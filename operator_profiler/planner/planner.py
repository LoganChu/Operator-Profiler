"""
ThetaPlanner — LLM-backed graph rewrite planner (θ_p).

Uses the Anthropic SDK with ``tool_choice`` to guarantee that the model
returns a valid JSON object matching the ``RewritePlan`` schema.  Falls
back to an empty ``RewritePlan`` on any parse or API error so the
optimization loop can continue.

Environment
-----------
Set ``ANTHROPIC_API_KEY`` before instantiating ``ThetaPlanner``, or pass
``api_key`` to ``PlannerConfig``.
"""
from __future__ import annotations

import json
import logging
from dataclasses import dataclass, field
from typing import TYPE_CHECKING, Literal

from operator_profiler.rewriter.dsl import RewritePlan
from operator_profiler.planner.schema import SearchCandidate
from operator_profiler.planner.system_prompt import build_system_prompt, build_gpu_context_section

if TYPE_CHECKING:
    import torch.fx
    from operator_profiler.schema.profile import OperatorAttributedProfile

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Tool schema for structured output
# ---------------------------------------------------------------------------

# We pass this as a "tool" to the Anthropic API with tool_choice="produce_rewrite_plan"
# so the model is forced to call it — guaranteeing JSON output.
# Tool schema for ranking memory candidates
_RANK_CANDIDATES_TOOL = {
    "name": "rank_memory_candidates",
    "description": (
        "Re-rank a list of retrieved memory candidates by semantic relevance to the "
        "current profile's bottleneck, graph structure, and metrics. Return the "
        "candidate IDs in descending order of relevance."
    ),
    "input_schema": {
        "type": "object",
        "properties": {
            "ranked_ids": {
                "type": "array",
                "items": {"type": "string"},
                "description": (
                    "Entry IDs from the candidates list, reordered from most to least "
                    "relevant for the current optimization context."
                ),
            },
            "reasoning": {
                "type": "string",
                "description": "One sentence explaining the top choice.",
            },
        },
        "required": ["ranked_ids"],
    },
}

_RANK_SYSTEM_PROMPT = (
    "You are a GPU optimization memory retrieval assistant. "
    "Given a current profile summary and a list of historical optimization entries, "
    "rank the entries by how useful they are for optimizing the current bottleneck. "
    "Prefer entries with: the same bottleneck class, similar op sequences, "
    "high speedup, and compatible graph structure. "
    "Respond only by calling the rank_memory_candidates tool."
)

_REWRITE_PLAN_TOOL = {
    "name": "produce_rewrite_plan",
    "description": (
        "Emit a RewritePlan JSON object that will be applied to the PyTorch "
        "FX graph to reduce GPU execution time."
    ),
    "input_schema": {
        "type": "object",
        "properties": {
            "plan_version": {
                "type": "string",
                "description": 'Must be "1.0".',
            },
            "source_profile_id": {
                "type": "string",
                "description": "schema_version/worst_operator_id from the profile.",
            },
            "description": {
                "type": "string",
                "description": "One sentence explaining the targeted bottleneck.",
            },
            "ops": {
                "type": "array",
                "description": "List of rewrite ops (fuse/reorder/change_layout/buffer_sharing).",
                "items": {"type": "object"},
            },
        },
        "required": ["plan_version", "ops"],
    },
}


# ---------------------------------------------------------------------------
# Config
# ---------------------------------------------------------------------------

@dataclass
class PlannerConfig:
    model: str = "claude-sonnet-4-6"
    max_tokens: int = 2048
    temperature: float = 0.3    # low temp → more deterministic structured output
    api_key: str | None = None  # falls back to ANTHROPIC_API_KEY env var


# ---------------------------------------------------------------------------
# ThetaPlanner
# ---------------------------------------------------------------------------

class ThetaPlanner:
    """
    Calls an LLM (Claude via Anthropic SDK) to generate a ``RewritePlan``
    given an ``OperatorAttributedProfile`` and a ``torch.fx.GraphModule``.

    Parameters
    ----------
    config:
        Model / API configuration.  Defaults to ``PlannerConfig()``.
    """

    def __init__(self, config: PlannerConfig | None = None) -> None:
        self._config = config or PlannerConfig()
        self._system_prompt = build_system_prompt()
        self._client = self._build_client()

    def _build_client(self):  # type: ignore[return]
        try:
            import anthropic
        except ImportError as exc:
            raise ImportError(
                "anthropic package is required for ThetaPlanner. "
                "Install it with: pip install anthropic"
            ) from exc
        kwargs = {}
        if self._config.api_key:
            kwargs["api_key"] = self._config.api_key
        return anthropic.Anthropic(**kwargs)

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def plan(
        self,
        gm: "torch.fx.GraphModule",
        profile: "OperatorAttributedProfile",
        candidates: list[SearchCandidate],
        strategy: Literal["explore", "refine"] = "explore",
        repair_context: str | None = None,
    ) -> RewritePlan:
        """
        Generate a ``RewritePlan`` for ``gm`` given the profiling data.

        Parameters
        ----------
        gm:
            The ``torch.fx.GraphModule`` to be rewritten.
        profile:
            ``OperatorAttributedProfile`` from the most recent profiling run.
        candidates:
            Similar successful rewrites retrieved from ``OptimizationMemory``.
            Pass an empty list for pure exploration.
        strategy:
            ``"explore"`` — generate a novel approach (no memory context).
            ``"refine"``  — build on retrieved patterns (candidates used).
        repair_context:
            Optional repair hint from ``VerifierAgent`` injected after a
            failed verification attempt.  When set, the planner is told
            specifically what went wrong and what to avoid on this retry.
        """
        user_message = self._build_user_message(gm, profile, candidates, strategy, repair_context)

        try:
            response = self._client.messages.create(
                model=self._config.model,
                max_tokens=self._config.max_tokens,
                temperature=self._config.temperature,
                system=self._system_prompt,
                tools=[_REWRITE_PLAN_TOOL],
                tool_choice={"type": "tool", "name": "produce_rewrite_plan"},
                messages=[{"role": "user", "content": user_message}],
            )
        except Exception as exc:
            logger.warning("Anthropic API call failed: %s — returning empty plan", exc)
            return RewritePlan()

        return self._parse_response(response)

    def rank_candidates(
        self,
        profile: "OperatorAttributedProfile",
        candidates: list[SearchCandidate],
        device_name: str | None = None,
    ) -> list[SearchCandidate]:
        """
        Re-rank ``candidates`` by semantic relevance to ``profile`` via an LLM call.

        Uses a separate, lightweight Anthropic call with ``tool_choice`` forcing
        ``rank_memory_candidates``.  Falls back to the original order on any
        API or parse error, or when there are fewer than 2 candidates.

        Parameters
        ----------
        profile:
            The current ``OperatorAttributedProfile`` being optimised.
        candidates:
            Broad-search candidates (from ``OptimizationMemory.broad_search``).
        device_name:
            Optional GPU device name for the context section.
        """
        if len(candidates) <= 1:
            return candidates

        gpu_ctx = build_gpu_context_section(profile, device_name)

        # Build a compact representation of each candidate for the LLM
        candidate_summaries = []
        for c in candidates:
            candidate_summaries.append({
                "entry_id": c.entry.entry_id,
                "bottleneck": c.entry.bottleneck,
                "op_sequence": c.entry.graph_pattern.op_sequence,
                "speedup": c.entry.speedup,
                "jaccard_similarity": round(c.similarity, 3),
                "rewrite_ops": [op.get("op", "unknown") for op in c.entry.rewrite_plan.model_dump().get("ops", [])],
            })

        user_message = (
            f"{gpu_ctx}\n\n"
            f"## Current Profile — Op Sequence\n"
            f"{[op.operator_name for op in profile.operators]}\n\n"
            f"## Memory Candidates (broad search, unfiltered)\n"
            f"{json.dumps(candidate_summaries, indent=2)}\n\n"
            "Rank these candidates from most to least relevant for the current optimization."
        )

        try:
            response = self._client.messages.create(
                model=self._config.model,
                max_tokens=512,
                temperature=0.0,
                system=_RANK_SYSTEM_PROMPT,
                tools=[_RANK_CANDIDATES_TOOL],
                tool_choice={"type": "tool", "name": "rank_memory_candidates"},
                messages=[{"role": "user", "content": user_message}],
            )
        except Exception as exc:
            logger.warning("rank_candidates API call failed: %s — using original order", exc)
            return candidates

        for block in response.content:
            if block.type == "tool_use" and block.name == "rank_memory_candidates":
                try:
                    ranked_ids: list[str] = block.input["ranked_ids"]
                    id_to_candidate = {c.entry.entry_id: c for c in candidates}
                    reordered = [
                        id_to_candidate[eid]
                        for eid in ranked_ids
                        if eid in id_to_candidate
                    ]
                    # Append any candidates not mentioned by the LLM at the end
                    mentioned = set(ranked_ids)
                    reordered += [c for c in candidates if c.entry.entry_id not in mentioned]
                    return reordered
                except Exception as exc:
                    logger.warning("rank_candidates parse error: %s — using original order", exc)
                    return candidates

        logger.warning("rank_candidates: no tool_use block — using original order")
        return candidates

    # ------------------------------------------------------------------
    # Private helpers
    # ------------------------------------------------------------------

    def _build_user_message(
        self,
        gm: "torch.fx.GraphModule",
        profile: "OperatorAttributedProfile",
        candidates: list[SearchCandidate],
        strategy: Literal["explore", "refine"],
        repair_context: str | None = None,
    ) -> str:
        memory_section: str
        if candidates and strategy == "refine":
            mem_data = [
                {
                    "speedup": c.entry.speedup,
                    "similarity": c.similarity,
                    "bottleneck": c.entry.bottleneck,
                    "rewrite_plan": c.entry.rewrite_plan.model_dump(),
                }
                for c in candidates
            ]
            memory_section = (
                "## Retrieved Memory (similar successful rewrites — use as inspiration)\n"
                + json.dumps(mem_data, indent=2)
            )
        else:
            memory_section = "## Retrieved Memory\nNone — generate a novel approach."

        strategy_instruction = (
            "**Strategy: REFINE** — build on the retrieved patterns above, "
            "adapting node names to the FX graph printed below."
            if strategy == "refine"
            else "**Strategy: EXPLORE** — generate a novel, creative rewrite approach."
        )

        repair_section = (
            f"\n\n{repair_context}"
            if repair_context is not None
            else ""
        )

        return (
            f"## Operator-Attributed Profile\n"
            f"{profile.model_dump_json(indent=2)}\n\n"
            f"## FX Graph\n"
            f"{gm.print_readable()}\n\n"
            f"{memory_section}\n\n"
            f"{strategy_instruction}"
            f"{repair_section}\n\n"
            "Produce a RewritePlan JSON by calling the `produce_rewrite_plan` tool."
        )

    def _parse_response(self, response) -> RewritePlan:  # type: ignore[return]
        for block in response.content:
            if block.type == "tool_use" and block.name == "produce_rewrite_plan":
                try:
                    return RewritePlan.model_validate(block.input)
                except Exception as exc:
                    logger.warning(
                        "Failed to parse RewritePlan from tool input: %s\nRaw: %s",
                        exc,
                        block.input,
                    )
                    return RewritePlan()
        logger.warning("No tool_use block in Anthropic response — returning empty plan")
        return RewritePlan()
