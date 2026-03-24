"""
RuleAgent — generates rich, causal OptimizationRule descriptions from
MemoryEntry records via an LLM.

Replaces the string-template rule_text in summarizer/rules.py with a
paragraph that explains WHY the rewrite worked — the mechanistic causal chain
— rather than just restating what the rewrite did.

Integration
-----------
entry_to_rule() and entries_to_rules() in summarizer/rules.py accept an
optional rule_agent keyword argument.  When provided, the agent enriches the
template-generated OptimizationRule with LLM-generated rule_text, conditions,
and recommended_action.  The original rule is returned unchanged on any API
or parse error.

Model choice: claude-haiku-4-5-20251001 by default — this task requires
narrative writing, not GPU metric reasoning, and Haiku is faster and cheaper
for generation tasks that don't require deep domain analysis.
"""
from __future__ import annotations

import logging
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from operator_profiler.planner.schema import MemoryEntry
    from operator_profiler.summarizer.schema import OptimizationRule

log = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Tool schema
# ---------------------------------------------------------------------------

_SYSTEM_PROMPT = """\
You are a GPU optimization educator writing documentation for a rewrite rule
library. Given a successful graph rewrite entry (operator pattern, bottleneck
class, rewrite applied, measured speedup), write a rich explanation that teaches
engineers WHY the rewrite improved performance — not just what it did.

Focus on the causal mechanism:
- For memory_bound: which DRAM round-trips were eliminated, which cache locality
  was improved, how DRAM bytes transferred changed.
- For compute_bound: how tensor core utilization improved, how pipeline parallelism
  increased, what scheduling conflict was resolved.
- For latency_bound: how many kernel dispatch round-trips were removed, why small
  kernels are expensive (occupancy, warp scheduling).

Write as if explaining to a senior ML engineer who knows PyTorch and GPU architecture
but hasn't seen this specific model. Be concrete and causal, not vague.

Respond only by calling the generate_optimization_rule tool."""

_GENERATE_TOOL = {
    "name": "generate_optimization_rule",
    "description": "Generate a rich, causal OptimizationRule explanation.",
    "input_schema": {
        "type": "object",
        "properties": {
            "rule_text": {
                "type": "string",
                "description": (
                    "Two to four sentences: (1) the bottleneck pattern and its cause, "
                    "(2) the mechanistic reason the rewrite helped (causal chain), "
                    "(3) the measured result. "
                    "Do NOT restate the inputs — explain the causal mechanism."
                ),
            },
            "conditions": {
                "type": "array",
                "items": {"type": "string"},
                "description": (
                    "2–4 specific metric conditions under which this rule applies. "
                    "Write as relative statements, e.g. "
                    "'arithmetic intensity is far below the GPU ridge point' "
                    "not 'AI < 5.0'. The rule should generalize across GPU generations."
                ),
            },
            "recommended_action": {
                "type": "string",
                "description": (
                    "One sentence: what to do and to which operator types. "
                    "Example: 'Fuse adjacent conv+bn+relu with inductor_fuse when all "
                    "three are memory-bound and share the same NVTX range.'"
                ),
            },
            "lessons_learned": {
                "type": "string",
                "description": (
                    "One sentence capturing the generalizable insight from this entry "
                    "that applies beyond the specific model. "
                    "Example: 'Elementwise ops following a conv are almost always "
                    "memory-bound and benefit from fusion regardless of batch size.'"
                ),
            },
        },
        "required": ["rule_text", "conditions", "recommended_action", "lessons_learned"],
    },
}


# ---------------------------------------------------------------------------
# Agent
# ---------------------------------------------------------------------------

class RuleAgent:
    """
    Generates rich OptimizationRule descriptions from MemoryEntry records.

    Parameters
    ----------
    model:
        Anthropic model string.  Defaults to Haiku — narrative generation
        does not require the heavier reasoning of Sonnet.
    api_key:
        Optional API key; falls back to ANTHROPIC_API_KEY env var.
    """

    def __init__(
        self,
        model: str = "claude-haiku-4-5-20251001",
        api_key: str | None = None,
    ) -> None:
        self._model = model
        self._api_key = api_key
        self._client = self._build_client()

    def _build_client(self):
        try:
            import anthropic
        except ImportError as exc:
            raise ImportError(
                "anthropic package is required for RuleAgent. "
                "Install with: pip install anthropic"
            ) from exc
        kwargs = {}
        if self._api_key:
            kwargs["api_key"] = self._api_key
        return anthropic.Anthropic(**kwargs)

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def enrich_rule(
        self,
        entry: "MemoryEntry",
        rule: "OptimizationRule",
    ) -> "OptimizationRule":
        """
        Return a copy of ``rule`` with LLM-generated rule_text, conditions,
        recommended_action, and lessons_learned.

        Falls back to the original ``rule`` unchanged on any API or parse error.
        """
        user_message = self._build_message(entry, rule)
        try:
            response = self._client.messages.create(
                model=self._model,
                max_tokens=600,
                temperature=0.3,
                system=_SYSTEM_PROMPT,
                tools=[_GENERATE_TOOL],
                tool_choice={"type": "tool", "name": "generate_optimization_rule"},
                messages=[{"role": "user", "content": user_message}],
            )
        except Exception as exc:
            log.warning("RuleAgent API call failed: %s — using template rule", exc)
            return rule

        return self._parse_response(response, rule)

    # ------------------------------------------------------------------
    # Private helpers
    # ------------------------------------------------------------------

    def _build_message(self, entry: "MemoryEntry", rule: "OptimizationRule") -> str:
        # Import locally to avoid a hard dependency at module level
        from operator_profiler.summarizer.rules import _summarise_rewrite_op

        ops_summary = "; ".join(
            _summarise_rewrite_op(op) for op in entry.rewrite_plan.ops
        ) or "(no ops)"

        return (
            f"## Memory Entry\n"
            f"Operator pattern: {entry.graph_pattern.op_sequence}\n"
            f"Bottleneck: {entry.bottleneck}\n"
            f"Rewrite applied: {ops_summary}\n"
            f"Measured speedup: {entry.speedup:.3f}×  ({(entry.speedup - 1) * 100:.1f}% faster)\n"
            f"Model: {entry.model_name or 'unknown'}\n"
            f"\n## Current Template Rule\n"
            f"{rule.rule_text}\n"
            f"\nGenerate a richer, causal explanation."
        )

    def _parse_response(
        self,
        response,
        rule: "OptimizationRule",
    ) -> "OptimizationRule":
        from operator_profiler.summarizer.schema import OptimizationRule as OR

        for block in response.content:
            if block.type == "tool_use" and block.name == "generate_optimization_rule":
                try:
                    inp = block.input
                    # Preserve all identity fields; replace text fields
                    return OR(
                        entry_id=rule.entry_id,
                        op_pattern=rule.op_pattern,
                        bottleneck=rule.bottleneck,
                        rewrite_op_summary=rule.rewrite_op_summary,
                        speedup=rule.speedup,
                        speedup_pct=rule.speedup_pct,
                        conditions=inp.get("conditions", rule.conditions),
                        recommended_action=inp.get("recommended_action", rule.recommended_action),
                        example_model=rule.example_model,
                        created_at=rule.created_at,
                        rule_text=inp.get("rule_text", rule.rule_text),
                    )
                except Exception as exc:
                    log.warning("RuleAgent parse error: %s — using template rule", exc)
                    return rule

        log.warning("RuleAgent: no tool_use block — using template rule")
        return rule
