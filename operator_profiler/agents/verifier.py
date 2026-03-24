"""
VerifierAgent — interprets VerificationResult.node_diffs and generates a
repair hint for ThetaPlanner.plan() to use on a single retry.

Failure categories
------------------
layout_error          Shape or stride mismatch introduced by a change_layout op.
numerical_instability Large absolute error without shape change; op semantics altered.
shape_mismatch        Output shape differs between original and rewritten graph.
op_semantics_error    Reorder changed execution order of dependent ops, or a fuse
                      merged ops with incompatible broadcasting.
unknown               Unclassifiable from available signals.

Integration
-----------
OptimizationLoop calls verifier_agent.diagnose(plan, ver_results) on the first
verification failure.  The resulting RepairContext.to_prompt_section() string is
passed to ThetaPlanner.plan(repair_context=...) for one retry.  If the retry
also fails, the plan is skipped as normal.
"""
from __future__ import annotations

import json
import logging
from dataclasses import dataclass, field
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from operator_profiler.rewriter.dsl import RewritePlan
    from operator_profiler.rewriter.verification import VerificationResult

log = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Tool schema
# ---------------------------------------------------------------------------

_SYSTEM_PROMPT = """\
You are a PyTorch FX graph rewrite debugger. A RewritePlan was applied to a
GraphModule but numerical verification failed — the rewritten graph produces
different outputs than the original.

Given the failed plan and the verification result details (error message, diverging
node names, shapes, max absolute errors), diagnose the root cause and write a
specific, actionable repair hint for the planner.

The repair hint should:
1. Name the specific node(s) or op(s) that caused the failure.
2. Explain the mechanical reason the applied op broke correctness.
3. Suggest a concrete alternative (different strategy, different target node,
   additional constraint to respect).

Respond only by calling the diagnose_verification_failure tool."""

_DIAGNOSE_TOOL = {
    "name": "diagnose_verification_failure",
    "description": (
        "Diagnose a rewrite verification failure and generate a repair hint for retry."
    ),
    "input_schema": {
        "type": "object",
        "properties": {
            "failure_category": {
                "type": "string",
                "enum": [
                    "layout_error",
                    "numerical_instability",
                    "shape_mismatch",
                    "op_semantics_error",
                    "unknown",
                ],
                "description": "Root cause category.",
            },
            "repair_hint": {
                "type": "string",
                "description": (
                    "One to two paragraphs of specific, actionable guidance. "
                    "Name the diverging node(s), explain why the applied op caused "
                    "the failure, and state what to do instead."
                ),
            },
            "avoid_ops": {
                "type": "array",
                "items": {"type": "string"},
                "description": (
                    "DSL op IDs from the failed plan that must NOT be retried verbatim."
                ),
            },
        },
        "required": ["failure_category", "repair_hint"],
    },
}


# ---------------------------------------------------------------------------
# Data class
# ---------------------------------------------------------------------------

@dataclass
class RepairContext:
    failure_category: str
    repair_hint: str
    avoid_ops: list[str] = field(default_factory=list)

    def to_prompt_section(self) -> str:
        """Format as a section injected into the ThetaPlanner user message."""
        lines = [
            "## Verification Failure — Repair Context",
            f"Failure category: {self.failure_category}",
            "",
            "The previous plan failed numerical verification. Guidance for this retry:",
            self.repair_hint,
        ]
        if self.avoid_ops:
            lines += [
                "",
                f"Do NOT reuse these exact op IDs verbatim: {', '.join(self.avoid_ops)}",
            ]
        lines += [
            "",
            "Generate a corrected plan that avoids the issues described above.",
        ]
        return "\n".join(lines)


# ---------------------------------------------------------------------------
# Agent
# ---------------------------------------------------------------------------

class VerifierAgent:
    """
    Interprets VerificationResult failures and produces RepairContext for
    a ThetaPlanner retry.

    Parameters
    ----------
    model:
        Anthropic model string.
    api_key:
        Optional API key; falls back to ANTHROPIC_API_KEY env var.
    """

    def __init__(
        self,
        model: str = "claude-sonnet-4-6",
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
                "anthropic package is required for VerifierAgent. "
                "Install with: pip install anthropic"
            ) from exc
        kwargs = {}
        if self._api_key:
            kwargs["api_key"] = self._api_key
        return anthropic.Anthropic(**kwargs)

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def diagnose(
        self,
        plan: "RewritePlan",
        ver_results: list["VerificationResult"],
    ) -> RepairContext:
        """
        Diagnose verification failure(s) and return a RepairContext.

        Parameters
        ----------
        plan:
            The RewritePlan that was applied and failed verification.
        ver_results:
            Verification results — only failed ones are used.

        Returns
        -------
        RepairContext
            Falls back to a generic hint derived from node_diffs on API error.
        """
        failed = [r for r in ver_results if not r.passed]
        if not failed:
            return RepairContext(
                failure_category="unknown",
                repair_hint="No failed verification results provided.",
            )

        user_message = self._build_message(plan, failed)
        try:
            response = self._client.messages.create(
                model=self._model,
                max_tokens=768,
                temperature=0.0,
                system=_SYSTEM_PROMPT,
                tools=[_DIAGNOSE_TOOL],
                tool_choice={"type": "tool", "name": "diagnose_verification_failure"},
                messages=[{"role": "user", "content": user_message}],
            )
        except Exception as exc:
            log.warning("VerifierAgent API call failed: %s — using generic hint", exc)
            return self._generic_fallback(failed)

        return self._parse_response(response, failed)

    # ------------------------------------------------------------------
    # Private helpers
    # ------------------------------------------------------------------

    def _build_message(
        self,
        plan: "RewritePlan",
        failed: list["VerificationResult"],
    ) -> str:
        plan_dump = plan.model_dump()
        lines = [
            "## Failed RewritePlan",
            f"Description: {plan.description or '(none)'}",
            f"Ops:\n{json.dumps(plan_dump.get('ops', []), indent=2)}",
            "",
            "## Verification Failures",
        ]
        for r in failed:
            lines.append(f"\n### Op ID: {r.op_id}")
            if r.error_message:
                lines.append(f"Error message: {r.error_message}")
            if r.max_abs_error is not None:
                lines.append(f"Max absolute error: {r.max_abs_error:.3e}")
            if r.node_diffs:
                lines.append(f"Diverging nodes ({len(r.node_diffs)}):")
                for nd in r.node_diffs[:6]:
                    lines.append(
                        f"  node={nd.node_name}  "
                        f"max_err={nd.max_abs_error:.3e}  "
                        f"orig_shape={nd.original_shape}  "
                        f"rw_shape={nd.rewritten_shape}"
                    )
        lines.append("\nDiagnose and provide a repair hint for a corrected retry.")
        return "\n".join(lines)

    def _parse_response(
        self,
        response,
        failed: list["VerificationResult"],
    ) -> RepairContext:
        for block in response.content:
            if block.type == "tool_use" and block.name == "diagnose_verification_failure":
                try:
                    inp = block.input
                    return RepairContext(
                        failure_category=inp.get("failure_category", "unknown"),
                        repair_hint=inp.get("repair_hint", ""),
                        avoid_ops=inp.get("avoid_ops", []),
                    )
                except Exception as exc:
                    log.warning("VerifierAgent parse error: %s", exc)
                    return self._generic_fallback(failed)
        log.warning("VerifierAgent: no tool_use block in response")
        return self._generic_fallback(failed)

    def _generic_fallback(self, failed: list["VerificationResult"]) -> RepairContext:
        """Construct a basic repair hint from raw node_diffs without LLM."""
        msgs: list[str] = []
        for r in failed:
            if r.error_message:
                msgs.append(r.error_message)
            elif r.node_diffs:
                nd = r.node_diffs[0]
                msgs.append(
                    f"Node '{nd.node_name}' diverged with "
                    f"max_abs_error={nd.max_abs_error:.3e} "
                    f"(orig_shape={nd.original_shape}, rw_shape={nd.rewritten_shape})"
                )
        hint = (
            "The previous plan failed numerical verification. "
            + " ".join(msgs)
            + " Try a different rewrite strategy or target a different operator."
        )
        return RepairContext(
            failure_category="unknown",
            repair_hint=hint,
            avoid_ops=[r.op_id for r in failed],
        )
