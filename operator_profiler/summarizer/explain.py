"""
Explain command logic — pure data-driven, no LLM API calls.

Produces a natural-language paragraph explaining what happened to a specific
node across the optimization run.
"""
from __future__ import annotations

from typing import TYPE_CHECKING

from operator_profiler.summarizer.schema import OperatorDiff, ProfileDiff

if TYPE_CHECKING:
    from operator_profiler.schema.profile import OperatorAttributedProfile
    from operator_profiler.planner.loop import LoopResult


def explain_node(
    node_id: str,
    diff: ProfileDiff,
    before: "OperatorAttributedProfile",
    loop_result: "LoopResult",
) -> str:
    """
    Return a natural-language explanation for a single node.

    Parameters
    ----------
    node_id:
        The operator_id as it appears in the before profile.
        Double-underscore form is normalised: ``aten__linear_0`` → ``aten::linear_0``.
    diff:
        Pre-computed ProfileDiff.
    before:
        Before profile for raw kernel-level detail.
    loop_result:
        LoopResult from the OptimizationLoop.
    """
    normalised_id = _normalise_node_id(node_id)

    op_diff = _find_op_diff(normalised_id, diff)
    if op_diff is None:
        available = ", ".join(
            d.operator_id_before for d in diff.operator_diffs
            if d.match_type not in ("new",)
        )
        return (
            f"Node '{node_id}' not found in the before profile.\n"
            f"Available nodes: {available or '(none)'}"
        )

    # Find the OperatorRecord from before for kernel details
    before_op = _find_before_op(normalised_id, before)
    history_context = _build_history_context(normalised_id, loop_result)

    return _format_explanation(op_diff, before_op, history_context)


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _normalise_node_id(node_id: str) -> str:
    """Converts ``aten__linear_0`` → ``aten::linear_0`` (double-underscore → ::)."""
    return node_id.replace("__", "::")


def _find_op_diff(node_id: str, diff: ProfileDiff) -> OperatorDiff | None:
    for d in diff.operator_diffs:
        if d.operator_id_before == node_id:
            return d
    return None


def _find_before_op(node_id: str, before: "OperatorAttributedProfile"):
    for op in before.operators:
        if op.operator_id == node_id:
            return op
    return None


def _build_history_context(node_id: str, loop_result: "LoopResult") -> str:
    """Find iterations where this operator was the worst bottleneck."""
    worst_iters = [
        h["iteration"]
        for h in loop_result.history
        if h.get("worst_op_id") == node_id
    ]
    if not worst_iters:
        return "This operator was not the primary bottleneck in any iteration."
    iters_str = ", ".join(str(i) for i in worst_iters)
    best_speedup_then = max(
        (h.get("best_speedup_so_far", 1.0) for h in loop_result.history if h["iteration"] in worst_iters),
        default=1.0,
    )
    plans_tried = sum(
        h.get("plans_tried", 0) for h in loop_result.history if h["iteration"] in worst_iters
    )
    return (
        f"This operator was the worst bottleneck in iteration(s) {iters_str}. "
        f"The optimizer tried {plans_tried} plan(s) in those iteration(s); "
        f"best speedup achieved so far at that point was {best_speedup_then:.3f}x."
    )


def _format_explanation(
    op_diff: OperatorDiff,
    before_op,
    history_context: str,
) -> str:
    lines: list[str] = []

    lines.append(
        f"Operator: {op_diff.operator_name} (call index {op_diff.call_index})"
    )
    lines.append("")

    # --- BEFORE ---
    lines.append("BEFORE optimization:")
    before_ms = op_diff.duration_before_ns / 1e6
    lines.append(f"  - Duration: {before_ms:.3f} ms")
    lines.append(f"  - Bottleneck: {op_diff.bottleneck_before}")

    if before_op is not None:
        kernel_count = len(before_op.kernels)
        lines.append(f"  - Kernels: {kernel_count} kernel(s) attributed")
        if before_op.kernels:
            top_k = max(before_op.kernels, key=lambda k: k.duration_ns)
            top_ms = top_k.duration_ns / 1e6
            kname = top_k.demangled_name or top_k.kernel_name
            lines.append(f"  - Top kernel: {kname} ({top_ms:.3f} ms)")
    lines.append("")

    # --- AFTER ---
    if op_diff.match_type == "fused_into":
        lines.append("AFTER optimization:")
        if op_diff.duration_after_ns is not None:
            after_ms = op_diff.duration_after_ns / 1e6
            speedup_str = f"{op_diff.speedup:.2f}x" if op_diff.speedup else "—"
            if op_diff.delta_duration_ns is not None and op_diff.duration_before_ns > 0:
                pct = op_diff.delta_duration_ns / op_diff.duration_before_ns * 100
                pct_str = f" ({pct:.1f}% faster)"
            else:
                pct_str = ""
            lines.append(
                f"  - Duration: {after_ms:.3f} ms  ({speedup_str} speedup{pct_str})"
            )
        bt_after = op_diff.bottleneck_after or "—"
        lines.append(f"  - Bottleneck: {bt_after}")
        if op_diff.rewrite_ops_applied:
            lines.append(
                f"  - Rewrite ops applied: {', '.join(op_diff.rewrite_ops_applied)}"
            )
        if op_diff.fusion_partners:
            lines.append(
                f"  - Fused with: {', '.join(op_diff.fusion_partners)}"
            )
    elif op_diff.match_type == "exact":
        lines.append("AFTER optimization:")
        if op_diff.duration_after_ns is not None:
            after_ms = op_diff.duration_after_ns / 1e6
            speedup_str = f"{op_diff.speedup:.2f}x" if op_diff.speedup else "—"
            if op_diff.delta_duration_ns is not None and op_diff.duration_before_ns > 0:
                pct = op_diff.delta_duration_ns / op_diff.duration_before_ns * 100
                pct_str = f", -{pct:.1f}%"
            else:
                pct_str = ""
            lines.append(
                f"  - Duration: {after_ms:.3f} ms  ({speedup_str} speedup{pct_str})"
            )
        bt_after = op_diff.bottleneck_after or "—"
        lines.append(f"  - Bottleneck: {bt_after}")
        if op_diff.rewrite_ops_applied:
            lines.append(
                f"  - Rewrite ops applied: {', '.join(op_diff.rewrite_ops_applied)}"
            )
    elif op_diff.match_type == "removed":
        lines.append("AFTER optimization: This operator was removed from the graph.")
    lines.append("")

    # --- CONTEXT ---
    lines.append("Optimization context:")
    lines.append(f"  - {history_context}")
    lines.append("")

    return "\n".join(lines)
