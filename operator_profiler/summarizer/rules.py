"""
OptimizationRule generation — distills MemoryEntry records into
human-readable rules for the Lessons Learned section.
"""
from __future__ import annotations

from typing import TYPE_CHECKING

from operator_profiler.summarizer.schema import OptimizationRule

if TYPE_CHECKING:
    from operator_profiler.planner.schema import MemoryEntry

# Bottleneck → list of human-readable threshold conditions
_CONDITIONS: dict[str, list[str]] = {
    "memory_bound": [
        "arithmetic_intensity < 5.0 FLOP/byte",
        "achieved_occupancy < 0.6",
    ],
    "compute_bound": [
        "tensor_core_active_pct < 50%",
        "arithmetic_intensity >= 5.0 FLOP/byte",
    ],
    "latency_bound": [
        "kernel_count > 3 per operator",
        "mean_achieved_occupancy < 0.4",
    ],
    "unknown": [],
}


def entry_to_rule(entry: "MemoryEntry") -> OptimizationRule:
    """Convert a single MemoryEntry into a human-readable OptimizationRule."""
    op_pattern = entry.graph_pattern.op_sequence
    rewrite_op_summary = "; ".join(
        _summarise_rewrite_op(op) for op in entry.rewrite_plan.ops
    )
    conditions = _CONDITIONS.get(entry.bottleneck, [])
    speedup_pct = round((entry.speedup - 1.0) * 100, 1)
    recommended_action = (
        f"Apply {rewrite_op_summary} to "
        f"{', '.join(op_pattern[:3])} operators"
    )
    rule_text = _build_rule_text(
        op_pattern=op_pattern,
        bottleneck=entry.bottleneck,
        rewrite_op_summary=rewrite_op_summary,
        speedup_pct=speedup_pct,
    )
    return OptimizationRule(
        entry_id=entry.entry_id,
        op_pattern=op_pattern,
        bottleneck=entry.bottleneck,
        rewrite_op_summary=rewrite_op_summary,
        speedup=entry.speedup,
        speedup_pct=speedup_pct,
        conditions=conditions,
        recommended_action=recommended_action,
        example_model=entry.model_name,
        created_at=entry.created_at,
        rule_text=rule_text,
    )


def entries_to_rules(
    entries: "list[MemoryEntry]",
    sort_by: str = "speedup",
    top_n: int | None = None,
) -> list[OptimizationRule]:
    """Convert a list of MemoryEntry records to OptimizationRules.

    Parameters
    ----------
    entries:
        Source memory entries.
    sort_by:
        ``"speedup"`` (default) or ``"created_at"`` (ISO 8601 lexicographic).
    top_n:
        If provided, return only the top N rules after sorting.
    """
    rules = [entry_to_rule(e) for e in entries]
    if sort_by == "speedup":
        rules.sort(key=lambda r: r.speedup, reverse=True)
    elif sort_by == "created_at":
        rules.sort(key=lambda r: r.created_at, reverse=True)
    if top_n is not None:
        rules = rules[:top_n]
    return rules


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _summarise_rewrite_op(op) -> str:  # op: AnyRewriteOp
    """One-line human-readable summary for any rewrite op."""
    from operator_profiler.rewriter.dsl import (
        FuseOp, ReorderOp, ChangeLayoutOp, BufferSharingOp,
    )
    if isinstance(op, FuseOp):
        return f"fuse({', '.join(op.nodes)}, strategy={op.strategy})"
    if isinstance(op, ReorderOp):
        anchor = f"before={op.before}" if op.before else f"after={op.after}"
        return f"reorder({op.node}, {anchor})"
    if isinstance(op, ChangeLayoutOp):
        return f"change_layout({op.target_node}, {op.current_format}→{op.target_format})"
    if isinstance(op, BufferSharingOp):
        return f"buffer_sharing({op.source_node}→{op.target_node})"
    return repr(op)


def _build_rule_text(
    op_pattern: list[str],
    bottleneck: str,
    rewrite_op_summary: str,
    speedup_pct: float,
) -> str:
    ops_str = ", ".join(op_pattern) if op_pattern else "(unknown operators)"
    return (
        f"When [{ops_str}] is {bottleneck}, "
        f"apply {rewrite_op_summary} "
        f"to achieve ~{speedup_pct:.1f}% speedup"
    )
