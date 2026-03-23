"""
ProfileDiff computation — matches before/after operators, handles fusion,
and computes speedups and bottleneck transitions.
"""
from __future__ import annotations

from typing import TYPE_CHECKING

from operator_profiler.summarizer.schema import OperatorDiff, ProfileDiff

if TYPE_CHECKING:
    from operator_profiler.schema.profile import OperatorAttributedProfile, OperatorRecord
    from operator_profiler.rewriter.dsl import RewritePlan, FuseOp


def compute_diff(
    before: "OperatorAttributedProfile",
    after: "OperatorAttributedProfile",
    plan: "RewritePlan | None",
    top_n: int = 5,
) -> ProfileDiff:
    """
    Compute a ProfileDiff from before/after profiles and an optional RewritePlan.

    Matching proceeds in four passes:
    1. Exact key match on (operator_name, call_index).
    2. Fusion resolution via FuseOp node lists and after.is_fused / fused_with.
    3. Remaining unmatched before ops → match_type="removed".
    4. Remaining unmatched after ops → match_type="new".
    """
    from operator_profiler.rewriter.dsl import FuseOp

    diffs: list[OperatorDiff] = []
    fusion_map = _build_fusion_map(plan)           # {node_id → [sibling_node_ids]}
    rewrite_ops_by_node = _index_rewrite_ops(plan) # {node_id → [op_ids]}

    # Build lookup dictionaries keyed by (operator_name, call_index)
    before_index: dict[tuple[str, int], OperatorRecord] = {
        (op.operator_name, op.call_index): op for op in before.operators
    }
    after_index: dict[tuple[str, int], OperatorRecord] = {
        (op.operator_name, op.call_index): op for op in after.operators
    }

    matched_before: set[str] = set()  # operator_id keys
    matched_after: set[str] = set()

    # ------------------------------------------------------------------
    # Pass 1 — exact key match
    # ------------------------------------------------------------------
    for key, before_op in before_index.items():
        if key in after_index:
            after_op = after_index[key]
            d = _make_operator_diff(
                before_op=before_op,
                after_op=after_op,
                rewrite_ops=rewrite_ops_by_node.get(before_op.operator_id, []),
                fusion_partners=[],
                match_type="exact",
            )
            diffs.append(d)
            matched_before.add(before_op.operator_id)
            matched_after.add(after_op.operator_id)

    # ------------------------------------------------------------------
    # Pass 2 — fusion resolution
    # ------------------------------------------------------------------
    # For each FuseOp, see if the constituent before-ops are unmatched
    # and whether a fused-together operator exists in after.
    if plan is not None:
        for fuse_op in (o for o in plan.ops if isinstance(o, FuseOp)):
            constituent_ids = fuse_op.nodes  # DSL node ids (operator_ids)

            # Filter to those that are actually in before and not yet matched
            unmatched_constituents = [
                cid for cid in constituent_ids
                if cid not in matched_before
            ]
            if not unmatched_constituents:
                continue

            # Find total before duration for all constituents
            before_total_ns = 0
            for cid in unmatched_constituents:
                bop = _find_op_by_id(cid, before.operators)
                if bop is not None and bop.aggregated is not None:
                    before_total_ns += bop.aggregated.total_duration_ns

            # Find the corresponding fused op in after (is_fused=True and fused_with overlap)
            fused_after_op = _find_fused_after_op(
                constituent_ids, after.operators, matched_after
            )

            after_ns: int | None = None
            after_op_id: str | None = None
            speedup: float | None = None
            if fused_after_op is not None and fused_after_op.aggregated is not None:
                after_ns = fused_after_op.aggregated.total_duration_ns
                after_op_id = fused_after_op.operator_id
                speedup = (before_total_ns / after_ns) if after_ns > 0 else None
                matched_after.add(fused_after_op.operator_id)

            # Emit one OperatorDiff per constituent
            for cid in unmatched_constituents:
                bop = _find_op_by_id(cid, before.operators)
                if bop is None:
                    continue
                bop_ns = bop.aggregated.total_duration_ns if bop.aggregated else 0
                partners = [x for x in constituent_ids if x != cid]
                d = OperatorDiff(
                    operator_id_before=bop.operator_id,
                    operator_id_after=after_op_id,
                    operator_name=bop.operator_name,
                    call_index=bop.call_index,
                    duration_before_ns=bop_ns,
                    duration_after_ns=after_ns,
                    delta_duration_ns=(bop_ns - after_ns) if after_ns is not None else None,
                    speedup=speedup,
                    bottleneck_before=(
                        bop.aggregated.bottleneck_classification if bop.aggregated else "unknown"
                    ),
                    bottleneck_after=(
                        fused_after_op.aggregated.bottleneck_classification
                        if fused_after_op and fused_after_op.aggregated
                        else None
                    ),
                    bottleneck_changed=False,
                    rewrite_ops_applied=rewrite_ops_by_node.get(bop.operator_id, []),
                    fusion_partners=partners,
                    match_type="fused_into",
                )
                diffs.append(d)
                matched_before.add(bop.operator_id)

    # ------------------------------------------------------------------
    # Pass 3 — removed operators
    # ------------------------------------------------------------------
    for op in before.operators:
        if op.operator_id not in matched_before:
            ns = op.aggregated.total_duration_ns if op.aggregated else 0
            diffs.append(
                OperatorDiff(
                    operator_id_before=op.operator_id,
                    operator_id_after=None,
                    operator_name=op.operator_name,
                    call_index=op.call_index,
                    duration_before_ns=ns,
                    duration_after_ns=None,
                    bottleneck_before=(
                        op.aggregated.bottleneck_classification if op.aggregated else "unknown"
                    ),
                    rewrite_ops_applied=rewrite_ops_by_node.get(op.operator_id, []),
                    match_type="removed",
                )
            )

    # ------------------------------------------------------------------
    # Pass 4 — new operators
    # ------------------------------------------------------------------
    for op in after.operators:
        if op.operator_id not in matched_after:
            ns = op.aggregated.total_duration_ns if op.aggregated else 0
            diffs.append(
                OperatorDiff(
                    operator_id_before=op.operator_id,  # use after id as placeholder
                    operator_id_after=op.operator_id,
                    operator_name=op.operator_name,
                    call_index=op.call_index,
                    duration_before_ns=0,
                    duration_after_ns=ns,
                    bottleneck_before="unknown",
                    bottleneck_after=(
                        op.aggregated.bottleneck_classification if op.aggregated else "unknown"
                    ),
                    match_type="new",
                )
            )

    # ------------------------------------------------------------------
    # Fix bottleneck_changed flag for exact matches
    # ------------------------------------------------------------------
    for d in diffs:
        if d.match_type == "exact" and d.bottleneck_after is not None:
            d.bottleneck_changed = d.bottleneck_before != d.bottleneck_after

    # ------------------------------------------------------------------
    # Totals
    # ------------------------------------------------------------------
    total_before = sum(
        op.aggregated.total_duration_ns
        for op in before.operators
        if op.aggregated is not None
    )
    total_after = sum(
        op.aggregated.total_duration_ns
        for op in after.operators
        if op.aggregated is not None
    )
    total_speedup = (total_before / total_after) if total_after > 0 else 1.0

    # Top bottlenecks: sort eligible diffs by duration_before_ns desc
    eligible = [
        d for d in diffs
        if d.match_type in ("exact", "fused_into") and d.duration_before_ns > 0
    ]
    top_bottlenecks = sorted(eligible, key=lambda d: d.duration_before_ns, reverse=True)[:top_n]

    return ProfileDiff(
        model_name=before.capture_metadata.model_name,
        device_name=before.capture_metadata.device_name,
        total_duration_before_ns=total_before,
        total_duration_after_ns=total_after,
        total_speedup=total_speedup,
        wall_time_saved_ns=total_before - total_after,
        operator_diffs=diffs,
        top_bottlenecks_before=top_bottlenecks,
        unmatched_before=[
            op.operator_id
            for op in before.operators
            if op.operator_id not in matched_before
        ],
        unmatched_after=[
            op.operator_id
            for op in after.operators
            if op.operator_id not in matched_after
        ],
    )


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _build_fusion_map(plan: "RewritePlan | None") -> dict[str, list[str]]:
    """Return {node_id: [sibling_ids]} for every FuseOp in plan."""
    from operator_profiler.rewriter.dsl import FuseOp
    result: dict[str, list[str]] = {}
    if plan is None:
        return result
    for op in plan.ops:
        if isinstance(op, FuseOp):
            for node_id in op.nodes:
                result[node_id] = [n for n in op.nodes if n != node_id]
    return result


def _index_rewrite_ops(plan: "RewritePlan | None") -> dict[str, list[str]]:
    """Return {operator_id: [dsl_op_ids]} mapping each operator to relevant DSL ops."""
    from operator_profiler.rewriter.dsl import (
        FuseOp, ReorderOp, ChangeLayoutOp, BufferSharingOp
    )
    result: dict[str, list[str]] = {}
    if plan is None:
        return result

    def _add(node_id: str, op_id: str) -> None:
        result.setdefault(node_id, []).append(op_id)

    for op in plan.ops:
        if isinstance(op, FuseOp):
            for nid in op.nodes:
                _add(nid, op.id)
        elif isinstance(op, ReorderOp):
            _add(op.node, op.id)
        elif isinstance(op, ChangeLayoutOp):
            _add(op.target_node, op.id)
        elif isinstance(op, BufferSharingOp):
            _add(op.source_node, op.id)
            _add(op.target_node, op.id)
    return result


def _find_op_by_id(
    operator_id: str,
    operators: "list[OperatorRecord]",
) -> "OperatorRecord | None":
    for op in operators:
        if op.operator_id == operator_id:
            return op
    return None


def _find_fused_after_op(
    constituent_ids: list[str],
    after_operators: "list[OperatorRecord]",
    already_matched: set[str],
) -> "OperatorRecord | None":
    """
    Find the fused operator in `after` whose `fused_with` contains at least one
    constituent operator name, and which has not already been matched.
    """
    constituent_names = set(constituent_ids)
    for op in after_operators:
        if op.operator_id in already_matched:
            continue
        if not op.is_fused:
            continue
        # fused_with stores operator names or operator_ids
        if constituent_names.intersection(set(op.fused_with)):
            return op
    return None


def _make_operator_diff(
    before_op: "OperatorRecord",
    after_op: "OperatorRecord | None",
    rewrite_ops: list[str],
    fusion_partners: list[str],
    match_type: str,
) -> OperatorDiff:
    before_ns = before_op.aggregated.total_duration_ns if before_op.aggregated else 0
    after_ns: int | None = None
    speedup: float | None = None
    delta: int | None = None
    bt_after: str | None = None

    if after_op is not None and after_op.aggregated is not None:
        after_ns = after_op.aggregated.total_duration_ns
        speedup = (before_ns / after_ns) if after_ns > 0 else None
        delta = before_ns - after_ns
        bt_after = after_op.aggregated.bottleneck_classification

    return OperatorDiff(
        operator_id_before=before_op.operator_id,
        operator_id_after=after_op.operator_id if after_op else None,
        operator_name=before_op.operator_name,
        call_index=before_op.call_index,
        duration_before_ns=before_ns,
        duration_after_ns=after_ns,
        delta_duration_ns=delta,
        speedup=speedup,
        bottleneck_before=(
            before_op.aggregated.bottleneck_classification
            if before_op.aggregated
            else "unknown"
        ),
        bottleneck_after=bt_after,
        bottleneck_changed=False,  # fixed after all diffs are computed
        rewrite_ops_applied=rewrite_ops,
        fusion_partners=fusion_partners,
        match_type=match_type,  # type: ignore[arg-type]
    )
