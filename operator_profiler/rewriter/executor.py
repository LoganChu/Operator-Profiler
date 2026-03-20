"""
HybridExecutor θ_e — applies a ``RewritePlan`` to a ``torch.fx.GraphModule``
deterministically and verifies each step against the original graph.

Algorithm (``run()``)
---------------------
1. Validate ``plan_version``.
2. Pre-flight **all** ops against the current node set before any mutation.
3. Deep-copy the original graph as the working graph.
4. For each op:
   - Checkpoint the working graph (``copy.deepcopy``).
   - Apply the op.
   - Verify the mutated graph against the **original** (not the prior step).
   - If passed: advance ``working_gm``.
   - If failed: atomic rollback to checkpoint; break unless
     ``continue_on_verification_failure=True``.
5. Return ``(working_gm, results)``.

The original ``GraphModule`` is **never mutated**.
"""
from __future__ import annotations

import copy
from dataclasses import dataclass, field
from typing import TYPE_CHECKING

from operator_profiler.rewriter.dsl import (
    AnyRewriteOp,
    BufferSharingOp,
    ChangeLayoutOp,
    DSL_VERSION,
    FuseOp,
    ReorderOp,
    RewritePlan,
    RewriteValidationError,
)
from operator_profiler.rewriter.verification import VerificationGate, VerificationResult

if TYPE_CHECKING:
    import torch.fx

# Re-export so callers can do ``from operator_profiler.rewriter.executor import ...``
__all__ = [
    "ExecutorConfig",
    "HybridExecutor",
    "RewriteValidationError",
    "PreFlightError",
]


class PreFlightError(RewriteValidationError):
    """Raised when pre-flight validation fails before any graph mutation."""


# ---------------------------------------------------------------------------
# Config
# ---------------------------------------------------------------------------

@dataclass
class ExecutorConfig:
    verification_atol: float = 1e-5
    verification_rtol: float = 1e-5
    device: str = "cpu"
    continue_on_verification_failure: bool = False
    skip_verification: bool = False


# ---------------------------------------------------------------------------
# Pre-flight helpers
# ---------------------------------------------------------------------------

def _node_names(gm: "torch.fx.GraphModule") -> set[str]:
    return {n.name for n in gm.graph.nodes}


def _preflight_op(op: AnyRewriteOp, gm: "torch.fx.GraphModule") -> None:
    """
    Validate ``op`` against the current graph state.  Raises
    ``PreFlightError`` if any node reference is missing or a constraint is
    violated.  No graph mutation occurs here.
    """
    from operator_profiler.rewriter.ops.buffer_sharing import _compute_liveness, _live_ranges_overlap
    from operator_profiler.rewriter.ops.change_layout import _is_layout_sensitive, _find_node as _cl_find
    from operator_profiler.rewriter.ops.reorder import _topo_reachable

    names = _node_names(gm)

    if isinstance(op, FuseOp):
        for name in op.nodes:
            if name not in names:
                raise PreFlightError(
                    f"Fuse op '{op.id}': node '{name}' not found in graph"
                )
        if op.strategy == "custom_op" and not op.custom_op_name:
            raise PreFlightError(
                f"Fuse op '{op.id}': custom_op_name required for strategy='custom_op'"
            )

    elif isinstance(op, ReorderOp):
        if op.node not in names:
            raise PreFlightError(
                f"Reorder op '{op.id}': node '{op.node}' not found in graph"
            )
        anchor_name = op.before if op.before is not None else op.after
        if anchor_name not in names:
            raise PreFlightError(
                f"Reorder op '{op.id}': anchor node '{anchor_name}' not found in graph"
            )
        # Topological dependency check
        node = next(n for n in gm.graph.nodes if n.name == op.node)
        anchor = next(n for n in gm.graph.nodes if n.name == anchor_name)
        if op.before is not None and _topo_reachable(anchor, node):
            raise PreFlightError(
                f"Reorder op '{op.id}': '{op.node}' depends on '{op.before}'; "
                f"cannot place before it"
            )
        if op.after is not None and _topo_reachable(node, anchor):
            raise PreFlightError(
                f"Reorder op '{op.id}': '{op.after}' depends on '{op.node}'; "
                f"cannot place after it"
            )

    elif isinstance(op, ChangeLayoutOp):
        if op.target_node not in names:
            raise PreFlightError(
                f"ChangeLayout op '{op.id}': node '{op.target_node}' not found in graph"
            )
        target = next(n for n in gm.graph.nodes if n.name == op.target_node)
        if not _is_layout_sensitive(target):
            raise PreFlightError(
                f"ChangeLayout op '{op.id}': node '{op.target_node}' "
                f"(target={target.target!r}) is not a layout-sensitive op"
            )

    elif isinstance(op, BufferSharingOp):
        for attr in ("source_node", "target_node"):
            name = getattr(op, attr)
            if name not in names:
                raise PreFlightError(
                    f"BufferSharing op '{op.id}': node '{name}' not found in graph"
                )
        if op.validate_liveness:
            liveness = _compute_liveness(gm)
            src_r = liveness.get(op.source_node)
            tgt_r = liveness.get(op.target_node)
            if src_r is None or tgt_r is None:
                raise PreFlightError(
                    f"BufferSharing op '{op.id}': cannot compute liveness"
                )
            if _live_ranges_overlap(src_r, tgt_r):
                raise PreFlightError(
                    f"BufferSharing op '{op.id}': live ranges of "
                    f"'{op.source_node}' {src_r} and '{op.target_node}' {tgt_r} overlap"
                )


# ---------------------------------------------------------------------------
# Dispatcher
# ---------------------------------------------------------------------------

def _apply_op(
    gm: "torch.fx.GraphModule", op: AnyRewriteOp
) -> "torch.fx.GraphModule":
    from operator_profiler.rewriter.ops.fuse import apply_fuse
    from operator_profiler.rewriter.ops.reorder import apply_reorder
    from operator_profiler.rewriter.ops.change_layout import apply_change_layout
    from operator_profiler.rewriter.ops.buffer_sharing import apply_buffer_sharing

    if isinstance(op, FuseOp):
        return apply_fuse(gm, op)
    if isinstance(op, ReorderOp):
        return apply_reorder(gm, op)
    if isinstance(op, ChangeLayoutOp):
        return apply_change_layout(gm, op)
    if isinstance(op, BufferSharingOp):
        return apply_buffer_sharing(gm, op)
    raise RewriteValidationError(f"Unknown op type: {type(op)}")  # pragma: no cover


# ---------------------------------------------------------------------------
# HybridExecutor
# ---------------------------------------------------------------------------

class HybridExecutor:
    def __init__(
        self,
        gm: "torch.fx.GraphModule",
        plan: RewritePlan,
        config: ExecutorConfig | None = None,
    ) -> None:
        self._original_gm = gm
        self._plan = plan
        self._config = config or ExecutorConfig()

    def run(self) -> tuple["torch.fx.GraphModule", list[VerificationResult]]:
        """
        Apply all ops in ``plan.ops`` and return the final graph + per-op
        verification results.
        """
        plan = self._plan
        config = self._config
        original_gm = self._original_gm

        # 1. Validate plan version
        if plan.plan_version != DSL_VERSION:
            raise RewriteValidationError(
                f"Plan version '{plan.plan_version}' does not match "
                f"executor DSL version '{DSL_VERSION}'"
            )

        # 2. Pre-flight all ops against the current graph (before any mutation)
        working_gm = copy.deepcopy(original_gm)
        for op in plan.ops:
            _preflight_op(op, working_gm)

        # 3. Apply ops one by one with checkpoint/rollback
        results: list[VerificationResult] = []

        for op in plan.ops:
            checkpoint_gm = copy.deepcopy(working_gm)

            try:
                mutated_gm = _apply_op(working_gm, op)
            except RewriteValidationError:
                working_gm = checkpoint_gm
                raise

            if config.skip_verification:
                results.append(
                    VerificationResult(op_id=op.id, passed=True, max_abs_error=None)
                )
                working_gm = mutated_gm
                continue

            gate = VerificationGate(
                original_gm=original_gm,
                rewritten_gm=mutated_gm,
                op_id=op.id,
                atol=config.verification_atol,
                rtol=config.verification_rtol,
                device=config.device,
            )
            result = gate.verify()
            results.append(result)

            if result.passed:
                working_gm = mutated_gm
            else:
                working_gm = checkpoint_gm
                if not config.continue_on_verification_failure:
                    break

        return working_gm, results
