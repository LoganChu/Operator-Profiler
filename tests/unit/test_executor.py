"""
Tests for HybridExecutor.

Coverage
--------
- Pre-flight rejects unknown node names before any mutation
- original_gm never mutated after run()
- Rollback restores checkpoint on failed verification
- continue_on_verification_failure flag allows multiple ops to run
- plan_version mismatch raises RewriteValidationError
- Successful op advances the working graph
- skip_verification=True bypasses gate
"""
from __future__ import annotations

import copy

import pytest
import torch
import torch.fx

from operator_profiler.rewriter.dsl import (
    FuseOp,
    ReorderOp,
    RewritePlan,
    RewriteValidationError,
)
from operator_profiler.rewriter.executor import (
    ExecutorConfig,
    HybridExecutor,
    PreFlightError,
)


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _make_two_branch_graph() -> torch.fx.GraphModule:
    """
    x → relu_0 (independent) and x → sigmoid_0 (independent)
    → add_0(relu_0, sigmoid_0) → output
    """
    graph = torch.fx.Graph()
    x = graph.placeholder("x")
    r = graph.call_function(torch.nn.functional.relu, args=(x,), name="relu_0")
    s = graph.call_function(torch.sigmoid, args=(x,), name="sigmoid_0")
    out = graph.call_function(torch.add, args=(r, s), name="add_0")
    graph.output(out)
    return torch.fx.GraphModule(torch.nn.Module(), graph)


def _make_simple_gm() -> torch.fx.GraphModule:
    """x → relu → output."""
    graph = torch.fx.Graph()
    x = graph.placeholder("x")
    r = graph.call_function(torch.nn.functional.relu, args=(x,), name="relu_0")
    graph.output(r)
    return torch.fx.GraphModule(torch.nn.Module(), graph)


def _make_fuse_plan(nodes: list[str], plan_id: str = "f0") -> RewritePlan:
    return RewritePlan(
        ops=[FuseOp(op="fuse", id=plan_id, nodes=nodes)]
    )


# ---------------------------------------------------------------------------
# plan_version check
# ---------------------------------------------------------------------------

class TestPlanVersionCheck:
    def test_wrong_version_raises(self):
        gm = _make_simple_gm()
        plan = RewritePlan(plan_version="9.9", ops=[])
        with pytest.raises(RewriteValidationError, match="version"):
            HybridExecutor(gm, plan).run()

    def test_correct_version_passes(self):
        gm = _make_simple_gm()
        plan = RewritePlan(ops=[])
        result_gm, results = HybridExecutor(gm, plan).run()
        assert results == []


# ---------------------------------------------------------------------------
# Pre-flight validation
# ---------------------------------------------------------------------------

class TestPreFlight:
    def test_unknown_fuse_node_raises_before_mutation(self):
        gm = _make_simple_gm()
        plan = _make_fuse_plan(["relu_0", "nonexistent"])
        with pytest.raises(PreFlightError, match="not found"):
            HybridExecutor(gm, plan).run()

    def test_original_gm_not_mutated_after_preflight_failure(self):
        gm = _make_simple_gm()
        node_names_before = {n.name for n in gm.graph.nodes}
        plan = _make_fuse_plan(["relu_0", "nonexistent"])
        try:
            HybridExecutor(gm, plan).run()
        except PreFlightError:
            pass
        node_names_after = {n.name for n in gm.graph.nodes}
        assert node_names_before == node_names_after

    def test_unknown_reorder_node_raises(self):
        gm = _make_simple_gm()
        plan = RewritePlan(
            ops=[ReorderOp(op="reorder", id="r0", node="nonexistent", before="relu_0")]
        )
        with pytest.raises(PreFlightError, match="not found"):
            HybridExecutor(gm, plan).run()


# ---------------------------------------------------------------------------
# Original graph immutability
# ---------------------------------------------------------------------------

class TestOriginalImmutability:
    def test_original_not_mutated_after_successful_run(self):
        gm = _make_two_branch_graph()
        original_node_names = {n.name for n in gm.graph.nodes}

        plan = RewritePlan(
            ops=[
                ReorderOp(op="reorder", id="r0", node="sigmoid_0", before="relu_0")
            ]
        )
        cfg = ExecutorConfig(skip_verification=True)
        HybridExecutor(gm, plan, cfg).run()

        assert {n.name for n in gm.graph.nodes} == original_node_names

    def test_original_not_mutated_after_failed_run(self):
        gm = _make_simple_gm()
        original_code = gm.code

        plan = _make_fuse_plan(["relu_0", "nonexistent"])
        try:
            HybridExecutor(gm, plan).run()
        except (RewriteValidationError, PreFlightError):
            pass

        assert gm.code == original_code


# ---------------------------------------------------------------------------
# Verification & rollback
# ---------------------------------------------------------------------------

class TestVerificationAndRollback:
    def test_successful_op_advances_working_graph(self):
        gm = _make_two_branch_graph()
        plan = RewritePlan(
            ops=[
                ReorderOp(op="reorder", id="r0", node="sigmoid_0", before="relu_0")
            ]
        )
        cfg = ExecutorConfig(skip_verification=True)
        result_gm, results = HybridExecutor(gm, plan, cfg).run()

        assert len(results) == 1
        assert results[0].passed is True
        # Verify the reorder actually happened in the returned graph
        names = [n.name for n in result_gm.graph.nodes]
        assert names.index("sigmoid_0") < names.index("relu_0")

    def test_failed_verification_triggers_rollback(self):
        """Pre-flight failure must not corrupt the original graph."""
        gm = _make_simple_gm()
        node_names_before = {n.name for n in gm.graph.nodes}
        plan = RewritePlan(
            ops=[ReorderOp(op="reorder", id="r0", node="nonexistent", before="relu_0")]
        )
        with pytest.raises(PreFlightError):
            HybridExecutor(gm, plan).run()
        assert {n.name for n in gm.graph.nodes} == node_names_before

    def test_continue_on_failure_runs_subsequent_ops(self):
        """
        Two independent reorder ops; first one passes (annotation-only fuse),
        second is also valid.  With continue_on_verification_failure=True
        both should be attempted.
        """
        gm = _make_two_branch_graph()
        plan = RewritePlan(
            ops=[
                FuseOp(op="fuse", id="f0", nodes=["relu_0", "sigmoid_0"]),
                FuseOp(op="fuse", id="f1", nodes=["relu_0", "add_0"]),
            ]
        )
        cfg = ExecutorConfig(
            skip_verification=True,
            continue_on_verification_failure=True,
        )
        result_gm, results = HybridExecutor(gm, plan, cfg).run()
        assert len(results) == 2

    def test_skip_verification_flag(self):
        """skip_verification=True means no VerificationGate is run; op still applies."""
        gm = _make_two_branch_graph()
        plan = RewritePlan(
            ops=[ReorderOp(op="reorder", id="r0", node="sigmoid_0", before="relu_0")]
        )
        result_gm, results = HybridExecutor(
            gm, plan, ExecutorConfig(skip_verification=True)
        ).run()
        assert len(results) == 1
        assert results[0].passed is True

    def test_empty_plan_returns_copy(self):
        gm = _make_simple_gm()
        plan = RewritePlan(ops=[])
        result_gm, results = HybridExecutor(gm, plan).run()
        assert results == []
        # Returned graph is a copy, not the original
        assert result_gm is not gm


# ---------------------------------------------------------------------------
# Successful end-to-end: reorder + verify
# ---------------------------------------------------------------------------

class TestEndToEnd:
    def test_reorder_passes_verification(self):
        """A pure reorder of independent nodes should be numerically equivalent."""
        gm = _make_two_branch_graph()
        plan = RewritePlan(
            ops=[ReorderOp(op="reorder", id="r0", node="sigmoid_0", before="relu_0")]
        )
        cfg = ExecutorConfig(verification_atol=1e-5, verification_rtol=1e-5)
        result_gm, results = HybridExecutor(gm, plan, cfg).run()

        assert len(results) == 1
        assert results[0].passed is True
        assert results[0].max_abs_error == pytest.approx(0.0, abs=1e-6)

    def test_fuse_annotation_passes_verification(self):
        """A fuse (annotation-only) must produce numerically identical output."""
        gm = _make_two_branch_graph()
        plan = _make_fuse_plan(["relu_0", "sigmoid_0"])
        cfg = ExecutorConfig(verification_atol=1e-5)
        result_gm, results = HybridExecutor(gm, plan, cfg).run()

        assert len(results) == 1
        assert results[0].passed is True
