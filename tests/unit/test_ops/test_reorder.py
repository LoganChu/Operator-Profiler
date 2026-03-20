"""
Tests for apply_reorder.

Coverage
--------
- Node appears at new position after reorder
- Data dependency violation raises RewriteValidationError
- graph.lint() passes after reorder
- Unknown node name raises RewriteValidationError
"""
from __future__ import annotations

import pytest
import torch
import torch.fx

from operator_profiler.rewriter.dsl import ReorderOp, RewriteValidationError
from operator_profiler.rewriter.ops.reorder import apply_reorder, _topo_reachable


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------

def _make_chain_graph() -> torch.fx.GraphModule:
    """
    Graph: x → relu → sigmoid → tanh → output
    Nodes: placeholder(x), relu, sigmoid, tanh, output
    """
    graph = torch.fx.Graph()
    x = graph.placeholder("x")
    r = graph.call_function(torch.nn.functional.relu, args=(x,), name="relu")
    s = graph.call_function(torch.sigmoid, args=(r,), name="sigmoid")
    t = graph.call_function(torch.tanh, args=(s,), name="tanh")
    graph.output(t)
    return torch.fx.GraphModule(torch.nn.Module(), graph)


def _node_order(gm: torch.fx.GraphModule) -> list[str]:
    return [n.name for n in gm.graph.nodes]


# ---------------------------------------------------------------------------
# Tests
# ---------------------------------------------------------------------------

class TestApplyReorder:
    def test_reorder_before_moves_node(self):
        gm = _make_chain_graph()
        # Move sigmoid before relu (sigmoid doesn't depend on relu in this move — wait, it does)
        # Instead: move tanh before sigmoid — but tanh depends on sigmoid.
        # Let's reorder a node that is truly independent.
        # We need a graph with an independent node.
        graph = torch.fx.Graph()
        x = graph.placeholder("x")
        # Two independent branches
        r = graph.call_function(torch.nn.functional.relu, args=(x,), name="relu")
        s = graph.call_function(torch.sigmoid, args=(x,), name="sigmoid")
        # Combine
        out = graph.call_function(torch.add, args=(r, s), name="add")
        graph.output(out)
        gm = torch.fx.GraphModule(torch.nn.Module(), graph)

        order_before = _node_order(gm)
        assert order_before.index("relu") < order_before.index("sigmoid")

        # Move sigmoid before relu (both depend only on x, so no violation)
        op = ReorderOp(op="reorder", id="r0", node="sigmoid", before="relu")
        apply_reorder(gm, op)

        order_after = _node_order(gm)
        assert order_after.index("sigmoid") < order_after.index("relu")

    def test_reorder_after_moves_node(self):
        graph = torch.fx.Graph()
        x = graph.placeholder("x")
        r = graph.call_function(torch.nn.functional.relu, args=(x,), name="relu")
        s = graph.call_function(torch.sigmoid, args=(x,), name="sigmoid")
        out = graph.call_function(torch.add, args=(r, s), name="add")
        graph.output(out)
        gm = torch.fx.GraphModule(torch.nn.Module(), graph)

        op = ReorderOp(op="reorder", id="r0", node="relu", after="sigmoid")
        apply_reorder(gm, op)

        order = _node_order(gm)
        assert order.index("relu") > order.index("sigmoid")

    def test_dependency_violation_before_raises(self):
        """Cannot move sigmoid before relu because sigmoid depends on relu."""
        gm = _make_chain_graph()
        op = ReorderOp(op="reorder", id="r0", node="sigmoid", before="relu")
        with pytest.raises(RewriteValidationError, match="depends on"):
            apply_reorder(gm, op)

    def test_dependency_violation_after_raises(self):
        """Cannot move relu after sigmoid because sigmoid depends on relu."""
        gm = _make_chain_graph()
        op = ReorderOp(op="reorder", id="r0", node="relu", after="sigmoid")
        with pytest.raises(RewriteValidationError, match="depends on"):
            apply_reorder(gm, op)

    def test_unknown_node_raises(self):
        gm = _make_chain_graph()
        op = ReorderOp(op="reorder", id="r0", node="nonexistent", before="relu")
        with pytest.raises(RewriteValidationError, match="not found"):
            apply_reorder(gm, op)

    def test_unknown_anchor_raises(self):
        gm = _make_chain_graph()
        op = ReorderOp(op="reorder", id="r0", node="relu", before="nonexistent")
        with pytest.raises(RewriteValidationError, match="not found"):
            apply_reorder(gm, op)

    def test_graph_lint_passes_after_reorder(self):
        graph = torch.fx.Graph()
        x = graph.placeholder("x")
        r = graph.call_function(torch.nn.functional.relu, args=(x,), name="relu")
        s = graph.call_function(torch.sigmoid, args=(x,), name="sigmoid")
        out = graph.call_function(torch.add, args=(r, s), name="add")
        graph.output(out)
        gm = torch.fx.GraphModule(torch.nn.Module(), graph)

        op = ReorderOp(op="reorder", id="r0", node="sigmoid", before="relu")
        apply_reorder(gm, op)
        # lint() raises if the graph is invalid — no exception = pass
        gm.graph.lint()


class TestTopoReachable:
    def test_direct_dependency(self):
        graph = torch.fx.Graph()
        x = graph.placeholder("x")
        r = graph.call_function(torch.nn.functional.relu, args=(x,), name="relu")
        s = graph.call_function(torch.sigmoid, args=(r,), name="sigmoid")
        graph.output(s)
        assert _topo_reachable(r, s) is True

    def test_no_dependency(self):
        graph = torch.fx.Graph()
        x = graph.placeholder("x")
        r = graph.call_function(torch.nn.functional.relu, args=(x,), name="relu")
        s = graph.call_function(torch.sigmoid, args=(x,), name="sigmoid")
        graph.output(graph.call_function(torch.add, args=(r, s)))
        assert _topo_reachable(r, s) is False
        assert _topo_reachable(s, r) is False

    def test_self_reachable(self):
        graph = torch.fx.Graph()
        x = graph.placeholder("x")
        assert _topo_reachable(x, x) is True
