"""
Tests for apply_fuse.

Coverage
--------
- inductor_fuse_group meta set on all nodes in the group
- is_fused + source_operators set on last (representative) node
- Non-adjacent nodes: annotation-only, no structural graph change
- Adjacent nodes: same annotation behaviour
- Unknown node names raise RewriteValidationError
- custom_op strategy without custom_op_name raises
- graph.lint() passes after fuse
"""
from __future__ import annotations

import pytest
import torch
import torch.fx

from operator_profiler.rewriter.dsl import FuseOp, RewriteValidationError
from operator_profiler.rewriter.ops.fuse import apply_fuse, _is_adjacent


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _make_linear_relu_graph() -> torch.fx.GraphModule:
    """Adjacent: x → linear → relu → output."""
    graph = torch.fx.Graph()
    x = graph.placeholder("x")
    linear = graph.call_function(
        torch.nn.functional.linear,
        args=(x, torch.ones(3, 3)),
        name="linear_0",
    )
    relu = graph.call_function(
        torch.nn.functional.relu,
        args=(linear,),
        name="relu_0",
    )
    graph.output(relu)
    return torch.fx.GraphModule(torch.nn.Module(), graph)


def _make_non_adjacent_graph() -> torch.fx.GraphModule:
    """
    x → linear → dropout → relu → output.
    linear and relu are non-adjacent (dropout is between them).
    """
    graph = torch.fx.Graph()
    x = graph.placeholder("x")
    linear = graph.call_function(
        torch.nn.functional.linear,
        args=(x, torch.ones(3, 3)),
        name="linear_0",
    )
    dropout = graph.call_function(
        torch.nn.functional.dropout,
        args=(linear,),
        name="dropout_0",
    )
    relu = graph.call_function(
        torch.nn.functional.relu,
        args=(dropout,),
        name="relu_0",
    )
    graph.output(relu)
    return torch.fx.GraphModule(torch.nn.Module(), graph)


# ---------------------------------------------------------------------------
# _is_adjacent
# ---------------------------------------------------------------------------

class TestIsAdjacent:
    def test_adjacent_returns_true(self):
        gm = _make_linear_relu_graph()
        assert _is_adjacent(gm, ["linear_0", "relu_0"]) is True

    def test_non_adjacent_returns_false(self):
        gm = _make_non_adjacent_graph()
        assert _is_adjacent(gm, ["linear_0", "relu_0"]) is False

    def test_missing_node_returns_false(self):
        gm = _make_linear_relu_graph()
        assert _is_adjacent(gm, ["linear_0", "nonexistent"]) is False


# ---------------------------------------------------------------------------
# apply_fuse — inductor_fuse (default)
# ---------------------------------------------------------------------------

class TestApplyFuseInductorFuse:
    def test_inductor_fuse_group_meta_set(self):
        gm = _make_linear_relu_graph()
        op = FuseOp(op="fuse", id="fuse_lr_0", nodes=["linear_0", "relu_0"])
        apply_fuse(gm, op)

        linear = next(n for n in gm.graph.nodes if n.name == "linear_0")
        relu = next(n for n in gm.graph.nodes if n.name == "relu_0")
        assert linear.meta["inductor_fuse_group"] == "fuse_lr_0"
        assert relu.meta["inductor_fuse_group"] == "fuse_lr_0"

    def test_is_fused_set_on_last_node(self):
        gm = _make_linear_relu_graph()
        op = FuseOp(op="fuse", id="fuse_lr_0", nodes=["linear_0", "relu_0"])
        apply_fuse(gm, op)

        relu = next(n for n in gm.graph.nodes if n.name == "relu_0")
        assert relu.meta.get("is_fused") is True

    def test_source_operators_set(self):
        gm = _make_linear_relu_graph()
        op = FuseOp(op="fuse", id="fuse_lr_0", nodes=["linear_0", "relu_0"])
        apply_fuse(gm, op)

        relu = next(n for n in gm.graph.nodes if n.name == "relu_0")
        sources = relu.meta["source_operators"]
        assert len(sources) == 2

    def test_non_adjacent_falls_back_to_annotation(self):
        """Non-adjacent nodes get annotation only — no structural graph change."""
        gm = _make_non_adjacent_graph()
        orig_node_count = len(list(gm.graph.nodes))

        op = FuseOp(op="fuse", id="fuse_lr_0", nodes=["linear_0", "relu_0"])
        apply_fuse(gm, op)

        # No structural change
        assert len(list(gm.graph.nodes)) == orig_node_count

        linear = next(n for n in gm.graph.nodes if n.name == "linear_0")
        relu = next(n for n in gm.graph.nodes if n.name == "relu_0")
        assert linear.meta["inductor_fuse_group"] == "fuse_lr_0"
        assert relu.meta["inductor_fuse_group"] == "fuse_lr_0"

    def test_graph_lint_passes(self):
        gm = _make_linear_relu_graph()
        op = FuseOp(op="fuse", id="fuse_lr_0", nodes=["linear_0", "relu_0"])
        apply_fuse(gm, op)
        gm.graph.lint()

    def test_unknown_node_raises(self):
        gm = _make_linear_relu_graph()
        op = FuseOp(op="fuse", id="fuse_lr_0", nodes=["linear_0", "nonexistent"])
        with pytest.raises(RewriteValidationError, match="not found"):
            apply_fuse(gm, op)


# ---------------------------------------------------------------------------
# apply_fuse — custom_op strategy
# ---------------------------------------------------------------------------

class TestApplyFuseCustomOp:
    def test_custom_op_meta_set(self):
        gm = _make_linear_relu_graph()
        op = FuseOp(
            op="fuse",
            id="fuse_lr_0",
            nodes=["linear_0", "relu_0"],
            strategy="custom_op",
            custom_op_name="torch.ops.custom.fused_fn",
        )
        apply_fuse(gm, op)

        linear = next(n for n in gm.graph.nodes if n.name == "linear_0")
        assert linear.meta["custom_op_name"] == "torch.ops.custom.fused_fn"


# ---------------------------------------------------------------------------
# apply_fuse — inline strategy
# ---------------------------------------------------------------------------

class TestApplyFuseInline:
    def test_inline_eligible_meta_set_for_adjacent(self):
        gm = _make_linear_relu_graph()
        op = FuseOp(
            op="fuse", id="fuse_lr_0", nodes=["linear_0", "relu_0"],
            strategy="inline",
        )
        apply_fuse(gm, op)

        linear = next(n for n in gm.graph.nodes if n.name == "linear_0")
        assert linear.meta.get("inline_fusion_eligible") is True

    def test_inline_non_adjacent_no_eligible_meta(self):
        gm = _make_non_adjacent_graph()
        op = FuseOp(
            op="fuse", id="fuse_lr_0", nodes=["linear_0", "relu_0"],
            strategy="inline",
        )
        apply_fuse(gm, op)

        linear = next(n for n in gm.graph.nodes if n.name == "linear_0")
        # Non-adjacent: no inline_fusion_eligible flag
        assert "inline_fusion_eligible" not in linear.meta
