"""
Tests for apply_change_layout.

Coverage
--------
- .to(memory_format=...) call inserted before the target node
- .contiguous() inserted when insert_contiguous_after=True
- .contiguous() NOT inserted when insert_contiguous_after=False
- Non-layout-sensitive op raises RewriteValidationError
- Unknown node name raises RewriteValidationError
- graph.lint() passes after transform
"""
from __future__ import annotations

import pytest
import torch
import torch.fx

from operator_profiler.rewriter.dsl import ChangeLayoutOp, RewriteValidationError
from operator_profiler.rewriter.ops.change_layout import (
    LAYOUT_SENSITIVE_ATEN_OPS,
    apply_change_layout,
    _is_layout_sensitive,
)


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _node_ops(gm: torch.fx.GraphModule) -> list[str]:
    """Return list of (op, target_str) pairs for all non-placeholder/output nodes."""
    return [
        f"{n.op}:{n.target}"
        for n in gm.graph.nodes
        if n.op not in ("placeholder", "output")
    ]


def _make_conv_graph() -> torch.fx.GraphModule:
    """
    Graph: x, weight (placeholders) → conv2d → output.
    Uses torch.nn.functional.conv2d so the node target string contains "conv2d".
    Two placeholders avoids any get_attr / constant tensor issues.
    """
    graph = torch.fx.Graph()
    x = graph.placeholder("x")
    w = graph.placeholder("weight")
    conv_node = graph.call_function(
        torch.nn.functional.conv2d,
        args=(x, w),
        name="conv2d_0",
    )
    graph.output(conv_node)
    gm = torch.fx.GraphModule(torch.nn.Module(), graph)
    return gm


def _make_relu_graph() -> torch.fx.GraphModule:
    """Graph: x → relu → output (relu is NOT layout-sensitive)."""
    graph = torch.fx.Graph()
    x = graph.placeholder("x")
    r = graph.call_function(torch.nn.functional.relu, args=(x,), name="relu_0")
    graph.output(r)
    return torch.fx.GraphModule(torch.nn.Module(), graph)


# ---------------------------------------------------------------------------
# _is_layout_sensitive
# ---------------------------------------------------------------------------

class TestIsLayoutSensitive:
    def test_conv_node_is_sensitive(self):
        gm = _make_conv_graph()
        conv_node = next(n for n in gm.graph.nodes if "conv" in n.name)
        assert _is_layout_sensitive(conv_node) is True

    def test_relu_node_is_not_sensitive(self):
        gm = _make_relu_graph()
        relu_node = next(n for n in gm.graph.nodes if n.op == "call_function")
        assert _is_layout_sensitive(relu_node) is False


# ---------------------------------------------------------------------------
# apply_change_layout
# ---------------------------------------------------------------------------

class TestApplyChangeLayout:
    def test_to_call_inserted(self):
        gm = _make_conv_graph()
        op = ChangeLayoutOp(
            op="change_layout",
            id="cl0",
            target_node="conv2d_0",
            current_format="NCHW",
            target_format="NHWC",
        )
        new_gm = apply_change_layout(gm, op)
        # The transformed graph should have a "to" call_method node
        method_names = [
            n.target for n in new_gm.graph.nodes if n.op == "call_method"
        ]
        assert "to" in method_names

    def test_contiguous_inserted_when_true(self):
        gm = _make_conv_graph()
        op = ChangeLayoutOp(
            op="change_layout",
            id="cl0",
            target_node="conv2d_0",
            current_format="NCHW",
            target_format="NHWC",
            insert_contiguous_after=True,
        )
        new_gm = apply_change_layout(gm, op)
        method_names = [
            n.target for n in new_gm.graph.nodes if n.op == "call_method"
        ]
        assert "contiguous" in method_names

    def test_contiguous_not_inserted_when_false(self):
        gm = _make_conv_graph()
        op = ChangeLayoutOp(
            op="change_layout",
            id="cl0",
            target_node="conv2d_0",
            current_format="NCHW",
            target_format="NHWC",
            insert_contiguous_after=False,
        )
        new_gm = apply_change_layout(gm, op)
        method_names = [
            n.target for n in new_gm.graph.nodes if n.op == "call_method"
        ]
        assert "contiguous" not in method_names

    def test_non_layout_sensitive_op_rejected(self):
        gm = _make_relu_graph()
        op = ChangeLayoutOp(
            op="change_layout",
            id="cl0",
            target_node="relu_0",
            current_format="NCHW",
            target_format="NHWC",
        )
        with pytest.raises(RewriteValidationError, match="not a layout-sensitive"):
            apply_change_layout(gm, op)

    def test_unknown_node_rejected(self):
        gm = _make_conv_graph()
        op = ChangeLayoutOp(
            op="change_layout",
            id="cl0",
            target_node="nonexistent",
            current_format="NCHW",
            target_format="NHWC",
        )
        with pytest.raises(RewriteValidationError, match="not found"):
            apply_change_layout(gm, op)

    def test_returns_new_graph_module(self):
        """Transformer returns a new GraphModule — original must be unchanged."""
        gm = _make_conv_graph()
        orig_node_count = len(list(gm.graph.nodes))
        op = ChangeLayoutOp(
            op="change_layout",
            id="cl0",
            target_node="conv2d_0",
            current_format="NCHW",
            target_format="NHWC",
        )
        new_gm = apply_change_layout(gm, op)
        assert new_gm is not gm
        # Original graph node count unchanged
        assert len(list(gm.graph.nodes)) == orig_node_count
        # New graph has more nodes (at least one .to() call added)
        assert len(list(new_gm.graph.nodes)) > orig_node_count

    def test_graph_lint_passes(self):
        gm = _make_conv_graph()
        op = ChangeLayoutOp(
            op="change_layout",
            id="cl0",
            target_node="conv2d_0",
            current_format="NCHW",
            target_format="NHWC",
        )
        new_gm = apply_change_layout(gm, op)
        new_gm.graph.lint()
