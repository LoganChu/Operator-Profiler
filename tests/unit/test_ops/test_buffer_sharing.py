"""
Tests for apply_buffer_sharing and liveness analysis.

Coverage
--------
- Non-overlapping live ranges → buffer_alias_of meta set
- Overlapping live ranges → RewriteValidationError raised
- validate_liveness=False skips the check
- meta["buffer_alias_of"] set correctly on target node
- Unknown node names raise
"""
from __future__ import annotations

import pytest
import torch
import torch.fx

from operator_profiler.rewriter.dsl import BufferSharingOp, RewriteValidationError
from operator_profiler.rewriter.ops.buffer_sharing import (
    _compute_liveness,
    _live_ranges_overlap,
    apply_buffer_sharing,
)


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _make_sequential_graph() -> torch.fx.GraphModule:
    """
    Sequential graph: x → relu → sigmoid → tanh → output.

    Live ranges:
      x:       def=0, last_use by relu and sigmoid and tanh (indirectly) → 0..1 (only used by relu)
      relu:    def=1, used by sigmoid → 1..2
      sigmoid: def=2, used by tanh → 2..3
      tanh:    def=3, used by output → 3..4
    """
    graph = torch.fx.Graph()
    x = graph.placeholder("x")
    r = graph.call_function(torch.nn.functional.relu, args=(x,), name="relu")
    s = graph.call_function(torch.sigmoid, args=(r,), name="sigmoid")
    t = graph.call_function(torch.tanh, args=(s,), name="tanh")
    graph.output(t)
    return torch.fx.GraphModule(torch.nn.Module(), graph)


def _make_fan_out_graph() -> torch.fx.GraphModule:
    """
    x → relu, x → sigmoid → add(relu, sigmoid) → output.

    relu and sigmoid: both defined after x; relu used only at add, sigmoid used only at add.
    They are live at the same time.
    """
    graph = torch.fx.Graph()
    x = graph.placeholder("x")
    r = graph.call_function(torch.nn.functional.relu, args=(x,), name="relu")
    s = graph.call_function(torch.sigmoid, args=(x,), name="sigmoid")
    out = graph.call_function(torch.add, args=(r, s), name="add")
    graph.output(out)
    return torch.fx.GraphModule(torch.nn.Module(), graph)


# ---------------------------------------------------------------------------
# Liveness unit tests
# ---------------------------------------------------------------------------

class TestComputeLiveness:
    def test_sequential_ranges(self):
        gm = _make_sequential_graph()
        liveness = _compute_liveness(gm)
        assert "relu" in liveness
        assert "sigmoid" in liveness
        # relu is defined before sigmoid and last used by sigmoid
        relu_def, relu_last = liveness["relu"]
        sig_def, sig_last = liveness["sigmoid"]
        assert relu_def < sig_def
        # relu's last use is at sigmoid's position
        assert relu_last == sig_def

    def test_all_nodes_present(self):
        gm = _make_sequential_graph()
        liveness = _compute_liveness(gm)
        names = {n.name for n in gm.graph.nodes}
        assert set(liveness.keys()) == names


class TestLiveRangesOverlap:
    def test_non_overlapping(self):
        assert _live_ranges_overlap((0, 2), (3, 5)) is False
        assert _live_ranges_overlap((3, 5), (0, 2)) is False

    def test_overlapping(self):
        assert _live_ranges_overlap((0, 3), (2, 5)) is True
        assert _live_ranges_overlap((2, 5), (0, 3)) is True

    def test_adjacent_not_overlapping(self):
        # [0,2] and [3,5] — no shared point
        assert _live_ranges_overlap((0, 2), (3, 5)) is False

    def test_touching(self):
        # [0,3] and [3,5] — share point 3
        assert _live_ranges_overlap((0, 3), (3, 5)) is True

    def test_contained(self):
        assert _live_ranges_overlap((1, 4), (2, 3)) is True


# ---------------------------------------------------------------------------
# apply_buffer_sharing tests
# ---------------------------------------------------------------------------

class TestApplyBufferSharing:
    def test_non_overlapping_sets_meta(self):
        """relu and tanh are sequential, non-overlapping."""
        gm = _make_sequential_graph()
        liveness = _compute_liveness(gm)
        relu_range = liveness["relu"]
        tanh_range = liveness["tanh"]
        # Confirm they don't overlap
        assert not _live_ranges_overlap(relu_range, tanh_range)

        op = BufferSharingOp(
            op="buffer_sharing",
            id="bs0",
            source_node="relu",
            target_node="tanh",
            validate_liveness=True,
        )
        apply_buffer_sharing(gm, op)

        tanh_node = next(n for n in gm.graph.nodes if n.name == "tanh")
        assert tanh_node.meta["buffer_alias_of"] == "relu"

    def test_overlapping_raises(self):
        """relu and sigmoid in fan-out graph are live at the same time."""
        gm = _make_fan_out_graph()
        op = BufferSharingOp(
            op="buffer_sharing",
            id="bs0",
            source_node="relu",
            target_node="sigmoid",
            validate_liveness=True,
        )
        with pytest.raises(RewriteValidationError, match="overlap"):
            apply_buffer_sharing(gm, op)

    def test_validate_liveness_false_skips_check(self):
        """With validate_liveness=False, overlapping ranges are allowed."""
        gm = _make_fan_out_graph()
        op = BufferSharingOp(
            op="buffer_sharing",
            id="bs0",
            source_node="relu",
            target_node="sigmoid",
            validate_liveness=False,
        )
        apply_buffer_sharing(gm, op)  # should not raise
        sig_node = next(n for n in gm.graph.nodes if n.name == "sigmoid")
        assert sig_node.meta["buffer_alias_of"] == "relu"

    def test_unknown_source_raises(self):
        gm = _make_sequential_graph()
        op = BufferSharingOp(
            op="buffer_sharing",
            id="bs0",
            source_node="nonexistent",
            target_node="tanh",
        )
        with pytest.raises(RewriteValidationError, match="not found"):
            apply_buffer_sharing(gm, op)

    def test_unknown_target_raises(self):
        gm = _make_sequential_graph()
        op = BufferSharingOp(
            op="buffer_sharing",
            id="bs0",
            source_node="relu",
            target_node="nonexistent",
        )
        with pytest.raises(RewriteValidationError, match="not found"):
            apply_buffer_sharing(gm, op)

    def test_graph_lint_passes(self):
        gm = _make_sequential_graph()
        op = BufferSharingOp(
            op="buffer_sharing",
            id="bs0",
            source_node="relu",
            target_node="tanh",
            validate_liveness=True,
        )
        apply_buffer_sharing(gm, op)
        gm.graph.lint()
