"""Unit tests for explain_node() (summarizer/explain.py)."""
from __future__ import annotations

import pytest

from operator_profiler.summarizer.diff import compute_diff
from operator_profiler.summarizer.explain import explain_node, _normalise_node_id


class TestNormaliseNodeId:
    def test_double_underscore_becomes_colon(self):
        assert _normalise_node_id("aten__linear_0") == "aten::linear_0"

    def test_already_colon_unchanged(self):
        assert _normalise_node_id("aten::linear_0") == "aten::linear_0"

    def test_no_underscore_unchanged(self):
        assert _normalise_node_id("relu") == "relu"


class TestExplainNodeFound:
    def test_returns_string(self, before_profile, after_profile_exact, loop_result_with_history):
        diff = compute_diff(before_profile, after_profile_exact, plan=None)
        result = explain_node("aten::linear_0", diff, before_profile, loop_result_with_history)
        assert isinstance(result, str)
        assert len(result) > 0

    def test_shows_operator_name(self, before_profile, after_profile_exact, loop_result_with_history):
        diff = compute_diff(before_profile, after_profile_exact, plan=None)
        result = explain_node("aten::linear_0", diff, before_profile, loop_result_with_history)
        assert "aten::linear" in result

    def test_shows_duration_before(self, before_profile, after_profile_exact, loop_result_with_history):
        diff = compute_diff(before_profile, after_profile_exact, plan=None)
        result = explain_node("aten::linear_0", diff, before_profile, loop_result_with_history)
        assert "3.000 ms" in result

    def test_shows_duration_after(self, before_profile, after_profile_exact, loop_result_with_history):
        diff = compute_diff(before_profile, after_profile_exact, plan=None)
        result = explain_node("aten::linear_0", diff, before_profile, loop_result_with_history)
        assert "1.500 ms" in result

    def test_shows_speedup(self, before_profile, after_profile_exact, loop_result_with_history):
        diff = compute_diff(before_profile, after_profile_exact, plan=None)
        result = explain_node("aten::linear_0", diff, before_profile, loop_result_with_history)
        assert "2.00x" in result

    def test_shows_bottleneck_before(self, before_profile, after_profile_exact, loop_result_with_history):
        diff = compute_diff(before_profile, after_profile_exact, plan=None)
        result = explain_node("aten::linear_0", diff, before_profile, loop_result_with_history)
        assert "memory_bound" in result

    def test_history_context_mentions_iteration(
        self, before_profile, after_profile_exact, loop_result_with_history
    ):
        diff = compute_diff(before_profile, after_profile_exact, plan=None)
        result = explain_node("aten::linear_0", diff, before_profile, loop_result_with_history)
        # linear_0 is worst_op_id in both iterations of the fixture
        assert "iteration" in result.lower() or "iteration(s)" in result

    def test_double_underscore_normalised(
        self, before_profile, after_profile_exact, loop_result_with_history
    ):
        diff = compute_diff(before_profile, after_profile_exact, plan=None)
        result1 = explain_node("aten::linear_0", diff, before_profile, loop_result_with_history)
        result2 = explain_node("aten__linear_0", diff, before_profile, loop_result_with_history)
        assert result1 == result2

    def test_shows_rewrite_ops_when_present(
        self, before_profile, after_profile_exact, fuse_plan, loop_result_with_history
    ):
        diff = compute_diff(before_profile, after_profile_exact, fuse_plan)
        result = explain_node("aten::linear_0", diff, before_profile, loop_result_with_history)
        assert "fuse_linear_relu" in result


class TestExplainNodeNotFound:
    def test_returns_error_message(self, before_profile, after_profile_exact, loop_result_with_history):
        diff = compute_diff(before_profile, after_profile_exact, plan=None)
        result = explain_node("aten::nonexistent_0", diff, before_profile, loop_result_with_history)
        assert "not found" in result.lower()

    def test_lists_available_nodes(self, before_profile, after_profile_exact, loop_result_with_history):
        diff = compute_diff(before_profile, after_profile_exact, plan=None)
        result = explain_node("aten::nonexistent_0", diff, before_profile, loop_result_with_history)
        assert "aten::linear_0" in result


class TestExplainFusedNode:
    def test_shows_fused_into(
        self, before_profile, after_profile_fused, fuse_plan, loop_result_with_history
    ):
        diff = compute_diff(before_profile, after_profile_fused, fuse_plan)
        result = explain_node("aten::linear_0", diff, before_profile, loop_result_with_history)
        # Fused-into nodes have duration_after_ns = combined after
        assert "2.000 ms" in result or "fused" in result.lower() or "AFTER" in result
