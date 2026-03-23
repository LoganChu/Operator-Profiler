"""Unit tests for ProfileDiff computation (summarizer/diff.py)."""
from __future__ import annotations

import pytest

from operator_profiler.rewriter.dsl import FuseOp, ReorderOp, RewritePlan
from operator_profiler.schema.profile import (
    AggregatedMetrics,
    CaptureMetadata,
    OperatorAttributedProfile,
    OperatorRecord,
)
from operator_profiler.summarizer.diff import compute_diff
from operator_profiler.summarizer.schema import ProfileDiff


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _meta():
    return CaptureMetadata(
        model_name="M",
        torch_version="2.3.0",
        capture_timestamp_utc="2026-03-22T00:00:00+00:00",
        device_name="A100",
    )


def _op(op_id, op_name, call_index, duration_ns, bottleneck="memory_bound"):
    return OperatorRecord(
        operator_id=op_id,
        operator_name=op_name,
        call_index=call_index,
        aggregated=AggregatedMetrics(
            total_duration_ns=duration_ns,
            kernel_count=1,
            bottleneck_classification=bottleneck,
        ),
    )


def _profile(*ops) -> OperatorAttributedProfile:
    return OperatorAttributedProfile(capture_metadata=_meta(), operators=list(ops))


# ---------------------------------------------------------------------------
# Pass 1: Exact matching
# ---------------------------------------------------------------------------

class TestExactMatching:
    def test_two_ops_match(self, before_profile, after_profile_exact):
        diff = compute_diff(before_profile, after_profile_exact, plan=None)
        exact = [d for d in diff.operator_diffs if d.match_type == "exact"]
        assert len(exact) == 2

    def test_speedup_computed(self, before_profile, after_profile_exact):
        diff = compute_diff(before_profile, after_profile_exact, plan=None)
        linear_diff = next(
            d for d in diff.operator_diffs if d.operator_name == "aten::linear"
        )
        # 3_000_000 / 1_500_000 == 2.0
        assert linear_diff.speedup == pytest.approx(2.0)

    def test_delta_duration_positive(self, before_profile, after_profile_exact):
        diff = compute_diff(before_profile, after_profile_exact, plan=None)
        for d in diff.operator_diffs:
            if d.match_type == "exact" and d.delta_duration_ns is not None:
                assert d.delta_duration_ns > 0

    def test_total_speedup(self, before_profile, after_profile_exact):
        diff = compute_diff(before_profile, after_profile_exact, plan=None)
        # before: 4_000_000 ns, after: 2_000_000 ns → 2.0x
        assert diff.total_speedup == pytest.approx(2.0)

    def test_wall_time_saved(self, before_profile, after_profile_exact):
        diff = compute_diff(before_profile, after_profile_exact, plan=None)
        assert diff.wall_time_saved_ns == 2_000_000

    def test_bottleneck_changed_flag(self, before_profile, after_profile_exact):
        diff = compute_diff(before_profile, after_profile_exact, plan=None)
        # linear: memory_bound → compute_bound, relu: compute_bound → compute_bound
        linear_diff = next(
            d for d in diff.operator_diffs if d.operator_name == "aten::linear"
        )
        relu_diff = next(
            d for d in diff.operator_diffs if d.operator_name == "aten::relu"
        )
        assert linear_diff.bottleneck_changed is True
        assert relu_diff.bottleneck_changed is False

    def test_model_name_from_before(self, before_profile, after_profile_exact):
        diff = compute_diff(before_profile, after_profile_exact, plan=None)
        assert diff.model_name == "TestModel"

    def test_no_unmatched_when_all_match(self, before_profile, after_profile_exact):
        diff = compute_diff(before_profile, after_profile_exact, plan=None)
        assert diff.unmatched_before == []
        assert diff.unmatched_after == []


# ---------------------------------------------------------------------------
# Pass 1: No plan (plan=None)
# ---------------------------------------------------------------------------

class TestNoPlan:
    def test_none_plan_is_ok(self, before_profile, after_profile_exact):
        diff = compute_diff(before_profile, after_profile_exact, plan=None)
        assert isinstance(diff, ProfileDiff)

    def test_rewrite_ops_empty_when_no_plan(self, before_profile, after_profile_exact):
        diff = compute_diff(before_profile, after_profile_exact, plan=None)
        for d in diff.operator_diffs:
            assert d.rewrite_ops_applied == []


# ---------------------------------------------------------------------------
# Pass 2: Fusion resolution
# ---------------------------------------------------------------------------

class TestFusedOperatorMatching:
    def test_constituents_get_fused_into(self, before_profile, after_profile_fused, fuse_plan):
        diff = compute_diff(before_profile, after_profile_fused, fuse_plan)
        fused = [d for d in diff.operator_diffs if d.match_type == "fused_into"]
        assert len(fused) == 2

    def test_fused_op_id_after_points_to_fused_record(
        self, before_profile, after_profile_fused, fuse_plan
    ):
        diff = compute_diff(before_profile, after_profile_fused, fuse_plan)
        fused = [d for d in diff.operator_diffs if d.match_type == "fused_into"]
        for d in fused:
            assert d.operator_id_after == "fused_linear_relu_0"

    def test_fusion_partners_populated(self, before_profile, after_profile_fused, fuse_plan):
        diff = compute_diff(before_profile, after_profile_fused, fuse_plan)
        linear_diff = next(
            d for d in diff.operator_diffs if d.operator_name == "aten::linear"
        )
        assert "aten::relu_0" in linear_diff.fusion_partners

    def test_speedup_from_sum_of_constituents(
        self, before_profile, after_profile_fused, fuse_plan
    ):
        diff = compute_diff(before_profile, after_profile_fused, fuse_plan)
        fused = [d for d in diff.operator_diffs if d.match_type == "fused_into"]
        # sum before = 4_000_000, fused after = 2_000_000 → speedup 2.0
        for d in fused:
            assert d.speedup == pytest.approx(2.0)

    def test_rewrite_op_id_attached(self, before_profile, after_profile_fused, fuse_plan):
        diff = compute_diff(before_profile, after_profile_fused, fuse_plan)
        linear_diff = next(
            d for d in diff.operator_diffs if d.operator_name == "aten::linear"
        )
        assert "fuse_linear_relu" in linear_diff.rewrite_ops_applied


# ---------------------------------------------------------------------------
# Pass 3 & 4: Removed / new operators
# ---------------------------------------------------------------------------

class TestUnmatchedOperators:
    def test_removed_when_after_is_empty(self, before_profile):
        empty_after = OperatorAttributedProfile(
            capture_metadata=before_profile.capture_metadata,
            operators=[],
        )
        diff = compute_diff(before_profile, empty_after, plan=None)
        removed = [d for d in diff.operator_diffs if d.match_type == "removed"]
        assert len(removed) == 2

    def test_new_when_extra_op_in_after(self, before_profile, after_profile_exact):
        extra_op = OperatorRecord(
            operator_id="aten::gelu_0",
            operator_name="aten::gelu",
            call_index=0,
            aggregated=AggregatedMetrics(
                total_duration_ns=500_000, kernel_count=1
            ),
        )
        after_with_extra = OperatorAttributedProfile(
            capture_metadata=after_profile_exact.capture_metadata,
            operators=after_profile_exact.operators + [extra_op],
        )
        diff = compute_diff(before_profile, after_with_extra, plan=None)
        new_ops = [d for d in diff.operator_diffs if d.match_type == "new"]
        assert any(d.operator_name == "aten::gelu" for d in new_ops)


# ---------------------------------------------------------------------------
# RewriteOp indexing
# ---------------------------------------------------------------------------

class TestRewriteOpIndexing:
    def test_reorder_op_indexed(self, before_profile, after_profile_exact):
        plan = RewritePlan(ops=[
            ReorderOp(op="reorder", id="r1", node="aten::relu_0", after="aten::linear_0")
        ])
        diff = compute_diff(before_profile, after_profile_exact, plan)
        relu_diff = next(
            d for d in diff.operator_diffs if d.operator_name == "aten::relu"
        )
        assert "r1" in relu_diff.rewrite_ops_applied

    def test_unrelated_op_has_no_rewrite_ops(self, before_profile, after_profile_exact):
        plan = RewritePlan(ops=[
            ReorderOp(op="reorder", id="r1", node="aten::relu_0", after="aten::linear_0")
        ])
        diff = compute_diff(before_profile, after_profile_exact, plan)
        linear_diff = next(
            d for d in diff.operator_diffs if d.operator_name == "aten::linear"
        )
        assert "r1" not in linear_diff.rewrite_ops_applied


# ---------------------------------------------------------------------------
# Top bottlenecks
# ---------------------------------------------------------------------------

class TestTopBottlenecks:
    def test_sorted_by_duration_before(self, before_profile, after_profile_exact):
        diff = compute_diff(before_profile, after_profile_exact, plan=None)
        bottlenecks = diff.top_bottlenecks_before
        assert bottlenecks[0].operator_name == "aten::linear"  # 3 ms > 1 ms

    def test_top_n_respected(self, before_profile, after_profile_exact):
        diff = compute_diff(before_profile, after_profile_exact, plan=None, top_n=1)
        assert len(diff.top_bottlenecks_before) == 1

    def test_top_n_zero_returns_empty(self, before_profile, after_profile_exact):
        diff = compute_diff(before_profile, after_profile_exact, plan=None, top_n=0)
        assert diff.top_bottlenecks_before == []
