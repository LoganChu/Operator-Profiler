"""Unit tests for ProvenanceViewer (summarizer/provenance.py)."""
from __future__ import annotations

import pytest

from operator_profiler.rewriter.dsl import FuseOp, RewritePlan
from operator_profiler.schema.profile import (
    AggregatedMetrics,
    CaptureMetadata,
    KernelMetrics,
    KernelRecord,
    OperatorAttributedProfile,
    OperatorRecord,
)
from operator_profiler.summarizer.provenance import (
    build_provenance_rows,
    render_provenance_html,
    render_provenance_text,
)


def _kernel(kid, dur=1_000_000):
    return KernelRecord(
        kernel_id=kid,
        kernel_name=f"kern_{kid}",
        demangled_name=f"dem_{kid}",
        stream_id=0,
        device_id=0,
        start_ns=0,
        end_ns=dur,
        duration_ns=dur,
        metrics=KernelMetrics(dram_bytes_read=512, achieved_occupancy=0.6),
    )


def _no_kernel_profile():
    return OperatorAttributedProfile(
        capture_metadata=CaptureMetadata(
            model_name="M",
            torch_version="2.3.0",
            capture_timestamp_utc="2026-03-22T00:00:00+00:00",
        ),
        operators=[
            OperatorRecord(
                operator_id="aten::linear_0",
                operator_name="aten::linear",
                call_index=0,
                kernels=[],  # no kernels
            )
        ],
    )


class TestBuildProvenanceRows:
    def test_one_row_per_kernel(self, before_profile):
        rows = build_provenance_rows(before_profile, plan=None)
        # before_profile has 2 ops, each with 1 kernel
        assert len(rows) == 2

    def test_op_with_no_kernels_emits_placeholder(self):
        profile = _no_kernel_profile()
        rows = build_provenance_rows(profile, plan=None)
        assert len(rows) == 1
        assert rows[0].kernel_name == "(no kernels)"

    def test_rewrite_ops_attached_to_fused_nodes(self, before_profile, fuse_plan):
        rows = build_provenance_rows(before_profile, fuse_plan)
        linear_rows = [r for r in rows if r.operator_id == "aten::linear_0"]
        assert len(linear_rows) == 1
        assert "fuse_linear_relu" in linear_rows[0].rewrite_ops

    def test_rewrite_ops_empty_for_unrelated(self, before_profile, fuse_plan):
        # There is no unrelated op in before_profile — both are in the fuse plan
        rows = build_provenance_rows(before_profile, plan=None)
        for r in rows:
            assert r.rewrite_ops == []

    def test_demangled_name_preferred(self, before_profile):
        rows = build_provenance_rows(before_profile, plan=None)
        for r in rows:
            if r.kernel_name != "(no kernels)":
                assert r.kernel_name.startswith("demangled_")

    def test_is_fused_flag(self, after_profile_fused, fuse_plan):
        rows = build_provenance_rows(after_profile_fused, plan=fuse_plan)
        fused_rows = [r for r in rows if r.is_fused]
        assert len(fused_rows) >= 1


class TestRenderProvenanceText:
    def test_contains_column_headers(self, before_profile):
        rows = build_provenance_rows(before_profile, plan=None)
        text = render_provenance_text(rows)
        assert "PyTorch Op" in text
        assert "Inductor Kernel" in text
        assert "Nsight Metrics" in text
        assert "Optimization Applied" in text

    def test_contains_operator_id(self, before_profile):
        rows = build_provenance_rows(before_profile, plan=None)
        text = render_provenance_text(rows)
        assert "aten::linear_0" in text

    def test_empty_rows_returns_message(self):
        text = render_provenance_text([])
        assert "No provenance rows" in text


class TestRenderProvenanceHtml:
    def test_starts_with_doctype(self, before_profile):
        rows = build_provenance_rows(before_profile, plan=None)
        html = render_provenance_html(rows)
        assert html.strip().startswith("<!DOCTYPE html>")

    def test_self_contained_no_external_links(self, before_profile):
        rows = build_provenance_rows(before_profile, plan=None)
        html = render_provenance_html(rows)
        assert "href=" not in html
        assert "src=" not in html

    def test_contains_all_operator_names(self, before_profile):
        rows = build_provenance_rows(before_profile, plan=None)
        html = render_provenance_html(rows)
        assert "aten::linear_0" in html
        assert "aten::relu_0" in html

    def test_contains_inline_style(self, before_profile):
        rows = build_provenance_rows(before_profile, plan=None)
        html = render_provenance_html(rows)
        assert "<style>" in html
