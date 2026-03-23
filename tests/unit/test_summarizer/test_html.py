"""Unit tests for HTML renderer (summarizer/html.py)."""
from __future__ import annotations

from operator_profiler.summarizer.diff import compute_diff
from operator_profiler.summarizer.html import render_html
from operator_profiler.summarizer.provenance import build_provenance_rows
from operator_profiler.summarizer.rules import entries_to_rules
from operator_profiler.summarizer.schema import SummaryReport


def _make_report(before_profile, after_profile_exact, loop_result_with_history, memory_store_with_entries):
    diff = compute_diff(before_profile, after_profile_exact, plan=None)
    rules = entries_to_rules(memory_store_with_entries.entries)
    return SummaryReport(
        diff=diff,
        rules=rules,
        lessons_learned=[r.rule_text for r in rules],
        loop_history=loop_result_with_history.history,
        best_speedup=loop_result_with_history.best_speedup,
    )


class TestRenderHtml:
    def test_starts_with_doctype(
        self, before_profile, after_profile_exact, loop_result_with_history, memory_store_with_entries
    ):
        report = _make_report(before_profile, after_profile_exact, loop_result_with_history, memory_store_with_entries)
        html = render_html(report)
        assert html.strip().startswith("<!DOCTYPE html>")

    def test_contains_model_name(
        self, before_profile, after_profile_exact, loop_result_with_history, memory_store_with_entries
    ):
        report = _make_report(before_profile, after_profile_exact, loop_result_with_history, memory_store_with_entries)
        html = render_html(report)
        assert "TestModel" in html

    def test_contains_inline_css(
        self, before_profile, after_profile_exact, loop_result_with_history, memory_store_with_entries
    ):
        report = _make_report(before_profile, after_profile_exact, loop_result_with_history, memory_store_with_entries)
        html = render_html(report)
        assert "<style>" in html

    def test_no_external_stylesheet_links(
        self, before_profile, after_profile_exact, loop_result_with_history, memory_store_with_entries
    ):
        report = _make_report(before_profile, after_profile_exact, loop_result_with_history, memory_store_with_entries)
        html = render_html(report)
        assert 'rel="stylesheet"' not in html

    def test_contains_provenance_rows(
        self, before_profile, after_profile_exact, loop_result_with_history, memory_store_with_entries
    ):
        report = _make_report(before_profile, after_profile_exact, loop_result_with_history, memory_store_with_entries)
        rows = build_provenance_rows(before_profile, plan=None)
        html = render_html(report, provenance_rows=rows)
        assert "aten::linear_0" in html

    def test_valid_html_structure(
        self, before_profile, after_profile_exact, loop_result_with_history, memory_store_with_entries
    ):
        report = _make_report(before_profile, after_profile_exact, loop_result_with_history, memory_store_with_entries)
        html = render_html(report)
        assert "<html" in html
        assert "</html>" in html
        assert "<body>" in html
        assert "</body>" in html

    def test_lessons_section_present(
        self, before_profile, after_profile_exact, loop_result_with_history, memory_store_with_entries
    ):
        report = _make_report(before_profile, after_profile_exact, loop_result_with_history, memory_store_with_entries)
        html = render_html(report)
        assert "Lessons Learned" in html
