"""Unit tests for Markdown renderer (summarizer/markdown.py)."""
from __future__ import annotations

import pytest

from operator_profiler.summarizer.diff import compute_diff
from operator_profiler.summarizer.markdown import render_markdown
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
        best_plan_description="Fuse linear+relu",
    )


class TestRenderMarkdown:
    def test_contains_model_name(
        self, before_profile, after_profile_exact, loop_result_with_history, memory_store_with_entries
    ):
        report = _make_report(
            before_profile, after_profile_exact, loop_result_with_history, memory_store_with_entries
        )
        md = render_markdown(report)
        assert "TestModel" in md

    def test_contains_total_speedup(
        self, before_profile, after_profile_exact, loop_result_with_history, memory_store_with_entries
    ):
        report = _make_report(
            before_profile, after_profile_exact, loop_result_with_history, memory_store_with_entries
        )
        md = render_markdown(report)
        assert "2.00x" in md

    def test_contains_top_bottlenecks_header(
        self, before_profile, after_profile_exact, loop_result_with_history, memory_store_with_entries
    ):
        report = _make_report(
            before_profile, after_profile_exact, loop_result_with_history, memory_store_with_entries
        )
        md = render_markdown(report)
        assert "## Top Bottlenecks" in md

    def test_contains_lessons_learned(
        self, before_profile, after_profile_exact, loop_result_with_history, memory_store_with_entries
    ):
        report = _make_report(
            before_profile, after_profile_exact, loop_result_with_history, memory_store_with_entries
        )
        md = render_markdown(report)
        assert "## Lessons Learned" in md

    def test_contains_iteration_history(
        self, before_profile, after_profile_exact, loop_result_with_history, memory_store_with_entries
    ):
        report = _make_report(
            before_profile, after_profile_exact, loop_result_with_history, memory_store_with_entries
        )
        md = render_markdown(report)
        assert "## Optimization Loop History" in md

    def test_history_rows_match_iterations(
        self, before_profile, after_profile_exact, loop_result_with_history, memory_store_with_entries
    ):
        report = _make_report(
            before_profile, after_profile_exact, loop_result_with_history, memory_store_with_entries
        )
        md = render_markdown(report)
        # 2 history entries → 2 data rows (each | 0 | and | 1 | in the table)
        assert "| 0 |" in md
        assert "| 1 |" in md

    def test_returns_string(
        self, before_profile, after_profile_exact, loop_result_with_history, memory_store_with_entries
    ):
        report = _make_report(
            before_profile, after_profile_exact, loop_result_with_history, memory_store_with_entries
        )
        md = render_markdown(report)
        assert isinstance(md, str)
        assert len(md) > 0

    def test_no_rules_skips_lessons_section(
        self, before_profile, after_profile_exact, loop_result_with_history
    ):
        diff = compute_diff(before_profile, after_profile_exact, plan=None)
        report = SummaryReport(
            diff=diff,
            rules=[],
            lessons_learned=[],
            loop_history=[],
            best_speedup=2.0,
        )
        md = render_markdown(report)
        assert "## Lessons Learned" not in md
