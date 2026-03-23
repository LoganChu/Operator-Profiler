"""Unit tests for RichDashboard and LiveProgressDashboard (summarizer/dashboard.py)."""
from __future__ import annotations

import sys
from unittest.mock import MagicMock, patch

import pytest

from operator_profiler.summarizer.diff import compute_diff
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


class TestRichDashboard:
    def test_render_does_not_raise(
        self, before_profile, after_profile_exact, loop_result_with_history, memory_store_with_entries
    ):
        report = _make_report(before_profile, after_profile_exact, loop_result_with_history, memory_store_with_entries)
        from operator_profiler.summarizer.dashboard import RichDashboard
        # Use a mock console to avoid stdout side-effects
        mock_console = MagicMock()
        dash = RichDashboard(report, console=mock_console)
        dash.render()  # should not raise

    def test_render_calls_console_print_when_rich_available(
        self, before_profile, after_profile_exact, loop_result_with_history, memory_store_with_entries
    ):
        from operator_profiler.summarizer import dashboard
        if not dashboard._RICH_AVAILABLE:
            pytest.skip("rich not installed")
        report = _make_report(before_profile, after_profile_exact, loop_result_with_history, memory_store_with_entries)
        mock_console = MagicMock()
        from operator_profiler.summarizer.dashboard import RichDashboard
        dash = RichDashboard(report, console=mock_console)
        dash.render()
        assert mock_console.print.called

    def test_render_plain_fallback_produces_markdown(
        self, before_profile, after_profile_exact, loop_result_with_history, memory_store_with_entries, capsys
    ):
        """When rich is absent, render() should produce text (via render_markdown)."""
        from operator_profiler.summarizer import dashboard as dash_mod
        orig = dash_mod._RICH_AVAILABLE
        try:
            dash_mod._RICH_AVAILABLE = False
            report = _make_report(before_profile, after_profile_exact, loop_result_with_history, memory_store_with_entries)
            from operator_profiler.summarizer.dashboard import RichDashboard
            dash = RichDashboard(report)
            dash.render()
            captured = capsys.readouterr().out
            assert "TestModel" in captured
        finally:
            dash_mod._RICH_AVAILABLE = orig


class TestLiveProgressDashboard:
    def test_update_does_not_raise_when_rich_present(self):
        from operator_profiler.summarizer import dashboard as dash_mod
        if not dash_mod._RICH_AVAILABLE:
            pytest.skip("rich not installed")
        mock_console = MagicMock()
        from operator_profiler.summarizer.dashboard import LiveProgressDashboard
        with LiveProgressDashboard(n_iterations=3, console=mock_console) as live:
            live.update(0, "memory_bound", 3, 1.5)
            live.update(1, "compute_bound", 2, 1.8)

    def test_update_fallback_logs_when_rich_absent(self, caplog):
        from operator_profiler.summarizer import dashboard as dash_mod
        orig = dash_mod._RICH_AVAILABLE
        try:
            dash_mod._RICH_AVAILABLE = False
            from operator_profiler.summarizer.dashboard import LiveProgressDashboard
            import logging
            with caplog.at_level(logging.INFO, logger="operator_profiler.summarizer.dashboard"):
                with LiveProgressDashboard(n_iterations=2) as live:
                    live.update(0, "latency_bound", 1, 1.1)
            assert any("latency_bound" in r.message for r in caplog.records)
        finally:
            dash_mod._RICH_AVAILABLE = orig

    def test_context_manager_exits_cleanly(self):
        from operator_profiler.summarizer.dashboard import LiveProgressDashboard
        with LiveProgressDashboard(n_iterations=1) as live:
            live.update(0, "unknown", 0, 1.0)
        # No exception = success
