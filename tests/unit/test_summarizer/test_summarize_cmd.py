"""Integration tests for operator-profiler summarize CLI command."""
from __future__ import annotations

import json
from pathlib import Path
from types import SimpleNamespace

import pytest

from operator_profiler.planner.schema import OptMemoryStore
from operator_profiler.cli.summarize_cmd import _run


def _write_fixtures(tmp_path, before_profile, after_profile_exact, loop_result_with_history):
    before_path = tmp_path / "before.json"
    after_path = tmp_path / "after.json"
    loop_path = tmp_path / "loop.json"
    memory_path = tmp_path / "memory.json"

    before_path.write_text(before_profile.model_dump_json(), encoding="utf-8")
    after_path.write_text(after_profile_exact.model_dump_json(), encoding="utf-8")
    loop_path.write_text(json.dumps(loop_result_with_history.to_dict()), encoding="utf-8")
    memory_path.write_text(OptMemoryStore().model_dump_json(), encoding="utf-8")

    return before_path, after_path, loop_path, memory_path


class TestSummarizeCmdMarkdown:
    def test_output_file_written(
        self, tmp_path, before_profile, after_profile_exact, loop_result_with_history
    ):
        before_path, after_path, loop_path, memory_path = _write_fixtures(
            tmp_path, before_profile, after_profile_exact, loop_result_with_history
        )
        out = tmp_path / "report.md"
        args = SimpleNamespace(
            before=str(before_path),
            after=str(after_path),
            loop_result=str(loop_path),
            memory=str(memory_path),
            output=str(out),
            format="markdown",
            top_n=5,
        )
        _run(args)
        assert out.exists()
        content = out.read_text()
        assert "TestModel" in content

    def test_markdown_contains_top_bottlenecks(
        self, tmp_path, before_profile, after_profile_exact, loop_result_with_history
    ):
        before_path, after_path, loop_path, memory_path = _write_fixtures(
            tmp_path, before_profile, after_profile_exact, loop_result_with_history
        )
        out = tmp_path / "report.md"
        args = SimpleNamespace(
            before=str(before_path),
            after=str(after_path),
            loop_result=str(loop_path),
            memory=str(memory_path),
            output=str(out),
            format="markdown",
            top_n=5,
        )
        _run(args)
        content = out.read_text()
        assert "Top Bottlenecks" in content


class TestSummarizeCmdHtml:
    def test_html_output_file_created(
        self, tmp_path, before_profile, after_profile_exact, loop_result_with_history
    ):
        before_path, after_path, loop_path, memory_path = _write_fixtures(
            tmp_path, before_profile, after_profile_exact, loop_result_with_history
        )
        out = tmp_path / "report.html"
        args = SimpleNamespace(
            before=str(before_path),
            after=str(after_path),
            loop_result=str(loop_path),
            memory=str(memory_path),
            output=str(out),
            format="html",
            top_n=5,
        )
        _run(args)
        assert out.exists()
        content = out.read_text()
        assert "<!DOCTYPE html>" in content
        assert "TestModel" in content


class TestSummarizeCmdRich:
    def test_rich_format_does_not_raise(
        self, tmp_path, before_profile, after_profile_exact, loop_result_with_history, capsys
    ):
        """Rich format should not raise (falls back to markdown if rich absent)."""
        before_path, after_path, loop_path, memory_path = _write_fixtures(
            tmp_path, before_profile, after_profile_exact, loop_result_with_history
        )
        args = SimpleNamespace(
            before=str(before_path),
            after=str(after_path),
            loop_result=str(loop_path),
            memory=str(memory_path),
            output=None,
            format="rich",
            top_n=5,
        )
        _run(args)  # should not raise
