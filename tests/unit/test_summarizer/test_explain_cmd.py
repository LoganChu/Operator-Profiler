"""Integration tests for operator-profiler explain CLI command."""
from __future__ import annotations

import json
from pathlib import Path
from types import SimpleNamespace

import pytest

from operator_profiler.cli.explain_cmd import _run


def _write_fixtures(tmp_path, before_profile, after_profile_exact, loop_result_with_history):
    before_path = tmp_path / "before.json"
    after_path = tmp_path / "after.json"
    loop_path = tmp_path / "loop.json"

    before_path.write_text(before_profile.model_dump_json(), encoding="utf-8")
    after_path.write_text(after_profile_exact.model_dump_json(), encoding="utf-8")
    loop_path.write_text(json.dumps(loop_result_with_history.to_dict()), encoding="utf-8")
    return before_path, after_path, loop_path


class TestExplainCmd:
    def test_found_node_prints_explanation(
        self, tmp_path, before_profile, after_profile_exact, loop_result_with_history, capsys
    ):
        before_path, after_path, loop_path = _write_fixtures(
            tmp_path, before_profile, after_profile_exact, loop_result_with_history
        )
        args = SimpleNamespace(
            node="aten::linear_0",
            before=str(before_path),
            after=str(after_path),
            loop_result=str(loop_path),
        )
        _run(args)
        captured = capsys.readouterr().out
        assert "aten::linear" in captured
        assert "3.000 ms" in captured

    def test_double_underscore_node_id(
        self, tmp_path, before_profile, after_profile_exact, loop_result_with_history, capsys
    ):
        before_path, after_path, loop_path = _write_fixtures(
            tmp_path, before_profile, after_profile_exact, loop_result_with_history
        )
        args = SimpleNamespace(
            node="aten__linear_0",
            before=str(before_path),
            after=str(after_path),
            loop_result=str(loop_path),
        )
        _run(args)
        captured = capsys.readouterr().out
        assert "aten::linear" in captured

    def test_unknown_node_prints_error(
        self, tmp_path, before_profile, after_profile_exact, loop_result_with_history, capsys
    ):
        before_path, after_path, loop_path = _write_fixtures(
            tmp_path, before_profile, after_profile_exact, loop_result_with_history
        )
        args = SimpleNamespace(
            node="aten::nonexistent_0",
            before=str(before_path),
            after=str(after_path),
            loop_result=str(loop_path),
        )
        _run(args)
        captured = capsys.readouterr().out
        assert "not found" in captured.lower()
