"""
Unit tests for ThetaPlanner — Anthropic client is fully mocked.

No GPU, no API key, no network required.
"""
from __future__ import annotations

from unittest.mock import MagicMock, patch

import pytest
import torch
import torch.nn as nn
import torch.fx

from operator_profiler.planner.planner import PlannerConfig, ThetaPlanner
from operator_profiler.planner.schema import SearchCandidate, MemoryEntry, GraphPattern
from operator_profiler.rewriter.dsl import FuseOp, RewritePlan
from operator_profiler.schema.profile import (
    AggregatedMetrics,
    CaptureMetadata,
    OperatorAttributedProfile,
    OperatorRecord,
)


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _make_profile() -> OperatorAttributedProfile:
    return OperatorAttributedProfile(
        capture_metadata=CaptureMetadata(
            model_name="M",
            torch_version="2.3.0",
            capture_timestamp_utc="2026-03-21T00:00:00+00:00",
        ),
        operators=[
            OperatorRecord(
                operator_id="aten::linear_0",
                operator_name="aten::linear",
                call_index=0,
                aggregated=AggregatedMetrics(
                    total_duration_ns=1000,
                    kernel_count=1,
                    bottleneck_classification="memory_bound",
                ),
            )
        ],
    )


def _make_gm() -> torch.fx.GraphModule:
    class M(nn.Module):
        def forward(self, x):
            return x + x
    return torch.fx.symbolic_trace(M())


def _make_tool_use_response(plan_dict: dict) -> MagicMock:
    """Build a mock Anthropic response containing a tool_use block."""
    block = MagicMock()
    block.type = "tool_use"
    block.name = "produce_rewrite_plan"
    block.input = plan_dict
    response = MagicMock()
    response.content = [block]
    return response


# ---------------------------------------------------------------------------
# Tests
# ---------------------------------------------------------------------------

@patch("operator_profiler.planner.planner.ThetaPlanner._build_client")
def test_plan_returns_rewrite_plan_on_success(mock_build_client):
    """plan() parses a well-formed tool_use response into a RewritePlan."""
    mock_client = MagicMock()
    mock_build_client.return_value = mock_client

    plan_dict = {
        "plan_version": "1.0",
        "source_profile_id": "1.0/aten::linear_0",
        "description": "Fuse linear ops.",
        "ops": [
            {
                "op": "fuse",
                "id": "f0",
                "nodes": ["add", "add_1"],
                "strategy": "inductor_fuse",
                "custom_op_name": None,
                "comment": None,
            }
        ],
    }
    mock_client.messages.create.return_value = _make_tool_use_response(plan_dict)

    planner = ThetaPlanner()
    gm = _make_gm()
    plan = planner.plan(gm, _make_profile(), [])

    assert isinstance(plan, RewritePlan)
    assert plan.plan_version == "1.0"
    assert len(plan.ops) == 1
    assert isinstance(plan.ops[0], FuseOp)


@patch("operator_profiler.planner.planner.ThetaPlanner._build_client")
def test_plan_returns_empty_plan_on_api_error(mock_build_client):
    """API errors are caught and an empty RewritePlan is returned."""
    mock_client = MagicMock()
    mock_build_client.return_value = mock_client
    mock_client.messages.create.side_effect = RuntimeError("network error")

    planner = ThetaPlanner()
    plan = planner.plan(_make_gm(), _make_profile(), [])
    assert isinstance(plan, RewritePlan)
    assert plan.ops == []


@patch("operator_profiler.planner.planner.ThetaPlanner._build_client")
def test_plan_returns_empty_plan_on_parse_error(mock_build_client):
    """Malformed tool input is caught and empty plan returned."""
    mock_client = MagicMock()
    mock_build_client.return_value = mock_client
    # bad plan_version
    bad_dict = {"plan_version": "99.0", "ops": [{"op": "unknown_op", "id": "x"}]}
    mock_client.messages.create.return_value = _make_tool_use_response(bad_dict)

    planner = ThetaPlanner()
    plan = planner.plan(_make_gm(), _make_profile(), [])
    assert isinstance(plan, RewritePlan)
    assert plan.ops == []


@patch("operator_profiler.planner.planner.ThetaPlanner._build_client")
def test_plan_returns_empty_plan_when_no_tool_use_block(mock_build_client):
    """Response without a tool_use block returns empty plan."""
    mock_client = MagicMock()
    mock_build_client.return_value = mock_client
    response = MagicMock()
    response.content = []
    mock_client.messages.create.return_value = response

    planner = ThetaPlanner()
    plan = planner.plan(_make_gm(), _make_profile(), [])
    assert plan.ops == []


@patch("operator_profiler.planner.planner.ThetaPlanner._build_client")
def test_plan_passes_refine_strategy_with_candidates(mock_build_client):
    """With strategy='refine', candidates are serialised into the user message."""
    mock_client = MagicMock()
    mock_build_client.return_value = mock_client
    mock_client.messages.create.return_value = _make_tool_use_response(
        {"plan_version": "1.0", "ops": []}
    )

    from operator_profiler.planner.memory import _make_pattern_hash
    entry = MemoryEntry(
        entry_id="abc",
        graph_pattern=GraphPattern(
            op_sequence=["aten::linear"],
            pattern_hash=_make_pattern_hash(["aten::linear"]),
        ),
        bottleneck="memory_bound",
        rewrite_plan=RewritePlan(),
        speedup=1.12,
        created_at="2026-03-21T00:00:00+00:00",
    )
    candidates = [SearchCandidate(entry=entry, similarity=0.9)]

    planner = ThetaPlanner()
    planner.plan(_make_gm(), _make_profile(), candidates, strategy="refine")

    call_kwargs = mock_client.messages.create.call_args[1]
    user_content = call_kwargs["messages"][0]["content"]
    assert "Retrieved Memory" in user_content
    assert "REFINE" in user_content


@patch("operator_profiler.planner.planner.ThetaPlanner._build_client")
def test_plan_explore_strategy_has_no_memory_context(mock_build_client):
    """With strategy='explore', the message says no memory context."""
    mock_client = MagicMock()
    mock_build_client.return_value = mock_client
    mock_client.messages.create.return_value = _make_tool_use_response(
        {"plan_version": "1.0", "ops": []}
    )

    planner = ThetaPlanner()
    planner.plan(_make_gm(), _make_profile(), [], strategy="explore")

    call_kwargs = mock_client.messages.create.call_args[1]
    user_content = call_kwargs["messages"][0]["content"]
    assert "EXPLORE" in user_content


@patch("operator_profiler.planner.planner.ThetaPlanner._build_client")
def test_planner_config_model_passed_to_api(mock_build_client):
    """PlannerConfig.model is forwarded to the Anthropic API call."""
    mock_client = MagicMock()
    mock_build_client.return_value = mock_client
    mock_client.messages.create.return_value = _make_tool_use_response(
        {"plan_version": "1.0", "ops": []}
    )

    cfg = PlannerConfig(model="claude-opus-4-6", max_tokens=512)
    planner = ThetaPlanner(cfg)
    planner.plan(_make_gm(), _make_profile(), [])

    call_kwargs = mock_client.messages.create.call_args[1]
    assert call_kwargs["model"] == "claude-opus-4-6"
    assert call_kwargs["max_tokens"] == 512


def test_planner_raises_on_missing_anthropic_package():
    """ImportError is raised with a helpful message when anthropic is not installed."""
    with patch.dict("sys.modules", {"anthropic": None}):
        with pytest.raises(ImportError, match="anthropic package is required"):
            ThetaPlanner()


# ---------------------------------------------------------------------------
# rank_candidates
# ---------------------------------------------------------------------------

def _make_candidates(entry_ids: list[str]) -> list[SearchCandidate]:
    """Build a list of SearchCandidates with the given entry IDs."""
    from operator_profiler.planner.memory import _make_pattern_hash
    result = []
    for eid in entry_ids:
        entry = MemoryEntry(
            entry_id=eid,
            graph_pattern=GraphPattern(
                op_sequence=["aten::linear"],
                pattern_hash=_make_pattern_hash(["aten::linear"]),
            ),
            bottleneck="memory_bound",
            rewrite_plan=RewritePlan(),
            speedup=1.2,
            created_at="2026-03-21T00:00:00+00:00",
        )
        result.append(SearchCandidate(entry=entry, similarity=0.8))
    return result


def _make_rank_response(ranked_ids: list[str]) -> MagicMock:
    block = MagicMock()
    block.type = "tool_use"
    block.name = "rank_memory_candidates"
    block.input = {"ranked_ids": ranked_ids, "reasoning": "top choice is most similar"}
    response = MagicMock()
    response.content = [block]
    return response


@patch("operator_profiler.planner.planner.ThetaPlanner._build_client")
def test_rank_candidates_reorders_by_llm_output(mock_build_client):
    """rank_candidates reorders candidates according to the LLM's ranked_ids."""
    mock_client = MagicMock()
    mock_build_client.return_value = mock_client
    mock_client.messages.create.return_value = _make_rank_response(["b", "a"])

    planner = ThetaPlanner()
    candidates = _make_candidates(["a", "b"])
    result = planner.rank_candidates(_make_profile(), candidates)

    assert [c.entry.entry_id for c in result] == ["b", "a"]


@patch("operator_profiler.planner.planner.ThetaPlanner._build_client")
def test_rank_candidates_single_entry_skips_api(mock_build_client):
    """With only 1 candidate, no API call is made and the list is returned as-is."""
    mock_client = MagicMock()
    mock_build_client.return_value = mock_client

    planner = ThetaPlanner()
    candidates = _make_candidates(["only"])
    result = planner.rank_candidates(_make_profile(), candidates)

    mock_client.messages.create.assert_not_called()
    assert result == candidates


@patch("operator_profiler.planner.planner.ThetaPlanner._build_client")
def test_rank_candidates_empty_list_skips_api(mock_build_client):
    """Empty candidate list is returned immediately without API call."""
    mock_client = MagicMock()
    mock_build_client.return_value = mock_client

    planner = ThetaPlanner()
    result = planner.rank_candidates(_make_profile(), [])

    mock_client.messages.create.assert_not_called()
    assert result == []


@patch("operator_profiler.planner.planner.ThetaPlanner._build_client")
def test_rank_candidates_api_failure_falls_back_to_original_order(mock_build_client):
    """On API error, rank_candidates returns the original order unchanged."""
    mock_client = MagicMock()
    mock_build_client.return_value = mock_client
    mock_client.messages.create.side_effect = RuntimeError("timeout")

    planner = ThetaPlanner()
    candidates = _make_candidates(["x", "y", "z"])
    result = planner.rank_candidates(_make_profile(), candidates)

    assert [c.entry.entry_id for c in result] == ["x", "y", "z"]


@patch("operator_profiler.planner.planner.ThetaPlanner._build_client")
def test_rank_candidates_partial_ids_appends_unmentioned(mock_build_client):
    """IDs not mentioned by the LLM are appended at the end."""
    mock_client = MagicMock()
    mock_build_client.return_value = mock_client
    # LLM only mentions "b" and "c", not "a"
    mock_client.messages.create.return_value = _make_rank_response(["b", "c"])

    planner = ThetaPlanner()
    candidates = _make_candidates(["a", "b", "c"])
    result = planner.rank_candidates(_make_profile(), candidates)

    ids = [c.entry.entry_id for c in result]
    assert ids[:2] == ["b", "c"]
    assert "a" in ids


@patch("operator_profiler.planner.planner.ThetaPlanner._build_client")
def test_rank_candidates_no_tool_use_block_falls_back(mock_build_client):
    """Response with no tool_use block returns original order."""
    mock_client = MagicMock()
    mock_build_client.return_value = mock_client
    response = MagicMock()
    response.content = []
    mock_client.messages.create.return_value = response

    planner = ThetaPlanner()
    candidates = _make_candidates(["p", "q"])
    result = planner.rank_candidates(_make_profile(), candidates)

    assert [c.entry.entry_id for c in result] == ["p", "q"]


@patch("operator_profiler.planner.planner.ThetaPlanner._build_client")
def test_rank_candidates_uses_rank_tool_not_rewrite_tool(mock_build_client):
    """rank_candidates must call rank_memory_candidates, not produce_rewrite_plan."""
    mock_client = MagicMock()
    mock_build_client.return_value = mock_client
    mock_client.messages.create.return_value = _make_rank_response(["a", "b"])

    planner = ThetaPlanner()
    planner.rank_candidates(_make_profile(), _make_candidates(["a", "b"]))

    call_kwargs = mock_client.messages.create.call_args[1]
    tool_names = [t["name"] for t in call_kwargs["tools"]]
    assert "rank_memory_candidates" in tool_names
    assert "produce_rewrite_plan" not in tool_names
    assert call_kwargs["tool_choice"]["name"] == "rank_memory_candidates"
