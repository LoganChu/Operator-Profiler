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
