"""
Unit tests for operator_profiler.planner.memory (OptimizationMemory).

No GPU required.  File I/O uses tmp_path (pytest fixture).
"""
from __future__ import annotations

import json
from pathlib import Path

import pytest

from operator_profiler.planner.memory import OptimizationMemory, _worst_bottleneck, _jaccard
from operator_profiler.planner.schema import GraphPattern, MemoryEntry
from operator_profiler.rewriter.dsl import RewritePlan
from operator_profiler.schema.profile import (
    AggregatedMetrics,
    CaptureMetadata,
    OperatorAttributedProfile,
    OperatorRecord,
)


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------

def _make_profile(
    op_names: list[str],
    bottleneck: str = "memory_bound",
    duration_ns: int = 1000,
) -> OperatorAttributedProfile:
    operators = [
        OperatorRecord(
            operator_id=f"{name}_{i}",
            operator_name=name,
            call_index=i,
            aggregated=AggregatedMetrics(
                total_duration_ns=duration_ns,
                kernel_count=1,
                bottleneck_classification=bottleneck,
            ),
        )
        for i, name in enumerate(op_names)
    ]
    return OperatorAttributedProfile(
        capture_metadata=CaptureMetadata(
            model_name="TestModel",
            torch_version="2.3.0",
            capture_timestamp_utc="2026-03-21T00:00:00+00:00",
        ),
        operators=operators,
    )


def _make_entry(
    op_names: list[str],
    bottleneck: str = "memory_bound",
    speedup: float = 1.1,
) -> MemoryEntry:
    from operator_profiler.planner.memory import _make_pattern_hash
    pattern = GraphPattern(
        op_sequence=op_names,
        pattern_hash=_make_pattern_hash(op_names),
    )
    return MemoryEntry(
        entry_id="abc123",
        graph_pattern=pattern,
        bottleneck=bottleneck,
        rewrite_plan=RewritePlan(),
        speedup=speedup,
        created_at="2026-03-21T00:00:00+00:00",
    )


# ---------------------------------------------------------------------------
# _jaccard
# ---------------------------------------------------------------------------

def test_jaccard_identical():
    assert _jaccard(["a", "b", "c"], ["a", "b", "c"]) == 1.0


def test_jaccard_disjoint():
    assert _jaccard(["a", "b"], ["c", "d"]) == 0.0


def test_jaccard_partial():
    result = _jaccard(["a", "b", "c"], ["b", "c", "d"])
    # intersection={b,c}, union={a,b,c,d} → 2/4 = 0.5
    assert result == pytest.approx(0.5)


def test_jaccard_empty():
    assert _jaccard([], []) == 1.0


# ---------------------------------------------------------------------------
# extract_pattern
# ---------------------------------------------------------------------------

def test_extract_pattern_op_sequence():
    profile = _make_profile(["aten::conv2d", "aten::relu"])
    mem = OptimizationMemory.__new__(OptimizationMemory)
    from operator_profiler.planner.schema import OptMemoryStore
    mem._store = OptMemoryStore()
    mem._path = Path("does_not_exist.json")
    pattern = mem.extract_pattern(profile)
    assert pattern.op_sequence == ["aten::conv2d", "aten::relu"]
    assert len(pattern.pattern_hash) == 64  # SHA-256 hex


def test_extract_pattern_hash_order_independent():
    from operator_profiler.planner.memory import _make_pattern_hash
    h1 = _make_pattern_hash(["aten::conv2d", "aten::relu"])
    h2 = _make_pattern_hash(["aten::relu", "aten::conv2d"])
    assert h1 == h2  # sorted before hashing


# ---------------------------------------------------------------------------
# search
# ---------------------------------------------------------------------------

def test_search_empty_memory(tmp_path):
    mem = OptimizationMemory(tmp_path / "opt.json")
    pattern = GraphPattern(op_sequence=["aten::linear"], pattern_hash="aaa")
    results = mem.search(pattern, "memory_bound")
    assert results == []


def test_search_filters_by_bottleneck(tmp_path):
    mem = OptimizationMemory(tmp_path / "opt.json")
    mem._store.entries.append(_make_entry(["aten::conv2d"], bottleneck="compute_bound"))
    mem._store.entries.append(_make_entry(["aten::conv2d"], bottleneck="memory_bound"))

    pattern = GraphPattern(op_sequence=["aten::conv2d"], pattern_hash="x")
    results = mem.search(pattern, "memory_bound")
    assert len(results) == 1
    assert results[0].entry.bottleneck == "memory_bound"


def test_search_returns_top_k(tmp_path):
    mem = OptimizationMemory(tmp_path / "opt.json")
    for i in range(5):
        mem._store.entries.append(
            _make_entry(["aten::conv2d", "aten::relu"], bottleneck="memory_bound", speedup=1.0 + i * 0.1)
        )
    pattern = GraphPattern(op_sequence=["aten::conv2d", "aten::relu"], pattern_hash="x")
    results = mem.search(pattern, "memory_bound", top_k=3)
    assert len(results) == 3


def test_search_ranks_by_similarity(tmp_path):
    mem = OptimizationMemory(tmp_path / "opt.json")
    # exact match
    mem._store.entries.append(_make_entry(["aten::conv2d", "aten::relu"], bottleneck="memory_bound"))
    # partial match
    mem._store.entries.append(_make_entry(["aten::conv2d", "aten::sigmoid"], bottleneck="memory_bound"))

    pattern = GraphPattern(op_sequence=["aten::conv2d", "aten::relu"], pattern_hash="x")
    results = mem.search(pattern, "memory_bound", top_k=2)
    assert results[0].similarity > results[1].similarity
    assert results[0].entry.graph_pattern.op_sequence == ["aten::conv2d", "aten::relu"]


# ---------------------------------------------------------------------------
# curate
# ---------------------------------------------------------------------------

def test_curate_below_threshold_returns_none(tmp_path):
    mem = OptimizationMemory(tmp_path / "opt.json")
    profile = _make_profile(["aten::linear"])
    result = mem.curate(profile, RewritePlan(), speedup=1.02, speedup_threshold=1.05)
    assert result is None
    assert len(mem) == 0


def test_curate_above_threshold_saves_entry(tmp_path):
    path = tmp_path / "opt.json"
    mem = OptimizationMemory(path)
    profile = _make_profile(["aten::linear", "aten::relu"], bottleneck="latency_bound")
    entry = mem.curate(profile, RewritePlan(), speedup=1.15, speedup_threshold=1.05)
    assert entry is not None
    assert entry.speedup == pytest.approx(1.15)
    assert entry.bottleneck == "latency_bound"
    assert len(mem) == 1


def test_curate_persists_to_disk(tmp_path):
    path = tmp_path / "opt.json"
    mem = OptimizationMemory(path)
    profile = _make_profile(["aten::linear"])
    mem.curate(profile, RewritePlan(), speedup=1.2)

    # Re-load from disk
    mem2 = OptimizationMemory(path)
    assert len(mem2) == 1
    assert mem2.entries[0].speedup == pytest.approx(1.2)


def test_curate_atomic_write_no_corruption(tmp_path):
    """Verify no .tmp file is left behind after a successful save."""
    path = tmp_path / "opt.json"
    mem = OptimizationMemory(path)
    profile = _make_profile(["aten::add"])
    mem.curate(profile, RewritePlan(), speedup=1.1)
    assert not (tmp_path / "opt.tmp").exists()
    assert path.exists()


# ---------------------------------------------------------------------------
# _worst_bottleneck
# ---------------------------------------------------------------------------

def test_worst_bottleneck_picks_slowest_op():
    profile = _make_profile(["aten::conv2d"], bottleneck="memory_bound", duration_ns=500)
    # Add a slower compute-bound op
    from operator_profiler.schema.profile import AggregatedMetrics, OperatorRecord
    profile.operators.append(
        OperatorRecord(
            operator_id="aten::mm_0",
            operator_name="aten::mm",
            call_index=1,
            aggregated=AggregatedMetrics(
                total_duration_ns=9999,
                kernel_count=1,
                bottleneck_classification="compute_bound",
            ),
        )
    )
    assert _worst_bottleneck(profile) == "compute_bound"


def test_worst_bottleneck_no_aggregated():
    from operator_profiler.schema.profile import OperatorRecord
    profile = OperatorAttributedProfile(
        capture_metadata=CaptureMetadata(
            model_name="M",
            torch_version="2.3.0",
            capture_timestamp_utc="2026-03-21T00:00:00+00:00",
        ),
        operators=[OperatorRecord(operator_id="x", operator_name="aten::relu", call_index=0)],
    )
    assert _worst_bottleneck(profile) == "unknown"
