"""
Unit tests for BeamSearch — UCB1 scoring, strategy partitioning, beam pruning.
"""
from __future__ import annotations

import math
import random

import pytest

from operator_profiler.planner.schema import BeamState
from operator_profiler.planner.search import BeamSearch
from operator_profiler.rewriter.dsl import RewritePlan


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _beam(speedup: float = 1.0, trial_count: int = 1) -> BeamState:
    return BeamState(plan=RewritePlan(), speedup=speedup, trial_count=trial_count)


# ---------------------------------------------------------------------------
# UCB1 scoring
# ---------------------------------------------------------------------------

def test_ucb1_score_higher_speedup_wins():
    search = BeamSearch(alpha=1.0)
    b1 = _beam(speedup=1.5, trial_count=3)
    b2 = _beam(speedup=1.2, trial_count=3)
    assert search.ucb1_score(b1, 10) > search.ucb1_score(b2, 10)


def test_ucb1_score_favours_under_sampled():
    """With equal speedup, the beam with fewer trials should score higher."""
    search = BeamSearch(alpha=1.0)
    b_few = _beam(speedup=1.1, trial_count=1)
    b_many = _beam(speedup=1.1, trial_count=50)
    assert search.ucb1_score(b_few, 100) > search.ucb1_score(b_many, 100)


def test_ucb1_score_zero_trials_no_crash():
    """trial_count=0 must not divide by zero."""
    search = BeamSearch()
    b = BeamState(plan=RewritePlan(), speedup=1.0, trial_count=0)
    score = search.ucb1_score(b, 5)
    assert math.isfinite(score)


# ---------------------------------------------------------------------------
# partition_strategies
# ---------------------------------------------------------------------------

def test_partition_sums_to_width():
    search = BeamSearch(width=3)
    for hits in [0, 1, 5]:
        for iteration in range(6):
            n_explore, n_refine = search.partition_strategies(hits, iteration)
            assert n_explore + n_refine == 3


def test_partition_all_explore_when_no_memory_hits():
    search = BeamSearch(width=4)
    n_explore, n_refine = search.partition_strategies(memory_hits=0, iteration=0)
    assert n_explore == 4
    assert n_refine == 0


def test_partition_explore_ratio_decays():
    search = BeamSearch(width=10, base_explore_ratio=0.7)
    n0, _ = search.partition_strategies(memory_hits=5, iteration=0)
    n5, _ = search.partition_strategies(memory_hits=5, iteration=5)
    # ratio at iter=5: max(0.7 - 0.35, 0.3) = 0.35 → ceil(10*0.35)=4
    # ratio at iter=0: max(0.7, 0.3) = 0.7 → ceil(10*0.7)=7
    assert n0 >= n5


def test_partition_explore_floor_at_30pct():
    search = BeamSearch(width=10, base_explore_ratio=0.7)
    # After many iterations the ratio should floor at 0.3
    n_explore, _ = search.partition_strategies(memory_hits=5, iteration=100)
    # floor 0.3 → ceil(10*0.3) = 3
    assert n_explore == 3


# ---------------------------------------------------------------------------
# select_strategy
# ---------------------------------------------------------------------------

def test_select_strategy_always_explore_when_no_hits():
    search = BeamSearch(seed=42)
    for _ in range(20):
        assert search.select_strategy(memory_hits=0, iteration=0) == "explore"


def test_select_strategy_returns_valid_values():
    search = BeamSearch(seed=0)
    for iteration in range(5):
        result = search.select_strategy(memory_hits=3, iteration=iteration)
        assert result in ("explore", "refine")


# ---------------------------------------------------------------------------
# top_beams
# ---------------------------------------------------------------------------

def test_top_beams_returns_at_most_width():
    search = BeamSearch(width=3)
    beams = [_beam(speedup=float(i)) for i in range(8)]
    top = search.top_beams(beams, total_trials=8)
    assert len(top) == 3


def test_top_beams_sorted_by_ucb1():
    search = BeamSearch(width=3, alpha=0.0)  # alpha=0 → pure speedup ranking
    beams = [_beam(speedup=1.5), _beam(speedup=1.1), _beam(speedup=1.3)]
    top = search.top_beams(beams, total_trials=3)
    assert top[0].speedup == pytest.approx(1.5)
    assert top[1].speedup == pytest.approx(1.3)


def test_top_beams_empty_input():
    search = BeamSearch(width=3)
    assert search.top_beams([], total_trials=0) == []


def test_top_beams_fewer_than_width():
    search = BeamSearch(width=5)
    beams = [_beam(speedup=1.2), _beam(speedup=1.1)]
    top = search.top_beams(beams, total_trials=2)
    assert len(top) == 2
