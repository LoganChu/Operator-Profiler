"""
BeamSearch — UCB1-based explore/refine strategy for the optimization loop.

The beam maintains up to ``width`` candidate ``BeamState`` objects.
At each iteration:

1. ``partition_strategies()`` decides how many new plans to generate
   via exploration (no memory context) vs. refinement (memory context).
   The explore ratio starts at ``base_explore_ratio`` (default 0.7) and
   decays by 7 percentage points per iteration, flooring at 0.3.

2. After profiling each new plan, ``top_beams()`` retains the top
   ``width`` candidates by UCB1 score:

       score = speedup + alpha * sqrt(log(total_trials + 1) / (trial_count + 1))

   This balances exploitation (high speedup) with exploration (under-
   sampled candidates).
"""
from __future__ import annotations

import math
import random
from dataclasses import dataclass, field
from math import ceil
from typing import Literal

from operator_profiler.planner.schema import BeamState


@dataclass
class BeamSearch:
    """
    Parameters
    ----------
    width:
        Maximum number of active beams (candidates) maintained across
        iterations.
    base_explore_ratio:
        Fraction of new plans generated via exploration in iteration 0.
        Decays by 0.07 per iteration; floor at 0.3.
    alpha:
        UCB1 exploration constant.  Larger values favour under-sampled
        candidates; smaller values favour high-speedup beams.
    seed:
        Optional random seed for deterministic behaviour in tests.
    """
    width: int = 3
    base_explore_ratio: float = 0.7
    alpha: float = 1.0
    seed: int | None = None

    def __post_init__(self) -> None:
        if self.seed is not None:
            random.seed(self.seed)

    # ------------------------------------------------------------------
    # UCB1 scoring
    # ------------------------------------------------------------------

    def ucb1_score(self, beam: BeamState, total_trials: int) -> float:
        """
        UCB1 score for a single beam.

        Higher speedup and lower trial_count both increase the score.
        """
        return beam.speedup + self.alpha * math.sqrt(
            math.log(total_trials + 1) / (beam.trial_count + 1)
        )

    # ------------------------------------------------------------------
    # Strategy selection
    # ------------------------------------------------------------------

    def _effective_explore_ratio(self, iteration: int) -> float:
        return max(self.base_explore_ratio - 0.07 * iteration, 0.3)

    def select_strategy(
        self, memory_hits: int, iteration: int
    ) -> Literal["explore", "refine"]:
        """
        Randomly select a strategy for a single plan, given the number of
        memory hits and the current iteration index.

        When ``memory_hits == 0`` exploration is always chosen.
        """
        if memory_hits == 0:
            return "explore"
        ratio = self._effective_explore_ratio(iteration)
        return "explore" if random.random() < ratio else "refine"

    def partition_strategies(
        self, memory_hits: int, iteration: int
    ) -> tuple[int, int]:
        """
        Return ``(n_explore, n_refine)`` that sums to ``self.width``.

        When ``memory_hits == 0`` all plans are exploratory.
        """
        if memory_hits == 0:
            return self.width, 0
        ratio = self._effective_explore_ratio(iteration)
        n_explore = ceil(self.width * ratio)
        n_refine = self.width - n_explore
        return n_explore, n_refine

    # ------------------------------------------------------------------
    # Beam pruning
    # ------------------------------------------------------------------

    def top_beams(
        self, beams: list[BeamState], total_trials: int
    ) -> list[BeamState]:
        """
        Return the top ``width`` beams sorted by UCB1 score (descending).
        """
        return sorted(
            beams,
            key=lambda b: self.ucb1_score(b, total_trials),
            reverse=True,
        )[: self.width]
