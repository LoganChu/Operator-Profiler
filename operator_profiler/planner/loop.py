"""
OptimizationLoop — the outer N-iteration beam search loop for θ_p.

Each iteration:
1.  Extract the graph pattern from the current profile.
2.  Identify the worst bottleneck operator.
3.  Query ``OptimizationMemory`` for similar successful rewrites.
4.  Use ``BeamSearch.partition_strategies()`` to decide how many plans to
    generate via exploration vs. refinement.
5.  For each candidate plan:
    a. Call ``ThetaPlanner.plan()`` (LLM API call).
    b. Apply via ``HybridExecutor``.
    c. If verification passes, lower to Inductor and measure with
       ``profiler_fn``.
    d. Curate memory if speedup > threshold.
    e. Track the best plan seen.
6.  Prune beams with ``BeamSearch.top_beams()``.
7.  Advance the baseline to the best beam (progressive baseline).

``profiler_fn`` is a ``Callable[[torch.fx.GraphModule], OperatorAttributedProfile]``
injected at construction time, making the loop fully testable without GPU
hardware (mock it in unit tests).
"""
from __future__ import annotations

import logging
from dataclasses import dataclass, field
from typing import Callable, TYPE_CHECKING

from operator_profiler.rewriter.dsl import RewritePlan
from operator_profiler.rewriter.executor import ExecutorConfig, HybridExecutor
from operator_profiler.planner.memory import OptimizationMemory, _worst_bottleneck
from operator_profiler.planner.planner import ThetaPlanner
from operator_profiler.planner.schema import BeamState
from operator_profiler.planner.search import BeamSearch

if TYPE_CHECKING:
    import torch.fx
    from operator_profiler.agents.verifier import VerifierAgent
    from operator_profiler.schema.profile import OperatorAttributedProfile

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Config & Result
# ---------------------------------------------------------------------------

@dataclass
class LoopConfig:
    n_iterations: int = 5
    beam_width: int = 3
    speedup_threshold: float = 1.05
    executor_config: ExecutorConfig = field(default_factory=ExecutorConfig)


@dataclass
class LoopResult:
    best_plan: RewritePlan | None
    best_speedup: float
    history: list[dict]    # per-iteration summary dicts

    def to_dict(self) -> dict:
        """Serialize to a JSON-safe dict."""
        return {
            "best_plan": self.best_plan.model_dump() if self.best_plan else None,
            "best_speedup": self.best_speedup,
            "history": self.history,
        }

    @classmethod
    def from_dict(cls, data: dict) -> "LoopResult":
        """Deserialize from a dict produced by :meth:`to_dict`."""
        return cls(
            best_plan=(
                RewritePlan.model_validate(data["best_plan"])
                if data.get("best_plan") is not None
                else None
            ),
            best_speedup=data["best_speedup"],
            history=data["history"],
        )


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _total_duration(profile: "OperatorAttributedProfile") -> int:
    """Sum of all operator ``total_duration_ns`` values."""
    total = 0
    for op in profile.operators:
        if op.aggregated is not None:
            total += op.aggregated.total_duration_ns
    return total


def _worst_operator_id(profile: "OperatorAttributedProfile") -> str:
    worst_op = None
    worst_duration = -1
    for op in profile.operators:
        if op.aggregated is not None and op.aggregated.total_duration_ns > worst_duration:
            worst_duration = op.aggregated.total_duration_ns
            worst_op = op
    return worst_op.operator_id if worst_op is not None else "unknown"


# ---------------------------------------------------------------------------
# OptimizationLoop
# ---------------------------------------------------------------------------

class OptimizationLoop:
    """
    Parameters
    ----------
    planner:
        ``ThetaPlanner`` instance (or a mock for testing).
    memory:
        ``OptimizationMemory`` instance; may be pre-loaded with prior entries.
    search:
        ``BeamSearch`` instance controlling width and explore/refine ratios.
    profiler_fn:
        Callable that takes a compiled ``torch.fx.GraphModule`` and returns
        a fresh ``OperatorAttributedProfile``.  In tests, mock this to return
        a pre-built fixture profile without running actual profiling.
    config:
        Loop hyperparameters.
    """

    def __init__(
        self,
        planner: ThetaPlanner,
        memory: OptimizationMemory,
        search: BeamSearch,
        profiler_fn: "Callable[[torch.fx.GraphModule], OperatorAttributedProfile]",
        config: LoopConfig | None = None,
        verifier_agent: "VerifierAgent | None" = None,
    ) -> None:
        self._planner = planner
        self._memory = memory
        self._search = search
        self._profiler_fn = profiler_fn
        self._config = config or LoopConfig()
        self._verifier_agent = verifier_agent

    def run(
        self,
        gm: "torch.fx.GraphModule",
        initial_profile: "OperatorAttributedProfile",
        example_inputs: list,
    ) -> LoopResult:
        """
        Run the optimization loop.

        Parameters
        ----------
        gm:
            Original ``torch.fx.GraphModule`` (never mutated).
        initial_profile:
            Baseline ``OperatorAttributedProfile`` from the original model.
        example_inputs:
            Representative input tensors for lowering / warm-up.

        Returns
        -------
        LoopResult
            Best plan found, its speedup over the original baseline, and a
            per-iteration history list.
        """
        cfg = self._config
        baseline_duration = _total_duration(initial_profile)
        current_profile = initial_profile
        best_plan: RewritePlan | None = None
        best_speedup = 1.0
        beams: list[BeamState] = []
        total_trials = 0
        history: list[dict] = []

        for iteration in range(cfg.n_iterations):
            logger.info("Optimization loop iteration %d/%d", iteration + 1, cfg.n_iterations)

            pattern = self._memory.extract_pattern(current_profile)
            bottleneck = _worst_bottleneck(current_profile)
            worst_op_id = _worst_operator_id(current_profile)
            candidates = self._memory.broad_search(pattern, top_k=15)
            candidates = self._planner.rank_candidates(current_profile, candidates)

            n_explore, n_refine = self._search.partition_strategies(
                len(candidates), iteration
            )
            logger.debug(
                "  bottleneck=%s, memory_hits=%d, n_explore=%d, n_refine=%d",
                bottleneck, len(candidates), n_explore, n_refine,
            )

            new_beams: list[BeamState] = []
            iter_plans_tried = 0

            for strategy, n in [("explore", n_explore), ("refine", n_refine)]:
                ctx_candidates = candidates if strategy == "refine" else []
                for _ in range(n):
                    plan = self._planner.plan(
                        gm, current_profile, ctx_candidates, strategy=strategy
                    )

                    # Apply & verify — with one VerifierAgent-guided repair retry
                    try:
                        executor = HybridExecutor(gm, plan, cfg.executor_config)
                        result_gm, ver_results = executor.run()
                    except Exception as exc:
                        logger.debug("Executor raised %s — skipping plan", exc)
                        iter_plans_tried += 1
                        total_trials += 1
                        continue

                    if not all(v.passed for v in ver_results):
                        if self._verifier_agent is not None:
                            repair_ctx = self._verifier_agent.diagnose(plan, ver_results)
                            logger.debug(
                                "VerifierAgent: %s — retrying with repair context",
                                repair_ctx.failure_category,
                            )
                            plan = self._planner.plan(
                                gm,
                                current_profile,
                                ctx_candidates,
                                strategy=strategy,
                                repair_context=repair_ctx.to_prompt_section(),
                            )
                            try:
                                executor = HybridExecutor(gm, plan, cfg.executor_config)
                                result_gm, ver_results = executor.run()
                            except Exception as exc:
                                logger.debug(
                                    "Repair retry executor raised %s — skipping plan", exc
                                )
                                iter_plans_tried += 1
                                total_trials += 1
                                continue

                        if not all(v.passed for v in ver_results):
                            logger.debug("Verification failed for plan — skipping")
                            iter_plans_tried += 1
                            total_trials += 1
                            continue

                    # Profile the rewritten model
                    try:
                        new_profile = self._profiler_fn(result_gm)
                    except Exception as exc:
                        logger.debug("profiler_fn raised %s — skipping plan", exc)
                        iter_plans_tried += 1
                        total_trials += 1
                        continue

                    new_duration = _total_duration(new_profile)
                    speedup = (
                        baseline_duration / new_duration
                        if new_duration > 0
                        else 1.0
                    )
                    total_trials += 1
                    iter_plans_tried += 1

                    logger.info(
                        "  [%s] speedup=%.3fx (baseline=%dns, new=%dns)",
                        strategy, speedup, baseline_duration, new_duration,
                    )

                    # Curate memory
                    self._memory.curate(
                        new_profile, plan, speedup, cfg.speedup_threshold
                    )

                    beam = BeamState(
                        plan=plan,
                        speedup=speedup,
                        trial_count=1,
                        strategy=strategy,
                    )
                    new_beams.append(beam)

                    if speedup > best_speedup:
                        best_speedup = speedup
                        best_plan = plan

            beams = self._search.top_beams(beams + new_beams, total_trials)

            # Progressive baseline: if the best beam improved, advance
            if beams and beams[0].speedup > 1.0:
                try:
                    best_gm, _ = HybridExecutor(
                        gm, beams[0].plan, cfg.executor_config
                    ).run()
                    current_profile = self._profiler_fn(best_gm)
                    baseline_duration = _total_duration(current_profile)
                    logger.info(
                        "  Advanced baseline to %.3fx (new baseline=%dns)",
                        beams[0].speedup, baseline_duration,
                    )
                except Exception as exc:
                    logger.debug("Could not advance baseline: %s", exc)

            history.append(
                {
                    "iteration": iteration,
                    "bottleneck": bottleneck,
                    "worst_op_id": worst_op_id,
                    "memory_hits": len(candidates),
                    "plans_tried": iter_plans_tried,
                    "best_speedup_so_far": best_speedup,
                    "beam_scores": [b.speedup for b in beams],
                }
            )

        logger.info(
            "Optimization loop complete: best_speedup=%.3fx over %d total trials",
            best_speedup, total_trials,
        )
        return LoopResult(
            best_plan=best_plan,
            best_speedup=best_speedup,
            history=history,
        )
