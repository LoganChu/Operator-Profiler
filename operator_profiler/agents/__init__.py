"""
LLM-backed agent modules for the Operator Profiler pipeline.

Each agent replaces a heuristic or static template with LLM reasoning,
while falling back gracefully on API errors so the pipeline stays robust.

  DiagnosisAgent      — bottleneck classification over full KernelMetrics set
  VerifierAgent       — repair hints from VerificationResult.node_diffs
  RuleAgent           — causal explanations for OptimizationRule entries
  MemoryCuratorAgent  — deduplication and compaction of OptimizationMemory
"""
from operator_profiler.agents.diagnosis import DiagnosisAgent, DiagnosisResult, ModelStats
from operator_profiler.agents.verifier import VerifierAgent, RepairContext
from operator_profiler.agents.rule import RuleAgent
from operator_profiler.agents.curator import MemoryCuratorAgent, CurationResult

__all__ = [
    "DiagnosisAgent",
    "DiagnosisResult",
    "ModelStats",
    "VerifierAgent",
    "RepairContext",
    "RuleAgent",
    "MemoryCuratorAgent",
    "CurationResult",
]
