"""
operator_profiler.rewriter — Hybrid Executor θ_e

Receives a structured JSON Rewrite Plan, applies it deterministically to a
``torch.fx.GraphModule``, and verifies numerical equivalence after each step.

Public API
----------
>>> from operator_profiler.rewriter import (
...     HybridExecutor, ExecutorConfig,
...     RewritePlan, RewriteValidationError,
...     VerificationGate, VerificationResult,
...     ProvenanceTracker,
...     lower_to_inductor, LoweringResult,
... )
"""
from operator_profiler.rewriter.dsl import (
    DSL_VERSION,
    AnyRewriteOp,
    FuseOp,
    ReorderOp,
    ChangeLayoutOp,
    BufferSharingOp,
    RewritePlan,
    RewriteValidationError,
)
from operator_profiler.rewriter.executor import ExecutorConfig, HybridExecutor, PreFlightError
from operator_profiler.rewriter.verification import (
    NodeDiff,
    VerificationGate,
    VerificationResult,
)
from operator_profiler.rewriter.provenance import (
    NodeProvenance,
    ProvenanceTracker,
    export_provenance_jsonl,
)
from operator_profiler.rewriter.lowering import LoweringResult, lower_to_inductor

__all__ = [
    # DSL
    "DSL_VERSION",
    "AnyRewriteOp",
    "FuseOp",
    "ReorderOp",
    "ChangeLayoutOp",
    "BufferSharingOp",
    "RewritePlan",
    "RewriteValidationError",
    # Executor
    "ExecutorConfig",
    "HybridExecutor",
    "PreFlightError",
    # Verification
    "NodeDiff",
    "VerificationGate",
    "VerificationResult",
    # Provenance
    "NodeProvenance",
    "ProvenanceTracker",
    "export_provenance_jsonl",
    # Lowering
    "LoweringResult",
    "lower_to_inductor",
]
