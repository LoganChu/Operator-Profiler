"""
Planner θ_p data models — Pydantic v2 schemas for the Optimization Memory,
beam search state, and retrieval candidates.
"""
from __future__ import annotations

from typing import Annotated, Literal

from pydantic import BaseModel, Field

from operator_profiler.rewriter.dsl import RewritePlan


# ---------------------------------------------------------------------------
# Graph Pattern
# ---------------------------------------------------------------------------

class GraphPattern(BaseModel):
    """
    A fingerprint of the operator sequence seen in a profiled graph.

    ``pattern_hash`` is the SHA-256 of ``"|".join(sorted(op_sequence))``
    so that two graphs with the same multiset of ops hash identically
    regardless of call order.
    """
    op_sequence: list[str]                    # e.g. ["aten::conv2d", "aten::relu"]
    pattern_hash: str                         # SHA-256 hex digest
    input_shapes: dict[str, list[int]] = Field(default_factory=dict)


# ---------------------------------------------------------------------------
# Memory Entry — one successful (pattern, bottleneck, plan, speedup) tuple
# ---------------------------------------------------------------------------

class MemoryEntry(BaseModel):
    entry_id: str                             # uuid4 hex string
    graph_pattern: GraphPattern
    bottleneck: Literal[
        "compute_bound", "memory_bound", "latency_bound", "unknown"
    ]
    rewrite_plan: RewritePlan
    speedup: float                            # measured speedup vs baseline
    profile_id: str | None = None            # source profile schema_version tag
    model_name: str | None = None
    created_at: str                           # ISO 8601 UTC


# ---------------------------------------------------------------------------
# Search Candidate — a memory entry with its retrieval similarity score
# ---------------------------------------------------------------------------

class SearchCandidate(BaseModel):
    entry: MemoryEntry
    similarity: float = Field(ge=0.0, le=1.0)  # Jaccard similarity, 0..1


# ---------------------------------------------------------------------------
# Beam State — one active hypothesis in the beam search
# ---------------------------------------------------------------------------

class BeamState(BaseModel):
    plan: RewritePlan
    speedup: float = 1.0
    trial_count: int = 0
    strategy: Literal["explore", "refine"] = "explore"


# ---------------------------------------------------------------------------
# Top-level memory store — written/read as opt_memory.json
# ---------------------------------------------------------------------------

class OptMemoryStore(BaseModel):
    schema_version: str = "1.0"
    entries: Annotated[list[MemoryEntry], Field(default_factory=list)]
