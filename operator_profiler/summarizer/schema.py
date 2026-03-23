"""
Summarizer θ_s data models — Pydantic v2 schemas for diff, rules, and reports.
"""
from __future__ import annotations

from typing import Literal

from pydantic import BaseModel, Field

SUMMARIZER_VERSION = "1.0"


class OperatorDiff(BaseModel):
    """Before/after comparison for a single matched operator."""

    operator_id_before: str
    operator_id_after: str | None = None   # None when the op was fused away
    operator_name: str
    call_index: int
    duration_before_ns: int
    duration_after_ns: int | None = None   # None when fused away
    delta_duration_ns: int | None = None   # before - after (positive = improvement)
    speedup: float | None = None           # before / after; None if fused away
    bottleneck_before: str
    bottleneck_after: str | None = None
    bottleneck_changed: bool = False
    rewrite_ops_applied: list[str] = Field(default_factory=list)  # DSL op IDs
    fusion_partners: list[str] = Field(default_factory=list)
    match_type: Literal["exact", "fused_into", "new", "removed"] = "exact"


class ProfileDiff(BaseModel):
    """Top-level before/after diff for a complete optimization run."""

    schema_version: str = SUMMARIZER_VERSION
    model_name: str
    device_name: str | None = None
    total_duration_before_ns: int
    total_duration_after_ns: int
    total_speedup: float
    wall_time_saved_ns: int
    operator_diffs: list[OperatorDiff] = Field(default_factory=list)
    top_bottlenecks_before: list[OperatorDiff] = Field(default_factory=list)
    unmatched_before: list[str] = Field(default_factory=list)
    unmatched_after: list[str] = Field(default_factory=list)


class OptimizationRule(BaseModel):
    """
    A human-readable rule distilled from a MemoryEntry.

    Example rule_text:
    'When [aten::conv2d, aten::relu] is memory_bound, apply
     fuse(conv2d_0, relu_0, strategy=inductor_fuse) to achieve ~18.0% speedup'
    """

    entry_id: str
    op_pattern: list[str]
    bottleneck: str
    rewrite_op_summary: str
    speedup: float
    speedup_pct: float           # (speedup - 1) * 100
    conditions: list[str]        # human-readable threshold strings
    recommended_action: str
    example_model: str | None = None
    created_at: str
    rule_text: str               # full assembled sentence


class SummaryReport(BaseModel):
    """Top-level container for a full optimization summary."""

    schema_version: str = SUMMARIZER_VERSION
    diff: ProfileDiff
    rules: list[OptimizationRule] = Field(default_factory=list)
    lessons_learned: list[str] = Field(default_factory=list)  # bullet strings
    loop_history: list[dict] = Field(default_factory=list)    # verbatim from LoopResult
    best_speedup: float
    best_plan_description: str | None = None
