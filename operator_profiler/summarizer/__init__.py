"""
Summarizer θ_s — post-optimization reporting and explanation stage.

Public API
----------
``compute_diff``       — before/after ProfileDiff computation
``entries_to_rules``   — OptimizationMemory → OptimizationRule list
``render_markdown``    — SummaryReport → Markdown string
``render_html``        — SummaryReport → self-contained HTML string
``RichDashboard``      — rich CLI summary dashboard
``LiveProgressDashboard`` — context manager for in-loop progress display
``explain_node``       — natural-language explanation for a specific node
``build_provenance_rows`` — join operator/kernel/metrics/plan into rows
"""
from operator_profiler.summarizer.schema import (
    OperatorDiff,
    OptimizationRule,
    ProfileDiff,
    SummaryReport,
)
from operator_profiler.summarizer.diff import compute_diff
from operator_profiler.summarizer.rules import entries_to_rules, entry_to_rule
from operator_profiler.summarizer.markdown import render_markdown
from operator_profiler.summarizer.html import render_html
from operator_profiler.summarizer.provenance import (
    ProvenanceRow,
    build_provenance_rows,
    render_provenance_text,
    render_provenance_rich,
    render_provenance_html,
)
from operator_profiler.summarizer.dashboard import RichDashboard, LiveProgressDashboard
from operator_profiler.summarizer.explain import explain_node

__all__ = [
    "OperatorDiff",
    "OptimizationRule",
    "ProfileDiff",
    "SummaryReport",
    "compute_diff",
    "entries_to_rules",
    "entry_to_rule",
    "render_markdown",
    "render_html",
    "ProvenanceRow",
    "build_provenance_rows",
    "render_provenance_text",
    "render_provenance_rich",
    "render_provenance_html",
    "RichDashboard",
    "LiveProgressDashboard",
    "explain_node",
]
