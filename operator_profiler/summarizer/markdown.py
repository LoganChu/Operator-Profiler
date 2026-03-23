"""
Markdown report renderer for θ_s.
"""
from __future__ import annotations

from datetime import datetime, timezone

from operator_profiler.summarizer.schema import (
    OptimizationRule,
    ProfileDiff,
    SummaryReport,
)


def render_markdown(report: SummaryReport) -> str:
    """Render a SummaryReport as a Markdown string."""
    sections = [
        _render_header(report.diff, report.best_speedup, report.best_plan_description),
        _render_top_bottlenecks_table(report.diff),
        _render_per_optimization_detail(report.diff),
        _render_iteration_history(report.loop_history),
        _render_lessons_learned(report.rules),
    ]
    return "\n\n".join(s for s in sections if s) + "\n"


# ---------------------------------------------------------------------------
# Section renderers
# ---------------------------------------------------------------------------

def _render_header(
    diff: ProfileDiff,
    best_speedup: float,
    best_plan_description: str | None,
) -> str:
    device = diff.device_name or "unknown"
    saved_ms = diff.wall_time_saved_ns / 1e6
    now = datetime.now(tz=timezone.utc).strftime("%Y-%m-%d %H:%M UTC")
    lines = [
        f"# Optimization Summary — {diff.model_name}",
        "",
        f"**Device**: {device}  ",
        f"**Total Speedup**: {diff.total_speedup:.2f}x  ",
        f"**Wall Time Saved**: {saved_ms:.2f} ms  ",
        f"**Best Speedup**: {best_speedup:.2f}x  ",
        f"**Date**: {now}  ",
    ]
    if best_plan_description:
        lines.append(f"**Best Plan**: {best_plan_description}  ")
    return "\n".join(lines)


def _render_top_bottlenecks_table(diff: ProfileDiff, top_n: int = 5) -> str:
    candidates = diff.top_bottlenecks_before[:top_n]
    if not candidates:
        return ""

    lines = [
        "## Top Bottlenecks",
        "",
        "| Rank | Operator | Before (ms) | After (ms) | Speedup | Δ% | Bottleneck Before → After |",
        "|------|----------|-------------|------------|---------|-----|--------------------------|",
    ]
    for i, d in enumerate(candidates, 1):
        before_ms = f"{d.duration_before_ns / 1e6:.3f}"
        after_ms = f"{d.duration_after_ns / 1e6:.3f}" if d.duration_after_ns is not None else "—"
        speedup = f"{d.speedup:.2f}x" if d.speedup is not None else "—"
        if d.delta_duration_ns is not None and d.duration_before_ns > 0:
            delta_pct = f"{d.delta_duration_ns / d.duration_before_ns * 100:.1f}%"
        else:
            delta_pct = "—"
        bt_after = d.bottleneck_after or "—"
        bt_str = f"{d.bottleneck_before} → {bt_after}"
        lines.append(
            f"| {i} | `{d.operator_name}` | {before_ms} | {after_ms} | {speedup} | {delta_pct} | {bt_str} |"
        )
    return "\n".join(lines)


def _render_per_optimization_detail(diff: ProfileDiff) -> str:
    changed = [
        d for d in diff.operator_diffs
        if d.match_type in ("exact", "fused_into") and (
            d.rewrite_ops_applied or d.fusion_partners or d.speedup not in (None, 1.0)
        )
    ]
    if not changed:
        return ""

    lines = ["## Optimization Detail", ""]
    for d in changed:
        before_ms = d.duration_before_ns / 1e6
        after_ms = d.duration_after_ns / 1e6 if d.duration_after_ns is not None else None
        speedup_str = f"{d.speedup:.2f}x" if d.speedup is not None else "—"
        bt_after = d.bottleneck_after or "—"

        lines.append(f"### `{d.operator_name}` (call {d.call_index})")
        lines.append("")
        lines.append(f"- **Before**: {before_ms:.3f} ms ({d.bottleneck_before})")
        if after_ms is not None:
            lines.append(f"- **After**: {after_ms:.3f} ms ({bt_after})")
        lines.append(f"- **Speedup**: {speedup_str}")
        if d.rewrite_ops_applied:
            lines.append(f"- **Applied**: {', '.join(d.rewrite_ops_applied)}")
        if d.fusion_partners:
            lines.append(f"- **Fused with**: {', '.join(d.fusion_partners)}")
        lines.append("")
    return "\n".join(lines).rstrip()


def _render_iteration_history(history: list[dict]) -> str:
    if not history:
        return ""
    lines = [
        "## Optimization Loop History",
        "",
        "| Iteration | Bottleneck | Memory Hits | Plans Tried | Best Speedup So Far |",
        "|-----------|------------|-------------|-------------|---------------------|",
    ]
    for h in history:
        it = h.get("iteration", "?")
        bottleneck = h.get("bottleneck", "?")
        mem_hits = h.get("memory_hits", "?")
        plans = h.get("plans_tried", "?")
        best = h.get("best_speedup_so_far", "?")
        best_str = f"{best:.3f}x" if isinstance(best, float) else str(best)
        lines.append(f"| {it} | {bottleneck} | {mem_hits} | {plans} | {best_str} |")
    return "\n".join(lines)


def _render_lessons_learned(rules: list[OptimizationRule]) -> str:
    if not rules:
        return ""
    lines = ["## Lessons Learned", ""]
    for rule in sorted(rules, key=lambda r: r.speedup, reverse=True):
        lines.append(f"> **Rule**: {rule.rule_text}  ")
        if rule.conditions:
            lines.append(f"> *Conditions*: {'; '.join(rule.conditions)}  ")
        if rule.example_model:
            lines.append(f"> *Example model*: {rule.example_model}  ")
        lines.append(">")
        lines.append("")
    return "\n".join(lines).rstrip()
