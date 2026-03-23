"""
Self-contained HTML report renderer. No external dependencies.
Produces a single .html file with all styles inlined.
"""
from __future__ import annotations

from operator_profiler.summarizer.schema import (
    OptimizationRule,
    ProfileDiff,
    SummaryReport,
)

_CSS = """
body { font-family: Arial, sans-serif; font-size: 14px; margin: 32px; color: #333; }
h1 { color: #1a202c; margin-bottom: 4px; }
h2 { color: #2d3748; border-bottom: 2px solid #e2e8f0; padding-bottom: 4px; margin-top: 32px; }
h3 { color: #4a5568; }
.meta { color: #718096; margin-bottom: 24px; }
.meta span { margin-right: 24px; }
.speedup-high { color: #276749; font-weight: bold; }
.speedup-med  { color: #975a16; font-weight: bold; }
.speedup-low  { color: #c53030; font-weight: bold; }
.bt-compute  { background: #bee3f8; color: #2a4365; padding: 2px 6px; border-radius: 3px; }
.bt-memory   { background: #fed7d7; color: #742a2a; padding: 2px 6px; border-radius: 3px; }
.bt-latency  { background: #fefcbf; color: #744210; padding: 2px 6px; border-radius: 3px; }
.bt-unknown  { background: #e2e8f0; color: #4a5568; padding: 2px 6px; border-radius: 3px; }
table { border-collapse: collapse; width: 100%; margin-bottom: 24px; }
th { background: #2d3748; color: #fff; padding: 10px 14px; text-align: left; }
td { padding: 8px 14px; border-bottom: 1px solid #e2e8f0; vertical-align: top; }
tr:nth-child(even) td { background: #f7fafc; }
.rule-card { background: #f0fff4; border-left: 4px solid #48bb78;
             padding: 12px 16px; margin-bottom: 12px; border-radius: 4px; }
.rule-card strong { color: #276749; }
.rule-card .cond { color: #718096; font-size: 12px; margin-top: 4px; }
code { background: #edf2f7; padding: 2px 6px; border-radius: 3px; font-size: 13px; }
"""


def render_html(
    report: SummaryReport,
    provenance_rows=None,
) -> str:
    """Render a SummaryReport (and optional provenance rows) as self-contained HTML."""
    body_parts = [
        _html_header(report.diff, report.best_speedup),
        _html_bottlenecks_table(report.diff),
        _html_detail_section(report.diff),
        _html_history_table(report.loop_history),
    ]
    if provenance_rows:
        body_parts.append(_html_provenance_table(provenance_rows))
    body_parts.append(_html_lessons_section(report.rules))

    body = "\n".join(body_parts)
    return (
        "<!DOCTYPE html>\n"
        "<html lang='en'>\n"
        "<head>\n"
        "<meta charset='utf-8'>\n"
        f"<title>Optimization Summary — {_esc(report.diff.model_name)}</title>\n"
        f"<style>{_CSS}</style>\n"
        "</head>\n"
        "<body>\n"
        f"{body}\n"
        "</body>\n"
        "</html>\n"
    )


# ---------------------------------------------------------------------------
# Section builders
# ---------------------------------------------------------------------------

def _html_header(diff: ProfileDiff, best_speedup: float) -> str:
    device = diff.device_name or "unknown"
    saved_ms = diff.wall_time_saved_ns / 1e6
    speedup_cls = _speedup_class(diff.total_speedup)
    return (
        f"<h1>Optimization Summary — {_esc(diff.model_name)}</h1>\n"
        f"<div class='meta'>"
        f"<span><strong>Device:</strong> {_esc(device)}</span>"
        f"<span><strong>Total Speedup:</strong> "
        f"<span class='{speedup_cls}'>{diff.total_speedup:.2f}x</span></span>"
        f"<span><strong>Wall Time Saved:</strong> {saved_ms:.2f} ms</span>"
        f"<span><strong>Best Speedup:</strong> {best_speedup:.2f}x</span>"
        f"</div>"
    )


def _html_bottlenecks_table(diff: ProfileDiff, top_n: int = 5) -> str:
    candidates = diff.top_bottlenecks_before[:top_n]
    if not candidates:
        return ""
    rows = []
    for i, d in enumerate(candidates, 1):
        before_ms = f"{d.duration_before_ns / 1e6:.3f}"
        after_ms = f"{d.duration_after_ns / 1e6:.3f}" if d.duration_after_ns is not None else "—"
        speedup = f"{d.speedup:.2f}x" if d.speedup is not None else "—"
        speedup_cls = _speedup_class(d.speedup or 1.0)
        bt_before_cls = _bt_class(d.bottleneck_before)
        bt_after_str = (
            f"<span class='{_bt_class(d.bottleneck_after)}'>{d.bottleneck_after}</span>"
            if d.bottleneck_after else "—"
        )
        rows.append(
            f"<tr>"
            f"<td>{i}</td>"
            f"<td><code>{_esc(d.operator_name)}</code></td>"
            f"<td>{before_ms}</td>"
            f"<td>{after_ms}</td>"
            f"<td class='{speedup_cls}'>{speedup}</td>"
            f"<td><span class='{bt_before_cls}'>{d.bottleneck_before}</span> → {bt_after_str}</td>"
            f"</tr>"
        )
    return (
        "<h2>Top Bottlenecks</h2>\n"
        "<table>\n"
        "<thead><tr><th>#</th><th>Operator</th><th>Before (ms)</th>"
        "<th>After (ms)</th><th>Speedup</th><th>Bottleneck</th></tr></thead>\n"
        f"<tbody>{''.join(rows)}</tbody>\n"
        "</table>"
    )


def _html_detail_section(diff: ProfileDiff) -> str:
    changed = [
        d for d in diff.operator_diffs
        if d.match_type in ("exact", "fused_into") and (
            d.rewrite_ops_applied or d.fusion_partners
        )
    ]
    if not changed:
        return ""
    parts = ["<h2>Optimization Detail</h2>"]
    for d in changed:
        before_ms = d.duration_before_ns / 1e6
        after_ms = d.duration_after_ns / 1e6 if d.duration_after_ns is not None else None
        speedup_str = f"{d.speedup:.2f}x" if d.speedup is not None else "—"
        speedup_cls = _speedup_class(d.speedup or 1.0)
        bt_after = d.bottleneck_after or "—"
        parts.append(
            f"<h3><code>{_esc(d.operator_name)}</code> (call {d.call_index})</h3>"
            f"<ul>"
            f"<li><strong>Before</strong>: {before_ms:.3f} ms "
            f"(<span class='{_bt_class(d.bottleneck_before)}'>{d.bottleneck_before}</span>)</li>"
            + (f"<li><strong>After</strong>: {after_ms:.3f} ms "
               f"(<span class='{_bt_class(bt_after)}'>{bt_after}</span>)</li>"
               if after_ms is not None else "")
            + f"<li><strong>Speedup</strong>: <span class='{speedup_cls}'>{speedup_str}</span></li>"
            + (f"<li><strong>Applied</strong>: {', '.join(_esc(x) for x in d.rewrite_ops_applied)}</li>"
               if d.rewrite_ops_applied else "")
            + (f"<li><strong>Fused with</strong>: {', '.join(_esc(x) for x in d.fusion_partners)}</li>"
               if d.fusion_partners else "")
            + "</ul>"
        )
    return "\n".join(parts)


def _html_history_table(history: list[dict]) -> str:
    if not history:
        return ""
    rows = []
    for h in history:
        it = h.get("iteration", "?")
        bottleneck = h.get("bottleneck", "?")
        mem_hits = h.get("memory_hits", "?")
        plans = h.get("plans_tried", "?")
        best = h.get("best_speedup_so_far", "?")
        best_str = f"{best:.3f}x" if isinstance(best, float) else str(best)
        rows.append(
            f"<tr><td>{it}</td><td>{_esc(str(bottleneck))}</td>"
            f"<td>{mem_hits}</td><td>{plans}</td><td>{best_str}</td></tr>"
        )
    return (
        "<h2>Optimization Loop History</h2>\n"
        "<table>\n"
        "<thead><tr><th>Iteration</th><th>Bottleneck</th>"
        "<th>Memory Hits</th><th>Plans Tried</th><th>Best Speedup</th></tr></thead>\n"
        f"<tbody>{''.join(rows)}</tbody>\n"
        "</table>"
    )


def _html_provenance_table(rows) -> str:
    """rows: list[ProvenanceRow]"""
    from operator_profiler.summarizer.provenance import render_provenance_html
    return "<h2>Provenance Viewer</h2>\n" + render_provenance_html(rows)


def _html_lessons_section(rules: list[OptimizationRule]) -> str:
    if not rules:
        return ""
    cards = []
    for rule in sorted(rules, key=lambda r: r.speedup, reverse=True):
        cond_html = (
            f"<p class='cond'>Conditions: {_esc('; '.join(rule.conditions))}</p>"
            if rule.conditions else ""
        )
        model_html = (
            f"<p class='cond'>Example model: {_esc(rule.example_model)}</p>"
            if rule.example_model else ""
        )
        cards.append(
            f"<div class='rule-card'>"
            f"<strong>Rule</strong>: {_esc(rule.rule_text)}"
            f"{cond_html}{model_html}"
            f"</div>"
        )
    return "<h2>Lessons Learned</h2>\n" + "\n".join(cards)


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _esc(s: str) -> str:
    return s.replace("&", "&amp;").replace("<", "&lt;").replace(">", "&gt;")


def _speedup_class(speedup: float) -> str:
    if speedup >= 1.2:
        return "speedup-high"
    if speedup >= 1.05:
        return "speedup-med"
    return "speedup-low"


def _bt_class(bottleneck: str) -> str:
    mapping = {
        "compute_bound": "bt-compute",
        "memory_bound": "bt-memory",
        "latency_bound": "bt-latency",
    }
    return mapping.get(bottleneck, "bt-unknown")
