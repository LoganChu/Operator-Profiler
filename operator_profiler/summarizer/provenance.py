"""
ProvenanceViewer — 4-column join of operator → kernel → metrics → rewrite op.

Columns: Original PyTorch Op | Inductor Kernel | Nsight Metrics | Optimization Applied
"""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from operator_profiler.schema.profile import OperatorAttributedProfile
    from operator_profiler.rewriter.dsl import RewritePlan


@dataclass
class ProvenanceRow:
    operator_id: str
    operator_name: str
    call_index: int
    is_fused: bool
    kernel_name: str                          # demangled_name or kernel_name
    duration_ns: int
    dram_bytes_read: int | None
    dram_bytes_written: int | None
    achieved_occupancy: float | None
    tensor_core_active_pct: float | None
    arithmetic_intensity: float | None
    attribution_method: str
    confidence: str
    rewrite_ops: list[str] = field(default_factory=list)  # DSL op IDs


def build_provenance_rows(
    profile: "OperatorAttributedProfile",
    plan: "RewritePlan | None",
) -> list[ProvenanceRow]:
    """Build provenance rows from a profile and optional rewrite plan."""
    from operator_profiler.rewriter.dsl import (
        FuseOp, ReorderOp, ChangeLayoutOp, BufferSharingOp,
    )

    # Pre-index DSL ops by operator_id they touch
    op_index: dict[str, list[str]] = {}
    if plan is not None:
        for dsl_op in plan.ops:
            node_ids: list[str] = []
            if isinstance(dsl_op, FuseOp):
                node_ids = list(dsl_op.nodes)
            elif isinstance(dsl_op, ReorderOp):
                node_ids = [dsl_op.node]
            elif isinstance(dsl_op, ChangeLayoutOp):
                node_ids = [dsl_op.target_node]
            elif isinstance(dsl_op, BufferSharingOp):
                node_ids = [dsl_op.source_node, dsl_op.target_node]
            for nid in node_ids:
                op_index.setdefault(nid, []).append(dsl_op.id)

    rows: list[ProvenanceRow] = []
    for op in profile.operators:
        rewrite_ops = op_index.get(op.operator_id, [])
        if not op.kernels:
            rows.append(
                ProvenanceRow(
                    operator_id=op.operator_id,
                    operator_name=op.operator_name,
                    call_index=op.call_index,
                    is_fused=op.is_fused,
                    kernel_name="(no kernels)",
                    duration_ns=0,
                    dram_bytes_read=None,
                    dram_bytes_written=None,
                    achieved_occupancy=None,
                    tensor_core_active_pct=None,
                    arithmetic_intensity=None,
                    attribution_method="n/a",
                    confidence="n/a",
                    rewrite_ops=rewrite_ops,
                )
            )
            continue
        for kernel in op.kernels:
            rows.append(
                ProvenanceRow(
                    operator_id=op.operator_id,
                    operator_name=op.operator_name,
                    call_index=op.call_index,
                    is_fused=op.is_fused,
                    kernel_name=kernel.demangled_name or kernel.kernel_name,
                    duration_ns=kernel.duration_ns,
                    dram_bytes_read=kernel.metrics.dram_bytes_read,
                    dram_bytes_written=kernel.metrics.dram_bytes_written,
                    achieved_occupancy=kernel.metrics.achieved_occupancy,
                    tensor_core_active_pct=kernel.metrics.tensor_core_active_pct,
                    arithmetic_intensity=kernel.metrics.arithmetic_intensity,
                    attribution_method=kernel.attribution_method.value,
                    confidence=kernel.confidence.value,
                    rewrite_ops=rewrite_ops,
                )
            )
    return rows


def render_provenance_text(rows: list[ProvenanceRow]) -> str:
    """Plain-text provenance table. No external dependencies."""
    if not rows:
        return "No provenance rows.\n"

    # Column widths
    col_op = max(len("PyTorch Op"), max(len(r.operator_id) for r in rows))
    col_kernel = max(len("Inductor Kernel"), max(len(r.kernel_name) for r in rows))
    col_metrics = 40
    col_opt = max(len("Optimization Applied"), max(
        len(", ".join(r.rewrite_ops) or "-") for r in rows
    ))

    header = (
        f"{'PyTorch Op':<{col_op}}  "
        f"{'Inductor Kernel':<{col_kernel}}  "
        f"{'Nsight Metrics':<{col_metrics}}  "
        f"{'Optimization Applied':<{col_opt}}"
    )
    sep = "-" * len(header)
    lines = [header, sep]

    for r in rows:
        metrics_parts = []
        if r.duration_ns:
            metrics_parts.append(f"{r.duration_ns / 1e6:.3f}ms")
        if r.achieved_occupancy is not None:
            metrics_parts.append(f"occ={r.achieved_occupancy:.0%}")
        if r.arithmetic_intensity is not None:
            metrics_parts.append(f"AI={r.arithmetic_intensity:.1f}")
        metrics_str = " ".join(metrics_parts) or "-"
        opt_str = ", ".join(r.rewrite_ops) or "-"

        lines.append(
            f"{r.operator_id:<{col_op}}  "
            f"{r.kernel_name:<{col_kernel}}  "
            f"{metrics_str:<{col_metrics}}  "
            f"{opt_str:<{col_opt}}"
        )
    return "\n".join(lines) + "\n"


def render_provenance_rich(rows: list[ProvenanceRow]) -> None:
    """Render provenance table using rich.Table.

    Raises
    ------
    ImportError
        If ``rich`` is not installed.
    """
    from rich.console import Console
    from rich.table import Table

    table = Table(title="Provenance Viewer", show_lines=True)
    table.add_column("PyTorch Op", style="cyan", no_wrap=True)
    table.add_column("Inductor Kernel", style="green")
    table.add_column("Nsight Metrics")
    table.add_column("Optimization Applied", style="yellow")

    for r in rows:
        metrics_parts = []
        if r.duration_ns:
            metrics_parts.append(f"{r.duration_ns / 1e6:.3f} ms")
        if r.achieved_occupancy is not None:
            metrics_parts.append(f"occ={r.achieved_occupancy:.0%}")
        if r.arithmetic_intensity is not None:
            metrics_parts.append(f"AI={r.arithmetic_intensity:.1f} FLOP/B")
        if r.dram_bytes_read is not None:
            total_dram = (r.dram_bytes_read or 0) + (r.dram_bytes_written or 0)
            metrics_parts.append(f"DRAM={total_dram / 1024:.1f}KB")
        metrics_str = "\n".join(metrics_parts) or "-"

        fused_tag = " [fused]" if r.is_fused else ""
        opt_str = "\n".join(r.rewrite_ops) if r.rewrite_ops else "-"
        table.add_row(
            f"{r.operator_id}{fused_tag}",
            r.kernel_name,
            metrics_str,
            opt_str,
        )

    Console().print(table)


def render_provenance_html(rows: list[ProvenanceRow]) -> str:
    """Render a self-contained HTML provenance table. No external deps."""
    _CSS = """
    body { font-family: monospace; font-size: 13px; margin: 24px; }
    h2 { color: #333; }
    table { border-collapse: collapse; width: 100%; }
    th { background: #2d3748; color: #fff; padding: 8px 12px; text-align: left; }
    td { padding: 6px 12px; border-bottom: 1px solid #e2e8f0; vertical-align: top; }
    tr:nth-child(even) td { background: #f7fafc; }
    .fused { color: #805ad5; font-weight: bold; }
    .opt { color: #d69e2e; }
    .metric { color: #2b6cb0; }
    """
    rows_html = []
    for r in rows:
        metrics_parts = []
        if r.duration_ns:
            metrics_parts.append(f"{r.duration_ns / 1e6:.3f} ms")
        if r.achieved_occupancy is not None:
            metrics_parts.append(f"occ={r.achieved_occupancy:.0%}")
        if r.arithmetic_intensity is not None:
            metrics_parts.append(f"AI={r.arithmetic_intensity:.1f}")
        if r.dram_bytes_read is not None:
            total_dram = (r.dram_bytes_read or 0) + (r.dram_bytes_written or 0)
            metrics_parts.append(f"DRAM={total_dram / 1024:.1f}KB")
        metrics_str = "<br>".join(metrics_parts) or "-"

        fused_class = ' class="fused"' if r.is_fused else ""
        opt_str = "<br>".join(r.rewrite_ops) if r.rewrite_ops else "-"
        rows_html.append(
            f"<tr>"
            f"<td{fused_class}>{_esc(r.operator_id)}</td>"
            f"<td>{_esc(r.kernel_name)}</td>"
            f"<td class='metric'>{metrics_str}</td>"
            f"<td class='opt'>{opt_str}</td>"
            f"</tr>"
        )
    body = "\n".join(rows_html)
    return (
        "<!DOCTYPE html>\n"
        "<html>\n<head><meta charset='utf-8'>"
        f"<style>{_CSS}</style>"
        "</head>\n<body>\n"
        "<h2>Provenance Viewer</h2>\n"
        "<table>\n"
        "<thead><tr>"
        "<th>PyTorch Op</th>"
        "<th>Inductor Kernel</th>"
        "<th>Nsight Metrics</th>"
        "<th>Optimization Applied</th>"
        "</tr></thead>\n"
        f"<tbody>{body}</tbody>\n"
        "</table>\n</body>\n</html>\n"
    )


def _esc(s: str) -> str:
    """Minimal HTML escaping."""
    return s.replace("&", "&amp;").replace("<", "&lt;").replace(">", "&gt;")
