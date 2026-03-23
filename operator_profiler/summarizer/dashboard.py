"""
Rich CLI dashboard for the Summarizer θ_s.

Optional dependency: rich>=13.0. Falls back to plain text if not installed.

Install the extra: pip install operator-profiler[rich]
"""
from __future__ import annotations

import logging
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from operator_profiler.summarizer.schema import SummaryReport

logger = logging.getLogger(__name__)

# Guard: import rich at module level so tests can monkeypatch sys.modules
try:
    from rich.console import Console
    from rich.panel import Panel
    from rich.table import Table
    from rich.live import Live
    from rich.text import Text
    _RICH_AVAILABLE = True
except ImportError:  # pragma: no cover
    _RICH_AVAILABLE = False


class RichDashboard:
    """
    Displays a post-optimization summary using Rich panels.

    Falls back to plain text if rich is not installed.

    Parameters
    ----------
    report:
        The SummaryReport to display.
    console:
        Injected rich.console.Console (useful for testing to capture output).
    """

    def __init__(self, report: "SummaryReport", console=None) -> None:
        self._report = report
        self._console = console  # may be None; resolved lazily in _render_rich

    def render(self) -> None:
        """Print the full dashboard. Falls back to plain text if rich absent."""
        if _RICH_AVAILABLE:
            self._render_rich()
        else:
            self._render_plain()

    # ------------------------------------------------------------------
    # Rich rendering
    # ------------------------------------------------------------------

    def _render_rich(self) -> None:
        console = self._console
        if console is None:
            console = Console()

        diff = self._report.diff
        speedup_str = f"{diff.total_speedup:.2f}x"
        saved_ms = diff.wall_time_saved_ns / 1e6

        # 1. Summary panel
        summary_text = (
            f"[bold]Model[/bold]: {diff.model_name}\n"
            f"[bold]Device[/bold]: {diff.device_name or 'unknown'}\n"
            f"[bold]Total Speedup[/bold]: [green]{speedup_str}[/green]\n"
            f"[bold]Wall Time Saved[/bold]: {saved_ms:.2f} ms\n"
            f"[bold]Best Speedup[/bold]: {self._report.best_speedup:.2f}x"
        )
        if self._report.best_plan_description:
            summary_text += f"\n[bold]Best Plan[/bold]: {self._report.best_plan_description}"
        console.print(Panel(summary_text, title="Optimization Summary", border_style="blue"))

        # 2. Top bottlenecks table
        console.print(self._make_bottlenecks_table())

        # 3. Iteration history
        if self._report.loop_history:
            console.print(self._make_history_table())

        # 4. Lessons learned
        if self._report.rules:
            console.print(self._make_lessons_panel())

    def _render_plain(self) -> None:
        from operator_profiler.summarizer.markdown import render_markdown
        print(render_markdown(self._report))

    def _make_bottlenecks_table(self) -> "Table":
        table = Table(title="Top Bottlenecks", show_header=True, header_style="bold magenta")
        table.add_column("Rank", style="dim", width=4)
        table.add_column("Operator", style="cyan")
        table.add_column("Before (ms)", justify="right")
        table.add_column("After (ms)", justify="right")
        table.add_column("Speedup", justify="right")
        table.add_column("Bottleneck")

        for i, d in enumerate(self._report.diff.top_bottlenecks_before, 1):
            before_ms = f"{d.duration_before_ns / 1e6:.3f}"
            after_ms = f"{d.duration_after_ns / 1e6:.3f}" if d.duration_after_ns else "—"
            speedup = d.speedup or 1.0
            if speedup >= 1.2:
                speedup_str = f"[green]{speedup:.2f}x[/green]"
            elif speedup >= 1.05:
                speedup_str = f"[yellow]{speedup:.2f}x[/yellow]"
            else:
                speedup_str = f"[red]{speedup:.2f}x[/red]"
            bt_after = d.bottleneck_after or "—"
            table.add_row(
                str(i),
                d.operator_name,
                before_ms,
                after_ms,
                speedup_str,
                f"{d.bottleneck_before} → {bt_after}",
            )
        return table

    def _make_history_table(self) -> "Table":
        table = Table(title="Iteration History", show_header=True, header_style="bold blue")
        table.add_column("Iter", width=4)
        table.add_column("Bottleneck")
        table.add_column("Memory Hits", justify="right")
        table.add_column("Plans Tried", justify="right")
        table.add_column("Best Speedup So Far", justify="right")

        for h in self._report.loop_history:
            it = str(h.get("iteration", "?"))
            bottleneck = str(h.get("bottleneck", "?"))
            mem_hits = str(h.get("memory_hits", "?"))
            plans = str(h.get("plans_tried", "?"))
            best = h.get("best_speedup_so_far", 1.0)
            best_str = f"{best:.3f}x" if isinstance(best, float) else str(best)
            table.add_row(it, bottleneck, mem_hits, plans, best_str)
        return table

    def _make_lessons_panel(self) -> "Panel":
        rules = sorted(self._report.rules, key=lambda r: r.speedup, reverse=True)[:10]
        lines = []
        for rule in rules:
            lines.append(f"• {rule.rule_text}")
            if rule.conditions:
                lines.append(f"  [dim]Conditions: {'; '.join(rule.conditions)}[/dim]")
        content = "\n".join(lines) or "(none)"
        return Panel(content, title="Lessons Learned", border_style="green")


class LiveProgressDashboard:
    """
    Context manager for in-loop progress display during OptimizationLoop.

    Uses rich.Live if available; falls back to logging-only otherwise.

    Usage
    -----
    ::

        with LiveProgressDashboard(n_iterations=5) as dash:
            for i in range(5):
                dash.update(iteration=i, bottleneck="memory_bound",
                            plans_tried=3, speedup=1.12)
    """

    def __init__(self, n_iterations: int, console=None) -> None:
        self._n = n_iterations
        self._console = console
        self._live = None
        self._table: "Table | None" = None

    def __enter__(self) -> "LiveProgressDashboard":
        if _RICH_AVAILABLE:
            self._table = Table(
                title="Optimization Progress",
                show_header=True,
                header_style="bold blue",
            )
            self._table.add_column("Iter", width=4)
            self._table.add_column("Bottleneck")
            self._table.add_column("Plans", justify="right")
            self._table.add_column("Best Speedup", justify="right")

            console = self._console or Console()
            self._live = Live(self._table, console=console, refresh_per_second=4)
            self._live.__enter__()
        return self

    def __exit__(self, *args) -> None:
        if self._live is not None:
            self._live.__exit__(*args)

    def update(
        self,
        iteration: int,
        bottleneck: str,
        plans_tried: int,
        speedup: float,
    ) -> None:
        if _RICH_AVAILABLE and self._table is not None:
            speedup_str = (
                f"[green]{speedup:.3f}x[/green]" if speedup >= 1.2
                else f"[yellow]{speedup:.3f}x[/yellow]" if speedup >= 1.05
                else f"[red]{speedup:.3f}x[/red]"
            )
            self._table.add_row(
                str(iteration),
                bottleneck,
                str(plans_tried),
                speedup_str,
            )
        else:
            logger.info(
                "Iter %d | bottleneck=%s | plans=%d | speedup=%.3fx",
                iteration, bottleneck, plans_tried, speedup,
            )
