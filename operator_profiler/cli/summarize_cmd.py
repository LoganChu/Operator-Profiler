"""
operator-profiler summarize

Generates a human-readable optimization summary report from before/after
profiles, a loop result, and an optimization memory store.
"""
from __future__ import annotations

import json
import logging
import sys
from pathlib import Path

log = logging.getLogger(__name__)


def add_parser(subparsers) -> None:
    p = subparsers.add_parser(
        "summarize",
        help="Generate a human-readable optimization summary report.",
    )
    p.add_argument("--before", required=True, help="Before profile JSON path")
    p.add_argument("--after", required=True, help="After profile JSON path")
    p.add_argument(
        "--loop-result",
        required=True,
        metavar="LOOP_RESULT",
        help="LoopResult JSON path (produced by operator-profiler optimize or LoopResult.to_dict())",
    )
    p.add_argument("--memory", required=True, help="OptMemoryStore JSON path")
    p.add_argument(
        "--output",
        default=None,
        help="Output file path. If omitted, writes to stdout (markdown) or auto-names (html).",
    )
    p.add_argument(
        "--format",
        choices=["markdown", "html", "rich"],
        default="rich",
        help="Output format (default: rich; falls back to markdown if rich not installed)",
    )
    p.add_argument("--top-n", type=int, default=5, dest="top_n",
                   help="Number of top bottlenecks to include (default: 5)")
    p.set_defaults(func=_run)


def _run(args) -> None:
    from operator_profiler.schema.profile import OperatorAttributedProfile
    from operator_profiler.planner.schema import OptMemoryStore
    from operator_profiler.planner.loop import LoopResult
    from operator_profiler.summarizer.diff import compute_diff
    from operator_profiler.summarizer.rules import entries_to_rules
    from operator_profiler.summarizer.schema import SummaryReport

    before_path = Path(args.before)
    after_path = Path(args.after)
    loop_path = Path(args.loop_result)
    memory_path = Path(args.memory)

    log.info("Loading before profile from %s", before_path)
    before = OperatorAttributedProfile.model_validate_json(before_path.read_text())

    log.info("Loading after profile from %s", after_path)
    after = OperatorAttributedProfile.model_validate_json(after_path.read_text())

    log.info("Loading loop result from %s", loop_path)
    loop_result = LoopResult.from_dict(json.loads(loop_path.read_text()))

    log.info("Loading optimization memory from %s", memory_path)
    store = OptMemoryStore.model_validate_json(memory_path.read_text())

    diff = compute_diff(
        before=before,
        after=after,
        plan=loop_result.best_plan,
        top_n=args.top_n,
    )

    rules = entries_to_rules(store.entries, sort_by="speedup")
    lessons = [r.rule_text for r in rules[:10]]

    report = SummaryReport(
        diff=diff,
        rules=rules,
        lessons_learned=lessons,
        loop_history=loop_result.history,
        best_speedup=loop_result.best_speedup,
        best_plan_description=(
            loop_result.best_plan.description if loop_result.best_plan else None
        ),
    )

    fmt = args.format
    output_path = Path(args.output) if args.output else None

    if fmt == "rich":
        from operator_profiler.summarizer.dashboard import RichDashboard
        RichDashboard(report).render()
        if output_path:
            # Also write markdown for persistence
            from operator_profiler.summarizer.markdown import render_markdown
            output_path.write_text(render_markdown(report), encoding="utf-8")
            log.info("Markdown report written to %s", output_path)

    elif fmt == "markdown":
        from operator_profiler.summarizer.markdown import render_markdown
        text = render_markdown(report)
        if output_path:
            output_path.write_text(text, encoding="utf-8")
            log.info("Report written to %s", output_path)
        else:
            sys.stdout.write(text)

    elif fmt == "html":
        from operator_profiler.summarizer.html import render_html
        from operator_profiler.summarizer.provenance import build_provenance_rows
        prov_rows = build_provenance_rows(before, loop_result.best_plan)
        html = render_html(report, prov_rows)
        if output_path is None:
            output_path = Path(f"summary_{before.capture_metadata.model_name}.html")
        output_path.write_text(html, encoding="utf-8")
        log.info("HTML report written to %s", output_path)
        print(f"HTML report saved to: {output_path}")
