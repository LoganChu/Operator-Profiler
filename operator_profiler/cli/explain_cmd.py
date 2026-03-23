"""
operator-profiler explain --node <node_id>

Outputs a natural-language explanation for a specific operator node.
Pure data-driven — no LLM API calls.
"""
from __future__ import annotations

import json
import logging
from pathlib import Path

log = logging.getLogger(__name__)


def add_parser(subparsers) -> None:
    p = subparsers.add_parser(
        "explain",
        help="Explain what happened to a specific operator node during optimization.",
    )
    p.add_argument(
        "--node",
        required=True,
        help=(
            "Operator node ID (e.g. aten__linear_0 or aten::linear_0). "
            "Double-underscore is accepted as a shell-safe alias for '::'."
        ),
    )
    p.add_argument("--before", required=True, help="Before profile JSON path")
    p.add_argument("--after", required=True, help="After profile JSON path")
    p.add_argument(
        "--loop-result",
        required=True,
        metavar="LOOP_RESULT",
        help="LoopResult JSON path",
    )
    p.set_defaults(func=_run)


def _run(args) -> None:
    from operator_profiler.schema.profile import OperatorAttributedProfile
    from operator_profiler.planner.loop import LoopResult
    from operator_profiler.summarizer.diff import compute_diff
    from operator_profiler.summarizer.explain import explain_node

    before = OperatorAttributedProfile.model_validate_json(
        Path(args.before).read_text()
    )
    after = OperatorAttributedProfile.model_validate_json(
        Path(args.after).read_text()
    )
    loop_result = LoopResult.from_dict(json.loads(Path(args.loop_result).read_text()))

    diff = compute_diff(before=before, after=after, plan=loop_result.best_plan)

    explanation = explain_node(
        node_id=args.node,
        diff=diff,
        before=before,
        loop_result=loop_result,
    )
    print(explanation)
