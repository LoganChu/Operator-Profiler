"""
apply_reorder — moves a node to a new topological position while preserving
data-dependency order.

Uses ``node.prepend(anchor)`` / ``node.append(anchor)`` from the torch.fx API.
A BFS reachability check over ``node.users`` rejects moves that would violate
data-dependency ordering.
"""
from __future__ import annotations

import torch.fx

from operator_profiler.rewriter.dsl import ReorderOp, RewriteValidationError


def _find_node(gm: torch.fx.GraphModule, name: str) -> torch.fx.Node:
    for n in gm.graph.nodes:
        if n.name == name:
            return n
    raise RewriteValidationError(f"Node '{name}' not found in graph")


def _topo_reachable(start: torch.fx.Node, target: torch.fx.Node) -> bool:
    """
    BFS over ``node.users`` edges.

    Returns True if ``target`` is reachable from ``start``, meaning ``start``
    is a (transitive) dependency of ``target``.
    """
    if start is target:
        return True
    visited: set[torch.fx.Node] = set()
    queue = list(start.users.keys())
    while queue:
        current = queue.pop()
        if current is target:
            return True
        if current in visited:
            continue
        visited.add(current)
        queue.extend(current.users.keys())
    return False


def apply_reorder(
    gm: torch.fx.GraphModule, op: ReorderOp
) -> torch.fx.GraphModule:
    """
    Move ``op.node`` immediately before ``op.before`` or immediately after
    ``op.after``.

    Raises ``RewriteValidationError`` if the move would violate a data
    dependency (e.g. moving a node before something it depends on).
    """
    node = _find_node(gm, op.node)

    if op.before is not None:
        anchor = _find_node(gm, op.before)
        # Invalid if node depends on anchor: anchor → ... → node
        if _topo_reachable(anchor, node):
            raise RewriteValidationError(
                f"Cannot reorder '{op.node}' before '{op.before}': "
                f"'{op.node}' transitively depends on '{op.before}'"
            )
        # anchor.prepend(node) places node immediately BEFORE anchor
        anchor.prepend(node)
    else:
        anchor = _find_node(gm, op.after)  # type: ignore[arg-type]
        # Invalid if anchor depends on node: node → ... → anchor
        if _topo_reachable(node, anchor):
            raise RewriteValidationError(
                f"Cannot reorder '{op.node}' after '{op.after}': "
                f"'{op.after}' transitively depends on '{op.node}'"
            )
        # anchor.append(node) places node immediately AFTER anchor
        anchor.append(node)

    gm.graph.lint()
    gm.recompile()
    return gm
