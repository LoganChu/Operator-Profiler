"""
apply_buffer_sharing — annotates two non-overlapping buffers as aliasable.

Sets ``node.meta["buffer_alias_of"] = source_node_name`` on the target node.
When ``validate_liveness=True`` (the default), a conservative liveness
analysis rejects the operation if the two nodes' live ranges overlap — false
negatives in liveness equal silent memory corruption, so over-refusal is safe.
"""
from __future__ import annotations

import torch.fx

from operator_profiler.rewriter.dsl import BufferSharingOp, RewriteValidationError


def _find_node(gm: torch.fx.GraphModule, name: str) -> torch.fx.Node:
    for n in gm.graph.nodes:
        if n.name == name:
            return n
    raise RewriteValidationError(f"Node '{name}' not found in graph")


def _compute_liveness(
    gm: torch.fx.GraphModule,
) -> dict[str, tuple[int, int]]:
    """
    Return ``{node_name: (first_def_index, last_use_index)}`` for all nodes.

    ``first_def_index`` is the node's position in topological order.
    ``last_use_index`` is the maximum position of any direct user (or
    ``len(nodes)-1`` if the node feeds into the graph output).
    """
    topo_list = list(gm.graph.nodes)
    index: dict[torch.fx.Node, int] = {n: i for i, n in enumerate(topo_list)}
    n_nodes = len(topo_list)
    liveness: dict[str, tuple[int, int]] = {}

    for i, node in enumerate(topo_list):
        last_use = i
        for user in node.users:
            if user.op == "output":
                last_use = n_nodes - 1
                break
            user_idx = index.get(user)
            if user_idx is not None:
                last_use = max(last_use, user_idx)
            else:
                last_use = n_nodes - 1
        liveness[node.name] = (i, last_use)

    return liveness


def _live_ranges_overlap(a: tuple[int, int], b: tuple[int, int]) -> bool:
    """True iff intervals [a1,a2] and [b1,b2] share at least one point."""
    a1, a2 = a
    b1, b2 = b
    return a1 <= b2 and b1 <= a2


def apply_buffer_sharing(
    gm: torch.fx.GraphModule, op: BufferSharingOp
) -> torch.fx.GraphModule:
    """
    Mark ``op.target_node`` as aliasing ``op.source_node``.

    If ``validate_liveness=True``, raises ``RewriteValidationError`` when the
    two nodes have overlapping live ranges (conservative safety check).
    """
    _find_node(gm, op.source_node)  # existence check
    target_node = _find_node(gm, op.target_node)

    if op.validate_liveness:
        liveness = _compute_liveness(gm)
        src_range = liveness.get(op.source_node)
        tgt_range = liveness.get(op.target_node)
        if src_range is None or tgt_range is None:
            raise RewriteValidationError(
                f"Cannot determine liveness for buffer_sharing op '{op.id}'"
            )
        if _live_ranges_overlap(src_range, tgt_range):
            raise RewriteValidationError(
                f"Buffer sharing op '{op.id}' rejected: "
                f"live ranges of '{op.source_node}' {src_range} and "
                f"'{op.target_node}' {tgt_range} overlap"
            )

    target_node.meta["buffer_alias_of"] = op.source_node
    gm.graph.lint()
    gm.recompile()
    return gm
