"""
apply_fuse — annotates a group of nodes for fusion.

Strategy mapping
----------------
``inductor_fuse`` (default)
    Annotates each node in the group with ``node.meta["inductor_fuse_group"]``.
    No structural graph change — Inductor consumes the annotation during
    compilation.  This is the safe fallback for both adjacent and
    non-adjacent nodes.

``inline``
    Adjacent nodes: same annotation as ``inductor_fuse`` plus an ``inline``
    hint in the meta.  ``replace_pattern``-based structural rewriting is not
    attempted here because it is brittle for ``call_module`` nodes
    (e.g. ``nn.Linear``).

``custom_op``
    Same annotation behaviour; ``custom_op_name`` is written to meta for
    downstream consumption.

In all strategies the **last** node in the group is treated as the
representative fused node for ``ProvenanceTracker``.
"""
from __future__ import annotations

import torch.fx

from operator_profiler.rewriter.dsl import FuseOp, RewriteValidationError
from operator_profiler.rewriter.provenance import ProvenanceTracker


def _find_node(gm: torch.fx.GraphModule, name: str) -> torch.fx.Node:
    for n in gm.graph.nodes:
        if n.name == name:
            return n
    raise RewriteValidationError(f"Node '{name}' not found in graph")


def _is_adjacent(
    gm: torch.fx.GraphModule, node_names: list[str]
) -> bool:
    """
    Return True iff ``node_names`` appear consecutively (in order) in the
    graph's topological node list.
    """
    node_list = list(gm.graph.nodes)
    name_to_idx: dict[str, int] = {n.name: i for i, n in enumerate(node_list)}
    indices = []
    for name in node_names:
        if name not in name_to_idx:
            return False
        indices.append(name_to_idx[name])
    return all(
        indices[i + 1] == indices[i] + 1 for i in range(len(indices) - 1)
    )


def apply_fuse(
    gm: torch.fx.GraphModule, op: FuseOp
) -> torch.fx.GraphModule:
    """
    Annotate ``op.nodes`` for fusion and record provenance on the last node.

    All nodes receive ``meta["inductor_fuse_group"] = op.id``.  Additional
    strategy-specific meta keys are set as needed.  The last node in the
    group is treated as the representative "fused" node for provenance.
    """
    tracker = ProvenanceTracker()
    nodes = [_find_node(gm, name) for name in op.nodes]

    # Snapshot BEFORE any mutation
    provenance = tracker.snapshot(nodes)

    # Validate custom_op strategy requirement
    if op.strategy == "custom_op" and not op.custom_op_name:
        raise RewriteValidationError(
            f"Fuse op '{op.id}': custom_op_name is required for strategy='custom_op'"
        )

    adjacent = _is_adjacent(gm, op.nodes)

    # Annotate all nodes in the group
    for node in nodes:
        node.meta["inductor_fuse_group"] = op.id
        node.meta["fuse_strategy"] = op.strategy

    if op.strategy == "custom_op" and op.custom_op_name:
        for node in nodes:
            node.meta["custom_op_name"] = op.custom_op_name

    if op.strategy == "inline" and adjacent:
        # Mark that structural inline fusion is possible
        for node in nodes:
            node.meta["inline_fusion_eligible"] = True

    # The last node is the representative fused node
    fused_node = nodes[-1]
    fused_node.meta["fuse_source_nodes"] = [n.name for n in nodes]

    # Write provenance AFTER annotation
    tracker.write(fused_node, provenance)

    gm.graph.lint()
    gm.recompile()
    return gm
