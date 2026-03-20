"""
ProvenanceTracker — captures node identity before graph mutation and writes
``source_operators`` metadata onto the resulting fused node.

After fusion a single Triton kernel may correspond to multiple original
``aten::`` operators.  The tracker writes:

    node.meta["source_operators"]  – flat list of original aten targets
    node.meta["source_node_names"] – corresponding fx.Node names
    node.meta["is_fused"]          – True

These mirror ``OperatorRecord.is_fused`` / ``fused_with`` in schema/profile.py.

Multi-hop fusion is supported: if a node was already the product of a prior
fusion (i.e. ``node.meta["source_operators"]`` is already set), ``snapshot()``
inherits that flat list, so the final chain is always one level deep.
"""
from __future__ import annotations

import json
from dataclasses import dataclass, field
from pathlib import Path
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    import torch.fx as fx


@dataclass
class NodeProvenance:
    original_name: str
    original_target: str
    original_op: str
    # Inherited from prior fusions — populated only for multi-hop nodes
    source_operators: list[str] = field(default_factory=list)


class ProvenanceTracker:
    """Two-phase tracker: snapshot before mutation, write after."""

    def snapshot(self, nodes: list["fx.Node"]) -> list[NodeProvenance]:
        """
        Called BEFORE any graph mutation.

        For each node, captures its name, target, and any already-recorded
        ``source_operators`` from ``node.meta`` (supports multi-hop fusion).
        """
        result: list[NodeProvenance] = []
        for node in nodes:
            target_str = str(node.target)
            existing: list[str] = list(node.meta.get("source_operators", []))
            result.append(
                NodeProvenance(
                    original_name=node.name,
                    original_target=target_str,
                    original_op=node.op,
                    source_operators=existing,
                )
            )
        return result

    def write(
        self, fused_node: "fx.Node", provenance: list[NodeProvenance]
    ) -> None:
        """
        Called AFTER fusion succeeds.

        Builds a flat ``source_operators`` list: if a provenance entry already
        carried inherited sources (multi-hop), those are expanded instead of
        using the entry's own ``original_target``.
        """
        source_operators: list[str] = []
        source_node_names: list[str] = []

        for p in provenance:
            source_node_names.append(p.original_name)
            if p.source_operators:
                # Multi-hop: flatten the previously fused chain
                source_operators.extend(p.source_operators)
            else:
                source_operators.append(p.original_target)

        fused_node.meta["source_operators"] = source_operators
        fused_node.meta["source_node_names"] = source_node_names
        fused_node.meta["is_fused"] = True


def export_provenance_jsonl(gm: "fx.GraphModule", path: str | Path) -> None:
    """
    Walk all nodes with ``meta["is_fused"] == True`` and emit a JSONL sidecar
    in the same format as ``INDUCTOR_PROVENANCE=1``::

        {"generated_kernel_name": "fused_linear_relu_0",
         "source_ops": ["aten::linear", "aten::relu"],
         "source_locations": []}

    This file can be passed to ``ManifestBuilder`` as
    ``provenance_jsonl_path`` — no changes required to the Mapper.
    """
    path = Path(path)
    lines: list[str] = []
    for node in gm.graph.nodes:
        if not node.meta.get("is_fused"):
            continue
        entry = {
            "generated_kernel_name": node.name,
            "source_ops": list(node.meta.get("source_operators", [])),
            "source_locations": [],
        }
        lines.append(json.dumps(entry))
    content = "\n".join(lines) + ("\n" if lines else "")
    path.write_text(content, encoding="utf-8")
