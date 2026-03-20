"""
apply_change_layout — wraps a layout-sensitive node with memory-format
conversion calls.

Returns a **new** ``GraphModule`` (non-mutating) with:
  1. ``input.to(memory_format=<target>)`` inserted before the target node.
  2. Optionally ``.contiguous()`` appended after the target node.

Direct graph manipulation (``inserting_before`` / ``inserting_after``) is
used instead of ``torch.fx.Transformer`` to avoid Proxy compatibility issues
across PyTorch versions.
"""
from __future__ import annotations

import copy

import torch
import torch.fx

from operator_profiler.rewriter.dsl import ChangeLayoutOp, RewriteValidationError


# ---------------------------------------------------------------------------
# Layout-sensitive aten ops
# ---------------------------------------------------------------------------

LAYOUT_SENSITIVE_ATEN_OPS: frozenset[str] = frozenset(
    {
        "conv1d",
        "conv2d",
        "conv3d",
        "convolution",
        "_convolution",
        "aten::conv1d",
        "aten::conv2d",
        "aten::conv3d",
        "aten::convolution",
        "aten::_convolution",
    }
)

_FORMAT_TO_MEMORY_FORMAT: dict[str, torch.memory_format] = {
    "NHWC": torch.channels_last,
    "NCHW": torch.contiguous_format,
    "NDHWC": torch.channels_last_3d,
    "NCDHW": torch.contiguous_format,
    "NCL": torch.contiguous_format,
    "NLC": torch.contiguous_format,
}


def _find_node(gm: torch.fx.GraphModule, name: str) -> torch.fx.Node:
    for n in gm.graph.nodes:
        if n.name == name:
            return n
    raise RewriteValidationError(f"Node '{name}' not found in graph")


def _is_layout_sensitive(node: torch.fx.Node) -> bool:
    """Return True if the node's target is a layout-sensitive aten convolution."""
    target_str = str(node.target).lower()
    for candidate in LAYOUT_SENSITIVE_ATEN_OPS:
        if candidate in target_str:
            return True
    # Fallback: any op whose name contains "conv"
    if "conv" in target_str:
        return True
    return False


# ---------------------------------------------------------------------------
# Public entry point (direct graph surgery — no Transformer)
# ---------------------------------------------------------------------------

def apply_change_layout(
    gm: torch.fx.GraphModule, op: ChangeLayoutOp
) -> torch.fx.GraphModule:
    """
    Return a new ``GraphModule`` with memory-format conversions inserted
    around ``op.target_node``.

    Raises ``RewriteValidationError`` if the node does not exist or is not a
    layout-sensitive convolution op.
    """
    target_orig = _find_node(gm, op.target_node)

    if not _is_layout_sensitive(target_orig):
        raise RewriteValidationError(
            f"Node '{op.target_node}' (target={target_orig.target!r}) is not "
            f"a layout-sensitive op. Supported ops: {LAYOUT_SENSITIVE_ATEN_OPS}"
        )

    # Work on a deep copy so original is never mutated
    new_gm = copy.deepcopy(gm)
    target = _find_node(new_gm, op.target_node)

    target_fmt = _FORMAT_TO_MEMORY_FORMAT.get(op.target_format, torch.contiguous_format)

    # 1. Insert .to(memory_format=...) before target, wrapping its first arg
    if target.args:
        first_arg = target.args[0]
        with new_gm.graph.inserting_before(target):
            to_node = new_gm.graph.call_method(
                "to",
                args=(first_arg,),
                kwargs={"memory_format": target_fmt},
            )
        # Redirect first arg of target to the new to_node
        new_args = list(target.args)
        new_args[0] = to_node
        target.args = tuple(new_args)

    # 2. Optionally insert .contiguous() after target
    if op.insert_contiguous_after:
        with new_gm.graph.inserting_after(target):
            cont_node = new_gm.graph.call_method("contiguous", args=(target,))
        # Replace all downstream uses of target with cont_node,
        # but leave target's own position (cont_node still references target)
        target.replace_all_uses_with(
            cont_node,
            delete_user_cb=lambda user: user is not cont_node,
        )

    new_gm.graph.lint()
    new_gm.recompile()
    return new_gm
