"""
VerificationGate — numerically compares a rewritten GraphModule against the
original to confirm equivalence within given tolerances.

Algorithm
---------
1. Generate dummy inputs from ``placeholder`` nodes in topological order
   (using ``node.meta["val"]`` shape/dtype when available, else defaults).
2. Run both graphs under ``torch.no_grad()`` / ``eval()`` mode.
3. Compare outputs with ``torch.testing.assert_close``.
4. On failure: use a recording ``Interpreter`` to identify the first
   diverging intermediate node.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

import torch
import torch.fx


# ---------------------------------------------------------------------------
# Result types
# ---------------------------------------------------------------------------

@dataclass
class NodeDiff:
    node_name: str
    max_abs_error: float
    original_shape: tuple[int, ...]
    rewritten_shape: tuple[int, ...]


@dataclass
class VerificationResult:
    op_id: str
    passed: bool
    max_abs_error: float | None
    node_diffs: list[NodeDiff] = field(default_factory=list)
    error_message: str | None = None


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _make_dummy_input(
    node: torch.fx.Node,
    input_shapes: dict[str, tuple[int, ...]] | None,
    input_dtypes: dict[str, torch.dtype] | None,
) -> torch.Tensor:
    """Generate a reproducible dummy tensor for a placeholder node."""
    val = node.meta.get("val")
    if val is not None and hasattr(val, "shape") and hasattr(val, "dtype"):
        shape = tuple(int(d) for d in val.shape)
        dtype = val.dtype
    elif input_shapes and node.name in input_shapes:
        shape = tuple(input_shapes[node.name])
        dtype = (input_dtypes or {}).get(node.name, torch.float32)
    else:
        shape = (2, 3)
        dtype = torch.float32

    if dtype in (torch.float16, torch.float32, torch.float64, torch.bfloat16):
        return torch.randn(shape, dtype=dtype)
    if dtype in (torch.int32, torch.int64, torch.int16, torch.int8, torch.uint8):
        return torch.randint(0, 10, shape, dtype=dtype)
    if dtype == torch.bool:
        return torch.randint(0, 2, shape, dtype=torch.uint8).bool()
    return torch.randn(shape, dtype=torch.float32)


def _normalize_output(out: Any) -> torch.Tensor:
    if isinstance(out, torch.Tensor):
        return out
    if isinstance(out, (tuple, list)) and len(out) > 0:
        first = out[0]
        if isinstance(first, torch.Tensor):
            return first
    return torch.tensor(float(out))


class _RecordingInterpreter(torch.fx.Interpreter):
    """Records per-node tensor outputs during forward execution."""

    def __init__(self, gm: torch.fx.GraphModule) -> None:
        super().__init__(gm)
        self.node_outputs: dict[str, torch.Tensor] = {}

    def run_node(self, n: torch.fx.Node) -> Any:
        result = super().run_node(n)
        if isinstance(result, torch.Tensor):
            self.node_outputs[n.name] = result.detach().clone()
        return result


# ---------------------------------------------------------------------------
# VerificationGate
# ---------------------------------------------------------------------------

class VerificationGate:
    def __init__(
        self,
        original_gm: torch.fx.GraphModule,
        rewritten_gm: torch.fx.GraphModule,
        op_id: str = "",
        atol: float = 1e-5,
        rtol: float = 1e-5,
        device: str = "cpu",
        input_shapes: dict[str, tuple[int, ...]] | None = None,
        input_dtypes: dict[str, torch.dtype] | None = None,
    ) -> None:
        self.original_gm = original_gm
        self.rewritten_gm = rewritten_gm
        self.op_id = op_id
        self.atol = atol
        self.rtol = rtol
        self.device = device
        self.input_shapes = input_shapes
        self.input_dtypes = input_dtypes

    def _generate_inputs(
        self, gm: torch.fx.GraphModule
    ) -> list[torch.Tensor]:
        torch.manual_seed(42)
        return [
            _make_dummy_input(n, self.input_shapes, self.input_dtypes).to(self.device)
            for n in gm.graph.nodes
            if n.op == "placeholder"
        ]

    def verify(self) -> VerificationResult:
        orig_gm = self.original_gm
        rw_gm = self.rewritten_gm
        orig_gm.eval()
        rw_gm.eval()

        orig_inputs = self._generate_inputs(orig_gm)
        rw_inputs = self._generate_inputs(rw_gm)

        with torch.no_grad():
            try:
                orig_raw = orig_gm(*orig_inputs)
            except Exception as exc:
                return VerificationResult(
                    op_id=self.op_id,
                    passed=False,
                    max_abs_error=None,
                    error_message=f"Original graph execution failed: {exc}",
                )
            try:
                rw_raw = rw_gm(*rw_inputs)
            except Exception as exc:
                return VerificationResult(
                    op_id=self.op_id,
                    passed=False,
                    max_abs_error=None,
                    error_message=f"Rewritten graph execution failed: {exc}",
                )

        orig_out = _normalize_output(orig_raw)
        rw_out = _normalize_output(rw_raw)

        if orig_out.shape != rw_out.shape:
            return VerificationResult(
                op_id=self.op_id,
                passed=False,
                max_abs_error=float("inf"),
                error_message=(
                    f"Shape mismatch: original {tuple(orig_out.shape)} "
                    f"vs rewritten {tuple(rw_out.shape)}"
                ),
            )

        try:
            torch.testing.assert_close(rw_out, orig_out, atol=self.atol, rtol=self.rtol)
            max_abs_error = float(
                (rw_out.float() - orig_out.float()).abs().max().item()
            )
            return VerificationResult(
                op_id=self.op_id,
                passed=True,
                max_abs_error=max_abs_error,
            )
        except AssertionError as exc:
            max_abs_error = float(
                (rw_out.float() - orig_out.float()).abs().max().item()
            )
            node_diffs = self._compute_node_diffs(
                orig_gm, rw_gm, orig_inputs, rw_inputs
            )
            return VerificationResult(
                op_id=self.op_id,
                passed=False,
                max_abs_error=max_abs_error,
                node_diffs=node_diffs,
                error_message=str(exc),
            )

    def _compute_node_diffs(
        self,
        orig_gm: torch.fx.GraphModule,
        rw_gm: torch.fx.GraphModule,
        orig_inputs: list[torch.Tensor],
        rw_inputs: list[torch.Tensor],
    ) -> list[NodeDiff]:
        orig_interp = _RecordingInterpreter(orig_gm)
        rw_interp = _RecordingInterpreter(rw_gm)

        with torch.no_grad():
            try:
                orig_interp.run(*orig_inputs)
            except Exception:
                return []
            try:
                rw_interp.run(*rw_inputs)
            except Exception:
                return []

        diffs: list[NodeDiff] = []
        for name, orig_t in orig_interp.node_outputs.items():
            if name not in rw_interp.node_outputs:
                continue
            rw_t = rw_interp.node_outputs[name]
            if orig_t.shape != rw_t.shape:
                diffs.append(
                    NodeDiff(
                        node_name=name,
                        max_abs_error=float("inf"),
                        original_shape=tuple(orig_t.shape),
                        rewritten_shape=tuple(rw_t.shape),
                    )
                )
            else:
                err = float((rw_t.float() - orig_t.float()).abs().max().item())
                if err > self.atol:
                    diffs.append(
                        NodeDiff(
                            node_name=name,
                            max_abs_error=err,
                            original_shape=tuple(orig_t.shape),
                            rewritten_shape=tuple(rw_t.shape),
                        )
                    )
        return diffs
