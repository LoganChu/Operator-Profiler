"""
Tests for lower_to_inductor and LoweringResult.

All tests use backend="eager" (CPU) to avoid CUDA dependency.
GPU integration test is in tests/integration/test_rewriter_gpu.py.

Coverage
--------
- lower_to_inductor returns a callable (not a GraphModule)
- Callable produces correct outputs (same as original gm)
- warmup_iters=2 completes without error
- provenance JSONL is emitted when provenance_output_path is set
- fullgraph=False allows graph-break gracefully
- LoweringResult fields are populated correctly
"""
from __future__ import annotations

import json
import tempfile
from pathlib import Path

import pytest
import torch
import torch.fx

from operator_profiler.rewriter.lowering import LoweringResult, lower_to_inductor
from operator_profiler.rewriter.provenance import ProvenanceTracker


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _make_simple_gm() -> tuple[torch.fx.GraphModule, list[torch.Tensor]]:
    """x → relu → output, with a (4,) float32 example input."""
    graph = torch.fx.Graph()
    x = graph.placeholder("x")
    r = graph.call_function(torch.nn.functional.relu, args=(x,), name="relu_0")
    graph.output(r)
    gm = torch.fx.GraphModule(torch.nn.Module(), graph)
    return gm, [torch.randn(4)]


def _make_fused_gm() -> tuple[torch.fx.GraphModule, list[torch.Tensor]]:
    """
    Two-node graph where the last node has is_fused provenance metadata
    (simulates a prior fuse op).  Uses two placeholders to avoid tensor-
    constant codegen issues with torch.compile.
    """
    graph = torch.fx.Graph()
    x = graph.placeholder("x")
    relu = graph.call_function(
        torch.nn.functional.relu,
        args=(x,),
        name="relu_0",
    )
    sigmoid = graph.call_function(
        torch.sigmoid,
        args=(relu,),
        name="sigmoid_0",
    )
    graph.output(sigmoid)
    gm = torch.fx.GraphModule(torch.nn.Module(), graph)

    # Simulate a prior fuse op: relu + sigmoid → sigmoid carries provenance
    tracker = ProvenanceTracker()
    nodes = [relu, sigmoid]
    prov = tracker.snapshot(nodes)
    tracker.write(sigmoid, prov)

    return gm, [torch.randn(4)]


# ---------------------------------------------------------------------------
# Tests
# ---------------------------------------------------------------------------

class TestLowerToInductor:
    def test_returns_callable(self):
        gm, inputs = _make_simple_gm()
        result = lower_to_inductor(
            gm, inputs, warmup_iters=1, backend="eager", device="cpu"
        )
        assert callable(result.compiled_model)
        assert not isinstance(result.compiled_model, torch.fx.GraphModule)

    def test_compiled_model_produces_correct_output(self):
        gm, inputs = _make_simple_gm()
        gm.eval()
        with torch.no_grad():
            expected = gm(*inputs)

        result = lower_to_inductor(
            gm, inputs, warmup_iters=1, backend="eager", device="cpu"
        )
        with torch.no_grad():
            actual = result.compiled_model(*inputs)

        torch.testing.assert_close(actual, expected, atol=1e-6, rtol=1e-6)

    def test_warmup_runs_without_error(self):
        gm, inputs = _make_simple_gm()
        lower_to_inductor(gm, inputs, warmup_iters=2, backend="eager", device="cpu")
        # No exception = pass

    def test_provenance_jsonl_emitted_when_path_given(self):
        gm, inputs = _make_fused_gm()
        with tempfile.NamedTemporaryFile(
            suffix=".jsonl", delete=False
        ) as f:
            path = Path(f.name)

        try:
            result = lower_to_inductor(
                gm,
                inputs,
                warmup_iters=1,
                provenance_output_path=path,
                backend="eager",
                device="cpu",
            )
            assert result.provenance_jsonl_path == str(path)
            assert path.exists()
            content = path.read_text().strip()
            assert content != ""
            record = json.loads(content.splitlines()[0])
            assert "generated_kernel_name" in record
            assert "source_ops" in record
        finally:
            path.unlink(missing_ok=True)

    def test_no_provenance_jsonl_when_path_none(self):
        gm, inputs = _make_simple_gm()
        result = lower_to_inductor(
            gm, inputs, warmup_iters=1, backend="eager", device="cpu"
        )
        assert result.provenance_jsonl_path is None

    def test_lowering_result_fields(self):
        gm, inputs = _make_simple_gm()
        result = lower_to_inductor(
            gm, inputs, warmup_iters=1, backend="eager", device="cpu"
        )
        assert isinstance(result, LoweringResult)
        assert result.compile_mode == "eager"
        assert "x" in result.input_shapes
        assert result.input_shapes["x"] == [4]

    def test_fullgraph_false_allows_partial_graph(self):
        """fullgraph=False should not raise even if the graph contains ops
        that might cause a graph break."""
        gm, inputs = _make_simple_gm()
        result = lower_to_inductor(
            gm,
            inputs,
            warmup_iters=1,
            fullgraph=False,
            backend="eager",
            device="cpu",
        )
        assert callable(result.compiled_model)
