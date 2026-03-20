"""
GPU integration tests for the Hybrid Executor rewriter pipeline.

Requires: CUDA GPU, torch with CUDA support.

Run with:
    conda run -n ml_env python -m pytest -m integration tests/integration/test_rewriter_gpu.py -v
"""
from __future__ import annotations

import pytest
import torch
import torch.fx

pytestmark = pytest.mark.integration


@pytest.fixture(scope="module", autouse=True)
def require_cuda():
    if not torch.cuda.is_available():
        pytest.skip("CUDA not available")


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _make_simple_gm() -> tuple[torch.fx.GraphModule, list[torch.Tensor]]:
    graph = torch.fx.Graph()
    x = graph.placeholder("x")
    r = graph.call_function(torch.nn.functional.relu, args=(x,), name="relu_0")
    graph.output(r)
    gm = torch.fx.GraphModule(torch.nn.Module(), graph)
    return gm, [torch.randn(4, device="cuda")]


# ---------------------------------------------------------------------------
# Tests
# ---------------------------------------------------------------------------

class TestLowerInductorCuda:
    def test_lower_inductor_cuda(self):
        """Full roundtrip: rewrite → lower → compiled callable produces correct CUDA output."""
        from operator_profiler.rewriter.lowering import lower_to_inductor

        gm, inputs = _make_simple_gm()
        gm.eval()
        with torch.no_grad():
            expected = gm(*inputs)

        result = lower_to_inductor(
            gm,
            inputs,
            warmup_iters=2,
            backend="inductor",
            device="cuda",
        )
        assert callable(result.compiled_model)

        with torch.no_grad():
            actual = result.compiled_model(*inputs)

        torch.testing.assert_close(actual, expected, atol=1e-5, rtol=1e-5)
        assert actual.device.type == "cuda"


class TestRewriterGpuRoundtrip:
    def test_reorder_then_lower(self):
        """Reorder independent nodes → lower to inductor on GPU."""
        from operator_profiler.rewriter.dsl import ReorderOp, RewritePlan
        from operator_profiler.rewriter.executor import ExecutorConfig, HybridExecutor
        from operator_profiler.rewriter.lowering import lower_to_inductor

        graph = torch.fx.Graph()
        x = graph.placeholder("x")
        r = graph.call_function(torch.nn.functional.relu, args=(x,), name="relu_0")
        s = graph.call_function(torch.sigmoid, args=(x,), name="sigmoid_0")
        out = graph.call_function(torch.add, args=(r, s), name="add_0")
        graph.output(out)
        gm = torch.fx.GraphModule(torch.nn.Module(), graph)

        plan = RewritePlan(
            ops=[ReorderOp(op="reorder", id="r0", node="sigmoid_0", before="relu_0")]
        )
        cfg = ExecutorConfig(device="cpu")  # verification on CPU
        result_gm, results = HybridExecutor(gm, plan, cfg).run()
        assert results[0].passed

        inputs = [torch.randn(4, device="cuda")]
        result_gm.to("cuda")
        lr = lower_to_inductor(
            result_gm, inputs, warmup_iters=2, backend="inductor", device="cuda"
        )
        assert callable(lr.compiled_model)
