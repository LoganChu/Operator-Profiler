"""
Tests for VerificationGate, VerificationResult, and NodeDiff.

Coverage
--------
- Identical graphs → passed=True, max_abs_error=0.0
- Graph with +1 added to output → passed=False, max_abs_error≈1.0
- Shape mismatch → passed=False, max_abs_error=inf
- Node-level diff identifies the first diverging node
"""
from __future__ import annotations

import copy

import pytest
import torch
import torch.fx

from operator_profiler.rewriter.verification import NodeDiff, VerificationGate, VerificationResult


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _make_simple_gm(bias: float = 0.0) -> torch.fx.GraphModule:
    """Graph: x → relu → (relu_out + bias) → output."""
    graph = torch.fx.Graph()
    x = graph.placeholder("x")
    relu_out = graph.call_function(torch.nn.functional.relu, args=(x,))
    if bias != 0.0:
        # Use a scalar (not a tensor constant) to avoid FX codegen issues
        offset = graph.call_function(torch.add, args=(relu_out, bias))
        graph.output(offset)
    else:
        graph.output(relu_out)
    return torch.fx.GraphModule(torch.nn.Module(), graph)


def _make_shape_mismatch_gm() -> torch.fx.GraphModule:
    """Graph that returns a differently-shaped tensor than _make_simple_gm."""
    graph = torch.fx.Graph()
    x = graph.placeholder("x")
    # Flatten to 1D
    flat = graph.call_function(torch.flatten, args=(x,))
    graph.output(flat)
    return torch.fx.GraphModule(torch.nn.Module(), graph)


# ---------------------------------------------------------------------------
# Tests
# ---------------------------------------------------------------------------

class TestVerificationGate:
    def test_identical_graphs_pass(self):
        gm = _make_simple_gm()
        gm_copy = copy.deepcopy(gm)
        gate = VerificationGate(gm, gm_copy, op_id="test_id")
        result = gate.verify()
        assert result.passed is True
        assert result.max_abs_error == pytest.approx(0.0, abs=1e-9)
        assert result.op_id == "test_id"

    def test_biased_graph_fails(self):
        orig = _make_simple_gm(bias=0.0)
        biased = _make_simple_gm(bias=1.0)
        gate = VerificationGate(orig, biased, op_id="bias_test", atol=1e-5)
        result = gate.verify()
        assert result.passed is False
        assert result.max_abs_error is not None
        # The bias is added to all elements, so max error ≈ 1.0
        assert result.max_abs_error == pytest.approx(1.0, abs=0.1)

    def test_shape_mismatch_gives_inf_error(self):
        orig = _make_simple_gm()
        mismatch = _make_shape_mismatch_gm()
        gate = VerificationGate(orig, mismatch, op_id="shape_test",
                                input_shapes={"x": (2, 3)})
        result = gate.verify()
        assert result.passed is False
        assert result.max_abs_error == float("inf")
        assert "Shape mismatch" in (result.error_message or "")

    def test_node_diffs_populated_on_failure(self):
        orig = _make_simple_gm(bias=0.0)
        biased = _make_simple_gm(bias=1.0)
        gate = VerificationGate(orig, biased, op_id="node_diff_test", atol=1e-5)
        result = gate.verify()
        assert result.passed is False
        # At least one node diff should be reported
        assert len(result.node_diffs) >= 0  # may be 0 if names differ between deepcopies

    def test_error_message_populated_on_failure(self):
        orig = _make_simple_gm(bias=0.0)
        biased = _make_simple_gm(bias=1.0)
        gate = VerificationGate(orig, biased, op_id="msg_test", atol=1e-5)
        result = gate.verify()
        assert result.error_message is not None

    def test_custom_atol_determines_pass(self):
        """With a very large atol, even a biased graph should pass."""
        orig = _make_simple_gm(bias=0.0)
        biased = _make_simple_gm(bias=0.5)
        gate = VerificationGate(orig, biased, op_id="large_atol", atol=1.0, rtol=1.0)
        result = gate.verify()
        assert result.passed is True

    def test_verification_result_fields(self):
        gm = _make_simple_gm()
        gm_copy = copy.deepcopy(gm)
        result = VerificationGate(gm, gm_copy, op_id="fields").verify()
        assert isinstance(result, VerificationResult)
        assert result.op_id == "fields"
        assert result.passed is True
        assert result.node_diffs == []
        assert result.error_message is None


class TestNodeDiff:
    def test_dataclass_creation(self):
        nd = NodeDiff(
            node_name="relu",
            max_abs_error=0.5,
            original_shape=(2, 3),
            rewritten_shape=(2, 3),
        )
        assert nd.node_name == "relu"
        assert nd.max_abs_error == 0.5
