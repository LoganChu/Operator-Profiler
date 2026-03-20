"""
Tests for ProvenanceTracker and export_provenance_jsonl.

Coverage
--------
- ``snapshot()`` captures correct original_target strings
- ``write()`` sets meta["source_operators"], meta["source_node_names"], meta["is_fused"]
- Multi-hop fusion flattens the chain
- ``export_provenance_jsonl()`` writes valid JSONL with correct keys
"""
from __future__ import annotations

import json
import tempfile
from pathlib import Path

import pytest
import torch
import torch.fx

from operator_profiler.rewriter.provenance import (
    ProvenanceTracker,
    export_provenance_jsonl,
)


# ---------------------------------------------------------------------------
# Fixture: minimal two-node graph  (x → relu → output)
# ---------------------------------------------------------------------------

def _make_two_node_graph() -> tuple[torch.fx.GraphModule, torch.fx.Node, torch.fx.Node]:
    """Return (gm, linear_node, relu_node) for a graph: x → linear → relu → out."""
    graph = torch.fx.Graph()
    x = graph.placeholder("x")
    linear = graph.call_function(torch.nn.functional.linear, args=(x, torch.ones(3, 3)))
    relu = graph.call_function(torch.nn.functional.relu, args=(linear,))
    graph.output(relu)
    gm = torch.fx.GraphModule(torch.nn.Module(), graph)
    return gm, linear, relu


# ---------------------------------------------------------------------------
# test_snapshot_captures_target
# ---------------------------------------------------------------------------

class TestSnapshot:
    def test_captures_target_strings(self):
        _gm, linear, relu = _make_two_node_graph()
        tracker = ProvenanceTracker()
        provenance = tracker.snapshot([linear, relu])
        assert len(provenance) == 2
        assert provenance[0].original_name == linear.name
        assert provenance[1].original_name == relu.name
        # targets are the callable objects — str() of them
        assert str(torch.nn.functional.linear) in provenance[0].original_target
        assert str(torch.nn.functional.relu) in provenance[1].original_target

    def test_captures_op_field(self):
        _gm, linear, relu = _make_two_node_graph()
        tracker = ProvenanceTracker()
        provenance = tracker.snapshot([linear, relu])
        assert all(p.original_op == "call_function" for p in provenance)

    def test_empty_source_operators_for_fresh_nodes(self):
        _gm, linear, relu = _make_two_node_graph()
        tracker = ProvenanceTracker()
        provenance = tracker.snapshot([linear, relu])
        assert provenance[0].source_operators == []
        assert provenance[1].source_operators == []


# ---------------------------------------------------------------------------
# test_write_sets_meta_keys
# ---------------------------------------------------------------------------

class TestWrite:
    def test_sets_is_fused(self):
        _gm, linear, relu = _make_two_node_graph()
        tracker = ProvenanceTracker()
        provenance = tracker.snapshot([linear, relu])
        tracker.write(relu, provenance)
        assert relu.meta["is_fused"] is True

    def test_sets_source_operators(self):
        _gm, linear, relu = _make_two_node_graph()
        tracker = ProvenanceTracker()
        provenance = tracker.snapshot([linear, relu])
        tracker.write(relu, provenance)
        src_ops = relu.meta["source_operators"]
        assert len(src_ops) == 2
        assert str(torch.nn.functional.linear) in src_ops[0]
        assert str(torch.nn.functional.relu) in src_ops[1]

    def test_sets_source_node_names(self):
        _gm, linear, relu = _make_two_node_graph()
        tracker = ProvenanceTracker()
        provenance = tracker.snapshot([linear, relu])
        tracker.write(relu, provenance)
        names = relu.meta["source_node_names"]
        assert linear.name in names
        assert relu.name in names


# ---------------------------------------------------------------------------
# test_multi_hop_flattens_chain
# ---------------------------------------------------------------------------

class TestMultiHop:
    def test_second_fusion_flattens(self):
        """
        First fusion: linear + relu → fused_A  source_operators = ["aten::linear", "aten::relu"]
        Second fusion: fused_A + gelu → fused_B  source_operators = ["aten::linear", "aten::relu", "aten::gelu"]
        """
        graph = torch.fx.Graph()
        x = graph.placeholder("x")
        linear = graph.call_function(torch.nn.functional.linear, args=(x, torch.ones(3, 3)))
        relu = graph.call_function(torch.nn.functional.relu, args=(linear,))
        gelu = graph.call_function(torch.nn.functional.gelu, args=(relu,))
        graph.output(gelu)
        gm = torch.fx.GraphModule(torch.nn.Module(), graph)

        # Simulate first fusion: linear + relu → relu carries provenance
        tracker1 = ProvenanceTracker()
        prov1 = tracker1.snapshot([linear, relu])
        tracker1.write(relu, prov1)

        # relu now has source_operators set
        assert relu.meta["is_fused"] is True
        first_sources = relu.meta["source_operators"]
        assert len(first_sources) == 2

        # Second fusion: relu (already fused) + gelu → gelu
        tracker2 = ProvenanceTracker()
        prov2 = tracker2.snapshot([relu, gelu])
        tracker2.write(gelu, prov2)

        final_sources = gelu.meta["source_operators"]
        # Must be flat: 2 from relu's chain + 1 from gelu
        assert len(final_sources) == 3
        # gelu's own target must be in the list
        assert str(torch.nn.functional.gelu) in final_sources[2]


# ---------------------------------------------------------------------------
# test_export_provenance_jsonl_format
# ---------------------------------------------------------------------------

class TestExportProvenanceJsonl:
    def test_writes_valid_jsonl(self):
        _gm, linear, relu = _make_two_node_graph()
        tracker = ProvenanceTracker()
        prov = tracker.snapshot([linear, relu])
        tracker.write(relu, prov)  # relu is the fused node

        with tempfile.NamedTemporaryFile(
            suffix=".jsonl", delete=False, mode="w"
        ) as f:
            path = Path(f.name)

        try:
            export_provenance_jsonl(_gm, path)
            lines = path.read_text().strip().splitlines()
            assert len(lines) == 1
            record = json.loads(lines[0])
            assert "generated_kernel_name" in record
            assert "source_ops" in record
            assert "source_locations" in record
            assert record["generated_kernel_name"] == relu.name
            assert len(record["source_ops"]) == 2
        finally:
            path.unlink(missing_ok=True)

    def test_no_fused_nodes_writes_empty_file(self):
        graph = torch.fx.Graph()
        x = graph.placeholder("x")
        out = graph.call_function(torch.nn.functional.relu, args=(x,))
        graph.output(out)
        gm = torch.fx.GraphModule(torch.nn.Module(), graph)

        with tempfile.NamedTemporaryFile(
            suffix=".jsonl", delete=False, mode="w"
        ) as f:
            path = Path(f.name)

        try:
            export_provenance_jsonl(gm, path)
            content = path.read_text()
            assert content == ""
        finally:
            path.unlink(missing_ok=True)

    def test_source_locations_is_empty_list(self):
        _gm, linear, relu = _make_two_node_graph()
        tracker = ProvenanceTracker()
        prov = tracker.snapshot([linear, relu])
        tracker.write(relu, prov)

        with tempfile.NamedTemporaryFile(
            suffix=".jsonl", delete=False, mode="w"
        ) as f:
            path = Path(f.name)

        try:
            export_provenance_jsonl(_gm, path)
            record = json.loads(path.read_text().strip())
            assert record["source_locations"] == []
        finally:
            path.unlink(missing_ok=True)
