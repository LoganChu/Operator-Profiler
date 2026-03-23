"""Unit tests for OptimizationRule generation (summarizer/rules.py)."""
from __future__ import annotations

import pytest

from operator_profiler.planner.schema import GraphPattern, MemoryEntry
from operator_profiler.rewriter.dsl import (
    BufferSharingOp,
    ChangeLayoutOp,
    FuseOp,
    ReorderOp,
    RewritePlan,
)
from operator_profiler.summarizer.rules import (
    _summarise_rewrite_op,
    entries_to_rules,
    entry_to_rule,
)


def _entry(entry_id, ops, bottleneck, speedup, model=None):
    plan = RewritePlan(
        ops=[FuseOp(op="fuse", id="f1", nodes=ops[:2], strategy="inductor_fuse")]
        if len(ops) >= 2 else []
    )
    return MemoryEntry(
        entry_id=entry_id,
        graph_pattern=GraphPattern(op_sequence=ops, pattern_hash="xx"),
        bottleneck=bottleneck,
        rewrite_plan=plan,
        speedup=speedup,
        model_name=model,
        created_at="2026-03-22T00:00:00+00:00",
    )


class TestEntryToRule:
    def test_speedup_pct(self):
        e = _entry("e1", ["aten::conv2d", "aten::relu"], "memory_bound", 1.18)
        rule = entry_to_rule(e)
        assert rule.speedup_pct == pytest.approx(18.0, abs=0.05)

    def test_rule_text_contains_bottleneck(self):
        e = _entry("e1", ["aten::linear"], "latency_bound", 1.3)
        rule = entry_to_rule(e)
        assert "latency_bound" in rule.rule_text

    def test_rule_text_contains_speedup_pct(self):
        e = _entry("e1", ["aten::conv2d", "aten::relu"], "memory_bound", 1.5)
        rule = entry_to_rule(e)
        assert "50.0%" in rule.rule_text

    def test_conditions_memory_bound(self):
        e = _entry("e1", ["aten::mm"], "memory_bound", 1.2)
        rule = entry_to_rule(e)
        assert any("arithmetic_intensity" in c for c in rule.conditions)

    def test_conditions_compute_bound(self):
        e = _entry("e1", ["aten::mm", "aten::add"], "compute_bound", 1.1)
        rule = entry_to_rule(e)
        assert any("tensor_core" in c for c in rule.conditions)

    def test_conditions_latency_bound(self):
        e = _entry("e1", ["aten::relu"], "latency_bound", 1.05)
        rule = entry_to_rule(e)
        assert any("kernel_count" in c for c in rule.conditions)

    def test_conditions_unknown_empty(self):
        e = _entry("e1", ["aten::relu"], "unknown", 1.0)
        rule = entry_to_rule(e)
        assert rule.conditions == []

    def test_example_model_forwarded(self):
        e = _entry("e1", ["aten::conv2d", "aten::relu"], "memory_bound", 1.3, model="ResNet50")
        rule = entry_to_rule(e)
        assert rule.example_model == "ResNet50"

    def test_entry_id_forwarded(self):
        e = _entry("my_entry_id", ["aten::linear", "aten::gelu"], "memory_bound", 1.2)
        rule = entry_to_rule(e)
        assert rule.entry_id == "my_entry_id"


class TestSummariseRewriteOp:
    def test_fuse_op(self):
        op = FuseOp(op="fuse", id="f1", nodes=["a", "b"], strategy="inductor_fuse")
        s = _summarise_rewrite_op(op)
        assert "fuse" in s
        assert "inductor_fuse" in s

    def test_reorder_op_before(self):
        op = ReorderOp(op="reorder", id="r1", node="relu", before="conv")
        s = _summarise_rewrite_op(op)
        assert "reorder" in s
        assert "before=conv" in s

    def test_reorder_op_after(self):
        op = ReorderOp(op="reorder", id="r1", node="relu", after="conv")
        s = _summarise_rewrite_op(op)
        assert "after=conv" in s

    def test_change_layout_op(self):
        op = ChangeLayoutOp(
            op="change_layout", id="cl1", target_node="n",
            current_format="NCHW", target_format="NHWC"
        )
        s = _summarise_rewrite_op(op)
        assert "NCHW" in s and "NHWC" in s

    def test_buffer_sharing_op(self):
        op = BufferSharingOp(op="buffer_sharing", id="bs1", source_node="a", target_node="b")
        s = _summarise_rewrite_op(op)
        assert "buffer_sharing" in s


class TestEntriesToRules:
    def test_sorted_by_speedup_desc(self, memory_store_with_entries):
        rules = entries_to_rules(memory_store_with_entries.entries)
        assert rules[0].speedup >= rules[-1].speedup

    def test_top_n(self, memory_store_with_entries):
        rules = entries_to_rules(memory_store_with_entries.entries, top_n=2)
        assert len(rules) == 2

    def test_empty_store_returns_empty(self):
        rules = entries_to_rules([])
        assert rules == []

    def test_sort_by_created_at(self, memory_store_with_entries):
        rules = entries_to_rules(
            memory_store_with_entries.entries, sort_by="created_at"
        )
        assert len(rules) == 3

    def test_all_entries_converted(self, memory_store_with_entries):
        rules = entries_to_rules(memory_store_with_entries.entries)
        assert len(rules) == len(memory_store_with_entries.entries)
