"""
Tests for the Rewrite Plan DSL (dsl.py).

Coverage
--------
- Discriminated union routing for all 4 op types
- All model_validator constraints
- JSON roundtrip for all 4 op types
- Unknown ``"op"`` discriminator raises ``ValidationError``
- ``RewritePlan`` accepts heterogeneous ops list
"""
from __future__ import annotations

import json

import pytest
from pydantic import ValidationError

from operator_profiler.rewriter.dsl import (
    DSL_VERSION,
    BufferSharingOp,
    ChangeLayoutOp,
    FuseOp,
    ReorderOp,
    RewritePlan,
)


# ---------------------------------------------------------------------------
# FuseOp
# ---------------------------------------------------------------------------

class TestFuseOp:
    def test_basic_valid(self):
        op = FuseOp(op="fuse", id="f0", nodes=["a", "b"])
        assert op.strategy == "inductor_fuse"
        assert op.custom_op_name is None

    def test_custom_op_requires_name(self):
        with pytest.raises(ValidationError, match="custom_op_name is required"):
            FuseOp(op="fuse", id="f0", nodes=["a", "b"], strategy="custom_op")

    def test_custom_op_name_forbidden_for_other_strategies(self):
        with pytest.raises(ValidationError, match="custom_op_name must be null"):
            FuseOp(
                op="fuse",
                id="f0",
                nodes=["a", "b"],
                strategy="inductor_fuse",
                custom_op_name="my_op",
            )

    def test_custom_op_valid(self):
        op = FuseOp(
            op="fuse",
            id="f0",
            nodes=["a", "b"],
            strategy="custom_op",
            custom_op_name="torch.ops.custom.fused_fn",
        )
        assert op.custom_op_name == "torch.ops.custom.fused_fn"

    def test_inline_strategy(self):
        op = FuseOp(op="fuse", id="f0", nodes=["a", "b"], strategy="inline")
        assert op.strategy == "inline"

    def test_nodes_min_length(self):
        with pytest.raises(ValidationError):
            FuseOp(op="fuse", id="f0", nodes=["a"])  # only 1 node

    def test_json_roundtrip(self):
        op = FuseOp(op="fuse", id="f0", nodes=["linear_0", "relu_0"])
        raw = op.model_dump_json()
        reloaded = FuseOp.model_validate_json(raw)
        assert reloaded.id == op.id
        assert reloaded.nodes == op.nodes


# ---------------------------------------------------------------------------
# ReorderOp
# ---------------------------------------------------------------------------

class TestReorderOp:
    def test_before_set(self):
        op = ReorderOp(op="reorder", id="r0", node="dropout_0", before="output")
        assert op.before == "output"
        assert op.after is None

    def test_after_set(self):
        op = ReorderOp(op="reorder", id="r0", node="dropout_0", after="relu_0")
        assert op.after == "relu_0"
        assert op.before is None

    def test_both_none_raises(self):
        with pytest.raises(ValidationError, match="Exactly one"):
            ReorderOp(op="reorder", id="r0", node="dropout_0")

    def test_both_set_raises(self):
        with pytest.raises(ValidationError, match="Exactly one"):
            ReorderOp(
                op="reorder", id="r0", node="dropout_0",
                before="output", after="relu_0"
            )

    def test_json_roundtrip(self):
        op = ReorderOp(op="reorder", id="r0", node="dropout_0", before="output")
        reloaded = ReorderOp.model_validate_json(op.model_dump_json())
        assert reloaded.node == "dropout_0"
        assert reloaded.before == "output"


# ---------------------------------------------------------------------------
# ChangeLayoutOp
# ---------------------------------------------------------------------------

class TestChangeLayoutOp:
    def test_valid(self):
        op = ChangeLayoutOp(
            op="change_layout",
            id="cl0",
            target_node="conv2d_0",
            current_format="NCHW",
            target_format="NHWC",
        )
        assert op.insert_contiguous_after is True

    def test_same_format_raises(self):
        with pytest.raises(ValidationError, match="must differ"):
            ChangeLayoutOp(
                op="change_layout",
                id="cl0",
                target_node="conv2d_0",
                current_format="NCHW",
                target_format="NCHW",
            )

    def test_insert_contiguous_false(self):
        op = ChangeLayoutOp(
            op="change_layout",
            id="cl0",
            target_node="conv2d_0",
            current_format="NCHW",
            target_format="NHWC",
            insert_contiguous_after=False,
        )
        assert op.insert_contiguous_after is False

    def test_3d_formats(self):
        op = ChangeLayoutOp(
            op="change_layout",
            id="cl0",
            target_node="conv3d_0",
            current_format="NCDHW",
            target_format="NDHWC",
        )
        assert op.current_format == "NCDHW"

    def test_json_roundtrip(self):
        op = ChangeLayoutOp(
            op="change_layout", id="cl0", target_node="n",
            current_format="NCHW", target_format="NHWC"
        )
        reloaded = ChangeLayoutOp.model_validate_json(op.model_dump_json())
        assert reloaded.target_node == "n"
        assert reloaded.target_format == "NHWC"


# ---------------------------------------------------------------------------
# BufferSharingOp
# ---------------------------------------------------------------------------

class TestBufferSharingOp:
    def test_valid(self):
        op = BufferSharingOp(
            op="buffer_sharing",
            id="bs0",
            source_node="relu_0",
            target_node="relu_2",
        )
        assert op.validate_liveness is True

    def test_validate_liveness_false(self):
        op = BufferSharingOp(
            op="buffer_sharing", id="bs0",
            source_node="relu_0", target_node="relu_2",
            validate_liveness=False,
        )
        assert op.validate_liveness is False

    def test_json_roundtrip(self):
        op = BufferSharingOp(
            op="buffer_sharing", id="bs0",
            source_node="relu_0", target_node="relu_2",
        )
        reloaded = BufferSharingOp.model_validate_json(op.model_dump_json())
        assert reloaded.source_node == "relu_0"


# ---------------------------------------------------------------------------
# RewritePlan — discriminated union routing
# ---------------------------------------------------------------------------

class TestRewritePlan:
    def test_heterogeneous_ops(self):
        plan = RewritePlan(
            ops=[
                {"op": "fuse", "id": "f0", "nodes": ["a", "b"]},
                {"op": "reorder", "id": "r0", "node": "c", "before": "d"},
                {
                    "op": "change_layout", "id": "cl0", "target_node": "conv",
                    "current_format": "NCHW", "target_format": "NHWC",
                },
                {
                    "op": "buffer_sharing", "id": "bs0",
                    "source_node": "relu_0", "target_node": "relu_2",
                },
            ]
        )
        assert len(plan.ops) == 4
        assert isinstance(plan.ops[0], FuseOp)
        assert isinstance(plan.ops[1], ReorderOp)
        assert isinstance(plan.ops[2], ChangeLayoutOp)
        assert isinstance(plan.ops[3], BufferSharingOp)

    def test_unknown_op_raises(self):
        with pytest.raises(ValidationError):
            RewritePlan(
                ops=[{"op": "explode", "id": "x0", "nodes": ["a"]}]
            )

    def test_default_plan_version(self):
        plan = RewritePlan()
        assert plan.plan_version == DSL_VERSION

    def test_empty_ops(self):
        plan = RewritePlan(ops=[])
        assert plan.ops == []

    def test_full_json_roundtrip(self):
        payload = {
            "plan_version": "1.0",
            "source_profile_id": "aten::linear_0:aten::relu_0",
            "description": "Fuse linear+relu",
            "ops": [
                {"op": "fuse", "id": "f0", "nodes": ["linear_0", "relu_0"]},
            ],
        }
        plan = RewritePlan.model_validate(payload)
        raw = json.loads(plan.model_dump_json())
        reloaded = RewritePlan.model_validate(raw)
        assert reloaded.source_profile_id == plan.source_profile_id
        assert len(reloaded.ops) == 1
        assert isinstance(reloaded.ops[0], FuseOp)
