"""
JSON Rewrite Plan DSL — Pydantic v2 models for the four op types.

The discriminated union ``AnyRewriteOp`` routes on the ``"op"`` field so that
``RewritePlan.ops`` can hold a heterogeneous list with full validation.
"""
from __future__ import annotations

from typing import Annotated, Literal, Union

from pydantic import BaseModel, Field, model_validator

DSL_VERSION = "1.0"

LAYOUT_FORMATS = Literal["NCHW", "NHWC", "NCL", "NLC", "NCDHW", "NDHWC"]


# ---------------------------------------------------------------------------
# Errors
# ---------------------------------------------------------------------------

class RewriteValidationError(ValueError):
    """Raised when a rewrite op fails pre-flight or runtime constraint checks."""


# ---------------------------------------------------------------------------
# Op types
# ---------------------------------------------------------------------------

class FuseOp(BaseModel):
    op: Literal["fuse"]
    id: str
    nodes: Annotated[list[str], Field(min_length=2)]
    strategy: Literal["inline", "custom_op", "inductor_fuse"] = "inductor_fuse"
    custom_op_name: str | None = None
    comment: str | None = None

    @model_validator(mode="after")
    def _check_custom_op_name(self) -> "FuseOp":
        if self.strategy == "custom_op" and not self.custom_op_name:
            raise ValueError("custom_op_name is required when strategy='custom_op'")
        if self.strategy != "custom_op" and self.custom_op_name is not None:
            raise ValueError(
                "custom_op_name must be null when strategy != 'custom_op'"
            )
        return self


class ReorderOp(BaseModel):
    op: Literal["reorder"]
    id: str
    node: str
    before: str | None = None
    after: str | None = None

    @model_validator(mode="after")
    def _check_before_after(self) -> "ReorderOp":
        if (self.before is None) == (self.after is None):
            raise ValueError(
                "Exactly one of 'before' or 'after' must be set (not both, not neither)"
            )
        return self


class ChangeLayoutOp(BaseModel):
    op: Literal["change_layout"]
    id: str
    target_node: str
    current_format: LAYOUT_FORMATS
    target_format: LAYOUT_FORMATS
    insert_contiguous_after: bool = True

    @model_validator(mode="after")
    def _check_formats_differ(self) -> "ChangeLayoutOp":
        if self.current_format == self.target_format:
            raise ValueError("current_format and target_format must differ")
        return self


class BufferSharingOp(BaseModel):
    op: Literal["buffer_sharing"]
    id: str
    source_node: str
    target_node: str
    validate_liveness: bool = True


# ---------------------------------------------------------------------------
# Discriminated union + top-level plan
# ---------------------------------------------------------------------------

AnyRewriteOp = Annotated[
    Union[FuseOp, ReorderOp, ChangeLayoutOp, BufferSharingOp],
    Field(discriminator="op"),
]


class RewritePlan(BaseModel):
    plan_version: str = DSL_VERSION
    source_profile_id: str | None = None
    description: str | None = None
    ops: list[AnyRewriteOp] = Field(default_factory=list)
