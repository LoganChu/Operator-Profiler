"""
Planner system prompt — teaches the LLM to interpret Nsight metrics and
produce valid ``RewritePlan`` JSON.

The module exposes two public names:

* ``METRIC_RULES`` — a structured constant used by tests to assert that
  all three bottleneck categories are represented.
* ``build_system_prompt()`` — returns the full multi-section system prompt
  string passed to the LLM on every ``ThetaPlanner.plan()`` call.
"""
from __future__ import annotations

from typing import TypedDict


# ---------------------------------------------------------------------------
# Metric rules (structured constant — also used by tests)
# ---------------------------------------------------------------------------

class MetricRule(TypedDict):
    bottleneck: str
    signals: list[str]
    thresholds: dict[str, object]
    recommended_ops: list[str]
    op_details: str


METRIC_RULES: list[MetricRule] = [
    {
        "bottleneck": "memory_bound",
        "signals": [
            "arithmetic_intensity < 30 FLOP/byte",
            "achieved_occupancy < 50%",
            "dram_bytes_read + dram_bytes_written is large",
            "l1_hit_rate < 60%",
        ],
        "thresholds": {
            "arithmetic_intensity_lt": 30.0,
            "achieved_occupancy_lt": 50.0,
            "l1_hit_rate_lt": 60.0,
        },
        "recommended_ops": ["change_layout", "fuse", "buffer_sharing"],
        "op_details": (
            "change_layout NCHW→NHWC for convolutions (improves L2 reuse); "
            "fuse adjacent elementwise ops to eliminate DRAM round-trips; "
            "buffer_sharing to alias tensors whose live ranges do not overlap."
        ),
    },
    {
        "bottleneck": "compute_bound",
        "signals": [
            "achieved_occupancy > 70%",
            "tensor_core_active_pct > 50%",
            "sm_throughput near peak",
        ],
        "thresholds": {
            "achieved_occupancy_gt": 70.0,
            "tensor_core_active_pct_gt": 50.0,
        },
        "recommended_ops": ["fuse", "reorder"],
        "op_details": (
            "fuse with strategy=inductor_fuse to reduce kernel launch overhead "
            "and let Inductor schedule tensor-core ops together; "
            "reorder to expose more pipeline parallelism."
        ),
    },
    {
        "bottleneck": "latency_bound",
        "signals": [
            "achieved_occupancy < 30%",
            "kernel_count > 5 per operator",
            "sm_active_cycles per kernel is small",
        ],
        "thresholds": {
            "achieved_occupancy_lt": 30.0,
            "kernel_count_gt": 5,
        },
        "recommended_ops": ["fuse"],
        "op_details": (
            "fuse with strategy=inline for adjacent elementwise ops "
            "(add, relu, gelu, sigmoid, mul) — eliminates per-kernel launch "
            "overhead and dispatcher round-trips for tiny wavefronts."
        ),
    },
]


# ---------------------------------------------------------------------------
# DSL reference (embedded verbatim in the prompt)
# ---------------------------------------------------------------------------

_DSL_REFERENCE = """
## RewritePlan DSL Reference

A RewritePlan is a JSON object with these top-level fields:
  plan_version   : "1.0"  (must match exactly)
  source_profile_id : string (set to "<schema_version>/<worst_operator_id>")
  description    : one sentence explaining your reasoning
  ops            : list of op objects (may be empty)

### Op types (discriminated by the "op" field)

#### fuse
{
  "op": "fuse",
  "id": "<unique string>",
  "nodes": ["<node_a>", "<node_b>", ...],      // min 2 node names from FX graph
  "strategy": "inline" | "custom_op" | "inductor_fuse",  // default: inductor_fuse
  "custom_op_name": null,   // required only when strategy="custom_op"
  "comment": null           // optional human note
}

#### reorder
{
  "op": "reorder",
  "id": "<unique string>",
  "node": "<node_to_move>",
  "before": "<anchor_node>",  // set EXACTLY ONE of before/after
  "after": null
}

#### change_layout
{
  "op": "change_layout",
  "id": "<unique string>",
  "target_node": "<conv_node>",
  "current_format": "NCHW",     // one of: NCHW NHWC NCL NLC NCDHW NDHWC
  "target_format": "NHWC",
  "insert_contiguous_after": true
}

#### buffer_sharing
{
  "op": "buffer_sharing",
  "id": "<unique string>",
  "source_node": "<node_a>",
  "target_node": "<node_b>",
  "validate_liveness": true
}

### Important constraints
- All node names must exist verbatim in the FX graph printed below.
- For reorder: you may not move a node before one of its own upstream
  dependencies, nor after one of its downstream consumers.
- For change_layout: target_node must be a convolution op
  (conv1d, conv2d, conv3d, or aten::_convolution).
- For buffer_sharing: source and target live ranges must not overlap
  (the executor will reject the plan if they do).
- An empty ops list is a valid plan (no rewrite, baseline is best).
"""


# ---------------------------------------------------------------------------
# Few-shot examples
# ---------------------------------------------------------------------------

_FEW_SHOT_EXAMPLES = """
## Examples

### Example 1 — Memory-bound convolution block
Profile: conv2d → batch_norm → relu, all memory_bound.
  Worst metric: arithmetic_intensity=8.2, achieved_occupancy=38%, l1_hit_rate=41%
FX graph nodes: conv2d_1, batch_norm_1, relu_1

Correct output:
{
  "plan_version": "1.0",
  "source_profile_id": "1.0/aten::conv2d_0",
  "description": "Convert conv2d to NHWC and fuse BN+ReLU to eliminate two DRAM round-trips.",
  "ops": [
    {
      "op": "change_layout",
      "id": "cl_0",
      "target_node": "conv2d_1",
      "current_format": "NCHW",
      "target_format": "NHWC",
      "insert_contiguous_after": true
    },
    {
      "op": "fuse",
      "id": "fuse_0",
      "nodes": ["batch_norm_1", "relu_1"],
      "strategy": "inductor_fuse",
      "custom_op_name": null,
      "comment": "Fuse BN+ReLU to avoid a DRAM read-write for the intermediate activation."
    }
  ]
}

### Example 2 — Latency-bound elementwise chain
Profile: add → relu → mul → sigmoid, all latency_bound.
  Worst metric: achieved_occupancy=18%, kernel_count=4
FX graph nodes: add_1, relu_1, mul_1, sigmoid_1

Correct output:
{
  "plan_version": "1.0",
  "source_profile_id": "1.0/aten::add_0",
  "description": "Fuse all four elementwise ops inline to eliminate kernel launch overhead.",
  "ops": [
    {
      "op": "fuse",
      "id": "fuse_0",
      "nodes": ["add_1", "relu_1", "mul_1", "sigmoid_1"],
      "strategy": "inline",
      "custom_op_name": null,
      "comment": null
    }
  ]
}
"""


# ---------------------------------------------------------------------------
# Public builder
# ---------------------------------------------------------------------------

def build_system_prompt() -> str:
    """
    Return the full system prompt passed to the Planner LLM.

    Sections
    --------
    1. Role definition
    2. Bottleneck taxonomy with metric thresholds
    3. DSL reference (all four op types with JSON examples)
    4. Output format instructions
    5. Few-shot examples
    """
    bottleneck_section = "\n".join(
        f"""
### {rule['bottleneck'].replace('_', ' ').title()}
Signals: {', '.join(rule['signals'])}
Recommended ops: {', '.join(rule['recommended_ops'])}
Details: {rule['op_details']}"""
        for rule in METRIC_RULES
    )

    return f"""\
You are an expert GPU kernel optimization planner for PyTorch models compiled with torch.compile (Inductor backend).

Your task is to analyse an Operator-Attributed Profile (JSON) and a torch.fx GraphModule (printed as readable code), then produce a **RewritePlan** JSON that will maximally reduce GPU execution time.

## Bottleneck Taxonomy

Use Nsight Compute (NCU) metrics to classify each operator before choosing a rewrite:
{bottleneck_section}

If metrics are absent (null), emit an empty plan (no ops) and set description to "Insufficient metrics; no rewrite generated."
{_DSL_REFERENCE}
## Output Format

Respond with **only** a single valid JSON object.  No markdown fences, no explanation outside the JSON.

Required fields:
- plan_version: exactly "1.0"
- source_profile_id: "<schema_version>/<operator_id of worst-bottleneck operator>"
- description: one sentence explaining which bottleneck you are targeting and why
- ops: list of op objects (may be empty)

Prioritise rewrites for the operator with the longest total_duration_ns.
Prefer a single well-targeted op over a long list of speculative ops.
{_FEW_SHOT_EXAMPLES}"""
