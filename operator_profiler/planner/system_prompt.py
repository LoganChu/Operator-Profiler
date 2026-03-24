"""
Planner system prompt — teaches the LLM to interpret Nsight metrics and
produce valid ``RewritePlan`` JSON.

Design principles
-----------------
1. No hard numerical thresholds.  Thresholds are GPU- and workload-specific.
   The ridge point (peak_compute / peak_bandwidth) is the hardware-correct
   boundary between memory-bound and compute-bound regimes, and it varies
   by an order of magnitude across GPU generations.  The LLM receives the
   actual ridge point for the profiled GPU and model-wide metric percentiles
   so it can reason *relatively* rather than against fixed constants.

2. Context over rules.  The system prompt defines *how* to reason; the user
   message supplies the *data* (profile, graph, GPU specs, candidate rewrites).

Public names
------------
* ``METRIC_RULES`` — qualitative signals per bottleneck class; used by
  the summarizer (``rules.py``) and by tests to assert coverage.
* ``build_system_prompt()`` — full system prompt string (static, no data).
* ``build_gpu_context_section(profile, device_name)`` — data-driven section
  injected into the *user* message, not the system prompt.
"""
from __future__ import annotations

import statistics
from typing import TYPE_CHECKING, TypedDict

if TYPE_CHECKING:
    from operator_profiler.schema.profile import OperatorAttributedProfile


# ---------------------------------------------------------------------------
# Metric rules — qualitative signals, no thresholds
# ---------------------------------------------------------------------------

class MetricRule(TypedDict):
    bottleneck: str
    signals: list[str]
    recommended_ops: list[str]
    op_details: str


METRIC_RULES: list[MetricRule] = [
    {
        "bottleneck": "memory_bound",
        "signals": [
            "arithmetic_intensity is far below the GPU ridge point",
            "achieved_occupancy is low relative to the model median",
            "DRAM bytes transferred are large relative to compute",
            "L1 hit rate is low, indicating poor data reuse",
        ],
        "recommended_ops": ["change_layout", "fuse", "buffer_sharing"],
        "op_details": (
            "change_layout NCHW→NHWC for convolutions (improves spatial locality); "
            "fuse adjacent elementwise ops to eliminate DRAM round-trips; "
            "buffer_sharing to alias tensors whose live ranges do not overlap."
        ),
    },
    {
        "bottleneck": "compute_bound",
        "signals": [
            "arithmetic_intensity is at or above the GPU ridge point",
            "achieved_occupancy is high relative to the model median",
            "tensor_core_active_pct is moderate, suggesting room to improve utilisation",
        ],
        "recommended_ops": ["fuse", "reorder"],
        "op_details": (
            "fuse with strategy=inductor_fuse to let Inductor co-schedule tensor-core ops; "
            "reorder to expose more pipeline parallelism between independent ops."
        ),
    },
    {
        "bottleneck": "latency_bound",
        "signals": [
            "many small kernels per operator (high kernel_count, low SM cycles per kernel)",
            "achieved_occupancy is very low relative to the model median",
            "each kernel's duration is small, suggesting dispatch overhead dominates",
        ],
        "recommended_ops": ["fuse"],
        "op_details": (
            "fuse with strategy=inline for adjacent elementwise ops "
            "(add, relu, gelu, sigmoid, mul) — eliminates per-kernel launch "
            "overhead and dispatcher round-trips."
        ),
    },
]


# ---------------------------------------------------------------------------
# GPU context builder — injected into the USER message
# ---------------------------------------------------------------------------

def build_gpu_context_section(
    profile: "OperatorAttributedProfile",
    device_name: str | None = None,
) -> str:
    """
    Build a data-driven GPU context section for injection into the user message.

    This section provides:
    - The GPU's ridge point (peak_compute / peak_bandwidth), the hardware-
      correct threshold separating memory-bound from compute-bound regimes.
    - Model-wide metric statistics (p25 / median / p75) so the LLM can judge
      each operator *relative to the distribution*, not against a fixed number.

    Parameters
    ----------
    profile:
        The current OperatorAttributedProfile being optimised.
    device_name:
        The device name from CaptureMetadata.  Used to look up GPU specs.
        Falls back to "unknown GPU" if not found.
    """
    from operator_profiler.aggregator.roofline import KNOWN_GPU_SPECS

    # ------------------------------------------------------------------
    # GPU specs lookup — fuzzy substring match
    # ------------------------------------------------------------------
    resolved_name = device_name or ""
    specs: dict[str, float] | None = None
    for key, val in KNOWN_GPU_SPECS.items():
        if key.lower() in resolved_name.lower() or resolved_name.lower() in key.lower():
            specs = val
            resolved_name = key
            break

    if specs is not None:
        peak_compute = specs["peak_compute_gflops"]
        peak_bw = specs["peak_bandwidth_gbs"]
        ridge_point = peak_compute / peak_bw
        hw_section = (
            f"GPU: {resolved_name}\n"
            f"Peak compute: {peak_compute / 1_000:.0f} TFLOP/s (FP16 tensor cores)\n"
            f"Peak bandwidth: {peak_bw:.0f} GB/s\n"
            f"Ridge point: {ridge_point:.1f} FLOP/byte\n"
            f"  → AI < {ridge_point:.1f} → memory-bound regime\n"
            f"  → AI > {ridge_point:.1f} → compute-bound regime"
        )
    else:
        hw_section = (
            f"GPU: {device_name or 'unknown'}\n"
            "Hardware specs not found in database.  "
            "Classify bottlenecks using the model-wide distribution below."
        )

    # ------------------------------------------------------------------
    # Model-wide metric distributions
    # ------------------------------------------------------------------
    ais, occs, tc_pcts, durations_ms = [], [], [], []
    for op in profile.operators:
        if op.aggregated is None:
            continue
        agg = op.aggregated
        durations_ms.append(agg.total_duration_ns / 1e6)
        if agg.mean_achieved_occupancy is not None:
            occs.append(agg.mean_achieved_occupancy)
        if agg.mean_tensor_core_active_pct is not None:
            tc_pcts.append(agg.mean_tensor_core_active_pct)
        for k in op.kernels:
            if k.metrics.arithmetic_intensity is not None:
                ais.append(k.metrics.arithmetic_intensity)

    def _pct(values: list[float], p: int) -> str:
        if not values:
            return "n/a"
        sorted_v = sorted(values)
        idx = max(0, min(len(sorted_v) - 1, int(p / 100 * len(sorted_v))))
        return f"{sorted_v[idx]:.2f}"

    def _med(values: list[float]) -> str:
        if not values:
            return "n/a"
        return f"{statistics.median(values):.2f}"

    total_duration_ms = sum(durations_ms) if durations_ms else 0.0
    worst_op = max(
        (op for op in profile.operators if op.aggregated),
        key=lambda op: op.aggregated.total_duration_ns,
        default=None,
    )

    stats_lines = [
        f"Total operators: {len(profile.operators)}",
        f"Total model duration: {total_duration_ms:.3f} ms",
        "",
        "Arithmetic intensity (FLOP/byte) across all kernels:",
        f"  p25={_pct(ais, 25)}  median={_med(ais)}  p75={_pct(ais, 75)}",
        "Achieved occupancy across operators:",
        f"  p25={_pct(occs, 25)}  median={_med(occs)}  p75={_pct(occs, 75)}",
        "Tensor core active % across operators:",
        f"  p25={_pct(tc_pcts, 25)}  median={_med(tc_pcts)}  p75={_pct(tc_pcts, 75)}",
    ]

    if worst_op and worst_op.aggregated:
        pct_of_total = (
            worst_op.aggregated.total_duration_ns / 1e6 / total_duration_ms * 100
            if total_duration_ms > 0 else 0.0
        )
        stats_lines += [
            "",
            f"Worst operator: {worst_op.operator_id}",
            f"  Duration: {worst_op.aggregated.total_duration_ns / 1e6:.3f} ms "
            f"({pct_of_total:.1f}% of total)",
            f"  Classification: {worst_op.aggregated.bottleneck_classification}",
            f"  Kernels: {worst_op.aggregated.kernel_count}",
        ]
        # Per-kernel AI for worst op
        worst_ais = [
            k.metrics.arithmetic_intensity
            for k in worst_op.kernels
            if k.metrics.arithmetic_intensity is not None
        ]
        if worst_ais and specs is not None:
            mean_ai = sum(worst_ais) / len(worst_ais)
            ridge_point = specs["peak_compute_gflops"] / specs["peak_bandwidth_gbs"]
            pct_of_ridge = mean_ai / ridge_point * 100
            stats_lines.append(
                f"  Mean AI: {mean_ai:.2f} FLOP/byte "
                f"({pct_of_ridge:.1f}% of GPU ridge point)"
            )

    return (
        "## Hardware & Profile Context\n\n"
        "### GPU Specifications\n"
        + hw_section
        + "\n\n### Model-Wide Metric Distribution\n"
        + "\n".join(stats_lines)
    )


# ---------------------------------------------------------------------------
# DSL reference
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
  GPU ridge point: 156 FLOP/byte (A100).
  Worst kernel AI=8.2 FLOP/byte (5.3% of ridge), achieved_occupancy=38%
  (model median occupancy=65%), l1_hit_rate=41%.
FX graph nodes: conv2d_1, batch_norm_1, relu_1

Reasoning: AI is 5% of ridge — deep in memory-bound regime.
Occupancy is well below model median (38% vs 65%). Two DRAM round-trips
can be eliminated by fusing BN+ReLU. Layout change to NHWC improves
spatial data reuse for the conv.

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
  GPU ridge point: 156 FLOP/byte (A100).
  Worst kernel AI=0.3 FLOP/byte (0.2% of ridge), achieved_occupancy=18%
  (model median=62%), kernel_count=4. Each kernel is tiny.
FX graph nodes: add_1, relu_1, mul_1, sigmoid_1

Reasoning: 4 separate kernel launches for 4 elementwise ops. Each one
is far below the memory bandwidth limit and barely uses the GPU at all
— occupancy (18%) is far below model median (62%). This is pure launch
overhead. Fusing all four inline eliminates 3 kernel dispatches.

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

    This prompt contains *instructions* only — no profile data, no GPU
    numbers, no operator names.  All data-specific context (ridge point,
    metric percentiles, candidate rewrites) is injected by
    ``build_gpu_context_section()`` into the *user* message.

    Sections
    --------
    1. Role definition
    2. How to classify bottlenecks (qualitative, relative reasoning — no
       hard thresholds)
    3. DSL reference
    4. Output format instructions
    5. Few-shot examples (with contextual reasoning shown explicitly)
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

## How to Classify Bottlenecks

You will be given GPU hardware specifications including the **ridge point** (peak_compute / peak_bandwidth in FLOP/byte) and **model-wide metric percentiles**.  Use these to reason *relatively*:

- An operator is **memory-bound** when its arithmetic intensity is far below the ridge point.
- An operator is **compute-bound** when its arithmetic intensity is near or above the ridge point.
- An operator is **latency-bound** when it launches many small kernels with occupancy far below the model median.

Do NOT apply fixed numerical thresholds.  What constitutes "low occupancy" depends on the model's own distribution and the GPU's characteristics.  An operator at 40% occupancy may be fine if the model median is 35%, or alarming if the median is 75%.

## Bottleneck Patterns and Recommended Actions
{bottleneck_section}

If metrics are absent (null), emit an empty plan (no ops) and set description to "Insufficient metrics; no rewrite generated."
{_DSL_REFERENCE}
## Output Format

Respond with **only** a single valid JSON object.  No markdown fences, no explanation outside the JSON.

Required fields:
- plan_version: exactly "1.0"
- source_profile_id: "<schema_version>/<operator_id of worst-bottleneck operator>"
- description: one sentence that names the bottleneck type, cites relative evidence
  (e.g., "AI is 3% of ridge point"), and states the chosen rewrite
- ops: list of op objects (may be empty)

Prioritise rewrites for the operator with the longest total_duration_ns.
Prefer a single well-targeted op over a long list of speculative ops.
{_FEW_SHOT_EXAMPLES}"""
