"""
demo_pipeline_report.py — Generate a realistic pipeline walkthrough report.

Simulates a full operator-profiler optimization run on a mini transformer
feed-forward block (Linear → ReLU → Linear → GELU) with realistic Nsight
metrics and prints a Markdown summary to stdout.

No GPU, no Anthropic API key required.  The profiler_fn and ThetaPlanner
are mocked; everything else uses real code paths.

Usage:
    conda run -n ml_env python scripts/demo_pipeline_report.py
    conda run -n ml_env python scripts/demo_pipeline_report.py > report.md
"""
from __future__ import annotations

import sys
import json
from pathlib import Path
from unittest.mock import MagicMock

import torch
import torch.nn as nn
import torch.fx

# Make sure operator_profiler is importable when run from the repo root
sys.path.insert(0, str(Path(__file__).parent.parent))

from operator_profiler.planner.loop import LoopConfig, LoopResult, OptimizationLoop
from operator_profiler.planner.memory import OptimizationMemory
from operator_profiler.planner.search import BeamSearch
from operator_profiler.rewriter.dsl import FuseOp, RewritePlan
from operator_profiler.rewriter.executor import ExecutorConfig, HybridExecutor
from operator_profiler.schema.profile import (
    AggregatedMetrics,
    CaptureMetadata,
    KernelMetrics,
    KernelRecord,
    OperatorAttributedProfile,
    OperatorRecord,
)
from operator_profiler.summarizer import (
    SummaryReport,
    build_provenance_rows,
    compute_diff,
    entries_to_rules,
    explain_node,
    render_markdown,
    render_provenance_text,
)


# ---------------------------------------------------------------------------
# 1. Model
# ---------------------------------------------------------------------------

class FFBlock(nn.Module):
    """Mini transformer feed-forward block: Linear→ReLU→Linear→GELU."""
    def __init__(self):
        super().__init__()
        self.fc1 = nn.Linear(256, 512)
        self.fc2 = nn.Linear(512, 256)

    def forward(self, x):
        y = self.fc1(x)
        y = torch.relu(y)
        y = self.fc2(y)
        y = torch.nn.functional.gelu(y)
        return y


# ---------------------------------------------------------------------------
# 2. Realistic before-profile (RTX 5090, memory_bound + latency_bound)
# ---------------------------------------------------------------------------
# RTX 5090: peak_compute = 838,600 GFLOP/s (FP16 dense),  peak_bw = 1,792 GB/s
# Ridge point = 838_600 / 1_792 ≈ 468 FLOP/byte
# FC layers at AI≈3-4 FLOP/byte are extremely deep in the memory-bound regime.

def _kernel(kid, dur_ns, dram_read_kb, dram_write_kb, occupancy, ai):
    return KernelRecord(
        kernel_id=kid,
        kernel_name=f"ampere_sgemm_{kid}",
        demangled_name=f"void ampere_sgemm<{kid}>(...)",
        stream_id=7,
        device_id=0,
        start_ns=0,
        end_ns=dur_ns,
        duration_ns=dur_ns,
        metrics=KernelMetrics(
            dram_bytes_read=dram_read_kb * 1024,
            dram_bytes_written=dram_write_kb * 1024,
            achieved_occupancy=occupancy,
            arithmetic_intensity=ai,
            tensor_core_active_pct=35.0 if ai > 1.0 else 5.0,
        ),
    )


def make_before_profile() -> OperatorAttributedProfile:
    return OperatorAttributedProfile(
        capture_metadata=CaptureMetadata(
            model_name="FFBlock (Linear256→512→256)",
            torch_version="2.10.0+cu130",
            cuda_version="12.8",
            compile_mode="inductor",
            capture_timestamp_utc="2026-03-22T09:15:00+00:00",
            device_name="RTX 5090",
        ),
        operators=[
            OperatorRecord(
                operator_id="aten::linear_0",
                operator_name="aten::linear",
                call_index=0,
                kernels=[
                    _kernel("fc1_fwd", 5_200_000, dram_read_kb=512, dram_write_kb=256,
                            occupancy=0.31, ai=3.2),
                ],
                aggregated=AggregatedMetrics(
                    total_duration_ns=5_200_000,
                    kernel_count=1,
                    total_dram_bytes_read=512 * 1024,
                    total_dram_bytes_written=256 * 1024,
                    mean_achieved_occupancy=0.31,
                    mean_tensor_core_active_pct=35.0,
                    bottleneck_classification="memory_bound",
                ),
            ),
            OperatorRecord(
                operator_id="aten::relu_0",
                operator_name="aten::relu",
                call_index=0,
                kernels=[
                    _kernel("relu_fwd", 420_000, dram_read_kb=256, dram_write_kb=256,
                            occupancy=0.18, ai=0.25),
                ],
                aggregated=AggregatedMetrics(
                    total_duration_ns=420_000,
                    kernel_count=1,
                    total_dram_bytes_read=256 * 1024,
                    total_dram_bytes_written=256 * 1024,
                    mean_achieved_occupancy=0.18,
                    mean_tensor_core_active_pct=4.0,
                    bottleneck_classification="latency_bound",
                ),
            ),
            OperatorRecord(
                operator_id="aten::linear_1",
                operator_name="aten::linear",
                call_index=1,
                kernels=[
                    _kernel("fc2_fwd", 3_800_000, dram_read_kb=768, dram_write_kb=128,
                            occupancy=0.29, ai=3.8),
                ],
                aggregated=AggregatedMetrics(
                    total_duration_ns=3_800_000,
                    kernel_count=1,
                    total_dram_bytes_read=768 * 1024,
                    total_dram_bytes_written=128 * 1024,
                    mean_achieved_occupancy=0.29,
                    mean_tensor_core_active_pct=33.0,
                    bottleneck_classification="memory_bound",
                ),
            ),
            OperatorRecord(
                operator_id="aten::gelu_0",
                operator_name="aten::gelu",
                call_index=0,
                kernels=[
                    _kernel("gelu_fwd", 510_000, dram_read_kb=128, dram_write_kb=128,
                            occupancy=0.14, ai=0.31),
                ],
                aggregated=AggregatedMetrics(
                    total_duration_ns=510_000,
                    kernel_count=1,
                    total_dram_bytes_read=128 * 1024,
                    total_dram_bytes_written=128 * 1024,
                    mean_achieved_occupancy=0.14,
                    mean_tensor_core_active_pct=3.0,
                    bottleneck_classification="latency_bound",
                ),
            ),
        ],
    )


# ---------------------------------------------------------------------------
# 3. After-profile — fused elementwise ops eliminate 2 DRAM round-trips
# ---------------------------------------------------------------------------

def make_after_profile() -> OperatorAttributedProfile:
    """
    Post-optimization profile after fusing ReLU into fc1 and GELU into fc2.
    The fused kernels avoid writing/reading intermediates back to DRAM, so
    total duration drops ~1.55×.
    """
    return OperatorAttributedProfile(
        capture_metadata=CaptureMetadata(
            model_name="FFBlock (Linear256→512→256)",
            torch_version="2.10.0+cu130",
            cuda_version="12.8",
            compile_mode="inductor",
            capture_timestamp_utc="2026-03-22T09:15:30+00:00",
            device_name="RTX 5090",
        ),
        operators=[
            OperatorRecord(
                operator_id="aten::linear_0",
                operator_name="aten::linear",
                call_index=0,
                is_fused=True,
                fused_with=["aten::linear_0", "aten::relu_0"],
                aggregated=AggregatedMetrics(
                    total_duration_ns=3_500_000,   # 5.2ms → 3.5ms
                    kernel_count=1,
                    mean_achieved_occupancy=0.54,  # occupancy improved post-fusion
                    mean_tensor_core_active_pct=61.0,
                    bottleneck_classification="compute_bound",
                ),
            ),
            OperatorRecord(
                operator_id="aten::linear_1",
                operator_name="aten::linear",
                call_index=1,
                is_fused=True,
                fused_with=["aten::linear_1", "aten::gelu_0"],
                aggregated=AggregatedMetrics(
                    total_duration_ns=2_600_000,   # 3.8ms → 2.6ms
                    kernel_count=1,
                    mean_achieved_occupancy=0.51,
                    mean_tensor_core_active_pct=58.0,
                    bottleneck_classification="compute_bound",
                ),
            ),
        ],
    )


# ---------------------------------------------------------------------------
# 4. RewritePlan the planner would generate
# ---------------------------------------------------------------------------

def make_fuse_plan(node_names: list[str]) -> RewritePlan:
    """
    The planner identifies two latency-bound elementwise ops (relu, gelu)
    and fuses them with their upstream linear layers.
    """
    # Find actual relu and gelu node names in the FX graph
    relu_nodes = [n for n in node_names if "relu" in n]
    gelu_nodes = [n for n in node_names if "gelu" in n]
    linear_nodes = [n for n in node_names if "fc" in n or "linear" in n or "addmm" in n or "linear" in n.lower()]

    ops = []

    # Fuse fc1 + relu
    if len(linear_nodes) >= 1 and relu_nodes:
        ops.append(FuseOp(
            op="fuse",
            id="fuse_fc1_relu",
            nodes=[linear_nodes[0], relu_nodes[0]],
            strategy="inductor_fuse",
            comment="Fuse fc1+relu: relu is latency_bound (AI=0.25, 0.05% of ridge), "
                    "occupancy 18% vs model median 24%. Fusing eliminates the 256KB "
                    "DRAM round-trip for the intermediate activation.",
        ))

    # Fuse fc2 + gelu
    if len(linear_nodes) >= 2 and gelu_nodes:
        ops.append(FuseOp(
            op="fuse",
            id="fuse_fc2_gelu",
            nodes=[linear_nodes[1], gelu_nodes[0]],
            strategy="inductor_fuse",
            comment="Fuse fc2+gelu: gelu is latency_bound (AI=0.31, 0.07% of ridge), "
                    "occupancy 14% vs model median 24%. Fusing eliminates the 128KB "
                    "DRAM round-trip for the gelu intermediate.",
        ))

    # Fallback: if we couldn't identify nodes, fuse all computation nodes
    if not ops and len(node_names) >= 2:
        ops.append(FuseOp(
            op="fuse",
            id="fuse_all",
            nodes=node_names,
            strategy="inductor_fuse",
        ))

    return RewritePlan(
        plan_version="1.0",
        source_profile_id="1.0/aten::linear_0",
        description=(
            "Fuse relu into fc1 and gelu into fc2 using inductor_fuse: both activation "
            "ops are latency_bound (AI far below RTX 5090 ridge of 468 FLOP/byte), "
            "eliminating 2 separate kernel launches and ~384 KB of unnecessary DRAM traffic."
        ),
        ops=ops,
    )


# ---------------------------------------------------------------------------
# 5. Wire up the loop with mocked planner + profiler
# ---------------------------------------------------------------------------

def run_pipeline(tmp_path: Path):
    # Trace the model to get real FX node names
    gm = torch.fx.symbolic_trace(FFBlock())
    node_names = [
        n.name for n in gm.graph.nodes
        if n.op not in ("placeholder", "output", "get_attr")
    ]

    before = make_before_profile()
    after  = make_after_profile()
    fuse_plan = make_fuse_plan(node_names)

    # Mock planner: always returns the pre-built fuse plan
    mock_planner = MagicMock()
    mock_planner.plan.return_value = fuse_plan
    mock_planner.rank_candidates.side_effect = lambda p, c, **kw: c

    # Mock profiler_fn: returns the faster after profile
    profiler_fn = MagicMock(return_value=after)

    mem = OptimizationMemory(tmp_path / "demo_memory.json")
    search = BeamSearch(width=3, seed=0)
    cfg = LoopConfig(
        n_iterations=3,
        beam_width=3,
        speedup_threshold=1.05,
        executor_config=ExecutorConfig(skip_verification=True),
    )

    loop = OptimizationLoop(
        planner=mock_planner,
        memory=mem,
        search=search,
        profiler_fn=profiler_fn,
        config=cfg,
    )

    loop_result = loop.run(gm, before, [torch.randn(8, 256)])

    # Apply the best plan
    executor = HybridExecutor(gm, loop_result.best_plan or RewritePlan(),
                              ExecutorConfig(skip_verification=True))
    result_gm, _ = executor.run()

    # Compute diff
    diff = compute_diff(before, after, loop_result.best_plan)

    # Optimization rules from curated memory
    rules = entries_to_rules(mem.entries)
    lessons = [r.rule_text for r in sorted(rules, key=lambda r: r.speedup, reverse=True)[:5]]

    report = SummaryReport(
        diff=diff,
        rules=rules,
        lessons_learned=lessons,
        loop_history=loop_result.history,
        best_speedup=loop_result.best_speedup,
        best_plan_description=fuse_plan.description,
    )

    return loop_result, diff, before, after, report, mem, node_names, fuse_plan


# ---------------------------------------------------------------------------
# 6. Print the full report
# ---------------------------------------------------------------------------

def main():
    import tempfile
    tmp = Path(tempfile.mkdtemp())

    print("Running pipeline...", file=sys.stderr)
    loop_result, diff, before, after, report, mem, node_names, fuse_plan = run_pipeline(tmp)
    rules = entries_to_rules(mem.entries)

    # ------------------------------------------------------------------
    # Section 0: Pipeline run banner
    # ------------------------------------------------------------------
    print("=" * 72)
    print("  OPERATOR PROFILER — DEMO PIPELINE REPORT")
    print("  Model: FFBlock (mini transformer feed-forward block)")
    print("  GPU:   RTX 5090  |  Ridge point: 468 FLOP/byte")
    print("=" * 72)
    print()

    # ------------------------------------------------------------------
    # Section 1: FX Graph
    # ------------------------------------------------------------------
    print("## FX Graph (torch.fx.symbolic_trace)")
    print()
    print(f"  Computation nodes: {node_names}")
    print()

    # ------------------------------------------------------------------
    # Section 2: Before Profile
    # ------------------------------------------------------------------
    print("## Before Profile — Operator-Level Metrics")
    print()
    total_before_ms = sum(op.aggregated.total_duration_ns for op in before.operators) / 1e6
    print(f"  Total execution time: {total_before_ms:.3f} ms across {len(before.operators)} operators")
    print()
    print(f"  {'Operator':<30} {'Duration':>10} {'Bottleneck':<16} {'Occupancy':>10} {'AI (FLOP/B)':>12}")
    print(f"  {'-'*30} {'-'*10} {'-'*16} {'-'*10} {'-'*12}")
    for op in before.operators:
        agg = op.aggregated
        k = op.kernels[0].metrics if op.kernels else None
        occ_str = f"{agg.mean_achieved_occupancy:.0%}" if agg.mean_achieved_occupancy else "n/a"
        ai_str  = f"{k.arithmetic_intensity:.2f}" if k and k.arithmetic_intensity else "n/a"
        ridge_pct = ""
        if k and k.arithmetic_intensity:
            pct = k.arithmetic_intensity / 468.0 * 100
            ai_str += f" ({pct:.1f}% of ridge)"
        print(f"  {op.operator_id:<30} {agg.total_duration_ns/1e6:>9.3f}ms "
              f"  {agg.bottleneck_classification:<16} {occ_str:>10} {ai_str:>20}")
    print()

    # ------------------------------------------------------------------
    # Section 3: Planner output
    # ------------------------------------------------------------------
    print("## Planner Output — RewritePlan")
    print()
    print(f"  Plan version: {fuse_plan.plan_version}")
    print(f"  Description:  {fuse_plan.description}")
    print(f"  Ops ({len(fuse_plan.ops)}):")
    for op in fuse_plan.ops:
        print(f"    [{op.id}]  op={op.op}  nodes={op.nodes}  strategy={op.strategy}")
        if hasattr(op, 'comment') and op.comment:
            for line in op.comment.split('. '):
                if line.strip():
                    print(f"      → {line.strip()}.")
    print()

    # ------------------------------------------------------------------
    # Section 4: Optimization loop history
    # ------------------------------------------------------------------
    print("## Optimization Loop — Iteration History")
    print()
    print(f"  {'Iter':>4}  {'Bottleneck':<16}  {'Memory Hits':>11}  {'Plans Tried':>11}  {'Best Speedup':>12}")
    print(f"  {'-'*4}  {'-'*16}  {'-'*11}  {'-'*11}  {'-'*12}")
    for h in loop_result.history:
        print(f"  {h['iteration']:>4}  {h['bottleneck']:<16}  {h['memory_hits']:>11}  "
              f"{h['plans_tried']:>11}  {h['best_speedup_so_far']:>11.3f}×")
    print()
    print(f"  Final best speedup: {loop_result.best_speedup:.3f}×")
    print(f"  Memory entries curated: {len(mem.entries)}")
    print()

    # ------------------------------------------------------------------
    # Section 5: After Profile & Diff
    # ------------------------------------------------------------------
    print("## After Profile — Post-Optimization Metrics")
    print()
    total_after_ms = sum(
        op.aggregated.total_duration_ns for op in after.operators
        if op.aggregated
    ) / 1e6
    print(f"  Total execution time: {total_after_ms:.3f} ms across {len(after.operators)} operators")
    print()
    print(f"  {'Operator':<30} {'Duration':>10} {'Bottleneck':<16} {'Occupancy':>10}  {'Fused With'}")
    print(f"  {'-'*30} {'-'*10} {'-'*16} {'-'*10}  {'-'*30}")
    for op in after.operators:
        agg = op.aggregated
        occ_str = f"{agg.mean_achieved_occupancy:.0%}" if agg.mean_achieved_occupancy else "n/a"
        fused = ", ".join(op.fused_with) if op.fused_with else "—"
        print(f"  {op.operator_id:<30} {agg.total_duration_ns/1e6:>9.3f}ms  "
              f"{agg.bottleneck_classification:<16} {occ_str:>10}  {fused}")
    print()

    print("## Profile Diff Summary")
    print()
    print(f"  Total speedup:      {diff.total_speedup:.3f}×")
    print(f"  Wall time saved:    {diff.wall_time_saved_ns/1e6:.3f} ms  "
          f"({total_before_ms:.3f} ms → {total_after_ms:.3f} ms)")
    print()
    print(f"  {'Operator':<30} {'Before':>8} {'After':>8} {'Speedup':>8}  Match Type")
    print(f"  {'-'*30} {'-'*8} {'-'*8} {'-'*8}  {'-'*12}")
    for d in diff.operator_diffs:
        before_ms = f"{d.duration_before_ns/1e6:.3f}" if d.duration_before_ns else "—"
        after_ms  = f"{d.duration_after_ns/1e6:.3f}"  if d.duration_after_ns  else "—"
        spd_str   = f"{d.speedup:.2f}×" if d.speedup else "—"
        print(f"  {d.operator_id_before:<30} {before_ms:>8} {after_ms:>8} {spd_str:>8}  {d.match_type}")
    print()

    # ------------------------------------------------------------------
    # Section 6: Provenance table
    # ------------------------------------------------------------------
    print("## Provenance Table (Before Profile)")
    print()
    rows = build_provenance_rows(before, fuse_plan)
    prov_text = render_provenance_text(rows)
    for line in prov_text.splitlines():
        print("  " + line)
    print()

    # ------------------------------------------------------------------
    # Section 7: explain_node
    # ------------------------------------------------------------------
    print("## Node Explanations")
    print()
    for node_id in ["aten::linear_0", "aten::relu_0"]:
        explanation = explain_node(node_id, diff, before, loop_result)
        print(f"  ### {node_id}")
        for line in explanation.splitlines():
            print("  " + line)
        print()

    # ------------------------------------------------------------------
    # Section 8: Optimization Rules from Memory
    # ------------------------------------------------------------------
    print("## Optimization Rules Learned")
    print()
    if rules:
        for rule in sorted(rules, key=lambda r: r.speedup, reverse=True):
            print(f"  [{rule.bottleneck}]  {rule.rule_text}")
            print(f"    Conditions: {', '.join(rule.conditions)}")
            print(f"    Action:     {rule.recommended_action}")
            print()
    else:
        print("  (No rules learned — speedup below curation threshold)")
    print()

    # ------------------------------------------------------------------
    # Section 9: Full Markdown Report
    # ------------------------------------------------------------------
    print("=" * 72)
    print("  FULL MARKDOWN REPORT (render_markdown)")
    print("=" * 72)
    print()
    md = render_markdown(report)
    print(md)

    # ------------------------------------------------------------------
    # Section 10: LoopResult serialization demo
    # ------------------------------------------------------------------
    print("=" * 72)
    print("  LOOPRESULT SERIALIZATION")
    print("=" * 72)
    print()
    d = loop_result.to_dict()
    restored = LoopResult.from_dict(d)
    print(f"  Original  best_speedup: {loop_result.best_speedup:.4f}×")
    print(f"  Restored  best_speedup: {restored.best_speedup:.4f}×")
    print(f"  JSON size: {len(json.dumps(d))} bytes")
    print(f"  History entries: {len(restored.history)}")
    print()
    print("  Serialized LoopResult (truncated):")
    raw = json.dumps(d, indent=2)
    for line in raw.splitlines()[:30]:
        print("    " + line)
    if len(raw.splitlines()) > 30:
        print(f"    ... ({len(raw.splitlines()) - 30} more lines)")
    print()

    print("Pipeline demo complete.", file=sys.stderr)


if __name__ == "__main__":
    main()
