"""
lower_to_inductor — compiles a rewritten ``GraphModule`` to Triton/CUDA kernels
via ``torch.compile`` and emits a provenance JSONL sidecar for the Mapper.

This is a **post-rewrite handoff step**, called once after
``HybridExecutor.run()`` returns the final verified ``GraphModule``.  It is
not part of the rewrite-and-verify loop.

``fullgraph=True`` (default) prevents torch.compile from graph-breaking on
unrecognized ops — the call raises immediately if Inductor cannot handle an op,
surfacing the problem before the profiling run.

The returned ``LoweringResult.compiled_model`` is a Python callable, not a
``GraphModule``.  All rewriting must be complete before calling this function.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import Callable

import torch
import torch.fx

from operator_profiler.rewriter.provenance import export_provenance_jsonl


@dataclass
class LoweringResult:
    """Output of ``lower_to_inductor``."""
    compiled_model: Callable
    provenance_jsonl_path: str | None
    compile_mode: str = "inductor"
    input_shapes: dict[str, list[int]] = field(default_factory=dict)


def lower_to_inductor(
    gm: torch.fx.GraphModule,
    example_inputs: list[torch.Tensor],
    warmup_iters: int = 2,
    provenance_output_path: str | Path | None = None,
    fullgraph: bool = True,
    device: str = "cuda",
    backend: str = "inductor",
) -> LoweringResult:
    """
    Lower a rewritten ``GraphModule`` to compiled kernels.

    Steps
    -----
    1. Export provenance JSONL from any fused-node metadata.
    2. Compile with ``torch.compile(gm, backend=backend, fullgraph=fullgraph)``.
    3. Run ``warmup_iters`` forward passes to trigger kernel compilation.
    4. Synchronise CUDA (if available).
    5. Return a ``LoweringResult``.

    Parameters
    ----------
    gm:
        Verified, fully rewritten ``GraphModule``.
    example_inputs:
        Representative input tensors (shapes / dtypes used for tracing).
    warmup_iters:
        Number of warm-up forward passes (≥2 recommended, consistent with
        ``NvtxCapture.warmup_iters``).
    provenance_output_path:
        Path for the JSONL sidecar.  If *None* the sidecar is not written.
    fullgraph:
        Passed directly to ``torch.compile``.  ``True`` raises on graph-break.
    device:
        Target device string (``"cuda"`` for production, ``"cpu"`` for tests).
    backend:
        Torch compile backend (``"inductor"`` for production, ``"eager"`` for
        CPU tests).
    """
    # 1. Emit provenance sidecar before compilation (metadata lives in gm)
    jsonl_path: str | None = None
    if provenance_output_path is not None:
        ppath = Path(provenance_output_path)
        export_provenance_jsonl(gm, ppath)
        jsonl_path = str(ppath)

    # 2. Compile
    compiled = torch.compile(gm, backend=backend, fullgraph=fullgraph)

    # 3. Warmup
    gm.eval()
    with torch.no_grad():
        for _ in range(warmup_iters):
            compiled(*example_inputs)

    # 4. Sync CUDA if available
    if torch.cuda.is_available():
        torch.cuda.synchronize()

    # 5. Build input_shapes metadata
    input_shapes: dict[str, list[int]] = {}
    placeholders = [n for n in gm.graph.nodes if n.op == "placeholder"]
    for node, tensor in zip(placeholders, example_inputs):
        input_shapes[node.name] = list(tensor.shape)

    return LoweringResult(
        compiled_model=compiled,
        provenance_jsonl_path=jsonl_path,
        compile_mode=backend,
        input_shapes=input_shapes,
    )
