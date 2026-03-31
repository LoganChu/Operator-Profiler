"""
Inductor provenance collector.

Runs a compiled model under torch.profiler, walks each CUDA kernel event's
cpu_parent chain to find the enclosing aten:: op, and writes a provenance
JSONL sidecar in the format expected by ManifestBuilder._load_provenance().

This must be run in a plain Python subprocess — NOT under nsys — because
torch.profiler and nsys both hook into CUPTI.  Running both simultaneously
causes nsys to intercept the activity buffers, leaving torch.profiler with
zero CUDA events.

Public API
----------
collect_kernel_provenance(model, x)  ->  dict[str, list[str]]
write_provenance_jsonl(kernel_to_ops, output_path)  ->  int
"""
from __future__ import annotations

import json
import re
from pathlib import Path


def normalize_to_short_name(name: str) -> str:
    """
    Convert a torch.profiler CUDA event name to the nsys shortName format.

    nsys stores the bare function identifier in CUPTI_ACTIVITY_KIND_KERNEL.shortName
    (no 'void' prefix, no template parameters, no namespace qualifiers), while
    torch.profiler exposes the full demangled name.  Examples:

      'void gemmSN_TN_kernel<float, 128, …>'  →  'gemmSN_TN_kernel'
      'triton_poi_fused_addmm_relu_0'          →  'triton_poi_fused_addmm_relu_0'
    """
    name = re.sub(r"^void\s+", "", name)
    base = re.split(r"[<(]", name)[0].rstrip()
    if "::" in base:
        parts = [p for p in base.split("::") if p]
        return parts[-1] if parts else base
    return base


def _walk_cpu_parent_for_aten(evt) -> str | None:
    """
    Walk the cpu_parent chain of a profiler FunctionEvent upward and return
    the first aten:: name found, or None.
    """
    cursor = getattr(evt, "cpu_parent", None)
    while cursor is not None:
        name = getattr(cursor, "name", "") or ""
        if name.startswith("aten::"):
            return name
        cursor = getattr(cursor, "cpu_parent", None)
    return None


def collect_kernel_provenance(model, x) -> dict[str, list[str]]:
    """
    Run one forward pass of *model* on input *x* inside torch.profiler.profile
    and return a mapping of CUDA kernel shortName → list of source aten:: ops.

    The kernel names match what nsys stores in CUPTI_ACTIVITY_KIND_KERNEL.shortName,
    so the dict keys align with what ManifestBuilder._attribute() looks up.

    Parameters
    ----------
    model:
        A compiled (torch.compile) PyTorch model, already warmed up so all
        Triton kernels have been JIT-compiled.
    x:
        Input tensor on the correct device.
    """
    import torch
    from torch.profiler import profile as TorchProfile, ProfilerActivity

    kernel_to_ops: dict[str, list[str]] = {}

    with TorchProfile(
        activities=[ProfilerActivity.CPU, ProfilerActivity.CUDA],
        record_shapes=False,
        with_stack=False,
    ) as prof:
        with torch.no_grad():
            _ = model(x)
    torch.cuda.synchronize()

    for evt in prof.events():
        if evt.device_type != torch.profiler.DeviceType.CUDA:
            continue
        kname = normalize_to_short_name(evt.name)
        if not kname:
            continue
        source_op = _walk_cpu_parent_for_aten(evt)
        if kname not in kernel_to_ops:
            kernel_to_ops[kname] = []
        if source_op and source_op not in kernel_to_ops[kname]:
            kernel_to_ops[kname].append(source_op)

    return kernel_to_ops


def write_provenance_jsonl(
    kernel_to_ops: dict[str, list[str]],
    output_path: str | Path,
) -> int:
    """
    Write provenance JSONL in the format expected by
    ManifestBuilder._load_provenance().

    Only writes entries where at least one aten:: source op was resolved.
    Kernels with no resolved source ops are omitted so _attribute() falls
    through to the NVTX enclosure tier rather than being pinned to an
    "unknown" provenance entry.

    Returns the number of entries written.
    """
    output_path = Path(output_path)
    lines: list[str] = []

    for kname, source_ops in kernel_to_ops.items():
        if not kname or not source_ops:
            continue
        locs = [
            {"file": "", "line": 0, "col": None, "op": op}
            for op in source_ops
        ]
        lines.append(json.dumps({
            "generated_kernel_name": kname,
            "source_ops": source_ops,
            "source_locations": locs,
        }))

    content = "\n".join(lines) + ("\n" if lines else "")
    output_path.write_text(content, encoding="utf-8")
    return len(lines)
