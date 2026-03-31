"""
collect_provenance.py — Inductor provenance sidecar collector.

Loads a user workload, compiles and warms up the model, then uses
torch.profiler to walk the cpu_parent chain of each CUDA kernel event
and write a provenance JSONL sidecar.

Must be run WITHOUT nsys — torch.profiler and nsys both hook into CUPTI
and cannot run simultaneously.

Workload interface
------------------
The workload script must expose:

    def get_model_and_input() -> tuple[model, input_tensor]:
        ...

The returned model should be a plain nn.Module on CUDA. Compilation and
warmup are handled by this tool via --compile-backend and --warmup-iters.

Usage:
    python scripts/collect_provenance.py \\
        --workload scripts/inductor_workload.py \\
        --output profile.provenance.jsonl
"""
from __future__ import annotations

import argparse
import importlib.util
import sys
from pathlib import Path

import torch

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from operator_profiler.capture.provenance_collector import (
    collect_kernel_provenance,
    write_provenance_jsonl,
)


def _load_workload(script_path: str):
    """Dynamically import a workload script and return its module."""
    path = Path(script_path).resolve()
    if not path.exists():
        print(f"[collect_provenance] ERROR: workload script not found: {path}", flush=True)
        sys.exit(1)
    spec = importlib.util.spec_from_file_location("_workload", path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    if not hasattr(module, "get_model_and_input"):
        print(
            f"[collect_provenance] ERROR: {path.name} does not expose "
            "get_model_and_input(). Add:\n\n"
            "    def get_model_and_input():\n"
            "        # return (model, input_tensor)\n",
            flush=True,
        )
        sys.exit(1)
    return module


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--workload", required=True,
        help="Path to a workload script that exposes get_model_and_input().",
    )
    parser.add_argument(
        "--output", default="workload.provenance.jsonl",
        help="Path to write the provenance JSONL sidecar.",
    )
    parser.add_argument(
        "--compile-backend", default="inductor",
        help="torch.compile backend to use (default: inductor). Pass 'none' to skip compilation.",
    )
    parser.add_argument(
        "--warmup-iters", type=int, default=5,
        help="Number of warmup iterations before provenance collection (default: 5).",
    )
    args = parser.parse_args()

    assert torch.cuda.is_available(), "CUDA required"
    print(f"[collect_provenance] GPU: {torch.cuda.get_device_name(0)}", flush=True)

    workload = _load_workload(args.workload)
    print(f"[collect_provenance] Loading workload: {args.workload}", flush=True)
    model, x = workload.get_model_and_input()

    if args.compile_backend != "none":
        print(f"[collect_provenance] Compiling with backend='{args.compile_backend}'...", flush=True)
        model = torch.compile(model, backend=args.compile_backend)

    print(f"[collect_provenance] Warmup ({args.warmup_iters} iters)...", flush=True)
    with torch.no_grad():
        for _ in range(args.warmup_iters):
            model(x)
    torch.cuda.synchronize()

    print("[collect_provenance] Collecting kernel provenance...", flush=True)
    kernel_to_ops = collect_kernel_provenance(model, x)
    n_written = write_provenance_jsonl(kernel_to_ops, args.output)

    print(f"[collect_provenance] {n_written} kernel entries → {args.output}", flush=True)
    if n_written == 0:
        print(
            "[collect_provenance] WARNING: no CUDA kernel events captured — "
            "torch.profiler may not have observed Triton kernel launches",
            flush=True,
        )


if __name__ == "__main__":
    main()
