"""
workload.py — Reference workload for the preprocessing pipeline.

Implements the workload interface used by the pipeline tools:

    def get_model_and_input() -> tuple[model, input_tensor]:
        ...

The returned model is a raw nn.Module on CUDA. Compilation and warmup
are handled by the pipeline tools (collect_provenance.py, run_workload.py).

The model is a TransformerBlock (attention + FFN + layer norm), representative
of LLM inference and covering a broad range of kernel types:
gemm, softmax, layer_norm, relu, gelu.

Can also be run directly under nsys for quick manual testing:
    nsys profile --trace=cuda,nvtx --output=<prefix> python scripts/workload.py
"""
from __future__ import annotations

import sys

import torch
import torch.nn as nn
import torch.autograd.profiler as autograd_profiler

DEVICE      = "cuda"
BATCH_SIZE  = 16
IN_FEATURES = 512
HIDDEN      = 2048
WARMUP      = 5
MEASURE     = 20


class FFBlock(nn.Module):
    """Transformer feed-forward block: Linear → ReLU → Linear → GELU."""
    def __init__(self):
        super().__init__()
        self.fc1 = nn.Linear(IN_FEATURES, HIDDEN, bias=True)
        self.fc2 = nn.Linear(HIDDEN, IN_FEATURES, bias=True)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return torch.nn.functional.gelu(self.fc2(torch.relu(self.fc1(x))))


class AttentionBlock(nn.Module):
    """Single-head attention projection pair for GEMM coverage."""
    def __init__(self):
        super().__init__()
        self.q_proj  = nn.Linear(IN_FEATURES, IN_FEATURES, bias=False)
        self.v_proj  = nn.Linear(IN_FEATURES, IN_FEATURES, bias=False)
        self.out_proj = nn.Linear(IN_FEATURES, IN_FEATURES, bias=False)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        q = torch.relu(self.q_proj(x))
        v = self.v_proj(x)
        scores = torch.softmax(q @ v.transpose(-1, -2) / (IN_FEATURES ** 0.5), dim=-1)
        return self.out_proj(scores @ v)


class TransformerBlock(nn.Module):
    """Attention + FFN with layer-norm — representative of LLM inference."""
    def __init__(self):
        super().__init__()
        self.attn = AttentionBlock()
        self.ff   = FFBlock()
        self.ln1  = nn.LayerNorm(IN_FEATURES)
        self.ln2  = nn.LayerNorm(IN_FEATURES)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = x + self.attn(self.ln1(x))
        x = x + self.ff(self.ln2(x))
        return x


def get_model_and_input() -> tuple:
    """
    Workload interface — return (raw_model, input_tensor).

    Returns an uncompiled, unwarmed model. The pipeline tools
    (collect_provenance.py, run_workload.py) handle compilation and warmup.
    """
    assert torch.cuda.is_available(), "CUDA required"
    model = TransformerBlock().to(DEVICE).eval()
    x     = torch.randn(BATCH_SIZE, IN_FEATURES, device=DEVICE)
    return model, x


def main() -> None:
    assert torch.cuda.is_available(), "CUDA required"
    print(f"[workload] GPU: {torch.cuda.get_device_name(0)}", flush=True)

    model, x = get_model_and_input()

    print("[workload] Compiling with inductor...", flush=True)
    compiled_model = torch.compile(model, backend="inductor")

    print(f"[workload] Warmup ({WARMUP} iters)...", flush=True)
    with torch.no_grad():
        for _ in range(WARMUP):
            _ = compiled_model(x)
    torch.cuda.synchronize()

    print(f"[workload] Capture ({MEASURE} iters with emit_nvtx)...", flush=True)
    with torch.no_grad():
        with autograd_profiler.emit_nvtx(record_shapes=True):
            for _ in range(MEASURE):
                _ = compiled_model(x)
    torch.cuda.synchronize()

    print("[workload] Done.", flush=True)


if __name__ == "__main__":
    main()
