import torch, sys
print(f"PyTorch:  {torch.__version__}")
print(f"CUDA available: {torch.cuda.is_available()}")
if torch.cuda.is_available():
    print(f"Device:   {torch.cuda.get_device_name(0)}")
    print(f"CUDA:     {torch.version.cuda}")
    print(f"VRAM:     {torch.cuda.get_device_properties(0).total_memory / 1e9:.1f} GB")
    t = torch.randn(1024, 1024, device='cuda')
    print(f"Tensor smoke test: {t.shape} on {t.device}  OK")
