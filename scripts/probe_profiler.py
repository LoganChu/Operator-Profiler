import torch, torch.nn as nn
from torch.profiler import profile, ProfilerActivity

model = nn.Linear(256, 512).cuda().eval()
x = torch.randn(32, 256, device='cuda')

with profile(activities=[ProfilerActivity.CPU, ProfilerActivity.CUDA], acc_events=True) as prof:
    with torch.no_grad():
        for _ in range(5):
            model(x)
torch.cuda.synchronize()

evts = prof.key_averages()
print(f"Total events: {len(evts)}")
if evts:
    e = evts[0]
    print("Attributes:", [a for a in dir(e) if not a.startswith('_') and 'time' in a.lower()])
    for evt in sorted(evts, key=lambda e: e.cpu_time_total, reverse=True)[:10]:
        print(f"  {evt.key:<35} cpu={evt.cpu_time_total:>8.0f}µs  "
              f"cuda={getattr(evt,'cuda_time_total',0):>8.0f}µs  "
              f"device={getattr(evt,'device_time_total',0):>8.0f}µs  "
              f"self_cpu={evt.self_cpu_time_total:>8.0f}µs")
