"""Benchmark the nanoGPT optimizer phase (clip_grad_norm_ + AdamW.step)
on the mojo device vs torch MPS.

The mojo backend has fused foreach kernels only for `_foreach_norm` /
`_foreach_mul_.Tensor` (grad clip) and fused AdamW; the `_foreach_*` family
AdamW's foreach path uses (`mul_.Scalar`, `add_.Scalar/Tensor`,
`lerp_.Scalar`, `addcmul_.Scalar`, `addcdiv_.ScalarList`,
`div_.ScalarList`, `sqrt`) falls back to sequential per-tensor dispatch —
~550 kernel launches per training step. This script measures that phase in
isolation on nanoGPT's exact 39-tensor parameter set.

    uv run python bench_foreach_optimizer.py mojo
    uv run python bench_foreach_optimizer.py mps

Reports: async wall time of the full optimizer phase, plus sync-mode
per-op attribution of every `_foreach_*` / optimizer-related aten op.
"""

import sys
import time

import torch
from torch.utils._python_dispatch import TorchDispatchMode

DEVICE = sys.argv[1] if len(sys.argv) > 1 else "mojo"

if DEVICE == "mojo":
    from torch_mojo_backend import register_mojo_devices

    register_mojo_devices()
    dev = torch.device("mojo")

    def sync() -> None:
        torch.mojo.synchronize(dev)

else:
    dev = torch.device("mps")

    def sync() -> None:
        torch.mps.synchronize()


# nanoGPT shakespeare-char parameter set: 26 decayed 2D tensors + 13 1D.
torch.manual_seed(0)
shapes_2d = [(65, 384), (256, 384)] + [
    (1152, 384),
    (384, 384),
    (1536, 384),
    (384, 1536),
] * 6
shapes_1d = [(384,)] * 13
params = [
    torch.nn.Parameter(torch.randn(*s, device=dev) * 0.02)
    for s in shapes_2d + shapes_1d
]
for p in params:
    p.grad = torch.randn_like(p) * 0.01

groups = [
    {"params": params[: len(shapes_2d)], "weight_decay": 0.1},
    {"params": params[len(shapes_2d) :], "weight_decay": 0.0},
]
# No explicit fused/foreach: same auto-selection nanoGPT's
# configure_optimizers gets on non-CUDA devices (foreach on both backends).
opt = torch.optim.AdamW(groups, lr=1e-3, betas=(0.9, 0.99))


def phase() -> None:
    torch.nn.utils.clip_grad_norm_(params, 1.0)
    opt.step()


for _ in range(5):
    phase()
sync()

n = 30
t0 = time.perf_counter()
for _ in range(n):
    phase()
sync()
wall_ms = (time.perf_counter() - t0) / n * 1000
print(f"{DEVICE}: optimizer phase (clip + step) async wall: {wall_ms:.2f} ms")

stats: dict[str, list[float]] = {}


class SyncProfiler(TorchDispatchMode):
    def __torch_dispatch__(self, func, types, args=(), kwargs=None):  # noqa: ANN001, ANN002
        sync()
        t0 = time.perf_counter()
        out = func(*args, **(kwargs or {}))
        sync()
        entry = stats.setdefault(str(func), [0, 0.0])
        entry[0] += 1
        entry[1] += time.perf_counter() - t0
        return out


reps = 3
with SyncProfiler():
    for _ in range(reps):
        phase()

print(f"{DEVICE}: sync-mode per-op attribution (per phase):")
total = 0.0
for name, (count, seconds) in sorted(stats.items(), key=lambda kv: -kv[1][1]):
    ms = seconds / reps * 1000
    total += ms
    if ms > 0.05:
        print(f"  {name.replace('aten.', ''):40s} x{count // reps:<4d} {ms:8.2f} ms")
print(f"  {'TOTAL':40s}       {total:8.2f} ms")
