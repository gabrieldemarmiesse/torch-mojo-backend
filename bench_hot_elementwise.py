"""Benchmark the remaining hot memory-bound ops of a nanoGPT training step:
mojo device vs torch MPS, at the exact shapes the step issues.

    uv run python bench_hot_elementwise.py mojo
    uv run python bench_hot_elementwise.py mps

Each case reports the async-wall per-call median over 50 calls (30 warmup).
"""

import sys
import time
from collections.abc import Callable

import torch

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


torch.manual_seed(0)


def bench(name: str, fn: Callable[[], object], n: int = 50) -> None:
    for _ in range(30):
        fn()
    sync()
    times = []
    for _ in range(n):
        sync()
        t0 = time.perf_counter()
        fn()
        sync()
        times.append(time.perf_counter() - t0)
    times.sort()
    print(f"{DEVICE}: {name:44s} {times[len(times) // 2] * 1e3:8.3f} ms")


# clone of a transposed 4D view — the post-SDPA `.contiguous()` (6x/step).
y = torch.randn(8, 6, 256, 64, device=dev)
yt = y.transpose(1, 2)
bench("clone transposed (8,256,6,64)", lambda: yt.contiguous())

# plain dense clone — autograd grad copies (2048, 1152) is the largest.
g = torch.randn(2048, 1152, device=dev)
bench("clone dense (2048,1152)", lambda: g.clone())

# gelu forward + backward at the MLP shape (6x/step each).
x = torch.randn(8, 256, 1536, device=dev)
bench("gelu fwd (8,256,1536)", lambda: torch.nn.functional.gelu(x))
xg = torch.randn(8, 256, 1536, device=dev, requires_grad=True)
go = torch.randn(8, 256, 1536, device=dev)
out = torch.nn.functional.gelu(xg)


def gelu_bwd() -> None:
    xg.grad = None
    out.backward(go, retain_graph=True)


bench("gelu fwd+bwd (8,256,1536)", gelu_bwd)

# layer_norm forward (13x/step).
ln_x = torch.randn(8, 256, 384, device=dev)
ln_w = torch.randn(384, device=dev)
bench(
    "layer_norm fwd (8,256,384)",
    lambda: torch.nn.functional.layer_norm(ln_x, (384,), ln_w, None),
)

# embedding backward: wte (vocab 65) and wpe (256 positions). Indices are
# created on CPU and transferred, as nanoGPT's get_batch does.
idx = torch.randint(0, 65, (8, 256)).to(dev)
eg = torch.randn(8, 256, 384, device=dev)
bench(
    "embedding_dense_backward vocab=65",
    lambda: torch.ops.aten.embedding_dense_backward(eg, idx, 65, -1, False),
)
pos = torch.arange(256, device=dev)
pg = torch.randn(256, 384, device=dev)
bench(
    "embedding_dense_backward vocab=256 (wpe)",
    lambda: torch.ops.aten.embedding_dense_backward(
        pg.unsqueeze(0), pos.unsqueeze(0), 256, -1, False
    ),
)

# cat — split backward re-assembling the c_attn grad (6x/step).
parts = [torch.randn(2048, 384, device=dev) for _ in range(3)]
bench("cat 3x(2048,384) dim=1", lambda: torch.cat(parts, dim=1))
