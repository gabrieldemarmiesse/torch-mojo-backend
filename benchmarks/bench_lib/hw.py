"""Hardware identity for the benchmark suite.

A baseline ratio only means something on the hardware that produced it,
against the stock backend it was compared with, measured the way it was
measured, under the torch build that supplied the stock leg.  All four
are therefore part of the key under which ratios are stored:

    "<device name> | <stock backend> | <timing method> | torch <version>"

The timing method matters because Apple exposes no public per-kernel GPU
counter: device time there is a saturated-queue proxy (enqueue a burst,
synchronize, divide).  Encoding the method in the key keeps a proxy-timed
Apple ratio from ever being read as directly comparable to a CUPTI-timed
CUDA ratio, without adding any extra fields to the baseline file.

The torch version matters because the ratio's denominator IS stock
PyTorch: an upgraded torch changes what every ratio means.  With the
exact version in the key, an upgraded box is simply an unseen config —
"never measured", so every case passes and records — the same safe rule
already applied to new hardware.

When the suite can pin the GPU core clock (bench_lib/clock.py), the pinned
frequency is appended to the method segment ("kineto-device-time@1395MHz"):
the stock cuBLAS leg is strongly clock-sensitive while ours barely is, so a
ratio at a pinned clock and one under ambient clock policy are different
measurements (S1-NN-tf32 stock leg: 567us pinned vs 612-788us ambient).  A
box that cannot pin keeps the bare method name — and on shared machines its
tf32 ratios inherit the ambient bimodality documented in clock.py.
"""

from __future__ import annotations

import dataclasses
import os
import platform
import subprocess

import torch

from bench_lib import clock


@dataclasses.dataclass(frozen=True)
class Hardware:
    """Identity of one benchmarkable configuration."""

    name: str  # e.g. "NVIDIA H100 PCIe", "AMD Instinct MI300X", "Apple M4"
    stock_device: str  # torch device string of the stock leg: cuda / mps / cpu
    stock_backend: str  # cuda / rocm / mps / cpu ("stock PyTorch" per platform)
    method: str  # how device time is obtained (see module docstring)
    pinned_clock_mhz: int | None = None  # core-clock pin, part of the method

    @property
    def key(self) -> str:
        """Baseline config key: hardware identity plus the torch build."""
        method = self.method
        if self.pinned_clock_mhz is not None:
            method = f"{method}@{self.pinned_clock_mhz}MHz"
        return (
            f"{self.name} | {self.stock_backend} | {method} | torch {torch.__version__}"
        )

    @property
    def is_accelerator(self) -> bool:
        return self.stock_backend != "cpu"


def _apple_chip_name() -> str:
    try:
        out = subprocess.run(
            ["sysctl", "-n", "machdep.cpu.brand_string"],
            capture_output=True,
            text=True,
            timeout=5,
        ).stdout.strip()
        if out:
            return out
    except OSError:
        pass
    return f"Apple {platform.machine()}"


def detect() -> Hardware | None:
    """The accelerator configuration of this machine, or None.

    CPU is a valid stock reference but is never benchmarked implicitly (a
    GEMM matrix sized for GPUs would run for hours): opting in requires
    TORCH_MOJO_BACKEND_BENCH_CPU=1.
    """
    if torch.cuda.is_available():
        name = torch.cuda.get_device_name(0)
        backend = "rocm" if torch.version.hip else "cuda"
        # ROCm pinning is not implemented; a ROCm box stays unpinned.
        pin = clock.probe_pinning() if backend == "cuda" else None
        return Hardware(name, "cuda", backend, "kineto-device-time", pin)
    mps = getattr(torch.backends, "mps", None)
    if mps is not None and mps.is_available():
        return Hardware(_apple_chip_name(), "mps", "mps", "queue-proxy-device-time")
    if os.environ.get("TORCH_MOJO_BACKEND_BENCH_CPU", "") == "1":
        return Hardware(
            platform.processor() or platform.machine(), "cpu", "cpu", "perf-counter"
        )
    return None
