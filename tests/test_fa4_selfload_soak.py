"""Race regression soaks for the self-loading FA4 fwd kernel (CI-lite).

Two of them, at two levels:

``test_selfload_soak_no_race`` drives the kernel through its Mojo launcher.
Ported from agent A's phase-2c harness
(``/scratch/fa4-fwd-harness-2c/soak_v7.mojo``), shortened to two shapes and
fewer reps for the permanent suite: bursts of back-to-back self-load kernel
launches with no inter-launch sync (so consecutive kernels' CTAs are
co-resident and warp scheduling gets perturbed), checked for bitwise
determinism across reps plus a tolerance check against the structurally
different phase-2b kernel. The full soak (more shapes, more reps, multiple
clock pins) stays in the harness; see ``tests/fa4_selfload_soak_probe.mojo``
for what this CI-lite variant covers and why (the PREFETCH invariant this
guards is documented next to its definition in
``fa4_fwd_selfload_kernel.mojo``).

``test_selfload_bhsd_production_soak`` drives the SAME kernel through the
route a user actually reaches: ``F.scaled_dot_product_attention`` on
contiguous (B, H, S, D) mojo tensors, which is the ONLY way into this
kernel (``fa4_ops.mojo``'s bhsd entry points, above the wave gate; the
launcher hardcodes ``bhsd=True``, so a strided/fused-qkv model such as
nanoGPT never lands here at all and this is the layout that does). It
proves the route with the profiler instead of assuming it, and it runs
longer than the Mojo probe because it is cheap: each launch is ~66us on an
H100, so the whole thing is a few seconds. ``FA4_SELFLOAD_SOAK_REPS`` in
the environment turns it up for a deliberate stress run.
"""

from __future__ import annotations

import os
import subprocess
from collections.abc import Callable
from pathlib import Path

import pytest
import torch
import torch.nn.functional as F
from torch.profiler import ProfilerActivity, profile

from scripts.compare_kernel_asm import build_env, mojo_cli
from torch_mojo_backend import get_accelerators
from torch_mojo_backend.mojo_device import torch_mojo_device_module

_REPO_ROOT = Path(__file__).resolve().parents[1]
_PROBE = Path(__file__).resolve().parent / "fa4_selfload_soak_probe.mojo"
_FA4_DIR = _REPO_ROOT / "torch_mojo_backend" / "eager_flash_attention"

# (batch, heads, seqlen) at head_dim 64, causal, bf16. Both clear the
# self-load wave gate (>= 2 waves of 3 CTAs/SM) on every H100 SKU: at 114
# SMs they are 4.4 and 3.3 waves, at 132 SMs 3.8 and 2.8 -- and the test
# asserts the route rather than trusting that arithmetic. The second shape
# is deliberately awkward: 1608 is not a multiple of BM=64, so its last
# m-tile is partial and the epilogue takes the clamped-store path, and
# 11 heads is coprime with everything in the scheduler's swizzle.
_PROD_SHAPES: tuple[tuple[int, int, int], ...] = ((8, 12, 1024), (4, 11, 1608))
# Launches per burst with NO sync between them: consecutive kernels' CTAs
# stay co-resident, which is the schedule perturbation the soak is for.
_PROD_BURST = 8
_PROD_REPS = int(os.environ.get("FA4_SELFLOAD_SOAK_REPS", "12"))
# torch.profiler reports the mojo-device kernels under the PrivateUse1
# backend (same list bench_lib/measure.py uses).
_DEVICE_EVENT_TYPES = ("DeviceType.CUDA", "DeviceType.PrivateUse1")


def _require_sm90a() -> None:
    accelerators = list(get_accelerators())
    if (
        not accelerators
        or accelerators[0].api != "cuda"
        or (accelerators[0].architecture_name != "sm_90a")
    ):
        pytest.skip("the self-load FA4 fwd kernel is H100 (sm_90a) only")


@pytest.fixture(scope="module")
def selfload_soak_binary(tmp_path_factory: pytest.TempPathFactory) -> Path:
    _require_sm90a()
    try:
        mojo = mojo_cli()
    except FileNotFoundError:
        pytest.skip("mojo compiler not found")

    out_dir = tmp_path_factory.mktemp("fa4_selfload_soak")
    out_path = out_dir / "fa4_selfload_soak_probe"
    command = [
        str(mojo),
        "build",
        str(_PROBE),
        "-I",
        str(_FA4_DIR),
        "-o",
        str(out_path),
    ]
    result = subprocess.run(
        command,
        cwd=str(_REPO_ROOT),
        env=build_env(),
        capture_output=True,
        text=True,
        timeout=600,
    )
    assert result.returncode == 0, (
        "mojo build failed for the self-load soak probe:\n"
        f"{result.stderr or result.stdout}"
    )
    return out_path


def test_selfload_soak_no_race(selfload_soak_binary: Path):
    """64 back-to-back self-load launches per shape must be bitwise
    deterministic and match the phase-2b kernel within tolerance."""
    result = subprocess.run(
        [str(selfload_soak_binary)],
        cwd=str(_REPO_ROOT),
        env=build_env(),
        capture_output=True,
        text=True,
        timeout=300,
    )
    output = result.stdout + result.stderr
    assert result.returncode == 0, f"self-load soak failed:\n{output}"
    assert "SOAK PASS" in output, f"self-load soak did not report PASS:\n{output}"


def _device_kernel_names(run: Callable[[], object]) -> set[str]:
    """Names of the device kernels `run` launches, per torch.profiler."""
    with profile(activities=[ProfilerActivity.CPU, ProfilerActivity.CUDA]) as prof:
        run()
        torch_mojo_device_module.synchronize()
    return {
        event.name
        for event in prof.events()
        if str(event.device_type) in _DEVICE_EVENT_TYPES
    }


@pytest.mark.parametrize("shape", _PROD_SHAPES, ids=lambda s: "B{}H{}S{}D64".format(*s))
def test_selfload_bhsd_production_soak(
    shape: tuple[int, int, int], mojo_gpu: str
) -> None:
    """The reachable-in-production path: contiguous BHSD SDPA, soaked.

    Every launch is checked bitwise against the first one (this kernel is
    deterministic by construction -- no atomics, no split-K -- so a single
    moved bit IS a race) and, once, against a CPU float32 reference so a
    corruption that happened to be bitwise-stable would still show.
    """
    _require_sm90a()
    batch, heads, seqlen = shape
    device = torch.device(mojo_gpu)
    torch.manual_seed(0xFA4)
    host = [
        torch.randn(batch, heads, seqlen, 64, dtype=torch.bfloat16) for _ in range(3)
    ]
    q, k, v = (tensor.to(device) for tensor in host)
    assert q.is_contiguous() and k.is_contiguous() and v.is_contiguous()

    def attend() -> torch.Tensor:
        return F.scaled_dot_product_attention(q, k, v, is_causal=True)

    # Prove the route instead of assuming the wave gate: this soak is
    # worthless if the shape quietly dispatched to the phase-2b kernel.
    kernels = _device_kernel_names(attend)
    assert any("selfload" in name for name in kernels), (
        f"the contiguous-BHSD SDPA call did not run the self-load kernel"
        f" (device kernels seen: {sorted(kernels)}) -- either the wave gate"
        " in fa4_ops.mojo moved or this shape no longer reaches it"
    )

    gold = attend().cpu()
    reference = F.scaled_dot_product_attention(
        host[0].float(), host[1].float(), host[2].float(), is_causal=True
    )
    # bf16 rounding of the output only; the same tolerance the Mojo probe
    # uses against the phase-2b kernel.
    assert (gold.float() - reference).abs().max().item() < 0.05

    for rep in range(_PROD_REPS):
        burst = [attend() for _ in range(_PROD_BURST)]
        for index, out in enumerate(burst):
            assert torch.equal(out.cpu(), gold), (
                f"self-load output moved on rep {rep}, burst slot {index}"
                f" (shape B{batch} H{heads} S{seqlen} D64): this kernel is"
                " deterministic, so a differing bit is a race -- most likely"
                " a TMA refill overtaking the wgmma.wait_group that frees"
                " its ring slot (see fa4_fwd_selfload_kernel.mojo's PREFETCH"
                " invariant)"
            )
