#!/usr/bin/env python3
"""Correctness gate for the eager Mojo SDPA against PyTorch-ROCm.

Attention has no closed-form gate like the GEMM harness's, so this uses the
repo's established rule (optimization_journal.md): compute an FP32 PyTorch-ROCm
result on identical inputs, then require the Mojo BF16 maximum absolute error
against it to be no more than twice the PyTorch-ROCm BF16 error against the same
FP32 reference.  It is applied separately to the forward output and to dQ, dK and
dV, at the real nanoGPT shape and at shapes that are not multiples of any tile.

The FP32 reference deliberately uses PyTorch's math SDPA backend, so it is a
plain ``softmax(QK^T / sqrt(d)) V`` in FP32 rather than a second fused kernel.

Known result, unchanged by the causal work: the ``noncausal`` case fails on the
output, dK and dV at 2.3-3.1x. That path does not use the causal row softmax; it
uses MAX's ``nn.softmax`` with the scale folded into an input lambda that rounds
``scores * scale`` back to the operand dtype before the reduction --- a second
BF16 rounding PyTorch's math backend does not pay, because it scales Q before the
BMM. The lambda's own comment already flags that round-trip as a known trade-off.
Every causal case, which is what the rewritten kernel handles and which keeps the
scaled value in float32, comes in at 0.76-1.84x.

Usage:
    uv run --no-sync python scripts/sdpa_correctness.py
    uv run --no-sync python scripts/sdpa_correctness.py --case nanogpt
"""

from __future__ import annotations

import argparse
import math
from dataclasses import dataclass

import torch
import torch.nn.functional as F


@dataclass(frozen=True)
class Case:
    label: str
    batch: int
    heads: int
    q_len: int
    head_dim: int
    is_causal: bool


CASES = (
    # The frozen workload.
    Case("nanogpt", 48, 12, 1024, 64, True),
    # Sequence lengths that are not multiples of the 128/64 output tiles, and
    # 1025 is not a multiple of the 8-element BF16 vector width either.
    Case("seq1000", 2, 4, 1000, 64, True),
    Case("seq1025", 2, 4, 1025, 64, True),
    # Batch 1 and a single head: the smallest grids the dispatch can see.
    Case("batch1", 1, 12, 1024, 64, True),
    Case("head1", 1, 1, 1024, 64, True),
    Case("head1_small", 1, 1, 129, 64, True),
    # Non-causal, to confirm the causal regimes are not doing the work of the
    # dense path by accident.
    Case("noncausal", 2, 4, 512, 64, False),
)


def _reference(
    case: Case, device: str, dtype: torch.dtype, seed: int
) -> tuple[torch.Tensor, ...]:
    """Forward output and the three input gradients on one device and dtype."""
    generator = torch.Generator(device="cpu").manual_seed(seed)
    shape = (case.batch, case.heads, case.q_len, case.head_dim)
    q = torch.randn(shape, generator=generator, dtype=torch.float32)
    k = torch.randn(shape, generator=generator, dtype=torch.float32)
    v = torch.randn(shape, generator=generator, dtype=torch.float32)
    grad_out = torch.randn(shape, generator=generator, dtype=torch.float32)

    qd = q.to(device=device, dtype=dtype).requires_grad_(True)
    kd = k.to(device=device, dtype=dtype).requires_grad_(True)
    vd = v.to(device=device, dtype=dtype).requires_grad_(True)
    gd = grad_out.to(device=device, dtype=dtype)

    if device == "cuda":
        with torch.nn.attention.sdpa_kernel(torch.nn.attention.SDPBackend.MATH):
            out = F.scaled_dot_product_attention(qd, kd, vd, is_causal=case.is_causal)
    else:
        out = F.scaled_dot_product_attention(qd, kd, vd, is_causal=case.is_causal)
    out.backward(gd)
    return (
        out.detach().float().cpu(),
        qd.grad.detach().float().cpu(),
        kd.grad.detach().float().cpu(),
        vd.grad.detach().float().cpu(),
    )


def _max_abs(a: torch.Tensor, b: torch.Tensor) -> float:
    return float((a - b).abs().max().item())


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--case", default="all")
    parser.add_argument("--seed", type=int, default=1234)
    parser.add_argument(
        "--tolerance-factor",
        type=float,
        default=2.0,
        help="allowed ratio of the Mojo BF16 error to the ROCm BF16 error",
    )
    args = parser.parse_args()

    # Every PyTorch-ROCm reference is computed *before* the Mojo device is
    # registered: registering it in-process perturbs the autograd engine's
    # stream bookkeeping and makes a subsequent cuda backward trip an internal
    # assert (`opt_ready_stream && opt_parent_stream`).
    selected = [c for c in CASES if args.case in ("all", c.label)]
    references: dict[
        str, tuple[tuple[torch.Tensor, ...], tuple[torch.Tensor, ...]]
    ] = {}
    for case in selected:
        references[case.label] = (
            _reference(case, "cuda", torch.float32, args.seed),
            _reference(case, "cuda", torch.bfloat16, args.seed),
        )

    from torch_mojo_backend import register_mojo_devices

    register_mojo_devices()

    names = ("output", "dQ", "dK", "dV")
    print(
        f"{'case':<13}{'quantity':<9}{'fp32 ref |x|max':>16}"
        f"{'rocm bf16 err':>15}{'mojo bf16 err':>15}{'ratio':>8}  verdict"
    )
    failures = 0
    for case in selected:
        fp32, rocm = references[case.label]
        mojo = _reference(case, "mojo", torch.bfloat16, args.seed)
        for name, ref, r, m in zip(names, fp32, rocm, mojo, strict=True):
            rocm_err = _max_abs(r, ref)
            mojo_err = _max_abs(m, ref)
            # A zero ROCm error would make the ratio meaningless; fall back to
            # one BF16 ulp of the reference's largest magnitude.
            budget = max(
                rocm_err * args.tolerance_factor,
                float(ref.abs().max().item()) * 2.0**-8,
            )
            ok = mojo_err <= budget and math.isfinite(mojo_err)
            failures += 0 if ok else 1
            ratio = mojo_err / rocm_err if rocm_err > 0 else float("inf")
            print(
                f"{case.label:<13}{name:<9}{float(ref.abs().max().item()):>16.6g}"
                f"{rocm_err:>15.6g}{mojo_err:>15.6g}{ratio:>8.3f}"
                f"  {'pass' if ok else 'FAIL'}"
            )
    if failures:
        print(f"\n{failures} comparison(s) FAILED")
        return 1
    print("\nall comparisons pass")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
