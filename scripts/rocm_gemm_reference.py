#!/usr/bin/env python3
"""Time the exact nanoGPT training-step GEMMs on PyTorch-ROCm.

Emits the target table the pure-Mojo GEMM harness compares itself against, so
an optimization agent never has to start a PyTorch process to know what it is
aiming at.  One row per GEMM the training step actually issues:

* ``fwd``   -- ``input @ weight.T``, the forward Linear
* ``dgrad`` -- ``grad_output @ weight``, the data gradient
* ``wgrad`` -- ``grad_output.T @ input``, the weight gradient

ROCm consumes the transposed operands of ``dgrad``/``wgrad`` as strided views,
so these timings include no transpose materialization.  That is deliberate: the
Mojo side currently materializes one, and the harness should show that cost as
part of the gap rather than hide it.

Layout flags describe the logical GEMM the kernel must perform, matching the
Mojo dispatch's own convention:
    transpose_a=1  the M x K operand is read from a K x M buffer
    transpose_b=1  the K x N operand is read from an N x K buffer

Usage:
    uv run --no-sync python scripts/rocm_gemm_reference.py \
        --output harness/nanogpt_train/rocm_gemm_targets.csv
"""

from __future__ import annotations

import argparse
import csv
import statistics
import time
from dataclasses import dataclass
from pathlib import Path

import torch

# nanoGPT GPT-2 124M, batch 48, block 1024: one row per Linear in the model,
# with its per-step call count.  in_features/out_features are the nn.Linear
# geometry; the weight buffer is always [out_features, in_features].
TOKENS = 48 * 1024


@dataclass(frozen=True)
class LinearLayer:
    name: str
    in_features: int
    out_features: int
    calls_per_step: int


LAYERS = (
    LinearLayer("attn_c_attn", 768, 2304, 12),
    LinearLayer("attn_c_proj", 768, 768, 12),
    LinearLayer("mlp_c_fc", 768, 3072, 12),
    LinearLayer("mlp_c_proj", 3072, 768, 12),
    LinearLayer("lm_head", 768, 50304, 1),
)


@dataclass(frozen=True)
class GemmCase:
    label: str
    role: str
    m: int
    n: int
    k: int
    transpose_a: int
    transpose_b: int
    calls_per_step: int


def build_cases(layers: tuple[LinearLayer, ...], tokens: int) -> list[GemmCase]:
    cases = []
    for layer in layers:
        in_features, out_features = layer.in_features, layer.out_features
        cases.append(
            GemmCase(
                f"{layer.name}_fwd",
                "fwd",
                tokens,
                out_features,
                in_features,
                0,
                1,
                layer.calls_per_step,
            )
        )
        cases.append(
            GemmCase(
                f"{layer.name}_dgrad",
                "dgrad",
                tokens,
                in_features,
                out_features,
                0,
                0,
                layer.calls_per_step,
            )
        )
        cases.append(
            GemmCase(
                f"{layer.name}_wgrad",
                "wgrad",
                out_features,
                in_features,
                tokens,
                1,
                0,
                layer.calls_per_step,
            )
        )
    return cases


def make_operands(
    case: GemmCase, device: torch.device
) -> tuple[torch.Tensor, torch.Tensor]:
    """Physical buffers plus the views the GEMM's layout flags describe."""
    if case.transpose_a:
        left = torch.randn(case.k, case.m, device=device, dtype=torch.bfloat16).t()
    else:
        left = torch.randn(case.m, case.k, device=device, dtype=torch.bfloat16)
    if case.transpose_b:
        right = torch.randn(case.n, case.k, device=device, dtype=torch.bfloat16).t()
    else:
        right = torch.randn(case.k, case.n, device=device, dtype=torch.bfloat16)
    return left, right


def time_case(case: GemmCase, warmup: int, iters: int) -> tuple[float, float, float]:
    device = torch.device("cuda")
    left, right = make_operands(case, device)
    for _ in range(warmup):
        torch.mm(left, right)
    torch.cuda.synchronize()
    samples = []
    for _ in range(iters):
        start = time.perf_counter()
        torch.mm(left, right)
        torch.cuda.synchronize()
        samples.append((time.perf_counter() - start) * 1e6)
    del left, right
    torch.cuda.empty_cache()
    ordered = sorted(samples)
    return (
        statistics.median(ordered),
        ordered[max(int(0.10 * (len(ordered) - 1)), 0)],
        ordered[int(0.90 * (len(ordered) - 1))],
    )


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--warmup", type=int, default=25)
    parser.add_argument("--iters", type=int, default=100)
    parser.add_argument("--tokens", type=int, default=TOKENS)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    if torch.version.hip is None:
        raise RuntimeError(
            f"A ROCm PyTorch build is required; found torch {torch.__version__}"
        )
    if not torch.cuda.is_available():
        raise RuntimeError("PyTorch cannot access the ROCm GPU")
    print(
        f"Reference: torch {torch.__version__}, {torch.cuda.get_device_name(0)}, "
        f"{args.warmup} warmups, {args.iters} timed iterations"
    )

    cases = build_cases(LAYERS, args.tokens)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    total_us = 0.0
    with args.output.open("w", newline="") as file:
        writer = csv.writer(file)
        writer.writerow(
            [
                "label",
                "role",
                "m",
                "n",
                "k",
                "transpose_a",
                "transpose_b",
                "calls_per_step",
                "rocm_us",
                "rocm_us_p10",
                "rocm_us_p90",
                "tflops",
            ]
        )
        for case in cases:
            median, p10, p90 = time_case(case, args.warmup, args.iters)
            tflops = 2.0 * case.m * case.n * case.k / (median * 1e-6) / 1e12
            total_us += median * case.calls_per_step
            writer.writerow(
                [
                    case.label,
                    case.role,
                    case.m,
                    case.n,
                    case.k,
                    case.transpose_a,
                    case.transpose_b,
                    case.calls_per_step,
                    f"{median:.2f}",
                    f"{p10:.2f}",
                    f"{p90:.2f}",
                    f"{tflops:.1f}",
                ]
            )
            print(
                f"{case.label:24s} m={case.m:6d} n={case.n:6d} k={case.k:6d} "
                f"ta={case.transpose_a} tb={case.transpose_b} "
                f"x{case.calls_per_step:2d}  {median:9.2f} us  {tflops:6.1f} TFLOP/s"
            )
    print(
        f"\nWrote {args.output}\nSum over one training step: {total_us / 1000:.3f} ms"
    )


if __name__ == "__main__":
    main()
