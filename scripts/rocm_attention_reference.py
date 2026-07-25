#!/usr/bin/env python3
"""PyTorch-ROCm reference times for the nanoGPT training step's attention.

Two different references, because the two backends do different things:

* ``--output-op`` records the whole-op target. PyTorch-ROCm runs one fused
  flash-attention kernel per direction (AOTriton ``attn_fwd`` and
  ``bwd_kernel_fuse``), never materializing the score matrix. This is the number
  acceptance is measured against: ratio <= 1.02 on the per-step total.
* ``--output-bmm`` records ``torch.bmm`` for each batched GEMM the Mojo
  decomposition issues. ROCm's attention does not use these, so they are not an
  acceptance target; they answer a different and still useful question -- how
  fast a batched BF16 GEMM of that exact shape can run on this hardware, which
  bounds what the current decomposition could ever reach.

Usage:
    uv run --no-sync python scripts/rocm_attention_reference.py \
        --output-op harness/nanogpt_train/rocm_attention_targets.csv \
        --output-bmm harness/nanogpt_train/rocm_attention_bmm_targets.csv
"""

from __future__ import annotations

import argparse
import csv
import statistics
import time
from dataclasses import dataclass
from pathlib import Path

import torch
import torch.nn.functional as F

# nanoGPT GPT-2 124M at batch 48, block 1024: 12 identical attention layers.
BATCH = 48
HEADS = 12
SEQ = 1024
HEAD_DIM = 64
LAYERS = 12


@dataclass(frozen=True)
class BmmCase:
    label: str
    role: str
    batch: int
    m: int
    n: int
    k: int
    transpose_b: int
    calls_per_step: int


def bmm_cases() -> list[BmmCase]:
    """The six batched GEMMs of the Mojo SDPA decomposition.

    Forward is scores = Q @ K^T then out = P @ V.  Backward is
    dV = (dO^T @ P)^T, dP = dO @ V^T, dQ = dS @ K, dK = (Q^T @ dS)^T; the two
    transposed forms materialize only the small O(L*D) operand, which is why
    their M extent is the head dimension.
    """
    folded = BATCH * HEADS
    return [
        BmmCase("fwd_scores_qkT", "fwd", folded, SEQ, SEQ, HEAD_DIM, 1, LAYERS),
        BmmCase("fwd_out_pv", "fwd", folded, SEQ, HEAD_DIM, SEQ, 0, LAYERS),
        BmmCase("bwd_dv_dotT_p", "bwd", folded, HEAD_DIM, SEQ, SEQ, 0, LAYERS),
        BmmCase("bwd_dp_do_vT", "bwd", folded, SEQ, SEQ, HEAD_DIM, 1, LAYERS),
        BmmCase("bwd_dq_ds_k", "bwd", folded, SEQ, HEAD_DIM, SEQ, 0, LAYERS),
        BmmCase("bwd_dk_qT_ds", "bwd", folded, HEAD_DIM, SEQ, SEQ, 0, LAYERS),
    ]


def percentiles(samples: list[float]) -> tuple[float, float, float]:
    ordered = sorted(samples)
    return (
        statistics.median(ordered),
        ordered[max(int(0.10 * (len(ordered) - 1)), 0)],
        ordered[int(0.90 * (len(ordered) - 1))],
    )


def time_bmm(case: BmmCase, warmup: int, iters: int) -> tuple[float, float, float]:
    device = torch.device("cuda")
    left = torch.randn(case.batch, case.m, case.k, device=device, dtype=torch.bfloat16)
    if case.transpose_b:
        right = torch.randn(
            case.batch, case.n, case.k, device=device, dtype=torch.bfloat16
        ).transpose(1, 2)
    else:
        right = torch.randn(
            case.batch, case.k, case.n, device=device, dtype=torch.bfloat16
        )
    for _ in range(warmup):
        torch.bmm(left, right)
    torch.cuda.synchronize()
    samples = []
    for _ in range(iters):
        start = time.perf_counter()
        torch.bmm(left, right)
        torch.cuda.synchronize()
        samples.append((time.perf_counter() - start) * 1e6)
    del left, right
    torch.cuda.empty_cache()
    return percentiles(samples)


def time_attention(warmup: int, iters: int) -> dict[str, tuple[float, float, float]]:
    """Time one causal BF16 flash-attention forward and backward per layer.

    Shapes and the causal flag match nanoGPT's CausalSelfAttention exactly:
    ``F.scaled_dot_product_attention(q, k, v, is_causal=True)`` on BHSD inputs.
    """
    device = torch.device("cuda")
    shape = (BATCH, HEADS, SEQ, HEAD_DIM)
    query = torch.randn(shape, device=device, dtype=torch.bfloat16, requires_grad=True)
    key = torch.randn(shape, device=device, dtype=torch.bfloat16, requires_grad=True)
    value = torch.randn(shape, device=device, dtype=torch.bfloat16, requires_grad=True)
    grad = torch.randn(shape, device=device, dtype=torch.bfloat16)

    def forward() -> torch.Tensor:
        return F.scaled_dot_product_attention(query, key, value, is_causal=True)

    for _ in range(warmup):
        out = forward()
        out.backward(grad)
        query.grad = key.grad = value.grad = None
    torch.cuda.synchronize()

    forward_samples = []
    for _ in range(iters):
        start = time.perf_counter()
        out = forward()
        torch.cuda.synchronize()
        forward_samples.append((time.perf_counter() - start) * 1e6)
        del out
    query.grad = key.grad = value.grad = None

    backward_samples = []
    for _ in range(iters):
        out = forward()
        torch.cuda.synchronize()
        start = time.perf_counter()
        out.backward(grad)
        torch.cuda.synchronize()
        backward_samples.append((time.perf_counter() - start) * 1e6)
        query.grad = key.grad = value.grad = None
        del out

    return {
        "sdpa_forward": percentiles(forward_samples),
        "sdpa_backward": percentiles(backward_samples),
    }


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output-op", type=Path, default=None)
    parser.add_argument("--output-bmm", type=Path, default=None)
    parser.add_argument("--warmup", type=int, default=25)
    parser.add_argument("--iters", type=int, default=100)
    return parser.parse_args()


def write_csv(path: Path, header: list[str], rows: list[list[object]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="") as file:
        writer = csv.writer(file)
        writer.writerow(header)
        writer.writerows(rows)
    print(f"wrote {path}")


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

    if args.output_op is not None:
        timings = time_attention(args.warmup, args.iters)
        rows = []
        for label, (median, p10, p90) in timings.items():
            rows.append(
                [
                    label,
                    BATCH,
                    HEADS,
                    SEQ,
                    HEAD_DIM,
                    LAYERS,
                    f"{median:.2f}",
                    f"{p10:.2f}",
                    f"{p90:.2f}",
                ]
            )
            print(
                f"{label:16s} b={BATCH} h={HEADS} s={SEQ} d={HEAD_DIM} "
                f"x{LAYERS}  {median:9.2f} us/layer  "
                f"{median * LAYERS / 1000:7.3f} ms/step"
            )
        write_csv(
            args.output_op,
            [
                "label",
                "batch",
                "heads",
                "seq",
                "head_dim",
                "calls_per_step",
                "rocm_us",
                "rocm_us_p10",
                "rocm_us_p90",
            ],
            rows,
        )

    if args.output_bmm is not None:
        rows = []
        total_us = 0.0
        for case in bmm_cases():
            median, p10, p90 = time_bmm(case, args.warmup, args.iters)
            flops = 2.0 * case.batch * case.m * case.n * case.k
            total_us += median * case.calls_per_step
            rows.append(
                [
                    case.label,
                    case.role,
                    case.batch,
                    case.m,
                    case.n,
                    case.k,
                    case.transpose_b,
                    case.calls_per_step,
                    f"{median:.2f}",
                    f"{p10:.2f}",
                    f"{p90:.2f}",
                    f"{flops / (median * 1e-6) / 1e12:.1f}",
                ]
            )
            print(
                f"{case.label:16s} batch={case.batch} m={case.m:5d} n={case.n:5d} "
                f"k={case.k:5d} tb={case.transpose_b} x{case.calls_per_step}  "
                f"{median:9.2f} us  {flops / (median * 1e-6) / 1e12:6.1f} TFLOP/s"
            )
        write_csv(
            args.output_bmm,
            [
                "label",
                "role",
                "batch",
                "m",
                "n",
                "k",
                "transpose_b",
                "calls_per_step",
                "rocm_us",
                "rocm_us_p10",
                "rocm_us_p90",
                "tflops",
            ],
            rows,
        )
        print(f"\nSum of batched GEMMs over one step: {total_us / 1000:.3f} ms")


if __name__ == "__main__":
    main()
