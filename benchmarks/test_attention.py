"""Attention benchmarks: the public SDPA entry point plus each internal
kernel entry (flash / efficient / math) and the flash backward.

The internal ops are called via torch.ops.aten so the exact registered
entry point is pinned — the public F.scaled_dot_product_attention node
additionally measures whatever backend selection stock PyTorch performs.
Shapes are (batch, heads, seq, head_dim): a GPT-2-like block and a long
single-sequence decode-prefill regime.  All cases are causal.

GQA_SHAPES cover the enable_gqa=True regime (see fix/gqa-sdpa-enable-gqa):
Q and K/V carry different head counts, so they get their own shape dict
keyed as B{batch}H{q_heads}KV{kv_heads}S{seq}D{head_dim} — the KV token
in the shape id is what lets a baseline diff show the ratio at a glance.
"""

from __future__ import annotations

import pytest
import torch
import torch.nn.functional as F
from bench_lib.cases import DTYPES, both
from bench_lib.check import Bench
from bench_lib.hw import Hardware

SHAPES: dict[str, tuple[int, int, int, int]] = {
    "B8H12S1024D64": (8, 12, 1024, 64),
    "B1H16S4096D128": (1, 16, 4096, 128),
}

# (batch, q_heads, kv_heads, seq, head_dim). Llama-3-8B's own ratio (32:8,
# head_dim 128) plus an MQA extreme (kv_heads=1) at the same seq/head_dim so
# the two rows differ only in the ratio being measured.
GQA_SHAPES: dict[str, tuple[int, int, int, int, int]] = {
    "B1H32KV8S4096D128": (1, 32, 8, 4096, 128),
    "B1H8KV1S4096D128": (1, 8, 1, 4096, 128),
}

COVERS: dict[str, str] = {
    "aten::scaled_dot_product_attention": "test_sdpa",
    "aten::_scaled_dot_product_flash_attention": "test_sdpa_flash",
    "aten::_scaled_dot_product_efficient_attention": "test_sdpa_efficient",
    "aten::_scaled_dot_product_attention_math": "test_sdpa_math",
    "aten::_scaled_dot_product_flash_attention_backward": "test_sdpa_flash_backward",
}

SKIPPED: dict[str, str] = {}


def _qkv(
    shape_id: str, dtype_id: str, hw: Hardware, mojo: torch.device
) -> tuple[list[torch.Tensor], list[torch.Tensor], float]:
    b, h, s, d = SHAPES[shape_id]
    dtype = DTYPES[dtype_id]
    refs, ours = [], []
    for _ in range(3):
        ref, our = both(torch.randn(b, h, s, d, dtype=dtype), hw, mojo)
        refs.append(ref)
        ours.append(our)
    flops = 4.0 * b * h * s * s * d / 2.0  # causal halves the score matrix
    return refs, ours, flops


@pytest.mark.parametrize("dtype_id", ("bf16", "f16"))
@pytest.mark.parametrize("shape_id", SHAPES)
@pytest.mark.bench_op("scaled_dot_product_attention")
def test_sdpa(
    shape_id: str, dtype_id: str, bench: Bench, hw: Hardware, mojo_device: torch.device
) -> None:
    refs, ours, flops = _qkv(shape_id, dtype_id, hw, mojo_device)
    bench.run(
        lambda: F.scaled_dot_product_attention(*refs, is_causal=True),
        lambda: F.scaled_dot_product_attention(*ours, is_causal=True),
        flops=flops,
    )


def _qkv_gqa(
    shape_id: str, dtype_id: str, hw: Hardware, mojo: torch.device
) -> tuple[list[torch.Tensor], list[torch.Tensor], float]:
    b, h, kv, s, d = GQA_SHAPES[shape_id]
    dtype = DTYPES[dtype_id]
    q_ref, q_our = both(torch.randn(b, h, s, d, dtype=dtype), hw, mojo)
    k_ref, k_our = both(torch.randn(b, kv, s, d, dtype=dtype), hw, mojo)
    v_ref, v_our = both(torch.randn(b, kv, s, d, dtype=dtype), hw, mojo)
    # FLOPs are driven by the query head count: each of the h query heads
    # still attends over the full (broadcast) K/V, same as the equal-head
    # case above -- the KV ratio changes memory traffic, not FLOPs.
    flops = 4.0 * b * h * s * s * d / 2.0  # causal halves the score matrix
    return [q_ref, k_ref, v_ref], [q_our, k_our, v_our], flops


@pytest.mark.parametrize("dtype_id", ("bf16", "f16"))
@pytest.mark.parametrize("shape_id", GQA_SHAPES)
@pytest.mark.bench_op("scaled_dot_product_attention")
def test_sdpa_gqa(
    shape_id: str, dtype_id: str, bench: Bench, hw: Hardware, mojo_device: torch.device
) -> None:
    """enable_gqa=True: K/V carry fewer heads than Q (see GQA_SHAPES)."""
    refs, ours, flops = _qkv_gqa(shape_id, dtype_id, hw, mojo_device)
    bench.run(
        lambda: F.scaled_dot_product_attention(*refs, is_causal=True, enable_gqa=True),
        lambda: F.scaled_dot_product_attention(*ours, is_causal=True, enable_gqa=True),
        flops=flops,
    )


@pytest.mark.parametrize("dtype_id", ("bf16", "f16"))
@pytest.mark.parametrize("shape_id", SHAPES)
@pytest.mark.bench_op("_scaled_dot_product_flash_attention")
def test_sdpa_flash(
    shape_id: str, dtype_id: str, bench: Bench, hw: Hardware, mojo_device: torch.device
) -> None:
    refs, ours, flops = _qkv(shape_id, dtype_id, hw, mojo_device)
    bench.run(
        lambda: torch.ops.aten._scaled_dot_product_flash_attention(
            *refs, 0.0, True, False
        ),
        lambda: torch.ops.aten._scaled_dot_product_flash_attention(
            *ours, 0.0, True, False
        ),
        flops=flops,
    )


@pytest.mark.parametrize("dtype_id", ("bf16", "f16"))
@pytest.mark.parametrize("shape_id", SHAPES)
@pytest.mark.bench_op("_scaled_dot_product_efficient_attention")
def test_sdpa_efficient(
    shape_id: str, dtype_id: str, bench: Bench, hw: Hardware, mojo_device: torch.device
) -> None:
    refs, ours, flops = _qkv(shape_id, dtype_id, hw, mojo_device)
    bench.run(
        lambda: torch.ops.aten._scaled_dot_product_efficient_attention(
            *refs, None, False, 0.0, True
        ),
        lambda: torch.ops.aten._scaled_dot_product_efficient_attention(
            *ours, None, False, 0.0, True
        ),
        flops=flops,
    )


@pytest.mark.parametrize("dtype_id", ("bf16", "f32"))
@pytest.mark.parametrize("shape_id", SHAPES)
@pytest.mark.bench_op("_scaled_dot_product_attention_math")
def test_sdpa_math(
    shape_id: str, dtype_id: str, bench: Bench, hw: Hardware, mojo_device: torch.device
) -> None:
    refs, ours, flops = _qkv(shape_id, dtype_id, hw, mojo_device)
    bench.run(
        lambda: torch.ops.aten._scaled_dot_product_attention_math(
            *refs, None, 0.0, True
        ),
        lambda: torch.ops.aten._scaled_dot_product_attention_math(
            *ours, None, 0.0, True
        ),
        flops=flops,
    )


@pytest.mark.parametrize("dtype_id", ("bf16", "f16"))
@pytest.mark.parametrize("shape_id", SHAPES)
@pytest.mark.bench_op("_scaled_dot_product_flash_attention_backward")
def test_sdpa_flash_backward(
    shape_id: str, dtype_id: str, bench: Bench, hw: Hardware, mojo_device: torch.device
) -> None:
    refs, ours, flops = _qkv(shape_id, dtype_id, hw, mojo_device)
    b, h, s, d = SHAPES[shape_id]
    g_ref, g_our = both(
        torch.randn(b, h, s, d, dtype=DTYPES[dtype_id]), hw, mojo_device
    )

    def forward(leg: list[torch.Tensor]) -> tuple:
        return torch.ops.aten._scaled_dot_product_flash_attention(
            *leg, 0.0, True, False
        )

    fwd_ref = forward(refs)
    try:
        # Un-timed setup outside bench.run's guard: skip, not error, when
        # the mojo device cannot run the forward this backward needs.
        fwd_our = forward(ours)
    except NotImplementedError as exc:
        pytest.skip(f"not supported on the mojo device: {exc}")

    def backward(grad: torch.Tensor, leg: list[torch.Tensor], fwd: tuple) -> tuple:
        out, logsumexp, cum_q, cum_k, max_q, max_k, seed, offset, _ = fwd
        return torch.ops.aten._scaled_dot_product_flash_attention_backward(
            grad,
            *leg,
            out,
            logsumexp,
            cum_q,
            cum_k,
            max_q,
            max_k,
            0.0,
            True,
            seed,
            offset,
        )

    bench.run(
        lambda: backward(g_ref, refs, fwd_ref),
        lambda: backward(g_our, ours, fwd_our),
        flops=2.5 * flops,
    )
