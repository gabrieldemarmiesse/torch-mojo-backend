"""Attention benchmarks: the public SDPA entry point plus each internal
kernel entry (flash / efficient / math) and the flash backward.

The internal ops are called via torch.ops.aten so the exact registered
entry point is pinned — the public F.scaled_dot_product_attention node
additionally measures whatever backend selection stock PyTorch performs.
Shapes are (batch, heads, seq, head_dim): a GPT-2-like block and a long
single-sequence decode-prefill regime.  All cases are causal.

GQA_SHAPES cover the enable_gqa=True regime: Q and K/V carry different
head counts, so they get their own shape dict keyed as
B{batch}H{q_heads}KV{kv_heads}S{seq}D{head_dim}.
"""

from __future__ import annotations

import copy
from collections.abc import Sequence

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
    "aten::_scaled_dot_product_flash_attention": "test_sdpa_flash",
    "aten::_scaled_dot_product_efficient_attention": "test_sdpa_efficient",
    "aten::_scaled_dot_product_flash_attention_backward": "test_sdpa_flash_backward",
    "aten::_scaled_dot_product_efficient_attention_backward": (
        "test_sdpa_efficient_backward"
    ),
    "aten::_flash_attention_forward": "test_flash_attention_forward",
    "aten::_efficient_attention_forward": "test_efficient_attention_forward",
    "aten::_scaled_dot_product_cudnn_attention": "test_sdpa_cudnn",
    "aten::_native_multi_head_attention": "test_native_multi_head_attention",
    "aten::_transform_bias_rescale_qkv": "test_transform_bias_rescale_qkv",
    "aten::_transformer_encoder_layer_fwd": "test_transformer_encoder_layer_fwd",
    "aten::_thnn_fused_lstm_cell": "test_fused_lstm_cell",
    "aten::_thnn_fused_lstm_cell_backward_impl": "test_fused_lstm_cell_backward",
    "aten::_thnn_fused_gru_cell": "test_fused_gru_cell",
    "aten::_thnn_fused_gru_cell_backward": "test_fused_gru_cell_backward",
}

_SAME_ROUTE = (
    "the math attention route test_sdpa_efficient_backward measures, behind "
    "a different argument layout"
)
SKIPPED: dict[str, str] = {
    "aten::_flash_attention_backward": _SAME_ROUTE,
    "aten::_efficient_attention_backward": _SAME_ROUTE,
    "aten::_scaled_dot_product_cudnn_attention_backward": _SAME_ROUTE,
    "aten::_cudnn_attention_backward": _SAME_ROUTE,
    "aten::_cudnn_attention_forward": (
        "the op test_sdpa_cudnn measures (_scaled_dot_product_cudnn_attention "
        "is a thin wrapper over it on CUDA, an alias here)"
    ),
    "aten::_flash_attention_forward_no_dropout_inplace": (
        "test_flash_attention_forward's route plus one copy into `out`"
    ),
}


def _qkv(
    shape_id: str, dtype_id: str, hw: Hardware, mojo: torch.device
) -> tuple[
    tuple[torch.Tensor, torch.Tensor, torch.Tensor],
    tuple[torch.Tensor, torch.Tensor, torch.Tensor],
    float,
]:
    b, h, s, d = SHAPES[shape_id]
    dtype = DTYPES[dtype_id]
    q_ref, q_our = both(torch.randn(b, h, s, d, dtype=dtype), hw, mojo)
    k_ref, k_our = both(torch.randn(b, h, s, d, dtype=dtype), hw, mojo)
    v_ref, v_our = both(torch.randn(b, h, s, d, dtype=dtype), hw, mojo)
    flops = 4.0 * b * h * s * s * d / 2.0  # causal halves the score matrix
    return (q_ref, k_ref, v_ref), (q_our, k_our, v_our), flops


# aten::_scaled_dot_product_flash_attention outputs (max_q / max_k are ints).
_FlashForwardOut = tuple[
    torch.Tensor,
    torch.Tensor,
    torch.Tensor,
    torch.Tensor,
    int,
    int,
    torch.Tensor,
    torch.Tensor,
    torch.Tensor,
]


@pytest.mark.parametrize("dtype_id", ("bf16", "f16"))
@pytest.mark.parametrize("shape_id", SHAPES)
@pytest.mark.bench_op("scaled_dot_product_attention")
def test_sdpa(
    shape_id: str, dtype_id: str, bench: Bench, hw: Hardware, mojo_device: torch.device
):
    refs, ours, flops = _qkv(shape_id, dtype_id, hw, mojo_device)
    bench.run(
        lambda: F.scaled_dot_product_attention(*refs, is_causal=True),
        lambda: F.scaled_dot_product_attention(*ours, is_causal=True),
        flops=flops,
    )


def _qkv_gqa(
    shape_id: str, dtype_id: str, hw: Hardware, mojo: torch.device
) -> tuple[
    tuple[torch.Tensor, torch.Tensor, torch.Tensor],
    tuple[torch.Tensor, torch.Tensor, torch.Tensor],
    float,
]:
    b, h, kv, s, d = GQA_SHAPES[shape_id]
    dtype = DTYPES[dtype_id]
    q_ref, q_our = both(torch.randn(b, h, s, d, dtype=dtype), hw, mojo)
    k_ref, k_our = both(torch.randn(b, kv, s, d, dtype=dtype), hw, mojo)
    v_ref, v_our = both(torch.randn(b, kv, s, d, dtype=dtype), hw, mojo)
    # FLOPs are driven by the query head count: each of the h query heads
    # still attends over the full (broadcast) K/V, same as the equal-head
    # case above -- the KV ratio changes memory traffic, not FLOPs.
    flops = 4.0 * b * h * s * s * d / 2.0  # causal halves the score matrix
    return (q_ref, k_ref, v_ref), (q_our, k_our, v_our), flops


@pytest.mark.parametrize("dtype_id", ("bf16", "f16"))
@pytest.mark.parametrize("shape_id", GQA_SHAPES)
@pytest.mark.bench_op("scaled_dot_product_attention")
def test_sdpa_gqa(
    shape_id: str, dtype_id: str, bench: Bench, hw: Hardware, mojo_device: torch.device
):
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
):
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
):
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
):
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
):
    refs, ours, flops = _qkv(shape_id, dtype_id, hw, mojo_device)
    b, h, s, d = SHAPES[shape_id]
    g_ref, g_our = both(
        torch.randn(b, h, s, d, dtype=DTYPES[dtype_id]), hw, mojo_device
    )

    def forward(leg: Sequence[torch.Tensor]) -> _FlashForwardOut:
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

    def backward(
        grad: torch.Tensor, leg: Sequence[torch.Tensor], fwd: _FlashForwardOut
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
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


@pytest.mark.parametrize("dtype_id", ("bf16",))
@pytest.mark.parametrize("shape_id", ("B8H12S1024D64",))
@pytest.mark.bench_op("_scaled_dot_product_efficient_attention_backward")
def test_sdpa_efficient_backward(
    shape_id: str, dtype_id: str, bench: Bench, hw: Hardware, mojo_device: torch.device
):
    refs, ours, flops = _qkv(shape_id, dtype_id, hw, mojo_device)
    b, h, s, d = SHAPES[shape_id]
    g_ref, g_our = both(
        torch.randn(b, h, s, d, dtype=DTYPES[dtype_id]), hw, mojo_device
    )
    aten = torch.ops.aten
    f_ref = aten._scaled_dot_product_efficient_attention(*refs, None, True, 0.0, True)
    f_our = aten._scaled_dot_product_efficient_attention(*ours, None, True, 0.0, True)

    def backward(
        g: torch.Tensor, leg: Sequence[torch.Tensor], f: Sequence[torch.Tensor]
    ) -> object:
        return aten._scaled_dot_product_efficient_attention_backward(
            g, *leg, None, f[0], f[1], f[2], f[3], 0.0, [True, True, True, False], True
        )

    bench.run(
        lambda: backward(g_ref, refs, f_ref),
        lambda: backward(g_our, ours, f_our),
        flops=2.5 * flops,
    )


@pytest.mark.parametrize("dtype_id", ("bf16",))
@pytest.mark.parametrize("shape_id", ("B8H12S1024D64",))
def test_flash_attention_forward(
    shape_id: str, dtype_id: str, bench: Bench, hw: Hardware, mojo_device: torch.device
):
    refs, ours, flops = _qkv(shape_id, dtype_id, hw, mojo_device)
    s = SHAPES[shape_id][2]
    aten = torch.ops.aten

    def run(leg: Sequence[torch.Tensor]) -> object:
        return aten._flash_attention_forward(
            *(t.transpose(1, 2) for t in leg), None, None, s, s, 0.0, True, False
        )

    bench.run(lambda: run(refs), lambda: run(ours), flops=flops)


@pytest.mark.parametrize("dtype_id", ("bf16", "f32"))
@pytest.mark.parametrize("shape_id", ("B8H12S1024D64",))
def test_efficient_attention_forward(
    shape_id: str, dtype_id: str, bench: Bench, hw: Hardware, mojo_device: torch.device
):
    refs, ours, flops = _qkv(shape_id, dtype_id, hw, mojo_device)
    aten = torch.ops.aten

    def run(leg: Sequence[torch.Tensor]) -> object:
        return aten._efficient_attention_forward(
            *(t.transpose(1, 2) for t in leg),
            None,
            None,
            None,
            None,
            None,
            0.0,
            1,
            True,
        )

    bench.run(lambda: run(refs), lambda: run(ours), flops=flops)


@pytest.mark.parametrize("dtype_id", ("bf16",))
@pytest.mark.parametrize("shape_id", ("B8H12S1024D64",))
@pytest.mark.bench_op("_scaled_dot_product_cudnn_attention")
def test_sdpa_cudnn(
    shape_id: str, dtype_id: str, bench: Bench, hw: Hardware, mojo_device: torch.device
):
    refs, ours, flops = _qkv(shape_id, dtype_id, hw, mojo_device)
    aten = torch.ops.aten
    bench.run(
        lambda: aten._scaled_dot_product_cudnn_attention(*refs, None, True, 0.0, True),
        lambda: aten._scaled_dot_product_cudnn_attention(*ours, None, True, 0.0, True),
        flops=flops,
    )


# (batch, seq, embed_dim, heads): a BERT-base-like encoder block.
TRANSFORMER_SHAPES: dict[str, tuple[int, int, int, int]] = {
    "B8T512E768H12": (8, 512, 768, 12)
}


@pytest.mark.parametrize("dtype_id", ("bf16", "f32"))
@pytest.mark.parametrize("shape_id", TRANSFORMER_SHAPES)
def test_native_multi_head_attention(
    shape_id: str, dtype_id: str, bench: Bench, hw: Hardware, mojo_device: torch.device
):
    b, t, e, h = TRANSFORMER_SHAPES[shape_id]
    dtype = DTYPES[dtype_id]
    x_ref, x_our = both(torch.randn(b, t, e, dtype=dtype), hw, mojo_device)
    ws = [torch.randn(3 * e, e, dtype=dtype) / e**0.5, torch.randn(3 * e, dtype=dtype)]
    ws += [torch.randn(e, e, dtype=dtype) / e**0.5, torch.randn(e, dtype=dtype)]
    w_ref = [w.to(hw.stock_device) for w in ws]
    w_our = [w.to(mojo_device) for w in ws]
    aten = torch.ops.aten
    bench.run(
        lambda: aten._native_multi_head_attention(
            x_ref, x_ref, x_ref, e, h, *w_ref, None, False
        ),
        lambda: aten._native_multi_head_attention(
            x_our, x_our, x_our, e, h, *w_our, None, False
        ),
        flops=8.0 * b * t * e * e + 4.0 * b * t * t * e,
    )


@pytest.mark.parametrize("dtype_id", ("bf16", "f32"))
@pytest.mark.parametrize("shape_id", TRANSFORMER_SHAPES)
def test_transform_bias_rescale_qkv(
    shape_id: str, dtype_id: str, bench: Bench, hw: Hardware, mojo_device: torch.device
):
    b, t, e, h = TRANSFORMER_SHAPES[shape_id]
    dtype = DTYPES[dtype_id]
    x_ref, x_our = both(torch.randn(b, t, 3 * e, dtype=dtype), hw, mojo_device)
    b_ref, b_our = both(torch.randn(3 * e, dtype=dtype), hw, mojo_device)
    aten = torch.ops.aten
    bench.run(
        lambda: aten._transform_bias_rescale_qkv(x_ref, b_ref, h),
        lambda: aten._transform_bias_rescale_qkv(x_our, b_our, h),
        flops=2.0 * b * t * 3 * e,
    )


@pytest.mark.parametrize("dtype_id", ("bf16", "f32"))
@pytest.mark.parametrize("shape_id", TRANSFORMER_SHAPES)
def test_transformer_encoder_layer_fwd(
    shape_id: str, dtype_id: str, bench: Bench, hw: Hardware, mojo_device: torch.device
):
    b, t, e, h = TRANSFORMER_SHAPES[shape_id]
    dtype = DTYPES[dtype_id]
    layer = torch.nn.TransformerEncoderLayer(e, h, 4 * e, batch_first=True).eval()
    layer = layer.to(dtype)
    ours = copy.deepcopy(layer).to(mojo_device)
    ref = layer.to(hw.stock_device)
    x_ref, x_our = both(torch.randn(b, t, e, dtype=dtype), hw, mojo_device)

    def run(mod: torch.nn.Module, x: torch.Tensor) -> torch.Tensor:
        with torch.no_grad():
            return mod(x)

    bench.run(
        lambda: run(ref, x_ref),
        lambda: run(ours, x_our),
        flops=24.0 * b * t * e * e + 4.0 * b * t * t * e,
    )


# (batch, hidden): one recurrent step of a 1024-wide LSTM/GRU layer.
RNN_SHAPES: dict[str, tuple[int, int]] = {
    "N64H1024": (64, 1024),
    "N357H789": (357, 789),
}


def _rnn_operands(
    shape_id: str, dtype_id: str, gates: int, hw: Hardware, mojo: torch.device
) -> list[tuple[torch.Tensor, torch.Tensor]]:
    """(input_gates, hidden_gates, cx/hx, input_bias, hidden_bias), both legs."""
    n, h = RNN_SHAPES[shape_id]
    dtype = DTYPES[dtype_id]
    ops = [both(torch.randn(n, gates * h, dtype=dtype), hw, mojo) for _ in range(2)]
    ops.append(both(torch.randn(n, h, dtype=dtype), hw, mojo))
    ops += [both(torch.randn(gates * h, dtype=dtype), hw, mojo) for _ in range(2)]
    return ops


@pytest.mark.parametrize("dtype_id", ("bf16", "f32"))
@pytest.mark.parametrize("shape_id", RNN_SHAPES)
@pytest.mark.bench_op("_thnn_fused_lstm_cell")
def test_fused_lstm_cell(
    shape_id: str, dtype_id: str, bench: Bench, hw: Hardware, mojo_device: torch.device
):
    ops = _rnn_operands(shape_id, dtype_id, 4, hw, mojo_device)
    fn = torch.ops.aten._thnn_fused_lstm_cell
    n, h = RNN_SHAPES[shape_id]
    bench.run(
        lambda: fn(*(r for r, _ in ops)),
        lambda: fn(*(o for _, o in ops)),
        flops=40.0 * n * h,
    )


@pytest.mark.parametrize("dtype_id", ("bf16", "f32"))
@pytest.mark.parametrize("shape_id", RNN_SHAPES)
@pytest.mark.bench_op("_thnn_fused_lstm_cell_backward_impl")
def test_fused_lstm_cell_backward(
    shape_id: str, dtype_id: str, bench: Bench, hw: Hardware, mojo_device: torch.device
):
    ops = _rnn_operands(shape_id, dtype_id, 4, hw, mojo_device)
    fn = torch.ops.aten._thnn_fused_lstm_cell
    _, cy_ref, ws_ref = fn(*(r for r, _ in ops))
    _, cy_our, ws_our = fn(*(o for _, o in ops))
    g_ref, g_our = ops[2]
    bwd = torch.ops.aten._thnn_fused_lstm_cell_backward_impl
    n, h = RNN_SHAPES[shape_id]
    bench.run(
        lambda: bwd(g_ref, g_ref, g_ref, cy_ref, ws_ref, True),
        lambda: bwd(g_our, g_our, g_our, cy_our, ws_our, True),
        flops=40.0 * n * h,
    )


@pytest.mark.parametrize("dtype_id", ("bf16", "f32"))
@pytest.mark.parametrize("shape_id", RNN_SHAPES)
@pytest.mark.bench_op("_thnn_fused_gru_cell")
def test_fused_gru_cell(
    shape_id: str, dtype_id: str, bench: Bench, hw: Hardware, mojo_device: torch.device
):
    ops = _rnn_operands(shape_id, dtype_id, 3, hw, mojo_device)
    fn = torch.ops.aten._thnn_fused_gru_cell
    n, h = RNN_SHAPES[shape_id]
    bench.run(
        lambda: fn(*(r for r, _ in ops)),
        lambda: fn(*(o for _, o in ops)),
        flops=30.0 * n * h,
    )


@pytest.mark.parametrize("dtype_id", ("bf16", "f32"))
@pytest.mark.parametrize("shape_id", RNN_SHAPES)
@pytest.mark.bench_op("_thnn_fused_gru_cell_backward")
def test_fused_gru_cell_backward(
    shape_id: str, dtype_id: str, bench: Bench, hw: Hardware, mojo_device: torch.device
):
    ops = _rnn_operands(shape_id, dtype_id, 3, hw, mojo_device)
    fn = torch.ops.aten._thnn_fused_gru_cell
    _, ws_ref = fn(*(r for r, _ in ops))
    _, ws_our = fn(*(o for _, o in ops))
    g_ref, g_our = ops[2]
    bwd = torch.ops.aten._thnn_fused_gru_cell_backward
    n, h = RNN_SHAPES[shape_id]
    bench.run(
        lambda: bwd(g_ref, ws_ref, True),
        lambda: bwd(g_our, ws_our, True),
        flops=30.0 * n * h,
    )
