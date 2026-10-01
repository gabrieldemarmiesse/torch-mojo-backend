"""Attention on the native mojo device: the flash ops (FA4), the fused
inference forward (decode + math), and the backend choice.

Everything goes through public torch APIs plus the ATen ops themselves --
`F.scaled_dot_product_attention` is CompositeImplicitAutograd, so the ops
registered here are the ones ATen calls underneath, and calling them directly
is what a test can pin. `native.op_count` asserts the native op really ran.
"""

import pytest
import torch
import torch.nn.functional as F

from torch_mojo_backend import get_accelerators, native, register_mojo_devices

aten = torch.ops.aten

# at::SDPBackend
MATH = 0
FLASH = 1
EFFICIENT = 2

# FA4 runs on Hopper only; every other GPU declines every flash route.
_HOPPER_ONLY = "the FA4 kernels are compiled for sm_90a"
# The fused MFMA flash forward/backward exists on CDNA3 only.
_GFX942_ONLY = "the fused MFMA flash kernels are compiled for gfx942"


def _grad(t: torch.Tensor) -> torch.Tensor:
    assert t.grad is not None
    return t.grad


def _counted(name: str) -> int:
    return native.op_count(f"aten::{name}")


@pytest.fixture
def counting():
    register_mojo_devices()  # idempotent; the op counters live in the shim
    native.op_counting(True)
    native.op_counts_reset()
    yield


def _arch(device: str) -> str:
    return get_accelerators()[int(device.split(":")[1])].architecture_name


def _qkv(
    device: str,
    dtype: torch.dtype,
    batch: int,
    heads: int,
    q_len: int,
    kv_len: int,
    head_dim: int,
    seed: int = 0,
) -> tuple[torch.Tensor, ...]:
    """Matching CPU-float32 and device tensors (the CPU ones are the rounded
    values the device actually sees, so the reference is exact)."""
    gen = torch.Generator().manual_seed(seed)
    shapes = (
        (batch, heads, q_len, head_dim),
        (batch, heads, kv_len, head_dim),
        (batch, heads, kv_len, head_dim),
    )
    ref = [torch.randn(s, generator=gen).to(dtype).float() for s in shapes]
    dev = [t.to(dtype).to(device) for t in ref]
    return (*ref, *dev)


def _tol(dtype: torch.dtype) -> float:
    """One absolute-and-relative tolerance against a float32 CPU reference
    computed from the same rounded inputs."""
    if dtype == torch.bfloat16:
        return 3e-2
    if dtype == torch.float16:
        return 5e-3
    return 1e-4


def _grad_tol(dtype: torch.dtype) -> float:
    """Same, for gradients: the backward rounds P and dS to `dtype` before its
    second GEMMs, so it carries about twice the forward's error."""
    if dtype == torch.bfloat16:
        return 6e-2
    if dtype == torch.float16:
        return 1e-2
    return 1e-4


# --------------------------------------------------------------------------
# _scaled_dot_product_flash_attention (FA4)
# --------------------------------------------------------------------------


@pytest.mark.parametrize("dtype", [torch.bfloat16, torch.float16])
@pytest.mark.parametrize("head_dim", [64, 128])
def test_flash_attention_forward(mojo_gpu, counting, dtype, head_dim):
    if _arch(mojo_gpu) != "sm_90a":
        pytest.skip(_HOPPER_ONLY)
    qr, kr, vr, q, k, v = _qkv(mojo_gpu, dtype, 2, 3, 256, 256, head_dim)
    out, lse = aten._scaled_dot_product_flash_attention(q, k, v, 0.0, True)[:2]
    assert _counted("_scaled_dot_product_flash_attention") == 1
    expected = F.scaled_dot_product_attention(qr, kr, vr, is_causal=True)
    torch.testing.assert_close(
        out.contiguous().cpu().float(), expected, atol=_tol(dtype), rtol=_tol(dtype)
    )
    # logsumexp is real FP32 data the backward consumes, never a placeholder.
    scores = (qr @ kr.transpose(-1, -2)) / head_dim**0.5
    mask = torch.ones(256, 256, dtype=torch.bool).tril()
    scores = scores.masked_fill(~mask, float("-inf"))
    torch.testing.assert_close(
        lse.cpu(), torch.logsumexp(scores, dim=-1), atol=3e-2, rtol=3e-2
    )


def test_flash_attention_partial_tail_block(mojo_gpu, counting):
    """A seqlen that is not a multiple of 128 is only served by the
    BHSD-native route, which zero-fills and clamps the last tile."""
    if _arch(mojo_gpu) != "sm_90a":
        pytest.skip(_HOPPER_ONLY)
    qr, kr, vr, q, k, v = _qkv(mojo_gpu, torch.bfloat16, 1, 2, 200, 200, 64)
    out = aten._scaled_dot_product_flash_attention(q, k, v, 0.0, True)[0]
    assert _counted("_scaled_dot_product_flash_attention") == 1
    torch.testing.assert_close(
        out.contiguous().cpu().float(),
        F.scaled_dot_product_attention(qr, kr, vr, is_causal=True),
        atol=_tol(torch.bfloat16),
        rtol=_tol(torch.bfloat16),
    )


def test_flash_attention_strided_qkv(mojo_gpu, counting):
    """A fused qkv projection hands over gapped (B, S, H, D) views; FA4 reads
    them through its strided TMA ABI instead of materializing three copies."""
    if _arch(mojo_gpu) != "sm_90a":
        pytest.skip(_HOPPER_ONLY)
    gen = torch.Generator().manual_seed(7)
    fused = torch.randn(2, 128, 3, 4, 64, generator=gen).to(torch.bfloat16)
    ref = [fused[:, :, i].float().transpose(1, 2) for i in range(3)]
    dev = fused.to(mojo_gpu)
    q, k, v = (dev[:, :, i].transpose(1, 2) for i in range(3))
    assert not q.is_contiguous()
    out = aten._scaled_dot_product_flash_attention(q, k, v, 0.0, True)[0]
    assert _counted("_scaled_dot_product_flash_attention") == 1
    torch.testing.assert_close(
        out.contiguous().cpu().float(),
        F.scaled_dot_product_attention(ref[0], ref[1], ref[2], is_causal=True),
        atol=_tol(torch.bfloat16),
        rtol=_tol(torch.bfloat16),
    )


@pytest.mark.parametrize("head_dim", [64, 128])
def test_flash_attention_backward(mojo_gpu, counting, head_dim):
    """ATen's own autograd formula ties the forward to
    `_scaled_dot_product_flash_attention_backward`; no Python node involved."""
    if _arch(mojo_gpu) != "sm_90a":
        pytest.skip(_HOPPER_ONLY)
    qr, kr, vr, q, k, v = _qkv(mojo_gpu, torch.bfloat16, 2, 2, 128, 128, head_dim)
    for t in (qr, kr, vr):
        t.requires_grad_()
    for t in (q, k, v):
        t.requires_grad_()
    out = aten._scaled_dot_product_flash_attention(q, k, v, 0.0, True)[0]
    grad = torch.ones_like(out)
    out.backward(grad)
    assert _counted("_scaled_dot_product_flash_attention_backward") == 1
    F.scaled_dot_product_attention(qr, kr, vr, is_causal=True).backward(
        torch.ones_like(qr)
    )
    for got, want in ((q, qr), (k, kr), (v, vr)):
        assert got.grad is not None and want.grad is not None
        torch.testing.assert_close(
            _grad(got).contiguous().cpu().float(), _grad(want), atol=6e-2, rtol=6e-2
        )


def test_flash_attention_float32_takes_the_math_route(mojo_gpu, counting):
    """No fused kernel takes float32 (outside gfx942), so the flash op runs
    the math route -- with a real logsumexp the backward can use."""
    qr, kr, vr, q, k, v = _qkv(mojo_gpu, torch.float32, 1, 2, 24, 24, 16)
    out, lse = aten._scaled_dot_product_flash_attention(q, k, v, 0.0, True)[:2]
    expected = F.scaled_dot_product_attention(qr, kr, vr, is_causal=True)
    torch.testing.assert_close(out.cpu(), expected, atol=1e-5, rtol=1e-5)
    scores = (qr @ kr.transpose(-1, -2)) / 4
    scores = scores.masked_fill(
        ~torch.ones(24, 24, dtype=torch.bool).tril(), -torch.inf
    )
    torch.testing.assert_close(lse.cpu(), torch.logsumexp(scores, -1))


def test_flash_attention_declines_dropout(mojo_gpu, counting):
    _, _, _, q, k, v = _qkv(mojo_gpu, torch.bfloat16, 1, 1, 128, 128, 64)
    with pytest.raises(NotImplementedError):
        aten._scaled_dot_product_flash_attention(q, k, v, 0.5, True)


def test_flash_attention_backward_partial_tail_takes_the_math_route(mojo_gpu, counting):
    """The FA4 backward tile machinery is unproven on a partial last tile,
    so an odd seqlen that the BHSD forward accepts gets its gradient from
    the math route (which recomputes the probabilities) instead."""
    if _arch(mojo_gpu) != "sm_90a":
        pytest.skip(_HOPPER_ONLY)
    qr, kr, vr, q, k, v = _qkv(mojo_gpu, torch.bfloat16, 1, 2, 200, 200, 64)
    for t in (qr, kr, vr, q, k, v):
        t.requires_grad_()
    out = aten._scaled_dot_product_flash_attention(q, k, v, 0.0, True)[0]
    out.backward(torch.ones_like(out))
    F.scaled_dot_product_attention(qr, kr, vr, is_causal=True).backward(
        torch.ones_like(qr)
    )
    for got, want in ((q, qr), (k, kr), (v, vr)):
        torch.testing.assert_close(
            _grad(got).cpu().float(), _grad(want), atol=6e-2, rtol=6e-2
        )


@pytest.mark.parametrize("is_causal", [False, True])
@pytest.mark.parametrize("dtype", [torch.bfloat16, torch.float16])
@pytest.mark.parametrize("shape", [(1, 2, 200, 64), (1, 2, 8, 8)])
def test_fused_flash_backward_partial_tail_gfx942(
    mojo_gpu, counting, shape, dtype, is_causal
):
    """The gfx942 MFMA backward accepts any seqlen. Its dK/dV kernel runs a
    partial last query tile (and a whole sequence shorter than one tile)
    through its masked path, which used to apply the causal bound even for
    non-causal attention and silently dropped every q < kv term of the tail
    keys (regression: 1x2x8x8 non-causal dK was off by 1.4 on a 1.8 range)."""
    if _arch(mojo_gpu) != "gfx942":
        pytest.skip(_GFX942_ONLY)
    batch, heads, seq, head_dim = shape
    qr, kr, vr, q, k, v = _qkv(mojo_gpu, dtype, batch, heads, seq, seq, head_dim)
    for t in (qr, kr, vr, q, k, v):
        t.requires_grad_()
    out = aten._scaled_dot_product_flash_attention(q, k, v, 0.0, is_causal)[0]
    out.backward(torch.ones_like(out))
    assert _counted("_scaled_dot_product_flash_attention_backward") == 1
    F.scaled_dot_product_attention(qr, kr, vr, is_causal=is_causal).backward(
        torch.ones_like(qr)
    )
    tol = _grad_tol(dtype)
    for got, want in ((q, qr), (k, kr), (v, vr)):
        assert got.grad is not None and want.grad is not None
        torch.testing.assert_close(
            _grad(got).contiguous().cpu().float(), _grad(want), atol=tol, rtol=tol
        )


# --------------------------------------------------------------------------
# _scaled_dot_product_efficient_attention (decode + math)
# --------------------------------------------------------------------------


@pytest.mark.parametrize("dtype", [torch.float32, torch.bfloat16])
def test_efficient_attention_decode_step(mojo_gpu, counting, dtype):
    """q_len == 1: one fused kernel instead of bmm + softmax + bmm."""
    qr, kr, vr, q, k, v = _qkv(mojo_gpu, dtype, 2, 4, 1, 130, 64)
    with torch.no_grad():
        out = aten._scaled_dot_product_efficient_attention(
            q, k, v, None, False, 0.0, False
        )[0]
    assert _counted("_scaled_dot_product_efficient_attention") == 1
    torch.testing.assert_close(
        out.contiguous().cpu().float(),
        F.scaled_dot_product_attention(qr, kr, vr),
        atol=_tol(dtype),
        rtol=_tol(dtype),
    )


@pytest.mark.parametrize("is_causal", [False, True])
@pytest.mark.parametrize("dtype", [torch.float32, torch.bfloat16])
def test_efficient_attention_math(mojo_device, counting, dtype, is_causal):
    """bmm + fused scale/causal softmax + bmm, no mask tensor materialized."""
    qr, kr, vr, q, k, v = _qkv(mojo_device, dtype, 2, 3, 17, 17, 40, seed=3)
    with torch.no_grad():
        out = aten._scaled_dot_product_efficient_attention(
            q, k, v, None, False, 0.0, is_causal
        )[0]
    assert _counted("_scaled_dot_product_efficient_attention") == 1
    torch.testing.assert_close(
        out.contiguous().cpu().float(),
        F.scaled_dot_product_attention(qr, kr, vr, is_causal=is_causal),
        atol=_tol(dtype),
        rtol=_tol(dtype),
    )


def test_efficient_attention_custom_scale(mojo_device, counting):
    qr, kr, vr, q, k, v = _qkv(mojo_device, torch.float32, 1, 2, 9, 12, 8)
    with torch.no_grad():
        out = aten._scaled_dot_product_efficient_attention(
            q, k, v, None, False, 0.0, False, scale=0.25
        )[0]
    torch.testing.assert_close(
        out.contiguous().cpu(),
        F.scaled_dot_product_attention(qr, kr, vr, scale=0.25),
        atol=_tol(torch.float32),
        rtol=_tol(torch.float32),
    )


def test_efficient_attention_trains(mojo_gpu, counting):
    """A grad-requiring call records `_scaled_dot_product_efficient_attention
    _backward`, which runs the math route (it recomputes the probabilities,
    so it needs no logsumexp from the forward)."""
    qr, kr, vr, q, k, v = _qkv(mojo_gpu, torch.float32, 1, 2, 8, 8, 8)
    for t in (qr, kr, vr, q, k, v):
        t.requires_grad_()
    for compute_lse in (False, True):
        out = aten._scaled_dot_product_efficient_attention(
            q, k, v, None, compute_lse, 0.0, True
        )[0]
        out.sum().backward()
    assert _counted("_scaled_dot_product_efficient_attention_backward") == 2
    F.scaled_dot_product_attention(qr, kr, vr, is_causal=True).sum().backward()
    for got, want in ((q, qr), (k, kr), (v, vr)):
        torch.testing.assert_close(
            _grad(got).cpu(), 2 * _grad(want), atol=1e-5, rtol=1e-5
        )


def test_efficient_attention_compute_log_sumexp(mojo_gpu, counting):
    """CUDA's memory-efficient layout: (B, H, ceil(L / 32) * 32), padded with
    +inf; (B, H, 0) when not asked for."""
    qr, kr, vr, q, k, v = _qkv(mojo_gpu, torch.float32, 1, 2, 5, 7, 8)
    with torch.no_grad():
        lse = aten._scaled_dot_product_efficient_attention(
            q, k, v, None, True, 0.0, False
        )[1]
        none = aten._scaled_dot_product_efficient_attention(
            q, k, v, None, False, 0.0, False
        )[1]
    assert lse.shape == (1, 2, 32) and none.shape == (1, 2, 0)
    want = torch.logsumexp((qr @ kr.transpose(-1, -2)) / 8**0.5, -1)
    torch.testing.assert_close(lse[..., :5].cpu(), want)
    assert torch.isinf(lse[..., 5:]).all() and (lse[..., 5:] > 0).all()


@pytest.mark.parametrize("dtype", [torch.float16, torch.bfloat16])
def test_math_attention_does_not_overflow_reduced_precision(mojo_gpu, dtype):
    """q @ k^T accumulates head_dim products: with q, k of magnitude 100 and
    head_dim 64 the scores reach 6.4e5, well past float16's 65504 ceiling.
    The scores and the softmax therefore run in float32."""
    shape = (1, 1, 8, 64)
    qr = torch.full(shape, 100.0, dtype=dtype)
    kr = torch.full(shape, 100.0, dtype=dtype)
    vr = torch.arange(8 * 64, dtype=torch.float32).reshape(shape).to(dtype) / 64.0
    q, k, v = qr.to(mojo_gpu), kr.to(mojo_gpu), vr.to(mojo_gpu)
    with torch.no_grad():
        out = aten._scaled_dot_product_efficient_attention(
            q, k, v, None, False, 0.0, False
        )[0]
    assert torch.isfinite(out.cpu().float()).all()
    torch.testing.assert_close(
        out.contiguous().cpu().float(),
        F.scaled_dot_product_attention(qr.float(), kr.float(), vr.float()),
        atol=1e-1,
        rtol=1e-1,
    )


@pytest.mark.parametrize("dtype", [torch.float32, torch.bfloat16])
def test_efficient_attention_bias_and_its_gradient(mojo_gpu, counting, dtype):
    qr, kr, vr, q, k, v = _qkv(mojo_gpu, dtype, 2, 2, 6, 9, 8)
    gen = torch.Generator().manual_seed(5)
    br = torch.randn(2, 1, 6, 9, generator=gen).to(dtype).float()
    b = br.to(dtype).to(mojo_gpu)
    for t in (qr, kr, vr, br, q, k, v, b):
        t.requires_grad_()
    out = aten._scaled_dot_product_efficient_attention(q, k, v, b, True, 0.0, False)[0]
    want = F.scaled_dot_product_attention(qr, kr, vr, attn_mask=br)
    torch.testing.assert_close(
        out.cpu().float(), want, atol=_tol(dtype), rtol=_tol(dtype)
    )
    g = torch.randn(want.shape, generator=gen)
    out.backward(g.to(dtype).to(mojo_gpu))
    want.backward(g)
    tol = _grad_tol(dtype)
    for got, ref in ((q, qr), (k, kr), (v, vr), (b, br)):
        assert _grad(got).shape == _grad(ref).shape
        torch.testing.assert_close(
            _grad(got).cpu().float(), _grad(ref), atol=tol, rtol=tol
        )


# --------------------------------------------------------------------------
# Grouped-query attention (enable_gqa=True)
# --------------------------------------------------------------------------


def _gqa_qkv(
    device: str,
    dtype: torch.dtype,
    q_heads: int,
    kv_heads: int,
    q_len: int,
    kv_len: int,
    head_dim: int = 64,
) -> tuple[torch.Tensor, ...]:
    """Like `_qkv`, with K/V carrying `kv_heads` heads instead of Q's."""
    gen = torch.Generator().manual_seed(3)
    shapes = (
        (2, q_heads, q_len, head_dim),
        (2, kv_heads, kv_len, head_dim),
        (2, kv_heads, kv_len, head_dim),
    )
    ref = [torch.randn(s, generator=gen).to(dtype).float() for s in shapes]
    return (*ref, *(t.to(dtype).to(device) for t in ref))


_GQA_HEADS = [(8, 2), (8, 1), (4, 4)]
_GQA_IDS = ["gqa8to2", "mqa8to1", "equal4"]


@pytest.mark.parametrize("is_causal", [True, False], ids=["causal", "full"])
@pytest.mark.parametrize("dtype", [torch.float32, torch.bfloat16])
@pytest.mark.parametrize("q_heads,kv_heads", _GQA_HEADS, ids=_GQA_IDS)
def test_sdpa_enable_gqa_inference(
    mojo_gpu, counting, dtype, q_heads, kv_heads, is_causal
):
    """An odd seqlen (257) so FA4 takes it through its BHSD tail route."""
    qr, kr, vr, q, k, v = _gqa_qkv(mojo_gpu, dtype, q_heads, kv_heads, 257, 257)
    with torch.no_grad():
        out = F.scaled_dot_product_attention(
            q, k, v, is_causal=is_causal, enable_gqa=True
        )
    if kv_heads != q_heads:
        assert _counted("_scaled_dot_product_efficient_attention") == 1
    assert _counted("clone") == 0  # no ATen repeat_interleave
    expected = F.scaled_dot_product_attention(
        qr, kr, vr, is_causal=is_causal, enable_gqa=True
    )
    torch.testing.assert_close(
        out.cpu().float(), expected, atol=_tol(dtype), rtol=_tol(dtype)
    )


@pytest.mark.parametrize("q_heads,kv_heads", [(8, 2), (8, 1)], ids=_GQA_IDS[:2])
def test_sdpa_enable_gqa_decode_step(mojo_gpu, counting, q_heads, kv_heads):
    qr, kr, vr, q, k, v = _gqa_qkv(mojo_gpu, torch.float32, q_heads, kv_heads, 1, 130)
    with torch.no_grad():
        out = F.scaled_dot_product_attention(q, k, v, enable_gqa=True)
    assert _counted("_scaled_dot_product_efficient_attention") == 1
    torch.testing.assert_close(
        out.cpu(),
        F.scaled_dot_product_attention(qr, kr, vr, enable_gqa=True),
        atol=1e-4,
        rtol=1e-4,
    )


def test_efficient_attention_repeats_strided_gqa_kv(mojo_gpu, counting):
    """K/V handed over as (B, S, H, D) storage viewed (B, H, S, D): the
    head repeat reads them through their own strides."""
    gen = torch.Generator().manual_seed(5)
    q_ref = torch.randn(1, 6, 40, 32, generator=gen)
    kv_ref = torch.randn(1, 40, 2, 2, 32, generator=gen)
    kv = kv_ref.to(mojo_gpu)
    k, v = kv[:, :, 0].transpose(1, 2), kv[:, :, 1].transpose(1, 2)
    assert not k.is_contiguous()
    out = aten._scaled_dot_product_efficient_attention(
        q_ref.to(mojo_gpu), k, v, None, False, 0.0, True
    )[0]
    expected = F.scaled_dot_product_attention(
        q_ref,
        kv_ref[:, :, 0].transpose(1, 2),
        kv_ref[:, :, 1].transpose(1, 2),
        is_causal=True,
        enable_gqa=True,
    )
    torch.testing.assert_close(out.cpu(), expected, atol=1e-4, rtol=1e-4)


def test_sdpa_enable_gqa_trains_through_math(mojo_gpu):
    """Under autograd GQA stays on ATen's math decomposition, whose
    repeat_interleave backward folds each query group's gradient back onto
    its KV head."""
    qr, kr, vr, q, k, v = _gqa_qkv(mojo_gpu, torch.float32, 4, 2, 24, 24, 16)
    for t in (qr, kr, vr, q, k, v):
        t.requires_grad_()
    F.scaled_dot_product_attention(
        q, k, v, is_causal=True, enable_gqa=True
    ).sum().backward()
    F.scaled_dot_product_attention(
        qr, kr, vr, is_causal=True, enable_gqa=True
    ).sum().backward()
    for got, want in ((q, qr), (k, kr), (v, vr)):
        assert got.grad is not None and want.grad is not None
        assert _grad(got).shape == _grad(want).shape
        torch.testing.assert_close(_grad(got).cpu(), _grad(want), atol=1e-4, rtol=1e-4)


def test_sdpa_enable_gqa_indivisible_heads_raise(mojo_gpu):
    _, _, _, q, k, v = _gqa_qkv(mojo_gpu, torch.float32, 6, 4, 8, 8, 16)
    with torch.no_grad(), pytest.raises(RuntimeError, match="divide"):
        F.scaled_dot_product_attention(q, k, v, enable_gqa=True)


# --------------------------------------------------------------------------
# _fused_sdp_choice
# --------------------------------------------------------------------------


def test_fused_sdp_choice_flash(mojo_gpu, counting):
    if _arch(mojo_gpu) != "sm_90a":
        pytest.skip(_HOPPER_ONLY)
    _, _, _, q, k, v = _qkv(mojo_gpu, torch.bfloat16, 1, 2, 128, 128, 64)
    assert aten._fused_sdp_choice(q, k, v, None, 0.0, True) == FLASH
    assert _counted("_fused_sdp_choice") == 1


def test_fused_sdp_choice_efficient_for_inference(mojo_gpu, counting):
    _, _, _, q, k, v = _qkv(mojo_gpu, torch.float32, 1, 2, 1, 64, 64)
    # gfx942's fused flash route takes float32 and q_len 1; nothing else does.
    want = FLASH if _arch(mojo_gpu) == "gfx942" else EFFICIENT
    assert aten._fused_sdp_choice(q, k, v, None, 0.0, False) == want


def test_fused_sdp_choice_math_when_grad_is_needed(mojo_gpu, counting):
    """Training goes to the differentiable decomposition: the efficient op's
    backward here is that same decomposition, so it would buy nothing."""
    _, _, _, q, k, v = _qkv(mojo_gpu, torch.float32, 1, 2, 8, 8, 8)
    q.requires_grad_()
    # gfx942's fused flash route has a float32 backward; nothing else does.
    want = FLASH if _arch(mojo_gpu) == "gfx942" else MATH
    assert aten._fused_sdp_choice(q, k, v, None, 0.0, False) == want


def test_fused_sdp_choice_math_for_masked_and_dropout(mojo_gpu, counting):
    _, _, _, q, k, v = _qkv(mojo_gpu, torch.bfloat16, 1, 2, 128, 128, 64)
    mask = torch.zeros(1, 1, 128, 128, dtype=torch.bfloat16, device=mojo_gpu)
    assert aten._fused_sdp_choice(q, k, v, mask, 0.0, False) == MATH
    assert aten._fused_sdp_choice(q, k, v, None, 0.25, True) == MATH


def test_fused_sdp_choice_gqa(mojo_gpu, counting):
    """Unequal heads: the efficient op for inference, math under autograd
    or when the heads do not divide. Equal heads ignore enable_gqa."""
    _, _, _, q, k, v = _gqa_qkv(mojo_gpu, torch.bfloat16, 8, 2, 128, 128)
    assert aten._fused_sdp_choice(q, k, v, None, 0.0, True, enable_gqa=True) == (
        EFFICIENT
    )
    q.requires_grad_()
    assert aten._fused_sdp_choice(q, k, v, None, 0.0, True, enable_gqa=True) == MATH
    _, _, _, q, k, v = _gqa_qkv(mojo_gpu, torch.bfloat16, 6, 4, 128, 128)
    assert aten._fused_sdp_choice(q, k, v, None, 0.0, True, enable_gqa=True) == MATH
    _, _, _, q, k, v = _qkv(mojo_gpu, torch.bfloat16, 1, 2, 128, 128, 64)
    assert aten._fused_sdp_choice(
        q, k, v, None, 0.0, True, enable_gqa=True
    ) == aten._fused_sdp_choice(q, k, v, None, 0.0, True)


def test_public_sdpa_takes_the_supported_route_and_trains(mojo_gpu: str):
    """F.scaled_dot_product_attention picks its backend through a C++
    DispatchStub the shim registers for this device, then calls the
    supported flash overloads on Hopper/gfx942 and math attention on Metal."""
    native.op_counting(True)
    before = native.op_count("aten::_scaled_dot_product_flash_attention_for_cpu")
    before_bwd = native.op_count(
        "aten::_scaled_dot_product_flash_attention_for_cpu_backward"
    )
    q = torch.randn(
        2, 4, 128, 64, dtype=torch.bfloat16, device=mojo_gpu, requires_grad=True
    )
    k = torch.randn_like(q, requires_grad=True)
    v = torch.randn_like(q, requires_grad=True)
    out = F.scaled_dot_product_attention(q, k, v, is_causal=True)
    out.float().sum().backward()
    # Metal uses math attention; the flash kernels support Hopper and gfx942.
    flash_calls = int(_arch(mojo_gpu) in ("sm_90a", "gfx942"))
    assert (
        native.op_count("aten::_scaled_dot_product_flash_attention_for_cpu")
        == before + flash_calls
    )
    assert (
        native.op_count("aten::_scaled_dot_product_flash_attention_for_cpu_backward")
        == before_bwd + flash_calls
    )
    ref_q = q.detach().cpu().float().requires_grad_(True)
    ref_k = k.detach().cpu().float().requires_grad_(True)
    ref_v = v.detach().cpu().float().requires_grad_(True)
    ref = F.scaled_dot_product_attention(ref_q, ref_k, ref_v, is_causal=True)
    ref.sum().backward()
    assert q.grad is not None and ref_q.grad is not None
    torch.testing.assert_close(out.cpu().float(), ref, atol=2e-2, rtol=2e-2)
    torch.testing.assert_close(q.grad.cpu().float(), ref_q.grad, atol=5e-2, rtol=5e-2)


# --------------------------------------------------------------------------
# The (B, S, H, D)-layout CUDA entry points and cuDNN's: the math route
# whenever no fused kernel takes the inputs. Expected values (layouts,
# causal alignment, fully masked rows) were measured on stock CUDA torch.
# --------------------------------------------------------------------------


def _ref_attention(qr, kr, vr, bias=None, offset=None):
    """(out, lse) in float64; `offset` is the causal diagonal (key j is
    visible to query i iff j <= i + offset)."""
    s = (qr.double() @ kr.double().transpose(-1, -2)) / qr.shape[-1] ** 0.5
    if bias is not None:
        s = s + bias.double()
    if offset is not None:
        keep = torch.ones(s.shape[-2], s.shape[-1], dtype=torch.bool).tril(offset)
        s = s.masked_fill(~keep, -torch.inf)
    lse = torch.logsumexp(s.detach(), -1)
    # A fully masked row attends to nothing: output 0 and no gradient (a
    # plain softmax would put NaN there and spread it through the GEMMs).
    dead = (lse == -torch.inf).unsqueeze(-1)
    p = torch.softmax(s.masked_fill(dead, 0.0), -1).masked_fill(dead, 0.0)
    return p @ vr.double(), lse


def _bshd(*ts):
    return [t.transpose(1, 2) for t in ts]


@pytest.mark.parametrize("dtype", [torch.float32, torch.float16, torch.bfloat16])
@pytest.mark.parametrize("is_causal", [False, True])
@pytest.mark.parametrize("lengths", [(5, 7), (7, 5)])
def test_flash_attention_forward_bshd(mojo_gpu, counting, dtype, is_causal, lengths):
    """CUDA's flash kernels align the causal mask bottom-right; a fully
    masked query row (L > S) has output 0 and logsumexp +inf."""
    lq, lk = lengths
    qr, kr, vr, q, k, v = _qkv(mojo_gpu, dtype, 2, 3, lq, lk, 16)
    for t in (qr, kr, vr, q, k, v):
        t.requires_grad_()
    out, lse, rng, unused, debug = aten._flash_attention_forward(
        *_bshd(q, k, v), None, None, lq, lk, 0.0, is_causal, False
    )
    assert out.shape == (2, lq, 3, 16) and out.is_contiguous()
    assert lse.shape == (2, 3, lq) and lse.dtype == torch.float32
    assert rng.shape == (2,) and rng.dtype == torch.uint64 and debug.numel() == 0
    want, want_lse = _ref_attention(qr, kr, vr, offset=lk - lq if is_causal else None)
    tol = _tol(dtype)
    torch.testing.assert_close(
        out.transpose(1, 2).cpu().double(), want, atol=tol, rtol=tol
    )
    torch.testing.assert_close(
        lse.cpu().double(),
        want_lse.masked_fill(want_lse == -torch.inf, torch.inf),
        atol=tol,
        rtol=tol,
    )
    g = torch.randn(want.shape, generator=torch.Generator().manual_seed(9))
    out.backward(g.transpose(1, 2).to(dtype).to(mojo_gpu))
    assert _counted("_flash_attention_backward") == 1
    want.backward(g.double())
    gt = _grad_tol(dtype)
    for got, ref in ((q, qr), (k, kr), (v, vr)):
        torch.testing.assert_close(
            _grad(got).cpu().double(), _grad(ref).double(), atol=gt, rtol=gt
        )


def test_flash_attention_forward_no_dropout_inplace(mojo_gpu, counting):
    if not hasattr(aten, "_flash_attention_forward_no_dropout_inplace"):
        pytest.skip("this torch predates _flash_attention_forward_no_dropout_inplace")
    qr, kr, vr, q, k, v = _qkv(mojo_gpu, torch.float16, 1, 2, 6, 6, 8)
    out = torch.empty(1, 6, 2, 8, dtype=torch.float16, device=mojo_gpu)
    lse = aten._flash_attention_forward_no_dropout_inplace(
        out, *_bshd(q, k, v), None, None, 6, 6, 0.0, True, False
    )
    want, want_lse = _ref_attention(qr, kr, vr, offset=0)
    torch.testing.assert_close(
        out.transpose(1, 2).cpu().double(), want, atol=5e-3, rtol=5e-3
    )
    torch.testing.assert_close(lse.cpu().double(), want_lse, atol=5e-3, rtol=5e-3)


def test_flash_attention_forward_declines_varlen_and_windows(mojo_gpu):
    _, _, _, q, k, v = _qkv(mojo_gpu, torch.float16, 1, 2, 6, 6, 8)
    cu = torch.tensor([0, 6], dtype=torch.int32, device=mojo_gpu)
    with pytest.raises(NotImplementedError, match="varlen"):
        aten._flash_attention_forward(*_bshd(q, k, v), cu, cu, 6, 6, 0.0, False, False)
    with pytest.raises(NotImplementedError, match="sliding window"):
        aten._flash_attention_forward(
            *_bshd(q, k, v), None, None, 6, 6, 0.0, False, False, window_size_left=2
        )
    with pytest.raises(NotImplementedError, match="dropout"):
        aten._flash_attention_forward(
            *_bshd(q, k, v), None, None, 6, 6, 0.5, False, False
        )


@pytest.mark.parametrize("dtype", [torch.float32, torch.bfloat16])
@pytest.mark.parametrize("mask_type", [0, 1, 2])
@pytest.mark.parametrize("with_bias", [False, True])
def test_efficient_attention_forward_bmhk(
    mojo_gpu, counting, dtype, mask_type, with_bias
):
    """custom_mask_type 1 is causal from the top left, 2 from the bottom
    right; a fully masked row (an all -inf bias row) has output 0 and
    logsumexp 0."""
    lq, lk = 5, 8
    qr, kr, vr, q, k, v = _qkv(mojo_gpu, dtype, 2, 2, lq, lk, 8)
    br = b = None
    if with_bias:
        br = torch.randn(2, 2, lq, lk, generator=torch.Generator().manual_seed(4))
        br[0, 1, 2] = -torch.inf
        br = br.to(dtype).float().requires_grad_()
        b = br.detach().to(dtype).to(mojo_gpu).requires_grad_()
    for t in (qr, kr, vr, q, k, v):
        t.requires_grad_()
    out, lse, seed, offset, mq, mk = aten._efficient_attention_forward(
        *_bshd(q, k, v), b, None, None, None, None, 0.0, mask_type, True
    )
    assert (mq, mk) == (lq, lk) and out.shape == (2, lq, 2, 8)
    assert lse.shape == (2, 2, 32)
    off = {0: None, 1: 0, 2: lk - lq}[mask_type]
    want, want_lse = _ref_attention(qr, kr, vr, br, off)
    tol = _tol(dtype)
    torch.testing.assert_close(
        out.transpose(1, 2).cpu().double(), want, atol=tol, rtol=tol
    )
    torch.testing.assert_close(
        lse[..., :lq].cpu().double(),
        want_lse.masked_fill(want_lse == -torch.inf, 0.0),
        atol=tol,
        rtol=tol,
    )
    g = torch.randn(want.shape, generator=torch.Generator().manual_seed(8))
    out.backward(g.transpose(1, 2).to(dtype).to(mojo_gpu))
    assert _counted("_efficient_attention_backward") == 1
    want.backward(g.double())
    gt = _grad_tol(dtype)
    pairs = [(q, qr), (k, kr), (v, vr)] + ([(b, br)] if with_bias else [])
    for got, ref in pairs:
        torch.testing.assert_close(
            _grad(got).cpu().double(), _grad(ref).double(), atol=gt, rtol=gt
        )


@pytest.mark.parametrize("dtype", [torch.float16, torch.bfloat16])
@pytest.mark.parametrize("is_causal", [False, True])
def test_cudnn_attention(mojo_gpu, counting, dtype, is_causal):
    """cuDNN's contract: logsumexp (B, H, L, 1) float32 when asked for, else
    undefined, like the cumulative-sequence tensors and the debug mask."""
    qr, kr, vr, q, k, v = _qkv(mojo_gpu, dtype, 1, 2, 6, 9, 16)
    br = torch.randn(6, 9, generator=torch.Generator().manual_seed(3)).to(dtype).float()
    for t in (qr, kr, vr, q, k, v):
        t.requires_grad_()
    res = aten._scaled_dot_product_cudnn_attention(
        q, k, v, br.to(dtype).to(mojo_gpu), True, 0.0, is_causal
    )
    out, lse = res[0], res[1]
    assert res[2] is None and res[3] is None and res[8] is None
    assert (res[4], res[5]) == (6, 9) and lse.shape == (1, 2, 6, 1)
    want, want_lse = _ref_attention(qr, kr, vr, br, 0 if is_causal else None)
    tol = _tol(dtype)
    torch.testing.assert_close(out.cpu().double(), want, atol=tol, rtol=tol)
    torch.testing.assert_close(lse[..., 0].cpu().double(), want_lse, atol=tol, rtol=tol)
    out.backward(torch.ones_like(out))
    assert _counted("_scaled_dot_product_cudnn_attention_backward") == 1
    want.backward(torch.ones_like(want))
    gt = _grad_tol(dtype)
    for got, ref in ((q, qr), (k, kr), (v, vr)):
        torch.testing.assert_close(
            _grad(got).cpu().double(), _grad(ref).double(), atol=gt, rtol=gt
        )
    with torch.no_grad():
        assert aten._scaled_dot_product_cudnn_attention(q, k, v, None, False)[1] is None
        fwd = aten._cudnn_attention_forward(q, k, v, None, None, None, 6, 9, True)
    torch.testing.assert_close(
        fwd[0].cpu().double(), _ref_attention(qr, kr, vr)[0], atol=tol, rtol=tol
    )


def test_cudnn_attention_rejects_float32_like_cuda(mojo_gpu):
    _, _, _, q, k, v = _qkv(mojo_gpu, torch.float32, 1, 2, 6, 6, 16)
    with pytest.raises(
        RuntimeError, match="only supports float16 and bfloat16, got Float"
    ):
        aten._scaled_dot_product_cudnn_attention(q, k, v, None, False)


def test_flash_and_efficient_attention_without_keys(mojo_gpu):
    """An empty key/value sequence: output 0, logsumexp +inf for flash (what
    CUDA returns) and 0 for the memory-efficient layout; zero gradients."""
    q = torch.randn(1, 3, 2, 8, dtype=torch.float16, device=mojo_gpu)
    kv = torch.randn(1, 0, 2, 8, dtype=torch.float16, device=mojo_gpu)
    q.requires_grad_()
    out, lse = aten._flash_attention_forward(
        q, kv, kv, None, None, 3, 0, 0.0, False, False
    )[:2]
    assert out.shape == (1, 3, 2, 8) and (out.cpu() == 0).all()
    assert lse.shape == (1, 2, 3) and (lse.cpu() == torch.inf).all()
    out.sum().backward()
    assert (_grad(q).cpu() == 0).all()
    out, lse = aten._efficient_attention_forward(
        q.detach(), kv, kv, None, None, None, None, None, 0.0, 0, True
    )[:2]
    assert (out.cpu() == 0).all() and (lse[..., :3].cpu() == 0).all()


def test_cudnn_attention_forward_returns_the_given_max_lengths(mojo_gpu):
    _, _, _, q, k, v = _qkv(mojo_gpu, torch.float16, 1, 2, 6, 9, 16)
    with torch.no_grad():
        res = aten._cudnn_attention_forward(q, k, v, None, None, None, 11, 13, True)
    assert (res[4], res[5]) == (11, 13)
