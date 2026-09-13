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
            got.grad.contiguous().cpu().float(), want.grad, atol=6e-2, rtol=6e-2
        )


def test_flash_attention_declines_float32(mojo_gpu, counting):
    _, _, _, q, k, v = _qkv(mojo_gpu, torch.float32, 1, 1, 128, 128, 64)
    with pytest.raises(NotImplementedError):
        aten._scaled_dot_product_flash_attention(q, k, v, 0.0, True)


def test_flash_attention_declines_dropout(mojo_gpu, counting):
    _, _, _, q, k, v = _qkv(mojo_gpu, torch.bfloat16, 1, 1, 128, 128, 64)
    with pytest.raises(NotImplementedError):
        aten._scaled_dot_product_flash_attention(q, k, v, 0.5, True)


def test_flash_attention_backward_refuses_partial_tail(mojo_gpu, counting):
    """The backward tile machinery is unproven on a partial last tile, so an
    odd seqlen that the BHSD forward accepts must fail loudly here rather
    than produce a silently wrong gradient."""
    if _arch(mojo_gpu) != "sm_90a":
        pytest.skip(_HOPPER_ONLY)
    _, _, _, q, k, v = _qkv(mojo_gpu, torch.bfloat16, 1, 2, 200, 200, 64)
    for t in (q, k, v):
        t.requires_grad_()
    out = aten._scaled_dot_product_flash_attention(q, k, v, 0.0, True)[0]
    with pytest.raises(NotImplementedError):
        out.backward(torch.ones_like(out))


# --------------------------------------------------------------------------
# _scaled_dot_product_efficient_attention (decode + math)
# --------------------------------------------------------------------------


@pytest.mark.parametrize("dtype", [torch.float32, torch.bfloat16])
def test_efficient_attention_decode_step(mojo_gpu, counting, dtype):
    """q_len == 1: one fused kernel instead of bmm + softmax + bmm (GPU only;
    the MAX CPU device takes the decomposition instead)."""
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


def test_efficient_attention_refuses_grad(mojo_gpu, counting):
    """Its ATen backward op has no kernel here, and a raise from inside the
    autograd engine is a far worse failure than one from the forward."""
    _, _, _, q, k, v = _qkv(mojo_gpu, torch.float32, 1, 1, 8, 8, 8)
    q.requires_grad_()
    with pytest.raises(NotImplementedError, match="torch.no_grad"):
        aten._scaled_dot_product_efficient_attention(q, k, v, None, True, 0.0, False)


def test_efficient_attention_declines_compute_log_sumexp(mojo_gpu, counting):
    """No route here produces an LSE, and the only consumer is a backward op
    this backend has no kernel for: returning zeros would hand a silently
    wrong saved value to a backward that cannot run anyway."""
    _, _, _, q, k, v = _qkv(mojo_gpu, torch.float32, 1, 1, 8, 8, 8)
    with torch.no_grad(), pytest.raises(NotImplementedError, match="log-sum-exp"):
        aten._scaled_dot_product_efficient_attention(q, k, v, None, True, 0.0, False)
    with torch.no_grad():
        lse = aten._scaled_dot_product_efficient_attention(
            q, k, v, None, False, 0.0, False
        )[1]
    assert lse.numel() == 0


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


def test_efficient_attention_declines_bias(mojo_gpu, counting):
    _, _, _, q, k, v = _qkv(mojo_gpu, torch.float32, 1, 1, 8, 8, 8)
    bias = torch.zeros(1, 1, 8, 8, device=mojo_gpu)
    with torch.no_grad(), pytest.raises(NotImplementedError):
        aten._scaled_dot_product_efficient_attention(q, k, v, bias, False, 0.0, False)


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
    assert aten._fused_sdp_choice(q, k, v, None, 0.0, False) == EFFICIENT


def test_fused_sdp_choice_math_when_grad_is_needed(mojo_gpu, counting):
    """`_scaled_dot_product_efficient_attention` is inference-only here, so a
    grad-requiring call must be sent to the differentiable decomposition."""
    _, _, _, q, k, v = _qkv(mojo_gpu, torch.float32, 1, 2, 8, 8, 8)
    q.requires_grad_()
    assert aten._fused_sdp_choice(q, k, v, None, 0.0, False) == MATH


def test_fused_sdp_choice_math_for_masked_and_dropout(mojo_gpu, counting):
    _, _, _, q, k, v = _qkv(mojo_gpu, torch.bfloat16, 1, 2, 128, 128, 64)
    mask = torch.zeros(1, 1, 128, 128, dtype=torch.bfloat16, device=mojo_gpu)
    assert aten._fused_sdp_choice(q, k, v, mask, 0.0, False) == MATH
    assert aten._fused_sdp_choice(q, k, v, None, 0.25, True) == MATH


def test_public_sdpa_takes_the_flash_route_and_trains(mojo_gpu):
    """F.scaled_dot_product_attention picks its backend through a C++
    DispatchStub the shim registers for this device, then calls the
    `_for_cpu` flash overloads (non-CUDA devices), which wrap the flash ops."""
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
    assert (
        native.op_count("aten::_scaled_dot_product_flash_attention_for_cpu")
        == before + 1
    )
    assert (
        native.op_count("aten::_scaled_dot_product_flash_attention_for_cpu_backward")
        == before_bwd + 1
    )
    ref_q = q.detach().cpu().float().requires_grad_(True)
    ref_k = k.detach().cpu().float().requires_grad_(True)
    ref_v = v.detach().cpu().float().requires_grad_(True)
    ref = F.scaled_dot_product_attention(ref_q, ref_k, ref_v, is_causal=True)
    ref.sum().backward()
    assert q.grad is not None and ref_q.grad is not None
    torch.testing.assert_close(out.cpu().float(), ref, atol=2e-2, rtol=2e-2)
    torch.testing.assert_close(q.grad.cpu().float(), ref_q.grad, atol=5e-2, rtol=5e-2)
