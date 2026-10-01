"""The transformer fast-path entry points on the native mojo device
(tmb/ops/transformer.mojo): `_transform_bias_rescale_qkv`,
`_native_multi_head_attention` and `_transformer_encoder_layer_fwd`, which
`nn.MultiheadAttention` / `nn.TransformerEncoderLayer` call in eval mode.
"""

import copy

import pytest
import torch

from tests.native.conftest import ran

aten = torch.ops.aten


def _tol(dtype: torch.dtype) -> dict[str, float]:
    if dtype == torch.bfloat16:
        return {"atol": 3e-2, "rtol": 3e-2}
    if dtype == torch.float16:
        return {"atol": 3e-3, "rtol": 3e-3}
    return {"atol": 2e-5, "rtol": 2e-5}


@pytest.mark.parametrize("dtype", [torch.float32, torch.float16, torch.bfloat16])
@pytest.mark.parametrize("shape", [(2, 5, 4, 12), (1, 3, 3, 7)])
def test_transform_bias_rescale_qkv(mojo_gpu, dtype, shape):
    b, t, nh, dh = shape
    d = nh * dh
    gen = torch.Generator().manual_seed(0)
    qkv = torch.randn(b, t, 3 * d, generator=gen).to(dtype)
    bias = torch.randn(3 * d, generator=gen).to(dtype)
    with ran("aten::_transform_bias_rescale_qkv"):
        got = aten._transform_bias_rescale_qkv(qkv.to(mojo_gpu), bias.to(mojo_gpu), nh)
    # CUDA: the 1/sqrt(dh) factor is rounded to the tensor dtype, the sum and
    # product run in float and round once.
    sqrt = torch.tensor(float(dh), dtype=dtype).float().sqrt().item()
    inv = torch.tensor(1.0 / sqrt, dtype=torch.float64).to(dtype).float()
    x = (qkv.float() + bias.float()).view(b, t, 3, nh, dh).permute(2, 0, 3, 1, 4)
    want = [x[0] * inv, x[1], x[2]]
    for g, w in zip(got, want):
        assert g.shape == (b, nh, t, dh) and g.dtype == dtype
        torch.testing.assert_close(g.cpu(), w.to(dtype), atol=0, rtol=0)


def test_transform_bias_rescale_qkv_bad_heads(mojo_gpu):
    qkv = torch.randn(1, 2, 12, device=mojo_gpu)
    with pytest.raises(RuntimeError, match="D % num_head == 0"):
        aten._transform_bias_rescale_qkv(qkv, torch.randn(12, device=mojo_gpu), 3)


@pytest.mark.parametrize("need_weights", [False, True])
@pytest.mark.parametrize("average", [True, False])
def test_mha_self_attention_matches_cpu(mojo_gpu, need_weights, average):
    torch.manual_seed(0)
    cpu = torch.nn.MultiheadAttention(32, 4, batch_first=True).eval()
    dev = copy.deepcopy(cpu).to(mojo_gpu)
    x = torch.randn(2, 7, 32)
    xd = x.to(mojo_gpu)
    with torch.no_grad(), ran("aten::_native_multi_head_attention"):
        o, w = cpu(x, x, x, need_weights=need_weights, average_attn_weights=average)
        od, wd = dev(
            xd, xd, xd, need_weights=need_weights, average_attn_weights=average
        )
    torch.testing.assert_close(od.cpu(), o, atol=1e-5, rtol=1e-4)
    if need_weights:
        torch.testing.assert_close(wd.cpu(), w, atol=1e-5, rtol=1e-4)
    else:
        assert wd is None


@pytest.mark.parametrize(
    "mask_kind", ["none", "src_mask", "key_padding", "full_bool", "full_float"]
)
@pytest.mark.parametrize("attention", ["self", "cross", "separate"])
def test_native_mha_matches_cpu(mojo_gpu, mask_kind, attention):
    """The op itself against CPU's: self-attention (one tensor), encoder-
    decoder (key is value) and three separate inputs, each mask type."""
    torch.manual_seed(0)
    b, t, e, h = 2, 7, 32, 4
    x, y, z = (torch.randn(b, t, e) for _ in range(3))
    q, k, v = {"self": (x, x, x), "cross": (x, y, y), "separate": (x, y, z)}[attention]
    ws = [torch.randn(3 * e, e) / e**0.5, torch.randn(3 * e)]
    ws += [torch.randn(e, e) / e**0.5, torch.randn(e)]
    mask, mask_type = None, None
    if mask_kind == "src_mask":
        mask, mask_type = torch.triu(torch.ones(t, t, dtype=torch.bool), 1), 0
    elif mask_kind == "key_padding":
        mask, mask_type = torch.zeros(b, t, dtype=torch.bool), 1
        mask[1, 4:] = True
    elif mask_kind == "full_bool":
        mask, mask_type = torch.rand(b, h, t, t) > 0.7, 2
        mask[..., 0] = False  # no fully masked row
    elif mask_kind == "full_float":
        mask, mask_type = (torch.rand(b, h, t, t) > 0.7).float(), 2
        mask[..., 0] = 0.0
    dev = {id(x): x.to(mojo_gpu), id(y): y.to(mojo_gpu), id(z): z.to(mojo_gpu)}
    qd, kd, vd = dev[id(q)], dev[id(k)], dev[id(v)]
    wd = [w.to(mojo_gpu) for w in ws]
    md = None if mask is None else mask.to(mojo_gpu)
    for need_weights, average in ((False, True), (True, True), (True, False)):
        want = aten._native_multi_head_attention(
            q, k, v, e, h, *ws, mask, need_weights, average, mask_type
        )
        with ran("aten::_native_multi_head_attention"):
            got = aten._native_multi_head_attention(
                qd, kd, vd, e, h, *wd, md, need_weights, average, mask_type
            )
        torch.testing.assert_close(got[0].cpu(), want[0], atol=1e-5, rtol=1e-4)
        if need_weights:
            torch.testing.assert_close(got[1].cpu(), want[1], atol=1e-5, rtol=1e-4)
        else:
            assert got[1] is None


def test_native_mha_errors_match_cuda(mojo_gpu):
    x = torch.randn(2, 5, 8, device=mojo_gpu)
    w = torch.randn(24, 8, device=mojo_gpu)
    b = torch.randn(24, device=mojo_gpu)
    pw, pb = torch.randn(8, 8, device=mojo_gpu), torch.randn(8, device=mojo_gpu)
    with pytest.raises(RuntimeError, match="must divide evenly"):
        aten._native_multi_head_attention(x, x, x, 8, 3, w, b, pw, pb)
    with pytest.raises(RuntimeError, match="didn't match last dim of query"):
        aten._native_multi_head_attention(x, x, x, 4, 2, w, b, pw, pb)
    with pytest.raises(RuntimeError, match="Mask Type should be defined"):
        aten._native_multi_head_attention(
            x,
            x,
            x,
            8,
            2,
            w,
            b,
            pw,
            pb,
            torch.zeros(5, 5, dtype=torch.bool, device=mojo_gpu),
        )


def _cuda_reference(layer):
    """The module's slow (Python) path with CUDA's fast-path activation:
    `_addmm_activation(use_gelu=True)` applies the tanh-approximated GELU
    there (cuBLASLt's epilogue), where CPU's is exact."""
    ref = copy.deepcopy(layer)
    if ref.activation_relu_or_gelu == 2:
        ref.activation = lambda t: torch.nn.functional.gelu(t, approximate="tanh")
    ref.activation_relu_or_gelu = 0  # disables the fast path
    return ref


@pytest.mark.parametrize("activation", ["relu", "gelu"])
@pytest.mark.parametrize("norm_first", [False, True])
@pytest.mark.parametrize("padding_mask", [False, True])
def test_transformer_encoder_layer_eval_fast_path(
    mojo_gpu, activation, norm_first, padding_mask
):
    torch.manual_seed(0)
    cpu = torch.nn.TransformerEncoderLayer(
        32, 4, 64, batch_first=True, activation=activation, norm_first=norm_first
    ).eval()
    dev = copy.deepcopy(cpu).to(mojo_gpu)
    x = torch.randn(2, 7, 32)
    kpm = None
    if padding_mask:
        kpm = torch.zeros(2, 7, dtype=torch.bool)
        kpm[1, 5:] = True
    with torch.no_grad(), ran("aten::_transformer_encoder_layer_fwd"):
        got = dev(
            x.to(mojo_gpu),
            src_key_padding_mask=None if kpm is None else kpm.to(mojo_gpu),
        )
    with torch.no_grad():
        want = _cuda_reference(cpu)(x, src_key_padding_mask=kpm)
    torch.testing.assert_close(got.cpu(), want, atol=2e-5, rtol=2e-4)


def test_transformer_encoder_layer_half(mojo_gpu):
    torch.manual_seed(0)
    cpu = torch.nn.TransformerEncoderLayer(32, 4, 64, batch_first=True).eval()
    dev = copy.deepcopy(cpu).to(mojo_gpu).half()
    x = torch.randn(2, 7, 32)
    with torch.no_grad():
        got = dev(x.half().to(mojo_gpu)).cpu().float()
        want = cpu(x.half().float())
    torch.testing.assert_close(got, want, atol=2e-2, rtol=2e-2)
