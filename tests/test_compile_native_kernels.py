"""torch.compile(backend=mojo_backend) on a GPU runs this repository's eager
kernels -- through the `tmb/graph` custom ops -- for the ops GPT-2
inference spends its time in: the Linear GEMMs (addmm / mm with a transposed
weight), bmm, softmax, layer norm and embedding.

Each test compiles a small torch function, compares it with stock eager torch
and, through a spy on the `custom_mojo_ops.native_*` wrapper, checks that the
graph took the native route: a MAX composition silently kept would pass the
numbers and fail nothing else.
"""

import contextlib
import math
from collections.abc import Callable, Iterator
from typing import TypedDict, TypeVar, cast
from unittest import mock

import pytest
import torch
import torch.nn.functional as F
from torch import nn
from torch._dynamo import mark_dynamic

from tests.conftest import require_cuda_autograd
from torch_mojo_backend import aten_functions, custom_mojo_ops, mojo_backend
from torch_mojo_backend.testing import CallChecker
from torch_mojo_backend.types import MaxTensor

_F = TypeVar("_F", bound=Callable[..., object])


class _Tolerance(TypedDict):
    rtol: float
    atol: float


# Same fp32 math as cuBLAS in another summation order: well inside 1e-4.
FP32 = _Tolerance(rtol=1e-4, atol=1e-4)
# TF32 tensor cores round both operands to 10 mantissa bits.
TF32 = _Tolerance(rtol=1e-2, atol=2e-2)
BF16 = _Tolerance(rtol=2e-2, atol=2e-2)


@pytest.fixture
def gpu(cuda_available: bool) -> str:
    if not cuda_available:
        pytest.skip("the native graph ops are accelerator routes")
    return "cuda"


@pytest.fixture(autouse=True)
def highest_matmul_precision():
    """torch's default; the TF32 test flips it and this puts it back."""
    before = torch.get_float32_matmul_precision()
    torch.set_float32_matmul_precision("highest")
    yield
    torch.set_float32_matmul_precision(before)


@contextlib.contextmanager
def routed_through(name: str) -> Iterator[mock.MagicMock]:
    """Spy on one `custom_mojo_ops.native_*` wrapper (aten_functions reaches
    it through the module attribute, so the spy sees every call the compile
    makes)."""
    original = getattr(custom_mojo_ops, name)
    with mock.patch.object(custom_mojo_ops, name, wraps=original) as spy:
        yield spy


def compiled(fn: _F) -> _F:
    return cast(_F, torch.compile(fn, backend=mojo_backend, fullgraph=True))


def dims(value: MaxTensor) -> list[int]:
    return [int(d) for d in value.shape]


def test_linear_is_one_native_gemm_over_the_stored_weight(gpu: str):
    """`F.linear` lowers to `addmm(bias, view(x), t(weight))`: the GEMM gets the
    (out, in) weight as stored plus `transpose_b`, never a transposed copy, and
    the bias rides in the same call."""

    def fn(x, w, b):
        return F.linear(x, w, b)

    x = torch.randn(2, 5, 64, device=gpu)
    w = torch.randn(96, 64, device=gpu)
    b = torch.randn(96, device=gpu)
    mark_dynamic(x, 0)
    mark_dynamic(x, 1)
    with routed_through("native_gemm") as spy:
        out = compiled(fn)(x, w, b)
    torch.testing.assert_close(out, fn(x, w, b), **FP32)
    assert spy.call_count == 1
    a, weight, bias = spy.call_args.args
    assert spy.call_args.kwargs == {"transpose_b": True, "tf32": False}
    assert dims(weight) == [96, 64]
    assert bias is not None


def test_mm_without_a_transpose(gpu: str):
    def fn(x, w):
        return x @ w

    x = torch.randn(7, 64, device=gpu)
    w = torch.randn(64, 33, device=gpu)
    mark_dynamic(x, 0)
    with routed_through("native_gemm") as spy:
        out = compiled(fn)(x, w)
    torch.testing.assert_close(out, fn(x, w), **FP32)
    assert spy.call_args.kwargs == {"transpose_b": False, "tf32": False}


def test_mm_single_row_takes_the_decode_route(gpu: str):
    """m == 1 is the lm_head of a decode step (the eager GEMV tier)."""

    def fn(x, w):
        return x @ w.t()

    x = torch.randn(1, 64, device=gpu)
    w = torch.randn(1000, 64, device=gpu)
    with routed_through("native_gemm") as spy:
        out = compiled(fn)(x, w)
    torch.testing.assert_close(out, fn(x, w), **FP32)
    assert spy.call_count == 1


def test_addmm_with_a_broadcast_bias_adds_after_the_gemm(gpu: str):
    """Only a (n,) bias fuses (tmb/ops/matmul.mojo's `_bias_fits`); a (1, n) one
    is added to the GEMM's result."""

    def fn(b, x, w):
        return torch.addmm(b, x, w)

    x = torch.randn(5, 64, device=gpu)
    w = torch.randn(64, 40, device=gpu)
    b = torch.randn(1, 40, device=gpu)
    with routed_through("native_gemm") as spy:
        out = compiled(fn)(b, x, w)
    torch.testing.assert_close(out, fn(b, x, w), **FP32)
    assert spy.call_args.args[2] is None


def test_addmm_with_scaling_keeps_the_max_composition(gpu: str):
    def fn(b, x, w):
        return torch.addmm(b, x, w, alpha=2.0, beta=0.5)

    x = torch.randn(5, 64, device=gpu)
    w = torch.randn(64, 40, device=gpu)
    b = torch.randn(40, device=gpu)
    with routed_through("native_gemm") as spy:
        out = compiled(fn)(b, x, w)
    torch.testing.assert_close(out, fn(b, x, w), **TF32)  # MAX's fp32 matmul is TF32
    assert spy.call_count == 0


def test_bmm(gpu: str):
    def fn(a, b):
        return torch.bmm(a, b)

    a = torch.randn(6, 9, 32, device=gpu)
    b = torch.randn(6, 32, 17, device=gpu)
    mark_dynamic(a, 0)
    mark_dynamic(a, 1)
    mark_dynamic(b, 0)
    with routed_through("native_bmm") as spy:
        out = compiled(fn)(a, b)
    torch.testing.assert_close(out, fn(a, b), **FP32)
    assert spy.call_args.kwargs == {"transpose_b": False, "tf32": False}


def test_bmm_reads_a_transposed_operand_in_place(gpu: str):
    def fn(a, b):
        return torch.bmm(a, b.transpose(1, 2))

    a = torch.randn(6, 9, 32, device=gpu)
    b = torch.randn(6, 17, 32, device=gpu)
    with routed_through("native_bmm") as spy:
        out = compiled(fn)(a, b)
    torch.testing.assert_close(out, fn(a, b), **FP32)
    assert spy.call_args.kwargs["transpose_b"] is True
    assert dims(spy.call_args.args[1]) == [6, 17, 32]


def test_softmax_over_the_last_dim(gpu: str):
    def fn(x):
        return torch.softmax(x, dim=-1)

    x = torch.randn(2, 3, 5, 7, device=gpu)
    mark_dynamic(x, 2)
    with routed_through("native_softmax_rows") as spy:
        out = compiled(fn)(x)
    torch.testing.assert_close(out, fn(x), **FP32)
    assert spy.call_count == 1
    assert int(spy.call_args.args[0].shape[-1]) == 7  # rows are symbolic


def test_softmax_over_another_dim_keeps_the_max_composition(gpu: str):
    def fn(x):
        return torch.softmax(x, dim=1)

    x = torch.randn(2, 3, 5, device=gpu)
    with routed_through("native_softmax_rows") as spy:
        out = compiled(fn)(x)
    torch.testing.assert_close(out, fn(x), **FP32)
    assert spy.call_count == 0


def test_softmax_half_to_float(gpu: str):
    def fn(x):
        return torch.softmax(x, dim=-1, dtype=torch.float32)

    x = torch.randn(4, 9, device=gpu, dtype=torch.bfloat16)
    with routed_through("native_softmax_rows") as spy:
        out = compiled(fn)(x)
    assert out.dtype == torch.float32
    torch.testing.assert_close(out, fn(x), **FP32)
    assert spy.call_count == 1


@pytest.mark.parametrize("dtype", [torch.float32, torch.bfloat16])
@pytest.mark.parametrize("dim", [0, -1])
def test_glu_backward(
    gpu: str, call_checker: CallChecker, dtype: torch.dtype, dim: int
):
    """Both gradient halves take the eager pointwise kinds: `glu_backward_a`
    through the binary custom op, `glu_backward_b` through the three-input
    `native_glu_backward_b`."""
    call_checker.register(aten_functions.aten_glu_backward)

    def fn(grad, x):
        return torch.ops.aten.glu_backward(grad, x, dim)

    x = (torch.randn(4, 6, 8, device=gpu) * 3).to(dtype)
    grad_shape = (2, 6, 8) if dim == 0 else (4, 6, 4)
    grad = torch.randn(grad_shape, device=gpu).to(dtype)
    with routed_through("native_glu_backward_b") as spy:
        out = compiled(fn)(grad, x)
    want = fn(grad.float(), x.float()).to(dtype)
    torch.testing.assert_close(out, want, **(BF16 if dtype == torch.bfloat16 else FP32))
    assert spy.call_count == 1


@pytest.fixture
def autograd_gpu(gpu: str) -> str:
    """`gpu`, in a process where CUDA autograd still works (a registered mojo
    device breaks it: see `require_cuda_autograd`). A fixture so the skip
    comes before `call_checker` is set up."""
    require_cuda_autograd(gpu)
    return gpu


def test_glu_through_autograd(autograd_gpu: str, call_checker: CallChecker):
    """F.glu's forward and backward under compile, both graph twins."""
    gpu = autograd_gpu
    call_checker.register(aten_functions.aten_glu, aten_functions.aten_glu_backward)

    def fn(x):
        return F.glu(x, -1)

    x = torch.randn(5, 10, device=gpu, requires_grad=True)
    grad = torch.randn(5, 5, device=gpu)
    with routed_through("native_glu_backward_b") as spy:
        compiled(fn)(x).backward(grad)
    assert x.grad is not None
    got = x.grad
    ref = x.detach().clone().requires_grad_(True)
    fn(ref).backward(grad)
    torch.testing.assert_close(got, ref.grad, **FP32)
    assert spy.call_count == 1


def test_layer_norm(gpu: str):
    def fn(x, w, b):
        return F.layer_norm(x, (64,), w, b, eps=1e-5)

    x = torch.randn(2, 5, 64, device=gpu)
    w = torch.randn(64, device=gpu)
    b = torch.randn(64, device=gpu)
    mark_dynamic(x, 1)
    with routed_through("native_layer_norm") as spy:
        out = compiled(fn)(x, w, b)
    torch.testing.assert_close(out, fn(x, w, b), **FP32)
    assert spy.call_count == 1


def test_native_layer_norm_returns_the_statistics(gpu: str):
    """The three ATen outputs, statistics float32 of the collapsed shape --
    what a backward graph reads -- not the placeholders the MAX fused path
    hands back."""

    def fn(x, w, b):
        return torch.native_layer_norm(x, (8, 16), w, b, 1e-5)

    x = torch.randn(3, 4, 8, 16, device=gpu)
    w = torch.randn(8, 16, device=gpu)
    b = torch.randn(8, 16, device=gpu)
    with routed_through("native_layer_norm") as spy:
        out, mean, rstd = compiled(fn)(x, w, b)
    ref_out, ref_mean, ref_rstd = fn(x, w, b)
    torch.testing.assert_close(out, ref_out, **FP32)
    assert mean.shape == ref_mean.shape == (3, 4, 1, 1)
    assert mean.dtype == rstd.dtype == torch.float32
    torch.testing.assert_close(mean, ref_mean, **FP32)
    torch.testing.assert_close(rstd, ref_rstd, **FP32)
    assert spy.call_count == 1


def test_layer_norm_without_affine_keeps_the_max_composition(gpu: str):
    def fn(x):
        return F.layer_norm(x, (64,))

    x = torch.randn(2, 5, 64, device=gpu)
    with routed_through("native_layer_norm") as spy:
        out = compiled(fn)(x)
    torch.testing.assert_close(out, fn(x), **FP32)
    assert spy.call_count == 0


@pytest.mark.parametrize("index_dtype", [torch.int64, torch.int32])
def test_embedding(gpu: str, index_dtype: torch.dtype):
    def fn(idx, table):
        return F.embedding(idx, table)

    table = torch.randn(50, 32, device=gpu)
    idx = torch.randint(0, 50, (2, 7), device=gpu, dtype=index_dtype)
    mark_dynamic(idx, 1)
    with routed_through("native_embedding") as spy:
        out = compiled(fn)(idx, table)
    torch.testing.assert_close(out, fn(idx, table))
    assert spy.call_count == 1
    assert out.shape == (2, 7, 32)


def test_tf32_follows_torch_matmul_precision(gpu: str):
    """Same knob as eager mode: any precision but "highest" lets float32
    GEMMs take the TF32 tensor-core bridge."""
    torch.set_float32_matmul_precision("high")

    def fn(x, w, b):
        return F.linear(x, w, b)

    x = torch.randn(64, 256, device=gpu)
    w = torch.randn(128, 256, device=gpu)
    b = torch.randn(128, device=gpu)
    with routed_through("native_gemm") as spy:
        out = compiled(fn)(x, w, b)
    torch.testing.assert_close(out, fn(x, w, b), **TF32)
    assert spy.call_args.kwargs["tf32"] is True


def test_switched_off_by_the_environment(gpu: str, monkeypatch: pytest.MonkeyPatch):
    monkeypatch.setenv("TORCH_MOJO_BACKEND_COMPILE_NATIVE_KERNELS", "0")

    def fn(x, w, b):
        return F.linear(x, w, b)

    x = torch.randn(5, 64, device=gpu)
    w = torch.randn(96, 64, device=gpu)
    b = torch.randn(96, device=gpu)
    with routed_through("native_gemm") as spy:
        out = compiled(fn)(x, w, b)
    torch.testing.assert_close(out, fn(x, w, b), **TF32)  # MAX's fp32 matmul is TF32
    assert spy.call_count == 0


def test_bf16_linear_and_bmm(gpu: str):
    """bfloat16 takes the gemm16 tensor-core bridge on an H100 and the SIMT
    routes elsewhere; either way the same numerics class as torch's."""

    def fn(x, w, b, q, k):
        return F.linear(x, w, b), torch.bmm(q, k.transpose(1, 2))

    x = torch.randn(3, 9, 128, device=gpu, dtype=torch.bfloat16)
    w = torch.randn(64, 128, device=gpu, dtype=torch.bfloat16)
    b = torch.randn(64, device=gpu, dtype=torch.bfloat16)
    q = torch.randn(4, 9, 32, device=gpu, dtype=torch.bfloat16)
    k = torch.randn(4, 9, 32, device=gpu, dtype=torch.bfloat16)
    with routed_through("native_gemm") as gemm, routed_through("native_bmm") as bmm:
        out, att = compiled(fn)(x, w, b, q, k)
    ref_out, ref_att = fn(x, w, b, q, k)
    torch.testing.assert_close(out, ref_out, **BF16)
    torch.testing.assert_close(att, ref_att, **BF16)
    assert gemm.call_count == 1
    assert bmm.call_count == 1


class _Block(nn.Module):
    """One GPT-2 block as demo_scripts/gpt2.py writes it: manual attention
    with a masked_fill causal mask, exact GELU, pre-LayerNorm."""

    bias: torch.Tensor

    def __init__(self, n_embd: int, n_head: int, block_size: int):
        super().__init__()
        self.ln_1 = nn.LayerNorm(n_embd)
        self.c_attn = nn.Linear(n_embd, 3 * n_embd)
        self.c_proj = nn.Linear(n_embd, n_embd)
        self.ln_2 = nn.LayerNorm(n_embd)
        self.c_fc = nn.Linear(n_embd, 4 * n_embd)
        self.c_proj2 = nn.Linear(4 * n_embd, n_embd)
        self.n_head = n_head
        self.register_buffer(
            "bias",
            torch.tril(torch.ones(block_size, block_size)).view(
                1, 1, block_size, block_size
            ),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        B, T, C = x.size()
        h = self.ln_1(x)
        q, k, v = self.c_attn(h).split(C, dim=2)
        k = k.view(B, T, self.n_head, C // self.n_head).transpose(1, 2)
        q = q.view(B, T, self.n_head, C // self.n_head).transpose(1, 2)
        v = v.view(B, T, self.n_head, C // self.n_head).transpose(1, 2)
        att = (q @ k.transpose(-2, -1)) * (1.0 / math.sqrt(k.size(-1)))
        att = att.masked_fill(self.bias[:, :, :T, :T] == 0, float("-inf"))
        att = F.softmax(att, dim=-1)
        y = (att @ v).transpose(1, 2).contiguous().view(B, T, C)
        x = x + self.c_proj(y)
        return x + self.c_proj2(F.gelu(self.c_fc(self.ln_2(x))))


class _TinyGPT2(nn.Module):
    def __init__(self):
        super().__init__()
        self.wte = nn.Embedding(100, 64)
        self.wpe = nn.Embedding(16, 64)
        self.block = _Block(64, 4, 16)
        self.ln_f = nn.LayerNorm(64)

    def forward(self, idx: torch.Tensor) -> torch.Tensor:
        pos = torch.arange(0, idx.shape[1], device=idx.device)
        x = self.block(self.wte(idx) + self.wpe(pos))
        x = self.ln_f(x)
        # The demo's inference head: the last position through the tied weight.
        return x[:, -1, :][:, None, :] @ self.wte.weight.t()


def test_gpt2_inference_runs_every_hot_op_natively(gpu: str):
    """The whole demo model shape, with dynamic batch and sequence: the five
    native routes all fire and the logits match eager torch."""
    torch.manual_seed(0)
    model = _TinyGPT2().to(gpu).eval()
    idx = torch.randint(0, 100, (2, 7), device=gpu)
    mark_dynamic(idx, 0)
    mark_dynamic(idx, 1)
    names = [
        "native_gemm",
        "native_bmm",
        "native_softmax_rows",
        "native_layer_norm",
        "native_embedding",
    ]
    with contextlib.ExitStack() as stack, torch.no_grad():
        spies = {name: stack.enter_context(routed_through(name)) for name in names}
        out = compiled(model.forward)(idx)
        ref = model(idx)
    torch.testing.assert_close(out, ref, rtol=1e-3, atol=1e-3)
    assert {name: spy.call_count for name, spy in spies.items()} == {
        "native_gemm": 5,  # c_attn, c_proj, c_fc, c_proj2 (addmm) and the head (mm)
        "native_bmm": 2,
        "native_softmax_rows": 1,
        "native_layer_norm": 3,
        "native_embedding": 2,
    }
