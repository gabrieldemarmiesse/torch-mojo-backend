"""Regression coverage for the in-place ("_") overloads this chunk was asked
to register: acos_, acosh_, asin_, asinh_, atan2_, atan_, atanh_,
bitwise_and_, bitwise_left_shift_, bitwise_not_, bitwise_or_,
bitwise_right_shift_, bitwise_xor_, ceil_, clamp_, clamp_max_, clamp_min_,
copysign_, cos_, cosh_, digamma_, div_, elu_, eq_, erf_, erfc_, erfinv_,
exp2_, exp_, expm1_, floor_, fmod_, frac_, gcd_, ge_, gelu_, gt_,
heaviside_, hypot_, i0_, igamma_, igammac_, lcm_, le_, lerp_, lgamma_,
log10_, log1p_, log2_, log_, logit_, lt_, mish_, ne_, neg_, nextafter_,
pow_, reciprocal_, remainder_, round_, rsqrt_, sgn_, sigmoid_, sign_,
silu_, sin_, sinc_, sinh_, sqrt_, tan_, tanh_, trunc_, xlogy_.

The surprise of this chunk: almost none of them need a registration of
their own. PyTorch's codegen gives every *structured* op group -- one
TORCH_META_FUNC / TORCH_IMPL_FUNC pair shared by the functional, `.out` and
in-place variants -- a generic `CompositeExplicitAutogradNonFunctional`
dispatch entry, and every hand-written in-place Scalar overload (e.g.
`bitwise_and_.Scalar`) is a `CompositeExplicitAutograd` composite that
wraps the scalar and calls the same route. Both alias keys apply to ANY
backend with no more specific entry -- PrivateUse1 ("mojo") included -- and
every op in this file already has its `.out` kernel registered on this
device, so `x.acos_(...)` already reaches `aten::acos.out` without any
plumbing of its own. Checked for every op/overload above with
`torch._C._dispatch_dump("aten::<op>_...")`: each shows one of those two
alias keys except `logit_`, whose native_functions.yaml entry hand-picks
`CPU, CUDA, XPU` with no generic fallback at all -- that one is the actual
gap this chunk closes (`op_logit_` in unary.mojo, registered as `"logit_"`
directly).

These tests exist to lock the already-working behavior in as a regression
guard (a future torch version, or a change to this backend's own `.out`
registrations, could silently stop the fallback from reaching our kernel),
and to cover `logit_`, the one case that is new.
"""

from __future__ import annotations

from collections.abc import Callable

import pytest
import torch

from tests.native.conftest import ran
from torch_mojo_backend import native


def _count(name: str) -> int:
    return native.op_count(f"aten::{name}")


def _reset():
    native.op_counting(True)
    native.op_counts_reset()


def _tol(dtype: torch.dtype) -> tuple[float, float]:
    if dtype in (torch.float16, torch.bfloat16):
        return 3e-2, 3e-2
    return 1e-4, 1e-5


def _sample(domain: str, shape: tuple[int, ...]) -> torch.Tensor:
    g = torch.Generator().manual_seed(0)
    if domain == "unit":
        return torch.empty(shape, dtype=torch.float64).uniform_(
            -0.85, 0.85, generator=g
        )
    if domain == "positive":
        return torch.empty(shape, dtype=torch.float64).uniform_(0.15, 4.0, generator=g)
    if domain == "above_1":
        return torch.empty(shape, dtype=torch.float64).uniform_(1.0, 6.0, generator=g)
    if domain == "above_neg1":
        return torch.empty(shape, dtype=torch.float64).uniform_(-0.9, 4.0, generator=g)
    return torch.randn(shape, dtype=torch.float64, generator=g) * 2


# ---------------------------------------------------------------------------
# unary: the promoting transcendental family (acos_ .. tanh_). Int/bool self
# promotes to float for the compute and must raise when cast back.
# ---------------------------------------------------------------------------

_PROMOTING_UNARY = [
    ("acos_", "acos.out", "unit"),
    ("acosh_", "acosh.out", "above_1"),
    ("asin_", "asin.out", "unit"),
    ("asinh_", "asinh.out", "signed"),
    ("atan_", "atan.out", "signed"),
    ("atanh_", "atanh.out", "unit"),
    ("cos_", "cos.out", "signed"),
    ("cosh_", "cosh.out", "signed"),
    ("digamma_", "digamma.out", "positive"),
    ("erf_", "erf.out", "signed"),
    ("erfc_", "erfc.out", "signed"),
    ("erfinv_", "erfinv.out", "unit"),
    ("exp_", "exp.out", "signed"),
    ("exp2_", "exp2.out", "signed"),
    ("expm1_", "expm1.out", "signed"),
    ("i0_", "i0.out", "signed"),
    ("lgamma_", "lgamma.out", "positive"),
    ("log_", "log.out", "positive"),
    ("log10_", "log10.out", "positive"),
    ("log1p_", "log1p.out", "above_neg1"),
    ("log2_", "log2.out", "positive"),
    ("reciprocal_", "reciprocal.out", "positive"),
    ("rsqrt_", "rsqrt.out", "positive"),
    ("sigmoid_", "sigmoid.out", "signed"),
    ("sin_", "sin.out", "signed"),
    ("sinc_", "sinc.out", "signed"),
    ("sinh_", "sinh.out", "signed"),
    ("sqrt_", "sqrt.out", "positive"),
    ("tan_", "tan.out", "unit"),
    ("tanh_", "tanh.out", "signed"),
]


@pytest.mark.parametrize("dtype", [torch.float32, torch.bfloat16])
@pytest.mark.parametrize("op,out_key,domain", _PROMOTING_UNARY)
def test_promoting_unary_inplace_matches_cpu(mojo_gpu, op, out_key, domain, dtype):
    x64 = _sample(domain, (3, 5))
    cpu = x64.to(dtype)
    getattr(cpu, op)()
    x = x64.to(dtype).to(mojo_gpu)
    _reset()
    with ran(f"aten::{out_key}"):
        y = getattr(x, op)()
    assert y.data_ptr() == x.data_ptr()
    rtol, atol = _tol(dtype)
    torch.testing.assert_close(x.cpu(), cpu, rtol=rtol, atol=atol, equal_nan=True)


@pytest.mark.parametrize("op,out_key,domain", _PROMOTING_UNARY)
def test_promoting_unary_inplace_int_self_errors_like_torch(
    mojo_gpu, op, out_key, domain
):
    """The promoted (float) result cannot cast back into an int64 self --
    `TensorIteratorBase::compute_types`'s in-place dtype check, independent
    of device, so CPU torch raises the exact same way first."""
    cpu = torch.tensor([1, 2, 3], dtype=torch.int64)
    with pytest.raises(RuntimeError, match="can't be cast"):
        getattr(cpu.clone(), op)()
    x = cpu.to(mojo_gpu)
    with pytest.raises(RuntimeError, match="can't be cast"):
        getattr(x, op)()


@pytest.mark.parametrize("op,out_key,domain", _PROMOTING_UNARY)
def test_promoting_unary_inplace_noncontiguous(mojo_gpu, op, out_key, domain):
    base64 = _sample(domain, (6, 4))
    cpu_t = base64.to(torch.float32).t()
    dev_t = base64.to(torch.float32).to(mojo_gpu).t()
    assert not cpu_t.is_contiguous() and not dev_t.is_contiguous()
    getattr(cpu_t, op)()
    getattr(dev_t, op)()
    torch.testing.assert_close(dev_t.cpu(), cpu_t, rtol=1e-4, atol=1e-5, equal_nan=True)


# ---------------------------------------------------------------------------
# unary: exact-dtype (no promotion) -- neg_, sign_, sgn_ -- work on int too.
# ---------------------------------------------------------------------------

_DIRECT_UNARY = [("neg_", "neg.out"), ("sign_", "sign.out"), ("sgn_", "sgn.out")]


@pytest.mark.parametrize("dtype", [torch.float32, torch.int64, torch.bool])
@pytest.mark.parametrize("op,out_key", _DIRECT_UNARY)
def test_direct_unary_inplace_matches_cpu(mojo_gpu, op, out_key, dtype):
    if dtype == torch.bool:
        x64 = torch.tensor([True, False, True, False])
    else:
        x64 = torch.randn(5, dtype=torch.float64) * 3
    if op == "neg_" and dtype == torch.bool:
        # Real CPU torch refuses this combination outright (negating a bool
        # doesn't mean anything) -- both legs must refuse the same way.
        with pytest.raises(RuntimeError, match="not supported"):
            x64.to(dtype).clone().neg_()
        with pytest.raises(RuntimeError, match="not supported"):
            x64.to(dtype).to(mojo_gpu).neg_()
        return
    cpu = x64.to(dtype)
    getattr(cpu, op)()
    x = x64.to(dtype).to(mojo_gpu)
    _reset()
    with ran(f"aten::{out_key}"):
        getattr(x, op)()
    assert torch.equal(x.cpu(), cpu)


# ---------------------------------------------------------------------------
# unary: ceil_ / floor_ / round_ / trunc_ -- identity on int/bool, rounds
# float. bitwise_not_ and frac_ alongside them (same single-tensor shape).
# ---------------------------------------------------------------------------

_ROUNDING_UNARY = [
    ("ceil_", "ceil.out"),
    ("floor_", "floor.out"),
    ("round_", "round.out"),
    ("trunc_", "trunc.out"),
]


@pytest.mark.parametrize("dtype", [torch.float32, torch.int32])
@pytest.mark.parametrize("op,out_key", _ROUNDING_UNARY)
def test_rounding_unary_inplace_matches_cpu(mojo_gpu, op, out_key, dtype):
    if dtype == torch.int32:
        x64 = torch.tensor([1, -2, 3, -4, 0], dtype=torch.int32)
    else:
        x64 = torch.tensor([1.4, -2.6, 0.5, -0.5, 2.5], dtype=torch.float64)
    cpu = x64.to(dtype)
    getattr(cpu, op)()
    x = x64.to(dtype).to(mojo_gpu)
    _reset()
    with ran(f"aten::{out_key}"):
        getattr(x, op)()
    torch.testing.assert_close(x.cpu(), cpu)


def test_round_decimals_inplace_matches_cpu(mojo_gpu):
    """round_.decimals is a direct registration from before this chunk
    (not a fallback case): confirm it still reaches its own kernel."""
    x64 = torch.tensor([1.456, 2.567, -3.1415], dtype=torch.float64)
    cpu = x64.to(torch.float32)
    cpu.round_(decimals=2)
    x = x64.to(torch.float32).to(mojo_gpu)
    _reset()
    with ran("aten::round_.decimals"):
        x.round_(decimals=2)
    torch.testing.assert_close(x.cpu(), cpu, rtol=1e-4, atol=1e-5)


def test_frac_inplace_matches_cpu(mojo_gpu):
    x64 = torch.tensor([1.7, -2.7, 0.0, 3.25], dtype=torch.float64)
    cpu = x64.to(torch.float32)
    cpu.frac_()
    x = x64.to(torch.float32).to(mojo_gpu)
    _reset()
    with ran("aten::frac.out"):
        x.frac_()
    torch.testing.assert_close(x.cpu(), cpu, rtol=1e-4, atol=1e-5)


def test_bitwise_not_inplace_matches_cpu(mojo_gpu):
    x64 = torch.tensor([1, 2, 3, -4], dtype=torch.int32)
    cpu = x64.clone()
    cpu.bitwise_not_()
    x = x64.to(mojo_gpu)
    _reset()
    with ran("aten::bitwise_not.out"):
        x.bitwise_not_()
    assert torch.equal(x.cpu(), cpu)


# ---------------------------------------------------------------------------
# unary activations with their own gaps closed elsewhere: elu_, mish_,
# gelu_, silu_ -- single-tensor, already structured-fallback.
# ---------------------------------------------------------------------------

_ACTIVATIONS = [
    ("elu_", "elu.out"),
    ("mish_", "mish.out"),
    ("gelu_", "gelu.out"),
    ("silu_", "silu.out"),
]


@pytest.mark.parametrize("op,out_key", _ACTIVATIONS)
def test_activation_inplace_matches_cpu(mojo_gpu, op, out_key):
    # None of these four are exposed as bound Tensor methods in this torch
    # version (only their non-in-place twins are, e.g. `F.elu`); the ATen
    # op itself is still reachable through `torch.ops.aten`.
    fn = getattr(torch.ops.aten, op)
    x64 = torch.tensor([-2.0, -0.5, 0.0, 0.5, 2.0], dtype=torch.float64)
    cpu = x64.to(torch.float32)
    fn(cpu)
    x = x64.to(torch.float32).to(mojo_gpu)
    _reset()
    with ran(f"aten::{out_key}"):
        fn(x)
    torch.testing.assert_close(x.cpu(), cpu, rtol=1e-4, atol=1e-5)


# ---------------------------------------------------------------------------
# logit_: the one genuine gap. CPU/CUDA/XPU-only dispatch, no generic
# fallback -- registered directly as `op_logit_` for this chunk.
# ---------------------------------------------------------------------------


def test_logit_inplace_matches_cpu(mojo_gpu):
    x64 = torch.tensor([0.1, 0.3, 0.5, 0.7, 0.9], dtype=torch.float64)
    cpu = x64.to(torch.float32)
    cpu.logit_(eps=1e-4)
    x = x64.to(torch.float32).to(mojo_gpu)
    _reset()
    with ran("aten::logit_"):
        x.logit_(eps=1e-4)
    torch.testing.assert_close(x.cpu(), cpu, rtol=1e-4, atol=1e-5)


def test_logit_inplace_no_eps_matches_cpu(mojo_gpu):
    x64 = torch.tensor([0.1, 0.3, 0.5, 0.7, 0.9], dtype=torch.float64)
    cpu = x64.to(torch.float32)
    cpu.logit_()
    x = x64.to(torch.float32).to(mojo_gpu)
    x.logit_()
    torch.testing.assert_close(x.cpu(), cpu, rtol=1e-4, atol=1e-5)


def test_logit_inplace_noncontiguous(mojo_gpu):
    base64 = torch.tensor([[0.1, 0.9, 0.3], [0.5, 0.2, 0.7]], dtype=torch.float64)
    cpu_t = base64.to(torch.float32).t()
    dev_t = base64.to(torch.float32).to(mojo_gpu).t()
    assert not cpu_t.is_contiguous() and not dev_t.is_contiguous()
    cpu_t.logit_()
    dev_t.logit_()
    torch.testing.assert_close(dev_t.cpu(), cpu_t, rtol=1e-4, atol=1e-5)


@pytest.mark.parametrize("eps", [None, 0.0, 0.05, -1.0])
def test_logit_inplace_clamps_at_eps(mojo_gpu, eps):
    """Values outside [eps, 1 - eps] and outside (0, 1): clamped, inf or NaN
    exactly as CPU torch (a negative eps, like None, does not clamp)."""
    base = torch.tensor([-0.5, 0.0, 0.01, 0.04, 0.5, 0.96, 0.99, 1.0, 1.5])
    cpu = base.clone()
    cpu.logit_(eps=eps)
    x = base.to(mojo_gpu)
    x.logit_(eps=eps)
    torch.testing.assert_close(x.cpu(), cpu, rtol=1e-5, atol=1e-6, equal_nan=True)


def test_logit_inplace_empty_and_0d(mojo_gpu):
    for base in (torch.empty(0), torch.tensor(0.25)):
        cpu = base.clone()
        cpu.logit_()
        x = base.to(mojo_gpu)
        x.logit_()
        assert x.shape == cpu.shape
        torch.testing.assert_close(x.cpu(), cpu)


@pytest.mark.parametrize("dtype", [torch.int64, torch.int32, torch.bool])
@pytest.mark.parametrize("default", [torch.float32, torch.float64])
def test_logit_inplace_refuses_integer_self(mojo_gpu, dtype, default):
    previous = torch.get_default_dtype()
    torch.set_default_dtype(default)
    try:
        base = torch.tensor([0, 1, 1], dtype=dtype)
        with pytest.raises(RuntimeError) as cpu_err:
            base.clone().logit_()
        with pytest.raises(RuntimeError) as dev_err:
            base.to(mojo_gpu).logit_()
    finally:
        torch.set_default_dtype(previous)
    assert "can't be cast to the desired output type" in str(cpu_err.value)
    # the device appends the op it raised from: "... [aten::logit_]"
    assert str(dev_err.value).startswith(str(cpu_err.value).splitlines()[0])


@pytest.mark.parametrize(
    "call",
    [
        lambda t: t.div_(2.5, rounding_mode="trunc"),
        lambda t: t.div_(-2.5, rounding_mode="floor"),
        lambda t: t.fmod_(1.5),
        lambda t: t.copysign_(-1.0),
        lambda t: torch.ops.aten.xlogy_.Scalar_Other(t, 3.0),
    ],
    ids=[
        "div_Scalar_mode_trunc",
        "div_Scalar_mode_floor",
        "fmod_Scalar",
        "copysign_Scalar",
        "xlogy_Scalar_Other",
    ],
)
def test_scalar_overloads_inplace_match_cpu(mojo_gpu, call):
    base = torch.tensor([5.0, -6.0, 7.5, -8.5, 0.0, 0.25])
    cpu = base.clone()
    call(cpu)
    x = base.to(mojo_gpu)
    call(x)
    torch.testing.assert_close(x.cpu(), cpu, rtol=1e-5, atol=1e-6)


# ---------------------------------------------------------------------------
# binary: Tensor-Tensor in-place math ops with no fast path of their own.
# ---------------------------------------------------------------------------


def _pos(shape: tuple[int, ...]) -> torch.Tensor:
    return torch.empty(shape, dtype=torch.float64).uniform_(0.2, 4.0)


def _signed(shape: tuple[int, ...]) -> torch.Tensor:
    return torch.randn(shape, dtype=torch.float64) * 2


_ShapeFn = Callable[[tuple[int, ...]], torch.Tensor]
_BINARY_TENSOR_INPLACE: list[tuple[str, str, _ShapeFn, _ShapeFn]] = [
    ("atan2_", "atan2.out", _signed, _signed),
    ("copysign_", "copysign.out", _signed, _signed),
    ("fmod_", "fmod.Tensor_out", _signed, lambda s: _pos(s)),
    ("heaviside_", "heaviside.out", _signed, _signed),
    ("hypot_", "hypot.out", _signed, _signed),
    ("igamma_", "igamma.out", _pos, _pos),
    ("igammac_", "igammac.out", _pos, _pos),
    ("nextafter_", "nextafter.out", _signed, _signed),
    ("xlogy_", "xlogy.OutTensor", _pos, _pos),
]


@pytest.mark.parametrize("op,out_key,mk_self,mk_other", _BINARY_TENSOR_INPLACE)
def test_binary_tensor_inplace_matches_cpu(mojo_gpu, op, out_key, mk_self, mk_other):
    a64 = mk_self((3, 4))
    b64 = mk_other((3, 4))
    cpu = a64.to(torch.float32)
    other_cpu = b64.to(torch.float32)
    getattr(cpu, op)(other_cpu)
    x = a64.to(torch.float32).to(mojo_gpu)
    other = b64.to(torch.float32).to(mojo_gpu)
    _reset()
    with ran(f"aten::{out_key}"):
        getattr(x, op)(other)
    torch.testing.assert_close(x.cpu(), cpu, rtol=1e-4, atol=1e-5, equal_nan=True)


@pytest.mark.parametrize("op,out_key,mk_self,mk_other", _BINARY_TENSOR_INPLACE)
def test_binary_tensor_inplace_broadcast_violation_errors(
    mojo_gpu, op, out_key, mk_self, mk_other
):
    """`self` cannot be broadcast UP to a bigger operand shape -- an
    in-place op never resizes its destination."""
    x_cpu = mk_self((3,)).to(torch.float32)
    other_cpu = mk_other((2, 3)).to(torch.float32)
    with pytest.raises(RuntimeError, match="doesn't match the broadcast shape"):
        getattr(x_cpu.clone(), op)(other_cpu)
    x = x_cpu.to(mojo_gpu)
    other = other_cpu.to(mojo_gpu)
    with pytest.raises(RuntimeError, match="doesn't match the broadcast shape"):
        getattr(x, op)(other)


@pytest.mark.parametrize("op,out_key,mk_self,mk_other", _BINARY_TENSOR_INPLACE)
def test_binary_tensor_inplace_self_aliasing_other(
    mojo_gpu, op, out_key, mk_self, mk_other
):
    """`x.op_(x)`: self and the other operand are the SAME tensor."""
    a64 = mk_self((2, 3)) if mk_self is mk_other else mk_self((2, 3)).abs() + 0.3
    cpu = a64.to(torch.float32)
    getattr(cpu, op)(cpu)
    x = a64.to(torch.float32).to(mojo_gpu)
    getattr(x, op)(x)
    torch.testing.assert_close(x.cpu(), cpu, rtol=1e-4, atol=1e-5, equal_nan=True)


# ---------------------------------------------------------------------------
# binary: int-only (gcd_, lcm_) and bitwise (and_/or_/xor_, shifts).
# ---------------------------------------------------------------------------


@pytest.mark.parametrize("op,out_key", [("gcd_", "gcd.out"), ("lcm_", "lcm.out")])
def test_int_binary_inplace_matches_cpu(mojo_gpu, op, out_key):
    a = torch.tensor([12, 18, 7, 100], dtype=torch.int32)
    b = torch.tensor([8, 9, 3, 60], dtype=torch.int32)
    cpu = a.clone()
    getattr(cpu, op)(b)
    x = a.to(mojo_gpu)
    y = b.to(mojo_gpu)
    _reset()
    with ran(f"aten::{out_key}"):
        getattr(x, op)(y)
    assert torch.equal(x.cpu(), cpu)


@pytest.mark.parametrize(
    "op,out_key",
    [
        ("bitwise_and_", "bitwise_and.Tensor_out"),
        ("bitwise_or_", "bitwise_or.Tensor_out"),
        ("bitwise_xor_", "bitwise_xor.Tensor_out"),
    ],
)
def test_bitwise_tensor_inplace_matches_cpu(mojo_gpu, op, out_key):
    a = torch.tensor([0b1010, 0b1100, 0b0011], dtype=torch.int32)
    b = torch.tensor([0b1100, 0b1010, 0b1111], dtype=torch.int32)
    cpu = a.clone()
    getattr(cpu, op)(b)
    x = a.to(mojo_gpu)
    y = b.to(mojo_gpu)
    _reset()
    with ran(f"aten::{out_key}"):
        getattr(x, op)(y)
    assert torch.equal(x.cpu(), cpu)


@pytest.mark.parametrize("op", ["bitwise_and_", "bitwise_or_", "bitwise_xor_"])
def test_bitwise_scalar_inplace_matches_cpu(mojo_gpu, op):
    a = torch.tensor([0b1010, 0b1100, 0b0011], dtype=torch.int32)
    cpu = a.clone()
    getattr(cpu, op)(0b0110)
    x = a.to(mojo_gpu)
    getattr(x, op)(0b0110)
    assert torch.equal(x.cpu(), cpu)


@pytest.mark.parametrize(
    "op,out_key",
    [
        ("bitwise_left_shift_", "bitwise_left_shift.Tensor_out"),
        ("bitwise_right_shift_", "bitwise_right_shift.Tensor_out"),
    ],
)
def test_bitwise_shift_tensor_inplace_matches_cpu(mojo_gpu, op, out_key):
    a = torch.tensor([1, 4, 32, 100], dtype=torch.int32)
    b = torch.tensor([2, 1, 3, 2], dtype=torch.int32)
    cpu = a.clone()
    getattr(cpu, op)(b)
    x = a.to(mojo_gpu)
    y = b.to(mojo_gpu)
    _reset()
    with ran(f"aten::{out_key}"):
        getattr(x, op)(y)
    assert torch.equal(x.cpu(), cpu)


@pytest.mark.parametrize("op", ["bitwise_left_shift_", "bitwise_right_shift_"])
def test_bitwise_shift_scalar_inplace_matches_cpu(mojo_gpu, op):
    a = torch.tensor([1, 4, 32, 100], dtype=torch.int32)
    cpu = a.clone()
    getattr(cpu, op)(2)
    x = a.to(mojo_gpu)
    getattr(x, op)(2)
    assert torch.equal(x.cpu(), cpu)


# ---------------------------------------------------------------------------
# div_ / pow_ / remainder_: Tensor and Scalar overloads.
# ---------------------------------------------------------------------------


@pytest.mark.parametrize("mode", [None, "trunc", "floor"])
def test_div_tensor_inplace_matches_cpu(mojo_gpu, mode):
    a = torch.tensor([5.0, -6.0, 7.5, -8.5])
    b = torch.tensor([2.0, 3.0, -2.0, 4.0])
    kwargs = {} if mode is None else {"rounding_mode": mode}
    cpu = a.clone()
    cpu.div_(b, **kwargs)
    x = a.to(mojo_gpu)
    y = b.to(mojo_gpu)
    out_key = "div.out" if mode is None else "div.out_mode"
    _reset()
    with ran(f"aten::{out_key}"):
        x.div_(y, **kwargs)
    torch.testing.assert_close(x.cpu(), cpu, rtol=1e-4, atol=1e-5)


def test_div_scalar_inplace_matches_cpu(mojo_gpu):
    a = torch.tensor([5.0, -6.0, 7.5, -8.5])
    cpu = a.clone()
    cpu.div_(2.5)
    x = a.to(mojo_gpu)
    x.div_(2.5)
    torch.testing.assert_close(x.cpu(), cpu, rtol=1e-4, atol=1e-5)


def test_pow_tensor_inplace_matches_cpu(mojo_gpu):
    a = torch.tensor([2.0, 3.0, 4.0, 1.5])
    b = torch.tensor([2.0, 0.5, 3.0, 2.0])
    cpu = a.clone()
    cpu.pow_(b)
    x = a.to(mojo_gpu)
    y = b.to(mojo_gpu)
    _reset()
    with ran("aten::pow.Tensor_Tensor_out"):
        x.pow_(y)
    torch.testing.assert_close(x.cpu(), cpu, rtol=1e-4, atol=1e-5)


def test_pow_scalar_inplace_matches_cpu(mojo_gpu):
    a = torch.tensor([2.0, 3.0, 4.0, 1.5])
    cpu = a.clone()
    cpu.pow_(2.0)
    x = a.to(mojo_gpu)
    _reset()
    with ran("aten::pow.Tensor_Scalar_out"):
        x.pow_(2.0)
    torch.testing.assert_close(x.cpu(), cpu, rtol=1e-4, atol=1e-5)


def test_remainder_tensor_inplace_matches_cpu(mojo_gpu):
    a = torch.tensor([5.0, -6.0, 7.5, -8.5])
    b = torch.tensor([3.0, 3.0, 2.0, 4.0])
    cpu = a.clone()
    cpu.remainder_(b)
    x = a.to(mojo_gpu)
    y = b.to(mojo_gpu)
    _reset()
    with ran("aten::remainder.Tensor_out"):
        x.remainder_(y)
    torch.testing.assert_close(x.cpu(), cpu, rtol=1e-4, atol=1e-5)


def test_remainder_scalar_inplace_matches_cpu(mojo_gpu):
    a = torch.tensor([5.0, -6.0, 7.5, -8.5])
    cpu = a.clone()
    cpu.remainder_(3.0)
    x = a.to(mojo_gpu)
    x.remainder_(3.0)
    torch.testing.assert_close(x.cpu(), cpu, rtol=1e-4, atol=1e-5)


# ---------------------------------------------------------------------------
# comparisons: eq_ / ne_ / lt_ / le_ / gt_ / ge_, Tensor and Scalar.
# ---------------------------------------------------------------------------

_COMPARE_INPLACE = [
    ("eq_", "eq"),
    ("ne_", "ne"),
    ("lt_", "lt"),
    ("le_", "le"),
    ("gt_", "gt"),
    ("ge_", "ge"),
]


@pytest.mark.parametrize("dtype", [torch.int64, torch.float32])
@pytest.mark.parametrize("op,base", _COMPARE_INPLACE)
def test_compare_tensor_inplace_matches_cpu(mojo_gpu, op, base, dtype):
    a = torch.tensor([1, 2, 3, 4], dtype=dtype)
    b = torch.tensor([1, 0, 3, 5], dtype=dtype)
    cpu = a.clone()
    getattr(cpu, op)(b)
    x = a.to(mojo_gpu)
    y = b.to(mojo_gpu)
    _reset()
    with ran(f"aten::{base}.Tensor_out"):
        getattr(x, op)(y)
    assert torch.equal(x.cpu(), cpu) and x.dtype == cpu.dtype == dtype


@pytest.mark.parametrize("op,base", _COMPARE_INPLACE)
def test_compare_scalar_inplace_matches_cpu(mojo_gpu, op, base):
    a = torch.tensor([1, 2, 3, 4], dtype=torch.int64)
    cpu = a.clone()
    getattr(cpu, op)(2)
    x = a.to(mojo_gpu)
    getattr(x, op)(2)
    assert torch.equal(x.cpu(), cpu)


def test_compare_inplace_casts_bool_result_into_self_dtype(mojo_gpu):
    """eq_/etc. compute a bool result and cast 0/1 back into self's own
    dtype (always safe -- any numeric dtype holds 0 or 1), unlike the
    promoting unary family above."""
    a = torch.tensor([1.0, 2.0, 3.0])
    b = torch.tensor([1.0, 0.0, 3.0])
    cpu = a.clone()
    cpu.eq_(b)
    x = a.to(mojo_gpu)
    x.eq_(b.to(mojo_gpu))
    assert torch.equal(x.cpu(), cpu) and x.dtype == torch.float32


# ---------------------------------------------------------------------------
# clamp_ / clamp_max_ / clamp_min_: Scalar and Tensor bounds.
# ---------------------------------------------------------------------------


@pytest.mark.parametrize(
    "op,out_key,kwargs",
    [
        ("clamp_", "clamp.out", {"min": 2.0, "max": 8.0}),
        ("clamp_max_", "clamp_max.out", {}),
        ("clamp_min_", "clamp_min.out", {}),
    ],
)
def test_clamp_scalar_inplace_matches_cpu(mojo_gpu, op, out_key, kwargs):
    a = torch.tensor([1.0, 5.0, 10.0, -3.0])
    bound_kwargs = kwargs if kwargs else {("max" if "max" in op else "min"): 8.0}
    cpu = a.clone()
    getattr(cpu, op)(**bound_kwargs)
    x = a.to(mojo_gpu)
    _reset()
    with ran(f"aten::{out_key}"):
        getattr(x, op)(**bound_kwargs)
    torch.testing.assert_close(x.cpu(), cpu)


@pytest.mark.parametrize(
    "op,out_key",
    [("clamp_max_", "clamp_max.Tensor_out"), ("clamp_min_", "clamp_min.Tensor_out")],
)
def test_clamp_max_min_tensor_inplace_matches_cpu(mojo_gpu, op, out_key):
    a = torch.tensor([1.0, 5.0, 10.0, -3.0])
    bound = torch.tensor([8.0, 8.0, 8.0, 8.0])
    cpu = a.clone()
    getattr(cpu, op)(bound)
    x = a.to(mojo_gpu)
    y = bound.to(mojo_gpu)
    _reset()
    with ran(f"aten::{out_key}"):
        getattr(x, op)(y)
    torch.testing.assert_close(x.cpu(), cpu)


def test_clamp_tensor_both_bounds_inplace_matches_cpu(mojo_gpu):
    a = torch.tensor([1.0, 5.0, 10.0, -3.0])
    lo = torch.tensor([0.0, 0.0, 0.0, 0.0])
    hi = torch.tensor([8.0, 8.0, 8.0, 8.0])
    cpu = a.clone()
    cpu.clamp_(min=lo, max=hi)
    x = a.to(mojo_gpu)
    _reset()
    with ran("aten::clamp.Tensor_out"):
        x.clamp_(min=lo.to(mojo_gpu), max=hi.to(mojo_gpu))
    torch.testing.assert_close(x.cpu(), cpu)


def test_clamp_tensor_one_bound_inplace_matches_cpu(mojo_gpu):
    """clamp_.Tensor with only one bound given routes through the
    maximum_stub / minimum_stub pair instead of the both-bounds iterator."""
    a = torch.tensor([1.0, 5.0, 10.0, -3.0])
    lo = torch.tensor([0.0, 0.0, 0.0, 0.0])
    cpu = a.clone()
    cpu.clamp_(min=lo)
    x = a.to(mojo_gpu)
    x.clamp_(min=lo.to(mojo_gpu))
    torch.testing.assert_close(x.cpu(), cpu)


# ---------------------------------------------------------------------------
# lerp_.Tensor (lerp_.Scalar was already registered before this chunk).
# ---------------------------------------------------------------------------


def test_lerp_tensor_inplace_matches_cpu(mojo_gpu):
    start = torch.tensor([0.0, 1.0, 2.0, 3.0])
    end = torch.tensor([10.0, 10.0, 10.0, 10.0])
    weight = torch.tensor([0.0, 0.25, 0.5, 1.0])
    cpu = start.clone()
    cpu.lerp_(end, weight)
    x = start.to(mojo_gpu)
    _reset()
    with ran("aten::lerp.Tensor_out"):
        x.lerp_(end.to(mojo_gpu), weight.to(mojo_gpu))
    torch.testing.assert_close(x.cpu(), cpu)
