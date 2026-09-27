"""The `out=` and in-place contract shared by every elementwise op, checked
over each registered `.out` overload against CPU torch.

TensorIterator gives every elementwise `out=` op the same meta checks, and
the mojo ops reproduce them in shared helpers (`_b_out_guard`,
`_b_store_out`, `_unary_out`, `_pw_out_of`), so they are tested here once,
generically, rather than op by op:

* an `out` with internal overlap (`torch.empty(1).expand(3)`) raises
  (`at::assert_no_internal_overlap`), as does one partially overlapping an
  input (`at::assert_no_partial_overlap`);
* an `out` that IS one of the inputs -- how `self.op_(other)` reaches the
  `.out` kernel -- is never resized: a broadcast shape larger than it raises
  ("output with shape [1, 3] doesn't match the broadcast shape [2, 3]");
* the result is cast into `out` exactly where torch casts it, and the out
  dtypes torch refuses are refused;
* a wrong-shaped `out` is resized, a strided one written where it lives;
* the value types agree with `torch.result_type` for the binary ops.

The table lists every elementwise `.out` overload; a case whose overload is
not registered on the mojo device (yet) is skipped, so each op PR of a stack
turns its own cases on. Every run asserts with `native.op_count` that the
registration under test ran: torch's composite kernels can route one
overload through another (`Scalar_out` through `Tensor_out`), which would
hide a missing registration.
"""

import contextlib
import zlib
from collections.abc import Callable, Iterator, Mapping
from typing import Any, NoReturn

import pytest
import torch

from torch_mojo_backend import get_accelerators, native

# The elementwise overloads writing one caller-supplied output.
OUT_OVERLOADS = [
    "abs.out",
    "acos.out",
    "acosh.out",
    "add.out",
    "addcdiv.out",
    "addcmul.out",
    "angle.out",
    "asin.out",
    "asinh.out",
    "atan.out",
    "atan2.out",
    "atanh.out",
    "binary_cross_entropy.out",
    "binary_cross_entropy_backward.grad_input",
    "binary_cross_entropy_with_logits.out",
    "bitwise_and.Scalar_out",
    "bitwise_and.Tensor_out",
    "bitwise_left_shift.Tensor_out",
    "bitwise_not.out",
    "bitwise_or.Scalar_out",
    "bitwise_or.Tensor_out",
    "bitwise_right_shift.Tensor_out",
    "bitwise_xor.Scalar_out",
    "bitwise_xor.Tensor_out",
    "ceil.out",
    "clamp.Tensor_out",
    "clamp.out",
    "clamp_max.Tensor_out",
    "clamp_max.out",
    "clamp_min.Tensor_out",
    "clamp_min.out",
    "copysign.Scalar_out",
    "copysign.out",
    "cos.out",
    "cosh.out",
    "deg2rad.out",
    "div.out",
    "div.out_mode",
    "elu.out",
    "elu_backward.grad_input",
    "erf.out",
    "erfc.out",
    "erfinv.out",
    "exp.out",
    "exp2.out",
    "expm1.out",
    "floor.out",
    "fmax.out",
    "fmin.out",
    "fmod.Scalar_out",
    "fmod.Tensor_out",
    "gcd.out",
    "gelu.out",
    "gelu_backward.grad_input",
    "hardshrink.out",
    "hardshrink_backward.grad_input",
    "hardsigmoid.out",
    "hardsigmoid_backward.grad_input",
    "hardswish.out",
    "hardtanh.out",
    "hardtanh_backward.grad_input",
    "heaviside.out",
    "huber_loss.out",
    "huber_loss_backward.out",
    "hypot.out",
    "igamma.out",
    "igammac.out",
    "isinf.out",
    "isnan.out",
    "isneginf.out",
    "isposinf.out",
    "lcm.out",
    "ldexp.out",
    "leaky_relu.out",
    "leaky_relu_backward.grad_input",
    "lerp.Scalar_out",
    "lerp.Tensor_out",
    "log.out",
    "log10.out",
    "log1p.out",
    "log2.out",
    "log_sigmoid_backward.grad_input",
    "logaddexp.out",
    "logaddexp2.out",
    "logical_and.out",
    "logical_not.out",
    "logical_or.out",
    "logical_xor.out",
    "maximum.out",
    "minimum.out",
    "mish.out",
    "mse_loss.out",
    "mse_loss_backward.grad_input",
    "mul.out",
    "nan_to_num.out",
    "neg.out",
    "nextafter.out",
    "pow.Scalar_out",
    "pow.Tensor_Scalar_out",
    "pow.Tensor_Tensor_out",
    "rad2deg.out",
    "reciprocal.out",
    "relu.out",
    "remainder.Scalar_out",
    "remainder.Tensor_out",
    "rsqrt.out",
    "rsub.Scalar_out",
    "rsub.Tensor_out",
    "sgn.out",
    "sigmoid.out",
    "sign.out",
    "signbit.out",
    "silu.out",
    "silu_backward.grad_input",
    "sin.out",
    "sinc.out",
    "sinh.out",
    "smooth_l1_loss.out",
    "smooth_l1_loss_backward.grad_input",
    "softplus.out",
    "softplus_backward.grad_input",
    "softshrink.out",
    "softshrink_backward.grad_input",
    "special_airy_ai.out",
    "special_bessel_j0.out",
    "special_bessel_j1.out",
    "special_bessel_y0.out",
    "special_bessel_y1.out",
    "special_chebyshev_polynomial_t.out",
    "special_chebyshev_polynomial_u.out",
    "special_chebyshev_polynomial_v.out",
    "special_chebyshev_polynomial_w.out",
    "special_entr.out",
    "special_erfcx.out",
    "special_hermite_polynomial_h.out",
    "special_hermite_polynomial_he.out",
    "special_i0e.out",
    "special_i1.out",
    "special_i1e.out",
    "special_laguerre_polynomial_l.out",
    "special_legendre_polynomial_p.out",
    "special_log_ndtr.out",
    "special_modified_bessel_i0.out",
    "special_modified_bessel_i1.out",
    "special_modified_bessel_k0.out",
    "special_modified_bessel_k1.out",
    "special_ndtri.out",
    "special_scaled_modified_bessel_k0.out",
    "special_scaled_modified_bessel_k1.out",
    "special_shifted_chebyshev_polynomial_t.out",
    "special_shifted_chebyshev_polynomial_u.out",
    "special_shifted_chebyshev_polynomial_v.out",
    "special_shifted_chebyshev_polynomial_w.out",
    "special_spherical_bessel_j0.out",
    "special_xlog1py.out",
    "special_zeta.out",
    "sqrt.out",
    "sub.out",
    "tan.out",
    "tanh.out",
    "threshold.out",
    "xlogy.OutTensor",
]

# Ops whose operands are integers (their float forms are a dtype error).
_INT_OPS = {
    "bitwise_and",
    "bitwise_left_shift",
    "bitwise_not",
    "bitwise_or",
    "bitwise_right_shift",
    "bitwise_xor",
    "gcd",
    "lcm",
}

# Required non-tensor arguments, by name (anything with a default is left
# to it). `reduction` 0 keeps the losses elementwise.
_SCALARS: dict[str, float | int | bool | str] = {
    "alpha": 1.0,
    "beta": 1.0,
    "delta": 1.0,
    "exponent": 1.5,
    "input_scale": 1.0,
    "is_result": False,
    "lambd": 0.5,
    "max": 0.6,
    "max_val": 1.0,
    "min": 0.3,
    "min_val": -1.0,
    "negative_slope": 0.01,
    "other": 0.75,
    "reduction": 0,
    "rounding_mode": "floor",
    "scale": 1.0,
    "self_is_result": False,
    "threshold": 0.5,
    "value": 0.25,
    "weight": 0.25,
}
_INT_SCALARS: dict[str, int] = {"other": 3}
_POLY = "polynomial"

# An op's keyword arguments: tensors and scalars, by its schema.
Kwargs = dict[str, Any]


def _schema(name: str) -> torch.FunctionSchema:
    packet, overload = name.split(".")
    return getattr(getattr(torch.ops.aten, packet), overload)._schema


def _op(name: str) -> Callable[..., torch.Tensor]:
    packet, overload = name.split(".")
    return getattr(getattr(torch.ops.aten, packet), overload)


def _out_name(name: str) -> str:
    (out,) = [
        a.name
        for a in _schema(name).arguments
        if a.kwarg_only and a.alias_info is not None and a.alias_info.is_write
    ]
    return out


def _tensor_args(name: str) -> list[str]:
    """The tensor inputs, in schema order (an optional `weight` left out)."""
    return [
        a.name
        for a in _schema(name).arguments
        if str(a.type) in ("Tensor", "Optional[Tensor]")
        and not (a.kwarg_only and a.alias_info is not None)
        and not (str(a.type) != "Tensor" and "weight" in a.name)
    ]


def _is_int_op(name: str) -> bool:
    return name.split(".")[0] in _INT_OPS


def _make(name: str, arg: str, shape: tuple[int, ...], dtype: torch.dtype):
    g = torch.Generator().manual_seed(zlib.crc32(f"{name}/{arg}/{shape}".encode()))
    if _is_int_op(name):
        return torch.randint(1, 8, shape, generator=g, dtype=dtype)
    if _POLY in name and arg == "n":
        return torch.randint(0, 5, shape, generator=g).to(dtype)
    x = torch.rand(shape, generator=g, dtype=torch.float32) * 0.6 + 0.2
    return x.to(dtype)


def _cpu_args(
    name: str,
    shapes: Mapping[str, tuple[int, ...]] | None = None,
    dtype: torch.dtype | None = None,
) -> Kwargs:
    """CPU keyword arguments of `name` (the out argument left out): tensors
    of `dtype` (float32 / int64 by default) in (0.2, 0.8), scalars from
    `_SCALARS`."""
    dtype = dtype or (torch.int64 if _is_int_op(name) else torch.float32)
    shapes = shapes or {}
    tensors = _tensor_args(name)
    kwargs = {}
    for a in _schema(name).arguments:
        if a.kwarg_only and a.alias_info is not None and a.alias_info.is_write:
            continue
        if a.name in tensors:
            kwargs[a.name] = _make(name, a.name, shapes.get(a.name, (2, 3)), dtype)
        elif name.startswith("pow.Scalar_out") and a.name == "self":
            kwargs[a.name] = 2.0
        elif a.name == "reduction":
            kwargs[a.name] = 0
        elif a.has_default_value() and a.name not in ("min",):
            continue
        elif _is_int_op(name) and a.name in _INT_SCALARS:
            kwargs[a.name] = _INT_SCALARS[a.name]
        elif a.name in _SCALARS:
            kwargs[a.name] = _SCALARS[a.name]
        else:
            raise AssertionError(f"no test value for {name}({a.name})")
    if name.startswith("log_sigmoid_backward"):
        # The CPU kernel reads the forward's buffer (CUDA's recomputes it).
        kwargs["buffer"] = torch.ops.aten.log_sigmoid_forward(kwargs["self"])[1]
    return kwargs


def _to(kwargs: Kwargs, device: str) -> Kwargs:
    return {
        k: v.to(device) if isinstance(v, torch.Tensor) else v for k, v in kwargs.items()
    }


def _registered_or_skip(name: str):
    if not torch._C._dispatch_has_kernel_for_dispatch_key(
        f"aten::{name}", "PrivateUse1"
    ):
        pytest.skip(f"aten::{name} is not registered on the mojo device")


def _metal(device: str) -> bool:
    idx = int(device.rsplit(":", 1)[-1])
    accelerators = list(get_accelerators())
    return idx < len(accelerators) and accelerators[idx].api == "metal"


@contextlib.contextmanager
def _ran(name: str) -> Iterator[None]:
    """The registration of `name` itself ran (whether or not it raised)."""
    native.op_counting(True)
    key = f"aten::{name}"
    before = native.op_count(key)
    try:
        yield
    finally:
        assert native.op_count(key) > before, f"{key} did not run natively"


def _outcome(fn: Callable[[], torch.Tensor]) -> torch.Tensor | Exception:
    try:
        return fn()
    except (RuntimeError, TypeError, ValueError, NotImplementedError) as e:
        return e


def _same_outcome(name, cpu, ours):
    """Both raised, or both produced the same values in the same dtype."""
    if isinstance(cpu, Exception):
        assert isinstance(ours, Exception), (
            f"{name}: CPU torch raised ({cpu}) but the mojo op did not"
        )
        assert not isinstance(ours, NotImplementedError), (
            f"{name}: declined ({ours}) where torch raises ({cpu})"
        )
        return
    if isinstance(ours, Exception):
        raise AssertionError(f"{name}: raised {ours!r}; CPU torch returned") from ours
    half = cpu.dtype in (torch.float16, torch.bfloat16)
    rtol, atol = (1e-2, 1e-3) if half else (1e-4, 1e-5)
    torch.testing.assert_close(ours.cpu(), cpu, equal_nan=True, rtol=rtol, atol=atol)


def _inputs(schema: torch.FunctionSchema) -> list[tuple[str, str]]:
    return [
        (a.name, str(a.type))
        for a in schema.arguments
        if not (a.kwarg_only and a.alias_info is not None and a.alias_info.is_write)
    ]


def _functional(name: str) -> Callable[..., torch.Tensor]:
    """The functional overload `name` is the `out=` form of: the one taking
    the same inputs."""
    packet, _ = name.split(".")
    ops = getattr(torch.ops.aten, packet)
    want = _inputs(_schema(name))
    for overload in ops.overloads():
        op = getattr(ops, overload)
        if (
            not any(a.is_out for a in op._schema.arguments)
            and _inputs(op._schema) == want
        ):
            return op
    raise AssertionError(f"no functional overload for {name}")


def _torch_does_not_check(name: str) -> NoReturn:
    pytest.skip(f"CPU torch does not make this check for {name}")


def _cpu_result(name: str, kwargs: Kwargs) -> torch.Tensor:
    """The functional op on CPU: torch's own result shape and dtype."""
    return _functional(name)(**kwargs)


def _natural_dtype(name: str, kwargs: Kwargs) -> torch.dtype:
    return _cpu_result(name, kwargs).dtype


_CASES = [pytest.param(n, id=n) for n in OUT_OVERLOADS]


@pytest.mark.parametrize("name", _CASES)
def test_out_resized_and_strided(mojo_gpu, name):
    """A wrong-shaped `out` is resized; a correctly shaped strided view of a
    larger buffer is written where it lives and nothing else is touched."""
    _registered_or_skip(name)
    kwargs = _cpu_args(name)
    expected = _cpu_result(name, kwargs)
    dtype = expected.dtype
    out = torch.empty(0, dtype=dtype, device=mojo_gpu)
    with _ran(name):
        got = _op(name)(**_to(kwargs, mojo_gpu), **{_out_name(name): out})
    assert got.shape == expected.shape
    _same_outcome(name, expected, got)
    # A transposed out and a row slice of a larger buffer.
    sentinel = torch.full((4, 3), 7).to(dtype)
    base = sentinel.to(mojo_gpu)
    strided = base[1:3]
    with _ran(name):
        _op(name)(**_to(kwargs, mojo_gpu), **{_out_name(name): strided})
    _same_outcome(name, expected, strided)
    assert torch.equal(base[0::3].cpu(), sentinel[0::3])
    transposed = torch.empty((3, 2), dtype=dtype, device=mojo_gpu).t()
    with _ran(name):
        _op(name)(**_to(kwargs, mojo_gpu), **{_out_name(name): transposed})
    _same_outcome(name, expected, transposed)


@pytest.mark.parametrize("name", _CASES)
def test_out_internal_overlap_raises(mojo_gpu, name):
    """`at::assert_no_internal_overlap`: an `out` whose elements share one
    memory location would be written by racing threads."""
    _registered_or_skip(name)
    kwargs = _cpu_args(name)
    dtype = _natural_dtype(name, kwargs)
    cpu_out = torch.empty(1, dtype=dtype).expand(2, 3)
    cpu = _outcome(lambda: _op(name)(**kwargs, **{_out_name(name): cpu_out}))
    if not isinstance(cpu, RuntimeError):
        # threshold's iterator runs with check_mem_overlap(false); the mojo
        # op still refuses the racing writes.
        _torch_does_not_check(name)
    out = torch.empty(1, dtype=dtype, device=mojo_gpu).expand(2, 3)
    with _ran(name), pytest.raises(RuntimeError, match="single memory location"):
        _op(name)(**_to(kwargs, mojo_gpu), **{_out_name(name): out})


@pytest.mark.parametrize("name", _CASES)
def test_out_partial_overlap_raises(mojo_gpu, name):
    """`at::assert_no_partial_overlap`: an `out` sharing some, but not all,
    of an input's memory."""
    _registered_or_skip(name)
    kwargs = _cpu_args(name)
    dtype = _natural_dtype(name, kwargs)
    first = _tensor_args(name)[0]
    if kwargs[first].dtype != dtype:
        pytest.skip("the result dtype differs from the input's")
    for device in ("cpu", mojo_gpu):
        base = _make(name, first, (7,), dtype).to(device)
        args = _to(kwargs, device)
        args[first] = base[:6].view(2, 3)
        out = base[1:7].view(2, 3)
        if device == "cpu":
            cpu = _outcome(lambda: _op(name)(**args, **{_out_name(name): out}))
            if not isinstance(cpu, RuntimeError):
                _torch_does_not_check(name)
            continue
        with _ran(name), pytest.raises(RuntimeError, match="single memory location"):
            _op(name)(**args, **{_out_name(name): out})


@pytest.mark.parametrize("name", _CASES)
def test_out_aliasing_an_input_is_never_resized(mojo_gpu, name):
    """`self.op_(other)` reaches the `.out` kernel with `out=self`: a
    broadcast shape larger than self raises instead of resizing it, and one
    that fits writes self in place."""
    _registered_or_skip(name)
    tensors = _tensor_args(name)
    if len(tensors) < 2:
        pytest.skip("one tensor input: the broadcast shape is self's")
    first = tensors[0]
    small = _cpu_args(name, shapes={first: (1, 3)})
    natural = _outcome(lambda: _cpu_result(name, small))
    if not isinstance(natural, torch.Tensor):
        _torch_does_not_check(name)  # CPU's oneDNN gelu_backward
    dtype = natural.dtype
    if small[first].dtype != dtype:
        pytest.skip("the result dtype differs from the input's")
    cpu = _outcome(lambda: _op(name)(**small, **{_out_name(name): small[first]}))
    if not isinstance(cpu, RuntimeError):
        # Not a TensorIterator op (the losses, rsub's composite): torch
        # resizes or broadcasts on its own terms there.
        _torch_does_not_check(name)
    args = _to(small, mojo_gpu)
    with (
        _ran(name),
        pytest.raises(RuntimeError, match="doesn't match the broadcast shape"),
    ):
        _op(name)(**args, **{_out_name(name): args[first]})
    assert args[first].shape == (1, 3)
    # Every other operand broadcasting into self: in place.
    big = _cpu_args(name, shapes={t: (1, 3) for t in tensors[1:]})
    expected = _op(name)(**big, **{_out_name(name): big[first].clone()})
    args = _to(big, mojo_gpu)
    with _ran(name):
        got = _op(name)(**args, **{_out_name(name): args[first]})
    assert got.data_ptr() == args[first].data_ptr()
    _same_outcome(name, expected, args[first])


_OUT_DTYPES = [torch.float16, torch.float32, torch.float64, torch.int64, torch.bool]


@pytest.mark.parametrize("out_dtype", _OUT_DTYPES)
@pytest.mark.parametrize("name", _CASES)
def test_out_dtype_follows_torch(mojo_gpu, name, out_dtype):
    """The result is cast into an `out` of another dtype exactly where torch
    casts it (`canCast`, or an op's own exact-dtype rule), and refused where
    torch refuses it."""
    _registered_or_skip(name)
    if out_dtype == torch.float64 and _metal(mojo_gpu):
        pytest.skip("Apple GPUs have no float64")
    if out_dtype == torch.float64 and name == "gelu_backward.grad_input":
        pytest.skip("CPU torch's oneDNN gelu_backward has no float64 output")
    kwargs = _cpu_args(name)
    cpu = _outcome(
        lambda: _op(name)(
            **kwargs, **{_out_name(name): torch.empty(0, dtype=out_dtype)}
        )
    )
    out = torch.empty(0, dtype=out_dtype, device=mojo_gpu)
    with _ran(name):
        ours = _outcome(
            lambda: _op(name)(**_to(kwargs, mojo_gpu), **{_out_name(name): out})
        )
    if isinstance(cpu, torch.Tensor):
        # The functional result cast: CPU's own out= of a loss with
        # reduction='none' is left unresized (binary_cross_entropy_out_cpu).
        cpu = _cpu_result(name, kwargs).to(out_dtype)
    _same_outcome(name, cpu, ours)
    if isinstance(ours, torch.Tensor):
        assert ours.dtype == out_dtype


# The registered in-place overloads of the elementwise batch (torch calls
# these directly; the composite ones reach a `.out` above with out=self).
INPLACE_OVERLOADS = [
    "__ilshift__.Tensor",
    "__irshift__.Tensor",
    "add_.Tensor",
    "addcdiv_.default",
    "addcmul_.default",
    "deg2rad_.default",
    "hardsigmoid_.default",
    "hardswish_.default",
    "hardtanh_.default",
    "ldexp_.default",
    "leaky_relu_.default",
    "lerp_.Scalar",
    "mul_.Scalar",
    "mul_.Tensor",
    "rad2deg_.default",
    "relu_.default",
    "sub_.Tensor",
    "threshold_.default",
]


def _inplace_args(name: str, shapes: dict[str, tuple[int, ...]], dtype=None):
    """`_cpu_args` for an in-place overload (its `self` is the output)."""
    schema = _schema(name)
    kwargs = {}
    int_op = name.startswith(("__ilshift__", "__irshift__"))
    dtype = dtype or (torch.int64 if int_op else torch.float32)
    for a in schema.arguments:
        if str(a.type) == "Tensor":
            g = torch.Generator().manual_seed(zlib.crc32(f"{name}/{a.name}".encode()))
            shape = shapes.get(a.name, (2, 3))
            if int_op:
                kwargs[a.name] = torch.randint(1, 4, shape, generator=g, dtype=dtype)
            else:
                kwargs[a.name] = (torch.rand(shape, generator=g) * 0.6 + 0.2).to(dtype)
        elif a.has_default_value():
            continue
        elif a.name in _SCALARS:
            kwargs[a.name] = _SCALARS[a.name]
        else:
            raise AssertionError(f"no test value for {name}({a.name})")
    return kwargs


@pytest.mark.parametrize("name", [pytest.param(n, id=n) for n in INPLACE_OVERLOADS])
def test_inplace_never_resizes_and_casts_like_torch(mojo_gpu, name):
    """`self.op_(other)`: an operand broadcasting self to a larger shape
    raises, one broadcasting into self writes it in place, and a result of
    another dtype is cast into self exactly where torch casts it."""
    if not torch._C._dispatch_has_kernel_for_dispatch_key(
        f"aten::{name.removesuffix('.default')}", "PrivateUse1"
    ):
        pytest.skip(f"aten::{name} is not registered on the mojo device")
    key = f"aten::{name.removesuffix('.default')}"
    tensors = [a.name for a in _schema(name).arguments if str(a.type) == "Tensor"]
    op = _op(name)
    if len(tensors) > 1:
        small = _inplace_args(name, {"self": (1, 3)})
        assert isinstance(_outcome(lambda: op(**small)), RuntimeError)
        args = _to(small, mojo_gpu)
        native.op_counting(True)
        before = native.op_count(key)
        with pytest.raises(RuntimeError):
            op(**args)
        assert native.op_count(key) > before, f"{key} did not run natively"
        assert args["self"].shape == (1, 3)
    fits = _inplace_args(name, {t: (1, 3) for t in tensors[1:]})
    expected = op(
        **{k: v.clone() if isinstance(v, torch.Tensor) else v for k, v in fits.items()}
    )
    args = _to(fits, mojo_gpu)
    self_ptr = args["self"].data_ptr()
    native.op_counting(True)
    before = native.op_count(key)
    got = op(**args)
    assert native.op_count(key) > before, f"{key} did not run natively"
    assert got.data_ptr() == self_ptr
    _same_outcome(name, expected, args["self"])
    # A float16 self with float32 operands: cast into self (a float32
    # result can't be cast into an int64 self, which raises).
    # (addcmul_ / addcdiv_ decline operands of mixed dtypes altogether.)
    if len(tensors) > 1 and not name.startswith(("__i", "lerp_", "addc")):
        for self_dtype in (torch.float16, torch.int64):
            mixed = dict(_inplace_args(name, {}))
            mixed["self"] = mixed["self"].to(self_dtype)
            cpu = _outcome(
                lambda: op(
                    **{
                        k: v.clone() if isinstance(v, torch.Tensor) else v
                        for k, v in mixed.items()
                    }
                )
            )
            args = _to(mixed, mojo_gpu)
            ours = _outcome(lambda: op(**args))
            _same_outcome(name, cpu, ours)


# Operand pairs whose promotion differs by rank: a dimensioned tensor
# outranks a 0-dim one of the same category, which outranks a Python
# number; a higher category (bool < integral < floating) always wins.
_PROMO_DTYPES = [
    torch.float16,
    torch.bfloat16,
    torch.float32,
    torch.float64,
    torch.int32,
    torch.int64,
    torch.bool,
]
# "cpu" is an explicit 0-dim CPU tensor (not a wrapped Python number: it
# promotes as a 0-dim tensor), "py" a Python number.
_PROMO_SHAPES = [
    ((2, 3), ()),
    ((), (2, 3)),
    ((2, 3), (1, 3)),
    ((2, 3), "cpu"),
    ((), "cpu"),
]

# Binary functional overloads covering every promotion route (the binary
# cascade, the pointwise family, the logical ops).
PROMO_OVERLOADS = [
    "__lshift__.Tensor",
    "atan2.default",
    "copysign.Tensor",
    "fmax.default",
    "hypot.default",
    "logaddexp.default",
    "maximum.default",
    "mul.Tensor",
    "nextafter.default",
    "remainder.Tensor",
    "rsub.Tensor",
    "sub.Tensor",
    "special_xlog1py.default",
    "xlogy.Tensor",
]


def _promo_input(dtype: torch.dtype, shape: tuple[int, ...], seed: int):
    g = torch.Generator().manual_seed(seed)
    if seed == 2 and dtype in (torch.int32, torch.int64):
        # A shift of 40 overflows an int32 result, not an int64 one.
        return torch.full(shape, 40, dtype=dtype)
    if dtype == torch.bool:
        return torch.rand(shape, generator=g) > 0.5
    if dtype.is_floating_point:
        return (torch.rand(shape, generator=g) * 3 + 0.5).to(dtype)
    return torch.randint(1, 5, shape, generator=g, dtype=dtype)


@pytest.mark.parametrize(
    "shapes",
    _PROMO_SHAPES,
    ids=["dim-0d", "0d-dim", "dim-dim", "dim-cpu0d", "0d-cpu0d"],
)
@pytest.mark.parametrize("name", [pytest.param(n, id=n) for n in PROMO_OVERLOADS])
def test_result_type_matches_torch(mojo_gpu, name, shapes):
    """The result dtype is `torch.result_type(a, b)` for every dtype pair,
    rank-aware (`float16[2, 3]` with a `float32[]` is float16), and the
    values follow."""
    key = f"aten::{name.removesuffix('.default')}"
    if not torch._C._dispatch_has_kernel_for_dispatch_key(key, "PrivateUse1"):
        pytest.skip(f"{key} is not registered on the mojo device")
    op = _op(name)
    metal = _metal(mojo_gpu)
    wrong = []
    for a_dtype in _PROMO_DTYPES:
        for b_dtype in _PROMO_DTYPES:
            if metal and torch.float64 in (a_dtype, b_dtype):
                continue
            a = _promo_input(a_dtype, shapes[0], 1)
            b_on_cpu = shapes[1] == "cpu"
            b = _promo_input(b_dtype, () if b_on_cpu else shapes[1], 2)
            cpu = _outcome(lambda: op(a, b))
            if isinstance(cpu, Exception):
                continue
            b_dev = b if b_on_cpu else b.to(mojo_gpu)
            ours = _outcome(lambda: op(a.to(mojo_gpu), b_dev))
            if isinstance(ours, NotImplementedError):
                continue  # a declined pair is not a wrong one
            if isinstance(ours, Exception):
                wrong.append(f"{a_dtype}/{b_dtype}: raised {ours}")
            elif ours.dtype != cpu.dtype:
                wrong.append(f"{a_dtype}/{b_dtype}: {ours.dtype} != {cpu.dtype}")
            else:
                try:
                    _same_outcome(name, cpu, ours)
                except AssertionError as e:
                    wrong.append(f"{a_dtype}/{b_dtype}: {str(e)[:120]}")
    assert not wrong, "\n".join(wrong)


def _registered(key: str) -> bool:
    return torch._C._dispatch_has_kernel_for_dispatch_key(key, "PrivateUse1")


@pytest.mark.parametrize(
    "fn",
    [
        lambda t: torch.clamp_min(t, 2**53 + 1),
        lambda t: torch.clamp_max(t, 2**53 + 1),
        lambda t: torch.clamp(t, -(2**53) - 1, 2**53 + 1),
        lambda t: torch.clamp_min(
            t, 2**53 + 1, out=torch.empty(0, dtype=t.dtype, device=t.device)
        ),
    ],
    ids=["clamp_min", "clamp_max", "clamp", "clamp_min.out"],
)
def test_clamp_int64_bounds_are_exact(mojo_gpu, fn):
    """An int64 bound past 2**53 has no exact float64: the bound reaches
    the kernel as the integer itself."""
    if not _registered("aten::clamp_min"):
        pytest.skip("clamp_min is not registered on the mojo device")
    x = torch.tensor([2**53, 2**53 + 2, -(2**53) - 2, 5, 2**60], dtype=torch.int64)
    assert torch.equal(fn(x.to(mojo_gpu)).cpu(), fn(x))


def test_rsub_scalar_out_matches_functional(mojo_gpu):
    """rsub.Scalar_out rounds like rsub.Scalar (`other` kept in opmath, not
    embedded in self's float16 first)."""
    if not _registered("aten::rsub.Scalar_out"):
        pytest.skip("rsub.Scalar_out is not registered on the mojo device")
    x = torch.tensor([0.1, 0.5, 1.0, 1.5, 3.3, -2.0], dtype=torch.float16).to(mojo_gpu)
    out = torch.empty(0, dtype=torch.float16, device=mojo_gpu)
    with _ran("rsub.Scalar_out"):
        torch.ops.aten.rsub.Scalar_out(x, 1.0001, out=out)
    assert torch.equal(out.cpu(), torch.ops.aten.rsub.Scalar(x, 1.0001).cpu())


@pytest.mark.parametrize(
    "dtype", [torch.uint16, torch.uint32, torch.uint64, torch.int8, torch.bool]
)
@pytest.mark.parametrize("op", [torch.isinf, torch.isfinite])
def test_inf_predicates_take_every_dtype(mojo_gpu, op, dtype):
    """isinf / isfinite of an integer (unsigned wide ones too) or bool
    tensor: never infinite, always finite, as the ATen composites give."""
    if not _registered("aten::isinf"):
        pytest.skip("isinf is not registered on the mojo device")
    x = torch.tensor([0, 1, 1], dtype=dtype)
    assert torch.equal(op(x.to(mojo_gpu)).cpu(), op(x))


@pytest.mark.parametrize("weight_dtype", [torch.float64, torch.float16, torch.bfloat16])
def test_lerp_promotes_a_0dim_weight(mojo_gpu, weight_dtype):
    """TORCH_META_FUNC(lerp_Tensor): a 0-dim weight of another float dtype
    is promoted with the operands (the dimensioned ones win: float32), and
    the result then casts into `out`; a dimensioned one must match."""
    if not _registered("aten::lerp.Tensor_out"):
        pytest.skip("lerp.Tensor is not registered on the mojo device")
    if weight_dtype == torch.float64 and _metal(mojo_gpu):
        pytest.skip("Apple GPUs have no float64")
    a, b = torch.rand(2, 3), torch.rand(2, 3)
    w = torch.tensor(0.3, dtype=weight_dtype)
    got = torch.lerp(a.to(mojo_gpu), b.to(mojo_gpu), w.to(mojo_gpu))
    _same_outcome("lerp", torch.lerp(a, b, w), got)
    out = torch.empty(0, dtype=torch.float16, device=mojo_gpu)
    with _ran("lerp.Tensor_out"):
        torch.lerp(a.to(mojo_gpu), b.to(mojo_gpu), w.to(mojo_gpu), out=out)
    _same_outcome("lerp", torch.lerp(a, b, w).half(), out)
    with pytest.raises(RuntimeError, match="for `weight`"):
        torch.lerp(
            a.to(mojo_gpu),
            b.to(mojo_gpu),
            torch.rand(2, 3, dtype=weight_dtype).to(mojo_gpu),
        )


@pytest.mark.parametrize("dtype", [torch.int32, torch.int64])
@pytest.mark.parametrize("op", ["threshold", "hardtanh"])
def test_integer_scalar_parameters_are_exact(mojo_gpu, op, dtype):
    """threshold / hardtanh on an integer tensor apply their Scalar
    parameters in scalar_t: 16777217 must not round through a float32."""
    key = f"aten::{op}"
    if not _registered(key):
        pytest.skip(f"{key} is not registered on the mojo device")
    x = torch.tensor([16777216, 16777217, 16777218, -5], dtype=dtype)
    if op == "threshold":
        fn = lambda t: torch.ops.aten.threshold(t, 16777217, 7)  # noqa: E731
    else:
        fn = lambda t: torch.ops.aten.hardtanh(t, 16777217, 16777217)  # noqa: E731
    assert torch.equal(fn(x.to(mojo_gpu)).cpu(), fn(x))


_B = 2**53
# Integer Scalars past 2**53 (no exact float64): each must reach the kernel
# as the int64 itself. (key: the registration the case needs.)
_EXACT_INT_CASES = {
    "threshold": (
        "aten::threshold",
        lambda t: torch.nn.functional.threshold(t, _B + 1, -7),
    ),
    "threshold_value": (
        "aten::threshold",
        lambda t: torch.nn.functional.threshold(t, 3, _B + 1),
    ),
    "hardtanh": (
        "aten::hardtanh",
        lambda t: torch.nn.functional.hardtanh(t, _B + 1, _B + 3),
    ),
    "clamp": ("aten::clamp", lambda t: torch.clamp(t, _B + 1, _B + 3)),
    "fmod": ("aten::fmod.Scalar", lambda t: torch.fmod(t, _B + 1)),
    "remainder": ("aten::remainder.Scalar", lambda t: torch.remainder(t, _B + 1)),
    "rsub": ("aten::rsub.Scalar", lambda t: torch.rsub(t, _B + 1)),
    "rsub_alpha": ("aten::rsub.Scalar", lambda t: torch.rsub(t, 1, alpha=_B + 1)),
    "sub_alpha": ("aten::sub.Tensor", lambda t: torch.sub(t, 1, alpha=_B + 1)),
    "add_alpha": ("aten::add.Tensor", lambda t: torch.add(t, t, alpha=_B + 1)),
    "maximum_0d": ("aten::maximum", lambda t: torch.maximum(t, torch.tensor(_B + 1))),
    "bitwise_and": ("aten::bitwise_and.Scalar", lambda t: torch.bitwise_and(t, _B + 1)),
    "floor_divide": (
        "aten::floor_divide.Scalar",
        lambda t: torch.floor_divide(t, _B + 1),
    ),
}


@pytest.mark.parametrize("case", list(_EXACT_INT_CASES))
def test_int64_scalars_past_2_53_are_exact(mojo_gpu, case):
    key, fn = _EXACT_INT_CASES[case]
    if not _registered(key):
        pytest.skip(f"{key} is not registered on the mojo device")
    x = torch.tensor([_B, _B + 1, _B + 2, 5], dtype=torch.int64)
    assert torch.equal(fn(x.to(mojo_gpu)).cpu(), fn(x))


_OVERFLOW_CASES = {
    "clamp_int32": ("aten::clamp", torch.int32, lambda t: t.clamp(0, 2**40)),
    "clamp_min_int8": ("aten::clamp_min", torch.int8, lambda t: t.clamp_min(300)),
    "threshold_int32": (
        "aten::threshold",
        torch.int32,
        lambda t: torch.nn.functional.threshold(t, 2**40, 1),
    ),
    "hardtanh_int32": (
        "aten::hardtanh",
        torch.int32,
        lambda t: torch.nn.functional.hardtanh(t, 0, 2**40),
    ),
    "sub_alpha_int8": (
        "aten::sub.Tensor",
        torch.int8,
        lambda t: torch.sub(t, 1, alpha=300),
    ),
}


@pytest.mark.parametrize("case", list(_OVERFLOW_CASES))
def test_integer_scalar_overflow_raises_like_torch(mojo_gpu, case):
    """`Scalar::to<scalar_t>()` refuses a value the dtype cannot hold."""
    key, dtype, fn = _OVERFLOW_CASES[case]
    if not _registered(key):
        pytest.skip(f"{key} is not registered on the mojo device")
    x = torch.tensor([1, 2, 3], dtype=dtype)
    with pytest.raises(RuntimeError, match="without overflow"):
        fn(x)
    with pytest.raises(RuntimeError, match="without overflow"):
        fn(x.to(mojo_gpu))


_FLOAT_TO_INT_CASES = [
    (torch.int8, -128.5),
    (torch.int8, 127.5),
    (torch.int8, -0.5),
    (torch.int8, -128.0),
    (torch.uint8, -2.0),
    (torch.uint8, -0.5),
    (torch.uint8, 127.5),
    (torch.uint8, 255.5),
    (torch.int32, 2147483647.5),
    (torch.int32, -2147483648.0),
    (torch.int8, float("nan")),
    (torch.int8, float("inf")),
]


@pytest.mark.parametrize(("dtype", "value"), _FLOAT_TO_INT_CASES)
def test_float_scalar_on_integer_dtype_converts_like_torch(mojo_gpu, dtype, value):
    """A float Scalar applied in an integer scalar_t (threshold's threshold
    and value) is refused outside the dtype's [lowest, max], NaN and inf
    included, and truncated inside it -- c10's checked_convert."""
    if not _registered("aten::threshold"):
        pytest.skip("threshold is not registered on the mojo device")
    x = torch.tensor([-3, 0, 1, 2, 100], dtype=dtype)
    for fn in (
        lambda t: torch.nn.functional.threshold(t, value, 1),
        lambda t: torch.nn.functional.threshold(t, 0, value),
    ):
        _same_outcome(
            "threshold", _outcome(lambda: fn(x)), _outcome(lambda: fn(x.to(mojo_gpu)))
        )


@pytest.mark.parametrize(
    ("dtype", "lo", "hi"),
    [
        (torch.uint8, -0.5, 3),
        (torch.uint8, 0, 3.9),
        (torch.uint8, -1, 3),
        (torch.uint8, -1.0, 3),
        (torch.uint8, 0, 300),
        (torch.int8, -0.5, 3.9),
        (torch.int8, -128.5, 3),
        (torch.int32, -(2**40), 2**40),
    ],
)
def test_hardtanh_integer_bounds_convert_like_torch(mojo_gpu, dtype, lo, hi):
    """hardtanh_out on an integer self takes each bound as `toLong()` (a
    float truncates: -0.5 is 0, which uint8 accepts), refuses a negative one
    on uint8, and clamps with the integers, checked against the dtype."""
    if not _registered("aten::hardtanh"):
        pytest.skip("hardtanh is not registered on the mojo device")
    x = torch.tensor([0, 1, 2, 5, 100], dtype=dtype)

    def fn(t: torch.Tensor) -> torch.Tensor:
        return torch.nn.functional.hardtanh(t, lo, hi)

    _same_outcome(
        "hardtanh", _outcome(lambda: fn(x)), _outcome(lambda: fn(x.to(mojo_gpu)))
    )


@pytest.mark.parametrize("grad_dtype", [torch.int64, torch.int32, torch.float16])
@pytest.mark.parametrize(("lo", "hi"), [(-1, 1), (-0.4, 0.4), (-1.0, 0.25)])
def test_hardtanh_backward_bounds_in_the_promoted_dtype(mojo_gpu, grad_dtype, lo, hi):
    """hardtanh_backward is a binary op over (grad_output, self): with an
    integer grad_output and a float self the bounds apply in the promoted
    float dtype, not in grad_output's (an int64 unit grad over [-0.5, 0,
    0.5] with bounds (-1, 1) is [1, 1, 1])."""
    if not _registered("aten::hardtanh_backward"):
        pytest.skip("hardtanh_backward is not registered on the mojo device")
    grad = torch.ones(5, dtype=grad_dtype)
    x = torch.tensor([-0.5, 0.0, 0.5, -1.0, 0.3])
    fn = torch.ops.aten.hardtanh_backward
    cpu = _outcome(lambda: fn(grad, x, lo, hi))
    ours = _outcome(lambda: fn(grad.to(mojo_gpu), x.to(mojo_gpu), lo, hi))
    _same_outcome("hardtanh_backward", cpu, ours)
    if isinstance(ours, torch.Tensor) and isinstance(cpu, torch.Tensor):
        assert ours.dtype == cpu.dtype


@pytest.mark.parametrize("op", ["add_", "mul_", "sub_", "__irshift__", "__ilshift__"])
def test_inplace_disjoint_strided_views_are_accepted(mojo_gpu, op):
    """`at::assert_no_partial_overlap` judges only non-overlapping-and-dense
    views: interleaved `x[::2]` / `x[1::2]` -- two views of ONE storage that
    share no element -- are accepted (torch calls them TooHard), while a
    genuinely partial overlap of one storage (`y[1:]` against `y[:-1]`)
    raises."""
    key = f"aten::{op}.Tensor"
    if not _registered(key):
        pytest.skip(f"{key} is not registered on the mojo device")
    # Even slots large, odd slots small shift counts / factors.
    cpu = torch.stack(
        [torch.arange(1, 9, dtype=torch.int64) * 64, torch.arange(8) % 5], dim=1
    ).flatten()
    x = cpu.to(mojo_gpu)
    a, b = x[::2], x[1::2]
    assert a.untyped_storage().data_ptr() == b.untyped_storage().data_ptr()
    native.op_counting(True)
    before = native.op_count(key)
    getattr(a, op)(b)
    assert native.op_count(key) > before, f"{key} did not run natively"
    getattr(cpu[::2], op)(cpu[1::2])
    assert torch.equal(x.cpu(), cpu)
    y = torch.arange(8, dtype=torch.int64, device=mojo_gpu)
    with pytest.raises(RuntimeError, match="single memory location"):
        getattr(y[1:], op)(y[:-1])
    assert torch.equal(y.cpu(), torch.arange(8))


@pytest.mark.parametrize("name", _CASES)
def test_independent_empty_out_is_not_an_input(mojo_gpu, name):
    """Out/input identity is tensor identity (`TensorBase::is_same`), not a
    data-pointer match: an independent empty `out` shares a null data
    pointer with an empty input yet is a different tensor, so it is resized
    to the broadcast shape like any other `out` --
    `torch.add(torch.empty(0), torch.ones(2, 1), out=torch.empty(0))` is
    (2, 0)."""
    _registered_or_skip(name)
    tensors = _tensor_args(name)
    if len(tensors) < 2:
        pytest.skip("one tensor input: nothing broadcasts past it")
    shapes = {t: (0,) if i == 0 else (2, 1) for i, t in enumerate(tensors)}
    kwargs = _cpu_args(name, shapes=shapes)
    natural = _outcome(lambda: _cpu_result(name, kwargs))
    if not isinstance(natural, torch.Tensor):
        _torch_does_not_check(name)
    cpu = _outcome(
        lambda: _op(name)(
            **kwargs, **{_out_name(name): torch.empty(0, dtype=natural.dtype)}
        )
    )
    args = _to(kwargs, mojo_gpu)
    out = torch.empty(0, dtype=natural.dtype, device=mojo_gpu)
    with _ran(name):
        ours = _outcome(lambda: _op(name)(**args, **{_out_name(name): out}))
    _same_outcome(name, cpu, ours)
    if isinstance(ours, torch.Tensor) and isinstance(cpu, torch.Tensor):
        assert ours.shape == cpu.shape


def test_inplace_casts_a_promoted_result_into_self(mojo_gpu):
    """int8 `<<=` int64: computed in int64 and cast back into the int8 self,
    as TensorIterator does; an int self with a float operand raises."""
    if not _registered("aten::__ilshift__.Tensor"):
        pytest.skip("__ilshift__ is not registered on the mojo device")
    cpu = torch.tensor([1, 2, 3, -4], dtype=torch.int8)
    shift = torch.tensor([1, 2, 3, 1], dtype=torch.int64)
    x = cpu.to(mojo_gpu)
    x <<= shift.to(mojo_gpu)
    cpu <<= shift
    assert x.dtype == torch.int8 and torch.equal(x.cpu(), cpu)
