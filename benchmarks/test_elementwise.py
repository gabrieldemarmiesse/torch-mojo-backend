"""Elementwise unary benchmarks: one node per (op, dtype, shape).

Every op here is a single memory-bound kernel; two shapes cover the two
regimes that matter: C_16777216 is a large contiguous vector (bandwidth
bound) and A_357x789 is small and awkward (launch/tail bound).  Operands
live in (0.05, 0.95) so one generator serves every op's domain
(acos/atanh need |x| < 1, log/rsqrt need x > 0), except acosh, which is
real only from 1 on and reads the same interval shifted to (1.05, 1.95).

The op axis carries a bench_op mark per param, so the baseline tree path
of node test_unary[abs-C_16777216-bf16] is bf16/abs/C_16777216/contig.
The separate test_unary_unaligned adds offset views without renaming any
existing contiguous benchmark node or baseline key.
"""

from __future__ import annotations

from collections.abc import Callable

import pytest
import torch
import torch.nn.functional as F
from bench_lib.cases import DTYPES, both, op_params, unit_interval
from bench_lib.check import Bench
from bench_lib.hw import Hardware

SHAPES: dict[str, tuple[int, ...]] = {
    "C_16777216": (16777216,),
    "A_357x789": (357, 789),
}

UNARY_OPS: dict[str, Callable[[torch.Tensor], torch.Tensor]] = {
    "abs": torch.abs,
    "acos": torch.acos,
    "acosh": torch.acosh,
    "angle": torch.angle,
    "asin": torch.asin,
    "asinh": torch.asinh,
    "atan": torch.atan,
    "atanh": torch.atanh,
    "ceil": torch.ceil,
    "cos": torch.cos,
    "cosh": torch.cosh,
    "deg2rad": torch.deg2rad,
    "erf": torch.erf,
    "erfc": torch.erfc,
    "erfinv": torch.erfinv,
    "exp": torch.exp,
    "exp2": torch.exp2,
    "expm1": torch.expm1,
    "floor": torch.floor,
    "frac": torch.frac,
    "gelu": F.gelu,
    "isfinite": torch.isfinite,
    "isinf": torch.isinf,
    "isnan": torch.isnan,
    "isneginf": torch.isneginf,
    "isposinf": torch.isposinf,
    "log": torch.log,
    "log10": torch.log10,
    "log1p": torch.log1p,
    "log2": torch.log2,
    "nan_to_num": torch.nan_to_num,
    "neg": torch.neg,
    "rad2deg": torch.rad2deg,
    "reciprocal": torch.reciprocal,
    "relu": torch.relu,
    "round": torch.round,
    "round.decimals": lambda x: torch.round(x, decimals=2),
    "rsqrt": torch.rsqrt,
    "sgn": torch.sgn,
    "sigmoid": torch.sigmoid,
    "sign": torch.sign,
    "signbit": torch.signbit,
    "silu": F.silu,
    "sin": torch.sin,
    "sinc": torch.sinc,
    "sinh": torch.sinh,
    "special_entr": torch.special.entr,
    "special_i0e": torch.special.i0e,
    "special_i1": torch.special.i1,
    "special_i1e": torch.special.i1e,
    "sqrt": torch.sqrt,
    "tan": torch.tan,
    "tanh": torch.tanh,
    "trunc": torch.trunc,
}

# Ops whose domain excludes unit_interval: its operand, shifted by this much.
DOMAIN_SHIFT: dict[str, float] = {"acosh": 1.0}


def _operand(op_name: str, shape: tuple[int, ...], dtype: torch.dtype) -> torch.Tensor:
    return (unit_interval(shape, torch.float32) + DOMAIN_SHIFT.get(op_name, 0.0)).to(
        dtype
    )


def _check_matches_stock(
    fn: Callable[[torch.Tensor], torch.Tensor], x_ref: torch.Tensor, x_our: torch.Tensor
):
    """Our result against the stock device's, whose kernels the mojo ones
    follow (CPU torch computes round.decimals and i0e differently in
    float16 / bfloat16). A dtype the stock device lacks skips the case, as
    `bench.run` would."""
    try:
        want = fn(x_ref).cpu()
    except NotImplementedError as exc:
        pytest.skip(f"not supported on the stock device: {exc}")
    torch.testing.assert_close(fn(x_our).cpu(), want)


# Special functions stock torch implements for float32/float64 only
# (AT_DISPATCH_FLOATING_TYPES): benchmarked in float32 alone.
FLOAT32_ONLY_OPS: dict[str, Callable[[torch.Tensor], torch.Tensor]] = {
    "special_airy_ai": torch.special.airy_ai,
    "special_bessel_j0": torch.special.bessel_j0,
    "special_bessel_j1": torch.special.bessel_j1,
    "special_bessel_y0": torch.special.bessel_y0,
    "special_bessel_y1": torch.special.bessel_y1,
    "special_erfcx": torch.special.erfcx,
    "special_log_ndtr": torch.special.log_ndtr,
    "special_modified_bessel_i0": torch.special.modified_bessel_i0,
    "special_modified_bessel_i1": torch.special.modified_bessel_i1,
    "special_modified_bessel_k0": torch.special.modified_bessel_k0,
    "special_modified_bessel_k1": torch.special.modified_bessel_k1,
    "special_ndtri": torch.special.ndtri,
    "special_scaled_modified_bessel_k0": torch.special.scaled_modified_bessel_k0,
    "special_scaled_modified_bessel_k1": torch.special.scaled_modified_bessel_k1,
    "special_spherical_bessel_j0": torch.special.spherical_bessel_j0,
}


# Registered elementwise ops NOT benchmarked here, and why.  Reconciled
# against the live registration table by test_coverage.py.
_UNARY_OUT = (
    "out= / in-place plumbing over the kernel test_unary measures (computed "
    "straight into a fitting destination)"
)
SKIPPED: dict[str, str] = {
    "aten::round_.decimals": (
        "in-place form of round.decimals: the same kernel, measured by"
        " test_unary[round.decimals]"
    )
} | {
    f"aten::{name}": _UNARY_OUT
    for name in (
        "deg2rad.out",
        "deg2rad_",
        "isinf.out",
        "isneginf.out",
        "isposinf.out",
        "rad2deg.out",
        "rad2deg_",
    )
}

COVERS: dict[str, str] = (
    {f"aten::{name}": "test_unary" for name in UNARY_OPS}
    | {f"aten::{name}": "test_unary_float32_only" for name in FLOAT32_ONLY_OPS}
    | {
        "aten::bitwise_not": "test_bitwise_not",
        "aten::logical_not": "test_logical_not",
        "aten::sqrt.out": "test_sqrt_contiguous_out",
    }
)


# The activations with scalar parameters of tmb/ops/pointwise.mojo, at
# their default parameters (the scalars travel in slots, so every value
# runs the same kernel). Operands are unit_interval shifted to (-0.45, 0.45)
# so both sides of every kink are timed.
ACTIVATION_OPS: dict[str, Callable[[torch.Tensor], torch.Tensor]] = {
    "elu": F.elu,
    "hardshrink": lambda x: F.hardshrink(x, 0.2),
    "hardsigmoid": F.hardsigmoid,
    "hardswish": F.hardswish,
    "hardtanh": lambda x: F.hardtanh(x, -0.25, 0.25),
    "leaky_relu": F.leaky_relu,
    "log_sigmoid_forward": F.logsigmoid,
    "mish": F.mish,
    "softplus": F.softplus,
    "softshrink": lambda x: F.softshrink(x, 0.2),
    "threshold": lambda x: F.threshold(x, 0.1, -1.0),
}
COVERS |= {f"aten::{name}": "test_activation" for name in ACTIVATION_OPS}
_ACT_OUT = (
    "out= / in-place plumbing over the activation kernel test_activation "
    "measures (computed straight into a fitting destination)"
)
SKIPPED |= {
    f"aten::{name}": _ACT_OUT
    for name in (
        "elu.out",
        "hardshrink.out",
        "hardsigmoid.out",
        "hardsigmoid_",
        "hardswish.out",
        "hardswish_",
        "hardtanh.out",
        "hardtanh_",
        "leaky_relu.out",
        "leaky_relu_",
        "log_sigmoid_forward.output",
        "mish.out",
        "rrelu_with_noise.out",
        "rrelu_with_noise_",
        "softplus.out",
        "softshrink.out",
        "threshold.out",
        "threshold_",
    )
} | {
    "aten::rrelu_with_noise": (
        "eval mode is the leaky_relu kernel test_activation measures; "
        "training is uniform_ (test_inplace's test_uniform_) plus two of the "
        "same pointwise launches"
    )
}


@pytest.mark.parametrize("dtype_id", ("bf16", "f32"))
@pytest.mark.parametrize("shape_id", SHAPES)
@pytest.mark.parametrize("op_name", op_params(ACTIVATION_OPS))
def test_activation(
    op_name: str,
    shape_id: str,
    dtype_id: str,
    bench: Bench,
    hw: Hardware,
    mojo_device: torch.device,
):
    fn = ACTIVATION_OPS[op_name]
    cpu = (unit_interval(SHAPES[shape_id], torch.float32) - 0.5).to(DTYPES[dtype_id])
    x_ref, x_our = both(cpu, hw, mojo_device)
    bench.run(lambda: fn(x_ref), lambda: fn(x_our), flops=float(cpu.numel()))


@pytest.mark.parametrize("dtype_id", ("f16", "bf16", "f32"))
@pytest.mark.parametrize("shape_id", SHAPES)
@pytest.mark.parametrize("op_name", op_params(UNARY_OPS))
def test_unary(
    op_name: str,
    shape_id: str,
    dtype_id: str,
    bench: Bench,
    hw: Hardware,
    mojo_device: torch.device,
):
    fn = UNARY_OPS[op_name]
    shape = SHAPES[shape_id]
    cpu = _operand(op_name, shape, DTYPES[dtype_id])
    x_ref, x_our = both(cpu, hw, mojo_device)
    for value in (x_ref, x_our):
        assert value.data_ptr() % (4 * cpu.element_size()) == 0
    _check_matches_stock(fn, x_ref, x_our)
    bench.run(lambda: fn(x_ref), lambda: fn(x_our), flops=float(x_ref.numel()))


@pytest.mark.parametrize("layout", ("offset_1",))
@pytest.mark.parametrize("dtype_id", ("f16", "bf16", "f32"))
@pytest.mark.parametrize("shape_id", SHAPES)
@pytest.mark.parametrize("op_name", op_params(UNARY_OPS))
def test_unary_unaligned(
    op_name: str,
    shape_id: str,
    dtype_id: str,
    layout: str,
    bench: Bench,
    hw: Hardware,
    mojo_device: torch.device,
):
    fn = UNARY_OPS[op_name]
    shape = SHAPES[shape_id]
    cpu = _operand(op_name, shape, DTYPES[dtype_id])
    # Slice after the device copy: copying a CPU view would realign it.
    storage = torch.cat((torch.zeros(1, dtype=cpu.dtype), cpu.flatten()))
    ref_storage, our_storage = both(storage, hw, mojo_device)
    x_ref, x_our = (value[1:].view(shape) for value in (ref_storage, our_storage))
    for value in (x_ref, x_our):
        assert value.is_contiguous()
        assert value.data_ptr() % (4 * cpu.element_size()) == cpu.element_size()
    _check_matches_stock(fn, x_ref, x_our)
    bench.run(lambda: fn(x_ref), lambda: fn(x_our), flops=float(cpu.numel()))


@pytest.mark.parametrize("dtype_id", ("f32",))
@pytest.mark.parametrize("shape_id", SHAPES)
@pytest.mark.parametrize("op_name", op_params(FLOAT32_ONLY_OPS))
def test_unary_float32_only(
    op_name: str,
    shape_id: str,
    dtype_id: str,
    bench: Bench,
    hw: Hardware,
    mojo_device: torch.device,
):
    fn = FLOAT32_ONLY_OPS[op_name]
    cpu = _operand(op_name, SHAPES[shape_id], DTYPES[dtype_id])
    x_ref, x_our = both(cpu, hw, mojo_device)
    _check_matches_stock(fn, x_ref, x_our)
    bench.run(lambda: fn(x_ref), lambda: fn(x_our), flops=float(x_ref.numel()))


@pytest.mark.bench_op("gelu")
@pytest.mark.parametrize("layout", ("contig", "offset_1"))
@pytest.mark.parametrize("dtype_id", ("f16", "bf16", "f32"))
@pytest.mark.parametrize("shape_id", ("C_16777216_tanh", "A_357x789_tanh"))
def test_gelu_tanh(
    shape_id: str,
    dtype_id: str,
    layout: str,
    bench: Bench,
    hw: Hardware,
    mojo_device: torch.device,
):
    # Approximation is a separate benchmark regime, encoded in the shape key
    # without renaming the existing default-GELU baseline entries.
    shape = SHAPES[shape_id.removesuffix("_tanh")]
    cpu = unit_interval(shape, DTYPES[dtype_id])
    if layout == "offset_1":
        storage = torch.cat((torch.zeros(1, dtype=cpu.dtype), cpu.flatten()))
        ref_storage, our_storage = both(storage, hw, mojo_device)
        x_ref, x_our = (value[1:].view(shape) for value in (ref_storage, our_storage))
    else:
        x_ref, x_our = both(cpu, hw, mojo_device)
    for value in (x_ref, x_our):
        expected_offset = cpu.element_size() if layout == "offset_1" else 0
        assert value.data_ptr() % (4 * cpu.element_size()) == expected_offset
    torch.testing.assert_close(
        F.gelu(x_our, approximate="tanh").cpu(), F.gelu(cpu, approximate="tanh")
    )
    bench.run(
        lambda: F.gelu(x_ref, approximate="tanh"),
        lambda: F.gelu(x_our, approximate="tanh"),
        flops=float(cpu.numel()),
    )


@pytest.mark.parametrize("dtype_id", ("i32",))
@pytest.mark.parametrize("shape_id", SHAPES)
def test_bitwise_not(
    shape_id: str, dtype_id: str, bench: Bench, hw: Hardware, mojo_device: torch.device
):
    shape = SHAPES[shape_id]
    x_ref, x_our = both(
        torch.randint(-1000, 1000, shape, dtype=DTYPES[dtype_id]), hw, mojo_device
    )
    bench.run(
        lambda: torch.bitwise_not(x_ref),
        lambda: torch.bitwise_not(x_our),
        flops=float(x_ref.numel()),
    )


@pytest.mark.parametrize("dtype_id", ("bool",))
@pytest.mark.parametrize("shape_id", SHAPES)
def test_logical_not(
    shape_id: str, dtype_id: str, bench: Bench, hw: Hardware, mojo_device: torch.device
):
    shape = SHAPES[shape_id]
    x_ref, x_our = both(torch.rand(shape) < 0.5, hw, mojo_device)
    bench.run(
        lambda: torch.logical_not(x_ref),
        lambda: torch.logical_not(x_our),
        flops=float(x_ref.numel()),
    )


_SQRT_OUT_SHAPES = {
    "small_800": (800, 0),
    "l2_3840000": (3840000, 0),
    "l2_5120000": (5120000, 0),
    "stream_40206400": (40206400, 0),
    "awkward_281673": (357 * 789, 0),
    "offset_281673": (357 * 789, 1),
}


@pytest.mark.bench_op("sqrt.out")
@pytest.mark.parametrize("dtype_id", ["f32"])
@pytest.mark.parametrize("shape_id", _SQRT_OUT_SHAPES)
def test_sqrt_contiguous_out(
    shape_id: str, dtype_id: str, bench: Bench, hw: Hardware, mojo_device: torch.device
):
    count, offset = _SQRT_OUT_SHAPES[shape_id]
    host = ((torch.arange(count + 16, dtype=torch.int64) * 7919 + 13) % 65521).to(
        DTYPES[dtype_id]
    ) / 4096
    ref_base, our_base = both(host, hw, mojo_device)
    ref_storage, our_storage = torch.empty_like(ref_base), torch.empty_like(our_base)
    ref, our = ref_base[offset : offset + count], our_base[offset : offset + count]
    ref_out = ref_storage[offset : offset + count]
    our_out = our_storage[offset : offset + count]
    bench.run(
        lambda: torch.sqrt(ref, out=ref_out),
        lambda: torch.sqrt(our, out=our_out),
        flops=float(count),
    )
