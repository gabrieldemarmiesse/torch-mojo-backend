"""Tests for the Mojo-extension fast path used by mojo eager mode."""

import functools
import math
import weakref
from pathlib import Path
from types import ModuleType, SimpleNamespace

import pytest
import torch
from max.driver import CPU
from torch.testing._internal.common_methods_invocations import op_db

from torch_mojo_backend import get_accelerators, register_mojo_devices

pytestmark = pytest.mark.xdist_group(name="group1")


@pytest.fixture(autouse=True)
def setup_max_device():
    register_mojo_devices()


BINARY_OPS = [torch.add, torch.sub, torch.mul, torch.div, torch.maximum, torch.minimum]
UNARY_OPS = [torch.relu, torch.exp]


def _spy_defined_native_calls(
    monkeypatch: pytest.MonkeyPatch, targets: set[tuple[str, str]]
) -> dict[tuple[str, str], list[tuple[tuple[object, ...], dict[str, object]]]]:
    """Observe descriptor calls at the stable one-function native ABI."""
    from torch_mojo_backend import eager_kernels

    calls = {target: [] for target in targets}
    original_load = eager_kernels.MOJO_EXTENSION_LOADER.load_canonical

    def load_canonical(
        mojo_file: Path, defines: eager_kernels.CanonicalDefines
    ) -> ModuleType:
        module = original_load(mojo_file, defines)
        key = (mojo_file.name, dict(defines).get("OP", ""))
        if key not in targets:
            return module
        native_call = module.call

        def call(*args: object, **kwargs: object) -> object:
            calls[key].append((args, kwargs))
            return native_call(*args, **kwargs)

        wrapped = ModuleType(f"{module.__name__}.spy")
        wrapped.call = call
        return wrapped

    monkeypatch.setattr(
        eager_kernels.MOJO_EXTENSION_LOADER, "load_canonical", load_canonical
    )
    return calls


def _replace_defined_native_calls(
    monkeypatch: pytest.MonkeyPatch, replacements: dict[tuple[str, str], object]
) -> None:
    """Replace selected constant `call` entry points without compiling them."""
    from torch_mojo_backend import eager_kernels

    original_load = eager_kernels.MOJO_EXTENSION_LOADER.load_canonical

    def load_canonical(
        mojo_file: Path, defines: eager_kernels.CanonicalDefines
    ) -> ModuleType:
        key = (mojo_file.name, dict(defines).get("OP", ""))
        replacement = replacements.get(key)
        if replacement is None:
            return original_load(mojo_file, defines)
        module = ModuleType(f"mock_{mojo_file.stem}_{key[1]}")
        module.call = replacement  # type: ignore[attr-defined]
        return module

    monkeypatch.setattr(
        eager_kernels.MOJO_EXTENSION_LOADER, "load_canonical", load_canonical
    )


@pytest.mark.parametrize("op", BINARY_OPS)
@pytest.mark.parametrize("dtype", [torch.float32, torch.float16])
def test_fast_binary_ops_match_cpu(mojo_device, op, dtype):
    x = torch.randn(33, 65).to(dtype)
    y = torch.randn(33, 65).to(dtype) + 1.5  # avoid div-by-~0
    result = op(x.to(mojo_device), y.to(mojo_device))
    torch.testing.assert_close(result.cpu(), op(x, y))


@pytest.mark.parametrize("op", UNARY_OPS)
@pytest.mark.parametrize("dtype", [torch.float32, torch.float16])
def test_fast_unary_ops_match_cpu(mojo_device, op, dtype):
    x = torch.randn(33, 65).to(dtype)
    result = op(x.to(mojo_device))
    torch.testing.assert_close(result.cpu(), op(x))


def test_fast_log1p_preserves_small_values(mojo_device):
    x = torch.tensor([1e-10, -1e-10, 1e-8, -1e-8, 1e-6, -1e-6])
    result = torch.log1p(x.to(mojo_device)).cpu()
    torch.testing.assert_close(result, torch.log1p(x), rtol=2e-6, atol=0)


@pytest.mark.parametrize("dtype", [torch.int32, torch.int64])
def test_fast_binary_int_dtypes(mojo_device, dtype):
    x = torch.arange(100, dtype=dtype)
    y = torch.arange(100, dtype=dtype) * 3
    result = (x.to(mojo_device) + y.to(mojo_device)).cpu()
    torch.testing.assert_close(result, x + y)


def test_fast_path_is_used(mojo_device, monkeypatch):
    """The eligible case must go through the Mojo kernel, not the fallback.

    Tensor-tensor adds route through the shared spec op, or through the
    Apple flat kernel selected during Metal device registration."""
    x = torch.randn(8, 8).to(mojo_device)
    y = torch.randn(8, 8).to(mojo_device)
    # Metal registration swaps fast_aten_add for the Apple flat kernel, so
    # watch both entry points and require exactly one of them to run.
    spec = ("logic_ops.mojo", "AddSpec")
    apple = ("elementwise_ops.mojo", "Add")
    native_calls = _spy_defined_native_calls(monkeypatch, {spec, apple})
    _ = x + y
    assert len(native_calls[spec]) + len(native_calls[apple]) == 1


@pytest.mark.parametrize(
    "module_name,spec_name,fn",
    [
        ("logic_ops", "SubSpec", lambda x, y: x - y),
        ("logic_ops", "EqSpec", lambda x, y: x == y),
        ("elementwise_ops", "SigmoidSpec", lambda x, y: torch.sigmoid(x)),
        ("elementwise_ops", "MulScalarSpec", lambda x, y: x * 2.0),
        ("reduction_ops", "SumSpec", lambda x, y: x.sum(-1)),
        ("nn_ops", "SoftmaxSpec", lambda x, y: torch.softmax(x, -1)),
        ("matmul_ops", "MatmulSpec", lambda x, y: x @ y),
    ],
)
def test_spec_path_is_used(mojo_device, module_name, spec_name, fn, monkeypatch):
    """One representative op per converted family must route through its
    spec entry (whole prologue in one Mojo call), not the classic chain."""
    target = (f"{module_name}.mojo", spec_name)
    native_calls = _spy_defined_native_calls(monkeypatch, {target})
    x = torch.randn(8, 8).to(mojo_device)
    y = torch.randn(8, 8).to(mojo_device)
    _ = fn(x, y)
    assert len(native_calls[target]) == 1


def test_fallback_broadcast(mojo_device):
    x = torch.randn(16, 16)
    y = torch.randn(16)
    result = (x.to(mojo_device) + y.to(mojo_device)).cpu()
    torch.testing.assert_close(result, x + y)


def test_fallback_scalar_other(mojo_device):
    x = torch.randn(16, 16)
    result = (x.to(mojo_device) + 2.5).cpu()
    torch.testing.assert_close(result, x + 2.5)


def test_fallback_alpha(mojo_device):
    x = torch.randn(16, 16)
    y = torch.randn(16, 16)
    result = torch.add(x.to(mojo_device), y.to(mojo_device), alpha=2.0).cpu()
    torch.testing.assert_close(result, torch.add(x, y, alpha=2.0))


def test_fallback_int_div(mojo_device):
    x = torch.arange(1, 65, dtype=torch.int32)
    y = torch.full((64,), 4, dtype=torch.int32)
    result = (x.to(mojo_device) / y.to(mojo_device)).cpu()
    # check_dtype=False: Mojo integer division currently promotes to float64
    # where torch gives float32.
    torch.testing.assert_close(result, x / y, check_dtype=False)


def test_fast_remainder_bfloat16_near_zero_divisor(mojo_device):
    """Regression test: a bf16 divisor near that format's smallest normal
    value (~1.18e-38 -- what OpInfo's exclude_zero substitutes for an
    exact-zero draw) used to send the naive trunc/multiply/subtract
    decomposition's quotient to +/-inf, even though the true remainder is
    always finite and bounded by |divisor|. -7.4375 and 91.0 both overflow
    a raw bf16 division by the tiny divisor (verified directly); the
    other two entries are ordinary values so this also covers the mixed
    case a broadcast triggers in practice."""
    x = torch.tensor([0.4922, -7.4375, 2.3125, 91.0], dtype=torch.bfloat16)
    y = torch.tensor([1.0, 1.0, 1.0, 1.1754943508222875e-38], dtype=torch.bfloat16)
    result = torch.remainder(x.to(mojo_device), y.to(mojo_device)).cpu()
    torch.testing.assert_close(result, torch.remainder(x, y))
    result_rmod = (y.to(mojo_device) % x.to(mojo_device)).cpu()
    torch.testing.assert_close(result_rmod, y % x)


def test_fast_remainder_float16_large_ratio(mojo_device):
    """Regression test: fp16's dynamic range alone (no divisor anywhere
    near a boundary) can push |a / b| past float32's 24-bit mantissa,
    e.g. 23456.0 / 0.0009937... ~= 2.36e7 > 2^24. Promoting to fp32
    without the bit-exact fmod used to lose the remainder's low bits
    entirely, rounding a true 0.0004178 down to exactly 0."""
    x = torch.tensor([23456.0], dtype=torch.float16)
    y = torch.tensor([0.0009937286376953125], dtype=torch.float16)
    result = torch.remainder(x.to(mojo_device), y.to(mojo_device)).cpu()
    torch.testing.assert_close(result, torch.remainder(x, y))


@pytest.mark.parametrize("dtype", [torch.bfloat16, torch.float16])
def test_fast_floor_divide_narrow_float_boundary(mojo_device, dtype):
    """Regression test for an off-by-one bug: computing floor(a / b) at
    bf16/fp16's own precision can round the quotient across an integer
    boundary before the floor is applied. E.g. -6.3125 / -1.0546875 has a
    true quotient of ~5.985, which used to round to 6.0 in bf16 before the
    floor, giving floor_divide = 6 instead of the correct 5."""
    x = torch.tensor([-6.3125, 91.0, 2.3125, -5.2812, 357.0], dtype=dtype)
    y = torch.tensor([-1.0546875, 3.375, 8.5, 1.0547, 6.789], dtype=dtype)
    result = torch.floor_divide(x.to(mojo_device), y.to(mojo_device)).cpu()
    torch.testing.assert_close(result, torch.floor_divide(x, y))


def test_fast_floor_divide_subnormal_quotient_underflow(mojo_device):
    """Regression test for a CPU-only bug distinct from the boundary-rounding
    one above: `a` sits exactly at float32's smallest NORMAL magnitude
    (2**-126, itself exactly representable in bf16 -- not subnormal), and
    dividing it by an O(1) `b` produces a true quotient that underflows into
    float32's subnormal range. The CPU `elementwise` codegen this kernel's
    CPU dispatch goes through was observed to flush that subnormal quotient
    to zero (a bare scalar `mojo run` of the same fp32 division does NOT
    flush -- this is specific to that compiled CPU path), giving floor(0) =
    0 instead of the correct -1. float64's subnormal threshold (~2**-1074)
    is nowhere near reachable from a bf16 operand, so the CPU dispatch
    widens to float64 instead of float32 for this op (`cpu_floordiv_f64` in
    logic_ops.mojo's `_bin_vec_op`); GPU device kernels are unaffected and
    stay on float32. bf16-only: fp16's own normal range bottoms out at
    2**-14, far above float32's subnormal cliff (~2**-149), so an fp16
    operand can never reach this failure mode through the same fp32
    upcast. Exact values from a fresh OpInfo conformance failure:
    `conformance/test_opinfo.py::test_matches_cpu_div_floor_rounding_mojo_bfloat16`,
    sample 5, index (3, 6, 2)."""
    dtype = torch.bfloat16
    x = torch.tensor([2.0**-126], dtype=dtype)
    y = torch.tensor([-4.71875], dtype=dtype)
    result = torch.floor_divide(x.to(mojo_device), y.to(mojo_device)).cpu()
    ref = torch.floor_divide(x, y)
    torch.testing.assert_close(result, ref)
    torch.testing.assert_close(ref, torch.tensor([-1.0], dtype=dtype))


@pytest.mark.parametrize("rounding_mode", ["floor", "trunc"])
@pytest.mark.parametrize(
    "dtype", [torch.float32, torch.float64, torch.int32, torch.int64, torch.uint8]
)
def test_fast_div_rounding_mode_signs(mojo_device, dtype, rounding_mode):
    """aten::div.Tensor_mode / div.out_mode: every sign combination of a
    non-exact division, for both float and integer dtypes. floor and trunc
    only diverge when the operands have opposite signs and the true
    quotient is inexact -- unsigned has no such case, included as the
    "always agree" control."""
    if dtype == torch.uint8:
        x = torch.tensor([7, 1, 200, 5], dtype=dtype)
        y = torch.tensor([2, 3, 7, 5], dtype=dtype)
    else:
        x = torch.tensor([7, -7, 7, -7, 8, -8, 6, -6], dtype=dtype)
        y = torch.tensor([2, 2, -2, -2, 3, 3, -3, -3], dtype=dtype)
    dev = torch.div(
        x.to(mojo_device), y.to(mojo_device), rounding_mode=rounding_mode
    ).cpu()
    ref = torch.div(x, y, rounding_mode=rounding_mode)
    torch.testing.assert_close(dev, ref)


def test_fast_div_rounding_mode_floor_trunc_disagree(mojo_device):
    """Pin the actual semantic difference between the two modes: a naive
    implementation that aliases trunc to floor (or vice versa) would still
    pass a same-sign-only test suite."""
    x = torch.tensor([-7, 7], dtype=torch.int64).to(mojo_device)
    y = torch.tensor([2, -2], dtype=torch.int64).to(mojo_device)
    floor_dev = torch.div(x, y, rounding_mode="floor").cpu()
    trunc_dev = torch.div(x, y, rounding_mode="trunc").cpu()
    assert not torch.equal(floor_dev, trunc_dev)
    torch.testing.assert_close(floor_dev, torch.tensor([-4, -4], dtype=torch.int64))
    torch.testing.assert_close(trunc_dev, torch.tensor([-3, -3], dtype=torch.int64))


@pytest.mark.parametrize("rounding_mode", ["floor", "trunc"])
def test_fast_div_rounding_mode_scalar(mojo_device, rounding_mode):
    """aten::div.Scalar_mode, routed through the same Tensor_mode kernel: a
    negative Python-scalar divisor against mixed-sign int operands."""
    x = torch.tensor([7, -7, 8, -8, 9, -9], dtype=torch.int32)
    dev = torch.div(x.to(mojo_device), -2, rounding_mode=rounding_mode).cpu()
    ref = torch.div(x, -2, rounding_mode=rounding_mode)
    torch.testing.assert_close(dev, ref)


def test_fast_div_rounding_mode_out(mojo_device):
    """aten::div.out_mode."""
    x = torch.tensor([7, -7, 8, -8], dtype=torch.int32).to(mojo_device)
    y = torch.tensor([2, 2, -3, -3], dtype=torch.int32).to(mojo_device)
    out = torch.empty_like(x)
    torch.div(x, y, rounding_mode="trunc", out=out)
    ref = torch.div(x.cpu(), y.cpu(), rounding_mode="trunc")
    torch.testing.assert_close(out.cpu(), ref)


@pytest.mark.parametrize("dtype", [torch.bfloat16, torch.float16])
def test_fast_div_trunc_mode_narrow_float_boundary(mojo_device, dtype):
    """Same operands as test_fast_floor_divide_narrow_float_boundary's
    boundary regression, but pinning the OPPOSITE fix for trunc mode:
    unlike floor_divide, torch's own tensor-tensor trunc kernel does not
    upcast bf16/fp16 to fp32 (see BOP_TRUNCDIV's comment in logic_ops.mojo),
    so a quotient of ~5.985 rounding to 6.0 in bf16 *before* truncation,
    giving 6 rather than the mathematically exact 5, is torch's actual
    output here and must be reproduced exactly, not "corrected"."""
    x = torch.tensor([-6.3125, 91.0, 2.3125, -5.2812, 357.0], dtype=dtype)
    y = torch.tensor([-1.0546875, 3.375, 8.5, 1.0547, 6.789], dtype=dtype)
    result = torch.div(
        x.to(mojo_device), y.to(mojo_device), rounding_mode="trunc"
    ).cpu()
    torch.testing.assert_close(result, torch.div(x, y, rounding_mode="trunc"))


@pytest.mark.parametrize("shape", [(0,), (1,), (7,), (0, 5)])
def test_edge_case_shapes(mojo_device, shape):
    x = torch.randn(*shape)
    y = torch.randn(*shape)
    result = (x.to(mojo_device) + y.to(mojo_device)).cpu()
    torch.testing.assert_close(result, x + y)


def test_chained_fast_ops(mojo_device):
    """Outputs of fast ops must be valid inputs to further fast ops."""
    x = torch.randn(32, 32)
    y = torch.randn(32, 32)
    device_result = x.to(mojo_device)
    for _ in range(5):
        device_result = torch.relu(
            device_result * y.to(mojo_device) + y.to(mojo_device)
        )
    expected = x
    for _ in range(5):
        expected = torch.relu(expected * y + y)
    torch.testing.assert_close(device_result.cpu(), expected)


@pytest.mark.parametrize("low_dtype", [torch.float16, torch.bfloat16])
def test_fast_binary_promotes_mixed_precision_residual_to_float32(mojo_gpu, low_dtype):
    """BF16/FP16 autocast outputs must add to FP32 residuals like CUDA."""
    generator = torch.Generator().manual_seed(20260718)
    residual = torch.randn(17, 65, generator=generator)
    branch = torch.randn(17, 65, generator=generator).to(low_dtype)
    actual = residual.to(mojo_gpu) + branch.to(mojo_gpu)
    expected = residual + branch
    assert actual.dtype == torch.float32
    torch.testing.assert_close(actual.cpu(), expected, atol=1e-6, rtol=1e-6)


@pytest.mark.parametrize("bf16_first", [False, True])
@pytest.mark.parametrize(
    "shape", [(), (0,), (0, 5), (1,), (7,), (17, 65), (3, 5, 7), (2, 3, 5, 7, 11)]
)
def test_fast_add_f32_bf16_fused_dynamic_shapes(
    mojo_gpu, shape, bf16_first, monkeypatch
):
    """Mixed residual adds convert BF16 values in registers, in either order."""
    generator = torch.Generator().manual_seed(20260720)
    fp32 = torch.randn(shape, generator=generator)
    bf16 = torch.randn(shape, generator=generator).to(torch.bfloat16)
    lhs, rhs = (bf16, fp32) if bf16_first else (fp32, bf16)

    fused_target = ("logic_ops.mojo", "AddF32Bf16Spec")
    cast_target = ("data_movement_ops.mojo", "CastSpec")
    calls = _spy_defined_native_calls(monkeypatch, {fused_target, cast_target})

    actual = lhs.to(mojo_gpu) + rhs.to(mojo_gpu)
    expected = lhs + rhs

    assert actual.dtype == torch.float32
    torch.testing.assert_close(actual.cpu(), expected, atol=0, rtol=0)
    assert len(calls[fused_target]) == 1
    assert not calls[cast_target]


def test_fast_add_f32_bf16_fused_spec_avoids_cast_temporary(mojo_gpu, monkeypatch):
    """A contiguous mixed add must be one fused spec launch, even with tails."""
    fp32_storage = torch.randn(1_106).to(mojo_gpu)
    bf16_storage = torch.randn(1_106).to(torch.bfloat16).to(mojo_gpu)
    fp32 = fp32_storage[1:]
    bf16 = bf16_storage[1:]

    fused_target = ("logic_ops.mojo", "AddF32Bf16Spec")
    cast_target = ("data_movement_ops.mojo", "CastSpec")
    calls = _spy_defined_native_calls(monkeypatch, {fused_target, cast_target})
    outputs = (fp32 + bf16, bf16 + fp32)

    expected = fp32_storage.cpu()[1:] + bf16_storage.cpu()[1:]
    for output in outputs:
        assert output.dtype == torch.float32
        torch.testing.assert_close(output.cpu(), expected, atol=0, rtol=0)
    assert len(calls[fused_target]) == 2
    assert not calls[cast_target]


def test_fast_add_f32_bf16_strided_preserves_general_fallback(mojo_gpu, monkeypatch):
    """Ineligible layouts remain correct through the existing general path."""
    fp32 = torch.randn(7, 11)
    bf16 = torch.randn(7, 11).to(torch.bfloat16)
    device_fp32 = fp32.to(mojo_gpu).t()
    device_bf16 = bf16.to(mojo_gpu).t()

    fused_target = ("logic_ops.mojo", "AddF32Bf16Spec")
    cast_target = ("data_movement_ops.mojo", "CastSpec")
    calls = _spy_defined_native_calls(monkeypatch, {fused_target, cast_target})
    actual = device_fp32 + device_bf16

    torch.testing.assert_close(actual.cpu(), fp32.t() + bf16.t(), atol=0, rtol=0)
    assert not calls[fused_target]
    assert len(calls[cast_target]) == 1


@pytest.fixture
def mojo_h100(mojo_gpu):
    """H100 Mojo device for architecture-gated tensor-core fast paths."""
    accelerator = list(get_accelerators())[0]
    if accelerator.api != "cuda" or accelerator.architecture_name != "sm_90a":
        pytest.skip("the pure-Mojo H100 tensor-core fast paths require an H100")
    return mojo_gpu


def test_wrapper_subclass_preserves_native_saved_output(mojo_device):
    """Native autograd saves a complete wrapper, not a holderless TensorImpl."""
    generator = torch.Generator().manual_seed(20260722)
    host_input = torch.randn(7, 11, generator=generator)
    host_gradient = torch.randn(7, 11, generator=generator)

    reference = host_input.clone().requires_grad_()
    torch.exp(reference).backward(host_gradient)

    actual = host_input.to(mojo_device).requires_grad_()
    output = torch.exp(actual)
    assert type(output.grad_fn).__name__ == "ExpBackward0"

    saved = output.grad_fn._saved_result
    assert isinstance(saved, type(output))
    assert saved is not output
    assert saved._holder is output._holder
    assert saved._ptr == output._ptr
    assert saved._shape == output._shape
    assert saved._mojo_strides == output._mojo_strides
    assert saved._device == output._device

    output.backward(host_gradient.to(mojo_device))
    assert actual.grad is not None
    torch.testing.assert_close(actual.grad.cpu(), reference.grad)


def test_wrapper_subclass_native_saved_output_tracks_mutation(mojo_device):
    output = torch.exp(torch.randn(7, 11).to(mojo_device).requires_grad_())

    with torch.no_grad():
        output.add_(torch.ones_like(output))

    with pytest.raises(RuntimeError, match="modified by an inplace operation"):
        output.backward(torch.ones_like(output))


def test_fast_view_family(mojo_device):
    x = torch.randn(2, 6, 768)
    xd = x.to(mojo_device)
    torch.testing.assert_close(xd.view(-1, 768).cpu(), x.view(-1, 768))
    torch.testing.assert_close(xd.reshape(12, 768).cpu(), x.reshape(12, 768))
    torch.testing.assert_close(xd.unsqueeze(0).cpu(), x.unsqueeze(0))


def test_fast_view_aliases_storage(mojo_device):
    """The fast view must alias, matching torch.Tensor.view semantics."""
    x = torch.zeros(4, 4).to(mojo_device)
    v = x.view(16)
    x += torch.ones(4, 4).to(mojo_device)
    torch.testing.assert_close(v.cpu(), torch.ones(16))


@pytest.mark.parametrize("dims", [(0, 1), (1, 2), (-1, -2)])
def test_fast_transpose(mojo_device, dims, monkeypatch):
    target = ("data_movement_ops.mojo", "PermuteCopy")
    calls = _spy_defined_native_calls(monkeypatch, {target})
    x = torch.randn(2, 3, 4)
    result = x.to(mojo_device).transpose(*dims).contiguous().cpu()
    torch.testing.assert_close(result, x.transpose(*dims).contiguous())
    assert len(calls[target]) == 1


def test_fast_t(mojo_device):
    x = torch.randn(50, 30)
    torch.testing.assert_close(
        x.to(mojo_device).t().contiguous().cpu(), x.t().contiguous()
    )


@pytest.mark.parametrize("split_size,dim", [(768, 2), (2, 0), ([1, 2, 3], 1)])
def test_fast_split(mojo_device, split_size, dim):
    x = torch.randn(4, 6, 2304)
    dev_parts = x.to(mojo_device).split(split_size, dim=dim)
    for dev_part, ref_part in zip(dev_parts, x.split(split_size, dim=dim)):
        torch.testing.assert_close(dev_part.cpu(), ref_part)


def test_fast_cat_skips_legacy_empty(mojo_device):
    empty = torch.empty(0)
    x = torch.randn(1, 12, 6, 64)
    result = torch.cat([empty.to(mojo_device), x.to(mojo_device)], dim=-2)
    torch.testing.assert_close(result.cpu(), torch.cat([empty, x], dim=-2))


# (label, per-input shapes, dim) for the batched N-input cat kernel. The
# input counts straddle the Mojo side's CAT_SEG_CAP (64 inputs per launch),
# and the odd lengths are the ones that cannot be copied 16 bytes at a time,
# exercising the element-wise instantiation of the same kernel.
_CAT_CASES = [
    ("single input", [(1000,)], 0),
    ("two aligned", [(4096,), (4096,)], 0),
    ("three", [(777,), (777,), (777,)], 0),
    ("full batch", [(5000,)] * 64, 0),
    ("past one batch", [(311,)] * 70, 0),
    ("past two batches", [(37,)] * 130, 0),
    ("wildly unequal", [(1,), (7,), (4096,), (3,), (100000,)], 0),
    ("odd lengths", [(12345,), (7,), (999,)], 0),
    ("zero along dim", [(0, 5), (3, 5)], 0),
    ("3-D middle dim", [(5, 2, 7), (5, 3, 7), (5, 4, 7)], 1),
    ("3-D trailing dim", [(5, 6, 2), (5, 6, 3), (5, 6, 4)], 2),
    ("3-D negative dim", [(5, 6, 2), (5, 6, 3)], -1),
    ("4-D middle dim", [(2, 3, 5, 64), (2, 3, 1, 64), (2, 3, 9, 64)], 2),
]


@pytest.mark.parametrize(
    "shapes,dim",
    [case[1:] for case in _CAT_CASES],
    ids=[case[0] for case in _CAT_CASES],
)
@pytest.mark.parametrize("dtype", [torch.float32, torch.bfloat16, torch.int64])
def test_fast_cat_batched(mojo_device, shapes, dim, dtype):
    host = [torch.randn(shape).to(dtype) for shape in shapes]
    device = [x.to(mojo_device) for x in host]
    torch.testing.assert_close(
        torch.cat(device, dim).cpu(), torch.cat(host, dim), rtol=0, atol=0
    )


def test_fast_cat_batches_every_input_count_into_one_bridge_call(mojo_gpu, monkeypatch):
    """One bridge call whatever the input count: the Mojo side cuts the
    inputs into launches itself, so 130 inputs never mean 130 calls."""
    target = ("data_movement_ops.mojo", "CatN")
    calls = _spy_defined_native_calls(monkeypatch, {target})
    for count in (2, 3, 64, 130):
        host = [torch.randn(1024) for _ in range(count)]
        device = [x.to(mojo_gpu) for x in host]
        torch.testing.assert_close(
            torch.cat(device).cpu(), torch.cat(host), rtol=0, atol=0
        )
    assert len(calls[target]) == 4


def test_fast_cat_strided_inputs_gather_without_materializing(mojo_gpu, monkeypatch):
    """A non-contiguous input keeps the per-input gather path (the KV-cache
    head transpose); the batched kernel never sees a strided source."""
    target = ("data_movement_ops.mojo", "CatN")
    calls = _spy_defined_native_calls(monkeypatch, {target})
    host = [torch.randn(64, 32), torch.randn(64, 32)]
    device = [x.to(mojo_gpu).t() for x in host]
    torch.testing.assert_close(
        torch.cat(device, 0).cpu(), torch.cat([x.t() for x in host], 0), rtol=0, atol=0
    )
    # Mixed contiguity takes the same per-input path.
    mixed_host = [torch.randn(32, 64), torch.randn(64, 32), torch.randn(32, 64)]
    mixed = [
        mixed_host[0].to(mojo_gpu),
        mixed_host[1].to(mojo_gpu).t(),
        mixed_host[2].to(mojo_gpu),
    ]
    torch.testing.assert_close(
        torch.cat(mixed, 0).cpu(),
        torch.cat([mixed_host[0], mixed_host[1].t(), mixed_host[2]], 0),
        rtol=0,
        atol=0,
    )
    assert calls[target] == []


def test_fast_cat_offset_views_and_legacy_empty(mojo_gpu):
    """Offset views (whose base address carries a misalignment the shapes do
    not show) and a legacy empty in the middle of the list."""
    host = [torch.randn(2048) for _ in range(4)]
    device = [x.to(mojo_gpu) for x in host]
    torch.testing.assert_close(
        torch.cat([x[3:1000] for x in device]).cpu(),
        torch.cat([x[3:1000] for x in host]),
        rtol=0,
        atol=0,
    )
    empty = torch.empty(0)
    mid = [torch.randn(4, 8), empty, torch.randn(3, 8)]
    torch.testing.assert_close(
        torch.cat([x.to(mojo_gpu) for x in mid], 0).cpu(),
        torch.cat(mid, 0),
        rtol=0,
        atol=0,
    )


@pytest.mark.parametrize("dim", [0, 1])
def test_fast_stack_uses_the_batched_cat(mojo_gpu, dim):
    host = [torch.randn(1024) for _ in range(33)]
    device = [x.to(mojo_gpu) for x in host]
    torch.testing.assert_close(
        torch.stack(device, dim).cpu(), torch.stack(host, dim), rtol=0, atol=0
    )


def test_fast_cat_narrow_copy_dst_cpu_alignment_regression(mojo_device):
    """Regression for a segfault (SIGSEGV) in the CPU fallback of
    `NarrowCopyDst` (the per-input path `fast_aten_cat` takes when the
    batched `CatN` kernel is unavailable, i.e. on a `mojo` device with no
    accelerator -- see `first._device.api != "cpu"` in `fast_aten_cat`).

    Root cause: a CPU-only "func4" fast path in `_narrow_copy_dst`
    (data_movement_ops.mojo) vectorized 4 elements per store whenever
    `copy_len`, `dst_stride` and `dst_offset` were all multiples of 4, using
    `elementwise[..., simd_width=1]`. That primitive does not guarantee
    calling back exactly `Coord(outer * copy_len // 4)` times with no
    overrun, and was observed writing one extra width-4 store past `out`'s
    allocation. Harmless while it lands in heap slack, it corrupts a live
    allocation once the heap is packed tightly enough. The fix removes the
    CPU fast path entirely and always stores one element at a time there;
    the accelerator fast path (a real GPU kernel launch, not this CPU
    `elementwise` call) is untouched.

    First reproduced via
    `conformance/test_opinfo.py::TestOpInfoConformancePRIVATEUSE1::test_matches_cpu_stack_mojo_int64`
    (exit 139), which iterates every `stack` OpInfo sample for int64 in one
    process. The overrun is real regardless of what runs around it -- it
    always writes past `out`'s allocation -- but whether that lands in
    unused heap space or corrupts something live depends on the process's
    whole allocation history, not just this test's own few small tensors: a
    hand-picked repro of only the "suspicious" (copy_len, dst_stride,
    offset) case, or even this same OpInfo replay run standalone outside
    `instantiate_device_type_tests`' class-wide setup, did not reliably
    crash, while running it through that real machinery (however lightly,
    e.g. via plain `unittest`, no pytest needed) did every time. So this is
    the exact OpInfo sample sequence that overran and corrupted a real
    allocation -- a faithful repro of the trigger conditions and a real
    correctness check (`assert_close`, not just "did not crash") -- even
    though a from-scratch process running only this function is not
    guaranteed to reproduce the crash exit code if the bug ever regresses.
    """
    op = next(
        candidate
        for candidate in op_db
        if candidate.formatted_name == "stack" and candidate.variant_test_name == ""
    )
    checked = 0
    for sample in op.sample_inputs("cpu", torch.int64):
        host = sample.input
        device = [tensor.to(mojo_device) for tensor in host]
        actual = torch.stack(device, *sample.args, **sample.kwargs).cpu()
        expected = torch.stack(host, *sample.args, **sample.kwargs)
        torch.testing.assert_close(actual, expected, rtol=0, atol=0)
        checked += 1
    assert checked > 0


def _repeat_fill(shape: tuple[int, ...], dtype: torch.dtype) -> torch.Tensor:
    """A deterministic input, never an RNG: consecutive elements differ, so a
    tiled copy that reads one element off is caught by VALUE.

    251 is prime and coprime with every row length used below, so the pattern
    never lines up with a row boundary; k/256 for k < 251 is exact in bfloat16
    as well as float32, which keeps every comparison a bit-exact one.
    """
    numel = 1
    for extent in shape:
        numel *= extent
    base = torch.arange(numel, dtype=torch.int64) % 251
    if dtype.is_floating_point:
        base = base.to(torch.float32) / 256.0
    return base.to(dtype).view(shape)


# (rows, cols, repeats) for the rank-2 tiled-copy path. `cols` of 789, 7, 1015
# and 31 divide by no vector width, which is the case a vectorized tiled copy
# gets silently wrong -- a wide load would straddle the `j % cols` wrap and
# read the next row -- and an ODD `cols` additionally lands successive output
# rows on different 16-byte phases, so "the first row is aligned" proves
# nothing. The rest cover a single input row, a single input COLUMN, rows
# shorter than a warp (which take the second, flat kernel), the r = (1, 1)
# identity, and repeats LONGER than the input rank, where torch left-pads the
# input shape with 1s and the extra leading factors multiply the output.
#
# The last four are the geometries where the launch, not the indexing, is the
# whole problem: almost no input rows with the copy count carrying the work,
# ONE output row whose width comes entirely from the trailing repeat factor,
# and `r1 == 1`, which flattens the input to a single row before launching.
# Each of those took a different branch of the dispatch, and a first version
# of these kernels was slower than the general path on all of them.
_REPEAT_CASES = [
    (357, 789, (2, 3)),
    (13, 7, (3, 5)),
    (64, 1024, (3, 2)),
    (1024, 64, (1, 16)),
    (100, 8, (1, 5)),
    (33, 33, (2, 2)),
    (1, 100, (5, 7)),
    (100, 1, (3, 7)),
    (17, 31, (1, 1)),
    (3, 4, (2, 3, 5)),
    (5, 6, (3, 1, 1)),
    (7, 128, (4, 1, 2)),
    (1, 3, (2000, 1)),
    (2, 3, (1000, 1)),
    (1, 1, (5000, 1)),
    (1, 64, (1, 300)),
]
_REPEAT_IDS = [
    f"{r}x{c}_r{'x'.join(str(k) for k in reps)}" for r, c, reps in _REPEAT_CASES
]


@pytest.mark.parametrize("rows,cols,reps", _REPEAT_CASES, ids=_REPEAT_IDS)
def test_fast_repeat_matches_torch(mojo_device, rows, cols, reps):
    x = _repeat_fill((rows, cols), torch.float32)
    torch.testing.assert_close(
        x.to(mojo_device).repeat(*reps).cpu(), x.repeat(*reps), rtol=0, atol=0
    )


@pytest.mark.parametrize(
    "dtype", [torch.float32, torch.bfloat16, torch.int64, torch.uint8]
)
@pytest.mark.parametrize("rows,cols,reps", [(357, 789, (2, 3)), (100, 8, (1, 5))])
def test_fast_repeat_every_element_size(mojo_gpu, dtype, rows, cols, reps):
    """The kernels dispatch on element SIZE, and the widest access is capped in
    BYTES, so a 1-byte dtype reaches a 16-wide vector the 4-byte one never
    instantiates. Run all four widths through both the segment and the flat
    kernel."""
    x = _repeat_fill((rows, cols), dtype)
    torch.testing.assert_close(
        x.to(mojo_gpu).repeat(*reps).cpu(), x.repeat(*reps), rtol=0, atol=0
    )


@pytest.mark.parametrize("offset", [1, 2, 3, 4])
@pytest.mark.parametrize("rows,cols,reps", [(357, 789, (2, 3)), (64, 1024, (3, 2))])
def test_fast_repeat_offset_views_are_not_assumed_aligned(
    mojo_gpu, offset, rows, cols, reps
):
    """A base that is not 16-byte aligned must drop to a narrower access rather
    than fault. An offset view satisfies every divisibility test the shapes can
    show -- only the runtime ADDRESS says otherwise -- and one float is 4 bytes,
    so offsets 1..3 cover every misaligned residue and 4 goes back to aligned.
    """
    base = _repeat_fill((rows * cols + 4,), torch.float32)
    x = base[offset : offset + rows * cols].view(rows, cols)
    device = base.to(mojo_gpu)[offset : offset + rows * cols].view(rows, cols)
    torch.testing.assert_close(
        device.repeat(*reps).cpu(), x.repeat(*reps), rtol=0, atol=0
    )


def test_fast_repeat_routes_only_rank2_tiles_to_the_fast_kernel(mojo_gpu, monkeypatch):
    """The tiled kernel takes a rank-<=2 input with any number of repeat
    factors (plus a higher-rank one whose leading extents are all 1); a
    genuinely higher-rank tile keeps the general rank-8 `_tile_copy`."""
    fast = ("data_movement_ops.mojo", "RepeatTiled")
    general = ("data_movement_ops.mojo", "TileCopy")
    calls = _spy_defined_native_calls(monkeypatch, {fast, general})

    for shape, reps in (
        ((64, 32), (2, 3)),  # rank 2
        ((64,), (5,)),  # rank 1, one repeat factor
        ((64,), (2, 5)),  # rank 1, left-padded to rank 2
        ((8, 16), (2, 3, 4)),  # rank 2, left-padded to rank 3
        ((1, 8, 16), (2, 3, 4)),  # rank 3, leading extent 1
    ):
        x = _repeat_fill(shape, torch.float32)
        torch.testing.assert_close(
            x.to(mojo_gpu).repeat(*reps).cpu(), x.repeat(*reps), rtol=0, atol=0
        )
    assert len(calls[fast]) == 5
    assert calls[general] == []

    for shape, reps in (
        ((4, 8, 16), (2, 3, 4)),  # a real rank-3 tile
        ((2, 3, 4, 5), (2, 2, 2, 2)),
    ):
        x = _repeat_fill(shape, torch.float32)
        torch.testing.assert_close(
            x.to(mojo_gpu).repeat(*reps).cpu(), x.repeat(*reps), rtol=0, atol=0
        )
    assert len(calls[fast]) == 5
    assert len(calls[general]) == 2


def test_fast_repeat_degenerate_extents(mojo_device):
    """A zero repeat factor or a zero input extent is an empty output, not a
    launch and not a crash. `repeat(x, [])` on a 0-d tensor is a legal 0-d
    copy with no last dim to tile along, and reaches the general kernel."""
    for shape, reps in (((4, 5), (0, 2)), ((4, 5), (2, 0)), ((0, 5), (2, 3))):
        x = _repeat_fill(shape, torch.float32)
        torch.testing.assert_close(
            x.to(mojo_device).repeat(*reps).cpu(), x.repeat(*reps), rtol=0, atol=0
        )
    scalar = torch.tensor(1.25)
    torch.testing.assert_close(
        torch.ops.aten.repeat(scalar.to(mojo_device), []).cpu(),
        torch.ops.aten.repeat(scalar, []),
        rtol=0,
        atol=0,
    )


@pytest.mark.parametrize("dtype", [torch.float32, torch.bfloat16])
def test_fast_batch_norm_inference_float32_running_stats(mojo_gpu, dtype):
    """A half/bfloat16 activation with FLOAT32 running statistics.

    That is exactly what `nn.BatchNorm2d` holds under AMP -- torch keeps the
    statistics in the parameter dtype while the activation is reduced
    precision -- and the pairing used to be refused outright because the
    bridge demanded that every operand share the input's dtype.
    """
    torch.manual_seed(0)
    x = torch.randn(3, 8, 5, 7, dtype=dtype)
    weight = torch.randn(8, dtype=dtype)
    bias = torch.randn(8, dtype=dtype)
    running_mean = torch.randn(8)
    running_var = torch.rand(8) + 0.5
    args = (weight, bias, running_mean, running_var)
    expected = torch.ops.aten._native_batch_norm_legit_no_training(
        x.float(), *(t.float() for t in args), 0.1, 1e-5
    )
    actual = torch.ops.aten._native_batch_norm_legit_no_training(
        x.to(mojo_gpu), *(t.to(mojo_gpu) for t in args), 0.1, 1e-5
    )
    tolerance = 1e-5 if dtype == torch.float32 else 8e-3
    torch.testing.assert_close(
        actual[0].cpu().float(), expected[0], atol=tolerance, rtol=tolerance
    )


@pytest.mark.parametrize("dtype", [torch.float32, torch.bfloat16])
@pytest.mark.parametrize("shape", [(3, 8, 5, 7), (2, 5, 13), (4, 3, 8, 8, 2)])
def test_fast_batch_norm_training_matches_torch(mojo_gpu, dtype, shape):
    """Every output and both in-place running statistics, against CPU torch.

    The three rules that are easy to get backwards and are all checked here:
    `save_invstd` inverts the BIASED variance, the running variance takes the
    UNBIASED one, and eps sits inside the square root.
    """
    torch.manual_seed(0)
    channels = shape[1]
    x = torch.randn(shape, dtype=dtype)
    weight = torch.randn(channels, dtype=dtype)
    bias = torch.randn(channels, dtype=dtype)
    running_mean = torch.randn(channels)
    running_var = torch.rand(channels) + 0.5

    ref_mean, ref_var = running_mean.clone(), running_var.clone()
    expected = torch.ops.aten.native_batch_norm(
        x.float(), weight.float(), bias.float(), ref_mean, ref_var, True, 0.13, 1e-5
    )
    our_mean = running_mean.to(mojo_gpu)
    our_var = running_var.to(mojo_gpu)
    actual = torch.ops.aten.native_batch_norm(
        x.to(mojo_gpu),
        weight.to(mojo_gpu),
        bias.to(mojo_gpu),
        our_mean,
        our_var,
        True,
        0.13,
        1e-5,
    )
    tolerance = 1e-5 if dtype == torch.float32 else 8e-3
    torch.testing.assert_close(
        actual[0].cpu().float(), expected[0], atol=tolerance, rtol=tolerance
    )
    for got, want in zip(actual[1:], expected[1:], strict=True):
        assert got.dtype == torch.float32  # ATen's acc_type, whatever x is
        torch.testing.assert_close(got.cpu(), want, atol=1e-4, rtol=1e-4)
    torch.testing.assert_close(our_mean.cpu(), ref_mean, atol=1e-5, rtol=1e-5)
    torch.testing.assert_close(our_var.cpu(), ref_var, atol=1e-5, rtol=1e-5)


def test_fast_batch_norm_training_running_stats_update_once(mojo_gpu):
    """The momentum update must be applied exactly once per call.

    The statistics kernel enqueues its cancellation re-pass unconditionally
    (the decision is made on the device), so the merge runs twice per call --
    and a running-stat update living in the wrong one of those two passes
    would compound `(1-m)*r + m*v` twice and stay plausible-looking.
    """
    torch.manual_seed(0)
    x = torch.randn(4, 6, 5, 5)
    running_mean = torch.zeros(6)
    running_var = torch.ones(6)
    ref_mean, ref_var = running_mean.clone(), running_var.clone()
    our_mean, our_var = running_mean.to(mojo_gpu), running_var.to(mojo_gpu)
    for _ in range(3):
        torch.ops.aten.native_batch_norm(
            x.float(), None, None, ref_mean, ref_var, True, 0.1, 1e-5
        )
        torch.ops.aten.native_batch_norm(
            x.to(mojo_gpu), None, None, our_mean, our_var, True, 0.1, 1e-5
        )
    torch.testing.assert_close(our_mean.cpu(), ref_mean, atol=1e-6, rtol=1e-6)
    torch.testing.assert_close(our_var.cpu(), ref_var, atol=1e-6, rtol=1e-6)


@pytest.mark.parametrize("magnitude", [1e6, -1e6])
@pytest.mark.parametrize("channels", [3, 200])
def test_fast_batch_norm_training_outlier_first_element(mojo_gpu, magnitude, channels):
    """Element (0, c, 0) is a wild outlier for every channel c.

    The statistics take their assumed mean from exactly that element, so this
    is the worst shift available and `M2 = q - s^2/n` cancels catastrophically
    without the adaptive re-pass.  Both channel counts are exercised because
    they take different routes: 3 channels cannot fill the device and go
    through the split workspace plus its merge, 200 take the fused
    one-block-per-channel path that re-scans in place.
    """
    torch.manual_seed(0)
    x = torch.randn(6, channels, 4, 9)
    x[0, :, 0, 0] = magnitude
    running_mean = torch.zeros(channels)
    running_var = torch.ones(channels)
    ref_mean, ref_var = running_mean.clone(), running_var.clone()
    expected = torch.ops.aten.native_batch_norm(
        x, None, None, ref_mean, ref_var, True, 0.1, 1e-5
    )
    our_mean, our_var = running_mean.to(mojo_gpu), running_var.to(mojo_gpu)
    actual = torch.ops.aten.native_batch_norm(
        x.to(mojo_gpu), None, None, our_mean, our_var, True, 0.1, 1e-5
    )
    for got, want in zip(actual[1:], expected[1:], strict=True):
        torch.testing.assert_close(got.cpu(), want, atol=0, rtol=1e-5)
    torch.testing.assert_close(our_var.cpu(), ref_var, atol=0, rtol=1e-5)


def test_fast_batch_norm_training_refuses_autograd_in_the_forward(mojo_gpu):
    """A training call that would need a gradient must fail at the FORWARD.

    `aten::native_batch_norm_backward` is not implemented here, and a raise
    from inside the autograd engine aborts the process on this backend instead
    of raising, so the refusal cannot wait for the backward node.
    """
    x = torch.randn(2, 4, 3, 3, device=mojo_gpu, requires_grad=True)
    running_mean = torch.zeros(4, device=mojo_gpu)
    running_var = torch.ones(4, device=mojo_gpu)
    with pytest.raises(NotImplementedError, match="native_batch_norm_backward"):
        torch.ops.aten.native_batch_norm(
            x, None, None, running_mean, running_var, True, 0.1, 1e-5
        )
    with torch.no_grad():
        torch.ops.aten.native_batch_norm(
            x, None, None, running_mean, running_var, True, 0.1, 1e-5
        )


# Every op here has a fast forward and no runnable native backward. Reaching
# the backward node aborts the process (exit 134, "terminate called without an
# active exception", no traceback), so each is refused from the forward
# instead; the table pairs the call with the backward op its message must name.
# One case per op, so a preflight that regresses names itself in the failing
# node id -- and can be re-run alone, which matters here: a broken preflight
# does not fail this test, it kills the pytest process.
_UNARY_AUTOGRAD_PREFLIGHT_CASES = [
    ("relu", (4, 8), torch.relu, "aten::threshold_backward"),
    ("tanh", (4, 8), torch.tanh, "aten::tanh_backward"),
    ("sigmoid", (4, 8), torch.sigmoid, "aten::sigmoid_backward"),
    (
        "softmax",
        (4, 8),
        functools.partial(torch.softmax, dim=-1),
        "aten::_softmax_backward_data",
    ),
    (
        "_softmax",
        (4, 8),
        functools.partial(torch.ops.aten._softmax, dim=-1, half_to_float=False),
        "aten::_softmax_backward_data",
    ),
    ("cumsum", (4, 5), functools.partial(torch.cumsum, dim=-1), "aten::flip"),
    (
        "avg_pool2d",
        (2, 3, 8, 8),
        functools.partial(torch.nn.functional.avg_pool2d, kernel_size=2),
        "aten::avg_pool2d_backward",
    ),
    (
        "max_pool2d",
        (2, 3, 8, 8),
        functools.partial(torch.nn.functional.max_pool2d, kernel_size=2),
        "aten::max_pool2d_with_indices_backward",
    ),
    (
        "adaptive_avg_pool2d",
        (2, 3, 8, 8),
        functools.partial(torch.nn.functional.adaptive_avg_pool2d, output_size=(2, 2)),
        "aten::_adaptive_avg_pool2d_backward",
    ),
    (
        "upsample_bilinear2d",
        (2, 3, 4, 4),
        functools.partial(
            torch.nn.functional.interpolate, scale_factor=2, mode="bilinear"
        ),
        "aten::upsample_bilinear2d_backward",
    ),
]


def _preflight_input(
    shape: tuple[int, ...], device: str, seed: int = 20260819
) -> tuple[torch.Tensor, torch.Tensor]:
    """The same values on the host and on the device, for an A/B comparison."""
    host = torch.randn(*shape, generator=torch.Generator().manual_seed(seed))
    return host, host.to(device)


@pytest.mark.parametrize(
    ("shape", "call", "backward_op"),
    [case[1:] for case in _UNARY_AUTOGRAD_PREFLIGHT_CASES],
    ids=[case[0] for case in _UNARY_AUTOGRAD_PREFLIGHT_CASES],
)
def test_autograd_preflight_refuses_unrunnable_backward_in_the_forward(
    mojo_gpu, shape, call, backward_op
):
    """A grad-requiring call must raise here, not abort in the engine later."""
    _, device_input = _preflight_input(shape, mojo_gpu)
    device_input.requires_grad_()
    with pytest.raises(NotImplementedError, match=backward_op):
        call(device_input)


@pytest.mark.parametrize(
    ("shape", "call", "backward_op"),
    [case[1:] for case in _UNARY_AUTOGRAD_PREFLIGHT_CASES],
    ids=[case[0] for case in _UNARY_AUTOGRAD_PREFLIGHT_CASES],
)
def test_autograd_preflight_leaves_the_gradless_forward_working(
    mojo_gpu, shape, call, backward_op
):
    """The preflight must not cost the forward: no grad wanted, no refusal.

    Both halves matter -- a grad-free call, and a grad-requiring call under
    `torch.no_grad()`, which is the workaround the refusal message advertises.
    """
    host_input, device_input = _preflight_input(shape, mojo_gpu)
    expected = call(host_input)

    torch.testing.assert_close(call(device_input).cpu(), expected, atol=1e-5, rtol=1e-5)
    with torch.no_grad():
        grad_wanted = device_input.detach().requires_grad_()
        torch.testing.assert_close(
            call(grad_wanted).cpu(), expected, atol=1e-5, rtol=1e-5
        )


def test_autograd_preflight_refuses_relu_inplace_in_the_forward(mojo_gpu):
    """`nn.ReLU(inplace=True)` records the same node the functional form does.

    The operand has to be a non-leaf: an in-place write to a leaf that requires
    grad is rejected by PyTorch before this backend is consulted.
    """
    _, device_input = _preflight_input((4, 8), mojo_gpu)
    device_input.requires_grad_()
    with torch.no_grad():
        torch.relu_(device_input.detach().clone())
    non_leaf = device_input * 1.0
    with pytest.raises(NotImplementedError, match="aten::threshold_backward"):
        torch.relu_(non_leaf)


def test_autograd_preflight_refuses_advanced_indexing_in_the_forward(mojo_gpu):
    """`index.Tensor`'s backward scatters through `_index_put_impl_`."""
    host_input, device_input = _preflight_input((4, 5), mojo_gpu)
    index = torch.tensor([0, 2], device=mojo_gpu)
    torch.testing.assert_close(device_input[index].cpu(), host_input[[0, 2]])
    device_input.requires_grad_()
    with pytest.raises(NotImplementedError, match="aten::_index_put_impl_"):
        device_input[index]


def test_autograd_preflight_refuses_scatter_src_only_for_the_src_gradient(mojo_gpu):
    """Only `src`'s gradient needs `gather`; a self-only call still runs.

    The `self` gradient is a scatter of zeros, which this backend can run, so
    preflighting on "any operand requires grad" would refuse a working call.
    """
    host_input, device_input = _preflight_input((4, 5), mojo_gpu)
    host_src, device_src = _preflight_input((4, 1), mojo_gpu, seed=7)
    index = torch.zeros(4, 1, dtype=torch.long, device=mojo_gpu)

    expected = host_input.scatter(1, torch.zeros(4, 1, dtype=torch.long), host_src)
    torch.testing.assert_close(
        device_input.scatter(1, index, device_src).cpu(), expected
    )
    # self-only: still allowed, and the node it records is runnable.
    device_input.requires_grad_()
    device_input.scatter(1, index, device_src).sum().backward()
    assert device_input.grad is not None

    with pytest.raises(NotImplementedError, match="aten::gather"):
        device_input.detach().scatter(1, index, device_src.requires_grad_())


def test_autograd_preflight_refuses_efficient_attention_in_the_forward(mojo_gpu):
    """The low-level efficient-attention op, reached only by a direct call.

    `F.scaled_dot_product_attention` goes through this backend's own Python
    autograd Function, whose backward raises normally; this op is what a
    caller reaching past that hits.
    """
    _, query = _preflight_input((1, 2, 4, 8), mojo_gpu)
    with torch.no_grad():
        torch.ops.aten._scaled_dot_product_efficient_attention(
            query, query, query, None, False
        )
    query.requires_grad_()
    with pytest.raises(
        NotImplementedError, match="_scaled_dot_product_efficient_attention_backward"
    ):
        torch.ops.aten._scaled_dot_product_efficient_attention(
            query, query, query, None, False
        )


@pytest.mark.parametrize("dtype", [torch.float32, torch.bfloat16])
@pytest.mark.parametrize("cols", [7, 789, 1024, 4096, 20000])
def test_fast_layer_norm_row_widths(mojo_gpu, dtype, cols):
    """Row widths across every launch route of the forward.

    789 is not a whole number of 16-byte vectors (so the row has a scalar head
    and tail), 1024 and 4096 sit in the register-cached ladder, and 20000
    exceeds it and falls to the shifted-moment streaming kernel.
    """
    torch.manual_seed(0)
    x = torch.randn(9, cols, dtype=dtype)
    weight = torch.randn(cols, dtype=dtype)
    bias = torch.randn(cols, dtype=dtype)
    expected = torch.ops.aten.native_layer_norm(
        x.float(), [cols], weight.float(), bias.float(), 1e-5
    )
    actual = torch.ops.aten.native_layer_norm(
        x.to(mojo_gpu), [cols], weight.to(mojo_gpu), bias.to(mojo_gpu), 1e-5
    )
    tolerance = 1e-5 if dtype == torch.float32 else 1e-2
    torch.testing.assert_close(
        actual[0].cpu().float(), expected[0], atol=tolerance, rtol=tolerance
    )
    # mean/rstd come back in `dtype` (ATen's own convention: same dtype as
    # the input, see fast_aten_native_layer_norm), but `expected` was
    # computed from float32 inputs above to get a clean reference for the
    # output tolerance check -- so its saved stats are float32 regardless of
    # `dtype` and must be cast down to compare. That final cast is a single
    # bf16 rounding step on each side of two independently-computed float32
    # values (our device reduction vs. CPU's), so the float32-era tolerances
    # below (tight enough there) must widen to a bf16 ULP or a boundary value
    # can round to adjacent bf16 representables and fail on a correct result.
    mean_tolerance = 1e-4 if dtype == torch.float32 else 2e-2
    rstd_tolerance = 1e-3 if dtype == torch.float32 else 2e-2
    mean_actual, rstd_actual = actual[1].cpu(), actual[2].cpu()
    torch.testing.assert_close(
        mean_actual,
        expected[1].to(mean_actual.dtype),
        atol=mean_tolerance,
        rtol=mean_tolerance,
    )
    torch.testing.assert_close(
        rstd_actual,
        expected[2].to(rstd_actual.dtype),
        atol=rstd_tolerance,
        rtol=rstd_tolerance,
    )


@pytest.mark.parametrize("offset", [1, 2, 3])
def test_fast_layer_norm_offset_view_row_phase(mojo_gpu, offset):
    """A row base that is not a multiple of the 16-byte vector width.

    Slicing a flat buffer gives the input a 16-byte phase its freshly
    allocated output does not share, which is what decides whether the cached
    route may store where it loaded.
    """
    torch.manual_seed(0)
    flat = torch.randn(5 * 64 + offset).to(mojo_gpu)
    x = flat[offset:].view(5, 64)
    weight = torch.randn(64).to(mojo_gpu)
    bias = torch.randn(64).to(mojo_gpu)
    expected = torch.ops.aten.native_layer_norm(
        x.cpu(), [64], weight.cpu(), bias.cpu(), 1e-5
    )
    actual = torch.ops.aten.native_layer_norm(x, [64], weight, bias, 1e-5)
    for got, want in zip(actual, expected, strict=True):
        torch.testing.assert_close(got.cpu(), want, atol=1e-5, rtol=1e-5)


def test_fast_layer_norm_large_mean_needs_the_moment_repass(mojo_gpu):
    """A row whose FIRST element is a wild outlier, on the streaming route.

    Rows too long for the register-cached route take their statistics from
    the shared shifted-moment scan about the row's first element; when that
    element is the outlier the variance loses most of its significand unless
    the kernel notices and re-reads about the accurate mean.
    """
    torch.manual_seed(0)
    cols = 40000  # past the cached ladder
    x = torch.randn(4, cols)
    x[:, 0] = 1e6
    expected = torch.ops.aten.native_layer_norm(x, [cols], None, None, 1e-5)
    actual = torch.ops.aten.native_layer_norm(x.to(mojo_gpu), [cols], None, None, 1e-5)
    torch.testing.assert_close(actual[2].cpu(), expected[2], atol=0, rtol=1e-4)
    torch.testing.assert_close(actual[0].cpu(), expected[0], atol=1e-4, rtol=1e-4)


@pytest.mark.parametrize("dtype", [torch.float32, torch.bfloat16])
@pytest.mark.parametrize(
    ("n", "c", "hxw", "groups"), [(2, 8, 5, 4), (3, 12, 49, 3), (2, 6, 4096, 2)]
)
def test_fast_group_norm_matches_torch(mojo_gpu, dtype, n, c, hxw, groups):
    torch.manual_seed(0)
    x = torch.randn(n, c, hxw, dtype=dtype)
    weight = torch.randn(c, dtype=dtype)
    bias = torch.randn(c, dtype=dtype)
    expected = torch.ops.aten.native_group_norm(
        x.float(), weight.float(), bias.float(), n, c, hxw, groups, 1e-5
    )
    actual = torch.ops.aten.native_group_norm(
        x.to(mojo_gpu), weight.to(mojo_gpu), bias.to(mojo_gpu), n, c, hxw, groups, 1e-5
    )
    tolerance = 1e-5 if dtype == torch.float32 else 1e-2
    torch.testing.assert_close(
        actual[0].cpu().float(), expected[0], atol=tolerance, rtol=tolerance
    )
    for got, want in zip(actual[1:], expected[1:], strict=True):
        torch.testing.assert_close(got.cpu(), want, atol=1e-3, rtol=1e-3)


def test_fast_group_norm_refuses_autograd_in_the_forward(mojo_gpu):
    """A grad-requiring group norm must fail at the FORWARD.

    `aten::native_group_norm_backward` is not implemented here, and a raise
    from inside the autograd engine aborts the process on this backend instead
    of propagating, so the refusal cannot wait for NativeGroupNormBackward0.
    Unlike batch norm there is no `training` flag to key on: every
    grad-requiring call is refused, and only turning grad off gets the forward.
    """
    host_input, device_input = _preflight_input((2, 4, 6, 6), mojo_gpu)
    host_weight, device_weight = _preflight_input((4,), mojo_gpu, seed=11)
    host_bias, device_bias = _preflight_input((4,), mojo_gpu, seed=13)

    expected = torch.nn.functional.group_norm(host_input, 2, host_weight, host_bias)
    torch.testing.assert_close(
        torch.nn.functional.group_norm(
            device_input, 2, device_weight, device_bias
        ).cpu(),
        expected,
        atol=1e-5,
        rtol=1e-5,
    )

    # Any one of the three operands wanting a gradient is enough to refuse:
    # the engine would come back for the same unrunnable node either way.
    for which in range(3):
        operands = [device_input.detach(), device_weight.detach(), device_bias.detach()]
        operands[which] = operands[which].requires_grad_()
        with pytest.raises(NotImplementedError, match="native_group_norm_backward"):
            torch.nn.functional.group_norm(operands[0], 2, operands[1], operands[2])

    with torch.no_grad():
        torch.testing.assert_close(
            torch.nn.functional.group_norm(
                device_input.detach().requires_grad_(), 2, device_weight, device_bias
            ).cpu(),
            expected,
            atol=1e-5,
            rtol=1e-5,
        )


def test_fast_group_norm_without_affine(mojo_gpu):
    """No weight and no bias: the kernel must not read the null pointers."""
    torch.manual_seed(0)
    x = torch.randn(2, 6, 32)
    expected = torch.ops.aten.native_group_norm(x, None, None, 2, 6, 32, 3, 1e-5)
    actual = torch.ops.aten.native_group_norm(
        x.to(mojo_gpu), None, None, 2, 6, 32, 3, 1e-5
    )
    for got, want in zip(actual, expected, strict=True):
        torch.testing.assert_close(got.cpu(), want, atol=1e-5, rtol=1e-5)


def test_fast_batch_norm_inference(mojo_device):
    x = torch.randn(2, 64, 14, 14)
    bn = torch.nn.BatchNorm2d(64).eval()
    bn.running_mean.normal_()
    bn.running_var.uniform_(0.5, 2.0)
    bn_dev = torch.nn.BatchNorm2d(64).eval()
    bn_dev.load_state_dict(bn.state_dict())
    bn_dev = bn_dev.to(mojo_device)
    with torch.no_grad():
        torch.testing.assert_close(
            bn_dev(x.to(mojo_device)).cpu(), bn(x), atol=1e-5, rtol=1e-5
        )


def test_fast_layer_norm(mojo_device):
    x = torch.randn(2, 6, 768)
    ln = torch.nn.LayerNorm(768).eval()
    with torch.no_grad():
        ln.weight.normal_()
        ln.bias.normal_()
    ln_dev = torch.nn.LayerNorm(768).eval()
    ln_dev.load_state_dict(ln.state_dict())
    ln_dev = ln_dev.to(mojo_device)
    with torch.no_grad():
        torch.testing.assert_close(
            ln_dev(x.to(mojo_device)).cpu(), ln(x), atol=1e-5, rtol=1e-5
        )


def test_fast_native_layer_norm_stats(mojo_device):
    x = torch.randn(1, 6, 768).to(mojo_device)
    w = torch.ones(768).to(mojo_device)
    b = torch.zeros(768).to(mojo_device)
    out, mean, rstd = torch.native_layer_norm(x, [768], w, b, 1e-5)
    ref_out, ref_mean, ref_rstd = torch.native_layer_norm(
        x.cpu(), [768], w.cpu(), b.cpu(), 1e-5
    )
    torch.testing.assert_close(out.cpu(), ref_out, atol=1e-5, rtol=1e-5)
    torch.testing.assert_close(mean.cpu(), ref_mean, atol=1e-5, rtol=1e-5)
    torch.testing.assert_close(rstd.cpu(), ref_rstd, atol=1e-4, rtol=1e-4)


@pytest.mark.parametrize(
    ("has_weight", "has_bias"),
    [(False, False), (True, False), (False, True), (True, True)],
)
@pytest.mark.parametrize(
    ("input_shape", "normalized_shape", "eps"),
    [((3, 7), (7,), 1e-5), ((2, 3, 4), (3, 4), 0.5)],
)
def test_fast_native_layer_norm_fp32_gpu_optional_affine_without_fill(
    mojo_gpu, monkeypatch, has_weight, has_bias, input_shape, normalized_shape, eps
):
    """The direct GPU ABI handles optional affine tensors without stand-ins."""
    from torch_mojo_backend.eager_kernels import aten_fast

    generator = torch.Generator().manual_seed(20260720)
    numel = math.prod(input_shape)
    cols = math.prod(normalized_shape)
    input_storage = torch.randn(numel + 1, generator=generator)
    input = input_storage[1:].view(input_shape)
    device_storage = input_storage.to(mojo_gpu)
    device_input = device_storage[1:].view(input_shape)

    weight_storage = torch.randn(cols + 1, generator=generator)
    bias_storage = torch.randn(cols + 1, generator=generator)
    weight = weight_storage[1:].view(normalized_shape) if has_weight else None
    bias = bias_storage[1:].view(normalized_shape) if has_bias else None
    device_weight = (
        weight_storage.to(mojo_gpu)[1:].view(normalized_shape) if has_weight else None
    )
    device_bias = (
        bias_storage.to(mojo_gpu)[1:].view(normalized_shape) if has_bias else None
    )
    expected = torch.native_layer_norm(input, normalized_shape, weight, bias, eps)

    def reject_affine_stand_in(*_args, **_kwargs):
        raise AssertionError("optional LayerNorm affine tensor was materialized")

    monkeypatch.setattr(aten_fast, "fast_filled", reject_affine_stand_in)
    actual = aten_fast.fast_aten_native_layer_norm(
        device_input, normalized_shape, device_weight, device_bias, eps
    )

    assert actual is not aten_fast.NOT_HANDLED
    for got, want, tolerance in zip(actual, expected, (1e-5, 1e-5, 1e-4), strict=True):
        torch.testing.assert_close(got.cpu(), want, atol=tolerance, rtol=tolerance)
    torch.testing.assert_close(device_storage.cpu(), input_storage, rtol=0, atol=0)


@pytest.mark.parametrize(
    ("has_weight", "has_bias"),
    [(False, False), (True, False), (False, True), (True, True)],
)
def test_fast_native_layer_norm_gpu_prologue_runs_without_a_gpu(
    monkeypatch, has_weight, has_bias
):
    """The GPU launch arguments must build for every affine combination.

    Every other test of this route needs the `mojo_gpu` fixture, which skips
    on any host without a MAX GPU -- CI included. That is how a name bound
    only inside the affine branch reached the launch-argument tuple. This
    runs the same prologue with no device at all, so the whole matrix is
    checked everywhere the suite runs.
    """
    from torch_mojo_backend.eager_kernels import aten_fast

    device = SimpleNamespace(label="gpu")
    next_ptr = iter(range(300, 400))

    def tensor(shape, dtype=None):
        shape = tuple(shape)
        return SimpleNamespace(
            _shape=shape,
            _mojo_strides=aten_fast._row_major_strides(shape),
            _offset=0,
            _dtype=aten_fast.DType.float32 if dtype is None else dtype,
            _device=device,
            _ptr=next(next_ptr),
            _itemsize=4,
            _numel=math.prod(shape),
            _is_contiguous=True,
        )

    input = tensor((3, 7))
    weight = tensor((7,)) if has_weight else None
    bias = tensor((7,)) if has_bias else None
    calls = []

    def reject_affine_stand_in(*_args, **_kwargs):
        raise AssertionError("optional LayerNorm affine tensor was materialized")

    monkeypatch.setattr(aten_fast, "_t", lambda value: value)
    monkeypatch.setattr(aten_fast, "_tc", lambda value: value)
    monkeypatch.setattr(
        aten_fast, "_alloc", lambda shape, dtype, _device: tensor(shape, dtype)
    )
    monkeypatch.setattr(aten_fast, "_ctx_ptr", lambda _device: 1234)
    monkeypatch.setattr(aten_fast, "fast_filled", reject_affine_stand_in)
    _replace_defined_native_calls(
        monkeypatch,
        {
            ("normalization_forward_ops.mojo", "LayerNormForward"): (
                lambda *args: calls.append(args)
            )
        },
    )

    actual = aten_fast.fast_aten_native_layer_norm(input, (7,), weight, bias, 1e-5)

    assert actual is not aten_fast.NOT_HANDLED
    out, mean, rstd = actual
    assert (out._shape, mean._shape, rstd._shape) == ((3, 7), (3, 1), (3, 1))
    assert len(calls) == 1
    launch = calls[0]
    assert launch[4] == (weight._ptr if has_weight else 0)
    assert launch[5] == (bias._ptr if has_bias else 0)
    assert launch[6:8] == (3, 7)
    assert launch[9:11] == (int(has_weight), int(has_bias))


def test_fast_native_layer_norm_fp32_gpu_noncontiguous_inputs(mojo_gpu):
    input_base = torch.randn(3, 2, 4)
    weight_base = torch.randn(4, 3)
    bias_base = torch.randn(4, 3)
    input = input_base.transpose(0, 1)
    weight = weight_base.t()
    bias = bias_base.t()
    assert not input.is_contiguous()
    assert not weight.is_contiguous()
    assert not bias.is_contiguous()
    expected = torch.native_layer_norm(input, (3, 4), weight, bias, 1e-5)

    device_input = input_base.to(mojo_gpu).transpose(0, 1)
    device_weight = weight_base.to(mojo_gpu).t()
    device_bias = bias_base.to(mojo_gpu).t()
    actual = torch.native_layer_norm(
        device_input, (3, 4), device_weight, device_bias, 1e-5
    )

    assert actual[0].is_contiguous()
    assert actual[1].is_contiguous()
    assert actual[2].is_contiguous()
    assert actual[1].shape == actual[2].shape == torch.Size([2, 1, 1])
    for got, want, tolerance in zip(actual, expected, (1e-5, 1e-5, 1e-4), strict=True):
        torch.testing.assert_close(got.cpu(), want, atol=tolerance, rtol=tolerance)


@pytest.mark.parametrize("cols", [768, 769])
@pytest.mark.parametrize("value", [float("nan"), float("inf"), float("-inf")])
def test_fast_native_layer_norm_fp32_gpu_nonfinite_rows(mojo_gpu, cols, value):
    input = torch.full((2, cols), value, dtype=torch.float32).to(mojo_gpu)

    output, mean, rstd = torch.native_layer_norm(input, (cols,), None, None, 1e-5)

    assert torch.isnan(output.cpu()).all()
    assert torch.isnan(mean.cpu()).all()
    assert torch.isnan(rstd.cpu()).all()


@pytest.mark.parametrize("dtype", [torch.float32, torch.bfloat16])
@pytest.mark.parametrize("cols", [5, 128, 257, 1023, 1024, 2047, 4096])
def test_fast_log_softmax_narrow_rows_match_cpu(mojo_device, dtype, cols):
    # Rows narrower than one full block pass (threads * simd_width) leave
    # threads with no elements; their -inf running max must not NaN the sum.
    x = torch.randn(8, cols).to(dtype)
    result = torch.log_softmax(x.to(mojo_device), dim=-1)
    torch.testing.assert_close(result.cpu(), torch.log_softmax(x, dim=-1))


def test_fast_log_softmax_masked_rows_match_cpu(mojo_device):
    # -inf logits (masking) must not NaN-poison the online rescale.
    x = torch.randn(4, 512)
    x[:, ::3] = float("-inf")
    result = torch.log_softmax(x.to(mojo_device), dim=-1)
    torch.testing.assert_close(result.cpu(), torch.log_softmax(x, dim=-1))


def test_fast_log_softmax_positive_inf_rows_match_cpu(mojo_gpu):
    # A +inf logit makes torch's denominator NaN, so the whole row is NaN. The
    # forward must not launder that away: guarding exp(a - b) with an a == b
    # select turns exp(inf - inf) into 1.0, which returns -inf for the finite
    # entries instead. The finite-sentinel seed keeps torch's answer, so compare
    # with equal_nan: where the NaNs land is the whole point of the case.
    #
    # GPU only. The CPU branch of this kernel (and of softmax) has the same
    # divergence for an unrelated reason -- its exp() does not propagate a NaN
    # argument -- which predates both fixes and wants its own change.
    x = torch.randn(4, 512)
    x[:, 7] = float("inf")
    result = torch.log_softmax(x.to(mojo_gpu), dim=-1)
    torch.testing.assert_close(
        result.cpu(), torch.log_softmax(x, dim=-1), equal_nan=True
    )


# --- aten::var.correction: the (outer, reduce, inner) moment reduction ------
#
# One mechanism serves every layout and regime, so the cases below walk the
# axes it dispatches on rather than any particular shape: reduced axis
# contiguous vs strided, enough outputs to fill the device vs few enough to
# need the reduced axis split across blocks, and both sides of each 16-byte
# vectorization boundary.  Shapes are deliberately awkward where they can be.

# (shape, dim) -- keyed by what the dispatch does with it.
_VAR_LAYOUTS = [
    ((357, 789), 1),  # contiguous reduce, awkward extents
    ((357, 789), 0),  # strided reduce (leading dim), awkward extents
    ((357, 789), None),  # full reduce
    ((4, 5, 6), 1),  # strided reduce with a real outer AND inner
    ((4, 5, 6), (1, 2)),  # adjacent trailing interval
    ((4, 5, 6), (0, 1)),  # adjacent leading interval
    ((4, 5, 6), (0, 2)),  # NON-adjacent: python permutes and materializes
    ((1, 513), 0),  # reduced extent of 1
    ((513, 1), 0),  # inner of 1, single column
    ((16, 33), 1),  # rows shorter than one 16-byte pass per thread
]


@pytest.mark.parametrize("dtype", [torch.float32, torch.bfloat16])
@pytest.mark.parametrize("keepdim", [False, True])
@pytest.mark.parametrize("shape,dim", _VAR_LAYOUTS)
def test_fast_var_layouts_match_cpu(mojo_device, shape, dim, keepdim, dtype):
    x = torch.randn(shape).to(dtype)
    kwargs = {} if dim is None else {"dim": dim}
    result = torch.var(x.to(mojo_device), correction=1, keepdim=keepdim, **kwargs)
    expected = torch.var(x, correction=1, keepdim=keepdim, **kwargs)
    assert result.shape == expected.shape
    # equal_nan: a reduced extent of 1 with correction=1 is nan on both sides.
    torch.testing.assert_close(
        result.cpu(), expected, atol=2e-2, rtol=2e-2, equal_nan=True
    )


@pytest.mark.parametrize(
    "shape,dim",
    [
        # Few outputs, long reduced axis: the split path, both layouts.  The
        # extents are prime-ish so the last shard is short and the merge has
        # to tolerate it.
        ((1048583,), None),
        ((3, 400001), 1),
        ((400001, 3), 0),
        ((2, 65537), 0),
        # Many outputs: no split, stage 1 finalizes on its own.
        ((4099, 1031), 1),
        ((1031, 4099), 0),
    ],
)
def test_fast_var_split_regimes_match_cpu(mojo_gpu, shape, dim):
    # The split factor is derived from the runtime SM count, so these cases
    # straddle the threshold on any card rather than at one fitted size.
    x = torch.rand(shape) * 0.9 + 0.05
    kwargs = {} if dim is None else {"dim": dim}
    result = torch.var(x.to(mojo_gpu), correction=1, **kwargs)
    expected = torch.var(x.double(), correction=1, **kwargs)
    torch.testing.assert_close(result.cpu().double(), expected, atol=1e-6, rtol=1e-4)


@pytest.mark.parametrize("shape,dim", [((1 << 22,), None), ((5, 300000), 1)])
def test_fast_var_survives_large_mean_offset(mojo_gpu, shape, dim):
    """Single-pass moments must not cancel away the answer.

    Accumulating raw sum and sum-of-squares would compute 1e8 - 1e8 here and
    keep almost no significant digits; the moments are taken about an assumed
    mean read from the slice itself, so the accumulator stays the size of the
    variance.
    """
    x = torch.rand(shape) - 0.5 + 1e4
    kwargs = {} if dim is None else {"dim": dim}
    result = torch.var(x.to(mojo_gpu), correction=1, **kwargs).cpu().double()
    expected = torch.var(x.double(), correction=1, **kwargs)
    torch.testing.assert_close(result, expected, atol=0, rtol=1e-3)


# Positions an outlier can occupy relative to the split geometry.  Only the
# FIRST element of a reduced slice is dangerous -- it is the assumed mean --
# but a shard boundary is where a wrong one would first show up as a merge
# bug, so all of them are probed.  n is large enough to take the split path.
_VAR_OUTLIER_N = 1 << 22
_VAR_OUTLIER_POSITIONS = [
    0,  # the assumed mean itself: the catastrophic case
    1,
    8192,
    _VAR_OUTLIER_N // 528,  # first shard boundary at 4 blocks/SM on 132 SMs
    _VAR_OUTLIER_N // 528 - 1,
    _VAR_OUTLIER_N // 2,
    _VAR_OUTLIER_N - 1,
]


@pytest.mark.parametrize("dtype", [torch.float32, torch.bfloat16])
@pytest.mark.parametrize("position", _VAR_OUTLIER_POSITIONS)
@pytest.mark.parametrize("magnitude", [1e6, -1e6])
def test_fast_var_outlier_anywhere_stays_accurate(mojo_gpu, position, magnitude, dtype):
    """A single wild element must not corrupt the reduction, wherever it sits.

    Shifting the moments by the slice's first element is what makes the single
    pass safe on data with a large mean, but it turns that first element into a
    load-bearing choice: if it is the outlier, `M2 = q - s^2/n` becomes a
    difference of two ~4e18 quantities and six digits vanish.  The kernel
    detects that cancellation and re-reads the slice about the mean it now
    knows exactly, so the answer must come out at float32's noise floor for
    every position -- not just the ones where the shift happened to be benign.

    Compared against an fp64 reference computed on the SAME quantized values,
    so this measures the algorithm and not the input dtype.
    """
    g = torch.Generator().manual_seed(0)
    x = torch.randn(_VAR_OUTLIER_N, generator=g, dtype=torch.float64)
    x[position] = magnitude
    quantized = x.to(dtype)
    truth = torch.var(quantized.double(), correction=1)
    ours = torch.var(quantized.to(mojo_gpu), correction=1).cpu().double()
    # bf16 quantizes the outlier itself, which sets the floor at ~2e-3 for any
    # algorithm (stock CUDA torch measures the same); f32 must reach 1e-6.
    rtol = 1e-6 if dtype == torch.float32 else 5e-3
    torch.testing.assert_close(ours, truth, atol=0, rtol=rtol)


@pytest.mark.parametrize("dim", [0, 1])
def test_fast_var_outlier_row_poisons_every_slice(mojo_gpu, dim):
    """The same defect, one per output: index 0 of EVERY reduced slice is wild.

    dim=0 puts the outliers in row 0 (the first element of every column's
    slice) and takes the strided kernel; dim=1 puts them in column 0 and takes
    the contiguous one.  Both must recover per output element -- one bad slice
    must not need, or be rescued by, a whole-tensor decision.
    """
    g = torch.Generator().manual_seed(0)
    x = torch.randn(2048, 2048, generator=g, dtype=torch.float64)
    if dim == 0:
        x[0, :] = 1e6
    else:
        x[:, 0] = 1e6
    quantized = x.to(torch.float32)
    truth = torch.var(quantized.double(), dim=dim, correction=1)
    ours = torch.var(quantized.to(mojo_gpu), dim=dim, correction=1).cpu().double()
    torch.testing.assert_close(ours, truth, atol=0, rtol=1e-5)


@pytest.mark.parametrize("dtype", [torch.float32, torch.bfloat16])
@pytest.mark.parametrize("value", [0.0, 3.0, -1e6])
@pytest.mark.parametrize("shape,dim", [((1 << 22,), None), ((512, 4096), 0)])
def test_fast_var_constant_slice_is_exactly_zero(mojo_gpu, shape, dim, value, dtype):
    """A constant slice has variance exactly 0, not an epsilon and not -0.

    This is the boundary of the cancellation detector: `q` and `M2` are both
    0, so a detector written as a ratio would divide by zero and a sloppy one
    would re-read every constant tensor forever.
    """
    x = torch.full(shape, value, dtype=dtype)
    kwargs = {} if dim is None else {"dim": dim}
    ours = torch.var(x.to(mojo_gpu), correction=1, **kwargs).cpu()
    assert bool((ours == 0).all()), ours
    assert not bool(ours.signbit().any()), "variance came back as -0.0"


@pytest.mark.parametrize("correction", [0, 1, 4, 5, 9])
def test_fast_var_correction_at_or_above_extent_matches_torch(mojo_gpu, correction):
    # torch clamps the divisor at 0 (WelfordOps::project), so correction >= n
    # yields inf for a non-constant sample and nan for a constant one -- not a
    # negative variance.
    x = torch.tensor([[1.0, 2.0, 3.0, 5.0], [7.0, 7.0, 7.0, 7.0]])
    result = torch.var(x.to(mojo_gpu), dim=1, correction=correction)
    expected = torch.var(x, dim=1, correction=correction)
    torch.testing.assert_close(result.cpu(), expected, equal_nan=True)


def test_fast_var_strided_axis_skips_materialization(mojo_gpu, monkeypatch):
    """Reducing a leading dim must reach the bridge with its ORIGINAL dims.

    If it did not, Python would have permuted the operand and materialized a
    full transposed copy first, and the bridge would see the trailing dim
    instead -- an extra read plus an extra write of the whole tensor.
    """
    target = ("reduction_ops.mojo", "VarSpec")
    native_calls = _spy_defined_native_calls(monkeypatch, {target})
    x = torch.randn(64, 48).to(mojo_gpu)
    _ = torch.var(x, dim=0)
    assert len(native_calls[target]) == 1
    assert native_calls[target][0][0][1] == (0,)


# ---------------------------------------------------------------------------
# cumsum: INNER (trailing/stride-1 dim, block.prefix_sum-cooperating blocks,
# with a 3-pass workspace scan once there are too few independent lines to
# fill the device) and OUTER (dim=0 on a contiguous rank-2 tensor, one
# thread per independent column, no cooperation needed) -- see
# torch_mojo_backend/eager_kernels/nn_ops/cumsum_kernels.mojo for the
# kernels and nn_ops.mojo's `_cumsum_spec_into_go` for the dispatch. Both
# families are CUDA-only (gated by `_device.api == "cuda"` in
# `fast_aten_cumsum`): non-CUDA devices, including the `mojo:cpu` test
# device, keep exactly the int64/int32/float32-trailing-dim-only surface
# `main` had before this file, through a portable (but unoptimized)
# `_parallel_for`-based kernel -- see `test_fast_cumsum_...non_cuda...`
# below.
#
# Shapes are chosen to land deliberately on both sides of the regime split
# (INNER "many independent lines" vs. the long-line workspace path, which
# is picked below `sm_count * FILL_WAVES` independent lines -- read at
# runtime, so a handful of rows lands in the workspace regime on any real
# datacenter GPU without hardcoding a specific SM count).
# ---------------------------------------------------------------------------

# (shape, dim) -- covers INNER many-lines, INNER workspace (few long lines),
# OUTER many-lines and OUTER narrow, plus edge cases (single element/line/
# column, an exact multiple of a tile size vs. one off it, and the awkward
# 357x789 shape AGENTS.md's benchmark-harness notes ask every kernel to
# include).
_CUMSUM_SHAPES = [
    ((4096, 4096), 1),  # INNER, many lines
    ((4096, 4096), 0),  # OUTER, many lines
    ((357, 789), 1),  # INNER, awkward extents
    ((357, 789), 0),  # OUTER, awkward extents
    ((1, 200_000), 1),  # INNER, workspace (one very long line)
    ((8, 100_003), 1),  # INNER, workspace (few long lines, off-by-one)
    ((4, 1024 * 5), 1),  # INNER, workspace, exact tile multiple
    ((200_003, 4), 0),  # OUTER, narrow (few columns, many rows)
    ((1, 4096), 1),  # single line
    ((500, 1), 1),  # single column, INNER framing
    ((1, 4096), 0),  # single row, OUTER framing
    ((4096, 1), 0),  # single column, OUTER framing
    ((1, 1), 1),  # single element
    ((500, 256), 1),  # exact tile multiple (256 threads)
    ((500, 257), 1),  # one past a tile multiple
]


@pytest.mark.parametrize("shape,dim", _CUMSUM_SHAPES)
def test_fast_cumsum_shapes_match_cpu(mojo_gpu, shape, dim):
    # Compared against a float64 reference computed from the SAME float32
    # input, not CPU's own float32 cumsum: two float32 accumulations done in
    # a different order (ours: one thread per line/column, strictly
    # sequential; CPU's: whatever order aten's own reduction takes) round
    # differently at every step and can legitimately disagree, without
    # either being wrong -- this tolerance measures the fast kernel's OWN
    # float32 accumulation error, the same idiom
    # test_fast_var_split_regimes_match_cpu above uses. A random walk this
    # long (up to 200,003 steps for the narrow-outer shape) crosses near
    # zero somewhere, where a purely RELATIVE tolerance is meaningless (a
    # ~1e-3 absolute difference against an expected value of 6e-4 is a
    # "250000%" relative error despite being an entirely unremarkable fp32
    # accumulation error at that depth) -- atol carries the bound there;
    # rtol matters for the large-magnitude tail. Seeded for reproducibility
    # (swept seeds 0-7 across every shape here to size this bound with
    # margin, not picked to make one run pass).
    torch.manual_seed(0)
    x = torch.randn(shape)
    result = torch.cumsum(x.to(mojo_gpu), dim=dim).cpu().double()
    expected = torch.cumsum(x.double(), dim=dim)
    torch.testing.assert_close(result, expected, rtol=2e-3, atol=2e-2)


# Measured worst relative error (against an fp64 reference, moderate
# accumulation depths up to 4096 elements -- see the PR description for the
# full sweep): bf16 up to ~0.39%, float16 up to ~0.90% (float16's narrower
# mantissa loses more on the OUTER kernel's longer per-thread serial
# accumulation). 3% is roughly a 3-4x margin over that measured worst case
# -- tightened from an ungrounded 6% bound used during kernel development
# (the fast kernels accumulate bf16/f16 in float32 internally, matching
# stock torch's own cumsum accumulation, so the error floor here is the
# INPUT's bf16/f16 quantization compounding over the scan, not anything the
# kernel design adds).
_CUMSUM_LOWP_RTOL = 3e-2
_CUMSUM_LOWP_ATOL = 3e-2


@pytest.mark.parametrize(
    "dtype", [torch.bfloat16, torch.float16, torch.int32, torch.int64]
)
@pytest.mark.parametrize("shape,dim", [((4096, 4096), 1), ((4096, 4096), 0)])
def test_fast_cumsum_dtypes_match_cpu(mojo_gpu, shape, dim, dtype):
    """Dtype sweep on the INNER 'many lines' and OUTER regimes."""
    torch.manual_seed(0)
    if dtype.is_floating_point:
        x = torch.randn(shape).to(dtype)
    else:
        x = torch.randint(-100, 100, shape, dtype=dtype)
    result = torch.cumsum(x.to(mojo_gpu), dim=dim)
    expected = torch.cumsum(x, dim=dim)
    assert result.dtype == expected.dtype
    if dtype.is_floating_point:
        torch.testing.assert_close(
            result.cpu(), expected, rtol=_CUMSUM_LOWP_RTOL, atol=_CUMSUM_LOWP_ATOL
        )
    else:
        torch.testing.assert_close(result.cpu(), expected, rtol=0, atol=0)


@pytest.mark.parametrize(
    "dtype", [torch.bfloat16, torch.float16, torch.int32, torch.int64]
)
@pytest.mark.parametrize("shape,dim", [((4, 5120), 1), ((8, 100_003), 1)])
def test_fast_cumsum_workspace_dtypes_match_cpu(mojo_gpu, shape, dim, dtype):
    """A2 follow-up: the long-line 3-pass workspace path was only ever
    correctness-tested in float32 before integration -- this covers
    bf16/f16/i32/i64 through the SAME regime (few rows, so `enqueue_
    cumsum_rows_workspace` is what actually runs, not the many-lines
    kernel)."""
    torch.manual_seed(0)
    if dtype.is_floating_point:
        x = torch.randn(shape).to(dtype)
    else:
        x = torch.randint(-100, 100, shape, dtype=dtype)
    result = torch.cumsum(x.to(mojo_gpu), dim=dim)
    expected = torch.cumsum(x, dim=dim)
    assert result.dtype == expected.dtype
    if dtype.is_floating_point:
        torch.testing.assert_close(
            result.cpu(), expected, rtol=_CUMSUM_LOWP_RTOL, atol=_CUMSUM_LOWP_ATOL
        )
    else:
        torch.testing.assert_close(result.cpu(), expected, rtol=0, atol=0)


@pytest.mark.parametrize("dim", [-1, -2])
def test_fast_cumsum_negative_dim(mojo_gpu, dim):
    torch.manual_seed(0)
    x = torch.randn(64, 4096)
    result = torch.cumsum(x.to(mojo_gpu), dim=dim).cpu().double()
    expected = torch.cumsum(x.double(), dim=dim)
    torch.testing.assert_close(result, expected, rtol=2e-3, atol=2e-2)


def test_fast_cumsum_dtype_kwarg_casts_before_accumulating(mojo_gpu):
    """torch casts the input to `dtype` first, then accumulates in it --
    verified against stock torch directly (not just a claim): an int64
    input with dtype=torch.float32 upcasts before scanning, and a float32
    input with dtype=torch.int32 TRUNCATES before scanning (not the other
    way around, i.e. not sum-then-cast)."""
    x = torch.arange(1, 9, dtype=torch.int64).reshape(2, 4)
    result = torch.cumsum(x.to(mojo_gpu), dim=1, dtype=torch.float32)
    expected = torch.cumsum(x, dim=1, dtype=torch.float32)
    assert result.dtype == torch.float32
    torch.testing.assert_close(result.cpu(), expected)

    y = torch.tensor([[1.7, 2.7, 3.7, 0.2]])
    result = torch.cumsum(y.to(mojo_gpu), dim=1, dtype=torch.int32)
    expected = torch.cumsum(y, dim=1, dtype=torch.int32)
    assert result.dtype == torch.int32
    torch.testing.assert_close(result.cpu(), expected)


@pytest.mark.parametrize("dtype", [torch.int32, torch.uint8, torch.bool])
def test_fast_cumsum_integer_promotes_to_int64(mojo_gpu, dtype):
    """No dtype= kwarg: torch promotes every integer/bool cumsum to int64
    (same rule fast_aten_sum already applies) -- verified directly against
    stock torch, not assumed. int8/int16 are NOT covered here: promoting
    them needs a cast through `_CAST_DTYPES`, which does not include those
    two dtypes -- a pre-existing gap `fast_aten_sum` has for the same
    reason (verified: `torch.sum` on an int8 mojo tensor also declines),
    not something specific to or fixed by this file."""
    if dtype == torch.bool:
        x = torch.randint(0, 2, (8, 16), dtype=torch.bool)
    else:
        x = torch.randint(0, 20, (8, 16), dtype=dtype)
    result = torch.cumsum(x.to(mojo_gpu), dim=1)
    expected = torch.cumsum(x, dim=1)
    assert result.dtype == torch.int64 == expected.dtype
    torch.testing.assert_close(result.cpu(), expected)


def test_fast_cumsum_noncontiguous_input_materializes_correctly(mojo_gpu):
    """A non-contiguous operand is NOT declined here: `fast_aten_cumsum`
    materializes it (`a._contig()`) before the boundary call, same as
    `_try_spec_unary` already did for this op pre-integration -- so the
    result must be numerically correct, not just non-crashing."""
    torch.manual_seed(0)
    x = torch.randn(64, 128).t()  # transpose of a contiguous tensor
    assert not x.is_contiguous()
    result = torch.cumsum(x.to(mojo_gpu), dim=1).cpu().double()
    expected = torch.cumsum(x.double(), dim=1)
    torch.testing.assert_close(result, expected, rtol=2e-3, atol=1e-2)


def test_fast_cumsum_declines_middle_dim_on_rank3(mojo_gpu):
    """rank>=3 non-trailing dims are out of this kernel family's scope and
    must raise (eager has no graph fallback), not silently compute the
    wrong axis."""
    x = torch.randn(4, 5, 6).to(mojo_gpu)
    with pytest.raises(NotImplementedError):
        torch.cumsum(x, dim=1)


def test_fast_cumsum_new_dtype_and_dim_decline_on_non_cuda_device():
    """dim=0 and bf16/f16 route through the new, CUDA-only fast kernels
    (`is_cuda` gate in `fast_aten_cumsum`) -- on a non-CUDA device, the
    `mojo:cpu` test device used here (no real GPU required, unlike AMD/Apple
    which this repo has no way to exercise in CI), these must decline
    exactly as they did on `main` before this file existed, not silently run
    an unmeasured kernel. int64/int32/float32 on the trailing dim is
    unaffected -- same portable `_parallel_for` kernel as always.
    """
    mojo_cpu = f"mojo:{len(get_accelerators()) - 1}"
    x32 = torch.randn(8, 16)
    with pytest.raises(NotImplementedError):
        torch.cumsum(x32.to(mojo_cpu), dim=0)
    xbf16 = torch.randn(8, 16).to(torch.bfloat16)
    with pytest.raises(NotImplementedError):
        torch.cumsum(xbf16.to(mojo_cpu), dim=1)
    # The pre-existing surface still works on that same device.
    result = torch.cumsum(x32.to(mojo_cpu), dim=1)
    torch.testing.assert_close(result.cpu(), torch.cumsum(x32, dim=1))


def test_fast_native_layer_norm_bf16_gpu_preserves_generic_path(mojo_gpu):
    input = torch.randn(3, 65).bfloat16()
    weight = torch.randn(65).bfloat16()
    bias = torch.randn(65).bfloat16()
    expected = torch.native_layer_norm(input, (65,), weight, bias, 1e-5)

    actual = torch.native_layer_norm(
        input.to(mojo_gpu), (65,), weight.to(mojo_gpu), bias.to(mojo_gpu), 1e-5
    )

    assert actual[0].dtype == torch.bfloat16
    assert actual[1].dtype == actual[2].dtype == torch.bfloat16
    for got, want in zip(actual, expected, strict=True):
        got_cpu = got.cpu()
        torch.testing.assert_close(
            got_cpu, want.to(got_cpu.dtype), atol=2e-2, rtol=2e-2
        )


def test_fast_native_layer_norm_weight_only_autograd(mojo_gpu):
    host_input = torch.randn(4, 65, requires_grad=True)
    host_weight = torch.randn(65, requires_grad=True)
    device_input = host_input.detach().to(mojo_gpu).requires_grad_()
    device_weight = host_weight.detach().to(mojo_gpu).requires_grad_()
    input_version = device_input._version
    weight_version = device_weight._version

    expected, expected_mean, expected_rstd = torch.native_layer_norm(
        host_input, (65,), host_weight, None, 1e-5
    )
    actual, mean, rstd = torch.native_layer_norm(
        device_input, (65,), device_weight, None, 1e-5
    )

    assert actual.requires_grad
    assert not mean.requires_grad and not rstd.requires_grad
    assert device_input._version == input_version
    assert device_weight._version == weight_version
    torch.testing.assert_close(actual.cpu(), expected.detach(), atol=1e-5, rtol=1e-5)
    torch.testing.assert_close(mean.cpu(), expected_mean, atol=1e-5, rtol=1e-5)
    torch.testing.assert_close(rstd.cpu(), expected_rstd, atol=1e-4, rtol=1e-4)

    grad_output = torch.randn_like(host_input)
    expected.backward(grad_output)
    actual.backward(grad_output.to(mojo_gpu))

    assert device_input._version == input_version
    assert device_weight._version == weight_version
    torch.testing.assert_close(
        device_input.grad.cpu(), host_input.grad, atol=2e-3, rtol=2e-3
    )
    torch.testing.assert_close(
        device_weight.grad.cpu(), host_weight.grad, atol=2e-3, rtol=2e-3
    )


@pytest.mark.parametrize(("rows", "cols", "storage_offset"), [(3, 7, 1), (257, 65, 0)])
@pytest.mark.parametrize(
    "output_mask",
    [
        (True, True, True),
        (True, False, False),
        (False, True, False),
        (False, False, True),
        (False, False, False),
    ],
)
def test_fast_native_layer_norm_backward_output_masks(
    mojo_gpu, rows, cols, storage_offset, output_mask
):
    generator = torch.Generator().manual_seed(20260718)
    input_storage = torch.randn(rows * cols + storage_offset, generator=generator)
    grad_storage = torch.randn(rows * cols + storage_offset, generator=generator)
    weight_storage = torch.randn(cols + storage_offset, generator=generator)
    bias_storage = torch.randn(cols + storage_offset, generator=generator)

    input = input_storage[storage_offset:].view(rows, cols)
    grad_output = grad_storage[storage_offset:].view(rows, cols)
    weight = weight_storage[storage_offset:]
    bias = bias_storage[storage_offset:]
    _, mean, rstd = torch.native_layer_norm(input, [cols], weight, bias, 1e-5)
    expected = torch.ops.aten.native_layer_norm_backward.default(
        grad_output, input, [cols], mean, rstd, weight, bias, output_mask
    )

    mojo_input = input_storage.to(mojo_gpu)[storage_offset:].view(rows, cols)
    mojo_grad = grad_storage.to(mojo_gpu)[storage_offset:].view(rows, cols)
    mojo_weight = weight_storage.to(mojo_gpu)[storage_offset:]
    mojo_bias = bias_storage.to(mojo_gpu)[storage_offset:]
    _, mojo_mean, mojo_rstd = torch.native_layer_norm(
        mojo_input, [cols], mojo_weight, mojo_bias, 1e-5
    )
    actual = torch.ops.aten.native_layer_norm_backward.default(
        mojo_grad,
        mojo_input,
        [cols],
        mojo_mean,
        mojo_rstd,
        mojo_weight,
        mojo_bias,
        output_mask,
    )

    for got, want, requested in zip(actual, expected, output_mask, strict=True):
        assert (got is not None) == requested
        if requested:
            torch.testing.assert_close(got.cpu(), want, atol=2e-3, rtol=2e-3)


@pytest.mark.parametrize(
    "invalid",
    [
        "grad_shape",
        "input_dtype",
        "mean_dtype",
        "mean_numel",
        "mean_device",
        "weight_shape",
        "bias_shape",
        "missing_weight",
        "missing_bias",
    ],
)
def test_fast_native_layer_norm_backward_rejects_metadata_before_materializing(
    mojo_gpu, monkeypatch, invalid
):
    from torch_mojo_backend.eager_kernels import aten_fast
    from torch_mojo_backend.mojo_device.torch_mojo_tensor import TorchMojoTensor

    rows, cols = 3, 4
    # Keep every kernel input noncontiguous so an accidental materialization
    # after incomplete validation is observable.
    input = torch.randn(rows, 2 * cols).to(mojo_gpu)[:, ::2]
    grad_output = torch.randn(rows, 2 * cols).to(mojo_gpu)[:, ::2]
    weight = torch.randn(2 * cols).to(mojo_gpu)[::2]
    bias = torch.randn(2 * cols).to(mojo_gpu)[::2]
    _, mean, rstd = torch.native_layer_norm(input, (cols,), weight, bias, 1e-5)
    output_mask = (True, True, True)

    if invalid == "grad_shape":
        grad_output = grad_output[:, :-1]
    elif invalid == "input_dtype":
        input = input.to(torch.float16)
    elif invalid == "mean_dtype":
        mean = mean.to(torch.float16)
    elif invalid == "mean_numel":
        mean = mean.view(-1)[:-1]
    elif invalid == "mean_device":
        mojo_cpu = f"mojo:{len(get_accelerators()) - 1}"
        mean = mean.cpu().to(mojo_cpu)
    elif invalid == "weight_shape":
        weight = weight.view(2, 2)
    elif invalid == "bias_shape":
        bias = bias.view(2, 2)
    elif invalid == "missing_weight":
        weight = None
    elif invalid == "missing_bias":
        bias = None

    def fail_materialization(_self):
        raise AssertionError("invalid LayerNorm metadata materialized a tensor")

    monkeypatch.setattr(
        TorchMojoTensor, "_materialize_contiguous", fail_materialization
    )
    assert (
        aten_fast.fast_aten_native_layer_norm_backward(
            grad_output, input, (cols,), mean, rstd, weight, bias, output_mask
        )
        is aten_fast.NOT_HANDLED
    )


def test_fast_native_layer_norm_backward_empty_rows(mojo_gpu):
    from torch_mojo_backend.eager_kernels import aten_fast

    cols = 65
    input = torch.empty(0, cols).to(mojo_gpu)
    grad_output = torch.empty_like(input)
    mean = torch.empty(0, 1).to(mojo_gpu)
    rstd = torch.empty(0, 1).to(mojo_gpu)
    weight = torch.randn(cols).to(mojo_gpu)
    bias = torch.randn(cols).to(mojo_gpu)

    grad_input, grad_weight, grad_bias = aten_fast.fast_aten_native_layer_norm_backward(
        grad_output, input, (cols,), mean, rstd, weight, bias, (True, True, True)
    )

    assert grad_input.shape == input.shape
    assert grad_input.numel() == 0
    torch.testing.assert_close(grad_weight.cpu(), torch.zeros(cols))
    torch.testing.assert_close(grad_bias.cpu(), torch.zeros(cols))


@pytest.mark.parametrize("affine", [False, True])
def test_fast_layer_norm_training_backward(mojo_gpu, affine):
    from torch_mojo_backend.mojo_device.mojo_device_aten_ops import EAGER_CALL_COUNTERS

    generator = torch.Generator().manual_seed(20260718)
    shape = (2, 16, 384)
    input = torch.randn(shape, generator=generator)
    grad_output = torch.randn(shape, generator=generator)
    weight = torch.randn(384, generator=generator) if affine else None
    bias = torch.randn(384, generator=generator) if affine else None

    reference_input = input.clone().requires_grad_()
    reference_weight = weight.clone().requires_grad_() if affine else None
    reference_bias = bias.clone().requires_grad_() if affine else None
    torch.nn.functional.layer_norm(
        reference_input, (384,), reference_weight, reference_bias, 1e-5
    ).backward(grad_output)

    mojo_input = input.to(mojo_gpu).requires_grad_()
    mojo_weight = weight.to(mojo_gpu).requires_grad_() if affine else None
    mojo_bias = bias.to(mojo_gpu).requires_grad_() if affine else None
    backward_counter = EAGER_CALL_COUNTERS["aten::native_layer_norm_backward"]
    calls_before = backward_counter.call_count
    mojo_output = torch.nn.functional.layer_norm(
        mojo_input, (384,), mojo_weight, mojo_bias, 1e-5
    )
    assert type(mojo_output.grad_fn).__name__ == "NativeLayerNormBackward0"
    assert backward_counter.call_count == calls_before
    mojo_output.backward(grad_output.to(mojo_gpu))
    assert backward_counter.call_count == calls_before + 1

    assert mojo_input.grad is not None
    torch.testing.assert_close(
        mojo_input.grad.cpu(), reference_input.grad, atol=2e-4, rtol=2e-4
    )
    if affine:
        assert mojo_weight.grad is not None
        assert mojo_bias.grad is not None
        torch.testing.assert_close(
            mojo_weight.grad.cpu(), reference_weight.grad, atol=2e-3, rtol=2e-3
        )
        torch.testing.assert_close(
            mojo_bias.grad.cpu(), reference_bias.grad, atol=2e-3, rtol=2e-3
        )


@pytest.mark.parametrize("requires", ["input", "weight", "bias"])
def test_fast_layer_norm_autograd_requests_only_needed_output(
    mojo_gpu, monkeypatch, requires
):
    input = torch.randn(3, 65).to(mojo_gpu)
    weight = torch.randn(65).to(mojo_gpu)
    bias = torch.randn(65).to(mojo_gpu)
    tensors = {"input": input, "weight": weight, "bias": bias}
    tensors[requires].requires_grad_()
    target = ("normalization_backward_ops.mojo", "LayerNormBackwardF32")
    native_calls = _spy_defined_native_calls(monkeypatch, {target})
    output = torch.nn.functional.layer_norm(input, (65,), weight, bias, 1e-5)
    output.backward(torch.ones(3, 65).to(mojo_gpu))

    expected_mask_bits = 1 << ("input", "weight", "bias").index(requires)
    assert [args[-2] for args, _ in native_calls[target]] == [expected_mask_bits]
    for name, tensor in tensors.items():
        assert (tensor.grad is not None) == (name == requires)


def test_fast_layer_norm_native_saved_tensor_hooks(mojo_gpu):
    generator = torch.Generator().manual_seed(20260722)
    host_input = torch.randn(3, 7, generator=generator)
    host_weight = torch.randn(7, generator=generator)
    host_bias = torch.randn(7, generator=generator)
    grad_output = torch.randn(3, 7, generator=generator)

    reference = [
        host_input.clone().requires_grad_(),
        host_weight.clone().requires_grad_(),
        host_bias.clone().requires_grad_(),
    ]
    torch.nn.functional.layer_norm(reference[0], (7,), *reference[1:]).backward(
        grad_output
    )

    actual = [
        host_input.to(mojo_gpu).requires_grad_(),
        host_weight.to(mojo_gpu).requires_grad_(),
        host_bias.to(mojo_gpu).requires_grad_(),
    ]
    hook_calls = []

    def pack(tensor):
        hook_calls.append(("pack", tensor.device.type))
        return tensor.cpu()

    def unpack(tensor):
        hook_calls.append(("unpack", tensor.device.type))
        return tensor.to(mojo_gpu)

    with torch.autograd.graph.saved_tensors_hooks(pack, unpack):
        output = torch.nn.functional.layer_norm(actual[0], (7,), *actual[1:])
        output.backward(grad_output.to(mojo_gpu))

    assert hook_calls.count(("pack", "mojo")) == 5
    assert hook_calls.count(("unpack", "cpu")) == 5
    for got, want in zip(actual, reference, strict=True):
        assert got.grad is not None
        torch.testing.assert_close(got.grad.cpu(), want.grad, atol=2e-3, rtol=2e-3)


def test_fast_layer_norm_native_double_backward(mojo_gpu):
    generator = torch.Generator().manual_seed(20260722)
    host_input = torch.randn(3, 7, generator=generator)
    host_weight = torch.randn(7, generator=generator)
    host_bias = torch.randn(7, generator=generator)
    first_seed = torch.randn(3, 7, generator=generator)
    second_seed = torch.randn(3, 7, generator=generator)

    def derivatives(input, weight, bias, seed1, seed2):
        output = torch.nn.functional.layer_norm(input, (7,), weight, bias)
        (input_grad,) = torch.autograd.grad(
            output, input, grad_outputs=seed1, create_graph=True
        )
        second_grads = torch.autograd.grad(
            input_grad, (input, weight), grad_outputs=seed2
        )
        return input_grad, *second_grads

    reference = [
        host_input.clone().requires_grad_(),
        host_weight.clone().requires_grad_(),
        host_bias.clone().requires_grad_(),
    ]
    expected = derivatives(*reference, first_seed, second_seed)

    actual = [
        host_input.to(mojo_gpu).requires_grad_(),
        host_weight.to(mojo_gpu).requires_grad_(),
        host_bias.to(mojo_gpu).requires_grad_(),
    ]
    got = derivatives(*actual, first_seed.to(mojo_gpu), second_seed.to(mojo_gpu))

    assert type(got[0].grad_fn).__name__ == "NativeLayerNormBackwardBackward0"
    for actual_grad, expected_grad in zip(got, expected, strict=True):
        torch.testing.assert_close(
            actual_grad.cpu(), expected_grad, atol=2e-3, rtol=2e-3
        )


@pytest.mark.parametrize("mutated", ["input", "weight", "bias", "mean", "rstd"])
def test_fast_layer_norm_backward_rejects_mutated_saved_tensor(mojo_gpu, mutated):
    input = torch.randn(4, 65).to(mojo_gpu).requires_grad_()
    weight = torch.randn(65).to(mojo_gpu).requires_grad_()
    bias = torch.randn(65).to(mojo_gpu).requires_grad_()
    output, mean, rstd = torch.native_layer_norm(input, (65,), weight, bias, 1e-5)
    assert not mean.requires_grad
    assert not rstd.requires_grad

    with torch.no_grad():
        target = {
            "input": input,
            "weight": weight,
            "bias": bias,
            "mean": mean,
            "rstd": rstd,
        }[mutated]
        target.add_(torch.ones_like(target))

    with pytest.raises(RuntimeError, match="modified by an inplace operation"):
        output.backward(torch.randn(4, 65).to(mojo_gpu))


def test_fast_layer_norm_backward_allows_mutated_forward_output(mojo_gpu):
    input = torch.randn(4, 65).to(mojo_gpu).requires_grad_()
    output = torch.nn.functional.layer_norm(input, (65,))

    with torch.no_grad():
        output.add_(torch.ones_like(output))
    output.backward(torch.randn(4, 65).to(mojo_gpu))

    assert input.grad is not None


def test_fast_native_layer_norm_rejects_wrong_affine_shape(mojo_gpu, monkeypatch):
    from torch_mojo_backend.eager_kernels import aten_fast

    input = torch.randn(2, 2, 3).to(mojo_gpu)
    wrong_shape_same_numel = torch.randn(6).to(mojo_gpu)
    bias = torch.randn(2, 3).to(mojo_gpu)

    def reject_materialization(*_args, **_kwargs):
        raise AssertionError("invalid LayerNorm call materialized a tensor")

    monkeypatch.setattr(aten_fast, "_tc", reject_materialization)
    monkeypatch.setattr(aten_fast, "_alloc", reject_materialization)

    assert (
        aten_fast.fast_aten_native_layer_norm(
            input, (2, 3), wrong_shape_same_numel, bias, 1e-5
        )
        is aten_fast.NOT_HANDLED
    )


@pytest.mark.parametrize(
    ("p", "train", "should_advance_rng"),
    [
        (0.0, True, True),
        (0.2, True, True),
        (1.0, True, False),
        (0.2, False, False),
        (1.0, False, False),
        (0.2, None, True),
    ],
)
def test_fast_native_dropout_forward_semantics(mojo_gpu, p, train, should_advance_rng):
    input = torch.linspace(-4.0, 4.0, 257)
    mojo_input = input.to(mojo_gpu)
    torch.mojo.manual_seed_all((1 << 63) + 20260718)
    before = torch.mojo.get_rng_state(mojo_input.device)

    output, mask = torch.ops.aten.native_dropout.default(mojo_input, p, train)
    after = torch.mojo.get_rng_state(mojo_input.device)

    assert output is not mojo_input
    assert output.shape == mojo_input.shape
    assert mask.shape == mojo_input.shape
    assert mask.dtype == torch.bool
    assert mask.device.type == "mojo"
    assert torch.equal(before, after) != should_advance_rng

    host_mask = mask.cpu()
    if train is False:
        assert host_mask.all()
        expected = input
    elif p == 1.0:
        assert not host_mask.any()
        expected = torch.zeros_like(input)
    else:
        scale = 1.0 / (1.0 - p)
        expected = input * host_mask * scale
    torch.testing.assert_close(output.cpu(), expected, atol=1e-6, rtol=1e-6)


@pytest.mark.parametrize("p", [-0.1, 1.1, float("nan")])
def test_fast_native_dropout_inference_ignores_probability(mojo_gpu, p):
    input = torch.linspace(-4.0, 4.0, 17)
    mojo_input = input.to(mojo_gpu)
    torch.mojo.manual_seed_all(20260718)
    before = torch.mojo.get_rng_state(mojo_input.device)

    output, mask = torch.ops.aten.native_dropout.default(mojo_input, p, False)

    assert output is not mojo_input
    torch.testing.assert_close(output.cpu(), input)
    assert mask.cpu().all()
    torch.testing.assert_close(torch.mojo.get_rng_state(mojo_input.device), before)


def test_fast_native_dropout_empty_does_not_advance_rng(mojo_gpu):
    input = torch.empty(0, 7).to(mojo_gpu)
    torch.mojo.manual_seed_all(20260718)
    before = torch.mojo.get_rng_state(input.device)
    output, mask = torch.ops.aten.native_dropout.default(input, 0.0, True)
    after = torch.mojo.get_rng_state(input.device)

    assert output is not input
    assert output.shape == input.shape
    assert mask.shape == input.shape
    assert mask.dtype == torch.bool
    torch.testing.assert_close(after, before)


def test_fast_native_dropout_rng_state_replays_exactly(mojo_gpu):
    input = torch.randn(4097).to(mojo_gpu)
    torch.mojo.manual_seed_all((1 << 63) + 0x12345)
    initial = torch.mojo.get_rng_state(input.device)
    first_output, first_mask = torch.ops.aten.native_dropout.default(input, 0.2, True)
    advanced = torch.mojo.get_rng_state(input.device)

    torch.mojo.set_rng_state(initial, input.device)
    replay_output, replay_mask = torch.ops.aten.native_dropout.default(input, 0.2, True)

    torch.testing.assert_close(replay_mask.cpu(), first_mask.cpu())
    torch.testing.assert_close(replay_output.cpu(), first_output.cpu())
    torch.testing.assert_close(torch.mojo.get_rng_state(input.device), advanced)


def _philox_rng_state(seed: int, counter: int) -> torch.Tensor:
    encoded = seed.to_bytes(8, "little") + counter.to_bytes(8, "little")
    return torch.tensor(list(encoded), dtype=torch.uint8)


def _decode_philox_rng_state(state: torch.Tensor) -> tuple[int, int]:
    encoded = bytes(state.reshape(-1).tolist())
    return int.from_bytes(encoded[:8], "little"), int.from_bytes(encoded[8:], "little")


def test_fast_native_dropout_reserves_exact_full_width_interval(mojo_gpu, monkeypatch):
    from torch_mojo_backend.eager_kernels import aten_fast

    calls = []
    _replace_defined_native_calls(
        monkeypatch,
        {("dropout_ops.mojo", "NativeDropoutF32"): lambda *args: calls.append(args)},
    )
    input = torch.arange(9, dtype=torch.float32).to(mojo_gpu)
    seed = (1 << 63) + 0x1234_5678
    counter = (1 << 63) + 0x9ABC_DEF0
    torch.mojo.set_rng_state(_philox_rng_state(seed, counter), input.device)

    output, mask = aten_fast.fast_aten_native_dropout(input, 0.0, True)

    assert output.shape == input.shape
    assert mask.shape == input.shape
    assert mask.dtype == torch.bool
    assert len(calls) == 1
    args = calls[0]
    assert args[3] == 9
    assert args[4] == 0.0
    assert args[5:9] == (
        seed & 0xFFFF_FFFF,
        seed >> 32,
        counter & 0xFFFF_FFFF,
        counter >> 32,
    )
    assert _decode_philox_rng_state(torch.mojo.get_rng_state(input.device)) == (
        seed,
        counter + 3,
    )


def test_fast_native_dropout_reservation_wrap_does_not_mutate_state(
    mojo_gpu, monkeypatch
):
    from torch_mojo_backend.eager_kernels import aten_fast

    calls = []
    _replace_defined_native_calls(
        monkeypatch,
        {("dropout_ops.mojo", "NativeDropoutF32"): lambda *args: calls.append(args)},
    )
    input = torch.arange(4, dtype=torch.float32).to(mojo_gpu)
    seed = (1 << 64) - 1
    torch.mojo.set_rng_state(_philox_rng_state(seed, (1 << 64) - 2), input.device)

    aten_fast.fast_aten_native_dropout(input, 0.0, True)
    endpoint = torch.mojo.get_rng_state(input.device)
    assert _decode_philox_rng_state(endpoint) == (seed, (1 << 64) - 1)
    assert len(calls) == 1

    with pytest.raises(OverflowError, match="wrap uint64"):
        aten_fast.fast_aten_native_dropout(input, 0.0, True)
    torch.testing.assert_close(torch.mojo.get_rng_state(input.device), endpoint)
    assert len(calls) == 1


@pytest.mark.parametrize("p", [-0.1, 1.1, float("nan"), float("inf")])
def test_fast_native_dropout_invalid_probability_does_not_touch_rng_or_input(
    mojo_gpu, monkeypatch, p
):
    from torch_mojo_backend.eager_kernels import aten_fast
    from torch_mojo_backend.mojo_device.torch_mojo_tensor import TorchMojoTensor

    input = torch.randn(3, 8).to(mojo_gpu)[:, ::2]
    torch.mojo.manual_seed_all(20260718)
    before = torch.mojo.get_rng_state(input.device)

    def fail_materialization(_self):
        raise AssertionError("invalid dropout metadata materialized the input")

    monkeypatch.setattr(
        TorchMojoTensor, "_materialize_contiguous", fail_materialization
    )
    with pytest.raises(RuntimeError, match="probability has to be between 0 and 1"):
        aten_fast.fast_aten_native_dropout(input, p, True)
    torch.testing.assert_close(torch.mojo.get_rng_state(input.device), before)


def test_fast_native_dropout_backward_multiplication_semantics(mojo_gpu):
    grad_output = torch.tensor([-0.0, -2.0, float("nan"), float("inf")])
    mask = torch.tensor([True, False, False, False])
    result = torch.ops.aten.native_dropout_backward.default(
        grad_output.to(mojo_gpu), mask.to(mojo_gpu), 2.0
    ).cpu()
    expected = grad_output * mask * 2.0

    torch.testing.assert_close(result, expected, equal_nan=True)
    assert torch.signbit(result[:2]).tolist() == torch.signbit(expected[:2]).tolist()


def test_fast_native_dropout_training_backward_and_saved_mask(mojo_gpu):
    from torch_mojo_backend.mojo_device.mojo_device_aten_ops import EAGER_CALL_COUNTERS

    generator = torch.Generator().manual_seed(20260718)
    input = torch.randn(3, 17, generator=generator).to(mojo_gpu).requires_grad_()
    grad_output = torch.randn(3, 17, generator=generator)
    backward_counter = EAGER_CALL_COUNTERS["aten::native_dropout_backward"]
    calls_before = backward_counter.call_count
    output, mask = torch.ops.aten.native_dropout.default(input, 0.2, True)
    assert type(output.grad_fn).__name__ == "NativeDropoutBackward0"
    assert backward_counter.call_count == calls_before
    output.backward(grad_output.to(mojo_gpu))
    assert backward_counter.call_count == calls_before + 1

    assert not mask.requires_grad
    assert input.grad is not None
    torch.testing.assert_close(
        input.grad.cpu(), grad_output * mask.cpu() * 1.25, atol=1e-6, rtol=1e-6
    )

    mutated_input = torch.randn(3, 17).to(mojo_gpu).requires_grad_()
    mutated_output, mutated_mask = torch.ops.aten.native_dropout.default(
        mutated_input, 0.2, True
    )
    mutated_mask.fill_(False)
    with pytest.raises(RuntimeError, match="modified by an inplace operation"):
        mutated_output.backward(torch.ones(3, 17).to(mojo_gpu))


@pytest.mark.parametrize(("train", "scale"), [(True, 1.25), (False, 1.0), (None, 1.0)])
def test_fast_native_dropout_autograd_optional_train_scale(mojo_gpu, train, scale):
    input = torch.randn(3, 17).to(mojo_gpu).requires_grad_()
    grad_output = torch.randn(3, 17)
    output, mask = torch.ops.aten.native_dropout.default(input, 0.2, train)
    output.backward(grad_output.to(mojo_gpu))

    assert input.grad is not None
    torch.testing.assert_close(
        input.grad.cpu(), grad_output * mask.cpu() * scale, atol=1e-6, rtol=1e-6
    )


def test_fast_native_dropout_backward_allows_mutated_output_and_input(mojo_gpu):
    input = torch.randn(3, 17).to(mojo_gpu).requires_grad_()
    grad_output = torch.randn(3, 17)
    output, mask = torch.ops.aten.native_dropout.default(input, 0.2, True)

    with torch.no_grad():
        output.add_(torch.ones_like(output))
        input.add_(torch.ones_like(input))
    output.backward(grad_output.to(mojo_gpu))

    assert input.grad is not None
    torch.testing.assert_close(
        input.grad.cpu(), grad_output * mask.cpu() * 1.25, atol=1e-6, rtol=1e-6
    )


def test_fast_native_dropout_double_backward(mojo_gpu):
    input = torch.randn(3, 17).to(mojo_gpu).requires_grad_()
    grad_output = torch.randn(3, 17).to(mojo_gpu).requires_grad_()
    second_seed = torch.randn(3, 17).to(mojo_gpu)
    output, mask = torch.ops.aten.native_dropout.default(input, 0.2, True)

    (grad_input,) = torch.autograd.grad(
        output, input, grad_outputs=grad_output, create_graph=True
    )
    (second_grad_output,) = torch.autograd.grad(
        grad_input, grad_output, grad_outputs=second_seed
    )

    assert grad_input.requires_grad
    torch.testing.assert_close(
        second_grad_output.cpu(), second_seed.cpu() * mask.cpu() * 1.25
    )


def test_fast_native_dropout_saved_tensor_hooks(mojo_gpu):
    input = torch.randn(3, 17).to(mojo_gpu).requires_grad_()
    grad_output = torch.randn(3, 17)
    hook_calls = []

    def pack(tensor):
        hook_calls.append(("pack", tensor.device.type, tuple(tensor.shape)))
        return tensor.cpu()

    def unpack(tensor):
        hook_calls.append(("unpack", tensor.device.type, tuple(tensor.shape)))
        return tensor.to(mojo_gpu)

    with torch.autograd.graph.saved_tensors_hooks(pack, unpack):
        output, mask = torch.ops.aten.native_dropout.default(input, 0.2, True)
        output.backward(grad_output.to(mojo_gpu))

    shape = (3, 17)
    assert hook_calls == [("pack", "mojo", shape), ("unpack", "cpu", shape)]
    torch.testing.assert_close(input.grad.cpu(), grad_output * mask.cpu() * 1.25)


def test_fast_nn_dropout_training_backward(mojo_gpu):
    input = torch.randn(2, 16, 384).to(mojo_gpu).requires_grad_()
    output = torch.nn.Dropout(0.2)(input)
    output.sum().backward()

    assert input.grad is not None
    assert torch.isfinite(output.cpu()).all()
    assert torch.isfinite(input.grad.cpu()).all()


UNIFORM_DTYPES = [torch.float32, torch.bfloat16, torch.float16, torch.float64]


@pytest.mark.parametrize("dtype", UNIFORM_DTYPES)
@pytest.mark.parametrize(("low", "high"), [(0.0, 1.0), (-100.0, 100.0), (1.0, 2.0)])
def test_fast_uniform_bounds_and_distribution(mojo_device, dtype, low, high):
    """Half-open [from, to) on a large sample, plus the first two moments."""
    torch.mojo.manual_seed_all(20260814)
    drawn = torch.empty(200_000, dtype=dtype, device=mojo_device).uniform_(low, high)
    host = drawn.cpu().double()

    assert host.min().item() >= low
    assert host.max().item() < high
    span = high - low
    assert abs(host.mean().item() - (low + high) / 2) < 0.01 * span
    assert abs(host.var().item() - span * span / 12) < 0.02 * span * span


@pytest.mark.parametrize("dtype", [torch.bfloat16, torch.float16])
def test_fast_uniform_narrow_dtype_never_reaches_to(mojo_device, dtype):
    """The endpoint fold, which only a narrow output dtype exercises.

    A draw above 1 - 2**-9 rounds to exactly 1.0 in bfloat16 (one in ~512;
    measured 407 of 200k with torch's own generator), so without the fold this
    sample would contain values equal to `to`. Every folded draw lands on
    `from` instead, and `from` is otherwise unreachable here: only an exact
    zero word produces it, with probability 2**-24 per element.
    """
    torch.mojo.manual_seed_all(20260814)
    host = (
        torch.empty(200_000, dtype=dtype, device=mojo_device)
        .uniform_(0.0, 1.0)
        .cpu()
        .double()
    )

    assert host.max().item() < 1.0
    assert int((host == 0.0).sum()) > 20


@pytest.mark.parametrize("dtype", UNIFORM_DTYPES)
def test_fast_uniform_from_equals_to_is_constant(mojo_gpu, dtype):
    drawn = torch.empty(37, dtype=dtype, device=mojo_gpu).uniform_(2.5, 2.5)
    assert torch.equal(drawn.cpu(), torch.full((37,), 2.5, dtype=dtype))


@pytest.mark.parametrize(
    ("numel", "dtype", "counters"),
    [
        (8, torch.float32, 2),
        # Not a multiple of the group: the ragged group consumes its whole
        # counter, so the next draw starts past it and cannot overlap.
        (7, torch.float32, 2),
        (1, torch.bfloat16, 1),
        # float64 spends two of the four words per element.
        (8, torch.float64, 4),
        (7, torch.float64, 4),
    ],
)
def test_fast_uniform_reserves_one_counter_per_philox_group(
    mojo_gpu, monkeypatch, numel, dtype, counters
):
    """The exact interval reserved, and the full-width words handed to Mojo."""
    calls = []
    _replace_defined_native_calls(
        monkeypatch,
        {("random_ops.mojo", "UniformFill"): lambda *args: calls.append(args)},
    )
    drawn = torch.empty(numel, dtype=dtype, device=mojo_gpu)
    seed = (1 << 63) + 0x1234_5678
    counter = (1 << 63) + 0x9ABC_DEF0
    torch.mojo.set_rng_state(_philox_rng_state(seed, counter), drawn.device)

    drawn.uniform_(-1.0, 3.0)

    assert len(calls) == 1
    args = calls[0]
    assert args[1:4] == (-1.0, 3.0, numel)
    assert args[5:9] == (
        seed & 0xFFFF_FFFF,
        seed >> 32,
        counter & 0xFFFF_FFFF,
        counter >> 32,
    )
    assert _decode_philox_rng_state(torch.mojo.get_rng_state(drawn.device)) == (
        seed,
        counter + counters,
    )


def test_fast_uniform_adjacent_draws_are_independent(mojo_gpu):
    """Two draws in a row must not share a single value of the stream.

    A reservation that rounded the ragged group down, or one that forgot to
    advance at all, shows up here and nowhere else: exact float32 collisions
    between two 8192-element draws are otherwise ~1 in 2**24 per element.
    """
    torch.mojo.manual_seed_all(20260814)
    first = torch.empty(8192, device=mojo_gpu).uniform_().cpu()
    second = torch.empty(8192, device=mojo_gpu).uniform_().cpu()

    assert int((first == second).sum()) == 0
    centered_first = first - first.mean()
    centered_second = second - second.mean()
    correlation = (centered_first * centered_second).sum() / (
        centered_first.norm() * centered_second.norm()
    )
    assert abs(correlation.item()) < 0.05


def test_fast_uniform_non_multiple_of_four_draws_do_not_overlap(mojo_gpu):
    """A ragged draw leaves the stream exactly two counters further on."""
    torch.mojo.manual_seed_all(20260814)
    state = torch.mojo.get_rng_state(mojo_gpu)
    seed, counter = _decode_philox_rng_state(state)

    first = torch.empty(7, device=mojo_gpu).uniform_().cpu()
    second = torch.empty(7, device=mojo_gpu).uniform_().cpu()

    torch.mojo.set_rng_state(_philox_rng_state(seed, counter + 2), mojo_gpu)
    replayed_second = torch.empty(7, device=mojo_gpu).uniform_().cpu()
    assert torch.equal(second, replayed_second)
    assert not torch.equal(first, second)


def test_fast_uniform_is_reproducible_under_manual_seed(mojo_device):
    torch.mojo.manual_seed_all(4242)
    first = torch.empty(1023, device=mojo_device).uniform_(-2.0, 5.0).cpu()
    torch.mojo.manual_seed_all(4242)
    second = torch.empty(1023, device=mojo_device).uniform_(-2.0, 5.0).cpu()

    assert torch.equal(first, second)


def test_fast_uniform_rng_state_round_trips(mojo_device):
    torch.mojo.manual_seed_all((1 << 63) + 0x54321)
    initial = torch.mojo.get_rng_state(mojo_device)
    first = torch.empty(1025, device=mojo_device).uniform_().cpu()
    advanced = torch.mojo.get_rng_state(mojo_device)
    assert not torch.equal(initial, advanced)

    torch.mojo.set_rng_state(initial, mojo_device)
    replayed = torch.empty(1025, device=mojo_device).uniform_().cpu()

    assert torch.equal(first, replayed)
    torch.testing.assert_close(torch.mojo.get_rng_state(mojo_device), advanced)


def test_fast_uniform_wide_and_scalar_store_paths_agree(mojo_gpu):
    """An offset base declines the vector store; the values must not change.

    `x[1:]` starts one float32 in, so the 16-byte store is unavailable and the
    scalar kernel runs instead. Both kernels index the stream the same way, so
    the same reservation has to produce the same values.
    """
    torch.mojo.manual_seed_all(20260814)
    state = torch.mojo.get_rng_state(mojo_gpu)
    aligned = torch.zeros(8, device=mojo_gpu)
    aligned.uniform_(-1.0, 1.0)

    torch.mojo.set_rng_state(state, mojo_gpu)
    storage = torch.zeros(9, device=mojo_gpu)
    offset_view = storage[1:]
    assert offset_view._ptr % 16 != 0
    offset_view.uniform_(-1.0, 1.0)

    host = storage.cpu()
    assert torch.equal(host[1:], aligned.cpu())
    assert host[0].item() == 0.0


def test_fast_uniform_strided_destination_matches_contiguous(mojo_gpu):
    """A view's values depend on its logical index, never on its layout."""
    torch.mojo.manual_seed_all(20260814)
    state = torch.mojo.get_rng_state(mojo_gpu)
    contiguous = torch.empty(4, 4, device=mojo_gpu).uniform_(10.0, 11.0)

    torch.mojo.set_rng_state(state, mojo_gpu)
    storage = torch.zeros(4, 8, device=mojo_gpu)
    view = storage[:, ::2]
    assert not view.is_contiguous()
    view.uniform_(10.0, 11.0)

    host = storage.cpu()
    assert torch.equal(host[:, ::2], contiguous.cpu())
    assert host[:, 1::2].abs().max().item() == 0.0


def test_fast_uniform_empty_does_not_advance_rng(mojo_gpu):
    drawn = torch.empty(0, 5, device=mojo_gpu)
    torch.mojo.manual_seed_all(20260814)
    before = torch.mojo.get_rng_state(drawn.device)

    assert drawn.uniform_() is drawn
    torch.testing.assert_close(torch.mojo.get_rng_state(drawn.device), before)


@pytest.mark.parametrize(
    ("dtype", "low", "high", "message"),
    [
        (torch.float32, 3.0, -1.0, r"\[from, to\) range, but found from=3 > to=-1"),
        (torch.float32, 3, -1, r"\[from, to\) range, but found from=3 > to=-1"),
        (torch.float16, -1e9, 1.0, "from is out of bounds for float16"),
        (torch.float16, 0.0, 1e9, "to is out of bounds for float16"),
        (torch.float32, -3e38, 3e38, "which result in to-from to exceed the limit"),
    ],
)
def test_fast_uniform_rejects_bad_bounds(mojo_gpu, dtype, low, high, message):
    """ATen's own checks, including on an empty tensor (ATen checks first)."""
    torch.mojo.manual_seed_all(20260814)
    before = torch.mojo.get_rng_state(mojo_gpu)
    for numel in (10, 0):
        drawn = torch.empty(numel, dtype=dtype, device=mojo_gpu)
        with pytest.raises(RuntimeError, match=message):
            drawn.uniform_(low, high)
    torch.testing.assert_close(torch.mojo.get_rng_state(mojo_gpu), before)


def test_fast_uniform_refuses_an_explicit_generator(mojo_gpu):
    torch.mojo.manual_seed_all(20260814)
    before = torch.mojo.get_rng_state(mojo_gpu)
    drawn = torch.empty(4, device=mojo_gpu)

    with pytest.raises(NotImplementedError, match="explicit generator"):
        drawn.uniform_(0.0, 1.0, generator=torch.Generator())
    torch.testing.assert_close(torch.mojo.get_rng_state(mojo_gpu), before)


def test_fast_uniform_declines_integer_dtypes(mojo_gpu):
    drawn = torch.zeros(4, dtype=torch.int64, device=mojo_gpu)
    with pytest.raises(NotImplementedError, match="aten::uniform_"):
        drawn.uniform_(0.0, 1.0)


def test_fast_uniform_initializes_a_module_like_nn_init(mojo_gpu):
    """The way a user actually reaches this op."""
    layer = torch.nn.Linear(64, 32).to(mojo_gpu)
    torch.nn.init.uniform_(layer.weight, -0.125, 0.125)
    host = layer.weight.detach().cpu()

    assert host.min().item() >= -0.125
    assert host.max().item() < 0.125
    assert host.std().item() > 0.05


def test_fast_uniform_carries_torch_rand_on_device(mojo_gpu):
    """`torch.rand` needs no registration of its own: ATen's own
    CompositeExplicitAutograd `rand` is `empty` plus `uniform_`, so it lands on
    this kernel. `torch.manual_seed` reaches the device generator through
    `torch.mojo.manual_seed_all`, which is what makes it reproducible."""
    torch.manual_seed(20260814)
    first = torch.rand(1000, device=mojo_gpu)
    like = torch.rand_like(first)
    torch.manual_seed(20260814)
    replayed = torch.rand(1000, device=mojo_gpu)

    assert first.device.type == "mojo"
    assert torch.equal(first.cpu(), replayed.cpu())
    assert not torch.equal(first.cpu(), like.cpu())
    host = first.cpu()
    assert host.min().item() >= 0.0
    assert host.max().item() < 1.0


# ---------------------------------------------------------------------------
# sort / topk. The kernel picks one of three launch routes from the row
# length and k, so the interesting cases are the route boundaries -- which is
# also why every reference below is a CPU *stable* sort: the kernel's order is
# total (ties broken by original index), so it must reproduce the stable
# answer exactly, not merely a correct one.
# ---------------------------------------------------------------------------

# Elements one block sorts in shared memory, mirrored from the kernel; the
# route boundaries sit at and just past it.
_SORT_TILE = 4096
_SORT_ROUTE_SIZES = (1, 2, 33, 255, 4095, 4096, 4097, 8192, 8193, 12000, 50257)


def _assert_same_ordering(actual: torch.Tensor, expected: torch.Tensor) -> None:
    """Bit-exact equality, with NaN counted equal to NaN.

    `torch.equal` has no equal_nan, and a sort that moves NaN correctly still
    has to be compared for NaN in the right *place*.
    """
    torch.testing.assert_close(actual, expected, rtol=0, atol=0, equal_nan=True)


def _sort_probe(rows: int, columns: int, seed: int = 7) -> torch.Tensor:
    """Values with deliberate ties, both zeros, and a NaN in longer rows."""
    generator = torch.Generator().manual_seed(seed)
    host = torch.randn(rows, columns, generator=generator)
    if columns >= 16:
        host[:, ::7] = 0.0
        host[:, ::11] = -0.0
        host[:, ::13] = 1.5
        host[:, 3] = float("nan")
    return host


@pytest.mark.parametrize("descending", [False, True])
@pytest.mark.parametrize("columns", _SORT_ROUTE_SIZES)
def test_fast_sort_matches_stable_cpu_on_every_route(mojo_gpu, columns, descending):
    host = _sort_probe(2, columns)
    expected = torch.sort(host, dim=-1, descending=descending, stable=True)
    actual = torch.sort(host.to(mojo_gpu), dim=-1, descending=descending, stable=True)
    _assert_same_ordering(actual.values.cpu(), expected.values)
    _assert_same_ordering(actual.indices.cpu(), expected.indices)


@pytest.mark.parametrize(
    ("columns", "k"),
    [
        # k small enough for the tournament route (one tile per row's worth of
        # candidates), then the first k that no longer fits and has to fall
        # back to the full sort, then a k spanning the whole row.
        (50257, 50),
        (50257, 315),
        (50257, 316),
        (50257, 2048),
        (8193, 4096),
        (4096, 4096),
        (33, 33),
    ],
)
@pytest.mark.parametrize("largest", [True, False])
def test_fast_topk_matches_stable_cpu_across_the_tournament_boundary(
    mojo_gpu, columns, k, largest
):
    """Values against ATen, indices against the stable sort's prefix.

    ATen leaves topk's tie order unspecified (CPU and CUDA disagree with each
    other on this data), so only the values are comparable to `torch.topk`.
    The indices are pinned to the stronger contract this kernel actually
    offers -- the first k of the stable sort -- which is what makes a tie
    reproducible here at all.
    """
    host = _sort_probe(2, columns, seed=11)
    expected = torch.topk(host, k, dim=-1, largest=largest)
    actual = torch.topk(host.to(mojo_gpu), k, dim=-1, largest=largest)
    _assert_same_ordering(actual.values.cpu(), expected.values)

    stable = torch.sort(host, dim=-1, descending=largest, stable=True)
    _assert_same_ordering(actual.indices.cpu(), stable.indices[..., :k])


def test_fast_sort_orders_nan_and_signed_zero_like_aten(mojo_gpu):
    """The two bit patterns the monotone key gets wrong on its own.

    A negative NaN's bits sit below -inf and -0.0's below +0.0, but ATen
    orders every NaN above every number and treats the two zeros as equal, so
    the kernel overrides both before comparing.
    """
    host = torch.tensor(
        [
            [
                0.0,
                -0.0,
                float("nan"),
                -float("nan"),
                float("inf"),
                -float("inf"),
                1.0,
                -1.0,
            ]
        ]
    )
    for descending in (False, True):
        expected = torch.sort(host, dim=-1, descending=descending, stable=True)
        actual = torch.sort(
            host.to(mojo_gpu), dim=-1, descending=descending, stable=True
        )
        _assert_same_ordering(actual.values.cpu(), expected.values)
        _assert_same_ordering(actual.indices.cpu(), expected.indices)


@pytest.mark.parametrize(
    "dtype",
    [
        torch.float32,
        torch.float64,
        torch.bfloat16,
        torch.float16,
        torch.int64,
        torch.int32,
        torch.int16,
        torch.int8,
        torch.uint8,
        torch.bool,
    ],
)
def test_fast_sort_covers_its_declared_dtypes(mojo_gpu, dtype):
    """Every kernel dtype, on a row long enough to cross tiles.

    The 64-bit dtypes matter most: their key is twice as wide, which halves
    the shared-memory tile and so moves every route boundary.
    """
    host = torch.arange(2 * 6000, dtype=torch.int64)
    host = ((host * 37 + 11) % 251).reshape(2, 6000)
    if dtype is torch.bool:
        host = host % 2 == 0
    else:
        host = (host - 125).to(dtype)
    expected = torch.sort(host, dim=-1, stable=True)
    actual = torch.sort(host.to(mojo_gpu), dim=-1, stable=True)
    assert torch.equal(actual.values.cpu(), expected.values)
    assert torch.equal(actual.indices.cpu(), expected.indices)


def test_fast_sort_and_topk_materialize_non_contiguous_and_non_last_dims(mojo_gpu):
    """A dim other than the last is transposed to last and back, and a view
    of the result must not read the wrong elements."""
    host = _sort_probe(9, 7, seed=3)
    device = host.to(mojo_gpu)
    for dim in (0, 1, -1, -2):
        expected = torch.sort(host, dim=dim, stable=True)
        actual = torch.sort(device, dim=dim, stable=True)
        _assert_same_ordering(actual.values.cpu(), expected.values)
        _assert_same_ordering(actual.indices.cpu(), expected.indices)

    transposed_expected = torch.topk(host.t(), 3, dim=-1)
    transposed_actual = torch.topk(device.t(), 3, dim=-1)
    _assert_same_ordering(transposed_actual.values.cpu(), transposed_expected.values)
    sliced_expected = torch.sort(host[:, 1::2], dim=-1, stable=True)
    sliced_actual = torch.sort(device[:, 1::2], dim=-1, stable=True)
    _assert_same_ordering(sliced_actual.values.cpu(), sliced_expected.values)


def test_fast_sort_and_topk_reject_out_of_range_arguments(mojo_gpu):
    tensor = torch.randn(4, 8).to(mojo_gpu)
    with pytest.raises(RuntimeError, match="selected index k out of range"):
        torch.topk(tensor, 9, dim=-1)
    with pytest.raises(IndexError, match="Dimension out of range"):
        torch.sort(tensor, dim=2)


# ---------------------------------------------------------------------------
# multinomial.  A sampler, so it is tested differently from a kernel: the
# mojo device's own draws are never compared to CPU's (different RNG
# streams -- that's expected, not a bug), only (1) determinism under a fixed
# seed, (2) the sampled DISTRIBUTION against the true weights, and (3) the
# ATen edge semantics that don't depend on which values got drawn.
#
# With replacement is cumsum + Philox uniforms (same generator `uniform_`
# already uses) + searchsorted; without replacement is Gumbel-top-k over
# #416's topk kernel. See `fast_aten_multinomial` in aten_fast.py for why
# each is a correct-DISTRIBUTION sampler without being ATen bit-identical.
# ---------------------------------------------------------------------------


def test_fast_multinomial_is_reproducible_under_manual_seed(mojo_device):
    weights = torch.tensor([0.1, 0.2, 0.3, 0.4], device=mojo_device)
    torch.mojo.manual_seed_all(20260819)
    first = torch.multinomial(weights, 4096, replacement=True).cpu()
    torch.mojo.manual_seed_all(20260819)
    second = torch.multinomial(weights, 4096, replacement=True).cpu()
    assert torch.equal(first, second)

    torch.mojo.manual_seed_all(20260819)
    first_wo = torch.multinomial(weights, 4, replacement=False).cpu()
    torch.mojo.manual_seed_all(20260819)
    second_wo = torch.multinomial(weights, 4, replacement=False).cpu()
    assert torch.equal(first_wo, second_wo)


def test_fast_multinomial_torch_manual_seed_also_reproduces(mojo_device):
    """`torch.manual_seed` (not just `torch.mojo.manual_seed_all`) reseeds
    the mojo device too, same as it already does for `uniform_`/`rand`."""
    weights = torch.tensor([1.0, 1.0, 1.0], device=mojo_device)
    torch.manual_seed(4242)
    first = torch.multinomial(weights, 1024, replacement=True).cpu()
    torch.manual_seed(4242)
    second = torch.multinomial(weights, 1024, replacement=True).cpu()
    assert torch.equal(first, second)


def test_fast_multinomial_with_replacement_matches_weight_distribution(mojo_device):
    """Frequency of each category over many draws, against the true
    (normalized) weights -- same moment-check style as
    `test_fast_uniform_bounds_and_distribution`, not a bit-parity check."""
    torch.mojo.manual_seed_all(20260819)
    weights = torch.tensor([1.0, 2.0, 3.0, 4.0], device=mojo_device)
    n = 200_000
    drawn = torch.multinomial(weights, n, replacement=True).cpu()
    observed = drawn.bincount(minlength=4).double() / n
    expected = weights.cpu().double() / weights.cpu().double().sum()
    assert (observed - expected).abs().max().item() < 0.01


def test_fast_multinomial_unnormalized_weights_accepted(mojo_device):
    """Weights need not sum to 1 -- ATen normalizes internally."""
    torch.mojo.manual_seed_all(20260819)
    weights = torch.tensor([10.0, 20.0, 30.0, 40.0], device=mojo_device)
    n = 100_000
    drawn = torch.multinomial(weights, n, replacement=True).cpu()
    observed = drawn.bincount(minlength=4).double() / n
    expected = torch.tensor([0.1, 0.2, 0.3, 0.4], dtype=torch.float64)
    assert (observed - expected).abs().max().item() < 0.01


@pytest.mark.parametrize("replacement", [True, False])
def test_fast_multinomial_zero_probability_categories_never_drawn(
    mojo_device, replacement
):
    torch.mojo.manual_seed_all(20260819)
    weights = torch.tensor([0.0, 1.0, 0.0, 1.0, 0.0], device=mojo_device)
    num_samples = 2 if replacement is False else 4000
    drawn = torch.multinomial(weights, num_samples, replacement=replacement).cpu()
    assert set(drawn.unique().tolist()) <= {1, 3}


def test_fast_multinomial_single_nonzero_always_sampled(mojo_device):
    torch.mojo.manual_seed_all(20260819)
    weights = torch.tensor([0.0, 0.0, 5.0, 0.0], device=mojo_device)
    drawn = torch.multinomial(weights, 500, replacement=True).cpu()
    assert bool((drawn == 2).all())


def test_fast_multinomial_without_replacement_never_repeats(mojo_device):
    torch.mojo.manual_seed_all(20260819)
    weights = torch.rand(64, device=mojo_device) + 0.01
    for _ in range(20):
        drawn = torch.multinomial(weights, 64, replacement=False).cpu()
        assert len(set(drawn.tolist())) == 64


def test_fast_multinomial_without_replacement_first_pick_matches_weight_distribution(
    mojo_device,
):
    """The Gumbel-top-k trick's FIRST selected index alone has exactly the
    plain categorical distribution -- a testable, well-known property of the
    algorithm, distinct from (and complementary to) the no-repeat check."""
    torch.mojo.manual_seed_all(20260819)
    weights = torch.tensor([1.0, 2.0, 3.0, 4.0], device=mojo_device)
    n = 50_000
    first_picks = torch.stack(
        [torch.multinomial(weights, 2, replacement=False)[0] for _ in range(n)]
    ).cpu()
    observed = first_picks.bincount(minlength=4).double() / n
    expected = weights.cpu().double() / weights.cpu().double().sum()
    assert (observed - expected).abs().max().item() < 0.02


def test_fast_multinomial_num_samples_between_nonzero_and_total_does_not_raise(
    mojo_device,
):
    """ATen's own multinomial(replacement=False) does NOT require enough
    positive-probability categories to cover num_samples -- it only raises
    once num_samples exceeds the TOTAL category count (verified against CPU
    torch: `multinomial([1,0,0,0], num_samples=4, replacement=False)`
    succeeds and includes zero-weight categories). This backend matches that
    permissiveness (Gumbel-top-k naturally ranks -inf-perturbed zero-weight
    categories below every positive one, deterministically) without
    promising the same fallback ORDER CPU's own implementation picks."""
    torch.mojo.manual_seed_all(20260819)
    weights = torch.tensor([1.0, 0.0, 0.0, 0.0], device=mojo_device)
    drawn = torch.multinomial(weights, 4, replacement=False).cpu()
    assert sorted(drawn.tolist()) == [0, 1, 2, 3]
    assert drawn[0].item() == 0  # the only positive-weight category wins first


@pytest.mark.parametrize(
    "dtype", [torch.float32, torch.float64, torch.bfloat16, torch.float16]
)
@pytest.mark.parametrize("replacement", [True, False])
def test_fast_multinomial_covers_its_declared_dtypes(mojo_gpu, dtype, replacement):
    """float64 weights are a real declared dtype (`multinomial only supports
    floating-point dtypes` on CPU accepts it too), but neither the
    validation reductions nor `aten::cumsum` have a float64 kernel --
    `_multinomial_validate_and_prepare` downcasts via a host round-trip
    first, same precedent as `mojo_device_normal_`."""
    torch.mojo.manual_seed_all(20260819)
    weights = torch.tensor([1.0, 2.0, 3.0, 4.0], dtype=dtype, device=mojo_gpu)
    num_samples = 3 if replacement is False else 16
    drawn = torch.multinomial(weights, num_samples, replacement=replacement)
    assert drawn.dtype == torch.int64
    assert drawn.min().item() >= 0
    assert drawn.max().item() < 4


def test_fast_multinomial_2d_batched_rows_are_independent(mojo_device):
    torch.mojo.manual_seed_all(20260819)
    weights = torch.tensor(
        [[1.0, 0.0, 0.0], [0.0, 1.0, 0.0], [0.0, 0.0, 1.0]], device=mojo_device
    )
    drawn = torch.multinomial(weights, 200, replacement=True).cpu()
    assert bool((drawn[0] == 0).all())
    assert bool((drawn[1] == 1).all())
    assert bool((drawn[2] == 2).all())


def test_fast_multinomial_without_replacement_full_row_is_a_permutation(mojo_device):
    torch.mojo.manual_seed_all(20260819)
    weights = torch.tensor(
        [[1.0, 1.0, 1.0, 1.0], [1.0, 1.0, 1.0, 1.0]], device=mojo_device
    )
    drawn = torch.multinomial(weights, 4, replacement=False).cpu()
    for row in drawn:
        assert sorted(row.tolist()) == [0, 1, 2, 3]


def test_fast_multinomial_decode_shape_realistic(mojo_device):
    """The `generate()` regime: batch 1, a GPT-2-sized vocabulary,
    num_samples=1, with replacement."""
    torch.mojo.manual_seed_all(20260819)
    weights = torch.rand(1, 50304, device=mojo_device)
    drawn = torch.multinomial(weights, 1, replacement=True)
    assert drawn.shape == (1, 1)
    assert drawn.dtype == torch.int64
    assert 0 <= drawn.item() < 50304


def test_fast_multinomial_rng_state_advances_and_round_trips(mojo_device):
    torch.mojo.manual_seed_all((1 << 63) + 0x777)
    initial = torch.mojo.get_rng_state(mojo_device)
    weights = torch.tensor([1.0, 1.0, 1.0, 1.0], device=mojo_device)
    first = torch.multinomial(weights, 8, replacement=True).cpu()
    advanced = torch.mojo.get_rng_state(mojo_device)
    assert not torch.equal(initial, advanced)

    torch.mojo.set_rng_state(initial, mojo_device)
    replayed = torch.multinomial(weights, 8, replacement=True).cpu()
    assert torch.equal(first, replayed)
    assert torch.equal(torch.mojo.get_rng_state(mojo_device), advanced)


# ---------------------------------------------------------------------------
# median.dim / kthvalue eager coverage: both reuse the sort/topk kernel
# family above, so this is a dtype/shape sweep on top of that kernel's own
# extensive route-boundary tests, not a new set of route boundaries.
# ---------------------------------------------------------------------------


@pytest.mark.parametrize(
    "dtype",
    [
        torch.float32,
        torch.float64,
        torch.bfloat16,
        torch.float16,
        torch.int64,
        torch.int32,
        torch.int16,
        torch.int8,
        torch.uint8,
    ],
)
def test_fast_median_and_kthvalue_cover_their_declared_dtypes(mojo_gpu, dtype):
    host = torch.arange(2 * 37, dtype=torch.int64)
    host = ((host * 37 + 11) % 251).reshape(2, 37)
    host = (host - 125).to(dtype)
    expected_median = torch.median(host, dim=-1)
    expected_kth = torch.kthvalue(host, 5, dim=-1)

    device = host.to(mojo_gpu)
    got_median = torch.median(device, dim=-1)
    got_kth = torch.kthvalue(device, 5, dim=-1)
    assert torch.equal(got_median.values.cpu(), expected_median.values)
    assert torch.equal(got_median.indices.cpu(), expected_median.indices)
    assert torch.equal(got_kth.values.cpu(), expected_kth.values)


def test_fast_median_declines_bool(mojo_gpu):
    """ATen itself doesn't implement median for bool ("median_out" not
    implemented for 'Bool') -- declining here is the matching behavior, not
    a gap."""
    tensor = (torch.arange(8) % 2 == 0).reshape(2, 4).to(mojo_gpu)
    with pytest.raises(NotImplementedError):
        torch.median(tensor, dim=-1)


def test_fast_lerp_scalar_out_and_inplace_alias(mojo_gpu):
    """AdamW's lerp out path preserves ATen's FP32 branch and aliases."""
    # This Python double rounds to exactly 0.5f. ATen narrows before choosing
    # its stable formula, and these values distinguish the two branches.
    weight = 0.5 - 2**-30
    start = torch.tensor(
        [[-1.0687099695205688, -2.0, 3.0], [4.0, -5.0, 6.0]], dtype=torch.float32
    )
    end = torch.tensor([[2.028475284576416, 8.0, -3.0]], dtype=torch.float32)
    expected = torch.lerp(start, end, weight)
    device_start = start.to(mojo_gpu)
    device_end = end.to(mojo_gpu)

    out = torch.empty_like(device_start)
    out_holder, out_ptr = out._holder, out._ptr
    returned = torch.lerp(device_start, device_end, weight, out=out)
    assert returned is out
    assert out._holder is out_holder
    assert out._ptr == out_ptr
    torch.testing.assert_close(out.cpu(), expected, rtol=0, atol=0)

    alias = device_start.view(3, 2)
    start_holder, start_ptr = device_start._holder, device_start._ptr
    returned = device_start.lerp_(device_end, weight)
    assert returned is device_start
    assert device_start._holder is start_holder
    assert device_start._ptr == start_ptr
    torch.testing.assert_close(alias.cpu().view_as(expected), expected, rtol=0, atol=0)
    torch.testing.assert_close(device_end.cpu(), end, rtol=0, atol=0)


def test_fast_l2_norm_out_and_mul_inplace_alias(mojo_gpu):
    """Gradient clipping keeps its norm on-device and mutates aliased grads."""
    contiguous = torch.linspace(-3.0, 4.0, 35).reshape(5, 7)
    input = contiguous.t()
    assert not input.is_contiguous()
    device_input = input.to(mojo_gpu)
    expected = torch.linalg.vector_norm(input)

    out = torch.empty((), dtype=torch.float32, device=mojo_gpu)
    out_holder, out_ptr = out._holder, out._ptr
    returned = torch.linalg.vector_norm(device_input, out=out)
    assert returned is out
    assert out._holder is out_holder
    assert out._ptr == out_ptr
    torch.testing.assert_close(out.cpu(), expected)

    empty = torch.empty((0, 7), dtype=torch.float32).to(mojo_gpu)
    torch.testing.assert_close(
        torch.linalg.vector_norm(empty).cpu(), torch.tensor(0.0), rtol=0, atol=0
    )

    base = torch.arange(12, dtype=torch.float32).to(mojo_gpu)
    view = base[::2]
    observer = base.view(3, 4)
    base_holder, base_ptr = base._holder, base._ptr
    coefficient = torch.tensor(0.25, dtype=torch.float32).to(mojo_gpu)
    view.mul_(coefficient)
    expected_base = torch.arange(12, dtype=torch.float32)
    expected_base[::2] *= 0.25
    assert base._holder is base_holder
    assert base._ptr == base_ptr
    torch.testing.assert_close(observer.cpu().reshape(-1), expected_base)


def _eager_registration_snapshot(op_name):
    from torch_mojo_backend.mojo_device.mojo_device_aten_ops import EAGER_CALL_COUNTERS

    assert op_name in EAGER_CALL_COUNTERS, f"missing eager registration for {op_name}"
    counter = EAGER_CALL_COUNTERS[op_name]
    return counter, counter.call_count


@pytest.mark.parametrize("dtype", [None, torch.float32], ids=["default", "float32"])
def test_fast_foreach_norm_l2_order_empty_nonfinite_and_chunk_boundary(mojo_gpu, dtype):
    """The fused route returns one independently owned scalar per input.

    Sixty-five inputs cross the descriptor cap used by the multi-tensor
    kernel.  Empty and non-finite tensors also pin the L2-norm semantics.
    """
    counter, calls_before = _eager_registration_snapshot("aten::_foreach_norm.Scalar")
    host_inputs = [
        torch.empty(0),
        torch.tensor([float("inf"), 1.0]),
        torch.tensor([float("nan"), 2.0]),
    ]
    host_inputs.extend(
        torch.tensor([3.0 * index, -4.0 * index]) for index in range(1, 62)
    )
    host_inputs.append(torch.linspace(-3.0, 4.0, 65_537))
    assert len(host_inputs) == 65
    device_inputs = [tensor.to(mojo_gpu) for tensor in host_inputs]

    actual = torch.ops.aten._foreach_norm.Scalar(device_inputs, 2, dtype=dtype)
    expected = torch.ops.aten._foreach_norm.Scalar(host_inputs, 2, dtype=dtype)

    assert counter.call_count == calls_before + 1
    assert len(actual) == len(device_inputs)
    for actual_scalar, expected_scalar in zip(actual, expected, strict=True):
        assert actual_scalar.device == torch.device(mojo_gpu)
        assert actual_scalar.shape == torch.Size([])
        assert actual_scalar.dtype == torch.float32
        torch.testing.assert_close(
            actual_scalar.cpu(), expected_scalar, equal_nan=True, rtol=2e-5, atol=1e-6
        )
    for index, output in enumerate(actual):
        for other in actual[index + 1 :]:
            assert output._holder is not other._holder
            assert output._ptr != other._ptr


def test_fast_foreach_norm_preserves_strided_fallback(mojo_gpu):
    counter, calls_before = _eager_registration_snapshot("aten::_foreach_norm.Scalar")
    host_bases = [
        torch.arange(24, dtype=torch.float32).reshape(4, 6),
        torch.linspace(-2.0, 3.0, 35).reshape(5, 7),
    ]
    host_inputs = [tensor.t() for tensor in host_bases]
    assert all(not tensor.is_contiguous() for tensor in host_inputs)
    device_inputs = [tensor.to(mojo_gpu).t() for tensor in host_bases]
    assert all(not tensor.is_contiguous() for tensor in device_inputs)

    actual = torch.ops.aten._foreach_norm.Scalar(device_inputs, 2)
    expected = torch.ops.aten._foreach_norm.Scalar(host_inputs, 2)

    assert counter.call_count == calls_before + 1
    for actual_scalar, expected_scalar in zip(actual, expected, strict=True):
        torch.testing.assert_close(actual_scalar.cpu(), expected_scalar)


@pytest.mark.parametrize(
    "dtype", [torch.float32, torch.float16, torch.bfloat16], ids=["f32", "f16", "bf16"]
)
def test_fast_foreach_add_scalar_inplace_chunk_boundary(mojo_gpu, dtype):
    """One launch mutates every input exactly once, across the descriptor cap.

    Sixty-six entries cross ``FOREACH_DESC_CAP`` (64) so the batching loop runs
    twice, and the last input crosses ``FOREACH_CHUNK_ELEMENTS`` (65536) so one
    tensor spans several blocks.
    """
    counter, calls_before = _eager_registration_snapshot("aten::_foreach_add_.Scalar")
    host_inputs = [torch.empty(0, dtype=dtype)] + [
        torch.tensor([float(index), -float(index)], dtype=dtype)
        for index in range(1, 65)
    ]
    host_inputs.append(torch.linspace(-3.0, 4.0, 65_537).to(dtype))
    device_inputs = [tensor.to(mojo_gpu) for tensor in host_inputs]
    allocation_state = [
        (tensor._holder, tensor._ptr, tensor._version) for tensor in device_inputs
    ]

    returned = torch.ops.aten._foreach_add_.Scalar(device_inputs, 1.5)

    assert returned is None
    assert counter.call_count == calls_before + 1
    for actual, expected, (holder, ptr, version) in zip(
        device_inputs, host_inputs, allocation_state, strict=True
    ):
        assert actual._holder is holder
        assert actual._ptr == ptr
        assert actual._version == version + 1
        # The kernel widens to FP32, adds, and narrows -- exactly what the
        # per-tensor `add_.Scalar` path it replaces does.
        torch.testing.assert_close(
            actual.cpu(), ((expected.float() + 1.5).to(dtype)), rtol=0, atol=0
        )


def test_fast_foreach_add_scalar_all_empty_and_integer_scalar(mojo_gpu):
    """Zero-work lists still record the mutation; an int scalar is accepted."""
    counter, calls_before = _eager_registration_snapshot("aten::_foreach_add_.Scalar")
    empties = [torch.empty(0, dtype=torch.float32).to(mojo_gpu) for _ in range(65)]
    versions = [tensor._version for tensor in empties]

    assert torch.ops.aten._foreach_add_.Scalar(empties, 1) is None

    assert counter.call_count == calls_before + 1
    assert [tensor._version for tensor in empties] == [v + 1 for v in versions]

    values = [torch.tensor([1.0, 2.0]).to(mojo_gpu)]
    torch.ops.aten._foreach_add_.Scalar(values, 3)
    torch.testing.assert_close(
        values[0].cpu(), torch.tensor([4.0, 5.0]), rtol=0, atol=0
    )


def test_fast_foreach_add_scalar_falls_back_where_it_must(mojo_gpu):
    """Aliasing, strided, mixed-dtype and integer lists keep ATen's semantics.

    Each of these would be wrong under one grid over the concatenation (a
    duplicate must be added twice, in order) or is simply outside the kernel's
    contract, so each has to reach the CompositeExplicitAutograd fallback and
    still produce the right answer.
    """
    counter, calls_before = _eager_registration_snapshot("aten::_foreach_add_.Scalar")

    duplicate = torch.tensor([2.0, -3.0, 5.0]).to(mojo_gpu)
    version = duplicate._version
    torch.ops.aten._foreach_add_.Scalar([duplicate, duplicate], 1.0)
    assert duplicate._version == version + 2
    torch.testing.assert_close(
        duplicate.cpu(), torch.tensor([4.0, -1.0, 7.0]), rtol=0, atol=0
    )

    strided = torch.arange(12, dtype=torch.float32).reshape(3, 4).to(mojo_gpu).t()
    torch.ops.aten._foreach_add_.Scalar([strided], 0.5)
    torch.testing.assert_close(
        strided.cpu(), torch.arange(12, dtype=torch.float32).reshape(3, 4).t() + 0.5
    )

    mixed = [
        torch.tensor([1.0], dtype=torch.float32).to(mojo_gpu),
        torch.tensor([1.0], dtype=torch.bfloat16).to(mojo_gpu),
    ]
    torch.ops.aten._foreach_add_.Scalar(mixed, 2.0)
    assert mixed[0].cpu().item() == 3.0
    assert mixed[1].cpu().item() == 3.0

    integers = [torch.tensor([1, 2], dtype=torch.int64).to(mojo_gpu)]
    torch.ops.aten._foreach_add_.Scalar(integers, 3)
    torch.testing.assert_close(
        integers[0].cpu(), torch.tensor([4, 5], dtype=torch.int64)
    )

    # Every one of the four went through the registered op.
    assert counter.call_count == calls_before + 4


def test_fast_foreach_add_scalar_matches_adamw_step_counters(mojo_gpu, monkeypatch):
    """The shape AdamW actually asks for: 75 one-element FP32 counters.

    Also the guard against the whole change being a no-op -- the bridge has to
    be entered once per call, not once per tensor.
    """
    counter, calls_before = _eager_registration_snapshot("aten::_foreach_add_.Scalar")
    target = ("optimizer_ops.mojo", "ForeachAdd")
    native_calls = _spy_defined_native_calls(monkeypatch, {target})
    steps = [torch.zeros((), dtype=torch.float32).to(mojo_gpu) for _ in range(75)]
    for _ in range(3):
        torch.ops.aten._foreach_add_.Scalar(steps, 1)

    assert counter.call_count == calls_before + 3
    bridge_calls = native_calls[target]
    assert len(bridge_calls) == 3
    bridge_args = [args for args, _ in bridge_calls]
    assert all(len(args[0]) == 150 for args in bridge_args)
    for step in steps:
        assert step.shape == torch.Size([])
        assert step.cpu().item() == 3.0


def test_fast_foreach_mul_tensor_inplace_chunk_boundary(mojo_gpu):
    """Every input keeps its allocation and receives exactly one mutation."""
    counter, calls_before = _eager_registration_snapshot("aten::_foreach_mul_.Tensor")
    host_inputs = [torch.empty(0)] + [
        torch.tensor([float(index), -float(index)]) for index in range(1, 64)
    ]
    host_inputs.append(torch.linspace(-3.0, 4.0, 65_537))
    device_inputs = [tensor.to(mojo_gpu) for tensor in host_inputs]
    allocation_state = [
        (tensor._holder, tensor._ptr, tensor._version) for tensor in device_inputs
    ]
    coefficient = torch.tensor(0.25, dtype=torch.float32).to(mojo_gpu)

    returned = torch.ops.aten._foreach_mul_.Tensor(device_inputs, coefficient)

    assert returned is None
    assert counter.call_count == calls_before + 1
    for actual, expected, (holder, ptr, version) in zip(
        device_inputs, host_inputs, allocation_state, strict=True
    ):
        assert actual._holder is holder
        assert actual._ptr == ptr
        assert actual._version == version + 1
        torch.testing.assert_close(actual.cpu(), expected * 0.25, rtol=0, atol=0)


def test_fast_foreach_norm_and_mul_all_empty_batches(mojo_gpu):
    """Zero-work batches still return outputs and record in-place mutations."""
    inputs = [torch.empty(0, dtype=torch.float32).to(mojo_gpu) for _ in range(65)]

    norms = torch.ops.aten._foreach_norm.Scalar(inputs, 2)

    assert len(norms) == len(inputs)
    assert all(norm.shape == torch.Size([]) for norm in norms)
    assert all(norm.item() == 0.0 for norm in norms)
    versions = [tensor._version for tensor in inputs]
    scalar = torch.tensor(0.25, dtype=torch.float32).to(mojo_gpu)

    returned = torch.ops.aten._foreach_mul_.Tensor(inputs, scalar)

    assert returned is None
    assert [tensor._version for tensor in inputs] == [
        version + 1 for version in versions
    ]


def test_fast_foreach_mul_tensor_duplicate_is_sequential(mojo_gpu):
    """A duplicate entry is multiplied twice and bumps its version twice."""
    counter, calls_before = _eager_registration_snapshot("aten::_foreach_mul_.Tensor")
    input = torch.tensor([2.0, -3.0, 5.0]).to(mojo_gpu)
    holder, ptr, version = input._holder, input._ptr, input._version
    coefficient = torch.tensor(0.5, dtype=torch.float32).to(mojo_gpu)

    torch.ops.aten._foreach_mul_.Tensor([input, input], coefficient)

    assert counter.call_count == calls_before + 1
    assert input._holder is holder
    assert input._ptr == ptr
    assert input._version == version + 2
    torch.testing.assert_close(
        input.cpu(), torch.tensor([0.5, -0.75, 1.25]), rtol=0, atol=0
    )


def test_fast_foreach_mul_tensor_validates_scalar_before_writes(mojo_gpu):
    counter, calls_before = _eager_registration_snapshot("aten::_foreach_mul_.Tensor")
    inputs = [
        torch.tensor([1.0, 2.0]).to(mojo_gpu),
        torch.tensor([-3.0, 4.0]).to(mojo_gpu),
    ]
    allocation_state = [
        (tensor._holder, tensor._ptr, tensor._version, tensor.cpu())
        for tensor in inputs
    ]
    invalid_scalar = torch.tensor([0.25], dtype=torch.float32).to(mojo_gpu)

    with pytest.raises(RuntimeError, match="scalar tensor|0 dim"):
        torch.ops.aten._foreach_mul_.Tensor(inputs, invalid_scalar)

    assert counter.call_count == calls_before + 1
    for actual, (holder, ptr, version, expected) in zip(
        inputs, allocation_state, strict=True
    ):
        assert actual._holder is holder
        assert actual._ptr == ptr
        assert actual._version == version
        torch.testing.assert_close(actual.cpu(), expected, rtol=0, atol=0)


def test_fast_foreach_mul_tensor_preserves_strided_fallback(mojo_gpu):
    counter, calls_before = _eager_registration_snapshot("aten::_foreach_mul_.Tensor")
    base = torch.arange(12, dtype=torch.float32).to(mojo_gpu)
    input = base[::2]
    assert not input.is_contiguous()
    holder, ptr, version = input._holder, input._ptr, input._version
    coefficient = torch.tensor(0.25, dtype=torch.float32).to(mojo_gpu)

    torch.ops.aten._foreach_mul_.Tensor([input], coefficient)

    expected = torch.arange(12, dtype=torch.float32)
    expected[::2] *= 0.25
    assert counter.call_count == calls_before + 1
    assert input._holder is holder
    assert input._ptr == ptr
    assert input._version == version + 1
    torch.testing.assert_close(base.cpu(), expected, rtol=0, atol=0)


def test_fast_foreach_mul_tensor_overlapping_views_are_sequential(mojo_gpu):
    counter, calls_before = _eager_registration_snapshot("aten::_foreach_mul_.Tensor")
    base = torch.tensor([1.0, 2.0, 3.0, 4.0, 5.0]).to(mojo_gpu)
    left = base[:4]
    right = base[1:]
    coefficient = torch.tensor(2.0).to(mojo_gpu)
    version = base._version

    torch.ops.aten._foreach_mul_.Tensor([left, right], coefficient)

    assert counter.call_count == calls_before + 1
    assert base._version == left._version == right._version == version + 2
    torch.testing.assert_close(
        base.cpu(), torch.tensor([2.0, 8.0, 12.0, 16.0, 10.0]), rtol=0, atol=0
    )


def test_fast_foreach_mul_tensor_rejects_scalar_alias_before_writes(mojo_gpu):
    counter, calls_before = _eager_registration_snapshot("aten::_foreach_mul_.Tensor")
    input = torch.tensor([2.0, 3.0, 4.0]).to(mojo_gpu)
    scalar_alias = input[0]
    holder, ptr, version = input._holder, input._ptr, input._version

    with pytest.raises(RuntimeError, match="single memory location|clone"):
        torch.ops.aten._foreach_mul_.Tensor([input], scalar_alias)

    assert counter.call_count == calls_before + 1
    assert input._holder is holder
    assert input._ptr == ptr
    assert input._version == version
    torch.testing.assert_close(
        input.cpu(), torch.tensor([2.0, 3.0, 4.0]), rtol=0, atol=0
    )


def test_fast_foreach_mul_tensor_rejects_dense_transpose_scalar_alias(mojo_gpu):
    base = torch.arange(1.0, 7.0).to(mojo_gpu)
    input = base.reshape(2, 3).t()
    scalar_alias = base[1]
    version = base._version

    with pytest.raises(RuntimeError, match="single memory location|clone"):
        torch.ops.aten._foreach_mul_.Tensor([input], scalar_alias)

    assert base._version == input._version == version
    torch.testing.assert_close(base.cpu(), torch.arange(1.0, 7.0), rtol=0, atol=0)


def test_fast_foreach_mul_tensor_allows_full_scalar_self_alias(mojo_gpu):
    input = torch.tensor(3.0).to(mojo_gpu)
    version = input._version

    torch.ops.aten._foreach_mul_.Tensor([input], input)

    assert input._version == version + 1
    torch.testing.assert_close(input.cpu(), torch.tensor(9.0), rtol=0, atol=0)


def test_fast_foreach_mul_tensor_allows_scalar_in_strided_hole(mojo_gpu):
    base = torch.arange(1.0, 7.0).to(mojo_gpu)
    input = base[::2]
    scalar_in_hole = base[1]
    version = base._version

    torch.ops.aten._foreach_mul_.Tensor([input], scalar_in_hole)

    assert base._version == input._version == version + 1
    torch.testing.assert_close(
        base.cpu(), torch.tensor([2.0, 2.0, 6.0, 4.0, 10.0, 6.0]), rtol=0, atol=0
    )


@pytest.mark.parametrize("foreach", [None, True, False])
def test_fast_clip_grad_norm_foreach_routing(mojo_gpu, foreach):
    norm_counter, norm_calls_before = _eager_registration_snapshot(
        "aten::_foreach_norm.Scalar"
    )
    mul_counter, mul_calls_before = _eager_registration_snapshot(
        "aten::_foreach_mul_.Tensor"
    )
    host_parameters = [
        torch.nn.Parameter(torch.zeros(3)),
        torch.nn.Parameter(torch.zeros(2, 2)),
    ]
    device_parameters = [
        torch.nn.Parameter(parameter.detach().to(mojo_gpu))
        for parameter in host_parameters
    ]
    gradients = (
        torch.tensor([3.0, 4.0, -2.0]),
        torch.tensor([[1.0, -2.0], [2.0, -1.0]]),
    )
    for host_parameter, device_parameter, gradient in zip(
        host_parameters, device_parameters, gradients, strict=True
    ):
        host_parameter.grad = gradient.clone()
        device_parameter.grad = gradient.to(mojo_gpu)

    expected_norm = torch.nn.utils.clip_grad_norm_(
        host_parameters, 1.25, foreach=foreach
    )
    actual_norm = torch.nn.utils.clip_grad_norm_(
        device_parameters, 1.25, foreach=foreach
    )

    expected_foreach_calls = int(foreach is not False)
    assert norm_counter.call_count == norm_calls_before + expected_foreach_calls
    assert mul_counter.call_count == mul_calls_before + expected_foreach_calls
    torch.testing.assert_close(actual_norm.cpu(), expected_norm)
    for actual, expected in zip(device_parameters, host_parameters, strict=True):
        torch.testing.assert_close(actual.grad.cpu(), expected.grad)


@pytest.mark.parametrize("keepdim", [True, False])
def test_fast_mean_trailing_dims(mojo_device, keepdim):
    x = torch.randn(1, 512, 7, 7)
    result = x.to(mojo_device).mean([-1, -2], keepdim=keepdim).cpu()
    torch.testing.assert_close(result, x.mean([-1, -2], keepdim=keepdim))


@pytest.mark.parametrize(
    ("shape", "dims", "keepdim"),
    [
        ((3, 5, 7), (0,), False),
        ((3, 5, 7), (0,), True),
        ((5, 3, 17), (0, 1), False),
        ((7, 17, 65), (0,), False),
        ((2, 3, 5, 7), (1, 2), False),
        ((2, 3, 5, 7), (1, 2), True),
        ((2, 257, 17), (1,), False),
    ],
)
def test_fast_sum_contiguous_adjacent_dims(mojo_device, shape, dims, keepdim):
    """Adjacent reductions operate directly on contiguous storage, including
    a nonzero storage offset, and preserve the input view and its guards."""
    elements = math.prod(shape)
    host_storage = torch.arange(elements + 2, dtype=torch.float32)
    expected_input = host_storage[1:-1].reshape(shape)
    device_storage = host_storage.to(mojo_device)
    device_input = device_storage[1:-1].view(shape)
    holder, ptr = device_input._holder, device_input._ptr

    actual = device_input.sum(dim=dims, keepdim=keepdim)
    expected = expected_input.sum(dim=dims, keepdim=keepdim)

    torch.testing.assert_close(actual.cpu(), expected, rtol=2e-6, atol=2e-6)
    assert device_input._holder is holder
    assert device_input._ptr == ptr
    torch.testing.assert_close(device_storage.cpu(), host_storage, rtol=0, atol=0)


def test_fast_sum_nonadjacent_or_strided_fallback(mojo_device):
    """Layouts outside the direct adjacent-dimension regime remain correct."""
    host = torch.randn(2, 3, 5, 7)
    device = host.to(mojo_device)

    torch.testing.assert_close(device.sum(dim=(0, 2)).cpu(), host.sum(dim=(0, 2)))

    host_strided = host.transpose(1, 2)
    device_strided = device.transpose(1, 2)
    torch.testing.assert_close(
        device_strided.sum(dim=(0, 2)).cpu(), host_strided.sum(dim=(0, 2))
    )


def test_fast_reduction_split_tier(mojo_device):
    """Few outputs, huge reduce extent: the split-the-reduce-axis path, where
    stage 1 writes per-shard accumulators to a workspace and a merge kernel
    folds them.  Integer-valued floats keep every f32 partial sum exact, so the
    comparisons are bit-exact whatever order the shards merge in — which is the
    point: an arbitrary split must reproduce the sequential answer."""
    # rows == 1: full-reduction layout, one output over millions of elements.
    x = torch.randint(-4, 5, (1, 2**20 + 7)).float()
    xd = x.to(mojo_device)
    torch.testing.assert_close(xd.sum(-1).cpu(), x.sum(-1))
    torch.testing.assert_close(xd.amax(-1).cpu(), x.amax(-1))
    torch.testing.assert_close(torch.any(xd, -1).cpu(), torch.any(x, -1))

    # A handful of rows: still under-saturated, so still split, but now the
    # merge has to keep per-row outputs apart.
    y = torch.randint(-4, 5, (128, 2**20)).float()
    y[5] = 0.0  # give any() a False row
    yd = y.to(mojo_device)
    torch.testing.assert_close(yd.sum(-1).cpu(), y.sum(-1))
    torch.testing.assert_close(yd.amax(-1).cpu(), y.amax(-1))
    torch.testing.assert_close(torch.any(yd, -1).cpu(), torch.any(y, -1))


def test_fast_anyall_nan_is_truthy(mojo_device):
    """torch treats NaN as truthy in any/all. Cover both launch regimes: the
    small shape fuses into one launch, the huge shape splits the reduce axis
    across blocks and merges."""
    small_any = torch.zeros(2, 100)
    small_any[0, 0] = float("nan")
    small_all = torch.full((2, 100), float("nan"))
    huge_any = torch.zeros(1, 2**20 + 7)
    huge_any[0, 12345] = float("nan")
    huge_all = torch.ones(1, 2**20 + 7)
    huge_all[0, 999] = float("nan")
    for x in (small_any, huge_any):
        got = torch.any(x.to(mojo_device), -1).cpu()
        torch.testing.assert_close(got, torch.any(x, -1))
    for x in (small_all, huge_all):
        got = torch.all(x.to(mojo_device), -1).cpu()
        torch.testing.assert_close(got, torch.all(x, -1))


# --------------------------------------------------------------------------
# The generic reduction skeleton (reduce_skeleton.mojo).  One accumulator per
# op over one (outer, reduce, inner) geometry, so these cases are written once
# and parametrized by the op rather than once per op.
#
# `_REDUCE_OPS` maps a name to (mojo callable, torch reference); everything
# below runs the whole table so a new accumulator inherits the coverage.
# --------------------------------------------------------------------------

_REDUCE_OPS = {
    "sum": lambda t, **kw: torch.sum(t, **kw),
    "mean": lambda t, **kw: torch.mean(t, **kw),
    "amax": lambda t, **kw: torch.amax(t, **kw),
    "amin": lambda t, **kw: torch.amin(t, **kw),
    "norm": lambda t, **kw: torch.linalg.vector_norm(t, **kw),
    "all": lambda t, **kw: torch.all(t, **kw),
    "any": lambda t, **kw: torch.any(t, **kw),
}


@pytest.mark.parametrize("op", list(_REDUCE_OPS))
@pytest.mark.parametrize(
    "shape,dim",
    [
        # Straddle the split threshold on the CONTIGUOUS axis: the split count
        # is derived from the runtime SM count, so these bracket it on any card
        # rather than at one fitted size.  Prime-ish extents leave the last
        # shard short, which is what an identity-seeded merge has to tolerate.
        ((1048583,), 0),  # one output, millions deep: maximum splitting
        ((3, 400001), 1),  # a few outputs: still split
        ((4099, 1031), 1),  # many outputs: fused, block per row
        ((5003, 37), 1),  # many SHORT rows: fused, warp per row
        # ... and on the STRIDED axis, where a non-trailing reduce dim is read
        # where it lies instead of being materialized transposed.
        ((400001, 3), 0),
        ((2, 65537), 0),
        ((1031, 4099), 0),
        ((37, 5003), 0),
        # Awkward rank-3 interiors: outer > 1 AND inner > 1.
        ((7, 129, 33), 1),
        ((3, 5, 7), 1),
    ],
)
def test_fast_reduce_skeleton_layouts_match_cpu(mojo_gpu, shape, dim, op):
    fn = _REDUCE_OPS[op]
    if op in ("all", "any"):
        x = torch.rand(shape) < 0.5
    else:
        x = torch.rand(shape) * 0.9 + 0.05
    ours = fn(x.to(mojo_gpu), dim=dim).cpu()
    if op in ("all", "any"):
        torch.testing.assert_close(ours, fn(x, dim=dim))
    else:
        # fp64 reference on the same values: this measures the reduction
        # order, not the input dtype.
        expected = fn(x.double(), dim=dim)
        torch.testing.assert_close(ours.double(), expected, atol=1e-6, rtol=1e-4)


@pytest.mark.parametrize("op", list(_REDUCE_OPS))
@pytest.mark.parametrize("dtype", [torch.float32, torch.bfloat16])
@pytest.mark.parametrize("dim", [0, 1])
def test_fast_reduce_skeleton_nonfinite_matches_torch(mojo_gpu, op, dtype, dim):
    """NaN and +/-inf, per op, against stock torch rather than against a rule.

    The rules genuinely differ: amax/amin PROPAGATE NaN (a plain `max()` does
    not, which is why the accumulator carries its own combine), sum and mean
    inherit it through the arithmetic, the L2 norm squares it, and any/all
    treat NaN as TRUTHY because their map is a nonzero test and not a
    comparison.  Every element of the grid below is exercised on both layouts,
    since the contiguous and strided kernels seed and combine separately.
    """
    if op in ("all", "any") and dtype is torch.bfloat16:
        pytest.skip("any/all take the bool fast path; dtype is not the axis")
    nan, inf = float("nan"), float("inf")
    x = torch.tensor(
        [
            [1.0, 2.0, 3.0, 4.0],
            [nan, 2.0, 3.0, 4.0],
            [1.0, inf, 3.0, 4.0],
            [1.0, 2.0, -inf, 4.0],
            [inf, -inf, 3.0, 4.0],
            [nan, inf, -inf, 0.0],
            [0.0, 0.0, 0.0, 0.0],
        ],
        dtype=dtype,
    )
    fn = _REDUCE_OPS[op]
    expected = fn(x, dim=dim)
    ours = fn(x.to(mojo_gpu), dim=dim).cpu()
    if ours.dtype == torch.bool:
        torch.testing.assert_close(ours, expected)
    else:
        torch.testing.assert_close(
            ours.float(), expected.float(), equal_nan=True, atol=1e-2, rtol=1e-2
        )


@pytest.mark.parametrize("op", list(_REDUCE_OPS))
@pytest.mark.parametrize(
    "shape,dim",
    [
        ((3, 1), 1),  # reduce extent of exactly 1
        ((1, 3), 1),  # one output
        ((1, 1), 1),
        ((3, 0), 1),  # EMPTY reduce axis: identity, or torch's own error
        ((0, 3), 1),  # empty output
        ((3, 0), 0),
    ],
)
def test_fast_reduce_skeleton_degenerate_extents_match_torch(mojo_gpu, shape, dim, op):
    """A zero-length reduce axis is where an identity-seeded accumulator and a
    zero-filled workspace part company, and where each op's own rule shows: sum
    answers 0, the L2 norm 0, all true, any false, mean nan (0/0) — and amax /
    amin refuse, because torch refuses ("Expected reduction dim to have
    non-zero size"): there is no element to select.  A reduction with no
    OUTPUTS is an error for nobody and writes nothing."""
    fn = _REDUCE_OPS[op]
    x = (torch.rand(shape) < 0.5) if op in ("all", "any") else torch.rand(shape)
    try:
        expected = fn(x, dim=dim)
    except (RuntimeError, IndexError) as exc:
        # The device declines the same cases; it reports its own
        # NotImplementedError rather than reproducing torch's message.
        with pytest.raises((type(exc), NotImplementedError)):
            fn(x.to(mojo_gpu), dim=dim)
        return
    ours = fn(x.to(mojo_gpu), dim=dim).cpu()
    if ours.dtype == torch.bool:
        torch.testing.assert_close(ours, expected)
    else:
        torch.testing.assert_close(ours, expected, equal_nan=True)


@pytest.mark.parametrize("op", ["all", "any"])
@pytest.mark.parametrize("size", [1, 255, 4096, (1 << 22) - 1, (1 << 22) + 1, 1 << 24])
def test_fast_bool_full_reduce_settles_early_and_late(mojo_gpu, op, size):
    """Full any()/all() over bool, including the aten cap at 4.2M elements that
    separates the AnyBool/AllBool entry from the AnySpec/AllSpec one, and the
    split threshold above it.  Both the settled-immediately case (element 0
    decides) and the settled-at-the-very-end case must give torch's answer: a
    reduction may not stop early in a way that skips a later element."""
    fn = torch.all if op == "all" else torch.any
    seed = (
        torch.ones(size, dtype=torch.bool)
        if op == "all"
        else torch.zeros(size, dtype=torch.bool)
    )
    for position in (0, size // 2, size - 1, None):
        x = seed.clone()
        if position is not None:
            x[position] = op != "all"
        assert bool(fn(x.to(mojo_gpu)).cpu()) == bool(fn(x)), (
            f"{op} size={size} flipped at {position}"
        )


@pytest.mark.parametrize(
    "spec,fn",
    [
        (("reduction_ops", "SumSpec"), lambda t: torch.sum(t, dim=0)),
        (("reduction_ops", "AmaxSpec"), lambda t: torch.amax(t, dim=0)),
        (("reduction_ops", "AminSpec"), lambda t: torch.amin(t, dim=0)),
        (("reduction_ops", "NormSpec"), lambda t: torch.linalg.vector_norm(t, dim=0)),
        (("reduction_ops", "MinDimSpec"), lambda t: torch.min(t, dim=0)),
        (("nn_ops", "MeanSpec"), lambda t: torch.mean(t, dim=0)),
    ],
)
def test_fast_reduce_strided_axis_skips_materialization(
    mojo_gpu, monkeypatch, spec, fn
):
    """Reducing a leading dim must reach the bridge with its ORIGINAL dims.

    If it did not, Python would have permuted the operand and materialized a
    full transposed copy first, and the bridge would see the trailing dim
    instead — an extra read plus an extra write of the whole tensor, which was
    the entire cost of a `dim=0` reduction before the strided-axis kernel.
    """
    target = (f"{spec[0]}.mojo", spec[1])
    native_calls = _spy_defined_native_calls(monkeypatch, {target})
    x = torch.randn(64, 48).to(mojo_gpu)
    _ = fn(x)
    assert len(native_calls[target]) == 1
    assert native_calls[target][0][0][1] == (0,)


@pytest.mark.parametrize("dtype", [torch.float32, torch.bfloat16])
@pytest.mark.parametrize(
    "shape,dim",
    [((4099, 1031), 1), ((5003, 37), 1), ((1031, 4099), 0), ((1 << 20,), 0)],
)
def test_fast_vector_norm_matches_torch(mojo_gpu, shape, dim, dtype):
    """linalg_vector_norm is one pass now (sum of squares, root in the
    finalize) rather than mul -> sum -> sqrt with a full-size temporary."""
    x = (torch.rand(shape) * 0.9 + 0.05).to(dtype)
    ours = torch.linalg.vector_norm(x.to(mojo_gpu), dim=dim).cpu()
    expected = torch.linalg.vector_norm(x.double(), dim=dim)
    torch.testing.assert_close(ours.double(), expected, atol=0, rtol=1e-2)


@pytest.mark.parametrize("dtype", [torch.float32, torch.bfloat16])
@pytest.mark.parametrize(
    "shape,dim",
    [((4099, 1031), 1), ((5003, 37), 1), ((1031, 4099), 0), ((1 << 20,), 0)],
)
def test_fast_min_dim_layouts_and_ties(mojo_gpu, shape, dim, dtype):
    """min.dim rides the (value, index) arg-reduction now, so it inherits the
    split path, the strided-axis kernel and first-occurrence tie-breaking."""
    x = (torch.rand(shape) * 0.9 + 0.05).to(dtype)
    # Force ties so the index answer is only right if the lower one wins.
    flat = x.reshape(-1)
    flat[: flat.numel() // 3] = 0.05
    values, indices = torch.min(x.to(mojo_gpu), dim=dim)
    exp_values, exp_indices = torch.min(x, dim=dim)
    torch.testing.assert_close(values.cpu(), exp_values)
    torch.testing.assert_close(indices.cpu(), exp_indices)


@pytest.mark.parametrize("dim", [0, 1])
def test_fast_min_dim_propagates_nan_like_torch(mojo_gpu, dim):
    """torch's min.dim answers with the FIRST NaN, not with the smallest
    number; the old block kernel's plain `<` silently answered the number."""
    nan = float("nan")
    x = torch.tensor(
        [[1.0, nan, -7.0, 3.0], [nan, nan, 2.0, 1.0], [5.0, 4.0, 3.0, 2.0]]
    )
    values, indices = torch.min(x.to(mojo_gpu), dim=dim)
    exp_values, exp_indices = torch.min(x, dim=dim)
    torch.testing.assert_close(values.cpu(), exp_values, equal_nan=True)
    torch.testing.assert_close(indices.cpu(), exp_indices)


def test_fast_max_pool2d(mojo_device):
    x = torch.randn(1, 64, 32, 32)
    result = torch.nn.functional.max_pool2d(x.to(mojo_device), 3, 2, 1).cpu()
    torch.testing.assert_close(result, torch.nn.functional.max_pool2d(x, 3, 2, 1))


def test_fast_max_pool2d_indices(mojo_device):
    x = torch.randn(1, 8, 16, 16)
    dev_vals, dev_idx = torch.nn.functional.max_pool2d(
        x.to(mojo_device), 2, 2, return_indices=True
    )
    ref_vals, ref_idx = torch.nn.functional.max_pool2d(x, 2, 2, return_indices=True)
    torch.testing.assert_close(dev_vals.cpu(), ref_vals)
    torch.testing.assert_close(dev_idx.cpu(), ref_idx)


def test_fast_embedding(mojo_device):
    weight = torch.randn(100, 32)
    idx = torch.randint(0, 100, (2, 5))
    result = torch.nn.functional.embedding(idx.to(mojo_device), weight.to(mojo_device))
    torch.testing.assert_close(result.cpu(), torch.nn.functional.embedding(idx, weight))


@pytest.mark.parametrize("strided", [False, True])
def test_fast_embedding_dense_backward_repeated_padding(mojo_gpu, strided):
    """The direct eager kernel accumulates duplicates and skips padding."""
    num_weights = 11
    padding_idx = 7
    indices = torch.tensor([[1, 7, 1, 4], [4, 1, 9, 7]], dtype=torch.int64)
    grad_output = torch.arange(1, indices.numel() * 5 + 1, dtype=torch.float32)
    grad_output = grad_output.reshape(*indices.shape, 5) / 16.0
    if strided:
        index_storage = torch.full((2, 8), 3, dtype=torch.int64)
        index_storage[:, ::2] = indices
        indices = index_storage[:, ::2]
        grad_storage = torch.empty(*indices.shape, 10)
        grad_storage[..., ::2] = grad_output
        grad_storage[..., 1::2] = -123.0
        grad_output = grad_storage[..., ::2]
        assert not indices.is_contiguous()
        assert not grad_output.is_contiguous()
        device_indices = index_storage.to(mojo_gpu)[:, ::2]
        device_grad_output = grad_storage.to(mojo_gpu)[..., ::2]
        assert not device_indices._is_contiguous
        assert not device_grad_output._is_contiguous
    else:
        device_indices = indices.to(mojo_gpu)
        device_grad_output = grad_output.to(mojo_gpu)

    expected = torch.ops.aten.embedding_dense_backward.default(
        grad_output, indices, num_weights, padding_idx, False
    )
    actual = torch.ops.aten.embedding_dense_backward.default(
        device_grad_output, device_indices, num_weights, padding_idx, False
    )
    torch.testing.assert_close(actual.cpu(), expected, rtol=0, atol=0)


def test_fast_embedding_dense_backward_strided_temporary_lifetime(
    mojo_gpu, monkeypatch
):
    """Internal contiguous copies may die once their launches are enqueued."""
    from torch_mojo_backend.mojo_device.torch_mojo_tensor import TorchMojoTensor

    num_weights = 13
    padding_idx = 5
    indices_storage = torch.tensor(
        [[2, 99, 5, 99, 2, 99], [8, 99, 2, 99, 8, 99]], dtype=torch.int64
    )
    grad_storage = torch.arange(1, 2 * 3 * 14 + 1, dtype=torch.float32).reshape(
        2, 3, 14
    )
    indices = indices_storage[:, ::2]
    grad_output = grad_storage[..., ::2]
    expected = torch.ops.aten.embedding_dense_backward.default(
        grad_output, indices, num_weights, padding_idx, False
    )
    device_indices = indices_storage.to(mojo_gpu)[:, ::2]
    device_grad_output = grad_storage.to(mojo_gpu)[..., ::2]
    assert not device_indices._is_contiguous
    assert not device_grad_output._is_contiguous

    materialized = []
    original = TorchMojoTensor._materialize_contiguous

    def record_materialized(tensor):
        result = original(tensor)
        materialized.append(weakref.ref(result))
        return result

    monkeypatch.setattr(TorchMojoTensor, "_materialize_contiguous", record_materialized)
    actual = torch.ops.aten.embedding_dense_backward.default(
        device_grad_output, device_indices, num_weights, padding_idx, False
    )

    assert len(materialized) == 2
    assert all(reference() is None for reference in materialized)
    torch.testing.assert_close(actual.cpu(), expected, rtol=0, atol=0)


def test_fast_embedding_dense_backward_host_bridge_abi(mojo_gpu, monkeypatch):
    """The host forwards nine runtime arguments and the tensor's own context."""
    from torch_mojo_backend.eager_kernels import _ctx_ptr, aten_fast

    grad_storage = torch.arange(2 * 3 * 5 + 3, dtype=torch.float32).to(mojo_gpu)
    index_storage = torch.tensor([99, 99, 1, 2, 1, 4, 2, 7], dtype=torch.int64).to(
        mojo_gpu
    )
    grad_output = grad_storage[3:].view(2, 3, 5)
    indices = index_storage[2:].view(2, 3)
    calls = []
    _replace_defined_native_calls(
        monkeypatch,
        {
            ("embedding_backward_ops.mojo", "EmbeddingDenseBackwardF32I64"): (
                lambda *args: calls.append(args)
            )
        },
    )

    output = aten_fast.fast_aten_embedding_dense_backward(
        grad_output, indices, 11, 7, False
    )

    assert len(calls) == 1
    assert calls[0] == (
        output._ptr,
        grad_output._ptr,
        indices._ptr,
        6,
        5,
        11,
        7,
        0,
        _ctx_ptr(grad_output._device),
    )
    assert calls[0][-1] == output._device._device_context_ptr()
    assert output._device == grad_output._device
    assert output._holder.get_nbytes() == 11 * 5 * torch.float32.itemsize


def test_fast_embedding_dense_backward_uses_each_gpu_context():
    """Explicit Mojo GPU indices must not silently launch on device zero."""
    gpu_count = sum(device.label == "gpu" for device in get_accelerators())
    if gpu_count < 2:
        pytest.skip("requires at least two MAX GPUs")

    indices = torch.tensor([[1, 3, 1], [4, 3, 2]], dtype=torch.int64)
    grad_output = torch.arange(1, indices.numel() * 5 + 1, dtype=torch.float32).view(
        *indices.shape, 5
    )
    expected = torch.ops.aten.embedding_dense_backward.default(
        grad_output, indices, 7, 3, False
    )
    for index in range(gpu_count):
        device = f"mojo:{index}"
        actual = torch.ops.aten.embedding_dense_backward.default(
            grad_output.to(device), indices.to(device), 7, 3, False
        )
        assert actual.device == torch.device(device)
        torch.testing.assert_close(actual.cpu(), expected, rtol=0, atol=0)

    with pytest.raises(NotImplementedError, match="embedding_dense_backward"):
        torch.ops.aten.embedding_dense_backward.default(
            grad_output.to("mojo:0"), indices.to("mojo:1"), 7, 3, False
        )


def test_fast_embedding_dense_backward_empty_indices(mojo_gpu):
    indices = torch.empty((0, 3), dtype=torch.int64)
    grad_output = torch.empty((0, 3, 5), dtype=torch.float32)
    actual = torch.ops.aten.embedding_dense_backward.default(
        grad_output.to(mojo_gpu), indices.to(mojo_gpu), 13, -1, False
    )
    assert tuple(actual.shape) == (13, 5)
    torch.testing.assert_close(actual.cpu(), torch.zeros(13, 5), rtol=0, atol=0)


def test_fast_embedding_training_backward(mojo_gpu):
    """F.embedding must keep the Mojo payload through SavedVariable unpack."""
    padding_idx = 3
    indices = torch.tensor([[1, 3, 1, 6], [6, 2, 1, 3]], dtype=torch.int64)
    weight = torch.randn(9, 7, requires_grad=True)
    grad_output = torch.arange(1, indices.numel() * 7 + 1, dtype=torch.float32)
    grad_output = grad_output.reshape(*indices.shape, 7) / 32.0

    expected = torch.nn.functional.embedding(indices, weight, padding_idx=padding_idx)
    expected.backward(grad_output)

    device_weight = weight.detach().to(mojo_gpu).requires_grad_()
    actual = torch.nn.functional.embedding(
        indices.to(mojo_gpu), device_weight, padding_idx=padding_idx
    )
    actual.backward(grad_output.to(mojo_gpu))
    torch.testing.assert_close(actual.cpu(), expected.detach(), rtol=0, atol=0)
    torch.testing.assert_close(device_weight.grad.cpu(), weight.grad, rtol=0, atol=0)


def test_fast_embedding_backward_rejects_unsupported_modes(mojo_gpu):
    indices = torch.tensor([0, 1, 1], dtype=torch.int64).to(mojo_gpu)
    grad_output = torch.ones(3, 4).to(mojo_gpu)
    with pytest.raises(NotImplementedError, match="scale_grad_by_freq"):
        torch.ops.aten.embedding_dense_backward.default(
            grad_output, indices, 3, -1, True
        )

    was_deterministic = torch.are_deterministic_algorithms_enabled()
    was_warn_only = torch.is_deterministic_algorithms_warn_only_enabled()
    try:
        torch.use_deterministic_algorithms(True)
        with pytest.raises(RuntimeError, match="does not have a deterministic"):
            torch.ops.aten.embedding_dense_backward.default(
                grad_output, indices, 3, -1, False
            )

        torch.use_deterministic_algorithms(True, warn_only=True)
        with pytest.warns(UserWarning, match="does not have a deterministic"):
            actual = torch.ops.aten.embedding_dense_backward.default(
                grad_output, indices, 3, -1, False
            )
        expected = torch.ops.aten.embedding_dense_backward.default(
            grad_output.cpu(), indices.cpu(), 3, -1, False
        )
        torch.testing.assert_close(actual.cpu(), expected, rtol=0, atol=0)
    finally:
        torch.use_deterministic_algorithms(was_deterministic, warn_only=was_warn_only)


def test_fast_embedding_autograd_reports_nondeterminism_at_forward(mojo_gpu):
    """The alert must fire while recording: a backward-time raise aborts.

    Exceptions thrown from a node the autograd engine runs on this backend
    escalate to std::terminate (the engine's stream guard restores streams
    through PyTorch's noexcept Python device guard during unwind).
    """
    indices = torch.tensor([0, 1, 1], dtype=torch.int64).to(mojo_gpu)
    weight = torch.randn(3, 4).to(mojo_gpu).requires_grad_()
    was_deterministic = torch.are_deterministic_algorithms_enabled()
    was_warn_only = torch.is_deterministic_algorithms_warn_only_enabled()
    try:
        torch.use_deterministic_algorithms(True)
        with pytest.raises(RuntimeError, match="does not have a deterministic"):
            torch.nn.functional.embedding(indices, weight)

        torch.use_deterministic_algorithms(True, warn_only=True)
        with pytest.warns(UserWarning, match="does not have a deterministic"):
            output = torch.nn.functional.embedding(indices, weight)
        output.sum().backward()
        assert weight.grad is not None
    finally:
        torch.use_deterministic_algorithms(was_deterministic, warn_only=was_warn_only)


@pytest.mark.parametrize(
    ("kwargs", "match"),
    [
        ({"scale_grad_by_freq": True}, "scale_grad_by_freq"),
        ({"sparse": True}, "sparse=True"),
    ],
)
def test_fast_embedding_autograd_rejects_unsafe_native_modes(mojo_gpu, kwargs, match):
    indices = torch.tensor([0, 1, 1], dtype=torch.int64).to(mojo_gpu)
    weight = torch.randn(3, 4).to(mojo_gpu).requires_grad_()

    with pytest.raises(NotImplementedError, match=match):
        torch.nn.functional.embedding(indices, weight, **kwargs)


def test_fast_scalar_elementwise(mojo_device):
    x = torch.randn(2, 6, 3072)
    xd = x.to(mojo_device)
    torch.testing.assert_close((xd * 0.5).cpu(), x * 0.5)
    torch.testing.assert_close((xd + 1.0).cpu(), x + 1.0)
    torch.testing.assert_close((xd**3.0).cpu(), x**3.0, atol=1e-4, rtol=1e-4)
    torch.testing.assert_close(torch.tanh(xd).cpu(), torch.tanh(x))


@pytest.mark.parametrize("approximate", ["none", "tanh"])
@pytest.mark.parametrize("storage_offset", [0, 1])
@pytest.mark.parametrize("shape", [(257,), (3, 5, 7)])
def test_fast_gelu_forward_bf16_direct_runtime_layout(
    mojo_gpu, monkeypatch, approximate, storage_offset, shape
):
    """The direct BF16 bridge covers aligned and two-byte-offset tails."""
    from torch_mojo_backend.eager_kernels.aten_fast import _ctx_ptr

    elements = math.prod(shape)
    backing = torch.linspace(-8.0, 8.0, elements + storage_offset, dtype=torch.bfloat16)
    input = backing[storage_offset:].view(shape)
    device_backing = backing.to(mojo_gpu)
    device_input = device_backing[storage_offset:].view(shape)
    input_ptr = device_input._ptr
    input_version = device_input._version
    target = ("activation_forward_ops.mojo", "GeluForwardBF16")
    native_calls = _spy_defined_native_calls(monkeypatch, {target})
    actual = torch.nn.functional.gelu(device_input, approximate=approximate)
    expected = torch.nn.functional.gelu(input, approximate=approximate)

    assert [args for args, _ in native_calls[target]] == [
        (
            actual._ptr,
            device_input._ptr,
            elements,
            int(approximate == "tanh"),
            _ctx_ptr(device_input._device),
        )
    ]
    assert actual.shape == input.shape
    assert actual.stride() == input.stride()
    assert actual.dtype == torch.bfloat16
    assert actual._ptr != device_input._ptr
    assert actual._version == 0
    assert device_input._ptr == input_ptr
    assert device_input._version == input_version
    torch.testing.assert_close(device_backing.cpu(), backing, atol=0, rtol=0)
    torch.testing.assert_close(actual.cpu(), expected, atol=2e-2, rtol=2e-2)


def _gelu_exact_reference(x: torch.Tensor) -> torch.Tensor:
    """FP64 exact GELU with no cancellation, for use as a test oracle.

    `torch.nn.functional.gelu(x.double())` is NOT usable as an oracle in the
    negative tail: it computes `0.5*x*(1 + erf(x/sqrt2))`, and `1 + erf` is
    quantized by the FP64 epsilon at 1.0, so it has lost half its significant
    bits by x = -8 and all of them by x = -8.5.  The identity
    `gelu(x) = relu(x) - 0.5*|x|*erfc(|x|/sqrt2)` never forms `1 - (1 - eps)`,
    and `torch.special.erfc` keeps full relative accuracy out to the FP64
    underflow, so this stays exact over the whole BF16 range.
    """
    wide = x.double()
    magnitude = wide.abs()
    correction = 0.5 * magnitude * torch.special.erfc(magnitude / math.sqrt(2.0))
    return wide.clamp(min=0.0) - correction


def test_fast_gelu_forward_bf16_exact_matches_double_reference(mojo_gpu):
    """Every finite BF16 value in [-12.5, 12.5] rounds like the true exact GELU.

    The exact path evaluates `relu(x) - 0.5*|x|*erfc(|x|/sqrt2)` from a fitted
    log2-domain polynomial and one `exp2` rather than `0.5*x*(1+erf(x/sqrt2))`,
    so the bound that matters is not "close to the previous kernel" but "rounds
    to the same BF16 as the true function".

    12.5 is where `exp2` reaches the smallest FP32 normal, so the last twelve
    BF16 inputs before the answer underflows BF16 altogether (|x| in
    [12.8, 13.5], true value under 1e-36) are outside the guarantee -- and are
    still nearer the truth than the form this replaces, which returns exactly
    zero for everything below x = -5.2.
    """
    values = torch.arange(0, 1 << 16, dtype=torch.int32).to(torch.uint16)
    grid = values.view(torch.bfloat16)
    grid = grid[torch.isfinite(grid) & (grid.abs() <= 12.5)].contiguous()
    assert grid.numel() > 30000

    actual = torch.nn.functional.gelu(grid.to(mojo_gpu), approximate="none").cpu()
    reference = _gelu_exact_reference(grid).bfloat16()

    actual_bits = actual.view(torch.int16).int()
    reference_bits = reference.view(torch.int16).int()
    differing = actual_bits != reference_bits
    worst = int((actual_bits - reference_bits).abs().max())
    assert worst <= 1, f"a BF16 result is {worst} ULP from the true exact GELU"
    # The survivors are genuine round-to-nearest ties, not a systematic bias.
    assert int(differing.sum()) <= 8, (
        f"{int(differing.sum())} of {grid.numel()} BF16 values differ by 1 ULP"
    )


def test_fast_gelu_forward_bf16_exact_resolves_the_negative_tail(mojo_gpu):
    """`0.5*x*(1 + erf)` loses the whole answer below x = -5; this must not.

    In FP32 `1 + erf(x/sqrt2)` is quantized by the epsilon at 1.0, so any form
    built on it returns exactly zero once `2*Phi(x)` drops under 1.2e-7, i.e.
    from about x = -5.2 -- while BF16 still resolves the true value down to
    x = -13.7.  This pins the property rather than the digits.
    """
    x = torch.arange(-12.5, -5.0, 0.0625, dtype=torch.bfloat16)
    actual = torch.nn.functional.gelu(x.to(mojo_gpu), approximate="none").cpu()

    assert bool((actual < 0).all()), "the negative tail collapsed to zero"
    # Strictly decreasing in x over this range, i.e. the decay has the shape of
    # the true function and not of a floor or a plateau.
    assert bool((actual[1:].double() < actual[:-1].double()).all())
    torch.testing.assert_close(
        actual.double(), _gelu_exact_reference(x), rtol=8e-3, atol=0
    )


@pytest.mark.parametrize("approximate", ["none", "tanh"])
def test_fast_gelu_forward_bf16_cuda_special_semantics(mojo_h100, approximate):
    """Signed zero, non-finites, and mode probes use frozen H100 results."""
    input_bits = torch.tensor(
        [
            0x0000,
            0x8000,
            0x7F80,
            0xFF80,
            0x7FC0,
            0x0001,
            0x8001,
            0x7F7F,
            0xFF7F,
            0x4005,
            0x4030,
        ],
        dtype=torch.uint16,
    )
    input = input_bits.view(torch.bfloat16)
    device_input = input.to(mojo_h100)

    actual = torch.nn.functional.gelu(device_input, approximate=approximate).cpu()
    actual_bits = actual.view(torch.uint16)

    assert int(actual_bits[0]) == 0x0000
    assert int(actual_bits[1]) == 0x8000
    assert torch.isposinf(actual[2])
    if approximate == "tanh":
        assert torch.isnan(actual[3])
    else:
        # CUDA's `0.5*x*(1 + erf(x/sqrt2))` reaches `-inf * 0` here and returns
        # NaN. The exact path no longer forms that product -- it computes
        # `relu(x) - 0.5*|x|*erfc(...)`, whose limit at -inf is the correct
        # -0.0 -- so this one input diverges from CUDA, deliberately, and in
        # the direction of the true function.
        assert int(actual_bits[3]) == 0x8000
    assert torch.isnan(actual[4])
    expected_probes = (0x4002, 0x402F) if approximate == "none" else (0x4003, 0x4030)
    assert tuple(int(value) for value in actual_bits[-2:]) == expected_probes
    torch.testing.assert_close(
        device_input.cpu().view(torch.int16), input.view(torch.int16), atol=0, rtol=0
    )


@pytest.mark.parametrize("approximate", ["none", "tanh"])
@pytest.mark.parametrize(
    "case", ["fp32_contiguous", "fp16_contiguous", "bf16_transpose", "bf16_gapped"]
)
def test_fast_gelu_forward_preserves_generic_fallbacks(
    mojo_gpu, monkeypatch, case, approximate
):
    """Other regimes retain the existing generic path and value behavior.

    Layout parity for noncontiguous inputs is outside this optimization: the
    existing generic path currently returns a row-major output.
    """
    if case == "fp32_contiguous":
        input = torch.randn(5, 7)
        device_input = input.to(mojo_gpu)
    elif case == "fp16_contiguous":
        input = torch.randn(5, 7, dtype=torch.float16)
        device_input = input.to(mojo_gpu)
    elif case == "bf16_transpose":
        backing = torch.randn(7, 5, dtype=torch.bfloat16)
        input = backing.t()
        device_input = backing.to(mojo_gpu).t()
        assert not device_input.is_contiguous()
    else:
        backing = torch.randn(71, dtype=torch.bfloat16)
        input = backing[1:71:2]
        device_input = backing.to(mojo_gpu)[1:71:2]
        assert not device_input.is_contiguous()

    def reject_direct(*_args):
        raise AssertionError("unsupported GELU input used the direct BF16 bridge")

    _replace_defined_native_calls(
        monkeypatch, {("activation_forward_ops.mojo", "GeluForwardBF16"): reject_direct}
    )
    actual = torch.nn.functional.gelu(device_input, approximate=approximate)
    expected = torch.nn.functional.gelu(input, approximate=approximate)

    tolerance = 5e-5 if input.dtype == torch.float32 else 2e-2
    torch.testing.assert_close(actual.cpu(), expected, atol=tolerance, rtol=tolerance)


@pytest.mark.parametrize("approximate", ["none", "tanh"])
def test_fast_gelu_forward_bf16_cpu_preserves_generic_path(monkeypatch, approximate):
    """BF16 on the MAX CPU device must not enter the accelerator bridge."""
    cpu_device = f"mojo:{len(list(get_accelerators())) - 1}"
    input = torch.randn(5, 7, dtype=torch.bfloat16)

    def reject_direct(*_args):
        raise AssertionError("MAX CPU GELU used the direct GPU bridge")

    _replace_defined_native_calls(
        monkeypatch, {("activation_forward_ops.mojo", "GeluForwardBF16"): reject_direct}
    )
    actual = torch.nn.functional.gelu(
        input.to(cpu_device), approximate=approximate
    ).cpu()
    expected = torch.nn.functional.gelu(input, approximate=approximate)

    torch.testing.assert_close(actual, expected, atol=2e-2, rtol=2e-2)


def test_fast_gelu_forward_bf16_empty_does_not_enqueue(mojo_gpu, monkeypatch):
    """Empty BF16 tensors preserve metadata without launching a GPU kernel."""

    def reject_direct(*_args):
        raise AssertionError("empty GELU forward enqueued a kernel")

    _replace_defined_native_calls(
        monkeypatch, {("activation_forward_ops.mojo", "GeluForwardBF16"): reject_direct}
    )
    input = torch.empty(0, 7, dtype=torch.bfloat16).to(mojo_gpu)

    actual = torch.nn.functional.gelu(input)

    assert actual.shape == input.shape
    assert actual.stride() == input.stride()
    assert actual.dtype == torch.bfloat16
    assert actual.device.type == "mojo"


def test_fast_gelu_forward_invalid_mode_rejects_before_materialization(
    mojo_gpu, monkeypatch
):
    """Invalid metadata is rejected before tensor or output work begins."""
    from torch_mojo_backend.eager_kernels import aten_fast

    input = torch.randn(17, dtype=torch.bfloat16).to(mojo_gpu)

    def reject_work(*_args, **_kwargs):
        raise AssertionError("invalid GELU mode performed tensor work")

    monkeypatch.setattr(aten_fast, "_t", reject_work)
    monkeypatch.setattr(aten_fast, "_alloc", reject_work)
    monkeypatch.setattr(aten_fast, "_unary_spec_op", reject_work)

    assert aten_fast.fast_aten_gelu(input, "invalid") is aten_fast.NOT_HANDLED
    with pytest.raises(NotImplementedError):
        torch.ops.aten.gelu.default(input, approximate="invalid")


@pytest.mark.parametrize("approximate", ["none", "tanh"])
@pytest.mark.parametrize("layout", ["contiguous_offset", "strided"])
def test_fast_gelu_backward_runtime_layouts(mojo_gpu, approximate, layout):
    """Cover arbitrary tails, storage offsets, and materialized strides."""
    elements = 257
    input_backing = torch.linspace(-8.0, 8.0, 2 * elements + 3)
    grad_backing = torch.linspace(2.0, -2.0, 2 * elements + 5)
    if layout == "contiguous_offset":
        input = input_backing[1 : elements + 1]
        grad_output = grad_backing[2 : elements + 2]
        device_input = input_backing.to(mojo_gpu)[1 : elements + 1]
        device_grad_output = grad_backing.to(mojo_gpu)[2 : elements + 2]
    else:
        input = input_backing[1 : 2 * elements + 1 : 2]
        grad_output = grad_backing[2 : 2 * elements + 2 : 2]
        device_input = input_backing.to(mojo_gpu)[1 : 2 * elements + 1 : 2]
        device_grad_output = grad_backing.to(mojo_gpu)[2 : 2 * elements + 2 : 2]

    expected = torch.ops.aten.gelu_backward(grad_output, input, approximate=approximate)
    actual = torch.ops.aten.gelu_backward(
        device_grad_output, device_input, approximate=approximate
    )
    torch.testing.assert_close(actual.cpu(), expected, atol=5e-5, rtol=5e-5)


@pytest.mark.parametrize("approximate", ["none", "tanh"])
@pytest.mark.parametrize("layout", ["contiguous_offset", "strided"])
def test_fast_gelu_backward_bf16_runtime_layouts(mojo_h100, approximate, layout):
    """BF16 covers odd pointer offsets, strides, both formulas, and a tail."""
    elements = 257
    input_backing = torch.linspace(-8.0, 8.0, 2 * elements + 3, dtype=torch.bfloat16)
    grad_backing = torch.linspace(2.0, -2.0, 2 * elements + 5, dtype=torch.bfloat16)
    if layout == "contiguous_offset":
        input = input_backing[1 : elements + 1]
        grad_output = grad_backing[2 : elements + 2]
        device_input = input_backing.to(mojo_h100)[1 : elements + 1]
        device_grad_output = grad_backing.to(mojo_h100)[2 : elements + 2]
    else:
        input = input_backing[1 : 2 * elements + 1 : 2]
        grad_output = grad_backing[2 : 2 * elements + 2 : 2]
        device_input = input_backing.to(mojo_h100)[1 : 2 * elements + 1 : 2]
        device_grad_output = grad_backing.to(mojo_h100)[2 : 2 * elements + 2 : 2]

    expected = torch.ops.aten.gelu_backward(grad_output, input, approximate=approximate)
    actual = torch.ops.aten.gelu_backward(
        device_grad_output, device_input, approximate=approximate
    )
    torch.testing.assert_close(actual.cpu(), expected, atol=2e-2, rtol=2e-2)


@pytest.mark.parametrize("approximate", ["none", "tanh"])
def test_fast_gelu_training_uses_direct_backward(mojo_gpu, monkeypatch, approximate):
    """Autograd must preserve the saved Mojo payload and call Fable's bridge."""
    input = torch.linspace(-8.0, 8.0, 257)
    grad_output = torch.linspace(2.0, -2.0, 257)
    reference = input.clone().requires_grad_()
    reference_output = torch.nn.functional.gelu(reference, approximate=approximate)
    reference_output.backward(grad_output)

    target = ("activation_backward_ops.mojo", "GeluBackwardF32")
    native_calls = _spy_defined_native_calls(monkeypatch, {target})
    actual = input.to(mojo_gpu).requires_grad_()
    actual_output = torch.nn.functional.gelu(actual, approximate=approximate)
    actual_output.backward(grad_output.to(mojo_gpu))

    assert len(native_calls[target]) == 1
    torch.testing.assert_close(actual_output.cpu(), reference_output)
    assert actual.grad is not None
    torch.testing.assert_close(actual.grad.cpu(), reference.grad, atol=5e-5, rtol=5e-5)


@pytest.mark.parametrize("approximate", ["none", "tanh"])
def test_fast_gelu_training_bf16_uses_direct_backward(
    mojo_h100, monkeypatch, approximate
):
    """BF16 autograd must call both dedicated Mojo bridges exactly once."""
    input = torch.linspace(-8.0, 8.0, 257, dtype=torch.bfloat16)
    grad_output = torch.linspace(2.0, -2.0, 257, dtype=torch.bfloat16)
    reference = input.clone().requires_grad_()
    reference_output = torch.nn.functional.gelu(reference, approximate=approximate)
    reference_output.backward(grad_output)

    forward_target = ("activation_forward_ops.mojo", "GeluForwardBF16")
    backward_target = ("activation_backward_ops.mojo", "GeluBackwardBF16")
    native_calls = _spy_defined_native_calls(
        monkeypatch, {forward_target, backward_target}
    )
    actual = input.to(mojo_h100).requires_grad_()
    actual_output = torch.nn.functional.gelu(actual, approximate=approximate)
    actual_output.backward(grad_output.to(mojo_h100))

    assert len(native_calls[forward_target]) == 1
    assert len(native_calls[backward_target]) == 1
    torch.testing.assert_close(
        actual_output.cpu(), reference_output, atol=2e-2, rtol=2e-2
    )
    assert actual.grad is not None
    torch.testing.assert_close(actual.grad.cpu(), reference.grad, atol=2e-2, rtol=2e-2)


def test_fast_gelu_backward_bf16_empty_does_not_enqueue(mojo_h100, monkeypatch):
    """Empty BF16 tensors preserve metadata without launching a GPU kernel."""

    def fail(*_args):
        raise AssertionError("empty GELU backward enqueued a kernel")

    _replace_defined_native_calls(
        monkeypatch, {("activation_backward_ops.mojo", "GeluBackwardBF16"): fail}
    )
    input = torch.empty(0, 7, dtype=torch.bfloat16).to(mojo_h100)
    grad_output = torch.empty_like(input)

    actual = torch.ops.aten.gelu_backward(grad_output, input)

    assert actual.shape == input.shape
    assert actual.dtype == torch.bfloat16
    assert actual.device.type == "mojo"


def test_fast_gelu_backward_rejects_mixed_dtype_before_materialization(
    mojo_h100, monkeypatch
):
    """A mismatched dtype must return NOT_HANDLED without touching storage."""
    from torch_mojo_backend.eager_kernels import aten_fast
    from torch_mojo_backend.mojo_device.torch_mojo_tensor import TorchMojoTensor

    input = torch.randn(17, dtype=torch.bfloat16).to(mojo_h100)
    grad_output = torch.randn(17, dtype=torch.float32).to(mojo_h100)

    def fail_materialization(_self):
        raise AssertionError("mixed-dtype GELU backward materialized an input")

    monkeypatch.setattr(
        TorchMojoTensor, "_materialize_contiguous", fail_materialization
    )
    assert (
        aten_fast.fast_aten_gelu_backward(grad_output, input) is aten_fast.NOT_HANDLED
    )


@pytest.mark.parametrize("value", [False, True])
def test_fast_bool_fill_scalar(mojo_device, value):
    actual = torch.empty(3, 5, dtype=torch.bool).to(mojo_device)
    returned = actual.t().fill_(value)
    assert returned is not actual
    assert returned._ptr == actual._ptr
    torch.testing.assert_close(actual.cpu(), torch.full((3, 5), value))


# One dtype per element width the fill kernels dispatch on: 8, 4, 2 and 1
# bytes. The kernel is layout-only below that -- the value's bits are cast on
# the host -- so widening this list tests the host cast, not the kernel.
FILL_WIDTH_DTYPES = [torch.int64, torch.float32, torch.bfloat16, torch.bool]


def _fill_check(built, dtype, value, mojo_device, base_numel=8192):
    """Fill `built(base)` on both CPU torch and the mojo device; compare the
    WHOLE base, so a kernel that writes outside the view is caught."""
    host = torch.zeros(base_numel, dtype=dtype)
    device = host.clone().to(mojo_device)
    built(host).fill_(value)
    built(device).fill_(value)
    assert torch.equal(device.cpu(), host)


@pytest.mark.parametrize("dtype", FILL_WIDTH_DTYPES)
def test_fast_fill_scalar_ranks_one_through_eight(mojo_device, dtype):
    """Every rank the rank-8 padded layout can carry collapses correctly."""
    value = True if dtype is torch.bool else 3
    for rank in range(1, 9):
        shape = tuple([2] * (rank - 1) + [5])
        numel = 5 * 2 ** (rank - 1)
        _fill_check(
            lambda base, s=shape, n=numel: base[:n].view(s), dtype, value, mojo_device
        )


@pytest.mark.parametrize("dtype", FILL_WIDTH_DTYPES)
def test_fast_fill_scalar_lengths_exercise_the_scalar_tail(mojo_device, dtype):
    """Lengths that are not a whole number of 16-byte stores, plus empty.

    A zero-element fill must be a no-op rather than a launch, and any length
    from 1 to one below the widest vector leaves only the tail.
    """
    value = True if dtype is torch.bool else 3
    for length in (0, 1, 2, 3, 5, 7, 9, 15, 16, 17, 31, 33, 4099):
        _fill_check(lambda base, n=length: base[:n], dtype, value, mojo_device)


@pytest.mark.parametrize("dtype", FILL_WIDTH_DTYPES)
def test_fast_fill_scalar_offset_views_at_every_alignment_phase(mojo_device, dtype):
    """A view can start at any element offset, so the vector width is chosen
    from the runtime base ADDRESS; every 16-byte phase must stay correct."""
    value = True if dtype is torch.bool else 3
    for offset in range(17):
        _fill_check(lambda base, o=offset: base[o : o + 333], dtype, value, mojo_device)


@pytest.mark.parametrize("dtype", FILL_WIDTH_DTYPES)
def test_fast_fill_scalar_strided_layouts(mojo_device, dtype):
    """Layouts the collapse must either flatten or hand to the strided arm.

    The transposed and permuted views are dense, so they must come out
    identical to a contiguous fill; the column, the step and the 2-D slice
    stay genuinely strided; `expand` repeats a stride-0 dimension, whose
    writes are redundant for a constant.
    """
    value = True if dtype is torch.bool else 3
    layouts = [
        lambda base: base[:6000].view(60, 100).t(),
        lambda base: base[:6000].view(60, 100)[:, 3],
        lambda base: base[:6000:7],
        lambda base: base[:6000].view(60, 100)[10:50, 20:90],
        lambda base: base[:5040].view(6, 7, 8, 15).permute(2, 0, 3, 1),
        lambda base: base[:100].view(1, 100).expand(7, 100),
        lambda base: base[:1].view(()),
    ]
    for build in layouts:
        _fill_check(build, dtype, value, mojo_device)


@pytest.mark.parametrize("dtype", FILL_WIDTH_DTYPES)
def test_fast_zero__delegates_to_the_fill_bridge(mojo_device, dtype):
    """aten::zero_ is aten::fill_.Scalar with 0, on any layout."""
    host = torch.ones(1234, dtype=dtype)
    device = host.clone().to(mojo_device)
    host.view(2, 617).t().zero_()
    device.view(2, 617).t().zero_()
    assert torch.equal(device.cpu(), host)


@pytest.mark.parametrize(
    "dtype", [torch.float32, torch.bfloat16, torch.int64, torch.int32, torch.uint8]
)
def test_fast_fill_scalar_value_conversion_matches_cpu(mojo_device, dtype):
    """The scalar is narrowed to the tensor's dtype exactly as ATen does:
    floats truncate into integer tensors, and `bool` -- a Python `int`
    subclass ATen accepts for every scalar fill -- lands as 0 or 1."""
    values = [0, 7, True, False, 0.0, 2.75]
    if dtype is not torch.uint8:
        values += [-1.5, -3]
    if dtype.is_floating_point:
        values += [float("inf"), float("-inf"), float("nan")]
    for value in values:
        host = torch.empty(33, dtype=dtype).fill_(value)
        device = torch.empty(33, dtype=dtype).to(mojo_device).fill_(value).cpu()
        assert torch.equal(device.isnan(), host.isnan())
        assert torch.equal(device.nan_to_num(0.0), host.nan_to_num(0.0))


def test_fast_gpu_portability_kernels(mojo_device):
    # These closures previously captured Float64 or instantiated host-only
    # code for the GPU target, which is rejected by Metal and gfx942.
    base = torch.arange(6, dtype=torch.float32).reshape(2, 3)
    device_base = base.to(mojo_device)
    device_base.t().fill_(2.5)
    torch.testing.assert_close(device_base.cpu(), torch.full_like(base, 2.5))

    index = torch.tensor([[0, 2, 1], [1, 0, 2]])
    source = torch.tensor([[10.0, 20.0, 30.0], [40.0, 50.0, 60.0]])
    scattered = (
        torch.zeros_like(source)
        .to(mojo_device)
        .scatter(1, index.to(mojo_device), source.to(mojo_device))
    )
    torch.testing.assert_close(
        scattered.cpu(), torch.zeros_like(source).scatter(1, index, source)
    )

    a = torch.randn(2, 3)
    b = torch.randn(2, 3)
    c = torch.randn(2, 3)
    result = torch.addcmul(
        a.to(mojo_device), b.to(mojo_device), c.to(mojo_device), value=0.125
    )
    torch.testing.assert_close(result.cpu(), torch.addcmul(a, b, c, value=0.125))


def test_fast_add_scalar_int(mojo_device):
    x = torch.arange(6)
    torch.testing.assert_close((x.to(mojo_device) + 3).cpu(), x + 3)


def test_fast_add_inplace(mojo_device):
    x = torch.randn(4, 4)
    y = torch.randn(4, 4)
    xd = x.clone().to(mojo_device)
    xd += y.to(mojo_device)
    torch.testing.assert_close(xd.cpu(), x + y)


def test_fast_all_and_item(mojo_device):
    ones = torch.ones(1, 6, dtype=torch.bool).to(mojo_device)
    assert bool(ones.all().item()) is True
    mixed = torch.tensor([[True, False, True]]).to(mojo_device)
    assert bool(mixed.all().item()) is False


def test_fast_arange(mojo_device):
    torch.testing.assert_close(
        torch.arange(6, device=mojo_device).cpu(), torch.arange(6)
    )
    torch.testing.assert_close(
        torch.arange(2, 20, 3, device=mojo_device).cpu(), torch.arange(2, 20, 3)
    )


def test_fast_arange_uses_device_accumulator(mojo_device):
    args = (16_777_217.0, 16_777_227.0, 1.0)
    result = torch.arange(*args, dtype=torch.float32, device=mojo_device).cpu()
    cpu_index = len(list(get_accelerators())) - 1
    if mojo_device == f"mojo:{cpu_index}":
        # PyTorch's CPU kernel specifies a float64 accumulator for float32.
        # Build that scalar reference explicitly: arm64's vectorized kernel
        # has platform-specific intermediate rounding at this boundary.
        expected = torch.tensor(
            [args[0] + i * args[2] for i in range(10)], dtype=torch.float32
        )
    elif torch.cuda.is_available():
        expected = torch.arange(*args, dtype=torch.float32, device="cuda").cpu()
    elif torch.backends.mps.is_available():
        expected = torch.arange(*args, dtype=torch.float32, device="mps").cpu()
    else:
        pytest.skip("no native GPU reference for MAX accelerator")
    assert torch.equal(result, expected)


def test_fast_cast(mojo_device):
    x = torch.randint(0, 3, (1, 6))
    torch.testing.assert_close(x.to(mojo_device).to(torch.bool).cpu(), x.to(torch.bool))
    f = torch.randn(3, 4)
    torch.testing.assert_close(
        f.to(mojo_device).to(torch.float16).cpu(), f.to(torch.float16)
    )


# The dtypes `CAST_DTYPES` in data_movement_ops.mojo dispatches on, both ends.
_FAST_CAST_DTYPES = [
    torch.float32,
    torch.float16,
    torch.bfloat16,
    torch.int64,
    torch.int32,
    torch.uint8,
    torch.bool,
]


@pytest.mark.parametrize("numel", [1, 3, 17, 1027, 4099])
@pytest.mark.parametrize("offset", [0, 1, 2, 3])
def test_fast_cast_is_exact_for_every_dtype_pair(mojo_device, numel, offset):
    """Every `CAST_DTYPES` pair, element for element, off a shifted base.

    The cast kernel moves several elements per thread in the widest vector the
    two base addresses admit, so this pins the three things that makes fragile:
    a count that is not a multiple of the vector (the scalar tail), a base
    address that is not vector-aligned (`offset`, which forces a narrower
    width or the scalar arm), and the `!= 0` bool destination. The pattern
    steps modulo 5, coprime with every power-of-two width, so a vector whose
    lanes are rotated or whose tail is left unwritten cannot pass.
    """
    values = torch.arange(numel + offset) % 5
    for src in _FAST_CAST_DTYPES:
        source = (values != 0) if src is torch.bool else values.to(src)
        on_device = source.to(mojo_device)
        view = on_device[offset : offset + numel]
        assert view.is_contiguous()
        expected_view = source[offset : offset + numel]
        for dst in _FAST_CAST_DTYPES:
            got = view.to(dst).cpu()
            assert torch.equal(got, expected_view.to(dst)), (
                f"{src} -> {dst}, numel={numel}, offset={offset}"
            )


@pytest.mark.parametrize("numel", [1027, 65_539])
def test_fast_cast_float_rounding_matches_cpu(mojo_device, numel):
    """Narrowing a float must round exactly as the CPU does, tail included."""
    generator = torch.Generator().manual_seed(numel)
    values = torch.randn(numel + 3, generator=generator)
    for offset in (0, 1, 3):
        on_device = values.to(mojo_device)[offset : offset + numel]
        expected = values[offset : offset + numel]
        for dst in (torch.bfloat16, torch.float16):
            assert torch.equal(on_device.to(dst).cpu(), expected.to(dst)), (
                f"float32 -> {dst}, numel={numel}, offset={offset}"
            )
            round_trip = on_device.to(dst).to(torch.float32).cpu()
            assert torch.equal(round_trip, expected.to(dst).to(torch.float32))


def test_fast_float64_factories_fill_scatter_and_arange(mojo_gpu):
    if list(get_accelerators())[0].api == "metal":
        pytest.skip("Metal does not support float64 kernels")

    ones = torch.ones(5, dtype=torch.float64, device=mojo_gpu)
    torch.testing.assert_close(ones.cpu(), torch.ones(5, dtype=torch.float64))

    values = torch.arange(5, dtype=torch.float64).to(mojo_gpu)
    values.fill_(2.5)
    torch.testing.assert_close(values.cpu(), torch.full((5,), 2.5, dtype=torch.float64))

    base = torch.zeros(5, dtype=torch.float64).to(mojo_gpu)
    index = torch.tensor([1, 3], dtype=torch.int64).to(mojo_gpu)
    source = torch.tensor([4.0, 7.0], dtype=torch.float64).to(mojo_gpu)
    scattered = base.scatter(0, index, source).cpu()
    torch.testing.assert_close(
        scattered, torch.tensor([0.0, 4.0, 0.0, 7.0, 0.0], dtype=torch.float64)
    )

    result = torch.arange(0.0, 2.0, 0.25, dtype=torch.float64, device=mojo_gpu)
    torch.testing.assert_close(
        result.cpu(), torch.arange(0.0, 2.0, 0.25, dtype=torch.float64)
    )


# ---- GPU-only fast paths (matmul / conv / attention via MAX kernel library)


def test_fast_mm_addmm(mojo_gpu):
    a = torch.randn(6, 768)
    b = torch.randn(768, 2304)
    bias = torch.randn(2304)
    dev = torch.addmm(bias.to(mojo_gpu), a.to(mojo_gpu), b.to(mojo_gpu)).cpu()
    # TF32-level tolerance: the MAX matmul kernels (same as graph mode) use
    # tensor cores for float32.
    torch.testing.assert_close(dev, torch.addmm(bias, a, b), atol=5e-2, rtol=5e-2)
    dev = (a.to(mojo_gpu) @ b.to(mojo_gpu)).cpu()
    torch.testing.assert_close(dev, a @ b, atol=5e-2, rtol=5e-2)


# Every layout arm of `_matmul_spec_operands_launch`: Python hands strided
# operands straight to it, so a copy-free route reading the wrong strides -- or
# a missing scratch copy -- shows up here rather than as a wrong training loss.
# Queued and inline launches both, because a queued launch cannot fall back.
def _strided_matmul_cases(device: torch.device) -> list[tuple[str, object, object]]:
    def to(x: torch.Tensor) -> torch.Tensor:
        return x.to(device)

    dense_a = torch.randn(64, 48)
    dense_b = torch.randn(48, 32)
    wide = torch.randn(48, 128)
    batched = torch.randn(4, 24, 32)
    batched_b = torch.randn(4, 24, 16)
    return [
        # arm 2: contiguous A, strided B
        ("strided_b", to(dense_a), to(wide)[:, ::2]),
        # arm 3: strided A, contiguous B -- the weight-gradient shape, and the
        # one the gfx942 TN and Apple TA routes claim
        ("transposed_a", to(torch.randn(48, 64)).t(), to(dense_b)),
        # arm 3 with an offset view, so the route may not assume offset 0
        ("narrowed_a", to(torch.randn(80, 64))[8:56].t(), to(dense_b)),
        # arm 4: both strided
        ("both_strided", to(torch.randn(48, 64)).t(), to(wide)[:, ::2]),
        # stride-0 broadcast reads
        ("expanded_a", to(torch.randn(1, 48)).expand(64, 48), to(dense_b)),
        # rank>2 activation with a strided leading dim
        ("rank3_strided", to(torch.randn(8, 64, 48))[::2], to(dense_b)),
        # batched, with the two matmul dims transposed
        ("bmm_transposed", to(batched).transpose(1, 2), to(batched_b)),
    ]


@pytest.mark.parametrize("queued", [False, True])
def test_matmul_every_strided_layout_arm(mojo_gpu, monkeypatch, queued):
    monkeypatch.setenv("TORCH_MOJO_BACKEND_KERNEL_QUEUE", "1" if queued else "0")
    for label, a, b in _strided_matmul_cases(mojo_gpu):
        expected = a.cpu() @ b.cpu()
        got = (a @ b).cpu()
        torch.testing.assert_close(
            got,
            expected,
            atol=5e-2,
            rtol=5e-2,
            msg=lambda m, label=label: f"{label}: {m}",
        )


def test_addmm_strided_bias_and_operands(mojo_gpu):
    a = torch.randn(48, 64).to(mojo_gpu).t()
    b = torch.randn(48, 32).to(mojo_gpu)
    bias = torch.randn(64).to(mojo_gpu)[::2]
    got = torch.addmm(bias, a, b).cpu()
    torch.testing.assert_close(
        got, torch.addmm(bias.cpu(), a.cpu(), b.cpu()), atol=5e-2, rtol=5e-2
    )


def test_matmul_float64_raises_at_the_call_and_not_at_the_drain(mojo_gpu):
    # The Mojo matmul carries no float64 instantiation.  Declining it in Python
    # turns what used to be a plausible-looking tensor plus an "unsupported
    # dtype" raised much later, at the next drain, into an error at the call
    # that caused it.
    a = torch.randn(32, 24, dtype=torch.float64).to(mojo_gpu)
    b = torch.randn(24, 16, dtype=torch.float64).to(mojo_gpu)
    with pytest.raises(NotImplementedError, match="aten::mm"):
        (a @ b).cpu()


@pytest.mark.parametrize(
    "in_features,out_features", [(768, 2304), (4096, 1024), (992, 3001), (768, 50257)]
)
def test_fast_linear_gfx942_dynamic_mfma(mojo_gpu, in_features, out_features):
    if list(get_accelerators())[0].architecture_name != "gfx942":
        pytest.skip("the dynamic MFMA kernels target gfx942")

    x = torch.randn(256, in_features)
    weight = torch.randn(out_features, in_features)
    bias = torch.randn(out_features)
    dev = torch.nn.functional.linear(
        x.to(mojo_gpu), weight.to(mojo_gpu), bias.to(mojo_gpu)
    ).cpu()
    ref = torch.nn.functional.linear(x, weight, bias)
    torch.testing.assert_close(dev, ref, atol=5e-2, rtol=5e-2)


@pytest.mark.parametrize(
    "in_features,out_features", [(768, 768), (1024, 4096), (4096, 1024), (992, 3001)]
)
def test_fast_addmm_gfx942_dynamic_mfma(mojo_gpu, in_features, out_features):
    if list(get_accelerators())[0].architecture_name != "gfx942":
        pytest.skip("the dynamic MFMA kernels target gfx942")

    x = torch.randn(256, in_features)
    weight = torch.randn(in_features, out_features)
    bias = torch.randn(out_features)
    dev_x = x.to(mojo_gpu)
    dev_weight = weight.to(mojo_gpu)
    dev_bias = bias.to(mojo_gpu)
    # Queue repeated launches before synchronizing. This catches invalid
    # tile schedules whose shared-memory race is hidden by a single launch.
    dev_outputs = [torch.addmm(dev_bias, dev_x, dev_weight) for _ in range(3)]
    dev = [output.cpu() for output in dev_outputs]
    ref = torch.addmm(bias, x, weight)
    for actual in dev:
        torch.testing.assert_close(actual, ref, atol=5e-2, rtol=5e-2)
    assert torch.equal(dev[0], dev[1])
    assert torch.equal(dev[0], dev[2])


@pytest.mark.parametrize("batch", [64, 257, 512])
def test_fast_addmm_gfx942_dynamic_batch_mfma(mojo_gpu, batch):
    if list(get_accelerators())[0].architecture_name != "gfx942":
        pytest.skip("the dynamic MFMA kernels target gfx942")

    # A K-dominant projection selects that shape regime without embedding
    # these dimensions in the kernel. The non-tile-aligned M covers its edge.
    x = torch.randn(batch, 4096)
    weight = torch.randn(4096, 1024)
    bias = torch.randn(1024)
    dev_x = x.to(mojo_gpu)
    dev_weight = weight.to(mojo_gpu)
    dev_bias = bias.to(mojo_gpu)
    outputs = [torch.addmm(dev_bias, dev_x, dev_weight) for _ in range(3)]
    actual = [output.cpu() for output in outputs]
    ref = torch.addmm(bias, x, weight)
    for output in actual:
        torch.testing.assert_close(output, ref, atol=5e-2, rtol=5e-2)
    assert torch.equal(actual[0], actual[1])
    assert torch.equal(actual[0], actual[2])


def test_fast_addmm_gfx942_unaligned_k(mojo_gpu):
    if list(get_accelerators())[0].architecture_name != "gfx942":
        pytest.skip("the dynamic MFMA dispatch targets gfx942")

    # K values outside the MFMA tile-alignment regime retain the general
    # dynamic GEMM path; this is a regime fallback, not a model-shape gate.
    x = torch.randn(65, 1000)
    weight = torch.randn(1000, 257)
    bias = torch.randn(257)
    actual = torch.addmm(bias.to(mojo_gpu), x.to(mojo_gpu), weight.to(mojo_gpu)).cpu()
    torch.testing.assert_close(
        actual, torch.addmm(bias, x, weight), atol=5e-2, rtol=5e-2
    )


def test_fast_gpt2_decode_attention_with_strided_kv(mojo_gpu):
    batch, heads, seq_len, capacity, head_dim = 4, 12, 8, 16, 64
    query = torch.randn(batch, heads, 1, head_dim)
    key_storage = torch.randn(batch, heads, capacity, head_dim)
    value_storage = torch.randn(batch, heads, capacity, head_dim)
    key = key_storage[:, :, :seq_len, :]
    value = value_storage[:, :, :seq_len, :]

    dev_key_storage = key_storage.to(mojo_gpu)
    dev_value_storage = value_storage.to(mojo_gpu)
    dev_key = dev_key_storage[:, :, :seq_len, :]
    dev_value = dev_value_storage[:, :, :seq_len, :]
    actual = torch.nn.functional.scaled_dot_product_attention(
        query.to(mojo_gpu), dev_key, dev_value
    ).cpu()
    ref = torch.nn.functional.scaled_dot_product_attention(query, key, value)
    torch.testing.assert_close(actual, ref, atol=2e-4, rtol=2e-4)


def test_fast_gpt2_logits_argmax(mojo_gpu):
    if list(get_accelerators())[0].architecture_name != "gfx942":
        pytest.skip("the GPT-2 argmax specialization targets gfx942")

    logits = torch.randn(256, 50257)
    logits[:, 123] = 100.0
    actual = torch.argmax(logits.to(mojo_gpu), dim=-1).cpu()
    torch.testing.assert_close(actual, torch.argmax(logits, dim=-1))


# ---------------------------------------------------------------------------
# argmin / argmax (argreduce_kernels.mojo)
#
# The arg-reduction splits the reduce axis across blocks and merges partials
# out of order, so every one of these checks a rule that an out-of-order merge
# could break: torch answers with the FIRST occurrence of the extremum, and a
# NaN anywhere beats every number.  Random data alone never produces a tie, so
# the tie cases are built by hand and placed on both sides of the split
# boundaries.
# ---------------------------------------------------------------------------

_ARGREDUCE_FNS = (torch.argmax, torch.argmin)
# Sizes that straddle the split decision (AR_MIN_CHUNK = 4096 elements is the
# smallest slice worth its own block): one block and a ragged vector tail, one
# block exactly, the first size that splits in two, and one that splits as many
# ways as the device has room for.
_ARGREDUCE_SPLIT_SHAPES = (4095, 4096, 8193, 1 << 20)


def _assert_argreduce_matches(
    device: str, cpu_tensor: torch.Tensor, dim: int | None = None
) -> None:
    device_tensor = cpu_tensor.to(device)
    for fn in _ARGREDUCE_FNS:
        expected = fn(cpu_tensor) if dim is None else fn(cpu_tensor, dim=dim)
        actual = (
            fn(device_tensor) if dim is None else fn(device_tensor, dim=dim)
        ).cpu()
        assert actual.dtype == expected.dtype
        assert torch.equal(actual, expected), f"{fn.__name__} dim={dim}"


@pytest.mark.parametrize("size", _ARGREDUCE_SPLIT_SHAPES)
def test_fast_argreduce_full_reduction_across_split_sizes(
    mojo_gpu: str, size: int
) -> None:
    """A full reduction is one output: it always takes the split path."""
    generator = torch.Generator().manual_seed(20260811)
    _assert_argreduce_matches(mojo_gpu, torch.randn(size, generator=generator))


@pytest.mark.parametrize("size", _ARGREDUCE_SPLIT_SHAPES)
def test_fast_argreduce_ties_take_the_lowest_index(mojo_gpu: str, size: int) -> None:
    """Every element equal, and the extremum repeated at known positions:
    both must answer with the FIRST of them however the axis was split."""
    _assert_argreduce_matches(mojo_gpu, torch.full((size,), 3.5))

    marked = torch.zeros(size)
    for position in (0, 1, size // 3, size // 2, size - 1):
        marked[position] = 1.0
    _assert_argreduce_matches(mojo_gpu, marked)  # first max at 0, first min at 2

    marked = torch.zeros(size)
    for position in (2, size // 5, size - 2):
        marked[position] = -1.0
    _assert_argreduce_matches(mojo_gpu, marked)


@pytest.mark.parametrize("size", (1000, 1 << 20))
def test_fast_argreduce_nan_beats_every_number(mojo_gpu: str, size: int) -> None:
    """torch propagates NaN: the index of the first NaN is the answer."""
    generator = torch.Generator().manual_seed(20260811)
    for position in (0, 3, size // 2, size - 1):
        values = torch.randn(size, generator=generator)
        values[position] = float("nan")
        _assert_argreduce_matches(mojo_gpu, values)

    two_nans = torch.randn(size, generator=generator)
    two_nans[5] = float("nan")
    two_nans[size - 5] = float("nan")
    _assert_argreduce_matches(mojo_gpu, two_nans)
    _assert_argreduce_matches(mojo_gpu, torch.full((size,), float("nan")))


@pytest.mark.parametrize("size", (1000, 1 << 20))
def test_fast_argreduce_saturated_identity_values(mojo_gpu: str, size: int) -> None:
    """A row of the identity element (-inf for argmax, +inf for argmin) still
    has an answer: index 0.  A lane seeded with the identity and a strict
    comparison would report "nothing found" here."""
    _assert_argreduce_matches(mojo_gpu, torch.full((size,), float("-inf")))
    _assert_argreduce_matches(mojo_gpu, torch.full((size,), float("inf")))


@pytest.mark.parametrize(
    "dtype", [torch.float32, torch.float16, torch.bfloat16, torch.int32, torch.int64]
)
def test_fast_argreduce_dtypes(mojo_gpu: str, dtype: torch.dtype) -> None:
    generator = torch.Generator().manual_seed(20260811)
    if dtype.is_floating_point:
        values = torch.randn(3, 5000, generator=generator).to(dtype)
    else:
        values = torch.randint(-1000, 1000, (3, 5000), generator=generator, dtype=dtype)
    _assert_argreduce_matches(mojo_gpu, values)
    _assert_argreduce_matches(mojo_gpu, values, dim=1)
    _assert_argreduce_matches(mojo_gpu, values, dim=0)


@pytest.mark.parametrize(
    ("shape", "dim"),
    [
        ((4096, 8), 0),  # inner below the coalescing floor: materialized
        ((4096, 16), 0),  # first inner extent the strided kernel takes
        ((4096, 33), 0),  # ragged tail column
        ((357, 789), 0),  # awkward, and one block per column tile
        ((5, 7, 33), 1),  # middle dim of a rank-3 tensor
        ((2, 3, 4, 5), 1),
        ((1 << 12, 1 << 12), 0),  # wide enough to split the strided axis
        ((70000, 2, 32), 1),  # outer past the 65535 cap on grid.y / grid.z
    ],
)
def test_fast_argreduce_strided_axis(
    mojo_gpu: str, shape: tuple[int, ...], dim: int
) -> None:
    """Non-trailing reduce dims: the strided kernel reads the source in place
    above the coalescing floor and the materialized route runs below it, and
    both must agree with torch (including on ties)."""
    generator = torch.Generator().manual_seed(20260811)
    _assert_argreduce_matches(
        mojo_gpu, torch.randn(shape, generator=generator), dim=dim
    )
    _assert_argreduce_matches(mojo_gpu, torch.zeros(shape), dim=dim)

    with_nan = torch.randn(shape, generator=generator)
    with_nan[(0,) * (len(shape) - 1) + (1,)] = float("nan")
    _assert_argreduce_matches(mojo_gpu, with_nan, dim=dim)


def test_fast_argreduce_strided_direct_gate_matches_the_kernel_regime(
    mojo_gpu: str,
) -> None:
    """The Python gate and the Mojo kernel must agree about which layouts go
    in place: a queued launch cannot fall back."""
    from torch_mojo_backend.eager_kernels.aten_fast import (
        _ARG_DIRECT_MIN_INNER,
        _arg_strided_direct_ok,
        _t,
    )

    contiguous = _t(torch.randn(8, _ARG_DIRECT_MIN_INNER).to(mojo_gpu))
    assert _arg_strided_direct_ok("ArgminSpec", contiguous, (0,))
    assert _arg_strided_direct_ok("ArgmaxSpec", contiguous, (0,))
    # trailing dims belong to the contiguous-axis kernels
    assert not _arg_strided_direct_ok("ArgmaxSpec", contiguous, (1,))
    # every other spec op keeps its own route
    assert not _arg_strided_direct_ok("SumSpec", contiguous, (0,))
    # below the coalescing floor, and non-contiguous operands, materialize
    narrow = _t(torch.randn(8, _ARG_DIRECT_MIN_INNER - 1).to(mojo_gpu))
    assert not _arg_strided_direct_ok("ArgmaxSpec", narrow, (0,))
    strided = _t(torch.randn(8, 64).to(mojo_gpu)[:, ::2])
    assert not _arg_strided_direct_ok("ArgmaxSpec", strided, (0,))
    # a non-adjacent dim interval is not a (outer, reduce, inner) view
    cube = _t(torch.randn(4, 5, 64).to(mojo_gpu))
    assert not _arg_strided_direct_ok("ArgmaxSpec", cube, (0, 2))


def test_fast_argreduce_views_and_keepdim(mojo_gpu: str) -> None:
    generator = torch.Generator().manual_seed(20260811)
    base = torch.randn(64, 128, generator=generator)
    for view in (base.t(), base[:, 3:70], base[::2, ::3]):
        for dim in (None, 0, 1):
            _assert_argreduce_matches(mojo_gpu, view, dim=dim)

    values = torch.randn(8, 300, 17, generator=generator)
    device_values = values.to(mojo_gpu)
    for dim in (0, 1, 2):
        for fn in _ARGREDUCE_FNS:
            expected = fn(values, dim=dim, keepdim=True)
            actual = fn(device_values, dim=dim, keepdim=True).cpu()
            assert actual.shape == expected.shape
            assert torch.equal(actual, expected)


def test_fast_bmm(mojo_gpu):
    a = torch.randn(12, 6, 64)
    b = torch.randn(12, 64, 6)
    dev = torch.bmm(a.to(mojo_gpu), b.to(mojo_gpu)).cpu()
    torch.testing.assert_close(dev, torch.bmm(a, b), atol=1e-2, rtol=1e-2)


@pytest.mark.parametrize("dtype", [torch.float32, torch.float16, torch.bfloat16])
@pytest.mark.parametrize(
    ("shape", "dim"),
    [
        ((33, 257), -1),  # odd cols: rows off 16B alignment take head/tail
        ((16, 4096), -1),  # pure 16B-aligned vector body
        ((2, 3, 129), 2),  # rank-3 trailing dim flattens without a view
    ],
)
def test_fast_log_softmax_backward_fused_matches_reference(mojo_gpu, dtype, shape, dim):
    """The fused trailing-dim kernel must match fp32-accumulated math."""
    generator = torch.Generator().manual_seed(20260722)
    source = torch.randn(shape, generator=generator)
    output = torch.log_softmax(source, dim=dim).to(dtype)
    grad_output = torch.randn(shape, generator=generator).to(dtype)
    expected = (
        grad_output.float()
        - output.float().exp() * grad_output.float().sum(dim=dim, keepdim=True)
    ).to(dtype)

    actual = torch.ops.aten._log_softmax_backward_data(
        grad_output.to(mojo_gpu), output.to(mojo_gpu), dim, dtype
    )

    assert actual.dtype == dtype
    tol = 2e-2 if dtype in (torch.bfloat16, torch.float16) else 2e-5
    torch.testing.assert_close(actual.cpu(), expected, atol=tol, rtol=tol)


def test_fast_log_softmax_backward_uses_fused_kernel(mojo_gpu, monkeypatch):
    """Contiguous trailing-dim same-dtype autograd must call the bridge once."""
    target = ("softmax_backward_ops.mojo", "LogSoftmaxBackwardData")
    native_calls = _spy_defined_native_calls(monkeypatch, {target})

    source = torch.randn(8, 640)
    grad_output = torch.randn(8, 640)
    reference = source.clone().requires_grad_()
    reference_output = torch.log_softmax(reference, dim=-1)
    reference_output.backward(grad_output)

    actual = source.to(mojo_gpu).requires_grad_()
    actual_output = torch.log_softmax(actual, dim=-1)
    actual_output.backward(grad_output.to(mojo_gpu))

    assert len(native_calls[target]) == 1
    assert actual.grad is not None
    torch.testing.assert_close(actual.grad.cpu(), reference.grad, atol=2e-5, rtol=2e-5)


def test_fast_log_softmax_backward_non_trailing_keeps_composed_path(
    mojo_gpu, monkeypatch
):
    """Non-trailing dims and the f32->f16 promotion stay on the composed path."""

    def fail(*_args):
        raise AssertionError("composed-path case reached the fused kernel")

    _replace_defined_native_calls(
        monkeypatch, {("softmax_backward_ops.mojo", "LogSoftmaxBackwardData"): fail}
    )

    source = torch.randn(6, 33, 5)
    output = torch.log_softmax(source, dim=1)
    grad_output = torch.randn(6, 33, 5)
    expected = torch.ops.aten._log_softmax_backward_data(
        grad_output, output, 1, torch.float32
    )
    actual = torch.ops.aten._log_softmax_backward_data(
        grad_output.to(mojo_gpu), output.to(mojo_gpu), 1, torch.float32
    )
    torch.testing.assert_close(actual.cpu(), expected, atol=2e-5, rtol=2e-5)

    # The CPU op rejects the f32-grad -> f16-target promotion, so compute
    # the reference directly (fp32 math, one final rounding).
    half_source = torch.randn(6, 40)
    half_output = torch.log_softmax(half_source, dim=-1)
    half_grad = torch.randn(6, 40)
    expected_half = (
        half_grad - half_output.exp() * half_grad.sum(dim=-1, keepdim=True)
    ).to(torch.float16)
    actual_half = torch.ops.aten._log_softmax_backward_data(
        half_grad.to(mojo_gpu), half_output.to(mojo_gpu), -1, torch.float16
    )
    assert actual_half.dtype == torch.float16
    torch.testing.assert_close(actual_half.cpu(), expected_half, atol=2e-3, rtol=2e-3)


# Column counts that walk the wide-row log-softmax dispatch. Rows above
# LSM_BIG_ROW_BYTES take the 1024-thread arm, whose grid is the L2 budget capped
# below by a device-filling floor; these counts straddle the point where the
# budget stops being the binding term. Non-multiples of every vector width are
# deliberate, and they make the per-row head and tail non-empty.
_WIDE_LSM_COLS = [12_501, 16_384, 16_385, 25_000, 33_000, 50_304, 70_001]


@pytest.mark.parametrize("dtype", [torch.float32, torch.float16, torch.bfloat16])
@pytest.mark.parametrize("cols", _WIDE_LSM_COLS)
@pytest.mark.parametrize("offset", [0, 1])
def test_fast_log_softmax_wide_rows_match_cpu(mojo_gpu, dtype, cols, offset):
    """Wide rows, both grid regimes, against CPU log_softmax.

    `offset` slices the storage so the row bases stop being 16-byte aligned,
    which is the only way to exercise the per-row scalar head and tail.
    """
    rows = 3
    generator = torch.Generator().manual_seed(20260726)
    flat = torch.randn(rows * cols + offset, generator=generator)
    expected = torch.log_softmax(flat[offset:].view(rows, cols).to(dtype), dim=-1)

    device_flat = flat.to(mojo_gpu).to(dtype)
    actual = torch.log_softmax(device_flat[offset:].view(rows, cols), dim=-1)

    assert actual.dtype == dtype
    assert not actual.isnan().any().item()
    tol = 3e-2 if dtype in (torch.bfloat16, torch.float16) else 2e-5
    torch.testing.assert_close(actual.cpu(), expected, atol=tol, rtol=tol)


@pytest.mark.parametrize("dtype", [torch.float32, torch.bfloat16])
@pytest.mark.parametrize("cols", [12_501, 33_000, 50_304, 70_001])
def test_fast_log_softmax_wide_rows_stay_finite(mojo_gpu, dtype, cols):
    """A wide row whose maximum is huge, and one that is entirely constant.

    The block maximum has to come from a finite sentinel: seeding it with
    `Float32.MIN` (which is -inf) turns an idle thread's `0 * exp(m - m)` into
    a NaN that the block reduction spreads over the whole row (defect D7). A
    constant row makes every log-probability -log(cols), which is the cheapest
    independent check that the denominator is the full row and not a fragment.
    """
    spiked = torch.full((1, cols), -3.0, dtype=dtype)
    spiked[0, cols // 3] = 60.0
    actual = torch.log_softmax(spiked.to(mojo_gpu), dim=-1)
    assert not actual.isnan().any().item()
    torch.testing.assert_close(
        actual.cpu(), torch.log_softmax(spiked, dim=-1), atol=3e-2, rtol=3e-2
    )

    constant = torch.full((2, cols), 0.25, dtype=dtype)
    flat_out = torch.log_softmax(constant.to(mojo_gpu), dim=-1).cpu().float()
    want = torch.full((2, cols), -math.log(cols))
    torch.testing.assert_close(flat_out, want, atol=3e-2, rtol=3e-2)


@pytest.mark.parametrize("dtype", [torch.float32, torch.float16, torch.bfloat16])
@pytest.mark.parametrize("cols", [3, 33_000, 50_304, 70_001])
@pytest.mark.parametrize("offset", [0, 1])
def test_fast_log_softmax_backward_wide_rows(mojo_gpu, dtype, cols, offset):
    """The register-staged backward arms, and the re-read fallback.

    cols == 3 is below one 16-byte vector, so the vector body is empty and the
    whole row is head/tail scalars (the 1-slot arm); the wide counts walk the
    4- and 8-slot arms and, at 70001, the fallback.
    """
    rows = 3
    generator = torch.Generator().manual_seed(20260726)
    flat_source = torch.randn(rows * cols + offset, generator=generator)
    flat_grad = torch.randn(rows * cols + offset, generator=generator)
    source = flat_source[offset:].view(rows, cols)
    grad = flat_grad[offset:].view(rows, cols).to(dtype)
    output = torch.log_softmax(source, dim=-1).to(dtype)
    expected = (
        grad.float() - output.float().exp() * grad.float().sum(-1, keepdim=True)
    ).to(dtype)

    device_grad = flat_grad.to(mojo_gpu).to(dtype)[offset:].view(rows, cols)
    device_output = output.reshape(-1).to(mojo_gpu).view(rows, cols)
    actual = torch.ops.aten._log_softmax_backward_data(
        device_grad, device_output, -1, dtype
    )

    assert actual.dtype == dtype
    assert not actual.isnan().any().item()
    tol = 3e-2 if dtype in (torch.bfloat16, torch.float16) else 2e-5
    torch.testing.assert_close(actual.cpu(), expected, atol=tol, rtol=tol)


def test_fast_binary_add_above_last_level_cache(mojo_gpu):
    """The flat 16-byte binary kernel's streaming grid arm.

    `_bw_flat_blocks` covers the vector slots exactly once the three operands
    exceed the 256 MiB last-level cache, and keeps the 4096-block cap below it.
    24000003 fp32 elements is 288 MB of traffic and not a multiple of the
    4-element vector, so the scalar tail rides on the streaming grid too.
    """
    total = 24_000_003
    left = torch.arange(total, dtype=torch.float32) % 1021 - 510.0
    right = torch.arange(total, dtype=torch.float32) % 733 - 366.0
    actual = (left.to(mojo_gpu) + right.to(mojo_gpu)).cpu()
    torch.testing.assert_close(actual, left + right)


# The (dtype -> out_dtype) vectorized kernels: a comparison reads 16 bytes per
# operand and writes a 1-byte mask, so its vector width, its ragged tail and
# its alignment gate are all different from the same-dtype arithmetic case.
# Sizes below straddle the vector width (16 // itemsize) in both directions;
# the offset views are what catches a gate written on `total % VW` instead of
# on the runtime base ADDRESS, which is the failure mode that silently
# corrupts rather than faulting.
_MASK_SIZES = [0, 1, 3, 4, 5, 15, 16, 17, 255, 256, 257, 4095, 100_003]
_COMPARE_OPS = {
    "eq": torch.eq,
    "ne": torch.ne,
    "lt": torch.lt,
    "le": torch.le,
    "gt": torch.gt,
    "ge": torch.ge,
}


def _mask_operands(n: int, dtype: torch.dtype) -> tuple[torch.Tensor, torch.Tensor]:
    """Deterministic operands that overlap on a third of their elements, so
    equality is exercised rather than being vacuously false everywhere."""
    if dtype.is_floating_point:
        left = (torch.arange(n, dtype=torch.float32) % 97 - 48.0).to(dtype)
        right = (torch.arange(n, dtype=torch.float32) % 61 - 30.0).to(dtype)
    else:
        left = (torch.arange(n) % 97 - 48).to(dtype)
        right = (torch.arange(n) % 61 - 30).to(dtype)
    right[: n // 3] = left[: n // 3]
    return left, right


@pytest.mark.parametrize("op_name", sorted(_COMPARE_OPS))
def test_fast_comparison_vector_width_boundaries_match_cpu(mojo_gpu, op_name):
    op = _COMPARE_OPS[op_name]
    for n in _MASK_SIZES:
        left, right = _mask_operands(n, torch.float32)
        actual = op(left.to(mojo_gpu), right.to(mojo_gpu)).cpu()
        assert actual.dtype == torch.bool
        assert torch.equal(actual, op(left, right)), f"tensor operands, n={n}"
        # 0-d operand: read once and splatted, not broadcast-indexed.
        scalar = op(left.to(mojo_gpu), 0.5).cpu()
        assert torch.equal(scalar, op(left, 0.5)), f"scalar operand, n={n}"


@pytest.mark.parametrize("offset", [1, 2, 3])
def test_fast_comparison_offset_views_match_cpu(mojo_gpu, offset):
    """A base address off by `offset` elements fails the 16-byte gate and must
    take the strided arm — with the same answers, including the tail."""
    for n in (5, 17, 100_003):
        left, right = _mask_operands(n + offset, torch.float32)
        device_left = left.to(mojo_gpu)[offset:]
        device_right = right.to(mojo_gpu)[offset:]
        expected = torch.lt(left[offset:], right[offset:])
        assert torch.equal(torch.lt(device_left, device_right).cpu(), expected)
        assert torch.equal(
            torch.lt(device_left, 0.5).cpu(), torch.lt(left[offset:], 0.5)
        )

        ints = (torch.arange(n + offset) % 251).to(torch.int32)
        device_ints = ints.to(mojo_gpu)[offset:]
        assert torch.equal(
            torch.bitwise_and(device_ints, device_ints).cpu(),
            torch.bitwise_and(ints[offset:], ints[offset:]),
        )
        assert torch.equal(
            torch.bitwise_not(device_ints).cpu(), torch.bitwise_not(ints[offset:])
        )
        bools = (torch.arange(n + offset) % 3) == 0
        device_bools = bools.to(mojo_gpu)[offset:]
        assert torch.equal(
            torch.logical_not(device_bools).cpu(), torch.logical_not(bools[offset:])
        )


def test_fast_logical_and_xor_bool_operands_match_cpu(mojo_gpu):
    """bool operands ride the uint8 kernel at the full 16-element width."""
    for n in _MASK_SIZES:
        left = (torch.arange(n) % 3) == 0
        right = (torch.arange(n) % 5) < 2
        for op in (torch.logical_and, torch.logical_xor):
            actual = op(left.to(mojo_gpu), right.to(mojo_gpu)).cpu()
            assert actual.dtype == torch.bool
            assert torch.equal(actual, op(left, right)), f"{op.__name__}, n={n}"


@pytest.mark.parametrize(
    "dtype", [torch.float32, torch.float16, torch.bfloat16, torch.int64, torch.int8]
)
def test_fast_comparison_dtypes_and_scalar_operands_match_cpu(mojo_gpu, dtype):
    # One dtype per storage width, each with a ragged tail (1003 is not a
    # multiple of any of the vector widths 2, 4, 8 or 16).
    left, right = _mask_operands(1003, dtype)
    for op in _COMPARE_OPS.values():
        assert torch.equal(
            op(left.to(mojo_gpu), right.to(mojo_gpu)).cpu(), op(left, right)
        )
        assert torch.equal(op(left.to(mojo_gpu), 3).cpu(), op(left, 3))


def test_fast_bitwise_scalar_operand_match_cpu(mojo_gpu):
    for n in (5, 17, 1003, 100_003):
        values = (torch.arange(n) * 2654435761 % (1 << 30)).to(torch.int32)
        device = values.to(mojo_gpu)
        for op in (torch.bitwise_and, torch.bitwise_or, torch.bitwise_xor):
            assert torch.equal(op(device, 21).cpu(), op(values, 21)), f"{op}, n={n}"


# masked_fill / where.self: the WhereSelect flat-vec tier (`_where_bcast` in
# data_movement_ops.mojo) and the MaskedFillScalar immediate-value fast path
# (masked_fill(_).Scalar only -- see `_masked_fill_scalar_operands` in
# aten_fast.py). Both were 4-6x slower than stock on the tiny A_357x789
# benchmark shape before those tiers existed: the old code went through the
# stdlib `elementwise` GPU dispatcher (per-call `compile_function`, plus a
# six-division rank-4 coordinate decomposition per element even for a fully
# contiguous, non-broadcasting case), and masked_fill's Scalar overload paid
# a second full kernel launch (a Fill kernel) just to materialize its value
# into a 0-d device tensor before WhereSelect could read it.
_WHERE_FAMILY_DTYPES = [torch.float32, torch.bfloat16, torch.float16]


def _where_family_operands(
    n: int, dtype: torch.dtype
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """Deterministic (input, other, mask): mask true on ~1/3 of elements."""
    left = (torch.arange(n, dtype=torch.float32) % 97 - 48.0).to(dtype)
    right = (torch.arange(n, dtype=torch.float32) % 61 - 30.0).to(dtype)
    mask = (torch.arange(n) % 3) == 0
    return left, right, mask


@pytest.mark.parametrize("dtype", _WHERE_FAMILY_DTYPES)
@pytest.mark.parametrize("n", _MASK_SIZES)
def test_fast_masked_fill_scalar_vector_width_boundaries_match_cpu(mojo_gpu, n, dtype):
    """The MaskedFillScalar immediate-value kernel's vector width and ragged
    tail, functional and in-place: `value` never touches device memory for
    this overload (see `_masked_fill_scalar_operands`)."""
    x, _other, mask = _where_family_operands(n, dtype)
    x_dev, mask_dev = x.to(mojo_gpu), mask.to(mojo_gpu)

    expected = x.masked_fill(mask, -1.5)
    actual = x_dev.masked_fill(mask_dev, -1.5).cpu()
    torch.testing.assert_close(actual, expected)

    expected_ip = x.clone()
    expected_ip.masked_fill_(mask, 2.25)
    actual_ip = x_dev.clone()
    actual_ip.masked_fill_(mask_dev, 2.25)
    torch.testing.assert_close(actual_ip.cpu(), expected_ip)


@pytest.mark.parametrize("dtype", _WHERE_FAMILY_DTYPES)
def test_fast_masked_fill_tensor_value_0d_matches_cpu(mojo_gpu, dtype):
    """masked_fill's Tensor overload: `value` is a real 0-d device tensor, a
    stride-0 broadcast operand for the WhereSelect flat-vec kernel (not an
    immediate) -- exercised across the same vector-width/tail sizes."""
    for n in _MASK_SIZES:
        x, _other, mask = _where_family_operands(n, dtype)
        value = torch.tensor(3.25, dtype=dtype)
        x_dev, mask_dev = x.to(mojo_gpu), mask.to(mojo_gpu)
        value_dev = value.to(mojo_gpu)

        expected = x.masked_fill(mask, value)
        actual = x_dev.masked_fill(mask_dev, value_dev).cpu()
        torch.testing.assert_close(actual, expected)

        expected_ip = x.clone()
        expected_ip.masked_fill_(mask, value)
        actual_ip = x_dev.clone()
        actual_ip.masked_fill_(mask_dev, value_dev)
        torch.testing.assert_close(actual_ip.cpu(), expected_ip)


@pytest.mark.parametrize("dtype", _WHERE_FAMILY_DTYPES)
def test_fast_where_self_vector_width_boundaries_match_cpu(mojo_gpu, dtype):
    for n in _MASK_SIZES:
        left, right, mask = _where_family_operands(n, dtype)
        actual = torch.where(
            mask.to(mojo_gpu), left.to(mojo_gpu), right.to(mojo_gpu)
        ).cpu()
        torch.testing.assert_close(actual, torch.where(mask, left, right))


@pytest.mark.parametrize("dtype", _WHERE_FAMILY_DTYPES)
@pytest.mark.parametrize("inplace", [False, True])
def test_fast_masked_fill_broadcast_mask_matches_cpu(mojo_gpu, dtype, inplace):
    """A mask that broadcasts up to the input (leading-dim broadcast) is a
    genuinely strided case for both the Scalar and Tensor overloads: it must
    fall back off the flat-vec tier (`cond_flat` is false) and still be
    correct."""
    x = (
        (torch.arange(3 * 4 * 5, dtype=torch.float32) % 23 - 11.0)
        .to(dtype)
        .reshape(3, 4, 5)
    )
    mask = (torch.arange(4 * 5) % 4 == 0).reshape(4, 5)
    x_dev, mask_dev = x.to(mojo_gpu), mask.to(mojo_gpu)

    for value in (7.0, torch.tensor(7.0, dtype=dtype)):
        value_dev = value.to(mojo_gpu) if isinstance(value, torch.Tensor) else value
        if inplace:
            expected = x.clone()
            expected.masked_fill_(mask, value)
            actual = x_dev.clone()
            actual.masked_fill_(mask_dev, value_dev)
            actual = actual.cpu()
        else:
            expected = x.masked_fill(mask, value)
            actual = x_dev.masked_fill(mask_dev, value_dev).cpu()
        torch.testing.assert_close(actual, expected)


@pytest.mark.parametrize("dtype", _WHERE_FAMILY_DTYPES)
def test_fast_masked_fill_noncontiguous_input_matches_cpu(mojo_gpu, dtype):
    """A transposed (non-contiguous) input must fail the flat-vec alignment
    gate and take the strided fallback, not silently corrupt. Built by
    transposing ON DEVICE, not via `.to()`: a cross-device copy is free to
    re-materialize its destination contiguously regardless of the source
    layout."""
    x = (torch.arange(60, dtype=torch.float32) % 41 - 20.0).to(dtype).reshape(6, 10)
    mask = (torch.arange(60) % 5 == 0).reshape(6, 10)
    x_dev_t = x.to(mojo_gpu).t()
    mask_dev_t = mask.to(mojo_gpu).t()
    assert not x_dev_t.is_contiguous()

    expected = x.t().masked_fill(mask.t(), -3.0)
    actual = x_dev_t.masked_fill(mask_dev_t, -3.0).cpu()
    torch.testing.assert_close(actual, expected)


@pytest.mark.parametrize("op_name", ["logical_not", "bitwise_not", "isnan"])
def test_fast_unary_mask_vector_tail_match_cpu(mojo_gpu, op_name):
    """The flat vectorized unary skeleton: same tail and alignment rules, and
    for logical_not/isnan the same (dtype -> 1-byte mask) narrowing."""
    op = getattr(torch, op_name)
    for n in _MASK_SIZES:
        if op_name == "bitwise_not":
            values = (torch.arange(n) % 251 - 125).to(torch.int32)
        elif op_name == "isnan":
            values = (torch.arange(n, dtype=torch.float32) % 7) - 3.0
            values[::5] = float("nan")
        else:
            values = ((torch.arange(n) % 4) == 0).to(torch.bool)
        actual = op(values.to(mojo_gpu)).cpu()
        assert torch.equal(actual, op(values)), f"{op_name}, n={n}"


@pytest.mark.parametrize(
    "dtype", [torch.float32, torch.float16, torch.bfloat16, torch.int32]
)
@pytest.mark.parametrize("shape", [(), (1,), (5,), (3, 7)])
def test_fast_scalar_inplace_add_and_mul(mojo_gpu, dtype, shape):
    """`add_`/`mul_` with a Python scalar must write straight into the input.

    The in-place spec skips the output allocation and the device-to-device copy
    the functional-plus-copy-back path pays; the int dtype is here because it
    has to keep falling through to that path.
    """
    if dtype.is_floating_point:
        base = torch.randn(shape, dtype=torch.float32).to(dtype)
    else:
        base = (
            torch.arange(1, math.prod(shape) + 1 if shape else 2)
            .reshape(shape if shape else ())[()]
            .to(dtype)
        )
    # torch itself refuses a float scalar on an integer in-place op.
    scalars = (2.5, -1, 0.0) if dtype.is_floating_point else (3, -1, 2)
    for scalar in scalars:
        want = base.clone()
        got = base.clone().to(mojo_gpu)
        keep = got
        want.add_(scalar)
        got.add_(scalar)
        assert got is keep, "add_ must return the same tensor object"
        torch.testing.assert_close(got.cpu(), want, atol=2e-2, rtol=2e-2)

        want.mul_(scalar)
        got.mul_(scalar)
        torch.testing.assert_close(got.cpu(), want, atol=2e-2, rtol=2e-2)


def test_fast_scalar_inplace_add_alpha_and_aliasing(mojo_gpu):
    """alpha folds into the scalar, and a view sees the in-place write."""
    base = torch.arange(12, dtype=torch.float32).reshape(3, 4)
    want = base.clone()
    got = base.clone().to(mojo_gpu)
    want.add_(3.0, alpha=-2)
    got.add_(3.0, alpha=-2)
    torch.testing.assert_close(got.cpu(), want)

    # A non-contiguous target must still be correct (it falls through to the
    # functional-plus-strided-copy path, which the in-place spec declines).
    strided_want = base.clone().t()
    strided_got = base.clone().to(mojo_gpu).t()
    strided_want.add_(0.5)
    strided_got.add_(0.5)
    torch.testing.assert_close(strided_got.cpu(), strided_want)

    # The write lands in the original storage, not a copy.
    holder = base.clone().to(mojo_gpu)
    row = holder[1]
    row.add_(100.0)
    torch.testing.assert_close(holder.cpu()[1], base[1] + 100.0)


@pytest.mark.parametrize("reduction", [0, 1, 2])
def test_fast_nll_loss_forward_and_backward_out(mojo_gpu, reduction):
    generator = torch.Generator().manual_seed(20260718)
    rows, classes = 17, 13
    log_probs = torch.log_softmax(
        torch.randn(rows, classes, generator=generator), dim=-1
    )
    target = torch.arange(rows, dtype=torch.int64) % classes
    target[::5] = -1
    output_shape = (rows,) if reduction == 0 else ()
    grad_output = torch.randn(output_shape, generator=generator)

    reference_output = torch.empty(output_shape)
    reference_total_weight = torch.empty(())
    torch.ops.aten.nll_loss_forward.output(
        log_probs,
        target,
        None,
        reduction,
        -1,
        output=reference_output,
        total_weight=reference_total_weight,
    )
    reference_grad_input = torch.empty_like(log_probs)
    torch.ops.aten.nll_loss_backward.grad_input(
        grad_output,
        log_probs,
        target,
        None,
        reduction,
        -1,
        reference_total_weight,
        grad_input=reference_grad_input,
    )

    device_log_probs = log_probs.to(mojo_gpu)
    target_backing = torch.zeros(rows * 2, dtype=torch.int64)
    target_backing[::2] = target
    device_target = target_backing.to(mojo_gpu)[::2]
    device_output = torch.empty(output_shape, device=mojo_gpu)
    device_total_weight = torch.empty((), device=mojo_gpu)
    returned_output, returned_total_weight = torch.ops.aten.nll_loss_forward.output(
        device_log_probs,
        device_target,
        None,
        reduction,
        -1,
        output=device_output,
        total_weight=device_total_weight,
    )
    device_grad_input = torch.empty_like(device_log_probs)
    returned_grad_input = torch.ops.aten.nll_loss_backward.grad_input(
        grad_output.to(mojo_gpu),
        device_log_probs,
        device_target,
        None,
        reduction,
        -1,
        device_total_weight,
        grad_input=device_grad_input,
    )

    assert returned_output is device_output
    assert returned_total_weight is device_total_weight
    assert returned_grad_input is device_grad_input
    torch.testing.assert_close(device_output.cpu(), reference_output)
    torch.testing.assert_close(device_total_weight.cpu(), reference_total_weight)
    torch.testing.assert_close(device_grad_input.cpu(), reference_grad_input)


@pytest.mark.parametrize(("rows", "classes"), [(257, 65), (3, 50304)])
def test_fast_nll_loss_runtime_class_regimes(mojo_gpu, rows, classes):
    """Exercise the unaligned scalar and GPT-2-width vector kernel regimes."""
    generator = torch.Generator().manual_seed(41 + classes)
    log_probs = torch.log_softmax(
        torch.randn(rows, classes, generator=generator), dim=-1
    )
    target = torch.arange(rows, dtype=torch.int64) * 17 % classes
    target[::7] = -1
    grad_output = torch.randn((), generator=generator)

    reference_output = torch.empty(())
    reference_total_weight = torch.empty(())
    torch.ops.aten.nll_loss_forward.output(
        log_probs,
        target,
        None,
        1,
        -1,
        output=reference_output,
        total_weight=reference_total_weight,
    )
    reference_grad_input = torch.empty_like(log_probs)
    torch.ops.aten.nll_loss_backward.grad_input(
        grad_output,
        log_probs,
        target,
        None,
        1,
        -1,
        reference_total_weight,
        grad_input=reference_grad_input,
    )

    device_log_probs = log_probs.to(mojo_gpu)
    device_target = target.to(mojo_gpu)
    device_output = torch.empty((), device=mojo_gpu)
    device_total_weight = torch.empty((), device=mojo_gpu)
    torch.ops.aten.nll_loss_forward.output(
        device_log_probs,
        device_target,
        None,
        1,
        -1,
        output=device_output,
        total_weight=device_total_weight,
    )
    device_grad_input = torch.empty_like(device_log_probs)
    torch.ops.aten.nll_loss_backward.grad_input(
        grad_output.to(mojo_gpu),
        device_log_probs,
        device_target,
        None,
        1,
        -1,
        device_total_weight,
        grad_input=device_grad_input,
    )

    torch.testing.assert_close(device_output.cpu(), reference_output)
    torch.testing.assert_close(device_total_weight.cpu(), reference_total_weight)
    torch.testing.assert_close(device_grad_input.cpu(), reference_grad_input)


@pytest.mark.parametrize("reduction", [0, 1, 2])
def test_fast_nll_loss_all_ignored(mojo_gpu, reduction):
    rows, classes = 19, 65
    log_probs = torch.log_softmax(torch.randn(rows, classes), dim=-1)
    target = torch.full((rows,), -1, dtype=torch.int64)
    output_shape = (rows,) if reduction == 0 else ()
    grad_output = torch.randn(output_shape)

    device_log_probs = log_probs.to(mojo_gpu)
    device_target = target.to(mojo_gpu)
    device_output = torch.empty(output_shape, device=mojo_gpu)
    device_total_weight = torch.empty((), device=mojo_gpu)
    torch.ops.aten.nll_loss_forward.output(
        device_log_probs,
        device_target,
        None,
        reduction,
        -1,
        output=device_output,
        total_weight=device_total_weight,
    )
    device_grad_input = torch.empty_like(device_log_probs)
    torch.ops.aten.nll_loss_backward.grad_input(
        grad_output.to(mojo_gpu),
        device_log_probs,
        device_target,
        None,
        reduction,
        -1,
        device_total_weight,
        grad_input=device_grad_input,
    )

    actual_output = device_output.cpu()
    if reduction == 1:
        assert actual_output.isnan().item()
    else:
        torch.testing.assert_close(actual_output, torch.zeros(output_shape))
    torch.testing.assert_close(device_total_weight.cpu(), torch.zeros(()))
    torch.testing.assert_close(device_grad_input.cpu(), torch.zeros_like(log_probs))


def test_fast_nll_loss_resizes_outs_and_preserves_identity(mojo_gpu):
    rows, classes = 11, 17
    log_probs = torch.log_softmax(torch.randn(rows, classes), dim=-1)
    target = torch.arange(rows, dtype=torch.int64) % classes
    # Reduced NLL accepts both a scalar and a one-element vector grad_output.
    grad_output = torch.randn(1)

    reference_output = torch.empty(())
    reference_total_weight = torch.empty(())
    torch.ops.aten.nll_loss_forward.output(
        log_probs,
        target,
        None,
        1,
        -1,
        output=reference_output,
        total_weight=reference_total_weight,
    )
    reference_grad_input = torch.empty_like(log_probs)
    torch.ops.aten.nll_loss_backward.grad_input(
        grad_output,
        log_probs,
        target,
        None,
        1,
        -1,
        reference_total_weight,
        grad_input=reference_grad_input,
    )

    device_log_probs = log_probs.to(mojo_gpu)
    device_target = target.to(mojo_gpu)
    device_output = torch.empty(7, device=mojo_gpu)
    device_total_weight = torch.empty(5, device=mojo_gpu)
    returned_output, returned_total_weight = torch.ops.aten.nll_loss_forward.output(
        device_log_probs,
        device_target,
        None,
        1,
        -1,
        output=device_output,
        total_weight=device_total_weight,
    )
    device_grad_input = torch.empty(3, device=mojo_gpu)
    returned_grad_input = torch.ops.aten.nll_loss_backward.grad_input(
        grad_output.to(mojo_gpu),
        device_log_probs,
        device_target,
        None,
        1,
        -1,
        device_total_weight,
        grad_input=device_grad_input,
    )

    assert returned_output is device_output
    assert returned_total_weight is device_total_weight
    assert returned_grad_input is device_grad_input
    assert tuple(device_output._shape) == ()
    assert tuple(device_total_weight._shape) == ()
    assert tuple(device_grad_input._shape) == (rows, classes)
    torch.testing.assert_close(device_output.cpu(), reference_output)
    torch.testing.assert_close(device_total_weight.cpu(), reference_total_weight)
    torch.testing.assert_close(device_grad_input.cpu(), reference_grad_input)


def test_fast_nll_loss_strided_outs(mojo_gpu):
    rows, classes = 13, 23
    sentinel = 123.0
    log_probs = torch.log_softmax(torch.randn(rows, classes), dim=-1)
    target = torch.arange(rows, dtype=torch.int64) % classes
    target[::4] = -1
    grad_output = torch.randn(rows)

    reference_output = torch.empty(rows)
    reference_total_weight = torch.empty(())
    torch.ops.aten.nll_loss_forward.output(
        log_probs,
        target,
        None,
        0,
        -1,
        output=reference_output,
        total_weight=reference_total_weight,
    )
    reference_grad_input = torch.empty_like(log_probs)
    torch.ops.aten.nll_loss_backward.grad_input(
        grad_output,
        log_probs,
        target,
        None,
        0,
        -1,
        reference_total_weight,
        grad_input=reference_grad_input,
    )

    device_log_probs = log_probs.to(mojo_gpu)
    device_target = target.to(mojo_gpu)
    output_backing = torch.full((rows * 2,), sentinel).to(mojo_gpu)
    device_output = output_backing[::2]
    device_total_weight = torch.empty((), device=mojo_gpu)
    torch.ops.aten.nll_loss_forward.output(
        device_log_probs,
        device_target,
        None,
        0,
        -1,
        output=device_output,
        total_weight=device_total_weight,
    )
    grad_backing = torch.full((rows, classes * 2), sentinel).to(mojo_gpu)
    device_grad_input = grad_backing[:, ::2]
    torch.ops.aten.nll_loss_backward.grad_input(
        grad_output.to(mojo_gpu),
        device_log_probs,
        device_target,
        None,
        0,
        -1,
        device_total_weight,
        grad_input=device_grad_input,
    )

    actual_output_backing = output_backing.cpu()
    actual_grad_backing = grad_backing.cpu()
    torch.testing.assert_close(actual_output_backing[::2], reference_output)
    torch.testing.assert_close(
        actual_output_backing[1::2], torch.full((rows,), sentinel)
    )
    torch.testing.assert_close(actual_grad_backing[:, ::2], reference_grad_input)
    torch.testing.assert_close(
        actual_grad_backing[:, 1::2], torch.full((rows, classes), sentinel)
    )


@pytest.mark.parametrize("reduction", [0, 1, 2])
def test_fast_nll_loss_empty_batch(mojo_gpu, reduction):
    classes = 7
    log_probs = torch.empty(0, classes)
    target = torch.empty(0, dtype=torch.int64)
    output_shape = (0,) if reduction == 0 else ()
    grad_output = torch.empty(0) if reduction == 0 else torch.ones(())

    device_log_probs = log_probs.to(mojo_gpu)
    device_target = target.to(mojo_gpu)
    device_output = torch.empty(output_shape, device=mojo_gpu)
    device_total_weight = torch.empty((), device=mojo_gpu)
    torch.ops.aten.nll_loss_forward.output(
        device_log_probs,
        device_target,
        None,
        reduction,
        -1,
        output=device_output,
        total_weight=device_total_weight,
    )
    device_grad_input = torch.empty_like(device_log_probs)
    torch.ops.aten.nll_loss_backward.grad_input(
        grad_output.to(mojo_gpu),
        device_log_probs,
        device_target,
        None,
        reduction,
        -1,
        device_total_weight,
        grad_input=device_grad_input,
    )

    actual_output = device_output.cpu()
    if reduction == 1:
        assert actual_output.isnan().item()
    else:
        torch.testing.assert_close(actual_output, torch.zeros(output_shape))
    torch.testing.assert_close(device_total_weight.cpu(), torch.zeros(()))
    assert tuple(device_grad_input.cpu().shape) == (0, classes)


def test_fast_nll_loss_rejects_unsupported_metadata_without_resizing(mojo_gpu):
    rows, classes = 5, 7
    log_probs = torch.log_softmax(torch.randn(rows, classes), dim=-1)
    target = torch.arange(rows, dtype=torch.int64) % classes
    device_log_probs = log_probs.to(mojo_gpu)
    device_target = target.to(mojo_gpu)
    output = torch.empty(3, device=mojo_gpu)
    bad_total_weight = torch.empty((), dtype=torch.float16, device=mojo_gpu)

    with pytest.raises(NotImplementedError, match="nll_loss_forward.output"):
        torch.ops.aten.nll_loss_forward.output(
            device_log_probs,
            device_target,
            None,
            1,
            -1,
            output=output,
            total_weight=bad_total_weight,
        )
    # The valid-but-wrong-shaped first output was not rebound before the
    # invalid second output caused the operation to reject the call.
    assert tuple(output._shape) == (3,)

    good_output = torch.empty((), device=mojo_gpu)
    good_total_weight = torch.empty((), device=mojo_gpu)
    invalid_calls = [
        lambda: torch.ops.aten.nll_loss_forward.output(
            device_log_probs,
            device_target,
            torch.ones(classes, device=mojo_gpu),
            1,
            -1,
            output=good_output,
            total_weight=good_total_weight,
        ),
        lambda: torch.ops.aten.nll_loss_forward.output(
            log_probs.half().to(mojo_gpu),
            device_target,
            None,
            1,
            -1,
            output=good_output,
            total_weight=good_total_weight,
        ),
        lambda: torch.ops.aten.nll_loss_forward.output(
            device_log_probs,
            target.to(torch.int32).to(mojo_gpu),
            None,
            1,
            -1,
            output=good_output,
            total_weight=good_total_weight,
        ),
        lambda: torch.ops.aten.nll_loss_forward.output(
            device_log_probs,
            device_target[:-1],
            None,
            1,
            -1,
            output=good_output,
            total_weight=good_total_weight,
        ),
    ]
    for invalid_call in invalid_calls:
        with pytest.raises(NotImplementedError, match="nll_loss_forward.output"):
            invalid_call()


@pytest.mark.parametrize("reduction", ["none", "mean", "sum"])
def test_fast_cross_entropy_training_uses_direct_nll_kernel(
    mojo_gpu, monkeypatch, reduction
):
    """End-to-end autograd must enqueue both direct NLL kernel bridges."""
    generator = torch.Generator().manual_seed(20260718)
    rows, classes = 19, 65
    logits = torch.randn(rows, classes, generator=generator)
    target = torch.arange(rows, dtype=torch.int64) * 11 % classes
    target[::6] = -1
    grad_output = (
        torch.randn(rows, generator=generator)
        if reduction == "none"
        else torch.randn((), generator=generator)
    )

    reference = logits.clone().requires_grad_()
    reference_loss = torch.nn.functional.cross_entropy(
        reference, target, reduction=reduction, ignore_index=-1
    )
    reference_loss.backward(grad_output)

    forward_target = ("loss_ops.mojo", "NllLossForwardF32")
    backward_target = ("loss_ops.mojo", "NllLossBackwardF32")
    native_calls = _spy_defined_native_calls(
        monkeypatch, {forward_target, backward_target}
    )

    actual = logits.to(mojo_gpu).requires_grad_()
    actual_loss = torch.nn.functional.cross_entropy(
        actual, target.to(mojo_gpu), reduction=reduction, ignore_index=-1
    )
    actual_loss.backward(grad_output.to(mojo_gpu))

    assert len(native_calls[forward_target]) == 1
    assert len(native_calls[backward_target]) == 1
    torch.testing.assert_close(actual_loss.cpu(), reference_loss)
    assert actual.grad is not None
    torch.testing.assert_close(actual.grad.cpu(), reference.grad, atol=2e-5, rtol=2e-5)


def test_fast_nll_loss_autograd_uses_saved_tensor_hooks(mojo_gpu):
    from torch_mojo_backend.mojo_device.mojo_device_aten_ops import EAGER_CALL_COUNTERS

    log_probs = torch.log_softmax(torch.randn(7, 11), dim=-1)
    target = torch.arange(7, dtype=torch.int64) % 11
    grad_output = torch.randn(())
    reference = log_probs.clone().requires_grad_()
    torch.nn.functional.nll_loss(reference, target).backward(grad_output)
    hook_calls = []

    def pack(tensor):
        hook_calls.append(("pack", tensor.device.type, tuple(tensor.shape)))
        return tensor.cpu()

    def unpack(tensor):
        hook_calls.append(("unpack", tensor.device.type, tuple(tensor.shape)))
        return tensor.to(mojo_gpu)

    actual = log_probs.to(mojo_gpu).requires_grad_()
    backward_counter = EAGER_CALL_COUNTERS["aten::nll_loss_backward.grad_input"]
    calls_before = backward_counter.call_count
    with torch.autograd.graph.saved_tensors_hooks(pack, unpack):
        output = torch.nn.functional.nll_loss(actual, target.to(mojo_gpu))
        assert type(output.grad_fn).__name__ == "NllLossBackward0"
        assert backward_counter.call_count == calls_before
        output.backward(grad_output.to(mojo_gpu))
        assert backward_counter.call_count == calls_before + 1

    expected_shapes = [(7, 11), (7,), ()]
    assert hook_calls == [("pack", "mojo", shape) for shape in expected_shapes] + [
        ("unpack", "cpu", shape) for shape in expected_shapes
    ]
    torch.testing.assert_close(actual.grad.cpu(), reference.grad)


def test_fast_nll_loss_native_double_backward(mojo_gpu):
    generator = torch.Generator().manual_seed(20260722)
    host_input = torch.randn(7, 11, generator=generator)
    target = torch.arange(7, dtype=torch.int64) % 11
    grad_output = torch.randn((), generator=generator)
    second_seed = torch.randn(7, 11, generator=generator)

    def derivatives(input, device_target, first_seed, seed2):
        output = torch.nn.functional.nll_loss(input, device_target)
        (input_grad,) = torch.autograd.grad(
            output, input, grad_outputs=first_seed, create_graph=True
        )
        (second_grad_output,) = torch.autograd.grad(
            input_grad, first_seed, grad_outputs=seed2
        )
        return input_grad, second_grad_output

    reference_input = host_input.clone().requires_grad_()
    reference_grad_output = grad_output.clone().requires_grad_()
    expected = derivatives(reference_input, target, reference_grad_output, second_seed)

    actual_input = host_input.to(mojo_gpu).requires_grad_()
    actual_grad_output = grad_output.to(mojo_gpu).requires_grad_()
    actual = derivatives(
        actual_input, target.to(mojo_gpu), actual_grad_output, second_seed.to(mojo_gpu)
    )

    assert type(actual[0].grad_fn).__name__ == "NllLossBackwardBackward0"
    for got, want in zip(actual, expected, strict=True):
        torch.testing.assert_close(got.cpu(), want)


@pytest.mark.parametrize("mutated", ["input", "target", "total_weight"])
def test_fast_nll_loss_backward_rejects_mutated_saved_tensor(mojo_gpu, mutated):
    log_probs = torch.log_softmax(torch.randn(7, 11), dim=-1).to(mojo_gpu)
    log_probs.requires_grad_()
    target = (torch.arange(7, dtype=torch.int64) % 11).to(mojo_gpu)
    output, total_weight = torch.ops.aten.nll_loss_forward.default(
        log_probs, target, None, 1, -100
    )

    with torch.no_grad():
        tensor = {"input": log_probs, "target": target, "total_weight": total_weight}[
            mutated
        ]
        tensor.add_(torch.ones_like(tensor))

    with pytest.raises(RuntimeError, match="modified by an inplace operation"):
        output.backward()


@pytest.mark.parametrize("dtype", [torch.float32, torch.float16])
def test_fast_mm_degenerate_dims(mojo_device, dtype):
    # n == 1 used to segfault the CPU library-matmul route (gemv special
    # case without a DeviceContext); m == 1 / k == 1 pinned as regression
    # guards for the library's other special-case routes.
    for m, k, n in [(37, 129, 1), (1, 129, 64), (64, 1, 33), (1, 129, 1)]:
        a = torch.randn(m, k).to(dtype)
        b = torch.randn(k, n).to(dtype)
        dev = torch.mm(a.to(mojo_device), b.to(mojo_device)).cpu()
        ref = (a.float() @ b.float()).to(dtype)
        torch.testing.assert_close(dev, ref, atol=5e-2, rtol=5e-2)
    # batched n == 1 shares the same path
    a3 = torch.randn(4, 8, 129).to(dtype)
    b3 = torch.randn(4, 129, 1).to(dtype)
    dev3 = torch.bmm(a3.to(mojo_device), b3.to(mojo_device)).cpu()
    ref3 = torch.bmm(a3.float(), b3.float()).to(dtype)
    torch.testing.assert_close(dev3, ref3, atol=5e-2, rtol=5e-2)


@pytest.mark.parametrize("dtype", [torch.float32, torch.float16])
def test_fast_mm_aligned_single_row(mojo_gpu, dtype):
    # This aligned shape selects GEVM on AMD.  Keep both the plain and bias
    # paths covered because GPT-2 decode uses the latter.
    a = torch.randn(1, 128).to(dtype)
    b = torch.randn(128, 64).to(dtype)
    bias = torch.randn(64).to(dtype)

    dev = torch.mm(a.to(mojo_gpu), b.to(mojo_gpu)).cpu()
    ref = (a.float() @ b.float()).to(dtype)
    torch.testing.assert_close(dev, ref, atol=5e-2, rtol=5e-2)

    dev_bias = torch.addmm(bias.to(mojo_gpu), a.to(mojo_gpu), b.to(mojo_gpu)).cpu()
    ref_bias = (a.float() @ b.float() + bias.float()).to(dtype)
    torch.testing.assert_close(dev_bias, ref_bias, atol=5e-2, rtol=5e-2)


@pytest.mark.parametrize("dtype", [torch.float32, torch.float16, torch.bfloat16])
def test_fast_linear_single_token(mojo_device, dtype):
    # m == 1 with bias: the decode-step GEMV route (on GPU this is
    # modular's gemv_gpu — GEMV_SPLIT_K for f16/bf16 aligned-k — plus the
    # row-broadcast bias epilogue).
    x = torch.randn(1, 768).to(dtype)
    w = torch.randn(96, 768).to(dtype)
    b = torch.randn(96).to(dtype)
    dev = torch.nn.functional.linear(
        x.to(mojo_device), w.to(mojo_device), b.to(mojo_device)
    ).cpu()
    ref = (x.float() @ w.float().t() + b.float()).to(dtype)
    torch.testing.assert_close(dev, ref, atol=5e-2, rtol=5e-2)


@pytest.mark.parametrize("with_bias", [False, True])
def test_fast_linear_training_backward(mojo_gpu, with_bias):
    from torch_mojo_backend.mojo_device.mojo_device_aten_ops import EAGER_CALL_COUNTERS

    generator = torch.Generator().manual_seed(20260718)
    x = torch.randn(2, 16, 32, generator=generator)
    weight = torch.randn(64, 32, generator=generator)
    bias = torch.randn(64, generator=generator) if with_bias else None
    grad_output = torch.randn(2, 16, 64, generator=generator)

    reference_inputs = [x.clone().requires_grad_(), weight.clone().requires_grad_()]
    reference_bias = bias.clone().requires_grad_() if bias is not None else None
    torch.nn.functional.linear(*reference_inputs, reference_bias).backward(grad_output)

    mojo_inputs = [tensor.to(mojo_gpu).requires_grad_() for tensor in (x, weight)]
    mojo_bias = bias.to(mojo_gpu).requires_grad_() if bias is not None else None
    backward_counter = EAGER_CALL_COUNTERS["aten::linear_backward"]
    calls_before = backward_counter.call_count
    mojo_output = torch.nn.functional.linear(*mojo_inputs, mojo_bias)
    assert type(mojo_output.grad_fn).__name__ == "LinearBackward0"
    assert backward_counter.call_count == calls_before

    mojo_output.backward(grad_output.to(mojo_gpu))
    assert backward_counter.call_count == calls_before + 1

    for actual, expected in zip(mojo_inputs, reference_inputs, strict=True):
        assert actual.grad is not None
        torch.testing.assert_close(
            actual.grad.cpu(), expected.grad, atol=2e-4, rtol=2e-4
        )
    if mojo_bias is not None:
        assert mojo_bias.grad is not None
        torch.testing.assert_close(
            mojo_bias.grad.cpu(), reference_bias.grad, atol=2e-4, rtol=2e-4
        )


@pytest.mark.parametrize(
    ("requires_grad", "expected_mm_calls", "expected_sum_calls"),
    [
        ((True, False, False), 1, 0),
        ((False, True, False), 1, 0),
        # PyTorch's linear_backward Meta/MPS contract couples the two
        # parameter outputs: requesting bias also computes grad_weight.
        ((False, False, True), 1, 1),
        ((True, True, True), 2, 1),
    ],
)
def test_fast_linear_native_backward_honors_output_mask_helper_calls(
    mojo_h100, monkeypatch, requires_grad, expected_mm_calls, expected_sum_calls
):
    from torch_mojo_backend.eager_kernels import aten_fast

    mm_calls = 0
    sum_calls = 0
    original_mm = aten_fast.fast_aten_mm
    original_sum = aten_fast.fast_aten_sum

    def counted_mm(*args, **kwargs):
        nonlocal mm_calls
        mm_calls += 1
        return original_mm(*args, **kwargs)

    def counted_sum(*args, **kwargs):
        nonlocal sum_calls
        sum_calls += 1
        return original_sum(*args, **kwargs)

    monkeypatch.setattr(aten_fast, "fast_aten_mm", counted_mm)
    monkeypatch.setattr(aten_fast, "fast_aten_sum", counted_sum)

    host_input = torch.randn(4, 7, dtype=torch.bfloat16)
    host_weight = torch.randn(11, 7, dtype=torch.bfloat16)
    host_bias = torch.randn(11, dtype=torch.bfloat16)
    host_grad_output = torch.randn(4, 11, dtype=torch.bfloat16)

    reference = [host_input.clone(), host_weight.clone(), host_bias.clone()]
    for tensor, requested in zip(reference, requires_grad, strict=True):
        tensor.requires_grad_(requested)
    torch.nn.functional.linear(*reference).backward(host_grad_output)

    input = host_input.to(mojo_h100)
    weight = host_weight.to(mojo_h100)
    bias = host_bias.to(mojo_h100)
    input.requires_grad_(requires_grad[0])
    weight.requires_grad_(requires_grad[1])
    bias.requires_grad_(requires_grad[2])
    grad_output = host_grad_output.to(mojo_h100)

    output = torch.nn.functional.linear(input, weight, bias)
    assert type(output.grad_fn).__name__ == "LinearBackward0"
    output.backward(grad_output)

    assert mm_calls == expected_mm_calls
    assert sum_calls == expected_sum_calls
    for actual, expected, requested in zip(
        (input, weight, bias), reference, requires_grad, strict=True
    ):
        if requested:
            assert actual.grad is not None
            torch.testing.assert_close(actual.grad.cpu(), expected.grad)
        else:
            assert actual.grad is None


def test_fast_linear_native_double_backward(mojo_gpu):
    generator = torch.Generator().manual_seed(20260722)
    host_input = torch.randn(2, 5, 7, generator=generator)
    host_weight = torch.randn(11, 7, generator=generator)
    first_seed = torch.randn(2, 5, 11, generator=generator)
    second_seed = torch.randn(2, 5, 7, generator=generator)

    def derivatives(input, weight, output_seed, input_grad_seed):
        output = torch.nn.functional.linear(input, weight)
        (input_grad,) = torch.autograd.grad(
            output, input, grad_outputs=output_seed, create_graph=True
        )
        (weight_second_grad,) = torch.autograd.grad(
            input_grad, weight, grad_outputs=input_grad_seed
        )
        return input_grad, weight_second_grad

    reference_input = host_input.clone().requires_grad_()
    reference_weight = host_weight.clone().requires_grad_()
    expected_first, expected_second = derivatives(
        reference_input, reference_weight, first_seed, second_seed
    )

    actual_input = host_input.to(mojo_gpu).requires_grad_()
    actual_weight = host_weight.to(mojo_gpu).requires_grad_()
    actual_first, actual_second = derivatives(
        actual_input, actual_weight, first_seed.to(mojo_gpu), second_seed.to(mojo_gpu)
    )

    torch.testing.assert_close(actual_first.cpu(), expected_first)
    torch.testing.assert_close(actual_second.cpu(), expected_second)


@pytest.mark.parametrize("mutated", ["input", "weight"])
def test_fast_linear_native_backward_rejects_mutated_saved_tensor(mojo_gpu, mutated):
    input = torch.randn(3, 7).to(mojo_gpu).requires_grad_()
    weight = torch.randn(11, 7).to(mojo_gpu).requires_grad_()
    output = torch.nn.functional.linear(input, weight)

    with torch.no_grad():
        (input if mutated == "input" else weight).add_(1.0)

    with pytest.raises(RuntimeError, match="modified by an inplace operation"):
        output.sum().backward()


def test_fast_linear_native_backward_does_not_save_bias(mojo_gpu):
    input = torch.randn(3, 7).to(mojo_gpu).requires_grad_()
    weight = torch.randn(11, 7).to(mojo_gpu).requires_grad_()
    bias = torch.randn(11).to(mojo_gpu).requires_grad_()
    grad_output = torch.randn(3, 11)
    output = torch.nn.functional.linear(input, weight, bias)

    with torch.no_grad():
        bias.add_(1.0)

    output.backward(grad_output.to(mojo_gpu))
    assert bias.grad is not None
    torch.testing.assert_close(bias.grad.cpu(), grad_output.sum(dim=0))


def test_fast_linear_native_saved_tensor_hooks(mojo_gpu):
    generator = torch.Generator().manual_seed(20260722)
    input = torch.randn(3, 7, generator=generator)
    weight = torch.randn(11, 7, generator=generator)
    bias = torch.randn(11, generator=generator)
    grad_output = torch.randn(3, 11, generator=generator)

    reference = [
        input.clone().requires_grad_(),
        weight.clone().requires_grad_(),
        bias.clone().requires_grad_(),
    ]
    torch.nn.functional.linear(reference[0], reference[1], reference[2]).backward(
        grad_output
    )

    actual = [
        input.to(mojo_gpu).requires_grad_(),
        weight.to(mojo_gpu).requires_grad_(),
        bias.to(mojo_gpu).requires_grad_(),
    ]
    hook_calls = []

    def pack(tensor):
        hook_calls.append(("pack", tensor.device.type))
        return tensor.cpu()

    def unpack(tensor):
        hook_calls.append(("unpack", tensor.device.type))
        return tensor.to(mojo_gpu)

    with torch.autograd.graph.saved_tensors_hooks(pack, unpack):
        output = torch.nn.functional.linear(actual[0], actual[1], actual[2])
        output.backward(grad_output.to(mojo_gpu))

    assert hook_calls.count(("pack", "mojo")) == 2
    assert hook_calls.count(("unpack", "cpu")) == 2
    for got, want in zip(actual, reference, strict=True):
        assert got.grad is not None
        torch.testing.assert_close(got.grad.cpu(), want.grad)


def test_fast_linear_skinny_m_large_output(mojo_gpu):
    # GPT-2's batch-32 lm_head takes Apple's 32-row simdgroup-matrix path.
    # Other GPUs retain the skinny-M C-transpose path.
    x = torch.randn(32, 1, 768)
    w = torch.randn(8192, 768)
    dev = torch.nn.functional.linear(x.to(mojo_gpu), w.to(mojo_gpu)).cpu()
    ref = x @ w.t()
    torch.testing.assert_close(dev, ref, atol=5e-2, rtol=5e-2)


@pytest.mark.parametrize(
    ("in_features", "out_features"), [(768, 2304), (768, 768), (768, 3072), (3072, 768)]
)
def test_fast_addmm_gpt2_batch32(mojo_gpu, in_features, out_features):
    x = torch.randn(32, in_features)
    w = torch.randn(in_features, out_features)
    bias = torch.randn(out_features)
    dev = torch.addmm(bias.to(mojo_gpu), x.to(mojo_gpu), w.to(mojo_gpu)).cpu()
    ref = torch.addmm(bias, x, w)
    torch.testing.assert_close(dev, ref, atol=5e-2, rtol=5e-2)


def test_tf32_module_is_available_for_lazy_import():
    from torch_mojo_backend.eager_kernels import aten_fast

    assert aten_fast._Tf32MatmulExtension.MOJO_FILE.name == "tf32_matmul_ops.mojo"


def test_bf16_module_is_available_for_lazy_import():
    from torch_mojo_backend.eager_kernels import aten_fast

    assert aten_fast._Gemm16MatmulExtension.MOJO_FILE.name == "gemm16_matmul_ops.mojo"


def test_bf16_v3_source_dependency_and_kernel_contract():
    """The lazy bridge includes v3 while v2 remains its explicit fallback."""
    from torch_mojo_backend.eager_kernels import aten_fast

    assert [path.name for path in aten_fast._GEMM16_SOURCE_PATHS] == [
        "gemm16_matmul_ops.mojo",
        "gemm16_v3_kernels.mojo",
        "gemm16_tn_v4_kernels.mojo",
        "gemm16_kernels.mojo",
    ]
    bridge_path, v3_path, tn_v4_path, fallback_path = aten_fast._GEMM16_SOURCE_PATHS
    bridge_source = bridge_path.read_text()
    v3_source = v3_path.read_text()
    tn_v4_source = tn_v4_path.read_text()
    fallback_source = fallback_path.read_text()

    assert "from gemm16_v3_kernels import" in bridge_source
    assert "from gemm16_kernels import (" in v3_source
    assert "from gemm16_tn_v4_kernels import" in v3_source
    # Kernel names carry the dtype the build was specialized for, so the
    # literal in the source is the tag interpolation, not "bf16".  A user
    # profiling a float16 model must read "f16_gemm_..." there.
    assert "from gemm16_dtype import _GEMM16_DT, _GEMM16_TAG" in v3_source
    for kernel_name in (
        "gemm_v3_nn_ws_m64n128_tma_s3",
        "gemm_v3_nn_ws_m128n256_tma_s3",
        "gemm_v3_nt_ws_m128n256_tma_s3",
        "gemm_v3_tn_ws_m64n128_tma_col_a_s3",
        "gemm_v3_tn_ws_m128n256_tma_col_a_s3",
    ):
        assert f'@__name(t"{{_GEMM16_TAG}}_{kernel_name}")' in v3_source

    for helper_name in (
        "_v3_enqueue_nn_ws_m64n128_tma_s3",
        "_v3_enqueue_nn_ws_m128n256_tma_s3",
    ):
        assert v3_source.count(f"{helper_name}(") == 2

    nt_kernel_start = v3_source.index(
        '@__name(t"{_GEMM16_TAG}_gemm_v3_nt_ws_m128n256_tma_s3")'
    )
    nt_kernel_end = v3_source.index(
        '@__name(t"{_GEMM16_TAG}_gemm_v3_tn_ws_m64n128_tma_col_a_s3")'
    )
    nt_source = v3_source[nt_kernel_start:nt_kernel_end]
    assert "b_tma.prefetch_descriptor()\n        barrier()" in nt_source
    assert "DeviceAttribute.MULTIPROCESSOR_COUNT" in v3_source
    for scratch_only in (
        "bf16_gemm_v3_nn_wgmma_tma_s2",
        "_v3_enqueue_nt_ws_m128n256_tma_s4",
        "candidate_bf16_gemm_accepted_v2",
        "GPT-5.6-SOL",
    ):
        assert scratch_only not in v3_source

    def _code_only(source: str) -> str:
        # The vendor-library ban is about imports and calls, not prose: a
        # comment saying a kernel was benchmarked against cuBLAS must not
        # trip it (it did once, via #392's tuning notes).
        return "\n".join(line.split("#", 1)[0] for line in source.splitlines())

    for source in (bridge_source, v3_source, tn_v4_source, fallback_source):
        code = _code_only(source).lower()
        for forbidden in (
            ".synchronize(",
            "devicecontext(",
            "from linalg.matmul",
            "cublas",
            "cudnn",
            "rocblas",
            "triton",
        ):
            assert forbidden not in code


@pytest.mark.parametrize("operation", ["gemm", "bmm"])
@pytest.mark.parametrize("failure_mode", ["missing_source", "import_error"])
def test_bf16_unavailable_bridge_falls_back_before_allocation(
    monkeypatch, operation, failure_mode
):
    """Missing sources skip early; a compiler failure is memoized and raised."""
    from torch_mojo_backend import eager_kernels
    from torch_mojo_backend.eager_kernels import aten_fast

    device = SimpleNamespace(label="gpu", api="cuda", architecture_name="sm_90a")

    def tensor(shape):
        shape = tuple(shape)
        return SimpleNamespace(
            _shape=shape,
            _mojo_strides=aten_fast._row_major_strides(shape),
            _dtype=aten_fast.DType.bfloat16,
            _device=device,
            _ptr=1234,
            _is_contiguous=True,
        )

    def fail_allocation(*_args, **_kwargs):
        raise AssertionError("unavailable BF16 bridge allocated an output")

    monkeypatch.setattr(aten_fast, "_t", lambda value: value)
    monkeypatch.setattr(aten_fast, "_ctx_ptr", lambda _device: 1)
    monkeypatch.setattr(
        aten_fast,
        "_GEMM16_SOURCE_PATHS",
        (SimpleNamespace(is_file=lambda: failure_mode != "missing_source"),),
    )

    def call():
        if operation == "gemm":
            return aten_fast._try_gemm16_mm(tensor((3, 4)), tensor((4, 5)))
        return aten_fast._try_gemm16_bmm(tensor((2, 3, 4)), tensor((2, 4, 5)))

    if failure_mode == "missing_source":
        monkeypatch.setattr(aten_fast, "_alloc", fail_allocation)
        assert call() is None
        assert call() is None
        return

    failure = ImportError("synthetic Mojo compiler failure")
    attempts = 0

    def fail_build(
        _source: Path, _defines: eager_kernels.CanonicalDefines | None
    ) -> Path:
        nonlocal attempts
        attempts += 1
        raise failure

    monkeypatch.setattr(
        aten_fast, "_alloc", lambda shape, _dtype, _device: tensor(shape)
    )
    monkeypatch.setattr(
        eager_kernels, "_ensure_tensor_holder", lambda: ModuleType("holder")
    )
    monkeypatch.setattr(eager_kernels, "_build_extension", fail_build)
    monkeypatch.setattr(
        eager_kernels, "MOJO_EXTENSION_LOADER", eager_kernels.MojoExtensionLoader()
    )
    for _ in range(2):
        with pytest.raises(ImportError) as raised:
            call()
        assert raised.value is failure
    assert attempts == 1


def test_bf16_matmul_family_precedes_tf32_and_tensorspec(monkeypatch):
    """Every eligible public entry point gives BF16 the first opportunity."""
    from torch_mojo_backend.eager_kernels import aten_fast

    lhs, rhs, bias, weight = object(), object(), object(), object()
    gemm_result, linear_result, bmm_result = object(), object(), object()
    gemm_calls = []
    linear_calls = []
    bmm_calls = []

    def try_gemm(*args, **kwargs):
        gemm_calls.append((args, kwargs))
        return gemm_result

    def try_linear(*args, **kwargs):
        linear_calls.append((args, kwargs))
        return linear_result

    def try_bmm(*args, **kwargs):
        bmm_calls.append((args, kwargs))
        return bmm_result

    def fail_later_route(*_args, **_kwargs):
        raise AssertionError("BF16-routed matmul reached TF32 or TensorSpec")

    monkeypatch.setattr(aten_fast, "_try_gemm16_mm", try_gemm)
    monkeypatch.setattr(aten_fast, "_try_gemm16_linear", try_linear)
    monkeypatch.setattr(aten_fast, "_try_gemm16_bmm", try_bmm)
    monkeypatch.setattr(aten_fast, "_try_tf32_gemm", fail_later_route)
    monkeypatch.setattr(aten_fast, "_try_tf32_linear", fail_later_route)
    monkeypatch.setattr(aten_fast, "_try_tf32_bmm", fail_later_route)
    monkeypatch.setattr(aten_fast, "_try_spec_matmul", fail_later_route)

    assert aten_fast.fast_aten_mm(lhs, rhs) is gemm_result
    # `lhs`/`rhs` are plain placeholders, not TorchMojoTensors, so
    # _gemm16_alignment_favors_split (which inspects real shapes) always
    # declines here and addmm takes its single bias-fused call, same as mm.
    assert aten_fast.fast_aten_addmm(bias, lhs, rhs) is gemm_result
    assert aten_fast.fast_aten_linear(lhs, weight, bias) is linear_result
    assert aten_fast.fast_aten_bmm(lhs, rhs) is bmm_result
    assert aten_fast._fast_aten_bmm_transpose_b(lhs, rhs) is bmm_result

    assert gemm_calls == [((lhs, rhs), {}), ((lhs, rhs, bias), {})]
    assert linear_calls == [((lhs, weight, bias), {})]
    assert bmm_calls == [((lhs, rhs), {}), ((lhs, rhs), {"transpose_b": True})]


def test_gemm16_alignment_favors_split_helper(monkeypatch):
    """The cheap pre-check gemm16's bias split relies on: every v3/v4
    tensor-core route needs m, n, and k each a multiple of 64, so a shape
    missing that can never benefit from splitting the mm from the bias."""
    from torch_mojo_backend.eager_kernels import aten_fast

    monkeypatch.setattr(aten_fast, "_t", lambda value: value)

    def tensor(shape):
        return SimpleNamespace(_shape=tuple(shape))

    assert aten_fast._gemm16_alignment_favors_split(
        tensor((128, 128)), tensor((128, 128))
    )
    # S5's exact shape (357x789x333): none of m, n, k is a multiple of 64.
    assert not aten_fast._gemm16_alignment_favors_split(
        tensor((357, 333)), tensor((333, 789))
    )
    # k mismatch between lhs and rhs.
    assert not aten_fast._gemm16_alignment_favors_split(
        tensor((128, 128)), tensor((64, 128))
    )
    # transpose_b reinterprets which rhs dim is k vs n.
    assert aten_fast._gemm16_alignment_favors_split(
        tensor((128, 128)), tensor((128, 128)), transpose_b=True
    )
    assert not aten_fast._gemm16_alignment_favors_split(tensor((3, 4)), tensor((4, 5)))


def test_addmm_skips_split_for_misaligned_shapes(monkeypatch):
    """A misaligned shape (S5's 357x789x333 regime) must take the single
    bias-fused gemm16 call directly rather than paying for a wasted
    bias-free mm plus a separate add: gemm16 always falls to the same
    "accepted" kernel either way, so the split only adds a kernel launch
    with no benefit.  Regression guard for the ~5us-per-launch tax that
    turned this exact shape's f16 addmm from an already-good <1.0x ratio
    into a 1.25-1.65x regression before this gate existed."""
    from torch_mojo_backend.eager_kernels import aten_fast

    def tensor(shape):
        return SimpleNamespace(_shape=tuple(shape))

    lhs, rhs, bias = tensor((357, 333)), tensor((333, 789)), object()
    fused_result = object()
    fused_calls = []

    def try_gemm(*args, **kwargs):
        fused_calls.append((args, kwargs))
        return fused_result

    def fail_add(*_args, **_kwargs):
        raise AssertionError("misaligned addmm should never try the bias-free split")

    monkeypatch.setattr(aten_fast, "_t", lambda value: value)
    monkeypatch.setattr(aten_fast, "_try_gemm16_mm", try_gemm)
    monkeypatch.setattr(aten_fast, "fast_aten_add", fail_add)

    assert aten_fast.fast_aten_addmm(bias, lhs, rhs) is fused_result
    assert fused_calls == [((lhs, rhs, bias), {})]


def test_addmm_tries_split_for_aligned_shapes(monkeypatch):
    """A 64-aligned addmm shape (unlike S5) does try the bias-free mm plus
    a separate add first, since it could plausibly reach a fast gemm16
    route -- see _gemm16_alignment_favors_split."""
    from torch_mojo_backend.eager_kernels import aten_fast

    def tensor(shape):
        return SimpleNamespace(_shape=tuple(shape))

    lhs, rhs, bias = tensor((128, 128)), tensor((128, 128)), object()
    mm_result, biased_result = object(), object()
    gemm_calls = []
    add_calls = []

    def try_gemm(*args, **kwargs):
        gemm_calls.append((args, kwargs))
        return mm_result

    def try_add(*args, **kwargs):
        add_calls.append((args, kwargs))
        return biased_result

    monkeypatch.setattr(aten_fast, "_t", lambda value: value)
    monkeypatch.setattr(aten_fast, "_try_gemm16_mm", try_gemm)
    monkeypatch.setattr(aten_fast, "fast_aten_add", try_add)

    assert aten_fast.fast_aten_addmm(bias, lhs, rhs) is biased_result
    assert gemm_calls == [((lhs, rhs), {})]
    assert add_calls == [((mm_result, bias), {})]


@pytest.mark.parametrize("operation", ["gemm", "bmm"])
@pytest.mark.parametrize("failure_mode", ["missing_source", "import_error"])
def test_tf32_unavailable_bridge_falls_back_before_allocation(
    monkeypatch, operation, failure_mode
):
    """Missing sources skip early; a compiler failure is memoized and raised."""
    from torch_mojo_backend import eager_kernels
    from torch_mojo_backend.eager_kernels import aten_fast

    device = SimpleNamespace(label="gpu", api="cuda", architecture_name="sm_90a")

    def tensor(shape):
        shape = tuple(shape)
        return SimpleNamespace(
            _shape=shape,
            _mojo_strides=aten_fast._row_major_strides(shape),
            _dtype=aten_fast.DType.float32,
            _device=device,
            _ptr=1234,
            _is_contiguous=True,
        )

    def fail_allocation(*_args, **_kwargs):
        raise AssertionError("unavailable TF32 bridge allocated an output")

    monkeypatch.setattr(aten_fast.torch, "get_float32_matmul_precision", lambda: "high")
    monkeypatch.setattr(aten_fast, "_t", lambda value: value)
    monkeypatch.setattr(aten_fast, "_ctx_ptr", lambda _device: 1)
    monkeypatch.setattr(
        aten_fast,
        "_TF32_SOURCE_PATHS",
        (SimpleNamespace(is_file=lambda: failure_mode != "missing_source"),),
    )

    def call():
        if operation == "gemm":
            return aten_fast._try_tf32_gemm(tensor((3, 4)), tensor((4, 5)))
        return aten_fast._try_tf32_bmm(tensor((2, 3, 4)), tensor((2, 4, 5)))

    if failure_mode == "missing_source":
        monkeypatch.setattr(aten_fast, "_alloc", fail_allocation)
        assert call() is None
        assert call() is None
        return

    failure = ImportError("synthetic Mojo compiler failure")
    attempts = 0

    def fail_build(
        _source: Path, _defines: eager_kernels.CanonicalDefines | None
    ) -> Path:
        nonlocal attempts
        attempts += 1
        raise failure

    monkeypatch.setattr(
        aten_fast, "_alloc", lambda shape, _dtype, _device: tensor(shape)
    )
    monkeypatch.setattr(
        eager_kernels, "_ensure_tensor_holder", lambda: ModuleType("holder")
    )
    monkeypatch.setattr(eager_kernels, "_build_extension", fail_build)
    monkeypatch.setattr(
        eager_kernels, "MOJO_EXTENSION_LOADER", eager_kernels.MojoExtensionLoader()
    )
    for _ in range(2):
        with pytest.raises(ImportError) as raised:
            call()
        assert raised.value is failure
    assert attempts == 1


def test_tf32_matmul_family_prefers_opt_in_routes(monkeypatch):
    """Eligible public matmul calls return before the TensorSpec fallback."""
    from torch_mojo_backend.eager_kernels import aten_fast

    lhs, rhs, bias, weight = object(), object(), object(), object()
    gemm_result, linear_result, bmm_result = object(), object(), object()
    gemm_calls = []
    linear_calls = []
    bmm_calls = []

    def try_gemm(*args, **kwargs):
        gemm_calls.append((args, kwargs))
        return gemm_result

    def try_linear(*args, **kwargs):
        linear_calls.append((args, kwargs))
        return linear_result

    def try_bmm(*args, **kwargs):
        bmm_calls.append((args, kwargs))
        return bmm_result

    def fail_spec(*_args, **_kwargs):
        raise AssertionError("TF32-routed matmul reached the TensorSpec fallback")

    monkeypatch.setattr(aten_fast, "_try_gemm16_mm", lambda *_args, **_kwargs: None)
    monkeypatch.setattr(aten_fast, "_try_gemm16_linear", lambda *_args, **_kwargs: None)
    monkeypatch.setattr(aten_fast, "_try_gemm16_bmm", lambda *_args, **_kwargs: None)
    monkeypatch.setattr(aten_fast, "_try_tf32_gemm", try_gemm)
    monkeypatch.setattr(aten_fast, "_try_tf32_linear", try_linear)
    monkeypatch.setattr(aten_fast, "_try_tf32_bmm", try_bmm)
    monkeypatch.setattr(aten_fast, "_try_spec_matmul", fail_spec)

    assert aten_fast.fast_aten_mm(lhs, rhs) is gemm_result
    assert aten_fast.fast_aten_addmm(bias, lhs, rhs) is gemm_result
    assert aten_fast.fast_aten_linear(lhs, weight, bias) is linear_result
    assert aten_fast.fast_aten_bmm(lhs, rhs) is bmm_result
    assert aten_fast._fast_aten_bmm_transpose_b(lhs, rhs) is bmm_result

    assert gemm_calls == [((lhs, rhs), {}), ((lhs, rhs, bias), {})]
    assert linear_calls == [((lhs, weight, bias), {})]
    assert bmm_calls == [((lhs, rhs), {}), ((lhs, rhs), {"transpose_b": True})]


def test_tf32_matmul_family_highest_retains_tensorspec_fallback(monkeypatch):
    """Strict FP32 declines TF32 without importing its extension, then uses
    SIMT. (gemm16's own cheap alignment pre-check -- see
    _gemm16_alignment_favors_split -- does inspect operand shapes via `_t`
    even here; that's a deliberate, cheap `isinstance` check, not the
    expensive TF32-extension import this test actually guards.)"""
    from torch_mojo_backend import eager_kernels
    from torch_mojo_backend.eager_kernels import aten_fast

    lhs, rhs, bias, input, weight = (object() for _ in range(5))
    fallback = object()
    spec_calls = []
    tf32_import_calls = []

    def fail_tf32_import(*_args: object, **_kwargs: object) -> ModuleType:
        tf32_import_calls.append("tf32_matmul_ops")
        raise AssertionError("strict FP32 lazily imported the TF32 extension")

    def spec(spec_name, tensors, transpose_b):
        spec_calls.append((spec_name, tensors, transpose_b))
        return fallback

    monkeypatch.setattr(
        eager_kernels.MOJO_EXTENSION_LOADER, "load_canonical", fail_tf32_import
    )
    monkeypatch.setattr(
        aten_fast.torch, "get_float32_matmul_precision", lambda: "highest"
    )
    monkeypatch.setattr(aten_fast, "_try_gemm16_mm", lambda *_args, **_kwargs: None)
    monkeypatch.setattr(aten_fast, "_try_gemm16_linear", lambda *_args, **_kwargs: None)
    monkeypatch.setattr(aten_fast, "_try_gemm16_bmm", lambda *_args, **_kwargs: None)
    monkeypatch.setattr(aten_fast, "_try_spec_matmul", spec)

    assert aten_fast.fast_aten_mm(lhs, rhs) is fallback
    assert aten_fast.fast_aten_addmm(bias, lhs, rhs) is fallback
    assert aten_fast.fast_aten_linear(input, weight, bias) is fallback
    assert aten_fast.fast_aten_bmm(lhs, rhs) is fallback
    assert aten_fast._fast_aten_bmm_transpose_b(lhs, rhs) is fallback

    assert spec_calls == [
        ("MatmulSpec", (lhs, rhs), 0),
        ("MatmulBiasSpec", (lhs, rhs, bias), 0),
        ("MatmulBiasSpec", (input, weight, bias), 1),
        ("BmmSpec", (lhs, rhs), 0),
        ("BmmSpec", (lhs, rhs), 1),
    ]
    assert tf32_import_calls == []


def test_matmul_spec_device_oom_is_not_disguised_as_unsupported(
    monkeypatch, fake_mojo_tensor
):
    from torch_mojo_backend import eager_kernels
    from torch_mojo_backend.eager_kernels import aten_fast
    from torch_mojo_backend.mojo_device.torch_mojo_tensor import TorchMojoTensor

    device = CPU()

    def fake_tensor(shape: tuple[int, ...]) -> TorchMojoTensor:
        return fake_mojo_tensor(device, shape=shape)

    lhs = fake_tensor((2, 3))
    rhs = fake_tensor((3, 4))

    def raise_allocator_oom(*_args: object, **_kwargs: object) -> None:
        raise NotImplementedError(
            "CUDA call failed: CUDA_ERROR_OUT_OF_MEMORY (out of memory)"
        )

    def load_oom_module(
        _mojo_file: Path, _defines: eager_kernels.CanonicalDefines
    ) -> ModuleType:
        module = ModuleType("matmul_oom_test_extension")
        module.call = raise_allocator_oom
        return module

    def fake_allocate_output(
        output_spec: aten_fast._TensorOutputSpec,
    ) -> TorchMojoTensor:
        return fake_tensor(output_spec.shape)

    monkeypatch.setattr(aten_fast, "_t", lambda tensor: tensor)
    monkeypatch.setattr(aten_fast, "_spec_of", lambda tensor: tensor)
    monkeypatch.setattr(aten_fast, "_allocate_output_spec", fake_allocate_output)
    monkeypatch.setattr(
        eager_kernels.MOJO_EXTENSION_LOADER, "load_canonical", load_oom_module
    )

    with pytest.raises(torch.OutOfMemoryError, match="CUDA_ERROR_OUT_OF_MEMORY"):
        aten_fast._try_spec_matmul("MatmulSpec", (lhs, rhs), 0)


def test_tf32_addmm_scalars_retain_existing_not_handled_contract(monkeypatch):
    from torch_mojo_backend.eager_kernels import aten_fast

    calls = []
    monkeypatch.setattr(
        aten_fast, "_try_gemm16_mm", lambda *args, **kwargs: calls.append("gemm16")
    )
    monkeypatch.setattr(
        aten_fast, "_try_tf32_gemm", lambda *args, **kwargs: calls.append("tf32")
    )
    monkeypatch.setattr(
        aten_fast, "_try_spec_matmul", lambda *args, **kwargs: calls.append("spec")
    )

    assert (
        aten_fast.fast_aten_addmm(object(), object(), object(), alpha=2.0)
        is aten_fast.NOT_HANDLED
    )
    assert calls == []


def test_tf32_linear_flattens_contiguous_gpt_input_as_zero_copy_view(monkeypatch):
    from torch_mojo_backend.eager_kernels import aten_fast

    input, weight, bias = object(), object(), object()
    holder = object()
    input_metadata = SimpleNamespace(
        _shape=(2, 3, 4, 8), _is_contiguous=True, _offset=7, _holder=holder
    )
    weight_metadata = SimpleNamespace(_shape=(11, 8))
    matrix_view = SimpleNamespace(_holder=holder)
    result = object()
    view_calls = []
    gemm_calls = []

    def as_tensor(value):
        return {input: input_metadata, weight: weight_metadata}.get(value)

    def view_of(*args, **kwargs):
        view_calls.append((args, kwargs))
        assert args[0]._holder is holder
        return matrix_view

    def try_gemm(*args, **kwargs):
        gemm_calls.append((args, kwargs))
        return result

    monkeypatch.setattr(aten_fast.torch, "get_float32_matmul_precision", lambda: "high")
    monkeypatch.setattr(aten_fast, "_t", as_tensor)
    monkeypatch.setattr(aten_fast, "_view_of", view_of)
    monkeypatch.setattr(aten_fast, "_try_tf32_gemm", try_gemm)

    assert aten_fast._try_tf32_linear(input, weight, bias) is result
    assert gemm_calls[0][0][0]._holder is input_metadata._holder
    assert view_calls == [((input_metadata, (24, 8), (8, 1), 7), {"contiguous": True})]
    assert gemm_calls == [
        (
            (matrix_view, weight, bias),
            {"transpose_b": True, "output_shape": (2, 3, 4, 11)},
        )
    ]


def test_tf32_linear_noncontiguous_batch_retains_tensorspec_path(monkeypatch):
    from torch_mojo_backend.eager_kernels import aten_fast

    input, weight = object(), object()
    input_metadata = SimpleNamespace(_shape=(2, 3, 8), _is_contiguous=False)
    weight_metadata = SimpleNamespace(_shape=(11, 8))
    fallback = object()

    def as_tensor(value):
        return {input: input_metadata, weight: weight_metadata}.get(value)

    def fail_tf32_work(*_args, **_kwargs):
        raise AssertionError("non-contiguous batched linear entered the TF32 path")

    monkeypatch.setattr(aten_fast.torch, "get_float32_matmul_precision", lambda: "high")
    monkeypatch.setattr(aten_fast, "_t", as_tensor)
    monkeypatch.setattr(aten_fast, "_view_of", fail_tf32_work)
    monkeypatch.setattr(aten_fast, "_try_gemm16_linear", lambda *_args, **_kwargs: None)
    monkeypatch.setattr(aten_fast, "_try_tf32_gemm", fail_tf32_work)
    monkeypatch.setattr(
        aten_fast, "_try_spec_matmul", lambda name, tensors, transpose_b: fallback
    )

    assert aten_fast.fast_aten_linear(input, weight) is fallback


@pytest.mark.parametrize("dropout_p", [0.0, 0.25, 1.0])
@pytest.mark.parametrize("tf32_available", [False, True])
def test_sdpa_forward_tf32_bmm_routing_preserves_raw_fallback(
    monkeypatch, tf32_available, dropout_p
):
    """Both SDPA BMMs route the effective probabilities in every dropout mode."""
    from torch_mojo_backend.eager_kernels import aten_fast

    device = SimpleNamespace(label="gpu")
    next_ptr = iter(range(100, 200))

    def tensor(shape, ptr=None, dtype=None):
        shape = tuple(shape)
        return SimpleNamespace(
            _shape=shape,
            _mojo_strides=aten_fast._row_major_strides(shape),
            _offset=0,
            _dtype=aten_fast.DType.float32 if dtype is None else dtype,
            _device=device,
            _ptr=next(next_ptr) if ptr is None else ptr,
            _itemsize=4,
            _numel=math.prod(shape),
            _is_contiguous=True,
        )

    q = tensor((2, 3, 5, 4))
    k = tensor((2, 3, 7, 4))
    v = tensor((2, 3, 7, 4))
    allocations = []
    tf32_calls = []
    raw_calls = []
    softmax_calls = []
    dropout_calls = []
    effective_probability_ptrs = []

    def alloc(shape, dtype, actual_device):
        allocations.append((tuple(shape), dtype, actual_device))
        return tensor(shape)

    def view_of(base, shape, strides, offset, contiguous=None):
        assert tuple(strides) == aten_fast._row_major_strides(shape)
        assert offset == base._offset
        return tensor(shape, base._ptr, base._dtype)

    def try_tf32(lhs, rhs, **kwargs):
        tf32_calls.append((lhs, rhs, kwargs))
        if not tf32_available:
            return None
        batch, rows, inner = lhs._shape
        transpose_b = kwargs.get("transpose_b", False)
        assert (rhs._shape[2] if transpose_b else rhs._shape[1]) == inner
        columns = rhs._shape[1] if transpose_b else rhs._shape[2]
        return tensor((batch, rows, columns))

    def native_dropout(probabilities, probability, train):
        assert probability == dropout_p == 0.25
        assert train is True
        output = tensor(probabilities._shape)
        mask = tensor(probabilities._shape, dtype=aten_fast.DType.bool)
        effective_probability_ptrs.append(output._ptr)
        dropout_calls.append(("native", probabilities._ptr, mask._ptr))
        return output, mask

    def multiply(probabilities, scalar):
        assert dropout_p == 1.0
        assert scalar == 0.0
        output = tensor(probabilities._shape)
        effective_probability_ptrs.append(output._ptr)
        dropout_calls.append(("multiply", probabilities._ptr))
        return output

    def filled(shape, value, dtype, actual_device):
        assert dropout_p == 1.0
        assert value is False
        assert dtype == aten_fast.DType.bool
        assert actual_device is device
        mask = tensor(shape, dtype=dtype)
        dropout_calls.append(("filled", mask._ptr))
        return mask

    monkeypatch.setattr(aten_fast, "_t", lambda value: value)
    monkeypatch.setattr(aten_fast, "_tc", lambda value: value)
    monkeypatch.setattr(aten_fast, "_alloc", alloc)
    monkeypatch.setattr(aten_fast, "_view_of", view_of)
    monkeypatch.setattr(aten_fast, "_try_gemm16_bmm", lambda *_args, **_kwargs: None)
    monkeypatch.setattr(aten_fast, "_try_tf32_bmm", try_tf32)
    monkeypatch.setattr(aten_fast, "_ctx_ptr", lambda _device: 1234)
    monkeypatch.setattr(aten_fast, "fast_aten_native_dropout", native_dropout)
    monkeypatch.setattr(aten_fast, "fast_aten_mul", multiply)
    monkeypatch.setattr(aten_fast, "fast_filled", filled)
    _replace_defined_native_calls(
        monkeypatch,
        {
            ("matmul_ops.mojo", "Bmm"): lambda *args: raw_calls.append(args),
            ("nn_ops.mojo", "SoftmaxRows"): lambda *args: softmax_calls.append(args),
        },
    )

    out, probabilities, mask = aten_fast._sdpa_math_forward_with_dropout(
        q, k, v, False, None, dropout_p
    )

    assert out._shape == (2, 3, 5, 4)
    assert probabilities._shape == (2, 3, 5, 7)
    assert (mask is not None) == (dropout_p > 0.0)
    if mask is not None:
        assert mask._shape == (2, 3, 5, 7)
        assert mask._dtype == aten_fast.DType.bool
    assert [(a._shape, b._shape, kwargs) for a, b, kwargs in tf32_calls] == [
        ((6, 5, 4), (6, 7, 4), {"transpose_b": True}),
        ((6, 5, 7), (6, 7, 4), {}),
    ]
    expected_effective_ptr = (
        probabilities._ptr if dropout_p == 0.0 else effective_probability_ptrs[0]
    )
    assert tf32_calls[1][0]._ptr == expected_effective_ptr
    assert len(softmax_calls) == 1
    if dropout_p == 0.0:
        assert dropout_calls == []
    elif dropout_p == 0.25:
        assert [call[0] for call in dropout_calls] == ["native"]
    else:
        assert [call[0] for call in dropout_calls] == ["multiply", "filled"]
    if tf32_available:
        assert [shape for shape, _, _ in allocations] == [(6, 5, 7)]
        assert raw_calls == []
    else:
        assert [shape for shape, _, _ in allocations] == [
            (6, 5, 7),
            (6, 5, 7),
            (6, 5, 4),
        ]
        assert [call[3] for call in raw_calls] == [(6, 5, 7, 4, 1), (6, 5, 4, 7, 0)]
        assert raw_calls[1][1] == expected_effective_ptr


@pytest.mark.parametrize("lhs_transposed", [False, True])
@pytest.mark.parametrize("rhs_transposed", [False, True])
@pytest.mark.parametrize("transpose_b", [False, True])
def test_bf16_gemm_host_bridge_layouts_offsets_context_and_highest(
    monkeypatch, lhs_transposed, rhs_transposed, transpose_b
):
    """BF16 preserves dense views and is independent of FP32 policy."""
    from torch_mojo_backend.eager_kernels import aten_fast

    device = SimpleNamespace(id=7, label="gpu", api="cuda", architecture_name="sm_90a")
    m, n, k = 6, 7, 5

    def matrix(shape, transposed, base_ptr, offset):
        rows, cols = shape
        return SimpleNamespace(
            _shape=shape,
            _mojo_strides=(1, rows) if transposed else (cols, 1),
            _offset=offset,
            _dtype=aten_fast.DType.bfloat16,
            _device=device,
            _ptr=base_ptr + 2 * offset,
            _itemsize=2,
            _numel=rows * cols,
            _is_contiguous=not transposed,
            _holder=object(),
        )

    lhs = matrix((m, k), lhs_transposed, 1000, 3)
    rhs_shape = (n, k) if transpose_b else (k, n)
    rhs = matrix(rhs_shape, rhs_transposed, 2000, 5)
    bias = SimpleNamespace(
        _shape=(n,),
        _mojo_strides=(1,),
        _offset=2,
        _dtype=aten_fast.DType.bfloat16,
        _device=device,
        _ptr=3004,
        _itemsize=2,
        _numel=n,
        _is_contiguous=True,
        _holder=object(),
    )
    calls = []
    allocations = []
    context_devices = []

    def alloc(shape, dtype, actual_device):
        assert dtype == aten_fast.DType.bfloat16
        assert actual_device is device
        shape = tuple(shape)
        allocations.append(shape)
        return SimpleNamespace(
            _shape=shape,
            _mojo_strides=aten_fast._row_major_strides(shape),
            _offset=0,
            _dtype=dtype,
            _device=actual_device,
            _ptr=9000,
            _itemsize=2,
            _numel=math.prod(shape),
            _is_contiguous=True,
            _holder=object(),
        )

    def context_ptr(actual_device):
        context_devices.append(actual_device)
        return 7007

    def fail_precision_query():
        raise AssertionError("BF16 consulted the float32 matmul precision policy")

    old_precision = torch.get_float32_matmul_precision()
    torch.set_float32_matmul_precision("highest")
    monkeypatch.setattr(
        aten_fast.torch, "get_float32_matmul_precision", fail_precision_query
    )
    monkeypatch.setattr(aten_fast, "_t", lambda value: value)
    monkeypatch.setattr(aten_fast, "_alloc", alloc)
    monkeypatch.setattr(aten_fast, "_ctx_ptr", context_ptr)
    _replace_defined_native_calls(
        monkeypatch,
        {("gemm16_matmul_ops.mojo", "Gemm16"): lambda *args: calls.append(args)},
    )

    try:
        out = aten_fast._try_gemm16_mm(
            lhs, rhs, bias, transpose_b=transpose_b, output_shape=(2, 3, n)
        )
    finally:
        torch.set_float32_matmul_precision(old_precision)

    assert out is not None
    assert out._shape == (2, 3, n)
    assert out._dtype == aten_fast.DType.bfloat16
    assert out._device is device
    assert allocations == [(2, 3, n)]
    assert context_devices == [device]
    assert calls == [
        (
            out._ptr,
            lhs._ptr,
            rhs._ptr,
            bias._ptr,
            m,
            n,
            k,
            int(lhs_transposed),
            int(rhs_transposed) ^ int(transpose_b),
            1,
            7007,
        )
    ]


def test_bf16_gemm_no_bias_uses_ignored_output_pointer(monkeypatch):
    """The 11-argument ABI always receives a valid fourth pointer."""
    from torch_mojo_backend.eager_kernels import aten_fast

    device = SimpleNamespace(label="gpu", api="cuda", architecture_name="sm_90a")

    def tensor(shape, ptr):
        shape = tuple(shape)
        return SimpleNamespace(
            _shape=shape,
            _mojo_strides=aten_fast._row_major_strides(shape),
            _dtype=aten_fast.DType.bfloat16,
            _device=device,
            _ptr=ptr,
            _is_contiguous=True,
        )

    lhs = tensor((6, 5), 1000)
    rhs = tensor((5, 7), 2000)
    output = tensor((6, 7), 9000)
    calls = []

    monkeypatch.setattr(aten_fast, "_t", lambda value: value)
    monkeypatch.setattr(aten_fast, "_alloc", lambda *_args: output)
    monkeypatch.setattr(aten_fast, "_ctx_ptr", lambda actual_device: 7007)
    _replace_defined_native_calls(
        monkeypatch,
        {("gemm16_matmul_ops.mojo", "Gemm16"): lambda *args: calls.append(args)},
    )

    assert aten_fast._try_gemm16_mm(lhs, rhs) is output
    assert len(calls) == 1
    assert calls[0][3] == output._ptr
    assert calls[0][9] == 0


# The 16-bit tensor-core GEMM family serves bfloat16 and float16 from one
# source, specialized at compile time by DTYPE_ARG_0.  These tests pin the
# float16 half of that contract: the same routes, the same regime dispatch and
# the same FP32 accumulation, at every layout the family claims.
_GEMM16_LAYOUTS = ("NN", "NT", "TN", "TT")


def _gemm16_operands(
    layout: str, m: int, n: int, k: int, dtype: torch.dtype, device: torch.device
) -> tuple[torch.Tensor, torch.Tensor]:
    """A (M,K) and B (K,N) as dense views with the requested physical layout."""

    def view(rows: int, cols: int, transposed: bool) -> torch.Tensor:
        host = torch.randn(cols, rows) if transposed else torch.randn(rows, cols)
        moved = host.to(dtype).to(device)
        return moved.t() if transposed else moved

    return view(m, k, layout[0] == "T"), view(k, n, layout[1] == "T")


@pytest.mark.parametrize("layout", _GEMM16_LAYOUTS)
def test_gemm16_float16_mm_matches_fp32_reference_on_every_layout(
    mojo_h100: torch.device, layout: str
) -> None:
    """float16 reaches the tensor-core family, not the generic tiled GEMM.

    The tolerance is the point of the assertion: the kernels accumulate in
    FP32, so a 333-deep contraction stays near 1e-3 relative.  A kernel that
    accumulated in float16 instead would miss it by an order of magnitude.
    """
    from torch_mojo_backend.eager_kernels import aten_fast

    m, n, k = 357, 789, 333  # awkward on purpose: exercises the edge tiles
    a, b = _gemm16_operands(layout, m, n, k, torch.float16, mojo_h100)

    out = aten_fast._try_gemm16_mm(a, b)
    assert out is not None, "float16 declined by the tensor-core GEMM family"
    assert out.dtype == torch.float16
    assert tuple(out.shape) == (m, n)
    assert out.stride() == (n, 1)

    expected = a.cpu().float() @ b.cpu().float()
    # atol covers the two error floors of a k=333 contraction of ~N(0,1)
    # operands (|result| ~ 18): the FP32 partial sums reordered against the
    # CPU reference (~2e-5) and the final round to float16 (half an ulp,
    # ~8e-3).  Both are ~20x below what accumulating in float16 would give,
    # which is what this bound is here to reject.
    torch.testing.assert_close(out.cpu().float(), expected, atol=2e-2, rtol=2e-3)


@pytest.mark.parametrize("op", ["mm", "addmm", "linear", "bmm", "bmm_transpose_b"])
def test_gemm16_float16_every_entry_point_launches_the_bridge(
    mojo_h100: torch.device, monkeypatch: pytest.MonkeyPatch, op: str
) -> None:
    """mm / addmm / linear / bmm all route float16 through the same bridge."""
    from torch_mojo_backend.eager_kernels import aten_fast

    m, n, k, batch = 64, 96, 128, 3
    calls = _spy_defined_native_calls(
        monkeypatch,
        {("gemm16_matmul_ops.mojo", "Gemm16"), ("gemm16_matmul_ops.mojo", "Bmm16")},
    )

    def half(*shape: int) -> torch.Tensor:
        return torch.randn(*shape).to(torch.float16).to(mojo_h100)

    if op == "mm":
        a, b = half(m, k), half(k, n)
        actual, expected = torch.mm(a, b), a.cpu().float() @ b.cpu().float()
        launched = ("gemm16_matmul_ops.mojo", "Gemm16")
    elif op == "addmm":
        a, b, c = half(m, k), half(k, n), half(n)
        actual = torch.addmm(c, a, b)
        expected = c.cpu().float() + a.cpu().float() @ b.cpu().float()
        launched = ("gemm16_matmul_ops.mojo", "Gemm16")
    elif op == "linear":
        x, w, c = half(2, m, k), half(n, k), half(n)
        actual = torch.nn.functional.linear(x, w, c)
        expected = torch.nn.functional.linear(
            x.cpu().float(), w.cpu().float(), c.cpu().float()
        )
        launched = ("gemm16_matmul_ops.mojo", "Gemm16")
    elif op == "bmm":
        a, b = half(batch, m, k), half(batch, k, n)
        actual, expected = torch.bmm(a, b), torch.bmm(a.cpu().float(), b.cpu().float())
        launched = ("gemm16_matmul_ops.mojo", "Bmm16")
    else:
        a, b = half(batch, m, k), half(batch, n, k)
        actual = aten_fast._fast_aten_bmm_transpose_b(a, b)
        expected = torch.bmm(a.cpu().float(), b.cpu().float().transpose(1, 2))
        launched = ("gemm16_matmul_ops.mojo", "Bmm16")

    assert actual.dtype == torch.float16
    assert len(calls[launched]) == 1, f"{op} did not reach the 16-bit GEMM bridge"
    torch.testing.assert_close(actual.cpu().float(), expected, atol=2e-2, rtol=2e-3)


def test_gemm16_float16_and_bfloat16_are_separate_specializations(
    mojo_h100: torch.device, monkeypatch: pytest.MonkeyPatch
) -> None:
    """One source, one .so per dtype: the define travels, the kernel does not.

    Mixing the two dtypes in one call must decline rather than reinterpret the
    operand bits, which have different exponent widths.
    """
    from torch_mojo_backend.eager_kernels import aten_fast

    calls = _spy_defined_native_calls(
        monkeypatch, {("gemm16_matmul_ops.mojo", "Gemm16")}
    )
    half_a = torch.randn(64, 128).to(torch.float16).to(mojo_h100)
    half_b = torch.randn(128, 96).to(torch.float16).to(mojo_h100)
    brain_b = torch.randn(128, 96).to(torch.bfloat16).to(mojo_h100)

    assert aten_fast._try_gemm16_mm(half_a, brain_b) is None
    assert not calls[("gemm16_matmul_ops.mojo", "Gemm16")]

    assert aten_fast._try_gemm16_mm(half_a, half_b) is not None
    assert aten_fast._try_gemm16_mm(half_a.bfloat16(), brain_b) is not None
    assert len(calls[("gemm16_matmul_ops.mojo", "Gemm16")]) == 2


def test_gemm16_float16_linear_backward_matches_fp32_reference(
    mojo_h100: torch.device,
) -> None:
    """linear_backward's dgrad/wgrad pair is the NN and TN route in float16."""
    from torch_mojo_backend.eager_kernels import aten_fast

    rows, in_features, out_features = 128, 192, 96

    def half(*shape: int) -> torch.Tensor:
        return torch.randn(*shape).to(torch.float16)

    host_input = half(rows, in_features)
    host_weight = half(out_features, in_features)
    host_grad = half(rows, out_features)

    grad_input, grad_weight, grad_bias = aten_fast.fast_aten_linear_backward(
        host_input.to(mojo_h100),
        host_grad.to(mojo_h100),
        host_weight.to(mojo_h100),
        [True, True, True],
    )
    assert grad_input is not aten_fast.NOT_HANDLED
    assert grad_input.dtype == torch.float16
    assert grad_weight.dtype == torch.float16

    torch.testing.assert_close(
        grad_input.cpu().float(),
        host_grad.float() @ host_weight.float(),
        atol=2e-2,
        rtol=2e-3,
    )
    torch.testing.assert_close(
        grad_weight.cpu().float(),
        host_grad.float().t() @ host_input.float(),
        atol=2e-2,
        rtol=2e-3,
    )
    torch.testing.assert_close(
        grad_bias.cpu().float(), host_grad.float().sum(0), atol=2e-2, rtol=2e-3
    )


@pytest.mark.parametrize(
    "invalid_case",
    [
        "bias_non_mojo",
        "bias_shape",
        "bias_dtype",
        "bias_device",
        "bias_noncontiguous",
        "rank",
        "inner",
        "zero",
        "dtype",
        "api",
        "architecture",
        "inner_stride",
        "output_shape",
    ],
)
def test_bf16_gemm_rejects_invalid_metadata_before_resolve_or_allocation(
    monkeypatch, invalid_case
):
    from torch_mojo_backend.eager_kernels import aten_fast

    h100 = SimpleNamespace(id=0, label="gpu", api="cuda", architecture_name="sm_90a")
    other = SimpleNamespace(id=1, label="gpu", api="cuda", architecture_name="sm_90a")

    def tensor(
        shape,
        *,
        device=h100,
        dtype=aten_fast.DType.bfloat16,
        strides=None,
        contiguous=True,
    ):
        shape = tuple(shape)
        return SimpleNamespace(
            _shape=shape,
            _mojo_strides=(
                aten_fast._row_major_strides(shape) if strides is None else strides
            ),
            _dtype=dtype,
            _device=device,
            _ptr=1234,
            _is_contiguous=contiguous,
        )

    lhs = tensor((6, 5))
    rhs = tensor((5, 7))
    bias = None
    output_shape = None
    non_mojo_bias = object()
    if invalid_case == "bias_non_mojo":
        bias = non_mojo_bias
    elif invalid_case == "bias_shape":
        bias = tensor((8,))
    elif invalid_case == "bias_dtype":
        bias = tensor((7,), dtype=aten_fast.DType.float16)
    elif invalid_case == "bias_device":
        bias = tensor((7,), device=other)
    elif invalid_case == "bias_noncontiguous":
        bias = tensor((7,), strides=(2,), contiguous=False)
    elif invalid_case == "rank":
        lhs = tensor((1, 6, 5))
    elif invalid_case == "inner":
        rhs = tensor((4, 7))
    elif invalid_case == "zero":
        lhs = tensor((0, 5))
    elif invalid_case == "dtype":
        rhs = tensor((5, 7), dtype=aten_fast.DType.float16)
    elif invalid_case == "api":
        non_cuda = SimpleNamespace(label="gpu", api="hip", architecture_name="gfx942")
        lhs = tensor((6, 5), device=non_cuda)
        rhs = tensor((5, 7), device=non_cuda)
    elif invalid_case == "architecture":
        non_h100 = SimpleNamespace(label="gpu", api="cuda", architecture_name="sm_89")
        lhs = tensor((6, 5), device=non_h100)
        rhs = tensor((5, 7), device=non_h100)
    elif invalid_case == "inner_stride":
        lhs = tensor((6, 5), strides=(10, 2), contiguous=False)
    else:
        output_shape = (5, 7)

    def as_tensor(value):
        return None if value is non_mojo_bias else value

    def fail_late_path(*_args, **_kwargs):
        raise AssertionError("invalid BF16 GEMM metadata reached a late path")

    monkeypatch.setattr(aten_fast, "_t", as_tensor)
    monkeypatch.setattr(aten_fast, "_call_mojo", fail_late_path)
    monkeypatch.setattr(aten_fast, "_alloc", fail_late_path)
    monkeypatch.setattr(aten_fast, "_ctx_ptr", fail_late_path)

    assert aten_fast._try_gemm16_mm(lhs, rhs, bias, output_shape=output_shape) is None


def test_bf16_linear_flattens_contiguous_gpt_input_without_precision_query(monkeypatch):
    from torch_mojo_backend.eager_kernels import aten_fast

    input, weight, bias = object(), object(), object()
    holder = object()
    input_metadata = SimpleNamespace(
        _shape=(2, 3, 4, 8), _is_contiguous=True, _offset=7, _holder=holder
    )
    weight_metadata = SimpleNamespace(_shape=(11, 8))
    matrix_view = SimpleNamespace(_holder=holder)
    result = object()
    view_calls = []
    gemm_calls = []

    def as_tensor(value):
        # `is` comparisons, not a dict keyed by `value`: matrix_view (passed
        # to _gemm16_alignment_favors_split too) is an unhashable
        # SimpleNamespace.
        if value is input:
            return input_metadata
        if value is weight:
            return weight_metadata
        return None

    def view_of(*args, **kwargs):
        view_calls.append((args, kwargs))
        return matrix_view

    def try_gemm(*args, **kwargs):
        gemm_calls.append((args, kwargs))
        return result

    def fail_precision_query():
        raise AssertionError("BF16 linear consulted the float32 precision policy")

    monkeypatch.setattr(
        aten_fast.torch, "get_float32_matmul_precision", fail_precision_query
    )
    monkeypatch.setattr(aten_fast, "_t", as_tensor)
    monkeypatch.setattr(aten_fast, "_view_of", view_of)
    monkeypatch.setattr(aten_fast, "_try_gemm16_mm", try_gemm)

    # matrix_view/weight are SimpleNamespaces `_t` doesn't resolve to a real
    # tensor, so _gemm16_alignment_favors_split declines and linear takes
    # its single bias-fused call, same as before this fix.
    assert aten_fast._try_gemm16_linear(input, weight, bias) is result
    assert view_calls == [((input_metadata, (24, 8), (8, 1), 7), {"contiguous": True})]
    assert gemm_calls == [
        (
            (matrix_view, weight, bias),
            {"transpose_b": True, "output_shape": (2, 3, 4, 11)},
        )
    ]


@pytest.mark.parametrize("lhs_transposed", [False, True])
@pytest.mark.parametrize("rhs_transposed", [False, True])
@pytest.mark.parametrize("transpose_b", [False, True])
def test_bf16_bmm_host_bridge_padded_layouts_offsets_and_logical_transpose(
    monkeypatch, lhs_transposed, rhs_transposed, transpose_b
):
    from torch_mojo_backend.eager_kernels import aten_fast

    device = SimpleNamespace(id=9, label="gpu", api="cuda", architecture_name="sm_90a")
    batch, m, n, k = 3, 7, 5, 9

    def batched(shape, transposed, gap, base_ptr, offset):
        batches, rows, cols = shape
        batch_stride = rows * cols + gap
        return (
            SimpleNamespace(
                _shape=shape,
                _mojo_strides=(batch_stride, 1, rows)
                if transposed
                else (batch_stride, cols, 1),
                _offset=offset,
                _dtype=aten_fast.DType.bfloat16,
                _device=device,
                _ptr=base_ptr + 2 * offset,
                _itemsize=2,
                _numel=batches * rows * cols,
                _is_contiguous=False,
                _holder=object(),
            ),
            batch_stride,
        )

    lhs, lhs_batch_stride = batched(
        (batch, m, k), lhs_transposed, gap=11, base_ptr=1000, offset=3
    )
    rhs_shape = (batch, n, k) if transpose_b else (batch, k, n)
    rhs, rhs_batch_stride = batched(
        rhs_shape, rhs_transposed, gap=17, base_ptr=2000, offset=6
    )
    calls = []
    allocations = []
    context_devices = []

    def alloc(shape, dtype, actual_device):
        shape = tuple(shape)
        allocations.append((shape, dtype, actual_device))
        return SimpleNamespace(
            _shape=shape,
            _mojo_strides=aten_fast._row_major_strides(shape),
            _offset=0,
            _dtype=dtype,
            _device=actual_device,
            _ptr=9000,
            _itemsize=2,
            _numel=math.prod(shape),
            _is_contiguous=True,
            _holder=object(),
        )

    def context_ptr(actual_device):
        context_devices.append(actual_device)
        return 8009

    def fail_precision_query():
        raise AssertionError("BF16 BMM consulted the float32 precision policy")

    monkeypatch.setattr(
        aten_fast.torch, "get_float32_matmul_precision", fail_precision_query
    )
    monkeypatch.setattr(aten_fast, "_t", lambda value: value)
    monkeypatch.setattr(aten_fast, "_alloc", alloc)
    monkeypatch.setattr(aten_fast, "_ctx_ptr", context_ptr)
    _replace_defined_native_calls(
        monkeypatch,
        {("gemm16_matmul_ops.mojo", "Bmm16"): lambda *args: calls.append(args)},
    )

    out = aten_fast._try_gemm16_bmm(lhs, rhs, transpose_b=transpose_b)

    assert out is not None
    assert out._shape == (batch, m, n)
    assert out._dtype == aten_fast.DType.bfloat16
    assert allocations == [((batch, m, n), aten_fast.DType.bfloat16, device)]
    assert context_devices == [device]
    assert calls == [
        (
            out._ptr,
            lhs._ptr,
            rhs._ptr,
            batch,
            m,
            n,
            k,
            m * n,
            lhs_batch_stride,
            rhs_batch_stride,
            int(lhs_transposed),
            int(rhs_transposed) ^ int(transpose_b),
            8009,
        )
    ]


@pytest.mark.parametrize(
    "invalid_case",
    [
        "inner_stride",
        "overlap",
        "batch",
        "inner",
        "rank",
        "zero",
        "dtype",
        "api",
        "architecture",
    ],
)
def test_bf16_bmm_rejects_invalid_metadata_before_resolve_or_allocation(
    monkeypatch, invalid_case
):
    from torch_mojo_backend.eager_kernels import aten_fast

    h100 = SimpleNamespace(label="gpu", api="cuda", architecture_name="sm_90a")

    def tensor(shape, *, device=h100, dtype=aten_fast.DType.bfloat16, strides=None):
        shape = tuple(shape)
        return SimpleNamespace(
            _shape=shape,
            _mojo_strides=(
                aten_fast._row_major_strides(shape) if strides is None else strides
            ),
            _dtype=dtype,
            _device=device,
            _ptr=1234,
            _is_contiguous=strides is None,
        )

    lhs = tensor((2, 7, 9))
    rhs = tensor((2, 9, 5))
    if invalid_case == "inner_stride":
        lhs = tensor((2, 7, 9), strides=(200, 20, 2))
    elif invalid_case == "overlap":
        lhs = tensor((2, 7, 9), strides=(62, 9, 1))
    elif invalid_case == "batch":
        rhs = tensor((3, 9, 5))
    elif invalid_case == "inner":
        rhs = tensor((2, 8, 5))
    elif invalid_case == "rank":
        rhs = tensor((9, 5))
    elif invalid_case == "zero":
        lhs = tensor((0, 7, 9))
        rhs = tensor((0, 9, 5))
    elif invalid_case == "dtype":
        rhs = tensor((2, 9, 5), dtype=aten_fast.DType.float16)
    elif invalid_case == "api":
        non_cuda = SimpleNamespace(label="gpu", api="hip", architecture_name="gfx942")
        lhs = tensor((2, 7, 9), device=non_cuda)
        rhs = tensor((2, 9, 5), device=non_cuda)
    elif invalid_case == "architecture":
        non_h100 = SimpleNamespace(label="gpu", api="cuda", architecture_name="sm_89")
        lhs = tensor((2, 7, 9), device=non_h100)
        rhs = tensor((2, 9, 5), device=non_h100)

    def fail_late_path(*_args, **_kwargs):
        raise AssertionError("invalid BF16 BMM metadata reached a late path")

    monkeypatch.setattr(aten_fast, "_t", lambda value: value)
    monkeypatch.setattr(aten_fast, "_call_mojo", fail_late_path)
    monkeypatch.setattr(aten_fast, "_alloc", fail_late_path)
    monkeypatch.setattr(aten_fast, "_ctx_ptr", fail_late_path)

    assert aten_fast._try_gemm16_bmm(lhs, rhs) is None


@pytest.mark.parametrize("operation", ["gemm", "bmm"])
def test_bf16_bridge_error_propagates_without_retry(monkeypatch, operation):
    """A failed enqueue must not allocate or invoke the bridge a second time."""
    from torch_mojo_backend.eager_kernels import aten_fast

    device = SimpleNamespace(label="gpu", api="cuda", architecture_name="sm_90a")

    def tensor(shape, ptr):
        shape = tuple(shape)
        return SimpleNamespace(
            _shape=shape,
            _mojo_strides=aten_fast._row_major_strides(shape),
            _dtype=aten_fast.DType.bfloat16,
            _device=device,
            _ptr=ptr,
            _is_contiguous=True,
        )

    allocations = []
    bridge_calls = []

    def allocate(shape, dtype, actual_device):
        allocations.append((tuple(shape), dtype, actual_device))
        return tensor(shape, 9000)

    def fail_bridge(*args, **kwargs):
        bridge_calls.append((args, kwargs))
        raise RuntimeError("synthetic BF16 enqueue failure")

    monkeypatch.setattr(aten_fast, "_t", lambda value: value)
    monkeypatch.setattr(aten_fast, "_alloc", allocate)
    monkeypatch.setattr(aten_fast, "_ctx_ptr", lambda actual_device: 7007)
    monkeypatch.setattr(aten_fast, "_call_mojo", fail_bridge)

    with pytest.raises(RuntimeError, match="synthetic BF16 enqueue failure"):
        if operation == "gemm":
            aten_fast._try_gemm16_mm(tensor((6, 5), 1000), tensor((5, 7), 2000))
        else:
            aten_fast._try_gemm16_bmm(tensor((2, 6, 5), 1000), tensor((2, 5, 7), 2000))

    assert len(allocations) == 1
    assert len(bridge_calls) == 1


@pytest.mark.parametrize("invalid_case", ["gemm_rhs", "gemm_bias", "bmm_rhs"])
def test_bf16_helpers_reject_cross_device_before_resolve_or_allocation(
    monkeypatch, invalid_case
):
    from torch_mojo_backend.eager_kernels import aten_fast

    devices = [
        SimpleNamespace(
            id=device_id, label="gpu", api="cuda", architecture_name="sm_90a"
        )
        for device_id in (4, 9)
    ]

    def tensor(shape, device):
        shape = tuple(shape)
        return SimpleNamespace(
            _shape=shape,
            _mojo_strides=aten_fast._row_major_strides(shape),
            _offset=0,
            _dtype=aten_fast.DType.bfloat16,
            _device=device,
            _ptr=1000 + device.id,
            _itemsize=2,
            _numel=math.prod(shape),
            _is_contiguous=True,
            _holder=object(),
        )

    def fail_late_path(*_args, **_kwargs):
        raise AssertionError("cross-device BF16 input reached resolve/allocation")

    monkeypatch.setattr(aten_fast, "_t", lambda value: value)
    monkeypatch.setattr(aten_fast, "_call_mojo", fail_late_path)
    monkeypatch.setattr(aten_fast, "_alloc", fail_late_path)
    monkeypatch.setattr(aten_fast, "_ctx_ptr", fail_late_path)

    if invalid_case == "gemm_rhs":
        result = aten_fast._try_gemm16_mm(
            tensor((3, 4), devices[0]), tensor((4, 5), devices[1])
        )
    elif invalid_case == "gemm_bias":
        result = aten_fast._try_gemm16_mm(
            tensor((3, 4), devices[0]),
            tensor((4, 5), devices[0]),
            tensor((5,), devices[1]),
        )
    else:
        result = aten_fast._try_gemm16_bmm(
            tensor((2, 3, 4), devices[0]), tensor((2, 4, 5), devices[1])
        )
    assert result is None


def test_tf32_helpers_route_each_fake_device_context_and_reject_cross_device(
    monkeypatch,
):
    """Context selection is operand-local and never falls back to device zero."""
    from torch_mojo_backend.eager_kernels import aten_fast

    devices = [
        SimpleNamespace(
            id=device_id, label="gpu", api="cuda", architecture_name="sm_90a"
        )
        for device_id in (4, 9)
    ]
    next_ptr = iter(range(1000, 1100))
    allocations = []
    context_devices = []
    gemm_calls = []
    bmm_calls = []

    def tensor(shape, device, ptr=None):
        shape = tuple(shape)
        return SimpleNamespace(
            _shape=shape,
            _mojo_strides=aten_fast._row_major_strides(shape),
            _offset=0,
            _dtype=aten_fast.DType.float32,
            _device=device,
            _ptr=next(next_ptr) if ptr is None else ptr,
            _itemsize=4,
            _numel=math.prod(shape),
            _is_contiguous=True,
            _holder=object(),
        )

    def alloc(shape, dtype, device):
        assert dtype == aten_fast.DType.float32
        allocations.append((tuple(shape), device))
        return tensor(shape, device)

    def context_ptr(device):
        context_devices.append(device)
        return 8000 + device.id

    monkeypatch.setattr(aten_fast.torch, "get_float32_matmul_precision", lambda: "high")
    monkeypatch.setattr(aten_fast, "_t", lambda value: value)
    monkeypatch.setattr(aten_fast, "_alloc", alloc)
    monkeypatch.setattr(aten_fast, "_ctx_ptr", context_ptr)
    _replace_defined_native_calls(
        monkeypatch,
        {
            ("tf32_matmul_ops.mojo", "Tf32GemmF32"): (
                lambda *args: gemm_calls.append(args)
            ),
            ("tf32_matmul_ops.mojo", "Tf32BmmF32"): (
                lambda *args: bmm_calls.append(args)
            ),
        },
    )

    per_device_tensors = []
    for device in devices:
        lhs = tensor((3, 4), device)
        rhs = tensor((4, 5), device)
        batched_lhs = tensor((2, 3, 4), device)
        batched_rhs = tensor((2, 4, 5), device)
        per_device_tensors.append((lhs, rhs, batched_lhs, batched_rhs))

        gemm_output = aten_fast._try_tf32_gemm(lhs, rhs)
        bmm_output = aten_fast._try_tf32_bmm(batched_lhs, batched_rhs)
        assert gemm_output._device is device
        assert bmm_output._device is device
        assert gemm_calls[-1][-1] == 8000 + device.id
        assert bmm_calls[-1][-1] == 8000 + device.id

    assert context_devices == [devices[0], devices[0], devices[1], devices[1]]
    assert allocations == [
        ((3, 5), devices[0]),
        ((2, 3, 5), devices[0]),
        ((3, 5), devices[1]),
        ((2, 3, 5), devices[1]),
    ]

    allocation_count = len(allocations)
    context_count = len(context_devices)
    gemm_count = len(gemm_calls)
    bmm_count = len(bmm_calls)
    lhs0, rhs0, batched_lhs0, batched_rhs0 = per_device_tensors[0]
    lhs1, rhs1, batched_lhs1, batched_rhs1 = per_device_tensors[1]
    assert aten_fast._try_tf32_gemm(lhs0, rhs1) is None
    assert aten_fast._try_tf32_gemm(lhs1, rhs0) is None
    assert aten_fast._try_tf32_gemm(lhs0, rhs0, tensor((5,), devices[1])) is None
    assert aten_fast._try_tf32_bmm(batched_lhs0, batched_rhs1) is None
    assert aten_fast._try_tf32_bmm(batched_lhs1, batched_rhs0) is None
    assert len(allocations) == allocation_count
    assert len(context_devices) == context_count
    assert len(gemm_calls) == gemm_count
    assert len(bmm_calls) == bmm_count


@pytest.mark.parametrize("lhs_transposed", [False, True])
@pytest.mark.parametrize("rhs_transposed", [False, True])
@pytest.mark.parametrize("transpose_b", [False, True])
def test_tf32_gemm_host_bridge_layouts(
    mojo_h100, monkeypatch, lhs_transposed, rhs_transposed, transpose_b
):
    """The host helper preserves dense views and passes physical layouts."""
    from torch_mojo_backend.eager_kernels import aten_fast

    m, n, k = 6, 7, 5

    def dense_view(shape, transposed):
        if transposed:
            return torch.randn(shape[1], shape[0]).to(mojo_h100).t()
        return torch.randn(*shape).to(mojo_h100)

    lhs = dense_view((m, k), lhs_transposed)
    rhs_shape = (n, k) if transpose_b else (k, n)
    rhs = dense_view(rhs_shape, rhs_transposed)
    bias = torch.randn(n).to(mojo_h100)
    calls = []

    def record(*args):
        calls.append(args)

    _replace_defined_native_calls(
        monkeypatch, {("tf32_matmul_ops.mojo", "Tf32GemmF32"): record}
    )
    old_precision = torch.get_float32_matmul_precision()
    torch.set_float32_matmul_precision("high")
    try:
        out = aten_fast._try_tf32_gemm(
            lhs, rhs, bias, transpose_b=transpose_b, output_shape=(2, 3, n)
        )
    finally:
        torch.set_float32_matmul_precision(old_precision)

    assert out is not None
    assert tuple(out.shape) == (2, 3, n)
    assert out.dtype == torch.float32
    assert out.device == torch.device(mojo_h100)
    assert len(calls) == 1
    args = calls[0]
    assert args[:7] == (out._ptr, lhs._ptr, rhs._ptr, bias._ptr, m, n, k)
    assert args[7:10] == (
        int(lhs_transposed),
        int(rhs_transposed) ^ int(transpose_b),
        1,
    )
    assert args[10] == aten_fast._ctx_ptr(lhs._device)


def test_tf32_gemm_host_bridge_strict_fp32_never_launches(mojo_gpu, monkeypatch):
    """The ``highest`` policy must retain the strict-FP32 SIMT path."""
    from torch_mojo_backend.eager_kernels import aten_fast

    calls = []
    _replace_defined_native_calls(
        monkeypatch,
        {("tf32_matmul_ops.mojo", "Tf32GemmF32"): lambda *args: calls.append(args)},
    )

    def fail_allocation(*_args, **_kwargs):
        raise AssertionError("strict FP32 GEMM allocated a TF32 output")

    monkeypatch.setattr(aten_fast, "_alloc", fail_allocation)
    lhs = torch.randn(6, 5).to(mojo_gpu)
    rhs = torch.randn(5, 7).to(mojo_gpu)
    old_precision = torch.get_float32_matmul_precision()
    torch.set_float32_matmul_precision("highest")
    try:
        assert aten_fast._try_tf32_gemm(lhs, rhs) is None
    finally:
        torch.set_float32_matmul_precision(old_precision)
    assert calls == []


def test_tf32_gemm_host_bridge_no_bias_uses_ignored_output_pointer(
    mojo_h100, monkeypatch
):
    from torch_mojo_backend.eager_kernels import aten_fast

    calls = []
    _replace_defined_native_calls(
        monkeypatch,
        {("tf32_matmul_ops.mojo", "Tf32GemmF32"): lambda *args: calls.append(args)},
    )
    lhs = torch.randn(6, 5).to(mojo_h100)
    rhs = torch.randn(5, 7).to(mojo_h100)
    old_precision = torch.get_float32_matmul_precision()
    torch.set_float32_matmul_precision("medium")
    try:
        out = aten_fast._try_tf32_gemm(lhs, rhs)
    finally:
        torch.set_float32_matmul_precision(old_precision)

    assert out is not None
    assert len(calls) == 1
    assert calls[0][3] == out._ptr
    assert calls[0][9] == 0


def test_tf32_gemm_rejects_non_mojo_bias_before_operand_inspection(monkeypatch):
    """A supplied CPU bias cannot be silently reinterpreted as no bias."""
    from torch_mojo_backend.eager_kernels import aten_fast

    lhs = object()
    rhs = object()
    cpu_bias = torch.randn(7)
    mojo_metadata = object()

    def fake_tensor(value):
        return mojo_metadata if value is lhs or value is rhs else None

    def fail_late_path(*_args, **_kwargs):
        raise AssertionError("invalid bias reached layout inspection or allocation")

    monkeypatch.setattr(aten_fast, "_t", fake_tensor)
    monkeypatch.setattr(aten_fast, "_tf32_dense_2d_layout", fail_late_path)
    monkeypatch.setattr(aten_fast, "_alloc", fail_late_path)
    old_precision = torch.get_float32_matmul_precision()
    torch.set_float32_matmul_precision("high")
    try:
        assert aten_fast._try_tf32_gemm(lhs, rhs, cpu_bias) is None
    finally:
        torch.set_float32_matmul_precision(old_precision)


@pytest.mark.parametrize(
    "invalid_case",
    [
        "bias_non_mojo",
        "bias_shape",
        "bias_dtype",
        "bias_device",
        "rank",
        "inner",
        "zero",
        "dtype",
        "device",
        "inner_stride",
        "output_shape",
    ],
)
def test_tf32_gemm_host_bridge_rejects_before_allocation(
    mojo_h100, monkeypatch, invalid_case
):
    from torch_mojo_backend.eager_kernels import aten_fast

    calls = []
    _replace_defined_native_calls(
        monkeypatch,
        {("tf32_matmul_ops.mojo", "Tf32GemmF32"): lambda *args: calls.append(args)},
    )

    def fail_allocation(*_args, **_kwargs):
        raise AssertionError("unsupported TF32 GEMM metadata allocated output")

    monkeypatch.setattr(aten_fast, "_alloc", fail_allocation)
    lhs = torch.randn(6, 5).to(mojo_h100)
    rhs = torch.randn(5, 7).to(mojo_h100)
    bias = None
    output_shape = None
    if invalid_case == "bias_non_mojo":
        bias = torch.randn(7)
    elif invalid_case == "bias_shape":
        bias = torch.randn(8).to(mojo_h100)
    elif invalid_case == "bias_dtype":
        bias = torch.randn(7, dtype=torch.float16).to(mojo_h100)
    elif invalid_case == "bias_device":
        mojo_cpu = f"mojo:{len(get_accelerators()) - 1}"
        bias = torch.randn(7).to(mojo_cpu)
    elif invalid_case == "rank":
        lhs = torch.randn(1, 6, 5).to(mojo_h100)
    elif invalid_case == "inner":
        rhs = torch.randn(4, 7).to(mojo_h100)
    elif invalid_case == "zero":
        lhs = torch.empty(0, 5).to(mojo_h100)
    elif invalid_case == "dtype":
        rhs = torch.randn(5, 7, dtype=torch.float16).to(mojo_h100)
    elif invalid_case == "device":
        mojo_cpu = f"mojo:{len(get_accelerators()) - 1}"
        rhs = torch.randn(5, 7).to(mojo_cpu)
    elif invalid_case == "inner_stride":
        storage = torch.empty(128).to(mojo_h100)
        lhs = aten_fast._view_of(storage, (6, 5), (10, 2), 1)
    else:
        output_shape = (5, 7)

    old_precision = torch.get_float32_matmul_precision()
    torch.set_float32_matmul_precision("high")
    try:
        assert (
            aten_fast._try_tf32_gemm(lhs, rhs, bias, output_shape=output_shape) is None
        )
    finally:
        torch.set_float32_matmul_precision(old_precision)
    assert calls == []


@pytest.mark.parametrize("lhs_transposed", [False, True])
@pytest.mark.parametrize("rhs_transposed", [False, True])
@pytest.mark.parametrize("transpose_b", [False, True])
def test_tf32_bmm_host_bridge_strided_dense_layouts(
    mojo_h100, monkeypatch, lhs_transposed, rhs_transposed, transpose_b
):
    """The dormant BMM bridge preserves offsets, batch gaps, and layouts."""
    from torch_mojo_backend.eager_kernels import aten_fast

    batch, m, n, k = 3, 7, 5, 9

    def dense_batched_view(shape, transposed, gap, offset):
        batches, rows, cols = shape
        matrix_elements = rows * cols
        batch_stride = matrix_elements + gap
        storage_elements = offset + (batches - 1) * batch_stride + matrix_elements
        storage = torch.empty(storage_elements + 4).to(mojo_h100)
        strides = (batch_stride, 1, rows) if transposed else (batch_stride, cols, 1)
        return aten_fast._view_of(storage, shape, strides, offset), batch_stride

    lhs, lhs_batch_stride = dense_batched_view(
        (batch, m, k), lhs_transposed, gap=5, offset=1
    )
    rhs_shape = (batch, n, k) if transpose_b else (batch, k, n)
    rhs, rhs_batch_stride = dense_batched_view(
        rhs_shape, rhs_transposed, gap=7, offset=2
    )
    calls = []
    _replace_defined_native_calls(
        monkeypatch,
        {("tf32_matmul_ops.mojo", "Tf32BmmF32"): lambda *args: calls.append(args)},
    )
    old_precision = torch.get_float32_matmul_precision()
    torch.set_float32_matmul_precision("high")
    try:
        out = aten_fast._try_tf32_bmm(lhs, rhs, transpose_b=transpose_b)
    finally:
        torch.set_float32_matmul_precision(old_precision)

    assert out is not None
    assert tuple(out.shape) == (batch, m, n)
    assert out.dtype == torch.float32
    assert out.device == torch.device(mojo_h100)
    assert len(calls) == 1
    args = calls[0]
    assert args[:7] == (out._ptr, lhs._ptr, rhs._ptr, batch, m, n, k)
    assert args[7:10] == (m * n, lhs_batch_stride, rhs_batch_stride)
    assert args[10:12] == (int(lhs_transposed), int(rhs_transposed) ^ int(transpose_b))
    assert args[12] == aten_fast._ctx_ptr(lhs._device)


def test_tf32_bmm_host_bridge_strict_fp32_never_launches(mojo_gpu, monkeypatch):
    from torch_mojo_backend.eager_kernels import aten_fast

    calls = []
    _replace_defined_native_calls(
        monkeypatch,
        {("tf32_matmul_ops.mojo", "Tf32BmmF32"): lambda *args: calls.append(args)},
    )

    def fail_allocation(*_args, **_kwargs):
        raise AssertionError("unsupported TF32 BMM metadata allocated output")

    monkeypatch.setattr(aten_fast, "_alloc", fail_allocation)
    lhs = torch.randn(2, 7, 9).to(mojo_gpu)
    rhs = torch.randn(2, 9, 5).to(mojo_gpu)
    old_precision = torch.get_float32_matmul_precision()
    torch.set_float32_matmul_precision("highest")
    try:
        assert aten_fast._try_tf32_bmm(lhs, rhs) is None
    finally:
        torch.set_float32_matmul_precision(old_precision)
    assert calls == []


@pytest.mark.parametrize(
    "invalid_case",
    ["inner_stride", "overlap", "batch", "inner", "rank", "zero", "dtype", "device"],
)
def test_tf32_bmm_host_bridge_rejects_unsupported_inputs(
    mojo_h100, monkeypatch, invalid_case
):
    from torch_mojo_backend.eager_kernels import aten_fast

    calls = []
    _replace_defined_native_calls(
        monkeypatch,
        {("tf32_matmul_ops.mojo", "Tf32BmmF32"): lambda *args: calls.append(args)},
    )

    def fail_allocation(*_args, **_kwargs):
        raise AssertionError("unsupported TF32 BMM metadata allocated output")

    monkeypatch.setattr(aten_fast, "_alloc", fail_allocation)
    lhs = torch.randn(2, 7, 9).to(mojo_h100)
    rhs = torch.randn(2, 9, 5).to(mojo_h100)
    if invalid_case == "inner_stride":
        storage = torch.empty(512).to(mojo_h100)
        lhs = aten_fast._view_of(storage, (2, 7, 9), (200, 20, 2), 1)
    elif invalid_case == "overlap":
        storage = torch.empty(256).to(mojo_h100)
        lhs = aten_fast._view_of(storage, (2, 7, 9), (62, 9, 1), 0)
    elif invalid_case == "batch":
        rhs = torch.randn(3, 9, 5).to(mojo_h100)
    elif invalid_case == "inner":
        rhs = torch.randn(2, 8, 5).to(mojo_h100)
    elif invalid_case == "rank":
        rhs = torch.randn(9, 5).to(mojo_h100)
    elif invalid_case == "zero":
        lhs = torch.empty(0, 7, 9).to(mojo_h100)
        rhs = torch.empty(0, 9, 5).to(mojo_h100)
    elif invalid_case == "dtype":
        rhs = torch.randn(2, 9, 5, dtype=torch.float16).to(mojo_h100)
    else:
        mojo_cpu = f"mojo:{len(get_accelerators()) - 1}"
        rhs = torch.randn(2, 9, 5).to(mojo_cpu)

    old_precision = torch.get_float32_matmul_precision()
    torch.set_float32_matmul_precision("medium")
    try:
        assert aten_fast._try_tf32_bmm(lhs, rhs) is None
    finally:
        torch.set_float32_matmul_precision(old_precision)
    assert calls == []


def _require_real_bf16_gemm_sources():
    """Skip before lazy import while an optional BF16 source is absent."""
    from torch_mojo_backend.eager_kernels import aten_fast

    missing = [
        path.name for path in aten_fast._GEMM16_SOURCE_PATHS if not path.is_file()
    ]
    if missing:
        pytest.skip(f"real BF16 GEMM sources are not installed: {', '.join(missing)}")


def _bf16_dense_matrix_pair(generator, shape, transposed, offset, mojo_h100):
    """Create matching stored-BF16 CPU/Mojo views with a pointer offset."""
    from torch_mojo_backend.eager_kernels import aten_fast

    rows, cols = shape
    storage = torch.randn(offset + rows * cols + 4, generator=generator).to(
        torch.bfloat16
    )
    strides = (1, rows) if transposed else (cols, 1)
    host = torch.as_strided(storage, shape, strides, offset)
    device = aten_fast._view_of(storage.to(mojo_h100), shape, strides, offset)
    return host, device


def _bf16_dense_batched_pair(
    generator, shape, transposed, gap, offset, mojo_h100, dtype=torch.bfloat16
):
    """Create stored 16-bit dense matrices separated by a runtime batch
    stride. ``gap`` > 0 pads consecutive batch items apart, i.e. a
    non-contiguous (non-packed) batch stride rather than ``rows * cols``.
    ``dtype`` defaults to bfloat16; pass ``torch.float16`` for the other
    16-bit route."""
    from torch_mojo_backend.eager_kernels import aten_fast

    batch, rows, cols = shape
    matrix_elements = rows * cols
    batch_stride = matrix_elements + gap
    storage_elements = offset + (batch - 1) * batch_stride + matrix_elements + 4
    storage = torch.randn(storage_elements, generator=generator).to(dtype)
    strides = (batch_stride, 1, rows) if transposed else (batch_stride, cols, 1)
    host = torch.as_strided(storage, shape, strides, offset)
    device = aten_fast._view_of(storage.to(mojo_h100), shape, strides, offset)
    return host, device


def _assert_bf16_fp32_accumulation_close(actual, expected):
    """Compare BF16 outputs after an explicit one-round FP32 oracle."""
    assert actual.dtype == expected.dtype == torch.bfloat16
    torch.testing.assert_close(actual, expected, atol=0.03125, rtol=0.02)


def _assert_gemm16_fp32_accumulation_close(actual, expected, dtype):
    """Compare 16-bit (bf16 or f16) GEMM outputs after a one-round FP32
    oracle. f16 has one more mantissa bit than bf16, so the bound
    ``_assert_bf16_fp32_accumulation_close`` uses stays safe for it too."""
    assert actual.dtype == expected.dtype == dtype
    torch.testing.assert_close(actual, expected, atol=0.03125, rtol=0.02)


@pytest.mark.parametrize(
    ("m", "n", "k", "lhs_transposed", "rhs_transposed"),
    [
        (1088, 128, 192, False, False),
        (1088, 128, 448, False, False),
        (4352, 256, 320, False, False),
        (1088, 256, 192, False, True),
        (192, 384, 256, False, True),
        (128, 256, 256, True, False),
        (576, 2048, 64, True, False),
        (1152, 1024, 64, True, False),
    ],
    ids=[
        "nn",
        "nn_small_five_k_tiles",
        "nn_large_five_k_tiles",
        "nt_three_stages",
        "nt_half_tiles_stage_reuse",
        "tn_small_stage_reuse",
        "tn_small_alignment_regime",
        "tn_large_occupancy_regime",
    ],
)
def test_bf16_real_v3_aligned_dynamic_gemm_routes(
    mojo_h100, monkeypatch, m, n, k, lhs_transposed, rhs_transposed
):
    """Production v3 routes match one-round FP32 references."""
    _require_real_bf16_gemm_sources()
    from torch_mojo_backend.eager_kernels import aten_fast

    generator = torch.Generator().manual_seed(20260719)
    lhs, mojo_lhs = _bf16_dense_matrix_pair(
        generator, (m, k), lhs_transposed, 0, mojo_h100
    )
    rhs, mojo_rhs = _bf16_dense_matrix_pair(
        generator, (k, n), rhs_transposed, 0, mojo_h100
    )
    expected = torch.mm(lhs.float(), rhs.float()).to(torch.bfloat16)

    def fail_later_route(*_args, **_kwargs):
        raise AssertionError("eligible BF16 v3 GEMM reached a later route")

    monkeypatch.setattr(aten_fast, "_try_tf32_gemm", fail_later_route)
    monkeypatch.setattr(aten_fast, "_try_spec_matmul", fail_later_route)
    old_precision = torch.get_float32_matmul_precision()
    torch.set_float32_matmul_precision("highest")
    try:
        actual = torch.mm(mojo_lhs, mojo_rhs)
    finally:
        torch.set_float32_matmul_precision(old_precision)

    _assert_bf16_fp32_accumulation_close(actual.cpu(), expected)


def test_bf16_real_tn_wgrad_tiny_m_huge_n_half_tile_n_regime(mojo_h100, monkeypatch):
    """The lm_head wgrad shape class: tiny-M TN with n % 256 == 128.

    nanoGPT's most expensive GEMM is the lm_head weight gradient -- TN,
    m=768, n=50304 (padded vocab), k=49152 (tokens).  Its defining dispatch
    property is n being a multiple of 128 but NOT of 256: the 256-wide TN
    split-K and persistent v4 rungs both decline it, and the narrow-tile
    192-wide rung declines too because the grid is multi-wave, so the call
    falls through the whole v4 ladder onto the v3 fallback.

    768 x 4224 x 1024 is the smallest shape that reproduces every one of
    those decisions rather than the giant original: n = 4224 = 33 * 128
    (n % 256 == 128, and also n % 192 == 0 like 50304, so the 192-wide rung
    is *evaluated*, not skipped) with (m/128) * (n/192) = 132 tiles > 114
    SMs, so that rung declines for the same multi-wave reason as 50304's
    1572 tiles -- one step smaller (n=2688, 84 tiles) engages the 192-wide
    direct kernel instead, a different route.  Verified on H100 to launch
    the same kernel as the full shape (CUPTI:
    bf16_gemm_v3_tn_ws_m64n128_tma_col_a_s3 at HEAD b99e74e).  With the
    ragged-n persistent TN rung added since, this shape still lands on v3:
    its 128x256 census (6 * ceil(4224/256) = 102 tiles) is single-wave on
    114 SMs, so that rung declines too -- keeping this the fall-through
    regression test it was.  Correctness only; timings live in the perf
    harness and benchmarks.
    """
    _require_real_bf16_gemm_sources()
    from torch_mojo_backend.eager_kernels import aten_fast

    m, n, k = 768, 4224, 1024
    generator = torch.Generator().manual_seed(20260810)
    lhs, mojo_lhs = _bf16_dense_matrix_pair(generator, (m, k), True, 0, mojo_h100)
    rhs, mojo_rhs = _bf16_dense_matrix_pair(generator, (k, n), False, 0, mojo_h100)
    expected = torch.mm(lhs.float(), rhs.float()).to(torch.bfloat16)

    def fail_later_route(*_args, **_kwargs):
        raise AssertionError("eligible BF16 TN GEMM reached TF32 or TensorSpec")

    monkeypatch.setattr(aten_fast, "_try_tf32_gemm", fail_later_route)
    monkeypatch.setattr(aten_fast, "_try_spec_matmul", fail_later_route)
    old_precision = torch.get_float32_matmul_precision()
    torch.set_float32_matmul_precision("highest")
    try:
        actual = torch.mm(mojo_lhs, mojo_rhs)
    finally:
        torch.set_float32_matmul_precision(old_precision)

    _assert_bf16_fp32_accumulation_close(actual.cpu(), expected)


@pytest.mark.parametrize(
    ("m", "n", "k"),
    [
        # Smallest multi-wave engage-side shape of the lm_head class:
        # n % 256 == 128 and 6 * ceil(4992/256) = 120 census tiles > 114
        # SMs, so the ragged-n persistent TN rung takes it (CUPTI on H100:
        # bf16_gemm_tn_v4_persistent_nclip -- the same route 768x50304x49152
        # takes; one census step smaller, n=4224, is the fall-through test
        # above).
        (768, 4992, 1024),
        # n % 256 == 64: before the ragged rung this fell past v3's
        # n % 128 == 0 gate onto the non-TMA wide fallback (a ~6x cliff on
        # H100); 12 * 17 = 204 census tiles, multi-wave.
        (1536, 4160, 1024),
        # m % 256 == 128: the trailing macro row is half empty, exercising
        # the persistent body's ragged-m TMA clamp inside the ragged-n
        # instantiation (3 * 51 = 153 census tiles).
        (384, 12928, 1024),
    ],
)
def test_bf16_real_tn_wgrad_ragged_n_persistent_route(mojo_h100, monkeypatch, m, n, k):
    """Multi-wave TN wgrad with n % 256 != 0: the ragged-n persistent rung.

    The n-clip instantiation of the shared persistent body (the one TT
    already used) serves TN here; its B TMA loads clamp past the n edge and
    the C TMA store clips the partial last 64-column chunk, so correctness
    at the ragged edge is exactly what this test guards.  Correctness only;
    timings live in the perf harness and benchmarks.
    """
    _require_real_bf16_gemm_sources()
    from torch_mojo_backend.eager_kernels import aten_fast

    generator = torch.Generator().manual_seed(20260811)
    lhs, mojo_lhs = _bf16_dense_matrix_pair(generator, (m, k), True, 0, mojo_h100)
    rhs, mojo_rhs = _bf16_dense_matrix_pair(generator, (k, n), False, 0, mojo_h100)
    expected = torch.mm(lhs.float(), rhs.float()).to(torch.bfloat16)

    def fail_later_route(*_args, **_kwargs):
        raise AssertionError("eligible BF16 TN GEMM reached TF32 or TensorSpec")

    monkeypatch.setattr(aten_fast, "_try_tf32_gemm", fail_later_route)
    monkeypatch.setattr(aten_fast, "_try_spec_matmul", fail_later_route)
    old_precision = torch.get_float32_matmul_precision()
    torch.set_float32_matmul_precision("highest")
    try:
        actual = torch.mm(mojo_lhs, mojo_rhs)
    finally:
        torch.set_float32_matmul_precision(old_precision)

    _assert_bf16_fp32_accumulation_close(actual.cpu(), expected)


@pytest.mark.parametrize(
    ("m", "n", "k"),
    [
        # Smallest NN engage-side shape of the lm_head dgrad class:
        # n % 256 == 128 with 60 macro-work clusters = 120 CTAs of
        # persistent fill >= 9/16 of 114 SMs, so the ragged-n NN rung takes
        # it (CUPTI on H100: bf16_gemm_nn_v4_persistent_nclip -- the same
        # route 768x50304x49152 takes).  One census step down (768x2432,
        # 60 CTAs) declines to the v3 small tile instead.
        (768, 4992, 1024),
        # n % 128 == 64: before the ragged rung this fell past the v3
        # small tile's n % 128 == 0 gate onto the non-TMA wide fallback (a
        # ~4.6x cliff on H100), so it must always engage regardless of
        # fill.
        (1536, 4160, 1024),
        # Ragged m AND ragged n at once: m % 64 != 0 rows exercise the
        # persistent body's row-clamp/row-predicated path inside the
        # n-clip instantiation (the v3 small tile cannot serve m % 64 !=
        # 0, so this engages on the m-gate alone).
        (300, 12928, 1024),
    ],
)
def test_bf16_real_nn_dgrad_ragged_n_persistent_route(mojo_h100, monkeypatch, m, n, k):
    """NN dgrad with n % 256 != 0: the ragged-n persistent rung.

    The n-clip instantiation of the shared persistent body (the one TN and
    TT already used) serves NN here; its B TMA loads clamp past the n edge
    and the C TMA store clips the partial last 64-column chunk, so
    correctness at the ragged edge is exactly what this test guards.
    Correctness only; timings live in the perf harness and benchmarks.
    """
    _require_real_bf16_gemm_sources()
    from torch_mojo_backend.eager_kernels import aten_fast

    generator = torch.Generator().manual_seed(20260812)
    lhs, mojo_lhs = _bf16_dense_matrix_pair(generator, (m, k), False, 0, mojo_h100)
    rhs, mojo_rhs = _bf16_dense_matrix_pair(generator, (k, n), False, 0, mojo_h100)
    expected = torch.mm(lhs.float(), rhs.float()).to(torch.bfloat16)

    def fail_later_route(*_args, **_kwargs):
        raise AssertionError("eligible BF16 NN GEMM reached TF32 or TensorSpec")

    monkeypatch.setattr(aten_fast, "_try_tf32_gemm", fail_later_route)
    monkeypatch.setattr(aten_fast, "_try_spec_matmul", fail_later_route)
    old_precision = torch.get_float32_matmul_precision()
    torch.set_float32_matmul_precision("highest")
    try:
        actual = torch.mm(mojo_lhs, mojo_rhs)
    finally:
        torch.set_float32_matmul_precision(old_precision)

    _assert_bf16_fp32_accumulation_close(actual.cpu(), expected)


@pytest.mark.parametrize("operation", ["mm", "addmm"])
@pytest.mark.parametrize("lhs_transposed", [False, True])
@pytest.mark.parametrize("rhs_transposed", [False, True])
def test_bf16_real_gemm_extension_handles_tails_offsets_and_all_layouts(
    mojo_h100, monkeypatch, operation, lhs_transposed, rhs_transposed
):
    """Real BF16 GEMM matches an FP32-accumulate, one-BF16-round oracle."""
    _require_real_bf16_gemm_sources()
    from torch_mojo_backend.eager_kernels import aten_fast

    generator = torch.Generator().manual_seed(20260719)
    m, n, k = 33, 65, 31
    lhs, mojo_lhs = _bf16_dense_matrix_pair(
        generator, (m, k), lhs_transposed, 1, mojo_h100
    )
    rhs, mojo_rhs = _bf16_dense_matrix_pair(
        generator, (k, n), rhs_transposed, 2, mojo_h100
    )
    bias_storage = torch.randn(n + 7, generator=generator).to(torch.bfloat16)
    bias = bias_storage[3 : 3 + n]
    mojo_bias = aten_fast._view_of(
        bias_storage.to(mojo_h100), (n,), (1,), 3, contiguous=True
    )

    product = torch.mm(lhs.float(), rhs.float())
    expected = (product if operation == "mm" else product + bias.float()).to(
        torch.bfloat16
    )

    def fail_later_route(*_args, **_kwargs):
        raise AssertionError("eligible BF16 GEMM reached TF32 or TensorSpec")

    monkeypatch.setattr(aten_fast, "_try_tf32_gemm", fail_later_route)
    monkeypatch.setattr(aten_fast, "_try_spec_matmul", fail_later_route)
    old_precision = torch.get_float32_matmul_precision()
    torch.set_float32_matmul_precision("highest")
    try:
        if operation == "mm":
            actual = torch.mm(mojo_lhs, mojo_rhs)
        else:
            actual = torch.addmm(mojo_bias, mojo_lhs, mojo_rhs)
    finally:
        torch.set_float32_matmul_precision(old_precision)

    _assert_bf16_fp32_accumulation_close(actual.cpu(), expected)


# float32 never reaches gemm16 (bf16/f16 only), so its addmm bias goes
# through the pre-existing TF32/TensorSpec routes this fix does not touch.
# Those routes already decline a 0-d or 1-element bias (a pre-existing gap,
# confirmed present on upstream/main before this change too) -- out of
# scope here, so float32 only exercises the (n,) row vector shape that path
# already supported, as a plain no-regression check.
_ADDMM_BROADCAST_CASES = [
    (dtype, bias_kind)
    for dtype in (torch.bfloat16, torch.float16)
    for bias_kind in ("scalar", "scalar_expandable", "row_vector")
] + [(torch.float32, "row_vector")]
_ADDMM_BROADCAST_IDS = [
    f"{dtype}".rsplit(".", 1)[-1] + f"-{bias_kind}"
    for dtype, bias_kind in _ADDMM_BROADCAST_CASES
]


@pytest.mark.parametrize(
    ("dtype", "bias_kind"), _ADDMM_BROADCAST_CASES, ids=_ADDMM_BROADCAST_IDS
)
@pytest.mark.parametrize("layout", _GEMM16_LAYOUTS)
def test_addmm_bias_broadcast_shapes_stay_correct(
    mojo_h100, monkeypatch, layout, dtype, bias_kind
):
    """Every bias shape aten::addmm broadcasts -- a 0-d scalar, a 1-element
    expandable vector, and the usual (n,) row vector -- stays correct on
    every physical layout, for bf16/f16.  These dtypes additionally reach
    the gemm16 tensor-core fast path rather than falling back to the slower
    bias-fused "accepted" kernel or TensorSpec: see fast_aten_addmm, which
    computes the bias-free mm on the fast path and adds the bias with the
    general broadcasting elementwise add (wider than the fused kernel's
    exact-(n,)-shape gate).
    """
    is_gemm16_dtype = dtype in (torch.bfloat16, torch.float16)
    if is_gemm16_dtype:
        _require_real_bf16_gemm_sources()
    from torch_mojo_backend.eager_kernels import aten_fast

    # 64-aligned in every dim: _gemm16_alignment_favors_split must consider
    # this shape worth splitting into a bias-free mm plus a separate add.
    m, n, k = 128, 128, 128
    a, b = _gemm16_operands(layout, m, n, k, dtype, mojo_h100)
    bias_shape = {"scalar": (), "scalar_expandable": (1,), "row_vector": (n,)}[
        bias_kind
    ]
    host_bias = torch.randn(bias_shape, dtype=dtype)
    mojo_bias = host_bias.to(mojo_h100)

    def fail_later_route(*_args, **_kwargs):
        raise AssertionError("eligible gemm16 addmm reached TF32 or TensorSpec")

    if is_gemm16_dtype:
        # float32 legitimately needs the TF32/TensorSpec routes below gemm16
        # (gemm16 only serves bf16/f16), so only bf16/f16 assert they never
        # reach them.
        monkeypatch.setattr(aten_fast, "_try_tf32_gemm", fail_later_route)
        monkeypatch.setattr(aten_fast, "_try_spec_matmul", fail_later_route)

    actual = torch.addmm(mojo_bias, a, b)
    expected = (a.cpu().float() @ b.cpu().float() + host_bias.float()).to(dtype)
    torch.testing.assert_close(actual.cpu(), expected, atol=5e-2, rtol=5e-2)


@pytest.mark.parametrize("dtype", [torch.bfloat16, torch.float16, torch.float32])
def test_addmm_alpha_beta_scaling_declines_explicitly(mojo_h100, dtype):
    """alpha/beta != 1 isn't implemented by the fast eager path -- it must
    decline loudly (NotImplementedError) rather than silently dropping the
    scaling, on every dtype the gemm16 dispatch fix touches (see the
    ``beta == 1 and alpha == 1`` gate in fast_aten_addmm, unchanged by this
    fix)."""
    a = torch.randn(8, 6, dtype=dtype)
    b = torch.randn(6, 5, dtype=dtype)
    bias = torch.randn(5, dtype=dtype)
    with pytest.raises(NotImplementedError, match="aten::addmm"):
        torch.addmm(
            bias.to(mojo_h100), a.to(mojo_h100), b.to(mojo_h100), alpha=2.0, beta=0.5
        )


@pytest.mark.parametrize("dtype", [torch.bfloat16, torch.float16])
@pytest.mark.parametrize("bias_kind", ["scalar", "scalar_expandable", "row_vector"])
def test_linear_bias_broadcast_shapes_use_gemm16_fast_path(
    mojo_h100, monkeypatch, dtype, bias_kind
):
    """F.linear's bias broadcasts the same way addmm's does; every shape
    stays on the gemm16 fast path -- see _try_gemm16_linear."""
    _require_real_bf16_gemm_sources()
    from torch_mojo_backend.eager_kernels import aten_fast

    # 64-aligned in every dim: _gemm16_alignment_favors_split must consider
    # this shape worth splitting into a bias-free mm plus a separate add.
    m, in_features, out_features = 128, 128, 128
    x = torch.randn(m, in_features, dtype=dtype)
    w = torch.randn(out_features, in_features, dtype=dtype)
    bias_shape = {
        "scalar": (),
        "scalar_expandable": (1,),
        "row_vector": (out_features,),
    }[bias_kind]
    host_bias = torch.randn(bias_shape, dtype=dtype)

    def fail_later_route(*_args, **_kwargs):
        raise AssertionError("eligible gemm16 linear reached TF32 or TensorSpec")

    monkeypatch.setattr(aten_fast, "_try_tf32_linear", fail_later_route)
    monkeypatch.setattr(aten_fast, "_try_spec_matmul", fail_later_route)

    actual = torch.nn.functional.linear(
        x.to(mojo_h100), w.to(mojo_h100), host_bias.to(mojo_h100)
    )
    expected = (x.float() @ w.float().t() + host_bias.float()).to(dtype)
    torch.testing.assert_close(actual.cpu(), expected, atol=5e-2, rtol=5e-2)


@pytest.mark.parametrize("lhs_transposed", [False, True])
@pytest.mark.parametrize("rhs_transposed", [False, True])
@pytest.mark.parametrize("transpose_b", [False, True])
def test_bf16_real_bmm_extension_handles_all_layouts_offsets_and_padded_batches(
    mojo_h100, monkeypatch, lhs_transposed, rhs_transposed, transpose_b
):
    """Real BF16 BMM covers physical layouts and a logical RHS transpose."""
    _require_real_bf16_gemm_sources()
    from torch_mojo_backend.eager_kernels import aten_fast

    generator = torch.Generator().manual_seed(20260719)
    batch, m, n, k = 3, 7, 5, 9
    lhs, mojo_lhs = _bf16_dense_batched_pair(
        generator, (batch, m, k), lhs_transposed, 5, 1, mojo_h100
    )
    rhs_shape = (batch, n, k) if transpose_b else (batch, k, n)
    rhs, mojo_rhs = _bf16_dense_batched_pair(
        generator, rhs_shape, rhs_transposed, 7, 2, mojo_h100
    )
    logical_rhs = rhs.transpose(1, 2) if transpose_b else rhs
    expected = torch.bmm(lhs.float(), logical_rhs.float()).to(torch.bfloat16)

    def fail_later_route(*_args, **_kwargs):
        raise AssertionError("eligible BF16 BMM reached TF32 or TensorSpec")

    monkeypatch.setattr(aten_fast, "_try_tf32_bmm", fail_later_route)
    monkeypatch.setattr(aten_fast, "_try_spec_matmul", fail_later_route)
    if transpose_b:
        actual = aten_fast._fast_aten_bmm_transpose_b(mojo_lhs, mojo_rhs)
    else:
        actual = torch.bmm(mojo_lhs, mojo_rhs)

    _assert_bf16_fp32_accumulation_close(actual.cpu(), expected)


# ---------------------------------------------------------------------------
# Batched TN wgmma route: TN previously had no batched tensor-core kernel at
# all (see enqueue_gemm16_bmm's module comment) and fell all the way back to
# the pre-wgmma accepted-v2 kernel that also serves NN/NT/TT above -- the
# worst-served of the four there. It now loops the mm-level TN wgmma ladder
# (_try_enqueue_gemm16_tn_route) once per batch item instead. The three
# aligned shapes below were profiler-verified (CUPTI on H100) to land on the
# v3 small-tile, v3 large-tile, and v4 split-K TN routes respectively; the
# awkward shape stays on today's accepted-v2 kernel, unchanged. NN/NT/TT are
# included at the same shapes to pin that their dispatch (and thus their
# performance) is untouched.
# ---------------------------------------------------------------------------


@pytest.mark.parametrize("dtype", [torch.bfloat16, torch.float16])
@pytest.mark.parametrize(
    ("m", "n", "k", "batch"),
    [
        (128, 256, 256, 3),  # v3 TN small-tile occupancy regime
        (1152, 1024, 64, 2),  # v3 TN large-tile occupancy regime
        (128, 256, 4096, 2),  # v4 TN split-K (deep-K, underfilled output)
        (357, 789, 333, 5),  # awkward/odd dims: every wgmma route declines
    ],
    ids=["tn_small_tile", "tn_large_tile", "tn_splitk", "awkward_odd_dims"],
)
@pytest.mark.parametrize("layout", ["NN", "NT", "TN", "TT"])
def test_bf16_real_bmm_batched_tn_wgmma_route(
    mojo_h100, monkeypatch, layout, m, n, k, batch, dtype
):
    """The strided-batch TN loop matches every other BMM layout's
    correctness at wgmma-triggering scale, batch > 1, non-contiguous
    (padded, not back-to-back) batch strides, and both 16-bit dtypes."""
    _require_real_bf16_gemm_sources()
    from torch_mojo_backend.eager_kernels import aten_fast

    lhs_transposed = layout[0] == "T"
    rhs_transposed = layout[1] == "T"
    generator = torch.Generator().manual_seed(20260816)
    lhs, mojo_lhs = _bf16_dense_batched_pair(
        generator, (batch, m, k), lhs_transposed, 5, 1, mojo_h100, dtype
    )
    rhs, mojo_rhs = _bf16_dense_batched_pair(
        generator, (batch, k, n), rhs_transposed, 7, 2, mojo_h100, dtype
    )
    expected = torch.bmm(lhs.float(), rhs.float()).to(dtype)

    def fail_later_route(*_args, **_kwargs):
        raise AssertionError("eligible 16-bit BMM reached TF32 or TensorSpec")

    monkeypatch.setattr(aten_fast, "_try_tf32_bmm", fail_later_route)
    monkeypatch.setattr(aten_fast, "_try_spec_matmul", fail_later_route)
    actual = torch.bmm(mojo_lhs, mojo_rhs)

    _assert_gemm16_fp32_accumulation_close(actual.cpu(), expected, dtype)


# ---------------------------------------------------------------------------
# Batched NN wgmma route (gemm16_bmm_v5_kernels.mojo). Unlike TN's per-item
# loop above, NN batches the WHOLE launch in one call: a persistent work list
# spans every batch item's tiles, addressed by rank-3 TMA descriptors that
# carry the batch dimension and clip ragged m/n/k per item, so a small
# per-item grid (attention, conv-shaped BMM) still fills the machine instead
# of underfilling once per loop iteration. Cases below walk
# `try_enqueue_bmm16_nn_batched`'s dispatch ladder: aligned direct-TMA,
# ragged m/n/k, the forced big-tile regime, padded (non-packed) batch
# strides, and the row-repack route (row stride/batch stride not a multiple
# of 8 elements) with its scalar-C epilogue -- plus a broadcast A operand
# (batch stride 0, the conv im2col shape) and an alignment-driven fallback,
# each in their own test below.
# ---------------------------------------------------------------------------


def test_tf32_dense_batched_layout_accepts_broadcast_stride_zero():
    """Batch stride 0 (a broadcast/expanded operand, e.g. conv's shared
    weight matrix or an `expand()`ed bmm argument) classifies as valid with
    the batch stride passed through unchanged, for both physical layouts.
    Genuine overlap (a stride strictly between 0 and matrix_elements) is
    still rejected -- only exact broadcast is a first-class case."""
    from torch_mojo_backend.eager_kernels import aten_fast

    h100 = SimpleNamespace(label="gpu", api="cuda", architecture_name="sm_90a")

    def tensor(shape, strides):
        return SimpleNamespace(
            _shape=tuple(shape),
            _mojo_strides=tuple(strides),
            _dtype=aten_fast.DType.bfloat16,
            _device=h100,
        )

    row_major = tensor((4, 7, 9), (0, 9, 1))
    assert aten_fast._tf32_dense_batched_layout(row_major) == (False, 0)

    transposed = tensor((4, 7, 9), (0, 1, 7))
    assert aten_fast._tf32_dense_batched_layout(transposed) == (True, 0)

    overlapping = tensor((4, 7, 9), (62, 9, 1))  # matrix_elements = 63
    assert aten_fast._tf32_dense_batched_layout(overlapping) is None

    non_broadcast_still_dense = tensor((4, 7, 9), (63, 9, 1))
    assert aten_fast._tf32_dense_batched_layout(non_broadcast_still_dense) == (
        False,
        63,
    )


@pytest.mark.parametrize("dtype", [torch.bfloat16, torch.float16])
@pytest.mark.parametrize(
    ("m", "n", "k", "batch", "gap_a", "gap_b"),
    [
        (64, 64, 64, 3, 0, 0),  # minimal aligned direct-TMA
        (100, 200, 104, 2, 0, 0),  # ragged m, n, k (per-item TMA clip)
        (300, 520, 128, 2, 0, 0),  # forced big-tile regime (m >= 256)
        (70, 96, 64, 2, 64, 128),  # padded (non-packed) batch strides
        (77, 89, 100, 2, 0, 0),  # fully unaligned: repack route + scalar-C
        (64, 71, 64, 2, 0, 0),  # odd n only: B repack + scalar C
        (64, 72, 65, 2, 0, 0),  # odd k only: A repack, TMA-store C
        (357, 789, 333, 8, 0, 0),  # awkward/odd dims (documented 1.09-1.11x)
    ],
    ids=[
        "aligned_min",
        "ragged_mnk",
        "big_tile",
        "padded_batch_strides",
        "unaligned_repack_scalar_c",
        "oddn_repack",
        "oddk_repack",
        "awkward_odd_dims",
    ],
)
def test_bf16_real_bmm_batched_nn_v5_route(
    mojo_h100, monkeypatch, m, n, k, batch, gap_a, gap_b, dtype
):
    """The batched NN route matches every other BMM layout's correctness
    across its whole dispatch ladder (gemm16_bmm_v5_kernels.mojo's
    `_b5_dispatch`): direct TMA, ragged-edge clipping, the big-tile
    instantiation, padded batch strides, and the row-repack path an odd n or
    k forces. Shapes mirror the reconciled harness table (see the PR body
    for the measured ratios vs cuBLAS)."""
    _require_real_bf16_gemm_sources()
    from torch_mojo_backend.eager_kernels import aten_fast

    generator = torch.Generator().manual_seed(20260816)
    lhs, mojo_lhs = _bf16_dense_batched_pair(
        generator, (batch, m, k), False, gap_a, 1, mojo_h100, dtype
    )
    rhs, mojo_rhs = _bf16_dense_batched_pair(
        generator, (batch, k, n), False, gap_b, 2, mojo_h100, dtype
    )
    expected = torch.bmm(lhs.float(), rhs.float()).to(dtype)

    def fail_later_route(*_args, **_kwargs):
        raise AssertionError("eligible 16-bit NN BMM reached TF32 or TensorSpec")

    monkeypatch.setattr(aten_fast, "_try_tf32_bmm", fail_later_route)
    monkeypatch.setattr(aten_fast, "_try_spec_matmul", fail_later_route)
    actual = torch.bmm(mojo_lhs, mojo_rhs)

    _assert_gemm16_fp32_accumulation_close(actual.cpu(), expected, dtype)


@pytest.mark.parametrize("dtype", [torch.bfloat16, torch.float16])
@pytest.mark.parametrize(
    ("m", "n", "k", "batch"),
    [
        (64, 128, 64, 4),  # broadcast A, direct TMA route
        (64, 89, 100, 3),  # broadcast A through the repack route (odd n/k)
        (64, 3136, 576, 2),  # conv-im2col-shaped: wide n
    ],
    ids=["bcastA_aligned", "bcastA_repack", "bcastA_conv_shaped"],
)
def test_bf16_real_bmm_batched_nn_v5_broadcast_a(
    mojo_h100, monkeypatch, m, n, k, batch, dtype
):
    """A broadcast (`expand()`ed, batch stride 0) A operand -- the shape
    conv's im2col GEMM produces, one shared weight matrix reused for every
    sample -- is a first-class input to the v5 route, not a decline: this
    used to be exactly the routing gap `_tf32_dense_batched_layout` produced
    for `fast_aten_bmm` (see that function's docstring; conv itself reaches
    the bridge through its own call site, not this classifier)."""
    _require_real_bf16_gemm_sources()
    from torch_mojo_backend.eager_kernels import aten_fast

    generator = torch.Generator().manual_seed(20260816)
    a2d = torch.randn(m, k, generator=generator).to(dtype)
    lhs = a2d.unsqueeze(0).expand(batch, m, k)
    mojo_a2d = a2d.to(mojo_h100)
    mojo_lhs = aten_fast._view_of(mojo_a2d, (batch, m, k), (0, k, 1), 0)
    rhs, mojo_rhs = _bf16_dense_batched_pair(
        generator, (batch, k, n), False, 0, 2, mojo_h100, dtype
    )
    expected = torch.bmm(lhs.float(), rhs.float()).to(dtype)

    def fail_later_route(*_args, **_kwargs):
        raise AssertionError(
            "eligible 16-bit broadcast-A NN BMM reached TF32 or TensorSpec"
        )

    monkeypatch.setattr(aten_fast, "_try_tf32_bmm", fail_later_route)
    monkeypatch.setattr(aten_fast, "_try_spec_matmul", fail_later_route)
    actual = torch.bmm(mojo_lhs, mojo_rhs)

    _assert_gemm16_fp32_accumulation_close(actual.cpu(), expected, dtype)


@pytest.mark.parametrize("dtype", [torch.bfloat16, torch.float16])
def test_bf16_real_bmm_batched_nn_v5_declines_on_misaligned_base_pointer(
    mojo_h100, monkeypatch, dtype
):
    """The v5 route's own runtime gate (`Int(a) % 16 != 0` etc. in
    `try_enqueue_bmm16_nn_batched`) checks the ACTUAL base pointer, not just
    the strides the host already validated -- a 1-element offset keeps every
    stride a multiple of 8 (the host-side repack gate would see this as
    TMA-eligible) while still breaking 16-byte pointer alignment. This must
    fall back to a still-correct route (the pre-wgmma accepted BMM kernel),
    not corrupt output or raise."""
    _require_real_bf16_gemm_sources()
    from torch_mojo_backend.eager_kernels import aten_fast

    generator = torch.Generator().manual_seed(20260816)
    batch, m, n, k = 2, 64, 128, 64
    lhs, mojo_lhs = _bf16_dense_batched_pair(
        generator, (batch, m, k), False, 0, 1, mojo_h100, dtype
    )
    rhs, mojo_rhs = _bf16_dense_batched_pair(
        generator, (batch, k, n), False, 0, 2, mojo_h100, dtype
    )
    expected = torch.bmm(lhs.float(), rhs.float()).to(dtype)

    def fail_later_route(*_args, **_kwargs):
        raise AssertionError("eligible 16-bit NN BMM reached TF32 or TensorSpec")

    monkeypatch.setattr(aten_fast, "_try_tf32_bmm", fail_later_route)
    monkeypatch.setattr(aten_fast, "_try_spec_matmul", fail_later_route)
    actual = torch.bmm(mojo_lhs, mojo_rhs)

    _assert_gemm16_fp32_accumulation_close(actual.cpu(), expected, dtype)


def test_bf16_real_linear_forward_backward_uses_three_gemm_routes(
    mojo_h100, monkeypatch
):
    """Linear forward, input-grad, and weight-grad all use real BF16 GEMM."""
    _require_real_bf16_gemm_sources()
    from torch_mojo_backend.eager_kernels import aten_fast

    generator = torch.Generator().manual_seed(20260719)
    input = torch.randn(2, 5, 73, generator=generator).to(torch.bfloat16)
    weight = torch.randn(67, 73, generator=generator).to(torch.bfloat16)
    bias = torch.randn(67, generator=generator).to(torch.bfloat16)
    grad_output = torch.randn(2, 5, 67, generator=generator).to(torch.bfloat16)

    input_matrix = input.reshape(-1, input.shape[-1])
    grad_matrix = grad_output.reshape(-1, grad_output.shape[-1])
    expected_output = (
        (input_matrix.float() @ weight.float().t() + bias.float())
        .to(torch.bfloat16)
        .reshape(2, 5, 67)
    )
    expected_input_grad = (
        (grad_matrix.float() @ weight.float()).to(torch.bfloat16).reshape_as(input)
    )
    expected_weight_grad = (grad_matrix.float().t() @ input_matrix.float()).to(
        torch.bfloat16
    )
    expected_bias_grad = grad_matrix.float().sum(dim=0).to(torch.bfloat16)

    mojo_input = input.to(mojo_h100).requires_grad_()
    mojo_weight = weight.to(mojo_h100).requires_grad_()
    mojo_bias = bias.to(mojo_h100).requires_grad_()
    mojo_grad_output = grad_output.to(mojo_h100)
    gemm_calls = []
    original_try_gemm16_mm = aten_fast._try_gemm16_mm

    def record_gemm16_mm(*args, **kwargs):
        result = original_try_gemm16_mm(*args, **kwargs)
        assert result is not None
        gemm_calls.append((tuple(args[0]._shape), tuple(args[1]._shape), kwargs))
        return result

    def fail_later_route(*_args, **_kwargs):
        raise AssertionError("eligible BF16 linear GEMM reached TF32 or TensorSpec")

    monkeypatch.setattr(aten_fast, "_try_gemm16_mm", record_gemm16_mm)
    monkeypatch.setattr(aten_fast, "_try_tf32_gemm", fail_later_route)
    monkeypatch.setattr(aten_fast, "_try_spec_matmul", fail_later_route)
    actual_output = torch.nn.functional.linear(mojo_input, mojo_weight, mojo_bias)
    actual_output.backward(mojo_grad_output)

    assert gemm_calls == [
        ((10, 73), (67, 73), {"transpose_b": True, "output_shape": (2, 5, 67)}),
        ((10, 67), (67, 73), {}),
        ((67, 10), (10, 73), {}),
    ]
    _assert_bf16_fp32_accumulation_close(actual_output.cpu(), expected_output)
    for actual, expected in (
        (mojo_input.grad, expected_input_grad),
        (mojo_weight.grad, expected_weight_grad),
        (mojo_bias.grad, expected_bias_grad),
    ):
        assert actual is not None
        _assert_bf16_fp32_accumulation_close(actual.cpu(), expected)


# ---------------------------------------------------------------------------
# Strict-fp32 TN GEMM (NVIDIA sm_90): a transposed-dense A — strides (1, m),
# linear-backward's dW = dY^T @ X — is read in place by the dedicated FFMA
# kernels (matmul_ops/tn_f32_gemm_kernels.mojo) instead of being materialized
# into a contiguous scratch first.  Python routing is unchanged: fp32 mm at
# "highest" precision reaches MatmulSpec and the Mojo side decides, so these
# tests only pin correctness per dispatch regime plus the regimes the route
# declines back to the copy path.
# ---------------------------------------------------------------------------


def _f32_transposed_dense_pair(
    generator: torch.Generator, m: int, k: int, offset: int, device: str
) -> tuple[torch.Tensor, torch.Tensor]:
    """Matching CPU/Mojo (m, k) views stored transposed-dense — strides
    (1, m) over a row-major (k, m) buffer — with a storage offset (offset 1
    makes the base pointer 16B-misaligned, forcing the 4-byte staging
    variants)."""
    from torch_mojo_backend.eager_kernels import aten_fast

    storage = torch.randn(offset + m * k + 4, generator=generator)
    host = torch.as_strided(storage, (m, k), (1, m), offset)
    dev = aten_fast._view_of(storage.to(device), (m, k), (1, m), offset)
    return host, dev


def _assert_f32_seq_k_close(
    actual: torch.Tensor, expected: torch.Tensor, k: int
) -> None:
    """Compare an fp32 GEMM against an fp64 oracle with the sequential-over-k
    fp32 accumulation tolerance (random-walk rounding grows ~sqrt(k))."""
    assert actual.dtype == torch.float32
    torch.testing.assert_close(
        actual.double(), expected, atol=5e-5 * math.sqrt(k), rtol=1e-5
    )


@pytest.mark.parametrize(
    ("m", "n", "k", "offset"),
    [
        (768, 768, 2048, 0),
        (768, 767, 1024, 0),
        (357, 789, 1571, 0),
        (256, 256, 512, 1),
        (128, 128, 40000, 0),
        (64, 48, 129, 0),
        (65, 63, 640, 0),
        (2, 4096, 64, 0),
        (127, 255, 63, 2),
    ],
    ids=[
        "aligned_fat_tile_split_core",
        "n_misaligned_quadrant_core",
        "awkward_vec1_quadrant_core",
        "misaligned_base_offset_view",
        "deepk_splitk_wide_reduce",
        "small_t64_k_tail",
        "small_t64_ragged_splitk",
        "tiny_m_t64",
        "ragged_offset_t64",
    ],
)
def test_f32_tn_transposed_a_route_matches_fp64_oracle(
    mojo_h100: str, monkeypatch: pytest.MonkeyPatch, m: int, n: int, k: int, offset: int
) -> None:
    """Every dispatch regime of the sm_90 fp32 TN route against fp64."""
    generator = torch.Generator().manual_seed(20260808)
    host_a, mojo_a = _f32_transposed_dense_pair(generator, m, k, offset, mojo_h100)
    host_b = torch.randn(k, n, generator=generator)
    mojo_b = host_b.to(mojo_h100)
    expected = torch.mm(host_a.double(), host_b.double())

    target = ("matmul_ops.mojo", "MatmulSpec")
    calls = _spy_defined_native_calls(monkeypatch, {target})
    old_precision = torch.get_float32_matmul_precision()
    torch.set_float32_matmul_precision("highest")
    try:
        actual = torch.mm(mojo_a, mojo_b)
    finally:
        torch.set_float32_matmul_precision(old_precision)

    # fp32 mm reached MatmulSpec with no Python-side routing change; the
    # Mojo dispatch decides between the in-place TN kernels and the copy.
    assert len(calls[target]) == 1
    _assert_f32_seq_k_close(actual.cpu(), expected, k)


def test_f32_tn_route_declined_regimes_stay_correct(mojo_h100: str) -> None:
    """Regimes the TN route declines (m == 1, bias, TT, bmm) keep the
    scratch-copy path and stay correct."""
    generator = torch.Generator().manual_seed(20260808)
    old_precision = torch.get_float32_matmul_precision()
    torch.set_float32_matmul_precision("highest")
    try:
        # m == 1 stays on the GEMV-after-copy path.
        host_a, mojo_a = _f32_transposed_dense_pair(generator, 1, 512, 0, mojo_h100)
        host_b = torch.randn(512, 384, generator=generator)
        _assert_f32_seq_k_close(
            torch.mm(mojo_a, host_b.to(mojo_h100)).cpu(),
            torch.mm(host_a.double(), host_b.double()),
            512,
        )

        # A bias (addmm -> MatmulBiasSpec) falls back to the copy path.
        host_a, mojo_a = _f32_transposed_dense_pair(generator, 96, 512, 0, mojo_h100)
        host_b = torch.randn(512, 128, generator=generator)
        bias = torch.randn(128, generator=generator)
        _assert_f32_seq_k_close(
            torch.addmm(bias.to(mojo_h100), mojo_a, host_b.to(mojo_h100)).cpu(),
            torch.addmm(bias.double(), host_a.double(), host_b.double()),
            512,
        )

        # TT (B strided too) falls back to the both-scratch path.
        host_bt = torch.randn(128, 384, generator=generator)
        host_a, mojo_a = _f32_transposed_dense_pair(generator, 96, 384, 0, mojo_h100)
        _assert_f32_seq_k_close(
            torch.mm(mojo_a, host_bt.to(mojo_h100).t()).cpu(),
            torch.mm(host_a.double(), host_bt.double().t()),
            384,
        )

        # Batched (bmm) falls back: the route only claims batch == 1.
        host_ab = torch.randn(3, 128, 64, generator=generator)
        host_bb = torch.randn(3, 128, 96, generator=generator)
        _assert_f32_seq_k_close(
            torch.bmm(
                host_ab.to(mojo_h100).transpose(1, 2), host_bb.to(mojo_h100)
            ).cpu(),
            torch.bmm(host_ab.double().transpose(1, 2), host_bb.double()),
            128,
        )
    finally:
        torch.set_float32_matmul_precision(old_precision)


def test_f32_tn_route_linear_weight_gradient(mojo_h100: str) -> None:
    """nn.Linear backward produces the dW = dY^T @ X layout the route claims;
    the full autograd round trip must stay correct."""
    generator = torch.Generator().manual_seed(20260808)
    host_input = torch.randn(512, 384, generator=generator)
    host_weight = torch.randn(768, 384, generator=generator)
    host_grad = torch.randn(512, 768, generator=generator)

    reference = host_input.clone().requires_grad_()
    ref_weight = host_weight.clone().requires_grad_()
    torch.nn.functional.linear(reference, ref_weight).backward(host_grad)

    mojo_input = host_input.to(mojo_h100).requires_grad_()
    mojo_weight = host_weight.to(mojo_h100).requires_grad_()
    old_precision = torch.get_float32_matmul_precision()
    torch.set_float32_matmul_precision("highest")
    try:
        output = torch.nn.functional.linear(mojo_input, mojo_weight)
        output.backward(host_grad.to(mojo_h100))
    finally:
        torch.set_float32_matmul_precision(old_precision)

    assert mojo_weight.grad is not None and mojo_input.grad is not None
    assert ref_weight.grad is not None and reference.grad is not None
    torch.testing.assert_close(
        mojo_weight.grad.cpu(), ref_weight.grad, atol=5e-3, rtol=5e-4
    )
    torch.testing.assert_close(
        mojo_input.grad.cpu(), reference.grad, atol=5e-3, rtol=5e-4
    )


def _tf32_dense_matrix_pair(generator, shape, transposed, offset, mojo_h100):
    """Create matching CPU/Mojo dense views with a nonzero storage offset."""
    from torch_mojo_backend.eager_kernels import aten_fast

    rows, cols = shape
    storage = torch.randn(offset + rows * cols + 4, generator=generator)
    strides = (1, rows) if transposed else (cols, 1)
    host = torch.as_strided(storage, shape, strides, offset)
    device = aten_fast._view_of(storage.to(mojo_h100), shape, strides, offset)
    return host, device


def _tf32_dense_batched_pair(generator, shape, transposed, gap, offset, mojo_h100):
    """Create dense per-matrix views separated by padding on both devices."""
    from torch_mojo_backend.eager_kernels import aten_fast

    batch, rows, cols = shape
    matrix_elements = rows * cols
    batch_stride = matrix_elements + gap
    storage_elements = offset + (batch - 1) * batch_stride + matrix_elements + 4
    storage = torch.randn(storage_elements, generator=generator)
    strides = (batch_stride, 1, rows) if transposed else (batch_stride, cols, 1)
    host = torch.as_strided(storage, shape, strides, offset)
    device = aten_fast._view_of(storage.to(mojo_h100), shape, strides, offset)
    return host, device


@pytest.mark.parametrize(
    ("operation", "lhs_transposed", "rhs_transposed"),
    [
        ("mm", False, False),
        ("mm", True, True),
        ("addmm", False, True),
        ("addmm", True, False),
    ],
)
def test_tf32_real_gemm_extension_handles_tails_offsets_and_layouts(
    mojo_h100, monkeypatch, operation, lhs_transposed, rhs_transposed
):
    """The lazily loaded extension, not SIMT fallback, computes real GEMMs."""
    from torch_mojo_backend.eager_kernels import aten_fast

    generator = torch.Generator().manual_seed(20260719)
    m, n, k = 33, 65, 31
    lhs, mojo_lhs = _tf32_dense_matrix_pair(
        generator, (m, k), lhs_transposed, 1, mojo_h100
    )
    rhs, mojo_rhs = _tf32_dense_matrix_pair(
        generator, (k, n), rhs_transposed, 2, mojo_h100
    )
    bias_storage = torch.randn(n + 7, generator=generator)
    bias = bias_storage[3 : 3 + n]
    mojo_bias = aten_fast._view_of(
        bias_storage.to(mojo_h100), (n,), (1,), 3, contiguous=True
    )

    def fail_spec(*_args, **_kwargs):
        raise AssertionError("eligible TF32 GEMM reached the SIMT fallback")

    monkeypatch.setattr(aten_fast, "_try_spec_matmul", fail_spec)
    old_precision = torch.get_float32_matmul_precision()
    torch.set_float32_matmul_precision("high")
    try:
        if operation == "mm":
            actual = torch.mm(mojo_lhs, mojo_rhs)
            expected = torch.mm(lhs, rhs)
        else:
            actual = torch.addmm(mojo_bias, mojo_lhs, mojo_rhs)
            expected = torch.addmm(bias, lhs, rhs)
    finally:
        torch.set_float32_matmul_precision(old_precision)

    torch.testing.assert_close(actual.cpu(), expected, atol=5e-2, rtol=5e-2)


@pytest.mark.parametrize("lhs_transposed", [False, True])
@pytest.mark.parametrize("rhs_transposed", [False, True])
@pytest.mark.parametrize("transpose_b", [False, True])
def test_tf32_real_bmm_extension_handles_layouts_offsets_and_padded_batches(
    mojo_h100, monkeypatch, lhs_transposed, rhs_transposed, transpose_b
):
    """Real BMM covers every physical layout and the logical RHS transpose."""
    from torch_mojo_backend.eager_kernels import aten_fast

    generator = torch.Generator().manual_seed(20260719)
    batch, m, n, k = 3, 7, 5, 9
    lhs, mojo_lhs = _tf32_dense_batched_pair(
        generator, (batch, m, k), lhs_transposed, 5, 1, mojo_h100
    )
    rhs_shape = (batch, n, k) if transpose_b else (batch, k, n)
    rhs, mojo_rhs = _tf32_dense_batched_pair(
        generator, rhs_shape, rhs_transposed, 7, 2, mojo_h100
    )

    def fail_spec(*_args, **_kwargs):
        raise AssertionError("eligible TF32 BMM reached the SIMT fallback")

    monkeypatch.setattr(aten_fast, "_try_spec_matmul", fail_spec)
    old_precision = torch.get_float32_matmul_precision()
    torch.set_float32_matmul_precision("high")
    try:
        if transpose_b:
            actual = aten_fast._fast_aten_bmm_transpose_b(mojo_lhs, mojo_rhs)
            expected = torch.bmm(lhs, rhs.transpose(1, 2))
        else:
            actual = torch.bmm(mojo_lhs, mojo_rhs)
            expected = torch.bmm(lhs, rhs)
    finally:
        torch.set_float32_matmul_precision(old_precision)

    torch.testing.assert_close(actual.cpu(), expected, atol=5e-2, rtol=5e-2)


def test_tf32_real_linear_forward_backward_uses_gemm_extension(mojo_h100, monkeypatch):
    """Linear forward, input-grad, and weight-grad all use the real TF32 GEMM."""
    from torch_mojo_backend.eager_kernels import aten_fast

    generator = torch.Generator().manual_seed(20260719)
    input = torch.randn(2, 5, 73, generator=generator)
    weight = torch.randn(67, 73, generator=generator)
    bias = torch.randn(67, generator=generator)
    grad_output = torch.randn(2, 5, 67, generator=generator)

    reference_input = input.clone().requires_grad_()
    reference_weight = weight.clone().requires_grad_()
    reference_bias = bias.clone().requires_grad_()
    reference_output = torch.nn.functional.linear(
        reference_input, reference_weight, reference_bias
    )
    reference_output.backward(grad_output)

    mojo_input = input.to(mojo_h100).requires_grad_()
    mojo_weight = weight.to(mojo_h100).requires_grad_()
    mojo_bias = bias.to(mojo_h100).requires_grad_()
    mojo_grad_output = grad_output.to(mojo_h100)
    gemm_calls = []
    original_try_tf32_gemm = aten_fast._try_tf32_gemm

    def record_tf32_gemm(*args, **kwargs):
        result = original_try_tf32_gemm(*args, **kwargs)
        assert result is not None
        gemm_calls.append((tuple(args[0]._shape), tuple(args[1]._shape), kwargs))
        return result

    def fail_spec(*_args, **_kwargs):
        raise AssertionError("eligible TF32 linear GEMM reached the SIMT fallback")

    monkeypatch.setattr(aten_fast, "_try_tf32_gemm", record_tf32_gemm)
    monkeypatch.setattr(aten_fast, "_try_spec_matmul", fail_spec)
    old_precision = torch.get_float32_matmul_precision()
    torch.set_float32_matmul_precision("high")
    try:
        actual_output = torch.nn.functional.linear(mojo_input, mojo_weight, mojo_bias)
        actual_output.backward(mojo_grad_output)
    finally:
        torch.set_float32_matmul_precision(old_precision)

    assert gemm_calls == [
        ((10, 73), (67, 73), {"transpose_b": True, "output_shape": (2, 5, 67)}),
        ((10, 67), (67, 73), {}),
        ((67, 10), (10, 73), {}),
    ]
    torch.testing.assert_close(
        actual_output.cpu(), reference_output, atol=5e-2, rtol=5e-2
    )
    for actual, expected in (
        (mojo_input.grad, reference_input.grad),
        (mojo_weight.grad, reference_weight.grad),
        (mojo_bias.grad, reference_bias.grad),
    ):
        assert actual is not None
        torch.testing.assert_close(actual.cpu(), expected, atol=5e-2, rtol=5e-2)


@pytest.mark.parametrize("dtype", [torch.float32, torch.float16])
def test_fast_linear_out_features_one(mojo_device, dtype):
    # out_features == 1 -> transposed-B GEMM with n == 1, plus bias.
    x = torch.randn(37, 129).to(dtype)
    w = torch.randn(1, 129).to(dtype)
    b = torch.randn(1).to(dtype)
    dev = torch.nn.functional.linear(
        x.to(mojo_device), w.to(mojo_device), b.to(mojo_device)
    ).cpu()
    ref = (x.float() @ w.float().t() + b.float()).to(dtype)
    torch.testing.assert_close(dev, ref, atol=5e-2, rtol=5e-2)


@pytest.mark.parametrize(
    "in_c,out_c,k,stride,padding,dilation,groups",
    [
        (3, 64, 7, 2, 3, 1, 1),
        (64, 64, 3, 1, 1, 1, 1),
        (64, 128, 1, 2, 0, 1, 1),
        (8, 12, 3, 1, 1, 2, 1),
        (8, 12, 3, 1, 1, 1, 2),
    ],
)
def test_fast_conv2d(mojo_gpu, in_c, out_c, k, stride, padding, dilation, groups):
    x = torch.randn(1, in_c, 32, 32)
    w = torch.randn(out_c, in_c // groups, k, k)
    b = torch.randn(out_c)
    dev = torch.nn.functional.conv2d(
        x.to(mojo_gpu),
        w.to(mojo_gpu),
        b.to(mojo_gpu),
        stride=stride,
        padding=padding,
        dilation=dilation,
        groups=groups,
    ).cpu()
    ref = torch.nn.functional.conv2d(
        x, w, b, stride=stride, padding=padding, dilation=dilation, groups=groups
    )
    torch.testing.assert_close(dev, ref, atol=5e-2, rtol=5e-2)


def test_fast_conv2d_batched_falls_back_correctly(mojo_gpu):
    x = torch.randn(3, 8, 16, 16)
    w = torch.randn(12, 8, 3, 3)
    dev = torch.nn.functional.conv2d(x.to(mojo_gpu), w.to(mojo_gpu), padding=1).cpu()
    ref = torch.nn.functional.conv2d(x, w, padding=1)
    torch.testing.assert_close(dev, ref, atol=5e-2, rtol=5e-2)


def test_fast_conv2d_refuses_autograd_in_the_forward(mojo_gpu):
    """A grad-requiring conv must fail at the FORWARD.

    `aten::convolution_backward` is not implemented here, and a raise from
    inside the autograd engine aborts the process on this backend (exit 134,
    no traceback) instead of propagating, so the refusal cannot wait for
    ConvolutionBackward0. A conv weight normally requires grad, so this refuses
    conv training outright -- the alternative was a dead process -- while
    inference under `torch.no_grad()` is untouched.
    """
    host_input, device_input = _preflight_input((2, 3, 8, 8), mojo_gpu)
    host_weight, device_weight = _preflight_input((4, 3, 3, 3), mojo_gpu, seed=11)
    host_bias, device_bias = _preflight_input((4,), mojo_gpu, seed=13)

    expected = torch.nn.functional.conv2d(host_input, host_weight, host_bias, padding=1)
    torch.testing.assert_close(
        torch.nn.functional.conv2d(
            device_input, device_weight, device_bias, padding=1
        ).cpu(),
        expected,
        atol=5e-2,
        rtol=5e-2,
    )

    # The weight-only case is the one every training model actually hits.
    for which in range(3):
        operands = [device_input.detach(), device_weight.detach(), device_bias.detach()]
        operands[which] = operands[which].requires_grad_()
        with pytest.raises(NotImplementedError, match="convolution_backward"):
            torch.nn.functional.conv2d(operands[0], operands[1], operands[2], padding=1)

    with torch.no_grad():
        torch.testing.assert_close(
            torch.nn.functional.conv2d(
                device_input.detach().requires_grad_(),
                device_weight,
                device_bias,
                padding=1,
            ).cpu(),
            expected,
            atol=5e-2,
            rtol=5e-2,
        )


@pytest.mark.parametrize("is_causal", [True, False])
# kv_len <= 32 exercises the library softmax's warp kernel, kv_len=64 the
# online/block kernel.
@pytest.mark.parametrize("kv_len", [6, 10, 64])
@pytest.mark.parametrize("dtype", [torch.float32, torch.float16])
def test_fast_sdpa(mojo_gpu, is_causal, kv_len, dtype):
    q = torch.randn(1, 12, 6, 64, dtype=dtype)
    k = torch.randn(1, 12, kv_len, 64, dtype=dtype)
    v = torch.randn(1, 12, kv_len, 64, dtype=dtype)
    dev = torch.nn.functional.scaled_dot_product_attention(
        q.to(mojo_gpu), k.to(mojo_gpu), v.to(mojo_gpu), is_causal=is_causal
    ).cpu()
    ref = torch.nn.functional.scaled_dot_product_attention(q, k, v, is_causal=is_causal)
    torch.testing.assert_close(dev, ref, atol=1e-2, rtol=1e-2)


_FA4_DTYPE_SUFFIX = {torch.bfloat16: "bf16", torch.float16: "f16"}


@pytest.mark.parametrize("head_dim", [64, 128], ids=["d64", "d128"])
@pytest.mark.parametrize("dtype", [torch.bfloat16, torch.float16], ids=["bf16", "f16"])
def test_fa4_causal_gapped_qkv_forward_backward(
    mojo_h100, monkeypatch, dtype, head_dim
):
    """The cached H100 path stays dynamic for nanoGPT's gapped QKV views.

    bf16 and f16 share the same kernel family (the f16 RS wgmma emitter is a
    vendored gap-filler for the stdlib's bf16-only register-operand overload,
    see ``fa4_wgmma_f16.mojo``), so both dtypes are exercised here. head_dim
    64 (GPT-2-class, BM=64, 1 MMA warpgroup, 2 CTAs/SM -- phase 2b's
    causal-tile-quantization fix, see ``fa4_fwd_common.mojo``) and 128
    (Llama-class, BM=128, 2 MMA warpgroups, unchanged) are two distinct
    comptime tile configs (see ``fa4_fwd_common.mojo``/``fa4_bwd_common.mojo``),
    not one kernel with a runtime dimension -- both get their own cached
    launch here.
    """
    from torch_mojo_backend.eager_flash_attention import load_fa4_ops
    from torch_mojo_backend.mojo_device.mojo_device_aten_ops import EAGER_CALL_COUNTERS

    module = load_fa4_ops()
    calls = {"forward": 0, "backward": 0}
    suffix = _FA4_DTYPE_SUFFIX[dtype]
    forward_name = f"flash_attention_fwd_{suffix}_d{head_dim}_causal_strided_qkv"
    backward_name = f"flash_attention_bwd_{suffix}_d{head_dim}_causal_strided_qkv"
    original_forward = getattr(module, forward_name)
    original_backward = getattr(module, backward_name)

    def forward_spy(*args):
        calls["forward"] += 1
        return original_forward(*args)

    def backward_spy(*args):
        calls["backward"] += 1
        return original_backward(*args)

    monkeypatch.setattr(module, forward_name, forward_spy)
    monkeypatch.setattr(module, backward_name, backward_spy)

    for seed, batch, seqlen, heads in ((20260718, 1, 128, 4), (20260719, 2, 256, 8)):
        width = heads * head_dim
        generator = torch.Generator().manual_seed(seed)
        fused_host = (
            torch.randn(
                batch, seqlen, 3 * width, generator=generator, dtype=torch.float32
            )
            * 0.25
        ).to(dtype)
        grad_output = torch.randn(
            batch, heads, seqlen, head_dim, generator=generator, dtype=torch.float32
        ).to(dtype)

        def qkv_views(fused):
            return tuple(
                part.view(batch, seqlen, heads, head_dim).transpose(1, 2)
                for part in fused.split(width, dim=2)
            )

        reference_inputs = [
            tensor.float().detach().requires_grad_() for tensor in qkv_views(fused_host)
        ]
        reference_output = torch.nn.functional.scaled_dot_product_attention(
            *reference_inputs, dropout_p=0.0, is_causal=True
        )
        reference_output.backward(grad_output.float())

        fused_mojo = fused_host.to(mojo_h100)
        mojo_inputs = list(qkv_views(fused_mojo))
        for tensor in mojo_inputs:
            tensor.requires_grad_()
            assert not tensor._is_contiguous

        # The first shape populates the compiled-function cache; the second
        # changes B, S, and H while exercising all five cached launches.
        backward_counter = EAGER_CALL_COUNTERS[
            "aten::_scaled_dot_product_flash_attention_backward"
        ]
        calls_before = backward_counter.call_count
        actual_output = torch.nn.functional.scaled_dot_product_attention(
            *mojo_inputs, dropout_p=0.0, is_causal=True
        )
        assert type(actual_output.grad_fn).__name__ == (
            "ScaledDotProductFlashAttentionBackward0"
        )
        assert backward_counter.call_count == calls_before
        assert actual_output.dtype == dtype
        assert not actual_output._is_contiguous
        assert actual_output.transpose(1, 2)._is_contiguous
        actual_output.backward(grad_output.to(mojo_h100))
        assert backward_counter.call_count == calls_before + 1

        torch.testing.assert_close(
            actual_output.cpu().float(), reference_output, atol=2e-2, rtol=2e-2
        )
        for actual, reference in zip(mojo_inputs, reference_inputs, strict=True):
            assert actual.grad is not None
            torch.testing.assert_close(
                actual.grad.cpu().float(), reference.grad, atol=5e-2, rtol=5e-2
            )

    assert calls == {"forward": 2, "backward": 2}


@pytest.mark.parametrize("head_dim", [64, 128], ids=["d64", "d128"])
@pytest.mark.parametrize("dtype", [torch.bfloat16, torch.float16], ids=["bf16", "f16"])
def test_fa4_direct_flash_aten_returns_real_logsumexp(mojo_h100, dtype, head_dim):
    """The low-level flash op must not substitute zero backward metadata."""
    generator = torch.Generator().manual_seed(20260721)
    shape = (1, 4, 128, head_dim)
    query, key, value = (
        (torch.randn(shape, generator=generator) * 0.25).to(dtype) for _ in range(3)
    )
    mojo_inputs = [
        tensor.to(mojo_h100).requires_grad_() for tensor in (query, key, value)
    ]
    result = torch.ops.aten._scaled_dot_product_flash_attention.default(
        *mojo_inputs, 0.0, True, False
    )

    scores = query.float() @ key.float().transpose(-2, -1)
    scores *= 1.0 / math.sqrt(shape[-1])
    causal = torch.ones(shape[-2], shape[-2], dtype=torch.bool).tril()
    expected_lse = scores.masked_fill(~causal, float("-inf")).logsumexp(-1)
    reference_inputs = [
        tensor.float().detach().requires_grad_() for tensor in (query, key, value)
    ]
    expected_output = torch.nn.functional.scaled_dot_product_attention(
        *reference_inputs, dropout_p=0.0, is_causal=True
    )

    output, logsumexp, cum_q, cum_k, max_q, max_k, rng, offset, debug = result
    torch.testing.assert_close(
        output.cpu().float(), expected_output, atol=2e-2, rtol=2e-2
    )
    torch.testing.assert_close(logsumexp.cpu(), expected_lse, atol=2e-2, rtol=2e-2)
    assert cum_q is None and cum_k is None
    assert (max_q, max_k) == (128, 128)
    assert rng.dtype == torch.uint64 and tuple(rng.shape) == (2,)
    assert offset.dtype == torch.uint64 and tuple(offset.shape) == ()
    assert debug.dtype == dtype and debug.numel() == 0

    grad_output = torch.randn(shape, generator=generator).to(dtype)
    expected_output.backward(grad_output.float())
    output.backward(grad_output.to(mojo_h100))
    for actual, expected in zip(mojo_inputs, reference_inputs, strict=True):
        assert actual.grad is not None
        torch.testing.assert_close(
            actual.grad.cpu().float(), expected.grad, atol=5e-2, rtol=5e-2
        )


# (batch, heads, seqlen): S residues mod BM reachable under the FA4
# seqlen % 128 == 0 eligibility gate -- ported from agent A's pure-Mojo fwd
# harness (test_fa4_fwd.mojo in /scratch/fa4-fwd-harness), which validates the
# same shapes' numerics against an fp64 host reference on sampled rows plus a
# bitwise BTHD-vs-BHSD leg comparison. Skips harness shape (1, 16, 4096): a
# full fp32 CPU SDPA reference there allocates a ~1 GB causal score matrix,
# which the pure-Mojo harness already covers far more cheaply. head_dim=128's
# BM is 128 (kFa4BlockM), which the seqlen % 128 == 0 gate always divides
# evenly -- no partial m-tile is reachable there, but the same shape set
# still stresses multi-plane LPT scheduling at the second tile config.
#
# NOTE (phase 2b, BM=64 at d64): the per-shape "tail N" comments below were
# written for the pre-phase-2b BM=192 tile, where seqlen % 128 == 0 still
# left a partial last m-block most of the time. 128 is a multiple of the new
# BM=64, so under this SAME seqlen % 128 == 0 gate every d64 block below is
# now a FULL 64-row block -- these shapes now only exercise multi-block/LPT
# scheduling at d64, not a partial tail. The genuine BM=64 partial-tail path
# (zero-fill load, store clamp, partial-row LSE predicate) needs seqlen NOT
# a multiple of 128, which the production eligibility gate never routes to
# FA4 -- see _FA4_BHSD_D64_TAIL_SHAPES below, which calls the bhsd bridge
# directly to reach it anyway.
_FA4_BHSD_TAIL_SHAPES = (
    (1, 1, 128),  # single m-block, single plane
    (1, 1, 256),  # 2 full BM=64 blocks at d64 (was: tail 64 at BM=192)
    (2, 3, 384),  # exact 2*BM at d128 / 6 full blocks at d64
    (1, 2, 640),  # 10 full BM=64 blocks at d64, 4 kv tiles + LPT
    (5, 7, 896),  # 14 full BM=64 blocks at d64, odd plane count
    (3, 5, 1152),  # exact 6*BM at d128 / 18 full blocks at d64 (awkward B/H)
    (2, 4, 1024),  # 16 full BM=64 blocks at d64, LPT swizzle > 1
)


@pytest.mark.parametrize("head_dim", [64, 128], ids=["d64", "d128"])
@pytest.mark.parametrize("dtype", [torch.bfloat16, torch.float16], ids=["bf16", "f16"])
@pytest.mark.parametrize("batch,heads,seqlen", _FA4_BHSD_TAIL_SHAPES)
def test_fa4_bhsd_native_forward_backward_matches_reference(
    mojo_h100, monkeypatch, batch, heads, seqlen, dtype, head_dim
):
    """Public contiguous, 16-byte-aligned (B, H, S, D) Q/K/V take the
    BHSD-native TMA path (no BTHD gather-copy materialization) across every
    BM=192 tail-residue class reachable under seqlen % 128 == 0, for both
    the d64 and d128 tile configs."""
    from torch_mojo_backend.eager_flash_attention import load_fa4_ops

    module = load_fa4_ops()
    suffix = _FA4_DTYPE_SUFFIX[dtype]
    bhsd_name = f"flash_attention_fwd_{suffix}_d{head_dim}_causal_bhsd"
    original_bhsd = getattr(module, bhsd_name)
    calls = {"count": 0}

    def bhsd_spy(*args):
        calls["count"] += 1
        return original_bhsd(*args)

    monkeypatch.setattr(module, bhsd_name, bhsd_spy)

    generator = torch.Generator().manual_seed(20260816 + seqlen)
    query, key, value = (
        (torch.randn(batch, heads, seqlen, head_dim, generator=generator) * 0.25).to(
            dtype
        )
        for _ in range(3)
    )
    grad_output = torch.randn(batch, heads, seqlen, head_dim, generator=generator).to(
        dtype
    )

    reference_inputs = [
        tensor.float().detach().requires_grad_() for tensor in (query, key, value)
    ]
    reference_output = torch.nn.functional.scaled_dot_product_attention(
        *reference_inputs, dropout_p=0.0, is_causal=True
    )
    reference_output.backward(grad_output.float())

    mojo_inputs = [
        tensor.to(mojo_h100).detach().requires_grad_() for tensor in (query, key, value)
    ]
    for tensor in mojo_inputs:
        assert tensor._is_contiguous
        assert tensor._ptr % 16 == 0

    actual_output = torch.nn.functional.scaled_dot_product_attention(
        *mojo_inputs, dropout_p=0.0, is_causal=True
    )
    assert calls["count"] == 1
    actual_output.backward(grad_output.to(mojo_h100))

    torch.testing.assert_close(
        actual_output.cpu().float(), reference_output, atol=2e-2, rtol=2e-2
    )
    for actual, reference in zip(mojo_inputs, reference_inputs, strict=True):
        assert actual.grad is not None
        torch.testing.assert_close(
            actual.grad.cpu().float(), reference.grad, atol=5e-2, rtol=5e-2
        )


# (batch, heads, seqlen): genuine BM=64 partial-tail shapes at d64 -- ported
# from agent A's phase-2b harness (test_fa4_fwd.mojo). None divides evenly
# by 128. The public eligibility gate now lets a no-grad, contiguous-BHSD
# call reach FA4 at these seqlens too (see test_aten_functions.py's
# test_sdpa_fa4_odd_seqlen_* for that path) -- this test instead calls the
# compiled bhsd bridge function directly with hand-built device buffers,
# bypassing the Python gate and its allocation/dtype/layout plumbing
# entirely, so the new BM=64 tail machinery (zero-fill load past seq_len,
# store clamp, the partial-row LSE predicate, and the causal boundary
# predicate for a warpgroup entirely past seq_len) is exercised in
# isolation.
_FA4_BHSD_D64_TAIL_SHAPES = (
    (2, 3, 65),  # ceil(65/64) = 2 m-blocks; last block has ONE valid row
    (2, 3, 127),  # ceil(127/64) = 2 m-blocks; last block has 63/64 valid rows
    (2, 3, 193),  # ceil(193/64) = 4 m-blocks; last block has ONE valid row,
    #                deeper into the KV walk than the 65 case
    # The three shapes above stay under one wave, so the bridge's wave gate
    # routes them to the v2b kernel -- its tail/clamp logic gets exercised,
    # but the SELFLOAD kernel carries its own copy of that logic and
    # selfload+partial-tail is reachable through the bridge (nothing below
    # the Python gate enforces seqlen % 128). batch*heads here is large
    # enough to cross the >=2-waves-at-3-CTAs/SM gate on any H100:
    # 16*12*ceil(193/64) = 768 CTAs >= 2*3*132.
    (16, 12, 193),
)


@pytest.mark.parametrize("dtype", [torch.bfloat16, torch.float16], ids=["bf16", "f16"])
@pytest.mark.parametrize("batch,heads,seqlen", _FA4_BHSD_D64_TAIL_SHAPES)
def test_fa4_bhsd_d64_direct_partial_tail_block(mojo_h100, batch, heads, seqlen, dtype):
    """BM=64 partial last m-block, calling the compiled bridge directly.

    This calls ``flash_attention_fwd_{bf16,f16}_d64_causal_bhsd`` the same
    way ``fast_fa4_16bit_d64_causal_forward`` does internally, with
    hand-built Q/K/V/out/LSE device buffers, to reach the BM=64 tail path
    the phase-2b harness measured directly rather than through the public
    SDPA/flash-op eligibility gate (see test_aten_functions.py for that
    coverage) or the forward's own allocation/dtype/layout plumbing.
    """
    from torch_mojo_backend.eager_flash_attention import load_fa4_ops
    from torch_mojo_backend.eager_kernels.aten_fast import _ctx_ptr

    module = load_fa4_ops()
    suffix = _FA4_DTYPE_SUFFIX[dtype]
    forward_fn = getattr(module, f"flash_attention_fwd_{suffix}_d64_causal_bhsd")

    head_dim = 64
    generator = torch.Generator().manual_seed(20260817 + seqlen)
    query, key, value = (
        (torch.randn(batch, heads, seqlen, head_dim, generator=generator) * 0.25).to(
            dtype
        )
        for _ in range(3)
    )
    reference_inputs = [tensor.float() for tensor in (query, key, value)]
    reference_output = torch.nn.functional.scaled_dot_product_attention(
        *reference_inputs, dropout_p=0.0, is_causal=True
    )

    q, k, v = (tensor.to(mojo_h100) for tensor in (query, key, value))
    for tensor in (q, k, v):
        assert tensor._is_contiguous
        assert tensor._ptr % 16 == 0
    out = torch.empty(batch, heads, seqlen, head_dim, dtype=dtype, device=mojo_h100)
    logsumexp = torch.empty(batch, heads, seqlen, dtype=torch.float32, device=mojo_h100)
    assert out._ptr % 16 == 0
    scale = 1.0 / math.sqrt(head_dim)

    result = forward_fn(
        q._ptr,
        k._ptr,
        v._ptr,
        out._ptr,
        logsumexp._ptr,
        batch,
        seqlen,
        heads,
        scale,
        _ctx_ptr(q._device),
    )
    assert result is None

    torch.testing.assert_close(
        out.cpu().float(), reference_output, atol=2e-2, rtol=2e-2
    )

    expected_lse = reference_inputs[0] @ reference_inputs[1].transpose(-2, -1) * scale
    causal_mask = torch.ones(seqlen, seqlen, dtype=torch.bool).tril()
    expected_lse = expected_lse.masked_fill(~causal_mask, float("-inf")).logsumexp(-1)
    torch.testing.assert_close(logsumexp.cpu(), expected_lse, atol=2e-2, rtol=2e-2)


# (batch, heads, seqlen): seqlens NOT a multiple of 128 -- the relaxed public
# eligibility gate (PR #391 review thread: "relax the seqlen % 128 == 0
# eligibility gate for the bhsd route") now reaches FA4 through the actual
# public API (torch.nn.functional.scaled_dot_product_attention / the flash
# op), not just the compiled bridge directly. Batch/heads stay modest at the
# larger seqlens so the CPU fp32 causal reference (O(seqlen^2) score matrix)
# stays cheap, same reasoning as _FA4_BHSD_TAIL_SHAPES above.
_FA4_ODD_SEQLEN_SHAPES = (
    (2, 3, 65),
    (2, 3, 127),
    (2, 3, 193),
    (2, 4, 1023),
    (1, 2, 4095),
)


@pytest.mark.parametrize("head_dim", [64, 128], ids=["d64", "d128"])
@pytest.mark.parametrize("dtype", [torch.bfloat16, torch.float16], ids=["bf16", "f16"])
@pytest.mark.parametrize("batch,heads,seqlen", _FA4_ODD_SEQLEN_SHAPES)
def test_fa4_sdpa_odd_seqlen_forward_matches_reference(
    mojo_h100, monkeypatch, batch, heads, seqlen, dtype, head_dim
):
    """No-grad SDPA at an odd seqlen reaches FA4's BHSD-native path.

    Public, contiguous, 16-byte-aligned Q/K/V with no gradient to track is
    exactly the regime the relaxed gate targets: this asserts the BHSD
    bridge actually ran (not just that the numbers happen to match, which
    the math decomposition would also produce) and that its output matches
    a CPU fp32 reference.
    """
    from torch_mojo_backend.eager_flash_attention import load_fa4_ops

    module = load_fa4_ops()
    suffix = _FA4_DTYPE_SUFFIX[dtype]
    bhsd_name = f"flash_attention_fwd_{suffix}_d{head_dim}_causal_bhsd"
    original_bhsd = getattr(module, bhsd_name)
    calls = {"count": 0}

    def bhsd_spy(*args):
        calls["count"] += 1
        return original_bhsd(*args)

    monkeypatch.setattr(module, bhsd_name, bhsd_spy)

    generator = torch.Generator().manual_seed(20260817 + seqlen)
    query, key, value = (
        (torch.randn(batch, heads, seqlen, head_dim, generator=generator) * 0.25).to(
            dtype
        )
        for _ in range(3)
    )
    reference_inputs = [tensor.float() for tensor in (query, key, value)]
    reference_output = torch.nn.functional.scaled_dot_product_attention(
        *reference_inputs, dropout_p=0.0, is_causal=True
    )

    mojo_inputs = [tensor.to(mojo_h100) for tensor in (query, key, value)]
    for tensor in mojo_inputs:
        assert tensor._is_contiguous
        assert tensor._ptr % 16 == 0

    actual_output = torch.nn.functional.scaled_dot_product_attention(
        *mojo_inputs, dropout_p=0.0, is_causal=True
    )

    assert calls["count"] == 1
    torch.testing.assert_close(
        actual_output.cpu().float(), reference_output, atol=2e-2, rtol=2e-2
    )


_FA4_ODD_SEQLEN_FLASH_OP_SHAPES = ((2, 3, 193), (2, 4, 1023))


@pytest.mark.parametrize("head_dim", [64, 128], ids=["d64", "d128"])
@pytest.mark.parametrize("dtype", [torch.bfloat16, torch.float16], ids=["bf16", "f16"])
@pytest.mark.parametrize("batch,heads,seqlen", _FA4_ODD_SEQLEN_FLASH_OP_SHAPES)
def test_fa4_direct_flash_aten_odd_seqlen_forward_matches_reference(
    mojo_h100, batch, heads, seqlen, dtype, head_dim
):
    """The lower ``aten::_scaled_dot_product_flash_attention`` op, called
    directly with no grad to track (as
    ``test_fa4_direct_flash_aten_returns_real_logsumexp`` does at
    seqlen % 128 == 0), also reaches FA4 at an odd seqlen -- this exercises
    the aten-op bridge's own metadata plumbing (LSE shape/dtype, rng/debug-
    mask stand-ins) at the relaxed gate, not just the higher-level SDPA
    wrapper above."""
    generator = torch.Generator().manual_seed(20260817 + seqlen)
    query, key, value = (
        (torch.randn(batch, heads, seqlen, head_dim, generator=generator) * 0.25).to(
            dtype
        )
        for _ in range(3)
    )
    mojo_inputs = [tensor.to(mojo_h100) for tensor in (query, key, value)]

    result = torch.ops.aten._scaled_dot_product_flash_attention.default(
        *mojo_inputs, 0.0, True, False
    )
    output, logsumexp, cum_q, cum_k, max_q, max_k, rng, offset, debug = result

    reference_inputs = [tensor.float() for tensor in (query, key, value)]
    reference_output = torch.nn.functional.scaled_dot_product_attention(
        *reference_inputs, dropout_p=0.0, is_causal=True
    )
    scores = reference_inputs[0] @ reference_inputs[1].transpose(-2, -1)
    scores *= 1.0 / math.sqrt(head_dim)
    causal = torch.ones(seqlen, seqlen, dtype=torch.bool).tril()
    expected_lse = scores.masked_fill(~causal, float("-inf")).logsumexp(-1)

    torch.testing.assert_close(
        output.cpu().float(), reference_output, atol=2e-2, rtol=2e-2
    )
    torch.testing.assert_close(logsumexp.cpu(), expected_lse, atol=2e-2, rtol=2e-2)
    assert cum_q is None and cum_k is None
    assert (max_q, max_k) == (seqlen, seqlen)
    assert rng.dtype == torch.uint64 and tuple(rng.shape) == (2,)
    assert offset.dtype == torch.uint64 and tuple(offset.shape) == ()
    assert debug.dtype == dtype and debug.numel() == 0


@pytest.mark.parametrize("head_dim", [64, 128], ids=["d64", "d128"])
@pytest.mark.parametrize("dtype", [torch.bfloat16, torch.float16], ids=["bf16", "f16"])
def test_fa4_sdpa_odd_seqlen_requires_grad_uses_decomposition(
    mojo_h100, monkeypatch, dtype, head_dim
):
    """A training-shaped call (grad enabled, requires_grad Q/K/V) at an odd
    seqlen must NOT take the native FA4 forward/backward pair: the bwd tile
    machinery has never been proven correct on a partial last tile (unlike
    the BHSD-native forward, see ``fast_fa4_16bit_d64_causal_forward``'s
    docstring), so a silently wrong gradient is exactly the outcome to
    avoid. The autograd wrapper's ``needs_backward`` branch in
    ``mojo_device_autograd.py`` keeps requiring seqlen % 128 == 0, so this
    falls through to the existing generic math/dropout
    ``_ScaledDotProductAttentionAutograd`` Function instead -- which still
    produces correct gradients, proven here against a CPU fp32 reference.
    """
    from torch_mojo_backend.eager_flash_attention import load_fa4_ops

    module = load_fa4_ops()
    suffix = _FA4_DTYPE_SUFFIX[dtype]
    watched_names = {
        f"flash_attention_fwd_{suffix}_d{head_dim}_causal_bhsd",
        f"flash_attention_fwd_{suffix}_d{head_dim}_causal",
        f"flash_attention_fwd_{suffix}_d{head_dim}_causal_strided_qkv",
        f"flash_attention_bwd_{suffix}_d{head_dim}_causal",
        f"flash_attention_bwd_{suffix}_d{head_dim}_causal_strided_qkv",
    }
    calls = dict.fromkeys(watched_names, 0)
    for name in watched_names:
        original = getattr(module, name)

        def wrapper(*args, _name=name, _original=original):
            calls[_name] += 1
            return _original(*args)

        monkeypatch.setattr(module, name, wrapper)

    batch, heads, seqlen = 2, 3, 193
    generator = torch.Generator().manual_seed(20260817 + seqlen)
    query, key, value = (
        (torch.randn(batch, heads, seqlen, head_dim, generator=generator) * 0.25).to(
            dtype
        )
        for _ in range(3)
    )
    grad_output = torch.randn(batch, heads, seqlen, head_dim, generator=generator).to(
        dtype
    )

    reference_inputs = [
        tensor.float().detach().requires_grad_() for tensor in (query, key, value)
    ]
    reference_output = torch.nn.functional.scaled_dot_product_attention(
        *reference_inputs, dropout_p=0.0, is_causal=True
    )
    reference_output.backward(grad_output.float())

    mojo_inputs = [
        tensor.to(mojo_h100).detach().requires_grad_() for tensor in (query, key, value)
    ]

    actual_output = torch.nn.functional.scaled_dot_product_attention(
        *mojo_inputs, dropout_p=0.0, is_causal=True
    )
    # The decomposition Function's autograd node, not the native flash pair
    # (see test_fa4_causal_gapped_qkv_forward_backward for that name).
    assert (
        type(actual_output.grad_fn).__name__
        != "ScaledDotProductFlashAttentionBackward0"
    )
    actual_output.backward(grad_output.to(mojo_h100))

    assert calls == dict.fromkeys(watched_names, 0)
    torch.testing.assert_close(
        actual_output.cpu().float(), reference_output, atol=2e-2, rtol=2e-2
    )
    for actual, reference in zip(mojo_inputs, reference_inputs, strict=True):
        assert actual.grad is not None
        torch.testing.assert_close(
            actual.grad.cpu().float(), reference.grad, atol=5e-2, rtol=5e-2
        )


@pytest.mark.parametrize("head_dim", [64, 128], ids=["d64", "d128"])
def test_fa4_bhsd_gate_rejects_misaligned_offset_view(mojo_h100, monkeypatch, head_dim):
    """A public (B, H, S, D) tensor that is a genuine offset slice of a
    larger buffer is fully contiguous but its base pointer is not 16-byte
    aligned -- ``_fa4_bhsd_layout`` must reject it (TMA descriptor creation
    needs 16-byte-aligned base addresses) and fall back to the existing
    BTHD-materialize-and-copy path. Still produces correct output. A 2-byte
    (one bf16 element) offset breaks 16-byte alignment independently of
    head_dim, so this is exercised at both tile configs."""
    from torch_mojo_backend.eager_flash_attention import load_fa4_ops

    module = load_fa4_ops()
    calls = {"bhsd": 0, "dense": 0}
    bhsd_name = f"flash_attention_fwd_bf16_d{head_dim}_causal_bhsd"
    dense_name = f"flash_attention_fwd_bf16_d{head_dim}_causal"
    original_bhsd = getattr(module, bhsd_name)
    original_dense = getattr(module, dense_name)

    def bhsd_spy(*args):
        calls["bhsd"] += 1
        return original_bhsd(*args)

    def dense_spy(*args):
        calls["dense"] += 1
        return original_dense(*args)

    monkeypatch.setattr(module, bhsd_name, bhsd_spy)
    monkeypatch.setattr(module, dense_name, dense_spy)

    batch, heads, seqlen = 1, 4, 128
    numel = batch * heads * seqlen * head_dim
    generator = torch.Generator().manual_seed(20260816)

    def offset_qkv() -> torch.Tensor:
        # One extra element (2 bytes) ahead of a fresh allocation: the
        # [1:] slice is still fully contiguous once viewed, but its base
        # pointer is no longer a multiple of 16.
        host = (
            torch.randn(numel + 1, generator=generator, dtype=torch.float32) * 0.25
        ).to(torch.bfloat16)
        view = host.to(mojo_h100)[1:].view(batch, heads, seqlen, head_dim)
        assert view._is_contiguous
        assert view._ptr % 16 != 0
        return view

    query, key, value = (offset_qkv() for _ in range(3))

    reference_inputs = [tensor.detach().cpu().float() for tensor in (query, key, value)]
    reference_output = torch.nn.functional.scaled_dot_product_attention(
        *reference_inputs, dropout_p=0.0, is_causal=True
    )

    actual_output = torch.nn.functional.scaled_dot_product_attention(
        query, key, value, dropout_p=0.0, is_causal=True
    )

    assert calls == {"bhsd": 0, "dense": 1}
    torch.testing.assert_close(
        actual_output.cpu().float(), reference_output, atol=2e-2, rtol=2e-2
    )


@pytest.mark.parametrize("mutated", ["query", "output"])
def test_fa4_autograd_rejects_saved_tensor_mutation(mojo_h100, mutated):
    """The physical FA4 copies must not bypass PyTorch version checks."""
    generator = torch.Generator().manual_seed(20260720)
    batch, seqlen, heads, head_dim = 1, 128, 4, 64
    width = heads * head_dim
    fused = torch.randn(
        batch, seqlen, 3 * width, generator=generator, dtype=torch.bfloat16
    ).to(mojo_h100)
    query, key, value = (
        part.view(batch, seqlen, heads, head_dim)
        .transpose(1, 2)
        .detach()
        .requires_grad_()
        for part in fused.split(width, dim=2)
    )
    output = torch.nn.functional.scaled_dot_product_attention(
        query, key, value, dropout_p=0.0, is_causal=True
    )
    with torch.no_grad():
        if mutated == "query":
            query.add_(1.0)
        else:
            output.add_(1.0)

    grad_output = torch.ones(output.shape, dtype=torch.bfloat16).to(mojo_h100)
    with pytest.raises(RuntimeError, match="modified by an inplace operation"):
        output.backward(grad_output)


def test_fast_sdpa_causal_training_backward(mojo_gpu):
    """Causal SDPA must propagate correct gradients to all three inputs."""
    generator = torch.Generator().manual_seed(20260718)
    shape = (2, 12, 128, 64)
    q = torch.randn(shape, generator=generator)
    k = torch.randn(shape, generator=generator)
    v = torch.randn(shape, generator=generator)
    grad_output = torch.randn(shape, generator=generator)

    # Keep the oracle on CPU: PyTorch 2.11's CUDA autograd currently trips an
    # internal stream assertion once a PrivateUse1 backend is registered.
    reference_device = "cpu"
    ref_inputs = [
        tensor.detach().clone().to(reference_device).requires_grad_()
        for tensor in (q, k, v)
    ]
    ref_output = torch.nn.functional.scaled_dot_product_attention(
        *ref_inputs, dropout_p=0.0, is_causal=True
    )
    ref_output.backward(grad_output.to(reference_device))
    ref_grads = [tensor.grad.cpu() for tensor in ref_inputs]

    mojo_inputs = [
        tensor.detach().clone().to(mojo_gpu).requires_grad_() for tensor in (q, k, v)
    ]
    mojo_output = torch.nn.functional.scaled_dot_product_attention(
        *mojo_inputs, dropout_p=0.0, is_causal=True
    )
    mojo_output.backward(grad_output.to(mojo_gpu))

    for name, tensor, expected_grad in zip(
        ("query", "key", "value"), mojo_inputs, ref_grads, strict=True
    ):
        assert tensor.grad is not None, f"{name} gradient was not computed"
        actual_grad = tensor.grad.cpu()
        assert torch.isfinite(actual_grad).all(), f"{name} gradient is not finite"
        torch.testing.assert_close(actual_grad, expected_grad, atol=2e-2, rtol=2e-2)


@pytest.mark.parametrize(
    (
        "requires",
        "expected_saved",
        "expected_bmm",
        "expected_transpose_b_bmm",
        "expected_fused_backward",
    ),
    [
        ("query", ("key", "value", "probabilities"), 1, 1, 1),
        ("key", ("query", "value", "probabilities"), 1, 1, 1),
        ("value", ("probabilities",), 1, 0, 0),
    ],
)
def test_fast_sdpa_partial_gradients_save_and_compute_only_dependencies(
    mojo_gpu,
    monkeypatch,
    requires,
    expected_saved,
    expected_bmm,
    expected_transpose_b_bmm,
    expected_fused_backward,
):
    """Each single-input gradient skips unrelated saves and BMM branches.

    This pins the DECOMPOSITION's internals -- which tensors it saves, how many
    batched GEMMs each requested gradient costs -- so the fused gfx942 path is
    disabled below.  That path is a different algorithm with a deliberately
    different profile: it saves Q/K/V/O/L, which is O(n*d) rather than the
    decomposition's O(n^2) probability matrix, and it produces dQ, dK and dV
    together because all three consume the same recomputed scores.  Its own
    contract is pinned by
    ``test_fused_flash_attention_partial_gradients_match_reference``.
    """
    from torch_mojo_backend.eager_kernels import aten_fast
    from torch_mojo_backend.mojo_device.torch_mojo_tensor import TorchMojoTensor

    monkeypatch.setattr(aten_fast, "_fused_fa_inputs", lambda *a, **k: None)

    generator = torch.Generator().manual_seed(20260718)
    batch, heads, query_length, key_length, head_dim = 2, 2, 5, 7, 3
    host_inputs = [
        torch.randn(batch, heads, query_length, head_dim, generator=generator),
        torch.randn(batch, heads, key_length, head_dim, generator=generator),
        torch.randn(batch, heads, key_length, head_dim, generator=generator),
    ]
    grad_output = torch.randn(batch, heads, query_length, head_dim, generator=generator)
    names = ("query", "key", "value")

    reference_inputs = [
        tensor.clone().requires_grad_(name == requires)
        for name, tensor in zip(names, host_inputs, strict=True)
    ]
    reference_output = torch.nn.functional.scaled_dot_product_attention(
        *reference_inputs, dropout_p=0.0, is_causal=True
    )
    reference_output.backward(grad_output)

    calls = {"bmm": 0, "transpose_b_bmm": 0, "fused_backward": 0}
    original_bmm = aten_fast.fast_aten_bmm
    original_causal_bmm = aten_fast._try_sdpa_causal_bmm
    original_transpose_b_bmm = aten_fast._fast_aten_bmm_transpose_b
    original_fused_backward = aten_fast.fast_sdpa_dropout_softmax_backward
    original_fused_route = aten_fast.fast_sdpa_backward
    original_materialize = TorchMojoTensor._materialize_contiguous
    materialized_shapes = []
    fused_route_handled = []

    def spy_bmm(*args):
        calls["bmm"] += 1
        return original_bmm(*args)

    def spy_causal_bmm(*args):
        # With is_causal=True the backward's plain batched GEMMs are replaced by
        # the causal ones, which skip the contraction indices the mask kills.
        # They play exactly the same role here, so count them the same way: this
        # test is about *how many* matmuls each requested gradient costs.
        result = original_causal_bmm(*args)
        if result is not None:
            calls["bmm"] += 1
        return result

    def spy_transpose_b_bmm(*args):
        calls["transpose_b_bmm"] += 1
        return original_transpose_b_bmm(*args)

    def spy_fused_backward(*args):
        calls["fused_backward"] += 1
        return original_fused_backward(*args)

    def spy_fused_route(*args, **kwargs):
        result = original_fused_route(*args, **kwargs)
        fused_route_handled.append(result is not aten_fast.NOT_HANDLED)
        return result

    def spy_materialize(self):
        materialized_shapes.append(tuple(self._shape))
        return original_materialize(self)

    monkeypatch.setattr(aten_fast, "fast_aten_bmm", spy_bmm)
    monkeypatch.setattr(aten_fast, "_try_sdpa_causal_bmm", spy_causal_bmm)
    monkeypatch.setattr(aten_fast, "_fast_aten_bmm_transpose_b", spy_transpose_b_bmm)
    monkeypatch.setattr(
        aten_fast, "fast_sdpa_dropout_softmax_backward", spy_fused_backward
    )
    monkeypatch.setattr(aten_fast, "fast_sdpa_backward", spy_fused_route)
    monkeypatch.setattr(TorchMojoTensor, "_materialize_contiguous", spy_materialize)

    actual_inputs = [
        tensor.to(mojo_gpu).requires_grad_(name == requires)
        for name, tensor in zip(names, host_inputs, strict=True)
    ]
    actual_output = torch.nn.functional.scaled_dot_product_attention(
        *actual_inputs, dropout_p=0.0, is_causal=True
    )
    assert actual_output.grad_fn.saved_names == expected_saved
    # The counts below are about the backward. The causal forward also issues a
    # batched GEMM through the same helper, so zero the counters here rather than
    # let a forward call be mistaken for a gradient's.
    calls.update(bmm=0, transpose_b_bmm=0, fused_backward=0)
    actual_output.backward(grad_output.to(mojo_gpu))

    if fused_route_handled == [True]:
        # The Apple fused route replaces every composed branch outright; its
        # own launches respect the same dependency pruning by construction.
        assert calls == {"bmm": 0, "transpose_b_bmm": 0, "fused_backward": 0}
    else:
        assert calls == {
            "bmm": expected_bmm,
            "transpose_b_bmm": expected_transpose_b_bmm,
            "fused_backward": expected_fused_backward,
        }
    # Neither the old P^T nor dScores^T (both SxL) may be materialized.
    assert (batch * heads, key_length, query_length) not in materialized_shapes
    for name, actual, reference in zip(
        names, actual_inputs, reference_inputs, strict=True
    ):
        assert (actual.grad is not None) == (name == requires)
        if name == requires:
            assert actual.grad._is_contiguous
            torch.testing.assert_close(
                actual.grad.cpu(), reference.grad, atol=3e-2, rtol=3e-2
            )


@pytest.mark.parametrize("requires", ["query", "key", "value"])
def test_fused_flash_attention_partial_gradients_match_reference(mojo_gpu, requires):
    """The fused path returns exactly the requested gradient, and it is right.

    Same shape as the decomposition's partial-gradient test, and deliberately
    non-square with is_causal=True: query_length 5 against key_length 7 is where
    top-left and bottom-right causal alignment disagree, so a kernel using the
    wrong convention fails here rather than silently passing on square inputs.
    """
    if list(get_accelerators())[0].architecture_name != "gfx942":
        pytest.skip("the fused flash-attention kernels target gfx942")

    from torch_mojo_backend.eager_kernels import aten_fast

    generator = torch.Generator().manual_seed(20260718)
    batch, heads, query_length, key_length, head_dim = 2, 2, 5, 7, 3
    host_inputs = [
        torch.randn(batch, heads, query_length, head_dim, generator=generator),
        torch.randn(batch, heads, key_length, head_dim, generator=generator),
        torch.randn(batch, heads, key_length, head_dim, generator=generator),
    ]
    grad_output = torch.randn(batch, heads, query_length, head_dim, generator=generator)
    names = ("query", "key", "value")

    reference_inputs = [
        tensor.clone().requires_grad_(name == requires)
        for name, tensor in zip(names, host_inputs, strict=True)
    ]
    torch.nn.functional.scaled_dot_product_attention(
        *reference_inputs, dropout_p=0.0, is_causal=True
    ).backward(grad_output)

    actual_inputs = [
        tensor.to(mojo_gpu).requires_grad_(name == requires)
        for name, tensor in zip(names, host_inputs, strict=True)
    ]
    # Only meaningful if the fused path actually claims these inputs.
    assert (
        aten_fast._fused_fa_inputs(*actual_inputs, None, 0.0, True, None, False)
        is not None
    ), "fused path declined the inputs this test exists to cover"

    torch.nn.functional.scaled_dot_product_attention(
        *actual_inputs, dropout_p=0.0, is_causal=True
    ).backward(grad_output.to(mojo_gpu))

    for name, actual, reference in zip(
        names, actual_inputs, reference_inputs, strict=True
    ):
        assert (actual.grad is not None) == (name == requires)
        if name == requires:
            assert torch.isfinite(actual.grad.cpu()).all()
            torch.testing.assert_close(
                actual.grad.cpu(), reference.grad, atol=3e-2, rtol=3e-2
            )


def test_fast_sdpa_saved_tensor_hooks_own_saved_allocations(mojo_gpu, monkeypatch):
    """CPU pack results, not ctx payloads, own SDPA's saved activations.

    Pins the DECOMPOSITION's saved set, so the fused gfx942 path is disabled
    here; the same ownership property is asserted for that path's own,
    different saved set by the test below.
    """
    from torch_mojo_backend.eager_kernels import aten_fast

    monkeypatch.setattr(aten_fast, "_fused_fa_inputs", lambda *a, **k: None)

    generator = torch.Generator().manual_seed(20260718)
    batch, heads, query_length, key_length, head_dim = 1, 2, 5, 7, 4
    host_inputs = [
        torch.randn(batch, heads, query_length, head_dim, generator=generator),
        torch.randn(batch, heads, key_length, head_dim, generator=generator),
        torch.randn(batch, heads, key_length, head_dim, generator=generator),
    ]
    grad_output = torch.randn(batch, heads, query_length, head_dim, generator=generator)
    reference_inputs = [tensor.clone().requires_grad_() for tensor in host_inputs]
    torch.nn.functional.scaled_dot_product_attention(
        *reference_inputs, dropout_p=0.0, is_causal=False
    ).backward(grad_output)

    hook_calls = []

    def pack(tensor):
        hook_calls.append(("pack", tensor.device.type, tuple(tensor.shape)))
        return tensor.cpu()

    def unpack(tensor):
        hook_calls.append(("unpack", tensor.device.type, tuple(tensor.shape)))
        return tensor

    actual_inputs = [tensor.to(mojo_gpu).requires_grad_() for tensor in host_inputs]
    with torch.autograd.graph.saved_tensors_hooks(pack, unpack):
        actual_output = torch.nn.functional.scaled_dot_product_attention(
            *actual_inputs, dropout_p=0.0, is_causal=False
        )
        assert actual_output.grad_fn.saved_names == (
            "query",
            "key",
            "value",
            "probabilities",
        )
        assert all(
            payload.holder is None for payload in actual_output.grad_fn.saved_payloads
        )
        actual_output.backward(grad_output.to(mojo_gpu))

    saved_shapes = [tuple(tensor.shape) for tensor in host_inputs] + [
        (batch, heads, query_length, key_length)
    ]
    assert hook_calls == [("pack", "mojo", shape) for shape in saved_shapes] + [
        ("unpack", "cpu", shape) for shape in saved_shapes
    ]
    for actual, reference in zip(actual_inputs, reference_inputs, strict=True):
        assert actual.grad is not None
        torch.testing.assert_close(
            actual.grad.cpu(), reference.grad, atol=3e-2, rtol=3e-2
        )


def test_fused_flash_attention_saved_tensor_hooks_own_saved_allocations(mojo_gpu):
    """The fused path's saved set is hook-owned too, and it is the smaller one.

    The decomposition saves the O(n^2) probability matrix; this path saves Q, K,
    V, the output and the per-row log-sum-exp, all O(n*d) except the scalar per
    query.  At the shape below that is the difference between packing a
    5x7 matrix and packing a length-5 vector, and the gap grows with sequence.
    """
    if list(get_accelerators())[0].architecture_name != "gfx942":
        pytest.skip("the fused flash-attention kernels target gfx942")

    from torch_mojo_backend.eager_kernels import aten_fast

    generator = torch.Generator().manual_seed(20260718)
    batch, heads, query_length, key_length, head_dim = 1, 2, 5, 7, 4
    host_inputs = [
        torch.randn(batch, heads, query_length, head_dim, generator=generator),
        torch.randn(batch, heads, key_length, head_dim, generator=generator),
        torch.randn(batch, heads, key_length, head_dim, generator=generator),
    ]
    grad_output = torch.randn(batch, heads, query_length, head_dim, generator=generator)
    reference_inputs = [tensor.clone().requires_grad_() for tensor in host_inputs]
    torch.nn.functional.scaled_dot_product_attention(
        *reference_inputs, dropout_p=0.0, is_causal=False
    ).backward(grad_output)

    hook_calls = []

    def pack(tensor):
        hook_calls.append(("pack", tensor.device.type, tuple(tensor.shape)))
        return tensor.cpu()

    def unpack(tensor):
        hook_calls.append(("unpack", tensor.device.type, tuple(tensor.shape)))
        return tensor

    actual_inputs = [tensor.to(mojo_gpu).requires_grad_() for tensor in host_inputs]
    assert (
        aten_fast._fused_fa_inputs(*actual_inputs, None, 0.0, False, None, False)
        is not None
    ), "fused path declined the inputs this test exists to cover"

    with torch.autograd.graph.saved_tensors_hooks(pack, unpack):
        actual_output = torch.nn.functional.scaled_dot_product_attention(
            *actual_inputs, dropout_p=0.0, is_causal=False
        )
        assert actual_output.grad_fn.saved_names == (
            "query",
            "key",
            "value",
            "output",
            "logsumexp",
        )
        # With hooks active the packed value owns the allocation, so no payload
        # may retain the original holder -- that is what would defeat a
        # CPU/offload hook.
        assert all(
            payload.holder is None for payload in actual_output.grad_fn.saved_payloads
        )
        actual_output.backward(grad_output.to(mojo_gpu))

    saved_shapes = [tuple(tensor.shape) for tensor in host_inputs] + [
        (batch, heads, query_length, head_dim),
        (batch, heads, query_length),
    ]
    assert hook_calls == [("pack", "mojo", shape) for shape in saved_shapes] + [
        ("unpack", "cpu", shape) for shape in saved_shapes
    ]
    for actual, reference in zip(actual_inputs, reference_inputs, strict=True):
        assert actual.grad is not None
        torch.testing.assert_close(
            actual.grad.cpu(), reference.grad, atol=3e-2, rtol=3e-2
        )


def test_fast_sdpa_saved_tensor_hook_rejects_holderless_mojo_result(mojo_gpu):
    """A malformed Mojo unpack result fails before any dangling pointer use."""
    shape = (1, 1, 3, 4)
    query = torch.randn(shape).to(mojo_gpu)
    key = torch.randn(shape).to(mojo_gpu)
    value = torch.randn(shape).to(mojo_gpu).requires_grad_()

    def pack(tensor):
        return tensor.cpu()

    def unpack(tensor):
        malformed = tensor.to(mojo_gpu)
        del malformed._holder
        return malformed

    with torch.autograd.graph.saved_tensors_hooks(pack, unpack):
        output = torch.nn.functional.scaled_dot_product_attention(query, key, value)
        with pytest.raises(
            RuntimeError,
            match="unusable Mojo tensor without a TorchMojoTensor allocation holder",
        ):
            output.backward(torch.ones(shape).to(mojo_gpu))


def test_sdpa_fused_backward_host_bridge_abi(mojo_gpu, monkeypatch):
    """The host helper forwards offset pointers and flattened runtime shape."""
    from torch_mojo_backend.eager_kernels import aten_fast

    shape = (2, 3, 5)
    elements = 30
    probs_storage = torch.randn(elements + 2).to(mojo_gpu)
    grad_storage = torch.randn(elements + 4).to(mojo_gpu)
    mask_storage = torch.ones(elements + 6, dtype=torch.bool).to(mojo_gpu)
    probabilities = probs_storage[1 : 1 + elements].view(shape)
    grad = grad_storage[2 : 2 + elements].view(shape)
    mask = mask_storage[3 : 3 + elements].view(shape)
    calls = []
    _replace_defined_native_calls(
        monkeypatch,
        {
            ("sdpa_backward_ops.mojo", "SDPADropoutSoftmaxBackward"): (
                lambda *args: calls.append(args)
            )
        },
    )

    out = aten_fast.fast_sdpa_dropout_softmax_backward(
        probabilities, grad, mask, 1.25, -0.5
    )

    assert out is not aten_fast.NOT_HANDLED
    assert tuple(out.shape) == shape
    assert len(calls) == 1
    args = calls[0]
    assert args[:4] == (out._ptr, probabilities._ptr, grad._ptr, mask._ptr)
    assert args[4:11] == (6, 5, 0, 1, 0, 1.25, -0.5)
    assert args[11] == probabilities._dtype.value
    assert args[12] == aten_fast._ctx_ptr(probabilities._device)


def test_sdpa_fused_backward_materializes_strided_operands(mojo_gpu, monkeypatch):
    """The raw bridge sees dense temporary pointers, never strided metadata."""
    from torch_mojo_backend.eager_kernels import aten_fast

    shape = (2, 3, 5)
    probabilities = torch.randn(shape).to(mojo_gpu).transpose(1, 2)
    grad = torch.randn(shape).to(mojo_gpu).transpose(1, 2)
    mask = torch.ones(shape, dtype=torch.bool).to(mojo_gpu).transpose(1, 2)
    assert not probabilities._is_contiguous
    assert not grad._is_contiguous
    assert not mask._is_contiguous
    original_ptrs = (probabilities._ptr, grad._ptr, mask._ptr)
    calls = []
    _replace_defined_native_calls(
        monkeypatch,
        {
            ("sdpa_backward_ops.mojo", "SDPADropoutSoftmaxBackward"): (
                lambda *args: calls.append(args)
            )
        },
    )

    out = aten_fast.fast_sdpa_dropout_softmax_backward(
        probabilities, grad, mask, 1.25, 0.125
    )

    assert out is not aten_fast.NOT_HANDLED
    assert tuple(out.shape) == (2, 5, 3)
    assert out._is_contiguous
    assert len(calls) == 1
    args = calls[0]
    assert all(actual != original for actual, original in zip(args[1:4], original_ptrs))
    assert args[4:11] == (10, 3, 0, 1, 0, 1.25, 0.125)
    assert args[11] == probabilities._dtype.value
    assert args[12] == aten_fast._ctx_ptr(probabilities._device)


def test_sdpa_fused_backward_no_mask_ignores_dropout_scale(mojo_gpu, monkeypatch):
    from torch_mojo_backend.eager_kernels import aten_fast

    calls = []
    _replace_defined_native_calls(
        monkeypatch,
        {
            ("sdpa_backward_ops.mojo", "SDPADropoutSoftmaxBackward"): (
                lambda *args: calls.append(args)
            )
        },
    )
    probabilities = torch.randn(3, 7).to(mojo_gpu)
    grad = torch.randn(3, 7).to(mojo_gpu)

    out = aten_fast.fast_sdpa_dropout_softmax_backward(
        probabilities, grad, None, object(), 0.0
    )

    assert out is not aten_fast.NOT_HANDLED
    assert len(calls) == 1
    assert calls[0][3:11] == (0, 3, 7, 0, 0, 0, 1.0, 0.0)


def test_sdpa_fused_backward_empty_skips_bridge(mojo_gpu, monkeypatch):
    from torch_mojo_backend.eager_kernels import aten_fast

    calls = []
    _replace_defined_native_calls(
        monkeypatch,
        {
            ("sdpa_backward_ops.mojo", "SDPADropoutSoftmaxBackward"): (
                lambda *args: calls.append(args)
            )
        },
    )
    probabilities = torch.empty(2, 0).to(mojo_gpu)
    grad = torch.empty(2, 0).to(mojo_gpu)
    out = aten_fast.fast_sdpa_dropout_softmax_backward(
        probabilities, grad, None, float("nan"), -2.0
    )

    assert out is not aten_fast.NOT_HANDLED
    assert tuple(out.shape) == (2, 0)
    assert calls == []


@pytest.mark.parametrize(
    "invalid",
    [
        "probabilities_dtype",
        "grad_dtype",
        "grad_shape",
        "mask_dtype",
        "mask_shape",
        "dropout_scale",
        "score_scale",
        "rank_zero",
    ],
)
def test_sdpa_fused_backward_validates_before_materializing(
    mojo_gpu, monkeypatch, invalid
):
    from torch_mojo_backend.eager_kernels import aten_fast

    probabilities = torch.randn(2, 3).to(mojo_gpu)
    grad = torch.randn(2, 3).to(mojo_gpu)
    mask = torch.ones(2, 3, dtype=torch.bool).to(mojo_gpu)
    dropout_scale = 1.25
    score_scale = 0.5
    if invalid == "probabilities_dtype":
        probabilities = torch.randn(2, 3, dtype=torch.float16).to(mojo_gpu)
    elif invalid == "grad_dtype":
        grad = torch.randn(2, 3, dtype=torch.float16).to(mojo_gpu)
    elif invalid == "grad_shape":
        grad = torch.randn(3, 2).to(mojo_gpu)
    elif invalid == "mask_dtype":
        mask = torch.ones(2, 3, dtype=torch.uint8).to(mojo_gpu)
    elif invalid == "mask_shape":
        mask = torch.ones(3, 2, dtype=torch.bool).to(mojo_gpu)
    elif invalid == "dropout_scale":
        dropout_scale = float("nan")
    elif invalid == "score_scale":
        score_scale = float("inf")
    else:
        probabilities = torch.randn(()).to(mojo_gpu)
        grad = torch.randn(()).to(mojo_gpu)
        mask = torch.ones((), dtype=torch.bool).to(mojo_gpu)

    def reject_materialization(_tensor):
        raise AssertionError("invalid metadata reached materialization")

    monkeypatch.setattr(aten_fast, "_tc", reject_materialization)
    assert (
        aten_fast.fast_sdpa_dropout_softmax_backward(
            probabilities, grad, mask, dropout_scale, score_scale
        )
        is aten_fast.NOT_HANDLED
    )


@pytest.mark.parametrize("is_causal", [False, True])
def test_fast_sdpa_dropout_matches_captured_mask_reference(
    mojo_gpu, monkeypatch, is_causal
):
    """Dropout belongs after softmax and before both value-gradient BMMs."""
    from torch_mojo_backend.eager_kernels import aten_fast

    generator = torch.Generator().manual_seed(20260718)
    batch, heads, length, head_dim = 2, 3, 7, 4
    shape = (batch, heads, length, head_dim)
    host_inputs = [torch.randn(shape, generator=generator) for _ in range(3)]
    grad_output = torch.randn(shape, generator=generator)
    mojo_inputs = [tensor.to(mojo_gpu).requires_grad_() for tensor in host_inputs]

    captured_masks = []
    native_dropout = aten_fast.fast_aten_native_dropout

    def capture_dropout(input, p, train):
        result = native_dropout(input, p, train)
        captured_masks.append(result[1])
        return result

    monkeypatch.setattr(aten_fast, "fast_aten_native_dropout", capture_dropout)
    torch.mojo.manual_seed_all((1 << 63) + 20260718)
    actual_output = torch.nn.functional.scaled_dot_product_attention(
        *mojo_inputs, dropout_p=0.2, is_causal=is_causal
    )
    assert len(captured_masks) == 1
    keep = captured_masks[0].cpu().reshape(batch, heads, length, length)
    state_after_forward = torch.mojo.get_rng_state(mojo_inputs[0].device)

    reference_inputs = [tensor.clone().requires_grad_() for tensor in host_inputs]
    query, key, value = reference_inputs
    scores = query @ key.transpose(-2, -1) * (head_dim**-0.5)
    if is_causal:
        causal_mask = torch.ones(length, length, dtype=torch.bool).tril()
        scores = scores.masked_fill(~causal_mask, float("-inf"))
    probabilities = torch.softmax(scores, dim=-1)
    reference_output = probabilities.mul(keep).mul(1.25) @ value

    reference_output.backward(grad_output)
    actual_output.backward(grad_output.to(mojo_gpu))

    torch.testing.assert_close(
        torch.mojo.get_rng_state(mojo_inputs[0].device), state_after_forward
    )
    torch.testing.assert_close(
        actual_output.cpu(), reference_output.detach(), atol=2e-2, rtol=2e-2
    )
    for name, actual, reference in zip(
        ("query", "key", "value"), mojo_inputs, reference_inputs, strict=True
    ):
        assert actual.grad is not None, f"{name} gradient was not computed"
        torch.testing.assert_close(
            actual.grad.cpu(), reference.grad, atol=3e-2, rtol=3e-2
        )


@pytest.mark.parametrize("dropout_p", [0.0, 0.2, 1.0])
def test_fast_sdpa_dropout_reserves_exact_probability_interval(mojo_gpu, dropout_p):
    batch, heads, query_length, key_length, head_dim = 2, 3, 5, 7, 4
    query = torch.randn(batch, heads, query_length, head_dim).to(mojo_gpu)
    key = torch.randn(batch, heads, key_length, head_dim).to(mojo_gpu)
    value = torch.randn(batch, heads, key_length, head_dim).to(mojo_gpu)
    seed = (1 << 63) + 0x1234
    counter = (1 << 63) + 0x5678
    torch.mojo.set_rng_state(_philox_rng_state(seed, counter), query.device)

    output = torch.nn.functional.scaled_dot_product_attention(
        query, key, value, dropout_p=dropout_p, is_causal=True
    )
    probability_elements = batch * heads * query_length * key_length
    expected_increment = (probability_elements + 3) // 4 if 0.0 < dropout_p < 1.0 else 0
    assert _decode_philox_rng_state(torch.mojo.get_rng_state(query.device)) == (
        seed,
        counter + expected_increment,
    )
    host_output = output.cpu()
    assert torch.isfinite(host_output).all()
    if dropout_p == 1.0:
        torch.testing.assert_close(host_output, torch.zeros_like(host_output))


def test_fast_sdpa_full_dropout_has_zero_gradients(mojo_gpu):
    shape = (1, 2, 8, 4)
    inputs = [torch.randn(shape).to(mojo_gpu).requires_grad_() for _ in range(3)]
    output = torch.nn.functional.scaled_dot_product_attention(
        *inputs, dropout_p=1.0, is_causal=True
    )
    output.backward(torch.randn(shape).to(mojo_gpu))

    torch.testing.assert_close(output.cpu(), torch.zeros(shape))
    for tensor in inputs:
        assert tensor.grad is not None
        torch.testing.assert_close(tensor.grad.cpu(), torch.zeros(shape))


def test_fast_sdpa_full_dropout_preserves_nonfinite_arithmetic(mojo_gpu):
    """Math SDPA uses ``P * 0`` rather than native-dropout's zero fill."""
    shape = (1, 1, 2, 2)
    host_inputs = [
        torch.full(shape, float("nan")),
        torch.tensor([[[[1.0, 2.0], [3.0, 4.0]]]]),
        torch.tensor([[[[5.0, 6.0], [7.0, 8.0]]]]),
    ]
    reference_inputs = [tensor.clone().requires_grad_() for tensor in host_inputs]
    actual_inputs = [tensor.to(mojo_gpu).requires_grad_() for tensor in host_inputs]

    reference_output = torch.nn.functional.scaled_dot_product_attention(
        *reference_inputs, dropout_p=1.0
    )
    actual_output = torch.nn.functional.scaled_dot_product_attention(
        *actual_inputs, dropout_p=1.0
    )
    reference_output.backward(torch.ones_like(reference_output))
    actual_output.backward(torch.ones(shape).to(mojo_gpu))

    torch.testing.assert_close(actual_output.cpu(), reference_output, equal_nan=True)
    for actual, reference in zip(actual_inputs, reference_inputs, strict=True):
        assert actual.grad is not None
        torch.testing.assert_close(actual.grad.cpu(), reference.grad, equal_nan=True)


def test_fast_sdpa_backward_rejects_mutated_saved_input(mojo_gpu):
    shape = (1, 2, 8, 16)
    inputs = [torch.randn(shape).to(mojo_gpu).requires_grad_() for _ in range(3)]
    output = torch.nn.functional.scaled_dot_product_attention(
        *inputs, dropout_p=0.0, is_causal=True
    )

    with torch.no_grad():
        inputs[0].add_(torch.ones_like(inputs[0]))

    with pytest.raises(RuntimeError, match="modified by an inplace operation"):
        output.backward(torch.randn(shape).to(mojo_gpu))


def test_fast_sdpa_nanogpt_shakespeare_dropout_backward(mojo_gpu):
    """Exercise nanoGPT Shakespeare's training-time attention configuration.

    The batch is reduced to keep the focused kernel test bounded, while the
    model geometry matches its six heads, 256-token context, and 64-wide heads.
    Dropout RNG differs by backend, so only gradient existence and finiteness
    are portable correctness requirements at this full attention geometry.
    """
    generator = torch.Generator().manual_seed(20260718)
    shape = (2, 6, 256, 64)
    inputs = [
        torch.randn(shape, generator=generator).to(mojo_gpu).requires_grad_()
        for _ in range(3)
    ]
    grad_output = torch.randn(shape, generator=generator).to(mojo_gpu)

    output = torch.nn.functional.scaled_dot_product_attention(
        *inputs, dropout_p=0.2, is_causal=True
    )
    output.backward(grad_output)

    for name, tensor in zip(("query", "key", "value"), inputs, strict=True):
        assert tensor.grad is not None, f"{name} gradient was not computed"
        assert torch.isfinite(tensor.grad.cpu()).all(), f"{name} gradient is not finite"


def test_fast_log_softmax_training_backward(mojo_gpu):
    """Autograd must retain a valid Mojo payload for the saved forward output."""
    generator = torch.Generator().manual_seed(20260718)
    x = torch.randn(32, 65, generator=generator)
    grad_output = torch.randn(32, 65, generator=generator)

    reference = x.clone().requires_grad_()
    torch.nn.functional.log_softmax(reference, dim=-1).backward(grad_output)

    actual = x.to(mojo_gpu).requires_grad_()
    output = torch.nn.functional.log_softmax(actual, dim=-1)
    assert type(output.grad_fn).__name__ == "LogSoftmaxBackward0"
    output.backward(grad_output.to(mojo_gpu))

    assert actual.grad is not None
    torch.testing.assert_close(actual.grad.cpu(), reference.grad, atol=2e-5, rtol=2e-5)


def test_fast_log_softmax_uses_saved_tensor_hooks(mojo_gpu):
    generator = torch.Generator().manual_seed(20260718)
    x = torch.randn(8, 17, generator=generator)
    grad_output = torch.randn(8, 17, generator=generator)
    reference = x.clone().requires_grad_()
    torch.nn.functional.log_softmax(reference, dim=-1).backward(grad_output)
    hook_calls = []

    def pack(tensor):
        hook_calls.append(("pack", tensor.device.type))
        return tensor.cpu()

    def unpack(tensor):
        hook_calls.append(("unpack", tensor.device.type))
        return tensor.to(mojo_gpu)

    actual = x.to(mojo_gpu).requires_grad_()
    with torch.autograd.graph.saved_tensors_hooks(pack, unpack):
        torch.nn.functional.log_softmax(actual, dim=-1).backward(
            grad_output.to(mojo_gpu)
        )

    assert hook_calls == [("pack", "mojo"), ("unpack", "cpu")]
    torch.testing.assert_close(actual.grad.cpu(), reference.grad, atol=2e-5, rtol=2e-5)


def test_fast_log_softmax_native_double_backward(mojo_gpu):
    generator = torch.Generator().manual_seed(20260722)
    host_input = torch.randn(4, 7, generator=generator)
    first_seed = torch.randn(4, 7, generator=generator)
    second_seed = torch.randn(4, 7, generator=generator)

    def derivatives(input, seed1, seed2):
        output = torch.nn.functional.log_softmax(input, dim=-1)
        (first,) = torch.autograd.grad(
            output, input, grad_outputs=seed1, create_graph=True
        )
        (second,) = torch.autograd.grad(first, input, grad_outputs=seed2)
        return first, second

    reference = host_input.clone().requires_grad_()
    expected_first, expected_second = derivatives(reference, first_seed, second_seed)

    actual = host_input.to(mojo_gpu).requires_grad_()
    actual_first, actual_second = derivatives(
        actual, first_seed.to(mojo_gpu), second_seed.to(mojo_gpu)
    )
    torch.testing.assert_close(actual_first.cpu(), expected_first, atol=2e-5, rtol=2e-5)
    torch.testing.assert_close(
        actual_second.cpu(), expected_second, atol=2e-5, rtol=2e-5
    )


def test_fast_log_softmax_backward_rejects_mutated_saved_output(mojo_gpu):
    output = torch.nn.functional.log_softmax(
        torch.randn(8, 17).to(mojo_gpu).requires_grad_(), dim=-1
    )
    with torch.no_grad():
        output.add_(torch.ones_like(output))

    with pytest.raises(RuntimeError, match="modified by an inplace operation"):
        output.backward(torch.randn(8, 17).to(mojo_gpu))


def test_fast_log_softmax_does_not_retain_python_output_cycle(mojo_gpu):
    input = torch.randn(8, 17).to(mojo_gpu).requires_grad_()
    output = torch.nn.functional.log_softmax(input, dim=-1)
    output_ref = weakref.ref(output)

    # Drop the reference the kernel-call queue is *entitled* to hold first: a
    # queued launch names its operands by raw pointer, so its item retains
    # every tensor it touches until the launch has run. That retention is
    # not the cycle under test -- it ends at the synchronize -- and without
    # this the assertion below would pass or fail on whether the queue
    # happened to be empty rather than on the Python graph.
    torch.accelerator.synchronize()

    del output

    # The backward node saves the OUTPUT, so a node held by the tensor that
    # also holds that tensor is the classic collectable cycle; refcounting
    # alone must be enough to free it.
    assert output_ref() is None


@pytest.mark.parametrize("dtype", [torch.float32, torch.float16])
def test_fast_sdpa_decode(mojo_gpu, dtype):
    # q_len == 1 selects the fused decode kernel used by GPT-2 generation.
    q = torch.randn(4, 12, 1, 64, dtype=dtype)
    k = torch.randn(4, 12, 128, 64, dtype=dtype)
    v = torch.randn(4, 12, 128, 64, dtype=dtype)
    dev = torch.nn.functional.scaled_dot_product_attention(
        q.to(mojo_gpu), k.to(mojo_gpu), v.to(mojo_gpu)
    ).cpu()
    ref = torch.nn.functional.scaled_dot_product_attention(q, k, v)
    torch.testing.assert_close(dev, ref, atol=1e-2, rtol=1e-2)
