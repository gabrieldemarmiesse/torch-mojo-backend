"""Tests for the Mojo-extension fast path used by mojo eager mode."""

import math
import weakref
from types import SimpleNamespace

import pytest
import torch

from torch_mojo_backend import get_accelerators, register_mojo_devices

pytestmark = pytest.mark.xdist_group(name="group1")


@pytest.fixture(autouse=True)
def setup_max_device():
    register_mojo_devices()


BINARY_OPS = [torch.add, torch.sub, torch.mul, torch.div, torch.maximum, torch.minimum]
UNARY_OPS = [torch.relu, torch.exp]


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


def test_fast_path_is_used(mojo_device):
    """The eligible case must go through the Mojo kernel, not the fallback.

    Tensor-tensor adds route through the shared spec op, or through the
    Apple flat kernel selected during Metal device registration."""
    from torch_mojo_backend import eager_kernels

    calls = []
    x = torch.randn(8, 8).to(mojo_device)
    y = torch.randn(8, 8).to(mojo_device)
    if x._device.api == "metal":
        module = eager_kernels.elementwise_ops
        name = "Add"
    else:
        module = eager_kernels.logic_ops
        name = "AddSpec"
    original = getattr(module, name)

    def spy(*args):
        calls.append(args)
        return original(*args)

    setattr(module, name, spy)
    try:
        _ = x + y
    finally:
        setattr(module, name, original)
    assert len(calls) == 1


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
def test_spec_path_is_used(mojo_device, module_name, spec_name, fn):
    """One representative op per converted family must route through its
    spec entry (whole prologue in one Mojo call), not the classic chain."""
    from torch_mojo_backend import eager_kernels

    module = getattr(eager_kernels, module_name)
    calls = []
    original = getattr(module, spec_name)

    def spy(*args):
        calls.append(args)
        return original(*args)

    setattr(module, spec_name, spy)
    try:
        x = torch.randn(8, 8).to(mojo_device)
        y = torch.randn(8, 8).to(mojo_device)
        _ = fn(x, y)
    finally:
        setattr(module, spec_name, original)
    assert len(calls) == 1


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
    from torch_mojo_backend import eager_kernels

    generator = torch.Generator().manual_seed(20260720)
    fp32 = torch.randn(shape, generator=generator)
    bf16 = torch.randn(shape, generator=generator).to(torch.bfloat16)
    lhs, rhs = (bf16, fp32) if bf16_first else (fp32, bf16)

    fused_calls = []
    cast_calls = []
    original_fused = eager_kernels.logic_ops.AddF32Bf16Spec
    original_cast = eager_kernels.data_movement_ops.CastSpec

    def fused_spy(*args):
        fused_calls.append(args)
        return original_fused(*args)

    def cast_spy(*args):
        cast_calls.append(args)
        return original_cast(*args)

    monkeypatch.setattr(eager_kernels.logic_ops, "AddF32Bf16Spec", fused_spy)
    monkeypatch.setattr(eager_kernels.data_movement_ops, "CastSpec", cast_spy)

    actual = lhs.to(mojo_gpu) + rhs.to(mojo_gpu)
    expected = lhs + rhs

    assert actual.dtype == torch.float32
    torch.testing.assert_close(actual.cpu(), expected, atol=0, rtol=0)
    assert len(fused_calls) == 1
    assert not cast_calls


def test_fast_add_f32_bf16_fused_spec_avoids_cast_temporary(mojo_gpu):
    """A contiguous mixed add must be one fused spec launch, even with tails."""
    from torch_mojo_backend import eager_kernels

    fp32_storage = torch.randn(1_106).to(mojo_gpu)
    bf16_storage = torch.randn(1_106).to(torch.bfloat16).to(mojo_gpu)
    fp32 = fp32_storage[1:]
    bf16 = bf16_storage[1:]

    fused_calls = []
    cast_calls = []
    original_fused = eager_kernels.logic_ops.AddF32Bf16Spec
    original_cast = eager_kernels.data_movement_ops.CastSpec

    def fused_spy(*args):
        fused_calls.append(args)
        return original_fused(*args)

    def cast_spy(*args):
        cast_calls.append(args)
        return original_cast(*args)

    eager_kernels.logic_ops.AddF32Bf16Spec = fused_spy
    eager_kernels.data_movement_ops.CastSpec = cast_spy
    try:
        outputs = (fp32 + bf16, bf16 + fp32)
    finally:
        eager_kernels.logic_ops.AddF32Bf16Spec = original_fused
        eager_kernels.data_movement_ops.CastSpec = original_cast

    expected = fp32_storage.cpu()[1:] + bf16_storage.cpu()[1:]
    for output in outputs:
        assert output.dtype == torch.float32
        torch.testing.assert_close(output.cpu(), expected, atol=0, rtol=0)
    assert len(fused_calls) == 2
    assert not cast_calls


def test_fast_add_f32_bf16_strided_preserves_general_fallback(mojo_gpu):
    """Ineligible layouts remain correct through the existing general path."""
    from torch_mojo_backend import eager_kernels

    fp32 = torch.randn(7, 11)
    bf16 = torch.randn(7, 11).to(torch.bfloat16)
    device_fp32 = fp32.to(mojo_gpu).t()
    device_bf16 = bf16.to(mojo_gpu).t()

    fused_calls = []
    cast_calls = []
    original_fused = eager_kernels.logic_ops.AddF32Bf16Spec
    original_cast = eager_kernels.data_movement_ops.CastSpec

    def fused_spy(*args):
        fused_calls.append(args)
        return original_fused(*args)

    def cast_spy(*args):
        cast_calls.append(args)
        return original_cast(*args)

    eager_kernels.logic_ops.AddF32Bf16Spec = fused_spy
    eager_kernels.data_movement_ops.CastSpec = cast_spy
    try:
        actual = device_fp32 + device_bf16
    finally:
        eager_kernels.logic_ops.AddF32Bf16Spec = original_fused
        eager_kernels.data_movement_ops.CastSpec = original_cast

    torch.testing.assert_close(actual.cpu(), fp32.t() + bf16.t(), atol=0, rtol=0)
    assert not fused_calls
    assert len(cast_calls) == 1


@pytest.fixture
def mojo_gpu(mojo_gpu_available: bool):
    """GPU mojo device only — for ops whose fast path is GPU-gated."""
    if not mojo_gpu_available:
        pytest.skip("You do not have a GPU supported by MAX")
    return "mojo:0"


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
    assert saved._strides == output._strides
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
def test_fast_transpose(mojo_device, dims):
    x = torch.randn(2, 3, 4)
    result = x.to(mojo_device).transpose(*dims).contiguous().cpu()
    torch.testing.assert_close(result, x.transpose(*dims).contiguous())


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


def test_fast_native_layer_norm_bf16_gpu_preserves_generic_path(mojo_gpu):
    input = torch.randn(3, 65).bfloat16()
    weight = torch.randn(65).bfloat16()
    bias = torch.randn(65).bfloat16()
    expected = torch.native_layer_norm(input, (65,), weight, bias, 1e-5)

    actual = torch.native_layer_norm(
        input.to(mojo_gpu), (65,), weight.to(mojo_gpu), bias.to(mojo_gpu), 1e-5
    )

    assert actual[0].dtype == torch.bfloat16
    assert actual[1].dtype == actual[2].dtype == torch.float32
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
    from torch_mojo_backend.eager_kernels import aten_fast

    input = torch.randn(3, 65).to(mojo_gpu)
    weight = torch.randn(65).to(mojo_gpu)
    bias = torch.randn(65).to(mojo_gpu)
    tensors = {"input": input, "weight": weight, "bias": bias}
    tensors[requires].requires_grad_()
    seen_mask_bits = []
    backward_ops = aten_fast.eager_kernels.normalization_backward_ops
    original = backward_ops.LayerNormBackwardF32

    def spy(*args):
        seen_mask_bits.append(args[-2])
        return original(*args)

    monkeypatch.setattr(backward_ops, "LayerNormBackwardF32", spy)
    output = torch.nn.functional.layer_norm(input, (65,), weight, bias, 1e-5)
    output.backward(torch.ones(3, 65).to(mojo_gpu))

    expected_mask_bits = 1 << ("input", "weight", "bias").index(requires)
    assert seen_mask_bits == [expected_mask_bits]
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


def _dropout_rng_state(seed: int, counter: int) -> torch.Tensor:
    encoded = seed.to_bytes(8, "little") + counter.to_bytes(8, "little")
    return torch.tensor(list(encoded), dtype=torch.uint8)


def _decode_dropout_rng_state(state: torch.Tensor) -> tuple[int, int]:
    encoded = bytes(state.reshape(-1).tolist())
    return int.from_bytes(encoded[:8], "little"), int.from_bytes(encoded[8:], "little")


def test_fast_native_dropout_reserves_exact_full_width_interval(mojo_gpu, monkeypatch):
    from torch_mojo_backend import eager_kernels
    from torch_mojo_backend.eager_kernels import aten_fast

    calls = []
    monkeypatch.setattr(
        eager_kernels,
        "dropout_ops",
        SimpleNamespace(NativeDropoutF32=lambda *args: calls.append(args)),
        raising=False,
    )
    input = torch.arange(9, dtype=torch.float32).to(mojo_gpu)
    seed = (1 << 63) + 0x1234_5678
    counter = (1 << 63) + 0x9ABC_DEF0
    torch.mojo.set_rng_state(_dropout_rng_state(seed, counter), input.device)

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
    assert _decode_dropout_rng_state(torch.mojo.get_rng_state(input.device)) == (
        seed,
        counter + 3,
    )


def test_fast_native_dropout_reservation_wrap_does_not_mutate_state(
    mojo_gpu, monkeypatch
):
    from torch_mojo_backend import eager_kernels
    from torch_mojo_backend.eager_kernels import aten_fast

    calls = []
    monkeypatch.setattr(
        eager_kernels,
        "dropout_ops",
        SimpleNamespace(NativeDropoutF32=lambda *args: calls.append(args)),
        raising=False,
    )
    input = torch.arange(4, dtype=torch.float32).to(mojo_gpu)
    seed = (1 << 64) - 1
    torch.mojo.set_rng_state(_dropout_rng_state(seed, (1 << 64) - 2), input.device)

    aten_fast.fast_aten_native_dropout(input, 0.0, True)
    endpoint = torch.mojo.get_rng_state(input.device)
    assert _decode_dropout_rng_state(endpoint) == (seed, (1 << 64) - 1)
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


def test_fast_reduction_library_tier(mojo_device):
    """Huge-col reductions route to the stdlib reduction library (GPU: rows <=
    128 and cols >= 2**20; MAX-CPU: always) — exercise that tier, which no other
    test reaches. Integer-valued floats keep every f32 partial sum exact, so the
    comparisons are bit-exact instead of tolerance-based."""
    # rows == 1: full-reduction layout (the two-phase GPU tier's main case).
    x = torch.randint(-4, 5, (1, 2**20 + 7)).float()
    xd = x.to(mojo_device)
    torch.testing.assert_close(xd.sum(-1).cpu(), x.sum(-1))
    torch.testing.assert_close(xd.amax(-1).cpu(), x.amax(-1))
    torch.testing.assert_close(torch.any(xd, -1).cpu(), torch.any(x, -1))

    # rows == 128 (gate boundary): per-row outputs must land in the right rows.
    y = torch.randint(-4, 5, (128, 2**20)).float()
    y[5] = 0.0  # give any() a False row
    yd = y.to(mojo_device)
    torch.testing.assert_close(yd.sum(-1).cpu(), y.sum(-1))
    torch.testing.assert_close(yd.amax(-1).cpu(), y.amax(-1))
    torch.testing.assert_close(torch.any(yd, -1).cpu(), torch.any(y, -1))


def test_fast_anyall_nan_is_truthy(mojo_device):
    """torch treats NaN as truthy in any/all. Cover both dispatch tiers: the
    small shape uses the block kernel on GPU (and the library on MAX-CPU), the
    huge shape uses the library tier everywhere."""
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
    from torch_mojo_backend import eager_kernels
    from torch_mojo_backend.eager_kernels import _ctx_ptr, aten_fast

    grad_storage = torch.arange(2 * 3 * 5 + 3, dtype=torch.float32).to(mojo_gpu)
    index_storage = torch.tensor([99, 99, 1, 2, 1, 4, 2, 7], dtype=torch.int64).to(
        mojo_gpu
    )
    grad_output = grad_storage[3:].view(2, 3, 5)
    indices = index_storage[2:].view(2, 3)
    calls = []
    monkeypatch.setitem(
        eager_kernels.__dict__,
        "embedding_backward_ops",
        SimpleNamespace(EmbeddingDenseBackwardF32I64=lambda *args: calls.append(args)),
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
    from torch_mojo_backend import eager_kernels
    from torch_mojo_backend.eager_kernels.aten_fast import _ctx_ptr

    elements = math.prod(shape)
    backing = torch.linspace(-8.0, 8.0, elements + storage_offset, dtype=torch.bfloat16)
    input = backing[storage_offset:].view(shape)
    device_backing = backing.to(mojo_gpu)
    device_input = device_backing[storage_offset:].view(shape)
    input_ptr = device_input._ptr
    input_version = device_input._version
    calls = []
    original = eager_kernels.activation_forward_ops.GeluForwardBF16

    def spy(*args):
        calls.append(args)
        return original(*args)

    monkeypatch.setattr(eager_kernels.activation_forward_ops, "GeluForwardBF16", spy)
    actual = torch.nn.functional.gelu(device_input, approximate=approximate)
    expected = torch.nn.functional.gelu(input, approximate=approximate)

    assert calls == [
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
    assert torch.isnan(actual[3])
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
    from torch_mojo_backend import eager_kernels

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

    monkeypatch.setitem(
        eager_kernels.__dict__,
        "activation_forward_ops",
        SimpleNamespace(GeluForwardBF16=reject_direct),
    )
    actual = torch.nn.functional.gelu(device_input, approximate=approximate)
    expected = torch.nn.functional.gelu(input, approximate=approximate)

    tolerance = 5e-5 if input.dtype == torch.float32 else 2e-2
    torch.testing.assert_close(actual.cpu(), expected, atol=tolerance, rtol=tolerance)


@pytest.mark.parametrize("approximate", ["none", "tanh"])
def test_fast_gelu_forward_bf16_cpu_preserves_generic_path(monkeypatch, approximate):
    """BF16 on the MAX CPU device must not enter the accelerator bridge."""
    from torch_mojo_backend import eager_kernels

    cpu_device = f"mojo:{len(list(get_accelerators())) - 1}"
    input = torch.randn(5, 7, dtype=torch.bfloat16)

    def reject_direct(*_args):
        raise AssertionError("MAX CPU GELU used the direct GPU bridge")

    monkeypatch.setitem(
        eager_kernels.__dict__,
        "activation_forward_ops",
        SimpleNamespace(GeluForwardBF16=reject_direct),
    )
    actual = torch.nn.functional.gelu(
        input.to(cpu_device), approximate=approximate
    ).cpu()
    expected = torch.nn.functional.gelu(input, approximate=approximate)

    torch.testing.assert_close(actual, expected, atol=2e-2, rtol=2e-2)


def test_fast_gelu_forward_bf16_empty_does_not_enqueue(mojo_gpu, monkeypatch):
    """Empty BF16 tensors preserve metadata without launching a GPU kernel."""
    from torch_mojo_backend import eager_kernels

    def reject_direct(*_args):
        raise AssertionError("empty GELU forward enqueued a kernel")

    monkeypatch.setattr(
        eager_kernels.activation_forward_ops, "GeluForwardBF16", reject_direct
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
    from torch_mojo_backend import eager_kernels

    input = torch.linspace(-8.0, 8.0, 257)
    grad_output = torch.linspace(2.0, -2.0, 257)
    reference = input.clone().requires_grad_()
    reference_output = torch.nn.functional.gelu(reference, approximate=approximate)
    reference_output.backward(grad_output)

    calls = 0
    original = eager_kernels.activation_backward_ops.GeluBackwardF32

    def spy(*args):
        nonlocal calls
        calls += 1
        return original(*args)

    monkeypatch.setattr(eager_kernels.activation_backward_ops, "GeluBackwardF32", spy)
    actual = input.to(mojo_gpu).requires_grad_()
    actual_output = torch.nn.functional.gelu(actual, approximate=approximate)
    actual_output.backward(grad_output.to(mojo_gpu))

    assert calls == 1
    torch.testing.assert_close(actual_output.cpu(), reference_output)
    assert actual.grad is not None
    torch.testing.assert_close(actual.grad.cpu(), reference.grad, atol=5e-5, rtol=5e-5)


@pytest.mark.parametrize("approximate", ["none", "tanh"])
def test_fast_gelu_training_bf16_uses_direct_backward(
    mojo_h100, monkeypatch, approximate
):
    """BF16 autograd must call both dedicated Mojo bridges exactly once."""
    from torch_mojo_backend import eager_kernels

    input = torch.linspace(-8.0, 8.0, 257, dtype=torch.bfloat16)
    grad_output = torch.linspace(2.0, -2.0, 257, dtype=torch.bfloat16)
    reference = input.clone().requires_grad_()
    reference_output = torch.nn.functional.gelu(reference, approximate=approximate)
    reference_output.backward(grad_output)

    forward_calls = 0
    backward_calls = 0
    original_forward = eager_kernels.activation_forward_ops.GeluForwardBF16
    original_backward = eager_kernels.activation_backward_ops.GeluBackwardBF16

    def forward_spy(*args):
        nonlocal forward_calls
        forward_calls += 1
        return original_forward(*args)

    def backward_spy(*args):
        nonlocal backward_calls
        backward_calls += 1
        return original_backward(*args)

    monkeypatch.setattr(
        eager_kernels.activation_forward_ops, "GeluForwardBF16", forward_spy
    )
    monkeypatch.setattr(
        eager_kernels.activation_backward_ops, "GeluBackwardBF16", backward_spy
    )
    actual = input.to(mojo_h100).requires_grad_()
    actual_output = torch.nn.functional.gelu(actual, approximate=approximate)
    actual_output.backward(grad_output.to(mojo_h100))

    assert forward_calls == 1
    assert backward_calls == 1
    torch.testing.assert_close(
        actual_output.cpu(), reference_output, atol=2e-2, rtol=2e-2
    )
    assert actual.grad is not None
    torch.testing.assert_close(actual.grad.cpu(), reference.grad, atol=2e-2, rtol=2e-2)


def test_fast_gelu_backward_bf16_empty_does_not_enqueue(mojo_h100, monkeypatch):
    """Empty BF16 tensors preserve metadata without launching a GPU kernel."""
    from torch_mojo_backend import eager_kernels

    def fail(*_args):
        raise AssertionError("empty GELU backward enqueued a kernel")

    monkeypatch.setattr(eager_kernels.activation_backward_ops, "GeluBackwardBF16", fail)
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
    from torch_mojo_backend import eager_kernels

    calls = 0
    original = eager_kernels.softmax_backward_ops.LogSoftmaxBackwardData

    def spy(*args):
        nonlocal calls
        calls += 1
        return original(*args)

    monkeypatch.setattr(
        eager_kernels.softmax_backward_ops, "LogSoftmaxBackwardData", spy
    )

    source = torch.randn(8, 640)
    grad_output = torch.randn(8, 640)
    reference = source.clone().requires_grad_()
    reference_output = torch.log_softmax(reference, dim=-1)
    reference_output.backward(grad_output)

    actual = source.to(mojo_gpu).requires_grad_()
    actual_output = torch.log_softmax(actual, dim=-1)
    actual_output.backward(grad_output.to(mojo_gpu))

    assert calls == 1
    assert actual.grad is not None
    torch.testing.assert_close(actual.grad.cpu(), reference.grad, atol=2e-5, rtol=2e-5)


def test_fast_log_softmax_backward_non_trailing_keeps_composed_path(
    mojo_gpu, monkeypatch
):
    """Non-trailing dims and the f32->f16 promotion stay on the composed path."""
    from torch_mojo_backend import eager_kernels

    def fail(*_args):
        raise AssertionError("composed-path case reached the fused kernel")

    monkeypatch.setattr(
        eager_kernels.softmax_backward_ops, "LogSoftmaxBackwardData", fail
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
    from torch_mojo_backend import eager_kernels

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

    calls = {"forward": 0, "backward": 0}
    original_forward = eager_kernels.loss_ops.NllLossForwardF32
    original_backward = eager_kernels.loss_ops.NllLossBackwardF32

    def spy_forward(*args):
        calls["forward"] += 1
        return original_forward(*args)

    def spy_backward(*args):
        calls["backward"] += 1
        return original_backward(*args)

    monkeypatch.setattr(eager_kernels.loss_ops, "NllLossForwardF32", spy_forward)
    monkeypatch.setattr(eager_kernels.loss_ops, "NllLossBackwardF32", spy_backward)

    actual = logits.to(mojo_gpu).requires_grad_()
    actual_loss = torch.nn.functional.cross_entropy(
        actual, target.to(mojo_gpu), reduction=reduction, ignore_index=-1
    )
    actual_loss.backward(grad_output.to(mojo_gpu))

    assert calls == {"forward": 1, "backward": 1}
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
    from torch_mojo_backend import eager_kernels

    assert "tf32_matmul_ops" in eager_kernels._MOJO_MODULES


def test_bf16_module_is_available_for_lazy_import():
    from torch_mojo_backend import eager_kernels

    assert "bf16_matmul_ops" in eager_kernels._MOJO_MODULES


def test_bf16_v3_source_dependency_and_kernel_contract():
    """The lazy bridge includes v3 while v2 remains its explicit fallback."""
    from torch_mojo_backend.eager_kernels import aten_fast

    assert [path.name for path in aten_fast._BF16_SOURCE_PATHS] == [
        "bf16_matmul_ops.mojo",
        "bf16_gemm_v3_kernels.mojo",
        "bf16_gemm_kernels.mojo",
    ]
    bridge_path, v3_path, fallback_path = aten_fast._BF16_SOURCE_PATHS
    bridge_source = bridge_path.read_text()
    v3_source = v3_path.read_text()
    fallback_source = fallback_path.read_text()

    assert "from bf16_gemm_v3_kernels import" in bridge_source
    assert "from bf16_gemm_kernels import (" in v3_source
    for kernel_name in (
        "nanogpt_bf16_gemm_v3_nn_ws_m64n128_tma_s3",
        "nanogpt_bf16_gemm_v3_nn_ws_m128n256_tma_s3",
        "nanogpt_bf16_gemm_v3_nt_ws_m128n256_tma_s3",
        "nanogpt_bf16_gemm_v3_tn_ws_m64n128_tma_col_a_s3",
        "nanogpt_bf16_gemm_v3_tn_ws_m128n256_tma_col_a_s3",
        "nanogpt_bf16_gemm_v3_tn_wgmma_tma_transpose_s2",
    ):
        assert f'@__name("{kernel_name}")' in v3_source

    for helper_name in (
        "_v3_enqueue_nn_ws_m64n128_tma_s3",
        "_v3_enqueue_nn_ws_m128n256_tma_s3",
    ):
        assert v3_source.count(f"{helper_name}(") == 2

    nt_kernel_start = v3_source.index(
        '@__name("nanogpt_bf16_gemm_v3_nt_ws_m128n256_tma_s3")'
    )
    nt_kernel_end = v3_source.index(
        '@__name("nanogpt_bf16_gemm_v3_tn_ws_m64n128_tma_col_a_s3")'
    )
    nt_source = v3_source[nt_kernel_start:nt_kernel_end]
    assert "b_tma.prefetch_descriptor()\n        barrier()" in nt_source
    assert "DeviceAttribute.MULTIPROCESSOR_COUNT" in v3_source
    for scratch_only in (
        "nanogpt_bf16_gemm_v3_nn_wgmma_tma_s2",
        "_v3_enqueue_nt_ws_m128n256_tma_s4",
        "candidate_bf16_gemm_accepted_v2",
        "GPT-5.6-SOL",
    ):
        assert scratch_only not in v3_source

    for source in (bridge_source, v3_source, fallback_source):
        for forbidden in (
            ".synchronize(",
            "devicecontext(",
            "from linalg.matmul",
            "cublas",
            "cudnn",
            "rocblas",
            "triton",
        ):
            assert forbidden not in source.lower()


@pytest.mark.parametrize("operation", ["gemm", "bmm"])
@pytest.mark.parametrize("failure_mode", ["missing_source", "import_error"])
def test_bf16_unavailable_bridge_falls_back_before_allocation(
    monkeypatch, operation, failure_mode
):
    """An unavailable optional BF16 bridge must not allocate or compile twice."""
    from torch_mojo_backend import eager_kernels
    from torch_mojo_backend.eager_kernels import aten_fast

    device = SimpleNamespace(label="gpu", api="cuda", architecture_name="sm_90a")

    def tensor(shape):
        shape = tuple(shape)
        return SimpleNamespace(
            _shape=shape,
            _strides=aten_fast._row_major_strides(shape),
            _dtype=aten_fast.DType.bfloat16,
            _device=device,
            _ptr=1234,
            _is_contiguous=True,
        )

    def fail_allocation(*_args, **_kwargs):
        raise AssertionError("unavailable BF16 bridge allocated an output")

    import_calls = []
    if failure_mode == "missing_source":
        sources = (SimpleNamespace(is_file=lambda: False),)

        def import_module(name):
            import_calls.append(name)
            raise AssertionError("missing BF16 source attempted a lazy import")

    else:
        sources = (SimpleNamespace(is_file=lambda: True),)

        def import_module(name):
            import_calls.append(name)
            raise ImportError("synthetic Mojo compiler failure")

    monkeypatch.setattr(aten_fast, "_t", lambda value: value)
    monkeypatch.setattr(aten_fast, "_alloc", fail_allocation)
    monkeypatch.setattr(aten_fast, "_BF16_SOURCE_PATHS", sources)
    monkeypatch.setattr(aten_fast, "_BF16_IMPORT_FAILED", False)
    monkeypatch.setattr(eager_kernels, "_import_mojo_module", import_module)
    monkeypatch.setitem(eager_kernels.__dict__, "tensor_holder", object())
    monkeypatch.delitem(eager_kernels.__dict__, "bf16_matmul_ops", raising=False)

    def call():
        if operation == "gemm":
            return aten_fast._try_bf16_gemm(tensor((3, 4)), tensor((4, 5)))
        return aten_fast._try_bf16_bmm(tensor((2, 3, 4)), tensor((2, 4, 5)))

    assert call() is None
    assert call() is None
    assert import_calls == (
        [] if failure_mode == "missing_source" else ["bf16_matmul_ops"]
    )


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

    monkeypatch.setattr(aten_fast, "_try_bf16_gemm", try_gemm)
    monkeypatch.setattr(aten_fast, "_try_bf16_linear", try_linear)
    monkeypatch.setattr(aten_fast, "_try_bf16_bmm", try_bmm)
    monkeypatch.setattr(aten_fast, "_try_tf32_gemm", fail_later_route)
    monkeypatch.setattr(aten_fast, "_try_tf32_linear", fail_later_route)
    monkeypatch.setattr(aten_fast, "_try_tf32_bmm", fail_later_route)
    monkeypatch.setattr(aten_fast, "_try_spec_matmul", fail_later_route)

    assert aten_fast.fast_aten_mm(lhs, rhs) is gemm_result
    assert aten_fast.fast_aten_addmm(bias, lhs, rhs) is gemm_result
    assert aten_fast.fast_aten_linear(lhs, weight, bias) is linear_result
    assert aten_fast.fast_aten_bmm(lhs, rhs) is bmm_result
    assert aten_fast._fast_aten_bmm_transpose_b(lhs, rhs) is bmm_result

    assert gemm_calls == [((lhs, rhs), {}), ((lhs, rhs, bias), {})]
    assert linear_calls == [((lhs, weight, bias), {})]
    assert bmm_calls == [((lhs, rhs), {}), ((lhs, rhs), {"transpose_b": True})]


@pytest.mark.parametrize("operation", ["gemm", "bmm"])
@pytest.mark.parametrize("failure_mode", ["missing_source", "import_error"])
def test_tf32_unavailable_bridge_falls_back_before_allocation(
    monkeypatch, operation, failure_mode
):
    """An optional TF32 bridge failure must retain the existing SIMT path."""
    from torch_mojo_backend import eager_kernels
    from torch_mojo_backend.eager_kernels import aten_fast

    device = SimpleNamespace(label="gpu", api="cuda", architecture_name="sm_90a")

    def tensor(shape):
        shape = tuple(shape)
        return SimpleNamespace(
            _shape=shape,
            _strides=aten_fast._row_major_strides(shape),
            _dtype=aten_fast.DType.float32,
            _device=device,
            _ptr=1234,
            _is_contiguous=True,
        )

    def fail_allocation(*_args, **_kwargs):
        raise AssertionError("unavailable TF32 bridge allocated an output")

    import_calls = []
    if failure_mode == "missing_source":
        sources = (SimpleNamespace(is_file=lambda: False),)

        def import_module(name):
            import_calls.append(name)
            raise AssertionError("missing TF32 source attempted a lazy import")

    else:
        sources = (SimpleNamespace(is_file=lambda: True),)

        def import_module(name):
            import_calls.append(name)
            raise ImportError("synthetic Mojo compiler failure")

    monkeypatch.setattr(aten_fast.torch, "get_float32_matmul_precision", lambda: "high")
    monkeypatch.setattr(aten_fast, "_t", lambda value: value)
    monkeypatch.setattr(aten_fast, "_alloc", fail_allocation)
    monkeypatch.setattr(aten_fast, "_TF32_SOURCE_PATHS", sources)
    monkeypatch.setattr(aten_fast, "_TF32_IMPORT_FAILED", False)
    monkeypatch.setattr(eager_kernels, "_import_mojo_module", import_module)
    monkeypatch.setitem(eager_kernels.__dict__, "tensor_holder", object())
    monkeypatch.delitem(eager_kernels.__dict__, "tf32_matmul_ops", raising=False)

    def call():
        if operation == "gemm":
            return aten_fast._try_tf32_gemm(tensor((3, 4)), tensor((4, 5)))
        return aten_fast._try_tf32_bmm(tensor((2, 3, 4)), tensor((2, 4, 5)))

    assert call() is None
    assert call() is None
    assert import_calls == (
        [] if failure_mode == "missing_source" else ["tf32_matmul_ops"]
    )


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

    monkeypatch.setattr(aten_fast, "_try_bf16_gemm", lambda *_args, **_kwargs: None)
    monkeypatch.setattr(aten_fast, "_try_bf16_linear", lambda *_args, **_kwargs: None)
    monkeypatch.setattr(aten_fast, "_try_bf16_bmm", lambda *_args, **_kwargs: None)
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
    """Strict FP32 rejects TF32 before inspecting operands, then uses SIMT."""
    from torch_mojo_backend import eager_kernels
    from torch_mojo_backend.eager_kernels import aten_fast

    lhs, rhs, bias, input, weight = (object() for _ in range(5))
    fallback = object()
    spec_calls = []
    tf32_import_calls = []

    def fail_tensor_inspection(*_args, **_kwargs):
        raise AssertionError("strict FP32 inspected a TF32 operand")

    def fail_tf32_import(name):
        tf32_import_calls.append(name)
        raise AssertionError("strict FP32 lazily imported the TF32 extension")

    def spec(spec_name, tensors, transpose_b):
        spec_calls.append((spec_name, tensors, transpose_b))
        return fallback

    monkeypatch.delitem(eager_kernels.__dict__, "tf32_matmul_ops", raising=False)
    monkeypatch.setattr(eager_kernels, "_import_mojo_module", fail_tf32_import)
    monkeypatch.setattr(
        aten_fast.torch, "get_float32_matmul_precision", lambda: "highest"
    )
    monkeypatch.setattr(aten_fast, "_try_bf16_gemm", lambda *_args, **_kwargs: None)
    monkeypatch.setattr(aten_fast, "_try_bf16_linear", lambda *_args, **_kwargs: None)
    monkeypatch.setattr(aten_fast, "_try_bf16_bmm", lambda *_args, **_kwargs: None)
    monkeypatch.setattr(aten_fast, "_t", fail_tensor_inspection)
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
    assert "tf32_matmul_ops" not in eager_kernels.__dict__


def test_matmul_spec_device_oom_is_not_disguised_as_unsupported(monkeypatch):
    from torch_mojo_backend import eager_kernels
    from torch_mojo_backend.eager_kernels import aten_fast

    lhs = object()
    rhs = object()

    def raise_allocator_oom(*_args):
        raise NotImplementedError(
            "CUDA call failed: CUDA_ERROR_OUT_OF_MEMORY (out of memory)"
        )

    monkeypatch.setattr(aten_fast, "_t", lambda tensor: tensor)
    monkeypatch.setattr(aten_fast, "_spec_of", lambda tensor: tensor)
    monkeypatch.setattr(
        eager_kernels,
        "matmul_ops",
        SimpleNamespace(MatmulSpec=raise_allocator_oom),
        raising=False,
    )

    with pytest.raises(torch.OutOfMemoryError, match="CUDA_ERROR_OUT_OF_MEMORY"):
        aten_fast._try_spec_matmul("MatmulSpec", (lhs, rhs), 0)


def test_tf32_addmm_scalars_retain_existing_not_handled_contract(monkeypatch):
    from torch_mojo_backend.eager_kernels import aten_fast

    calls = []
    monkeypatch.setattr(
        aten_fast, "_try_bf16_gemm", lambda *args, **kwargs: calls.append("bf16")
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
    monkeypatch.setattr(aten_fast, "_try_bf16_linear", lambda *_args, **_kwargs: None)
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
    from torch_mojo_backend import eager_kernels
    from torch_mojo_backend.eager_kernels import aten_fast

    device = SimpleNamespace(label="gpu")
    next_ptr = iter(range(100, 200))

    def tensor(shape, ptr=None, dtype=None):
        shape = tuple(shape)
        return SimpleNamespace(
            _shape=shape,
            _strides=aten_fast._row_major_strides(shape),
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
    monkeypatch.setattr(aten_fast, "_try_bf16_bmm", lambda *_args, **_kwargs: None)
    monkeypatch.setattr(aten_fast, "_try_tf32_bmm", try_tf32)
    monkeypatch.setattr(aten_fast, "_ctx_ptr", lambda _device: 1234)
    monkeypatch.setattr(aten_fast, "fast_aten_native_dropout", native_dropout)
    monkeypatch.setattr(aten_fast, "fast_aten_mul", multiply)
    monkeypatch.setattr(aten_fast, "fast_filled", filled)
    monkeypatch.setitem(
        eager_kernels.__dict__,
        "matmul_ops",
        SimpleNamespace(Bmm=lambda *args: raw_calls.append(args)),
    )
    monkeypatch.setitem(
        eager_kernels.__dict__,
        "nn_ops",
        SimpleNamespace(SoftmaxRows=lambda *args: softmax_calls.append(args)),
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
    from torch_mojo_backend import eager_kernels
    from torch_mojo_backend.eager_kernels import aten_fast

    device = SimpleNamespace(id=7, label="gpu", api="cuda", architecture_name="sm_90a")
    m, n, k = 6, 7, 5

    def matrix(shape, transposed, base_ptr, offset):
        rows, cols = shape
        return SimpleNamespace(
            _shape=shape,
            _strides=(1, rows) if transposed else (cols, 1),
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
        _strides=(1,),
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
            _strides=aten_fast._row_major_strides(shape),
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
    monkeypatch.setitem(
        eager_kernels.__dict__,
        "bf16_matmul_ops",
        SimpleNamespace(Bf16GemmBF16=lambda *args: calls.append(args)),
    )

    try:
        out = aten_fast._try_bf16_gemm(
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
    from torch_mojo_backend import eager_kernels
    from torch_mojo_backend.eager_kernels import aten_fast

    device = SimpleNamespace(label="gpu", api="cuda", architecture_name="sm_90a")

    def tensor(shape, ptr):
        shape = tuple(shape)
        return SimpleNamespace(
            _shape=shape,
            _strides=aten_fast._row_major_strides(shape),
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
    monkeypatch.setitem(
        eager_kernels.__dict__,
        "bf16_matmul_ops",
        SimpleNamespace(Bf16GemmBF16=lambda *args: calls.append(args)),
    )

    assert aten_fast._try_bf16_gemm(lhs, rhs) is output
    assert len(calls) == 1
    assert calls[0][3] == output._ptr
    assert calls[0][9] == 0


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
            _strides=(
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
    monkeypatch.setattr(aten_fast, "_resolve_bf16_bridge", fail_late_path)
    monkeypatch.setattr(aten_fast, "_alloc", fail_late_path)
    monkeypatch.setattr(aten_fast, "_ctx_ptr", fail_late_path)

    assert aten_fast._try_bf16_gemm(lhs, rhs, bias, output_shape=output_shape) is None


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
        return {input: input_metadata, weight: weight_metadata}.get(value)

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
    monkeypatch.setattr(aten_fast, "_try_bf16_gemm", try_gemm)

    assert aten_fast._try_bf16_linear(input, weight, bias) is result
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
    from torch_mojo_backend import eager_kernels
    from torch_mojo_backend.eager_kernels import aten_fast

    device = SimpleNamespace(id=9, label="gpu", api="cuda", architecture_name="sm_90a")
    batch, m, n, k = 3, 7, 5, 9

    def batched(shape, transposed, gap, base_ptr, offset):
        batches, rows, cols = shape
        batch_stride = rows * cols + gap
        return (
            SimpleNamespace(
                _shape=shape,
                _strides=(batch_stride, 1, rows)
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
            _strides=aten_fast._row_major_strides(shape),
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
    monkeypatch.setitem(
        eager_kernels.__dict__,
        "bf16_matmul_ops",
        SimpleNamespace(Bf16BmmBF16=lambda *args: calls.append(args)),
    )

    out = aten_fast._try_bf16_bmm(lhs, rhs, transpose_b=transpose_b)

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
            _strides=(
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
    monkeypatch.setattr(aten_fast, "_resolve_bf16_bridge", fail_late_path)
    monkeypatch.setattr(aten_fast, "_alloc", fail_late_path)
    monkeypatch.setattr(aten_fast, "_ctx_ptr", fail_late_path)

    assert aten_fast._try_bf16_bmm(lhs, rhs) is None


@pytest.mark.parametrize("operation", ["gemm", "bmm"])
def test_bf16_bridge_error_propagates_without_retry(monkeypatch, operation):
    """A failed enqueue must not allocate or invoke the bridge a second time."""
    from torch_mojo_backend.eager_kernels import aten_fast

    device = SimpleNamespace(label="gpu", api="cuda", architecture_name="sm_90a")

    def tensor(shape, ptr):
        shape = tuple(shape)
        return SimpleNamespace(
            _shape=shape,
            _strides=aten_fast._row_major_strides(shape),
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

    def fail_bridge(*args):
        bridge_calls.append(args)
        raise RuntimeError("synthetic BF16 enqueue failure")

    monkeypatch.setattr(aten_fast, "_t", lambda value: value)
    monkeypatch.setattr(aten_fast, "_alloc", allocate)
    monkeypatch.setattr(aten_fast, "_ctx_ptr", lambda actual_device: 7007)
    monkeypatch.setattr(aten_fast, "_resolve_bf16_bridge", lambda _name: fail_bridge)

    with pytest.raises(RuntimeError, match="synthetic BF16 enqueue failure"):
        if operation == "gemm":
            aten_fast._try_bf16_gemm(tensor((6, 5), 1000), tensor((5, 7), 2000))
        else:
            aten_fast._try_bf16_bmm(tensor((2, 6, 5), 1000), tensor((2, 5, 7), 2000))

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
            _strides=aten_fast._row_major_strides(shape),
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
    monkeypatch.setattr(aten_fast, "_resolve_bf16_bridge", fail_late_path)
    monkeypatch.setattr(aten_fast, "_alloc", fail_late_path)
    monkeypatch.setattr(aten_fast, "_ctx_ptr", fail_late_path)

    if invalid_case == "gemm_rhs":
        result = aten_fast._try_bf16_gemm(
            tensor((3, 4), devices[0]), tensor((4, 5), devices[1])
        )
    elif invalid_case == "gemm_bias":
        result = aten_fast._try_bf16_gemm(
            tensor((3, 4), devices[0]),
            tensor((4, 5), devices[0]),
            tensor((5,), devices[1]),
        )
    else:
        result = aten_fast._try_bf16_bmm(
            tensor((2, 3, 4), devices[0]), tensor((2, 4, 5), devices[1])
        )
    assert result is None


def test_tf32_helpers_route_each_fake_device_context_and_reject_cross_device(
    monkeypatch,
):
    """Context selection is operand-local and never falls back to device zero."""
    from torch_mojo_backend import eager_kernels
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
            _strides=aten_fast._row_major_strides(shape),
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
    monkeypatch.setitem(
        eager_kernels.__dict__,
        "tf32_matmul_ops",
        SimpleNamespace(
            Tf32GemmF32=lambda *args: gemm_calls.append(args),
            Tf32BmmF32=lambda *args: bmm_calls.append(args),
        ),
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
    from torch_mojo_backend import eager_kernels
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

    monkeypatch.setitem(
        eager_kernels.__dict__, "tf32_matmul_ops", SimpleNamespace(Tf32GemmF32=record)
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
    from torch_mojo_backend import eager_kernels
    from torch_mojo_backend.eager_kernels import aten_fast

    calls = []
    monkeypatch.setitem(
        eager_kernels.__dict__,
        "tf32_matmul_ops",
        SimpleNamespace(Tf32GemmF32=lambda *args: calls.append(args)),
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
    from torch_mojo_backend import eager_kernels
    from torch_mojo_backend.eager_kernels import aten_fast

    calls = []
    monkeypatch.setitem(
        eager_kernels.__dict__,
        "tf32_matmul_ops",
        SimpleNamespace(Tf32GemmF32=lambda *args: calls.append(args)),
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
    from torch_mojo_backend import eager_kernels
    from torch_mojo_backend.eager_kernels import aten_fast

    calls = []
    monkeypatch.setitem(
        eager_kernels.__dict__,
        "tf32_matmul_ops",
        SimpleNamespace(Tf32GemmF32=lambda *args: calls.append(args)),
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
    from torch_mojo_backend import eager_kernels
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
    monkeypatch.setitem(
        eager_kernels.__dict__,
        "tf32_matmul_ops",
        SimpleNamespace(Tf32BmmF32=lambda *args: calls.append(args)),
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
    from torch_mojo_backend import eager_kernels
    from torch_mojo_backend.eager_kernels import aten_fast

    calls = []
    monkeypatch.setitem(
        eager_kernels.__dict__,
        "tf32_matmul_ops",
        SimpleNamespace(Tf32BmmF32=lambda *args: calls.append(args)),
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
    from torch_mojo_backend import eager_kernels
    from torch_mojo_backend.eager_kernels import aten_fast

    calls = []
    monkeypatch.setitem(
        eager_kernels.__dict__,
        "tf32_matmul_ops",
        SimpleNamespace(Tf32BmmF32=lambda *args: calls.append(args)),
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

    missing = [path.name for path in aten_fast._BF16_SOURCE_PATHS if not path.is_file()]
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


def _bf16_dense_batched_pair(generator, shape, transposed, gap, offset, mojo_h100):
    """Create stored-BF16 dense matrices separated by runtime batch padding."""
    from torch_mojo_backend.eager_kernels import aten_fast

    batch, rows, cols = shape
    matrix_elements = rows * cols
    batch_stride = matrix_elements + gap
    storage_elements = offset + (batch - 1) * batch_stride + matrix_elements + 4
    storage = torch.randn(storage_elements, generator=generator).to(torch.bfloat16)
    strides = (batch_stride, 1, rows) if transposed else (batch_stride, cols, 1)
    host = torch.as_strided(storage, shape, strides, offset)
    device = aten_fast._view_of(storage.to(mojo_h100), shape, strides, offset)
    return host, device


def _assert_bf16_fp32_accumulation_close(actual, expected):
    """Compare BF16 outputs after an explicit one-round FP32 oracle."""
    assert actual.dtype == expected.dtype == torch.bfloat16
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
    from torch_mojo_backend import eager_kernels
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

    monkeypatch.delitem(eager_kernels.__dict__, "bf16_matmul_ops", raising=False)
    monkeypatch.setattr(aten_fast, "_BF16_IMPORT_FAILED", False)
    monkeypatch.setattr(aten_fast, "_try_tf32_gemm", fail_later_route)
    monkeypatch.setattr(aten_fast, "_try_spec_matmul", fail_later_route)
    old_precision = torch.get_float32_matmul_precision()
    torch.set_float32_matmul_precision("highest")
    try:
        actual = torch.mm(mojo_lhs, mojo_rhs)
    finally:
        torch.set_float32_matmul_precision(old_precision)

    module = eager_kernels.__dict__.get("bf16_matmul_ops")
    assert getattr(module, "__name__", "") == (
        "torch_mojo_backend.eager_kernels.bf16_matmul_ops"
    )
    _assert_bf16_fp32_accumulation_close(actual.cpu(), expected)


@pytest.mark.parametrize("operation", ["mm", "addmm"])
@pytest.mark.parametrize("lhs_transposed", [False, True])
@pytest.mark.parametrize("rhs_transposed", [False, True])
def test_bf16_real_gemm_extension_handles_tails_offsets_and_all_layouts(
    mojo_h100, monkeypatch, operation, lhs_transposed, rhs_transposed
):
    """Real BF16 GEMM matches an FP32-accumulate, one-BF16-round oracle."""
    _require_real_bf16_gemm_sources()
    from torch_mojo_backend import eager_kernels
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

    monkeypatch.delitem(eager_kernels.__dict__, "bf16_matmul_ops", raising=False)
    monkeypatch.setattr(aten_fast, "_BF16_IMPORT_FAILED", False)
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

    module = eager_kernels.__dict__.get("bf16_matmul_ops")
    assert getattr(module, "__name__", "") == (
        "torch_mojo_backend.eager_kernels.bf16_matmul_ops"
    )
    _assert_bf16_fp32_accumulation_close(actual.cpu(), expected)


@pytest.mark.parametrize("lhs_transposed", [False, True])
@pytest.mark.parametrize("rhs_transposed", [False, True])
@pytest.mark.parametrize("transpose_b", [False, True])
def test_bf16_real_bmm_extension_handles_all_layouts_offsets_and_padded_batches(
    mojo_h100, monkeypatch, lhs_transposed, rhs_transposed, transpose_b
):
    """Real BF16 BMM covers physical layouts and a logical RHS transpose."""
    _require_real_bf16_gemm_sources()
    from torch_mojo_backend import eager_kernels
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

    monkeypatch.delitem(eager_kernels.__dict__, "bf16_matmul_ops", raising=False)
    monkeypatch.setattr(aten_fast, "_BF16_IMPORT_FAILED", False)
    monkeypatch.setattr(aten_fast, "_try_tf32_bmm", fail_later_route)
    monkeypatch.setattr(aten_fast, "_try_spec_matmul", fail_later_route)
    if transpose_b:
        actual = aten_fast._fast_aten_bmm_transpose_b(mojo_lhs, mojo_rhs)
    else:
        actual = torch.bmm(mojo_lhs, mojo_rhs)

    module = eager_kernels.__dict__.get("bf16_matmul_ops")
    assert getattr(module, "__name__", "") == (
        "torch_mojo_backend.eager_kernels.bf16_matmul_ops"
    )
    _assert_bf16_fp32_accumulation_close(actual.cpu(), expected)


def test_bf16_real_linear_forward_backward_uses_three_gemm_routes(
    mojo_h100, monkeypatch
):
    """Linear forward, input-grad, and weight-grad all use real BF16 GEMM."""
    _require_real_bf16_gemm_sources()
    from torch_mojo_backend import eager_kernels
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
    original_try_bf16_gemm = aten_fast._try_bf16_gemm

    def record_bf16_gemm(*args, **kwargs):
        result = original_try_bf16_gemm(*args, **kwargs)
        assert result is not None
        gemm_calls.append((tuple(args[0]._shape), tuple(args[1]._shape), kwargs))
        return result

    def fail_later_route(*_args, **_kwargs):
        raise AssertionError("eligible BF16 linear GEMM reached TF32 or TensorSpec")

    monkeypatch.delitem(eager_kernels.__dict__, "bf16_matmul_ops", raising=False)
    monkeypatch.setattr(aten_fast, "_BF16_IMPORT_FAILED", False)
    monkeypatch.setattr(aten_fast, "_try_bf16_gemm", record_bf16_gemm)
    monkeypatch.setattr(aten_fast, "_try_tf32_gemm", fail_later_route)
    monkeypatch.setattr(aten_fast, "_try_spec_matmul", fail_later_route)
    actual_output = torch.nn.functional.linear(mojo_input, mojo_weight, mojo_bias)
    actual_output.backward(mojo_grad_output)

    assert gemm_calls == [
        ((10, 73), (67, 73), {"transpose_b": True, "output_shape": (2, 5, 67)}),
        ((10, 67), (67, 73), {}),
        ((67, 10), (10, 73), {}),
    ]
    module = eager_kernels.__dict__.get("bf16_matmul_ops")
    assert getattr(module, "__name__", "") == (
        "torch_mojo_backend.eager_kernels.bf16_matmul_ops"
    )
    _assert_bf16_fp32_accumulation_close(actual_output.cpu(), expected_output)
    for actual, expected in (
        (mojo_input.grad, expected_input_grad),
        (mojo_weight.grad, expected_weight_grad),
        (mojo_bias.grad, expected_bias_grad),
    ):
        assert actual is not None
        _assert_bf16_fp32_accumulation_close(actual.cpu(), expected)


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
    from torch_mojo_backend import eager_kernels
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

    monkeypatch.delitem(eager_kernels.__dict__, "tf32_matmul_ops", raising=False)
    monkeypatch.setattr(aten_fast, "_TF32_IMPORT_FAILED", False)
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

    module = eager_kernels.__dict__.get("tf32_matmul_ops")
    assert getattr(module, "__name__", "") == (
        "torch_mojo_backend.eager_kernels.tf32_matmul_ops"
    )
    torch.testing.assert_close(actual.cpu(), expected, atol=5e-2, rtol=5e-2)


@pytest.mark.parametrize("lhs_transposed", [False, True])
@pytest.mark.parametrize("rhs_transposed", [False, True])
@pytest.mark.parametrize("transpose_b", [False, True])
def test_tf32_real_bmm_extension_handles_layouts_offsets_and_padded_batches(
    mojo_h100, monkeypatch, lhs_transposed, rhs_transposed, transpose_b
):
    """Real BMM covers every physical layout and the logical RHS transpose."""
    from torch_mojo_backend import eager_kernels
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

    monkeypatch.delitem(eager_kernels.__dict__, "tf32_matmul_ops", raising=False)
    monkeypatch.setattr(aten_fast, "_TF32_IMPORT_FAILED", False)
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

    module = eager_kernels.__dict__.get("tf32_matmul_ops")
    assert getattr(module, "__name__", "") == (
        "torch_mojo_backend.eager_kernels.tf32_matmul_ops"
    )
    torch.testing.assert_close(actual.cpu(), expected, atol=5e-2, rtol=5e-2)


def test_tf32_real_linear_forward_backward_uses_gemm_extension(mojo_h100, monkeypatch):
    """Linear forward, input-grad, and weight-grad all use the real TF32 GEMM."""
    from torch_mojo_backend import eager_kernels
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

    monkeypatch.delitem(eager_kernels.__dict__, "tf32_matmul_ops", raising=False)
    monkeypatch.setattr(aten_fast, "_TF32_IMPORT_FAILED", False)
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
    module = eager_kernels.__dict__.get("tf32_matmul_ops")
    assert getattr(module, "__name__", "") == (
        "torch_mojo_backend.eager_kernels.tf32_matmul_ops"
    )
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


def test_fa4_bf16_causal_gapped_qkv_forward_backward(mojo_h100, monkeypatch):
    """The cached H100 path stays dynamic for nanoGPT's gapped QKV views."""
    from torch_mojo_backend.eager_flash_attention import load_fa4_ops
    from torch_mojo_backend.mojo_device.mojo_device_aten_ops import EAGER_CALL_COUNTERS

    module = load_fa4_ops()
    calls = {"forward": 0, "backward": 0}
    original_forward = module.flash_attention_fwd_bf16_d64_causal_strided_qkv
    original_backward = module.flash_attention_bwd_bf16_d64_causal_strided_qkv

    def forward_spy(*args):
        calls["forward"] += 1
        return original_forward(*args)

    def backward_spy(*args):
        calls["backward"] += 1
        return original_backward(*args)

    monkeypatch.setattr(
        module, "flash_attention_fwd_bf16_d64_causal_strided_qkv", forward_spy
    )
    monkeypatch.setattr(
        module, "flash_attention_bwd_bf16_d64_causal_strided_qkv", backward_spy
    )

    head_dim = 64
    for seed, batch, seqlen, heads in ((20260718, 1, 128, 4), (20260719, 2, 256, 8)):
        width = heads * head_dim
        generator = torch.Generator().manual_seed(seed)
        fused_host = (
            torch.randn(
                batch, seqlen, 3 * width, generator=generator, dtype=torch.float32
            )
            * 0.25
        ).to(torch.bfloat16)
        grad_output = torch.randn(
            batch, heads, seqlen, head_dim, generator=generator, dtype=torch.float32
        ).to(torch.bfloat16)

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
        assert actual_output.dtype == torch.bfloat16
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


def test_fa4_direct_flash_aten_returns_real_logsumexp(mojo_h100):
    """The low-level flash op must not substitute zero backward metadata."""
    generator = torch.Generator().manual_seed(20260721)
    shape = (1, 4, 128, 64)
    query, key, value = (
        (torch.randn(shape, generator=generator) * 0.25).to(torch.bfloat16)
        for _ in range(3)
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
    assert debug.dtype == torch.bfloat16 and debug.numel() == 0

    grad_output = torch.randn(shape, generator=generator).to(torch.bfloat16)
    expected_output.backward(grad_output.float())
    output.backward(grad_output.to(mojo_h100))
    for actual, expected in zip(mojo_inputs, reference_inputs, strict=True):
        assert actual.grad is not None
        torch.testing.assert_close(
            actual.grad.cpu().float(), expected.grad, atol=5e-2, rtol=5e-2
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
    """Each single-input gradient skips unrelated saves and BMM branches."""
    from torch_mojo_backend.eager_kernels import aten_fast
    from torch_mojo_backend.mojo_device.torch_mojo_tensor import TorchMojoTensor

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
    original_transpose_b_bmm = aten_fast._fast_aten_bmm_transpose_b
    original_fused_backward = aten_fast.fast_sdpa_dropout_softmax_backward
    original_materialize = TorchMojoTensor._materialize_contiguous
    materialized_shapes = []

    def spy_bmm(*args):
        calls["bmm"] += 1
        return original_bmm(*args)

    def spy_transpose_b_bmm(*args):
        calls["transpose_b_bmm"] += 1
        return original_transpose_b_bmm(*args)

    def spy_fused_backward(*args):
        calls["fused_backward"] += 1
        return original_fused_backward(*args)

    def spy_materialize(self):
        materialized_shapes.append(tuple(self._shape))
        return original_materialize(self)

    monkeypatch.setattr(aten_fast, "fast_aten_bmm", spy_bmm)
    monkeypatch.setattr(aten_fast, "_fast_aten_bmm_transpose_b", spy_transpose_b_bmm)
    monkeypatch.setattr(
        aten_fast, "fast_sdpa_dropout_softmax_backward", spy_fused_backward
    )
    monkeypatch.setattr(TorchMojoTensor, "_materialize_contiguous", spy_materialize)

    actual_inputs = [
        tensor.to(mojo_gpu).requires_grad_(name == requires)
        for name, tensor in zip(names, host_inputs, strict=True)
    ]
    actual_output = torch.nn.functional.scaled_dot_product_attention(
        *actual_inputs, dropout_p=0.0, is_causal=True
    )
    assert actual_output.grad_fn.saved_names == expected_saved
    actual_output.backward(grad_output.to(mojo_gpu))

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


def test_fast_sdpa_saved_tensor_hooks_own_saved_allocations(mojo_gpu):
    """CPU pack results, not ctx payloads, own SDPA's saved activations."""
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
    from torch_mojo_backend import eager_kernels
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
    monkeypatch.setattr(
        eager_kernels,
        "sdpa_backward_ops",
        SimpleNamespace(SDPADropoutSoftmaxBackwardF32=lambda *args: calls.append(args)),
        raising=False,
    )

    out = aten_fast.fast_sdpa_dropout_softmax_backward(
        probabilities, grad, mask, 1.25, -0.5
    )

    assert out is not aten_fast.NOT_HANDLED
    assert tuple(out.shape) == shape
    assert len(calls) == 1
    args = calls[0]
    assert args[:4] == (out._ptr, probabilities._ptr, grad._ptr, mask._ptr)
    assert args[4:9] == (6, 5, 1, 1.25, -0.5)
    assert args[9] == aten_fast._ctx_ptr(probabilities._device)


def test_sdpa_fused_backward_materializes_strided_operands(mojo_gpu, monkeypatch):
    """The raw bridge sees dense temporary pointers, never strided metadata."""
    from torch_mojo_backend import eager_kernels
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
    monkeypatch.setattr(
        eager_kernels,
        "sdpa_backward_ops",
        SimpleNamespace(SDPADropoutSoftmaxBackwardF32=lambda *args: calls.append(args)),
        raising=False,
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
    assert args[4:9] == (10, 3, 1, 1.25, 0.125)
    assert args[9] == aten_fast._ctx_ptr(probabilities._device)


def test_sdpa_fused_backward_no_mask_ignores_dropout_scale(mojo_gpu, monkeypatch):
    from torch_mojo_backend import eager_kernels
    from torch_mojo_backend.eager_kernels import aten_fast

    calls = []
    monkeypatch.setattr(
        eager_kernels,
        "sdpa_backward_ops",
        SimpleNamespace(SDPADropoutSoftmaxBackwardF32=lambda *args: calls.append(args)),
        raising=False,
    )
    probabilities = torch.randn(3, 7).to(mojo_gpu)
    grad = torch.randn(3, 7).to(mojo_gpu)

    out = aten_fast.fast_sdpa_dropout_softmax_backward(
        probabilities, grad, None, object(), 0.0
    )

    assert out is not aten_fast.NOT_HANDLED
    assert len(calls) == 1
    assert calls[0][3:9] == (0, 3, 7, 0, 1.0, 0.0)


def test_sdpa_fused_backward_empty_skips_bridge(mojo_gpu, monkeypatch):
    from torch_mojo_backend import eager_kernels
    from torch_mojo_backend.eager_kernels import aten_fast

    calls = []
    monkeypatch.setattr(
        eager_kernels,
        "sdpa_backward_ops",
        SimpleNamespace(SDPADropoutSoftmaxBackwardF32=lambda *args: calls.append(args)),
        raising=False,
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
    torch.mojo.set_rng_state(_dropout_rng_state(seed, counter), query.device)

    output = torch.nn.functional.scaled_dot_product_attention(
        query, key, value, dropout_p=dropout_p, is_causal=True
    )
    probability_elements = batch * heads * query_length * key_length
    expected_increment = (probability_elements + 3) // 4 if 0.0 < dropout_p < 1.0 else 0
    assert _decode_dropout_rng_state(torch.mojo.get_rng_state(query.device)) == (
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

    del output

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
