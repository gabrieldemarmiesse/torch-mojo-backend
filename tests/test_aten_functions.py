import math
from collections.abc import Callable

import pytest
import torch
import torch.nn.functional
from torch._dynamo import mark_dynamic
from torch._dynamo.exc import BackendCompilerFailed
from torch.ops import aten

from torch_mojo_backend import aten_functions, mojo_backend, register_mojo_devices
from torch_mojo_backend.testing import (
    CallChecker,
    Conf,
    check_functions_are_equivalent,
    check_outputs,
)


@pytest.mark.parametrize("dtype", [torch.bfloat16, torch.float16])
def test_scaled_dot_product_flash_attention_basic(
    conf: Conf, dtype: torch.dtype, call_checker: CallChecker
):
    """Half-precision scaled dot product attention through the public API.

    The op the backend serves this with depends on the device: the compile path
    lowers to ``aten::_scaled_dot_product_flash_attention``, while mojo eager
    keeps the high-level ``aten::scaled_dot_product_attention`` and only reaches
    a flash kernel inside it on the architectures that have one (FA4 is CUDA
    sm_90a / BF16 / head_dim 64 only). Both twins are registered so the
    assertion is "a Mojo SDPA implementation ran", whichever variant the device
    routes to.
    """
    call_checker.register(
        aten_functions.aten__scaled_dot_product_flash_attention,
        aten_functions.aten_scaled_dot_product_attention,
    )

    def fn(q, k, v):
        return torch.nn.functional.scaled_dot_product_attention(
            q, k, v, dropout_p=0.0, is_causal=False
        )

    batch_size, num_heads, seq_len, head_dim = 2, 4, 8, 16
    q = torch.randn(batch_size, num_heads, seq_len, head_dim, dtype=dtype)
    k = torch.randn(batch_size, num_heads, seq_len, head_dim, dtype=dtype)
    v = torch.randn(batch_size, num_heads, seq_len, head_dim, dtype=dtype)

    # TensorFloat-32 tensor cores are used by default, lowering precision
    check_outputs(fn, conf, [q, k, v], atol=1e-2, rtol=1e-2)


@pytest.mark.parametrize("dtype", [torch.bfloat16, torch.float16])
def test_scaled_dot_product_flash_attention_with_causal(conf: Conf, dtype: str):
    """Test _scaled_dot_product_flash_attention with causal masking"""
    # Flash attention only works on CUDA
    if conf.device != "cuda:0":
        pytest.skip("Flash attention is only supported on CUDA")

    def fn(q, k, v):
        return torch.ops.aten._scaled_dot_product_flash_attention(
            q, k, v, dropout_p=0.0, is_causal=True, return_debug_mask=False
        )[0]  # For the moment we support only training

    batch_size, num_heads, seq_len, head_dim = 1, 2, 4, 8
    q = torch.randn(batch_size, num_heads, seq_len, head_dim, dtype=dtype)
    k = torch.randn(batch_size, num_heads, seq_len, head_dim, dtype=dtype)
    v = torch.randn(batch_size, num_heads, seq_len, head_dim, dtype=dtype)

    # TensorFloat-32 tensor cores are used by default, lowering precision
    check_outputs(fn, conf, [q, k, v], atol=1e-2, rtol=1e-2)


@pytest.mark.parametrize("dtype", [torch.bfloat16, torch.float16])
def test_scaled_dot_product_flash_attention_with_scale(conf: Conf, dtype):
    """Test _scaled_dot_product_flash_attention with custom scale"""
    # Flash attention only works on CUDA
    if conf.device != "cuda:0":
        pytest.skip("Flash attention is only supported on CUDA")

    def fn(q, k, v):
        return torch.ops.aten._scaled_dot_product_flash_attention(
            q,
            k,
            v,
            dropout_p=0.0,
            is_causal=False,
            return_debug_mask=False,
            scale=0.125,
        )[0]  # For the moment we support only training

    batch_size, num_heads, seq_len, head_dim = 1, 1, 4, 8
    q = torch.randn(batch_size, num_heads, seq_len, head_dim, dtype=dtype)
    k = torch.randn(batch_size, num_heads, seq_len, head_dim, dtype=dtype)
    v = torch.randn(batch_size, num_heads, seq_len, head_dim, dtype=dtype)

    # TensorFloat-32 tensor cores are used by default, lowering precision
    check_outputs(fn, conf, [q, k, v], atol=1e-2, rtol=1e-2)


def test_scaled_dot_product_attention_math_additive_mask(
    conf: Conf, call_checker: CallChecker
):
    """Test aten._scaled_dot_product_attention_math with an additive float mask"""
    call_checker.register(aten_functions.aten__scaled_dot_product_attention_math)

    def fn(q, k, v, attn_mask):
        out, _ = torch.ops.aten._scaled_dot_product_attention_math(
            q, k, v, attn_mask=attn_mask
        )
        return out

    batch_size, num_heads, seq_len, head_dim = 2, 4, 6, 16
    q = torch.randn(batch_size, num_heads, seq_len, head_dim)
    k = torch.randn(batch_size, num_heads, seq_len, head_dim)
    v = torch.randn(batch_size, num_heads, seq_len, head_dim)

    attn_mask = torch.tril(torch.ones(seq_len, seq_len))
    attn_mask = attn_mask.masked_fill(attn_mask == 0, float("-inf"))
    attn_mask = attn_mask.masked_fill(attn_mask == 1, 0.0)

    check_outputs(fn, conf, [q, k, v, attn_mask], atol=1e-4, rtol=1e-4)


def test_scaled_dot_product_attention_math_broadcast_mask(
    conf: Conf, call_checker: CallChecker
):
    """Test aten._scaled_dot_product_attention_math with a broadcastable mask"""
    call_checker.register(aten_functions.aten__scaled_dot_product_attention_math)

    def fn(q, k, v, attn_mask):
        out, _ = torch.ops.aten._scaled_dot_product_attention_math(
            q, k, v, attn_mask=attn_mask
        )
        return out

    batch_size, num_heads, seq_len, head_dim = 2, 4, 6, 16
    q = torch.randn(batch_size, num_heads, seq_len, head_dim)
    k = torch.randn(batch_size, num_heads, seq_len, head_dim)
    v = torch.randn(batch_size, num_heads, seq_len, head_dim)

    # Per-head mask that broadcasts across batch: shape [num_heads, seq_len, seq_len]
    attn_mask = torch.randn(num_heads, seq_len, seq_len)

    check_outputs(fn, conf, [q, k, v, attn_mask], atol=1e-4, rtol=1e-4)


def test_scaled_dot_product_attention_math_no_mask(
    conf: Conf, call_checker: CallChecker
):
    """Test aten._scaled_dot_product_attention_math without attn_mask still works"""
    call_checker.register(aten_functions.aten__scaled_dot_product_attention_math)

    def fn(q, k, v):
        out, _ = torch.ops.aten._scaled_dot_product_attention_math(q, k, v)
        return out

    batch_size, num_heads, seq_len, head_dim = 2, 4, 6, 16
    q = torch.randn(batch_size, num_heads, seq_len, head_dim)
    k = torch.randn(batch_size, num_heads, seq_len, head_dim)
    v = torch.randn(batch_size, num_heads, seq_len, head_dim)

    check_outputs(fn, conf, [q, k, v], atol=1e-4, rtol=1e-4)


def test_scaled_dot_product_attention_no_mask(conf: Conf, call_checker: CallChecker):
    call_checker.register(aten_functions.aten_scaled_dot_product_attention)

    def fn(q, k, v):
        return torch.nn.functional.scaled_dot_product_attention(q, k, v, dropout_p=0.0)

    batch_size, num_heads, seq_len, head_dim = 2, 4, 6, 16
    q = torch.randn(batch_size, num_heads, seq_len, head_dim)
    k = torch.randn(batch_size, num_heads, seq_len, head_dim)
    v = torch.randn(batch_size, num_heads, seq_len, head_dim)
    check_outputs(fn, conf, [q, k, v], atol=1e-4, rtol=1e-4)


def test_scaled_dot_product_attention_float_mask(conf: Conf, call_checker: CallChecker):
    call_checker.register(aten_functions.aten_scaled_dot_product_attention)

    def fn(q, k, v, attn_mask):
        return torch.nn.functional.scaled_dot_product_attention(
            q, k, v, attn_mask=attn_mask, dropout_p=0.0
        )

    batch_size, num_heads, seq_len, head_dim = 2, 4, 6, 16
    q = torch.randn(batch_size, num_heads, seq_len, head_dim)
    k = torch.randn(batch_size, num_heads, seq_len, head_dim)
    v = torch.randn(batch_size, num_heads, seq_len, head_dim)
    attn_mask = torch.tril(torch.ones(seq_len, seq_len))
    attn_mask = attn_mask.masked_fill(attn_mask == 0, float("-inf")).masked_fill(
        attn_mask == 1, 0.0
    )
    check_outputs(fn, conf, [q, k, v, attn_mask], atol=1e-4, rtol=1e-4)


def test_scaled_dot_product_attention_bool_mask(conf: Conf, call_checker: CallChecker):
    call_checker.register(aten_functions.aten_scaled_dot_product_attention)

    def fn(q, k, v, attn_mask):
        return torch.nn.functional.scaled_dot_product_attention(
            q, k, v, attn_mask=attn_mask, dropout_p=0.0
        )

    batch_size, num_heads, seq_len, head_dim = 2, 4, 6, 16
    q = torch.randn(batch_size, num_heads, seq_len, head_dim)
    k = torch.randn(batch_size, num_heads, seq_len, head_dim)
    v = torch.randn(batch_size, num_heads, seq_len, head_dim)
    attn_mask = torch.tril(torch.ones(seq_len, seq_len, dtype=torch.bool))
    check_outputs(fn, conf, [q, k, v, attn_mask], atol=1e-4, rtol=1e-4)


@pytest.mark.parametrize("mask_batch", [1, 2])
def test_scaled_dot_product_attention_cross_attention_4d_mask(
    conf: Conf, call_checker: CallChecker, mask_batch: int
) -> None:
    """Cross attention (key length != query length) with a rank-4 mask.

    ``mask_batch=1`` additionally exercises a mask that broadcasts over the
    batch dimension of the score matrix rather than matching it exactly.
    """
    call_checker.register(aten_functions.aten_scaled_dot_product_attention)

    def fn(q, k, v, attn_mask):
        return torch.nn.functional.scaled_dot_product_attention(
            q, k, v, attn_mask=attn_mask, dropout_p=0.0
        )

    batch_size, num_heads, q_len, kv_len, head_dim = 2, 4, 6, 11, 16
    q = torch.randn(batch_size, num_heads, q_len, head_dim)
    k = torch.randn(batch_size, num_heads, kv_len, head_dim)
    v = torch.randn(batch_size, num_heads, kv_len, head_dim)
    attn_mask = torch.randn(mask_batch, num_heads, q_len, kv_len)
    check_outputs(fn, conf, [q, k, v, attn_mask], atol=1e-4, rtol=1e-4)


@pytest.mark.parametrize("dtype", [torch.float32, torch.float64])
def test_native_batch_norm_legit_no_training_basic(device: str, dtype: torch.dtype):
    """Test basic batch normalization inference with different dtypes"""

    def fn(input_tensor, weight, bias, running_mean, running_var):
        outputs = aten._native_batch_norm_legit_no_training.default(
            input_tensor, weight, bias, running_mean, running_var, 0.1, 1e-5
        )
        # We don't support returning the saved mean and variance yet.
        # It's not sure we'll ever support returning those, notably because of
        # https://github.com/pytorch/pytorch/issues/85960
        return outputs[0]

    # Create test tensors
    batch_size, channels, height, width = 2, 3, 4, 4
    input_tensor = torch.randn(
        batch_size, channels, height, width, dtype=dtype, device=device
    )
    weight = torch.randn(channels, dtype=dtype, device=device)
    bias = torch.randn(channels, dtype=dtype, device=device)
    running_mean = torch.randn(channels, dtype=dtype, device=device)
    running_var = torch.abs(torch.randn(channels, dtype=dtype, device=device)) + 1e-5
    check_functions_are_equivalent(
        fn, device, [input_tensor, weight, bias, running_mean, running_var]
    )


@pytest.mark.parametrize("channels", [1, 4, 16])
def test_native_batch_norm_legit_no_training_different_channels(
    device: str, channels: int
):
    """Test batch norm with different numbers of channels"""

    def fn(input_tensor, weight, bias, running_mean, running_var):
        output = aten._native_batch_norm_legit_no_training.default(
            input_tensor, weight, bias, running_mean, running_var, 0.1, 1e-5
        )
        # We don't support returning the saved mean and variance yet.
        # It's not sure we'll ever support returning those, notably because of
        # https://github.com/pytorch/pytorch/issues/85960
        return output[0]

    # Create test tensors with varying channel dimensions
    batch_size, height, width = 2, 8, 8
    input_tensor = torch.randn(batch_size, channels, height, width, device=device)
    weight = torch.randn(channels, device=device)
    bias = torch.randn(channels, device=device)
    running_mean = torch.randn(channels, device=device)
    running_var = torch.abs(torch.randn(channels, device=device)) + 1e-5

    # Test that compilation works and outputs match
    check_functions_are_equivalent(
        fn, device, [input_tensor, weight, bias, running_mean, running_var]
    )


def test_native_batch_norm_legit_no_training_none_weight_bias(device: str):
    """Test batch norm with None weight and bias"""

    def fn(input_tensor, running_mean, running_var):
        output = aten._native_batch_norm_legit_no_training.default(
            input_tensor, None, None, running_mean, running_var, 0.1, 1e-5
        )
        # We don't support returning the saved mean and variance yet.
        # It's not sure we'll ever support returning those, notably because of
        # https://github.com/pytorch/pytorch/issues/85960
        return output[0]

    # Create test tensors
    batch_size, channels, height, width = 2, 3, 4, 4
    input_tensor = torch.randn(batch_size, channels, height, width, device=device)
    running_mean = torch.randn(channels, device=device)
    running_var = torch.abs(torch.randn(channels, device=device)) + 1e-5

    # Test that compilation works and outputs match
    check_functions_are_equivalent(
        fn, device, [input_tensor, running_mean, running_var]
    )


@pytest.mark.parametrize("eps", [1e-5, 1e-3])
def test_native_batch_norm_legit_no_training_different_eps(device: str, eps: float):
    """Test batch norm with different epsilon values"""

    def fn(input_tensor, weight, bias, running_mean, running_var):
        output = aten._native_batch_norm_legit_no_training.default(
            input_tensor, weight, bias, running_mean, running_var, 0.1, eps
        )
        # We don't support returning the saved mean and variance yet.
        # It's not sure we'll ever support returning those, notably because of
        # https://github.com/pytorch/pytorch/issues/85960
        return output[0]

    # Create test tensors
    batch_size, channels, height, width = 2, 3, 4, 4
    input_tensor = torch.randn(batch_size, channels, height, width, device=device)
    weight = torch.randn(channels, device=device)
    bias = torch.randn(channels, device=device)
    running_mean = torch.randn(channels, device=device)
    running_var = torch.abs(torch.randn(channels, device=device)) + eps * 10

    # Test that compilation works and outputs match
    check_functions_are_equivalent(
        fn, device, [input_tensor, weight, bias, running_mean, running_var]
    )


def test_native_batch_norm_legit_no_training_2d_input(device: str):
    """Test batch norm with 2D input (N, C)"""

    def fn(input_tensor, weight, bias, running_mean, running_var):
        output = aten._native_batch_norm_legit_no_training.default(
            input_tensor, weight, bias, running_mean, running_var, 0.1, 1e-5
        )
        # We don't support returning the saved mean and variance yet.
        # It's not sure we'll ever support returning those, notably because of
        # https://github.com/pytorch/pytorch/issues/85960
        return output[0]

    # Create 2D test tensors (batch_size, channels)
    batch_size, channels = 10, 5
    input_tensor = torch.randn(batch_size, channels, device=device)
    weight = torch.randn(channels, device=device)
    bias = torch.randn(channels, device=device)
    running_mean = torch.randn(channels, device=device)
    running_var = torch.abs(torch.randn(channels, device=device)) + 1e-5

    # Test that compilation works and outputs match
    check_functions_are_equivalent(
        fn, device, [input_tensor, weight, bias, running_mean, running_var]
    )


def test_aten_native_batch_norm_inference(conf: Conf, call_checker: CallChecker):
    """Test aten.native_batch_norm in inference mode (training=False)."""
    from torch_mojo_backend.eager_kernels import aten_fast

    call_checker.register(
        aten_functions.aten_native_batch_norm, aten_fast.fast_aten_native_batch_norm
    )

    def fn(input_tensor, weight, bias, running_mean, running_var):
        return aten.native_batch_norm(
            input_tensor, weight, bias, running_mean, running_var, False, 0.1, 1e-5
        )[0]

    batch_size, channels, height, width = 2, 3, 4, 4
    input_tensor = torch.randn(batch_size, channels, height, width)
    weight = torch.randn(channels)
    bias = torch.randn(channels)
    running_mean = torch.randn(channels)
    running_var = torch.abs(torch.randn(channels)) + 1e-5

    check_outputs(fn, conf, [input_tensor, weight, bias, running_mean, running_var])


def test_aten_native_batch_norm_training(conf: Conf, call_checker: CallChecker):
    """Test aten.native_batch_norm in training mode (training=True, batch stats)."""
    call_checker.register(aten_functions.aten_native_batch_norm)

    def fn(input_tensor, weight, bias, running_mean, running_var):
        return aten.native_batch_norm(
            input_tensor, weight, bias, running_mean, running_var, True, 0.1, 1e-5
        )[0]

    batch_size, channels, height, width = 4, 3, 5, 5
    input_tensor = torch.randn(batch_size, channels, height, width)
    weight = torch.randn(channels)
    bias = torch.randn(channels)
    running_mean = torch.randn(channels)
    running_var = torch.abs(torch.randn(channels)) + 1e-5

    check_outputs(fn, conf, [input_tensor, weight, bias, running_mean, running_var])


@pytest.mark.parametrize("dtype", [torch.float32, torch.float64])
def test_aten_native_layer_norm_basic(
    conf: Conf, dtype: torch.dtype, call_checker: CallChecker
):
    """Test aten.native_layer_norm returns (output, mean, rstd)"""
    from torch_mojo_backend.eager_kernels import aten_fast

    call_checker.register(
        aten_functions.aten_native_layer_norm, aten_fast.fast_aten_native_layer_norm
    )

    def fn(x, weight, bias):
        out, mean, rstd = aten.native_layer_norm(x, [10], weight, bias, 1e-5)
        return out, mean, rstd

    x = torch.randn(5, 10, dtype=dtype)
    weight = torch.randn(10, dtype=dtype)
    bias = torch.randn(10, dtype=dtype)
    check_outputs(fn, conf, [x, weight, bias])


def test_aten_native_layer_norm_none_weight_bias(conf: Conf, call_checker: CallChecker):
    """Test aten.native_layer_norm with None weight and bias"""
    call_checker.register(aten_functions.aten_native_layer_norm)

    def fn(x):
        out, mean, rstd = aten.native_layer_norm(x, [10], None, None, 1e-5)
        return out, mean, rstd

    x = torch.randn(5, 10)
    check_outputs(fn, conf, [x])


def test_aten_native_layer_norm_multidim(conf: Conf, call_checker: CallChecker):
    """Test aten.native_layer_norm with multi-dim normalized_shape"""
    call_checker.register(aten_functions.aten_native_layer_norm)

    def fn(x, weight, bias):
        out, mean, rstd = aten.native_layer_norm(x, [3, 4], weight, bias, 1e-5)
        return out, mean, rstd

    x = torch.randn(2, 5, 3, 4)
    weight = torch.randn(3, 4)
    bias = torch.randn(3, 4)
    check_outputs(fn, conf, [x, weight, bias])


@pytest.mark.parametrize("eps", [1e-5, 1e-3])
def test_aten_native_layer_norm_different_eps(
    conf: Conf, eps: float, call_checker: CallChecker
):
    """Test aten.native_layer_norm with different epsilon values"""
    from torch_mojo_backend.eager_kernels import aten_fast

    call_checker.register(
        aten_functions.aten_native_layer_norm, aten_fast.fast_aten_native_layer_norm
    )

    def fn(x, weight, bias):
        out, mean, rstd = aten.native_layer_norm(x, [10], weight, bias, eps)
        return out, mean, rstd

    x = torch.randn(5, 10)
    weight = torch.randn(10)
    bias = torch.randn(10)
    check_outputs(fn, conf, [x, weight, bias])


def test_aten_normal_default(conf: Conf, call_checker: CallChecker):
    """Test aten.normal_ with default mean=0 and std=1"""
    call_checker.register(aten_functions.aten_normal_)

    x = torch.zeros(1000, dtype=torch.float32).to(conf.device)
    aten.normal_(x)

    assert x.shape == (1000,)
    assert x.dtype == torch.float32
    assert x.device == torch.device(conf.device)
    x_cpu = x.to("cpu")
    assert abs(x_cpu.mean().item()) < 0.2
    assert abs(x_cpu.std().item() - 1.0) < 0.2


@pytest.mark.parametrize("mean,std", [(2.0, 3.0), (-1.0, 0.5)])
def test_aten_normal_custom_params(
    conf: Conf, call_checker: CallChecker, mean: float, std: float
):
    """Test aten.normal_ with custom mean and std"""
    call_checker.register(aten_functions.aten_normal_)

    x = torch.zeros(1000, dtype=torch.float32).to(conf.device)
    aten.normal_(x, mean, std)

    assert x.shape == (1000,)
    assert x.dtype == torch.float32
    assert x.device == torch.device(conf.device)
    x_cpu = x.to("cpu")
    assert abs(x_cpu.mean().item() - mean) < 0.3 * std + 0.3
    assert abs(x_cpu.std().item() - std) < 0.3 * std + 0.3


def test_aten_normal_multidim(conf: Conf, call_checker: CallChecker):
    """Test aten.normal_ with a multi-dimensional tensor"""
    call_checker.register(aten_functions.aten_normal_)

    x = torch.zeros(10, 20, 5, dtype=torch.float32).to(conf.device)
    aten.normal_(x)

    assert x.shape == (10, 20, 5)
    assert x.dtype == torch.float32
    assert x.device == torch.device(conf.device)
    x_cpu = x.to("cpu")
    assert abs(x_cpu.mean().item()) < 0.2
    assert abs(x_cpu.std().item() - 1.0) < 0.2


@pytest.mark.parametrize("dim,keepdim", [(0, False), (1, True), ([0, 1], False)])
def test_aten_mean_out(conf: Conf, call_checker: CallChecker, dim, keepdim: bool):
    call_checker.register(aten_functions.aten_mean_out)

    def fn(x):
        out = x.new_empty(x.mean(dim=dim, keepdim=keepdim).shape)
        return aten.mean(x, dim=dim, keepdim=keepdim, out=out)

    x = torch.randn(3, 4, 5)
    check_outputs(fn, conf, [x])


@pytest.mark.parametrize("dtype", [torch.float32, torch.float16, torch.bfloat16])
def test_aten_empty_like(conf: Conf, call_checker: CallChecker, dtype: torch.dtype):
    if conf.device == "cpu" and dtype == torch.float16:
        pytest.skip("float16 not supported on CPU in MAX")
    call_checker.register(aten_functions.aten_empty_like)

    x = torch.randn(3, 4, dtype=dtype).to(conf.device)
    result = aten.empty_like(x)
    call_checker.check_was_called()
    assert result.shape == x.shape
    assert result.dtype == x.dtype
    assert result.device == torch.device(conf.device)


@pytest.mark.parametrize("target_dtype", [torch.float32, torch.bfloat16])
def test_aten_empty_like_different_dtype(
    conf: Conf, call_checker: CallChecker, target_dtype: torch.dtype
):
    call_checker.register(aten_functions.aten_empty_like)

    x = torch.randn(2, 5, dtype=torch.float32).to(conf.device)
    result = aten.empty_like(x, dtype=target_dtype)
    call_checker.check_was_called()
    assert result.shape == x.shape
    assert result.dtype == target_dtype
    assert result.device == torch.device(conf.device)


@pytest.mark.parametrize("dtype", [torch.float32, torch.float16, torch.bfloat16])
def test_aten_ones_like(conf: Conf, call_checker: CallChecker, dtype: torch.dtype):
    if conf.device == "cpu" and dtype == torch.float16:
        pytest.skip("float16 not supported on CPU in MAX")
    call_checker.register(aten_functions.aten_ones_like)

    def fn(x):
        return aten.ones_like(x)

    x = torch.randn(3, 4, dtype=dtype)
    check_outputs(fn, conf, [x])


@pytest.mark.parametrize("shape", [(1,), (4, 5), (2, 3, 7)])
def test_aten_ones_like_shape(conf: Conf, call_checker: CallChecker, shape: tuple):
    call_checker.register(aten_functions.aten_ones_like)

    def fn(x):
        return aten.ones_like(x)

    x = torch.randn(*shape)
    check_outputs(fn, conf, [x])


@pytest.mark.parametrize("dtype", [torch.float32, torch.bfloat16])
def test_aten_acos_basic(conf: Conf, dtype: torch.dtype):
    """Test aten.acos basic functionality with values in valid domain [-1, 1]"""
    # Skip float16 on CPU as MAX doesn't support f16 on CPU
    if conf.device == "cpu" and dtype == torch.float16:
        pytest.skip("float16 not supported on CPU in MAX")

    def fn(x):
        return aten.acos(x)

    # Test with values in valid domain [-1, 1]
    x = torch.tensor([-1.0, -0.5, 0.0, 0.5, 1.0, 0.8], dtype=dtype)
    check_outputs(fn, conf, [x])


@pytest.mark.parametrize("dtype", [torch.float32, torch.float64])
def test_aten_acos_special_values(conf: Conf, dtype: torch.dtype):
    """Test aten.acos with special mathematical values"""

    if dtype == torch.float64:
        pytest.xfail(
            "Bug: could not find LLVM intrinsic: 'llvm.nvvm.sqrt.approx.d', see https://github.com/modular/modular/issues/6434"
        )

    def fn(x):
        return aten.acos(x)

    # Test known mathematical values
    # acos(1.0) = 0.0
    # acos(0.0) = π/2 ≈ 1.5708
    # acos(-1.0) = π ≈ 3.1416
    x = torch.tensor([1.0, 0.0, -1.0], dtype=dtype)
    check_outputs(fn, conf, [x])


def test_aten_acos_2d_tensor(conf: Conf):
    """Test aten.acos with 2D tensor"""

    def fn(x):
        return aten.acos(x)

    x = torch.tensor([[-1.0, -0.5], [0.0, 0.5], [0.8, 1.0]], dtype=torch.float32)
    check_outputs(fn, conf, [x])


def test_aten_acos_3d_tensor(conf: Conf):
    """Test aten.acos with 3D tensor"""

    def fn(x):
        return aten.acos(x)

    # Random values in [-1, 1] range
    x = torch.rand(2, 3, 4, dtype=torch.float32) * 2 - 1
    check_outputs(fn, conf, [x])


def test_aten_acos_edge_domain_values(conf: Conf):
    """Test aten.acos with values near domain boundaries"""

    def fn(x):
        return aten.acos(x)

    # Test values very close to -1 and 1
    x = torch.tensor([-0.999, -0.99, 0.99, 0.999], dtype=torch.float32)
    check_outputs(fn, conf, [x])


def test_aten_acos_single_element(conf: Conf):
    """Test aten.acos with single element tensor"""

    def fn(x):
        return aten.acos(x)

    x = torch.tensor([0.5], dtype=torch.float32)
    check_outputs(fn, conf, [x])


@pytest.mark.parametrize("dtype", [torch.float32, torch.bfloat16])
def test_aten_amax_all_dims(conf: Conf, dtype: torch.dtype):
    """Test aten_amax with default empty dim list (reduces over all dimensions)"""
    # Skip float16 on CPU as MAX doesn't support f16 on CPU
    if conf.device == "cpu" and dtype == torch.float16:
        pytest.skip("float16 not supported on CPU in MAX")

    def fn(x):
        return aten.amax(x)

    # Test with different shapes
    x = torch.randn(3, 4, 5, dtype=dtype)
    check_outputs(fn, conf, [x])

    # Test with 1D tensor
    x1d = torch.randn(10, dtype=dtype)
    check_outputs(fn, conf, [x1d])


@pytest.mark.parametrize(
    "dtype", [torch.uint8, torch.int8, torch.int16, torch.int32, torch.int64]
)
def test_aten_bitwise_not(conf: Conf, dtype: torch.dtype):
    def fn(x):
        return aten.bitwise_not(x)

    # Create test tensors
    x = torch.randint(0, 10, (3, 4), dtype=dtype)

    check_outputs(fn, conf, [x])


def test_aten_bitwise_not_bool(conf: Conf):
    dtype = torch.bool

    def fn(x):
        return aten.bitwise_not(x)

    # Create test tensors
    x = torch.randint(0, 2, (3, 4), dtype=dtype)

    check_outputs(fn, conf, [x])


@pytest.mark.parametrize(
    "dtype", [torch.uint8, torch.int8, torch.int16, torch.int32, torch.int64]
)
def test_aten_bitwise_and_scalar(conf: Conf, dtype: torch.dtype):
    def fn(x):
        return aten.bitwise_and(x, 6)

    # Create test tensors
    x = torch.randint(0, 10, (3, 4), dtype=dtype)

    check_outputs(fn, conf, [x])


@pytest.mark.parametrize("bool_value", [True, False])
def test_aten_bitwise_and_scalar_bool(conf: Conf, bool_value: bool):
    dtype = torch.bool

    def fn(x):
        return aten.bitwise_and(x, bool_value)

    # Create test tensors
    x = torch.randint(0, 2, (3, 4), dtype=dtype)

    check_outputs(fn, conf, [x])


@pytest.mark.parametrize(
    "dtype", [torch.uint8, torch.int8, torch.int16, torch.int32, torch.int64]
)
def test_aten_bitwise_and(conf: Conf, dtype: torch.dtype):
    def fn(x, y):
        return aten.bitwise_and(x, y)

    # Create test tensors
    x = torch.randint(0, 10, (3, 4), dtype=dtype)
    y = torch.randint(0, 10, (3, 4), dtype=dtype)

    check_outputs(fn, conf, [x, y])


def test_aten_bitwise_and_bool(conf: Conf):
    dtype = torch.bool

    def fn(x, y):
        return aten.bitwise_and(x, y)

    # Create test tensors
    x = torch.randint(0, 2, (3, 4), dtype=dtype)
    y = torch.randint(0, 2, (3, 4), dtype=dtype)

    check_outputs(fn, conf, [x, y])


def test_aten_bitwise_and_broadcasting(conf: Conf):
    def fn(x, y):
        return aten.bitwise_and(x, y)

    # Create test tensors with broadcasting shapes
    x = torch.randint(0, 10, (3, 4, 5), dtype=torch.int32)
    y = torch.randint(0, 10, (5,), dtype=torch.int32)

    check_outputs(fn, conf, [x, y])


def test_aten_bitwise_and_broadcasting_ones(conf: Conf):
    def fn(x, y):
        return aten.bitwise_and(x, y)

    # Create test tensors with broadcasting shapes
    x = torch.randint(0, 100, (3, 1, 5), dtype=torch.int32)
    y = torch.randint(0, 100, (1, 4, 5), dtype=torch.int32)

    check_outputs(fn, conf, [x, y])


def test_aten_bitwise_and_broadcasting_ones_pad(conf: Conf):
    def fn(x, y):
        return aten.bitwise_and(x, y)

    # Create test tensors with broadcasting shapes
    x = torch.randint(0, 100, (8, 3, 1, 5), dtype=torch.int32)
    y = torch.randint(0, 100, (1, 4, 5), dtype=torch.int32)

    check_outputs(fn, conf, [x, y])


def test_aten_bitwise_and_broadcasting_ones_pad_dynamic_dim(conf: Conf):
    def fn(x, y):
        return aten.bitwise_and(x, y)

    # Create test tensors with broadcasting shapes
    x = torch.randint(0, 100, (8, 3, 1, 5), dtype=torch.int32)
    mark_dynamic(x, 0)
    mark_dynamic(x, 1)
    y = torch.randint(0, 100, (1, 4, 5), dtype=torch.int32)
    mark_dynamic(y, 1)

    check_outputs(fn, conf, [x, y])


# Tests for bitwise_or operations
@pytest.mark.parametrize(
    "dtype", [torch.uint8, torch.int8, torch.int16, torch.int32, torch.int64]
)
def test_aten_bitwise_or_scalar(conf: Conf, dtype: torch.dtype):
    def fn(x):
        return aten.bitwise_or(x, 6)

    # Create test tensors
    x = torch.randint(0, 10, (3, 4), dtype=dtype)

    check_outputs(fn, conf, [x])


@pytest.mark.parametrize("bool_value", [True, False])
def test_aten_bitwise_or_scalar_bool(conf: Conf, bool_value: bool):
    dtype = torch.bool

    def fn(x):
        return aten.bitwise_or(x, bool_value)

    # Create test tensors
    x = torch.randint(0, 2, (3, 4), dtype=dtype)

    check_outputs(fn, conf, [x])


@pytest.mark.parametrize(
    "dtype", [torch.uint8, torch.int8, torch.int16, torch.int32, torch.int64]
)
def test_aten_bitwise_or(conf: Conf, dtype: torch.dtype):
    def fn(x, y):
        return aten.bitwise_or(x, y)

    # Create test tensors
    x = torch.randint(0, 10, (3, 4), dtype=dtype)
    y = torch.randint(0, 10, (3, 4), dtype=dtype)

    check_outputs(fn, conf, [x, y])


def test_aten_bitwise_or_bool(conf: Conf):
    dtype = torch.bool

    def fn(x, y):
        return aten.bitwise_or(x, y)

    # Create test tensors
    x = torch.randint(0, 2, (3, 4), dtype=dtype)
    y = torch.randint(0, 2, (3, 4), dtype=dtype)

    check_outputs(fn, conf, [x, y])


def test_aten_bitwise_or_broadcasting(conf: Conf):
    def fn(x, y):
        return aten.bitwise_or(x, y)

    # Create test tensors with broadcasting shapes
    x = torch.randint(0, 10, (3, 4, 5), dtype=torch.int32)
    y = torch.randint(0, 10, (4, 5), dtype=torch.int32)

    check_outputs(fn, conf, [x, y])


# Tests for bitwise_xor operations
@pytest.mark.parametrize(
    "dtype", [torch.uint8, torch.int8, torch.int16, torch.int32, torch.int64]
)
def test_aten_bitwise_xor_scalar(conf: Conf, dtype: torch.dtype):
    def fn(x):
        return aten.bitwise_xor(x, 6)

    # Create test tensors
    x = torch.randint(0, 10, (3, 4), dtype=dtype)

    check_outputs(fn, conf, [x])


@pytest.mark.parametrize("bool_value", [True, False])
def test_aten_bitwise_xor_scalar_bool(conf: Conf, bool_value: bool):
    dtype = torch.bool

    def fn(x):
        return aten.bitwise_xor(x, bool_value)

    # Create test tensors
    x = torch.randint(0, 2, (3, 4), dtype=dtype)

    check_outputs(fn, conf, [x])


@pytest.mark.parametrize(
    "dtype", [torch.uint8, torch.int8, torch.int16, torch.int32, torch.int64]
)
def test_aten_bitwise_xor(conf: Conf, dtype: torch.dtype):
    def fn(x, y):
        return aten.bitwise_xor(x, y)

    # Create test tensors
    x = torch.randint(0, 10, (3, 4), dtype=dtype)
    y = torch.randint(0, 10, (3, 4), dtype=dtype)

    check_outputs(fn, conf, [x, y])


def test_aten_bitwise_xor_bool(conf: Conf):
    dtype = torch.bool

    def fn(x, y):
        return aten.bitwise_xor(x, y)

    # Create test tensors
    x = torch.randint(0, 2, (3, 4), dtype=dtype)
    y = torch.randint(0, 2, (3, 4), dtype=dtype)

    check_outputs(fn, conf, [x, y])


def test_aten_bitwise_xor_broadcasting(conf: Conf):
    def fn(x, y):
        return aten.bitwise_xor(x, y)

    # Create test tensors with broadcasting shapes
    x = torch.randint(0, 10, (3, 4, 5), dtype=torch.int32)
    y = torch.randint(0, 10, (4, 5), dtype=torch.int32)

    check_outputs(fn, conf, [x, y])


@pytest.mark.parametrize("dtype", [torch.float32, torch.float64])
def test_foreach_add_scalar(conf: Conf, dtype: torch.dtype):
    """Test _foreach_add.Scalar - adds scalar to each tensor in list"""

    def fn(x, y, z):
        tensors = [x, y, z]
        return aten._foreach_add.Scalar(tensors, 2.5)

    x = torch.randn(3, 4, dtype=dtype)
    y = torch.randn(2, 5, dtype=dtype)
    z = torch.randn(4, dtype=dtype)

    check_outputs(fn, conf, [x, y, z])


@pytest.mark.parametrize("dtype", [torch.float32, torch.float64])
def test_foreach_add_list(conf: Conf, dtype: torch.dtype):
    """Test _foreach_add.List - adds corresponding tensors with alpha scaling"""

    def fn(x1, y1, z1, x2, y2, z2):
        self_tensors = [x1, y1, z1]
        other_tensors = [x2, y2, z2]
        return aten._foreach_add.List(self_tensors, other_tensors, alpha=1.0)

    x1 = torch.randn(3, 4, dtype=dtype)
    y1 = torch.randn(2, 5, dtype=dtype)
    z1 = torch.randn(4, dtype=dtype)
    x2 = torch.randn(3, 4, dtype=dtype)
    y2 = torch.randn(2, 5, dtype=dtype)
    z2 = torch.randn(4, dtype=dtype)

    check_outputs(fn, conf, [x1, y1, z1, x2, y2, z2])


@pytest.mark.parametrize("alpha", [1.0, 2.0, -0.5])
def test_foreach_add_list_alpha(conf: Conf, alpha: float):
    """Test _foreach_add.List with different alpha values"""

    def fn(x1, y1, x2, y2):
        self_tensors = [x1, y1]
        other_tensors = [x2, y2]
        return aten._foreach_add.List(self_tensors, other_tensors, alpha=alpha)

    x1 = torch.randn(3, 4)
    y1 = torch.randn(2, 5)
    x2 = torch.randn(3, 4)
    y2 = torch.randn(2, 5)

    check_outputs(fn, conf, [x1, y1, x2, y2])


@pytest.mark.parametrize("dtype", [torch.float32, torch.float64])
def test_foreach_add_scalarlist(conf: Conf, dtype: torch.dtype):
    """Test _foreach_add.ScalarList - adds corresponding scalar to each tensor"""

    def fn(x, y, z):
        tensors = [x, y, z]
        scalars = [1.5, -2.0, 3.5]
        return aten._foreach_add.ScalarList(tensors, scalars)

    x = torch.randn(3, 4, dtype=dtype)
    y = torch.randn(2, 5, dtype=dtype)
    z = torch.randn(4, dtype=dtype)

    check_outputs(fn, conf, [x, y, z])


@pytest.mark.parametrize("dtype", [torch.float32, torch.float64])
def test_foreach_add_tensor(conf: Conf, dtype: torch.dtype):
    """Test _foreach_add.Tensor - broadcasts single 0-d tensor to all tensors in list"""

    def fn(x, y, z, other):
        tensors = [x, y, z]
        return aten._foreach_add.Tensor(tensors, other, alpha=1.0)

    x = torch.randn(3, 4, dtype=dtype)
    y = torch.randn(2, 5, dtype=dtype)
    z = torch.randn(4, dtype=dtype)
    other = torch.tensor(2.5, dtype=dtype)  # 0-d tensor

    check_outputs(fn, conf, [x, y, z, other])


@pytest.mark.parametrize("alpha", [1.0, 2.0, -0.5])
def test_foreach_add_tensor_alpha(conf: Conf, alpha: float):
    """Test _foreach_add.Tensor with different alpha values"""

    def fn(x, y, other):
        tensors = [x, y]
        return aten._foreach_add.Tensor(tensors, other, alpha=alpha)

    x = torch.randn(3, 4)
    y = torch.randn(2, 5)
    other = torch.tensor(1.5)  # 0-d tensor

    check_outputs(fn, conf, [x, y, other])


@pytest.mark.parametrize("dtype", [torch.float32, torch.float64])
def test_foreach_sub_scalar(conf: Conf, dtype: torch.dtype):
    """Test _foreach_sub.Scalar - subtracts scalar from each tensor in list"""

    def fn(x, y, z):
        tensors = [x, y, z]
        return aten._foreach_sub.Scalar(tensors, 2.5)

    x = torch.randn(3, 4, dtype=dtype)
    y = torch.randn(2, 5, dtype=dtype)
    z = torch.randn(4, dtype=dtype)

    check_outputs(fn, conf, [x, y, z])


@pytest.mark.parametrize("dtype", [torch.float32, torch.float64])
def test_foreach_sub_list(conf: Conf, dtype: torch.dtype):
    """Test _foreach_sub.List - subtracts corresponding tensors with alpha scaling"""

    def fn(x1, y1, z1, x2, y2, z2):
        self_tensors = [x1, y1, z1]
        other_tensors = [x2, y2, z2]
        return aten._foreach_sub.List(self_tensors, other_tensors, alpha=1.0)

    x1 = torch.randn(3, 4, dtype=dtype)
    y1 = torch.randn(2, 5, dtype=dtype)
    z1 = torch.randn(4, dtype=dtype)
    x2 = torch.randn(3, 4, dtype=dtype)
    y2 = torch.randn(2, 5, dtype=dtype)
    z2 = torch.randn(4, dtype=dtype)

    check_outputs(fn, conf, [x1, y1, z1, x2, y2, z2])


@pytest.mark.parametrize("alpha", [1.0, 2.0, -0.5])
def test_foreach_sub_list_alpha(conf: Conf, alpha: float):
    """Test _foreach_sub.List with different alpha values"""

    def fn(x1, y1, x2, y2):
        self_tensors = [x1, y1]
        other_tensors = [x2, y2]
        return aten._foreach_sub.List(self_tensors, other_tensors, alpha=alpha)

    x1 = torch.randn(3, 4)
    y1 = torch.randn(2, 5)
    x2 = torch.randn(3, 4)
    y2 = torch.randn(2, 5)

    check_outputs(fn, conf, [x1, y1, x2, y2])


@pytest.mark.parametrize("dtype", [torch.float32, torch.float64])
def test_foreach_sub_scalarlist(conf: Conf, dtype: torch.dtype):
    """Test _foreach_sub.ScalarList - subtracts corresponding scalar from each tensor"""

    def fn(x, y, z):
        tensors = [x, y, z]
        scalars = [1.5, -2.0, 3.5]
        return aten._foreach_sub.ScalarList(tensors, scalars)

    x = torch.randn(3, 4, dtype=dtype)
    y = torch.randn(2, 5, dtype=dtype)
    z = torch.randn(4, dtype=dtype)

    check_outputs(fn, conf, [x, y, z])


@pytest.mark.parametrize("dtype", [torch.float32, torch.float64])
def test_foreach_mul_scalar(conf: Conf, dtype: torch.dtype):
    """Test _foreach_mul.Scalar - multiplies each tensor in list by scalar"""

    def fn(x, y, z):
        tensors = [x, y, z]
        return aten._foreach_mul.Scalar(tensors, 2.5)

    x = torch.randn(3, 4, dtype=dtype)
    y = torch.randn(2, 5, dtype=dtype)
    z = torch.randn(4, dtype=dtype)

    check_outputs(fn, conf, [x, y, z])


@pytest.mark.parametrize("dtype", [torch.float32, torch.float64])
def test_foreach_mul_list(conf: Conf, dtype: torch.dtype):
    """Test _foreach_mul.List - multiplies corresponding tensors"""

    def fn(x1, y1, z1, x2, y2, z2):
        self_tensors = [x1, y1, z1]
        other_tensors = [x2, y2, z2]
        return aten._foreach_mul.List(self_tensors, other_tensors)

    x1 = torch.randn(3, 4, dtype=dtype)
    y1 = torch.randn(2, 5, dtype=dtype)
    z1 = torch.randn(4, dtype=dtype)
    x2 = torch.randn(3, 4, dtype=dtype)
    y2 = torch.randn(2, 5, dtype=dtype)
    z2 = torch.randn(4, dtype=dtype)

    check_outputs(fn, conf, [x1, y1, z1, x2, y2, z2])


@pytest.mark.parametrize("dtype", [torch.float32, torch.float64])
def test_foreach_mul_scalarlist(conf: Conf, dtype: torch.dtype):
    """Test _foreach_mul.ScalarList - multiplies each tensor by corresponding scalar"""

    def fn(x, y, z):
        tensors = [x, y, z]
        scalars = [1.5, -2.0, 3.5]
        return aten._foreach_mul.ScalarList(tensors, scalars)

    x = torch.randn(3, 4, dtype=dtype)
    y = torch.randn(2, 5, dtype=dtype)
    z = torch.randn(4, dtype=dtype)

    check_outputs(fn, conf, [x, y, z])


@pytest.mark.parametrize("dtype", [torch.float32, torch.float64])
def test_foreach_mul_tensor(conf: Conf, dtype: torch.dtype):
    """Test _foreach_mul.Tensor - broadcasts single 0-d tensor to all tensors in list"""

    def fn(x, y, z, other):
        tensors = [x, y, z]
        return aten._foreach_mul.Tensor(tensors, other)

    x = torch.randn(3, 4, dtype=dtype)
    y = torch.randn(2, 5, dtype=dtype)
    z = torch.randn(4, dtype=dtype)
    other = torch.tensor(2.5, dtype=dtype)  # 0-d tensor

    check_outputs(fn, conf, [x, y, z, other])


@pytest.mark.parametrize("dtype", [torch.float32, torch.float64])
def test_foreach_pow_scalar(conf: Conf, dtype: torch.dtype):
    """Test _foreach_pow.Scalar - raises each tensor in list to scalar power"""

    def fn(x, y, z):
        tensors = [x, y, z]
        return aten._foreach_pow.Scalar(tensors, 2.0)

    x = torch.randn(3, 4, dtype=dtype).abs() + 0.1
    y = torch.randn(2, 5, dtype=dtype).abs() + 0.1
    z = torch.randn(4, dtype=dtype).abs() + 0.1

    check_outputs(fn, conf, [x, y, z])


@pytest.mark.parametrize("dtype", [torch.float32, torch.float64])
def test_foreach_pow_list(conf: Conf, dtype: torch.dtype):
    """Test _foreach_pow.List - raises corresponding tensors to powers"""

    def fn(x1, y1, z1, x2, y2, z2):
        self_tensors = [x1, y1, z1]
        exponent_tensors = [x2, y2, z2]
        return aten._foreach_pow.List(self_tensors, exponent_tensors)

    x1 = torch.randn(3, 4, dtype=dtype).abs() + 0.1
    y1 = torch.randn(2, 5, dtype=dtype).abs() + 0.1
    z1 = torch.randn(4, dtype=dtype).abs() + 0.1
    x2 = torch.randn(3, 4, dtype=dtype)
    y2 = torch.randn(2, 5, dtype=dtype)
    z2 = torch.randn(4, dtype=dtype)

    check_outputs(fn, conf, [x1, y1, z1, x2, y2, z2])


@pytest.mark.parametrize("dtype", [torch.float32, torch.float64])
def test_foreach_pow_scalarlist(conf: Conf, dtype: torch.dtype):
    """Test _foreach_pow.ScalarList - raises each tensor to corresponding scalar power"""

    def fn(x, y, z):
        tensors = [x, y, z]
        exponents = [2.0, 3.0, 0.5]
        return aten._foreach_pow.ScalarList(tensors, exponents)

    x = torch.randn(3, 4, dtype=dtype).abs() + 0.1
    y = torch.randn(2, 5, dtype=dtype).abs() + 0.1
    z = torch.randn(4, dtype=dtype).abs() + 0.1

    check_outputs(fn, conf, [x, y, z])


@pytest.mark.parametrize("dtype", [torch.float32, torch.float64])
def test_foreach_pow_scalarandtensor(device: str, dtype: torch.dtype):
    """Test _foreach_pow.ScalarAndTensor - raises scalar to tensor powers"""

    def fn(x, y, z):
        exponent_tensors = [x, y, z]
        return aten._foreach_pow.ScalarAndTensor(2.0, exponent_tensors)

    x = torch.randn(3, 4, dtype=dtype, device=device)
    y = torch.randn(2, 5, dtype=dtype, device=device)
    z = torch.randn(4, dtype=dtype, device=device)

    check_functions_are_equivalent(fn, device, [x, y, z])


@pytest.mark.parametrize("dtype", [torch.float32, torch.float64])
def test_foreach_div_scalar(conf: Conf, dtype: torch.dtype):
    """Test _foreach_div.Scalar - divides each tensor in list by scalar"""

    def fn(x, y, z):
        tensors = [x, y, z]
        return aten._foreach_div.Scalar(tensors, 2.5)

    x = torch.randn(3, 4, dtype=dtype)
    y = torch.randn(2, 5, dtype=dtype)
    z = torch.randn(4, dtype=dtype)

    check_outputs(fn, conf, [x, y, z])


@pytest.mark.parametrize("dtype", [torch.float32, torch.float64])
def test_foreach_div_list(conf: Conf, dtype: torch.dtype):
    """Test _foreach_div.List - divides corresponding tensors"""

    def fn(x1, y1, z1, x2, y2, z2):
        self_tensors = [x1, y1, z1]
        other_tensors = [x2, y2, z2]
        return aten._foreach_div.List(self_tensors, other_tensors)

    x1 = torch.randn(3, 4, dtype=dtype)
    y1 = torch.randn(2, 5, dtype=dtype)
    z1 = torch.randn(4, dtype=dtype)
    x2 = torch.randn(3, 4, dtype=dtype) + 0.1
    y2 = torch.randn(2, 5, dtype=dtype) + 0.1
    z2 = torch.randn(4, dtype=dtype) + 0.1

    check_outputs(fn, conf, [x1, y1, z1, x2, y2, z2])


@pytest.mark.parametrize("dtype", [torch.float32, torch.float64])
def test_foreach_div_scalarlist(conf: Conf, dtype: torch.dtype):
    """Test _foreach_div.ScalarList - divides each tensor by corresponding scalar"""

    def fn(x, y, z):
        tensors = [x, y, z]
        scalars = [2.0, 3.0, 1.5]
        return aten._foreach_div.ScalarList(tensors, scalars)

    x = torch.randn(3, 4, dtype=dtype)
    y = torch.randn(2, 5, dtype=dtype)
    z = torch.randn(4, dtype=dtype)

    check_outputs(fn, conf, [x, y, z])


@pytest.mark.parametrize("dtype", [torch.float32, torch.float64])
def test_foreach_div_tensor(conf: Conf, dtype: torch.dtype):
    """Test _foreach_div.Tensor - broadcasts single 0-d tensor to all tensors in list"""

    def fn(x, y, z, other):
        tensors = [x, y, z]
        return aten._foreach_div.Tensor(tensors, other)

    x = torch.randn(3, 4, dtype=dtype)
    y = torch.randn(2, 5, dtype=dtype)
    z = torch.randn(4, dtype=dtype)
    other = torch.tensor(2.5, dtype=dtype)  # 0-d tensor

    check_outputs(fn, conf, [x, y, z, other])


def test_foreach_sqrt(conf: Conf):
    """Test _foreach_sqrt - computes square root of each tensor in list"""

    def fn(x, y, z):
        tensors = [x, y, z]
        return aten._foreach_sqrt(tensors)

    x = torch.randn(3, 4).abs() + 0.1
    y = torch.randn(2, 5).abs() + 0.1
    z = torch.randn(4).abs() + 0.1

    check_outputs(fn, conf, [x, y, z])


@pytest.mark.parametrize("dtype", [torch.float32, torch.float64])
def test_foreach_neg(conf: Conf, dtype: torch.dtype):
    """Test _foreach_neg - computes negation of each tensor in list"""

    def fn(x, y, z):
        tensors = [x, y, z]
        return aten._foreach_neg(tensors)

    x = torch.randn(3, 4, dtype=dtype)
    y = torch.randn(2, 5, dtype=dtype)
    z = torch.randn(4, dtype=dtype)

    check_outputs(fn, conf, [x, y, z])


@pytest.mark.parametrize("dtype", [torch.float32, torch.float64])
def test_foreach_reciprocal(conf: Conf, dtype: torch.dtype):
    """Test _foreach_reciprocal - computes reciprocal (1/x) of each tensor in list"""

    def fn(x, y, z):
        tensors = [x, y, z]
        return aten._foreach_reciprocal(tensors)

    # Add small offset to avoid division by zero
    x = torch.randn(3, 4, dtype=dtype) + 0.5
    y = torch.randn(2, 5, dtype=dtype) + 0.5
    z = torch.randn(4, dtype=dtype) + 0.5

    check_outputs(fn, conf, [x, y, z])


@pytest.mark.parametrize("dtype", [torch.float32, torch.float64])
def test_foreach_addcmul_scalar(conf: Conf, dtype: torch.dtype):
    """Test _foreach_addcmul.Scalar - adds element-wise product scaled by scalar"""

    def fn(x1, y1, z1, x2, y2, z2, x3, y3, z3):
        self_tensors = [x1, y1, z1]
        tensor1_list = [x2, y2, z2]
        tensor2_list = [x3, y3, z3]
        return aten._foreach_addcmul.Scalar(
            self_tensors, tensor1_list, tensor2_list, 2.0
        )

    x1 = torch.randn(3, 4, dtype=dtype)
    y1 = torch.randn(2, 5, dtype=dtype)
    z1 = torch.randn(4, dtype=dtype)
    x2 = torch.randn(3, 4, dtype=dtype)
    y2 = torch.randn(2, 5, dtype=dtype)
    z2 = torch.randn(4, dtype=dtype)
    x3 = torch.randn(3, 4, dtype=dtype)
    y3 = torch.randn(2, 5, dtype=dtype)
    z3 = torch.randn(4, dtype=dtype)

    check_outputs(fn, conf, [x1, y1, z1, x2, y2, z2, x3, y3, z3])


@pytest.mark.parametrize("dtype", [torch.float32, torch.float64])
def test_foreach_addcmul_scalarlist(conf: Conf, dtype: torch.dtype):
    """Test _foreach_addcmul.ScalarList - adds element-wise products scaled by corresponding scalars"""

    def fn(x1, y1, z1, x2, y2, z2, x3, y3, z3):
        self_tensors = [x1, y1, z1]
        tensor1_list = [x2, y2, z2]
        tensor2_list = [x3, y3, z3]
        scalars = [1.0, 2.0, 0.5]
        return aten._foreach_addcmul.ScalarList(
            self_tensors, tensor1_list, tensor2_list, scalars
        )

    x1 = torch.randn(3, 4, dtype=dtype)
    y1 = torch.randn(2, 5, dtype=dtype)
    z1 = torch.randn(4, dtype=dtype)
    x2 = torch.randn(3, 4, dtype=dtype)
    y2 = torch.randn(2, 5, dtype=dtype)
    z2 = torch.randn(4, dtype=dtype)
    x3 = torch.randn(3, 4, dtype=dtype)
    y3 = torch.randn(2, 5, dtype=dtype)
    z3 = torch.randn(4, dtype=dtype)

    check_outputs(fn, conf, [x1, y1, z1, x2, y2, z2, x3, y3, z3])


# NOTE: _foreach_addcmul.Tensor is NOT tested
# The .Tensor variant requires a 1-D CPU tensor with concrete values to extract scalars,
# which is incompatible with torch.compile's meta tensor tracing.
# See: https://github.com/pytorch/pytorch/issues/139795


@pytest.mark.parametrize("dtype", [torch.float32, torch.float64])
def test_foreach_addcdiv_scalar(conf: Conf, dtype: torch.dtype):
    """Test _foreach_addcdiv.Scalar - adds element-wise quotient scaled by scalar"""

    def fn(x1, y1, z1, x2, y2, z2, x3, y3, z3):
        self_tensors = [x1, y1, z1]
        tensor1_list = [x2, y2, z2]
        tensor2_list = [x3, y3, z3]
        return aten._foreach_addcdiv.Scalar(
            self_tensors, tensor1_list, tensor2_list, value=2.0
        )

    x1 = torch.randn(3, 4, dtype=dtype)
    y1 = torch.randn(2, 5, dtype=dtype)
    z1 = torch.randn(4, dtype=dtype)
    x2 = torch.randn(3, 4, dtype=dtype)
    y2 = torch.randn(2, 5, dtype=dtype)
    z2 = torch.randn(4, dtype=dtype)
    x3 = torch.randn(3, 4, dtype=dtype) + 0.5
    y3 = torch.randn(2, 5, dtype=dtype) + 0.5
    z3 = torch.randn(4, dtype=dtype) + 0.5

    check_outputs(fn, conf, [x1, y1, z1, x2, y2, z2, x3, y3, z3])


@pytest.mark.parametrize("dtype", [torch.float32, torch.float64])
def test_foreach_addcdiv_scalarlist(conf: Conf, dtype: torch.dtype):
    """Test _foreach_addcdiv.ScalarList - adds element-wise quotient scaled by scalar list"""

    def fn(x1, y1, z1, x2, y2, z2, x3, y3, z3):
        self_tensors = [x1, y1, z1]
        tensor1_list = [x2, y2, z2]
        tensor2_list = [x3, y3, z3]
        scalars = [1.0, 2.0, 0.5]
        return aten._foreach_addcdiv.ScalarList(
            self_tensors, tensor1_list, tensor2_list, scalars
        )

    x1 = torch.randn(3, 4, dtype=dtype)
    y1 = torch.randn(2, 5, dtype=dtype)
    z1 = torch.randn(4, dtype=dtype)
    x2 = torch.randn(3, 4, dtype=dtype)
    y2 = torch.randn(2, 5, dtype=dtype)
    z2 = torch.randn(4, dtype=dtype)
    x3 = torch.randn(3, 4, dtype=dtype) + 0.5
    y3 = torch.randn(2, 5, dtype=dtype) + 0.5
    z3 = torch.randn(4, dtype=dtype) + 0.5

    check_outputs(fn, conf, [x1, y1, z1, x2, y2, z2, x3, y3, z3])


# NOTE: _foreach_addcdiv.Tensor is NOT tested
# The .Tensor variant requires a 1-D CPU tensor with concrete values to extract scalars,
# which is incompatible with torch.compile's meta tensor tracing.
# See: https://github.com/pytorch/pytorch/issues/139795


def test_aten_masked_fill__inplace_scalar(conf: Conf):
    """Test aten.masked_fill_ with scalar values"""

    def fn(x, mask):
        aten.masked_fill_(x, mask, 5.0)
        # Modified in-place
        return x

    x = torch.randn(2, 3)
    mask = torch.tensor([[True, False, True], [False, True, False]])
    check_outputs(fn, conf, [x, mask])


def test_aten_masked_fill__inplace_tensor(conf: Conf):
    """Test aten.masked_fill_ with tensor values"""

    def fn(x, mask, value):
        aten.masked_fill_(x, mask, value)
        # Modified in-place
        return x

    x = torch.randn(2, 3)
    mask = torch.tensor([[True, False, True], [False, True, False]])
    value = torch.tensor(5.0)
    check_outputs(fn, conf, [x, mask, value])


@pytest.mark.parametrize("dtype", [torch.float32, torch.bfloat16])
def test_aten_ceil_basic(conf: Conf, dtype: torch.dtype):
    """Test aten.ceil basic functionality with floating point numbers"""
    # Skip float16 on CPU as MAX doesn't support f16 on CPU
    if conf.device == "cpu" and dtype == torch.float16:
        pytest.skip("float16 not supported on CPU in MAX")

    def fn(x):
        return aten.ceil(x)

    # Test with positive and negative floating point values
    x = torch.tensor([-2.7, -1.5, -1.0, -0.3, 0.0, 0.3, 1.0, 1.5, 2.7], dtype=dtype)
    check_outputs(fn, conf, [x])


@pytest.mark.parametrize("dtype", [torch.int32, torch.int64])
def test_aten_ceil_integer_types(conf: Conf, dtype: torch.dtype):
    """Test aten.ceil with integer types (should return copy)"""

    def fn(x):
        return aten.ceil(x)

    # Integer types should return a copy with no change
    x = torch.tensor([-5, -1, 0, 1, 5], dtype=dtype)
    check_outputs(fn, conf, [x])


def test_aten_ceil_2d_tensor(conf: Conf):
    """Test aten.ceil with 2D tensor"""

    def fn(x):
        return aten.ceil(x)

    x = torch.tensor([[-2.7, -1.3], [0.5, 1.8], [2.1, 3.9]], dtype=torch.float32)
    check_outputs(fn, conf, [x])


def test_aten_ceil_3d_tensor(conf: Conf):
    """Test aten.ceil with 3D tensor"""

    def fn(x):
        return aten.ceil(x)

    x = torch.randn(2, 3, 4, dtype=torch.float32)
    check_outputs(fn, conf, [x])


def test_aten_ceil_edge_cases(conf: Conf):
    """Test aten.ceil with edge cases"""

    def fn(x):
        return aten.ceil(x)

    # Test with already integer values, zero, and boundary cases
    x = torch.tensor([-3.0, -2.0, -1.0, 0.0, 1.0, 2.0, 3.0], dtype=torch.float32)
    check_outputs(fn, conf, [x])


def test_aten_ceil_large_values(conf: Conf):
    """Test aten.ceil with large floating point values"""

    def fn(x):
        return aten.ceil(x)

    # Test with large positive and negative values
    x = torch.tensor([-1000.1, -100.9, 100.1, 1000.9], dtype=torch.float32)
    check_outputs(fn, conf, [x])


def test_aten_ceil_small_fractional_values(conf: Conf):
    """Test aten.ceil with small fractional values"""

    def fn(x):
        return aten.ceil(x)

    # Test with small positive and negative fractional values
    x = torch.tensor([-0.001, -0.5, -0.999, 0.001, 0.5, 0.999], dtype=torch.float32)
    check_outputs(fn, conf, [x])


def test_aten_ceil_single_element(conf: Conf):
    """Test aten.ceil with single element tensor"""

    def fn(x):
        return aten.ceil(x)

    x = torch.tensor([2.3], dtype=torch.float32)
    check_outputs(fn, conf, [x])


def test_aten_ceil_scalar_tensor(conf: Conf):
    """Test aten.ceil with scalar tensor"""

    def fn(x):
        return aten.ceil(x)

    x = torch.tensor(2.7, dtype=torch.float32)
    check_outputs(fn, conf, [x])


@pytest.mark.parametrize("rounding_mode", ["floor", "trunc"])
@pytest.mark.parametrize(
    "dtype", [torch.float32, torch.float64, torch.int32, torch.int64]
)
def test_aten_div_rounding_mode_tensor(
    conf: Conf, call_checker: CallChecker, dtype: torch.dtype, rounding_mode: str
):
    """aten::div.Tensor_mode: floor/trunc rounding, incl. the sign
    combinations where the two modes disagree (only when the true quotient
    is negative and inexact)."""
    call_checker.register(aten_functions.aten_div)

    def fn(x, y):
        return torch.div(x, y, rounding_mode=rounding_mode)

    x = torch.tensor([7, -7, 7, -7, 8, -8, 6, -6], dtype=dtype)
    y = torch.tensor([2, 2, -2, -2, 3, 3, -3, -3], dtype=dtype)
    check_outputs(fn, conf, [x, y])


@pytest.mark.parametrize("rounding_mode", ["floor", "trunc"])
def test_aten_div_rounding_mode_scalar(
    conf: Conf, call_checker: CallChecker, rounding_mode: str
):
    """aten::div.Scalar_mode (a negative Python-scalar divisor)."""
    call_checker.register(aten_functions.aten_div)

    def fn(x):
        return torch.div(x, -3, rounding_mode=rounding_mode)

    x = torch.tensor([7, -7, 8, -8, 9, -9], dtype=torch.int64)
    check_outputs(fn, conf, [x])


def test_aten_div_rounding_mode_floor_trunc_disagree(
    conf: Conf, call_checker: CallChecker
):
    """Documents (and pins) the actual semantic difference: floor and trunc
    only disagree when the operands have opposite signs and the division is
    inexact -- a naive implementation that aliases one mode to the other
    would still pass same-sign cases."""
    call_checker.register(aten_functions.aten_div)

    def fn(x, y):
        return torch.div(x, y, rounding_mode="floor"), torch.div(
            x, y, rounding_mode="trunc"
        )

    x = torch.tensor([-7, 7], dtype=torch.int64)
    y = torch.tensor([2, -2], dtype=torch.int64)
    check_outputs(fn, conf, [x, y])


@pytest.mark.parametrize("dtype", [torch.float32, torch.float64, torch.bfloat16])
@pytest.mark.parametrize("approximate", ["none", "tanh"])
def test_aten_gelu_backward_basic(conf: Conf, dtype: torch.dtype, approximate: str):
    """Test aten.gelu_backward with different approximations"""
    # Skip float16 on CPU as MAX doesn't support f16 on CPU
    if conf.device == "cpu" and dtype == torch.float16:
        pytest.skip("float16 not supported on CPU in MAX")

    def fn(grad_output, x):
        return aten.gelu_backward(grad_output, x, approximate=approximate)

    # Test with varied input values covering negative, zero, and positive ranges
    x = torch.tensor([-3.0, -2.0, -1.0, -0.5, 0.0, 0.5, 1.0, 2.0, 3.0], dtype=dtype)
    grad_output = torch.ones_like(x)

    # bfloat16 requires higher tolerance due to lower precision
    if dtype == torch.bfloat16:
        check_outputs(fn, conf, [grad_output, x], atol=1e-2, rtol=5e-2)
    else:
        check_outputs(fn, conf, [grad_output, x])


@pytest.mark.parametrize("dtype", [torch.float32, torch.bfloat16])
def test_aten_gelu_backward_2d_tensor(conf: Conf, dtype: torch.dtype):
    """Test aten.gelu_backward with 2D tensor"""
    if conf.device == "cpu" and dtype == torch.float16:
        pytest.skip("float16 not supported on CPU in MAX")

    def fn(grad_output, x):
        return aten.gelu_backward(grad_output, x, approximate="none")

    x = torch.randn(3, 4, dtype=dtype)
    grad_output = torch.randn(3, 4, dtype=dtype)

    # bfloat16 requires higher tolerance due to lower precision
    if dtype == torch.bfloat16:
        check_outputs(fn, conf, [grad_output, x], atol=1e-2, rtol=5e-2)
    else:
        check_outputs(fn, conf, [grad_output, x])


@pytest.mark.parametrize("dtype", [torch.float32, torch.bfloat16])
def test_aten_gelu_backward_3d_tensor(conf: Conf, dtype: torch.dtype):
    """Test aten.gelu_backward with 3D tensor"""
    if conf.device == "cpu" and dtype == torch.float16:
        pytest.skip("float16 not supported on CPU in MAX")

    def fn(grad_output, x):
        return aten.gelu_backward(grad_output, x, approximate="none")

    x = torch.randn(2, 3, 4, dtype=dtype)
    grad_output = torch.randn(2, 3, 4, dtype=dtype)

    # bfloat16 requires higher tolerance due to lower precision
    if dtype == torch.bfloat16:
        check_outputs(fn, conf, [grad_output, x], atol=1e-2, rtol=5e-2)
    else:
        check_outputs(fn, conf, [grad_output, x])


def test_aten_gelu_backward_tanh_approx(conf: Conf):
    """Test aten.gelu_backward with tanh approximation"""

    def fn(grad_output, x):
        return aten.gelu_backward(grad_output, x, approximate="tanh")

    x = torch.randn(10, dtype=torch.float32)
    grad_output = torch.ones_like(x)
    check_outputs(fn, conf, [grad_output, x])


def test_aten_gelu_backward_edge_values(conf: Conf):
    """Test aten.gelu_backward with edge case values"""

    def fn(grad_output, x):
        return aten.gelu_backward(grad_output, x, approximate="none")

    # Test with large values, small values, and exact zeros
    x = torch.tensor([-10.0, -5.0, -1e-6, 0.0, 1e-6, 5.0, 10.0], dtype=torch.float32)
    grad_output = torch.ones_like(x)
    check_outputs(fn, conf, [grad_output, x])


def test_aten_gelu_backward_different_grad_outputs(conf: Conf):
    """Test aten.gelu_backward with non-uniform grad_output"""

    def fn(grad_output, x):
        return aten.gelu_backward(grad_output, x, approximate="none")

    x = torch.randn(8, dtype=torch.float32)
    # Test with varied gradient values
    grad_output = torch.tensor(
        [0.1, 0.5, 1.0, 2.0, -0.5, -1.0, 0.0, 3.0], dtype=torch.float32
    )
    check_outputs(fn, conf, [grad_output, x])


def test_aten_gelu_backward_scalar_tensor(conf: Conf):
    """Test aten.gelu_backward with scalar tensor"""

    def fn(grad_output, x):
        return aten.gelu_backward(grad_output, x, approximate="none")

    x = torch.tensor(1.5, dtype=torch.float32)
    grad_output = torch.tensor(1.0, dtype=torch.float32)
    check_outputs(fn, conf, [grad_output, x])


TRIGON_FUNCTIONS = [aten.asinh, aten.cosh, aten.sinh, aten.tan, aten.tanh]


@pytest.mark.parametrize("dtype", [torch.float32, torch.float64, torch.bfloat16])
@pytest.mark.parametrize("fn", TRIGON_FUNCTIONS)
def test_aten_trigon_basic(conf: Conf, fn: Callable, dtype: torch.dtype):
    """Test trigonometric functions basic functionality with floating point numbers"""

    if dtype == torch.float64 and fn in (aten.asinh, aten.tan):
        pytest.xfail(
            "could not find LLVM intrinsic: 'llvm.nvvm.sqrt.approx.d', see https://github.com/modular/modular/issues/6434"
        )

    # Test with positive, negative, and zero values
    # cosh(0) = 1, cosh is even function: cosh(-x) = cosh(x)
    x = torch.tensor([-2.0, -1.0, -0.5, 0.0, 0.5, 1.0, 2.0], dtype=dtype)
    check_outputs(fn, conf, [x])


@pytest.mark.parametrize("fn", TRIGON_FUNCTIONS)
def test_aten_trigon_2d_tensor(conf: Conf, fn: Callable):
    """Test trigonometric functions with 2D tensor"""

    x = torch.tensor([[-1.5, -0.5], [0.0, 1.0], [1.5, 2.5]], dtype=torch.float32)
    check_outputs(fn, conf, [x])


@pytest.mark.parametrize("fn", TRIGON_FUNCTIONS)
def test_aten_trigon_3d_tensor(conf: Conf, fn: Callable):
    """Test trigonometric functions with 3D tensor"""

    x = torch.randn(2, 3, 4, dtype=torch.float32)
    check_outputs(fn, conf, [x])


@pytest.mark.parametrize("fn", TRIGON_FUNCTIONS)
def test_aten_trigon_large_values(conf: Conf, fn: Callable):
    """Test trigonometric functions with large values (may approach infinity)"""

    # large values will produce large results due to exponential growth
    x = torch.tensor([-5.0, -3.0, 3.0, 5.0], dtype=torch.float32)
    check_outputs(fn, conf, [x])


@pytest.mark.parametrize("fn", TRIGON_FUNCTIONS)
def test_aten_trigon_small_values(conf: Conf, fn: Callable):
    """Test trigonometric functions with small values near zero"""

    # for small x, cosh(x) ≈ 1 + x²/2
    x = torch.tensor([-0.1, -0.01, -0.001, 0.001, 0.01, 0.1], dtype=torch.float32)
    check_outputs(fn, conf, [x])


@pytest.mark.parametrize("fn", TRIGON_FUNCTIONS)
def test_aten_trigon_single_element(conf: Conf, fn: Callable):
    """Test trigonometric functions with single element tensor"""

    x = torch.tensor([1.5], dtype=torch.float32)
    check_outputs(fn, conf, [x])


@pytest.mark.parametrize("fn", TRIGON_FUNCTIONS)
def test_aten_trigon_scalar_tensor(conf: Conf, fn: Callable):
    """Test trigonometric functions with scalar tensor"""

    x = torch.tensor(1.0, dtype=torch.float32)
    check_outputs(fn, conf, [x])


@pytest.mark.parametrize("dtype", [torch.float32, torch.bfloat16, torch.float16])
def test_aten_addr_basic(conf: Conf, dtype: torch.dtype, call_checker: CallChecker):
    """aten.addr: beta*self + alpha*outer(vec1, vec2).

    Regression test for the fused fast path added because ATen's own
    CompositeExplicitAutograd fallback for backends without a native addr
    kernel (`math_addr`) composes outer/scale/add in a different
    multiplication order than the native CPU/CUDA kernel, which drifted
    enough to fail OpInfo conformance for fp16/bf16 (up to 18% of sampled
    elements): conformance/test_opinfo.py::test_matches_cpu_addr_mojo_float16
    and ..._bfloat16. beta/alpha values below match the failing OpInfo
    sample exactly.
    """
    from torch_mojo_backend.mojo_device.mojo_device_aten_ops import EAGER_CALL_COUNTERS

    call_checker.register(EAGER_CALL_COUNTERS["aten::addr"])

    def fn(self, vec1, vec2):
        return aten.addr(self, vec1, vec2, beta=0.6, alpha=0.2)

    self_ = torch.randn(5, 10, dtype=dtype)
    vec1 = torch.randn(5, dtype=dtype)
    vec2 = torch.randn(10, dtype=dtype)
    check_outputs(fn, conf, [self_, vec1, vec2])
    call_checker.check_was_called()


@pytest.mark.parametrize("dtype", [torch.float32, torch.bfloat16, torch.float16])
def test_aten_addr_default_beta_alpha(
    conf: Conf, dtype: torch.dtype, call_checker: CallChecker
):
    """aten.addr with default beta=alpha=1 (self + outer(vec1, vec2))."""
    from torch_mojo_backend.mojo_device.mojo_device_aten_ops import EAGER_CALL_COUNTERS

    call_checker.register(EAGER_CALL_COUNTERS["aten::addr"])

    def fn(self, vec1, vec2):
        return aten.addr(self, vec1, vec2)

    self_ = torch.randn(4, 6, dtype=dtype)
    vec1 = torch.randn(4, dtype=dtype)
    vec2 = torch.randn(6, dtype=dtype)
    check_outputs(fn, conf, [self_, vec1, vec2])
    call_checker.check_was_called()


def test_aten_addr_beta_zero(conf: Conf, call_checker: CallChecker):
    """beta=0 must ignore `self` entirely, including nan/inf in it (matches
    ATen's own addr contract, see aten/src/ATen/native/LinearAlgebra.cpp)."""
    from torch_mojo_backend.mojo_device.mojo_device_aten_ops import EAGER_CALL_COUNTERS

    call_checker.register(EAGER_CALL_COUNTERS["aten::addr"])

    def fn(self, vec1, vec2):
        return aten.addr(self, vec1, vec2, beta=0.0, alpha=1.5)

    self_ = torch.full((3, 4), float("nan"), dtype=torch.float32)
    vec1 = torch.randn(3, dtype=torch.float32)
    vec2 = torch.randn(4, dtype=torch.float32)
    check_outputs(fn, conf, [self_, vec1, vec2])
    call_checker.check_was_called()


def test_aten_addr_self_broadcast(conf: Conf, call_checker: CallChecker):
    """`self` broadcastable to (len(vec1), len(vec2)) but not that exact
    shape (here: 0-d): the fast path's own right-alignment handles this
    directly (see `fast_aten_addr`), so this still goes through the fused
    kernel, not the composite fallback -- verified via call_checker."""
    from torch_mojo_backend.mojo_device.mojo_device_aten_ops import EAGER_CALL_COUNTERS

    call_checker.register(EAGER_CALL_COUNTERS["aten::addr"])

    def fn(self, vec1, vec2):
        return aten.addr(self, vec1, vec2, beta=0.5, alpha=2.0)

    self_ = torch.randn((), dtype=torch.float32)  # 0-d, broadcasts to (n, m)
    vec1 = torch.randn(3, dtype=torch.float32)
    vec2 = torch.randn(5, dtype=torch.float32)
    check_outputs(fn, conf, [self_, vec1, vec2])
    call_checker.check_was_called()


def test_aten_addr_dtype_mismatch_fallback(conf: Conf):
    """`self`, vec1, vec2 with different dtypes: `fast_aten_addr` requires
    them to already match (see its docstring) and declines, so this
    exercises the actual redispatch-to-ATen's-composite-fallback path in
    `mojo_device_addr` -- confirmed to return NOT_HANDLED for this input
    and to redispatch to a correct result, not just declared to."""

    def fn(self, vec1, vec2):
        return aten.addr(self, vec1, vec2, beta=0.5, alpha=2.0)

    self_ = torch.randn(3, 5, dtype=torch.float32)
    vec1 = torch.randn(3, dtype=torch.bfloat16)
    vec2 = torch.randn(5, dtype=torch.bfloat16)
    check_outputs(fn, conf, [self_, vec1, vec2], rtol=1e-2, atol=1e-2)


def test_aten_isin_basic(conf: Conf):
    """Test aten.isin basic functionality with 1D tensors"""

    def fn(elements, test_elements):
        return aten.isin(elements, test_elements, assume_unique=False, invert=False)

    elements = torch.tensor([1, 2, 3, 4, 5])
    test_elements = torch.tensor([2, 4])

    # Expected: elements 2 and 4 are in test_elements
    # Result should be [False, True, False, True, False]
    check_outputs(fn, conf, [elements, test_elements])


@pytest.mark.parametrize(
    "dtype", [torch.float32, torch.float64, torch.int32, torch.int64]
)
def test_aten_isin_different_dtypes(conf: Conf, dtype: torch.dtype):
    """Test aten.isin with different numeric data types"""

    def fn(elements, test_elements):
        return aten.isin(elements, test_elements, assume_unique=False, invert=False)

    if dtype in (torch.float32, torch.float64):
        elements = torch.tensor([1.0, 2.5, 3.0, 4.5], dtype=dtype)
        test_elements = torch.tensor([2.5, 4.5], dtype=dtype)
    else:
        elements = torch.tensor([1, 2, 3, 4], dtype=dtype)
        test_elements = torch.tensor([2, 4], dtype=dtype)

    check_outputs(fn, conf, [elements, test_elements])


def test_aten_isin_with_invert(conf: Conf):
    """Test aten.isin with invert=True"""

    def fn(elements, test_elements):
        return aten.isin(elements, test_elements, assume_unique=False, invert=True)

    elements = torch.tensor([1, 2, 3, 4, 5])
    test_elements = torch.tensor([2, 4])

    # With invert=True: return True for elements NOT in test_elements
    # Expected: [True, False, True, False, True]
    check_outputs(fn, conf, [elements, test_elements])


def test_aten_isin_empty_test_elements(conf: Conf):
    """Test aten.isin with empty test_elements tensor"""

    def fn(elements, test_elements):
        return aten.isin(elements, test_elements, assume_unique=False, invert=False)

    elements = torch.tensor([1, 2, 3, 4, 5])
    test_elements = torch.tensor([])

    # No elements should be found in empty test_elements
    # Expected: [False, False, False, False, False]
    check_outputs(fn, conf, [elements, test_elements])


def test_aten_isin_empty_elements(conf: Conf):
    """Test aten.isin with empty elements tensor"""

    def fn(elements, test_elements):
        return aten.isin(elements, test_elements, assume_unique=False, invert=False)

    elements = torch.tensor([])
    test_elements = torch.tensor([1, 2, 3])

    # Empty elements should return empty result
    check_outputs(fn, conf, [elements, test_elements])


def test_aten_isin_all_match(conf: Conf):
    """Test aten.isin when all elements are in test_elements"""

    def fn(elements, test_elements):
        return aten.isin(elements, test_elements, assume_unique=False, invert=False)

    elements = torch.tensor([1, 2, 3])
    test_elements = torch.tensor([1, 2, 3, 4, 5])

    # All elements should be found
    # Expected: [True, True, True]
    check_outputs(fn, conf, [elements, test_elements])


# ---------------------------------------------------------------------------
# The dim-indexed family: gather / index_select / scatter_add / index_add /
# index_put.  All of them are the same two kernels over an index space
# described by strides (GatherDim reads, ScatterDim[+add] writes), so the
# axes worth parametrizing are the ones that change that description: the
# rank, WHICH dim is indexed (0 has a dedicated row fast path, the last is
# the contiguous-inner regime, a negative dim exercises normalization), the
# payload dtype, and whether the operands are contiguous.
# ---------------------------------------------------------------------------

_INDEX_DTYPES = [torch.float32, torch.bfloat16, torch.int64]


def _index_payload(shape: tuple[int, ...], dtype: torch.dtype) -> torch.Tensor:
    """A tensor of `dtype` holding exactly representable values: bfloat16 has
    8 mantissa bits, so a scatter_add of random floats would fail on rounding
    rather than on anything this family of ops does."""
    numel = math.prod(shape)
    values = torch.arange(numel, dtype=torch.int64) % 17 - 8
    return values.reshape(shape).to(dtype)


@pytest.mark.parametrize("dtype", _INDEX_DTYPES)
@pytest.mark.parametrize(
    ("shape", "dim"),
    [((6, 5), 0), ((6, 5), 1), ((6, 5), -1), ((3, 4, 5), 1), ((2, 3, 4, 5), 2)],
)
def test_aten_gather(
    conf: Conf,
    dtype: torch.dtype,
    shape: tuple[int, ...],
    dim: int,
    call_checker: CallChecker,
):
    call_checker.register(aten_functions.aten_gather)

    def fn(x, index):
        return aten.gather(x, dim, index)

    x = _index_payload(shape, dtype)
    # `index.size(d) <= self.size(d)` is legal for gather on EVERY dim, not
    # only the indexed one, and the output is shaped like the INDEX — so a
    # kernel that walked `self`'s shape, or took the output's strides from
    # `self`, is wrong here while an equal-shaped index would hide it.
    index_shape = [3 if d == dim else max(1, s - 1) for d, s in enumerate(shape)]
    index = torch.randint(0, shape[dim], index_shape, dtype=torch.int64)
    check_outputs(fn, conf, [x, index])


def test_aten_gather_strided_input(conf: Conf, call_checker: CallChecker):
    """A transposed (non-contiguous) `self`: the kernel reads it through its
    real strides instead of materializing a copy first."""
    call_checker.register(aten_functions.aten_gather)

    def fn(x, index):
        return aten.gather(x.transpose(0, 1), 1, index)

    x = torch.randn(7, 5)
    index = torch.randint(0, 7, (5, 7), dtype=torch.int64)
    check_outputs(fn, conf, [x, index])


def test_aten_take_along_dim(conf: Conf, call_checker: CallChecker):
    """`take_along_dim` is a composite that lowers straight to gather — and,
    unlike gather itself, it DOES define negative indices (it normalizes them
    with `remainder` before calling gather)."""
    call_checker.register(aten_functions.aten_gather)

    def fn(x, index):
        return torch.take_along_dim(x, index, dim=1)

    x = torch.randn(4, 6)
    index = torch.tensor([[0, -1, 2], [-2, 1, 0], [3, 3, -6], [5, 0, 1]])
    check_outputs(fn, conf, [x, index])


@pytest.mark.parametrize("dtype", _INDEX_DTYPES)
@pytest.mark.parametrize(
    ("shape", "dim"),
    [((6, 5), 0), ((6, 5), 1), ((6, 5), -1), ((3, 4, 5), 1), ((2, 3, 4, 5, 2), 2)],
)
def test_aten_index_select(
    conf: Conf,
    dtype: torch.dtype,
    shape: tuple[int, ...],
    dim: int,
    call_checker: CallChecker,
):
    call_checker.register(aten_functions.aten_index_select)

    def fn(x, index):
        return aten.index_select(x, dim, index)

    x = _index_payload(shape, dtype)
    # Duplicated and out-of-order indices, and a length that differs from the
    # indexed extent (index_select is the one op in the family whose output is
    # shaped like neither input).
    index = torch.tensor([2, 0, 2, 1, 0, 1, 2], dtype=torch.int64)
    check_outputs(fn, conf, [x, index])


def test_aten_index_select_int32_index(conf: Conf, call_checker: CallChecker):
    """index_select accepts an int32 index; gather/scatter_add do not."""
    call_checker.register(aten_functions.aten_index_select)

    def fn(x, index):
        return aten.index_select(x, 0, index)

    x = torch.randn(5, 3)
    index = torch.tensor([4, 0, 2], dtype=torch.int32)
    check_outputs(fn, conf, [x, index])


def test_aten_index_select_strided_input(conf: Conf, call_checker: CallChecker):
    call_checker.register(aten_functions.aten_index_select)

    def fn(x, index):
        return aten.index_select(x.transpose(0, 1), 1, index)

    x = torch.randn(6, 4)
    index = torch.tensor([5, 5, 1, 0], dtype=torch.int64)
    check_outputs(fn, conf, [x, index])


@pytest.mark.parametrize("dtype", _INDEX_DTYPES)
@pytest.mark.parametrize(
    ("shape", "dim"), [((6, 5), 0), ((6, 5), 1), ((6, 5), -1), ((3, 4, 5), 1)]
)
def test_aten_scatter_add(
    conf: Conf,
    dtype: torch.dtype,
    shape: tuple[int, ...],
    dim: int,
    call_checker: CallChecker,
):
    """Every index here is duplicated many times over: the whole point of
    scatter_add is that colliding writes SUM instead of clobbering, and a test
    with unique indices would pass against a plain scatter."""
    call_checker.register(aten_functions.aten_scatter_add)

    def fn(x, index, src):
        return aten.scatter_add(x, dim, index, src)

    x = _index_payload(shape, dtype)
    index = torch.zeros(shape, dtype=torch.int64)
    index.narrow(dim % len(shape), 1, shape[dim] - 1).fill_(1)
    src = _index_payload(shape, dtype)
    check_outputs(fn, conf, [x, index, src])


def test_aten_scatter_add_inplace(conf: Conf, call_checker: CallChecker):
    """`scatter_add_` and `scatter_add` both route through
    `aten::scatter_add.out`; in the in-place one the out= tensor IS self, so
    the "start from a copy of self" step must be a no-op rather than a copy of
    an already-scattered buffer."""
    call_checker.register(aten_functions.aten_scatter_add)

    def fn(x, index, src):
        out = x.clone()
        out.scatter_add_(1, index, src)
        return out

    x = _index_payload((4, 6), torch.float32)
    index = torch.randint(0, 6, (4, 6), dtype=torch.int64)
    src = _index_payload((4, 6), torch.float32)
    check_outputs(fn, conf, [x, index, src])


@pytest.mark.parametrize("dtype", _INDEX_DTYPES)
@pytest.mark.parametrize(("shape", "dim"), [((6, 5), 0), ((6, 5), 1), ((3, 4, 5), 1)])
def test_aten_index_add(
    conf: Conf,
    dtype: torch.dtype,
    shape: tuple[int, ...],
    dim: int,
    call_checker: CallChecker,
):
    """index_add is scatter_add with a broadcast index, and it is also what
    autograd calls for index_select's backward — an unimplemented backward on
    this device aborts the process instead of raising, so it stops being
    optional the moment index_select exists."""
    call_checker.register(aten_functions.aten_index_add)

    def fn(x, index, source):
        return aten.index_add(x, dim, index, source)

    x = _index_payload(shape, dtype)
    index = torch.tensor([0, 0, 1, 0], dtype=torch.int64)
    source_shape = list(shape)
    source_shape[dim] = 4
    source = _index_payload(tuple(source_shape), dtype)
    check_outputs(fn, conf, [x, index, source])


@pytest.mark.parametrize("accumulate", [False, True])
@pytest.mark.parametrize("dtype", _INDEX_DTYPES)
def test_aten_index_put(
    conf: Conf, dtype: torch.dtype, accumulate: bool, call_checker: CallChecker
):
    """`x[idx] = v` / `x[idx] += v`. With `accumulate=False` the indices are
    unique on purpose: torch leaves the winner of colliding writes undefined
    on an accelerator, so a duplicate there would be testing nothing."""
    call_checker.register(aten_functions.aten_index_put)

    def fn(x, index, values):
        return aten.index_put(x, [index], values, accumulate)

    x = _index_payload((6, 4), dtype)
    index = torch.tensor(
        [3, 0, 3, 1] if accumulate else [3, 0, 5, 1], dtype=torch.int64
    )
    values = _index_payload((4, 4), dtype)
    check_outputs(fn, conf, [x, index, values])


def test_aten_index_put_negative_indices(conf: Conf, call_checker: CallChecker):
    """Advanced indexing — unlike gather/index_select/scatter_add — DOES
    define negative indices, and wraps them python-style."""
    call_checker.register(aten_functions.aten_index_put)

    def fn(x, index, values):
        return aten.index_put(x, [index], values, False)

    x = torch.randn(5, 3)
    index = torch.tensor([-1, -5, 2], dtype=torch.int64)
    values = torch.randn(3, 3)
    check_outputs(fn, conf, [x, index, values])


def test_aten_index_put_broadcast_values(conf: Conf, call_checker: CallChecker):
    call_checker.register(aten_functions.aten_index_put)

    def fn(x, index, values):
        return aten.index_put(x, [index], values, False)

    x = torch.randn(5, 3)
    index = torch.tensor([1, 4], dtype=torch.int64)
    values = torch.randn(3)  # broadcast over the indexed rows
    check_outputs(fn, conf, [x, index, values])


def test_aten_index_put_bool_mask_scalar(conf: Conf):
    """`x[mask] = scalar` does NOT reach a bool-mask index_put kernel: torch's
    own `_index_put_impl_` re-routes exactly this shape to `masked_fill_`
    (canDispatchToMaskedFill), and so does the mojo one — which is why no
    index_put implementation is asserted here."""

    def fn(x):
        out = x.clone()
        out[x > 0] = -1.0
        return out

    check_outputs(fn, conf, [torch.randn(4, 5)])


def test_aten_index_put_bool_row_mask_scalar(conf: Conf):
    """A mask that covers only the LEADING dim, on a SQUARE tensor.

    A mask index selects leading dims, but broadcasting is right-aligned, so
    routing a shape-(4,) mask straight into masked_fill_ on a (4, 4) tensor
    fills the wrong axis — and only a square tensor exposes it, because any
    other shape raises instead of answering wrongly.
    """

    def fn(x, keep):
        out = x.clone()
        out[keep] = 0.0
        return out

    x = torch.arange(16, dtype=torch.float32).reshape(4, 4)
    keep = torch.tensor([True, False, False, True])
    check_outputs(fn, conf, [x, keep])


def test_aten_isin_none_match(conf: Conf):
    """Test aten.isin when no elements are in test_elements"""

    def fn(elements, test_elements):
        return aten.isin(elements, test_elements, assume_unique=False, invert=False)

    elements = torch.tensor([1, 2, 3])
    test_elements = torch.tensor([4, 5, 6])

    # No elements should be found
    # Expected: [False, False, False]
    check_outputs(fn, conf, [elements, test_elements])


def test_aten_isin_with_duplicates(conf: Conf):
    """Test aten.isin with duplicates in both tensors"""

    def fn(elements, test_elements):
        return aten.isin(elements, test_elements, assume_unique=False, invert=False)

    elements = torch.tensor([1, 2, 2, 3, 3, 3])
    test_elements = torch.tensor([2, 2, 4])

    # Elements 2 and 2 should be found (duplicated values treated separately)
    # Note: behavior depends on assume_unique, with False it should handle duplicates
    check_outputs(fn, conf, [elements, test_elements])


def test_aten_isin_2d_tensor(conf: Conf):
    """Test aten.isin with 2D tensor"""

    def fn(elements, test_elements):
        return aten.isin(elements, test_elements, assume_unique=False, invert=False)

    elements = torch.tensor([[1, 2, 3], [4, 5, 6]])
    test_elements = torch.tensor([2, 5, 7])

    # Expected: [[False, True, False], [False, True, False]]
    check_outputs(fn, conf, [elements, test_elements])


def test_aten_isin_3d_tensor(conf: Conf):
    """Test aten.isin with 3D tensor"""

    def fn(elements, test_elements):
        return aten.isin(elements, test_elements, assume_unique=False, invert=False)

    elements = torch.tensor([[[1, 2], [3, 4]], [[5, 6], [7, 8]]])
    test_elements = torch.tensor([2, 4, 6, 8])

    # Expected: [[[False, True], [False, True]], [[False, True], [False, True]]]
    check_outputs(fn, conf, [elements, test_elements])


@pytest.mark.parametrize("dtype", [torch.float32, torch.bfloat16])
@pytest.mark.parametrize(("shape", "dim"), [((64, 40), -1), ((4, 5, 64), 1)])
def test_aten__log_softmax_backward_data_autograd(
    conf: Conf,
    call_checker: CallChecker,
    dtype: torch.dtype,
    shape: tuple[int, ...],
    dim: int,
):
    """Backward through log_softmax reaches the dedicated backward op."""
    call_checker.register(aten_functions.aten__log_softmax_backward_data)

    def fn(x, grad_output):
        leaf = x.detach().requires_grad_(True)
        y = torch.log_softmax(leaf, dim=dim)
        (grad_input,) = torch.autograd.grad(y, leaf, grad_output)
        return grad_input

    x = torch.randn(shape, dtype=dtype)
    grad_output = torch.randn(shape, dtype=dtype)
    # bf16 mismatch is dominated by the CPU reference's own rounding: a 1-ulp
    # wobble in its forward output scales with |rowsum(grad_output)|, which is
    # unbounded for unseeded inputs (worst mismatch ~0.04 over 2000 seeds while
    # our side stays within 1 ulp of the fp64 ground truth; a broken backward
    # shows O(1) errors).
    tolerance = 6e-2 if dtype == torch.bfloat16 else 2e-3
    check_outputs(fn, conf, [x, grad_output], atol=tolerance, rtol=tolerance)


def test_aten_isin_with_assume_unique(conf: Conf):
    """Test aten.isin with assume_unique=True"""

    def fn(elements, test_elements):
        return aten.isin(elements, test_elements, assume_unique=True, invert=False)

    elements = torch.tensor([1, 2, 3, 4, 5])
    test_elements = torch.tensor([2, 4])

    # Result should be the same as assume_unique=False for valid unique inputs
    # Expected: [False, True, False, True, False]
    check_outputs(fn, conf, [elements, test_elements])


@pytest.mark.parametrize("dtype", [torch.float32, torch.int32])
def test_aten_isin_numeric_types(conf: Conf, dtype: torch.dtype):
    """Test aten.isin with different numeric types"""

    def fn(elements, test_elements):
        return aten.isin(elements, test_elements, assume_unique=False, invert=False)

    if dtype == torch.float32:
        elements = torch.tensor([1.5, 2.5, 3.5], dtype=dtype)
        test_elements = torch.tensor([2.5, 4.5], dtype=dtype)
    else:
        elements = torch.tensor([10, 20, 30], dtype=dtype)
        test_elements = torch.tensor([20, 40], dtype=dtype)

    check_outputs(fn, conf, [elements, test_elements])


@pytest.mark.parametrize("dtype", [torch.float32, torch.float64])
def test_aten_select_scatter_basic_2d(conf: Conf, dtype: torch.dtype):
    """Test aten.select_scatter basic functionality with 2D tensors"""

    def fn(self, src):
        return aten.select_scatter(self, src, dim=0, index=1)

    # Replace row 1 with new values
    self = torch.zeros(3, 4, dtype=dtype)
    src = torch.ones(4, dtype=dtype) * 5

    check_outputs(fn, conf, [self, src])


@pytest.mark.parametrize("dtype", [torch.float32, torch.int32, torch.int64])
def test_aten_select_scatter_dim1(conf: Conf, dtype: torch.dtype):
    """Test aten.select_scatter along dimension 1"""

    def fn(self, src):
        return aten.select_scatter(self, src, dim=1, index=2)

    # Replace column 2 with new values
    self = torch.zeros(3, 4, dtype=dtype)
    src = torch.ones(3, dtype=dtype) * 7

    check_outputs(fn, conf, [self, src])


@pytest.mark.parametrize("dtype", [torch.float32, torch.bfloat16])
def test_aten_select_scatter_3d_tensor(conf: Conf, dtype: torch.dtype):
    """Test aten.select_scatter with 3D tensor"""
    if conf.device == "cpu" and dtype == torch.float16:
        pytest.skip("float16 not supported on CPU in MAX")

    def fn(self, src):
        return aten.select_scatter(self, src, dim=1, index=2)

    # Replace middle slice along dimension 1
    self = torch.zeros(2, 4, 3, dtype=dtype)
    src = torch.randn(2, 3, dtype=dtype)

    check_outputs(fn, conf, [self, src])


def test_aten_select_scatter_negative_dim(conf: Conf):
    """Test aten.select_scatter with negative dimension"""

    def fn(self, src):
        return aten.select_scatter(self, src, dim=-1, index=1)

    # Negative dimension indexing (dim=-1 is last dimension)
    self = torch.zeros(3, 4, dtype=torch.float32)
    src = torch.ones(3, dtype=torch.float32) * 2

    check_outputs(fn, conf, [self, src])


def test_aten_select_scatter_negative_index(conf: Conf):
    """Test aten.select_scatter with negative index"""

    def fn(self, src):
        return aten.select_scatter(self, src, dim=0, index=-1)

    # Negative index (index=-1 is last index)
    self = torch.zeros(3, 4, dtype=torch.float32)
    src = torch.ones(4, dtype=torch.float32) * 3

    check_outputs(fn, conf, [self, src])


def test_aten_select_scatter_scalar_src(conf: Conf):
    """Test aten.select_scatter with scalar src (1D self)"""

    def fn(self, src):
        return aten.select_scatter(self, src, dim=0, index=1)

    # When self is 1D, src is a scalar (0D tensor)
    self = torch.zeros(3, dtype=torch.float32)
    src = torch.tensor(5.0, dtype=torch.float32)

    check_outputs(fn, conf, [self, src])


@pytest.mark.parametrize("repeats", [1, 2, 3, 5])
@pytest.mark.parametrize("dim", [0, 1, -1])
def test_aten_repeat_interleave_basic(device: str, repeats: int, dim: int):
    """Test aten.repeat_interleave with basic parameters"""

    def fn(x):
        return aten.repeat_interleave(x, repeats, dim)

    x = torch.randn(3, 4, device=device)
    check_functions_are_equivalent(fn, device, [x])


@pytest.mark.parametrize("dtype", [torch.float32, torch.int32, torch.bool])
def test_aten_repeat_interleave_different_dtypes(device: str, dtype: torch.dtype):
    """Test aten.repeat_interleave with different data types"""

    def fn(x):
        return aten.repeat_interleave(x, 2, 0)

    if dtype == torch.bool:
        x = torch.randint(0, 2, (3, 4), dtype=dtype, device=device)
    elif dtype == torch.int32:
        x = torch.randint(0, 10, (3, 4), dtype=dtype, device=device)
    else:
        x = torch.randn(3, 4, dtype=dtype, device=device)

    check_functions_are_equivalent(fn, device, [x])


def test_aten_repeat_interleave_1d(device: str):
    """Test aten.repeat_interleave with 1D tensor"""

    def fn(x):
        return aten.repeat_interleave(x, 3, 0)

    x = torch.randn(5, device=device)
    check_functions_are_equivalent(fn, device, [x])


def test_aten_repeat_interleave_3d(device: str):
    """Test aten.repeat_interleave with 3D tensor"""

    def fn(x):
        return aten.repeat_interleave(x, 2, 1)

    x = torch.randn(2, 3, 4, device=device)
    check_functions_are_equivalent(fn, device, [x])


@pytest.mark.parametrize("shape", [(1, 5), (5, 1), (1, 1)])
def test_aten_repeat_interleave_edge_cases(device: str, shape: tuple):
    """Test aten.repeat_interleave with edge case shapes"""

    def fn(x):
        return aten.repeat_interleave(x, 2, 0)

    x = torch.randn(*shape, device=device)
    check_functions_are_equivalent(fn, device, [x])


def test_aten_repeat_interleave_large_repeats(device: str):
    """Test aten.repeat_interleave with large repeat count"""

    def fn(x):
        return aten.repeat_interleave(x, 10, 0)

    x = torch.randn(2, 3, device=device)
    check_functions_are_equivalent(fn, device, [x])


@pytest.mark.parametrize("dtype", [torch.float32, torch.float64, torch.bfloat16])
def test_aten_pow_tensor_tensor_basic(conf: Conf, dtype: torch.dtype):
    """Test pow.Tensor_Tensor basic functionality with floating point numbers"""
    # Skip float16 on CPU as MAX doesn't support f16 on CPU
    if conf.device == "cpu" and dtype == torch.float16:
        pytest.skip("float16 not supported on CPU in MAX")

    def fn(base, exponent):
        return aten.pow.Tensor_Tensor(base, exponent)

    # Test with various base and exponent values
    base = torch.tensor([1.0, 2.0, 3.0, 4.0, 5.0], dtype=dtype)
    exponent = torch.tensor([2.0, 3.0, 0.5, -1.0, 0.0], dtype=dtype)
    check_outputs(fn, conf, [base, exponent])


@pytest.mark.parametrize("dtype", [torch.int32, torch.int64])
def test_aten_pow_tensor_tensor_integer(conf: Conf, dtype: torch.dtype):
    """Test pow.Tensor_Tensor with integer types and positive exponents"""

    def fn(base, exponent):
        return aten.pow.Tensor_Tensor(base, exponent)

    # Test with integer values and positive exponents
    base = torch.tensor([1, 2, 3, 4, 5], dtype=dtype)
    exponent = torch.tensor([2, 3, 1, 0, 2], dtype=dtype)
    check_outputs(fn, conf, [base, exponent])


def test_aten_pow_tensor_tensor_broadcasting_basic(conf: Conf):
    """Test pow.Tensor_Tensor with broadcasting"""

    def fn(base, exponent):
        return aten.pow.Tensor_Tensor(base, exponent)

    # Test broadcasting: (3, 1) with (3, 4)
    base = torch.tensor([[2.0], [3.0], [4.0]], dtype=torch.float32)
    exponent = torch.tensor([[1.0, 2.0, 3.0, 0.5]], dtype=torch.float32)
    check_outputs(fn, conf, [base, exponent])


def test_aten_pow_tensor_tensor_broadcasting_scalar(conf: Conf):
    """Test pow.Tensor_Tensor with scalar-like broadcasting"""

    def fn(base, exponent):
        return aten.pow.Tensor_Tensor(base, exponent)

    # Test broadcasting: (4,) with (1,)
    base = torch.tensor([1.0, 2.0, 3.0, 4.0], dtype=torch.float32)
    exponent = torch.tensor([2.0], dtype=torch.float32)
    check_outputs(fn, conf, [base, exponent])


def test_aten_pow_tensor_tensor_2d_tensor(conf: Conf):
    """Test pow.Tensor_Tensor with 2D tensors"""

    def fn(base, exponent):
        return aten.pow.Tensor_Tensor(base, exponent)

    base = torch.tensor([[1.0, 2.0, 3.0], [4.0, 5.0, 6.0]], dtype=torch.float32)
    exponent = torch.tensor([[2.0, 3.0, 1.0], [0.5, 2.0, -1.0]], dtype=torch.float32)
    check_outputs(fn, conf, [base, exponent])


def test_aten_pow_tensor_tensor_3d_tensor(conf: Conf):
    """Test pow.Tensor_Tensor with 3D tensors"""

    def fn(base, exponent):
        return aten.pow.Tensor_Tensor(base, exponent)

    base = torch.randn(2, 3, 4, dtype=torch.float32).abs() + 1.0  # Ensure positive
    exponent = torch.randn(2, 3, 4, dtype=torch.float32) * 2.0
    check_outputs(fn, conf, [base, exponent])


def test_aten_pow_tensor_tensor_special_exponents(conf: Conf):
    """Test pow.Tensor_Tensor with special exponent values (0, 1, 2, -1)"""

    def fn(base, exponent):
        return aten.pow.Tensor_Tensor(base, exponent)

    # Test special exponents
    base = torch.tensor([2.0, 3.0, 4.0, 5.0], dtype=torch.float32)
    exponent = torch.tensor([0.0, 1.0, 2.0, -1.0], dtype=torch.float32)
    check_outputs(fn, conf, [base, exponent])


def test_aten_pow_tensor_tensor_special_bases(conf: Conf):
    """Test pow.Tensor_Tensor with special base values (0, 1, -1)"""

    def fn(base, exponent):
        return aten.pow.Tensor_Tensor(base, exponent)

    # Test special bases (note: 0^0 should be 1 in PyTorch)
    base = torch.tensor([0.0, 1.0, 1.0, -1.0, -1.0], dtype=torch.float32)
    exponent = torch.tensor([0.0, 5.0, -5.0, 2.0, 3.0], dtype=torch.float32)
    check_outputs(fn, conf, [base, exponent])


def test_aten_pow_tensor_tensor_fractional_exponents(conf: Conf):
    """Test pow.Tensor_Tensor with fractional exponents (roots)"""

    def fn(base, exponent):
        return aten.pow.Tensor_Tensor(base, exponent)

    # Test fractional exponents (roots) - use positive bases
    base = torch.tensor([1.0, 4.0, 9.0, 16.0, 25.0], dtype=torch.float32)
    exponent = torch.tensor([0.5, 0.5, 0.5, 0.25, 0.333], dtype=torch.float32)
    check_outputs(fn, conf, [base, exponent])


def test_aten_pow_tensor_tensor_large_values(conf: Conf):
    """Test pow.Tensor_Tensor with larger values"""

    def fn(base, exponent):
        return aten.pow.Tensor_Tensor(base, exponent)

    # Test with larger bases and smaller exponents to avoid overflow
    base = torch.tensor([10.0, 5.0, 3.0, 2.0], dtype=torch.float32)
    exponent = torch.tensor([2.0, 3.0, 4.0, 10.0], dtype=torch.float32)
    check_outputs(fn, conf, [base, exponent])


def test_aten_pow_tensor_tensor_negative_base_integer_exponent(conf: Conf):
    """Test pow.Tensor_Tensor with negative base and integer exponents"""

    def fn(base, exponent):
        return aten.pow.Tensor_Tensor(base, exponent)

    # Negative bases with integer exponents
    base = torch.tensor([-2.0, -3.0, -2.0, -3.0], dtype=torch.float32)
    exponent = torch.tensor([2.0, 2.0, 3.0, 3.0], dtype=torch.float32)
    check_outputs(fn, conf, [base, exponent])


def test_aten_pow_tensor_tensor_single_element(conf: Conf):
    """Test pow.Tensor_Tensor with single element tensors"""

    def fn(base, exponent):
        return aten.pow.Tensor_Tensor(base, exponent)

    base = torch.tensor([2.0], dtype=torch.float32)
    exponent = torch.tensor([3.0], dtype=torch.float32)
    check_outputs(fn, conf, [base, exponent])


@pytest.mark.parametrize(("right", "out_int32"), [(False, False), (True, True)])
@pytest.mark.parametrize(
    "dtype", [torch.float32, torch.bfloat16, torch.float16, torch.int32, torch.int64]
)
def test_aten_bucketize_tensor(
    mojo_device: str,
    call_checker: CallChecker,
    dtype: torch.dtype,
    right: bool,
    out_int32: bool,
) -> None:
    register_mojo_devices()
    call_checker.register(aten_functions.aten_bucketize)

    def fn(values: torch.Tensor, boundaries: torch.Tensor) -> torch.Tensor:
        return aten.bucketize.Tensor(
            values, boundaries, right=right, out_int32=out_int32
        )

    boundaries = torch.tensor([-5, -1, 0, 2, 2, 9], dtype=dtype)
    values = torch.tensor([[-6, -1, 1], [2, 8, 10]], dtype=dtype)
    check_outputs(fn, Conf(mojo_device, False), [values, boundaries])


@pytest.mark.parametrize(("side", "out_int32"), [("left", True), ("right", False)])
@pytest.mark.parametrize(
    "dtype", [torch.float32, torch.bfloat16, torch.float16, torch.int32, torch.int64]
)
def test_aten_searchsorted_batched(
    mojo_device: str,
    call_checker: CallChecker,
    dtype: torch.dtype,
    side: str,
    out_int32: bool,
) -> None:
    register_mojo_devices()
    call_checker.register(aten_functions.aten_searchsorted)

    def fn(boundaries: torch.Tensor, values: torch.Tensor) -> torch.Tensor:
        return aten.searchsorted.Tensor(
            boundaries, values, side=side, out_int32=out_int32
        )

    boundaries = torch.tensor([[-3, 0, 0, 7], [1, 4, 8, 12]], dtype=dtype)
    values = torch.tensor([[-4, 0, 6], [1, 9, 20]], dtype=dtype)
    check_outputs(fn, Conf(mojo_device, False), [boundaries, values])


@pytest.mark.parametrize(
    "dtype", [torch.float32, torch.bfloat16, torch.float16, torch.int32, torch.int64]
)
def test_aten_searchsorted_sorter(
    mojo_device: str, call_checker: CallChecker, dtype: torch.dtype
) -> None:
    register_mojo_devices()
    call_checker.register(aten_functions.aten_searchsorted)

    def fn(
        boundaries: torch.Tensor, values: torch.Tensor, sorter: torch.Tensor
    ) -> torch.Tensor:
        return aten.searchsorted.Tensor(boundaries, values, sorter=sorter)

    boundaries = torch.tensor([[30, 10, 20], [6, -2, 3]], dtype=dtype)
    sorter = torch.tensor([[1, 2, 0], [1, 2, 0]], dtype=torch.int64)
    values = torch.tensor([[5, 10, 25, 35], [-3, 2, 6, 8]], dtype=dtype)
    check_outputs(fn, Conf(mojo_device, False), [boundaries, values, sorter])


@pytest.mark.parametrize(
    ("boundary_dtype", "value_dtype"),
    [
        (torch.float16, torch.bfloat16),
        (torch.int64, torch.float16),
        (torch.int32, torch.int64),
    ],
)
def test_aten_searchsorted_dtype_promotion(
    mojo_device: str,
    call_checker: CallChecker,
    boundary_dtype: torch.dtype,
    value_dtype: torch.dtype,
) -> None:
    register_mojo_devices()
    call_checker.register(aten_functions.aten_searchsorted)

    def fn(boundaries: torch.Tensor, values: torch.Tensor) -> torch.Tensor:
        return aten.searchsorted.Tensor(boundaries, values, right=True)

    boundaries = torch.tensor([-5, 0, 2, 9], dtype=boundary_dtype)
    values = torch.tensor([-6, 0, 1, 10], dtype=value_dtype)
    check_outputs(fn, Conf(mojo_device, False), [boundaries, values])


@pytest.mark.parametrize("out_int32", [False, True])
def test_aten_searchsorted_and_bucketize_scalar_overloads(
    conf: Conf, call_checker: CallChecker, out_int32: bool
) -> None:
    call_checker.register(
        aten_functions.aten_searchsorted, aten_functions.aten_bucketize
    )

    def fn(boundaries: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        return (
            aten.searchsorted.Scalar(
                boundaries, 2.0, side="right", out_int32=out_int32
            ),
            aten.bucketize.Scalar(float("nan"), boundaries, out_int32=out_int32),
        )

    boundaries = torch.tensor(
        [float("-inf"), 0, 2, 2, float("inf")], dtype=torch.float32
    )
    check_outputs(fn, conf, [boundaries])


@pytest.mark.parametrize("right", [False, True])
def test_aten_searchsorted_and_bucketize_empty_edges(
    conf: Conf, call_checker: CallChecker, right: bool
) -> None:
    call_checker.register(
        aten_functions.aten_searchsorted, aten_functions.aten_bucketize
    )

    def fn(
        empty_boundaries: torch.Tensor,
        values: torch.Tensor,
        batched_boundaries: torch.Tensor,
        batched_values: torch.Tensor,
        nonempty_boundaries: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        return (
            aten.searchsorted.Tensor(empty_boundaries, values, right=right),
            aten.bucketize.Tensor(values, empty_boundaries, right=right),
            aten.searchsorted.Tensor(batched_boundaries, batched_values, right=right),
            aten.bucketize.Tensor(values[:0], nonempty_boundaries, right=right),
        )

    empty_boundaries = torch.empty(0)
    values = torch.tensor([float("-inf"), 0.0, float("inf"), float("nan")])
    batched_boundaries = torch.empty(2, 0)
    batched_values = torch.randn(2, 3)
    nonempty_boundaries = torch.tensor([0.0, 1.0])
    check_outputs(
        fn,
        conf,
        [
            empty_boundaries,
            values,
            batched_boundaries,
            batched_values,
            nonempty_boundaries,
        ],
    )


def test_aten_searchsorted_and_bucketize_eager_cpu_and_gpu(
    mojo_device: str, call_checker: CallChecker
) -> None:
    register_mojo_devices()
    call_checker.register(
        aten_functions.aten_searchsorted, aten_functions.aten_bucketize
    )

    def fn(
        boundaries: torch.Tensor, values: torch.Tensor
    ) -> tuple[torch.Tensor, torch.Tensor]:
        return (
            aten.searchsorted.Tensor(boundaries, values, right=True),
            aten.bucketize.Tensor(values, boundaries, out_int32=True),
        )

    boundaries = torch.tensor([-4.0, 0.0, 3.0, 8.0])
    values = torch.tensor([[float("-inf"), 0.0, 4.0, float("inf"), float("nan")]])
    check_outputs(fn, Conf(mojo_device, False), [boundaries, values])


@pytest.mark.parametrize("right", [False, True])
def test_aten_searchsorted_and_bucketize_nan_boundaries_eager(
    mojo_device: str, call_checker: CallChecker, right: bool
) -> None:
    register_mojo_devices()
    call_checker.register(
        aten_functions.aten_searchsorted, aten_functions.aten_bucketize
    )

    def fn(
        boundaries: torch.Tensor, values: torch.Tensor
    ) -> tuple[torch.Tensor, torch.Tensor]:
        return (
            aten.searchsorted.Tensor(boundaries, values, right=right),
            aten.bucketize.Tensor(values, boundaries, right=right),
        )

    boundaries = torch.tensor([1.0, 2.0, float("nan")])
    values = torch.tensor([0.5, 1.5, 3.0, float("nan")])
    check_outputs(fn, Conf(mojo_device, False), [boundaries, values])


@pytest.mark.parametrize("out_int32", [False, True])
def test_aten_searchsorted_and_bucketize_out_variants(
    mojo_device: str, call_checker: CallChecker, out_int32: bool
) -> None:
    register_mojo_devices()
    call_checker.register(
        aten_functions.aten_searchsorted, aten_functions.aten_bucketize
    )
    dtype = torch.int32 if out_int32 else torch.int64
    boundaries = torch.tensor([1.0, 2.0, 4.0]).to(mojo_device)
    values = torch.tensor([0.5, 2.0, 3.0, 5.0]).to(mojo_device)
    tensor_out = torch.empty(0, dtype=dtype, device=mojo_device)
    scalar_out = torch.empty((), dtype=dtype, device=mojo_device)

    expected_tensor = torch.tensor([0, 1, 2, 3], dtype=dtype)
    expected_scalar = torch.tensor(2, dtype=dtype)
    assert (
        aten.searchsorted.Tensor_out(
            boundaries, values, out_int32=out_int32, out=tensor_out
        )
        is tensor_out
    )
    torch.testing.assert_close(tensor_out.cpu(), expected_tensor)
    assert (
        aten.searchsorted.Scalar_out(
            boundaries, 3.0, out_int32=out_int32, out=scalar_out
        )
        is scalar_out
    )
    torch.testing.assert_close(scalar_out.cpu(), expected_scalar)

    tensor_out = torch.empty(0, dtype=dtype, device=mojo_device)
    scalar_out = torch.empty((), dtype=dtype, device=mojo_device)
    assert (
        aten.bucketize.Tensor_out(
            values, boundaries, out_int32=out_int32, out=tensor_out
        )
        is tensor_out
    )
    torch.testing.assert_close(tensor_out.cpu(), expected_tensor)
    assert (
        aten.bucketize.Scalar_out(3.0, boundaries, out_int32=out_int32, out=scalar_out)
        is scalar_out
    )
    torch.testing.assert_close(scalar_out.cpu(), expected_scalar)


def test_aten_searchsorted_and_bucketize_errors(
    mojo_device: str, call_checker: CallChecker
) -> None:
    register_mojo_devices()
    call_checker.register(
        aten_functions.aten_searchsorted, aten_functions.aten_bucketize
    )
    boundaries = torch.tensor([1.0, 3.0, 7.0]).to(mojo_device)
    values = torch.tensor([0.0, 4.0]).to(mojo_device)

    with pytest.raises(RuntimeError, match="side can only be 'left' or 'right'"):
        aten.searchsorted.Tensor(boundaries, values, side="middle")
    with pytest.raises(RuntimeError, match="side and right can't be set to opposites"):
        aten.searchsorted.Tensor(boundaries, values, right=True, side="left")
    with pytest.raises(RuntimeError, match="sorter must be a tensor of long dtype"):
        aten.searchsorted.Tensor(
            boundaries,
            values,
            sorter=torch.tensor([0, 1, 2], dtype=torch.int32).to(mojo_device),
        )
    with pytest.raises(RuntimeError, match="sorter index out of range"):
        aten.searchsorted.Tensor(
            boundaries,
            values,
            sorter=torch.tensor([0, 1, 3], dtype=torch.int64).to(mojo_device),
        )
    with pytest.raises(RuntimeError, match="should have same device type"):
        aten.searchsorted.Tensor(boundaries, torch.tensor([0.0, 4.0]))
    with pytest.raises(
        RuntimeError, match="sorter and boundary tensors should have same device type"
    ):
        aten.searchsorted.Tensor(
            boundaries, values, sorter=torch.tensor([0, 1, 2], dtype=torch.int64)
        )
    with pytest.raises(RuntimeError, match="boundaries tensor must be 1 dimension"):
        aten.bucketize.Tensor(values, boundaries.unsqueeze(0))


def test_aten_searchsorted_and_bucketize_compile_backend(
    call_checker: CallChecker,
) -> None:
    call_checker.register(
        aten_functions.aten_searchsorted, aten_functions.aten_bucketize
    )

    def fn(
        batched_boundaries: torch.Tensor,
        batched_values: torch.Tensor,
        boundaries: torch.Tensor,
        values: torch.Tensor,
        empty_boundaries: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        return (
            aten.searchsorted.Tensor(batched_boundaries, batched_values, side="right"),
            aten.bucketize.Tensor(values, boundaries, out_int32=True),
            aten.searchsorted.Scalar(boundaries, 2.0, side="left"),
            aten.bucketize.Scalar(float("nan"), boundaries, right=True),
            aten.searchsorted.Tensor(empty_boundaries, values),
        )

    batched_boundaries = torch.tensor([[10.0, 20.0, 30.0], [-2.0, 3.0, 6.0]])
    batched_values = torch.tensor([[5.0, 10.0, 25.0], [-3.0, 2.0, 8.0]])
    boundaries = torch.tensor([-2.0, 0.0, 4.0, 9.0])
    values = torch.tensor([float("-inf"), 0.0, 5.0, float("inf"), float("nan")])
    empty_boundaries = torch.empty(0)
    check_outputs(
        fn,
        Conf("cpu", True),
        [batched_boundaries, batched_values, boundaries, values, empty_boundaries],
    )


@pytest.mark.parametrize("right", [False, True])
def test_aten_searchsorted_and_bucketize_nan_boundaries_compile_backend(
    call_checker: CallChecker, right: bool
) -> None:
    call_checker.register(
        aten_functions.aten_searchsorted, aten_functions.aten_bucketize
    )

    def fn(
        boundaries: torch.Tensor, values: torch.Tensor
    ) -> tuple[torch.Tensor, torch.Tensor]:
        return (
            aten.searchsorted.Tensor(boundaries, values, right=right),
            aten.bucketize.Tensor(values, boundaries, right=right),
        )

    boundaries = torch.tensor([1.0, 2.0, float("nan")])
    values = torch.tensor([0.5, 1.5, 3.0, float("nan")])
    check_outputs(fn, Conf("cpu", True), [boundaries, values])


def test_aten_searchsorted_compile_backend_declines_unchecked_sorter() -> None:
    def fn(
        boundaries: torch.Tensor, values: torch.Tensor, sorter: torch.Tensor
    ) -> torch.Tensor:
        return aten.searchsorted.Tensor(boundaries, values, sorter=sorter)

    boundaries = torch.tensor([1.0, 2.0, 3.0])
    values = torch.tensor([0.5])
    out_of_range_sorter = torch.tensor([0, 1, 9], dtype=torch.int64)
    with pytest.raises(
        BackendCompilerFailed, match="sorter bounds cannot be validated"
    ):
        torch.compile(fn, backend=mojo_backend)(boundaries, values, out_of_range_sorter)


@pytest.mark.parametrize("dtype", [torch.float32, torch.float64])
def test_aten_scatter_src_basic_2d(conf: Conf, dtype: torch.dtype):
    """Test aten.scatter.src basic functionality with 2D tensors"""

    def fn(self, index, src):
        return aten.scatter.src(self, dim=1, index=index, src=src)

    # Basic 2D scatter along dim=1
    self = torch.zeros(3, 5, dtype=dtype)
    src = torch.ones(3, 2, dtype=dtype)
    index = torch.tensor([[0, 2], [1, 4], [3, 0]], dtype=torch.long)

    check_outputs(fn, conf, [self, index, src])


@pytest.mark.parametrize("dtype", [torch.float32, torch.int32, torch.int64])
def test_aten_scatter_src_dim0(conf: Conf, dtype: torch.dtype):
    """Test aten.scatter.src along dimension 0"""

    def fn(self, index, src):
        return aten.scatter.src(self, dim=0, index=index, src=src)

    # Scatter along dim=0
    self = torch.zeros(4, 4, dtype=dtype)
    src = torch.ones(2, 4, dtype=dtype) * 5
    index = torch.tensor([[1, 0, 2, 3], [2, 1, 3, 0]], dtype=torch.long)

    check_outputs(fn, conf, [self, index, src])


@pytest.mark.parametrize("dtype", [torch.float32, torch.bfloat16])
def test_aten_scatter_src_3d_tensor(conf: Conf, dtype: torch.dtype):
    """Test aten.scatter.src with 3D tensor"""
    if conf.device == "cpu" and dtype == torch.float16:
        pytest.skip("float16 not supported on CPU in MAX")

    def fn(self, index, src):
        return aten.scatter.src(self, dim=1, index=index, src=src)

    # 3D scatter along middle dimension
    self = torch.zeros(2, 4, 3, dtype=dtype)
    src = torch.randn(2, 2, 3, dtype=dtype)
    index = torch.tensor(
        [[[0, 2, 1], [3, 1, 2]], [[2, 0, 3], [1, 3, 0]]], dtype=torch.long
    )

    check_outputs(fn, conf, [self, index, src])


def test_aten_scatter_src_negative_dim(conf: Conf):
    """Test aten.scatter.src with negative dimension"""

    def fn(self, index, src):
        return aten.scatter.src(self, dim=-1, index=index, src=src)

    # Negative dimension indexing (dim=-1 is last dimension)
    self = torch.zeros(3, 4, dtype=torch.float32)
    src = torch.ones(3, 2, dtype=torch.float32) * 2
    index = torch.tensor([[0, 3], [1, 2], [2, 1]], dtype=torch.long)

    check_outputs(fn, conf, [self, index, src])


def test_aten_scatter_value_basic(conf: Conf):
    """Test aten.scatter.value with scalar value - basic functionality"""

    def fn(x, index):
        return aten.scatter.value(x, 0, index, 1.0)

    # Create base tensor of zeros
    x = torch.zeros(3, 5)
    # Indices where we want to write along dimension 0
    index = torch.tensor([[0, 1, 2, 0, 0], [2, 0, 0, 1, 2]], dtype=torch.long)
    check_outputs(fn, conf, [x, index])


def test_aten_scatter_value_diagonal(conf: Conf):
    """Test aten.scatter.value with scalar value - create diagonal pattern"""

    def fn(x, index):
        return aten.scatter.value(x, 0, index, 5.0)

    x = torch.zeros(3, 3)
    index = torch.tensor([[0, 1, 2]], dtype=torch.long)
    check_outputs(fn, conf, [x, index])


@pytest.mark.parametrize(
    "dtype", [torch.float32, torch.float64, torch.int32, torch.int64]
)
def test_aten_scatter_value_dtypes(conf: Conf, dtype: torch.dtype):
    """Test aten.scatter.value with scalar value - different dtypes"""

    def fn(x, index):
        if dtype in [torch.int32, torch.int64]:
            return aten.scatter.value(x, 0, index, 7)
        else:
            return aten.scatter.value(x, 0, index, 3.5)

    if dtype in [torch.int32, torch.int64]:
        x = torch.zeros(4, 4, dtype=dtype)
    else:
        x = torch.zeros(4, 4, dtype=dtype)

    index = torch.tensor([[0, 1, 2, 3]], dtype=torch.long)
    check_outputs(fn, conf, [x, index])


def test_aten_scatter_value_dim1(conf: Conf):
    """Test aten.scatter.value with scalar value along dimension 1"""

    def fn(x, index):
        return aten.scatter.value(x, 1, index, 2.5)

    x = torch.zeros(3, 5)
    index = torch.tensor([[0, 2, 4], [1, 3, 4], [0, 1, 2]], dtype=torch.long)
    check_outputs(fn, conf, [x, index])


def test_aten_scatter_value_3d(conf: Conf):
    """Test aten.scatter.value with scalar value on 3D tensor"""

    def fn(x, index):
        return aten.scatter.value(x, 1, index, 9.0)

    x = torch.zeros(2, 3, 4)
    index = torch.tensor(
        [[[0, 1, 2, 0], [1, 2, 0, 1]], [[2, 0, 1, 2], [0, 1, 2, 1]]], dtype=torch.long
    )
    check_outputs(fn, conf, [x, index])


def test_aten_split_with_sizes_basic(conf: Conf):
    """Test aten.split_with_sizes with basic splits"""

    def fn(x):
        return aten.split_with_sizes(x, [2, 3, 5], dim=0)

    x = torch.randn(10, 4)
    check_outputs(fn, conf, [x])


@pytest.mark.parametrize("dim", [0, 1, 2])
def test_aten_split_with_sizes_different_dims(conf: Conf, dim: int):
    """Test aten.split_with_sizes on different dimensions"""

    def fn(x):
        # Adjust split sizes based on dimension
        if dim == 0:
            return aten.split_with_sizes(x, [1, 2, 1], dim=dim)  # sum=4, dim size=4
        elif dim == 1:
            return aten.split_with_sizes(x, [1, 2, 1], dim=dim)  # sum=4, dim size=4
        else:  # dim == 2
            return aten.split_with_sizes(x, [2, 2, 1], dim=dim)  # sum=5, dim size=5

    x = torch.randn(4, 4, 5)
    check_outputs(fn, conf, [x])


@pytest.mark.parametrize("dim", [-1, -2, -3])
def test_aten_split_with_sizes_negative_dims(conf: Conf, dim: int):
    """Test aten.split_with_sizes with negative dimension indices"""

    def fn(x):
        # Adjust split sizes based on dimension
        if dim == -1:  # last dim, size=5
            return aten.split_with_sizes(x, [2, 2, 1], dim=dim)
        elif dim == -2:  # middle dim, size=4
            return aten.split_with_sizes(x, [1, 2, 1], dim=dim)
        else:  # dim == -3, first dim, size=4
            return aten.split_with_sizes(x, [1, 2, 1], dim=dim)

    x = torch.randn(4, 4, 5)
    check_outputs(fn, conf, [x])


def test_aten_split_with_sizes_uneven(conf: Conf):
    """Test aten.split_with_sizes with uneven splits"""

    def fn(x):
        return aten.split_with_sizes(x, [1, 3, 2, 4], dim=1)

    x = torch.randn(3, 10)
    check_outputs(fn, conf, [x])


def test_aten_split_with_sizes_single_split(conf: Conf):
    """Test aten.split_with_sizes with single split (entire tensor)"""

    def fn(x):
        return aten.split_with_sizes(x, [5], dim=0)

    x = torch.randn(5, 3)
    check_outputs(fn, conf, [x])


@pytest.mark.parametrize("dtype", [torch.float32, torch.int32, torch.bool])
def test_aten_split_with_sizes_dtypes(conf: Conf, dtype: torch.dtype):
    """Test aten.split_with_sizes with different data types"""

    def fn(x):
        return aten.split_with_sizes(x, [2, 2, 2], dim=0)

    if dtype == torch.bool:
        x = torch.randint(0, 2, (6, 4), dtype=dtype)
    elif dtype == torch.int32:
        x = torch.randint(0, 10, (6, 4), dtype=dtype)
    else:
        x = torch.randn(6, 4, dtype=dtype)

    check_outputs(fn, conf, [x])


def test_aten_split_with_sizes_1d(conf: Conf):
    """Test aten.split_with_sizes with 1D tensor"""

    def fn(x):
        return aten.split_with_sizes(x, [3, 3, 4], dim=0)

    x = torch.randn(10)
    check_outputs(fn, conf, [x])


def test_aten_split_with_sizes_3d(conf: Conf):
    """Test aten.split_with_sizes with 3D tensor"""

    def fn(x):
        return aten.split_with_sizes(x, [1, 1, 2], dim=2)

    x = torch.randn(2, 3, 4)
    check_outputs(fn, conf, [x])


def test_aten_split_with_sizes_many_splits(conf: Conf):
    """Test aten.split_with_sizes with many small splits"""

    def fn(x):
        return aten.split_with_sizes(x, [1, 1, 1, 1, 1, 1, 1, 1], dim=0)

    x = torch.randn(8, 3)
    check_outputs(fn, conf, [x])


@pytest.mark.parametrize("split_sizes", [[2, 0, 3], [0, 5, 0], [0, 0, 5]])
def test_aten_split_with_sizes_zero_size(conf: Conf, split_sizes: list[int]):
    """Test aten.split_with_sizes with zero-sized splits"""

    def fn(x):
        return aten.split_with_sizes(x, split_sizes, dim=0)

    x = torch.randn(5, 3)
    check_outputs(fn, conf, [x])


def test_aten_split_with_sizes_exact_split(conf: Conf):
    """Test aten.split_with_sizes where sizes exactly match dimension"""

    def fn(x):
        return aten.split_with_sizes(x, [2, 2, 2, 2, 2], dim=1)

    x = torch.randn(3, 10, 4)
    check_outputs(fn, conf, [x])


@pytest.mark.parametrize("dtype", [torch.float32, torch.float16, torch.bfloat16])
def test_aten_square_basic(conf: Conf, dtype: torch.dtype):
    """Test aten.square basic functionality with different dtypes"""

    def fn(x):
        return aten.square(x)

    x = torch.randn(3, 4, dtype=dtype)
    check_outputs(fn, conf, [x])


@pytest.mark.parametrize("shape", [(5,), (3, 4), (2, 3, 4), (2, 3, 4, 5)])
def test_aten_square_different_shapes(conf: Conf, shape: tuple):
    """Test aten.square with different tensor shapes"""

    def fn(x):
        return aten.square(x)

    x = torch.randn(shape)
    check_outputs(fn, conf, [x])


def test_aten_square_scalar(conf: Conf):
    """Test aten.square with scalar (0D tensor)"""

    def fn(x):
        return aten.square(x)

    x = torch.tensor(3.5)
    check_outputs(fn, conf, [x])


def test_aten_square_negative_values(conf: Conf):
    """Test aten.square with negative values"""

    def fn(x):
        return aten.square(x)

    x = torch.tensor([-2.0, -1.5, -1.0, 0.0, 1.0, 1.5, 2.0])
    check_outputs(fn, conf, [x])


def test_aten_square_zero_tensor(conf: Conf):
    """Test aten.square with tensor of zeros"""

    def fn(x):
        return aten.square(x)

    x = torch.zeros(3, 4)
    check_outputs(fn, conf, [x])


def test_aten_squeeze_single_dim(conf: Conf):
    """Test aten.squeeze with single dimension"""

    def fn(x):
        return aten.squeeze(x, 1)

    x = torch.randn(3, 1, 5)
    check_outputs(fn, conf, [x])


@pytest.mark.parametrize("dim", [0, 1, 2, 3])
def test_aten_squeeze_different_dims(conf: Conf, dim: int):
    """Test aten.squeeze on different dimensions"""

    def fn(x):
        return aten.squeeze(x, dim)

    x = torch.randn(1, 3, 1, 5)
    check_outputs(fn, conf, [x])


def test_aten_squeeze_negative_dim(conf: Conf):
    """Test aten.squeeze with negative dimension"""

    def fn(x):
        return aten.squeeze(x, -2)

    x = torch.randn(3, 1, 5)
    check_outputs(fn, conf, [x])


def test_aten_squeeze_multiple_dims(device: str):
    """Test aten.squeeze with multiple dimensions"""

    def fn(x):
        return aten.squeeze(x, [0, 2])

    x = torch.randn(1, 3, 1, 5, device=device)
    check_functions_are_equivalent(fn, device, [x])


def test_aten_squeeze_no_change(conf: Conf):
    """Test aten.squeeze when dimension is not size 1"""

    def fn(x):
        return aten.squeeze(x, 1)

    x = torch.randn(3, 4, 5)
    check_outputs(fn, conf, [x])


@pytest.mark.parametrize("dtype", [torch.float32, torch.int32, torch.bool])
def test_aten_squeeze_different_dtypes(conf: Conf, dtype: torch.dtype):
    """Test aten.squeeze with different data types"""

    def fn(x):
        return aten.squeeze(x, 1)

    if dtype == torch.bool:
        x = torch.randint(0, 2, (3, 1, 5), dtype=dtype)
    elif dtype == torch.int32:
        x = torch.randint(0, 10, (3, 1, 5), dtype=dtype)
    else:
        x = torch.randn(3, 1, 5, dtype=dtype)

    check_outputs(fn, conf, [x])


def test_aten_squeeze_all_ones(device: str):
    """Test aten.squeeze with tensor of all size-1 dimensions"""

    def fn(x):
        return aten.squeeze(x, [0, 1, 2, 3])

    x = torch.randn(1, 1, 1, 1, device=device)
    check_functions_are_equivalent(fn, device, [x])


def test_aten_squeeze_2d(conf: Conf):
    """Test aten.squeeze with 2D tensor"""

    def fn(x):
        return aten.squeeze(x, 0)

    x = torch.randn(1, 5)
    check_outputs(fn, conf, [x])


def test_aten_squeeze_5d(device: str):
    """Test aten.squeeze with 5D tensor"""

    def fn(x):
        return aten.squeeze(x, [1, 3])

    x = torch.randn(2, 1, 3, 1, 4, device=device)
    check_functions_are_equivalent(fn, device, [x])


def test_aten_squeeze_empty_dims(device: str):
    """Test aten.squeeze with empty dimensions list"""

    def fn(x):
        return aten.squeeze(x, [])

    x = torch.randn(1, 3, 1, 5, device=device)
    check_functions_are_equivalent(fn, device, [x])


@pytest.mark.parametrize("shape", [(1,), (1, 1), (1, 1, 1)])
def test_aten_squeeze_edge_cases(device: str, shape: tuple):
    """Test aten.squeeze with edge case shapes"""

    def fn(x):
        return aten.squeeze(x, list(range(len(shape))))

    x = torch.randn(*shape, device=device)
    check_functions_are_equivalent(fn, device, [x])


def test_aten_triu_basic(conf: Conf):
    """Test aten.triu with default diagonal=0"""

    def fn(x):
        return aten.triu(x, diagonal=0)

    x = torch.randn(5, 5)
    check_outputs(fn, conf, [x])


@pytest.mark.parametrize("diagonal", [-2, -1, 0, 1, 2])
def test_aten_triu_different_diagonals(conf: Conf, diagonal: int):
    """Test aten.triu with different diagonal values"""

    def fn(x):
        return aten.triu(x, diagonal=diagonal)

    x = torch.randn(6, 6)
    check_outputs(fn, conf, [x])


@pytest.mark.parametrize("shape", [(3, 5), (5, 3), (7, 7)])
def test_aten_triu_rectangular(conf: Conf, shape: tuple):
    """Test aten.triu with rectangular matrices"""

    def fn(x):
        return aten.triu(x, diagonal=0)

    x = torch.randn(*shape)
    check_outputs(fn, conf, [x])


@pytest.mark.parametrize(
    "dtype", [torch.float32, torch.float64, torch.int32, torch.bool]
)
def test_aten_triu_different_dtypes(conf: Conf, dtype: torch.dtype):
    """Test aten.triu with different data types"""

    def fn(x):
        return aten.triu(x, diagonal=0)

    if dtype == torch.bool:
        x = torch.randint(0, 2, (4, 4), dtype=dtype)
    elif dtype == torch.int32:
        x = torch.randint(0, 10, (4, 4), dtype=dtype)
    else:
        x = torch.randn(4, 4, dtype=dtype)

    check_outputs(fn, conf, [x])


def test_aten_triu_3d(conf: Conf):
    """Test aten.triu with 3D tensor (batch of matrices)"""

    def fn(x):
        return aten.triu(x, diagonal=0)

    x = torch.randn(3, 4, 4)
    check_outputs(fn, conf, [x])


@pytest.mark.parametrize("diagonal", [-1, 0, 1])
def test_aten_triu_3d_different_diagonals(conf: Conf, diagonal: int):
    """Test aten.triu with 3D tensor and different diagonals"""

    def fn(x):
        return aten.triu(x, diagonal=diagonal)

    x = torch.randn(2, 5, 5)
    check_outputs(fn, conf, [x])


def test_aten_triu_large_diagonal(conf: Conf):
    """Test aten.triu with diagonal larger than matrix size"""

    def fn(x):
        return aten.triu(x, diagonal=10)

    x = torch.randn(5, 5)
    check_outputs(fn, conf, [x])


def test_aten_triu_negative_large_diagonal(conf: Conf):
    """Test aten.triu with large negative diagonal"""

    def fn(x):
        return aten.triu(x, diagonal=-10)

    x = torch.randn(5, 5)
    check_outputs(fn, conf, [x])


def test_aten_triu_small_matrix(conf: Conf):
    """Test aten.triu with small matrices"""

    def fn(x):
        return aten.triu(x, diagonal=0)

    x = torch.randn(2, 2)
    check_outputs(fn, conf, [x])


def test_aten_triu_single_element(conf: Conf):
    """Test aten.triu with 1x1 matrix"""

    def fn(x):
        return aten.triu(x, diagonal=0)

    x = torch.randn(1, 1)
    check_outputs(fn, conf, [x])


@pytest.mark.parametrize("diagonal", [10, -10])
def test_aten_triu_dynamic_dimensions_large_diagonal(device: str, diagonal: int):
    """Test aten.triu with dynamic dimensions and large diagonal"""

    def fn(x):
        return aten.triu(x, diagonal=diagonal)

    x = torch.randn(5, 7, device=device)
    # Mark both dimensions as dynamic
    mark_dynamic(x, 0)
    mark_dynamic(x, 1)
    check_functions_are_equivalent(fn, device, [x])


def test_aten_triu_dynamic_batch_dimension(conf: Conf):
    """Test aten.triu with dynamic batch dimension"""

    def fn(x):
        return aten.triu(x, diagonal=1)

    x = torch.randn(3, 4, 4)
    # Mark only the batch dimension as dynamic
    mark_dynamic(x, 0)
    check_outputs(fn, conf, [x])


def test_aten_logical_and_bool_tensors(conf: Conf):
    """Test aten.logical_and with boolean tensors"""

    def fn(x, y):
        return aten.logical_and(x, y)

    x = torch.tensor([True, False, True, False])
    y = torch.tensor([True, True, False, False])
    check_outputs(fn, conf, [x, y])


def test_aten_logical_and_numeric_tensors(conf: Conf):
    """Test aten.logical_and with numeric tensors (converted to bool)"""

    def fn(x, y):
        return aten.logical_and(x, y)

    x = torch.tensor([1, 0, 2, -1], dtype=torch.float32)
    y = torch.tensor([3, 0, 0, 4], dtype=torch.float32)
    check_outputs(fn, conf, [x, y])


@pytest.mark.parametrize(
    "dtype", [torch.int32, torch.int64, torch.float32, torch.float64]
)
def test_aten_logical_and_different_dtypes(conf: Conf, dtype: torch.dtype):
    """Test aten.logical_and with different numeric data types"""

    def fn(x, y):
        return aten.logical_and(x, y)

    x = torch.tensor([1, 0, 2, -1], dtype=dtype)
    y = torch.tensor([3, 0, 0, 4], dtype=dtype)
    check_outputs(fn, conf, [x, y])


def test_aten_logical_and_2d_tensors(conf: Conf):
    """Test aten.logical_and with 2D tensors"""

    def fn(x, y):
        return aten.logical_and(x, y)

    x = torch.randint(0, 2, (3, 4), dtype=torch.bool)
    y = torch.randint(0, 2, (3, 4), dtype=torch.bool)
    check_outputs(fn, conf, [x, y])


def test_aten_logical_and_broadcasting(conf: Conf):
    """Test aten.logical_and with broadcasting"""

    def fn(x, y):
        return aten.logical_and(x, y)

    x = torch.randint(0, 3, (3, 4), dtype=torch.int32)
    y = torch.randint(0, 3, (4,), dtype=torch.int32)
    check_outputs(fn, conf, [x, y])


def test_aten_logical_and_mixed_types(conf: Conf):
    """Test aten.logical_and with mixed boolean and numeric types"""

    def fn(x, y):
        return aten.logical_and(x, y)

    x = torch.tensor([True, False, True, False])
    y = torch.tensor([1, 0, 2, 0], dtype=torch.float32)
    check_outputs(fn, conf, [x, y])


def test_aten_logical_and_zeros_and_ones(conf: Conf):
    """Test aten.logical_and with patterns of zeros and ones"""

    def fn(x, y):
        return aten.logical_and(x, y)

    x = torch.tensor([0, 1, 0, 1], dtype=torch.float32)
    y = torch.tensor([0, 0, 1, 1], dtype=torch.float32)
    check_outputs(fn, conf, [x, y])


def test_aten_logical_and_all_false(conf: Conf):
    """Test aten.logical_and with all false values"""

    def fn(x, y):
        return aten.logical_and(x, y)

    x = torch.zeros(4, dtype=torch.float32)
    y = torch.zeros(4, dtype=torch.float32)
    check_outputs(fn, conf, [x, y])


def test_aten_logical_and_all_true(conf: Conf):
    """Test aten.logical_and with all true values"""

    def fn(x, y):
        return aten.logical_and(x, y)

    x = torch.ones(4, dtype=torch.float32)
    y = torch.ones(4, dtype=torch.float32)
    check_outputs(fn, conf, [x, y])


def test_aten_logical_and_negative_values(conf: Conf):
    """Test aten.logical_and with negative values (should be treated as true)"""

    def fn(x, y):
        return aten.logical_and(x, y)

    x = torch.tensor([-1, -2, 0, 3], dtype=torch.float32)
    y = torch.tensor([4, 0, -5, 6], dtype=torch.float32)
    check_outputs(fn, conf, [x, y])


def test_aten_logical_and_3d_tensors(conf: Conf):
    """Test aten.logical_and with 3D tensors"""

    def fn(x, y):
        return aten.logical_and(x, y)

    x = torch.randint(0, 2, (2, 3, 4), dtype=torch.bool)
    y = torch.randint(0, 2, (2, 3, 4), dtype=torch.bool)
    check_outputs(fn, conf, [x, y])


def test_aten_logical_and_scalar_like(conf: Conf):
    """Test aten.logical_and with scalar-like tensors"""

    def fn(x, y):
        return aten.logical_and(x, y)

    x = torch.tensor(True)
    y = torch.tensor(False)
    check_outputs(fn, conf, [x, y])


@pytest.mark.parametrize(
    "shape_pair", [((3, 1), (3, 4)), ((1, 4), (3, 4)), ((3, 1, 1), (3, 4, 5))]
)
def test_aten_logical_and_broadcasting_shapes(conf: Conf, shape_pair: tuple):
    """Test aten.logical_and with various broadcasting shapes"""

    def fn(x, y):
        return aten.logical_and(x, y)

    x_shape, y_shape = shape_pair
    x = torch.randint(0, 2, x_shape, dtype=torch.int32)
    y = torch.randint(0, 2, y_shape, dtype=torch.int32)
    check_outputs(fn, conf, [x, y])


@pytest.mark.parametrize("dim", [0, 1, 2])
@pytest.mark.parametrize("keepdim", [True, False])
def test_aten_amax_single_dim(conf: Conf, dim: int, keepdim: bool):
    """Test aten_amax with single dimension"""

    def fn(x):
        return aten.amax(x, dim=[dim], keepdim=keepdim)

    x = torch.randn(3, 4, 5)
    check_outputs(fn, conf, [x])


@pytest.mark.parametrize("dims", [[0, 1], [1, 2], [0, 2]])
@pytest.mark.parametrize("keepdim", [True, False])
def test_aten_amax_multiple_dims(conf: Conf, dims: list[int], keepdim: bool):
    """Test aten_amax with multiple dimensions"""

    def fn(x):
        return aten.amax(x, dim=dims, keepdim=keepdim)

    x = torch.randn(3, 4, 5)
    check_outputs(fn, conf, [x])


def test_aten_max_no_dim(conf: Conf):
    """Test aten_max without dimension (returns single value)"""

    def fn(x):
        return aten.max(x)

    x = torch.randn(3, 4, 5)
    check_outputs(fn, conf, [x])


@pytest.mark.parametrize("dim", [0, 1, 2])
@pytest.mark.parametrize("keepdim", [True, False])
def test_aten_max_with_dim(device: str, dim: int, keepdim: bool):
    """Test aten_max with dimension (returns values and indices tuple)"""

    def fn(x):
        return aten.max(x, dim=dim, keepdim=keepdim)

    x = torch.randn(3, 4, 5, device=device)
    check_functions_are_equivalent(fn, device, [x])


@pytest.mark.parametrize("dtype", [torch.int32, torch.int64, torch.float32])
def test_aten_max_different_dtypes(device: str, dtype: torch.dtype):
    """Test aten_max with different data types"""

    def fn(x):
        return aten.max(x, dim=1, keepdim=False)

    if dtype.is_floating_point:
        x = torch.randn(3, 4, dtype=dtype, device=device)
    else:
        x = torch.randint(-10, 10, (3, 4), dtype=dtype, device=device)

    check_functions_are_equivalent(fn, device, [x])


@pytest.mark.parametrize("dtype", [torch.float32, torch.bfloat16])
def test_aten_amin_all_dims(conf: Conf, dtype: torch.dtype):
    """Test aten_amin with default empty dim list (reduces over all dimensions)"""
    # Skip float16 on CPU as MAX doesn't support f16 on CPU
    if conf.device == "cpu" and dtype == torch.float16:
        pytest.skip("float16 not supported on CPU in MAX")

    def fn(x):
        return aten.amin(x)

    # Test with different shapes
    x = torch.randn(3, 4, 5, dtype=dtype)
    check_outputs(fn, conf, [x])

    # Test with 1D tensor
    x1d = torch.randn(10, dtype=dtype)
    check_outputs(fn, conf, [x1d])


@pytest.mark.parametrize("dim", [0, 1, 2])
@pytest.mark.parametrize("keepdim", [True, False])
def test_aten_amin_single_dim(conf: Conf, dim: int, keepdim: bool):
    """Test aten_amin with single dimension"""

    def fn(x):
        return aten.amin(x, dim=[dim], keepdim=keepdim)

    x = torch.randn(3, 4, 5)
    check_outputs(fn, conf, [x])


@pytest.mark.parametrize("dims", [[0, 1], [1, 2], [0, 2]])
@pytest.mark.parametrize("keepdim", [True, False])
def test_aten_amin_multiple_dims(
    conf: Conf, dims: list[int], keepdim: bool, call_checker: CallChecker
):
    """Test aten_amin with multiple dimensions"""
    call_checker.register(aten_functions.aten_amin)

    def fn(x):
        return aten.amin(x, dim=dims, keepdim=keepdim)

    x = torch.randn(3, 4, 5)
    check_outputs(fn, conf, [x])


def test_aten_min_no_dim(conf: Conf, call_checker: CallChecker):
    """Test aten_min without dimension (returns single value)"""
    call_checker.register(aten_functions.aten_min)

    def fn(x):
        return aten.min(x)

    x = torch.randn(3, 4, 5)
    check_outputs(fn, conf, [x])


@pytest.mark.parametrize("dim", [0, 1, 2])
@pytest.mark.parametrize("keepdim", [True, False])
def test_aten_min_with_dim(
    conf: Conf, dim: int, keepdim: bool, call_checker: CallChecker
):
    """Test aten_min with dimension (returns values and indices tuple)"""
    call_checker.register(aten_functions.aten_min)

    def fn(x):
        return aten.min(x, dim=dim, keepdim=keepdim)

    x = torch.randn(3, 4, 5)
    check_outputs(fn, conf, [x])


@pytest.mark.parametrize("dtype", [torch.int32, torch.int64, torch.float32])
def test_aten_min_different_dtypes(
    conf: Conf, dtype: torch.dtype, call_checker: CallChecker
):
    """Test aten_min with different data types"""
    call_checker.register(aten_functions.aten_min)

    def fn(x):
        return aten.min(x, dim=1, keepdim=False)

    if dtype.is_floating_point:
        x = torch.randn(3, 4, dtype=dtype)
    else:
        x = torch.randint(-10, 10, (3, 4), dtype=dtype)

    check_outputs(fn, conf, [x])


@pytest.mark.parametrize("dtype", [torch.bool, torch.uint8, torch.int32, torch.int64])
def test_aten_sum_integral_dtypes(
    conf: Conf, dtype: torch.dtype, call_checker: CallChecker
):
    """Test aten.sum over integral dtypes (torch promotes bool/small-int sums to int64)"""
    call_checker.register(aten_functions.aten_sum)

    def fn(x):
        return aten.sum(x)

    x = (torch.arange(12) % 2).reshape(3, 4).to(dtype)
    check_outputs(fn, conf, [x])


@pytest.mark.parametrize("with_bias", [True, False])
def test_aten_linear(conf: Conf, with_bias: bool, call_checker: CallChecker):
    """Test aten.linear (input @ weight^T + bias) with 2D input"""
    call_checker.register(aten_functions.aten_linear)

    if with_bias:

        def fn(x, w, b):
            return aten.linear(x, w, b)

        inputs = [torch.randn(5, 16), torch.randn(9, 16), torch.randn(9)]
    else:

        def fn(x, w):
            return aten.linear(x, w)

        inputs = [torch.randn(5, 16), torch.randn(9, 16)]
    check_outputs(fn, conf, inputs)


@pytest.mark.parametrize("with_bias", [False, True])
def test_aten_linear_1d(conf: Conf, with_bias: bool, call_checker: CallChecker):
    call_checker.register(aten_functions.aten_linear)

    if with_bias:

        def fn(x, w, b):
            return aten.linear(x, w, b)

        inputs = [torch.randn(16), torch.randn(9, 16), torch.randn(9)]
    else:

        def fn(x, w):
            return aten.linear(x, w)

        inputs = [torch.randn(16), torch.randn(9, 16)]

    check_outputs(fn, conf, inputs)


@pytest.mark.parametrize(
    ("in_features", "out_features", "with_bias"),
    [(0, 5, False), (0, 5, True), (7, 0, False), (7, 0, True)],
)
def test_aten_linear_1d_degenerate(
    conf: Conf,
    call_checker: CallChecker,
    in_features: int,
    out_features: int,
    with_bias: bool,
):
    call_checker.register(aten_functions.aten_linear)

    if with_bias:

        def fn(x, w, b):
            return aten.linear(x, w, b)

        inputs = [
            torch.randn(in_features),
            torch.randn(out_features, in_features),
            torch.randn(out_features),
        ]
    else:

        def fn(x, w):
            return aten.linear(x, w)

        inputs = [torch.randn(in_features), torch.randn(out_features, in_features)]

    check_outputs(fn, conf, inputs)


def test_aten_linear_3d_input(conf: Conf, call_checker: CallChecker):
    """Test aten.linear with batched (3D) input, as transformers call it"""
    call_checker.register(aten_functions.aten_linear)

    def fn(x, w, b):
        return aten.linear(x, w, b)

    inputs = [torch.randn(3, 7, 16), torch.randn(11, 16), torch.randn(11)]
    check_outputs(fn, conf, inputs)


@pytest.mark.parametrize("input_shape", [(16,), (5, 16), (3, 7, 16)])
def test_aten_linear_backward(
    conf: Conf, call_checker: CallChecker, input_shape: tuple[int, ...]
):
    call_checker.register(aten_functions.aten_linear_backward)
    out_features = 11
    input = torch.randn(input_shape)
    weight = torch.randn(out_features, input_shape[-1])
    grad_output = torch.randn((*input_shape[:-1], out_features))
    rows = input.numel() // input_shape[-1]
    input_matrix = input.reshape(rows, input_shape[-1])
    grad_matrix = grad_output.reshape(rows, out_features)
    expected = (
        (grad_matrix @ weight).reshape_as(input),
        grad_matrix.t() @ input_matrix,
        grad_matrix.sum(dim=0),
    )

    actual = aten.linear_backward(
        input.to(conf.device),
        grad_output.to(conf.device),
        weight.to(conf.device),
        [True, True, True],
    )

    for got, want in zip(actual, expected, strict=True):
        torch.testing.assert_close(got.cpu(), want)


@pytest.mark.parametrize(
    "output_mask",
    [
        (True, False, False),
        (False, True, False),
        (False, False, True),
        (True, True, True),
        (False, False, False),
    ],
)
def test_aten_linear_backward_output_mask(
    conf: Conf, call_checker: CallChecker, output_mask: tuple[bool, bool, bool]
):
    call_checker.register(aten_functions.aten_linear_backward)
    input = torch.randn(5, 7)
    weight = torch.randn(11, 7)
    grad_output = torch.randn(5, 11)
    all_expected = (
        grad_output @ weight,
        grad_output.t() @ input,
        grad_output.sum(dim=0),
    )

    actual = aten.linear_backward(
        input.to(conf.device),
        grad_output.to(conf.device),
        weight.to(conf.device),
        output_mask,
    )

    need_parameter_grads = output_mask[1] or output_mask[2]
    defined_mask = (output_mask[0], need_parameter_grads, need_parameter_grads)
    for index, (got, want, defined) in enumerate(
        zip(actual, all_expected, defined_mask, strict=True)
    ):
        assert (got is not None) == defined
        if defined and (index != 2 or output_mask[2]):
            torch.testing.assert_close(got.cpu(), want)
        elif defined:
            assert got.shape == want.shape


def test_aten_linear_backward_empty_batch(conf: Conf, call_checker: CallChecker):
    call_checker.register(aten_functions.aten_linear_backward)
    input = torch.empty(0, 7)
    weight = torch.randn(11, 7)
    grad_output = torch.empty(0, 11)

    actual = aten.linear_backward(
        input.to(conf.device),
        grad_output.to(conf.device),
        weight.to(conf.device),
        [True, True, True],
    )

    expected = (torch.empty_like(input), torch.zeros_like(weight), torch.zeros(11))
    for got, want in zip(actual, expected, strict=True):
        torch.testing.assert_close(got.cpu(), want)


def test_aten_linear_backward_noncontiguous(conf: Conf, call_checker: CallChecker):
    call_checker.register(aten_functions.aten_linear_backward)
    input = torch.randn(7, 5).t()
    weight = torch.randn(7, 11).t()
    grad_output = torch.randn(11, 5).t()
    assert not input.is_contiguous()
    assert not weight.is_contiguous()
    assert not grad_output.is_contiguous()
    expected = (grad_output @ weight, grad_output.t() @ input, grad_output.sum(dim=0))

    actual = aten.linear_backward(
        input.to(conf.device),
        grad_output.to(conf.device),
        weight.to(conf.device),
        [True, True, True],
    )

    for got, want in zip(actual, expected, strict=True):
        torch.testing.assert_close(got.cpu(), want)


@pytest.mark.parametrize(
    ("input_shape", "out_features"), [((0,), 5), ((3, 0), 5), ((3, 7), 0)]
)
def test_aten_linear_backward_degenerate_features(
    conf: Conf,
    call_checker: CallChecker,
    input_shape: tuple[int, ...],
    out_features: int,
):
    call_checker.register(aten_functions.aten_linear_backward)
    input = torch.randn(input_shape)
    weight = torch.randn(out_features, input_shape[-1])
    grad_output = torch.randn((*input_shape[:-1], out_features))
    rows = 1 if len(input_shape) == 1 else math.prod(input_shape[:-1])
    input_matrix = input.reshape(rows, input_shape[-1])
    grad_matrix = grad_output.reshape(rows, out_features)
    expected = (
        (grad_matrix @ weight).reshape_as(input),
        grad_matrix.t() @ input_matrix,
        grad_matrix.sum(dim=0),
    )

    actual = aten.linear_backward(
        input.to(conf.device),
        grad_output.to(conf.device),
        weight.to(conf.device),
        [True, True, True],
    )

    for got, want in zip(actual, expected, strict=True):
        torch.testing.assert_close(got.cpu(), want)


@pytest.mark.parametrize("dtype", [torch.float32, torch.float64])
def test_aten_var_no_dim(conf: Conf, dtype: torch.dtype, call_checker: CallChecker):
    """Test aten.var over all dimensions (default correction=1)"""
    call_checker.register(aten_functions.aten_var)

    def fn(x):
        return torch.var(x)

    x = torch.randn(3, 4, 5, dtype=dtype)
    check_outputs(fn, conf, [x])


@pytest.mark.parametrize("dim", [0, 1, 2, -1])
@pytest.mark.parametrize("keepdim", [True, False])
def test_aten_var_with_dim(
    conf: Conf, dim: int, keepdim: bool, call_checker: CallChecker
):
    """Test aten.var with a single dimension"""
    call_checker.register(aten_functions.aten_var)

    def fn(x):
        return torch.var(x, dim=dim, keepdim=keepdim)

    x = torch.randn(3, 4, 5)
    check_outputs(fn, conf, [x])


@pytest.mark.parametrize("correction", [0, 1, 2])
def test_aten_var_correction(conf: Conf, correction: int, call_checker: CallChecker):
    """Test aten.var with different correction values (0 = population, 1 = Bessel)"""
    call_checker.register(aten_functions.aten_var)

    def fn(x):
        return torch.var(x, dim=1, correction=correction, keepdim=False)

    x = torch.randn(3, 5, 4)
    check_outputs(fn, conf, [x])


@pytest.mark.parametrize("dim", [(0, 1), (1, 2), (0, 2)])
@pytest.mark.parametrize("keepdim", [True, False])
def test_aten_var_multi_dim(
    conf: Conf, dim: tuple, keepdim: bool, call_checker: CallChecker
):
    """Test aten.var reducing over multiple dimensions"""
    call_checker.register(aten_functions.aten_var)

    def fn(x):
        return torch.var(x, dim=list(dim), keepdim=keepdim)

    x = torch.randn(3, 4, 5)
    check_outputs(fn, conf, [x])


def test_aten_var_correction_zero_no_dim(conf: Conf, call_checker: CallChecker):
    """Test aten.var over all dims with correction=0 (population variance)"""
    call_checker.register(aten_functions.aten_var)

    def fn(x):
        return torch.var(x, correction=0)

    x = torch.randn(3, 4)
    check_outputs(fn, conf, [x])


@pytest.mark.parametrize("dtype", [torch.float32, torch.float64])
@pytest.mark.parametrize("shape", [(5,), (3, 4), (2, 3, 4)])
def test_aten_silu(
    conf: Conf, dtype: torch.dtype, shape: tuple, call_checker: CallChecker
):
    """Test aten.silu (x * sigmoid(x)) across dtypes and shapes"""
    call_checker.register(aten_functions.aten_silu)

    def fn(x):
        return aten.silu(x)

    x = torch.randn(shape, dtype=dtype)
    check_outputs(fn, conf, [x])


@pytest.mark.parametrize("dtype", [torch.float32, torch.float64])
@pytest.mark.parametrize("shape", [(2, 3), (1, 4, 4)])
@pytest.mark.parametrize("value", [-1.5, 42])
def test_fill_scalar_basic(
    conf: Conf,
    dtype: torch.dtype,
    shape: tuple,
    value: float,
    call_checker: CallChecker,
):
    """Test basic fill.Scalar functionality with different dtypes, shapes, and values"""
    from torch_mojo_backend.eager_kernels import aten_fast

    call_checker.register(
        aten_functions.aten_fill_scalar, aten_fast.fast_aten_fill_scalar
    )

    def fn(x):
        return aten.fill.Scalar(x, value)

    # Create input tensor
    x = torch.randn(shape, dtype=dtype)

    check_outputs(fn, conf, [x])


@pytest.mark.parametrize("dtype", [torch.int32, torch.int64])
@pytest.mark.parametrize("shape", [(2, 3), (1, 4, 4)])
@pytest.mark.parametrize("value", [-5, 42])
def test_fill_scalar_integer_dtypes(
    conf: Conf, dtype: torch.dtype, shape: tuple, value: int, call_checker: CallChecker
):
    """Test fill.Scalar functionality with integer dtypes"""
    from torch_mojo_backend.eager_kernels import aten_fast

    call_checker.register(
        aten_functions.aten_fill_scalar, aten_fast.fast_aten_fill_scalar
    )

    def fn(x):
        return aten.fill.Scalar(x, value)

    # Create input tensor with integer values
    x = torch.zeros(shape, dtype=dtype) + 1

    check_outputs(fn, conf, [x])


@pytest.mark.parametrize("value", [-5, 100])
def test_fill_scalar_integer_values(conf: Conf, value: int, call_checker: CallChecker):
    """Test fill.Scalar with integer values"""
    from torch_mojo_backend.eager_kernels import aten_fast

    call_checker.register(
        aten_functions.aten_fill_scalar, aten_fast.fast_aten_fill_scalar
    )

    def fn(x):
        return aten.fill.Scalar(x, value)

    # Test with float tensor
    x = torch.randn(3, 4)

    check_outputs(fn, conf, [x])


def test_fill_scalar_single_element(conf: Conf, call_checker: CallChecker):
    """Test fill.Scalar with single element tensor"""
    from torch_mojo_backend.eager_kernels import aten_fast

    call_checker.register(
        aten_functions.aten_fill_scalar, aten_fast.fast_aten_fill_scalar
    )

    def fn(x):
        return torch.ops.aten.fill.Scalar(x, 7.5)

    # Single element tensor
    x = torch.tensor([1.0])

    check_outputs(fn, conf, [x])


def test_fill_scalar_zero_dim(conf: Conf):
    """Test fill.Scalar with single element tensor"""

    def fn(x):
        return torch.ops.aten.fill.Scalar(x, 7.5)

    # Single element tensor
    x = torch.tensor(1.0)

    check_outputs(fn, conf, [x])


def test_fill__scalar_inplace(conf: Conf, call_checker: CallChecker):
    """Test fill_.Scalar fills tensor in-place"""
    from torch_mojo_backend.eager_kernels import aten_fast

    call_checker.register(
        aten_functions.aten_fill__scalar, aten_fast.fast_aten_fill__scalar
    )

    def fn(x):
        aten.fill_(x, 3.5)
        return x

    x = torch.zeros(3, 4)
    check_outputs(fn, conf, [x])


@pytest.mark.xfail(reason="Fixme, currently off to support eager mode")
def test_max_pool2d_error_message_not_supported_output(conf: Conf):
    def fn(x):
        return aten.max_pool2d_with_indices(x, kernel_size=2, stride=2)

    # Test different sizes
    batch_size, channels = 1, 2
    x = torch.randn(batch_size, channels, 16, 16)
    with pytest.raises(
        BackendCompilerFailed,
        match="The implementation of aten.max_pool2d_with_indices doesn't support returning indices yet.",
    ):
        check_outputs(fn, conf, [x])


@pytest.mark.xfail(reason="Fixme, currently off to support eager mode")
def test_max_pool2d_error_message_not_supported_in_graph(conf: Conf):
    def fn(x):
        return aten.max_pool2d_with_indices(x, kernel_size=2, stride=2)[1] * 2

    # Test different sizes
    batch_size, channels = 1, 2
    x = torch.randn(batch_size, channels, 16, 16)
    with pytest.raises(
        BackendCompilerFailed,
        match="The implementation of aten.max_pool2d_with_indices doesn't support returning indices yet.",
    ):
        check_outputs(fn, conf, [x])


# aten._log_softmax(Tensor self, int dim, bool half_to_float) -> Tensor
def test_log_softmax_basic(conf: Conf):
    """Test _log_softmax basic functionality."""

    def fn(x):
        return aten._log_softmax(x, -1, False)

    x = torch.randn(3, 4, 5)
    check_outputs(fn, conf, [x])


@pytest.mark.parametrize("dtype", [torch.float32, torch.float16, torch.bfloat16])
@pytest.mark.parametrize(
    ("shape", "dim"),
    [
        ((3, 4, 5), -1),
        ((3, 4, 5), 1),
        ((), -1),
        ((2, 0, 5), 1),
        ((2, 2, 1, 2, 1), 1),
        ((2, 1, 2, 1, 2, 1, 2, 1), 6),
    ],
)
def test_aten__log_softmax_backward_data(
    conf: Conf,
    call_checker: CallChecker,
    dtype: torch.dtype,
    shape: tuple[int, ...],
    dim: int,
):
    call_checker.register(aten_functions.aten__log_softmax_backward_data)

    def fn(grad_output, output):
        return aten._log_softmax_backward_data(grad_output, output, dim, dtype)

    source = torch.randn(shape, dtype=dtype)
    output = torch.log_softmax(source, dim=dim)
    grad_output = torch.randn_like(output)
    tolerance = 2e-2 if dtype == torch.bfloat16 else 2e-3
    check_outputs(fn, conf, [grad_output, output], atol=tolerance, rtol=tolerance)


def test_aten__log_softmax_backward_data_noncontiguous(
    conf: Conf, call_checker: CallChecker
):
    call_checker.register(aten_functions.aten__log_softmax_backward_data)

    def fn(grad_output, output):
        return aten._log_softmax_backward_data(grad_output, output, 1, torch.float32)

    source = torch.randn(2, 3, 2, 4, 5)
    output = torch.log_softmax(source, dim=1).transpose(0, 4)
    grad_output = torch.randn_like(source).transpose(0, 4)
    assert not output.is_contiguous()
    assert not grad_output.is_contiguous()
    check_outputs(fn, conf, [grad_output, output])


@pytest.mark.parametrize("grad_value", [float("inf"), -float("inf"), float("nan")])
def test_aten__log_softmax_backward_data_scalar_nonfinite(
    conf: Conf, call_checker: CallChecker, grad_value: float
):
    call_checker.register(aten_functions.aten__log_softmax_backward_data)
    grad_output = torch.tensor(grad_value, device=conf.device)
    output = torch.tensor(0.0, device=conf.device)

    actual = aten._log_softmax_backward_data(grad_output, output, -1, torch.float32)

    assert torch.isnan(actual.cpu())


def test_aten__log_softmax_backward_data_half_to_float(
    conf: Conf, call_checker: CallChecker
):
    call_checker.register(aten_functions.aten__log_softmax_backward_data)
    source = torch.randn(3, 7)
    output = torch.log_softmax(source, dim=-1)
    grad_output = torch.randn_like(output)
    expected = (grad_output - output.exp() * grad_output.sum(dim=-1, keepdim=True)).to(
        torch.float16
    )

    actual = aten._log_softmax_backward_data(
        grad_output.to(conf.device), output.to(conf.device), -1, torch.float16
    )

    assert actual.dtype == torch.float16
    torch.testing.assert_close(
        actual.cpu().float(), expected.float(), atol=2e-3, rtol=2e-3
    )


def test_log_softmax_numerical_stability(conf: Conf):
    """Test _log_softmax with large values to verify numerical stability."""

    def fn(x):
        return aten._log_softmax(x, -1, False)

    # Create tensor with large values that could cause overflow without max subtraction
    x = torch.randn(2, 3, dtype=torch.float32) * 100
    check_outputs(fn, conf, [x])


@pytest.mark.parametrize("dim", [-1, 0, 1])
def test_log_softmax_half_to_float_true(conf: Conf, dim: int):
    """Test _log_softmax with half_to_float=True.

    When half_to_float=True:
    - Input must be float16
    - Computation is done in float32
    - Output is float32 (not converted back to float16)
    """
    if not torch.cuda.is_available():
        pytest.skip(
            "CUDA is required for half_to_float=True tests"
            " as the cpu does not have a reference implementation."
        )

    def fn(x):
        initial_device = x.device
        if x.device.type == "cpu" and not torch.compiler.is_compiling():
            # We're in the reference eager cpu execution, which doesn't work on
            # cpu. We move the computation to cuda for reference.
            x = x.to("cuda")

        output = aten._log_softmax(x, dim, True)
        return output.to(initial_device)

    # half_to_float=True requires float16 input
    x = torch.randn(3, 4, 5, dtype=torch.float16)
    check_outputs(fn, conf, [x])


@pytest.mark.parametrize("dtype", [torch.float16, torch.float32])
@pytest.mark.parametrize("dim", [-1, 0])
def test_log_softmax_half_to_float_false(conf: Conf, dtype: torch.dtype, dim: int):
    """Test _log_softmax with half_to_float=False.

    When half_to_float=False:
    - Input can be any dtype
    - Output dtype matches input dtype
    - For float16 inputs, computation happens in float32 but result is converted back
    """

    def fn(x):
        return aten._log_softmax(x, dim, False)

    x = torch.randn(3, 4, 5, dtype=dtype)
    check_outputs(fn, conf, [x], atol=1e-3, rtol=1e-2)


@pytest.mark.parametrize("dtype", [torch.float32, torch.float16])
def test_aten_erf_basic(conf: Conf, dtype: torch.dtype):
    """Test aten.erf basic functionality with different dtypes"""

    def fn(x):
        return aten.erf(x)

    x = torch.randn(3, 4, dtype=dtype)
    check_outputs(fn, conf, [x])


def test_aten__unsafe_view(conf: Conf, call_checker: CallChecker):
    from torch_mojo_backend.eager_kernels import aten_fast

    call_checker.register(
        aten_functions.aten__unsafe_view, aten_fast.fast_aten__unsafe_view
    )

    def fn(x):
        return aten._unsafe_view(x, [2, 6])

    x = torch.randn(3, 4)
    check_outputs(fn, conf, [x])


@pytest.mark.parametrize("dtype", [torch.float32, torch.float16])
def test_aten__unsafe_view_dtypes(
    conf: Conf, dtype: torch.dtype, call_checker: CallChecker
):
    from torch_mojo_backend.eager_kernels import aten_fast

    call_checker.register(
        aten_functions.aten__unsafe_view, aten_fast.fast_aten__unsafe_view
    )

    def fn(x):
        return aten._unsafe_view(x, [4, -1])

    x = torch.randn(2, 3, 4, dtype=dtype)
    check_outputs(fn, conf, [x])
