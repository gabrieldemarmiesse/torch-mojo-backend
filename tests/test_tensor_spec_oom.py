"""Host-only regression tests for TensorSpec allocator-error propagation."""

from pathlib import Path
from types import ModuleType

import pytest
import torch
from max.driver import CPU

from torch_mojo_backend import eager_kernels
from torch_mojo_backend.eager_kernels import aten_fast
from torch_mojo_backend.mojo_device.torch_mojo_tensor import TorchMojoTensor

_CUDA_OOM = "CUDA call failed: CUDA_ERROR_OUT_OF_MEMORY (out of memory)"


@pytest.mark.parametrize(
    "path",
    [
        "binary",
        "binary_cast",
        "binary_fill_rhs",
        "binary_fill_lhs",
        "unary",
        "reduce",
        "matmul",
        "scalar",
        "int_scalar",
        "logical",
        "batch_norm",
        "min_dim",
        "attention_decode",
    ],
)
def test_tensor_spec_fallbacks_propagate_device_oom(
    monkeypatch, path, fake_mojo_tensor
):
    """Fallback-only errors stay recoverable, but allocator OOM never does."""
    device = CPU()
    _tensor = fake_mojo_tensor
    lhs = _tensor(device)
    rhs = _tensor(device)
    matmul_rhs = _tensor(device, shape=(3, 2))
    bool_lhs = _tensor(device, dtype=aten_fast.DType.bool)
    int_lhs = _tensor(device, dtype=aten_fast.DType.int64)
    query = _tensor(device, shape=(1, 1, 1, 4), strides=(4, 4, 4, 1))
    key = _tensor(device, shape=(1, 1, 2, 4), strides=(8, 8, 4, 1))
    value = _tensor(device, shape=(1, 1, 2, 4), strides=(8, 8, 4, 1))
    tensors = (lhs, rhs, matmul_rhs, bool_lhs, int_lhs, query, key, value)

    def as_tensor(candidate: object) -> object | None:
        return candidate if any(candidate is tensor for tensor in tensors) else None

    def raise_allocator_oom(*_args: object, **_kwargs: object) -> None:
        raise NotImplementedError(_CUDA_OOM)

    def load_oom_module(
        _mojo_file: Path, _defines: eager_kernels.CanonicalDefines
    ) -> ModuleType:
        module = ModuleType("oom_test_extension")
        module.call = raise_allocator_oom
        return module

    def fake_allocate_output(
        output_spec: aten_fast._TensorOutputSpec,
    ) -> TorchMojoTensor:
        return _tensor(
            output_spec.device, dtype=output_spec.dtype, shape=output_spec.shape
        )

    def fake_alloc(
        shape: tuple[int, ...], dtype: aten_fast.DType, actual_device: object
    ) -> TorchMojoTensor:
        return _tensor(actual_device, dtype=dtype, shape=shape)

    monkeypatch.setattr(aten_fast, "_t", as_tensor)
    monkeypatch.setattr(aten_fast, "_on_gpu", lambda _tensor: True)
    monkeypatch.setattr(aten_fast, "_spec_of", lambda tensor: tensor)
    monkeypatch.setattr(aten_fast, "_ctx_ptr", lambda _device: 1)
    monkeypatch.setattr(aten_fast, "_allocate_output_spec", fake_allocate_output)
    monkeypatch.setattr(aten_fast, "_alloc", fake_alloc)
    monkeypatch.setattr(
        eager_kernels.MOJO_EXTENSION_LOADER, "load_canonical", load_oom_module
    )
    calls = {
        "binary": lambda: aten_fast._try_spec_binary("SubSpec", lhs, rhs),
        "binary_cast": lambda: aten_fast._try_spec_binary("SubSpec", bool_lhs, rhs),
        "binary_fill_rhs": lambda: aten_fast._try_spec_binary("SubSpec", lhs, 1.0),
        "binary_fill_lhs": lambda: aten_fast._try_spec_binary("SubSpec", 1.0, rhs),
        "unary": lambda: aten_fast._try_spec_unary("NegSpec", lhs),
        "reduce": lambda: aten_fast._try_spec_reduce("SumSpec", lhs, (1,), False),
        "matmul": lambda: aten_fast._try_spec_matmul(
            "MatmulSpec", (lhs, matmul_rhs), 0
        ),
        "scalar": lambda: aten_fast._try_spec_scalar("AddScalarSpec", lhs, 1.0),
        "int_scalar": lambda: aten_fast._try_spec_int_scalar(
            "AddScalarIntSpec", int_lhs, 1
        ),
        "logical": lambda: aten_fast._try_logical("LogicalAndSpec", bool_lhs, bool_lhs),
        "batch_norm": lambda: aten_fast._fast_batch_norm_inference(
            lhs, lhs, lhs, lhs, lhs, 1e-5
        ),
        "min_dim": lambda: aten_fast.fast_aten_min_dim(lhs, 1),
        "attention_decode": lambda: aten_fast.fast_aten_scaled_dot_product_attention(
            query, key, value
        ),
    }

    with pytest.raises(torch.OutOfMemoryError, match="CUDA_ERROR_OUT_OF_MEMORY"):
        calls[path]()


def test_tensor_spec_unsupported_metadata_still_uses_fallback(
    monkeypatch, fake_mojo_tensor
):
    device = CPU()
    tensor = fake_mojo_tensor(device)

    def raise_unsupported(*_args: object, **_kwargs: object) -> None:
        raise NotImplementedError("mojo spec neg: strided input is unsupported")

    def load_unsupported_module(
        _mojo_file: Path, _defines: eager_kernels.CanonicalDefines
    ) -> ModuleType:
        module = ModuleType("unsupported_test_extension")
        module.call = raise_unsupported
        return module

    def fake_allocate_output(
        output_spec: aten_fast._TensorOutputSpec,
    ) -> TorchMojoTensor:
        return fake_mojo_tensor(
            output_spec.device, dtype=output_spec.dtype, shape=output_spec.shape
        )

    monkeypatch.setattr(aten_fast, "_t", lambda _candidate: tensor)
    monkeypatch.setattr(aten_fast, "_spec_of", lambda _tensor: object())
    monkeypatch.setattr(aten_fast, "_allocate_output_spec", fake_allocate_output)
    monkeypatch.setattr(
        eager_kernels.MOJO_EXTENSION_LOADER, "load_canonical", load_unsupported_module
    )

    assert aten_fast._try_spec_unary("NegSpec", tensor) is None
