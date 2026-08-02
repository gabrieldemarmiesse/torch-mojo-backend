"""Eager out= resize, aliasing, and dtype-policy regressions."""

import pytest
import torch

from torch_mojo_backend.testing import CallChecker

pytestmark = pytest.mark.xdist_group(name="group1")


def _watch_eager_op(call_checker: CallChecker, op_name: str) -> None:
    from torch_mojo_backend.mojo_device.mojo_device_aten_ops import EAGER_CALL_COUNTERS

    call_checker.register(EAGER_CALL_COUNTERS[op_name])


@pytest.mark.parametrize("storage_offset", [0, 2])
def test_mul_out_resize_preserves_existing_storage_alias(
    mojo_gpu: str, call_checker: CallChecker, storage_offset: int
):
    _watch_eager_op(call_checker, "aten::mul.out")
    base = torch.arange(8, dtype=torch.float32).to(mojo_gpu)
    out = base[storage_offset:storage_offset]
    lhs = torch.tensor([2.0, 3.0], device=mojo_gpu)
    rhs = torch.tensor([5.0, 7.0], device=mojo_gpu)
    holder = base._holder

    returned = torch.ops.aten.mul.out(lhs, rhs, out=out)

    assert returned is out
    assert out._holder is holder is base._holder
    assert out.shape == (2,)
    assert torch.numel(out) == 2
    torch.testing.assert_close(out.cpu(), torch.tensor([10.0, 21.0]))
    expected_base = torch.arange(8, dtype=torch.float32)
    expected_base[storage_offset : storage_offset + 2] = torch.tensor([10.0, 21.0])
    torch.testing.assert_close(base.cpu(), expected_base)


def test_resized_out_invalidates_cached_tensor_spec(
    mojo_gpu: str, call_checker: CallChecker
):
    """A spec operation after resize must use the new pointer and shape."""
    _watch_eager_op(call_checker, "aten::mul.out")
    from torch_mojo_backend.eager_kernels.aten_fast import _spec_of

    out = torch.empty((), dtype=torch.float32, device=mojo_gpu)
    _spec_of(out)
    assert "_spec" in out.__dict__

    lhs = torch.tensor([2.0, 3.0], device=mojo_gpu)
    rhs = torch.tensor([5.0, 7.0], device=mojo_gpu)
    torch.ops.aten.mul.out(lhs, rhs, out=out)

    assert "_spec" not in out.__dict__
    assert out.shape == (2,)
    incremented = torch.add(out, 1.0)
    torch.testing.assert_close(incremented.cpu(), torch.tensor([11.0, 22.0]))


@pytest.mark.parametrize(
    ("op_name", "valid_dtype", "invalid_dtype"),
    [
        ("aten::any.out", torch.uint8, torch.int64),
        ("aten::isin.Tensor_Tensor_out", torch.bool, torch.int64),
    ],
)
def test_out_variants_enforce_operation_specific_dtype_contracts(
    mojo_gpu: str,
    call_checker: CallChecker,
    op_name: str,
    valid_dtype: torch.dtype,
    invalid_dtype: torch.dtype,
):
    _watch_eager_op(call_checker, op_name)
    if op_name == "aten::any.out":
        input = torch.tensor([[0, 2, 0], [0, 0, 0]], device=mojo_gpu)

        def invoke(out):
            return torch.ops.aten.any.out(input, 1, False, out=out)

        expected = torch.tensor([1, 0], dtype=valid_dtype)
    else:
        input = torch.tensor([1, 2, 3], device=mojo_gpu)
        test_elements = torch.tensor([2, 4], device=mojo_gpu)

        def invoke(out):
            return torch.ops.aten.isin.Tensor_Tensor_out(
                input, test_elements, assume_unique=False, invert=False, out=out
            )

        expected = torch.tensor([False, True, False], dtype=valid_dtype)

    valid_out = torch.empty(0, dtype=valid_dtype, device=mojo_gpu)
    assert invoke(valid_out) is valid_out
    torch.testing.assert_close(valid_out.cpu(), expected)

    invalid_out = torch.empty(0, dtype=invalid_dtype, device=mojo_gpu)
    with pytest.raises(RuntimeError):
        invoke(invalid_out)
