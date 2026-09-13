"""Eager `out=` resize, aliasing, and dtype-policy contracts, on the native
backend.

Per docs/native_backend.md / the porting brief: an `out=` op computes into
the caller's tensor when it already has the right shape/dtype/contiguity,
else computes then `copy_strided_into`s -- so a same-shape `out=` on a view
must keep writing through the *original* storage (no silent reallocation),
and dtype-checked `out=` ops must reject a mismatched output dtype with a
clear error rather than a wrong answer.
"""

import pytest
import torch

from torch_mojo_backend import aten_functions
from torch_mojo_backend.testing import CallChecker

pytestmark = pytest.mark.xdist_group(name="group1")


@pytest.mark.xfail(strict=False, reason="op not ported yet: aten::add.out")
@pytest.mark.parametrize("storage_offset", [0, 2])
def test_add_out_resize_preserves_existing_storage_alias(
    mojo_gpu: str, call_checker: CallChecker, storage_offset: int
):
    call_checker.register(aten_functions.aten_add)
    base = torch.arange(8, dtype=torch.float32).to(mojo_gpu)
    out = base[storage_offset : storage_offset + 2]
    base_ptr = base.data_ptr()

    lhs = torch.tensor([2.0, 3.0], device=mojo_gpu)
    rhs = torch.tensor([5.0, 7.0], device=mojo_gpu)
    returned = torch.add(lhs, rhs, out=out)

    assert returned is out
    assert out.data_ptr() == base_ptr + storage_offset * base.element_size()
    assert out.shape == (2,)
    torch.testing.assert_close(out.cpu(), torch.tensor([7.0, 10.0]))
    expected_base = torch.arange(8, dtype=torch.float32)
    expected_base[storage_offset : storage_offset + 2] = torch.tensor([7.0, 10.0])
    torch.testing.assert_close(base.cpu(), expected_base)


@pytest.mark.xfail(strict=False, reason="op not ported yet: aten::mul.out")
def test_mul_out_resizes_a_mismatched_output(mojo_gpu: str, call_checker: CallChecker):
    """An `out=` tensor with the wrong shape must be resized, not reused."""
    call_checker.register(aten_functions.aten_mul)
    out = torch.empty((), dtype=torch.float32, device=mojo_gpu)

    lhs = torch.tensor([2.0, 3.0], device=mojo_gpu)
    rhs = torch.tensor([5.0, 7.0], device=mojo_gpu)
    torch.mul(lhs, rhs, out=out)

    assert out.shape == (2,)
    torch.testing.assert_close(out.cpu(), torch.tensor([10.0, 21.0]))
    incremented = torch.add(out, out)
    torch.testing.assert_close(incremented.cpu(), torch.tensor([20.0, 42.0]))


@pytest.mark.xfail(
    strict=False,
    reason="op not ported yet: aten::any.out / aten::isin.Tensor_Tensor_out",
)
@pytest.mark.parametrize(
    ("op_name", "valid_dtype", "invalid_dtype"),
    [
        ("aten::any.out", torch.uint8, torch.int64),
        ("aten::isin.Tensor_Tensor_out", torch.bool, torch.int64),
    ],
)
def test_out_variants_enforce_operation_specific_dtype_contracts(
    mojo_gpu: str, op_name: str, valid_dtype: torch.dtype, invalid_dtype: torch.dtype
):
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
