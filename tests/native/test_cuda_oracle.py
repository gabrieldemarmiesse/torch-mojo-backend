"""The live CUDA oracle's comparison (tests/native/cuda_oracle.py): the
policies on CPU tensors, and one round trip on both devices."""

import pytest
import torch

from tests.native.cuda_oracle import (
    CudaOracle,
    Raised,
    assert_bits_equal,
    tensor_mismatch,
)


def _f32(*bits: int) -> torch.Tensor:
    return torch.tensor(bits, dtype=torch.int64).to(torch.int32).view(torch.float32)


def test_nan_payload_and_sign_are_bits():
    quiet, other, negative = _f32(0x7FC00000), _f32(0x7FFFFFFF), _f32(-0x400000)
    assert tensor_mismatch(quiet, quiet) is None
    report = tensor_mismatch(quiet, other)
    assert report is not None
    assert "ours 0x7fc00000 (nan), CUDA 0x7fffffff (nan)" in report
    assert tensor_mismatch(quiet, negative) is not None
    assert tensor_mismatch(quiet, other, nan="any") is None
    assert tensor_mismatch(quiet, negative, nan="any") is None


def test_signed_zero():
    pos, neg = torch.tensor([0.0, 1.0]), torch.tensor([-0.0, 1.0])
    assert tensor_mismatch(pos, neg) is not None
    assert tensor_mismatch(pos, neg, signed_zero=False) is None
    assert tensor_mismatch(torch.tensor([0.0]), torch.tensor([1e-45])) is not None


@pytest.mark.parametrize(
    "dtype",
    [
        torch.bfloat16, torch.float16, torch.float64, torch.float8_e4m3fn,
        torch.float8_e5m2, torch.int8, torch.uint8, torch.int64, torch.uint16,
        torch.bool,
    ],
)  # fmt: skip
def test_every_dtype_reports_its_first_mismatch(dtype: torch.dtype):
    a = torch.tensor([[0, 1], [1, 0]]).to(dtype)
    b = a.clone()
    b[1, 0] = 0
    report = tensor_mismatch(a, b, inputs=[a.float()])
    assert report is not None
    assert report.startswith(f"1 of 4 elements differ (dtype {dtype}):")
    assert "[1, 0]: ours 0x" in report and "input 0 0x3f800000 (1.0)" in report


def test_complex_names_the_part():
    a = torch.tensor([1 + 2j, 3 + 4j], dtype=torch.complex64)
    b = torch.tensor([1 + 2j, 3 - 4j], dtype=torch.complex64)
    report = tensor_mismatch(a, b)
    assert report is not None and "[1].imag: ours 0x40800000 (4.0)" in report
    assert tensor_mismatch(a.conj(), a.conj().resolve_conj()) is None
    assert tensor_mismatch(a.conj(), a.conj(), nan="any", signed_zero=False) is None


def test_dtype_shape_and_stride():
    a = torch.zeros(2, 3)
    assert "dtype" in str(tensor_mismatch(a, a.double()))
    assert "shape" in str(tensor_mismatch(a, a.reshape(3, 2)))
    t = torch.zeros(3, 2).t()
    assert tensor_mismatch(a, t) is None
    assert "stride" in str(tensor_mismatch(a, t, check_stride=True))


def test_structures_scalars_and_errors():
    one = torch.ones(2)
    assert_bits_equal((one, [1, 2.5]), (one.clone(), [1, 2.5]))
    with pytest.raises(AssertionError, match="structure"):
        assert_bits_equal((one,), [one])
    with pytest.raises(AssertionError, match="output 1: ours -0.0, CUDA 0.0"):
        assert_bits_equal((one, -0.0), (one, 0.0))
    assert_bits_equal(-0.0, 0.0, signed_zero=False)
    error = Raised(RuntimeError("bad cuda:0\nmore"))
    assert_bits_equal(error, Raised(RuntimeError("other")))
    with pytest.raises(AssertionError, match="ours RuntimeError: bad DEV"):
        assert_bits_equal(error, Raised(RuntimeError("other")), check_message=True)
    with pytest.raises(AssertionError, match="CUDA returned"):
        assert_bits_equal(Raised(NotImplementedError("x")), one)


def test_round_trip_on_both_devices(cuda_oracle: CudaOracle):
    x = torch.linspace(-3, 3, 101)
    cuda_oracle.assert_bitwise_equal(torch.neg, x)
    out = torch.empty(101)
    cuda_oracle.assert_bitwise_equal(torch.neg, x, kwargs={"out": out})
    cuda_oracle.assert_bitwise_equal_by_device(
        lambda d: torch.rand(7, device=d), seed=3
    )
    with pytest.raises(AssertionError, match="101 of 101 elements differ"):
        cuda_oracle.assert_bitwise_equal_by_device(
            lambda d: torch.neg(x.to(d)) if d.startswith("cuda") else x.to(d)
        )


def test_inputs_alias_and_errors_keep_state(cuda_oracle: CudaOracle):
    x = torch.arange(4.0)
    # one tensor passed twice is one tensor on each device, a fresh copy
    # even when it already lives there
    seen = []

    def same(a: torch.Tensor, b: torch.Tensor) -> torch.Tensor:
        seen.append(a is b)
        return a.add_(1)

    on_mojo = x.to(cuda_oracle.mojo)
    cuda_oracle.assert_bitwise_equal(same, on_mojo, on_mojo)
    assert seen == [True, True]
    assert torch.equal(on_mojo.cpu(), x)

    # a raise after an out= resize compares the out's shape, not its bytes
    cuda_oracle.assert_bitwise_equal(
        lambda a, out: torch.neg(a, out=out),
        torch.ones(64).to(torch.float8_e4m3fn),
        kwargs={"out": torch.empty(0, dtype=torch.float8_e4m3fn)},
    )

    # both raise, but only the mojo side wrote its input first
    def wrote(t: torch.Tensor):
        if t.device.type == "mojo":
            t.zero_()
        raise RuntimeError("same")

    with pytest.raises(AssertionError, match="3 of 4 elements differ"):
        cuda_oracle.assert_bitwise_equal(wrote, x)

    # and a draw before the raise moves only the mojo generator
    def draw_then_raise(d: str):
        if d.startswith("mojo"):
            torch.rand(4, device=d)
        raise RuntimeError("same")

    with pytest.raises(AssertionError, match="differ"):
        cuda_oracle.assert_bitwise_equal_by_device(draw_then_raise, seed=1)
