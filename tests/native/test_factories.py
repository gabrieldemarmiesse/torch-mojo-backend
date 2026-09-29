"""Native backend: factories group (arange.start_out, uniform_, normal_,
native_dropout, native_dropout_backward, multinomial), plus a correctness smoke test of
the composite-decomposed factories this group deliberately does not
register (see ops_factories.mojo's module docstring and the final report).

Public torch API only, per the porting brief: no `aten_fast`,
`TorchMojoTensor`, or `_ctx_ptr` internals.
"""

import math
import re
from fractions import Fraction

import numpy as np
import pytest
import torch

from tests.native.conftest import is_metal as _metal, skip_if_metal
from torch_mojo_backend import aten_functions, native
from torch_mojo_backend.native import device_module


def _op_count_delta(name: str):
    """A context manager-less before/after pair for one native op's call
    count (agents_docs/native_backend.md "Test support"), for ops with no
    `aten_functions` twin for `CallChecker` to key off."""
    native.op_counting(True)
    before = native.op_count(name)

    def _ran() -> bool:
        return native.op_count(name) > before

    return _ran


# ---------------------------------------------------------------------------
# Composite-decomposed factories: full/zeros/ones/new_*/scalar_tensor/
# empty_like/*_like are not registered here (see ops_factories.mojo) --
# ATen's own CompositeExplicitAutograd calls `empty.memory_format` +
# `fill_.Scalar`, both registered by ops_core.mojo. This just checks that
# decomposition actually produces correct values end to end.
# ---------------------------------------------------------------------------


def test_full_zeros_ones_and_likes_decompose_correctly(mojo_gpu):
    ran_empty = _op_count_delta("aten::empty.memory_format")
    ran_fill = _op_count_delta("aten::fill_.Scalar")

    full = torch.full((2, 3), 5.0, device=mojo_gpu)
    torch.testing.assert_close(full.cpu(), torch.full((2, 3), 5.0))
    assert full.dtype == torch.float32

    zeros = torch.zeros(4, dtype=torch.int64, device=mojo_gpu)
    assert zeros.cpu().tolist() == [0, 0, 0, 0]

    ones = torch.ones(3, 2, device=mojo_gpu)
    assert ones.cpu().tolist() == [[1.0, 1.0]] * 3

    scalar = torch.scalar_tensor(7, dtype=torch.int32, device=mojo_gpu)
    assert scalar.cpu().item() == 7

    like_src = torch.randn(2, 5, device=mojo_gpu)
    torch.testing.assert_close(torch.zeros_like(like_src).cpu(), torch.zeros(2, 5))
    torch.testing.assert_close(torch.ones_like(like_src).cpu(), torch.ones(2, 5))
    torch.testing.assert_close(
        torch.full_like(like_src, 2.5).cpu(), torch.full((2, 5), 2.5)
    )
    assert torch.empty_like(like_src).shape == like_src.shape

    self_ = torch.randn(3, device=mojo_gpu)
    assert self_.new_zeros(2, 2).cpu().tolist() == [[0.0, 0.0]] * 2
    assert self_.new_ones(2).cpu().tolist() == [1.0, 1.0]
    assert self_.new_full((2,), 9.0).cpu().tolist() == [9.0, 9.0]
    assert self_.new_empty(2).shape == (2,)

    assert ran_empty(), "empty.memory_format never ran natively"
    assert ran_fill(), "fill_.Scalar never ran natively"


# ---------------------------------------------------------------------------
# arange.start_out
# ---------------------------------------------------------------------------


@pytest.mark.parametrize(
    ("start", "end", "step", "dtype"),
    [
        (0, 10, 1, torch.int64),
        (2, 20, 3, torch.int32),
        (10, 0, -2, torch.float32),
        (0.0, 1.0, 0.1, torch.float32),
        (0, 5, 1, torch.float64),
        (0, 5, 1, torch.float16),
        (0, 5, 1, torch.bfloat16),
    ],
)
def test_arange_matches_cpu(mojo_gpu, start, end, step, dtype):
    ours = torch.arange(start, end, step, dtype=dtype, device=mojo_gpu)
    ref = torch.arange(start, end, step, dtype=dtype)
    torch.testing.assert_close(ours.cpu(), ref, atol=1e-3, rtol=1e-3)


def test_arange_single_and_no_step_args(mojo_gpu):
    torch.testing.assert_close(torch.arange(7, device=mojo_gpu).cpu(), torch.arange(7))
    torch.testing.assert_close(
        torch.arange(3, 9, device=mojo_gpu).cpu(), torch.arange(3, 9)
    )


def test_arange_empty_range(mojo_gpu):
    out = torch.arange(5, 5, device=mojo_gpu)
    assert out.shape == (0,)
    out = torch.arange(5, 5, -1, device=mojo_gpu)
    assert out.shape == (0,)


def test_arange_out_resizes_a_mismatched_tensor(mojo_gpu):
    """The composite path hands `start_out` a size-0 `out`; a direct
    `out=` call with a wrongly-shaped tensor must resize it too."""
    out = torch.empty(3, device=mojo_gpu)
    result = torch.arange(0, 10, 2, out=out)
    assert result.shape == (5,)
    assert result.data_ptr() == out.data_ptr()
    torch.testing.assert_close(result.cpu(), torch.arange(0, 10, 2).float())


def test_arange_rejects_zero_step(mojo_gpu):
    with pytest.raises(RuntimeError, match="step must be nonzero"):
        torch.arange(0, 5, 0, device=mojo_gpu)


def test_arange_rejects_inconsistent_sign(mojo_gpu):
    with pytest.raises(RuntimeError):
        torch.arange(0, 5, -1, device=mojo_gpu)


def test_arange_huge_int_uses_host_fallback(mojo_gpu):
    """Values cross into the device kernel as Float64, so one beyond 2**53
    cannot round-trip exactly: the host fallback (`_host_arange_tensor` in
    the old code) must be used instead, and still produce exact values.
    """
    start = 2**60
    out = torch.arange(start, start + 4, dtype=torch.int64, device=mojo_gpu)
    torch.testing.assert_close(
        out.cpu(), torch.arange(start, start + 4, dtype=torch.int64)
    )


def test_arange_strided_out(mojo_gpu):
    storage = torch.zeros(10, device=mojo_gpu)
    view = storage[::2]
    torch.arange(0, 5, out=view)
    assert torch.equal(storage.cpu()[::2], torch.arange(0, 5).float())
    assert storage.cpu()[1::2].abs().max().item() == 0.0


def test_arange_start_out_native_call_counted(mojo_gpu):
    call_checker_ran = _op_count_delta("aten::arange.start_out")
    _ = torch.arange(12, device=mojo_gpu)
    assert call_checker_ran()
    assert aten_functions.aten_arange is not None  # the compile-backend twin


# ---------------------------------------------------------------------------
# uniform_
# ---------------------------------------------------------------------------

UNIFORM_DTYPES = [torch.float32, torch.bfloat16, torch.float16, torch.float64]


@pytest.mark.parametrize("dtype", UNIFORM_DTYPES)
@pytest.mark.parametrize(("low", "high"), [(0.0, 1.0), (-100.0, 100.0), (1.0, 2.0)])
def test_uniform_bounds_and_distribution(mojo_device, dtype, low, high):
    if dtype == torch.float64:
        skip_if_metal(mojo_device, "uniform_ of dtype float64 is declined on Apple GPU")
    device_module.manual_seed_all(20260814)
    drawn = torch.empty(200_000, dtype=dtype, device=mojo_device).uniform_(low, high)
    host = drawn.cpu().double()

    assert host.min().item() >= low
    assert host.max().item() < high
    span = high - low
    assert abs(host.mean().item() - (low + high) / 2) < 0.01 * span
    assert abs(host.var().item() - span * span / 12) < 0.02 * span * span


@pytest.mark.parametrize("dtype", UNIFORM_DTYPES)
def test_uniform_from_equals_to_is_constant(mojo_gpu, dtype):
    if dtype == torch.float64:
        skip_if_metal(mojo_gpu, "uniform_ of dtype float64 is declined on Apple GPU")
    drawn = torch.empty(37, dtype=dtype, device=mojo_gpu).uniform_(2.5, 2.5)
    assert torch.equal(drawn.cpu(), torch.full((37,), 2.5, dtype=dtype))


def test_uniform_is_reproducible_under_manual_seed(mojo_device):
    device_module.manual_seed_all(4242)
    first = torch.empty(1023, device=mojo_device).uniform_(-2.0, 5.0).cpu()
    device_module.manual_seed_all(4242)
    second = torch.empty(1023, device=mojo_device).uniform_(-2.0, 5.0).cpu()
    torch.testing.assert_close(first, second)


def test_uniform_rng_state_round_trips(mojo_device):
    device_module.manual_seed_all((1 << 63) + 0x54321)
    initial = device_module.get_rng_state(mojo_device)
    first = torch.empty(1025, device=mojo_device).uniform_().cpu()
    advanced = device_module.get_rng_state(mojo_device)
    assert not torch.equal(initial, advanced)

    device_module.set_rng_state(initial, mojo_device)
    replayed = torch.empty(1025, device=mojo_device).uniform_().cpu()

    torch.testing.assert_close(first, replayed)
    torch.testing.assert_close(device_module.get_rng_state(mojo_device), advanced)


def test_uniform_adjacent_draws_are_independent(mojo_gpu):
    device_module.manual_seed_all(20260814)
    first = torch.empty(8192, device=mojo_gpu).uniform_().cpu()
    second = torch.empty(8192, device=mojo_gpu).uniform_().cpu()
    assert int((first == second).sum()) == 0


def test_uniform_strided_destination_matches_contiguous(mojo_gpu):
    device_module.manual_seed_all(20260814)
    state = device_module.get_rng_state(mojo_gpu)
    contiguous = torch.empty(4, 4, device=mojo_gpu).uniform_(10.0, 11.0)

    device_module.set_rng_state(state, mojo_gpu)
    storage = torch.zeros(4, 8, device=mojo_gpu)
    view = storage[:, ::2]
    assert not view.is_contiguous()
    view.uniform_(10.0, 11.0)

    host = storage.cpu()
    torch.testing.assert_close(host[:, ::2], contiguous.cpu())
    assert host[:, 1::2].abs().max().item() == 0.0


def test_uniform_empty_does_not_advance_rng(mojo_gpu):
    drawn = torch.empty(0, 5, device=mojo_gpu)
    device_module.manual_seed_all(20260814)
    before = device_module.get_rng_state(drawn.device)
    assert drawn.uniform_() is drawn
    torch.testing.assert_close(device_module.get_rng_state(drawn.device), before)


@pytest.mark.parametrize(
    ("dtype", "low", "high"),
    [
        (torch.float32, 3.0, -1.0),
        (torch.float16, -1e9, 1.0),
        (torch.float16, 0.0, 1e9),
        (torch.float32, -3e38, 3e38),
    ],
)
def test_uniform_rejects_bad_bounds(mojo_gpu, dtype, low, high):
    device_module.manual_seed_all(20260814)
    before = device_module.get_rng_state(mojo_gpu)
    for numel in (10, 0):
        drawn = torch.empty(numel, dtype=dtype, device=mojo_gpu)
        with pytest.raises(RuntimeError):
            drawn.uniform_(low, high)
    torch.testing.assert_close(device_module.get_rng_state(mojo_gpu), before)


def test_uniform_declines_integer_dtypes(mojo_gpu):
    drawn = torch.zeros(4, dtype=torch.int64, device=mojo_gpu)
    with pytest.raises(NotImplementedError, match="aten::uniform_"):
        drawn.uniform_(0.0, 1.0)


def test_uniform_initializes_a_module_like_nn_init(mojo_gpu):
    layer = torch.nn.Linear(64, 32).to(mojo_gpu)
    torch.nn.init.uniform_(layer.weight, -0.125, 0.125)
    host = layer.weight.detach().cpu()
    assert host.min().item() >= -0.125
    assert host.max().item() < 0.125
    assert host.std().item() > 0.05


def test_uniform_carries_torch_rand_on_device(mojo_gpu):
    """`torch.rand` needs no registration of its own: ATen's own
    CompositeExplicitAutograd `rand` is `empty` + `uniform_`.

    `.min()`/`.max()` run on the CPU copy: `aten::min`/`aten::max` are a
    different group's ops, not registered by this one.
    """
    torch.manual_seed(20260814)
    host = torch.rand(1000, device=mojo_gpu).cpu()
    assert host.min().item() >= 0.0
    assert host.max().item() < 1.0


def test_uniform_explicit_generator_is_independent_and_reproducible(mojo_gpu):
    """Unlike the old eager path (which could not construct
    `torch.Generator(device="mojo")` at all), the native backend's
    `getNewGenerator` makes this a real, independently seeded stream --
    `uniform_` now honors it instead of declining it."""
    g1 = torch.Generator(device=mojo_gpu)
    g1.manual_seed(111)
    g2 = torch.Generator(device=mojo_gpu)
    g2.manual_seed(222)

    a = torch.empty(512, device=mojo_gpu).uniform_(0.0, 1.0, generator=g1).cpu()
    b = torch.empty(512, device=mojo_gpu).uniform_(0.0, 1.0, generator=g2).cpu()
    assert int((a == b).sum()) == 0

    g1.manual_seed(111)
    a_again = torch.empty(512, device=mojo_gpu).uniform_(0.0, 1.0, generator=g1).cpu()
    torch.testing.assert_close(a, a_again)


def test_uniform_native_call_counted(mojo_gpu):
    ran = _op_count_delta("aten::uniform_")
    torch.empty(4, device=mojo_gpu).uniform_()
    assert ran()


# ---------------------------------------------------------------------------
# normal_
# ---------------------------------------------------------------------------


def test_normal_mean_and_std(mojo_device):
    torch.manual_seed(20260908)
    drawn = torch.empty(200_000, device=mojo_device).normal_(2.0, 3.0)
    host = drawn.cpu().double()
    assert abs(host.mean().item() - 2.0) < 0.05
    assert abs(host.std().item() - 3.0) < 0.05


def test_normal_default_mean_std(mojo_device):
    torch.manual_seed(20260908)
    drawn = torch.empty(100_000, device=mojo_device).normal_()
    host = drawn.cpu().double()
    assert abs(host.mean().item()) < 0.05
    assert abs(host.std().item() - 1.0) < 0.05


def test_normal_strided_destination(mojo_gpu):
    storage = torch.zeros(4, 8, device=mojo_gpu)
    view = storage[:, ::2]
    view.normal_(0.0, 1.0)
    host = storage.cpu()
    assert host[:, 1::2].abs().max().item() == 0.0
    assert host[:, ::2].std().item() > 0.0


def test_normal_explicit_generator_is_independent_and_reproducible(mojo_gpu):
    """normal_ draws from the generator it is given, not the default one."""
    g = torch.Generator(device=mojo_gpu).manual_seed(111)
    device_module.manual_seed_all(999)
    before = device_module.get_rng_state(mojo_gpu)
    a = torch.empty(512, device=mojo_gpu).normal_(generator=g).cpu()
    torch.testing.assert_close(device_module.get_rng_state(mojo_gpu), before)
    g.manual_seed(111)
    torch.testing.assert_close(
        torch.empty(512, device=mojo_gpu).normal_(generator=g).cpu(), a
    )


def test_normal_native_call_counted(mojo_gpu):
    ran = _op_count_delta("aten::normal_")
    torch.empty(4, device=mojo_gpu).normal_()
    assert ran()
    assert aten_functions.aten_normal_ is not None  # the compile-backend twin


# ---------------------------------------------------------------------------
# native_dropout / native_dropout_backward
# ---------------------------------------------------------------------------


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
def test_native_dropout_forward_semantics(mojo_gpu, p, train, should_advance_rng):
    input = torch.linspace(-4.0, 4.0, 257)
    mojo_input = input.to(mojo_gpu)
    device_module.manual_seed_all((1 << 63) + 20260718)
    before = device_module.get_rng_state(mojo_input.device)

    output, mask = torch.ops.aten.native_dropout.default(mojo_input, p, train)
    after = device_module.get_rng_state(mojo_input.device)

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
def test_native_dropout_inference_ignores_probability(mojo_gpu, p):
    input = torch.linspace(-4.0, 4.0, 17)
    mojo_input = input.to(mojo_gpu)
    device_module.manual_seed_all(20260718)
    before = device_module.get_rng_state(mojo_input.device)

    output, mask = torch.ops.aten.native_dropout.default(mojo_input, p, False)

    assert output is not mojo_input
    torch.testing.assert_close(output.cpu(), input)
    assert mask.cpu().all()
    torch.testing.assert_close(device_module.get_rng_state(mojo_input.device), before)


def test_native_dropout_empty_does_not_advance_rng(mojo_gpu):
    input = torch.empty(0, 7).to(mojo_gpu)
    device_module.manual_seed_all(20260718)
    before = device_module.get_rng_state(input.device)
    output, mask = torch.ops.aten.native_dropout.default(input, 0.0, True)
    after = device_module.get_rng_state(input.device)

    assert output is not input
    assert output.shape == input.shape
    assert mask.shape == input.shape
    assert mask.dtype == torch.bool
    torch.testing.assert_close(after, before)


def test_native_dropout_rng_state_replays_exactly(mojo_gpu):
    input = torch.randn(4097).to(mojo_gpu)
    device_module.manual_seed_all((1 << 63) + 0x12345)
    initial = device_module.get_rng_state(input.device)
    first_output, first_mask = torch.ops.aten.native_dropout.default(input, 0.2, True)
    advanced = device_module.get_rng_state(input.device)

    device_module.set_rng_state(initial, input.device)
    replay_output, replay_mask = torch.ops.aten.native_dropout.default(input, 0.2, True)

    torch.testing.assert_close(replay_mask.cpu(), first_mask.cpu())
    torch.testing.assert_close(replay_output.cpu(), first_output.cpu())
    torch.testing.assert_close(device_module.get_rng_state(input.device), advanced)


@pytest.mark.parametrize("p", [-0.1, 1.1, float("nan")])
def test_native_dropout_invalid_probability_does_not_touch_rng(mojo_gpu, p):
    input = torch.randn(3, 8).to(mojo_gpu)[:, ::2]
    device_module.manual_seed_all(20260718)
    before = device_module.get_rng_state(input.device)
    with pytest.raises(RuntimeError, match="probability has to be between 0 and 1"):
        torch.ops.aten.native_dropout.default(input, p, True)
    torch.testing.assert_close(device_module.get_rng_state(input.device), before)


def test_native_dropout_declines_integer_dtypes(mojo_gpu):
    input = torch.ones(4, dtype=torch.int32, device=mojo_gpu)
    with pytest.raises(NotImplementedError, match="native_dropout"):
        torch.ops.aten.native_dropout.default(input, 0.5, True)


def test_native_dropout_backward_multiplication_semantics(mojo_gpu):
    grad_output = torch.tensor([-0.0, -2.0, float("nan"), float("inf")])
    mask = torch.tensor([True, False, False, False])
    result = torch.ops.aten.native_dropout_backward.default(
        grad_output.to(mojo_gpu), mask.to(mojo_gpu), 2.0
    ).cpu()
    expected = grad_output * mask * 2.0

    torch.testing.assert_close(result, expected, equal_nan=True)
    assert torch.signbit(result[:2]).tolist() == torch.signbit(expected[:2]).tolist()


def test_native_dropout_training_backward_and_saved_mask(mojo_gpu):
    generator = torch.Generator().manual_seed(20260718)
    input = torch.randn(3, 17, generator=generator).to(mojo_gpu).requires_grad_()
    grad_output = torch.randn(3, 17, generator=generator)
    ran = _op_count_delta("aten::native_dropout_backward")

    output, mask = torch.ops.aten.native_dropout.default(input, 0.2, True)
    assert type(output.grad_fn).__name__ == "NativeDropoutBackward0"
    assert not ran()
    output.backward(grad_output.to(mojo_gpu))
    assert ran()

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
def test_native_dropout_autograd_optional_train_scale(mojo_gpu, train, scale):
    input = torch.randn(3, 17).to(mojo_gpu).requires_grad_()
    grad_output = torch.randn(3, 17)
    output, mask = torch.ops.aten.native_dropout.default(input, 0.2, train)
    output.backward(grad_output.to(mojo_gpu))
    assert input.grad is not None
    torch.testing.assert_close(
        input.grad.cpu(), grad_output * mask.cpu() * scale, atol=1e-6, rtol=1e-6
    )


# ---------------------------------------------------------------------------
# RNG stream edges the tests above (8192- and 1025-element draws from an
# aligned base) cannot reach.
# ---------------------------------------------------------------------------


def test_uniform_non_multiple_of_four_draws_do_not_overlap(mojo_gpu):
    """A 7-element draw still advances the counter by a whole reservation, or
    the next draw repeats it."""
    device_module.manual_seed_all(20260814)
    first = torch.empty(7, device=mojo_gpu).uniform_().cpu()
    state = device_module.get_rng_state(mojo_gpu)
    second = torch.empty(7, device=mojo_gpu).uniform_().cpu()
    assert not set(first.tolist()) & set(second.tolist())

    device_module.set_rng_state(state, mojo_gpu)
    replayed = torch.empty(7, device=mojo_gpu).uniform_().cpu()
    torch.testing.assert_close(replayed, second)


def test_uniform_misaligned_destination_indexes_the_stream_the_same_way(mojo_gpu):
    """The stream mapping is independent of the destination's alignment."""
    device_module.manual_seed_all(20260814)
    state = device_module.get_rng_state(mojo_gpu)
    aligned = torch.zeros(8, device=mojo_gpu).uniform_(-1.0, 1.0).cpu()

    device_module.set_rng_state(state, mojo_gpu)
    storage = torch.zeros(9, device=mojo_gpu)
    storage[1:].uniform_(-1.0, 1.0)
    host = storage.cpu()
    assert torch.equal(host[1:], aligned)
    assert float(host[0]) == 0.0


def test_torch_manual_seed_reaches_the_device_generator(mojo_gpu):
    """`torch.manual_seed` (not just device_module.manual_seed_all) must seed
    the mojo generator."""
    torch.manual_seed(20260913)
    first = torch.rand(1000, device=mojo_gpu).cpu()
    torch.manual_seed(20260913)
    replayed = torch.rand(1000, device=mojo_gpu).cpu()
    torch.testing.assert_close(first, replayed)
    assert not torch.equal(torch.rand(1000, device=mojo_gpu).cpu(), first)


def test_arange_needs_a_wide_accumulator(mojo_gpu):
    """Past 2**24 an fp32 running sum can no longer add 1.0.

    The reference is the vendor backend's own answer for this accelerator;
    the test skips where there is none to compare with rather than inventing
    one.
    """
    args = (16_777_217.0, 16_777_227.0, 1.0)
    result = torch.arange(*args, dtype=torch.float32, device=mojo_gpu).cpu()
    if torch.cuda.is_available():
        expected = torch.arange(*args, dtype=torch.float32, device="cuda").cpu()
    elif torch.backends.mps.is_available():
        expected = torch.arange(*args, dtype=torch.float32, device="mps").cpu()
    else:
        pytest.skip("no native GPU reference for this MAX accelerator")
    assert torch.equal(result, expected)


def test_float64_factories_fill_scatter_and_arange(mojo_gpu):
    """fp64 is a separate kernel specialization from fp32 for each of these."""
    skip_if_metal(mojo_gpu, "Metal has no float64")
    ones = torch.ones(5, dtype=torch.float64, device=mojo_gpu)
    assert ones.dtype == torch.float64
    torch.testing.assert_close(ones.cpu(), torch.ones(5, dtype=torch.float64))

    filled = torch.empty(5, dtype=torch.float64, device=mojo_gpu).fill_(2.5)
    torch.testing.assert_close(filled.cpu(), torch.full((5,), 2.5, dtype=torch.float64))

    scattered = torch.zeros(5, dtype=torch.float64, device=mojo_gpu).scatter(
        0,
        torch.tensor([1, 3], device=mojo_gpu),
        torch.tensor([4.0, 7.0], dtype=torch.float64, device=mojo_gpu),
    )
    torch.testing.assert_close(
        scattered.cpu(), torch.tensor([0.0, 4.0, 0.0, 7.0, 0.0], dtype=torch.float64)
    )

    ranged = torch.arange(0.0, 2.0, 0.25, dtype=torch.float64, device=mojo_gpu)
    torch.testing.assert_close(
        ranged.cpu(), torch.arange(0.0, 2.0, 0.25, dtype=torch.float64)
    )


def test_randint_and_random_on_the_device(mojo_device):
    """random_.from / .to / random_ draw on the device (values in range)."""
    torch.manual_seed(0)
    x = torch.randint(3, 9, (200,), device=mojo_device)
    assert x.dtype == torch.int64
    vals = x.cpu()
    assert vals.min() >= 3 and vals.max() < 9 and vals.unique().numel() == 6
    y = torch.empty(64, dtype=torch.int32, device=mojo_device).random_(5)
    assert set(y.cpu().tolist()) <= set(range(5))
    z = torch.empty(64, dtype=torch.uint8, device=mojo_device).random_()
    assert z.cpu().max() <= 255
    strided = torch.zeros(4, 6, dtype=torch.int64, device=mojo_device).t()
    strided.random_(1, 3)
    assert set(strided.cpu().unique().tolist()) <= {1, 2}


@pytest.mark.parametrize("dtype", [torch.bfloat16, torch.float16])
def test_native_dropout_half_precision_round_trips_through_float32(mojo_gpu, dtype):
    # 1 + randn: no input is exactly zero, so a zero output means "dropped"
    x = (1.0 + torch.rand(64, 32)).to(dtype).to(mojo_gpu).requires_grad_(True)
    y = torch.nn.functional.dropout(x, p=0.5, training=True)
    assert y.dtype == dtype
    kept = y.detach().cpu().float() != 0
    torch.testing.assert_close(
        y.detach().cpu().float()[kept], 2 * x.detach().cpu().float()[kept]
    )
    assert 0.3 < kept.float().mean() < 0.7
    y.float().sum().backward()
    assert x.grad is not None
    assert x.grad.dtype == dtype
    torch.testing.assert_close(x.grad.cpu().float(), 2 * kept.float(), atol=0, rtol=0)


# ---------------------------------------------------------------------------
# multinomial. A draw is random, so it is validated statistically -- fixed
# seeds, a chi-square bound far from the edge (p < 1e-6 for a correct
# sampler) -- plus what no draw may ever do: repeat without replacement,
# leave the range, or pick a zero-probability category while a positive one
# is left. ATen's fast path (no replacement, or one sample) is also
# bit-for-bit stock CUDA's for the same seed, checked when CUDA is present.
# ---------------------------------------------------------------------------

_MULTINOMIAL_PROBS = torch.tensor([0.0, 0.1, 0.2, 0.0, 0.3, 0.4, 0.0])
_MULTINOMIAL_DTYPES = [torch.float32, torch.bfloat16, torch.float16, torch.float64]


def _chi2(samples: torch.Tensor, probs: torch.Tensor) -> float:
    counts = torch.bincount(samples.reshape(-1).cpu(), minlength=probs.numel())
    counts = counts.double()
    expected = probs.double() / probs.double().sum() * counts.sum()
    assert (counts[expected == 0] == 0).all(), counts
    support = expected > 0
    return float((((counts - expected) ** 2)[support] / expected[support]).sum())


def _multinomial(probs: torch.Tensor, n: int, replacement: bool, **kwargs):
    ran = _op_count_delta("aten::multinomial")
    out = torch.multinomial(probs, n, replacement, **kwargs)
    assert ran()
    assert out.dtype == torch.int64 and out.device == probs.device
    return out


@pytest.mark.parametrize("dtype", _MULTINOMIAL_DTYPES)
@pytest.mark.parametrize(
    ("n", "replacement", "rows"),
    [(1, False, 20000), (1, True, 20000), (3, False, 20000), (50, True, 400)],
)
def test_multinomial_frequencies(mojo_gpu, dtype, n, replacement, rows):
    """Every first draw follows p (for top-k Gumbel sampling the first pick
    is the argmax, a draw from p), and with replacement every draw does."""
    if dtype is torch.float64:
        skip_if_metal(mojo_gpu, "float64 is unavailable on Apple GPUs")
    torch.manual_seed(1234)
    probs = _MULTINOMIAL_PROBS.to(dtype)
    out = _multinomial(probs.expand(rows, -1).contiguous().to(mojo_gpu), n, replacement)
    assert out.shape == (rows, n)
    drawn = out if replacement else out[:, 0]
    assert _chi2(drawn, probs.float()) < 35  # 3 degrees of freedom


def test_multinomial_one_long_row_with_replacement(mojo_gpu):
    """The inverse-CDF sampler over a vocabulary-sized row whose CDF spans
    many per-thread chunks, with runs of zeros across chunk boundaries."""
    torch.manual_seed(7)
    probs = torch.rand(50257)
    probs[1000:1500] = 0.0
    probs[::97] = 0.0
    out = _multinomial(probs.to(mojo_gpu), 200000, True).cpu()
    assert out.shape == (200000,)
    assert (probs[out] > 0).all()
    # Coarse bins of ~50 categories keep the chi-square meaningful.
    bins = torch.arange(50257) // 50
    counts = torch.bincount(bins[out], minlength=int(bins[-1]) + 1).double()
    expected = torch.zeros_like(counts).index_add_(0, bins, probs.double())
    expected = expected / expected.sum() * out.numel()
    support = expected > 0
    dof = int(support.sum()) - 1
    chi2 = float((((counts - expected) ** 2)[support] / expected[support]).sum())
    assert chi2 < dof + 8 * dof**0.5, (chi2, dof)


def test_multinomial_without_replacement_never_repeats(mojo_gpu):
    torch.manual_seed(3)
    probs = torch.rand(64, 37).to(mojo_gpu)
    out = _multinomial(probs, 37, False).cpu()
    assert torch.equal(out.sort(dim=1).values, torch.arange(37).expand(64, -1))
    positives = torch.tensor([[0.0, 2.0, 0.0, 1.0, 3.0]] * 500).to(mojo_gpu)
    out = _multinomial(positives, 3, False).cpu()
    assert set(out.reshape(-1).tolist()) == {1, 3, 4}


@pytest.mark.parametrize("replacement", [False, True])
def test_multinomial_is_reproducible(mojo_gpu, replacement):
    probs = torch.rand(5, 11).to(mojo_gpu)
    n = 4
    torch.manual_seed(99)
    first = _multinomial(probs, n, replacement).cpu()
    second = _multinomial(probs, n, replacement).cpu()
    torch.manual_seed(99)
    assert torch.equal(_multinomial(probs, n, replacement).cpu(), first)
    assert not torch.equal(first, second)
    g = torch.Generator(device=mojo_gpu)
    g.manual_seed(5)
    a = _multinomial(probs, n, replacement, generator=g).cpu()
    g.manual_seed(5)
    assert torch.equal(_multinomial(probs, n, replacement, generator=g).cpu(), a)


@pytest.mark.parametrize(("n", "replacement"), [(1, False), (1, True), (6, False)])
def test_multinomial_fast_path_matches_stock_cuda(mojo_gpu, n, replacement):
    """ATen's fast path composes exponential_, div and argmax/topk, all of
    them CUDA's bits here, so a seeded draw is stock CUDA's own."""
    if not torch.cuda.is_available():
        pytest.skip("no CUDA device to compare against")
    skip_if_metal(mojo_gpu, "CUDA bit-parity is only claimed on NVIDIA GPUs")
    probs = torch.rand(6, 50304, generator=torch.Generator().manual_seed(0))
    torch.manual_seed(11)
    ours = _multinomial(probs.to(mojo_gpu), n, replacement).cpu()
    torch.manual_seed(11)
    theirs = torch.multinomial(probs.cuda(), n, replacement).cpu()
    assert torch.equal(ours, theirs)


def test_multinomial_shapes_layouts_and_out(mojo_gpu):
    torch.manual_seed(0)
    one_d = _multinomial(torch.rand(9).to(mojo_gpu), 4, True)
    assert one_d.shape == (4,)
    strided = torch.rand(9, 5).to(mojo_gpu).t()  # rows of stride 5
    out = _multinomial(strided, 2, False).cpu()
    assert out.shape == (5, 2) and out.max() < 9
    assert _multinomial(torch.rand(0, 5).to(mojo_gpu), 3, True).shape == (0, 3)
    dest = torch.empty(0, dtype=torch.int64, device=mojo_gpu)
    ran = _op_count_delta("aten::multinomial.out")
    torch.multinomial(torch.rand(2, 5).to(mojo_gpu), 3, out=dest)
    assert ran()
    assert dest.shape == (2, 3) and dest.cpu().max() < 5


@pytest.mark.parametrize(
    "probs",
    [[0.1, -0.1, 0.2], [0.1, float("nan"), 0.2], [0.1, float("inf"), 0.2], [0.0, 0.0]],
    ids=["negative", "nan", "inf", "zero_sum"],
)
@pytest.mark.parametrize(("n", "replacement"), [(1, False), (2, True)])
def test_multinomial_rejects_invalid_distributions_like_cpu(
    mojo_gpu, probs, n, replacement
):
    """The same message as CPU torch, raised synchronously (both of ATen's
    paths, whose messages differ)."""
    host = torch.tensor(probs)
    with pytest.raises(RuntimeError) as cpu_error:
        torch.multinomial(host, n, replacement)
    message = str(cpu_error.value).splitlines()[0]
    with pytest.raises(RuntimeError, match=re.escape(message)):
        torch.multinomial(host.to(mojo_gpu), n, replacement)


def test_multinomial_argument_errors(mojo_gpu):
    probs = torch.rand(2, 3).to(mojo_gpu)
    with pytest.raises(RuntimeError, match="without replacement"):
        torch.multinomial(probs, 4)
    with pytest.raises(RuntimeError, match="n_sample <= 0"):
        torch.multinomial(probs, 0, True)
    with pytest.raises(RuntimeError, match="1 or 2 dim"):
        torch.multinomial(torch.rand(2, 2, 2).to(mojo_gpu), 1)
    with pytest.raises(RuntimeError, match="floating-point dtypes"):
        torch.multinomial(torch.ones(3, dtype=torch.int64).to(mojo_gpu), 1)


@pytest.mark.parametrize("dtype", [torch.int64, torch.bool])
def test_native_dropout_backward_integral_grad_raises(mojo_gpu, dtype):
    """CUDA's dropout_backward dispatches on floating gradients only."""
    grad = torch.tensor([[1, 0, 3], [4, 5, 0]]).to(dtype)
    mask = torch.tensor([[True, False, True], [False, True, True]])
    with pytest.raises(RuntimeError, match='"masked_scale" not implemented'):
        torch.ops.aten.native_dropout_backward(
            grad.to(mojo_gpu), mask.to(mojo_gpu), 1.25
        )


# ---------------------------------------------------------------------------
# eye.out / eye.m_out, linspace.out / logspace.out, tril_indices /
# triu_indices -- the functional forms are ATen composites that end here.
# ---------------------------------------------------------------------------

_RANGE_DTYPES = [
    torch.float32,
    torch.float16,
    torch.bfloat16,
    torch.float64,
    torch.int64,
    torch.int32,
    torch.int8,
    torch.uint8,
]


@pytest.mark.parametrize(
    "dtype", [torch.float32, torch.bfloat16, torch.int64, torch.bool]
)
@pytest.mark.parametrize(("n", "m"), [(3, None), (3, 5), (5, 2), (0, 3), (1, 1)])
def test_eye(mojo_gpu, dtype, n, m):
    # torch.eye(n) itself resolves to eye.m_out; eye.out is reached directly.
    ran = _op_count_delta("aten::eye.m_out")
    args = (n,) if m is None else (n, m)
    got = torch.eye(*args, dtype=dtype, device=mojo_gpu)
    assert ran()
    assert torch.equal(got.cpu(), torch.eye(*args, dtype=dtype))
    if m is None:
        ran_out = _op_count_delta("aten::eye.out")
        out = torch.empty(0, dtype=dtype, device=mojo_gpu)
        torch.ops.aten.eye.out(n, out=out)
        assert ran_out()
        assert torch.equal(out.cpu(), torch.eye(n, dtype=dtype))
    # An out= of another layout is written where it lives.
    out = torch.full((6, 6), 7, dtype=dtype, device=mojo_gpu)[1::2, ::2]
    torch.eye(3, 3, out=out)
    assert torch.equal(out.cpu(), torch.eye(3, dtype=dtype))


def test_eye_errors(mojo_gpu):
    with pytest.raises(RuntimeError, match="n must be greater or equal to 0, got -1"):
        torch.eye(-1, device=mojo_gpu)
    with pytest.raises(RuntimeError, match="m must be greater or equal to 0, got -3"):
        torch.eye(0, -3, device=mojo_gpu)


def _round_f32(x: Fraction) -> float:
    """`x` correctly rounded to float32 (nearest, ties to even): one
    rounding, as a hardware fma does."""
    guess = float(np.float32(float(x)))
    candidates = {guess}
    for direction in (-np.inf, np.inf):
        candidates.add(float(np.nextafter(np.float32(guess), np.float32(direction))))
    return min(
        candidates,
        key=lambda c: (abs(Fraction(c) - x), int(np.float32(c).view(np.uint32)) & 1),
    )


def _cuda_linspace(
    start: float, end: float, steps: int, dtype: torch.dtype
) -> list[float]:
    """ATen's `linspace_cuda_out` (native/cuda/RangeFactories.cu) evaluated
    exactly on the host: the first `steps // 2` elements count up from
    `start`, the rest down from `end`; float32/float64 contract each
    `a + step * i` into one fma, the half types round every c10 operator,
    integers use a float32 step and truncate."""
    half = steps // 2
    n = steps - 1
    if dtype in (torch.float16, torch.bfloat16):

        def r(x: float) -> float:
            return torch.tensor(x, dtype=torch.float32).to(dtype).item()

        s, e = r(start), r(end)
        step = r(r(e - s) / r(float(n)))
        out = []
        for i in range(steps):
            if i < half:
                out.append(r(s + r(step * r(float(i)))))
            else:
                out.append(r(e - r(step * r(float(n - i)))))
        return out
    if dtype == torch.float64:
        step = (end - start) / n
        return [
            float(Fraction(start) + Fraction(step) * i)
            if i < half
            else float(Fraction(end) - Fraction(step) * (n - i))
            for i in range(steps)
        ]
    if dtype == torch.float32:
        s, e = _round_f32(Fraction(start)), _round_f32(Fraction(end))
        step = _round_f32(Fraction(e - s) / n)
    else:
        info = torch.iinfo(dtype)
        s = int(start) % 256 if dtype == torch.uint8 else int(start)
        e = int(end) % 256 if dtype == torch.uint8 else int(end)
        assert info.min <= s <= info.max and info.min <= e <= info.max
        s, e = float(np.float32(s)), float(np.float32(e))
        step = _round_f32(Fraction(_round_f32(Fraction(e) - Fraction(s))) / n)
    out = []
    for i in range(steps):
        if i < half:
            value = _round_f32(Fraction(s) + Fraction(step) * i)
        else:
            value = _round_f32(Fraction(e) - Fraction(step) * (n - i))
        out.append(value if dtype == torch.float32 else float(int(value)))
    return out


@pytest.mark.parametrize("dtype", _RANGE_DTYPES)
@pytest.mark.parametrize(
    ("start", "end", "steps"),
    [
        (0, 1, 5),
        (-3.7, 11.2, 17),
        (3, -10, 101),
        (0, 100, 3001),
        (2, 2, 1),
        (0, 5, 0),
        (1.5, 7, 2),
        (-1, 1, 3),
    ],
)
def test_linspace(mojo_gpu, dtype, start, end, steps):
    if dtype == torch.float64:
        skip_if_metal(mojo_gpu, "Apple GPUs have no float64")
    if not dtype.is_floating_point and (
        isinstance(start, float) and start < 0 and dtype == torch.uint8
    ):
        pytest.skip("a negative float endpoint overflows uint8 (checked below)")
    if dtype in (torch.int8, torch.uint8) and max(abs(start), abs(end)) > 127:
        pytest.skip(
            "out of the dtype's range (checked in test_linspace_out_and_errors)"
        )
    ran = _op_count_delta("aten::linspace.out")
    got = torch.linspace(start, end, steps, dtype=dtype, device=mojo_gpu)
    assert ran()
    assert got.dtype == dtype and got.shape == (steps,)
    if steps == 1:
        expected = torch.linspace(start, end, steps, dtype=dtype)
    else:
        expected = torch.tensor(
            _cuda_linspace(start, end, steps, dtype), dtype=torch.float64
        ).to(dtype)
    assert torch.equal(got.cpu(), expected), (got.cpu(), expected)


def test_linspace_unsigned_wraps_negative_integers(mojo_gpu):
    """An integral Scalar converts to uint8 modulo 256 (c10::overflows lets
    -255..-1 through), as on CPU and CUDA."""
    got = torch.linspace(-1, 1, 3, dtype=torch.uint8, device=mojo_gpu)
    assert got.cpu().tolist() == [255, 128, 1]
    got = torch.linspace(-1, 2.5, 4, dtype=torch.uint8, device=mojo_gpu)
    assert got.cpu().tolist() == torch.linspace(-1, 2.5, 4, dtype=torch.uint8).tolist()


@pytest.mark.parametrize("dtype", _RANGE_DTYPES)
@pytest.mark.parametrize(
    ("start", "end", "steps", "base"),
    [(0, 1, 5, 10.0), (-2, 3, 11, 2.0), (1, 0.5, 7, 3.0), (2, 2, 1, 10.0)],
)
def test_logspace(mojo_gpu, dtype, start, end, steps, base):
    if dtype == torch.float64:
        skip_if_metal(mojo_gpu, "Apple GPUs have no float64")
    if dtype == torch.uint8 and min(start, end) < 0:
        pytest.skip(
            "a negative float exponent is fine, a negative endpoint is not tested here"
        )
    ran = _op_count_delta("aten::logspace.out")
    got = torch.logspace(start, end, steps, base=base, dtype=dtype, device=mojo_gpu)
    assert ran()
    if steps == 1:
        assert torch.equal(
            got.cpu(), torch.logspace(start, end, steps, base=base, dtype=dtype)
        )
        return
    # The exponents are linspace's exact values (`_cuda_linspace`); the power
    # is float `powf` (float64 `pow`), within an ulp of the correctly rounded
    # value, so the half and float types get one ulp of their dtype.
    if dtype.is_floating_point:
        exponents = _cuda_linspace(start, end, steps, dtype)
    else:
        # Integers: the endpoints convert to the integer type first and the
        # exponents are float32 (`static_cast<float>(end - start)` step).
        exponents = _cuda_linspace(int(start), int(end), steps, torch.float32)
    power = torch.tensor([base**x for x in exponents], dtype=torch.float64)
    if dtype.is_floating_point:
        tol = {
            torch.float64: 1e-15,
            torch.float32: 2.4e-7,
            torch.float16: 9.8e-4,
            torch.bfloat16: 7.9e-3,
        }[dtype]
        torch.testing.assert_close(
            got.cpu().double(), power.to(dtype).double(), rtol=tol, atol=0
        )
    else:
        # powf, then truncation toward zero: the truncated value may land one
        # below an exactly integral power when powf rounds down.
        expected = power.floor()
        diff = expected - got.cpu().double()
        assert ((diff == 0) | ((diff == 1) & (power == power.round()))).all(), (
            got,
            power,
        )


def test_range_one_step_and_empty(mojo_gpu):
    """steps 0 and 1 take effect before the dtype dispatch (bool included),
    and the one-step value goes through fill's conversion check."""
    for fn, args in [
        (torch.linspace, (0, 1, 0)),
        (torch.linspace, (1, 1, 1)),
        (torch.linspace, (0, 1, 1)),
        (torch.logspace, (0, 1, 1)),
    ]:
        got = fn(*args, dtype=torch.bool, device=mojo_gpu)
        assert torch.equal(got.cpu(), fn(*args, dtype=torch.bool))
    got = torch.logspace(-1, 0, 1, dtype=torch.uint8, device=mojo_gpu)
    assert got.cpu().tolist() == [0]
    with pytest.raises(RuntimeError, match="without overflow"):
        torch.logspace(3, 1, 1, dtype=torch.uint8, device=mojo_gpu)
    with pytest.raises(RuntimeError, match="not implemented for 'Bool'"):
        torch.linspace(0, 1, 3, dtype=torch.bool, device=mojo_gpu)


def test_linspace_out_and_errors(mojo_gpu):
    out = torch.zeros(10, device=mojo_gpu)[::2]
    torch.linspace(0, 4, 5, out=out)
    assert out.cpu().tolist() == [0.0, 1.0, 2.0, 3.0, 4.0]
    with pytest.raises(RuntimeError, match="number of steps must be non-negative"):
        torch.linspace(0, 1, -1, device=mojo_gpu)
    for start, end, dtype in [
        (0, 1000, torch.int8),
        (-300, 1, torch.uint8),
        (-1.0, 1, torch.uint8),
        (0, 1e6, torch.float16),
    ]:
        with pytest.raises(RuntimeError, match="without overflow"):
            torch.linspace(start, end, 3, dtype=dtype, device=mojo_gpu)


@pytest.mark.parametrize("upper", [False, True])
@pytest.mark.parametrize("dtype", [torch.int64, torch.int32])
@pytest.mark.parametrize(
    ("row", "col", "offset"),
    [(4, 5, 0), (4, 5, -2), (3, 3, 2), (0, 3, 0), (5, 2, 1), (3, 4, -9)],
)
def test_tri_indices(mojo_gpu, upper, dtype, row, col, offset):
    fn = torch.triu_indices if upper else torch.tril_indices
    ran = _op_count_delta("aten::triu_indices" if upper else "aten::tril_indices")
    got = fn(row, col, offset, dtype=dtype, device=mojo_gpu)
    assert ran()
    assert got.dtype == dtype
    assert torch.equal(got.cpu(), fn(row, col, offset, dtype=dtype))


def test_tri_indices_errors(mojo_gpu):
    with pytest.raises(RuntimeError, match="row must be non-negative"):
        torch.tril_indices(-1, 3, device=mojo_gpu)
    with pytest.raises(RuntimeError, match="col must be non-negative"):
        torch.triu_indices(3, -1, device=mojo_gpu)


# ---------------------------------------------------------------------------
# bernoulli.out, _fused_dropout, _fill_mem_eff_dropout_mask_
# ---------------------------------------------------------------------------


def test_bernoulli_out_is_bernoulli_tensor(mojo_gpu):
    """bernoulli.out = out.resize_(p.shape).bernoulli_(p): the same draw as
    the functional form from the same generator state, any out dtype."""
    torch.manual_seed(0)
    p = torch.rand(33, 7).to(mojo_gpu)
    gen = torch.Generator(device=mojo_gpu)
    gen.manual_seed(123)
    want = torch.bernoulli(p, generator=gen).cpu()
    for dtype in (torch.float32, torch.int64, torch.bool, torch.float16):
        gen.manual_seed(123)
        out = torch.empty(0, dtype=dtype, device=mojo_gpu)
        torch.bernoulli(p, generator=gen, out=out)
        assert out.shape == p.shape and out.dtype == dtype
        assert torch.equal(out.cpu().float(), want)
    # A correctly shaped strided out is written where it lives.
    gen.manual_seed(123)
    base = torch.full((7, 33), 5.0, device=mojo_gpu)
    torch.bernoulli(p, generator=gen, out=base.t())
    assert torch.equal(base.t().cpu(), want)
    # p = 0 / 1 are exact; the frequency of 0.3 is.
    q = torch.tensor([0.0, 1.0, 0.3]).repeat_interleave(20000).to(mojo_gpu)
    out = torch.empty(0, device=mojo_gpu)
    torch.bernoulli(q, out=out)
    r = out.cpu().reshape(3, -1)
    assert r[0].sum() == 0 and r[1].sum() == 20000
    assert abs(r[2].mean().item() - 0.3) < 0.015


def test_bernoulli_out_errors_and_aliasing(mojo_gpu):
    x = torch.rand(1, device=mojo_gpu).expand(6)
    with pytest.raises(RuntimeError, match="unsupported operation"):
        torch.bernoulli(torch.rand(6, device=mojo_gpu), out=x)
    with pytest.raises(RuntimeError, match="floating type"):
        torch.bernoulli(
            torch.ones(3, dtype=torch.long, device=mojo_gpu),
            out=torch.empty(3, device=mojo_gpu),
        )
    # out shares storage with p and needs a resize: p is read first.
    base = torch.tensor([1.0, 1.0, 0.0, 0.0]).to(mojo_gpu)
    p = base[:2]
    torch.bernoulli(p, out=base)
    assert base.cpu().tolist() == [1.0, 1.0]


def test_fused_dropout_matches_native_dropout(mojo_gpu):
    """_fused_dropout(x, keep) is dropout_cuda<uint8_t> with p = keep: the
    same draws as native_dropout(x, 1 - keep) from the same state, the mask
    as uint8."""
    torch.manual_seed(1)
    for x in (torch.randn(4097), torch.randn(8, 9).t(), torch.randn(3, 5)):
        xd = x.to(mojo_gpu)
        device_module.manual_seed_all(99)
        out, mask = torch.ops.aten._fused_dropout(xd, 0.75)
        after = device_module.get_rng_state(xd.device)
        device_module.manual_seed_all(99)
        nout, nmask = torch.ops.aten.native_dropout(xd, 0.25, True)
        assert mask.dtype == torch.uint8
        assert torch.equal(mask.cpu(), nmask.cpu().to(torch.uint8))
        assert torch.equal(out.cpu(), nout.cpu())
        assert torch.equal(device_module.get_rng_state(xd.device), after)
    gen = torch.Generator(device=mojo_gpu)
    gen.manual_seed(5)
    a = torch.ops.aten._fused_dropout(xd, 0.5, gen)
    gen.manual_seed(5)
    b = torch.ops.aten._fused_dropout(xd, 0.5, gen)
    assert torch.equal(a[1].cpu(), b[1].cpu())
    e, em = torch.ops.aten._fused_dropout(torch.empty(0, 3, device=mojo_gpu), 0.5)
    assert e.shape == (0, 3) and em.dtype == torch.uint8
    with pytest.raises(RuntimeError, match='"fused_dropout" not implemented'):
        torch.ops.aten._fused_dropout(
            torch.ones(3, dtype=torch.long, device=mojo_gpu), 0.5
        )


def _philox_uniform(seed: int, positions: np.ndarray) -> np.ndarray:
    """curand_uniform of word `p` of subsequence 0 of Philox4x32-10(seed),
    in numpy: the reference for _fill_mem_eff_dropout_mask_."""
    m0, m1 = np.uint64(0xD2511F53), np.uint64(0xCD9E8D57)
    w0, w1 = np.uint32(0x9E3779B9), np.uint32(0xBB67AE85)
    ctr = positions.astype(np.uint64) >> np.uint64(2)
    c0 = (ctr & np.uint64(0xFFFFFFFF)).astype(np.uint32)
    c1 = (ctr >> np.uint64(32)).astype(np.uint32)
    c2 = np.zeros_like(c0)
    c3 = np.zeros_like(c0)
    k0 = np.full_like(c0, seed & 0xFFFFFFFF)
    k1 = np.full_like(c0, (seed >> 32) & 0xFFFFFFFF)
    for _ in range(10):
        p0 = m0 * c0.astype(np.uint64)
        p1 = m1 * c2.astype(np.uint64)
        hi0, lo0 = (p0 >> np.uint64(32)).astype(np.uint32), p0.astype(np.uint32)
        hi1, lo1 = (p1 >> np.uint64(32)).astype(np.uint32), p1.astype(np.uint32)
        c0, c1, c2, c3 = hi1 ^ c1 ^ k0, lo1, hi0 ^ c3 ^ k1, lo0
        k0 = k0 + w0
        k1 = k1 + w1
    words = np.stack([c0, c1, c2, c3])[positions % 4, np.arange(len(positions))]
    inv = np.float32(2.3283064e-10)
    return words.astype(np.float32) * inv + inv / np.float32(2)


@pytest.mark.parametrize("offset", [0, 4, 7])
def test_fill_mem_eff_dropout_mask_is_the_philox_stream(mojo_gpu, offset):
    """Element L is curand_uniform of word offset + L of subsequence 0: the
    values CUDA's rand_uniform_kernel writes, bit for bit (n_keys = 5 is
    not a multiple of 4, so rows start mid-block)."""
    seed = (1 << 40) + 12345
    t = torch.full((2, 3, 4, 5), -1.0, device=mojo_gpu)
    with np.errstate(over="ignore"):
        want = _philox_uniform(seed, np.arange(t.numel()) + offset)
    out = torch.ops.aten._fill_mem_eff_dropout_mask_(t, 0.3, seed, offset)
    assert out is t
    got = t.cpu().numpy().reshape(-1)
    assert np.array_equal(got, want)
    with pytest.raises(RuntimeError, match="is_contiguous"):
        torch.ops.aten._fill_mem_eff_dropout_mask_(t.transpose(2, 3), 0.1, 1, 0)
    with pytest.raises(RuntimeError, match="Float"):
        torch.ops.aten._fill_mem_eff_dropout_mask_(t.half(), 0.1, 1, 0)


# ---------------------------------------------------------------------------
# poisson / _standard_gamma / binomial: distribution checks with fixed seeds
# ---------------------------------------------------------------------------

_N = 200_000


def _moments(x: torch.Tensor) -> tuple[float, float]:
    x = x.double()
    return x.mean().item(), x.var().item()


@pytest.mark.parametrize(
    "dtype", [torch.float32, torch.float16, torch.bfloat16, torch.float64]
)
@pytest.mark.parametrize("rate", [0.0, 0.3, 4.0, 9.99, 10.0, 37.5, 1e4])
def test_poisson_moments(mojo_gpu, dtype, rate):
    if dtype == torch.float64:
        skip_if_metal(mojo_gpu, "Metal has no float64")
    if dtype != torch.float32 and rate == 1e4:
        pytest.skip("1e4 is not exact in the half dtypes' samples")
    torch.manual_seed(2)
    lam = torch.full((_N,), rate, dtype=dtype, device=mojo_gpu)
    x = torch.poisson(lam).cpu()
    assert x.dtype == dtype
    assert (x >= 0).all() and torch.equal(x, x.round())
    mean, var = _moments(x)
    if rate == 0.0:
        assert mean == 0.0 and var == 0.0
        return
    se = math.sqrt(rate / _N)
    assert abs(mean - rate) < 6 * se + 1e-3 * rate
    assert abs(var - rate) < 0.05 * rate


def test_poisson_seeding_generator_and_errors(mojo_gpu):
    lam = torch.rand(1000, device=mojo_gpu) * 20
    torch.manual_seed(3)
    a = torch.poisson(lam)
    torch.manual_seed(3)
    b = torch.poisson(lam)
    assert torch.equal(a.cpu(), b.cpu())
    c = torch.poisson(lam)
    assert not torch.equal(a.cpu(), c.cpu())
    gen = torch.Generator(device=mojo_gpu)
    gen.manual_seed(4)
    d = torch.poisson(lam, gen)
    gen.manual_seed(4)
    assert torch.equal(d.cpu(), torch.poisson(lam, gen).cpu())
    # per element rates, broadcast shape kept, strided input
    lam2 = torch.tensor([[1.0, 50.0]]).expand(50000, 2).to(mojo_gpu)
    m = torch.poisson(lam2.t()).cpu().double().mean(dim=1)
    assert abs(m[0] - 1.0) < 0.03 and abs(m[1] - 50.0) < 0.2
    assert torch.poisson(torch.empty(0, 2, device=mojo_gpu)).shape == (0, 2)
    for bad in (-1.0, float("nan")):
        with pytest.raises(RuntimeError, match="invalid Poisson rate"):
            torch.poisson(torch.tensor([1.0, bad], device=mojo_gpu))
    with pytest.raises(RuntimeError, match='"poisson_cuda" not implemented'):
        torch.poisson(torch.ones(3, dtype=torch.long, device=mojo_gpu))
    # CUDA's curand_poisson is a uint32 sampler: an infinite rate saturates
    # at UINT32_MAX, then converts to the output dtype.
    inf = torch.tensor([float("inf")], device=mojo_gpu)
    assert torch.poisson(inf).item() == float(torch.tensor(4294967295.0).float())
    if not _metal(mojo_gpu):
        assert torch.poisson(inf.double()).item() == 4294967295.0
    # An empty self returns before its probabilities are read.
    empty = torch.empty(0, device=mojo_gpu)
    assert empty.bernoulli_(torch.tensor(2.0, device=mojo_gpu)) is empty


@pytest.mark.parametrize(
    "dtype", [torch.float32, torch.float16, torch.bfloat16, torch.float64]
)
@pytest.mark.parametrize("alpha", [1e-3, 0.2, 1.0, 3.7, 50.0, 1e6])
def test_standard_gamma_moments(mojo_gpu, dtype, alpha):
    """Gamma(alpha, 1): mean alpha, variance alpha; samples positive (at
    least the dtype's smallest normal, CUDA's clamp)."""
    if dtype == torch.float64:
        skip_if_metal(mojo_gpu, "Metal has no float64")
    if alpha == 1e6 and dtype not in (torch.float32, torch.float64):
        pytest.skip("1e6 overflows float16; bfloat16 spacing there dwarfs sd 1e3")
    torch.manual_seed(5)
    a = torch.full((_N,), alpha, dtype=dtype, device=mojo_gpu)
    x = torch._standard_gamma(a).cpu()
    assert x.dtype == dtype
    assert (x >= torch.finfo(dtype).tiny).all() and torch.isfinite(x).all()
    mean, var = _moments(x)
    a_q = torch.tensor(alpha, dtype=dtype).double().item()  # alpha as stored
    rel = 0.02 if dtype in (torch.float32, torch.float64) else 0.03
    if alpha < 0.01:
        # Most mass below the smallest normal of the half types: only the
        # mean of the float types is meaningful.
        if dtype in (torch.float32, torch.float64):
            assert mean < 0.01
        return
    assert abs(mean - a_q) < 6 * math.sqrt(a_q / _N) + rel * 0.1 * a_q
    assert abs(var - a_q) < 0.06 * a_q


def test_standard_gamma_edges_and_errors(mojo_gpu):
    x = torch._standard_gamma(
        torch.tensor([0.0, float("nan"), float("inf")], device=mojo_gpu)
    ).cpu()
    # alpha = 0 samples 0, which CUDA's kernel clamps to the smallest normal.
    assert x[0] == torch.finfo(torch.float32).tiny
    assert math.isnan(x[1]) and x[2] == float("inf")
    # So an all-zero Dirichlet concentration gives 1 / k, not 0 / 0.
    d0 = torch._sample_dirichlet(torch.zeros(2, 4, device=mojo_gpu)).cpu()
    torch.testing.assert_close(d0, torch.full((2, 4), 0.25))
    with pytest.raises(RuntimeError, match='"gamma_cuda" not implemented'):
        torch._standard_gamma(torch.ones(2, dtype=torch.long, device=mojo_gpu))
    torch.manual_seed(6)
    a = (
        torch.distributions.Gamma(
            torch.full((_N,), 2.0, device=mojo_gpu),
            torch.full((_N,), 4.0, device=mojo_gpu),
        )
        .sample()
        .cpu()
    )
    assert abs(a.double().mean().item() - 0.5) < 0.01
    d = torch.distributions.Dirichlet(torch.tensor([1.0, 2.0, 3.0], device=mojo_gpu))
    s = d.sample((20000,)).cpu().double()
    torch.testing.assert_close(s.sum(-1), torch.ones(20000, dtype=torch.float64))
    torch.testing.assert_close(
        s.mean(0),
        torch.tensor([1 / 6, 2 / 6, 3 / 6], dtype=torch.float64),
        atol=0.01,
        rtol=0,
    )


@pytest.mark.parametrize(
    "dtype", [torch.float32, torch.float16, torch.bfloat16, torch.float64]
)
@pytest.mark.parametrize(
    "count,prob",
    [
        (0.0, 0.5),
        (10.0, 0.0),
        (10.0, 1.0),
        (7.0, 0.3),
        (40.0, 0.2),
        (100.0, 0.5),
        (60.0, 0.9),
        (1000.0, 0.97),
    ],
)
def test_binomial_moments(mojo_gpu, dtype, count, prob):
    """Inversion (count * p < 10) and BTRS, both tails of p, the p = 0 / 1
    and count = 0 edges."""
    if dtype == torch.float64:
        skip_if_metal(mojo_gpu, "Metal has no float64")
    torch.manual_seed(7)
    c = torch.full((_N,), count, dtype=dtype, device=mojo_gpu)
    p = torch.full((_N,), prob, dtype=dtype, device=mojo_gpu)
    x = torch.binomial(c, p).cpu()
    assert x.dtype == dtype
    assert (x >= 0).all() and (x <= count).all() and torch.equal(x, x.round())
    p_q = torch.tensor(prob, dtype=dtype).double().item()
    mean, var = _moments(x)
    want_mean, want_var = count * p_q, count * p_q * (1 - p_q)
    if want_var == 0:
        assert mean == want_mean and var == 0
        return
    assert abs(mean - want_mean) < 6 * math.sqrt(want_var / _N) + 1e-3
    assert abs(var - want_var) < 0.05 * want_var


def test_binomial_broadcast_seed_and_errors(mojo_gpu):
    count = torch.tensor([5.0, 50.0, 500.0], device=mojo_gpu)
    prob = torch.full((20000, 1), 0.4, device=mojo_gpu)
    torch.manual_seed(8)
    x = torch.binomial(count.expand(20000, 3), prob.expand(20000, 3))
    torch.manual_seed(8)
    y = torch.binomial(count.expand(20000, 3), prob.expand(20000, 3))
    assert torch.equal(x.cpu(), y.cpu())
    m = x.cpu().double().mean(0)
    torch.testing.assert_close(
        m, torch.tensor([2.0, 20.0, 200.0], dtype=torch.float64), rtol=0.02, atol=0.05
    )
    assert torch.binomial(count[:1].expand(4, 3), prob[:4]).shape == (4, 3)
    with pytest.raises(RuntimeError, match="floating-point dtypes for count"):
        torch.binomial(torch.ones(3, dtype=torch.long, device=mojo_gpu), count)
    with pytest.raises(RuntimeError, match="floating-point dtypes for prob"):
        torch.binomial(count, torch.ones(3, dtype=torch.long, device=mojo_gpu))
    with pytest.raises(RuntimeError, match="Found dtype Double but expected Float"):
        torch.binomial(count, count.double())
    nan = torch.binomial(
        torch.tensor([float("nan"), 3.0], device=mojo_gpu),
        torch.tensor([0.5, float("nan")], device=mojo_gpu),
    )
    assert nan.isnan().all()


def test_empty_samplers_advance_the_generator(mojo_gpu):
    """CUDA reserves the Philox counters before it looks at numel: a draw
    after an empty one comes from the advanced stream."""
    lam = torch.full((64,), 5.0, device=mojo_gpu)
    empty = torch.empty(0, device=mojo_gpu)
    for fn in (torch.poisson, torch._standard_gamma, lambda t: torch.binomial(t, t)):
        device_module.manual_seed_all(11)
        before = device_module.get_rng_state(lam.device)
        fn(empty)
        assert not torch.equal(device_module.get_rng_state(lam.device), before)


def test_binomial_boundaries_before_nan(mojo_gpu):
    """sample_binomial's `count <= 0 || prob <= 0` and `prob >= 1` come
    before its NaN case: (0, NaN) and (NaN, 0) are 0, (NaN, 1) is NaN."""
    nan = float("nan")
    c = torch.tensor([0.0, nan, nan, 4.0], device=mojo_gpu)
    p = torch.tensor([nan, 0.0, 1.0, nan], device=mojo_gpu)
    got = torch.binomial(c, p).cpu()
    assert got[0] == 0 and got[1] == 0 and got[2].isnan() and got[3].isnan()


@pytest.mark.parametrize("bad", [-0.1, 1.5, float("nan")])
def test_bernoulli_rejects_probabilities_outside_0_1(mojo_gpu, bad):
    p = torch.tensor([0.5, bad, 0.2], device=mojo_gpu)
    with pytest.raises(RuntimeError, match="Expected p_in >= 0 && p_in <= 1"):
        torch.bernoulli(p, out=torch.empty(3, device=mojo_gpu))
    with pytest.raises(RuntimeError, match="Expected p_in >= 0 && p_in <= 1"):
        torch.empty(3, device=mojo_gpu).bernoulli_(p)
    ok = torch.tensor(
        [0.0, 1.0],
        dtype=torch.float64 if not _metal(mojo_gpu) else torch.float32,
        device=mojo_gpu,
    )
    assert torch.bernoulli(ok).cpu().tolist() == [0.0, 1.0]
