"""Native backend: factories group (arange.start_out, uniform_, normal_,
native_dropout, native_dropout_backward, multinomial), plus a correctness smoke test of
the composite-decomposed factories this group deliberately does not
register (see ops_factories.mojo's module docstring and the final report).

Public torch API only, per the porting brief: no `aten_fast`,
`TorchMojoTensor`, or `_ctx_ptr` internals.
"""

import re

import pytest
import torch

from tests.native.conftest import skip_if_metal
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
