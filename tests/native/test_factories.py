"""Native backend: factories group (arange.start_out, uniform_, normal_,
native_dropout, native_dropout_backward), plus a correctness smoke test of
the composite-decomposed factories this group deliberately does not
register (see ops_factories.mojo's module docstring and the final report).

Public torch API only, per the porting brief: no `aten_fast`,
`TorchMojoTensor`, or `_ctx_ptr` internals.
"""

import pytest
import torch

from torch_mojo_backend import aten_functions, get_accelerators, native
from torch_mojo_backend.native import device_module


def _op_count_delta(name: str):
    """A context manager-less before/after pair for one native op's call
    count (docs/native_backend.md "Test support"), for ops with no
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


def test_normal_explicit_generator_declined(mojo_gpu):
    """The mojo generator's Philox state can't feed the host CPU draw this
    op performs, so an explicit generator is still rejected (see
    op_normal_'s comment)."""
    drawn = torch.empty(4, device=mojo_gpu)
    with pytest.raises(NotImplementedError, match="explicit generator"):
        drawn.normal_(0.0, 1.0, generator=torch.Generator(device=mojo_gpu))


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


def test_native_dropout_declines_non_float32(mojo_gpu):
    input = torch.randn(4, dtype=torch.float64, device=mojo_gpu)
    with pytest.raises(NotImplementedError, match="native_dropout"):
        torch.ops.aten.native_dropout.default(input, 0.5, True)


def test_native_dropout_declines_on_the_cpu_device(mojo_device):
    """`mojo:cpu` (the MAX CPU device) is declined too -- old eager path's
    `_on_gpu` gate, ported verbatim."""
    if torch.device(mojo_device) != device_module.cpu():
        pytest.skip("only meaningful on the mojo:cpu device")
    input = torch.randn(4, dtype=torch.float32, device=mojo_device)
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
    """7 is not a multiple of the 4-wide philox group: the ragged group must
    still consume its whole counter, or the next draw repeats it."""
    device_module.manual_seed_all(20260814)
    first = torch.empty(7, device=mojo_gpu).uniform_().cpu()
    state = device_module.get_rng_state(mojo_gpu)
    second = torch.empty(7, device=mojo_gpu).uniform_().cpu()
    assert not set(first.tolist()) & set(second.tolist())

    device_module.set_rng_state(state, mojo_gpu)
    replayed = torch.empty(7, device=mojo_gpu).uniform_().cpu()
    torch.testing.assert_close(replayed, second)


def test_uniform_misaligned_destination_indexes_the_stream_the_same_way(mojo_gpu):
    """An offset base cannot take the 16-byte vector store; the scalar store
    kernel must index the philox stream identically."""
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


def test_arange_needs_a_wide_accumulator(mojo_device):
    """Past 2**24 an fp32 running sum can no longer add 1.0.

    The reference depends on the device, because the accumulator width does:
    torch's CPU kernel specifies float64 for a float32 arange, so the MAX CPU
    device is compared against that scalar sequence (built explicitly --
    arm64's vectorized kernel rounds differently at this boundary); an
    accelerator is compared against the vendor backend's own answer, and the
    test skips where there is none to compare with rather than inventing one.
    """
    args = (16_777_217.0, 16_777_227.0, 1.0)
    result = torch.arange(*args, dtype=torch.float32, device=mojo_device).cpu()
    cpu_index = len(list(get_accelerators())) - 1
    if mojo_device == f"mojo:{cpu_index}":
        expected = torch.tensor(
            [args[0] + i * args[2] for i in range(10)], dtype=torch.float32
        )
    elif torch.cuda.is_available():
        expected = torch.arange(*args, dtype=torch.float32, device="cuda").cpu()
    elif torch.backends.mps.is_available():
        expected = torch.arange(*args, dtype=torch.float32, device="mps").cpu()
    else:
        pytest.skip("no native GPU reference for this MAX accelerator")
    assert torch.equal(result, expected)


def test_float64_factories_fill_scatter_and_arange(mojo_gpu):
    """fp64 is a separate kernel specialization from fp32 for each of these."""
    if list(get_accelerators())[0].api == "metal":
        pytest.skip("Metal has no float64")
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
    """random_.from / .to / random_ draw on the host and copy over."""
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
