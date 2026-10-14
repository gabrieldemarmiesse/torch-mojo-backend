"""Native `_foreach_*` / fused-optimizer ops (`tmb/ops/foreach.mojo`).

Migrated from `tests/test_eager_optimizer_ops.py` (deleted): only the tests
that exercise ops this group registers. That file also covered `lerp.Scalar`,
`sub.out`, `mul.out`, `linalg_vector_norm.out`, `any.out` and
`isin.Tensor_Tensor_out` directly -- those belong to the binary/reductions/
compare groups' own native test files, not here.

`native.op_count("aten::<op>")` (not `CallChecker`) verifies a call reached
this group's boxed kernel: the in-place foreach overloads (`_foreach_add_.
Scalar`, ...) have no `aten_functions` twin for `CallChecker`'s name-derived
matching to key off (torch.compile always sees the functionalized, non-
underscore form), and `_foreach_norm.Scalar` / `_fused_adamw_` have none
either. `_foreach_sqrt` is the one op here with a matching
`aten_functions.aten__foreach_sqrt` twin, so it uses `call_checker` like the
other groups' tests do.

Some correctness tests below exercise the SEQUENTIAL fallback (mixed dtype,
non-contiguous, non-fp32), which calls the per-tensor op (`add_.Scalar`,
`mul_.Tensor`, `sqrt`, `linalg_vector_norm`, ...) through `tmb_call_op`.
Until the groups that register those land in the integrated tree, those
particular tests raise NotImplementedError from the dispatcher (no
PrivateUse1 kernel and no CompositeExplicitAutograd registration for a bare
`Tensor.mul_(Scalar)` et al.) -- that is an integration gap, not a bug in
this file.
"""

import contextlib
import math
import re
from collections.abc import Sequence

import pytest
import torch

from tests.native.conftest import is_metal, ran, skip_if_metal
from torch_mojo_backend import aten_functions, get_accelerators, native
from torch_mojo_backend.testing import CallChecker

pytestmark = pytest.mark.xdist_group(name="group_native_foreach")


def _watch(op_name: str):
    """Reset the shim's per-op call counters and start counting."""
    del op_name  # kept as a parameter so call sites read like a doc comment
    native.op_counting(True)
    native.op_counts_reset()


def _assert_ran(op_name: str):
    assert native.op_count(op_name) > 0, f"{op_name} did not run natively"


def _sequential_copies(device: str, pairs: int) -> int:
    """The `aten::_copy_from` calls a batchable `_foreach_copy_` makes: none
    where the batched kernel runs, one per pair on Apple GPUs, which do not
    take that route yet (see `_batched_copy_device` in tmb/ops/foreach.mojo)."""
    return pairs if is_metal(device) else 0


_COPY_CAST_LISTS = [
    [0, 1, 2, 3, 7, 8, 9, 2047, 2048, 2049, 4095, 4096, 4097],
    [0, 0, 0],
    [513] * 65,
    [2049] * 129,
    [800, 800, 3840000, 2400, 1280000, 800, 800, 800, 5120000, 3200, 5120000, 800],
    [800] * 12,
    [357 * 789, 7 * 1025, 1025, 17, 1],
    [1048576],
    [0, 17, 0, 1025],
    [0 if i % 3 == 0 else i * 357 % 5001 for i in range(197)],
]


@pytest.mark.parametrize("sizes", _COPY_CAST_LISTS)
@pytest.mark.parametrize("non_blocking", [False, True])
def test_foreach_copy_cast_lists(mojo_gpu: str, sizes: list[int], non_blocking: bool):
    _check_foreach_copy_cast(mojo_gpu, sizes, 3, 5, non_blocking)


@pytest.mark.parametrize("sizes", _COPY_CAST_LISTS)
def test_foreach_copy_bf16_bits(mojo_gpu: str, sizes: list[int]):
    hosts = [
        (torch.arange(n + 9, dtype=torch.int64) * 7919 + 13).to(torch.int16)
        for n in sizes
    ]
    sources = [bits.view(torch.bfloat16).to(mojo_gpu) for bits in hosts]
    guards = [
        torch.full((n + 11,), -9, dtype=torch.bfloat16, device=mojo_gpu) for n in sizes
    ]
    srcs = [t[3 : 3 + n] for t, n in zip(sources, sizes, strict=True)]
    dsts = [t[5 : 5 + n] for t, n in zip(guards, sizes, strict=True)]
    versions = [t._version for t in dsts]
    _watch("aten::_foreach_copy_")
    torch._foreach_copy_(dsts, srcs)
    _assert_ran("aten::_foreach_copy_")
    for dst, src, host, guard, n, version in zip(
        dsts, sources, hosts, guards, sizes, versions, strict=True
    ):
        torch.testing.assert_close(
            dst.cpu().view(torch.int16), host[3 : 3 + n], rtol=0, atol=0
        )
        torch.testing.assert_close(src.cpu().view(torch.int16), host, rtol=0, atol=0)
        observed = guard.cpu()
        assert bool((observed[:5] == -9).all()) and bool(
            (observed[5 + n :] == -9).all()
        )
        assert dst._version == version + 1


@pytest.mark.parametrize("source_offset", range(4))
@pytest.mark.parametrize("destination_offset", range(4))
def test_foreach_copy_cast_alignment(
    mojo_gpu: str, source_offset: int, destination_offset: int
):
    _check_foreach_copy_cast(
        mojo_gpu, _COPY_CAST_LISTS[0], source_offset, destination_offset, False
    )


_BATCH_COPY_DTYPES = [
    torch.float64,
    torch.float32,
    torch.float16,
    torch.bfloat16,
    torch.int64,
    torch.int32,
    torch.int16,
    torch.int8,
    torch.uint8,
    torch.bool,
]
# Every dtype into and out of float32, the 16-bit floats both ways, rounding
# through float (float64 -> bf16, int64 -> half), wrapping integer narrowing,
# and bool normalization. Each pair is its own kernel build.
_BATCH_COPY_PAIRS = sorted(
    {(dtype, torch.float32) for dtype in _BATCH_COPY_DTYPES}
    | {(torch.float32, dtype) for dtype in _BATCH_COPY_DTYPES}
    | {
        (torch.bfloat16, torch.float16),
        (torch.float16, torch.bfloat16),
        (torch.float64, torch.bfloat16),
        (torch.int64, torch.float16),
        (torch.int64, torch.int8),
        (torch.int32, torch.uint8),
        (torch.int64, torch.bool),
        (torch.bool, torch.bfloat16),
    }
    | {(dtype, dtype) for dtype in _BATCH_COPY_DTYPES + [torch.uint16, torch.uint32]},
    key=str,
)
_BATCH_COPY_SIZES = [0, 1, 7, 17, 2047, 2049, 8193, 357 * 789]


def _batch_copy_source(n: int, src: torch.dtype, dst: torch.dtype) -> torch.Tensor:
    """Values whose conversion is defined behaviour in C++ for this pair."""
    index = torch.arange(n, dtype=torch.int64)
    if src == torch.bool:
        return index % 3 == 1
    if src.is_floating_point:
        values = ((index * 7919 + 13) % 20001 - 10000).double() / 97
        if dst in (torch.uint8, torch.uint16, torch.uint32):
            values = values.abs()
        if not dst.is_floating_point and dst != torch.bool:
            values = values.clamp(-127, 127)
        elif dst.is_floating_point and n >= 4:
            values[:4] = torch.tensor([float("nan"), float("inf"), -float("inf"), -0.0])
        return values.to(src)
    bits = (index * 2654435761 + 12345) % (1 << 40) - (1 << 39)
    if src in (torch.uint8, torch.uint16, torch.uint32):
        bits = bits.abs()
    return bits.to(src)


@pytest.mark.parametrize(("src_dtype", "dst_dtype"), _BATCH_COPY_PAIRS, ids=str)
@pytest.mark.parametrize("offsets", [(0, 0), (1, 3)])
def test_foreach_copy_batched_dtypes(mojo_gpu: str, src_dtype, dst_dtype, offsets):
    if torch.float64 in (src_dtype, dst_dtype):
        skip_if_metal(mojo_gpu, "Apple GPUs have no float64")
    source_offset, destination_offset = offsets
    hosts = [_batch_copy_source(n + 8, src_dtype, dst_dtype) for n in _BATCH_COPY_SIZES]
    sources = [host.to(mojo_gpu) for host in hosts]
    guards = [
        torch.full((n + 8,), 5, dtype=dst_dtype, device=mojo_gpu)
        for n in _BATCH_COPY_SIZES
    ]
    spans = list(zip(sources, guards, _BATCH_COPY_SIZES, strict=True))
    srcs = [s[source_offset : source_offset + n] for s, _, n in spans]
    dsts = [g[destination_offset : destination_offset + n] for _, g, n in spans]
    versions = [t._version for t in dsts]
    _watch("aten::_foreach_copy_")
    torch._foreach_copy_(dsts, srcs)
    _assert_ran("aten::_foreach_copy_")
    assert native.op_count("aten::_copy_from") == _sequential_copies(
        mojo_gpu, len(dsts)
    ), "took the wrong route"
    for host, guard, n, dst, version in zip(
        hosts, guards, _BATCH_COPY_SIZES, dsts, versions, strict=True
    ):
        expected = torch.full((n + 8,), 5, dtype=dst_dtype)
        expected[destination_offset : destination_offset + n].copy_(
            host[source_offset : source_offset + n]
        )
        torch.testing.assert_close(
            guard.cpu(), expected, rtol=0, atol=0, equal_nan=True
        )
        assert dst._version == version + 1


@pytest.mark.parametrize("shift", [0, 1, 5])
def test_foreach_copy_batched_interleaved_views(mojo_gpu: str, shift: int):
    """Sources and destinations alternate slots of one buffer, in shuffled
    pair order: disjoint, they batch; one source shifted into the next slot
    (partly or wholly) meets another pair's destination, and the whole list
    takes the sequential route."""
    slots, width = 200, 5
    host = torch.arange(slots * width + 2, dtype=torch.float32)
    base = host.to(mojo_gpu)
    order = torch.randperm(slots // 2, generator=torch.Generator().manual_seed(0))

    def views(tensor):
        dsts = [tensor[1 + 2 * i * width :][:width] for i in order.tolist()]
        srcs = [tensor[1 + (2 * i + 1) * width :][:width] for i in order.tolist()]
        srcs[37] = tensor[1 + (2 * order[37].item() + 1) * width + shift :][:width]
        return dsts, srcs

    expected_dsts, expected_srcs = views(host)
    for destination, source in zip(expected_dsts, expected_srcs, strict=True):
        destination.copy_(source)
    _watch("aten::_foreach_copy_")
    torch._foreach_copy_(*views(base))
    sequential = slots // 2 if shift else _sequential_copies(mojo_gpu, slots // 2)
    assert native.op_count("aten::_copy_from") == sequential
    torch.testing.assert_close(base.cpu(), host, rtol=0, atol=0)


def test_foreach_copy_batched_keeps_sequential_order(mojo_gpu: str):
    """Stock CUDA's batched `_foreach_copy_` races here (pair 1 reads what
    pair 0 writes); ours must fall back and match one-by-one copy_."""
    n = 1 << 20
    buffer = torch.zeros(2 * n, device=mojo_gpu)
    ones = torch.ones(n, dtype=torch.bfloat16, device=mojo_gpu)
    _watch("aten::_foreach_copy_")
    torch._foreach_copy_([buffer[:n], buffer[n:]], [ones, buffer[:n]])
    assert native.op_count("aten::_copy_from") == 2
    assert bool((buffer == 1).all())


@pytest.mark.parametrize(
    "dtype", [torch.float32, torch.float16, torch.bfloat16, torch.int64, torch.uint8]
)
@pytest.mark.parametrize("sizes", [[0, 357, 789, 17, 1, 0], [800] * 12, [0, 0, 0]])
@pytest.mark.parametrize("offsets", [(0, 0), (1, 3)])
def test_foreach_copy_adjacent_views(mojo_gpu: str, dtype, sizes, offsets):
    source_offset, destination_offset = offsets
    size = sum(sizes)
    host = ((torch.arange(size + 8) * 7919 + 13) % 127).to(dtype)
    source = host.to(mojo_gpu)
    destination = torch.full((size + 8,), 53, dtype=dtype, device=mojo_gpu)
    srcs = source[source_offset : source_offset + size].split(sizes)
    dsts = destination[destination_offset : destination_offset + size].split(sizes)
    pointers = [tensor.data_ptr() for tensor in dsts]
    source_version, destination_version = source._version, destination._version
    returned = torch._foreach_copy_(dsts, srcs)
    # The torch 2.11 stub says None, but its Python binding returns dsts.
    assert isinstance(returned, tuple)
    assert all(a is b for a, b in zip(returned, dsts, strict=True))
    expected = torch.full((size + 8,), 53, dtype=dtype)
    expected[destination_offset : destination_offset + size] = host[
        source_offset : source_offset + size
    ]
    torch.testing.assert_close(destination.cpu(), expected, rtol=0, atol=0)
    torch.testing.assert_close(source.cpu(), host, rtol=0, atol=0)
    assert [tensor.data_ptr() for tensor in dsts] == pointers
    assert source._version == source_version
    assert destination._version == destination_version + len(dsts)


@pytest.mark.parametrize(
    "mode",
    [
        "self",
        "disjoint",
        "dependency",
        "reversed",
        "strided",
        "source_gap",
        "destination_gap",
    ],
)
def test_foreach_copy_adjacent_aliases_and_fallbacks(mojo_gpu: str, mode: str):
    host = torch.arange(1, 25, dtype=torch.float32)
    base = host.to(mojo_gpu)

    def views(tensor):
        if mode == "self":
            return tensor[:12].split(3), tensor[:12].split(3)
        if mode == "disjoint":
            return tensor[12:].split(3), tensor[:12].split(3)
        if mode == "dependency":
            return [tensor[1:2], tensor[2:3]], [tensor[:1], tensor[1:2]]
        if mode == "reversed":
            return tensor[12:].split(3), list(reversed(tensor[:12].split(3)))
        if mode == "source_gap":
            return tensor[12:16].split(2), [tensor[:2], tensor[4:6]]
        if mode == "destination_gap":
            return [tensor[12:14], tensor[16:18]], tensor[:4].split(2)
        return tensor[1::2].split(3), tensor[::2].split(3)

    dsts, srcs = views(base)
    expected_dsts, expected_srcs = views(host)
    version = base._version
    for destination, source in zip(expected_dsts, expected_srcs, strict=True):
        destination.copy_(source)
    torch._foreach_copy_(dsts, srcs)
    torch.testing.assert_close(base.cpu(), host, rtol=0, atol=0)
    assert base._version == version + len(dsts)


def test_foreach_copy_adjacent_later_shape_error(mojo_gpu: str):
    source_cpu = torch.arange(5, dtype=torch.float32)
    destination_cpu = torch.full((4,), 53.0)
    source = source_cpu.to(mojo_gpu)
    destination = destination_cpu.to(mojo_gpu)
    reference_before = destination_cpu._version
    with pytest.raises(RuntimeError):
        for dst, src in zip(
            destination_cpu.split(2), source_cpu.split([2, 3]), strict=True
        ):
            dst.copy_(src)
    before = destination._version
    with pytest.raises(RuntimeError):
        torch._foreach_copy_(destination.split(2), source.split([2, 3]))
    torch.testing.assert_close(destination.cpu(), destination_cpu, rtol=0, atol=0)
    assert destination._version - before == destination_cpu._version - reference_before


def _check_foreach_copy_cast(
    device: str,
    sizes: list[int],
    source_offset: int,
    destination_offset: int,
    non_blocking: bool,
):
    patterns = torch.tensor(
        [
            0,
            0x80000000,
            1,
            0x80000001,
            0x8000,
            0x8001,
            0x18000,
            0x7FFFFF,
            0x800000,
            0x3F808000,
            0x3F818000,
            0xBF808000,
            0xBF818000,
            0x7F7FFFFF,
            0xFF7FFFFF,
            0x7F800000,
            0xFF800000,
            0x7FC00000,
            0xFFC00000,
            0x7F800001,
            0xFFFFFFFF,
        ],
        dtype=torch.int64,
    ).to(torch.int32)
    host_sources, device_sources, guards, srcs, dsts = [], [], [], [], []
    for n in sizes:
        bits = patterns[torch.arange(n + 32) % len(patterns)]
        host = bits.view(torch.float32)
        source = host.to(device)
        guard = torch.full((n + 32,), -9, dtype=torch.bfloat16, device=device)
        host_sources.append(host)
        device_sources.append(source)
        guards.append(guard)
        srcs.append(source[source_offset : source_offset + n])
        dsts.append(guard[destination_offset : destination_offset + n])
    versions = [t._version for t in dsts]
    _watch("aten::_foreach_copy_")
    assert torch.ops.aten._foreach_copy_.default(dsts, srcs, non_blocking) is None
    _assert_ran("aten::_foreach_copy_")
    for host, source, guard, dst, n, version in zip(
        host_sources, device_sources, guards, dsts, sizes, versions, strict=True
    ):
        expected = host[source_offset : source_offset + n].bfloat16()
        actual = dst.cpu()
        torch.testing.assert_close(actual, expected, rtol=0, atol=0, equal_nan=True)
        finite = ~torch.isnan(expected)
        torch.testing.assert_close(
            actual.view(torch.int16)[finite],
            expected.view(torch.int16)[finite],
            rtol=0,
            atol=0,
        )
        torch.testing.assert_close(
            source.cpu().view(torch.int32), host.view(torch.int32), rtol=0, atol=0
        )
        observed = guard.cpu()
        assert bool((observed[:destination_offset] == -9).all())
        assert bool((observed[destination_offset + n :] == -9).all())
        assert dst._version == version + 1


@pytest.mark.parametrize(
    "mode",
    [
        "strided",
        "broadcast",
        "mixed_dtype",
        "duplicate",
        "empty_duplicate",
        "chain",
        "cpu_source",
        "cpu_destination",
    ],
)
def test_foreach_copy_fallback(mojo_gpu: str, mode: str):
    source = torch.arange(14, dtype=torch.float32).reshape(2, 7)
    srcs = [source.to(mojo_gpu), (source + 20).to(mojo_gpu)]
    dsts = [
        torch.zeros((2, 7), device=mojo_gpu, dtype=torch.bfloat16) for _ in range(2)
    ]
    if mode == "strided":
        srcs = [t.t() for t in srcs]
        dsts = [t.t() for t in dsts]
    elif mode == "broadcast":
        srcs = [t[:1] for t in srcs]
    elif mode == "mixed_dtype":
        dsts[1] = torch.zeros((2, 7), device=mojo_gpu, dtype=torch.float32)
    elif mode == "duplicate":
        dsts[1] = dsts[0]
    elif mode == "empty_duplicate":
        dsts = [dsts[0][:0]] * 2
        srcs = [t[:0] for t in srcs]
    elif mode == "chain":
        dsts = [srcs[1], torch.zeros_like(srcs[1])]
    elif mode == "cpu_source":
        srcs = [t.cpu() for t in srcs]
    elif mode == "cpu_destination":
        dsts = [t.cpu() for t in dsts]
    versions = [t._version for t in dsts]
    expected = [
        torch.empty(t.shape, dtype=t.dtype).copy_(s.cpu())
        for t, s in zip(dsts, srcs, strict=True)
    ]
    if mode == "duplicate":
        expected[0] = expected[1]
    if mode == "chain":
        expected[1] = expected[0]
    torch.ops.aten._foreach_copy_.default(dsts, srcs, False)
    for actual, want, version in zip(dsts, expected, versions, strict=True):
        torch.testing.assert_close(actual.cpu(), want, rtol=0, atol=0)
        assert actual._version == version + (
            2 if mode in ("duplicate", "empty_duplicate") else 1
        )


def test_foreach_copy_list_errors(mojo_gpu: str):
    tensor = torch.empty(3, device=mojo_gpu)
    with pytest.raises(RuntimeError, match="at least one|empty"):
        torch._foreach_copy_([], [])
    with pytest.raises(RuntimeError, match="same number|same length|must not be empty"):
        torch._foreach_copy_([tensor], [])
    with pytest.raises(RuntimeError, match="same number|same length"):
        torch._foreach_copy_([tensor], [tensor, tensor])


def _fused_adamw_case(device: str, *, amsgrad: bool):
    """Nonuniform, nonzero AdamW state without shared storage (empty, 0-d,
    1-d, and 2-d shapes; deterministic values, no RNG)."""
    shapes = ((), (0,), (7,), (17, 65))

    def values(shape: tuple[int, ...], *, scale: float, offset: float) -> torch.Tensor:
        numel = torch.empty(shape).numel()
        return (
            torch.arange(numel, dtype=torch.float32)
            .mul(scale)
            .add(offset)
            .reshape(shape)
        )

    parameters = [
        values(shape, scale=0.003, offset=-0.75 + index * 0.1)
        for index, shape in enumerate(shapes)
    ]
    gradients = [
        values(shape, scale=-0.0007, offset=0.3 - index * 0.02)
        for index, shape in enumerate(shapes)
    ]
    exp_avgs = [
        values(shape, scale=0.0002, offset=-0.08 + index * 0.01)
        for index, shape in enumerate(shapes)
    ]
    exp_avg_sqs = [
        values(shape, scale=0.00001, offset=0.01 + index * 0.001)
        for index, shape in enumerate(shapes)
    ]
    max_exp_avg_sqs = [value.mul(1.25) for value in exp_avg_sqs] if amsgrad else []
    state_steps = [
        torch.tensor(float(step), dtype=torch.float32) for step in (3, 5, 11, 17)
    ]
    groups = (
        parameters,
        gradients,
        exp_avgs,
        exp_avg_sqs,
        max_exp_avg_sqs,
        state_steps,
    )
    return tuple([[value.to(device) for value in group] for group in groups])


@pytest.mark.parametrize(
    ("amsgrad", "maximize"), [(False, False), (False, True), (True, False)]
)
def test_fused_adamw_matches_cpu(mojo_gpu: str, amsgrad: bool, maximize: bool):
    _watch("aten::_fused_adamw_")
    cpu_groups = _fused_adamw_case("cpu", amsgrad=amsgrad)
    mojo_groups = _fused_adamw_case(mojo_gpu, amsgrad=amsgrad)
    kwargs = {
        "lr": 0.025,
        "beta1": 0.8,
        "beta2": 0.95,
        "weight_decay": 0.1,
        "eps": 1e-8,
        "amsgrad": amsgrad,
        "maximize": maximize,
    }

    assert torch.ops.aten._fused_adamw_.default(*cpu_groups, **kwargs) is None
    assert torch.ops.aten._fused_adamw_.default(*mojo_groups, **kwargs) is None

    for cpu_group, mojo_group in zip(cpu_groups, mojo_groups, strict=True):
        for expected, actual in zip(cpu_group, mojo_group, strict=True):
            torch.testing.assert_close(actual.cpu(), expected, rtol=2e-6, atol=2e-7)
    _assert_ran("aten::_fused_adamw_")


def test_fused_adamw_tensor_lr_matches_cpu(mojo_gpu: str):
    _watch("aten::_fused_adamw_.tensor_lr")
    cpu_groups = _fused_adamw_case("cpu", amsgrad=False)
    mojo_groups = _fused_adamw_case(mojo_gpu, amsgrad=False)
    cpu_lr = torch.tensor(0.0125, dtype=torch.float32)
    mojo_lr = cpu_lr.to(mojo_gpu)
    kwargs = {
        "beta1": 0.8,
        "beta2": 0.95,
        "weight_decay": 0.1,
        "eps": 1e-8,
        "amsgrad": False,
        "maximize": False,
    }

    torch.ops.aten._fused_adamw_.tensor_lr(*cpu_groups, lr=cpu_lr, **kwargs)
    torch.ops.aten._fused_adamw_.tensor_lr(*mojo_groups, lr=mojo_lr, **kwargs)

    for cpu_group, mojo_group in zip(cpu_groups, mojo_groups, strict=True):
        for expected, actual in zip(cpu_group, mojo_group, strict=True):
            torch.testing.assert_close(actual.cpu(), expected, rtol=2e-6, atol=2e-7)
    _assert_ran("aten::_fused_adamw_.tensor_lr")


def test_fused_adamw_grad_scale_and_found_inf(mojo_gpu: str):
    """found_inf=1 must skip every write; found_inf=0 with grad_scale halves
    the (unwritten) gradient's effective magnitude but otherwise runs."""
    _watch("aten::_fused_adamw_")
    for found_inf_value in (0.0, 1.0):
        cpu_groups = _fused_adamw_case("cpu", amsgrad=True)
        mojo_groups = _fused_adamw_case(mojo_gpu, amsgrad=True)
        before = [[t.cpu().clone() for t in group] for group in mojo_groups]
        cpu_grad_scale = torch.tensor(2.0, dtype=torch.float32)
        mojo_grad_scale = cpu_grad_scale.to(mojo_gpu)
        cpu_found_inf = torch.tensor(found_inf_value, dtype=torch.float32)
        mojo_found_inf = cpu_found_inf.to(mojo_gpu)
        kwargs = {
            "lr": 0.025,
            "beta1": 0.8,
            "beta2": 0.95,
            "weight_decay": 0.1,
            "eps": 1e-8,
            "amsgrad": True,
            "maximize": False,
        }
        torch.ops.aten._fused_adamw_.default(
            *cpu_groups, grad_scale=cpu_grad_scale, found_inf=cpu_found_inf, **kwargs
        )
        torch.ops.aten._fused_adamw_.default(
            *mojo_groups, grad_scale=mojo_grad_scale, found_inf=mojo_found_inf, **kwargs
        )
        for cpu_group, mojo_group in zip(cpu_groups, mojo_groups, strict=True):
            for expected, actual in zip(cpu_group, mojo_group, strict=True):
                torch.testing.assert_close(actual.cpu(), expected, rtol=2e-6, atol=2e-7)
        if found_inf_value == 1.0:
            for before_group, after_group in zip(before, mojo_groups, strict=True):
                for expected, actual in zip(before_group, after_group, strict=True):
                    torch.testing.assert_close(actual.cpu(), expected, rtol=0, atol=0)
    _assert_ran("aten::_fused_adamw_")


def test_fused_adamw_bumps_the_version_of_every_list_it_writes(mojo_gpu: str):
    """With grad_scale the kernel writes the UNSCALED gradient back into
    `grads`, so autograd's version counter has to move there too -- otherwise
    a saved gradient that this mutated still looks pristine to
    `_version`-based checks."""
    _watch("aten::_fused_adamw_")
    groups = _fused_adamw_case(mojo_gpu, amsgrad=False)
    parameters, grads, exp_avgs, exp_avg_sqs, _, _ = groups
    before = [
        [t._version for t in group]
        for group in (parameters, grads, exp_avgs, exp_avg_sqs)
    ]
    torch.ops.aten._fused_adamw_.default(
        *groups,
        lr=0.025,
        beta1=0.8,
        beta2=0.95,
        weight_decay=0.1,
        eps=1e-8,
        amsgrad=False,
        maximize=False,
        grad_scale=torch.tensor(2.0, device=mojo_gpu),
        found_inf=torch.tensor(0.0, device=mojo_gpu),
    )
    after = [
        [t._version for t in group]
        for group in (parameters, grads, exp_avgs, exp_avg_sqs)
    ]
    names = ("params", "grads", "exp_avgs", "exp_avg_sqs")
    for name, old, new in zip(names, before, after, strict=True):
        assert all(b < a for b, a in zip(old, new, strict=True)), (
            f"{name}: versions {old} -> {new}"
        )
    _assert_ran("aten::_fused_adamw_")


def test_fused_adamw_validates_every_tensor_before_write(mojo_gpu: str):
    _watch("aten::_fused_adamw_")
    parameters = [
        torch.tensor([1.0, -2.0], device=mojo_gpu),
        torch.tensor([3.0, -4.0, 5.0], device=mojo_gpu),
    ]
    grads = [torch.ones_like(p) for p in parameters]
    exp_avgs = [torch.zeros_like(p) for p in parameters]
    exp_avg_sqs = [torch.ones_like(p) for p in parameters]
    # Only the SECOND entry is malformed: a partial-launch implementation
    # would already have updated the valid first one before discovering this.
    exp_avg_sqs[1] = torch.ones(4, device=mojo_gpu)
    steps = [torch.tensor(1.0, device=mojo_gpu) for _ in parameters]
    mutable = (parameters, grads, exp_avgs, exp_avg_sqs)
    snapshot = [[t.cpu().clone() for t in group] for group in mutable]

    with pytest.raises(RuntimeError, match="same dtype, device, and layout"):
        torch.ops.aten._fused_adamw_.default(
            parameters,
            grads,
            exp_avgs,
            exp_avg_sqs,
            [],
            steps,
            lr=0.01,
            beta1=0.9,
            beta2=0.95,
            weight_decay=0.1,
            eps=1e-8,
            amsgrad=False,
            maximize=False,
        )
    for snap_group, actual_group in zip(snapshot, mutable, strict=True):
        for snap, actual in zip(snap_group, actual_group, strict=True):
            torch.testing.assert_close(actual.cpu(), snap, rtol=0, atol=0)


def test_fused_adamw_optimizer_two_groups_matches_cpu(mojo_gpu: str):
    """The public torch.optim.AdamW(fused=True) entry point end to end."""
    _watch("aten::_fused_adamw_")
    initial = [
        torch.linspace(-1.0, 1.0, 31, dtype=torch.float32),
        torch.linspace(0.5, -0.75, 35, dtype=torch.float32).reshape(5, 7),
    ]
    cpu_parameters = [torch.nn.Parameter(v.clone()) for v in initial]
    mojo_parameters = [torch.nn.Parameter(v.to(mojo_gpu)) for v in initial]
    cpu_optimizer = torch.optim.AdamW(
        [
            {"params": [cpu_parameters[0]], "weight_decay": 0.1},
            {"params": [cpu_parameters[1]], "weight_decay": 0.0},
        ],
        lr=0.0125,
        betas=(0.8, 0.95),
        eps=1e-8,
        fused=True,
    )
    mojo_optimizer = torch.optim.AdamW(
        [
            {"params": [mojo_parameters[0]], "weight_decay": 0.1},
            {"params": [mojo_parameters[1]], "weight_decay": 0.0},
        ],
        lr=0.0125,
        betas=(0.8, 0.95),
        eps=1e-8,
        fused=True,
    )

    for step in range(2):
        for index, (cpu_parameter, mojo_parameter) in enumerate(
            zip(cpu_parameters, mojo_parameters, strict=True)
        ):
            gradient = torch.linspace(
                -0.2 + step * 0.03,
                0.35 - index * 0.02,
                cpu_parameter.numel(),
                dtype=torch.float32,
            ).reshape(cpu_parameter.shape)
            cpu_parameter.grad = gradient.clone()
            mojo_parameter.grad = gradient.to(mojo_gpu)
        cpu_optimizer.step()
        mojo_optimizer.step()

    assert all(group["fused"] is True for group in mojo_optimizer.param_groups)
    for cpu_parameter, mojo_parameter in zip(
        cpu_parameters, mojo_parameters, strict=True
    ):
        torch.testing.assert_close(
            mojo_parameter.cpu(), cpu_parameter, rtol=2e-6, atol=2e-7
        )
        cpu_state = cpu_optimizer.state[cpu_parameter]
        mojo_state = mojo_optimizer.state[mojo_parameter]
        for name in ("exp_avg", "exp_avg_sq", "step"):
            torch.testing.assert_close(
                mojo_state[name].cpu(), cpu_state[name], rtol=2e-6, atol=2e-7
            )
        assert mojo_state["step"].device == torch.device(mojo_gpu)
    _assert_ran("aten::_fused_adamw_")


def _amp_grads(dtype: torch.dtype, *, poison: float | None) -> list[torch.Tensor]:
    """CPU grads: an empty tensor, a misaligned view, and sizes around the
    batched kernel's chunk and vector boundaries; `poison` lands in the last."""
    base = torch.arange(65_541, dtype=torch.float32).mul(0.37).remainder(9.0).sub(4.5)
    grads = [
        base[:7].clone(),
        base[1:1106].clone().reshape(17, 65),
        base[:0].clone(),
        base[3:30].clone().reshape(3, 3, 3),
        base[2:65_541].clone(),
    ]
    if poison is not None:
        grads[-1][40_000] = poison
    return [g.to(dtype) for g in grads]


@pytest.mark.parametrize(
    "dtype", [torch.float32, torch.float16, torch.bfloat16], ids=["f32", "f16", "bf16"]
)
@pytest.mark.parametrize("inv_scale", [0.25, 1.0, 1.0 / 3.0])
@pytest.mark.parametrize(
    "poison", [None, float("inf"), float("-inf"), float("nan")], ids=str
)
def test_amp_foreach_non_finite_check_and_unscale_matches_cpu(
    mojo_gpu: str, dtype: torch.dtype, inv_scale: float, poison: float | None
):
    if dtype != torch.float32:
        skip_if_metal(mojo_gpu, "the Metal batched foreach arm is float32 only")
    _watch("aten::_amp_foreach_non_finite_check_and_unscale_")
    cpu = _amp_grads(dtype, poison=poison)
    # Offset views put the batched kernel's vector body off a 16-byte boundary.
    storage = [torch.empty(g.numel() + 1, dtype=dtype, device=mojo_gpu) for g in cpu]
    ours = [s[1:].view(g.shape).copy_(g) for s, g in zip(storage, cpu, strict=True)]
    cpu_found = torch.zeros(())
    cpu_inv = torch.tensor(inv_scale)
    found = cpu_found.to(mojo_gpu)
    inv = cpu_inv.to(mojo_gpu)
    versions = [g._version for g in ours]
    torch._amp_foreach_non_finite_check_and_unscale_(cpu, cpu_found, cpu_inv)
    torch._amp_foreach_non_finite_check_and_unscale_(ours, found, inv)
    for expected, actual in zip(cpu, ours, strict=True):
        torch.testing.assert_close(
            actual.cpu(), expected, rtol=0, atol=0, equal_nan=True
        )
    assert found.item() == cpu_found.item() == (0.0 if poison is None else 1.0)
    assert all(g._version > v for g, v in zip(ours, versions, strict=True))
    _assert_ran("aten::_amp_foreach_non_finite_check_and_unscale_")


def test_amp_unscale_mixed_dtypes_and_non_dense_fall_back(mojo_gpu: str):
    """A non-dense view (through a contiguous temporary) and a list that mixes
    dtypes (one launch per tensor) give the CPU result; found_inf already set
    stays set."""
    _watch("aten::_amp_foreach_non_finite_check_and_unscale_")
    cpu_base = torch.arange(96, dtype=torch.float32).reshape(8, 12).sub(40.0)
    cpu_base[5, 6] = float("inf")
    cpu = [cpu_base.t()[::2, 1:], torch.linspace(-3, 3, 37), cpu_base[:, ::3]]
    base = cpu_base.to(mojo_gpu)
    ours = [base.t()[::2, 1:], cpu[1].to(mojo_gpu), base[:, ::3]]
    if get_accelerators()[0].api != "metal":
        cpu[1] = cpu[1].to(torch.bfloat16)
        ours[1] = ours[1].to(torch.bfloat16)
    cpu_found, cpu_inv = torch.zeros(()), torch.tensor(0.5)
    found, inv = cpu_found.to(mojo_gpu), cpu_inv.to(mojo_gpu)
    torch._amp_foreach_non_finite_check_and_unscale_(cpu, cpu_found, cpu_inv)
    torch._amp_foreach_non_finite_check_and_unscale_(ours, found, inv)
    torch.testing.assert_close(base.cpu(), cpu_base, rtol=0, atol=0)
    torch.testing.assert_close(ours[1].cpu(), cpu[1], rtol=0, atol=0)
    assert found.item() == 1.0

    finite = [torch.ones(5, device=mojo_gpu)]
    torch._amp_foreach_non_finite_check_and_unscale_(finite, found, inv)
    assert found.item() == 1.0, "a set found_inf must never be cleared"
    torch._amp_foreach_non_finite_check_and_unscale_([], found, inv)
    _assert_ran("aten::_amp_foreach_non_finite_check_and_unscale_")


def test_amp_unscale_rejects_bad_scalars(mojo_gpu: str):
    grads = [torch.ones(3, device=mojo_gpu)]
    with pytest.raises(RuntimeError, match="found_inf"):
        torch._amp_foreach_non_finite_check_and_unscale_(
            grads, torch.zeros(2, device=mojo_gpu), torch.ones((), device=mojo_gpu)
        )
    with pytest.raises(RuntimeError, match="inv_scale"):
        torch._amp_foreach_non_finite_check_and_unscale_(
            grads,
            torch.zeros((), device=mojo_gpu),
            torch.ones((), dtype=torch.float16, device=mojo_gpu),
        )


def test_amp_update_scale_matches_cpu(mojo_gpu: str):
    """Every branch of the state machine against ATen's CPU kernel: growth
    after `growth_interval` clean steps, backoff (and tracker reset) on inf,
    and a growth that would overflow float32 leaving the scale unchanged."""
    _watch("aten::_amp_update_scale_")
    found_seq = [0, 0, 0, 1, 0, 0, 0, 0, 1, 1, 0, 0, 0]
    for initial in (65536.0, 3.0e38, 0.1):
        cpu_scale = torch.tensor(initial)
        cpu_tracker = torch.zeros((), dtype=torch.int32)
        scale = cpu_scale.to(mojo_gpu)
        tracker = cpu_tracker.to(mojo_gpu)
        for found_value in found_seq:
            cpu_found = torch.tensor(float(found_value))
            args = (2.0, 0.5, 3)
            torch._amp_update_scale_(cpu_scale, cpu_tracker, cpu_found, *args)
            version = tracker._version
            returned = torch._amp_update_scale_(
                scale, tracker, cpu_found.to(mojo_gpu), *args
            )
            assert returned.data_ptr() == scale.data_ptr()
            assert tracker._version > version
            assert scale.item() == cpu_scale.item()
            assert tracker.item() == cpu_tracker.item()
    _assert_ran("aten::_amp_update_scale_")


def test_amp_update_scale_rejects_bad_tracker(mojo_gpu: str):
    with pytest.raises(RuntimeError, match="growth_tracker"):
        torch._amp_update_scale_(
            torch.ones((), device=mojo_gpu),
            torch.zeros((), device=mojo_gpu),  # float32, not int32
            torch.zeros((), device=mojo_gpu),
            2.0,
            0.5,
            10,
        )


def _grad_scaler_run(
    device: str, fused: bool, steps: int
) -> tuple[list[torch.Tensor], list[float]]:
    torch.manual_seed(0)
    model = torch.nn.Sequential(
        torch.nn.Linear(16, 32), torch.nn.ReLU(), torch.nn.Linear(32, 4)
    )
    model.to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=1e-2, fused=fused)
    scaler = torch.amp.GradScaler(
        torch.device(device).type, init_scale=2.0**10, growth_interval=2
    )
    x = torch.linspace(-1.0, 1.0, 8 * 16).reshape(8, 16).to(device)
    target = torch.arange(8).remainder(4).to(device)
    scales = []
    for step in range(steps):
        loss = torch.nn.functional.cross_entropy(model(x), target)
        if step == 3:
            loss = loss * float("inf")  # this step must be skipped, scale halved
        scaler.scale(loss).backward()
        if step % 2:
            scaler.unscale_(optimizer)  # the explicit unscale-then-clip recipe
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        scaler.step(optimizer)
        scaler.update()
        optimizer.zero_grad()
        scales.append(scaler.get_scale())
    return [p.detach().cpu() for p in model.parameters()], scales


@pytest.mark.parametrize("fused", [False, True], ids=["foreach", "fused"])
def test_grad_scaler_training_matches_cpu(mojo_gpu: str, fused: bool):
    """torch.amp.GradScaler end to end, against the same run on CPU: growth,
    an inf step that is skipped with backoff, explicit unscale_ + clipping,
    and both the foreach and the fused AdamW routes."""
    skip_if_metal(mojo_gpu, "GradScaler.unscale_ needs float64")
    _watch("aten::_amp_update_scale_")
    cpu_params, cpu_scales = _grad_scaler_run("cpu", fused, steps=8)
    params, scales = _grad_scaler_run(mojo_gpu, fused, steps=8)
    assert scales == cpu_scales
    assert scales[3] == scales[2] / 2
    for expected, actual in zip(cpu_params, params, strict=True):
        torch.testing.assert_close(actual, expected, rtol=1e-5, atol=1e-6)
    _assert_ran("aten::_amp_update_scale_")
    _assert_ran("aten::_amp_foreach_non_finite_check_and_unscale_")


def test_grad_scaler_with_float16_autocast(mojo_gpu: str):
    """The recipe from torch's AMP docs trains on the mojo device."""
    skip_if_metal(mojo_gpu, "GradScaler.unscale_ needs float64")
    device = torch.device(mojo_gpu)
    torch.manual_seed(0)
    model = torch.nn.Linear(16, 4).to(device)
    optimizer = torch.optim.SGD(model.parameters(), lr=0.5)
    scaler = torch.amp.GradScaler(device.type)
    x = torch.randn(64, 16).to(device)
    target = x[:, :4].argmax(dim=1)
    losses = []
    for _ in range(30):
        with torch.autocast(device.type, dtype=torch.float16):
            loss = torch.nn.functional.cross_entropy(model(x), target)
        scaler.scale(loss).backward()
        scaler.step(optimizer)
        scaler.update()
        optimizer.zero_grad()
        losses.append(loss.item())
    assert losses[-1] < 0.5 * losses[0]
    assert scaler.get_scale() > 0


def _foreach_lists(device: str) -> list[list[torch.Tensor]]:
    """Nonuniform FP32 lists: empties, multi-dim, and one tensor crossing the
    65_536-element chunk boundary. The third list stays bounded away from 0
    (used as a divisor by callers that need one)."""
    shapes = ((7,), (17, 65), (0,), (5, 3, 2), (65_539,))

    def values(shape: tuple[int, ...], *, scale: float, offset: float) -> torch.Tensor:
        numel = torch.empty(shape).numel()
        return (
            torch.arange(numel, dtype=torch.float32)
            .mul(scale)
            .add(offset)
            .reshape(shape)
        )

    mutated = [
        values(shape, scale=0.003, offset=-0.4 + index * 0.1)
        for index, shape in enumerate(shapes)
    ]
    operands = [
        values(shape, scale=-0.0007, offset=0.3 - index * 0.02)
        for index, shape in enumerate(shapes)
    ]
    divisors = [
        values(shape, scale=0.0002, offset=-0.08 + index * 0.01).abs().add(0.5)
        for index, shape in enumerate(shapes)
    ]
    return [[t.to(device) for t in group] for group in (mutated, operands, divisors)]


_FOREACH_BATCHED_OPS = [
    ("aten::_foreach_add_.Scalar", lambda a, b, c: torch._foreach_add_(a, 1e-3)),
    (
        "aten::_foreach_addcmul_.Scalar",
        lambda a, b, c: torch._foreach_addcmul_(a, b, c, value=0.01),
    ),
    ("aten::_foreach_lerp_.Scalar", lambda a, b, c: torch._foreach_lerp_(a, b, 0.1)),
    ("aten::_foreach_mul_.Scalar", lambda a, b, c: torch._foreach_mul_(a, 0.998)),
]


@pytest.mark.parametrize(("op_name", "apply"), _FOREACH_BATCHED_OPS)
def test_batched_foreach_elementwise_matches_cpu(mojo_gpu: str, op_name: str, apply):
    """The batched kernel launch runs and matches CPU's sequential result.

    Not checked here: `Tensor(a!)[]` mutations get no automatic `_version`
    bump the way a single `Tensor(a!)` op does (`add_.Scalar`/`zero_` do;
    `_foreach_add_.Scalar` measurably does not) -- see this group's final
    report for why (it needs a C-level primitive the shim does not expose
    yet, and this file cannot add one).
    """
    _watch(op_name)
    cpu_lists = _foreach_lists("cpu")
    mojo_lists = _foreach_lists(mojo_gpu)
    apply(*cpu_lists)
    apply(*mojo_lists)
    for expected, actual in zip(cpu_lists[0], mojo_lists[0], strict=True):
        torch.testing.assert_close(actual.cpu(), expected, rtol=2e-6, atol=2e-7)
    _assert_ran(op_name)


@pytest.mark.parametrize("dtype", [torch.float16, torch.bfloat16], ids=["f16", "bf16"])
def test_batched_foreach_mul_scalar_reduced_precision(
    mojo_gpu: str, dtype: torch.dtype
):
    """Half/bfloat16 lists also take the batched path (mul_/add_ widen to
    FP32 and narrow back, matching the per-tensor `mul_.Scalar` route)."""
    _watch("aten::_foreach_mul_.Scalar")
    host = [torch.linspace(-3.0, 4.0, length).to(dtype) for length in (7, 65_539, 1)]
    device = [tensor.to(mojo_gpu) for tensor in host]
    torch._foreach_mul_(device, 0.998)
    for expected, actual in zip(host, device, strict=True):
        torch.testing.assert_close(
            actual.cpu(), (expected.float() * 0.998).to(dtype), rtol=0, atol=0
        )
    _assert_ran("aten::_foreach_mul_.Scalar")


def test_foreach_mul_tensor_matches_cpu(mojo_gpu: str):
    _watch("aten::_foreach_mul_.Tensor")
    cpu_lists = _foreach_lists("cpu")
    mojo_lists = _foreach_lists(mojo_gpu)
    cpu_scalar = torch.tensor(0.25, dtype=torch.float32)
    mojo_scalar = cpu_scalar.to(mojo_gpu)
    torch._foreach_mul_(cpu_lists[0], cpu_scalar)
    torch._foreach_mul_(mojo_lists[0], mojo_scalar)
    for expected, actual in zip(cpu_lists[0], mojo_lists[0], strict=True):
        torch.testing.assert_close(actual.cpu(), expected, rtol=2e-6, atol=2e-7)
    _assert_ran("aten::_foreach_mul_.Tensor")


def test_foreach_mul_tensor_rejects_non_0d_other(mojo_gpu: str):
    tensors = [torch.tensor([1.0, 2.0], device=mojo_gpu)]
    other = torch.tensor([1.0, 2.0], device=mojo_gpu)
    with pytest.raises(RuntimeError, match="0 dim"):
        torch._foreach_mul_(tensors, other)


def test_batched_foreach_sqrt_matches_cpu(mojo_gpu: str, call_checker: CallChecker):
    call_checker.register(aten_functions.aten__foreach_sqrt)
    # abs() on the CPU tensor, then transfer: aten::abs is another group's op,
    # not yet registered in this worktree.
    cpu_inputs = [tensor.abs() for tensor in _foreach_lists("cpu")[0]]
    mojo_inputs = [tensor.to(mojo_gpu) for tensor in cpu_inputs]
    # Correctly-rounded float32 sqrt (via float64), not torch's CPU float32
    # sqrt (MKL vsSqrt is documented <=1ulp and is not itself bit-exact).
    expected = [tensor.double().sqrt().to(tensor.dtype) for tensor in cpu_inputs]
    actual = torch._foreach_sqrt(mojo_inputs)
    for expected_out, actual_out in zip(expected, actual, strict=True):
        torch.testing.assert_close(actual_out.cpu(), expected_out, rtol=0, atol=0)


def test_foreach_norm_scalar_matches_cpu(mojo_gpu: str):
    _watch("aten::_foreach_norm.Scalar")
    tensors = [t.abs().add(0.1) for t in _foreach_lists("cpu")[0] if t.numel() > 0]
    mojo_tensors = [t.to(mojo_gpu) for t in tensors]
    expected = torch._foreach_norm(tensors, 2)
    actual = torch._foreach_norm(mojo_tensors, 2)
    for e, a in zip(expected, actual, strict=True):
        torch.testing.assert_close(a.cpu(), e, rtol=2e-6, atol=2e-7)
    _assert_ran("aten::_foreach_norm.Scalar")


def test_foreach_norm_scalar_ord_one(mojo_gpu: str):
    """ord 1 is a batched reduction too (CUDA's L1 LpNormFunctor)."""
    tensors = [t.abs().add(0.1) for t in _foreach_lists("cpu")[0] if t.numel() > 0]
    mojo_tensors = [t.to(mojo_gpu) for t in tensors]
    expected = torch._foreach_norm(tensors, 1)
    actual = torch._foreach_norm(mojo_tensors, 1)
    for e, a in zip(expected, actual, strict=True):
        torch.testing.assert_close(a.cpu(), e, rtol=2e-6, atol=2e-7)


def test_batched_foreach_falls_back_for_non_f32(mojo_gpu: str):
    """Unsupported regimes (here: integer dtype) reach ATen's ordinary
    per-tensor semantics unchanged, via the sequential fallback."""
    cpu_tensors = [torch.arange(5), torch.arange(3)]
    mojo_tensors = [tensor.to(mojo_gpu) for tensor in cpu_tensors]
    torch._foreach_mul_(cpu_tensors, 3)
    torch._foreach_mul_(mojo_tensors, 3)
    for expected, actual in zip(cpu_tensors, mojo_tensors, strict=True):
        torch.testing.assert_close(actual.cpu(), expected, rtol=0, atol=0)


def test_batched_foreach_mixed_dtype_list_falls_back(mojo_gpu: str):
    """A two-dtype list is not one launch; still gets ATen's answer via the
    per-tensor fallback."""
    mixed = [
        torch.tensor([1.5, -2.0]).to(mojo_gpu),
        torch.tensor([1.5, -2.0], dtype=torch.bfloat16).to(mojo_gpu),
    ]
    torch._foreach_mul_(mixed, 2.0)
    torch.testing.assert_close(
        mixed[0].cpu(), torch.tensor([3.0, -4.0]), rtol=0, atol=0
    )
    torch.testing.assert_close(
        mixed[1].cpu(), torch.tensor([3.0, -4.0], dtype=torch.bfloat16), rtol=0, atol=0
    )


@pytest.mark.parametrize(
    ("op_name", "apply"),
    [
        ("div", lambda a, b, c: torch._foreach_div_(a, [0.5, -1.5, 2.0, 0.25, -0.125])),
        (
            "addcdiv",
            lambda a, b, c: torch._foreach_addcdiv_(
                a, b, c, [0.5, -1.5, 2.0, 0.25, -0.125]
            ),
        ),
    ],
)
def test_foreach_scalarlist_ops_match_cpu(mojo_gpu: str, op_name: str, apply):
    """`_foreach_div_.ScalarList` / `_foreach_addcdiv_.ScalarList`: a
    `Scalar[]` reaches the boxed kernel element by element."""
    cpu_lists = _foreach_lists("cpu")
    mojo_lists = _foreach_lists(mojo_gpu)
    apply(*cpu_lists)
    apply(*mojo_lists)
    for expected, actual in zip(cpu_lists[0], mojo_lists[0], strict=True):
        torch.testing.assert_close(actual.cpu(), expected, rtol=2e-6, atol=2e-7)


def test_batched_inplace_foreach_bumps_the_version_counter(mojo_gpu):
    """The dispatcher bumps `Tensor(a!)` arguments but not the members of a
    `Tensor(a!)[]`; the batched kernels do it themselves so autograd notices
    an in-place update of a saved tensor."""
    xs = [torch.ones(8, device=mojo_gpu) for _ in range(3)]
    before = [x._version for x in xs]
    torch._foreach_add_(xs, 1.0)
    torch._foreach_mul_(xs, 2.0)
    assert [x._version - v for x, v in zip(xs, before)] == [2, 2, 2]
    assert xs[0].cpu().tolist() == [4.0] * 8


# ---------------------------------------------------------------------------
# Past the per-launch descriptor cap, and the aliasing rules.
#
# The lists above are 5 tensors, which fits one batch: nothing here crosses
# the cap (64 descriptors per launch) and nothing checks what happens when a
# list member aliases another or the scalar operand.
# ---------------------------------------------------------------------------


def _over_cap_lists(device: str, dtype=torch.float32):
    """66 entries: an empty, 64 small ones (the cap), and one large."""
    cpu = (
        [torch.empty(0, dtype=dtype)]
        + [torch.tensor([3.0 * i, -4.0 * i], dtype=dtype) for i in range(64)]
        + [torch.linspace(-3, 4, 65_537).to(dtype)]
    )
    return cpu, [t.to(device) for t in cpu]


@pytest.mark.parametrize("dtype_arg", [None, torch.float32])
def test_foreach_norm_over_the_descriptor_cap(mojo_gpu: str, dtype_arg):
    """65 inputs including an empty one and two non-finite ones: the batch
    boundary must not reorder or drop a result."""
    cpu = (
        [
            torch.empty(0),
            torch.tensor([float("inf"), 1.0]),
            torch.tensor([float("nan"), 2.0]),
        ]
        + [torch.tensor([3.0 * i, -4.0 * i]) for i in range(1, 62)]
        + [torch.linspace(-3, 4, 65_537)]
    )
    assert len(cpu) == 65
    device = [t.to(mojo_gpu) for t in cpu]
    kwargs = {} if dtype_arg is None else {"dtype": dtype_arg}
    got = torch._foreach_norm(device, 2, **kwargs)
    want = torch._foreach_norm(cpu, 2, **kwargs)
    assert len(got) == len(want)
    for index, (g, w) in enumerate(zip(got, want, strict=True)):
        assert g.dtype == torch.float32 and tuple(g.shape) == ()
        (
            torch.testing.assert_close(
                g.cpu(), w.float(), equal_nan=True, rtol=2e-5, atol=1e-6
            ),
            index,
        )
    pointers = [t.data_ptr() for t in got]
    assert len(set(pointers)) == len(pointers)  # distinct allocations


def test_foreach_norm_strided_inputs(mojo_gpu: str):
    cpu = [
        torch.arange(24, dtype=torch.float32).reshape(4, 6).t(),
        torch.linspace(-2, 3, 35).reshape(5, 7).t(),
    ]
    device = [
        torch.arange(24, dtype=torch.float32).to(mojo_gpu).reshape(4, 6).t(),
        torch.linspace(-2, 3, 35).to(mojo_gpu).reshape(5, 7).t(),
    ]
    got = torch._foreach_norm(device, 2)
    want = torch._foreach_norm(cpu, 2)
    for g, w in zip(got, want, strict=True):
        torch.testing.assert_close(g.cpu(), w, rtol=2e-5, atol=1e-6)


def test_foreach_norm_then_mul_over_the_cap_all_empty(mojo_gpu: str):
    device = [torch.empty(0, device=mojo_gpu) for _ in range(65)]
    versions = [t._version for t in device]
    norms = torch._foreach_norm(device, 2)
    for norm in norms:
        assert tuple(norm.shape) == ()
        assert float(norm.cpu()) == 0.0
    torch._foreach_mul_(device, torch.tensor(2.0, device=mojo_gpu))
    for tensor, before in zip(device, versions, strict=True):
        assert tensor._version == before + 1


@pytest.mark.parametrize("dtype", [torch.float32, torch.float16, torch.bfloat16])
def test_foreach_add_scalar_over_the_cap_is_bit_exact(mojo_gpu: str, dtype):
    """The batched kernel widens to fp32 and narrows, which is what the
    per-tensor `add_.Scalar` does: bit-exactly the same answer."""
    cpu, device = _over_cap_lists(mojo_gpu, dtype)
    pointers = [t.data_ptr() for t in device]
    versions = [t._version for t in device]
    torch._foreach_add_(device, 1.5)
    for tensor, source, pointer, version in zip(
        device, cpu, pointers, versions, strict=True
    ):
        assert tensor.data_ptr() == pointer
        assert tensor._version == version + 1
        expected = (source.float() + 1.5).to(dtype)
        assert torch.equal(tensor.cpu(), expected)


def test_foreach_mul_tensor_over_the_cap_is_bit_exact(mojo_gpu: str):
    cpu, device = _over_cap_lists(mojo_gpu)
    pointers = [t.data_ptr() for t in device]
    versions = [t._version for t in device]
    torch._foreach_mul_(device, torch.tensor(2.5, device=mojo_gpu))
    for tensor, source, pointer, version in zip(
        device, cpu, pointers, versions, strict=True
    ):
        assert tensor.data_ptr() == pointer
        assert tensor._version == version + 1
        assert torch.equal(tensor.cpu(), (source.float() * 2.5).to(torch.float32))


def test_foreach_add_scalar_duplicate_entry_is_applied_twice(mojo_gpu: str):
    """ATen's contract for a list with the same tensor twice: the op runs per
    ENTRY, not per distinct allocation."""
    x = torch.tensor([2.0, -3.0, 5.0], device=mojo_gpu)
    version = x._version
    torch._foreach_add_([x, x], 1.0)
    torch.testing.assert_close(x.cpu(), torch.tensor([4.0, -1.0, 7.0]))
    assert x._version == version + 2


def test_foreach_mul_tensor_overlapping_views_are_sequential(mojo_gpu: str):
    base = torch.tensor([1.0, 2.0, 3.0, 4.0, 5.0], device=mojo_gpu)
    version = base._version
    torch._foreach_mul_([base[:4], base[1:]], torch.tensor(2.0, device=mojo_gpu))
    torch.testing.assert_close(base.cpu(), torch.tensor([2.0, 8.0, 12.0, 16.0, 10.0]))
    assert base._version == version + 2


def test_foreach_mul_tensor_rejects_a_scalar_that_aliases_an_input(mojo_gpu: str):
    """A scalar read from inside a tensor the launch is about to overwrite:
    the result would depend on launch order, so it must decline BEFORE
    writing anything."""
    x = torch.tensor([2.0, 3.0, 4.0], device=mojo_gpu)
    before = x.cpu().clone()
    version = x._version
    with pytest.raises((NotImplementedError, RuntimeError)):
        torch._foreach_mul_([x], x[0])
    torch.testing.assert_close(x.cpu(), before)
    assert x._version == version


def test_foreach_mul_tensor_allows_a_full_self_alias(mojo_gpu: str):
    """The whole 0-d tensor being its own operand is well defined."""
    x = torch.tensor(3.0, device=mojo_gpu)
    version = x._version
    torch._foreach_mul_([x], x)
    assert float(x.cpu()) == 9.0
    assert x._version == version + 1


def test_foreach_mul_tensor_allows_a_scalar_in_a_strided_hole(mojo_gpu: str):
    """`base[1]` is not covered by `base[::2]`, so nothing overwrites it.

    The foreach fast path declines (its self-overlap check compares address
    ranges); the per-tensor `mul_` it falls back to follows torch's overlap
    rule, which calls a non-dense output too hard to judge and runs it.
    """
    base = torch.arange(1.0, 7.0, device=mojo_gpu)
    version = base._version
    torch._foreach_mul_([base[::2]], base[1])
    torch.testing.assert_close(
        base.cpu(), torch.tensor([2.0, 2.0, 6.0, 4.0, 10.0, 6.0])
    )
    assert base._version == version + 1


# --- fused optimizers: _fused_{adam,adamw,sgd,adagrad}_ -----------------------

_FUSED_DTYPES = {
    "f32": torch.float32,
    "f16": torch.float16,
    "bf16": torch.bfloat16,
    "f64": torch.float64,
}
# (shape, transpose): empty, 0-d, 1-d, a tensor longer than one vector step
# with a scalar tail, and a transposed (dense, non-contiguous) one.
_FUSED_SHAPES = (
    ((0,), False),
    ((), False),
    ((7,), False),
    ((17, 65), False),
    ((9, 13), True),
)


def _fused_tolerance(dtype: torch.dtype) -> tuple[float | None, float | None]:
    """(rtol, atol); None, None is torch.testing's per-dtype default: about
    one ulp of a half type."""
    if dtype == torch.float64:
        return 1e-12, 1e-14
    if dtype == torch.float32:
        return 4e-6, 4e-7
    return None, None


def _fused_values(
    shape: tuple[int, ...],
    transpose: bool,
    dtype: torch.dtype,
    *,
    scale: float,
    offset: float,
    positive: bool = False,
) -> torch.Tensor:
    """Deterministic, nonuniform values (no RNG); `transpose` stores the
    tensor transposed, so it is dense but not contiguous."""
    stored = shape[::-1] if transpose else shape
    numel = torch.empty(stored).numel()
    x = torch.arange(numel, dtype=torch.float64).mul(scale).add(offset)
    x = x.sin().mul(0.5).add(offset) if not positive else x.cos().abs().add(0.05)
    x = x.to(dtype).reshape(stored)
    return x.t() if transpose else x


def _fused_lists(
    name: str, n_lists: int, dtype: torch.dtype, state_dtype: torch.dtype | None = None
) -> list[list[torch.Tensor]]:
    """params, grads, then `n_lists - 2` state lists (the squared-gradient
    accumulators -- Adam's exp_avg_sqs and max_exp_avg_sqs, Adagrad's
    state_sums -- kept positive)."""
    state_dtype = state_dtype or dtype
    first_positive = 2 if "adagrad" in name else 3
    lists = []
    for k in range(n_lists):
        dt = dtype if k < 2 else state_dtype
        lists.append(
            [
                _fused_values(
                    shape,
                    t,
                    dt,
                    scale=0.37 + 0.11 * k,
                    offset=0.2 * k - 0.3 + 0.05 * i,
                    positive=k >= first_positive,
                )
                for i, (shape, t) in enumerate(_FUSED_SHAPES)
            ]
        )
    return lists


def _fused_steps(device: str) -> list[torch.Tensor]:
    return [torch.tensor(float(s), device=device) for s in (1, 3, 5, 11, 17)]


def _to(lists: list[list[torch.Tensor]], device: str) -> list[list[torch.Tensor]]:
    # clone() keeps the transposed layout (preserve_format on a dense tensor).
    return [[t.clone().to(device) for t in group] for group in lists]


def _assert_lists_close(actual, expected, dtype: torch.dtype, exact: bool = False):
    rtol, atol = (0.0, 0.0) if exact else _fused_tolerance(dtype)
    for a_group, e_group in zip(actual, expected, strict=True):
        for a, e in zip(a_group, e_group, strict=True):
            assert a.stride() == e.stride() or a.numel() <= 1
            torch.testing.assert_close(a.cpu(), e.cpu(), rtol=rtol, atol=atol)


def _call_fused(name: str, lists, steps, device: str, **kwargs):
    # torch.<op>, not torch.ops.aten.<op>: the overload packet would take a
    # 0-d tensor lr as a float and never reach `.tensor_lr`.
    fn = getattr(torch, name)
    if name in ("_fused_adam_", "_fused_adamw_"):
        params, grads, avgs, sqs = lists[:4]
        maxs = lists[4] if len(lists) > 4 else []
        return fn(params, grads, avgs, sqs, maxs, steps, **kwargs)
    if name == "_fused_sgd_":
        return fn(lists[0], lists[1], lists[2] if len(lists) > 2 else [], **kwargs)
    return fn(lists[0], lists[1], lists[2], steps, **kwargs)


# (op, number of tensor lists, kwargs): every flag the kernels branch on.
_FUSED_CASES = [
    (
        "_fused_adam_",
        4,
        dict(
            lr=0.025,
            beta1=0.8,
            beta2=0.95,
            weight_decay=0.1,
            eps=1e-8,
            amsgrad=False,
            maximize=False,
        ),
    ),
    (
        "_fused_adam_",
        5,
        dict(
            lr=0.01,
            beta1=0.9,
            beta2=0.999,
            weight_decay=0.0,
            eps=1e-6,
            amsgrad=True,
            maximize=True,
        ),
    ),
    (
        "_fused_adamw_",
        4,
        dict(
            lr=0.025,
            beta1=0.8,
            beta2=0.95,
            weight_decay=0.1,
            eps=1e-8,
            amsgrad=False,
            maximize=True,
        ),
    ),
    (
        "_fused_adamw_",
        5,
        dict(
            lr=0.01,
            beta1=0.9,
            beta2=0.99,
            weight_decay=0.02,
            eps=1e-8,
            amsgrad=True,
            maximize=False,
        ),
    ),
    (
        "_fused_sgd_",
        2,
        dict(
            weight_decay=0.1,
            momentum=0.0,
            lr=0.05,
            dampening=0.0,
            nesterov=False,
            maximize=True,
            is_first_step=False,
        ),
    ),
    (
        "_fused_sgd_",
        3,
        dict(
            weight_decay=0.0,
            momentum=0.9,
            lr=0.05,
            dampening=0.1,
            nesterov=False,
            maximize=False,
            is_first_step=False,
        ),
    ),
    (
        "_fused_sgd_",
        3,
        dict(
            weight_decay=0.01,
            momentum=0.8,
            lr=0.05,
            dampening=0.0,
            nesterov=True,
            maximize=False,
            is_first_step=True,
        ),
    ),
    (
        "_fused_adagrad_",
        3,
        dict(lr=0.05, lr_decay=0.01, weight_decay=0.1, eps=1e-10, maximize=False),
    ),
    (
        "_fused_adagrad_",
        3,
        dict(lr=0.05, lr_decay=0.0, weight_decay=0.0, eps=1e-10, maximize=True),
    ),
]
_FUSED_CASE_IDS = [
    "adam",
    "adam-amsgrad-max",
    "adamw-max",
    "adamw-amsgrad",
    "sgd-nomomentum",
    "sgd-momentum",
    "sgd-nesterov-first",
    "adagrad",
    "adagrad-max",
]


@pytest.mark.parametrize("dtype_id", list(_FUSED_DTYPES))
@pytest.mark.parametrize(
    ("name", "n_lists", "kwargs"), _FUSED_CASES, ids=_FUSED_CASE_IDS
)
def test_fused_optimizer_matches_cpu(
    mojo_gpu: str,
    name: str,
    n_lists: int,
    kwargs: dict[str, float | bool],
    dtype_id: str,
):
    dtype = _FUSED_DTYPES[dtype_id]
    if dtype == torch.float64:
        skip_if_metal(mojo_gpu, "Apple GPUs have no float64")
    lists = _fused_lists(name, n_lists, dtype)
    mojo = _to(lists, mojo_gpu)
    # Half types: CUDA computes in float from the stored values and rounds
    # each result once on store, which is the float32 op on the widened
    # values, rounded. (CPU's own reduced-precision fused SGD is not a usable
    # reference: its vectorized loop leaves the params unchanged.)
    reduced = dtype in (torch.float16, torch.bfloat16)
    cpu = [[t.float() if reduced else t.clone() for t in group] for group in lists]
    with ran(f"aten::{name}"):
        assert (
            _call_fused(name, mojo, _fused_steps(mojo_gpu), mojo_gpu, **kwargs) is None
        )
    _call_fused(name, cpu, _fused_steps("cpu"), "cpu", **kwargs)
    _assert_lists_close(mojo, [[t.to(dtype) for t in group] for group in cpu], dtype)


@pytest.mark.parametrize("found_inf", [0.0, 1.0])
@pytest.mark.parametrize(
    ("name", "n_lists", "kwargs"), _FUSED_CASES, ids=_FUSED_CASE_IDS
)
def test_fused_optimizer_grad_scale_found_inf_and_tensor_lr(
    mojo_gpu: str,
    name: str,
    n_lists: int,
    kwargs: dict[str, float | bool],
    found_inf: float,
):
    """GradScaler's fused path: grads are unscaled in place (and written
    back), found_inf == 1 skips every write; lr is a device tensor (the
    `.tensor_lr` overload), then a CPU tensor (read as a float)."""
    lists = _fused_lists(name, n_lists, torch.float32)
    kwargs = dict(kwargs)
    lr = kwargs.pop("lr")
    for lr_device in (mojo_gpu, "cpu"):
        cpu, mojo = _to(lists, "cpu"), _to(lists, mojo_gpu)
        with ran(f"aten::{name}.tensor_lr"):
            _call_fused(
                name,
                mojo,
                _fused_steps(mojo_gpu),
                mojo_gpu,
                lr=torch.tensor(lr, device=lr_device),
                grad_scale=torch.tensor(4.0, device=mojo_gpu),
                found_inf=torch.tensor(found_inf, device=mojo_gpu),
                **kwargs,
            )
        _call_fused(
            name,
            cpu,
            _fused_steps("cpu"),
            "cpu",
            lr=torch.tensor(lr),
            grad_scale=torch.tensor(4.0),
            found_inf=torch.tensor(found_inf),
            **kwargs,
        )
        _assert_lists_close(mojo, cpu, torch.float32)
        if found_inf == 1.0:
            _assert_lists_close(mojo, lists, torch.float32, exact=True)


@pytest.mark.parametrize(
    ("name", "n_lists", "kwargs"), _FUSED_CASES, ids=_FUSED_CASE_IDS
)
def test_fused_optimizer_functional_and_out_variants(
    mojo_gpu: str, name: str, n_lists: int, kwargs: dict[str, float | bool]
):
    """The autogen `_fused_*` (functional) and `.out` overloads compose over
    the in-place kernel."""
    lists = _fused_lists(name, n_lists, torch.float32)
    cpu, mojo = _to(lists, "cpu"), _to(lists, mojo_gpu)
    functional = name.rstrip("_")
    fn = getattr(torch.ops.aten, functional)

    def call(ls, device):
        steps = _fused_steps(device)
        if name in ("_fused_adam_", "_fused_adamw_"):
            maxs = ls[4] if len(ls) > 4 else []
            return fn(ls[0], ls[1], ls[2], ls[3], maxs, steps, **kwargs)
        if name == "_fused_sgd_":
            return fn(ls[0], ls[1], ls[2] if len(ls) > 2 else [], **kwargs)
        return fn(ls[0], ls[1], ls[2], steps, **kwargs)

    expected = call(cpu, "cpu")
    with ran(f"aten::{name}"):
        actual = call(mojo, mojo_gpu)
    for a_group, e_group in zip(actual, expected, strict=True):
        for a, e in zip(a_group, e_group, strict=True):
            rtol, atol = _fused_tolerance(torch.float32)
            torch.testing.assert_close(a.cpu(), e, rtol=rtol, atol=atol)
    # The functional form leaves its inputs alone.
    _assert_lists_close(mojo, lists, torch.float32, exact=True)


@pytest.mark.parametrize("out_mode", ["fresh", "alias_self"])
@pytest.mark.parametrize(
    ("name", "n_lists", "kwargs"), _FUSED_CASES, ids=_FUSED_CASE_IDS
)
def test_fused_optimizer_out_variants(
    mojo_gpu: str,
    name: str,
    n_lists: int,
    kwargs: dict[str, float | bool],
    out_mode: str,
):
    """The autogen `.out` (and `.tensor_lr_out`) overloads: `self` is read
    only, the state lists are updated in place, the new params land in the
    caller's `out` tensors (their storage kept), and `out` may be `self`
    itself."""
    lists = _fused_lists(name, n_lists, torch.float32)
    packet = getattr(torch.ops.aten, name.rstrip("_"))
    kwargs = dict(kwargs)
    lr = kwargs.pop("lr")
    results = {}
    for device in ("cpu", mojo_gpu):
        ls = _to(lists, device)
        if out_mode == "alias_self":
            out = ls[0]
        else:
            out = [torch.full_like(t, 7.0) for t in ls[0]]
        ptrs = [t.data_ptr() for t in out]
        steps = _fused_steps(device)
        for overload, lr_value in (("out", lr), ("tensor_lr_out", torch.tensor(lr))):
            fn = getattr(packet, overload)
            if name in ("_fused_adam_", "_fused_adamw_"):
                maxs = ls[4] if len(ls) > 4 else []
                args = (ls[0], ls[1], ls[2], ls[3], maxs, steps)
            elif name == "_fused_sgd_":
                args = (ls[0], ls[1], ls[2] if len(ls) > 2 else [])
            else:
                args = (ls[0], ls[1], ls[2], steps)
            if device == "cpu":
                fn(*args, lr=lr_value, out=out, **kwargs)
            else:
                with ran(f"aten::{name}", f"aten::{name}.tensor_lr"):
                    fn(*args, lr=lr_value, out=out, **kwargs)
        assert [t.data_ptr() for t in out] == ptrs
        results[device] = (ls, out)
    (cpu_ls, cpu_out), (mojo_ls, mojo_out) = results["cpu"], results[mojo_gpu]
    _assert_lists_close([mojo_out], [cpu_out], torch.float32)
    _assert_lists_close(mojo_ls, cpu_ls, torch.float32)
    if out_mode == "fresh":
        # `self` is not written by the out= form.
        _assert_lists_close([mojo_ls[0]], [lists[0]], torch.float32, exact=True)


def _alias_pairs(n_lists: int) -> list[tuple[int, int]]:
    return [(i, j) for i in range(n_lists) for j in range(i + 1, n_lists)]


_ALIAS_CASES = [
    (name, n_lists, kwargs, pair)
    for (name, n_lists, kwargs), case_id in zip(
        _FUSED_CASES, _FUSED_CASE_IDS, strict=True
    )
    if case_id in ("adam-amsgrad-max", "adamw-amsgrad", "sgd-momentum", "adagrad")
    for pair in _alias_pairs(n_lists)
]


@pytest.mark.skipif(not torch.cuda.is_available(), reason="needs stock CUDA")
@pytest.mark.parametrize(("name", "n_lists", "kwargs", "pair"), _ALIAS_CASES)
def test_fused_optimizer_aliased_lists_match_stock_cuda(
    mojo_gpu: str,
    name: str,
    n_lists: int,
    kwargs: dict[str, float | bool],
    pair: tuple[int, int],
):
    """Every pair of tensor lists passed as the same tensors: the functor
    loads every operand before storing any, and stores in its fixed order,
    so the list CUDA writes last is what survives."""
    lists = _fused_lists(name, n_lists, torch.float32)
    lists[pair[1]] = lists[pair[0]]
    out = {}
    for device in ("cuda", mojo_gpu):
        ls = [[t.clone().to(device) for t in group] for group in lists]
        ls[pair[1]] = ls[pair[0]]
        _call_fused(
            name,
            ls,
            _fused_steps(device),
            device,
            grad_scale=torch.tensor(4.0, device=device),
            **kwargs,
        )
        out[device] = ls
    # Aliasing a gradient with a sum of squares can take a square root of a
    # negative sum: NaN on both sides.
    rtol, atol = _fused_tolerance(torch.float32)
    for a_group, e_group in zip(out[mojo_gpu], out["cuda"], strict=True):
        for a, e in zip(a_group, e_group, strict=True):
            torch.testing.assert_close(
                a.cpu(), e.cpu(), rtol=rtol, atol=atol, equal_nan=True
            )


def test_fused_optimizer_aliased_lists_keep_the_last_store(mojo_gpu: str):
    """The same, CUDA-free, on two hand-computed cases."""
    # AMSGrad with max_exp_avg_sqs == exp_avg_sqs: exp_avg_sq decays to 5,
    # the max (10) is stored after it and survives.
    sq = [torch.full((5,), 10.0, device=mojo_gpu)]
    params = [torch.ones(5, device=mojo_gpu)]
    _call_fused(
        "_fused_adam_",
        [
            params,
            [torch.zeros(5, device=mojo_gpu)],
            [torch.zeros(5, device=mojo_gpu)],
            sq,
            sq,
        ],
        [torch.tensor(1.0, device=mojo_gpu)],
        mojo_gpu,
        lr=0.1,
        beta1=0.9,
        beta2=0.5,
        weight_decay=0.0,
        eps=1e-8,
        amsgrad=True,
        maximize=False,
    )
    assert torch.equal(sq[0].cpu(), torch.full((5,), 10.0))
    # SGD with grads == momentum buffers and grad_scale: the momentum update
    # reads the buffer as loaded (2), not the unscaled grad stored first
    # (0.5); the buffer (0.9 * 2 + 0.5) is stored last.
    shared = [torch.full((5,), 2.0, device=mojo_gpu)]
    params = [torch.ones(5, device=mojo_gpu)]
    torch._fused_sgd_(
        params,
        shared,
        shared,
        weight_decay=0.0,
        momentum=0.9,
        lr=0.1,
        dampening=0.0,
        nesterov=False,
        maximize=False,
        is_first_step=False,
        grad_scale=torch.tensor(4.0, device=mojo_gpu),
    )
    torch.testing.assert_close(shared[0].cpu(), torch.full((5,), 2.3))
    torch.testing.assert_close(params[0].cpu(), torch.full((5,), 1 - 0.1 * 2.3))


@pytest.mark.parametrize("name", ["_fused_adam_", "_fused_adamw_", "_fused_adagrad_"])
def test_fused_optimizer_requires_one_step_per_tensor(mojo_gpu: str, name: str):
    """An empty (or short) state_steps list is refused before any launch:
    the kernel reads one step per tensor."""
    n_lists = 3 if name == "_fused_adagrad_" else 4
    lists = _to(_fused_lists(name, n_lists, torch.float32), mojo_gpu)
    kwargs = (
        dict(lr=0.1, lr_decay=0.0, weight_decay=0.0, eps=1e-10, maximize=False)
        if name == "_fused_adagrad_"
        else dict(
            lr=0.1,
            beta1=0.9,
            beta2=0.99,
            weight_decay=0.0,
            eps=1e-8,
            amsgrad=False,
            maximize=False,
        )
    )
    for steps in ([], _fused_steps(mojo_gpu)[:2]):
        with pytest.raises(RuntimeError, match="same number of tensors"):
            _call_fused(name, lists, steps, mojo_gpu, **kwargs)


@pytest.mark.parametrize(
    "lr",
    [
        torch.tensor(0.05, dtype=torch.float16),
        torch.tensor(0.05, dtype=torch.bfloat16),
        torch.tensor(1, dtype=torch.int64),
        torch.tensor(True),
        torch.tensor(0.05, dtype=torch.float64),
    ],
    ids=["f16", "bf16", "int64", "bool", "f64"],
)
def test_fused_optimizer_cpu_lr_of_any_dtype(mojo_gpu: str, lr: torch.Tensor):
    """A CPU lr tensor is `lr.item<double>()`, whatever its dtype."""
    lists = _fused_lists("_fused_sgd_", 2, torch.float32)
    cpu, mojo = _to(lists, "cpu"), _to(lists, mojo_gpu)
    kwargs = dict(
        weight_decay=0.0,
        momentum=0.0,
        dampening=0.0,
        nesterov=False,
        maximize=False,
        is_first_step=False,
    )
    _call_fused("_fused_sgd_", mojo, [], mojo_gpu, lr=lr, **kwargs)
    _call_fused("_fused_sgd_", cpu, [], "cpu", lr=lr.item(), **kwargs)
    _assert_lists_close(mojo, cpu, torch.float32)


def _bf16_round(x: torch.Tensor) -> torch.Tensor:
    return x.to(torch.bfloat16).to(torch.float32)


def _fma32(a, b, c) -> torch.Tensor:
    """float32 fma: the double product of two floats is exact."""
    return (
        torch.as_tensor(a, dtype=torch.float64) * torch.as_tensor(b).double()
        + torch.as_tensor(c).double()
    ).float()


@pytest.mark.parametrize("amsgrad", [False, True])
@pytest.mark.parametrize("name", ["_fused_adam_", "_fused_adamw_"])
def test_fused_adam_mixed_precision_matches_cuda_math(
    mojo_gpu: str, name: str, amsgrad: bool
):
    """float32 params and grads with bfloat16 states (CUDA's
    FusedAdamMathFunctorMP; CPU has no such mode): bias corrections in
    double, the update in float32, states rounded to bfloat16 on store."""
    lists = _fused_lists(name, 5 if amsgrad else 4, torch.float32, torch.bfloat16)
    mojo = _to(lists, mojo_gpu)
    steps = [3.0, 1.0, 5.0, 11.0, 17.0]
    lr, beta1, beta2, wd, eps = 0.02, 0.85, 0.97, 0.1, 1e-8
    with ran(f"aten::{name}"):
        _call_fused(
            name,
            mojo,
            [torch.tensor(s, device=mojo_gpu) for s in steps],
            mojo_gpu,
            lr=lr,
            beta1=beta1,
            beta2=beta2,
            weight_decay=wd,
            eps=eps,
            amsgrad=amsgrad,
            maximize=False,
        )
    f = torch.float32
    for i, step in enumerate(steps):
        p, g = lists[0][i].float(), lists[1][i].float()
        m, v = lists[2][i].float(), lists[3][i].float()
        bc1 = torch.tensor(1 - beta1**step, dtype=f)
        bc2s = torch.tensor((1 - beta2**step) ** 0.5, dtype=f)
        b1, b2 = torch.tensor(beta1, dtype=f), torch.tensor(beta2, dtype=f)
        if name == "_fused_adam_":
            g = g + p * torch.tensor(wd, dtype=f)
        else:
            p = p - torch.tensor(lr, dtype=f) * torch.tensor(wd, dtype=f) * p
        m = _fma32(b1, m, _fma32(-b1, g, g))
        v = _fma32(b2, v, _fma32(-b2, g * g, g * g))
        denom_src = v
        if amsgrad:
            mx = torch.maximum(lists[4][i].float(), v)
            denom_src = mx
            torch.testing.assert_close(mojo[4][i].cpu().float(), _bf16_round(mx))
        denom = denom_src.sqrt() / bc2s + torch.tensor(eps, dtype=f)
        p = p - (torch.tensor(lr, dtype=f) / bc1) * m / denom
        torch.testing.assert_close(mojo[0][i].cpu(), p, rtol=4e-6, atol=4e-7)
        assert mojo[2][i].dtype == torch.bfloat16
        torch.testing.assert_close(mojo[2][i].cpu().float(), _bf16_round(m))
        torch.testing.assert_close(mojo[3][i].cpu().float(), _bf16_round(v))


def _torch_at_least(major: int, minor: int) -> bool:
    version = torch.__version__.split("+")[0].split(".")
    return (int(version[0]), int(version[1])) >= (major, minor)


@pytest.mark.skipif(not torch.cuda.is_available(), reason="needs stock CUDA")
@pytest.mark.parametrize("dtype_id", ["f32", "f16", "bf16", "f64", "f32-bf16"])
@pytest.mark.parametrize(
    ("name", "n_lists", "kwargs"), _FUSED_CASES, ids=_FUSED_CASE_IDS
)
def test_fused_optimizer_matches_stock_cuda(
    mojo_gpu: str,
    name: str,
    n_lists: int,
    kwargs: dict[str, float | bool],
    dtype_id: str,
):
    """Against stock torch's own CUDA kernels. Not bit for bit: nvcc contracts
    `grad + param * weight_decay` and SGD's momentum update into an fma in
    some ILP lanes and code paths and not in others (read off the sm_90 SASS
    of FusedAdamMathFunctor<float, 4, ORIGINAL>: lane 0 of the unscaled path
    is a mul then an add, lanes 1-3 an fma), so a result can sit an ulp or so
    away where that choice differs."""
    if dtype_id == "f32-bf16":
        if name not in ("_fused_adam_", "_fused_adamw_"):
            pytest.skip("mixed precision is an Adam/AdamW mode")
        if not _torch_at_least(2, 14):
            pytest.skip("stock torch has mixed-precision fused Adam from 2.14")
        lists = _fused_lists(name, n_lists, torch.float32, torch.bfloat16)
    else:
        lists = _fused_lists(name, n_lists, _FUSED_DTYPES[dtype_id])
    for scaled in (False, True):
        extra = {}
        if scaled:
            extra = dict(grad_scale=4.0, found_inf=0.0)
        cuda, mojo = _to(lists, "cuda"), _to(lists, mojo_gpu)
        for device, ls in (("cuda", cuda), (mojo_gpu, mojo)):
            _call_fused(
                name,
                ls,
                _fused_steps(device),
                device,
                **kwargs,
                **{k: torch.tensor(v, device=device) for k, v in extra.items()},
            )
        for a_group, e_group in zip(mojo, cuda, strict=True):
            for a, e in zip(a_group, e_group, strict=True):
                if e.numel() == 0:
                    continue
                eps = torch.finfo(a.dtype).eps
                scale = e.abs().max().item()
                torch.testing.assert_close(
                    a.cpu(), e.cpu(), rtol=4 * eps, atol=4 * eps * scale
                )


def test_fused_optimizer_errors_match_cuda(mojo_gpu: str):
    params = [torch.ones(3, device=mojo_gpu), torch.ones(4, device=mojo_gpu)]
    grads = [torch.ones_like(p) for p in params]
    steps = [torch.tensor(1.0, device=mojo_gpu) for _ in params]
    adam = dict(
        lr=0.1,
        beta1=0.9,
        beta2=0.99,
        weight_decay=0.0,
        eps=1e-8,
        amsgrad=False,
        maximize=False,
    )

    # A dtype mismatch across lists, and a non-dense tensor.
    with pytest.raises(RuntimeError, match="must have same dtype, device, and layout"):
        _call_fused(
            "_fused_adam_",
            [params, [g.half() for g in grads], params, params],
            steps,
            mojo_gpu,
            **adam,
        )
    # float32 params with other states is the mixed-precision mode, which
    # takes bfloat16 states only.
    with pytest.raises(
        RuntimeError,
        match="Mixed-precision fused Adam requires bfloat16 optimizer states, got Half",
    ):
        halves = [p.half() for p in params]
        _call_fused(
            "_fused_adam_", [params, grads, halves, halves], steps, mojo_gpu, **adam
        )
    strided = [torch.ones(6, device=mojo_gpu)[::2], torch.ones(4, device=mojo_gpu)]
    with pytest.raises(RuntimeError, match="must have same dtype, device, and layout"):
        _call_fused(
            "_fused_adamw_", [strided, grads, params, params], steps, mojo_gpu, **adam
        )
    # Mixed precision is float32 params with bfloat16 states only.
    halves = [p.half() for p in params]
    with pytest.raises(
        RuntimeError,
        match="Mixed-precision fused Adam requires float32 params, got Half",
    ):
        _call_fused(
            "_fused_adam_",
            [halves, [g.half() for g in grads], params, params],
            steps,
            mojo_gpu,
            **adam,
        )
    # Integer params: the dispatch has no such kernel.
    ints = [p.int() for p in params]
    with pytest.raises(
        RuntimeError, match="\"fused_adagrad_kernel_cuda\" not implemented for 'Int'"
    ):
        torch._fused_adagrad_(
            ints,
            [p.int() for p in params],
            [p.int() for p in params],
            steps,
            lr=0.1,
            lr_decay=0.0,
            weight_decay=0.0,
            eps=1e-10,
            maximize=False,
        )
    with pytest.raises(RuntimeError, match="found_inf must be on the same GPU"):
        torch._fused_adagrad_(
            params,
            grads,
            [torch.zeros_like(p) for p in params],
            steps,
            lr=torch.tensor(0.1, device=mojo_gpu),
            lr_decay=0.0,
            weight_decay=0.0,
            eps=1e-10,
            maximize=False,
            found_inf=torch.tensor(0.0),
        )
    with pytest.raises(RuntimeError, match="grad_scale must be on the same GPU"):
        torch._fused_sgd_(
            params,
            grads,
            [],
            weight_decay=0.0,
            momentum=0.0,
            lr=0.1,
            dampening=0.0,
            nesterov=False,
            maximize=False,
            is_first_step=False,
            grad_scale=torch.tensor(1.0),
        )
    with pytest.raises(RuntimeError, match=r"momentum > 0"):
        torch._fused_sgd_(
            params,
            grads,
            [torch.zeros_like(p) for p in params],
            weight_decay=0.0,
            momentum=0.0,
            lr=0.1,
            dampening=0.0,
            nesterov=False,
            maximize=False,
            is_first_step=False,
        )
    with pytest.raises(RuntimeError, match=r"momentum == 0"):
        torch._fused_sgd_(
            params,
            grads,
            [],
            weight_decay=0.0,
            momentum=0.9,
            lr=0.1,
            dampening=0.0,
            nesterov=False,
            maximize=False,
            is_first_step=False,
        )


def _optimizer_run(
    device: str, make, dtype: torch.dtype, steps: int, scaler: bool = False
) -> tuple[list[torch.Tensor], dict[int, dict[str, torch.Tensor]]]:
    initial = [
        torch.linspace(-1.0, 1.0, 31, dtype=torch.float64),
        torch.linspace(0.5, -0.75, 35, dtype=torch.float64).reshape(5, 7),
        torch.linspace(0.1, 0.3, 12, dtype=torch.float64).reshape(4, 3).t(),
    ]
    # On the CPU, reduced-precision runs are emulated the way CUDA computes
    # them: float32 math on the stored values, every result rounded to `dtype`
    # after each step (CPU's own reduced-precision fused SGD is broken).
    emulate = device == "cpu" and dtype in (torch.float16, torch.bfloat16)
    work = torch.float32 if emulate else dtype
    params = [torch.nn.Parameter(v.to(dtype).to(work).to(device)) for v in initial]
    optimizer = make(params)
    grad_scaler = None
    if scaler:
        grad_scaler = torch.amp.GradScaler(
            torch.device(device).type, init_scale=2.0**8, growth_interval=2
        )
        grad_scaler.scale(torch.ones((), device=device))  # its lazy init
    for step in range(steps):
        for index, p in enumerate(params):
            # Bounded away from -weight_decay * param: a gradient that the
            # decay cancels to ~0 makes Adagrad's g / sqrt(g * g) ill-posed.
            grad = torch.linspace(
                0.15 + step * 0.03, 0.45 - index * 0.02, p.numel(), dtype=torch.float64
            )
            if grad_scaler is not None:
                grad = grad * grad_scaler.get_scale()
                if step == 2:
                    grad[0] = float("inf")  # a skipped step
            grad = grad.reshape(p.shape).to(dtype).to(work)
            p.grad = torch.empty_like(p).copy_(grad)  # the param's layout
        if grad_scaler is not None:
            grad_scaler.step(optimizer)
            grad_scaler.update()
        else:
            optimizer.step()
        if emulate:
            with torch.no_grad():
                for p in params:
                    p.copy_(p.to(dtype))
                    for key, value in optimizer.state[p].items():
                        if key != "step" and isinstance(value, torch.Tensor):
                            value.copy_(value.to(dtype))
    state = {
        i: {
            k: v.to(dtype) if emulate and k != "step" else v
            for k, v in optimizer.state[p].items()
        }
        for i, p in enumerate(params)
    }
    return [p.detach().cpu().to(dtype) for p in params], state


_OPTIMIZERS = {
    "adam-amsgrad-wd-max": (
        lambda ps: torch.optim.Adam(
            ps,
            lr=0.02,
            betas=(0.8, 0.95),
            weight_decay=0.1,
            amsgrad=True,
            maximize=True,
            fused=True,
        ),
        "aten::_fused_adam_",
    ),
    "adamw": (
        lambda ps: torch.optim.AdamW(
            ps, lr=0.02, betas=(0.8, 0.95), weight_decay=0.1, fused=True
        ),
        "aten::_fused_adamw_",
    ),
    "sgd-nesterov": (
        lambda ps: torch.optim.SGD(
            ps, lr=0.05, momentum=0.9, nesterov=True, weight_decay=0.01, fused=True
        ),
        "aten::_fused_sgd_",
    ),
    "sgd-dampening-max": (
        lambda ps: torch.optim.SGD(
            ps, lr=0.05, momentum=0.5, dampening=0.2, maximize=True, fused=True
        ),
        "aten::_fused_sgd_",
    ),
    "sgd-plain": (
        lambda ps: torch.optim.SGD(ps, lr=0.05, fused=True),
        "aten::_fused_sgd_",
    ),
    "adagrad": (
        lambda ps: torch.optim.Adagrad(
            ps, lr=0.05, lr_decay=0.01, weight_decay=0.1, fused=True
        ),
        "aten::_fused_adagrad_",
    ),
}


@pytest.mark.parametrize("dtype_id", ["f32", "f16", "bf16"])
@pytest.mark.parametrize("opt_id", list(_OPTIMIZERS))
def test_fused_torch_optim_matches_cpu(mojo_gpu: str, opt_id: str, dtype_id: str):
    """`torch.optim.*(fused=True)` end to end over several steps, against the
    same optimizer's fused CPU kernels."""
    make, op_name = _OPTIMIZERS[opt_id]
    dtype = _FUSED_DTYPES[dtype_id]
    with ran(op_name):
        params, state = _optimizer_run(mojo_gpu, make, dtype, steps=4)
    cpu_params, cpu_state = _optimizer_run("cpu", make, dtype, steps=4)
    rtol, atol = _fused_tolerance(dtype)
    if dtype != torch.float32:
        rtol, atol = 2 * torch.finfo(dtype).eps, 1e-5
    for actual, expected in zip(params, cpu_params, strict=True):
        assert actual.stride() == expected.stride()
        torch.testing.assert_close(actual, expected, rtol=rtol, atol=atol)
    for i, s in state.items():
        for key, value in s.items():
            assert value.device == torch.device(mojo_gpu)
            torch.testing.assert_close(
                value.cpu(), cpu_state[i][key], rtol=rtol, atol=atol
            )


@pytest.mark.parametrize("opt_id", ["adam-amsgrad-wd-max", "sgd-nesterov", "adagrad"])
def test_fused_torch_optim_with_grad_scaler_matches_cpu(mojo_gpu: str, opt_id: str):
    """GradScaler hands fused optimizers grad_scale/found_inf (no host sync):
    the unscale happens in the kernel, an inf step is skipped."""
    skip_if_metal(mojo_gpu, "GradScaler needs float64")
    make, op_name = _OPTIMIZERS[opt_id]
    with ran(op_name):
        params, _ = _optimizer_run(mojo_gpu, make, torch.float32, steps=5, scaler=True)
    cpu_params, _ = _optimizer_run("cpu", make, torch.float32, steps=5, scaler=True)
    rtol, atol = _fused_tolerance(torch.float32)
    for actual, expected in zip(params, cpu_params, strict=True):
        torch.testing.assert_close(actual, expected, rtol=rtol, atol=atol)


@pytest.mark.skipif(not torch.cuda.is_available(), reason="needs stock CUDA")
@pytest.mark.parametrize(
    ("name", "n_lists", "kwargs"), _FUSED_CASES, ids=_FUSED_CASE_IDS
)
def test_fused_optimizer_nonfinite_matches_stock_cuda(
    mojo_gpu: str, name: str, n_lists: int, kwargs: dict[str, float | bool]
):
    """NaN / inf in every operand propagate as on CUDA (amsgrad's
    `std::max` keeps a NaN running max, 0 * inf is NaN, ...)."""
    nan, inf = float("nan"), float("inf")
    rows = [
        [1.0, 1.0, 1.0, 1.0, 1.0, nan, 0.0, 1.0],
        [nan, inf, -inf, 0.0, 1.0, 1.0, 0.0, 1.0],
        [0.0, 0.0, 0.0, nan, 0.0, 0.0, inf, 0.0],
        [0.0, 0.0, 0.0, 0.0, nan, 0.0, 0.0, 0.1],
        [nan, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, nan],
    ]
    out = {}
    for device in ("cuda", mojo_gpu):
        lists = [[torch.tensor(row, device=device)] for row in rows[:n_lists]]
        steps = [torch.tensor(2.0, device=device)]
        _call_fused(name, lists, steps, device, **kwargs)
        out[device] = lists
    for a, e in zip(out[mojo_gpu], out["cuda"], strict=True):
        torch.testing.assert_close(a[0].cpu(), e[0].cpu(), equal_nan=True)


# ---------------------------------------------------------------------------
# The generic multi-tensor apply (tmb/kernels/foreach): every elementwise
# `_foreach_*` overload of FOREACH_OVERLOADS in tmb/ops/foreach.mojo, against
# CPU torch, and its fast-route / fallback split. One block per family, in
# the order they joined the table.
# ---------------------------------------------------------------------------


# Ragged sizes: empty, single, and around the chunk grain (512), the minimum
# chunk (1024) and the maximum chunk (65_536) of `foreach_chunk_elements`.
_MTA_SIZES = [7, 0, 1, 511, 1025, 2049, 65_539]
_MTA_FLOATS = {
    "f32": torch.float32,
    "f16": torch.float16,
    "bf16": torch.bfloat16,
    "f64": torch.float64,
}


def _mta_tol(dtype: torch.dtype) -> dict[str, float]:
    return {
        torch.float32: {"rtol": 1e-5, "atol": 1e-6},
        torch.float16: {"rtol": 1e-3, "atol": 1e-3},
        torch.bfloat16: {"rtol": 1.6e-2, "atol": 1e-2},
        torch.float64: {"rtol": 1e-12, "atol": 1e-14},
    }.get(dtype, {"rtol": 0, "atol": 0})


def _mta_lists(
    device: str,
    dtype: torch.dtype,
    lo: float,
    hi: float,
    seed: int,
    sizes: list[int] = _MTA_SIZES,
    offset: int = 1,
) -> tuple[list[torch.Tensor], list[torch.Tensor]]:
    """(cpu, device) lists of the same values. The device tensors are views
    `offset` elements into their allocation, so the kernel's alignment peel
    runs; the CPU ones are plain copies."""
    g = torch.Generator().manual_seed(seed)
    cpu, dev = [], []
    for n in sizes:
        if dtype.is_floating_point:
            base = torch.empty(n + offset, dtype=torch.float64).uniform_(
                lo, hi, generator=g
            )
        else:
            base = torch.randint(int(lo), int(hi), (n + offset,), generator=g)
        base = base.to(dtype)
        cpu.append(base[offset:].clone())
        dev.append(base.to(device)[offset:])
    return cpu, dev


@contextlib.contextmanager
def _one_launch(name: str):
    """The call ran as `aten::<name>` alone: no per-tensor op was dispatched,
    so the multi-tensor apply took the list."""
    native.op_counting(True)
    native.op_counts_reset()
    try:
        yield
        counts = native.op_counts()
    finally:
        native.op_counting(False)
    assert set(counts) == {"aten::" + name}, counts


def _assert_lists(
    actual: Sequence[torch.Tensor],
    expected: Sequence[torch.Tensor],
    rtol: float | None = None,
    atol: float | None = None,
):
    assert len(actual) == len(expected)
    for a, e in zip(actual, expected, strict=True):
        assert a.dtype == e.dtype and a.shape == e.shape
        torch.testing.assert_close(a.cpu(), e, equal_nan=True, rtol=rtol, atol=atol)


# --- core: the overloads the multi-tensor apply started with -------------------


_MTA_CORE = {
    "_foreach_add_.Scalar": (
        lambda a, b, c, s: torch._foreach_add_(a, 0.75),
        lambda a, b, c: [t + 0.75 for t in a],
    ),
    "_foreach_mul_.Scalar": (
        lambda a, b, c, s: torch._foreach_mul_(a, 0.75),
        lambda a, b, c: [t * 0.75 for t in a],
    ),
    "_foreach_mul_.Tensor": (
        lambda a, b, c, s: torch._foreach_mul_(a, s),
        lambda a, b, c: [t * torch.tensor(1.25, dtype=t.dtype) for t in a],
    ),
    "_foreach_addcmul_.Scalar": (
        lambda a, b, c, s: torch._foreach_addcmul_(a, b, c, value=-0.75),
        lambda a, b, c: [
            torch.addcmul(x, y, z, value=-0.75) for x, y, z in zip(a, b, c)
        ],
    ),
    "_foreach_lerp_.Scalar": (
        lambda a, b, c, s: torch._foreach_lerp_(a, b, 0.3),
        lambda a, b, c: [torch.lerp(x, y, 0.3) for x, y in zip(a, b)],
    ),
    "_foreach_sqrt": (
        lambda a, b, c, s: torch._foreach_sqrt(c),
        lambda a, b, c: [t.sqrt() for t in c],
    ),
}


@pytest.mark.parametrize("dtype_id", list(_MTA_FLOATS))
@pytest.mark.parametrize("name", list(_MTA_CORE))
def test_mta_core_overloads_match_cpu(mojo_gpu: str, name: str, dtype_id: str):
    """The overloads the multi-tensor apply started with, in every floating
    dtype, over ragged and misaligned lists."""
    dtype = _MTA_FLOATS[dtype_id]
    if dtype == torch.float64:
        skip_if_metal(mojo_gpu, "no float64 on Apple GPUs")
    call, reference = _MTA_CORE[name]
    cpu, dev = _mta_lists(mojo_gpu, dtype, -3, 3, seed=61)
    cpu1, dev1 = _mta_lists(mojo_gpu, dtype, -2, 2, seed=62)
    cpu2, dev2 = _mta_lists(mojo_gpu, dtype, 0.5, 2, seed=63)
    scalar = torch.tensor(1.25, dtype=dtype, device=mojo_gpu)
    expected = reference(cpu, cpu1, cpu2)
    with _one_launch(name):
        result = call(dev, dev1, dev2, scalar)
    _assert_lists(dev if result is None else result, expected, **_mta_tol(dtype))


def test_mta_core_long_lists_span_several_launches(mojo_gpu: str):
    """130 tensors: three descriptor batches (FE_DESC_CAP = 48) of one call."""
    sizes = [(i * 37) % 700 for i in range(130)]
    cpu, dev = _mta_lists(mojo_gpu, torch.float32, -2, 2, seed=41, sizes=sizes)
    cpu_o, dev_o = _mta_lists(mojo_gpu, torch.float32, -2, 2, seed=42, sizes=sizes)
    with _one_launch("_foreach_add_.Scalar"):
        torch._foreach_add_(dev, -0.5)
    with _one_launch("_foreach_addcmul_.Scalar"):
        torch._foreach_addcmul_(dev, dev_o, dev_o, value=0.25)
    expected = [torch.addcmul(a - 0.5, b, b, value=0.25) for a, b in zip(cpu, cpu_o)]
    _assert_lists(dev, expected, **_mta_tol(torch.float32))


# (list dtype, CPU `other`): float/int lists against float/int 0-d and
# one-element 1-d CPU tensors.
_CPU_SCALARS = [
    (torch.float32, torch.tensor(3.0)),
    (torch.float32, torch.tensor(3)),
    (torch.int64, torch.tensor(3)),
    (torch.int64, torch.tensor(3.0)),
    (torch.bfloat16, torch.tensor([0.25], dtype=torch.float64)),
]
_CPU_SCALAR_IDS = ["f32-f32", "f32-i64", "i64-i64", "i64-f32", "bf16-f64-1d"]
_CPU_SCALAR_OPS = ["mul", "div", "add"]
_CPU_SCALAR_INPLACE = [False, True]


@pytest.mark.parametrize(
    "inplace", _CPU_SCALAR_INPLACE, ids=lambda b: "inplace" if b else "out"
)
@pytest.mark.parametrize(("dtype", "other"), _CPU_SCALARS, ids=_CPU_SCALAR_IDS)
@pytest.mark.parametrize("op", _CPU_SCALAR_OPS)
def test_mta_tensor_overload_with_a_cpu_scalar(
    mojo_gpu: str, op: str, dtype: torch.dtype, other: torch.Tensor, inplace: bool
):
    """The `.Tensor` overloads with a CPU `other` (called by schema: Python
    would pick `.Scalar` for a 0-d tensor). mul / div take its item() and
    run the Scalar overload of the same op, functional or in place
    (ForeachBinaryOpScalarTensor.cu); add's, of the list dtype, is refused
    on device, of another dtype the per-tensor op. Errors match the CPU
    per-tensor op (an in-place promotion) or CUDA's device check."""
    cpu = [torch.full((5,), 2, dtype=dtype), (torch.arange(1030) % 9).to(dtype)]
    dev = [t.to(mojo_gpu) for t in cpu]
    schema = getattr(torch.ops.aten, f"_foreach_{op}_" if inplace else f"_foreach_{op}")
    if op == "add" and other.dim() != 0:
        with pytest.raises(RuntimeError, match="scalar tensor expected to be 0 dim"):
            schema.Tensor(dev, other)
        return
    if op == "add" and other.dtype == dtype:
        word = "is expected" if inplace else "expected"
        with pytest.raises(RuntimeError, match=f"scalar tensor {word} to be on mojo:0"):
            schema.Tensor(dev, other)
        return
    ref = getattr(torch, op)
    value = other.item() if op != "add" else other
    try:
        expected = (
            [ref(t, value) for t in cpu]
            if not inplace
            else [getattr(t.clone(), f"{op}_")(value) for t in cpu]
        )
    except RuntimeError as error:
        assert "can't be cast" in str(error)
        with pytest.raises(RuntimeError, match="(?i)can't be cast"):
            schema.Tensor(dev, other)
        return
    result = schema.Tensor(dev, other)
    _assert_lists(dev if inplace else result, expected)


def test_mta_core_argument_errors(mojo_gpu: str):
    xs = [torch.ones(3, device=mojo_gpu), torch.ones(4, device=mojo_gpu)]
    with pytest.raises(RuntimeError, match="at least one tensor"):
        torch._foreach_sqrt([])
    with pytest.raises(RuntimeError, match="same number of tensors, got 2 and 1"):
        torch._foreach_addcmul_(xs, xs, xs[:1])
    with pytest.raises(RuntimeError, match="without overflow"):
        torch._foreach_add_(xs, 1e300)


def test_mta_empty_tensors_only(mojo_gpu: str):
    dev = [torch.empty(0, device=mojo_gpu), torch.empty(0, 3, device=mojo_gpu)]
    with _one_launch("_foreach_sqrt"):
        out = torch._foreach_sqrt(dev)
    assert [tuple(t.shape) for t in out] == [(0,), (0, 3)]
    versions = [t._version for t in dev]
    torch._foreach_add_(dev, 1.0)
    assert [t._version - v for t, v in zip(dev, versions)] == [1, 1]


def _stock_cuda_parity(dtype: torch.dtype, mojo_gpu: str, calls):
    """Run `calls(a, b, c, w)` on stock CUDA and on the mojo device over the
    same lists, and require bit-identical results."""
    results = {}
    for device in ("cuda", mojo_gpu):
        g = torch.Generator().manual_seed(7)

        def mk(lo: float, hi: float) -> list[torch.Tensor]:
            return [
                torch.empty(n, dtype=torch.float64)
                .uniform_(lo, hi, generator=g)
                .to(dtype)
                .to(device)
                for n in (1, 1025, 4099)
            ]

        results[device] = calls(mk(-3, 3), mk(-3, 3), mk(0.5, 3), mk(0, 1))
    for ours, stock in zip(results[mojo_gpu], results["cuda"], strict=True):
        torch.testing.assert_close(ours.cpu(), stock.cpu(), rtol=0, atol=0)


def _core_calls(a, b, c, w) -> list[torch.Tensor]:
    torch._foreach_add_(a, 0.3)
    torch._foreach_mul_(b, 0.7)
    torch._foreach_addcmul_(a, b, c, value=0.3)
    torch._foreach_lerp_(b, a, 0.7)
    return [*a, *b, *torch._foreach_sqrt(c)]


@pytest.mark.skipif(not torch.cuda.is_available(), reason="needs stock CUDA")
@pytest.mark.parametrize("dtype", [torch.float32, torch.float16, torch.bfloat16])
def test_mta_core_is_bit_exact_with_stock_cuda(mojo_gpu: str, dtype: torch.dtype):
    """The element math rounds like CUDA's foreach functors: opmath, one fma
    where nvcc contracts (addcmul's explicit one included), one rounding on
    the store."""
    _stock_cuda_parity(dtype, mojo_gpu, _core_calls)


@pytest.mark.parametrize(
    "make",
    [
        lambda p: torch.optim.Adam(p, lr=1e-2, foreach=True),
        lambda p: torch.optim.AdamW(
            p, lr=1e-2, weight_decay=0.1, amsgrad=True, foreach=True
        ),
        lambda p: torch.optim.SGD(
            p, lr=1e-2, momentum=0.9, nesterov=True, foreach=True
        ),
        lambda p: torch.optim.Adagrad(p, lr=1e-2, foreach=True),
        lambda p: torch.optim.RMSprop(
            p, lr=1e-2, momentum=0.5, centered=True, foreach=True
        ),
        lambda p: torch.optim.NAdam(p, lr=1e-2, foreach=True),
        lambda p: torch.optim.RAdam(p, lr=1e-2, foreach=True),
        lambda p: torch.optim.Adamax(p, lr=1e-2, foreach=True),
    ],
    ids=[
        "adam",
        "adamw-amsgrad",
        "sgd-nesterov",
        "adagrad",
        "rmsprop",
        "nadam",
        "radam",
        "adamax",
    ],
)
def test_mta_torch_optim_foreach_matches_cpu(mojo_gpu: str, make):
    """torch.optim's default foreach implementation, a few steps, against the
    same optimizer on CPU."""
    results = {}
    for device in ("cpu", mojo_gpu):
        torch.manual_seed(0)
        params = [
            torch.nn.Parameter(torch.randn(shape).to(device))
            for shape in ((17, 33), (5,), (1025,), (3, 3, 3))
        ]
        opt = make(params)
        for step in range(4):
            for i, p in enumerate(params):
                p.grad = (
                    torch.sin(
                        torch.arange(p.numel(), dtype=torch.float32) * (i + 1) + step
                    )
                    .reshape(p.shape)
                    .to(device)
                )
            opt.step()
        results[device] = [p.detach().cpu() for p in params]
    for a, e in zip(results[mojo_gpu], results["cpu"], strict=True):
        torch.testing.assert_close(a, e, rtol=2e-5, atol=2e-6)


# --- unary ---------------------------------------------------------------------


_MTA_UNARY = {
    "abs": (-3, 3),
    "acos": (-0.99, 0.99),
    "asin": (-0.99, 0.99),
    "atan": (-3, 3),
    "ceil": (-3, 3),
    "cos": (-3, 3),
    "cosh": (-3, 3),
    "erf": (-2, 2),
    "erfc": (-2, 2),
    "exp": (-3, 3),
    "expm1": (-1, 1),
    "floor": (-3, 3),
    "frac": (-3, 3),
    "lgamma": (0.1, 5),
    "log": (0.1, 5),
    "log10": (0.1, 5),
    "log1p": (-0.5, 3),
    "log2": (0.1, 5),
    "neg": (-3, 3),
    "reciprocal": (0.5, 3),
    "round": (-3, 3),
    "rsqrt": (0.1, 5),
    "sigmoid": (-3, 3),
    "sign": (-3, 3),
    "sin": (-3, 3),
    "sinh": (-2, 2),
    "sqrt": (0, 5),
    "tan": (-1, 1),
    "tanh": (-3, 3),
    "trunc": (-3, 3),
}
# The unary kinds whose float64 math lowers on the GPU (the rest decline
# float64 per tensor too, so their foreach falls back and declines).
_MTA_UNARY_F64 = {
    "abs", "ceil", "floor", "frac", "neg", "reciprocal", "round",
    "sign", "sqrt", "trunc",
}  # fmt: skip
_MTA_UNARY_CASES = [
    (name, dtype_id)
    for name in _MTA_UNARY
    for dtype_id in _MTA_FLOATS
    if dtype_id != "f64" or name in _MTA_UNARY_F64
]


@pytest.mark.parametrize("inplace", [False, True], ids=["out", "inplace"])
@pytest.mark.parametrize(("name", "dtype_id"), _MTA_UNARY_CASES)
def test_mta_unary_matches_cpu(mojo_gpu: str, name: str, dtype_id: str, inplace: bool):
    dtype = _MTA_FLOATS[dtype_id]
    if dtype == torch.float64:
        skip_if_metal(mojo_gpu, "no float64 on Apple GPUs")
    lo, hi = _MTA_UNARY[name]
    cpu, dev = _mta_lists(mojo_gpu, dtype, lo, hi, seed=len(name))
    if inplace:
        expected = [getattr(torch, name)(t) for t in cpu]
        before = [t._version for t in dev]
        with _one_launch(f"_foreach_{name}_"):
            getattr(torch, f"_foreach_{name}_")(dev)
        actual = dev
        assert [t._version - v for t, v in zip(dev, before)] == [1] * len(dev)
    else:
        expected = [getattr(torch, name)(t) for t in cpu]
        with _one_launch(f"_foreach_{name}"):
            actual = getattr(torch, f"_foreach_{name}")(dev)
    _assert_lists(actual, expected, **_mta_tol(dtype))


@pytest.mark.parametrize("dtype", [torch.int8, torch.int32, torch.int64, torch.uint8])
@pytest.mark.parametrize("name", ["abs", "neg"])
def test_mta_unary_integers(mojo_gpu: str, name: str, dtype: torch.dtype):
    lo, hi = (0, 200) if dtype == torch.uint8 else (-100, 100)
    cpu, dev = _mta_lists(mojo_gpu, dtype, lo, hi, seed=3)
    with _one_launch(f"_foreach_{name}"):
        actual = getattr(torch, f"_foreach_{name}")(dev)
    _assert_lists(actual, [getattr(torch, name)(t) for t in cpu])


def test_mta_abs_and_zero_on_bool(mojo_gpu: str):
    """CUDA's abs takes bool (the CPU kernel does not), as the identity."""
    cpu = [torch.tensor([True, False, True]), torch.zeros(1025, dtype=torch.bool)]
    dev = [t.to(mojo_gpu) for t in cpu]
    with _one_launch("_foreach_abs"):
        actual = torch._foreach_abs(dev)
    _assert_lists(actual, cpu)
    with _one_launch("_foreach_zero_"):
        torch._foreach_zero_(dev)
    _assert_lists(dev, [torch.zeros_like(t) for t in cpu])


@pytest.mark.parametrize(
    "dtype", [torch.float32, torch.float16, torch.float64, torch.int64, torch.bool]
)
def test_mta_zero_(mojo_gpu: str, dtype: torch.dtype):
    if dtype == torch.float64:
        skip_if_metal(mojo_gpu, "no float64 on Apple GPUs")
    cpu, dev = _mta_lists(mojo_gpu, dtype, 1, 5, seed=1)
    with _one_launch("_foreach_zero_"):
        torch._foreach_zero_(dev)
    _assert_lists(dev, [torch.zeros_like(t) for t in cpu])


def test_mta_unary_fallbacks_match_cpu(mojo_gpu: str):
    """Lists the fast route does not take still get ATen's answer."""
    # Mixed dtypes.
    cpu = [torch.randn(5), torch.randn(9).to(torch.bfloat16)]
    dev = [t.to(mojo_gpu) for t in cpu]
    _assert_lists(
        torch._foreach_exp(dev), [t.exp() for t in cpu], **_mta_tol(torch.bfloat16)
    )
    # Not dense.
    base = torch.randn(40)
    dev_base = base.to(mojo_gpu)
    _assert_lists(
        torch._foreach_sigmoid([dev_base[::2], dev_base[1::4]]),
        [base[::2].sigmoid(), base[1::4].sigmoid()],
    )
    # An integer list: the per-tensor op promotes it.
    ints = [torch.arange(5), torch.arange(1025) % 7]
    _assert_lists(
        torch._foreach_exp([t.to(mojo_gpu) for t in ints]), [t.exp() for t in ints]
    )


# --- binary: Scalar, ScalarList, List, Tensor ----------------------------------


# Binary overloads: (op, the overloads it has, (lo, hi) of self, of other).
_MTA_BINARY = {
    "add": (("Scalar", "ScalarList", "List", "Tensor"), (-3, 3), (-3, 3)),
    "sub": (("Scalar", "ScalarList", "List"), (-3, 3), (-3, 3)),
    "mul": (("Scalar", "ScalarList", "List", "Tensor"), (-3, 3), (-3, 3)),
    "div": (("Scalar", "ScalarList", "List", "Tensor"), (-3, 3), (0.5, 3)),
    "clamp_max": (("Scalar", "ScalarList", "List"), (-3, 3), (-1, 1)),
    "clamp_min": (("Scalar", "ScalarList", "List"), (-3, 3), (-1, 1)),
    "maximum": (("Scalar", "ScalarList", "List"), (-3, 3), (-1, 1)),
    "minimum": (("Scalar", "ScalarList", "List"), (-3, 3), (-1, 1)),
    "pow": (("Scalar", "ScalarList", "List"), (0.25, 2), (-2, 2)),
}
_MTA_BINARY_CASES = [
    (op, overload, dtype_id)
    for op, (overloads, _, _) in _MTA_BINARY.items()
    for overload in overloads
    for dtype_id in _MTA_FLOATS
]
_MTA_INT_DTYPES = {"i32": torch.int32, "i64": torch.int64, "u8": torch.uint8}


def _per_tensor_ref(op: str):
    """What ATen's slow kernel calls per tensor for `_foreach_<op>`."""
    return {"maximum": torch.clamp_min, "minimum": torch.clamp_max}.get(
        op, getattr(torch, op)
    )


def _binary_call(
    op: str,
    overload: str,
    inplace: bool,
    selves: list[torch.Tensor],
    others: list[torch.Tensor],
    scalar: float | int | bool,
    scalars: list[float] | list[int],
    device: torch.Tensor,
    alpha: float | int | None,
):
    fn = getattr(torch, f"_foreach_{op}_" if inplace else f"_foreach_{op}")
    kwargs = {} if alpha is None else {"alpha": alpha}
    if overload == "Scalar":
        return fn(selves, scalar)
    if overload == "ScalarList":
        return fn(selves, scalars)
    if overload == "List":
        return fn(selves, others, **kwargs)
    return fn(selves, device, **kwargs)


def _binary_expected(op, overload, selves, others, scalar, scalars, alpha):
    ref = _per_tensor_ref(op)
    kwargs = {} if alpha is None else {"alpha": alpha}
    out = []
    for i, t in enumerate(selves):
        if overload == "Scalar":
            out.append(ref(t, scalar))
        elif overload == "ScalarList":
            out.append(ref(t, scalars[i]))
        elif overload == "List":
            out.append(ref(t, others[i], **kwargs))
        else:
            out.append(ref(t, torch.tensor(scalar, dtype=t.dtype), **kwargs))
    return out


def _run_binary(mojo_gpu, op, overload, dtype, inplace, scalar, scalars, alpha):
    _, lo_hi, other_lo_hi = _MTA_BINARY[op]
    cpu, dev = _mta_lists(mojo_gpu, dtype, *lo_hi, seed=11)
    cpu_o, dev_o = _mta_lists(mojo_gpu, dtype, *other_lo_hi, seed=12)
    if op == "div" and not dtype.is_floating_point:
        pytest.skip("integer division promotes: not a fast-route case")
    alpha = alpha if op in ("add", "sub") and overload in ("List", "Tensor") else None
    expected = _binary_expected(op, overload, cpu, cpu_o, scalar, scalars, alpha)
    suffix = "_" if inplace else ""
    other = torch.tensor(scalar, dtype=dtype, device=mojo_gpu)
    with _one_launch(f"_foreach_{op}{suffix}.{overload}"):
        result = _binary_call(
            op, overload, inplace, dev, dev_o, scalar, scalars, other, alpha
        )
    actual = dev if inplace else result
    tol = _mta_tol(dtype)
    if op == "pow" and dtype == torch.float32:
        tol = {"rtol": 2e-5, "atol": 1e-6}
    _assert_lists(actual, expected, **tol)


@pytest.mark.parametrize("inplace", [False, True], ids=["out", "inplace"])
@pytest.mark.parametrize(("op", "overload", "dtype_id"), _MTA_BINARY_CASES)
def test_mta_binary_matches_cpu(
    mojo_gpu: str, op: str, overload: str, dtype_id: str, inplace: bool
):
    dtype = _MTA_FLOATS[dtype_id]
    if dtype == torch.float64:
        skip_if_metal(mojo_gpu, "no float64 on Apple GPUs")
    scalars = [0.5 + 0.25 * i for i in range(len(_MTA_SIZES))]
    if op == "div":
        scalars = [s + 0.1 for s in scalars]
    _run_binary(mojo_gpu, op, overload, dtype, inplace, 0.75, scalars, alpha=-1.5)


@pytest.mark.parametrize("inplace", [False, True], ids=["out", "inplace"])
@pytest.mark.parametrize("dtype_id", list(_MTA_INT_DTYPES))
@pytest.mark.parametrize(
    ("op", "overload"),
    [(op, overload) for op in ("add", "sub", "mul") for overload in _MTA_BINARY[op][0]],
)
def test_mta_binary_integers(
    mojo_gpu: str, op: str, overload: str, dtype_id: str, inplace: bool
):
    """CUDA's add / sub / mul take integer lists on the fast route (an
    integral Scalar keeps the dtype)."""
    dtype = _MTA_INT_DTYPES[dtype_id]
    scalars = [i - 3 for i in range(len(_MTA_SIZES))]
    _run_binary(mojo_gpu, op, overload, dtype, inplace, 3, scalars, alpha=2)


@pytest.mark.parametrize("op", ["add", "mul"])
def test_mta_binary_bool(mojo_gpu: str, op: str):
    cpu, dev = _mta_lists(mojo_gpu, torch.bool, 0, 2, seed=5)
    cpu_o, dev_o = _mta_lists(mojo_gpu, torch.bool, 0, 2, seed=6)
    ref = getattr(torch, op)
    with _one_launch(f"_foreach_{op}.Scalar"):
        actual = getattr(torch, f"_foreach_{op}")(dev, True)
    _assert_lists(actual, [ref(t, True) for t in cpu])
    with _one_launch(f"_foreach_{op}_.ScalarList"):
        getattr(torch, f"_foreach_{op}_")(dev, [i % 2 == 0 for i in range(len(dev))])
    expected = [ref(t, i % 2 == 0) for i, t in enumerate(cpu)]
    _assert_lists(dev, expected)
    # A list operand: add's default alpha=1 is an int, which promotes a bool
    # list (ForeachUtils.h), so only mul takes the fast route here.
    actual = getattr(torch, f"_foreach_{op}")(dev, dev_o)
    _assert_lists(actual, [ref(t, o) for t, o in zip(expected, cpu_o)])


@pytest.mark.parametrize("dtype_id", list(_MTA_FLOATS))
def test_mta_pow_scalar_and_tensor(mojo_gpu: str, dtype_id: str):
    dtype = _MTA_FLOATS[dtype_id]
    if dtype == torch.float64:
        skip_if_metal(mojo_gpu, "no float64 on Apple GPUs")
    cpu, dev = _mta_lists(mojo_gpu, dtype, -2, 2, seed=9)
    with _one_launch("_foreach_pow.ScalarAndTensor"):
        actual = torch._foreach_pow(1.75, dev)
    _assert_lists(actual, [torch.pow(1.75, t) for t in cpu], **_mta_tol(dtype))


def test_mta_pow_scalar_and_integer_tensors_promote(mojo_gpu: str):
    """`_foreach_pow(1.5, int_list)` follows result_type, like CPU torch and
    the per-tensor `pow(Scalar, Tensor)`: float32 [1.5, 2.25]. Stock CUDA
    (torch 2.11, H100) returns int64 [1, 1]: its fast route
    (ForeachBinaryOpScalar.cu:184, `can_use_fast_route(exponent)` without
    the scalar) keeps the integer list and the functor's
    `scalar.to<opmath_t>()` (:47) truncates 1.5 to 1. Pinned on purpose."""
    exponents = [torch.tensor([1, 2]), torch.arange(5)]
    actual = torch._foreach_pow(1.5, [t.to(mojo_gpu) for t in exponents])
    _assert_lists(actual, [torch.pow(1.5, t) for t in exponents])
    assert actual[0].dtype == torch.float32
    assert actual[0].cpu().tolist() == [1.5, 2.25]


_SIGNED_ZEROS = [(0.0, -0.0), (-0.0, 0.0), (0.0, 0.0), (-0.0, -0.0)]


@pytest.mark.parametrize("overload", ["Scalar", "ScalarList", "List"])
@pytest.mark.parametrize("op", ["maximum", "minimum", "clamp_min", "clamp_max"])
@pytest.mark.parametrize("dtype", [torch.float32, torch.bfloat16], ids=str)
def test_mta_min_max_signed_zeros_and_nan(
    mojo_gpu: str, dtype: torch.dtype, op: str, overload: str
):
    """ForeachMinMaxFunctors.cuh: `(isnan(a) || a > b) ? a : b`. Equal
    operands give the second one, so maximum(+0, -0) is -0 and
    maximum(-0, +0) is +0 (measured on stock CUDA, every op, overload and
    argument order) -- unlike the per-tensor ops; NaN on either side wins."""
    nan = float("nan")
    pairs = [*_SIGNED_ZEROS, (nan, 1.0), (1.0, nan), (2.0, 1.0), (1.0, 2.0)]
    selves = [torch.tensor([a], dtype=dtype, device=mojo_gpu) for a, _ in pairs]
    others = [b for _, b in pairs]
    fn = getattr(torch, f"_foreach_{op}")
    if overload == "Scalar":
        actual = [fn([t], b)[0] for t, b in zip(selves, others)]
    elif overload == "ScalarList":
        actual = fn(selves, others)
    else:
        actual = fn(
            selves, [torch.tensor([b], dtype=dtype, device=mojo_gpu) for b in others]
        )
    keep_max = op in ("maximum", "clamp_min")
    for (a, b), r in zip(pairs, actual, strict=True):
        if math.isnan(a) or math.isnan(b):
            want = nan
        elif a == b:
            want = b
        else:
            want = max(a, b) if keep_max else min(a, b)
        got = r.item()
        if math.isnan(want):
            assert math.isnan(got)
        else:
            assert got == want and math.copysign(1, got) == math.copysign(1, want)


def test_mta_sub_bool_rules(mojo_gpu: str):
    """sub_check like CUDA: the ScalarList overload checks every pair before
    anything runs (the float tensor stays untouched), the Scalar one the
    first tensor; the List overload computes bool lists with a bool alpha on
    its fast route and raises in the per-tensor sub otherwise."""
    a = torch.tensor([10.0], device=mojo_gpu)
    b = torch.tensor([True], device=mojo_gpu)
    with pytest.raises((RuntimeError, NotImplementedError), match="with a bool tensor"):
        torch._foreach_sub_([a, b], [1, 1])
    assert a.cpu().tolist() == [10.0]
    with pytest.raises((RuntimeError, NotImplementedError), match="with a bool tensor"):
        torch._foreach_sub_([b, a], 1)
    assert a.cpu().tolist() == [10.0]
    x = torch.tensor([True, False], device=mojo_gpu)
    y = torch.tensor([True, True], device=mojo_gpu)
    # Stock CUDA: [False, True] and [True, False].
    assert torch._foreach_sub([x], [y], alpha=True)[0].cpu().tolist() == [False, True]
    assert torch._foreach_sub([x], [y], alpha=False)[0].cpu().tolist() == [True, False]
    # The per-tensor sub.Tensor declines bool operands (CUDA: "two bool
    # tensors"); either way nothing is computed.
    with pytest.raises((RuntimeError, NotImplementedError)):
        torch._foreach_sub([x], [y])


def test_mta_dense_non_contiguous_lists(mojo_gpu: str):
    """Transposed (dense, non-contiguous) tensors with matching strides take
    the fast route and keep their layout in the out-of-place result."""
    cpu = [torch.randn(5, 7).t(), torch.randn(33, 65).t()]
    other = [torch.randn(5, 7).t(), torch.randn(33, 65).t()]
    dev = [t.to(mojo_gpu) for t in cpu]
    dev_o = [t.to(mojo_gpu) for t in other]
    assert all(t.stride() == o.stride() for t, o in zip(dev, dev_o))
    with _one_launch("_foreach_mul.List"):
        actual = torch._foreach_mul(dev, dev_o)
    for a, t in zip(actual, dev):
        assert a.stride() == t.stride()
    _assert_lists(actual, [a * b for a, b in zip(cpu, other)])


def test_mta_inplace_aliasing_is_sequential(mojo_gpu: str):
    """An in-place list reading another entry's tensor (here a swap) has the
    sequential result: the second update sees the first."""
    a, b = torch.randn(1000), torch.randn(1000)
    da, db = a.to(mojo_gpu), b.to(mojo_gpu)
    torch._foreach_add_([da, db], [db, da])
    a.add_(b)
    b.add_(a)
    _assert_lists([da, db], [a, b])
    # The same tensor at the same index is elementwise and takes the launch.
    x = torch.randn(3000)
    dx = x.to(mojo_gpu)
    with _one_launch("_foreach_mul_.List"):
        torch._foreach_mul_([dx], [dx])
    _assert_lists([dx], [x * x])


def test_mta_scalar_tensor_overloads(mojo_gpu: str):
    xs = [torch.randn(5), torch.randn(1030)]
    dev = [t.to(mojo_gpu) for t in xs]
    # A CPU tensor is its `item()` (div by its reciprocal, as CUDA does).
    two = torch.tensor(2.0)
    _assert_lists(torch._foreach_mul(dev, two), [t * 2.0 for t in xs])
    _assert_lists(torch._foreach_div(dev, torch.tensor([4.0])), [t / 4.0 for t in xs])
    # A device 0-d tensor of another dtype falls back to the per-tensor op.
    s = torch.tensor(3, device=mojo_gpu)
    _assert_lists(torch._foreach_mul(dev, s), [t * 3 for t in xs])
    with pytest.raises(RuntimeError, match="0 dim"):
        torch._foreach_add(dev, torch.ones(2, device=mojo_gpu))
    half = torch.tensor(0.5, device=mojo_gpu)
    with _one_launch("_foreach_add_.Tensor"):
        torch._foreach_add_(dev, half, alpha=-2)
    _assert_lists(dev, [t - 1.0 for t in xs])


def test_mta_binary_fallbacks_match_cpu(mojo_gpu: str):
    # Mismatched layouts across lists.
    a, b = torch.randn(4, 6), torch.randn(6, 4).t()
    _assert_lists(torch._foreach_add([a.to(mojo_gpu)], [b.to(mojo_gpu)]), [a + b])
    # An integer list with a float Scalar promotes (out of place) ...
    ints = [torch.arange(5), torch.arange(1025) % 7]
    dev_ints = [t.to(mojo_gpu) for t in ints]
    _assert_lists(torch._foreach_add(dev_ints, 1.5), [t + 1.5 for t in ints])
    _assert_lists(
        torch._foreach_mul(dev_ints, [0.5, 2.0]), [ints[0] * 0.5, ints[1] * 2.0]
    )
    _assert_lists(torch._foreach_div(dev_ints, 4), [t / 4 for t in ints])
    # ... and in place it raises like the per-tensor op.
    with pytest.raises(RuntimeError, match="can't be cast"):
        torch._foreach_div_(dev_ints, 4)


def test_mta_binary_argument_errors(mojo_gpu: str):
    xs = [torch.ones(3, device=mojo_gpu), torch.ones(4, device=mojo_gpu)]
    with pytest.raises(RuntimeError, match="same number of tensors, got 2 and 1"):
        torch._foreach_add(xs, xs[:1])
    with pytest.raises(RuntimeError, match="same number of elements as scalar list"):
        torch._foreach_mul_(xs, [1.0])


def _binary_calls(a, b, c, w) -> list[torch.Tensor]:
    return [
        *torch._foreach_add(a, b, alpha=0.3),
        *torch._foreach_sub(a, 0.7),
        *torch._foreach_mul(a, [0.1, 0.2, 0.3]),
        *torch._foreach_div(a, c),
        *torch._foreach_div(a, 3.0),
        *torch._foreach_maximum(a, b),
        *torch._foreach_clamp_max(a, 0.25),
    ]


@pytest.mark.skipif(not torch.cuda.is_available(), reason="needs stock CUDA")
@pytest.mark.parametrize("dtype", [torch.float32, torch.float16, torch.bfloat16])
def test_mta_binary_is_bit_exact_with_stock_cuda(mojo_gpu: str, dtype: torch.dtype):
    """add / sub with an alpha are one fma, div by a Scalar is CUDA's
    multiply by the reciprocal, maximum / minimum propagate NaN."""
    _stock_cuda_parity(dtype, mojo_gpu, _binary_calls)


# --- pointwise (addcmul, addcdiv) and ternary (lerp) ---------------------------


_MTA_POINTWISE_CASES = [
    (op, overload, dtype_id)
    for op in ("addcmul", "addcdiv")
    for overload in ("Scalar", "ScalarList", "Tensor")
    for dtype_id in _MTA_FLOATS
]


@pytest.mark.parametrize("inplace", [False, True], ids=["out", "inplace"])
@pytest.mark.parametrize(("op", "overload", "dtype_id"), _MTA_POINTWISE_CASES)
def test_mta_pointwise_matches_cpu(
    mojo_gpu: str, op: str, overload: str, dtype_id: str, inplace: bool
):
    dtype = _MTA_FLOATS[dtype_id]
    if dtype == torch.float64:
        skip_if_metal(mojo_gpu, "no float64 on Apple GPUs")
    cpu, dev = _mta_lists(mojo_gpu, dtype, -3, 3, seed=21)
    cpu1, dev1 = _mta_lists(mojo_gpu, dtype, -2, 2, seed=22)
    cpu2, dev2 = _mta_lists(mojo_gpu, dtype, 0.5, 2, seed=23)
    values = [0.5 - 0.3 * i for i in range(len(cpu))]
    if overload == "Scalar":
        arg, per = -0.75, [-0.75] * len(cpu)
    elif overload == "ScalarList":
        arg, per = values, values
    else:
        arg, per = torch.tensor(values, dtype=torch.float64), values
    expected = [
        getattr(torch, op)(t, a, b, value=v)
        for t, a, b, v in zip(cpu, cpu1, cpu2, per, strict=True)
    ]
    name = f"_foreach_{op}_" if inplace else f"_foreach_{op}"
    with _one_launch(f"{name}.{overload}"):
        result = getattr(torch, name)(dev, dev1, dev2, arg)
    _assert_lists(dev if inplace else result, expected, **_mta_tol(dtype))


@pytest.mark.parametrize("inplace", [False, True], ids=["out", "inplace"])
@pytest.mark.parametrize("dtype_id", list(_MTA_FLOATS))
@pytest.mark.parametrize("overload", ["List", "Scalar", "ScalarList"])
def test_mta_lerp_matches_cpu(
    mojo_gpu: str, overload: str, dtype_id: str, inplace: bool
):
    dtype = _MTA_FLOATS[dtype_id]
    if dtype == torch.float64:
        skip_if_metal(mojo_gpu, "no float64 on Apple GPUs")
    cpu, dev = _mta_lists(mojo_gpu, dtype, -3, 3, seed=31)
    cpu_e, dev_e = _mta_lists(mojo_gpu, dtype, -3, 3, seed=32)
    cpu_w, dev_w = _mta_lists(mojo_gpu, dtype, -0.5, 1.5, seed=33)
    weights = [0.1, 0.4999, 0.5, 0.9, -0.25, 1.25, 0.75]
    if overload == "List":
        arg, per = dev_w, cpu_w
    elif overload == "Scalar":
        arg, per = 0.3, [0.3] * len(cpu)
    else:
        arg, per = weights, weights
    expected = [torch.lerp(t, e, w) for t, e, w in zip(cpu, cpu_e, per)]
    name = "_foreach_lerp_" if inplace else "_foreach_lerp"
    with _one_launch(f"{name}.{overload}"):
        result = getattr(torch, name)(dev, dev_e, arg)
    _assert_lists(dev if inplace else result, expected, **_mta_tol(dtype))


def test_mta_pointwise_long_lists_span_several_launches(mojo_gpu: str):
    sizes = [(i * 37) % 700 for i in range(130)]
    cpu, dev = _mta_lists(mojo_gpu, torch.float32, -2, 2, seed=41, sizes=sizes)
    cpu_o, dev_o = _mta_lists(mojo_gpu, torch.float32, -2, 2, seed=42, sizes=sizes)
    with _one_launch("_foreach_lerp.List"):
        actual = torch._foreach_lerp(dev, dev_o, dev_o)
    _assert_lists(actual, [torch.lerp(a, b, b) for a, b in zip(cpu, cpu_o)])
    with _one_launch("_foreach_addcdiv_.ScalarList"):
        torch._foreach_addcdiv_(dev, dev_o, dev, [0.01 * i for i in range(130)])
    for a, b, i in zip(cpu, cpu_o, range(130)):
        a.addcdiv_(b, a.clone(), value=0.01 * i)
    _assert_lists(dev, cpu, **_mta_tol(torch.float32))


def test_mta_pointwise_argument_errors(mojo_gpu: str):
    xs = [torch.ones(3, device=mojo_gpu), torch.ones(4, device=mojo_gpu)]
    with pytest.raises(RuntimeError, match="Expected scalars to be on CPU"):
        torch._foreach_addcmul(xs, xs, xs, torch.ones(2, device=mojo_gpu))
    with pytest.raises(RuntimeError, match="Expected length of scalars"):
        torch._foreach_addcmul(xs, xs, xs, torch.ones(3))
    with pytest.raises(RuntimeError, match="same number of elements as scalar list"):
        torch._foreach_lerp(xs, xs, [0.5])


def _pointwise_calls(a, b, c, w) -> list[torch.Tensor]:
    return [
        *torch._foreach_addcmul(a, b, c, 0.3),
        *torch._foreach_addcmul(a, b, c),
        *torch._foreach_addcdiv(a, b, c, [0.1, -0.2, 0.3]),
        *torch._foreach_addcdiv(a, b, c),
        *torch._foreach_lerp(a, b, w),
        *torch._foreach_lerp(a, b, 0.7),
        *torch._foreach_lerp(a, b, [0.2, 0.5, 0.9]),
    ]


@pytest.mark.skipif(not torch.cuda.is_available(), reason="needs stock CUDA")
@pytest.mark.parametrize("dtype", [torch.float32, torch.float16, torch.bfloat16])
def test_mta_pointwise_is_bit_exact_with_stock_cuda(mojo_gpu: str, dtype: torch.dtype):
    _stock_cuda_parity(dtype, mojo_gpu, _pointwise_calls)


# --- the reductions: _foreach_norm, _foreach_powsum, _foreach_max -----------

_MTA_NORM_DTYPES = {
    "f32": (torch.float32, None),
    "f16": (torch.float16, None),
    "bf16": (torch.bfloat16, None),
    "f64": (torch.float64, None),
    "f16-to-f32": (torch.float16, torch.float32),
    "f32-to-f64": (torch.float32, torch.float64),
}


@pytest.mark.parametrize(
    "ord_", [0, 1, 2, float("inf")], ids=["l0", "l1", "l2", "linf"]
)
@pytest.mark.parametrize("dtype_id", list(_MTA_NORM_DTYPES))
def test_mta_norm_matches_cpu(mojo_gpu: str, dtype_id: str, ord_: float):
    dtype, out_dtype = _MTA_NORM_DTYPES[dtype_id]
    if torch.float64 in (dtype, out_dtype):
        skip_if_metal(mojo_gpu, "no float64 on Apple GPUs")
    sizes = [7, 1, 65_537, 131_072, 3, 200_001]
    if ord_ != float("inf"):
        sizes.insert(1, 0)
    cpu, dev = _mta_lists(mojo_gpu, dtype, -2, 2, seed=51, sizes=sizes)
    cpu[0][3] = 0.0
    dev[0][3] = 0.0
    kwargs = {} if out_dtype is None else {"dtype": out_dtype}
    with _one_launch("_foreach_norm.Scalar"):
        actual = torch._foreach_norm(dev, ord_, **kwargs)
    expected = [
        torch.linalg.vector_norm(t.double(), ord_).to(out_dtype or dtype) for t in cpu
    ]
    tol = _mta_tol(out_dtype or dtype)
    if (out_dtype or dtype) == torch.float32:
        tol = {"rtol": 2e-5, "atol": 1e-5}
    _assert_lists(actual, expected, **tol)


@pytest.mark.parametrize("ord_", [1, 2])
@pytest.mark.parametrize(
    ("dtype", "out_dtype"),
    [
        (torch.float32, None),
        (torch.bfloat16, None),
        (torch.float16, torch.float32),
        (torch.float32, torch.float64),
    ],
    ids=["f32", "bf16", "f16-to-f32", "f32-to-f64"],
)
def test_mta_powsum_matches_cpu(
    mojo_gpu: str, dtype: torch.dtype, out_dtype: torch.dtype | None, ord_: int
):
    if out_dtype == torch.float64:
        skip_if_metal(mojo_gpu, "no float64 on Apple GPUs")
    cpu, dev = _mta_lists(mojo_gpu, dtype, -2, 2, seed=52)
    kwargs = {} if out_dtype is None else {"dtype": out_dtype}
    with _one_launch("_foreach_powsum.Scalar"):
        actual = torch._foreach_powsum(dev, ord_, **kwargs)
    result = out_dtype or dtype
    expected = [
        torch.linalg.vector_norm(t.double(), ord_).pow(ord_).to(result) for t in cpu
    ]
    tol = (
        {"rtol": 2e-5, "atol": 1e-5}
        if result != dtype or dtype == torch.float32
        else _mta_tol(dtype)
    )
    _assert_lists(actual, expected, **tol)


def test_mta_norm_fallbacks_and_errors(mojo_gpu: str):
    cpu = [torch.randn(10), torch.randn(65_540)]
    dev = [t.to(mojo_gpu) for t in cpu]
    # An ord with no batched functor is the per-tensor linalg_vector_norm.
    _assert_lists(
        torch._foreach_norm(dev, 3.5),
        [torch.linalg.vector_norm(t, 3.5) for t in cpu],
        rtol=2e-5,
        atol=1e-5,
    )
    _assert_lists(
        torch._foreach_powsum(dev, 3),
        [torch.linalg.vector_norm(t.double(), 3).pow(3).float() for t in cpu],
        rtol=2e-5,
        atol=1e-4,
    )
    with pytest.raises(RuntimeError, match="infinity norm on an empty tensor"):
        torch._foreach_norm([torch.empty(0, device=mojo_gpu)], float("inf"))
    with pytest.raises(RuntimeError, match="convertible without narrowing"):
        torch._foreach_norm(dev, 2, dtype=torch.float16)
    with pytest.raises(
        RuntimeError, match="should be real for real inputs, but got ComplexFloat"
    ):
        torch._foreach_norm(dev, 2, dtype=torch.complex64)
    with pytest.raises(RuntimeError, match="floating point or complex, but got Int"):
        torch._foreach_norm(dev, 2, dtype=torch.int32)
    with pytest.raises(
        NotImplementedError, match="\"powsum_cuda\" not implemented for 'Long'"
    ):
        torch._foreach_powsum([torch.arange(5, device=mojo_gpu)], 2)
    with pytest.raises(RuntimeError, match="expects ord to be integer or float"):
        torch._foreach_norm(dev, True)


@pytest.mark.parametrize(
    "dtype",
    [torch.float32, torch.float16, torch.bfloat16, torch.float64, torch.int8,
     torch.int16, torch.int32, torch.int64, torch.uint8, torch.bool],
    ids=str,
)  # fmt: skip
def test_mta_max_matches_cpu(mojo_gpu: str, dtype: torch.dtype):
    if dtype == torch.float64:
        skip_if_metal(mojo_gpu, "no float64 on Apple GPUs")
    sizes = [7, 1, 65_537, 131_072, 3, 200_001]
    lo, hi = (0, 2) if dtype == torch.bool else (-100, 100)
    if dtype == torch.uint8:
        lo = 0
    cpu, dev = _mta_lists(mojo_gpu, dtype, lo, hi, seed=53, sizes=sizes)
    if dtype.is_floating_point:
        cpu[3][70_000] = float("nan")
        dev[3][70_000] = float("nan")
        cpu[4][:] = float("-inf")
        dev[4][:] = float("-inf")
    with _one_launch("_foreach_max"):
        actual = torch._foreach_max(dev)
    _assert_lists(actual, [t.max() for t in cpu])
    with pytest.raises(RuntimeError, match="Expected reduction dim to be specified"):
        torch._foreach_max([*dev, torch.empty(0, dtype=dtype, device=mojo_gpu)])


def test_mta_max_of_negative_infinities_is_negative_infinity(mojo_gpu: str):
    """-inf, as CPU torch and the per-tensor max. Stock CUDA (torch 2.11,
    H100) returns -3.4028e38 for float32 (-1.7977e308 for float64): its
    reduction is seeded with `std::numeric_limits<T>::lowest()`
    (ForeachReduceOp.cu:77/:105/:127, the scratch fill at :192). Pinned on
    purpose."""
    for dtype in (torch.float32, torch.bfloat16):
        x = torch.full((3,), float("-inf"), dtype=dtype)
        (actual,) = torch._foreach_max([x.to(mojo_gpu)])
        assert actual.cpu().item() == float("-inf")
        assert torch._foreach_max([x])[0].item() == float("-inf")


def _reduce_outcome(fn, tensors: list[torch.Tensor], ord_: float, dtype) -> str:
    """A reduction's result (value, dtype) or its error (type, message
    without the mojo device's `[aten::...]` suffix), as one comparable
    string."""
    kwargs = {} if dtype is None else {"dtype": dtype}
    try:
        (r,) = fn(tensors, ord_, **kwargs)
    except (RuntimeError, NotImplementedError) as error:
        message = str(error).split(" [aten::")[0].split("\n")[0]
        return f"{type(error).__name__}: {message}"
    return f"{r.cpu().item()!r} {r.dtype}"


_REDUCE_ORDS = (2, 3, -2, 0, 1, 2.5, -1, float("inf"), float("-inf"), 0.5,
                float("nan"), 1e100, -1e100, 1e-100, -1 / 3)  # fmt: skip
_REDUCE_INPUTS = [torch.int64, torch.int32, torch.int16, torch.int8, torch.uint8,
                  torch.uint16, torch.uint32, torch.uint64, torch.bool,
                  torch.float16, torch.bfloat16, torch.float32,
                  torch.float64]  # fmt: skip
_REDUCE_DTYPES = [None, torch.int64, torch.int32, torch.bool, torch.float16,
                  torch.bfloat16, torch.float32, torch.float64, torch.complex64,
                  torch.complex128]  # fmt: skip


@pytest.mark.skipif(not torch.cuda.is_available(), reason="needs stock CUDA")
@pytest.mark.parametrize("fn_name", ["norm", "powsum"])
@pytest.mark.parametrize("dtype", _REDUCE_DTYPES, ids=str)
@pytest.mark.parametrize("input_dtype", _REDUCE_INPUTS, ids=str)
def test_mta_norm_and_powsum_match_stock_cuda(
    mojo_gpu: str, input_dtype: torch.dtype, dtype: torch.dtype | None, fn_name: str
):
    """Every (input dtype, `dtype=`, ord, size) against stock CUDA: its fast
    route, and the per-tensor linalg_vector_norm / linalg__powsum of its slow
    route -- their errors (an integer computation dtype has no powsum
    kernel, an integer input no norm) and their arithmetic (the elements
    cast to the result dtype, accumulated in its opmath, rounded once)."""
    fn = getattr(torch, f"_foreach_{fn_name}")
    for n in (5, 0, 1):
        base = torch.arange(1, n + 1) if n != 1 else torch.tensor([-3])
        for ord_ in _REDUCE_ORDS:
            if dtype == torch.float64 and is_metal(mojo_gpu):
                continue
            outcomes = [
                _reduce_outcome(fn, [base.to(input_dtype).to(device)], ord_, dtype)
                for device in ("cuda", mojo_gpu)
            ]
            stock, ours = (o.replace("powsum_cuda", "powsum") for o in outcomes)
            assert ours == stock, (n, ord_)


def test_mta_powsum_rounds_once(mojo_gpu: str):
    """An int32 list summed in float16: each 10000 ** -2 is a float16
    subnormal, but the powers accumulate in float32 and the sum rounds once
    -- 1.0013580322265625e-05 on stock CUDA."""
    x = [torch.full((1000,), 10000, dtype=torch.int32, device=mojo_gpu)]
    (r,) = torch._foreach_powsum(x, -2, dtype=torch.float16)
    assert r.dtype == torch.float16
    assert r.cpu().item() == 1.0013580322265625e-05
    with pytest.raises(
        NotImplementedError, match="\"powsum_cuda\" not implemented for 'Long'"
    ):
        torch._foreach_powsum(x, 2, dtype=torch.int64)


# CUDA-free pins of the stock-CUDA behaviours above (torch 2.11, H100),
# so a machine without stock CUDA still checks them.


@pytest.mark.parametrize("ord_", [float("nan"), 1e100, 3.0, 0.5, 1e-100])
@pytest.mark.parametrize("fn_name", ["norm", "powsum"])
def test_mta_empty_reductions_are_zero(mojo_gpu: str, fn_name: str, ord_: float):
    """No kernel runs on an empty tensor: the result is the identity, 0,
    with no root -- for ord nan and 1e100 too (not nan, not 1)."""
    fn = getattr(torch, f"_foreach_{fn_name}")
    (r,) = fn([torch.empty(0, device=mojo_gpu)], ord_)
    assert r.dtype == torch.float32 and r.cpu().item() == 0.0


def test_mta_reductions_on_integer_lists_without_cuda(mojo_gpu: str):
    """Ord 3 is off the fast route, whose own dtype check (ord 2) refuses an
    integer `dtype` with a different message."""
    x = [torch.arange(5, device=mojo_gpu)]
    with pytest.raises(
        RuntimeError,
        match="Expected a floating point or complex tensor as input. Got Long",
    ):
        torch._foreach_norm(x, 2, dtype=torch.float32)
    (zero,) = torch._foreach_powsum(
        [torch.empty(0, dtype=torch.int32, device=mojo_gpu)], 2
    )
    assert zero.dtype == torch.int64 and zero.cpu().item() == 0
    (zero,) = torch._foreach_powsum(
        [torch.empty(0, device=mojo_gpu)], 3, dtype=torch.int32
    )
    assert zero.dtype == torch.int32 and zero.cpu().item() == 0
    with pytest.raises(
        NotImplementedError, match="\"powsum_cuda\" not implemented for 'Int'"
    ):
        torch._foreach_powsum([torch.ones(3, device=mojo_gpu)], 3, dtype=torch.int32)
    (r,) = torch._foreach_powsum(x, 2, dtype=torch.float32)
    assert r.cpu().item() == 30.0
    for ord_, text in ((-1 / 3, "-0.333333"), (-1e100, "-1e+100"), (-2.5, "-2.5"),
                       (-123456789.0, "-1.23457e+08"), (-0.0001234567, "-0.000123457"),
                       (-2, "-2")):  # fmt: skip
        with pytest.raises(
            RuntimeError, match=f"cannot compute the {re.escape(text)} norm"
        ):
            torch._foreach_norm([torch.empty(0, device=mojo_gpu)], ord_)


def test_mta_max_refusals(mojo_gpu: str):
    for dtype, name in ((torch.uint16, "UInt16"), (torch.uint32, "UInt32")):
        x = torch.ones(3, dtype=dtype, device=mojo_gpu)
        with pytest.raises(
            NotImplementedError,
            match=f"\"foreach_tensor_max_cuda_scalar_type\" not implemented for '{name}'",
        ):
            torch._foreach_max([x])
    with pytest.raises(RuntimeError, match="Expected reduction dim to be specified"):
        torch._foreach_max(
            [torch.ones(2, device=mojo_gpu), torch.empty(0, device=mojo_gpu)]
        )


def test_mta_reductions_on_mixed_device_lists(mojo_gpu: str):
    """A list mixing the mojo device and the CPU takes the per-tensor route:
    each result on its input's device, as on stock CUDA."""
    a = torch.tensor([3.0, -4.0])
    b = torch.tensor([1.5, 2.0])
    lists = [a.to(mojo_gpu), b]
    for fn, args, want in (
        (torch._foreach_norm, (2,), [5.0, 2.5]),
        (
            torch._foreach_norm,
            (3,),
            [
                torch.linalg.vector_norm(a, 3).item(),
                torch.linalg.vector_norm(b, 3).item(),
            ],
        ),
        (torch._foreach_norm, (float("inf"),), [4.0, 2.0]),
        (torch._foreach_powsum, (2,), [25.0, 6.25]),
        (lambda ts, *_: torch._foreach_max(ts), (), [3.0, 2.0]),
    ):
        got = fn(lists, *args)
        assert [t.device.type for t in got] == [torch.device(mojo_gpu).type, "cpu"]
        assert [t.cpu().item() for t in got] == pytest.approx(want, rel=1e-6)
    got = torch._foreach_add(lists, [b.to(mojo_gpu), a], alpha=2)
    assert [t.device.type for t in got] == [torch.device(mojo_gpu).type, "cpu"]
    torch.testing.assert_close(got[1], a * 2 + b)


@pytest.mark.parametrize(
    ("dtype", "real"),
    [(torch.complex64, torch.float32), (torch.complex128, torch.float64),
     (torch.complex32, torch.float16)],
    ids=["c64", "c128", "c32"],
)  # fmt: skip
def test_mta_powsum_complex_dtype_is_its_real_dtype(
    mojo_gpu: str, dtype: torch.dtype, real: torch.dtype
):
    """Off the fast route (ord 1 / 2 refuse a complex `dtype` for a real
    list), powsum computes in the complex dtype's real value dtype, as on
    stock CUDA: `_foreach_powsum([ones(2)], 3, dtype=complex64)` is float32
    2.0; norm refuses it (linalg.vector_norm's check)."""
    if real == torch.float64:
        skip_if_metal(mojo_gpu, "no float64 on Apple GPUs")
    ones = [torch.ones(2, device=mojo_gpu)]
    for ord_, want in ((3, 2.0), (0, 2.0), (-1, 2.0), (float("inf"), 2.0)):
        (r,) = torch._foreach_powsum(ones, ord_, dtype=dtype)
        assert r.dtype == real and r.cpu().item() == want
    (r,) = torch._foreach_powsum([torch.arange(3, device=mojo_gpu)], 2, dtype=dtype)
    assert r.dtype == real and r.cpu().item() == 5.0
    (r,) = torch._foreach_powsum([torch.empty(0, device=mojo_gpu)], 3, dtype=dtype)
    assert r.dtype == real and r.cpu().item() == 0.0
    with pytest.raises(RuntimeError, match="dtype should be real for real inputs"):
        torch._foreach_powsum(ones, 2, dtype=dtype)
    with pytest.raises(
        RuntimeError, match="linalg.vector_norm: dtype should be real for real inputs"
    ):
        torch._foreach_norm(ones, 3, dtype=dtype)


def test_mta_norm_message_prints_integer_orders_exactly(mojo_gpu: str):
    with pytest.raises(RuntimeError, match="the -9007199254740993 norm"):
        torch._foreach_norm([torch.empty(0, device=mojo_gpu)], -9007199254740993)


@pytest.mark.skipif(not torch.cuda.is_available(), reason="needs stock CUDA")
@pytest.mark.parametrize("input_dtype", _REDUCE_INPUTS, ids=str)
def test_mta_max_matches_stock_cuda(mojo_gpu: str, input_dtype: torch.dtype):
    """_foreach_max of every input dtype the device holds, on and off the
    fast route (dense / strided), against stock CUDA."""
    if input_dtype == torch.float64 and is_metal(mojo_gpu):
        pytest.skip("no float64 on Apple GPUs")

    def outcome(tensors: list[torch.Tensor]) -> str:
        try:
            (r,) = torch._foreach_max(tensors)
        except (RuntimeError, NotImplementedError) as error:
            message = str(error).split(" [aten::")[0].split("\n")[0]
            return f"{type(error).__name__}: {message}"
        return f"{r.cpu().item()!r} {r.dtype}"

    for n in (5, 0, 1):
        base = torch.arange(1, 2 * n + 1).to(input_dtype)
        for view in (lambda t: t[:n], lambda t: t[::2]):
            stock, ours = (
                outcome([view(base.to(device))]) for device in ("cuda", mojo_gpu)
            )
            assert ours == stock, n


@pytest.mark.parametrize("dtype", [torch.uint16, torch.uint32, torch.uint64], ids=str)
def test_mta_wide_unsigned_norm_and_powsum(mojo_gpu: str, dtype: torch.dtype):
    """Wide unsigned lists take the same slow route as the other integers:
    cast to the result dtype, accumulated in its opmath, rounded once (the
    values stock CUDA returns)."""
    ones = [torch.ones(2, dtype=dtype, device=mojo_gpu)]
    (r,) = torch._foreach_powsum(ones, 3, dtype=torch.complex64)
    assert r.dtype == torch.float32 and r.cpu().item() == 2.0
    x = [torch.full((1000,), 10000, dtype=dtype, device=mojo_gpu)]
    (r,) = torch._foreach_powsum(x, -2, dtype=torch.float16)
    assert r.cpu().item() == 1.0013580322265625e-05
    with pytest.raises(NotImplementedError, match="not implemented for 'Long'"):
        torch._foreach_powsum(ones, 2)
    names = {torch.uint16: "UInt16", torch.uint32: "UInt32", torch.uint64: "UInt64"}
    with pytest.raises(RuntimeError, match=f"Got {names[dtype]}"):
        torch._foreach_norm(ones, 2, dtype=torch.float32)
