"""Native backend: data_movement group (see docs/native_backend.md and
torch_mojo_backend/mojo/tmb/ops/data_movement.mojo).

Public-API checks only (no `TorchMojoTensor`/`aten_fast`/old-eager
internals): every op is exercised through ordinary `torch` calls on tensors
living on a `mojo` device, and `call_checker` confirms the native op (not
some other route) actually ran.
"""

import os
import re
import subprocess
import sys
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

import pytest
import torch

from torch_mojo_backend import aten_functions, get_accelerators

# tests/native/conftest.py registers devices at fixture setup. Registration
# during collection would also affect deselected tests and break CUDA autograd
# in the separate CUDA compiler job.


def _fill(shape: tuple[int, ...], dtype: torch.dtype) -> torch.Tensor:
    """A deterministic, non-RNG input: consecutive elements differ, so a
    kernel that reads/writes one element off is caught by VALUE, not just by
    shape."""
    numel = 1
    for extent in shape:
        numel *= extent
    base = torch.arange(numel, dtype=torch.int64) % 251
    if dtype.is_floating_point:
        base = base.to(torch.float32) / 256.0
    return base.to(dtype).view(shape)


# ---------------------------------------------------------------------------
# clone
# ---------------------------------------------------------------------------


def test_clone_contiguous(mojo_device, call_checker):
    call_checker.register(aten_functions.aten_clone)
    x = _fill((3, 4), torch.float32)
    dev = x.to(mojo_device)
    cloned = dev.clone()
    torch.testing.assert_close(cloned.cpu(), x)
    # An independent allocation: mutating one leaves the other untouched.
    # (fill_ rather than add_: add.out is a different group's op.)
    cloned.fill_(99.0)
    torch.testing.assert_close(dev.cpu(), x)


def test_clone_strided(mojo_device, call_checker):
    call_checker.register(aten_functions.aten_clone)
    x = _fill((5, 7), torch.float32)
    dev = x.to(mojo_device).t()
    cloned = dev.clone()
    # preserve_format, and a transpose IS dense: torch keeps the strides.
    assert cloned.stride() == x.t().clone().stride()
    torch.testing.assert_close(cloned.cpu(), x.t())


@pytest.mark.parametrize(
    ("src_dtype", "dst_dtype"),
    [
        (torch.float32, torch.bfloat16),
        (torch.bfloat16, torch.float32),
        (torch.float32, torch.float16),
        (torch.int32, torch.float32),
    ],
)
@pytest.mark.parametrize("shape", [(357, 789), (0,), (17,)])
def test_copy_cast_into_offset_destination(mojo_gpu, src_dtype, dst_dtype, shape):
    source = _fill(shape, src_dtype)
    n = source.numel()
    src_storage = torch.full((n + 8,), 3, dtype=src_dtype, device=mojo_gpu)
    src = src_storage[3 : 3 + n].view(shape)
    src.copy_(source)
    before_source = src_storage.cpu()
    dst_storage = torch.full((n + 12,), -7, dtype=dst_dtype, device=mojo_gpu)
    dst = dst_storage[5 : 5 + n].view(shape)
    version = dst._version
    ptr = dst.data_ptr()
    result = dst.copy_(src)
    assert result is dst
    assert dst.data_ptr() == ptr
    assert dst._version == version + 1
    expected = torch.full((n + 12,), -7, dtype=dst_dtype)
    expected[5 : 5 + n].copy_(source.flatten())
    torch.testing.assert_close(dst_storage.cpu(), expected, rtol=0, atol=0)
    torch.testing.assert_close(src_storage.cpu(), before_source, rtol=0, atol=0)


@pytest.mark.parametrize("rank", [1, 2, 3, 4, 5])
def test_clone_every_rank(mojo_gpu, rank):
    """rank<=4 takes the PermuteCopy fast path, rank>4 the general one."""
    shape = tuple(range(2, 2 + rank))
    x = _fill(shape, torch.bfloat16)
    dev = x.to(mojo_gpu).permute(*reversed(range(rank)))
    torch.testing.assert_close(dev.clone().cpu(), x.permute(*reversed(range(rank))))


@pytest.mark.parametrize("dtype", [torch.bfloat16, torch.float16, torch.int16])
@pytest.mark.parametrize(
    "rows,cols,pitch,source_offset,destination_offset",
    [
        (2, 5120000, 15370400, 0, 0),
        (2, 3840000, 15370400, 1600, 0),
        (2, 800, 15370400, 0, 0),
        (357, 789, 811, 3, 5),
        (7, 1025, 1041, 1, 3),
        (5, 32768, 32781, 0, 0),
        (0, 17, 19, 1, 3),
        (7, 0, 9, 2, 3),
        (1, 1, 1, 0, 0),
        (2, 7, 9, 0, 0),
        (2, 8, 16, 1, 0),
        (2, 8, 16, 0, 1),
        (2, 9, 17, 3, 5),
        (3, 2049, 2056, 0, 0),
        (3, 2048, 2056, 8, 16),
        (65536, 1, 2, 1, 3),
    ],
)
def test_copy_row_strided_storage_bits(
    mojo_gpu: str,
    dtype: torch.dtype,
    rows: int,
    cols: int,
    pitch: int,
    source_offset: int,
    destination_offset: int,
):
    _check_row_copy_storage_bits(
        mojo_gpu, dtype, rows, cols, pitch, source_offset, destination_offset
    )


def _check_row_copy_storage_bits(
    mojo_gpu: str,
    dtype: torch.dtype,
    rows: int,
    cols: int,
    pitch: int,
    source_offset: int,
    destination_offset: int,
):
    # Compare storage words so NaN payloads, signed zero and subnormals survive.
    word = (
        torch.int32
        if dtype in (torch.float32, torch.int32, torch.uint32)
        else torch.int16
    )
    multiplier = 2654435761 if word == torch.int32 else 7919
    size = max(1, rows * pitch + source_offset + 7)
    bits = (torch.arange(size, dtype=torch.int64) * multiplier + 13).to(word)
    if word == torch.int32:
        special = torch.tensor(
            [
                0,
                0x80000000,
                0x7F800000,
                0xFF800000,
                0x7FC00001,
                0x7F800001,
                1,
                0x7FFFFF,
            ],
            dtype=torch.int64,
        ).to(word)
        bits[: min(8, size)] = special[: min(8, size)]
    source_base = bits.view(dtype).to(mojo_gpu)
    source = source_base.as_strided((rows, cols), (pitch, 1), source_offset)
    guard = torch.full((rows * cols + destination_offset + 7,), -535, dtype=word)
    destination_base = guard.view(dtype).to(mojo_gpu)
    destination = destination_base[
        destination_offset : destination_offset + rows * cols
    ].view(rows, cols)
    expected = guard.clone()
    expected[destination_offset : destination_offset + rows * cols].view(
        rows, cols
    ).copy_(bits.as_strided((rows, cols), (pitch, 1), source_offset))
    version, pointer = destination._version, destination.data_ptr()
    assert destination.copy_(source) is destination
    assert destination._version == version + 1
    assert destination.data_ptr() == pointer
    torch.testing.assert_close(
        destination_base.cpu().view(word), expected, rtol=0, atol=0
    )
    torch.testing.assert_close(source_base.cpu().view(word), bits, rtol=0, atol=0)


_FP32_ROW_COPY_CASES = [
    (2, 40206400, 41027200, 0, 0),
    (2, 41027200, 41027200, 0, 0),
    (2, 5120000, 15370400, 0, 0),
    (357, 789, 811, 3, 5),
    (7, 1025, 1041, 1, 3),
    (5, 32768, 32781, 0, 0),
    (0, 17, 19, 1, 3),
    (7, 0, 9, 2, 3),
    (1, 1, 1, 0, 0),
    *[
        (3, n, n + 4, 0, 0)
        for n in (
            3,
            4,
            5,
            7,
            8,
            9,
            255,
            256,
            257,
            511,
            512,
            513,
            1023,
            1024,
            1025,
            2047,
            2048,
            2049,
        )
    ],
    *[(3, 2048, 2056, src, dst) for src in range(4) for dst in range(4)],
    (65536, 1, 2, 1, 3),
    (65536, 4, 8, 0, 0),
]


@pytest.mark.parametrize(
    "rows,cols,pitch,source_offset,destination_offset", _FP32_ROW_COPY_CASES
)
def test_copy_row_strided_fp32_storage_bits(
    mojo_gpu: str,
    rows: int,
    cols: int,
    pitch: int,
    source_offset: int,
    destination_offset: int,
):
    _check_row_copy_storage_bits(
        mojo_gpu, torch.float32, rows, cols, pitch, source_offset, destination_offset
    )


@pytest.mark.parametrize("dtype", [torch.int32, torch.uint32])
@pytest.mark.parametrize("aligned", [False, True])
def test_copy_row_strided_word32_alias_dtypes(
    mojo_gpu: str, dtype: torch.dtype, aligned: bool
):
    # CopyStrided dispatches by storage width, so integer32 uses this kernel too.
    _check_row_copy_storage_bits(
        mojo_gpu,
        dtype,
        7,
        2048 if aligned else 2049,
        2056 if aligned else 2057,
        0 if aligned else 1,
        0 if aligned else 3,
    )


@pytest.mark.parametrize("dtype", [torch.bfloat16, torch.float16, torch.float32])
@pytest.mark.parametrize(
    "layout", ["leading_batch", "transpose", "strided_output", "broadcast", "alias"]
)
def test_copy_row_strided_fallback(mojo_gpu: str, dtype: torch.dtype, layout: str):
    host = _fill((2, 3, 7), dtype)
    source_base = host.to(mojo_gpu)
    if layout == "leading_batch":
        source, expected = source_base, host
    elif layout == "transpose":
        source, expected = source_base[0].t(), host[0].t()
    elif layout == "broadcast":
        source, expected = source_base[0, :1].expand(3, 7), host[0, :1].expand(3, 7)
    else:
        source, expected = source_base[0], host[0]
    if layout == "alias":
        destination = source
    elif layout == "strided_output":
        destination = torch.full((3, 14), -9, dtype=dtype, device=mojo_gpu)[:, ::2]
    else:
        destination = torch.empty(source.shape, dtype=dtype, device=mojo_gpu)
    destination.copy_(source)
    torch.testing.assert_close(destination.cpu(), expected, rtol=0, atol=0)
    torch.testing.assert_close(source_base.cpu(), host, rtol=0, atol=0)


_SPLIT_ROW_CASES = [
    (2, [800, 800, 3840000, 2400, 1280000, 800, 800, 800, 5120000, 3200, 5120000, 800]),
    (2, [800] * 12),
    (7, [357, 789, 17, 1, 0]),
    (1, [1048576]),
    (3, [513] * 65),
    (65536, [1, 0, 2]),
    (0, [17, 0, 1025]),
    (3, []),
    (3, [0, 0, 0]),
    (1, [2049] * 129),
    (3, [17] * 80),
    (3, [513] * 81),
    (3, [0 if i % 3 == 0 else i * 357 % 5001 for i in range(197)]),
]


@pytest.mark.parametrize("rows,sizes", _SPLIT_ROW_CASES)
def test_split_copy_row_cases(mojo_gpu: str, rows: int, sizes: list[int]):
    _check_split_copy_bits(mojo_gpu, rows, sizes, 3, 5)


@pytest.mark.parametrize("source_offset", range(4))
@pytest.mark.parametrize("destination_offset", range(4))
def test_split_copy_row_alignment(
    mojo_gpu: str, source_offset: int, destination_offset: int
):
    _check_split_copy_bits(
        mojo_gpu,
        3,
        [0, 1, 2, 7, 8, 9, 2047, 2048, 2049, 4095, 4096, 4097],
        source_offset,
        destination_offset,
    )


def _check_split_copy_bits(
    device: str,
    rows: int,
    sizes: list[int],
    source_offset: int,
    destination_offset: int,
    dtype: torch.dtype = torch.bfloat16,
):
    count = rows * sum(sizes)
    bits_dtype = _BITS_DTYPES[dtype.itemsize]
    bits = (torch.arange(count + source_offset + 7, dtype=torch.int64) * 7919 + 13).to(
        bits_dtype
    )
    if dtype == torch.bool:
        bits = bits % 2
    host = bits.view(dtype)
    source_base = host.to(device)
    source = source_base[source_offset : source_offset + count].view(rows, sum(sizes))
    guard_value = True if dtype == torch.bool else -9
    guards = [
        torch.full(
            (rows * n + destination_offset + 7,),
            guard_value,
            dtype=dtype,
            device=device,
        )
        for n in sizes
    ]
    outputs = [
        base[destination_offset : destination_offset + rows * n].view(rows, n)
        for base, n in zip(guards, sizes, strict=True)
    ]
    versions = [out._version for out in outputs]
    assert (
        torch.ops.aten.split_with_sizes_copy.out(source, sizes, 1, out=outputs) is None
    )
    expected = (
        host[source_offset : source_offset + count]
        .view(rows, sum(sizes))
        .split(sizes, dim=1)
    )
    for actual, want, guard, n, version in zip(
        outputs, expected, guards, sizes, versions, strict=True
    ):
        torch.testing.assert_close(
            actual.cpu().view(bits_dtype),
            want.contiguous().view(bits_dtype),
            rtol=0,
            atol=0,
        )
        observed = guard.cpu()
        assert bool((observed[:destination_offset] == guard_value).all())
        assert bool((observed[destination_offset + rows * n :] == guard_value).all())
        assert actual._version == version + 1
    torch.testing.assert_close(source_base.cpu().view(bits_dtype), bits, rtol=0, atol=0)


_BITS_DTYPES = {1: torch.uint8, 2: torch.int16, 4: torch.int32, 8: torch.int64}


# A same-dtype copy moves bits, one kernel build per element width; bool also
# checks its uint8 storage. Short rows under one tile, and rows that straddle
# it with more rectangles than one launch holds.
@pytest.mark.parametrize(
    "dtype",
    [
        torch.float16,
        torch.int16,
        torch.float32,
        torch.int32,
        torch.float64,
        torch.int64,
        torch.int8,
        torch.uint8,
        torch.bool,
    ],
)
@pytest.mark.parametrize("rows,sizes", [(3, [17] * 80), (2, [2049, 1, 0, 4096])])
def test_split_copy_row_other_dtypes(
    mojo_gpu: str, dtype: torch.dtype, rows: int, sizes: list[int]
):
    _check_split_copy_bits(mojo_gpu, rows, sizes, 3, 5, dtype)


@pytest.mark.parametrize(
    "shape,dim,sizes",
    [
        ((2, 5, 3), 1, [2, 0, 3]),
        ((2, 5, 3), -2, [1, 4]),
        ((2, 3, 4), 0, [1, 1]),
        ((2, 3, 4), -1, [1, 3]),
    ],
)
@pytest.mark.parametrize("dtype", [torch.bfloat16, torch.float32])
def test_split_copy_dimensions(
    mojo_gpu: str,
    shape: tuple[int, ...],
    dim: int,
    sizes: list[int],
    dtype: torch.dtype,
):
    host = _fill(shape, dtype)
    expected = list(host.split(sizes, dim))
    outputs = [torch.empty(t.shape, dtype=dtype, device=mojo_gpu) for t in expected]
    torch.ops.aten.split_with_sizes_copy.out(host.to(mojo_gpu), sizes, dim, out=outputs)
    for actual, want in zip(outputs, expected, strict=True):
        torch.testing.assert_close(actual.cpu(), want, rtol=0, atol=0)


@pytest.mark.parametrize(
    "mode",
    [
        "strided_input",
        "strided_output",
        "resize",
        "resize_empty",
        "duplicate",
        "source_alias",
        "source_dependency",
    ],
)
def test_split_copy_fallback(mojo_device: str, mode: str):
    def operands(device: str) -> tuple[torch.Tensor, list[torch.Tensor]]:
        source = torch.arange(12, dtype=torch.bfloat16).view(3, 4).to(device)
        outputs = [
            torch.empty((3, 2), dtype=torch.bfloat16, device=device) for _ in range(2)
        ]
        if mode == "strided_input":
            source = (
                torch.arange(24, dtype=torch.bfloat16).view(3, 8).to(device)[:, ::2]
            )
        elif mode == "strided_output":
            outputs = [
                torch.zeros((3, 4), dtype=torch.bfloat16, device=device)[:, ::2]
                for _ in range(2)
            ]
        elif mode == "resize":
            outputs = [
                torch.empty(1, dtype=torch.bfloat16, device=device) for _ in range(2)
            ]
        elif mode == "resize_empty":
            outputs = [
                torch.empty(0, dtype=torch.bfloat16, device=device) for _ in range(2)
            ]
        elif mode == "duplicate":
            outputs[1] = outputs[0]
        elif mode == "source_alias":
            outputs = list(source.split([2, 2], dim=1))
        elif mode == "source_dependency":
            outputs[0] = source[:, 2:]
        return source, outputs

    host, reference = operands("cpu")
    source, outputs = operands(mojo_device)
    ref_versions, versions = (
        [t._version for t in reference],
        [t._version for t in outputs],
    )
    torch.ops.aten.split_with_sizes_copy.out(host, [2, 2], 1, out=reference)
    torch.ops.aten.split_with_sizes_copy.out(source, [2, 2], 1, out=outputs)
    for actual, want, before, ref_before in zip(
        outputs, reference, versions, ref_versions, strict=True
    ):
        torch.testing.assert_close(actual.cpu(), want, rtol=0, atol=0)
        assert actual._version - before == want._version - ref_before
    torch.testing.assert_close(source.cpu(), host, rtol=0, atol=0)


def _split_copy_error_operands(
    mode: str, device: str
) -> tuple[torch.Tensor, list[int], int, list[torch.Tensor]]:
    source = torch.ones((2, 4), dtype=torch.bfloat16, device=device)
    sizes, dim = [2, 2], 1
    outputs = [
        torch.empty((2, 2), dtype=torch.bfloat16, device=device) for _ in range(2)
    ]
    if mode == "negative_size":
        sizes = [-1, 5]
    elif mode == "sum":
        sizes = [1, 2]
    elif mode == "dimension":
        dim = 2
    elif mode == "negative_dimension":
        dim = -3
    elif mode == "scalar":
        source = source[0, 0]
    elif mode == "output_count":
        outputs.pop()
    elif mode == "dtype":
        outputs[1] = torch.empty((2, 2), dtype=torch.float32, device=device)
    return source, sizes, dim, outputs


def _message(error: pytest.ExceptionInfo[BaseException]) -> str:
    """The first line, without the ` [aten::<op>]` the shim appends to every
    error a mojo op raises (`shim_dispatch.cpp`)."""
    return re.sub(r" \[aten::[^]]+\]$", "", str(error.value).splitlines()[0])


# Registering the op retires ATen's composite kernel, so these messages are no
# longer ATen's by construction -- they are ours, and a user reads them instead
# of the ones CPU torch prints. Compare the text against CPU rather than only
# the exception type, or a message can silently lose the sizes or the dim.
@pytest.mark.parametrize(
    "mode",
    [
        "negative_size",
        "sum",
        "dimension",
        "negative_dimension",
        "scalar",
        "output_count",
        "dtype",
    ],
)
def test_split_copy_errors_read_like_cpu(mojo_gpu: str, mode: str):
    # ATen raises the dim check with TORCH_CHECK_INDEX, which reaches python as
    # IndexError; every error out of a mojo op is a RuntimeError, so the type
    # differs for that one case and only the text is compared.
    messages = []
    for device in ("cpu", mojo_gpu):
        source, sizes, dim, outputs = _split_copy_error_operands(mode, device)
        with pytest.raises((RuntimeError, IndexError)) as error:
            torch.ops.aten.split_with_sizes_copy.out(source, sizes, dim, out=outputs)
        messages.append(_message(error))
    assert messages[1] == messages[0]


def test_split_copy_error_names_the_wrong_device(mojo_gpu: str):
    source = torch.ones((2, 4), dtype=torch.bfloat16, device=mojo_gpu)
    outputs = [
        torch.empty((2, 2), dtype=torch.bfloat16, device=mojo_gpu),
        torch.empty((2, 2), dtype=torch.bfloat16),
    ]
    with pytest.raises(RuntimeError) as error:
        torch.ops.aten.split_with_sizes_copy.out(source, [2, 2], 1, out=outputs)
    assert _message(error) == (
        f"Expected out tensor to have device {source.device}, but got cpu instead"
    )


@pytest.fixture(params=[(0, 1), (1, 0)], ids=["0-to-1", "1-to-0"])
def mojo_pair(request: pytest.FixtureRequest) -> tuple[str, str]:
    if len(get_accelerators()) < 2:
        pytest.skip("requires two mojo GPUs")
    src, dst = request.param
    return f"mojo:{src}", f"mojo:{dst}"


@pytest.mark.parametrize(
    "dtype", [torch.float32, torch.float16, torch.bfloat16, torch.int64, torch.bool]
)
@pytest.mark.parametrize("shape", [(357, 789), (), (0, 3)])
def test_cross_device_to(
    mojo_pair: tuple[str, str], dtype: torch.dtype, shape: tuple[int, ...]
):
    src, dst = mojo_pair
    expected = _fill(shape, dtype)
    actual = expected.to(src).to(dst)
    assert str(actual.device) == dst
    torch.testing.assert_close(actual.cpu(), expected)


@pytest.mark.parametrize("layout", ["transpose", "channels_last", "slice", "offset"])
@pytest.mark.parametrize("dtype", [torch.float32, torch.float16])
@pytest.mark.parametrize(
    "memory_format",
    [torch.preserve_format, torch.contiguous_format, torch.channels_last],
)
def test_cross_device_to_layout(
    mojo_pair: tuple[str, str],
    layout: str,
    dtype: torch.dtype,
    memory_format: torch.memory_format,
):
    src, dst = mojo_pair
    cpu = _fill((3, 5, 7, 11), torch.float32)
    gpu = cpu.to(src)
    if layout == "transpose":
        cpu, gpu = cpu.transpose(1, 3), gpu.transpose(1, 3)
    elif layout == "channels_last":
        cpu = cpu.contiguous(memory_format=torch.channels_last)
        gpu = gpu.contiguous(memory_format=torch.channels_last)
    elif layout == "slice":
        cpu, gpu = cpu[:, :, 1::2, 1::2], gpu[:, :, 1::2, 1::2]
    else:
        cpu, gpu = cpu[1:], gpu[1:]
    expected = cpu.to(dtype=dtype, memory_format=memory_format, copy=True)
    actual = gpu.to(dst, dtype=dtype, memory_format=memory_format)
    assert actual.stride() == expected.stride()
    torch.testing.assert_close(actual.cpu(), expected)


@pytest.mark.parametrize("strided", [False, True])
@pytest.mark.parametrize(
    "dtype", [torch.float32, torch.float16, torch.int64, torch.bool]
)
@pytest.mark.parametrize("reshape", [False, True])
def test_cross_device_copy(
    mojo_pair: tuple[str, str], strided: bool, dtype: torch.dtype, reshape: bool
):
    src, dst = mojo_pair
    cpu = _fill((19, 37), torch.float32).t()
    source = _fill((19, 37), torch.float32).to(src).t()
    host_base = torch.full((37, 38), -1, dtype=dtype)
    device_base = host_base.to(dst)
    expected = host_base[:, ::2] if strided else host_base[:, :19].contiguous()
    actual = device_base[:, ::2] if strided else device_base[:, :19].contiguous()
    if reshape:
        expected, actual = expected.unsqueeze(0), actual.unsqueeze(0)
    expected.copy_(cpu)
    result = actual.copy_(source)
    assert result is actual
    torch.testing.assert_close(actual.cpu(), expected)
    if strided:
        torch.testing.assert_close(device_base.cpu(), host_base)


@pytest.mark.parametrize(
    "src_dtype,dst_dtype",
    [
        (torch.float64, torch.float32),
        (torch.float32, torch.float64),
        (torch.int16, torch.int64),
        (torch.int64, torch.uint64),
        (torch.uint64, torch.int64),
        (torch.int8, torch.float32),
        (torch.float32, torch.int16),
        (torch.uint32, torch.int64),
    ],
)
@pytest.mark.parametrize("strided", [False, True])
def test_cross_device_copy_other_dtypes(
    mojo_pair: tuple[str, str],
    src_dtype: torch.dtype,
    dst_dtype: torch.dtype,
    strided: bool,
):
    src, dst = mojo_pair
    values = torch.tensor([2**60 + 3, 2**60 + 5, 0, 1, 19, 251]).reshape(2, 3)
    host = values.to(src_dtype)
    expected = torch.empty((2, 3), dtype=dst_dtype).copy_(host)
    actual = torch.empty((2, 6) if strided else (2, 3), dtype=dst_dtype, device=dst)
    if strided:
        actual = actual[:, ::2]
    actual.copy_(host.to(src))
    torch.testing.assert_close(actual.cpu(), expected, rtol=0, atol=0)


@pytest.mark.parametrize("copy_into", [False, True])
@pytest.mark.parametrize("non_blocking", [False, True])
def test_cross_device_stream_lifetime(
    mojo_pair: tuple[str, str], copy_into: bool, non_blocking: bool
):
    src, dst = mojo_pair
    n = 64 * 1024 * 1024  # 256 MiB; queued producers outlast host dispatch.
    source_stream = torch.Stream(device=src)
    destination_stream = torch.Stream(device=dst)
    with source_stream:
        source = torch.full((n,), 1.0, device=src)
        for _ in range(8):
            source.add_(1.0)
        with destination_stream:
            if copy_into:
                actual = torch.full((n,), -3.0, device=dst)
                actual.copy_(source, non_blocking=non_blocking)
            else:
                actual = source.to(dst, non_blocking=non_blocking)
            del source
            # Reuse source-side allocations while the destination consumes.
            scratch = torch.empty((n,), device=src)
            scratch.fill_(-11.0)
            actual.add_(1.0)
            torch.testing.assert_close(actual.cpu(), torch.full((n,), 10.0))
    source_stream.synchronize()
    destination_stream.synchronize()


def test_cross_device_autograd(mojo_pair: tuple[str, str]):
    src, dst = mojo_pair
    x = _fill((13, 7), torch.float32).requires_grad_()
    w1 = _fill((7, 11), torch.float32).requires_grad_()
    w2 = _fill((11, 5), torch.float32).requires_grad_()
    gx = x.detach().to(src).requires_grad_()
    gw1 = w1.detach().to(src).requires_grad_()
    gw2 = w2.detach().to(dst).requires_grad_()
    expected = ((x @ w1).relu() @ w2).square().mean()
    actual = ((gx @ gw1).relu().to(dst) @ gw2).square().mean()
    expected.backward()
    actual.backward()
    torch.testing.assert_close(actual.cpu(), expected.detach())
    for got, ref in [(gx, x), (gw1, w1), (gw2, w2)]:
        assert got.grad is not None and ref.grad is not None
        torch.testing.assert_close(got.grad.cpu(), ref.grad, rtol=2e-4, atol=2e-4)


@pytest.mark.parametrize("same_device", [False, True])
@pytest.mark.parametrize("operation", ["to", "copy"])
@pytest.mark.parametrize(
    "source_dtype,target_dtype",
    [(torch.int64, torch.uint64), (torch.uint64, torch.int64)],
)
def test_cross_device_to_exact_integer_cast(
    mojo_pair: tuple[str, str],
    same_device: bool,
    operation: str,
    source_dtype: torch.dtype,
    target_dtype: torch.dtype,
):
    src, dst = mojo_pair
    cpu = torch.tensor([2**60 + 3, 2**60 + 5, 2**53 + 1, 0, 19], dtype=source_dtype)
    source = cpu.to(src)
    destination = src if same_device else dst
    if operation == "to":
        actual = source.to(destination, dtype=target_dtype)
    else:
        actual = torch.empty(cpu.shape, device=destination, dtype=target_dtype)
        actual.copy_(source)
    torch.testing.assert_close(actual.cpu(), cpu.to(target_dtype), rtol=0, atol=0)


@pytest.mark.parametrize("layout", ["contiguous", "transpose", "cast"])
@pytest.mark.parametrize("copy_into", [False, True])
@pytest.mark.parametrize("non_blocking", [False, True])
def test_cross_device_allocation_stream(
    mojo_pair: tuple[str, str], layout: str, copy_into: bool, non_blocking: bool
):
    src, dst = mojo_pair
    owner_src, owner_dst = torch.Stream(device=src), torch.Stream(device=dst)
    current_src, current_dst = torch.Stream(device=src), torch.Stream(device=dst)
    cpu = _fill((1021, 4093), torch.float32)
    with owner_src:
        source = cpu.to(src)
    with owner_dst:
        actual = torch.empty_like(cpu, device=dst)
    current_src.wait_stream(owner_src)
    current_dst.wait_stream(owner_dst)
    with current_src, current_dst:
        source.add_(1)
        expected = cpu + 1
        if layout == "transpose":
            source, expected, actual = source.t(), expected.t(), actual.t()
        dtype = torch.float16 if layout == "cast" else torch.float32
        if copy_into:
            if dtype != actual.dtype:
                with owner_dst:
                    actual = actual.to(dtype=dtype)
                current_dst.wait_stream(owner_dst)
            actual.copy_(source, non_blocking=non_blocking)
        else:
            actual = source.to(dst, dtype=dtype, non_blocking=non_blocking)
        del source
        # Churn on the allocation streams, before either current stream drains.
        with owner_src:
            torch.empty_like(cpu, device=src).fill_(-111)
        if copy_into:
            # Immediately free destination storage after enqueueing a consumer.
            consumed = actual + 2
            del actual
            with owner_dst:
                torch.empty_like(cpu, device=dst, dtype=dtype).fill_(-222)
            actual = consumed
            expected = expected.to(dtype) + 2
        else:
            expected = expected.to(dtype)
        torch.testing.assert_close(actual.cpu(), expected, rtol=0, atol=0)
        assert (
            torch.tensor(19, device=src).to(dst, non_blocking=non_blocking).item() == 19
        )
    owner_src.synchronize()
    owner_dst.synchronize()


def _peer_autograd_stress(src: str, dst: str, iterations: int):
    a, b = torch.Stream(device=src), torch.Stream(device=dst)
    with a, b:
        for _ in range(iterations):
            x = torch.full((257, 31), 0.25, device=src, requires_grad=True)
            y = (x * 2).to(dst)
            y.square().sum().backward()
            assert x.grad is not None
            torch.testing.assert_close(x.grad.cpu(), torch.full((257, 31), 2.0))


def test_cross_device_interleaved_stress(mojo_pair: tuple[str, str]):
    src, dst = mojo_pair
    owners = [torch.Stream(device=d) for d in (src, dst)]
    streams = [torch.Stream(device=d) for d in (src, dst)]
    # 16 MiB per transfer, delayed checks and owner-stream churn force real reuse.
    shape = (1021, 4093)
    pending = []
    with ThreadPoolExecutor(max_workers=1) as pool:
        worker = pool.submit(_peer_autograd_stress, src, dst, 100)
        for i in range(300):
            direction = i % 2
            source_device, dest_device = (src, dst) if direction == 0 else (dst, src)
            owner, target_owner = owners[direction], owners[1 - direction]
            current, target_current = streams[direction], streams[1 - direction]
            with owner:
                source = torch.full(shape, float(i % 19), device=source_device)
            with target_owner:
                target = torch.empty(shape, device=dest_device)
            current.wait_stream(owner)
            target_current.wait_stream(target_owner)
            with current, target_current:
                source.add_(1)
                if i % 3 == 0:
                    source, target = source.t(), target.t()
                if i % 4 < 2:
                    target.copy_(source, non_blocking=True)
                else:
                    target = source.to(dest_device, non_blocking=True)
                del source
                result = target + 2
                del target
                pending.append((result, float(i % 19 + 3), target_current))
            with owner:
                torch.empty(shape, device=source_device).fill_(-999)
            with target_owner:
                torch.empty(shape, device=dest_device).fill_(-888)
            if len(pending) == 16:
                for result, value, stream in pending:
                    with stream:
                        torch.testing.assert_close(
                            result.cpu(), torch.full(result.shape, value)
                        )
                pending.clear()
        for result, value, stream in pending:
            with stream:
                torch.testing.assert_close(
                    result.cpu(), torch.full(result.shape, value)
                )
        worker.result()
    for stream in owners + streams:
        stream.synchronize()


@pytest.mark.parametrize("mode", [None, "trace", "host", "enable_error"])
def test_cross_device_route(mojo_pair: tuple[str, str], mode: str | None):
    # The test hook and peer cache are per backend.
    code = r"""
import ctypes
import os
import sys
import torch
from torch_mojo_backend import register_mojo_devices
register_mojo_devices()
# Independent capability query: losing the backend's direct route must fail.
try:
    driver = ctypes.CDLL("libcuda.so.1")
    assert driver.cuInit(0) == 0
    capable = ctypes.c_int()
    for dst, src in [(1, 0), (0, 1)]:
        assert driver.cuDeviceCanAccessPeer(ctypes.byref(capable), dst, src) == 0
        if not capable.value:
            sys.exit(77)
except OSError:
    sys.exit(77)
if not capable.value:
    sys.exit(77)
mode = os.environ.get("TORCH_MOJO_BACKEND_TEST_PEER_COPY", "")
os.environ["TORCH_MOJO_BACKEND_TEST_PEER_COPY"] = "host" if mode != "host" else "trace"
for src, dst in [("mojo:0", "mojo:1"), ("mojo:1", "mojo:0")]:
    cpu = torch.arange(357 * 79, dtype=torch.float32).reshape(357, 79)
    source = cpu.to(src)
    for _ in range(3):
        torch.testing.assert_close(source.to(dst).cpu(), cpu)
        target = torch.empty((357, 158), device=dst)[:, ::2]
        target.copy_(source)
        torch.testing.assert_close(target.cpu(), cpu)
"""
    env = dict(os.environ)
    env.pop("TORCH_MOJO_BACKEND_TEST_PEER_COPY", None)
    if mode is not None:
        env["TORCH_MOJO_BACKEND_TEST_PEER_COPY"] = mode
    result = subprocess.run(
        [sys.executable, "-c", code],
        cwd=Path(__file__).resolve().parents[2],
        env=env,
        capture_output=True,
        text=True,
        timeout=600,
    )
    if result.returncode == 77:
        pytest.skip("route assertion requires a CUDA peer-capable pair")
    assert result.returncode == 0, result.stdout + result.stderr
    output = result.stdout
    if mode is None:
        assert "peer " not in output and "P2P_" not in output, output
        return
    assert output.count("peer probe") == 2, output
    assert output.count("peer enable failed") == (2 if mode == "enable_error" else 0), (
        output
    )
    route = "direct" if mode == "trace" else "host"
    assert output.count(f"peer copy {route}") == 12, output
    assert f"peer copy {'host' if route == 'direct' else 'direct'}" not in output


# ---------------------------------------------------------------------------
# memory formats (empty.memory_format / clone / _to_copy)
# ---------------------------------------------------------------------------

# (shape, memory format): rank 4 for channels_last, rank 5 for
# channels_last_3d, plus the degenerate extents whose strides torch still
# spells out in full (a size-1 channel, an empty batch).
_MEMORY_FORMATS = [
    ((2, 3, 4, 5), torch.channels_last),
    ((3, 7, 5, 11), torch.channels_last),
    ((2, 1, 4, 5), torch.channels_last),
    ((0, 3, 4, 5), torch.channels_last),
    ((2, 3, 4, 5, 2), torch.channels_last_3d),
    ((2, 5, 3, 7, 3), torch.channels_last_3d),
]


def _memory_format_id(val):
    if isinstance(val, torch.memory_format):
        return str(val).removeprefix("torch.")
    return "x".join(str(d) for d in val)


@pytest.mark.parametrize("shape,memory_format", _MEMORY_FORMATS, ids=_memory_format_id)
def test_empty_memory_format_strides_match_cpu(mojo_device, shape, memory_format):
    made = torch.empty(shape, device=mojo_device, memory_format=memory_format)
    assert made.stride() == torch.empty(shape, memory_format=memory_format).stride()
    assert made.is_contiguous(memory_format=memory_format)


@pytest.mark.parametrize(
    "factory", [torch.empty_like, torch.zeros_like, torch.ones_like, torch.rand_like]
)
def test_like_factories_keep_the_memory_format(mojo_device, factory):
    """ATen's `*_like` composites resolve the format themselves and land on
    `empty.memory_format` / `empty_strided`."""
    x = torch.arange(120.0).reshape(2, 3, 4, 5).to(memory_format=torch.channels_last)
    dev = x.to(mojo_device)
    assert factory(dev).stride() == factory(x).stride()
    for memory_format in (
        torch.preserve_format,
        torch.contiguous_format,
        torch.channels_last,
    ):
        got = factory(dev, memory_format=memory_format)
        assert got.stride() == factory(x, memory_format=memory_format).stride()


@pytest.mark.parametrize("shape,memory_format", _MEMORY_FORMATS, ids=_memory_format_id)
def test_clone_memory_format_matches_cpu(
    mojo_device, shape, memory_format, call_checker
):
    call_checker.register(aten_functions.aten_clone)
    x = _fill(shape, torch.float32)
    expected = x.clone(memory_format=memory_format)
    got = x.to(mojo_device).clone(memory_format=memory_format)
    assert got.stride() == expected.stride()
    assert got.is_contiguous(memory_format=memory_format)
    torch.testing.assert_close(got.cpu(), expected)


@pytest.mark.parametrize("shape,memory_format", _MEMORY_FORMATS, ids=_memory_format_id)
def test_to_memory_format_matches_cpu(mojo_device, shape, memory_format, call_checker):
    call_checker.register(aten_functions.aten__to_copy)
    x = _fill(shape, torch.float32)
    expected = x.to(memory_format=memory_format)
    got = x.to(mojo_device).to(memory_format=memory_format)
    assert got.stride() == expected.stride()
    torch.testing.assert_close(got.cpu(), expected)


@pytest.mark.parametrize("shape,memory_format", _MEMORY_FORMATS, ids=_memory_format_id)
def test_contiguous_memory_format_matches_cpu(mojo_device, shape, memory_format):
    """`Tensor.contiguous(memory_format=...)` is a composite torch lowers to
    `clone(memory_format)` -- but only when the tensor is not already in that
    format, so it also covers the no-copy answer."""
    x = _fill(shape, torch.float32)
    dev = x.to(mojo_device)
    for tensor, reference in (
        (dev, x),
        (dev.contiguous(memory_format=memory_format), x),
    ):
        got = tensor.contiguous(memory_format=memory_format)
        expected = reference.contiguous(memory_format=memory_format)
        assert got.stride() == expected.stride()
        torch.testing.assert_close(got.cpu(), expected)


@pytest.mark.parametrize("dtype", [torch.float16, torch.bfloat16, torch.int64])
def test_to_dtype_and_memory_format_matches_cpu(mojo_device, dtype):
    x = _fill((2, 3, 4, 5), torch.float32)
    expected = x.to(dtype, memory_format=torch.channels_last)
    got = x.to(mojo_device).to(dtype, memory_format=torch.channels_last)
    assert got.stride() == expected.stride()
    assert got.is_contiguous(memory_format=torch.channels_last)
    torch.testing.assert_close(got.cpu(), expected)


def test_memory_format_from_a_permuted_input(mojo_device):
    """A non-contiguous source: the strided read has to feed the strided
    write, neither of them row-major."""
    x = _fill((2, 3, 4, 5), torch.float32).permute(0, 2, 3, 1)
    dev = _fill((2, 3, 4, 5), torch.float32).to(mojo_device).permute(0, 2, 3, 1)
    for expected, got in (
        (
            x.clone(memory_format=torch.channels_last),
            dev.clone(memory_format=torch.channels_last),
        ),
        (
            x.to(torch.float16, memory_format=torch.channels_last),
            dev.to(torch.float16, memory_format=torch.channels_last),
        ),
        (x.contiguous(), dev.contiguous()),
    ):
        assert got.stride() == expected.stride()
        torch.testing.assert_close(got.cpu(), expected)


def test_preserve_format_keeps_the_input_layout(mojo_device):
    """torch's preserve_format: a non-overlapping-and-dense input keeps its
    strides exactly, anything else gets `infer_dense_strides`."""
    x = _fill((2, 3, 4, 5), torch.float32)
    dev = x.to(mojo_device)
    cases = [
        (
            x.to(memory_format=torch.channels_last),
            dev.to(memory_format=torch.channels_last),
        ),
        (x.permute(0, 2, 3, 1), dev.permute(0, 2, 3, 1)),
        # Not dense: a strided slice of a transpose.
        (x.reshape(24, 5).t()[:, ::2], dev.reshape(24, 5).t()[:, ::2]),
    ]
    for expected_src, got_src in cases:
        for expected, got in (
            (expected_src.clone(), got_src.clone()),  # clone
            (expected_src.half(), got_src.half()),  # _to_copy
        ):
            assert got.stride() == expected.stride()
            torch.testing.assert_close(got.cpu(), expected)


def test_channels_last_survives_a_device_round_trip(mojo_device):
    x = _fill((2, 3, 4, 5), torch.float32).to(memory_format=torch.channels_last)
    dev = x.to(mojo_device)
    assert dev.stride() == x.stride()
    back = dev.cpu()
    assert back.stride() == x.stride()
    assert torch.equal(back, x)


@pytest.mark.gpu
def test_channels_last_survives_a_move_between_mojo_devices():
    """The cross-device leg stages a contiguous buffer and lays the layout out
    again on the destination."""
    if len(list(get_accelerators())) < 2:
        pytest.skip("needs two mojo devices")
    x = _fill((2, 3, 4, 5), torch.float32).to(memory_format=torch.channels_last)
    moved = x.to("mojo:0").to("mojo:1")
    assert moved.stride() == x.stride()
    torch.testing.assert_close(moved.cpu(), x)


@pytest.mark.parametrize(
    "shape,memory_format",
    [((2, 3, 4), torch.channels_last), ((2, 3, 4, 5), torch.channels_last_3d)],
    ids=_memory_format_id,
)
def test_memory_format_rank_is_checked_like_torch(mojo_device, shape, memory_format):
    x = _fill(shape, torch.float32)
    dev = x.to(mojo_device)
    calls = [
        lambda t: torch.empty(shape, device=t.device, memory_format=memory_format),
        lambda t: t.clone(memory_format=memory_format),
        lambda t: t.to(torch.float16, memory_format=memory_format),
        lambda t: t.contiguous(memory_format=memory_format),
    ]
    for call in calls:
        with pytest.raises(RuntimeError) as cpu_error:
            call(x)
        with pytest.raises(RuntimeError) as mojo_error:
            call(dev)
        assert str(cpu_error.value) in str(mojo_error.value)


def test_empty_rejects_preserve_format_like_torch(mojo_device):
    with pytest.raises(RuntimeError) as cpu_error:
        torch.empty(3, memory_format=torch.preserve_format)
    with pytest.raises(RuntimeError) as mojo_error:
        torch.empty(3, device=mojo_device, memory_format=torch.preserve_format)
    assert str(cpu_error.value) in str(mojo_error.value)


# ---------------------------------------------------------------------------
# _to_copy: dtype casts (same device) and device moves
# ---------------------------------------------------------------------------


@pytest.mark.parametrize(
    "src_dtype,dst_dtype",
    [
        (torch.float32, torch.bfloat16),
        (torch.bfloat16, torch.float16),
        (torch.float32, torch.int64),
        (torch.int64, torch.int32),
        (torch.int32, torch.uint8),
        (torch.uint8, torch.bool),
        # Exotic pairs outside the fast CastSpec kernel: the host round trip.
        (torch.float32, torch.float64),
        (torch.float64, torch.float32),
        (torch.float32, torch.int16),
        (torch.int16, torch.int8),
        (torch.uint8, torch.uint16),
    ],
)
def test_to_copy_dtype_cast(mojo_device, src_dtype, dst_dtype, call_checker):
    call_checker.register(aten_functions.aten__to_copy)
    x = _fill((3, 5), src_dtype if not src_dtype.is_floating_point else torch.float32)
    x = x.to(src_dtype)
    dev = x.to(mojo_device)
    torch.testing.assert_close(dev.to(dst_dtype).cpu(), x.to(dst_dtype))


def test_to_copy_always_returns_a_fresh_tensor(mojo_gpu):
    # `Tensor.to(dtype)` short-circuits to `self` in Python when nothing
    # would change, without ever reaching `_to_copy`; call the aten op
    # directly to exercise its own "always a fresh tensor" contract.
    x = torch.randn(4).to(mojo_gpu)
    same = torch.ops.aten._to_copy.default(x, dtype=torch.float32)
    assert same.data_ptr() != x.data_ptr()


def test_to_copy_device_round_trip(mojo_device):
    x = _fill((4, 6), torch.float32)
    dev = x.to(mojo_device)
    back = dev.to("cpu")
    torch.testing.assert_close(back, x)


# ---------------------------------------------------------------------------
# cat
# ---------------------------------------------------------------------------


def test_cat_skips_legacy_empty(mojo_device, call_checker):
    call_checker.register(aten_functions.aten_cat)
    empty = torch.empty(0)
    x = torch.randn(1, 12, 6, 8)
    result = torch.cat([empty.to(mojo_device), x.to(mojo_device)], dim=-2)
    torch.testing.assert_close(result.cpu(), torch.cat([empty, x], dim=-2))


_CAT_CASES = [
    ("single input", [(1000,)], 0),
    ("two aligned", [(4096,), (4096,)], 0),
    ("three", [(777,), (777,), (777,)], 0),
    ("past one batch", [(311,)] * 70, 0),
    ("wildly unequal", [(1,), (7,), (4096,), (3,), (10000,)], 0),
    ("odd lengths", [(12345,), (7,), (999,)], 0),
    ("zero along dim", [(0, 5), (3, 5)], 0),
    ("3-D middle dim", [(5, 2, 7), (5, 3, 7), (5, 4, 7)], 1),
    ("3-D trailing dim", [(5, 6, 2), (5, 6, 3), (5, 6, 4)], 2),
    ("3-D negative dim", [(5, 6, 2), (5, 6, 3)], -1),
]


@pytest.mark.parametrize(
    "shapes,dim",
    [case[1:] for case in _CAT_CASES],
    ids=[case[0] for case in _CAT_CASES],
)
@pytest.mark.parametrize("dtype", [torch.float32, torch.bfloat16, torch.int64])
def test_cat_batched(mojo_device, shapes, dim, dtype):
    host = [torch.randn(shape).to(dtype) for shape in shapes]
    device = [x.to(mojo_device) for x in host]
    torch.testing.assert_close(
        torch.cat(device, dim).cpu(), torch.cat(host, dim), rtol=0, atol=0
    )


def test_cat_strided_inputs(mojo_gpu):
    """A non-contiguous input takes the per-input strided-view path (the
    batched kernel only ever sees contiguous inputs)."""
    host = [torch.randn(64, 32), torch.randn(64, 32)]
    device = [x.to(mojo_gpu).t() for x in host]
    torch.testing.assert_close(
        torch.cat(device, 0).cpu(), torch.cat([x.t() for x in host], 0), rtol=0, atol=0
    )


def test_cat_offset_views_and_legacy_empty(mojo_gpu):
    host = [torch.randn(2048) for _ in range(4)]
    device = [x.to(mojo_gpu) for x in host]
    torch.testing.assert_close(
        torch.cat([x[3:1000] for x in device]).cpu(),
        torch.cat([x[3:1000] for x in host]),
        rtol=0,
        atol=0,
    )
    empty = torch.empty(0)
    mid = [torch.randn(4, 8), empty, torch.randn(3, 8)]
    torch.testing.assert_close(
        torch.cat([x.to(mojo_gpu) for x in mid], 0).cpu(),
        torch.cat(mid, 0),
        rtol=0,
        atol=0,
    )


# ---------------------------------------------------------------------------
# stack
# ---------------------------------------------------------------------------


@pytest.mark.parametrize("dim", [0, 1, -1])
def test_stack(mojo_device, dim, call_checker):
    call_checker.register(aten_functions.aten_stack)
    host = [torch.randn(6, 5) for _ in range(5)]
    device = [x.to(mojo_device) for x in host]
    torch.testing.assert_close(
        torch.stack(device, dim).cpu(), torch.stack(host, dim), rtol=0, atol=0
    )


def test_stack_many_inputs(mojo_gpu):
    host = [torch.randn(1024) for _ in range(33)]
    device = [x.to(mojo_gpu) for x in host]
    torch.testing.assert_close(
        torch.stack(device, 0).cpu(), torch.stack(host, 0), rtol=0, atol=0
    )


# ---------------------------------------------------------------------------
# repeat
# ---------------------------------------------------------------------------

_REPEAT_CASES = [
    (357, 789, (2, 3)),
    (13, 7, (3, 5)),
    (64, 1024, (3, 2)),
    (1024, 64, (1, 16)),
    (33, 33, (2, 2)),
    (1, 100, (5, 7)),
    (17, 31, (1, 1)),
    (3, 4, (2, 3, 5)),
    (5, 6, (3, 1, 1)),
    (1, 3, (2000, 1)),
]
_REPEAT_IDS = [
    f"{r}x{c}_r{'x'.join(str(k) for k in reps)}" for r, c, reps in _REPEAT_CASES
]


@pytest.mark.parametrize("rows,cols,reps", _REPEAT_CASES, ids=_REPEAT_IDS)
def test_repeat_matches_torch(mojo_device, rows, cols, reps, call_checker):
    call_checker.register(aten_functions.aten_repeat)
    x = _fill((rows, cols), torch.float32)
    torch.testing.assert_close(
        x.to(mojo_device).repeat(*reps).cpu(), x.repeat(*reps), rtol=0, atol=0
    )


@pytest.mark.parametrize(
    "dtype", [torch.float32, torch.bfloat16, torch.int64, torch.uint8]
)
@pytest.mark.parametrize("rows,cols,reps", [(357, 789, (2, 3)), (100, 8, (1, 5))])
def test_repeat_every_element_size(mojo_gpu, dtype, rows, cols, reps):
    x = _fill((rows, cols), dtype)
    torch.testing.assert_close(
        x.to(mojo_gpu).repeat(*reps).cpu(), x.repeat(*reps), rtol=0, atol=0
    )


@pytest.mark.parametrize("offset", [1, 2, 3, 4])
def test_repeat_offset_views_are_not_assumed_aligned(mojo_gpu, offset):
    rows, cols, reps = 357, 789, (2, 3)
    base = _fill((rows * cols + 4,), torch.float32)
    x = base[offset : offset + rows * cols].view(rows, cols)
    device = base.to(mojo_gpu)[offset : offset + rows * cols].view(rows, cols)
    torch.testing.assert_close(
        device.repeat(*reps).cpu(), x.repeat(*reps), rtol=0, atol=0
    )


def test_repeat_degenerate_extents(mojo_device):
    """A zero repeat factor or a zero input extent is an empty output, not a
    launch and not a crash. `repeat(x, [])` on a 0-d tensor is a legal 0-d
    copy with no last dim to tile along."""
    for shape, reps in (((4, 5), (0, 2)), ((4, 5), (2, 0)), ((0, 5), (2, 3))):
        x = _fill(shape, torch.float32)
        torch.testing.assert_close(
            x.to(mojo_device).repeat(*reps).cpu(), x.repeat(*reps), rtol=0, atol=0
        )
    scalar = torch.tensor(1.25)
    torch.testing.assert_close(
        torch.ops.aten.repeat(scalar.to(mojo_device), []).cpu(),
        torch.ops.aten.repeat(scalar, []),
        rtol=0,
        atol=0,
    )


# ---------------------------------------------------------------------------
# tril / triu
# ---------------------------------------------------------------------------


@pytest.mark.parametrize("diagonal", [-2, -1, 0, 1, 2])
@pytest.mark.parametrize("shape", [(5, 5), (357, 41), (2, 6, 4)])
def test_tril(mojo_device, shape, diagonal, call_checker):
    call_checker.register(aten_functions.aten_tril)
    x = _fill(shape, torch.float32)
    torch.testing.assert_close(x.to(mojo_device).tril(diagonal).cpu(), x.tril(diagonal))


@pytest.mark.parametrize("diagonal", [-2, -1, 0, 1, 2])
@pytest.mark.parametrize("shape", [(5, 5), (41, 357), (2, 6, 4)])
def test_triu(mojo_device, shape, diagonal, call_checker):
    call_checker.register(aten_functions.aten_triu)
    x = _fill(shape, torch.float32)
    torch.testing.assert_close(x.to(mojo_device).triu(diagonal).cpu(), x.triu(diagonal))


def test_triu_every_dtype(mojo_gpu):
    for dtype in (torch.bfloat16, torch.int64, torch.uint8, torch.bool):
        x = (_fill((6, 6), torch.int64) % 2).to(dtype)
        torch.testing.assert_close(x.to(mojo_gpu).triu(1).cpu(), x.triu(1))


# ---------------------------------------------------------------------------
# select_scatter
# ---------------------------------------------------------------------------


def test_select_scatter_basic(mojo_device, call_checker):
    call_checker.register(aten_functions.aten_select_scatter)
    a = _fill((4, 5), torch.float32)
    src = torch.full((5,), -1.0)
    expected = a.select_scatter(src, 0, 2)
    dev = a.to(mojo_device).select_scatter(src.to(mojo_device), 0, 2)
    torch.testing.assert_close(dev.cpu(), expected)
    # `a` itself is untouched (select_scatter is functional).
    torch.testing.assert_close(a, _fill((4, 5), torch.float32))


def test_select_scatter_negative_dim_and_index(mojo_device):
    a = _fill((3, 4, 5), torch.float32)
    src = torch.full((3, 5), 7.0)
    expected = a.select_scatter(src, -2, -1)
    dev = a.to(mojo_device).select_scatter(src.to(mojo_device), -2, -1)
    torch.testing.assert_close(dev.cpu(), expected)


def test_select_scatter_casts_src(mojo_gpu):
    # float16 (not float64): the fast CastSpec kernel's dtype set is what
    # select_scatter's src-cast uses, matching the old eager path's
    # `_cast_tensor` (pre-gated on the same set, never a host round trip).
    a = _fill((4, 5), torch.float32)
    src = torch.full((5,), 9.0, dtype=torch.float16)
    expected = a.select_scatter(src, 0, 1)
    dev = a.to(mojo_gpu).select_scatter(src.to(mojo_gpu), 0, 1)
    torch.testing.assert_close(dev.cpu(), expected)


def test_select_scatter_rejects_a_src_of_the_wrong_shape(mojo_gpu):
    """`select_scatter_symint` checks `slice.sizes() == src.sizes()`: it does
    not broadcast, and neither may this backend."""
    a = _fill((4, 5), torch.float32).to(mojo_gpu)
    src = torch.tensor(9.0, device=mojo_gpu)  # 0-d against a (5,) slice
    with pytest.raises(RuntimeError, match="size equal to the slice"):
        a.select_scatter(src, 0, 1)


# ---------------------------------------------------------------------------
# scatter.src / scatter.value
# ---------------------------------------------------------------------------


@pytest.mark.parametrize("dim", [0, 1, -1])
def test_scatter_src(mojo_device, dim, call_checker):
    call_checker.register(aten_functions.aten_scatter_src)
    a = _fill((4, 5), torch.float32)
    # Indices with no duplicate destination within a scatter group: torch's
    # own semantics call a colliding scatter "nondeterministic" (whichever
    # write lands last wins), so a randint index could legitimately disagree
    # with the cpu reference by write-order alone -- that is not a kernel bug.
    # Each row (dim=1/-1) or column (dim=0) is instead a distinct sub-permutation.
    if dim == 0:
        index = torch.stack([torch.randperm(4)[:3] for _ in range(5)], dim=1)
    elif dim == 1:
        index = torch.stack([torch.randperm(5)[:4] for _ in range(3)], dim=0)
    else:
        index = torch.stack([torch.randperm(5)[:3] for _ in range(4)], dim=0)
    index = index.to(torch.int64)
    src = _fill(tuple(index.shape), torch.float32) + 100
    expected = a.scatter(dim, index, src)
    dev = a.to(mojo_device).scatter(dim, index.to(mojo_device), src.to(mojo_device))
    torch.testing.assert_close(dev.cpu(), expected)


def test_scatter_value(mojo_device, call_checker):
    call_checker.register(aten_functions.aten_scatter_value)
    a = _fill((4, 5), torch.float32)
    index = torch.randint(0, 4, (2, 5)).to(torch.int64)
    expected = a.scatter(0, index, -3.5)
    dev = a.to(mojo_device).scatter(0, index.to(mojo_device), -3.5)
    torch.testing.assert_close(dev.cpu(), expected)


def test_scatter_value_bool(mojo_gpu):
    a = torch.zeros(4, 5, dtype=torch.bool)
    index = torch.randint(0, 4, (2, 5)).to(torch.int64)
    expected = a.scatter(0, index, True)
    dev = a.to(mojo_gpu).scatter(0, index.to(mojo_gpu), True)
    assert dev.cpu().tolist() == expected.tolist()


def test_scatter_rejects_an_out_of_range_index(mojo_gpu):
    """The kernel skips the write rather than scribbling outside the tensor,
    and raises the flag the host reads back after the launch."""
    a = torch.zeros(4, 5, device=mojo_gpu)
    src = torch.ones(2, 5, device=mojo_gpu)
    for bad in (4, -1, 1 << 40):
        index = torch.zeros(2, 5, dtype=torch.int64, device=mojo_gpu)
        index[0, 0] = bad
        with pytest.raises(RuntimeError, match="index out of range"):
            a.scatter(0, index, src)
    with pytest.raises(RuntimeError, match="index out of range"):
        index = torch.full((2, 5), 9, dtype=torch.int64, device=mojo_gpu)
        a.scatter(0, index, -3.5)
    # A valid scatter still works after the flagged one (the flag is fresh
    # per launch, not a sticky per-device bit).
    ok = torch.zeros(2, 5, dtype=torch.int64, device=mojo_gpu)
    torch.testing.assert_close(
        a.scatter(0, ok, src).cpu(), torch.zeros(4, 5).scatter(0, ok.cpu(), src.cpu())
    )


def test_scatter_rejects_an_index_bigger_than_self(mojo_gpu):
    """ATen's `scatter_shape_check`: index.size(d) <= self.size(d) off `dim`,
    and index.size(d) <= src.size(d) everywhere."""
    a = torch.zeros(4, 5, device=mojo_gpu)
    index = torch.zeros(2, 9, dtype=torch.int64, device=mojo_gpu)
    with pytest.raises(RuntimeError, match="smaller than self"):
        a.scatter(0, index, 1.0)
    index = torch.zeros(2, 5, dtype=torch.int64, device=mojo_gpu)
    with pytest.raises(RuntimeError, match="smaller than src"):
        a.scatter(0, index, torch.ones(1, 5, device=mojo_gpu))


def test_index_tensor_out_of_range_index_stays_in_bounds(mojo_gpu):
    """An out-of-range gather index is clamped in the kernel: the read stays
    inside the table, so the process survives a bad index instead of reading
    device memory it does not own."""
    table = _fill((4, 3), torch.float32).to(mojo_gpu)
    idx = torch.tensor([0, 99, -99], dtype=torch.int64, device=mojo_gpu)
    out = table[idx]
    assert tuple(out.shape) == (3, 3)
    torch.testing.assert_close(out[0].cpu(), table[0].cpu())
    # The clamped rows are unspecified in value but must be real table rows.
    rows = table.cpu().tolist()
    assert out[1].cpu().tolist() in rows
    assert out[2].cpu().tolist() in rows


def test_scatter_rejects_rank_beyond_4(mojo_gpu):
    a = torch.zeros(2, 2, 2, 2, 2).to(mojo_gpu)
    index = torch.zeros(2, 2, 2, 2, 2, dtype=torch.int64).to(mojo_gpu)
    with pytest.raises(NotImplementedError):
        a.scatter(0, index, 1.0)


# ---------------------------------------------------------------------------
# index.Tensor
# ---------------------------------------------------------------------------


@pytest.mark.parametrize("idx_dtype", [torch.int64, torch.int32])
def test_index_tensor_gather_rows(mojo_device, idx_dtype, call_checker):
    call_checker.register(aten_functions.aten_index)
    x = _fill((10, 3, 4), torch.float32)
    idx = torch.tensor([3, 0, 7, 7], dtype=idx_dtype)
    expected = x[idx]
    dev = x.to(mojo_device)[idx.to(mojo_device)]
    torch.testing.assert_close(dev.cpu(), expected)


def test_index_tensor_negative_indices(mojo_gpu):
    x = _fill((6, 4), torch.float32)
    idx = torch.tensor([-1, -2, 0], dtype=torch.int64)
    expected = x[idx]
    dev = x.to(mojo_gpu)[idx.to(mojo_gpu)]
    torch.testing.assert_close(dev.cpu(), expected)


def test_index_tensor_bool_mask_full_rank(mojo_device, call_checker):
    call_checker.register(aten_functions.aten_index)
    x = _fill((4, 5), torch.float32)
    mask = _fill((4, 5), torch.int64) % 3 == 0
    expected = x[mask]
    dev = x.to(mojo_device)[mask.to(mojo_device)]
    torch.testing.assert_close(dev.cpu(), expected)


def test_index_tensor_bool_mask_partial_rank(mojo_gpu):
    x = _fill((4, 5, 3), torch.float32)
    mask = torch.tensor([True, False, True, False])
    expected = x[mask]
    dev = x.to(mojo_gpu)[mask.to(mojo_gpu)]
    torch.testing.assert_close(dev.cpu(), expected)


def test_index_tensor_bool_mask_all_false(mojo_gpu):
    x = _fill((4, 5), torch.float32)
    mask = torch.zeros(4, 5, dtype=torch.bool)
    expected = x[mask]
    dev = x.to(mojo_gpu)[mask.to(mojo_gpu)]
    assert dev.cpu().shape == expected.shape


# ---------------------------------------------------------------------------
# nonzero
# ---------------------------------------------------------------------------


def test_nonzero_basic(mojo_device, call_checker):
    call_checker.register(aten_functions.aten_nonzero)
    x = torch.tensor([[1, 0, 2], [0, 3, 0], [4, 0, 5]], dtype=torch.float32)
    torch.testing.assert_close(x.to(mojo_device).nonzero().cpu(), x.nonzero())


def test_nonzero_all_zeros(mojo_device):
    x = torch.zeros(3, 3, dtype=torch.float32)
    torch.testing.assert_close(x.to(mojo_device).nonzero().cpu(), x.nonzero())


def test_nonzero_all_nonzero(mojo_device):
    x = torch.ones(2, 3, dtype=torch.float32)
    torch.testing.assert_close(x.to(mojo_device).nonzero().cpu(), x.nonzero())


@pytest.mark.parametrize("shape", [(2,), (3, 4), (2, 3, 4)])
def test_nonzero_shapes(mojo_gpu, shape):
    x = _fill(shape, torch.float32)
    x = x * (x > 0.5)  # scatter in some real zeros
    torch.testing.assert_close(x.to(mojo_gpu).nonzero().cpu(), x.nonzero())


@pytest.mark.parametrize("value", [0, 1])
def test_nonzero_scalar_has_no_coordinate_column(mojo_gpu, value):
    """A 0-d tensor has no coordinates: ATen reports (n, 0), not (n, 1)."""
    want = torch.nonzero(torch.tensor(value))
    got = torch.nonzero(torch.tensor(value, device=mojo_gpu))
    assert tuple(got.shape) == tuple(want.shape)
    assert got.cpu().tolist() == want.tolist()


def _check_cat_cast(device, rows, widths, source_offset=0, destination_offset=0):
    hosts, sources, source_bases = [], [], []
    for index, width in enumerate(widths):
        bits = (
            (
                torch.arange(rows * width + 32, dtype=torch.int64) * 7919
                + index * 357
                + 13
            )
            % 65536
        ).to(torch.int16)
        host = bits.view(torch.bfloat16)
        base = host.to(device)
        hosts.append(
            host[source_offset : source_offset + rows * width].view(rows, width)
        )
        sources.append(
            base[source_offset : source_offset + rows * width].view(rows, width)
        )
        source_bases.append((base, bits))
    size = rows * sum(widths)
    backing = torch.full((size + 32,), 17.0, device=device)
    output = backing[destination_offset : destination_offset + size].view(
        rows, sum(widths)
    )
    version, pointer = output._version, output.data_ptr()
    assert torch.cat(sources, 1, out=output) is output
    expected = torch.cat(hosts, 1).float()
    actual = output.cpu()
    torch.testing.assert_close(actual, expected, rtol=0, atol=0, equal_nan=True)
    valid = ~torch.isnan(expected)
    torch.testing.assert_close(
        actual.view(torch.int32)[valid],
        expected.view(torch.int32)[valid],
        rtol=0,
        atol=0,
    )
    assert output._version == version + 1
    assert output.data_ptr() == pointer
    assert torch.all(backing[:destination_offset].cpu() == 17)
    assert torch.all(backing[destination_offset + size :].cpu() == 17)
    for source, bits in source_bases:
        torch.testing.assert_close(source.cpu().view(torch.int16), bits, rtol=0, atol=0)


@pytest.mark.parametrize("source_offset", range(8))
@pytest.mark.parametrize("destination_offset", range(4))
def test_cat_cast_offsets(mojo_gpu, source_offset, destination_offset):
    _check_cat_cast(
        mojo_gpu,
        3,
        [0, 1, 2, 3, 4, 7, 8, 9, 255, 256, 257, 1023, 1024, 1025, 4095, 4096, 4097],
        source_offset,
        destination_offset,
    )


@pytest.mark.parametrize(
    "rows,widths,source_offset,destination_offset",
    [
        (3, [0, 0, 0], 3, 1),
        (0, [17, 0, 1025], 3, 1),
        (1, [8] * 64, 0, 0),
        (3, [513] * 65, 1, 3),
        (2, [1031] * 129, 3, 1),
        (2, [0 if i % 3 == 0 else (i * 357) % 5001 for i in range(197)], 7, 3),
        (65536, [1, 0, 2], 1, 1),
        (
            2,
            [
                800,
                800,
                3840000,
                2400,
                1280000,
                800,
                800,
                800,
                5120000,
                3200,
                5120000,
                800,
            ],
            0,
            0,
        ),
        (
            2,
            [
                384,
                384,
                884736,
                1152,
                294912,
                384,
                384,
                384,
                1179648,
                1536,
                1179648,
                384,
            ],
            0,
            0,
        ),
        (1, [8388608, 8388608], 0, 0),
        (3, [357, 789, 13, 1601], 0, 0),
        (2, [357, 789, 13, 1601], 0, 1),
        (1, [0, 7, 0, 1031], 0, 0),
        (2, [8] * 65, 0, 0),
        (3, [0 if i % 3 == 0 else 8 * (i % 17 + 1) for i in range(129)], 0, 0),
        (65536, [8, 0, 16], 0, 0),
    ],
)
def test_cat_cast_regimes(mojo_gpu, rows, widths, source_offset, destination_offset):
    _check_cat_cast(mojo_gpu, rows, widths, source_offset, destination_offset)


@pytest.mark.parametrize("dim", [0, 1, 2, -1])
def test_cat_cast_dimensions(mojo_gpu, dim):
    parts = [
        torch.arange(2 * 3 * 5, dtype=torch.float32).view(2, 3, 5).bfloat16(),
        torch.ones(2, 3, 5, dtype=torch.bfloat16),
    ]
    expected = torch.cat(parts, dim).float()
    out = torch.empty_like(expected, device=mojo_gpu)
    torch.cat([x.to(mojo_gpu) for x in parts], dim, out=out)
    torch.testing.assert_close(out.cpu(), expected, rtol=0, atol=0)


@pytest.mark.parametrize(
    "kind",
    [
        "resized",
        "strided_input",
        "strided_output",
        "same_dtype",
        "overlap",
        "shared_inputs",
    ],
)
def test_cat_cast_fallbacks(mojo_gpu, kind):
    hosts = [
        torch.arange(24, dtype=torch.float32).reshape(3, 8).bfloat16(),
        torch.ones(3, 8, dtype=torch.bfloat16),
    ]
    parts = [x.to(mojo_gpu) for x in hosts]
    if kind == "shared_inputs":
        parts[1] = parts[0]
        hosts[1] = hosts[0]
    if kind == "strided_input":
        parts = [x[:, ::2] for x in parts]
        hosts = [x[:, ::2] for x in hosts]
    expected = torch.cat(hosts, 1)
    dtype = torch.bfloat16 if kind == "same_dtype" else torch.float32
    if kind == "resized":
        out = torch.empty(1, device=mojo_gpu, dtype=dtype)
    elif kind == "strided_output":
        out = torch.empty(expected.numel() * 2, device=mojo_gpu, dtype=dtype)[::2].view(
            expected.shape
        )
    elif kind == "overlap":
        # Existing temporary route safely reads BF16 views before writing
        # overlapping FP32 output. Preserve that backend behavior.
        out = torch.empty_like(expected, device=mojo_gpu, dtype=dtype)
        source = out.view(torch.bfloat16).reshape(-1)[:48].view(3, 16)
        source.copy_(torch.cat(hosts, 1).to(mojo_gpu))
        parts = [source]
    else:
        out = torch.empty_like(expected, device=mojo_gpu, dtype=dtype)
    torch.cat(parts, 1, out=out)
    torch.testing.assert_close(out.cpu(), expected.to(dtype), rtol=0, atol=0)


@pytest.mark.parametrize(
    "kind", ["empty_list", "wrong_shape", "wrong_dtype", "wrong_device"]
)
def test_cat_cast_rejects_before_writing(mojo_gpu, kind):
    parts = [
        torch.ones(3, 8, dtype=torch.bfloat16, device=mojo_gpu),
        torch.ones(3, 8, dtype=torch.bfloat16, device=mojo_gpu),
    ]
    out = torch.full((3, 16), 17.0, device=mojo_gpu)
    if kind == "empty_list":
        parts = []
    elif kind == "wrong_shape":
        parts[1] = torch.ones(4, 8, dtype=torch.bfloat16, device=mojo_gpu)
    elif kind == "wrong_dtype":
        parts[1] = parts[1].float()
    else:
        parts[1] = parts[1].cpu()
    with pytest.raises((RuntimeError, NotImplementedError)):
        torch.cat(parts, 1, out=out)
    assert torch.all(out.cpu() == 17)


_CAT_OUT_DTYPES = [
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
# Every pair c10::canCast allows into or out of float32 and bfloat16, the
# same-dtype copy of each width, and integer widening/narrowing and bool.
_CAT_OUT_PAIRS = sorted(
    {
        (a, b)
        for a in _CAT_OUT_DTYPES
        for b in _CAT_OUT_DTYPES
        if (a == b or {a, b} & {torch.float32, torch.bfloat16})
        and not (a.is_floating_point and not b.is_floating_point)
        and not (b == torch.bool and a != torch.bool)
    }
    | {
        (torch.int64, torch.int8),
        (torch.uint8, torch.int32),
        (torch.bool, torch.int64),
    },
    key=str,
)


def _cat_out_values(n: int, dtype: torch.dtype, seed: int) -> torch.Tensor:
    index = torch.arange(n, dtype=torch.int64) * 7919 + 13 + seed
    if dtype == torch.bool:
        return index % 3 == 1
    if dtype.is_floating_point:
        return ((index % 20001 - 10000).double() / 97).to(dtype)
    return (index % 251 - 125).to(dtype)


@pytest.mark.parametrize(("src_dtype", "dst_dtype"), _CAT_OUT_PAIRS, ids=str)
@pytest.mark.parametrize(
    "dim,widths,offsets",
    [
        (1, [0, 1, 7, 17, 2049, 8193], (0, 0)),
        (1, [5, 1031, 3], (1, 3)),
        (0, [4, 1], (3, 1)),
    ],
)
def test_cat_out_batched_dtypes(mojo_gpu, src_dtype, dst_dtype, dim, widths, offsets):
    """One batched rectangle copy per call: rows of every input straight into
    the (possibly converting) contiguous output, at aligned and odd offsets."""
    rows = 3
    source_offset, destination_offset = offsets
    hosts, sources = [], []
    for index, width in enumerate(widths):
        shape = (rows, width, 2) if dim == 1 else (width, rows, 2)
        numel = rows * width * 2
        host = _cat_out_values(numel + source_offset, src_dtype, index)
        hosts.append(host[source_offset:].view(shape))
        sources.append(host.to(mojo_gpu)[source_offset:].view(shape))
    expected = torch.cat(hosts, dim).to(dst_dtype)
    backing = torch.zeros(
        expected.numel() + destination_offset + 4, dtype=dst_dtype, device=mojo_gpu
    )
    out = backing[destination_offset:][: expected.numel()].view(expected.shape)
    version, pointer = out._version, out.data_ptr()
    assert torch.cat(sources, dim, out=out) is out
    torch.testing.assert_close(out.cpu(), expected, rtol=0, atol=0, equal_nan=True)
    assert out._version == version + 1 and out.data_ptr() == pointer
    guard = backing.cpu()
    assert not guard[:destination_offset].any()
    assert not guard[destination_offset + expected.numel() :].any()


@pytest.mark.parametrize(
    ("src_dtype", "dst_dtype"),
    [
        (torch.float32, torch.int64),
        (torch.bfloat16, torch.uint8),
        (torch.int32, torch.bool),
    ],
    ids=str,
)
def test_cat_out_rejects_what_torch_cannot_cast(mojo_gpu, src_dtype, dst_dtype):
    parts = [torch.ones(2, 3, dtype=src_dtype), torch.ones(2, 5, dtype=src_dtype)]
    with pytest.raises(TypeError, match="can't be cast to the desired output type"):
        torch.cat(parts, 1, out=torch.empty(2, 8, dtype=dst_dtype))
    before = torch.full((2, 8), 3, dtype=dst_dtype)
    out = before.to(mojo_gpu)
    with pytest.raises(
        (TypeError, RuntimeError), match="can't be cast to the desired output type"
    ):
        torch.cat([x.to(mojo_gpu) for x in parts], 1, out=out)
    torch.testing.assert_close(out.cpu(), before, rtol=0, atol=0)


def test_cat_out_keeps_a_matching_out_where_it_is(mojo_gpu):
    """`cat.out` must not resize an out that already has the right shape:
    resizing resets sizes, strides AND offset, which would send the result to
    the front of the base storage instead of into the caller's view."""
    base = torch.zeros(16, device=mojo_gpu)
    out = base[4:8]
    parts = [torch.ones(2, device=mojo_gpu), torch.full((2,), 2.0, device=mojo_gpu)]
    torch.cat(parts, 0, out=out)
    assert out.cpu().tolist() == [1.0, 1.0, 2.0, 2.0]
    assert base.cpu().tolist() == [0.0] * 4 + [1.0, 1.0, 2.0, 2.0] + [0.0] * 8


def test_cat_out_resizes_a_mismatching_out(mojo_gpu):
    out = torch.empty(0, device=mojo_gpu)
    parts = [torch.ones(2, device=mojo_gpu), torch.full((3,), 2.0, device=mojo_gpu)]
    torch.cat(parts, 0, out=out)
    assert out.cpu().tolist() == [1.0, 1.0, 2.0, 2.0, 2.0]


@pytest.mark.parametrize("dtype", [torch.float16, torch.bfloat16])
@pytest.mark.parametrize("strided", [False, True], ids=["contiguous", "strided"])
def test_chunk_cat_mixed_precision_out(
    mojo_device: str, dtype: torch.dtype, strided: bool
):
    """FSDP2 packs half-precision, uneven parameter gradients into fp32."""
    inputs = [_fill((5, 3), dtype), _fill((7,), dtype)]
    expected = torch.empty(2, 13, dtype=torch.float32)
    torch._chunk_cat(inputs, dim=0, num_chunks=2, out=expected)
    storage = torch.full((2, 15 if strided else 13), -123.0, device=mojo_device)
    out = storage[:, 1:-1] if strided else storage
    torch._chunk_cat([x.to(mojo_device) for x in inputs], dim=0, num_chunks=2, out=out)
    torch.testing.assert_close(out.cpu(), expected, rtol=0, atol=0)
    if strided:
        assert (storage.cpu()[:, (0, -1)] == -123).all()


def test_nonzero_int_dtype(mojo_gpu):
    x = torch.tensor([1, 0, 3, 0, 5], dtype=torch.int64)
    torch.testing.assert_close(x.to(mojo_gpu).nonzero().cpu(), x.nonzero())


# ---------------------------------------------------------------------------
# set_.source_Tensor
# ---------------------------------------------------------------------------


def test_set_source_tensor_adopts_the_allocation(mojo_gpu):
    # Built on the host and uploaded (torch.arange(..., device=mojo) is the
    # factories group's op, not this one's) rather than through arange/add_
    # directly on the mojo device.
    destination = torch.zeros(8, device=mojo_gpu)
    source = (torch.arange(4, dtype=torch.float32) + 50).to(mojo_gpu)
    returned = destination.set_(source)  # ty: ignore[invalid-argument-type]
    assert returned is destination
    assert tuple(destination.shape) == (4,)
    assert destination.cpu().tolist() == [50.0, 51.0, 52.0, 53.0]
    # Sharing the allocation, not a copy of it.
    source.fill_(99.0)
    assert destination.cpu().tolist() == [99.0] * 4


def test_set_source_tensor_keeps_its_own_dtype(mojo_gpu):
    """Matches upstream `set_tensor_`'s `set_storage_keep_dtype`: self's own
    dtype survives, only storage/sizes/strides move -- a raw bit
    reinterpretation, same as real ATen. `Tensor.set_`'s Python method adds
    its own dtype-equality check ahead of the dispatcher for this overload,
    so this goes through the aten op directly, the way FSDP1's C++-side caller
    does."""
    destination = torch.zeros(4, dtype=torch.int32, device=mojo_gpu)
    source = torch.arange(4, dtype=torch.float32).to(mojo_gpu)
    torch.ops.aten.set_.source_Tensor(destination, source)
    assert destination.dtype == torch.int32
    assert destination.cpu().tolist() == [0, 1065353216, 1073741824, 1077936128]


# ---------------------------------------------------------------------------
# empty_permuted
# ---------------------------------------------------------------------------


def test_empty_permuted_shape_dtype_device(mojo_device, call_checker):
    call_checker.register(aten_functions.aten_empty_permuted)
    out = torch.ops.aten.empty_permuted(
        [2, 3, 4], [1, 0, 2], dtype=torch.float16, device=mojo_device
    )
    assert tuple(out.shape) == (2, 3, 4)
    assert out.dtype == torch.float16
    assert out.device.type == "mojo"
    # physical_layout is the whole point of the op: dim 1 is outermost, then
    # dim 0, then dim 2 -- the same strides CPU torch produces.
    reference = torch.ops.aten.empty_permuted([2, 3, 4], [1, 0, 2])
    assert out.stride() == reference.stride()
    assert not out.is_contiguous()
    assert out.permute(1, 0, 2).is_contiguous()


@pytest.mark.parametrize(
    "layout", [[0, 1, 2], [2, 1, 0], [1, 2, 0], [0, 2, 1], [2, 0, 1]]
)
def test_empty_permuted_every_layout_matches_cpu(mojo_gpu, layout):
    out = torch.ops.aten.empty_permuted([2, 3, 4], layout, device=mojo_gpu)
    reference = torch.ops.aten.empty_permuted([2, 3, 4], layout)
    assert out.stride() == reference.stride()
    assert out.is_contiguous() == reference.is_contiguous()


def test_empty_permuted_rejects_a_bad_layout(mojo_gpu):
    with pytest.raises(RuntimeError, match="Duplicate dim"):
        torch.ops.aten.empty_permuted([2, 3], [0, 0], device=mojo_gpu)
    with pytest.raises(RuntimeError, match="physical_layout"):
        torch.ops.aten.empty_permuted([2, 3], [0], device=mojo_gpu)


# ---------------------------------------------------------------------------
# Cast exactness across every dtype pair, length and storage offset.
#
# `test_to_copy_dtype_cast` above is 11 fixed pairs at shape (3,5) from
# offset 0. The values here are `arange % 5`, whose period is coprime with
# every power-of-two vector width, so a rotated lane or an unwritten tail
# cannot pass; every comparison is exact.
# ---------------------------------------------------------------------------

_CAST_DTYPES = [
    torch.float32,
    torch.float16,
    torch.bfloat16,
    torch.int64,
    torch.int32,
    torch.uint8,
    torch.bool,
]


@pytest.mark.parametrize("src_dtype", _CAST_DTYPES)
def test_cast_is_exact_for_every_dtype_pair(mojo_gpu, src_dtype):
    for dst_dtype in _CAST_DTYPES:
        for numel in (1, 3, 17, 1027, 4099):
            for offset in (0, 1, 2, 3):
                base = torch.arange(numel + offset) % 5
                src_cpu = (base != 0) if src_dtype == torch.bool else base.to(src_dtype)
                view_cpu = src_cpu[offset : offset + numel]
                view = src_cpu.to(mojo_gpu)[offset : offset + numel]
                assert view.is_contiguous()
                assert torch.equal(view.to(dst_dtype).cpu(), view_cpu.to(dst_dtype)), (
                    src_dtype,
                    dst_dtype,
                    numel,
                    offset,
                )


@pytest.mark.parametrize("dst_dtype", [torch.bfloat16, torch.float16])
def test_float_narrowing_rounds_like_cpu(mojo_gpu, dst_dtype):
    """Rounding mode, not just range: 65_539 values at three base alignments,
    down and back up, exactly equal to CPU."""
    for numel in (1027, 65_539):
        for offset in (0, 1, 3):
            src_cpu = torch.randn(numel + offset, dtype=torch.float32) * 8.0
            view_cpu = src_cpu[offset:]
            view = src_cpu.to(mojo_gpu)[offset:]
            narrowed = view.to(dst_dtype)
            assert torch.equal(narrowed.cpu(), view_cpu.to(dst_dtype)), (numel, offset)
            assert torch.equal(
                narrowed.to(torch.float32).cpu(), view_cpu.to(dst_dtype).float()
            )


# ---------------------------------------------------------------------------
# cat / repeat / stack batching edges
# ---------------------------------------------------------------------------


@pytest.mark.parametrize("dtype", [torch.float32, torch.bfloat16, torch.int64])
@pytest.mark.parametrize(
    "shapes,dim",
    [
        ([(5000,)] * 64, 0),  # exactly the per-launch segment cap
        ([(37,)] * 130, 0),  # two batches past it
        ([(2, 3, 5, 64), (2, 3, 1, 64), (2, 3, 9, 64)], 2),  # 4-D middle dim
        ([(3, 4), (100_000, 4), (1, 4)], 0),  # wildly unequal members
    ],
)
def test_cat_batching_edges(mojo_gpu, shapes, dim, dtype):
    parts = [_fill(shape, dtype) for shape in shapes]
    expected = torch.cat(parts, dim=dim)
    actual = torch.cat([p.to(mojo_gpu) for p in parts], dim=dim)
    torch.testing.assert_close(actual.cpu(), expected, rtol=0, atol=0)


def test_cat_mixed_contiguity_in_one_list(mojo_gpu):
    """One launch must not assume every member has the same layout."""
    a = _fill((32, 64), torch.float32)
    b = _fill((64, 32), torch.float32)
    c = _fill((32, 64), torch.float32)
    expected = torch.cat([a, b.t(), c], dim=0)
    actual = torch.cat([a.to(mojo_gpu), b.to(mojo_gpu).t(), c.to(mojo_gpu)], dim=0)
    torch.testing.assert_close(actual.cpu(), expected, rtol=0, atol=0)


@pytest.mark.parametrize(
    "shape,reps",
    [
        ((100, 1), (3, 7)),  # a single input column
        ((7, 128), (4, 1, 2)),  # more reps than input dims
        ((2, 3), (1000, 1)),
        ((1, 5000), (5000, 1)),
        ((1, 64), (1, 300)),
        ((64,), (5,)),  # rank-1 input
        ((64,), (2, 5)),  # rank-1 input, left-padded
        ((1, 8, 16), (2, 3, 4)),  # leading extent 1
        ((4, 8, 16), (2, 3, 4)),  # genuine rank-3 tile
        ((2, 3, 4, 5), (2, 2, 2, 2)),  # rank 4
    ],
)
def test_repeat_geometries(mojo_gpu, shape, reps):
    cpu = _fill(shape, torch.float32)
    torch.testing.assert_close(
        cpu.to(mojo_gpu).repeat(*reps).cpu(), cpu.repeat(*reps), rtol=0, atol=0
    )


def test_stack_opinfo_samples_do_not_corrupt_the_heap(mojo_gpu):
    """Regression for a SIGSEGV (exit 139), not just a wrong value.

    The destination-side narrow copy vectorized 4 elements wide on CPU and
    could issue one width-4 store PAST the destination allocation; it only
    corrupts when the heap is packed, so the faithful repro is the whole
    OpInfo sample sequence plus a value check.
    """
    from torch.testing._internal.common_methods_invocations import (  # noqa: PLC0415 -- importing op_db at module scope would pull torch's OpInfo database into every collection of this file
        op_db,
    )

    infos = [info for info in op_db if info.name == "stack"]
    assert infos, "no `stack` OpInfo in this torch"
    ran = 0
    for sample in infos[0].sample_inputs(torch.device("cpu"), torch.int64):
        parts = list(sample.input)
        expected = torch.stack(parts, *sample.args, **sample.kwargs)
        actual = torch.stack(
            [p.to(mojo_gpu) for p in parts], *sample.args, **sample.kwargs
        )
        torch.testing.assert_close(actual.cpu(), expected, rtol=0, atol=0)
        ran += 1
    assert ran > 0


@pytest.mark.parametrize(
    "dtype", [torch.float32, torch.float16, torch.bfloat16, torch.int64]
)
@pytest.mark.parametrize("axis", [0, 1, 2])
def test_index_put_single_axis(mojo_gpu: str, dtype: torch.dtype, axis: int):
    data = torch.arange(5 * 7 * 9).reshape(5, 7, 9).to(dtype)
    indices = torch.tensor([4, 0, 2], dtype=torch.int64)
    value_shape = list(data.shape)
    value_shape[axis] = indices.numel()
    values = torch.full(value_shape, -7, dtype=dtype)
    expected = data.clone()
    cpu_indexer = (slice(None),) * axis + (indices,) + (slice(None),) * (2 - axis)
    expected[cpu_indexer] = values
    ours = data.to(mojo_gpu)
    pointer = ours.data_ptr()
    gpu_indexer = (
        (slice(None),) * axis + (indices.to(mojo_gpu),) + (slice(None),) * (2 - axis)
    )
    ours[gpu_indexer] = values.to(mojo_gpu)
    assert ours.data_ptr() == pointer
    torch.testing.assert_close(ours.cpu(), expected)


@pytest.mark.parametrize("rowlen", [3, 4, 7, 8, 9, 257, 789, 1024, 1025, 1028, 2056])
@pytest.mark.parametrize(
    "target_offset,source_offset", [(0, 0), (1, 0), (0, 1), (1, 1)]
)
@pytest.mark.parametrize("dtype", [torch.float32, torch.bfloat16])
def test_index_put_row_copy_offsets(
    mojo_gpu: str,
    rowlen: int,
    target_offset: int,
    source_offset: int,
    dtype: torch.dtype,
):
    n, k = 11, 5
    indices = torch.tensor([8, 0, 10, 3, 5])
    source_host = ((torch.arange(k * rowlen + source_offset + 1) % 31).float() / 8).to(
        dtype
    )
    source_storage = source_host.to(mojo_gpu)
    values = source_storage[source_offset : source_offset + k * rowlen].reshape(
        k, rowlen
    )
    storage = torch.full((n * rowlen + target_offset + 1,), -99.0, dtype=dtype).to(
        mojo_gpu
    )
    ours = storage[target_offset : target_offset + n * rowlen].reshape(n, rowlen)
    expected = torch.full((n, rowlen), -99.0, dtype=dtype)
    expected[indices] = source_host[source_offset : source_offset + k * rowlen].reshape(
        k, rowlen
    )
    pointer = ours.data_ptr()
    returned = ours.index_put_((indices.to(mojo_gpu),), values)
    assert returned.data_ptr() == pointer
    torch.testing.assert_close(ours.cpu(), expected, rtol=0, atol=0)
    actual_storage = storage.cpu()
    torch.testing.assert_close(
        actual_storage[:target_offset], torch.full((target_offset,), -99.0, dtype=dtype)
    )
    assert actual_storage[-1] == -99
    torch.testing.assert_close(source_storage.cpu(), source_host, rtol=0, atol=0)


@pytest.mark.parametrize("bad_index", [-1, 11])
@pytest.mark.parametrize("dtype", [torch.float32, torch.bfloat16])
def test_index_put_row_copy_invalid_leaves_target(
    mojo_gpu: str, bad_index: int, dtype: torch.dtype
):
    data = torch.full((11, 1024), -31.0, dtype=dtype)
    ours = data.to(mojo_gpu)
    indices = torch.tensor([2, bad_index, 0]).to(mojo_gpu)
    values = torch.full((3, 1024), 17.0, dtype=dtype).to(mojo_gpu)
    with pytest.raises(NotImplementedError, match="nonnegative"):
        ours.index_put_((indices,), values)
    torch.testing.assert_close(ours.cpu(), data, rtol=0, atol=0)


@pytest.mark.parametrize("dtype", [torch.float32, torch.bfloat16])
def test_index_put_row_copy_broadcast_fallback(mojo_gpu: str, dtype: torch.dtype):
    indices = torch.tensor([8, 0, 10, 3, 5])
    values = torch.arange(5, dtype=dtype).reshape(5, 1)
    expected = torch.full((11, 257), -13.0, dtype=dtype)
    expected[indices] = values
    ours = torch.full((11, 257), -13.0, dtype=dtype).to(mojo_gpu)
    ours.index_put_((indices.to(mojo_gpu),), values.to(mojo_gpu))
    torch.testing.assert_close(ours.cpu(), expected, rtol=0, atol=0)


@pytest.mark.parametrize("dtype", [torch.float32, torch.bfloat16])
def test_index_put_row_copy_strided_index_fallback(mojo_gpu: str, dtype: torch.dtype):
    storage = torch.tensor([8, 9, 0, 9, 10, 9, 3, 9, 5, 9])
    indices = storage.to(mojo_gpu)[::2]
    values = torch.arange(5 * 257, dtype=dtype).reshape(5, 257)
    expected = torch.full((11, 257), -13.0, dtype=dtype)
    expected[storage[::2]] = values
    ours = torch.full((11, 257), -13.0, dtype=dtype).to(mojo_gpu)
    ours.index_put_((indices,), values.to(mojo_gpu))
    torch.testing.assert_close(ours.cpu(), expected, rtol=0, atol=0)


@pytest.mark.parametrize("value_shape", [(), (1,), (1, 4), (2, 1)])
def test_index_put_broadcast(mojo_gpu: str, value_shape: tuple[int, ...]):
    data = torch.arange(20, dtype=torch.float32).reshape(5, 4)
    indices = torch.tensor([1, 4])
    values = torch.full(value_shape, -3.0)
    expected = data.clone()
    expected[indices] = values
    ours = data.to(mojo_gpu)
    returned = ours.index_put_((indices.to(mojo_gpu),), values.to(mojo_gpu))
    assert returned.data_ptr() == ours.data_ptr()
    torch.testing.assert_close(ours.cpu(), expected)


def test_index_put_empty(mojo_gpu: str):
    data = torch.arange(20, dtype=torch.float32).reshape(5, 4)
    ours = data.to(mojo_gpu)
    ours.index_put_(
        (torch.empty(0, dtype=torch.int64).to(mojo_gpu),),
        torch.empty(0, 4).to(mojo_gpu),
    )
    torch.testing.assert_close(ours.cpu(), data)


def test_index_put_noncontiguous(mojo_gpu: str):
    data = torch.arange(20, dtype=torch.float32).reshape(4, 5)
    indices = torch.tensor([1, 4])
    values = torch.full((2, 4), -3.0)
    expected = data.t().clone()
    expected[indices] = values
    ours = data.to(mojo_gpu).t()
    try:
        ours.index_put_((indices.to(mojo_gpu),), values.to(mojo_gpu))
    except NotImplementedError as exc:
        assert "contigu" in str(exc).lower()
    else:
        torch.testing.assert_close(ours.cpu(), expected)


@pytest.mark.parametrize("case", ["int32", "negative", "out_of_bounds", "accumulate"])
def test_index_put_declined_inputs(mojo_gpu: str, case: str):
    indices = torch.tensor(
        [-1] if case == "negative" else [9] if case == "out_of_bounds" else [1],
        dtype=torch.int32 if case == "int32" else torch.int64,
    )
    data = torch.zeros(3, 4).to(mojo_gpu)
    values = torch.ones(1, 4).to(mojo_gpu)
    with pytest.raises(NotImplementedError):
        data.index_put_(
            (indices.to(mojo_gpu),), values, accumulate=case == "accumulate"
        )
    torch.testing.assert_close(data.cpu(), torch.zeros(3, 4))
