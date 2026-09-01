"""Data-movement kernels: cat / stack / repeat / strided clone / tril /
triu / arange / dtype cast.

clone is benchmarked on STRIDED inputs only: a contiguous clone is a
device memcpy, which measure.py excludes from device time by design (the
node would raise NoDeviceKernels).  _to_copy is benchmarked only in its
on-device dtype-cast regime (the vectorized cast kernel); its device-
move regimes are memcpys and unmeasurable here for the same reason.
arange runs entirely on-device (the fast_arange kernel, hot in HF decode
loops) — the mojo leg builds the tensor on the mojo device directly.
"""

from __future__ import annotations

import pytest
import torch
from bench_lib.cases import DTYPES, both
from bench_lib.check import Bench
from bench_lib.hw import Hardware

# (pieces, elements per piece).  The piece COUNT is a regime axis of its own:
# CatN batches up to CAT_SEG_CAP inputs per launch, so 64 is the last count
# that fits one launch and 65 is the first that does not — benchmark both
# sides of that edge, or a change to the cap only ever gets measured on its
# good side.  The odd piece length is the unaligned regime: 281673 = 357*789
# elements is not a multiple of 16 bytes in either dtype, so it exercises the
# element/tail path instead of the wide vector path.
CAT_SHAPES: dict[str, tuple[int, int]] = {
    "P_2x8388608": (2, 8388608),
    "P_64x262144": (64, 262144),
    "P_65x262144": (65, 262144),
    "P_512x2048": (512, 2048),
    "P_64x281673": (64, 281673),
}
STACK_SHAPES: dict[str, tuple[int, int]] = {
    "P_8x1048576": (8, 1048576),
    "P_32x65536": (32, 65536),
}
# (rows, cols, repeats).  The ASPECT RATIO of the input is a regime axis of
# its own, and the two square shapes this dict started with hid every one of
# them.  repeat dispatches on the row length in bytes AND on how much grid
# the resulting geometry has, so all four of these are separately reachable:
#
#   8192x64 r(1,16)   a 256-byte row: 16 vector slots to spread over a block
#   64x8192 r(16,1)   one row fills whole blocks and is repeated DOWN
#   2x3    r(100000,1)  a 12-byte row and almost no input rows -- the copy
#                       axis, not the row axis, is where the parallelism is
#   1x64   r(1,9375)    ONE output row, all of its width coming from the
#                       repeat factor, which the segment kernel's serial
#                       inner loop cannot spread over the grid at all
#
# The last two are here because they are what a regression looks like: a
# first version of the tiled kernel ran them at 1.9x the general kernel it
# replaced while every recorded shape improved 24-41x.
REPEAT_SHAPES: dict[str, tuple[int, int, tuple[int, int]]] = {
    "S_1024x1024_r4x4": (1024, 1024, (4, 4)),
    "S_357x789_r3x5": (357, 789, (3, 5)),
    "S_8192x64_r1x16": (8192, 64, (1, 16)),
    "S_64x8192_r16x1": (64, 8192, (16, 1)),
    "S_2x3_r100000x1": (2, 3, (100000, 1)),
    "S_1x64_r1x9375": (1, 64, (1, 9375)),
}
TRI_SHAPES: dict[str, tuple[int, int]] = {"S_8192x8192": (8192, 8192)}
ARANGE_N = 16777216

# (N, C, H, W). Two round shapes plus one awkward one (357x789, from the
# repo-wide convention of covering an unaligned regime) -- all comfortably
# above PAD2D_PADDING's largest side (5), so reflect's "pad < input
# dimension" validation never trips.
PAD2D_SHAPES: dict[str, tuple[int, int, int, int]] = {
    "S_8x64x64x64": (8, 64, 64, 64),
    "S_32x128x32x32": (32, 128, 32, 32),
    "S_16x3x357x789": (16, 3, 357, 789),
}
# Asymmetric on every side (left, right, top, bottom) -- the normal case for
# F.pad's 4-tuple, not just the symmetric special case.
PAD2D_PADDING = (3, 5, 2, 4)

COVERS: dict[str, str] = {
    "aten::cat": "test_cat",
    "aten::stack": "test_stack",
    "aten::repeat": "test_repeat",
    "aten::clone": "test_clone (strided inputs; contiguous clone is a memcpy)",
    "aten::tril": "test_tril",
    "aten::triu": "test_triu",
    "aten::arange": "test_arange",
    "aten::_to_copy": "test_to_copy_cast (dtype-cast regime only)",
    "aten::reflection_pad2d": "test_reflection_pad2d",
    "aten::replication_pad2d": "test_replication_pad2d",
}

SKIPPED: dict[str, str] = {}


@pytest.mark.parametrize("dtype_id", ("bf16", "f32"))
@pytest.mark.parametrize("shape_id", CAT_SHAPES)
def test_cat(
    shape_id: str, dtype_id: str, bench: Bench, hw: Hardware, mojo_device: torch.device
) -> None:
    pieces, elems = CAT_SHAPES[shape_id]
    dtype = DTYPES[dtype_id]
    refs, ours = [], []
    for _ in range(pieces):
        ref, our = both(torch.randn(elems, dtype=dtype), hw, mojo_device)
        refs.append(ref)
        ours.append(our)
    bench.run(
        lambda: torch.cat(refs), lambda: torch.cat(ours), flops=float(pieces * elems)
    )


@pytest.mark.parametrize("dtype_id", ("bf16", "f32"))
@pytest.mark.parametrize("shape_id", STACK_SHAPES)
def test_stack(
    shape_id: str, dtype_id: str, bench: Bench, hw: Hardware, mojo_device: torch.device
) -> None:
    pieces, elems = STACK_SHAPES[shape_id]
    dtype = DTYPES[dtype_id]
    refs, ours = [], []
    for _ in range(pieces):
        ref, our = both(torch.randn(elems, dtype=dtype), hw, mojo_device)
        refs.append(ref)
        ours.append(our)
    bench.run(
        lambda: torch.stack(refs),
        lambda: torch.stack(ours),
        flops=float(pieces * elems),
    )


@pytest.mark.parametrize("dtype_id", ("bf16", "f32"))
@pytest.mark.parametrize("shape_id", REPEAT_SHAPES)
def test_repeat(
    shape_id: str, dtype_id: str, bench: Bench, hw: Hardware, mojo_device: torch.device
) -> None:
    rows, cols, reps = REPEAT_SHAPES[shape_id]
    x_ref, x_our = both(
        torch.randn(rows, cols, dtype=DTYPES[dtype_id]), hw, mojo_device
    )
    bench.run(
        lambda: x_ref.repeat(*reps),
        lambda: x_our.repeat(*reps),
        flops=float(rows * cols * reps[0] * reps[1]),
    )


@pytest.mark.parametrize("dtype_id", ("bf16", "f32"))
@pytest.mark.parametrize("layout", ("T", "sliced"))
@pytest.mark.parametrize("shape_id", ("S_4096x4096",))
def test_clone(
    shape_id: str,
    layout: str,
    dtype_id: str,
    bench: Bench,
    hw: Hardware,
    mojo_device: torch.device,
) -> None:
    dtype = DTYPES[dtype_id]
    if layout == "T":
        base_ref, base_our = both(torch.randn(4096, 4096, dtype=dtype), hw, mojo_device)
        x_ref, x_our = base_ref.t(), base_our.t()
    else:
        base_ref, base_our = both(torch.randn(8192, 4096, dtype=dtype), hw, mojo_device)
        x_ref, x_our = base_ref[::2], base_our[::2]
    bench.run(lambda: x_ref.clone(), lambda: x_our.clone(), flops=float(x_ref.numel()))


@pytest.mark.parametrize("dtype_id", ("bf16", "f32"))
@pytest.mark.parametrize("shape_id", TRI_SHAPES)
def test_tril(
    shape_id: str, dtype_id: str, bench: Bench, hw: Hardware, mojo_device: torch.device
) -> None:
    x_ref, x_our = both(
        torch.randn(TRI_SHAPES[shape_id], dtype=DTYPES[dtype_id]), hw, mojo_device
    )
    bench.run(
        lambda: torch.tril(x_ref), lambda: torch.tril(x_our), flops=float(x_ref.numel())
    )


@pytest.mark.parametrize("dtype_id", ("bf16", "f32"))
@pytest.mark.parametrize("shape_id", TRI_SHAPES)
def test_triu(
    shape_id: str, dtype_id: str, bench: Bench, hw: Hardware, mojo_device: torch.device
) -> None:
    x_ref, x_our = both(
        torch.randn(TRI_SHAPES[shape_id], dtype=DTYPES[dtype_id]), hw, mojo_device
    )
    bench.run(
        lambda: torch.triu(x_ref), lambda: torch.triu(x_our), flops=float(x_ref.numel())
    )


@pytest.mark.parametrize("dtype_id", ("f32", "i64"))
@pytest.mark.parametrize("shape_id", (f"N_{ARANGE_N}",))
def test_arange(
    shape_id: str, dtype_id: str, bench: Bench, hw: Hardware, mojo_device: torch.device
) -> None:
    dtype = DTYPES[dtype_id]
    bench.run(
        lambda: torch.arange(ARANGE_N, dtype=dtype, device=hw.stock_device),
        lambda: torch.arange(ARANGE_N, dtype=dtype, device=mojo_device),
        flops=float(ARANGE_N),
    )


# source dtype axis; the cast target is folded into the layout token.
CAST_TARGETS: dict[str, tuple[str, torch.dtype]] = {
    "bf16": ("to_f32", torch.float32),
    "f32": ("to_bf16", torch.bfloat16),
}


@pytest.mark.parametrize("dtype_id", ("bf16", "f32"))
@pytest.mark.parametrize("shape_id", ("C_16777216", "A_357x789"))
@pytest.mark.bench_op("_to_copy")
def test_to_copy_cast(
    shape_id: str, dtype_id: str, bench: Bench, hw: Hardware, mojo_device: torch.device
) -> None:
    shape = (16777216,) if shape_id == "C_16777216" else (357, 789)
    _, target = CAST_TARGETS[dtype_id]
    x_ref, x_our = both(torch.randn(shape, dtype=DTYPES[dtype_id]), hw, mojo_device)
    bench.run(
        lambda: x_ref.to(target), lambda: x_our.to(target), flops=float(x_ref.numel())
    )


def _pad2d_out_numel(shape: tuple[int, int, int, int]) -> float:
    n, c, h, w = shape
    pad_l, pad_r, pad_t, pad_b = PAD2D_PADDING
    return float(n * c * (h + pad_t + pad_b) * (w + pad_l + pad_r))


@pytest.mark.parametrize("dtype_id", ("bf16", "f32"))
@pytest.mark.parametrize("shape_id", PAD2D_SHAPES)
def test_reflection_pad2d(
    shape_id: str, dtype_id: str, bench: Bench, hw: Hardware, mojo_device: torch.device
) -> None:
    shape = PAD2D_SHAPES[shape_id]
    x_ref, x_our = both(torch.randn(shape, dtype=DTYPES[dtype_id]), hw, mojo_device)
    bench.run(
        lambda: torch.nn.functional.pad(x_ref, PAD2D_PADDING, mode="reflect"),
        lambda: torch.nn.functional.pad(x_our, PAD2D_PADDING, mode="reflect"),
        flops=_pad2d_out_numel(shape),
    )


@pytest.mark.parametrize("dtype_id", ("bf16", "f32"))
@pytest.mark.parametrize("shape_id", PAD2D_SHAPES)
def test_replication_pad2d(
    shape_id: str, dtype_id: str, bench: Bench, hw: Hardware, mojo_device: torch.device
) -> None:
    shape = PAD2D_SHAPES[shape_id]
    x_ref, x_our = both(torch.randn(shape, dtype=DTYPES[dtype_id]), hw, mojo_device)
    bench.run(
        lambda: torch.nn.functional.pad(x_ref, PAD2D_PADDING, mode="replicate"),
        lambda: torch.nn.functional.pad(x_our, PAD2D_PADDING, mode="replicate"),
        flops=_pad2d_out_numel(shape),
    )
