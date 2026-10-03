"""Data-movement kernels: cat / stack / repeat / strided clone / tril /
triu / arange / dtype cast / masked_select.

clone is benchmarked on STRIDED inputs only: a contiguous clone is a
device memcpy, which measure.py excludes from device time by design (the
node would raise NoDeviceKernels).  _to_copy is benchmarked only in its
on-device dtype-cast regime (the vectorized cast kernel); its device-
move regimes are memcpys and unmeasurable here for the same reason.
arange runs entirely on-device (the fast_arange kernel, hot in HF decode
loops) — the mojo leg builds the tensor on the mojo device directly.
"""

from __future__ import annotations

import math

import pytest
import torch
from bench_lib.cases import DTYPES, both, op_params
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
# The 1-d and 3-d pads, and every pad's backward (called directly, a
# gather over grad_output). Each rank's shapes keep every side above the
# largest pad of that rank, again with one awkward shape.
PAD1D_SHAPES: dict[str, tuple[int, ...]] = {
    "S_32x64x4096": (32, 64, 4096),
    "S_16x3x357789": (16, 3, 357789),
}
PAD1D_PADDING: tuple[int, ...] = (3, 5)
PAD3D_SHAPES: dict[str, tuple[int, ...]] = {
    "S_4x32x32x64x64": (4, 32, 32, 64, 64),
    "S_2x3x37x57x89": (2, 3, 37, 57, 89),
}
PAD3D_PADDING: tuple[int, ...] = (3, 5, 2, 4, 1, 2)
PAD_BY_RANK = {
    1: (PAD1D_SHAPES, PAD1D_PADDING),
    2: (PAD2D_SHAPES, PAD2D_PADDING),
    3: (PAD3D_SHAPES, PAD3D_PADDING),
}
PAD_MODES = {"reflection": "reflect", "replication": "replicate"}
PAD1D_OPS = {"reflection_pad1d": 1, "replication_pad1d": 1}
PAD3D_OPS = {"reflection_pad3d": 3, "replication_pad3d": 3}
PAD1D_BACKWARD_OPS = {f"{m}_pad1d_backward": 1 for m in PAD_MODES}
PAD2D_BACKWARD_OPS = {f"{m}_pad2d_backward": 2 for m in PAD_MODES}
PAD3D_BACKWARD_OPS = {f"{m}_pad3d_backward": 3 for m in PAD_MODES}

COVERS: dict[str, str] = {
    "aten::split_with_sizes_copy.out": "test_split_copy_rows",
    "aten::_copy_from": "test_copy_row_strided (same-device strided copies; contiguous/device moves are memcpy)",
    "aten::cat": "test_cat",
    "aten::stack": "test_stack",
    "aten::repeat": "test_repeat",
    "aten::clone": "test_clone (strided inputs; contiguous clone is a memcpy)",
    "aten::tril": "test_tril",
    "aten::triu": "test_triu",
    "aten::arange.start_out": (
        "test_arange (torch.arange reaches the device through the out overload)"
    ),
    "aten::_to_copy": "test_to_copy_cast (dtype-cast regime only)",
    "aten::_index_put_impl_": "test_index_put",
    "aten::reflection_pad2d": "test_reflection_pad2d",
    "aten::replication_pad2d": "test_replication_pad2d",
    **{f"aten::{name}": "test_pad1d" for name in PAD1D_OPS},
    **{f"aten::{name}": "test_pad3d" for name in PAD3D_OPS},
    **{f"aten::{name}": "test_pad1d_backward" for name in PAD1D_BACKWARD_OPS},
    **{f"aten::{name}": "test_pad2d_backward" for name in PAD2D_BACKWARD_OPS},
    **{f"aten::{name}": "test_pad3d_backward" for name in PAD3D_BACKWARD_OPS},
    "aten::masked_select": "test_masked_select",
    "aten::masked_select.out": (
        "test_masked_select (same kernels, the result copied into out)"
    ),
}

# The indexing group (tmb/ops/indexing.mojo) and the range factories.
COVERS |= {
    "aten::flip": "test_flip",
    "aten::roll": "test_roll",
    "aten::channel_shuffle": "test_channel_shuffle",
    "aten::take": "test_take",
    "aten::take.out": "test_take (same gather, into the caller's out)",
    "aten::put_": "test_put",
    "aten::index_fill_.int_Scalar": "test_index_fill",
    "aten::index_fill_.int_Tensor": (
        "test_index_fill (the same scatter after one read of the 0-d value)"
    ),
    "aten::index_copy": "test_index_copy",
    "aten::index_copy_": "test_index_copy (the same scatter, into self)",
    "aten::index_copy.out": "test_index_copy (a copy of self, then the same scatter)",
    "aten::masked_scatter_": "test_masked_scatter",
    "aten::_unique2": "test_unique",
    "aten::repeat_interleave.Tensor": "test_repeat_interleave",
    "aten::unfold_backward": "test_unfold_backward",
    "aten::linspace.out": "test_linspace",
    "aten::logspace.out": "test_logspace",
    "aten::eye.m_out": "test_eye",
    "aten::eye.out": "test_eye (torch.eye(n) resolves to eye.m_out; same fills)",
}

_PAD_OUT = (
    "out-variant plumbing over an already-benchmarked functional impl: the "
    "same resample kernel, written into the caller's tensor"
)
SKIPPED: dict[str, str] = {
    "aten::unfold": "pure view/metadata op: a storage-sharing as_strided, no kernel",
    "aten::tril_indices": (
        "no device kernel: the coordinates are written by a host loop and "
        "uploaded with one H2D memcpy, which device time excludes"
    ),
    "aten::triu_indices": (
        "no device kernel: the coordinates are written by a host loop and "
        "uploaded with one H2D memcpy, which device time excludes"
    ),
    "aten::equal": (
        "eq + all + a host read through the dispatcher: the kernels are the "
        "benchmarked eq.Tensor and all, the rest is the sync"
    ),
    "aten::trace": (
        "a strided diagonal view summed by the registered sum reduction, "
        "benchmarked in test_reduction"
    ),
    "aten::dot": (
        "mul + sum through the dispatcher (float32 for the half types), both "
        "benchmarked in their own families"
    ),
    "aten::vdot": "dot for the real dtypes: mul + sum, both benchmarked",
    "aten::linalg_cross": (
        "six mul.Tensor, three sub.Tensor and three strided copies through "
        "the dispatcher, each benchmarked in its own family"
    ),
    "aten::linalg_cross.out": "linalg_cross, written into the caller's out",
    **{
        f"aten::{name}{suffix}": (
            "torch's test-only op (TestOps.cpp): a host loop, or a clone"
        )
        for name in (
            "_test_optional_intlist",
            "_test_optional_filled_intlist",
            "_test_optional_floatlist",
            "_test_functorch_fallback",
        )
        for suffix in ("", ".out")
    },
    "aten::tril.out": _PAD_OUT.replace("resample", "TriangularCopy"),
    "aten::triu.out": _PAD_OUT.replace("resample", "TriangularCopy"),
    "aten::tril_": "test_tril's TriangularCopy, then a strided copy into self",
    "aten::triu_": "test_triu's TriangularCopy, then a strided copy into self",
    "aten::range.out": (
        "the Arange kernel test_arange measures, with an inclusive end"
    ),
    "aten::randperm.generator_out": (
        "random_ (test_inplace) and sort.stable (test_reduction) through the "
        "dispatcher: no kernel of its own"
    ),
    "aten::nonzero.out": (
        "nonzero's host round trip (no device kernel), copied into out"
    ),
    "aten::nonzero_static": (
        "nonzero's host round trip (no device kernel), a fill and a memcpy"
    ),
    "aten::nonzero_static.out": ("nonzero_static, copied into the caller's out"),
    "aten::index.Tensor_out": (
        "index.Tensor (test_embedding's test_index), copied into out"
    ),
    "aten::narrow_copy.out": (
        "a narrow view's strided copy (test_copy_row_strided's kernel), copied into out"
    ),
    "aten::_unique": "test_unique's kernels without the counts",
    "aten::unique_dim": (
        "one stable sort per column plus test_unique's group passes: the "
        "sorts dominate, and they are the reductions suite's"
    ),
    "aten::unique_consecutive": "test_unique's group passes without the sort",
    "aten::unique_dim_consecutive": (
        "test_unique's group passes over rows, then index_select"
    ),
    "aten::fill_.Tensor": (
        "a one-element read of the value, then fill_.Scalar's fill kernel"
    ),
    **{
        f"aten::{m}_pad{r}d{suffix}": _PAD_OUT
        for m in PAD_MODES
        for r in (1, 2, 3)
        for suffix in (".out", "_backward.grad_input")
    },
}


SPLIT_COPY_SHAPES = {
    "S_2x15370400_mixed": (
        2,
        [800, 800, 3840000, 2400, 1280000, 800, 800, 800, 5120000, 3200, 5120000, 800],
    ),
    "S_2x9600_12pieces": (2, [800] * 12),
    "S_7x1164_awkward": (7, [357, 789, 17, 1, 0]),
    "S_1x1048576_single": (1, [1048576]),
    "S_3x33345_65pieces": (3, [513] * 65),
    "S_65536x3_empty_piece": (65536, [1, 0, 2]),
}


@pytest.mark.bench_op("split_with_sizes_copy.out")
@pytest.mark.parametrize("dtype_id", ("bf16", "f32"))
@pytest.mark.parametrize("shape_id", SPLIT_COPY_SHAPES)
def test_split_copy_rows(
    shape_id: str, dtype_id: str, bench: Bench, hw: Hardware, mojo_device: torch.device
):
    rows, sizes = SPLIT_COPY_SHAPES[shape_id]
    src_ref, src_our = both(
        torch.randn(rows, sum(sizes), dtype=DTYPES[dtype_id]), hw, mojo_device
    )
    outputs = [
        both(torch.empty(rows, n, dtype=DTYPES[dtype_id]), hw, mojo_device)
        for n in sizes
    ]
    dst_ref = [pair[0] for pair in outputs]
    dst_our = [pair[1] for pair in outputs]
    bench.run(
        lambda: torch.ops.aten.split_with_sizes_copy.out(
            src_ref, sizes, 1, out=dst_ref
        ),
        lambda: torch.ops.aten.split_with_sizes_copy.out(
            src_our, sizes, 1, out=dst_our
        ),
        flops=float(rows * sum(sizes)),
    )


# Few wide rows exercise all-gather unpacking; odd pitches and offsets cover
# scalar tails and unaligned views without making launch shapes model-specific.
ROW_COPY_SHAPES = {
    "S_2x40206400_p41027200_o0x0": (2, 40206400, 41027200, 0, 0),
    "S_2x5120000_p15370400_o0x0": (2, 5120000, 15370400, 0, 0),
    "S_2x3840000_p15370400_o1600x0": (2, 3840000, 15370400, 1600, 0),
    "S_2x800_p15370400_o0x0": (2, 800, 15370400, 0, 0),
    "S_357x789_p811_o3x5": (357, 789, 811, 3, 5),
    "S_7x1025_p1041_o1x3": (7, 1025, 1041, 1, 3),
    "S_5x32768_p32781_o0x0": (5, 32768, 32781, 0, 0),
}


@pytest.mark.bench_op("_copy_from")
@pytest.mark.parametrize("dtype_id", ("bf16", "f16", "f32"))
@pytest.mark.parametrize("shape_id", ROW_COPY_SHAPES)
def test_copy_row_strided(
    shape_id: str, dtype_id: str, bench: Bench, hw: Hardware, mojo_device: torch.device
):
    rows, cols, pitch, source_offset, destination_offset = ROW_COPY_SHAPES[shape_id]
    dtype = DTYPES[dtype_id]
    source_ref, source_our = both(
        torch.randn(rows * pitch + source_offset, dtype=dtype), hw, mojo_device
    )
    destination_ref, destination_our = both(
        torch.empty(rows * cols + destination_offset, dtype=dtype), hw, mojo_device
    )
    source_ref = source_ref.as_strided((rows, cols), (pitch, 1), source_offset)
    source_our = source_our.as_strided((rows, cols), (pitch, 1), source_offset)
    destination_ref = destination_ref[destination_offset:].view(rows, cols)
    destination_our = destination_our[destination_offset:].view(rows, cols)
    bench.run(
        lambda: destination_ref.copy_(source_ref),
        lambda: destination_our.copy_(source_our),
        flops=float(rows * cols),
    )


@pytest.mark.bench_op("cat.out")
@pytest.mark.parametrize("dtype_id", ("f32",))
@pytest.mark.parametrize("shape_id", ("R2_W41027200_P1",))
@pytest.mark.parametrize("layout", ("contiguous_out",))
def test_cat_out_contiguous_fp32(
    shape_id: str,
    dtype_id: str,
    layout: str,
    bench: Bench,
    hw: Hardware,
    mojo_device: torch.device,
):
    # The backend's existing cat.out reaches an internal contiguous copy.
    # Measure the complete public op here; isolated copy time belongs to the
    # pure Mojo harness. Plain contiguous copy_ continues to use DMA.
    source_ref, source_our = both(torch.randn(2, 41027200), hw, mojo_device)
    out_ref, out_our = both(torch.empty(2, 41027200), hw, mojo_device)
    bench.run(
        lambda: torch.cat([source_ref], 1, out=out_ref),
        lambda: torch.cat([source_our], 1, out=out_our),
        flops=float(source_ref.numel()),
    )


@pytest.mark.parametrize("dtype_id", ("bf16", "f32"))
@pytest.mark.parametrize("shape_id", CAT_SHAPES)
def test_cat(
    shape_id: str, dtype_id: str, bench: Bench, hw: Hardware, mojo_device: torch.device
):
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


CAT_CAST_SHAPES: dict[str, tuple[int, tuple[int, ...], int]] = {
    "R2_W15370400_P12": (
        2,
        (800, 800, 3840000, 2400, 1280000, 800, 800, 800, 5120000, 3200, 5120000, 800),
        0,
    ),
    "R2_W3543936_P12": (
        2,
        (384, 384, 884736, 1152, 294912, 384, 384, 384, 1179648, 1536, 1179648, 384),
        0,
    ),
    "R1_W16777216_P2": (1, (8388608, 8388608), 0),
    "R3_W2760_P4": (3, (357, 789, 13, 1601), 0),
    "R2_W2760_P4_offset1": (2, (357, 789, 13, 1601), 1),
    "R1_W1038_P4_empty": (1, (0, 7, 0, 1031), 0),
}


# (input, output) dtypes: the mixed-precision cast both ways and a
# same-dtype cat.out, which also writes straight into `out`.
CAT_OUT_DTYPES: dict[str, tuple[torch.dtype, torch.dtype]] = {
    "bf16_f32": (torch.bfloat16, torch.float32),
    "f32_bf16": (torch.float32, torch.bfloat16),
    "f32": (torch.float32, torch.float32),
}


@pytest.mark.parametrize("dtype_id", CAT_OUT_DTYPES)
@pytest.mark.parametrize("shape_id", CAT_CAST_SHAPES)
@pytest.mark.parametrize("layout", ("contiguous_cast_out",))
@pytest.mark.bench_op("cat.out")
def test_cat_cast_out(
    shape_id: str,
    dtype_id: str,
    layout: str,
    bench: Bench,
    hw: Hardware,
    mojo_device: torch.device,
):
    rows, widths, offset = CAT_CAST_SHAPES[shape_id]
    src_dtype, dst_dtype = CAT_OUT_DTYPES[dtype_id]
    refs, ours = [], []
    for index, width in enumerate(widths):
        host = (
            (
                (
                    (torch.arange(rows * width, dtype=torch.int64) * 7919 + 13 + index)
                    % 65521
                ).float()
                / 65536
                - 0.5
            )
            .to(src_dtype)
            .view(rows, width)
        )
        ref, our = both(host, hw, mojo_device)
        refs.append(ref)
        ours.append(our)
    size = rows * sum(widths)
    ref_base, our_base = both(torch.empty(size + 16, dtype=dst_dtype), hw, mojo_device)
    ref_out = ref_base[offset : offset + size].view(rows, sum(widths))
    our_out = our_base[offset : offset + size].view(rows, sum(widths))
    bench.run(
        lambda: torch.cat(refs, 1, out=ref_out),
        lambda: torch.cat(ours, 1, out=our_out),
        flops=float(size),
    )


@pytest.mark.parametrize("dtype_id", ("bf16", "f32"))
@pytest.mark.parametrize("shape_id", STACK_SHAPES)
def test_stack(
    shape_id: str, dtype_id: str, bench: Bench, hw: Hardware, mojo_device: torch.device
):
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
):
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
):
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
):
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
):
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
):
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
):
    shape = (16777216,) if shape_id == "C_16777216" else (357, 789)
    _, target = CAST_TARGETS[dtype_id]
    x_ref, x_our = both(torch.randn(shape, dtype=DTYPES[dtype_id]), hw, mojo_device)
    bench.run(
        lambda: x_ref.to(target), lambda: x_our.to(target), flops=float(x_ref.numel())
    )


INDEX_PUT_SHAPES = {
    "N1000C256H7W7_K500": ((1000, 256, 7, 7), 500),
    "N357C789_K119": ((357, 789), 119),
}


@pytest.mark.parametrize("dtype_id", ["bf16", "f32"])
@pytest.mark.parametrize("shape_id", INDEX_PUT_SHAPES)
@pytest.mark.bench_op("_index_put_impl_")
def test_index_put(
    shape_id: str, dtype_id: str, bench: Bench, hw: Hardware, mojo_device: torch.device
):
    shape, count = INDEX_PUT_SHAPES[shape_id]
    dtype = DTYPES[dtype_id]
    data = torch.zeros(shape, dtype=dtype)
    indices = torch.arange(count, dtype=torch.int64) * 2
    values = torch.randn((count, *shape[1:]), dtype=dtype)
    d_ref, d_our = both(data, hw, mojo_device)
    i_ref, i_our = both(indices, hw, mojo_device)
    v_ref, v_our = both(values, hw, mojo_device)
    bench.run(
        lambda: d_ref.index_put_((i_ref,), v_ref),
        lambda: d_our.index_put_((i_our,), v_our),
        flops=float(values.numel()),
    )


def _pad2d_out_numel(shape: tuple[int, int, int, int]) -> float:
    n, c, h, w = shape
    pad_l, pad_r, pad_t, pad_b = PAD2D_PADDING
    return float(n * c * (h + pad_t + pad_b) * (w + pad_l + pad_r))


@pytest.mark.parametrize("dtype_id", ("bf16", "f32"))
@pytest.mark.parametrize("shape_id", PAD2D_SHAPES)
def test_reflection_pad2d(
    shape_id: str, dtype_id: str, bench: Bench, hw: Hardware, mojo_device: torch.device
):
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
):
    shape = PAD2D_SHAPES[shape_id]
    x_ref, x_our = both(torch.randn(shape, dtype=DTYPES[dtype_id]), hw, mojo_device)
    bench.run(
        lambda: torch.nn.functional.pad(x_ref, PAD2D_PADDING, mode="replicate"),
        lambda: torch.nn.functional.pad(x_our, PAD2D_PADDING, mode="replicate"),
        flops=_pad2d_out_numel(shape),
    )


def _pad_fn(name: str) -> tuple[str, int]:
    """(F.pad mode, rank) of a `<mode>_pad<r>d[_backward]` op name."""
    base, rest = name.split("_pad", 1)
    return PAD_MODES[base], int(rest[0])


def _pad_out_shape(shape: tuple[int, ...], padding: tuple[int, ...]) -> list[int]:
    """F.pad's output shape: pairs (lo, hi) run from the last dim."""
    out = list(shape)
    for k in range(len(padding) // 2):
        out[-1 - k] += padding[2 * k] + padding[2 * k + 1]
    return out


def _bench_pad(
    name: str,
    shape_id: str,
    dtype_id: str,
    bench: Bench,
    hw: Hardware,
    mojo_device: torch.device,
):
    mode, rank = _pad_fn(name)
    shapes, padding = PAD_BY_RANK[rank]
    shape = shapes[shape_id]
    x_ref, x_our = both(torch.randn(shape, dtype=DTYPES[dtype_id]), hw, mojo_device)
    bench.run(
        lambda: torch.nn.functional.pad(x_ref, padding, mode=mode),
        lambda: torch.nn.functional.pad(x_our, padding, mode=mode),
        flops=float(math.prod(_pad_out_shape(shape, padding))),
    )


def _bench_pad_backward(
    name: str,
    shape_id: str,
    dtype_id: str,
    bench: Bench,
    hw: Hardware,
    mojo_device: torch.device,
):
    _, rank = _pad_fn(name)
    shapes, padding = PAD_BY_RANK[rank]
    shape = shapes[shape_id]
    op = getattr(torch.ops.aten, name)
    out_shape = _pad_out_shape(shape, padding)
    dtype = DTYPES[dtype_id]
    x_ref, x_our = both(torch.randn(shape, dtype=dtype), hw, mojo_device)
    g_ref, g_our = both(torch.randn(out_shape, dtype=dtype), hw, mojo_device)
    bench.run(
        lambda: op(g_ref, x_ref, list(padding)),
        lambda: op(g_our, x_our, list(padding)),
        flops=float(math.prod(out_shape)),
    )


@pytest.mark.parametrize("dtype_id", ("bf16", "f32"))
@pytest.mark.parametrize("shape_id", PAD1D_SHAPES)
@pytest.mark.parametrize("op_name", op_params(PAD1D_OPS))
def test_pad1d(
    op_name: str,
    shape_id: str,
    dtype_id: str,
    bench: Bench,
    hw: Hardware,
    mojo_device: torch.device,
):
    _bench_pad(op_name, shape_id, dtype_id, bench, hw, mojo_device)


@pytest.mark.parametrize("dtype_id", ("bf16", "f32"))
@pytest.mark.parametrize("shape_id", PAD3D_SHAPES)
@pytest.mark.parametrize("op_name", op_params(PAD3D_OPS))
def test_pad3d(
    op_name: str,
    shape_id: str,
    dtype_id: str,
    bench: Bench,
    hw: Hardware,
    mojo_device: torch.device,
):
    _bench_pad(op_name, shape_id, dtype_id, bench, hw, mojo_device)


@pytest.mark.parametrize("dtype_id", ("bf16", "f32"))
@pytest.mark.parametrize("shape_id", PAD1D_SHAPES)
@pytest.mark.parametrize("op_name", op_params(PAD1D_BACKWARD_OPS))
def test_pad1d_backward(
    op_name: str,
    shape_id: str,
    dtype_id: str,
    bench: Bench,
    hw: Hardware,
    mojo_device: torch.device,
):
    _bench_pad_backward(op_name, shape_id, dtype_id, bench, hw, mojo_device)


@pytest.mark.parametrize("dtype_id", ("bf16", "f32"))
@pytest.mark.parametrize("shape_id", PAD2D_SHAPES)
@pytest.mark.parametrize("op_name", op_params(PAD2D_BACKWARD_OPS))
def test_pad2d_backward(
    op_name: str,
    shape_id: str,
    dtype_id: str,
    bench: Bench,
    hw: Hardware,
    mojo_device: torch.device,
):
    _bench_pad_backward(op_name, shape_id, dtype_id, bench, hw, mojo_device)


@pytest.mark.parametrize("dtype_id", ("bf16", "f32"))
@pytest.mark.parametrize("shape_id", PAD3D_SHAPES)
@pytest.mark.parametrize("op_name", op_params(PAD3D_BACKWARD_OPS))
def test_pad3d_backward(
    op_name: str,
    shape_id: str,
    dtype_id: str,
    bench: Bench,
    hw: Hardware,
    mojo_device: torch.device,
):
    _bench_pad_backward(op_name, shape_id, dtype_id, bench, hw, mojo_device)


MASKED_SELECT_SHAPES: dict[str, tuple[int, ...]] = {
    "C_16777216": (16777216,),
    "A_357x789": (357, 789),
}


@pytest.mark.parametrize("dtype_id", ("bf16", "f32"))
@pytest.mark.parametrize("shape_id", MASKED_SELECT_SHAPES)
def test_masked_select(
    shape_id: str, dtype_id: str, bench: Bench, hw: Hardware, mojo_device: torch.device
):
    # Fixed 50% density under the seeded fixture, as for nonzero: the output
    # size is identical on both legs and across runs. Device time only, so
    # the host read of the per-tile counts is not measured.
    shape = MASKED_SELECT_SHAPES[shape_id]
    x_ref, x_our = both(torch.randn(shape, dtype=DTYPES[dtype_id]), hw, mojo_device)
    m_ref, m_our = both(torch.rand(shape) < 0.5, hw, mojo_device)
    bench.run(
        lambda: torch.masked_select(x_ref, m_ref),
        lambda: torch.masked_select(x_our, m_our),
        flops=float(x_ref.numel()),
    )


# ---------------------------------------------------------------------------
# The indexing group: flip / roll / channel_shuffle are strided copies,
# take / put_ / index_fill_ / index_copy the gather and scatter kernels,
# masked_scatter_ and repeat_interleave compositions around cumsum.
# ---------------------------------------------------------------------------

INDEXING_SHAPES: dict[str, tuple[int, int]] = {
    "S_4096x4096": (4096, 4096),
    "A_357x789": (357, 789),
}


@pytest.mark.parametrize("dtype_id", ("bf16", "f32"))
@pytest.mark.parametrize("shape_id", INDEXING_SHAPES)
@pytest.mark.parametrize("layout", ("dim0", "dim1", "both"))
def test_flip(
    shape_id: str,
    dtype_id: str,
    layout: str,
    bench: Bench,
    hw: Hardware,
    mojo_device: torch.device,
):
    dims = {"dim0": (0,), "dim1": (1,), "both": (0, 1)}[layout]
    shape = INDEXING_SHAPES[shape_id]
    x_ref, x_our = both(torch.randn(shape, dtype=DTYPES[dtype_id]), hw, mojo_device)
    bench.run(
        lambda: torch.flip(x_ref, dims),
        lambda: torch.flip(x_our, dims),
        flops=float(x_ref.numel()),
    )


@pytest.mark.parametrize("dtype_id", ("bf16", "f32"))
@pytest.mark.parametrize("shape_id", INDEXING_SHAPES)
@pytest.mark.parametrize("layout", ("dim1", "both", "flat"))
def test_roll(
    shape_id: str,
    dtype_id: str,
    layout: str,
    bench: Bench,
    hw: Hardware,
    mojo_device: torch.device,
):
    shape = INDEXING_SHAPES[shape_id]
    args = {"dim1": ((37,), (1,)), "both": ((5, 37), (0, 1)), "flat": ((12345,), ())}
    shifts, dims = args[layout]
    x_ref, x_our = both(torch.randn(shape, dtype=DTYPES[dtype_id]), hw, mojo_device)
    bench.run(
        lambda: torch.roll(x_ref, shifts, dims),
        lambda: torch.roll(x_our, shifts, dims),
        flops=float(x_ref.numel()),
    )


CHANNEL_SHUFFLE_SHAPES: dict[str, tuple[tuple[int, int, int, int], int]] = {
    "S_32x256x28x28_g4": ((32, 256, 28, 28), 4),
    "A_8x116x19x23_g2": ((8, 116, 19, 23), 2),
}


@pytest.mark.parametrize("dtype_id", ("bf16", "f32"))
@pytest.mark.parametrize("shape_id", CHANNEL_SHUFFLE_SHAPES)
def test_channel_shuffle(
    shape_id: str, dtype_id: str, bench: Bench, hw: Hardware, mojo_device: torch.device
):
    shape, groups = CHANNEL_SHUFFLE_SHAPES[shape_id]
    x_ref, x_our = both(torch.randn(shape, dtype=DTYPES[dtype_id]), hw, mojo_device)
    bench.run(
        lambda: torch.nn.functional.channel_shuffle(x_ref, groups),
        lambda: torch.nn.functional.channel_shuffle(x_our, groups),
        flops=float(x_ref.numel()),
    )


@pytest.mark.parametrize("dtype_id", ("bf16", "f32"))
@pytest.mark.parametrize("shape_id", INDEXING_SHAPES)
def test_take(
    shape_id: str, dtype_id: str, bench: Bench, hw: Hardware, mojo_device: torch.device
):
    shape = INDEXING_SHAPES[shape_id]
    numel = shape[0] * shape[1]
    x_ref, x_our = both(torch.randn(shape, dtype=DTYPES[dtype_id]), hw, mojo_device)
    i_ref, i_our = both(torch.randint(-numel, numel, (numel // 2,)), hw, mojo_device)
    bench.run(
        lambda: torch.take(x_ref, i_ref),
        lambda: torch.take(x_our, i_our),
        flops=float(i_ref.numel()),
    )


@pytest.mark.parametrize("dtype_id", ("bf16", "f32"))
@pytest.mark.parametrize("shape_id", INDEXING_SHAPES)
@pytest.mark.parametrize("layout", ("set", "accumulate"))
def test_put(
    shape_id: str,
    dtype_id: str,
    layout: str,
    bench: Bench,
    hw: Hardware,
    mojo_device: torch.device,
):
    shape = INDEXING_SHAPES[shape_id]
    numel = shape[0] * shape[1]
    dtype = DTYPES[dtype_id]
    x_ref, x_our = both(torch.randn(shape, dtype=dtype), hw, mojo_device)
    i_ref, i_our = both(torch.randperm(numel)[: numel // 2], hw, mojo_device)
    s_ref, s_our = both(torch.randn(numel // 2, dtype=dtype), hw, mojo_device)
    acc = layout == "accumulate"
    bench.run(
        lambda: x_ref.put_(i_ref, s_ref, accumulate=acc),
        lambda: x_our.put_(i_our, s_our, accumulate=acc),
        flops=float(i_ref.numel()),
    )


@pytest.mark.parametrize("dtype_id", ("bf16", "f32"))
@pytest.mark.parametrize("shape_id", INDEXING_SHAPES)
@pytest.mark.parametrize("layout", ("dim0", "dim1"))
def test_index_fill(
    shape_id: str,
    dtype_id: str,
    layout: str,
    bench: Bench,
    hw: Hardware,
    mojo_device: torch.device,
):
    dim = 0 if layout == "dim0" else 1
    shape = INDEXING_SHAPES[shape_id]
    x_ref, x_our = both(torch.randn(shape, dtype=DTYPES[dtype_id]), hw, mojo_device)
    idx = torch.randperm(shape[dim])[: shape[dim] // 3]
    i_ref, i_our = both(idx, hw, mojo_device)
    bench.run(
        lambda: x_ref.index_fill_(dim, i_ref, -1.0),
        lambda: x_our.index_fill_(dim, i_our, -1.0),
        flops=float(idx.numel() * x_ref.numel() // shape[dim]),
    )


@pytest.mark.parametrize("dtype_id", ("bf16", "f32"))
@pytest.mark.parametrize("shape_id", INDEXING_SHAPES)
@pytest.mark.parametrize("layout", ("dim0", "dim1"))
def test_index_copy(
    shape_id: str,
    dtype_id: str,
    layout: str,
    bench: Bench,
    hw: Hardware,
    mojo_device: torch.device,
):
    dim = 0 if layout == "dim0" else 1
    shape = INDEXING_SHAPES[shape_id]
    dtype = DTYPES[dtype_id]
    x_ref, x_our = both(torch.randn(shape, dtype=dtype), hw, mojo_device)
    idx = torch.randperm(shape[dim])[: shape[dim] // 3]
    src_shape = list(shape)
    src_shape[dim] = idx.numel()
    i_ref, i_our = both(idx, hw, mojo_device)
    s_ref, s_our = both(torch.randn(src_shape, dtype=dtype), hw, mojo_device)
    bench.run(
        lambda: x_ref.index_copy_(dim, i_ref, s_ref),
        lambda: x_our.index_copy_(dim, i_our, s_our),
        flops=float(s_ref.numel()),
    )


@pytest.mark.parametrize("dtype_id", ("bf16", "f32"))
@pytest.mark.parametrize("shape_id", INDEXING_SHAPES)
def test_masked_scatter(
    shape_id: str, dtype_id: str, bench: Bench, hw: Hardware, mojo_device: torch.device
):
    shape = INDEXING_SHAPES[shape_id]
    dtype = DTYPES[dtype_id]
    x_ref, x_our = both(torch.randn(shape, dtype=dtype), hw, mojo_device)
    m_ref, m_our = both(torch.rand(shape) < 0.5, hw, mojo_device)
    s_ref, s_our = both(torch.randn(shape, dtype=dtype), hw, mojo_device)
    bench.run(
        lambda: x_ref.masked_scatter_(m_ref, s_ref),
        lambda: x_our.masked_scatter_(m_our, s_our),
        flops=float(x_ref.numel()),
    )


REPEAT_INTERLEAVE_SHAPES: dict[str, int] = {"N_65536": 65536, "N_789": 789}


@pytest.mark.parametrize("dtype_id", ("i64",))
@pytest.mark.parametrize("shape_id", REPEAT_INTERLEAVE_SHAPES)
def test_repeat_interleave(
    shape_id: str, dtype_id: str, bench: Bench, hw: Hardware, mojo_device: torch.device
):
    # `output_size` given: no host read of the total on either leg.
    repeats = torch.randint(0, 8, (REPEAT_INTERLEAVE_SHAPES[shape_id],))
    total = int(repeats.sum())
    r_ref, r_our = both(repeats.to(DTYPES[dtype_id]), hw, mojo_device)
    bench.run(
        lambda: torch.repeat_interleave(r_ref, output_size=total),
        lambda: torch.repeat_interleave(r_our, output_size=total),
        flops=float(total),
    )


UNFOLD_SHAPES: dict[str, tuple[int, int, int]] = {
    "S_16x65536_k8s1": (16, 65536, 8),
    "A_357x789_k5s1": (357, 789, 5),
}


@pytest.mark.parametrize("dtype_id", ("bf16", "f32"))
@pytest.mark.parametrize("shape_id", UNFOLD_SHAPES)
def test_unfold_backward(
    shape_id: str, dtype_id: str, bench: Bench, hw: Hardware, mojo_device: torch.device
):
    rows, cols, size = UNFOLD_SHAPES[shape_id]
    grad_shape = (rows, cols - size + 1, size)
    g_ref, g_our = both(
        torch.randn(grad_shape, dtype=DTYPES[dtype_id]), hw, mojo_device
    )
    bench.run(
        lambda: torch.ops.aten.unfold_backward(g_ref, [rows, cols], 1, size, 1),
        lambda: torch.ops.aten.unfold_backward(g_our, [rows, cols], 1, size, 1),
        flops=float(g_ref.numel()),
    )


@pytest.mark.parametrize("dtype_id", ("f32", "bf16", "i64"))
@pytest.mark.parametrize("shape_id", (f"N_{ARANGE_N}",))
def test_linspace(
    shape_id: str, dtype_id: str, bench: Bench, hw: Hardware, mojo_device: torch.device
):
    dtype = DTYPES[dtype_id]
    bench.run(
        lambda: torch.linspace(-3, 1000, ARANGE_N, dtype=dtype, device=hw.stock_device),
        lambda: torch.linspace(-3, 1000, ARANGE_N, dtype=dtype, device=mojo_device),
        flops=float(ARANGE_N),
    )


@pytest.mark.parametrize("dtype_id", ("f32", "bf16"))
@pytest.mark.parametrize("shape_id", (f"N_{ARANGE_N}",))
def test_logspace(
    shape_id: str, dtype_id: str, bench: Bench, hw: Hardware, mojo_device: torch.device
):
    dtype = DTYPES[dtype_id]
    bench.run(
        lambda: torch.logspace(-3, 3, ARANGE_N, dtype=dtype, device=hw.stock_device),
        lambda: torch.logspace(-3, 3, ARANGE_N, dtype=dtype, device=mojo_device),
        flops=float(ARANGE_N),
    )


@pytest.mark.parametrize("dtype_id", ("f32", "bf16"))
@pytest.mark.parametrize("shape_id", ("S_4096x4096", "A_357x789"))
def test_eye(
    shape_id: str, dtype_id: str, bench: Bench, hw: Hardware, mojo_device: torch.device
):
    n, m = INDEXING_SHAPES[shape_id]
    dtype = DTYPES[dtype_id]
    bench.run(
        lambda: torch.eye(n, m, dtype=dtype, device=hw.stock_device),
        lambda: torch.eye(n, m, dtype=dtype, device=mojo_device),
        flops=float(n * m),
    )


UNIQUE_SHAPES: dict[str, tuple[int, int]] = {
    "N_16777216_K1000000": (16_777_216, 1_000_000),
    "N_357789_K97": (357_789, 97),
}


@pytest.mark.parametrize("dtype_id", ("i64", "f32"))
@pytest.mark.parametrize("shape_id", UNIQUE_SHAPES)
@pytest.mark.bench_op("_unique2")
def test_unique(
    shape_id: str, dtype_id: str, bench: Bench, hw: Hardware, mojo_device: torch.device
):
    """Sorted unique with inverse and counts: the sort, the group passes and
    the one read of the group count, on both legs."""
    n, k = UNIQUE_SHAPES[shape_id]
    x_ref, x_our = both(torch.randint(0, k, (n,)).to(DTYPES[dtype_id]), hw, mojo_device)
    bench.run(
        lambda: torch.unique(x_ref, return_inverse=True, return_counts=True),
        lambda: torch.unique(x_our, return_inverse=True, return_counts=True),
        flops=float(n),
    )
