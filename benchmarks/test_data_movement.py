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
from bench_lib.measure import gpu_lock

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
}

SKIPPED: dict[str, str] = {}


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
    with gpu_lock():
        d_ref, d_our = both(data, hw, mojo_device)
        i_ref, i_our = both(indices, hw, mojo_device)
        v_ref, v_our = both(values, hw, mojo_device)
    bench.run(
        lambda: d_ref.index_put_((i_ref,), v_ref),
        lambda: d_our.index_put_((i_our,), v_our),
        flops=float(values.numel()),
    )
