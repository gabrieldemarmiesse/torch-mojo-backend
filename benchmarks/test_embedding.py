"""Embedding / index / scatter benchmarks.

Index tensors are generated once under the seeded fixture and shared by
both legs, so gather/scatter locality is identical.  Shape tokens fold
the index-count axis in (design rule): e.g. V50304xD768_T49152 is the
nanoGPT token-embedding regime (vocab x dim, T tokens looked up).
"""

from __future__ import annotations

import pytest
import torch
import torch.nn.functional as F
from bench_lib.cases import DTYPES, both
from bench_lib.check import Bench
from bench_lib.hw import Hardware

# (vocab, dim, tokens)
EMB_SHAPES: dict[str, tuple[int, int, int]] = {
    "V50304xD768_T49152": (50304, 768, 49152),
    "V1000xD64_T4096": (1000, 64, 4096),
}
# (rows, row_width, gathered)
INDEX_SHAPES: dict[str, tuple[int, int, int]] = {
    "R_262144x64_I1048576": (262144, 64, 1048576),
    "R_1000x64_I4096": (1000, 64, 4096),
}
# (rows, row_width, scattered_rows)
SCATTER_SHAPES: dict[str, tuple[int, int, int]] = {
    "R_262144x64_S65536": (262144, 64, 65536),
    "R_1000x64_S512": (1000, 64, 512),
}
# (outer, rows, cols)
SELECT_SCATTER_SHAPES: dict[str, tuple[int, int, int]] = {
    "S32x2048x1024": (32, 2048, 1024),
    "S8x357x789": (8, 357, 789),
}
# (rows, cols, dim).  WHICH dim is indexed is THE regime axis of the
# dim-indexed family, not just a parameter: indexing the inner dim keeps every
# thread's read inside one row (and, for scatter_add, concentrates the atomics
# on `cols` slots per row), while indexing the outer dim spreads both across
# the whole allocation.  The awkward 357x789 case keeps a non-round extent in
# the suite.
DIM_INDEX_SHAPES: dict[str, tuple[int, int, int]] = {
    "R_262144x64_D1": (262144, 64, 1),
    "R_4096x4096_D0": (4096, 4096, 0),
    "R_357x789_D0": (357, 789, 0),
}
# (rows, row_width, scattered_rows) of index_put(accumulate=True), the atomic
# regime of `_index_put_impl_` (the plain-store one is test_data_movement's
# test_index_put). `_acc` in the token keeps its baseline key apart from those.
PUT_ACC_SHAPES: dict[str, tuple[int, int, int]] = {
    "R_262144x64_S65536_acc": (262144, 64, 65536),
    "R_357x789_S119_acc": (357, 789, 119),
}
# (rows, cols, selected, dim).  dim 0 is the row-gather fast path (the
# GatherRows kernel index.Tensor already uses); dim 1 is GatherDim over the
# folded (outer, selected, inner) view, where each selected element is `rows`
# separate short reads.
SELECT_SHAPES: dict[str, tuple[int, int, int, int]] = {
    "R_262144x64_S1048576_D0": (262144, 64, 1048576, 0),
    "R_4096x4096_S8192_D1": (4096, 4096, 8192, 1),
    "R_357x789_S119_D1": (357, 789, 119, 1),
}

COVERS: dict[str, str] = {
    "aten::embedding": "test_embedding",
    "aten::embedding_dense_backward": "test_embedding_backward",
    "aten::gather": "test_gather",
    "aten::index.Tensor": "test_index",
    "aten::index_add": "test_index_add",
    "aten::index_select": "test_index_select",
    "aten::scatter.src": "test_scatter_src",
    "aten::scatter.value": "test_scatter_value",
    "aten::scatter_add": "test_scatter_add",
    "aten::select_scatter": "test_select_scatter",
    "aten::scatter_reduce.two": "test_scatter_reduce",
    "aten::scatter.reduce": (
        "test_scatter_reduce (reduce='add'/'multiply' are its sum/prod launches)"
    ),
    "aten::index_reduce": "test_index_reduce",
    "aten::_embedding_bag_forward_only": "test_embedding_bag",
}

_SAME_KERNEL_OUT = (
    "out= overload of a benchmarked functional op: the same single kernel "
    "launch, writing a caller-supplied destination (through its own strides) "
    "instead of an allocated one"
)
_SAME_KERNEL_INPLACE = (
    "in-place overload of a benchmarked functional op: the same single kernel "
    "launch, minus the clone of self"
)

SKIPPED: dict[str, str] = {
    "aten::gather.out": _SAME_KERNEL_OUT,
    "aten::index_add.out": _SAME_KERNEL_OUT,
    "aten::index_add_": _SAME_KERNEL_INPLACE,
    "aten::index_select.out": _SAME_KERNEL_OUT,
    "aten::scatter.src_out": _SAME_KERNEL_OUT,
    "aten::scatter_.src": _SAME_KERNEL_INPLACE,
    "aten::scatter_add.out": _SAME_KERNEL_OUT,
    "aten::scatter_add_": _SAME_KERNEL_INPLACE,
    "aten::scatter.value_out": _SAME_KERNEL_OUT,
    "aten::scatter_.value": _SAME_KERNEL_INPLACE,
    "aten::scatter.reduce_out": _SAME_KERNEL_OUT,
    "aten::scatter_.reduce": _SAME_KERNEL_INPLACE,
    "aten::scatter.value_reduce": (
        "scatter.reduce with a scalar in place of src: the same launch"
    ),
    "aten::scatter.value_reduce_out": _SAME_KERNEL_OUT,
    "aten::scatter_.value_reduce": _SAME_KERNEL_INPLACE,
    "aten::scatter_reduce.two_out": _SAME_KERNEL_OUT,
    "aten::scatter_reduce_.two": _SAME_KERNEL_INPLACE,
    "aten::index_reduce.out": _SAME_KERNEL_OUT,
    "aten::index_reduce_": _SAME_KERNEL_INPLACE,
    "aten::_embedding_bag": (
        "test_embedding_bag's kernel: the same launch, taken when the weight "
        "requires grad"
    ),
    "aten::_embedding_bag_backward": (
        "argument checks, then _embedding_bag_dense_backward"
    ),
    "aten::_embedding_bag_dense_backward": (
        "index_select / mul / div / index_add (or scatter_add for max) "
        "through the dispatcher, each benchmarked: no kernel of its own"
    ),
    "aten::_embedding_bag_per_sample_weights_backward": (
        "index_select / mul / sum through the dispatcher, each benchmarked"
    ),
    "aten::embedding_renorm_": (
        "a host dedup of the indices (one read) and a per-row rescale kernel; "
        "the read dominates"
    ),
}


@pytest.mark.parametrize("dtype_id", ("bf16", "f32"))
@pytest.mark.parametrize("shape_id", EMB_SHAPES)
def test_embedding(
    shape_id: str, dtype_id: str, bench: Bench, hw: Hardware, mojo_device: torch.device
):
    vocab, dim, tokens = EMB_SHAPES[shape_id]
    w_ref, w_our = both(
        torch.randn(vocab, dim, dtype=DTYPES[dtype_id]), hw, mojo_device
    )
    idx_ref, idx_our = both(torch.randint(0, vocab, (tokens,)), hw, mojo_device)
    bench.run(
        lambda: F.embedding(idx_ref, w_ref),
        lambda: F.embedding(idx_our, w_our),
        flops=float(tokens * dim),
    )


@pytest.mark.parametrize("dtype_id", ("bf16", "f32"))
@pytest.mark.parametrize("shape_id", EMB_SHAPES)
@pytest.mark.bench_op("embedding_dense_backward")
def test_embedding_backward(
    shape_id: str, dtype_id: str, bench: Bench, hw: Hardware, mojo_device: torch.device
):
    vocab, dim, tokens = EMB_SHAPES[shape_id]
    g_ref, g_our = both(
        torch.randn(tokens, dim, dtype=DTYPES[dtype_id]), hw, mojo_device
    )
    idx_ref, idx_our = both(torch.randint(0, vocab, (tokens,)), hw, mojo_device)
    bench.run(
        lambda: torch.ops.aten.embedding_dense_backward(
            g_ref, idx_ref, vocab, -1, False
        ),
        lambda: torch.ops.aten.embedding_dense_backward(
            g_our, idx_our, vocab, -1, False
        ),
        flops=float(tokens * dim),
    )


@pytest.mark.parametrize("dtype_id", ("bf16", "f32"))
@pytest.mark.parametrize("shape_id", INDEX_SHAPES)
@pytest.mark.bench_op("index.Tensor")
def test_index(
    shape_id: str, dtype_id: str, bench: Bench, hw: Hardware, mojo_device: torch.device
):
    rows, width, gathered = INDEX_SHAPES[shape_id]
    x_ref, x_our = both(
        torch.randn(rows, width, dtype=DTYPES[dtype_id]), hw, mojo_device
    )
    idx_ref, idx_our = both(torch.randint(0, rows, (gathered,)), hw, mojo_device)
    bench.run(
        lambda: x_ref[idx_ref], lambda: x_our[idx_our], flops=float(gathered * width)
    )


def _scatter_case(
    shape_id: str, dtype_id: str, hw: Hardware, mojo: torch.device
) -> tuple[dict[str, torch.Tensor], dict[str, torch.Tensor]]:
    rows, width, scattered = SCATTER_SHAPES[shape_id]
    dtype = DTYPES[dtype_id]
    base = torch.randn(rows, width, dtype=dtype)
    index = torch.randint(0, rows, (scattered, width))
    src = torch.randn(scattered, width, dtype=dtype)
    ref, our = {}, {}
    for name, tensor in (("base", base), ("index", index), ("src", src)):
        ref[name], our[name] = both(tensor, hw, mojo)
    return ref, our


@pytest.mark.parametrize("dtype_id", ("bf16", "f32"))
@pytest.mark.parametrize("shape_id", SCATTER_SHAPES)
@pytest.mark.bench_op("scatter.src")
def test_scatter_src(
    shape_id: str, dtype_id: str, bench: Bench, hw: Hardware, mojo_device: torch.device
):
    ref, our = _scatter_case(shape_id, dtype_id, hw, mojo_device)
    bench.run(
        lambda: ref["base"].scatter(0, ref["index"], ref["src"]),
        lambda: our["base"].scatter(0, our["index"], our["src"]),
        flops=float(ref["base"].numel()),
    )


@pytest.mark.parametrize("dtype_id", ("bf16", "f32"))
@pytest.mark.parametrize("shape_id", SCATTER_SHAPES)
@pytest.mark.bench_op("scatter.value")
def test_scatter_value(
    shape_id: str, dtype_id: str, bench: Bench, hw: Hardware, mojo_device: torch.device
):
    ref, our = _scatter_case(shape_id, dtype_id, hw, mojo_device)
    bench.run(
        lambda: ref["base"].scatter(0, ref["index"], 1.0),
        lambda: our["base"].scatter(0, our["index"], 1.0),
        flops=float(ref["base"].numel()),
    )


@pytest.mark.parametrize("dtype_id", ("bf16", "f32"))
@pytest.mark.parametrize("shape_id", DIM_INDEX_SHAPES)
def test_gather(
    shape_id: str, dtype_id: str, bench: Bench, hw: Hardware, mojo_device: torch.device
):
    rows, cols, dim = DIM_INDEX_SHAPES[shape_id]
    x_ref, x_our = both(
        torch.randn(rows, cols, dtype=DTYPES[dtype_id]), hw, mojo_device
    )
    idx_ref, idx_our = both(
        torch.randint(0, (rows, cols)[dim], (rows, cols)), hw, mojo_device
    )
    bench.run(
        lambda: torch.gather(x_ref, dim, idx_ref),
        lambda: torch.gather(x_our, dim, idx_our),
        flops=float(rows * cols),
    )


@pytest.mark.parametrize("dtype_id", ("bf16", "f32"))
@pytest.mark.parametrize("shape_id", SELECT_SHAPES)
def test_index_select(
    shape_id: str, dtype_id: str, bench: Bench, hw: Hardware, mojo_device: torch.device
):
    rows, cols, selected, dim = SELECT_SHAPES[shape_id]
    x_ref, x_our = both(
        torch.randn(rows, cols, dtype=DTYPES[dtype_id]), hw, mojo_device
    )
    idx_ref, idx_our = both(
        torch.randint(0, (rows, cols)[dim], (selected,)), hw, mojo_device
    )
    bench.run(
        lambda: torch.index_select(x_ref, dim, idx_ref),
        lambda: torch.index_select(x_our, dim, idx_our),
        flops=float(selected * (cols if dim == 0 else rows)),
    )


@pytest.mark.parametrize("dtype_id", ("bf16", "f32"))
@pytest.mark.parametrize("shape_id", DIM_INDEX_SHAPES)
def test_scatter_add(
    shape_id: str, dtype_id: str, bench: Bench, hw: Hardware, mojo_device: torch.device
):
    """Indices are drawn uniformly over the indexed extent, so collisions are
    the norm — for D1 (64 columns, 262144 rows) every row's 64 writes land in
    64 slots. Both legs accumulate with atomics, so the ratio is a comparison
    of two atomic schemes, not of atomics against a sorted reduction."""
    rows, cols, dim = DIM_INDEX_SHAPES[shape_id]
    dtype = DTYPES[dtype_id]
    x_ref, x_our = both(torch.randn(rows, cols, dtype=dtype), hw, mojo_device)
    src_ref, src_our = both(torch.randn(rows, cols, dtype=dtype), hw, mojo_device)
    idx_ref, idx_our = both(
        torch.randint(0, (rows, cols)[dim], (rows, cols)), hw, mojo_device
    )
    bench.run(
        lambda: torch.scatter_add(x_ref, dim, idx_ref, src_ref),
        lambda: torch.scatter_add(x_our, dim, idx_our, src_our),
        flops=float(rows * cols),
    )


@pytest.mark.parametrize("dtype_id", ("bf16", "f32"))
@pytest.mark.parametrize("shape_id", SELECT_SHAPES)
def test_index_add(
    shape_id: str, dtype_id: str, bench: Bench, hw: Hardware, mojo_device: torch.device
):
    """index_add is index_select's mirror image (and its backward): the same
    broadcast index, the same geometry, writes instead of reads."""
    rows, cols, selected, dim = SELECT_SHAPES[shape_id]
    dtype = DTYPES[dtype_id]
    x_ref, x_our = both(torch.randn(rows, cols, dtype=dtype), hw, mojo_device)
    source_shape = (selected, cols) if dim == 0 else (rows, selected)
    s_ref, s_our = both(torch.randn(source_shape, dtype=dtype), hw, mojo_device)
    idx_ref, idx_our = both(
        torch.randint(0, (rows, cols)[dim], (selected,)), hw, mojo_device
    )
    bench.run(
        lambda: torch.index_add(x_ref, dim, idx_ref, s_ref),
        lambda: torch.index_add(x_our, dim, idx_our, s_our),
        flops=float(selected * (cols if dim == 0 else rows)),
    )


@pytest.mark.parametrize("dtype_id", ("bf16", "f32"))
@pytest.mark.parametrize("shape_id", PUT_ACC_SHAPES)
@pytest.mark.bench_op("_index_put_impl_")
def test_index_put_accumulate(
    shape_id: str, dtype_id: str, bench: Bench, hw: Hardware, mojo_device: torch.device
):
    rows, width, scattered = PUT_ACC_SHAPES[shape_id]
    dtype = DTYPES[dtype_id]
    x_ref, x_our = both(torch.randn(rows, width, dtype=dtype), hw, mojo_device)
    v_ref, v_our = both(torch.randn(scattered, width, dtype=dtype), hw, mojo_device)
    idx_ref, idx_our = both(torch.randint(0, rows, (scattered,)), hw, mojo_device)
    bench.run(
        lambda: torch.index_put(x_ref, [idx_ref], v_ref, True),
        lambda: torch.index_put(x_our, [idx_our], v_our, True),
        flops=float(scattered * width),
    )


@pytest.mark.parametrize("dtype_id", ("bf16", "f32"))
@pytest.mark.parametrize("shape_id", SELECT_SCATTER_SHAPES)
def test_select_scatter(
    shape_id: str, dtype_id: str, bench: Bench, hw: Hardware, mojo_device: torch.device
):
    outer, rows, cols = SELECT_SCATTER_SHAPES[shape_id]
    dtype = DTYPES[dtype_id]
    x_ref, x_our = both(torch.randn(outer, rows, cols, dtype=dtype), hw, mojo_device)
    s_ref, s_our = both(torch.randn(rows, cols, dtype=dtype), hw, mojo_device)
    bench.run(
        lambda: torch.select_scatter(x_ref, s_ref, 0, outer // 2),
        lambda: torch.select_scatter(x_our, s_our, 0, outer // 2),
        flops=float(x_ref.numel()),
    )


@pytest.mark.parametrize("dtype_id", ("bf16", "f32"))
@pytest.mark.parametrize("shape_id", DIM_INDEX_SHAPES)
@pytest.mark.parametrize("layout", ("sum", "prod", "amax", "mean_noself"))
def test_scatter_reduce(
    shape_id: str,
    dtype_id: str,
    layout: str,
    bench: Bench,
    hw: Hardware,
    mojo_device: torch.device,
):
    """scatter_add's geometry with each reduction: sum is the atomic add,
    prod / amax the compare-and-swap loop, and mean without self adds the
    identity fill, a count scatter and a division."""
    rows, cols, dim = DIM_INDEX_SHAPES[shape_id]
    dtype = DTYPES[dtype_id]
    reduce = layout.removesuffix("_noself")
    include_self = not layout.endswith("_noself")
    x_ref, x_our = both(torch.randn(rows, cols, dtype=dtype), hw, mojo_device)
    src_ref, src_our = both(torch.randn(rows, cols, dtype=dtype), hw, mojo_device)
    idx_ref, idx_our = both(
        torch.randint(0, (rows, cols)[dim], (rows, cols)), hw, mojo_device
    )
    bench.run(
        lambda: torch.scatter_reduce(
            x_ref, dim, idx_ref, src_ref, reduce, include_self=include_self
        ),
        lambda: torch.scatter_reduce(
            x_our, dim, idx_our, src_our, reduce, include_self=include_self
        ),
        flops=float(rows * cols),
    )


@pytest.mark.parametrize("dtype_id", ("bf16", "f32"))
@pytest.mark.parametrize("shape_id", SELECT_SHAPES)
@pytest.mark.parametrize("layout", ("amax", "mean"))
def test_index_reduce(
    shape_id: str,
    dtype_id: str,
    layout: str,
    bench: Bench,
    hw: Hardware,
    mojo_device: torch.device,
):
    """index_add's geometry with a reduction (more than 16 indices: the
    atomic route)."""
    rows, cols, selected, dim = SELECT_SHAPES[shape_id]
    dtype = DTYPES[dtype_id]
    x_ref, x_our = both(torch.randn(rows, cols, dtype=dtype), hw, mojo_device)
    source_shape = (selected, cols) if dim == 0 else (rows, selected)
    s_ref, s_our = both(torch.randn(source_shape, dtype=dtype), hw, mojo_device)
    idx_ref, idx_our = both(
        torch.randint(0, (rows, cols)[dim], (selected,)), hw, mojo_device
    )
    bench.run(
        lambda: torch.index_reduce(x_ref, dim, idx_ref, s_ref, layout),
        lambda: torch.index_reduce(x_our, dim, idx_our, s_our, layout),
        flops=float(selected * (cols if dim == 0 else rows)),
    )


# (vocab, dim, bags, bag length)
EMB_BAG_SHAPES: dict[str, tuple[int, int, int, int]] = {
    "V50304xD768_B1024xL48": (50304, 768, 1024, 48),
    "V1000xD64_B357xL7": (1000, 64, 357, 7),
}


@pytest.mark.parametrize("dtype_id", ("bf16", "f32"))
@pytest.mark.parametrize("shape_id", EMB_BAG_SHAPES)
@pytest.mark.parametrize("layout", ("sum", "mean", "max"))
@pytest.mark.bench_op("_embedding_bag_forward_only")
def test_embedding_bag(
    shape_id: str,
    dtype_id: str,
    layout: str,
    bench: Bench,
    hw: Hardware,
    mojo_device: torch.device,
):
    vocab, dim, bags, length = EMB_BAG_SHAPES[shape_id]
    w_ref, w_our = both(
        torch.randn(vocab, dim, dtype=DTYPES[dtype_id]), hw, mojo_device
    )
    i_ref, i_our = both(torch.randint(0, vocab, (bags * length,)), hw, mojo_device)
    o_ref, o_our = both(torch.arange(0, bags * length, length), hw, mojo_device)
    bench.run(
        lambda: F.embedding_bag(i_ref, w_ref, o_ref, mode=layout),
        lambda: F.embedding_bag(i_our, w_our, o_our, mode=layout),
        flops=float(bags * length * dim),
    )
