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
# the whole allocation.  No awkward-length shape here, unlike the rest of this
# suite: these kernels move ONE element per thread with no vector path at all,
# so a length that is not a multiple of a vector width is not a distinct
# regime for them -- it is the only regime they have.
DIM_INDEX_SHAPES: dict[str, tuple[int, int, int]] = {
    "R_262144x64_D1": (262144, 64, 1),
    "R_4096x4096_D0": (4096, 4096, 0),
}
# (rows, row_width, scattered_rows, accumulate).  `accumulate` is a REGIME of
# index_put -- plain scattered stores vs atomic adds, two different kernels --
# so it is folded into the shape token rather than parametrized separately: a
# baseline key is (op, dtype, shape, layout) and nothing else, so two nodes
# sharing a shape token would overwrite each other's entry and the suite would
# flap between them on every run.
PUT_SHAPES: dict[str, tuple[int, int, int, bool]] = {
    "R_262144x64_S65536_set": (262144, 64, 65536, False),
    "R_262144x64_S65536_acc": (262144, 64, 65536, True),
}
# (rows, cols, selected, dim).  dim 0 is the row-gather fast path (the
# GatherRows kernel index.Tensor already uses); dim 1 is the general strided
# kernel, where each selected element is `rows` separate short reads.
SELECT_SHAPES: dict[str, tuple[int, int, int, int]] = {
    "R_262144x64_S1048576_D0": (262144, 64, 1048576, 0),
    "R_4096x4096_S8192_D1": (4096, 4096, 8192, 1),
}

COVERS: dict[str, str] = {
    "aten::_index_put_impl_": "test_index_put",
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
}

_SAME_KERNEL_OUT = (
    "out= overload of a benchmarked functional op: the same single kernel "
    "launch, writing a caller-supplied destination instead of an allocated "
    "one (no extra copy, unlike the _register_out wrappers)"
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
    "aten::scatter_add.out": _SAME_KERNEL_OUT,
    "aten::scatter_add_": _SAME_KERNEL_INPLACE,
}


@pytest.mark.parametrize("dtype_id", ("bf16", "f32"))
@pytest.mark.parametrize("shape_id", EMB_SHAPES)
def test_embedding(
    shape_id: str, dtype_id: str, bench: Bench, hw: Hardware, mojo_device: torch.device
) -> None:
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
) -> None:
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
) -> None:
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
) -> None:
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
) -> None:
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
) -> None:
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
) -> None:
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
) -> None:
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
) -> None:
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
@pytest.mark.parametrize("shape_id", PUT_SHAPES)
@pytest.mark.bench_op("_index_put_impl_")
def test_index_put(
    shape_id: str, dtype_id: str, bench: Bench, hw: Hardware, mojo_device: torch.device
) -> None:
    rows, width, scattered, accumulate = PUT_SHAPES[shape_id]
    dtype = DTYPES[dtype_id]
    x_ref, x_our = both(torch.randn(rows, width, dtype=dtype), hw, mojo_device)
    v_ref, v_our = both(torch.randn(scattered, width, dtype=dtype), hw, mojo_device)
    idx_ref, idx_our = both(torch.randint(0, rows, (scattered,)), hw, mojo_device)
    bench.run(
        lambda: torch.index_put(x_ref, [idx_ref], v_ref, accumulate),
        lambda: torch.index_put(x_our, [idx_our], v_our, accumulate),
        flops=float(scattered * width),
    )


@pytest.mark.parametrize("dtype_id", ("bf16", "f32"))
@pytest.mark.parametrize("shape_id", SELECT_SCATTER_SHAPES)
def test_select_scatter(
    shape_id: str, dtype_id: str, bench: Bench, hw: Hardware, mojo_device: torch.device
) -> None:
    outer, rows, cols = SELECT_SCATTER_SHAPES[shape_id]
    dtype = DTYPES[dtype_id]
    x_ref, x_our = both(torch.randn(outer, rows, cols, dtype=dtype), hw, mojo_device)
    s_ref, s_our = both(torch.randn(rows, cols, dtype=dtype), hw, mojo_device)
    bench.run(
        lambda: torch.select_scatter(x_ref, s_ref, 0, outer // 2),
        lambda: torch.select_scatter(x_our, s_our, 0, outer // 2),
        flops=float(x_ref.numel()),
    )
