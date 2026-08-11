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

COVERS: dict[str, str] = {
    "aten::embedding": "test_embedding",
    "aten::embedding_dense_backward": "test_embedding_backward",
    "aten::index.Tensor": "test_index",
    "aten::scatter.src": "test_scatter_src",
    "aten::scatter.value": "test_scatter_value",
    "aten::select_scatter": "test_select_scatter",
}

SKIPPED: dict[str, str] = {}


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
