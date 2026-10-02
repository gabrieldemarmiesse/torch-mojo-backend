"""GEMM performance-regression benchmarks: mm / bmm / addmm / linear.

One pytest node per (shape, layout, dtype, op) case; the node id names the
kernel regime, so the list of failing tests is the list of regressed
regimes.  Select with ordinary pytest, e.g.:

    uv run pytest benchmarks/test_gemm.py -k "test_mm and S7 and TN and bf16"
    uv run pytest "benchmarks/test_gemm.py::test_mm[S1_4096x4096x4096-NN-bf16]"

Shapes come from the 2026-08 GEMM campaign.  S7 is the nanoGPT lm_head
weight gradient (GPT-2 124M, 48x1024 tokens, vocab padded to 50304):
tiny-M / huge-N / huge-K, the regime the campaign initially missed.

Layout letters follow BLAS: for A (M,K) x B (K,N), 'T' means the operand
is stored transposed and reached through a .t() view, exactly as
linear_backward produces it (NN = dgrad, NT = forward linear, TN = wgrad).

The "tf32" dtype id is float32 run at torch.set_float32_matmul_precision
("high") — the TF32 route — while "f32" is precision "highest".
"""

from __future__ import annotations

import contextlib
from collections.abc import Callable, Iterator

import pytest
import torch
from bench_lib.check import Bench
from bench_lib.hw import Hardware

# M, N, K
SHAPES = {
    "S1_4096x4096x4096": (4096, 4096, 4096),
    "S2_8192x2048x2048": (8192, 2048, 2048),
    "S3_2048x8192x2048": (2048, 8192, 2048),
    "S4_1024x1024x8192": (1024, 1024, 8192),  # deep-K
    "S5_357x789x333": (357, 789, 333),  # awkward, exercises edge tiles
    "S6_32768x768x768": (32768, 768, 768),  # tall-skinny
    "S7_768x50304x49152": (768, 50304, 49152),  # lm_head wgrad: tiny-M/huge-N/huge-K
    # GPT-2 XL (1600 embedding, B=16 x T=1024 tokens) projection sites, as
    # aten::linear sees them: M x out_features x in_features.  1600, 4800 and
    # 6400 are all 64 modulo 128 -- a residue no shape above has in ANY
    # dimension, and the one the gemm16 candidates (rolling NN, the widened TN
    # selection, the fused-bias NT) are gated on.  test_linear measures the
    # fused forward; test_linear_backward's two legs are the dX (M x K x N) and
    # dW (N x K x M) products, which is where the NN and TN candidates run.
    "S8_16384x4800x1600": (16384, 4800, 1600),  # c_attn
    "S9_16384x1600x1600": (16384, 1600, 1600),  # attn.c_proj
    "S10_16384x6400x1600": (16384, 6400, 1600),  # mlp.c_fc
    "S11_16384x1600x6400": (16384, 1600, 6400),  # mlp.c_proj
    # lm_head, the fifth site of the same step and bias-free in the model:
    # measured at parity with cuda before the candidates went in, and outside
    # every one of their gates (the aspect limits reject a 50304-wide
    # projection). A regression guard for the routes it already had, not a
    # candidate target. test_linear passes a bias on every shape, so this node
    # measures the biased call rather than the model's bias-free one.
    "S12_16384x50304x1600": (16384, 50304, 1600),
    # The TN persistent-rolling geometry dispatcher's own six shapes
    # (gemm16_tn_v4_kernels.mojo::_try_enqueue_tn_rolling_geom), as the
    # wgrad GEMM itself sees them: out_features x in_features x tokens (S8
    # above uses the SAME weights but as test_linear's tokens x out x in --
    # a "TN"-layout test_mm case here reduces over tokens directly, which is
    # what the standalone engagement measured with a bare torch.mm(g.t(), x)
    # and what test_linear_backward's dW leg bundles in with dX and dbias).
    "S13_4800x1600x8192": (4800, 1600, 8192),  # c_attn dW
    "S14_1600x1600x8192": (1600, 1600, 8192),  # attn.c_proj dW
    "S15_6400x1600x8192": (6400, 1600, 8192),  # mlp.c_fc dW
    "S16_1600x6400x8192": (1600, 6400, 8192),  # mlp.c_proj dW
    "S17_4800x1600x16384": (4800, 1600, 16384),  # c_attn dW, the deeper batch
    # Ragged M: 4808 % 64 == 8, so every aligned TN rung declines and only
    # the rolling dispatcher's TMA clip (m % 8 == 0) reaches it -- measured
    # 942 us (generic fallback) -> 149 us (rolling route) on H100 SXM.
    "S18_4808x1600x6592": (4808, 1600, 6592),
    # Low-occupancy TN regression guards (agent-C review of dff066a): an
    # aligned m (% 128 == 0) whose rolling-geometry work census barely
    # dents the available clusters must fall through to split-K / the
    # narrow-tile-192 rung / v3, not take the rolling dispatcher
    # unconditionally.  Measured H100 SXM, sm:1500 MHz, before the
    # dispatcher's occupancy decline -> after:
    "S19_768x768x12288": (768, 768, 12288),  # 35.3 -> 89.0 us (+152%)
    "S20_256x256x65536": (256, 256, 65536),  # 84.4 -> 443.1 us (+425%);
    # also a split-K case (m % 128 == 0, n % 256 == 0) that must be tried
    # before the rolling dispatcher, not after.
    "S21_128x128x8192": (128, 128, 8192),  # 37.1 -> 59.8 us (+61%)
}

# Batched, these four would be 2-3 TFLOP and several GB per leg for a regime
# the candidates never serve: the single-matrix GEMM entry is where they are
# installed, so every BMM item keeps its existing route by construction.
BMM_EXCLUDED = {
    "S7_768x50304x49152",  # ~40 GB per leg
    "S8_16384x4800x1600",
    "S9_16384x1600x1600",
    "S10_16384x6400x1600",
    "S11_16384x1600x6400",
    "S13_4800x1600x8192",
    "S14_1600x1600x8192",
    "S15_6400x1600x8192",
    "S16_1600x6400x8192",
    "S17_4800x1600x16384",
    "S18_4808x1600x6592",
    "S12_16384x50304x1600",  # tens of GB per leg batched
}

LAYOUTS = ("NN", "NT", "TN", "TT")

# id -> (torch dtype, float32 matmul precision during the case)
DTYPES = {
    "bf16": (torch.bfloat16, "highest"),
    "f16": (torch.float16, "highest"),
    "f32": (torch.float32, "highest"),
    "tf32": (torch.float32, "high"),
}

COVERS: dict[str, str] = {
    "aten::mm": "test_mm",
    "aten::bmm": "test_bmm",
    "aten::addmm": "test_addmm",
    "aten::linear": "test_linear",
    "aten::linear_backward": "test_linear_backward",
    # the BLAS family past the plain GEMMs (test_blas)
    "aten::addmm.dtype": "test_blas",
    "aten::_addmm_activation": "test_blas",
    "aten::addmv": "test_blas",
    "aten::addbmm": "test_blas",
    "aten::baddbmm": "test_blas",
    "aten::baddbmm.dtype": "test_blas",
    "aten::bmm.dtype": "test_blas",
    "aten::mm.dtype": "test_blas",
    "aten::_int_mm": "test_blas",
    "aten::_weight_int8pack_mm": "test_blas",
    "aten::_weight_int4pack_mm": "test_blas",
}

_BLAS_VARIANT = (
    "the same GEMM and epilogue as its functional overload, which test_blas "
    "measures, written into the caller's tensor"
)
SKIPPED: dict[str, str] = {
    "aten::addmm_": _BLAS_VARIANT,
    "aten::addmm.dtype_out": _BLAS_VARIANT,
    "aten::_addmm_activation.out": _BLAS_VARIANT,
    "aten::addmv.out": _BLAS_VARIANT,
    "aten::addmv_": _BLAS_VARIANT,
    "aten::addbmm.out": _BLAS_VARIANT,
    "aten::addbmm_": _BLAS_VARIANT,
    "aten::baddbmm.out": _BLAS_VARIANT,
    "aten::baddbmm_": _BLAS_VARIANT,
    "aten::baddbmm.dtype_out": _BLAS_VARIANT,
    "aten::bmm.dtype_out": _BLAS_VARIANT,
    "aten::mm.dtype_out": _BLAS_VARIANT,
    "aten::_int_mm.out": _BLAS_VARIANT,
    "aten::_convert_weight_to_int4pack": (
        "a one-time weight repack (a byte copy into the packed tensor), not "
        "on any step's critical path"
    ),
}

BMM_BATCH = 8
BMM_SHAPES = {
    f"{tag.split('_')[0]}_{BMM_BATCH}x{tag.split('_')[1]}": dims
    for tag, dims in SHAPES.items()
    if tag not in BMM_EXCLUDED
}


@contextlib.contextmanager
def matmul_precision(precision: str) -> Iterator[None]:
    previous = torch.get_float32_matmul_precision()
    torch.set_float32_matmul_precision(precision)
    try:
        yield
    finally:
        torch.set_float32_matmul_precision(previous)


def _mat(
    rows: int,
    cols: int,
    transposed: bool,
    dtype: torch.dtype,
    device: str | torch.device,
) -> torch.Tensor:
    if transposed:
        return torch.randn(cols, rows, dtype=dtype, device=device).t()
    return torch.randn(rows, cols, dtype=dtype, device=device)


def _bmat(
    batch: int,
    rows: int,
    cols: int,
    transposed: bool,
    dtype: torch.dtype,
    device: str | torch.device,
) -> torch.Tensor:
    if transposed:
        return torch.randn(batch, cols, rows, dtype=dtype, device=device).transpose(
            1, 2
        )
    return torch.randn(batch, rows, cols, dtype=dtype, device=device)


def _operand_pair(
    layout: str, m: int, n: int, k: int, dtype: torch.dtype, device: str | torch.device
) -> tuple[torch.Tensor, torch.Tensor]:
    a = _mat(m, k, layout[0] == "T", dtype, device)
    b = _mat(k, n, layout[1] == "T", dtype, device)
    return a, b


@pytest.mark.parametrize("dtype_id", DTYPES)
@pytest.mark.parametrize("layout", LAYOUTS)
@pytest.mark.parametrize("shape_id", SHAPES)
def test_mm(
    shape_id: str,
    layout: str,
    dtype_id: str,
    bench: Bench,
    hw: Hardware,
    mojo_device: torch.device,
):
    m, n, k = SHAPES[shape_id]
    dtype, precision = DTYPES[dtype_id]
    with matmul_precision(precision):
        a_ref, b_ref = _operand_pair(layout, m, n, k, dtype, hw.stock_device)
        a_our, b_our = _operand_pair(layout, m, n, k, dtype, mojo_device)
        bench.run(
            lambda: torch.mm(a_ref, b_ref),
            lambda: torch.mm(a_our, b_our),
            flops=2.0 * m * n * k,
        )


@pytest.mark.parametrize("dtype_id", DTYPES)
@pytest.mark.parametrize("layout", LAYOUTS)
@pytest.mark.parametrize("shape_id", BMM_SHAPES)
def test_bmm(
    shape_id: str,
    layout: str,
    dtype_id: str,
    bench: Bench,
    hw: Hardware,
    mojo_device: torch.device,
):
    m, n, k = BMM_SHAPES[shape_id]
    dtype, precision = DTYPES[dtype_id]
    with matmul_precision(precision):
        a_ref = _bmat(BMM_BATCH, m, k, layout[0] == "T", dtype, hw.stock_device)
        b_ref = _bmat(BMM_BATCH, k, n, layout[1] == "T", dtype, hw.stock_device)
        a_our = _bmat(BMM_BATCH, m, k, layout[0] == "T", dtype, mojo_device)
        b_our = _bmat(BMM_BATCH, k, n, layout[1] == "T", dtype, mojo_device)
        bench.run(
            lambda: torch.bmm(a_ref, b_ref),
            lambda: torch.bmm(a_our, b_our),
            flops=2.0 * BMM_BATCH * m * n * k,
        )


@pytest.mark.parametrize("dtype_id", DTYPES)
@pytest.mark.parametrize("layout", LAYOUTS)
@pytest.mark.parametrize("shape_id", SHAPES)
def test_addmm(
    shape_id: str,
    layout: str,
    dtype_id: str,
    bench: Bench,
    hw: Hardware,
    mojo_device: torch.device,
):
    m, n, k = SHAPES[shape_id]
    dtype, precision = DTYPES[dtype_id]
    with matmul_precision(precision):
        a_ref, b_ref = _operand_pair(layout, m, n, k, dtype, hw.stock_device)
        a_our, b_our = _operand_pair(layout, m, n, k, dtype, mojo_device)
        bias_ref = torch.randn(n, dtype=dtype, device=hw.stock_device)
        bias_our = torch.randn(n, dtype=dtype, device=mojo_device)
        bench.run(
            lambda: torch.addmm(bias_ref, a_ref, b_ref),
            lambda: torch.addmm(bias_our, a_our, b_our),
            flops=2.0 * m * n * k,
        )


# The BLAS family past mm / bmm / addmm: the alpha / beta epilogues, the
# `.dtype` overloads (half in, float32 out), the integer GEMM and the int8 /
# int4 weight-only GEMMs. One square and one awkward shape each.
BLAS_SHAPES = {tag: SHAPES[tag] for tag in ("S1_4096x4096x4096", "S5_357x789x333")}
BLAS_OPS = (
    "addmm_scaled",
    "addmm_dtype",
    "addmm_relu",
    "addmv",
    "addbmm",
    "baddbmm",
    "baddbmm_dtype",
    "bmm_dtype",
    "mm_dtype",
    "int_mm",
    "int8pack",
    "int4pack",
)
BLAS_DTYPES = {"bf16": torch.bfloat16, "f32": torch.float32}
# Ops whose stock CUDA kernel takes one dtype only.
_BLAS_ONE_DTYPE = {
    "addmm_dtype",
    "baddbmm_dtype",
    "bmm_dtype",
    "mm_dtype",
    "int_mm",
    "int4pack",
}


def _blas_case(
    op: str, m: int, n: int, k: int, dtype: torch.dtype, device: str | torch.device
) -> tuple[Callable[[], torch.Tensor], float]:
    """(callable, flops) of one BLAS-family case on `device`."""
    gen = torch.Generator().manual_seed(0)

    def rand(*shape: int, dt: torch.dtype = dtype) -> torch.Tensor:
        return torch.randn(*shape, generator=gen).to(dt).to(device)

    f32 = torch.float32
    if op == "addmm_scaled":
        c, a, b = rand(m, n), rand(m, k), rand(k, n)
        return (lambda: torch.addmm(c, a, b, beta=0.5, alpha=0.25)), 2.0 * m * n * k
    if op == "addmm_dtype":
        c, a, b = rand(n, dt=f32), rand(m, k), rand(k, n)
        return (lambda: torch.addmm(c, a, b, out_dtype=f32)), 2.0 * m * n * k
    if op == "addmm_relu":
        c, a, b = rand(n), rand(m, k), rand(k, n)
        return (lambda: torch.ops.aten._addmm_activation(c, a, b)), 2.0 * m * n * k
    if op == "addmv":
        c, a, v = rand(m), rand(m, k), rand(k)
        return (lambda: torch.addmv(c, a, v, beta=0.5, alpha=2)), 2.0 * m * k
    if op == "addbmm":
        c, a, b = rand(m, n), rand(4, m, k // 4), rand(4, k // 4, n)
        return (lambda: torch.addbmm(c, a, b, beta=0.5)), 2.0 * m * n * k
    if op in ("baddbmm", "baddbmm_dtype", "bmm_dtype"):
        bm = max(m // 8, 1)
        c = rand(8, bm, n, dt=f32 if op != "baddbmm" else dtype)
        a, b = rand(8, bm, k), rand(8, k, n)
        flops = 2.0 * 8 * bm * n * k
        if op == "baddbmm":
            return (lambda: torch.baddbmm(c, a, b, beta=0, alpha=0.125)), flops
        if op == "baddbmm_dtype":
            return (lambda: torch.baddbmm(c, a, b, out_dtype=f32)), flops
        return (lambda: torch.bmm(a, b, out_dtype=f32)), flops
    if op == "mm_dtype":
        a, b = rand(m, k), rand(k, n)
        return (lambda: torch.mm(a, b, out_dtype=f32)), 2.0 * m * n * k
    if op == "int_mm":
        a = torch.randint(-128, 128, (m, k - k % 8), dtype=torch.int8, generator=gen)
        b = torch.randint(
            -128, 128, (k - k % 8, n - n % 8), dtype=torch.int8, generator=gen
        )
        a, b = a.to(device), b.to(device)
        return (lambda: torch._int_mm(a, b)), 2.0 * m * n * k
    if op == "int8pack":
        x = rand(m, k)
        w = torch.randint(-128, 128, (n, k), dtype=torch.int8, generator=gen).to(device)
        s = rand(n)
        return (lambda: torch.ops.aten._weight_int8pack_mm(x, w, s)), 2.0 * m * n * k
    # int4pack: k a multiple of 256 (innerKTiles 8 x 16 and the 128-group)
    k4 = max(k - k % 256, 256)
    q = torch.randint(0, 256, (n - n % 8, k4 // 2), dtype=torch.uint8, generator=gen)
    w4 = torch.ops.aten._convert_weight_to_int4pack(q.to(device), 8)
    x = rand(m, k4)
    sz = rand(k4 // 128, n - n % 8, 2)
    return (
        lambda: torch.ops.aten._weight_int4pack_mm(x, w4, 128, sz)
    ), 2.0 * m * n * k4


@pytest.mark.parametrize("dtype_id", BLAS_DTYPES)
@pytest.mark.parametrize("op", BLAS_OPS)
@pytest.mark.parametrize("shape_id", BLAS_SHAPES)
def test_blas(
    shape_id: str,
    op: str,
    dtype_id: str,
    bench: Bench,
    hw: Hardware,
    mojo_device: torch.device,
):
    if op in _BLAS_ONE_DTYPE and dtype_id != "bf16":
        pytest.skip(f"{op}: stock CUDA takes bfloat16 operands only here")
    m, n, k = BLAS_SHAPES[shape_id]
    dtype = BLAS_DTYPES[dtype_id]
    ref, flops = _blas_case(op, m, n, k, dtype, hw.stock_device)
    ours, _ = _blas_case(op, m, n, k, dtype, mojo_device)
    bench.run(ref, ours, flops=flops)


# linear has no layout axis: x @ weight.t() + bias with weight stored
# (out_features, in_features) IS the NT regime by construction.
@pytest.mark.parametrize("dtype_id", DTYPES)
@pytest.mark.parametrize("shape_id", SHAPES)
def test_linear(
    shape_id: str, dtype_id: str, bench: Bench, hw: Hardware, mojo_device: torch.device
):
    m, n, k = SHAPES[shape_id]
    dtype, precision = DTYPES[dtype_id]
    with matmul_precision(precision):
        x_ref = torch.randn(m, k, dtype=dtype, device=hw.stock_device)
        w_ref = torch.randn(n, k, dtype=dtype, device=hw.stock_device)
        bias_ref = torch.randn(n, dtype=dtype, device=hw.stock_device)
        x_our = torch.randn(m, k, dtype=dtype, device=mojo_device)
        w_our = torch.randn(n, k, dtype=dtype, device=mojo_device)
        bias_our = torch.randn(n, dtype=dtype, device=mojo_device)
        bench.run(
            lambda: torch.nn.functional.linear(x_ref, w_ref, bias_ref),
            lambda: torch.nn.functional.linear(x_our, w_our, bias_our),
            flops=2.0 * m * n * k,
        )


NT_BIAS_SHAPES = {
    "NTB1_8192x1600x1600": (8192, 1600, 1600),
    "NTB2_8192x4800x1600": (8192, 4800, 1600),
    "NTB3_8192x6400x1600": (8192, 6400, 1600),
    "NTB4_8192x1600x6400": (8192, 1600, 6400),
    "NTB5_4096x1536x3072": (4096, 1536, 3072),
    "NTB6_357x789x544": (357, 789, 544),
}


@pytest.mark.bench_op("linear")
@pytest.mark.parametrize("dtype_id", ["bf16"])
@pytest.mark.parametrize("shape_id", NT_BIAS_SHAPES)
def test_linear_nt_bias(
    shape_id: str, dtype_id: str, bench: Bench, hw: Hardware, mojo_device: torch.device
):
    """The six NT+bias tuning regimes, including a masked output edge."""
    m, n, k = NT_BIAS_SHAPES[shape_id]
    dtype, _ = DTYPES[dtype_id]
    operands = [torch.randn(*shape, dtype=dtype) for shape in ((m, k), (n, k), (n,))]
    reference = [t.to(hw.stock_device) for t in operands]
    native = [t.to(mojo_device) for t in operands]
    bench.run(
        lambda: torch.nn.functional.linear(*reference),
        lambda: torch.nn.functional.linear(*native),
        flops=2.0 * m * n * k,
    )


# aten::linear_backward has no CUDA registration in stock PyTorch (CUDA
# decomposes linear to addmm, so its backward is matmul nodes); the stock
# reference leg therefore composes the exact equivalent three-op sequence:
# dgrad g @ w, wgrad g.t() @ x, bgrad g.sum(0).
@pytest.mark.parametrize("dtype_id", DTYPES)
@pytest.mark.parametrize("shape_id", SHAPES)
def test_linear_backward(
    shape_id: str, dtype_id: str, bench: Bench, hw: Hardware, mojo_device: torch.device
):
    m, n, k = SHAPES[shape_id]
    dtype, precision = DTYPES[dtype_id]
    with matmul_precision(precision):
        x_ref = torch.randn(m, k, dtype=dtype, device=hw.stock_device)
        w_ref = torch.randn(n, k, dtype=dtype, device=hw.stock_device)
        g_ref = torch.randn(m, n, dtype=dtype, device=hw.stock_device)
        x_our = torch.randn(m, k, dtype=dtype, device=mojo_device)
        w_our = torch.randn(n, k, dtype=dtype, device=mojo_device)
        g_our = torch.randn(m, n, dtype=dtype, device=mojo_device)

        def ref_leg() -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
            return g_ref @ w_ref, g_ref.t() @ x_ref, g_ref.sum(0)

        bench.run(
            ref_leg,
            lambda: torch.ops.aten.linear_backward(
                x_our, g_our, w_our, [True, True, True]
            ),
            flops=4.0 * m * n * k,
        )


# Medium-row training and ragged leading dimensions. These focused bf16
# cases complement the large-M campaign without multiplying every BMM and
# float32 configuration by the same shapes.
MEDIUM_BF16_MM_CASES = (
    ("R1_1024x1600x4800", "NT", (1024, 1600, 4800)),
    ("R2_1024x1600x1600", "NT", (1024, 1600, 1600)),
    ("R3_1024x1600x6400", "NT", (1024, 1600, 6400)),
    ("R4_1024x6400x1600", "NT", (1024, 6400, 1600)),
    ("R5_1600x4800x1024", "TN", (1600, 4800, 1024)),
    ("R6_1600x1600x1024", "TN", (1600, 1600, 1024)),
    ("R7_1600x6400x1024", "TN", (1600, 6400, 1024)),
    ("R8_6400x1600x1024", "TN", (6400, 1600, 1024)),
    ("R9_1024x50257x1600", "NT", (1024, 50257, 1600)),
    ("R10_1024x1600x50257", "NN", (1024, 1600, 50257)),
    ("R11_50257x1600x1024", "TN", (50257, 1600, 1024)),
    ("R12_512x768x8193", "NN", (512, 768, 8193)),
    ("R13_1009x1592x4095", "NN", (1009, 1592, 4095)),
)


@pytest.mark.bench_op("mm")
@pytest.mark.parametrize("dtype_id", ["bf16"])
@pytest.mark.parametrize("shape_id,layout,dims", MEDIUM_BF16_MM_CASES)
def test_mm_medium_rows(
    shape_id: str,
    layout: str,
    dims: tuple[int, int, int],
    dtype_id: str,
    bench: Bench,
    hw: Hardware,
    mojo_device: torch.device,
):
    m, n, k = dims
    dtype, _ = DTYPES[dtype_id]
    a_ref, b_ref = _operand_pair(layout, m, n, k, dtype, hw.stock_device)
    a_our, b_our = _operand_pair(layout, m, n, k, dtype, mojo_device)
    bench.run(
        lambda: torch.mm(a_ref, b_ref),
        lambda: torch.mm(a_our, b_our),
        flops=2.0 * m * n * k,
    )


@pytest.mark.bench_op("addmm")
@pytest.mark.parametrize("dtype_id", ["bf16"])
@pytest.mark.parametrize("layout", ["NN"])
@pytest.mark.parametrize(
    "shape_id,dims",
    [
        ("R1_1024x4800x1600", (1024, 4800, 1600)),
        ("R2_1024x1600x1600", (1024, 1600, 1600)),
        ("R3_1024x6400x1600", (1024, 6400, 1600)),
        ("R4_1024x1600x6400", (1024, 1600, 6400)),
        ("R5_1009x1617x1599", (1009, 1617, 1599)),
    ],
)
def test_addmm_medium_rows(
    shape_id: str,
    dims: tuple[int, int, int],
    layout: str,
    dtype_id: str,
    bench: Bench,
    hw: Hardware,
    mojo_device: torch.device,
):
    m, n, k = dims
    dtype, _ = DTYPES[dtype_id]
    a_ref, b_ref = _operand_pair(layout, m, n, k, dtype, hw.stock_device)
    a_our, b_our = _operand_pair(layout, m, n, k, dtype, mojo_device)
    bias_ref = torch.randn(n, dtype=dtype, device=hw.stock_device)
    bias_our = torch.randn(n, dtype=dtype, device=mojo_device)
    bench.run(
        lambda: torch.addmm(bias_ref, a_ref, b_ref),
        lambda: torch.addmm(bias_our, a_our, b_our),
        flops=2.0 * m * n * k,
    )
