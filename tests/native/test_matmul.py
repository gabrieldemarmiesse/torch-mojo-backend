"""End-to-end checks of the matmul group on the native backend: mm, bmm,
addmm, linear, linear_backward, addr and the convolution forward and
backward.

Public torch API only — every assertion compares the mojo device against the
same computation on CPU, and `assert_ran` proves the native op is what
produced it (rather than a decomposition into something else).
"""

import contextlib
import copy
import os
import pathlib
import re
import subprocess
import sys
from collections.abc import Callable

import numpy as np
import pytest
import torch

from tests.native.conftest import side_stream_or_skip, skip_if_metal
from torch_mojo_backend import aten_functions, get_accelerators, native
from torch_mojo_backend.native import device_module
from torch_mojo_backend.testing import CallChecker


@contextlib.contextmanager
def assert_ran(*op_names: str):
    """Assert that each aten op ran as a native boxed kernel in the block."""
    native.op_counting(True)
    before = native.op_counts()
    yield
    after = native.op_counts()
    for name in op_names:
        assert after.get(name, 0) > before.get(name, 0), (
            f"{name} did not run natively (counted: {sorted(after)})"
        )


@pytest.fixture
def mojo_h100(mojo_gpu):
    """H100 mojo device: the gemm16 / tf32 tensor-core bridges are gated to
    compute capability 9.0 and decline everywhere else."""
    accelerator = list(get_accelerators())[0]
    if accelerator.api != "cuda" or accelerator.architecture_name != "sm_90a":
        pytest.skip("the pure-Mojo tensor-core fast paths require an H100")
    return mojo_gpu


def _tol(dtype: torch.dtype) -> tuple[float, float]:
    """(atol, rtol): fp32 to fp32 accuracy, 16-bit to 16-bit rounding."""
    if dtype == torch.float32:
        return 1e-4, 1e-4
    return 5e-2, 5e-2


def _ref(*tensors: torch.Tensor) -> list[torch.Tensor]:
    """CPU float32 copies: the reference is always accumulated in fp32, the
    way every kernel in this family does."""
    return [t.cpu().float() for t in tensors]


# --- mm -----------------------------------------------------------------------


@pytest.mark.parametrize("dtype", [torch.float32, torch.bfloat16])
def test_mm(mojo_device, dtype, call_checker: CallChecker):
    call_checker.register(aten_functions.aten_mm)
    a = torch.randn(64, 128).to(dtype)
    b = torch.randn(128, 96).to(dtype)
    got = torch.mm(a.to(mojo_device), b.to(mojo_device)).cpu()
    assert got.dtype == dtype
    ra, rb = _ref(a, b)
    atol, rtol = _tol(dtype)
    torch.testing.assert_close((ra @ rb).to(dtype), got, atol=atol, rtol=rtol)


def test_mm_transposed_operands(mojo_device):
    """A `.t()` operand is a dense transposed layout, which every route reads
    in place — no materialized transpose."""
    a = torch.randn(48, 32)
    bt = torch.randn(64, 32)
    with assert_ran("aten::mm"):
        got = torch.mm(a.to(mojo_device), bt.to(mojo_device).t()).cpu()
    torch.testing.assert_close(got, a @ bt.t(), atol=1e-4, rtol=1e-4)
    with assert_ran("aten::mm"):
        got2 = torch.mm(a.to(mojo_device).t().contiguous().t(), bt.to(mojo_device).t())
    torch.testing.assert_close(got2.cpu(), a @ bt.t(), atol=1e-4, rtol=1e-4)


@pytest.mark.parametrize("shape", [(17, 37, 23), (1, 129, 31), (19, 35, 1)])
@pytest.mark.parametrize(
    "transpose_a,transpose_b",
    [(False, False), (True, False), (False, True), (True, True)],
)
def test_mm_float64(
    mojo_gpu: str, shape: tuple[int, int, int], transpose_a: bool, transpose_b: bool
):
    skip_if_metal(mojo_gpu, "Metal does not support float64")
    m, k, n = shape
    generator = torch.Generator().manual_seed(211)
    a = torch.randn(m, k, dtype=torch.float64, generator=generator)
    b = torch.randn(k, n, dtype=torch.float64, generator=generator)
    device_a = a.t().contiguous().to(mojo_gpu).t() if transpose_a else a.to(mojo_gpu)
    device_b = b.t().contiguous().to(mojo_gpu).t() if transpose_b else b.to(mojo_gpu)
    with assert_ran("aten::mm"):
        result = torch.mm(device_a, device_b)
    assert result.dtype == torch.float64
    torch.testing.assert_close(result.cpu(), torch.mm(a, b), atol=2e-12, rtol=2e-12)


def test_mm_float64_accumulation(mojo_gpu: str):
    skip_if_metal(mojo_gpu, "Metal does not support float64")
    a = torch.tensor([[1.0 + 2**-40, 1.0, -1.0]], dtype=torch.float64)
    b = torch.tensor([[1.0], [2**-40], [1.0]], dtype=torch.float64)
    result = torch.mm(a.to(mojo_gpu), b.to(mojo_gpu))
    torch.testing.assert_close(
        result.cpu(), torch.tensor([[2**-39]], dtype=torch.float64), atol=0, rtol=0
    )


@pytest.mark.parametrize(
    "shape",
    [
        (64, 64, 64),
        (96, 128, 64),
        (256, 256, 256),
        # deep K (k >= 2048 and k >= 2n) with k % 64 != 0: the second fp32
        # geometry that used to pick a single-MMA (16x16) warp tile
        (64, 2080, 64),
        (128, 4128, 96),
    ],
)
def test_mm_float32_tensor_core_regime(mojo_device, shape):
    """fp32 mm on the shapes that reach a matrix-core route (m >= 64, k % 32).

    On gfx942 the non-transposed route used to select a 16x32 warp tile, which
    is a single MMA tall for the fp32 16x16x4 matrix core and is miscompiled:
    a k-step vanished from part of the accumulator, so one output element in
    eight was wrong by O(1) (max abs error 24 at 256x256x256) while every
    shape below the route's cutoffs stayed exact.
    """
    m, k, n = shape
    a = torch.randn(m, k)
    b = torch.randn(k, n)
    # The bar grows like sqrt(k): the kernels accumulate in fp32 in k order,
    # so at k = 4128 a correct result sits ~2e-4 from torch's blocked sum on
    # a few elements (measured on gfx942 and on the CPU device alike).
    tol = 1e-4 * max(1.0, (k / 64) ** 0.5)
    with assert_ran("aten::mm"):
        got = torch.mm(a.to(mojo_device), b.to(mojo_device)).cpu()
    torch.testing.assert_close(got, a @ b, atol=tol, rtol=tol)

    # the same geometries through the fused-bias route
    bias = torch.randn(n)
    with assert_ran("aten::addmm"):
        got_bias = torch.addmm(
            bias.to(mojo_device), a.to(mojo_device), b.to(mojo_device)
        ).cpu()
    torch.testing.assert_close(got_bias, a @ b + bias, atol=tol, rtol=tol)


@pytest.mark.parametrize(
    "shape,layout,bias,offsets",
    [
        ((1024, 1600, 50257), "NN", False, (0, 0)),
        ((257, 520, 4095), "NN", False, (1, 1)),
        ((193, 264, 8193), "NN", False, (1, 1)),
        ((1009, 1592, 8193), "NN", False, (1, 4)),
        ((1009, 1592, 8223), "NN", False, (3, 0)),
        ((1024, 1600, 6400), "NT", False, (0, 0)),
        ((1600, 1600, 1024), "TN", False, (0, 0)),
        ((1600, 4800, 1024), "TN", False, (0, 0)),
        ((2047, 1600, 1024), "TN", False, (0, 0)),
        ((1009, 1592, 1024), "TN", False, (0, 0)),
        ((1009, 1592, 1024), "TN", False, (1, 3)),
        ((1600, 4800, 33), "TN", False, (0, 0)),
        ((1600, 4800, 1025), "TN", False, (0, 0)),
        ((1600, 4800, 40), "TN", False, (0, 0)),
        ((1600, 4800, 40), "NT", False, (0, 0)),
        ((1600, 4800, 40), "NN", False, (0, 0)),
        ((1009, 1617, 1599), "NN", True, (1, 1)),
        ((1024, 1600, 1600), "NN", True, (0, 0)),
        ((1009, 1032, 1600), "NN", True, (0, 0)),
        ((520, 776, 3200), "NN", True, (0, 0)),
        ((1024, 1600, 1600), "NN", True, (1, 0)),
    ],
)
def test_bf16_gemm_leading_dimensions(mojo_gpu, shape, layout, bias, offsets):
    """Odd strides, offset pointers, K tails, and matrix-core tile boundaries.

    Covers the gfx942 routes for an odd K (whole k tiles on the MFMA core,
    the tail in the reduction), the fused-bias NN routes (unsplit 128x128,
    split-K with the bias in the reduction, a misaligned operand), NT read in
    place, TN on 256x256, 128x128 and 64x256 tiles including odd leading
    dimensions (m = 2047, 1009) read without a copy, and ragged extents whose
    edge tiles are shifted back.  Sample an
    fp64 CPU product so the large contraction stays inexpensive, at the
    boundaries of the 32- to 256-element tiles and the last row/column;
    sentinels and input comparisons catch unintended writes.
    """
    m, n, k = shape
    generator = torch.Generator().manual_seed(731)

    def operand(rows, cols, transposed, offset):
        physical = (cols, rows) if transposed else (rows, cols)
        data = torch.randn(physical, generator=generator).to(torch.bfloat16)
        storage = torch.cat(
            [torch.full((offset,), 11, dtype=torch.bfloat16), data.reshape(-1)]
        )
        device_storage = storage.to(mojo_gpu)
        view = device_storage[offset:].view(physical)
        return (
            (data.t() if transposed else data),
            (view.t() if transposed else view),
            storage,
            device_storage,
        )

    a, da, a_storage, da_storage = operand(m, k, layout[0] == "T", offsets[0])
    b, db, b_storage, db_storage = operand(k, n, layout[1] == "T", offsets[1])
    bias_cpu = torch.randn(n, generator=generator).to(torch.bfloat16)
    with assert_ran("aten::addmm" if bias else "aten::mm"):
        result = (
            torch.addmm(bias_cpu.to(mojo_gpu), da, db) if bias else torch.mm(da, db)
        )
    edges = {0, 1, 31, 32, 63, 64, 127, 128, 255, 256}
    rows = sorted({r for r in edges if r < m} | {m // 2, m - 1})
    cols = sorted({c for c in edges if c < n} | {n // 2, n - 1})
    expected = a[rows].double() @ b[:, cols].double()
    if bias:
        expected += bias_cpu[cols].double()
    actual = result.cpu()[rows][:, cols]
    torch.testing.assert_close(
        actual, expected.to(torch.bfloat16), atol=0.03125, rtol=0.0078125
    )
    assert result.dtype == torch.bfloat16
    torch.testing.assert_close(da_storage.cpu(), a_storage, atol=0, rtol=0)
    torch.testing.assert_close(db_storage.cpu(), b_storage, atol=0, rtol=0)


@pytest.mark.parametrize(
    "shape,layout",
    [
        ((1600, 4800, 33), "TN"),
        ((1600, 4800, 1), "TN"),
        ((1600, 4800, 17), "TN"),
        ((1600, 4800, 1025), "TN"),
        ((1600, 4800, 40), "TN"),
        ((4096, 4096, 33), "TN"),
        ((1600, 4800, 33), "NN"),
        ((1600, 4800, 1025), "NN"),
        ((1600, 4800, 40), "NN"),
        ((1600, 4800, 33), "NT"),
        ((1600, 4800, 40), "NT"),
    ],
)
def test_bf16_gemm_k_tail_only(mojo_gpu, shape, layout):
    """Only the LAST k term is nonzero, so the exact product is all ones.

    A kernel that drops the K tail returns zeros: on gfx942 the masked TN
    tile loads k in pairs, and guarding a pair as a unit threw away row
    K - 1 whenever an odd K ends inside one (1, 17, 33, 1025 here; 40 is the
    even tail).  (1600, 4800) fills half of a 228-CU MI300A with 256x256
    tiles and (4096, 4096) all of it, so both reach the unsplit masked
    plan; the NN and NT rows cover the neighbouring routes' tails.
    """
    m, n, k = shape
    a = torch.zeros(m, k, dtype=torch.bfloat16)
    a[:, -1] = 1
    b = torch.zeros(k, n, dtype=torch.bfloat16)
    b[-1] = 1
    da = a.t().contiguous().to(mojo_gpu).t() if layout[0] == "T" else a.to(mojo_gpu)
    db = b.t().contiguous().to(mojo_gpu).t() if layout[1] == "T" else b.to(mojo_gpu)
    with assert_ran("aten::mm"):
        result = torch.mm(da, db).cpu()
    assert result.dtype == torch.bfloat16
    wrong = (result != 1).sum().item()
    assert wrong == 0, f"{wrong} of {m * n} outputs differ from 1"


@pytest.mark.parametrize("dtype", [torch.float32, torch.float16])
def test_mm_degenerate_dims(mojo_device, dtype):
    # n == 1 used to segfault the CPU library-matmul route (gemv special case
    # without a DeviceContext); m == 1 / k == 1 pin the library's other
    # special-case routes.
    for m, k, n in [(37, 129, 1), (1, 129, 64), (64, 1, 33), (1, 129, 1)]:
        a = torch.randn(m, k).to(dtype)
        b = torch.randn(k, n).to(dtype)
        got = torch.mm(a.to(mojo_device), b.to(mojo_device)).cpu()
        ra, rb = _ref(a, b)
        torch.testing.assert_close(got, (ra @ rb).to(dtype), atol=5e-2, rtol=5e-2)


def test_mm_unsupported_dtype_raises(mojo_device):
    """Integer matmul has no kernel in this family: the op declines, and the
    decline reaches python as NotImplementedError (never a wrong answer)."""
    a = torch.arange(6, dtype=torch.int64).reshape(2, 3).to(mojo_device)
    b = torch.arange(6, dtype=torch.int64).reshape(3, 2).to(mojo_device)
    with pytest.raises(NotImplementedError):
        torch.mm(a, b)


# --- bmm ----------------------------------------------------------------------


@pytest.mark.parametrize("dtype", [torch.float32, torch.bfloat16])
def test_bmm(mojo_device, dtype, call_checker: CallChecker):
    call_checker.register(aten_functions.aten_bmm)
    a = torch.randn(3, 64, 128).to(dtype)
    b = torch.randn(3, 128, 96).to(dtype)
    got = torch.bmm(a.to(mojo_device), b.to(mojo_device)).cpu()
    ra, rb = _ref(a, b)
    atol, rtol = _tol(dtype)
    torch.testing.assert_close(got, torch.bmm(ra, rb).to(dtype), atol=atol, rtol=rtol)


def test_bmm_expanded_batch(mojo_device):
    """A stride-0 batch dimension is one matrix shared by every item, which
    the batched routes address directly instead of materializing."""
    a = torch.randn(4, 12, 20)
    b = torch.randn(20, 8)
    # as_strided rather than expand(): only the view ops this group's own
    # tests may rely on (`aten::as_strided`) are guaranteed registered here.
    shared = torch.as_strided(b.to(mojo_device), (4, 20, 8), (0, 8, 1))
    with assert_ran("aten::bmm"):
        got = torch.bmm(a.to(mojo_device), shared).cpu()
    ref = torch.bmm(a, torch.as_strided(b, (4, 20, 8), (0, 8, 1)))
    torch.testing.assert_close(got, ref)


def test_bmm_transposed_rhs(mojo_device):
    a = torch.randn(2, 16, 24)
    b = torch.randn(2, 10, 24)
    got = torch.bmm(a.to(mojo_device), b.to(mojo_device).transpose(1, 2)).cpu()
    torch.testing.assert_close(
        got, torch.bmm(a, b.transpose(1, 2)), atol=1e-4, rtol=1e-4
    )


# --- addmm --------------------------------------------------------------------


@pytest.mark.parametrize("dtype", [torch.float32, torch.bfloat16])
def test_addmm(mojo_device, dtype, call_checker: CallChecker):
    call_checker.register(aten_functions.aten_addmm)
    bias = torch.randn(96).to(dtype)
    a = torch.randn(64, 128).to(dtype)
    b = torch.randn(128, 96).to(dtype)
    got = torch.addmm(bias.to(mojo_device), a.to(mojo_device), b.to(mojo_device)).cpu()
    rbias, ra, rb = _ref(bias, a, b)
    atol, rtol = _tol(dtype)
    torch.testing.assert_close(got, (ra @ rb + rbias).to(dtype), atol=atol, rtol=rtol)


@pytest.mark.parametrize("dtype", [torch.bfloat16, torch.float32])
@pytest.mark.parametrize(
    "shape", [(1024, 1600, 1600), (256, 1600, 6400), (512, 256, 256)]
)
def test_addmm_adds_its_bias_without_the_dispatcher(mojo_gpu, dtype, shape):
    """The shapes whose bias is added after an unbiased mm: that add is one
    launch into the product, not a second `aten::add` and a second buffer."""
    m, k, n = shape
    bias = (torch.randn(n) * 0.1).to(dtype)
    a = (torch.randn(m, k) * 0.1).to(dtype)
    b = (torch.randn(k, n) * 0.1).to(dtype)
    dev_bias, dev_a, dev_b = bias.to(mojo_gpu), a.to(mojo_gpu), b.to(mojo_gpu)
    with assert_no_bias_add():
        got = torch.addmm(dev_bias, dev_a, dev_b)
    ref = a.float() @ b.float() + bias.float()
    assert got.dtype == dtype
    assert got.stride() == (n, 1)
    assert _rel_err(got, ref) < _bf16_bound(k)
    # the same product without the bias, to show the bias landed at all
    plain = torch.mm(dev_a, dev_b)
    assert _rel_err(got - dev_bias, plain.cpu().float()) < _bf16_bound(k)
    out = torch.empty(m, n, dtype=dtype, device=mojo_gpu)
    assert torch.addmm(dev_bias, dev_a, dev_b, out=out) is out
    torch.testing.assert_close(out.cpu(), got.cpu(), rtol=0, atol=0)


@pytest.mark.parametrize("dtype", [torch.float16, torch.bfloat16])
@pytest.mark.parametrize("beta,alpha", [(1, 1), (0.6, 0.2), (0, 0.5)])
def test_addr_half_is_bit_identical_to_cpu(mojo_gpu: str, dtype, beta, alpha):
    """CPU rounds to the dtype after each product; a contracted `a + b*c`
    rounds once and was off on 15-23% of elements."""
    generator = torch.Generator().manual_seed(0)
    a = torch.randn(64, 97, dtype=dtype, generator=generator)
    b = torch.randn(64, dtype=dtype, generator=generator)
    c = torch.randn(97, dtype=dtype, generator=generator)
    got = torch.addr(
        a.to(mojo_gpu), b.to(mojo_gpu), c.to(mojo_gpu), beta=beta, alpha=alpha
    )
    assert torch.equal(got.cpu(), torch.addr(a, b, c, beta=beta, alpha=alpha))


def test_addmm_scaled_declines(mojo_device):
    """beta/alpha scaling is not implemented by this family; the decline is a
    NotImplementedError, not a silently dropped scale."""
    bias = torch.randn(8).to(mojo_device)
    a = torch.randn(4, 6).to(mojo_device)
    b = torch.randn(6, 8).to(mojo_device)
    with pytest.raises(NotImplementedError):
        torch.addmm(bias, a, b, beta=0.5)
    with pytest.raises(NotImplementedError):
        torch.addmm(bias, a, b, alpha=2.0)


@pytest.fixture
def mojo_gfx942(mojo_gpu):
    accelerator = list(get_accelerators())[0]
    if accelerator.architecture_name != "gfx942":
        pytest.skip("the NT MFMA bias route requires gfx942")
    return mojo_gpu


NT_MFMA_BIAS_SHAPES = [
    (8192, 1600, 1600),
    (8192, 4800, 1600),
    (8192, 6400, 1600),
    (8192, 1600, 6400),
    (4096, 1536, 3072),
    (357, 789, 544),
]


def _nt_bias_hash(shape: tuple[int, ...], seed: int) -> torch.Tensor:
    """The standalone harness's exact binary fractions, without an RNG."""
    size = int(np.prod(shape))
    x = np.arange(size, dtype=np.uint64) ^ np.uint64(
        (seed * 0x9E3779B97F4A7C15) % (1 << 64)
    )
    x = (x ^ (x >> np.uint64(30))) * np.uint64(0xBF58476D1CE4E5B9)
    x = (x ^ (x >> np.uint64(27))) * np.uint64(0x94D049BB133111EB)
    x ^= x >> np.uint64(31)
    values = ((x % np.uint64(17)).astype(np.float32) - 8) / 64
    return torch.from_numpy(values.reshape(shape)).bfloat16()


@pytest.mark.parametrize("m,n,k", NT_MFMA_BIAS_SHAPES)
def test_nt_mfma_bias_harness(mojo_gfx942, m, n, k):
    """Six measured regimes: exact sampled fp64 reference and full input checks.

    Every dot product is exact in fp32 for these binary fractions. Rounding
    before adding bias therefore fails the zero-tolerance comparison.
    """
    for seed in (1, 4):
        x = _nt_bias_hash((m, k), seed)
        w = _nt_bias_hash((n, k), seed + 1)
        bias = _nt_bias_hash((n,), seed + 2)
        dx, dw, db = [t.to(mojo_gfx942) for t in (x, w, bias)]
        with assert_ran("aten::linear"):
            got = torch.nn.functional.linear(dx, dw, db).cpu()
        for sample in range(72):
            row = (sample * 131 + sample * sample * 17) % m
            col = (sample * 397 + sample * sample * 29) % n
            if sample >= 64:
                row = min(m - 1, (sample - 64) * 128) if sample < 68 else m - 1
                col = n - 1 if sample < 68 else min(n - 1, (sample - 68) * 256)
            ref = (x[row].double() @ w[col].double() + bias[col].double()).bfloat16()
            torch.testing.assert_close(got[row, col], ref, atol=0, rtol=0)
        for actual, original in zip((dx, dw, db), (x, w, bias), strict=True):
            torch.testing.assert_close(actual.cpu(), original, atol=0, rtol=0)


@pytest.mark.parametrize(
    "m,n,k",
    [
        (1, 3, 7),
        (1, 17, 1),
        (3, 19, 15),
        (17, 35, 31),
        (7, 13, 40),
        (35, 67, 72),
        (67, 131, 333),
        (357, 789, 544),
    ],
)
@pytest.mark.parametrize("offset", [0, 1])
@pytest.mark.parametrize("op", ["linear", "addmm"])
def test_nt_mfma_bias_edges(mojo_gfx942, m, n, k, offset, op):
    """K tails, partial MFMA tiles and contiguous offset views must stay safe."""
    x = _nt_bias_hash((m, k), 4)
    w = _nt_bias_hash((n, k), 5)
    bias = _nt_bias_hash((n,), 6)
    originals = [
        torch.cat((torch.zeros(offset, dtype=t.dtype), t.flatten()))
        for t in (x, w, bias)
    ]
    storage = [t.to(mojo_gfx942) for t in originals]
    dx, dw, db = [
        torch.as_strided(t, ref.shape, ref.stride(), offset)
        for t, ref in zip(storage, (x, w, bias), strict=True)
    ]
    if offset:
        assert dx.data_ptr() % 16 != 0 and dw.data_ptr() % 16 != 0
    with assert_ran("aten::" + op):
        got = (
            torch.nn.functional.linear(dx, dw, db)
            if op == "linear"
            else torch.addmm(db, dx, dw.t())
        ).cpu()
    ref = (x.double() @ w.double().t() + bias.double()).bfloat16()
    torch.testing.assert_close(got, ref, atol=0, rtol=0)
    for actual, original in zip(storage, originals, strict=True):
        torch.testing.assert_close(actual.cpu(), original, atol=0, rtol=0)


def test_nt_mfma_bias_multiexponent(mojo_gfx942):
    """Nonuniform exponents also exercise cancellation in fp32 accumulators."""
    generator = torch.Generator().manual_seed(773)
    x = (
        torch.randn(137, 296, generator=generator)
        * torch.exp2(torch.randint(-4, 4, (137, 296), generator=generator).float())
    ).bfloat16()
    w = (
        torch.randn(259, 296, generator=generator)
        * torch.exp2(torch.randint(-4, 4, (259, 296), generator=generator).float())
    ).bfloat16()
    bias = torch.randn(259, generator=generator).bfloat16()
    dev = [t.to(mojo_gfx942) for t in (x, w, bias)]
    got = torch.nn.functional.linear(*dev).cpu()
    ref = (x.double() @ w.double().t() + bias.double()).bfloat16()
    torch.testing.assert_close(got, ref, rtol=8e-3, atol=2e-3)
    for actual, original in zip(dev, (x, w, bias), strict=True):
        torch.testing.assert_close(actual.cpu(), original, atol=0, rtol=0)


# --- the residue-64 bf16 candidates (GPT-2 XL linear sites) -------------------
#
# 1600 / 4800 / 6400 are 64 modulo 128, the regime the three measured
# candidates in gemm16_candidate_dispatch.mojo were fitted for: the rolling
# NN kernel (dX), the widened TN selection (dW) and the fused-bias NT kernel
# (forward).  M is scaled down from the model's 16384 to the gate's minimum
# so these stay unit tests; every N/K width is the model's, because those are
# what select the route.
# ---------------------------------------------------------------------------

# (name, out_features, in_features)
XL_SITES = [
    ("c_attn", 4800, 1600),
    ("attn.c_proj", 1600, 1600),
    ("mlp.c_fc", 6400, 1600),
    ("mlp.c_proj", 1600, 6400),
]
XL_M = 4096  # the fused NT gate's minimum m, and a multiple of 128


def _bf16_bound(k: int) -> float:
    """Max relative error of a K-long dot product with bf16 storage and fp32
    accumulation: bf16's epsilon is 2^-8 and the tile sums grow like sqrt(K).
    A failure here is a kernel bug, not a tolerance bug."""
    return 2**-8 * (k**0.5) * 0.5


def _rel_err(got: torch.Tensor, ref32: torch.Tensor) -> float:
    scale = ref32.abs().max().clamp(min=1.0)
    return float((got.cpu().float() - ref32).abs().max() / scale)


@contextlib.contextmanager
def assert_no_bias_add():
    """Assert no aten::add ran in the block: the fused NT kernel computes
    `A @ B.T + bias` in one launch, and the split-bias fallback it replaces
    would show up here as the broadcasting add it used to need."""
    native.op_counting(True)
    before = native.op_counts()
    yield
    after = native.op_counts()
    added = after.get("aten::add.Tensor", 0) - before.get("aten::add.Tensor", 0)
    assert added == 0, f"a second add ran on the fused output ({added} launches)"


@pytest.mark.parametrize("site,n,k", XL_SITES)
@pytest.mark.parametrize("bias", [True, False])
def test_gemm16_linear_residue64_sites(mojo_h100, site, n, k, bias):
    """Forward linear at each GPT-2 XL width, with and without a bias.

    With a bias this is the fused NT route and the result must be the single
    fp32-rounded product -- there is no second add to account for.
    """
    x = torch.randn(XL_M, k, dtype=torch.bfloat16)
    w = torch.randn(n, k, dtype=torch.bfloat16)
    b = torch.randn(n, dtype=torch.bfloat16) if bias else None
    dev_b = b.to(mojo_h100) if b is not None else None
    with assert_ran("aten::linear"):
        if bias:
            with assert_no_bias_add():
                got = torch.nn.functional.linear(
                    x.to(mojo_h100), w.to(mojo_h100), dev_b
                )
        else:
            got = torch.nn.functional.linear(x.to(mojo_h100), w.to(mojo_h100), dev_b)
    ref = torch.nn.functional.linear(
        x.float(), w.float(), b.float() if b is not None else None
    )
    assert got.dtype == torch.bfloat16
    assert _rel_err(got, ref) < _bf16_bound(k)


@pytest.mark.parametrize("site,n,k", XL_SITES)
def test_gemm16_linear_backward_residue64_sites(mojo_h100, site, n, k):
    """dX (rolling NN), dW (the widened TN selection) and the unchanged
    row-sum bias gradient, at the same widths."""
    x = torch.randn(XL_M, k, dtype=torch.bfloat16)
    w = torch.randn(n, k, dtype=torch.bfloat16)
    g = torch.randn(XL_M, n, dtype=torch.bfloat16)
    with assert_ran("aten::linear_backward"):
        dx, dw, db = torch.ops.aten.linear_backward(
            x.to(mojo_h100), g.to(mojo_h100), w.to(mojo_h100), [True, True, True]
        )
    assert _rel_err(dx, g.float() @ w.float()) < _bf16_bound(n)
    assert _rel_err(dw, g.float().t() @ x.float()) < _bf16_bound(XL_M)
    assert _rel_err(db, g.float().sum(0)) < _bf16_bound(XL_M)


def test_gemm16_fused_bias_awkward_shape_falls_back(mojo_h100):
    """357 x 789 x 1231: no dimension is a multiple of 64, so every candidate
    declines on metadata alone and the pre-existing route serves the call."""
    x = torch.randn(357, 1231, dtype=torch.bfloat16)
    w = torch.randn(789, 1231, dtype=torch.bfloat16)
    b = torch.randn(789, dtype=torch.bfloat16)
    got = torch.nn.functional.linear(x.to(mojo_h100), w.to(mojo_h100), b.to(mojo_h100))
    ref = torch.nn.functional.linear(x.float(), w.float(), b.float())
    assert _rel_err(got, ref) < _bf16_bound(1231)
    # and the matching backward, which reaches mm with ragged m/n/k
    g = torch.randn(357, 789, dtype=torch.bfloat16)
    dx, dw, db = torch.ops.aten.linear_backward(
        x.to(mojo_h100), g.to(mojo_h100), w.to(mojo_h100), [True, True, True]
    )
    assert _rel_err(dx, g.float() @ w.float()) < _bf16_bound(789)
    assert _rel_err(dw, g.float().t() @ x.float()) < _bf16_bound(357)
    assert _rel_err(db, g.float().sum(0)) < _bf16_bound(357)


def test_gemm16_fused_bias_needs_a_physical_nt_pair(mojo_h100):
    """A weight reached through a `.t()` view makes the linear physically NN,
    which is a different kernel's call: the fused NT route must decline it and
    the split-bias path must produce the same answer."""
    x = torch.randn(XL_M, 1600, dtype=torch.bfloat16)
    wt = torch.randn(1600, 4800, dtype=torch.bfloat16)  # (k, n), viewed as (n, k)
    b = torch.randn(4800, dtype=torch.bfloat16)
    w_view = wt.to(mojo_h100).t()
    assert not w_view.is_contiguous()
    got = torch.nn.functional.linear(x.to(mojo_h100), w_view, b.to(mojo_h100))
    ref = x.float() @ wt.float() + b.float()
    assert _rel_err(got, ref) < _bf16_bound(1600)


@pytest.mark.parametrize("dtype", [torch.float32, torch.float16])
def test_gemm16_fused_bias_is_bfloat16_only(mojo_h100, dtype):
    """fp32 (TF32 / strict SIMT) and float16 keep the routes they had: the
    candidate is gated to bfloat16, so both must still add the bias
    separately and still be right."""
    x = torch.randn(XL_M, 1600).to(dtype)
    w = torch.randn(4800, 1600).to(dtype)
    b = torch.randn(4800).to(dtype)
    got = torch.nn.functional.linear(x.to(mojo_h100), w.to(mojo_h100), b.to(mojo_h100))
    ref = x.float() @ w.float().t() + b.float()
    assert got.dtype == dtype
    bound = 1e-4 if dtype == torch.float32 else _bf16_bound(1600)
    assert _rel_err(got, ref) < bound


@pytest.mark.parametrize("site,n,k", XL_SITES)
def test_gemm16_addmm_residue64_sites(mojo_h100, site, n, k):
    """The same fused route through addmm, where the NT pair arrives as a
    transposed `mat2` rather than as a linear's weight."""
    a = torch.randn(XL_M, k, dtype=torch.bfloat16)
    bt = torch.randn(n, k, dtype=torch.bfloat16)
    bias = torch.randn(n, dtype=torch.bfloat16)
    with assert_ran("aten::addmm"):
        with assert_no_bias_add():
            got = torch.addmm(bias.to(mojo_h100), a.to(mojo_h100), bt.to(mojo_h100).t())
    ref = a.float() @ bt.float().t() + bias.float()
    assert _rel_err(got, ref) < _bf16_bound(k)


# --- TF32 NT on Hopper WGMMA (gemm16's float32 build) -------------------------
#
# Under a TF32 matmul precision an fp32 NT GEMM (A (m, k) row-major, B stored
# (n, k)) runs gemm16's warp-specialized WGMMA kernels compiled for a 4-byte
# operand; every other fp32 GEMM keeps the SM80-class tf32_matmul family.
#
# The oracle is EXACT: operands are multiples of 1/32 in [-1, 1), exact in
# TF32's 10 mantissa bits, so every product is a multiple of 1/1024 and every
# k-deep sum here stays inside fp32's exact range. Any mantissa loss, lost
# k-step or dropped tile is a nonzero error, not a tolerance call.
#
# Which route ran is read off the numerics: WGMMA reads the top 19 bits of an
# fp32 operand (it truncates to TF32), the SM80-class kernel rounds to nearest
# even, and strict fp32 does neither. 1 + 3 * 2**-12 is 0.75 of a TF32 ulp
# above 1, so it comes back as 1.0, 1 + 2**-10 and itself respectively.
# -------------------------------------------------------------------------------

_TF32_PROBE = 1.0 + 3 * 2.0**-12
_ROUTE_WGMMA = 1.0
_ROUTE_SM80 = 1.0 + 2.0**-10
_ROUTE_STRICT = _TF32_PROBE

TF32_WGMMA_SHAPES = {
    "aligned_256x512x256": (256, 512, 256),
    # ragged m, ragged (even) n, k not a multiple of BK = 32: the three
    # relaxations over the 16-bit routes' tile-multiple gate, at once
    "ragged_357x790x336": (357, 790, 336),
    "awkward_357x790x1020": (357, 790, 1020),
    # deep K, few output tiles: the split-K workspace + reduce route
    "deep_k_256x256x8192": (256, 256, 8192),
    # k = 4, the smallest the TMA rule admits: one partial k-tile, zero-filled
    "min_k_128x256x4": (128, 256, 4),
}

# Declined by the host gate: k * 4 not a multiple of 16 bytes (no TMA
# descriptor can describe it) or an odd n (the pair store would misalign).
TF32_DECLINED_SHAPES = {
    "awkward_357x789x1023": (357, 789, 1023),
    "unaligned_k_357x790x333": (357, 790, 333),
    "odd_n_128x257x256": (128, 257, 256),
}


@contextlib.contextmanager
def _matmul_precision(precision: str):
    previous = torch.get_float32_matmul_precision()
    torch.set_float32_matmul_precision(precision)
    try:
        yield
    finally:
        torch.set_float32_matmul_precision(previous)


def _tf32_exact(shape: tuple[int, ...], seed: int) -> torch.Tensor:
    """Multiples of 1/32 in [-1, 1): exact in TF32, so the reference is."""
    g = torch.Generator().manual_seed(seed)
    return torch.randint(-31, 32, shape, generator=g).float() / 32.0


def _nt_exact(m: int, n: int, k: int) -> tuple[torch.Tensor, torch.Tensor]:
    a = _tf32_exact((m, k), m * 7 + k)
    b = _tf32_exact((n, k), n * 11 + k)
    return a, b


def _exact_ref(a: torch.Tensor, b_nk: torch.Tensor, bias=None) -> torch.Tensor:
    out = a.double() @ b_nk.double().t()
    if bias is not None:
        out = out + bias.double()
    return out.float()


@pytest.mark.parametrize("shape_id", sorted(TF32_WGMMA_SHAPES))
def test_tf32_wgmma_nt_mm_is_bit_exact(mojo_h100, shape_id):
    m, n, k = TF32_WGMMA_SHAPES[shape_id]
    a, b = _nt_exact(m, n, k)
    with _matmul_precision("high"), assert_ran("aten::mm"):
        got = (a.to(mojo_h100) @ b.to(mojo_h100).t()).cpu()
    torch.testing.assert_close(got, _exact_ref(a, b), atol=0, rtol=0)


@pytest.mark.parametrize("shape_id", sorted(TF32_WGMMA_SHAPES))
@pytest.mark.parametrize("op", ["linear", "addmm", "linear_3d"])
def test_tf32_wgmma_nt_bias_is_bit_exact(mojo_h100, shape_id, op):
    """linear / addmm with a bias: the WGMMA product plus a separate bias add,
    exact on exact operands. linear_3d is a batched (rank-3) input, which the
    route flattens into one (batch * m, k) matrix without a copy."""
    m, n, k = TF32_WGMMA_SHAPES[shape_id]
    a, w = _nt_exact(m, n, k)
    bias = _tf32_exact((n,), n)
    ref = _exact_ref(a, w, bias)
    da, dw, db = a.to(mojo_h100), w.to(mojo_h100), bias.to(mojo_h100)
    with _matmul_precision("high"):
        if op == "linear":
            got = torch.nn.functional.linear(da, dw, db)
        elif op == "addmm":
            got = torch.addmm(db, da, dw.t())
        else:
            if m % 2:
                pytest.skip("an odd m has no two-batch split")
            got = torch.nn.functional.linear(da.view(2, m // 2, k), dw, db)
            ref = ref.view(2, m // 2, n)
    torch.testing.assert_close(got.cpu(), ref, atol=0, rtol=0)


@pytest.mark.parametrize("shape_id", sorted(TF32_DECLINED_SHAPES))
def test_tf32_declined_nt_shapes_stay_exact(mojo_h100, shape_id):
    m, n, k = TF32_DECLINED_SHAPES[shape_id]
    a, b = _nt_exact(m, n, k)
    bias = _tf32_exact((n,), n)
    with _matmul_precision("high"):
        got = (a.to(mojo_h100) @ b.to(mojo_h100).t()).cpu()
        got_bias = torch.nn.functional.linear(
            a.to(mojo_h100), b.to(mojo_h100), bias.to(mojo_h100)
        ).cpu()
    torch.testing.assert_close(got, _exact_ref(a, b), atol=0, rtol=0)
    torch.testing.assert_close(got_bias, _exact_ref(a, b, bias), atol=0, rtol=0)


def _probe_nt(m: int, n: int, k: int) -> tuple[torch.Tensor, torch.Tensor]:
    """A (m, k) whose column 0 is the probe and B (n, k) of ones in column 0:
    every output element is the probe as the route rounded it."""
    a = torch.zeros(m, k)
    a[:, 0] = _TF32_PROBE
    b = torch.zeros(n, k)
    b[:, 0] = 1.0
    return a, b


def _probe_value(out: torch.Tensor) -> float:
    flat = out.cpu().reshape(-1)
    assert torch.all(flat == flat[0]), "the route did not treat rows alike"
    return float(flat[0])


def _misaligned(t: torch.Tensor, device: str) -> torch.Tensor:
    """The same matrix at a 4-byte offset into its storage: not 16B-aligned,
    so no TMA descriptor may be built over it."""
    flat = torch.zeros(t.numel() + 1)
    flat[1:] = t.reshape(-1)
    view = flat.to(device)[1:].view(t.shape)
    assert view.data_ptr() % 16 != 0
    return view


@pytest.mark.parametrize(
    "case,expected",
    [
        ("mm_nt", _ROUTE_WGMMA),
        ("mm_nt_ragged", _ROUTE_WGMMA),
        ("linear", _ROUTE_WGMMA),
        ("linear_bias", _ROUTE_WGMMA),
        ("linear_3d", _ROUTE_WGMMA),
        ("addmm_nt", _ROUTE_WGMMA),
        ("mm_nt_k_not_4", _ROUTE_SM80),
        ("mm_nt_odd_n", _ROUTE_SM80),
        ("mm_nt_misaligned_a", _ROUTE_SM80),
        ("mm_nt_misaligned_b", _ROUTE_SM80),
        ("mm_nn", _ROUTE_SM80),
        ("mm_tn", _ROUTE_SM80),
        ("bmm_nt", _ROUTE_SM80),
    ],
)
def test_tf32_route_selection(mojo_h100, case, expected):
    """Which kernel family served each fp32 GEMM under precision "high"."""
    m, n, k = (357, 790, 1020) if case == "mm_nt_ragged" else (256, 512, 256)
    if case == "mm_nt_k_not_4":
        k = 255
    if case == "mm_nt_odd_n":
        n = 511
    a, b = _probe_nt(m, n, k)
    d = mojo_h100
    with _matmul_precision("high"):
        if case.startswith("mm_nt") and "misaligned" not in case:
            out = a.to(d) @ b.to(d).t()
        elif case == "mm_nt_misaligned_a":
            out = _misaligned(a, d) @ b.to(d).t()
        elif case == "mm_nt_misaligned_b":
            out = a.to(d) @ _misaligned(b, d).t()
        elif case == "linear":
            out = torch.nn.functional.linear(a.to(d), b.to(d))
        elif case == "linear_bias":
            out = torch.nn.functional.linear(a.to(d), b.to(d), torch.zeros(n).to(d))
        elif case == "linear_3d":
            out = torch.nn.functional.linear(a.to(d).view(2, m // 2, k), b.to(d))
        elif case == "addmm_nt":
            out = torch.addmm(torch.zeros(n).to(d), a.to(d), b.to(d).t())
        elif case == "mm_nn":
            out = a.to(d) @ b.t().contiguous().to(d)
        elif case == "mm_tn":
            out = a.t().contiguous().to(d).t() @ b.t().contiguous().to(d)
        else:
            out = torch.bmm(a.to(d).expand(2, m, k), b.to(d).expand(2, n, k).mT)
    assert _probe_value(out) == expected


@pytest.mark.parametrize("op", ["mm", "linear", "addmm"])
def test_tf32_wgmma_not_used_at_highest_precision(mojo_h100, op):
    """ "highest" (torch's default) is the user asking for real fp32: an NT
    GEMM the WGMMA route would take must not drop a single mantissa bit."""
    m, n, k = 256, 512, 256
    a, b = _probe_nt(m, n, k)
    d = mojo_h100
    assert torch.get_float32_matmul_precision() == "highest"
    if op == "mm":
        out = a.to(d) @ b.to(d).t()
    elif op == "linear":
        out = torch.nn.functional.linear(a.to(d), b.to(d), torch.zeros(n).to(d))
    else:
        out = torch.addmm(torch.zeros(n).to(d), a.to(d), b.to(d).t())
    assert _probe_value(out) == _ROUTE_STRICT


# --- the NT+bias 192x192 rolling kernel (GPT-2 XL forward linear sites) ------
#
# gemm16_candidate_dispatch.mojo's try_enqueue_candidate_nt_bias now tries the
# 192x192 rolling kernel (gemm16_rolling_kernels.mojo's kmaj_b + has_bias
# instantiation, `bf16_gemm_nt_bias_rolling_ws_m192n192_s3c2wg3g8`) ahead of
# the 128-row maybe_enqueue_gemm16_nt_bias_v4 fallback; both -- and the
# generic bf16 route below them for shapes neither accepts, e.g. m not a
# multiple of 128 -- fuse the bias into the accumulator, so assert_no_bias_add
# holds regardless of which one actually served a given shape.
#
# Shapes are exactly the ones the standalone engagement measured on H100 SXM
# (worst ratio 1.052 against cuBLAS), plus two smaller ones exercising the
# ragged-N tile boundary (1600 and 6400 are not multiples of BN=192; only
# 4800 is) at a cheaper M for test time.
# -------------------------------------------------------------------------------

NT_BIAS_192_SHAPES = [
    (8192, 4800, 1600),  # c_attn
    (8192, 1600, 1600),  # c_proj
    (8192, 6400, 1600),  # c_fc
    (8192, 1600, 6400),  # mlp_proj
    (16384, 4800, 1600),  # c_attn at the larger batch -- kept full-size
    (6600, 4800, 1600),  # ragged M (not a multiple of 128 or of 384)
    (4096, 1664, 1600),  # small, ragged N (1664 % 192 != 0)
    (4224, 4800, 1664),  # small, one tile past the regime's 4096 minimum
]


@pytest.mark.parametrize("m,n,k", NT_BIAS_192_SHAPES)
def test_gemm16_nt_bias_rolling_192(mojo_h100, m, n, k):
    """Forward linear in bf16 with a bias at the 192x192 rolling kernel's own
    shapes: fused in one launch regardless of which NT+bias route inside the
    regime actually serves it (see the module comment above)."""
    x = torch.randn(m, k, dtype=torch.bfloat16)
    w = torch.randn(n, k, dtype=torch.bfloat16)
    b = torch.randn(n, dtype=torch.bfloat16)
    with assert_ran("aten::linear"):
        with assert_no_bias_add():
            got = torch.nn.functional.linear(
                x.to(mojo_h100), w.to(mojo_h100), b.to(mojo_h100)
            )
    ref = torch.nn.functional.linear(x.float(), w.float(), b.float())
    assert got.dtype == torch.bfloat16
    assert _rel_err(got, ref) < _bf16_bound(k)


@pytest.mark.parametrize("m,n,k", NT_BIAS_192_SHAPES)
def test_gemm16_nt_bias_rolling_192_bias_only(mojo_h100, m, n, k):
    """Zero products: the output IS the bias, so a dropped, shifted or
    misindexed bias column shows up exactly, across every column-tile
    boundary. (The random-input test above tolerates bf16 accumulation error
    of the size of a unit bias.)"""
    x = torch.zeros(m, k, dtype=torch.bfloat16)
    w = torch.zeros(n, k, dtype=torch.bfloat16)
    b = (torch.arange(n, dtype=torch.float32) % 251 - 125).to(torch.bfloat16)
    with assert_ran("aten::linear"):
        with assert_no_bias_add():
            got = torch.nn.functional.linear(
                x.to(mojo_h100), w.to(mojo_h100), b.to(mojo_h100)
            )
    torch.testing.assert_close(got.cpu(), b.expand(m, n), rtol=0, atol=0)


# --- the TN persistent-rolling geometry dispatcher (GPT-2 XL dW sites) -------
#
# gemm16_tn_v4_kernels.mojo's try_enqueue_gemm16_gemm_tn_v4 now routes a
# multi-wave TN (weight-gradient) GEMM to one of three tuned geometries of
# the shared persistent-rolling body (192x192 / 128x256 / 128x192, chosen by
# a runtime cost model -- see _try_enqueue_tn_rolling_geom's module comment
# in that file), replacing the old fixed-128x256 rung, and clips a ragged M
# (m % 8 == 0) via TMA instead of declining it the way the split-K and
# narrow-tile rungs below it still do.
#
# Shapes are the standalone engagement's six (out_features, in_features,
# tokens); tokens is scaled down on all but the first the way this file's
# other site tables scale down whatever axis dominates test time (XL_M
# above) -- the geometry choice is a function of out_features/in_features
# and the GPU's SM count only, never of tokens.
# -------------------------------------------------------------------------------

TN_ROLLING_SHAPES = [
    (4800, 1600, 8192),  # c_attn dW -- full-size tokens, the flagship shape
    (1600, 1600, 1024),  # c_proj dW
    (6400, 1600, 1024),  # c_fc dW
    (1600, 6400, 1024),  # mlp_proj dW
    (4800, 1600, 2048),  # c_attn dW at the model's deeper reduction (16384)
    (4808, 1600, 1024),  # ragged M -- not a multiple of 64; measured 942 us
    # (generic fallback) -> 149 us (rolling route) at full size, job 250131
]


@pytest.mark.parametrize("out_features,in_features,tokens", TN_ROLLING_SHAPES)
def test_gemm16_tn_rolling_geometry_dispatch(
    mojo_h100, out_features, in_features, tokens
):
    """dW through linear_backward at the rolling dispatcher's own shapes."""
    x = torch.randn(tokens, in_features, dtype=torch.bfloat16)
    w = torch.randn(out_features, in_features, dtype=torch.bfloat16)
    g = torch.randn(tokens, out_features, dtype=torch.bfloat16)
    with assert_ran("aten::linear_backward"):
        _dx, dw, _db = torch.ops.aten.linear_backward(
            x.to(mojo_h100), g.to(mojo_h100), w.to(mojo_h100), [True, True, True]
        )
    ref = g.float().t() @ x.float()
    assert dw.dtype == torch.bfloat16
    assert _rel_err(dw, ref) < _bf16_bound(tokens)


def test_gemm16_tn_rolling_small_ragged_m(mojo_h100):
    """A small ragged M (64 <= m < 1600) was not part of the standalone
    engagement's measured sweep (an A2 review finding); this is the
    smallest case that still clears the dispatcher's m >= 64 floor."""
    out_features, in_features, tokens = 136, 1600, 512
    x = torch.randn(tokens, in_features, dtype=torch.bfloat16)
    w = torch.randn(out_features, in_features, dtype=torch.bfloat16)
    g = torch.randn(tokens, out_features, dtype=torch.bfloat16)
    with assert_ran("aten::linear_backward"):
        _dx, dw, _db = torch.ops.aten.linear_backward(
            x.to(mojo_h100), g.to(mojo_h100), w.to(mojo_h100), [True, True, True]
        )
    ref = g.float().t() @ x.float()
    assert _rel_err(dw, ref) < _bf16_bound(tokens)


# n=320 divides none of 128/192/256 -- no fallback rung (split-K, narrow-
# tile-192, v3's aligned or small-tile routes) exists anywhere in the TN
# ladder, so the occupancy decline must keep the rolling dispatcher
# regardless of how low its own modeled occupancy is (a Codex review
# finding: it used to fall all the way to the generic non-TMA route).
# m=64, n=9600 is the opposite failure mode of the same decline: n % 128
# == 0 makes v3's small-tile route (bm=64, an exact fit) eligible, and on
# 132 SMs this shape clears the three-quarters occupancy floor outright
# (50 of 66 clusters) -- so occupancy alone was not enough to stop the
# rolling dispatcher from choosing a 128-row geometry that pads m=64 to
# 128 (4x the small-tile route's padding-free work), also a Codex finding.
# (64, 320, 4096) and (2048, 320, 4096) are the two sides of the third
# finding (agent C): with no fallback rung the generic route is still the
# right answer for a small output above the ladder's 128-row floor -- so
# (256, 320, 4096) now DECLINES to it (33.4 -> 20.0 us) while these two,
# one under the floor and one far past _V4_TN_ROLL_MIN_AREA, keep rolling
# (0.26x and 0.35x of generic). All four must be numerically right
# whichever route the dispatcher picks for them.
TN_ROLLING_ESCAPE_HATCH_SHAPES = [
    (256, 320, 4096),
    (64, 9600, 64),
    (64, 320, 4096),
    (2048, 320, 4096),
]


@pytest.mark.parametrize(
    "out_features,in_features,tokens", TN_ROLLING_ESCAPE_HATCH_SHAPES
)
def test_gemm16_tn_rolling_occupancy_decline_escape_hatch(
    mojo_h100, out_features, in_features, tokens
):
    """Correctness at the two shapes the occupancy decline's escape hatch
    (_try_enqueue_tn_rolling_geom's docstring) must get right: one with no
    fallback rung at all, one where the fallback exists but the rolling
    dispatcher's own geometry would pad m severely worse than it does."""
    x = torch.randn(tokens, in_features, dtype=torch.bfloat16)
    w = torch.randn(out_features, in_features, dtype=torch.bfloat16)
    g = torch.randn(tokens, out_features, dtype=torch.bfloat16)
    with assert_ran("aten::linear_backward"):
        _dx, dw, _db = torch.ops.aten.linear_backward(
            x.to(mojo_h100), g.to(mojo_h100), w.to(mojo_h100), [True, True, True]
        )
    ref = g.float().t() @ x.float()
    assert _rel_err(dw, ref) < _bf16_bound(tokens)


def test_gemm16_tn_rolling_m4800_boundary_guard(mojo_h100):
    """m = 4800 lands exactly on the 192x192/cluster-2 geometry's macro-row
    boundary: 4800 % (192 * 2) == 192, so the second cluster rank's box for
    the grid's last macro row starts exactly at m -- entirely out of bounds
    for the A load and the C store, not merely a partial tile (an A2 review
    finding on this engagement, documented in _try_enqueue_tn_rolling_geom's
    docstring, gemm16_tn_v4_kernels.mojo).

    Geometry selection is SM-count dependent (see
    _try_enqueue_tn_rolling_geom's cost model); this (out, in) pair picks
    the 192x192 geometry on an H100 SXM (132 SMs), the hardware this
    engagement was measured on -- on a different SM count the correctness
    check below still holds for whichever geometry actually ran.

    What this can and cannot detect (a Codex review finding): `out=`
    (ops_matmul.mojo's op_mm_out / _store_out) computes into a freshly
    allocated, exactly (m, n)-sized temporary and then copy_strided_intos
    it into `view`; that copy is itself bounded by (m, n), so the nonzero
    canary below guards the COPY against overrunning `view`, not the GEMM
    kernel's own TMA store against overrunning ITS temporary -- the canary
    rows never border the kernel's real destination memory, so an
    out-of-bounds *write* by the kernel itself would not reach them (nor
    would a zero canary catch an out-of-bounds write of zero, which is why
    this one is not zero). Real evidence for the kernel's own store comes
    from a clean `compute-sanitizer --tool memcheck` run over the direct
    (non-`out=`) `torch.mm(grad.t(), x)` path at this m and the ragged
    m=4808, cited in this change's commit message. This test remains a
    regression guard for the `out=` copy path (a real thing that could
    still break on its own), not a substitute for that sanitizer evidence.
    """
    m, n, k = 4800, 1600, 1024
    canary = -12345.0
    base = torch.full((m + 192, n), canary, dtype=torch.bfloat16, device=mojo_h100)
    view = base[:m]
    g = torch.randn(k, m, dtype=torch.bfloat16)
    x = torch.randn(k, n, dtype=torch.bfloat16)
    torch.mm(g.to(mojo_h100).t(), x.to(mojo_h100), out=view)
    ref = g.float().t() @ x.float()
    assert _rel_err(view, ref) < _bf16_bound(k)
    expected_guard = torch.full((192, n), canary, dtype=torch.bfloat16)
    assert torch.equal(base[m:].cpu(), expected_guard), "the out= copy wrote past m"


# --- the out= overloads (TorchInductor's extern kernels) ----------------------


def _out_case(
    op: str, device: str
) -> tuple[Callable[[torch.Tensor], torch.Tensor], torch.Tensor]:
    """(call taking the `out=` tensor, CPU reference) for one out= overload.

    Inductor reaches mm / bmm / addmm through exactly these three
    (`extern_kernels.mm(a, b, out=buf)`), so each gets the same checks.
    """
    if op == "mm":
        a, b = torch.randn(12, 20), torch.randn(20, 8)
        return lambda out: torch.mm(a.to(device), b.to(device), out=out), a @ b
    if op == "bmm":
        a, b = torch.randn(3, 12, 20), torch.randn(3, 20, 8)
        return lambda out: torch.bmm(a.to(device), b.to(device), out=out), a @ b
    # a 1-D bias: the fused MatmulBiasSpec kernel reads one row-broadcast
    # vector, and a 2-D `self` sends addmm down a route this group declines.
    c, a, b = torch.randn(8), torch.randn(12, 20), torch.randn(20, 8)
    return (
        lambda out: torch.addmm(c.to(device), a.to(device), b.to(device), out=out),
        torch.addmm(c, a, b),
    )


def _mojo_devices() -> list[str]:
    return [f"mojo:{i}" for i in range(len(list(get_accelerators())))]


@pytest.mark.parametrize("op", ["mm", "bmm", "addmm"])
def test_out_writes_the_callers_tensor(mojo_device, op):
    run, ref = _out_case(op, mojo_device)
    out = torch.empty(ref.shape, device=mojo_device)
    with assert_ran(f"aten::{op}.out"):
        got = run(out)
    assert got.data_ptr() == out.data_ptr()
    torch.testing.assert_close(out.cpu(), ref, atol=1e-4, rtol=1e-4)


@pytest.mark.parametrize("op", ["mm", "bmm", "addmm"])
def test_out_with_the_wrong_dtype_raises_torchs_message(mojo_device, op):
    """torch's generated `resize_out` requires the `out=` tensor to ALREADY
    hold the result dtype -- it never casts, so a float result can never be
    truncated into an integer buffer. The text is taken from the same call on
    CPU so the two backends cannot drift apart."""
    for wrong in (torch.float64, torch.int64):
        run_cpu, _ = _out_case(op, "cpu")
        with pytest.raises(RuntimeError) as cpu_err:
            run_cpu(torch.empty(0, dtype=wrong))
        run_mojo, _ = _out_case(op, mojo_device)
        with pytest.raises(RuntimeError, match=re.escape(str(cpu_err.value))):
            run_mojo(torch.empty(0, dtype=wrong, device=mojo_device))


def test_mm_out_does_not_cast_bfloat16_up_into_a_float_out(mojo_device):
    a, b = torch.randn(8, 16).bfloat16(), torch.randn(16, 4).bfloat16()
    with pytest.raises(RuntimeError) as cpu_err:
        torch.mm(a, b, out=torch.empty(8, 4))
    with pytest.raises(RuntimeError, match=re.escape(str(cpu_err.value))):
        torch.mm(
            a.to(mojo_device),
            b.to(mojo_device),
            out=torch.empty(8, 4, device=mojo_device),
        )


@pytest.mark.parametrize("op", ["mm", "bmm", "addmm"])
def test_out_on_the_cpu_raises(mojo_device, op):
    run, ref = _out_case(op, mojo_device)
    msg = f"Expected out tensor to have device {mojo_device}, but got cpu instead"
    with pytest.raises(RuntimeError, match=re.escape(msg)):
        run(torch.empty(ref.shape))


@pytest.mark.gpu
def test_out_on_another_mojo_device_raises():
    """A cross-device `out=` would launch the copy in the destination's
    context with no ordering against the stream that produced the result;
    torch rejects it in `resize_out` and so must this backend."""
    devices = _mojo_devices()
    if len(devices) < 2:
        pytest.skip("needs two mojo devices")
    a, b = torch.randn(12, 20), torch.randn(20, 8)
    msg = (
        f"Expected out tensor to have device {devices[0]}, but got {devices[1]} instead"
    )
    with pytest.raises(RuntimeError, match=re.escape(msg)):
        torch.mm(
            a.to(devices[0]),
            b.to(devices[0]),
            out=torch.empty(12, 8, device=devices[1]),
        )


@pytest.mark.parametrize("op", ["mm", "bmm", "addmm"])
def test_out_of_the_wrong_shape_is_resized(mojo_device, op):
    """`resize_output`: an empty `out` is the common Inductor case, and a
    non-empty one of the wrong shape is resized too (torch warns and does it
    anyway)."""
    for start in ((0,), (3, 3)):
        run, ref = _out_case(op, mojo_device)
        out = torch.zeros(start, device=mojo_device)
        run(out)
        assert tuple(out.shape) == tuple(ref.shape)
        torch.testing.assert_close(out.cpu(), ref, atol=1e-4, rtol=1e-4)


def test_out_of_the_right_shape_keeps_its_own_strides(mojo_device):
    """A correctly shaped `out` is never re-laid-out: a slice of a bigger
    tensor is written where it lives, leaving the rest of the base alone."""
    base = torch.zeros(12, 16, device=mojo_device)
    view = base[:, :8]
    a, b = torch.randn(12, 20), torch.randn(20, 8)
    torch.mm(a.to(mojo_device), b.to(mojo_device), out=view)
    assert view.stride() == (16, 1)
    torch.testing.assert_close(view.cpu(), a @ b, atol=1e-4, rtol=1e-4)
    assert bool((base[:, 8:] == 0).all()), "the resize scribbled over the base"


def test_mm_out_under_allocator_churn_on_a_side_stream(mojo_device):
    """`out=` on a side stream, 48 times, each with fresh inputs and a fresh
    product of exactly the recycled size, with the host never syncing inside
    the loop -- the hardest case for the helper that copies the product into
    the caller's tensor.

    It is a guard, not a proof of the lifetime fix in `_store_out`. Measured
    on an H100: with the `_ = held^` keepalive deleted this still passes,
    because both the (stream-ordered) release of the product and the copy out
    of it are enqueued on the same stream, and the stream runs the copy before
    any later kernel can write the recycled block. The keepalive is still
    required: nothing in an op may depend on that allocator detail.
    """
    side = side_stream_or_skip(mojo_device)
    outs, refs = [], []
    with device_module.stream(side):
        for _ in range(48):
            a, b = torch.randn(96, 128), torch.randn(128, 64)
            out = torch.empty(96, 64, device=mojo_device)
            torch.mm(a.to(mojo_device), b.to(mojo_device), out=out)
            outs.append(out)
            refs.append(a @ b)
    torch.accelerator.synchronize()
    for i, (out, ref) in enumerate(zip(outs, refs, strict=True)):
        torch.testing.assert_close(
            out.cpu(),
            ref,
            atol=1e-3,
            rtol=1e-3,
            msg=lambda m, i=i: f"iteration {i}: {m}",
        )


# --- linear -------------------------------------------------------------------


@pytest.mark.parametrize("dtype", [torch.float32, torch.bfloat16])
@pytest.mark.parametrize("shape", [(64, 128), (2, 64, 128), (128,)])
def test_linear(mojo_device, dtype, shape, call_checker: CallChecker):
    call_checker.register(aten_functions.aten_linear)
    x = torch.randn(*shape).to(dtype)
    w = torch.randn(96, 128).to(dtype)
    bias = torch.randn(96).to(dtype)
    for b in (None, bias):
        got = torch.nn.functional.linear(
            x.to(mojo_device),
            w.to(mojo_device),
            None if b is None else b.to(mojo_device),
        ).cpu()
        rx, rw = _ref(x, w)
        ref = torch.nn.functional.linear(rx, rw, None if b is None else b.cpu().float())
        assert got.shape == ref.shape
        atol, rtol = _tol(dtype)
        torch.testing.assert_close(got, ref.to(dtype), atol=atol, rtol=rtol)


def test_linear_is_not_decomposed_to_addmm(mojo_device):
    """nn.Linear reaches aten::linear, not addmm: that is what keeps its
    backward the fused aten::linear_backward node."""
    x = torch.randn(5, 16)
    layer = torch.nn.Linear(16, 24)
    native.op_counting(True)
    before = native.op_counts()
    got = torch.nn.functional.linear(
        x.to(mojo_device),
        layer.weight.detach().to(mojo_device),
        layer.bias.detach().to(mojo_device),
    ).cpu()
    after = native.op_counts()
    assert after.get("aten::linear", 0) > before.get("aten::linear", 0)
    assert after.get("aten::addmm", 0) == before.get("aten::addmm", 0)
    torch.testing.assert_close(got, layer(x), atol=1e-4, rtol=1e-4)


def test_linear_empty_features(mojo_device):
    """The rank-1 vector route's own edge cases: a zero-width output, and a
    zero-length input whose result is just the bias."""
    x0 = torch.randn(0)
    w0 = torch.randn(7, 0)
    bias = torch.randn(7)
    got = torch.nn.functional.linear(
        x0.to(mojo_device), w0.to(mojo_device), bias.to(mojo_device)
    ).cpu()
    torch.testing.assert_close(got, bias)
    got_nobias = torch.nn.functional.linear(
        x0.to(mojo_device), w0.to(mojo_device)
    ).cpu()
    torch.testing.assert_close(got_nobias, torch.zeros(7))


# --- linear_backward ----------------------------------------------------------


@pytest.mark.parametrize("dtype", [torch.float32, torch.bfloat16])
def test_linear_backward_through_autograd(mojo_device, dtype):
    """nn.Linear's backward IS aten::linear_backward here: registering
    aten::linear keeps the layer from decomposing to addmm, so the recorded
    node is the fused one."""
    x = torch.randn(8, 32).to(dtype)
    w = torch.randn(16, 32).to(dtype)
    b = torch.randn(16).to(dtype)
    upstream = torch.randn(8, 16).to(dtype)

    def run(device):
        xi = x.to(device).requires_grad_()
        wi = w.to(device).requires_grad_()
        bi = b.to(device).requires_grad_()
        torch.nn.functional.linear(xi, wi, bi).backward(upstream.to(device))
        return xi.grad, wi.grad, bi.grad

    with assert_ran("aten::linear", "aten::linear_backward"):
        got = run(mojo_device)
    ref = run("cpu")
    for g, r in zip(got, ref, strict=True):
        torch.testing.assert_close(g.cpu().float(), r.float(), atol=5e-2, rtol=5e-2)


def test_linear_backward_higher_rank_input(mojo_device):
    x = torch.randn(2, 5, 12)
    w = torch.randn(7, 12)
    upstream = torch.randn(2, 5, 7)

    def run(device):
        xi = x.to(device).requires_grad_()
        wi = w.to(device).requires_grad_()
        torch.nn.functional.linear(xi, wi).backward(upstream.to(device))
        return xi.grad, wi.grad

    got = run(mojo_device)
    ref = run("cpu")
    for g, r in zip(got, ref, strict=True):
        torch.testing.assert_close(g.cpu(), r, atol=1e-3, rtol=1e-3)


@pytest.mark.parametrize(
    "mask", [(True, False, False), (False, True, True), (True, True, True)]
)
def test_linear_backward_output_mask(mojo_device, mask, call_checker: CallChecker):
    """Called directly: only the requested gradients have to be right, and
    PyTorch's contract defines both parameter outputs when either is asked
    for."""
    call_checker.register(aten_functions.aten_linear_backward)
    x = torch.randn(6, 10)
    w = torch.randn(4, 10)
    grad = torch.randn(6, 4)
    out = torch.ops.aten.linear_backward(
        x.to(mojo_device), grad.to(mojo_device), w.to(mojo_device), list(mask)
    )
    if mask[0]:
        torch.testing.assert_close(out[0].cpu(), grad @ w, atol=1e-4, rtol=1e-4)
    if mask[1]:
        torch.testing.assert_close(out[1].cpu(), grad.t() @ x, atol=1e-4, rtol=1e-4)
    if mask[2]:
        torch.testing.assert_close(out[2].cpu(), grad.sum(dim=0), atol=1e-4, rtol=1e-4)


# --- addr ---------------------------------------------------------------------


@pytest.mark.parametrize("dtype", [torch.float32, torch.bfloat16, torch.float16])
def test_addr(mojo_device, dtype):
    """The fused kernel reproduces CPU's own addr_kernel op order and per-op
    rounding: ATen's composite fallback multiplies in a different order,
    which drifted enough to fail OpInfo conformance for fp16/bf16. beta and
    alpha below are the failing OpInfo sample."""
    self_ = torch.randn(5, 10).to(dtype)
    vec1 = torch.randn(5).to(dtype)
    vec2 = torch.randn(10).to(dtype)
    with assert_ran("aten::addr"):
        got = torch.addr(
            self_.to(mojo_device),
            vec1.to(mojo_device),
            vec2.to(mojo_device),
            beta=0.6,
            alpha=0.2,
        ).cpu()
    # The tolerance is one ULP of the *intermediates* (beta*self, alpha*outer),
    # which a cancelling output element can show in full: on the GPU one of 50
    # elements lands a bf16 ULP away. ATen's composite gets the order itself
    # wrong, which moved a fifth of the elements much further than this.
    atol = 1e-5 if dtype == torch.float32 else 8e-3
    torch.testing.assert_close(
        got, torch.addr(self_, vec1, vec2, beta=0.6, alpha=0.2), atol=atol, rtol=2e-2
    )


def test_addr_default_beta_alpha(mojo_device):
    self_ = torch.randn(4, 6)
    vec1 = torch.randn(4)
    vec2 = torch.randn(6)
    got = torch.addr(
        self_.to(mojo_device), vec1.to(mojo_device), vec2.to(mojo_device)
    ).cpu()
    torch.testing.assert_close(got, torch.addr(self_, vec1, vec2))


def test_addr_beta_zero_ignores_self(mojo_device):
    """beta=0 must ignore `self` entirely, nan included (ATen's own addr
    contract, aten/src/ATen/native/LinearAlgebra.cpp)."""
    self_ = torch.full((3, 4), float("nan"))
    vec1 = torch.randn(3)
    vec2 = torch.randn(4)
    got = torch.addr(
        self_.to(mojo_device),
        vec1.to(mojo_device),
        vec2.to(mojo_device),
        beta=0.0,
        alpha=1.5,
    ).cpu()
    torch.testing.assert_close(got, torch.addr(self_, vec1, vec2, beta=0.0, alpha=1.5))


def test_addr_self_broadcast(mojo_device):
    """`self` broadcastable to (len(vec1), len(vec2)) but not that exact shape
    (here 0-d): the fused kernel's own right-alignment handles it, so this
    does not reach the composite."""
    self_ = torch.randn(())
    vec1 = torch.randn(3)
    vec2 = torch.randn(5)
    got = torch.addr(
        self_.to(mojo_device),
        vec1.to(mojo_device),
        vec2.to(mojo_device),
        beta=0.5,
        alpha=2.0,
    ).cpu()
    torch.testing.assert_close(got, torch.addr(self_, vec1, vec2, beta=0.5, alpha=2.0))


def test_addr_integer_uses_the_composite(mojo_device):
    """A dtype the fused kernel does not cover: the op falls back to ATen's
    own `math_addr` composition instead of declining, so support is
    unchanged."""
    self_ = torch.arange(12, dtype=torch.int64).reshape(3, 4)
    vec1 = torch.arange(3, dtype=torch.int64)
    vec2 = torch.arange(4, dtype=torch.int64)
    got = torch.addr(
        self_.to(mojo_device), vec1.to(mojo_device), vec2.to(mojo_device)
    ).cpu()
    torch.testing.assert_close(got, torch.addr(self_, vec1, vec2))


# --- convolution --------------------------------------------------------------


def test_conv2d_basic(mojo_device, call_checker: CallChecker):
    call_checker.register(aten_functions.aten_convolution)
    x = torch.randn(1, 3, 16, 16)
    w = torch.randn(6, 3, 5, 5)
    bias = torch.randn(6)
    got = torch.nn.functional.conv2d(
        x.to(mojo_device), w.to(mojo_device), bias=bias.to(mojo_device), padding=2
    ).cpu()
    torch.testing.assert_close(
        got,
        torch.nn.functional.conv2d(x, w, bias=bias, padding=2),
        atol=1e-4,
        rtol=1e-3,
    )


@pytest.mark.parametrize(
    "kwargs",
    [
        {},
        {"stride": 2},
        {"padding": 1},
        {"stride": (2, 1), "padding": (1, 2)},
        {"dilation": 2},
        {"stride": 2, "padding": 2, "dilation": 2},
    ],
)
def test_conv2d_geometry(mojo_device, kwargs):
    x = torch.randn(2, 3, 12, 14)
    w = torch.randn(4, 3, 3, 3)
    got = torch.nn.functional.conv2d(x.to(mojo_device), w.to(mojo_device), **kwargs)
    ref = torch.nn.functional.conv2d(x, w, **kwargs)
    assert got.shape == ref.shape
    torch.testing.assert_close(got.cpu(), ref, atol=1e-4, rtol=1e-3)


def test_conv2d_1x1_reuses_the_input_as_the_patch_matrix(mojo_device):
    """A 1x1 stride-1 conv needs no im2col: NCHW already is the col matrix."""
    x = torch.randn(2, 5, 7, 9)
    w = torch.randn(3, 5, 1, 1)
    got = torch.nn.functional.conv2d(x.to(mojo_device), w.to(mojo_device)).cpu()
    torch.testing.assert_close(
        got, torch.nn.functional.conv2d(x, w), atol=1e-4, rtol=1e-3
    )


def test_conv2d_grouped(mojo_device):
    x = torch.randn(2, 6, 10, 10)
    w = torch.randn(8, 3, 3, 3)
    bias = torch.randn(8)
    got = torch.nn.functional.conv2d(
        x.to(mojo_device), w.to(mojo_device), bias=bias.to(mojo_device), groups=2
    ).cpu()
    torch.testing.assert_close(
        got, torch.nn.functional.conv2d(x, w, bias=bias, groups=2), atol=1e-4, rtol=1e-3
    )


def test_conv2d_depthwise(mojo_device):
    x = torch.randn(1, 4, 8, 8)
    w = torch.randn(4, 1, 3, 3)
    got = torch.nn.functional.conv2d(
        x.to(mojo_device), w.to(mojo_device), groups=4, padding=1
    ).cpu()
    torch.testing.assert_close(
        got, torch.nn.functional.conv2d(x, w, groups=4, padding=1), atol=1e-4, rtol=1e-3
    )


# conv1d is the rank-3 aten::convolution: the 2-D im2col + GEMM path with a
# unit H axis. Odd lengths and channel counts on purpose.
@pytest.mark.parametrize("dtype", [torch.float32, torch.float16, torch.bfloat16])
@pytest.mark.parametrize(
    "in_c,out_c,length,k,stride,padding,dilation,groups,bias",
    [
        (3, 8, 17, 3, 1, 0, 1, 1, True),
        (8, 8, 17, 3, 2, 1, 1, 1, True),
        (8, 12, 25, 3, 1, 2, 2, 1, False),
        (8, 12, 25, 3, 1, 1, 1, 2, True),
        (6, 6, 19, 5, 2, 2, 1, 6, True),  # depthwise
        (80, 16, 357, 3, 1, 1, 1, 1, True),
        (5, 7, 13, 1, 1, 0, 1, 1, True),  # 1x1: no im2col
        (4, 6, 11, 4, 3, 3, 2, 1, False),
    ],
)
def test_conv1d(
    mojo_device, dtype, in_c, out_c, length, k, stride, padding, dilation, groups, bias
):
    x = torch.randn(2, in_c, length)
    w = torch.randn(out_c, in_c // groups, k)
    b = torch.randn(out_c) if bias else None
    args = (stride, padding, dilation, groups)
    with assert_ran("aten::convolution"):
        got = torch.nn.functional.conv1d(
            x.to(dtype).to(mojo_device),
            w.to(dtype).to(mojo_device),
            None if b is None else b.to(dtype).to(mojo_device),
            *args,
        )
    ref = torch.nn.functional.conv1d(
        x.to(dtype).double(),
        w.to(dtype).double(),
        None if b is None else b.to(dtype).double(),
        *args,
    )
    assert got.shape == ref.shape and got.dtype == dtype
    tol = 1e-4 if dtype == torch.float32 else 3e-2
    torch.testing.assert_close(got.cpu().double(), ref, atol=tol, rtol=tol)


def test_conv1d_non_contiguous_input_and_module(mojo_device):
    """A transposed (N, L, C) view goes through the contiguous copy, and
    nn.Conv1d reaches the same op."""
    conv = torch.nn.Conv1d(6, 10, 3, padding=1)
    x = torch.randn(3, 23, 6).transpose(1, 2)
    with torch.no_grad():
        ref = conv(x)
        conv = conv.to(mojo_device)
        with assert_ran("aten::convolution"):
            got = conv(x.to(mojo_device))
    torch.testing.assert_close(got.cpu(), ref, atol=1e-4, rtol=1e-3)


def test_conv1d_transposed_declines(mojo_device):
    x = torch.randn(1, 3, 8).to(mojo_device)
    w = torch.randn(3, 2, 3).to(mojo_device)
    with pytest.raises(NotImplementedError):
        torch.nn.functional.conv_transpose1d(x, w)


def test_conv_transposed_declines(mojo_device):
    """The transposed forward has no kernel here: it must raise, not produce
    a plain convolution."""
    x = torch.randn(1, 3, 8, 8).to(mojo_device)
    w = torch.randn(3, 2, 3, 3).to(mojo_device)
    with pytest.raises(NotImplementedError):
        torch.nn.functional.conv_transpose2d(x, w)


# --- convolution_backward -----------------------------------------------------

# (n, c, spatial, out_c, kernel, stride, padding, dilation, groups): odd and
# rectangular extents, every stride/padding/dilation regime, grouped and
# depthwise, batch 1 (where grad_output needs no permuting copy) and the 1x1
# stride-1 case (where im2col and col2im are the identity).
_CONV_BWD_CASES = {
    "k3s1p1": (2, 6, (9, 11), 8, (3, 3), 1, 1, 1, 1),
    "stride2": (2, 6, (11, 11), 8, (3, 3), 2, 1, 1, 1),
    "unpadded": (3, 4, (10, 13), 6, (3, 3), 1, 0, 1, 1),
    "dilation2": (2, 4, (13, 13), 6, (3, 3), 1, 2, 2, 1),
    "groups2": (2, 8, (9, 9), 12, (3, 3), 1, 1, 1, 2),
    "depthwise": (2, 6, (9, 9), 6, (3, 3), 1, 1, 1, 6),
    "pointwise": (2, 5, (7, 7), 4, (1, 1), 1, 0, 1, 1),
    "pointwise_n1": (1, 5, (7, 9), 4, (1, 1), 1, 0, 1, 1),
    "pointwise_grouped_n1": (1, 4, (6, 6), 8, (1, 1), 1, 0, 1, 2),
    # (11 + 2 - 3) // 2 + 1 == 5 drops a row: the last input row has no tap.
    "truncating_n1": (1, 4, (11, 9), 6, (3, 3), 2, 1, 1, 1),
    "rect_per_axis": (2, 3, (12, 10), 5, (5, 3), (2, 1), (2, 1), (1, 2), 1),
    "grouped_n1": (1, 6, (8, 8), 6, (3, 3), 1, 1, 1, 3),
    "1d": (2, 3, (17,), 8, (3,), 1, 0, 1, 1),
    "1d_stride2_groups2": (2, 8, (25,), 12, (3,), 2, 1, 1, 2),
    "1d_depthwise": (2, 6, (19,), 6, (5,), 2, 2, 1, 6),
    "1d_pointwise_n1": (1, 5, (13,), 7, (1,), 1, 0, 1, 1),
    "1d_dilated": (3, 4, (11,), 6, (4,), 3, 3, 2, 1),
}


def _conv_bwd_args(case: str) -> tuple[list[torch.Tensor], list[object]]:
    n, c, spatial, out_c, kernel, stride, padding, dilation, groups = _CONV_BWD_CASES[
        case
    ]
    rank = len(spatial)

    def per_axis(v):
        return list(v) if isinstance(v, tuple) else [v] * rank

    stride, padding, dilation = per_axis(stride), per_axis(padding), per_axis(dilation)
    out = [
        (spatial[i] + 2 * padding[i] - dilation[i] * (kernel[i] - 1) - 1) // stride[i]
        + 1
        for i in range(rank)
    ]
    gen = torch.Generator().manual_seed(0)
    tensors = [
        torch.randn(n, out_c, *out, generator=gen),
        torch.randn(n, c, *spatial, generator=gen),
        torch.randn(out_c, c // groups, *kernel, generator=gen),
    ]
    rest = [[out_c], stride, padding, dilation, False, [0] * rank, groups]
    return tensors, rest


def _conv_bwd_bound(dtype: torch.dtype, k: int) -> float:
    """Per-element error allowance relative to |go| (x) |x| (x) |w|, the
    same backward over absolute values.

    fp32 is an fp32 accumulation of `k` terms: ~sqrt(k) half-ulps, x8 for
    margin. A 16-bit dtype also rounds the output once, and the data
    gradient's intermediate columns once before col2im sums them: two
    half-ulps of the 16-bit format, x2 for margin.
    """
    acc = 8 * torch.finfo(torch.float32).eps * k**0.5
    if dtype == torch.float32:
        return acc
    return acc + 2 * torch.finfo(dtype).eps


def _check_conv_bwd(case: str, dtype: torch.dtype, mask: list[bool], device: str):
    tensors, rest = _conv_bwd_args(case)
    tensors = [t.to(dtype) for t in tensors]
    with assert_ran("aten::convolution_backward"):
        got = torch.ops.aten.convolution_backward(
            *[t.to(device) for t in tensors], *rest, mask
        )
    ref = torch.ops.aten.convolution_backward(
        *[t.double() for t in tensors], *rest, [True, True, True]
    )
    mag = torch.ops.aten.convolution_backward(
        *[t.double().abs() for t in tensors], *rest, [True, True, True]
    )
    grad_output, x, weight = tensors
    groups = _CONV_BWD_CASES[case][-1]
    k_in = weight.shape[0] // groups * weight[0, 0].numel()
    k_param = grad_output.numel() // grad_output.shape[1]
    for slot, wanted in enumerate(mask):
        if not wanted:
            assert got[slot] is None
            continue
        assert got[slot].dtype == dtype
        assert got[slot].shape == ref[slot].shape
        bound = mag[slot] * _conv_bwd_bound(dtype, k_in if slot == 0 else k_param)
        err = (got[slot].cpu().double() - ref[slot]).abs()
        assert (err <= bound + 1e-30).all(), (
            f"slot {slot}: worst error {err.max().item()} at bound "
            f"{bound.flatten()[err.argmax()].item()}"
        )


@pytest.mark.parametrize("case", _CONV_BWD_CASES)
@pytest.mark.parametrize("dtype", [torch.float32, torch.bfloat16])
def test_convolution_backward(mojo_device, case, dtype):
    _check_conv_bwd(case, dtype, [True, True, True], mojo_device)


@pytest.mark.parametrize(
    "mask",
    [
        [True, False, False],
        [False, True, False],
        [False, False, True],
        [True, True, False],
        [True, False, True],
        [False, True, True],
        [False, False, False],
    ],
    ids=str,
)
@pytest.mark.parametrize("case", ["stride2", "groups2", "pointwise_n1", "1d"])
def test_convolution_backward_output_mask(mojo_device, case, mask):
    """Masked-off slots come back undefined (None), and the requested ones do
    not depend on which others were skipped."""
    _check_conv_bwd(case, torch.float32, mask, mojo_device)


def test_convolution_backward_float16(mojo_device):
    _check_conv_bwd("rect_per_axis", torch.float16, [True, True, True], mojo_device)


def test_convolution_backward_strided_operands(mojo_device):
    """A channels-last grad_output, a transposed-view input and a
    channels-last weight all go through the same gradients."""
    tensors, rest = _conv_bwd_args("stride2")
    grad_output, x, weight = tensors
    ref = torch.ops.aten.convolution_backward(
        grad_output.double(), x.double(), weight.double(), *rest, [True] * 3
    )
    go_m = grad_output.to(mojo_device).to(memory_format=torch.channels_last)
    x_m = x.transpose(2, 3).contiguous().to(mojo_device).transpose(2, 3)
    w_m = weight.to(mojo_device).to(memory_format=torch.channels_last)
    assert not go_m.is_contiguous() and not x_m.is_contiguous()
    with assert_ran("aten::convolution_backward"):
        got = torch.ops.aten.convolution_backward(go_m, x_m, w_m, *rest, [True] * 3)
    for g, r in zip(got, ref, strict=True):
        torch.testing.assert_close(g.cpu().double(), r, atol=1e-4, rtol=1e-4)


def test_convolution_backward_transposed_declines(mojo_device):
    x = torch.randn(1, 3, 8, 8).to(mojo_device)
    w = torch.randn(3, 2, 3, 3).to(mojo_device)
    go = torch.randn(1, 2, 10, 10).to(mojo_device)
    with pytest.raises(NotImplementedError):
        torch.ops.aten.convolution_backward(
            go, x, w, [2], [1, 1], [0, 0], [1, 1], True, [0, 0], 1, [True] * 3
        )


@pytest.mark.parametrize("dtype", [torch.float32, torch.bfloat16])
@pytest.mark.parametrize("bias", [True, False])
@pytest.mark.parametrize(
    "make",
    [
        lambda: torch.nn.Conv2d(5, 7, 3, stride=2, padding=1),
        lambda: torch.nn.Conv2d(6, 6, 3, padding=1, groups=6, bias=False),
        lambda: torch.nn.Conv2d(4, 8, (3, 5), padding=(1, 2), dilation=(1, 2)),
        lambda: torch.nn.Conv1d(5, 7, 5, stride=2, padding=2),
        lambda: torch.nn.Conv1d(6, 9, 3, groups=3),
    ],
    ids=[
        "conv2d_s2",
        "conv2d_depthwise",
        "conv2d_rect_dilated",
        "conv1d_s2",
        "conv1d_g3",
    ],
)
def test_conv_module_trains(mojo_device, make, bias, dtype):
    """nn.Conv forward then .backward(): every parameter's .grad and the
    input's match CPU, and the backward is the native op."""
    torch.manual_seed(0)
    conv = make()
    if not bias:
        conv.bias = None
    spatial = (13, 11) if isinstance(conv, torch.nn.Conv2d) else (23,)
    x = torch.randn(3, conv.in_channels, *spatial)

    def run(module, device, dt):
        module = module.to(device=device, dtype=dt)
        xi = x.to(device=device, dtype=dt).requires_grad_()
        out = module(xi)
        out.backward(torch.ones_like(out) * 0.5)
        return [xi.grad] + [p.grad for p in module.parameters()]

    ref = run(copy.deepcopy(conv), "cpu", torch.float64)
    with assert_ran("aten::convolution", "aten::convolution_backward"):
        got = run(conv, mojo_device, dtype)
    tol = 1e-4 if dtype == torch.float32 else 5e-2
    assert len(got) == len(ref)
    for g, r in zip(got, ref, strict=True):
        assert g is not None and g.dtype == dtype
        torch.testing.assert_close(g.cpu().double(), r, atol=tol, rtol=tol)


# --- torch.matmul decomposes onto the registered ops --------------------------


def test_matmul_decomposes_to_mm_and_bmm(mojo_device):
    """aten::matmul is CompositeImplicitAutograd: it is not registered here,
    it reaches mm / bmm."""
    a2, b2 = torch.randn(6, 8), torch.randn(8, 4)
    with assert_ran("aten::mm"):
        got2 = torch.matmul(a2.to(mojo_device), b2.to(mojo_device)).cpu()
    torch.testing.assert_close(got2, a2 @ b2, atol=1e-4, rtol=1e-4)

    a3, b3 = torch.randn(3, 6, 8), torch.randn(3, 8, 4)
    with assert_ran("aten::bmm"):
        got3 = torch.matmul(a3.to(mojo_device), b3.to(mojo_device)).cpu()
    torch.testing.assert_close(got3, a3 @ b3, atol=1e-4, rtol=1e-4)


# --- the architecture-gated tensor-core bridges -------------------------------


@pytest.mark.parametrize("dtype", [torch.bfloat16, torch.float16])
@pytest.mark.parametrize("op", ["mm", "addmm", "linear", "bmm"])
def test_gemm16_entry_points(mojo_h100, dtype, op):
    """Every 16-bit entry point on aligned shapes, where the warp-specialized
    tensor-core routes are the ones that take the call."""
    m, n, k, batch = 128, 192, 256, 3

    def dev(*shape: int) -> torch.Tensor:
        return torch.randn(*shape).to(dtype).to(mojo_h100)

    if op == "mm":
        a, b = dev(m, k), dev(k, n)
        got, ref = torch.mm(a, b), a.cpu().float() @ b.cpu().float()
    elif op == "addmm":
        a, b, c = dev(m, k), dev(k, n), dev(n)
        got = torch.addmm(c, a, b)
        ref = a.cpu().float() @ b.cpu().float() + c.cpu().float()
    elif op == "linear":
        x, w, c = dev(2, m, k), dev(n, k), dev(n)
        got = torch.nn.functional.linear(x, w, c)
        ref = torch.nn.functional.linear(
            x.cpu().float(), w.cpu().float(), c.cpu().float()
        )
    else:
        a, b = dev(batch, m, k), dev(batch, k, n)
        got, ref = torch.bmm(a, b), torch.bmm(a.cpu().float(), b.cpu().float())

    assert got.dtype == dtype
    torch.testing.assert_close(got.cpu().float(), ref, atol=2e-1, rtol=2e-2)


def test_tf32_bridge_opt_in(mojo_h100):
    """fp32 stays on the strict SIMT path under torch's default matmul
    precision ("highest": TF32 drops mantissa bits) and reaches the
    tensor-core route once torch.set_float32_matmul_precision allows it."""
    a = torch.randn(128, 256)
    b = torch.randn(256, 192)
    ref = a @ b
    previous = torch.get_float32_matmul_precision()
    torch.set_float32_matmul_precision("high")
    try:
        got = torch.mm(a.to(mojo_h100), b.to(mojo_h100)).cpu()
    finally:
        torch.set_float32_matmul_precision(previous)
    # TF32 keeps 10 mantissa bits, so the tolerance is bf16-like, not fp32.
    torch.testing.assert_close(got, ref, atol=2e-1, rtol=2e-2)
    strict = torch.mm(a.to(mojo_h100), b.to(mojo_h100)).cpu()
    torch.testing.assert_close(strict, ref, atol=1e-3, rtol=1e-4)


# ---------------------------------------------------------------------------
# Strided-operand arms and degenerate GEMM shapes.
#
# The layout tests above use `.t()` from offset 0 and an expanded bmm batch.
# Every arm below is a different route: a gapped B, an offset-view A (the
# route may not assume offset 0), a stride-0 broadcast read, a rank-3
# activation, and the m/n/k == 1 shapes decode paths take.
# ---------------------------------------------------------------------------


@pytest.mark.parametrize(
    "arm",
    [
        "b_gapped",
        "a_transposed",
        "a_offset_transposed",
        "both_strided",
        "a_broadcast",
        "a_rank3_gapped",
        "batched_transposed",
    ],
)
def test_matmul_every_strided_arm(mojo_gpu, arm):
    if arm == "batched_transposed":
        a_cpu = torch.randn(4, 24, 32)
        b_cpu = torch.randn(4, 24, 16)
        got = torch.bmm(a_cpu.to(mojo_gpu).transpose(1, 2), b_cpu.to(mojo_gpu))
        ref = torch.bmm(a_cpu.transpose(1, 2), b_cpu)
        torch.testing.assert_close(got.cpu(), ref, atol=5e-2, rtol=5e-2)
        return

    b_base = torch.randn(64, 64)
    if arm == "b_gapped":
        a_cpu, b_cpu = torch.randn(48, 64), b_base[:, ::2]
        a = a_cpu.to(mojo_gpu)
        b = b_base.to(mojo_gpu)[:, ::2]
    elif arm == "a_transposed":
        a_cpu, b_cpu = torch.randn(48, 64).t(), torch.randn(48, 32)
        a = a_cpu.t().contiguous().to(mojo_gpu).t()
        b = b_cpu.to(mojo_gpu)
    elif arm == "a_offset_transposed":
        base = torch.randn(80, 64)
        a_cpu, b_cpu = base[8:56].t(), torch.randn(48, 32)
        a = base.to(mojo_gpu)[8:56].t()
        b = b_cpu.to(mojo_gpu)
    elif arm == "both_strided":
        base = torch.randn(80, 64)
        # A is (64, 48) after the transpose, so B must have 48 rows.
        a_cpu, b_cpu = base[8:56].t(), b_base[:48, ::2]
        a = base.to(mojo_gpu)[8:56].t()
        b = b_base.to(mojo_gpu)[:48, ::2]
    elif arm == "a_broadcast":
        row = torch.randn(1, 48)
        a_cpu, b_cpu = row.expand(64, 48), torch.randn(48, 32)
        a = row.to(mojo_gpu).expand(64, 48)
        b = b_cpu.to(mojo_gpu)
    else:  # a_rank3_gapped
        base = torch.randn(16, 64, 48)
        a_cpu, b_cpu = base[::2], torch.randn(48, 32)
        a = base.to(mojo_gpu)[::2]
        b = b_cpu.to(mojo_gpu)

    got = a @ b
    torch.testing.assert_close(got.cpu(), a_cpu @ b_cpu, atol=5e-2, rtol=5e-2)


def test_addmm_strided_bias_and_operands(mojo_gpu):
    bias_base = torch.randn(64)  # [::2] is the 32 columns of the product
    a_base = torch.randn(48, 64)
    b_cpu = torch.randn(48, 32)
    bias_cpu, a_cpu = bias_base[::2], a_base.t()
    got = torch.addmm(
        bias_base.to(mojo_gpu)[::2], a_base.to(mojo_gpu).t(), b_cpu.to(mojo_gpu)
    )
    torch.testing.assert_close(
        got.cpu(), torch.addmm(bias_cpu, a_cpu, b_cpu), atol=5e-2, rtol=5e-2
    )


@pytest.mark.parametrize("dtype", [torch.float32, torch.float16])
@pytest.mark.parametrize("case", ["gevm", "gevm_bias", "out_features_one", "batch_n1"])
def test_degenerate_gemm_shapes(mojo_gpu, dtype, case):
    """m == 1 (decode), n == 1, and the batched n == 1: each is a separate
    route from the general tile ladder."""
    atol, rtol = _tol(dtype)
    if case == "gevm":
        a = torch.randn(1, 128).to(dtype)
        b = torch.randn(128, 64).to(dtype)
        got = torch.mm(a.to(mojo_gpu), b.to(mojo_gpu))
        ref = a.float() @ b.float()
    elif case == "gevm_bias":
        a = torch.randn(1, 128).to(dtype)
        b = torch.randn(128, 64).to(dtype)
        c = torch.randn(64).to(dtype)
        got = torch.addmm(c.to(mojo_gpu), a.to(mojo_gpu), b.to(mojo_gpu))
        ref = a.float() @ b.float() + c.float()
    elif case == "out_features_one":
        x = torch.randn(37, 129).to(dtype)
        w = torch.randn(1, 129).to(dtype)
        c = torch.randn(1).to(dtype)
        got = torch.nn.functional.linear(x.to(mojo_gpu), w.to(mojo_gpu), c.to(mojo_gpu))
        ref = torch.nn.functional.linear(x.float(), w.float(), c.float())
    else:
        a = torch.randn(4, 8, 129).to(dtype)
        b = torch.randn(4, 129, 1).to(dtype)
        got = torch.bmm(a.to(mojo_gpu), b.to(mojo_gpu))
        ref = torch.bmm(a.float(), b.float())
    assert got.dtype == dtype
    torch.testing.assert_close(got.cpu().float(), ref, atol=atol * 4, rtol=rtol * 4)


@pytest.mark.parametrize("dtype", [torch.float32, torch.float16, torch.bfloat16])
def test_linear_single_token(mojo_gpu, dtype):
    """m == 1 through F.linear: the GPT-2 decode step."""
    x = torch.randn(1, 768).to(dtype)
    w = torch.randn(96, 768).to(dtype)
    c = torch.randn(96).to(dtype)
    got = torch.nn.functional.linear(x.to(mojo_gpu), w.to(mojo_gpu), c.to(mojo_gpu))
    ref = torch.nn.functional.linear(x.float(), w.float(), c.float())
    torch.testing.assert_close(got.cpu().float(), ref, atol=2e-1, rtol=5e-2)


@pytest.mark.parametrize(
    "in_features,out_features", [(768, 2304), (768, 768), (768, 3072), (3072, 768)]
)
def test_addmm_model_shapes(mojo_gpu, in_features, out_features):
    """Skinny-M against a large N, at the four shapes a transformer block
    actually issues."""
    x = torch.randn(32, in_features)
    w = torch.randn(in_features, out_features)
    c = torch.randn(out_features)
    got = torch.addmm(c.to(mojo_gpu), x.to(mojo_gpu), w.to(mojo_gpu))
    torch.testing.assert_close(got.cpu(), torch.addmm(c, x, w), atol=5e-2, rtol=5e-2)


def test_linear_skinny_m_large_output(mojo_gpu):
    x = torch.randn(32, 1, 768)
    w = torch.randn(8192, 768)
    got = torch.nn.functional.linear(x.to(mojo_gpu), w.to(mojo_gpu))
    torch.testing.assert_close(
        got.cpu(), torch.nn.functional.linear(x, w), atol=5e-2, rtol=5e-2
    )


# --- the dynamic tile scheduler (gemm16_sched_pool.mojo) ----------------------
#
# The three persistent bodies -- `_rolling_persistent_ws`,
# `_nt_bias_rolling_ws` (gemm16_rolling_kernels.mojo) and
# `_v4_nn_persistent_ws` (gemm16_nn_v4_kernels.mojo) -- take their output
# tiles from a global ticket counter instead of owning a static share, so a
# cluster that cannot launch (NCCL holding its SMs on a DDP job's comm
# stream) costs its tiles' latency rather than the whole kernel's.  The
# counter lives in a per-(device, stream) block that the kernel itself
# resets: the last cluster to fetch a ticket stores 0 back.  What that makes
# testable, beyond the ordinary correctness the tables above already cover:
#
#   * a MISSED reset shows up only on the SECOND launch that shares the
#     counter (it would start mid-count and skip tiles), so the tests below
#     launch repeatedly without a sync in between and check the last result;
#   * the tile order is dynamic but each tile's K order is not, so two runs
#     of one GEMM must agree BIT for bit -- a stronger assertion than the
#     bf16 bound, and the one that would catch a torn ticket handing two
#     clusters the same tile;
#   * a second stream gets its own counter, which is what makes reuse safe
#     without reasoning about occupancy.
# -----------------------------------------------------------------------------

# (m, n, k) through the NN rolling body: m % 128 == 0, n % 256 == 64,
# 4n <= m <= 32n and n <= k <= 8n is try_enqueue_candidate_nn's own gate.
SCHED_NN_DX = (8192, 1600, 4800)  # c_attn dX
SCHED_NN_DX_SMALL = (6400, 1600, 1600)  # attn.c_proj dX, cheap enough to loop
SCHED_TN_DW = (4800, 1600, 8192)  # c_attn dW, through the TN rolling body


def _sched_mm(device, m, n, k):
    """One bf16 NN GEMM on `device`, with its fp32 CPU reference."""
    a = torch.randn(m, k, dtype=torch.bfloat16)
    b = torch.randn(k, n, dtype=torch.bfloat16)
    return a.to(device), b.to(device), a.float() @ b.float()


@pytest.mark.parametrize("m,n,k", [(8192, 4800, 1600), (8192, 1600, 6400)])
def test_gemm16_sched_fused_forward(mojo_h100, m, n, k):
    """The fused NT+bias 192x192 kernel through the scheduler.

    That instantiation (`_nt_bias_rolling_ws`, the only one with a live
    `bias` argument) was compiled but never launched while the scheduler was
    developed outside the tree, so it gets its own test: the bias still fuses
    into the one launch, the result still clears the bf16 bound, and two runs
    agree bit for bit."""
    x = torch.randn(m, k, dtype=torch.bfloat16)
    w = torch.randn(n, k, dtype=torch.bfloat16)
    b = torch.randn(n, dtype=torch.bfloat16)
    dx, dw, db = x.to(mojo_h100), w.to(mojo_h100), b.to(mojo_h100)
    with assert_ran("aten::linear"):
        with assert_no_bias_add():
            got = torch.nn.functional.linear(dx, dw, db)
    again = torch.nn.functional.linear(dx, dw, db)
    assert torch.equal(got.cpu(), again.cpu()), (
        "two runs of one GEMM disagree: the tile ORDER is dynamic but each "
        "tile's K order is not, so the result must be bit-identical"
    )
    ref = torch.nn.functional.linear(x.float(), w.float(), b.float())
    assert _rel_err(got, ref) < _bf16_bound(k)


def test_gemm16_sched_dx_mm(mojo_h100):
    """dX as a bare mm: the NN rolling body's own route."""
    m, n, k = SCHED_NN_DX
    a, b, ref = _sched_mm(mojo_h100, m, n, k)
    with assert_ran("aten::mm"):
        got = torch.mm(a, b)
    assert torch.equal(got.cpu(), torch.mm(a, b).cpu())
    assert _rel_err(got, ref) < _bf16_bound(k)


def test_gemm16_sched_dw_mm(mojo_h100):
    """dW as `mm(grad.t(), x)`: the TN (col_a) instantiation of the same
    body, reached through the rolling geometry dispatcher."""
    out_features, in_features, tokens = SCHED_TN_DW
    x = torch.randn(tokens, in_features, dtype=torch.bfloat16)
    g = torch.randn(tokens, out_features, dtype=torch.bfloat16)
    dx, dg = x.to(mojo_h100), g.to(mojo_h100)
    with assert_ran("aten::mm"):
        got = torch.mm(dg.t(), dx)
    assert torch.equal(got.cpu(), torch.mm(dg.t(), dx).cpu())
    assert _rel_err(got, g.float().t() @ x.float()) < _bf16_bound(tokens)


def test_gemm16_sched_counter_is_reset_between_launches(mojo_h100):
    """64 back-to-back launches sharing one ticket counter, each computing a
    DIFFERENT product, then a GEMM with a different work census.

    Nothing synchronizes between the launches, so they are exactly the
    same-stream sequence the reset argument relies on: launch j+1 may only
    start once launch j's last fetcher has stored 0 back.  A counter left
    dirty makes launch j+1 start mid-count and never issue its first tiles.

    Every launch slides the A operand down one row, so consecutive launches
    have different answers everywhere. That is what makes the final
    comparison able to see a skipped tile at all: torch.mm allocates a fresh
    output each call and the caching allocator hands back the block the
    previous result just freed, so a launch that recomputed the SAME product
    would find correct bytes already sitting in the tiles it never wrote (an
    agent-C review finding on the first cut of this test)."""
    m, n, k = SCHED_NN_DX_SMALL
    laps = 64
    a = torch.randn(m + laps, k, dtype=torch.bfloat16)
    b = torch.randn(k, n, dtype=torch.bfloat16)
    dev_a, dev_b = a.to(mojo_h100), b.to(mojo_h100)
    got = torch.mm(dev_a[:m], dev_b)
    for lap in range(1, laps):
        got = torch.mm(dev_a[lap : lap + m], dev_b)
    ref = a[laps - 1 : laps - 1 + m].float() @ b.float()
    assert _rel_err(got, ref) < _bf16_bound(k)
    # A different census on the same counter: the TN body's work loop is a
    # different length, so a stale count would land somewhere else entirely.
    out_features, in_features, tokens = 1600, 1600, 8192
    x = torch.randn(tokens, in_features, dtype=torch.bfloat16)
    g = torch.randn(tokens, out_features, dtype=torch.bfloat16)
    dw = torch.mm(g.to(mojo_h100).t(), x.to(mojo_h100))
    assert _rel_err(dw, g.float().t() @ x.float()) < _bf16_bound(tokens)


def test_gemm16_sched_on_a_side_stream(mojo_h100):
    """Two streams running persistent GEMMs AT THE SAME TIME.

    The whole reuse argument is "launches on one stream are ordered", so a
    GEMM issued on a side stream must not share the default stream's counter:
    if it did, two genuinely concurrent launches would dispense from one
    counter and two clusters would compute the same tile while another tile
    went unwritten.  Only genuinely overlapping launches can show that, so
    the streams are fenced once for INPUT readiness and then each issues its
    own burst with no fence in between -- an earlier cut of this test fenced
    the side stream behind the default stream's GEMM, which serialized the
    two and tested nothing (an agent-C/Codex review finding).

    Outputs are held until after the barrier: freeing a side stream's tensor
    is ordered on its owner stream only, so dropping them inside the burst
    would hand live memory back to the allocator."""
    side = side_stream_or_skip(mojo_h100)
    m, n, k = SCHED_NN_DX_SMALL
    a, b, ref = _sched_mm(mojo_h100, m, n, k)
    bound = _bf16_bound(k)
    # Prewarm both streams (kernel build, allocator) and preallocate every
    # output, so the bursts below are nothing but back-to-back launches.
    with device_module.stream(side):
        warm_side = torch.mm(a, b)
    warm_default = torch.mm(a, b)
    on_side = [torch.empty(m, n, dtype=a.dtype, device=a.device) for _ in range(24)]
    on_default = [torch.empty(m, n, dtype=a.dtype, device=a.device) for _ in range(24)]
    # The inputs were filled on the default stream; that is the only
    # cross-stream dependency, and this is the last fence before the bursts.
    torch.accelerator.synchronize()
    # Interleave the two streams' submissions so neither burst can drain
    # before the other starts (a side burst issued whole before the default
    # one could finish first and never overlap -- a Codex review finding).
    # Overlap is made likely, not proven: nothing here observes the device
    # timeline, so a shared counter would show only as a wrong tile.
    for out_side, out_default in zip(on_side, on_default, strict=True):
        with device_module.stream(side):
            torch.mm(a, b, out=out_side)
        torch.mm(a, b, out=out_default)
    torch.accelerator.synchronize()
    del warm_side, warm_default
    for got in on_default:
        assert _rel_err(got, ref) < bound
    first = on_default[0].cpu()
    for got in on_side:
        assert _rel_err(got, ref) < bound
        # Same operands, same per-tile K order: the answer is bit-identical
        # however the tiles were shared out, on either stream.
        assert torch.equal(got.cpu(), first)


# The 650-launch body runs in a FRESH process: the counter table is
# process-global and never evicts, so a test sharing a process with the rest
# of this file could find it already exhausted and then measure two runs of
# the SAME fallback kernel -- equal times, a green test, and the bug intact
# (an agent-C/Codex review finding). A subprocess also keeps the timing away
# from whatever else the session has left resident.
_SLOT_STABILITY_PROGRAM = """
import os, time
os.environ.setdefault("MODULAR_TELEMETRY_ENABLED", "0")
import torch
from torch_mojo_backend import register_mojo_devices

register_mojo_devices()
dev = "mojo:0"
grad = torch.randn(8192, 1600, dtype=torch.bfloat16, device=dev)
x = torch.randn(8192, 1600, dtype=torch.bfloat16, device=dev)


def burst(n):
    torch.mojo.synchronize()
    start = time.perf_counter()
    for _ in range(n):
        out = torch.mm(grad.t(), x)
    torch.mojo.synchronize()
    return (time.perf_counter() - start) / n, out


burst(8)
first, _ = burst(50)
for _ in range(600):
    torch.mm(grad.t(), x)
last, out = burst(50)
ref = grad.t().float().cpu() @ x.float().cpu()
scale = ref.abs().max().clamp(min=1.0)
rel = float((out.cpu().float() - ref).abs().max() / scale)
print("RESULT", first, last, rel, flush=True)
"""


def test_gemm16_sched_slot_is_stable_across_many_launches(mojo_h100):
    """The scheduler's counter slot is keyed on the (device, stream) context
    handle. Keyed on a per-call object it appended one table entry per launch
    -- a device allocation and a memset on the hot path -- until the 512-slot
    table filled, after which every persistent route declined to the fallback
    kernels for the rest of the process (the GPT-2 XL step went 227 -> 460
    ms). Launch well past that count in a fresh process and check the last
    launches are as fast as the first (a decline is a 3-9x cliff, not a few
    percent).

    What this can NOT see is a table that fills for some other reason and
    takes both bursts down with it; that is what the fresh process is for,
    and why the absolute per-launch time is printed on failure."""
    out = subprocess.run(
        [sys.executable, "-c", _SLOT_STABILITY_PROGRAM],
        capture_output=True,
        text=True,
        timeout=900,
        env={**os.environ, "PYTHONPATH": str(pathlib.Path(__file__).parents[2])},
    )
    assert out.returncode == 0, f"subprocess failed:\n{out.stdout}\n{out.stderr}"
    line = [x for x in out.stdout.splitlines() if x.startswith("RESULT ")]
    assert line, f"no RESULT line:\n{out.stdout}\n{out.stderr}"
    _, first, last, rel = line[-1].split()
    first, last, rel = float(first), float(last), float(rel)
    assert last < 1.5 * first, f"per-launch {first * 1e6:.0f} -> {last * 1e6:.0f} us"
    assert rel < _bf16_bound(8192), f"relative error {rel}"
