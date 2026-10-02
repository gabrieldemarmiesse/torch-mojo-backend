"""Reduction benchmarks.

The reduced-dim axis is folded into the shape token (design rule: any
extra axis an op needs goes into the shape id): S_4096x4096_d0 reduces
the strided dim, S_4096x4096_d1 the contiguous dim, C_16777216_all is the
full reduction of a large vector.  all/any's .dim/.dims overloads and
mean.dim are covered by the same fold.

nonzero uses a fixed 50% density mask from the seeded fixture: the
output-shape host sync is a real cost of the op, but this suite measures
device kernel time only, which is exactly what the design pinned.
"""

from __future__ import annotations

import pytest
import torch
from bench_lib.cases import DTYPES, both, unit_interval
from bench_lib.check import Bench
from bench_lib.hw import Hardware

# shape id -> (tensor shape, reduced dim or None for a full reduction)
DIM_SHAPES: dict[str, tuple[tuple[int, ...], int | None]] = {
    "S_4096x4096_d0": ((4096, 4096), 0),
    "S_4096x4096_d1": ((4096, 4096), 1),
    "C_16777216_all": ((16777216,), None),
}
FULL_SHAPES: dict[str, tuple[int, ...]] = {
    "C_16777216": (16777216,),
    "A_357x789": (357, 789),
}
LASTDIM_SHAPES: dict[str, tuple[tuple[int, ...], int]] = {
    "S_4096x4096_d0": ((4096, 4096), 0),
    "S_4096x4096_d1": ((4096, 4096), 1),
}
# Selection along the last dim. k and the direction are folded into the shape
# token, per the design rule above that any extra axis an op needs goes into
# the shape id. V_ rows are a GPT-2 vocabulary -- the regime HF generate()
# actually runs, one row per sequence with k either a sampling cutoff or a
# beam width -- and A_ is the awkward-shape control. k also selects the launch
# route: K50 fits the tournament (one pass over the row), K2048 does not and
# falls back to the full sort, which is why both are measured. The _min /
# _desc tokens only pick the other comptime direction of the same kernels;
# one of each is enough to notice if that stops being true.
TOPK_SHAPES: dict[str, tuple[tuple[int, ...], int, bool]] = {
    "V_1x50304_K50": ((1, 50304), 50, True),
    "V_8x50304_K2048": ((8, 50304), 2048, True),
    "V_8x50304_K2048_min": ((8, 50304), 2048, False),
    "A_357x789_K32": ((357, 789), 32, True),
}
SORT_SHAPES: dict[str, tuple[tuple[int, ...], bool]] = {
    "V_8x50304": ((8, 50304), False),
    "V_8x50304_desc": ((8, 50304), True),
    "S_4096x4096": ((4096, 4096), False),
    "A_357x789": ((357, 789), False),
}

COVERS: dict[str, str] = {
    "aten::sum": "test_sum (full-reduction case)",
    "aten::sum.dim_IntList": "test_sum (dim cases)",
    "aten::nansum": "test_nansum (same kernel as sum, NaN-zeroing map)",
    "aten::mean": "test_mean (full-reduction case)",
    "aten::mean.dim": "test_mean (dim cases)",
    "aten::prod": "test_prod (full-reduction case)",
    "aten::prod.dim_int": "test_prod (dim cases)",
    "aten::max": "test_max",
    "aten::min": "test_min",
    "aten::amax": "test_amax",
    "aten::amin": "test_amin",
    "aten::argmax": "test_argmax",
    "aten::argmin": "test_argmin",
    "aten::all": "test_all (full-reduction case)",
    "aten::all.dim": "test_all (dim case)",
    "aten::all.dims": "test_all (same fast impl as .dim)",
    "aten::any": "test_any (full-reduction case)",
    "aten::any.dim": "test_any (dim case)",
    "aten::any.dims": "test_any (same fast impl as .dim)",
    "aten::count_nonzero.dim_IntList": "test_count_nonzero",
    "aten::min.dim": "test_min_dim",
    "aten::var.correction": "test_var",
    "aten::linalg_vector_norm": "test_vector_norm",
    "aten::linalg_vector_norm.out": (
        "test_vector_norm (same kernel, out-variant plumbing)"
    ),
    "aten::cumsum": "test_cumsum",
    "aten::topk": "test_topk",
    "aten::sort.stable": "test_sort",
    "aten::nonzero": "test_nonzero",
    "aten::multinomial": "test_multinomial",
    "aten::median.dim": "test_median",
    "aten::kthvalue": "test_kthvalue",
    "aten::max.dim": "test_max_dim",
    "aten::aminmax": "test_aminmax",
    "aten::std.correction": "test_std",
    "aten::var_mean.correction": "test_var_mean",
    "aten::std_mean.correction": "test_std_mean",
    "aten::cumprod": "test_cumprod",
    "aten::_cummax_helper": "test_cummax",
    "aten::_logcumsumexp": "test_logcumsumexp",
    "aten::mode": "test_mode",
    "aten::hash_tensor": "test_hash_tensor",
    "aten::histc": "test_histc",
    "aten::bincount": "test_bincount",
    "aten::renorm": "test_renorm",
    "aten::segment_reduce": "test_segment_reduce",
    "aten::_weight_norm_interface": "test_weight_norm",
    "aten::_compute_linear_combination": "test_linear_combination",
}

_CPU_ONLY = (
    "stock PyTorch has no CUDA/ROCm kernel for it (CPU and MPS only), so there "
    "is no accelerator reference leg to time against"
)
_SAME_KERNEL_OUT = (
    "out= overload of a benchmarked functional op: the same kernel launches, "
    "written straight into (or copied into) the caller's tensors"
)
SKIPPED: dict[str, str] = {
    "aten::max.unary_out": _SAME_KERNEL_OUT,
    "aten::sum.IntList_out": _SAME_KERNEL_OUT,
    "aten::topk.values": _SAME_KERNEL_OUT,
    "aten::nansum.out": _SAME_KERNEL_OUT,
    "aten::min.unary_out": _SAME_KERNEL_OUT,
    "aten::sort.values_stable": _SAME_KERNEL_OUT,
    "aten::multinomial.out": _SAME_KERNEL_OUT,
    "aten::median.dim_values": _SAME_KERNEL_OUT,
    "aten::kthvalue.values": _SAME_KERNEL_OUT,
    "aten::nanmedian.dim_values": _SAME_KERNEL_OUT,
    "aten::nanmedian.dim": (
        "median.dim's launches (the full sort, then one read per row) with the "
        "NaN-skipping select mode: the same binary search median.dim already "
        "runs on every float row"
    ),
    "aten::median": (
        "median.dim's kernels over the flattened tensor as one row: the same "
        "full-sort route test_median and test_sort measure"
    ),
    "aten::nanmedian": "aten::median's route with nanmedian.dim's select mode",
    "aten::norm.Scalar": "legacy norm overload -> the same ord-2 vector_norm kernel",
    "aten::norm.ScalarOpt_dtype": (
        "legacy norm overload -> the same ord-2 vector_norm kernel"
    ),
    "aten::norm.ScalarOpt_dim": (
        "legacy norm overload -> the same ord-2 vector_norm kernel"
    ),
    "aten::norm.ScalarOpt_dim_dtype": (
        "legacy norm overload -> the same ord-2 vector_norm kernel"
    ),
    "aten::norm.out": _SAME_KERNEL_OUT,
    "aten::norm.dtype_out": _SAME_KERNEL_OUT,
    "aten::max.dim_max": _SAME_KERNEL_OUT,
    "aten::aminmax.out": _SAME_KERNEL_OUT,
    "aten::_aminmax": "deprecated alias: aminmax's full-reduction route",
    "aten::_aminmax.dim": "deprecated alias: aminmax's dim route (test_aminmax)",
    "aten::argmax.out": _SAME_KERNEL_OUT,
    "aten::argmin.out": _SAME_KERNEL_OUT,
    "aten::std.correction_out": _SAME_KERNEL_OUT,
    "aten::var.correction_out": _SAME_KERNEL_OUT,
    "aten::cumsum.out": _SAME_KERNEL_OUT,
    "aten::cumsum_": "cumsum's kernels, then a copy back into self",
    "aten::cumprod.out": _SAME_KERNEL_OUT,
    "aten::cumprod_": "cumprod's kernels, then a copy back into self",
    "aten::_cummin_helper": (
        "the cummax scan kernels with the comparison flipped (test_cummax)"
    ),
    "aten::_logcumsumexp.out": _SAME_KERNEL_OUT,
    "aten::mode.values": _SAME_KERNEL_OUT,
    "aten::hash_tensor.out": _SAME_KERNEL_OUT,
    "aten::histc.out": _SAME_KERNEL_OUT,
    "aten::renorm.out": _SAME_KERNEL_OUT,
    "aten::renorm_": "renorm's launches, then a copy back into self",
    "aten::_compute_linear_combination.out": _SAME_KERNEL_OUT,
    "aten::_segment_reduce_backward": (
        "one thread per segment output over the forward's geometry; "
        "test_segment_reduce measures that launch shape"
    ),
    "aten::_weight_norm_interface_backward": (
        "composed of the mul/sum/reciprocal kernels this suite already "
        "measures (tmb/ops/stats.mojo), like test_weight_norm"
    ),
    "aten::_fused_rms_norm_backward": (
        "composed of the mul/sum kernels this suite already measures "
        "(tmb/ops/stats.mojo); no fused kernel of its own yet"
    ),
    "aten::histogram.bin_ct": _CPU_ONLY,
    "aten::histogram.bin_ct_out": _CPU_ONLY,
    "aten::histogram.bins_tensor": _CPU_ONLY,
    "aten::histogram.bins_tensor_out": _CPU_ONLY,
    "aten::_histogramdd_bin_edges": _CPU_ONLY,
    "aten::_histogramdd_from_bin_cts": _CPU_ONLY,
    "aten::_histogramdd_from_bin_tensors": _CPU_ONLY,
    "aten::isin.Tensor_Scalar": "ATen's redispatch to eq.Scalar / ne.Scalar",
    "aten::isin.Tensor_Scalar_out": (
        "ATen's redispatch to eq.Scalar_out / ne.Scalar_out"
    ),
    "aten::isin.Scalar_Tensor": "eq.Scalar over test_elements, then any()",
    "aten::isin.Scalar_Tensor_out": _SAME_KERNEL_OUT,
}


def _dim_case(
    shape_id: str, dtype_id: str, hw: Hardware, mojo: torch.device
) -> tuple[torch.Tensor, torch.Tensor, int | None]:
    shape, dim = DIM_SHAPES[shape_id]
    x_ref, x_our = both(unit_interval(shape, DTYPES[dtype_id]), hw, mojo)
    return x_ref, x_our, dim


@pytest.mark.parametrize("dtype_id", ("bf16", "f32"))
@pytest.mark.parametrize("shape_id", DIM_SHAPES)
@pytest.mark.bench_op("sum.dim_IntList")
def test_sum(
    shape_id: str, dtype_id: str, bench: Bench, hw: Hardware, mojo_device: torch.device
):
    x_ref, x_our, dim = _dim_case(shape_id, dtype_id, hw, mojo_device)
    if dim is None:
        bench.run(
            lambda: torch.sum(x_ref),
            lambda: torch.sum(x_our),
            flops=float(x_ref.numel()),
        )
    else:
        bench.run(
            lambda: torch.sum(x_ref, dim=dim),
            lambda: torch.sum(x_our, dim=dim),
            flops=float(x_ref.numel()),
        )


@pytest.mark.parametrize("dtype_id", ("bf16", "f32"))
@pytest.mark.parametrize("shape_id", DIM_SHAPES)
def test_nansum(
    shape_id: str, dtype_id: str, bench: Bench, hw: Hardware, mojo_device: torch.device
):
    x_ref, x_our, dim = _dim_case(shape_id, dtype_id, hw, mojo_device)
    if dim is None:
        bench.run(
            lambda: torch.nansum(x_ref),
            lambda: torch.nansum(x_our),
            flops=float(x_ref.numel()),
        )
    else:
        bench.run(
            lambda: torch.nansum(x_ref, dim=dim),
            lambda: torch.nansum(x_our, dim=dim),
            flops=float(x_ref.numel()),
        )


@pytest.mark.parametrize("dtype_id", ("bf16", "f32"))
@pytest.mark.parametrize("shape_id", DIM_SHAPES)
def test_mean(
    shape_id: str, dtype_id: str, bench: Bench, hw: Hardware, mojo_device: torch.device
):
    x_ref, x_our, dim = _dim_case(shape_id, dtype_id, hw, mojo_device)
    if dim is None:
        bench.run(
            lambda: torch.mean(x_ref),
            lambda: torch.mean(x_our),
            flops=float(x_ref.numel()),
        )
    else:
        bench.run(
            lambda: torch.mean(x_ref, dim=dim),
            lambda: torch.mean(x_our, dim=dim),
            flops=float(x_ref.numel()),
        )


@pytest.mark.parametrize("dtype_id", ("bf16", "f32"))
@pytest.mark.parametrize("shape_id", DIM_SHAPES)
def test_prod(
    shape_id: str, dtype_id: str, bench: Bench, hw: Hardware, mojo_device: torch.device
):
    x_ref, x_our, dim = _dim_case(shape_id, dtype_id, hw, mojo_device)
    if dim is None:
        bench.run(
            lambda: torch.prod(x_ref),
            lambda: torch.prod(x_our),
            flops=float(x_ref.numel()),
        )
    else:
        bench.run(
            lambda: torch.prod(x_ref, dim=dim),
            lambda: torch.prod(x_our, dim=dim),
            flops=float(x_ref.numel()),
        )


@pytest.mark.parametrize("dtype_id", ("bf16", "f32"))
@pytest.mark.parametrize("shape_id", FULL_SHAPES)
def test_max(
    shape_id: str, dtype_id: str, bench: Bench, hw: Hardware, mojo_device: torch.device
):
    x_ref, x_our = both(
        unit_interval(FULL_SHAPES[shape_id], DTYPES[dtype_id]), hw, mojo_device
    )
    bench.run(
        lambda: torch.max(x_ref), lambda: torch.max(x_our), flops=float(x_ref.numel())
    )


@pytest.mark.parametrize("dtype_id", ("bf16", "f32"))
@pytest.mark.parametrize("shape_id", FULL_SHAPES)
def test_min(
    shape_id: str, dtype_id: str, bench: Bench, hw: Hardware, mojo_device: torch.device
):
    x_ref, x_our = both(
        unit_interval(FULL_SHAPES[shape_id], DTYPES[dtype_id]), hw, mojo_device
    )
    bench.run(
        lambda: torch.min(x_ref), lambda: torch.min(x_our), flops=float(x_ref.numel())
    )


@pytest.mark.parametrize("dtype_id", ("bool", "f32"))
@pytest.mark.parametrize("shape_id", DIM_SHAPES)
def test_count_nonzero(
    shape_id: str, dtype_id: str, bench: Bench, hw: Hardware, mojo_device: torch.device
):
    shape, dim = DIM_SHAPES[shape_id]
    src = (
        torch.rand(shape) < 0.5
        if dtype_id == "bool"
        else unit_interval(shape, DTYPES[dtype_id])
    )
    x_ref, x_our = both(src, hw, mojo_device)
    d = 0 if dim is None else dim
    bench.run(
        lambda: torch.count_nonzero(x_ref, dim=d),
        lambda: torch.count_nonzero(x_our, dim=d),
        flops=float(x_ref.numel()),
    )


@pytest.mark.parametrize("dtype_id", ("bf16", "f32"))
@pytest.mark.parametrize("shape_id", LASTDIM_SHAPES)
def test_amax(
    shape_id: str, dtype_id: str, bench: Bench, hw: Hardware, mojo_device: torch.device
):
    shape, dim = LASTDIM_SHAPES[shape_id]
    x_ref, x_our = both(unit_interval(shape, DTYPES[dtype_id]), hw, mojo_device)
    bench.run(
        lambda: torch.amax(x_ref, dim=dim),
        lambda: torch.amax(x_our, dim=dim),
        flops=float(x_ref.numel()),
    )


@pytest.mark.parametrize("dtype_id", ("bf16", "f32"))
@pytest.mark.parametrize("shape_id", LASTDIM_SHAPES)
def test_amin(
    shape_id: str, dtype_id: str, bench: Bench, hw: Hardware, mojo_device: torch.device
):
    shape, dim = LASTDIM_SHAPES[shape_id]
    x_ref, x_our = both(unit_interval(shape, DTYPES[dtype_id]), hw, mojo_device)
    bench.run(
        lambda: torch.amin(x_ref, dim=dim),
        lambda: torch.amin(x_our, dim=dim),
        flops=float(x_ref.numel()),
    )


@pytest.mark.parametrize("dtype_id", ("f32",))
@pytest.mark.parametrize("shape_id", DIM_SHAPES)
def test_argmax(
    shape_id: str, dtype_id: str, bench: Bench, hw: Hardware, mojo_device: torch.device
):
    x_ref, x_our, dim = _dim_case(shape_id, dtype_id, hw, mojo_device)
    bench.run(
        lambda: torch.argmax(x_ref, dim=dim),
        lambda: torch.argmax(x_our, dim=dim),
        flops=float(x_ref.numel()),
    )


@pytest.mark.parametrize("dtype_id", ("f32",))
@pytest.mark.parametrize("shape_id", DIM_SHAPES)
def test_argmin(
    shape_id: str, dtype_id: str, bench: Bench, hw: Hardware, mojo_device: torch.device
):
    x_ref, x_our, dim = _dim_case(shape_id, dtype_id, hw, mojo_device)
    bench.run(
        lambda: torch.argmin(x_ref, dim=dim),
        lambda: torch.argmin(x_our, dim=dim),
        flops=float(x_ref.numel()),
    )


@pytest.mark.parametrize("dtype_id", ("bool",))
@pytest.mark.parametrize("shape_id", DIM_SHAPES)
def test_all(
    shape_id: str, dtype_id: str, bench: Bench, hw: Hardware, mojo_device: torch.device
):
    shape, dim = DIM_SHAPES[shape_id]
    x_ref, x_our = both(torch.rand(shape) < 0.999, hw, mojo_device)
    if dim is None:
        bench.run(
            lambda: torch.all(x_ref),
            lambda: torch.all(x_our),
            flops=float(x_ref.numel()),
        )
    else:
        bench.run(
            lambda: torch.all(x_ref, dim=dim),
            lambda: torch.all(x_our, dim=dim),
            flops=float(x_ref.numel()),
        )


@pytest.mark.parametrize("dtype_id", ("bool",))
@pytest.mark.parametrize("shape_id", DIM_SHAPES)
def test_any(
    shape_id: str, dtype_id: str, bench: Bench, hw: Hardware, mojo_device: torch.device
):
    shape, dim = DIM_SHAPES[shape_id]
    x_ref, x_our = both(torch.rand(shape) < 0.001, hw, mojo_device)
    if dim is None:
        bench.run(
            lambda: torch.any(x_ref),
            lambda: torch.any(x_our),
            flops=float(x_ref.numel()),
        )
    else:
        bench.run(
            lambda: torch.any(x_ref, dim=dim),
            lambda: torch.any(x_our, dim=dim),
            flops=float(x_ref.numel()),
        )


@pytest.mark.parametrize("dtype_id", ("bf16", "f32"))
@pytest.mark.parametrize("shape_id", LASTDIM_SHAPES)
@pytest.mark.bench_op("min.dim")
def test_min_dim(
    shape_id: str, dtype_id: str, bench: Bench, hw: Hardware, mojo_device: torch.device
):
    shape, dim = LASTDIM_SHAPES[shape_id]
    x_ref, x_our = both(unit_interval(shape, DTYPES[dtype_id]), hw, mojo_device)
    bench.run(
        lambda: torch.min(x_ref, dim=dim),
        lambda: torch.min(x_our, dim=dim),
        flops=float(x_ref.numel()),
    )


@pytest.mark.parametrize("dtype_id", ("bf16", "f32"))
@pytest.mark.parametrize("shape_id", DIM_SHAPES)
@pytest.mark.bench_op("var.correction")
def test_var(
    shape_id: str, dtype_id: str, bench: Bench, hw: Hardware, mojo_device: torch.device
):
    x_ref, x_our, dim = _dim_case(shape_id, dtype_id, hw, mojo_device)
    if dim is None:
        bench.run(
            lambda: torch.var(x_ref, correction=1),
            lambda: torch.var(x_our, correction=1),
            flops=float(x_ref.numel()),
        )
    else:
        bench.run(
            lambda: torch.var(x_ref, dim=dim, correction=1),
            lambda: torch.var(x_our, dim=dim, correction=1),
            flops=float(x_ref.numel()),
        )


@pytest.mark.parametrize("dtype_id", ("bf16", "f32"))
@pytest.mark.parametrize("shape_id", FULL_SHAPES)
@pytest.mark.bench_op("linalg_vector_norm")
def test_vector_norm(
    shape_id: str, dtype_id: str, bench: Bench, hw: Hardware, mojo_device: torch.device
):
    x_ref, x_our = both(
        unit_interval(FULL_SHAPES[shape_id], DTYPES[dtype_id]), hw, mojo_device
    )
    bench.run(
        lambda: torch.linalg.vector_norm(x_ref),
        lambda: torch.linalg.vector_norm(x_our),
        flops=float(x_ref.numel()),
    )


@pytest.mark.parametrize("dtype_id", ("bf16", "f32"))
@pytest.mark.parametrize("shape_id", LASTDIM_SHAPES)
def test_cumsum(
    shape_id: str, dtype_id: str, bench: Bench, hw: Hardware, mojo_device: torch.device
):
    shape, dim = LASTDIM_SHAPES[shape_id]
    x_ref, x_our = both(unit_interval(shape, DTYPES[dtype_id]), hw, mojo_device)
    bench.run(
        lambda: torch.cumsum(x_ref, dim=dim),
        lambda: torch.cumsum(x_our, dim=dim),
        flops=float(x_ref.numel()),
    )


@pytest.mark.parametrize("dtype_id", ("bool",))
@pytest.mark.parametrize("shape_id", FULL_SHAPES)
def test_nonzero(
    shape_id: str, dtype_id: str, bench: Bench, hw: Hardware, mojo_device: torch.device
):
    # Fixed 50% density under the seeded fixture: the output size, and so
    # the kernel work, is identical on both legs and across runs.
    x_ref, x_our = both(torch.rand(FULL_SHAPES[shape_id]) < 0.5, hw, mojo_device)
    bench.run(
        lambda: torch.nonzero(x_ref),
        lambda: torch.nonzero(x_our),
        flops=float(x_ref.numel()),
    )


@pytest.mark.parametrize("dtype_id", ("bf16", "f32"))
@pytest.mark.parametrize("shape_id", TOPK_SHAPES)
@pytest.mark.bench_op("topk")
def test_topk(
    shape_id: str, dtype_id: str, bench: Bench, hw: Hardware, mojo_device: torch.device
):
    shape, k, largest = TOPK_SHAPES[shape_id]
    x_ref, x_our = both(unit_interval(shape, DTYPES[dtype_id]), hw, mojo_device)
    bench.run(
        lambda: torch.topk(x_ref, k, dim=-1, largest=largest),
        lambda: torch.topk(x_our, k, dim=-1, largest=largest),
        flops=float(x_ref.numel()),
    )


@pytest.mark.parametrize("dtype_id", ("bf16", "f32"))
@pytest.mark.parametrize("shape_id", SORT_SHAPES)
@pytest.mark.bench_op("sort.stable")
def test_sort(
    shape_id: str, dtype_id: str, bench: Bench, hw: Hardware, mojo_device: torch.device
):
    shape, descending = SORT_SHAPES[shape_id]
    x_ref, x_our = both(unit_interval(shape, DTYPES[dtype_id]), hw, mojo_device)
    bench.run(
        lambda: torch.sort(x_ref, dim=-1, descending=descending, stable=True),
        lambda: torch.sort(x_our, dim=-1, descending=descending, stable=True),
        flops=float(x_ref.numel()),
    )


# The HF `generate()` regime this op exists for is the first: batch 1, a
# GPT-2-sized vocabulary, one sample -- one draw per decode step (ATen's
# fast path: exponential_, div, argmax). _NOREP takes the same path with
# topk for the argmax; _REP with more than one sample is the inverse-CDF
# kernel of its own.
MULTINOMIAL_SHAPES: dict[str, tuple[tuple[int, ...], int, bool]] = {
    "V_1x50304_N1": ((1, 50304), 1, True),
    "V_8x50304_N4_NOREP": ((8, 50304), 4, False),
    "V_8x50304_N64_REP": ((8, 50304), 64, True),
    "A_357x789_N3_REP": ((357, 789), 3, True),
}


@pytest.mark.parametrize("dtype_id", ("bf16", "f32"))
@pytest.mark.parametrize("shape_id", MULTINOMIAL_SHAPES)
@pytest.mark.bench_op("multinomial")
def test_multinomial(
    shape_id: str, dtype_id: str, bench: Bench, hw: Hardware, mojo_device: torch.device
):
    # Device time only, like every other case in this suite: the two legs
    # draw from independent RNG streams, so the sampled INDICES are never
    # compared here -- only how long each backend takes to produce them.
    # Correctness (determinism, distribution, ATen edge semantics) is
    # covered in tests/native/test_factories.py.
    shape, num_samples, replacement = MULTINOMIAL_SHAPES[shape_id]
    x_ref, x_our = both(unit_interval(shape, DTYPES[dtype_id]), hw, mojo_device)
    bench.run(
        lambda: torch.multinomial(x_ref, num_samples, replacement=replacement),
        lambda: torch.multinomial(x_our, num_samples, replacement=replacement),
        flops=float(x_ref.numel()),
    )


# One order statistic per row, read off the sort kernel: median's rows are
# sorted whole (V_: a vocabulary row, several tiles; S_: one tile per row),
# kthvalue's small k takes the topk tournament and its middle k the full sort.
MEDIAN_SHAPES: dict[str, tuple[int, ...]] = {
    "V_8x50304": (8, 50304),
    "S_4096x4096": (4096, 4096),
    "A_357x789": (357, 789),
}
KTHVALUE_SHAPES: dict[str, tuple[tuple[int, ...], int]] = {
    "V_8x50304_K50": ((8, 50304), 50),
    "V_8x50304_K25152": ((8, 50304), 25152),
    "A_357x789_K32": ((357, 789), 32),
}


@pytest.mark.parametrize("dtype_id", ("bf16", "f32"))
@pytest.mark.parametrize("shape_id", MEDIAN_SHAPES)
@pytest.mark.bench_op("median.dim")
def test_median(
    shape_id: str, dtype_id: str, bench: Bench, hw: Hardware, mojo_device: torch.device
):
    shape = MEDIAN_SHAPES[shape_id]
    x_ref, x_our = both(unit_interval(shape, DTYPES[dtype_id]), hw, mojo_device)
    bench.run(
        lambda: torch.median(x_ref, dim=-1),
        lambda: torch.median(x_our, dim=-1),
        flops=float(x_ref.numel()),
    )


@pytest.mark.parametrize("dtype_id", ("bf16", "f32"))
@pytest.mark.parametrize("shape_id", KTHVALUE_SHAPES)
@pytest.mark.bench_op("kthvalue")
def test_kthvalue(
    shape_id: str, dtype_id: str, bench: Bench, hw: Hardware, mojo_device: torch.device
):
    shape, k = KTHVALUE_SHAPES[shape_id]
    x_ref, x_our = both(unit_interval(shape, DTYPES[dtype_id]), hw, mojo_device)
    bench.run(
        lambda: torch.kthvalue(x_ref, k, dim=-1),
        lambda: torch.kthvalue(x_our, k, dim=-1),
        flops=float(x_ref.numel()),
    )


@pytest.mark.parametrize("dtype_id", ("bf16", "f32"))
@pytest.mark.parametrize("shape_id", LASTDIM_SHAPES)
@pytest.mark.bench_op("max.dim")
def test_max_dim(
    shape_id: str, dtype_id: str, bench: Bench, hw: Hardware, mojo_device: torch.device
):
    shape, dim = LASTDIM_SHAPES[shape_id]
    x_ref, x_our = both(unit_interval(shape, DTYPES[dtype_id]), hw, mojo_device)
    bench.run(
        lambda: torch.max(x_ref, dim=dim),
        lambda: torch.max(x_our, dim=dim),
        flops=float(x_ref.numel()),
    )


@pytest.mark.parametrize("dtype_id", ("bf16", "f32"))
@pytest.mark.parametrize("shape_id", LASTDIM_SHAPES)
@pytest.mark.bench_op("aminmax")
def test_aminmax(
    shape_id: str, dtype_id: str, bench: Bench, hw: Hardware, mojo_device: torch.device
):
    shape, dim = LASTDIM_SHAPES[shape_id]
    x_ref, x_our = both(unit_interval(shape, DTYPES[dtype_id]), hw, mojo_device)
    bench.run(
        lambda: torch.aminmax(x_ref, dim=dim),
        lambda: torch.aminmax(x_our, dim=dim),
        flops=float(x_ref.numel()),
    )


@pytest.mark.parametrize("dtype_id", ("bf16", "f32"))
@pytest.mark.parametrize("shape_id", DIM_SHAPES)
@pytest.mark.bench_op("std.correction")
def test_std(
    shape_id: str, dtype_id: str, bench: Bench, hw: Hardware, mojo_device: torch.device
):
    x_ref, x_our, dim = _dim_case(shape_id, dtype_id, hw, mojo_device)
    bench.run(
        lambda: torch.std(x_ref, dim=dim),
        lambda: torch.std(x_our, dim=dim),
        flops=float(x_ref.numel()),
    )


@pytest.mark.parametrize("dtype_id", ("bf16", "f32"))
@pytest.mark.parametrize("shape_id", DIM_SHAPES)
@pytest.mark.bench_op("var_mean.correction")
def test_var_mean(
    shape_id: str, dtype_id: str, bench: Bench, hw: Hardware, mojo_device: torch.device
):
    x_ref, x_our, dim = _dim_case(shape_id, dtype_id, hw, mojo_device)
    bench.run(
        lambda: torch.var_mean(x_ref, dim=dim),
        lambda: torch.var_mean(x_our, dim=dim),
        flops=float(x_ref.numel()),
    )


@pytest.mark.parametrize("dtype_id", ("bf16", "f32"))
@pytest.mark.parametrize("shape_id", DIM_SHAPES)
@pytest.mark.bench_op("std_mean.correction")
def test_std_mean(
    shape_id: str, dtype_id: str, bench: Bench, hw: Hardware, mojo_device: torch.device
):
    x_ref, x_our, dim = _dim_case(shape_id, dtype_id, hw, mojo_device)
    bench.run(
        lambda: torch.std_mean(x_ref, dim=dim),
        lambda: torch.std_mean(x_our, dim=dim),
        flops=float(x_ref.numel()),
    )


@pytest.mark.parametrize("dtype_id", ("bf16", "f32"))
@pytest.mark.parametrize("shape_id", LASTDIM_SHAPES)
@pytest.mark.bench_op("cumprod")
def test_cumprod(
    shape_id: str, dtype_id: str, bench: Bench, hw: Hardware, mojo_device: torch.device
):
    shape, dim = LASTDIM_SHAPES[shape_id]
    # Factors near 1 keep the running product finite over 4096 steps.
    src = (torch.rand(shape) * 0.002 + 0.999).to(DTYPES[dtype_id])
    x_ref, x_our = both(src, hw, mojo_device)
    bench.run(
        lambda: torch.cumprod(x_ref, dim=dim),
        lambda: torch.cumprod(x_our, dim=dim),
        flops=float(x_ref.numel()),
    )


@pytest.mark.parametrize("dtype_id", ("bf16", "f32"))
@pytest.mark.parametrize("shape_id", LASTDIM_SHAPES)
@pytest.mark.bench_op("_cummax_helper")
def test_cummax(
    shape_id: str, dtype_id: str, bench: Bench, hw: Hardware, mojo_device: torch.device
):
    shape, dim = LASTDIM_SHAPES[shape_id]
    x_ref, x_our = both(unit_interval(shape, DTYPES[dtype_id]), hw, mojo_device)
    bench.run(
        lambda: torch.cummax(x_ref, dim=dim),
        lambda: torch.cummax(x_our, dim=dim),
        flops=float(x_ref.numel()),
    )


@pytest.mark.parametrize("dtype_id", ("bf16", "f32"))
@pytest.mark.parametrize("shape_id", LASTDIM_SHAPES)
@pytest.mark.bench_op("_logcumsumexp")
def test_logcumsumexp(
    shape_id: str, dtype_id: str, bench: Bench, hw: Hardware, mojo_device: torch.device
):
    shape, dim = LASTDIM_SHAPES[shape_id]
    x_ref, x_our = both(unit_interval(shape, DTYPES[dtype_id]), hw, mojo_device)
    bench.run(
        lambda: torch.logcumsumexp(x_ref, dim=dim),
        lambda: torch.logcumsumexp(x_our, dim=dim),
        flops=float(x_ref.numel()),
    )


# mode sorts each row whole (the same routes as median), then one thread per
# row walks the runs. Values drawn from a few hundred integers so runs repeat.
MODE_SHAPES: dict[str, tuple[int, ...]] = {
    "V_8x50304": (8, 50304),
    "S_4096x4096": (4096, 4096),
    "A_357x789": (357, 789),
}


@pytest.mark.parametrize("dtype_id", ("f32", "i64"))
@pytest.mark.parametrize("shape_id", MODE_SHAPES)
@pytest.mark.bench_op("mode")
def test_mode(
    shape_id: str, dtype_id: str, bench: Bench, hw: Hardware, mojo_device: torch.device
):
    shape = MODE_SHAPES[shape_id]
    src = torch.randint(0, 300, shape).to(DTYPES[dtype_id])
    x_ref, x_our = both(src, hw, mojo_device)
    bench.run(
        lambda: torch.mode(x_ref, dim=-1),
        lambda: torch.mode(x_our, dim=-1),
        flops=float(x_ref.numel()),
    )


@pytest.mark.parametrize("dtype_id", ("bf16", "f32"))
@pytest.mark.parametrize("shape_id", DIM_SHAPES)
@pytest.mark.bench_op("hash_tensor")
def test_hash_tensor(
    shape_id: str, dtype_id: str, bench: Bench, hw: Hardware, mojo_device: torch.device
):
    x_ref, x_our, dim = _dim_case(shape_id, dtype_id, hw, mojo_device)
    dims = [] if dim is None else [dim]
    bench.run(
        lambda: torch.hash_tensor(x_ref, dims),
        lambda: torch.hash_tensor(x_our, dims),
        flops=float(x_ref.numel()),
    )


HIST_SHAPES: dict[str, tuple[int, ...]] = {
    "C_16777216": (16777216,),
    "A_357x789": (357, 789),
}


@pytest.mark.parametrize("dtype_id", ("f32",))
@pytest.mark.parametrize("shape_id", HIST_SHAPES)
@pytest.mark.bench_op("histc")
def test_histc(
    shape_id: str, dtype_id: str, bench: Bench, hw: Hardware, mojo_device: torch.device
):
    x_ref, x_our = both(
        unit_interval(HIST_SHAPES[shape_id], DTYPES[dtype_id]), hw, mojo_device
    )
    bench.run(
        lambda: torch.histc(x_ref, 100, 0.0, 1.0),
        lambda: torch.histc(x_our, 100, 0.0, 1.0),
        flops=float(x_ref.numel()),
    )


@pytest.mark.parametrize("dtype_id", ("i64",))
@pytest.mark.parametrize("shape_id", ("C_16777216",))
@pytest.mark.bench_op("bincount")
def test_bincount(
    shape_id: str, dtype_id: str, bench: Bench, hw: Hardware, mojo_device: torch.device
):
    x_ref, x_our = both(torch.randint(0, 1000, HIST_SHAPES[shape_id]), hw, mojo_device)
    bench.run(
        lambda: torch.bincount(x_ref),
        lambda: torch.bincount(x_our),
        flops=float(x_ref.numel()),
    )


@pytest.mark.parametrize("dtype_id", ("bf16", "f32"))
@pytest.mark.parametrize("shape_id", LASTDIM_SHAPES)
@pytest.mark.bench_op("renorm")
def test_renorm(
    shape_id: str, dtype_id: str, bench: Bench, hw: Hardware, mojo_device: torch.device
):
    shape, dim = LASTDIM_SHAPES[shape_id]
    x_ref, x_our = both(unit_interval(shape, DTYPES[dtype_id]), hw, mojo_device)
    bench.run(
        lambda: torch.renorm(x_ref, 2, dim, 1.0),
        lambda: torch.renorm(x_our, 2, dim, 1.0),
        flops=float(x_ref.numel()),
    )


# Segments of length 1..64 along dim 0 of a (rows, 256) table.
SEGMENT_SHAPES: dict[str, tuple[int, int]] = {
    "S_65536x256": (65536, 256),
    "A_3570x789": (3570, 789),
}


@pytest.mark.parametrize("dtype_id", ("bf16", "f32"))
@pytest.mark.parametrize("shape_id", SEGMENT_SHAPES)
@pytest.mark.bench_op("segment_reduce")
def test_segment_reduce(
    shape_id: str, dtype_id: str, bench: Bench, hw: Hardware, mojo_device: torch.device
):
    rows, cols = SEGMENT_SHAPES[shape_id]
    lengths = torch.randint(1, 65, (rows,))
    lengths = lengths[torch.cumsum(lengths, 0) <= rows]
    lengths = torch.cat([lengths, torch.tensor([rows - int(lengths.sum())])])
    data = unit_interval((rows, cols), DTYPES[dtype_id])
    x_ref, x_our = both(data, hw, mojo_device)
    l_ref, l_our = both(lengths, hw, mojo_device)
    bench.run(
        lambda: torch.segment_reduce(x_ref, "sum", lengths=l_ref, unsafe=True),
        lambda: torch.segment_reduce(x_our, "sum", lengths=l_our, unsafe=True),
        flops=float(x_ref.numel()),
    )


@pytest.mark.parametrize("dtype_id", ("bf16", "f32"))
@pytest.mark.parametrize("shape_id", LASTDIM_SHAPES)
@pytest.mark.bench_op("_weight_norm_interface")
def test_weight_norm(
    shape_id: str, dtype_id: str, bench: Bench, hw: Hardware, mojo_device: torch.device
):
    shape, dim = LASTDIM_SHAPES[shape_id]
    v_ref, v_our = both(unit_interval(shape, DTYPES[dtype_id]), hw, mojo_device)
    g_shape = [1, 1]
    g_shape[dim] = shape[dim]
    g_ref, g_our = both(
        unit_interval(tuple(g_shape), DTYPES[dtype_id]), hw, mojo_device
    )
    bench.run(
        lambda: torch._weight_norm_interface(v_ref, g_ref, dim),
        lambda: torch._weight_norm_interface(v_our, g_our, dim),
        flops=float(v_ref.numel()),
    )


@pytest.mark.parametrize("dtype_id", ("bf16", "f32"))
@pytest.mark.parametrize("shape_id", ("S_8x1024x1024",))
@pytest.mark.bench_op("_compute_linear_combination")
def test_linear_combination(
    shape_id: str, dtype_id: str, bench: Bench, hw: Hardware, mojo_device: torch.device
):
    # matrix_exp's use: 8 matrix powers combined with an 8x8 coefficient table.
    i_ref, i_our = both(
        unit_interval((8, 1024, 1024), DTYPES[dtype_id]), hw, mojo_device
    )
    c_ref, c_our = both(unit_interval((8, 8), DTYPES[dtype_id]), hw, mojo_device)
    bench.run(
        lambda: torch.ops.aten._compute_linear_combination(i_ref, c_ref),
        lambda: torch.ops.aten._compute_linear_combination(i_our, c_our),
        flops=float(i_ref.numel() * 8),
    )
