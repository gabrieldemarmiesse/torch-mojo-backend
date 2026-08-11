"""Reconciliation: every op the mojo device registers is either
benchmarked or explicitly skipped with a reason.

Each family module declares COVERS (aten op -> the test that measures
it) and may declare module-local SKIPPED entries; suite-wide skips whose
reasons span families live here.  This test compares the union against
the live registration table, so a newly registered op that nobody
classified fails the suite instead of silently becoming a coverage gap,
and a deregistered op cannot leave a stale classification behind.
"""

from __future__ import annotations

import importlib

FAMILY_MODULES = (
    "test_gemm",
    "test_elementwise",
    "test_binary",
    "test_inplace",
    "test_reduction",
    "test_softmax",
    "test_norm",
    "test_attention",
    "test_embedding",
    "test_dropout_loss",
    "test_foreach",
    "test_vision",
    "test_data_movement",
)

_VIEW = "pure view/metadata op: zero-copy wrapper math, no kernel launched"
_ALLOC = "pure allocation, no kernel; the allocator is Modular's, not ours to gate"
_FILL = "alloc + the same fill kernel already benchmarked via fill_.Scalar"
_MEMCPY = (
    "device transfer/memcpy: kineto device time excludes memcpy events, "
    "unmeasurable by the suite's own rules"
)
_OUT = (
    "out-variant plumbing over an already-benchmarked functional fast impl "
    "(compute + _copy_into)"
)

# Registered ops that are deliberately NOT benchmarked, with the defense.
SKIPPED_OPS: dict[str, str] = {
    # -- views ------------------------------------------------------------
    "aten::alias": _VIEW,
    "aten::detach": _VIEW,
    "aten::view": _VIEW,
    "aten::_unsafe_view": _VIEW,
    "aten::expand": _VIEW,
    "aten::permute": _VIEW,
    "aten::transpose.int": _VIEW,
    "aten::t": _VIEW,
    "aten::select.int": _VIEW,
    "aten::slice.Tensor": _VIEW,
    "aten::unsqueeze": _VIEW,
    "aten::squeeze.dim": _VIEW,
    "aten::split.Tensor": _VIEW + " (returns views)",
    "aten::split_with_sizes": _VIEW + " (returns views)",
    "aten::unbind.int": _VIEW + " (returns views)",
    # -- allocation -------------------------------------------------------
    "aten::empty.memory_format": _ALLOC,
    "aten::empty_strided": _ALLOC,
    "aten::empty_strided.memory_format": _ALLOC,
    "aten::empty_permuted": _ALLOC,
    "aten::empty_like": _ALLOC,
    "aten::new_empty": _ALLOC,
    # -- alloc + fill -----------------------------------------------------
    "aten::zeros": _FILL,
    "aten::ones": _FILL,
    "aten::full": _FILL,
    "aten::zeros_like": _FILL,
    "aten::ones_like": _FILL,
    "aten::full_like": _FILL,
    "aten::new_zeros": _FILL,
    "aten::new_ones": _FILL,
    "aten::new_full": _FILL,
    "aten::scalar_tensor": _FILL,
    "aten::zero_": _FILL + " (delegates to fill_)",
    "aten::fill.Scalar": _FILL,
    # -- transfers / sync -------------------------------------------------
    "aten::_copy_from": _MEMCPY + " (H2D/D2H/D2D)",
    "aten::arange.start_out": _MEMCPY + " (arange + copy plumbing)",
    "aten::_local_scalar_dense": (
        "scalar extraction / sync primitive: the cost is the sync, not a kernel"
    ),
    "aten::normal_": "host-side torch RNG + upload; no device kernel of ours",
    # -- out-variant plumbing --------------------------------------------
    "aten::addcdiv.out": _OUT,
    "aten::addcmul.out": _OUT,
    "aten::div.out": _OUT,
    "aten::mul.out": _OUT,
    "aten::mean.out": _OUT,
    "aten::sub.out": _OUT,
    "aten::any.out": _OUT,
    "aten::lerp.Scalar_out": _OUT,
    "aten::isin.Tensor_Tensor_out": _OUT,
    "aten::min.dim_min": _OUT,
    # -- not implemented --------------------------------------------------
    "aten::_adaptive_avg_pool2d_backward": (
        "registered as an explicit raiser (_register_missing): no fast impl "
        "exists, nothing to measure"
    ),
}


def test_every_registered_op_is_classified() -> None:
    from torch_mojo_backend.mojo_device import mojo_device_aten_ops as reg

    registered = {name for name, _ in reg._aten_ops_registry}
    covered: dict[str, str] = {}
    skipped: dict[str, str] = dict(SKIPPED_OPS)
    for module_name in FAMILY_MODULES:
        module = importlib.import_module(module_name)
        covered.update(module.COVERS)
        skipped.update(module.SKIPPED)

    overlap = sorted(set(covered) & set(skipped))
    assert not overlap, f"ops classified as both covered and skipped: {overlap}"

    unclassified = sorted(registered - set(covered) - set(skipped))
    assert not unclassified, (
        "newly registered ops with no benchmark and no documented skip "
        f"reason: {unclassified}. Add a benchmark to the matching family "
        "module (and its COVERS entry), or a reasoned entry to SKIPPED_OPS "
        "in benchmarks/test_coverage.py."
    )

    stale = sorted((set(covered) | set(skipped)) - registered)
    assert not stale, (
        f"classified ops that are no longer registered: {stale}. Remove the "
        "stale COVERS/SKIPPED entries (and any benchmark of a dropped op)."
    )
