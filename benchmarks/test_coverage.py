"""Two reconciliations, both GPU-free and both about things that fail
SILENTLY rather than loudly.

1. Every op the mojo device registers is either benchmarked or
   explicitly skipped with a reason.  Each family module declares COVERS
   (aten op -> the test that measures it) and may declare module-local
   SKIPPED entries; suite-wide skips whose reasons span families live
   here.  The union is compared against the live registration table, so a
   newly registered op that nobody classified fails the suite instead of
   becoming a coverage gap, and a deregistered op cannot leave a stale
   classification behind.

2. Every recorded baseline is still addressed by a test node.  A key
   whose node disappeared — renamed shape token, dropped parametrize
   case, deleted test — never fails anything: it just stops protecting
   its kernel regime, and the op silently re-measures as a "new entry" on
   the machine that owns the baselines.  Nothing else in the suite
   notices, which is exactly why this check exists.
"""

from __future__ import annotations

import importlib
import os
import re
import subprocess
import sys
import tempfile
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))

from bench_lib import baselines

# conftest is resolved via the sys.path.insert above (this script also runs
# standalone, not just under pytest's conftest auto-discovery); ty's module
# resolver doesn't follow that runtime path manipulation. (A real fix is
# `environment.root = ["benchmarks"]` in [tool.ty], mirroring the sys.path
# insert -- left for a repo-wide config change rather than a local workaround.)
from conftest import KEY_DUMP_ENV  # ty: ignore[unresolved-import]

BENCH_DIR = Path(__file__).resolve().parent
NATIVE_MOJO_DIR = BENCH_DIR.parent / "torch_mojo_backend" / "native" / "mojo"

# `impl[op_add_tensor](lib, "add.Tensor")` in an ops_*.mojo file IS the
# registration (see docs/native_backend.md): the backend registers its ops
# from Mojo, so the list is read from the source rather than imported.
# `impl[op_x, "name.overload"](site)` in each register_<group> (registry.mojo);
# the formatter may break the call over several lines
_IMPL_RE = re.compile(r'impl\[\s*\w+\s*,\s*"([^"]+)"\s*,?\s*\]\s*\(', re.S)


def registered_ops() -> set[str]:
    names: set[str] = set()
    for path in sorted(NATIVE_MOJO_DIR.glob("*.mojo")):
        names |= {f"aten::{name}" for name in _IMPL_RE.findall(path.read_text())}
    if not names:
        raise AssertionError(f"no aten op registrations found under {NATIVE_MOJO_DIR}")
    return names


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

_VIEW = "pure view/metadata op: zero-copy metadata math, no kernel launched"
_ALLOC = "pure allocation, no kernel; the allocator is Modular's, not ours to gate"
_FILL = "alloc + the same fill kernel already benchmarked via fill_.Scalar"
_MEMCPY = (
    "device transfer/memcpy: kineto device time excludes memcpy events, "
    "unmeasurable by the suite's own rules"
)
_OUT = (
    "out-variant plumbing over an already-benchmarked functional impl "
    "(compute, then copy_strided_into when the caller's tensor isn't the "
    "right shape/dtype/layout to compute into directly)"
)

_COMPOSED = (
    "no kernel of its own: ops_composed.mojo builds it from ops this suite "
    "already measures (a few extra launches, nothing new to regress against)"
)
_HOST_RNG = "host-side torch RNG + upload; no device kernel of ours"

# Registered ops that are deliberately NOT benchmarked, with the defense.
SKIPPED_OPS: dict[str, str] = {
    # -- views ------------------------------------------------------------
    "aten::as_strided": _VIEW,
    "aten::view": _VIEW,
    "aten::_unsafe_view": _VIEW,
    "aten::_reshape_alias": _VIEW,
    # -- allocation -------------------------------------------------------
    "aten::empty.memory_format": _ALLOC,
    "aten::empty_strided": _ALLOC,
    "aten::empty_permuted": _ALLOC,
    # -- alloc + fill -----------------------------------------------------
    "aten::zero_": _FILL + " (delegates to fill_)",
    "aten::fill.Scalar": _FILL,
    # -- transfers / sync -------------------------------------------------
    "aten::_copy_from": _MEMCPY + " (H2D/D2H/D2D)",
    "aten::_local_scalar_dense": (
        "scalar extraction / sync primitive: the cost is the sync, not a kernel"
    ),
    "aten::normal_": _HOST_RNG,
    "aten::record_stream": (
        "stream-lifetime bookkeeping, not compute: records a MAX event on "
        "the named stream so a buffer's free is fenced behind a foreign "
        "reader (native/mojo/device.mojo). No kernel launches, so there is "
        "no device time to measure"
    ),
    # -- metadata-only mutation -------------------------------------------
    "aten::set_.source_Tensor": (
        "repoints a tensor at another's allocation in place: storage swap "
        "plus TensorImpl metadata, zero-copy and no kernel"
    ),
    # -- dispatch decision, not compute -----------------------------------
    "aten::_fused_sdp_choice": (
        "returns which SDPA backend to use as an int; the attention kernel "
        "it selects is what test_attention measures"
    ),
    # -- CPU-device-only attention ----------------------------------------
    "aten::_scaled_dot_product_flash_attention_for_cpu": (
        "ATen routes this overload only on the MAX CPU device; the suite "
        "measures the accelerator against stock GPU torch, so there is no "
        "comparable reference leg"
    ),
    "aten::_scaled_dot_product_flash_attention_for_cpu_backward": (
        "backward of the CPU-device-only overload above, same reason"
    ),
    # -- out-variant plumbing --------------------------------------------
    "aten::abs.out": _OUT,
    "aten::acos.out": _OUT,
    "aten::add.out": _OUT,
    "aten::addcdiv.out": _OUT,
    "aten::addcmul.out": _OUT,
    "aten::addmm.out": _OUT,
    "aten::any.out": _OUT,
    "aten::asinh.out": _OUT,
    "aten::atanh.out": _OUT,
    "aten::bitwise_not.out": _OUT,
    "aten::bmm.out": _OUT,
    "aten::bucketize.Scalar_out": _OUT,
    "aten::bucketize.Tensor_out": _OUT,
    "aten::cat.out": _OUT,
    "aten::ceil.out": _OUT,
    "aten::cos.out": _OUT,
    "aten::cosh.out": _OUT,
    "aten::div.out": _OUT,
    "aten::div.out_mode": _OUT,
    "aten::eq.Scalar_out": _OUT,
    "aten::eq.Tensor_out": _OUT,
    "aten::erf.out": _OUT,
    "aten::exp.out": _OUT,
    "aten::floor.out": _OUT,
    "aten::ge.Scalar_out": _OUT,
    "aten::ge.Tensor_out": _OUT,
    "aten::gelu.out": _OUT,
    "aten::gt.Scalar_out": _OUT,
    "aten::gt.Tensor_out": _OUT,
    "aten::isin.Tensor_Tensor_out": _OUT,
    "aten::isnan.out": _OUT,
    "aten::le.Scalar_out": _OUT,
    "aten::le.Tensor_out": _OUT,
    "aten::lerp.Scalar_out": _OUT,
    "aten::log.out": _OUT,
    "aten::log1p.out": _OUT,
    "aten::logical_not.out": _OUT,
    "aten::lt.Scalar_out": _OUT,
    "aten::lt.Tensor_out": _OUT,
    "aten::masked_fill.Scalar_out": _OUT,
    "aten::masked_fill.Tensor_out": _OUT,
    "aten::mean.out": _OUT,
    "aten::min.dim_min": _OUT,
    "aten::mm.out": _OUT,
    "aten::mul.out": _OUT,
    "aten::ne.Scalar_out": _OUT,
    "aten::ne.Tensor_out": _OUT,
    "aten::neg.out": _OUT,
    "aten::reciprocal.out": _OUT,
    "aten::relu.out": _OUT,
    "aten::rsqrt.out": _OUT,
    "aten::searchsorted.Scalar_out": _OUT,
    "aten::searchsorted.Tensor_out": _OUT,
    "aten::sigmoid.out": _OUT,
    "aten::sign.out": _OUT,
    "aten::silu.out": _OUT,
    "aten::sin.out": _OUT,
    "aten::sinh.out": _OUT,
    "aten::sqrt.out": _OUT,
    "aten::sub.out": _OUT,
    "aten::where.self_out": _OUT,
    "aten::tan.out": _OUT,
    "aten::tanh.out": _OUT,
    # -- composed from already-benchmarked ops ------------------------------
    "aten::threshold_backward": _COMPOSED,
    "aten::threshold_backward.grad_input": _COMPOSED,
    "aten::sigmoid_backward": _COMPOSED,
    "aten::sigmoid_backward.grad_input": _COMPOSED,
    "aten::tanh_backward": _COMPOSED,
    "aten::tanh_backward.grad_input": _COMPOSED,
    "aten::isneginf": _COMPOSED,
    "aten::isneginf.out": _COMPOSED,
    "aten::isposinf": _COMPOSED,
    "aten::isposinf.out": _COMPOSED,
    # -- host-side RNG ------------------------------------------------------
    "aten::random_": _HOST_RNG,
    "aten::random_.from": _HOST_RNG,
    "aten::random_.to": _HOST_RNG,
    # -- new op, no benchmark yet -------------------------------------------
    "aten::_softmax_backward_data": (
        "newly registered; test_softmax covers the forward and the backward "
        "shares _log_softmax_backward_data's reduce-and-scale shape, but it "
        "has no benchmark node of its own yet"
    ),
    "aten::native_batch_norm_backward": (
        "newly registered; test_batch_norm covers the forward, and the "
        "backward has no benchmark node of its own yet"
    ),
    "aten::addr": (
        "newly added fast kernel (see fix-addr-fp16-bf16-precision) fixes "
        "fp16/bf16 rounding-order drift vs CPU; it has no prior native "
        "kernel to regress against (ATen's own CompositeExplicitAutograd "
        "fallback ran before this, composing already-benchmarked mul/add), "
        "so there is no throughput baseline to protect yet. A dedicated "
        "benchmark is a reasonable follow-up, not required for a precision "
        "fix that doesn't touch performance-critical code paths."
    ),
}


def test_every_registered_op_is_classified():
    registered = registered_ops()
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


def _addressable_keys() -> set[str]:
    """Every baseline path the suite can produce, from a collection run.

    Collection only — no benchmark runs, no GPU, no device touched.  It
    happens in a SUBPROCESS over the whole benchmarks/ directory on
    purpose: reading this session's own items would make the answer depend
    on how the suite was selected, and `-k something` would then report
    every unselected baseline as an orphan.
    """
    with tempfile.TemporaryDirectory() as tmp:
        dump = Path(tmp) / "keys.txt"
        proc = subprocess.run(
            [sys.executable, "-m", "pytest", str(BENCH_DIR), "--collect-only", "-q"],
            env={**os.environ, KEY_DUMP_ENV: str(dump)},
            cwd=BENCH_DIR.parent,
            capture_output=True,
            text=True,
        )
        assert dump.exists(), (
            "collecting the benchmark suite in a subprocess produced no key "
            f"dump (exit {proc.returncode}).\nstdout:\n{proc.stdout[-2000:]}\n"
            f"stderr:\n{proc.stderr[-2000:]}"
        )
        return set(dump.read_text().split())


AXES = ("op", "dtype", "shape", "layout")


def _diagnose(orphans: list[str], keys: set[str]) -> str:
    """Name what went missing, not just which paths broke.

    Renaming one shape token orphans every op that uses it, so the raw
    list is hundreds of paths with one cause.  Report the axis VALUES the
    suite no longer produces, with their blast radius, and fall back to
    listing paths when every value still exists and only the combination
    is gone (a dropped parametrize case).
    """
    live = [{key.split("/")[i] for key in keys} for i in range(len(AXES))]
    unknown: dict[str, int] = {}
    combinations = []
    for orphan in orphans:
        parts = orphan.split("/")
        missing = [
            f"{AXES[i]} {part!r}" for i, part in enumerate(parts) if part not in live[i]
        ]
        if missing:
            for token in missing:
                unknown[token] = unknown.get(token, 0) + 1
        else:
            combinations.append(orphan)
    lines = [
        f"  no test node produces {token} — {count} recorded entr"
        f"{'y' if count == 1 else 'ies'} under it"
        for token, count in sorted(unknown.items(), key=lambda kv: -kv[1])
    ]
    if combinations:
        shown = combinations[:10]
        lines.append(
            f"  {len(combinations)} entr{'y' if len(combinations) == 1 else 'ies'} "
            f"whose axes all exist but whose combination is gone: {shown}"
            + (" (first 10)" if len(combinations) > len(shown) else "")
        )
    return "\n".join(lines)


def test_every_recorded_baseline_is_still_addressable():
    keys = _addressable_keys()
    assert keys, "the collection run found no benchmark nodes at all"

    # Across every hardware config: a machine that measured a case still
    # owns that baseline even when this machine never runs it.
    recorded = {
        str(key)
        for config in baselines.load().configs.values()
        for key in config.leaves()
    }
    orphans = sorted(recorded - keys)
    assert not orphans, (
        f"{len(orphans)} of {len(recorded)} recorded baselines are no longer "
        f"addressed by any test node:\n{_diagnose(orphans, keys)}\n"
        "Each one silently protects nothing — the case it was measured for "
        "will re-record as a new entry on the machine that owns it. If a "
        "token was renamed, rename the recorded entries with it (a shape id "
        "IS a baseline key); if the case was dropped for good, delete its "
        "entries from the data block of benchmarks/baselines.html."
    )
