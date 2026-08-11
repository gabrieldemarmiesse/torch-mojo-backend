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
import subprocess
import sys
import tempfile
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))

from bench_lib import baselines
from conftest import KEY_DUMP_ENV

BENCH_DIR = Path(__file__).resolve().parent

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


def test_every_recorded_baseline_is_still_addressable() -> None:
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
