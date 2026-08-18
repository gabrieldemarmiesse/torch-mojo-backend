"""Codegen guard: FA4's softmax exponent must not be a fused multiply-add.

FA4's online softmax computes ``P = exp2(<scaled score> - <row statistic>)``.
The scaled score must be ROUNDED on its own before the subtraction: a single
``fma(s, scale_log2, -m)`` keeps the product exact while ``m`` is rounded,
which leaves the winning element's exponent off zero by up to half an f32 ulp
of ``|s * scale_log2|`` -- an error that widens with the logit magnitude and,
at the magnitudes of the 2026-08 nanoGPT attention-collapse incident, cost a
16x saturated-row rms regression against cuda flash. The source therefore
spells the product as ``s.fma[FastMathFlag.NONE](scale_log2, 0)`` and
subtracts afterwards.

That spelling is a request, not a guarantee: floating-point contraction is a
compiler privilege, ``SIMD.fma`` defaults to ``FastMathFlag.CONTRACT``, and
``s * scale_log2 - m`` written plainly IS contracted straight back into the
defective single fma by today's toolchain. So the source form alone cannot be
trusted to survive a toolchain upgrade, and a numerical regression test only
covers the instantiations it can reach at runtime. This test reads the
emitted PTX instead and covers EVERY instantiation the module compiles --
both forward kernels and the backward P-recompute, bf16 and f16, d64 and
d128 -- by asserting the property directly on the machine code:

    no ``ex2.approx`` input may be produced by an ``fma``.

Cross-compiles with ``mojo build --emit asm --target-accelerator sm_90a``
(needs no GPU, never touches ``/tmp/gpu_lock_0.lock``), the same way
``scripts/compare_kernel_asm.py`` and ``test_fa4_selfload_ptx_ordering.py``
do. Measured against the stock kernels this check is a total discriminator:
1728 of the 1728 ``ex2`` sites are fma-fed before the fix and 0 after.

Builds land in a STABLE directory (``tests/__mojocache__/...``) for the
reason spelled out in ``test_fa4_selfload_ptx_ordering.py``: Mojo's transform
cache replays a kernel's sidecar to the path it was FIRST built at, so a
fresh temp dir per run goes empty on the second run in a session.
"""

from __future__ import annotations

import hashlib
import re
import subprocess
from pathlib import Path

import pytest

from scripts.compare_kernel_asm import build_env, mojo_cli

_REPO_ROOT = Path(__file__).resolve().parents[1]
_FA4_DIR = _REPO_ROOT / "torch_mojo_backend" / "eager_flash_attention"
_ENTRY = _FA4_DIR / "fa4_ops.mojo"
_BUILD_DIR = Path(__file__).resolve().parent / "__mojocache__" / "fa4_rounded_product"
# Ties the transform-cache entry to THIS checkout's output path; see the
# ASM_CACHE_BUST comment in _build_fa4_ptx.
_BUILD_TAG = hashlib.sha256(str(_BUILD_DIR).encode()).hexdigest()[:16]

# `<op> %dst, <args>;` -- enough to attribute each register to its last writer.
_DEFINITION = re.compile(r"^\s*([a-z0-9.]+)\s+(%[A-Za-z0-9_]+)\s*,\s*(.*);\s*$")
_EX2 = re.compile(r"ex2\.approx\.ftz\.f32\s+%[A-Za-z0-9_]+\s*,\s*(%[A-Za-z0-9_]+)")
# The three kernel families that carry a P = exp2(...) expression, keyed by a
# substring of their sidecar file name. Sidecars are named
# `<module>_<kernel>_<hash>.ptx` and the kernel half is truncated, so these
# match on the module name plus whatever survives of the kernel name.
_P_LOOP_KERNELS = (
    "fa4_fwd_kernel_fwd_fa4_kernel",
    "fa4_fwd_selfload_kernel",
    "fa4_bwd_kernel_bwd_main_kernel",
)


def _build_fa4_ptx() -> list[Path]:
    """Cross-compile fa4_ops for sm_90a; return its per-kernel PTX sidecars."""
    try:
        mojo = mojo_cli()
    except FileNotFoundError:
        pytest.skip("mojo compiler not found")
    _BUILD_DIR.mkdir(parents=True, exist_ok=True)
    # The directory is stable across runs, so sidecars from an older source
    # revision would otherwise linger and be scanned as if they were current.
    for stale in list(_BUILD_DIR.glob("*.ptx")) + list(_BUILD_DIR.glob("*.s")):
        stale.unlink()
    result = subprocess.run(
        [
            str(mojo),
            "build",
            str(_ENTRY),
            "-I",
            str(_FA4_DIR),
            "--emit",
            "asm",
            "--target-accelerator",
            "sm_90a",
            # Mojo's transform cache keys on source and defines but NOT on the
            # output path, and services a hit by rewriting the sidecars to the
            # path the entry was FIRST built at. An identical fa4_ops built
            # earlier by scripts/compare_kernel_asm.py (into its own --work-dir,
            # since deleted) therefore makes this build fail with "could not
            # open offload output file". Nothing reads this define, so it cannot
            # move the generated code; it only makes the cache entry belong to
            # this output path, which stays a hit on every later run.
            "-D",
            f"ASM_CACHE_BUST={_BUILD_TAG}",
            "-o",
            str(_BUILD_DIR / "fa4_ops.s"),
        ],
        cwd=str(_REPO_ROOT),
        env=build_env(),
        capture_output=True,
        text=True,
        timeout=1800,
    )
    assert result.returncode == 0, (
        f"mojo build --emit asm failed for fa4_ops:\n{result.stderr or result.stdout}"
    )
    sidecars = sorted(_BUILD_DIR.glob("*.ptx"))
    assert sidecars, (
        "mojo build --emit asm produced no .ptx sidecar -- no GPU kernel was"
        " instantiated, so this check would be vacuous"
    )
    return sidecars


@pytest.fixture(scope="module")
def fa4_ptx() -> dict[str, str]:
    """Every FA4 GPU kernel's PTX, keyed by sidecar file name."""
    return {path.name: path.read_text() for path in _build_fa4_ptx()}


def _classify_exp2_inputs(ptx: str) -> tuple[int, int, int]:
    """Count ``ex2`` sites by what produced their input.

    Returns ``(fma_fed, rounded_product_fed, other)``:

    - ``fma_fed`` is the defect signature -- ``fma.rn.f32 %d, %s, %c, %negm``
      feeding ``ex2`` directly, i.e. an exact product against a rounded row
      statistic;
    - ``rounded_product_fed`` is the fixed form -- ``sub.f32`` of an
      ``fma.rn.f32 ..., 0f00000000`` (the correctly rounded product) and the
      row statistic;
    - ``other`` is every remaining ``ex2``: the online-rescale
      ``exp2(old_max - new_max)`` corrections and the masked-row constants,
      which are subtractions of two already-rounded values and are not part
      of the P expression.
    """
    definitions: dict[str, tuple[str, str]] = {}
    fma_fed = rounded = other = 0
    for line in ptx.splitlines():
        consumed = _EX2.search(line)
        if consumed is not None:
            operation, arguments = definitions.get(consumed.group(1), ("", ""))
            if operation.startswith("fma"):
                fma_fed += 1
            elif operation == "sub.f32":
                minuend = arguments.split(",")[0].strip()
                producer, producer_arguments = definitions.get(minuend, ("", ""))
                if producer.startswith("fma") and producer_arguments.rstrip().endswith(
                    "0f00000000"
                ):
                    rounded += 1
                else:
                    other += 1
            else:
                other += 1
        defined = _DEFINITION.match(line)
        if defined is not None:
            definitions[defined.group(2)] = (defined.group(1), defined.group(3))
    return fma_fed, rounded, other


def test_no_fa4_exp2_input_is_produced_by_an_fma(fa4_ptx: dict[str, str]) -> None:
    """The defect signature must not appear in any emitted FA4 kernel."""
    offenders = {
        name: _classify_exp2_inputs(ptx)[0]
        for name, ptx in fa4_ptx.items()
        if _classify_exp2_inputs(ptx)[0]
    }
    assert not offenders, (
        "an FA4 exp2 input is produced directly by an fma, so the scaled"
        " score is an EXACT product being compared against a ROUNDED row"
        " statistic -- the extreme-logit defect this fix removed. Either the"
        " source regressed to fma(s, scale_log2, -m), or the toolchain"
        " contracted the rounded product back into the subtraction."
        f" Offending kernels: {offenders}"
    )


def test_every_fa4_p_loop_uses_a_separately_rounded_product(
    fa4_ptx: dict[str, str],
) -> None:
    """The positive half: the rounded-product form is actually present.

    Without this, deleting the whole P loop (or emitting exp2 of something
    unrelated) would silently satisfy the negative check above.
    """
    for kernel in _P_LOOP_KERNELS:
        sidecars = [name for name in fa4_ptx if kernel in name]
        assert sidecars, (
            f"no PTX sidecar for {kernel!r} -- fa4_ops stopped instantiating"
            " a kernel this guard is supposed to cover, or the sidecar naming"
            " changed"
        )
        for name in sidecars:
            _, rounded, _ = _classify_exp2_inputs(fa4_ptx[name])
            assert rounded >= 8, (
                f"{name} has only {rounded} exp2 site(s) fed by a separately"
                " rounded product; the P loop of this kernel should have"
                " dozens, so the guard is not actually watching it"
            )
