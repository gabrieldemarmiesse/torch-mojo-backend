"""The float64 specs of the unary ops whose math has no GPU lowering still build.

float64 acos, atanh, cos, sin, sinh and tan have no GPU lowering: on NVIDIA,
std.math refuses them ("libm operations are only available on CPU targets",
"DType.float64 is not supported for cos on NVIDIA GPU", "no libcall
available for facos"), and on AMD LLVM crashes ("Cannot select: f64 =
fcos"). The elementwise family's dtype gate never admits those float64
specs, so its float64 route (`_unary_float64_on` in
`tmb/kernels/elementwise/entry.mojo`) must not instantiate their kernel
either; a build of such a spec compiles the gate's error path and nothing
else.

Cross-compiles with ``mojo build --emit asm --target-accelerator sm_90a``
like ``test_fa4_selfload_ptx_ordering.py``: no GPU needed, no GPU lock.
"""

from __future__ import annotations

import subprocess
from pathlib import Path

import pytest

from scripts.compare_kernel_asm import build_env, mojo_cli

_REPO_ROOT = Path(__file__).resolve().parents[1]
_MOJO_ROOT = _REPO_ROOT / "torch_mojo_backend" / "mojo"
_ENTRY = _MOJO_ROOT / "tmb" / "kernels" / "elementwise" / "entry.mojo"
# Stable, gitignored build directory: see test_fa4_selfload_ptx_ordering.py
# for why a fresh temp dir per run breaks Mojo's transform cache.
_BUILD_DIR = Path(__file__).resolve().parent / "__mojocache__" / "unary_float64"


@pytest.mark.parametrize(
    "op", ["AcosSpec", "AtanhSpec", "CosSpec", "SinSpec", "SinhSpec", "TanSpec"]
)
def test_libm_float64_unary_spec_builds_for_gpu(op: str):
    try:
        mojo = mojo_cli()
    except FileNotFoundError:
        pytest.skip("mojo compiler not found")
    out_dir = _BUILD_DIR / op
    out_dir.mkdir(parents=True, exist_ok=True)
    result = subprocess.run(
        [
            str(mojo),
            "build",
            str(_ENTRY),
            "-I",
            str(_MOJO_ROOT),
            "--emit",
            "asm",
            "--target-accelerator",
            "sm_90a",
            "-D",
            f"OP={op}",
            "-D",
            "DTYPE_ARG_0=float64",
            "-o",
            str(out_dir / "entry.s"),
        ],
        cwd=str(_REPO_ROOT),
        env=build_env(),
        capture_output=True,
        text=True,
        timeout=600,
    )
    assert result.returncode == 0, (
        f"the float64 {op} elementwise spec does not build for the GPU:\n"
        f"{result.stderr or result.stdout}"
    )
