"""The eager-mode house rules that are properties of the kernel SOURCES.

AGENTS.md, "Rules about the eager mode": the point of this project is that a
CPU-only PyTorch install plus a Mojo compiler drives the GPU, so no kernel may
reach for a vendor BLAS/DNN library; and a kernel runs on the DeviceContext
its caller hands it, so it may not synchronize on its own -- that would
serialize a user's stream behind our op.

Nothing else catches a violation: an import of a vendor-backed routine
compiles and runs, and the answer is right. It just quietly makes the wheel
depend on cuBLAS. This check used to live inside one gemm16 test in
tests/test_eager_kernels.py (deleted with the old eager path); it applies to
every family, so it is repo-wide here.
"""

import re
from pathlib import Path

import pytest

PACKAGE = Path(__file__).resolve().parent.parent / "torch_mojo_backend"
KERNEL_ROOTS = (PACKAGE / "eager_kernels", PACKAGE / "eager_flash_attention")

# Lowercased substrings that must not appear in kernel CODE. Prose is
# exempt: a comment or docstring recording that a kernel was benchmarked
# against cuBLAS is a measurement note, not a dependency (an earlier version
# of this check tripped on exactly that).
VENDOR_LIBRARIES = ("cublas", "cudnn", "rocblas", "miopen", "triton")

# Modular's own Mojo kernels under `linalg` are fine -- that is the
# documented way to reuse them. `linalg.vendor_blas` is the one subpackage
# that dispatches to the vendor library instead.
VENDOR_ROUTES = ("from linalg.vendor_blas", "import vendor_blas")

_DOCSTRING = re.compile(r'"""(?:.|\n)*?"""')


def _kernel_sources() -> list[Path]:
    paths = [p for root in KERNEL_ROOTS for p in sorted(root.rglob("*.mojo"))]
    assert paths, f"no kernel sources found under {[str(r) for r in KERNEL_ROOTS]}"
    return paths


def _code_only(source: str) -> str:
    """Drop docstrings and `#` comments: the rules are about imports and
    calls, not about what a tuning note mentions."""
    without_docstrings = _DOCSTRING.sub("", source)
    return "\n".join(
        line.split("#", 1)[0] for line in without_docstrings.splitlines()
    ).lower()


@pytest.mark.parametrize(
    "path", _kernel_sources(), ids=lambda p: str(p.relative_to(PACKAGE))
)
def test_kernel_source_takes_no_vendor_library(path: Path):
    code = _code_only(path.read_text())
    for forbidden in VENDOR_LIBRARIES + VENDOR_ROUTES:
        assert forbidden not in code, (
            f"{path.relative_to(PACKAGE)} reaches for {forbidden!r}. The whole "
            "point of this backend is that a CPU-only torch install drives the "
            "GPU (AGENTS.md, 'Rules about the eager mode', rule 1)."
        )


@pytest.mark.parametrize(
    "path", _kernel_sources(), ids=lambda p: str(p.relative_to(PACKAGE))
)
def test_kernel_source_does_not_synchronize(path: Path):
    code = _code_only(path.read_text())
    assert ".synchronize(" not in code, (
        f"{path.relative_to(PACKAGE)} synchronizes: a kernel runs on the "
        "DeviceContext its caller hands it (native/mojo/abi.mojo's `ctx_for`) "
        "and returns; blocking inside an op serializes the caller's stream."
    )
