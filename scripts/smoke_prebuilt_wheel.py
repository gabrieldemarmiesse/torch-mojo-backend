"""Check an installed wheel: prebuilt libraries used, no compiler run, op works.

Run it against a venv that has the wheel and one torch version installed (and
not from the source tree: it insists on an installed package). It hides every
C++ compiler from the package, registers the mojo device, reads the
registration trace, and refuses a run in which the C++ shim or the Mojo base
library was compiled instead of taken from `native/prebuilt/`. Then it runs
one op on the MAX CPU device -- the last mojo device, and the only one on a
machine with no accelerator -- so a CI runner without a GPU exercises the
whole stack down to a kernel build (a Mojo build, which needs no C++).

    python scripts/smoke_prebuilt_wheel.py
"""

from __future__ import annotations

import contextlib
import io
import os
import sys
from pathlib import Path

import torch

import torch_mojo_backend
from torch_mojo_backend import native, register_mojo_devices

_REQUIRED = ("using prebuilt C++ shim", "using prebuilt Mojo base library")
# The " in" keeps these from matching "using prebuilt C++ shim ...".
_FORBIDDEN = ("built C++ shim in", "built Mojo backend in")


def _hide_compilers():
    """Registration must succeed as if no C++ compiler were installed:
    `_find_cxx` is the package's one discovery point."""
    os.environ.pop("CXX", None)

    def no_compiler() -> list[str] | None:
        return None

    native._find_cxx = no_compiler  # ty: ignore[invalid-assignment]


def main() -> int:
    package = Path(torch_mojo_backend.__file__).resolve()
    if "site-packages" not in package.parts:
        raise SystemExit(f"expected an installed wheel, got {package}")
    print(f"torch {torch.__version__}, package {package}")
    _hide_compilers()

    # The registration trace goes to sys.stderr through print(), so redirecting
    # it here captures the two lines this test is about.
    trace = io.StringIO()
    with contextlib.redirect_stderr(trace):
        register_mojo_devices()
    log = trace.getvalue()
    print(log, end="", file=sys.stderr)
    for line in _REQUIRED:
        if line not in log:
            raise SystemExit(f"the registration trace never said {line!r}")
    for line in _FORBIDDEN:
        if line in log:
            raise SystemExit(f"the wheel compiled its own {line!r}")

    count = torch.mojo.device_count()  # ty: ignore[unresolved-attribute] -- registered by register_mojo_devices()
    cpu_device = f"mojo:{count - 1}"
    result = (torch.ones(3, device=cpu_device) * 2).cpu().tolist()
    if result != [2.0, 2.0, 2.0]:
        raise SystemExit(f"{cpu_device}: ones(3) * 2 gave {result}")
    print(f"{cpu_device}: ones(3) * 2 == {result}  OK")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
