"""Guard against silent error swallowing in the Mojo kernel sources.

A dispatcher that catches a kernel error and reports success returns an
apparently valid result backed by uninitialized memory, and it defeats the
call queue's only error channel: a queued launch that raises nothing never
reaches ``_HELD_ERROR``, so ``drain()`` has nothing to re-raise and the
failure is invisible to Python. 36 such handlers were found and fixed at
once; this test keeps the pattern from coming back.

An exception handler must do something observable: propagate the error to
Python (``_spec_unsupported``), re-raise, abort, or apply an explicit,
commented fallback value. A body consisting only of ``pass`` (comments
aside) is never acceptable, and neither is the success-sentinel spelling of
the same mistake, ``return _raw_ret_none()`` -- the value an `abi("C")`
dispatcher returns when the kernel ran fine.
"""

import re
from pathlib import Path

_BACKEND_DIR = Path(__file__).parent.parent / "torch_mojo_backend"

# `except ...:` then a body whose first non-comment statement claims success:
# either `pass` or the `-> PyObjectPtr` success sentinel `_raw_ret_none()`.
_SWALLOW_RE = re.compile(
    r"^(?P<indent>[ \t]*)except[^\n]*:\n"
    r"(?:(?P=indent)[ \t]+#[^\n]*\n)*"
    r"(?P=indent)[ \t]+(?:pass|return[ \t]+_raw_ret_none\(\))[ \t]*(?:#[^\n]*)?\n",
    re.M,
)


def test_no_success_reporting_exception_handlers() -> None:
    sources = sorted(_BACKEND_DIR.rglob("*.mojo"))
    assert sources, f"no .mojo sources found under {_BACKEND_DIR}"
    offenders = []
    for path in sources:
        text = path.read_text()
        for match in _SWALLOW_RE.finditer(text):
            line = text.count("\n", 0, match.start()) + 1
            offenders.append(f"{path.relative_to(_BACKEND_DIR.parent)}:{line}")
    assert not offenders, (
        "these exception handlers report success after a kernel error "
        "(uninitialized results, and nothing for the call queue to re-raise "
        "at the next drain). Propagate via _spec_unsupported, re-raise, or "
        "apply an explicit fallback value instead:\n  " + "\n  ".join(offenders)
    )
