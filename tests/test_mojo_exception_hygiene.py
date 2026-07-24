"""Guard against silent error swallowing in the Mojo kernel sources.

A dispatcher that catches a kernel error and does nothing returns an
apparently valid result backed by uninitialized memory, and disables the
loader's unsupported-dtype escalation (which relies on the error reaching
Python). 36 such handlers were found and fixed at once; this test keeps the
pattern from coming back.

An exception handler must do something observable: propagate the error to
Python (``_spec_unsupported``), re-raise, abort, or apply an explicit,
commented fallback value. A body consisting only of ``pass`` (comments
aside) is never acceptable.
"""

import re
from pathlib import Path

_BACKEND_DIR = Path(__file__).parent.parent / "torch_mojo_backend"

# `except ...:` then a body whose first non-comment statement is `pass`.
_SWALLOW_RE = re.compile(
    r"^(?P<indent>[ \t]*)except[^\n]*:\n"
    r"(?:(?P=indent)[ \t]+#[^\n]*\n)*"
    r"(?P=indent)[ \t]+pass[ \t]*(?:#[^\n]*)?\n",
    re.M,
)


def test_no_pass_only_exception_handlers() -> None:
    sources = sorted(_BACKEND_DIR.rglob("*.mojo"))
    assert sources, f"no .mojo sources found under {_BACKEND_DIR}"
    offenders = []
    for path in sources:
        text = path.read_text()
        for match in _SWALLOW_RE.finditer(text):
            line = text.count("\n", 0, match.start()) + 1
            offenders.append(f"{path.relative_to(_BACKEND_DIR.parent)}:{line}")
    assert not offenders, (
        "pass-only exception handlers swallow kernel errors (uninitialized "
        "results, broken dtype escalation). Propagate via _spec_unsupported, "
        "re-raise, or apply an explicit fallback value instead:\n  "
        + "\n  ".join(offenders)
    )
