"""Ban `-> None` return annotations.

Ruff's `suppress-none-returning` (pyproject.toml) exempts functions that
return nothing from the ANN rules, so the annotation carries no information
and is left off everywhere; this keeps it from creeping back.
"""

import re
from pathlib import Path

_REPO_DIR = Path(__file__).parent.parent
_SKIPPED_DIRS = {".venv", "deeplink", "__mojocache__", ".claude"}
_NONE_RETURN_RE = re.compile(r"\)\s*->\s*None\s*:")


def test_no_none_return_annotations():
    sources = sorted(
        path
        for path in _REPO_DIR.rglob("*.py")
        if not _SKIPPED_DIRS & set(path.relative_to(_REPO_DIR).parts)
    )
    assert sources, f"no .py sources found under {_REPO_DIR}"
    offenders = []
    for path in sources:
        text = path.read_text()
        for match in _NONE_RETURN_RE.finditer(text):
            line = text.count("\n", 0, match.start()) + 1
            offenders.append(f"{path.relative_to(_REPO_DIR)}:{line}")
    assert not offenders, "`-> None` return annotations found:\n" + "\n".join(offenders)
