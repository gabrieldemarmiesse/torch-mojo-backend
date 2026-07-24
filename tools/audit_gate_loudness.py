"""Audit: every dtype-gated comptime loop must be LOUD on miss.

A gated-out dtype must surface as a raised error (which the loader catches to
escalate), never as silently-unwritten output. For each `comptime if _dt_on[`
site, find the enclosing function and check that a `raise` exists after the
loop (the `handled`-flag idiom) — flag functions where it doesn't.
"""

import re
import sys


def audit(path: str) -> list[str]:
    lines = open(path).read().splitlines()
    # map: function start line -> (name, indent)
    suspects = []
    gate_lines = [i for i, line in enumerate(lines) if "comptime if _dt_on[" in line]
    for gl in gate_lines:
        # enclosing def: nearest previous line matching top-level or nested def
        fn_line, fn_name, fn_indent = None, "?", 0
        for j in range(gl, -1, -1):
            m = re.match(r"^(\s*)def (\w+)", lines[j])
            if m and len(m.group(1)) < len(lines[gl]) - len(lines[gl].lstrip()):
                fn_line, fn_name, fn_indent = j, m.group(2), len(m.group(1))
                break
        # function end: next line at indent <= fn_indent that starts a new def/decorator
        end = len(lines)
        for j in range(fn_line + 1, len(lines)):
            s = lines[j]
            if (
                s.strip()
                and (len(s) - len(s.lstrip())) <= fn_indent
                and not s.lstrip().startswith(")")
            ):
                if re.match(r"^\s*(def |@|comptime |var |struct )", s) and j > gl:
                    end = j
                    break
        body = "\n".join(lines[fn_line:end])
        if "raise" not in body:
            suspects.append(f"{path.rsplit('/', 1)[-1]}:{gl + 1} in {fn_name}()")
    return suspects


bad = []
for p in sys.argv[1:]:
    bad += audit(p)
print(f"{len(bad)} silent gated sites:")
for s in bad:
    print(" ", s)
