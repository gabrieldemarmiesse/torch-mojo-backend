"""Report Mojo dispatch gates that can never be reached.

The host dispatchers in ``eager_kernels`` are long ``if`` ladders: each branch
tests a shape/alignment predicate, launches a kernel and returns, so a later
branch only runs when every earlier one declined. A branch whose predicate is
identical to an earlier sibling's is therefore dead, along with every kernel it
launches -- and nothing else catches it, because the kernel, its enqueue helper
and the branch are all still *referenced*. Mojo has no unreachable-code or
dead-function diagnostic (checked against ``mojo build --help-hidden``), and a
launch profile only tells you a kernel did not run, not that it could not.

This is what made ``_v3_tn_wgmma_tma_transpose_s2`` survive review: its gate was
term-for-term the TN small-tile gate ~90 lines above it, spelled with a
different set of comptime aliases that happened to hold the same values
(``_V3_TN_SMALL_BM/BN/BK`` vs ``_V3_BM/_V3_BN/_V3_BK``, both 64/128/64).

So the check resolves comptime integer aliases to their values before comparing
predicates. Constants are resolved per file: several names (``_BK``,
``_THREADS``, ``_BLOCK``, ``_VEC``) are defined with different values in
different modules, and a name defined twice within one file is left unresolved
rather than guessed.

Default mode reports only IDENTICAL predicates, which is what a dead branch
looks like. ``--report-subsumes`` additionally reports a later gate strictly
stronger than an earlier one; that is a hint, not a defect, because an earlier
branch whose body returns conditionally does let control fall through.
"""

from __future__ import annotations

import argparse
import re
import sys
from pathlib import Path

CONST_RE = re.compile(r"^\s*comptime\s+([A-Za-z_]\w*)\s*=\s*([0-9_]+)\s*$")
DEF_RE = re.compile(r"^(\s*)(?:def|fn)\s+([A-Za-z_]\w*)\s*[\(\[]")
IF_RE = re.compile(r"^(\s*)(if|elif)\s+(.*)$")
LADDER_CONTINUATION_RE = re.compile(r"(if\b|elif\b|else\b|\))")

# A gate with fewer terms than this is an ordinary conditional, not a dispatch
# admission predicate; comparing those is all noise.
MIN_GATE_TERMS = 3


class Gate:
    """One ``if``/``elif`` predicate, flattened to its conjunction of terms."""

    def __init__(
        self, function: str, line: int, indent: int, terms: frozenset[str]
    ) -> None:
        self.function = function
        self.line = line
        self.indent = indent
        self.terms = terms


def file_constants(text: str) -> dict[str, int]:
    """Integer ``comptime`` aliases declared in one file.

    A name declared more than once with differing values is dropped: resolving
    it would silently pick one scope's value (``BLOCK_M`` is both 32 and 64 in
    ``matmul_ops.mojo``).
    """
    seen: dict[str, int] = {}
    conflicting: set[str] = set()
    for line in text.splitlines():
        match = CONST_RE.match(line)
        if match is None:
            continue
        name, value = match.group(1), int(match.group(2).replace("_", ""))
        if name in seen and seen[name] != value:
            conflicting.add(name)
        seen[name] = value
    return {name: value for name, value in seen.items() if name not in conflicting}


def split_conjunction(condition: str) -> list[str]:
    """Split a predicate on its top-level ``and``, ignoring nested parens."""
    condition = condition.strip()
    while condition.startswith("(") and condition.endswith(")"):
        depth = 0
        wraps_whole = True
        for index, char in enumerate(condition):
            depth += (char == "(") - (char == ")")
            if depth == 0 and index < len(condition) - 1:
                wraps_whole = False
                break
        if not wraps_whole:
            break
        condition = condition[1:-1].strip()

    parts: list[str] = []
    current = ""
    depth = 0
    for token in re.split(r"(\band\b|\(|\))", condition):
        if token == "(":
            depth += 1
            current += token
        elif token == ")":
            depth -= 1
            current += token
        elif token == "and" and depth <= 0:
            parts.append(current)
            current = ""
        else:
            current += token
    parts.append(current)
    return [
        re.sub(r"\s+", " ", part).strip().strip("()").strip()
        for part in parts
        if part.strip()
    ]


def resolve_constants(term: str, constants: dict[str, int]) -> str:
    """Rewrite comptime alias names to their integer values."""
    return re.sub(
        r"\b[A-Za-z_]\w*\b",
        lambda m: str(constants[m.group(0)]) if m.group(0) in constants else m.group(0),
        term,
    )


def parse_gates(lines: list[str], constants: dict[str, int]) -> list[Gate]:
    """Collect every multi-term ``if``/``elif`` predicate in a file."""
    gates: list[Gate] = []
    function = "?"
    index = 0
    while index < len(lines):
        definition = DEF_RE.match(lines[index])
        if definition is not None:
            function = definition.group(2)

        opening = IF_RE.match(lines[index])
        if opening is None:
            index += 1
            continue

        condition = opening.group(3)
        depth = 0
        end = index
        while True:
            depth += lines[end].count("(") - lines[end].count(")")
            if depth <= 0 and lines[end].rstrip().endswith(":"):
                break
            end += 1
            if end >= len(lines):
                break
            condition += " " + lines[end].strip()

        terms = frozenset(
            resolve_constants(term, constants)
            for term in split_conjunction(condition.rstrip().rstrip(":"))
        )
        if len(terms) >= MIN_GATE_TERMS:
            gates.append(Gate(function, index + 1, len(opening.group(1)), terms))
        index = end + 1
    return gates


def same_ladder(lines: list[str], earlier: Gate, later: Gate) -> bool:
    """Whether no statement between the two gates leaves their ``if`` ladder.

    Any statement dedenting past the gates' own indent, or sitting at that
    indent without being part of the ladder, means the later gate is reached by
    a different path and the two are not alternatives.
    """
    for line in lines[earlier.line : later.line - 1]:
        stripped = line.lstrip()
        if not stripped or stripped.startswith("#"):
            continue
        indent = len(line) - len(stripped)
        if indent < earlier.indent:
            return False
        if indent == earlier.indent and not LADDER_CONTINUATION_RE.match(stripped):
            return False
    return True


def check_file(path: Path, report_subsumes: bool) -> list[str]:
    """Report gate pairs in one file where the later gate cannot be reached."""
    text = path.read_text()
    lines = text.splitlines()
    gates = parse_gates(lines, file_constants(text))

    by_ladder: dict[tuple[str, int], list[Gate]] = {}
    for gate in gates:
        by_ladder.setdefault((gate.function, gate.indent), []).append(gate)

    problems: list[str] = []
    for (function, _), siblings in by_ladder.items():
        for position, earlier in enumerate(siblings):
            for later in siblings[position + 1 :]:
                if not earlier.terms <= later.terms:
                    continue
                identical = earlier.terms == later.terms
                if not identical and not report_subsumes:
                    continue
                if not same_ladder(lines, earlier, later):
                    continue
                kind = "unreachable" if identical else "possibly unreachable"
                extra = sorted(later.terms - earlier.terms)
                problems.append(
                    f"{path}:{later.line}: {kind} gate in {function}() -- the gate at "
                    f"line {earlier.line} admits everything this one does"
                    + (f"; extra terms here: {extra}" if extra else "")
                )
    return problems


def iter_sources(targets: list[Path]) -> list[Path]:
    """Expand directories to the ``.mojo`` files under them."""
    sources: list[Path] = []
    for target in targets:
        if target.is_dir():
            sources.extend(sorted(target.rglob("*.mojo")))
        elif target.suffix == ".mojo":
            sources.append(target)
    return sources


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "targets",
        nargs="*",
        type=Path,
        default=[Path("torch_mojo_backend")],
        help="`.mojo` files or directories to scan (pre-commit passes files)",
    )
    parser.add_argument(
        "--report-subsumes",
        action="store_true",
        help="also report later gates that are strictly stronger (warn-only signal)",
    )
    args = parser.parse_args()

    problems: list[str] = []
    for source in iter_sources(args.targets):
        problems.extend(check_file(source, args.report_subsumes))

    for problem in problems:
        print(problem)
    if not problems:
        return 0
    if args.report_subsumes:
        # Strictly-stronger gates are only a hint: an earlier branch whose body
        # returns conditionally does let control reach them. Never fail on these.
        print(
            f"\n{len(problems)} gate pair(s) to eyeball; check whether the earlier branch"
        )
        print("returns unconditionally before treating any of them as dead.")
        return 0
    print(
        f"\n{len(problems)} unreachable dispatch gate(s). A gate whose predicate an "
        "earlier branch already admits can never run, so the kernels it launches are "
        "dead. Delete the branch, or make its predicate actually narrower."
    )
    return 1


if __name__ == "__main__":
    sys.exit(main())
