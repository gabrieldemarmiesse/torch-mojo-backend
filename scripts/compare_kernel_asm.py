"""Diff the GPU assembly two source trees emit, for an accelerator you lack.

A kernel is usually tuned on whatever GPU the machine has, and the risk is a
change that quietly rewrites the code emitted for a GPU that cannot be measured
here.  `mojo build --emit asm --target-accelerator <arch>` cross-compiles, so
that question is answerable without the device: build both trees for the absent
architecture and compare.  Identical assembly means the change cannot have moved
that architecture's kernels; different assembly means it needs either a
measurement there or an arch gate.

`--emit asm` writes host assembly to the `-o` path and one sidecar per GPU
kernel beside it: `.ptx` for NVIDIA, `.amdgcn` for AMD, `.ll` for Metal.  Each
is named `<module>_<kernel>_<hash>.ptx`, and that hash also mangles every symbol
inside.  Kernels are paired by name with the hash stripped and the hash is
masked in the body, so a kernel whose mangling moved but whose code did not
still compares equal.

This covers device code only.  Launch geometry -- grid size, block count, the
thresholds that pick between kernels -- is host code, and a change keyed off
`sm_count` or an L2 budget can retarget an architecture while emitting
byte-identical PTX.  Read the dispatch by hand as well.

Usage:
    uv run python scripts/compare_kernel_asm.py \\
        --before /path/to/main_worktree --after . --accelerator sm_90a

`mojo --print-supported-accelerators` lists the valid names (sm_90a, gfx942...).
"""

from __future__ import annotations

import argparse
import difflib
import re
import shutil
from pathlib import Path

from mojo.run import subprocess_run_mojo

DEFAULT_KERNEL_DIR = Path("torch_mojo_backend/eager_kernels")
SIDECAR_SUFFIXES = (".ptx", ".amdgcn", ".ll")
# Trailing mangling hash, on the file name and on every symbol inside it.
HASH_RE = re.compile(r"_[0-9a-f]{8}\b")


def emit_asm(tree: Path, module: Path, accelerator: str, out_dir: Path) -> str | None:
    """Build one module for ``accelerator``; return an error line, or None.

    Both trees are built with this interpreter's Mojo, so the comparison
    isolates the source change.  Run it under a venv whose Mojo matches the one
    the trees pin, or a toolchain difference shows up as kernel churn.  Going
    through ``subprocess_run_mojo`` is required rather than tidy: invoking the
    ``mojo`` binary directly leaves the prelude unconfigured and every builtin
    parses as an unknown declaration.
    """
    out_dir.mkdir(parents=True, exist_ok=True)
    result = subprocess_run_mojo(
        [
            "build",
            str(module),
            "--emit",
            "asm",
            "--target-accelerator",
            accelerator,
            "-o",
            str(out_dir / f"{module.stem}.s"),
        ],
        cwd=str(tree),
        capture_output=True,
        text=True,
    )
    if result.returncode != 0:
        message = (result.stderr or result.stdout).strip().splitlines()
        return message[-1][:160] if message else f"exit {result.returncode}"
    return None


def collect(out_dir: Path) -> dict[str, str]:
    """Map each kernel's hash-stripped name to its hash-masked assembly."""
    kernels = {}
    for path in sorted(out_dir.iterdir()):
        if path.suffix in SIDECAR_SUFFIXES:
            kernels[HASH_RE.sub("", path.stem)] = HASH_RE.sub("_HASH", path.read_text())
    return kernels


def print_diff(kernel: str, before: str, after: str) -> None:
    """Print a unified diff of one kernel's assembly."""
    print(
        "\n".join(
            difflib.unified_diff(
                before.splitlines(),
                after.splitlines(),
                fromfile=f"before/{kernel}",
                tofile=f"after/{kernel}",
                lineterm="",
                n=2,
            )
        )
    )


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--before", type=Path, required=True, help="baseline tree")
    parser.add_argument("--after", type=Path, required=True, help="tree under review")
    parser.add_argument("--accelerator", default="sm_90a")
    parser.add_argument("--kernel-dir", type=Path, default=DEFAULT_KERNEL_DIR)
    parser.add_argument(
        "--modules",
        nargs="*",
        default=None,
        help="module stems to check; default is every PyInit_ module in --kernel-dir",
    )
    parser.add_argument("--work-dir", type=Path, default=Path("/tmp/kernel_asm_diff"))
    parser.add_argument("--show-diff", action="store_true")
    parser.add_argument("--keep", action="store_true", help="keep emitted assembly")
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    shutil.rmtree(args.work_dir, ignore_errors=True)

    stems = args.modules
    if stems is None:
        # Only the modules that define a `PyInit_` are built on their own; the
        # rest are libraries whose kernels are instantiated by an importer of
        # theirs, and building one in isolation emits no sidecars at all.
        stems = sorted(
            path.stem
            for path in (args.after / args.kernel_dir).glob("*.mojo")
            if "def PyInit_" in path.read_text()
        )

    rows = []
    total_changed = 0
    failed = []
    for stem in stems:
        module = args.kernel_dir / f"{stem}.mojo"
        sides = {}
        errors = {}
        # A module present on one side only is an added or deleted file, not a
        # failure: the PR that introduces a kernel module is the common case,
        # and every kernel in it is new for this architecture.
        present = {
            side: tree
            for side, tree in (("before", args.before), ("after", args.after))
            if (tree / module).is_file()
        }
        for side, tree in present.items():
            error = emit_asm(
                tree, module, args.accelerator, args.work_dir / side / stem
            )
            if error:
                errors[side] = error
            else:
                sides[side] = collect(args.work_dir / side / stem)
        if errors:
            detail = "; ".join(f"{side}: {text}" for side, text in errors.items())
            rows.append((stem, "-", "-", "-", detail))
            failed.append(stem)
            continue

        before = sides.get("before", {})
        after = sides.get("after", {})
        if "before" not in present or "after" not in present:
            note = "NEW module" if "before" not in present else "REMOVED module"
            total_changed += len(after) + len(before)
            rows.append(
                (
                    stem,
                    str(len(after)),
                    "-",
                    f"+{len(after)}/-{len(before)}",
                    f"{note}, every kernel is new for {args.accelerator}",
                )
            )
            continue

        changed = sorted(
            k for k in before.keys() & after.keys() if before[k] != after[k]
        )
        removed = sorted(before.keys() - after.keys())
        added = sorted(after.keys() - before.keys())
        total_changed += len(changed) + len(removed) + len(added)
        for kernel in changed if args.show_diff else []:
            print_diff(kernel, before[kernel], after[kernel])
        detail = ", ".join(changed[:3]) + (" ..." if len(changed) > 3 else "")
        rows.append(
            (
                stem,
                str(len(after)),
                str(len(changed)),
                f"+{len(added)}/-{len(removed)}",
                detail,
            )
        )

    width = max((len(row[0]) for row in rows), default=6)
    print(f"\n{args.accelerator}: {args.before} -> {args.after}\n")
    print(f"{'module'.ljust(width)}  kernels  changed  new/gone  detail")
    for stem, count, changed, delta, detail in rows:
        print(f"{stem.ljust(width)}  {count:>7}  {changed:>7}  {delta:>8}  {detail}")

    if not args.keep:
        shutil.rmtree(args.work_dir, ignore_errors=True)

    if failed:
        # A module that never compiled contributes no kernels, and reporting
        # that as "nothing differs" would be a false clean bill of health.
        print(
            f"\nFAILED to build {len(failed)} module(s): {', '.join(failed)}.\n"
            "The comparison is incomplete. The usual cause is a Mojo mismatch: "
            "run this under a venv whose Mojo matches the one the trees pin."
        )
        return 2
    print(f"\n{total_changed} kernel(s) differ for {args.accelerator}.")
    return 1 if total_changed else 0


if __name__ == "__main__":
    raise SystemExit(main())
