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
still compares equal.  Both trees are built into one shared directory per
specialization, which is load-bearing: see `build_both_sides`.

Specializations, and why a module is never built bare
-----------------------------------------------------
Every kernel in this repository sits behind a compile-time gate from
`variant_gates.mojo`: `comptime if _op_on["Name"]()` selects the single
operation a build carries, and `_dtype_arg_on[i, dt]()` / `_dtype_out_on[i, dt]`
select its dtypes.  All of those gates read `-D` defines that default to the
empty string, so `mojo build` with no `-D` instantiates *nothing*, exits 0 and
writes no sidecar at all.  Comparing modules built that way reports "0 kernel(s)
differ" for every possible change -- a false all-clear.  This script therefore
enumerates specializations: it scans each entry module for `_op_on["NAME"]` and
builds one variant per operation with `-D OP=NAME`, plus one dtype per gated
axis (`-D DTYPE_ARG_0=...`, `-D DTYPE_OUT=...`) for the modules that gate dtypes.

Coverage is partial by construction and the run says so.  By default each dtype
axis of a module is set uniformly to `float32`, then to `bfloat16` -- two
variants per operation of a dtype-gated module, one for the rest.  Kernels
reachable only under some other dtype (float16, float64, the integer dtypes) or
only under a mixed-dtype signature (an int64 index argument next to a float
value argument) are NOT built unless you ask: `--dtypes float32 bfloat16 int64`.
The summary lists every variant that was built, every variant that emitted no
device kernel, and the dtypes that were left out, so partial coverage can never
be misread as full coverage.

This covers device code only.  Launch geometry -- grid size, block count, the
thresholds that pick between kernels -- is host code, and a change keyed off
`sm_count` or an L2 budget can retarget an architecture while emitting
byte-identical PTX.  Read the dispatch by hand as well.

Usage:
    uv run python scripts/compare_kernel_asm.py \\
        --before /path/to/main_worktree --after . --accelerator sm_90a

    # what a full run would build, without building it
    uv run python scripts/compare_kernel_asm.py --before .. --after . --dry-run

    # one module, one dtype: the fast loop while iterating on a kernel
    uv run python scripts/compare_kernel_asm.py --before .. --after . \\
        --modules matmul_ops --dtypes bfloat16

A full default run is a few hundred `mojo build` invocations per tree; they run
`--jobs` at a time (each peaks near 4.5 GB RSS).  Narrow it with `--modules`,
`--ops` and `--dtypes` while iterating, and run it wide before merging.

`mojo --print-supported-accelerators` lists the valid names (sm_90a, gfx942...).
"""

from __future__ import annotations

import argparse
import difflib
import functools
import os
import re
import shutil
import subprocess
import sys
import textwrap
import threading
import uuid
from concurrent.futures import ThreadPoolExecutor
from dataclasses import dataclass
from pathlib import Path

DEFAULT_KERNEL_DIR = Path("torch_mojo_backend/eager_kernels")
SIDECAR_SUFFIXES = (".ptx", ".amdgcn", ".ll")
# Trailing mangling hash, on the file name and on every symbol inside it.
HASH_RE = re.compile(r"_[0-9a-f]{8}\b")
# The compile-time gates of variant_gates.mojo, as written at the call sites.
OP_GATE_RE = re.compile(r'_op_on\["(\w+)"\]')
ARG_GATE_RE = re.compile(r"_dtype_arg(?:_abi|_width)?_on\[\s*(\d+)")
OUT_GATE_RE = re.compile(r"_dtype_out_on\[\s*(\d+)")
# A gate whose operation name or argument index is not a literal cannot be
# enumerated by reading the source, and silently enumerating the rest would
# understate coverage without saying so.
OPAQUE_GATE_RE = re.compile(
    r"_op_on\[\s*(?!\")|_dtype_arg(?:_abi|_width)?_on\[\s*(?!\d)|_dtype_out_on\[\s*(?!\d)"
)
# One dtype per pass, applied to every gated axis of a module. float32 reaches
# the widest set of kernels; bfloat16 reaches the ones a float32 build gates
# out (the bf16 MFMA/WGMMA paths) and the 16-bit width branches.
DEFAULT_DTYPES = ("float32", "bfloat16")


@dataclass(frozen=True)
class Gates:
    """What one entry module's compile-time gates can select."""

    ops: tuple[str, ...]
    arg_axes: tuple[int, ...]
    out_axes: tuple[int, ...]
    opaque: bool


@dataclass(frozen=True)
class Variant:
    """One (operation, dtype) specialization of a module, built on both sides."""

    op: str | None
    dtype: str | None
    defines: tuple[tuple[str, str], ...]

    @property
    def label(self) -> str:
        return ".".join(part for part in (self.op or "ungated", self.dtype) if part)


@dataclass
class ModulePlan:
    """Everything decided about one module before any build runs."""

    stem: str
    modules: dict[str, Path]
    gates: Gates
    variants: list[Variant]
    notes: list[str]


@functools.cache
def mojo_cli() -> Path:
    """The mojo CLI of the environment providing `max`.

    A copy of ``eager_kernels._find_mojo``, kept local so this script does not
    drag in the device layer (torch, max.driver, the kernel loader) just to
    spawn a compiler.
    """
    candidates = []
    if sys.executable:  # None/'' in embedded interpreters
        candidates.append(Path(sys.executable).parent / "mojo")
    import max as max_package

    for base in list(getattr(max_package, "__path__", ())):
        candidates.extend(parent / "bin" / "mojo" for parent in Path(base).parents)
    found = shutil.which("mojo")
    if found is not None:
        candidates.append(Path(found))
    for candidate in candidates:
        if candidate.is_file():
            return candidate
    raise FileNotFoundError("mojo executable not found")


def build_env() -> dict[str, str]:
    """Environment for `mojo build`, identical to ``eager_kernels._build_env``.

    Once the MAX runtime is loaded it exports MODULAR_*PACKAGE_ROOT/IMPORT_PATH
    overrides meant for embedded payloads.  They shadow the CLI's own stdlib, and
    then `variant_gates.mojo` -- which every entry module imports -- fails to
    parse with "use of unknown declaration 'PyCFunction'".  Stripping them is
    what the on-device builds do, so building them here the same way is also the
    only way this comparison sees the same source the runtime compiles.
    """
    return {
        key: value
        for key, value in os.environ.items()
        if not (
            key.startswith("MODULAR_")
            and ("PACKAGE_ROOT" in key or "IMPORT_PATH" in key)
        )
    }


def emit_asm(
    tree: Path,
    kernel_dir: Path,
    module: Path,
    accelerator: str,
    out_dir: Path,
    defines: tuple[tuple[str, str], ...],
) -> str | None:
    """Build one specialization for ``accelerator``; return an error line, or None.

    Both trees are built with this interpreter's Mojo, so the comparison
    isolates the source change.  Run it under a venv whose Mojo matches the one
    the trees pin, or a toolchain difference shows up as kernel churn.
    """
    out_dir.mkdir(parents=True, exist_ok=True)
    command = [
        str(mojo_cli()),
        "build",
        str(module),
        "-I",
        str(module.parent),
        "-I",
        str(kernel_dir),
        "--emit",
        "asm",
        "--target-accelerator",
        accelerator,
    ]
    for name, value in defines:
        command += ["-D", f"{name}={value}"]
    command += ["-o", str(out_dir / f"{module.stem}.s")]
    result = subprocess.run(
        command, cwd=str(tree), env=build_env(), capture_output=True, text=True
    )
    if result.returncode != 0:
        lines = (result.stderr or result.stdout).strip().splitlines()
        # The driver's own last line is a generic "failed to parse ..."; the
        # first `error:` diagnostic is the one that says what is wrong.
        diagnostics = [line for line in lines if "error:" in line] or lines
        return (
            diagnostics[0].strip()[:160] if diagnostics else f"exit {result.returncode}"
        )
    return None


def find_entry_modules(tree: Path, kernel_dir: Path) -> dict[str, Path]:
    """Map each entry module's stem to its path relative to ``tree``.

    Only the modules that define a `PyInit_` are built on their own; the rest
    are libraries whose kernels are instantiated by an importer of theirs, and
    building one in isolation emits no sidecars at all.
    """
    entries: dict[str, Path] = {}
    for path in sorted((tree / kernel_dir).rglob("*.mojo")):
        if "def PyInit_" not in path.read_text():
            continue
        relative = path.relative_to(tree)
        if path.stem in entries:
            raise ValueError(
                f"multiple entry modules named {path.stem!r}: "
                f"{entries[path.stem]}, {relative}"
            )
        entries[path.stem] = relative
    return entries


def gate_sources(tree: Path, kernel_dir: Path, module: Path) -> list[Path]:
    """The files whose gates decide what one entry module instantiates.

    An operation directory owns its entry point and its private helpers, and a
    gate may sit in either, so the whole directory is scanned.  An entry module
    that lives at the package root (tensor_holder.mojo) is scanned alone: its
    siblings there are the shared libraries, including variant_gates.mojo.
    """
    entry = tree / module
    if entry.parent.resolve() == (tree / kernel_dir).resolve():
        return [entry]
    return sorted(entry.parent.glob("*.mojo"))


def scan_gates(sources: list[Path]) -> Gates:
    """Read the operation names and dtype axes out of the gate call sites."""
    ops = set()
    arg_axes = set()
    out_axes = set()
    opaque = False
    for path in sources:
        text = path.read_text()
        ops.update(OP_GATE_RE.findall(text))
        arg_axes.update(int(index) for index in ARG_GATE_RE.findall(text))
        out_axes.update(int(index) for index in OUT_GATE_RE.findall(text))
        opaque = opaque or OPAQUE_GATE_RE.search(text) is not None
    return Gates(
        tuple(sorted(ops)), tuple(sorted(arg_axes)), tuple(sorted(out_axes)), opaque
    )


def variant_defines(
    gates: Gates, op: str | None, dtype: str | None, extra: tuple[tuple[str, str], ...]
) -> tuple[tuple[str, str], ...]:
    """The `-D` set that turns one operation of one module on."""
    defines: dict[str, str] = {}
    if op is not None:
        defines["OP"] = op
    if dtype is not None:
        for index in gates.arg_axes:
            defines[f"DTYPE_ARG_{index}"] = dtype
        for index in gates.out_axes:
            # `_dtype_out_on[0, ...]` accepts DTYPE_OUT or DTYPE_OUT_0, and the
            # runtime spells a single output DTYPE_OUT (see exact_call_defines).
            defines["DTYPE_OUT" if index == 0 else f"DTYPE_OUT_{index}"] = dtype
    defines.update(extra)
    return tuple(sorted(defines.items()))


def plan_module(
    stem: str,
    modules: dict[str, Path],
    gates: dict[str, Gates],
    dtypes: tuple[str, ...],
    only_ops: tuple[str, ...] | None,
    extra_defines: tuple[tuple[str, str], ...],
) -> ModulePlan:
    """Enumerate the specializations to build for one module, and say what is odd."""
    notes = []
    merged = Gates(
        tuple(sorted({op for side in gates.values() for op in side.ops})),
        tuple(sorted({axis for side in gates.values() for axis in side.arg_axes})),
        tuple(sorted({axis for side in gates.values() for axis in side.out_axes})),
        any(side.opaque for side in gates.values()),
    )
    if merged.opaque:
        notes.append(
            "a gate here is called with a non-literal operation name or "
            "argument index, so reading the source cannot enumerate what this "
            "module can instantiate: the variants below may miss kernels."
        )
    for side, other in (("before", "after"), ("after", "before")):
        if side in gates and other in gates:
            only_here = sorted(set(gates[side].ops) - set(gates[other].ops))
            if only_here:
                notes.append(
                    f"operation(s) present only in {side}: {', '.join(only_here)} "
                    f"(every kernel of theirs counts as new or gone)"
                )

    ops: list[str | None] = list(merged.ops)
    if not ops:
        # Not necessarily a bug -- tensor_holder.mojo is deliberately ungated --
        # but a module that lost its gates would otherwise compare as empty and
        # silently read as "nothing differs", which is the failure mode this
        # whole script exists to avoid. Build it bare and say so out loud.
        notes.append(
            "no _op_on[...] gate found, so nothing here can be selected by name."
            + (
                " Built ONCE with no OP define, which is correct only for a"
                " deliberately ungated module; otherwise the gate scan is broken"
                " and this module's kernels are NOT covered."
                if only_ops is None
                else " --ops was given, so this module was skipped entirely."
            )
        )
        ops = [None] if only_ops is None else []
    if only_ops is not None:
        kept = [op for op in ops if op in only_ops]
        if len(kept) != len(ops):
            notes.append(f"--ops skipped {len(ops) - len(kept)} of {len(ops)} op(s)")
        ops = kept

    axes = bool(merged.arg_axes or merged.out_axes)
    variants = [
        Variant(
            op,
            dtype if axes else None,
            variant_defines(merged, op, dtype if axes else None, extra_defines),
        )
        for op in ops
        for dtype in (dtypes if axes else dtypes[:1])
    ]
    return ModulePlan(stem, modules, merged, variants, notes)


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


def default_jobs() -> int:
    """Concurrent `mojo build` subprocesses.

    A build peaks around 4.5 GB RSS and uses ~2.5-3 cores -- the figures
    `eager_kernels._pool_size` is tuned to -- so cap by available RAM (5 GiB per
    slot with headroom) and by cores; never fewer than 1, never more than 16.
    """
    mem_gib = 8.0
    try:
        with open("/proc/meminfo") as handle:
            for line in handle:
                if line.startswith("MemAvailable"):
                    mem_gib = int(line.split()[1]) / (1024 * 1024)
                    break
    except OSError:
        pass
    return max(1, min(int(mem_gib // 5), (os.cpu_count() or 4) // 3, 16))


def parse_define(text: str) -> tuple[str, str]:
    name, separator, value = text.partition("=")
    if not separator or not re.fullmatch(r"[A-Z][A-Z0-9_]*", name):
        raise argparse.ArgumentTypeError(
            f"--define takes UPPER_CASE_NAME=value, got {text!r}"
        )
    return name, value


def parse_dtype(text: str) -> str:
    if not re.fullmatch(r"[a-z][a-z0-9_]*", text):
        raise argparse.ArgumentTypeError(f"not a MAX dtype name: {text!r}")
    return text


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
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
    parser.add_argument(
        "--ops",
        nargs="*",
        default=None,
        type=str,
        help="only build these OP gates; default is every gate found in the source",
    )
    parser.add_argument(
        "--dtypes",
        nargs="+",
        default=list(DEFAULT_DTYPES),
        type=parse_dtype,
        help=(
            "one build per dtype, applied to every gated dtype axis of a module "
            f"(default: {' '.join(DEFAULT_DTYPES)})"
        ),
    )
    parser.add_argument(
        "--define",
        action="append",
        default=[],
        type=parse_define,
        metavar="NAME=VALUE",
        help="extra -D applied to every variant (for flag-selected variants)",
    )
    parser.add_argument(
        "--jobs",
        type=int,
        default=default_jobs(),
        help=(
            "specializations to build concurrently, one `mojo build` each "
            f"(default here: {default_jobs()}, from RAM and cores)"
        ),
    )
    parser.add_argument(
        "--work-dir",
        type=Path,
        default=Path("/tmp/kernel_asm_diff"),
        help=(
            "build directory; keep it stable across runs so a Mojo cache hit "
            "replays its sidecars to a path that still exists"
        ),
    )
    parser.add_argument("--show-diff", action="store_true")
    parser.add_argument(
        "--keep",
        action="store_true",
        help="copy the emitted assembly to <work-dir>/keep/<side>/<module>/<variant>",
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="print the specializations, build nothing",
    )
    parser.add_argument("--quiet", action="store_true", help="no per-build progress")
    return parser.parse_args()


def wrap(prefix: str, items: list[str]) -> str:
    return textwrap.fill(
        prefix + ", ".join(items),
        width=100,
        initial_indent="    ",
        subsequent_indent="        ",
    )


def build_plans(args: argparse.Namespace) -> list[ModulePlan]:
    """Discover every module on both sides and enumerate its specializations."""
    trees = {"before": args.before, "after": args.after}
    entries = {
        side: find_entry_modules(tree, args.kernel_dir) for side, tree in trees.items()
    }
    stems = args.modules
    if stems is None:
        stems = sorted(set(entries["before"]) | set(entries["after"]))
    plans = []
    for stem in stems:
        # A module present on one side only is an added or deleted file, not a
        # failure: the PR that introduces a kernel module is the common case,
        # and every kernel in it is new for this architecture.
        modules = {side: entries[side][stem] for side in trees if stem in entries[side]}
        if not modules:
            raise ValueError(f"no entry module named {stem!r} in either tree")
        gates = {
            side: scan_gates(gate_sources(trees[side], args.kernel_dir, module))
            for side, module in modules.items()
        }
        plans.append(
            plan_module(
                stem,
                modules,
                gates,
                tuple(args.dtypes),
                None if args.ops is None else tuple(args.ops),
                tuple(args.define),
            )
        )
    return plans


def build_both_sides(
    args: argparse.Namespace, trees: dict[str, Path], plan: ModulePlan, variant: Variant
) -> tuple[dict[str, dict[str, str]], dict[str, str]]:
    """Build one specialization on every side it exists, and read its kernels.

    Both sides are built one after the other **into the same directory**, which
    looks redundant and is not.  Mojo's transform cache (`$MODULAR_HOME`,
    `.mojo_cache/mojo/transform`) keys on the source and the defines but not on
    the output path, and it services a hit by rewriting the sidecars to the path
    recorded when the entry was created.  Two trees whose module is byte
    identical -- the normal case for most modules of any PR -- are one such hit,
    so building them into separate directories leaves the second one empty and
    reports every kernel of an unchanged module as deleted.  Sharing the
    directory makes the recorded path and the wanted path the same path, hit or
    miss; the sidecars are read into memory and the directory is emptied between
    sides so nothing of the first side can be mistaken for the second's.
    """
    kernels: dict[str, dict[str, str]] = {}
    errors: dict[str, str] = {}
    out_dir = args.work_dir / plan.stem / variant.label
    for side in plan.modules:
        shutil.rmtree(out_dir, ignore_errors=True)
        out_dir.mkdir(parents=True, exist_ok=True)
        error = emit_asm(
            trees[side],
            args.kernel_dir,
            plan.modules[side],
            args.accelerator,
            out_dir,
            variant.defines,
        )
        if error is not None and "offload output file" in error:
            # A hit recorded by an older run, whose --work-dir is gone. Nothing
            # reads this define, so it cannot move the generated code; it only
            # makes the cache miss and compile for real, into the path we want.
            error = emit_asm(
                trees[side],
                args.kernel_dir,
                plan.modules[side],
                args.accelerator,
                out_dir,
                variant.defines + (("ASM_CACHE_BUST", uuid.uuid4().hex),),
            )
        if error is not None:
            errors[side] = error
            continue
        kernels[side] = collect(out_dir)
        if args.keep:
            shutil.copytree(
                out_dir, args.work_dir / "keep" / side / plan.stem / variant.label
            )
    shutil.rmtree(out_dir, ignore_errors=True)
    return kernels, errors


def run_builds(
    args: argparse.Namespace, plans: list[ModulePlan]
) -> tuple[dict[tuple[str, str, str], dict[str, str]], dict[tuple[str, str, str], str]]:
    """Build every (module, variant) in parallel; return kernels and errors."""
    trees = {"before": args.before, "after": args.after}
    jobs = [(plan, variant) for plan in plans for variant in plan.variants]
    kernels: dict[tuple[str, str, str], dict[str, str]] = {}
    errors: dict[tuple[str, str, str], str] = {}
    lock = threading.Lock()
    done = 0

    def build(job: tuple[ModulePlan, Variant]) -> None:
        nonlocal done
        plan, variant = job
        built, failures = build_both_sides(args, trees, plan, variant)
        with lock:
            done += 1
            for side, found in built.items():
                kernels[(side, plan.stem, variant.label)] = found
            for side, error in failures.items():
                errors[(side, plan.stem, variant.label)] = error
            if not args.quiet:
                counts = [f"{side} {len(found)}" for side, found in built.items()]
                counts += [f"{side} FAILED" for side in failures]
                print(
                    f"[{done:>4}/{len(jobs)}] {plan.stem}/{variant.label}"
                    f" -- kernels: {', '.join(counts)}",
                    file=sys.stderr,
                    flush=True,
                )

    print(
        f"{sum(len(plan.modules) for plan, _ in jobs)} build(s) = {len(jobs)} "
        f"specialization(s) over {len(plans)} module(s), "
        f"{args.jobs} specialization(s) at a time (both sides of one run "
        "sequentially, in one directory -- see build_both_sides).",
        file=sys.stderr,
        flush=True,
    )
    with ThreadPoolExecutor(max_workers=max(1, args.jobs)) as pool:
        list(pool.map(build, jobs))
    return kernels, errors


def main() -> int:
    args = parse_args()
    plans = build_plans(args)

    coverage = []
    for plan in plans:
        axes = sorted(
            [f"DTYPE_ARG_{index}" for index in plan.gates.arg_axes]
            + [
                "DTYPE_OUT" if index == 0 else f"DTYPE_OUT_{index}"
                for index in plan.gates.out_axes
            ]
        )
        coverage.append(
            f"{plan.stem}: {len(plan.variants)} variant(s)"
            f"; dtype axes: {', '.join(axes) if axes else 'none (dtype-independent)'}"
        )
        coverage.append(wrap("ops built: ", [v.label for v in plan.variants]))
        coverage.extend(wrap(f"NOTE: {note}", []) for note in plan.notes)

    if args.dry_run:
        print("\n".join(coverage))
        print(
            f"\n{sum(len(plan.variants) for plan in plans)} specialization(s) over "
            f"{len(plans)} module(s); dtypes {', '.join(args.dtypes)}. Nothing built."
        )
        return 0

    if not any(plan.variants for plan in plans):
        print("\n".join(coverage))
        print(
            "\nNo specialization was selected, so nothing was compared. Check "
            "--modules and --ops against the names --dry-run lists."
        )
        return 2

    shutil.rmtree(args.work_dir, ignore_errors=True)
    kernels, errors = run_builds(args, plans)

    rows = []
    total_changed = 0
    total_kernels = 0
    failed = []
    empty = []
    for plan in plans:
        changed_names = []
        counts = {"kernels": 0, "changed": 0, "added": 0, "removed": 0}
        broken = []
        for variant in plan.variants:
            sides = {}
            for side in plan.modules:
                key = (side, plan.stem, variant.label)
                if key in errors:
                    broken.append(f"{variant.label} [{side}: {errors[key]}]")
                else:
                    sides[side] = kernels[key]
            if len(sides) != len(plan.modules):
                continue
            before = sides.get("before", {})
            after = sides.get("after", {})
            counts["kernels"] += len(after)
            total_kernels += len(before) + len(after)
            if not before and not after:
                empty.append(f"{plan.stem}/{variant.label}")
                continue
            if "before" not in sides or "after" not in sides:
                # Added or removed module: every kernel of it is new or gone.
                counts["added"] += len(after)
                counts["removed"] += len(before)
                continue
            changed = sorted(
                name
                for name in before.keys() & after.keys()
                if before[name] != after[name]
            )
            counts["changed"] += len(changed)
            counts["added"] += len(after.keys() - before.keys())
            counts["removed"] += len(before.keys() - after.keys())
            changed_names += [f"{variant.label}/{name}" for name in changed]
            for name in changed if args.show_diff else []:
                print_diff(
                    f"{plan.stem}/{variant.label}/{name}", before[name], after[name]
                )

        total_changed += counts["changed"] + counts["added"] + counts["removed"]
        if broken:
            failed.append((plan.stem, broken))
        detail = ", ".join(changed_names[:3]) + (
            " ..." if len(changed_names) > 3 else ""
        )
        if not detail and broken:
            detail = f"{len(broken)} variant(s) FAILED to build"
        if not detail and not plan.variants:
            detail = "no variant built -- see its NOTE under COVERAGE"
        if not detail and "before" not in plan.modules:
            detail = f"NEW module, every kernel is new for {args.accelerator}"
        if not detail and "after" not in plan.modules:
            detail = f"REMOVED module, every kernel is gone for {args.accelerator}"
        rows.append(
            (
                plan.stem,
                str(len(plan.variants)),
                str(counts["kernels"]),
                str(counts["changed"]),
                f"+{counts['added']}/-{counts['removed']}",
                detail,
            )
        )

    width = max((len(row[0]) for row in rows), default=6)
    print(f"\n{args.accelerator}: {args.before} -> {args.after}\n")
    print(f"{'module'.ljust(width)}  variants  kernels  changed  new/gone  detail")
    for stem, variants, count, changed, delta, detail in rows:
        print(
            f"{stem.ljust(width)}  {variants:>8}  {count:>7}  {changed:>7}"
            f"  {delta:>8}  {detail}"
        )

    print("\nCOVERAGE (what was actually built)\n")
    print("\n".join(coverage))
    print("\nNOT COVERED\n")
    print(wrap("every gated dtype axis was set, uniformly, to: ", list(args.dtypes)))
    print(
        "    Kernels gated on any other dtype, or on a mixed-dtype signature\n"
        "    (an int64 index next to a float value), were NOT built, and are\n"
        "    neither verified changed nor verified unchanged. Pass --dtypes to\n"
        "    cover them."
    )
    if empty:
        print(
            f"\n{len(empty)} specialization(s) emitted NO device kernel for "
            f"{args.accelerator} -- unverified, not verified-unchanged:"
        )
        print(wrap("", sorted(empty)))

    if not args.keep:
        shutil.rmtree(args.work_dir, ignore_errors=True)

    if failed:
        # A module that never compiled contributes no kernels, and reporting
        # that as "nothing differs" would be a false clean bill of health.
        print(f"\nFAILED to build {sum(len(b) for _, b in failed)} specialization(s):")
        for stem, broken in failed:
            print(wrap(f"{stem}: ", broken))
        print(
            "The comparison is incomplete. The usual cause is a Mojo mismatch: "
            "run this under a venv whose Mojo matches the one the trees pin."
        )
        return 2
    if total_kernels == 0:
        # Exactly the failure this enumeration exists to prevent: gates that
        # match nothing compile cleanly and emit no sidecar, so an empty run
        # would otherwise print a confident "0 kernel(s) differ".
        print(
            f"\nNO device kernel was emitted for {args.accelerator} by ANY of the "
            f"{sum(len(plan.variants) for plan in plans)} specialization(s). This is "
            "NOT a clean bill of health: either the gate scan no longer matches "
            "variant_gates.mojo, or nothing selected here has device code. Check "
            "with --dry-run and --keep before trusting any result."
        )
        return 2
    print(
        f"\n{total_changed} kernel(s) differ for {args.accelerator} across "
        f"{sum(len(plan.variants) for plan in plans)} specialization(s) of "
        f"{len(plans)} module(s) ({total_kernels} sidecar(s) emitted in total)."
    )
    return 1 if total_changed else 0


if __name__ == "__main__":
    raise SystemExit(main())
