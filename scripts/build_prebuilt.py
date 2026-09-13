"""Build the native libraries that ship prebuilt in the wheel.

`register_mojo_devices()` needs two fixed libraries before any op runs: the
C++ shim (`native/csrc/`, one g++ invocation against the installed torch
headers) and the Mojo base library (`native/mojo/`, one `mojo build`). On a
fresh install they cost ~5 s and ~13 s of compiling, and the shim needs a C++
compiler on the box. Neither depends on the accelerator and neither contains
device code, so both can be built here and shipped:

* the shim links only libtorch_cpu, libc10 and the C/C++ runtimes -- no
  Python -- so one file per (torch major.minor, platform, machine, libstdc++
  ABI flag) serves every Python version and both the CPU and the CUDA wheel
  of that torch series;
* the base library depends on MAX, which is pinned exactly in pyproject, so
  it is one file per platform per release.

What this script does, per torch version: make a throwaway venv holding that
version's CPU wheel and nothing else, and run the shim build *in that venv*,
so its autocast policy table and its C++ standard come from those exact
headers. The base library is built once, with the MAX of the environment the
script itself runs in -- or, with `--backend-python`, in a throwaway venv of
its own. Every artefact lands in
`torch_mojo_backend/native/prebuilt/` with an entry in `manifest.json`
recording what it was built from; `torch_mojo_backend.native` uses a file
only when that entry matches the running environment and the hash of the
sources shipped beside it.

    # the usual local run
    uv run python scripts/build_prebuilt.py --torch 2.11 2.14

    # a portable Linux shim: an old-glibc container, no MAX in it
    docker run --rm -v "$PWD:/src" -w /src quay.io/pypa/manylinux_2_28_x86_64 \
        bash -c 'pip install uv && python scripts/build_prebuilt.py \
                 --torch 2.11 --no-backend --out /src/prebuilt-x86_64'

    # the base library, in a venv of its own (no project sync, no CUDA torch)
    python3 scripts/build_prebuilt.py --backend-python 3.12 --out prebuilt-out

    # CI: each platform job builds its own, the release job merges them
    python3 scripts/build_prebuilt.py --merge artifacts/

The shim venvs hold torch alone -- no MAX, no install of this package -- and
the build reaches `torch_mojo_backend.native` through `_native_module()`
below. That is what lets the same command run in a manylinux_2_28 container,
where MAX cannot be installed (its wheels are manylinux_2_34) and where a
build that must load on old systems belongs.

`--emit-one` is how the script re-enters itself inside a venv it just made;
it is not meant to be typed by hand.
"""

from __future__ import annotations

import argparse
import datetime
import importlib.util
import json
import os
import re
import shutil
import subprocess
import sys
import tempfile
import tomllib
from pathlib import Path
from types import ModuleType

_ROOT = Path(__file__).resolve().parents[1]
_DEFAULT_OUT = _ROOT / "torch_mojo_backend" / "native" / "prebuilt"
_CPU_INDEX = "https://download.pytorch.org/whl/cpu"
_PYPI = "https://pypi.org/simple"


def _log(msg: str):
    print(f"[build_prebuilt] {msg}", flush=True)


def _run(cmd: list[str], *, env: dict[str, str] | None = None) -> str:
    _log("$ " + " ".join(cmd))
    proc = subprocess.run(cmd, text=True, capture_output=True, env=env)
    if proc.returncode != 0:
        raise RuntimeError(
            f"command failed ({proc.returncode}): {' '.join(cmd)}\n"
            + proc.stdout
            + proc.stderr
        )
    if proc.stderr.strip():
        print(proc.stderr, file=sys.stderr, flush=True)
    return proc.stdout


def _native_module() -> ModuleType:
    """`torch_mojo_backend.native`, with as little of the package as possible.

    Building the shim needs torch headers and a C++ compiler and nothing
    else -- but the package's `__init__` imports MAX, which is absent from
    the torch-only venvs this script makes and uninstallable in the
    manylinux_2_28 container (MAX ships manylinux_2_34 wheels). So load the
    module from its file when the ordinary import is not available. The
    module itself imports only torch at import time.
    """
    try:
        from torch_mojo_backend import native  # noqa: PLC0415 -- optional path

        return native
    except ImportError:
        path = _ROOT / "torch_mojo_backend" / "native" / "__init__.py"
        spec = importlib.util.spec_from_file_location("torch_mojo_backend.native", path)
        if spec is None or spec.loader is None:
            raise
        module = importlib.util.module_from_spec(spec)
        sys.modules[spec.name] = module
        spec.loader.exec_module(module)
        return module


def _pins() -> list[str]:
    """This project's max/mojo requirements, so a venv made here builds the
    base library against the toolchain the wheel pins."""
    deps = tomllib.loads((_ROOT / "pyproject.toml").read_text())["project"][
        "dependencies"
    ]
    return [d for d in deps if re.match(r"^(max|mojo)\s*==", d)]


def _package_version() -> str:
    """From pyproject, not importlib.metadata: the shim venvs hold torch
    alone, so this package is not installed in them."""
    return str(
        tomllib.loads((_ROOT / "pyproject.toml").read_text())["project"]["version"]
    )


def _torch_requirement(version: str) -> str:
    """`2.11` means the whole series (its newest patch release); `2.11.0`
    means that one."""
    return f"torch=={version}" if version.count(".") >= 2 else f"torch=={version}.*"


def _glibc_floor(lib: Path) -> str | None:
    """The newest GLIBC_x.y symbol version the library imports, i.e. the
    oldest glibc that can load it. `objdump` only: None when it is missing or
    the platform has no such versioning (macOS)."""
    if sys.platform != "linux" or shutil.which("objdump") is None:
        return None
    out = subprocess.run(
        ["objdump", "-T", str(lib)], text=True, capture_output=True
    ).stdout
    versions = {
        tuple(int(p) for p in m.group(1).split("."))
        for m in re.finditer(r"GLIBC_(\d+(?:\.\d+)+)", out)
    }
    return ".".join(str(p) for p in max(versions)) if versions else None


def _emit_one(kind: str, out_dir: Path) -> int:
    """Build one artefact in *this* interpreter and copy it into `out_dir`.

    Runs inside the venv the orchestrator just made (shim) or in the project
    venv (base library), and prints its manifest entry as one JSON line.
    """
    native = _native_module()
    out_dir.mkdir(parents=True, exist_ok=True)
    if kind == "shim":
        spec = native.prebuilt_shim_spec()
        built = native.build_shim(prebuilt=False)
        extra = {
            "torch_build": native.torch.__version__,
            "cxx_standard": native._cxx_standard().removeprefix("-std="),
            "compiler": native._cxx_identity(native._cxx()),
        }
    else:
        spec = native.prebuilt_backend_spec()
        built = native.build_backend(prebuilt=False)
        extra = {
            "mojo": native._pkg_version("mojo-compiler"),
            "target_cpu": native.portable_target_cpu(),
        }
    name = native.prebuilt_file_name(spec)
    shutil.copy2(built, out_dir / name)
    entry = {
        **spec,
        "file": name,
        "size": (out_dir / name).stat().st_size,
        "built": datetime.date.today().isoformat(),
        "package_version": _package_version(),
        **extra,
    }
    floor = _glibc_floor(out_dir / name)
    if floor is not None:
        entry["glibc_floor"] = floor
    print("MANIFEST_ENTRY " + json.dumps(entry), flush=True)
    return 0


def _entry_of(output: str) -> dict[str, object]:
    for line in output.splitlines():
        if line.startswith("MANIFEST_ENTRY "):
            entry = json.loads(line[len("MANIFEST_ENTRY ") :])
            if not isinstance(entry, dict):
                raise RuntimeError(f"malformed manifest entry: {line}")
            return entry
    raise RuntimeError("the build printed no manifest entry:\n" + output)


def _child_env(cache_dir: Path) -> dict[str, str]:
    """A build subprocess writes its .so into a throwaway cache -- the copy in
    `prebuilt/` is the artefact -- and never reads prebuilt files itself.
    PYTHONPATH carries the sources into venvs that do not have the package."""
    return {
        **os.environ,
        "PYTHONPATH": str(_ROOT),
        "TORCH_MOJO_BACKEND_CACHE_DIR": str(cache_dir),
        "TORCH_MOJO_BACKEND_PREBUILT": "0",
    }


def build_shim_for(
    version: str, out_dir: Path, work_dir: Path, python: str, index_url: str
) -> dict[str, object]:
    """One throwaway venv with that torch, then the shim build inside it."""
    venv = work_dir / f"venv-torch{version}"
    _run(["uv", "venv", "--python", python, str(venv)])
    py = venv / "bin" / "python"
    _run(
        [
            "uv",
            "pip",
            "install",
            "--python",
            str(py),
            "--index-url",
            index_url,
            _torch_requirement(version),
        ]
    )
    out = _run(
        [
            str(py),
            str(Path(__file__).resolve()),
            "--emit-one",
            "shim",
            "--out",
            str(out_dir),
        ],
        env=_child_env(work_dir / f"cache-torch{version}"),
    )
    return _entry_of(out)


def build_backend(
    out_dir: Path, work_dir: Path, python: str | None, index_url: str
) -> dict[str, object]:
    """The base library, built with the MAX of the environment this script
    runs in -- or, with `python`, in a throwaway venv holding this project's
    max/mojo pins and a CPU torch. CI uses the venv: a full project sync
    would pull the CUDA torch and its several GB of nvidia wheels, none of
    which this build reads (torch is imported only for its version)."""
    interpreter = sys.executable
    if python is not None:
        venv = work_dir / "venv-backend"
        _run(["uv", "venv", "--python", python, str(venv)])
        interpreter = str(venv / "bin" / "python")
        _run(
            [
                "uv",
                "pip",
                "install",
                "--python",
                interpreter,
                # max and mojo live on PyPI, torch on the CPU index; the local
                # version segment of `2.14.0+cpu` sorts above plain `2.14.0`,
                # so the CPU wheel wins the best-match too.
                "--index-url",
                index_url,
                "--extra-index-url",
                _PYPI,
                "--index-strategy",
                "unsafe-best-match",
                "torch",
                *_pins(),
            ]
        )
    out = _run(
        [
            interpreter,
            str(Path(__file__).resolve()),
            "--emit-one",
            "backend",
            "--out",
            str(out_dir),
        ],
        env=_child_env(work_dir / "cache-backend"),
    )
    return _entry_of(out)


def _read_entries(manifest: Path) -> list[dict[str, object]]:
    try:
        data = json.loads(manifest.read_text())
    except (OSError, ValueError):
        return []
    entries = data.get("entries", [])
    return entries if isinstance(entries, list) else []


def write_manifest(out_dir: Path, entries: list[dict[str, object]]):
    """Merge new entries into the manifest: one entry per file name, and no
    entry for a library that is no longer in the directory."""
    manifest = out_dir / "manifest.json"
    fresh = {str(e["file"]): e for e in entries}
    merged = [
        e
        for e in _read_entries(manifest)
        if str(e.get("file")) not in fresh and (out_dir / str(e.get("file"))).exists()
    ]
    merged += fresh.values()
    merged.sort(key=lambda e: (str(e.get("kind")), str(e.get("file"))))
    manifest.write_text(
        json.dumps({"manifest_version": 1, "entries": merged}, indent=2) + "\n"
    )
    _log(f"wrote {manifest} ({len(merged)} entries)")


def merge_artifacts(src: Path, out_dir: Path):
    """Collect what the per-platform CI jobs uploaded -- each a directory of
    libraries and one manifest.json -- into one prebuilt directory."""
    out_dir.mkdir(parents=True, exist_ok=True)
    entries: list[dict[str, object]] = []
    for manifest in sorted(src.rglob("manifest.json")):
        for entry in _read_entries(manifest):
            lib = manifest.parent / str(entry.get("file"))
            if not lib.exists():
                raise RuntimeError(
                    f"{manifest} names {entry.get('file')}, which is not there"
                )
            shutil.copy2(lib, out_dir / lib.name)
            entries.append(entry)
            _log(f"merged {lib.name} ({lib.stat().st_size} bytes)")
    if not entries:
        raise RuntimeError(f"no manifest.json under {src}")
    write_manifest(out_dir, entries)


def _summarize(entries: list[dict[str, object]], skipped: list[str]):
    for e in entries:
        floor = e.get("glibc_floor")
        _log(
            f"{e['kind']:8} {e['file']} {int(str(e['size'])) // 1024} KiB"
            + (f" glibc>={floor}" if floor else "")
        )
    for s in skipped:
        _log(f"skipped: {s}")


def report(out_dir: Path, max_glibc: str | None, kind: str | None) -> int:
    """Print what the directory holds, and fail when an artefact needs a newer
    glibc than `max_glibc` -- the floor a Linux build must not exceed, checked
    on the entries the build recorded (`glibc_floor`, the newest GLIBC_x.y
    symbol version the library imports)."""
    entries = _read_entries(out_dir / "manifest.json")
    if not entries:
        raise RuntimeError(f"no manifest entries in {out_dir}")
    _summarize(entries, [])
    if max_glibc is None:
        return 0
    limit = tuple(int(p) for p in max_glibc.split("."))
    checked = [e for e in entries if kind is None or e.get("kind") == kind]
    # a Linux entry with no recorded floor is a build that could not be
    # inspected: fail closed rather than ship it unmeasured
    unmeasured = [
        e
        for e in checked
        if e.get("platform") == "linux" and e.get("glibc_floor") is None
    ]
    too_new = [
        e
        for e in checked
        if e.get("glibc_floor") is not None
        and tuple(int(p) for p in str(e["glibc_floor"]).split(".")) > limit
    ]
    for e in unmeasured:
        _log(f"FAIL {e['file']} records no glibc floor")
    for e in too_new:
        _log(f"FAIL {e['file']} needs glibc {e['glibc_floor']} > {max_glibc}")
    return 1 if too_new or unmeasured else 0


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--torch",
        nargs="*",
        default=[],
        metavar="VERSION",
        help="torch versions to build a shim for (2.11 = the newest 2.11.x)",
    )
    parser.add_argument(
        "--backend",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="also build the Mojo base library (default: yes)",
    )
    parser.add_argument("--out", type=Path, default=_DEFAULT_OUT)
    parser.add_argument(
        "--work-dir",
        type=Path,
        help="where the throwaway venvs go (default: a temporary directory)",
    )
    parser.add_argument("--index-url", default=_CPU_INDEX, help="the torch CPU index")
    parser.add_argument(
        "--python",
        default=f"{sys.version_info.major}.{sys.version_info.minor}",
        help="python version for the throwaway venvs",
    )
    parser.add_argument(
        "--backend-python",
        metavar="X.Y",
        help="build the base library in a throwaway venv of this python "
        "instead of in the interpreter running this script",
    )
    parser.add_argument(
        "--strict",
        action="store_true",
        help="fail instead of skipping a torch version with no wheel here",
    )
    parser.add_argument("--merge", type=Path, metavar="DIR", help="merge CI artifacts")
    parser.add_argument(
        "--report",
        action="store_true",
        help="print what --out holds instead of building anything",
    )
    parser.add_argument(
        "--max-glibc",
        metavar="X.Y",
        help="with --report: fail if an artefact needs a newer glibc than this",
    )
    parser.add_argument(
        "--kind", choices=("shim", "backend"), help="with --report: check only these"
    )
    parser.add_argument(
        "--emit-one", choices=("shim", "backend"), help=argparse.SUPPRESS
    )
    args = parser.parse_args()

    if args.emit_one:
        return _emit_one(args.emit_one, args.out)
    if args.merge:
        merge_artifacts(args.merge, args.out)
        return 0
    if args.report:
        return report(args.out, args.max_glibc, args.kind)

    args.out.mkdir(parents=True, exist_ok=True)
    with tempfile.TemporaryDirectory(prefix="tmb-prebuilt-") as tmp:
        work = args.work_dir or Path(tmp)
        work.mkdir(parents=True, exist_ok=True)
        entries: list[dict[str, object]] = []
        skipped: list[str] = []
        for version in args.torch:
            try:
                entries.append(
                    build_shim_for(version, args.out, work, args.python, args.index_url)
                )
            except RuntimeError as exc:
                if args.strict or "uv pip install" not in str(exc):
                    raise
                skipped.append(f"torch {version}: no CPU wheel here ({exc})")
        if args.backend:
            entries.append(
                build_backend(args.out, work, args.backend_python, args.index_url)
            )
        write_manifest(args.out, entries)
        _summarize(entries, skipped)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
