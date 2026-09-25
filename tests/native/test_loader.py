"""The native backend's on-demand build cache (torch_mojo_backend/native),
exercised through public behavior only: env-var relocation, a second process
reusing a build, and a missing/corrupt cached `.so`.

Replaces the old `tests/test_eager_kernel_loader.py`, which unit-tested the
Python `MojoExtensionLoader`/`MojoExtension`/`_DefinedUnit` machinery
directly. That machinery does not exist in the native backend: builds are
driven from Mojo (`tmb/backend/loader.mojo`) and the two backend shims from
`torch_mojo_backend/native/__init__.py`, with no Python-level descriptor or
unit cache object to import and poke at. What is left to test is the cache's
*observable* contract -- same one `TORCH_MOJO_BACKEND_CACHE_DIR`, same on-disk
`.so` files, whatever process asks -- so every test here runs the real
backend in a subprocess against a throwaway cache directory.

These tests build real Mojo libraries (a cold run compiles the C++ shim,
the Mojo backend, and one kernel-family variant), so they are slow and need the GPU allocation like
every other native test.
"""

from __future__ import annotations

import os
import shutil
import subprocess
import sys
from pathlib import Path

import pytest

from torch_mojo_backend import native

pytestmark = pytest.mark.xdist_group(name="group1")

_WORKTREE = Path(__file__).resolve().parents[2]
_SHIM_SUFFIX = ".dylib" if sys.platform == "darwin" else ".so"

# One tiny script, run in a fresh process: register the backend and run one
# op (`add`) on the mojo GPU device. Real correctness is covered elsewhere;
# this only needs to exercise the build-or-reuse path for the two backend
# shims and the `logic` kernel family (AddSpec/float32).
_RUN_ADD = """
import torch
from torch_mojo_backend import register_mojo_devices
register_mojo_devices()
# Broadcasting exercises logic on Metal too; equal-shape contiguous
# add uses a different Metal kernel family.
x = torch.tensor([[1.0], [2.0]], device="mojo:0")
y = torch.tensor([[1.0, 2.0]], device="mojo:0")
result = (x + y).cpu().tolist()
assert result == [[2.0, 3.0], [3.0, 4.0]], result
print("OK")
"""


def _run(cache_dir: Path, *, trace: bool = True) -> subprocess.CompletedProcess[str]:
    env = dict(os.environ)
    env["TORCH_MOJO_BACKEND_CACHE_DIR"] = str(cache_dir)
    env["PYTHONPATH"] = str(_WORKTREE)
    env["TORCH_MOJO_BACKEND_TRACE"] = "1" if trace else "0"
    # These tests are about compiling and caching, so never let a prebuilt
    # library short-circuit a build (a checkout that ran
    # scripts/build_prebuilt.py has them; a plain clone does not).
    env["TORCH_MOJO_BACKEND_PREBUILT"] = "0"
    return subprocess.run(
        [sys.executable, "-c", _RUN_ADD],
        env=env,
        capture_output=True,
        text=True,
        timeout=900,
    )


def _family_sos(cache_dir: Path) -> list[Path]:
    return sorted(cache_dir.glob("logic.*.so"))


def _assert_ok(proc: subprocess.CompletedProcess[str]):
    assert proc.returncode == 0, proc.stdout + proc.stderr
    assert "OK" in proc.stdout, proc.stdout + proc.stderr


_COLD = []


@pytest.fixture
def cold(
    tmp_path_factory: pytest.TempPathFactory, mojo_gpu: str
) -> tuple[Path, subprocess.CompletedProcess[str]]:
    """The one cold build of the module: an empty cache directory filled by a
    single process. Building it per test was 15-60 s, four times over."""
    if not _COLD:
        cache_dir = tmp_path_factory.mktemp("cold") / "cache"
        proc = _run(cache_dir)
        _assert_ok(proc)
        _COLD.append((cache_dir, proc))
    return _COLD[0]


@pytest.fixture
def built_cache(
    cold: tuple[Path, subprocess.CompletedProcess[str]], tmp_path: Path
) -> Path:
    """A private copy of the cold build (mtimes kept) for a test to damage."""
    return Path(shutil.copytree(cold[0], tmp_path / "cache"))


def test_cache_dir_env_var_relocates_every_build(
    cold: tuple[Path, subprocess.CompletedProcess[str]],
):
    """`TORCH_MOJO_BACKEND_CACHE_DIR` is the only place anything is written:
    the C++ shim, the Mojo backend, and the per-family kernel variant."""
    cache_dir, proc = cold

    assert list(cache_dir.glob(f"libtmb_shim.hash-*{_SHIM_SUFFIX}")), (
        "C++ shim not cached here"
    )
    assert list(cache_dir.glob(f"libtmb_backend.hash-*{_SHIM_SUFFIX}")), (
        "Mojo backend not cached here"
    )
    family_sos = _family_sos(cache_dir)
    assert family_sos, "logic (AddSpec) kernel variant not cached here"
    assert not list(cache_dir.glob("tmbop.*")), (
        "op bodies live in the backend library, nothing is built per op"
    )
    # A cold run must have actually built all of them, not found them by luck.
    assert "built C++ shim" in proc.stdout + proc.stderr
    assert "built Mojo backend" in proc.stdout + proc.stderr
    assert "built  logic" in proc.stdout + proc.stderr


def test_second_process_reuses_every_build(built_cache: Path):
    """A warm second process dlopens the cached `.so`s; it builds nothing."""
    cache_dir = built_cache
    mtimes_before = {
        p: p.stat().st_mtime_ns
        for p in cache_dir.iterdir()
        if p.suffix in (".so", ".dylib")
    }
    assert mtimes_before

    second = _run(cache_dir)
    _assert_ok(second)

    combined = second.stdout + second.stderr
    assert "built C++ shim" not in combined
    assert "built Mojo backend" not in combined
    assert "built  logic" not in combined
    mtimes_after = {
        p: p.stat().st_mtime_ns
        for p in cache_dir.iterdir()
        if p.suffix in (".so", ".dylib")
    }
    assert mtimes_after == mtimes_before, "a warm run rewrote a cached .so"


def test_missing_family_so_is_rebuilt(built_cache: Path):
    """Deleting the cached kernel-variant `.so` (but not the two backend
    shims) makes the next process rebuild only that piece."""
    cache_dir = built_cache
    family_sos = _family_sos(cache_dir)
    assert family_sos
    for so in family_sos:
        so.unlink()
    shim_mtimes_before = {
        p: p.stat().st_mtime_ns
        for p in (
            *cache_dir.glob(f"libtmb_shim.hash-*{_SHIM_SUFFIX}"),
            *cache_dir.glob(f"libtmb_backend.hash-*{_SHIM_SUFFIX}"),
        )
    }

    proc = _run(cache_dir)
    _assert_ok(proc)

    combined = proc.stdout + proc.stderr
    assert "built  logic" in combined, "missing kernel .so was not rebuilt"
    assert "built C++ shim" not in combined, (
        "the shim was still on disk; it must not rebuild"
    )
    assert "built Mojo backend" not in combined, (
        "the backend was still on disk; it must not rebuild"
    )
    assert _family_sos(cache_dir), "rebuild did not reinstall the .so"
    shim_mtimes_after = {
        p: p.stat().st_mtime_ns
        for p in (
            *cache_dir.glob(f"libtmb_shim.hash-*{_SHIM_SUFFIX}"),
            *cache_dir.glob(f"libtmb_backend.hash-*{_SHIM_SUFFIX}"),
        )
    }
    assert shim_mtimes_after == shim_mtimes_before


def test_corrupt_family_so_fails_clearly_and_recovers_once_removed(built_cache: Path):
    """A corrupted-but-present `.so` is not silently used: the loader only
    rebuilds a *missing* file (`if not exists(so): build`), so a file that
    exists but fails to load surfaces a clear error instead of running
    garbage code or hanging. Removing it (as a user would after a truncated
    build/transfer) lets the very next process rebuild and succeed, so the
    cache is never permanently wedged by one bad file.
    """
    cache_dir = built_cache
    family_sos = _family_sos(cache_dir)
    assert family_sos
    corrupted = family_sos[0]
    corrupted.write_bytes(b"not a shared library")

    proc = _run(cache_dir)
    assert proc.returncode != 0, "a corrupt .so must not silently succeed"
    combined = proc.stdout + proc.stderr
    assert "OK" not in proc.stdout
    # Never a segfault/hang: a clean non-zero exit with a message, not -SIGSEGV.
    assert proc.returncode > 0, combined
    assert combined.strip(), "the failure produced no diagnostic at all"

    corrupted.unlink()
    recovered = _run(cache_dir)
    _assert_ok(recovered)
    assert "built  logic" in recovered.stdout + recovered.stderr


def test_compiler_env_drops_the_runtime_interpreter_variables(monkeypatch):
    """The MAX runtime exports PYTHONEXECUTABLE for the interpreter it found on
    PATH; a `mojo` launcher started with it against another venv's prefix dies
    with "Could not find platform independent libraries"."""
    monkeypatch.setenv("PYTHONEXECUTABLE", "/usr/bin/python3")
    monkeypatch.setenv("PYTHONHOME", "/usr")
    env = native.compiler_env()
    assert "PYTHONEXECUTABLE" not in env and "PYTHONHOME" not in env
    assert env["MODULAR_HOME"] and env["MODULAR_CACHE_DIR"]


def test_kernel_call_defines_and_owned_spec_lifetimes(tmp_path):
    """Lazy specialization metadata preserves defines and stable spec pointers."""
    binary = tmp_path / "kernel_call_probe"
    subprocess.run(
        [
            "mojo",
            "build",
            str(Path(__file__).with_name("kernel_call_probe.mojo")),
            "-I",
            str(_WORKTREE / "torch_mojo_backend/mojo"),
            "--Werror",
            "-o",
            str(binary),
        ],
        env=native.compiler_env(),
        check=True,
        capture_output=True,
        text=True,
        timeout=180,
    )
    subprocess.run([str(binary)], check=True, timeout=30)


def test_werror_flag_reaches_every_python_driven_mojo_build(monkeypatch, tmp_path):
    """TORCH_MOJO_BACKEND_WERROR=1 (what conftest sets) turns compiler warnings
    into build failures; unset, a user's build must not carry --Werror. The
    Mojo-side builds (loader.mojo, kernel families) read
    the same variable."""
    monkeypatch.delenv("TORCH_MOJO_BACKEND_WERROR", raising=False)
    assert "--Werror" not in native.backend_build_command(tmp_path / "a.so")
    assert native.mojo_diagnostic_flags() == []
    monkeypatch.setenv("TORCH_MOJO_BACKEND_WERROR", "1")
    assert "--Werror" in native.backend_build_command(tmp_path / "a.so")
    assert native.mojo_diagnostic_flags() == ["--Werror"]


def test_cxx_standard_follows_the_torch_version(monkeypatch):
    for version, expected in (
        ("2.11.0+cpu", "-std=c++17"),
        ("2.13.0", "-std=c++17"),
        ("2.14.0a0+git0", "-std=c++20"),
        ("2.14.0+cpu", "-std=c++20"),
    ):
        monkeypatch.setattr(native.torch, "__version__", version)
        assert native._cxx_standard() == expected, version
