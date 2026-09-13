"""The native backend's on-demand build cache (torch_mojo_backend/native),
exercised through public behavior only: env-var relocation, a second process
reusing a build, and a missing/corrupt cached `.so`.

Replaces the old `tests/test_eager_kernel_loader.py`, which unit-tested the
Python `MojoExtensionLoader`/`MojoExtension`/`_DefinedUnit` machinery
directly. That machinery does not exist in the native backend: builds are
driven from Mojo (`native/mojo/loader.mojo`) and the two backend shims from
`torch_mojo_backend/native/__init__.py`, with no Python-level descriptor or
unit cache object to import and poke at. What is left to test is the cache's
*observable* contract -- same one `TORCH_MOJO_BACKEND_CACHE_DIR`, same on-disk
`.so` files, whatever process asks -- so every test here runs the real
backend in a subprocess against a throwaway cache directory.

These tests build real Mojo extensions (a cold run compiles the C++ shim,
the Mojo backend, one extension per aten op the script touches, and one
kernel-family variant), so they are slow and need the GPU allocation like
every other native test.
"""

from __future__ import annotations

import os
import subprocess
import sys
from pathlib import Path

import pytest

from torch_mojo_backend import native

pytestmark = pytest.mark.xdist_group(name="group1")

_WORKTREE = Path(__file__).resolve().parents[2]

# One tiny script, run in a fresh process: register the backend and run one
# op (`add`) on the mojo GPU device. Real correctness is covered elsewhere;
# this only needs to exercise the build-or-reuse path for the two backend
# shims and the `logic_ops` kernel family (AddSpec/float32).
_RUN_ADD = """
import torch
from torch_mojo_backend import register_mojo_devices
register_mojo_devices()
x = torch.tensor([1.0, 2.0], device="mojo:0")
result = (x + x).cpu().tolist()
assert result == [2.0, 4.0], result
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
    return sorted(cache_dir.glob("logic_ops.*.so"))


def _op_sos(cache_dir: Path) -> list[Path]:
    """The per-op extensions: one `mojo build` of an ops_*.mojo per aten op
    (native/mojo/registry.mojo), built at that op's first call."""
    return sorted(cache_dir.glob("tmbop.*.so"))


def _assert_ok(proc: subprocess.CompletedProcess[str]):
    assert proc.returncode == 0, proc.stdout + proc.stderr
    assert "OK" in proc.stdout, proc.stdout + proc.stderr


def test_cache_dir_env_var_relocates_every_build(tmp_path: Path):
    """`TORCH_MOJO_BACKEND_CACHE_DIR` is the only place anything is written:
    the C++ shim, the Mojo backend, and the per-family kernel variant."""
    cache_dir = tmp_path / "cache"
    proc = _run(cache_dir)
    _assert_ok(proc)

    assert list(cache_dir.glob("libtmb_shim.hash-*.so")), "C++ shim not cached here"
    assert list(cache_dir.glob("libtmb_backend.hash-*.so")), (
        "Mojo backend not cached here"
    )
    family_sos = _family_sos(cache_dir)
    assert family_sos, "logic_ops (AddSpec) kernel variant not cached here"
    op_sos = _op_sos(cache_dir)
    assert op_sos, "no op extension cached here"
    # add is one of the ops the script runs, and its body lives in its own
    # extension rather than in the backend library.
    assert any(p.name.startswith("tmbop.ops_binary.add.Tensor.") for p in op_sos), [
        p.name for p in op_sos
    ]
    # A cold run must have actually built all of them, not found them by luck.
    assert "built C++ shim" in proc.stdout + proc.stderr
    assert "built Mojo backend" in proc.stdout + proc.stderr
    assert "built  logic_ops" in proc.stdout + proc.stderr
    assert "built  ops_binary add.Tensor" in proc.stdout + proc.stderr


def test_second_process_reuses_every_build(tmp_path: Path):
    """A warm second process dlopens the cached `.so`s; it builds nothing."""
    cache_dir = tmp_path / "cache"
    first = _run(cache_dir)
    _assert_ok(first)
    mtimes_before = {p: p.stat().st_mtime_ns for p in cache_dir.glob("*.so")}
    assert mtimes_before

    second = _run(cache_dir)
    _assert_ok(second)

    combined = second.stdout + second.stderr
    assert "built C++ shim" not in combined
    assert "built Mojo backend" not in combined
    assert "built  logic_ops" not in combined
    assert "built  ops_" not in combined, "an op extension was rebuilt warm"
    mtimes_after = {p: p.stat().st_mtime_ns for p in cache_dir.glob("*.so")}
    assert mtimes_after == mtimes_before, "a warm run rewrote a cached .so"


def test_missing_family_so_is_rebuilt(tmp_path: Path):
    """Deleting the cached kernel-variant `.so` (but not the two backend
    shims) makes the next process rebuild only that piece."""
    cache_dir = tmp_path / "cache"
    _assert_ok(_run(cache_dir))
    family_sos = _family_sos(cache_dir)
    assert family_sos
    for so in family_sos:
        so.unlink()
    shim_mtimes_before = {
        p: p.stat().st_mtime_ns
        for p in (
            *cache_dir.glob("libtmb_shim.hash-*.so"),
            *cache_dir.glob("libtmb_backend.hash-*.so"),
        )
    }

    proc = _run(cache_dir)
    _assert_ok(proc)

    combined = proc.stdout + proc.stderr
    assert "built  logic_ops" in combined, "missing kernel .so was not rebuilt"
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
            *cache_dir.glob("libtmb_shim.hash-*.so"),
            *cache_dir.glob("libtmb_backend.hash-*.so"),
        )
    }
    assert shim_mtimes_after == shim_mtimes_before


def test_corrupt_family_so_fails_clearly_and_recovers_once_removed(tmp_path: Path):
    """A corrupted-but-present `.so` is not silently used: the loader only
    rebuilds a *missing* file (`if not exists(so): build`), so a file that
    exists but fails to load surfaces a clear error instead of running
    garbage code or hanging. Removing it (as a user would after a truncated
    build/transfer) lets the very next process rebuild and succeed, so the
    cache is never permanently wedged by one bad file.
    """
    cache_dir = tmp_path / "cache"
    _assert_ok(_run(cache_dir))
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
    assert "built  logic_ops" in recovered.stdout + recovered.stderr


def test_missing_op_extension_is_rebuilt_alone(tmp_path: Path):
    """Deleting one op's extension makes the next process rebuild that op and
    nothing else: op bodies are cached per op, independently of the backend
    library and of the kernel families."""
    cache_dir = tmp_path / "cache"
    _assert_ok(_run(cache_dir))
    add_sos = sorted(cache_dir.glob("tmbop.ops_binary.add.Tensor.*.so"))
    assert add_sos
    for so in add_sos:
        so.unlink()
    others_before = {
        p: p.stat().st_mtime_ns
        for p in cache_dir.glob("*.so")
        if not p.name.startswith("tmbop.ops_binary.add.Tensor.")
    }

    proc = _run(cache_dir)
    _assert_ok(proc)

    combined = proc.stdout + proc.stderr
    assert "built  ops_binary add.Tensor" in combined
    assert "built Mojo backend" not in combined
    assert "built  logic_ops" not in combined
    assert sorted(cache_dir.glob("tmbop.ops_binary.add.Tensor.*.so"))
    others_after = {
        p: p.stat().st_mtime_ns
        for p in cache_dir.glob("*.so")
        if not p.name.startswith("tmbop.ops_binary.add.Tensor.")
    }
    assert others_after == others_before


def test_compiler_env_drops_the_runtime_interpreter_variables(monkeypatch):
    """The MAX runtime exports PYTHONEXECUTABLE for the interpreter it found on
    PATH; a `mojo` launcher started with it against another venv's prefix dies
    with "Could not find platform independent libraries"."""
    monkeypatch.setenv("PYTHONEXECUTABLE", "/usr/bin/python3")
    monkeypatch.setenv("PYTHONHOME", "/usr")
    env = native.compiler_env()
    assert "PYTHONEXECUTABLE" not in env and "PYTHONHOME" not in env
    assert env["MODULAR_HOME"]


def test_cxx_standard_follows_the_torch_version(monkeypatch):
    for version, expected in (
        ("2.11.0+cpu", "-std=c++17"),
        ("2.13.0", "-std=c++17"),
        ("2.14.0a0+git0", "-std=c++20"),
        ("2.14.0+cpu", "-std=c++20"),
    ):
        monkeypatch.setattr(native.torch, "__version__", version)
        assert native._cxx_standard() == expected, version
