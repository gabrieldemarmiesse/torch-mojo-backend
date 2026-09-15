"""`torch-mojo-backend cache dir` / `cache clean` against a relocated cache."""

import os
import subprocess
import sys
from pathlib import Path


def _run(cache_dir: Path, *args: str) -> subprocess.CompletedProcess[str]:
    # conftest turns on VERBOSE, which prints import-time diagnostics to
    # stdout; the CLI's stdout must be exactly the answer.
    env = dict(
        os.environ,
        TORCH_MOJO_BACKEND_CACHE_DIR=str(cache_dir),
        TORCH_MOJO_BACKEND_VERBOSE="0",
    )
    return subprocess.run(
        [sys.executable, "-m", "torch_mojo_backend.cli", *args],
        env=env,
        capture_output=True,
        text=True,
        check=True,
    )


def test_cache_dir_prints_the_relocated_path(tmp_path: Path):
    cache_dir = tmp_path / "cache"
    assert _run(cache_dir, "cache", "dir").stdout.strip() == str(cache_dir)


def test_cache_clean_removes_the_directory(tmp_path: Path):
    cache_dir = tmp_path / "cache"
    cache_dir.mkdir()
    (cache_dir / "libtmb_shim.hash-0.so").write_bytes(b"stale")
    proc = _run(cache_dir, "cache", "clean")
    assert not cache_dir.exists()
    assert str(cache_dir) in proc.stderr


def test_cache_clean_on_a_missing_directory_succeeds(tmp_path: Path):
    proc = _run(tmp_path / "absent", "cache", "clean")
    assert "nothing to remove" in proc.stderr
