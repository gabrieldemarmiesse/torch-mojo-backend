"""TORCH_MOJO_BACKEND_WERROR=1 holds on a cache hit.

Every Mojo build records the compiler's warnings beside its artifact
(`<artifact>.warnings`, empty when clean), and WERROR is checked against that
record, fresh build or cached -- so an artifact built once without WERROR
cannot hide its warnings from a later WERROR run. A record-less artifact
(cached before records existed) is rebuilt once under WERROR; without WERROR
the record is never read.

Both build paths are covered with a tiny Mojo source carrying an unused
variable: the Python-driven builds through `native.build_library` (the
backend base library and the graph package go through the same helpers),
and the kernel-family builds through a standalone probe that drives
`tmb/backend/loader.mojo`'s `Loader` over a fake source tree. Neither needs
a GPU.
"""

from __future__ import annotations

import json
import os
import re
import subprocess
from pathlib import Path

import pytest

from torch_mojo_backend import native

_HERE = Path(__file__).resolve().parent

_WARNS = """\
@export
def tmb_call() abi("C") -> Int32:
    var unused = 1
    return 0
"""
_CLEAN = """\
@export
def tmb_call() abi("C") -> Int32:
    return 0
"""
_WARNING = "assignment to 'unused' was never used"

# What the compiler prints, as observed from mojo 1.1 (`mojo build` with an
# unused variable, in an imported module too, and a deprecated call), with
# the noise around it.
_COMPILER_OUTPUT = """\
Failed to initialize Crashpad.  Crash reporting will not be available.  \
Cause: while locating crashpad handler: unable to locate crashpad handler
/src/a.mojo:4:2: warning: @export requires an explicit 'abi()' effect
@export
 ^
/src/a.mojo:7:12: warning: use new_thing
    return old_thing[n]()
           ^~~~~~~~~
Imported from /src/a.mojo:1:
/src/a.mojo:2:5: note: 'old_thing' declared here
def old_thing[n: Int]() -> Int:
    ^
[TRACE] built logic in 1.0s
Included from /src/entry.mojo:1:
/src/b.mojo:9:1: error: an error is not a warning
    var x: Int = "a"
                 ^
Imported from /src/entry.mojo:1:
/src/b.mojo:3:1: note: nor is its note
    def f():
    ^
Included from /src/entry.mojo:1:
/src/b.mojo:2:18: warning: assignment to 'unused' was never used
    var unused = 1
                 ^
/a path/with spaces/c.mojo:1:1: warning: a spaced path
x = 1
^
/usr/bin/mojo: warning: a warning with no location
ld: warning: a linker's, which --Werror never counted
warning: a bare one, which mojo does not print
ptxas info    : Used 10 registers
some text: warning: not a diagnostic, its prefix has spaces
"""
_EXPECTED = """\
/src/a.mojo:4:2: warning: @export requires an explicit 'abi()' effect
@export
 ^
/src/a.mojo:7:12: warning: use new_thing
    return old_thing[n]()
           ^~~~~~~~~
Imported from /src/a.mojo:1:
/src/a.mojo:2:5: note: 'old_thing' declared here
def old_thing[n: Int]() -> Int:
    ^
Included from /src/entry.mojo:1:
/src/b.mojo:2:18: warning: assignment to 'unused' was never used
    var unused = 1
                 ^
/a path/with spaces/c.mojo:1:1: warning: a spaced path
x = 1
^
/usr/bin/mojo: warning: a warning with no location
"""


def test_the_filter_keeps_warnings_and_nothing_else():
    assert native.warning_diagnostics(_COMPILER_OUTPUT) == _EXPECTED
    assert native.warning_diagnostics("") == ""
    assert native.warning_diagnostics("Failed to initialize Crashpad.\n") == ""


# --- The Python-driven builds ---------------------------------------------


@pytest.fixture
def cache(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> Path:
    cache = tmp_path / "cache"
    monkeypatch.setattr(native, "_CACHE_DIR", cache)
    monkeypatch.setenv("TORCH_MOJO_BACKEND_TRACE", "1")
    return cache


def _entry(tmp_path: Path, text: str) -> Path:
    entry = tmp_path / "src" / "entry.mojo"
    entry.parent.mkdir(exist_ok=True)
    entry.write_text(text)
    return entry


def _build(
    entry: Path,
    werror: bool,
    monkeypatch: pytest.MonkeyPatch,
    capsys: pytest.CaptureFixture[str],
) -> tuple[Path | None, str | None, bool]:
    """(artifact, error, compiled): one `build_library` call."""
    monkeypatch.setenv("TORCH_MOJO_BACKEND_WERROR", "1" if werror else "0")
    capsys.readouterr()
    try:
        out, error = native.build_library(entry, "warnprobe"), None
    except RuntimeError as exc:
        out, error = None, str(exc)
    return out, error, "built libwarnprobe" in capsys.readouterr().err


def test_a_warning_cached_without_werror_fails_a_werror_hit(
    tmp_path: Path, cache: Path, monkeypatch, capsys
):
    entry = _entry(tmp_path, _WARNS)
    out, error, compiled = _build(entry, False, monkeypatch, capsys)
    assert out is not None and error is None and compiled
    record = native.warnings_record(out)
    assert _WARNING in record.read_text()
    assert "Crashpad" not in record.read_text()

    hit, error, compiled = _build(entry, True, monkeypatch, capsys)
    assert hit is None and not compiled
    assert error is not None
    assert error.startswith("building libwarnprobe failed:\n")
    assert "TORCH_MOJO_BACKEND_WERROR=1" in error and _WARNING in error


def test_a_fresh_werror_build_and_a_hit_fail_alike(
    tmp_path: Path, cache: Path, monkeypatch, capsys
):
    entry = _entry(tmp_path, _WARNS)
    _, fresh, compiled = _build(entry, True, monkeypatch, capsys)
    assert fresh is not None and compiled
    _, cached, compiled = _build(entry, True, monkeypatch, capsys)
    assert cached == fresh and not compiled
    # The artifact itself is fine: without WERROR it is used as built.
    out, error, compiled = _build(entry, False, monkeypatch, capsys)
    assert out is not None and error is None and not compiled


def test_a_clean_build_passes_werror_with_an_empty_record(
    tmp_path: Path, cache: Path, monkeypatch, capsys
):
    entry = _entry(tmp_path, _CLEAN)
    out, error, compiled = _build(entry, True, monkeypatch, capsys)
    assert out is not None and error is None and compiled
    assert native.warnings_record(out).read_text() == ""
    again, error, compiled = _build(entry, True, monkeypatch, capsys)
    assert again == out and error is None and not compiled


@pytest.mark.parametrize("source", [_CLEAN, _WARNS], ids=["clean", "warns"])
def test_a_record_less_artifact_is_rebuilt_once_under_werror(
    tmp_path: Path, cache: Path, monkeypatch, capsys, source: str
):
    entry = _entry(tmp_path, source)
    out, _, _ = _build(entry, False, monkeypatch, capsys)
    assert out is not None
    native.warnings_record(out).unlink()  # what an older version left behind
    out.write_bytes(b"legacy")  # tells the old artifact from a rebuilt one

    _, error, compiled = _build(entry, True, monkeypatch, capsys)
    assert compiled
    # Same key, same build: the one in place is kept (another process may
    # have it mapped), and only the record is added.
    assert out.read_bytes() == b"legacy"
    assert native.warnings_record(out).exists()
    assert (error is not None) == (source == _WARNS)
    _, again, compiled = _build(entry, True, monkeypatch, capsys)
    assert again == error and not compiled


def _unreadable(path: Path) -> bool:
    path.chmod(0)
    return not os.access(path, os.R_OK)  # root reads it anyway


def test_an_unreadable_record_is_rebuilt_under_werror(
    tmp_path: Path, cache: Path, monkeypatch, capsys
):
    entry = _entry(tmp_path, _WARNS)
    out, _, _ = _build(entry, False, monkeypatch, capsys)
    assert out is not None
    if not _unreadable(native.warnings_record(out)):
        pytest.skip("this user reads any file")
    _, error, compiled = _build(entry, True, monkeypatch, capsys)
    assert compiled and error is not None and _WARNING in error


def test_a_hit_without_werror_never_reads_the_record(
    tmp_path: Path, cache: Path, monkeypatch, capsys
):
    entry = _entry(tmp_path, _WARNS)
    out, _, _ = _build(entry, False, monkeypatch, capsys)
    assert out is not None
    record = native.warnings_record(out)
    record.unlink()
    record.mkdir()  # reading this raises; a WERROR lookup would rebuild
    hit, error, compiled = _build(entry, False, monkeypatch, capsys)
    assert hit == out and error is None and not compiled


def test_the_record_is_published_after_the_artifact(
    tmp_path: Path, cache: Path, monkeypatch, capsys
):
    """The record is published after the artifact, so an install interrupted
    between the two leaves an entry WERROR rebuilds rather than trusts."""
    entry = _entry(tmp_path, _CLEAN)
    out, _, _ = _build(entry, False, monkeypatch, capsys)
    assert out is not None
    native.warnings_record(out).unlink()
    seen = []
    replace = native._replace_from_scratch

    def spy(tmp: Path, dst: Path):
        seen.append(native.warnings_record(dst).exists())
        replace(tmp, dst)

    monkeypatch.setattr(native, "_replace_from_scratch", spy)
    out.unlink()
    native.warnings_record(out).write_text("a stale record\n")
    _, error, compiled = _build(entry, True, monkeypatch, capsys)
    assert compiled and error is None
    assert seen == [False]
    assert native.warnings_record(out).read_text() == ""


@pytest.fixture
def clean_backend(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> Path:
    """`build_backend` compiling a tiny clean library instead of the base
    library, and a wheel shipping a prebuilt one for this environment."""
    source = _entry(tmp_path, _CLEAN)

    def command(out: Path, accelerator: str | None = None) -> list[str]:
        mojo = native._find_mojo()
        return [mojo, "build", str(source), "--emit", "shared-lib", "-o", str(out)]

    monkeypatch.setattr(native, "backend_build_command", command)
    prebuilt = tmp_path / "prebuilt"
    prebuilt.mkdir()
    spec = native.prebuilt_backend_spec()
    name = native.prebuilt_file_name(spec)
    (prebuilt / name).write_bytes(b"prebuilt")
    (prebuilt / "manifest.json").write_text(
        json.dumps({"manifest_version": 1, "entries": [{**spec, "file": name}]})
    )
    monkeypatch.setattr(native, "_PREBUILT", prebuilt)
    monkeypatch.setattr(native, "_PREBUILT_MANIFEST", prebuilt / "manifest.json")
    monkeypatch.delenv("TORCH_MOJO_BACKEND_PREBUILT", raising=False)
    return prebuilt


def test_werror_compiles_the_backend_instead_of_taking_the_prebuilt(
    clean_backend: Path, cache: Path, monkeypatch
):
    monkeypatch.setenv("TORCH_MOJO_BACKEND_WERROR", "1")
    out = native.build_backend()
    assert out.read_bytes() != b"prebuilt"
    assert native.warnings_record(out).read_text() == ""
    assert not (cache / (out.name + native._PREBUILT_MARK)).exists()


def test_werror_records_a_cached_prebuilt_copy_without_replacing_it(
    clean_backend: Path, cache: Path, monkeypatch, capsys
):
    monkeypatch.setenv("TORCH_MOJO_BACKEND_WERROR", "0")
    out = native.build_backend()
    mark = cache / (out.name + native._PREBUILT_MARK)
    assert out.read_bytes() == b"prebuilt" and mark.exists()
    assert not native.warnings_record(out).exists()

    monkeypatch.setenv("TORCH_MOJO_BACKEND_WERROR", "1")
    capsys.readouterr()
    assert native.build_backend() == out
    assert "built Mojo backend" in capsys.readouterr().err
    # Another process may have the copy mapped: it stays, mark and all.
    assert out.read_bytes() == b"prebuilt" and mark.exists()
    assert native.warnings_record(out).read_text() == ""
    assert native.build_backend() == out
    assert "built Mojo backend" not in capsys.readouterr().err


def test_the_graph_package_replays_its_warnings(
    tmp_path: Path, cache: Path, monkeypatch
):
    package = tmp_path / "pkgwarn"
    package.mkdir()
    (package / "__init__.mojo").write_text("")
    (package / "ops.mojo").write_text(
        "def helper() -> Int:\n    var unused = 1\n    return 2\n"
    )
    sources = sorted(package.glob("*.mojo"))

    monkeypatch.setenv("TORCH_MOJO_BACKEND_WERROR", "0")
    out = native._precompile_package(package, sources, [])
    assert _WARNING in native.warnings_record(out).read_text()
    monkeypatch.setenv("TORCH_MOJO_BACKEND_WERROR", "1")
    with pytest.raises(RuntimeError, match="precompiling pkgwarn") as failure:
        native._precompile_package(package, sources, [])
    assert _WARNING in str(failure.value)


def test_no_mojo_build_passes_werror_to_the_compiler(monkeypatch, tmp_path: Path):
    """The check is on the record, so the compiler never gets `--Werror`:
    the artifact of a build that warned is cached like any other."""
    monkeypatch.setenv("TORCH_MOJO_BACKEND_WERROR", "1")
    assert "--Werror" not in native.backend_build_command(tmp_path / "a.so")


# --- The kernel-family builds (tmb/backend/loader.mojo) -------------------


@pytest.fixture(scope="module")
def probe(tmp_path_factory: pytest.TempPathFactory) -> Path:
    binary = tmp_path_factory.mktemp("probe") / "loader_werror_probe"
    proc = subprocess.run(
        [
            native._find_mojo(),
            "build",
            str(_HERE / "loader_werror_probe.mojo"),
            "-I",
            str(native._MOJO_ROOT),
            "--Werror",
            "-o",
            str(binary),
        ],
        env=native.compiler_env(),
        capture_output=True,
        text=True,
        timeout=600,
    )
    assert proc.returncode == 0, proc.stdout + proc.stderr
    return binary


@pytest.mark.parametrize(
    "output",
    [
        _COMPILER_OUTPUT,
        _COMPILER_OUTPUT.rstrip("\n"),
        _COMPILER_OUTPUT.replace("\n", "\r\n"),
    ],
    ids=["lf", "no-final-newline", "crlf"],
)
def test_the_loader_filter_is_the_python_filter(
    probe: Path, tmp_path: Path, output: str
):
    sample = tmp_path / "output.txt"
    sample.write_bytes(output.encode())
    proc = subprocess.run(
        [str(probe), "filter", str(sample)], capture_output=True, timeout=60, check=True
    )
    assert proc.stdout.decode() == native.warning_diagnostics(output) == _EXPECTED


class _Family:
    """A fake source tree holding one kernel family, and a cache for it."""

    def __init__(self, probe: Path, tmp_path: Path, name: str, source: str):
        self.probe, self.name, self.tmp = probe, name, tmp_path
        self.root = tmp_path / "root"
        family = self.root / "tmb" / "kernels" / name
        family.mkdir(parents=True)
        (family / "entry.mojo").write_text(source)
        self.cache = tmp_path / "cache"

    def entry(self, werror: bool) -> tuple[str, bool]:
        """(the probe's last line, whether the loader compiled)."""
        env = native.compiler_env()  # node-local compiler caches, kept warm
        env["TMPDIR"] = str(self.tmp)  # source snapshots of this test only
        env["TORCH_MOJO_BACKEND_WERROR"] = "1" if werror else "0"
        proc = subprocess.run(
            [
                str(self.probe),
                "entry",
                str(self.root),
                str(self.cache),
                native._find_mojo(),
                self.name,
            ],
            env=env,
            capture_output=True,
            text=True,
            timeout=600,
        )
        assert proc.returncode == 0, proc.stdout + proc.stderr
        failed = proc.stdout.find("FAILED: ")
        result = proc.stdout[failed:] if failed >= 0 else proc.stdout.splitlines()[-1]
        built = re.search(rf"\[TRACE\] built +{self.name} ", proc.stdout)
        return result.strip(), built is not None

    def so(self) -> Path:
        (so,) = self.cache.glob(f"{self.name}.*.so")
        return so


def test_loader_a_warning_cached_without_werror_fails_a_werror_hit(
    probe: Path, tmp_path: Path
):
    fam = _Family(probe, tmp_path, "warnprobe", _WARNS)
    assert fam.entry(werror=False) == ("OK", True)
    record = native.warnings_record(fam.so())
    assert _WARNING in record.read_text()
    assert "Crashpad" not in record.read_text()

    result, compiled = fam.entry(werror=True)
    assert not compiled
    assert result.startswith("FAILED: mojo build of warnprobe -D ")
    assert "TORCH_MOJO_BACKEND_WERROR=1" in result and _WARNING in result


def test_loader_a_fresh_werror_build_and_a_hit_fail_alike(probe: Path, tmp_path: Path):
    fam = _Family(probe, tmp_path, "warnprobe", _WARNS)
    fresh, compiled = fam.entry(werror=True)
    assert fresh.startswith("FAILED: ") and compiled
    assert fam.entry(werror=True) == (fresh, False)
    assert fam.entry(werror=False) == ("OK", False)


def test_loader_a_clean_build_passes_werror(probe: Path, tmp_path: Path):
    fam = _Family(probe, tmp_path, "cleanprobe", _CLEAN)
    assert fam.entry(werror=True) == ("OK", True)
    assert native.warnings_record(fam.so()).read_text() == ""
    assert fam.entry(werror=True) == ("OK", False)


def test_loader_a_record_less_so_is_rebuilt_once_under_werror(
    probe: Path, tmp_path: Path
):
    fam = _Family(probe, tmp_path, "warnprobe", _WARNS)
    assert fam.entry(werror=False) == ("OK", True)
    native.warnings_record(fam.so()).unlink()
    # Without WERROR the missing record is not even noticed.
    assert fam.entry(werror=False) == ("OK", False)
    before = fam.so().stat()
    result, compiled = fam.entry(werror=True)
    assert result.startswith("FAILED: ") and compiled
    after = fam.so().stat()  # kept, as on the Python side
    assert (after.st_ino, after.st_mtime_ns) == (before.st_ino, before.st_mtime_ns)
    assert native.warnings_record(fam.so()).exists()
    assert fam.entry(werror=True) == (result, False)


def test_loader_an_unreadable_record_is_rebuilt_under_werror(
    probe: Path, tmp_path: Path
):
    fam = _Family(probe, tmp_path, "warnprobe", _WARNS)
    assert fam.entry(werror=False) == ("OK", True)
    if not _unreadable(native.warnings_record(fam.so())):
        pytest.skip("this user reads any file")
    result, compiled = fam.entry(werror=True)
    assert result.startswith("FAILED: ") and compiled and _WARNING in result


def test_loader_a_hit_without_werror_never_reads_the_record(
    probe: Path, tmp_path: Path
):
    fam = _Family(probe, tmp_path, "warnprobe", _WARNS)
    assert fam.entry(werror=False) == ("OK", True)
    record = native.warnings_record(fam.so())
    record.unlink()
    record.mkdir()  # reading this fails
    assert fam.entry(werror=False) == ("OK", False)
