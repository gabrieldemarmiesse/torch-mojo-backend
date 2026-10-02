"""Every build's cache key is the hash of the entry's import closure, so an
edit to shared code (the SIMD math in tmb/kernels/common that both the graph
backend and the eager kernels use, the op_utils every family imports) must
invalidate the kernel builds AND the backend library that reach it.

The two walkers -- `Loader._closure` in tmb/backend/loader.mojo for the
on-demand kernel builds and `native.mojo_import_closure` for the
Python-driven ones -- implement one rule: `from tmb.a.b import` names
`<root>/tmb/a/b.mojo`, `from .b import` inside a package names a sibling,
anything else is the toolchain's. Both are exercised here on the same
throwaway tree, and on the real tree they must agree.
"""

import subprocess
from pathlib import Path

import pytest

from torch_mojo_backend import native

PACKAGE = Path(__file__).resolve().parents[1] / "torch_mojo_backend"
REAL_ROOT = PACKAGE / "mojo"


def _fake_root(tmp_path: Path) -> tuple[Path, Path, Path]:
    """A miniature of the real layout: a family entry importing op_utils,
    which imports the shared math, which imports a sibling relatively (the
    grammar tmb/graph's modules still use, so the walkers must follow it)."""
    root = tmp_path / "mojo"
    (root / "tmb/kernels/example").mkdir(parents=True)
    (root / "tmb/kernels/common").mkdir()
    (root / "tmb/backend").mkdir()
    (root / "tmb/kernels/example/entry.mojo").write_text(
        "from tmb.kernels.common.op_utils import Argv\n"
        "from std.ffi import external_call\n"
    )
    (root / "tmb/kernels/common/op_utils.mojo").write_text(
        "from tmb.kernels.common.unary_math import elementwise_unary\n"
    )
    unary = root / "tmb/kernels/common/unary_math.mojo"
    unary.write_text("from .math_utils import ieee_sqrt\n# first implementation\n")
    math = root / "tmb/kernels/common/math_utils.mojo"
    math.write_text("# original math\n")
    (root / "tmb/backend/entry.mojo").write_text(
        "from tmb.kernels.common.op_utils import Argv\n"
    )
    return root, unary, math


def test_python_walker_follows_absolute_and_relative_imports(
    tmp_path: Path, monkeypatch
):
    root, unary, math = _fake_root(tmp_path)
    monkeypatch.setattr(native, "_MOJO_ROOT", root)
    closure = native.mojo_import_closure(root / "tmb/kernels/example/entry.mojo")
    assert [p.relative_to(root).as_posix() for p in closure] == [
        "tmb/kernels/common/math_utils.mojo",
        "tmb/kernels/common/op_utils.mojo",
        "tmb/kernels/common/unary_math.mojo",
        "tmb/kernels/example/entry.mojo",
    ]


def test_shared_math_invalidates_backend_cache(tmp_path: Path, monkeypatch):
    root, unary, math = _fake_root(tmp_path)
    monkeypatch.setattr(native, "_MOJO_ROOT", root)
    entry = root / "tmb/backend/entry.mojo"
    monkeypatch.setattr(native, "_BACKEND_ENTRY", entry)
    before = native._hash_files(native.mojo_import_closure(entry), "test")
    math.write_text("# changed math\n")
    assert native._hash_files(native.mojo_import_closure(entry), "test") != before


@pytest.fixture(scope="module")
def cache_probe(tmp_path_factory: pytest.TempPathFactory) -> Path:
    """A tiny executable around the real Loader: prints one family's source
    hash for the root given on the command line."""
    tmp = tmp_path_factory.mktemp("probe")
    source = tmp / "cache_probe.mojo"
    source.write_text(
        "from tmb.backend.loader import Loader\n"
        "from std.sys import argv\n\n"
        "def main() raises:\n"
        '    var loader = Loader(argv()[1], "", "", "test", False)\n'
        "    print(loader.source_hash(argv()[2]))\n"
    )
    executable = tmp / "cache_probe"
    build = subprocess.run(
        [
            native._find_mojo(),
            "build",
            str(source),
            "-I",
            str(REAL_ROOT),
            "-o",
            str(executable),
            "--Werror",
        ],
        capture_output=True,
        text=True,
        env=native.compiler_env(),
    )
    assert build.returncode == 0, build.stdout + build.stderr
    return executable


@pytest.mark.parametrize("edited", ["unary_math", "math_utils"])
def test_shared_math_invalidates_native_cache(
    tmp_path: Path, cache_probe: Path, edited: str
):
    root, unary, math = _fake_root(tmp_path)

    def source_hash() -> str:
        return subprocess.check_output(
            [str(cache_probe), str(root), "example"], text=True
        ).strip()

    before = source_hash()
    assert source_hash() == before
    (unary if edited == "unary_math" else math).write_text("# changed\n")
    assert source_hash() != before


def test_both_walkers_agree_on_the_real_tree(cache_probe: Path, tmp_path: Path):
    """The Mojo loader and the Python builder must hash the same files for a
    family, or an edit could rebuild one and not the other. Compared through
    a Mojo probe that lists the closure the same way the loader hashes it."""
    source = tmp_path / "closure_probe.mojo"
    source.write_text(
        "from tmb.backend.loader import Loader\n"
        "from std.sys import argv\n\n"
        "def main() raises:\n"
        '    var loader = Loader(argv()[1], "", "", "test", False)\n'
        "    for f in loader._closure_texts(argv()[2])[0]:\n"
        "        print(f)\n"
    )
    executable = tmp_path / "closure_probe"
    build = subprocess.run(
        [
            native._find_mojo(),
            "build",
            str(source),
            "-I",
            str(REAL_ROOT),
            "-o",
            str(executable),
            "--Werror",
        ],
        capture_output=True,
        text=True,
        env=native.compiler_env(),
    )
    assert build.returncode == 0, build.stdout + build.stderr
    for family in ("logic", "matmul", "fa4"):
        entry = REAL_ROOT / "tmb/kernels" / family / "entry.mojo"
        from_mojo = sorted(
            subprocess.check_output(
                [str(executable), str(REAL_ROOT), str(entry)], text=True
            ).split()
        )
        from_python = sorted(str(p) for p in native.mojo_import_closure(entry))
        assert from_mojo == from_python, family
        assert any(p.endswith("tmb/kernels/common/op_utils.mojo") for p in from_python)
        assert any(
            p.endswith("tmb/kernels/common/math_utils.mojo") for p in from_python
        )
