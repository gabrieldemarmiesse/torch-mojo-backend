"""Unit tests for the per-operation Mojo extension loader."""

import subprocess
from pathlib import Path
from types import ModuleType

import pytest

from torch_mojo_backend import eager_kernels


def test_build_variant_compiles_original_source(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    package_dir = tmp_path / "eager_kernels"
    package_dir.mkdir()
    source = package_dir / "sample_ops.mojo"
    source_text = (
        '@export\ndef PyInit_sample_ops() abi("C") -> PythonObject:\n'
        "    return PythonObject(None)\n"
    )
    source.write_text(source_text)
    commands: list[list[str]] = []

    def fake_find_mojo() -> Path:
        return Path("/fake/mojo")

    def fake_run(args: list[str], **kwargs: object) -> subprocess.CompletedProcess[str]:
        commands.append(args)
        Path(args[-1]).write_bytes(b"fake shared library")
        return subprocess.CompletedProcess(args, 0, "", "")

    monkeypatch.setattr(eager_kernels, "_PACKAGE_DIR", package_dir)
    monkeypatch.setattr(eager_kernels, "_CACHE_DIR", package_dir / "__mojocache__")
    monkeypatch.setattr(eager_kernels, "_find_mojo", fake_find_mojo)
    monkeypatch.setattr(eager_kernels.subprocess, "run", fake_run)

    result = eager_kernels._build_variant(
        "sample_ops", frozenset({"SampleOp"}), frozenset({"float32"})
    )

    assert result.is_file()
    assert len(commands) == 1
    assert commands[0][2] == str(source)
    assert source.read_text() == source_text
    assert not list(package_dir.glob("_tmbv_*.mojo"))


def test_variants_load_with_original_pyinit_name(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    loaded: list[tuple[str, Path]] = []

    def fake_build_variant(
        name: str, ops: frozenset[str] | None, dtypes: frozenset[str] | None
    ) -> Path:
        assert name == "elementwise_ops"
        assert ops is not None
        assert dtypes == frozenset({"float32"})
        return tmp_path / f"{next(iter(ops))}.so"

    def fake_load_extension(module_name: str, so_path: Path) -> ModuleType:
        loaded.append((module_name, so_path))
        return ModuleType(f"loaded_{so_path.stem}")

    holder_state = eager_kernels._STATES["tensor_holder"]
    monkeypatch.setattr(holder_state, "module", ModuleType("tensor_holder"))
    monkeypatch.setattr(eager_kernels, "_build_variant", fake_build_variant)
    monkeypatch.setattr(eager_kernels, "_load_extension", fake_load_extension)

    first = eager_kernels._import_mojo_module(
        "elementwise_ops", frozenset({"ReluSpec"}), frozenset({"float32"})
    )
    second = eager_kernels._import_mojo_module(
        "elementwise_ops", frozenset({"ExpSpec"}), frozenset({"float32"})
    )

    expected_name = f"{eager_kernels.__name__}.elementwise_ops"
    assert loaded == [
        (expected_name, tmp_path / "ReluSpec.so"),
        (expected_name, tmp_path / "ExpSpec.so"),
    ]
    assert first is not second


def test_module_hash_includes_loader_cache_abi(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    source = tmp_path / "sample_ops.mojo"
    source.write_text("def PyInit_sample_ops():\n    pass\n")
    monkeypatch.setattr(eager_kernels, "_PACKAGE_DIR", tmp_path)

    initial = eager_kernels._module_hash("sample_ops")
    monkeypatch.setattr(eager_kernels, "_VARIANT_CACHE_ABI", b"next-loader-abi")

    assert eager_kernels._module_hash("sample_ops") != initial
