"""Unit tests for stateless, exactly specialized Mojo extensions."""

import inspect
import subprocess
from collections import deque
from dataclasses import dataclass
from pathlib import Path
from types import ModuleType

import pytest

from torch_mojo_backend import eager_kernels


def test_defines_are_canonical_independent_of_mapping_order() -> None:
    first = {
        "OP": "AddSpec",
        "DTYPE_ARG_1": "bfloat16",
        "INPLACE": False,
        "DTYPE_ARG_0": "float32",
    }
    second = {
        "DTYPE_ARG_0": "float32",
        "INPLACE": False,
        "DTYPE_ARG_1": "bfloat16",
        "OP": "AddSpec",
    }

    expected = (
        ("DTYPE_ARG_0", "float32"),
        ("DTYPE_ARG_1", "bfloat16"),
        ("INPLACE", "0"),
        ("OP", "AddSpec"),
    )
    assert eager_kernels.normalize_defines(first) == expected
    assert eager_kernels.normalize_defines(second) == expected
    assert eager_kernels.defines_cache_string(
        first
    ) == eager_kernels.defines_cache_string(second)


def test_exact_call_defines_use_ordered_argument_and_output_roles() -> None:
    defines = eager_kernels.exact_call_defines(
        "WhereSelect",
        ("bool", "float32", "bfloat16"),
        output_dtypes=("float32",),
        flags={"INPLACE": False},
    )

    assert defines == {
        "DTYPE_ARG_0": "bool",
        "DTYPE_ARG_1": "float32",
        "DTYPE_ARG_2": "bfloat16",
        "DTYPE_OUT": "float32",
        "INPLACE": "0",
        "OP": "WhereSelect",
    }
    assert not any(name.startswith("DTYPE_COMPILE") for name in defines)


@pytest.mark.parametrize("name", ["op", "HAS-DASH", "1_DTYPE", ""])
def test_define_names_must_be_explicit_compiler_identifiers(name: str) -> None:
    with pytest.raises(ValueError, match="define name"):
        eager_kernels.normalize_defines({name: "value"})


@pytest.mark.parametrize("value", [None, 1.5, object()])
def test_define_values_reject_types_outside_the_compiler_contract(
    value: object,
) -> None:
    normalize_value = inspect.unwrap(eager_kernels._normalize_define_value)
    with pytest.raises(TypeError, match="must be bool, int, or str"):
        normalize_value(value)


def test_build_extension_compiles_original_source(
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

    # As they arrive here: already canonical.
    defines = eager_kernels.normalize_defines(
        {
            "OP": "AddSpec",
            "DTYPE_ARG_1": "float32",
            "DTYPE_OUT": "bfloat16",
            "DTYPE_ARG_0": "float32",
        }
    )
    result = eager_kernels._build_extension(source, defines)

    assert result.is_file()
    assert len(commands) == 1
    assert commands[0][2] == str(source)
    define_args = [
        commands[0][index + 1]
        for index, value in enumerate(commands[0][:-1])
        if value == "-D"
    ]
    assert define_args == [
        "DTYPE_ARG_0=float32",
        "DTYPE_ARG_1=float32",
        "DTYPE_OUT=bfloat16",
        "OP=AddSpec",
    ]
    assert source.read_text() == source_text
    assert not list(package_dir.glob("_tmbv_*.mojo"))


# A source whose gates read OP and DTYPE_ARG_0 and nothing else, so the
# loader can tell which defines select its generated code.
_GATED_SOURCE = (
    "from variant_gates import _op_on, _dtype_arg_on, _register_call\n"
    "\n"
    "def PyInit_elementwise_ops():\n"
    '    comptime if _op_on["AddSpec"]() and _dtype_arg_on[0, DType.float32]():\n'
    "        pass\n"
)


def test_loader_reuses_one_variant_per_distinct_define_set(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    """One .so per distinct define set, and exactly one.

    The key is the defines as sent, so order must not matter but any
    difference in name or value must fork a build. A define the source never
    gates on is not special-cased: it forks a byte-identical build, which is
    a wasted first-use compile and the reason not to emit dead defines at the
    call site.
    """
    source = tmp_path / "elementwise_ops.mojo"
    source.write_text(_GATED_SOURCE)
    builds: list[tuple[Path, eager_kernels.CanonicalDefines | None]] = []
    loaded: list[ModuleType] = []

    def fake_build_extension(
        mojo_file: Path, defines: eager_kernels.CanonicalDefines | None
    ) -> Path:
        builds.append((mojo_file, defines))
        return tmp_path / f"variant-{len(builds)}.so"

    def fake_load_extension(module_name: str, so_path: Path) -> ModuleType:
        module = ModuleType(f"loaded_{so_path.stem}")
        module.call = lambda: so_path.name  # type: ignore[attr-defined]
        loaded.append(module)
        return module

    def fake_holder() -> ModuleType:
        return ModuleType("tensor_holder")

    monkeypatch.setattr(eager_kernels, "_ensure_tensor_holder", fake_holder)
    monkeypatch.setattr(eager_kernels, "_build_extension", fake_build_extension)
    monkeypatch.setattr(eager_kernels, "_load_extension", fake_load_extension)

    loader = eager_kernels.MojoExtensionLoader()

    def load(**defines: eager_kernels.DefineValue) -> ModuleType:
        return loader.load_canonical(source, eager_kernels.normalize_defines(defines))

    base = load(OP="AddSpec", DTYPE_ARG_0="float32", INPLACE=False)
    same_defines_different_order = load(
        INPLACE=False, DTYPE_ARG_0="float32", OP="AddSpec"
    )
    unread_define_differs = load(OP="AddSpec", DTYPE_ARG_0="float32", INPLACE=True)
    different_dtype = load(OP="AddSpec", DTYPE_ARG_0="bfloat16", INPLACE=False)
    different_op = load(OP="MulSpec", DTYPE_ARG_0="float32", INPLACE=False)

    assert same_defines_different_order is base
    assert all(
        module is not base
        for module in (unread_define_differs, different_dtype, different_op)
    )
    assert len(builds) == 4
    assert len(loaded) == 4
    # Every define reaches the compiler, including the one no gate reads.
    assert sum(
        1 for _, defines in builds for name, _ in defines or () if name == "INPLACE"
    ) == len(builds)
    assert callable(base.call)
    assert not hasattr(base, "AddSpec")


def test_source_hash_includes_loader_cache_abi(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    source = tmp_path / "sample_ops.mojo"
    source.write_text("def PyInit_sample_ops():\n    pass\n")
    monkeypatch.setattr(eager_kernels, "_PACKAGE_DIR", tmp_path)

    initial = eager_kernels._source_hash(source)
    monkeypatch.setattr(eager_kernels, "_VARIANT_CACHE_ABI", b"next-loader-abi")

    assert eager_kernels._source_hash(source) != initial


def test_source_hash_includes_compiler_toolchain_identity(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    source = tmp_path / "sample_ops.mojo"
    source.write_text("def PyInit_sample_ops():\n    pass\n")
    monkeypatch.setattr(eager_kernels, "_PACKAGE_DIR", tmp_path)

    initial = eager_kernels._source_hash(source)
    monkeypatch.setattr(eager_kernels, "_TOOLCHAIN_IDENTITY", b"another-compiler")

    assert eager_kernels._source_hash(source) != initial


def test_nested_module_hash_includes_private_and_shared_dependencies(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    package_dir = tmp_path / "eager_kernels"
    operation_dir = package_dir / "sample_ops"
    operation_dir.mkdir(parents=True)
    source = operation_dir / "sample_ops.mojo"
    private_helper = operation_dir / "private_helper.mojo"
    shared_helper = package_dir / "shared_helper.mojo"
    source.write_text(
        "from private_helper import private_call\n"
        "from shared_helper import shared_call\n"
    )
    private_helper.write_text("fn private_call():\n    pass\n")
    shared_helper.write_text("fn shared_call():\n    pass\n")
    monkeypatch.setattr(eager_kernels, "_PACKAGE_DIR", package_dir)

    initial = eager_kernels._source_hash(source)
    private_helper.write_text("fn private_call():\n    return\n")
    after_private_change = eager_kernels._source_hash(source)
    shared_helper.write_text("fn shared_call():\n    return\n")

    assert after_private_change != initial
    assert eager_kernels._source_hash(source) != after_private_change


def test_defined_unit_memoizes_one_failed_build_across_all_waiters(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    source = tmp_path / "elementwise_ops.mojo"
    source.write_text("def PyInit_elementwise_ops():\n    pass\n")
    defines = eager_kernels.normalize_defines(
        {"OP": "AddSpec", "DTYPE_ARG_0": "float32"}
    )
    failure = ImportError("compiler failed once")
    attempts = 0

    def fail_build(
        mojo_file: Path, call_defines: eager_kernels.CanonicalDefines | None
    ) -> Path:
        nonlocal attempts
        assert mojo_file == source
        assert call_defines == defines
        attempts += 1
        raise failure

    def fake_holder() -> ModuleType:
        return ModuleType("tensor_holder")

    monkeypatch.setattr(eager_kernels, "_ensure_tensor_holder", fake_holder)
    monkeypatch.setattr(eager_kernels, "_build_extension", fail_build)
    unit = eager_kernels._DefinedUnit(source, defines)

    job = unit.request_async()
    with pytest.raises(ImportError) as first:
        job.wait()
    assert first.value is failure
    assert unit.request_async() is job
    with pytest.raises(ImportError) as second:
        unit.request_async().wait()
    assert second.value is failure
    with pytest.raises(ImportError) as blocking:
        unit.load_blocking()
    assert blocking.value is failure
    assert attempts == 1


@dataclass(frozen=True)
class _TensorMetadata:
    shape: tuple[int, ...]
    dtype: str


class _ElementwiseAdd(eager_kernels.MojoExtension[_TensorMetadata, _TensorMetadata]):
    MOJO_FILE = Path("elementwise_ops.mojo")

    @classmethod
    def make_defines(
        cls, left: _TensorMetadata, right: _TensorMetadata, inplace: bool = False
    ) -> dict[str, eager_kernels.DefineValue]:
        return {
            "OP": "AddSpec",
            "DTYPE_ARG_0": left.dtype,
            "DTYPE_ARG_1": right.dtype,
            "DTYPE_OUT": left.dtype,
            "INPLACE": inplace,
        }

    @classmethod
    def expected_output_specs(
        cls, left: _TensorMetadata, right: _TensorMetadata, inplace: bool = False
    ) -> _TensorMetadata:
        del right, inplace
        return _TensorMetadata(left.shape, left.dtype)

    @classmethod
    def call_extension(
        cls,
        extension: ModuleType,
        output_specs: _TensorMetadata,
        left: _TensorMetadata,
        right: _TensorMetadata,
        inplace: bool = False,
    ) -> _TensorMetadata:
        return extension.call(  # type: ignore[attr-defined, no-any-return]
            left, right, output_specs, inplace
        )


def test_descriptor_prepares_output_metadata_without_loading_and_has_no_state(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    def unexpected_build(
        source: Path, defines: eager_kernels.CanonicalDefines | None
    ) -> Path:
        raise AssertionError(f"prepare compiled {source} with {defines}")

    monkeypatch.setattr(eager_kernels, "_build_extension", unexpected_build)
    class_attributes = set(_ElementwiseAdd.__dict__)
    small = _ElementwiseAdd.prepare(
        _TensorMetadata((2, 3), "float32"), _TensorMetadata((2, 3), "bfloat16")
    )
    large = _ElementwiseAdd.prepare(
        _TensorMetadata((128, 256), "float32"), _TensorMetadata((128, 256), "bfloat16")
    )

    assert _ElementwiseAdd.MOJO_FILE == Path("elementwise_ops.mojo")
    assert small.extension is _ElementwiseAdd
    assert small.output_specs == _TensorMetadata((2, 3), "float32")
    assert large.output_specs == _TensorMetadata((128, 256), "float32")
    assert small.defines == large.defines
    loader = eager_kernels.MojoExtensionLoader()
    assert loader._unit(_ElementwiseAdd.MOJO_FILE, small.defines) is loader._unit(
        _ElementwiseAdd.MOJO_FILE, large.defines
    )
    assert set(_ElementwiseAdd.__dict__) == class_attributes
    with pytest.raises(TypeError, match="stateless descriptor"):
        _ElementwiseAdd()


def test_prepared_call_invokes_only_constant_call_entrypoint() -> None:
    calls: list[tuple[object, ...]] = []
    module = ModuleType("defined_elementwise_add")

    def call(*args: object) -> _TensorMetadata:
        calls.append(args)
        output = args[2]
        assert isinstance(output, _TensorMetadata)
        return output

    module.call = call  # type: ignore[attr-defined]

    class FakeLoader(eager_kernels.MojoExtensionLoader):
        def load_canonical(
            self, mojo_file: Path, defines: eager_kernels.CanonicalDefines
        ) -> ModuleType:
            assert mojo_file == _ElementwiseAdd.MOJO_FILE
            assert ("OP", "AddSpec") in defines
            return module

    left = _TensorMetadata((4, 8), "float32")
    right = _TensorMetadata((4, 8), "float32")
    prepared = _ElementwiseAdd.prepare(left, right, True)

    assert prepared.execute(FakeLoader()) == _TensorMetadata((4, 8), "float32")
    assert calls == [(left, right, prepared.output_specs, True)]
    assert not hasattr(module, "AddSpec")


def test_prepared_calls_enqueue_in_fifo_order(monkeypatch: pytest.MonkeyPatch) -> None:
    launches: list[str] = []

    class FakeJob:
        def wait(self) -> None:
            return None

    class QueuedUnit:
        """The whole contract the queue needs of a unit: a module once its
        build lands, and a job to wait on until then."""

        def __init__(self) -> None:
            self.ext: ModuleType | None = None
            self.job = FakeJob()

        def request_async(self) -> FakeJob:
            return self.job

    unit = QueuedUnit()

    class FakeLoader(eager_kernels.MojoExtensionLoader):
        def unit_canonical(
            self, mojo_file: Path, defines: eager_kernels.CanonicalDefines
        ) -> object:
            assert mojo_file == _ElementwiseAdd.MOJO_FILE
            assert ("OP", "AddSpec") in defines
            return unit

    queue = eager_kernels.call_queue
    monkeypatch.setattr(queue, "_QUEUE", deque())
    monkeypatch.setattr(queue, "_HELD_ERROR", [])
    monkeypatch.setattr(queue, "_DEVICE_THREAD", [None])
    monkeypatch.setattr(queue, "_QUEUE_LAUNCH_THREAD", [None])

    loader = FakeLoader()
    first = _ElementwiseAdd.prepare(
        _TensorMetadata((2,), "float32"), _TensorMetadata((2,), "float32")
    )
    second = _ElementwiseAdd.prepare(
        _TensorMetadata((9,), "float32"), _TensorMetadata((9,), "float32")
    )
    first.enqueue_into(("first",), (), loader)
    second.enqueue_into(("second",), (), loader)
    assert queue.active()

    module = ModuleType("queued_elementwise_add")

    def call(label: str) -> None:
        launches.append(label)

    module.call = call  # type: ignore[attr-defined]
    unit.ext = module
    queue.drain()

    assert launches == ["first", "second"]
    assert not queue.active()


def test_spec_descriptor_canonical_defines_match_make_defines() -> None:
    """The memoized canonical defines must stay field-for-field equal to the
    literal ``make_defines`` dicts: the define-gate scanner reads the dicts,
    the hot path uses the memo, and the two must never drift."""
    from dataclasses import dataclass, field

    from max.driver import CPU
    from max.dtype import DType

    from torch_mojo_backend.eager_kernels import aten_fast

    cpu = CPU()

    @dataclass
    class _FakePayload:
        _dtype: DType
        _shape: tuple[int, ...] = (2, 3)
        _strides: tuple[int, ...] = (3, 1)
        _device: object = field(default_factory=lambda: cpu)

    a = _FakePayload(DType.float32)
    b = _FakePayload(DType.bfloat16)
    cases = [
        (aten_fast._FillSpecExtension, ((4,), 1.0, DType.float32, cpu)),
        (aten_fast._CastSpecExtension, (a, DType.bfloat16)),
        (aten_fast._BinarySpecExtension, ("AddSpec", a, b, DType.float32)),
        (aten_fast._ElementwiseUnarySpecExtension, ("ReluSpec", a, DType.float32)),
        (
            aten_fast._ReductionOpsSpecExtension,
            ("SumSpec", a, (1,), False, (), DType.float32),
        ),
        (aten_fast._MinDimSpecExtension, (a, 1, False)),
        (aten_fast._MatmulSpecExtension, ("MatmulSpec", (a, b), 1)),
    ]
    for extension, args in cases:
        expected = eager_kernels.normalize_defines(extension.make_defines(*args))
        assert extension.make_canonical_defines(*args) == expected, extension.__name__
