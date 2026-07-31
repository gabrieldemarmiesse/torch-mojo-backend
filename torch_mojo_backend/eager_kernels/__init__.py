"""Fast eager-mode kernels for mojo_device, compiled as CPython extensions.

The `.mojo` modules in this package are built on demand with
`mojo build --emit shared-lib` into per-op units: every gated .so compiles
exactly one extension entry point named `call`. Canonical compiler defines
such as `OP`, `DTYPE_ARG_0`, and `DTYPE_OUT` select the operation, ordered
argument dtypes, and compile-time flags for one GPU-call family. Shapes and
strides are runtime arguments and never enter the specialization key.

Each Python operation descriptor is a stateless `MojoExtension` class. It
selects one immutable specialization and invokes that extension's constant
`call` entry point. A different shape reuses the same .so; a different dtype
tuple or flag set selects another .so. There is no dtype escalation or retry
build.

An op call here is one CPython extension call that receives raw data
pointers (from `TorchMojoTensor._ptr`) plus sizes/dtypes as plain ints, and
enqueues a kernel on MAX's own DeviceContext, so it stays correctly ordered
with every other MAX driver operation on that device.

`tensor_holder` is exempt from gating: it registers the process-wide
`TensorHolder`/`TensorSpec` Python types (a duplicate `add_type` aborts the
process), so it is always built complete and never escalated — which also
means direct references to its functions stay valid forever.
"""

import fcntl
import hashlib
import importlib.machinery
import importlib.metadata
import importlib.util
import json
import os
import platform
import re
import subprocess
import sys
import threading
from abc import ABC, abstractmethod
from collections.abc import Mapping
from dataclasses import dataclass
from pathlib import Path
from types import ModuleType
from typing import ClassVar, Generic, TypeVar

from max import driver
from max.dtype import DType

from . import call_queue

_PACKAGE_DIR = Path(__file__).parent
_CACHE_DIR = _PACKAGE_DIR / "__mojocache__"


_MOJO_EXE_CACHE: list = []


def _find_mojo() -> Path:
    """The mojo CLI of the environment providing `max`. Resolved lazily at
    first build: sys.executable can be unset during worker bootstrap
    (pytest-xdist) at import time. `max` may be a namespace package
    (__file__ is None), so walk up from its __path__ entries."""
    if _MOJO_EXE_CACHE:
        return _MOJO_EXE_CACHE[0]
    candidates: list[Path] = []
    if sys.executable:  # None/'' in embedded interpreters and workers
        candidates.append(Path(sys.executable).parent / "mojo")
    import max as _max_pkg

    for base in list(getattr(_max_pkg, "__path__", ())) or (
        [_max_pkg.__file__] if getattr(_max_pkg, "__file__", None) else []
    ):
        candidates.extend(parent / "bin" / "mojo" for parent in Path(base).parents)
    for cand in candidates:
        if cand.is_file():
            _MOJO_EXE_CACHE.append(cand)
            return cand
    import shutil

    which = shutil.which("mojo")
    if which is not None:
        _MOJO_EXE_CACHE.append(Path(which))
        return _MOJO_EXE_CACHE[0]
    raise FileNotFoundError("mojo executable not found for kernel builds")


# Part of every extension cache key. Bump whenever the loader ABI changes
# without necessarily changing a Mojo source file. Version 2 keeps each
# source's original PyInit_<module> symbol instead of compiling a mangled copy.
_VARIANT_CACHE_ABI = b"stateless-mojo-extension-v5"


def _toolchain_identity() -> bytes:
    """Stable host/compiler identity for native-extension cache entries.

    The MAX and Mojo package versions identify the compiler and bundled Mojo
    libraries without invoking the compiler during a cache lookup.  The
    CPython ABI and host platform prevent a shared library built by another
    interpreter or host architecture from being reused.  Accelerator-specific
    identity is deliberately absent until the compiler's target API is known.
    """
    package_versions: list[str] = []
    for package in ("mojo-compiler", "max-core", "max-mojo-libs"):
        try:
            version = importlib.metadata.version(package)
        except importlib.metadata.PackageNotFoundError:
            version = "missing"
        package_versions.append(f"{package}={version}")
    fields = (
        *package_versions,
        f"python={sys.implementation.cache_tag}",
        f"platform={sys.platform}",
        f"machine={platform.machine()}",
    )
    return "|".join(fields).encode()


_TOOLCHAIN_IDENTITY = _toolchain_identity()

_IMPORT_RE = re.compile(r"^(?:from|import)\s+(\w+)", re.M)


DefineValue = bool | int | str
CanonicalDefines = tuple[tuple[str, str], ...]


def _normalize_define_value(value: DefineValue) -> str:
    if not isinstance(value, bool | int | str):
        raise TypeError(
            "compiler define values must be bool, int, or str, "
            f"got {type(value).__name__}"
        )
    if isinstance(value, bool):
        return "1" if value else "0"
    result = str(value)
    if "\x00" in result or "\n" in result or "\r" in result:
        raise ValueError(f"invalid compiler define value {result!r}")
    return result


def normalize_defines(defines: Mapping[str, DefineValue]) -> CanonicalDefines:
    """Validate and canonicalize compiler defines independently of dict order."""
    normalized: list[tuple[str, str]] = []
    for name, value in defines.items():
        if not re.fullmatch(r"[A-Z][A-Z0-9_]*", name):
            raise ValueError(f"invalid compiler define name {name!r}")
        normalized.append((name, _normalize_define_value(value)))
    normalized.sort()
    return tuple(normalized)


def defines_cache_string(defines: Mapping[str, DefineValue]) -> str:
    """Unambiguous, stable serialization used to identify a specialization."""
    return json.dumps(normalize_defines(defines), separators=(",", ":"))


def exact_call_defines(
    op: str,
    arg_dtypes: tuple[DType | str, ...],
    *,
    output_dtypes: tuple[DType | str, ...] = (),
    flags: Mapping[str, DefineValue] | None = None,
) -> dict[str, DefineValue]:
    """Build the shape-independent compiler definitions for one exact call."""
    values: dict[str, DefineValue] = {"OP": op}
    values.update(
        (f"DTYPE_ARG_{index}", _dtype_name(dtype))
        for index, dtype in enumerate(arg_dtypes)
    )
    if len(output_dtypes) == 1:
        values["DTYPE_OUT"] = _dtype_name(output_dtypes[0])
    else:
        values.update(
            (f"DTYPE_OUT_{index}", _dtype_name(dtype))
            for index, dtype in enumerate(output_dtypes)
        )
    for name, value in (flags or {}).items():
        define_name = name.upper()
        if define_name in values:
            raise ValueError(f"duplicate specialization define {define_name!r}")
        values[define_name] = value
    # Validate here so malformed values fail while the call is prepared, not
    # in a background compiler thread. The returned mapping remains convenient
    # for MojoExtension.make_defines().
    return dict(normalize_defines(values))


def _canonical_cache_string(defines: CanonicalDefines) -> str:
    return json.dumps(defines, separators=(",", ":"))


def _dtype_name(dtype: DType | str) -> str:
    name = dtype if isinstance(dtype, str) else dtype.name
    if not re.fullmatch(r"[A-Za-z][A-Za-z0-9_]*", name):
        raise ValueError(f"invalid specialization dtype {name!r}")
    return name


def _dep_closure(source: Path) -> list[Path]:
    """Return the local Mojo dependency surface for one extension source.

    An operation directory owns its entry point and private helpers, while
    shared helpers live at the eager-kernel package root. Both locations are
    searched in the same order as the compiler command.
    """
    source = source.resolve()
    package_dir = _PACKAGE_DIR.resolve()
    search_roots = [source.parent]
    if source.is_relative_to(package_dir) and source.parent != package_dir:
        search_roots.append(package_dir)
    seen: set[Path] = set()
    todo = [source]
    files: list[Path] = []
    while todo:
        path = todo.pop().resolve()
        if path in seen:
            continue
        seen.add(path)
        if not path.is_file():
            continue
        files.append(path)
        for dep in _IMPORT_RE.findall(path.read_text()):
            for root in search_roots:
                candidate = root / f"{dep}.mojo"
                if candidate.is_file():
                    todo.append(candidate)
                    break
    for root in search_roots:
        op_utils = root / "op_utils"
        if op_utils.is_dir():
            files.extend(
                path for path in sorted(op_utils.rglob("*.mojo")) if path not in seen
            )
    return files


def _module_hash(name: str) -> str:
    return _source_hash(_PACKAGE_DIR / f"{name}.mojo")


def _source_hash(source: Path) -> str:
    hasher = hashlib.sha256()
    hasher.update(_VARIANT_CACHE_ABI)
    hasher.update(_TOOLCHAIN_IDENTITY)
    for path in _dep_closure(source):
        try:
            relative_path = path.relative_to(_PACKAGE_DIR.resolve())
        except ValueError:
            relative_path = path.relative_to(source.resolve().parent)
        hasher.update(str(relative_path).encode())
        hasher.update(path.read_bytes())
    return hasher.hexdigest()[:16]


def _defines_tag(defines: CanonicalDefines | None) -> str:
    if defines is None:
        return "full"
    return hashlib.sha256(_canonical_cache_string(defines).encode()).hexdigest()[:12]


def _build_env() -> dict[str, str]:
    """Environment for `mojo build` subprocesses. Once the MAX runtime is
    loaded it exports MODULAR_*PACKAGE_ROOT/IMPORT_PATH overrides meant for
    embedded payloads; they break the standalone CLI's own package discovery
    (`No module named 'mojo'`), so strip them."""
    return {
        k: v
        for k, v in os.environ.items()
        if not (
            k.startswith("MODULAR_") and ("PACKAGE_ROOT" in k or "IMPORT_PATH" in k)
        )
    }


def _extension_cmd(src: Path, defines: CanonicalDefines | None, out: Path) -> list[str]:
    cmd = [str(_find_mojo()), "build", str(src), "--emit", "shared-lib"]
    import_roots = [src.parent]
    if _PACKAGE_DIR not in import_roots:
        import_roots.append(_PACKAGE_DIR)
    for import_root in import_roots:
        cmd += ["-I", str(import_root)]
    for name, value in defines or ():
        cmd += ["-D", f"{name}={value}"]
    return cmd + ["-o", str(out)]


def _extension_path(src: Path, defines: CanonicalDefines | None) -> Path:
    return _CACHE_DIR / (
        f"{src.stem}.{_defines_tag(defines)}.hash-{_source_hash(src)}.so"
    )


def _build_extension(src: Path, defines: CanonicalDefines | None) -> Path:
    """Compile one immutable defined extension and return its cache path."""
    out = _extension_path(src, defines)
    if out.is_file():
        return out
    _CACHE_DIR.mkdir(exist_ok=True)
    with open(_CACHE_DIR / f".{out.stem}.lock", "w") as lock:
        fcntl.flock(lock, fcntl.LOCK_EX)
        if out.is_file():
            return out
        scope = "full" if defines is None else dict(defines).get("OP", "defined")
        import time as _time

        print(
            f"torch-mojo-backend: compiling {src.stem}.{scope} on demand..."
            + (f" t={_time.monotonic():.2f}" if os.environ.get("TMB_TRACE") else ""),
            file=sys.stderr,
        )
        # Build to a temp path and rename into place: `mojo build -o` writes
        # the output non-atomically, and the fast-path existence check above
        # runs WITHOUT the lock — a reader must never see a partial .so.
        # The temp name keeps the .so suffix (mojo normalizes others) and a
        # leading dot so no cache scan ever picks it up.
        tmp = out.with_name(f".tmp{os.getpid()}.{out.name}")
        try:
            proc = subprocess.run(
                _extension_cmd(src, defines, tmp),
                capture_output=True,
                text=True,
                env=_build_env(),
            )
            if os.environ.get("TMB_TRACE"):
                print(
                    f"[TRACE] built {src.stem}.{scope} t={_time.monotonic():.2f}",
                    file=sys.stderr,
                    flush=True,
                )
            if proc.returncode != 0:
                raise ImportError(
                    f"mojo build failed for {src.stem} "
                    f"({_defines_tag(defines)}):\n{proc.stderr}"
                )
            os.replace(tmp, out)
        finally:
            tmp.unlink(missing_ok=True)
        return out


def _load_extension(module_name: str, so_path: Path) -> ModuleType:
    loader = importlib.machinery.ExtensionFileLoader(module_name, str(so_path))
    spec = importlib.util.spec_from_file_location(
        module_name, str(so_path), loader=loader
    )
    module = importlib.util.module_from_spec(spec)
    loader.exec_module(module)
    return module


_FULL_MODULE_LOCK = threading.Lock()
_FULL_MODULE_CACHE: dict[str, ModuleType] = {}
_FULL_MODULE_FAILURES: dict[str, BaseException] = {}


def _load_full_module(name: str) -> ModuleType:
    """Load a deliberately ungated singleton extension exactly once."""
    with _FULL_MODULE_LOCK:
        module = _FULL_MODULE_CACHE.get(name)
        if module is not None:
            return module
        failure = _FULL_MODULE_FAILURES.get(name)
        if failure is not None:
            raise failure
        try:
            source = _PACKAGE_DIR / f"{name}.mojo"
            module = _load_extension(
                f"{__name__}.{name}", _build_extension(source, None)
            )
        except BaseException as exc:
            _FULL_MODULE_FAILURES[name] = exc
            raise
        _FULL_MODULE_CACHE[name] = module
        return module


def _ensure_tensor_holder() -> ModuleType:
    """Register the process-wide TensorHolder/TensorSpec Python types."""
    return _load_full_module("tensor_holder")


def _resolve_mojo_file(mojo_file: Path) -> Path:
    return mojo_file if mojo_file.is_absolute() else _PACKAGE_DIR / mojo_file


class _AsyncLoadJob:
    def __init__(self, unit: "_DefinedUnit") -> None:
        self.unit = unit
        self.done = threading.Event()
        self.error: BaseException | None = None

    def run(self) -> None:
        try:
            with _ASYNC_BUILD_SLOTS:
                self.unit.load_blocking()
        except BaseException as exc:
            self.error = exc
        finally:
            self.done.set()

    def wait(self) -> ModuleType:
        self.done.wait()
        if self.error is not None:
            raise self.error
        if self.unit.module is None:
            raise RuntimeError("extension load completed without a module")
        return self.unit.module


class _DefinedUnit:
    def __init__(self, mojo_file: Path, defines: CanonicalDefines) -> None:
        self.mojo_file = mojo_file
        self.defines = defines
        self.lock = threading.Lock()
        self.load_lock = threading.Lock()
        self.module: ModuleType | None = None
        self.failure: BaseException | None = None
        self.job: _AsyncLoadJob | None = None

    def load_blocking(self) -> ModuleType:
        # Serialize direct and background callers. The filesystem lock avoids
        # duplicate compiler processes, but this lock also prevents duplicate
        # imports and gives the unit one permanent success-or-failure result.
        with self.load_lock:
            with self.lock:
                if self.module is not None:
                    return self.module
                if self.failure is not None:
                    raise self.failure
            try:
                if self.mojo_file.stem != "tensor_holder":
                    _ensure_tensor_holder()
                so_path = _build_extension(self.mojo_file, self.defines)
                module = _load_extension(f"{__name__}.{self.mojo_file.stem}", so_path)
                if not callable(getattr(module, "call", None)):
                    raise ImportError(
                        f"specialized extension {self.mojo_file.stem!r} does "
                        "not expose call(); check its OP define"
                    )
            except BaseException as exc:
                with self.lock:
                    if self.failure is None:
                        self.failure = exc
                    failure = self.failure
                raise failure
            with self.lock:
                self.module = module
                return module

    @property
    def ext(self) -> ModuleType | None:
        """Queue-unit compatibility: the native loaded module, if ready."""
        return self.module

    def resolve(self, attr: str) -> object:
        if attr != "call":
            raise AttributeError(f"specialized modules expose call(), not {attr!r}")
        module = self.load_blocking() if self.module is None else self.module
        return getattr(module, "call")

    def request_async(self) -> _AsyncLoadJob:
        with self.lock:
            job = self.job
            if job is None:
                job = _AsyncLoadJob(self)
                self.job = job
                if self.module is not None:
                    job.done.set()
                    return job
                if self.failure is not None:
                    job.error = self.failure
                    job.done.set()
                    return job
                threading.Thread(
                    target=job.run,
                    name=f"mojo-build-{self.mojo_file.stem}",
                    daemon=True,
                ).start()
            return job


class MojoExtensionLoader:
    """Stateful build/module cache shared by stateless operation classes."""

    def __init__(self) -> None:
        self._lock = threading.Lock()
        self._units: dict[tuple[Path, CanonicalDefines], _DefinedUnit] = {}

    def _unit(self, mojo_file: Path, defines: CanonicalDefines) -> _DefinedUnit:
        source = _resolve_mojo_file(mojo_file).resolve()
        key = (source, defines)
        with self._lock:
            unit = self._units.get(key)
            if unit is None:
                unit = self._units[key] = _DefinedUnit(source, defines)
            return unit

    def load(self, mojo_file: Path, defines: Mapping[str, DefineValue]) -> ModuleType:
        return self._unit(mojo_file, normalize_defines(defines)).load_blocking()

    def load_canonical(self, mojo_file: Path, defines: CanonicalDefines) -> ModuleType:
        return self._unit(mojo_file, defines).load_blocking()

    def request_async(
        self, mojo_file: Path, defines: Mapping[str, DefineValue]
    ) -> _AsyncLoadJob:
        return self._unit(mojo_file, normalize_defines(defines)).request_async()

    def request_canonical_async(
        self, mojo_file: Path, defines: CanonicalDefines
    ) -> _AsyncLoadJob:
        return self._unit(mojo_file, defines).request_async()

    def unit_canonical(
        self, mojo_file: Path, defines: CanonicalDefines
    ) -> _DefinedUnit:
        """Return the queue-compatible unit for an already canonical key."""
        return self._unit(mojo_file, defines)


_OutputSpecs = TypeVar("_OutputSpecs")
_ExtensionResult = TypeVar("_ExtensionResult")


@dataclass(frozen=True)
class PreparedExtensionCall(Generic[_OutputSpecs, _ExtensionResult]):
    extension: type["MojoExtension"]
    defines: CanonicalDefines
    output_specs: _OutputSpecs
    args: tuple[object, ...]
    kwargs: tuple[tuple[str, object], ...]

    def request_module_async(
        self, loader: MojoExtensionLoader | None = None
    ) -> _AsyncLoadJob:
        selected_loader = loader or MOJO_EXTENSION_LOADER
        return selected_loader.request_canonical_async(
            self.extension.MOJO_FILE, self.defines
        )

    def get_loaded_module(
        self, loader: MojoExtensionLoader | None = None
    ) -> ModuleType:
        selected_loader = loader or MOJO_EXTENSION_LOADER
        return selected_loader.load_canonical(self.extension.MOJO_FILE, self.defines)

    def execute(self, loader: MojoExtensionLoader | None = None) -> _ExtensionResult:
        module = self.get_loaded_module(loader)
        return self.extension.call_extension(
            module, self.output_specs, *self.args, **dict(self.kwargs)
        )

    def enqueue_into(
        self,
        extension_args: tuple[object, ...],
        loader: MojoExtensionLoader | None = None,
    ) -> _OutputSpecs:
        """Queue a non-returning `call(..., out)` and return inferred outputs."""
        selected_loader = loader or MOJO_EXTENSION_LOADER
        unit = selected_loader.unit_canonical(self.extension.MOJO_FILE, self.defines)
        call_queue.kernel_call_into(unit, "call", extension_args)
        return self.output_specs


class MojoExtension(ABC, Generic[_OutputSpecs, _ExtensionResult]):
    """Stateless operation descriptor for one Mojo source specialization."""

    __slots__ = ()
    MOJO_FILE: ClassVar[Path]

    def __new__(cls) -> "MojoExtension":
        raise TypeError(
            f"{cls.__name__} is a stateless descriptor and cannot be instantiated"
        )

    @staticmethod
    def str_from_defined_dict(defines: Mapping[str, DefineValue]) -> str:
        return defines_cache_string(defines)

    @classmethod
    @abstractmethod
    def make_defines(cls, *args: object, **kwargs: object) -> Mapping[str, DefineValue]:
        """Capture every compile-time value; never include shapes or strides."""

    @classmethod
    @abstractmethod
    def expected_output_specs(cls, *args: object, **kwargs: object) -> _OutputSpecs:
        """Infer output metadata without compiling or executing the extension."""

    @classmethod
    @abstractmethod
    def call_extension(
        cls,
        extension: ModuleType,
        output_specs: _OutputSpecs,
        *args: object,
        **kwargs: object,
    ) -> _ExtensionResult:
        """Invoke extension.call using the same explicit operation arguments."""

    @classmethod
    def prepare(
        cls, *args: object, **kwargs: object
    ) -> PreparedExtensionCall[_OutputSpecs, _ExtensionResult]:
        output_specs = cls.expected_output_specs(*args, **kwargs)
        defines = normalize_defines(cls.make_defines(*args, **kwargs))
        return PreparedExtensionCall(
            cls, defines, output_specs, args, tuple(kwargs.items())
        )

    @classmethod
    def get_loaded_module(cls, *args: object, **kwargs: object) -> ModuleType:
        defines = normalize_defines(cls.make_defines(*args, **kwargs))
        return MOJO_EXTENSION_LOADER.load_canonical(cls.MOJO_FILE, defines)

    @classmethod
    def execute(cls, *args: object, **kwargs: object) -> _ExtensionResult:
        return cls.prepare(*args, **kwargs).execute()


class MojoFileExtension(MojoExtension[object, object]):
    """Reusable stateless descriptor for a Mojo file's raw call ABIs.

    Operation-specific Python code supplies the exact dtypes, flags, already
    inferred output metadata, and native argument tuple. This is the common
    path for kernels whose outputs are allocated before the extension call.
    """

    @classmethod
    def make_defines(
        cls,
        op: str,
        extension_args: tuple[object, ...],
        *,
        arg_dtypes: tuple[DType | str, ...],
        output_dtypes: tuple[DType | str, ...] = (),
        flags: Mapping[str, DefineValue] | None = None,
        result_specs: object = None,
    ) -> Mapping[str, DefineValue]:
        del extension_args, result_specs
        return exact_call_defines(
            op, arg_dtypes, output_dtypes=output_dtypes, flags=flags
        )

    @classmethod
    def expected_output_specs(
        cls,
        op: str,
        extension_args: tuple[object, ...],
        *,
        arg_dtypes: tuple[DType | str, ...],
        output_dtypes: tuple[DType | str, ...] = (),
        flags: Mapping[str, DefineValue] | None = None,
        result_specs: object = None,
    ) -> object:
        del op, extension_args, arg_dtypes, output_dtypes, flags
        return result_specs

    @classmethod
    def call_extension(
        cls,
        extension: ModuleType,
        output_specs: object,
        op: str,
        extension_args: tuple[object, ...],
        *,
        arg_dtypes: tuple[DType | str, ...],
        output_dtypes: tuple[DType | str, ...] = (),
        flags: Mapping[str, DefineValue] | None = None,
        result_specs: object = None,
    ) -> object:
        del (output_specs, op, arg_dtypes, output_dtypes, flags, result_specs)
        return extension.call(*extension_args)  # type: ignore[attr-defined, no-any-return]

    @classmethod
    def invoke(
        cls,
        op: str,
        extension_args: tuple[object, ...],
        *,
        arg_dtypes: tuple[DType | str, ...],
        output_dtypes: tuple[DType | str, ...] = (),
        flags: Mapping[str, DefineValue] | None = None,
        result_specs: object = None,
    ) -> object:
        """Compile/load this exact variant, preserving the launch FIFO."""
        prepared = cls.prepare(
            op,
            extension_args,
            arg_dtypes=arg_dtypes,
            output_dtypes=output_dtypes,
            flags=flags,
            result_specs=result_specs,
        )
        if call_queue.enabled():
            prepared.enqueue_into(extension_args)
            return None
        return prepared.execute()


MOJO_EXTENSION_LOADER = MojoExtensionLoader()


def _pool_size() -> int:
    """Concurrent `mojo build` subprocesses. Each build peaks around 4.5 GB
    RSS and uses ~2.5-3 cores, so cap by available RAM (5 GiB per slot with
    headroom) and by cores; never fewer than 1, never more than 16."""
    mem_gib = 8.0
    try:
        with open("/proc/meminfo") as f:
            for line in f:
                if line.startswith("MemAvailable"):
                    mem_gib = int(line.split()[1]) / (1024 * 1024)
                    break
    except OSError:
        pass
    by_mem = int(mem_gib // 5)
    by_cpu = (os.cpu_count() or 4) // 3
    return max(1, min(by_mem, by_cpu, 16))


_ASYNC_BUILD_SLOTS = threading.Semaphore(_pool_size())


def __getattr__(name: str) -> object:
    if name == "tensor_holder":
        holder = _ensure_tensor_holder()
        globals()[name] = holder
        return holder
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")


_CTX_PTR_CACHE: dict[driver.Device, int] = {}


def _ctx_ptr(device: driver.Device) -> int:
    ptr = _CTX_PTR_CACHE.get(device)
    if ptr is None:
        ptr = device._device_context_ptr()
        _CTX_PTR_CACHE[device] = ptr
    return ptr
