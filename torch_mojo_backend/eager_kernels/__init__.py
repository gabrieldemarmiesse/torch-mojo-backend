"""Fast eager-mode kernels for mojo_device, compiled as CPython extensions.

The `.mojo` modules in this package are built on demand with
`mojo build --emit shared-lib` into per-op units: every gated .so compiles
exactly one extension entry point named `call`. Canonical compiler defines
such as `OP`, `DTYPE_ARG_0`, and `DTYPE_OUT` select the operation, ordered
argument dtypes, and compile-time flags for one GPU-call family. Shapes and
strides are runtime arguments and never enter the specialization key.

A specialization is identified by the defines Python sends, verbatim, plus
the hash of the source closure they are compiled against. The key is
deliberately not narrowed to the defines the sources are believed to read:
deciding that required parsing Mojo with regexes, which had to be taught
every new gate form and silently reverted to the unnarrowed key whenever it
could not (see git history). Sending a define no `comptime` gate consumes
can therefore fork a second, byte-identical `mojo build` -- a first-use
compile, never a wrong answer -- and the fix is to stop emitting the dead
define at its call site, where the fact is known, rather than to infer it
here.

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
import time
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


# Part of every extension cache key. Bump whenever the loader changes what a
# cached .so means without necessarily changing a Mojo source file — the
# current value covers one `call` entry point per .so, keeping each source's
# original PyInit_<module> symbol, and a specialization key restricted to the
# defines the Mojo sources read.
_VARIANT_CACHE_ABI = b"stateless-mojo-extension-v6"

# Loaded .so modules are registered under this prefix rather than under
# `torch_mojo_backend.eager_kernels`, whose `<op>_ops` names belong to the
# Python packages holding the descriptor classes: CPython inserts
# single-phase extension modules into sys.modules under the requested name,
# so sharing a name would replace the package with the native module.
_VARIANT_NAMESPACE = "torch_mojo_backend._mojo_kernels"


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

# Compiled once: these run on every specialization key built for a launch.
_DEFINE_NAME_RE = re.compile(r"[A-Z][A-Z0-9_]*")
_DTYPE_NAME_RE = re.compile(r"[A-Za-z][A-Za-z0-9_]*")


DefineValue = bool | int | str
CanonicalDefines = tuple[tuple[str, str], ...]

_DTYPE_NAMES: dict[DType, str] = {dtype: dtype.name for dtype in DType}


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
        if not _DEFINE_NAME_RE.fullmatch(name):
            raise ValueError(f"invalid compiler define name {name!r}")
        normalized.append((name, _normalize_define_value(value)))
    normalized.sort()
    return tuple(normalized)


def defines_cache_string(defines: Mapping[str, DefineValue]) -> str:
    """Unambiguous, stable serialization used to identify a specialization."""
    return json.dumps(normalize_defines(defines), separators=(",", ":"))


# Substrings that identify device-memory exhaustion across backends. Mojo
# reports launch and allocation failures as generic exceptions, so the
# message text is the only portable signal.
_DEVICE_OOM_MARKERS = (
    "cuda_error_out_of_memory",
    "hiperroroutofmemory",
    "out of memory",
    "failed to allocate device memory",
    "halerror (code = -13",
)


def is_device_oom(exc: BaseException) -> bool:
    """Whether an exception reports device-memory exhaustion (any backend)."""
    folded = str(exc).casefold()
    return any(marker in folded for marker in _DEVICE_OOM_MARKERS)


_CALL_DEFINES_CACHE: dict[tuple[object, ...], CanonicalDefines] = {}


def _build_call_defines(
    op: str,
    arg_dtypes: tuple[DType | str, ...],
    output_dtypes: tuple[DType | str, ...],
    flag_items: tuple[tuple[str, DefineValue], ...],
) -> CanonicalDefines:
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
    for name, value in flag_items:
        define_name = name.upper()
        if define_name in values:
            raise ValueError(f"duplicate specialization define {define_name!r}")
        values[define_name] = value
    # Validate here so malformed values fail while the call is prepared, not
    # in a background compiler thread.
    return normalize_defines(values)


def _canonical_call_defines(
    op: str,
    arg_dtypes: tuple[DType | str, ...],
    output_dtypes: tuple[DType | str, ...],
    flag_items: tuple[tuple[str, DefineValue], ...],
) -> CanonicalDefines:
    """Memoized canonical key for one exact call.

    A launch repeats the same (op, dtypes, flags) combination for the whole
    life of the process, so the regex validation and the sort below run once
    per specialization instead of once per kernel call.
    """
    key = (op, arg_dtypes, output_dtypes, flag_items)
    try:
        cached = _CALL_DEFINES_CACHE.get(key)
    except TypeError:
        # An unhashable define value: skip the memo so the build below
        # rejects it with the documented TypeError.
        return _build_call_defines(op, arg_dtypes, output_dtypes, flag_items)
    if cached is None:
        cached = _build_call_defines(op, arg_dtypes, output_dtypes, flag_items)
        _CALL_DEFINES_CACHE[key] = cached
    return cached


def exact_call_defines(
    op: str,
    arg_dtypes: tuple[DType | str, ...],
    *,
    output_dtypes: tuple[DType | str, ...] = (),
    flags: Mapping[str, DefineValue] | None = None,
) -> dict[str, DefineValue]:
    """Build the shape-independent compiler definitions for one exact call."""
    return dict(
        _canonical_call_defines(
            op,
            tuple(arg_dtypes),
            tuple(output_dtypes),
            tuple(flags.items()) if flags else (),
        )
    )


def _canonical_cache_string(defines: CanonicalDefines) -> str:
    return json.dumps(defines, separators=(",", ":"))


def _dtype_name(dtype: DType | str) -> str:
    if isinstance(dtype, DType):
        return _DTYPE_NAMES[dtype]  # enum member names are valid by construction
    if not _DTYPE_NAME_RE.fullmatch(dtype):
        raise ValueError(f"invalid specialization dtype {dtype!r}")
    return dtype


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


SourceStamp = tuple[str, int, int]

_CLOSURE_CACHE: dict[Path, tuple[tuple[Path, ...], tuple[SourceStamp, ...]]] = {}
_SOURCE_HASH_CACHE: dict[tuple[object, ...], str] = {}


def _source_stamps(files: tuple[Path, ...]) -> tuple[SourceStamp, ...]:
    """Cheap change detector for a dependency closure: one stat per file."""
    stamps: list[SourceStamp] = []
    for path in files:
        try:
            info = path.stat()
        except OSError:
            stamps.append((str(path), -1, -1))
        else:
            stamps.append((str(path), info.st_mtime_ns, info.st_size))
    return tuple(stamps)


def _closure_of(source: Path) -> tuple[tuple[Path, ...], tuple[SourceStamp, ...]]:
    """The dependency closure of an already resolved source, cached per path.

    Hashing reads every file in the closure, which costs milliseconds per
    source; the cached list is reused only while
    every file in it still has the same size and mtime, so editing any of
    them — including editing one to add an import — invalidates the entry.
    An edit that preserves both size and mtime is invisible, the same
    trade-off every build cache makes.
    """
    cached = _CLOSURE_CACHE.get(source)
    if cached is not None:
        files, stamps = cached
        if _source_stamps(files) == stamps:
            return files, stamps
    files = tuple(_dep_closure(source))
    stamps = _source_stamps(files)
    _CLOSURE_CACHE[source] = (files, stamps)
    return files, stamps


def _source_hash(source: Path) -> str:
    resolved = source.resolve()
    files, stamps = _closure_of(resolved)
    key = (resolved, _PACKAGE_DIR, _VARIANT_CACHE_ABI, _TOOLCHAIN_IDENTITY, stamps)
    cached = _SOURCE_HASH_CACHE.get(key)
    if cached is not None:
        return cached
    package_dir = _PACKAGE_DIR.resolve()
    hasher = hashlib.sha256()
    hasher.update(_VARIANT_CACHE_ABI)
    hasher.update(_TOOLCHAIN_IDENTITY)
    for path in files:
        try:
            relative_path = path.relative_to(package_dir)
        except ValueError:
            relative_path = path.relative_to(resolved.parent)
        hasher.update(str(relative_path).encode())
        hasher.update(path.read_bytes())
    digest = hasher.hexdigest()[:16]
    _SOURCE_HASH_CACHE[key] = digest
    return digest


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


_BUILD_NOTICE_LOCK = threading.Lock()
_BUILD_NOTICE_SHOWN = False


def _trace(message: str) -> None:
    """Timestamped build tracing, on by default; TORCH_MOJO_BACKEND_TRACE=0
    silences it."""
    if os.environ.get("TORCH_MOJO_BACKEND_TRACE", "1") != "0":
        sys.stderr.write(f"[TRACE] {message} t={time.monotonic():.2f}\n")


def _announce_build() -> None:
    """One notice per process, not one per specialization.

    A cold process compiles dozens of variants across up to `_pool_size()`
    threads; naming each one buries the only thing a user needs to know,
    which is that the wait happens once and is cached. The per-variant
    detail is the [TRACE] lines (on by default). A single write keeps concurrent
    builder threads from splicing their lines together.
    """
    global _BUILD_NOTICE_SHOWN
    with _BUILD_NOTICE_LOCK:
        if _BUILD_NOTICE_SHOWN:
            return
        _BUILD_NOTICE_SHOWN = True
    sys.stderr.write(
        "torch-mojo-backend: compiling eager Mojo kernels on demand "
        "(first use only, cached afterwards in __mojocache__)...\n"
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
        _announce_build()
        _trace(f"building {src.stem}.{scope}")
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
            _trace(f"built {src.stem}.{scope}")
            if proc.returncode != 0:
                raise ImportError(
                    f"mojo build failed for {src.stem} "
                    f"({_defines_tag(defines)}):\n{proc.stderr}"
                )
            os.replace(tmp, out)
        finally:
            tmp.unlink(missing_ok=True)
        return out


def _variant_module_name(src: Path, defines: CanonicalDefines | None) -> str:
    """sys.modules name for one loaded .so.

    CPython derives the init symbol from the last dotted component only, so
    the stem must stay last; the specialization tag in front gives every
    variant of a source its own entry instead of each one replacing the
    previous, and `_VARIANT_NAMESPACE` keeps all of them clear of the
    importable `<op>_ops` packages.
    """
    return f"{_VARIANT_NAMESPACE}.{_defines_tag(defines)}.{src.stem}"


def _load_extension(module_name: str, so_path: Path) -> ModuleType:
    loader = importlib.machinery.ExtensionFileLoader(module_name, str(so_path))
    spec = importlib.util.spec_from_file_location(
        module_name, str(so_path), loader=loader
    )
    module = importlib.util.module_from_spec(spec)
    loader.exec_module(module)
    return module


_TENSOR_HOLDER_LOCK = threading.Lock()
_TENSOR_HOLDER: list[ModuleType | BaseException] = []


def _ensure_tensor_holder() -> ModuleType:
    """Register the process-wide TensorHolder/TensorSpec Python types.

    Deliberately ungated and loaded exactly once: a second `add_type` of the
    same name aborts the process, and the file lock in `_build_extension`
    guards the build but not the dlopen.
    """
    with _TENSOR_HOLDER_LOCK:
        if _TENSOR_HOLDER:
            loaded = _TENSOR_HOLDER[0]
            if isinstance(loaded, BaseException):
                raise loaded
            return loaded
        source = _PACKAGE_DIR / "tensor_holder.mojo"
        try:
            module = _load_extension(
                _variant_module_name(source, None), _build_extension(source, None)
            )
        except BaseException as exc:
            _TENSOR_HOLDER.append(exc)
            raise
        _TENSOR_HOLDER.append(module)
        return module


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
                module = _load_extension(
                    _variant_module_name(self.mojo_file, self.defines), so_path
                )
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
        """The loaded native module, or None while it is still building.
        `call_queue` reads this to decide whether an item can launch."""
        return self.module

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
        self._requests: dict[tuple[Path, CanonicalDefines], _DefinedUnit] = {}

    def _unit(self, mojo_file: Path, defines: CanonicalDefines) -> _DefinedUnit:
        """The unit for one requested specialization.

        This is on the path of every kernel launch, so a repeated request is
        one dict lookup and nothing else. Everything that turns a request
        into a build identity — resolving the source path and reading its
        dependency closure — happens once per distinct (file, defines) pair
        on the miss path below.
        """
        unit = self._requests.get((mojo_file, defines))
        if unit is None:
            unit = self._resolve_unit(mojo_file, defines)
        return unit

    def _resolve_unit(self, mojo_file: Path, defines: CanonicalDefines) -> _DefinedUnit:
        source = _resolve_mojo_file(mojo_file).resolve()
        key = (source, defines)
        with self._lock:
            unit = self._units.get(key)
            if unit is None:
                unit = self._units[key] = _DefinedUnit(*key)
            self._requests[(mojo_file, defines)] = unit
            return unit

    def load_canonical(self, mojo_file: Path, defines: CanonicalDefines) -> ModuleType:
        return self._unit(mojo_file, defines).load_blocking()

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

    def get_loaded_module(
        self, loader: MojoExtensionLoader | None = None
    ) -> ModuleType:
        selected_loader = loader or MOJO_EXTENSION_LOADER
        return selected_loader.load_canonical(self.extension.MOJO_FILE, self.defines)

    def execute(self, loader: MojoExtensionLoader | None = None) -> _ExtensionResult:
        module = self.get_loaded_module(loader)
        return self.extension.call_extension(module, self.output_specs, *self.args)

    def enqueue_into(
        self,
        extension_args: tuple[object, ...],
        keepalive: tuple[object, ...],
        loader: MojoExtensionLoader | None = None,
    ) -> None:
        """Queue a non-returning `call(..., out)` into preallocated outputs.

        `keepalive` names the objects whose buffers `extension_args`'
        raw pointers reference; the queued item retains them until it
        launches (queue rule 3)."""
        selected_loader = loader or MOJO_EXTENSION_LOADER
        unit = selected_loader.unit_canonical(self.extension.MOJO_FILE, self.defines)
        call_queue.kernel_call_into(unit, extension_args, keepalive)


class MojoExtension(ABC, Generic[_OutputSpecs, _ExtensionResult]):
    """Stateless operation descriptor for one Mojo source specialization."""

    __slots__ = ()
    MOJO_FILE: ClassVar[Path]

    def __new__(cls) -> "MojoExtension":
        raise TypeError(
            f"{cls.__name__} is a stateless descriptor and cannot be instantiated"
        )

    @classmethod
    @abstractmethod
    def make_defines(cls, *args: object, **kwargs: object) -> Mapping[str, DefineValue]:
        """Capture every compile-time value; never include shapes or strides."""

    @classmethod
    def make_canonical_defines(
        cls, *args: object, **kwargs: object
    ) -> CanonicalDefines:
        """The specialization key, canonical and validated exactly once.

        Descriptors whose `make_defines` already returns a validated mapping
        override this to hand the canonical form straight over instead of
        having it re-validated per launch.
        """
        return normalize_defines(cls.make_defines(*args, **kwargs))

    @classmethod
    @abstractmethod
    def expected_output_specs(cls, *args: object, **kwargs: object) -> _OutputSpecs:
        """Infer output metadata without compiling or executing the extension."""

    @classmethod
    @abstractmethod
    def extension_args(
        cls, out: object, *args: object, **kwargs: object
    ) -> tuple[object, ...]:
        """The native argument tuple for `call`, given the allocated outputs.

        This is the per-operation contract the queue depends on: a queued
        launch is serialized here, from Python-side allocated outputs, and
        never goes through `call_extension`.
        """

    @classmethod
    def call_extension(
        cls,
        extension: ModuleType,
        output_specs: _OutputSpecs,
        *args: object,
        **kwargs: object,
    ) -> _ExtensionResult:
        """Allocate the outputs, invoke ``extension.call``, return them.

        The synchronous twin of the queued path: the same
        ``allocate_outputs`` allocation and ``extension_args`` serialization,
        executed inline. Descriptors with a different call ABI
        (``MojoFileExtension``) override it.
        """
        out = cls.allocate_outputs(output_specs)
        extension.call(*cls.extension_args(out, *args, **kwargs))  # type: ignore[attr-defined]
        return out

    @classmethod
    def allocate_outputs(cls, output_specs: _OutputSpecs) -> _ExtensionResult:
        """Allocate this descriptor's outputs from their inferred specs.

        The default covers every single-output family: one tensor from one
        spec. Descriptors with several outputs override this with the same
        allocation their ``call_extension`` performs.
        """
        return _allocate_single_output(output_specs)  # type: ignore[return-value]

    @classmethod
    def prepare(
        cls, *args: object, **kwargs: object
    ) -> PreparedExtensionCall[_OutputSpecs, _ExtensionResult]:
        output_specs = cls.expected_output_specs(*args, **kwargs)
        defines = cls.make_canonical_defines(*args, **kwargs)
        # Keyword arguments (MojoFileExtension's dtype/flag declarations) are
        # consumed above, by spec inference and the defines; the prepared call
        # itself carries only the positional runtime arguments.
        return PreparedExtensionCall(cls, defines, output_specs, args)


def _allocate_single_output(spec: object) -> object:
    """``output_specs._allocate_output_spec``, resolved on first use.

    ``output_specs`` imports this package, so it cannot be imported at module
    scope here; rebinding the global on the first allocation leaves the
    steady state at one plain lookup (the same trick as ``output_specs``'s
    own ``_alloc``).
    """
    global _allocate_single_output
    from torch_mojo_backend.eager_kernels.output_specs import _allocate_output_spec

    _allocate_single_output = _allocate_output_spec
    return _allocate_output_spec(spec)


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
    ) -> Mapping[str, DefineValue]:
        del extension_args
        return exact_call_defines(
            op, arg_dtypes, output_dtypes=output_dtypes, flags=flags
        )

    @classmethod
    def make_canonical_defines(
        cls,
        op: str,
        extension_args: tuple[object, ...],
        *,
        arg_dtypes: tuple[DType | str, ...],
        output_dtypes: tuple[DType | str, ...] = (),
        flags: Mapping[str, DefineValue] | None = None,
    ) -> CanonicalDefines:
        del extension_args
        return _canonical_call_defines(
            op,
            tuple(arg_dtypes),
            tuple(output_dtypes),
            tuple(flags.items()) if flags else (),
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
    ) -> object:
        del op, extension_args, arg_dtypes, output_dtypes, flags
        return None  # this family's outputs are allocated by its caller

    @classmethod
    def extension_args(
        cls,
        out: object,
        op: str,
        extension_args: tuple[object, ...],
        *,
        arg_dtypes: tuple[DType | str, ...],
        output_dtypes: tuple[DType | str, ...] = (),
        flags: Mapping[str, DefineValue] | None = None,
    ) -> tuple[object, ...]:
        del out, op, arg_dtypes, output_dtypes, flags
        return extension_args  # already built by the caller, outputs included

    @classmethod
    def call_extension(
        cls,
        extension: ModuleType,
        output_specs: object,
        op: str,
        extension_args: tuple[object, ...],
        *,
        arg_dtypes: tuple[DType | str, ...] = (),
        output_dtypes: tuple[DType | str, ...] = (),
        flags: Mapping[str, DefineValue] | None = None,
    ) -> object:
        # The dtype/flag declarations select the variant at prepare() time;
        # execution needs only the runtime arguments, so they default empty.
        del output_specs, op, arg_dtypes, output_dtypes, flags
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
        keepalive: tuple[object, ...],
    ) -> object:
        """Compile/load this exact variant, preserving the launch FIFO.

        `keepalive` names the tensors whose buffers `extension_args`' raw
        specs and pointers reference (queue rule 3). Required and
        keyword-only so no call site can forget it."""
        prepared = cls.prepare(
            op,
            extension_args,
            arg_dtypes=arg_dtypes,
            output_dtypes=output_dtypes,
            flags=flags,
        )
        if call_queue.enabled():
            prepared.enqueue_into(extension_args, keepalive)
            return None
        return prepared.execute()


MOJO_EXTENSION_LOADER = MojoExtensionLoader()


def _available_memory_gib() -> float | None:
    """Memory the build pool may assume, or None when the host cannot say.

    Linux answers with MemAvailable; macOS has no procfs, so fall back to
    total physical memory through sysconf, which overestimates but still
    scales with the machine. Returning None rather than a fixed guess is what
    lets the caller fall back to the core count instead of silently
    serializing every build on hosts neither source covers.
    """
    try:
        with open("/proc/meminfo") as meminfo:
            for line in meminfo:
                if line.startswith("MemAvailable"):
                    return int(line.split()[1]) / (1024 * 1024)
    except OSError:
        pass
    try:
        return os.sysconf("SC_PAGE_SIZE") * os.sysconf("SC_PHYS_PAGES") / 2**30
    except (OSError, ValueError, AttributeError):
        return None


def _pool_size() -> int:
    """Concurrent `mojo build` subprocesses. Each build peaks around 4.5 GB
    RSS and uses ~2.5-3 cores, so cap by available RAM (5 GiB per slot with
    headroom) and by cores; never fewer than 1, never more than 16."""
    by_cpu = (os.cpu_count() or 4) // 3
    memory_gib = _available_memory_gib()
    by_mem = by_cpu if memory_gib is None else int(memory_gib // 5)
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
