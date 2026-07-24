"""Fast eager-mode kernels for mojo_device, compiled as CPython extensions.

The `.mojo` modules in this package are built on demand with
`mojo build --emit shared-lib` into per-op units: every gated .so compiles
EXACTLY ONE extension entry point (`-D TMB_OPS=<op>`) with a dtype set
(`-D TMB_DTYPES=<csv>`). One op per .so means every unit builds
independently — full build parallelism across ops of the same module, no
escalation chains, and op-granular caching (a new workload rebuilds only
the ops it newly demands, never a whole module variant).

Demand discovery is transparent to callers: `eager_kernels.<module>` returns
a proxy whose attribute lookup builds/loads the unit for that op on first
touch, and whose call wrappers rebuild the unit with the full dtype set when
a Mojo kernel reports an unsupported dtype, then retry. Demanded ops/dtypes
are persisted in `__mojocache__/demand_profile.json` (telemetry, and the
default dtype seed for first builds).

An op call here is one CPython extension call that receives raw data
pointers (from `TorchMojoTensor._ptr`) plus sizes/dtypes as plain ints, and
enqueues a kernel on MAX's own DeviceContext, so it stays correctly ordered
with every other MAX driver operation on that device.

`tensor_holder` is exempt from gating: it registers the process-wide
`TensorHolder`/`TensorSpec` Python types (a duplicate `add_type` aborts the
process), so it is always built complete and never escalated — which also
means direct references to its functions stay valid forever.
"""

import atexit
import fcntl
import hashlib
import importlib.machinery
import importlib.util
import json
import os
import re
import subprocess
import sys
import threading
from pathlib import Path
from types import ModuleType

from max import driver

_PACKAGE_DIR = Path(__file__).parent
_CACHE_DIR = _PACKAGE_DIR / "__mojocache__"
_PROFILE_PATH = _CACHE_DIR / "demand_profile.json"


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

_MOJO_MODULES = (
    "tensor_holder",
    "activation_backward_ops",
    "activation_forward_ops",
    "dropout_ops",
    "embedding_backward_ops",
    "elementwise_ops",
    "nn_ops",
    "data_movement_ops",
    "logic_ops",
    "matmul_ops",
    "bf16_matmul_ops",
    "tf32_matmul_ops",
    "conv_ops",
    "reduction_ops",
    "loss_ops",
    "softmax_backward_ops",
    "normalization_forward_ops",
    "normalization_backward_ops",
    "optimizer_ops",
    "sdpa_backward_ops",
)
# Owns the process-wide Python type registry: always full, never escalated.
_FULL_MODULES = frozenset({"tensor_holder"})
# Dtypes a first-touch variant compiles when the profile has none recorded:
# what every torch workload touches (bf16/f32 compute, i64 indices, masks).
_DEFAULT_DTYPES = ("bfloat16", "bool", "float32", "int64", "uint8")

_IMPORT_RE = re.compile(r"^(?:from|import)\s+(\w+)", re.M)
_REGISTRATION_RE = re.compile(
    r"\.def_(?:py_c_)?function\((?:[^\"]*?)\"([A-Za-z0-9_]+)\"", re.S
)


def _registered_ops(name: str) -> frozenset[str]:
    """Entry-point names a module's PyInit can register (parsed from source),
    so a probe for a nonexistent attribute fails fast instead of triggering a
    futile variant build."""
    cached = _REGISTERED_OPS_CACHE.get(name)
    if cached is None:
        source = (_PACKAGE_DIR / f"{name}.mojo").read_text()
        cached = frozenset(_REGISTRATION_RE.findall(source))
        _REGISTERED_OPS_CACHE[name] = cached
    return cached


_REGISTERED_OPS_CACHE: dict[str, frozenset[str]] = {}


def _dep_closure(name: str) -> list[Path]:
    """The module's source plus every sibling .mojo (and op_utils) it can
    reach through imports — the correct per-module cache-key surface."""
    seen: set[str] = set()
    todo = [name]
    files: list[Path] = []
    while todo:
        stem = todo.pop()
        if stem in seen:
            continue
        seen.add(stem)
        path = _PACKAGE_DIR / f"{stem}.mojo"
        if not path.is_file():
            continue
        files.append(path)
        for dep in _IMPORT_RE.findall(path.read_text()):
            if dep not in seen and (_PACKAGE_DIR / f"{dep}.mojo").is_file():
                todo.append(dep)
    files.extend(sorted((_PACKAGE_DIR / "op_utils").rglob("*.mojo")))
    return files


def _module_hash(name: str) -> str:
    hasher = hashlib.sha256()
    for path in _dep_closure(name):
        hasher.update(path.name.encode())
        hasher.update(path.read_bytes())
    return hasher.hexdigest()[:16]


def _load_profile() -> dict:
    try:
        return json.loads(_PROFILE_PATH.read_text())
    except (OSError, ValueError):
        return {}


def _save_profile() -> None:
    _CACHE_DIR.mkdir(exist_ok=True)
    with open(_CACHE_DIR / ".profile.lock", "w") as lock:
        fcntl.flock(lock, fcntl.LOCK_EX)
        merged = _load_profile()
        for name, state in _STATES.items():
            if not state.demanded_ops:
                continue
            entry = merged.setdefault(name, {"ops": [], "dtypes": []})
            entry["ops"] = sorted(set(entry["ops"]) | state.demanded_ops)
            dtypes: set[str] = set()
            for unit in state.units.values():
                dtypes |= set(unit.dtypes or ())
            entry["dtypes"] = sorted(set(entry["dtypes"]) | dtypes)
        _PROFILE_PATH.write_text(json.dumps(merged, indent=1, sort_keys=True))


def _variant_tag(ops: frozenset[str] | None, dtypes: frozenset[str] | None) -> str:
    if ops is None and dtypes is None:
        return "full"
    key = (
        f"{sorted(ops) if ops is not None else 'all'}"
        f"|{sorted(dtypes) if dtypes is not None else 'all'}"
    )
    return hashlib.sha256(key.encode()).hexdigest()[:10]


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


def _variant_cmd(
    name: str,
    src: Path,
    ops: frozenset[str] | None,
    dtypes: frozenset[str] | None,
    out: Path,
) -> list[str]:
    cmd = [str(_find_mojo()), "build", str(src), "--emit", "shared-lib"]
    if ops is not None:
        cmd += ["-D", f"TMB_OPS={','.join(sorted(ops)) or '__none__'}"]
    if dtypes is not None:
        cmd += ["-D", f"TMB_DTYPES={','.join(sorted(dtypes))}"]
    return cmd + ["-o", str(out)]


def _unit_symbol(name: str, ops: frozenset[str] | None, dtypes: frozenset[str] | None) -> str:
    """Module name (= PyInit suffix) for a build. Full builds keep the
    source's own symbol; gated per-op builds get a deterministic tag-based
    mangle so every unit coexists in one process AND a cached .so is
    loadable under the same name by any later process."""
    if ops is None and dtypes is None:
        return name
    return f"_tmbv_{name}_{_variant_tag(ops, dtypes)}"


def _variant_path(
    name: str,
    ops: frozenset[str] | None,
    dtypes: frozenset[str] | None,
) -> Path:
    tag = _variant_tag(ops, dtypes)
    return _CACHE_DIR / (f"{name}.{tag}.hash-{_module_hash(name)}.so")


def _build_variant(
    name: str,
    ops: frozenset[str] | None,
    dtypes: frozenset[str] | None,
) -> Path:
    """Compile one unit .so (blocking); returns the cache path.

    A gated build compiles exactly ONE op — multi-op .so files are not
    representable in this loader. `ops is None` (with dtypes None) is the
    full build, reserved for the type-registry module (tensor_holder).
    """
    if ops is not None and len(ops) != 1:
        raise ValueError(
            f"per-op loader: a gated build must contain exactly one op, "
            f"got {sorted(ops)!r} for {name}"
        )
    out = _variant_path(name, ops, dtypes)
    if out.is_file():
        return out
    _CACHE_DIR.mkdir(exist_ok=True)
    with open(_CACHE_DIR / f".{out.stem}.lock", "w") as lock:
        fcntl.flock(lock, fcntl.LOCK_EX)
        if out.is_file():
            return out
        src = _PACKAGE_DIR / f"{name}.mojo"
        scratch: Path | None = None
        symbol = _unit_symbol(name, ops, dtypes)
        if symbol != name:
            scratch = _PACKAGE_DIR / f"{symbol}.mojo"
            scratch.write_text(
                src.read_text().replace(f"def PyInit_{name}", f"def PyInit_{symbol}")
            )
            src = scratch
        scope = "full" if ops is None else next(iter(ops))
        import time as _time

        print(
            f"torch-mojo-backend: compiling {name}.{scope} on demand..."
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
                _variant_cmd(name, src, ops, dtypes, tmp),
                capture_output=True,
                text=True,
                env=_build_env(),
            )
            if os.environ.get("TMB_TRACE"):
                print(
                    f"[TRACE] built {name}.{scope} t={_time.monotonic():.2f}",
                    file=sys.stderr,
                    flush=True,
                )
            if proc.returncode != 0:
                raise ImportError(
                    f"mojo build failed for {name} "
                    f"({_variant_tag(ops, dtypes)}):\n{proc.stderr}"
                )
            os.replace(tmp, out)
        finally:
            tmp.unlink(missing_ok=True)
            if scratch is not None:
                scratch.unlink(missing_ok=True)
        return out


def _import_mojo_module(
    name: str,
    ops: frozenset[str] | None = None,
    dtypes: frozenset[str] | None = None,
) -> ModuleType:
    """Compatibility seam kept from the previous loaders: the single point a
    unit is built (if needed) and loaded. Tests patch this to simulate
    compiler failure / unavailable extensions."""
    if name != "tensor_holder":
        # tensor_holder registers the process-wide TensorHolder/TensorSpec
        # Python types every other module's spec ops consume; it must be
        # loaded and finalized before any kernel module.
        _STATES["tensor_holder"].ensure_loaded(None)
    so_path = _build_variant(name, ops, dtypes)
    return _load_extension(f"{__name__}.{_unit_symbol(name, ops, dtypes)}", so_path)


def _load_extension(module_name: str, so_path: Path) -> ModuleType:
    loader = importlib.machinery.ExtensionFileLoader(module_name, str(so_path))
    spec = importlib.util.spec_from_file_location(
        module_name, str(so_path), loader=loader
    )
    module = importlib.util.module_from_spec(spec)
    loader.exec_module(module)
    return module


class KernelPending(BaseException):
    """The kernel the current op needs is compiling asynchronously.

    Raised (instead of blocking) when a variant miss happens under
    ``__torch_dispatch__`` and the deferred-execution layer is available.
    Deliberately a BaseException so intermediate ``except Exception`` blocks
    in the dispatch stack cannot swallow it; the deferred layer catches it
    at the dispatch boundary (main thread) or in the launcher's retry loop.
    """

    def __init__(self, state: "_ModuleState", job: "_AsyncOpJob") -> None:
        super().__init__(f"a kernel unit of {state.name} is compiling")
        self.state = state
        self.job = job


_DISPATCH_TLS = threading.local()


def _in_torch_dispatch() -> bool:
    return getattr(_DISPATCH_TLS, "depth", 0) > 0


def _may_raise_pending() -> bool:
    """Whether a compile miss under dispatch may raise KernelPending for the
    deferred-execution layer. Under the unit-test suite kernel loads block
    inline instead: a KernelPending retry restarts the WHOLE aten op, so a
    multi-kernel op with several cold units would re-run its side work
    (duplicate casts/allocations) once per unit — test spies see those. The
    force knob re-enables deferral for its dedicated harnesses."""
    from torch_mojo_backend.is_running_tests import IS_RUNNING_TESTS

    return not IS_RUNNING_TESTS or bool(os.environ.get("TMB_FORCE_DEFER"))


class _dispatch_scope:
    """Marks 'this extension call came through __torch_dispatch__', which is
    the only context where a variant miss may raise KernelPending instead of
    compiling synchronously (direct callers, e.g. tests, still block)."""

    def __enter__(self) -> None:
        _DISPATCH_TLS.depth = getattr(_DISPATCH_TLS, "depth", 0) + 1

    def __exit__(self, *exc: object) -> None:
        _DISPATCH_TLS.depth -= 1


class _AsyncOpJob:
    """One background build of one (module, op) unit. Every unit builds
    independently through the slot pool — full parallelism across ops."""

    def __init__(self, unit: "_OpUnit") -> None:
        self.unit = unit
        self.done = threading.Event()
        self.error: BaseException | None = None

    def run(self) -> None:
        unit = self.unit
        try:
            with _ASYNC_BUILD_SLOTS:
                with unit.lock:
                    dtypes = None if unit.want_all_dtypes else unit.dtypes
                ext = _import_mojo_module(
                    unit.state.name, frozenset({unit.op}), dtypes
                )
                with unit.lock:
                    unit.dtypes = dtypes
                    unit.ext = ext
                    # Invalidate the proxy's cached resolution for this op
                    # so the next lookup binds the fresh extension.
                    _PROXIES[unit.state.name].__dict__.pop(unit.op, None)
        except BaseException as exc:  # surfaced to every waiter
            self.error = exc
        finally:
            with unit.lock:
                if unit.job is self:
                    unit.job = None
                more = unit.want_all_dtypes and unit.dtypes is not None
            self.done.set()
            if more and self.error is None:
                unit.request_async()  # dtype demand arrived mid-build

    def wait(self) -> None:
        self.done.wait()
        if self.error is not None:
            raise self.error


class _OpUnit:
    """Build/load bookkeeping for one (module, op) compilation unit."""

    def __init__(self, state: "_ModuleState", op: str) -> None:
        self.state = state
        self.op = op
        self.lock = threading.Lock()
        self.ext: ModuleType | None = None
        profile = _PROFILE.get(state.name, {})
        self.dtypes: frozenset[str] | None = frozenset(
            profile.get("dtypes", ()) or _DEFAULT_DTYPES
        )
        self.want_all_dtypes = False
        self.job: _AsyncOpJob | None = None

    def _satisfied(self) -> bool:
        return self.ext is not None and not (
            self.want_all_dtypes and self.dtypes is not None
        )

    def request_async(self, all_dtypes: bool = False) -> _AsyncOpJob:
        """Make sure a build covering the current demand is in flight."""
        with self.lock:
            if all_dtypes:
                self.want_all_dtypes = True
            job = self.job
            if job is None:
                if self._satisfied():  # already loaded: a completed no-op job
                    job = _AsyncOpJob(self)
                    job.done.set()
                    return job
                job = self.job = _AsyncOpJob(self)
                threading.Thread(
                    target=job.run,
                    name=f"tmb-build-{self.state.name}.{self.op}",
                    daemon=True,
                ).start()
            return job

    def load_blocking(self, all_dtypes: bool = False) -> ModuleType:
        """Build (if needed) and load this unit synchronously."""
        with self.lock:
            if all_dtypes:
                self.want_all_dtypes = True
            if self._satisfied():
                return self.ext
            dtypes = None if self.want_all_dtypes else self.dtypes
        ext = _import_mojo_module(self.state.name, frozenset({self.op}), dtypes)
        with self.lock:
            if not self._satisfied():
                self.dtypes = dtypes
                self.ext = ext
                _PROXIES[self.state.name].__dict__.pop(self.op, None)
            return self.ext


class _ModuleState:
    """Bookkeeping for one .mojo module: a unit per demanded op (gated
    modules) or one full extension (tensor_holder)."""

    def __init__(self, name: str) -> None:
        self.name = name
        self.lock = threading.Lock()
        self.module: ModuleType | None = None  # full modules only
        self.units: dict[str, _OpUnit] = {}
        self.demanded_ops: set[str] = set()

    def unit(self, op: str) -> _OpUnit:
        """Get or create the unit for `op` (validated against the source's
        registration list so probes fail fast instead of building)."""
        if op not in _registered_ops(self.name):
            raise AttributeError(f"module {self.name!r} has no entry point {op!r}")
        with self.lock:
            unit = self.units.get(op)
            if unit is None:
                unit = self.units[op] = _OpUnit(self, op)
            return unit

    def any_loaded_ext(self) -> ModuleType | None:
        with self.lock:
            for unit in self.units.values():
                if unit.ext is not None:
                    return unit.ext
        return None

    def ensure_loaded(self, first_op: str | None) -> ModuleType:
        """Full modules: build+load the complete extension (tensor_holder).
        Gated modules: resolve the unit for `first_op`."""
        if self.name in _FULL_MODULES:
            with self.lock:
                if self.module is None:
                    self.module = _import_mojo_module(self.name)
                return self.module
        if first_op is None:
            raise AttributeError(
                f"{self.name} is per-op loaded; an op name is required"
            )
        unit = self.unit(first_op)
        if unit.ext is None and _in_torch_dispatch() and _may_raise_pending():
            raise KernelPending(self, unit.request_async())
        return unit.load_blocking()


def _wrap_call(unit: _OpUnit, attr: str, fn: object) -> object:
    def call(*args: object, **kwargs: object) -> object:
        try:
            return fn(*args, **kwargs)
        except Exception as exc:  # Mojo errors surface as plain Exception
            if "unsupported dtype" not in str(exc) or unit.dtypes is None:
                raise
            if _in_torch_dispatch() and _may_raise_pending():
                raise KernelPending(
                    unit.state, unit.request_async(all_dtypes=True)
                ) from exc
            module = unit.load_blocking(all_dtypes=True)
            return getattr(module, attr)(*args, **kwargs)

    return call


class _ModuleProxy:
    """Stands in for one extension module; routes each attribute to its
    per-op unit, building units on demand."""

    def __init__(self, state: _ModuleState) -> None:
        self.__dict__["_state"] = state
        # Real module metadata: tests and tooling identify the extension by
        # its canonical module name, which the proxy stands in for.
        self.__dict__["__name__"] = f"{__name__}.{state.name}"

    def __getattr__(self, attr: str) -> object:
        state: _ModuleState = self.__dict__["_state"]
        if attr.startswith("__"):
            raise AttributeError(attr)
        if attr not in _registered_ops(state.name):
            # Not an entry point: serve module-level symbols from any loaded
            # unit (every unit carries the ungated module-level defs).
            ext = state.any_loaded_ext()
            if ext is not None and hasattr(ext, attr):
                value = getattr(ext, attr)
                self.__dict__[attr] = value
                return value
            raise AttributeError(
                f"module {state.name!r} has no entry point {attr!r}"
            )
        unit = state.unit(attr)
        if unit.ext is None:
            if _in_torch_dispatch() and _may_raise_pending():
                raise KernelPending(state, unit.request_async())
            unit.load_blocking()
        state.demanded_ops.add(attr)
        value = getattr(unit.ext, attr)
        if type(value).__name__ == "builtin_function_or_method":
            value = _wrap_call(unit, attr, value)
        self.__dict__[attr] = value  # later lookups skip __getattr__
        return value


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


_PROFILE = _load_profile()
_ASYNC_BUILD_SLOTS = threading.Semaphore(_pool_size())
_STATES: dict[str, _ModuleState] = {n: _ModuleState(n) for n in _MOJO_MODULES}
_PROXIES: dict[str, _ModuleProxy] = {
    n: _ModuleProxy(_STATES[n]) for n in _MOJO_MODULES if n not in _FULL_MODULES
}
atexit.register(_save_profile)


_CACHED_IN_DICT: set[str] = set()


def __getattr__(name: str) -> object:
    if name in _MOJO_MODULES:
        if name == "tensor_holder":
            holder = _STATES["tensor_holder"].ensure_loaded(None)
            globals()[name] = holder
            return holder
        proxy = _PROXIES[name]
        state = _STATES[name]
        if name in _CACHED_IN_DICT and any(
            u.ext is not None for u in state.units.values()
        ):
            # A previously *successful* resolution was explicitly deleted
            # from the package dict (tests use this to force a fresh
            # import): drop every loaded unit so resolution goes back
            # through _import_mojo_module.
            with state.lock:
                state.units.clear()
            proxy.__dict__.clear()
            proxy.__dict__["_state"] = state
            proxy.__dict__["__name__"] = f"{__name__}.{name}"
            _CACHED_IN_DICT.discard(name)
        # Resolution is always lazy: units are per-op, so nothing is built
        # until an attribute access names the op — build errors surface at
        # first attribute use, not at resolution. Caching the proxy is
        # therefore free in every context (it performs no work), and keeps
        # `eager_kernels.__dict__.get(<module>)` a valid presence probe.
        globals()[name] = proxy
        _CACHED_IN_DICT.add(name)
        return proxy
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")


_CTX_PTR_CACHE: dict[driver.Device, int] = {}


def _ctx_ptr(device: driver.Device) -> int:
    ptr = _CTX_PTR_CACHE.get(device)
    if ptr is None:
        ptr = device._device_context_ptr()
        _CTX_PTR_CACHE[device] = ptr
    return ptr
