"""Fast eager-mode kernels for mojo_device, compiled as CPython extensions.

The `.mojo` modules in this package are built on demand with
`mojo build --emit shared-lib` into *variants*: each variant compiles only
the extension entry points (`-D TMB_OPS=<csv>`) and dtypes
(`-D TMB_DTYPES=<csv>`) that the workload has actually demanded, which cuts
zero-cache cold starts by an order of magnitude (a gated-out op or dtype is
never elaborated, so its GPU kernels are never compiled).

Demand discovery is transparent to callers: `eager_kernels.<module>` returns
a proxy whose attribute lookup escalates to a wider variant when an op is
missing, and whose call wrappers escalate to the full dtype set when a Mojo
kernel reports an unsupported dtype, then retry. Demanded ops/dtypes are
persisted in `__mojocache__/demand_profile.json`; later cold starts compile
each module's profiled variant once, in parallel, in the background.

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
_MOJO_EXE = Path(sys.executable).parent / "mojo"

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
            entry["dtypes"] = sorted(set(entry["dtypes"]) | set(state.dtypes or ()))
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
    cmd = [str(_MOJO_EXE), "build", str(src), "--emit", "shared-lib"]
    if ops is not None:
        cmd += ["-D", f"TMB_OPS={','.join(sorted(ops)) or '__none__'}"]
    if dtypes is not None:
        cmd += ["-D", f"TMB_DTYPES={','.join(sorted(dtypes))}"]
    return cmd + ["-o", str(out)]


def _variant_path(
    name: str,
    ops: frozenset[str] | None,
    dtypes: frozenset[str] | None,
    generation: int,
) -> Path:
    tag = _variant_tag(ops, dtypes)
    return _CACHE_DIR / (f"{name}.{tag}.g{generation}.hash-{_module_hash(name)}.so")


def _build_variant(
    name: str,
    ops: frozenset[str] | None,
    dtypes: frozenset[str] | None,
    generation: int,
) -> Path:
    """Compile one variant .so (blocking); returns the cache path.

    generation 0 keeps the source's own PyInit symbol; later generations
    build from a copy with a renamed PyInit so multiple variants of the same
    module can coexist in one process.
    """
    out = _variant_path(name, ops, dtypes, generation)
    if out.is_file():
        return out
    _CACHE_DIR.mkdir(exist_ok=True)
    with open(_CACHE_DIR / f".{out.stem}.lock", "w") as lock:
        fcntl.flock(lock, fcntl.LOCK_EX)
        if out.is_file():
            return out
        src = _PACKAGE_DIR / f"{name}.mojo"
        scratch: Path | None = None
        if generation > 0:
            scratch = _PACKAGE_DIR / f"_tmbv_{name}_g{generation}.mojo"
            scratch.write_text(
                src.read_text().replace(
                    f"def PyInit_{name}", f"def PyInit__tmbv_{name}_g{generation}"
                )
            )
            src = scratch
        scope = "full" if ops is None else f"{len(ops)} ops"
        import time as _time

        print(
            f"torch-mojo-backend: compiling {name} [{scope}] on demand..."
            + (f" t={_time.monotonic():.2f}" if os.environ.get("TMB_TRACE") else ""),
            file=sys.stderr,
        )
        try:
            proc = subprocess.run(
                _variant_cmd(name, src, ops, dtypes, out),
                capture_output=True,
                text=True,
                env=_build_env(),
            )
            if os.environ.get("TMB_TRACE"):
                import time as _time

                print(
                    f"[TRACE] built {name} t={_time.monotonic():.2f}",
                    file=sys.stderr,
                    flush=True,
                )
            if proc.returncode != 0:
                raise ImportError(
                    f"mojo build failed for {name} "
                    f"({_variant_tag(ops, dtypes)}):\n{proc.stderr}"
                )
        finally:
            if scratch is not None:
                scratch.unlink(missing_ok=True)
        return out


def _import_mojo_module(name: str) -> ModuleType:
    """Compatibility seam kept from the previous loader: the single point a
    module's target variant is built (if needed) and loaded. Tests patch this
    to simulate compiler failure / unavailable extensions."""
    if name != "tensor_holder":
        # tensor_holder registers the process-wide TensorHolder/TensorSpec
        # Python types every other module's spec ops consume; it must be
        # loaded and finalized before any kernel module.
        _STATES["tensor_holder"].ensure_loaded(None)
    state = _STATES[name]
    so_path = _build_variant(state.name, state.ops, state.dtypes, state.generation)
    suffix = (
        state.name
        if state.generation == 0
        else f"_tmbv_{state.name}_g{state.generation}"
    )
    return _load_extension(f"{__name__}.{suffix}", so_path)


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

    def __init__(self, state: "_ModuleState", job: "_AsyncVariantJob") -> None:
        super().__init__(f"kernel variant of {state.name} is compiling")
        self.state = state
        self.job = job


_DISPATCH_TLS = threading.local()


def _in_torch_dispatch() -> bool:
    return getattr(_DISPATCH_TLS, "depth", 0) > 0


class _dispatch_scope:
    """Marks 'this extension call came through __torch_dispatch__', which is
    the only context where a variant miss may raise KernelPending instead of
    compiling synchronously (direct callers, e.g. tests, still block)."""

    def __enter__(self) -> None:
        _DISPATCH_TLS.depth = getattr(_DISPATCH_TLS, "depth", 0) + 1

    def __exit__(self, *exc: object) -> None:
        _DISPATCH_TLS.depth -= 1


class _AsyncVariantJob:
    """One background build of a wider variant for one module. Requests that
    arrive while it runs accumulate in the state's wanted-set; a follow-up
    job starts automatically when this one completes with wants left over."""

    def __init__(self, state: "_ModuleState") -> None:
        self.state = state
        self.done = threading.Event()
        self.urgent = threading.Event()
        self.error: BaseException | None = None

    def run(self) -> None:
        state = self.state
        try:
            # Merge window: let demands accumulate briefly so one build
            # covers several ops. A blocked waiter cuts the window short.
            self.urgent.wait(timeout=0.5)
            with _ASYNC_BUILD_SLOTS:
                with state.lock:
                    profile = _PROFILE.get(state.name, {})
                    first_build = state.module is None
                    base_ops = set(state.ops or ())
                    if first_build:
                        base_ops |= set(profile.get("ops", ()))
                    target_ops = frozenset(base_ops | state.wanted_ops)
                    if state.wanted_all_dtypes or (
                        not first_build and state.dtypes is None
                    ):
                        target_dtypes: frozenset[str] | None = None
                    elif first_build:
                        target_dtypes = frozenset(
                            profile.get("dtypes", ()) or _DEFAULT_DTYPES
                        )
                    else:
                        target_dtypes = state.dtypes
                    generation = state.generation + 1
                path = _build_variant(state.name, target_ops, target_dtypes, generation)
                with state.lock:
                    state.generation = generation
                    state.ops = target_ops
                    state.dtypes = target_dtypes
                    state.module = _load_extension(
                        f"{__name__}._tmbv_{state.name}_g{generation}", path
                    )
                    state.wanted_ops -= set(target_ops)
                    if target_dtypes is None:
                        state.wanted_all_dtypes = False
                    proxy = _PROXIES[state.name]
                    proxy.__dict__.clear()
                    proxy.__dict__["_state"] = state
                    proxy.__dict__["__name__"] = f"{__name__}.{state.name}"
        except BaseException as exc:  # surfaced to every waiter
            self.error = exc
        finally:
            with state.lock:
                if state.async_job is self:
                    state.async_job = None
                more = state.wanted_ops or state.wanted_all_dtypes
            self.done.set()
            if more and self.error is None:
                state.request_async()

    def wait(self) -> None:
        self.urgent.set()  # a blocked consumer: start building now
        self.done.wait()
        if self.error is not None:
            raise self.error


class _ModuleState:
    """Loaded-variant bookkeeping for one .mojo module."""

    def __init__(self, name: str) -> None:
        self.name = name
        self.lock = threading.Lock()
        self.module: ModuleType | None = None
        self.ops: frozenset[str] | None = None  # None = all
        self.dtypes: frozenset[str] | None = None  # None = all
        self.demanded_ops: set[str] = set()
        self.generation = 0
        self.wanted_ops: set[str] = set()
        self.wanted_all_dtypes = False
        self.async_job: _AsyncVariantJob | None = None

    def request_async(
        self, add_op: str | None = None, all_dtypes: bool = False
    ) -> _AsyncVariantJob:
        """Record a demand and make sure a build covering it is in flight."""
        if add_op is not None and add_op not in _registered_ops(self.name):
            raise AttributeError(f"module {self.name!r} has no entry point {add_op!r}")
        with self.lock:
            if add_op is not None:
                self.wanted_ops.add(add_op)
            if all_dtypes:
                self.wanted_all_dtypes = True
            job = self.async_job
            if job is None:
                job = self.async_job = _AsyncVariantJob(self)
                threading.Thread(
                    target=job.run, name=f"tmb-build-{self.name}", daemon=True
                ).start()
            return job

    def ensure_loaded(self, first_op: str | None) -> ModuleType:
        if (
            self.module is None
            and self.name not in _FULL_MODULES
            and _in_torch_dispatch()
            and not _PREWARM.has_build_for(self.name)
        ):
            raise KernelPending(self, self.request_async(add_op=first_op))
        with self.lock:
            if self.module is None:
                _PREWARM.wait_for(self.name)
                profile = _PROFILE.get(self.name, {})
                if self.name in _FULL_MODULES:
                    ops: frozenset[str] | None = None
                    dtypes: frozenset[str] | None = None
                else:
                    wanted = set(profile.get("ops", ()))
                    if first_op is not None:
                        wanted.add(first_op)
                    ops = frozenset(wanted)
                    dtypes = frozenset(profile.get("dtypes", ()) or _DEFAULT_DTYPES)
                self.ops, self.dtypes = ops, dtypes
                self.module = _import_mojo_module(self.name)
            return self.module

    def escalate(
        self, add_op: str | None = None, all_dtypes: bool = False
    ) -> ModuleType:
        if self.name in _FULL_MODULES:
            raise AttributeError(
                f"{self.name} is built complete; no attribute {add_op!r}"
            )
        if add_op is not None and add_op not in _registered_ops(self.name):
            raise AttributeError(f"module {self.name!r} has no entry point {add_op!r}")
        with self.lock:
            ops = set(self.ops or ())
            if add_op is not None:
                ops.add(add_op)
            dtypes = None if all_dtypes else self.dtypes
            rollback = (self.ops, self.dtypes, self.generation, self.module)
            self.generation += 1
            self.ops = frozenset(ops)
            self.dtypes = dtypes
            try:
                self.module = _import_mojo_module(self.name)
            except BaseException:
                self.ops, self.dtypes, self.generation, self.module = rollback
                raise
            proxy = _PROXIES[self.name]
            proxy.__dict__.clear()
            proxy.__dict__["_state"] = self
            proxy.__dict__["__name__"] = f"{__name__}.{self.name}"
            return self.module


def _wrap_call(state: _ModuleState, attr: str, fn: object) -> object:
    def call(*args: object, **kwargs: object) -> object:
        try:
            return fn(*args, **kwargs)
        except Exception as exc:  # Mojo errors surface as plain Exception
            if "unsupported dtype" not in str(exc) or state.dtypes is None:
                raise
            if _in_torch_dispatch():
                raise KernelPending(
                    state, state.request_async(all_dtypes=True)
                ) from exc
            module = state.escalate(all_dtypes=True)
            return getattr(module, attr)(*args, **kwargs)

    return call


class _ModuleProxy:
    """Stands in for one extension module; escalates variants on demand."""

    def __init__(self, state: _ModuleState) -> None:
        self.__dict__["_state"] = state
        # Real module metadata: tests and tooling identify the extension by
        # its canonical module name, which the proxy stands in for.
        self.__dict__["__name__"] = f"{__name__}.{state.name}"

    def __getattr__(self, attr: str) -> object:
        state: _ModuleState = self.__dict__["_state"]
        if attr.startswith("__"):
            raise AttributeError(attr)
        module = state.ensure_loaded(attr)
        try:
            value = getattr(module, attr)
        except AttributeError:
            if state.name not in _FULL_MODULES and _in_torch_dispatch():
                raise KernelPending(state, state.request_async(add_op=attr)) from None
            module = state.escalate(add_op=attr)
            value = getattr(module, attr)
        state.demanded_ops.add(attr)
        if type(value).__name__ == "builtin_function_or_method":
            value = _wrap_call(state, attr, value)
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


class _Prewarm:
    """Background builds of every profiled variant at import time, run
    through a slot pool sized to the machine (`_pool_size`)."""

    def __init__(self) -> None:
        self.lock = threading.Lock()
        self.slots = _pool_size()
        self.pending: list[tuple[str, list[str]]] = []
        self.running: dict[str, subprocess.Popen] = {}
        for name, entry in _PROFILE.items():
            if name in _FULL_MODULES or name not in _MOJO_MODULES:
                continue
            ops = frozenset(entry.get("ops", ()))
            if not ops:
                continue
            dtypes = frozenset(entry.get("dtypes", ()) or _DEFAULT_DTYPES)
            out = _variant_path(name, ops, dtypes, 0)
            if out.is_file():
                continue
            _CACHE_DIR.mkdir(exist_ok=True)
            self.pending.append(
                (
                    name,
                    _variant_cmd(name, _PACKAGE_DIR / f"{name}.mojo", ops, dtypes, out),
                )
            )
        if self.pending:
            print(
                f"torch-mojo-backend: prewarming {len(self.pending)} kernel "
                f"variants in the background ({self.slots} build slots)...",
                file=sys.stderr,
            )
            self._pump()
            threading.Thread(target=self._reaper, daemon=True).start()

    def _pump(self) -> None:
        with self.lock:
            self.running = {n: p for n, p in self.running.items() if p.poll() is None}
            while self.pending and len(self.running) < self.slots:
                name, cmd = self.pending.pop(0)
                self.running[name] = subprocess.Popen(
                    cmd,
                    stdout=subprocess.DEVNULL,
                    stderr=subprocess.DEVNULL,
                    env=_build_env(),
                )

    def _reaper(self) -> None:
        while True:
            with self.lock:
                live = any(p.poll() is None for p in self.running.values())
                idle = not self.pending and not live
            if idle:
                return
            self._pump()
            threading.Event().wait(0.5)

    def has_build_for(self, name: str) -> bool:
        """A prewarm build for this module is pending or running (its result
        is imminent, so first-touch should wait for it rather than defer)."""
        with self.lock:
            return name in self.running or any(n == name for n, _ in self.pending)

    def wait_for(self, name: str) -> None:
        """Block until this module's prewarm build (if any) has finished.
        A still-pending build is promoted to run immediately."""
        with self.lock:
            for i, (n, cmd) in enumerate(self.pending):
                if n == name:
                    del self.pending[i]
                    self.running[name] = subprocess.Popen(
                        cmd,
                        stdout=subprocess.DEVNULL,
                        stderr=subprocess.DEVNULL,
                        env=_build_env(),
                    )
                    break
            proc = self.running.get(name)
        if proc is not None:
            proc.wait()
            self._pump()


_PROFILE = _load_profile()
_ASYNC_BUILD_SLOTS = threading.Semaphore(_pool_size())
_PREWARM = _Prewarm()
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
        if name in _CACHED_IN_DICT and state.module is not None:
            # A previously *successful* resolution was explicitly deleted
            # from the package dict (tests use this to force a fresh
            # import): drop the loaded module so resolution goes back
            # through _import_mojo_module.
            with state.lock:
                state.module = None
            proxy.__dict__.clear()
            proxy.__dict__["_state"] = state
            proxy.__dict__["__name__"] = f"{__name__}.{name}"
            _CACHED_IN_DICT.discard(name)
        # Import at resolution time, like a real module attribute: an
        # unavailable extension raises HERE and is not cached, so callers'
        # ImportError handling and failure-flag caching behave as before.
        # Under __torch_dispatch__ resolution stays lazy instead — the first
        # attribute access carries the op name, so the variant built covers
        # it (resolution alone would build a useless empty variant).
        if not _in_torch_dispatch():
            state.ensure_loaded(None)
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
