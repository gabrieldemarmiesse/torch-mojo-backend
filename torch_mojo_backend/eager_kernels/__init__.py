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
        print(
            f"torch-mojo-backend: compiling {name} [{scope}] on demand...",
            file=sys.stderr,
        )
        try:
            proc = subprocess.run(
                _variant_cmd(name, src, ops, dtypes, out),
                capture_output=True,
                text=True,
                env=_build_env(),
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


def _load_extension(module_name: str, so_path: Path) -> ModuleType:
    loader = importlib.machinery.ExtensionFileLoader(module_name, str(so_path))
    spec = importlib.util.spec_from_file_location(
        module_name, str(so_path), loader=loader
    )
    module = importlib.util.module_from_spec(spec)
    loader.exec_module(module)
    return module


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

    def ensure_loaded(self, first_op: str | None) -> ModuleType:
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
                so_path = _build_variant(self.name, ops, dtypes, 0)
                self.module = _load_extension(f"{__name__}.{self.name}", so_path)
                self.ops, self.dtypes = ops, dtypes
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
            self.generation += 1
            so_path = _build_variant(self.name, frozenset(ops), dtypes, self.generation)
            self.module = _load_extension(
                f"{__name__}._tmbv_{self.name}_g{self.generation}", so_path
            )
            self.ops = frozenset(ops)
            self.dtypes = dtypes
            proxy = _PROXIES[self.name]
            proxy.__dict__.clear()
            proxy.__dict__["_state"] = self
            return self.module


def _wrap_call(state: _ModuleState, attr: str, fn: object) -> object:
    def call(*args: object, **kwargs: object) -> object:
        try:
            return fn(*args, **kwargs)
        except Exception as exc:  # Mojo errors surface as plain Exception
            if "unsupported dtype" not in str(exc) or state.dtypes is None:
                raise
            module = state.escalate(all_dtypes=True)
            return getattr(module, attr)(*args, **kwargs)

    return call


class _ModuleProxy:
    """Stands in for one extension module; escalates variants on demand."""

    def __init__(self, state: _ModuleState) -> None:
        self.__dict__["_state"] = state

    def __getattr__(self, attr: str) -> object:
        state: _ModuleState = self.__dict__["_state"]
        if attr.startswith("__"):
            raise AttributeError(attr)
        module = state.ensure_loaded(attr)
        try:
            value = getattr(module, attr)
        except AttributeError:
            module = state.escalate(add_op=attr)
            value = getattr(module, attr)
        state.demanded_ops.add(attr)
        if type(value).__name__ == "builtin_function_or_method":
            value = _wrap_call(state, attr, value)
        self.__dict__[attr] = value  # later lookups skip __getattr__
        return value


class _Prewarm:
    """Background parallel builds of every profiled variant at import time."""

    def __init__(self) -> None:
        self.procs: dict[str, subprocess.Popen] = {}
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
            self.procs[name] = subprocess.Popen(
                _variant_cmd(name, _PACKAGE_DIR / f"{name}.mojo", ops, dtypes, out),
                stdout=subprocess.DEVNULL,
                stderr=subprocess.DEVNULL,
                env=_build_env(),
            )
        if self.procs:
            print(
                f"torch-mojo-backend: prewarming {len(self.procs)} kernel "
                "variants in the background...",
                file=sys.stderr,
            )

    def wait_for(self, name: str) -> None:
        proc = self.procs.pop(name, None)
        if proc is not None:
            proc.wait()


_PROFILE = _load_profile()
_PREWARM = _Prewarm()
_STATES: dict[str, _ModuleState] = {n: _ModuleState(n) for n in _MOJO_MODULES}
_PROXIES: dict[str, _ModuleProxy] = {
    n: _ModuleProxy(_STATES[n]) for n in _MOJO_MODULES if n not in _FULL_MODULES
}
atexit.register(_save_profile)


def __getattr__(name: str) -> object:
    if name in _MOJO_MODULES:
        # tensor_holder must be imported and finalized before any other
        # module: it registers the process-wide TensorHolder/TensorSpec
        # Python types every other module's spec ops consume.
        holder = _STATES["tensor_holder"].ensure_loaded(None)
        if name == "tensor_holder":
            globals()[name] = holder
            return holder
        proxy = _PROXIES[name]
        globals()[name] = proxy
        return proxy
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")


_CTX_PTR_CACHE: dict[driver.Device, int] = {}


def _ctx_ptr(device: driver.Device) -> int:
    ptr = _CTX_PTR_CACHE.get(device)
    if ptr is None:
        ptr = device._device_context_ptr()
        _CTX_PTR_CACHE[device] = ptr
    return ptr
