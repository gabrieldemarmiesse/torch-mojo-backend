"""Select the newest PTX assembler compatible with the NVIDIA driver and GPUs.

Discover installed assemblers and MAX's bundled compiler; an unknown MAX
compiler version is a last resort. Check the driver at import and the GPUs
at device registration, preserving explicit ``MODULAR_NVPTX_COMPILER_PATH``
settings. Report rejected candidates and installation advice when none fits.
"""

from __future__ import annotations

import ctypes
import functools
import importlib.metadata
import importlib.util
import os
import re
import shutil
import subprocess
import sys
import warnings
from dataclasses import dataclass
from pathlib import Path

from torch_mojo_backend import _cuda_driver

ENV_VAR = "MODULAR_NVPTX_COMPILER_PATH"
TRITON_ENV_VAR = "TRITON_PTXAS_PATH"
CHECK_ENV_VAR = "TORCH_MOJO_BACKEND_PTXAS_CHECK"
AUTO_ENV_VAR = "TORCH_MOJO_BACKEND_PTXAS_AUTO"

# Development pins and suggested optional installs. The cu12 series is the
# default because its cubins load on every r525+ driver; CUDA
# 13 renamed the project (`nvidia-cuda-nvcc-cu13` is a deprecated stub).
WHEELS: dict[int, tuple[str, str]] = {
    11: ("nvidia-cuda-nvcc-cu11", "11.8.*"),
    12: ("nvidia-cuda-nvcc-cu12", "12.8.*"),
    13: ("nvidia-cuda-nvcc", "13.0.*"),
}
PINNED_WHEEL = WHEELS[12]

BUILTIN_SOURCE = "MAX built-in libnvptxcompiler"
# What TORCH_MOJO_BACKEND_PTXAS_AUTO holds when our pick was "no ptxas at all":
# not a path, since the variable it marks is then absent.
BUILTIN_MARK = "<max built-in>"

# The driver each CUDA major needs, from NVIDIA's minor version compatibility
# table: one number per major, because within a major any toolkit's cubins
# load on the major's minimum driver.
MIN_DRIVER: dict[int, str] = {11: "r450", 12: "r525", 13: "r580"}

# First CUDA release that can target an architecture -- the *name we need*,
# `target_name` below, not the bare sm number: CUDA 11.8 introduced sm_90 but
# only 12.0 added sm_90a, which is the one MAX asks for on an H100. Advice
# only -- a candidate is judged by what its own --help lists, never by this
# table -- so it is needed exactly when there is no usable ptxas left to ask.
# tests/test_ptxas_selection.py holds it up against the real binaries.
ARCH_MIN_CUDA: dict[int, tuple[int, int]] = {
    50: (6, 0),
    52: (6, 5),
    53: (7, 0),
    60: (8, 0),
    61: (8, 0),
    62: (8, 0),
    70: (9, 0),
    72: (9, 1),
    75: (10, 0),
    80: (11, 0),
    86: (11, 1),
    87: (11, 4),
    88: (13, 0),
    89: (11, 8),
    90: (12, 0),  # sm_90 is 11.8, sm_90a -- what we need -- is 12.0
    100: (12, 8),
    101: (12, 8),
    103: (12, 9),
    110: (13, 0),
    120: (12, 8),
    121: (12, 9),
}

# CUDA 13 dropped Maxwell, Pascal and Volta -- and sm_101 -- so these need a
# 12.x assembler even on a driver new enough for 13.
ARCH_LAST_MAJOR: dict[int, int] = dict.fromkeys(
    (50, 52, 53, 60, 61, 62, 70, 72, 101), 12
)

# Architectures with an `a` (architecture-specific) form. MAX targets that
# form when it exists -- `gpu-query --target-accelerator` answers
# `nvidia:sm_90a` on an H100 -- and it is a *separate* --gpu-name value: CUDA
# 11.8 takes sm_90 and rejects sm_90a, which is how a too-old assembler fails
# the build. So this is the name a candidate has to support, not `sm_90`.
ARCH_A_VARIANT: frozenset[int] = frozenset({90, 100, 101, 103, 110, 120, 121})


def target_name(arch: int) -> str:
    """The --target-accelerator name MAX assembles an sm number for."""
    return f"sm_{arch}a" if arch in ARCH_A_VARIANT else f"sm_{arch}"


def _trace(message: str):
    """`native._trace` without the import: native.py imports this module."""
    if os.environ.get("TORCH_MOJO_BACKEND_TRACE", "1") != "0":
        print(f"[TRACE] {message}", file=sys.stderr, flush=True)


@dataclass(frozen=True)
class Ptxas:
    """One assembler found on the machine."""

    path: Path
    source: str  # where it came from, for the report
    # None for an unversioned built-in compiler or an external one that failed.
    version: tuple[int, int, int] | None

    @property
    def release(self) -> str:
        if self.version is None:
            if self.is_builtin:
                return "CUDA version unknown (last-resort fallback)"
            return "no such file" if not self.path.is_file() else "--version failed"
        return f"CUDA {self.version[0]}.{self.version[1]}"

    @property
    def is_builtin(self) -> bool:
        """MAX's own compiler: selected by *unsetting* the variable."""
        return self.source == BUILTIN_SOURCE

    @property
    def label(self) -> str:
        """How a message names this assembler."""
        return "MAX's built-in assembler" if self.is_builtin else str(self.path)

    @property
    def mark(self) -> str:
        """The value ``adopt`` records for this choice."""
        return BUILTIN_MARK if self.is_builtin else str(self.path)


def table_arches(version: tuple[int, int, int]) -> frozenset[str]:
    """The targets a CUDA release supports, per the tables above.

    For the built-in compiler only, which has no ``--help`` to ask; every
    external ptxas is judged by :func:`arches_of` instead.
    """
    return frozenset(
        target_name(arch)
        for arch, first in ARCH_MIN_CUDA.items()
        if first <= version[:2] and version[0] <= ARCH_LAST_MAJOR.get(arch, 99)
    )


def _version_of(path: Path) -> tuple[int, int, int] | None:
    """The release ptxas reports, or None when it cannot be run at all."""
    try:
        out = subprocess.run(
            [str(path), "--version"], capture_output=True, text=True, timeout=30
        )
    except (OSError, subprocess.SubprocessError):
        return None
    match = re.search(r"V(\d+)\.(\d+)\.(\d+)", out.stdout + out.stderr)
    if match is None:
        return None
    return (int(match[1]), int(match[2]), int(match[3]))


@functools.cache
def arches_of(path: Path) -> frozenset[str]:
    """The --gpu-name values this ptxas accepts, from its own ``--help``.

    Names keep their suffix (`sm_90` and `sm_90a` are two different values,
    and only the second one builds on an H100). Empty when the output cannot
    be parsed, which callers read as "unknown" and never as "targets
    nothing": a wording change in a future CUDA must not start rejecting
    working assemblers.
    """
    try:
        out = subprocess.run(
            [str(path), "--help"], capture_output=True, text=True, timeout=30
        )
    except (OSError, subprocess.SubprocessError):
        return frozenset()
    return frozenset(re.findall(r"'(sm_\d+[a-z]?)'", out.stdout))


def _nvidia_wheel_ptxas() -> list[tuple[Path, str]]:
    """Every ptxas under the `nvidia` namespace package.

    The cu12 nvcc wheel installs `nvidia/cuda_nvcc/bin/ptxas`, the CUDA 13
    one `nvidia/cu13/bin/ptxas`; globbing the namespace covers both and
    whatever layout comes next.
    """
    try:
        spec = importlib.util.find_spec("nvidia")
    except (ImportError, ValueError):  # no `nvidia` namespace package (macOS)
        return []
    if spec is None or not spec.submodule_search_locations:
        return []
    found = []
    for location in spec.submodule_search_locations:
        for candidate in sorted(Path(location).glob("*/bin/ptxas")):
            layout = candidate.parent.parent.name
            found.append(
                (
                    candidate,
                    f"{PINNED_WHEEL[0]} wheel"
                    if layout == "cuda_nvcc"
                    else f"nvidia-cuda-nvcc wheel ({layout})",
                )
            )
    return found


def wheel_ptxas() -> Path | None:
    """The ptxas shipped by nvidia-cuda-nvcc-cu12, or None when not installed."""
    try:
        spec = importlib.util.find_spec("nvidia.cuda_nvcc")
    except ModuleNotFoundError:  # no `nvidia` namespace package at all (macOS)
        return None
    if spec is None or not spec.submodule_search_locations:
        return None
    for location in spec.submodule_search_locations:
        candidate = Path(location) / "bin" / "ptxas"
        if candidate.is_file():
            return candidate
    return None


def torch_wheel_ptxas() -> Path | None:
    """The ptxas inside the installed torch wheel, which is what
    ``torch/_inductor/runtime/compile_tasks.py``'s ``_set_triton_ptxas_path``
    puts in ``TRITON_PTXAS_PATH`` -- at *import* of that module, so it is
    usually already set by the time anything of ours runs."""
    spec = importlib.util.find_spec("torch")
    if spec is None or not spec.submodule_search_locations:
        return None
    for location in spec.submodule_search_locations:
        candidate = Path(location) / "bin" / "ptxas"
        if candidate.is_file():
            return candidate
    return None


def _triton_ptxas() -> Path | None:
    spec = importlib.util.find_spec("triton")
    if spec is None or not spec.submodule_search_locations:
        return None
    for location in spec.submodule_search_locations:
        candidate = Path(location) / "backends" / "nvidia" / "bin" / "ptxas"
        if candidate.is_file():
            return candidate
    return None


def _builtin_version(path: Path) -> tuple[int, int, int] | None:
    """Query MAX's bundled compiler without importing MAX or initializing CUDA.

    NVIDIA's API reports the CUDA Toolkit major.minor, not the PTX ISA
    version. It exposes no patch version, so use zero for that component.
    Older wheels may not expose a shared library or this symbol.
    """
    try:
        lib = ctypes.CDLL(str(path))
        get_version = lib.nvPTXCompilerGetVersion
    except (OSError, AttributeError):
        return None
    get_version.argtypes = [
        ctypes.POINTER(ctypes.c_uint),
        ctypes.POINTER(ctypes.c_uint),
    ]
    get_version.restype = ctypes.c_int
    major, minor = ctypes.c_uint(), ctypes.c_uint()
    if get_version(ctypes.byref(major), ctypes.byref(minor)) != 0 or major.value == 0:
        return None
    return major.value, minor.value, 0


def builtin_ptxas() -> Ptxas | None:
    """The compiler bundled with the installed ``max``, or None without MAX.

    Found without importing ``max`` -- this runs at import, before it loads.
    """
    try:
        spec = importlib.util.find_spec("max")
    except (ImportError, ValueError):
        return None
    if spec is None or not spec.submodule_search_locations:
        return None
    try:
        dist = importlib.metadata.distribution("max-core")
    except importlib.metadata.PackageNotFoundError:
        # Older layouts may embed the compiler in MAX instead of shipping
        # libNVPTX.so. Keep them as candidates with unknown compatibility.
        path = Path(next(iter(spec.submodule_search_locations)))
        return Ptxas(path=path, source=BUILTIN_SOURCE, version=None)
    path = Path(str(dist.locate_file("modular/lib/libNVPTX.so")))
    return Ptxas(path=path, source=BUILTIN_SOURCE, version=_builtin_version(path))


def _system_ptxas() -> list[tuple[Path, str]]:
    """ptxas from a CUDA toolkit installed on the machine."""
    found = []
    for var, root in (
        ("CUDA_HOME", os.environ.get("CUDA_HOME")),
        ("CUDA_PATH", os.environ.get("CUDA_PATH")),
    ):
        if root:
            found.append((Path(root) / "bin" / "ptxas", f"${var}"))
    on_path = shutil.which("ptxas")
    if on_path:
        found.append((Path(on_path), "$PATH"))
    for root in sorted(Path("/usr/local").glob("cuda*"), reverse=True):
        found.append((root / "bin" / "ptxas", "system CUDA"))
    return found


def adopt(path: str):
    """Assemble with `path`, and record that the choice was ours.

    Both variables go into the environment, because every child inherits it:
    a rank spawned by torchrun, an Inductor compile worker, a `mojo build`.
    Without the second one a child sees an inherited automatic choice as a
    setting of the user's and refuses to move off it, which is the opposite
    of what its own GPU may need.
    """
    os.environ[ENV_VAR] = path
    os.environ[AUTO_ENV_VAR] = path


def adopt_builtin():
    """Assemble with MAX's own compiler: the variable *absent*, and marked so.

    A child that inherits the environment then sees no setting and the mark,
    and re-selects for its own GPU like the parent did.
    """
    os.environ.pop(ENV_VAR, None)
    os.environ[AUTO_ENV_VAR] = BUILTIN_MARK


def _adopt(ptxas: Ptxas):
    if ptxas.is_builtin:
        adopt_builtin()
    else:
        adopt(str(ptxas.path))


def explicit_choice() -> str | None:
    """``MODULAR_NVPTX_COMPILER_PATH`` as the user set it, not as we did."""
    value = os.environ.get(ENV_VAR)
    if not value or value == os.environ.get(AUTO_ENV_VAR):
        return None
    return value


def candidates() -> list[Ptxas]:
    """Every ptxas on this machine, in discovery order, each asked its version.

    The environment's own choice comes first so a report always explains the
    setting in force; MAX's own compiler comes last, since it is what runs
    when nothing is set.
    """
    found = list[tuple[Path, str]]()
    explicit = explicit_choice()
    if explicit:
        found.append((Path(explicit), ENV_VAR))
    found += _nvidia_wheel_ptxas()
    torch_ptxas = torch_wheel_ptxas()
    if torch_ptxas is not None:
        found.append((torch_ptxas, "torch wheel"))
    triton_ptxas = _triton_ptxas()
    if triton_ptxas is not None:
        found.append((triton_ptxas, "triton wheel"))
    found += _system_ptxas()

    seen = set[Path]()
    result = []
    for path, source in found:
        try:
            resolved = path.resolve()
        except OSError:
            continue
        # A path the user gave us is reported even when it is not there:
        # "no such file" is the answer they need. Ours are skipped silently.
        if resolved in seen or (not path.is_file() and source != ENV_VAR):
            continue
        seen.add(resolved)
        result.append(Ptxas(path=path, source=source, version=_version_of(path)))
    builtin = builtin_ptxas()
    if builtin is not None:
        result.append(builtin)
    return result


def driver_cuda_version() -> tuple[int, int] | None:
    """The CUDA version this NVIDIA driver supports, or None without one.

    ``cuDriverGetVersion`` is one of the few entry points callable before
    ``cuInit``, so this creates no context and touches no GPU: it is safe at
    import, in a process that later forks, and on a machine whose GPUs are
    all busy.
    """
    try:
        lib = ctypes.CDLL("libcuda.so.1")
    except OSError:
        return None
    version = ctypes.c_int()
    try:
        if lib.cuDriverGetVersion(ctypes.byref(version)) != 0:
            return None
    except AttributeError:
        return None
    major, minor = divmod(version.value // 10, 100)
    return (major, minor)


def driver_release() -> str | None:
    """The r-number of the loaded driver (``570.211.01``), for the report."""
    try:
        return Path("/sys/module/nvidia/version").read_text().strip()
    except OSError:
        return None


_arches: tuple[tuple[int, str], ...] = ()


def device_arches() -> tuple[tuple[int, str], ...]:
    """(sm number, name) of every visible NVIDIA GPU.

    Needs ``cuInit``, which is real initialization and not a passive probe,
    so it is called at registration -- where the driver is about to be
    initialized anyway -- and never at import. The answer is remembered, but
    a failure is not: a driver that was busy or wedged at the first attempt
    must not leave the process believing for good that there are no GPUs.
    """
    global _arches
    if _arches:
        return _arches
    try:
        lib = ctypes.CDLL("libcuda.so.1")
    except OSError:
        return ()
    if lib.cuInit(0) != 0:
        return ()
    count = ctypes.c_int()
    if lib.cuDeviceGetCount(ctypes.byref(count)) != 0:
        return ()
    found = []
    for index in range(count.value):
        device = ctypes.c_int()
        if lib.cuDeviceGet(ctypes.byref(device), index) != 0:
            continue
        major, minor = ctypes.c_int(), ctypes.c_int()
        cc_major = _cuda_driver.COMPUTE_CAPABILITY_MAJOR
        cc_minor = _cuda_driver.COMPUTE_CAPABILITY_MINOR
        if lib.cuDeviceGetAttribute(ctypes.byref(major), cc_major, device) != 0:
            continue
        if lib.cuDeviceGetAttribute(ctypes.byref(minor), cc_minor, device) != 0:
            continue
        found.append((major.value * 10 + minor.value, _cuda_driver.name(device, lib)))
    _arches = tuple(found)
    return _arches


def rejection(
    ptxas: Ptxas, driver: tuple[int, int] | None, arches: tuple[int, ...]
) -> str | None:
    """Why this ptxas cannot be used here, or None when it can."""
    if ptxas.version is None:
        if ptxas.is_builtin:
            # Compatibility is unknown, so allow it only as a last resort
            # through choose()'s ranking, without inventing version bounds.
            return None
        return ptxas.release  # "no such file" or "--version failed"
    major, minor, _ = ptxas.version
    if driver is not None and major > driver[0]:
        floor = MIN_DRIVER.get(major, f"the CUDA {major} minimum")
        return (
            f"too new for this driver: its cubins need driver {floor} or newer, "
            f"and this driver supports CUDA {driver[0]}.{driver[1]}"
        )
    supported = (
        table_arches(ptxas.version) if ptxas.is_builtin else arches_of(ptxas.path)
    )
    if supported:  # empty means --help could not be parsed: not a rejection
        missing = {target_name(arch) for arch in arches} - supported
        if missing:
            return f"too old for this GPU: cannot target {', '.join(sorted(missing))}"
    return None


def choose(
    driver: tuple[int, int] | None,
    arches: tuple[int, ...] = (),
    found: list[Ptxas] | None = None,
) -> tuple[Ptxas | None, list[tuple[Ptxas, str]]]:
    """The best usable ptxas, and every rejected one with its reason.

    Among known versions that fit, the highest release wins. An unversioned
    MAX compiler ranks last: try it only when no known assembler fits,
    leaving MAX to check compatibility at runtime.
    """
    if found is None:
        found = candidates()
    usable, rejected = [], []
    for ptxas in found:
        why = rejection(ptxas, driver, arches)
        if why is None:
            usable.append(ptxas)
        else:
            rejected.append((ptxas, why))
    if not usable:
        return None, rejected

    return max(usable, key=lambda ptxas: ptxas.version or (-1, -1, -1)), rejected


def install_advice(
    driver: tuple[int, int] | None, arches: tuple[int, ...]
) -> list[str]:
    """The wheel to install, given what this driver and these GPUs allow."""
    if driver is None:
        wheel, spec = PINNED_WHEEL
        return [f'pip install "{wheel}=={spec}"']
    needed = max((ARCH_MIN_CUDA.get(arch, (0, 0)) for arch in arches), default=(0, 0))
    ceiling = min((ARCH_LAST_MAJOR.get(arch, 99) for arch in arches), default=99)
    major = min(driver[0], ceiling)
    names = ", ".join(f"sm_{arch}" for arch in sorted(set(arches)))
    if needed[0] > ceiling:
        # Not the driver's fault and not installable: these GPUs have no
        # assembler in common, one needing a release that dropped another.
        return [
            f"no CUDA release targets {names} at once: one needs "
            f"{needed[0]}.{needed[1]}, which no longer targets the other.",
            "Run on one architecture at a time (CUDA_VISIBLE_DEVICES).",
        ]
    if needed[0] > major:
        # No wheel can help: the architecture is newer than anything this
        # driver loads.
        return [
            f"{names} needs CUDA {needed[0]}.{needed[1]}, whose cubins need "
            f"driver {MIN_DRIVER.get(needed[0], 'a newer one')} or newer, and "
            f"this driver supports CUDA {driver[0]}.{driver[1]}.",
            "Update the NVIDIA driver; no assembler bridges this gap.",
        ]
    wheel, spec = WHEELS.get(major, PINNED_WHEEL)
    pinned = tuple(int(part) for part in spec.rstrip(".*").split("."))
    spec = f">={needed[0]}.{needed[1]},<{major + 1}" if needed > pinned else f"=={spec}"
    advice = [f'pip install "{wheel}{spec}"  (uv add "{wheel}{spec}")']
    if major < 12:
        advice.append(
            f"Driver {driver[0]}.{driver[1]} predates CUDA 12, which MAX's own "
            "runtime may need regardless of the assembler; updating it is the "
            "surer fix."
        )
    return advice


def report(
    driver: tuple[int, int] | None = None,
    devices: tuple[tuple[int, str], ...] | None = None,
    found: list[Ptxas] | None = None,
) -> str:
    """The whole picture: driver, GPUs, every ptxas and its verdict.

    Printed by ``torch-mojo-backend ptxas`` and quoted in the error raised
    when nothing works.
    """
    if driver is None:
        driver = driver_cuda_version()
    if devices is None:
        devices = device_arches()
    if found is None:
        found = candidates()
    arches = tuple(arch for arch, _ in devices)
    chosen, rejected = choose(driver, arches, found)
    reasons = {id(ptxas): why for ptxas, why in rejected}
    # A setting of the user's own is what runs, whatever we would have picked
    # ourselves -- check() leaves a working one alone -- and when it cannot
    # work nothing runs at all, so nothing is marked as in force.
    env_entry = next((p for p in found if p.source == ENV_VAR), None)
    if env_entry is not None:
        in_force = None if id(env_entry) in reasons else env_entry
    else:
        in_force = chosen

    lines = []
    if driver is None:
        lines.append("  driver    no NVIDIA driver on this machine (libcuda.so.1)")
    else:
        release = driver_release()
        named = f"{release} " if release else ""
        lines.append(f"  driver    {named}(supports CUDA {driver[0]}.{driver[1]})")
    if devices:
        for index, (arch, name) in enumerate(devices):
            lines.append(f"  gpu {index}     {name} (sm_{arch})")
    else:
        lines.append("  gpu       none visible")
    lines.append("")
    lines.append(
        f"  ptxas found, newest first, use {ENV_VAR} to override the automatic selection"
    )
    if not found:
        lines.append("    none")
    # Newest first; the ones whose version could not be read close the list.
    for ptxas in sorted(found, key=lambda p: p.version or (-1, 0, 0), reverse=True):
        why = reasons.get(id(ptxas))
        if why is not None:
            mark = "no"
        elif ptxas is in_force:
            mark = "USED"
        elif in_force is None and ptxas is chosen:
            mark = "pick"  # what would be used, once the setting is gone
        else:
            mark = "ok"
        detail = f"{ptxas.release}, from {ptxas.source}"
        lines.append(f"    [{mark:>4}] {ptxas.path}")
        if why is not None and ptxas.version is None:
            why = None  # the release column already says what went wrong
        lines.append(f"           {detail}" + (f" -- {why}" if why else ""))
    if chosen is None:
        lines.append("")
        lines.append("  fix")
        for line in install_advice(driver, arches):
            lines.append(f"    {line}")
        lines.append(f"    or set {ENV_VAR} to a ptxas of your own.")
    return "\n".join(lines)


def diagnose(message: str) -> str:
    """The ptxas picture, when a build log suggests the assembler is at fault.

    ``ptxas fatal : Value 'sm_90a' is not defined for option 'gpu-name'`` is
    the whole explanation a user gets otherwise, three screens into a
    compiler dump, with nothing to act on.
    """
    if "ptxas" not in message and ENV_VAR not in message:
        return ""
    return "\n\nThe assembler this was built with:\n\n" + report() + "\n"


def apply_default():
    """Point MAX at the best ptxas for this driver unless the user chose one.

    Runs at import, before ``max`` is loaded, so it uses only what can be
    known without initializing CUDA: the driver's version bound, not the
    GPU's architecture. :func:`check` revisits the choice at registration.
    """
    if explicit_choice():
        return
    driver = driver_cuda_version()
    if driver is None:
        # No NVIDIA driver: nothing here will ever load a cubin, so asking
        # every assembler on the machine its version would be a subprocess
        # each -- and a hung toolkit wrapper on a network mount would hold up
        # an AMD or CPU-only import for its timeout. The wheel's path is
        # still set, without running anything, because a build node with no
        # GPU cross-compiles with it.
        wheel = wheel_ptxas()
        if wheel is not None:
            adopt(str(wheel))
        return
    chosen, _ = choose(driver)
    if chosen is not None:
        _adopt(chosen)


class PtxasError(RuntimeError):
    """No ptxas on this machine can assemble kernels this GPU will run."""


_checked = False


def check(force: bool = False):
    """Confirm the ptxas in force suits the GPUs that are actually here.

    Called by ``register_mojo_devices()`` before anything is built. The
    architecture is only knowable once the driver is initialized, so this is
    the first moment the GPU's lower bound can be applied -- and it is still
    before the first kernel build, which is where a mismatch would otherwise
    surface as a ``mojo build`` dump or, worse, as
    ``CUDA_ERROR_INVALID_IMAGE`` at the first op.

    Re-selects when the import-time choice does not fit, raises
    :class:`PtxasError` when nothing does. ``TORCH_MOJO_BACKEND_PTXAS_CHECK=0``
    turns the refusal into a warning, for a machine whose rules we got wrong.
    """
    global _checked
    if _checked and not force:
        return
    _check_now()
    # Only a check that got through is remembered: registration is retryable
    # by design (`register_mojo_devices` leaves a failure retryable), and a
    # second attempt must hit this refusal again rather than sail past it and
    # fail later for a reason that says nothing.
    _checked = True


def _check_now():
    driver = driver_cuda_version()
    devices = device_arches()
    if driver is None or not devices:  # not an NVIDIA machine: nothing to pick
        return
    arches = tuple(arch for arch, _ in devices)
    found = candidates()
    explicit = explicit_choice()
    current = next((p for p in found if p.source == ENV_VAR), None)
    chosen, _ = choose(driver, arches, found)

    if explicit:
        why = rejection(current, driver, arches) if current else "no such file"
        if why is None:
            return  # the user's choice works here; nothing to say
        message = f"{ENV_VAR}={explicit} cannot be used here: {why}.\n"
        if chosen is not None:
            compatibility = (
                "compatibility is unknown"
                if chosen.version is None
                else "which works on this machine"
            )
            message += (
                f"Unset it and {chosen.label} ({chosen.release}) is used "
                f"instead, {compatibility}.\n"
            )
        message += "\n" + report(driver, devices, found)
    elif chosen is not None:
        previous = os.environ.get(AUTO_ENV_VAR)
        _adopt(chosen)
        if previous is not None and previous != chosen.mark:
            # Silently assembling with something other than the wheel we ship
            # is exactly the kind of thing a bug report needs to mention.
            before = next((p for p in found if p.mark == previous), None)
            why = rejection(before, driver, arches) if before else "is gone"
            _trace(f"ptxas: using {chosen.label} ({chosen.release}); {previous} {why}")
        return
    else:
        message = (
            "no usable ptxas: the mojo device assembles its kernels with ptxas, "
            "and none of the ones on this machine produces cubins these GPUs "
            "and this driver can both run.\n\n" + report(driver, devices, found)
        )
    message += "\n\n  `torch-mojo-backend ptxas` prints this table at any time."
    if os.environ.get(CHECK_ENV_VAR) == "0":
        warnings.warn(message, RuntimeWarning, stacklevel=2)
        return
    raise PtxasError(message)


def apply_triton_default():
    """Assemble Triton's kernels with the same ptxas as the device's own.

    Torch's choice tracks its wheel's CUDA rather than the driver: the cu130
    wheel on an r570 driver assembles cubins with ELF ABI version 8, and
    loading one fails with "device kernel image is invalid". MAX's ptxas is
    already pinned to something this driver loads, and a Triton kernel
    launched on the mojo device has exactly the same constraint, so it gets
    the same assembler -- but only over torch's default, recognized by its
    path. A ``TRITON_PTXAS_PATH`` pointing anywhere else is the user's and
    stands.
    """
    ptxas = os.environ.get(ENV_VAR)
    current = os.environ.get(TRITON_ENV_VAR)
    if not ptxas:
        return
    if current and Path(current) != torch_wheel_ptxas():
        return
    os.environ[TRITON_ENV_VAR] = ptxas


apply_default()
