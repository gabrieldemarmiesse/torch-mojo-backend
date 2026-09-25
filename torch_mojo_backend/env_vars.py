"""Every environment variable this project reads, in one place.

``register_mojo_devices()`` calls :func:`warn_about_unknown_env_vars`, which
holds the process environment up against :data:`OWN_ENV_VARS`. A variable
spelled ``TORCH_MOJO_BACKEND_*`` or ``MOJOCCL_*`` that is not in that table
can only be a typo — nothing anywhere would ever read it — so the user gets a
warning naming it, plus a "did you mean" whenever one of the real names is
close enough. Silence on a misspelled knob is the failure mode this exists to
kill: the setting simply does nothing, and the default it was meant to
override stays in force with no sign that anything went wrong.

``tests/test_env_vars_are_registered.py`` scans the repository and fails when
a name reachable through ``os.environ`` / ``getenv`` is missing from the
tables below, so the list cannot quietly fall behind the code.

The Mojo side keeps its own copy of the names it reads — one file per
compiled library, ``mojo/tmb/backend/env_vars.mojo`` and
``mojo/tmb/ccl/env_vars.mojo``, since those two build separately and
neither can import Python. This file is the union of all three, and the
scanner is what ties them together.
"""

from __future__ import annotations

import difflib
import warnings
from collections.abc import Mapping
from os import environ

# A variable starting with one of these is ours to recognize, so an unknown
# one is a user error worth reporting. `PYTORCH_MOJO_BACKEND_` is deliberately
# absent: it is a legacy alias spelling, not a namespace we invite people into.
CHECKED_PREFIXES = ("TORCH_MOJO_BACKEND_", "MOJOCCL_")

# Variables this project defines. Name -> what setting it does. Add here first
# and the scanner stays green; the value is what a user reads when they ask
# what a knob is for, so write it for them.
OWN_ENV_VARS: dict[str, str] = {
    # -- torch-mojo-backend: user-facing ---------------------------------
    "TORCH_MOJO_BACKEND_CACHE_DIR": (
        "Directory holding every cached native build (the C++ shim, the Mojo "
        "base library, the kernel specializations). Defaults to the user "
        "cache directory; point it at shared scratch on a cluster. "
        "`torch-mojo-backend cache dir` prints the value in force."
    ),
    "TORCH_MOJO_BACKEND_CCL": (
        "`mojo` runs collectives through the in-repo Mojo implementation "
        "(libmojoccl.so) instead of vendor NCCL/RCCL. Anything else, or "
        "unset, keeps the vendor library."
    ),
    "TORCH_MOJO_BACKEND_COMPILE_NATIVE_KERNELS": (
        "`0` makes the torch.compile backend compose MAX's own matmul / "
        "softmax / layer-norm / embedding ops instead of calling this "
        "repository's eager kernels through `tmb/graph`. On by "
        "default; the switch exists to compare the two and as an escape hatch."
    ),
    "TORCH_MOJO_BACKEND_DEBUG_GRAPH": (
        "`1` dumps the FX graph the torch.compile backend received."
    ),
    "TORCH_MOJO_BACKEND_NCCL_LIB": (
        "Absolute path of libnccl.so.2, overriding the search order "
        "(the nvidia-nccl-cu12 wheel, then the system loader)."
    ),
    "TORCH_MOJO_BACKEND_PREBUILT": (
        "`0` ignores the libraries shipped prebuilt in the wheel and compiles "
        "the shim and base library here instead."
    ),
    "TORCH_MOJO_BACKEND_PROFILE": (
        "`1` prints per-op timings from the torch.compile backend."
    ),
    "TORCH_MOJO_BACKEND_PTXAS_AUTO": (
        "Set by the package, not by you: the MODULAR_NVPTX_COMPILER_PATH it "
        "chose itself, or `<max built-in>` when it chose to leave that unset "
        "for MAX's own compiler. Every child process inherits the environment, and this "
        "is what lets one tell an inherited automatic choice from a setting "
        "of yours, which is never overridden."
    ),
    "TORCH_MOJO_BACKEND_PTXAS_CHECK": (
        "`0` downgrades the refusal to register a device whose ptxas cannot "
        "assemble for this driver and GPU into a warning, for a machine whose "
        "assembler rules we got wrong. The build then fails on its own terms."
    ),
    "TORCH_MOJO_BACKEND_RCCL_LIB": (
        "Absolute path of librccl.so.1, overriding the ROCm search order "
        "($ROCM_PATH, /opt/rocm, then the system loader)."
    ),
    "TORCH_MOJO_BACKEND_TRACE": (
        "`0` silences the build-timing and collectives trace lines that are "
        "printed on stderr by default."
    ),
    "TORCH_MOJO_BACKEND_TRITON": (
        "`0` leaves Triton alone: the mojo device does not install its Triton "
        "driver hook, so Triton keeps whatever backend it found."
    ),
    "TORCH_MOJO_BACKEND_VERBOSE": (
        "`1` prints the graph structures the torch.compile backend builds."
    ),
    "TORCH_MOJO_BACKEND_WERROR": (
        "`1` makes every Mojo build fail on a compiler warning. Off by "
        "default — a newer toolchain warning about something new must not "
        "cost a user their device — and on under pytest."
    ),
    # -- torch-mojo-backend: legacy aliases -------------------------------
    # Read alongside the TORCH_MOJO_BACKEND_ spellings above; either one being
    # truthy turns the flag on (flags.py).
    "PYTORCH_MOJO_BACKEND_DEBUG_GRAPH": (
        "Legacy alias for TORCH_MOJO_BACKEND_DEBUG_GRAPH."
    ),
    "PYTORCH_MOJO_BACKEND_PROFILE": "Legacy alias for TORCH_MOJO_BACKEND_PROFILE.",
    "PYTORCH_MOJO_BACKEND_VERBOSE": "Legacy alias for TORCH_MOJO_BACKEND_VERBOSE.",
    # -- torch-mojo-backend: test and benchmark hooks ---------------------
    "TORCH_MOJO_BACKEND_BENCH_CPU": (
        "`1` lets benchmarks/ run with no accelerator present, for developing "
        "the harness itself. The numbers are not comparable to a GPU run."
    ),
    "TORCH_MOJO_BACKEND_BENCH_DUMP_KEYS": (
        "Path where benchmarks/ collection writes every node's baseline key, "
        "one per line, measuring nothing. Drives test_coverage.py."
    ),
    "TORCH_MOJO_BACKEND_BENCH_UPDATE": (
        "`1`/`improve` or `force`: the --update-baselines modes of "
        "benchmarks/, for callers that cannot pass the flag."
    ),
    "TORCH_MOJO_BACKEND_TESTING": (
        "`1` marks the process as the test suite (tests/conftest.py sets it). "
        "Enables assertions too expensive for production."
    ),
    "TORCH_MOJO_BACKEND_TEST_PEER_COPY": (
        "Test hook forcing the device-to-device copy route ('host', 'trace', "
        "...) instead of letting the backend pick."
    ),
    "TORCH_MOJO_BACKEND_TEST_PEER_GATE_FD": (
        "Test hook: a file descriptor the peer-copy path blocks on, so a test "
        "can hold a copy open and observe the state around it."
    ),
    # -- mojoccl: transport selection -------------------------------------
    "MOJOCCL_NET": (
        "`verbs` or `fabric` pins the inter-node transport. Unset probes: "
        "verbs first (the measured path), then libfabric."
    ),
    "MOJOCCL_LIBFABRIC": (
        "Absolute path of libfabric.so.1, overriding the search. Unset tries "
        "the system loader, then the Cray installation."
    ),
    "MOJOCCL_SOCKET_IFNAME": (
        "Network interface the TCP bootstrap binds to, when the automatic "
        "choice picks the wrong one. Unset prefers an up, non-loopback "
        "interface with a default route."
    ),
    "MOJOCCL_BOOTSTRAP_TIMEOUT_S": (
        "Seconds the TCP bootstrap waits for every rank to check in; defaults to 120."
    ),
    # -- mojoccl: InfiniBand / verbs --------------------------------------
    "MOJOCCL_IB_HCA": (
        "Keeps only the named IB device, when a host has several and the "
        "automatic pick is wrong. Unset uses GPU/NIC PCI affinity and "
        "the local rank as a tiebreaker."
    ),
    "MOJOCCL_IB_RELAXED_ORDERING": (
        "`0` registers memory regions without IBV_ACCESS_RELAXED_ORDERING, "
        "for fabrics where it misbehaves. Enabled by default."
    ),
    "MOJOCCL_IB_TIMEOUT_S": (
        "Seconds a collective waits for its peers before raising the abort "
        "word. Defaults to 60. Raising it releases every waiter, so it is "
        "process-wide."
    ),
    "MOJOCCL_IB_TRACE": (
        "`1` prints what each rank negotiated — transport, device, ports, "
        "credit counters — one line per communicator. Defaults to `0`."
    ),
    # -- mojoccl: libfabric -----------------------------------------------
    "MOJOCCL_FABRIC_DOMAIN": (
        "Keeps only the named libfabric domain, so one process per NIC can "
        "each drive their own (`cxi0`, `cxi1`, ...). Unset uses PCI affinity, "
        "then the local rank modulo the available domains."
    ),
    "MOJOCCL_FABRIC_PROVIDER": "libfabric provider name; defaults to `cxi`.",
    # -- mojoccl: NVLink SHARP (multicast) ---------------------------------
    "MOJOCCL_NVLS": (
        "`0` turns the NVLink-SHARP multicast path off, region and all. "
        "Enabled by default when every rank supports it."
    ),
    "MOJOCCL_REGION_MB": (
        "Size of the registered staging region, in MiB. Any positive 4 KiB "
        "multiple; rounded up to the allocation granularity. Defaults to "
        "64 MiB on gfx942 (MI300A), 256 MiB elsewhere, including NVIDIA."
    ),
    # -- mojoccl: standalone probe artifacts (not library controls) --------
    "MOJOCCL_LIBRARY": (
        "Path of the prebuilt mojoccl library for the standalone stream-order "
        "probe. Required by that probe; the production library never reads it."
    ),
    "MOJOCCL_STATE_PROBE_LIBRARY": (
        "Path of the prebuilt communicator-state helper for the standalone "
        "stream-order probe. Required by that probe; not a library control."
    ),
}

# Variables owned by torch, MAX, the vendor runtimes or the OS that this
# project reads. Listed so the scanner can tell "read but not ours" from
# "forgotten"; they take no part in the typo check, since a misspelling of
# one of these is not ours to diagnose.
FOREIGN_ENV_VARS: dict[str, str] = {
    "CUDA_VISIBLE_DEVICES": "NVIDIA runtime device visibility; narrowed per rank.",
    "CXX": "C++ compiler for the shim; tried before c++, g++, clang++.",
    "HIP_PATH": "Older spelling of ROCM_PATH, searched for the HIP runtime.",
    "HIP_VISIBLE_DEVICES": "HIP runtime device visibility; narrowed per rank.",
    "LOCAL_RANK": "Set by torchrun; picks this worker's GPU out of the visible set.",
    "MODULAR_HOME": (
        "The Mojo compiler's module cache. Defaulted to node-local scratch so "
        "concurrent compilers on an NFS $HOME cannot evict each other."
    ),
    "MODULAR_CACHE_DIR": (
        "The Mojo compiler's compile cache (default $MODULAR_HOME/cache). "
        "Defaulted to node-local scratch for the same reason as MODULAR_HOME."
    ),
    "MODULAR_MOJO_MAX_IMPORT_PATH": (
        "The comma-separated import path the Mojo toolchain -- MAX's in-process "
        "graph compiler included -- resolves `from X import` along. Extended at "
        "package import with the Mojo source root (`_mojo_import_path.py`) so the "
        "torch.compile backend's custom ops can call the eager kernels; a value "
        "you set is extended, never replaced."
    ),
    "CUDA_HOME": "A CUDA toolkit root, searched for a ptxas to assemble with.",
    "CUDA_PATH": "Older spelling of CUDA_HOME, searched the same way.",
    "MODULAR_NVPTX_COMPILER_PATH": (
        "The ptxas MAX assembles with. Defaulted to the newest on this "
        "machine that suits both the driver and the GPU, and left unset when "
        "that is MAX's own compiler. An explicit value wins and is only checked; "
        "`torch-mojo-backend ptxas` shows the choice and the alternatives."
    ),
    "ROCM_PATH": "ROCm install root, searched for the HIP runtime and librccl.",
    "ROCR_VISIBLE_DEVICES": "HSA-level device visibility; narrowed per rank.",
    "TMPDIR": "Scratch directory for the Mojo loader's intermediate build files.",
    "TORCHINDUCTOR_COMPILE_THREADS": (
        "Inductor's worker count. Left alone when the user set it; otherwise "
        "forced to 1 for worker start methods that would fork the device."
    ),
    "TRITON_PTXAS_PATH": (
        "The ptxas Triton assembles with. Aligned with MAX's, but only when "
        "it still holds torch's own default."
    ),
}


class UnknownEnvVarWarning(UserWarning):
    """A `TORCH_MOJO_BACKEND_*` / `MOJOCCL_*` variable nothing reads is set."""


def known_env_vars() -> frozenset[str]:
    """Every variable name this project reads, ours and other people's."""
    return frozenset(OWN_ENV_VARS) | frozenset(FOREIGN_ENV_VARS)


def suggest(name: str) -> str | None:
    """The real variable `name` was most likely a typo of, if any.

    Matching runs on the part *after* the shared prefix. Whole names all
    start with the same twenty-odd characters, so comparing them whole scores
    every pair high enough to clear any cutoff and would answer a confident
    suggestion for a name that resembles nothing.
    """
    for prefix in CHECKED_PREFIXES:
        if not name.startswith(prefix):
            continue
        suffixes = {
            known[len(prefix) :]: known
            for known in OWN_ENV_VARS
            if known.startswith(prefix)
        }
        close = difflib.get_close_matches(name[len(prefix) :], suffixes, 1, 0.6)
        return suffixes[close[0]] if close else None
    return None


def unknown_env_vars(
    environment: Mapping[str, str] | None = None,
) -> list[tuple[str, str | None]]:
    """The `(name, suggestion)` of every variable set in `environment` that
    is spelled like one of ours but is read by nothing."""
    environment = environ if environment is None else environment
    return [
        (name, suggest(name))
        for name in sorted(environment)
        if name.startswith(CHECKED_PREFIXES) and name not in OWN_ENV_VARS
    ]


def warn_about_unknown_env_vars(environment: Mapping[str, str] | None = None):
    """Warn about each misspelled variable. Called by `register_mojo_devices()`
    under its lock, so a normal process warns once; a registration that failed
    and is retried from another source line warns again, which is the right
    way round -- the second attempt is a second chance to read it.

    A warning, not an error: the environment is not always the user's to
    clean — a scheduler prologue or a shared module file can export anything
    — and refusing to register the device over a stray variable would be a
    far worse trade than a line on stderr.
    """
    for name, suggestion in unknown_env_vars(environment):
        hint = (
            f" Did you mean {suggestion}?"
            if suggestion
            else " See torch_mojo_backend/env_vars.py for the ones that exist."
        )
        warnings.warn(
            f"{name} is set, but torch-mojo-backend reads no such environment"
            f" variable, so it has no effect.{hint}",
            UnknownEnvVarWarning,
            stacklevel=3,
        )
