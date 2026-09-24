"""`torch.mojo`: the device module torch looks up by backend name.

Devices, streams and events are torch's own generic ones (the C++ shim
implements the device guard), so this module is the thin torch.cuda-shaped
surface: device selection, RNG state, synchronize, amp dtypes, memory info.
"""

from __future__ import annotations

import ctypes
from collections import OrderedDict
from dataclasses import dataclass
from typing import Never, TypeAlias

import torch

from torch_mojo_backend import native

_PRIVATEUSE1 = 20  # c10::DeviceType::PrivateUse1

DeviceLike = "int | str | torch.device | None"


def _index(device: int | str | torch.device | None) -> int:
    if device is None:
        idx = current_device()
    elif isinstance(device, int):
        idx = device
    else:
        d = torch.device(device)
        if d.type != "mojo":
            raise ValueError(f"expected a mojo device, got {d}")
        idx = current_device() if d.index is None else d.index
    if idx < 0 or idx >= device_count():
        raise ValueError(
            f"Invalid mojo device index {idx}; expected 0 <= index < {device_count()}"
        )
    return idx


MemoryStats: TypeAlias = dict[str, "int | MemoryStats"]


def _require_memory_binding(name: str, minimum_version: str = "2.9"):
    """Explain unsupported torch builds before accessing their memory bindings."""
    if not hasattr(torch._C, name):
        raise RuntimeError(
            f"torch {torch.__version__} is missing torch._C.{name}; "
            f"this Mojo memory API needs torch's accelerator memory bindings "
            f"(torch>={minimum_version}; torch-mojo-backend requires torch>=2.10)."
        )


def memory_stats_as_nested_dict(
    device: int | str | torch.device | None = None,
) -> MemoryStats:
    """Allocator statistics from torch's DeviceAllocator binding.

    Only ``all`` is populated. Requested and reserved bytes equal allocated
    bytes: we neither round requests nor own MAX's caching arena. Pool,
    segment, active, inactive-split and oversize statistics are structurally
    zero; they describe allocator concepts this backend does not implement.
    """
    _require_memory_binding("_accelerator_getDeviceStats")
    return torch._C._accelerator_getDeviceStats(_index(device))


def memory_stats(
    device: int | str | torch.device | None = None,
) -> OrderedDict[str, int]:
    """Sorted, flattened statistics with torch.cuda's key spelling."""
    pairs = []

    def flatten(prefix: str, value: int | MemoryStats):
        if isinstance(value, dict):
            for key, child in value.items():
                flatten(f"{prefix}.{key}" if prefix else key, child)
        else:
            pairs.append((prefix, value))

    flatten("", memory_stats_as_nested_dict(device))
    return OrderedDict(sorted(pairs))


def memory_allocated(device: int | str | torch.device | None = None) -> int:
    """Bytes in live storage allocated by this backend on this device."""
    return memory_stats(device)["allocated_bytes.all.current"]


def max_memory_allocated(device: int | str | torch.device | None = None) -> int:
    """Peak live storage bytes since the last peak reset."""
    return memory_stats(device)["allocated_bytes.all.peak"]


def memory_reserved(device: int | str | torch.device | None = None) -> int:
    """Equals memory_allocated: MAX's arena reservation is not exposed."""
    return memory_stats(device)["reserved_bytes.all.current"]


def max_memory_reserved(device: int | str | torch.device | None = None) -> int:
    """Equals max_memory_allocated; this backend has no caching arena."""
    return memory_stats(device)["reserved_bytes.all.peak"]


def reset_peak_memory_stats(device: int | str | torch.device | None = None):
    """Reset peaks to current usage, including storage still alive."""
    _require_memory_binding("_accelerator_resetPeakStats")
    torch._C._accelerator_resetPeakStats(_index(device))


def reset_accumulated_memory_stats(device: int | str | torch.device | None = None):
    """Zero cumulative allocation, free, retry and OOM counts; keep current/peak."""
    _require_memory_binding("_accelerator_resetAccumulatedStats")
    torch._C._accelerator_resetAccumulatedStats(_index(device))


def empty_cache():
    """Release completed deferred host staging buffers on every device.

    Does not synchronize or return device memory to the OS: MAX owns the
    arena and exposes no trim API. Live tensor storage is unaffected.
    Like torch.accelerator.empty_cache, this is a no-op in a forked child.
    """
    _require_memory_binding("_accelerator_emptyCache")
    torch._C._accelerator_emptyCache()


def mem_get_info(device: int | str | torch.device | None = None) -> tuple[int, int]:
    """MAX's reported (free, total) bytes, separate from our storage counters.

    A device whose MAX runtime cannot supply capacity raises explicitly
    rather than returning fabricated zeros.
    """
    _require_memory_binding("_accelerator_getMemoryInfo", minimum_version="2.10")
    return torch._C._accelerator_getMemoryInfo(_index(device))


def memory_summary(
    device: int | str | torch.device | None = None, abbreviated: bool = False
) -> str:
    """Format allocator counters in bytes/counts, with their Mojo meanings."""
    idx = _index(device)
    stats = memory_stats(idx)
    lines = [
        f"Mojo memory summary, device {idx} (bytes unless marked count)",
        f"{'Metric':<28} {'Current':>16} {'Peak':>16} {'Allocated':>16} {'Freed':>16}",
    ]
    metrics = [
        ("Allocated memory", "allocated_bytes"),
        ("Reserved memory", "reserved_bytes"),
    ]
    if not abbreviated:
        metrics += [
            ("Requested memory", "requested_bytes"),
            ("Allocations (count)", "allocation"),
        ]
    for label, metric in metrics:
        values = " ".join(
            f"{stats[f'{metric}.all.{field}']:>16,}"
            for field in ("current", "peak", "allocated", "freed")
        )
        lines.append(f"{label:<28} {values}")
    for key in ("num_device_alloc", "num_device_free", "num_alloc_retries", "num_ooms"):
        lines.append(f"{key}: {stats[key]:,}")
    lines += [
        "Reserved = allocated: MAX owns the arena; its cached slack is not exposed.",
        "Pool, segment, active, inactive_split and oversize breakdowns are structurally zero.",
    ]
    return "\n".join(lines)


@dataclass(frozen=True)
class MojoDeviceProperties:
    """Cached MAX properties; unavailable fields are None, clock_rate is kHz.

    Inductor's separate MojoDeviceProperties in inductor.py has the exact
    field set its Triton autotuner requires; this is the public MAX view.
    """

    name: str
    major: int | None
    minor: int | None
    total_memory: int | None
    multi_processor_count: int | None
    max_threads_per_multi_processor: int | None
    warp_size: int | None
    regs_per_multiprocessor: int | None
    gcnArchName: str | None
    api: str
    arch_name: str | None
    max_threads_per_block: int | None
    regs_per_block: int | None
    shared_memory_per_block: int | None
    shared_memory_per_block_optin: int | None
    shared_memory_per_multiprocessor: int | None
    max_blocks_per_multi_processor: int | None
    clock_rate: int | None
    max_grid_dim_x: int | None


def _gfx_version(arch: str) -> tuple[int | None, int | None]:
    """HIP's `hipDeviceProp_t` (major, minor): the gfx ISA version, whose last
    two hex digits are minor and stepping ("gfx90a" -> 9, 0; "gfx1100" -> 11, 0)."""
    digits = arch.split(":")[0].removeprefix("gfx")
    if len(digits) < 3 or not digits[:-2].isdigit():
        return None, None
    return int(digits[:-2]), int(digits[-2], 16)


def get_device_properties(
    device: int | str | torch.device | None = None,
) -> MojoDeviceProperties:
    """Properties cached per device in Mojo."""
    idx = _index(device)
    values = (ctypes.c_int64 * 16)()  # tmb.h TMB_DEVICE_PROPS_SLOTS
    text = ctypes.create_string_buffer(4096)
    fn = native.shim().tmb_device_properties
    fn.argtypes = [
        ctypes.c_int32,
        ctypes.POINTER(ctypes.c_int64),
        ctypes.c_int32,
        ctypes.POINTER(ctypes.c_char),
        ctypes.c_int32,
    ]
    fn.restype = ctypes.c_int32
    if fn(idx, values, len(values), text, len(text)):
        raise RuntimeError(native.last_error())
    name, api, arch = (
        part.decode("utf-8", errors="replace") for part in text.raw.split(b"\0")[:3]
    )

    def value(slot: int) -> int | None:
        return values[slot] if values[slot] >= 0 else None

    major, minor = value(0), value(1)
    if api == "hip":
        major, minor = _gfx_version(arch)
    return MojoDeviceProperties(
        name=name,
        major=major,
        minor=minor,
        total_memory=value(2),
        multi_processor_count=value(3),
        max_threads_per_multi_processor=value(4),
        warp_size=value(5),
        regs_per_multiprocessor=value(6),
        gcnArchName=(arch or None) if api == "hip" else None,
        api=api,
        arch_name=arch or None,
        max_threads_per_block=value(7),
        regs_per_block=value(8),
        shared_memory_per_block=value(9),
        shared_memory_per_block_optin=value(10),
        shared_memory_per_multiprocessor=value(11),
        max_blocks_per_multi_processor=value(12),
        clock_rate=value(13),
        max_grid_dim_x=value(14),
    )


def get_device_name(device: int | str | torch.device | None = None) -> str:
    """MAX's device name."""
    return get_device_properties(device).name


def get_device_capability(
    device: int | str | torch.device | None = None,
) -> tuple[int | None, int | None]:
    """torch.cuda's (major, minor) on CUDA and ROCm; (None, None) on Metal."""
    props = get_device_properties(device)
    return props.major, props.minor


def memory_snapshot() -> Never:
    raise NotImplementedError(
        "MAX does not expose its arena blocks or allocation history"
    )


def _record_memory_history(
    enabled: str | None = "all",
    context: str | None = "all",
    stacks: str = "all",
    max_entries: int = 9223372036854775807,
    device: int | str | torch.device | None = None,
    clear_history: bool = False,
    compile_context: bool = False,
    global_record_annotations: bool = False,
    skip_actions: list[str] | None = None,
) -> Never:
    _index(device)
    raise NotImplementedError(
        "Mojo tracks counters, not allocation histories or stack traces"
    )


def _dump_snapshot(
    filename: str = "dump_snapshot.pickle", augment_with_fx_traces: bool = False
) -> Never:
    raise NotImplementedError("MAX does not expose an allocator snapshot to dump")


def caching_allocator_alloc(
    size: int,
    device: int | str | torch.device | None = None,
    stream: int | torch.Stream | None = None,
) -> Never:
    _index(device)
    raise NotImplementedError(
        "Mojo allocations require owned storage handles, not raw pointers"
    )


def caching_allocator_delete(mem_ptr: int) -> Never:
    raise NotImplementedError("Mojo frees owned storage handles, not raw pointers")


def set_per_process_memory_fraction(
    fraction: float, device: int | str | torch.device | None = None
) -> Never:
    _index(device)
    raise NotImplementedError(
        "MAX owns the arena and exposes no per-process memory limit"
    )


def list_gpu_processes(device: int | str | torch.device | None = None) -> Never:
    _index(device)
    raise NotImplementedError("MAX does not expose per-process GPU memory usage")


def host_memory_stats() -> Never:
    raise NotImplementedError("MAX does not expose pinned host allocator statistics")


def host_memory_stats_as_nested_dict() -> Never:
    raise NotImplementedError("MAX does not expose pinned host allocator statistics")


def reset_accumulated_host_memory_stats() -> Never:
    raise NotImplementedError("MAX does not expose pinned host allocator statistics")


def reset_peak_host_memory_stats() -> Never:
    raise NotImplementedError("MAX does not expose pinned host allocator statistics")


def is_available() -> bool:
    return native.is_registered() and native.device_count() > 0


def is_initialized() -> bool:
    return native.is_registered()


def _lazy_init():
    """torch's `device_lazy_init(PrivateUse1)` hook: it imports `torch.mojo`
    and calls this, then marks the device initialized.

    Nothing to do here (registration already built everything), but the method
    must EXIST: without it torch never sets its per-device-type initialized
    flag, and `torch._C._accelerator_synchronizeDevice` -- which returns early
    for an uninitialized lazy-init device -- silently does nothing. That made
    `torch.accelerator.synchronize()` and `torch.mojo.synchronize()` no-ops,
    so a readback after work on a non-default stream could race it.
    """


def _is_in_bad_fork() -> bool:
    """True in a child forked after registration, where the runtime is
    unusable (agents_docs/native_backend.md, "Fork"). torch.manual_seed consults it
    before seeding this device from a forked DataLoader worker."""
    return native.is_registered() and bool(native.shim().tmb_is_in_bad_fork())


def device_count() -> int:
    return native.device_count()


def current_device() -> int:
    return int(native.shim().tmb_current_device())


def set_device(device: int | str | torch.device):
    idx = _index(device)
    if idx < 0 or idx >= device_count():
        raise ValueError(f"Invalid device index {idx}")
    native.shim().tmb_set_current_device(ctypes.c_int32(idx))


class device:
    """Context manager swapping the current device (torch.serialization needs
    it on the backend module for map_location="mojo")."""

    def __init__(self, device: int | str | torch.device | None):
        self.idx = -1 if device is None else _index(device)
        self.prev = -1

    def __enter__(self):
        self.prev = current_device()
        if self.idx >= 0:
            set_device(self.idx)

    def __exit__(self, *exc: object) -> bool:
        if self.idx >= 0:
            set_device(self.prev)
        return False


def synchronize(device: int | str | torch.device | None = None):
    torch._C._accelerator_synchronizeDevice(_index(device))


def manual_seed(seed: int):
    native.shim().tmb_rng_manual_seed(
        ctypes.c_int32(current_device()), ctypes.c_uint64(seed & ((1 << 64) - 1))
    )


def manual_seed_all(seed: int):
    native.shim().tmb_rng_manual_seed(
        ctypes.c_int32(-1), ctypes.c_uint64(seed & ((1 << 64) - 1))
    )


def seed():
    manual_seed(
        int.from_bytes(torch.randint(0, 2**62, (1,)).numpy().tobytes(), "little")
    )


def seed_all():
    manual_seed_all(
        int.from_bytes(torch.randint(0, 2**62, (1,)).numpy().tobytes(), "little")
    )


def initial_seed() -> int:
    return int.from_bytes(bytes(get_rng_state().tolist()[:8]), "little")


def get_rng_state(device: int | str | torch.device | None = None) -> torch.Tensor:
    """16 bytes: Philox seed then counter, little-endian."""
    buf = (ctypes.c_uint8 * 16)()
    native.shim().tmb_rng_get_state(ctypes.c_int32(_index(device)), buf)
    return torch.tensor(list(buf), dtype=torch.uint8)


def set_rng_state(
    new_state: torch.Tensor, device: int | str | torch.device | None = None
):
    if not isinstance(new_state, torch.Tensor):
        raise TypeError("Mojo RNG state must be a torch.Tensor")
    state = new_state.detach().cpu().contiguous()
    if state.dtype != torch.uint8 or state.numel() != 16:
        raise ValueError("Mojo RNG state must be a 16-element uint8 tensor")
    buf = (ctypes.c_uint8 * 16)(*state.reshape(-1).tolist())
    native.shim().tmb_rng_set_state(ctypes.c_int32(_index(device)), buf)


def get_rng_state_all() -> list[torch.Tensor]:
    return [get_rng_state(i) for i in range(device_count())]


def set_rng_state_all(states: list[torch.Tensor]):
    for i, s in enumerate(states):
        set_rng_state(s, i)


def get_amp_supported_dtype() -> list[torch.dtype]:
    return [torch.float16, torch.bfloat16]


def is_bf16_supported() -> bool:
    return True


def current_stream(device: int | str | torch.device | None = None) -> torch.Stream:
    return torch.accelerator.current_stream(_index(device))


def default_stream(device: int | str | torch.device | None = None) -> torch.Stream:
    """Stream id 0 is every device's default stream."""
    return torch.Stream(
        stream_id=0, device_index=_index(device), device_type=_PRIVATEUSE1
    )


def stream_native_handle(stream: torch.Stream) -> int:
    """The vendor (CUDA/HIP) handle of a mojo stream, for code that launches
    on it outside torch (Triton). `torch.Stream.native_handle` is the same
    value on torch >= 2.11; this works on any version."""
    fn = native.shim().tmb_stream_native_handle
    fn.restype = ctypes.c_void_p
    fn.argtypes = [ctypes.c_int32, ctypes.c_int64]
    return fn(stream.device_index, stream.stream_id) or 0


def set_stream(stream: torch.Stream):
    torch.accelerator.set_stream(stream)


class StreamContext:
    def __init__(self, stream: torch.Stream | None):
        self.stream = stream
        self.prev: torch.Stream | None = None
        self.prev_device: int | None = None

    def __enter__(self):
        if self.stream is None:
            return
        # set_stream also makes the stream's device current (torch semantics):
        # both are restored on exit
        self.prev_device = current_device()
        self.prev = torch.accelerator.current_stream(self.stream.device.index)
        torch.accelerator.set_stream(self.stream)

    def __exit__(self, *exc: object) -> bool:
        if self.prev is not None:
            torch.accelerator.set_stream(self.prev)
        if self.prev_device is not None:
            set_device(self.prev_device)
        return False


def stream(stream: torch.Stream | None) -> StreamContext:
    return StreamContext(stream)


Stream = torch.Stream
Event = torch.Event
