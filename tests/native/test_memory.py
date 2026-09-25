"""Storage accounting and MAX properties through the public device APIs."""

import gc
import os
import select
import subprocess
import sys
import threading
import time
import weakref
from collections import OrderedDict
from concurrent.futures import ThreadPoolExecutor
from dataclasses import FrozenInstanceError
from pathlib import Path
from queue import Queue
from statistics import median

import pytest
import torch
from torch._subclasses.fake_tensor import FakeTensorMode
from torch.utils.data import DataLoader, TensorDataset

from tests.native.conftest import side_stream_or_skip
from tests.native.test_fork import _in_forked_child
from torch_mojo_backend import get_accelerators, register_mojo_devices
from torch_mojo_backend.inductor import MojoInterface
from torch_mojo_backend.native import device_module

_DEVICE_APIS = (
    "memory_allocated",
    "max_memory_allocated",
    "memory_reserved",
    "max_memory_reserved",
    "memory_stats",
    "memory_stats_as_nested_dict",
    "reset_peak_memory_stats",
    "reset_accumulated_memory_stats",
    "mem_get_info",
    "memory_summary",
    "get_device_properties",
    "get_device_name",
    "get_device_capability",
)
_ACCELERATOR_APIS = (
    "memory_allocated",
    "max_memory_allocated",
    "memory_reserved",
    "max_memory_reserved",
    "memory_stats",
    "reset_peak_memory_stats",
    "reset_accumulated_memory_stats",
    "get_memory_info",
)


def _settle(device: str | torch.device | int):
    """Release cycles, then finish every stream on the owning device (plan B/F)."""
    gc.collect()
    device_module.synchronize(device)


def _fresh(code: str, *args: str):
    """Use the project interpreter, with a deadline only to detect hangs."""
    subprocess.run([sys.executable, "-c", code, *args], check=True, timeout=180)


def _assert_shared(device: str | torch.device | int, expected: int):
    """Integration checks AFTER a test-owned byte ledger establishes truth."""
    for api in (device_module, torch.accelerator):
        assert api.memory_allocated(device) == expected
        assert api.memory_reserved(device) == expected
        stats = api.memory_stats(device)
        assert stats["allocated_bytes.all.current"] == expected
        assert api.max_memory_allocated(device) == stats["allocated_bytes.all.peak"]
        assert api.max_memory_reserved(device) == stats["reserved_bytes.all.peak"]
    assert device_module.memory_stats(device) == torch.accelerator.memory_stats(device)


def test_odd_sizes_non_lifo_ledger(mojo_device: str):
    """01: the expected 5641 bytes come from literal requests, not storage APIs."""
    _settle(mojo_device)
    base = device_module.memory_allocated(mojo_device)
    sizes = (1, 7, 511, 512, 513, 4097)
    tensors = {
        i: torch.empty(size, dtype=torch.uint8, device=mojo_device)
        for i, size in enumerate(sizes)
    }
    assert sum(sizes) == 5641
    assert len({x.data_ptr() for x in tensors.values()}) == len(sizes)
    for i, tensor in tensors.items():
        assert tensor.untyped_storage().nbytes() == sizes[i]
    del tensor
    live = 5641
    _assert_shared(mojo_device, base + live)
    for i in (2, 0, 5, 1, 4, 3):
        del tensors[i]
        live -= sizes[i]
        device_module.synchronize(mojo_device)
        _assert_shared(mojo_device, base + live)


@pytest.mark.parametrize(
    "dtype,width",
    [
        (torch.bool, 1),
        (torch.int16, 2),
        (torch.float32, 4),
        (torch.int64, 8),
        (torch.complex64, 8),
    ],
)
def test_scalar_and_zero_storage_history(
    mojo_device: str, dtype: torch.dtype, width: int
):
    """01/20: a scalar owns one element; fresh empty storages own no bytes."""
    _settle(mojo_device)
    before = device_module.memory_stats(mojo_device)
    if dtype.is_complex:
        # Complex ScalarTypes are not supported by the native backend. A
        # declined factory must not manufacture even a zero-storage charge.
        for shape in ((), (0,), (4, 0, 7)):
            with pytest.raises(NotImplementedError, match="dtype.*not supported"):
                torch.empty(shape, dtype=dtype, device=mojo_device)
            assert device_module.memory_stats(mojo_device) == before
            with pytest.raises(NotImplementedError, match="dtype.*not supported"):
                torch.empty_strided(
                    shape, (1,) * len(shape), dtype=dtype, device=mojo_device
                )
            assert device_module.memory_stats(mojo_device) == before
        return
    zeros = [
        torch.empty(shape, dtype=dtype, device=mojo_device)
        for shape in ((0,), (4, 0, 7))
        for _ in range(16)
    ]
    assert all(x.untyped_storage().nbytes() == 0 for x in zeros)
    assert device_module.memory_stats(mojo_device) == before
    scalar = torch.empty((), dtype=dtype, device=mojo_device)
    _assert_shared(mojo_device, before["allocated_bytes.all.current"] + width)
    assert scalar.untyped_storage().nbytes() == width
    del zeros, scalar
    _settle(mojo_device)
    _assert_shared(mojo_device, before["allocated_bytes.all.current"])


def test_expansion_and_empty_alias_own_storage(mojo_device: str):
    """02/20: even an empty last alias retains the entire nonempty storage."""
    _settle(mojo_device)
    base = device_module.memory_allocated(mojo_device)
    x = torch.ones(1, device=mojo_device)
    expanded = x.expand(2**30)
    detached = x.detach()
    empty = x[1:]
    before = device_module.memory_stats(mojo_device)
    assert before["allocated_bytes.all.current"] == base + 4
    assert expanded.numel() == 2**30 and empty.numel() == 0
    clone = x.clone()
    _assert_shared(mojo_device, base + 8)
    del clone, x, expanded, detached
    _settle(mojo_device)
    _assert_shared(mojo_device, base + 4)
    del empty
    _settle(mojo_device)
    _assert_shared(mojo_device, base)


@pytest.mark.parametrize(
    "first",
    ["allocation", "generic_allocation", "info", "properties", "name", "capability"],
)
def test_registration_and_first_memory_operation(first: str, mojo_gpu_available: bool):
    """09/10: registration eagerly initializes MAX, before any storage exists.

    There is no public registered-but-cold mojo allocator. Before registration
    torch.mojo is absent and generic accelerator queries target another backend.
    The genuinely cold generic early returns are exercised in the fork test.
    """
    if not mojo_gpu_available:
        pytest.skip("You do not have a GPU supported by MAX")
    _fresh(
        """
import sys
import torch
from torch_mojo_backend import native, register_mojo_devices
assert not native.is_registered()
assert not hasattr(torch, "mojo")
assert torch._C._get_privateuse1_backend_name() != "mojo"
register_mojo_devices()
assert native.is_registered()
assert torch._C._accelerator_isAllocatorInitialized()
assert torch.accelerator.current_accelerator().type == "mojo"
device = "mojo:0"
first = sys.argv[1]
if first in ("allocation", "generic_allocation"):
    x = torch.empty(513, dtype=torch.uint8, device=device)
    api = torch.accelerator if first == "generic_allocation" else torch.mojo
    assert api.memory_allocated(device) == 513
    assert api.memory_stats(device)["allocation.all.current"] == 1
    del x
elif first == "info":
    free, total = torch.accelerator.get_memory_info(device)
    assert 0 < free <= total
elif first == "properties":
    assert torch.mojo.get_device_properties(device).name
elif first == "name":
    assert torch.mojo.get_device_name(device)
else:
    capability = torch.mojo.get_device_capability(device)
    assert isinstance(capability, tuple) and len(capability) == 2
for api in (torch.mojo, torch.accelerator):
    stats = api.memory_stats(device)
    assert stats and all(type(v) is int for v in stats.values())
    assert stats["allocated_bytes.all.current"] == 0
    assert api.memory_allocated(device) == api.memory_reserved(device) == 0
    api.reset_peak_memory_stats(device)
    api.reset_accumulated_memory_stats(device)
    api.empty_cache()
    # Initialization is a runtime-lifecycle flag, not "has allocated before".
    assert torch._C._accelerator_isAllocatorInitialized()
    try:
        api.memory_stats("cpu")
    except ValueError:
        pass
    else:
        raise AssertionError("initialized allocator must validate device arguments")
x = torch.empty(4097, dtype=torch.uint8, device=device)
assert torch.mojo.memory_allocated(device) == 4097
assert torch.accelerator.memory_allocated(device) == 4097
del x
torch.mojo.synchronize(device)
assert torch.accelerator.memory_allocated(device) == 0
""",
        first,
    )


def test_schema_and_detached_snapshots(mojo_device: str):
    """07: independently enumerate the C++ binding's schema and nested paths."""
    _settle(mojo_device)
    base = device_module.memory_allocated(mojo_device)
    x = torch.empty(511, dtype=torch.uint8, device=mojo_device)
    expected_keys = {
        f"{family}.{pool}.{field}"
        for family in (
            "allocation",
            "segment",
            "active",
            "inactive_split",
            "allocated_bytes",
            "reserved_bytes",
            "active_bytes",
            "inactive_split_bytes",
            "requested_bytes",
        )
        for pool in ("all", "small_pool", "large_pool")
        for field in ("current", "peak", "allocated", "freed")
    }
    expected_keys.update(
        f"{family}.{field}"
        for family in ("oversize_allocations", "oversize_segments")
        for field in ("current", "peak", "allocated", "freed")
    )
    expected_keys.update(
        (
            "num_alloc_retries",
            "num_ooms",
            "num_sync_all_streams",
            "num_device_alloc",
            "num_device_free",
            "max_split_size",
        )
    )
    flat = device_module.memory_stats(mojo_device)
    nested = device_module.memory_stats_as_nested_dict(mojo_device)
    assert set(flat) == expected_keys
    assert all(type(value) is int and value >= 0 for value in flat.values())
    for path, value in flat.items():
        leaf = nested
        for component in path.split("."):
            assert isinstance(leaf, dict)
            leaf = leaf[component]
        assert leaf == value
    _assert_shared(mojo_device, base + 511)
    saved = flat.copy()
    flat["allocated_bytes.all.current"] = -7
    allocation_bytes = nested["allocated_bytes"]
    assert isinstance(allocation_bytes, dict)
    all_pool = allocation_bytes["all"]
    assert isinstance(all_pool, dict)
    all_pool["current"] = -13
    assert device_module.memory_stats(mojo_device) == saved
    del x
    _settle(mojo_device)
    assert saved["allocated_bytes.all.current"] == base + 511
    _assert_shared(mojo_device, base)


@pytest.mark.parametrize(
    "form", ["int", "string", "device", "omitted", "none", "bare_string", "bare_device"]
)
def test_valid_device_arguments_and_resets(mojo_device: str, form: str):
    """11/06: literal ownership anchors every spelling, including keywords."""
    idx = torch.device(mojo_device).index
    assert idx is not None
    explicit = form in ("int", "string", "device")
    argument = {
        "int": idx,
        "string": mojo_device,
        "device": torch.device(mojo_device),
        "none": None,
        "bare_string": "mojo",
        "bare_device": torch.device("mojo"),
        "omitted": None,
    }[form]
    _settle(mojo_device)
    base = device_module.memory_allocated(idx)
    other = (idx + 1) % device_module.device_count()
    with device_module.device(other if explicit else idx):
        x = torch.empty(513 + idx, dtype=torch.uint8, device=mojo_device)
        for api, names in (
            (device_module, _DEVICE_APIS),
            (torch.accelerator, _ACCELERATOR_APIS),
        ):
            for name in names:
                fn = getattr(api, name)
                before = device_module.memory_stats(idx)
                if form == "omitted":
                    got = fn()
                elif api is device_module:
                    got = fn(device=argument)
                else:
                    got = fn(argument)
                if name == "reset_peak_memory_stats":
                    after = device_module.memory_stats(idx)
                    for key, value in before.items():
                        assert after[key] == (
                            before[key.removesuffix("peak") + "current"]
                            if key.endswith(".peak")
                            else value
                        )
                elif name == "reset_accumulated_memory_stats":
                    after = device_module.memory_stats(idx)
                    for key, value in before.items():
                        assert after[key] == (
                            0
                            if key.endswith((".allocated", ".freed"))
                            or key.startswith("num_")
                            else value
                        )
                elif name in ("mem_get_info", "get_memory_info"):
                    free, total = got
                    assert type(free) is int and type(total) is int
                    assert 0 < free <= total
                    assert total == device_module.mem_get_info(idx)[1]
                else:
                    assert got == getattr(device_module, name)(idx)
                assert device_module.current_device() == (other if explicit else idx)
                _assert_shared(idx, base + 513 + idx)
        del x
    _settle(idx)
    _assert_shared(idx, base)


@pytest.mark.skipif(len(list(get_accelerators())) < 2, reason="needs two GPUs")
@pytest.mark.parametrize("outer_api", ["mojo", "accelerator"])
def test_nested_contexts_and_per_device_resets(mojo_gpu: str, outer_api: str):
    """13/35: unequal per-device histories reject a single global counter."""
    second = torch.device("mojo:1")
    devices = (mojo_gpu, second)
    for dev in devices:
        _settle(dev)
    bases = [device_module.memory_allocated(dev) for dev in devices]
    x = torch.empty(513, dtype=torch.uint8, device=mojo_gpu)
    y = torch.empty(4097, dtype=torch.uint8, device=second)
    for dev, size in zip(devices, (8193, 16385), strict=True):
        temp = torch.empty(size, dtype=torch.uint8, device=dev)
        del temp
        _settle(dev)
    snapshots = [device_module.memory_stats(dev) for dev in devices]
    original = device_module.current_device()
    outer = (
        device_module.device if outer_api == "mojo" else torch.accelerator.device_index
    )
    inner = (
        torch.accelerator.device_index if outer_api == "mojo" else device_module.device
    )
    with outer(torch.device(mojo_gpu).index):
        assert device_module.memory_allocated() == bases[0] + 513
        with pytest.raises(RuntimeError, match="context exit"):
            with inner(second.index):
                assert device_module.memory_allocated() == bases[1] + 4097
                z = torch.empty(7, dtype=torch.uint8, device="mojo")
                assert z.device == second
                del z
                _settle(second)
                with device_module.device(None):
                    assert device_module.current_device() == second.index
                torch.accelerator.reset_peak_memory_stats(mojo_gpu)
                assert device_module.max_memory_allocated(mojo_gpu) == bases[0] + 513
                assert (
                    device_module.max_memory_allocated(second)
                    == snapshots[1]["allocated_bytes.all.peak"]
                )
                gpu_reset = device_module.memory_stats(mojo_gpu)
                device_module.reset_accumulated_memory_stats()
                assert device_module.memory_stats(mojo_gpu) == gpu_reset
                assert (
                    device_module.memory_stats(second)["allocated_bytes.all.allocated"]
                    == 0
                )
                raise RuntimeError("context exit")
        assert device_module.current_device() == torch.device(mojo_gpu).index
        _assert_shared(mojo_gpu, bases[0] + 513)
    assert device_module.current_device() == original
    stable = [device_module.memory_stats(dev) for dev in devices]
    for dev in (*devices, *reversed(devices)):
        with device_module.device(dev):
            assert [device_module.memory_stats(d) for d in devices] == stable
    del y
    _settle(second)
    _assert_shared(second, bases[1])
    _assert_shared(mojo_gpu, bases[0] + 513)
    del x
    _settle(mojo_gpu)
    _assert_shared(mojo_gpu, bases[0])


def test_memory_info_allocation_pressure(mojo_gpu: str):
    """14: MAX reports an arena budget; free == total is valid when idle.

    Isolate the arena so prior tests' cached chunks cannot satisfy the request.
    A large request exercises actual capacity, independently of live counters.
    """
    _fresh(
        """
import sys
import torch
from torch_mojo_backend import register_mojo_devices
register_mojo_devices()
d = sys.argv[1]
free, total = torch.mojo.mem_get_info(d)
assert type(free) is int and type(total) is int and 0 < free <= total
assert torch.mojo.mem_get_info(d)[1] == total
size = min(2 * 1024**3, free // 4)
assert size > 0
x = torch.empty(size, dtype=torch.uint8, device=d)
torch.mojo.synchronize(d)
after_free, after_total = torch.mojo.mem_get_info(d)
assert after_total == total
assert 0 < after_free <= total
assert free - after_free >= size, (free, after_free, size)
assert torch.accelerator.get_memory_info(d)[1] == total
""",
        mojo_gpu,
    )


def test_capability_contracts_and_inductor_properties(mojo_device: str):
    """15/17/18: public architecture, compiler cc, and dtype support differ."""
    props = device_module.get_device_properties(mojo_device)
    capability = device_module.get_device_capability(mojo_device)
    assert type(capability) is tuple and len(capability) == 2
    if props.api == "cuda":
        assert all(type(part) is int for part in capability)
        cc = MojoInterface.get_compute_capability(mojo_device)
        assert type(cc) is int
        assert capability[0] is not None and capability[1] is not None
        assert cc == capability[0] * 10 + capability[1]
    elif props.api == "hip":
        cc = MojoInterface.get_compute_capability(mojo_device)
        assert isinstance(cc, str) and cc.startswith("gfx")
        assert all(type(part) is int for part in capability)
        if torch.cuda.is_available():
            idx = torch.device(mojo_device).index
            assert capability == torch.cuda.get_device_capability(idx)
    else:
        assert capability == (None, None)
    if not hasattr(torch.accelerator, "get_device_capability"):
        return  # torch < 2.10
    # Dtype support, not the SM or gfx architecture (stock CUDA and ROCm
    # raise here: CUDAGuardImpl keeps DeviceGuardImplInterface's default).
    index = torch.device(mojo_device).index
    accelerator_capability = torch.accelerator.get_device_capability(index)
    assert set(accelerator_capability) == {"supported_dtypes"}
    expected = _DEVICE_DTYPES - ({torch.float64} if props.api == "metal" else set())
    assert accelerator_capability["supported_dtypes"] == expected
    with torch.accelerator.device_index(index):
        assert torch.accelerator.get_device_capability() == accelerator_capability


# What torch.accelerator.get_device_capability() lists on CUDA and HIP; Metal
# lists the same minus float64.
_DEVICE_DTYPES = {
    torch.bool,
    torch.uint8,
    torch.int8,
    torch.int16,
    torch.int32,
    torch.int64,
    torch.uint16,
    torch.uint32,
    torch.uint64,
    torch.float16,
    torch.bfloat16,
    torch.float32,
    torch.float64,
}
# Declined on every backend: no mojo tensor of these dtypes can exist.
_NEVER_ON_DEVICE = (
    torch.complex32,
    torch.complex64,
    torch.complex128,
    torch.float8_e4m3fn,
    torch.float8_e5m2,
    torch.float8_e4m3fnuz,
    torch.float8_e5m2fnuz,
    torch.qint8,
    torch.quint8,
    torch.qint32,
)


def _values_exact_in(a: torch.dtype, b: torch.dtype) -> torch.Tensor:
    """0/1 when either side is bool, else small non-negative integers: exact
    in every supported dtype (bfloat16 up to 256, int8 up to 127)."""
    if torch.bool in (a, b):
        return torch.tensor([0, 1, 1, 0, 1])
    return torch.tensor([0, 1, 2, 3, 5, 8, 13, 21, 34, 55, 89, 127])


@pytest.mark.skipif(
    not hasattr(torch.accelerator, "get_device_capability"),
    reason="torch.accelerator.get_device_capability() needs torch 2.10",
)
def test_accelerator_capability_dtypes_allocate_and_convert(mojo_device: str):
    """Every dtype torch.accelerator.get_device_capability() lists allocates
    on the device and converts to and from every other listed one, by `to`
    and by `copy_`; the ones it leaves out are absent."""
    props = device_module.get_device_properties(mojo_device)
    index = torch.device(mojo_device).index
    supported = torch.accelerator.get_device_capability(index)["supported_dtypes"]
    assert supported and all(isinstance(dtype, torch.dtype) for dtype in supported)
    assert not supported.intersection(_NEVER_ON_DEVICE)
    if props.api == "metal":
        assert torch.float64 not in supported
    with pytest.raises(NotImplementedError):
        torch.empty(2, dtype=torch.complex64, device=mojo_device)

    dtypes = sorted(
        (dtype for dtype in supported if isinstance(dtype, torch.dtype)), key=str
    )
    failures = []
    for src in dtypes:
        allocated = torch.empty(3, dtype=src, device=mojo_device)
        assert allocated.dtype == src and allocated.device == torch.device(mojo_device)
        for dst in dtypes:
            host = _values_exact_in(src, dst).to(src)
            expected = host.to(dst)
            on_device = host.to(mojo_device)
            converted = on_device.to(dst)
            copied = torch.empty(host.shape, dtype=dst, device=mojo_device)
            copied.copy_(on_device)
            back = converted.to(src)
            for what, got, want in (
                ("to", converted, expected),
                ("copy_", copied, expected),
                ("round trip", back, host),
            ):
                got = got.cpu()
                # tolist(): CPU torch.equal lacks the uint16/32/64 kernels.
                if got.dtype != want.dtype or got.tolist() != want.tolist():
                    failures.append(f"{src} -> {dst} {what}: {got.tolist()}")
    assert not failures, failures


@pytest.mark.parametrize(
    ("arch", "expected"),
    [
        ("gfx942:sramecc+:xnack-", (9, 4)),
        ("gfx90a", (9, 0)),
        ("gfx950", (9, 5)),
        ("gfx1030", (10, 3)),
        ("gfx1100", (11, 0)),
        ("gfx1201", (12, 0)),
        ("", (None, None)),
        ("sm_90a", (None, None)),
    ],
)
def test_gfx_version_matches_hip_device_prop(
    arch: str, expected: tuple[int | None, int | None]
):
    assert device_module._gfx_version(arch) == expected


@pytest.mark.cpu_torch
def test_inductor_properties_after_explicit_enable(mojo_gpu: str):
    """18: Inductor registration is opt-in, separate from device registration."""
    if torch.cuda.is_available():
        pytest.skip("enable_inductor requires a CPU torch wheel; CUDA is active")
    if device_module.get_device_properties(mojo_gpu).api not in ("cuda", "hip"):
        pytest.skip("Inductor/Triton integration supports CUDA and HIP GPUs")
    _fresh(
        """
import sys
import torch
from torch._inductor.runtime.hints import DeviceProperties
from torch_mojo_backend import register_mojo_devices
from torch_mojo_backend.inductor import enable_inductor, MojoInterface
register_mojo_devices()
enable_inductor()
d = torch.device(sys.argv[1])
props = DeviceProperties.create(d)
assert props.type == "mojo" and props.index == d.index
assert type(props.multi_processor_count) is int and props.multi_processor_count > 0
assert props.cc == MojoInterface.get_compute_capability(d)
assert props.warp_size in (32, 64)
x = torch.arange(17, dtype=torch.float32).to(d)
fn = torch.compile(lambda x: x + 1, backend="inductor", fullgraph=True)
y = fn(x)
torch.testing.assert_close(y.cpu(), torch.arange(17, dtype=torch.float32) + 1)
""",
        mojo_gpu,
    )


def test_storage_extent_offset_and_materialization(mojo_device: str):
    """21: holes and offsets describe storage extent, not logical numel."""
    _settle(mojo_device)
    base = device_module.memory_allocated(mojo_device)
    x = torch.empty_strided((2, 3), (100, 2), dtype=torch.float32, device=mojo_device)
    x.fill_(3)
    assert x.untyped_storage().nbytes() == 420  # (1 + 100 + 2*2) * 4
    view = x.as_strided((2, 2), (100, 2), storage_offset=2)
    assert view.storage_offset() == 2
    assert view.untyped_storage().data_ptr() == x.untyped_storage().data_ptr()
    _assert_shared(mojo_device, base + 420)
    before = device_module.memory_stats(mojo_device)
    with pytest.raises(RuntimeError, match="bounds|storage"):
        x.as_strided((2, 3), (100, 2), storage_offset=1)
    assert device_module.memory_stats(mojo_device) == before
    dense = view.contiguous()
    assert dense.untyped_storage().nbytes() == 16
    assert dense.contiguous() is dense
    torch.testing.assert_close(dense.cpu(), torch.full((2, 2), 3.0))
    _assert_shared(mojo_device, base + 420 + 16)
    del x, view
    _settle(mojo_device)
    _assert_shared(mojo_device, base + 16)
    del dense
    _settle(mojo_device)
    _assert_shared(mojo_device, base)


def test_unsupported_tensor_resize_preserves_storage(mojo_device: str):
    """22: tensor resize_ is not registered (storage resize_ IS supported).

    Pin a clean decline instead of adding unrelated operator support here.
    """
    reference = torch.arange(513, dtype=torch.float32)
    _settle(mojo_device)
    base = device_module.memory_allocated(mojo_device)
    x = reference.to(mojo_device)
    _settle(mojo_device)
    before = device_module.memory_stats(mojo_device)
    for size in (7, 0, 513, 1031):
        with pytest.raises(NotImplementedError, match="aten::resize_"):
            x.resize_(size)
        assert x.untyped_storage().nbytes() == 513 * 4
        assert device_module.memory_stats(mojo_device) == before
    torch.testing.assert_close(x.cpu(), reference)
    del x
    _settle(mojo_device)
    _assert_shared(mojo_device, base)


def test_standalone_storage_resize_and_lifetime(mojo_device: str):
    """23: an UntypedStorage owns bytes independently of a Tensor wrapper."""
    _settle(mojo_device)
    base = device_module.memory_allocated(mojo_device)
    x = torch.arange(17, dtype=torch.uint8).to(mojo_device)
    storage = x.untyped_storage()
    del x
    _settle(mojo_device)
    _assert_shared(mojo_device, base + 17)
    for size in (1031, 7, 0):
        storage.resize_(size)
        _settle(mojo_device)
        assert storage.nbytes() == size
        _assert_shared(mojo_device, base + size)
        # Sizes and the literal byte ledger are independent oracles.
    del storage
    _settle(mojo_device)
    _assert_shared(mojo_device, base)


@pytest.mark.parametrize("keep_alias", [False, True])
def test_set_replaces_only_one_storage_owner(mojo_device: str, keep_alias: bool):
    """24: set_(Tensor) adopts shared storage, including its nonzero offset."""
    _settle(mojo_device)
    base = device_module.memory_allocated(mojo_device)
    x = torch.empty(17, dtype=torch.float32, device=mojo_device)
    y = torch.arange(1031, dtype=torch.float32).to(mojo_device)
    alias = x.detach() if keep_alias else None
    torch.ops.aten.set_.source_Tensor(x, y[3:10])
    assert x.storage_offset() == 3
    assert x.untyped_storage().data_ptr() == y.untyped_storage().data_ptr()
    x.fill_(42)
    torch.testing.assert_close(y[3:10].cpu(), torch.full((7,), 42.0))
    _settle(mojo_device)
    _assert_shared(mojo_device, base + 1031 * 4 + (17 * 4 if keep_alias else 0))
    del alias
    _settle(mojo_device)
    _assert_shared(mojo_device, base + 1031 * 4)
    del y
    _assert_shared(mojo_device, base + 1031 * 4)
    del x
    _settle(mojo_device)
    _assert_shared(mojo_device, base)


@pytest.mark.skipif(len(list(get_accelerators())) < 2, reason="needs two GPUs")
def test_set_cross_device_failure_preserves_both_ledgers(mojo_gpu: str):
    second = torch.device("mojo:1")
    x = torch.ones(17, device=mojo_gpu)
    y = torch.ones(31, device=second)
    for dev in (mojo_gpu, second):
        _settle(dev)
    before = [device_module.memory_stats(dev) for dev in (mojo_gpu, second)]
    with pytest.raises((RuntimeError, NotImplementedError), match="same mojo|device"):
        torch.ops.aten.set_.source_Tensor(x, y)
    assert [device_module.memory_stats(dev) for dev in (mojo_gpu, second)] == before
    torch.testing.assert_close(x.cpu(), torch.ones(17))
    torch.testing.assert_close(y.cpu(), torch.ones(31))


@pytest.mark.parametrize("owner_kind", ["list", "closure", "cycle", "detach"])
def test_python_owners_and_cycles(mojo_device: str, owner_kind: str):
    """25: addresses and weakrefs do not own storage, strong references do."""
    _settle(mojo_device)
    base = device_module.memory_allocated(mojo_device)
    x = torch.empty(4097, dtype=torch.uint8, device=mojo_device)
    ref = weakref.ref(x)
    address = x.data_ptr()
    if owner_kind == "closure":

        def owner(tensor: torch.Tensor = x) -> torch.Tensor:
            return tensor
    elif owner_kind == "detach":
        owner = x.detach()
    else:
        owner = [x]
        if owner_kind == "cycle":
            owner.append(owner)
    del x
    gc.collect()
    assert address != 0
    _assert_shared(mojo_device, base + 4097)
    assert (ref() is None) == (owner_kind == "detach")
    del owner
    _settle(mojo_device)
    assert ref() is None
    _assert_shared(mojo_device, base)


class _SaveKnownBuffer(torch.autograd.Function):
    """26's independent ledger: scalar output + 1031 float32 saved elements."""

    @staticmethod
    def forward(
        ctx: torch.autograd.function.BackwardCFunction, x: torch.Tensor
    ) -> torch.Tensor:
        saved = torch.full((1031,), 3.0, dtype=torch.float32, device=x.device)
        ctx.save_for_backward(saved)
        return x.clone()

    @staticmethod
    def backward(
        ctx: torch.autograd.function.BackwardCFunction, grad: torch.Tensor
    ) -> torch.Tensor:
        (saved,) = ctx.saved_tensors
        return grad * saved[0]


def test_autograd_saved_buffers_and_retain_graph(mojo_device: str):
    # Warm forward/backward paths before taking B; kernel/runtime caches
    # are deliberately outside the test-owned storage ledger.
    warm = torch.tensor(2.0, device=mojo_device, requires_grad=True)
    _SaveKnownBuffer.apply(warm).backward()
    del warm
    _settle(mojo_device)
    base = device_module.memory_allocated(mojo_device)
    x = torch.tensor(2.0, device=mojo_device, requires_grad=True)
    y = _SaveKnownBuffer.apply(x)
    _settle(mojo_device)
    _assert_shared(mojo_device, base + 4 + 4 + 1031 * 4)
    y.backward(retain_graph=True)
    _settle(mojo_device)
    _assert_shared(mojo_device, base + 4 + 4 + 1031 * 4 + 4)
    assert x.grad is not None
    assert x.grad.cpu().item() == 3.0
    y.backward()
    _settle(mojo_device)
    _assert_shared(mojo_device, base + 12)
    assert x.grad.cpu().item() == 6.0
    x.grad = None
    _settle(mojo_device)
    _assert_shared(mojo_device, base + 8)
    del y, x
    _settle(mojo_device)
    _assert_shared(mojo_device, base)


@pytest.mark.parametrize("factory", ["empty", "zeros", "zero_"])
def test_factory_final_storage_cost(mojo_device: str, factory: str):
    """27: in-place fill changes contents, not storage ownership."""
    warm = torch.zeros(17, device=mojo_device)
    del warm
    _settle(mojo_device)
    base = device_module.memory_allocated(mojo_device)
    if factory == "zeros":
        x = torch.zeros(513, dtype=torch.float32, device=mojo_device)
    else:
        x = torch.empty(513, dtype=torch.float32, device=mojo_device)
        if factory == "zero_":
            x.zero_()
    _settle(mojo_device)
    assert x.untyped_storage().nbytes() == 513 * 4
    _assert_shared(mojo_device, base + 513 * 4)
    if factory != "empty":
        torch.testing.assert_close(x.cpu(), torch.zeros(513))
    del x
    _settle(mojo_device)
    _assert_shared(mojo_device, base)


@pytest.mark.skipif(len(list(get_accelerators())) < 2, reason="needs two GPUs")
@pytest.mark.parametrize("reverse", [False, True])
@pytest.mark.parametrize("source_first", [False, True])
def test_cross_device_copy_ownership(mojo_gpu: str, reverse: bool, source_first: bool):
    """30: two real mojo GPUs cover independent owners."""
    other = torch.device("mojo:1")
    src, dst = (
        (other, torch.device(mojo_gpu)) if reverse else (torch.device(mojo_gpu), other)
    )
    for dev in (src, dst):
        _settle(dev)
    bases = [device_module.memory_allocated(dev) for dev in (src, dst)]
    reference = torch.arange(513, dtype=torch.float32)
    x = reference.to(src)
    assert x.to(src) is x
    copied = x.to(src, copy=True)
    _assert_shared(src, bases[0] + 2 * 513 * 4)
    del copied
    y = x.to(dst)
    for dev in (src, dst):
        _settle(dev)
    _assert_shared(src, bases[0] + 513 * 4)
    _assert_shared(dst, bases[1] + 513 * 4)
    y.fill_(7)
    torch.testing.assert_close(x.cpu(), reference)
    torch.testing.assert_close(y.cpu(), torch.full((513,), 7.0))
    if source_first:
        del x
        _settle(src)
        _assert_shared(src, bases[0])
        _assert_shared(dst, bases[1] + 513 * 4)
        del y
    else:
        del y
        _settle(dst)
        _assert_shared(dst, bases[1])
        _assert_shared(src, bases[0] + 513 * 4)
        del x
    for dev, base in zip((src, dst), bases, strict=True):
        _settle(dev)
        _assert_shared(dev, base)


@pytest.mark.parametrize("offset", [0, 3])
def test_final_release_with_pending_recorded_work(mojo_gpu: str, offset: int):
    """08/31: logical accounting ends when final storage returns to MAX.

    Physical reuse is stream-ordered by MAX. Querying before synchronization
    must already exclude X; synchronization then proves the consumer's data
    survived. No assumption that the GPU is still busy when we sample.
    """
    side = side_stream_or_skip(mojo_gpu)
    out = torch.empty(4093, device=mojo_gpu)
    _settle(mojo_gpu)
    base = device_module.memory_allocated(mojo_gpu)  # includes the output
    x = torch.full((4096,), 7.0, device=mojo_gpu)
    view = x[offset : offset + 4093]
    side.wait_stream(device_module.current_stream(mojo_gpu))
    with device_module.stream(side):
        out.copy_(view)
    view.record_stream(side)
    view.record_stream(side)
    empty_view = x[3:3]
    empty_view.record_stream(side)
    del empty_view
    _assert_shared(mojo_gpu, base + 4096 * 4)
    del view, x
    _assert_shared(mojo_gpu, base)
    # Competing owner-stream allocation/fill gives premature reuse a chance
    # to corrupt the consumer. We require contents, never pointer reuse or
    # proof that a race happened on this particular run.
    competing = torch.full((4096,), -9.0, device=mojo_gpu)
    _assert_shared(mojo_gpu, base + 4096 * 4)
    del competing
    device_module.synchronize(mojo_gpu)
    torch.testing.assert_close(out.cpu(), torch.full((4093,), 7.0))
    _assert_shared(mojo_gpu, base)


def test_nondefault_stream_allocation_and_manual_ordering(mojo_gpu: str):
    """32: manual stream dependencies are the alternative to record_stream."""
    producer = side_stream_or_skip(mojo_gpu)
    consumer = side_stream_or_skip(mojo_gpu)
    out = torch.empty(1031, device=mojo_gpu)
    _settle(mojo_gpu)
    base = device_module.memory_allocated(mojo_gpu)
    with device_module.stream(producer):
        x = torch.full((1031,), 11.0, device=mojo_gpu)
    _assert_shared(mojo_gpu, base + 1031 * 4)
    consumer.wait_stream(producer)
    with device_module.stream(consumer):
        out.copy_(x)
    producer.wait_stream(consumer)
    del x
    # Device synchronization covers BOTH side streams; default-only would not.
    device_module.synchronize(mojo_gpu)
    _assert_shared(mojo_gpu, base)
    torch.testing.assert_close(out.cpu(), torch.full((1031,), 11.0))


def test_foreign_thread_free_and_concurrent_counters(mojo_device: str):
    """33: final ownership crosses a queue; barriers define exact snapshots."""
    idx = torch.device(mojo_device).index
    assert idx is not None
    other = (idx + 1) % device_module.device_count()
    for dev in {idx, other}:
        _settle(dev)
    base = device_module.memory_allocated(idx)
    other_before = device_module.memory_stats(other)
    queue = Queue()
    queue.put(torch.empty(4097, dtype=torch.uint8, device=mojo_device))
    _assert_shared(idx, base + 4097)

    def release() -> int:
        with device_module.device(other):
            tensor = queue.get(timeout=30)
            del tensor
            assert device_module.current_device() == other
        return other

    with ThreadPoolExecutor(max_workers=4) as pool:
        assert pool.submit(release).result(timeout=60) == other
        _settle(idx)
        _assert_shared(idx, base)
        if idx != other:
            assert device_module.memory_stats(other) == other_before
        held = threading.Barrier(5, timeout=30)
        release_all = threading.Event()
        sizes = (1, 511, 4097, 65539)

        def worker(size: int):
            with device_module.device(other):
                tensor = torch.empty(size, dtype=torch.uint8, device=mojo_device)
                held.wait()
                assert release_all.wait(timeout=60)
                del tensor

        futures = [pool.submit(worker, size) for size in sizes]
        try:
            held.wait()
            _assert_shared(idx, base + sum(sizes))
        finally:
            release_all.set()
        for future in futures:
            future.result(timeout=60)
    _settle(idx)
    _assert_shared(idx, base)
    if idx != other:
        assert device_module.memory_stats(other) == other_before


def test_invalid_and_overflowing_requests_preserve_counters(mojo_device: str):
    """38: signed extent overflow is rejected without physical pressure."""
    _fresh(
        """
import sys
import torch
from torch_mojo_backend import register_mojo_devices
register_mojo_devices()
d = sys.argv[1]
sentinel = torch.arange(17, dtype=torch.float32).to(d)
torch.mojo.synchronize(d)
for shape in ((-1,), (2**62, 8), (2**63,)):
    before = torch.mojo.memory_stats(d)
    try:
        torch.empty(shape, dtype=torch.float32, device=d)
    except (RuntimeError, TypeError, OverflowError):
        pass
    else:
        raise AssertionError(("invalid shape succeeded", shape))
    assert torch.mojo.memory_stats(d) == before
before = torch.mojo.memory_stats(d)
try:
    sentinel.untyped_storage().resize_(-1)
except (RuntimeError, TypeError, OverflowError):
    pass
else:
    raise AssertionError("negative storage size succeeded")
assert torch.mojo.memory_stats(d) == before
torch.testing.assert_close(sentinel.cpu(), torch.arange(17, dtype=torch.float32))
x = torch.empty(513, dtype=torch.uint8, device=d)
assert torch.mojo.memory_allocated(d) == before["allocated_bytes.all.current"] + 513
del x
torch.mojo.synchronize(d)
assert torch.mojo.memory_allocated(d) == before["allocated_bytes.all.current"]
""",
        mojo_device,
    )


@pytest.mark.filterwarnings("ignore:This process:DeprecationWarning")
@pytest.mark.parametrize("pin_memory", [False, True])
def test_dataloader_parent_memory_boundaries(mojo_device: str, pin_memory: bool):
    """40: CPU fork workers and persistent/pinned loading preserve parent B."""
    _settle(mojo_device)
    sentinel = torch.empty(17, dtype=torch.uint8, device=mojo_device)
    base = device_module.memory_allocated(mojo_device)
    source = torch.arange(64.0).reshape(16, 4)
    loader = DataLoader(
        TensorDataset(source),
        batch_size=4,
        num_workers=2,
        multiprocessing_context="fork",
        persistent_workers=True,
        pin_memory=pin_memory,
        timeout=60,
    )
    for _ in range(2):
        for i, (host,) in enumerate(loader):
            assert host.device.type == "cpu"
            batch = host.to(mojo_device, non_blocking=pin_memory)
            _settle(mojo_device)
            _assert_shared(mojo_device, base + 4 * 4 * 4)
            torch.testing.assert_close(batch.cpu(), source[4 * i : 4 * (i + 1)])
            del batch
            _settle(mojo_device)
            _assert_shared(mojo_device, base)
    del loader, sentinel
    _settle(mojo_device)


def test_spawned_process_has_independent_counters(mojo_device: str):
    """41: pipe barriers keep independent child/parent ownership observable."""
    _settle(mojo_device)
    sentinel = torch.empty(513, dtype=torch.uint8, device=mojo_device)
    before = device_module.memory_stats(mojo_device)
    code = """
import contextlib
import sys
with contextlib.redirect_stdout(sys.stderr):
    import torch
    from torch_mojo_backend import register_mojo_devices
    register_mojo_devices()
d = sys.argv[1]
base = torch.mojo.memory_allocated(d)
x = torch.empty(4097, dtype=torch.uint8, device=d)
assert torch.mojo.memory_allocated(d) == base + 4097
stats = torch.mojo.memory_stats(d)
print("held", flush=True)
assert input() == "reset"
assert torch.mojo.memory_stats(d) == stats
torch.mojo.reset_peak_memory_stats(d)
torch.mojo.reset_accumulated_memory_stats(d)
print("reset", flush=True)
assert input() == "release"
del x
torch.mojo.synchronize(d)
assert torch.mojo.memory_allocated(d) == base
print("released", flush=True)
"""
    with subprocess.Popen(
        [sys.executable, "-c", code, mojo_device],
        stdin=subprocess.PIPE,
        stdout=subprocess.PIPE,
        text=True,
    ) as child:
        assert child.stdout is not None and child.stdin is not None

        def receive(expected: str):
            assert child.stdout is not None
            assert select.select([child.stdout], [], [], 120)[0], "child hung"
            assert child.stdout.readline().strip() == expected

        try:
            receive("held")
            assert device_module.memory_stats(mojo_device) == before
            device_module.reset_peak_memory_stats(mojo_device)
            device_module.reset_accumulated_memory_stats(mojo_device)
            reset = device_module.memory_stats(mojo_device)
            child.stdin.write("reset\n")
            child.stdin.flush()
            receive("reset")
            assert device_module.memory_stats(mojo_device) == reset
            child.stdin.write("release\n")
            child.stdin.flush()
            receive("released")
            assert child.wait(timeout=60) == 0
            assert device_module.memory_stats(mojo_device) == reset
        finally:
            if child.poll() is None:
                child.kill()
                child.wait(timeout=30)
    del sentinel
    _settle(mojo_device)


def test_fake_and_meta_tensors_do_not_charge(mojo_device: str):
    """43: logical shapes in fake/meta mode have no owned device payload."""
    with FakeTensorMode():
        torch.empty(7, device=mojo_device).view(1, 7)
    _settle(mojo_device)
    sentinel = torch.empty(17, dtype=torch.uint8, device=mojo_device)
    before = device_module.memory_stats(mojo_device)
    meta = torch.empty((2**20, 2**20), device="meta")
    with FakeTensorMode():
        fake = torch.empty((2**20, 2**20), device=mojo_device)
        view = fake.view(-1)
        assert view.numel() == 2**40
        assert fake.device == torch.device(mojo_device)
    assert meta.device.type == "meta"
    assert device_module.memory_stats(mojo_device) == before
    real = torch.empty(1031, dtype=torch.float32, device=mojo_device)
    _assert_shared(mojo_device, before["allocated_bytes.all.current"] + 1031 * 4)
    del real
    _settle(mojo_device)
    _assert_shared(mojo_device, before["allocated_bytes.all.current"])
    del sentinel


def test_cumulative_bytes_exceed_32_bits(mojo_device: str):
    """46: >4 GiB traffic, only 16 MiB live; exact independent arithmetic."""
    _settle(mojo_device)
    sentinel = torch.empty(17, dtype=torch.uint8, device=mojo_device)
    base = device_module.memory_allocated(mojo_device)
    device_module.reset_accumulated_memory_stats(mojo_device)
    device_module.reset_peak_memory_stats(mojo_device)
    size, count = 16 * 1024**2, 257
    for _ in range(count):
        x = torch.empty(size, dtype=torch.uint8, device=mojo_device)
        assert device_module.memory_allocated(mojo_device) == base + size
        del x
        device_module.synchronize(mojo_device)
        assert device_module.memory_allocated(mojo_device) == base
    for api in (device_module, torch.accelerator):
        stats = api.memory_stats(mojo_device)
        for family in ("allocated_bytes", "requested_bytes", "reserved_bytes"):
            for field in ("allocated", "freed"):
                value = stats[f"{family}.all.{field}"]
                assert type(value) is int and value == size * count > 2**32
            assert stats[f"{family}.all.peak"] == base + size
        assert stats["num_device_alloc"] == stats["num_device_free"] == count
    del sentinel
    _settle(mojo_device)


@pytest.mark.skipif(
    not Path("/proc/self/statm").exists(), reason="current RSS oracle needs Linux /proc"
)
def test_repeated_release_current_residency(mojo_device: str):
    """47/H: live counters plus CURRENT RSS and independent MAX arena usage.

    This is a bounded residency regression check, not a raw-allocation ledger
    or proof against arbitrarily small leaks. Allow one warmed batch of host
    arena retention, and one MAX arena chunk; repeated leaked batches exceed
    both. Neither peak counters nor ru_maxrss are used as a leak oracle.
    """
    _fresh(
        """
import gc
import os
import sys
from pathlib import Path
import torch
from torch_mojo_backend import register_mojo_devices
register_mojo_devices()
d = sys.argv[1]
size, count = 1024**2, 32
def batch():
    tensors = [torch.empty(size, dtype=torch.uint8, device=d) for _ in range(count)]
    tensors.clear()
    gc.collect()
    torch.mojo.synchronize(d)
def rss():
    return int(Path("/proc/self/statm").read_text().split()[1]) * os.sysconf("SC_PAGE_SIZE")
for _ in range(8):
    batch()
base = torch.mojo.memory_stats(d)
rss_before = rss()
free_before, total = torch.mojo.mem_get_info(d)
for _ in range(64):
    batch()
    stats = torch.mojo.memory_stats(d)
    assert stats["allocated_bytes.all.current"] == base["allocated_bytes.all.current"]
    assert stats["allocation.all.current"] == base["allocation.all.current"]
    # A bounded diagnostic allowance for host/runtime arenas and metadata.
    assert rss() <= rss_before + size * count
    free, current_total = torch.mojo.mem_get_info(d)
    assert current_total == total
    # MAX's observed minimum arena chunk is 256 MiB (see docs).
    assert free >= free_before - 256 * 1024**2
""",
        mojo_device,
    )


@pytest.mark.parametrize("shape", [(357, 19), (1,), (0, 7)])
@pytest.mark.parametrize("dtype", [torch.uint8, torch.float32, torch.float64])
def test_storage_bytes_and_counts(
    mojo_device: str, shape: tuple[int, ...], dtype: torch.dtype
):
    device_module.synchronize(mojo_device)
    before = device_module.memory_stats(mojo_device)
    x = torch.empty(shape, dtype=dtype, device=mojo_device)
    nbytes = x.untyped_storage().nbytes()
    assert nbytes == torch.empty(shape, dtype=dtype).untyped_storage().nbytes()
    during = device_module.memory_stats(mojo_device)
    for metric in ("allocated_bytes", "requested_bytes", "reserved_bytes"):
        assert (
            during[f"{metric}.all.current"] == before[f"{metric}.all.current"] + nbytes
        )
        assert (
            during[f"{metric}.all.allocated"]
            == before[f"{metric}.all.allocated"] + nbytes
        )
    count = int(nbytes > 0)
    assert during["allocation.all.current"] == before["allocation.all.current"] + count
    assert during["num_device_alloc"] == before["num_device_alloc"] + count
    del x
    device_module.synchronize(mojo_device)
    after = device_module.memory_stats(mojo_device)
    assert after["allocated_bytes.all.current"] == before["allocated_bytes.all.current"]
    assert (
        after["allocated_bytes.all.freed"]
        == before["allocated_bytes.all.freed"] + nbytes
    )
    assert after["allocation.all.current"] == before["allocation.all.current"]
    assert after["num_device_free"] == before["num_device_free"] + count


@pytest.mark.parametrize("api_name", ["mojo", "accelerator"])
def test_peak_and_accumulated_resets(mojo_device: str, api_name: str):
    api = device_module if api_name == "mojo" else torch.accelerator
    base = device_module.memory_allocated(mojo_device)
    keep = torch.empty(13, device=mojo_device)
    current = base + keep.untyped_storage().nbytes()
    api.reset_peak_memory_stats(mojo_device)
    large = torch.empty(1024 * 1024, dtype=torch.uint8, device=mojo_device)
    peak = current + large.untyped_storage().nbytes()
    assert device_module.max_memory_allocated(mojo_device) == peak
    del large
    device_module.synchronize(mojo_device)
    assert device_module.memory_allocated(mojo_device) == current
    assert device_module.max_memory_allocated(mojo_device) == peak
    assert device_module.max_memory_reserved(mojo_device) == peak
    api.reset_accumulated_memory_stats(mojo_device)
    stats = device_module.memory_stats(mojo_device)
    assert stats["allocated_bytes.all.current"] == current
    assert stats["allocated_bytes.all.peak"] == peak
    for key, value in stats.items():
        if key.endswith((".allocated", ".freed")) or key.startswith("num_"):
            assert value == 0, key
    api.reset_peak_memory_stats(mojo_device)
    assert device_module.max_memory_allocated(mojo_device) == current
    assert device_module.max_memory_reserved(mojo_device) == current
    del keep
    device_module.synchronize(mojo_device)
    assert device_module.memory_allocated(mojo_device) == base
    # A free after a reset still counts, even when its allocation was before it.
    assert device_module.memory_stats(mojo_device)["allocated_bytes.all.freed"] == 52


@pytest.mark.parametrize("peak_first", [False, True])
def test_reset_orders_are_idempotent_with_live_storage(
    mojo_device: str, peak_first: bool
):
    """04/05: resetting either way preserves live owners and resets every leaf."""
    _settle(mojo_device)
    base = device_module.memory_allocated(mojo_device)
    x = torch.empty(53, dtype=torch.uint8, device=mojo_device)
    temporary = torch.empty(1031, dtype=torch.uint8, device=mojo_device)
    del temporary
    _settle(mojo_device)
    before = device_module.memory_stats(mojo_device)
    resets = (
        device_module.reset_peak_memory_stats,
        torch.accelerator.reset_accumulated_memory_stats,
    )
    for reset in resets if peak_first else reversed(resets):
        reset(mojo_device)
    after = device_module.memory_stats(mojo_device)
    for key, value in before.items():
        if key.endswith(".peak"):
            assert after[key] == before[key.removesuffix("peak") + "current"]
        elif key.endswith((".allocated", ".freed")) or key.startswith("num_"):
            assert after[key] == 0
        else:
            assert after[key] == value
    for reset in resets:
        reset(mojo_device)
        assert device_module.memory_stats(mojo_device) == after
    _assert_shared(mojo_device, base + 53)
    del x
    _settle(mojo_device)
    _assert_shared(mojo_device, base)
    assert device_module.memory_stats(mojo_device)["allocated_bytes.all.freed"] == 53


def test_views_and_strided_storage(mojo_device: str):
    # Collect tensors left in cycles by earlier tests before recording the baseline.
    _settle(mojo_device)
    base = device_module.memory_allocated(mojo_device)
    x = torch.empty_strided((5, 7), (19, 2), device=mojo_device)
    reference = torch.empty_strided((5, 7), (19, 2))
    nbytes = reference.untyped_storage().nbytes()
    assert device_module.memory_allocated(mojo_device) == base + nbytes
    views = [x.view_as(x), x[1:, 2:], torch.as_strided(x, (2, 3), (19, 2), 1)]
    assert all(
        v.untyped_storage().data_ptr() == x.untyped_storage().data_ptr() for v in views
    )
    assert device_module.memory_allocated(mojo_device) == base + nbytes
    del x
    assert device_module.memory_allocated(mojo_device) == base + nbytes
    del views
    device_module.synchronize(mojo_device)
    assert device_module.memory_allocated(mojo_device) == base


@pytest.mark.parametrize("non_blocking", [False, True])
def test_host_memory_is_excluded(mojo_device: str, non_blocking: bool):
    # Unreachable tensor cycles from earlier tests must not be collected
    # after we snapshot the live-storage baseline.
    _settle(mojo_device)
    base = device_module.memory_allocated(mojo_device)
    host = torch.arange(513, dtype=torch.float32)
    pinned = torch.empty(4096, pin_memory=True)
    assert pinned.device.type == "cpu"
    assert device_module.memory_allocated(mojo_device) == base
    x = host.to(mojo_device, non_blocking=non_blocking)
    during = base + x.untyped_storage().nbytes()
    copied = x.to("cpu", non_blocking=non_blocking)
    device_module.synchronize(mojo_device)
    torch.testing.assert_close(copied, host)
    assert device_module.memory_allocated(mojo_device) == during
    del x
    device_module.synchronize(mojo_device)
    assert device_module.memory_allocated(mojo_device) == base


def test_stats_identities_and_summary(mojo_device: str):
    x = torch.empty(4099, dtype=torch.uint8, device=mojo_device)
    stats = device_module.memory_stats(mojo_device)
    assert isinstance(stats, OrderedDict)
    assert list(stats) == sorted(stats)
    nested = device_module.memory_stats_as_nested_dict(mojo_device)
    for metric, read_current, read_peak in (
        (
            "allocated_bytes",
            device_module.memory_allocated,
            device_module.max_memory_allocated,
        ),
        (
            "reserved_bytes",
            device_module.memory_reserved,
            device_module.max_memory_reserved,
        ),
    ):
        assert stats[f"{metric}.all.current"] == read_current(mojo_device)
        assert stats[f"{metric}.all.peak"] == read_peak(mojo_device)
    assert nested == torch._C._accelerator_getDeviceStats(
        torch.device(mojo_device).index
    )
    for key, value in stats.items():
        assert isinstance(value, int) and value >= 0
        if (
            ".small_pool." in key
            or ".large_pool." in key
            or key.startswith(
                (
                    "active",
                    "inactive_split",
                    "oversize",
                    "segment",
                    "max_split_size",
                    "num_sync_all_streams",
                )
            )
        ):
            assert value == 0, key
    for abbreviated in (False, True):
        summary = device_module.memory_summary(mojo_device, abbreviated=abbreviated)
        assert f"{device_module.memory_allocated(mojo_device):,}" in summary
        assert "MAX owns the arena" in summary
        assert ("Requested memory" in summary) is not abbreviated
    del x


def test_empty_cache_before_any_allocation(mojo_gpu_available: bool):
    if not mojo_gpu_available:
        pytest.skip("You do not have a GPU supported by MAX")
    code = """
import torch
from torch_mojo_backend import register_mojo_devices
register_mojo_devices()
assert torch._C._accelerator_isAllocatorInitialized()
for api in (torch.mojo, torch.accelerator):
    assert api.memory_allocated() == 0
    api.empty_cache()
    api.empty_cache()
    assert api.memory_allocated() == 0
"""
    subprocess.run([sys.executable, "-c", code], check=True, timeout=120)


def test_empty_cache_preserves_live_storage(mojo_device: str):
    """Live device storage survives a flush; staging is checked separately."""
    x = torch.arange(1031, dtype=torch.float32).to(mojo_device)
    current = device_module.memory_allocated(mojo_device)
    for api in (device_module, torch.accelerator):
        api.empty_cache()
        api.empty_cache()
        assert device_module.memory_allocated(mojo_device) == current
    device_module.synchronize(mojo_device)
    before = device_module.memory_stats(mojo_device)
    device_module.empty_cache()
    torch.accelerator.empty_cache()
    assert device_module.memory_stats(mojo_device) == before
    torch.testing.assert_close(x.cpu(), torch.arange(1031, dtype=torch.float32))


def _current_rss() -> int:
    """Resident bytes now, never the process's high-water mark."""
    for line in Path("/proc/self/status").read_text().splitlines():
        if line.startswith("VmRSS:"):
            return int(line.split()[1]) * 1024
    raise AssertionError("/proc/self/status did not report VmRSS")


def _check_empty_cache_staging(device: str, api_name: str):
    """Worker shared with mutation probes; see the test for oracle limits."""
    register_mojo_devices()
    api = device_module if api_name == "mojo" else torch.accelerator
    stream = torch.Stream(device=device)
    size = 64 * 1024**2
    source = torch.full((size,), 37, dtype=torch.uint8)
    expected = source.clone()
    uploaded = torch.empty_like(source, device=device)

    # Warm driver setup, pinned allocation and readback before measuring RSS.
    with stream:
        uploaded.copy_(source, non_blocking=True)
    stream.synchronize()
    torch.testing.assert_close(uploaded.cpu(), expected)
    device_module.synchronize(device)
    stream.synchronize()
    baseline = _current_rss()
    current = device_module.memory_allocated(device)
    with stream:
        uploaded.copy_(source, non_blocking=True)
    # A device synchronize also drains staging, masking a no-op empty_cache.
    # Stream synchronization only completes the transfer and its callback.
    stream.synchronize()
    completed = _current_rss()
    assert completed >= baseline + size * 3 // 4, "staging was not resident"
    api.empty_cache()
    # MAX queues the physical free on the owning stream even with caching off.
    # Finish that free without calling a second staging-drain entry point.
    stream.synchronize()
    released = completed - _current_rss()
    assert released >= size * 3 // 4, (
        f"completed staging was not released: {released} of {size} bytes"
    )
    api.empty_cache()
    assert device_module.memory_allocated(device) == current
    torch.testing.assert_close(uploaded.cpu(), expected)

    with stream:
        a = torch.ones((4096, 4096), device=device)
        b = torch.ones_like(a)
        out = torch.empty_like(a)
        torch.mm(a, b, out=out)
    stream.synchronize()
    flush_times = []
    sync_times = []
    pending = []
    retained = []
    # Four samples per leg, in ABBA order, so a clock/temperature ramp does
    # not favor one leg. Medians tolerate a scheduling outlier. Time only the
    # call after enqueueing identical work, not Python enqueue overhead.
    for wait in (True, False, False, True) * 2:
        source.fill_(37)
        device_module.synchronize(device)
        stream.synchronize()
        baseline = _current_rss()
        with stream:
            for _ in range(64):
                torch.mm(a, b, out=out)
            uploaded.copy_(source, non_blocking=True)
        # The pageable source is no longer the transfer's source of truth.
        source.fill_(99)
        staged = _current_rss()
        assert staged >= baseline + size * 3 // 4, "staging was not resident"
        assert not stream.query(), "queued work finished before the timing sample"
        started = time.perf_counter()
        if wait:
            stream.synchronize()
            sync_times.append(time.perf_counter() - started)
        else:
            api.empty_cache()
            flush_times.append(time.perf_counter() - started)
            pending.append(not stream.query())
            retained.append(_current_rss() >= staged - size // 4)
        stream.synchronize()
        if not wait:
            # An early free handed to MAX can stay resident until this sync.
            # Require retention here too, before any post-completion drain.
            retained[-1] &= _current_rss() >= staged - size // 4
        torch.testing.assert_close(uploaded.cpu(), expected)
    flush_seconds, sync_seconds = median(flush_times), median(sync_times)
    assert flush_seconds < sync_seconds * 0.5, (
        f"empty_cache waited for pending work: median {flush_seconds:.6f}s, "
        f"synchronize {sync_seconds:.6f}s"
    )
    assert sum(pending) >= 3, "empty_cache did not return while work was pending"
    assert all(kept for busy, kept in zip(pending, retained) if busy), (
        "pending staging was released before a post-completion drain"
    )
    device_module.synchronize(device)
    stream.synchronize()


@pytest.mark.skipif(
    not Path("/proc/self/status").exists(),
    reason="current RSS oracle needs Linux /proc",
)
@pytest.mark.parametrize("api_name", ["mojo", "accelerator"])
def test_empty_cache_staging_contract(mojo_gpu: str, api_name: str):
    """Completed staging is released; pending data stays resident without a wait.

    MAX's default host cache retains even multi-GiB freed buffers, hiding
    release from RSS. Disable it in a fresh process, before MAX initializes,
    using the same host-manager setting as MAX's own allocator tests. Only
    public torch operations create/complete the transfers; current VmRSS is
    the independent release oracle, with 16 MiB allowance for runtime noise.

    Check pending residency both at return and after stream completion: MAX
    can defer an early buffer free until then. Copied values must also survive
    overwriting the pageable source. Internal staging-wrapper ownership is
    not exposed publicly; these checks establish buffer residency and data
    integrity, not the identity/lifetime of that private bookkeeping object.
    CPU and Metal have synchronous uploads without this staging path.
    """
    if device_module.get_device_properties(mojo_gpu).api not in ("cuda", "hip"):
        pytest.skip("requires the asynchronous pinned-staging upload path")
    subprocess.run(
        [
            sys.executable,
            "-c",
            "from tests.native.test_memory import _check_empty_cache_staging; "
            "import sys; _check_empty_cache_staging(sys.argv[1], sys.argv[2])",
            mojo_gpu,
            api_name,
        ],
        env={**os.environ, "MODULAR_DEVICE_CONTEXT_HOST_MEMORY_MANAGER_SIZE": "0"},
        check=True,
        timeout=180,
    )


def test_properties_and_memory_info(mojo_device: str):
    props = device_module.get_device_properties(mojo_device)
    assert props == device_module.get_device_properties(torch.device(mojo_device))
    assert props.name and props.name == device_module.get_device_name(mojo_device)
    assert device_module.get_device_capability(mojo_device) == (
        props.major,
        props.minor,
    )
    free, total = device_module.mem_get_info(mojo_device)
    assert 0 < free <= total
    assert props.total_memory == total
    with pytest.raises(FrozenInstanceError):
        setattr(props, "name", "changed")
    assert props.multi_processor_count is not None and props.multi_processor_count > 0
    if props.api == "metal":
        # MAX does not expose Metal's SIMD-group width as a device attribute.
        assert props.warp_size is None
    else:
        assert props.warp_size in (32, 64)
    if props.api == "cuda":
        assert props.major is not None and props.major > 0
        if torch.cuda.is_available():
            idx = torch.device(mojo_device).index
            assert (props.major, props.minor) == torch.cuda.get_device_capability(idx)
        assert props.arch_name is not None
        assert props.arch_name.startswith(f"sm_{props.major}{props.minor}")


@pytest.mark.parametrize("name", _DEVICE_APIS)
@pytest.mark.parametrize("bad", [-1, "count", 10**30, "cpu"])
def test_invalid_device(name: str, bad: int | str):
    index = device_module.device_count() if bad == "count" else bad
    with pytest.raises(ValueError, match="device"):
        getattr(device_module, name)(index)


@pytest.mark.parametrize("name", _ACCELERATOR_APIS)
@pytest.mark.parametrize("bad", [-1, "count", 10**30])
def test_accelerator_invalid_device(name: str, bad: int | str):
    index = device_module.device_count() if bad == "count" else bad
    with pytest.raises((ValueError, RuntimeError, TypeError), match="device|arguments"):
        getattr(torch.accelerator, name)(index)


def test_accelerator_agrees(mojo_device: str):
    """Each surface must match a byte/event ledger, including both resets."""
    with device_module.device(mojo_device):
        assert torch._C._accelerator_isAllocatorInitialized()
        _settle(mojo_device)
        apis = (device_module, torch.accelerator)
        baselines = [
            (api.memory_allocated(), api.memory_stats()["allocation.all.current"])
            for api in apis
        ]

        def check(
            live_bytes: int,
            peak_bytes: int,
            allocated: int,
            freed: int,
            live_count: int,
            peak_count: int,
            allocations: int,
            frees: int,
        ):
            for api, (base_bytes, base_count) in zip(apis, baselines):
                assert api.memory_allocated() == base_bytes + live_bytes
                assert api.memory_reserved() == base_bytes + live_bytes
                assert api.max_memory_allocated() == base_bytes + peak_bytes
                assert api.max_memory_reserved() == base_bytes + peak_bytes
                stats = api.memory_stats()
                for family in ("allocated_bytes", "reserved_bytes", "requested_bytes"):
                    assert stats[f"{family}.all.current"] == base_bytes + live_bytes
                    assert stats[f"{family}.all.peak"] == base_bytes + peak_bytes
                    assert stats[f"{family}.all.allocated"] == allocated
                    assert stats[f"{family}.all.freed"] == freed
                assert stats["allocation.all.current"] == base_count + live_count
                assert stats["allocation.all.peak"] == base_count + peak_count
                assert stats["allocation.all.allocated"] == allocations
                assert stats["allocation.all.freed"] == frees
                assert stats["num_device_alloc"] == allocations
                assert stats["num_device_free"] == frees
                assert stats["num_alloc_retries"] == stats["num_ooms"] == 0
            assert torch.accelerator.memory_stats() == device_module.memory_stats()

        # Exercise resets through EACH surface, reading both independently.
        for api in apis:
            api.reset_peak_memory_stats()
            api.reset_accumulated_memory_stats()
            check(0, 0, 0, 0, 0, 0, 0, 0)
            x = torch.empty(8193, dtype=torch.uint8, device=mojo_device)
            check(8193, 8193, 8193, 0, 1, 1, 1, 0)
            temporary = torch.empty(257, dtype=torch.uint8, device=mojo_device)
            check(8450, 8450, 8450, 0, 2, 2, 2, 0)
            del temporary
            _settle(mojo_device)
            check(8193, 8450, 8450, 257, 1, 2, 2, 1)
            api.reset_accumulated_memory_stats()
            check(8193, 8450, 0, 0, 1, 2, 0, 0)
            api.reset_peak_memory_stats()
            check(8193, 8193, 0, 0, 1, 1, 0, 0)
            api.empty_cache()
            check(8193, 8193, 0, 0, 1, 1, 0, 0)
            del x
            _settle(mojo_device)
            check(0, 8193, 0, 8193, 0, 1, 0, 1)

        expected = device_module.mem_get_info()
        got = torch.accelerator.get_memory_info()
        # Free host memory can change between adjacent CPU queries.
        assert got[1] == expected[1]
        assert 0 < got[0] <= got[1]


@pytest.mark.skipif(len(list(get_accelerators())) < 3, reason="needs two GPUs")
def test_gpu_isolation(mojo_gpu: str):
    before = device_module.memory_stats(mojo_gpu)
    other_before = device_module.memory_allocated(1)
    x = torch.empty(12345, dtype=torch.uint8, device="mojo:1")
    assert device_module.memory_stats(mojo_gpu) == before
    assert device_module.memory_allocated(1) == other_before + 12345
    device_module.reset_peak_memory_stats(1)
    device_module.reset_accumulated_memory_stats(1)
    assert device_module.memory_stats(mojo_gpu) == before
    del x
    device_module.synchronize(1)
    assert device_module.memory_allocated(1) == other_before


def test_record_stream_does_not_double_count(mojo_gpu: str):
    side = side_stream_or_skip(mojo_gpu)
    baseline = device_module.memory_allocated(mojo_gpu)
    x = torch.empty(73, device=mojo_gpu)
    x.record_stream(side)
    x.record_stream(side)
    assert device_module.memory_allocated(mojo_gpu) == baseline + 292
    del x
    side.synchronize()
    device_module.synchronize(mojo_gpu)
    assert device_module.memory_allocated(mojo_gpu) == baseline


def test_oom_counts_retry_without_allocating(mojo_gpu: str):
    _, total = device_module.mem_get_info(mojo_gpu)
    sentinel = torch.arange(17, dtype=torch.float32).to(mojo_gpu)
    _settle(mojo_gpu)
    for operation in ("empty", "storage_growth"):
        before = device_module.memory_stats(mojo_gpu)
        # The boxed empty C ABI carries allocator errors as RuntimeError;
        # direct storage growth preserves torch.OutOfMemoryError's subclass.
        with pytest.raises(RuntimeError, match="out of memory"):
            if operation == "empty":
                torch.empty(2 * total, dtype=torch.uint8, device=mojo_gpu)
            else:
                sentinel.untyped_storage().resize_(2 * total)
        after = device_module.memory_stats(mojo_gpu)
        for key, value in before.items():
            assert after[key] == value + int(
                key in ("num_alloc_retries", "num_ooms")
            ), key
        assert sentinel.untyped_storage().nbytes() == 17 * 4
        torch.testing.assert_close(
            sentinel.cpu(), torch.arange(17, dtype=torch.float32)
        )
    recovery = torch.empty(7, dtype=torch.uint8, device=mojo_gpu)
    _assert_shared(mojo_gpu, before["allocated_bytes.all.current"] + 7)
    del recovery
    _settle(mojo_gpu)
    device_module.reset_accumulated_memory_stats(mojo_gpu)
    reset = device_module.memory_stats(mojo_gpu)
    assert all(value == 0 for key, value in reset.items() if key.startswith("num_"))


def test_inductor_memory_allocated(mojo_device: str):
    before = MojoInterface.memory_allocated(mojo_device)
    x = torch.empty(2049, dtype=torch.uint8, device=mojo_device)
    assert MojoInterface.memory_allocated(mojo_device) == before + 2049
    assert MojoInterface.memory_allocated(
        mojo_device
    ) == device_module.memory_allocated(mojo_device)
    del x


def test_memory_apis_after_fork(mojo_device: str):
    held = [torch.empty(7, device=mojo_device)]
    # Populate the cache too: cached properties must still honor bad-fork checks.
    device_module.get_device_properties(mojo_device)

    def child() -> str:
        assert not torch._C._accelerator_isAllocatorInitialized()
        for name in _DEVICE_APIS:
            with pytest.raises(RuntimeError, match="spawn"):
                getattr(device_module, name)(mojo_device)
        # 09/39: this is the reachable cold mojo allocator. Stats skip
        # argument validation. Resets gained that guard in torch 2.13
        # (upstream 29d6170e7c7); 2.11/2.12 docstrings promised it too early.
        cold_resets = tuple(map(int, torch.__version__.split(".")[:2])) >= (2, 13)
        for argument in (mojo_device, "cpu", "cuda:0", 2**64, "malformed"):
            for name in _ACCELERATOR_APIS:
                if name == "get_memory_info":
                    continue
                if name.startswith("reset_") and not cold_resets:
                    with pytest.raises((RuntimeError, ValueError, TypeError)) as error:
                        getattr(torch.accelerator, name)(argument)
                    if argument == mojo_device:
                        assert "spawn" in str(error.value)
                    continue
                result = getattr(torch.accelerator, name)(argument)
                if name == "memory_stats":
                    assert isinstance(result, OrderedDict) and not result
                elif name.startswith("reset_"):
                    assert result is None
                else:
                    assert type(result) is int and result == 0
        with pytest.raises(RuntimeError, match="spawn"):
            torch.accelerator.get_memory_info(mojo_device)
        for _ in range(2):
            assert device_module.empty_cache() is None
            assert torch.accelerator.empty_cache() is None
        assert not torch._C._accelerator_isAllocatorInitialized()
        # Dropping inherited storage must also avoid the inherited mutex.
        held.clear()
        return "ok"

    out = _in_forked_child(child)
    assert out == "ok"


@pytest.mark.parametrize(
    "name,args",
    [
        ("memory_snapshot", ()),
        ("_record_memory_history", ()),
        ("_dump_snapshot", ()),
        ("caching_allocator_alloc", (1024,)),
        ("caching_allocator_delete", (0,)),
        ("set_per_process_memory_fraction", (0.5,)),
        ("list_gpu_processes", ()),
        ("host_memory_stats", ()),
        ("host_memory_stats_as_nested_dict", ()),
        ("reset_accumulated_host_memory_stats", ()),
        ("reset_peak_host_memory_stats", ()),
    ],
)
def test_unsupported_memory_apis(
    name: str, args: tuple[int | float, ...], mojo_gpu_available: bool
):
    if not mojo_gpu_available:
        pytest.skip("You do not have a GPU supported by MAX")
    with pytest.raises(NotImplementedError, match="MAX|Mojo"):
        getattr(device_module, name)(*args)
