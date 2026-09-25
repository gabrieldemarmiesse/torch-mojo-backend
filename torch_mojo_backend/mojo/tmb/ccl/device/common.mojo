# Rewrite of: https://github.com/NVIDIA/nccl/blob/master/src/device/common.h
#
# Every spin is bounded (`MOJOCCL_IB_TIMEOUT_S`, default 60 s, measured with
# the GPU's own timer -- never compared across GPUs).  On timeout the kernel
# stores a nonzero code into its own region's error word (byte
# `error_offset()`), latches the failure in the communicator's pinned status
# page (`publish_fault`, so the host can see it without synchronizing a
# stream) and returns instead of hanging the node.  The same spins also leave
# early when the host raises the communicator's abort word
# (`install_status_page`, `ncclCommAbort`), which is what makes abort prompt
# instead of costing a full deadline.

from std.atomic import Ordering, fence, Atomic
from std.sys import llvm_intrinsic, has_amd_gpu_accelerator
from std.ffi import _get_global_or_null, external_call
from std.os import getenv
from std.memory.alloc import unsafe_alloc
from std.time import global_perf_counter_ns
from max.gpu.host import DeviceAttribute, DeviceContext, DeviceStream
from std.builtin.device_passable import DevicePassable
from max.gpu.host.launch_attribute import (
    LaunchAttribute,
    LaunchAttributeID,
    LaunchAttributeValue,
)
from max.gpu import grid_dim

from tmb.ccl.env_vars import MOJOCCL_IB_TIMEOUT_S
from tmb.ccl.include.device import (
    BLOCK,
    DEFAULT_TIMEOUT_NS,
    FAULT_ARENA,
    FAULT_BLOCK,
    FAULT_CODE,
    FAULT_PEER,
    FAULT_PHASE,
    FAULT_SEEN,
    FAULT_TARGET,
    STATUS_ABORT_WORD,
    STATUS_FAULT_WORD,
    _AMD,
    _GFX942,
    _STATUS_PTR_OFFSET,
)


# gfx942 polls relaxed and acquires once (`poll_acquire`). MI300A, job
# 5447705: FSDP2 comm busy 209.5 -> 188.5 ms/step (agents_docs/distributed.md).
comptime _POLL_ORDER = Ordering.RELAXED if _GFX942 else Ordering.ACQUIRE


@always_inline
def poll_pause():
    """RCCL 2.22.3's gfx942 waitPeer sleep (prims_simple.h); no other target."""
    comptime if _GFX942:
        llvm_intrinsic["llvm.amdgcn.s.sleep", NoneType, has_side_effect=True](
            Int32(1)
        )


@always_inline
def poll_acquire():
    """Acquire the release observed by a successful relaxed flag poll.

    The system-scope atomic read is sequenced before this system acquire
    fence, so the release it reads synchronizes with the fence (the standard
    atomic-to-fence rule). LLVM lowers gfx942's monotonic system load to
    `global_load ... sc0 sc1`, and this fence to waitcnt + buffer_inv sc0 sc1.
    The invalidate happens once, before consuming payload, instead of on
    every failed poll while independent compute is using L2. Every polling
    thread fences, including an already-satisfied flag, before the block
    barrier passes the acquire to the payload-reading threads.

    Call it where the wave has reconverged, never inside the branch that
    polls: there the spin loop's exit leaves EXEC = 0 and the gfx942
    invalidate ran with no lanes (agents_docs/mojo_collectives_kernel_results.md
    section 7).
    """
    comptime if _GFX942:
        fence[ordering=Ordering.ACQUIRE]()


def spin_timeout_ns() -> UInt64:
    """The deadline every device spin in this library is launched with.

    `MOJOCCL_IB_TIMEOUT_S` governs it, the same variable the inter-node
    transport's own waits use: a rank waiting on a peer that stopped
    answering should give up after one interval, not two different ones
    depending on which side of the hierarchy the peer is. Read from the
    environment once per process and cached in a process global -- a getenv
    and a float parse per collective would be a measurable slice of a 27 MiB
    allreduce's 164 us.
    """
    var g = _get_global_or_null("CCL_SPIN_TIMEOUT_NS")
    if g:
        return g.value().unsafe_bitcast[UInt64]()[unsafe_offset=0]
    var ns = UInt64(DEFAULT_TIMEOUT_NS)
    var raw = getenv(MOJOCCL_IB_TIMEOUT_S, String(""))
    if raw != String(""):
        try:
            var seconds = Float64(raw)
            if seconds > 0.0:
                ns = UInt64(seconds * 1.0e9)
        except:
            # Unparseable value: keep the 60 s default rather than a silent 0.
            ns = UInt64(DEFAULT_TIMEOUT_NS)
    var slot = unsafe_alloc[UInt64](1)
    slot[unsafe_offset=0] = ns
    external_call["KGEN_CompilerRT_InsertGlobal", NoneType](
        StringSlice("CCL_SPIN_TIMEOUT_NS"), slot.unsafe_bitcast[NoneType]()
    )
    return ns


@always_inline
def status_page(region: Pointer[UInt8, MutAnyOrigin]) -> Int:
    """Device address of this communicator's status page, or 0 if none was
    installed (a region built by a test harness, or one whose header has not
    been published yet)."""
    return Int(
        region.unsafe_offset(_STATUS_PTR_OFFSET).unsafe_bitcast[UInt64]()[
            unsafe_offset=0
        ]
    )


@always_inline
def status_word(page: Int, index: Int) -> Pointer[UInt64, MutAnyOrigin]:
    return Pointer[UInt64, MutAnyOrigin](unsafe_from_address=page + index * 8)


@always_inline
def abort_raised(page: Int) -> Bool:
    """Whether this communicator has stopped -- by `ncclCommAbort`, or by a
    device deadline (`publish_fault` raises the same word).

    The one predicate a kernel needs before it does anything irreversible,
    and deliberately one word: a reader on the device pays a PCIe round trip
    per load of this page.
    """
    if page == 0:
        return False
    return (
        Atomic[Scalar[DType.uint64]].load[ordering=Ordering.ACQUIRE](
            status_word(page, STATUS_ABORT_WORD)
        )
        != 0
    )


@always_inline
def fault_latched(page: Int) -> Bool:
    """Whether a device deadline has already been latched here.

    Once it has, nothing this communicator does is trustworthy any more: some
    block gave up waiting for a peer, so an arena holds bytes nobody produced.

    Distinguishes a deadline from an abort, which `abort_raised` does not (a
    deadline raises the abort word too). Only `publish_fault`'s own guard
    needs the distinction on the device; every other device reader wants
    `abort_raised`, which is half the loads.
    """
    if page == 0:
        return False
    return (
        Atomic[Scalar[DType.uint64]].load[ordering=Ordering.ACQUIRE](
            status_word(page, STATUS_FAULT_WORD + FAULT_CODE)
        )
        != 0
    )


@always_inline
def _abort_raised(region: Pointer[UInt8, MutAnyOrigin]) -> Bool:
    """`abort_raised` for a caller that has a region rather than a page.

    Two dependent loads, and only from the slow path of a spin: the page's
    device address out of my own region's header (written once at communicator
    init, never again) and then the abort word itself, which lives in host
    memory and is where `ncclCommAbort` stores.
    """
    return abort_raised(status_page(region))


@always_inline
def latch_arena_error(
    err_word: Pointer[UInt64, MutAnyOrigin], code: Int, phase: Int
) -> Bool:
    """Store `code * 1_000_000 + phase` into an arena's error word if it is
    still clear; True if this thread is the one that did it.

    First writer wins, deliberately. The first deadline is the one that
    explains the run: once a rank stops publishing flags, every later
    collective on that arena times out too, and overwriting would leave only
    the last consequence. Device memory, so this is an ordinary global atomic.
    """
    var expected = UInt64(0)
    return Atomic[Scalar[DType.uint64]].compare_exchange[
        success_ordering=Ordering.RELEASE,
        failure_ordering=Ordering.RELAXED,
    ](err_word, expected, UInt64(code) * 1_000_000 + UInt64(phase))


@always_inline
def publish_fault(
    page: Int,
    code: Int,
    phase: Int,
    block: Int,
    peer: UInt64,
    seen: UInt64,
    target: UInt64,
    arena: Int,
):
    """Latch a device deadline in the status page, for the host to print.

    Plain stores for the detail words and one release store for the code --
    the same kind of write `_proxy_request_kernel` has always made into pinned
    host memory, so this needs nothing of the hardware that the transport does
    not already need. The guard is a read of the code word rather than a
    compare-exchange: host-memory atomics are a portability question this
    library does not have to open, and the caller has already won a
    compare-exchange on its arena's error word, so the only race left is two
    ARENAS failing within the same microsecond -- two descriptions of one
    episode, not two episodes.
    """
    if page == 0 or fault_latched(page):
        return
    status_word(page, STATUS_FAULT_WORD + FAULT_PHASE)[
        unsafe_offset=0
    ] = UInt64(phase)
    status_word(page, STATUS_FAULT_WORD + FAULT_BLOCK)[
        unsafe_offset=0
    ] = UInt64(block)
    status_word(page, STATUS_FAULT_WORD + FAULT_PEER)[unsafe_offset=0] = peer
    status_word(page, STATUS_FAULT_WORD + FAULT_SEEN)[unsafe_offset=0] = seen
    status_word(page, STATUS_FAULT_WORD + FAULT_TARGET)[
        unsafe_offset=0
    ] = target
    status_word(page, STATUS_FAULT_WORD + FAULT_ARENA)[
        unsafe_offset=0
    ] = UInt64(arena)
    # Last, and with release: a nonzero code promises the six words above.
    Atomic[Scalar[DType.uint64]].store[ordering=Ordering.RELEASE](
        status_word(page, STATUS_FAULT_WORD + FAULT_CODE), UInt64(code)
    )
    # And raise the abort word: "this communicator has failed" is then ONE
    # word for every device reader, which matters on the inter-node release
    # path, where each load of this pinned page is a PCIe round trip per
    # exchange (see `proxy._proxy_main`). Ordered after the code, so a
    # reader that sees the word raised finds a complete record.
    Atomic[Scalar[DType.uint64]].store[ordering=Ordering.RELEASE](
        status_word(page, STATUS_ABORT_WORD), UInt64(1)
    )


@always_inline
def device_now_ns() -> UInt64:
    """Device-side clock for the spin deadlines, wrap-safe on AMD.

    The stdlib's `global_perf_counter_ns` on AMD returns
    `(s_memrealtime_ticks * 1_000_000_000) // 100_000_000` in UInt64: the
    product overflows 184 s after the GPU's counter started, and from then on
    the value is a saw-tooth with a 184.47 s period. A spin whose start and
    poll straddle a wrap computes `now - t0` as an enormous unsigned number
    and fires its deadline at once -- the block records the error word and
    returns while its peers wait a real 60 s for flags it never publishes,
    and the collective completes with garbage on that node. Measured on
    Adastra (2x4 MI300A): about one 40 s stress run in five corrupted, always
    a run 60-120 s longer than a clean one. Reading the 100 MHz counter
    directly and scaling by 10 keeps differences exact for centuries. NVIDIA's
    `globaltimer` is nanoseconds already and is left as it was.
    """
    comptime if has_amd_gpu_accelerator():
        return (
            llvm_intrinsic[
                "llvm.amdgcn.s.memrealtime", UInt64, has_side_effect=True
            ]()
            * 10
        )
    else:
        return global_perf_counter_ns()


# ===-------------------------------------------------------------------=== #
# Cached launch (compile_function costs ~180 us per call; do it once)
# ===-------------------------------------------------------------------=== #


@always_inline
def _cached_function[
    declared_arg_types: TypeList[Trait=AnyType, ...],
    //,
    func: def(* args: * declared_arg_types) thin -> None,
](ctx: DeviceContext, key: String) raises -> Pointer[
    type_of(ctx.compile_function[func]()), MutUntrackedOrigin
]:
    """Compile `func` at most once per process and context (same caching
    pattern as the repo's eager kernels)."""
    var name = String(t"CCL_KERNEL_{key}_{ctx.id()}")
    comptime FuncT = type_of(ctx.compile_function[func]())
    var global_ptr = _get_global_or_null(name)
    if global_ptr:
        return global_ptr.value().unsafe_bitcast[FuncT]()
    var compiled = ctx.compile_function[func]()
    var fptr = unsafe_alloc[FuncT](1)
    fptr.unsafe_write(compiled^)
    external_call["KGEN_CompilerRT_InsertGlobal", NoneType](
        StringSlice(name), fptr.unsafe_bitcast[NoneType]()
    )
    return fptr


def _cached_occupancy[
    declared_arg_types: TypeList[Trait=AnyType, ...],
    //,
    func: def(* args: * declared_arg_types) thin -> None,
](ctx: DeviceContext, key: String, threads: Int) raises -> Int:
    """The driver's occupancy answer for `func` at `threads` per block --
    blocks per multiprocessor that can be active at once -- which is what a
    kernel whose blocks wait for each other has to size its grid by
    (`all_reduce_gin.mojo`). Asked once per (kernel, context)."""
    var name = String(t"CCL_OCC_{key}_{ctx.id()}")
    var global_ptr = _get_global_or_null(name)
    if global_ptr:
        return global_ptr.value().unsafe_bitcast[Int]()[]
    var f = _cached_function[func](ctx, key)
    var occ = f[].occupancy_max_active_blocks_per_multiprocessor(threads, 0)
    var p = unsafe_alloc[Int](1)
    p.unsafe_write(occ)
    external_call["KGEN_CompilerRT_InsertGlobal", NoneType](
        StringSlice(name), p.unsafe_bitcast[NoneType]()
    )
    return occ


def _enqueue_cached[
    declared_arg_types: TypeList[Trait=AnyType, ...],
    //,
    func: def(* args: * declared_arg_types) thin -> None,
    *Ts: DevicePassable,
](
    ctx: DeviceContext,
    stream: DeviceStream,
    key: String,
    blocks: Int,
    *args: *Ts,
) raises:
    """`_enqueue_cached_dim` at this file's `BLOCK` threads per block."""
    _enqueue_cached_dim[func](ctx, stream, key, blocks, BLOCK, False, *args)


def _enqueue_cached_dim[
    declared_arg_types: TypeList[Trait=AnyType, ...],
    //,
    func: def(* args: * declared_arg_types) thin -> None,
    *Ts: DevicePassable,
](
    ctx: DeviceContext,
    stream: DeviceStream,
    key: String,
    blocks: Int,
    threads: Int,
    cooperative: Bool,
    *args: *Ts,
) raises:
    """Enqueue `func` on `stream`. `ctx` is only the compilation/caching
    handle -- the launch always goes to the stream the caller handed us, which
    in production is a foreign `cudaStream_t` wrapped by
    `DeviceContext.create_external_stream`. `threads` must not exceed the
    kernel's MAX_THREADS_PER_BLOCK_METADATA.

    `cooperative` asks the driver for a co-resident grid (CUDA's
    `CU_LAUNCH_ATTRIBUTE_COOPERATIVE`, where the device supports it): the
    launch is refused outright, instead of deadlocking at the first grid
    barrier, when the grid cannot be resident at once. MAX's launch
    attributes are CUDA-only, so on AMD the occupancy bound the caller
    applied is the whole guarantee."""
    var f = _cached_function[func](ctx, key)
    var attrs = List[LaunchAttribute]()
    comptime if not _AMD:
        if (
            cooperative
            and ctx.get_attribute(DeviceAttribute.COOPERATIVE_LAUNCH) != 0
        ):
            attrs.append(
                LaunchAttribute(
                    id=LaunchAttributeID.COOPERATIVE,
                    value=LaunchAttributeValue(True),
                )
            )
    stream.enqueue_function(
        f[],
        *args,
        grid_dim=(blocks,),
        block_dim=(threads,),
        attributes=attrs^,
    )
