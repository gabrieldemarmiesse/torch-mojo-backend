# Rewrite of: https://github.com/NVIDIA/nccl/blob/master/src/device/all_reduce.h
#
# NVLS (NVSwitch multicast) allreduce: the one collective in this library
# whose traffic goes through the switch's reduction engine instead of over
# unicast NVLink.
#
# Ported from the measured prototype (`nvls/mckernels.mojo` in the
# mojo_collectives worktree, schedule `-D nvls_pipe=1`, the shape its
# RESULTS.md section 6.4 recorded at 786 us / 168 MiB and 2217 us / 512 MiB
# against NCCL's 749 / 2143 on 8xH100). Three differences from the prototype,
# all of them because this is library code:
#
#   * the counters live in the region's existing signal area (the first page,
#     which include/device.mojo reserves) instead of a private header, so
#     the unicast kernels and this one share one region;
#   * they are UInt64, not UInt32 -- the prototype's u32 would wrap after
#     ~10^8 calls, about a day of DDP;
#   * the grid, the reducer share and the chunk size are runtime arguments,
#     not `-D` defines, so one build serves every message size.
#
# Shape = NCCL's runNVLS (nccl:src/device/all_reduce.h:410-457): every rank
# copies its input into its own bound region, rank r `multimem.ld_reduce`s
# slice r (the switch sums the eight contributions and returns one value) and
# `multimem.st`s the result back into all eight regions, and every rank copies
# the region out to its user tensor. Per GPU that is `bytes` of NVLink each
# way against `2*(world-1)/world*bytes` for a unicast reduce-scatter +
# all-gather -- 1.75x less at world 8 -- paid for with ~1.5x the HBM traffic,
# which is why it only wins above `NVLS_MIN_BYTES` (48 MiB, measured).
#
# The region must be VMM memory bound to a multicast object (transport/nvls.mojo); `mc`
# is the multicast VA, which only `multimem.*` may touch, and `uc` is a plain
# mapping of the same physical bytes, which is what the copies and the flag
# spin use. Everything here is behind `NVLS_ARCH` (sm_90+, where `multimem`
# exists) and, at runtime, behind the communicator having brought that region
# up; AMD and pre-Hopper NVIDIA keep the unicast kernels.
#
# Builds for both targets:
#   uv run --no-sync mojo build device/all_reduce.mojo --target-accelerator sm_90a
#   uv run --no-sync mojo build device/all_reduce.mojo --target-accelerator gfx942

from std.memory import AddressSpace, stack_allocation
from std.atomic import Atomic, Ordering
from max.gpu.memory import (
    Consistency,
    ReduceOp,
    multimem_ld_reduce,
    multimem_st,
)
from max.gpu.host import DeviceContext, DeviceStream
from max.gpu import (
    MAX_THREADS_PER_BLOCK_METADATA,
    block_idx,
    grid_dim,
    thread_idx,
)
from max.gpu.intrinsics import Scope
from std.utils import StaticTuple
from std.sys.info import _has_sm_9x_or_newer
from max.gpu.sync import barrier
from std.time import global_perf_counter_ns
from std.sys._assembly import inlined_assembly
from std.sys import size_of

from tmb.ccl.device.common import (
    _abort_raised,
    _enqueue_cached,
    abort_raised,
    latch_arena_error,
    publish_fault,
    spin_timeout_ns,
    status_page,
)
from tmb.ccl.include.device import BLOCK, FAULT_NO_PEER, signal_bytes


comptime NVLS_ARCH = _has_sm_9x_or_newer()
"""`multimem` is an sm_90+ instruction and RCCL has no equivalent, so every
line that emits one is behind this. A build for gfx942 or sm_80 elaborates
none of it; `nvls_available()` is the runtime half of the same gate."""

# ===-------------------------------------------------------------------=== #
# Region geometry. The three counters live in the first page of the signal
# area, which include/device.mojo holds free next to the error word at
# offset 0 (its own flag matrix starts at 4096).
# ===-------------------------------------------------------------------=== #

comptime FLAG_OFF = 64
"""Cross-GPU arrival counter, the address `multimem.red.add` targets."""
comptime ARRIVE_OFF = 128
"""This GPU's per-block arrival counter (plain memory, device scope)."""
comptime RELEASE_OFF = 192
"""What the last block of this GPU publishes to free the others."""

comptime ERR_NVLS_SYNC = 6
"""Error code written to the region's error word on a barrier timeout;
continues include/device.mojo's ERR_* numbering."""

comptime NVLS_MIN_BYTES = 48 * 1024 * 1024
"""Below this the unicast push/reduce/pull kernels win and keep the traffic.

Measured crossover, ABBA-interleaved against the unicast kernels over the same
user tensors on 8xH100 (prototype RESULTS.md 6.5): NVLS/unicast is 1.043 at
40 MiB and 0.963 at 48 MiB, then falls monotonically to 0.773 at 512 MiB. fp32
and bf16 land on the same crossover to within 0.3%, so one constant covers
both. Fitted on H100/NVSwitch; re-fit on another fabric."""

comptime NVLS_MAX_BLOCKS = 216
"""Grid cap, fitted on H100 (132 SMs). The barrier below is a FULL barrier --
every block of the grid must be resident at once or it deadlocks -- so
`nvls_blocks()` also clamps to what two blocks per SM allows, and
`nvvm.minctasm=2` makes ptxas leave room for two."""

comptime NVLS_REDUCE_PCT = 25
"""Share of the grid that drives the switch; the rest drives HBM.

Not 50: the switch phase saturates at about 33k threads (the whole-grid sweep
of the unpipelined kernel moves 2% between 64 and 264 blocks) while the copies
want every thread they can get. Measured at 168 MiB: 60% 927 us, 50% 877,
40% 843, 30% 824, 25% 819, 20% 805 -- and 25% wins once the chunk size is also
tuned (786 vs 805). H100."""

comptime _CHUNK_DIV = 4
comptime _CHUNK_MIN = 21 * 1024 * 1024
comptime _CHUNK_MAX = 86 * 1024 * 1024
"""Pipeline chunk = clamp(bytes / 4, 21 MiB, 86 MiB).

One barrier per chunk, worth about 10 us, against a shorter overlap window per
chunk. Measured at 168 MiB: 2.6 MiB chunks 1374 us, 5 MiB 1163, 10 MiB 1030,
21 MiB 927, 43 MiB 786; at 512 MiB 43 MiB gives 2217 and 86 MiB 2191. The
clamp fits every point measured (H100)."""

comptime _U = 2
"""16-byte vectors in flight per thread in the HBM copy loops."""

comptime _MMU = 2
"""multimem loads in flight per thread.

`multimem_ld_reduce`/`multimem_st` are `inlined_assembly` with a `~{memory}`
clobber, so the compiler may not reorder them: issuing `_MMU` loads before the
matching stores is the only way to get more than one switch round trip in
flight. Deeper does not help -- the switch phase at 168 MiB is 684 us with 2,
726 with 4, 777 with 8, i.e. a throughput ceiling of 250-270 GB/s per
direction, not a latency wall."""

comptime _SPIN_CHECK = 4096


def nvls_available() -> Bool:
    """Whether this build can emit `multimem` at all. The communicator also
    has to have brought a multicast region up; see transport/nvls.mojo."""
    return NVLS_ARCH


def nvls_min_bytes() -> Int:
    return NVLS_MIN_BYTES


def nvls_blocks(sm_count: Int) -> Int:
    """Grid for the NVLS kernel on a device with `sm_count` SMs.

    The full barrier needs the whole grid resident, so this never exceeds two
    blocks per SM (which `nvvm.minctasm=2` and the 256-thread block make fit at
    up to 128 registers). A grid that is not fully resident deadlocks -- the
    prototype hung at 216 blocks of 512 threads and at 396 blocks of 512, and
    ran at every geometry that stayed inside occupancy.
    """
    var by_occupancy = 2 * sm_count if sm_count > 0 else NVLS_MAX_BLOCKS
    return max(2, min(NVLS_MAX_BLOCKS, by_occupancy))


def nvls_chunk_bytes(total_bytes: Int) -> Int:
    """`clamp(total_bytes / 4, 21 MiB, 86 MiB)`; see `_CHUNK_DIV`."""
    var c = total_bytes // _CHUNK_DIV
    c = max(c, _CHUNK_MIN)
    return min(c, _CHUNK_MAX)


def nvls_chunk_vecs(chunk_bytes: Int, world: Int) -> Int:
    """Chunk size in 16-byte vectors, rounded to a whole number of per-rank
    slices so the reduce phase of every chunk divides evenly by `world`."""
    var q = max(1, world)
    var v = max(q, chunk_bytes // 16)
    return v // q * q


def nvls_padded_vecs(numel: Int, world: Int, esize: Int) -> Int:
    """16-byte vectors the payload occupies once padded up to a whole number
    of per-rank slices. `multimem.ld_reduce.v4` needs a full-width, 16-byte
    aligned operand (nccl:src/device/common_kernel.h:208-255 falls back to
    scalar multimem otherwise, at a large cost), so the copy-in zero-fills the
    pad rather than leaving a ragged tail."""
    var w = 16 // esize
    var q = max(1, world * w)
    return ((numel + q - 1) // q * q) // w


def nvls_barriers_per_call(numel: Int, world: Int, esize: Int, cv: Int) -> Int:
    """How far the host must advance the flag target for one call: the start
    barrier, the one that publishes the first chunk, and one per chunk."""
    var nvec = nvls_padded_vecs(numel, world, esize)
    return (nvec + cv - 1) // cv + 2


# ===-------------------------------------------------------------------=== #
# Device-side primitives
# ===-------------------------------------------------------------------=== #


@always_inline
def _mm_red_add_u64(
    addr: Pointer[UInt64, MutAnyOrigin, address_space=AddressSpace.GLOBAL],
    v: UInt64,
):
    """Add `v` to this address on every device bound to the multicast object.

    The Mojo stdlib wraps `multimem.ld_reduce` and `multimem.st` but not
    `multimem.red`. `.release.sys` so the payload stores this block made are
    visible to the other GPUs before the counter moves.
    """
    inlined_assembly[
        "multimem.red.release.sys.global.add.u64 [$0], $1;",
        NoneType,
        constraints="l,l,~{memory}",
        has_side_effect=True,
    ](addr.unsafe_bitcast[NoneType](), v)


@always_inline
def _record_error(uc: Pointer[UInt8, MutAnyOrigin], phase: Int, target: UInt64):
    """Record a multicast-barrier failure the way the unicast kernels do.

    The arena word for `ncclCommGetAsyncError`, and -- unless the host asked
    for this by raising the abort word -- the communicator's status page, so
    the next collective fails loudly instead of returning the reduction of an
    arena a peer never finished writing. There is no peer to name here: the
    barrier is a multicast counter, so what a reader gets is the block and the
    counter value it was waiting for.
    """
    if thread_idx.x == 0:
        if not latch_arena_error(
            uc.unsafe_bitcast[UInt64](), ERR_NVLS_SYNC, phase
        ):
            return
        var page = status_page(uc)
        if abort_raised(page):
            return
        publish_fault(
            page,
            ERR_NVLS_SYNC,
            phase,
            Int(block_idx.x),
            FAULT_NO_PEER,
            UInt64(0),
            target,
            Int(uc),
        )


@always_inline
def _nvls_sync(
    mc: Pointer[UInt8, MutAnyOrigin],
    uc: Pointer[UInt8, MutAnyOrigin],
    target: UInt64,
    t0: UInt64,
    timeout_ns: UInt64,
) -> Bool:
    """Full barrier: every block of every rank.

    It has to be full, not the block-index-matched barrier the unicast kernels
    use: there, block b only ever consumes bytes block b of a peer produced,
    but here the reduce phase reads a contiguous slice that EVERY block of
    every peer helped stage. The prototype measured the index-matched version
    passing every small case and failing from n = 65537 up, 3-6 wrong elements
    per rank -- exactly the kind of bug that ships.

    Level 1 is a device-scope arrival counter: the last block of this GPU to
    arrive has, by the release/acquire pair on it, every other block's payload
    stores in front of it. Level 2 is one
    `multimem.red.release.sys.global.add.u64`, which posts that GPU's arrival
    to all eight counters in one instruction; the same thread then waits on its
    own copy through the plain mapping (so the wait costs no fabric traffic)
    and releases the local blocks. Targets advance by `world` per barrier and
    are never reset, so a rank a whole call ahead cannot deadlock one behind.

    Returns False if the deadline passed or the host raised the abort word.
    The whole block learns that through shared memory so no thread is stuck
    inside a `barrier()`, and the local release is published anyway so the
    other blocks are not wedged either.
    """
    var failed = stack_allocation[
        1, DType.uint32, address_space=AddressSpace.SHARED
    ]()
    if thread_idx.x == 0:
        failed[unsafe_offset=0] = 0
    barrier()

    if thread_idx.x == 0:
        var nb = Int(grid_dim.x)
        var arrive = uc.unsafe_offset(ARRIVE_OFF).unsafe_bitcast[UInt64]()
        var release = uc.unsafe_offset(RELEASE_OFF).unsafe_bitcast[UInt64]()
        var mine = uc.unsafe_offset(FLAG_OFF).unsafe_bitcast[UInt64]()
        var seen = Atomic[Scalar[DType.uint64]].fetch_add(arrive, UInt64(1))
        var spins = 0
        # The grid is the same on every call of a process (`nvls_blocks` reads
        # only the device), so arrivals group cleanly into runs of `nb`.
        if (Int(seen) + 1) % nb == 0:
            comptime if NVLS_ARCH:
                _mm_red_add_u64(
                    mc.unsafe_offset(FLAG_OFF)
                    .unsafe_bitcast[UInt64]()
                    .unsafe_address_space_cast[AddressSpace.GLOBAL](),
                    UInt64(1),
                )
            while (
                Atomic[Scalar[DType.uint64]].load[ordering=Ordering.ACQUIRE](
                    mine
                )
                < target
            ):
                spins += 1
                if spins >= _SPIN_CHECK:
                    spins = 0
                    if _abort_raised(uc):
                        failed[unsafe_offset=0] = 1
                        break
                    # Same-GPU timer difference only, never across GPUs.
                    if global_perf_counter_ns() - t0 > timeout_ns:
                        failed[unsafe_offset=0] = 1
                        break
            Atomic[Scalar[DType.uint64]].store[ordering=Ordering.RELEASE](
                release, target
            )
        else:
            while (
                Atomic[Scalar[DType.uint64]].load[ordering=Ordering.ACQUIRE](
                    release
                )
                < target
            ):
                spins += 1
                if spins >= _SPIN_CHECK:
                    spins = 0
                    if _abort_raised(uc):
                        failed[unsafe_offset=0] = 1
                        break
                    if global_perf_counter_ns() - t0 > timeout_ns:
                        failed[unsafe_offset=0] = 1
                        break
    barrier()
    return failed[unsafe_offset=0] == 0


@always_inline
def _copy_vec[
    dtype: DType, W: Int
](
    dst: Pointer[Scalar[dtype], MutAnyOrigin],
    src: Pointer[Scalar[dtype], MutAnyOrigin],
    nvec: Int,
    tid: Int,
    stride: Int,
):
    """Grid-stride copy of `nvec` 16-byte vectors, `_U` of them in flight."""
    var v = tid
    var lim = nvec - (_U - 1) * stride
    while v < lim:
        var tmp = StaticTuple[SIMD[dtype, W], _U]()
        comptime for u in range(_U):
            tmp[u] = src.unsafe_load[width=W, alignment=16](
                (v + u * stride) * W
            )
        comptime for u in range(_U):
            dst.unsafe_store[width=W, alignment=16](
                (v + u * stride) * W, tmp[u]
            )
        v += _U * stride
    while v < nvec:
        dst.unsafe_store[width=W, alignment=16](
            v * W, src.unsafe_load[width=W, alignment=16](v * W)
        )
        v += stride


@always_inline
def _copy_vec_scaled[
    dtype: DType, W: Int
](
    dst: Pointer[Scalar[dtype], MutAnyOrigin],
    src: Pointer[Scalar[dtype], MutAnyOrigin],
    nvec: Int,
    tid: Int,
    stride: Int,
    scale: Float32,
):
    """`_copy_vec` times `scale`, through an fp32 accumulator for the half
    dtypes."""
    comptime accum = DType.float32 if (
        dtype == DType.bfloat16 or dtype == DType.float16
    ) else dtype
    var sv = SIMD[accum, W](scale.cast[accum]())
    var v = tid
    var lim = nvec - (_U - 1) * stride
    while v < lim:
        var tmp = StaticTuple[SIMD[dtype, W], _U]()
        comptime for u in range(_U):
            tmp[u] = src.unsafe_load[width=W, alignment=16](
                (v + u * stride) * W
            )
        comptime for u in range(_U):
            dst.unsafe_store[width=W, alignment=16](
                (v + u * stride) * W, (tmp[u].cast[accum]() * sv).cast[dtype]()
            )
        v += _U * stride
    while v < nvec:
        dst.unsafe_store[width=W, alignment=16](
            v * W,
            (
                src.unsafe_load[width=W, alignment=16](v * W).cast[accum]() * sv
            ).cast[dtype](),
        )
        v += stride


@always_inline
def _copy_in_span[
    dtype: DType, W: Int
](
    uc_pay: Pointer[Scalar[dtype], MutAnyOrigin],
    in_ptr: Pointer[Scalar[dtype], MutAnyOrigin],
    v0: Int,
    v1: Int,
    n: Int,
    tid: Int,
    stride: Int,
    scale: Float32,
):
    """Stage vectors [v0, v1) times `scale`; the vectors past element `n` are
    zero filled so the multimem phase runs on whole 16-byte operands (see
    `nvls_padded_vecs`).

    `scale` is applied HERE, to each rank's input, and not to the reduced
    value: `multimem.ld_reduce` accumulates in fp32 but returns the sum
    narrowed to the wire dtype, so two fp16 ranks averaging 40000 would read
    inf before any scaling. Pre-scaling the inputs is NCCL's PreMulSum for
    AVG (nccl:src/enqueue/enqueue.cc:2517); for a power-of-two world x/world
    is exact, so it costs no rounding.
    """
    comptime accum = DType.float32 if (
        dtype == DType.bfloat16 or dtype == DType.float16
    ) else dtype
    var nfull = n // W
    var lo = min(v0, nfull)
    var hi = min(v1, nfull)
    if scale == Float32(1.0):
        _copy_vec[dtype, W](
            uc_pay.unsafe_offset(lo * W),
            in_ptr.unsafe_offset(lo * W),
            hi - lo,
            tid,
            stride,
        )
    else:
        _copy_vec_scaled[dtype, W](
            uc_pay.unsafe_offset(lo * W),
            in_ptr.unsafe_offset(lo * W),
            hi - lo,
            tid,
            stride,
            scale,
        )
    var v = hi + tid
    while v < v1:
        var x = SIMD[dtype, W](0)
        comptime for k in range(W):
            var idx = v * W + k
            if idx < n:
                x[k] = (
                    in_ptr[unsafe_offset=idx].cast[accum]()
                    * scale.cast[accum]()
                ).cast[dtype]()
        uc_pay.unsafe_store[width=W, alignment=16](v * W, x)
        v += stride


@always_inline
def _copy_out_span[
    dtype: DType, W: Int
](
    out_ptr: Pointer[Scalar[dtype], MutAnyOrigin],
    uc_pay: Pointer[Scalar[dtype], MutAnyOrigin],
    v0: Int,
    v1: Int,
    n: Int,
    tid: Int,
    stride: Int,
):
    var nfull = n // W
    var lo = min(v0, nfull)
    var hi = min(v1, nfull)
    _copy_vec[dtype, W](
        out_ptr.unsafe_offset(lo * W),
        uc_pay.unsafe_offset(lo * W),
        hi - lo,
        tid,
        stride,
    )
    # The `n % W` elements no 16-byte vector covers, once, on the last chunk.
    if v1 * W >= n and n % W != 0:
        var i = nfull * W + tid
        while i < n:
            out_ptr[unsafe_offset=i] = uc_pay[unsafe_offset=i]
            i += stride


@always_inline
def _reduce_span[
    dtype: DType, W: Int
](
    mc_pay: Pointer[Scalar[dtype], MutAnyOrigin],
    v0: Int,
    v1: Int,
    tid: Int,
    stride: Int,
):
    """`multimem.ld_reduce` + `multimem.st` over vectors [v0, v1).

    Only the rank that owns the span ever touches it, so its read and its write
    need no ordering against each other across ranks: one barrier before (every
    rank's copy-in is in) and one after (every rank's store landed). Any AVG
    scale was applied at copy-in (`_copy_in_span`); the switch's sum is stored
    as is. Integer dtypes never reach this kernel.
    """
    var v = v0 + tid
    var lim = v1 - (_MMU - 1) * stride
    while v < lim:
        var acc = StaticTuple[SIMD[dtype, W], _MMU]()
        comptime for u in range(_MMU):
            acc[u] = multimem_ld_reduce[
                dtype,
                simd_width=W,
                reduction=ReduceOp.ADD,
                scope=Scope.SYSTEM,
                consistency=Consistency.RELAXED,
            ](
                mc_pay.unsafe_offset(
                    (v + u * stride) * W
                ).unsafe_address_space_cast[AddressSpace.GLOBAL]()
            )
        comptime for u in range(_MMU):
            multimem_st[
                dtype,
                simd_width=W,
                scope=Scope.SYSTEM,
                consistency=Consistency.RELAXED,
            ](
                mc_pay.unsafe_offset(
                    (v + u * stride) * W
                ).unsafe_address_space_cast[AddressSpace.GLOBAL](),
                acc[u],
            )
        v += _MMU * stride
    while v < v1:
        var a = mc_pay.unsafe_offset(v * W).unsafe_address_space_cast[
            AddressSpace.GLOBAL
        ]()
        var one = multimem_ld_reduce[
            dtype,
            simd_width=W,
            reduction=ReduceOp.ADD,
            scope=Scope.SYSTEM,
            consistency=Consistency.RELAXED,
        ](a)
        multimem_st[
            dtype,
            simd_width=W,
            scope=Scope.SYSTEM,
            consistency=Consistency.RELAXED,
        ](a, one)
        v += stride


# ===-------------------------------------------------------------------=== #
# The kernel
# ===-------------------------------------------------------------------=== #


@__llvm_metadata(
    MAX_THREADS_PER_BLOCK_METADATA=StaticTuple[Int32, 1](Int32(BLOCK)),
    `nvvm.minctasm`=SIMDLength(2),
)
@__name(t"ccl_nvls_allreduce_multimem_{dtype}")
def _nvls_ar_kernel[
    dtype: DType, W: Int
](
    mc: Pointer[UInt8, MutAnyOrigin],
    uc: Pointer[UInt8, MutAnyOrigin],
    in_ptr: Pointer[Scalar[dtype], MutAnyOrigin],
    out_ptr: Pointer[Scalar[dtype], MutAnyOrigin],
    numel: Int64,
    chunk_vecs: Int64,
    payload_off: Int64,
    rank_i: Int32,
    world_i: Int32,
    reduce_blocks: Int32,
    scale: Float32,
    target: UInt64,
    timeout_ns: UInt64,
):
    """Split grid: the low `reduce_blocks` blocks only drive the switch, the
    rest only drive HBM, so chunk c's reduction runs at the same time as chunk
    c+1's copy-in and chunk c-1's copy-out. One barrier per chunk covers all
    three -- they touch disjoint byte ranges of the region.

    The alternative (every thread both reduces and copies, with the copies
    issued inside the switch round trip) was built and measured 6% slower --
    830 us against 786 at 168 MiB -- because a thread's copy pairs are a chain
    of dependent load-then-store pairs that serialize on HBM latency. See the
    prototype's section 6.3.
    """
    var t0 = global_perf_counter_ns()
    var n = Int(numel)
    var world = Int(world_i)
    var rank = Int(rank_i)
    var q = world * W
    var nvec = ((n + q - 1) // q * q) // W
    var uc_pay = uc.unsafe_offset(Int(payload_off)).unsafe_bitcast[
        Scalar[dtype]
    ]()
    var mc_pay = mc.unsafe_offset(Int(payload_off)).unsafe_bitcast[
        Scalar[dtype]
    ]()

    var nb = Int(grid_dim.x)
    var rb = min(max(1, Int(reduce_blocks)), nb - 1)
    var is_reducer = Int(block_idx.x) < rb
    var rtid = Int(block_idx.x) * BLOCK + Int(thread_idx.x)
    var rstride = rb * BLOCK
    var ctid = (Int(block_idx.x) - rb) * BLOCK + Int(thread_idx.x)
    var cstride = (nb - rb) * BLOCK
    var cv = Int(chunk_vecs)
    var nch = (nvec + cv - 1) // cv
    var bar = target

    # Start barrier, before a single byte of staging is written.
    #
    # The whole `2 * cap` arena is shared scratch, and the one rule that keeps
    # collectives of different shapes and sizes from colliding in it is that
    # every kernel opens with a barrier: no generation writes the arena until
    # every rank has finished READING it for the previous one. This kernel
    # needs that rule for the same reason the unicast ones do -- a broadcast or
    # an all-gather retiring just before it has its peers reading my staging,
    # which is exactly where the copy-in below writes. It is one barrier
    # (~8 us) per call, not per chunk.
    if not _nvls_sync(mc, uc, bar, t0, timeout_ns):
        _record_error(uc, 0, bar)
        return
    bar += UInt64(world)

    if not is_reducer:
        _copy_in_span[dtype, W](
            uc_pay, in_ptr, 0, min(cv, nvec), n, ctid, cstride, scale
        )
    if not _nvls_sync(mc, uc, bar, t0, timeout_ns):
        _record_error(uc, 1, bar)
        return
    bar += UInt64(world)

    for c in range(nch):
        var s0 = c * cv
        var s1 = min(s0 + cv, nvec)
        if is_reducer:
            var per = (s1 - s0) // world
            _reduce_span[dtype, W](
                mc_pay,
                s0 + rank * per,
                s0 + rank * per + per,
                rtid,
                rstride,
            )
        else:
            if c + 1 < nch:
                _copy_in_span[dtype, W](
                    uc_pay,
                    in_ptr,
                    s1,
                    min(s1 + cv, nvec),
                    n,
                    ctid,
                    cstride,
                    scale,
                )
            if c > 0:
                _copy_out_span[dtype, W](
                    out_ptr, uc_pay, (c - 1) * cv, s0, n, ctid, cstride
                )
        if not _nvls_sync(mc, uc, bar, t0, timeout_ns):
            _record_error(uc, c + 2, bar)
            return
        bar += UInt64(world)

    if not is_reducer:
        _copy_out_span[dtype, W](
            out_ptr, uc_pay, (nch - 1) * cv, nvec, n, ctid, cstride
        )


# ===-------------------------------------------------------------------=== #
# Public API
# ===-------------------------------------------------------------------=== #


def nvls_allreduce[
    dtype: DType
](
    ctx: DeviceContext,
    stream: DeviceStream,
    rank: Int,
    world: Int,
    mc_base: Int,
    uc_base: Int,
    in_ptr: Int,
    out_ptr: Int,
    numel: Int,
    payload_bytes: Int,
    scale: Float32,
    target: Int,
    blocks: Int,
    chunk_vecs: Int,
) raises:
    """Elementwise `out[i] = scale * sum over ranks of in_r[i]` through the
    NVSwitch, on `stream`. `in_ptr` may equal `out_ptr`.

    `mc_base`/`uc_base` are this rank's multicast and unicast mappings of the
    same region; the payload starts at `signal_bytes()` and may use
    `payload_bytes`, which is the WHOLE `2 * cap` arena -- this kernel needs
    one buffer, not two, and the start barrier above is what lets it spread
    over the same bytes a broadcast or a one-shot allreduce uses. The counters
    live in the signal area. `target` is the flag value the FIRST of this
    call's barriers waits for; the caller advances its own counter by
    `world * nvls_barriers_per_call(...)`.
    """
    comptime if not NVLS_ARCH:
        raise Error("collectives: this build has no NVLS path (needs sm_90+)")
    comptime W = 16 // size_of[dtype]()
    comptime if dtype.is_integral():
        raise Error("collectives: the NVLS path is float only")
    if numel <= 0:
        return
    if world < 2 or world > 8:
        raise Error("collectives: the NVLS path needs 2..8 local ranks")
    if rank < 0 or rank >= world:
        raise Error("collectives: rank out of range")
    if (in_ptr | out_ptr) % 16 != 0:
        raise Error("collectives: NVLS needs 16-byte aligned in_ptr/out_ptr")
    if mc_base == 0 or uc_base == 0:
        raise Error("collectives: NVLS called without a multicast region")
    var nvec = nvls_padded_vecs(numel, world, size_of[dtype]())
    # The pad rounds the payload up to a whole per-rank slice, so the region
    # has to hold slightly more than the message.
    if nvec * 16 > payload_bytes:
        raise Error("collectives: NVLS message exceeds the staging arena")
    if chunk_vecs < world or chunk_vecs % world != 0:
        raise Error("collectives: NVLS chunk must be whole per-rank slices")
    var rb = max(1, min(blocks - 1, blocks * NVLS_REDUCE_PCT // 100))
    comptime if NVLS_ARCH:
        _enqueue_cached[_nvls_ar_kernel[dtype, W]](
            ctx,
            stream,
            String(t"nvls_{dtype}"),
            blocks,
            Pointer[UInt8, MutAnyOrigin](unsafe_from_address=mc_base),
            Pointer[UInt8, MutAnyOrigin](unsafe_from_address=uc_base),
            Pointer[Scalar[dtype], MutAnyOrigin](unsafe_from_address=in_ptr),
            Pointer[Scalar[dtype], MutAnyOrigin](unsafe_from_address=out_ptr),
            Int64(numel),
            Int64(chunk_vecs),
            Int64(signal_bytes()),
            Int32(rank),
            Int32(world),
            Int32(rb),
            scale,
            UInt64(target),
            spin_timeout_ns(),
        )
