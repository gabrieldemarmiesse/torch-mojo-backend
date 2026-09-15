# Intra-node collectives (allreduce / broadcast / allgather) over IPC-mapped
# peer regions -- a Mojo replacement for the NCCL/RCCL calls a DDP
# ProcessGroup makes inside one node.
#
# One library-owned device allocation per rank ("the region"), shared to the
# peers by cuIpc/hipIpc (the plumbing layer owns that half), laid out as
#
#   [0, SIGNAL_BYTES)                            signal area, zero at creation
#   [SIGNAL_BYTES, SIGNAL_BYTES + cap)           "stage_in"
#   [SIGNAL_BYTES + cap, SIGNAL_BYTES + 2*cap)   "stage_out"
#
# The two cap-sized halves are library scratch; this file uses them as one
# 2*cap byte staging arena and places its own sub-areas inside it, always
# within those bounds:
#   allreduce two-shot: `world` push slots of max-shard size, then the reduced
#                       shard (`_launch_allreduce`, which raises if that ever
#                       fails to fit -- it cannot, since world*shard ~ numel
#                       <= cap and the arena is 2*cap).  On AMD that last area
#                       is `world-1` slots instead of one -- the gather slots
#                       peers push their reduced shards into -- at the same
#                       base offset, so the NVIDIA layout is byte for byte
#                       what it was;
#   allreduce one-shot: two generation-parity halves of `world` whole-message
#                       slots (used only when they fit);
#   broadcast/allgather: the base of the arena, one message-sized stage per
#                        rank (the caller chunks anything larger than cap).
#   split allreduce:    `world-1` compacted push slots at the base of
#                       stage_in, and the reduced shard at its own element
#                       offset inside stage_out (see `shard_range`), so the
#                       inter-node library can rewrite it in place between the
#                       two halves.
#
# Why the data path looks like it does
# ------------------------------------
# The user's tensors live in MAX-allocated memory, which cannot be exported
# with legacy IPC (measured: `cuIpcGetMemHandle` -> CUDA_ERROR_INVALID_VALUE),
# so peers can only ever read the region.  The naive way to bridge that -- copy
# the input into stage_in, run a direct reduce-scatter + all-gather over the
# regions, copy the result back out -- costs two extra full HBM round trips
# (+45 us on the 27 MiB GPT-2 bucket, measured in the feasibility study, 210 us
# vs 165 us direct).  Both copies are avoidable:
#
#   phase 1  PUSH   every rank reads its own input straight out of user memory
#                   and writes shard s into peer s's slot `rank` of the arena.
#                   That write IS the copy-in, and it travels over NVLink.
#   phase 2  REDUCE each rank sums the `world` contributions to its own shard
#                   (its own straight from user memory, the peers' from the
#                   arena), scales, and writes the result to the arena and to
#                   its slice of the user output.
#   phase 3  PULL   each rank reads the other ranks' reduced shards out of
#                   their arenas directly into its user output.
#
# NVLink traffic is 2*(world-1)/world * bytes per GPU -- the unicast minimum,
# the same as a direct reduce-scatter + all-gather -- and no byte is copied
# locally that the direct kernel would not also copy.  The staging is free.
#
# Link direction: AMD (gfx942 / MI300A) takes a different phase 3
# ------------------------------------------------------------------
# On an xGMI mesh a GPU-initiated remote *read* does not scale across links
# while a remote *write* does.  Measured on a 4x MI300A node with the copy
# loop below (perf-work/linkbw.mojo, 168 MiB, per-GPU GB/s): one link 91
# either way; three peers written at once 233; three peers read at once 93,
# and a ring of simultaneous readers 56.  RCCL reaches the same 236 GB/s at
# 512 MiB and gets there the same way -- its P2P transport hard-wires
# `read = 0` on AMD (rccl:src/graph/paths.cc:441 only lets compCap 80 read),
# so the sender stores into the receiver's buffer and no rank ever loads
# across a link.
#
# So on AMD phase 3 is a second push instead of a pull:
#
#   phase 2' each rank writes its reduced shard into EVERY peer's gather slot
#            (and into its own slice of the user output),
#   phase 3' each rank copies the `world-1` gather slots of its OWN region
#            into the user output -- a local HBM copy, because the user's
#            output cannot be IPC-mapped and so a peer cannot write it
#            directly.
#
# The cross-link traffic is bit for bit the same 2*(world-1)/world * bytes; it
# has only changed direction.  The price is that local copy, and it is small:
# the region is `hipDeviceMallocUncached` but reads out of it at full HBM rate
# (1434 GB/s measured against 1453 for a normal buffer), so 0.75 * message
# costs ~90 us at 168 MiB against the ~1130 us the wire needs.
#
# NVIDIA keeps the pull: behind a switch every direction is equivalent, the
# pull needs no gather slots and no local copy, and the H100 numbers in
# docs/mojo_collectives_kernel_results.md were measured with it.  The split is
# a `comptime if has_amd_gpu_accelerator()` in the kernel and in the one host
# line that sizes the arena, so NVIDIA device code is unchanged.
#
# Synchronisation
# ---------------
# The signal area holds one UInt64 flag per (block, writer rank).  Rank r's
# block b publishes `generation * PHASES_PER_GEN + phase` into every peer's
# flags[b][r] with a system-scope release store and then waits for its own
# flags[b][p] to reach that value for every peer p.  Flag values are derived
# from `generation`, never reset, and compared with `>=`, so a peer that is a
# whole call ahead can never deadlock a peer that is behind and nothing has to
# be cleared between calls.  Every collective opens with a start barrier, so
# one rule covers the whole arena: no generation writes it until every rank has
# finished reading the previous generation.  That is what lets collectives of
# different kinds and sizes share the staging area -- see the invariant above
# `_ar_twoshot_kernel`.
#
# The hierarchical (multi-node) allreduce splits that into
# `reduce_scatter_stage` (phases 1-2) and `allgather_finish` (phase 3), with
# the vendor library's inter-node allreduce of one shard in between; the split
# spends two generations and one extra launch. See the block comment above
# `_rs_stage_kernel`.
#
# Broadcast is scatter + all-gather, not "root stages, everyone reads": the
# latter puts (world-1) x nbytes on the root's one outbound link and measured
# 3.6x slower.  Allgather is a local stage + a peer gather, which is already
# the unicast minimum.
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
#
# Portability: NVIDIA and AMD share every line of the device code except the
# two places the header calls out (phase 3 of the allreduce, and the fence
# discipline).  On NVIDIA ordering is `Atomic[...].store[RELEASE]` /
# `load[ACQUIRE]` at default (system) scope, which lowers to
# `st.release.sys.global` / `ld.acquire.sys.global` on sm_90a.  On gfx942 the
# region is uncached and the flags are published with `s_waitcnt lgkmcnt(0)
# vmcnt(0)` plus a relaxed store -- RCCL's "cheap post-send fence", see
# `_sync`.  Blocks are 256 threads (RCCL's gfx942 maximum) and every layout is
# wave-64 safe.
#
# Every host function takes the DeviceStream to enqueue on (production wraps
# the caller's foreign cudaStream_t with `DeviceContext.create_external_stream`);
# `ctx` is only the handle used to compile and cache the DeviceFunction.
#
# Builds for both targets:
#   uv run --no-sync mojo build collectives_kernels.mojo --target-accelerator sm_90a
#   uv run --no-sync mojo build collectives_kernels.mojo --target-accelerator gfx942

from std.atomic import Atomic, Ordering, fence
from std.builtin.device_passable import DevicePassable
from std.collections import InlineArray
from std.ffi import _get_global_or_null, external_call
from std.os import getenv
from std.gpu import (
    MAX_THREADS_PER_BLOCK_METADATA,
    block_idx,
    global_idx,
    grid_dim,
    thread_idx,
)
from max.gpu.host import DeviceAttribute, DeviceContext, DeviceStream
from max.gpu.host.launch_attribute import (
    LaunchAttribute,
    LaunchAttributeID,
    LaunchAttributeValue,
)
from max.gpu.sync import barrier
from std.memory import AddressSpace, stack_allocation
from std.memory.alloc import unsafe_alloc
from std.sys import (
    get_defined_int,
    has_amd_gpu_accelerator,
    size_of,
)
from std.sys import llvm_intrinsic
from std.time import global_perf_counter_ns
from std.utils import StaticTuple

# ===-------------------------------------------------------------------=== #
# Compile-time configuration
# ===-------------------------------------------------------------------=== #

comptime _AMD = has_amd_gpu_accelerator()
"""Whether this build targets AMD.  Every behavioural difference in this file
is behind it, so the NVIDIA path is exactly what it was before the MI300A work
(see the "Link direction" note in the module header)."""

comptime MAX_WORLD = 8
"""Largest world size a single region can address (one flag column per rank)."""

comptime BLOCK = 256
"""Threads per block. 256 is RCCL's gfx942 maximum and is wave-64 safe."""

comptime MAX_BLOCKS = 1024
"""Flag rows in the signal area; every grid this file launches is <= this."""

comptime _FLAG_BYTE_OFFSET = 4096
"""Start of the flag matrix inside the signal area (the first page holds the
error word, the NVLS barrier counters and the abort-word pointer)."""

comptime _STATUS_PTR_OFFSET = 256
"""Header slot holding the DEVICE address of the communicator's pinned host
status page (below). Zero until `install_status_page` publishes one, and every
spin reads zero as "this region has no status page". 64/128/192 are the NVLS
barrier counters (nvls_kernels.mojo), so 256 is the first free line."""

comptime _GRIDBAR_ARRIVE_OFFSET = 320
comptime _GRIDBAR_RELEASE_OFFSET = 384
"""The two words of `grid_barrier`, on separate cache lines so the arrival
counter's traffic never invalidates the line every block is spinning on.
Purely rank-local -- no peer ever reads them -- and zeroed by `region_init`
with the rest of the signal area. Only arena 0's pair is ever used: the fused
inter-node kernel spans arenas but is one grid. They are a fixed rendezvous,
not generation-tagged, so two collectives of one communicator must never be
on the device at once. mojoccl (not NCCL, which allows groups across
streams) enforces that itself: a collective issued on a stream other than
the communicator's previous one is made to wait, through an event, for the
previous collective (`_order_before` / `_order_after`, mojoccl.mojo), so
every collective of a communicator runs in one total order whatever streams
the caller uses. The process group issues everything on one comm stream
anyway."""

comptime _POISON_OFFSET = 448
"""Device word one block of a persistent kernel raises to tell the others to
give up (internode_fused.mojo). Device memory, unlike the status page's abort
word: every block reads it once per chunk, and that read has to be an L2 hit
rather than a PCIe round trip."""

comptime _SIGNAL_BYTES = 128 * 1024
"""Signal-area size: 4 KiB header + MAX_BLOCKS*MAX_WORLD*8 B of flags = 68 KiB,
rounded to 128 KiB. Two orders of magnitude below MAX's 24.75 MiB `Signal`."""

comptime PHASES_PER_GEN = 8
"""Flag values are `generation * PHASES_PER_GEN + phase`; this bounds the
number of syncs one collective call may perform (three, for the two-shot
allreduce). Small on purpose: the flag is a UInt64 and the generation counter
never resets, so headroom costs nothing, but a tight bound documents the
contract."""

comptime DEFAULT_TIMEOUT_NS = 60_000_000_000
"""Default spin-loop deadline (60 s), measured with this GPU's own timer.

`MOJOCCL_IB_TIMEOUT_S` overrides it (`spin_timeout_ns`): one variable for
"how long a rank waits for a peer that stopped answering", whether the peer
is on this node's NVLink or on the other end of the fabric."""

comptime _SPIN_CHECK = 4096
"""Spins between two reads of the (not free) global timer."""

# The tuning constants below carry a `-D` override so the benchmark harness
# can sweep them without editing this file; the defaults are what a plain
# build (and therefore production) uses.

comptime _UNROLL = get_defined_int["ccl_unroll", 4]()
"""16-byte vectors in flight per thread in the NVLink copy loops."""

comptime _ONESHOT_MAX_BYTES = get_defined_int["ccl_oneshot_max", 512 * 1024]()
"""At or below this an allreduce uses the one-shot path: (world-1)x the NVLink
bytes but one sync instead of two, which wins while the transfer is
latency-bound. Measured crossover on 8xH100/NVSwitch (us, one-shot vs
two-shot): 128 KiB 9.3/19.0, 256 KiB 12.3/19.3, 512 KiB 18.1/19.8,
1 MiB 30.0/20.5 -- so the crossover sits just above 512 KiB. The path is taken
only if 2*world message-sized slots also fit the region."""

comptime _AR_MAX_BLOCKS = get_defined_int[
    "ccl_ar_blocks", 128 if has_amd_gpu_accelerator() else 216
]()
"""Grid cap for allreduce, fitted on H100 (132 SMs) / NVSwitch; see the block
sweep in RESULTS.md. Not portable: re-fit it on another card. The AMD value
was swept on one 4x MI300A node (228 CUs) with the push/reduce/push-back
schedule and the barrier this file uses there, fp32, 4 ranks, us at
9 / 27 MiB: 64 -> 118 / 285, **128 -> 121 / 254**, 224 -> 144 / 258. The cost
of the grid here is the barrier's `buffer_wbl2` per thread, which is why the
best value moved down from 224 once the release fence went back to every
thread; 64 starves the 27 MiB transfer. Re-fit it on another card, and re-fit
it if the barrier changes."""

comptime _AR_BIG_BYTES = get_defined_int["ccl_ar_big_bytes", 64 * 1024 * 1024]()
"""Above this message size the allreduce grid drops to `_AR_BIG_BLOCKS`."""

comptime _AR_BIG_BLOCKS = get_defined_int[
    "ccl_ar_big_blocks", 912 if has_amd_gpu_accelerator() else 128
]()
"""Grid cap for large allreduces. A grid that fits in one wave of an H100's
132 SMs measured 8% faster at 512 MiB than 216 blocks (2825 vs 3065 us) and
the same at 168 MiB, because the barrier is per block index: with more blocks
than SMs the second wave runs the whole collective after the first, on fewer
SMs. Fitted on H100 (132 SMs); re-fit on another card.

MI300A wants the opposite, and not gently. Swept on one 4x MI300A node with
the push/reduce/push-back schedule, fp32, 4 ranks, us at 168 / 512 MiB:
64 -> 1987 / 6638, 96 -> 1393 / 5465, 128 -> 1886 / 5381, 160 -> 1302 / 4804,
224 -> 2224 / 6053, 456 -> 1262 / 5093, **912 -> 1196 / 3705**. 912 is four
waves of the 228 CUs and is the only value that is best at both sizes; the
response in between is not monotonic (224, one block per CU, is the worst
point measured) so do not interpolate -- re-sweep."""

comptime _COPY_MAX_BLOCKS = get_defined_int["ccl_copy_blocks", 432]()
"""Grid cap for the pure-copy collectives (broadcast / allgather)."""

# Error codes written to the region's error word (`error_offset()`).
comptime ERR_ALLREDUCE_SYNC = 1
comptime ERR_BROADCAST_SYNC = 2
comptime ERR_ALLGATHER_SYNC = 3
comptime ERR_RS_STAGE_SYNC = 4
comptime ERR_AG_FINISH_SYNC = 5
# 6 is nvls_kernels.mojo's ERR_NVLS_SYNC; 7 and 8 are free.
comptime ERR_PROXY_WAIT = 9
"""`internode_kernels.mojo`'s wait for the inter-node progress thread. The
value is what that kernel has always written, so old logs still decode."""

comptime ERR_FUSED_GRID = 10
"""A grid barrier inside the fused inter-node allreduce (internode_fused.mojo)
gave up: this rank's own blocks stopped arriving, which only happens because
another spin in the same kernel already failed and returned."""

comptime ERR_HOST_LAUNCH = 11
"""The host failed to launch the fused inter-node allreduce after reserving
its exchange counters (mojoccl.mojo `_do_allreduce_fused`): the peers will
wait for flags and exchanges that are never coming, so the communicator is
failed from the host through the same status page a device deadline uses."""


# ===-------------------------------------------------------------------=== #
# Region geometry (host + device agree; every rank computes the same numbers)
# ===-------------------------------------------------------------------=== #


def signal_bytes() -> Int:
    """Bytes of signal/flag area at the base of every rank's region."""
    comptime assert (
        _AR_MAX_BLOCKS <= MAX_BLOCKS
        and _AR_BIG_BLOCKS <= MAX_BLOCKS
        and _COPY_MAX_BLOCKS <= MAX_BLOCKS
    ), "grid caps must fit the flag matrix"
    comptime assert BLOCK >= MAX_WORLD, "the sync needs one thread per peer"
    comptime assert (
        _FLAG_BYTE_OFFSET + MAX_BLOCKS * MAX_WORLD * 8 <= _SIGNAL_BYTES
    ), "the flag matrix must fit the signal area"
    return _SIGNAL_BYTES


def poison_offset() -> Int:
    """Byte offset, inside the signal area, of `_POISON_OFFSET`'s word."""
    comptime assert (
        _POISON_OFFSET + 8 <= _FLAG_BYTE_OFFSET
    ), "the header words must stay inside the first page"
    return _POISON_OFFSET


def error_offset() -> Int:
    """Byte offset, inside the signal area, of the UInt64 error word.

    Zero means "no error". A nonzero value is `code * 1_000_000 + phase`, with
    `code` one of the `ERR_*` constants above; it means some block gave up
    waiting for a peer and the collective's result is undefined from that
    generation on.
    """
    return 0


def status_ptr_offset() -> Int:
    """Byte offset, inside the signal area, of the status-page pointer slot."""
    comptime assert (
        _STATUS_PTR_OFFSET + 8 <= _FLAG_BYTE_OFFSET
    ), "the status pointer must fit the header page"
    return _STATUS_PTR_OFFSET


# ===-------------------------------------------------------------------=== #
# The communicator's status page: pinned host memory, mapped on the device
# ===-------------------------------------------------------------------=== #
#
# One page per communicator, allocated by mojoccl.mojo and published in every
# arena header (`install_status_page`). Two cache lines, and the split is the
# point: line 0 is the abort word the HOST writes and every device spin reads,
# line 1 is the fault record the DEVICE writes and the host reads.
#
# The fault record is what makes a device deadline loud. Before it, the only
# trace a timed-out barrier left was the arena's error word -- device memory,
# which the host can only read by launching a copy kernel and synchronizing
# the stream (`_read_error_word`), i.e. by blocking behind the very kernels it
# wants to report. Nobody could afford that per collective, so nobody read it,
# so a rank that gave up after 60 s went on to run every later kernel of the
# run and the collective completed with whatever the arena happened to hold.
# Pinned host memory a kernel stores into needs no copy and no synchronize:
# every collective entry point can test one word for a few nanoseconds, and
# `proxy_request` / `proxy_wait` can test it on the device for the price of a
# load in a kernel that already touches this page.

comptime STATUS_PAGE_BYTES = 128
"""Two cache lines: the abort word must stay on a line of its own, or raising
it would invalidate the line a spin is reading the fault record from."""

comptime STATUS_ABORT_WORD = 0
"""Word index of the abort word. `ncclCommAbort` stores 1 into it."""

comptime STATUS_FAULT_WORD = 8
"""Word index of the first word of the fault record (second cache line)."""

comptime STATUS_HOST_FAULT_WORD = 15
"""Host-only record: bit 63 = device fault already observed, bits 32..62 =
code, bits 0..31 = exchange. The host cannot join the device arena's claim,
so it never writes the device record. `_report_fault` keeps the first fully
published fault observed when the host latches this word."""

# Fault record layout, as word offsets from `STATUS_FAULT_WORD`. `FAULT_CODE`
# is written LAST, with a release store, so a nonzero code also means the six
# detail words are final; it is the "has this communicator failed" predicate
# the host and the inter-node kernels test. There is no rank field: the page
# belongs to one process, and that process knows its own rank.
comptime FAULT_CODE = 0
"""One of the `ERR_*` codes -- which collective, i.e. which kernel."""
comptime FAULT_PHASE = 1
"""Which barrier inside that collective (`_flag_target`'s phase)."""
comptime FAULT_BLOCK = 2
"""The block that gave up. Barriers here are matched by block index."""
comptime FAULT_PEER = 3
"""The peer whose flag never arrived, or `FAULT_NO_PEER`."""
comptime FAULT_SEEN = 4
"""The value that peer's flag actually held when the deadline fired."""
comptime FAULT_TARGET = 5
"""The value it was waited for: `generation * PHASES_PER_GEN + phase`."""
comptime FAULT_ARENA = 6
"""Base ADDRESS of the arena, which the host turns into an arena index (it is
the only party that knows the stride)."""

comptime FAULT_NO_PEER = UInt64(0xFFFF_FFFF_FFFF_FFFF)
"""`FAULT_PEER` for a wait that is not on a peer's flag -- the inter-node
exchange wait, whose counterpart is this rank's own progress thread."""


@always_inline
def _align_up(x: Int, a: Int) -> Int:
    return (x + a - 1) // a * a


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
    var raw = getenv("MOJOCCL_IB_TIMEOUT_S", String(""))
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
def _flag_target(generation: Int, phase: Int) -> UInt64:
    """The flag value that marks `phase` of `generation`.

    Strictly increasing in (generation, phase), never reset, so waiters compare
    with `>=` and a peer running ahead is harmless.
    """
    return UInt64(generation) * UInt64(PHASES_PER_GEN) + UInt64(phase)


# ===-------------------------------------------------------------------=== #
# Device-side primitives
# ===-------------------------------------------------------------------=== #


@always_inline
def _flags(
    region: Pointer[UInt8, MutAnyOrigin],
) -> Pointer[UInt64, MutAnyOrigin]:
    """flags[block][writer_rank], row-major, MAX_WORLD columns."""
    return region.unsafe_offset(_FLAG_BYTE_OFFSET).unsafe_bitcast[UInt64]()


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
        Atomic[DType.uint64].load[ordering=Ordering.ACQUIRE](
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
        Atomic[DType.uint64].load[ordering=Ordering.ACQUIRE](
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
    return Atomic[DType.uint64].compare_exchange[
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
    Atomic[DType.uint64].store[ordering=Ordering.RELEASE](
        status_word(page, STATUS_FAULT_WORD + FAULT_CODE), UInt64(code)
    )
    # And raise the abort word: "this communicator has failed" is then ONE
    # word for every device reader, which matters on the inter-node release
    # path, where each load of this pinned page is a PCIe round trip per
    # exchange (see `internode._proxy_main`). Ordered after the code, so a
    # reader that sees the word raised finds a complete record.
    Atomic[DType.uint64].store[ordering=Ordering.RELEASE](
        status_word(page, STATUS_ABORT_WORD), UInt64(1)
    )


@always_inline
def _record_deadline(
    regions: InlineArray[Pointer[UInt8, MutAnyOrigin], MAX_WORLD],
    rank: Int,
    code: Int,
    target: UInt64,
    peer: UInt64,
    seen: UInt64,
    arena_off: Int = 0,
):
    """A barrier gave up: record it in the arena and latch it for the host.

    Called by the thread that hit the deadline, which is the only one that
    knows which peer it was waiting for and what that peer's flag held. The
    arena word is what it always was (`ncclCommGetAsyncError` reads it); the
    status page is what makes the next collective on this communicator fail
    loudly instead of returning garbage.
    """
    var region = regions[rank].unsafe_offset(arena_off)
    var phase = Int(target % UInt64(PHASES_PER_GEN))
    # `error_offset()` is 0: the error word is the first word of the region.
    if not latch_arena_error(region.unsafe_bitcast[UInt64](), code, phase):
        return
    publish_fault(
        status_page(region),
        code,
        phase,
        Int(block_idx.x),
        peer,
        seen,
        target,
        Int(region),
    )


@always_inline
def _sync(
    regions: InlineArray[Pointer[UInt8, MutAnyOrigin], MAX_WORLD],
    world: Int,
    rank: Int,
    code: Int,
    target: UInt64,
    t0: UInt64,
    timeout_ns: UInt64,
    arena_off: Int = 0,
) -> Bool:
    """Block-scoped barrier across the same block index on every rank.

    Thread `p` (p < world) publishes `target` into peer p's flags[bid][rank]
    -- after every payload write this block made into peer memory has landed,
    which is what the fence below is for -- then waits for peer p's flag in my
    own region to reach `target`.

    Blocks are matched by index: every collective in this file gives block b of
    every rank exactly the same grid-stride slice of the index space, so block b
    only ever consumes bytes block b of a peer produced.

    Returns False if any participating thread hit the deadline; the whole block
    learns that through shared memory so no thread is left inside a `barrier()`.

    `code` is the caller's `ERR_*` constant, and recording the failure is this
    function's job rather than the caller's: only the thread that gave up knows
    which peer it was waiting for and what that peer's flag held, and that is
    most of what a reader of the message needs. Every caller used to follow a
    False with the same two lines; folding them in is also how a new caller
    stops being able to forget them.
    """
    var failed = stack_allocation[
        1, DType.uint32, address_space=AddressSpace.SHARED
    ]()
    if thread_idx.x == 0:
        failed[unsafe_offset=0] = 0
    comptime if _AMD:
        # gfx942's `s_barrier` is emitted with `s_waitcnt lgkmcnt(0)` only, so
        # another wave's payload stores can still be in flight when one thread
        # publishes the flag; RCCL puts `vmcnt(0)` inside its block barrier for
        # exactly this reason (rccl:src/device/prims_simple.h:193-210). A
        # release fence in every thread is the portable spelling and lowers to
        # `s_waitcnt vmcnt(0)` + `buffer_wbl2 sc0 sc1`, i.e. the writeback that
        # makes this block's payload stores visible to the peer that is about
        # to be told they are there.
        #
        # Both cheaper spellings were tried and both are wrong on this box, in
        # the same way and only for small payloads:
        #   * no writeback at all (RCCL's `skip_fence` for cudaArch 940, which
        #     is sound for RCCL because its P2P buffers are uncached) --
        #     broke every broadcast;
        #   * the writeback moved after the barrier into the `world` threads
        #     that publish flags -- fixed the broadcast at 4 ranks but still
        #     failed a 1-element allreduce and a broadcast at 2 ranks.
        # Our region is `hipDeviceMallocUncached`, but the mapping a peer
        # writes *through* comes from `hipIpcOpenMemHandle` and does not carry
        # that memory type, so a few bytes can still be sitting in the writer's
        # cache. Megabyte payloads drain on their own, which is why only the
        # small collectives ever failed. NVIDIA needs nothing: `bar.sync` is a
        # CTA-scope fence and the release store below is cumulative over it,
        # which is what NCCL's postPeer relies on.
        fence[ordering=Ordering.RELEASE]()
    barrier()

    if Int(thread_idx.x) < world:
        var peer = Int(thread_idx.x)
        var bid = Int(block_idx.x)
        # The acquire stays an acquire *load*, per iteration. Spinning on a
        # relaxed load (still `sc0 sc1`, so it cannot read a stale flag) and
        # invalidating once after the wait looks exactly as strong, is worth
        # a lot -- it is the difference between 243 us and 465 us at 27 MiB
        # when the grid is 1024 blocks, because `buffer_inv sc0 sc1` throws
        # the payload out of L2 for every block still working -- and was
        # measured to leave a one-element allreduce at 2 ranks failing 2 runs
        # in 13, against 0 in 12 with this spelling. Neither sample proves
        # anything on its own (Fisher p ~ 0.5), but there is no argument for
        # why the cheap version is sound on this hardware, and two cheaper
        # release spellings already turned out unsound here in exactly this
        # way -- small payloads only. So: correctness, and the large messages
        # pay for it. See docs/mojo_collectives_kernel_results.md section 7
        # for the experiment that would settle it.
        Atomic[DType.uint64].store[ordering=Ordering.RELEASE](
            _flags(regions[peer].unsafe_offset(arena_off)).unsafe_offset(
                bid * MAX_WORLD + rank
            ),
            target,
        )
        var mine = _flags(regions[rank].unsafe_offset(arena_off)).unsafe_offset(
            bid * MAX_WORLD + peer
        )
        var spins = 0
        while (
            Atomic[DType.uint64].load[ordering=Ordering.ACQUIRE](mine) < target
        ):
            spins += 1
            if spins >= _SPIN_CHECK:
                spins = 0
                if _abort_raised(regions[rank].unsafe_offset(arena_off)):
                    # Abort is a request, not a failure: record it in the
                    # arena word the way this file always has, but do not
                    # latch a fault. `ncclCommAbort` already told the host
                    # what happened, and an aborted communicator has to keep
                    # answering NCCL_INVALID_USAGE rather than start
                    # answering NCCL_REMOTE_ERROR.
                    _ = latch_arena_error(
                        regions[rank]
                        .unsafe_offset(arena_off)
                        .unsafe_bitcast[UInt64](),
                        code,
                        Int(target % UInt64(PHASES_PER_GEN)),
                    )
                    failed[unsafe_offset=0] = 1
                    break
                # Same-GPU timer difference only; never compared across GPUs.
                if device_now_ns() - t0 > timeout_ns:
                    _record_deadline(
                        regions,
                        rank,
                        code,
                        target,
                        UInt64(peer),
                        Atomic[DType.uint64].load[ordering=Ordering.ACQUIRE](
                            mine
                        ),
                        arena_off,
                    )
                    failed[unsafe_offset=0] = 1
                    break
    barrier()
    return failed[unsafe_offset=0] == 0


@always_inline
def grid_barrier(
    region: Pointer[UInt8, MutAnyOrigin],
    poison: Pointer[UInt64, MutAnyOrigin],
    nblocks: Int,
    code: Int,
    t0: UInt64,
    timeout_ns: UInt64,
) -> Bool:
    """Rendezvous of every block of ONE grid on ONE GPU; True if it completed.

    What a kernel boundary used to provide. The split inter-node schedule got
    "every block of this rank has finished phase P" from launching P+1 as a
    separate kernel; the fused kernel (internode_fused.mojo) runs the whole
    pipeline in one launch and has to say it itself -- before it hands a shard
    to the NIC, before it reads what the NIC delivered, and at the end of each
    chunk, where it restores exactly the invariant the launch boundary had.

    THE CALLER MUST GUARANTEE THAT EVERY BLOCK IS RESIDENT. Blocks that have
    arrived spin, so a grid larger than the device can hold at once deadlocks:
    the launcher sizes the grid at one block per SM/CU for that reason.

    Sense reversal, not a target count: the last block to arrive resets the
    counter and bumps a release word every other block is waiting to see
    change. Nothing is carried across launches and nothing has to agree with
    the host, which a "wait for arrival number N" barrier would need -- and
    would hang on for ever if the host's arithmetic and the kernel's loop ever
    disagreed by one.

    `poison` is how a block that has already given up -- in a spin of its own,
    or on a peer that stopped answering -- releases the blocks waiting here
    instead of leaving each of them to burn its own full deadline. It is a
    device word, read only from the slow path of the spin.

    Ordering: each block's arrival is a release RMW after its payload writes,
    the last arriver's RMW acquires the release sequence of all of them, and
    its release store of the sense word is what the waiters acquire. So a
    block that leaves this barrier sees every write every block made before
    entering it -- including, for block 0, the shard the NIC is about to read.
    """
    var failed = stack_allocation[
        1, DType.uint32, address_space=AddressSpace.SHARED
    ]()
    if thread_idx.x == 0:
        failed[unsafe_offset=0] = 0
    comptime if _AMD:
        # Same reason as `_sync`: gfx942's `s_barrier` does not wait on
        # outstanding vector stores, so the block's payload writes need an
        # explicit release before one thread announces them.
        fence[ordering=Ordering.RELEASE]()
    barrier()

    if thread_idx.x == 0:
        var arrive = region.unsafe_offset(
            _GRIDBAR_ARRIVE_OFFSET
        ).unsafe_bitcast[UInt64]()
        var release = region.unsafe_offset(
            _GRIDBAR_RELEASE_OFFSET
        ).unsafe_bitcast[UInt64]()
        var seen = Atomic[DType.uint64].load[ordering=Ordering.RELAXED](release)
        var was = Atomic[DType.uint64].fetch_add[
            ordering=Ordering.ACQUIRE_RELEASE
        ](arrive, UInt64(1))
        if Int(was) == nblocks - 1:
            # Last in. Reset the counter first: no block can arrive at the
            # next barrier before it has seen the sense word below change,
            # and the release store orders this plain one ahead of it.
            Atomic[DType.uint64].store[ordering=Ordering.RELAXED](
                arrive, UInt64(0)
            )
            Atomic[DType.uint64].store[ordering=Ordering.RELEASE](
                release, seen + 1
            )
        else:
            var page = status_page(region)
            var spins = 0
            while (
                Atomic[DType.uint64].load[ordering=Ordering.ACQUIRE](release)
                == seen
            ):
                comptime if _AMD:
                    # On gfx942 the acquire load above is a `buffer_inv sc0
                    # sc1` -- an L1+L2 invalidate -- per iteration, and
                    # nblocks-1 threads spin here for a whole network round
                    # trip (internode_kernels.mojo has the measurement for
                    # one such thread). Sleep between polls; the ordering
                    # stays as it is.
                    llvm_intrinsic[
                        "llvm.amdgcn.s.sleep", NoneType, has_side_effect=True
                    ](Int32(2))
                spins += 1
                if spins >= _SPIN_CHECK:
                    spins = 0
                    if (
                        Atomic[DType.uint64].load[ordering=Ordering.ACQUIRE](
                            poison
                        )
                        != 0
                    ):
                        failed[unsafe_offset=0] = 1
                        break
                    if abort_raised(page):
                        _ = latch_arena_error(
                            region.unsafe_bitcast[UInt64](), code, 0
                        )
                        failed[unsafe_offset=0] = 1
                        break
                    if device_now_ns() - t0 > timeout_ns:
                        if latch_arena_error(
                            region.unsafe_bitcast[UInt64](), code, 0
                        ):
                            publish_fault(
                                page,
                                code,
                                0,
                                Int(block_idx.x),
                                FAULT_NO_PEER,
                                seen,
                                seen + 1,
                                Int(region),
                            )
                        failed[unsafe_offset=0] = 1
                        break
        # Read once, by one thread, for the whole block: a block that has
        # been released may still be one another block just poisoned, and
        # the answer has to be the same for every thread of this CTA or part
        # of it would leave while the rest reaches the next `barrier()`.
        if (
            failed[unsafe_offset=0] == 0
            and Atomic[DType.uint64].load[ordering=Ordering.ACQUIRE](poison)
            != 0
        ):
            failed[unsafe_offset=0] = 1
    barrier()
    return failed[unsafe_offset=0] == 0


@always_inline
def _peer_step(i: Int, world: Int) -> Int:
    """Peer offset for loop index `i` (1 <= i < world), rotated by block index.

    Every block still touches exactly the same bytes for every peer -- only
    the order of the peers differs -- so the block-matched sync invariant is
    untouched. What changes is link usage on a point-to-point topology such
    as the MI300A's xGMI mesh (one link per GPU pair): with the plain order
    the whole grid queues on one peer's link at a time and the other
    `world-2` links sit idle; rotated, the blocks spread over all of them at
    once. Behind a switch (NVSwitch) the order is irrelevant. Measured on
    4x MI300A: see docs/distributed.md, "Cluster notes (AMD MI300A)".
    """
    comptime if has_amd_gpu_accelerator():
        return 1 + (i - 1 + Int(block_idx.x)) % (world - 1)
    else:
        # Behind NVSwitch the order is irrelevant; keep the H100-measured
        # device code byte-identical.
        return i


@always_inline
def _peer_step0(i: Int, world: Int) -> Int:
    """`_peer_step` for loops that include the rank itself (0 <= i < world)."""
    comptime if has_amd_gpu_accelerator():
        return (i + Int(block_idx.x)) % world
    else:
        return i


@always_inline
def _gather_slot(writer: Int, owner: Int) -> Int:
    """Index of `writer`'s slot inside `owner`'s gather area (AMD phase 3').

    Compacted, exactly like the split allreduce's push slots: `owner` never
    writes its own slot (its own reduced shard goes straight to the user
    output), so `world-1` slots suffice and the area stays inside the arena
    even when the message is exactly `cap` bytes.
    """
    return writer if writer < owner else writer - 1


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


@always_inline
def _copy_vec[
    dtype: DType, W: Int, U: Int
](
    dst: Pointer[Scalar[dtype], MutAnyOrigin],
    src: Pointer[Scalar[dtype], MutAnyOrigin],
    nvec: Int,
    tid: Int,
    stride: Int,
):
    """Grid-stride copy of `nvec` 16-byte vectors, `U` of them in flight.

    Both pointers must be 16-byte aligned, which every address this file forms
    is (region sub-areas are multiples of 16 B from a page-aligned base; user
    pointers come from MAX's allocator; shard starts are multiples of W).
    """
    var v = tid
    var lim = nvec - (U - 1) * stride
    while v < lim:
        var tmp = InlineArray[SIMD[dtype, W], U](uninitialized=True)
        comptime for u in range(U):
            tmp[u] = src.unsafe_load[width=W, alignment=16](
                (v + u * stride) * W
            )
        comptime for u in range(U):
            dst.unsafe_store[width=W, alignment=16](
                (v + u * stride) * W, tmp[u]
            )
        v += U * stride
    while v < nvec:
        dst.unsafe_store[width=W, alignment=16](
            v * W, src.unsafe_load[width=W, alignment=16](v * W)
        )
        v += stride


@always_inline
def _copy_scalar_tail[
    dtype: DType
](
    dst: Pointer[Scalar[dtype], MutAnyOrigin],
    src: Pointer[Scalar[dtype], MutAnyOrigin],
    base: Int,
    count: Int,
    tid: Int,
    stride: Int,
):
    """The `numel % W` elements a 16-byte vector loop cannot cover."""
    for i in range(tid, count, stride):
        dst[unsafe_offset=base + i] = src[unsafe_offset=base + i]


@always_inline
def _copy_bytes[
    U: Int
](
    dst: Pointer[UInt8, MutAnyOrigin],
    src: Pointer[UInt8, MutAnyOrigin],
    nbytes: Int,
    tid: Int,
    stride: Int,
):
    """dtype-agnostic byte copy: 16-byte vectors when both sides allow it.

    Both paths walk the *same* 16-byte chunks in the same grid-stride order, so
    the chunk -> block mapping does not depend on which path a given call takes.
    That matters: for a byte collective the writer and the reader are different
    ranks looking at different pointer pairs (the root's `send` and a peer's
    `recv`), so they can disagree about alignment, and the per-block barrier
    only orders block b against block b. A fallback that walked single bytes
    would let block 3 read a chunk block 0 wrote.
    """
    var nvec = nbytes // 16
    if (Int(dst) | Int(src)) % 16 == 0:
        _copy_vec[DType.uint8, 16, U](dst, src, nvec, tid, stride)
    else:
        for v in range(tid, nvec, stride):
            comptime for j in range(16):
                dst[unsafe_offset=v * 16 + j] = src[unsafe_offset=v * 16 + j]
    _copy_scalar_tail(dst, src, nvec * 16, nbytes - nvec * 16, tid, stride)


@always_inline
def _copy_span[
    dtype: DType, W: Int, U: Int
](
    dst: Pointer[Scalar[dtype], MutAnyOrigin],
    src: Pointer[Scalar[dtype], MutAnyOrigin],
    count: Int,
    tid: Int,
    stride: Int,
):
    """`count` elements: the 16-byte vectors, then the `count % W` tail."""
    var vc = count // W
    _copy_vec[dtype, W, U](dst, src, vc, tid, stride)
    _copy_scalar_tail(dst, src, vc * W, count - vc * W, tid, stride)


@always_inline
def _copy_span_scaled[
    dtype: DType, W: Int, U: Int
](
    dst: Pointer[Scalar[dtype], MutAnyOrigin],
    src: Pointer[Scalar[dtype], MutAnyOrigin],
    count: Int,
    tid: Int,
    stride: Int,
    scale: Float32,
):
    """`_copy_span` times `scale`; integer dtypes ignore `scale` (as NCCL's
    ncclAvg does, and as the fused allreduce does). `scale == 1` takes the
    plain copy, so the all-gather half of a SUM allreduce costs no more than a
    peer copy."""
    comptime accum = DType.float32 if (
        dtype == DType.bfloat16 or dtype == DType.float16
    ) else dtype
    comptime if accum.is_floating_point():
        if scale != Float32(1.0):
            var sv = SIMD[accum, W](scale.cast[accum]())
            var vc = count // W
            var v = tid
            var lim = vc - (U - 1) * stride
            while v < lim:
                var tmp = InlineArray[SIMD[dtype, W], U](uninitialized=True)
                comptime for u in range(U):
                    tmp[u] = src.unsafe_load[width=W, alignment=16](
                        (v + u * stride) * W
                    )
                comptime for u in range(U):
                    dst.unsafe_store[width=W, alignment=16](
                        (v + u * stride) * W,
                        (tmp[u].cast[accum]() * sv).cast[dtype](),
                    )
                v += U * stride
            while v < vc:
                dst.unsafe_store[width=W, alignment=16](
                    v * W,
                    (
                        src.unsafe_load[width=W, alignment=16](v * W).cast[
                            accum
                        ]()
                        * sv
                    ).cast[dtype](),
                )
                v += stride
            for i in range(tid, count - vc * W, stride):
                var k = vc * W + i
                dst[unsafe_offset=k] = (
                    src[unsafe_offset=k].cast[accum]() * scale.cast[accum]()
                ).cast[dtype]()
            return
    _copy_span[dtype, W, U](dst, src, count, tid, stride)


# ===-------------------------------------------------------------------=== #
# Shard partition. Every rank derives the same table from (numel, world).
# ===-------------------------------------------------------------------=== #


@always_inline
def _vstart(s: Int, q: Int, rem: Int) -> Int:
    """First 16-byte vector of rank `s`'s shard."""
    return s * q + min(s, rem)


@always_inline
def _vcount(s: Int, q: Int, rem: Int) -> Int:
    """16-byte vectors in rank `s`'s shard (the `numel % W` scalar tail, if
    any, belongs to the last rank and is counted separately)."""
    return q + (1 if s < rem else 0)


# ===-------------------------------------------------------------------=== #
# Split shard partition -- the one the hierarchical (multi-node) path uses
# ===-------------------------------------------------------------------=== #
#
# `shard_range` is a *different* partition from `_vstart`/`_vcount` above and
# deliberately so. The fused allreduce spreads the `nvec % world` leftover
# vectors one each over the first ranks, which balances best but makes every
# shard's offset depend on the whole remainder table. The split path hands its
# shard offset to a foreign library (NCCL/RCCL, which allreduces shard `r`
# across nodes among the ranks whose local index is `r`), so the rule has to be
# something a caller can state in one line and every rank must derive the same
# answer from (numel, world, elem_bytes) alone:
#
#     equal shards of `per` elements, `per` rounded up to the 16-byte vector
#     width, the last non-empty shard short, the ranks past the end empty.
#
# Imbalance is at most one vector per rank against the balanced split -- below
# the noise at every size that reaches the split path -- and in exchange every
# shard starts 16-byte aligned, which the vector loops and the vendor library
# both want.


@always_inline
def _shard_per(numel: Int, world: Int, W: Int) -> Int:
    """Elements per shard, rounded up to the 16-byte vector width `W`."""
    if numel <= 0 or world <= 0:
        return 0
    var c = (numel + world - 1) // world
    return (c + W - 1) // W * W


@always_inline
def _shard_off(numel: Int, per: Int, s: Int) -> Int:
    """First element of rank `s`'s shard (== numel once the shards run out)."""
    var o = s * per
    return o if o < numel else numel


@always_inline
def _shard_cnt(numel: Int, per: Int, s: Int) -> Int:
    """Elements in rank `s`'s shard; 0 for the ranks past the end."""
    var rest = numel - _shard_off(numel, per, s)
    return per if per < rest else rest


def shard_range(
    numel: Int, world: Int, rank: Int, elem_bytes: Int
) -> Tuple[Int, Int]:
    """`(offset_elems, count_elems)` of rank `rank`'s shard of a `numel` buffer.

    Pure host arithmetic, no device state, identical on every rank -- the ABI
    layer calls it to address the shard `reduce_scatter_stage` leaves in
    stage_out and to size the inter-node collective it runs on it.

    Shards are contiguous and cover `[0, numel)` in rank order; every offset is
    16-byte aligned for this dtype (so the shard pointer is as well, given a
    16-byte aligned buffer); the last non-empty shard may be shorter and the
    ranks past the end get `(numel, 0)`. `elem_bytes` must divide 16 (all
    supported dtypes are 1, 2, 4 or 8 bytes wide).
    """
    if elem_bytes <= 0 or elem_bytes > 16 or 16 % elem_bytes != 0:
        return Tuple(0, 0)
    var per = _shard_per(numel, world, 16 // elem_bytes)
    if per == 0 or rank < 0 or rank >= world:
        return Tuple(0, 0)
    return Tuple(_shard_off(numel, per, rank), _shard_cnt(numel, per, rank))


# ===-------------------------------------------------------------------=== #
# Allreduce -- two-shot (push / reduce / pull)
# ===-------------------------------------------------------------------=== #
#
# Buffer-reuse invariant. Every collective in this file opens with a start
# barrier, so the whole arena obeys one rule:
#
#     no rank writes an arena byte for generation g until every rank has
#     finished reading arena bytes for generation g-1
#
# (a rank reaches generation g's start barrier only after its own generation
# g-1 work has retired, and nobody passes that barrier until all have arrived).
# That is what lets collectives of different kinds and sizes share the arena
# and interleave freely -- which they do: DDP issues a 4-byte one-shot
# allreduce and an 8-byte allgather in between 27 MiB two-shot allreduces, and
# their staging layouts overlap. Dropping the start barrier saves ~2.5 us and
# is only sound for a run of identically shaped collectives; it was measured
# (166.4 us vs 169 us at 27 MiB) and rejected as a correctness trap.
#
# Within a call the two data syncs order the three phases:
#     start(g) < push(g) < A(g) < reduce(g) < B(g) < pull(g) < start(g+1)
# (on AMD: reduce-and-push-back instead of reduce, local gather instead of
# pull -- same three phases, same two data syncs) and a sync is a full N-way
# rendezvous of matching block indices. In-place (in_ptr == out_ptr) is safe
# because the phases are block-matched: block b writes exactly the elements
# block b read.


@__llvm_metadata(
    MAX_THREADS_PER_BLOCK_METADATA=StaticTuple[Int32, 1](Int32(BLOCK))
)
@__name(t"ccl_allreduce_push_reduce_pull_{dtype}_w{NW}")
def _ar_twoshot_kernel[
    dtype: DType, W: Int, U: Int, NW: Int
](
    regions: InlineArray[Pointer[UInt8, MutAnyOrigin], MAX_WORLD],
    in_ptr: Pointer[Scalar[dtype], MutAnyOrigin],
    out_ptr: Pointer[Scalar[dtype], MutAnyOrigin],
    numel: Int64,
    slot_stride_b: Int64,
    push_off_b: Int64,
    shard_off_b: Int64,
    world_i: Int32,
    rank_i: Int32,
    flag_base: UInt64,
    scale: Float32,
    timeout_ns: UInt64,
):
    comptime accum = DType.float32 if (
        dtype == DType.bfloat16 or dtype == DType.float16
    ) else dtype
    var t0 = device_now_ns()
    var world = NW if NW > 0 else Int(world_i)
    var rank = Int(rank_i)
    var tid = Int(global_idx.x)
    var stride = Int(grid_dim.x) * BLOCK
    var n = Int(numel)
    var nvec = n // W
    var q = nvec // world
    var rem = nvec % world
    var tail = n - nvec * W
    var slot_stride = Int(slot_stride_b)
    var push_off = Int(push_off_b)
    var shard_off = Int(shard_off_b)

    # --- phase 0: start barrier -- nobody writes the arena for generation g
    # until every rank has finished reading it for generation g-1 ------------
    if not _sync(
        regions, world, rank, ERR_ALLREDUCE_SYNC, flag_base, t0, timeout_ns
    ):
        return

    # --- phase 1: push shard s of my input into peer s's slot `rank` --------
    for i in range(1, world):
        var s = rank + _peer_step(i, world)
        if s >= world:
            s -= world
        var src = in_ptr.unsafe_offset(_vstart(s, q, rem) * W)
        var vc = _vcount(s, q, rem)
        var dst = (
            regions[s]
            .unsafe_offset(push_off + slot_stride * rank)
            .unsafe_bitcast[Scalar[dtype]]()
        )
        _copy_vec[dtype, W, U](dst, src, vc, tid, stride)
        if tail > 0 and s == world - 1:
            _copy_scalar_tail(dst, src, vc * W, tail, tid, stride)

    if not _sync(
        regions, world, rank, ERR_ALLREDUCE_SYNC, flag_base + 1, t0, timeout_ns
    ):
        return

    # --- phase 2: reduce my shard, to the arena and to the user output ------
    var my_vs = _vstart(rank, q, rem)
    var my_vc = _vcount(rank, q, rem)
    var my_tail = tail if rank == world - 1 else 0
    var uin = in_ptr.unsafe_offset(my_vs * W)
    var uout = out_ptr.unsafe_offset(my_vs * W)
    # NVIDIA publishes the reduced shard in its own region for the peers to
    # pull.  AMD pushes it into theirs instead (phase 2b below), so there is
    # nothing to publish locally; `shard` then aliases `uout` and the store to
    # it is elided at compile time.
    var shard = (
        uout if _AMD else regions[rank]
        .unsafe_offset(shard_off)
        .unsafe_bitcast[Scalar[dtype]]()
    )

    # Slot pointers are formed by arithmetic inside the unrolled loop, never
    # held in an array: a stack array of `world` pointers is demoted to local
    # memory (MOCO-1431) and turns every payload load into a generic-address
    # `ld.v4.b32` plus an `ld.local.b64` of the pointer itself.
    var slots = regions[rank].unsafe_offset(push_off)

    # AMD only: the peers' gather slots this rank pushes its reduced shard
    # into, hoisted out of the element loop.  A `comptime for` writes and
    # reads this array at constant indices only, so SROA keeps the `world-1`
    # pointers in registers -- the MOCO-1431 demotion above bites when the
    # index is a runtime value, which is why the reduce's own source pointers
    # are still formed by arithmetic.  `NW == 0` (world 3, 5, 6, 7) has no
    # comptime bound and falls back to a second pass over the shard.
    var gout = InlineArray[Pointer[Scalar[dtype], MutAnyOrigin], MAX_WORLD](
        uninitialized=True
    )
    comptime if _AMD and NW > 0:
        comptime for j in range(1, NW):
            var pj = rank + _peer_step(j, NW)
            if pj >= NW:
                pj -= NW
            gout[j] = (
                regions[pj]
                .unsafe_offset(shard_off + slot_stride * _gather_slot(rank, pj))
                .unsafe_bitcast[Scalar[dtype]]()
            )

    for v in range(tid, my_vc, stride):
        var acc = uin.unsafe_load[width=W, alignment=16](v * W).cast[accum]()
        comptime if NW > 0:
            comptime for j in range(1, NW):
                var p = rank + j
                if p >= NW:
                    p -= NW
                acc += (
                    slots.unsafe_offset(slot_stride * p)
                    .unsafe_bitcast[Scalar[dtype]]()
                    .unsafe_load[width=W, alignment=16](v * W)
                    .cast[accum]()
                )
        else:
            for j in range(1, world):
                var p = rank + j
                if p >= world:
                    p -= world
                acc += (
                    slots.unsafe_offset(slot_stride * p)
                    .unsafe_bitcast[Scalar[dtype]]()
                    .unsafe_load[width=W, alignment=16](v * W)
                    .cast[accum]()
                )
        comptime if accum.is_floating_point():
            acc *= SIMD[accum, W](scale.cast[accum]())
        var res = acc.cast[dtype]()
        comptime if not _AMD:
            shard.unsafe_store[width=W, alignment=16](v * W, res)
        uout.unsafe_store[width=W, alignment=16](v * W, res)
        # One reduce, `world` stores: my output slice and every peer's gather
        # slot.  This is NCCL's MULTIDSTS shape (rccl:src/device/
        # common_kernel.h reduceCopyPacks stores to all destinations from one
        # accumulator) and it keeps the local reduce traffic inside the wire
        # transfer instead of adding a pass in front of it.
        comptime if _AMD and NW > 0:
            comptime for j in range(1, NW):
                gout[j].unsafe_store[width=W, alignment=16](v * W, res)

    for i in range(tid, my_tail, stride):
        var k = my_vc * W + i
        var a = uin[unsafe_offset=k].cast[accum]()
        for j in range(1, world):
            var p = rank + j
            if p >= world:
                p -= world
            a += (
                slots.unsafe_offset(slot_stride * p)
                .unsafe_bitcast[Scalar[dtype]]()[unsafe_offset=k]
                .cast[accum]()
            )
        comptime if accum.is_floating_point():
            a *= scale.cast[accum]()
        comptime if not _AMD:
            shard[unsafe_offset=k] = a.cast[dtype]()
        uout[unsafe_offset=k] = a.cast[dtype]()
        comptime if _AMD and NW > 0:
            comptime for j in range(1, NW):
                gout[j][unsafe_offset=k] = a.cast[dtype]()

    comptime if _AMD and NW == 0:
        # --- phase 2b (AMD, generic world): push my reduced shard into every
        # peer's gather slot in a second pass.  Only worlds 3, 5, 6 and 7 come
        # here; 2, 4 and 8 fuse the stores into the reduce above.  Thread
        # `tid` reads back only the elements thread `tid` just wrote (both
        # loops walk `{tid, tid+stride, ...}`), so no fence is involved.
        for i in range(1, world):
            var p = rank + _peer_step(i, world)
            if p >= world:
                p -= world
            var dst = (
                regions[p]
                .unsafe_offset(shard_off + slot_stride * _gather_slot(rank, p))
                .unsafe_bitcast[Scalar[dtype]]()
            )
            _copy_vec[dtype, W, U](dst, uout, my_vc, tid, stride)
            if my_tail > 0:
                _copy_scalar_tail(dst, uout, my_vc * W, my_tail, tid, stride)

    if not _sync(
        regions, world, rank, ERR_ALLREDUCE_SYNC, flag_base + 2, t0, timeout_ns
    ):
        return

    # --- phase 3: the peers' reduced shards into the user output ------------
    # NVIDIA reads them across the fabric; AMD reads them out of its own
    # region, where phase 2b's pushes left them (module header, "Link
    # direction").
    for i in range(1, world):
        var p = rank + _peer_step(i, world)
        if p >= world:
            p -= world
        var vs = _vstart(p, q, rem)
        var vc = _vcount(p, q, rem)
        var src: Pointer[Scalar[dtype], MutAnyOrigin]
        comptime if _AMD:
            src = (
                regions[rank]
                .unsafe_offset(shard_off + slot_stride * _gather_slot(p, rank))
                .unsafe_bitcast[Scalar[dtype]]()
            )
        else:
            src = (
                regions[p]
                .unsafe_offset(shard_off)
                .unsafe_bitcast[Scalar[dtype]]()
            )
        var dst = out_ptr.unsafe_offset(vs * W)
        _copy_vec[dtype, W, U](dst, src, vc, tid, stride)
        if tail > 0 and p == world - 1:
            _copy_scalar_tail(dst, src, vc * W, tail, tid, stride)


# ===-------------------------------------------------------------------=== #
# Allreduce -- one-shot (push whole input / reduce), for small messages
# ===-------------------------------------------------------------------=== #
#
# (world-1)x the NVLink bytes of the two-shot path but one data sync instead of
# two, which wins while latency dominates: measured 9.3 us against 19.0 us at
# 128 KiB, with the crossover just above 512 KiB.


@__llvm_metadata(
    MAX_THREADS_PER_BLOCK_METADATA=StaticTuple[Int32, 1](Int32(BLOCK))
)
@__name(t"ccl_allreduce_oneshot_{dtype}_w{NW}")
def _ar_oneshot_kernel[
    dtype: DType, W: Int, U: Int, NW: Int
](
    regions: InlineArray[Pointer[UInt8, MutAnyOrigin], MAX_WORLD],
    in_ptr: Pointer[Scalar[dtype], MutAnyOrigin],
    out_ptr: Pointer[Scalar[dtype], MutAnyOrigin],
    numel: Int64,
    slot_stride_b: Int64,
    push_off_b: Int64,
    world_i: Int32,
    rank_i: Int32,
    flag_base: UInt64,
    scale: Float32,
    timeout_ns: UInt64,
):
    comptime accum = DType.float32 if (
        dtype == DType.bfloat16 or dtype == DType.float16
    ) else dtype
    var t0 = device_now_ns()
    var world = NW if NW > 0 else Int(world_i)
    var rank = Int(rank_i)
    var tid = Int(global_idx.x)
    var stride = Int(grid_dim.x) * BLOCK
    var n = Int(numel)
    var nvec = n // W
    var tail = n - nvec * W
    var slot_stride = Int(slot_stride_b)
    var push_off = Int(push_off_b)

    if not _sync(
        regions, world, rank, ERR_ALLREDUCE_SYNC, flag_base, t0, timeout_ns
    ):
        return

    for i in range(1, world):
        var s = rank + _peer_step(i, world)
        if s >= world:
            s -= world
        var dst = (
            regions[s]
            .unsafe_offset(push_off + slot_stride * rank)
            .unsafe_bitcast[Scalar[dtype]]()
        )
        _copy_vec[dtype, W, U](dst, in_ptr, nvec, tid, stride)
        if tail > 0:
            _copy_scalar_tail(dst, in_ptr, nvec * W, tail, tid, stride)

    if not _sync(
        regions, world, rank, ERR_ALLREDUCE_SYNC, flag_base + 1, t0, timeout_ns
    ):
        return

    var slots = regions[rank].unsafe_offset(push_off)

    for v in range(tid, nvec, stride):
        var acc = in_ptr.unsafe_load[width=W, alignment=16](v * W).cast[accum]()
        comptime if NW > 0:
            comptime for j in range(1, NW):
                var p = rank + j
                if p >= NW:
                    p -= NW
                acc += (
                    slots.unsafe_offset(slot_stride * p)
                    .unsafe_bitcast[Scalar[dtype]]()
                    .unsafe_load[width=W, alignment=16](v * W)
                    .cast[accum]()
                )
        else:
            for j in range(1, world):
                var p = rank + j
                if p >= world:
                    p -= world
                acc += (
                    slots.unsafe_offset(slot_stride * p)
                    .unsafe_bitcast[Scalar[dtype]]()
                    .unsafe_load[width=W, alignment=16](v * W)
                    .cast[accum]()
                )
        comptime if accum.is_floating_point():
            acc *= SIMD[accum, W](scale.cast[accum]())
        out_ptr.unsafe_store[width=W, alignment=16](v * W, acc.cast[dtype]())

    for i in range(tid, tail, stride):
        var k = nvec * W + i
        var a = in_ptr[unsafe_offset=k].cast[accum]()
        for j in range(1, world):
            var p = rank + j
            if p >= world:
                p -= world
            a += (
                slots.unsafe_offset(slot_stride * p)
                .unsafe_bitcast[Scalar[dtype]]()[unsafe_offset=k]
                .cast[accum]()
            )
        comptime if accum.is_floating_point():
            a *= scale.cast[accum]()
        out_ptr[unsafe_offset=k] = a.cast[dtype]()


# ===-------------------------------------------------------------------=== #
# Split allreduce -- reduce-scatter stage, then all-gather finish
# ===-------------------------------------------------------------------=== #
#
# The hierarchical (multi-node) allreduce is
#
#   reduce_scatter_stage(g)   my shard, summed over the node, into my stage_out
#   <inter-node step>         NCCL/RCCL allreduces that shard across nodes,
#                             in place, on the same stream, among the ranks
#                             that share my local index
#   allgather_finish(g+1)     pull every rank's now-global shard into my output
#
# Layout. Shards follow `shard_range` (equal, 16-byte aligned, last one short).
#   push slots  stage_in base, `world-1` slots of `per * elem_bytes` bytes.
#               Slot indices are *compacted*: writer r stores into destination
#               rank s's slot `r if r < s else r-1`, because s never writes its
#               own slot (it reads its own contribution from user memory). The
#               compaction is not cosmetic: `world` uncompacted slots can be
#               up to 16*world bytes larger than stage_in when
#               numel*elem_bytes == cap (the per-shard round-up to the vector
#               width, times world), which would spill onto rank 0's shard in
#               stage_out. `world-1` of them never can -- checked exhaustively
#               for every dtype width, world and cap.
#   shard       stage_out + offset(rank) * elem_bytes, i.e. stage_out is an
#               image of the whole buffer of which only my shard is live. That
#               placement is what makes the ABI layer's job one line -- the
#               inter-node collective gets `stage_out + offset*elem_bytes` and
#               `count` -- and it makes the pull address the same on both
#               sides. It always fits: offset+count <= numel and
#               numel*elem_bytes <= cap_bytes.
#
# Ordering. Three things happen between the two kernels that the fused
# allreduce never has to think about, and all three are covered without a new
# protocol:
#
#  1. A foreign library writes my stage_out shard in place. Nobody may read it
#     before that write lands. `allgather_finish`'s start barrier is the fence:
#     a rank publishes its generation g+1 flags only from inside that kernel,
#     which its stream starts only after its inter-node op has completed, so
#     seeing peer p's flag implies p's shard is final.
#  2. Peer p's shard was written by a *previous kernel* (p's reduce_scatter_
#     stage), not by the thread that publishes the flag. Stream order makes
#     that kernel's writes happen-before the flag's release store, and the
#     release/acquire pair is system-scoped and cumulative, so they are visible
#     to the acquiring reader. (On gfx942 `_sync`'s AMD-only release fence adds
#     the `buffer_wbl2 sc0 sc1` writeback that the workgroup barrier omits.)
#     Note the consequence: block-index matching, which the fused allreduce
#     relies on *within* a kernel, is not needed across this boundary -- the
#     two kernels may be launched with different grids, and they are.
#  3. Arena reuse after the pulls. Nothing extra is needed: the next collective
#     of any kind opens with a start barrier, and a rank reaches it only after
#     its own `allgather_finish` retired, so no generation g+2 write can race a
#     generation g+1 pull. That is the same one rule as everywhere else in this
#     file -- the split pair just spends two generations instead of one.
#
# Each call consumes one generation. `reduce_scatter_stage` uses flag phases
# 0 and 1, `allgather_finish` phase 0.


@always_inline
def _rs_stage_body[
    dtype: DType, W: Int, U: Int, NW: Int
](
    regions: InlineArray[Pointer[UInt8, MutAnyOrigin], MAX_WORLD],
    arena_off: Int,
    in_ptr: Pointer[Scalar[dtype], MutAnyOrigin],
    n: Int,
    per: Int,
    slot_stride: Int,
    push_off: Int,
    out_off: Int,
    world: Int,
    rank: Int,
    tid: Int,
    stride: Int,
    flag_base: UInt64,
    scale: Float32,
    t0: UInt64,
    timeout_ns: UInt64,
) -> Bool:
    """The reduce-scatter half, as a body two kernels share.

    `_rs_stage_kernel` is one launch of it; the pipelined inter-node kernel
    (internode_fused.mojo) runs it once per chunk with `arena_off` naming the
    arena and `tid`/`stride` the caller's grid-stride slice. Everything the
    launcher used to compute from `Int64` arguments arrives here as `Int`.

    False means a barrier gave up (deadline or abort); it has already
    recorded why, and the caller must return without touching the arena.
    """
    comptime accum = DType.float32 if (
        dtype == DType.bfloat16 or dtype == DType.float16
    ) else dtype
    comptime esize = size_of[dtype]()

    # --- phase 0: start barrier (the arena-reuse invariant) -----------------
    if not _sync(
        regions,
        world,
        rank,
        ERR_RS_STAGE_SYNC,
        flag_base,
        t0,
        timeout_ns,
        arena_off,
    ):
        return False

    # --- phase 1: push shard s of my input into peer s's slot for me --------
    for i in range(1, world):
        var s = rank + _peer_step(i, world)
        if s >= world:
            s -= world
        var off = _shard_off(n, per, s)
        var cnt = _shard_cnt(n, per, s)
        if cnt <= 0:
            continue
        var dst = (
            regions[s]
            .unsafe_offset(
                arena_off
                + push_off
                + slot_stride * (rank if rank < s else rank - 1)
            )
            .unsafe_bitcast[Scalar[dtype]]()
        )
        _copy_span[dtype, W, U](
            dst, in_ptr.unsafe_offset(off), cnt, tid, stride
        )

    if not _sync(
        regions,
        world,
        rank,
        ERR_RS_STAGE_SYNC,
        flag_base + 1,
        t0,
        timeout_ns,
        arena_off,
    ):
        return False

    # --- phase 2: sum the `world` contributions to my shard into stage_out --
    # Times `scale`, applied in the fp32 accumulator BEFORE the store narrows
    # to the wire dtype: the inter-node step sums these node partials again in
    # that dtype, and an unscaled fp16 sum of 8 x 10000 is already inf. This
    # is NCCL's PreMulSum for AVG (nccl:src/enqueue/enqueue.cc:2517), and for
    # a power-of-two communicator x/world is exact, so it rounds no more than
    # the plain sum would. `allgather_finish` then runs with scale 1.
    var my_off = _shard_off(n, per, rank)
    var my_cnt = _shard_cnt(n, per, rank)
    if my_cnt <= 0:
        return True
    var uin = in_ptr.unsafe_offset(my_off)
    var shard = (
        regions[rank]
        .unsafe_offset(arena_off + out_off + my_off * esize)
        .unsafe_bitcast[Scalar[dtype]]()
    )
    # Slot pointers are formed arithmetically, never held in a stack array:
    # such an array is demoted to local memory (MOCO-1431) and every payload
    # load becomes a generic-address `ld.v4.b32` plus an `ld.local.b64`.
    var slots = regions[rank].unsafe_offset(arena_off + push_off)
    var my_vc = my_cnt // W

    for v in range(tid, my_vc, stride):
        var acc = uin.unsafe_load[width=W, alignment=16](v * W).cast[accum]()
        comptime if NW > 0:
            comptime for j in range(1, NW):
                var p = rank + j
                if p >= NW:
                    p -= NW
                acc += (
                    slots.unsafe_offset(
                        slot_stride * (p if p < rank else p - 1)
                    )
                    .unsafe_bitcast[Scalar[dtype]]()
                    .unsafe_load[width=W, alignment=16](v * W)
                    .cast[accum]()
                )
        else:
            for j in range(1, world):
                var p = rank + j
                if p >= world:
                    p -= world
                acc += (
                    slots.unsafe_offset(
                        slot_stride * (p if p < rank else p - 1)
                    )
                    .unsafe_bitcast[Scalar[dtype]]()
                    .unsafe_load[width=W, alignment=16](v * W)
                    .cast[accum]()
                )
        comptime if accum.is_floating_point():
            acc *= SIMD[accum, W](scale.cast[accum]())
        shard.unsafe_store[width=W, alignment=16](v * W, acc.cast[dtype]())

    for i in range(tid, my_cnt - my_vc * W, stride):
        var k = my_vc * W + i
        var a = uin[unsafe_offset=k].cast[accum]()
        for j in range(1, world):
            var p = rank + j
            if p >= world:
                p -= world
            a += (
                slots.unsafe_offset(slot_stride * (p if p < rank else p - 1))
                .unsafe_bitcast[Scalar[dtype]]()[unsafe_offset=k]
                .cast[accum]()
            )
        comptime if accum.is_floating_point():
            a *= scale.cast[accum]()
        shard[unsafe_offset=k] = a.cast[dtype]()
    return True


@__llvm_metadata(
    MAX_THREADS_PER_BLOCK_METADATA=StaticTuple[Int32, 1](Int32(BLOCK))
)
@__name(t"ccl_reduce_scatter_stage_{dtype}_w{NW}")
def _rs_stage_kernel[
    dtype: DType, W: Int, U: Int, NW: Int
](
    regions: InlineArray[Pointer[UInt8, MutAnyOrigin], MAX_WORLD],
    in_ptr: Pointer[Scalar[dtype], MutAnyOrigin],
    numel: Int64,
    per_e: Int64,
    slot_stride_b: Int64,
    push_off_b: Int64,
    out_off_b: Int64,
    world_i: Int32,
    rank_i: Int32,
    flag_base: UInt64,
    scale: Float32,
    timeout_ns: UInt64,
):
    _ = _rs_stage_body[dtype, W, U, NW](
        regions,
        0,
        in_ptr,
        Int(numel),
        Int(per_e),
        Int(slot_stride_b),
        Int(push_off_b),
        Int(out_off_b),
        NW if NW > 0 else Int(world_i),
        Int(rank_i),
        Int(global_idx.x),
        Int(grid_dim.x) * BLOCK,
        flag_base,
        scale,
        device_now_ns(),
        timeout_ns,
    )


@always_inline
def _ag_finish_body[
    dtype: DType, W: Int, U: Int
](
    regions: InlineArray[Pointer[UInt8, MutAnyOrigin], MAX_WORLD],
    arena_off: Int,
    out_ptr: Pointer[Scalar[dtype], MutAnyOrigin],
    n: Int,
    per: Int,
    out_off: Int,
    world: Int,
    rank: Int,
    tid: Int,
    stride: Int,
    flag_base: UInt64,
    scale: Float32,
    t0: UInt64,
    timeout_ns: UInt64,
) -> Bool:
    """The all-gather half, as a body two kernels share (see
    `_rs_stage_body`)."""
    comptime esize = size_of[dtype]()

    # Start barrier. Doubles as the wait for every peer's inter-node step: a
    # peer publishes this flag from inside this kernel, which its stream runs
    # after that step.
    if not _sync(
        regions,
        world,
        rank,
        ERR_AG_FINISH_SYNC,
        flag_base,
        t0,
        timeout_ns,
        arena_off,
    ):
        return False

    # My own shard is pulled out of my own stage_out like everyone else's: the
    # inter-node step rewrote it, so the reduce-scatter's result in the user
    # buffer would be stale even if it had been written there.
    for i in range(world):
        var p = rank + _peer_step0(i, world)
        if p >= world:
            p -= world
        var off = _shard_off(n, per, p)
        var cnt = _shard_cnt(n, per, p)
        if cnt <= 0:
            continue
        var src = (
            regions[p]
            .unsafe_offset(arena_off + out_off + off * esize)
            .unsafe_bitcast[Scalar[dtype]]()
        )
        _copy_span_scaled[dtype, W, U](
            out_ptr.unsafe_offset(off), src, cnt, tid, stride, scale
        )
    return True


@__llvm_metadata(
    MAX_THREADS_PER_BLOCK_METADATA=StaticTuple[Int32, 1](Int32(BLOCK))
)
@__name(t"ccl_allgather_finish_{dtype}_w{NW}")
def _ag_finish_kernel[
    dtype: DType, W: Int, U: Int, NW: Int
](
    regions: InlineArray[Pointer[UInt8, MutAnyOrigin], MAX_WORLD],
    out_ptr: Pointer[Scalar[dtype], MutAnyOrigin],
    numel: Int64,
    per_e: Int64,
    out_off_b: Int64,
    world_i: Int32,
    rank_i: Int32,
    flag_base: UInt64,
    scale: Float32,
    timeout_ns: UInt64,
):
    _ = _ag_finish_body[dtype, W, U](
        regions,
        0,
        out_ptr,
        Int(numel),
        Int(per_e),
        Int(out_off_b),
        NW if NW > 0 else Int(world_i),
        Int(rank_i),
        Int(global_idx.x),
        Int(grid_dim.x) * BLOCK,
        flag_base,
        scale,
        device_now_ns(),
        timeout_ns,
    )


# ===-------------------------------------------------------------------=== #
# Broadcast and allgather -- one peer-copy kernel each, chunked to `cap`
# ===-------------------------------------------------------------------=== #
#
# Both walk the buffer in chunks of `cap` bytes and alternate between the two
# cap-sized halves of the arena, so the write of chunk c races only against
# reads of chunk c-1 (the other half); reads of chunk c-2 are ordered before
# it by the sync of chunk c-1. A leading sync separates the first two chunks
# from the previous call's reads.


@__llvm_metadata(
    MAX_THREADS_PER_BLOCK_METADATA=StaticTuple[Int32, 1](Int32(BLOCK))
)
@__name("ccl_broadcast_scatter_gather_bytes")
def _bcast_kernel[
    U: Int
](
    regions: InlineArray[Pointer[UInt8, MutAnyOrigin], MAX_WORLD],
    send: Pointer[UInt8, MutAnyOrigin],
    recv: Pointer[UInt8, MutAnyOrigin],
    nbytes: Int64,
    stage_off_b: Int64,
    world_i: Int32,
    rank_i: Int32,
    root_i: Int32,
    flag_base: UInt64,
    timeout_ns: UInt64,
):
    """Broadcast as scatter + all-gather.

    The obvious design -- root stages the whole message, everybody reads it --
    puts `(world-1) * nbytes` on the root's single outbound link and measured
    552 us on 27 MiB where the fabric could do 150. Instead the root scatters
    shard p into rank p's stage (nbytes out of the root, spread over the peers)
    and every other rank gathers the `world` shards, so the read side is a
    permutation too and no link carries more than nbytes.

    Out-of-place capable (ncclBroadcast): the root reads `send`, everyone
    writes `recv`, and the root copies `send` to `recv` locally when they
    differ (cheaper than gathering its own message back over NVLink).
    """
    var t0 = device_now_ns()
    var world = Int(world_i)
    var rank = Int(rank_i)
    var root = Int(root_i)
    var tid = Int(global_idx.x)
    var stride = Int(grid_dim.x) * BLOCK
    var n = Int(nbytes)
    var stage_off = Int(stage_off_b)
    var nv = n // 16
    var q = nv // world
    var rem = nv % world
    var mtail = n - nv * 16

    if not _sync(
        regions, world, rank, ERR_BROADCAST_SYNC, flag_base, t0, timeout_ns
    ):
        return

    if rank == root:
        for i in range(world):
            var p = root + i
            if p >= world:
                p -= world
            var vc = _vcount(p, q, rem)
            _copy_bytes[U](
                regions[p].unsafe_offset(stage_off),
                send.unsafe_offset(_vstart(p, q, rem) * 16),
                vc * 16 + (mtail if p == world - 1 else 0),
                tid,
                stride,
            )
        if Int(send) != Int(recv):
            _copy_bytes[U](recv, send, n, tid, stride)

    if not _sync(
        regions, world, rank, ERR_BROADCAST_SYNC, flag_base + 1, t0, timeout_ns
    ):
        return

    comptime if _AMD:
        # The gather half is a push too (module header, "Link direction").
        # After the scatter every rank holds its own shard in the scatter area
        # at `stage_off`; each then writes that shard into every OTHER
        # non-root rank's gather slot, and everyone assembles locally.  The
        # root needs nothing back -- it copied `send` to `recv` above -- so it
        # is skipped as a destination, which is also why `world-1` compacted
        # slots are enough.  The whole staging is `world` shard slots, i.e.
        # about `nbytes`, so the caller's chunking is unchanged.
        var sslot = (_vcount(0, q, rem) * 16 + mtail + 15) // 16 * 16
        var gbase = stage_off + sslot
        var my_vc = _vcount(rank, q, rem)
        var my_bytes = my_vc * 16 + (mtail if rank == world - 1 else 0)
        var mine = regions[rank].unsafe_offset(stage_off)
        if rank != root:
            _copy_bytes[U](
                recv.unsafe_offset(_vstart(rank, q, rem) * 16),
                mine,
                my_bytes,
                tid,
                stride,
            )
        for i in range(1, world):
            var p = rank + _peer_step(i, world)
            if p >= world:
                p -= world
            if p == root:
                continue
            _copy_bytes[U](
                regions[p].unsafe_offset(gbase + sslot * _gather_slot(rank, p)),
                mine,
                my_bytes,
                tid,
                stride,
            )
        if not _sync(
            regions,
            world,
            rank,
            ERR_BROADCAST_SYNC,
            flag_base + 2,
            t0,
            timeout_ns,
        ):
            return
        if rank != root:
            for i in range(1, world):
                var p = rank + _peer_step(i, world)
                if p >= world:
                    p -= world
                var vc = _vcount(p, q, rem)
                _copy_bytes[U](
                    recv.unsafe_offset(_vstart(p, q, rem) * 16),
                    regions[rank].unsafe_offset(
                        gbase + sslot * _gather_slot(p, rank)
                    ),
                    vc * 16 + (mtail if p == world - 1 else 0),
                    tid,
                    stride,
                )
        return

    if rank != root:
        for i in range(world):
            var p = rank + _peer_step0(i, world)
            if p >= world:
                p -= world
            var vc = _vcount(p, q, rem)
            _copy_bytes[U](
                recv.unsafe_offset(_vstart(p, q, rem) * 16),
                regions[p].unsafe_offset(stage_off),
                vc * 16 + (mtail if p == world - 1 else 0),
                tid,
                stride,
            )


@__llvm_metadata(
    MAX_THREADS_PER_BLOCK_METADATA=StaticTuple[Int32, 1](Int32(BLOCK))
)
@__name("ccl_allgather_bytes")
def _allgather_kernel[
    U: Int
](
    regions: InlineArray[Pointer[UInt8, MutAnyOrigin], MAX_WORLD],
    in_ptr: Pointer[UInt8, MutAnyOrigin],
    out_ptr: Pointer[UInt8, MutAnyOrigin],
    nbytes: Int64,
    stride_b: Int64,
    stage_off_b: Int64,
    world_i: Int32,
    rank_i: Int32,
    flag_base: UInt64,
    timeout_ns: UInt64,
):
    """Local stage + peer gather -- already the unicast minimum: `nbytes` of
    local copy and `(world-1)*nbytes` of peer reads per GPU.

    Rank r's contribution lands at `out_ptr + r*stride_b`; `stride_b` is the
    output layout's true per-rank size, which differs from `nbytes` when the
    caller splits one rank's contribution across several calls.
    """
    var t0 = device_now_ns()
    var world = Int(world_i)
    var rank = Int(rank_i)
    var tid = Int(global_idx.x)
    var stride = Int(grid_dim.x) * BLOCK
    var n = Int(nbytes)
    var out_stride = Int(stride_b)
    var stage_off = Int(stage_off_b)

    comptime if _AMD:
        # Push instead of pull (module header, "Link direction"): my
        # contribution goes into every peer's slot for me and straight into my
        # own output slice, and the gather is then a local copy out of my own
        # region.  `world-1` compacted slots, addressed by `_gather_slot`, so
        # the staging is `(world-1) * nbytes` -- which is why
        # `allgather_max_bytes` chunks smaller here than on NVIDIA.
        var slot = (n + 15) // 16 * 16
        if not _sync(
            regions, world, rank, ERR_ALLGATHER_SYNC, flag_base, t0, timeout_ns
        ):
            return
        _copy_bytes[U](
            out_ptr.unsafe_offset(rank * out_stride), in_ptr, n, tid, stride
        )
        for i in range(1, world):
            var p = rank + _peer_step(i, world)
            if p >= world:
                p -= world
            _copy_bytes[U](
                regions[p].unsafe_offset(
                    stage_off + slot * _gather_slot(rank, p)
                ),
                in_ptr,
                n,
                tid,
                stride,
            )
        if not _sync(
            regions,
            world,
            rank,
            ERR_ALLGATHER_SYNC,
            flag_base + 1,
            t0,
            timeout_ns,
        ):
            return
        for i in range(1, world):
            var p = rank + _peer_step(i, world)
            if p >= world:
                p -= world
            _copy_bytes[U](
                out_ptr.unsafe_offset(p * out_stride),
                regions[rank].unsafe_offset(
                    stage_off + slot * _gather_slot(p, rank)
                ),
                n,
                tid,
                stride,
            )
        return

    if not _sync(
        regions, world, rank, ERR_ALLGATHER_SYNC, flag_base, t0, timeout_ns
    ):
        return

    _copy_bytes[U](
        regions[rank].unsafe_offset(stage_off), in_ptr, n, tid, stride
    )
    _copy_bytes[U](
        out_ptr.unsafe_offset(rank * out_stride), in_ptr, n, tid, stride
    )

    if not _sync(
        regions, world, rank, ERR_ALLGATHER_SYNC, flag_base + 1, t0, timeout_ns
    ):
        return

    for i in range(1, world):
        var p = rank + _peer_step(i, world)
        if p >= world:
            p -= world
        _copy_bytes[U](
            out_ptr.unsafe_offset(p * out_stride),
            regions[p].unsafe_offset(stage_off),
            n,
            tid,
            stride,
        )


# ===-------------------------------------------------------------------=== #
# region_init
# ===-------------------------------------------------------------------=== #


@__llvm_metadata(
    MAX_THREADS_PER_BLOCK_METADATA=StaticTuple[Int32, 1](Int32(BLOCK))
)
@__name("ccl_zero_signal_area")
def _zero_kernel(ptr: Pointer[UInt32, MutAnyOrigin], nwords: Int32):
    var i = Int(global_idx.x)
    if i < Int(nwords):
        ptr[unsafe_offset=i] = 0


@__llvm_metadata(
    MAX_THREADS_PER_BLOCK_METADATA=StaticTuple[Int32, 1](Int32(BLOCK))
)
@__name("ccl_store_u64")
def _store_u64_kernel(dst: Pointer[UInt64, MutAnyOrigin], value: UInt64):
    if global_idx.x == 0:
        dst[unsafe_offset=0] = value


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
    (`internode_fused.mojo`). Asked once per (kernel, context)."""
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


@always_inline
def _region_ptrs(
    regions: StaticTuple[Int, MAX_WORLD], rank: Int, world: Int
) -> InlineArray[Pointer[UInt8, MutAnyOrigin], MAX_WORLD]:
    """Peer region bases as mapped in this process. Unused slots are filled
    with my own region so no kernel can ever hold a null pointer."""
    var out = InlineArray[Pointer[UInt8, MutAnyOrigin], MAX_WORLD](
        uninitialized=True
    )
    for r in range(MAX_WORLD):
        var addr = regions[r] if r < world else regions[rank]
        out[r] = Pointer[UInt8, MutAnyOrigin](unsafe_from_address=addr)
    return out^


def _check_common(
    rank: Int, world: Int, cap_bytes: Int, generation: Int
) raises:
    if world < 1 or world > MAX_WORLD:
        raise Error("collectives: world must be in 1.." + String(MAX_WORLD))
    if rank < 0 or rank >= world:
        raise Error("collectives: rank out of range")
    if generation < 1:
        raise Error("collectives: generation must start at 1")
    if cap_bytes <= 0 or cap_bytes % 4096 != 0:
        raise Error("collectives: cap_bytes must be a positive 4 KiB multiple")


# ===-------------------------------------------------------------------=== #
# Public API
# ===-------------------------------------------------------------------=== #


def region_init(ctx: DeviceContext, region: Int) raises:
    """Zero the signal area of this rank's region and block until it is done.

    Called once per rank at communicator creation, on `ctx`'s own queue -- no
    `stream` parameter, unlike the collectives: this must be complete before
    any peer writes a flag into the area, and the caller only has to make the
    ranks meet (its own rendezvous) after calling it.
    """
    var words = _SIGNAL_BYTES // 4
    _enqueue_cached[_zero_kernel](
        ctx,
        ctx.stream(),
        "zero",
        (words + BLOCK - 1) // BLOCK,
        Pointer[UInt32, MutAnyOrigin](unsafe_from_address=region),
        Int32(words),
    )
    ctx.synchronize()


def install_status_page(ctx: DeviceContext, region: Int, dev_addr: Int) raises:
    """Publish the device mapping of the communicator's pinned status page in
    this region's header, and block until it has landed.

    The spins read it out of the region rather than take it as a kernel
    argument because the launcher signatures below are fixed; the region is
    raw driver memory, so the host cannot poke the slot directly. Called once
    per arena, right after `region_init` zeroed it and before any peer can
    reach the region.
    """
    _enqueue_cached[_store_u64_kernel](
        ctx,
        ctx.stream(),
        "statusptr",
        1,
        Pointer[UInt64, MutAnyOrigin](
            unsafe_from_address=region + _STATUS_PTR_OFFSET
        ),
        UInt64(dev_addr),
    )
    ctx.synchronize()


@always_inline
def _launch_allreduce[
    dtype: DType, W: Int, NW: Int
](
    ctx: DeviceContext,
    stream: DeviceStream,
    regions: InlineArray[Pointer[UInt8, MutAnyOrigin], MAX_WORLD],
    in_ptr: Int,
    out_ptr: Int,
    numel: Int,
    world: Int,
    rank: Int,
    cap_bytes: Int,
    scale: Float32,
    generation: Int,
    one_shot: Bool,
) raises:
    var arena = _SIGNAL_BYTES
    var arena_end = _SIGNAL_BYTES + 2 * cap_bytes
    var esize = size_of[dtype]()
    var nvec = numel // W
    var tail = numel - nvec * W
    var ip = Pointer[Scalar[dtype], MutAnyOrigin](unsafe_from_address=in_ptr)
    var op = Pointer[Scalar[dtype], MutAnyOrigin](unsafe_from_address=out_ptr)

    if one_shot:
        # `world` slots, each a whole message, at the base of the arena. They
        # overlap the two-shot staging on purpose: the start barrier orders
        # every generation's writes after the previous generation's reads, so
        # collectives of different shapes and sizes may interleave freely.
        var slot = _align_up(numel * esize, 16)
        if world * slot > 2 * cap_bytes:
            raise Error("collectives: one-shot slots exceed the region")
        var push_off = arena
        var blocks = min(_AR_MAX_BLOCKS, max(1, (nvec + BLOCK - 1) // BLOCK))
        _enqueue_cached[_ar_oneshot_kernel[dtype, W, _UNROLL, NW]](
            ctx,
            stream,
            String(t"ar1_{dtype}_{NW}"),
            blocks,
            regions,
            ip,
            op,
            Int64(numel),
            Int64(slot),
            Int64(push_off),
            Int32(world),
            Int32(rank),
            _flag_target(generation, 0),
            scale,
            spin_timeout_ns(),
        )
        return

    var q = nvec // world
    var rem = nvec % world
    var max_shard_elems = max((q + (1 if rem > 0 else 0)) * W, q * W + tail)
    var slot = _align_up(max_shard_elems * esize, 16)
    var push_off = arena
    var shard_off = arena + world * slot
    # NVIDIA parks one reduced shard at `shard_off` for the peers to pull;
    # AMD parks `world-1` gather slots there for the peers to push into
    # (module header, "Link direction").  `world*slot` is already about
    # `numel*esize` <= cap, so `world` more slots would not fit an arena of
    # 2*cap when the message is exactly cap bytes -- hence the compacted
    # `world-1`, whose worst case is cap*(2 - 1/world) plus alignment.
    comptime tail_slots = 1
    var end = shard_off + (max(world - 1, 1) if _AMD else tail_slots) * slot
    if end > arena_end:
        raise Error("collectives: allreduce staging exceeds the region")
    var cap_blocks = (
        _AR_BIG_BLOCKS if numel * esize >= _AR_BIG_BYTES else _AR_MAX_BLOCKS
    )
    var blocks = min(cap_blocks, max(1, (q + 1 + BLOCK - 1) // BLOCK))
    _enqueue_cached[_ar_twoshot_kernel[dtype, W, _UNROLL, NW]](
        ctx,
        stream,
        String(t"ar2_{dtype}_{NW}"),
        blocks,
        regions,
        ip,
        op,
        Int64(numel),
        Int64(slot),
        Int64(push_off),
        Int64(shard_off),
        Int32(world),
        Int32(rank),
        _flag_target(generation, 0),
        scale,
        spin_timeout_ns(),
    )


def allreduce[
    dtype: DType
](
    ctx: DeviceContext,
    stream: DeviceStream,
    rank: Int,
    world: Int,
    regions: StaticTuple[Int, MAX_WORLD],
    in_ptr: Int,
    out_ptr: Int,
    numel: Int,
    cap_bytes: Int,
    scale: Float32,
    generation: Int,
) raises:
    """Elementwise `out[i] = scale * sum over ranks of in_r[i]`, on `stream`.

    `in_ptr` may equal `out_ptr`. `numel * size_of[dtype]()` must be <=
    `cap_bytes`. `scale` is applied on the final write and ignored for integer
    dtypes (the caller passes 1.0 there).
    """
    _check_common(rank, world, cap_bytes, generation)
    if numel == 0:
        return
    if numel < 0:
        raise Error("collectives: numel must be >= 0")
    comptime W = 16 // size_of[dtype]()
    if numel * size_of[dtype]() > cap_bytes:
        raise Error("collectives: allreduce message exceeds cap_bytes")
    if (in_ptr | out_ptr) % 16 != 0:
        # The payload loops use 16-byte vector loads/stores, which fault (or
        # silently misbehave) on a misaligned address. Every allocator-returned
        # pointer satisfies this; a mid-tensor view may not, and the caller
        # must stage such a tensor into an aligned buffer rather than have this
        # kernel guess. Byte collectives have a scalar fallback and need no
        # such rule.
        raise Error(
            "collectives: allreduce needs 16-byte aligned in_ptr and out_ptr"
        )
    var rp = _region_ptrs(regions, rank, world)
    # world == 1 needs no special case: the push and pull loops are empty, the
    # sync is a self-rendezvous, and the reduce degenerates to out = scale*in.
    var bytes = numel * size_of[dtype]()
    var one_shot = bytes <= _ONESHOT_MAX_BYTES and (
        world * _align_up(bytes, 16) <= 2 * cap_bytes
    )

    if world == 8:
        _launch_allreduce[dtype, W, 8](
            ctx,
            stream,
            rp,
            in_ptr,
            out_ptr,
            numel,
            world,
            rank,
            cap_bytes,
            scale,
            generation,
            one_shot,
        )
    elif world == 4:
        _launch_allreduce[dtype, W, 4](
            ctx,
            stream,
            rp,
            in_ptr,
            out_ptr,
            numel,
            world,
            rank,
            cap_bytes,
            scale,
            generation,
            one_shot,
        )
    elif world == 2:
        _launch_allreduce[dtype, W, 2](
            ctx,
            stream,
            rp,
            in_ptr,
            out_ptr,
            numel,
            world,
            rank,
            cap_bytes,
            scale,
            generation,
            one_shot,
        )
    else:
        _launch_allreduce[dtype, W, 0](
            ctx,
            stream,
            rp,
            in_ptr,
            out_ptr,
            numel,
            world,
            rank,
            cap_bytes,
            scale,
            generation,
            one_shot,
        )


@always_inline
def _split_blocks[dtype: DType, W: Int](numel: Int, per: Int) -> Int:
    """Grid for both halves of the split allreduce: sized by the shard, capped
    by the same one-wave rule the fused kernel uses (RESULTS.md block sweep).
    The two halves need not agree -- see ordering note 2 above the kernels."""
    var cap_blocks = (
        _AR_BIG_BLOCKS if numel * size_of[dtype]()
        >= _AR_BIG_BYTES else _AR_MAX_BLOCKS
    )
    return min(cap_blocks, max(1, (per // W + 1 + BLOCK - 1) // BLOCK))


def _check_split[
    dtype: DType, W: Int
](
    rank: Int,
    world: Int,
    ptr: Int,
    numel: Int,
    cap_bytes: Int,
    generation: Int,
    what: String,
) raises -> Int:
    """Shared preconditions of the two split entry points; returns `per`."""
    _check_common(rank, world, cap_bytes, generation)
    if numel < 0:
        raise Error("collectives: numel must be >= 0")
    if numel * size_of[dtype]() > cap_bytes:
        raise Error("collectives: " + what + " message exceeds cap_bytes")
    if ptr % 16 != 0:
        # Same rule as `allreduce`: the payload loops use 16-byte vectors.
        raise Error("collectives: " + what + " needs a 16-byte aligned buffer")
    var per = _shard_per(numel, world, W)
    if (world - 1) * per * size_of[dtype]() > cap_bytes:
        # Cannot happen -- the compacted slot table is ~(world-1)/world of the
        # message -- but the arena has no guard page, so check it anyway.
        raise Error("collectives: split push slots exceed the region")
    return per


@always_inline
def _launch_rs_stage[
    dtype: DType, W: Int, NW: Int
](
    ctx: DeviceContext,
    stream: DeviceStream,
    regions: InlineArray[Pointer[UInt8, MutAnyOrigin], MAX_WORLD],
    in_ptr: Int,
    numel: Int,
    per: Int,
    world: Int,
    rank: Int,
    cap_bytes: Int,
    generation: Int,
    scale: Float32,
) raises:
    _enqueue_cached[_rs_stage_kernel[dtype, W, _UNROLL, NW]](
        ctx,
        stream,
        String(t"rs_{dtype}_{NW}"),
        _split_blocks[dtype, W](numel, per),
        regions,
        Pointer[Scalar[dtype], MutAnyOrigin](unsafe_from_address=in_ptr),
        Int64(numel),
        Int64(per),
        Int64(per * size_of[dtype]()),
        Int64(_SIGNAL_BYTES),
        Int64(_SIGNAL_BYTES + cap_bytes),
        Int32(world),
        Int32(rank),
        _flag_target(generation, 0),
        scale,
        spin_timeout_ns(),
    )


@always_inline
def _launch_ag_finish[
    dtype: DType, W: Int, NW: Int
](
    ctx: DeviceContext,
    stream: DeviceStream,
    regions: InlineArray[Pointer[UInt8, MutAnyOrigin], MAX_WORLD],
    out_ptr: Int,
    numel: Int,
    per: Int,
    world: Int,
    rank: Int,
    cap_bytes: Int,
    scale: Float32,
    generation: Int,
) raises:
    _enqueue_cached[_ag_finish_kernel[dtype, W, _UNROLL, NW]](
        ctx,
        stream,
        String(t"agf_{dtype}_{NW}"),
        _split_blocks[dtype, W](numel, per),
        regions,
        Pointer[Scalar[dtype], MutAnyOrigin](unsafe_from_address=out_ptr),
        Int64(numel),
        Int64(per),
        Int64(_SIGNAL_BYTES + cap_bytes),
        Int32(world),
        Int32(rank),
        _flag_target(generation, 0),
        scale,
        spin_timeout_ns(),
    )


def reduce_scatter_stage[
    dtype: DType
](
    ctx: DeviceContext,
    stream: DeviceStream,
    rank: Int,
    world: Int,
    regions: StaticTuple[Int, MAX_WORLD],
    in_ptr: Int,
    numel: Int,
    cap_bytes: Int,
    generation: Int,
    scale: Float32 = Float32(1.0),
) raises:
    """First half of a hierarchical allreduce: push + local reduce, on `stream`.

    When the enqueued work completes, this rank's shard --
    `shard_range(numel, world, rank, size_of[dtype]())` -- summed over the
    `world` node-local ranks and multiplied by `scale` (ignored for integer
    dtypes), sits in this rank's own stage_out, at element offset `offset`
    from `region + signal_bytes() + cap_bytes`. The caller then runs the
    inter-node collective in place on exactly that range, on this same
    stream, and calls `allgather_finish` with `generation + 1` and scale 1.

    `scale` goes here rather than into `allgather_finish` so that an AVG never
    stores an unscaled sum in a narrow wire dtype (NCCL's PreMulSum); pass
    the communicator-wide 1/world, and the inter-node SUM of the node
    partials is the average.

    `rank` / `world` / `regions` are the node-local group. Preconditions are
    `allreduce`'s: 16-byte aligned `in_ptr`, `numel * size_of[dtype]() <=
    cap_bytes`, strictly increasing `generation`.
    """
    comptime W = 16 // size_of[dtype]()
    var per = _check_split[dtype, W](
        rank, world, in_ptr, numel, cap_bytes, generation, "reduce_scatter"
    )
    if numel == 0:
        return
    var rp = _region_ptrs(regions, rank, world)
    # world == 1 needs no special case: the push loop is empty, the sync is a
    # self-rendezvous and the reduce copies the input into stage_out.
    if world == 8:
        _launch_rs_stage[dtype, W, 8](
            ctx,
            stream,
            rp,
            in_ptr,
            numel,
            per,
            world,
            rank,
            cap_bytes,
            generation,
            scale,
        )
    elif world == 4:
        _launch_rs_stage[dtype, W, 4](
            ctx,
            stream,
            rp,
            in_ptr,
            numel,
            per,
            world,
            rank,
            cap_bytes,
            generation,
            scale,
        )
    elif world == 2:
        _launch_rs_stage[dtype, W, 2](
            ctx,
            stream,
            rp,
            in_ptr,
            numel,
            per,
            world,
            rank,
            cap_bytes,
            generation,
            scale,
        )
    else:
        _launch_rs_stage[dtype, W, 0](
            ctx,
            stream,
            rp,
            in_ptr,
            numel,
            per,
            world,
            rank,
            cap_bytes,
            generation,
            scale,
        )


def allgather_finish[
    dtype: DType
](
    ctx: DeviceContext,
    stream: DeviceStream,
    rank: Int,
    world: Int,
    regions: StaticTuple[Int, MAX_WORLD],
    out_ptr: Int,
    numel: Int,
    cap_bytes: Int,
    scale: Float32,
    generation: Int,
) raises:
    """Second half: start barrier, then pull every rank's shard, on `stream`.

    The start barrier is also the wait for the peers' inter-node steps -- a
    peer publishes this generation's flags from inside this kernel, which its
    stream runs only after that step. Then rank p's shard is read from p's
    stage_out (mine included, since the inter-node step rewrote it) into
    `out_ptr` at the same element offset, times `scale` (ignored for integer
    dtypes). Same `numel`, `world` and preconditions as the matching
    `reduce_scatter_stage`; `generation` is that call's plus one.

    In place is safe: `out_ptr` may be the `in_ptr` the matching
    `reduce_scatter_stage` read, because the two are separate launches on one
    stream and no rank ever touches another rank's user memory.

    No exit protocol is needed: the next collective's start barrier already
    orders any arena reuse after every peer's pulls.
    """
    comptime W = 16 // size_of[dtype]()
    var per = _check_split[dtype, W](
        rank, world, out_ptr, numel, cap_bytes, generation, "allgather_finish"
    )
    if numel == 0:
        return
    var rp = _region_ptrs(regions, rank, world)
    if world == 8:
        _launch_ag_finish[dtype, W, 8](
            ctx,
            stream,
            rp,
            out_ptr,
            numel,
            per,
            world,
            rank,
            cap_bytes,
            scale,
            generation,
        )
    elif world == 4:
        _launch_ag_finish[dtype, W, 4](
            ctx,
            stream,
            rp,
            out_ptr,
            numel,
            per,
            world,
            rank,
            cap_bytes,
            scale,
            generation,
        )
    elif world == 2:
        _launch_ag_finish[dtype, W, 2](
            ctx,
            stream,
            rp,
            out_ptr,
            numel,
            per,
            world,
            rank,
            cap_bytes,
            scale,
            generation,
        )
    else:
        _launch_ag_finish[dtype, W, 0](
            ctx,
            stream,
            rp,
            out_ptr,
            numel,
            per,
            world,
            rank,
            cap_bytes,
            scale,
            generation,
        )


def broadcast(
    ctx: DeviceContext,
    stream: DeviceStream,
    rank: Int,
    root: Int,
    world: Int,
    regions: StaticTuple[Int, MAX_WORLD],
    send_ptr: Int,
    recv_ptr: Int,
    nbytes: Int,
    cap_bytes: Int,
    generation: Int,
) raises:
    """Every rank's `recv` <- `root`'s `send`, dtype-agnostic, on `stream`.

    Out-of-place capable: `send_ptr` and `recv_ptr` may differ (only the root
    reads `send_ptr`). `nbytes` must be <= `cap_bytes`; the caller chunks
    anything larger.
    """
    _check_common(rank, world, cap_bytes, generation)
    if root < 0 or root >= world:
        raise Error("collectives: root out of range")
    if nbytes == 0:
        return
    if nbytes < 0:
        raise Error("collectives: nbytes must be >= 0")
    if nbytes > cap_bytes:
        raise Error("collectives: broadcast message exceeds cap_bytes")
    var rp = _region_ptrs(regions, rank, world)
    var blocks = min(
        _COPY_MAX_BLOCKS,
        max(1, (nbytes // 16 // world + BLOCK - 1) // BLOCK),
    )
    _enqueue_cached[_bcast_kernel[_UNROLL]](
        ctx,
        stream,
        "bcast",
        blocks,
        rp,
        Pointer[UInt8, MutAnyOrigin](unsafe_from_address=send_ptr),
        Pointer[UInt8, MutAnyOrigin](unsafe_from_address=recv_ptr),
        Int64(nbytes),
        Int64(_SIGNAL_BYTES),
        Int32(world),
        Int32(rank),
        Int32(root),
        _flag_target(generation, 0),
        spin_timeout_ns(),
    )


def allgather_max_bytes(cap_bytes: Int, world: Int) -> Int:
    """Largest per-rank contribution one `allgather` call may carry.

    NVIDIA stages one message-sized buffer per rank in its own region and
    reads the peers', so `cap_bytes` is the bound and this is the identity.
    AMD pushes instead, which needs `world-1` message-sized slots inside the
    `2*cap_bytes` arena; the caller chunks to that.
    """
    comptime if _AMD:
        if world <= 2:
            return cap_bytes
        return min(cap_bytes, (2 * cap_bytes // (world - 1)) // 16 * 16)
    return cap_bytes


def allgather(
    ctx: DeviceContext,
    stream: DeviceStream,
    rank: Int,
    world: Int,
    regions: StaticTuple[Int, MAX_WORLD],
    in_ptr: Int,
    out_ptr: Int,
    nbytes_per_rank: Int,
    cap_bytes: Int,
    generation: Int,
    stride_bytes: Int = -1,
) raises:
    """Gather: `out[r*stride_bytes ...]` <- rank r's input, on `stream`.

    `stride_bytes` defaults to `nbytes_per_rank` and is the output layout's
    true per-rank size, which differs when the caller splits one rank's
    contribution across several calls. `nbytes_per_rank` must be <=
    `cap_bytes`.
    """
    _check_common(rank, world, cap_bytes, generation)
    if nbytes_per_rank == 0:
        return
    if nbytes_per_rank < 0:
        raise Error("collectives: nbytes_per_rank must be >= 0")
    if nbytes_per_rank > allgather_max_bytes(cap_bytes, world):
        raise Error("collectives: allgather message exceeds cap_bytes")
    var stride = stride_bytes if stride_bytes >= 0 else nbytes_per_rank
    if stride < nbytes_per_rank:
        raise Error("collectives: stride_bytes < nbytes_per_rank")
    var rp = _region_ptrs(regions, rank, world)
    var blocks = min(
        _COPY_MAX_BLOCKS,
        max(1, (nbytes_per_rank // 16 + BLOCK - 1) // BLOCK),
    )
    _enqueue_cached[_allgather_kernel[_UNROLL]](
        ctx,
        stream,
        "allgather",
        blocks,
        rp,
        Pointer[UInt8, MutAnyOrigin](unsafe_from_address=in_ptr),
        Pointer[UInt8, MutAnyOrigin](unsafe_from_address=out_ptr),
        Int64(nbytes_per_rank),
        Int64(stride),
        Int64(_SIGNAL_BYTES),
        Int32(world),
        Int32(rank),
        _flag_target(generation, 0),
        spin_timeout_ns(),
    )
