# Rewrite of: https://github.com/NVIDIA/nccl/blob/master/src/include/device.h
#
# One library-owned device allocation per rank ("the region"), shared to the
# peers by cuIpc/hipIpc (the plumbing layer owns that half), laid out as
#
#   [0, SIGNAL_BYTES)                            signal area, zero at creation
#   [SIGNAL_BYTES, SIGNAL_BYTES + cap)           "stage_in"
#   [SIGNAL_BYTES + cap, SIGNAL_BYTES + 2*cap)   "stage_out"
#
# The two cap-sized halves are library scratch; the kernels use them as one
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

from std.sys.info import _accelerator_arch
from std.sys import has_amd_gpu_accelerator


# ===-------------------------------------------------------------------=== #
# Compile-time configuration
# ===-------------------------------------------------------------------=== #

comptime _AMD = has_amd_gpu_accelerator()
"""Whether this build targets AMD.  Every behavioural difference in this file
is behind it, so the NVIDIA path is exactly what it was before the MI300A work
(see the "Link direction" note in the module header)."""

comptime _GFX942 = (
    _accelerator_arch() == "gfx942" or _accelerator_arch() == "amdgpu:gfx942"
)
"""Whether this build targets gfx942, the one spelling of it in the CCL.

The host pass uses the bare --target-accelerator name, while device
compilation can use the target-qualified spelling; both are the same
architecture. gfx942 covers the MI300A APU and the discrete MI300X and
MI325X alike. The multi-node grid rule RCCL applies only to the APU is
chosen at run time (`_node_grids` in init.mojo), not here."""


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
barrier counters (device/all_reduce.mojo), so 256 is the first free line."""

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
previous collective (`_order_before` / `_order_after`, enqueue.mojo), so
every collective of a communicator runs in one total order whatever streams
the caller uses. The process group issues everything on one comm stream
anyway."""

comptime _POISON_OFFSET = 448
"""Device word one block of a persistent kernel raises to tell the others to
give up (all_reduce_gin.mojo). Device memory, unlike the status page's abort
word: every block reads it once per chunk, and that read has to be an L2 hit
rather than a PCIe round trip."""

comptime _AG_ARRIVE_OFFSET = 4032
"""Arrival counter of the mapped all-gather's stage phase: the last block to
finish staging this rank's contribution releases the chunk's RDMA exchange
(`_allgather_body`). Per arena, rank-local, zeroed with the signal area and
reset by the last arriver, so nothing carries across launches."""

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

# Measured algorithm constants. Changes are fitted and reviewed in source.

comptime _UNROLL = 4
"""16-byte vectors in flight per thread in the NVLink copy loops."""

comptime _ONESHOT_MAX_BYTES = 512 * 1024
"""At or below this an allreduce uses the one-shot path: (world-1)x the NVLink
bytes but one sync instead of two, which wins while the transfer is
latency-bound. Measured crossover on 8xH100/NVSwitch (us, one-shot vs
two-shot): 128 KiB 9.3/19.0, 256 KiB 12.3/19.3, 512 KiB 18.1/19.8,
1 MiB 30.0/20.5 -- so the crossover sits just above 512 KiB. The path is taken
only if 2*world message-sized slots also fit the region."""

comptime _AR_MAX_BLOCKS = 128 if has_amd_gpu_accelerator() else 216
"""Grid cap for allreduce, fitted on H100 (132 SMs) / NVSwitch; see the block
sweep in RESULTS.md. Not portable: re-fit it on another card. The AMD value
was swept on one 4x MI300A node (228 CUs) with the push/reduce/push-back
schedule and the barrier this file uses there, fp32, 4 ranks, us at
9 / 27 MiB: 64 -> 118 / 285, **128 -> 121 / 254**, 224 -> 144 / 258. The cost
of the grid here is the barrier's `buffer_wbl2` per thread, which is why the
best value moved down from 224 once the release fence went back to every
thread; 64 starves the 27 MiB transfer. Re-fit it on another card, and re-fit
it if the barrier changes."""

comptime _AR_BIG_BYTES = 64 * 1024 * 1024
"""Above this message size the allreduce grid drops to `_AR_BIG_BLOCKS`."""

comptime _AR_BIG_BLOCKS = 912 if has_amd_gpu_accelerator() else 128
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

comptime _COPY_MAX_BLOCKS = 432
"""Grid cap for the pure-copy collectives (broadcast / allgather)."""

# Error codes written to the region's error word (`error_offset()`).
comptime ERR_ALLREDUCE_SYNC = 1
comptime ERR_BROADCAST_SYNC = 2
comptime ERR_ALLGATHER_SYNC = 3
comptime ERR_RS_STAGE_SYNC = 4
comptime ERR_AG_FINISH_SYNC = 5
# 6 is device/all_reduce.mojo's ERR_NVLS_SYNC; 8 is free.
comptime ERR_REDUCE_SCATTER_SYNC = 7
"""`reduce_scatter`'s barriers: the push/reduce kernel that writes the user's
output directly, not the split allreduce's `reduce_scatter_stage` (code 4)."""
comptime ERR_PROXY_WAIT = 9
"""`gin_proxy.mojo`'s wait for the inter-node progress thread. The
value is what that kernel has always written, so old logs still decode."""

comptime ERR_FUSED_GRID = 10
"""A grid barrier inside the fused inter-node allreduce (all_reduce_gin.mojo)
gave up: this rank's own blocks stopped arriving, which only happens because
another spin in the same kernel already failed and returned."""

comptime ERR_HOST_LAUNCH = 11
"""The host failed to launch the fused inter-node allreduce after reserving
its exchange counters (enqueue.mojo `_do_allreduce_fused`): the peers will
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
# One page per communicator, allocated by init.mojo and published in every
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
