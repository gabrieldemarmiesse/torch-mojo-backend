"""Dynamic persistent-tile scheduler: counter pool (host) + ticket protocol
(device).

The problem this solves: `_rolling_persistent_body`
(gemm16_rolling_kernels.mojo) and `_v4_nn_persistent_ws`
(gemm16_nn_v4_kernels.mojo) used to assign output tiles STATICALLY --
CTA/cluster `c` owned works `c, c + num_clusters, c + 2 * num_clusters,
...`.  The grid is one cluster per pair of SMs and each CTA holds ~213 KiB
of shared memory, so exactly one CTA fits per SM: when another kernel holds
K SMs -- which is the normal state of a DDP backward pass, where NCCL's
allreduce kernels run on the comm stream while these GEMMs run on the
compute stream -- K CTAs cannot launch at all.  Their static share is not
redistributed; it runs only after the resident CTAs retire, at the very end,
so a kernel that should have lost `K / sms` of its throughput instead ran
nearly twice as long (measured: +78% to +93% at K = 16 across eight GPT-2 XL
dX/dW shapes, against cuBLAS's +13% to +55% on the same hog).

The fix is the CUTLASS persistent-tile-scheduler shape: one global atomic
counter per launch, each cluster taking the next work index when it is ready
for one.  A cluster that never launches takes zero works; the makespan
becomes `ceil(total_works / clusters_that_actually_ran)` rounds instead of
`static_share + tail`.

## Who fetches, and how the cluster agrees

Both CTAs of a `cluster_m = 2` cluster must process the SAME work index:
they split one macro-row (`m0 = macro_row * bm * cluster_m + rank * bm`) and
multicast each other's halves of the B tile.  Two independent atomicAdds
would hand them different tiles, so exactly one thread in the whole cluster
fetches:

* rank 0's producer thread does the `atomicAdd` and publishes the ticket in
  its own shared-memory ring;
* rank != 0's producer thread reads rank 0's ring through DSMEM
  (`mapa.shared::cluster` + `ld.relaxed.cluster.shared::cluster.b32`) and
  republishes it in its own ring;
* every consumer thread of either CTA reads its OWN CTA's ring.

so the cross-CTA traffic is one 32-bit DSMEM read per work per cluster, and
the consumers never touch DSMEM at all.

## Ring, tags and the overwrite bound

A ring slot is one 32-bit word, `((round % SCHED_TAG_PERIOD) + 1) << 22 |
(w + 1)`: packing tag and payload into ONE word makes publication atomic at
the hardware's 32-bit granularity, so a reader either sees the old word or
the whole new one and no fence is needed between "data" and "flag".  Tag 0
is never produced, so the zero-initialised ring cannot be mistaken for a
published round 0.  22 bits cover any `total_works <= SCHED_MAX_WORKS`; the
host gate `sched_supported()` refuses anything larger rather than aliasing
silently, and the enqueuers then decline so the dispatch ladder's next rung
serves the shape.

Slot `j % SCHED_RING` is rewritten by round `j + SCHED_RING`.  The producer
publishes round `j + 1` at the top of its iteration `j` (a one-round
lookahead, so the peer rank's poll is already satisfied when it looks), and
its issue of round `j` is gated by `empty_barriers`, which every consumer of
every CTA in the cluster arrives at -- so publication runs at most
`stages + 1` rounds ahead of the slowest reader in the cluster.
`SCHED_RING >= stages + 4` therefore guarantees a slot is re-used only after
every reader has read it; a comptime assert in each kernel enforces it.

## The counter, and why no atomic detects the end of a launch

Every cluster fetches exactly (its works + 1) tickets -- one per work plus
the one past the end that stops its loop -- so a launch issues exactly
`total_works + num_clusters` tickets, and the cluster whose terminating
ticket is `total_works + num_clusters - 1` holds the highest ticket the
launch will ever issue.  Nobody fetches after that, so that cluster simply
stores 0 into the counter and it is clean for the next launch; the store
needs no ordering against the other clusters, which may still be computing,
because none of them will read the word again.

This matters more than it looks.  An earlier revision used the textbook
"second `done` counter, last cluster to arrive resets": four atomics per
cluster at the end of the kernel, on the SAME four addresses across 66
clusters, all issued after the final `cluster_sync()` and therefore squarely
on the critical path to kernel completion.  nsys measured that as a FIXED
4-9 us per launch -- +8% on a 66 us GEMM, +2.8% on a 250 us one -- which was
the entire residual uncontended regression of the dynamic scheduler.

## One counter per (device, stream), and why that is safe unconditionally

A counter may be shared by two launches only if they cannot overlap.
`sched_slot_ptr` keys the counter on the (device, STREAM) the launch is
enqueued on, which makes that unconditional: launches on one stream are
strictly ordered, so launch N's `sched_finish` store of 0 happens before
launch N+1 starts, and no two launches sharing a counter are ever in flight
together.  Nothing about occupancy, tile counts or kernel duration enters
the argument, and a second stream (a side `torch.Stream`, a second process
rank's compute stream) gets its own counter rather than racing for one.

The host table needs no lock for this.  It is append-only: an entry is
claimed with one `fetch_add` on the slot counter (so two threads never write
the same entry) and published with a release store of its `valid` word,
which readers load with acquire before trusting the entry's fields.  Two
threads racing on the FIRST launch for one stream can both miss the scan and
create two counters for it; that is harmless, because the two are
interchangeable -- every launch leaves whichever one it used back at 0, and
same-stream launches do not overlap, so any of them is correct at any time.
The same argument covers the (equally rare) race to install the table itself
in the process-global registry, which mirrors what `_enqueue_cached` already
does for its compiled functions.

Everything here is `M, N, K`-agnostic: the scheduler sees only
`total_works`, which the kernel computes from the runtime shape.
"""

from std.atomic import Atomic, Ordering
from std.ffi import _get_global_or_null, external_call
from std.memory import AddressSpace
from std.memory.alloc import unsafe_alloc
from std.sys import inlined_assembly, is_amd_gpu, is_nvidia_gpu

from max.gpu.host import DeviceBuffer, DeviceContext


# --- layout constants ------------------------------------------------------

# Int32 words per counter: only word 0 is live (the ticket dispenser).  It
# gets a whole 128-byte L2 line to itself so two streams' counters, which ARE
# hit concurrently by 66 clusters each, never share one -- which means the
# base must be ALIGNED to 128 bytes, not merely 128 bytes long: nothing
# promises what alignment `enqueue_create_buffer` hands back, so twice the
# size is allocated and the base is rounded up inside it.
comptime SCHED_WORDS = 32
comptime SCHED_ALLOC_WORDS = 2 * SCHED_WORDS
comptime SCHED_LINE = 4 * SCHED_WORDS

# Ring depth (power of two: the slot index is a mask, never a modulo) and the
# tag period, a MULTIPLE of the depth so one counter `rm` gives both the slot
# (`rm & (SCHED_RING - 1)`) and the tag (`rm + 1`).  That single live UInt32
# is the whole per-round scheduler state the producer warp carries across the
# mainloop, which matters: the producer runs under `warpgroup_reg_dealloc`,
# so anything it keeps live in the TMA-issue loop is charged against that
# budget and spills to local memory beyond it.  An earlier revision carried
# `round_idx: Int` and computed `% 1023` / `% SCHED_RING` with 64-bit signed
# division; it cost up to +16% uncontended (measured: tn_mlp_proj 247 -> 287
# us) purely in producer spill traffic.
comptime SCHED_RING = 8
comptime SCHED_TAG_PERIOD = 1016

# Payload width of a ring word; the remaining 10 bits carry the round tag.
comptime SCHED_W_BITS = 22
comptime SCHED_W_MASK = UInt32((1 << SCHED_W_BITS) - 1)
comptime SCHED_TAG_MASK = ~SCHED_W_MASK
# Margin for the tickets past the end that clusters fetch before exiting (at
# most one per cluster, plus the lookahead): with it, no clamp is needed on
# the fetch path at all, which keeps a 64-bit `min` out of the producer.
comptime SCHED_MAX_WORKS = (1 << SCHED_W_BITS) - 4096

comptime SCHED_PTR = Pointer[Scalar[DType.int32], MutAnyOrigin]
# `stack_allocation` hands back MutUntrackedOrigin; the ring type follows it
# so no origin cast is needed at every call site inside the kernels.
comptime SCHED_SMEM_PTR = Pointer[
    Scalar[DType.uint32],
    MutUntrackedOrigin,
    address_space=AddressSpace.SHARED,
]


@always_inline
def _cluster_remote_smem_addr(local_addr: UInt32, peer_rank: UInt32) -> UInt32:
    """`mapa.shared::cluster.u32`: this CTA's shared address -> the same
    object in CTA `peer_rank`'s window.  Pure address arithmetic, no access.

    Written out rather than imported: MAX 26.5 had no
    `cluster_remote_smem_addr`, and 26.6 has one (identical body) in
    `max.gpu.primitives.cluster` but does not export it from
    `max.gpu.primitives`. Importing it from the module is a follow-up.
    """
    return inlined_assembly[
        "mapa.shared::cluster.u32 $0, $1, $2;",
        UInt32,
        constraints="=r,r,r",
        has_side_effect=False,
    ](local_addr, peer_rank)


@always_inline
def _sched_scope() -> StaticString:
    comptime if is_nvidia_gpu():
        return "device"
    elif is_amd_gpu():
        return "agent"
    else:
        return ""


# --- device side -----------------------------------------------------------


@always_inline
def sched_word(rm: UInt32, w: UInt32) -> UInt32:
    """Ring word for round `rm`: tag `rm + 1` in the top 10 bits, `w + 1`
    below.  One word, so publication is atomic at the hardware's 32-bit
    granularity and no fence separates "data" from "flag"; tag 0 is never
    produced, so a zero-initialised slot is never mistaken for a round."""
    return ((rm + 1) << SCHED_W_BITS) | ((w + 1) & SCHED_W_MASK)


@always_inline
def sched_want(rm: UInt32) -> UInt32:
    return (rm + 1) << SCHED_W_BITS


@always_inline
def sched_advance(rm: UInt32) -> UInt32:
    """Next round counter.  A compare-and-reset, never a modulo."""
    var r = rm + 1
    if r == UInt32(SCHED_TAG_PERIOD):
        r = 0
    return r


@always_inline
def sched_slot(ring: SCHED_SMEM_PTR, rm: UInt32) -> SCHED_SMEM_PTR:
    return ring.unsafe_offset(Int(rm & UInt32(SCHED_RING - 1)))


@always_inline
def sched_store_ring(ptr: SCHED_SMEM_PTR, val: UInt32):
    """Publish one ring word, visible to every CTA of the cluster.

    Cluster-scoped rather than CTA-scoped because rank 0's ring is read by
    its peer's producer through `mapa`; the other ranks' rings have only
    local readers but use the same instruction, which costs the same.
    """
    inlined_assembly[
        "st.relaxed.cluster.shared::cluster.b32 [$0], $1;",
        NoneType,
        constraints="r,r",
        has_side_effect=True,
    ](UInt32(Int(ptr)), val)


@always_inline
def sched_load_peer(ptr: SCHED_SMEM_PTR, peer_rank: UInt32) -> UInt32:
    """Read one ring word from CTA `peer_rank`'s copy of the same object.
    `has_side_effect` keeps the load inside a spin loop."""
    var addr = _cluster_remote_smem_addr(UInt32(Int(ptr)), peer_rank)
    return inlined_assembly[
        "ld.relaxed.cluster.shared::cluster.b32 $0, [$1];",
        UInt32,
        constraints="=r,r",
        has_side_effect=True,
    ](addr)


@always_inline
def sched_load_local(ptr: SCHED_SMEM_PTR) -> UInt32:
    """Read one ring word from THIS CTA's ring: no `mapa`, plain shared."""
    return inlined_assembly[
        "ld.relaxed.cta.shared.b32 $0, [$1];",
        UInt32,
        constraints="=r,r",
        has_side_effect=True,
    ](UInt32(Int(ptr)))


@always_inline
def sched_poll_local(ring: SCHED_SMEM_PTR, rm: UInt32) -> Int:
    """Spin on this CTA's ring until round `rm` is published (consumers)."""
    var slot = sched_slot(ring, rm)
    var want = sched_want(rm)
    var v = sched_load_local(slot)
    while (v & SCHED_TAG_MASK) != want:
        v = sched_load_local(slot)
    return Int(v & SCHED_W_MASK) - 1


@always_inline
def sched_poll_peer(ring: SCHED_SMEM_PTR, rm: UInt32) -> UInt32:
    """Spin on rank 0's ring until round `rm` is published (peer producers).
    Returns the raw payload field, which the caller republishes as-is."""
    var slot = sched_slot(ring, rm)
    var want = sched_want(rm)
    var v = sched_load_peer(slot, UInt32(0))
    while (v & SCHED_TAG_MASK) != want:
        v = sched_load_peer(slot, UInt32(0))
    return (v & SCHED_W_MASK) - 1


@always_inline
def sched_publish(ring: SCHED_SMEM_PTR, rm: UInt32, w: UInt32):
    """Publish an already-fetched ticket for round `rm` in this CTA's ring."""
    sched_store_ring(sched_slot(ring, rm), sched_word(rm, w))


@always_inline
def sched_read_local(ring: SCHED_SMEM_PTR, rm: UInt32) -> Int:
    """Read a round this CTA's own producer has already published: one
    shared load, no spin.  Re-reading beats keeping the value in a register
    across the mainloop, which the producer warp cannot afford."""
    return Int(sched_load_local(sched_slot(ring, rm)) & SCHED_W_MASK) - 1


@always_inline
def sched_fetch_add(ptr: SCHED_PTR, delta: Int32) -> Int32:
    return Atomic[Scalar[DType.int32], scope=_sched_scope()].fetch_add[
        ordering=Ordering.RELAXED
    ](ptr, delta)


@always_inline
def sched_store_global(ptr: SCHED_PTR, val: Int32):
    """Relaxed device-scope store; used only for the end-of-launch reset."""
    inlined_assembly[
        "st.relaxed.gpu.global.b32 [$0], $1;",
        NoneType,
        constraints="l,r",
        has_side_effect=True,
    ](UInt64(Int(ptr)), val)


@always_inline
def sched_init_ring(ring: SCHED_SMEM_PTR):
    """Zero the ring (tag 0 = "nothing published"). One thread per CTA, and
    the caller must `cluster_sync()` -- the ORDERED cluster barrier, not
    `cluster_sync_relaxed` -- before any peer reads it: without the fence the
    peer may still see the pre-launch contents of that shared word, and one
    arbitrary word in 1024 carries round 0's tag."""
    comptime for i in range(SCHED_RING):
        ring[unsafe_offset=i] = UInt32(0)


@always_inline
def sched_finish(
    slot: SCHED_PTR, last_ticket: Int, total_works: Int, num_clusters: Int
):
    """End-of-launch reset, from rank 0's producer thread only.

    `last_ticket` is the past-the-end ticket that stopped this cluster's work
    loop.  Exactly `total_works + num_clusters` tickets are issued per launch,
    so the holder of the highest one is the last fetcher: it resets the
    counter for the next launch, with no `done` counter and no extra atomic.

    The reset is IN BAND, so a launch that never runs to completion -- a
    device-side fault, a context teardown mid-flight -- leaves the counter at
    whatever it had reached, and the next launch on that same stream would
    start mid-count and skip its first tiles.  That is not a case worth
    guarding: a fault has already poisoned the CUDA context and every
    subsequent launch on it fails anyway.  A host-side memset per launch is
    the alternative, and it is exactly the hot-path cost this design exists
    to avoid.
    """
    if last_ticket == total_works + num_clusters - 1:
        sched_store_global(slot, 0)


@always_inline
def sched_publish_round(
    sched: SCHED_PTR, ring: SCHED_SMEM_PTR, rank: Int, rm: UInt32
):
    """Publish round `rm`'s cluster-uniform work index in this CTA's ring.

    Rank 0 dispenses (one relaxed device-scope atomicAdd); every other rank
    waits for rank 0's ring through DSMEM and republishes the same value, so
    the consumers of either CTA only ever read their own CTA's shared memory.
    Returns nothing: the caller reads the value back out of the ring with
    `sched_read_local`, which is one shared load and costs the producer warp
    no live register across the mainloop.
    """
    var w: UInt32
    if rank == 0:
        w = UInt32(sched_fetch_add(sched, 1))
    else:
        w = sched_poll_peer(ring, rm)
    sched_publish(ring, rm, w)


@always_inline
def sched_publish_first(
    ring: SCHED_SMEM_PTR, rank: Int, rm: UInt32, ticket: UInt32
):
    """Round 0, whose ticket rank 0 already fetched in the prologue."""
    var w = ticket
    if rank != 0:
        w = sched_poll_peer(ring, rm)
    sched_publish(ring, rm, w)


# --- host side -------------------------------------------------------------


@always_inline
def sched_supported(total_works: Int) -> Bool:
    """The ring word's 22-bit payload bounds the work census."""
    return total_works <= SCHED_MAX_WORKS


# Process-global table of counters, one per (device, stream).  ONE registry
# lookup per launch under a 13-character literal name -- short enough for
# Mojo's small-string optimisation, so the hot path formats no string and
# allocates nothing.  (An earlier revision built a name from `ctx.id()` twice
# per launch; the enqueue path already spends three `cuTensorMapEncodeTiled`
# driver calls per launch, so host time there is not free.)
#
# Words, all Int64:
#     [0]              slots claimed so far (atomic, `fetch_add` to claim);
#                      also the release/acquire pair that publishes the
#                      zeroed table to a thread that found it in the registry
#     [1]              "already warned about exhaustion" flag
#     [2 + 4 * i + 0]  valid flag, release-stored after the three below
#     [2 + 4 * i + 1]  device id
#     [2 + 4 * i + 2]  context (device, stream) key
#     [2 + 4 * i + 3]  counter pointer
#
# The slot count bounds how many distinct (device, stream) contexts can ever
# take a persistent route in one process.  Entries are reused -- a stream that
# comes back finds its own entry -- but never evicted, because a counter must
# outlive every launch that can still be in flight and the backend's
# `Dev.views` never shrinks either, so a process that creates more than
# `_SCHED_MAX_SLOTS` distinct streams and runs a persistent GEMM on each would
# put every later one on the fallback kernels for good.  512 is far past what
# torch's stream pool hands out; the decline warns once when it happens rather
# than silently getting slower.
comptime _SCHED_REG = "TMB_SCHEDPOOL"
comptime _SCHED_MAX_SLOTS = 512
comptime _SCHED_HEADER = 2
comptime _SCHED_TABLE_WORDS = _SCHED_HEADER + 4 * _SCHED_MAX_SLOTS
comptime _SCHED_TABLE = Pointer[Scalar[DType.int64], MutAnyOrigin]


def _sched_table() raises -> _SCHED_TABLE:
    var cached = _get_global_or_null(String(_SCHED_REG))
    if cached:
        return (
            cached.value()
            .unsafe_bitcast[Scalar[DType.int64]]()
            .as_unsafe_any_origin()
        )
    var table = unsafe_alloc[Scalar[DType.int64]](_SCHED_TABLE_WORDS)
    for i in range(1, _SCHED_TABLE_WORDS):
        table[unsafe_offset=i] = Int64(0)
    # Word 0 last, with RELEASE: every reader's first act is an ACQUIRE load
    # of it (`sched_slot_ptr`), so that pair is what makes the zeroed words
    # above visible to a thread that finds the table through the registry
    # rather than building it.
    Atomic[Scalar[DType.int64]].store[ordering=Ordering.RELEASE](
        table, Int64(0)
    )
    external_call["KGEN_CompilerRT_InsertGlobal", NoneType](
        StringSlice(_SCHED_REG), table.unsafe_bitcast[NoneType]()
    )
    # Re-read rather than returning `table`: if another thread inserted first,
    # ITS table is what every later lookup sees, so use that one and leak this
    # one (the registry has no removal, and this happens at most once per
    # process).  Two tables in flight would only mean a duplicate counter per
    # stream, which the module docstring shows is harmless.
    var winner = _get_global_or_null(String(_SCHED_REG))
    if winner:
        return (
            winner.value()
            .unsafe_bitcast[Scalar[DType.int64]]()
            .as_unsafe_any_origin()
        )
    return table.as_unsafe_any_origin()


def _sched_stream_key(ctx: DeviceContext) raises -> Optional[Int]:
    """Identity of the stream this context submits to, or nothing.

    The key is the context's own C++ handle: the backend keeps one
    DeviceContext per (device, stream) and hands out copies of it
    (`Dev.views`), so the handle is stable across launches on a stream and
    distinct between streams.  It must NOT be `ctx.stream()._handle`:
    `stream()` wraps a fresh DeviceStream object on every call, so that key
    appended one table entry -- a buffer allocation and a memset on the hot
    path -- per launch until the table filled, after which every persistent
    route declined to the fallback kernels for the rest of the process (the
    GPT-2 XL step went from 227 to 460 ms).  A context with no handle returns
    nothing rather than a shared sentinel: two such contexts would be
    indistinguishable here, and a counter shared between two streams that CAN
    overlap is the one thing this key exists to prevent.
    """
    var handle = ctx._handle
    if not handle:
        return None
    return Int(handle.value())


def _sched_warn_exhausted(table: _SCHED_TABLE):
    """Say once, on the process's first exhaustion, why the persistent GEMM
    routes have quietly stopped being chosen.

    Prefix and destination match the backend's only other runtime warning
    (`_warn` in tmb/backend/device.mojo), which prints to stdout; a decline is
    a slowdown, not an error, so it must not raise.
    """
    var flag = table.unsafe_offset(1)
    if (
        Atomic[Scalar[DType.int64]].fetch_add[ordering=Ordering.RELAXED](
            flag, Int64(1)
        )
        != 0
    ):
        return
    print(
        "torch-mojo-backend: more than ",
        _SCHED_MAX_SLOTS,
        " (device, stream) pairs have run a persistent bf16 GEMM; the tile",
        " scheduler's counter table is full, so those routes will decline to",
        " slower kernels from now on. Reuse streams instead of creating new",
        " ones, or raise _SCHED_MAX_SLOTS in gemm16_sched_pool.mojo.",
    )


def sched_slot_ptr(ctx: DeviceContext) raises -> Optional[SCHED_PTR]:
    """The ticket counter launches on this (device, stream) use, or nothing
    when the table is full (the caller then declines and the dispatch
    ladder's next rung serves the shape).

    The counter is allocated on the first such launch and never freed: it must
    outlive every launch that can still be in flight, which for a process-wide
    pool means the process, the same lifetime `_enqueue_cached` gives its
    DeviceFunctions.  See the module docstring for why no lock is needed.
    """
    comptime BufT = DeviceBuffer[DType.int32]
    var table = _sched_table()
    var device = Int(ctx.id())
    var stream = _sched_stream_key(ctx)
    if not stream:
        return None
    var key = stream.value()
    var claimed = Int(
        Atomic[Scalar[DType.int64]].load[ordering=Ordering.ACQUIRE](table)
    )
    var scan = min(claimed, _SCHED_MAX_SLOTS)
    for i in range(scan):
        var entry = table.unsafe_offset(_SCHED_HEADER + 4 * i)
        if (
            Atomic[Scalar[DType.int64]].load[ordering=Ordering.ACQUIRE](entry)
            == 0
        ):
            continue
        if (
            Int(entry[unsafe_offset=1]) == device
            and Int(entry[unsafe_offset=2]) == key
        ):
            return SCHED_PTR(unsafe_from_address=Int(entry[unsafe_offset=3]))
    var idx = Int(
        Atomic[Scalar[DType.int64]].fetch_add[ordering=Ordering.RELAXED](
            table, Int64(1)
        )
    )
    if idx >= _SCHED_MAX_SLOTS:
        _sched_warn_exhausted(table)
        return None
    var buf = ctx.enqueue_create_buffer[DType.int32](SCHED_ALLOC_WORDS)
    # Stream-ordered against the launch that follows on this same context, so
    # no host synchronize is needed to know the kernel sees zeros.
    ctx.enqueue_memset(buf, Int32(0))
    var held = unsafe_alloc[BufT](1)
    held.unsafe_write(buf^)
    # Round the base up to a 128-byte line: the counter word is hit by every
    # cluster of every launch on this stream, and a neighbour's counter in
    # the same line would bounce it.
    var raw = Int(held[].unsafe_ptr())
    var base = SCHED_PTR(
        unsafe_from_address=raw + (SCHED_LINE - raw % SCHED_LINE) % SCHED_LINE
    )
    var entry = table.unsafe_offset(_SCHED_HEADER + 4 * idx)
    entry[unsafe_offset=1] = Int64(device)
    entry[unsafe_offset=2] = Int64(key)
    entry[unsafe_offset=3] = Int64(Int(base))
    Atomic[Scalar[DType.int64]].store[ordering=Ordering.RELEASE](
        entry, Int64(1)
    )
    return base
