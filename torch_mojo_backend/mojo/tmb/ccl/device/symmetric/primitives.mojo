# Rewrite of: https://github.com/NVIDIA/nccl/blob/master/src/device/symmetric/primitives.cuh

from max.gpu import block_idx
from std.collections import Array
from std.utils import StaticTuple

from tmb.ccl.include.device import MAX_WORLD


@always_inline
def _peer_step(i: Int, world: Int) -> Int:
    """Peer offset for loop index `i` (1 <= i < world), rotated by block index.

    Every block still touches exactly the same bytes for every peer -- only
    the order of the peers differs -- so the block-matched sync invariant is
    untouched. What changes is link usage on a point-to-point topology such
    as the MI300A's xGMI mesh (one link per GPU pair): with the plain order
    the whole grid queues on one peer's link at a time and the other
    `world-2` links sit idle; rotated, the blocks spread over all of them at
    once. Measured on 4x MI300A: see agents_docs/distributed.md, "Cluster notes
    (AMD MI300A)". Behind NVSwitch it is not irrelevant either: with every
    block of a rank storing into the same peer at once the 2x8 H100 fused
    reduce-scatter's push ran at 291 GB/s per GPU, rotated 326 (block fp32
    isolated 632 -> 618 us), so both vendors rotate.
    """
    return 1 + (i - 1 + Int(block_idx.x)) % (world - 1)


@always_inline
def _peer_step0(i: Int, world: Int) -> Int:
    """`_peer_step` for loops that include the rank itself (0 <= i < world)."""
    return (i + Int(block_idx.x)) % world


@always_inline
def _gather_slot(writer: Int, owner: Int) -> Int:
    """Index of `writer`'s slot inside `owner`'s gather area (AMD phase 3').

    Compacted, exactly like the split allreduce's push slots: `owner` never
    writes its own slot (its own reduced shard goes straight to the user
    output), so `world-1` slots suffice and the area stays inside the arena
    even when the message is exactly `cap` bytes.
    """
    return writer if writer < owner else writer - 1


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


@always_inline
def _region_ptrs(
    regions: StaticTuple[Int, MAX_WORLD], rank: Int, world: Int
) -> Array[Pointer[UInt8, MutAnyOrigin], MAX_WORLD]:
    """Peer region bases as mapped in this process. Unused slots are filled
    with my own region so no kernel can ever hold a null pointer."""
    var out = Array[Pointer[UInt8, MutAnyOrigin], MAX_WORLD](uninitialized=True)
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
