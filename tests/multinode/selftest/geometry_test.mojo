# Region-geometry sweep: no GPU, no InfiniBand, no peers. Checks that every
# address a multi-node collective forms stays inside the area it belongs to,
# over the whole configuration matrix this library claims to support --
# region 1 MiB..1 GiB, local_world 1..8, 2..16 nodes, every allreduce dtype
# width, and message sizes from 4 bytes to the chunk cap.
#
# The arithmetic is the library's own (`mojoccl.region_layout` and friends,
# which the communicator calls through one-line wrappers), so this is a check
# of the shipped code and not of a copy of it.
from std.sys import size_of

from rs_fused import reduce_scatter_fused_plan
from rs_multinode import reduce_scatter_nodes_max_count, reduce_scatter_rank_ids

from collectives_kernels import MAX_WORLD, shard_range, signal_bytes
from internode import CREDIT_AREA_BYTES, CREDIT_SLOT_BYTES, MAX_NODES
from mojoccl import (
    EMPTY_SHARD_BYTES,
    INBOX_SLOTS,
    PIPE_ARENAS,
    allgather_mapped_max_bytes,
    allgather_mapped_pipeline_plan,
    inbox_group_bytes,
    max_chunk_bytes,
    net_stage_bytes,
    PIPE_SPLIT_UNIT,
    pipeline_chunk_bytes,
    region_layout,
)


def _align_up(x: Int, a: Int) -> Int:
    return (x + a - 1) // a * a


def _check(ok: Bool, what: String, mut bad: Int):
    if not ok:
        bad += 1
        if bad < 20:
            print("FAIL:", what)


def main() raises:
    var caps = List[Int]()
    var mb = 1
    while mb <= 1024:
        caps.append(mb * 1024 * 1024)
        mb *= 2
    # A few non-power-of-two region sizes too: MOJOCCL_REGION_MB takes any
    # 4 KiB multiple.
    caps.append(3 * 1024 * 1024)
    caps.append(48 * 1024 * 1024)
    caps.append(200 * 1024 * 1024)
    caps.append(768 * 1024 * 1024)

    var widths = List[Int]()
    widths.append(1)
    widths.append(2)
    widths.append(4)
    widths.append(8)

    var bad = 0
    var cases = 0
    # A nontrivial topology catches large-tuple runtime indexing regressions.
    for nnodes in range(1, MAX_NODES + 1):
        for lw in range(1, MAX_WORLD + 1):
            var topo = List[Int]()
            for node in range(nnodes):
                for local in range(lw):
                    topo.append(local * nnodes + node)
            var ids = reduce_scatter_rank_ids(topo)
            for i in range(nnodes * lw):
                _check(Int(ids[i]) == topo[i], "reduce-scatter rank map", bad)
            cases += 1
    for ci in range(len(caps)):
        var cap = caps[ci]
        for nnodes in range(1, MAX_NODES + 1):
            var lay = region_layout(cap, nnodes)
            var narenas = lay[0]
            var arena_cap = lay[1]
            var stride = lay[2]
            var region_bytes = lay[3]
            var net_off = narenas * stride
            _check(
                narenas == (PIPE_ARENAS if nnodes > 1 else 1),
                "arena count",
                bad,
            )
            _check(
                arena_cap % 4096 == 0 and arena_cap > 0, "arena_cap page", bad
            )
            _check(
                stride == signal_bytes() + 2 * arena_cap, "arena stride", bad
            )
            _check(
                region_bytes == net_off + (cap if nnodes > 1 else 0),
                "region size",
                bad,
            )
            # The staging total must not grow: that is what keeps the region
            # the size it was before the pipeline.
            _check(narenas * 2 * arena_cap <= 2 * cap, "staging total", bad)
            if nnodes == 1:
                _check(arena_cap == cap, "single node keeps the old cap", bad)
                _check(
                    region_bytes == signal_bytes() + 2 * cap,
                    "single node region",
                    bad,
                )
                continue

            var group = inbox_group_bytes(cap, INBOX_SLOTS)
            _check(group > 0, "inbox group nonempty", bad)
            _check(INBOX_SLOTS * group <= cap // 2, "inbox groups fit", bad)
            _check(
                CREDIT_AREA_BYTES + net_stage_bytes(cap) <= cap // 2,
                "credits + staging fit the first half",
                bad,
            )
            _check(
                (MAX_NODES - 1) * CREDIT_SLOT_BYTES <= CREDIT_AREA_BYTES,
                "credit slots fit the credit area",
                bad,
            )
            var npeers = nnodes - 1
            var ag_bytes = allgather_mapped_max_bytes(arena_cap, group, npeers)
            _check(
                ag_bytes > 0 and ag_bytes % 16 == 0, "mapped gather chunk", bad
            )
            _check(ag_bytes <= arena_cap, "mapped gather stage fits", bad)
            _check(npeers * ag_bytes <= group, "mapped gather inbox fits", bad)
            _check(
                ag_bytes + 16 > arena_cap or npeers * (ag_bytes + 16) > group,
                "mapped gather maximal chunk",
                bad,
            )
            for tail in range(1, 32):
                var chunk = min(tail, ag_bytes)
                _check(
                    npeers * _align_up(chunk, 16) <= group,
                    "mapped gather tail inbox fits",
                    bad,
                )
            cases += 1
            for li in range(1, MAX_WORLD + 1):
                var lw = li
                var ag_counts = List[Int]()
                ag_counts.append(1)
                ag_counts.append(16)
                ag_counts.append(12345)
                ag_counts.append(3_840_000)
                ag_counts.append(20_512_400)
                ag_counts.append(357 * 789 * 4)
                ag_counts.append(ag_bytes - 1)
                ag_counts.append(ag_bytes)
                ag_counts.append(ag_bytes + 1)
                ag_counts.append(ag_bytes * 600 + 1)
                for ai in range(len(ag_counts)):
                    var count = ag_counts[ai]
                    var plan = allgather_mapped_pipeline_plan(
                        count,
                        ag_bytes,
                        narenas,
                        INBOX_SLOTS,
                        PIPE_SPLIT_UNIT * lw,
                    )
                    var chunk = plan[0]
                    var nchunks = plan[1]
                    var depth = plan[2]
                    _check(
                        chunk <= ag_bytes and chunk % 16 == 0,
                        "gather pipeline capacity/alignment",
                        bad,
                    )
                    _check(
                        (nchunks - 1) * chunk < count <= nchunks * chunk,
                        "gather pipeline coverage",
                        bad,
                    )
                    _check(
                        1 <= depth <= 2
                        and depth <= narenas
                        and depth <= nchunks
                        and depth < INBOX_SLOTS,
                        "gather pipeline depth/credit window",
                        bad,
                    )
                    if count < PIPE_SPLIT_UNIT * lw and count <= ag_bytes:
                        _check(nchunks == 1, "gather small path unchanged", bad)
                    var completed = List[Int](length=depth, fill=-1)
                    var active = List[Int](length=depth, fill=-1)
                    for k in range(nchunks + depth - 1):
                        if k < nchunks:
                            var arena = k % depth
                            _check(
                                k < depth or completed[arena] == k - depth,
                                "gather arena consumed before reuse",
                                bad,
                            )
                            active[arena] = k
                            var size = min(chunk, count - k * chunk)
                            _check(
                                size <= arena_cap
                                and npeers * _align_up(size, 16) <= group,
                                "gather pipeline tail capacity",
                                bad,
                            )
                        var j = k - (depth - 1)
                        if j >= 0:
                            _check(
                                active[j % depth] == j,
                                "gather consumes matching arena",
                                bad,
                            )
                            completed[j % depth] = j
                    cases += 1
                var maxchunk = max_chunk_bytes(
                    cap, arena_cap, INBOX_SLOTS, lw, nnodes
                )
                _check(maxchunk % 4096 == 0, "chunk page aligned", bad)
                for wi in range(len(widths)):
                    var item = widths[wi]
                    var rs_count = reduce_scatter_nodes_max_count(
                        arena_cap, lw, item, nnodes
                    )
                    var rs_slot = _align_up(rs_count * item, 16)
                    _check(rs_count > 0, "reduce-scatter nonempty chunk", bad)
                    var rs_output = (lw - 1) * nnodes * rs_slot
                    _check(
                        rs_output + nnodes * rs_slot <= 2 * arena_cap,
                        "reduce-scatter pushes + node outputs fit arena",
                        bad,
                    )
                    var next_slot = _align_up((rs_count + 1) * item, 16)
                    _check(
                        nnodes * lw * next_slot > 2 * arena_cap,
                        "reduce-scatter maximal chunk",
                        bad,
                    )
                    for d in range(nnodes):
                        _check(
                            rs_output + d * rs_slot + rs_count * item
                            <= 2 * arena_cap,
                            "reduce-scatter node output address",
                            bad,
                        )
                    # Final chunks may use a different aligned slot stride.
                    for count in range(1, min(rs_count, 37) + 1):
                        var slot = _align_up(count * item, 16)
                        var output = (lw - 1) * nnodes * slot
                        _check(
                            output + nnodes * slot <= 2 * arena_cap,
                            "reduce-scatter short chunk fits arena",
                            bad,
                        )
                        _check(
                            output % 16 == 0,
                            "reduce-scatter packed output aligned",
                            bad,
                        )
                    if item == 4:
                        var counts = List[Int]()
                        counts.append(1)
                        counts.append(12345)
                        counts.append(32768)
                        counts.append(PIPE_SPLIT_UNIT // item - 1)
                        counts.append(PIPE_SPLIT_UNIT // item)
                        counts.append(rs_count * 7 + 123)
                        for pi in range(len(counts)):
                            var count = counts[pi]
                            var plan = reduce_scatter_fused_plan(
                                count,
                                rs_count,
                                narenas,
                                PIPE_SPLIT_UNIT,
                            )
                            _check(
                                plan[0] <= rs_count,
                                "fused reduce-scatter chunk fits packed arena",
                                bad,
                            )
                            _check(
                                plan[0] * plan[1] >= count
                                and plan[0] * (plan[1] - 1) < count,
                                "fused reduce-scatter covers input exactly",
                                bad,
                            )
                            _check(
                                plan[1] == 1 or plan[0] * item % 16 == 0,
                                "fused reduce-scatter chunk offsets aligned",
                                bad,
                            )
                            _check(
                                plan[2] <= narenas and plan[2] <= plan[1],
                                "fused reduce-scatter arena reuse bounded",
                                bad,
                            )
                            cases += 1
                    cases += 1
                    # Message sizes: tiny, ragged, and right at the cap.
                    var sizes = List[Int]()
                    sizes.append(item)
                    sizes.append(3 * item)
                    sizes.append(1003 * item)
                    sizes.append(maxchunk // 2 + item)
                    sizes.append(maxchunk - item)
                    sizes.append(maxchunk)
                    sizes.append(4 * maxchunk + 7 * item)
                    for si in range(len(sizes)):
                        var total = sizes[si]
                        var chunk_bytes = pipeline_chunk_bytes(
                            maxchunk, lw, total, PIPE_SPLIT_UNIT
                        )
                        _check(chunk_bytes <= maxchunk, "chunk <= max", bad)
                        _check(
                            chunk_bytes <= arena_cap, "chunk fits an arena", bad
                        )
                        _check(
                            chunk_bytes % 4096 == 0, "chunk page aligned", bad
                        )
                        var chunk_elems = max(1, chunk_bytes // item)
                        var count = max(1, total // item)
                        var nchunks = (count + chunk_elems - 1) // chunk_elems
                        _check(nchunks >= 1, "at least one chunk", bad)
                        for k in range(nchunks):
                            var off = k * chunk_elems
                            var cnt = min(chunk_elems, count - off)
                            # Every chunk offset must stay 16-byte aligned:
                            # the split kernels use 16-byte vectors.
                            _check(
                                (off * item) % 16 == 0,
                                "chunk offset align",
                                bad,
                            )
                            _check(
                                cnt * item <= arena_cap,
                                "chunk elems fit the arena cap",
                                bad,
                            )
                            for r in range(lw):
                                var sr = shard_range(cnt, lw, r, item)
                                var nbytes = sr[1] * item
                                var slot = _align_up(
                                    max(nbytes, EMPTY_SHARD_BYTES), 16
                                )
                                _check(
                                    npeers * slot <= group,
                                    "inbox slots fit the group",
                                    bad,
                                )
                                # The shard lives in stage_out at its own
                                # element offset; it must stay inside it.
                                _check(
                                    sr[0] * item + nbytes <= arena_cap,
                                    "shard inside stage_out",
                                    bad,
                                )
                                # Compacted push slots at the base of
                                # stage_in (RESULTS.md 10.1).
                                var per = shard_range(cnt, lw, 0, item)[1]
                                _check(
                                    (lw - 1) * per * item <= arena_cap,
                                    "push slots inside stage_in",
                                    bad,
                                )
                            cases += 1
    print("geometry cases", cases, "failures", bad)
    print("PASS" if bad == 0 else "FAIL")
