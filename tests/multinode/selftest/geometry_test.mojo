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

from collectives_kernels import MAX_WORLD, shard_range, signal_bytes
from internode import CREDIT_AREA_BYTES, CREDIT_SLOT_BYTES, MAX_NODES
from mojoccl import (
    EMPTY_SHARD_BYTES,
    INBOX_SLOTS,
    PIPE_ARENAS,
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
            for li in range(1, MAX_WORLD + 1):
                var lw = li
                var maxchunk = max_chunk_bytes(
                    cap, arena_cap, INBOX_SLOTS, lw, nnodes
                )
                _check(maxchunk % 4096 == 0, "chunk page aligned", bad)
                for wi in range(len(widths)):
                    var item = widths[wi]
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
