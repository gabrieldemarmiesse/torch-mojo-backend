"""The streaming reduce-scatter's arena layout may not depend on the chunk.

`reduce_scatter_gin_stream.mojo` hands a chunk's staging slots to its peers
under a credit that is per BLOCK INDEX: before writing the arena at chunk k a
block waits only for the SAME block index on every peer to have released chunk
`k - narenas`. That is sound exactly while block b owns the same bytes of the
arena in both chunks. Deriving the slot stride or the per-block partition
from a short final chunk's element count breaks it -- block b's write then
lands on bytes block b-1 of a peer is still reducing -- and the failure is a
race that a 16-rank run reproduces only sometimes. So the invariant is
pinned here, in source, where the check is deterministic and needs no GPU.
"""

import re
from pathlib import Path

MOJOCCL = Path(__file__).resolve().parents[1] / "torch_mojo_backend/mojo/tmb/ccl"
STREAM = (MOJOCCL / "device/symmetric/reduce_scatter_gin_stream.mojo").read_text()
ENQUEUE = (MOJOCCL / "enqueue.mojo").read_text()
INIT = (MOJOCCL / "init.mojo").read_text()
COMM = (MOJOCCL / "include/comm.mojo").read_text()


def _body(source: str, start: str, end: str) -> str:
    begin = source.index(start)
    return source[begin : source.index(end, begin)]


def _constant(source: str, name: str) -> int:
    """The NVIDIA value of a `comptime NAME = <amd> if _MI300A else <nvidia>`."""
    found = re.search(rf"^comptime {name} = (.+)$", source, re.M)
    assert found is not None, f"{name} is no longer declared where this reads it"
    return int(found.group(1).rsplit(" ", 1)[-1])


def test_streaming_kernel_sizes_its_slots_from_the_full_chunk():
    kernel = _body(STREAM, "def _stream_rs_kernel[", "\ndef _resident[")
    assert "_align_up(ce * 4, 16)" in kernel, (
        "the staging slot stride must come from chunk_elems, so that a short"
        " final chunk reuses an arena without re-cutting it"
    )
    assert "_align_up(cnt * 4" not in kernel, (
        "a slot stride derived from this chunk's own count moves every slot"
        " base when the last chunk is short; the per-block FREE credit does"
        " not cover that"
    )


def test_streaming_kernel_partitions_by_the_full_chunk():
    kernel = _body(STREAM, "def _stream_rs_kernel[", "\ndef _resident[")
    assert "(vc_full + nblocks - 1) // nblocks" in kernel
    assert "(vc + nblocks - 1) // nblocks" not in kernel, (
        "a partition derived from this chunk's own vector count hands block b"
        " bytes that block b-1 owned in the chunk sharing its arena"
    )


def test_host_launcher_uses_one_slot_stride_for_the_whole_call():
    launcher = _body(
        ENQUEUE, "def _do_reduce_scatter_stream(", "\ndef _do_reduce_scatter_nodes["
    )
    assert "var slot = _align_up(chunk_elems * 4, 16)" in launcher
    loop = _body(launcher, "for k in range(nchunks):", "    try:")
    assert "slot = " not in loop, (
        "the RDMA source offset and the inbox stride are the kernel's slot"
        " stride; both sides must use the one the whole call uses"
    )


def test_a_short_final_chunk_reuses_an_arena_at_reachable_sizes():
    """The geometry that broke is an ordinary FSDP2-sized call, not a corner."""
    region = _constant(INIT, "DEFAULT_REGION_MB") * 1024 * 1024
    narenas = _constant(COMM, "PIPE_ARENAS")
    threads = 512  # RS_STREAM_THREADS, the fused kernel's
    blocks = _constant(STREAM, "RS_STREAM_BIG_BLOCKS")
    arena_cap = region // narenas // 4096 * 4096
    # `reduce_scatter_nodes_max_count` for 8 local ranks on 2 nodes, fp32.
    chunk = (2 * arena_cap // (2 * 8) // 16 * 16) // 4
    count = 4 * chunk + chunk // 2  # 36 MiB per rank at 16 ranks
    nchunks = -(-count // chunk)
    assert nchunks == 5 and nchunks > narenas, (nchunks, narenas)
    last = count - (nchunks - 1) * chunk
    assert last * 2 == chunk, last
    # The grid cap binds, so both chunks are cut into `blocks` ranges.
    assert blocks <= -(-(chunk // 4) // threads)
    full_vpb = -(-(chunk // 4) // blocks)
    short_vpb = -(-(last // 4) // blocks)
    vectors = last // 4

    def cut(vpb: int, block: int, vc: int) -> tuple[int, int]:
        return min(vc, block * vpb), min(vc, (block + 1) * vpb)

    # Cutting the short chunk on its own count starts block 1 inside the range
    # block 0 owns in the chunk it shares an arena with, and block 1 waits
    # only for block 1 of its peers.
    assert cut(short_vpb, 1, vectors)[0] < cut(full_vpb, 0, chunk // 4)[1]
    # Cutting it on the full chunk's count does not: every block keeps its
    # start and a short chunk only leaves the tail of the layout unwritten.
    assert cut(full_vpb, 1, vectors)[0] == cut(full_vpb, 1, chunk // 4)[0]
    assert cut(full_vpb, blocks - 1, vectors) == (vectors, vectors)
    # The slot stride would move as well, taking every slot base with it.
    assert -(-last * 4 // 16) * 16 != -(-chunk * 4 // 16) * 16
