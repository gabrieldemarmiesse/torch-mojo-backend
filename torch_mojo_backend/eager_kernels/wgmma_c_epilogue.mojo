"""The shared C-tile epilogue of the 128-thread wgmma bodies.

One 128-thread warp group holding a 64 x `bn` fp32 accumulator writes it out
in exactly one of two ways, and both are pure functions of the accumulator
fragment layout, so every kernel with that body shape wants the same code:

  * `tma_store=True` — stage the tile into the 128 B-swizzled shared buffer
    the TMA store expects, then `async_store_3d` it 64 columns at a time.
    Needs a 16 B aligned output row pitch (the descriptor requirement), and
    TMA clips the ragged edges against the descriptor's own extents.
  * `tma_store=False` — the scalar fallback for an output row pitch a TMA
    store descriptor cannot express: 4 B pairs straight to global, with the
    row and column bounds tested in software.

The accumulator element -> (row, col) mapping is the wgmma m64nNk16 fragment
layout: lane l of warp w owns rows `w*16 + l/4` (+8) and column pairs
`(l%4)*2 + 8*q`.  It is the one piece of this that is easy to get subtly
wrong, which is why it lives in one place instead of once per kernel.

The tile is addressed as one item of a rank-3 (batch, m, n) output:
`out[bidx, m0 + row, n0 + col]`, with `c_bs` the element stride between
batch items.  A conv fprop whose GEMM computes (out_c x pixels) per sample
is exactly that with `m = out_c`, `n = OH*OW`, `c_bs = out_c*OH*OW`, which
is why the implicit-GEMM conv kernel shares this epilogue verbatim rather
than carrying a second copy.
"""

from std.gpu import thread_idx
from max.gpu.memory import fence_async_view_proxy
from max.gpu.sync import barrier
from std.memory import AddressSpace
from std.utils.index import Index

from layout import Layout, LayoutTensor
from layout.tma_async import TMATensorTile


@always_inline
def _wgmma_store_c_tile[
    dtype: DType,
    bm: Int,
    bn: Int,
    tma_store: Bool,
](
    c_tma: TMATensorTile[dtype, 3, Index(1, bm, 64), Index(1, bm, 64)],
    c_smem: UnsafePointer[
        Scalar[dtype], MutAnyOrigin, address_space=AddressSpace.SHARED
    ],
    accum: UnsafePointer[
        Scalar[DType.float32], MutAnyOrigin, address_space=AddressSpace.LOCAL
    ],
    output: UnsafePointer[Scalar[dtype], MutAnyOrigin],
    m: Int,
    n: Int,
    c_bs: Int,
    m0: Int,
    n0: Int,
    bidx: Int,
):
    comptime assert bn % 64 == 0, "the C tile is stored 64 columns at a time"
    comptime CFRAG = 64 * bn // 128
    var tid = Int(thread_idx.x)
    var warp = tid // 32
    var lane = tid % 32
    var base_row = warp * 16 + lane // 4
    var base_col = (lane % 4) * 2
    comptime if tma_store:
        comptime for q in range(CFRAG // 2):
            var e = q * 2
            var row = base_row + (q % 2) * 8
            var col = base_col + (q // 2) * 8
            var pair = SIMD[dtype, 2](
                accum[e].cast[dtype](),
                accum[e + 1].cast[dtype](),
            )
            var lcol = col % 64
            var elem = (
                (col // 64) * (bm * 64)
                + row * 64
                + ((lcol // 8) ^ (row % 8)) * 8
                + lcol % 8
            )
            c_smem.store[alignment=4](elem, pair)
        fence_async_view_proxy()
        barrier()
        if thread_idx.x == 0:
            comptime for chunk in range(bn // 64):
                var c_chunk = LayoutTensor[
                    dtype,
                    Layout.row_major(bm, 64),
                    MutAnyOrigin,
                    address_space=AddressSpace.SHARED,
                    alignment=128,
                ](c_smem + chunk * bm * 64)
                c_tma.async_store_3d(c_chunk, (n0 + chunk * 64, m0, bidx))
            c_tma.commit_group()
            c_tma.wait_group[0]()
    else:
        comptime for q in range(CFRAG // 2):
            var e = q * 2
            var row = base_row + (q % 2) * 8
            var col = base_col + (q // 2) * 8
            if m0 + row < m:
                var off = bidx * c_bs + (m0 + row) * n + n0 + col
                if n0 + col + 1 < n:
                    output.store[alignment=2](
                        off,
                        SIMD[dtype, 2](
                            accum[e].cast[dtype](),
                            accum[e + 1].cast[dtype](),
                        ),
                    )
                elif n0 + col < n:
                    output[off] = accum[e].cast[dtype]()
