"""Candidate H100 BF16 GEMM/BMM built on mma.sync m16n8k16 tensor cores.

Four shared-memory tile regimes (128x128x32, 128x64x32, 64x128x32, 64x64x32)
serve every runtime shape, layout, and batch; the host picks per launch from
runtime dims and the SM count only (narrow tiles when one extent is <= 64,
64x64 when the full tile underfills the GPU but a 64x64 grid still fits one
wave, wide otherwise).
Eight warps own 32-row m16n8k16 fragment grids with FP32 accumulators that
live across the entire K loop; the only BF16 rounding is the single final
store (plus optional FP32 bias add before it). A two-stage pipeline
prefetches the next K tile into registers while the current shared tile is
consumed, with one barrier per K tile.

Each regime compiles twice. FASTK builds carry a host-side proof of 16B
staging alignment (contiguous-dim % 8, base % 16, batch strides % 8, plus
4B-aligned pair stores) for every batch: guarded branches compile out,
m/n-contiguous operands stage k-major so all staging keeps 16B vector
stores, and Int32 tile-index math keeps the footprint at 126-128 registers
so two-plus 256-thread blocks co-reside per SM. Guarded builds keep the v1
dim-major layouts and per-batch alignment re-proofs (16B fast loads, then
element-aligned 8-wide loads, then per-element guarded loads); their larger
register budget buys ILP for sub-one-wave grids, so the host routes grids
of at most 132 blocks (one per H100 SM) and any unproved launch to them.
All shared pitches stay 16 bytes mod 128 for conflict-free 16B staging.
FASTK BMM maps one batch per grid.z block; batch counts above 65,535
grid-stride through the guarded kernels with a barrier and accumulator
reset between batches. Tail loads zero-fill and every store is bounds
guarded, so batch padding and harness guard cells are never touched.

The host admits these Int32 kernels only after a machine-width proof that
no narrowed value can wrap: positive dims, m <= 2**31-1 - (BM-1),
n <= 2**31-1 - (BN-1), and k <= 2**31-1 - 31 for the selected tile, a
guarded block-count product bounding grid.x by 2**31-1, and for BMM
batch_count <= 2**31 - grid_z plus a proof that every last-batch base and
matrix-local offset fits machine Int. Any launch outside that proof takes the
*_wide kernels, mechanically adapted
from the accepted v1 source: every dim, tile origin, K index, batch
index, and flat offset stays machine Int, a logical linear tile index
grid-strides past the 2_147_483_647 physical grid.x cap, grid.z
grid-strides batches, both loops advance via remaining-distance tests
that cannot wrap, and the host raises before enqueue if any launch or
addressing product cannot fit in machine Int.
"""

from std.collections import InlineArray
from max.gpu.sync import barrier
from std.gpu import block_idx, grid_dim, thread_idx
from max.gpu.compute.mma import mma
from max.gpu.host import DeviceAttribute, DeviceContext
from std.memory import AddressSpace
from std.memory import stack_allocation

comptime _BM = 128
comptime _BN = 128
comptime _BK = 32
comptime _LDS = _BK + 8
comptime _STAGE_A = _BM * _LDS
comptime _STAGE_B = _BN * _LDS
comptime _THREADS = 256
comptime _GROUP_M = 8
comptime _BF16 = DType.bfloat16
comptime _F32 = DType.float32
comptime _Ptr = UnsafePointer[Scalar[_BF16], MutAnyOrigin]
comptime _F32Ptr = UnsafePointer[Scalar[_F32], MutAnyOrigin]
comptime _I32_MAX = 2_147_483_647
comptime _I64_MAX = 9_223_372_036_854_775_807


@always_inline
def _g2r_kc[
    CH: Int, FAST: Bool
](
    src: _Ptr,
    row0: Int32,
    rows: Int32,
    k0: Int32,
    kdim: Int32,
    tid: Int32,
    fast: Int32,
    mut regs: InlineArray[SIMD[_BF16, 8], CH],
):
    # K-contiguous operand: element (r, kk) lives at src[r * kdim + kk].
    # Quad q takes row (4q % 64) + q // 16 so the paired quads of one shared
    # 16B-store phase land 4 rows (320B = 64 mod 128B) apart: conflict-free.
    # Index math stays in Int32 (dims are < 2^31) so loop-carried values take
    # one register, not a 64-bit pair; flat offsets widen to Int at the use.
    @parameter
    for it in range(CH):
        var q = tid // 4
        var r = (q * 4) % 64 + q // 16 + Int32(it * 64)
        var kc = (tid % 4) * 8
        var gr = row0 + r
        var gk = k0 + kc
        var v = SIMD[_BF16, 8]()

        @parameter
        if FAST:
            # FAST proves 16B base alignment and kdim % 8 == 0 for every
            # batch, so each 8-wide chunk is aligned and entirely in or out
            # of bounds.
            if gr < rows and gk < kdim:
                v = src.load[width=8, alignment=16](
                    Int(gr) * Int(kdim) + Int(gk)
                )
        else:
            if fast != 0:
                # Per-batch proof of the same property.
                if gr < rows and gk < kdim:
                    v = src.load[width=8, alignment=16](
                        Int(gr) * Int(kdim) + Int(gk)
                    )
            else:
                if gr < rows and gk < kdim:
                    if gk + 8 <= kdim:
                        # Whole chunk in bounds; 2B element alignment only.
                        v = src.load[width=8, alignment=2](
                            Int(gr) * Int(kdim) + Int(gk)
                        )
                    else:
                        var flat = Int(gr) * Int(kdim) + Int(gk)

                        @parameter
                        for e in range(8):
                            if gk + Int32(e) < kdim:
                                v[e] = src[flat + e]
        regs[it] = v


@always_inline
def _g2r_mc[
    CH: Int, FAST: Bool, QUAD: Bool = False
](
    src: _Ptr,
    row0: Int32,
    rows: Int32,
    k0: Int32,
    kdim: Int32,
    tid: Int32,
    fast: Int32,
    mut regs: InlineArray[SIMD[_BF16, 8], CH],
):
    # Row-contiguous operand: element (r, kk) lives at src[kk * rows + r].
    #
    # QUAD assigns the four lanes of a quad to the four consecutive 8-element
    # chunks of one k-row (32 contiguous elements, 64 bytes) instead of
    # giving each lane its own k-row.  The guarded loads of an awkward shape
    # scalarize (2B element alignment only), and under the default mapping
    # every lane then owns a private 32B sector (~3 of 32 bytes used); the
    # quad mapping packs four lanes into each pair of sectors.  Used by the
    # small-tile guarded builds together with the k-major staging stores in
    # store_tile, whose addressing must match this mapping exactly.
    @parameter
    for it in range(CH):
        var kr: Int32
        var rc: Int32

        @parameter
        if QUAD:
            kr = (tid // 4) % Int32(_BK)
            rc = (tid % 4) * 8 + (tid // 128) * 32 + Int32(it * 64)
        else:
            var item = tid + Int32(it * _THREADS)
            kr = item % _BK
            rc = (item // _BK) * 8
        var gk = k0 + kr
        var gr = row0 + rc
        var v = SIMD[_BF16, 8]()

        @parameter
        if FAST:
            # FAST proves 16B base alignment and rows % 8 == 0 for every
            # batch.
            if gk < kdim and gr < rows:
                v = src.load[width=8, alignment=16](
                    Int(gk) * Int(rows) + Int(gr)
                )
        else:
            if fast != 0:
                # Per-batch proof of the same property.
                if gk < kdim and gr < rows:
                    v = src.load[width=8, alignment=16](
                        Int(gk) * Int(rows) + Int(gr)
                    )
            else:
                if gk < kdim and gr < rows:
                    if gr + 8 <= rows:
                        # Whole chunk in bounds; 2B element alignment only.
                        v = src.load[width=8, alignment=2](
                            Int(gk) * Int(rows) + Int(gr)
                        )
                    else:
                        var flat = Int(gk) * Int(rows) + Int(gr)

                        @parameter
                        for e in range(8):
                            if gr + Int32(e) < rows:
                                v[e] = src[flat + e]
        regs[it] = v


@always_inline
def _mma_tile_impl[
    TA: Bool,
    TB: Bool,
    BM: Int,
    BN: Int,
    FASTK: Bool,
    BATCHED: Bool,
    SPLITK: Bool = False,
](
    output: _Ptr,
    a: _Ptr,
    b: _Ptr,
    bias: _Ptr,
    m: Int,
    n: Int,
    k: Int,
    c_bstride: Int,
    a_bstride: Int,
    b_bstride: Int,
    has_bias: Int,
    a_fast: Int,
    b_fast: Int,
    c_pair: Int,
    batch_count: Int,
    ws: _F32Ptr,
    chunk_tiles: Int,
    ws_pitch: Int,
):
    # SPLITK builds tile the K loop over grid.y: block (x, s) computes the
    # partial product of its output tile over K tiles [s * chunk_tiles,
    # (s + 1) * chunk_tiles) and stores FP32 partials to workspace slice s
    # (ws + s * ws_pitch, one m*n image per slice, pitch rounded so every
    # slice keeps 16B alignment for the reduce).  bias / c_pair / batching
    # are not part of this mode: the reduce kernel owns the bias add and the
    # bf16 store, and the host only routes non-batched GEMM launches here.
    comptime assert not (SPLITK and BATCHED), "split-K GEMM is not batched"
    comptime assert not (SPLITK and FASTK), "split-K uses the guarded loads"
    # Eight warps arrange as (BM/32) x (8/(BM/32)); each owns a 32-row by
    # 8*NT-column fragment grid. ACH/BCH: 8-element staging chunks per thread.
    comptime WARPS_M = BM // 32
    comptime WARPS_N = 8 // WARPS_M
    comptime NT = BN // (WARPS_N * 8)
    comptime ACH = BM // 64
    comptime BCH = BN // 64
    # An operand staged from its m/n-contiguous form lives k-major in shared
    # ([_BK][dim + 8]) so staging keeps 16B vector stores; a k-contiguous
    # operand stays dim-major ([dim][_LDS]). Both pitches are 16 bytes mod
    # 128, so 16B staging stores and fragment gathers are conflict-free.
    comptime LDA_K = BM + 8
    comptime LDB_K = BN + 8
    # Stage strides use the dim-major footprint for both layouts; the k-major
    # image (_BK * (dim + 8)) is strictly smaller, so it always fits.
    comptime STAGE_A = BM * _LDS
    comptime STAGE_B = BN * _LDS

    # Tile-local index math runs in Int32 (each dim is < 2^31) so loop-carried
    # values cost one register instead of a 64-bit pair; flat global offsets
    # widen back to Int at every pointer use.
    var tid = Int32(Int(thread_idx.x))
    var lane = tid % 32
    var warp = tid // 32
    var g = lane // 4
    var tg = lane % 4
    var wm = (warp % Int32(WARPS_M)) * 32
    var wn = (warp // Int32(WARPS_M)) * Int32(NT * 8)

    var mi = Int32(m)
    var ni = Int32(n)
    var ki = Int32(k)

    # Grouped block ordering: sweep _GROUP_M M-blocks per N column so the
    # active A rows stay resident in L2 while B streams.
    var blocks_m = (mi + Int32(BM - 1)) // Int32(BM)
    var blocks_n = (ni + Int32(BN - 1)) // Int32(BN)
    var lin = Int32(Int(block_idx.x))
    var group_span = Int32(_GROUP_M) * blocks_n
    var gid = lin // group_span
    var rem = lin % group_span
    var rows_in_group = min(Int32(_GROUP_M), blocks_m - gid * Int32(_GROUP_M))
    var bm0 = (gid * Int32(_GROUP_M) + rem % rows_in_group) * Int32(BM)
    var bn0 = (rem // rows_in_group) * Int32(BN)

    # grid.z is capped at 65,535, so each block grid-strides over batches.
    var gdz = Int32(Int(grid_dim.z))
    var bz = Int32(Int(block_idx.z))
    var ap = a
    var bp = b
    var cp = output
    # FASTK instantiations carry a host-side proof (divisibility, base and
    # batch-stride alignment) covering every batch; guarded instantiations
    # re-prove base alignment per batch so individually aligned batches keep
    # the vector paths even when the batch stride breaks alignment.
    var af: Int32 = 0
    var bf: Int32 = 0
    var cpair: Int32 = 0

    var smem_a = stack_allocation[
        2 * STAGE_A, _BF16, alignment=16, address_space=AddressSpace.SHARED
    ]()
    var smem_b = stack_allocation[
        2 * STAGE_B, _BF16, alignment=16, address_space=AddressSpace.SHARED
    ]()

    var acc = InlineArray[SIMD[_F32, 4], 2 * NT](fill=SIMD[_F32, 4]())
    var va = InlineArray[SIMD[_BF16, 8], ACH](fill=SIMD[_BF16, 8]())
    var vb = InlineArray[SIMD[_BF16, 8], BCH](fill=SIMD[_BF16, 8]())

    # QKMAJ: in the guarded small-tile regime, m/n-contiguous operands use
    # the quad load mapping (see _g2r_mc) plus k-major staging with one 16B
    # conflict-free vector store per chunk, and the k-major fragment gathers
    # in compute_tile.  This quarters the global load sectors of the
    # scalarized guarded loads and removes the same-word conflicts of the
    # dim-major scalar staging stores.  Measured on H100 PCIe for the
    # latency-bound 64x64 regime; the larger-tile and FASTK kernels keep
    # their existing layouts byte-for-byte.
    comptime QKMAJ = (not FASTK) and BM == 64 and BN == 64

    @parameter
    @always_inline
    def load_tile(k0: Int32):
        @parameter
        if TA:
            _g2r_mc[ACH, FASTK, QKMAJ](ap, bm0, mi, k0, ki, tid, af, va)
        else:
            _g2r_kc[ACH, FASTK](ap, bm0, mi, k0, ki, tid, af, va)

        @parameter
        if TB:
            _g2r_kc[BCH, FASTK](bp, bn0, ni, k0, ki, tid, bf, vb)
        else:
            _g2r_mc[BCH, FASTK, QKMAJ](bp, bn0, ni, k0, ki, tid, bf, vb)

    @parameter
    @always_inline
    def store_tile(stage: Int):
        var base_a = Int32(stage * STAGE_A)
        var base_b = Int32(stage * STAGE_B)

        @parameter
        if TA and FASTK:

            @parameter
            for it in range(ACH):
                var item = tid + Int32(it * _THREADS)
                var kr = item % _BK
                var rc = (item // _BK) * 8
                smem_a.store[alignment=16](
                    Int(base_a + kr * Int32(LDA_K) + rc), va[it]
                )
        elif TA and QKMAJ:
            # Must mirror the QUAD mapping in _g2r_mc.  rc is a multiple of
            # 8 and the pitch is 144B, so every store keeps 16B alignment.
            @parameter
            for it in range(ACH):
                var kr = (tid // 4) % Int32(_BK)
                var rc = (tid % 4) * 8 + (tid // 128) * 32 + Int32(it * 64)
                smem_a.store[alignment=16](
                    Int(base_a + kr * Int32(LDA_K) + rc), va[it]
                )
        elif TA:

            @parameter
            for it in range(ACH):
                var item = tid + Int32(it * _THREADS)
                var kr = item % _BK
                var rc = (item // _BK) * 8

                @parameter
                for e in range(8):
                    smem_a[
                        Int(base_a + (rc + Int32(e)) * Int32(_LDS) + kr)
                    ] = va[it][e]
        else:

            @parameter
            for it in range(ACH):
                var q = tid // 4
                var r = (q * 4) % 64 + q // 16 + Int32(it * 64)
                var kc = (tid % 4) * 8
                smem_a.store[alignment=16](
                    Int(base_a + r * Int32(_LDS) + kc), va[it]
                )

        @parameter
        if TB:

            @parameter
            for it in range(BCH):
                var q = tid // 4
                var r = (q * 4) % 64 + q // 16 + Int32(it * 64)
                var kc = (tid % 4) * 8
                smem_b.store[alignment=16](
                    Int(base_b + r * Int32(_LDS) + kc), vb[it]
                )
        elif FASTK:

            @parameter
            for it in range(BCH):
                var item = tid + Int32(it * _THREADS)
                var kr = item % _BK
                var rc = (item // _BK) * 8
                smem_b.store[alignment=16](
                    Int(base_b + kr * Int32(LDB_K) + rc), vb[it]
                )
        elif QKMAJ:
            # Must mirror the QUAD mapping in _g2r_mc (see the A branch).
            @parameter
            for it in range(BCH):
                var kr = (tid // 4) % Int32(_BK)
                var rc = (tid % 4) * 8 + (tid // 128) * 32 + Int32(it * 64)
                smem_b.store[alignment=16](
                    Int(base_b + kr * Int32(LDB_K) + rc), vb[it]
                )
        else:

            @parameter
            for it in range(BCH):
                var item = tid + Int32(it * _THREADS)
                var kr = item % _BK
                var rc = (item // _BK) * 8

                @parameter
                for e in range(8):
                    smem_b[
                        Int(base_b + (rc + Int32(e)) * Int32(_LDS) + kr)
                    ] = vb[it][e]

    @parameter
    @always_inline
    def compute_tile(stage: Int):
        var base_a = Int32(stage * STAGE_A)
        var base_b = Int32(stage * STAGE_B)

        @parameter
        for ks in range(2):
            var kb = Int32(ks * 16) + 2 * tg
            var afr = InlineArray[SIMD[_BF16, 8], 2](fill=SIMD[_BF16, 8]())

            @parameter
            for mt in range(2):
                var row = wm + Int32(mt * 16) + g

                @parameter
                if TA and (FASTK or QKMAJ):
                    var c0 = Int(base_a + kb * Int32(LDA_K) + row)
                    afr[mt] = SIMD[_BF16, 8](
                        smem_a[c0],
                        smem_a[c0 + LDA_K],
                        smem_a[c0 + 8],
                        smem_a[c0 + LDA_K + 8],
                        smem_a[c0 + 8 * LDA_K],
                        smem_a[c0 + 9 * LDA_K],
                        smem_a[c0 + 8 * LDA_K + 8],
                        smem_a[c0 + 9 * LDA_K + 8],
                    )
                else:
                    var a01 = smem_a.load[width=2, alignment=4](
                        Int(base_a + row * Int32(_LDS) + kb)
                    )
                    var a23 = smem_a.load[width=2, alignment=4](
                        Int(base_a + (row + 8) * Int32(_LDS) + kb)
                    )
                    var a45 = smem_a.load[width=2, alignment=4](
                        Int(base_a + row * Int32(_LDS) + kb + 8)
                    )
                    var a67 = smem_a.load[width=2, alignment=4](
                        Int(base_a + (row + 8) * Int32(_LDS) + kb + 8)
                    )
                    afr[mt] = a01.join(a23).join(a45.join(a67))

            @parameter
            for nt in range(NT):
                var nr = wn + Int32(nt * 8) + g
                var bfr = SIMD[_BF16, 4]()

                @parameter
                if TB or not (FASTK or QKMAJ):
                    var b01 = smem_b.load[width=2, alignment=4](
                        Int(base_b + nr * Int32(_LDS) + kb)
                    )
                    var b23 = smem_b.load[width=2, alignment=4](
                        Int(base_b + nr * Int32(_LDS) + kb + 8)
                    )
                    bfr = b01.join(b23)
                else:
                    var d0 = Int(base_b + kb * Int32(LDB_K) + nr)
                    bfr = SIMD[_BF16, 4](
                        smem_b[d0],
                        smem_b[d0 + LDB_K],
                        smem_b[d0 + 8 * LDB_K],
                        smem_b[d0 + 9 * LDB_K],
                    )

                @parameter
                for mt in range(2):
                    mma(acc[mt * NT + nt], afr[mt], bfr, acc[mt * NT + nt])

    var kt = (ki + Int32(_BK - 1)) // Int32(_BK)
    # Non-SPLITK builds keep (t0, t_end) = (0, kt): both stay compile-time
    # constants after inlining, so the pre-existing kernels are unchanged.
    var t0: Int32 = 0
    var t_end: Int32 = kt

    @parameter
    if SPLITK:
        t0 = Int32(Int(block_idx.y)) * Int32(chunk_tiles)
        t_end = min(kt, t0 + Int32(chunk_tiles))

    @parameter
    @always_inline
    def run_tiles():
        @parameter
        for i in range(2 * NT):
            acc[i] = SIMD[_F32, 4]()

        load_tile(t0 * Int32(_BK))
        store_tile(0)
        barrier()
        var cur = 0
        var t: Int32 = t0 + 1
        while t < t_end:
            load_tile(t * Int32(_BK))
            compute_tile(cur)
            # The other stage was last read before the previous barrier, and
            # the stage being computed is untouched, so one barrier per tile
            # suffices.
            store_tile(1 - cur)
            barrier()
            cur = 1 - cur
            t += 1
        compute_tile(cur)

        @parameter
        if SPLITK:
            # Tile-blocked FP32 partial store: each block owns a private
            # BM x BN image at (slice, logical row-major tile id), so every
            # pair store is 8B-aligned and fully coalesced no matter how
            # awkward n is.  The reduce kernel owns the bounds masking, the
            # bias add and the bf16 rounding, so no clipping happens here;
            # out-of-range lanes hold zeros (their staged loads zero-fill).
            var blocks_n_t = (ni + Int32(BN - 1)) // Int32(BN)
            var tile_id = (bm0 // Int32(BM)) * blocks_n_t + bn0 // Int32(BN)
            var sp = (
                ws
                + Int(Int32(Int(block_idx.y))) * ws_pitch
                + Int(tile_id) * (BM * BN)
            )

            @parameter
            for mt in range(2):
                var r0 = wm + Int32(mt * 16) + g

                @parameter
                for nt in range(NT):
                    var c = wn + Int32(nt * 8) + 2 * tg
                    var frag = acc[mt * NT + nt]

                    @parameter
                    for h in range(2):
                        var r = r0 + Int32(h * 8)
                        var pair = SIMD[_F32, 2](frag[2 * h], frag[2 * h + 1])
                        sp.store[alignment=8](Int(r) * BN + Int(c), pair)
            return

        @parameter
        for mt in range(2):
            var row0 = bm0 + wm + Int32(mt * 16) + g

            @parameter
            for nt in range(NT):
                var col = bn0 + wn + Int32(nt * 8) + 2 * tg
                if col < ni:
                    var frag = acc[mt * NT + nt]
                    var add0 = Float32(0)
                    var add1 = Float32(0)
                    if has_bias != 0:
                        add0 = bias[Int(col)].cast[_F32]()
                        if col + 1 < ni:
                            add1 = bias[Int(col) + 1].cast[_F32]()

                    @parameter
                    for h in range(2):
                        var row = row0 + Int32(h * 8)
                        if row < mi:
                            var base_idx = Int(row) * n + Int(col)
                            var v0 = (frag[2 * h] + add0).cast[_BF16]()
                            if cpair != 0 and col + 1 < ni:
                                var pair = SIMD[_BF16, 2](
                                    v0, (frag[2 * h + 1] + add1).cast[_BF16]()
                                )
                                cp.store[alignment=4](base_idx, pair)
                            else:
                                cp[base_idx] = v0
                                if col + 1 < ni:
                                    cp[base_idx + 1] = (
                                        frag[2 * h + 1] + add1
                                    ).cast[_BF16]()

    @parameter
    @always_inline
    def set_flags():
        @parameter
        if FASTK:
            af = 1
            bf = 1
            cpair = 1
        else:
            af = 1 if (a_fast != 0 and Int(ap) % 16 == 0) else 0
            bf = 1 if (b_fast != 0 and Int(bp) % 16 == 0) else 0
            cpair = 1 if (c_pair != 0 and Int(cp) % 4 == 0) else 0

    @parameter
    if BATCHED:
        var bc = Int32(batch_count)
        var first_batch = True
        while bz < bc:
            ap = a + Int(bz) * a_bstride
            bp = b + Int(bz) * b_bstride
            cp = output + Int(bz) * c_bstride
            set_flags()
            if not first_batch:
                # The previous batch's final compute_tile still reads shared
                # stages; fence before restaging them for this batch.
                barrier()
            first_batch = False
            run_tiles()
            bz += gdz
    else:
        # One batch per grid.z block; GEMM passes zero strides so this folds
        # to the base pointers. Launches with more than 65,535 batches take
        # the grid-striding BATCHED kernels instead.
        ap = a + Int(bz) * a_bstride
        bp = b + Int(bz) * b_bstride
        cp = output + Int(bz) * c_bstride
        set_flags()
        run_tiles()


@always_inline
def _gemm_layout_tag[TA: Bool, TB: Bool]() -> StaticString:
    comptime if TA:
        return "tt" if TB else "tn"
    else:
        return "nt" if TB else "nn"


@always_inline
def _gemm_regime_tag[BM: Int, BN: Int]() -> StaticString:
    # The 64x64 case must be tested first: testing BN or BM alone is not
    # injective at (64, 64) and would collide with the _w64 tag.
    comptime if BM == 64 and BN == 64:
        return "_s64"
    else:
        comptime if BN == 64:
            return "_w64"
        else:
            comptime if BM == 64:
                return "_m64"
            else:
                return ""


@always_inline
def _gemm_fastk_tag[FASTK: Bool]() -> StaticString:
    return "_f" if FASTK else ""


# The GEMM entry point, one instantiation per (layout, tile regime, FASTK)
# the host dispatcher can select. The name is assembled from the same three
# axes the twenty-four hand-written wrappers spelled out, so every symbol a
# GPU profile or scripts/compare_kernel_asm.py knows is unchanged.
#
# GEMM has a single matrix, passed as batch_count with zero batch strides:
# the guarded builds take the grid-striding BATCHED body (its batch loop runs
# once) and the FASTK builds take the one-batch-per-grid.z body, so BATCHED is
# exactly the complement of FASTK here.
@__name(
    t"bf16_gemm_{_gemm_layout_tag[TA, TB]()}{_gemm_regime_tag[BM, BN]()}{_gemm_fastk_tag[FASTK]()}"
)
def _gemm_entry[
    TA: Bool, TB: Bool, BM: Int, BN: Int, FASTK: Bool
](
    output: _Ptr,
    a: _Ptr,
    b: _Ptr,
    bias: _Ptr,
    m_arg: Int64,
    n_arg: Int64,
    k_arg: Int64,
    has_bias_arg: Int64,
    a_fast_arg: Int64,
    b_fast_arg: Int64,
    c_pair_arg: Int64,
    batch_count_arg: Int64,
):
    # Int is not device-passable (host/device width mismatch); scalars cross
    # the launch ABI as Int64 and index math stays in Int.
    var m = Int(m_arg)
    var n = Int(n_arg)
    var k = Int(k_arg)
    var has_bias = Int(has_bias_arg)
    var a_fast = Int(a_fast_arg)
    var b_fast = Int(b_fast_arg)
    var c_pair = Int(c_pair_arg)
    var batch_count = Int(batch_count_arg)
    # The trailing FP32 pointer is the split-K workspace, unread outside
    # SPLITK builds; `output` stands in as the dummy.
    _mma_tile_impl[TA, TB, BM, BN, FASTK, not FASTK](
        output,
        a,
        b,
        bias,
        m,
        n,
        k,
        0,
        0,
        0,
        has_bias,
        a_fast,
        b_fast,
        c_pair,
        batch_count,
        output.bitcast[Scalar[_F32]](),
        0,
        0,
    )


# Split-K entry for underfilled small grids: same guarded body and tile
# regimes as _gemm_entry, but grid.y slices the K loop and each block writes
# FP32 partials to its workspace slice.  Instantiated for the 128x128 and
# 64x64 regimes only (the ones _pick_regime can select for a small grid with
# both extents above 64); narrow-extent shapes keep their existing routes.
@__name(
    t"bf16_gemm_{_gemm_layout_tag[TA, TB]()}{_gemm_regime_tag[BM, BN]()}_splitk"
)
def _gemm_splitk_entry[
    TA: Bool, TB: Bool, BM: Int, BN: Int
](
    ws: _F32Ptr,
    a: _Ptr,
    b: _Ptr,
    m_arg: Int64,
    n_arg: Int64,
    k_arg: Int64,
    chunk_tiles_arg: Int64,
    ws_pitch_arg: Int64,
    a_fast_arg: Int64,
    b_fast_arg: Int64,
):
    # Int is not device-passable (host/device width mismatch); scalars cross
    # the launch ABI as Int64 and index math stays in Int.
    var m = Int(m_arg)
    var n = Int(n_arg)
    var k = Int(k_arg)
    var chunk_tiles = Int(chunk_tiles_arg)
    var ws_pitch = Int(ws_pitch_arg)
    var a_fast = Int(a_fast_arg)
    var b_fast = Int(b_fast_arg)
    # `output`/`bias` are unread under SPLITK; the workspace and `a` stand in
    # as dummies the same way BMM reuses `a` for its unread bias pointer.
    _mma_tile_impl[TA, TB, BM, BN, False, False, SPLITK=True](
        ws.bitcast[Scalar[_BF16]](),
        a,
        b,
        a,
        m,
        n,
        k,
        0,
        0,
        0,
        0,
        a_fast,
        b_fast,
        0,
        1,
        ws,
        chunk_tiles,
        ws_pitch,
    )


# Reduction of the split-K FP32 workspace slices into the bf16 output, with
# the optional bias add folded in.  The workspace is tile-blocked (see the
# SPLITK epilogue in _mma_tile_impl): slice s starts at s * pitch and holds
# one contiguous BM x BN FP32 image per logical row-major output tile, so
# every vector load here is 16B-aligned regardless of how awkward m and n
# are.  Block (t, q) owns a slab of tile t; each thread sums one vec4 lane
# per group across the slices with two accumulators (paired slice loads in
# flight), then scatters the clipped bf16 result (plus bias) into the
# row-major output.  On H100 PCIe one group of 256x4 elements per block was
# fastest (2 groups halves the grid and lost ~15%); the kernel is launch- and
# L2-latency-bound at these sizes, not bandwidth-bound.
comptime _SPLITK_RED_THREADS = 256
comptime _SPLITK_RED_GROUPS = 1


@__name(t"bf16_gemm_splitk_reduce{_gemm_regime_tag[BM, BN]()}")
def _gemm_splitk_reduce[
    BM: Int, BN: Int
](
    output: _Ptr,
    ws: _F32Ptr,
    bias: _Ptr,
    m_arg: Int64,
    n_arg: Int64,
    pitch_arg: Int64,
    splits_arg: Int64,
    has_bias_arg: Int64,
):
    # Int is not device-passable (host/device width mismatch); scalars cross
    # the launch ABI as Int64 and index math stays in Int.
    var m = Int(m_arg)
    var n = Int(n_arg)
    var pitch = Int(pitch_arg)
    var splits = Int(splits_arg)
    var has_bias = Int(has_bias_arg)
    comptime TILE = BM * BN
    comptime GSTRIDE = _SPLITK_RED_THREADS * 4
    var blocks_n = (n - 1) // BN + 1
    var t = Int(block_idx.x)
    var idx0 = (
        Int(block_idx.y) * (GSTRIDE * _SPLITK_RED_GROUPS)
        + Int(thread_idx.x) * 4
    )
    var acc = InlineArray[SIMD[_F32, 4], _SPLITK_RED_GROUPS](
        fill=SIMD[_F32, 4]()
    )
    var acc_b = InlineArray[SIMD[_F32, 4], _SPLITK_RED_GROUPS](
        fill=SIMD[_F32, 4]()
    )

    @parameter
    for gr in range(_SPLITK_RED_GROUPS):
        acc[gr] = ws.load[width=4, alignment=16](t * TILE + idx0 + gr * GSTRIDE)
    # Two accumulators per group keep pairs of slice loads in flight instead
    # of one serial load-add chain.
    var s = 1
    while s + 1 < splits:

        @parameter
        for gr in range(_SPLITK_RED_GROUPS):
            var base = t * TILE + idx0 + gr * GSTRIDE
            acc[gr] += ws.load[width=4, alignment=16](s * pitch + base)
            acc_b[gr] += ws.load[width=4, alignment=16]((s + 1) * pitch + base)
        s += 2
    if s < splits:

        @parameter
        for gr in range(_SPLITK_RED_GROUPS):
            acc[gr] += ws.load[width=4, alignment=16](
                s * pitch + t * TILE + idx0 + gr * GSTRIDE
            )

    @parameter
    for gr in range(_SPLITK_RED_GROUPS):
        var idx = idx0 + gr * GSTRIDE
        var acc4 = acc[gr] + acc_b[gr]
        # idx is a multiple of 4 and BN is a multiple of 4, so the vec4
        # never crosses a tile row: one (row, col0) pair covers all lanes.
        var row = (t // blocks_n) * BM + idx // BN
        var col0 = (t % blocks_n) * BN + idx % BN
        if row < m:
            if has_bias != 0:

                @parameter
                for e in range(4):
                    if col0 + e < n:
                        acc4[e] += bias[col0 + e].cast[_F32]()
            var obase = row * n + col0
            if col0 + 4 <= n:
                output.store[alignment=2](obase, acc4.cast[_BF16]())
            else:

                @parameter
                for e in range(4):
                    if col0 + e < n:
                        output[obase + e] = acc4[e].cast[_BF16]()


@always_inline
def _bmm_layout_tag[TA: Bool, TB: Bool]() -> StaticString:
    comptime if TA:
        return "tt" if TB else "tn"
    else:
        return "nt" if TB else "nn"


@always_inline
def _bmm_regime_tag[BM: Int, BN: Int]() -> StaticString:
    # The four tile regimes: 128x128 unsuffixed, 128x64 narrow-N, 64x128
    # narrow-M, 64x64 small.  The 64x64 case must be tested first: testing
    # BN or BM alone is not injective at (64, 64) and would collide with
    # the _w64 tag.
    comptime if BM == 64 and BN == 64:
        return "_s64"
    elif BN == 64:
        return "_w64"
    elif BM == 64:
        return "_m64"
    else:
        return ""


@always_inline
def _bmm_fastk_tag[FASTK: Bool]() -> StaticString:
    comptime if FASTK:
        return "_f"
    else:
        return ""


# One generic entry carries all 24 BMM specializations; only the comptime
# tuple (layout pair, tile regime, FASTK) ever varied between them. BATCHED
# is the complement of FASTK rather than a fourth axis: a FASTK launch maps
# one batch per grid.z block, so it never runs the kernel's batch loop, while
# a guarded launch grid-strides batches through it.
@__name(
    t"bf16_bmm_{_bmm_layout_tag[TA, TB]()}{_bmm_regime_tag[BM, BN]()}{_bmm_fastk_tag[FASTK]()}"
)
def _bmm_entry[
    TA: Bool, TB: Bool, BM: Int, BN: Int, FASTK: Bool
](
    output: _Ptr,
    a: _Ptr,
    b: _Ptr,
    m_arg: Int64,
    n_arg: Int64,
    k_arg: Int64,
    c_bstride_arg: Int64,
    a_bstride_arg: Int64,
    b_bstride_arg: Int64,
    a_fast_arg: Int64,
    b_fast_arg: Int64,
    c_pair_arg: Int64,
    batch_count_arg: Int64,
):
    # Int is not device-passable (host/device width mismatch); scalars cross
    # the launch ABI as Int64 and index math stays in Int.
    var m = Int(m_arg)
    var n = Int(n_arg)
    var k = Int(k_arg)
    var c_bstride = Int(c_bstride_arg)
    var a_bstride = Int(a_bstride_arg)
    var b_bstride = Int(b_bstride_arg)
    var a_fast = Int(a_fast_arg)
    var b_fast = Int(b_fast_arg)
    var c_pair = Int(c_pair_arg)
    var batch_count = Int(batch_count_arg)
    # BMM carries no bias: `a` stands in as an unread bias pointer under
    # has_bias = 0, and the batch strides are the real ones.  The trailing
    # FP32 pointer is the split-K workspace, unread outside SPLITK builds.
    _mma_tile_impl[TA, TB, BM, BN, FASTK, not FASTK](
        output,
        a,
        b,
        a,
        m,
        n,
        k,
        c_bstride,
        a_bstride,
        b_bstride,
        0,
        a_fast,
        b_fast,
        c_pair,
        batch_count,
        output.bitcast[Scalar[_F32]](),
        0,
        0,
    )


@always_inline
def _g2r_kc_wide(
    src: _Ptr,
    row0: Int,
    rows: Int,
    k0: Int,
    kdim: Int,
    tid: Int,
    fast: Int,
    mut regs: InlineArray[SIMD[_BF16, 8], 2],
):
    # Full-width copy of the accepted-v1 K-contiguous staging: element
    # (r, kk) lives at src[r * kdim + kk]. Every coordinate and flat offset
    # stays machine Int; the host proved rows * kdim fits in Int.
    @parameter
    for it in range(2):
        var q = tid // 4
        var r = (q * 4) % 64 + q // 16 + it * 64
        var kc = (tid % 4) * 8
        var gr = row0 + r
        var gk = k0 + kc
        var v = SIMD[_BF16, 8]()
        if fast != 0:
            # fast proves 16B base alignment and kdim % 8 == 0, so each
            # 8-wide chunk is aligned and entirely in or out of bounds.
            if gr < rows and gk < kdim:
                v = src.load[width=8, alignment=16](gr * kdim + gk)
        else:
            if gr < rows and gk < kdim:
                if gk + 8 <= kdim:
                    # Whole chunk in bounds; only 2B element alignment holds.
                    v = src.load[width=8, alignment=2](gr * kdim + gk)
                else:

                    @parameter
                    for e in range(8):
                        if gk + e < kdim:
                            v[e] = src[gr * kdim + gk + e]
        regs[it] = v


@always_inline
def _g2r_mc_wide(
    src: _Ptr,
    row0: Int,
    rows: Int,
    k0: Int,
    kdim: Int,
    tid: Int,
    fast: Int,
    mut regs: InlineArray[SIMD[_BF16, 8], 2],
):
    # Full-width copy of the accepted-v1 row-contiguous staging: element
    # (r, kk) lives at src[kk * rows + r].
    @parameter
    for it in range(2):
        var item = tid + it * _THREADS
        var kr = item % _BK
        var rc = (item // _BK) * 8
        var gk = k0 + kr
        var gr = row0 + rc
        var v = SIMD[_BF16, 8]()
        if fast != 0:
            # fast proves 16B base alignment and rows % 8 == 0.
            if gk < kdim and gr < rows:
                v = src.load[width=8, alignment=16](gk * rows + gr)
        else:
            if gk < kdim and gr < rows:
                if gr + 8 <= rows:
                    # Whole chunk in bounds; only 2B element alignment holds.
                    v = src.load[width=8, alignment=2](gk * rows + gr)
                else:

                    @parameter
                    for e in range(8):
                        if gr + e < rows:
                            v[e] = src[gk * rows + gr + e]
        regs[it] = v


@always_inline
def _mma_tile_wide[
    TA: Bool, TB: Bool
](
    output: _Ptr,
    a: _Ptr,
    b: _Ptr,
    bias: _Ptr,
    m: Int,
    n: Int,
    k: Int,
    c_bstride: Int,
    a_bstride: Int,
    b_bstride: Int,
    has_bias: Int,
    a_fast: Int,
    b_fast: Int,
    c_pair: Int,
    batch_count: Int,
):
    # Full-width fallback adapted from the accepted-v1 128x128x32 kernel for
    # launches the Int32 host proof cannot cover. Runtime dims, the logical
    # linear tile index, tile origins, K indices, batch indices, and every
    # flat pointer offset stay machine Int; only lane/thread/shared-local
    # values are narrow by construction. The host caps physical grid.x at
    # 2_147_483_647 and proves every product formed here fits in Int, so
    # blocks grid-stride the logical tile index (and grid.z the batch index)
    # with remaining-distance tests that never form a wrapping add.
    var tid = Int(thread_idx.x)
    var lane = tid % 32
    var warp = tid // 32
    var g = lane // 4
    var tg = lane % 4
    var wm = (warp % 4) * 32
    var wn = (warp // 4) * 64

    # Grouped block ordering as v1: sweep _GROUP_M M-blocks per N column so
    # the active A rows stay resident in L2 while B streams. The ceiling
    # divisions never form dim + tile - 1, so they cannot wrap.
    var blocks_m = (m - 1) // _BM + 1
    var blocks_n = (n - 1) // _BN + 1
    var total_tiles = blocks_m * blocks_n
    var group_span = _GROUP_M * blocks_n
    var gdx = Int(grid_dim.x)
    var gdz = Int(grid_dim.z)
    var lin = Int(block_idx.x)

    var bm0 = 0
    var bn0 = 0
    var ap = a
    var bp = b
    var cp = output
    # a_fast/b_fast/c_pair prove divisibility only; base alignment is checked
    # per batch so individually aligned batches keep the vector paths even
    # when the batch stride breaks alignment for others.
    var af = 0
    var bf = 0
    var cpair = 0

    var smem_a = stack_allocation[
        2 * _STAGE_A, _BF16, alignment=16, address_space=AddressSpace.SHARED
    ]()
    var smem_b = stack_allocation[
        2 * _STAGE_B, _BF16, alignment=16, address_space=AddressSpace.SHARED
    ]()

    var acc = InlineArray[SIMD[_F32, 4], 16](fill=SIMD[_F32, 4]())
    var va = InlineArray[SIMD[_BF16, 8], 2](fill=SIMD[_BF16, 8]())
    var vb = InlineArray[SIMD[_BF16, 8], 2](fill=SIMD[_BF16, 8]())

    @parameter
    @always_inline
    def load_tile(k0: Int):
        @parameter
        if TA:
            _g2r_mc_wide(ap, bm0, m, k0, k, tid, af, va)
        else:
            _g2r_kc_wide(ap, bm0, m, k0, k, tid, af, va)

        @parameter
        if TB:
            _g2r_kc_wide(bp, bn0, n, k0, k, tid, bf, vb)
        else:
            _g2r_mc_wide(bp, bn0, n, k0, k, tid, bf, vb)

    @parameter
    @always_inline
    def store_tile(stage: Int):
        var base_a = stage * _STAGE_A
        var base_b = stage * _STAGE_B

        @parameter
        if TA:

            @parameter
            for it in range(2):
                var item = tid + it * _THREADS
                var kr = item % _BK
                var rc = (item // _BK) * 8

                @parameter
                for e in range(8):
                    smem_a[base_a + (rc + e) * _LDS + kr] = va[it][e]
        else:

            @parameter
            for it in range(2):
                var q = tid // 4
                var r = (q * 4) % 64 + q // 16 + it * 64
                var kc = (tid % 4) * 8
                smem_a.store[alignment=16](base_a + r * _LDS + kc, va[it])

        @parameter
        if TB:

            @parameter
            for it in range(2):
                var q = tid // 4
                var r = (q * 4) % 64 + q // 16 + it * 64
                var kc = (tid % 4) * 8
                smem_b.store[alignment=16](base_b + r * _LDS + kc, vb[it])
        else:

            @parameter
            for it in range(2):
                var item = tid + it * _THREADS
                var kr = item % _BK
                var rc = (item // _BK) * 8

                @parameter
                for e in range(8):
                    smem_b[base_b + (rc + e) * _LDS + kr] = vb[it][e]

    @parameter
    @always_inline
    def compute_tile(stage: Int):
        var base_a = stage * _STAGE_A
        var base_b = stage * _STAGE_B

        @parameter
        for ks in range(2):
            var kb = ks * 16 + 2 * tg
            var afr = InlineArray[SIMD[_BF16, 8], 2](fill=SIMD[_BF16, 8]())

            @parameter
            for mt in range(2):
                var row = wm + mt * 16 + g
                var a01 = smem_a.load[width=2, alignment=4](
                    base_a + row * _LDS + kb
                )
                var a23 = smem_a.load[width=2, alignment=4](
                    base_a + (row + 8) * _LDS + kb
                )
                var a45 = smem_a.load[width=2, alignment=4](
                    base_a + row * _LDS + kb + 8
                )
                var a67 = smem_a.load[width=2, alignment=4](
                    base_a + (row + 8) * _LDS + kb + 8
                )
                afr[mt] = a01.join(a23).join(a45.join(a67))

            @parameter
            for nt in range(8):
                var nr = wn + nt * 8 + g
                var b01 = smem_b.load[width=2, alignment=4](
                    base_b + nr * _LDS + kb
                )
                var b23 = smem_b.load[width=2, alignment=4](
                    base_b + nr * _LDS + kb + 8
                )
                var bfr = b01.join(b23)

                @parameter
                for mt in range(2):
                    mma(acc[mt * 8 + nt], afr[mt], bfr, acc[mt * 8 + nt])

    var kt = (k - 1) // _BK + 1
    var first_work = True
    while lin < total_tiles:
        var gid = lin // group_span
        var rem = lin % group_span
        var rows_in_group = min(_GROUP_M, blocks_m - gid * _GROUP_M)
        bm0 = (gid * _GROUP_M + rem % rows_in_group) * _BM
        bn0 = (rem // rows_in_group) * _BN

        var bz = Int(block_idx.z)
        while bz < batch_count:
            ap = a + bz * a_bstride
            bp = b + bz * b_bstride
            cp = output + bz * c_bstride
            af = 1 if (a_fast != 0 and Int(ap) % 16 == 0) else 0
            bf = 1 if (b_fast != 0 and Int(bp) % 16 == 0) else 0
            cpair = 1 if (c_pair != 0 and Int(cp) % 4 == 0) else 0

            if not first_work:
                # The previous (tile, batch) pair's final compute_tile still
                # reads the shared stages; fence before restaging. This
                # covers batch-to-batch moves and the hop from one logical
                # tile's last batch to the next tile's first batch.
                barrier()
            first_work = False

            @parameter
            for i in range(16):
                acc[i] = SIMD[_F32, 4]()

            load_tile(0)
            store_tile(0)
            barrier()
            var cur = 0
            for t in range(1, kt):
                load_tile(t * _BK)
                compute_tile(cur)
                # The other stage was last read before the previous barrier,
                # and the stage being computed is untouched, so one barrier
                # per tile suffices.
                store_tile(1 - cur)
                barrier()
                cur = 1 - cur
            compute_tile(cur)

            @parameter
            for mt in range(2):
                var row0 = bm0 + wm + mt * 16 + g

                @parameter
                for nt in range(8):
                    var col = bn0 + wn + nt * 8 + 2 * tg
                    if col < n:
                        var frag = acc[mt * 8 + nt]
                        var add0 = Float32(0)
                        var add1 = Float32(0)
                        if has_bias != 0:
                            add0 = bias[col].cast[_F32]()
                            if col + 1 < n:
                                add1 = bias[col + 1].cast[_F32]()

                        @parameter
                        for h in range(2):
                            var row = row0 + h * 8
                            if row < m:
                                var base_idx = row * n + col
                                var v0 = (frag[2 * h] + add0).cast[_BF16]()
                                if cpair != 0 and col + 1 < n:
                                    var pair = SIMD[_BF16, 2](
                                        v0,
                                        (frag[2 * h + 1] + add1).cast[_BF16](),
                                    )
                                    cp.store[alignment=4](base_idx, pair)
                                else:
                                    cp[base_idx] = v0
                                    if col + 1 < n:
                                        cp[base_idx + 1] = (
                                            frag[2 * h + 1] + add1
                                        ).cast[_BF16]()

            # Overflow-safe batch advance: compare the remaining distance
            # first so bz + gdz is never formed past the final batch.
            if batch_count - bz <= gdz:
                break
            bz += gdz

        # Overflow-safe logical-tile advance, same shape as the batch loop:
        # lin + gdx is never formed past the final logical tile.
        if total_tiles - lin <= gdx:
            break
        lin += gdx


@always_inline
def _wide_layout_tag[TA: Bool, TB: Bool]() -> StaticString:
    # Kernel-symbol layout suffix: A then B, "t" transposed, "n" not.
    comptime if TA:
        return "tt" if TB else "tn"
    else:
        return "nt" if TB else "nn"


@__name(t"bf16_gemm_{_wide_layout_tag[TA, TB]()}_wide")
def _gemm_wide[
    TA: Bool, TB: Bool
](
    output: _Ptr,
    a: _Ptr,
    b: _Ptr,
    bias: _Ptr,
    m_arg: Int64,
    n_arg: Int64,
    k_arg: Int64,
    has_bias_arg: Int64,
    a_fast_arg: Int64,
    b_fast_arg: Int64,
    c_pair_arg: Int64,
    batch_count_arg: Int64,
):
    # Int is not device-passable (host/device width mismatch); scalars cross
    # the launch ABI as Int64 and index math stays in Int.
    var m = Int(m_arg)
    var n = Int(n_arg)
    var k = Int(k_arg)
    var has_bias = Int(has_bias_arg)
    var a_fast = Int(a_fast_arg)
    var b_fast = Int(b_fast_arg)
    var c_pair = Int(c_pair_arg)
    var batch_count = Int(batch_count_arg)
    _mma_tile_wide[TA, TB](
        output,
        a,
        b,
        bias,
        m,
        n,
        k,
        0,
        0,
        0,
        has_bias,
        a_fast,
        b_fast,
        c_pair,
        batch_count,
    )


@__name(t"bf16_bmm_{_wide_layout_tag[TA, TB]()}_wide")
def _bmm_wide[
    TA: Bool, TB: Bool
](
    output: _Ptr,
    a: _Ptr,
    b: _Ptr,
    m_arg: Int64,
    n_arg: Int64,
    k_arg: Int64,
    c_bstride_arg: Int64,
    a_bstride_arg: Int64,
    b_bstride_arg: Int64,
    a_fast_arg: Int64,
    b_fast_arg: Int64,
    c_pair_arg: Int64,
    batch_count_arg: Int64,
):
    # Int is not device-passable (host/device width mismatch); scalars cross
    # the launch ABI as Int64 and index math stays in Int.
    var m = Int(m_arg)
    var n = Int(n_arg)
    var k = Int(k_arg)
    var c_bstride = Int(c_bstride_arg)
    var a_bstride = Int(a_bstride_arg)
    var b_bstride = Int(b_bstride_arg)
    var a_fast = Int(a_fast_arg)
    var b_fast = Int(b_fast_arg)
    var c_pair = Int(c_pair_arg)
    var batch_count = Int(batch_count_arg)
    _mma_tile_wide[TA, TB](
        output,
        a,
        b,
        a,
        m,
        n,
        k,
        c_bstride,
        a_bstride,
        b_bstride,
        0,
        a_fast,
        b_fast,
        c_pair,
        batch_count,
    )


@always_inline
def _a_fast_flag(m: Int, k: Int, transpose_a: Bool) -> Int:
    # Row divisibility only; guarded kernels prove base alignment per batch.
    var contig = m if transpose_a else k
    return 1 if contig % 8 == 0 else 0


@always_inline
def _b_fast_flag(n: Int, k: Int, transpose_b: Bool) -> Int:
    # Row divisibility only; guarded kernels prove base alignment per batch.
    var contig = k if transpose_b else n
    return 1 if contig % 8 == 0 else 0


@always_inline
def _fast_proof(
    output: _Ptr,
    a: _Ptr,
    b: _Ptr,
    m: Int,
    n: Int,
    k: Int,
    c_bstride: Int,
    a_bstride: Int,
    b_bstride: Int,
    transpose_a: Bool,
    transpose_b: Bool,
) -> Bool:
    # A FASTK launch bakes in: 16B-aligned vector staging for both operands
    # in every batch, and 4B-aligned pair stores of the output. Strides are
    # in elements (2B), so 16B alignment needs stride % 8 == 0.
    var a_ok = (
        _a_fast_flag(m, k, transpose_a) == 1
        and Int(a) % 16 == 0
        and a_bstride % 8 == 0
    )
    var b_ok = (
        _b_fast_flag(n, k, transpose_b) == 1
        and Int(b) % 16 == 0
        and b_bstride % 8 == 0
    )
    var c_ok = n % 2 == 0 and Int(output) % 4 == 0 and c_bstride % 2 == 0
    return a_ok and b_ok and c_ok


@always_inline
def _pick_regime(m: Int, n: Int, batch_count: Int, sm_count: Int) -> Int:
    # 0: 128x128, 1: 128x64, 2: 64x128, 3: 64x64. Narrow tiles only where
    # the wide tile wastes at least half of one extent (small n or small m);
    # the wide tile's compute-to-traffic ratio wins elsewhere, including
    # long-K weight-gradient shapes where halving a tile doubles operand
    # re-reads.
    if n <= 64:
        return 1
    if m <= 64:
        return 2
    # Small/awkward regime, keyed on shape, batch count, and SM count only:
    # take 64x64 tiles when the whole 128x128 grid -- blocks per matrix
    # times the batch count, since batched launches run one grid.z slice
    # per batch -- cannot fill the GPU (fewer blocks than SMs) AND the
    # 64x64 grid still fits in a single wave, so the 4x CTA count is pure
    # occupancy with no extra wave of operand re-reads. Shapes whose 64x64
    # grid would spill past one wave (e.g. deep-K 1024x1024, or attention
    # -prefill bmm batches whose per-matrix underfill is already covered by
    # grid.z) keep the wide tile's better compute-to-traffic ratio. The
    # non-batched GEMM path passes batch_count=1.
    # The one-wave bound was fitted on an H100 PCIe (114 SMs).  The
    # batch_count <= sm_count and _I32_MAX guards keep the block products
    # far from machine-Int wrap; anything larger fails the Int32 proof and
    # takes the wide fallback regardless of regime.
    if (
        sm_count > 0
        and batch_count > 0
        and batch_count <= sm_count
        and m <= _I32_MAX
        and n <= _I32_MAX
    ):
        var blocks_full = (
            ((m - 1) // _BM + 1) * ((n - 1) // _BN + 1) * batch_count
        )
        var blocks_s64 = ((m - 1) // 64 + 1) * ((n - 1) // 64 + 1) * batch_count
        if blocks_full < sm_count and blocks_s64 <= sm_count:
            return 3
    return 0


@always_inline
def _regime_bm(regime: Int) -> Int:
    # Single source of truth for the regime -> (BM, BN) tile mapping
    # (0: 128x128, 1: 128x64, 2: 64x128, 3: 64x64), shared by the runtime
    # dispatch prologues and the comptime instantiation ladders so the two
    # cannot drift apart.
    return 64 if (regime == 2 or regime == 3) else 128


@always_inline
def _regime_bn(regime: Int) -> Int:
    # See _regime_bm: the BN half of the shared regime -> tile mapping.
    return 64 if (regime == 1 or regime == 3) else 128


@always_inline
def _opt_grid_x(m: Int, n: Int, k: Int, bm: Int, bn: Int) -> Int:
    # Machine-width proof that the Int32 optimized kernels cannot wrap for
    # this launch; returns grid.x, or 0 when the proof fails. Ordered so no
    # step can itself overflow: positivity first, then the selected-regime
    # caps (so mi + bm - 1, ni + bn - 1, ki + 31, every staged coordinate
    # dim0 + 126, kt * _BK, and 8 * blocks_n stay under 2**31), then block
    # counts, then a guarded product so grid.x is in (0, 2**31 - 1] and the
    # kernel's grouped bm0/bn0 decomposition of block_idx.x stays in range.
    if m <= 0 or n <= 0 or k <= 0:
        return 0
    if m > _I32_MAX - (bm - 1) or n > _I32_MAX - (bn - 1):
        return 0
    if k > _I32_MAX - 31:
        return 0
    var blocks_m = (m - 1) // bm + 1
    var blocks_n = (n - 1) // bn + 1
    # The grouped block decoder narrows _GROUP_M * blocks_n to Int32.
    if blocks_n > _I32_MAX // _GROUP_M:
        return 0
    if blocks_m > _I32_MAX // blocks_n:
        return 0
    return blocks_m * blocks_n


@always_inline
def _opt_bmm_address_proof(
    m: Int,
    n: Int,
    k: Int,
    batch_count: Int,
    c_bstride: Int,
    a_bstride: Int,
    b_bstride: Int,
) -> Bool:
    # Called only after _opt_grid_x and the ordered batch-count proof have
    # established positive Int32-safe dimensions and batch_count. Prove the
    # machine-Int pointer arithmetic separately: both each flat matrix span
    # and the last batch base plus that span must be representable. Keep the
    # checks ordered so no product or sum is evaluated before its bound.
    if c_bstride < 0 or a_bstride < 0 or b_bstride < 0:
        return False
    if k > _I64_MAX // m:
        return False
    if k > _I64_MAX // n:
        return False
    if n > _I64_MAX // m:
        return False
    var a_span = m * k
    var b_span = k * n
    var c_span = m * n
    var last_batch = batch_count - 1
    if a_bstride > 0:
        if last_batch > (_I64_MAX - a_span) // a_bstride:
            return False
    if b_bstride > 0:
        if last_batch > (_I64_MAX - b_span) // b_bstride:
            return False
    if c_bstride > 0:
        if last_batch > (_I64_MAX - c_span) // c_bstride:
            return False
    return True


@always_inline
def _wide_grid_x(
    m: Int,
    n: Int,
    k: Int,
    batch_count: Int,
    c_bstride: Int,
    a_bstride: Int,
    b_bstride: Int,
) raises -> Int:
    # Pre-launch proof for the full-width fallback: raise instead of letting
    # any launch or addressing value wrap machine Int. Returns the physical
    # grid.x: the logical tile count capped at the CUDA 2_147_483_647 limit;
    # blocks grid-stride the remaining logical tiles.
    if m <= 0 or n <= 0 or k <= 0 or batch_count <= 0:
        raise Error("bf16 gemm/bmm: dims and batch count must be positive")
    if c_bstride < 0 or a_bstride < 0 or b_bstride < 0:
        raise Error("bf16 gemm/bmm: batch strides must be non-negative")
    # Tile-local coordinate offsets (< one 128x128x32 tile) must not wrap.
    if m > _I64_MAX - _BM or n > _I64_MAX - _BN or k > _I64_MAX - _BK:
        raise Error("bf16 gemm/bmm: dimension too large for machine Int")
    # Flat spans m*k, k*n, and m*n bound every row*kdim + gk, gk*rows + gr,
    # and row*n + col offset the kernel forms; each must fit in Int.
    if k > _I64_MAX // m or k > _I64_MAX // n or n > _I64_MAX // m:
        raise Error("bf16 gemm/bmm: matrix span exceeds machine Int")
    var a_span = m * k
    var b_span = k * n
    var c_span = m * n
    var blocks_m = (m - 1) // _BM + 1
    var blocks_n = (n - 1) // _BN + 1
    # The kernel forms group_span = _GROUP_M * blocks_n and the logical tile
    # count blocks_m * blocks_n; guard both products.
    if blocks_n > _I64_MAX // _GROUP_M or blocks_m > _I64_MAX // blocks_n:
        raise Error("bf16 gemm/bmm: logical tile count exceeds machine Int")
    # Last-batch base offset plus the in-matrix span must fit machine Int.
    var last_batch = batch_count - 1
    if a_bstride > 0:
        if last_batch > (_I64_MAX - a_span) // a_bstride:
            raise Error("bf16 gemm/bmm: A batch span exceeds machine Int")
    if b_bstride > 0:
        if last_batch > (_I64_MAX - b_span) // b_bstride:
            raise Error("bf16 gemm/bmm: B batch span exceeds machine Int")
    if c_bstride > 0:
        if last_batch > (_I64_MAX - c_span) // c_bstride:
            raise Error("bf16 gemm/bmm: output batch span exceeds machine Int")
    return min(blocks_m * blocks_n, _I32_MAX)


def _enqueue_gemm_wide(
    output: _Ptr,
    a: _Ptr,
    b: _Ptr,
    bias: _Ptr,
    m: Int,
    n: Int,
    k: Int,
    hb: Int,
    a_fast: Int,
    b_fast: Int,
    c_pair: Int,
    transpose_a: Bool,
    transpose_b: Bool,
    ctx: DeviceContext,
) raises:
    var grid_x = _wide_grid_x(m, n, k, 1, 0, 0, 0)

    # One leaf per layout: the launch arguments are the same for all four, so
    # only the comptime (TA, TB) specialization varies.
    comptime for layout in range(4):
        comptime TA = layout >= 2
        comptime TB = layout % 2 == 1
        if transpose_a == TA and transpose_b == TB:
            ctx.enqueue_function[_gemm_wide[TA, TB]](
                output,
                a,
                b,
                bias,
                Int64(m),
                Int64(n),
                Int64(k),
                Int64(hb),
                Int64(a_fast),
                Int64(b_fast),
                Int64(c_pair),
                Int64(1),
                grid_dim=(grid_x,),
                block_dim=(_THREADS,),
            )
            return


def _enqueue_bmm_wide(
    output: _Ptr,
    a: _Ptr,
    b: _Ptr,
    batch_count: Int,
    m: Int,
    n: Int,
    k: Int,
    c_bstride: Int,
    a_bstride: Int,
    b_bstride: Int,
    a_fast: Int,
    b_fast: Int,
    c_pair: Int,
    transpose_a: Bool,
    transpose_b: Bool,
    ctx: DeviceContext,
) raises:
    var grid_x = _wide_grid_x(
        m, n, k, batch_count, c_bstride, a_bstride, b_bstride
    )
    # CUDA grid.z tops out at 65,535; blocks grid-stride over extra batches.
    var grid_z = min(batch_count, 65535)

    # One leaf per layout: the launch arguments are the same for all four, so
    # only the comptime (TA, TB) specialization varies.
    comptime for layout in range(4):
        comptime TA = layout >= 2
        comptime TB = layout % 2 == 1
        if transpose_a == TA and transpose_b == TB:
            ctx.enqueue_function[_bmm_wide[TA, TB]](
                output,
                a,
                b,
                Int64(m),
                Int64(n),
                Int64(k),
                Int64(c_bstride),
                Int64(a_bstride),
                Int64(b_bstride),
                Int64(a_fast),
                Int64(b_fast),
                Int64(c_pair),
                Int64(batch_count),
                grid_dim=(grid_x, 1, grid_z),
                block_dim=(_THREADS,),
            )
            return


def _enqueue_gemm_splitk(
    output: _Ptr,
    a: _Ptr,
    b: _Ptr,
    bias: _Ptr,
    m: Int,
    n: Int,
    k: Int,
    hb: Int,
    a_fast: Int,
    b_fast: Int,
    bm: Int,
    bn: Int,
    grid_x: Int,
    splits_in: Int,
    transpose_a: Bool,
    transpose_b: Bool,
    ctx: DeviceContext,
) raises:
    # Recompute splits from the rounded-up chunk so no grid.y slice is empty
    # (e.g. kt = 5 with splits_in = 4 collapses to 3 slices of 2, 2, 1).
    var kt = (k - 1) // _BK + 1
    var chunk_tiles = (kt + splits_in - 1) // splits_in
    var splits = (kt + chunk_tiles - 1) // chunk_tiles

    comptime for ri in range(4):
        comptime if ri == 0 or ri == 3:
            comptime BM = _regime_bm(ri)
            comptime BN = _regime_bn(ri)
            if bm == BM and bn == BN:
                # One contiguous BM x BN FP32 image per (slice, tile); the
                # tile-blocked layout keeps both the epilogue stores and the
                # reduce loads aligned and coalesced (see the SPLITK
                # epilogue in _mma_tile_impl).
                var pitch = grid_x * (BM * BN)
                var ws = ctx.enqueue_create_buffer[DType.float32](
                    splits * pitch
                )
                var ws_ptr = ws.unsafe_ptr().as_unsafe_any_origin()

                comptime for li in range(4):
                    comptime TA = li >= 2
                    comptime TB = li % 2 == 1
                    if transpose_a == TA and transpose_b == TB:
                        ctx.enqueue_function[
                            _gemm_splitk_entry[TA, TB, BM, BN]
                        ](
                            ws_ptr,
                            a,
                            b,
                            Int64(m),
                            Int64(n),
                            Int64(k),
                            Int64(chunk_tiles),
                            Int64(pitch),
                            Int64(a_fast),
                            Int64(b_fast),
                            grid_dim=(grid_x, splits),
                            block_dim=(_THREADS,),
                        )
                ctx.enqueue_function[_gemm_splitk_reduce[BM, BN]](
                    output,
                    ws_ptr,
                    bias,
                    Int64(m),
                    Int64(n),
                    Int64(pitch),
                    Int64(splits),
                    Int64(hb),
                    grid_dim=(
                        grid_x,
                        (BM * BN)
                        // (_SPLITK_RED_THREADS * 4 * _SPLITK_RED_GROUPS),
                    ),
                    block_dim=(_SPLITK_RED_THREADS,),
                )
                # Normal release after both stream-ordered consumers are
                # enqueued.
                _ = ws^
                return
    # Only the 128x128 and 64x64 regimes are instantiated above; falling
    # through would enqueue nothing and return with the output buffer
    # unwritten -- a silent wrong result.  Unreachable while the caller
    # restricts this route to regimes 0 and 3, but refuse loudly if the
    # dispatch and the instantiation ladder ever drift apart.
    raise Error("bf16 gemm split-K: no kernel instantiated for this tile")


def enqueue_bf16_gemm(
    output: UnsafePointer[Scalar[DType.bfloat16], MutAnyOrigin],
    a: UnsafePointer[Scalar[DType.bfloat16], MutAnyOrigin],
    b: UnsafePointer[Scalar[DType.bfloat16], MutAnyOrigin],
    bias: UnsafePointer[Scalar[DType.bfloat16], MutAnyOrigin],
    m: Int,
    n: Int,
    k: Int,
    transpose_a: Bool,
    transpose_b: Bool,
    has_bias: Bool,
    ctx: DeviceContext,
) raises:
    var a_fast = _a_fast_flag(m, k, transpose_a)
    var b_fast = _b_fast_flag(n, k, transpose_b)
    var c_pair = 1 if n % 2 == 0 else 0
    var hb = 1 if has_bias else 0
    var sm_count = ctx.get_attribute(DeviceAttribute.MULTIPROCESSOR_COUNT)
    var regime = _pick_regime(m, n, 1, sm_count)
    var bm = _regime_bm(regime)
    var bn = _regime_bn(regime)
    # The Int32 kernels are legal only under the machine-width proof; a zero
    # grid means some narrowed value could wrap, so route to the full-width
    # fallback instead.
    var grid_x = _opt_grid_x(m, n, k, bm, bn)
    if grid_x == 0:
        _enqueue_gemm_wide(
            output,
            a,
            b,
            bias,
            m,
            n,
            k,
            hb,
            a_fast,
            b_fast,
            c_pair,
            transpose_a,
            transpose_b,
            ctx,
        )
        return
    # Split-K route for underfilled small grids: with fewer blocks than SMs
    # every CTA runs alone on its SM and the whole K loop is one serial
    # latency chain, so slicing K over grid.y multiplies both the block
    # count and the number of chains in flight.  Applied only when both
    # extents exceed 64 (the 128x128 and 64x64 regimes; narrow-extent
    # shapes keep their existing routes) and when at least two slices of
    # two K tiles each exist.  The 3x-SMs block target and the cap of 8
    # slices were fitted on an H100 PCIe (114 SMs, 50 MB L2: the FP32
    # workspace round-trip stays L2-resident); stock PyTorch picks the
    # same split=4 shape for 357x789x333.
    if (regime == 0 or regime == 3) and sm_count > 0 and grid_x < sm_count:
        var kt_total = (k - 1) // _BK + 1
        # The 64x64 regime profits when K is deep enough for at least two
        # slices of three tiles AND the direct 64x64 grid leaves a good
        # chunk of the GPU idle (measured on H100 PCIe: 500x500x123 at kt=4
        # loses ~7% to the workspace round-trip, 357x789x333 at kt=11 wins
        # 1.4-1.6x).  Two refinements from a 3-round interleaved A/B sweep
        # of m,n in [450, 600] x k in [160, 320], all four layouts, on an
        # H100 PCIe (114 SMs):
        #   - Near-full grids: at 100 direct blocks (600x600, 88% of one
        #     wave) the split loses every cell, both layouts, +13% to +47%;
        #     at 81 blocks (550x550) it is mixed and at <= 72 it wins.
        #     Hence engage only while the direct grid is under 3/4 of the
        #     SM count.
        #   - Shallow K (kt 6..7): only TN profits (-5% to -20% at 64..81
        #     blocks); NN loses +4% to +15% (500x500x200, this defect's
        #     corner, was a wash: 8.04 vs 8.14 us) and NT loses up to +44%
        #     (500x500x200: 6.86 vs 4.77 us).  TT was not in that sweep: at
        #     the time it was rewritten as NN upstream, and it now arrives
        #     directly only when the aligned v4 TT routes decline, so it
        #     rides the generic kt >= 8 arm unfitted.  From kt = 8 up, every
        #     swept layout wins
        #     at <= 81 blocks (-6% to -34%), except NN 550x550x320 (+4.3%),
        #     the one loss this fit knowingly keeps: excluding it needs a
        #     tighter grid cutoff that costs the larger TN wins next to it.
        var split_ok = (
            regime == 3
            and 4 * grid_x <= 3 * sm_count
            and (
                kt_total >= 8
                or (transpose_a and not transpose_b and kt_total >= 6)
            )
        ) or (
            regime == 0
            and ((2 * grid_x <= sm_count and kt_total >= 8) or kt_total >= 64)
        )
        var splits = (3 * sm_count) // grid_x
        if splits > (kt_total + 1) // 2:
            splits = (kt_total + 1) // 2
        if splits > 8:
            splits = 8
        if split_ok and splits >= 2:
            _enqueue_gemm_splitk(
                output,
                a,
                b,
                bias,
                m,
                n,
                k,
                hb,
                a_fast,
                b_fast,
                bm,
                bn,
                grid_x,
                splits,
                transpose_a,
                transpose_b,
                ctx,
            )
            return
    var fastk = _fast_proof(
        output, a, b, m, n, k, 0, 0, 0, transpose_a, transpose_b
    )
    # Below one block per SM (132 on H100) occupancy cannot rise, and the
    # guarded kernel's larger register budget buys more ILP per block, so
    # the low-register FASTK build only wins past that point.
    fastk = fastk and grid_x > 132

    # Leaf selection, in the order the hand-written if-tree tested it: the
    # FASTK proof, then the tile regime, then the two transposes. `bm`/`bn`
    # are the regime's tile extents computed above, so the 128x128 arm is
    # reached exactly when the old `else` was. Every arm launches the same
    # arguments; only the specialization differs.
    comptime for fi in range(2):
        comptime FASTK = fi == 1
        if fastk == FASTK:
            comptime for ri in range(4):
                comptime BM = _regime_bm(ri)
                comptime BN = _regime_bn(ri)
                if bm == BM and bn == BN:
                    comptime for li in range(4):
                        comptime TA = li >= 2
                        comptime TB = li % 2 == 1
                        if transpose_a == TA and transpose_b == TB:
                            ctx.enqueue_function[
                                _gemm_entry[TA, TB, BM, BN, FASTK]
                            ](
                                output,
                                a,
                                b,
                                bias,
                                Int64(m),
                                Int64(n),
                                Int64(k),
                                Int64(hb),
                                Int64(a_fast),
                                Int64(b_fast),
                                Int64(c_pair),
                                Int64(1),
                                grid_dim=(grid_x,),
                                block_dim=(_THREADS,),
                            )


def enqueue_bf16_bmm(
    output: UnsafePointer[Scalar[DType.bfloat16], MutAnyOrigin],
    a: UnsafePointer[Scalar[DType.bfloat16], MutAnyOrigin],
    b: UnsafePointer[Scalar[DType.bfloat16], MutAnyOrigin],
    batch_count: Int,
    m: Int,
    n: Int,
    k: Int,
    output_batch_stride: Int,
    a_batch_stride: Int,
    b_batch_stride: Int,
    transpose_a: Bool,
    transpose_b: Bool,
    ctx: DeviceContext,
) raises:
    var a_fast = _a_fast_flag(m, k, transpose_a)
    var b_fast = _b_fast_flag(n, k, transpose_b)
    var c_pair = 1 if n % 2 == 0 else 0
    var sm_count = ctx.get_attribute(DeviceAttribute.MULTIPROCESSOR_COUNT)
    var regime = _pick_regime(m, n, batch_count, sm_count)
    var bm = _regime_bm(regime)
    var bn = _regime_bn(regime)
    var grid_x = 0
    var grid_z = 0
    var opt_ok = False
    # Prove batch_count before forming any batch-derived value. CUDA grid.z
    # tops out at 65,535; guarded blocks grid-stride over extra batches.
    if batch_count > 0:
        grid_z = min(batch_count, 65535)
        # The Int32 batch loop's final step forms bz + grid_dim.z with
        # bz <= batch_count - 1; this overflow-safe comparison bounds that
        # sum by 2**31 - 1 without ever evaluating batch_count - 1 + grid_z.
        if batch_count <= 2_147_483_648 - grid_z:
            grid_x = _opt_grid_x(m, n, k, bm, bn)
            if grid_x > 0:
                opt_ok = _opt_bmm_address_proof(
                    m,
                    n,
                    k,
                    batch_count,
                    output_batch_stride,
                    a_batch_stride,
                    b_batch_stride,
                )
    if not opt_ok:
        _enqueue_bmm_wide(
            output,
            a,
            b,
            batch_count,
            m,
            n,
            k,
            output_batch_stride,
            a_batch_stride,
            b_batch_stride,
            a_fast,
            b_fast,
            c_pair,
            transpose_a,
            transpose_b,
            ctx,
        )
        return
    var fastk = _fast_proof(
        output,
        a,
        b,
        m,
        n,
        k,
        output_batch_stride,
        a_batch_stride,
        b_batch_stride,
        transpose_a,
        transpose_b,
    )
    # Same one-block-per-SM threshold as the GEMM path, counting batches.
    # Compare by division instead of forming grid_x * grid_z. FASTK BMM
    # kernels map one batch per grid.z block, so huge batch counts fall back
    # to the grid-striding guarded kernels.
    if grid_x <= 132 // grid_z:
        fastk = False
    if batch_count > 65535:
        fastk = False

    # One comptime ladder in place of the 24-leaf if-tree: it instantiates
    # exactly the same (layout, regime, FASTK) entries and launches each with
    # the same argument list. The regime arm is matched on the bm/bn the
    # prologue already derived from it, which keeps the old tree's
    # else-catch-all: anything that is not the narrow-N or narrow-M regime
    # runs the 128x128 tiles.
    @parameter
    for RI in range(4):
        comptime BM = _regime_bm(RI)
        comptime BN = _regime_bn(RI)

        @parameter
        for FI in range(2):
            comptime FASTK = FI == 1

            @parameter
            for TAI in range(2):
                comptime TA = TAI == 1

                @parameter
                for TBI in range(2):
                    comptime TB = TBI == 1
                    if (
                        bm == BM
                        and bn == BN
                        and fastk == FASTK
                        and transpose_a == TA
                        and transpose_b == TB
                    ):
                        ctx.enqueue_function[_bmm_entry[TA, TB, BM, BN, FASTK]](
                            output,
                            a,
                            b,
                            Int64(m),
                            Int64(n),
                            Int64(k),
                            Int64(output_batch_stride),
                            Int64(a_batch_stride),
                            Int64(b_batch_stride),
                            Int64(a_fast),
                            Int64(b_fast),
                            Int64(c_pair),
                            Int64(batch_count),
                            grid_dim=(grid_x, 1, grid_z),
                            block_dim=(_THREADS,),
                        )
                        return
