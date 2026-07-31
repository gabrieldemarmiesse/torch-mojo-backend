"""Candidate H100 BF16 GEMM/BMM built on mma.sync m16n8k16 tensor cores.

Three shared-memory tile regimes (128x128x32, 128x64x32, 64x128x32) serve
every runtime shape, layout, and batch; the host picks per launch from
runtime dims only (narrow tiles when one extent is <= 64, wide otherwise).
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
from std.gpu import barrier, block_idx, grid_dim, thread_idx
from std.gpu.compute.mma import mma
from std.gpu.host import DeviceContext
from std.gpu.memory import AddressSpace
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
    # Row-contiguous operand: element (r, kk) lives at src[kk * rows + r].
    @parameter
    for it in range(CH):
        var item = tid + Int32(it * _THREADS)
        var kr = item % _BK
        var rc = (item // _BK) * 8
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
    TA: Bool, TB: Bool, BM: Int, BN: Int, FASTK: Bool, BATCHED: Bool
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

    @parameter
    @always_inline
    def load_tile(k0: Int32):
        @parameter
        if TA:
            _g2r_mc[ACH, FASTK](ap, bm0, mi, k0, ki, tid, af, va)
        else:
            _g2r_kc[ACH, FASTK](ap, bm0, mi, k0, ki, tid, af, va)

        @parameter
        if TB:
            _g2r_kc[BCH, FASTK](bp, bn0, ni, k0, ki, tid, bf, vb)
        else:
            _g2r_mc[BCH, FASTK](bp, bn0, ni, k0, ki, tid, bf, vb)

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
                if TA and FASTK:
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
                if TB or not FASTK:
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

    @parameter
    @always_inline
    def run_tiles():
        @parameter
        for i in range(2 * NT):
            acc[i] = SIMD[_F32, 4]()

        load_tile(0)
        store_tile(0)
        barrier()
        var cur = 0
        var t: Int32 = 1
        while t < kt:
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


@__name("nanogpt_bf16_gemm_nn")
def _gemm_nn(
    output: _Ptr,
    a: _Ptr,
    b: _Ptr,
    bias: _Ptr,
    m: Int,
    n: Int,
    k: Int,
    has_bias: Int,
    a_fast: Int,
    b_fast: Int,
    c_pair: Int,
    batch_count: Int,
):
    _mma_tile_impl[False, False, 128, 128, False, True](
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


@__name("nanogpt_bf16_gemm_nn_f")
def _gemm_nn_f(
    output: _Ptr,
    a: _Ptr,
    b: _Ptr,
    bias: _Ptr,
    m: Int,
    n: Int,
    k: Int,
    has_bias: Int,
    a_fast: Int,
    b_fast: Int,
    c_pair: Int,
    batch_count: Int,
):
    _mma_tile_impl[False, False, 128, 128, True, False](
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


@__name("nanogpt_bf16_gemm_nn_w64")
def _gemm_nn_w64(
    output: _Ptr,
    a: _Ptr,
    b: _Ptr,
    bias: _Ptr,
    m: Int,
    n: Int,
    k: Int,
    has_bias: Int,
    a_fast: Int,
    b_fast: Int,
    c_pair: Int,
    batch_count: Int,
):
    _mma_tile_impl[False, False, 128, 64, False, True](
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


@__name("nanogpt_bf16_gemm_nn_w64_f")
def _gemm_nn_w64_f(
    output: _Ptr,
    a: _Ptr,
    b: _Ptr,
    bias: _Ptr,
    m: Int,
    n: Int,
    k: Int,
    has_bias: Int,
    a_fast: Int,
    b_fast: Int,
    c_pair: Int,
    batch_count: Int,
):
    _mma_tile_impl[False, False, 128, 64, True, False](
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


@__name("nanogpt_bf16_gemm_nn_m64")
def _gemm_nn_m64(
    output: _Ptr,
    a: _Ptr,
    b: _Ptr,
    bias: _Ptr,
    m: Int,
    n: Int,
    k: Int,
    has_bias: Int,
    a_fast: Int,
    b_fast: Int,
    c_pair: Int,
    batch_count: Int,
):
    _mma_tile_impl[False, False, 64, 128, False, True](
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


@__name("nanogpt_bf16_gemm_nn_m64_f")
def _gemm_nn_m64_f(
    output: _Ptr,
    a: _Ptr,
    b: _Ptr,
    bias: _Ptr,
    m: Int,
    n: Int,
    k: Int,
    has_bias: Int,
    a_fast: Int,
    b_fast: Int,
    c_pair: Int,
    batch_count: Int,
):
    _mma_tile_impl[False, False, 64, 128, True, False](
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


@__name("nanogpt_bf16_gemm_nt")
def _gemm_nt(
    output: _Ptr,
    a: _Ptr,
    b: _Ptr,
    bias: _Ptr,
    m: Int,
    n: Int,
    k: Int,
    has_bias: Int,
    a_fast: Int,
    b_fast: Int,
    c_pair: Int,
    batch_count: Int,
):
    _mma_tile_impl[False, True, 128, 128, False, True](
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


@__name("nanogpt_bf16_gemm_nt_f")
def _gemm_nt_f(
    output: _Ptr,
    a: _Ptr,
    b: _Ptr,
    bias: _Ptr,
    m: Int,
    n: Int,
    k: Int,
    has_bias: Int,
    a_fast: Int,
    b_fast: Int,
    c_pair: Int,
    batch_count: Int,
):
    _mma_tile_impl[False, True, 128, 128, True, False](
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


@__name("nanogpt_bf16_gemm_nt_w64")
def _gemm_nt_w64(
    output: _Ptr,
    a: _Ptr,
    b: _Ptr,
    bias: _Ptr,
    m: Int,
    n: Int,
    k: Int,
    has_bias: Int,
    a_fast: Int,
    b_fast: Int,
    c_pair: Int,
    batch_count: Int,
):
    _mma_tile_impl[False, True, 128, 64, False, True](
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


@__name("nanogpt_bf16_gemm_nt_w64_f")
def _gemm_nt_w64_f(
    output: _Ptr,
    a: _Ptr,
    b: _Ptr,
    bias: _Ptr,
    m: Int,
    n: Int,
    k: Int,
    has_bias: Int,
    a_fast: Int,
    b_fast: Int,
    c_pair: Int,
    batch_count: Int,
):
    _mma_tile_impl[False, True, 128, 64, True, False](
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


@__name("nanogpt_bf16_gemm_nt_m64")
def _gemm_nt_m64(
    output: _Ptr,
    a: _Ptr,
    b: _Ptr,
    bias: _Ptr,
    m: Int,
    n: Int,
    k: Int,
    has_bias: Int,
    a_fast: Int,
    b_fast: Int,
    c_pair: Int,
    batch_count: Int,
):
    _mma_tile_impl[False, True, 64, 128, False, True](
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


@__name("nanogpt_bf16_gemm_nt_m64_f")
def _gemm_nt_m64_f(
    output: _Ptr,
    a: _Ptr,
    b: _Ptr,
    bias: _Ptr,
    m: Int,
    n: Int,
    k: Int,
    has_bias: Int,
    a_fast: Int,
    b_fast: Int,
    c_pair: Int,
    batch_count: Int,
):
    _mma_tile_impl[False, True, 64, 128, True, False](
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


@__name("nanogpt_bf16_gemm_tn")
def _gemm_tn(
    output: _Ptr,
    a: _Ptr,
    b: _Ptr,
    bias: _Ptr,
    m: Int,
    n: Int,
    k: Int,
    has_bias: Int,
    a_fast: Int,
    b_fast: Int,
    c_pair: Int,
    batch_count: Int,
):
    _mma_tile_impl[True, False, 128, 128, False, True](
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


@__name("nanogpt_bf16_gemm_tn_f")
def _gemm_tn_f(
    output: _Ptr,
    a: _Ptr,
    b: _Ptr,
    bias: _Ptr,
    m: Int,
    n: Int,
    k: Int,
    has_bias: Int,
    a_fast: Int,
    b_fast: Int,
    c_pair: Int,
    batch_count: Int,
):
    _mma_tile_impl[True, False, 128, 128, True, False](
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


@__name("nanogpt_bf16_gemm_tn_w64")
def _gemm_tn_w64(
    output: _Ptr,
    a: _Ptr,
    b: _Ptr,
    bias: _Ptr,
    m: Int,
    n: Int,
    k: Int,
    has_bias: Int,
    a_fast: Int,
    b_fast: Int,
    c_pair: Int,
    batch_count: Int,
):
    _mma_tile_impl[True, False, 128, 64, False, True](
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


@__name("nanogpt_bf16_gemm_tn_w64_f")
def _gemm_tn_w64_f(
    output: _Ptr,
    a: _Ptr,
    b: _Ptr,
    bias: _Ptr,
    m: Int,
    n: Int,
    k: Int,
    has_bias: Int,
    a_fast: Int,
    b_fast: Int,
    c_pair: Int,
    batch_count: Int,
):
    _mma_tile_impl[True, False, 128, 64, True, False](
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


@__name("nanogpt_bf16_gemm_tn_m64")
def _gemm_tn_m64(
    output: _Ptr,
    a: _Ptr,
    b: _Ptr,
    bias: _Ptr,
    m: Int,
    n: Int,
    k: Int,
    has_bias: Int,
    a_fast: Int,
    b_fast: Int,
    c_pair: Int,
    batch_count: Int,
):
    _mma_tile_impl[True, False, 64, 128, False, True](
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


@__name("nanogpt_bf16_gemm_tn_m64_f")
def _gemm_tn_m64_f(
    output: _Ptr,
    a: _Ptr,
    b: _Ptr,
    bias: _Ptr,
    m: Int,
    n: Int,
    k: Int,
    has_bias: Int,
    a_fast: Int,
    b_fast: Int,
    c_pair: Int,
    batch_count: Int,
):
    _mma_tile_impl[True, False, 64, 128, True, False](
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


@__name("nanogpt_bf16_gemm_tt")
def _gemm_tt(
    output: _Ptr,
    a: _Ptr,
    b: _Ptr,
    bias: _Ptr,
    m: Int,
    n: Int,
    k: Int,
    has_bias: Int,
    a_fast: Int,
    b_fast: Int,
    c_pair: Int,
    batch_count: Int,
):
    _mma_tile_impl[True, True, 128, 128, False, True](
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


@__name("nanogpt_bf16_gemm_tt_f")
def _gemm_tt_f(
    output: _Ptr,
    a: _Ptr,
    b: _Ptr,
    bias: _Ptr,
    m: Int,
    n: Int,
    k: Int,
    has_bias: Int,
    a_fast: Int,
    b_fast: Int,
    c_pair: Int,
    batch_count: Int,
):
    _mma_tile_impl[True, True, 128, 128, True, False](
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


@__name("nanogpt_bf16_gemm_tt_w64")
def _gemm_tt_w64(
    output: _Ptr,
    a: _Ptr,
    b: _Ptr,
    bias: _Ptr,
    m: Int,
    n: Int,
    k: Int,
    has_bias: Int,
    a_fast: Int,
    b_fast: Int,
    c_pair: Int,
    batch_count: Int,
):
    _mma_tile_impl[True, True, 128, 64, False, True](
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


@__name("nanogpt_bf16_gemm_tt_w64_f")
def _gemm_tt_w64_f(
    output: _Ptr,
    a: _Ptr,
    b: _Ptr,
    bias: _Ptr,
    m: Int,
    n: Int,
    k: Int,
    has_bias: Int,
    a_fast: Int,
    b_fast: Int,
    c_pair: Int,
    batch_count: Int,
):
    _mma_tile_impl[True, True, 128, 64, True, False](
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


@__name("nanogpt_bf16_gemm_tt_m64")
def _gemm_tt_m64(
    output: _Ptr,
    a: _Ptr,
    b: _Ptr,
    bias: _Ptr,
    m: Int,
    n: Int,
    k: Int,
    has_bias: Int,
    a_fast: Int,
    b_fast: Int,
    c_pair: Int,
    batch_count: Int,
):
    _mma_tile_impl[True, True, 64, 128, False, True](
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


@__name("nanogpt_bf16_gemm_tt_m64_f")
def _gemm_tt_m64_f(
    output: _Ptr,
    a: _Ptr,
    b: _Ptr,
    bias: _Ptr,
    m: Int,
    n: Int,
    k: Int,
    has_bias: Int,
    a_fast: Int,
    b_fast: Int,
    c_pair: Int,
    batch_count: Int,
):
    _mma_tile_impl[True, True, 64, 128, True, False](
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


@__name("nanogpt_bf16_bmm_nn")
def _bmm_nn(
    output: _Ptr,
    a: _Ptr,
    b: _Ptr,
    m: Int,
    n: Int,
    k: Int,
    c_bstride: Int,
    a_bstride: Int,
    b_bstride: Int,
    a_fast: Int,
    b_fast: Int,
    c_pair: Int,
    batch_count: Int,
):
    _mma_tile_impl[False, False, 128, 128, False, True](
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


@__name("nanogpt_bf16_bmm_nn_f")
def _bmm_nn_f(
    output: _Ptr,
    a: _Ptr,
    b: _Ptr,
    m: Int,
    n: Int,
    k: Int,
    c_bstride: Int,
    a_bstride: Int,
    b_bstride: Int,
    a_fast: Int,
    b_fast: Int,
    c_pair: Int,
    batch_count: Int,
):
    _mma_tile_impl[False, False, 128, 128, True, False](
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


@__name("nanogpt_bf16_bmm_nn_w64")
def _bmm_nn_w64(
    output: _Ptr,
    a: _Ptr,
    b: _Ptr,
    m: Int,
    n: Int,
    k: Int,
    c_bstride: Int,
    a_bstride: Int,
    b_bstride: Int,
    a_fast: Int,
    b_fast: Int,
    c_pair: Int,
    batch_count: Int,
):
    _mma_tile_impl[False, False, 128, 64, False, True](
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


@__name("nanogpt_bf16_bmm_nn_w64_f")
def _bmm_nn_w64_f(
    output: _Ptr,
    a: _Ptr,
    b: _Ptr,
    m: Int,
    n: Int,
    k: Int,
    c_bstride: Int,
    a_bstride: Int,
    b_bstride: Int,
    a_fast: Int,
    b_fast: Int,
    c_pair: Int,
    batch_count: Int,
):
    _mma_tile_impl[False, False, 128, 64, True, False](
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


@__name("nanogpt_bf16_bmm_nn_m64")
def _bmm_nn_m64(
    output: _Ptr,
    a: _Ptr,
    b: _Ptr,
    m: Int,
    n: Int,
    k: Int,
    c_bstride: Int,
    a_bstride: Int,
    b_bstride: Int,
    a_fast: Int,
    b_fast: Int,
    c_pair: Int,
    batch_count: Int,
):
    _mma_tile_impl[False, False, 64, 128, False, True](
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


@__name("nanogpt_bf16_bmm_nn_m64_f")
def _bmm_nn_m64_f(
    output: _Ptr,
    a: _Ptr,
    b: _Ptr,
    m: Int,
    n: Int,
    k: Int,
    c_bstride: Int,
    a_bstride: Int,
    b_bstride: Int,
    a_fast: Int,
    b_fast: Int,
    c_pair: Int,
    batch_count: Int,
):
    _mma_tile_impl[False, False, 64, 128, True, False](
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


@__name("nanogpt_bf16_bmm_nt")
def _bmm_nt(
    output: _Ptr,
    a: _Ptr,
    b: _Ptr,
    m: Int,
    n: Int,
    k: Int,
    c_bstride: Int,
    a_bstride: Int,
    b_bstride: Int,
    a_fast: Int,
    b_fast: Int,
    c_pair: Int,
    batch_count: Int,
):
    _mma_tile_impl[False, True, 128, 128, False, True](
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


@__name("nanogpt_bf16_bmm_nt_f")
def _bmm_nt_f(
    output: _Ptr,
    a: _Ptr,
    b: _Ptr,
    m: Int,
    n: Int,
    k: Int,
    c_bstride: Int,
    a_bstride: Int,
    b_bstride: Int,
    a_fast: Int,
    b_fast: Int,
    c_pair: Int,
    batch_count: Int,
):
    _mma_tile_impl[False, True, 128, 128, True, False](
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


@__name("nanogpt_bf16_bmm_nt_w64")
def _bmm_nt_w64(
    output: _Ptr,
    a: _Ptr,
    b: _Ptr,
    m: Int,
    n: Int,
    k: Int,
    c_bstride: Int,
    a_bstride: Int,
    b_bstride: Int,
    a_fast: Int,
    b_fast: Int,
    c_pair: Int,
    batch_count: Int,
):
    _mma_tile_impl[False, True, 128, 64, False, True](
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


@__name("nanogpt_bf16_bmm_nt_w64_f")
def _bmm_nt_w64_f(
    output: _Ptr,
    a: _Ptr,
    b: _Ptr,
    m: Int,
    n: Int,
    k: Int,
    c_bstride: Int,
    a_bstride: Int,
    b_bstride: Int,
    a_fast: Int,
    b_fast: Int,
    c_pair: Int,
    batch_count: Int,
):
    _mma_tile_impl[False, True, 128, 64, True, False](
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


@__name("nanogpt_bf16_bmm_nt_m64")
def _bmm_nt_m64(
    output: _Ptr,
    a: _Ptr,
    b: _Ptr,
    m: Int,
    n: Int,
    k: Int,
    c_bstride: Int,
    a_bstride: Int,
    b_bstride: Int,
    a_fast: Int,
    b_fast: Int,
    c_pair: Int,
    batch_count: Int,
):
    _mma_tile_impl[False, True, 64, 128, False, True](
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


@__name("nanogpt_bf16_bmm_nt_m64_f")
def _bmm_nt_m64_f(
    output: _Ptr,
    a: _Ptr,
    b: _Ptr,
    m: Int,
    n: Int,
    k: Int,
    c_bstride: Int,
    a_bstride: Int,
    b_bstride: Int,
    a_fast: Int,
    b_fast: Int,
    c_pair: Int,
    batch_count: Int,
):
    _mma_tile_impl[False, True, 64, 128, True, False](
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


@__name("nanogpt_bf16_bmm_tn")
def _bmm_tn(
    output: _Ptr,
    a: _Ptr,
    b: _Ptr,
    m: Int,
    n: Int,
    k: Int,
    c_bstride: Int,
    a_bstride: Int,
    b_bstride: Int,
    a_fast: Int,
    b_fast: Int,
    c_pair: Int,
    batch_count: Int,
):
    _mma_tile_impl[True, False, 128, 128, False, True](
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


@__name("nanogpt_bf16_bmm_tn_f")
def _bmm_tn_f(
    output: _Ptr,
    a: _Ptr,
    b: _Ptr,
    m: Int,
    n: Int,
    k: Int,
    c_bstride: Int,
    a_bstride: Int,
    b_bstride: Int,
    a_fast: Int,
    b_fast: Int,
    c_pair: Int,
    batch_count: Int,
):
    _mma_tile_impl[True, False, 128, 128, True, False](
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


@__name("nanogpt_bf16_bmm_tn_w64")
def _bmm_tn_w64(
    output: _Ptr,
    a: _Ptr,
    b: _Ptr,
    m: Int,
    n: Int,
    k: Int,
    c_bstride: Int,
    a_bstride: Int,
    b_bstride: Int,
    a_fast: Int,
    b_fast: Int,
    c_pair: Int,
    batch_count: Int,
):
    _mma_tile_impl[True, False, 128, 64, False, True](
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


@__name("nanogpt_bf16_bmm_tn_w64_f")
def _bmm_tn_w64_f(
    output: _Ptr,
    a: _Ptr,
    b: _Ptr,
    m: Int,
    n: Int,
    k: Int,
    c_bstride: Int,
    a_bstride: Int,
    b_bstride: Int,
    a_fast: Int,
    b_fast: Int,
    c_pair: Int,
    batch_count: Int,
):
    _mma_tile_impl[True, False, 128, 64, True, False](
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


@__name("nanogpt_bf16_bmm_tn_m64")
def _bmm_tn_m64(
    output: _Ptr,
    a: _Ptr,
    b: _Ptr,
    m: Int,
    n: Int,
    k: Int,
    c_bstride: Int,
    a_bstride: Int,
    b_bstride: Int,
    a_fast: Int,
    b_fast: Int,
    c_pair: Int,
    batch_count: Int,
):
    _mma_tile_impl[True, False, 64, 128, False, True](
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


@__name("nanogpt_bf16_bmm_tn_m64_f")
def _bmm_tn_m64_f(
    output: _Ptr,
    a: _Ptr,
    b: _Ptr,
    m: Int,
    n: Int,
    k: Int,
    c_bstride: Int,
    a_bstride: Int,
    b_bstride: Int,
    a_fast: Int,
    b_fast: Int,
    c_pair: Int,
    batch_count: Int,
):
    _mma_tile_impl[True, False, 64, 128, True, False](
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


@__name("nanogpt_bf16_bmm_tt")
def _bmm_tt(
    output: _Ptr,
    a: _Ptr,
    b: _Ptr,
    m: Int,
    n: Int,
    k: Int,
    c_bstride: Int,
    a_bstride: Int,
    b_bstride: Int,
    a_fast: Int,
    b_fast: Int,
    c_pair: Int,
    batch_count: Int,
):
    _mma_tile_impl[True, True, 128, 128, False, True](
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


@__name("nanogpt_bf16_bmm_tt_f")
def _bmm_tt_f(
    output: _Ptr,
    a: _Ptr,
    b: _Ptr,
    m: Int,
    n: Int,
    k: Int,
    c_bstride: Int,
    a_bstride: Int,
    b_bstride: Int,
    a_fast: Int,
    b_fast: Int,
    c_pair: Int,
    batch_count: Int,
):
    _mma_tile_impl[True, True, 128, 128, True, False](
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


@__name("nanogpt_bf16_bmm_tt_w64")
def _bmm_tt_w64(
    output: _Ptr,
    a: _Ptr,
    b: _Ptr,
    m: Int,
    n: Int,
    k: Int,
    c_bstride: Int,
    a_bstride: Int,
    b_bstride: Int,
    a_fast: Int,
    b_fast: Int,
    c_pair: Int,
    batch_count: Int,
):
    _mma_tile_impl[True, True, 128, 64, False, True](
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


@__name("nanogpt_bf16_bmm_tt_w64_f")
def _bmm_tt_w64_f(
    output: _Ptr,
    a: _Ptr,
    b: _Ptr,
    m: Int,
    n: Int,
    k: Int,
    c_bstride: Int,
    a_bstride: Int,
    b_bstride: Int,
    a_fast: Int,
    b_fast: Int,
    c_pair: Int,
    batch_count: Int,
):
    _mma_tile_impl[True, True, 128, 64, True, False](
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


@__name("nanogpt_bf16_bmm_tt_m64")
def _bmm_tt_m64(
    output: _Ptr,
    a: _Ptr,
    b: _Ptr,
    m: Int,
    n: Int,
    k: Int,
    c_bstride: Int,
    a_bstride: Int,
    b_bstride: Int,
    a_fast: Int,
    b_fast: Int,
    c_pair: Int,
    batch_count: Int,
):
    _mma_tile_impl[True, True, 64, 128, False, True](
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


@__name("nanogpt_bf16_bmm_tt_m64_f")
def _bmm_tt_m64_f(
    output: _Ptr,
    a: _Ptr,
    b: _Ptr,
    m: Int,
    n: Int,
    k: Int,
    c_bstride: Int,
    a_bstride: Int,
    b_bstride: Int,
    a_fast: Int,
    b_fast: Int,
    c_pair: Int,
    batch_count: Int,
):
    _mma_tile_impl[True, True, 64, 128, True, False](
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


@__name("nanogpt_bf16_gemm_nn_wide")
def _gemm_nn_wide(
    output: _Ptr,
    a: _Ptr,
    b: _Ptr,
    bias: _Ptr,
    m: Int,
    n: Int,
    k: Int,
    has_bias: Int,
    a_fast: Int,
    b_fast: Int,
    c_pair: Int,
    batch_count: Int,
):
    _mma_tile_wide[False, False](
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


@__name("nanogpt_bf16_gemm_nt_wide")
def _gemm_nt_wide(
    output: _Ptr,
    a: _Ptr,
    b: _Ptr,
    bias: _Ptr,
    m: Int,
    n: Int,
    k: Int,
    has_bias: Int,
    a_fast: Int,
    b_fast: Int,
    c_pair: Int,
    batch_count: Int,
):
    _mma_tile_wide[False, True](
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


@__name("nanogpt_bf16_gemm_tn_wide")
def _gemm_tn_wide(
    output: _Ptr,
    a: _Ptr,
    b: _Ptr,
    bias: _Ptr,
    m: Int,
    n: Int,
    k: Int,
    has_bias: Int,
    a_fast: Int,
    b_fast: Int,
    c_pair: Int,
    batch_count: Int,
):
    _mma_tile_wide[True, False](
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


@__name("nanogpt_bf16_gemm_tt_wide")
def _gemm_tt_wide(
    output: _Ptr,
    a: _Ptr,
    b: _Ptr,
    bias: _Ptr,
    m: Int,
    n: Int,
    k: Int,
    has_bias: Int,
    a_fast: Int,
    b_fast: Int,
    c_pair: Int,
    batch_count: Int,
):
    _mma_tile_wide[True, True](
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


@__name("nanogpt_bf16_bmm_nn_wide")
def _bmm_nn_wide(
    output: _Ptr,
    a: _Ptr,
    b: _Ptr,
    m: Int,
    n: Int,
    k: Int,
    c_bstride: Int,
    a_bstride: Int,
    b_bstride: Int,
    a_fast: Int,
    b_fast: Int,
    c_pair: Int,
    batch_count: Int,
):
    _mma_tile_wide[False, False](
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


@__name("nanogpt_bf16_bmm_nt_wide")
def _bmm_nt_wide(
    output: _Ptr,
    a: _Ptr,
    b: _Ptr,
    m: Int,
    n: Int,
    k: Int,
    c_bstride: Int,
    a_bstride: Int,
    b_bstride: Int,
    a_fast: Int,
    b_fast: Int,
    c_pair: Int,
    batch_count: Int,
):
    _mma_tile_wide[False, True](
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


@__name("nanogpt_bf16_bmm_tn_wide")
def _bmm_tn_wide(
    output: _Ptr,
    a: _Ptr,
    b: _Ptr,
    m: Int,
    n: Int,
    k: Int,
    c_bstride: Int,
    a_bstride: Int,
    b_bstride: Int,
    a_fast: Int,
    b_fast: Int,
    c_pair: Int,
    batch_count: Int,
):
    _mma_tile_wide[True, False](
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


@__name("nanogpt_bf16_bmm_tt_wide")
def _bmm_tt_wide(
    output: _Ptr,
    a: _Ptr,
    b: _Ptr,
    m: Int,
    n: Int,
    k: Int,
    c_bstride: Int,
    a_bstride: Int,
    b_bstride: Int,
    a_fast: Int,
    b_fast: Int,
    c_pair: Int,
    batch_count: Int,
):
    _mma_tile_wide[True, True](
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
def _pick_regime(m: Int, n: Int) -> Int:
    # 0: 128x128, 1: 128x64, 2: 64x128. Narrow tiles only where the wide
    # tile wastes at least half of one extent (small n or small m); the wide
    # tile's compute-to-traffic ratio wins elsewhere, including long-K
    # weight-gradient shapes where halving a tile doubles operand re-reads.
    if n <= 64:
        return 1
    if m <= 64:
        return 2
    return 0


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
    if transpose_a:
        if transpose_b:
            ctx.enqueue_function[_gemm_tt_wide](
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
                1,
                grid_dim=(grid_x,),
                block_dim=(_THREADS,),
            )
        else:
            ctx.enqueue_function[_gemm_tn_wide](
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
                1,
                grid_dim=(grid_x,),
                block_dim=(_THREADS,),
            )
    else:
        if transpose_b:
            ctx.enqueue_function[_gemm_nt_wide](
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
                1,
                grid_dim=(grid_x,),
                block_dim=(_THREADS,),
            )
        else:
            ctx.enqueue_function[_gemm_nn_wide](
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
                1,
                grid_dim=(grid_x,),
                block_dim=(_THREADS,),
            )


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
    if transpose_a:
        if transpose_b:
            ctx.enqueue_function[_bmm_tt_wide](
                output,
                a,
                b,
                m,
                n,
                k,
                c_bstride,
                a_bstride,
                b_bstride,
                a_fast,
                b_fast,
                c_pair,
                batch_count,
                grid_dim=(grid_x, 1, grid_z),
                block_dim=(_THREADS,),
            )
        else:
            ctx.enqueue_function[_bmm_tn_wide](
                output,
                a,
                b,
                m,
                n,
                k,
                c_bstride,
                a_bstride,
                b_bstride,
                a_fast,
                b_fast,
                c_pair,
                batch_count,
                grid_dim=(grid_x, 1, grid_z),
                block_dim=(_THREADS,),
            )
    else:
        if transpose_b:
            ctx.enqueue_function[_bmm_nt_wide](
                output,
                a,
                b,
                m,
                n,
                k,
                c_bstride,
                a_bstride,
                b_bstride,
                a_fast,
                b_fast,
                c_pair,
                batch_count,
                grid_dim=(grid_x, 1, grid_z),
                block_dim=(_THREADS,),
            )
        else:
            ctx.enqueue_function[_bmm_nn_wide](
                output,
                a,
                b,
                m,
                n,
                k,
                c_bstride,
                a_bstride,
                b_bstride,
                a_fast,
                b_fast,
                c_pair,
                batch_count,
                grid_dim=(grid_x, 1, grid_z),
                block_dim=(_THREADS,),
            )


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
    var regime = _pick_regime(m, n)
    var bm = 64 if regime == 2 else 128
    var bn = 64 if regime == 1 else 128
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
    var fastk = _fast_proof(
        output, a, b, m, n, k, 0, 0, 0, transpose_a, transpose_b
    )
    # Below one block per SM (132 on H100) occupancy cannot rise, and the
    # guarded kernel's larger register budget buys more ILP per block, so
    # the low-register FASTK build only wins past that point.
    fastk = fastk and grid_x > 132
    if fastk:
        if regime == 1:
            if transpose_a:
                if transpose_b:
                    ctx.enqueue_function[_gemm_tt_w64_f](
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
                        1,
                        grid_dim=(grid_x,),
                        block_dim=(_THREADS,),
                    )
                else:
                    ctx.enqueue_function[_gemm_tn_w64_f](
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
                        1,
                        grid_dim=(grid_x,),
                        block_dim=(_THREADS,),
                    )
            else:
                if transpose_b:
                    ctx.enqueue_function[_gemm_nt_w64_f](
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
                        1,
                        grid_dim=(grid_x,),
                        block_dim=(_THREADS,),
                    )
                else:
                    ctx.enqueue_function[_gemm_nn_w64_f](
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
                        1,
                        grid_dim=(grid_x,),
                        block_dim=(_THREADS,),
                    )
        elif regime == 2:
            if transpose_a:
                if transpose_b:
                    ctx.enqueue_function[_gemm_tt_m64_f](
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
                        1,
                        grid_dim=(grid_x,),
                        block_dim=(_THREADS,),
                    )
                else:
                    ctx.enqueue_function[_gemm_tn_m64_f](
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
                        1,
                        grid_dim=(grid_x,),
                        block_dim=(_THREADS,),
                    )
            else:
                if transpose_b:
                    ctx.enqueue_function[_gemm_nt_m64_f](
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
                        1,
                        grid_dim=(grid_x,),
                        block_dim=(_THREADS,),
                    )
                else:
                    ctx.enqueue_function[_gemm_nn_m64_f](
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
                        1,
                        grid_dim=(grid_x,),
                        block_dim=(_THREADS,),
                    )
        else:
            if transpose_a:
                if transpose_b:
                    ctx.enqueue_function[_gemm_tt_f](
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
                        1,
                        grid_dim=(grid_x,),
                        block_dim=(_THREADS,),
                    )
                else:
                    ctx.enqueue_function[_gemm_tn_f](
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
                        1,
                        grid_dim=(grid_x,),
                        block_dim=(_THREADS,),
                    )
            else:
                if transpose_b:
                    ctx.enqueue_function[_gemm_nt_f](
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
                        1,
                        grid_dim=(grid_x,),
                        block_dim=(_THREADS,),
                    )
                else:
                    ctx.enqueue_function[_gemm_nn_f](
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
                        1,
                        grid_dim=(grid_x,),
                        block_dim=(_THREADS,),
                    )
    else:
        if regime == 1:
            if transpose_a:
                if transpose_b:
                    ctx.enqueue_function[_gemm_tt_w64](
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
                        1,
                        grid_dim=(grid_x,),
                        block_dim=(_THREADS,),
                    )
                else:
                    ctx.enqueue_function[_gemm_tn_w64](
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
                        1,
                        grid_dim=(grid_x,),
                        block_dim=(_THREADS,),
                    )
            else:
                if transpose_b:
                    ctx.enqueue_function[_gemm_nt_w64](
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
                        1,
                        grid_dim=(grid_x,),
                        block_dim=(_THREADS,),
                    )
                else:
                    ctx.enqueue_function[_gemm_nn_w64](
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
                        1,
                        grid_dim=(grid_x,),
                        block_dim=(_THREADS,),
                    )
        elif regime == 2:
            if transpose_a:
                if transpose_b:
                    ctx.enqueue_function[_gemm_tt_m64](
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
                        1,
                        grid_dim=(grid_x,),
                        block_dim=(_THREADS,),
                    )
                else:
                    ctx.enqueue_function[_gemm_tn_m64](
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
                        1,
                        grid_dim=(grid_x,),
                        block_dim=(_THREADS,),
                    )
            else:
                if transpose_b:
                    ctx.enqueue_function[_gemm_nt_m64](
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
                        1,
                        grid_dim=(grid_x,),
                        block_dim=(_THREADS,),
                    )
                else:
                    ctx.enqueue_function[_gemm_nn_m64](
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
                        1,
                        grid_dim=(grid_x,),
                        block_dim=(_THREADS,),
                    )
        else:
            if transpose_a:
                if transpose_b:
                    ctx.enqueue_function[_gemm_tt](
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
                        1,
                        grid_dim=(grid_x,),
                        block_dim=(_THREADS,),
                    )
                else:
                    ctx.enqueue_function[_gemm_tn](
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
                        1,
                        grid_dim=(grid_x,),
                        block_dim=(_THREADS,),
                    )
            else:
                if transpose_b:
                    ctx.enqueue_function[_gemm_nt](
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
                        1,
                        grid_dim=(grid_x,),
                        block_dim=(_THREADS,),
                    )
                else:
                    ctx.enqueue_function[_gemm_nn](
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
                        1,
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
    var regime = _pick_regime(m, n)
    var bm = 64 if regime == 2 else 128
    var bn = 64 if regime == 1 else 128
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
    if fastk:
        if regime == 1:
            if transpose_a:
                if transpose_b:
                    ctx.enqueue_function[_bmm_tt_w64_f](
                        output,
                        a,
                        b,
                        m,
                        n,
                        k,
                        output_batch_stride,
                        a_batch_stride,
                        b_batch_stride,
                        a_fast,
                        b_fast,
                        c_pair,
                        batch_count,
                        grid_dim=(grid_x, 1, grid_z),
                        block_dim=(_THREADS,),
                    )
                else:
                    ctx.enqueue_function[_bmm_tn_w64_f](
                        output,
                        a,
                        b,
                        m,
                        n,
                        k,
                        output_batch_stride,
                        a_batch_stride,
                        b_batch_stride,
                        a_fast,
                        b_fast,
                        c_pair,
                        batch_count,
                        grid_dim=(grid_x, 1, grid_z),
                        block_dim=(_THREADS,),
                    )
            else:
                if transpose_b:
                    ctx.enqueue_function[_bmm_nt_w64_f](
                        output,
                        a,
                        b,
                        m,
                        n,
                        k,
                        output_batch_stride,
                        a_batch_stride,
                        b_batch_stride,
                        a_fast,
                        b_fast,
                        c_pair,
                        batch_count,
                        grid_dim=(grid_x, 1, grid_z),
                        block_dim=(_THREADS,),
                    )
                else:
                    ctx.enqueue_function[_bmm_nn_w64_f](
                        output,
                        a,
                        b,
                        m,
                        n,
                        k,
                        output_batch_stride,
                        a_batch_stride,
                        b_batch_stride,
                        a_fast,
                        b_fast,
                        c_pair,
                        batch_count,
                        grid_dim=(grid_x, 1, grid_z),
                        block_dim=(_THREADS,),
                    )
        elif regime == 2:
            if transpose_a:
                if transpose_b:
                    ctx.enqueue_function[_bmm_tt_m64_f](
                        output,
                        a,
                        b,
                        m,
                        n,
                        k,
                        output_batch_stride,
                        a_batch_stride,
                        b_batch_stride,
                        a_fast,
                        b_fast,
                        c_pair,
                        batch_count,
                        grid_dim=(grid_x, 1, grid_z),
                        block_dim=(_THREADS,),
                    )
                else:
                    ctx.enqueue_function[_bmm_tn_m64_f](
                        output,
                        a,
                        b,
                        m,
                        n,
                        k,
                        output_batch_stride,
                        a_batch_stride,
                        b_batch_stride,
                        a_fast,
                        b_fast,
                        c_pair,
                        batch_count,
                        grid_dim=(grid_x, 1, grid_z),
                        block_dim=(_THREADS,),
                    )
            else:
                if transpose_b:
                    ctx.enqueue_function[_bmm_nt_m64_f](
                        output,
                        a,
                        b,
                        m,
                        n,
                        k,
                        output_batch_stride,
                        a_batch_stride,
                        b_batch_stride,
                        a_fast,
                        b_fast,
                        c_pair,
                        batch_count,
                        grid_dim=(grid_x, 1, grid_z),
                        block_dim=(_THREADS,),
                    )
                else:
                    ctx.enqueue_function[_bmm_nn_m64_f](
                        output,
                        a,
                        b,
                        m,
                        n,
                        k,
                        output_batch_stride,
                        a_batch_stride,
                        b_batch_stride,
                        a_fast,
                        b_fast,
                        c_pair,
                        batch_count,
                        grid_dim=(grid_x, 1, grid_z),
                        block_dim=(_THREADS,),
                    )
        else:
            if transpose_a:
                if transpose_b:
                    ctx.enqueue_function[_bmm_tt_f](
                        output,
                        a,
                        b,
                        m,
                        n,
                        k,
                        output_batch_stride,
                        a_batch_stride,
                        b_batch_stride,
                        a_fast,
                        b_fast,
                        c_pair,
                        batch_count,
                        grid_dim=(grid_x, 1, grid_z),
                        block_dim=(_THREADS,),
                    )
                else:
                    ctx.enqueue_function[_bmm_tn_f](
                        output,
                        a,
                        b,
                        m,
                        n,
                        k,
                        output_batch_stride,
                        a_batch_stride,
                        b_batch_stride,
                        a_fast,
                        b_fast,
                        c_pair,
                        batch_count,
                        grid_dim=(grid_x, 1, grid_z),
                        block_dim=(_THREADS,),
                    )
            else:
                if transpose_b:
                    ctx.enqueue_function[_bmm_nt_f](
                        output,
                        a,
                        b,
                        m,
                        n,
                        k,
                        output_batch_stride,
                        a_batch_stride,
                        b_batch_stride,
                        a_fast,
                        b_fast,
                        c_pair,
                        batch_count,
                        grid_dim=(grid_x, 1, grid_z),
                        block_dim=(_THREADS,),
                    )
                else:
                    ctx.enqueue_function[_bmm_nn_f](
                        output,
                        a,
                        b,
                        m,
                        n,
                        k,
                        output_batch_stride,
                        a_batch_stride,
                        b_batch_stride,
                        a_fast,
                        b_fast,
                        c_pair,
                        batch_count,
                        grid_dim=(grid_x, 1, grid_z),
                        block_dim=(_THREADS,),
                    )
    else:
        if regime == 1:
            if transpose_a:
                if transpose_b:
                    ctx.enqueue_function[_bmm_tt_w64](
                        output,
                        a,
                        b,
                        m,
                        n,
                        k,
                        output_batch_stride,
                        a_batch_stride,
                        b_batch_stride,
                        a_fast,
                        b_fast,
                        c_pair,
                        batch_count,
                        grid_dim=(grid_x, 1, grid_z),
                        block_dim=(_THREADS,),
                    )
                else:
                    ctx.enqueue_function[_bmm_tn_w64](
                        output,
                        a,
                        b,
                        m,
                        n,
                        k,
                        output_batch_stride,
                        a_batch_stride,
                        b_batch_stride,
                        a_fast,
                        b_fast,
                        c_pair,
                        batch_count,
                        grid_dim=(grid_x, 1, grid_z),
                        block_dim=(_THREADS,),
                    )
            else:
                if transpose_b:
                    ctx.enqueue_function[_bmm_nt_w64](
                        output,
                        a,
                        b,
                        m,
                        n,
                        k,
                        output_batch_stride,
                        a_batch_stride,
                        b_batch_stride,
                        a_fast,
                        b_fast,
                        c_pair,
                        batch_count,
                        grid_dim=(grid_x, 1, grid_z),
                        block_dim=(_THREADS,),
                    )
                else:
                    ctx.enqueue_function[_bmm_nn_w64](
                        output,
                        a,
                        b,
                        m,
                        n,
                        k,
                        output_batch_stride,
                        a_batch_stride,
                        b_batch_stride,
                        a_fast,
                        b_fast,
                        c_pair,
                        batch_count,
                        grid_dim=(grid_x, 1, grid_z),
                        block_dim=(_THREADS,),
                    )
        elif regime == 2:
            if transpose_a:
                if transpose_b:
                    ctx.enqueue_function[_bmm_tt_m64](
                        output,
                        a,
                        b,
                        m,
                        n,
                        k,
                        output_batch_stride,
                        a_batch_stride,
                        b_batch_stride,
                        a_fast,
                        b_fast,
                        c_pair,
                        batch_count,
                        grid_dim=(grid_x, 1, grid_z),
                        block_dim=(_THREADS,),
                    )
                else:
                    ctx.enqueue_function[_bmm_tn_m64](
                        output,
                        a,
                        b,
                        m,
                        n,
                        k,
                        output_batch_stride,
                        a_batch_stride,
                        b_batch_stride,
                        a_fast,
                        b_fast,
                        c_pair,
                        batch_count,
                        grid_dim=(grid_x, 1, grid_z),
                        block_dim=(_THREADS,),
                    )
            else:
                if transpose_b:
                    ctx.enqueue_function[_bmm_nt_m64](
                        output,
                        a,
                        b,
                        m,
                        n,
                        k,
                        output_batch_stride,
                        a_batch_stride,
                        b_batch_stride,
                        a_fast,
                        b_fast,
                        c_pair,
                        batch_count,
                        grid_dim=(grid_x, 1, grid_z),
                        block_dim=(_THREADS,),
                    )
                else:
                    ctx.enqueue_function[_bmm_nn_m64](
                        output,
                        a,
                        b,
                        m,
                        n,
                        k,
                        output_batch_stride,
                        a_batch_stride,
                        b_batch_stride,
                        a_fast,
                        b_fast,
                        c_pair,
                        batch_count,
                        grid_dim=(grid_x, 1, grid_z),
                        block_dim=(_THREADS,),
                    )
        else:
            if transpose_a:
                if transpose_b:
                    ctx.enqueue_function[_bmm_tt](
                        output,
                        a,
                        b,
                        m,
                        n,
                        k,
                        output_batch_stride,
                        a_batch_stride,
                        b_batch_stride,
                        a_fast,
                        b_fast,
                        c_pair,
                        batch_count,
                        grid_dim=(grid_x, 1, grid_z),
                        block_dim=(_THREADS,),
                    )
                else:
                    ctx.enqueue_function[_bmm_tn](
                        output,
                        a,
                        b,
                        m,
                        n,
                        k,
                        output_batch_stride,
                        a_batch_stride,
                        b_batch_stride,
                        a_fast,
                        b_fast,
                        c_pair,
                        batch_count,
                        grid_dim=(grid_x, 1, grid_z),
                        block_dim=(_THREADS,),
                    )
            else:
                if transpose_b:
                    ctx.enqueue_function[_bmm_nt](
                        output,
                        a,
                        b,
                        m,
                        n,
                        k,
                        output_batch_stride,
                        a_batch_stride,
                        b_batch_stride,
                        a_fast,
                        b_fast,
                        c_pair,
                        batch_count,
                        grid_dim=(grid_x, 1, grid_z),
                        block_dim=(_THREADS,),
                    )
                else:
                    ctx.enqueue_function[_bmm_nn](
                        output,
                        a,
                        b,
                        m,
                        n,
                        k,
                        output_batch_stride,
                        a_batch_stride,
                        b_batch_stride,
                        a_fast,
                        b_fast,
                        c_pair,
                        batch_count,
                        grid_dim=(grid_x, 1, grid_z),
                        block_dim=(_THREADS,),
                    )
