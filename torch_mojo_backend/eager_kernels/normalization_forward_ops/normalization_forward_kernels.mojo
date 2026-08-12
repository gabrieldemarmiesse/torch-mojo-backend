"""Normalization forward device kernels: layer norm and group norm.

Both ops reduce ONE contiguous run per output row and then rewrite that run
scaled by the row's own statistics, so they are one mechanism here with two
comptime knobs — the dtype and how the affine parameters are indexed:

  * `AFFINE_COL`  gamma[j] / beta[j]                      (layer norm)
  * `AFFINE_CHAN` gamma[g*cpg + j//hxw] / beta[...]       (group norm)

and two memory regimes, chosen at launch from the row length:

  * `_norm_rows_cached_kernel` — the row FITS IN REGISTERS (16-byte vector
    loads, `vecs` vectors per lane, a comptime block width picked so no lane
    idles). One read plus one write is the whole HBM traffic; the mean and the
    variance are then two exact passes over registers, which costs nothing and
    is what ATen's own accumulator does.
  * `_norm_rows_moments_kernel` — the row does not fit, is not a whole number
    of 16-byte vectors, or sits at an unaligned address. The statistics come
    from ONE read through the shared shifted-moment scan
    (`op_utils._moments_scan_contig`, the same core `aten.var.correction`
    uses), including its adaptive re-pass when the assumed mean turns out to
    be an outlier, and the row is re-read once to write the output. Two reads
    and one write, against the three reads the predecessor block-per-row
    kernel paid.

Both write the per-row `mean` and `rstd` (always float32, whatever the input
dtype) that `aten.native_layer_norm` / `aten.native_group_norm` return and
their backward passes consume; statistics accumulate in float32 for every
input dtype, matching torch.

Apple keeps a third, warp-per-row route: its rows are short enough that a
whole block per row leaves most lanes idle and pays two block-wide barriers,
while one warp per row holds the row in registers and reduces with two
shuffles.
"""

from std.gpu import (
    MAX_THREADS_PER_BLOCK_METADATA,
    WARP_SIZE,
    block_idx,
    grid_dim,
    lane_id,
    thread_idx,
    warp_id,
)
from max.gpu.host import DeviceContext
from max.gpu.primitives import block
from std.gpu.primitives import warp
from std.math import ceildiv, max, min
from std.sys.info import has_apple_gpu_accelerator, size_of
from std.utils.static_tuple import StaticTuple

from op_utils import (
    _enqueue_cached,
    _make_ptr,
    _moment_cancels,
    _moment_partition,
    _moments_scan_contig,
    _vec16_phase,
    ieee_sqrt,
)


comptime _MAX_GRID = 65535

# Affine indexing modes (comptime): see the module docstring.
comptime AFFINE_COL = 0
comptime AFFINE_CHAN = 1

# Block width of the streaming (non-cacheable) route. 256 is what every other
# block-per-row kernel in this package uses, and the route is bandwidth-bound
# on rows long enough that the exact width does not select the performance.
comptime NORM_THREADS = 256

# Widths the cached route may pick from, smallest first: the launcher takes the
# first one that gives every lane a vector, so a 1024-column bf16 row runs
# 128 lanes x one 8-element vector instead of 256 lanes half of which idle.
comptime _CACHED_WIDTHS = [128, 256, 512, 1024]
comptime _CACHED_MAX_WIDTH = 1024
# Vectors per lane at the widest block: the cached route therefore reaches
# `1024 * 2 * V` columns, i.e. 8192 f32 / 16384 bf16 — wider than any layer-
# norm or group-norm row a real model has. Past it a row would need more
# registers than it is worth and the streaming route takes over.
comptime _CACHED_MAX_VECS = 2

# Warp-per-row fast path (Apple): see the module docstring.
comptime _WARPS_PER_BLOCK = 8 if WARP_SIZE <= 32 else 4


@always_inline
def _affine_vec_ok[
    V: Int, affine: Int, ragged: Bool
](col0: Int, hxw: Int, buffers_aligned: Bool) -> Bool:
    """Whether the V affine coefficients for columns `[col0, col0 + V)` can be
    taken in one go.

    `AFFINE_COL` needs a genuine 16-byte boundary of the affine buffer, which
    is a per-ROW property: a row whose base is not a multiple of V shifts
    every one of its vectors. `AFFINE_CHAN` instead needs the whole vector to
    fall inside ONE channel, which a `hxw` that is not a multiple of V breaks
    for some vectors and not others.

    Both hold unconditionally on the non-ragged route, whose launch condition
    is exactly that every row starts on a 16-byte boundary with 16-byte
    aligned affine buffers and (for the per-channel affine) `hxw % V == 0`;
    the comptime answer there keeps the gather arm out of the hot kernel
    entirely.
    """
    comptime if not ragged:
        return True
    elif affine == AFFINE_COL:
        return buffers_aligned and col0 % V == 0
    else:
        return col0 // hxw == (col0 + V - 1) // hxw


@always_inline
def _affine_vec[
    dtype: DType, //, V: Int, affine: Int
](
    p: UnsafePointer[Scalar[dtype], ImmutAnyOrigin],
    col0: Int,
    gbase: Int,
    hxw: Int,
    vec_ok: Bool,
) -> SIMD[DType.float32, V]:
    """The V affine coefficients that multiply columns `[col0, col0 + V)`.

    `vec_ok` is `_affine_vec_ok`; a false there gathers the coefficients
    element-wise, which is always legal and is cheap because the affine
    buffers are small and stay in cache.
    """
    var out = SIMD[DType.float32, V](0)
    comptime if affine == AFFINE_COL:
        if vec_ok:
            return p.load[width=V, alignment=V * size_of[dtype]()](col0).cast[
                DType.float32
            ]()
        return p.load[width=V, alignment=size_of[dtype]()](col0).cast[
            DType.float32
        ]()
    else:
        if vec_ok:
            return SIMD[DType.float32, V](
                p[gbase + col0 // hxw].cast[DType.float32]()
            )
        comptime for k in range(V):
            out[k] = p[gbase + (col0 + k) // hxw].cast[DType.float32]()
        return out


@always_inline
def _affine_scalar[
    dtype: DType, //, affine: Int
](
    p: UnsafePointer[Scalar[dtype], ImmutAnyOrigin],
    col: Int,
    gbase: Int,
    hxw: Int,
) -> Float32:
    """One column's affine coefficient (the scalar head/tail of a row)."""
    comptime if affine == AFFINE_COL:
        return p[col].cast[DType.float32]()
    else:
        return p[gbase + col // hxw].cast[DType.float32]()


@always_inline
def _affine_base[affine: Int](row: Int, group: Int, cpg: Int) -> Int:
    """First channel of this row's group; 0 when the affine is per column."""
    comptime if affine == AFFINE_COL:
        return 0
    else:
        return (row % group) * cpg


@__llvm_metadata(
    MAX_THREADS_PER_BLOCK_METADATA=StaticTuple[Int32, 1](Int32(threads))
)
@__name(t"norm_rows_cached_{dtype}_{affine}_t{threads}_v{vecs}_r{ragged}")
def _norm_rows_cached_kernel[
    dtype: DType, threads: Int, vecs: Int, affine: Int, ragged: Bool
](
    out_ptr: UnsafePointer[Scalar[dtype], MutAnyOrigin],
    mean_ptr: UnsafePointer[Scalar[DType.float32], MutAnyOrigin],
    rstd_ptr: UnsafePointer[Scalar[DType.float32], MutAnyOrigin],
    in_ptr: UnsafePointer[Scalar[dtype], ImmutAnyOrigin],
    gamma_ptr: UnsafePointer[Scalar[dtype], ImmutAnyOrigin],
    beta_ptr: UnsafePointer[Scalar[dtype], ImmutAnyOrigin],
    rows_arg: Int64,
    cols_arg: Int64,
    eps: Float32,
    hxw_arg: Int64,
    cpg_arg: Int64,
    group_arg: Int64,
    has_weight_arg: Int64,
    has_bias_arg: Int64,
    affine_aligned_arg: Int64,
):
    """One block per row, the whole row retained in registers between the two
    reductions and the store, so the row is read from HBM exactly once.

    `ragged` says the rows are NOT all 16-byte aligned runs of whole vectors.
    Such a row is `head` scalars, `n_vec` 16-byte vectors and `tail` scalars —
    the shared `_moment_partition` split, on the ADDRESS rather than the
    column index — so a row length that is not a whole number of vectors (or a
    row whose base lands mid-vector) still runs here rather than falling back
    to a second read. `head` and `tail` are shorter than one vector, hence at
    most one extra scalar per lane on either side. It is a comptime knob and
    not a runtime branch because the whole partition, the two predicated
    scalars and the element-wise affine gather then vanish from the aligned
    kernel, which is the one every training-hot shape takes: keeping them as
    runtime branches measured 120us against 85us on a bf16 32768x1024
    LayerNorm forward.
    """
    # Int is not device-passable (host/device width mismatch); scalars cross
    # the launch ABI as Int64 and index math stays in Int.
    var rows = Int(rows_arg)
    var cols = Int(cols_arg)
    var hxw = Int(hxw_arg)
    var cpg = Int(cpg_arg)
    var group = Int(group_arg)
    var has_weight = Int(has_weight_arg) != 0
    var has_bias = Int(has_bias_arg) != 0
    var affine_aligned = Int(affine_aligned_arg) != 0
    comptime V = 16 // size_of[dtype]()
    comptime vec_align = V * size_of[dtype]()  # 16 bytes
    var tid = Int(thread_idx.x)
    var row = Int(block_idx.x)
    var row_stride = Int(grid_dim.x)
    var inv_cols = 1.0 / Float32(cols)
    while row < rows:
        var base = row * cols
        var head = 0
        var n_vec = cols // V
        var vec_start = base
        var tail_start = cols
        comptime if ragged:
            # `out_ptr` shares `in_ptr`'s 16-byte phase (the launcher checked
            # it), so one partition addresses both buffers.
            _moment_partition[dtype](
                Int(in_ptr), base, cols, head, n_vec, vec_start, tail_start
            )
        var body_col = vec_start - base
        var head_col = tid
        var tail_col = tail_start + tid
        var has_head = ragged and head_col < head
        var has_tail = ragged and tail_col < cols

        var x = InlineArray[SIMD[dtype, V], vecs](fill=SIMD[dtype, V](0))
        var xh = Float32(0)
        var xt = Float32(0)
        var acc = SIMD[DType.float32, V](0)
        comptime for u in range(vecs):
            var v = tid + u * threads
            if v < n_vec:
                x[u] = in_ptr.load[width=V, alignment=vec_align](
                    vec_start + v * V
                )
                acc += x[u].cast[DType.float32]()
        var thread_sum = acc.reduce_add()
        if has_head:
            xh = in_ptr[base + head_col].cast[DType.float32]()
            thread_sum += xh
        if has_tail:
            xt = in_ptr[base + tail_col].cast[DType.float32]()
            thread_sum += xt
        # Row-uniform: every lane of the block reaches both reductions.
        var mean = block.sum[block_size=threads](thread_sum) * inv_cols

        var vacc = SIMD[DType.float32, V](0)
        comptime for u in range(vecs):
            var v = tid + u * threads
            if v < n_vec:
                var d = x[u].cast[DType.float32]() - mean
                vacc = d.fma(d, vacc)
        var thread_var = vacc.reduce_add()
        if has_head:
            thread_var += (xh - mean) * (xh - mean)
        if has_tail:
            thread_var += (xt - mean) * (xt - mean)
        var variance = block.sum[block_size=threads](thread_var) * inv_cols
        var rstd = 1.0 / ieee_sqrt(variance + eps)
        # Match ATen: NaN/inf rows report NaN for both statistics (the
        # centered-square reduction is nonnegative for finite rows).
        if variance != variance:
            mean = variance
        if tid == 0:
            mean_ptr[row] = mean
            rstd_ptr[row] = rstd

        var gbase = _affine_base[affine](row, group, cpg)
        comptime for u in range(vecs):
            var v = tid + u * threads
            if v < n_vec:
                var col0 = body_col + v * V
                var aff_vec_ok = _affine_vec_ok[V, affine, ragged](
                    col0, hxw, affine_aligned
                )
                var r = (x[u].cast[DType.float32]() - mean) * rstd
                if has_weight:
                    r *= _affine_vec[V=V, affine=affine](
                        gamma_ptr, col0, gbase, hxw, aff_vec_ok
                    )
                if has_bias:
                    r += _affine_vec[V=V, affine=affine](
                        beta_ptr, col0, gbase, hxw, aff_vec_ok
                    )
                out_ptr.store[width=V, alignment=vec_align](
                    vec_start + v * V, r.cast[dtype]()
                )
        if has_head:
            _norm_write_element[affine=affine](
                out_ptr,
                in_ptr,
                gamma_ptr,
                beta_ptr,
                base,
                head_col,
                gbase,
                hxw,
                mean,
                rstd,
                has_weight,
                has_bias,
            )
        if has_tail:
            _norm_write_element[affine=affine](
                out_ptr,
                in_ptr,
                gamma_ptr,
                beta_ptr,
                base,
                tail_col,
                gbase,
                hxw,
                mean,
                rstd,
                has_weight,
                has_bias,
            )
        row += row_stride


@always_inline
def _norm_write_element[
    dtype: DType, //, affine: Int
](
    out_ptr: UnsafePointer[Scalar[dtype], MutAnyOrigin],
    in_ptr: UnsafePointer[Scalar[dtype], ImmutAnyOrigin],
    gamma_ptr: UnsafePointer[Scalar[dtype], ImmutAnyOrigin],
    beta_ptr: UnsafePointer[Scalar[dtype], ImmutAnyOrigin],
    base: Int,
    col: Int,
    gbase: Int,
    hxw: Int,
    mean: Float32,
    rstd: Float32,
    has_weight: Bool,
    has_bias: Bool,
):
    """Normalize and store one element of a row (scalar head/tail)."""
    var r = (in_ptr[base + col].cast[DType.float32]() - mean) * rstd
    if has_weight:
        r *= _affine_scalar[affine=affine](gamma_ptr, col, gbase, hxw)
    if has_bias:
        r += _affine_scalar[affine=affine](beta_ptr, col, gbase, hxw)
    out_ptr[base + col] = r.cast[dtype]()


@__llvm_metadata(
    MAX_THREADS_PER_BLOCK_METADATA=StaticTuple[Int32, 1](Int32(NORM_THREADS))
)
@__name(t"norm_rows_moments_{dtype}_{affine}")
def _norm_rows_moments_kernel[
    dtype: DType, affine: Int
](
    out_ptr: UnsafePointer[Scalar[dtype], MutAnyOrigin],
    mean_ptr: UnsafePointer[Scalar[DType.float32], MutAnyOrigin],
    rstd_ptr: UnsafePointer[Scalar[DType.float32], MutAnyOrigin],
    in_ptr: UnsafePointer[Scalar[dtype], ImmutAnyOrigin],
    gamma_ptr: UnsafePointer[Scalar[dtype], ImmutAnyOrigin],
    beta_ptr: UnsafePointer[Scalar[dtype], ImmutAnyOrigin],
    rows_arg: Int64,
    cols_arg: Int64,
    eps: Float32,
    hxw_arg: Int64,
    cpg_arg: Int64,
    group_arg: Int64,
    has_weight_arg: Int64,
    has_bias_arg: Int64,
):
    """One block per row for rows too long (or too unaligned) to cache.

    The statistics are the shared shifted moment pair about the row's first
    element, so the reduction reads the row ONCE; `_moment_cancels` catches
    the case where that first element was a poor stand-in for the mean and
    re-scans about the accurate one, which is the only thing that keeps a
    single-pass formulation honest. The output pass then re-reads the row,
    which the block just touched and L2 therefore still holds.
    """
    # Int is not device-passable (host/device width mismatch); scalars cross
    # the launch ABI as Int64 and index math stays in Int.
    var rows = Int(rows_arg)
    var cols = Int(cols_arg)
    var hxw = Int(hxw_arg)
    var cpg = Int(cpg_arg)
    var group = Int(group_arg)
    var has_weight = Int(has_weight_arg) != 0
    var has_bias = Int(has_bias_arg) != 0
    comptime V = 16 // size_of[dtype]()
    comptime vec_align = V * size_of[dtype]()  # 16 bytes
    var tid = Int(thread_idx.x)
    var row = Int(block_idx.x)
    var row_stride = Int(grid_dim.x)
    var nf = Float32(cols)

    # Input and output are independent allocations with independent 16-byte
    # phases (a sliced input can sit at `ptr % 16 == 4` while its freshly
    # allocated output sits at 0). The output pass loads and stores the SAME
    # row-local index, so it can only vectorize when both agree; otherwise it
    # walks the row element-wise. Loop-invariant.
    var in_phase = _vec16_phase[dtype](Int(in_ptr))
    var vec_io = in_phase >= 0 and in_phase == _vec16_phase[dtype](Int(out_ptr))

    while row < rows:
        var base = row * cols
        var head = 0
        var n_vec = 0
        var vec_start = 0
        var tail_start = 0
        _moment_partition[dtype](
            Int(in_ptr), base, cols, head, n_vec, vec_start, tail_start
        )
        var shift = in_ptr[base].cast[DType.float32]()
        var s = Float32(0)
        var q = Float32(0)
        _moments_scan_contig[V=V, vec_align=vec_align, threads=NORM_THREADS](
            in_ptr,
            base,
            cols,
            head,
            n_vec,
            vec_start,
            tail_start,
            tid,
            shift,
            s,
            q,
        )
        # The block reductions sit outside the scan's own range guards: every
        # thread of the block must reach them (they barrier internally).
        var bs = block.sum[block_size=NORM_THREADS](s)
        var bq = block.sum[block_size=NORM_THREADS](q)
        # Cold path; `block.sum` broadcasts, so the whole block agrees and the
        # early-exit-free branch cannot desynchronize the barriers below.
        if _moment_cancels(bs, bq, cols):
            shift += bs / nf
            _moments_scan_contig[
                V=V, vec_align=vec_align, threads=NORM_THREADS
            ](
                in_ptr,
                base,
                cols,
                head,
                n_vec,
                vec_start,
                tail_start,
                tid,
                shift,
                s,
                q,
            )
            bs = block.sum[block_size=NORM_THREADS](s)
            bq = block.sum[block_size=NORM_THREADS](q)

        var mean = shift + bs / nf
        var m2 = bq - bs * bs / nf
        if m2 < 0:  # a few ulps below zero on a constant row
            m2 = Float32(0)
        var variance = m2 / nf
        var rstd = 1.0 / ieee_sqrt(variance + eps)
        # Match ATen: NaN/inf rows report NaN for both statistics.
        if variance != variance:
            mean = variance
        if tid == 0:
            mean_ptr[row] = mean
            rstd_ptr[row] = rstd

        var gbase = _affine_base[affine](row, group, cpg)
        if vec_io:
            var jh = tid
            while jh < head:
                _norm_write_element[affine=affine](
                    out_ptr,
                    in_ptr,
                    gamma_ptr,
                    beta_ptr,
                    base,
                    jh,
                    gbase,
                    hxw,
                    mean,
                    rstd,
                    has_weight,
                    has_bias,
                )
                jh += NORM_THREADS
            var v = tid
            while v < n_vec:
                var off = vec_start + v * V
                var col0 = off - base
                var r = (
                    in_ptr.load[width=V, alignment=vec_align](off).cast[
                        DType.float32
                    ]()
                    - mean
                ) * rstd
                # The affine buffers have their own 16-byte phase and the
                # column index carries the row's, so the per-column route
                # never vector-loads them here; they are tiny and stay in
                # cache.
                var aff_vec_ok = _affine_vec_ok[V, affine, True](
                    col0, hxw, False
                )
                if has_weight:
                    r *= _affine_vec[V=V, affine=affine](
                        gamma_ptr, col0, gbase, hxw, aff_vec_ok
                    )
                if has_bias:
                    r += _affine_vec[V=V, affine=affine](
                        beta_ptr, col0, gbase, hxw, aff_vec_ok
                    )
                out_ptr.store[width=V, alignment=vec_align](
                    off, r.cast[dtype]()
                )
                v += NORM_THREADS
            var jt = tail_start + tid
            while jt < cols:
                _norm_write_element[affine=affine](
                    out_ptr,
                    in_ptr,
                    gamma_ptr,
                    beta_ptr,
                    base,
                    jt,
                    gbase,
                    hxw,
                    mean,
                    rstd,
                    has_weight,
                    has_bias,
                )
                jt += NORM_THREADS
        else:
            var j = tid
            while j < cols:
                _norm_write_element[affine=affine](
                    out_ptr,
                    in_ptr,
                    gamma_ptr,
                    beta_ptr,
                    base,
                    j,
                    gbase,
                    hxw,
                    mean,
                    rstd,
                    has_weight,
                    has_bias,
                )
                j += NORM_THREADS
        row += row_stride


@always_inline
def _ln_fwd_warp_rows[
    dtype: DType, //, chunks: Int
](
    output: UnsafePointer[Scalar[dtype], MutAnyOrigin],
    mean: UnsafePointer[Scalar[DType.float32], MutAnyOrigin],
    rstd: UnsafePointer[Scalar[DType.float32], MutAnyOrigin],
    input: UnsafePointer[Scalar[dtype], ImmutAnyOrigin],
    weight: UnsafePointer[Scalar[dtype], ImmutAnyOrigin],
    bias: UnsafePointer[Scalar[dtype], ImmutAnyOrigin],
    rows: Int,
    cols: Int,
    vec_cols: Int,
    epsilon: Float32,
    has_weight: Int,
    has_bias: Int,
):
    comptime V = 16 // size_of[dtype]()
    comptime vec_align = V * size_of[dtype]()  # 16 bytes
    var lane = Int(lane_id())
    var row = Int(block_idx.x) * _WARPS_PER_BLOCK + Int(warp_id())
    var row_stride = Int(grid_dim.x) * _WARPS_PER_BLOCK
    var inv_cols = 1.0 / Float32(cols)
    while row < rows:
        var base = row * cols
        # The whole row stays in registers between the two reductions and
        # the store pass; `chunks` is comptime so the array indexes are
        # static.
        var x = InlineArray[SIMD[dtype, V], chunks](fill=SIMD[dtype, V](0))
        var acc = SIMD[DType.float32, V](0.0)
        comptime for u in range(chunks):
            var c = lane + u * WARP_SIZE
            if c < vec_cols:
                x[u] = input.load[width=V, alignment=vec_align](base + c * V)
                acc += x[u].cast[DType.float32]()
        # Row condition is warp-uniform: every lane reaches the shuffles.
        var row_mean = warp.sum(acc.reduce_add()) * inv_cols
        var vacc = SIMD[DType.float32, V](0.0)
        comptime for u in range(chunks):
            var c = lane + u * WARP_SIZE
            if c < vec_cols:
                var centered = x[u].cast[DType.float32]() - row_mean
                vacc = centered.fma(centered, vacc)
        var variance = warp.sum(vacc.reduce_add()) * inv_cols
        var row_rstd = 1.0 / ieee_sqrt(variance + epsilon)
        # Match ATen: NaN/inf rows report NaN for both statistics (the
        # centered-square reduction is nonnegative for finite rows).
        if variance != variance:
            row_mean = variance
        if lane == 0:
            mean[row] = row_mean
            rstd[row] = row_rstd
        comptime for u in range(chunks):
            var c = lane + u * WARP_SIZE
            if c < vec_cols:
                var result = (x[u].cast[DType.float32]() - row_mean) * row_rstd
                if has_weight != 0:
                    result *= weight.load[width=V, alignment=vec_align](
                        c * V
                    ).cast[DType.float32]()
                if has_bias != 0:
                    result += bias.load[width=V, alignment=vec_align](
                        c * V
                    ).cast[DType.float32]()
                output.store[width=V, alignment=vec_align](
                    base + c * V, result.cast[dtype]()
                )
        row += row_stride


@__name(t"layer_norm_warp_{dtype}_c{chunks}")
def _ln_fwd_warp_kernel[
    dtype: DType, chunks: Int
](
    output: UnsafePointer[Scalar[dtype], MutAnyOrigin],
    mean: UnsafePointer[Scalar[DType.float32], MutAnyOrigin],
    rstd: UnsafePointer[Scalar[DType.float32], MutAnyOrigin],
    input: UnsafePointer[Scalar[dtype], ImmutAnyOrigin],
    weight: UnsafePointer[Scalar[dtype], ImmutAnyOrigin],
    bias: UnsafePointer[Scalar[dtype], ImmutAnyOrigin],
    rows_arg: Int64,
    cols_arg: Int64,
    vec_cols_arg: Int64,
    epsilon: Float32,
    has_weight_arg: Int64,
    has_bias_arg: Int64,
):
    # Int is not device-passable (host/device width mismatch); scalars cross
    # the launch ABI as Int64 and index math stays in Int.
    _ln_fwd_warp_rows[chunks](
        output,
        mean,
        rstd,
        input,
        weight,
        bias,
        Int(rows_arg),
        Int(cols_arg),
        Int(vec_cols_arg),
        epsilon,
        Int(has_weight_arg),
        Int(has_bias_arg),
    )


@always_inline
def _enqueue_norm_cached[
    dtype: DType, threads: Int, vecs: Int, affine: Int, ragged: Bool
](
    out_ptr: UnsafePointer[Scalar[dtype], MutAnyOrigin],
    mean_ptr: UnsafePointer[Scalar[DType.float32], MutAnyOrigin],
    rstd_ptr: UnsafePointer[Scalar[DType.float32], MutAnyOrigin],
    in_ptr: UnsafePointer[Scalar[dtype], ImmutAnyOrigin],
    gamma_ptr: UnsafePointer[Scalar[dtype], ImmutAnyOrigin],
    beta_ptr: UnsafePointer[Scalar[dtype], ImmutAnyOrigin],
    rows: Int,
    cols: Int,
    eps: Float32,
    hxw: Int,
    cpg: Int,
    group: Int,
    has_weight: Int,
    has_bias: Int,
    affine_aligned: Int,
    ctx: DeviceContext,
) raises:
    _enqueue_cached[
        _norm_rows_cached_kernel[dtype, threads, vecs, affine, ragged]
    ](
        ctx,
        String(t"norm_rows_cached_{dtype}_{affine}_{threads}_{vecs}_{ragged}"),
        max(1, min(rows, _MAX_GRID)),
        1,
        1,
        threads,
        out_ptr,
        mean_ptr,
        rstd_ptr,
        in_ptr,
        gamma_ptr,
        beta_ptr,
        Int64(rows),
        Int64(cols),
        eps,
        Int64(hxw),
        Int64(cpg),
        Int64(group),
        Int64(has_weight),
        Int64(has_bias),
        Int64(affine_aligned),
    )


@always_inline
def _enqueue_norm_warp[
    dtype: DType, chunks: Int
](
    out_ptr: UnsafePointer[Scalar[dtype], MutAnyOrigin],
    mean_ptr: UnsafePointer[Scalar[DType.float32], MutAnyOrigin],
    rstd_ptr: UnsafePointer[Scalar[DType.float32], MutAnyOrigin],
    in_ptr: UnsafePointer[Scalar[dtype], ImmutAnyOrigin],
    gamma_ptr: UnsafePointer[Scalar[dtype], ImmutAnyOrigin],
    beta_ptr: UnsafePointer[Scalar[dtype], ImmutAnyOrigin],
    rows: Int,
    cols: Int,
    vec_cols: Int,
    eps: Float32,
    has_weight: Int,
    has_bias: Int,
    ctx: DeviceContext,
) raises:
    _enqueue_cached[_ln_fwd_warp_kernel[dtype, chunks]](
        ctx,
        String(t"ln_fwd_warp_{dtype}_c{chunks}"),
        max(1, min(ceildiv(rows, _WARPS_PER_BLOCK), _MAX_GRID)),
        1,
        1,
        WARP_SIZE * _WARPS_PER_BLOCK,
        out_ptr,
        mean_ptr,
        rstd_ptr,
        in_ptr,
        gamma_ptr,
        beta_ptr,
        Int64(rows),
        Int64(cols),
        Int64(vec_cols),
        eps,
        Int64(has_weight),
        Int64(has_bias),
    )


def enqueue_norm_rows[
    dtype: DType, affine: Int
](
    out_addr: Int,
    mean_addr: Int,
    rstd_addr: Int,
    in_addr: Int,
    gamma_addr: Int,
    beta_addr: Int,
    rows: Int,
    cols: Int,
    eps: Float32,
    hxw: Int,
    cpg: Int,
    group: Int,
    has_weight: Bool,
    has_bias: Bool,
    ctx: DeviceContext,
) raises:
    """Normalize `rows` contiguous runs of `cols` elements, writing the output
    plus the per-row float32 mean and rstd.

    `hxw`, `cpg` and `group` are read only by the `AFFINE_CHAN` route (group
    norm); layer norm passes them as 1 / 1 / 1.
    """
    if rows <= 0 or cols <= 0:
        return
    var out_ptr = _make_ptr[dtype](out_addr).as_unsafe_any_origin()
    var mean_ptr = _make_ptr[DType.float32](mean_addr).as_unsafe_any_origin()
    var rstd_ptr = _make_ptr[DType.float32](rstd_addr).as_unsafe_any_origin()
    var in_ptr = _make_ptr[dtype](in_addr).as_unsafe_any_origin().as_immutable()
    var gamma_ptr = (
        _make_ptr[dtype](gamma_addr).as_unsafe_any_origin().as_immutable()
    )
    var beta_ptr = (
        _make_ptr[dtype](beta_addr).as_unsafe_any_origin().as_immutable()
    )
    var hw = 1 if has_weight else 0
    var hb = 1 if has_bias else 0

    comptime V = 16 // size_of[dtype]()
    # The cached route stores where it loaded, so it needs the two buffers to
    # share a 16-byte phase; the row length itself may be ragged (its head and
    # tail are one scalar per lane). The affine buffers have their own phase,
    # which only decides whether their coefficients are vector-loaded.
    var in_phase = _vec16_phase[dtype](in_addr)
    var row_ok = in_phase >= 0 and in_phase == _vec16_phase[dtype](out_addr)
    var affine_aligned = 1
    if has_weight and gamma_addr % 16 != 0:
        affine_aligned = 0
    if has_bias and beta_addr % 16 != 0:
        affine_aligned = 0
    # Vectors a row can need, whatever its phase (a head shortens the body).
    var need_vecs = ceildiv(cols, V)

    comptime if has_apple_gpu_accelerator():
        comptime if affine == AFFINE_COL:
            if (
                row_ok
                and in_phase == 0
                and cols % V == 0
                and affine_aligned != 0
                and cols <= 8 * WARP_SIZE * V
            ):
                var vec_cols = cols // V
                var chunks = ceildiv(vec_cols, WARP_SIZE)
                # Smallest ladder entry that covers `chunks`.
                if chunks <= 1:
                    _enqueue_norm_warp[dtype, 1](
                        out_ptr,
                        mean_ptr,
                        rstd_ptr,
                        in_ptr,
                        gamma_ptr,
                        beta_ptr,
                        rows,
                        cols,
                        vec_cols,
                        eps,
                        hw,
                        hb,
                        ctx,
                    )
                elif chunks <= 2:
                    _enqueue_norm_warp[dtype, 2](
                        out_ptr,
                        mean_ptr,
                        rstd_ptr,
                        in_ptr,
                        gamma_ptr,
                        beta_ptr,
                        rows,
                        cols,
                        vec_cols,
                        eps,
                        hw,
                        hb,
                        ctx,
                    )
                elif chunks <= 3:
                    _enqueue_norm_warp[dtype, 3](
                        out_ptr,
                        mean_ptr,
                        rstd_ptr,
                        in_ptr,
                        gamma_ptr,
                        beta_ptr,
                        rows,
                        cols,
                        vec_cols,
                        eps,
                        hw,
                        hb,
                        ctx,
                    )
                elif chunks <= 4:
                    _enqueue_norm_warp[dtype, 4](
                        out_ptr,
                        mean_ptr,
                        rstd_ptr,
                        in_ptr,
                        gamma_ptr,
                        beta_ptr,
                        rows,
                        cols,
                        vec_cols,
                        eps,
                        hw,
                        hb,
                        ctx,
                    )
                elif chunks <= 6:
                    _enqueue_norm_warp[dtype, 6](
                        out_ptr,
                        mean_ptr,
                        rstd_ptr,
                        in_ptr,
                        gamma_ptr,
                        beta_ptr,
                        rows,
                        cols,
                        vec_cols,
                        eps,
                        hw,
                        hb,
                        ctx,
                    )
                else:
                    _enqueue_norm_warp[dtype, 8](
                        out_ptr,
                        mean_ptr,
                        rstd_ptr,
                        in_ptr,
                        gamma_ptr,
                        beta_ptr,
                        rows,
                        cols,
                        vec_cols,
                        eps,
                        hw,
                        hb,
                        ctx,
                    )
                return

    if row_ok and need_vecs <= _CACHED_MAX_VECS * _CACHED_MAX_WIDTH:
        # Every row a 16-byte-aligned run of whole vectors, with vector-loadable
        # affine coefficients: the specialization that carries no partition and
        # no gather arm at all.
        var straight = in_phase == 0 and cols % V == 0 and affine_aligned != 0
        comptime if affine == AFFINE_CHAN:
            straight = straight and hxw % V == 0
        # Narrowest block that still gives every lane a vector, then as many
        # vectors per lane as the row needs.
        comptime for w in range(len(_CACHED_WIDTHS)):
            comptime threads = _CACHED_WIDTHS[w]
            if need_vecs <= threads:
                if straight:
                    _enqueue_norm_cached[dtype, threads, 1, affine, False](
                        out_ptr,
                        mean_ptr,
                        rstd_ptr,
                        in_ptr,
                        gamma_ptr,
                        beta_ptr,
                        rows,
                        cols,
                        eps,
                        hxw,
                        cpg,
                        group,
                        hw,
                        hb,
                        affine_aligned,
                        ctx,
                    )
                else:
                    _enqueue_norm_cached[dtype, threads, 1, affine, True](
                        out_ptr,
                        mean_ptr,
                        rstd_ptr,
                        in_ptr,
                        gamma_ptr,
                        beta_ptr,
                        rows,
                        cols,
                        eps,
                        hxw,
                        cpg,
                        group,
                        hw,
                        hb,
                        affine_aligned,
                        ctx,
                    )
                return
        if straight:
            _enqueue_norm_cached[
                dtype, _CACHED_MAX_WIDTH, _CACHED_MAX_VECS, affine, False
            ](
                out_ptr,
                mean_ptr,
                rstd_ptr,
                in_ptr,
                gamma_ptr,
                beta_ptr,
                rows,
                cols,
                eps,
                hxw,
                cpg,
                group,
                hw,
                hb,
                affine_aligned,
                ctx,
            )
        else:
            _enqueue_norm_cached[
                dtype, _CACHED_MAX_WIDTH, _CACHED_MAX_VECS, affine, True
            ](
                out_ptr,
                mean_ptr,
                rstd_ptr,
                in_ptr,
                gamma_ptr,
                beta_ptr,
                rows,
                cols,
                eps,
                hxw,
                cpg,
                group,
                hw,
                hb,
                affine_aligned,
                ctx,
            )
        return

    _enqueue_cached[_norm_rows_moments_kernel[dtype, affine]](
        ctx,
        String(t"norm_rows_moments_{dtype}_{affine}"),
        max(1, min(rows, _MAX_GRID)),
        1,
        1,
        NORM_THREADS,
        out_ptr,
        mean_ptr,
        rstd_ptr,
        in_ptr,
        gamma_ptr,
        beta_ptr,
        Int64(rows),
        Int64(cols),
        eps,
        Int64(hxw),
        Int64(cpg),
        Int64(group),
        Int64(hw),
        Int64(hb),
    )
