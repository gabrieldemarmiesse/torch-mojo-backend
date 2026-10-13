# ===----------------------------------------------------------------------=== #
# The two element-wise halves of torch._scaled_mm around a float32 GEMM.
#
#   Dequant   out_f32[r, c] = float(x[r, c]), x a float8 matrix of any
#             strides (exact: a float8 value has at most three mantissa
#             bits).
#   BlockAccum  acc[i, j] (+)= (p[i, j] * sa(i, kb)) * sb(j, kb)
#             one K block's partial product folded into the accumulator with
#             that block's scales, the way cuBLASLt applies 1x128 / 128x128
#             float32 block scales and MX 1x32 float8_e8m0fnu ones (in the
#             SWIZZLE_32_4_4 layout): a partial sum per K block, scaled, then
#             accumulated.
#   Epilogue  out[i, j] = cvt((p[i, j] * (sa[i] * sb[j]) + bias[j]) * sr)
#             with sa / sb a scalar, a vector or absent, a float16 /
#             bfloat16 bias, and a float8 output saturated to its largest
#             finite value the way cuBLAS writes one.
#
# The GEMM between them is the float32 one of tmb/ops/matmul.mojo. A float8
# value has at most three mantissa bits, so its float32 copy is exact and
# every product of two of them is exact in float32: with the scales left
# for the epilogue (tensorwise, rowwise), the float32 GEMM computes exactly
# what a float8 tensor-core GEMM with a float32 accumulator does.
# ===----------------------------------------------------------------------=== #

from std.memory import bitcast
from max.gpu import block_dim, block_idx, grid_dim, thread_idx
from max.gpu.host import DeviceContext

from tmb.kernels.common.dtype_convert import f32_to_bf16
from tmb.kernels.common.float8 import f32_to_fp8, fp8_to_f32, is_float8
from tmb.kernels.common.op_utils import (
    Arg,
    Argv,
    GS_THREADS,
    _enqueue_cached,
    _gs_blocks,
    _make_ptr,
    _raw_ctx,
    _raw_dtype_int,
    _raw_int,
    _spec_dispatcher7,
    _spec_dispatcher16,
)
from tmb.kernels.common.variant_gates import (
    ErrBuf,
    NO_OP_COMPILED,
    _dtype_arg_on,
    _dtype_out_on,
    _op_on,
    _tmb_entry_error,
)

# Block-scale recipes of `BlockAccum` (the op side spells the same numbers).
comptime RECIPE_NONE = 0
comptime RECIPE_TENSOR = 1
comptime RECIPE_ROW = 2
comptime RECIPE_BLOCK_1X128 = 3
comptime RECIPE_BLOCK_128X128 = 4
comptime RECIPE_MX_SWIZZLED = 5

comptime FP8_MATRIX_DTYPES = [
    DType.float8_e4m3fn,
    DType.float8_e5m2,
    DType.float8_e4m3fnuz,
    DType.float8_e5m2fnuz,
]

comptime EPILOGUE_OUT_DTYPES = [
    DType.float32,
    DType.bfloat16,
    DType.float16,
    DType.float8_e4m3fn,
    DType.float8_e5m2,
    DType.float8_e4m3fnuz,
    DType.float8_e5m2fnuz,
]


@always_inline
def _mx_swizzled_index(r: Int, kb: Int, kblocks: Int) -> Int:
    """Where scale (row r, K-block kb) sits in torch's `to_blocked` layout
    (cuBLAS SWIZZLE_32_4_4): 128x4 tiles in row-major tile order, each
    stored as 32 rows of 16 -- row r % 32, column (r % 128) // 32 * 4 + kb % 4.
    `kblocks` is ceil(K / 32); tiles per row of tiles round it up to 4."""
    var col_tiles = (kblocks + 3) // 4
    var tile = (r // 128) * col_tiles + kb // 4
    var rin = r % 128
    return tile * 512 + (rin % 32) * 16 + (rin // 32) * 4 + kb % 4


def _dequant_kernel[
    src: DType
](
    out_ptr: Pointer[Float32, MutAnyOrigin],
    in_ptr: Pointer[Scalar[DType.uint8], ImmutAnyOrigin],
    rows_a: Int64,
    cols_a: Int64,
    rs_a: Int64,
    cs_a: Int64,
):
    var cols = Int(cols_a)
    var total = Int(rows_a) * cols
    var i = Int(block_idx.x) * Int(block_dim.x) + Int(thread_idx.x)
    var step = Int(grid_dim.x) * Int(block_dim.x)
    while i < total:
        var r = i // cols
        var c = i - r * cols
        out_ptr[unsafe_offset=i] = fp8_to_f32[src, 1](
            SIMD[DType.uint8, 1](
                in_ptr[unsafe_offset=r * Int(rs_a) + c * Int(cs_a)]
            )
        )[0]
        i += step


def _dequant_go(
    out_o: Arg,
    in_o: Arg,
    rows_o: Arg,
    cols_o: Arg,
    rs_o: Arg,
    cs_o: Arg,
    ctx_o: Arg,
) raises:
    """`Dequant`: slots (out f32, in, rows, cols, in row/col strides,
    ctx). Scales are applied later (`BlockAccum`, `Epilogue`)."""
    var ctx = _raw_ctx(ctx_o)
    var total = _raw_int(rows_o) * _raw_int(cols_o)
    if total == 0:
        return
    var handled = False
    comptime for dt in FP8_MATRIX_DTYPES:
        comptime if _dtype_arg_on[0, dt]():
            _enqueue_cached[_dequant_kernel[dt]](
                ctx,
                _gs_blocks(total),
                1,
                1,
                GS_THREADS,
                _make_ptr[DType.float32](
                    _raw_int(out_o)
                ).as_unsafe_any_origin(),
                _make_ptr[DType.uint8](_raw_int(in_o))
                .as_unsafe_any_origin()
                .as_imm(),
                Int64(_raw_int(rows_o)),
                Int64(_raw_int(cols_o)),
                Int64(_raw_int(rs_o)),
                Int64(_raw_int(cs_o)),
            )
            handled = True
    if not handled:
        raise Error("scaled_mm Dequant: unsupported float8 dtype")


@always_inline
def _block_scale(
    ptr: Pointer[Scalar[DType.uint8], ImmutAnyOrigin],
    recipe: Int,
    ss0: Int,
    ss1: Int,
    row: Int,
    kb: Int,
    kblocks: Int,
) -> Float32:
    """The scale of (`row`, K block `kb`) under a block recipe."""
    if recipe == RECIPE_MX_SWIZZLED:
        var at = _mx_swizzled_index(row, kb, kblocks)
        return fp8_to_f32[DType.float8_e8m0fnu, 1](
            SIMD[DType.uint8, 1](ptr[unsafe_offset=at])
        )[0]
    var at = row * ss0 + kb * ss1
    if recipe == RECIPE_BLOCK_128X128:
        at = (row // 128) * ss0 + kb * ss1
    return ptr.unsafe_bitcast[Float32]()[unsafe_offset=at]


def _block_accum_kernel(
    acc_ptr: Pointer[Float32, MutAnyOrigin],
    p_ptr: Pointer[Float32, ImmutAnyOrigin],
    m_a: Int64,
    n_a: Int64,
    kb_a: Int64,
    first_a: Int64,
    sa_ptr: Pointer[Scalar[DType.uint8], ImmutAnyOrigin],
    sa_recipe: Int64,
    sa_ss0: Int64,
    sa_ss1: Int64,
    sb_ptr: Pointer[Scalar[DType.uint8], ImmutAnyOrigin],
    sb_recipe: Int64,
    sb_ss0: Int64,
    sb_ss1: Int64,
    kblocks_a: Int64,
):
    var n = Int(n_a)
    var total = Int(m_a) * n
    var kb = Int(kb_a)
    var kblocks = Int(kblocks_a)
    var i = Int(block_idx.x) * Int(block_dim.x) + Int(thread_idx.x)
    var step = Int(grid_dim.x) * Int(block_dim.x)
    while i < total:
        var r = i // n
        var c = i - r * n
        var sa = _block_scale(
            sa_ptr, Int(sa_recipe), Int(sa_ss0), Int(sa_ss1), r, kb, kblocks
        )
        var sb = _block_scale(
            sb_ptr, Int(sb_recipe), Int(sb_ss0), Int(sb_ss1), c, kb, kblocks
        )
        # (partial * sa) * sb: measured on stock CUDA, 1x128 scales 1e37
        # and 1e-37 overflow (inf), 1e-37 and 1e37 do not.
        var v = (p_ptr[unsafe_offset=i] * sa) * sb
        if first_a != 0:
            acc_ptr[unsafe_offset=i] = v
        else:
            acc_ptr[unsafe_offset=i] = acc_ptr[unsafe_offset=i] + v
        i += step


def _block_accum_go(
    acc_o: Arg,
    p_o: Arg,
    m_o: Arg,
    n_o: Arg,
    kb_o: Arg,
    first_o: Arg,
    sa_o: Arg,
    sar_o: Arg,
    sa0_o: Arg,
    sa1_o: Arg,
    sb_o: Arg,
    sbr_o: Arg,
    sb0_o: Arg,
    sb1_o: Arg,
    kblocks_o: Arg,
    ctx_o: Arg,
) raises:
    """`BlockAccum`: slots (acc f32, p f32 (both m x n contiguous), m, n,
    K block, first, then for scale_a and scale_b: address, recipe, row
    stride, K-block stride; the K-block count (MX layout), ctx)."""
    var ctx = _raw_ctx(ctx_o)
    var total = _raw_int(m_o) * _raw_int(n_o)
    if total == 0:
        return
    _enqueue_cached[_block_accum_kernel](
        ctx,
        _gs_blocks(total),
        1,
        1,
        GS_THREADS,
        _make_ptr[DType.float32](_raw_int(acc_o)).as_unsafe_any_origin(),
        _make_ptr[DType.float32](_raw_int(p_o)).as_unsafe_any_origin().as_imm(),
        Int64(_raw_int(m_o)),
        Int64(_raw_int(n_o)),
        Int64(_raw_int(kb_o)),
        Int64(_raw_int(first_o)),
        _make_ptr[DType.uint8](_raw_int(sa_o)).as_unsafe_any_origin().as_imm(),
        Int64(_raw_int(sar_o)),
        Int64(_raw_int(sa0_o)),
        Int64(_raw_int(sa1_o)),
        _make_ptr[DType.uint8](_raw_int(sb_o)).as_unsafe_any_origin().as_imm(),
        Int64(_raw_int(sbr_o)),
        Int64(_raw_int(sb0_o)),
        Int64(_raw_int(sb1_o)),
        Int64(_raw_int(kblocks_o)),
    )


@always_inline
def _or(addr: Int, fallback: Int) -> Int:
    return addr if addr != 0 else fallback


@always_inline
def _float8_max[dt: DType]() -> Float32:
    """The largest finite value, what cuBLAS saturates a float8 output to."""
    comptime if dt == DType.float8_e4m3fn:
        return 448.0
    elif dt == DType.float8_e4m3fnuz:
        return 240.0
    else:
        return 57344.0


@always_inline
def _to_out[dst: DType](v: Float32) -> Scalar[dst]:
    comptime if is_float8(dst):
        # Saturate (NaN stays NaN: both comparisons are false for it).
        comptime top = _float8_max[dst]()
        var c = v
        if c > top:
            c = top
        elif c < -top:
            c = -top
        return bitcast[dst, 1](f32_to_fp8[dst, 1](SIMD[DType.float32, 1](c)))
    elif dst == DType.bfloat16:
        return rebind[Scalar[dst]](f32_to_bf16[1](SIMD[DType.float32, 1](v)))
    else:
        return v.cast[dst]()


def _epilogue_kernel[
    dst: DType
](
    out_ptr: Pointer[Scalar[dst], MutAnyOrigin],
    p_ptr: Pointer[Float32, ImmutAnyOrigin],
    m_a: Int64,
    n_a: Int64,
    ors_a: Int64,
    ocs_a: Int64,
    sa: Pointer[Float32, ImmutAnyOrigin],
    sa_stride: Int64,
    sb: Pointer[Float32, ImmutAnyOrigin],
    sb_stride: Int64,
    bias: Pointer[Scalar[DType.uint8], ImmutAnyOrigin],
    sr: Pointer[Float32, ImmutAnyOrigin],
    flags: Int64,
    mode: Int64,
    bias_stride: Int64,
):
    var m = Int(m_a)
    var n = Int(n_a)
    var total = m * n
    var bf = bias.unsafe_bitcast[BFloat16]()
    var bh = bias.unsafe_bitcast[Float16]()
    var has_sa = (flags & 1) != 0
    var has_sb = (flags & 2) != 0
    var has_bias = (flags & 4) != 0
    var has_sr = (flags & 8) != 0
    var a_first = (mode & 2) != 0
    var b_first = (mode & 4) != 0
    var i = Int(block_idx.x) * Int(block_dim.x) + Int(thread_idx.x)
    var step = Int(grid_dim.x) * Int(block_dim.x)
    while i < total:
        var r = i // n
        var c = i - r * n
        var v = p_ptr[unsafe_offset=i]
        # The outer scales' order is each CUDA path's own, measured on stock
        # CUDA with 1e37 / 1e-37 scales (which order overflows):
        #   tensorwise (cuBLAS alpha)    acc * (sa * sb)     mode 0
        #   grouped (CUTLASS EVT)        (acc * sa) * sb     mode bit 1
        #   rowwise _scaled_mm           (acc * sb) * sa     mode bit 2
        if a_first:
            if has_sa:
                v = v * sa[unsafe_offset=r * Int(sa_stride)]
            if has_sb:
                v = v * sb[unsafe_offset=c * Int(sb_stride)]
        elif b_first:
            if has_sb:
                v = v * sb[unsafe_offset=c * Int(sb_stride)]
            if has_sa:
                v = v * sa[unsafe_offset=r * Int(sa_stride)]
        else:
            var scale = Float32(1)
            if has_sa:
                scale = sa[unsafe_offset=r * Int(sa_stride)]
            if has_sb:
                scale = scale * sb[unsafe_offset=c * Int(sb_stride)]
            if has_sa or has_sb:
                v = v * scale
        if has_bias:
            if (mode & 1) != 0:
                v = (
                    v
                    + bh[unsafe_offset=c * Int(bias_stride)].cast[
                        DType.float32
                    ]()
                )
            else:
                v = (
                    v
                    + bf[unsafe_offset=c * Int(bias_stride)].cast[
                        DType.float32
                    ]()
                )
        if has_sr:
            v = v * sr[]
        out_ptr[unsafe_offset=r * Int(ors_a) + c * Int(ocs_a)] = _to_out[dst](v)
        i += step


def _epilogue_go(
    out_o: Arg,
    p_o: Arg,
    m_o: Arg,
    n_o: Arg,
    ors_o: Arg,
    ocs_o: Arg,
    sa_o: Arg,
    sas_o: Arg,
    sb_o: Arg,
    sbs_o: Arg,
    bias_o: Arg,
    mode_o: Arg,
    bias_stride_o: Arg,
    sr_o: Arg,
    out_dtype_o: Arg,
    ctx_o: Arg,
) raises:
    """`Epilogue`: slots (out, p f32 contiguous, m, n, out row/col strides,
    scale_a address + element stride (0: one scalar), scale_b likewise,
    bias address (0: none), mode (bit 0: a float16 bias, else bfloat16;
    bit 1: scale_a then scale_b, bit 2: scale_b then scale_a, neither: one
    combined factor) and bias stride, scale_result address (0: none),
    out dtype, ctx)."""
    var ctx = _raw_ctx(ctx_o)
    var total = _raw_int(m_o) * _raw_int(n_o)
    if total == 0:
        return
    var out_dtype = _raw_dtype_int(out_dtype_o)
    var p = _raw_int(p_o)  # stands in for an absent operand's pointer
    var handled = False
    comptime for dt in EPILOGUE_OUT_DTYPES:
        comptime if _dtype_out_on[0, dt]():
            if out_dtype == dt:
                _enqueue_cached[_epilogue_kernel[dt]](
                    ctx,
                    _gs_blocks(total),
                    1,
                    1,
                    GS_THREADS,
                    _make_ptr[dt](_raw_int(out_o)).as_unsafe_any_origin(),
                    _make_ptr[DType.float32](_raw_int(p_o))
                    .as_unsafe_any_origin()
                    .as_imm(),
                    Int64(_raw_int(m_o)),
                    Int64(_raw_int(n_o)),
                    Int64(_raw_int(ors_o)),
                    Int64(_raw_int(ocs_o)),
                    _make_ptr[DType.float32](_or(_raw_int(sa_o), p))
                    .as_unsafe_any_origin()
                    .as_imm(),
                    Int64(_raw_int(sas_o)),
                    _make_ptr[DType.float32](_or(_raw_int(sb_o), p))
                    .as_unsafe_any_origin()
                    .as_imm(),
                    Int64(_raw_int(sbs_o)),
                    _make_ptr[DType.uint8](_or(_raw_int(bias_o), p))
                    .as_unsafe_any_origin()
                    .as_imm(),
                    _make_ptr[DType.float32](_or(_raw_int(sr_o), p))
                    .as_unsafe_any_origin()
                    .as_imm(),
                    Int64(
                        (1 if _raw_int(sa_o) != 0 else 0)
                        | (2 if _raw_int(sb_o) != 0 else 0)
                        | (4 if _raw_int(bias_o) != 0 else 0)
                        | (8 if _raw_int(sr_o) != 0 else 0)
                    ),
                    Int64(_raw_int(mode_o)),
                    Int64(_raw_int(bias_stride_o)),
                )
                handled = True
    if not handled:
        raise Error("scaled_mm Epilogue: unsupported output dtype")


@export
def tmb_call(argv: Argv, argc: Int, err: ErrBuf, errcap: Int) abi("C") -> Int32:
    """C entry of this family: one kernel per build (see `OP`)."""
    try:
        comptime if _op_on["Dequant"]():
            _spec_dispatcher7[_dequant_go, "Dequant"](argv, argc)
            return 0
        elif _op_on["BlockAccum"]():
            _spec_dispatcher16[_block_accum_go, "BlockAccum"](argv, argc)
            return 0
        elif _op_on["Epilogue"]():
            _spec_dispatcher16[_epilogue_go, "Epilogue"](argv, argc)
            return 0
        raise Error(NO_OP_COMPILED)
    except e:
        return _tmb_entry_error(err, errcap, e)
