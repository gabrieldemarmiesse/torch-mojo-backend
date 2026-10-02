# ===----------------------------------------------------------------------=== #
# Fast eager-mode NN kernels for mojo_device: batch norm (inference),
# layer norm, row softmax (with optional causal mask), spatial mean,
# embedding gather, and boolean all-reduce.
#
# Same architecture as elementwise.mojo: Python-visible functions get raw
# integer pointers (tensor `._ptr`, offset pre-applied) plus dtype ints and
# the device's DeviceContext pointer, and enqueue work on MAX's own device
# queue (fire and forget, no sync).
#
# Most kernels here are written as a parallel-for over independent output
# elements or rows (`elementwise` with an inner sequential loop), so the same
# code runs on CPU and GPU with fully dynamic shapes. The row-reduction ops
# (layer norm, max) additionally have explicit GPU kernels that
# launch one thread block per row: their row counts are far too small
# (batch * seq_len) for a thread-per-row launch to fill the GPU. Row softmax
# instead delegates its GPU path to modular's nn.softmax grid-stride kernels,
# and argmax to `argreduce_kernels.mojo`, shared with reduction' argmin,
# which splits the reduce axis when the rows alone cannot fill the device.
# ===----------------------------------------------------------------------=== #

from std.os import abort
from max.gpu.sync import barrier
from max.gpu import (
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
from max.gpu.primitives import warp
from std.math import ceildiv, exp, floor
from std.memory import stack_allocation
from std.memory.alloc import unsafe_alloc
from std.sys.info import (
    _accelerator_arch,
    has_accelerator,
    has_apple_gpu_accelerator,
    size_of,
)
from std.utils.coord import Coord
from std.utils.index import IndexList
from std.utils.numerics import min_finite
from std.utils.static_tuple import StaticTuple

from tmb.kernels.common.gpu_elementwise import elementwise
from tmb.kernels.reduction.argreduce import _argreduce_spec_into
from tmb.kernels.nn.cumsum_kernels import (
    CUMSUM_DTYPES,
    FILL_WAVES,
    _acc_dtype,
    cumsum_workspace_lines,
    enqueue_cumsum_cols,
    enqueue_cumsum_rows,
    enqueue_cumsum_rows_workspace,
)
from tmb.kernels.nn.gather_kernels import _gather0
from tmb.kernels.reduction.reduce_skeleton import (
    AllOp,
    AnyOp,
    MaxOp,
    MeanOp,
    _reduce_generic,
    _rowred_spec_into_go,
)
from tmb.kernels.nn.softmax_rows_kernels import (
    _softmax_rows,
    _softmax_rows_dropout_go,
    softmax_rows_f64,
)
from layout import TileTensor, row_major
from tmb.kernels.random.dropout_kernels import _philox4x32_10


from tmb.kernels.common.op_utils import (
    Arg,
    Argv,
    FLOAT_DTYPES,
    MAX_RANK,
    _check_into,
    _check_into_sized,
    _device_sm_count,
    _enqueue_cached,
    _make_ptr,
    _parallel_for,
    _raw_ctx,
    _raw_dtype_int,
    _raw_f64,
    _raw_int,
    _raw_tuple_f64,
    _raw_tuple_int,
    _reduce_spec_geom,
    _spec_dispatcher2,
    _spec_dispatcher3,
    _spec_dispatcher4,
    _spec_dispatcher5,
    _spec_dispatcher7,
    _spec_ptr,
    ieee_sqrt,
)

from tmb.kernels.common.variant_gates import (
    ErrBuf,
    NO_OP_COMPILED,
    _dtype_arg_on,
    _dtype_supported,
    _op_on,
    _tmb_entry_error,
)


# ---------------------------------------------------------------------------
# Batch norm, inference mode: out = (x - mean[c]) / sqrt(var[c] + eps) * g + b
# Input is NC... contiguous; `inner` is the product of the dims after C.
# ---------------------------------------------------------------------------


@always_inline
def _batch_norm[
    dtype: DType
](
    out_addr: Int,
    in_addr: Int,
    mean_addr: Int,
    var_addr: Int,
    gamma_addr: Int,
    beta_addr: Int,
    eps: Float32,
    channels: Int,
    inner: Int,
    total: Int,
    ctx: DeviceContext,
) raises:
    var out_ptr = _make_ptr[dtype](out_addr)
    var in_ptr = _make_ptr[dtype](in_addr)
    var mean_ptr = _make_ptr[dtype](mean_addr)
    var var_ptr = _make_ptr[dtype](var_addr)
    var gamma_ptr = _make_ptr[dtype](gamma_addr)
    var beta_ptr = _make_ptr[dtype](beta_addr)

    @always_inline
    @__parameter
    @__copy_capture(out_ptr, in_ptr, mean_ptr, var_ptr, gamma_ptr, beta_ptr)
    def func[width: Int, alignment: Int = 1](idx: Coord):
        var i = Int(idx[0].value())
        var c = (i // inner) % channels
        var m = mean_ptr[unsafe_offset=c].cast[DType.float32]()
        var v = var_ptr[unsafe_offset=c].cast[DType.float32]()
        var g = gamma_ptr[unsafe_offset=c].cast[DType.float32]()
        var b = beta_ptr[unsafe_offset=c].cast[DType.float32]()
        var scale = g / ieee_sqrt(v + eps)
        var a = in_ptr[unsafe_offset=i].cast[DType.float32]()
        out_ptr[unsafe_offset=i] = ((a - m) * scale + b).cast[dtype]()

    _parallel_for[func](total, ctx)


def _batch_norm_go(
    out_ptr_obj: Arg,
    in_ptr_obj: Arg,
    mean_ptr_obj: Arg,
    var_ptr_obj: Arg,
    gamma_ptr_obj: Arg,
    beta_ptr_obj: Arg,
    params: Arg,  # (eps, channels, inner, total)
    dtype_obj: Arg,
    device_context_ptr: Arg,
) raises:
    var dtype = _raw_dtype_int(dtype_obj)
    var out_addr = _raw_int(out_ptr_obj)
    var in_addr = _raw_int(in_ptr_obj)
    var mean_addr = _raw_int(mean_ptr_obj)
    var var_addr = _raw_int(var_ptr_obj)
    var gamma_addr = _raw_int(gamma_ptr_obj)
    var beta_addr = _raw_int(beta_ptr_obj)
    var eps_val = Float32(_raw_tuple_f64(params, 0))
    var channels_val = _raw_tuple_int(params, 1)
    var inner_val = _raw_tuple_int(params, 2)
    var total = _raw_tuple_int(params, 3)
    var ctx = _raw_ctx(device_context_ptr)

    var handled = False
    comptime for dt in FLOAT_DTYPES:
        comptime if _dtype_arg_on[0, dt]():
            if dtype == dt:
                _batch_norm[dt](
                    out_addr,
                    in_addr,
                    mean_addr,
                    var_addr,
                    gamma_addr,
                    beta_addr,
                    eps_val,
                    channels_val,
                    inner_val,
                    total,
                    ctx,
                )
                handled = True
    if not handled:
        raise Error("unsupported dtype for fast batch_norm: " + String(dtype))


def _softmax_rows_go(
    out_ptr_obj: Arg,
    in_ptr_obj: Arg,
    rows: Arg,
    cols: Arg,
    scale: Arg,
    causal: Arg,
    q_len: Arg,
    dtype_obj: Arg,
    device_context_ptr: Arg,
) raises:
    var dtype = _raw_dtype_int(dtype_obj)
    var out_addr = _raw_int(out_ptr_obj)
    var in_addr = _raw_int(in_ptr_obj)
    var rows_val = _raw_int(rows)
    var cols_val = _raw_int(cols)
    var scale_val = Float32(_raw_f64(scale))
    var causal_val = _raw_int(causal)
    var q_len_val = _raw_int(q_len)
    var ctx = _raw_ctx(device_context_ptr)

    var handled = False
    comptime for dt in FLOAT_DTYPES:
        comptime if _dtype_arg_on[0, dt]():
            if dtype == dt:
                _softmax_rows[dt](
                    out_addr,
                    in_addr,
                    rows_val,
                    cols_val,
                    scale_val,
                    causal_val,
                    q_len_val,
                    ctx,
                )
                handled = True
    if not handled:
        raise Error("unsupported dtype for fast softmax: " + String(dtype))


# ---------------------------------------------------------------------------
# Fused single-query attention (decode step): out = softmax(scale * q @ K^T)
# @ V for q_len == 1, one thread block per (batch * head). Replaces the
# bmm + softmax + bmm chain, whose m=1 GEMMs read K one row per thread
# (uncoalesced) and which costs three kernel launches plus two scratch
# buffers per call.
# ---------------------------------------------------------------------------

comptime ATTN_THREADS = 256
comptime ATTN_MAX_KV = 4096
comptime ATTN_MAX_HD = 256
comptime APPLE_ATTN_THREADS = 64
comptime APPLE_ATTN_MAX_KV = 256
comptime APPLE_ATTN_MAX_HD = 64


@__llvm_metadata(
    MAX_THREADS_PER_BLOCK_METADATA=StaticTuple[Int32, 1](Int32(THREADS))
)
@__name(t"attn_decode_{dtype}_{THREADS}_{MAX_KV}_{MAX_HD}_{RED_STAGES}")
def _attn_decode_kernel[
    dtype: DType,
    THREADS: Int = ATTN_THREADS,
    MAX_KV: Int = ATTN_MAX_KV,
    MAX_HD: Int = ATTN_MAX_HD,
    RED_STAGES: Int = 8,
](
    out_ptr: Pointer[Scalar[dtype], MutAnyOrigin],
    q_ptr: Pointer[Scalar[dtype], ImmutAnyOrigin],
    k_ptr: Pointer[Scalar[dtype], ImmutAnyOrigin],
    v_ptr: Pointer[Scalar[dtype], ImmutAnyOrigin],
    kv_len_arg: Int64,
    head_dim_arg: Int64,
    scale: Float32,
    heads_arg: Int64,
    q_b_stride_arg: Int64,
    q_h_stride_arg: Int64,
    k_b_stride_arg: Int64,
    k_h_stride_arg: Int64,
    k_s_stride_arg: Int64,
    v_b_stride_arg: Int64,
    v_h_stride_arg: Int64,
    v_s_stride_arg: Int64,
):
    """out is (BH, 1, head_dim) contiguous. Q/K/V have unit head-dimension
    stride and explicit batch/head/sequence strides; this consumes both the
    fused-QKV transpose views and padded strided K/V storage without a gather.
    Scores for the block's row are staged in shared memory (hence the
    ATTN_MAX_KV cap), softmax uses f32 max/sum shared-memory tree reductions,
    and the V pass has lane d accumulate output element d so V reads coalesce
    across lanes."""
    # Int is not device-passable (host/device width mismatch); scalars cross
    # the launch ABI as Int64 and index math stays in Int.
    var kv_len = Int(kv_len_arg)
    var head_dim = Int(head_dim_arg)
    var heads = Int(heads_arg)
    var q_b_stride = Int(q_b_stride_arg)
    var q_h_stride = Int(q_h_stride_arg)
    var k_b_stride = Int(k_b_stride_arg)
    var k_h_stride = Int(k_h_stride_arg)
    var k_s_stride = Int(k_s_stride_arg)
    var v_b_stride = Int(v_b_stride_arg)
    var v_h_stride = Int(v_h_stride_arg)
    var v_s_stride = Int(v_s_stride_arg)
    comptime vec_align = 4 * size_of[dtype]()
    var bh = block_idx.x
    var tid = thread_idx.x
    var out_base = bh * head_dim
    var batch = bh // heads
    var head = bh % heads
    var q_base = batch * q_b_stride + head * q_h_stride
    var k_base = batch * k_b_stride + head * k_h_stride
    var v_base = batch * v_b_stride + head * v_h_stride

    var q_smem = stack_allocation[
        MAX_HD, DType.float32, address_space=AddressSpace.SHARED
    ]()
    var s_smem = stack_allocation[
        MAX_KV, DType.float32, address_space=AddressSpace.SHARED
    ]()
    var red = stack_allocation[
        THREADS, DType.float32, address_space=AddressSpace.SHARED
    ]()
    var bcast = stack_allocation[
        2, DType.float32, address_space=AddressSpace.SHARED
    ]()

    for d in range(tid, head_dim, THREADS):
        q_smem[unsafe_offset=d] = q_ptr[unsafe_offset=q_base + d].cast[
            DType.float32
        ]()
    barrier()

    var m = Float32.MIN
    for j in range(tid, kv_len, THREADS):
        var krow = k_base + j * k_s_stride
        var acc = Float32(0)
        for d in range(0, head_dim, 4):
            var k4 = k_ptr.unsafe_load[width=4, alignment=vec_align](
                krow + d
            ).cast[DType.float32]()
            var q4 = q_smem.unsafe_load[width=4, alignment=16](d)
            acc += (q4 * k4).reduce_add()
        var s = acc * scale
        s_smem[unsafe_offset=j] = s
        if s > m:
            m = s
    red[unsafe_offset=tid] = m
    barrier()
    var stride = THREADS // 2
    for _ in range(RED_STAGES):
        if tid < stride:
            if red[unsafe_offset=tid + stride] > red[unsafe_offset=tid]:
                red[unsafe_offset=tid] = red[unsafe_offset=tid + stride]
        barrier()
        stride //= 2
    if tid == 0:
        bcast[unsafe_offset=0] = red[unsafe_offset=0]
    barrier()
    m = bcast[unsafe_offset=0]

    var s = Float32(0)
    for j in range(tid, kv_len, THREADS):
        var e = exp(s_smem[unsafe_offset=j] - m)
        s_smem[unsafe_offset=j] = e
        s += e
    red[unsafe_offset=tid] = s
    barrier()
    stride = THREADS // 2
    for _ in range(RED_STAGES):
        if tid < stride:
            red[unsafe_offset=tid] += red[unsafe_offset=tid + stride]
        barrier()
        stride //= 2
    if tid == 0:
        bcast[unsafe_offset=1] = red[unsafe_offset=0]
    barrier()
    var inv_denom = 1.0 / bcast[unsafe_offset=1]

    # GPT-2's D=64 matches an AMD wavefront. Partition the V reduction over
    # all four wavefronts so the bandwidth-heavy pass uses the whole block,
    # then combine four partial vectors in the now-free reduction scratch.
    # The Apple specialization instantiates this kernel with 64 threads, so
    # keep the four-wavefront reduction entirely out of non-gfx942 builds.
    comptime if _accelerator_arch() == "amdgpu:gfx942":
        if THREADS == 256 and head_dim == 64:
            var lane = tid % 64
            var wave = tid // 64
            var acc = Float32(0)
            for j in range(wave, kv_len, 4):
                acc += (
                    s_smem[unsafe_offset=j]
                    * v_ptr[unsafe_offset=v_base + j * v_s_stride + lane].cast[
                        DType.float32
                    ]()
                )
            red[unsafe_offset=tid] = acc
            barrier()
            if wave == 0:
                acc = (
                    red[unsafe_offset=lane]
                    + red[unsafe_offset=64 + lane]
                    + red[unsafe_offset=128 + lane]
                    + red[unsafe_offset=192 + lane]
                )
                out_ptr[unsafe_offset=out_base + lane] = (acc * inv_denom).cast[
                    dtype
                ]()
            return

    for d in range(tid, head_dim, THREADS):
        var acc = Float32(0)
        for j in range(kv_len):
            acc += (
                s_smem[unsafe_offset=j]
                * v_ptr[unsafe_offset=v_base + j * v_s_stride + d].cast[
                    DType.float32
                ]()
            )
        out_ptr[unsafe_offset=out_base + d] = (acc * inv_denom).cast[dtype]()


@always_inline
def _attn_decode[
    dtype: DType
](
    out_addr: Int,
    q_addr: Int,
    k_addr: Int,
    v_addr: Int,
    bh: Int,
    kv_len: Int,
    head_dim: Int,
    scale: Float32,
    heads: Int,
    q_b_stride: Int,
    q_h_stride: Int,
    k_b_stride: Int,
    k_h_stride: Int,
    k_s_stride: Int,
    v_b_stride: Int,
    v_h_stride: Int,
    v_s_stride: Int,
    ctx: DeviceContext,
) raises:
    comptime if has_accelerator():
        comptime if has_apple_gpu_accelerator():
            if kv_len <= APPLE_ATTN_MAX_KV and head_dim <= APPLE_ATTN_MAX_HD:
                _enqueue_cached[
                    _attn_decode_kernel[
                        dtype,
                        APPLE_ATTN_THREADS,
                        APPLE_ATTN_MAX_KV,
                        APPLE_ATTN_MAX_HD,
                        6,
                    ]
                ](
                    ctx,
                    bh,
                    1,
                    1,
                    APPLE_ATTN_THREADS,
                    _make_ptr[dtype](out_addr).as_unsafe_any_origin(),
                    _make_ptr[dtype](q_addr).as_unsafe_any_origin().as_imm(),
                    _make_ptr[dtype](k_addr).as_unsafe_any_origin().as_imm(),
                    _make_ptr[dtype](v_addr).as_unsafe_any_origin().as_imm(),
                    Int64(kv_len),
                    Int64(head_dim),
                    scale,
                    Int64(heads),
                    Int64(q_b_stride),
                    Int64(q_h_stride),
                    Int64(k_b_stride),
                    Int64(k_h_stride),
                    Int64(k_s_stride),
                    Int64(v_b_stride),
                    Int64(v_h_stride),
                    Int64(v_s_stride),
                )
                return
        _enqueue_cached[_attn_decode_kernel[dtype]](
            ctx,
            bh,
            1,
            1,
            ATTN_THREADS,
            _make_ptr[dtype](out_addr).as_unsafe_any_origin(),
            _make_ptr[dtype](q_addr).as_unsafe_any_origin().as_imm(),
            _make_ptr[dtype](k_addr).as_unsafe_any_origin().as_imm(),
            _make_ptr[dtype](v_addr).as_unsafe_any_origin().as_imm(),
            Int64(kv_len),
            Int64(head_dim),
            scale,
            Int64(heads),
            Int64(q_b_stride),
            Int64(q_h_stride),
            Int64(k_b_stride),
            Int64(k_h_stride),
            Int64(k_s_stride),
            Int64(v_b_stride),
            Int64(v_h_stride),
            Int64(v_s_stride),
        )
    else:
        raise Error("no GPU accelerator available at compile time")


@always_inline
def _gather0_data_dispatch[
    idx_dtype: DType
](
    dtype: DType,
    out_addr: Int,
    weight_addr: Int,
    indices_addr: Int,
    num_indices: Int,
    row_len: Int,
    num_rows: Int,
    ctx: DeviceContext,
) raises:
    var handled = False
    comptime for dt in FLOAT_DTYPES:
        comptime if _dtype_arg_on[0, dt]():
            if dtype == dt:
                _gather0[dt, idx_dtype](
                    out_addr,
                    weight_addr,
                    indices_addr,
                    num_indices,
                    row_len,
                    num_rows,
                    ctx,
                )
                handled = True
    if not handled:
        raise Error("unsupported dtype for fast embedding: " + String(dtype))


def _gather0_go(
    out_ptr_obj: Arg,
    weight_ptr_obj: Arg,
    indices_ptr_obj: Arg,
    idx_dtype_obj: Arg,
    num_indices: Arg,
    row_len: Arg,
    num_rows: Arg,
    dtype_obj: Arg,
    device_context_ptr: Arg,
) raises:
    var dtype = _raw_dtype_int(dtype_obj)
    var idx_dtype = _raw_dtype_int(idx_dtype_obj)
    var out_addr = _raw_int(out_ptr_obj)
    var weight_addr = _raw_int(weight_ptr_obj)
    var indices_addr = _raw_int(indices_ptr_obj)
    var num_indices_val = _raw_int(num_indices)
    var row_len_val = _raw_int(row_len)
    var num_rows_val = _raw_int(num_rows)
    var ctx = _raw_ctx(device_context_ptr)

    comptime if _dtype_arg_on[1, DType.int64]():
        if idx_dtype != DType.int64:
            raise Error("embedding specialization/index dtype mismatch")
        _gather0_data_dispatch[DType.int64](
            dtype,
            out_addr,
            weight_addr,
            indices_addr,
            num_indices_val,
            row_len_val,
            num_rows_val,
            ctx,
        )
    elif _dtype_arg_on[1, DType.int32]():
        if idx_dtype != DType.int32:
            raise Error("embedding specialization/index dtype mismatch")
        _gather0_data_dispatch[DType.int32](
            dtype,
            out_addr,
            weight_addr,
            indices_addr,
            num_indices_val,
            row_len_val,
            num_rows_val,
            ctx,
        )
    else:
        raise Error(
            "unsupported index dtype for fast embedding: " + String(idx_dtype)
        )


# ---------------------------------------------------------------------------
# Full any()/all() over a bool tensor -> scalar bool. Viewed as one row of
# `size` bools reduced over the trailing axis, i.e. the generic reduction
# skeleton's (outer=1, reduce=size, inner=1) geometry with the AnyOp / AllOp
# accumulator. aten caps this path at < 4.2M elements, so it is almost always
# the small regime, where the split chooser degrades to a single 256-thread
# block and no merge launch -- which matters because the HF sdpa mask check
# runs this every decode step.
# ---------------------------------------------------------------------------


def _all_bool(
    out_addr: Int, in_addr: Int, size: Int, ctx: DeviceContext
) raises:
    """Full all() over a bool tensor -> scalar bool (AND)."""
    _reduce_generic[AllOp, DType.bool](out_addr, in_addr, 1, size, 1, ctx)


def _all_bool_go(
    out_ptr_obj: Arg,
    in_ptr_obj: Arg,
    size: Arg,
    device_context_ptr: Arg,
) raises:
    _all_bool(
        _raw_int(out_ptr_obj),
        _raw_int(in_ptr_obj),
        _raw_int(size),
        _raw_ctx(device_context_ptr),
    )


# ---------------------------------------------------------------------------
# any() over a bool tensor -> scalar bool: same block reduction as all().
# ---------------------------------------------------------------------------


@always_inline
def _any_bool(
    out_addr: Int, in_addr: Int, size: Int, ctx: DeviceContext
) raises:
    """Full any() over a bool tensor -> scalar bool (OR)."""
    _reduce_generic[AnyOp, DType.bool](out_addr, in_addr, 1, size, 1, ctx)


def _any_bool_go(
    out_ptr_obj: Arg,
    in_ptr_obj: Arg,
    size: Arg,
    device_context_ptr: Arg,
) raises:
    _any_bool(
        _raw_int(out_ptr_obj),
        _raw_int(in_ptr_obj),
        _raw_int(size),
        _raw_ctx(device_context_ptr),
    )


# Row-wise argmax lives in `argreduce_kernels.mojo` (`_argreduce_rows`,
# with `_argreduce_cols` for a strided reduce axis): one
# comptime-parametrized mechanism shared with reduction' argmin, so
# the two ops cannot drift apart in semantics or launch geometry.


# Row-wise max (values only, aten.max() with no dim) is the generic reduction
# skeleton's MaxOp: no kernel of its own here, only the MaxSpec registration
# below.


# ---------------------------------------------------------------------------
# Cumulative sum, INNER (trailing dim, any rank) or OUTER (dim=0, rank 2)
# family — see cumsum_kernels.mojo for the fast, NVIDIA-measured kernels
# (one block cooperating over a "line" via block.prefix_sum for INNER, one
# thread per independent column for OUTER, plus a 3-pass workspace scan for
# the few-very-long-lines regime).
#
# `_cumsum_rows_portable`/`_cumsum_cols_portable` below are a DIFFERENT,
# simpler thing: a plain one-task-per-line serial accumulate through
# `_parallel_for`, used whenever the GPU `ctx.api()` is not `"cuda"` (AMD,
# Apple): `block.prefix_sum`/`block.sum` are portable MAX primitives and this
# whole kernel family DOES cross-compile for gfx942 (verified with
# scripts/compare_kernel_asm.py), but the fast kernels were only ever
# MEASURED on NVIDIA (H100) — see the PR that added this file. Per
# AGENTS.md's "To check a kernel change against a GPU you do not have", an
# unmeasured architecture gets the change gated off, not shipped on faith, so
# non-CUDA GPUs keep running the exact naive kernel `main` ran for cumsum
# before this file existed. AMD (gfx942) was later measured correct on this
# portable path for every dtype and both routes (INNER and OUTER dim=0) --
# see `_is_cumsum_dtype`'s `fast_ok` gate in ops_reductions.mojo, which is
# where "cuda" or "hip" reaches the OUTER route and the bf16/f16 dtypes at
# all; Metal stays on the pre-existing (int64/int32/float32, trailing-dim)
# surface, unmeasured.
# ---------------------------------------------------------------------------


@always_inline
def _cumsum_rows_portable[
    dtype: DType
](out_addr: Int, in_addr: Int, rows: Int, cols: Int, ctx: DeviceContext) raises:
    comptime acc = _acc_dtype[dtype]()
    var out_ptr = _make_ptr[dtype](out_addr)
    var in_ptr = _make_ptr[dtype](in_addr)

    @always_inline
    @__parameter
    @__copy_capture(out_ptr, in_ptr)
    def func[width: Int, alignment: Int = 1](idx: Coord):
        var r = Int(idx[0].value())
        var base = r * cols
        var total = Scalar[acc](0)
        for j in range(cols):
            total += in_ptr[unsafe_offset=base + j].cast[acc]()
            out_ptr[unsafe_offset=base + j] = total.cast[dtype]()

    _parallel_for[func](rows, ctx)


@always_inline
def _cumsum_cols_portable[
    dtype: DType
](out_addr: Int, in_addr: Int, rows: Int, cols: Int, ctx: DeviceContext) raises:
    """Portable (CPU or non-CUDA GPU) fallback for dim=0: one parallel task
    per column. Unlike `_cumsum_rows_portable`, this has no naive-kernel
    precedent on `main` (dim=0 simply raised NotImplementedError there).
    `op_cumsum`'s `fast_ok` gate (ops_reductions.mojo) reaches dim=0 on CUDA
    and HIP devices (this body serves HIP there, since `ctx.api()` picks the
    fast block.prefix_sum kernel only for `"cuda"`) and declines it
    elsewhere (the CPU mojo device included), so on today's dispatch this
    body runs for HIP only; it stays a real, tested implementation (not a
    stub) so the gate can be loosened further (e.g. Metal) without a new
    kernel."""
    comptime acc = _acc_dtype[dtype]()
    var out_ptr = _make_ptr[dtype](out_addr)
    var in_ptr = _make_ptr[dtype](in_addr)

    @always_inline
    @__parameter
    @__copy_capture(out_ptr, in_ptr)
    def func[width: Int, alignment: Int = 1](idx: Coord):
        var c = Int(idx[0].value())
        var total = Scalar[acc](0)
        var i = 0
        while i < rows:
            var addr = i * cols + c
            total += in_ptr[unsafe_offset=addr].cast[acc]()
            out_ptr[unsafe_offset=addr] = total.cast[dtype]()
            i += 1

    _parallel_for[func](cols, ctx)


@always_inline
def _cumsum_inner_into[
    dtype: DType
](out_addr: Int, in_addr: Int, rows: Int, cols: Int, ctx: DeviceContext) raises:
    """INNER family entry: dim is the trailing (stride-1) axis.

    The fast `block.prefix_sum`-based kernels below are NVIDIA-only by
    measurement, not by portability (`block.prefix_sum`/`block.sum` cross-
    compile fine for AMD -- verified with scripts/compare_kernel_asm.py,
    --accelerator gfx942). `ctx.api() == "cuda"` gates the fast kernels off
    on anything else (HIP included, on today's measurement -- it runs the
    portable fallback below instead), independently of `op_cumsum`'s own
    `fast_ok` dim/dtype gate in ops_reductions.mojo, which is what decides
    whether HIP reaches this function at all for bf16/f16 or dim=0.
    """
    if ctx.api() == "cuda":
        comptime if has_accelerator():
            var gout = _make_ptr[dtype](out_addr).as_unsafe_any_origin()
            var gin = _make_ptr[dtype](in_addr).as_unsafe_any_origin().as_imm()
            # sm_count_floor: below this many independent lines, one block
            # per line cannot fill the device — see FILL_WAVES in
            # cumsum_kernels.mojo. Read at runtime (not the compile-time
            # `default_device_info` table) because two cards of the same
            # architecture can differ here (H100 PCIe: 114 SMs, H100 SXM:
            # 132 — the table reports 132 for both).
            var sm_count_floor = _device_sm_count(ctx) * FILL_WAVES
            if rows >= sm_count_floor:
                enqueue_cumsum_rows[dtype](ctx, gout, gin, rows, cols)
            else:
                comptime acc = _acc_dtype[dtype]()
                var ws_n = max(1, cumsum_workspace_lines[dtype](rows, cols))
                var ws = ctx.enqueue_create_buffer[acc](ws_n)
                enqueue_cumsum_rows_workspace[dtype](
                    ctx,
                    gout,
                    gin,
                    rows,
                    cols,
                    ws.unsafe_ptr().as_unsafe_any_origin(),
                )
                # Keep the workspace alive until its free is enqueued after
                # the finish kernel (stream-ordered, matches the GEMM
                # split-K workspace pattern in tmb/kernels/matmul/entry.mojo).
                _ = ws^
        else:
            raise Error("no GPU accelerator available at compile time")
    else:
        _cumsum_rows_portable[dtype](out_addr, in_addr, rows, cols, ctx)


@always_inline
def _cumsum_outer_into[
    dtype: DType
](out_addr: Int, in_addr: Int, rows: Int, cols: Int, ctx: DeviceContext) raises:
    """OUTER family entry: dim=0 on a contiguous rank-2 tensor. Same
    CUDA-only fast-path gate as `_cumsum_inner_into` above."""
    if ctx.api() == "cuda":
        comptime if has_accelerator():
            var gout = _make_ptr[dtype](out_addr).as_unsafe_any_origin()
            var gin = _make_ptr[dtype](in_addr).as_unsafe_any_origin().as_imm()
            enqueue_cumsum_cols[dtype](ctx, gout, gin, rows, cols)
        else:
            raise Error("no GPU accelerator available at compile time")
    else:
        _cumsum_cols_portable[dtype](out_addr, in_addr, rows, cols, ctx)


# ---------------------------------------------------------------------------
# METH_FASTCALL wrappers: raw CPython argument unpacking (no owning
# PythonObject per argument). Argument types are guaranteed by the internal
# Python callers; raise sites are unsupported-dtype guards gated upstream.
# ---------------------------------------------------------------------------


def _batch_norm_dispatcher(argv: Argv, argc: Int) raises:
    var args = argv
    _batch_norm_go(
        args[unsafe_offset=0],
        args[unsafe_offset=1],
        args[unsafe_offset=2],
        args[unsafe_offset=3],
        args[unsafe_offset=4],
        args[unsafe_offset=5],
        args[unsafe_offset=6],
        args[unsafe_offset=7],
        args[unsafe_offset=8],
    )


def _softmax_rows_dispatcher(argv: Argv, argc: Int) raises:
    var args = argv
    _softmax_rows_go(
        args[unsafe_offset=0],
        args[unsafe_offset=1],
        args[unsafe_offset=2],
        args[unsafe_offset=3],
        args[unsafe_offset=4],
        args[unsafe_offset=5],
        args[unsafe_offset=6],
        args[unsafe_offset=7],
        args[unsafe_offset=8],
    )


def _softmax_rows_dropout_dispatcher(argv: Argv, argc: Int) raises:
    var args = argv
    _softmax_rows_dropout_go(
        args[unsafe_offset=0],
        args[unsafe_offset=1],
        args[unsafe_offset=2],
        args[unsafe_offset=3],
        args[unsafe_offset=4],
        args[unsafe_offset=5],
        args[unsafe_offset=6],
        args[unsafe_offset=7],
        args[unsafe_offset=8],
        args[unsafe_offset=9],
        args[unsafe_offset=10],
        args[unsafe_offset=11],
        args[unsafe_offset=12],
        args[unsafe_offset=13],
        args[unsafe_offset=14],
    )


def _gather0_dispatcher(argv: Argv, argc: Int) raises:
    var args = argv
    _gather0_go(
        args[unsafe_offset=0],
        args[unsafe_offset=1],
        args[unsafe_offset=2],
        args[unsafe_offset=3],
        args[unsafe_offset=4],
        args[unsafe_offset=5],
        args[unsafe_offset=6],
        args[unsafe_offset=7],
        args[unsafe_offset=8],
    )


def _all_bool_dispatcher(argv: Argv, argc: Int) raises:
    var args = argv
    _all_bool_go(
        args[unsafe_offset=0],
        args[unsafe_offset=1],
        args[unsafe_offset=2],
        args[unsafe_offset=3],
    )


def _any_bool_dispatcher(argv: Argv, argc: Int) raises:
    var args = argv
    _any_bool_go(
        args[unsafe_offset=0],
        args[unsafe_offset=1],
        args[unsafe_offset=2],
        args[unsafe_offset=3],
    )


comptime SPEC_MAXROWS_DTYPES: List[DType] = [
    DType.float32,
    DType.float16,
    DType.bfloat16,
    DType.int64,
    DType.int32,
    DType.float64,
]


def _argmax_spec_into_go(
    a_o: Arg,
    rdims_t: Arg,
    keepdim_o: Arg,
    out_o: Arg,
) raises:
    ref a = _spec_ptr(a_o)[]
    ref out = _spec_ptr(out_o)[]
    if not _dtype_supported[SPEC_MAXROWS_DTYPES](a.dtype):
        raise Error("mojo spec argmax: unsupported dtype ", a.dtype)
    if a.numel == 0:
        raise Error("mojo spec argmax: empty input")
    _argreduce_spec_into[SPEC_MAXROWS_DTYPES, False](a, out, rdims_t, a.ctx())


def _cumsum_spec_into_go(a_o: Arg, dim_o: Arg, out_o: Arg) raises:
    """Cumulative sum over the trailing dim (any rank) or dim=0 of a rank-2
    tensor; full-shape output. Python (`fast_aten_cumsum`) normalizes the
    dim and pre-materializes non-contiguous/other-dim inputs, but this
    boundary re-validates rather than trusting the caller, matching every
    other spec bridge in this file."""
    ref a = _spec_ptr(a_o)[]
    ref out = _spec_ptr(out_o)[]
    if not _dtype_supported[CUMSUM_DTYPES](a.dtype):
        raise Error("mojo spec cumsum: unsupported dtype ", a.dtype)
    if a.rank < 1 or a.numel == 0:
        raise Error("mojo spec cumsum: empty or rank-0 input")
    var dim = _raw_int(dim_o)
    if dim < 0 or dim >= a.rank:
        raise Error("mojo spec cumsum: dim out of range")
    if not a.contig:
        raise Error(
            "mojo spec cumsum: input must be contiguous"
            " (Python pre-materializes)"
        )
    _check_into(a, out, a.dtype)
    var addr = out.ptr
    var ctx = a.ctx()

    if dim == a.rank - 1:
        var cols = a.shape[MAX_RANK - 1]
        var rows = a.numel // cols
        comptime for dt in CUMSUM_DTYPES:
            comptime if _dtype_arg_on[0, dt]():
                if a.dtype == dt:
                    _cumsum_inner_into[dt](addr, a.ptr, rows, cols, ctx)
    elif dim == 0 and a.rank == 2:
        var rows = a.dim(0)
        var cols = a.dim(1)
        comptime for dt in CUMSUM_DTYPES:
            comptime if _dtype_arg_on[0, dt]():
                if a.dtype == dt:
                    _cumsum_outer_into[dt](addr, a.ptr, rows, cols, ctx)
    else:
        raise Error(
            "mojo spec cumsum: dim must be the trailing dim, or dim=0 on a"
            " rank-2 tensor"
        )


def _batch_norm_spec_into_go(
    in_o: Arg,
    mean_o: Arg,
    var_o: Arg,
    gamma_o: Arg,
    beta_o: Arg,
    eps_o: Arg,
    out_o: Arg,
) raises:
    """Inference batch norm: geometry (channels/inner) derived from the
    input spec, output alloc and launch in one boundary call, reusing the
    `_batch_norm` kernel above."""
    ref inp = _spec_ptr(in_o)[]
    ref out = _spec_ptr(out_o)[]
    ref meanp = _spec_ptr(mean_o)[]
    ref varp = _spec_ptr(var_o)[]
    ref gammap = _spec_ptr(gamma_o)[]
    ref betap = _spec_ptr(beta_o)[]
    var eps = Float32(_raw_f64(eps_o))

    if inp.rank < 2:
        raise Error("mojo spec batch_norm: input rank must be >= 2")
    if inp.numel == 0:
        raise Error("mojo spec batch_norm: empty input")
    if not (
        inp.contig
        and meanp.contig
        and varp.contig
        and gammap.contig
        and betap.contig
    ):
        raise Error("mojo spec batch_norm: all inputs must be contiguous")
    if (
        meanp.dtype != inp.dtype
        or varp.dtype != inp.dtype
        or gammap.dtype != inp.dtype
        or betap.dtype != inp.dtype
    ):
        raise Error("mojo spec batch_norm: stat/affine dtypes must match input")
    if not _dtype_supported[List[DType](FLOAT_DTYPES)](inp.dtype):
        raise Error("mojo spec batch_norm: unsupported dtype ", inp.dtype)

    var channels = inp.dim(1)
    var inner = 1
    for i in range(MAX_RANK - inp.rank + 2, MAX_RANK):
        inner *= inp.shape[i]

    var ctx = inp.ctx()
    var nbytes = inp.numel * inp.itemsize
    _ = nbytes
    _check_into(inp, out, inp.dtype)
    var addr = out.ptr
    comptime for dt in FLOAT_DTYPES:
        comptime if _dtype_arg_on[0, dt]():
            if inp.dtype == dt:
                _batch_norm[dt](
                    addr,
                    inp.ptr,
                    meanp.ptr,
                    varp.ptr,
                    gammap.ptr,
                    betap.ptr,
                    eps,
                    channels,
                    inner,
                    inp.numel,
                    ctx,
                )


def _softmax_spec_into_go(a_o: Arg, out_o: Arg) raises:
    """Plain softmax over the trailing dim (scale=1, no causal mask);
    full-shape output. The non-trailing dim transpose recursion and the
    half_to_float cast stay in Python."""
    ref a = _spec_ptr(a_o)[]
    ref out = _spec_ptr(out_o)[]
    if not _dtype_supported[List[DType](FLOAT_DTYPES)](a.dtype):
        raise Error("mojo spec softmax: unsupported dtype ", a.dtype)
    if a.rank < 1 or a.numel == 0:
        raise Error("mojo spec softmax: empty or rank-0 input")

    var cols = a.shape[MAX_RANK - 1]
    var rows = a.numel // cols
    var ctx = a.ctx()
    var nbytes = a.numel * a.itemsize
    _ = nbytes
    _check_into(a, out, a.dtype)
    var addr = out.ptr
    if a.contig:
        comptime for dt in FLOAT_DTYPES:
            comptime if _dtype_arg_on[0, dt]():
                if a.dtype == dt:
                    _softmax_rows[dt](
                        addr, a.ptr, rows, cols, Float32(1.0), 0, 1, ctx
                    )
    else:
        raise Error(
            "mojo spec softmax: input must be contiguous"
            " (Python pre-materializes)"
        )


def _softmax_f64_spec_into_go(a_o: Arg, out_o: Arg, log_o: Arg) raises:
    """float64 softmax (log_o != 0: log-softmax) over the trailing dim of a
    contiguous operand into a same-shape float64 output: the `dtype=`
    route of softmax / log_softmax. The other dtypes stay on `SoftmaxSpec`
    / the reduction family's `LogSoftmaxSpec`."""
    ref a = _spec_ptr(a_o)[]
    ref out = _spec_ptr(out_o)[]
    if a.dtype != DType.float64 or a.rank < 1 or a.numel == 0:
        raise Error("mojo spec softmax_f64: a non-empty float64 input")
    if not a.contig:
        raise Error("mojo spec softmax_f64: input must be contiguous")
    _check_into(a, out, a.dtype)
    var cols = a.shape[MAX_RANK - 1]
    softmax_rows_f64(
        out.ptr, a.ptr, a.numel // cols, cols, _raw_int(log_o) != 0, a.ctx()
    )


def _attn_decode_spec_into_go(
    q_o: Arg,
    k_o: Arg,
    v_o: Arg,
    scale_o: Arg,
    out_o: Arg,
) raises:
    """Fused decode attention (q_len == 1, not causal), GPU only — one
    boundary call replacing the Python gates + geometry + alloc + launch.
    Q/K/V read through their real strides; the innermost head dimension must
    remain contiguous for vector loads."""
    ref q = _spec_ptr(q_o)[]
    ref out = _spec_ptr(out_o)[]
    ref k = _spec_ptr(k_o)[]
    ref v = _spec_ptr(v_o)[]
    var scale = Float32(_raw_f64(scale_o))

    if q.rank != 4 or k.rank != 4 or v.rank != 4:
        raise Error("mojo spec attn_decode: rank != 4")
    if q.dtype != k.dtype or q.dtype != v.dtype:
        raise Error("mojo spec attn_decode: dtypes differ")
    if not _dtype_supported[List[DType](FLOAT_DTYPES)](q.dtype):
        raise Error("mojo spec attn_decode: unsupported dtype ", q.dtype)
    var b = q.shape[MAX_RANK - 4]
    var h = q.shape[MAX_RANK - 3]
    var q_len = q.shape[MAX_RANK - 2]
    var head_dim = q.shape[MAX_RANK - 1]
    var kv_len = k.shape[MAX_RANK - 2]
    if q_len != 1:
        raise Error("mojo spec attn_decode: q_len != 1")
    for i in range(4):
        if k.shape[MAX_RANK - 4 + i] != v.shape[MAX_RANK - 4 + i]:
            raise Error("mojo spec attn_decode: k/v shapes differ")
    if (
        b != k.shape[MAX_RANK - 4]
        or h != k.shape[MAX_RANK - 3]
        or head_dim != k.shape[MAX_RANK - 1]
    ):
        raise Error("mojo spec attn_decode: q/k shapes incompatible")
    if b * h * kv_len * head_dim == 0:
        raise Error("mojo spec attn_decode: empty input")
    if (
        q.strides[MAX_RANK - 1] != 1
        or k.strides[MAX_RANK - 1] != 1
        or v.strides[MAX_RANK - 1] != 1
        or k.strides[MAX_RANK - 2] != head_dim
        or v.strides[MAX_RANK - 2] != head_dim
    ):
        raise Error("mojo spec attn_decode: unsupported q/k/v strides")

    var ctx = q.ctx()
    if head_dim % 4 != 0 or head_dim > ATTN_MAX_HD or kv_len > ATTN_MAX_KV:
        raise Error("mojo spec attn_decode: size caps")

    var numel = b * h * head_dim
    var nbytes = numel * q.itemsize
    _ = nbytes
    _check_into_sized(q, out, numel, q.dtype)
    var addr = out.ptr
    comptime for dt in FLOAT_DTYPES:
        comptime if _dtype_arg_on[0, dt]():
            if q.dtype == dt:
                _attn_decode[dt](
                    addr,
                    q.ptr,
                    k.ptr,
                    v.ptr,
                    b * h,
                    kv_len,
                    head_dim,
                    scale,
                    h,
                    q.strides[MAX_RANK - 4],
                    q.strides[MAX_RANK - 3],
                    k.strides[MAX_RANK - 4],
                    k.strides[MAX_RANK - 3],
                    k.strides[MAX_RANK - 2],
                    v.strides[MAX_RANK - 4],
                    v.strides[MAX_RANK - 3],
                    v.strides[MAX_RANK - 2],
                    ctx,
                )
    var oshape = IndexList[MAX_RANK](1)
    oshape[MAX_RANK - 4] = b
    oshape[MAX_RANK - 3] = h
    oshape[MAX_RANK - 2] = 1
    oshape[MAX_RANK - 1] = head_dim


# ---------------------------------------------------------------------------
# Python module definition
# ---------------------------------------------------------------------------


@export
def tmb_call(argv: Argv, argc: Int, err: ErrBuf, errcap: Int) abi("C") -> Int32:
    """C entry of this family: one kernel per build (see `OP`).
    Slots are described in op_utils (`Arg`); errors come back as (rc=1, message).
    """
    try:
        comptime if _op_on["MeanSpec"]():
            _spec_dispatcher4[
                _rowred_spec_into_go[MeanOp], "a scalar-reduction spec op"
            ](argv, argc)
            return 0
        comptime if _op_on["MaxSpec"]():
            _spec_dispatcher4[
                _rowred_spec_into_go[MaxOp], "a scalar-reduction spec op"
            ](argv, argc)
            return 0
        comptime if _op_on["ArgmaxSpec"]():
            _spec_dispatcher4[_argmax_spec_into_go, "ArgmaxSpec"](argv, argc)
            return 0
        comptime if _op_on["CumsumSpec"]():
            _spec_dispatcher3[_cumsum_spec_into_go, "CumsumSpec"](argv, argc)
            return 0
        comptime if _op_on["BatchNormSpec"]():
            _spec_dispatcher7[_batch_norm_spec_into_go, "BatchNormSpec"](
                argv, argc
            )
            return 0
        comptime if _op_on["SoftmaxSpec"]():
            _spec_dispatcher2[_softmax_spec_into_go, "SoftmaxSpec"](argv, argc)
            return 0
        comptime if _op_on["SoftmaxF64Spec"]():
            _spec_dispatcher3[_softmax_f64_spec_into_go, "SoftmaxF64Spec"](
                argv, argc
            )
            return 0
        comptime if _op_on["AttnDecodeSpec"]():
            _spec_dispatcher5[_attn_decode_spec_into_go, "AttnDecodeSpec"](
                argv, argc
            )
            return 0
        comptime if _op_on["BatchNormInference"]():
            _batch_norm_dispatcher(argv, argc)
            return 0
        comptime if _op_on["SoftmaxRows"]():
            _softmax_rows_dispatcher(argv, argc)
            return 0
        comptime if _op_on["SoftmaxRowsDropoutF32"]():
            _softmax_rows_dropout_dispatcher(argv, argc)
            return 0
        comptime if _op_on["Gather0"]():
            _gather0_dispatcher(argv, argc)
            return 0
        comptime if _op_on["AllBool"]():
            _all_bool_dispatcher(argv, argc)
            return 0
        comptime if _op_on["AnyBool"]():
            _any_bool_dispatcher(argv, argc)
            return 0
        raise Error(NO_OP_COMPILED)
    except e:
        return _tmb_entry_error(err, errcap, e)
