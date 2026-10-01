# ===----------------------------------------------------------------------=== #
# aten::_embedding_bag / _embedding_bag_forward_only on the mojo device: a
# port of ATen's EmbeddingBag_updateOutputKernel_sum_mean / _max
# (native/cuda/EmbeddingBag.cu). One thread owns one (bag, feature) pair
# and walks the bag's indices in order, so every output is deterministic and
# reduced in the same order as CUDA: sum / mean accumulate in the dtype's
# accumulator (float for the half types), max keeps CUDA's first-wins
# comparison. Indices and offsets arrive as int64 (the op promotes them);
# offset2bag, bag_size and max_indices are written as int64.
#
# An index outside [0, numRows) is skipped and raises the int32 flag at
# `err` (CUDA device-asserts), which the op reads back and raises.
# ===----------------------------------------------------------------------=== #

from std.math import pow
from max.gpu import (
    MAX_THREADS_PER_BLOCK_METADATA,
    block_dim,
    block_idx,
    grid_dim,
    thread_idx,
)
from std.sys.info import has_accelerator, has_apple_gpu_accelerator
from std.utils.static_tuple import StaticTuple

from tmb.kernels.common.op_utils import (
    Arg,
    Argv,
    GS_THREADS,
    _enqueue_cached,
    _gs_blocks,
    _make_ptr,
    _raw_ctx,
    _raw_dtype_int,
    _raw_f64,
    _raw_int,
    _raw_tuple_int,
    _spec_dispatcher10,
    _spec_dispatcher12,
)
from tmb.kernels.common.variant_gates import (
    ErrBuf,
    NO_OP_COMPILED,
    _dtype_arg_on,
    _op_on,
    _tmb_entry_error,
)

comptime BAG_DTYPES = [
    DType.float32,
    DType.float16,
    DType.bfloat16,
    DType.float64,
]

# EmbeddingBagMode (native/EmbeddingBag.h)
comptime MODE_SUM = 0
comptime MODE_MEAN = 1
comptime MODE_MAX = 2


@always_inline
def _acc_dtype[dtype: DType]() -> DType:
    """ATen's `acc_type<scalar_t, true>`: float for the half types."""
    comptime if dtype == DType.float64:
        return DType.float64
    else:
        return DType.float32


@__name(t"embedding_bag_fwd_{dtype}_t{GS_THREADS}")
@__llvm_metadata(
    MAX_THREADS_PER_BLOCK_METADATA=StaticTuple[Int32, 1](Int32(GS_THREADS))
)
def _embedding_bag_kernel[
    dtype: DType
](
    output: Pointer[Scalar[dtype], MutAnyOrigin],
    offset2bag: Pointer[Int64, MutAnyOrigin],
    bag_size: Pointer[Int64, MutAnyOrigin],
    max_indices: Pointer[Int64, MutAnyOrigin],
    psw: Pointer[Scalar[dtype], MutAnyOrigin],
    indices: Pointer[Int64, MutAnyOrigin],
    offsets: Pointer[Int64, MutAnyOrigin],
    weight: Pointer[Scalar[dtype], MutAnyOrigin],
    err: Pointer[Int32, MutAnyOrigin],
    num_indices: Int64,
    num_bags: Int64,
    feature_size: Int64,
    weight_stride0: Int64,
    weight_stride1: Int64,
    mode: Int64,
    padding_idx: Int64,
    num_rows: Int64,
    psw_stride: Int64,
    has_psw: Int64,
):
    comptime acc_t = _acc_dtype[dtype]()
    var features = Int(feature_size)
    var total = Int(num_bags) * features
    var i = Int(block_idx.x) * Int(block_dim.x) + Int(thread_idx.x)
    var step = Int(grid_dim.x) * Int(block_dim.x)
    while i < total:
        var bag = i // features
        var feature = i - bag * features
        var begin = 0 if bag == 0 else Int(offsets[unsafe_offset=bag])
        var end = Int(num_indices)
        if bag < Int(num_bags) - 1:
            end = Int(offsets[unsafe_offset=bag + 1])
        var col = feature * Int(weight_stride1)
        var count = 0
        if mode == MODE_MAX:
            var best = Scalar[dtype](0)
            var best_word = -1
            for emb in range(begin, end):
                var word = Int(indices[unsafe_offset=emb])
                if word < 0 or word >= Int(num_rows):
                    err[] = 1
                    continue
                var pad = word == Int(padding_idx)
                var v = weight[unsafe_offset=word * Int(weight_stride0) + col]
                if count == 0 or v > best:
                    if not pad:
                        best = v
                        best_word = word
                if not pad:
                    count += 1
                if feature == 0:
                    offset2bag[unsafe_offset=emb] = Int64(bag)
            max_indices[unsafe_offset=i] = Int64(best_word)
            output[unsafe_offset=i] = best
        else:
            var acc = Scalar[acc_t](0)
            for emb in range(begin, end):
                var word = Int(indices[unsafe_offset=emb])
                if word < 0 or word >= Int(num_rows):
                    err[] = 1
                    continue
                var pad = word == Int(padding_idx)
                var v = weight[unsafe_offset=word * Int(weight_stride0) + col]
                if pad:
                    v = Scalar[dtype](0)
                if has_psw != 0:
                    # nvcc contracts CUDA's `sum += w * v` into one fma.
                    acc = (
                        psw[unsafe_offset=emb * Int(psw_stride)]
                        .cast[acc_t]()
                        .fma(v.cast[acc_t](), acc)
                    )
                else:
                    acc += v.cast[acc_t]()
                if not pad:
                    count += 1
                if feature == 0:
                    offset2bag[unsafe_offset=emb] = Int64(bag)
            if mode == MODE_MEAN and count != 0:
                acc = acc / Scalar[acc_t](count)
            output[unsafe_offset=i] = acc.cast[dtype]()
        if end < begin:
            err[] = 2
        bag_size[unsafe_offset=bag] = Int64(count)
        i += step


def _embedding_bag_go(
    output_o: Arg,
    offset2bag_o: Arg,
    bag_size_o: Arg,
    max_indices_o: Arg,
    psw_o: Arg,
    indices_o: Arg,
    offsets_o: Arg,
    weight_o: Arg,
    err_o: Arg,
    # (num_indices, num_bags, feature_size, weight_stride0, weight_stride1,
    #  mode, padding_idx, num_rows, psw_stride, has_psw)
    params: Arg,
    dtype_o: Arg,
    ctx_o: Arg,
) raises:
    var dtype = _raw_dtype_int(dtype_o)
    var ctx = _raw_ctx(ctx_o)
    var num_bags = _raw_tuple_int(params, 1)
    var feature_size = _raw_tuple_int(params, 2)
    var handled = False
    comptime for dt in BAG_DTYPES:
        comptime if _dtype_arg_on[0, dt]():
            if dtype == dt:
                handled = True
                comptime if not has_accelerator():
                    raise Error("no GPU accelerator available at compile time")
                elif dt == DType.float64 and has_apple_gpu_accelerator():
                    raise Error("float64 is not supported on Apple GPU")
                else:
                    _enqueue_cached[_embedding_bag_kernel[dt]](
                        ctx,
                        _gs_blocks(num_bags * feature_size),
                        1,
                        1,
                        GS_THREADS,
                        _make_ptr[dt](
                            _raw_int(output_o)
                        ).as_unsafe_any_origin(),
                        _make_ptr[DType.int64](
                            _raw_int(offset2bag_o)
                        ).as_unsafe_any_origin(),
                        _make_ptr[DType.int64](
                            _raw_int(bag_size_o)
                        ).as_unsafe_any_origin(),
                        _make_ptr[DType.int64](
                            _raw_int(max_indices_o)
                        ).as_unsafe_any_origin(),
                        _make_ptr[dt](_raw_int(psw_o)).as_unsafe_any_origin(),
                        _make_ptr[DType.int64](
                            _raw_int(indices_o)
                        ).as_unsafe_any_origin(),
                        _make_ptr[DType.int64](
                            _raw_int(offsets_o)
                        ).as_unsafe_any_origin(),
                        _make_ptr[dt](
                            _raw_int(weight_o)
                        ).as_unsafe_any_origin(),
                        _make_ptr[DType.int32](
                            _raw_int(err_o)
                        ).as_unsafe_any_origin(),
                        Int64(_raw_tuple_int(params, 0)),
                        Int64(num_bags),
                        Int64(feature_size),
                        Int64(_raw_tuple_int(params, 3)),
                        Int64(_raw_tuple_int(params, 4)),
                        Int64(_raw_tuple_int(params, 5)),
                        Int64(_raw_tuple_int(params, 6)),
                        Int64(_raw_tuple_int(params, 7)),
                        Int64(_raw_tuple_int(params, 8)),
                        Int64(_raw_tuple_int(params, 9)),
                    )
    if not handled:
        raise Error("EmbeddingBagForward: unsupported dtype ", dtype)


# ---------------------------------------------------------------------------
# EmbeddingRenorm: ATen's `renorm_kernel` (native/cuda/Embedding.cu) with one
# thread per (already unique, in-range) row instead of one block: the row's
# p-norm in the accumulator dtype (|x| for p=1, x*x for p=2, pow(x, p)
# otherwise -- no abs, as upstream), and when it exceeds max_norm the row
# is scaled by `max_norm / (norm + 1e-7)` rounded to the dtype first.
# ---------------------------------------------------------------------------


@__name(t"embedding_renorm_{dtype}_t{GS_THREADS}")
@__llvm_metadata(
    MAX_THREADS_PER_BLOCK_METADATA=StaticTuple[Int32, 1](Int32(GS_THREADS))
)
def _embedding_renorm_kernel[
    dtype: DType
](
    weights: Pointer[Scalar[dtype], MutAnyOrigin],
    rows: Pointer[Int64, MutAnyOrigin],
    num_rows: Int64,
    dim: Int64,
    stride0: Int64,
    stride1: Int64,
    max_norm: Scalar[_acc_dtype[dtype]()],
    norm_type: Scalar[_acc_dtype[dtype]()],
):
    comptime acc_t = _acc_dtype[dtype]()
    var i = Int(block_idx.x) * Int(block_dim.x) + Int(thread_idx.x)
    var step = Int(grid_dim.x) * Int(block_dim.x)
    while i < Int(num_rows):
        var base = Int(rows[unsafe_offset=i]) * Int(stride0)
        var v = Scalar[acc_t](0)
        for k in range(Int(dim)):
            var x = weights[unsafe_offset=base + k * Int(stride1)].cast[acc_t]()
            if norm_type == 1:
                v += abs(x)
            elif norm_type == 2:
                v += x * x
            else:
                v += pow(x, norm_type)
        var norm = pow(v, Scalar[acc_t](1) / norm_type)
        if norm > max_norm:
            var factor = (max_norm / (norm + Scalar[acc_t](1e-7))).cast[dtype]()
            for k in range(Int(dim)):
                var at = base + k * Int(stride1)
                weights[unsafe_offset=at] = weights[unsafe_offset=at] * factor
        i += step


def _embedding_renorm_go(
    weights_o: Arg,
    rows_o: Arg,
    num_rows_o: Arg,
    dim_o: Arg,
    stride0_o: Arg,
    stride1_o: Arg,
    max_norm_o: Arg,
    norm_type_o: Arg,
    dtype_o: Arg,
    ctx_o: Arg,
) raises:
    var dtype = _raw_dtype_int(dtype_o)
    var ctx = _raw_ctx(ctx_o)
    var num_rows = _raw_int(num_rows_o)
    var handled = False
    comptime for dt in BAG_DTYPES:
        comptime if _dtype_arg_on[0, dt]():
            if dtype == dt:
                handled = True
                comptime if not has_accelerator():
                    raise Error("no GPU accelerator available at compile time")
                elif dt == DType.float64 and has_apple_gpu_accelerator():
                    raise Error("float64 is not supported on Apple GPU")
                else:
                    comptime acc_t = _acc_dtype[dt]()
                    _enqueue_cached[_embedding_renorm_kernel[dt]](
                        ctx,
                        _gs_blocks(num_rows),
                        1,
                        1,
                        GS_THREADS,
                        _make_ptr[dt](
                            _raw_int(weights_o)
                        ).as_unsafe_any_origin(),
                        _make_ptr[DType.int64](
                            _raw_int(rows_o)
                        ).as_unsafe_any_origin(),
                        Int64(num_rows),
                        Int64(_raw_int(dim_o)),
                        Int64(_raw_int(stride0_o)),
                        Int64(_raw_int(stride1_o)),
                        _raw_f64(max_norm_o).cast[acc_t](),
                        _raw_f64(norm_type_o).cast[acc_t](),
                    )
    if not handled:
        raise Error("EmbeddingRenorm: unsupported dtype ", dtype)


@export
def tmb_call(argv: Argv, argc: Int, err: ErrBuf, errcap: Int) abi("C") -> Int32:
    """C entry of this family: one kernel per build (see `OP`).
    Slots are described in op_utils (`Arg`); errors come back as (rc=1, message).
    """
    try:
        comptime if _op_on["EmbeddingBagForward"]():
            _spec_dispatcher12[_embedding_bag_go, "EmbeddingBagForward"](
                argv, argc
            )
            return 0
        comptime if _op_on["EmbeddingRenorm"]():
            _spec_dispatcher10[_embedding_renorm_go, "EmbeddingRenorm"](
                argv, argc
            )
            return 0
        raise Error(NO_OP_COMPILED)
    except e:
        return _tmb_entry_error(err, errcap, e)
