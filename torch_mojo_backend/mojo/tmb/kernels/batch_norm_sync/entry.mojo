# C entry of SyncBatchNorm's building blocks (kernels: kernels.mojo).
# Slots, all pointers 0 when absent:
#   BnStats           mean, var_or_invstd, running_mean, running_var, input,
#                     params (C, N, HxW, mode, has_running, layout), eps,
#                     momentum,
#                     ctx                     DTYPE_ARG_0 input, 1 running
#   BnElemt           out, input, weight, bias, mean, invstd,
#                     params (C, HxW, numel, layout), ctx
#                                             DTYPE_ARG_0 input, 1 stats,
#                                             2 affine
#   BnGather          save_mean, save_invstd, mean, invstd, running_mean,
#                     running_var, counts, params (world, features), eps,
#                     momentum, ctx           DTYPE_ARG_0 stats, 1 scalar_t
#   BnBackwardReduce  sum_dy, sum_dy_xmu, grad_weight, grad_bias, input,
#                     grad_out, mean, invstd, params (flags, C, N, HxW), ctx
#                                             DTYPE_ARG_0 input, 1 stats,
#                                             2 weight
#   BnBackwardElemt   grad_input, grad_out, input, mean, invstd, weight,
#                     sum_dy, sum_dy_xmu, count (int32), params (world, C,
#                     HxW, numel, layout), ctx        DTYPE_ARG_0 input, 1 stats,
#                                             2 weight

from tmb.kernels.batch_norm_sync.kernels import (
    bn_backward_elemt,
    bn_backward_reduce,
    bn_elemt,
    bn_gather,
    bn_stats,
)
from tmb.kernels.common.op_utils import (
    Argv,
    _raw_ctx,
    _raw_f64,
    _raw_int,
    _raw_tuple_int,
)
from tmb.kernels.common.variant_gates import (
    ErrBuf,
    NO_OP_COMPILED,
    _dtype_arg_on,
    _op_on,
    _tmb_entry_error,
)

comptime BN_DTYPES = [
    DType.float32,
    DType.float16,
    DType.bfloat16,
    DType.float64,
]


@always_inline
def _a(argv: Argv, i: Int) -> Int:
    return _raw_int(argv[unsafe_offset=i])


def _go2[d0: DType, d1: DType](argv: Argv) raises:
    comptime if _op_on["BnStats"]():
        var p = argv[unsafe_offset=5]
        bn_stats[d0, d1](
            _a(argv, 0),
            _a(argv, 1),
            _a(argv, 2),
            _a(argv, 3),
            _a(argv, 4),
            _raw_tuple_int(p, 0),
            _raw_tuple_int(p, 1),
            _raw_tuple_int(p, 2),
            _raw_tuple_int(p, 3),
            _raw_tuple_int(p, 4) != 0,
            _raw_f64(argv[unsafe_offset=6]),
            _raw_f64(argv[unsafe_offset=7]),
            _raw_tuple_int(p, 5),
            _raw_ctx(argv[unsafe_offset=8]),
        )
    elif _op_on["BnGather"]():
        var p = argv[unsafe_offset=7]
        bn_gather[d0, d1](
            _a(argv, 0),
            _a(argv, 1),
            _a(argv, 2),
            _a(argv, 3),
            _a(argv, 4),
            _a(argv, 5),
            _a(argv, 6),
            _raw_tuple_int(p, 0),
            _raw_tuple_int(p, 1),
            _raw_f64(argv[unsafe_offset=8]),
            _raw_f64(argv[unsafe_offset=9]),
            _raw_ctx(argv[unsafe_offset=10]),
        )
    else:
        raise Error(NO_OP_COMPILED)


def _go3[d0: DType, d1: DType, d2: DType](argv: Argv) raises:
    comptime if _op_on["BnElemt"]():
        var p = argv[unsafe_offset=6]
        bn_elemt[d0, d1, d2](
            _a(argv, 0),
            _a(argv, 1),
            _a(argv, 2),
            _a(argv, 3),
            _a(argv, 4),
            _a(argv, 5),
            _raw_tuple_int(p, 0),
            _raw_tuple_int(p, 1),
            _raw_tuple_int(p, 2),
            _raw_tuple_int(p, 3),
            _raw_ctx(argv[unsafe_offset=7]),
        )
    elif _op_on["BnBackwardReduce"]():
        var p = argv[unsafe_offset=8]
        bn_backward_reduce[d0, d1, d2](
            _a(argv, 0),
            _a(argv, 1),
            _a(argv, 2),
            _a(argv, 3),
            _a(argv, 4),
            _a(argv, 5),
            _a(argv, 6),
            _a(argv, 7),
            _raw_tuple_int(p, 0),
            _raw_tuple_int(p, 1),
            _raw_tuple_int(p, 2),
            _raw_tuple_int(p, 3),
            _raw_ctx(argv[unsafe_offset=9]),
        )
    elif _op_on["BnBackwardElemt"]():
        var p = argv[unsafe_offset=9]
        bn_backward_elemt[d0, d1, d2](
            _a(argv, 0),
            _a(argv, 1),
            _a(argv, 2),
            _a(argv, 3),
            _a(argv, 4),
            _a(argv, 5),
            _a(argv, 6),
            _a(argv, 7),
            _a(argv, 8),
            _raw_tuple_int(p, 0),
            _raw_tuple_int(p, 1),
            _raw_tuple_int(p, 2),
            _raw_tuple_int(p, 3),
            _raw_tuple_int(p, 4),
            _raw_ctx(argv[unsafe_offset=10]),
        )
    else:
        raise Error(NO_OP_COMPILED)


@export
def tmb_call(argv: Argv, argc: Int, err: ErrBuf, errcap: Int) abi("C") -> Int32:
    try:
        comptime if _op_on["BnStats"]() or _op_on["BnGather"]():
            comptime for d0 in BN_DTYPES:
                comptime if _dtype_arg_on[0, d0]():
                    comptime for d1 in BN_DTYPES:
                        comptime if _dtype_arg_on[1, d1]():
                            _go2[d0, d1](argv)
                            return 0
        else:
            comptime for d0 in BN_DTYPES:
                comptime if _dtype_arg_on[0, d0]():
                    comptime for d1 in BN_DTYPES:
                        comptime if _dtype_arg_on[1, d1]():
                            comptime for d2 in BN_DTYPES:
                                comptime if _dtype_arg_on[2, d2]():
                                    _go3[d0, d1, d2](argv)
                                    return 0
        raise Error(NO_OP_COMPILED)
    except e:
        return _tmb_entry_error(err, errcap, e)
