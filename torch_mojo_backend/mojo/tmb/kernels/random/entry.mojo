# ===----------------------------------------------------------------------=== #
# C entry of the on-device random generators (kernels: distribution_kernels.mojo,
# multinomial_kernels.mojo, curand_philox.mojo). Slots are unpacked here and nothing is read from the
# host or synchronized: the generator state was reserved before the call.
# ===----------------------------------------------------------------------=== #

from tmb.kernels.random.distribution_kernels import (
    DIST_BERNOULLI,
    DIST_CAUCHY,
    DIST_EXPONENTIAL,
    DIST_GEOMETRIC,
    DIST_LOG_NORMAL,
    DIST_NORMAL,
    DIST_RANDOM_32,
    DIST_RANDOM_64,
    DIST_RANDOM_FROM_TO_32,
    DIST_RANDOM_FROM_TO_64,
    DIST_RANDOM_FULL_64,
    DIST_UNIFORM,
    I64x8,
    acc_dtype,
    enqueue_bernoulli_tensor,
    enqueue_distribution,
)
from tmb.kernels.random.sampler_kernels import (
    SAMPLE_BINOMIAL,
    SAMPLE_GAMMA,
    SAMPLE_POISSON,
    enqueue_sampler,
)
from tmb.kernels.random.multinomial_kernels import (
    enqueue_multinomial_check,
    enqueue_multinomial_draw,
)
from tmb.kernels.common.op_utils import (
    Arg,
    Argv,
    _raw_ctx,
    _raw_dtype_int,
    _raw_f64,
    _raw_int,
    _raw_tuple_int,
    _raw_tuple_len,
    _spec_dispatcher6,
    _spec_dispatcher10,
    _spec_dispatcher12,
    _spec_dispatcher15,
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

comptime ALL_DTYPES = [
    DType.float32,
    DType.bfloat16,
    DType.float16,
    DType.float64,
    DType.int64,
    DType.int32,
    DType.int16,
    DType.int8,
    DType.uint8,
    DType.uint16,
    DType.uint32,
    DType.uint64,
    DType.bool,
]
comptime PROB_DTYPES = [DType.float32, DType.float64]
# aten::multinomial's input dtypes (ATen's floating dispatch).
comptime MULTINOMIAL_DTYPES = [
    DType.float32,
    DType.bfloat16,
    DType.float16,
    DType.float64,
]


@always_inline
def _join_u64(lo: Int, hi: Int) -> UInt64:
    return UInt64(lo) | (UInt64(hi) << 32)


@always_inline
def _tuple_i64x8(t: Arg) -> I64x8:
    var out = I64x8(fill=Int64(0))
    var n = _raw_tuple_len(t)
    for i in range(min(n, 8)):
        out[i] = Int64(_raw_tuple_int(t, i))
    return out^


@always_inline
def _narrow[dtype: DType](x: Float64) -> Scalar[dtype]:
    """`static_cast<scalar_t>(double)`: at::Half / BFloat16 construct from
    float, so the 16-bit types round twice."""
    comptime if dtype == DType.float16 or dtype == DType.bfloat16:
        return x.cast[DType.float32]().cast[dtype]()
    else:
        return x.cast[dtype]()


def _launch[
    dtype: DType, DIST: Int
](
    ctx_obj: Arg,
    dst: Int,
    numel: Int,
    ndim: Int,
    sizes: I64x8,
    strides: I64x8,
    grid: Int,
    pf0: Float64,
    pf1: Float64,
    i0: Int64,
    i1: Int64,
    seed: UInt64,
    offset: UInt64,
) raises:
    comptime ACC = acc_dtype[dtype]()
    var p_out0 = Scalar[dtype]()
    var p_out1 = Scalar[dtype]()
    var p_acc0: Scalar[ACC]
    var p_acc1: Scalar[ACC]
    comptime if DIST == DIST_UNIFORM:
        p_out0 = _narrow[dtype](pf0)
        p_out1 = _narrow[dtype](pf1)
        p_acc0 = p_out0.cast[ACC]()
        # `to - from` is scalar_t arithmetic: at::Half / BFloat16 round the
        # difference before opmath sees it.
        p_acc1 = (p_out1.cast[ACC]() - p_acc0).cast[dtype]().cast[ACC]()
    else:
        p_acc0 = pf0.cast[ACC]()
        p_acc1 = pf1.cast[ACC]()
    enqueue_distribution[dtype, DIST](
        _raw_ctx(ctx_obj),
        dst,
        numel,
        ndim,
        sizes,
        strides,
        grid,
        p_out0,
        p_out1,
        p_acc0,
        p_acc1,
        i0,
        i1,
        seed,
        offset,
    )


@always_inline
def _dist_supported[DIST: Int, dt: DType]() -> Bool:
    """ATen's dispatch macro of each kernel."""
    comptime if (
        DIST == DIST_UNIFORM
        or DIST == DIST_NORMAL
        or DIST == DIST_LOG_NORMAL
        or DIST == DIST_CAUCHY
        or DIST == DIST_EXPONENTIAL
    ):
        return dt.is_floating_point()
    elif DIST == DIST_RANDOM_FULL_64:
        return (
            dt == DType.int64
            or dt == DType.float64
            or dt == DType.float32
            or dt == DType.bfloat16
        )
    elif DIST == DIST_RANDOM_32:
        return dt != DType.int64 and dt != DType.float64
    elif DIST == DIST_RANDOM_64:
        return dt == DType.int64 or dt == DType.float64
    elif DIST == DIST_BERNOULLI or DIST == DIST_GEOMETRIC:
        return not (
            dt == DType.uint16 or dt == DType.uint32 or dt == DType.uint64
        )
    else:  # random_from_to: AT_ALL_TYPES + bool/half/bf16 + barebones unsigned
        return True


def _dist_go[
    DIST: Int
](
    dst_obj: Arg,
    numel_obj: Arg,
    ndim_obj: Arg,
    sizes_obj: Arg,
    strides_obj: Arg,
    grid_obj: Arg,
    pf0_obj: Arg,
    pf1_obj: Arg,
    pi0_obj: Arg,
    pi1_obj: Arg,
    seed_lo_obj: Arg,
    seed_hi_obj: Arg,
    offset_lo_obj: Arg,
    offset_hi_obj: Arg,
    dtype_obj: Arg,
    ctx_obj: Arg,
) raises:
    var dtype = _raw_dtype_int(dtype_obj)
    var seed = _join_u64(_raw_int(seed_lo_obj), _raw_int(seed_hi_obj))
    var offset = _join_u64(_raw_int(offset_lo_obj), _raw_int(offset_hi_obj))
    var handled = False

    comptime for dt in ALL_DTYPES:
        comptime if _dtype_out_on[0, dt]() and _dist_supported[DIST, dt]():
            if dtype == dt:
                _launch[dt, DIST](
                    ctx_obj,
                    _raw_int(dst_obj),
                    _raw_int(numel_obj),
                    _raw_int(ndim_obj),
                    _tuple_i64x8(sizes_obj),
                    _tuple_i64x8(strides_obj),
                    _raw_int(grid_obj),
                    _raw_f64(pf0_obj),
                    _raw_f64(pf1_obj),
                    Int64(_raw_int(pi0_obj)),
                    Int64(_raw_int(pi1_obj)),
                    seed,
                    offset,
                )
                handled = True
    if not handled:
        raise Error("unsupported dtype for on-device random draw: ", dtype)


def _bernoulli_tensor_go(
    dst_obj: Arg,
    p_obj: Arg,
    numel_obj: Arg,
    ndim_obj: Arg,
    sizes_obj: Arg,
    dst_strides_obj: Arg,
    p_strides_obj: Arg,
    grid_obj: Arg,
    seed_lo_obj: Arg,
    seed_hi_obj: Arg,
    offset_lo_obj: Arg,
    offset_hi_obj: Arg,
    dtype_obj: Arg,
    pdtype_obj: Arg,
    ctx_obj: Arg,
) raises:
    var dtype = _raw_dtype_int(dtype_obj)
    var pdtype = _raw_dtype_int(pdtype_obj)
    var seed = _join_u64(_raw_int(seed_lo_obj), _raw_int(seed_hi_obj))
    var offset = _join_u64(_raw_int(offset_lo_obj), _raw_int(offset_hi_obj))
    var handled = False

    comptime for dt in ALL_DTYPES:
        comptime if _dtype_out_on[0, dt]():
            comptime for pdt in PROB_DTYPES:
                comptime if _dtype_arg_on[0, pdt]():
                    if dtype == dt and pdtype == pdt:
                        enqueue_bernoulli_tensor[dt, pdt](
                            _raw_ctx(ctx_obj),
                            _raw_int(dst_obj),
                            _raw_int(p_obj),
                            _raw_int(numel_obj),
                            _raw_int(ndim_obj),
                            _tuple_i64x8(sizes_obj),
                            _tuple_i64x8(dst_strides_obj),
                            _tuple_i64x8(p_strides_obj),
                            _raw_int(grid_obj),
                            seed,
                            offset,
                        )
                        handled = True
    if not handled:
        raise Error(
            "unsupported dtypes for on-device bernoulli_.Tensor: ",
            dtype,
            " / ",
            pdtype,
        )


def _multinomial_check_go(
    flag_obj: Arg,
    probs_obj: Arg,
    rows_obj: Arg,
    n_obj: Arg,
    dtype_obj: Arg,
    ctx_obj: Arg,
) raises:
    var dtype = _raw_dtype_int(dtype_obj)
    var handled = False
    comptime for dt in MULTINOMIAL_DTYPES:
        comptime if _dtype_arg_on[0, dt]():
            if dtype == dt:
                enqueue_multinomial_check[dt](
                    _raw_ctx(ctx_obj),
                    _raw_int(flag_obj),
                    _raw_int(probs_obj),
                    _raw_int(rows_obj),
                    _raw_int(n_obj),
                )
                handled = True
    if not handled:
        raise Error("unsupported dtype for multinomial: ", dtype)


def _multinomial_draw_go(
    dst_obj: Arg,
    cdf_obj: Arg,
    probs_obj: Arg,
    rows_obj: Arg,
    n_obj: Arg,
    n_sample_obj: Arg,
    seed_lo_obj: Arg,
    seed_hi_obj: Arg,
    offset_lo_obj: Arg,
    offset_hi_obj: Arg,
    dtype_obj: Arg,
    ctx_obj: Arg,
) raises:
    var dtype = _raw_dtype_int(dtype_obj)
    var handled = False
    comptime for dt in MULTINOMIAL_DTYPES:
        comptime if _dtype_arg_on[0, dt]():
            if dtype == dt:
                enqueue_multinomial_draw[dt](
                    _raw_ctx(ctx_obj),
                    _raw_int(dst_obj),
                    _raw_int(cdf_obj),
                    _raw_int(probs_obj),
                    _raw_int(rows_obj),
                    _raw_int(n_obj),
                    _raw_int(n_sample_obj),
                    _join_u64(_raw_int(seed_lo_obj), _raw_int(seed_hi_obj)),
                    _join_u64(_raw_int(offset_lo_obj), _raw_int(offset_hi_obj)),
                )
                handled = True
    if not handled:
        raise Error("unsupported dtype for multinomial: ", dtype)


# poisson / _standard_gamma / binomial: ATen's floating dispatch with the
# half types (AT_DISPATCH_FLOATING_TYPES_AND2(Half, BFloat16)).
comptime SAMPLER_DTYPES = [
    DType.float32,
    DType.bfloat16,
    DType.float16,
    DType.float64,
]


def _sampler_go[
    KIND: Int
](
    dst_obj: Arg,
    a_obj: Arg,
    b_obj: Arg,
    flag_obj: Arg,
    numel_obj: Arg,
    seed_lo_obj: Arg,
    seed_hi_obj: Arg,
    offset_lo_obj: Arg,
    offset_hi_obj: Arg,
    ctx_obj: Arg,
) raises:
    var handled = False
    comptime for dt in SAMPLER_DTYPES:
        comptime if _dtype_out_on[0, dt]():
            enqueue_sampler[dt, KIND](
                _raw_ctx(ctx_obj),
                _raw_int(dst_obj),
                _raw_int(a_obj),
                _raw_int(b_obj),
                _raw_int(flag_obj),
                _raw_int(numel_obj),
                _join_u64(_raw_int(seed_lo_obj), _raw_int(seed_hi_obj)),
                _join_u64(_raw_int(offset_lo_obj), _raw_int(offset_hi_obj)),
            )
            handled = True
    if not handled:
        raise Error("sampler: no dtype compiled into this module")


@export
def tmb_call(argv: Argv, argc: Int, err: ErrBuf, errcap: Int) abi("C") -> Int32:
    try:
        comptime if _op_on["Uniform"]():
            _spec_dispatcher16[_dist_go[DIST_UNIFORM], "Uniform"](argv, argc)
            return 0
        comptime if _op_on["Normal"]():
            _spec_dispatcher16[_dist_go[DIST_NORMAL], "Normal"](argv, argc)
            return 0
        comptime if _op_on["LogNormal"]():
            _spec_dispatcher16[_dist_go[DIST_LOG_NORMAL], "LogNormal"](
                argv, argc
            )
            return 0
        comptime if _op_on["Cauchy"]():
            _spec_dispatcher16[_dist_go[DIST_CAUCHY], "Cauchy"](argv, argc)
            return 0
        comptime if _op_on["Exponential"]():
            _spec_dispatcher16[_dist_go[DIST_EXPONENTIAL], "Exponential"](
                argv, argc
            )
            return 0
        comptime if _op_on["Geometric"]():
            _spec_dispatcher16[_dist_go[DIST_GEOMETRIC], "Geometric"](
                argv, argc
            )
            return 0
        comptime if _op_on["Bernoulli"]():
            _spec_dispatcher16[_dist_go[DIST_BERNOULLI], "Bernoulli"](
                argv, argc
            )
            return 0
        comptime if _op_on["RandomFromTo32"]():
            _spec_dispatcher16[
                _dist_go[DIST_RANDOM_FROM_TO_32], "RandomFromTo32"
            ](argv, argc)
            return 0
        comptime if _op_on["RandomFromTo64"]():
            _spec_dispatcher16[
                _dist_go[DIST_RANDOM_FROM_TO_64], "RandomFromTo64"
            ](argv, argc)
            return 0
        comptime if _op_on["RandomFull64"]():
            _spec_dispatcher16[_dist_go[DIST_RANDOM_FULL_64], "RandomFull64"](
                argv, argc
            )
            return 0
        comptime if _op_on["Random32"]():
            _spec_dispatcher16[_dist_go[DIST_RANDOM_32], "Random32"](argv, argc)
            return 0
        comptime if _op_on["Random64"]():
            _spec_dispatcher16[_dist_go[DIST_RANDOM_64], "Random64"](argv, argc)
            return 0
        comptime if _op_on["BernoulliTensor"]():
            _spec_dispatcher15[_bernoulli_tensor_go, "BernoulliTensor"](
                argv, argc
            )
            return 0
        comptime if _op_on["MultinomialCheck"]():
            _spec_dispatcher6[_multinomial_check_go, "MultinomialCheck"](
                argv, argc
            )
            return 0
        comptime if _op_on["MultinomialDraw"]():
            _spec_dispatcher12[_multinomial_draw_go, "MultinomialDraw"](
                argv, argc
            )
            return 0
        comptime if _op_on["Poisson"]():
            _spec_dispatcher10[_sampler_go[SAMPLE_POISSON], "Poisson"](
                argv, argc
            )
            return 0
        comptime if _op_on["Gamma"]():
            _spec_dispatcher10[_sampler_go[SAMPLE_GAMMA], "Gamma"](argv, argc)
            return 0
        comptime if _op_on["Binomial"]():
            _spec_dispatcher10[_sampler_go[SAMPLE_BINOMIAL], "Binomial"](
                argv, argc
            )
            return 0
        raise Error(NO_OP_COMPILED)
    except e:
        return _tmb_entry_error(err, errcap, e)
