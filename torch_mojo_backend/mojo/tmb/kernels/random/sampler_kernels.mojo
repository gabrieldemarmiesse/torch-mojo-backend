"""`poisson`, `_standard_gamma` and `binomial`: one rejection sampler per
element, on the device Philox stream.

The samplers are ATen's (aten/src/ATen/native/Distributions.h: Marsaglia and
Tsang's `sample_gamma` and the BTRS / inversion `sample_binomial` that the
CUDA kernels run, with their accumulate type and curand's float uniforms and
normals; Distributions.cpp's `sample_poisson`, Hoermann's PTRS, in double as
the CPU runs it -- CUDA calls curand's own `curand_poisson` there). The
stream is not CUDA's: element `i` draws from curand subsequence `i` at the
reserved offset (CUDA keys it by thread and redraws the same subsequence for
every element a thread visits), so values match stock CUDA in distribution,
not bit for bit. The generator advances by what CUDA reserves (poisson 20,
gamma 10, binomial 42; the caller's `philox_reserve`).

Every loop is bounded: a NaN parameter never enters one, and an exhausted
bound (probability far below 1e-30 for finite parameters) returns NaN rather
than spinning a kernel.
"""
from max.gpu.host import DeviceContext
from max.gpu import block_dim, block_idx, grid_dim, thread_idx
from std.math import ceil, floor
from std.sys.info import has_accelerator, has_apple_gpu_accelerator
from std.utils.numerics import inf, isinf, isnan, nan

from tmb.kernels.common.libdevice_port import (
    nv_exp,
    nv_expf,
    nv_log,
    nv_log1p,
    nv_log1pf,
    nv_logf,
)
from tmb.kernels.common.math_utils import ieee_sqrt
from tmb.kernels.common.op_utils import _enqueue_cached, _make_ptr
from tmb.kernels.common.pow_math import pow_c99
from tmb.kernels.random.philox import (
    CURAND_2POW32_INV,
    U32x2,
    U32x4,
    curand4,
    curand_box_muller,
    curand_ctr,
    curand_key,
)

comptime SAMPLE_POISSON = 0
comptime SAMPLE_GAMMA = 1
comptime SAMPLE_BINOMIAL = 2

comptime BLOCK = 256
# Rejection rounds before a sampler gives up (see the module docstring).
comptime MAX_ROUNDS = 4096


@always_inline
def acc_of[dtype: DType]() -> DType:
    """ATen's `acc_type<scalar_t, /*is_cuda=*/true>`."""
    comptime if dtype == DType.float64:
        return DType.float64
    else:
        return DType.float32


@always_inline
def poisson_acc() -> DType:
    """`sample_poisson` runs in double; Apple GPUs have none."""
    comptime if has_apple_gpu_accelerator():
        return DType.float32
    else:
        return DType.float64


struct _Stream(Movable):
    """curand's Philox4x32-10 state as a word stream: `curand4` blocks of
    subsequence `ctr[2:4]`, consumed one 32-bit word at a time, and
    `curand_normal`'s Box-Muller pair with its cached second value."""

    var ctr: U32x4
    var key: U32x2
    var k: UInt64
    var buf: U32x4
    var pos: Int
    var has_normal: Bool
    var normal_cache: Float32

    @always_inline
    def __init__(out self, seed: UInt64, offset: UInt64, subsequence: UInt64):
        self.ctr = curand_ctr(offset, subsequence)
        self.key = curand_key(seed)
        self.k = 0
        self.buf = U32x4(0)
        self.pos = 4
        self.has_normal = False
        self.normal_cache = 0

    @always_inline
    def word(mut self) -> UInt32:
        if self.pos == 4:
            self.buf = curand4(self.ctr, self.key, self.k)
            self.k += 1
            self.pos = 0
        var w = self.buf[self.pos]
        self.pos += 1
        return w

    @always_inline
    def uniform(mut self) -> Float32:
        """`curand_uniform`: (0, 1]."""
        return self.word().cast[DType.float32]() * CURAND_2POW32_INV + (
            CURAND_2POW32_INV / 2
        )

    @always_inline
    def uniform24(mut self) -> Float32:
        """Distributions.cu's `curand_uniform_wrapper`: [0, 1) from the low
        24 bits of one word."""
        return (self.word() & UInt32(0xFFFFFF)).cast[DType.float32]() * Float32(
            1.0 / 16777216.0
        )

    @always_inline
    def normal(mut self) -> Float32:
        """`curand_normal`: a Box-Muller pair per two words, the second
        value kept for the next call."""
        if self.has_normal:
            self.has_normal = False
            return self.normal_cache
        var x = self.word()
        var y = self.word()
        var pair = curand_box_muller(x, y)
        self.normal_cache = pair[1]
        self.has_normal = True
        return pair[0]


@always_inline
def _log[w: DType](x: Scalar[w]) -> Scalar[w]:
    comptime if w == DType.float64:
        return nv_log(x.cast[DType.float64]()).cast[w]()
    else:
        return nv_logf(x.cast[DType.float32]()).cast[w]()


@always_inline
def _log1p[w: DType](x: Scalar[w]) -> Scalar[w]:
    comptime if w == DType.float64:
        return nv_log1p(x.cast[DType.float64]()).cast[w]()
    else:
        return nv_log1pf(x.cast[DType.float32]()).cast[w]()


@always_inline
def _exp[w: DType](x: Scalar[w]) -> Scalar[w]:
    comptime if w == DType.float64:
        return nv_exp(x.cast[DType.float64]()).cast[w]()
    else:
        return nv_expf(x.cast[DType.float32]()).cast[w]()


@always_inline
def _loggam[w: DType](x: Scalar[w]) -> Scalar[w]:
    """log Gamma(x) for x >= 1 (numpy's `random_loggam`: Stirling's series
    after shifting x to at least 7), all `sample_poisson` asks of lgamma."""
    comptime A = [
        8.333333333333333e-02,
        -2.777777777777778e-03,
        7.936507936507937e-04,
        -5.952380952380952e-04,
        8.417508417508418e-04,
        -1.917526917526918e-03,
        6.410256410256410e-03,
        -2.955065359477124e-02,
        1.796443723688307e-01,
        -1.39243221690590e00,
    ]
    if x == 1 or x == 2:
        return 0
    var x0 = x
    var n = 0
    if x <= 7:
        n = Int(7 - x)
        x0 = x + Scalar[w](n)
    var x2 = 1 / (x0 * x0)
    comptime a9 = A[9]
    var gl0 = Scalar[w](a9)
    comptime for i in range(8, -1, -1):
        comptime ai = A[i]
        gl0 = gl0 * x2 + Scalar[w](ai)
    var gl = (
        gl0 / x0
        + Scalar[w](0.9189385332046727)  # log(2 pi) / 2
        + (x0 - Scalar[w](0.5)) * _log(x0)
        - x0
    )
    if x <= 7:
        for _ in range(n):
            x0 -= 1
            gl -= _log(x0)
    return gl


@always_inline
def _poisson(
    lam: Scalar[poisson_acc()], mut s: _Stream
) -> Scalar[poisson_acc()]:
    """Distributions.cpp's `sample_poisson`."""
    comptime w = poisson_acc()
    comptime V = Scalar[w]
    if isnan(lam) or lam < 0:
        return nan[w]()
    if isinf(lam):
        return lam
    if lam >= 10:
        # Transformed rejection method (Hoermann, 1993).
        var slam = ieee_sqrt(lam)
        var loglam = _log(lam)
        var b = V(0.931) + V(2.53) * slam
        var a = V(-0.059) + V(0.02483) * b
        var invalpha = V(1.1239) + V(1.1328) / (b - V(3.4))
        var vr = V(0.9277) - V(3.6224) / (b - 2)
        for _ in range(MAX_ROUNDS):
            var U = s.uniform().cast[w]() - V(0.5)
            var Vv = s.uniform().cast[w]()
            var us = V(0.5) - abs(U)
            var k = floor((2 * a / us + b) * U + lam + V(0.43))
            if us >= V(0.07) and Vv <= vr:
                return k
            if k < 0 or (us < V(0.013) and Vv > us):
                continue
            if (_log(Vv) + _log(invalpha) - _log(a / (us * us) + b)) <= (
                -lam + k * loglam - _loggam[w](k + 1)
            ):
                return k
        return nan[w]()
    if lam == 0:
        return 0
    var enlam = _exp(-lam)
    var x = V(0)
    var prod = V(1)
    for _ in range(MAX_ROUNDS):
        prod *= s.uniform().cast[w]()
        if prod > enlam:
            x += 1
        else:
            return x
    return nan[w]()


@always_inline
def _gamma[
    dtype: DType
](alpha_in: Scalar[dtype], mut s: _Stream) -> Scalar[dtype]:
    """Distributions.h's `sample_gamma` as gamma_cuda_kernel runs it, then
    the kernel's clamp to the dtype's smallest normal."""
    comptime w = acc_of[dtype]()
    comptime V = Scalar[w]
    var alpha = alpha_in.cast[w]()
    if isnan(alpha):
        return nan[dtype]()
    var scale = V(1)
    if alpha < 1:
        if alpha == 0:
            return Scalar[dtype](0)
        scale *= pow_c99[w](1 - s.uniform().cast[w](), 1 / alpha)
        alpha += 1
    var d = alpha - V(1.0 / 3.0)
    var c = 1 / ieee_sqrt(9 * d)
    if isnan(c) or not (d > 0):
        return nan[dtype]()
    var sample = nan[w]()
    for _ in range(MAX_ROUNDS):
        var x = V(0)
        var y = V(0)
        for _ in range(MAX_ROUNDS):
            x = s.normal().cast[w]()
            y = 1 + c * x
            if y > 0:
                break
        var v = y * y * y
        var u = 1 - s.uniform().cast[w]()
        var xx = x * x
        if u < 1 - V(0.0331) * xx * xx:
            sample = scale * d * v
            break
        if _log(u) < V(0.5) * xx + d * (1 - v + _log(v)):
            sample = scale * d * v
            break
    var r = sample.cast[dtype]()
    if isnan(r):
        return r
    var min_value: Scalar[dtype]
    comptime if dtype == DType.float16:
        min_value = Scalar[dtype](6.103515625e-05)
    elif dtype == DType.bfloat16:
        min_value = Scalar[dtype](1.1754943508222875e-38)
    elif dtype == DType.float32:
        min_value = Scalar[dtype](1.1754943508222875e-38)
    else:
        min_value = Scalar[dtype](2.2250738585072014e-308)
    return min_value if min_value > r else r


@always_inline
def _stirling_approx_tail[w: DType](k: Scalar[w]) -> Scalar[w]:
    comptime TAIL = [
        0.0810614667953272,
        0.0413406959554092,
        0.0276779256849983,
        0.02079067210376509,
        0.0166446911898211,
        0.0138761288230707,
        0.0118967099458917,
        0.0104112652619720,
        0.00925546218271273,
        0.00833056343336287,
    ]
    if k < 10:
        comptime t0 = TAIL[0]
        var r = Scalar[w](t0)
        comptime for i in range(1, 10):
            comptime ti = TAIL[i]
            if Int(k) == i:
                r = Scalar[w](ti)
        return r
    var kp1sq = (k + 1) * (k + 1)
    return (
        Scalar[w](1.0 / 12)
        - (Scalar[w](1.0 / 360) - Scalar[w](1.0 / 1260) / kp1sq) / kp1sq
    ) / (k + 1)


@always_inline
def _binomial_inversion[
    dtype: DType
](count: Scalar[dtype], prob: Scalar[dtype], mut s: _Stream) -> Scalar[dtype]:
    comptime w = acc_of[dtype]()
    var geom_sum = Scalar[w](0)
    var num_geom = Scalar[dtype](0)
    var logprob = _log1p(-prob.cast[w]())
    for _ in range(MAX_ROUNDS):
        var U = s.uniform24().cast[w]()
        var geom = ceil(_log(U) / logprob)
        geom_sum += geom
        if geom_sum > count.cast[w]():
            return num_geom
        num_geom = num_geom + 1
    return nan[dtype]()


@always_inline
def _btrs[
    dtype: DType
](count_in: Scalar[dtype], prob_in: Scalar[dtype], mut s: _Stream) -> Scalar[
    dtype
]:
    comptime w = acc_of[dtype]()
    comptime V = Scalar[w]
    var count = count_in.cast[w]()
    var prob = prob_in.cast[w]()
    var stddev = ieee_sqrt(count * prob * (1 - prob))
    var b = V(1.15) + V(2.53) * stddev
    var a = V(-0.0873) + V(0.0248) * b + V(0.01) * prob
    var c = count * prob + V(0.5)
    var v_r = V(0.92) - V(4.2) / b
    var r = prob / (1 - prob)
    var alpha = (V(2.83) + V(5.1) / b) * stddev
    var m = floor((count + 1) * prob)
    for _ in range(MAX_ROUNDS):
        var U = s.uniform24().cast[w]() - V(0.5)
        var Vv = s.uniform24().cast[w]()
        var us = V(0.5) - abs(U)
        var k = floor((2 * a / us + b) * U + c).cast[dtype]().cast[w]()
        if k < 0 or k > count:
            continue
        if us >= V(0.07) and Vv <= v_r:
            return k.cast[dtype]()
        Vv = _log(Vv * alpha / (a / (us * us) + b))
        var upperbound = (
            (m + V(0.5)) * _log((m + 1) / (r * (count - m + 1)))
            + (count + 1) * _log((count - m + 1) / (count - k + 1))
            + (k + V(0.5)) * _log(r * (count - k + 1) / (k + 1))
            + _stirling_approx_tail[w](m)
            + _stirling_approx_tail[w](count - m)
            - _stirling_approx_tail[w](k)
            - _stirling_approx_tail[w](count - k)
        )
        if Vv <= upperbound:
            return k.cast[dtype]()
    return nan[dtype]()


@always_inline
def _binomial[
    dtype: DType
](count: Scalar[dtype], prob: Scalar[dtype], mut s: _Stream) -> Scalar[dtype]:
    """Distributions.h's `sample_binomial`."""
    if isnan(count) or isnan(prob):
        return nan[dtype]()
    if count <= 0 or prob <= 0:
        return Scalar[dtype](0)
    if prob >= 1:
        return count
    if isinf(count):
        return count
    if prob <= Scalar[dtype](0.5):
        if count.cast[acc_of[dtype]()]() * prob.cast[acc_of[dtype]()]() >= 10:
            return _btrs(count, prob, s)
        return _binomial_inversion(count, prob, s)
    var qprob = Scalar[dtype](1.0) - prob
    if count.cast[acc_of[dtype]()]() * qprob.cast[acc_of[dtype]()]() >= 10:
        return count - _btrs(count, qprob, s)
    return count - _binomial_inversion(count, qprob, s)


@__name(t"philox_sampler{KIND}_{dtype}")
def _sampler_kernel[
    dtype: DType, KIND: Int
](
    dst: Pointer[Scalar[dtype], MutAnyOrigin],
    a: Pointer[Scalar[dtype], MutAnyOrigin],
    b: Pointer[Scalar[dtype], MutAnyOrigin],
    flag: Pointer[Int32, MutAnyOrigin],
    numel_arg: Int64,
    seed: UInt64,
    offset: UInt64,
):
    var numel = Int(numel_arg)
    var i = Int(block_idx.x) * Int(block_dim.x) + Int(thread_idx.x)
    var stride = Int(grid_dim.x) * Int(block_dim.x)
    while i < numel:
        var s = _Stream(seed, offset, UInt64(i))
        comptime if KIND == SAMPLE_POISSON:
            var lam = a[unsafe_offset=i]
            if not (lam >= 0):
                # CUDA's device assert, CPU's TORCH_CHECK: the op reads
                # this back and raises. Every writer stores the same 1.
                flag[unsafe_offset=0] = 1
            var lam_w = lam.cast[poisson_acc()]()
            dst[unsafe_offset=i] = _poisson(lam_w, s).cast[dtype]()
        elif KIND == SAMPLE_GAMMA:
            dst[unsafe_offset=i] = _gamma[dtype](a[unsafe_offset=i], s)
        else:
            dst[unsafe_offset=i] = _binomial[dtype](
                a[unsafe_offset=i], b[unsafe_offset=i], s
            )
        i += stride


def enqueue_sampler[
    dtype: DType, KIND: Int
](
    ctx: DeviceContext,
    dst_addr: Int,
    a_addr: Int,
    b_addr: Int,
    flag_addr: Int,
    numel: Int,
    seed: UInt64,
    offset: UInt64,
) raises:
    if numel <= 0:
        return
    comptime if dtype == DType.float64 and has_apple_gpu_accelerator():
        raise Error("float64 is not supported on Apple GPU")
    else:
        comptime if not has_accelerator():
            raise Error("no GPU accelerator available at compile time")
        else:
            var grid = min((numel + BLOCK - 1) // BLOCK, 65535)
            _enqueue_cached[_sampler_kernel[dtype, KIND]](
                ctx,
                grid,
                1,
                1,
                BLOCK,
                _make_ptr[dtype](dst_addr).as_unsafe_any_origin(),
                _make_ptr[dtype](a_addr).as_unsafe_any_origin(),
                _make_ptr[dtype](b_addr).as_unsafe_any_origin(),
                _make_ptr[DType.int32](flag_addr).as_unsafe_any_origin(),
                Int64(numel),
                seed,
                offset,
            )
