"""Runtime-dynamic pure-Mojo BF16 GELU-forward candidate.

Buffers proven 16-byte aligned use one 32-byte BF16 vector per thread.  The
same kernels handle an aligned tail, or an entirely scalar unaligned range,
without any out-of-bounds access.  GELU widens BF16 lanes to FP32 internally
and rounds once on return.  Device functions are cached per supplied context.
"""

from nn.activations import gelu_tanh
from std.ffi import _get_global_or_null, external_call
from std.gpu import block_idx, grid_dim, thread_idx
from max.gpu.host import DeviceContext
from std.math import ceildiv, exp2
from std.math.polynomial import polynomial_evaluate
from std.memory import alloc, bitcast


comptime _BLOCK = 256
comptime _VEC = 16

# Largest `|x|` the exponent polynomial below is fitted over.  Past it the true
# GELU underflows BF16 on the negative side -- `gelu(-13.9)` is 4e-43, under the
# smallest BF16 subnormal -- so the fit only has to stay monotone out there, not
# accurate, and `_GELU_A` is placed where accuracy stops mattering rather than
# where the inputs stop.  It also keeps the prefactor finite, which is what
# makes `gelu(+inf) == +inf` and `gelu(-inf) == 0` fall out without a guard.
comptime _GELU_A = Scalar[DType.float32](12.75)
comptime _GELU_NEG_ZERO = bitcast[DType.float32, 1](UInt32(0x8000_0000))


@always_inline
def _gelu_exact[
    width: SIMDLength, //
](x: SIMD[DType.float32, width]) -> SIMD[DType.float32, width]:
    """The EXACT (erf) GELU, evaluated as one polynomial and one `v_exp_f32`.

    Not the `tanh` approximation: this computes the same mathematical function
    `x * Phi(x)` that `nn.activations.gelu` does, only far more cheaply.  The
    identity it is built on is

        gelu(x) = relu(x) - 0.5 * |x| * erfc(|x| / sqrt(2))

    -- for `x >= 0` the second term is `x - x*Phi(x)`, for `x < 0` it is
    `-x*Phi(x)` with `relu` contributing nothing -- so ONE branch of `erfc`
    serves both signs and the `select` between two polynomial branches
    disappears.  Writing that `erfc` as `exp2(-a * R(a))` and fitting `R`
    directly in the log2 domain removes, against `gelu(std.math.erf(x / sqrt2))`:

    * the `x / 1.4142135` divide.  `pop.div` carries no `arcp`, so LLVM may not
      fold a divide by a non-power-of-two into a multiply and gfx942 expands it
      to the IEEE `v_div_scale`/`v_rcp`/3x`v_fma`/`v_div_fmas`/`v_div_fixup`
      sequence -- about ten instructions per lane, sixteen lanes per thread.
      Here the `sqrt(2)` is folded into the fitted coefficients and costs zero.
    * the second polynomial.  `erf` evaluates BOTH its branches unconditionally
      (the branch point is |x| > 0.921875 and the selection is `v_cndmask`, not
      control flow), so its small-|x| branch costs six more FMAs on every lane
      that does not use it.  A wave-uniform skip cannot fire at 1024 elements
      per wave.
    * the `1 - exp(...)` and the `copysign`.

    What is left is 8 FMAs, one `v_exp_f32`, and five other VALU ops.

    It is also MORE accurate than the shipped path, not less, and in the place
    that matters: `0.5*x*(1 + erf(x/sqrt2))` cancels catastrophically once
    `erf -> -1`, because `1 + erf` is quantized by the FP32 epsilon at 1.0, so
    the shipped form collapses to zero below about `x = -5.2` while the true
    value is still 1e-7 and BF16 still resolves it. This form never forms
    `1 - (1 - eps)`; it computes the small quantity directly.  Measured over
    every one of the 33761 BF16 values in [-30, 30] against 50-digit `mpmath`:
    197 of them round to the wrong BF16 through the shipped path and 26 through
    this one, and all 26 are past `x = -13.7` where the answer is subnormal.
    """
    var a = abs(x)
    # The clamp feeds the polynomial only; `s2` keeps the unclamped `a` so the
    # exponent keeps growing past the fit and the tail keeps decaying to zero.
    var am = min(a, _GELU_A)
    # No exponent bias here, deliberately.  Moving binades out of `exp2` and
    # into the prefactor would cost nothing (the multiply becomes an FMA) and
    # would fix the last twelve BF16 values, near x = -13, where `exp2` bottoms
    # out at the smallest FP32 normal.  It was measured and REJECTED: scaling
    # the prefactor by 2^-24 underflows it for a SUBNORMAL input, where the
    # correction is the whole answer, and that broke 2304 of the 65280 BF16
    # inputs to fix 12 whose true value is under 1e-36.
    var s2 = a * polynomial_evaluate[
        [
            Scalar[DType.float32](1.151188374e00),
            4.584094882e-01,
            5.425748229e-02,
            -8.789988235e-03,
            1.049621147e-03,
            -8.759975753e-05,
            4.782452834e-06,
            -1.524164617e-07,
            2.140817079e-09,
        ],
    ](am)
    # `exp2` is one `llvm.amdgcn.exp2.f32` on gfx942; the log2(e) that `exp`
    # would multiply by is already inside the fitted coefficients.
    # `max(x, -0.0)` rather than `max(x, 0)`: for every `x` past the point
    # where the correction underflows, and for `-0.0` itself, the result is a
    # zero whose SIGN this is the only thing that carries.  `maxNum(-0.0, -0.0)`
    # is `-0.0` and `maxNum(+0.0, -0.0)` is `+0.0`, so both signs come out right
    # and no positive `x` is touched.
    return max(x, _GELU_NEG_ZERO) - (0.5 * am) * exp2(-s2)


@always_inline
def _gelu_exact_bf16[
    width: SIMDLength, //
](x: SIMD[DType.bfloat16, width]) -> SIMD[DType.bfloat16, width]:
    """Widen to FP32, evaluate the exact GELU, round once on return."""
    return _gelu_exact(x.cast[DType.float32]()).cast[DType.bfloat16]()


@__name("gelu_forward_bf16_exact_vec16")
def _gelu_forward_bf16_exact(
    output: UnsafePointer[Scalar[DType.bfloat16], MutAnyOrigin],
    input: UnsafePointer[Scalar[DType.bfloat16], MutAnyOrigin],
    elements_arg: Int64,
    vec_count_arg: Int64,
):
    # Int is not device-passable (host/device width mismatch); scalars cross
    # the launch ABI as Int64 and index math stays in Int.
    var elements = Int(elements_arg)
    var vec_count = Int(vec_count_arg)
    var gid = Int(block_idx.x) * _BLOCK + Int(thread_idx.x)
    if gid < vec_count:
        var base = gid * _VEC
        var values = input.load[width=_VEC, alignment=16](base)
        output.store[width=_VEC, alignment=16](base, _gelu_exact_bf16(values))
    var index = vec_count * _VEC + gid
    var stride = Int(grid_dim.x) * _BLOCK
    while index < elements:
        output[index] = _gelu_exact_bf16(input.load[width=1](index))[0]
        index += stride


@__name("gelu_forward_bf16_tanh_vec16")
def _gelu_forward_bf16_tanh(
    output: UnsafePointer[Scalar[DType.bfloat16], MutAnyOrigin],
    input: UnsafePointer[Scalar[DType.bfloat16], MutAnyOrigin],
    elements_arg: Int64,
    vec_count_arg: Int64,
):
    # Int is not device-passable (host/device width mismatch); scalars cross
    # the launch ABI as Int64 and index math stays in Int.
    var elements = Int(elements_arg)
    var vec_count = Int(vec_count_arg)
    var gid = Int(block_idx.x) * _BLOCK + Int(thread_idx.x)
    if gid < vec_count:
        var base = gid * _VEC
        var values = input.load[width=_VEC, alignment=16](base)
        output.store[width=_VEC, alignment=16](base, gelu_tanh(values))
    var index = vec_count * _VEC + gid
    var stride = Int(grid_dim.x) * _BLOCK
    while index < elements:
        output[index] = gelu_tanh(input[index])
        index += stride


def _enqueue_exact_cached(
    output: UnsafePointer[Scalar[DType.bfloat16], MutAnyOrigin],
    input: UnsafePointer[Scalar[DType.bfloat16], MutAnyOrigin],
    elements: Int,
    vec_count: Int,
    blocks: Int,
    ctx: DeviceContext,
) raises:
    var cache_name = String(t"GELU_FORWARD_BF16_EXACT_VEC16_V1_{ctx.id()}")
    comptime FuncT = type_of(ctx.compile_function[_gelu_forward_bf16_exact]())
    if global_ptr := _get_global_or_null(cache_name):
        var cached = global_ptr.value().bitcast[FuncT]()
        ctx.enqueue_function(
            cached[],
            output,
            input,
            Int64(elements),
            Int64(vec_count),
            grid_dim=(blocks,),
            block_dim=(_BLOCK,),
        )
        return
    var compiled = ctx.compile_function[_gelu_forward_bf16_exact]()
    var cached = alloc[FuncT](1)
    cached.init_pointee_move(compiled^)
    external_call["KGEN_CompilerRT_InsertGlobal", NoneType](
        StringSlice(cache_name), cached.bitcast[NoneType]()
    )
    ctx.enqueue_function(
        cached[],
        output,
        input,
        Int64(elements),
        Int64(vec_count),
        grid_dim=(blocks,),
        block_dim=(_BLOCK,),
    )


def _enqueue_tanh_cached(
    output: UnsafePointer[Scalar[DType.bfloat16], MutAnyOrigin],
    input: UnsafePointer[Scalar[DType.bfloat16], MutAnyOrigin],
    elements: Int,
    vec_count: Int,
    blocks: Int,
    ctx: DeviceContext,
) raises:
    var cache_name = String(t"GELU_FORWARD_BF16_TANH_VEC16_V1_{ctx.id()}")
    comptime FuncT = type_of(ctx.compile_function[_gelu_forward_bf16_tanh]())
    if global_ptr := _get_global_or_null(cache_name):
        var cached = global_ptr.value().bitcast[FuncT]()
        ctx.enqueue_function(
            cached[],
            output,
            input,
            Int64(elements),
            Int64(vec_count),
            grid_dim=(blocks,),
            block_dim=(_BLOCK,),
        )
        return
    var compiled = ctx.compile_function[_gelu_forward_bf16_tanh]()
    var cached = alloc[FuncT](1)
    cached.init_pointee_move(compiled^)
    external_call["KGEN_CompilerRT_InsertGlobal", NoneType](
        StringSlice(cache_name), cached.bitcast[NoneType]()
    )
    ctx.enqueue_function(
        cached[],
        output,
        input,
        Int64(elements),
        Int64(vec_count),
        grid_dim=(blocks,),
        block_dim=(_BLOCK,),
    )


def enqueue_gelu_forward_bf16(
    output: UnsafePointer[Scalar[DType.bfloat16], MutAnyOrigin],
    input: UnsafePointer[Scalar[DType.bfloat16], MutAnyOrigin],
    elements: Int,
    tanh_approx: Bool,
    ctx: DeviceContext,
) raises:
    if elements <= 0:
        return
    var aligned = (Int(output) | Int(input)) % 16 == 0
    var vec_count = elements // _VEC if aligned else 0
    var blocks = ceildiv(vec_count, _BLOCK) if vec_count > 0 else ceildiv(
        elements, _BLOCK
    )
    if tanh_approx:
        _enqueue_tanh_cached(output, input, elements, vec_count, blocks, ctx)
    else:
        _enqueue_exact_cached(output, input, elements, vec_count, blocks, ctx)
