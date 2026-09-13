# ===----------------------------------------------------------------------=== #
# Fast eager-mode GELU backward kernels for mojo_device (float32 and bfloat16,
# exact erf and tanh approximation modes), ported from the validated Fable
# candidate.
#
# Same architecture as loss_ops.mojo: the Python-visible function gets raw
# integer pointers (tensor `._ptr`, offset pre-applied) plus element-count and
# mode ints and the device's DeviceContext pointer, and enqueues work on the
# device queue (fire and forget, no sync, no host reads, no host allocation).
#
# The op is memory-bound: two f32 reads plus one f32 write per element is the
# minimum global traffic, and NCU confirms the vectorized kernel already moves
# exactly that. Each mode uses one kernel. When the host proves all three base
# pointers 16-byte aligned, the body runs one 16-byte vector load/store chunk
# per thread (flat launch; capped-grid grid-stride and higher ILP measured
# equal-or-slower at nanoGPT sizes). The same kernel then finishes the tail --
# or, for unaligned pointers, the whole range -- with a scalar grid-stride
# loop, so arbitrary positive sizes and unproven alignment stay correct. The
# bf16 kernels widen loads to f32, evaluate the derivative in f32, and round
# once on the store.
# ===----------------------------------------------------------------------=== #

from std.gpu import block_dim, block_idx, grid_dim, thread_idx
from max.gpu.host import DeviceContext
from std.math import ceildiv, erf, exp, tanh
from std.os import abort
from std.sys.info import has_accelerator, has_apple_gpu_accelerator

from op_utils import (
    Arg,
    Argv,
    _enqueue_cached,
    _make_ptr,
    _raw_ctx,
    _raw_int,
)

from variant_gates import ErrBuf, NO_OP_COMPILED, _op_on, _tmb_entry_error


comptime _BLOCK = 256
comptime _VEC = 4
comptime _VEC_BF16 = 8

comptime _SQRT_HALF = Float32(0.7071067811865476)
comptime _INV_SQRT_2PI = Float32(0.3989422804014327)
comptime _BETA = Float32(0.7978845608028654)
comptime _KAPPA = Float32(0.044715)


@always_inline
def _exact_grad[
    width: Int
](x: SIMD[DType.float32, width], g: SIMD[DType.float32, width]) -> SIMD[
    DType.float32, width
]:
    var cdf = 0.5 * (1.0 + erf(x * _SQRT_HALF))
    var pdf = _INV_SQRT_2PI * exp(-0.5 * (x * x))
    return g * (cdf + x * pdf)


@always_inline
def _tanh_grad[
    width: Int
](x: SIMD[DType.float32, width], g: SIMD[DType.float32, width]) -> SIMD[
    DType.float32, width
]:
    var x2 = x * x
    var inner = _BETA * (x + _KAPPA * (x2 * x))
    var t = tanh(inner)
    var derivative = 0.5 * (1.0 + t) + 0.5 * x * (1.0 - t * t) * _BETA * (
        1.0 + 3.0 * _KAPPA * x2
    )
    return g * derivative


@__name("gelu_backward_exact")
def _gelu_backward_exact(
    output: Pointer[Scalar[DType.float32], MutAnyOrigin],
    grad_output: Pointer[Scalar[DType.float32], MutAnyOrigin],
    input: Pointer[Scalar[DType.float32], MutAnyOrigin],
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
        var x = input.unsafe_load[width=_VEC, alignment=16](base)
        var g = grad_output.unsafe_load[width=_VEC, alignment=16](base)
        output.unsafe_store[width=_VEC, alignment=16](
            base, _exact_grad[_VEC](x, g)
        )
    var i = vec_count * _VEC + gid
    var stride = Int(grid_dim.x) * _BLOCK
    while i < elements:
        output[unsafe_offset=i] = _exact_grad[1](
            input[unsafe_offset=i], grad_output[unsafe_offset=i]
        )
        i += stride


@__name("gelu_backward_tanh")
def _gelu_backward_tanh(
    output: Pointer[Scalar[DType.float32], MutAnyOrigin],
    grad_output: Pointer[Scalar[DType.float32], MutAnyOrigin],
    input: Pointer[Scalar[DType.float32], MutAnyOrigin],
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
        var x = input.unsafe_load[width=_VEC, alignment=16](base)
        var g = grad_output.unsafe_load[width=_VEC, alignment=16](base)
        output.unsafe_store[width=_VEC, alignment=16](
            base, _tanh_grad[_VEC](x, g)
        )
    var i = vec_count * _VEC + gid
    var stride = Int(grid_dim.x) * _BLOCK
    while i < elements:
        output[unsafe_offset=i] = _tanh_grad[1](
            input[unsafe_offset=i], grad_output[unsafe_offset=i]
        )
        i += stride


@always_inline
def _grad_val[
    width: Int, tanh_mode: Bool
](x: SIMD[DType.float32, width], g: SIMD[DType.float32, width]) -> SIMD[
    DType.float32, width
]:
    comptime if tanh_mode:
        return _tanh_grad[width](x, g)
    else:
        return _exact_grad[width](x, g)


def _gelu_backward_f32_g4[
    tanh_mode: Bool
](
    output: Pointer[Scalar[DType.float32], MutAnyOrigin],
    grad_output: Pointer[Scalar[DType.float32], MutAnyOrigin],
    input: Pointer[Scalar[DType.float32], MutAnyOrigin],
    elements_arg: Int64,
    vec_count_arg: Int64,
):
    # Int is not device-passable (host/device width mismatch); scalars cross
    # the launch ABI as Int64 and index math stays in Int.
    var elements = Int(elements_arg)
    var vec_count = Int(vec_count_arg)
    # Apple variant: each thread owns 4 consecutive 16-byte chunks (one
    # sequential 64-byte stream per operand with the loads issued together),
    # which this GPU needs to stream at full rate.  The trailing chunk and
    # the scalar tail (or, unaligned, the whole range) ride the same launch.
    var gid = Int(block_idx.x) * Int(block_dim.x) + Int(thread_idx.x)
    var gstride = Int(grid_dim.x) * Int(block_dim.x)
    var groups = vec_count // 4
    var g = gid
    while g < groups:
        var b = g * 4
        var x0 = input.unsafe_load[width=_VEC, alignment=16](b * 4)
        var x1 = input.unsafe_load[width=_VEC, alignment=16]((b + 1) * 4)
        var x2 = input.unsafe_load[width=_VEC, alignment=16]((b + 2) * 4)
        var x3 = input.unsafe_load[width=_VEC, alignment=16]((b + 3) * 4)
        var g0 = grad_output.unsafe_load[width=_VEC, alignment=16](b * 4)
        var g1 = grad_output.unsafe_load[width=_VEC, alignment=16]((b + 1) * 4)
        var g2 = grad_output.unsafe_load[width=_VEC, alignment=16]((b + 2) * 4)
        var g3 = grad_output.unsafe_load[width=_VEC, alignment=16]((b + 3) * 4)
        output.unsafe_store[width=_VEC, alignment=16](
            b * 4, _grad_val[_VEC, tanh_mode](x0, g0)
        )
        output.unsafe_store[width=_VEC, alignment=16](
            (b + 1) * 4, _grad_val[_VEC, tanh_mode](x1, g1)
        )
        output.unsafe_store[width=_VEC, alignment=16](
            (b + 2) * 4, _grad_val[_VEC, tanh_mode](x2, g2)
        )
        output.unsafe_store[width=_VEC, alignment=16](
            (b + 3) * 4, _grad_val[_VEC, tanh_mode](x3, g3)
        )
        g += gstride
    var c = groups * 4 + gid
    if c < vec_count:
        var x = input.unsafe_load[width=_VEC, alignment=16](c * 4)
        var gg = grad_output.unsafe_load[width=_VEC, alignment=16](c * 4)
        output.unsafe_store[width=_VEC, alignment=16](
            c * 4, _grad_val[_VEC, tanh_mode](x, gg)
        )
    var i = vec_count * _VEC + gid
    while i < elements:
        output[unsafe_offset=i] = _grad_val[1, tanh_mode](
            input[unsafe_offset=i], grad_output[unsafe_offset=i]
        )
        i += gstride


@__name("gelu_backward_exact_bf16")
def _gelu_backward_exact_bf16(
    output: Pointer[Scalar[DType.bfloat16], MutAnyOrigin],
    grad_output: Pointer[Scalar[DType.bfloat16], MutAnyOrigin],
    input: Pointer[Scalar[DType.bfloat16], MutAnyOrigin],
    elements_arg: Int64,
    vec_count_arg: Int64,
):
    # Int is not device-passable (host/device width mismatch); scalars cross
    # the launch ABI as Int64 and index math stays in Int.
    var elements = Int(elements_arg)
    var vec_count = Int(vec_count_arg)
    var gid = Int(block_idx.x) * _BLOCK + Int(thread_idx.x)
    if gid < vec_count:
        var base = gid * _VEC_BF16
        var x = input.unsafe_load[width=_VEC_BF16, alignment=16](base).cast[
            DType.float32
        ]()
        var g = grad_output.unsafe_load[width=_VEC_BF16, alignment=16](
            base
        ).cast[DType.float32]()
        output.unsafe_store[width=_VEC_BF16, alignment=16](
            base, _exact_grad[_VEC_BF16](x, g).cast[DType.bfloat16]()
        )
    var i = vec_count * _VEC_BF16 + gid
    var stride = Int(grid_dim.x) * _BLOCK
    while i < elements:
        output[unsafe_offset=i] = _exact_grad[1](
            input[unsafe_offset=i].cast[DType.float32](),
            grad_output[unsafe_offset=i].cast[DType.float32](),
        ).cast[DType.bfloat16]()
        i += stride


@__name("gelu_backward_tanh_bf16")
def _gelu_backward_tanh_bf16(
    output: Pointer[Scalar[DType.bfloat16], MutAnyOrigin],
    grad_output: Pointer[Scalar[DType.bfloat16], MutAnyOrigin],
    input: Pointer[Scalar[DType.bfloat16], MutAnyOrigin],
    elements_arg: Int64,
    vec_count_arg: Int64,
):
    # Int is not device-passable (host/device width mismatch); scalars cross
    # the launch ABI as Int64 and index math stays in Int.
    var elements = Int(elements_arg)
    var vec_count = Int(vec_count_arg)
    var gid = Int(block_idx.x) * _BLOCK + Int(thread_idx.x)
    if gid < vec_count:
        var base = gid * _VEC_BF16
        var x = input.unsafe_load[width=_VEC_BF16, alignment=16](base).cast[
            DType.float32
        ]()
        var g = grad_output.unsafe_load[width=_VEC_BF16, alignment=16](
            base
        ).cast[DType.float32]()
        output.unsafe_store[width=_VEC_BF16, alignment=16](
            base, _tanh_grad[_VEC_BF16](x, g).cast[DType.bfloat16]()
        )
    var i = vec_count * _VEC_BF16 + gid
    var stride = Int(grid_dim.x) * _BLOCK
    while i < elements:
        output[unsafe_offset=i] = _tanh_grad[1](
            input[unsafe_offset=i].cast[DType.float32](),
            grad_output[unsafe_offset=i].cast[DType.float32](),
        ).cast[DType.bfloat16]()
        i += stride


def enqueue_gelu_backward_f32(
    output: Pointer[Scalar[DType.float32], MutAnyOrigin],
    grad_output: Pointer[Scalar[DType.float32], MutAnyOrigin],
    input: Pointer[Scalar[DType.float32], MutAnyOrigin],
    elements: Int,
    tanh_approx: Bool,
    ctx: DeviceContext,
) raises:
    comptime if not has_accelerator():
        raise Error("no GPU accelerator available at compile time")
    else:
        if elements <= 0:
            return
        var aligned = (Int(output) | Int(grad_output) | Int(input)) % 16 == 0
        var vec_count = elements // _VEC if aligned else 0
        comptime if has_apple_gpu_accelerator():
            # Apple: grouped sequential-stream kernel (see
            # _gelu_backward_f32_g4).
            var span = max(vec_count // 4, 1) if vec_count > 0 else elements
            var grid_g4 = max(1, min(ceildiv(span, _BLOCK), 4096))
            if tanh_approx:
                _enqueue_cached[_gelu_backward_f32_g4[True]](
                    ctx,
                    "gelu_bwd_tanh_g4",
                    grid_g4,
                    1,
                    1,
                    _BLOCK,
                    output,
                    grad_output,
                    input,
                    Int64(elements),
                    Int64(vec_count),
                )
            else:
                _enqueue_cached[_gelu_backward_f32_g4[False]](
                    ctx,
                    "gelu_bwd_exact_g4",
                    grid_g4,
                    1,
                    1,
                    _BLOCK,
                    output,
                    grad_output,
                    input,
                    Int64(elements),
                    Int64(vec_count),
                )
            return
        # The grid covers the vector body; the <=3-element aligned tail rides
        # on the same launch via the scalar loop. With vec_count == 0 the grid
        # covers every element for the scalar path.
        var grid = ceildiv(vec_count, _BLOCK) if vec_count > 0 else ceildiv(
            elements, _BLOCK
        )
        if tanh_approx:
            _enqueue_cached[_gelu_backward_tanh](
                ctx,
                "gelu_bwd_tanh",
                grid,
                1,
                1,
                _BLOCK,
                output,
                grad_output,
                input,
                Int64(elements),
                Int64(vec_count),
            )
        else:
            _enqueue_cached[_gelu_backward_exact](
                ctx,
                "gelu_bwd_exact",
                grid,
                1,
                1,
                _BLOCK,
                output,
                grad_output,
                input,
                Int64(elements),
                Int64(vec_count),
            )


def enqueue_gelu_backward_bf16(
    output: Pointer[Scalar[DType.bfloat16], MutAnyOrigin],
    grad_output: Pointer[Scalar[DType.bfloat16], MutAnyOrigin],
    input: Pointer[Scalar[DType.bfloat16], MutAnyOrigin],
    elements: Int,
    tanh_approx: Bool,
    ctx: DeviceContext,
) raises:
    comptime if not has_accelerator():
        raise Error("no GPU accelerator available at compile time")
    else:
        if elements <= 0:
            return
        var aligned = (Int(output) | Int(grad_output) | Int(input)) % 16 == 0
        var vec_count = elements // _VEC_BF16 if aligned else 0
        # The grid covers the vector body; the <=7-element aligned tail rides
        # on the same launch via the scalar loop. With vec_count == 0 the grid
        # covers every element for the scalar path.
        var grid = ceildiv(vec_count, _BLOCK) if vec_count > 0 else ceildiv(
            elements, _BLOCK
        )
        if tanh_approx:
            _enqueue_cached[_gelu_backward_tanh_bf16](
                ctx,
                "gelu_bwd_tanh_bf16",
                grid,
                1,
                1,
                _BLOCK,
                output,
                grad_output,
                input,
                Int64(elements),
                Int64(vec_count),
            )
        else:
            _enqueue_cached[_gelu_backward_exact_bf16](
                ctx,
                "gelu_bwd_exact_bf16",
                grid,
                1,
                1,
                _BLOCK,
                output,
                grad_output,
                input,
                Int64(elements),
                Int64(vec_count),
            )


# ---------------------------------------------------------------------------
# Raw CPython entry points
# ---------------------------------------------------------------------------


def _gelu_backward_go(
    output_ptr_obj: Arg,
    grad_output_ptr_obj: Arg,
    input_ptr_obj: Arg,
    elements_obj: Arg,
    tanh_mode_obj: Arg,
    device_context_ptr: Arg,
) raises:
    var output = _make_ptr[DType.float32](
        _raw_int(output_ptr_obj)
    ).as_unsafe_any_origin()
    var grad_output = _make_ptr[DType.float32](
        _raw_int(grad_output_ptr_obj)
    ).as_unsafe_any_origin()
    var input = _make_ptr[DType.float32](
        _raw_int(input_ptr_obj)
    ).as_unsafe_any_origin()
    var ctx = _raw_ctx(device_context_ptr)
    enqueue_gelu_backward_f32(
        output,
        grad_output,
        input,
        _raw_int(elements_obj),
        _raw_int(tanh_mode_obj) != 0,
        ctx,
    )


def _gelu_backward_dispatcher(argv: Argv, argc: Int) raises:
    var args = argv
    _gelu_backward_go(
        args[unsafe_offset=0],
        args[unsafe_offset=1],
        args[unsafe_offset=2],
        args[unsafe_offset=3],
        args[unsafe_offset=4],
        args[unsafe_offset=5],
    )


def _gelu_backward_bf16_go(
    output_ptr_obj: Arg,
    grad_output_ptr_obj: Arg,
    input_ptr_obj: Arg,
    elements_obj: Arg,
    tanh_mode_obj: Arg,
    device_context_ptr: Arg,
) raises:
    var output = _make_ptr[DType.bfloat16](
        _raw_int(output_ptr_obj)
    ).as_unsafe_any_origin()
    var grad_output = _make_ptr[DType.bfloat16](
        _raw_int(grad_output_ptr_obj)
    ).as_unsafe_any_origin()
    var input = _make_ptr[DType.bfloat16](
        _raw_int(input_ptr_obj)
    ).as_unsafe_any_origin()
    var ctx = _raw_ctx(device_context_ptr)
    enqueue_gelu_backward_bf16(
        output,
        grad_output,
        input,
        _raw_int(elements_obj),
        _raw_int(tanh_mode_obj) != 0,
        ctx,
    )


def _gelu_backward_bf16_dispatcher(argv: Argv, argc: Int) raises:
    var args = argv
    _gelu_backward_bf16_go(
        args[unsafe_offset=0],
        args[unsafe_offset=1],
        args[unsafe_offset=2],
        args[unsafe_offset=3],
        args[unsafe_offset=4],
        args[unsafe_offset=5],
    )


# ---------------------------------------------------------------------------
# Python module definition
# ---------------------------------------------------------------------------


@export
def tmb_call(argv: Argv, argc: Int, err: ErrBuf, errcap: Int) abi("C") -> Int32:
    """C entry of this family: one kernel per build (see `OP`).
    Slots are described in op_utils (`Arg`); errors come back as (rc=1, message).
    """
    try:
        comptime if _op_on["GeluBackwardF32"]():
            _gelu_backward_dispatcher(argv, argc)
            return 0
        comptime if _op_on["GeluBackwardBF16"]():
            _gelu_backward_bf16_dispatcher(argv, argc)
            return 0
        raise Error(NO_OP_COMPILED)
    except e:
        return _tmb_entry_error(err, errcap, e)
