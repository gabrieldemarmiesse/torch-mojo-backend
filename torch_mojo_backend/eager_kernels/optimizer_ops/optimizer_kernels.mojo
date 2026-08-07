"""Runtime-dynamic, multi-tensor FP32 AdamW GPU kernel.

This accepted kernel is independent of PyTorch and vendor libraries. A
descriptor batch is passed by value, each GPU block owns one fixed-size chunk,
and the block resolves that chunk to its runtime tensor descriptor. Tensor
sizes and addresses are runtime data and never compilation keys.
"""

from std.collections import InlineArray
from std.ffi import _get_global_or_null, external_call
from std.gpu import block_idx, thread_idx
from std.gpu.host import DeviceContext
from std.math import min, pow
from std.memory import alloc
from std.sys.info import has_apple_gpu_accelerator

from op_utils import _enqueue_cached, ieee_sqrt
from optimizer_contract import (
    ADAMW_CHUNK_ELEMENTS,
    ADAMW_DESC_CAP,
    ADAMW_THREADS,
    AdamWDesc,
)


comptime _VEC = 4


@always_inline
def _ptr(addr: Int) -> UnsafePointer[Scalar[DType.float32], MutUntrackedOrigin]:
    return UnsafePointer[Scalar[DType.float32], MutUntrackedOrigin](
        unsafe_from_address=addr
    )


@always_inline
def _adamw_update[
    width: Int
](
    param: SIMD[DType.float32, width],
    grad: SIMD[DType.float32, width],
    exp_avg: SIMD[DType.float32, width],
    exp_avg_sq: SIMD[DType.float32, width],
    max_exp_avg_sq: SIMD[DType.float32, width],
    lr: Float32,
    beta1: Float32,
    beta2: Float32,
    weight_decay: Float32,
    eps: Float32,
    bias1: Float32,
    sqrt_bias2: Float32,
    amsgrad: Bool,
) -> Tuple[
    SIMD[DType.float32, width],
    SIMD[DType.float32, width],
    SIMD[DType.float32, width],
    SIMD[DType.float32, width],
]:
    # The decay is mathematically neutral at zero, but evaluating the
    # expression unconditionally is not IEEE-neutral: 0 * +/-Inf is NaN.
    # Match ATen/CUDA by bypassing the multiply for weight_decay == 0.
    var updated_param = param
    if weight_decay != 0.0:
        updated_param -= lr * weight_decay * param
    var updated_avg = beta1 * exp_avg + (1.0 - beta1) * grad
    var updated_avg_sq = beta2 * exp_avg_sq + (1.0 - beta2) * grad * grad
    var denominator_v = updated_avg_sq
    var updated_max = max_exp_avg_sq
    if amsgrad:
        updated_max = max(updated_max, updated_avg_sq)
        denominator_v = updated_max
    var denominator = ieee_sqrt(denominator_v) / sqrt_bias2 + eps
    updated_param -= (lr / bias1) * updated_avg / denominator
    return updated_param, updated_avg, updated_avg_sq, updated_max


@__name("fused_adamw_f32_dynamic_multitensor")
def _fused_adamw_f32(
    descs: InlineArray[AdamWDesc, ADAMW_DESC_CAP],
    desc_count: Int,
    lr_scalar: Float32,
    lr_ptr_addr: Int,
    beta1: Float32,
    beta2: Float32,
    weight_decay: Float32,
    eps: Float32,
    amsgrad_int: Int,
    maximize_int: Int,
    grad_scale_ptr_addr: Int,
    found_inf_ptr_addr: Int,
):
    # found_inf must gate every mutation, including gradient writeback.
    if found_inf_ptr_addr != 0 and _ptr(found_inf_ptr_addr)[0] == 1.0:
        return

    var chunk = Int(block_idx.x)
    var desc_index = 0
    while desc_index + 1 < desc_count and chunk >= descs[desc_index].chunk_end:
        desc_index += 1

    var desc = descs[desc_index]
    var first_chunk = 0
    if desc_index != 0:
        first_chunk = descs[desc_index - 1].chunk_end
    var begin = (chunk - first_chunk) * ADAMW_CHUNK_ELEMENTS
    var end = min(begin + ADAMW_CHUNK_ELEMENTS, desc.numel)

    var params = _ptr(desc.param_addr)
    var grads = _ptr(desc.grad_addr)
    var exp_avgs = _ptr(desc.exp_avg_addr)
    var exp_avg_sqs = _ptr(desc.exp_avg_sq_addr)
    var max_exp_avg_sqs = _ptr(desc.max_exp_avg_sq_addr)
    var steps = _ptr(desc.step_addr)

    var lr = lr_scalar
    if lr_ptr_addr != 0:
        lr = _ptr(lr_ptr_addr)[0]
    var inv_grad_scale = Float32(1.0)
    if grad_scale_ptr_addr != 0:
        inv_grad_scale /= _ptr(grad_scale_ptr_addr)[0]
    var grad_sign = Float32(-1.0) if maximize_int != 0 else Float32(1.0)
    var step = steps[0]
    var bias1 = Float32(1.0) - pow(beta1, step)
    var sqrt_bias2 = ieee_sqrt(Float32(1.0) - pow(beta2, step))
    var amsgrad = amsgrad_int != 0

    var index = begin + Int(thread_idx.x) * _VEC
    var stride = ADAMW_THREADS * _VEC
    while index + _VEC <= end:
        var p = params.load[width=_VEC, alignment=4](index)
        var g = grads.load[width=_VEC, alignment=4](index)
        if grad_scale_ptr_addr != 0:
            g *= inv_grad_scale
            grads.store[width=_VEC, alignment=4](index, g)
        g *= grad_sign
        var m = exp_avgs.load[width=_VEC, alignment=4](index)
        var v = exp_avg_sqs.load[width=_VEC, alignment=4](index)
        var max_v = v
        if amsgrad:
            max_v = max_exp_avg_sqs.load[width=_VEC, alignment=4](index)
        var new_p, new_m, new_v, new_max = _adamw_update[_VEC](
            p,
            g,
            m,
            v,
            max_v,
            lr,
            beta1,
            beta2,
            weight_decay,
            eps,
            bias1,
            sqrt_bias2,
            amsgrad,
        )
        params.store[width=_VEC, alignment=4](index, new_p)
        exp_avgs.store[width=_VEC, alignment=4](index, new_m)
        exp_avg_sqs.store[width=_VEC, alignment=4](index, new_v)
        if amsgrad:
            max_exp_avg_sqs.store[width=_VEC, alignment=4](index, new_max)
        index += stride

    # Only the final chunk of a tensor can have a scalar tail. Starting it at
    # the first lane after the vector region keeps stores disjoint.
    index = begin + ((end - begin) // _VEC) * _VEC + Int(thread_idx.x)
    while index < end:
        var p = params[index]
        var g = grads[index]
        if grad_scale_ptr_addr != 0:
            g *= inv_grad_scale
            grads[index] = g
        g *= grad_sign
        var m = exp_avgs[index]
        var v = exp_avg_sqs[index]
        var max_v = v
        if amsgrad:
            max_v = max_exp_avg_sqs[index]
        var new_p, new_m, new_v, new_max = _adamw_update[1](
            p,
            g,
            m,
            v,
            max_v,
            lr,
            beta1,
            beta2,
            weight_decay,
            eps,
            bias1,
            sqrt_bias2,
            amsgrad,
        )
        params[index] = new_p[0]
        exp_avgs[index] = new_m[0]
        exp_avg_sqs[index] = new_v[0]
        if amsgrad:
            max_exp_avg_sqs[index] = new_max[0]
        index += ADAMW_THREADS


# Apple/Metal per-tensor variant. Metal only translates pointer-typed kernel
# *arguments* into valid GPU addresses/bindings; the descriptor batch above
# smuggles raw addresses as plain data, which read back zeros there. Each
# tensor gets its own launch with real pointer arguments; chunk grid and
# update math are unchanged. Optional pointers (lr / grad_scale / found_inf)
# are passed as a valid dummy plus a has_* flag, never as null.
@__name("fused_adamw_f32_tensor_apple_v1")
def _fused_adamw_f32_tensor_apple(
    params: UnsafePointer[Scalar[DType.float32], MutAnyOrigin],
    grads: UnsafePointer[Scalar[DType.float32], MutAnyOrigin],
    exp_avgs: UnsafePointer[Scalar[DType.float32], MutAnyOrigin],
    exp_avg_sqs: UnsafePointer[Scalar[DType.float32], MutAnyOrigin],
    max_exp_avg_sqs: UnsafePointer[Scalar[DType.float32], MutAnyOrigin],
    steps: UnsafePointer[Scalar[DType.float32], ImmutAnyOrigin],
    lr_ptr: UnsafePointer[Scalar[DType.float32], ImmutAnyOrigin],
    grad_scale_ptr: UnsafePointer[Scalar[DType.float32], ImmutAnyOrigin],
    found_inf_ptr: UnsafePointer[Scalar[DType.float32], ImmutAnyOrigin],
    numel: Int,
    lr_scalar: Float32,
    has_lr_ptr: Int,
    beta1: Float32,
    beta2: Float32,
    weight_decay: Float32,
    eps: Float32,
    amsgrad_int: Int,
    maximize_int: Int,
    has_grad_scale: Int,
    has_found_inf: Int,
):
    # found_inf must gate every mutation, including gradient writeback.
    if has_found_inf != 0 and found_inf_ptr[0] == 1.0:
        return

    var begin = Int(block_idx.x) * ADAMW_CHUNK_ELEMENTS
    var end = min(begin + ADAMW_CHUNK_ELEMENTS, numel)

    var lr = lr_scalar
    if has_lr_ptr != 0:
        lr = lr_ptr[0]
    var inv_grad_scale = Float32(1.0)
    if has_grad_scale != 0:
        inv_grad_scale /= grad_scale_ptr[0]
    var grad_sign = Float32(-1.0) if maximize_int != 0 else Float32(1.0)
    var step = steps[0]
    var bias1 = Float32(1.0) - pow(beta1, step)
    var sqrt_bias2 = ieee_sqrt(Float32(1.0) - pow(beta2, step))
    var amsgrad = amsgrad_int != 0

    var index = begin + Int(thread_idx.x) * _VEC
    var stride = ADAMW_THREADS * _VEC
    while index + _VEC <= end:
        var p = params.load[width=_VEC, alignment=4](index)
        var g = grads.load[width=_VEC, alignment=4](index)
        if has_grad_scale != 0:
            g *= inv_grad_scale
            grads.store[width=_VEC, alignment=4](index, g)
        g *= grad_sign
        var m = exp_avgs.load[width=_VEC, alignment=4](index)
        var v = exp_avg_sqs.load[width=_VEC, alignment=4](index)
        var max_v = v
        if amsgrad:
            max_v = max_exp_avg_sqs.load[width=_VEC, alignment=4](index)
        var new_p, new_m, new_v, new_max = _adamw_update[_VEC](
            p,
            g,
            m,
            v,
            max_v,
            lr,
            beta1,
            beta2,
            weight_decay,
            eps,
            bias1,
            sqrt_bias2,
            amsgrad,
        )
        params.store[width=_VEC, alignment=4](index, new_p)
        exp_avgs.store[width=_VEC, alignment=4](index, new_m)
        exp_avg_sqs.store[width=_VEC, alignment=4](index, new_v)
        if amsgrad:
            max_exp_avg_sqs.store[width=_VEC, alignment=4](index, new_max)
        index += stride

    # Only the final chunk of a tensor can have a scalar tail. Starting it at
    # the first lane after the vector region keeps stores disjoint.
    index = begin + ((end - begin) // _VEC) * _VEC + Int(thread_idx.x)
    while index < end:
        var p = params[index]
        var g = grads[index]
        if has_grad_scale != 0:
            g *= inv_grad_scale
            grads[index] = g
        g *= grad_sign
        var m = exp_avgs[index]
        var v = exp_avg_sqs[index]
        var max_v = v
        if amsgrad:
            max_v = max_exp_avg_sqs[index]
        var new_p, new_m, new_v, new_max = _adamw_update[1](
            p,
            g,
            m,
            v,
            max_v,
            lr,
            beta1,
            beta2,
            weight_decay,
            eps,
            bias1,
            sqrt_bias2,
            amsgrad,
        )
        params[index] = new_p[0]
        exp_avgs[index] = new_m[0]
        exp_avg_sqs[index] = new_v[0]
        if amsgrad:
            max_exp_avg_sqs[index] = new_max[0]
        index += ADAMW_THREADS


def enqueue_fused_adamw_f32(
    descs: InlineArray[AdamWDesc, ADAMW_DESC_CAP],
    desc_count: Int,
    total_chunks: Int,
    lr_scalar: Float32,
    lr_ptr_addr: Int,
    beta1: Float32,
    beta2: Float32,
    weight_decay: Float32,
    eps: Float32,
    amsgrad_int: Int,
    maximize_int: Int,
    grad_scale_ptr_addr: Int,
    found_inf_ptr_addr: Int,
    ctx: DeviceContext,
) raises:
    comptime if has_apple_gpu_accelerator():
        var first_chunk = 0
        for desc_index in range(desc_count):
            var desc = descs[desc_index]
            var chunk_count = desc.chunk_end - first_chunk
            first_chunk = desc.chunk_end
            if chunk_count <= 0:
                continue
            var max_addr = desc.max_exp_avg_sq_addr
            if max_addr == 0:
                max_addr = desc.param_addr  # unused when amsgrad == 0
            var lr_addr = lr_ptr_addr if lr_ptr_addr != 0 else desc.param_addr
            var gs_addr = (
                grad_scale_ptr_addr if grad_scale_ptr_addr
                != 0 else desc.param_addr
            )
            var fi_addr = (
                found_inf_ptr_addr if found_inf_ptr_addr
                != 0 else desc.param_addr
            )
            _enqueue_cached[_fused_adamw_f32_tensor_apple](
                ctx,
                String("FUSED_ADAMW_TENSOR_APPLE_F32_V1"),
                chunk_count,
                1,
                1,
                ADAMW_THREADS,
                _ptr(desc.param_addr).as_unsafe_any_origin(),
                _ptr(desc.grad_addr).as_unsafe_any_origin(),
                _ptr(desc.exp_avg_addr).as_unsafe_any_origin(),
                _ptr(desc.exp_avg_sq_addr).as_unsafe_any_origin(),
                _ptr(max_addr).as_unsafe_any_origin(),
                _ptr(desc.step_addr).as_unsafe_any_origin().as_immutable(),
                _ptr(lr_addr).as_unsafe_any_origin().as_immutable(),
                _ptr(gs_addr).as_unsafe_any_origin().as_immutable(),
                _ptr(fi_addr).as_unsafe_any_origin().as_immutable(),
                desc.numel,
                lr_scalar,
                1 if lr_ptr_addr != 0 else 0,
                beta1,
                beta2,
                weight_decay,
                eps,
                amsgrad_int,
                maximize_int,
                1 if grad_scale_ptr_addr != 0 else 0,
                1 if found_inf_ptr_addr != 0 else 0,
            )
        return

    # compile_function is cached explicitly per context and kernel ABI. No
    # runtime shape, descriptor count, or tensor address participates in this
    # key, so dynamic input sizes never trigger recompilation.
    var cache_name = String(t"FUSED_ADAMW_F32_V5_ABI1_{ctx.id()}")
    comptime FuncT = type_of(ctx.compile_function[_fused_adamw_f32]())
    if global_ptr := _get_global_or_null(cache_name):
        var cached = global_ptr.value().bitcast[FuncT]()
        ctx.enqueue_function(
            cached[],
            descs,
            desc_count,
            lr_scalar,
            lr_ptr_addr,
            beta1,
            beta2,
            weight_decay,
            eps,
            amsgrad_int,
            maximize_int,
            grad_scale_ptr_addr,
            found_inf_ptr_addr,
            grid_dim=(total_chunks,),
            block_dim=(ADAMW_THREADS,),
        )
        return

    var compiled = ctx.compile_function[_fused_adamw_f32]()
    var cached = alloc[FuncT](1)
    cached.init_pointee_move(compiled^)
    external_call["KGEN_CompilerRT_InsertGlobal", NoneType](
        StringSlice(cache_name),
        cached.bitcast[NoneType](),
    )
    ctx.enqueue_function(
        cached[],
        descs,
        desc_count,
        lr_scalar,
        lr_ptr_addr,
        beta1,
        beta2,
        weight_decay,
        eps,
        amsgrad_int,
        maximize_int,
        grad_scale_ptr_addr,
        found_inf_ptr_addr,
        grid_dim=(total_chunks,),
        block_dim=(ADAMW_THREADS,),
    )
