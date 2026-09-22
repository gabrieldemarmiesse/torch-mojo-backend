from max.gpu.host import DeviceContext
from std.gpu import block_dim, block_idx, grid_dim, thread_idx
from std.math import ceildiv
from std.sys.info import _has_sm_9x
from tmb.kernels.common.op_utils import _enqueue_cached, _make_ptr


@__name(t"tanh_backward_contig_f32_v4_a{aligned}")
def tanh_backward_kernel[
    aligned: Bool
](
    dst: Pointer[Float32, MutAnyOrigin],
    grad: Pointer[Float32, ImmutAnyOrigin],
    output: Pointer[Float32, ImmutAnyOrigin],
    size_arg: Int64,
):
    comptime alignment = 16 if aligned else 4
    var size = Int(size_arg)
    var nvec = size // 4
    var tid = Int(block_idx.x) * Int(block_dim.x) + Int(thread_idx.x)
    var index = tid
    while index < nvec:
        var i = index * 4
        var g = grad.unsafe_load[width=4, alignment=alignment](i)
        var y = output.unsafe_load[width=4, alignment=alignment](i)
        # ATen CUDA contracts 1-y*y (BinaryMiscBackwardOpsKernels.cu).
        # Keep its single-rounded factor and separate final multiplication.
        var factor = (-y).fma(y, SIMD[DType.float32, 4](1))
        dst.unsafe_store[width=4, alignment=alignment](i, g * factor)
        index += Int(grid_dim.x) * Int(block_dim.x)
    if tid < size - nvec * 4:
        var i = nvec * 4 + tid
        var g = grad[unsafe_offset=i]
        var y = output[unsafe_offset=i]
        dst[unsafe_offset=i] = g * (-y).fma(y, Float32(1))


@__name("tanh_backward_contig_f32_v4_input_peel")
def tanh_backward_peel_kernel(
    dst: Pointer[Float32, MutAnyOrigin],
    grad: Pointer[Float32, ImmutAnyOrigin],
    output: Pointer[Float32, ImmutAnyOrigin],
    size_arg: Int64,
    head_arg: Int64,
):
    var size = Int(size_arg)
    var head = Int(head_arg)
    var nvec = (size - head) // 4
    var tid = Int(block_idx.x) * Int(block_dim.x) + Int(thread_idx.x)
    var index = tid
    while index < nvec:
        var i = head + index * 4
        var g = grad.unsafe_load[width=4, alignment=16](i)
        var y = output.unsafe_load[width=4, alignment=16](i)
        var factor = (-y).fma(y, SIMD[DType.float32, 4](1))
        dst.unsafe_store[width=4, alignment=4](i, g * factor)
        index += Int(grid_dim.x) * Int(block_dim.x)
    if tid < head:
        var y = output[unsafe_offset=tid]
        dst[unsafe_offset=tid] = grad[unsafe_offset=tid] * (-y).fma(
            y, Float32(1)
        )
    var tail = head + nvec * 4
    if tid < size - tail:
        var i = tail + tid
        var y = output[unsafe_offset=i]
        dst[unsafe_offset=i] = grad[unsafe_offset=i] * (-y).fma(y, Float32(1))


def enqueue_tanh_backward_f32(
    dst: Int, grad: Int, output: Int, size: Int, ctx: DeviceContext
) raises:
    if size == 0:
        return
    comptime if _has_sm_9x():
        # Grid and alignment routes measured on H100; all sizes are dynamic.
        var blocks = min(max(1, ceildiv(size // 4, 256)), 1 << 22)
        var dp = _make_ptr[DType.float32](dst).as_unsafe_any_origin()
        var gp = _make_ptr[DType.float32](grad).as_unsafe_any_origin().as_imm()
        var yp = (
            _make_ptr[DType.float32](output).as_unsafe_any_origin().as_imm()
        )
        if (dst | grad | output) % 16 == 0:
            _enqueue_cached[tanh_backward_kernel[True]](
                ctx,
                blocks,
                1,
                1,
                256,
                dp,
                gp,
                yp,
                Int64(size),
            )
        elif grad % 16 == output % 16:
            var head = min(size, ((16 - grad % 16) % 16) // 4)
            var peel_blocks = min(
                max(1, ceildiv((size - head) // 4, 256)), 1 << 22
            )
            _enqueue_cached[tanh_backward_peel_kernel](
                ctx,
                peel_blocks,
                1,
                1,
                256,
                dp,
                gp,
                yp,
                Int64(size),
                Int64(head),
            )
        else:
            _enqueue_cached[tanh_backward_kernel[False]](
                ctx,
                blocks,
                1,
                1,
                256,
                dp,
                gp,
                yp,
                Int64(size),
            )
    else:
        raise Error("tanh backward FP32 fused path requires Hopper")
