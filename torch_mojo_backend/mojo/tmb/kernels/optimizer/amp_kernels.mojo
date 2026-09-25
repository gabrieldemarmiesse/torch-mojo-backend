"""GradScaler's scale update (`aten::_amp_update_scale_`) as one thread.

ATen's `amp_update_scale_cuda_kernel` is a <<<1, 1>>> launch; this is the same
state machine on the same three device scalars, so `GradScaler.update()` never
reads anything back to the host. The pointers are real kernel arguments, as
Metal requires (see `foreach_elementwise_kernels`).
"""

from max.gpu.host import DeviceContext
from std.sys.info import has_accelerator, has_apple_gpu_accelerator
from std.utils.numerics import isfinite

from tmb.kernels.common.op_utils import _enqueue_cached


comptime _F32Ptr = Pointer[Float32, MutAnyOrigin]
comptime _I32Ptr = Pointer[Int32, MutAnyOrigin]

# ATen multiplies the float scale by the double factor, then narrows. Apple
# GPUs have no float64, so there the factor is narrowed first: the same result
# for the power-of-two factors GradScaler defaults to.
comptime _FACTOR = (
    DType.float32 if has_apple_gpu_accelerator() else DType.float64
)


@always_inline
def _times(scale: Float32, factor: Scalar[_FACTOR]) -> Float32:
    return (scale.cast[_FACTOR]() * factor).cast[DType.float32]()


@always_inline
def _ptr[dtype: DType](addr: Int) -> Pointer[Scalar[dtype], MutAnyOrigin]:
    return Pointer[Scalar[dtype], MutUntrackedOrigin](
        unsafe_from_address=addr
    ).as_unsafe_any_origin()


@__name("amp_update_scale_t1")
def _amp_update_scale_kernel(
    scale_ptr: _F32Ptr,
    growth_tracker_ptr: _I32Ptr,
    found_inf_ptr: _F32Ptr,
    growth_factor: Scalar[_FACTOR],
    backoff_factor: Scalar[_FACTOR],
    growth_interval: Int32,
):
    if found_inf_ptr[unsafe_offset=0] != 0.0:
        scale_ptr[unsafe_offset=0] = _times(
            scale_ptr[unsafe_offset=0], backoff_factor
        )
        growth_tracker_ptr[unsafe_offset=0] = 0
    else:
        var successful = growth_tracker_ptr[unsafe_offset=0] + 1
        if successful == growth_interval:
            var new_scale = _times(scale_ptr[unsafe_offset=0], growth_factor)
            if isfinite(new_scale):
                scale_ptr[unsafe_offset=0] = new_scale
            growth_tracker_ptr[unsafe_offset=0] = 0
        else:
            growth_tracker_ptr[unsafe_offset=0] = successful


def enqueue_amp_update_scale(
    scale_addr: Int,
    growth_tracker_addr: Int,
    found_inf_addr: Int,
    growth_factor: Float64,
    backoff_factor: Float64,
    growth_interval: Int,
    ctx: DeviceContext,
) raises:
    comptime if has_accelerator():
        _enqueue_cached[_amp_update_scale_kernel](
            ctx,
            1,
            1,
            1,
            1,
            _ptr[DType.float32](scale_addr),
            _ptr[DType.int32](growth_tracker_addr),
            _ptr[DType.float32](found_inf_addr),
            Scalar[_FACTOR](growth_factor),
            Scalar[_FACTOR](backoff_factor),
            Int32(growth_interval),
        )
    else:
        raise Error("no GPU accelerator available at compile time")
