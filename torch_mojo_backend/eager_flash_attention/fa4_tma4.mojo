"""Rank-4 split-TMA creator with THREE runtime dims — stdlib gap-filler.

The stdlib's `create_split_tma` overloads stop at two runtime
dimensions; the hdim64 fwd kernel needs 4-D Q/O descriptors over
(batch, seqlen, nheads, head_dim) — S as its OWN dimension so TMA
OOB handling clamps partial tail tiles (BM=192 does not divide the
seqlen envelope; with the flattened (B*S, H, D) form a tail O-store
would CLOBBER the next batch's rows). Mirrors the stdlib's
`_split_tma_gmem_tensor` + `create_tensor_tile` pattern
(modular/max/kernels/src/layout/tma_async.mojo:4711-4871) 1:1, with
one more runtime dim.

The `_strided` variants additionally take runtime ELEMENT strides
(innermost stride 1) so zero-copy views of fused QKV storage can be
described without materializing dense copies; they return the same
`SplitLastDimTMATensorTile` types as the row-major creators, so kernel
specializations are shared.
"""

from max.gpu.host import DeviceContext
from max.gpu.host.nvidia.tma import TensorMapSwizzle
from std.utils.index import IndexList

from layout import (
    IntTuple,
    Layout,
    LayoutTensor,
    RuntimeLayout,
    RuntimeTuple,
    UNKNOWN_VALUE,
)
from layout.tma_async import SplitLastDimTMATensorTile, create_tensor_tile

# Fully runtime gmem shapes AND strides; only the smem/descriptor
# boxes stay compile time.
comptime _STRIDED_LAYOUT_3 = Layout(
    IntTuple(UNKNOWN_VALUE, UNKNOWN_VALUE, UNKNOWN_VALUE),
    IntTuple(UNKNOWN_VALUE, UNKNOWN_VALUE, UNKNOWN_VALUE),
)
comptime _STRIDED_LAYOUT_4 = Layout(
    IntTuple(UNKNOWN_VALUE, UNKNOWN_VALUE, UNKNOWN_VALUE, UNKNOWN_VALUE),
    IntTuple(UNKNOWN_VALUE, UNKNOWN_VALUE, UNKNOWN_VALUE, UNKNOWN_VALUE),
)


def create_split_tma_4d[
    dtype: DType,
    //,
    smem_shape: IndexList[4],
    gmem_shape: IndexList[4],
    swizzle_mode: TensorMapSwizzle,
](
    ctx: DeviceContext,
    ptr: UnsafePointer[Scalar[dtype], ImmutAnyOrigin],
    runtime_dim0: Int,
    runtime_dim1: Int,
    runtime_dim2: Int,
    out res: SplitLastDimTMATensorTile[
        dtype,
        smem_shape,
        swizzle_mode,
    ],
) raises:
    """Rank-4 TMA tile whose first THREE gmem dims are runtime."""
    var runtime_shape: IndexList[4] = {}
    runtime_shape[0] = runtime_dim0
    runtime_shape[1] = runtime_dim1
    runtime_shape[2] = runtime_dim2
    runtime_shape[3] = gmem_shape[3]
    var tensor = LayoutTensor[
        dtype,
        Layout.row_major(gmem_shape),
        ImmutAnyOrigin,
    ](ptr, RuntimeLayout[Layout.row_major(gmem_shape)].row_major(runtime_shape))
    res = create_tensor_tile[
        res.tile_shape,
        swizzle_mode=swizzle_mode,
        __tile_shape = res.tile_shape,
        __desc_shape = res.desc_shape,
    ](ctx, tensor)


def create_split_tma_3d_strided[
    dtype: DType,
    //,
    smem_shape: IndexList[3],
    swizzle_mode: TensorMapSwizzle,
](
    ctx: DeviceContext,
    ptr: UnsafePointer[Scalar[dtype], ImmutAnyOrigin],
    dim0: Int,
    dim1: Int,
    dim2: Int,
    stride0: Int,
    stride1: Int,
    stride2: Int,
    out res: SplitLastDimTMATensorTile[
        dtype,
        smem_shape,
        swizzle_mode,
    ],
) raises:
    """Rank-3 TMA tile with runtime gmem shape AND element strides."""
    var runtime_layout = RuntimeLayout[_STRIDED_LAYOUT_3](
        RuntimeTuple[_STRIDED_LAYOUT_3.shape](dim0, dim1, dim2),
        RuntimeTuple[_STRIDED_LAYOUT_3.stride](stride0, stride1, stride2),
    )
    var tensor = LayoutTensor[dtype, _STRIDED_LAYOUT_3, ImmutAnyOrigin](
        ptr, runtime_layout
    )
    res = create_tensor_tile[
        res.tile_shape,
        swizzle_mode=swizzle_mode,
        __tile_shape = res.tile_shape,
        __desc_shape = res.desc_shape,
    ](ctx, tensor)


def create_split_tma_4d_strided[
    dtype: DType,
    //,
    smem_shape: IndexList[4],
    swizzle_mode: TensorMapSwizzle,
](
    ctx: DeviceContext,
    ptr: UnsafePointer[Scalar[dtype], ImmutAnyOrigin],
    dim0: Int,
    dim1: Int,
    dim2: Int,
    dim3: Int,
    stride0: Int,
    stride1: Int,
    stride2: Int,
    stride3: Int,
    out res: SplitLastDimTMATensorTile[
        dtype,
        smem_shape,
        swizzle_mode,
    ],
) raises:
    """Rank-4 TMA tile with runtime gmem shape AND element strides."""
    var runtime_layout = RuntimeLayout[_STRIDED_LAYOUT_4](
        RuntimeTuple[_STRIDED_LAYOUT_4.shape](dim0, dim1, dim2, dim3),
        RuntimeTuple[_STRIDED_LAYOUT_4.stride](
            stride0, stride1, stride2, stride3
        ),
    )
    var tensor = LayoutTensor[dtype, _STRIDED_LAYOUT_4, ImmutAnyOrigin](
        ptr, runtime_layout
    )
    res = create_tensor_tile[
        res.tile_shape,
        swizzle_mode=swizzle_mode,
        __tile_shape = res.tile_shape,
        __desc_shape = res.desc_shape,
    ](ctx, tensor)
