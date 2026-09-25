# Rewrite of: https://github.com/NVIDIA/nccl/blob/master/src/nccl.h.in


comptime UID_BYTES = 128


# ncclResult_t (nccl.h.in:44-53)
comptime NCCL_SUCCESS: Int32 = 0
comptime NCCL_UNHANDLED_CUDA_ERROR: Int32 = 1
comptime NCCL_SYSTEM_ERROR: Int32 = 2
comptime NCCL_INTERNAL_ERROR: Int32 = 3
comptime NCCL_INVALID_ARGUMENT: Int32 = 4
comptime NCCL_INVALID_USAGE: Int32 = 5
comptime NCCL_REMOTE_ERROR: Int32 = 6
comptime NCCL_IN_PROGRESS: Int32 = 7

# ncclDataType_t (nccl.h.in:466-479). AllReduce runs a real kernel and only
# instantiates the five below; Broadcast/AllGather move raw bytes (item size
# x count -> nbytes) and accept every type in the enum.
comptime NCCL_INT8: Int32 = 0
comptime NCCL_UINT8: Int32 = 1
comptime NCCL_INT32: Int32 = 2
comptime NCCL_UINT32: Int32 = 3
comptime NCCL_INT64: Int32 = 4
comptime NCCL_UINT64: Int32 = 5
comptime NCCL_FLOAT16: Int32 = 6
comptime NCCL_FLOAT32: Int32 = 7
comptime NCCL_FLOAT64: Int32 = 8
comptime NCCL_BFLOAT16: Int32 = 9

# ncclRedOp_t (nccl.h.in:448-463) -- values this library implements.
comptime NCCL_SUM: Int32 = 0
comptime NCCL_AVG: Int32 = 4


def _dtype_item_bytes(nccl_dtype: Int32) -> Int:
    """Item size for the five dtypes AllReduce's kernel is instantiated for."""
    if nccl_dtype == NCCL_INT32 or nccl_dtype == NCCL_FLOAT32:
        return 4
    elif nccl_dtype == NCCL_INT64:
        return 8
    elif nccl_dtype == NCCL_FLOAT16 or nccl_dtype == NCCL_BFLOAT16:
        return 2
    else:
        return 0


def _any_dtype_item_bytes(nccl_dtype: Int32) -> Int:
    """Item size for every ncclDataType_t -- Broadcast/AllGather move raw
    bytes and never look at the dtype beyond this."""
    if nccl_dtype == NCCL_INT8 or nccl_dtype == NCCL_UINT8:
        return 1
    elif (
        nccl_dtype == NCCL_INT32
        or nccl_dtype == NCCL_UINT32
        or nccl_dtype == NCCL_FLOAT32
    ):
        return 4
    elif (
        nccl_dtype == NCCL_INT64
        or nccl_dtype == NCCL_UINT64
        or nccl_dtype == NCCL_FLOAT64
    ):
        return 8
    elif nccl_dtype == NCCL_FLOAT16 or nccl_dtype == NCCL_BFLOAT16:
        return 2
    else:
        return 0
