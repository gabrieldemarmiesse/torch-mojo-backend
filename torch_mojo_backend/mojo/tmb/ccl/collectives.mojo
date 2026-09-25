# Rewrite of: https://github.com/NVIDIA/nccl/blob/master/src/collectives.cc

from tmb.ccl.enqueue import (
    _allgather_locked,
    _allreduce_locked,
    _broadcast_locked,
    _order_after,
    _reduce_scatter_locked,
)
from tmb.ccl.include.comm import (
    _comm_ptr,
    _fail_submission,
    _lock,
    _submission_exception_code,
    _unlock,
)
from tmb.ccl.nccl import (
    NCCL_AVG,
    NCCL_INT32,
    NCCL_INT64,
    NCCL_INTERNAL_ERROR,
    NCCL_INVALID_ARGUMENT,
    NCCL_INVALID_USAGE,
    NCCL_SUCCESS,
    NCCL_SUM,
    _any_dtype_item_bytes,
    _dtype_item_bytes,
)


def ncclAllReduce(
    sendbuff: Int64,
    recvbuff: Int64,
    count: Int64,
    datatype: Int32,
    op: Int32,
    comm: Int64,
    stream: Int64,
) -> Int32:
    try:
        var item = _dtype_item_bytes(datatype)
        if item == 0:
            return NCCL_INVALID_ARGUMENT
        if op != NCCL_SUM and op != NCCL_AVG:
            return NCCL_INVALID_USAGE
        # The payload loops use 16-byte vector loads/stores on
        # in_ptr/out_ptr and fault on a misaligned address (RESULTS.md
        # section 9). Every allocator-returned pointer and every chunk
        # offset this function forms satisfy that (the chunk size is a
        # multiple of 4096 bytes) -- only a mid-tensor view the caller
        # passes directly can violate it, so reject that case here with a
        # clear error instead of letting the kernel raise.
        if Int(sendbuff) % 16 != 0 or Int(recvbuff) % 16 != 0:
            return NCCL_INVALID_ARGUMENT
        ref state = _comm_ptr(comm)[]
        _lock(state)
        var rc = NCCL_INTERNAL_ERROR
        try:
            rc = _allreduce_locked(
                comm, sendbuff, recvbuff, count, datatype, op, stream
            )
        except e:
            rc = _submission_exception_code(state)
            _unlock(state)
            print("mojoccl: ncclAllReduce failed:", e)
            return rc
        if rc == NCCL_SUCCESS:
            try:
                _order_after(state, stream)
            except e:
                _fail_submission(state)
                _unlock(state)
                raise e
        _unlock(state)
        return rc
    except e:
        print("mojoccl: ncclAllReduce failed:", e)
        return NCCL_INTERNAL_ERROR


def ncclBroadcast(
    sendbuff: Int64,
    recvbuff: Int64,
    count: Int64,
    datatype: Int32,
    root: Int32,
    comm: Int64,
    stream: Int64,
) -> Int32:
    try:
        var item = _any_dtype_item_bytes(datatype)
        if item == 0:
            return NCCL_INVALID_ARGUMENT
        ref state = _comm_ptr(comm)[]
        _lock(state)
        var rc = NCCL_INTERNAL_ERROR
        try:
            rc = _broadcast_locked(
                comm, sendbuff, recvbuff, Int(count) * item, root, stream
            )
        except e:
            rc = _submission_exception_code(state)
            _unlock(state)
            print("mojoccl: ncclBroadcast failed:", e)
            return rc
        if rc == NCCL_SUCCESS:
            try:
                _order_after(state, stream)
            except e:
                _fail_submission(state)
                _unlock(state)
                raise e
        _unlock(state)
        return rc
    except e:
        print("mojoccl: ncclBroadcast failed:", e)
        return NCCL_INTERNAL_ERROR


def ncclAllGather(
    sendbuff: Int64,
    recvbuff: Int64,
    sendcount: Int64,
    datatype: Int32,
    comm: Int64,
    stream: Int64,
) -> Int32:
    try:
        var item = _any_dtype_item_bytes(datatype)
        if item == 0:
            return NCCL_INVALID_ARGUMENT
        ref state = _comm_ptr(comm)[]
        _lock(state)
        var rc = NCCL_INTERNAL_ERROR
        try:
            rc = _allgather_locked(
                comm, sendbuff, recvbuff, Int(sendcount) * item, stream
            )
        except e:
            rc = _submission_exception_code(state)
            _unlock(state)
            print("mojoccl: ncclAllGather failed:", e)
            return rc
        if rc == NCCL_SUCCESS:
            try:
                _order_after(state, stream)
            except e:
                _fail_submission(state)
                _unlock(state)
                raise e
        _unlock(state)
        return rc
    except e:
        print("mojoccl: ncclAllGather failed:", e)
        return NCCL_INTERNAL_ERROR


# ---------------------------------------------------------------------------
# Not implemented: Reduce and point-to-point operations (+
# barrier, which routes to gloo -- see process_group.py). Returning
# ncclInvalidUsage rather than silently mis-computing is the point.
# ---------------------------------------------------------------------------


def ncclReduce(
    sendbuff: Int64,
    recvbuff: Int64,
    count: Int64,
    datatype: Int32,
    op: Int32,
    root: Int32,
    comm: Int64,
    stream: Int64,
) -> Int32:
    return NCCL_INVALID_USAGE


def ncclReduceScatter(
    sendbuff: Int64,
    recvbuff: Int64,
    count: Int64,
    datatype: Int32,
    op: Int32,
    comm: Int64,
    stream: Int64,
) -> Int32:
    try:
        var item = _dtype_item_bytes(datatype)
        if item == 0 or count < 0:
            return NCCL_INVALID_ARGUMENT
        if op != NCCL_SUM and op != NCCL_AVG:
            return NCCL_INVALID_USAGE
        if op == NCCL_AVG and (
            datatype == NCCL_INT32 or datatype == NCCL_INT64
        ):
            return NCCL_INVALID_USAGE
        ref state = _comm_ptr(comm)[]
        _lock(state)
        var rc = NCCL_INTERNAL_ERROR
        try:
            rc = _reduce_scatter_locked(
                comm, sendbuff, recvbuff, Int(count), datatype, op, stream
            )
        except e:
            rc = _submission_exception_code(state)
            _unlock(state)
            print("mojoccl: ncclReduceScatter failed:", e)
            return rc
        if rc == NCCL_SUCCESS:
            try:
                _order_after(state, stream)
            except e:
                _fail_submission(state)
                _unlock(state)
                raise e
        _unlock(state)
        return rc
    except e:
        print("mojoccl: ncclReduceScatter failed:", e)
        return NCCL_INTERNAL_ERROR


def ncclSend(
    sendbuff: Int64,
    count: Int64,
    datatype: Int32,
    peer: Int32,
    comm: Int64,
    stream: Int64,
) -> Int32:
    return NCCL_INVALID_USAGE


def ncclRecv(
    recvbuff: Int64,
    count: Int64,
    datatype: Int32,
    peer: Int32,
    comm: Int64,
    stream: Int64,
) -> Int32:
    return NCCL_INVALID_USAGE
