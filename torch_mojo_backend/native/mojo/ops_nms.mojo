"""Non-maximum suppression with device sort and IoU masks."""

from std.utils import IndexList
from abi import (
    ST_INT64,
    Values,
    cpu_empty,
    new_tensor,
    own,
    ret_owned,
    unsupported,
    v_f64,
    v_tensor,
)
from device import copy_from_host, copy_to_host, ctx_for, ctx_ptr
from kernels import KernelCall
from op_utils import MAX_RANK
from registry import Site, impl


def _shape(n: Int) -> IndexList[MAX_RANK]:
    var shape = IndexList[MAX_RANK](1)
    shape[MAX_RANK - 1] = n
    return shape


# torchvision::nms(Tensor dets, Tensor scores, float iou_threshold) -> Tensor
def op_nms(args: Values, n_args: Int, rets: Values, n_rets: Int) raises:
    var boxes = v_tensor(args[unsafe_offset=0])
    var scores = v_tensor(args[unsafe_offset=1])
    var threshold = v_f64(args[unsafe_offset=2])
    if (
        not boxes.on_mojo()
        or not scores.on_mojo()
        or boxes.device != scores.device
    ):
        unsupported("nms: boxes and scores must be on the same mojo GPU")
    if (
        boxes.rank != 2
        or boxes.dim(1) != 4
        or scores.rank != 1
        or scores.dim(0) != boxes.dim(0)
    ):
        unsupported("nms: expected boxes [N, 4] and scores [N]")
    if boxes.dtype != scores.dtype or (
        boxes.dtype != DType.float16
        and boxes.dtype != DType.float32
        and boxes.dtype != DType.float64
    ):
        unsupported(
            "nms: boxes and scores must have the same float16, float32, or"
            " float64 dtype"
        )
    if not boxes.contig or not scores.contig:
        unsupported("nms: boxes and scores must be contiguous")
    var ctx = ctx_for(boxes.device)
    if ctx.api() == "cpu" or (
        ctx.api() == "metal" and boxes.dtype == DType.float64
    ):
        unsupported("nms: requires a GPU supporting the input dtype")
    var n = boxes.dim(0)
    if n == 0:
        var empty = own(new_tensor(_shape(0), 1, ST_INT64, boxes.device))
        ret_owned(rets, 0, empty)
        return
    var columns = (n + 63) // 64
    var order = own(new_tensor(_shape(n), 1, ST_INT64, boxes.device))
    var temporary = own(new_tensor(_shape(n), 1, ST_INT64, boxes.device))
    var mask = own(new_tensor(_shape(n * columns), 1, ST_INT64, boxes.device))
    var call = KernelCall("nms_ops", "NmsMask")
    call.arg_dtype(0, boxes.dtype)
    call.int(boxes.ptr)
    call.int(scores.ptr)
    call.int(order.t.ptr)
    call.int(temporary.t.ptr)
    call.int(mask.t.ptr)
    call.int(n)
    call.f64(threshold)
    call.int(ctx_ptr(ctx))
    call.run()
    var host_mask = own(cpu_empty(_shape(n * columns), 1, ST_INT64))
    copy_to_host(ctx, mask.t.ptr, host_mask.t.ptr, n * columns * 8)
    var masks = Pointer[UInt64, MutUntrackedOrigin](
        unsafe_from_address=host_mask.t.ptr
    )
    var removed = List[UInt64](capacity=columns)
    for _ in range(columns):
        removed.append(0)
    var host_keep = own(cpu_empty(_shape(n), 1, ST_INT64))
    var kept = Pointer[Int64, MutUntrackedOrigin](
        unsafe_from_address=host_keep.t.ptr
    )
    var count = 0
    for i in range(n):
        var word = i // 64
        if removed[word] & (UInt64(1) << UInt64(i % 64)) != 0:
            continue
        kept[unsafe_offset=count] = Int64(i)
        count += 1
        for j in range(word, columns):
            removed[j] |= masks[unsafe_offset=i * columns + j]
    var keep = own(new_tensor(_shape(count), 1, ST_INT64, boxes.device))
    copy_from_host(boxes.device, ctx, keep.t.ptr, host_keep.t.ptr, count * 8)
    var out = own(new_tensor(_shape(count), 1, ST_INT64, boxes.device))
    var width = 1
    var odd = False
    while width < n:
        odd = not odd
        width *= 2
    var gather = KernelCall("nms_ops", "NmsGather")
    gather.int(temporary.t.ptr if odd else order.t.ptr)
    gather.int(keep.t.ptr)
    gather.int(out.t.ptr)
    gather.int(count)
    gather.int(ctx_ptr(ctx))
    gather.run()
    _ = ctx
    _ = order^
    _ = temporary^
    _ = mask^
    _ = host_mask^
    _ = host_keep^
    _ = keep^
    ret_owned(rets, 0, out)


def register_nms(site: Site) raises:
    impl[op_nms, "torchvision::nms"](site)
