# Rewrite of: https://github.com/NVIDIA/nccl/blob/master/src/include/ibvwrap.h

from tmb.ccl.include.ibvcore import (
    CTX_POLL_CQ,
    CTX_POST_RECV,
    CTX_POST_SEND,
    QP_CONTEXT,
)
from tmb.ccl.misc.utils import P8, as_fn, ld64


# ===-------------------------------------------------------------------=== #
# Data path -- ctx->ops.<fn>, no dlsym
# ===-------------------------------------------------------------------=== #


@always_inline
def post_send(qp: Int, wr: P8, bad_wr: P8) -> Int32:
    """`qp->context->ops.post_send(qp, wr, &bad_wr)`; 0 or an errno."""
    var f = as_fn[def(Int, P8, P8) thin abi("C") -> Int32](
        ld64(
            P8(
                unsafe_from_address=ld64(P8(unsafe_from_address=qp), QP_CONTEXT)
            ),
            CTX_POST_SEND,
        )
    )
    return f(qp, wr, bad_wr)


@always_inline
def post_recv(qp: Int, wr: P8, bad_wr: P8) -> Int32:
    var f = as_fn[def(Int, P8, P8) thin abi("C") -> Int32](
        ld64(
            P8(
                unsafe_from_address=ld64(P8(unsafe_from_address=qp), QP_CONTEXT)
            ),
            CTX_POST_RECV,
        )
    )
    return f(qp, wr, bad_wr)


@always_inline
def poll_cq(cq: Int, num_entries: Int, wc: P8) -> Int32:
    """Number of completions written into `wc`, or negative on error."""
    var f = as_fn[def(Int, Int32, P8) thin abi("C") -> Int32](
        ld64(
            P8(
                unsafe_from_address=ld64(P8(unsafe_from_address=cq), QP_CONTEXT)
            ),
            CTX_POLL_CQ,
        )
    )
    return f(cq, Int32(num_entries), wc)
