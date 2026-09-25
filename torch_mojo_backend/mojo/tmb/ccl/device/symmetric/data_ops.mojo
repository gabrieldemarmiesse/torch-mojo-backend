# Rewrite of: https://github.com/NVIDIA/nccl/blob/master/src/device/symmetric/data_ops.cuh

from std.collections import Array


@always_inline
def _copy_vec[
    dtype: DType, W: Int, U: Int
](
    dst: Pointer[Scalar[dtype], MutAnyOrigin],
    src: Pointer[Scalar[dtype], MutAnyOrigin],
    nvec: Int,
    tid: Int,
    stride: Int,
):
    """Grid-stride copy of `nvec` 16-byte vectors, `U` of them in flight.

    Both pointers must be 16-byte aligned, which every address this file forms
    is (region sub-areas are multiples of 16 B from a page-aligned base; user
    pointers come from MAX's allocator; shard starts are multiples of W).
    """
    var v = tid
    var lim = nvec - (U - 1) * stride
    while v < lim:
        var tmp = Array[SIMD[dtype, W], U](uninitialized=True)
        comptime for u in range(U):
            tmp[u] = src.unsafe_load[width=W, alignment=16](
                (v + u * stride) * W
            )
        comptime for u in range(U):
            dst.unsafe_store[width=W, alignment=16](
                (v + u * stride) * W, tmp[u]
            )
        v += U * stride
    # The remainder is at most U-1 vectors per thread: issue every load
    # before the first store, as the main loop does, rather than one round
    # trip per vector. Same vector -> thread mapping either way.
    if v < nvec:
        var tmp = Array[SIMD[dtype, W], U](uninitialized=True)
        comptime for u in range(U):
            if v + u * stride < nvec:
                tmp[u] = src.unsafe_load[width=W, alignment=16](
                    (v + u * stride) * W
                )
        comptime for u in range(U):
            if v + u * stride < nvec:
                dst.unsafe_store[width=W, alignment=16](
                    (v + u * stride) * W, tmp[u]
                )


@always_inline
def _copy_scalar_tail[
    dtype: DType
](
    dst: Pointer[Scalar[dtype], MutAnyOrigin],
    src: Pointer[Scalar[dtype], MutAnyOrigin],
    base: Int,
    count: Int,
    tid: Int,
    stride: Int,
):
    """The `numel % W` elements a 16-byte vector loop cannot cover."""
    for i in range(tid, count, stride):
        dst[unsafe_offset=base + i] = src[unsafe_offset=base + i]


@always_inline
def _copy_bytes[
    U: Int
](
    dst: Pointer[UInt8, MutAnyOrigin],
    src: Pointer[UInt8, MutAnyOrigin],
    nbytes: Int,
    tid: Int,
    stride: Int,
):
    """dtype-agnostic byte copy: 16-byte vectors when both sides allow it.

    Both paths walk the *same* 16-byte chunks in the same grid-stride order, so
    the chunk -> block mapping does not depend on which path a given call takes.
    That matters: for a byte collective the writer and the reader are different
    ranks looking at different pointer pairs (the root's `send` and a peer's
    `recv`), so they can disagree about alignment, and the per-block barrier
    only orders block b against block b. A fallback that walked single bytes
    would let block 3 read a chunk block 0 wrote.
    """
    var nvec = nbytes // 16
    if (Int(dst) | Int(src)) % 16 == 0:
        _copy_vec[DType.uint8, 16, U](dst, src, nvec, tid, stride)
    elif (Int(dst) | Int(src)) % 4 == 0:
        # Four 4-byte accesses per chunk instead of sixteen 1-byte ones: the
        # common miss is a view whose element offset is not a multiple of the
        # 16-byte width, which is still 4-byte aligned for every dtype wider
        # than a byte.
        var d4 = dst.unsafe_bitcast[UInt32]()
        var s4 = src.unsafe_bitcast[UInt32]()
        for v in range(tid, nvec, stride):
            comptime for j in range(4):
                d4[unsafe_offset=v * 4 + j] = s4[unsafe_offset=v * 4 + j]
    else:
        for v in range(tid, nvec, stride):
            comptime for j in range(16):
                dst[unsafe_offset=v * 16 + j] = src[unsafe_offset=v * 16 + j]
    _copy_scalar_tail(dst, src, nvec * 16, nbytes - nvec * 16, tid, stride)


@always_inline
def _copy_bytes2[
    U: Int
](
    dst_a: Pointer[UInt8, MutAnyOrigin],
    dst_b: Pointer[UInt8, MutAnyOrigin],
    src: Pointer[UInt8, MutAnyOrigin],
    nbytes: Int,
    tid: Int,
    stride: Int,
):
    """`_copy_bytes` to two destinations, reading the source once.

    The all-gather's local half writes my contribution both into my own
    region (for the peers to read) and into my own slice of the output, and
    the two copies together were two reads of it. One read and two stores
    is a quarter less HBM traffic in that phase. Same 16-byte chunk ->
    thread mapping as `_copy_bytes`, whatever path any of them takes.
    """
    var nvec = nbytes // 16
    if (Int(dst_a) | Int(dst_b) | Int(src)) % 16 == 0:
        var v = tid
        var lim = nvec - (U - 1) * stride
        while v < lim:
            var tmp = Array[SIMD[DType.uint8, 16], U](uninitialized=True)
            comptime for u in range(U):
                tmp[u] = src.unsafe_load[width=16, alignment=16](
                    (v + u * stride) * 16
                )
            comptime for u in range(U):
                dst_a.unsafe_store[width=16, alignment=16](
                    (v + u * stride) * 16, tmp[u]
                )
                dst_b.unsafe_store[width=16, alignment=16](
                    (v + u * stride) * 16, tmp[u]
                )
            v += U * stride
        if v < nvec:
            var tmp = Array[SIMD[DType.uint8, 16], U](uninitialized=True)
            comptime for u in range(U):
                if v + u * stride < nvec:
                    tmp[u] = src.unsafe_load[width=16, alignment=16](
                        (v + u * stride) * 16
                    )
            comptime for u in range(U):
                if v + u * stride < nvec:
                    dst_a.unsafe_store[width=16, alignment=16](
                        (v + u * stride) * 16, tmp[u]
                    )
                    dst_b.unsafe_store[width=16, alignment=16](
                        (v + u * stride) * 16, tmp[u]
                    )
    elif (Int(dst_a) | Int(dst_b) | Int(src)) % 4 == 0:
        var a4 = dst_a.unsafe_bitcast[UInt32]()
        var b4 = dst_b.unsafe_bitcast[UInt32]()
        var s4 = src.unsafe_bitcast[UInt32]()
        for v in range(tid, nvec, stride):
            comptime for j in range(4):
                var x = s4[unsafe_offset=v * 4 + j]
                a4[unsafe_offset=v * 4 + j] = x
                b4[unsafe_offset=v * 4 + j] = x
    else:
        for v in range(tid, nvec, stride):
            comptime for j in range(16):
                var x = src[unsafe_offset=v * 16 + j]
                dst_a[unsafe_offset=v * 16 + j] = x
                dst_b[unsafe_offset=v * 16 + j] = x
    var base = nvec * 16
    for i in range(tid, nbytes - base, stride):
        var x = src[unsafe_offset=base + i]
        dst_a[unsafe_offset=base + i] = x
        dst_b[unsafe_offset=base + i] = x


@always_inline
def _copy_span[
    dtype: DType, W: Int, U: Int
](
    dst: Pointer[Scalar[dtype], MutAnyOrigin],
    src: Pointer[Scalar[dtype], MutAnyOrigin],
    count: Int,
    tid: Int,
    stride: Int,
):
    """`count` elements: the 16-byte vectors, then the `count % W` tail."""
    var vc = count // W
    _copy_vec[dtype, W, U](dst, src, vc, tid, stride)
    _copy_scalar_tail(dst, src, vc * W, count - vc * W, tid, stride)


@always_inline
def _copy_span_flex[
    dtype: DType, W: Int, U: Int
](
    dst: Pointer[Scalar[dtype], MutAnyOrigin],
    src: Pointer[Scalar[dtype], MutAnyOrigin],
    count: Int,
    tid: Int,
    stride: Int,
    vec: Bool,
):
    """`_copy_span` where 16-byte vectors are allowed, else the same
    W-element groups moved one element at a time.

    Both paths hand group `g` to the same thread, so a reader taking the other
    path still reads only what its own block index wrote -- the rule
    `_copy_bytes` documents, here because the two ranks of one push are
    different pointers and may disagree about alignment.
    """
    if vec:
        _copy_span[dtype, W, U](dst, src, count, tid, stride)
        return
    var vc = count // W
    for v in range(tid, vc, stride):
        comptime for e in range(W):
            dst[unsafe_offset=v * W + e] = src[unsafe_offset=v * W + e]
    _copy_scalar_tail(dst, src, vc * W, count - vc * W, tid, stride)


@always_inline
def _copy_span_scaled[
    dtype: DType, W: Int, U: Int
](
    dst: Pointer[Scalar[dtype], MutAnyOrigin],
    src: Pointer[Scalar[dtype], MutAnyOrigin],
    count: Int,
    tid: Int,
    stride: Int,
    scale: Float32,
):
    """`_copy_span` times `scale`; integer dtypes ignore `scale` (as NCCL's
    ncclAvg does, and as the fused allreduce does). `scale == 1` takes the
    plain copy, so the all-gather half of a SUM allreduce costs no more than a
    peer copy."""
    comptime accum = DType.float32 if (
        dtype == DType.bfloat16 or dtype == DType.float16
    ) else dtype
    comptime if accum.is_floating_point():
        if scale != Float32(1.0):
            var sv = SIMD[accum, W](scale.cast[accum]())
            var vc = count // W
            var v = tid
            var lim = vc - (U - 1) * stride
            while v < lim:
                var tmp = Array[SIMD[dtype, W], U](uninitialized=True)
                comptime for u in range(U):
                    tmp[u] = src.unsafe_load[width=W, alignment=16](
                        (v + u * stride) * W
                    )
                comptime for u in range(U):
                    dst.unsafe_store[width=W, alignment=16](
                        (v + u * stride) * W,
                        (tmp[u].cast[accum]() * sv).cast[dtype](),
                    )
                v += U * stride
            if v < vc:
                var tmp = Array[SIMD[dtype, W], U](uninitialized=True)
                comptime for u in range(U):
                    if v + u * stride < vc:
                        tmp[u] = src.unsafe_load[width=W, alignment=16](
                            (v + u * stride) * W
                        )
                comptime for u in range(U):
                    if v + u * stride < vc:
                        dst.unsafe_store[width=W, alignment=16](
                            (v + u * stride) * W,
                            (tmp[u].cast[accum]() * sv).cast[dtype](),
                        )
            for i in range(tid, count - vc * W, stride):
                var k = vc * W + i
                dst[unsafe_offset=k] = (
                    src[unsafe_offset=k].cast[accum]() * scale.cast[accum]()
                ).cast[dtype]()
            return
    _copy_span[dtype, W, U](dst, src, count, tid, stride)


@always_inline
def _share[
    accum: DType, W: Int
](v: SIMD[accum, W], scale: Float32) -> SIMD[accum, W]:
    """One rank's contribution to an AVG: scaled BEFORE it joins the sum.

    Scaling the finished sum instead overflows where NCCL does not -- two fp32
    ranks contributing 2**127 average to inf rather than to 2**127 -- and for
    a power-of-two world x/world is exact away from the subnormal range (a
    contribution below 2**-149 * world rounds to zero before the sum), so the
    result matches post-scaling everywhere else. This is NCCL's PreMulSum for
    AVG (nccl:src/enqueue/enqueue.cc:2517) applied in the fp32 accumulator
    rather than in the tensor dtype, and what the NVLS path already does at
    copy-in. Integer dtypes ignore `scale`, as ncclAvg does; `scale` is 1 for
    SUM, and the multiply by it is exact.
    """
    comptime if accum.is_floating_point():
        return v * SIMD[accum, W](scale.cast[accum]())
    else:
        return v
