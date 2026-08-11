"""Fable-owned FP32/int64 embedding dense backward candidate API.

Semantics (pinned PyTorch commit 70d99e998b4955e0049d13a98d77ae1b14db1f45,
`scale_grad_by_freq=False` only): zero a dense logical
`[num_weights, embedding_dim]` grad_weight, then for every row `i` with
`indices[i] != padding_idx` add `grad_output[i, :]` into row `indices[i]`.
Duplicate indices accumulate; the accumulation order is unspecified, matching
the nondeterministic-algorithms speed protocol.

Three runtime regimes, chosen only from runtime metadata (shapes and pointer
alignment); no shape is compiled in:

  * table (num_weights small enough for a 48 KiB shared table and enough
    indices to expect duplicates): stage A blocks own a (row chunk, column
    chunk) tile, accumulate rows into a per-block shared table with shared
    atomics (only _TABLE_ROWG row-groups can collide, and the 8-deep staging
    keeps global loads in flight), then write the table to context-owned
    scratch; stage B splits the chunk dimension across _RED_TY thread rows
    and writes every output element with plain stores.  Stage B rewrites the
    complete logical output, so this path needs no separate zero pass and no
    global atomics, which is what makes collision-heavy token tables fast.
    Scalar loads make stage A independent of alignment and of
    `embedding_dim % 4`.
  * histogram (large vec4-capable outputs): count each target row once, zero
    only rows whose count is not exactly one, then scatter with a plain
    128-bit store for single-hit rows and a 128-bit vector reduction for
    duplicated rows.  For mostly-unique large tables this removes both the
    zero-pass writes to touched rows and the atomic read-modify-write
    traffic, the two DRAM streams NCU shows dominating the plain path.
  * zero + scatter (everything else): a grid-stride zero over the flat
    output (vec4 body with runtime scalar head/tail around the 16-byte
    alignment boundary), then a grid-stride scatter over flat grad_output
    with relaxed device-scope atomic adds.  The vec4 scatter needs
    `embedding_dim % 4 == 0` plus 16-byte-aligned grad_output and
    grad_weight bases (row starts then inherit base alignment) and uses a
    single 128-bit vector reduction per 4 columns on sm_90+ NVIDIA
    (compile-time guarded, with a portable 4x scalar-atomic fallback);
    anything else takes the scalar variant.
  * ownership (Apple only, small tables): Metal has no threadgroup float
    atomics and its device-scope float atomics serialize badly on
    collision-heavy tables, so each thread instead *owns* one
    (row, column-chunk) output cell, scans the (threadgroup-staged) index
    list, accumulates matching grad_output rows in registers, and writes
    its cell with one plain store.  No atomics, no zero pass; work scales
    with num_weights * num_indices, so the regime is gated to small
    tables and index lists.

Empty logical outputs return before any launch, and empty `indices` still
zeroes the complete output.
"""

from std.atomic import Atomic, Ordering
from std.ffi import _get_global_or_null, external_call
from max.gpu.sync import barrier
from std.gpu import block_idx, grid_dim, thread_idx
from max.gpu.host import DeviceAttribute, DeviceContext
from std.math import ceildiv
from std.memory import AddressSpace, alloc, stack_allocation
from std.sys import inlined_assembly, is_amd_gpu, is_nvidia_gpu
from std.sys.info import _is_sm_9x_or_newer, has_apple_gpu_accelerator

from op_utils import _enqueue_cached, _enqueue_cached_2d

comptime _BLOCK = 256
comptime _VEC = 4
# 48 KiB static shared table: portable per-block budget, sized in f32 elements.
comptime _TABLE_COLS = 128
comptime _TABLE_MAX_ROWS = 96
comptime _TABLE_ROWG = 4
comptime _TABLE_UNROLL = 8
# Grid-y is limited to 65535; row-per-block kernels stride over it.
comptime _MAX_GRID_Y = 65535
comptime _RED_TX = 64
comptime _RED_TY = 4
# Histogram pays two extra small launches; only large outputs amortize them.
comptime _HIST_MIN_OUT = 1 << 22
# Ownership regime (Apple): every thread owns one (row, column-chunk) output
# cell and scans the complete index list, so compare work grows with
# num_weights * num_indices; both dimensions are capped to keep that scan
# cheaper than the atomic scatter it replaces.
comptime _OWN_TX = 32
comptime _OWN_TY = 8
comptime _OWN_TILE = _OWN_TX * _OWN_TY
comptime _OWN_MAX_ROWS = 1024
comptime _OWN_MAX_INDICES = 16384


def _cached_max_grid(ctx: DeviceContext) raises -> Int:
    """max_grid (16 blocks per SM) with the device query cached per context.

    `ctx.get_attribute` costs ~1 ms per call on Metal — it must never sit on
    the per-op path.
    """
    var name = String(t"TMB_EMB_BWD_MAXGRID_{ctx.id()}")
    if global_ptr := _get_global_or_null(name):
        return global_ptr.value().bitcast[Int]()[]
    var sm_count: Int
    try:
        sm_count = ctx.get_attribute(DeviceAttribute.MULTIPROCESSOR_COUNT)
    except:
        sm_count = 64
    var value = max(1, sm_count) * 16
    var cached = alloc[Int](1)
    cached[] = value
    external_call["KGEN_CompilerRT_InsertGlobal", NoneType](
        StringSlice(name), cached.bitcast[NoneType]()
    )
    return value


@always_inline
def _device_scope() -> StaticString:
    # Device-scope atomics skip system-scope ordering cost on the discrete
    # targets; every other accelerator keeps the portable default scope.
    comptime if is_nvidia_gpu():
        return "device"
    elif is_amd_gpu():
        return "agent"
    else:
        return ""


@always_inline
def _atomic_add_f32(
    ptr: UnsafePointer[Scalar[DType.float32], MutAnyOrigin], value: Float32
):
    _ = Atomic[DType.float32, scope=_device_scope()].fetch_add[
        ordering=Ordering.RELAXED
    ](ptr, value)


@always_inline
def _atomic_add_f32_vec4(
    ptr: UnsafePointer[Scalar[DType.float32], MutAnyOrigin],
    value: SIMD[DType.float32, _VEC],
):
    # One 128-bit vector reduction replaces four scalar L2 atomic ops; the
    # scatter regimes are L2-atomic-throughput bound, not DRAM bound.  The
    # address must be 16-byte aligned, which the vec4 dispatch guarantees.
    comptime if is_nvidia_gpu() and _is_sm_9x_or_newer():
        inlined_assembly[
            "red.relaxed.gpu.global.add.v4.f32 [$0], {$1, $2, $3, $4};",
            NoneType,
            constraints="l,f,f,f,f",
            has_side_effect=True,
        ](UInt64(Int(ptr)), value[0], value[1], value[2], value[3])
    else:
        comptime for k in range(_VEC):
            _atomic_add_f32(ptr + k, value[k])


@__name("embedding_dense_backward_zero_vec4")
def _zero_vec4(
    grad_weight: UnsafePointer[Scalar[DType.float32], MutAnyOrigin],
    head_arg: Int64,
    vec_count_arg: Int64,
    tail_arg: Int64,
):
    # Int is not device-passable (host/device width mismatch); scalars cross
    # the launch ABI as Int64 and index math stays in Int.
    var head = Int(head_arg)
    var vec_count = Int(vec_count_arg)
    var tail = Int(tail_arg)
    var tid = Int(block_idx.x) * _BLOCK + Int(thread_idx.x)
    var stride = Int(grid_dim.x) * _BLOCK
    # head/tail are < _VEC scalars around the 16-byte-aligned body.
    if tid < head:
        grad_weight[tid] = 0.0
    if tid < tail:
        grad_weight[head + vec_count * _VEC + tid] = 0.0
    var zeros = SIMD[DType.float32, _VEC](0.0)
    var v = tid
    while v < vec_count:
        grad_weight.store[width=_VEC, alignment=16](head + v * _VEC, zeros)
        v += stride


@__name("embedding_dense_backward_zero_scalar")
def _zero_scalar(
    grad_weight: UnsafePointer[Scalar[DType.float32], MutAnyOrigin],
    elements_arg: Int64,
):
    # Int is not device-passable (host/device width mismatch); scalars cross
    # the launch ABI as Int64 and index math stays in Int.
    var elements = Int(elements_arg)
    var index = Int(block_idx.x) * _BLOCK + Int(thread_idx.x)
    var stride = Int(grid_dim.x) * _BLOCK
    while index < elements:
        grad_weight[index] = 0.0
        index += stride


@__name("embedding_dense_backward_scatter_vec4")
def _scatter_vec4(
    grad_weight: UnsafePointer[Scalar[DType.float32], MutAnyOrigin],
    grad_output: UnsafePointer[Scalar[DType.float32], MutAnyOrigin],
    indices: UnsafePointer[Scalar[DType.int64], MutAnyOrigin],
    num_indices_arg: Int64,
    vec_cols_arg: Int64,
    padding_idx_arg: Int64,
):
    # Int is not device-passable (host/device width mismatch); scalars cross
    # the launch ABI as Int64 and index math stays in Int.
    var num_indices = Int(num_indices_arg)
    var vec_cols = Int(vec_cols_arg)
    var padding_idx = Int(padding_idx_arg)
    var e = Int(block_idx.x) * _BLOCK + Int(thread_idx.x)
    var stride = Int(grid_dim.x) * _BLOCK
    var total = num_indices * vec_cols
    while e < total:
        var row = e // vec_cols
        var col = e - row * vec_cols
        var target = Int(indices[row])
        if target != padding_idx:
            var v = grad_output.load[width=_VEC, alignment=16](e * _VEC)
            _atomic_add_f32_vec4(
                grad_weight + (target * vec_cols + col) * _VEC, v
            )
        e += stride


@__name("embedding_dense_backward_scatter_scalar")
def _scatter_scalar(
    grad_weight: UnsafePointer[Scalar[DType.float32], MutAnyOrigin],
    grad_output: UnsafePointer[Scalar[DType.float32], MutAnyOrigin],
    indices: UnsafePointer[Scalar[DType.int64], MutAnyOrigin],
    num_indices_arg: Int64,
    embedding_dim_arg: Int64,
    padding_idx_arg: Int64,
):
    # Int is not device-passable (host/device width mismatch); scalars cross
    # the launch ABI as Int64 and index math stays in Int.
    var num_indices = Int(num_indices_arg)
    var embedding_dim = Int(embedding_dim_arg)
    var padding_idx = Int(padding_idx_arg)
    var e = Int(block_idx.x) * _BLOCK + Int(thread_idx.x)
    var stride = Int(grid_dim.x) * _BLOCK
    var total = num_indices * embedding_dim
    while e < total:
        var row = e // embedding_dim
        var col = e - row * embedding_dim
        var target = Int(indices[row])
        if target != padding_idx:
            _atomic_add_f32(
                grad_weight + target * embedding_dim + col, grad_output[e]
            )
        e += stride


@__name("embedding_dense_backward_count_zero")
def _count_zero(
    counts: UnsafePointer[Scalar[DType.int32], MutAnyOrigin],
    num_weights_arg: Int64,
):
    # Int is not device-passable (host/device width mismatch); scalars cross
    # the launch ABI as Int64 and index math stays in Int.
    var num_weights = Int(num_weights_arg)
    var index = Int(block_idx.x) * _BLOCK + Int(thread_idx.x)
    var stride = Int(grid_dim.x) * _BLOCK
    while index < num_weights:
        counts[index] = 0
        index += stride


@__name("embedding_dense_backward_count")
def _count(
    counts: UnsafePointer[Scalar[DType.int32], MutAnyOrigin],
    indices: UnsafePointer[Scalar[DType.int64], MutAnyOrigin],
    num_indices_arg: Int64,
    padding_idx_arg: Int64,
):
    # Int is not device-passable (host/device width mismatch); scalars cross
    # the launch ABI as Int64 and index math stays in Int.
    var num_indices = Int(num_indices_arg)
    var padding_idx = Int(padding_idx_arg)
    var index = Int(block_idx.x) * _BLOCK + Int(thread_idx.x)
    var stride = Int(grid_dim.x) * _BLOCK
    while index < num_indices:
        var target = Int(indices[index])
        if target != padding_idx:
            _ = Atomic[DType.int32, scope=_device_scope()].fetch_add[
                ordering=Ordering.RELAXED
            ](counts + target, 1)
        index += stride


@__name("embedding_dense_backward_zero_untouched")
def _zero_untouched(
    grad_weight: UnsafePointer[Scalar[DType.float32], MutAnyOrigin],
    counts: UnsafePointer[Scalar[DType.int32], MutAnyOrigin],
    num_weights_arg: Int64,
    vec_cols_arg: Int64,
):
    # Rows hit exactly once are fully overwritten by the histogram scatter's
    # plain stores, so only every other row needs the zero pass.  The 2D
    # (column chunk, row) grid replaces the flat form's per-element integer
    # division, which cost half the SM issue slots.
    # Int is not device-passable (host/device width mismatch); scalars cross
    # the launch ABI as Int64 and index math stays in Int.
    var num_weights = Int(num_weights_arg)
    var vec_cols = Int(vec_cols_arg)
    var col = Int(block_idx.x) * _BLOCK + Int(thread_idx.x)
    var row = Int(block_idx.y)
    var row_stride = Int(grid_dim.y)
    var zeros = SIMD[DType.float32, _VEC](0.0)
    if col >= vec_cols:
        return
    while row < num_weights:
        if counts[row] != 1:
            grad_weight.store[width=_VEC, alignment=16](
                (row * vec_cols + col) * _VEC, zeros
            )
        row += row_stride


@__name("embedding_dense_backward_scatter_hist_vec4")
def _scatter_hist_vec4(
    grad_weight: UnsafePointer[Scalar[DType.float32], MutAnyOrigin],
    grad_output: UnsafePointer[Scalar[DType.float32], MutAnyOrigin],
    indices: UnsafePointer[Scalar[DType.int64], MutAnyOrigin],
    counts: UnsafePointer[Scalar[DType.int32], MutAnyOrigin],
    num_indices_arg: Int64,
    vec_cols_arg: Int64,
    padding_idx_arg: Int64,
):
    # Int is not device-passable (host/device width mismatch); scalars cross
    # the launch ABI as Int64 and index math stays in Int.
    var num_indices = Int(num_indices_arg)
    var vec_cols = Int(vec_cols_arg)
    var padding_idx = Int(padding_idx_arg)
    var col = Int(block_idx.x) * _BLOCK + Int(thread_idx.x)
    var row = Int(block_idx.y)
    var row_stride = Int(grid_dim.y)
    if col >= vec_cols:
        return
    while row < num_indices:
        var target = Int(indices[row])
        if target != padding_idx:
            var v = grad_output.load[width=_VEC, alignment=16](
                (row * vec_cols + col) * _VEC
            )
            var dst = (target * vec_cols + col) * _VEC
            if counts[target] == 1:
                # Sole contributor: plain store skips the atomic
                # read-modify-write and the zero-pass write for this row.
                grad_weight.store[width=_VEC, alignment=16](dst, v)
            else:
                _atomic_add_f32_vec4(grad_weight + dst, v)
        row += row_stride


@__name("embedding_dense_backward_owner_vec4")
def _owner_vec4(
    grad_weight: UnsafePointer[Scalar[DType.float32], MutAnyOrigin],
    grad_output: UnsafePointer[Scalar[DType.float32], MutAnyOrigin],
    indices: UnsafePointer[Scalar[DType.int64], MutAnyOrigin],
    num_indices_arg: Int64,
    vec_cols_arg: Int64,
    num_weights_arg: Int64,
    padding_idx_arg: Int64,
):
    # Thread (tx, ty) owns output cell (row block_y*_OWN_TY + ty, vec column
    # block_x*_OWN_TX + tx).  Index tiles are staged in threadgroup memory as
    # int32 (valid targets fit: the regime caps num_weights; padding rows are
    # encoded -1, which no owned row matches).  Inactive threads still run
    # every tile iteration so the barriers stay uniform.
    # Int is not device-passable (host/device width mismatch); scalars cross
    # the launch ABI as Int64 and index math stays in Int.
    var num_indices = Int(num_indices_arg)
    var vec_cols = Int(vec_cols_arg)
    var num_weights = Int(num_weights_arg)
    var padding_idx = Int(padding_idx_arg)
    var tile = stack_allocation[
        _OWN_TILE, DType.int32, address_space=AddressSpace.SHARED
    ]()
    var tx = Int(thread_idx.x)
    var ty = Int(thread_idx.y)
    var tid = ty * _OWN_TX + tx
    var col = Int(block_idx.x) * _OWN_TX + tx
    var row = Int(block_idx.y) * _OWN_TY + ty
    var active = row < num_weights and col < vec_cols
    var acc = SIMD[DType.float32, _VEC](0.0)
    var base = 0
    while base < num_indices:
        var count = min(_OWN_TILE, num_indices - base)
        if tid < count:
            var t = Int(indices[base + tid])
            tile[tid] = Int32(-1) if t == padding_idx else Int32(t)
        barrier()
        if active:
            for k in range(count):
                if Int(tile[k]) == row:
                    acc += grad_output.load[width=_VEC, alignment=16](
                        ((base + k) * vec_cols + col) * _VEC
                    )
        barrier()
        base += _OWN_TILE
    if active:
        # Full ownership: the plain store doubles as the zero pass for rows
        # no index touched.
        grad_weight.store[width=_VEC, alignment=16](
            (row * vec_cols + col) * _VEC, acc
        )


@__name("embedding_dense_backward_owner_scalar")
def _owner_scalar(
    grad_weight: UnsafePointer[Scalar[DType.float32], MutAnyOrigin],
    grad_output: UnsafePointer[Scalar[DType.float32], MutAnyOrigin],
    indices: UnsafePointer[Scalar[DType.int64], MutAnyOrigin],
    num_indices_arg: Int64,
    embedding_dim_arg: Int64,
    num_weights_arg: Int64,
    padding_idx_arg: Int64,
):
    # Int is not device-passable (host/device width mismatch); scalars cross
    # the launch ABI as Int64 and index math stays in Int.
    var num_indices = Int(num_indices_arg)
    var embedding_dim = Int(embedding_dim_arg)
    var num_weights = Int(num_weights_arg)
    var padding_idx = Int(padding_idx_arg)
    var tile = stack_allocation[
        _OWN_TILE, DType.int32, address_space=AddressSpace.SHARED
    ]()
    var tx = Int(thread_idx.x)
    var ty = Int(thread_idx.y)
    var tid = ty * _OWN_TX + tx
    var col = Int(block_idx.x) * _OWN_TX + tx
    var row = Int(block_idx.y) * _OWN_TY + ty
    var active = row < num_weights and col < embedding_dim
    var acc = Float32(0.0)
    var base = 0
    while base < num_indices:
        var count = min(_OWN_TILE, num_indices - base)
        if tid < count:
            var t = Int(indices[base + tid])
            tile[tid] = Int32(-1) if t == padding_idx else Int32(t)
        barrier()
        if active:
            for k in range(count):
                if Int(tile[k]) == row:
                    acc += grad_output[(base + k) * embedding_dim + col]
        barrier()
        base += _OWN_TILE
    if active:
        grad_weight[row * embedding_dim + col] = acc


@__name("embedding_dense_backward_table_accum")
def _table_accum(
    scratch: UnsafePointer[Scalar[DType.float32], MutAnyOrigin],
    grad_output: UnsafePointer[Scalar[DType.float32], MutAnyOrigin],
    indices: UnsafePointer[Scalar[DType.int64], MutAnyOrigin],
    num_indices_arg: Int64,
    embedding_dim_arg: Int64,
    num_weights_arg: Int64,
    padding_idx_arg: Int64,
    dim_pad_arg: Int64,
    rows_per_block_arg: Int64,
):
    # Int is not device-passable (host/device width mismatch); scalars cross
    # the launch ABI as Int64 and index math stays in Int.
    var num_indices = Int(num_indices_arg)
    var embedding_dim = Int(embedding_dim_arg)
    var num_weights = Int(num_weights_arg)
    var padding_idx = Int(padding_idx_arg)
    var dim_pad = Int(dim_pad_arg)
    var rows_per_block = Int(rows_per_block_arg)
    var table = stack_allocation[
        _TABLE_MAX_ROWS * _TABLE_COLS,
        DType.float32,
        alignment=16,
        address_space=AddressSpace.SHARED,
    ]()
    var tx = Int(thread_idx.x)
    var ty = Int(thread_idx.y)
    var tid = ty * _TABLE_COLS + tx
    var col = Int(block_idx.x) * _TABLE_COLS + tx

    var used = num_weights * _TABLE_COLS
    var i = tid
    while i < used:
        table[i] = 0.0
        i += _TABLE_COLS * _TABLE_ROWG
    barrier()

    # Thread (tx, ty) owns one grad_output column for row-group ty, so the
    # only same-address races are between the _TABLE_ROWG row-groups; shared
    # atomics resolve those.  The _TABLE_UNROLL-deep staging keeps several
    # independent global loads in flight per thread despite the
    # shared-capacity-limited occupancy.
    var row_end = min(
        Int(block_idx.y) * rows_per_block + rows_per_block, num_indices
    )
    if col < embedding_dim:
        var row = Int(block_idx.y) * rows_per_block + ty
        while row + (_TABLE_UNROLL - 1) * _TABLE_ROWG < row_end:
            var t = InlineArray[Int, _TABLE_UNROLL](uninitialized=True)
            var v = InlineArray[Float32, _TABLE_UNROLL](uninitialized=True)
            comptime for u in range(_TABLE_UNROLL):
                t[u] = Int(indices[row + u * _TABLE_ROWG])
            comptime for u in range(_TABLE_UNROLL):
                v[u] = grad_output[
                    (row + u * _TABLE_ROWG) * embedding_dim + col
                ]
            comptime for u in range(_TABLE_UNROLL):
                if t[u] != padding_idx:
                    _ = Atomic[DType.float32].fetch_add[
                        ordering=Ordering.RELAXED
                    ](table + t[u] * _TABLE_COLS + tx, v[u])
            row += _TABLE_UNROLL * _TABLE_ROWG
        while row < row_end:
            var t = Int(indices[row])
            if t != padding_idx:
                _ = Atomic[DType.float32].fetch_add[ordering=Ordering.RELAXED](
                    table + t * _TABLE_COLS + tx,
                    grad_output[row * embedding_dim + col],
                )
            row += _TABLE_ROWG
    barrier()

    # Columns in [embedding_dim, dim_pad) flush the zeros they were
    # initialized with, keeping stage B's padded vec4 reads deterministic.
    if col < dim_pad:
        var chunk_base = Int(block_idx.y) * num_weights
        var r = ty
        while r < num_weights:
            scratch[(chunk_base + r) * dim_pad + col] = table[
                r * _TABLE_COLS + tx
            ]
            r += _TABLE_ROWG


@__name("embedding_dense_backward_table_reduce")
def _table_reduce(
    grad_weight: UnsafePointer[Scalar[DType.float32], MutAnyOrigin],
    scratch: UnsafePointer[Scalar[DType.float32], MutAnyOrigin],
    num_weights_arg: Int64,
    embedding_dim_arg: Int64,
    dim_pad_arg: Int64,
    chunks_arg: Int64,
    out_vec_ok_arg: Int64,
):
    # thread_idx.y splits the chunk dimension so the small output space still
    # produces enough in-flight loads; a shared tile folds the _RED_TY
    # partials before one thread row stores the row.
    # Int is not device-passable (host/device width mismatch); scalars cross
    # the launch ABI as Int64 and index math stays in Int.
    var num_weights = Int(num_weights_arg)
    var embedding_dim = Int(embedding_dim_arg)
    var dim_pad = Int(dim_pad_arg)
    var chunks = Int(chunks_arg)
    var out_vec_ok = Int(out_vec_ok_arg)
    var partials = stack_allocation[
        _RED_TX * _RED_TY * _VEC,
        DType.float32,
        alignment=16,
        address_space=AddressSpace.SHARED,
    ]()
    var tx = Int(thread_idx.x)
    var ty = Int(thread_idx.y)
    var vec_pad = dim_pad // _VEC
    var total = num_weights * vec_pad
    var chunk_stride = num_weights * dim_pad
    var block_base = Int(block_idx.x) * _RED_TX
    var grid_stride = Int(grid_dim.x) * _RED_TX
    while block_base < total:
        var e = block_base + tx
        var acc0 = SIMD[DType.float32, _VEC](0.0)
        var acc1 = SIMD[DType.float32, _VEC](0.0)
        var acc2 = SIMD[DType.float32, _VEC](0.0)
        var acc3 = SIMD[DType.float32, _VEC](0.0)
        if e < total:
            var r = e // vec_pad
            var base = r * dim_pad + (e - r * vec_pad) * _VEC
            var k = ty
            while k + 3 * _RED_TY < chunks:
                acc0 += scratch.load[width=_VEC, alignment=16](
                    base + k * chunk_stride
                )
                acc1 += scratch.load[width=_VEC, alignment=16](
                    base + (k + _RED_TY) * chunk_stride
                )
                acc2 += scratch.load[width=_VEC, alignment=16](
                    base + (k + 2 * _RED_TY) * chunk_stride
                )
                acc3 += scratch.load[width=_VEC, alignment=16](
                    base + (k + 3 * _RED_TY) * chunk_stride
                )
                k += 4 * _RED_TY
            while k < chunks:
                acc0 += scratch.load[width=_VEC, alignment=16](
                    base + k * chunk_stride
                )
                k += _RED_TY
        partials.store[width=_VEC, alignment=16](
            (ty * _RED_TX + tx) * _VEC, (acc0 + acc1) + (acc2 + acc3)
        )
        barrier()
        if ty == 0 and e < total:
            var acc = SIMD[DType.float32, _VEC](0.0)
            comptime for y in range(_RED_TY):
                acc += partials.load[width=_VEC, alignment=16](
                    (y * _RED_TX + tx) * _VEC
                )
            var r = e // vec_pad
            var c = (e - r * vec_pad) * _VEC
            var out_offset = r * embedding_dim + c
            if out_vec_ok != 0:
                grad_weight.store[width=_VEC, alignment=16](out_offset, acc)
            else:
                comptime for j in range(_VEC):
                    if c + j < embedding_dim:
                        grad_weight[out_offset + j] = acc[j]
        barrier()
        block_base += grid_stride


def enqueue_embedding_dense_backward_f32_i64(
    grad_weight: UnsafePointer[Scalar[DType.float32], MutAnyOrigin],
    grad_output: UnsafePointer[Scalar[DType.float32], MutAnyOrigin],
    indices: UnsafePointer[Scalar[DType.int64], MutAnyOrigin],
    num_indices: Int,
    embedding_dim: Int,
    num_weights: Int,
    padding_idx: Int,
    scale_grad_by_freq: Bool,
    ctx: DeviceContext,
) raises:
    if scale_grad_by_freq:
        raise Error(
            "embedding_dense_backward candidate: scale_grad_by_freq=True is"
            " unsupported"
        )
    var output_elements = num_weights * embedding_dim
    if output_elements <= 0:
        # Empty logical output: nothing to zero and nothing to scatter.
        return

    var max_grid = _cached_max_grid(ctx)
    var sm_count = max_grid // 16
    var out_addr = Int(grad_weight)
    var vec_ok = (
        embedding_dim % _VEC == 0 and (Int(grad_output) | out_addr) % 16 == 0
    )

    # Ownership regime (Apple): no float atomics at all — each thread owns
    # one output cell, scans the staged index list, and writes once.  The
    # store pass covers the complete output, so no zero pass either.  Gated
    # to small tables/index lists because compare work is
    # num_weights * num_indices per column chunk.
    comptime if has_apple_gpu_accelerator():
        if num_weights <= _OWN_MAX_ROWS and num_indices <= _OWN_MAX_INDICES:
            var blocks_y = ceildiv(num_weights, _OWN_TY)
            if vec_ok:
                var vec_cols = embedding_dim // _VEC
                _enqueue_cached_2d[_owner_vec4](
                    ctx,
                    "emb_bwd_owner_vec4",
                    ceildiv(vec_cols, _OWN_TX),
                    blocks_y,
                    1,
                    _OWN_TX,
                    _OWN_TY,
                    grad_weight,
                    grad_output,
                    indices,
                    Int64(num_indices),
                    Int64(vec_cols),
                    Int64(num_weights),
                    Int64(padding_idx),
                )
            else:
                _enqueue_cached_2d[_owner_scalar](
                    ctx,
                    "emb_bwd_owner_scalar",
                    ceildiv(embedding_dim, _OWN_TX),
                    blocks_y,
                    1,
                    _OWN_TX,
                    _OWN_TY,
                    grad_weight,
                    grad_output,
                    indices,
                    Int64(num_indices),
                    Int64(embedding_dim),
                    Int64(num_weights),
                    Int64(padding_idx),
                )
            return

    # Table regime: the whole weight table fits a 48 KiB shared block and
    # duplicates are expected, so shared-memory pre-aggregation avoids the
    # L2 same-address atomic serialization that dominates collision-heavy
    # scatters.  Stage B rewrites the complete output, replacing the zero
    # pass.  Apple GPUs have no threadgroup (local) float atomics — Metal
    # rejects the pipeline at compile time — so they always take the
    # device-scope-atomic regimes below instead.
    comptime table_regime_supported = not has_apple_gpu_accelerator()
    if (
        table_regime_supported
        and num_weights <= _TABLE_MAX_ROWS
        and num_indices >= 2 * num_weights
        and num_indices >= 128
    ):
        var dim_pad = ceildiv(embedding_dim, _VEC) * _VEC
        var col_chunks = ceildiv(embedding_dim, _TABLE_COLS)
        var blocks_y = max(
            1,
            min(
                ceildiv(max(1, sm_count) * 2, col_chunks),
                ceildiv(num_indices, 4 * _TABLE_ROWG * _TABLE_UNROLL),
            ),
        )
        var rows_per_block = ceildiv(num_indices, blocks_y)
        blocks_y = ceildiv(num_indices, rows_per_block)
        var scratch = ctx.enqueue_create_buffer[DType.float32](
            blocks_y * num_weights * dim_pad
        )
        var scratch_ptr = scratch.unsafe_ptr().as_unsafe_any_origin()
        _enqueue_cached_2d[_table_accum](
            ctx,
            "emb_bwd_table_accum",
            col_chunks,
            blocks_y,
            1,
            _TABLE_COLS,
            _TABLE_ROWG,
            scratch_ptr,
            grad_output,
            indices,
            Int64(num_indices),
            Int64(embedding_dim),
            Int64(num_weights),
            Int64(padding_idx),
            Int64(dim_pad),
            Int64(rows_per_block),
        )
        var out_vec_ok = 1 if (
            embedding_dim % _VEC == 0 and out_addr % 16 == 0
        ) else 0
        var total_vec = num_weights * (dim_pad // _VEC)
        _enqueue_cached_2d[_table_reduce](
            ctx,
            "emb_bwd_table_reduce",
            max(1, min(ceildiv(total_vec, _RED_TX), max_grid)),
            1,
            1,
            _RED_TX,
            _RED_TY,
            grad_weight,
            scratch_ptr,
            Int64(num_weights),
            Int64(embedding_dim),
            Int64(dim_pad),
            Int64(blocks_y),
            Int64(out_vec_ok),
        )
        # Normal release after both stream-ordered consumers are enqueued.
        _ = scratch^
        return

    # Histogram regime: for large vec4-capable outputs the plain path's cost
    # is dominated by the full zero pass plus atomic read-modify-write; a
    # cheap per-row count lets single-hit rows skip both.
    if num_indices > 0 and vec_ok and output_elements >= _HIST_MIN_OUT:
        var vec_cols = embedding_dim // _VEC
        var counts = ctx.enqueue_create_buffer[DType.int32](num_weights)
        var counts_ptr = counts.unsafe_ptr().as_unsafe_any_origin()
        _enqueue_cached[_count_zero](
            ctx,
            "emb_bwd_count_zero",
            max(1, min(ceildiv(num_weights, _BLOCK), max_grid)),
            1,
            1,
            _BLOCK,
            counts_ptr,
            Int64(num_weights),
        )
        _enqueue_cached[_count](
            ctx,
            "emb_bwd_count",
            max(1, min(ceildiv(num_indices, _BLOCK), max_grid)),
            1,
            1,
            _BLOCK,
            counts_ptr,
            indices,
            Int64(num_indices),
            Int64(padding_idx),
        )
        var col_blocks = ceildiv(vec_cols, _BLOCK)
        _enqueue_cached[_zero_untouched](
            ctx,
            "emb_bwd_zero_untouched",
            col_blocks,
            min(num_weights, _MAX_GRID_Y),
            1,
            _BLOCK,
            grad_weight,
            counts_ptr,
            Int64(num_weights),
            Int64(vec_cols),
        )
        _enqueue_cached[_scatter_hist_vec4](
            ctx,
            "emb_bwd_scatter_hist_vec4",
            col_blocks,
            min(num_indices, _MAX_GRID_Y),
            1,
            _BLOCK,
            grad_weight,
            grad_output,
            indices,
            counts_ptr,
            Int64(num_indices),
            Int64(vec_cols),
            Int64(padding_idx),
        )
        _ = counts^
        return

    # Stage 1: zero the complete logical output on every invocation.
    if out_addr % 4 == 0:
        # Scalar head up to the first 16-byte boundary, vec4 body, scalar
        # tail; head/tail stay below _VEC by construction.
        var head = min(((16 - out_addr % 16) % 16) // 4, output_elements)
        var body = output_elements - head
        var vec_count = body // _VEC
        var tail = body - vec_count * _VEC
        var grid = max(1, min(ceildiv(vec_count, _BLOCK), max_grid))
        _enqueue_cached[_zero_vec4](
            ctx,
            "emb_bwd_zero_vec4",
            grid,
            1,
            1,
            _BLOCK,
            grad_weight,
            Int64(head),
            Int64(vec_count),
            Int64(tail),
        )
    else:
        var grid = max(1, min(ceildiv(output_elements, _BLOCK), max_grid))
        _enqueue_cached[_zero_scalar](
            ctx,
            "emb_bwd_zero_scalar",
            grid,
            1,
            1,
            _BLOCK,
            grad_weight,
            Int64(output_elements),
        )

    if num_indices <= 0:
        return

    # Stage 2: scatter-accumulate grad_output rows.  With
    # embedding_dim % 4 == 0 every row start inherits its base alignment, so
    # one runtime check on both bases covers all rows.
    if vec_ok:
        var vec_cols = embedding_dim // _VEC
        var total = num_indices * vec_cols
        var grid = max(1, min(ceildiv(total, _BLOCK), max_grid))
        _enqueue_cached[_scatter_vec4](
            ctx,
            "emb_bwd_scatter_vec4",
            grid,
            1,
            1,
            _BLOCK,
            grad_weight,
            grad_output,
            indices,
            Int64(num_indices),
            Int64(vec_cols),
            Int64(padding_idx),
        )
    else:
        var total = num_indices * embedding_dim
        var grid = max(1, min(ceildiv(total, _BLOCK), max_grid))
        _enqueue_cached[_scatter_scalar](
            ctx,
            "emb_bwd_scatter_scalar",
            grid,
            1,
            1,
            _BLOCK,
            grad_weight,
            grad_output,
            indices,
            Int64(num_indices),
            Int64(embedding_dim),
            Int64(padding_idx),
        )
