# ===----------------------------------------------------------------------=== #
# Fused eager-mode log_softmax backward kernels for mojo_device
# (float32/float16/bfloat16 GPU), validated in
# kernel_bench/bench_log_softmax_bwd.mojo on H100.
#
#   grad_input = grad_output - exp(output) * rowsum(grad_output)
#
# over the trailing dim, with fp32 accumulation. `output` holds log-probs
# (<= 0), so exp(output) is in [0, 1] and never overflows in any dtype.
#
# The op is memory-bound; the 3-stream minimum DRAM traffic is read grad +
# read output + write grad_input. The primary kernel stages the grad row in
# dynamic shared memory during the rowsum pass so the emit pass never
# re-reads grad from DRAM, and reaches ~95% of HBM roofline at the nanogpt
# cross-entropy shape (32768x50304 bf16: 5.2 ms vs torch-CUDA's fused
# kernel at 6.9 ms on the same box). One thread block processes one row:
# 16-byte vectorized loads/stores over the aligned row body (base pointers
# are 16B-aligned, enforced by the host; odd-cols rows get a scalar
# head/tail), a warp-shuffle block.sum for the fp32 rowsum, and a
# measured-on-H100 block-size heuristic (long rows want 1024 threads —
# full occupancy at 2 blocks/SM with ~100 KB smem each; short rows want
# smaller blocks; tiny grids want the fattest block available since
# blocks, not threads, are the scarce resource).
#
# Rows too long for the device's opt-in shared-memory capacity (or devices
# where the opt-in attribute is unavailable) fall back to a no-staging
# variant of the same kernel that re-reads grad through L2/DRAM.
#
# Shapes are fully dynamic: any rows/cols >= 1, any of the three dtypes.
# ===----------------------------------------------------------------------=== #

from std.builtin.device_passable import DevicePassable
from std.ffi import _get_global_or_null, external_call
from std.gpu import (
    MAX_THREADS_PER_BLOCK_METADATA,
    block_idx,
    grid_dim,
    thread_idx,
)
from std.gpu.host import DeviceAttribute, DeviceContext, FuncAttribute
from std.gpu.memory import AddressSpace, external_memory
from std.gpu.primitives import block
from std.math import exp
from std.memory import alloc
from std.sys.info import has_accelerator, size_of
from std.utils.static_tuple import StaticTuple

from op_utils import _enqueue_cached

# NVIDIA's default dynamic shared-memory limit; anything above needs the
# MAX_DYNAMIC_SHARED_SIZE_BYTES opt-in function attribute.
comptime _DEFAULT_DYN_SMEM = 48 * 1024

comptime _NOSMEM_THREADS = 256


@__llvm_metadata(
    MAX_THREADS_PER_BLOCK_METADATA=StaticTuple[Int32, 1](Int32(threads))
)
@__name(t"lsm_bwd_smem_{dtype}_{threads}")
def _log_softmax_bwd_smem_kernel[
    dtype: DType, threads: Int
](
    gi: UnsafePointer[Scalar[dtype], MutAnyOrigin],
    g: UnsafePointer[Scalar[dtype], ImmutAnyOrigin],
    o: UnsafePointer[Scalar[dtype], ImmutAnyOrigin],
    rows: Int,
    cols: Int,
):
    # One row per block. Phase 1 stages the vectorized body of the grad row
    # into dynamic shared memory while accumulating the fp32 rowsum, so
    # phase 2 never re-reads grad from DRAM: total traffic is the 3-stream
    # minimum. Each thread re-reads exactly the shared words it wrote (the
    # same v-loop), so the staging itself needs no barrier. The (at most
    # VEC-1) head/tail scalars of unaligned rows re-read global memory and
    # hit L1/L2.
    comptime VEC = 16 // size_of[dtype]()
    var smem_g = external_memory[
        Scalar[dtype], address_space=AddressSpace.SHARED, alignment=16
    ]()
    var tid = Int(thread_idx.x)
    var row = Int(block_idx.x)
    var base = row * cols
    var head = (VEC - (base % VEC)) % VEC
    if head > cols:
        head = cols
    var nvec = (cols - head) // VEC
    var tail_start = head + nvec * VEC

    var vsum = SIMD[DType.float32, VEC](0)
    var ssum = Float32(0)
    if tid < head:
        ssum += g[base + tid].cast[DType.float32]()
    var v = tid
    while v < nvec:
        var gv = g.load[width=VEC, alignment=16](base + head + v * VEC)
        smem_g.store[width=VEC, alignment=16](v * VEC, gv)
        vsum += gv.cast[DType.float32]()
        v += threads
    var j = tail_start + tid
    while j < cols:
        ssum += g[base + j].cast[DType.float32]()
        j += threads
    var srow = block.sum[block_size=threads, broadcast=True](
        vsum.reduce_add() + ssum
    )

    if tid < head:
        var idx = base + tid
        gi[idx] = (
            g[idx].cast[DType.float32]()
            - exp(o[idx].cast[DType.float32]()) * srow
        ).cast[dtype]()
    v = tid
    while v < nvec:
        var idx = base + head + v * VEC
        var gv = smem_g.load[width=VEC, alignment=16](v * VEC).cast[
            DType.float32
        ]()
        var ov = o.load[width=VEC, alignment=16](idx).cast[DType.float32]()
        gi.store[width=VEC, alignment=16](
            idx, (gv - exp(ov) * srow).cast[dtype]()
        )
        v += threads
    j = tail_start + tid
    while j < cols:
        var idx = base + j
        gi[idx] = (
            g[idx].cast[DType.float32]()
            - exp(o[idx].cast[DType.float32]()) * srow
        ).cast[dtype]()
        j += threads


@__llvm_metadata(
    MAX_THREADS_PER_BLOCK_METADATA=StaticTuple[Int32, 1](Int32(_NOSMEM_THREADS))
)
@__name(t"lsm_bwd_nosmem_{dtype}")
def _log_softmax_bwd_nosmem_kernel[
    dtype: DType
](
    gi: UnsafePointer[Scalar[dtype], MutAnyOrigin],
    g: UnsafePointer[Scalar[dtype], ImmutAnyOrigin],
    o: UnsafePointer[Scalar[dtype], ImmutAnyOrigin],
    rows: Int,
    cols: Int,
):
    # Fallback for rows whose vector body exceeds the dynamic shared-memory
    # capacity: same structure, but phase 2 re-reads grad through L2/DRAM.
    comptime VEC = 16 // size_of[dtype]()
    var tid = Int(thread_idx.x)
    var row = Int(block_idx.x)
    var base = row * cols
    var head = (VEC - (base % VEC)) % VEC
    if head > cols:
        head = cols
    var nvec = (cols - head) // VEC
    var tail_start = head + nvec * VEC

    var vsum = SIMD[DType.float32, VEC](0)
    var ssum = Float32(0)
    if tid < head:
        ssum += g[base + tid].cast[DType.float32]()
    var v = tid
    while v < nvec:
        var gv = g.load[width=VEC, alignment=16](base + head + v * VEC)
        vsum += gv.cast[DType.float32]()
        v += _NOSMEM_THREADS
    var j = tail_start + tid
    while j < cols:
        ssum += g[base + j].cast[DType.float32]()
        j += _NOSMEM_THREADS
    var srow = block.sum[block_size=_NOSMEM_THREADS, broadcast=True](
        vsum.reduce_add() + ssum
    )

    if tid < head:
        var idx = base + tid
        gi[idx] = (
            g[idx].cast[DType.float32]()
            - exp(o[idx].cast[DType.float32]()) * srow
        ).cast[dtype]()
    v = tid
    while v < nvec:
        var idx = base + head + v * VEC
        var gv = g.load[width=VEC, alignment=16](idx).cast[DType.float32]()
        var ov = o.load[width=VEC, alignment=16](idx).cast[DType.float32]()
        gi.store[width=VEC, alignment=16](
            idx, (gv - exp(ov) * srow).cast[dtype]()
        )
        v += _NOSMEM_THREADS
    j = tail_start + tid
    while j < cols:
        var idx = base + j
        gi[idx] = (
            g[idx].cast[DType.float32]()
            - exp(o[idx].cast[DType.float32]()) * srow
        ).cast[dtype]()
        j += _NOSMEM_THREADS


@always_inline
def _enqueue_cached_smem[
    declared_arg_types: TypeList[Trait=AnyType, ...],
    //,
    func: def(* args: * declared_arg_types) thin -> None,
    *Ts: DevicePassable,
](
    ctx: DeviceContext,
    key: String,
    grid: Int,
    threads: Int,
    smem_bytes: Int,
    attr_bytes: Int,
    *args: *Ts,
) raises:
    """`op_utils._enqueue_cached` with dynamic shared memory.

    The MAX_DYNAMIC_SHARED_SIZE_BYTES function attribute is baked in when
    the `DeviceFunction` is first compiled and cached, so the caller must
    fold the attribute regime into `key` (kernels launched both above and
    below the 48 KB opt-in boundary need distinct keys) and must pass the
    same `attr_bytes` for every call with the same key.
    """
    var name = String(t"TMB_KERNEL_{key}_{ctx.id()}")
    comptime FuncT = type_of(ctx.compile_function[func]())

    if global_ptr := _get_global_or_null(name):
        var fptr = global_ptr.value().bitcast[FuncT]()
        ctx.enqueue_function(
            fptr[],
            *args,
            grid_dim=(grid,),
            block_dim=(threads,),
            shared_mem_bytes=smem_bytes,
        )
        return

    var compiled: FuncT
    if attr_bytes > 0:
        compiled = ctx.compile_function[func](
            func_attribute=FuncAttribute.MAX_DYNAMIC_SHARED_SIZE_BYTES(
                UInt32(attr_bytes)
            )
        )
    else:
        compiled = ctx.compile_function[func]()
    var fptr = alloc[FuncT](1)
    fptr.init_pointee_move(compiled^)
    external_call["KGEN_CompilerRT_InsertGlobal", NoneType](
        StringSlice(name),
        fptr.bitcast[NoneType](),
    )
    ctx.enqueue_function(
        fptr[],
        *args,
        grid_dim=(grid,),
        block_dim=(threads,),
        shared_mem_bytes=smem_bytes,
    )


@always_inline
def _dyn_smem_capacity(ctx: DeviceContext) -> Int:
    """Usable per-block dynamic shared memory on this device.

    Above the 48 KB default this queries the NVIDIA opt-in limit (minus the
    1 KB system reservation). Where the attribute is unavailable (e.g. AMD)
    it conservatively reports the default, routing long rows to the
    no-staging kernel.
    """
    var capacity = _DEFAULT_DYN_SMEM
    try:
        var opt_in = ctx.get_attribute(
            DeviceAttribute.MAX_SHARED_MEMORY_PER_BLOCK_OPTIN
        )
        if opt_in - 1024 > capacity:
            capacity = opt_in - 1024
    except:
        # Deliberate fallback: the attribute is unavailable on this device
        # (e.g. AMD); keep the conservative default capacity.
        capacity = _DEFAULT_DYN_SMEM
    return capacity


def enqueue_log_softmax_backward[
    dtype: DType
](
    gi: UnsafePointer[Scalar[dtype], MutAnyOrigin],
    g: UnsafePointer[Scalar[dtype], ImmutAnyOrigin],
    o: UnsafePointer[Scalar[dtype], ImmutAnyOrigin],
    rows: Int,
    cols: Int,
    ctx: DeviceContext,
) raises:
    """Fire-and-forget trailing-dim log_softmax backward.

    Contract (validated by the Python caller): contiguous row-major 2D
    views [rows, cols] with rows, cols >= 1; all three base pointers
    16-byte aligned; rows < 2**31 (one thread block per row).
    """
    comptime if not has_accelerator():
        raise Error("no GPU accelerator available at compile time")
    else:
        comptime VEC = 16 // size_of[dtype]()
        # Worst-case vector-body bytes over rows (head == 0 rows).
        var smem_bytes = (cols // VEC) * 16
        var use_smem = False
        var attr_bytes = 0
        if smem_bytes > 0 and smem_bytes <= _DEFAULT_DYN_SMEM:
            use_smem = True
        elif smem_bytes > _DEFAULT_DYN_SMEM:
            var capacity = _dyn_smem_capacity(ctx)
            if smem_bytes <= capacity:
                use_smem = True
                attr_bytes = capacity
        if use_smem:
            # Measured-on-H100 block-size heuristic; see file header.
            var nvec_max = cols // VEC
            var suffix = "l" if attr_bytes > 0 else "s"
            if nvec_max >= 2048 or rows <= 228:
                _enqueue_cached_smem[_log_softmax_bwd_smem_kernel[dtype, 1024]](
                    ctx,
                    String(t"lsm_bwd_smem_{dtype}_1024_{suffix}"),
                    rows,
                    1024,
                    smem_bytes,
                    attr_bytes,
                    gi,
                    g,
                    o,
                    rows,
                    cols,
                )
            elif nvec_max >= 1024:
                _enqueue_cached_smem[_log_softmax_bwd_smem_kernel[dtype, 512]](
                    ctx,
                    String(t"lsm_bwd_smem_{dtype}_512_{suffix}"),
                    rows,
                    512,
                    smem_bytes,
                    attr_bytes,
                    gi,
                    g,
                    o,
                    rows,
                    cols,
                )
            else:
                _enqueue_cached_smem[_log_softmax_bwd_smem_kernel[dtype, 256]](
                    ctx,
                    String(t"lsm_bwd_smem_{dtype}_256_{suffix}"),
                    rows,
                    256,
                    smem_bytes,
                    attr_bytes,
                    gi,
                    g,
                    o,
                    rows,
                    cols,
                )
        else:
            _enqueue_cached[_log_softmax_bwd_nosmem_kernel[dtype]](
                ctx,
                String(t"lsm_bwd_nosmem_{dtype}"),
                rows,
                1,
                1,
                _NOSMEM_THREADS,
                gi,
                g,
                o,
                rows,
                cols,
            )
