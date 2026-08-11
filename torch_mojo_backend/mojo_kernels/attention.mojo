from max.gpu.sync import barrier
from std.gpu import MAX_THREADS_PER_BLOCK_METADATA, block_idx, thread_idx
from max.gpu.host import DeviceContext
from std.memory import AddressSpace
from std.math import exp, sqrt
from std.memory import stack_allocation
from std.sys.info import size_of
from std.utils.static_tuple import StaticTuple

import extensibility as compiler
from extensibility import InputTensor, OutputTensor
from layout import Layout, LayoutTensor


comptime THREADS = 256
comptime MAX_KV = 4096
comptime MAX_HEAD_DIM = 256


@__llvm_metadata(
    MAX_THREADS_PER_BLOCK_METADATA=StaticTuple[Int32, 1](Int32(THREADS))
)
def _gpt2_decode_attention_kernel[
    dtype: DType,
    o_layout: Layout,
    q_layout: Layout,
    k_layout: Layout,
    v_layout: Layout,
    mask_layout: Layout,
](
    output: LayoutTensor[dtype, o_layout, MutAnyOrigin],
    query: LayoutTensor[dtype, q_layout, MutAnyOrigin],
    key: LayoutTensor[dtype, k_layout, MutAnyOrigin],
    value: LayoutTensor[dtype, v_layout, MutAnyOrigin],
    mask: LayoutTensor[dtype, mask_layout, MutAnyOrigin],
    kv_len_arg: Int64,
    head_dim_arg: Int64,
    heads_arg: Int64,
    mask_batch_arg: Int64,
):
    """Single-query attention for contiguous [B, H, S, D] tensors."""
    # Int is not device-passable (host/device width mismatch); scalars cross
    # the launch ABI as Int64 and index math stays in Int.
    var kv_len = Int(kv_len_arg)
    var head_dim = Int(head_dim_arg)
    var heads = Int(heads_arg)
    var mask_batch = Int(mask_batch_arg)
    comptime vec_align = 4 * size_of[dtype]()
    var bh = block_idx.x
    var tid = thread_idx.x
    var out_base = bh * head_dim
    var q_base = bh * head_dim
    var kv_base = bh * kv_len * head_dim
    var mask_base = (0 if mask_batch == 1 else bh // heads) * kv_len
    var scale = 1.0 / sqrt(Float32(head_dim))

    var q_smem = stack_allocation[
        MAX_HEAD_DIM, DType.float32, address_space=AddressSpace.SHARED
    ]()
    var scores = stack_allocation[
        MAX_KV, DType.float32, address_space=AddressSpace.SHARED
    ]()
    var reduction = stack_allocation[
        THREADS, DType.float32, address_space=AddressSpace.SHARED
    ]()
    var broadcast = stack_allocation[
        2, DType.float32, address_space=AddressSpace.SHARED
    ]()

    for d in range(tid, head_dim, THREADS):
        q_smem[d] = query.ptr[q_base + d].cast[DType.float32]()
    barrier()

    var row_max = Float32.MIN
    for j in range(tid, kv_len, THREADS):
        var krow = kv_base + j * head_dim
        var dot = Float32(0)
        for d in range(0, head_dim, 4):
            var k4 = key.ptr.load[width=4, alignment=vec_align](krow + d).cast[
                DType.float32
            ]()
            var q4 = q_smem.load[width=4, alignment=16](d)
            dot += (q4 * k4).reduce_add()
        var score = dot * scale + mask.ptr[mask_base + j].cast[DType.float32]()
        scores[j] = score
        row_max = max(row_max, score)

    reduction[tid] = row_max
    barrier()
    var stride = THREADS // 2
    comptime for _ in range(8):
        if tid < stride:
            reduction[tid] = max(reduction[tid], reduction[tid + stride])
        barrier()
        stride //= 2
    if tid == 0:
        broadcast[0] = reduction[0]
    barrier()
    row_max = broadcast[0]

    var row_sum = Float32(0)
    for j in range(tid, kv_len, THREADS):
        var probability = exp(scores[j] - row_max)
        scores[j] = probability
        row_sum += probability
    reduction[tid] = row_sum
    barrier()
    stride = THREADS // 2
    comptime for _ in range(8):
        if tid < stride:
            reduction[tid] += reduction[tid + stride]
        barrier()
        stride //= 2
    if tid == 0:
        broadcast[1] = reduction[0]
    barrier()
    var inv_sum = 1.0 / broadcast[1]

    # GPT-2 has D=64 on AMD wave64. Split the KV reduction across all four
    # wavefronts instead of leaving three quarters of the block idle during
    # the bandwidth-heavy V pass, then combine four partial output vectors.
    var lane = tid % 64
    var wave = tid // 64
    var acc = Float32(0)
    if lane < head_dim:
        for j in range(wave, kv_len, 4):
            acc += (
                scores[j]
                * value.ptr[kv_base + j * head_dim + lane].cast[DType.float32]()
            )
    reduction[tid] = acc
    barrier()
    if wave == 0 and lane < head_dim:
        acc = (
            reduction[lane]
            + reduction[64 + lane]
            + reduction[128 + lane]
            + reduction[192 + lane]
        )
        output.ptr[out_base + lane] = (acc * inv_sum).cast[dtype]()


@compiler.register("gpt2_decode_attention")
struct GPT2DecodeAttention:
    @staticmethod
    def execute[
        dtype: DType,
        rank: Int,
        //,
        target: StaticString,
    ](
        output: OutputTensor[dtype=dtype, rank=rank, ...],
        query: InputTensor[dtype=dtype, rank=rank, ...],
        key: InputTensor[dtype=dtype, rank=rank, ...],
        value: InputTensor[dtype=dtype, rank=rank, ...],
        mask: InputTensor[dtype=dtype, rank=rank, ...],
        ctx: DeviceContext,
    ) raises:
        comptime assert rank == 4, "attention inputs must have rank 4"
        comptime assert target == "gpu", "decode attention is GPU-only"

        var shape = query.shape()
        var key_shape = key.shape()
        var mask_shape = mask.shape()
        var batch = shape[0]
        var heads = shape[1]
        var q_len = shape[2]
        var head_dim = shape[3]
        var kv_len = key_shape[2]
        if q_len != 1 or head_dim > MAX_HEAD_DIM or kv_len > MAX_KV:
            raise Error("unsupported decode attention shape")

        var o = output.to_layout_tensor()
        var q = query.to_layout_tensor()
        var k = key.to_layout_tensor()
        var v = value.to_layout_tensor()
        var m = mask.to_layout_tensor()
        comptime kernel = _gpt2_decode_attention_kernel[
            dtype, o.layout, q.layout, k.layout, v.layout, m.layout
        ]
        ctx.enqueue_function[kernel](
            o,
            q,
            k,
            v,
            m,
            Int64(kv_len),
            Int64(head_dim),
            Int64(heads),
            Int64(mask_shape[0]),
            grid_dim=batch * heads,
            block_dim=THREADS,
        )
