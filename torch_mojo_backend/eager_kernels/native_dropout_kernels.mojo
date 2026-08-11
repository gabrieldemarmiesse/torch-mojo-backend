"""Production pure-Mojo kernels for eager FP32 ``aten::native_dropout`` and
``aten::native_dropout_backward`` (training mode, contiguous).

Ported unchanged from the validated showcase candidate
``candidate_native_dropout.mojo`` (full correctness matrix + H100 profiling).

RNG design
==========
Counter-based Philox4x32-10 (Salmon et al., SC'11) — the same stateless family
CUDA/PyTorch use for dropout.  Mapping from ``(seed, base_offset, index i)``:

- Key ``(k0, k1) = (seed[31:0], seed[63:32])``: every seed bit, including bit
  63, keys the stream.
- Element ``i`` belongs to Philox block ``g = i // 4`` with lane ``i % 4``.
  Its 128-bit counter is ``(c0, c1, c2, c3) = ((base_offset + g)[31:0],
  (base_offset + g)[63:32], 0, 0)``, so bumping ``base_offset`` by one shifts
  the logical stream by exactly four lanes and counter bit 63 lands in the top
  bit of ``c1``.  The mapping is independent of launch geometry, vector width,
  and tail handling.
- One Philox4x32-10 evaluation yields four independent 32-bit words; word
  ``i % 4`` supplies element ``i``.

Threshold convention
====================
The keep probability is formed exactly as the CUDA FP32 kernel does:
``keep_f32 = Float32(1.0 - p)`` with the subtraction in Float64 followed by a
single narrowing.  A lane is kept iff ``u32 < floor(keep_f32 * 2^32)`` using
all 32 random bits, so P(keep) equals ``keep_f32`` up to 2^-32 quantization
and ``p == 0`` maps to threshold ``2^32`` (always keep).  On device the
comparison is evaluated branch-free as ``(u64(u32) - threshold) >> 63`` (both
operands are below 2^63, so the borrow bit is exactly ``u32 < threshold``).
The ``p == 1`` endpoint is decided on the exact Float64 argument and takes
PyTorch's zeros_like path; drop probabilities that merely round to 1 in
Float32 stay stochastic.

Arithmetic
==========
Forward (``0 <= p < 1``): ``output[i] = (input[i] * Float32(mask[i])) *
(Float32(1) / keep_f32)`` — plain multiplication so dropped -0/NaN/Inf follow
IEEE semantics.  Backward: ``grad_input[i] = (grad_output[i] *
Float32(mask[i])) * Float32(scale)`` for every element, no branches to +0.

Vector dispatch is a host-side runtime decision: the 16-byte float4 path is
used only when every participating FP32 pointer is 16-byte aligned and the
mask pointer is 4-byte aligned; otherwise a scalar generic kernel with the
identical index->lane mapping runs.
"""

from std.gpu import block_idx, thread_idx
from max.gpu.host import DeviceContext
from std.math import ceildiv


comptime _BLOCK = 256

comptime _PHILOX_ROUNDS = 10
comptime _PHILOX_M0 = UInt32(0xD2511F53)
comptime _PHILOX_M1 = UInt32(0xCD9E8D57)
comptime _PHILOX_W0 = UInt32(0x9E3779B9)
comptime _PHILOX_W1 = UInt32(0xBB67AE85)


@always_inline
def _philox4x32_10(counter: UInt64, seed: UInt64) -> SIMD[DType.uint32, 4]:
    var c0 = (counter & 0xFFFF_FFFF).cast[DType.uint32]()
    var c1 = (counter >> 32).cast[DType.uint32]()
    var c2 = UInt32(0)
    var c3 = UInt32(0)
    var k0 = (seed & 0xFFFF_FFFF).cast[DType.uint32]()
    var k1 = (seed >> 32).cast[DType.uint32]()

    comptime for _round in range(_PHILOX_ROUNDS):
        var prod0 = _PHILOX_M0.cast[DType.uint64]() * c0.cast[DType.uint64]()
        var prod1 = _PHILOX_M1.cast[DType.uint64]() * c2.cast[DType.uint64]()
        var hi0 = (prod0 >> 32).cast[DType.uint32]()
        var lo0 = (prod0 & 0xFFFF_FFFF).cast[DType.uint32]()
        var hi1 = (prod1 >> 32).cast[DType.uint32]()
        var lo1 = (prod1 & 0xFFFF_FFFF).cast[DType.uint32]()
        var next0 = hi1 ^ c1 ^ k0
        var next2 = hi0 ^ c3 ^ k1
        c0 = next0
        c1 = lo1
        c2 = next2
        c3 = lo0
        k0 += _PHILOX_W0
        k1 += _PHILOX_W1
    return SIMD[DType.uint32, 4](c0, c1, c2, c3)


@__name("native_dropout_forward_philox_vec4")
def _forward_vec4(
    output: UnsafePointer[Scalar[DType.float32], MutAnyOrigin],
    mask: UnsafePointer[Scalar[DType.bool], MutAnyOrigin],
    input: UnsafePointer[Scalar[DType.float32], MutAnyOrigin],
    elements_arg: Int64,
    seed: UInt64,
    base_offset: UInt64,
    threshold: UInt64,
    scale: Float32,
):
    # Int is not device-passable (host/device width mismatch); scalars cross
    # the launch ABI as Int64 and index math stays in Int.
    var elements = Int(elements_arg)
    var group = Int(block_idx.x) * _BLOCK + Int(thread_idx.x)
    var base = group * 4
    if base >= elements:
        return
    var rnd = _philox4x32_10(base_offset + UInt64(group), seed)
    if base + 4 <= elements:
        var keep_bits = (
            rnd.cast[DType.uint64]() - SIMD[DType.uint64, 4](threshold)
        ) >> 63
        var x = input.load[width=4, alignment=16](base)
        var result = x * keep_bits.cast[DType.float32]() * scale
        output.store[alignment=16](base, result)
        mask.bitcast[Scalar[DType.uint8]]().store[alignment=4](
            base, keep_bits.cast[DType.uint8]()
        )
    else:
        comptime for lane in range(4):
            var idx = base + lane
            if idx < elements:
                var keep_bit = (
                    rnd[lane].cast[DType.uint64]() - threshold
                ) >> 63
                output[idx] = (
                    input[idx] * keep_bit.cast[DType.float32]() * scale
                )
                mask[idx] = Scalar[DType.bool](keep_bit != 0)


@__name("native_dropout_forward_philox_generic")
def _forward_generic(
    output: UnsafePointer[Scalar[DType.float32], MutAnyOrigin],
    mask: UnsafePointer[Scalar[DType.bool], MutAnyOrigin],
    input: UnsafePointer[Scalar[DType.float32], MutAnyOrigin],
    elements_arg: Int64,
    seed: UInt64,
    base_offset: UInt64,
    threshold: UInt64,
    scale: Float32,
):
    # Int is not device-passable (host/device width mismatch); scalars cross
    # the launch ABI as Int64 and index math stays in Int.
    var elements = Int(elements_arg)
    var group = Int(block_idx.x) * _BLOCK + Int(thread_idx.x)
    var base = group * 4
    if base >= elements:
        return
    var rnd = _philox4x32_10(base_offset + UInt64(group), seed)

    comptime for lane in range(4):
        var idx = base + lane
        if idx < elements:
            var keep_bit = (rnd[lane].cast[DType.uint64]() - threshold) >> 63
            output[idx] = input[idx] * keep_bit.cast[DType.float32]() * scale
            mask[idx] = Scalar[DType.bool](keep_bit != 0)


@__name("native_dropout_forward_zero_fill")
def _forward_zero_fill(
    output: UnsafePointer[Scalar[DType.float32], MutAnyOrigin],
    mask: UnsafePointer[Scalar[DType.bool], MutAnyOrigin],
    elements_arg: Int64,
):
    # Int is not device-passable (host/device width mismatch); scalars cross
    # the launch ABI as Int64 and index math stays in Int.
    var elements = Int(elements_arg)
    var base = (Int(block_idx.x) * _BLOCK + Int(thread_idx.x)) * 4

    comptime for lane in range(4):
        var idx = base + lane
        if idx < elements:
            output[idx] = Float32(0.0)
            mask[idx] = Scalar[DType.bool](False)


@__name("native_dropout_backward_vec4")
def _backward_vec4(
    grad_input: UnsafePointer[Scalar[DType.float32], MutAnyOrigin],
    grad_output: UnsafePointer[Scalar[DType.float32], MutAnyOrigin],
    mask: UnsafePointer[Scalar[DType.bool], MutAnyOrigin],
    elements_arg: Int64,
    scale: Float32,
):
    # Int is not device-passable (host/device width mismatch); scalars cross
    # the launch ABI as Int64 and index math stays in Int.
    var elements = Int(elements_arg)
    var base = (Int(block_idx.x) * _BLOCK + Int(thread_idx.x)) * 4
    if base >= elements:
        return
    if base + 4 <= elements:
        var g = grad_output.load[width=4, alignment=16](base)
        var mask_bytes = mask.bitcast[Scalar[DType.uint8]]().load[
            width=4, alignment=4
        ](base)
        # Bool storage is 0/1 by contract, so the byte cast is Float32(mask).
        var m = mask_bytes.cast[DType.float32]()
        grad_input.store[alignment=16](base, g * m * scale)
    else:
        comptime for lane in range(4):
            var idx = base + lane
            if idx < elements:
                grad_input[idx] = (
                    grad_output[idx] * mask[idx].cast[DType.float32]() * scale
                )


@__name("native_dropout_backward_generic")
def _backward_generic(
    grad_input: UnsafePointer[Scalar[DType.float32], MutAnyOrigin],
    grad_output: UnsafePointer[Scalar[DType.float32], MutAnyOrigin],
    mask: UnsafePointer[Scalar[DType.bool], MutAnyOrigin],
    elements_arg: Int64,
    scale: Float32,
):
    # Int is not device-passable (host/device width mismatch); scalars cross
    # the launch ABI as Int64 and index math stays in Int.
    var elements = Int(elements_arg)
    var base = (Int(block_idx.x) * _BLOCK + Int(thread_idx.x)) * 4
    if base >= elements:
        return

    comptime for lane in range(4):
        var idx = base + lane
        if idx < elements:
            grad_input[idx] = (
                grad_output[idx] * mask[idx].cast[DType.float32]() * scale
            )


@always_inline
def _is_aligned(address: Int, alignment: Int) -> Bool:
    return address % alignment == 0


def enqueue_native_dropout_f32(
    output: UnsafePointer[Scalar[DType.float32], MutAnyOrigin],
    mask: UnsafePointer[Scalar[DType.bool], MutAnyOrigin],
    input: UnsafePointer[Scalar[DType.float32], MutAnyOrigin],
    elements: Int,
    p: Float64,
    seed: UInt64,
    base_offset: UInt64,
    ctx: DeviceContext,
) raises:
    # NaN fails both comparisons, so this single conjunction rejects NaN and
    # every value outside [0, 1] before any launch.
    if not (p >= 0.0 and p <= 1.0):
        raise Error(
            "native_dropout_kernels: invalid dropout probability; p must"
            " satisfy 0 <= p <= 1 (NaN rejected)"
        )
    if elements <= 0:
        return

    var groups = ceildiv(elements, 4)
    var grid = ceildiv(groups, _BLOCK)

    if p == 1.0:
        # Exact Float64 endpoint only: PyTorch's zeros_like shortcut with an
        # all-false mask and no divide-by-zero.
        ctx.enqueue_function[_forward_zero_fill](
            output,
            mask,
            Int64(elements),
            grid_dim=(grid,),
            block_dim=(_BLOCK,),
        )
        return

    var keep_f32 = Float32(1.0 - p)
    var scale = Float32(1.0) / keep_f32
    var threshold = (Float64(keep_f32) * 4294967296.0).cast[DType.uint64]()

    if (
        _is_aligned(Int(output), 16)
        and _is_aligned(Int(input), 16)
        and _is_aligned(Int(mask), 4)
    ):
        ctx.enqueue_function[_forward_vec4](
            output,
            mask,
            input,
            Int64(elements),
            seed,
            base_offset,
            threshold,
            scale,
            grid_dim=(grid,),
            block_dim=(_BLOCK,),
        )
    else:
        ctx.enqueue_function[_forward_generic](
            output,
            mask,
            input,
            Int64(elements),
            seed,
            base_offset,
            threshold,
            scale,
            grid_dim=(grid,),
            block_dim=(_BLOCK,),
        )


def enqueue_native_dropout_backward_f32(
    grad_input: UnsafePointer[Scalar[DType.float32], MutAnyOrigin],
    grad_output: UnsafePointer[Scalar[DType.float32], MutAnyOrigin],
    mask: UnsafePointer[Scalar[DType.bool], MutAnyOrigin],
    elements: Int,
    scale: Float64,
    ctx: DeviceContext,
) raises:
    if elements <= 0:
        return

    var groups = ceildiv(elements, 4)
    var grid = ceildiv(groups, _BLOCK)
    var scale_f32 = Float32(scale)

    if (
        _is_aligned(Int(grad_input), 16)
        and _is_aligned(Int(grad_output), 16)
        and _is_aligned(Int(mask), 4)
    ):
        ctx.enqueue_function[_backward_vec4](
            grad_input,
            grad_output,
            mask,
            Int64(elements),
            scale_f32,
            grid_dim=(grid,),
            block_dim=(_BLOCK,),
        )
    else:
        ctx.enqueue_function[_backward_generic](
            grad_input,
            grad_output,
            mask,
            Int64(elements),
            scale_f32,
            grid_dim=(grid,),
            block_dim=(_BLOCK,),
        )
