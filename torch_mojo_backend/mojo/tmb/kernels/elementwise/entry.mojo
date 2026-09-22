# ===----------------------------------------------------------------------=== #
# Fast eager-mode elementwise kernels for mojo_device.
#
# This module is built on demand by `eager_kernels.MojoExtensionLoader`, one
# `mojo build --emit shared-lib` per specialization: the `OP` and `DTYPE_*`
# compiler defines pick exactly one registration below (see variant_gates.mojo)
# and every .so exposes that single entry point under the constant name
# `call`. Dtype selection is therefore *compile time*, not a runtime switch
# over every dtype; a different dtype tuple is a different .so.
#
# The design mirrors `max._interpreter_ops.elementwise_binary_ops` (the MO
# interpreter's own op bindings): each Python-visible function receives raw
# tensor metadata (`TensorSpec` handles, or plain ints with the storage offset
# already applied) plus the device's DeviceContext pointer — there are no
# `max.driver.Buffer` objects and no attribute access at all — and enqueues the
# kernel on MAX's own device context, so ordering with regular MAX driver
# operations (copies, other kernels) comes for free.
#
# Every kernel here works on *contiguous* buffers with fully dynamic sizes:
# shapes and strides are runtime arguments and never enter the specialization
# key, so one compiled variant serves every shape with zero recompilation.
# ===----------------------------------------------------------------------=== #

from tmb.graph.unary_math import elementwise_predicate, elementwise_unary

from std.os import abort
from std.gpu import block_dim, block_idx, grid_dim, thread_idx
from max.gpu.host import DeviceContext
from std.math import ceildiv, pow
from std.sys.info import (
    has_accelerator,
    has_apple_gpu_accelerator,
    has_nvidia_gpu_accelerator,
    simd_width_of,
    size_of,
)
from std.utils.index import IndexList
from std.utils.coord import Coord

from max.algorithm import elementwise

from tmb.kernels.common.op_utils import (
    Arg,
    Argv,
    FLOAT_DTYPES,
    GS_THREADS,
    MAX_RANK,
    _check_into,
    _enqueue_cached,
    _fill_bits,
    _fill_bits_dtype,
    _fill_contig,
    _flat_vec_unary,
    _gs_blocks,
    _l2_wave_blocks,
    _make_ptr,
    _raw_ctx,
    _raw_dtype_int,
    _raw_f64,
    _raw_int,
    _raw_tuple_int,
    _raw_tuple_len,
    _spec_dispatcher2,
    _spec_dispatcher3,
    _spec_ptr,
)

from tmb.kernels.common.variant_gates import (
    ErrBuf,
    NO_OP_COMPILED,
    _dtype_arg_abi_on,
    _dtype_arg_on,
    _dtype_out_on,
    _dtype_supported,
    _op_on,
    _tmb_entry_error,
)
from tmb.graph.math_utils import ieee_sqrt
from std.sys.info import _has_sm_9x


# ---------------------------------------------------------------------------
# Raw-pointer calling convention: every Python-visible kernel below receives
# tensor operands as a single int (the `._ptr` address, storage offset
# already applied), unpacked with `_raw_int` and turned into a typed
# pointer with `_make_ptr[dt]`; numel and dtype are explicit int args
# (`_raw_int` / `_raw_dtype_int`); `ctx_ptr` (int) is always last
# (`_raw_ctx`). The dispatchers register as METH_FASTCALL functions
# (`def_py_c_function`), skipping the owning PythonObject wrappers of the
# `def_function` path entirely.
# ---------------------------------------------------------------------------


# ---------------------------------------------------------------------------
# Binary elementwise kernels
# ---------------------------------------------------------------------------

comptime OP_ADD = 0
comptime OP_SUB = 1
comptime OP_MUL = 2
comptime OP_DIV = 3
comptime OP_MAX = 4
comptime OP_MIN = 5


def _bin_contig_kernel4[
    dtype: DType, op_code: Int
](
    out_ptr: Pointer[Scalar[dtype], MutAnyOrigin],
    lhs_ptr: Pointer[Scalar[dtype], ImmutAnyOrigin],
    rhs_ptr: Pointer[Scalar[dtype], ImmutAnyOrigin],
    n4_arg: Int64,
):
    # Int is not device-passable (host/device width mismatch); scalars cross
    # the launch ABI as Int64 and index math stays in Int.
    var n4 = Int(n4_arg)
    comptime vec_align = 4 * size_of[dtype]()
    var c = Int(block_idx.x) * Int(block_dim.x) + Int(thread_idx.x)
    var gstride = Int(grid_dim.x) * Int(block_dim.x)
    while c < n4:
        var i = c * 4
        var a = lhs_ptr.unsafe_load[width=4, alignment=vec_align](i)
        var b = rhs_ptr.unsafe_load[width=4, alignment=vec_align](i)
        comptime if op_code == OP_ADD:
            out_ptr.unsafe_store[width=4, alignment=vec_align](i, a + b)
        comptime if op_code == OP_SUB:
            out_ptr.unsafe_store[width=4, alignment=vec_align](i, a - b)
        comptime if op_code == OP_MUL:
            out_ptr.unsafe_store[width=4, alignment=vec_align](i, a * b)
        comptime if op_code == OP_DIV:
            comptime if dtype.is_floating_point():
                out_ptr.unsafe_store[width=4, alignment=vec_align](i, a / b)
        comptime if op_code == OP_MAX:
            out_ptr.unsafe_store[width=4, alignment=vec_align](i, max(a, b))
        comptime if op_code == OP_MIN:
            out_ptr.unsafe_store[width=4, alignment=vec_align](i, min(a, b))
        c += gstride


def _bin_contig_kernel[
    dtype: DType, op_code: Int
](
    out_ptr: Pointer[Scalar[dtype], MutAnyOrigin],
    lhs_ptr: Pointer[Scalar[dtype], ImmutAnyOrigin],
    rhs_ptr: Pointer[Scalar[dtype], ImmutAnyOrigin],
    size_arg: Int64,
):
    # Int is not device-passable (host/device width mismatch); scalars cross
    # the launch ABI as Int64 and index math stays in Int.
    var size = Int(size_arg)
    var i = Int(block_idx.x) * Int(block_dim.x) + Int(thread_idx.x)
    var gstride = Int(grid_dim.x) * Int(block_dim.x)
    while i < size:
        var a = lhs_ptr[unsafe_offset=i]
        var b = rhs_ptr[unsafe_offset=i]
        comptime if op_code == OP_ADD:
            out_ptr[unsafe_offset=i] = a + b
        comptime if op_code == OP_SUB:
            out_ptr[unsafe_offset=i] = a - b
        comptime if op_code == OP_MUL:
            out_ptr[unsafe_offset=i] = a * b
        comptime if op_code == OP_DIV:
            comptime if dtype.is_floating_point():
                out_ptr[unsafe_offset=i] = a / b
        comptime if op_code == OP_MAX:
            out_ptr[unsafe_offset=i] = max(a, b)
        comptime if op_code == OP_MIN:
            out_ptr[unsafe_offset=i] = min(a, b)
        i += gstride


@always_inline
def _bin_elementwise[
    dtype: DType, op_code: Int
](
    out_ptr: Pointer[Scalar[dtype], MutUntrackedOrigin],
    lhs_ptr: Pointer[Scalar[dtype], MutUntrackedOrigin],
    rhs_ptr: Pointer[Scalar[dtype], MutUntrackedOrigin],
    size: Int,
    ctx: DeviceContext,
) raises:
    """out = op(lhs, rhs) over `size` contiguous elements."""

    comptime if op_code == OP_DIV and not dtype.is_floating_point():
        raise Error("integer/bool div is not supported in the fast path")
    else:
        comptime if has_accelerator():
            comptime if dtype != DType.float64:
                # The 4-wide body loads and stores at `4 * itemsize`, so
                # it needs the runtime addresses aligned, not just a numel
                # divisible by 4: a contiguous operand at an odd storage
                # offset (any offset view) would fault the context with
                # CUDA_ERROR_MISALIGNED_ADDRESS. The unary twin gates the
                # same way; unaligned operands take the scalar body.
                comptime vec_align = 4 * size_of[dtype]()
                if (
                    size % 4 == 0
                    and (Int(out_ptr) | Int(lhs_ptr) | Int(rhs_ptr)) % vec_align
                    == 0
                ):
                    var n4 = size // 4
                    _enqueue_cached[_bin_contig_kernel4[dtype, op_code]](
                        ctx,
                        String(t"ew_bin4_{op_code}_{dtype}"),
                        _gs_blocks(n4),
                        1,
                        1,
                        GS_THREADS,
                        out_ptr.as_unsafe_any_origin(),
                        lhs_ptr.as_unsafe_any_origin().as_imm(),
                        rhs_ptr.as_unsafe_any_origin().as_imm(),
                        Int64(n4),
                    )
                    return
                _enqueue_cached[_bin_contig_kernel[dtype, op_code]](
                    ctx,
                    String(t"ew_bin_{op_code}_{dtype}"),
                    _gs_blocks(size),
                    1,
                    1,
                    GS_THREADS,
                    out_ptr.as_unsafe_any_origin(),
                    lhs_ptr.as_unsafe_any_origin().as_imm(),
                    rhs_ptr.as_unsafe_any_origin().as_imm(),
                    Int64(size),
                )
            else:
                raise Error("float64 is not supported on GPU")
        else:
            raise Error("no GPU accelerator available at compile time")


def _bin_go[
    op_code: Int
](
    out_ptr: Arg,
    lhs_ptr: Arg,
    rhs_ptr: Arg,
    numel: Arg,
    dtype_val: Arg,
    ctx_ptr: Arg,
) raises:
    var out_addr = _raw_int(out_ptr)
    var lhs_addr = _raw_int(lhs_ptr)
    var rhs_addr = _raw_int(rhs_ptr)
    var size = _raw_int(numel)
    var dtype = _raw_dtype_int(dtype_val)
    var ctx = _raw_ctx(ctx_ptr)

    var handled = False
    comptime for dt in [
        DType.float32,
        DType.float16,
        DType.bfloat16,
        DType.float64,
        DType.int8,
        DType.int16,
        DType.int32,
        DType.int64,
        DType.uint8,
    ]:
        comptime if _dtype_arg_on[0, dt]():
            if dtype == dt:
                _bin_elementwise[dt, op_code](
                    _make_ptr[dt](out_addr),
                    _make_ptr[dt](lhs_addr),
                    _make_ptr[dt](rhs_addr),
                    size,
                    ctx,
                )
                handled = True
    if not handled:
        # A miss means Python selected the wrong immutable specialization.
        raise Error("unsupported dtype for fast binary elementwise op: ", dtype)


# ---------------------------------------------------------------------------
# Unary elementwise kernels
#
# Opcodes fall in three buckets:
#   * RELU / ABS / NEG / SIGN work on integer *and* float dtypes and compute
#     directly in the tensor dtype (no float round-trip).
#   * every other opcode is float-only (shared `unary_math._float_unary`): half-precision
#     inputs are promoted to float32, computed, and cast back — matching
#     torch's numerics and keeping the polynomial math accurate.
# Two of the ops deserve a note: `tan` and `asinh` cannot call the std.math
# primitive of the same name, because those lower to libm (`_call_libm`) which
# `comptime assert`s CPU-only and would refuse to compile for the GPU target.
# `asinh` is composed from log/sqrt in shared math; `tan` routes through
# the shared `custom_tan`, which selects math per target and dtype.
# ---------------------------------------------------------------------------

comptime UOP_RELU = 0
comptime UOP_EXP = 1
comptime UOP_TANH = 2
comptime UOP_ABS = 3
comptime UOP_NEG = 4
comptime UOP_SIGN = 5
comptime UOP_CEIL = 6
comptime UOP_FLOOR = 7
comptime UOP_ACOS = 8
comptime UOP_ASINH = 9
comptime UOP_ATANH = 10
comptime UOP_COS = 11
comptime UOP_COSH = 12
comptime UOP_ERF = 13
comptime UOP_LOG = 14
comptime UOP_LOG1P = 15
comptime UOP_RECIPROCAL = 16
comptime UOP_RSQRT = 17
comptime UOP_SIGMOID = 18
comptime UOP_SILU = 19
comptime UOP_SIN = 20
comptime UOP_SINH = 21
comptime UOP_SQRT = 22
comptime UOP_TAN = 23
comptime UOP_GELU_NONE = 24
comptime UOP_GELU_TANH = 25
comptime UOP_LOG2 = 26


def _unary_contig_kernel[
    dtype: DType, op_code: Int
](
    out_ptr: Pointer[Scalar[dtype], MutAnyOrigin],
    in_ptr: Pointer[Scalar[dtype], ImmutAnyOrigin],
    size_arg: Int64,
):
    # Int is not device-passable (host/device width mismatch); scalars cross
    # the launch ABI as Int64 and index math stays in Int.
    var size = Int(size_arg)
    var i = Int(block_idx.x) * Int(block_dim.x) + Int(thread_idx.x)
    var gstride = Int(grid_dim.x) * Int(block_dim.x)
    while i < size:
        var a = in_ptr[unsafe_offset=i]
        out_ptr[unsafe_offset=i] = _unary_apply[dtype, 1, op_code](a)
        i += gstride


@always_inline
def _unary_apply[
    dtype: DType, width: Int, op_code: Int
](a: SIMD[dtype, width]) -> SIMD[dtype, width]:
    """Dispatch a native opcode to the shared graph/native SIMD expression."""
    comptime if op_code == UOP_RELU:
        return elementwise_unary["relu"](a)
    elif op_code == UOP_EXP:
        return elementwise_unary["exp"](a)
    elif op_code == UOP_TANH:
        return elementwise_unary["tanh"](a)
    elif op_code == UOP_ABS:
        return elementwise_unary["abs"](a)
    elif op_code == UOP_NEG:
        return elementwise_unary["neg"](a)
    elif op_code == UOP_SIGN:
        return elementwise_unary["sign"](a)
    elif op_code == UOP_CEIL:
        return elementwise_unary["ceil"](a)
    elif op_code == UOP_FLOOR:
        return elementwise_unary["floor"](a)
    elif op_code == UOP_ACOS:
        return elementwise_unary["acos"](a)
    elif op_code == UOP_ASINH:
        return elementwise_unary["asinh"](a)
    elif op_code == UOP_ATANH:
        return elementwise_unary["atanh"](a)
    elif op_code == UOP_COS:
        return elementwise_unary["cos"](a)
    elif op_code == UOP_COSH:
        return elementwise_unary["cosh"](a)
    elif op_code == UOP_ERF:
        return elementwise_unary["erf"](a)
    elif op_code == UOP_LOG:
        return elementwise_unary["log"](a)
    elif op_code == UOP_LOG1P:
        return elementwise_unary["log1p"](a)
    elif op_code == UOP_RECIPROCAL:
        return elementwise_unary["reciprocal"](a)
    elif op_code == UOP_RSQRT:
        return elementwise_unary["rsqrt"](a)
    elif op_code == UOP_SIGMOID:
        return elementwise_unary["sigmoid"](a)
    elif op_code == UOP_SILU:
        return elementwise_unary["silu"](a)
    elif op_code == UOP_SIN:
        return elementwise_unary["sin"](a)
    elif op_code == UOP_SINH:
        return elementwise_unary["sinh"](a)
    elif op_code == UOP_SQRT:
        return elementwise_unary["sqrt"](a)
    elif op_code == UOP_TAN:
        return elementwise_unary["tan"](a)
    elif op_code == UOP_GELU_NONE:
        return elementwise_unary["gelu_none"](a)
    elif op_code == UOP_GELU_TANH:
        return elementwise_unary["gelu_tanh"](a)
    elif op_code == UOP_LOG2:
        return elementwise_unary["log2"](a)
    else:
        comptime assert False, "unknown unary opcode"


def _unary_contig_kernel4[
    dtype: DType, op_code: Int
](
    out_ptr: Pointer[Scalar[dtype], MutAnyOrigin],
    in_ptr: Pointer[Scalar[dtype], ImmutAnyOrigin],
    size_arg: Int64,
    vec_count_arg: Int64,
):
    # Int is not device-passable (host/device width mismatch); scalars cross
    # the launch ABI as Int64 and index math stays in Int.
    var size = Int(size_arg)
    var vec_count = Int(vec_count_arg)
    # Vector body (4-element chunks when the host proved alignment,
    # vec_count == 0 otherwise) plus a grid-stride scalar loop that covers
    # the tail — or, with vec_count == 0, the entire range.  Each thread
    # owns 4 consecutive chunks: a sequential 4*vec_align-byte stream with
    # four independent loads in flight, which this GPU needs to stream at
    # full rate.
    comptime vec_align = 4 * size_of[dtype]()
    var gid = Int(block_idx.x) * Int(block_dim.x) + Int(thread_idx.x)
    var gstride = Int(grid_dim.x) * Int(block_dim.x)
    var groups = vec_count // 4
    var g = gid
    while g < groups:
        var b = g * 4
        var a0 = in_ptr.unsafe_load[width=4, alignment=vec_align](b * 4)
        var a1 = in_ptr.unsafe_load[width=4, alignment=vec_align]((b + 1) * 4)
        var a2 = in_ptr.unsafe_load[width=4, alignment=vec_align]((b + 2) * 4)
        var a3 = in_ptr.unsafe_load[width=4, alignment=vec_align]((b + 3) * 4)
        out_ptr.unsafe_store[width=4, alignment=vec_align](
            b * 4, _unary_apply[dtype, 4, op_code](a0)
        )
        out_ptr.unsafe_store[width=4, alignment=vec_align](
            (b + 1) * 4, _unary_apply[dtype, 4, op_code](a1)
        )
        out_ptr.unsafe_store[width=4, alignment=vec_align](
            (b + 2) * 4, _unary_apply[dtype, 4, op_code](a2)
        )
        out_ptr.unsafe_store[width=4, alignment=vec_align](
            (b + 3) * 4, _unary_apply[dtype, 4, op_code](a3)
        )
        g += gstride
    var c = groups * 4 + gid
    if c < vec_count:
        var a = in_ptr.unsafe_load[width=4, alignment=vec_align](c * 4)
        out_ptr.unsafe_store[width=4, alignment=vec_align](
            c * 4, _unary_apply[dtype, 4, op_code](a)
        )
    var i = vec_count * 4 + gid
    while i < size:
        out_ptr[unsafe_offset=i] = _unary_apply[dtype, 1, op_code](
            in_ptr[unsafe_offset=i]
        )
        i += gstride


@__name("sqrt_contig_f32_v4_peel")
def _sqrt_peel_kernel(
    dst: Pointer[Float32, MutAnyOrigin],
    src: Pointer[Float32, ImmutAnyOrigin],
    size_arg: Int64,
    head_arg: Int64,
):
    var size = Int(size_arg)
    var head = Int(head_arg)
    var vectors = (size - head) // 4
    var tid = Int(block_idx.x) * Int(block_dim.x) + Int(thread_idx.x)
    var vector = tid
    while vector < vectors:
        var index = head + vector * 4
        dst.unsafe_store[width=4, alignment=16](
            index, ieee_sqrt(src.unsafe_load[width=4, alignment=16](index))
        )
        vector += Int(grid_dim.x) * Int(block_dim.x)
    if tid < head:
        dst[unsafe_offset=tid] = ieee_sqrt(src[unsafe_offset=tid])
    var tail = head + vectors * 4
    if tid < size - tail:
        var index = tail + tid
        dst[unsafe_offset=index] = ieee_sqrt(src[unsafe_offset=index])


@always_inline
def _unary_elementwise[
    dtype: DType, op_code: Int
](
    out_ptr: Pointer[Scalar[dtype], MutUntrackedOrigin],
    in_ptr: Pointer[Scalar[dtype], MutUntrackedOrigin],
    size: Int,
    ctx: DeviceContext,
) raises:
    comptime is_direct = (
        op_code == UOP_RELU
        or op_code == UOP_ABS
        or op_code == UOP_NEG
        or op_code == UOP_SIGN
    )
    comptime if not is_direct and not dtype.is_floating_point():
        # Transcendentals / ceil / floor / gelu require a float dtype; the
        # Python side already gates on this, so this only ever fires as a
        # defensive guard (and keeps the float math out of int instantiations).
        raise Error("this unary op requires a floating point dtype")
    else:
        comptime if has_accelerator():
            comptime if dtype != DType.float64 or op_code == UOP_LOG2:
                # Public elementwise owns launch geometry on every GPU.
                # SIMD-4 needs BOTH pointers aligned; offset views retain the
                # existing scalar/vector fallback and its cached launch.
                @always_inline
                @parameter
                @__copy_capture(out_ptr, in_ptr)
                def gpu_func[width: Int, alignment: Int = 1](idx: Coord):
                    var i = Int(idx[0].value())
                    # Only the aligned branch requests width > 1. The
                    # launcher calls the same body at width 1 for tails.
                    comptime byte_alignment = (
                        min(16, width * size_of[dtype]()) if width
                        > 4 else width * size_of[dtype]()
                    )
                    var a = in_ptr.unsafe_load[
                        width=width, alignment=byte_alignment
                    ](i)
                    out_ptr.unsafe_store[width=width, alignment=byte_alignment](
                        i, _unary_apply[dtype, width, op_code](a)
                    )

                # Measured on H100: these wider public bodies avoid excess
                # waves for expensive half math and the float32 log1p body.
                # Keep SIMD4 for 8-byte-aligned half views and other GPUs.
                comptime wider_nvidia = has_nvidia_gpu_accelerator() and (
                    (
                        (dtype == DType.float16 or dtype == DType.bfloat16)
                        and (
                            op_code == UOP_ACOS
                            or op_code == UOP_GELU_NONE
                            or op_code == UOP_GELU_TANH
                            or op_code == UOP_LOG2
                        )
                    )
                    or (
                        (
                            dtype == DType.float16
                            or dtype == DType.bfloat16
                            or dtype == DType.float32
                        )
                        and op_code == UOP_LOG1P
                    )
                )
                comptime preferred_width = (16 if op_code == UOP_ACOS else 8)
                comptime if wider_nvidia:
                    if (Int(out_ptr) | Int(in_ptr)) % 16 == 0:
                        elementwise[
                            gpu_func,
                            simd_width=preferred_width,
                            target="gpu",
                            _trace_description="modular_unary",
                        ](Coord(size), ctx)
                        return
                if (Int(out_ptr) | Int(in_ptr)) % (4 * size_of[dtype]()) == 0:
                    elementwise[
                        gpu_func,
                        simd_width=4,
                        target="gpu",
                        _trace_description="modular_unary",
                    ](Coord(size), ctx)
                    return
            comptime if (
                op_code == UOP_SQRT and dtype == DType.float32 and _has_sm_9x()
            ):
                # Measured on H100: one vector per thread with the shared
                # L2/HBM grid improves sqrt while retaining ieee_sqrt.
                if ctx.api() == "cuda":
                    if _flat_vec_unary[
                        dtype,
                        dtype,
                        _unary_apply[dtype, _, op_code],
                        "sqrt",
                    ](Int(out_ptr), Int(in_ptr), size, ctx):
                        return
                    if size > 0 and Int(out_ptr) % 16 == Int(in_ptr) % 16:
                        var head = min(
                            size, ((16 - Int(in_ptr) % 16) % 16) // 4
                        )
                        # Equal residues permit a common scalar prefix,
                        # making both vector bases 16-byte aligned.
                        _enqueue_cached[_sqrt_peel_kernel](
                            ctx,
                            "sqrt_contig_f32_v4_peel",
                            _l2_wave_blocks(
                                max(1, (size - head) // 4), size * 8, ctx
                            ),
                            1,
                            1,
                            GS_THREADS,
                            out_ptr.as_unsafe_any_origin(),
                            in_ptr.as_unsafe_any_origin().as_imm(),
                            Int64(size),
                            Int64(head),
                        )
                        return
            comptime if (
                op_code == UOP_LOG2
                and (dtype == DType.float32 or dtype == DType.bfloat16)
                and has_nvidia_gpu_accelerator()
            ):
                if _flat_vec_unary[
                    dtype,
                    dtype,
                    _unary_apply[dtype, _, op_code],
                    "log2",
                ](Int(out_ptr), Int(in_ptr), size, ctx):
                    return
            comptime if (
                op_code == UOP_LOG2 or (is_direct and dtype == DType.float64)
            ) and not has_apple_gpu_accelerator():
                # Preserve log2's upstream scalar fallback, including
                # float64; the existing unary ops keep their 4-wide route.
                # float64 abs/neg/sign/relu land here too: the ops accept the
                # dtype, and only Apple GPUs lack it.
                _enqueue_cached[_unary_contig_kernel[dtype, op_code]](
                    ctx,
                    String(t"ew_unary_{op_code}_{dtype}"),
                    _gs_blocks(size),
                    1,
                    1,
                    GS_THREADS,
                    out_ptr.as_unsafe_any_origin(),
                    in_ptr.as_unsafe_any_origin().as_imm(),
                    Int64(size),
                )
            elif dtype != DType.float64:
                # 4-wide vector body when both pointers are vector-
                # aligned; the scalar grid-stride tail in the same kernel
                # keeps arbitrary sizes and unproven alignment correct.
                # (Was Apple-only: on H100 the scalar kernel streamed a
                # bf16 gelu at 1.65 TB/s against cuDNN's 2.8.)
                comptime vec_align = 4 * size_of[dtype]()
                var aligned = (Int(out_ptr) | Int(in_ptr)) % vec_align == 0
                var vec_count = size // 4 if aligned else 0
                var span = max(vec_count // 4, 1) if vec_count > 0 else size
                _enqueue_cached[_unary_contig_kernel4[dtype, op_code]](
                    ctx,
                    String(t"ew_unary4_{op_code}_{dtype}"),
                    _gs_blocks(span),
                    1,
                    1,
                    GS_THREADS,
                    out_ptr.as_unsafe_any_origin(),
                    in_ptr.as_unsafe_any_origin().as_imm(),
                    Int64(size),
                    Int64(vec_count),
                )
            else:
                raise Error("float64 is not supported on GPU")
        else:
            raise Error("no GPU accelerator available at compile time")


comptime BUOP_ISNAN = 0
comptime BUOP_LOGICAL_NOT = 1


@always_inline
def _unary_bool_vec[
    dtype: DType, op_code: Int, w: Int
](a: SIMD[dtype, w]) -> SIMD[DType.uint8, w]:
    """The bool-output unary ops at an arbitrary SIMD width.

    Module level, not a nested closure: the vectorized skeleton compiles
    this into a device function, and a capturing closure would capture by
    reference on GPU. The mask is cast to uint8 (0/1) and stored through a
    uint8 pointer -- that IS torch's bool memory format, while storing
    SIMD[bool, w] would offer LLVM a packed i1 vector.
    """
    comptime if op_code == BUOP_ISNAN:
        # `numerics.isnan` is bit-based (llvm.is.fpclass), so it survives the
        # fast-math flags that would fold `a != a` to False; it also returns
        # all-False for integer dtypes.
        return elementwise_predicate["isnan"](a).cast[DType.uint8]()
    else:
        return elementwise_predicate["logical_not"](a).cast[DType.uint8]()


@always_inline
def _unary_bool[
    dtype: DType, op_code: Int
](
    out_ptr: Pointer[Scalar[DType.bool], MutUntrackedOrigin],
    in_ptr: Pointer[Scalar[dtype], MutUntrackedOrigin],
    size: Int,
    ctx: DeviceContext,
) raises:
    @always_inline
    @parameter
    @__copy_capture(out_ptr, in_ptr)
    def func[width: Int, alignment: Int = 1](idx: Coord):
        # Same body as the vectorized path above, through the same helper:
        # the 0/1 uint8 it returns is bit-identical to the bool stored here.
        var i = Int(idx[0].value())
        out_ptr.unsafe_store[width=width](
            i,
            _unary_bool_vec[dtype, op_code, width](
                in_ptr.unsafe_load[width=width](i)
            ).cast[DType.bool](),
        )

    comptime if has_accelerator():
        comptime if (dtype == DType.float64 and has_apple_gpu_accelerator()):
            raise Error("float64 is not supported on Apple GPU")
        else:

            @always_inline
            @parameter
            @__copy_capture(out_ptr, in_ptr)
            def gpu_bool[width: Int, alignment: Int = 1](idx: Coord):
                var i = Int(idx[0].value())
                var a = in_ptr.unsafe_load[
                    width=width, alignment=min(16, width * size_of[dtype]())
                ](i)
                out_ptr.unsafe_bitcast[UInt8]().unsafe_store[
                    width=width, alignment=min(16, width)
                ](i, _unary_bool_vec[dtype, op_code, width](a))

            # H100 measurements favor 16 input bytes for half predicates,
            # and 32 byte-sized inputs per thread for logical_not.
            comptime preferred_width = 32 if size_of[
                dtype
            ]() == 1 else 16 // size_of[dtype]()
            comptime if has_nvidia_gpu_accelerator() and preferred_width > 4:
                if (
                    Int(in_ptr) % 16 == 0
                    and Int(out_ptr) % min(16, preferred_width) == 0
                ):
                    elementwise[
                        gpu_bool, simd_width=preferred_width, target="gpu"
                    ](Coord(size), ctx)
                    return
            # Keep 64-bit inputs on the previous 16-byte/SIMD2 regime.
            comptime vector_width = min(4, 16 // size_of[dtype]())
            if (
                Int(in_ptr) % (vector_width * size_of[dtype]()) == 0
                and Int(out_ptr) % vector_width == 0
            ):
                elementwise[gpu_bool, simd_width=vector_width, target="gpu"](
                    Coord(size), ctx
                )
                return
            elementwise[func, simd_width=1, target="gpu"](Coord(size), ctx)
    else:
        raise Error("no GPU accelerator available at compile time")


comptime SOP_ADD = 0
comptime SOP_MUL = 1
comptime SOP_POW = 2


@always_inline
def _scalar_elementwise[
    dtype: DType, op_code: Int
](
    out_ptr: Pointer[Scalar[dtype], MutUntrackedOrigin],
    in_ptr: Pointer[Scalar[dtype], MutUntrackedOrigin],
    scalar: Float32,
    size: Int,
    ctx: DeviceContext,
) raises:
    comptime if not dtype.is_floating_point():
        raise Error("scalar elementwise ops require a floating point dtype")
    else:

        @always_inline
        @parameter
        @__copy_capture(out_ptr, in_ptr, scalar)
        def body[width: Int, al: Int](i: Int):
            var a = in_ptr.unsafe_load[width=width, alignment=al](i).cast[
                DType.float32
            ]()
            var s = SIMD[DType.float32, width](scalar)
            comptime if op_code == SOP_ADD:
                out_ptr.unsafe_store[width=width, alignment=al](
                    i, (a + s).cast[dtype]()
                )
            comptime if op_code == SOP_MUL:
                out_ptr.unsafe_store[width=width, alignment=al](
                    i, (a * s).cast[dtype]()
                )
            comptime if op_code == SOP_POW:
                # float32 through float64, as in logic' BOP_POW (the
                # float32 exp(y * log x) is up to 16 ulp off on H100)
                comptime if dtype == DType.float32 and not has_apple_gpu_accelerator():
                    out_ptr.unsafe_store[width=width, alignment=al](
                        i,
                        pow(
                            a.cast[DType.float64](), s.cast[DType.float64]()
                        ).cast[dtype](),
                    )
                else:
                    out_ptr.unsafe_store[width=width, alignment=al](
                        i, pow(a, s).cast[dtype]()
                    )

        # Element alignment only: the CPU lanes and the GPU scalar lanes may
        # start at any element (a bucket view starts wherever the previous
        # parameter ended).
        @always_inline
        @parameter
        def func[width: Int, alignment: Int = 1](idx: Coord):
            body[width, size_of[dtype]()](Int(idx[0].value()))

        # 16-byte vectors, launched only once both bases proved aligned.
        @always_inline
        @parameter
        def func_vec[width: Int, alignment: Int = 1](idx: Coord):
            body[width, 16](Int(idx[0].value()))

        comptime if has_accelerator():
            # 16-byte vectors when both bases allow them and there is no
            # tail (a bucket view starts wherever the previous parameter
            # ended): scalar lanes moved 1.5 TB/s on H100, vectors ~3.
            comptime vec = 16 // size_of[dtype]()
            if (
                Int(out_ptr) % 16 == 0
                and Int(in_ptr) % 16 == 0
                and size % vec == 0
            ):
                elementwise[func_vec, simd_width=vec, target="gpu"](
                    Coord(size), ctx
                )
            else:
                elementwise[func, simd_width=1, target="gpu"](Coord(size), ctx)
        else:
            raise Error("no GPU accelerator available at compile time")


comptime IOP_ADD = 0
comptime IOP_MUL = 1


@always_inline
def _int_scalar_elementwise[
    dtype: DType, op_code: Int
](
    out_ptr: Pointer[Scalar[dtype], MutUntrackedOrigin],
    in_ptr: Pointer[Scalar[dtype], MutUntrackedOrigin],
    scalar: Int,
    size: Int,
    ctx: DeviceContext,
) raises:
    @always_inline
    @parameter
    @__copy_capture(out_ptr, in_ptr, scalar)
    def func[width: Int, alignment: Int = 1](idx: Coord):
        var i = Int(idx[0].value())
        var a = in_ptr.unsafe_load[width=width](i)
        comptime if op_code == IOP_ADD:
            out_ptr.unsafe_store[width=width](i, a + SIMD[dtype, width](scalar))
        comptime if op_code == IOP_MUL:
            out_ptr.unsafe_store[width=width](i, a * SIMD[dtype, width](scalar))

    comptime if has_accelerator():
        elementwise[func, simd_width=1, target="gpu"](Coord(size), ctx)
    else:
        raise Error("no GPU accelerator available at compile time")


@always_inline
def _fill[
    dtype: DType
](out_addr: Int, value: Float64, size: Int, ctx: DeviceContext) raises:
    """`out[i] = value` over a contiguous buffer.

    Shares the whole fill implementation with the in-place `StridedFill`
    bridge (`op_utils._fill_contig`): both write one repeated bit pattern,
    so both store it through the same-width unsigned integer type at the
    widest vector width the base address admits.
    """
    comptime if dtype == DType.float64 and has_apple_gpu_accelerator():
        raise Error("float64 is not supported on Apple GPU")
    comptime BITS = _fill_bits_dtype[dtype]()
    _fill_contig[BITS](out_addr, _fill_bits[dtype, BITS](value), size, ctx)


@always_inline
def _arange[
    dtype: DType
](
    out_ptr: Pointer[Scalar[dtype], MutUntrackedOrigin],
    start: Float64,
    step: Float64,
    size: Int,
    ctx: DeviceContext,
) raises:
    # Match PyTorch's GPU accumulator types: f32 accumulates in f32;
    # half/bfloat16 use f32; integral outputs use int64. Metal must never see
    # the f64 closure or even capture a Float64 value.
    comptime if dtype == DType.float32:
        var start_f32 = start.cast[DType.float32]()
        var step_f32 = step.cast[DType.float32]()

        @always_inline
        @parameter
        @__copy_capture(out_ptr, start_f32, step_f32)
        def gpu_f32[width: Int, alignment: Int = 1](idx: Coord):
            var i = Int(idx[0].value())
            out_ptr[unsafe_offset=i] = (
                start_f32 + Scalar[DType.float32](i) * step_f32
            ).cast[dtype]()

        comptime if has_accelerator():
            elementwise[gpu_f32, simd_width=1, target="gpu"](Coord(size), ctx)
        else:
            raise Error("no GPU accelerator available at compile time")
    elif dtype == DType.float16 or dtype == DType.bfloat16:
        var start_f32 = start.cast[DType.float32]()
        var step_f32 = step.cast[DType.float32]()

        @always_inline
        @parameter
        @__copy_capture(out_ptr, start_f32, step_f32)
        def lowp[width: Int, alignment: Int = 1](idx: Coord):
            var i = Int(idx[0].value())
            out_ptr[unsafe_offset=i] = (
                start_f32 + Scalar[DType.float32](i) * step_f32
            ).cast[dtype]()

        comptime if has_accelerator():
            elementwise[lowp, simd_width=1, target="gpu"](Coord(size), ctx)
        else:
            raise Error("no GPU accelerator available at compile time")
    elif dtype.is_integral():
        var start_i64 = start.cast[DType.int64]()
        var step_i64 = step.cast[DType.int64]()

        @always_inline
        @parameter
        @__copy_capture(out_ptr, start_i64, step_i64)
        def integral[width: Int, alignment: Int = 1](idx: Coord):
            var i = Int(idx[0].value())
            out_ptr[unsafe_offset=i] = (
                start_i64 + Scalar[DType.int64](i) * step_i64
            ).cast[dtype]()

        comptime if has_accelerator():
            elementwise[integral, simd_width=1, target="gpu"](Coord(size), ctx)
        else:
            raise Error("no GPU accelerator available at compile time")
    else:

        @always_inline
        @parameter
        @__copy_capture(out_ptr, start, step)
        def f64[width: Int, alignment: Int = 1](idx: Coord):
            var i = Int(idx[0].value())
            out_ptr[unsafe_offset=i] = (start + Float64(i) * step).cast[dtype]()

        comptime if has_apple_gpu_accelerator():
            raise Error("float64 is not supported on Apple GPU")
        elif has_accelerator():
            elementwise[f64, simd_width=1, target="gpu"](Coord(size), ctx)
        else:
            raise Error("no GPU accelerator available at compile time")


def _arange_go(
    out_ptr: Arg,
    start: Arg,
    step: Arg,
    numel: Arg,
    dtype_val: Arg,
    ctx_ptr: Arg,
) raises:
    var out_addr = _raw_int(out_ptr)
    var start_val = _raw_f64(start)
    var step_val = _raw_f64(step)
    var size = _raw_int(numel)
    var dtype = _raw_dtype_int(dtype_val)
    var ctx = _raw_ctx(ctx_ptr)

    var handled = False
    comptime for dt in [
        DType.float32,
        DType.float16,
        DType.bfloat16,
        DType.float64,
        DType.int64,
        DType.int32,
        DType.int16,
        DType.int8,
        DType.uint8,
    ]:
        comptime if _dtype_out_on[0, dt]():
            if dtype == dt:
                _arange[dt](
                    _make_ptr[dt](out_addr), start_val, step_val, size, ctx
                )
                handled = True
    if not handled:
        # A miss means Python selected the wrong immutable specialization.
        raise Error("unsupported dtype for fast arange: ", dtype)


# ---------------------------------------------------------------------------
# METH_FASTCALL wrappers: raw CPython argument unpacking (no owning
# PythonObject per argument). Argument types are guaranteed by the internal
# Python callers in aten_fast.py; errors cannot cross the C ABI, and the
# only raise sites are unsupported-dtype guards already gated upstream.
# ---------------------------------------------------------------------------


def _bin_dispatcher[op_code: Int](argv: Argv, argc: Int) raises:
    var args = argv
    _bin_go[op_code](
        args[unsafe_offset=0],
        args[unsafe_offset=1],
        args[unsafe_offset=2],
        args[unsafe_offset=3],
        args[unsafe_offset=4],
        args[unsafe_offset=5],
    )


def _arange_dispatcher(argv: Argv, argc: Int) raises:
    var args = argv
    _arange_go(
        args[unsafe_offset=0],
        args[unsafe_offset=1],
        args[unsafe_offset=2],
        args[unsafe_offset=3],
        args[unsafe_offset=4],
        args[unsafe_offset=5],
    )


# ---------------------------------------------------------------------------
# TensorSpec entries (docs/tensor_spec_design.md): the whole op prologue —
# input checks, output alloc, kernel launch — in one boundary call over
# cached TensorSpecs, reusing the contiguous kernels above. Failed checks
# raise a real NotImplementedError into Python ("take the classic path");
# nothing is swallowed on spec paths.
# ---------------------------------------------------------------------------

# Dtypes the unary spec entries dispatch on for the "direct" (in-dtype) ops
# and the bool-output ops; the transcendental ops gate down to FLOAT_DTYPES.
# float64 works on the CPU device (the kernels comptime-refuse it on GPU).
# Annotated List[DType]: rc1 infers bare `[...]` literals as Array, which no
# longer binds to variant_gates._dtype_supported's `List[DType]` parameter.
comptime INT_SCALAR_DTYPES: List[DType] = [DType.int32, DType.int64]

comptime SPEC_UNARY_DTYPES: List[DType] = [
    DType.float32,
    DType.float16,
    DType.bfloat16,
    DType.float64,
    DType.int8,
    DType.int16,
    DType.int32,
    DType.int64,
    DType.uint8,
]


def _unary_spec_into_go[op_code: Int](a_o: Arg, out_o: Arg) raises:
    ref a = _spec_ptr(a_o)[]
    ref out = _spec_ptr(out_o)[]

    comptime is_direct = (
        op_code == UOP_RELU
        or op_code == UOP_ABS
        or op_code == UOP_NEG
        or op_code == UOP_SIGN
    )
    var supported = False
    comptime if op_code == UOP_LOG2:
        supported = _dtype_supported[
            [DType.float16, DType.bfloat16, DType.float32, DType.float64]
        ](a.dtype)
    elif is_direct:
        supported = _dtype_supported[SPEC_UNARY_DTYPES](a.dtype)
    else:
        supported = _dtype_supported[List[DType](FLOAT_DTYPES)](a.dtype)
    if not supported:
        raise Error("mojo spec unary: unsupported dtype ", a.dtype)

    var ctx = a.ctx()
    var nbytes = a.numel * a.itemsize
    _ = nbytes
    _check_into(a, out, a.dtype)
    var addr = out.ptr
    if a.numel > 0:
        if not a.contig:
            raise Error(
                "mojo spec unary: input must be contiguous (Python"
                " pre-materializes)"
            )
        comptime for dt in SPEC_UNARY_DTYPES:
            comptime if _dtype_arg_on[0, dt]():
                if a.dtype == dt:
                    _unary_elementwise[dt, op_code](
                        _make_ptr[dt](addr),
                        _make_ptr[dt](a.ptr),
                        a.numel,
                        ctx,
                    )


def _unary_bool_spec_into_go[op_code: Int](a_o: Arg, out_o: Arg) raises:
    ref a = _spec_ptr(a_o)[]
    ref out = _spec_ptr(out_o)[]
    # bool inputs are read through their uint8 storage (bit-compatible).
    var kdtype = a.dtype
    if a.dtype == DType.bool:
        kdtype = DType.uint8
    var supported = False
    comptime for dt in SPEC_UNARY_DTYPES:
        comptime if _dtype_arg_abi_on[0, dt]():
            if kdtype == dt:
                supported = True
    if not supported:
        raise Error("mojo spec unary bool: unsupported dtype ", a.dtype)

    var ctx = a.ctx()
    var nbytes = a.numel  # bool output, itemsize 1
    _ = nbytes
    _check_into(a, out, DType.bool)
    var addr = out.ptr
    if a.numel > 0:
        if not a.contig:
            raise Error(
                "mojo spec unary bool: input must be contiguous (Python"
                " pre-materializes)"
            )
        comptime for dt in SPEC_UNARY_DTYPES:
            comptime if _dtype_arg_abi_on[0, dt]():
                if kdtype == dt:
                    _unary_bool[dt, op_code](
                        _make_ptr[DType.bool](addr),
                        _make_ptr[dt](a.ptr),
                        a.numel,
                        ctx,
                    )


def _scalar_spec_into_go[
    op_code: Int
](a_o: Arg, scalar_o: Arg, out_o: Arg) raises:
    ref a = _spec_ptr(a_o)[]
    ref out = _spec_ptr(out_o)[]
    if not _dtype_supported[List[DType](FLOAT_DTYPES)](a.dtype):
        raise Error("mojo spec scalar: unsupported dtype ", a.dtype)

    var scalar = Float32(_raw_f64(scalar_o))
    var ctx = a.ctx()
    var nbytes = a.numel * a.itemsize
    _ = nbytes
    _check_into(a, out, a.dtype)
    var addr = out.ptr
    if a.numel > 0:
        if not a.contig:
            raise Error(
                "mojo spec scalar: input must be contiguous (Python"
                " pre-materializes)"
            )
        comptime for dt in FLOAT_DTYPES:
            comptime if _dtype_arg_on[0, dt]():
                if a.dtype == dt:
                    _scalar_elementwise[dt, op_code](
                        _make_ptr[dt](addr),
                        _make_ptr[dt](a.ptr),
                        scalar,
                        a.numel,
                        ctx,
                    )


def _scalar_inplace_go[op_code: Int](a_o: Arg, scalar_o: Arg) raises:
    """`a op= scalar` for a contiguous float tensor, in place.

    The functional spec above allocates an output buffer, and the ATen in-place
    wrapper then copies it back over `a` -- an allocation and a
    device-to-device copy per call. That is invisible next to a real tensor but
    dominates a one-element tensor: nanoGPT's fused AdamW bumps 75 scalar step
    counters per step through `_foreach_add_.Scalar`, which ATen decomposes into
    75 `add_.Scalar`, and the copies alone cost ~376 us/step of GPU time.
    """
    ref a = _spec_ptr(a_o)[]
    var supported = False
    comptime for dt in FLOAT_DTYPES:
        if a.dtype == dt:
            supported = True
    if not supported:
        raise Error("mojo spec scalar inplace: unsupported dtype ", a.dtype)
    if not a.contig:
        raise Error("mojo spec scalar inplace: input is not contiguous")

    var scalar = Float32(_raw_f64(scalar_o))
    if a.numel > 0:
        var ctx = a.ctx()
        comptime for dt in FLOAT_DTYPES:
            if a.dtype == dt:
                _scalar_elementwise[dt, op_code](
                    _make_ptr[dt](a.ptr),
                    _make_ptr[dt](a.ptr),
                    scalar,
                    a.numel,
                    ctx,
                )
    return


def _scalar_inplace_dispatcher[op_code: Int](argv: Argv, argc: Int) raises:
    var args = argv
    _scalar_inplace_go[op_code](args[unsafe_offset=0], args[unsafe_offset=1])


def _int_scalar_spec_into_go[
    op_code: Int
](a_o: Arg, scalar_o: Arg, out_o: Arg) raises:
    ref a = _spec_ptr(a_o)[]
    ref out = _spec_ptr(out_o)[]
    if not _dtype_supported[INT_SCALAR_DTYPES](a.dtype):
        raise Error("mojo spec int scalar: unsupported dtype ", a.dtype)

    var scalar = _raw_int(scalar_o)
    var ctx = a.ctx()
    var nbytes = a.numel * a.itemsize
    _ = nbytes
    _check_into(a, out, a.dtype)
    var addr = out.ptr
    if a.numel > 0:
        if not a.contig:
            raise Error(
                "mojo spec int scalar: input must be contiguous (Python"
                " pre-materializes)"
            )
        comptime for dt in [DType.int32, DType.int64]:
            comptime if _dtype_arg_on[0, dt]():
                if a.dtype == dt:
                    _int_scalar_elementwise[dt, op_code](
                        _make_ptr[dt](addr),
                        _make_ptr[dt](a.ptr),
                        scalar,
                        a.numel,
                        ctx,
                    )


comptime SPEC_FILL_DTYPES = [
    DType.float32,
    DType.float16,
    DType.bfloat16,
    DType.float64,
    DType.int64,
    DType.int32,
    DType.int16,
    DType.int8,
    DType.uint8,
    DType.bool,
]


def _fill_spec_into_go(value_o: Arg, out_o: Arg) raises:
    """Fill a caller-allocated contiguous output; dtype/extent come from the
    output spec."""
    ref out = _spec_ptr(out_o)[]
    var value = _raw_f64(value_o)
    var supported = False
    comptime for dt in SPEC_FILL_DTYPES:
        comptime if _dtype_out_on[0, dt]():
            if out.dtype == dt:
                supported = True
    if not supported:
        raise Error("mojo spec fill into: unsupported dtype ", out.dtype)
    if not out.contig:
        raise Error("mojo spec fill into: output must be contiguous")
    if out.dtype == DType.bool:
        value = Float64(1) if value != 0 else Float64(0)
    var ctx = out.ctx()
    if out.numel > 0:
        comptime for dt in SPEC_FILL_DTYPES:
            comptime if _dtype_out_on[0, dt]():
                if out.dtype == dt:
                    _fill[dt](out.ptr, value, out.numel, ctx)


# ---------------------------------------------------------------------------
# Python module definition
# ---------------------------------------------------------------------------


@export
def tmb_call(argv: Argv, argc: Int, err: ErrBuf, errcap: Int) abi("C") -> Int32:
    """C entry of this family: one kernel per build (see `OP`).
    Slots are described in op_utils (`Arg`); errors come back as (rc=1, message).
    """
    try:
        comptime if _op_on["ReluSpec"]():
            _spec_dispatcher2[_unary_spec_into_go[UOP_RELU], "a unary spec op"](
                argv, argc
            )
            return 0
        comptime if _op_on["ExpSpec"]():
            _spec_dispatcher2[_unary_spec_into_go[UOP_EXP], "a unary spec op"](
                argv, argc
            )
            return 0
        comptime if _op_on["TanhSpec"]():
            _spec_dispatcher2[_unary_spec_into_go[UOP_TANH], "a unary spec op"](
                argv, argc
            )
            return 0
        comptime if _op_on["AbsSpec"]():
            _spec_dispatcher2[_unary_spec_into_go[UOP_ABS], "a unary spec op"](
                argv, argc
            )
            return 0
        comptime if _op_on["NegSpec"]():
            _spec_dispatcher2[_unary_spec_into_go[UOP_NEG], "a unary spec op"](
                argv, argc
            )
            return 0
        comptime if _op_on["SignSpec"]():
            _spec_dispatcher2[_unary_spec_into_go[UOP_SIGN], "a unary spec op"](
                argv, argc
            )
            return 0
        comptime if _op_on["CeilSpec"]():
            _spec_dispatcher2[_unary_spec_into_go[UOP_CEIL], "a unary spec op"](
                argv, argc
            )
            return 0
        comptime if _op_on["FloorSpec"]():
            _spec_dispatcher2[
                _unary_spec_into_go[UOP_FLOOR], "a unary spec op"
            ](argv, argc)
            return 0
        comptime if _op_on["AcosSpec"]():
            _spec_dispatcher2[_unary_spec_into_go[UOP_ACOS], "a unary spec op"](
                argv, argc
            )
            return 0
        comptime if _op_on["AsinhSpec"]():
            _spec_dispatcher2[
                _unary_spec_into_go[UOP_ASINH], "a unary spec op"
            ](argv, argc)
            return 0
        comptime if _op_on["AtanhSpec"]():
            _spec_dispatcher2[
                _unary_spec_into_go[UOP_ATANH], "a unary spec op"
            ](argv, argc)
            return 0
        comptime if _op_on["CosSpec"]():
            _spec_dispatcher2[_unary_spec_into_go[UOP_COS], "a unary spec op"](
                argv, argc
            )
            return 0
        comptime if _op_on["CoshSpec"]():
            _spec_dispatcher2[_unary_spec_into_go[UOP_COSH], "a unary spec op"](
                argv, argc
            )
            return 0
        comptime if _op_on["ErfSpec"]():
            _spec_dispatcher2[_unary_spec_into_go[UOP_ERF], "a unary spec op"](
                argv, argc
            )
            return 0
        comptime if _op_on["LogSpec"]():
            _spec_dispatcher2[_unary_spec_into_go[UOP_LOG], "a unary spec op"](
                argv, argc
            )
            return 0
        comptime if _op_on["Log2Spec"]():
            _spec_dispatcher2[_unary_spec_into_go[UOP_LOG2], "a unary spec op"](
                argv, argc
            )
            return 0
        comptime if _op_on["Log1pSpec"]():
            _spec_dispatcher2[
                _unary_spec_into_go[UOP_LOG1P], "a unary spec op"
            ](argv, argc)
            return 0
        comptime if _op_on["ReciprocalSpec"]():
            _spec_dispatcher2[
                _unary_spec_into_go[UOP_RECIPROCAL], "a unary spec op"
            ](argv, argc)
            return 0
        comptime if _op_on["RsqrtSpec"]():
            _spec_dispatcher2[
                _unary_spec_into_go[UOP_RSQRT], "a unary spec op"
            ](argv, argc)
            return 0
        comptime if _op_on["SigmoidSpec"]():
            _spec_dispatcher2[
                _unary_spec_into_go[UOP_SIGMOID], "a unary spec op"
            ](argv, argc)
            return 0
        comptime if _op_on["SiluSpec"]():
            _spec_dispatcher2[_unary_spec_into_go[UOP_SILU], "a unary spec op"](
                argv, argc
            )
            return 0
        comptime if _op_on["SinSpec"]():
            _spec_dispatcher2[_unary_spec_into_go[UOP_SIN], "a unary spec op"](
                argv, argc
            )
            return 0
        comptime if _op_on["SinhSpec"]():
            _spec_dispatcher2[_unary_spec_into_go[UOP_SINH], "a unary spec op"](
                argv, argc
            )
            return 0
        comptime if _op_on["SqrtSpec"]():
            _spec_dispatcher2[_unary_spec_into_go[UOP_SQRT], "a unary spec op"](
                argv, argc
            )
            return 0
        comptime if _op_on["TanSpec"]():
            _spec_dispatcher2[_unary_spec_into_go[UOP_TAN], "a unary spec op"](
                argv, argc
            )
            return 0
        comptime if _op_on["GeluNoneSpec"]():
            _spec_dispatcher2[
                _unary_spec_into_go[UOP_GELU_NONE], "a unary spec op"
            ](argv, argc)
            return 0
        comptime if _op_on["GeluTanhSpec"]():
            _spec_dispatcher2[
                _unary_spec_into_go[UOP_GELU_TANH], "a unary spec op"
            ](argv, argc)
            return 0
        comptime if _op_on["IsNanSpec"]():
            _spec_dispatcher2[
                _unary_bool_spec_into_go[BUOP_ISNAN],
                "a bool-output unary spec op",
            ](argv, argc)
            return 0
        comptime if _op_on["LogicalNotSpec"]():
            _spec_dispatcher2[
                _unary_bool_spec_into_go[BUOP_LOGICAL_NOT],
                "a bool-output unary spec op",
            ](argv, argc)
            return 0
        comptime if _op_on["AddScalarSpec"]():
            _spec_dispatcher3[
                _scalar_spec_into_go[SOP_ADD], "a float-scalar spec op"
            ](argv, argc)
            return 0
        comptime if _op_on["MulScalarSpec"]():
            _spec_dispatcher3[
                _scalar_spec_into_go[SOP_MUL], "a float-scalar spec op"
            ](argv, argc)
            return 0
        comptime if _op_on["PowScalarSpec"]():
            _spec_dispatcher3[
                _scalar_spec_into_go[SOP_POW], "a float-scalar spec op"
            ](argv, argc)
            return 0
        comptime if _op_on["AddScalarInplace"]():
            _scalar_inplace_dispatcher[SOP_ADD](argv, argc)
            return 0
        comptime if _op_on["MulScalarInplace"]():
            _scalar_inplace_dispatcher[SOP_MUL](argv, argc)
            return 0
        comptime if _op_on["AddScalarIntSpec"]():
            _spec_dispatcher3[
                _int_scalar_spec_into_go[IOP_ADD], "an int-scalar spec op"
            ](argv, argc)
            return 0
        comptime if _op_on["MulScalarIntSpec"]():
            _spec_dispatcher3[
                _int_scalar_spec_into_go[IOP_MUL], "an int-scalar spec op"
            ](argv, argc)
            return 0
        comptime if _op_on["FillSpec"]():
            _spec_dispatcher2[_fill_spec_into_go, "FillSpec"](argv, argc)
            return 0
        comptime if _op_on["Add"]():
            _bin_dispatcher[OP_ADD](argv, argc)
            return 0
        comptime if _op_on["Arange"]():
            _arange_dispatcher(argv, argc)
            return 0
        raise Error(NO_OP_COMPILED)
    except e:
        return _tmb_entry_error(err, errcap, e)
