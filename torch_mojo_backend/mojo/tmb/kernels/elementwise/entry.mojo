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

from tmb.kernels.common.unary_math import (
    elementwise_predicate,
    elementwise_unary,
    elementwise_unary_param,
    is_rounding,
    param_compute_dtype,
)

from std.os import abort
from max.gpu import block_dim, block_idx, grid_dim, thread_idx
from max.gpu.host import DeviceContext
from std.math import ceildiv
from std.sys.info import (
    has_accelerator,
    has_apple_gpu_accelerator,
    has_nvidia_gpu_accelerator,
    simd_width_of,
    size_of,
)
from std.utils.index import IndexList
from std.utils.coord import Coord

from tmb.kernels.common.gpu_elementwise import elementwise

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
    _spec_dispatcher5,
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
from tmb.kernels.common.div_math import floor_div, trunc_div
from tmb.kernels.common.math_utils import ieee_sqrt
from tmb.kernels.common.pow_math import torch_pow
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
# Three of the ops deserve a note: `tan`, `acosh` and `asinh` cannot call the
# std.math primitive of the same name, because those lower to libm
# (`_call_libm`) which `comptime assert`s CPU-only and would refuse to compile
# for the GPU target. `acosh` and `asinh` are composed from log/sqrt in shared
# math; `tan` routes through
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
comptime UOP_ACOSH = 27
# The float-only ops whose math is a scalar port of torch's CUDA routine
# (`unary_math.is_scalar_special`) or a rounding (`unary_math.is_rounding`):
# opcode UOP_TABLE_BASE + i computes kind _TABLE_UOP_KINDS[i], and its spec op
# is _TABLE_UOP_SPECS[i].
comptime UOP_TABLE_BASE = 28
comptime _TABLE_UOP_KINDS: List[StaticString] = [
    "airy_ai",
    "angle",
    "asin",
    "atan",
    "bessel_j0",
    "bessel_j1",
    "bessel_y0",
    "bessel_y1",
    "digamma",
    "entr",
    "erfc",
    "erfcx",
    "erfinv",
    "exp2",
    "expm1",
    "frac",
    "i0",
    "i0e",
    "i1",
    "i1e",
    "lgamma",
    "log10",
    "log_ndtr",
    "modified_bessel_i0",
    "modified_bessel_i1",
    "modified_bessel_k0",
    "modified_bessel_k1",
    "ndtri",
    "round",
    "scaled_modified_bessel_k0",
    "scaled_modified_bessel_k1",
    "sinc",
    "spherical_bessel_j0",
    "trunc",
]
comptime _TABLE_UOP_SPECS: List[StaticString] = [
    "AiryAiSpec",
    "AngleSpec",
    "AsinSpec",
    "AtanSpec",
    "BesselJ0Spec",
    "BesselJ1Spec",
    "BesselY0Spec",
    "BesselY1Spec",
    "DigammaSpec",
    "EntrSpec",
    "ErfcSpec",
    "ErfcxSpec",
    "ErfinvSpec",
    "Exp2Spec",
    "Expm1Spec",
    "FracSpec",
    "I0Spec",
    "I0eSpec",
    "I1Spec",
    "I1eSpec",
    "LgammaSpec",
    "Log10Spec",
    "LogNdtrSpec",
    "ModifiedBesselI0Spec",
    "ModifiedBesselI1Spec",
    "ModifiedBesselK0Spec",
    "ModifiedBesselK1Spec",
    "NdtriSpec",
    "RoundSpec",
    "ScaledModifiedBesselK0Spec",
    "ScaledModifiedBesselK1Spec",
    "SincSpec",
    "SphericalBesselJ0Spec",
    "TruncSpec",
]


@always_inline
def _table_uop_kind[op_code: Int]() -> StaticString:
    comptime assert (
        op_code >= UOP_TABLE_BASE
        and op_code < UOP_TABLE_BASE + len(_TABLE_UOP_KINDS)
    ), "not a table opcode"
    comptime kind = _TABLE_UOP_KINDS[op_code - UOP_TABLE_BASE]
    return kind


# Below this many elements the expensive half-precision bodies
# (`is_expensive_half` in `_unary_elementwise`) run faster at W4 than at the
# full 16-bytes/sizeof width: per-thread body latency, not block dispatch,
# limits such launches. Fitted on H100 PCIe (e.g. acosh f16, 281673 elements:
# 4.1 us at W4, 4.7 us at W8); it sits above 357*789 = 281673, the awkward
# shape of benchmarks/test_elementwise.py. Unmeasured on other GPUs.
comptime _NARROW_TRANSCENDENTAL_THRESHOLD = 300000


# The float32 scalar-port kinds below take 8 lanes per thread, two 16-byte
# vectors, from `_WIDE_UNARY_MIN` elements up: for these heavy branchy
# bodies two independent chains per thread beat twice the threads once the
# grid fills the GPU. Chosen kind by kind with ncu on H100 PCIe at base
# clocks, 16M uniform [0, 1) inputs, 4 -> 8 lanes (stock torch's vec4
# kernel in brackets), in us: lgamma 106 -> 90 (87), erfc 83 -> 71 (70),
# i0 128 -> 111 (109), i0e 117 -> 102 (97), i1 128 -> 112 (110), i1e 85 ->
# 71 (71), sinc 79 -> 70 (70), log_ndtr 118 -> 102 (105), atan 75 -> 70
# (68), ndtri 212 -> 199 (211), bessel_j0 77 -> 71 (68), bessel_y0 165 ->
# 132 (123), bessel_y1 143 -> 125, modified_bessel_i0 128 -> 112 (110),
# modified_bessel_i1 127 -> 111 (109), modified_bessel_k1 216 -> 200 (193),
# scaled_modified_bessel_k0 215 -> 199 (188), scaled_modified_bessel_k1
# 229 -> 214 (205), spherical_bessel_j0 85 -> 72 (72). expm1, entr,
# digamma, erfinv, airy_ai, bessel_j1 and erfcx measured the same or
# slower at 8 lanes and stay at 4. Below the threshold the 4-lane launch
# keeps more threads in flight (small grids).
comptime _WIDE_UNARY_MIN = 1 << 21


@always_inline
def _unary_heavy[op_code: Int]() -> Bool:
    """Unary bodies expensive enough for gpu_elementwise's small-grid block
    choice (`_policy_block`, 8-lane TINY launches): the scalar-port table
    kinds and the transcendentals that measured faster with it (asinh,
    atanh, cosh, erf, silu: benchmarks/, f16 / bf16 at 281673 elements, e.g.
    atanh 1.15 -> 1.05x torch); cos, log, sqrt, reciprocal and the cheap
    ones measured slower (log f16 0.82 -> 0.89x)."""
    return (
        op_code == UOP_ASINH
        or op_code == UOP_ATANH
        or op_code == UOP_COSH
        or op_code == UOP_ERF
        or op_code == UOP_SILU
        or _table_special[op_code]()
    )


@always_inline
def _param_heavy[kind: StaticString]() -> Bool:
    return kind == "logit" or kind == "polygamma" or kind == "mvlgamma"


@always_inline
def _table_special[op_code: Int]() -> Bool:
    """A table kind computed by a scalar port (not a rounding): on a half
    dtype it joins `is_expensive_half` (digamma f16 281673 elements: 19.5
    us at 8 lanes, torch 16.5; H100 PCIe, ncu base clocks)."""
    comptime if op_code < UOP_TABLE_BASE:
        return False
    else:
        return not is_rounding[_table_uop_kind[op_code]()]()


@always_inline
def _wide_unary_f32[dtype: DType, op_code: Int]() -> Bool:
    comptime if dtype != DType.float32 or op_code < UOP_TABLE_BASE:
        return False
    else:
        comptime kind = _table_uop_kind[op_code]()
        return (
            kind == "lgamma"
            or kind == "erfc"
            or kind == "i0"
            or kind == "i0e"
            or kind == "i1"
            or kind == "i1e"
            or kind == "sinc"
            or kind == "log_ndtr"
            or kind == "atan"
            or kind == "ndtri"
            or kind == "bessel_j0"
            or kind == "bessel_y0"
            or kind == "bessel_y1"
            or kind == "modified_bessel_i0"
            or kind == "modified_bessel_i1"
            or kind == "modified_bessel_k1"
            or kind == "scaled_modified_bessel_k0"
            or kind == "scaled_modified_bessel_k1"
            or kind == "spherical_bessel_j0"
        )


@always_inline
def _unary_is_direct[op_code: Int]() -> Bool:
    """RELU / ABS / NEG / SIGN: computed in the tensor dtype, integers too."""
    return (
        op_code == UOP_RELU
        or op_code == UOP_ABS
        or op_code == UOP_NEG
        or op_code == UOP_SIGN
    )


@always_inline
def _unary_float64_on[op_code: Int]() -> Bool:
    """Whether `op_code` takes float64: the dtype gate admits it and the
    float64 route of `_unary_elementwise` instantiates its kernel.

    Both read this one list so they cannot drift apart: the float64 kernel of
    an opcode outside it must not be built, because float64 acos, atanh, cos,
    sin, sinh and tan have no GPU lowering (std.math: "libm operations are
    only available on CPU targets", "DType.float64 is not supported for cos
    on NVIDIA GPU"; LLVM on AMD: "Cannot select: f64 = fcos").
    """
    comptime if op_code >= UOP_TABLE_BASE:
        # trunc / round / frac / angle are exact in float64.
        return is_rounding[_table_uop_kind[op_code]()]()
    else:
        return (
            _unary_is_direct[op_code]()
            or op_code == UOP_LOG2
            or op_code == UOP_RECIPROCAL
            or op_code == UOP_CEIL
            or op_code == UOP_FLOOR
        )


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
    elif op_code == UOP_ACOSH:
        return elementwise_unary["acosh"](a)
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
    elif op_code >= UOP_TABLE_BASE:
        return elementwise_unary[_table_uop_kind[op_code]()](a)
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
    comptime is_direct = _unary_is_direct[op_code]()
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
                @__parameter
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

                # Width, measured on H100 with gpu_elementwise: 16 bytes /
                # sizeof(dtype) (W8 f16/bf16, W4 f32) once both pointers are
                # 16-byte aligned, except the expensive half bodies below,
                # which take W4 under `_NARROW_TRANSCENDENTAL_THRESHOLD`
                # elements. 8-byte-aligned half views and every non-NVIDIA
                # GPU keep W4 (unmeasured there).
                comptime is_expensive_half = (
                    dtype == DType.float16 or dtype == DType.bfloat16
                ) and (
                    op_code == UOP_ACOS
                    or op_code == UOP_ACOSH
                    or op_code == UOP_GELU_NONE
                    or op_code == UOP_GELU_TANH
                    or op_code == UOP_LOG2
                    or op_code == UOP_LOG1P
                    or op_code == UOP_SINH
                    or op_code == UOP_TAN
                    or _table_special[op_code]()
                )
                comptime if has_nvidia_gpu_accelerator():
                    comptime full_width = 16 // size_of[dtype]()
                    if (Int(out_ptr) | Int(in_ptr)) % 16 == 0:
                        comptime if _wide_unary_f32[dtype, op_code]():
                            if size >= _WIDE_UNARY_MIN:
                                elementwise[
                                    gpu_func,
                                    simd_width=8,
                                    target="gpu",
                                    _trace_description="modular_unary",
                                    _heavy=_unary_heavy[op_code](),
                                ](Coord(size), ctx)
                                return
                        comptime if is_expensive_half:
                            if size < _NARROW_TRANSCENDENTAL_THRESHOLD:
                                elementwise[
                                    gpu_func,
                                    simd_width=4,
                                    target="gpu",
                                    _trace_description="modular_unary",
                                    _heavy=_unary_heavy[op_code](),
                                ](Coord(size), ctx)
                                return
                        elementwise[
                            gpu_func,
                            simd_width=full_width,
                            target="gpu",
                            _trace_description="modular_unary",
                            _heavy=_unary_heavy[op_code](),
                        ](Coord(size), ctx)
                        return
                    if (
                        Int(out_ptr) % 16 == 0
                        and Int(in_ptr) % (4 * size_of[dtype]()) != 0
                    ):
                        # An input view off the vector grid (a slice that
                        # starts at an odd element): element-aligned loads,
                        # still one W-lane body and one aligned vector
                        # store per thread.
                        @always_inline
                        @__parameter
                        @__copy_capture(out_ptr, in_ptr)
                        def gpu_func_ua[
                            width: Int, alignment: Int = 1
                        ](idx: Coord):
                            var i = Int(idx[0].value())
                            comptime st_align = min(
                                16, width * size_of[dtype]()
                            )
                            var a = in_ptr.unsafe_load[
                                width=width, alignment=size_of[dtype]()
                            ](i)
                            out_ptr.unsafe_store[
                                width=width, alignment=st_align
                            ](i, _unary_apply[dtype, width, op_code](a))

                        # 16 store bytes per thread from 300k elements up:
                        # H100 PCIe, ncu base clocks, offset_1 16M: f16 neg
                        # / exp2 / atan 37 / 40 / 47 us at 4 lanes, 33 / 33
                        # / 40 at 8 (torch's unrolled kernel 39 / 40 / 59);
                        # f32 is best at 4. Below, 4 lanes: the half-type
                        # transcendentals ran 1.2-1.6x slower at 8
                        # (benchmarks/, 281673 elements: digamma, rsqrt,
                        # sinh, log1p) and the cheap bodies measured even.
                        comptime if size_of[dtype]() == 2:
                            if size < _NARROW_TRANSCENDENTAL_THRESHOLD:
                                elementwise[
                                    gpu_func_ua,
                                    simd_width=4,
                                    target="gpu",
                                    _trace_description="modular_unary_ua",
                                    _heavy=_unary_heavy[op_code](),
                                ](Coord(size), ctx)
                                return
                        elementwise[
                            gpu_func_ua,
                            simd_width=16 // size_of[dtype](),
                            target="gpu",
                            _trace_description="modular_unary_ua",
                            _heavy=_unary_heavy[op_code](),
                        ](Coord(size), ctx)
                        return
                if (Int(out_ptr) | Int(in_ptr)) % (4 * size_of[dtype]()) == 0:
                    elementwise[
                        gpu_func,
                        simd_width=4,
                        target="gpu",
                        _trace_description="modular_unary",
                        _heavy=_unary_heavy[op_code](),
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
                op_code == UOP_LOG2
                or (dtype == DType.float64 and _unary_float64_on[op_code]())
            ) and not has_apple_gpu_accelerator():
                # Preserve log2's upstream scalar fallback, including
                # float64; the existing unary ops keep their 4-wide route.
                # Every float64 op the dtype gate admits lands here too
                # (`_unary_float64_on`): only Apple GPUs lack it.
                _enqueue_cached[_unary_contig_kernel[dtype, op_code]](
                    ctx,
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
comptime BUOP_SIGNBIT = 2
comptime BUOP_ISINF = 3
comptime BUOP_ISFINITE = 4
comptime BUOP_ISPOSINF = 5
comptime BUOP_ISNEGINF = 6


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
    elif op_code == BUOP_SIGNBIT:
        return elementwise_predicate["signbit"](a).cast[DType.uint8]()
    elif op_code == BUOP_ISINF:
        return elementwise_predicate["isinf"](a).cast[DType.uint8]()
    elif op_code == BUOP_ISFINITE:
        return elementwise_predicate["isfinite"](a).cast[DType.uint8]()
    elif op_code == BUOP_ISPOSINF:
        return elementwise_predicate["isposinf"](a).cast[DType.uint8]()
    elif op_code == BUOP_ISNEGINF:
        return elementwise_predicate["isneginf"](a).cast[DType.uint8]()
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
    @__parameter
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
            @__parameter
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
            # and 32 byte-sized inputs per thread for logical_not. float32
            # takes 8 lanes (32 input bytes), not 4: with the NVIDIA
            # gpu_elementwise launcher, 8-byte bool stores per thread beat
            # 4-byte ones by ~5.5% at 16M elements (48.4 vs 51.2 us
            # streamed; W16 for half measured no better than W8), fitted on
            # H100 PCIe at 1395 MHz, not measured elsewhere.
            comptime preferred_width = (
                32 if size_of[dtype]()
                == 1 else (
                    8 if size_of[dtype]() == 4 else 16 // size_of[dtype]()
                )
            )
            comptime if has_nvidia_gpu_accelerator() and preferred_width > 4:
                if (
                    Int(in_ptr) % 16 == 0
                    and Int(out_ptr) % min(16, preferred_width) == 0
                ):
                    elementwise[
                        gpu_bool, simd_width=preferred_width, target="gpu"
                    ](Coord(size), ctx)
                    return
                if (
                    Int(out_ptr) % 8 == 0
                    and Int(in_ptr) % (4 * size_of[dtype]()) != 0
                ):
                    # An input view off the vector grid: element-aligned
                    # loads, one aligned bool vector store per thread (as
                    # in `_unary_elementwise`'s `gpu_func_ua`).
                    @always_inline
                    @__parameter
                    @__copy_capture(out_ptr, in_ptr)
                    def gpu_bool_ua[width: Int, alignment: Int = 1](idx: Coord):
                        var i = Int(idx[0].value())
                        var a = in_ptr.unsafe_load[
                            width=width, alignment=size_of[dtype]()
                        ](i)
                        out_ptr.unsafe_bitcast[UInt8]().unsafe_store[
                            width=width, alignment=width
                        ](i, _unary_bool_vec[dtype, op_code, width](a))

                    # 8 lanes for 2-byte inputs, 4 for float32 (H100,
                    # offset_1 signbit: f16 16M 33 -> 26 us at 8, torch
                    # 36; f32 281674 elements 5.2 us at 4, 5.4 at 8).
                    comptime ua_w = 8 if size_of[dtype]() <= 2 else 4
                    elementwise[gpu_bool_ua, simd_width=ua_w, target="gpu"](
                        Coord(size), ctx
                    )
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
# Rounding divisions by a scalar, for bf16/fp16 tensors: ATen's CPU
# div_floor_kernel / div_trunc_kernel take a separate path when the divisor
# is a scalar (`iter.is_scalar(2)`) and the dtype is a reduced float, and
# divide in float32 by the scalar's ORIGINAL value (`original_scalar_value
# <opmath_t>`, a Python number never rounded to bf16). That is this family's
# float32 body exactly; the logic family's broadcast route would round the
# scalar to the tensor's dtype first, and trunc there divides in bf16.
comptime SOP_FLOORDIV = 3
comptime SOP_TRUNCDIV = 4
# rsub(Tensor, Scalar other) with alpha 1: other - self in opmath, one
# rounding (the CUDA sub kernel's fma(-1, self, other)).
comptime SOP_RSUB = 5


@__name("scalar_mul_contig_f32_v4_peel")
def _scalar_mul_peel_kernel(
    dst: Pointer[Float32, MutAnyOrigin],
    src: Pointer[Float32, ImmutAnyOrigin],
    scalar: Float32,
    size_arg: Int64,
    head_arg: Int64,
):
    var size = Int(size_arg)
    var head = Int(head_arg)
    var nvec = (size - head) // 4
    var tid = Int(block_idx.x) * Int(block_dim.x) + Int(thread_idx.x)
    var index = tid
    while index < nvec:
        var i = head + index * 4
        dst.unsafe_store[width=4, alignment=16](
            i,
            src.unsafe_load[width=4, alignment=16](i)
            * SIMD[DType.float32, 4](scalar),
        )
        index += Int(grid_dim.x) * Int(block_dim.x)
    if tid < head:
        dst[unsafe_offset=tid] = src[unsafe_offset=tid] * scalar
    var tail = head + nvec * 4
    if tid < size - tail:
        var i = tail + tid
        dst[unsafe_offset=i] = src[unsafe_offset=i] * scalar


comptime _POW_GENERAL = 0
comptime _POW_SQUARE = 1
comptime _POW_CUBE = 2
comptime _POW_INV_SQUARE = 3
comptime _POW_SQRT = 4
comptime _POW_RSQRT = 5
comptime _POW_RECIPROCAL = 6


@always_inline
def _pow_scalar_body[
    dtype: DType, pk: Int, w: Int
](a: SIMD[DType.float32, w], s: SIMD[DType.float32, w]) -> SIMD[dtype, w]:
    """pow(a, s) for a scalar exponent, `pk` naming the special exponent
    the host found (PowKernel.cu's pow_tensor_scalar_kernel_impl computes
    those in scalar_t: each product rounded to the tensor dtype)."""
    comptime f32 = DType.float32
    comptime if pk == _POW_SQUARE:
        return (a * a).cast[dtype]()
    elif pk == _POW_CUBE:
        var sq = (a * a).cast[dtype]().cast[f32]()
        return (sq * a).cast[dtype]()
    elif pk == _POW_INV_SQUARE:
        # `1.0 / (base * base)`: a double quotient of the rounded square,
        # which rounds to the IEEE float quotient.
        var sq = (a * a).cast[dtype]().cast[f32]()
        return (1 / sq).cast[dtype]()
    elif pk == _POW_SQRT:
        return elementwise_unary["sqrt"](a).cast[dtype]()
    elif pk == _POW_RSQRT:
        return elementwise_unary["rsqrt"](a).cast[dtype]()
    elif pk == _POW_RECIPROCAL:
        return elementwise_unary["reciprocal"](a).cast[dtype]()
    else:
        # pow_math.torch_pow, as in logic's BOP_POW: C's special cases.
        return torch_pow(a, s).cast[dtype]()


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
        comptime if dtype == DType.float32 and op_code == SOP_MUL and _has_sm_9x():
            # H100 measurements select a common alignment peel for arrays
            # of at least 1024 elements. Keep the original aligned/divisible
            # route, other operations/dtypes and non-Hopper targets intact.
            if (
                ctx.api() == "cuda"
                and size >= 1024
                and Int(in_ptr) % 16 == Int(out_ptr) % 16
                and (Int(in_ptr) % 16 != 0 or size % 4 != 0)
            ):
                var head = min(size, ((16 - Int(in_ptr) % 16) % 16) // 4)
                var nvec = (size - head) // 4
                _enqueue_cached[_scalar_mul_peel_kernel](
                    ctx,
                    min(ceildiv(nvec, 256), 1 << 22),
                    1,
                    1,
                    256,
                    out_ptr.as_unsafe_any_origin(),
                    in_ptr.as_unsafe_any_origin().as_imm(),
                    scalar,
                    Int64(size),
                    Int64(head),
                )
                return

        # pow: PowKernel.cu converts the exponent to scalar_t
        # (`exp_scalar.to<scalar_t>()`), so a half tensor's exponent is
        # rounded to its dtype before the special-exponent tests and the pow
        # (ROCm runs the same source). torch MPS passes it as a float
        # (UnaryKernel.mm `pow_tensor_scalar_kernel`), unrounded.
        var sv = scalar
        comptime if (
            op_code == SOP_POW
            and not has_apple_gpu_accelerator()
            and (dtype == DType.float16 or dtype == DType.bfloat16)
        ):
            sv = scalar.cast[dtype]().cast[DType.float32]()

        @always_inline
        @__parameter
        @__copy_capture(out_ptr, in_ptr, sv)
        def body[width: Int, al: Int, pk: Int](i: Int):
            var a = in_ptr.unsafe_load[width=width, alignment=al](i).cast[
                DType.float32
            ]()
            var s = SIMD[DType.float32, width](sv)
            comptime if op_code == SOP_ADD:
                out_ptr.unsafe_store[width=width, alignment=al](
                    i, (a + s).cast[dtype]()
                )
            comptime if op_code == SOP_MUL:
                out_ptr.unsafe_store[width=width, alignment=al](
                    i, (a * s).cast[dtype]()
                )
            comptime if op_code == SOP_FLOORDIV:
                out_ptr.unsafe_store[width=width, alignment=al](
                    i, floor_div(a, s).cast[dtype]()
                )
            comptime if op_code == SOP_TRUNCDIV:
                out_ptr.unsafe_store[width=width, alignment=al](
                    i, trunc_div(a, s).cast[dtype]()
                )
            comptime if op_code == SOP_POW:
                out_ptr.unsafe_store[width=width, alignment=al](
                    i, _pow_scalar_body[dtype, pk](a, s)
                )
            comptime if op_code == SOP_RSUB:
                out_ptr.unsafe_store[width=width, alignment=al](
                    i, (s - a).cast[dtype]()
                )

        # Element alignment only: the CPU lanes and the GPU scalar lanes may
        # start at any element (a bucket view starts wherever the previous
        # parameter ended).
        @always_inline
        @__parameter
        def launch[pk: Int]() raises:
            @always_inline
            @__parameter
            def func[width: Int, alignment: Int = 1](idx: Coord):
                body[width, size_of[dtype](), pk](Int(idx[0].value()))

            # 16-byte vectors once both bases proved aligned; the launcher's
            # width-1 tail lanes are element-aligned only.
            @always_inline
            @__parameter
            def func_vec[width: Int, alignment: Int = 1](idx: Coord):
                comptime al = 16 if width > 1 else size_of[dtype]()
                body[width, al, pk](Int(idx[0].value()))

            # Every vector at 16-byte alignment: only launched when there
            # is no tail (the route other GPUs keep).
            @always_inline
            @__parameter
            def func_vec_no_tail[width: Int, alignment: Int = 1](idx: Coord):
                body[width, 16, pk](Int(idx[0].value()))

            comptime if has_accelerator():
                # scalar lanes moved 1.5 TB/s on H100, vectors ~3.
                comptime vec = 16 // size_of[dtype]()
                var aligned = Int(out_ptr) % 16 == 0 and Int(in_ptr) % 16 == 0
                comptime if has_nvidia_gpu_accelerator():
                    # Sizes off the vector multiple vectorize too, their
                    # tail lanes element-aligned (`func_vec`).
                    if aligned:
                        elementwise[func_vec, simd_width=vec, target="gpu"](
                            Coord(size), ctx
                        )
                        return
                else:
                    if aligned and size % vec == 0:
                        elementwise[
                            func_vec_no_tail, simd_width=vec, target="gpu"
                        ](Coord(size), ctx)
                        return
                elementwise[func, simd_width=1, target="gpu"](Coord(size), ctx)
            else:
                raise Error("no GPU accelerator available at compile time")

        comptime if op_code == SOP_POW and has_apple_gpu_accelerator():
            # UnaryKernel.mm's pow_tensor_scalar_kernel: 2 is `sqr`, -1 /
            # -0.5 / 0.5 the reciprocal / rsqrt / sqrt kernels; the rest
            # (3 and -2 included) runs the full float pow.
            if sv == 2:
                launch[_POW_SQUARE]()
            elif sv == 0.5:
                launch[_POW_SQRT]()
            elif sv == -0.5:
                launch[_POW_RSQRT]()
            elif sv == -1:
                launch[_POW_RECIPROCAL]()
            else:
                launch[_POW_GENERAL]()
        elif op_code == SOP_POW:
            # PowKernel.cu's pow_tensor_scalar_kernel (CUDA and ROCm):
            # exponents 2, 3 and -2 are products, 0.5 / -0.5 / -1 the sqrt /
            # rsqrt / reciprocal kernels; only the rest runs the full pow.
            if sv == 2:
                launch[_POW_SQUARE]()
            elif sv == 3:
                launch[_POW_CUBE]()
            elif sv == -2:
                launch[_POW_INV_SQUARE]()
            elif sv == 0.5:
                launch[_POW_SQRT]()
            elif sv == -0.5:
                launch[_POW_RSQRT]()
            elif sv == -1:
                launch[_POW_RECIPROCAL]()
            else:
                launch[_POW_GENERAL]()
        else:
            launch[_POW_GENERAL]()


# ---------------------------------------------------------------------------
# Unary ops with runtime scalar arguments (`unary_math.elementwise_unary_param`):
# kind _PARAM_UOP_KINDS[i] is spec op _PARAM_UOP_SPECS[i], and takes three
# float64 slots after its input spec (converted on the host to the kernel's
# compute type, `unary_math.param_compute_dtype`). round_decimals and
# nan_to_num are exact in any float dtype and take float64 too; the others
# are float-only.
# ---------------------------------------------------------------------------

comptime _PARAM_UOP_KINDS: List[StaticString] = [
    "logit",
    "mvlgamma",
    "nan_to_num",
    "polygamma",
    "round_decimals",
]
comptime _PARAM_UOP_SPECS: List[StaticString] = [
    "LogitSpec",
    "MvlgammaSpec",
    "NanToNumSpec",
    "PolygammaSpec",
    "RoundDecimalsSpec",
]


@always_inline
def _param_float64_on[index: Int]() -> Bool:
    comptime kind = _PARAM_UOP_KINDS[index]
    return kind == "nan_to_num" or kind == "round_decimals"


@always_inline
def _param_unary_elementwise[
    dtype: DType, index: Int
](
    out_ptr: Pointer[Scalar[dtype], MutUntrackedOrigin],
    in_ptr: Pointer[Scalar[dtype], MutUntrackedOrigin],
    p0: Float64,
    p1: Float64,
    p2: Float64,
    size: Int,
    ctx: DeviceContext,
) raises:
    comptime kind = _PARAM_UOP_KINDS[index]
    comptime if not dtype.is_floating_point():
        raise Error("parameterized unary ops require a floating point dtype")
    else:
        # Converted here, on the host: the kernel never sees a double unless
        # the tensor is float64 (see `param_compute_dtype`).
        comptime ct = param_compute_dtype[dtype]()
        var q0 = p0.cast[ct]()
        var q1 = p1.cast[ct]()
        var q2 = p2.cast[ct]()

        @always_inline
        @__parameter
        @__copy_capture(out_ptr, in_ptr, q0, q1, q2)
        def body[width: Int, lal: Int, sal: Int](i: Int):
            out_ptr.unsafe_store[width=width, alignment=sal](
                i,
                elementwise_unary_param[kind](
                    in_ptr.unsafe_load[width=width, alignment=lal](i),
                    q0,
                    q1,
                    q2,
                ),
            )

        comptime esz = size_of[dtype]()

        @always_inline
        @__parameter
        def func[width: Int, alignment: Int = 1](idx: Coord):
            body[width, esz, esz](Int(idx[0].value()))

        @always_inline
        @__parameter
        def func_vec[width: Int, alignment: Int = 1](idx: Coord):
            body[width, 16, 16](Int(idx[0].value()))

        # An input view off the vector grid: element-aligned loads, aligned
        # vector stores.
        @always_inline
        @__parameter
        def func_ua[width: Int, alignment: Int = 1](idx: Coord):
            comptime al = 16 if width > 1 else esz
            body[width, esz, al](Int(idx[0].value()))

        comptime if has_accelerator():
            comptime vec = 16 // size_of[dtype]()
            if (Int(out_ptr) | Int(in_ptr)) % 16 == 0:
                # polygamma / mvlgamma on a half dtype: a loop per element,
                # so small launches keep more threads at 4 lanes (as the
                # unary family's `is_expensive_half`; polygamma(2, x) f16
                # 281673 elements: 1.18x torch at 8 lanes). NVIDIA only.
                comptime if (
                    has_nvidia_gpu_accelerator()
                    and (kind == "polygamma" or kind == "mvlgamma")
                    and (dtype == DType.float16 or dtype == DType.bfloat16)
                ):
                    if size < _NARROW_TRANSCENDENTAL_THRESHOLD:
                        elementwise[
                            func_vec,
                            simd_width=4,
                            target="gpu",
                            _trace_description="modular_param_unary",
                            _heavy=_param_heavy[kind](),
                        ](Coord(size), ctx)
                        return
                elementwise[
                    func_vec,
                    simd_width=vec,
                    target="gpu",
                    _trace_description="modular_param_unary",
                    _heavy=_param_heavy[kind](),
                ](Coord(size), ctx)
            else:
                comptime if has_nvidia_gpu_accelerator():
                    if Int(out_ptr) % 16 == 0:
                        comptime if (
                            kind == "polygamma" or kind == "mvlgamma"
                        ) and (
                            dtype == DType.float16 or dtype == DType.bfloat16
                        ):
                            if size < _NARROW_TRANSCENDENTAL_THRESHOLD:
                                elementwise[
                                    func_ua,
                                    simd_width=4,
                                    target="gpu",
                                    _trace_description="modular_param_unary_ua",
                                    _heavy=_param_heavy[kind](),
                                ](Coord(size), ctx)
                                return
                        elementwise[
                            func_ua,
                            simd_width=vec,
                            target="gpu",
                            _trace_description="modular_param_unary_ua",
                            _heavy=_param_heavy[kind](),
                        ](Coord(size), ctx)
                        return
                elementwise[
                    func,
                    simd_width=1,
                    target="gpu",
                    _trace_description="modular_param_unary",
                    _heavy=_param_heavy[kind](),
                ](Coord(size), ctx)
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
    @__parameter
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
        @__parameter
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
        @__parameter
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
        @__parameter
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
        @__parameter
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
# TensorSpec entries (agents_docs/tensor_spec_design.md): the whole op prologue —
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

    comptime is_direct = _unary_is_direct[op_code]()
    var supported = False
    comptime if is_direct:
        supported = _dtype_supported[SPEC_UNARY_DTYPES](a.dtype)
    elif _unary_float64_on[op_code]():
        supported = _dtype_supported[
            [DType.float16, DType.bfloat16, DType.float32, DType.float64]
        ](a.dtype)
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


def _param_unary_spec_into_go[
    index: Int
](a_o: Arg, p0_o: Arg, p1_o: Arg, p2_o: Arg, out_o: Arg) raises:
    ref a = _spec_ptr(a_o)[]
    ref out = _spec_ptr(out_o)[]
    var supported = False
    comptime if _param_float64_on[index]():
        supported = _dtype_supported[
            [DType.float16, DType.bfloat16, DType.float32, DType.float64]
        ](a.dtype)
    else:
        supported = _dtype_supported[List[DType](FLOAT_DTYPES)](a.dtype)
    if not supported:
        raise Error("mojo parameterized unary: unsupported dtype ", a.dtype)
    var ctx = a.ctx()
    _check_into(a, out, a.dtype)
    var addr = out.ptr
    if a.numel > 0:
        if not a.contig:
            raise Error("mojo parameterized unary: input must be contiguous")
        comptime for dt in [
            DType.float16,
            DType.bfloat16,
            DType.float32,
            DType.float64,
        ]:
            comptime if _dtype_arg_on[0, dt]():
                if a.dtype == dt:
                    _param_unary_elementwise[dt, index](
                        _make_ptr[dt](addr),
                        _make_ptr[dt](a.ptr),
                        _raw_f64(p0_o),
                        _raw_f64(p1_o),
                        _raw_f64(p2_o),
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
        comptime if _op_on["AcoshSpec"]():
            _spec_dispatcher2[
                _unary_spec_into_go[UOP_ACOSH], "a unary spec op"
            ](argv, argc)
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
        comptime for i in range(len(_TABLE_UOP_SPECS)):
            comptime if _op_on[_TABLE_UOP_SPECS[i]]():
                _spec_dispatcher2[
                    _unary_spec_into_go[UOP_TABLE_BASE + i], "a unary spec op"
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
        comptime if _op_on["SignbitSpec"]():
            _spec_dispatcher2[
                _unary_bool_spec_into_go[BUOP_SIGNBIT],
                "a bool-output unary spec op",
            ](argv, argc)
            return 0
        comptime if _op_on["IsInfSpec"]():
            _spec_dispatcher2[
                _unary_bool_spec_into_go[BUOP_ISINF],
                "a bool-output unary spec op",
            ](argv, argc)
            return 0
        comptime if _op_on["IsFiniteSpec"]():
            _spec_dispatcher2[
                _unary_bool_spec_into_go[BUOP_ISFINITE],
                "a bool-output unary spec op",
            ](argv, argc)
            return 0
        comptime if _op_on["IsPosInfSpec"]():
            _spec_dispatcher2[
                _unary_bool_spec_into_go[BUOP_ISPOSINF],
                "a bool-output unary spec op",
            ](argv, argc)
            return 0
        comptime if _op_on["IsNegInfSpec"]():
            _spec_dispatcher2[
                _unary_bool_spec_into_go[BUOP_ISNEGINF],
                "a bool-output unary spec op",
            ](argv, argc)
            return 0
        comptime for i in range(len(_PARAM_UOP_SPECS)):
            comptime if _op_on[_PARAM_UOP_SPECS[i]]():
                _spec_dispatcher5[
                    _param_unary_spec_into_go[i], "a parameterized unary op"
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
        comptime if _op_on["RsubScalarSpec"]():
            _spec_dispatcher3[
                _scalar_spec_into_go[SOP_RSUB], "a float-scalar spec op"
            ](argv, argc)
            return 0
        comptime if _op_on["FloorDivScalarSpec"]():
            _spec_dispatcher3[
                _scalar_spec_into_go[SOP_FLOORDIV], "a float-scalar spec op"
            ](argv, argc)
            return 0
        comptime if _op_on["TruncDivScalarSpec"]():
            _spec_dispatcher3[
                _scalar_spec_into_go[SOP_TRUNCDIV], "a float-scalar spec op"
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
