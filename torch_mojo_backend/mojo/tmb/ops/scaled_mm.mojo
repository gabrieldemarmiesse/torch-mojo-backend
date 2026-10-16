"""torch._scaled_mm / _scaled_mm_v2 on the mojo device: float8 GEMMs with
the scaling recipes stock CUDA accepts (aten/src/ATen/native/cuda/
ScaledBlas.cpp, ScaledBlasUtils.cpp, torch 2.14).

    out = cvt(((dequant(A) @ dequant(B)) * (scale_a * scale_b) + bias)
              * scale_result)

Each operand is dequantized to float32 (tmb/kernels/scaled_mm, `Dequant`),
the product is the float32 GEMM of tmb/ops/matmul.mojo and one `Epilogue`
pass applies the outer scales, the bias and the output conversion.

* Tensorwise and rowwise scales factor out of the K sum, so they stay out of
  the operands: the float32 copies of float8 values are exact (three
  mantissa bits at most) and so are their products, which makes the float32
  GEMM -- and the TF32 one, whose 10-bit mantissa still holds every float8
  value -- compute what a float8 tensor-core GEMM with a float32 accumulator
  does. That route takes the TF32 tensor-core bridge whatever
  `torch.get_float32_matmul_precision()` says: it loses nothing here.
* Block scales (1x128, 128x128 with float32 scales; MX 1x32 with
  float8_e8m0fnu scales in the SWIZZLE_32_4_4 layout) vary along K: like
  cuBLASLt, each K block's exact partial product is multiplied by that
  block's scale_a, then scale_b, and accumulated in float32 (`BlockAccum`).

`use_fast_accum` is accepted and ignored (the accumulation is always full
float32, which the option allows). A float8 output saturates to its largest
finite value, as cuBLAS writes one. `scale_result` multiplies a float8
output (cuBLAS's D scale) and is ignored for any other: torch 2.14
behaviour (pytorch#190979, test_float8_scale_result); torch 2.11-2.13 ignore
it for every output. Tensorwise scales are combined before they meet the
accumulator, as cuBLAS's alpha is, so 1e38 * 1e-38 does not overflow;
rowwise scales multiply it in turn, scale_b then scale_a, as CUDA's rowwise
kernel does; block scales are applied per K block (above).

_scaled_mm_v2 validates its scale lists like torch 2.14's structured meta
(validate_scaled_mm_v2_inputs, which runs before CUDA's kernel): an empty or
two-entry list for a tensorwise / rowwise recipe raises "scale_a must have 1
Float element" (resp. "... N Float elements"), and a recipe list of another
length "unsupported recipe(s)". torch 2.11's kernel reads scale_a[0] of an
empty list (it aborts) and says "Invalid scaling configuration." for the
others.

Where this is more permissive than an H100: a rowwise recipe takes a float8
output (cuBLAS >= 12.9 does too), cuBLASLt's row-major x
column-major layout rule is not enforced (any strides work), and recipes
CUDA gates by architecture (DeepSeek blocks: SM90 only; MX: SM100) run on
every device. The fnuz float8 dtypes (ROCm's) are accepted as operands.
torch raises `ValueError` for several of these checks; the boxed-kernel
ABI has no ValueError channel, so they surface as `RuntimeError` with the
same message.
"""
from std.utils import IndexList

from tmb.backend.abi import (
    Owned,
    TAG_NONE,
    TAG_OPT_TENSOR_LIST,
    TAG_TENSOR,
    TAG_TENSOR_LIST,
    TAG_TENSOR_REF,
    ST_BFLOAT16,
    ST_FLOAT16,
    ST_FLOAT32,
    ST_INT32,
    ST_FLOAT8_E5M2,
    ST_FLOAT8_E8M0FNU,
    T,
    Values,
    IntList,
    cpu_empty,
    dtype_code,
    max_dtype,
    new_strided,
    new_tensor,
    own,
    own_if_new,
    ret_ref,
    ret_tensor,
    unsupported,
    v_dtype_or,
    v_is_none,
    v_opt_tensor,
    v_tensor,
    v_tensor_list,
    view_strided,
)
from tmb.backend.device import copy_to_host, ctx_for, ctx_ptr
from tmb.backend.kernel_call import KernelCall
from tmb.backend.registry import Site, impl
from tmb.kernels.common.float8 import is_float8
from tmb.kernels.common.op_utils import MAX_RANK
from tmb.ops.common import (
    check_out_as,
    contiguous,
    copy_strided_into,
    shares_storage,
    device_str,
    fill_value,
    resize_out,
)
from tmb.ops.data_movement import _scalar_type_name
from tmb.ops.matmul import (
    _dense_2d,
    _spec_matmul,
    _sm90_cuda,
    _tf32_available,
    _tf32_bridge,
)

# at::blas::ScalingType
comptime TENSOR_WISE = 0
comptime ROW_WISE = 1
comptime BLOCK_1X16 = 2
comptime BLOCK_1X32 = 3
comptime BLOCK_1X128 = 4
comptime BLOCK_128X128 = 5
# at::blas::SwizzleType
comptime SWIZZLE_32_4_4 = 1

# The order the epilogue applies scale_a and scale_b in -- each CUDA path's
# own, measured on stock CUDA with 1e37 / 1e-37 scales (which overflows):
# tensorwise is cuBLAS's combined alpha, the grouped CUTLASS epilogue is
# ScaleB * (ScaleA * acc), rowwise _scaled_mm multiplies by scale_b first.
comptime ORDER_COMBINED = 0
comptime ORDER_A_THEN_B = 2
comptime ORDER_B_THEN_A = 4

# `Dequant`'s recipes (tmb/kernels/scaled_mm/entry.mojo).
comptime DQ_NONE = 0
comptime DQ_TENSOR = 1
comptime DQ_ROW = 2
comptime DQ_BLOCK_1X128 = 3
comptime DQ_BLOCK_128X128 = 4
comptime DQ_MX_SWIZZLED = 5


def _cdiv(a: Int, b: Int) -> Int:
    return (a + b - 1) // b


def _round_up(a: Int, b: Int) -> Int:
    return _cdiv(a, b) * b


def _shape2(r: Int, c: Int) -> IndexList[MAX_RANK]:
    var s = IndexList[MAX_RANK](1)
    s[MAX_RANK - 2] = r
    s[MAX_RANK - 1] = c
    return s


def _strides2(r: Int, c: Int) -> IndexList[MAX_RANK]:
    var s = IndexList[MAX_RANK](0)
    s[MAX_RANK - 2] = r
    s[MAX_RANK - 1] = c
    return s


def _contig2(d0: Int, d1: Int, s0: Int, s1: Int) -> Bool:
    """`is_contiguous()` of a 2-D layout (size-1 dims are free)."""
    if d0 * d1 == 0:
        return True
    if d1 != 1 and s1 != 1:
        return False
    return d0 == 1 or s0 == d1


@fieldwise_init
struct _Sc(Copyable, ImplicitlyCopyable, Movable):
    """A scale tensor as the recipe predicates see it, possibly through a
    `.t()`: rank, the two (logical) sizes and strides, numel, dtype,
    contiguity and data address."""

    var rank: Int
    var d0: Int
    var d1: Int
    var s0: Int
    var s1: Int
    var numel: Int
    var stype: Int32
    var contig: Bool
    var addr: Int

    def size(self, i: Int) -> Int:
        return self.d0 if i == 0 else self.d1

    def stride(self, i: Int) -> Int:
        return self.s0 if i == 0 else self.s1


def _sc(t: T) -> _Sc:
    var d0 = t.dim(0) if t.rank >= 1 else 1
    var d1 = t.dim(1) if t.rank >= 2 else 1
    var s0 = t.stride(0) if t.rank >= 1 else 0
    var s1 = t.stride(1) if t.rank >= 2 else 0
    return _Sc(t.rank, d0, d1, s0, s1, t.numel, t.stype, t.contig, t.ptr)


def _sc_t(t: T) raises -> _Sc:
    """The `.t()` view of a scale tensor; like `Tensor.t()`, refuses rank > 2
    (get_joint_scaling transposes scale_b before it checks any recipe)."""
    if t.rank > 2:
        raise Error(
            "t() expects a tensor with <= 2 dimensions, but self is ",
            t.rank,
            "D",
        )
    if t.rank < 2:
        return _sc(t)
    return _Sc(
        2,
        t.dim(1),
        t.dim(0),
        t.stride(1),
        t.stride(0),
        t.numel,
        t.stype,
        _contig2(t.dim(1), t.dim(0), t.stride(1), t.stride(0)),
        t.ptr,
    )


def _vec_stride(s: _Sc, n: Int) -> Int:
    """The element stride of a scale holding one value per index of a
    length-`n` dimension. torch only checks such a scale's element count
    (a rowwise v2 scale_b of shape (2, 8) for 16 columns passes), so a
    contiguous one is read flat; a non-contiguous one reaches here only
    with its `n` values along a single dim (callers flatten the rest, see
    `_flat_scale`)."""
    if s.rank == 0 or n <= 1:
        return 0
    if s.contig:
        return 1
    if s.rank == 1:
        return s.s0
    return s.s1 if s.d1 == n else s.s0


def _flat_scale(t: T, n: Int) raises -> Owned:
    """A per-row / per-column scale of `n` values in logical (flattened)
    order: `t` itself when contiguous or one-dimensional in effect, else a
    contiguous copy. cuBLAS would read `n` consecutive floats from the data
    pointer whatever the strides; reading the tensor's own elements is the
    stride-correct semantics (and never reads past the view)."""
    var single = t.rank <= 1 or (
        t.rank == 2 and (t.dim(0) == n or t.dim(1) == n)
    )
    if t.contig or single:
        return own_if_new(t.copy(), t)
    return own_if_new(contiguous(t), t)


# --- CUDA's recipe predicates (ScaledBlas.cpp, is_*_scaling) ---------------


def _ss(s: _Sc, dim: Int, size: Int, stride: Int) -> Bool:
    """check_size_stride: the stride of a size-<=1 dim is free."""
    return s.size(dim) == size and (size <= 1 or s.stride(dim) == stride)


def _is_scaling(
    rows: Int, cols: Int, mat_st: Int32, s: _Sc, kind: Int
) raises -> Bool:
    """Whether scale `s` is recipe `kind` for a `rows` x `cols` float8
    matrix of dtype `mat_st` (the outer dim is rows, K is cols)."""
    if not is_float8(max_dtype(mat_st)):
        return False
    if kind == BLOCK_1X32:
        return (
            s.stype == ST_FLOAT8_E8M0FNU
            and s.numel == _round_up(rows, 128) * _round_up(_cdiv(cols, 32), 4)
            and s.contig
        )
    if kind == BLOCK_1X16 or s.stype != ST_FLOAT32:
        return False  # 1x16 is float4-only
    if kind == TENSOR_WISE:
        return s.numel == 1
    if s.rank != 2:
        return False
    if kind == ROW_WISE:
        return s.d0 == rows and s.d1 == 1 and s.contig
    if kind == BLOCK_1X128:
        return _ss(s, 0, rows, 1) and _ss(s, 1, _cdiv(cols, 128), rows)
    return _ss(s, 0, _cdiv(rows, 128), _cdiv(cols, 128)) and _ss(
        s, 1, _cdiv(cols, 128), 1
    )


# --- shared validation -------------------------------------------------------


def _mat_str(t: T) -> String:
    return String(t.dim(0)) + "x" + String(t.dim(1))


def _shape_str(t: T) -> String:
    var s = String("[")
    for i in range(t.rank):
        if i:
            s += ", "
        s += String(t.dim(i))
    return s + "]"


def _stride_str(t: T) -> String:
    var s = String("[")
    for i in range(t.rank):
        if i:
            s += ", "
        s += String(t.stride(i))
    return s + "]"


def _tname(t: T) -> String:
    return _scalar_type_name(t.dtype)


def _check_bias(bias: Optional[T], out_st: Int32) raises:
    """_scaled_mm_out_cuda's bias rules, in its order."""
    if not bias:
        return
    var b = bias.value().copy()
    var out_name = _scalar_type_name(max_dtype(out_st))
    if out_st == ST_FLOAT32:
        raise Error("Bias is not supported when out_dtype is set to Float32")
    if b.stype != ST_BFLOAT16 and b.stype != ST_FLOAT16:
        raise Error("Bias must be BFloat16 or Half, but got ", _tname(b))
    if out_st == ST_BFLOAT16 and b.stype != ST_BFLOAT16:
        raise Error(
            "Bias must be BFloat16 to compute ",
            out_name,
            " output, but got ",
            _tname(b),
        )
    if out_st == ST_FLOAT16 and b.stype != ST_FLOAT16:
        raise Error(
            "Bias must be Float16 to compute ",
            out_name,
            " output, but got ",
            _tname(b),
        )


def _check_same_device(ts: List[T]) raises:
    """checkAllSameGPU over the defined tensors."""
    for i in range(len(ts)):
        if not ts[i].on_mojo() or ts[i].device != ts[0].device:
            raise Error(
                (
                    "Expected all tensors to be on the same device, but found"
                    " at least two devices, "
                ),
                device_str(ts[0]),
                " and ",
                device_str(ts[i]),
                "!",
            )


def _check_mat_dtypes(
    a: T, b: T, a_name: StaticString, b_name: StaticString
) raises:
    if not is_float8(a.dtype):
        raise Error(
            "Expected ",
            a_name,
            " to be Float8 or Float4_x2 matrix got ",
            _tname(a),
        )
    if not is_float8(b.dtype):
        raise Error(
            "Expected ",
            b_name,
            " to be Float8 or Float4_x2 matrix got ",
            _tname(b),
        )
    if a.stype == ST_FLOAT8_E5M2 and b.stype == ST_FLOAT8_E5M2:
        raise Error(
            "Multiplication of two Float8_e5m2 matrices is not supported"
        )
    # (CUDA reaches this only past its empty-result early return.)
    var empty = a.numel == 0 or b.numel == 0
    if (
        a.stype == ST_FLOAT8_E8M0FNU or b.stype == ST_FLOAT8_E8M0FNU
    ) and not empty:
        unsupported(
            "_scaled_mm of Float8_e8m0fnu matrices (a scale dtype, not a data"
            " dtype)"
        )


def _check_out_dtype(out_st: Int32) raises:
    var dt = max_dtype(out_st)
    if not (
        out_st == ST_FLOAT32
        or out_st == ST_BFLOAT16
        or out_st == ST_FLOAT16
        or (is_float8(dt) and out_st != ST_FLOAT8_E8M0FNU)
    ):
        unsupported("_scaled_mm with a " + _scalar_type_name(dt) + " output")


# --- the computation ---------------------------------------------------------


@fieldwise_init
struct _Dq(Copyable, ImplicitlyCopyable, Movable):
    """How `Dequant` reads one operand's scale (DQ_NONE: not at all)."""

    var recipe: Int
    var addr: Int
    var ss0: Int
    var ss1: Int


def _dq_none() -> _Dq:
    return _Dq(DQ_NONE, 0, 0, 0)


def _dequant(src: T, rows: Int, cols: Int, rs: Int, cs: Int) raises -> Owned:
    """A contiguous float32 `rows` x `cols` copy of the float8 matrix at
    `src` (element strides rs, cs): exact, unscaled."""
    var out = own(new_tensor(_shape2(rows, cols), 2, ST_FLOAT32, src.device))
    var ctx = ctx_for(src.device)
    var cp = ctx_ptr(ctx)
    var call = KernelCall("scaled_mm", "Dequant")
    call.arg_dtype(0, src.dtype)
    call.int(out.t.ptr)
    call.int(src.ptr)
    call.int(rows)
    call.int(cols)
    call.int(rs)
    call.int(cs)
    call.int(cp)
    call.run()
    _ = ctx
    return out^


def _gemm_f32(a32: T, bt32: T, exact_tf32: Bool) raises -> Owned:
    """`a32 @ bt32.T` (both contiguous float32, a32 M x K, bt32 N x K)."""
    var k = a32.dim(1)
    var n = bt32.dim(0)
    var b = own(
        view_strided(bt32, _shape2(k, n), _strides2(1, k), 2, bt32.offset)
    )
    if exact_tf32 and _sm90_cuda(a32.device) and _tf32_available():
        var am = _dense_2d(a32)
        var bm = _dense_2d(b.t)
        if am and bm:
            var r = _tf32_bridge(
                a32, am.value(), bm.value(), False, None, List[Int]()
            )
            if r:
                _ = b^
                return own(r.value().copy())
    # Block-scaled operands are no longer exact in TF32: always strict
    # float32 here, whatever torch.get_float32_matmul_precision() says (the
    # cuBLAS float8 GEMM accumulates in full float32 regardless).
    var r = _spec_matmul("MatmulSpec", a32, b.t, None, 0)
    _ = b^
    if not r:
        unsupported("a float32 GEMM for _scaled_mm on this device")
    return own(r.value().copy())


@fieldwise_init
struct _Outer(Copyable, ImplicitlyCopyable, Movable):
    """An outer (epilogue) scale: address (0 = none), element stride
    (0 = one scalar)."""

    var addr: Int
    var stride: Int


def _scaled_gemm(
    dest: T,
    a: T,
    b: T,
    dq_a: _Dq,
    dq_b: _Dq,
    sa: _Outer,
    sb: _Outer,
    bias: Optional[T],
    scale_result: Int = 0,
    order: Int = ORDER_COMBINED,
    grouped: Bool = False,
) raises:
    """dest (M x N, any strides) = cvt((dequant(a) @ dequant(b) * (sa * sb)
    + bias) * scale_result), the outer scales applied in `order` (see the
    ORDER_* constants: each CUDA path has its own). `b` is mat2 (K x N); its scale view `dq_b` is
    indexed (n, k). `scale_result` is a float32 scalar's address (0: none)."""
    var m = a.dim(0)
    var k = a.dim(1)
    var n = b.dim(1)
    if m == 0 or n == 0:
        return
    if k == 0 and not grouped:
        # _scaled_mm: `out.zero_()` and nothing else, bias included.
        fill_value(dest, 0.0)
        return
    var p: Owned
    if k == 0:
        # A grouped GEMM's empty-K group still runs its epilogue on a zero
        # accumulator (0 * inf scale is NaN there, as on CUDA).
        p = own(new_tensor(_shape2(m, n), 2, ST_FLOAT32, dest.device))
        fill_value(p.t, 0.0)
    elif dq_a.recipe != DQ_NONE or dq_b.recipe != DQ_NONE:
        p = _block_scaled_product(a, b, dq_a, dq_b)
    else:
        var a32 = _dequant(a, m, k, a.stride(0), a.stride(1))
        # B transposed: row n of B^T is column n of B.
        var bt32 = _dequant(b, n, k, b.stride(1), b.stride(0))
        p = _gemm_f32(a32.t, bt32.t, True)
        _ = a32^
        _ = bt32^
    var ctx = ctx_for(dest.device)
    var cp = ctx_ptr(ctx)
    var call = KernelCall("scaled_mm", "Epilogue")
    call.out_dtype(dest.dtype)
    call.int(dest.ptr)
    call.int(p.t.ptr)
    call.int(m)
    call.int(n)
    call.int(dest.stride(0))
    call.int(dest.stride(1))
    call.int(sa.addr)
    call.int(sa.stride)
    call.int(sb.addr)
    call.int(sb.stride)
    # torch only checks `bias.numel() == N`, so a (2, 8) bias of a 16-column
    # GEMM is its 16 elements in order: read it flattened. Deliberate
    # divergence for a NON-contiguous bias: cuBLAS reads N consecutive
    # elements from the data pointer, ignoring strides (measured, torch
    # 2.11 / H100: bias = arange(32)[::2] adds 0, 1, ..., 15, values the
    # view does not even contain). We read the view's own elements
    # (0, 2, ..., 30) -- stride-correct, and never past the view's storage.
    var bsrc = bias.value().copy() if bias else p.t.copy()  # p: contiguous
    var flat_bias = own_if_new(contiguous(bsrc), bsrc)
    if bias:
        var bv = flat_bias.t.copy()
        call.int(bv.ptr)
        call.int((1 if bv.dtype == DType.float16 else 0) | order)
        call.int(1)
    else:
        call.int(0)
        call.int(order)
        call.int(0)
    call.int(scale_result)
    call.int(dtype_code(dest.dtype))
    call.int(cp)
    call.run()
    _ = ctx
    _ = p^  # read by the epilogue: alive past the launch
    _ = flat_bias^


def _block_scaled_product(a: T, b: T, dq_a: _Dq, dq_b: _Dq) raises -> Owned:
    """`a @ b` under block scales, the way cuBLASLt computes it: per K
    block (128 for the float32 1x128 / 128x128 recipes, 32 for MX), an exact
    partial product of the unscaled float8 values, scaled by that block's
    scales and added to a float32 accumulator (`BlockAccum`). Pre-scaling
    the operands instead changes the arithmetic: a 1e37 scale overflows a
    partial sum on CUDA where it would not in a pre-scaled operand."""
    var m = a.dim(0)
    var k = a.dim(1)
    var n = b.dim(1)
    var bk = 32 if dq_a.recipe == DQ_MX_SWIZZLED else 128
    var nkb = _cdiv(k, bk)
    var acc = own(new_tensor(_shape2(m, n), 2, ST_FLOAT32, a.device))
    for kb in range(nkb):
        var k0 = kb * bk
        var kl = min(bk, k - k0)
        var av = own(
            view_strided(
                a,
                _shape2(m, kl),
                _strides2(a.stride(0), a.stride(1)),
                2,
                a.offset + k0 * a.stride(1),
            )
        )
        var bv = own(
            view_strided(
                b,
                _shape2(kl, n),
                _strides2(b.stride(0), b.stride(1)),
                2,
                b.offset + k0 * b.stride(0),
            )
        )
        var a32 = _dequant(av.t, m, kl, a.stride(0), a.stride(1))
        var bt32 = _dequant(bv.t, n, kl, b.stride(1), b.stride(0))
        _ = av^
        _ = bv^
        var part = _gemm_f32(a32.t, bt32.t, True)
        _ = a32^
        _ = bt32^
        var ctx = ctx_for(a.device)
        var call = KernelCall("scaled_mm", "BlockAccum")
        call.int(acc.t.ptr)
        call.int(part.t.ptr)
        call.int(m)
        call.int(n)
        call.int(kb)
        call.int(1 if kb == 0 else 0)
        call.int(dq_a.addr)
        call.int(dq_a.recipe)
        call.int(dq_a.ss0)
        call.int(dq_a.ss1)
        call.int(dq_b.addr)
        call.int(dq_b.recipe)
        call.int(dq_b.ss0)
        call.int(dq_b.ss1)
        call.int(_cdiv(k, 32))
        call.int(ctx_ptr(ctx))
        call.run()
        _ = ctx
        _ = part^  # read by the launch
    return acc^


def _plan(kind: Int, s: _Sc, outer: Int) -> Tuple[_Dq, _Outer]:
    """A recipe's split between `Dequant` and the epilogue. `s` is the scale
    as indexed (outer index, K block); `outer` is M for A, N for B."""
    if kind == TENSOR_WISE:
        return (_dq_none(), _Outer(s.addr, 0))
    if kind == ROW_WISE:
        return (_dq_none(), _Outer(s.addr, _vec_stride(s, outer)))
    if kind == BLOCK_1X128:
        return (_Dq(DQ_BLOCK_1X128, s.addr, s.s0, s.s1), _Outer(0, 0))
    if kind == BLOCK_128X128:
        return (_Dq(DQ_BLOCK_128X128, s.addr, s.s0, s.s1), _Outer(0, 0))
    return (_Dq(DQ_MX_SWIZZLED, s.addr, 0, 0), _Outer(0, 0))


def _run_recipe(
    dest: T,
    a: T,
    b: T,
    kind_a: Int,
    sa: _Sc,
    kind_b: Int,
    sb: _Sc,
    bias: Optional[T],
    scale_result: Int = 0,
) raises:
    var pa = _plan(kind_a, sa, a.dim(0))
    var pb = _plan(kind_b, sb, b.dim(1))
    _scaled_gemm(
        dest,
        a,
        b,
        pa[0],
        pb[0],
        pa[1],
        pb[1],
        bias,
        scale_result,
        ORDER_B_THEN_A if kind_a == ROW_WISE else ORDER_COMBINED,
    )


# --- aten::_scaled_mm (v1) ---------------------------------------------------


def _invalid_scaling(a: T, b: T, sa: T, sb: T) raises:
    var m = a.dim(0)
    var k = a.dim(1)
    var n = b.dim(1)
    raise Error(
        (
            "Invalid scaling configuration.\n- For TensorWise scaling, a and b"
            " should be float8, scales should be float and singletons.\n- For"
            " RowWise scaling, a and b should be float8, scales should be"
            " float, scale_a should be ("
        ),
        m,
        ", 1) and scale_b should be (1, ",
        n,
        (
            "), and both should be contiguous.\n- For BlockWise 1x128 scaling,"
            " a and b should be float8, scales should be float, scale_a should"
            " be ("
        ),
        m,
        ", ",
        _cdiv(k, 128),
        ") and scale_b should be (",
        _cdiv(b.dim(0), 128),
        ", ",
        n,
        (
            "), and both should be outer-dim-major.\n- For BlockWise 128x128"
            " scaling, a and b should be float8, scales should be float,"
            " scale_a should be ("
        ),
        _cdiv(m, 128),
        ", ",
        _cdiv(k, 128),
        ") and scale_b should be (",
        _cdiv(b.dim(0), 128),
        ", ",
        _cdiv(n, 128),
        (
            "), and both should be near-inner-dim-major (with 16-byte aligned"
            " strides).\n- For Blockwise 1x32 scaling, a and b should be"
            " float8, scales should be float8_e8m0fnu, scale_a should have "
        ),
        _round_up(m, 128) * _round_up(_cdiv(k, 32), 4),
        " elements and scale_b should have ",
        _round_up(n, 128) * _round_up(_cdiv(b.dim(0), 32), 4),
        " elements, and both should be contiguous.\nGot a.dtype()=",
        _tname(a),
        ", scale_a.dtype()=",
        _tname(sa),
        ", scale_a.size()=",
        _shape_str(sa),
        ", scale_a.stride()=",
        _stride_str(sa),
        ", b.dtype()=",
        _tname(b),
        ", scale_b.dtype()=",
        _tname(sb),
        ", scale_b.size()=",
        _shape_str(sb),
        " and scale_b.stride()=",
        _stride_str(sb),
    )


def _v1_joint_scaling(
    a: T, b: T, scale_a: T, scale_b: T
) raises -> Tuple[Int, Int]:
    """get_joint_scaling over CUDA's option list, in its priority order."""
    var pairs: List[Tuple[Int, Int]] = [
        (TENSOR_WISE, TENSOR_WISE),
        (ROW_WISE, ROW_WISE),
        (BLOCK_128X128, BLOCK_1X128),
        (BLOCK_1X128, BLOCK_128X128),
        (BLOCK_1X128, BLOCK_1X128),
        (BLOCK_1X32, BLOCK_1X32),
    ]
    var m = a.dim(0)
    var k = a.dim(1)
    var n = b.dim(1)
    for p in pairs:
        var lhs = p[0]
        var rhs = p[1]
        # 1x32 scales are swizzled over the original dims, never transposed.
        var sb = _sc(scale_b) if rhs == BLOCK_1X32 else _sc_t(scale_b)
        if _is_scaling(m, k, a.stype, _sc(scale_a), lhs) and _is_scaling(
            n, b.dim(0), b.stype, sb, rhs
        ):
            return (lhs, rhs)
    _invalid_scaling(a, b, scale_a, scale_b)
    return (0, 0)


def _scaled_mm_v1(
    a: T,
    b: T,
    scale_a: T,
    scale_b: T,
    bias: Optional[T],
    scale_result: Optional[T],
    out_dtype: Int32,
    has_out_dtype: Bool,
    mut dest: T,
) raises:
    """_scaled_mm_out_cuda, check for check."""
    if a.rank != 2:
        raise Error("mat1 must be a matrix")
    if b.rank != 2:
        raise Error("mat2 must be a matrix")
    if a.dim(1) != b.dim(0):
        raise Error(
            "mat1 and mat2 shapes cannot be multiplied (",
            _mat_str(a),
            " and ",
            _mat_str(b),
            ")",
        )
    var kinds = _v1_joint_scaling(a, b, scale_a, scale_b)
    if scale_result:
        var sr = scale_result.value().copy()
        if sr.numel != 1 or sr.stype != ST_FLOAT32:
            raise Error("scale_result must be a float scalar")
    var n = b.dim(1)
    if bias and bias.value().numel != n:
        raise Error("Bias must be size ", n, " but got ", bias.value().numel)
    if a.dim(1) % 16 != 0:
        raise Error(
            (
                "Expected trailing dimension of mat1 to be divisible by 16 but"
                " got mat1 shape: ("
            ),
            _mat_str(a),
            ").",
        )
    if b.dim(0) % 16 != 0 or b.dim(1) % 16 != 0:
        raise Error("mat2 shape (", _mat_str(b), ") must be divisible by 16")
    if has_out_dtype and out_dtype != dest.stype:
        raise Error("out_dtype must match output matrix type")
    _check_mat_dtypes(a, b, "mat1", "mat2")
    _check_bias(bias, dest.stype)
    var devs: List[T] = [dest.copy(), a.copy(), b.copy()]
    if bias:
        devs.append(bias.value().copy())
    devs.append(scale_a.copy())
    devs.append(scale_b.copy())
    if scale_result:
        devs.append(scale_result.value().copy())
    _check_same_device(devs)
    if a.numel > 0 and b.numel > 0:
        # CUDA's output-dtype limits are cuBLAS's, past the empty early return.
        _check_out_dtype(dest.stype)
    resize_out(dest, _shape2(a.dim(0), n), 2)
    var sb = _sc(scale_b) if kinds[1] == BLOCK_1X32 else _sc_t(scale_b)
    # ScaledBlas.cpp hands scale_result to cuBLAS as the D scale only for a
    # float8 output (torch 2.14: test_float8_scale_result multiplies).
    var sr = 0
    if scale_result and is_float8(dest.dtype):
        sr = scale_result.value().ptr
    _run_recipe(dest, a, b, kinds[0], _sc(scale_a), kinds[1], sb, bias, sr)


# aten::_scaled_mm(Tensor self, Tensor mat2, Tensor scale_a, Tensor scale_b,
#   Tensor? bias=None, Tensor? scale_result=None, ScalarType? out_dtype=None,
#   bool use_fast_accum=False) -> Tensor
def op_scaled_mm(args: Values, n_args: Int, rets: Values, n_rets: Int) raises:
    var a = v_tensor(args[unsafe_offset=0])
    var b = v_tensor(args[unsafe_offset=1])
    var has_dt = not v_is_none(args[unsafe_offset=6])
    var out_st = v_dtype_or(args[unsafe_offset=6], a.stype)
    _ = max_dtype(out_st)  # a dtype the device cannot hold declines here
    var out = own(new_tensor(_shape2(0, 0), 2, out_st, a.device))
    _scaled_mm_v1(
        a,
        b,
        v_tensor(args[unsafe_offset=2]),
        v_tensor(args[unsafe_offset=3]),
        v_opt_tensor(args[unsafe_offset=4]),
        v_opt_tensor(args[unsafe_offset=5]),
        out_st,
        has_dt,
        out.t,
    )
    ret_tensor(rets, 0, out.take())


def _arg_tensors(args: Values, idxs: List[Int]) raises -> List[T]:
    """The tensors among the given schema arguments (lists flattened)."""
    var out = List[T]()
    for i in idxs:
        var v = args[unsafe_offset=i].copy()
        if v.tag == TAG_NONE:
            continue
        if v.tag == TAG_TENSOR_LIST or v.tag == TAG_OPT_TENSOR_LIST:
            for t in v_tensor_list(v):
                out.append(t.copy())
        elif v.tag == TAG_TENSOR or v.tag == TAG_TENSOR_REF:
            out.append(v_tensor(v))
    return out^


def _aliases_any(dest: T, ts: List[T]) -> Bool:
    """Whether `out` shares storage with an input: resizing it first could
    reallocate that storage and leave the input's pointer stale (and even
    without a resize, an input read after part of `out` was written would
    see the new values)."""
    for t in ts:
        if shares_storage(dest, t):
            return True
    return False


def _land(mut dest: T, res: T) raises:
    """Hand a result computed into a temporary to the caller's `out`."""
    resize_out(dest, res.shape, res.rank)
    if res.numel > 0:
        copy_strided_into(dest, res)


# aten::_scaled_mm.out(..., bool use_fast_accum=False, *, Tensor(a!) out)
def op_scaled_mm_out(
    args: Values, n_args: Int, rets: Values, n_rets: Int
) raises:
    var a = v_tensor(args[unsafe_offset=0])
    var b = v_tensor(args[unsafe_offset=1])
    var dest = v_tensor(args[unsafe_offset=8])
    var has_dt = not v_is_none(args[unsafe_offset=6])
    var out_st = v_dtype_or(args[unsafe_offset=6], dest.stype)
    if _aliases_any(dest, _arg_tensors(args, [0, 1, 2, 3, 4, 5])):
        var tmp = own(new_tensor(_shape2(0, 0), 2, dest.stype, dest.device))
        _scaled_mm_v1(
            a,
            b,
            v_tensor(args[unsafe_offset=2]),
            v_tensor(args[unsafe_offset=3]),
            v_opt_tensor(args[unsafe_offset=4]),
            v_opt_tensor(args[unsafe_offset=5]),
            out_st,
            has_dt,
            tmp.t,
        )
        _land(dest, tmp.t)
        _ = tmp^
    else:
        _scaled_mm_v1(
            a,
            b,
            v_tensor(args[unsafe_offset=2]),
            v_tensor(args[unsafe_offset=3]),
            v_opt_tensor(args[unsafe_offset=4]),
            v_opt_tensor(args[unsafe_offset=5]),
            out_st,
            has_dt,
            dest,
        )
    ret_ref(rets, 0, v_tensor(args[unsafe_offset=8]))


# --- aten::_scaled_mm_v2 -----------------------------------------------------


def _ints(l: IntList) -> List[Int]:
    return l.to_list()


def _v2_validate(
    a: T,
    b: T,
    scale_a: List[T],
    recipe_a: List[Int],
    swizzle_a: List[Int],
    scale_b: List[T],
    recipe_b: List[Int],
    swizzle_b: List[Int],
    bias: Optional[T],
    contraction_dim: List[Int],
) raises:
    """TORCH_META_FUNC(_scaled_mm_v2) and validate_scaled_mm_v2_inputs
    (non-ROCm, non-XPU)."""
    if a.rank != 2:
        raise Error("mat_a must be a matrix")
    if b.rank != 2:
        raise Error("mat_b must be a matrix")
    if len(contraction_dim) > 0:
        if len(contraction_dim) != 2:
            raise Error("contraction_dim must have exactly 2 elements")
        var da = contraction_dim[0] + (2 if contraction_dim[0] < 0 else 0)
        var db = contraction_dim[1] + (2 if contraction_dim[1] < 0 else 0)
        if da < 0 or da > 1 or db < 0 or db > 1:
            raise Error("contraction_dim out of range")
        if a.dim(da) != b.dim(db):
            raise Error(
                "mat_a and mat_b shapes cannot be multiplied (",
                _mat_str(a),
                " and ",
                _mat_str(b),
                ") with contraction dims mat_a: ",
                contraction_dim[0],
                ", mat_b: ",
                contraction_dim[1],
            )
        if da != 1 or db != 0:
            unsupported(
                "_scaled_mm_v2 with a non-default contraction_dim for float8"
                " operands"
            )
    elif a.dim(1) != b.dim(0):
        raise Error(
            "mat_a and mat_b shapes cannot be multiplied (",
            _mat_str(a),
            " and ",
            _mat_str(b),
            ")",
        )
    var n = b.dim(1)
    if bias and bias.value().numel != n:
        raise Error("Bias must be size ", n, " but got ", bias.value().numel)
    if not is_float8(a.dtype) or not is_float8(b.dtype):
        raise Error(
            "Expected both inputs to be fp8 or fp4 types but got mat_a.dtype=",
            _tname(a),
            " and mat_b.dtype=",
            _tname(b),
        )
    var m = a.dim(0)
    var k = a.dim(1)

    @__parameter
    def single(ea: Int, eb: Int) -> Bool:
        return (
            len(recipe_a) == 1
            and len(recipe_b) == 1
            and recipe_a[0] == ea
            and recipe_b[0] == eb
        )

    var is_tw = single(TENSOR_WISE, TENSOR_WISE)
    var is_rw = single(ROW_WISE, ROW_WISE)
    var is_mx = single(BLOCK_1X32, BLOCK_1X32)
    var is_ds = (
        single(BLOCK_1X128, BLOCK_1X128)
        or single(BLOCK_1X128, BLOCK_128X128)
        or single(BLOCK_128X128, BLOCK_1X128)
    )
    if is_tw:
        if (
            len(scale_a) != 1
            or scale_a[0].numel != 1
            or scale_a[0].stype != ST_FLOAT32
        ):
            raise Error("scale_a must have 1 Float element")
        if (
            len(scale_b) != 1
            or scale_b[0].numel != 1
            or scale_b[0].stype != ST_FLOAT32
        ):
            raise Error("scale_b must have 1 Float element")
    elif is_rw:
        if (
            len(scale_a) != 1
            or scale_a[0].numel != m
            or scale_a[0].stype != ST_FLOAT32
        ):
            raise Error(
                "scale_a must have ",
                m,
                " Float elements, got ",
                scale_a[0].numel if len(scale_a) else 0,
            )
        if (
            len(scale_b) != 1
            or scale_b[0].numel != n
            or scale_b[0].stype != ST_FLOAT32
        ):
            raise Error(
                "scale_b must have ",
                n,
                " Float elements, got ",
                scale_b[0].numel if len(scale_b) else 0,
            )
    elif is_mx:
        var ea = _round_up(m, 128) * _round_up(_cdiv(k, 32), 4)
        var eb = _round_up(n, 128) * _round_up(_cdiv(k, 32), 4)
        if (
            len(scale_a) != 1
            or scale_a[0].numel != ea
            or scale_a[0].stype != ST_FLOAT8_E8M0FNU
        ):
            raise Error(
                "For Blockwise scaling scale_a should have ",
                ea,
                " elements, got: ",
                scale_a[0].numel if len(scale_a) else 0,
            )
        if (
            len(scale_b) != 1
            or scale_b[0].numel != eb
            or scale_b[0].stype != ST_FLOAT8_E8M0FNU
        ):
            raise Error(
                "For Blockwise scaling scale_b should have ",
                eb,
                " elements, got: ",
                scale_b[0].numel if len(scale_b) else 0,
            )
        if not scale_a[0].contig or not scale_b[0].contig:
            raise Error(
                "For Blockwise scaling both scales should be contiguous"
            )
        if len(swizzle_a) != 1:
            raise Error("swizzle_a must have 1 value, got ", len(swizzle_a))
        if len(swizzle_b) != 1:
            raise Error("swizzle_b must have 1 value, got ", len(swizzle_b))
        if swizzle_a[0] != SWIZZLE_32_4_4:
            raise Error("scale_a must be swizzled to SWIZZLE_32_4_4 format")
        if swizzle_b[0] != SWIZZLE_32_4_4:
            raise Error("scale_b must be swizzled to SWIZZLE_32_4_4 format")
    elif not is_ds:
        var is_fp4_recipe = (
            single(BLOCK_1X16, BLOCK_1X16)
            or len(recipe_a) == 2
            and len(recipe_b) == 2
        )
        if is_fp4_recipe:
            unsupported("_scaled_mm_v2 NVFP4 recipes (float4 operands)")
        raise Error(
            (
                "Invalid scaling configuration for _scaled_mm_v2: unsupported"
                " recipe(s). Got mat_a.dtype="
            ),
            _tname(a),
            ", mat_b.dtype=",
            _tname(b),
            ", recipe_a.size()=",
            len(recipe_a),
            ", recipe_b.size()=",
            len(recipe_b),
        )


def _v2_invalid(a: T, b: T, sa: T, sb: T) raises:
    var m = a.dim(0)
    var k = a.dim(1)
    var n = b.dim(1)
    raise Error(
        (
            "Invalid scaling configuration.\n- For TensorWise scaling, a and b"
            " should be float8, scales should be float and singletons.\n- For"
            " RowWise scaling, a and b should be float8, scales should be"
            " float, scale_a should be ("
        ),
        m,
        ", 1) and scale_b should be (1, ",
        n,
        (
            "), and both should be contiguous.\n- For BlockWise 1x128 scaling,"
            " a and b should be float8, scales should be float, scale_a should"
            " be ("
        ),
        m,
        ", ",
        _cdiv(k, 128),
        ") and scale_b should be (",
        _cdiv(b.dim(0), 128),
        ", ",
        n,
        (
            "), and both should be outer-dim-major.\n- For BlockWise 128x128"
            " scaling, a and b should be float8, scales should be float,"
            " scale_a should be ("
        ),
        _cdiv(m, 128),
        ", ",
        _cdiv(k, 128),
        ") and scale_b should be (",
        _cdiv(b.dim(0), 128),
        ", ",
        _cdiv(n, 128),
        (
            "), and both should be near-inner-dim-major (with 16-byte aligned"
            " strides).\nGot mat_a.dtype()="
        ),
        _tname(a),
        ", scale_a[0].dtype()=",
        _tname(sa),
        ", scale_a[0].size()=",
        _shape_str(sa),
        ", scale_a[0].stride()=",
        _stride_str(sa),
        ", mat_b.dtype()=",
        _tname(b),
        ", scale_b[0].dtype()=",
        _tname(sb),
        ", scale_b[0].size()=",
        _shape_str(sb),
        " and scale_b[0].stride()=",
        _stride_str(sb),
    )


def _v2_deepseek_checks(
    kind_a: Int, kind_b: Int, a: T, b: T, sa: T, sb: T
) raises:
    """_scaled_block{1x128,128x128}_block{...}: the scale shapes and
    strides, in CUDA's convention (block scales are outer-dim-major; the
    128x128 side is (round_up(K/128, 4), outer/128) with K blocks
    contiguous)."""
    var m = a.dim(0)
    var k = a.dim(1)
    var n = b.dim(1)
    var kb = _cdiv(k, 128)
    var kb4 = _round_up(kb, 4)

    @__parameter
    def check_1x128(s: T, outer: Int, name: StaticString) raises:
        if (
            s.rank != 2
            or s.dim(0) != outer
            or s.dim(1) != kb
            or s.stype != ST_FLOAT32
        ):
            raise Error(
                name,
                " must have shape ",
                outer,
                " x ",
                kb,
                " Float elements, got ",
                _shape_str(s),
            )
        if not (
            s.stride(0) == 1
            and (s.stride(1) == outer or (s.dim(1) == 1 and s.stride(1) == 1))
        ):
            raise Error(
                name, " strides must be (1, ", outer, "); got: ", _stride_str(s)
            )

    @__parameter
    def check_128x128(s: T, outer: Int, name: StaticString) raises:
        var ob = _cdiv(outer, 128)
        if (
            s.rank != 2
            or s.dim(0) != kb4
            or s.dim(1) != ob
            or s.stype != ST_FLOAT32
        ):
            raise Error(
                name,
                " must have shape ",
                kb4,
                " x ",
                ob,
                " Float elements, got ",
                _shape_str(s),
            )
        if not (
            s.stride(0) == 1
            and (s.stride(1) == kb4 or (s.dim(1) == 1 and s.stride(1) == 1))
        ):
            raise Error(
                name, " must have strides (1, ", kb4, "); got ", _stride_str(s)
            )

    if kind_a == BLOCK_1X128:
        check_1x128(sa, m, "scale_a")
    else:
        check_128x128(sa, m, "scale_a")
    if kind_b == BLOCK_1X128:
        check_1x128(sb, n, "scale_b")
    else:
        check_128x128(sb, n, "scale_b")


def _v2_scale_view(kind: Int, s: T) raises -> _Sc:
    """A v2 scale as `Dequant` indexes it, (outer index or block, K block):
    the 128x128 layout is stored (K blocks, outer blocks), so transposed."""
    if kind == BLOCK_128X128:
        return _sc_t(s)
    return _sc(s)


def _scaled_mm_v2(args: Values, mut dest: T) raises:
    var a = v_tensor(args[unsafe_offset=0])
    var b = v_tensor(args[unsafe_offset=1])
    var scale_a = v_tensor_list(args[unsafe_offset=2])
    var recipe_a = _ints(IntList(args[unsafe_offset=3]))
    var swizzle_a = _ints(IntList(args[unsafe_offset=4]))
    var scale_b = v_tensor_list(args[unsafe_offset=5])
    var recipe_b = _ints(IntList(args[unsafe_offset=6]))
    var swizzle_b = _ints(IntList(args[unsafe_offset=7]))
    var bias = v_opt_tensor(args[unsafe_offset=8])
    var contraction = _ints(IntList(args[unsafe_offset=10]))
    # The structured op's generated device check runs before its meta
    # function: nothing resizes `out` or returns early on mixed devices.
    var devs: List[T] = [dest.copy(), a.copy(), b.copy()]
    if bias:
        devs.append(bias.value().copy())
    for i in range(len(scale_a)):
        devs.append(scale_a[i].copy())
    for i in range(len(scale_b)):
        devs.append(scale_b[i].copy())
    _check_same_device(devs)
    _v2_validate(
        a,
        b,
        scale_a,
        recipe_a,
        swizzle_a,
        scale_b,
        recipe_b,
        swizzle_b,
        bias,
        contraction,
    )
    var m = a.dim(0)
    var k = a.dim(1)
    var n = b.dim(1)
    resize_out(dest, _shape2(m, n), 2)
    if m == 0 or k == 0 or n == 0:
        if k == 0:
            fill_value(dest, 0.0)
        return
    if k % 16 != 0:
        raise Error(
            (
                "Expected trailing dimension of mat1 to be divisible by 16 but"
                " got mat1 shape: ("
            ),
            _mat_str(a),
            ").",
        )
    if b.dim(0) % 16 != 0 or n % 16 != 0:
        raise Error("mat2 shape (", _mat_str(b), ") must be divisible by 16")
    _check_mat_dtypes(a, b, "mat_a", "mat_b")
    _check_bias(bias, dest.stype)
    _check_out_dtype(dest.stype)
    # find_scaled_gemm_impl over CUDA's table: the deepseek and MX recipes
    # take float8_e4m3fn operands only.
    var kind_a = recipe_a[0]
    var kind_b = recipe_b[0]
    if len(scale_a) != 1 or len(scale_b) != 1:
        # Reached by the DeepSeek recipes, which the meta does not validate:
        # find_scaled_gemm_impl matches no implementation (torch 2.14's
        # message then reads scale_a[0] of an empty list).
        raise Error(
            "Invalid scaling configuration. Expected one scale per side, got ",
            len(scale_a),
            " and ",
            len(scale_b),
        )
    var sa = scale_a[0].copy()
    var sb = scale_b[0].copy()
    var e4m3_both = (
        a.dtype == DType.float8_e4m3fn and b.dtype == DType.float8_e4m3fn
    )
    var found = False
    if kind_a == TENSOR_WISE and kind_b == TENSOR_WISE:
        found = sa.stype == ST_FLOAT32 and sb.stype == ST_FLOAT32
    elif kind_a == ROW_WISE and kind_b == ROW_WISE:
        found = sa.stype == ST_FLOAT32 and sb.stype == ST_FLOAT32
    elif kind_a == BLOCK_1X32 and kind_b == BLOCK_1X32:
        found = (
            e4m3_both
            and sa.stype == ST_FLOAT8_E8M0FNU
            and sb.stype == ST_FLOAT8_E8M0FNU
        )
    elif (kind_a == BLOCK_1X128 or kind_a == BLOCK_128X128) and (
        kind_b == BLOCK_1X128 or kind_b == BLOCK_128X128
    ):
        found = (
            e4m3_both
            and sa.stype == ST_FLOAT32
            and sb.stype == ST_FLOAT32
            and not (kind_a == BLOCK_128X128 and kind_b == BLOCK_128X128)
        )
    # Every fp8 recipe check (check_*_recipe) wants exactly one scale and
    # one recipe per side; extra entries select no implementation.
    if (
        len(scale_a) != 1
        or len(scale_b) != 1
        or len(recipe_a) != 1
        or len(recipe_b) != 1
    ):
        found = False
    if not found:
        _v2_invalid(a, b, sa, sb)
    if kind_a == TENSOR_WISE:
        pass  # numel and dtype checked by the validation above
    elif kind_a == ROW_WISE:
        if sa.rank != 2 or sa.dim(0) != m or sa.dim(1) != 1:
            raise Error(
                "scale_a must have shape [",
                m,
                ", 1], got [",
                _shape_str(sa),
                "]",
            )
        if not (sa.stride(1) == 1 or sa.dim(1) == 1):
            raise Error(
                "expected scale_a.stride(1) to be 1, but got ", sa.stride(1)
            )
        if sb.rank != 2 or sb.stride(1) != 1:
            raise Error(
                "expected scale_b.stride(1) to be 1, but got ",
                sb.stride(1) if sb.rank == 2 else 0,
            )
    elif kind_a == BLOCK_1X32:
        pass  # element counts, contiguity and swizzle checked above
    else:
        _v2_deepseek_checks(kind_a, kind_b, a, b, sa, sb)
    if kind_a == ROW_WISE:
        var fa = _flat_scale(sa, m)
        var fb = _flat_scale(sb, n)
        _run_recipe(dest, a, b, kind_a, _sc(fa.t), kind_b, _sc(fb.t), bias)
        _ = fa^
        _ = fb^
        return
    _run_recipe(
        dest,
        a,
        b,
        kind_a,
        _v2_scale_view(kind_a, sa),
        kind_b,
        _v2_scale_view(kind_b, sb),
        bias,
    )


# aten::_scaled_mm_v2(Tensor self, Tensor mat2, Tensor[] scale_a,
#   int[] recipe_a, int[] swizzle_a, Tensor[] scale_b, int[] recipe_b,
#   int[] swizzle_b, Tensor? bias, ScalarType? out_dtype,
#   int[] contraction_dim=[], bool use_fast_accum=False) -> Tensor
def op_scaled_mm_v2(
    args: Values, n_args: Int, rets: Values, n_rets: Int
) raises:
    var a = v_tensor(args[unsafe_offset=0])
    var out_st = v_dtype_or(args[unsafe_offset=9], a.stype)
    _ = max_dtype(out_st)
    var out = own(new_tensor(_shape2(0, 0), 2, out_st, a.device))
    _scaled_mm_v2(args, out.t)
    ret_tensor(rets, 0, out.take())


# aten::_scaled_mm_v2.out(..., bool use_fast_accum=False, *, Tensor(a!) out)
def op_scaled_mm_v2_out(
    args: Values, n_args: Int, rets: Values, n_rets: Int
) raises:
    var a = v_tensor(args[unsafe_offset=0])
    var dest = v_tensor(args[unsafe_offset=12])
    var out_st = v_dtype_or(args[unsafe_offset=9], a.stype)
    check_out_as(dest, out_st, a)
    if _aliases_any(dest, _arg_tensors(args, [0, 1, 2, 5, 8])):
        var tmp = own(new_tensor(_shape2(0, 0), 2, out_st, a.device))
        _scaled_mm_v2(args, tmp.t)
        _land(dest, tmp.t)
        _ = tmp^
    else:
        _scaled_mm_v2(args, dest)
    ret_ref(rets, 0, v_tensor(args[unsafe_offset=12]))


# --- aten::_scaled_grouped_mm (v1, v2) --------------------------------------
# GroupedBlas.cpp. Only the rowwise recipe (float32 scales) runs: it is the
# one an H100 build has (`_f8_f8_bf16_rowwise_grouped_mm`); MX grouped GEMMs
# need MSLK on CUDA too and are declined here. Each group is one
# `_scaled_gemm` over views, the way ATen's `_grouped_mm_fallback` loops;
# `offs` is read back to the host once (a synchronizing 4-byte-per-group
# copy).


def _check_grouped_e4m3(a: T, b: T) raises:
    """_f8_f8_bf16_rowwise_grouped_mm_cuda: both operands e4m3fn (CUDA
    raises ValueError; RuntimeError here, same text)."""
    if a.dtype != DType.float8_e4m3fn:
        raise Error("Expected mat_a to be Float8_e4m3 matrix got ", _tname(a))
    if b.dtype != DType.float8_e4m3fn:
        raise Error("Expected mat_b to be Float8_e4m3 matrix got ", _tname(b))


def _valid_strides_transposed(m: T) raises -> Bool:
    """check_valid_strides_and_return_transposed (GroupedMMUtils.h)."""
    var e = m.rank - 1
    var align = 16 // m.itemsize
    if m.ptr % 16 != 0:
        raise Error("expected data_ptr to be aligned to 16 bytes")
    if m.rank == 3 and m.stride(0) % align != 0:
        raise Error(
            (
                "batch stride must yield 16-byte-aligned per-batch pointers,"
                " got stride(0)="
            ),
            m.stride(0),
            " with element size ",
            m.itemsize,
        )
    if m.stride(e - 1) == 1 and m.stride(e) >= max(1, m.dim(e - 1)):
        if m.stride(e) % align != 0:
            raise Error("strides should be multiple of 16 bytes")
        return True
    if m.stride(e) == 1 and m.stride(e - 1) >= max(1, m.dim(e)):
        if m.stride(e - 1) % align != 0:
            raise Error("strides should be multiple of 16 bytes")
        return False
    raise Error(
        "Invalid strides/sizes, got ",
        _stride_str(m),
        " for strides and ",
        _shape_str(m),
        " for sizes",
    )


def _check_rowwise_grouped_scale(
    mat: T, scale: T, dim: Int, arg: Int, multiplier: Int
) raises:
    """_check_scales_fp8_rowwise."""
    if mat.rank == 2:
        if scale.rank != 1:
            raise Error(
                "scale must be a 1D tensor, but got ",
                scale.rank,
                "D, arg ",
                arg,
            )
        if not scale.contig:
            raise Error("scale must be contiguous for arg ", arg)
        if scale.dim(0) != mat.dim(dim) * multiplier:
            raise Error("scale must have the same length as mat for arg ", arg)
    else:
        if scale.rank != 2:
            raise Error(
                "scale must be a 2D tensor, but got ",
                scale.rank,
                "D for arg ",
                arg,
            )
        if scale.stride(1) != 1:
            raise Error(
                "scale must be contiguous in the last dimension for arg ", arg
            )
        if scale.dim(0) != mat.dim(0):
            raise Error(
                "scale must have the same batch dimension as mat for arg ", arg
            )
        if scale.dim(1) != mat.dim(1 + dim):
            raise Error(
                "scale must have the same first dimension as mat for arg ", arg
            )


def _grouped_common_checks(
    a: T, b: T, offs: Optional[T], has_bias: Bool
) raises:
    """The recipe-independent checks both overloads run, in v1's order."""
    if _valid_strides_transposed(a):
        raise Error("Expected mat1 to not be transposed")
    if not _valid_strides_transposed(b):
        raise Error("Expected mat2 to be transposed")
    if a.rank != 2 and a.rank != 3:
        raise Error("mat_a has to be 2 or 3d")
    if b.rank != 2 and b.rank != 3:
        raise Error("mat_b has to be 2 or 3d")
    var both_2d = a.rank == 2 and b.rank == 2
    if not both_2d and a.dim(a.rank - 1) != b.dim(b.rank - 2):
        raise Error("contraction dimension of mat_a and mat_b must match")
    if a.dim(a.rank - 1) % 16 != 0:
        raise Error(
            (
                "Expected trailing dimension of mat_a to be divisible by 16 but"
                " got mat1 shape: ("
            ),
            _shape_str(a),
            ").",
        )
    if b.dim(b.rank - 2) % 16 != 0 or b.dim(b.rank - 1) % 16 != 0:
        raise Error(
            "Expected mat_b shape to be divisible by 16 but got mat_b shape: (",
            _shape_str(b),
            ").",
        )
    if has_bias:
        raise Error("Bias not supported yet")
    if Bool(offs) != (a.rank == 2 or b.rank == 2):
        raise Error("Have to provide offsets if there is a 2d matrix")
    if offs:
        if offs.value().rank != 1:
            raise Error("offs has to be 1D")
        if offs.value().stype != ST_INT32:
            raise Error("Offsets have to be int32")
        if offs.value().stride(0) != 1:
            raise Error("Offsets must have stride (1,)")


def _read_offs(offs: T) raises -> List[Int]:
    var out = List[Int]()
    var n = offs.dim(0)
    if n == 0:
        return out^
    var host = own(cpu_empty(IndexList[MAX_RANK](n), 1, ST_INT32))
    var dense = own_if_new(contiguous(offs), offs)
    var ctx = ctx_for(offs.device)
    copy_to_host(ctx, dense.t.ptr, host.t.ptr, n * 4)
    _ = ctx
    _ = dense^
    var p = Pointer[Int32, MutUntrackedOrigin](unsafe_from_address=host.t.ptr)
    for i in range(n):
        out.append(Int(p[unsafe_offset=i]))
    _ = host^
    return out^


def _view2(
    t: T, offset_elems: Int, r: Int, c: Int, rs: Int, cs: Int
) raises -> Owned:
    return own(
        view_strided(
            t, _shape2(r, c), _strides2(rs, cs), 2, t.offset + offset_elems
        )
    )


def _view1(t: T, offset_elems: Int, n: Int, st: Int) raises -> Owned:
    var shape = IndexList[MAX_RANK](1)
    shape[MAX_RANK - 1] = n
    var strides = IndexList[MAX_RANK](0)
    strides[MAX_RANK - 1] = st
    return own(view_strided(t, shape, strides, 1, t.offset + offset_elems))


def _grouped_dims(a: T, b: T, offs: Optional[T]) raises -> List[Int]:
    """compute_grouped_gemm_output_size_stride's sizes."""
    var g = offs.value().dim(0) if offs else 0
    var dims: List[Int]
    if a.rank == 2 and b.rank == 2:
        dims = [g, a.dim(0), b.dim(1)]
    elif a.rank == 2:
        if g != b.dim(0):
            raise Error("matrix batch sizes have to match")
        dims = [a.dim(0), b.dim(2)]
    elif b.rank == 2:
        if g != a.dim(0):
            raise Error("matrix batch sizes have to match")
        dims = [a.dim(1), b.dim(1)]
    else:
        if a.dim(0) != b.dim(0):
            raise Error("batched dimension has to match")
        dims = [a.dim(0), a.dim(1), b.dim(2)]
    return dims^


def _grouped_shape(dims: List[Int]) -> IndexList[MAX_RANK]:
    var shape = IndexList[MAX_RANK](1)
    for i in range(len(dims)):
        shape[MAX_RANK - len(dims) + i] = dims[i]
    return shape


def _grouped_out(a: T, b: T, offs: Optional[T]) raises -> Owned:
    """create_grouped_gemm_output_tensor for a bfloat16 result: the last dim
    padded to 16 bytes in the strides."""
    var dims = _grouped_dims(a, b, offs)
    var rank = len(dims)
    var shape = IndexList[MAX_RANK](1)
    var strides = IndexList[MAX_RANK](0)
    var acc = 1
    for i in range(rank - 1, -1, -1):
        shape[MAX_RANK - rank + i] = dims[i]
        strides[MAX_RANK - rank + i] = acc
        var extent = dims[i]
        if i == rank - 1:
            extent = _round_up(extent, 8)  # 16 bytes of bfloat16
        acc *= max(extent, 1)
    return own(new_strided(shape, strides, rank, ST_BFLOAT16, a.device))


def _grouped_device_check(
    a: T, b: T, sa: T, sb: T, offs: Optional[T], dest: T
) raises:
    """checkAllSameGPU: no host scale or offset reaches a device kernel."""
    var devs: List[T] = [dest.copy(), a.copy(), b.copy(), sa.copy(), sb.copy()]
    if offs:
        devs.append(offs.value().copy())
    _check_same_device(devs)


def _grouped_rowwise(
    a: T, b: T, sa: T, sb: T, offs: Optional[T], dest: T
) raises:
    """Every group of a rowwise grouped GEMM into `dest`."""
    _grouped_device_check(a, b, sa, sb, offs, dest)
    var none = Optional[T](None)

    @__parameter
    def one(av: T, bv: T, ov: T, sav: T, sbv: T) raises:
        _scaled_gemm(
            ov,
            av,
            bv,
            _dq_none(),
            _dq_none(),
            _Outer(sav.ptr, sav.stride(0) if sav.rank >= 1 else 0),
            _Outer(sbv.ptr, sbv.stride(0) if sbv.rank >= 1 else 0),
            none,
            order=ORDER_A_THEN_B,
            grouped=True,
        )

    if a.rank == 3 and b.rank == 3:
        for g in range(a.dim(0)):
            var av = _view2(
                a, g * a.stride(0), a.dim(1), a.dim(2), a.stride(1), a.stride(2)
            )
            var bv = _view2(
                b, g * b.stride(0), b.dim(1), b.dim(2), b.stride(1), b.stride(2)
            )
            var ov = _view2(
                dest,
                g * dest.stride(0),
                dest.dim(1),
                dest.dim(2),
                dest.stride(1),
                dest.stride(2),
            )
            var sav = _view1(sa, g * sa.stride(0), sa.dim(1), sa.stride(1))
            var sbv = _view1(sb, g * sb.stride(0), sb.dim(1), sb.stride(1))
            one(av.t, bv.t, ov.t, sav.t, sbv.t)
            _ = av^
            _ = bv^
            _ = ov^
            _ = sav^
            _ = sbv^
        return
    var ends = _read_offs(offs.value())
    var start = 0
    for g in range(len(ends)):
        var end = ends[g]
        if end < start:
            raise Error("offsets must be non-decreasing")
        if a.rank == 2 and b.rank == 2:
            # Groups along K: out[g] = A[:, s:e] @ B[s:e, :].
            if end > a.dim(1) or end > b.dim(0):
                raise Error("offsets exceed the grouped dimension")
            var m = a.dim(0)
            var n = b.dim(1)
            var av = _view2(
                a, start * a.stride(1), m, end - start, a.stride(0), a.stride(1)
            )
            var bv = _view2(
                b, start * b.stride(0), end - start, n, b.stride(0), b.stride(1)
            )
            var ov = _view2(
                dest, g * dest.stride(0), m, n, dest.stride(1), dest.stride(2)
            )
            var sav = _view1(sa, g * m * sa.stride(0), m, sa.stride(0))
            var sbv = _view1(sb, g * n * sb.stride(0), n, sb.stride(0))
            one(av.t, bv.t, ov.t, sav.t, sbv.t)
            _ = av^
            _ = bv^
            _ = ov^
            _ = sav^
            _ = sbv^
        elif a.rank == 2:
            # Groups along M: out[s:e] = A[s:e] @ B[g].
            if end > a.dim(0):
                raise Error("offsets exceed the grouped dimension")
            var av = _view2(
                a,
                start * a.stride(0),
                end - start,
                a.dim(1),
                a.stride(0),
                a.stride(1),
            )
            var bv = _view2(
                b, g * b.stride(0), b.dim(1), b.dim(2), b.stride(1), b.stride(2)
            )
            var ov = _view2(
                dest,
                start * dest.stride(0),
                end - start,
                dest.dim(1),
                dest.stride(0),
                dest.stride(1),
            )
            var sav = _view1(
                sa, start * sa.stride(0), end - start, sa.stride(0)
            )
            var sbv = _view1(sb, g * sb.stride(0), sb.dim(1), sb.stride(1))
            one(av.t, bv.t, ov.t, sav.t, sbv.t)
            _ = av^
            _ = bv^
            _ = ov^
            _ = sav^
            _ = sbv^
        else:
            # Groups along N: out[:, s:e] = A[g] @ B[:, s:e].
            if end > b.dim(1):
                raise Error("offsets exceed the grouped dimension")
            var av = _view2(
                a, g * a.stride(0), a.dim(1), a.dim(2), a.stride(1), a.stride(2)
            )
            var bv = _view2(
                b,
                start * b.stride(1),
                b.dim(0),
                end - start,
                b.stride(0),
                b.stride(1),
            )
            var ov = _view2(
                dest,
                start * dest.stride(1),
                dest.dim(0),
                end - start,
                dest.stride(0),
                dest.stride(1),
            )
            var sav = _view1(sa, g * sa.stride(0), sa.dim(1), sa.stride(1))
            var sbv = _view1(
                sb, start * sb.stride(0), end - start, sb.stride(0)
            )
            one(av.t, bv.t, ov.t, sav.t, sbv.t)
            _ = av^
            _ = bv^
            _ = ov^
            _ = sav^
            _ = sbv^
        start = end


# aten::_scaled_grouped_mm(Tensor self, Tensor mat2, Tensor scale_a,
#   Tensor scale_b, Tensor? offs=None, Tensor? bias=None,
#   Tensor? scale_result=None, ScalarType? out_dtype=None,
#   bool use_fast_accum=False) -> Tensor
def op_scaled_grouped_mm(
    args: Values, n_args: Int, rets: Values, n_rets: Int
) raises:
    var a = v_tensor(args[unsafe_offset=0])
    var b = v_tensor(args[unsafe_offset=1])
    var sa = v_tensor(args[unsafe_offset=2])
    var sb = v_tensor(args[unsafe_offset=3])
    var offs = v_opt_tensor(args[unsafe_offset=4])
    var bias = v_opt_tensor(args[unsafe_offset=5])
    var scale_result = v_opt_tensor(args[unsafe_offset=6])
    var out_st = v_dtype_or(args[unsafe_offset=7], ST_BFLOAT16)
    _grouped_common_checks(a, b, offs, Bool(bias))
    if scale_result:
        raise Error("Scale result not supported yet")
    var f32 = sa.stype == ST_FLOAT32 and sb.stype == ST_FLOAT32
    var mx = sa.stype == ST_FLOAT8_E8M0FNU and sb.stype == ST_FLOAT8_E8M0FNU
    if not f32 and not mx:
        raise Error(
            "For FP8 tensorwise and rowwise, both scales must both be float32"
            " tensors. For MXFP8, scales must both be float8_e8m0fnu tensors."
        )
    if mx:
        unsupported(
            "_scaled_grouped_mm with MXFP8 (float8_e8m0fnu) scales: CUDA runs"
            " it through MSLK only"
        )
    var mult = offs.value().dim(0) if a.rank == 2 and b.rank == 2 else 1
    _check_rowwise_grouped_scale(a, sa, 0, 0, mult)
    _check_rowwise_grouped_scale(b, sb, 1, 1, mult)
    if out_st != ST_BFLOAT16:
        raise Error(
            "Only bf16 high precision output types are supported for grouped"
            " gemm"
        )
    _check_grouped_e4m3(a, b)
    var out = _grouped_out(a, b, offs)
    _grouped_rowwise(a, b, sa, sb, offs, out.t)
    ret_tensor(rets, 0, out.take())


def _grouped_v2_checks(args: Values) raises:
    """validate_scaled_grouped_mm_v2_inputs, then the CUDA IMPL's checks up
    to the recipe: only rowwise (float32 scales) is implemented."""
    var a = v_tensor(args[unsafe_offset=0])
    var b = v_tensor(args[unsafe_offset=1])
    var scale_a = v_tensor_list(args[unsafe_offset=2])
    var recipe_a = _ints(IntList(args[unsafe_offset=3]))
    var scale_b = v_tensor_list(args[unsafe_offset=5])
    var recipe_b = _ints(IntList(args[unsafe_offset=6]))
    var offs = v_opt_tensor(args[unsafe_offset=8])
    var bias = v_opt_tensor(args[unsafe_offset=9])
    var out_st = v_dtype_or(args[unsafe_offset=10], ST_BFLOAT16)
    var contraction = _ints(IntList(args[unsafe_offset=11]))
    if a.rank != 2 and a.rank != 3:
        raise Error("mat_a has to be 2 or 3d")
    if b.rank != 2 and b.rank != 3:
        raise Error("mat_b has to be 2 or 3d")
    if (a.rank != 2 or b.rank != 2) and len(contraction) > 0:
        if len(contraction) != 2:
            raise Error("contraction_dim must have exactly 2 elements")
        if contraction[0] != -1 or contraction[1] != -2:
            raise Error("Currently contraction dims must be (-1, -2) only")
    if out_st != ST_BFLOAT16:
        raise Error(
            "Only bf16 high precision output types are supported for grouped"
            " gemm"
        )
    _grouped_common_checks(a, b, offs, Bool(bias))
    var rowwise = (
        is_float8(a.dtype)
        and is_float8(b.dtype)
        and len(scale_a) == 1
        and len(scale_b) == 1
        and len(recipe_a) == 1
        and len(recipe_b) == 1
        and recipe_a[0] == ROW_WISE
        and recipe_b[0] == ROW_WISE
        and scale_a[0].stype == ST_FLOAT32
        and scale_b[0].stype == ST_FLOAT32
    )
    if not rowwise:
        if len(recipe_a) >= 1 and recipe_a[0] == BLOCK_1X32:
            unsupported(
                "_scaled_grouped_mm_v2 MXFP8 recipe: CUDA runs it through MSLK"
                " only"
            )
        raise Error("No gemm implementation was found")
    var mult = offs.value().dim(0) if a.rank == 2 and b.rank == 2 else 1
    _check_rowwise_grouped_scale(a, scale_a[0], 0, 0, mult)
    _check_rowwise_grouped_scale(b, scale_b[0], 1, 1, mult)
    _check_grouped_e4m3(a, b)


# aten::_scaled_grouped_mm_v2(Tensor self, Tensor mat2, Tensor[] scale_a,
#   int[] recipe_a, int[] swizzle_a, Tensor[] scale_b, int[] recipe_b,
#   int[] swizzle_b, Tensor? offs=None, Tensor? bias=None,
#   ScalarType? out_dtype=None, int[] contraction_dim=[],
#   bool use_fast_accum=False) -> Tensor
def op_scaled_grouped_mm_v2(
    args: Values, n_args: Int, rets: Values, n_rets: Int
) raises:
    _grouped_v2_checks(args)
    var a = v_tensor(args[unsafe_offset=0])
    var b = v_tensor(args[unsafe_offset=1])
    var offs = v_opt_tensor(args[unsafe_offset=8])
    var out = _grouped_out(a, b, offs)
    _grouped_rowwise(
        a,
        b,
        v_tensor_list(args[unsafe_offset=2])[0],
        v_tensor_list(args[unsafe_offset=5])[0],
        offs,
        out.t,
    )
    ret_tensor(rets, 0, out.take())


# aten::_scaled_grouped_mm_v2.out(..., *, Tensor(a!) out) -> Tensor(a!)
def op_scaled_grouped_mm_v2_out(
    args: Values, n_args: Int, rets: Values, n_rets: Int
) raises:
    _grouped_v2_checks(args)
    var a = v_tensor(args[unsafe_offset=0])
    var b = v_tensor(args[unsafe_offset=1])
    var offs = v_opt_tensor(args[unsafe_offset=8])
    var dest = v_tensor(args[unsafe_offset=13])
    check_out_as(dest, ST_BFLOAT16, a)
    _grouped_device_check(
        a,
        b,
        v_tensor_list(args[unsafe_offset=2])[0],
        v_tensor_list(args[unsafe_offset=5])[0],
        offs,
        dest,
    )
    var dims = _grouped_dims(a, b, offs)
    if _aliases_any(dest, _arg_tensors(args, [0, 1, 2, 5, 8])):
        var tmp = _grouped_out(a, b, offs)
        _grouped_rowwise(
            a,
            b,
            v_tensor_list(args[unsafe_offset=2])[0],
            v_tensor_list(args[unsafe_offset=5])[0],
            offs,
            tmp.t,
        )
        _land(dest, tmp.t)
        _ = tmp^
    else:
        resize_out(dest, _grouped_shape(dims), len(dims))
        _grouped_rowwise(
            a,
            b,
            v_tensor_list(args[unsafe_offset=2])[0],
            v_tensor_list(args[unsafe_offset=5])[0],
            offs,
            dest,
        )
    ret_ref(rets, 0, v_tensor(args[unsafe_offset=13]))


def register_scaled_mm(site: Site) raises:
    impl[op_scaled_mm, "_scaled_mm"](site)
    impl[op_scaled_mm_out, "_scaled_mm.out"](site)
    impl[op_scaled_mm_v2, "_scaled_mm_v2"](site)
    impl[op_scaled_mm_v2_out, "_scaled_mm_v2.out"](site)
    impl[op_scaled_grouped_mm, "_scaled_grouped_mm"](site)
    impl[op_scaled_grouped_mm_v2, "_scaled_grouped_mm_v2"](site)
    impl[op_scaled_grouped_mm_v2_out, "_scaled_grouped_mm_v2.out"](site)
