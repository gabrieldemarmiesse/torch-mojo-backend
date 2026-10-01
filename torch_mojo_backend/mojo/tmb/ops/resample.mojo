"""ATen ops: resample group (see agents_docs/native_backend.md).

Reflection / replication padding (1-d, 2-d, 3-d) and nearest, nearest-exact,
linear, bilinear, bicubic, trilinear and antialiased bilinear / bicubic /
lanczos upsampling,
each with its `.out` overload, its backward and the backward's
`.grad_input` overload -- the ops behind `F.pad(mode="reflect" |
"replicate")` and `F.interpolate`. The `.vec`
overloads are CompositeImplicitAutograd over these and need nothing here.
Also `grid_sampler_2d` / `grid_sampler_3d` (F.grid_sample), their `.out`,
backwards and backward `.out`, on their own family `grid_sample`
(tmb/kernels/grid_sample/entry.mojo); see the section at the bottom.

One kernel family, `resample` (tmb/kernels/resample/entry.mojo), serves all
of them: every op views its operands as contiguous (planes, D, H, W) with the
missing spatial extents set to 1, and the MODE / RANK / REFLECT defines pick
the formula. Shape checks and error messages follow the structured meta
functions (UpSample.h, Upsample*.cpp, ReflectionPad.cpp,
ReplicationPadding.cpp, Padding.h at v2.14.0); the scales follow the CUDA
host code (compute_scales_value, area_pixel_compute_scale), rounded to the
kernel's accumulation type here so the device reads them exactly.

Outputs are contiguous: CUDA keeps a channels-last input's memory format for
upsampling, the values are the same.
"""
from std.utils import IndexList

from tmb.backend.abi import (
    Owned,
    ST_UINT8,
    TAG_BOOL_LIST,
    TAG_NONE,
    T,
    Value,
    Values,
    IntList,
    alert_not_deterministic,
    index_error,
    is_floating,
    new_like,
    new_tensor,
    own,
    own_if_new,
    ret_owned,
    ret_ref,
    unsupported,
    v_bool,
    v_f64,
    v_int,
    v_is_none,
    v_tensor,
)
from tmb.backend.device import ctx_for, ctx_ptr, dev
from tmb.backend.kernel_call import KernelCall
from tmb.kernels.common.op_utils import MAX_RANK, _f64_slot
from tmb.ops.common import (
    OVERLAP_FULL,
    OVERLAP_PARTIAL,
    assert_no_internal_overlap,
    assert_no_overlap,
    assert_no_partial_overlap,
    cast_to,
    check_out,
    check_out_as,
    device_str,
    fill_value,
    overlap_status,
    resized_geometry,
    check_out,
    same_view,
    shares_storage,
    contiguous,
    copy_strided_into,
    resize_out,
)
from tmb.ops.data_movement import _scalar_type_name
from tmb.backend.registry import Site, impl

comptime NEAREST = 0
comptime NEAREST_EXACT = 1
comptime LINEAR = 2
comptime CUBIC = 3
comptime BILINEAR_AA = 4
comptime BICUBIC_AA = 5
comptime LANCZOS_AA = 6


def _sizes_str(t: T) -> String:
    """`IntArrayRef` as torch prints it: [2, 3, 4]."""
    var s = String("[")
    for i in range(t.rank):
        if i:
            s += ", "
        s += String(t.dim(i))
    return s + "]"


def _list_str(l: List[Int]) -> String:
    var s = String("[")
    for i in range(len(l)):
        if i:
            s += ", "
        s += String(l[i])
    return s + "]"


def _pick(i: Int, a: StaticString, b: StaticString, c: StaticString) -> String:
    """The i-th of three names (spatial axes D, H, W)."""
    if i == 0:
        return String(a)
    if i == 1:
        return String(b)
    return String(c)


def _shape(dims: List[Int]) -> IndexList[MAX_RANK]:
    var shape = IndexList[MAX_RANK](1)
    for i in range(len(dims)):
        shape[MAX_RANK - len(dims) + i] = dims[i]
    return shape


def _require_mojo(t: T, what: String) raises:
    if not t.on_mojo():
        unsupported(what + ": operand is not on the mojo device")


def _dest(
    args: Values,
    i: Int,
    like: T,
    dims: List[Int],
    inputs: List[T],
    internal_check: Bool,
    copy_shortcut: Bool,
) raises -> T:
    """The caller's `out=` / `grad_input=` tensor, checked and resized as
    the "Upsample / pad" rows of the overlap table in tmb/ops/common.mojo
    say (the structured meta resizes the out first; only what the CUDA
    kernel then does checks anything): `copy_shortcut` is the kernel's
    same-size `output.copy_(input)` (copy_'s internal-overlap check, then
    its partial-overlap check against the input); `internal_check` a kernel
    that copies a non-contiguous out back with copy_.

    The out is resized for real, before the op runs anything, exactly as
    ATen does: the caller then re-reads its inputs (a resize may move a
    storage they share) and executes ATen's sequence on this tensor -- no
    temporary unless ATen uses one -- so an out aliasing an input sees the
    same values it would on CUDA."""
    var dst = v_tensor(args[unsafe_offset=i])
    check_out(dst, like)
    var shape = _shape(dims)
    var post = resized_geometry(dst, shape, len(dims))
    if copy_shortcut:
        assert_no_internal_overlap(post)
        for k in range(len(inputs)):
            assert_no_partial_overlap(post, inputs[k])
    elif internal_check and not post.contig:
        assert_no_internal_overlap(post)
    resize_out(dst, shape, len(dims))
    return dst^


def _into[
    MODE: Int, RANK: Int, BACKWARD: Bool
](
    dst: T,
    src: T,
    identity: Bool,
    in_dims: List[Int],
    out_dims: List[Int],
    align: Bool,
    scales: List[Float64],
) raises:
    """Run the upsample kernel (or its identity copy) into `dst` -- through
    a temporary copied back when `dst` is not contiguous, as the CUDA
    kernels' `output_c` does."""
    if dst.numel == 0:
        return
    var target = Optional[Owned](None)
    var into = dst.copy()
    if not dst.contig:
        target = own(new_like(dst))
        into = target.value().t.copy()
    if identity:
        copy_strided_into(into, src)
    else:
        _run_upsample[MODE, RANK, BACKWARD](
            into, src, in_dims, out_dims, align, scales
        )
    if target:
        copy_strided_into(dst, into)
    _ = target^


# ---------------------------------------------------------------------------
# Upsampling
# ---------------------------------------------------------------------------


def _up_name[MODE: Int, RANK: Int]() -> String:
    comptime if MODE == NEAREST:
        return String("upsample_nearest", RANK, "d")
    elif MODE == NEAREST_EXACT:
        return String("_upsample_nearest_exact", RANK, "d")
    elif MODE == CUBIC:
        return String("upsample_bicubic2d")
    elif MODE == BILINEAR_AA:
        return String("_upsample_bilinear2d_aa")
    elif MODE == BICUBIC_AA:
        return String("_upsample_bicubic2d_aa")
    elif MODE == LANCZOS_AA:
        return String("_upsample_lanczos2d_aa")
    else:
        comptime if RANK == 1:
            return String("upsample_linear1d")
        elif RANK == 2:
            return String("upsample_bilinear2d")
        else:
            return String("upsample_trilinear3d")


def _up_check[
    RANK: Int
](input_size: List[Int], osize: IntList) raises -> List[Int]:
    """upsample_{1,2,3}d_common_check: the full output size."""
    if len(osize) != RANK:
        raise Error(
            "It is expected output_size equals to ",
            RANK,
            ", but got size ",
            len(osize),
        )
    if len(input_size) != RANK + 2:
        raise Error(
            "It is expected input_size equals to ",
            RANK + 2,
            ", but got size ",
            len(input_size),
        )
    var positive = True
    for k in range(RANK):
        if input_size[2 + k] <= 0 or osize[k] <= 0:
            positive = False
    if not positive:
        var ins = String()
        var outs = String()
        for k in range(RANK):
            var label = _pick(3 - RANK + k, "D", "H", "W")
            if k:
                ins += ", "
                outs += ", "
            ins += String(label, ": ", input_size[2 + k])
            outs += String(label, ": ", osize[k])
        comptime if RANK == 1:
            raise Error(
                (
                    "Input and output sizes should be greater than 0, but got"
                    " input ("
                ),
                ins,
                ") and output (",
                outs,
                ")",
            )
        else:
            raise Error(
                (
                    "Input and output sizes should be greater than 0, but got"
                    " input ("
                ),
                ins,
                ") output (",
                outs,
                ")",
            )
    var full = List[Int]()
    full.append(input_size[0])
    full.append(input_size[1])
    for k in range(RANK):
        full.append(osize[k])
    return full^


def _opt_scale(v: Value) raises -> Float64:
    """An optional `float? scales_*` argument; 0 stands for None."""
    if v_is_none(v):
        return 0.0
    return v_f64(v)


def _kernel_scale[
    MODE: Int, BACKWARD: Bool
](
    in_size: Int, out_size: Int, align: Bool, scale: Float64, f64: Bool
) -> Float64:
    """The per-axis ratio the kernel reads, rounded to the type the CUDA
    kernel holds it in (float for nearest; accscalar_t otherwise)."""
    comptime if MODE <= NEAREST_EXACT:
        # compute_scales_value<float> / compute_scales_value_backwards<float>
        comptime if BACKWARD:
            if scale > 0.0:
                return Float64(Float32(scale))
            return Float64(Float32(out_size) / Float32(in_size))
        else:
            if scale > 0.0:
                return Float64(Float32(1.0 / scale))
            return Float64(Float32(in_size) / Float32(out_size))
    else:
        # area_pixel_compute_scale<accscalar_t>
        if align:
            if out_size <= 1:
                return 0.0
            if f64:
                return Float64(in_size - 1) / Float64(out_size - 1)
            return Float64(Float32(in_size - 1) / Float32(out_size - 1))
        if scale > 0.0:
            return 1.0 / scale if f64 else Float64(Float32(1.0 / scale))
        if f64:
            return Float64(in_size) / Float64(out_size)
        return Float64(Float32(in_size) / Float32(out_size))


def _up_dtype_ok[MODE: Int](t: T) -> Bool:
    if is_floating(t.stype):
        return True
    comptime if MODE <= NEAREST_EXACT:
        return t.stype == ST_UINT8
    else:
        return False


def _run_upsample[
    MODE: Int, RANK: Int, BACKWARD: Bool
](
    dst: T,
    src: T,
    in_dims: List[Int],
    out_dims: List[Int],
    align: Bool,
    scales: List[Float64],
) raises:
    """Launch Upsample / UpsampleBackward: `src` and `dst` contiguous, the
    dims full (N, C, spatial...) sizes of the forward input and output."""
    var geom = List[Int]()
    var sc = List[Int]()
    for k in range(3 - RANK):
        _ = k
        geom.append(1)
    for k in range(RANK):
        geom.append(in_dims[2 + k])
    for k in range(3 - RANK):
        _ = k
        geom.append(1)
    for k in range(RANK):
        geom.append(out_dims[2 + k])
    var f64 = src.dtype == DType.float64
    for k in range(3 - RANK):
        _ = k
        sc.append(_f64_slot(1.0))
    for k in range(RANK):
        sc.append(
            _f64_slot(
                _kernel_scale[MODE, BACKWARD](
                    in_dims[2 + k], out_dims[2 + k], align, scales[k], f64
                )
            )
        )
    var ctx = ctx_for(src.device)
    var call = KernelCall(
        "resample", "UpsampleBackward" if BACKWARD else "Upsample"
    )
    call.arg_dtype(0, src.dtype)
    call.flag("MODE", MODE)
    call.flag("RANK", RANK)
    call.int(dst.ptr)
    call.int(src.ptr)
    call.int(in_dims[0] * in_dims[1])
    call.tuple(geom)
    call.tuple(sc)
    call.int(1 if align else 0)
    call.int(ctx_ptr(ctx))
    call.run()
    _ = ctx


def _has_copy_shortcut[MODE: Int, RANK: Int]() -> Bool:
    """The CUDA kernels with a host-side same-size `copy_`:
    upsample_nearest2d / _upsample_nearest_exact2d and upsample_bilinear2d,
    forward and backward (UpSampleNearest2d.cu, UpSampleBilinear2d.cu)."""
    return RANK == 2 and (
        MODE == NEAREST or MODE == NEAREST_EXACT or MODE == LINEAR
    )


def _copies_back[MODE: Int, RANK: Int, BACKWARD: Bool]() -> Bool:
    """The CUDA kernels that compute a non-contiguous out into a temporary
    and `copy_` it back (so refuse an internally overlapping out): nearest
    2-d (both ways) and 3-d forward, bilinear 2-d backward, trilinear
    backward, antialiased forward."""
    comptime if MODE == NEAREST or MODE == NEAREST_EXACT:
        return RANK == 2 or (RANK == 3 and not BACKWARD)
    elif MODE == LINEAR:
        return BACKWARD and RANK >= 2
    elif MODE == BILINEAR_AA:
        return not BACKWARD
    else:
        return False


def _same_dims(a: List[Int], b: List[Int]) -> Bool:
    if len(a) != len(b):
        return False
    for k in range(len(a)):
        if a[k] != b[k]:
            return False
    return True


def _is_identity[
    MODE: Int, RANK: Int, BACKWARD: Bool
](in_dims: List[Int], out_dims: List[Int], scales: List[Float64]) -> Bool:
    """Whether CUDA copies the input unchanged, per kernel: the host-side
    `copy_` of upsample_nearest2d / _upsample_nearest_exact2d /
    upsample_bilinear2d (forward and backward), the in-kernel "just copy" of
    the linear 1-d, trilinear and bicubic kernels, and the antialiased
    backward's. The nearest 1-d / 3-d kernels have no shortcut, but at an
    unchanged size with a unit scale their source index is the identity.
    The antialiased CUDA forward has none; lanczos, a CPU-only kernel,
    copies in its forward (upsample_separable_Nd_kernel_impl) and not in its
    backward."""
    for k in range(RANK):
        if in_dims[2 + k] != out_dims[2 + k]:
            return False
    comptime if MODE <= NEAREST_EXACT and RANK != 2:
        for k in range(RANK):
            var n = in_dims[2 + k]
            if (
                _kernel_scale[MODE, BACKWARD](n, n, False, scales[k], False)
                != 1.0
            ):
                return False
        return True
    elif MODE == LANCZOS_AA:
        return not BACKWARD
    elif MODE >= BILINEAR_AA:
        return BACKWARD
    else:
        return True


def op_upsample[
    MODE: Int, RANK: Int, OUT: Bool
](args: Values, n_args: Int, rets: Values, n_rets: Int) raises:
    """aten::upsample_nearest{1,2,3}d, _upsample_nearest_exact{1,2,3}d,
    upsample_linear1d, upsample_bilinear2d, upsample_bicubic2d,
    upsample_trilinear3d and their `.out`:
    (Tensor self, SymInt[R] output_size, [bool align_corners,]
    float? scales..., [*, Tensor(a!) out])."""
    comptime name = _up_name[MODE, RANK]()
    comptime interp = MODE >= LINEAR
    comptime first_scale = 3 if interp else 2
    var a = v_tensor(args[unsafe_offset=0])
    _require_mojo(a, name)
    var osize = IntList(args[unsafe_offset=1])
    var align = v_bool(args[unsafe_offset=2]) if interp else False
    var scales = List[Float64]()
    for k in range(RANK):
        scales.append(_opt_scale(args[unsafe_offset=first_scale + k]))
    var in_dims = List[Int]()
    for k in range(a.rank):
        in_dims.append(a.dim(k))
    var out_dims = _up_check[RANK](in_dims, osize)
    if a.numel == 0 and in_dims[1] == 0:
        raise Error(
            "Non-empty ",
            RANK + 2,
            "D data tensor expected but got a tensor with sizes ",
            _sizes_str(a),
        )
    if not _up_dtype_ok[MODE](a):
        unsupported(String(name, ": dtype ", a.dtype))
    var identity = _is_identity[MODE, RANK, False](in_dims, out_dims, scales)
    var shortcut = _has_copy_shortcut[MODE, RANK]() and _same_dims(
        in_dims, out_dims
    )
    comptime if OUT:
        var dst = _dest(
            args,
            first_scale + RANK,
            a,
            out_dims,
            [a.copy()],
            _copies_back[MODE, RANK, False](),
            shortcut and a.numel > 0,
        )
        a = T(a.h)  # the resize may have moved a storage `a` shares
        # The CUDA template's literal sequence on the caller's tensor.
        comptime if MODE == NEAREST or MODE == NEAREST_EXACT:
            if RANK == 2 and a.numel == 0:
                ret_ref(rets, 0, dst)
                return
        if shortcut:
            if dst.numel > 0 and dst.impl() != a.impl():
                copy_strided_into(dst, a)  # output.copy_(input)
        else:
            var src = own_if_new(contiguous(a), a)
            _into[MODE, RANK, False](
                dst, src.t, identity, in_dims, out_dims, align, scales
            )
            _ = src^
        ret_ref(rets, 0, dst)
    else:
        var dst = new_tensor(_shape(out_dims), RANK + 2, a.stype, a.device)
        if dst.numel > 0:
            var src = own_if_new(contiguous(a), a)
            _into[MODE, RANK, False](
                dst, src.t, identity, in_dims, out_dims, align, scales
            )
            _ = src^
        var o = own(dst^)
        ret_owned(rets, 0, o)


def op_upsample_backward[
    MODE: Int, RANK: Int, OUT: Bool
](args: Values, n_args: Int, rets: Values, n_rets: Int) raises:
    """The `_backward` / `_backward.grad_input` overloads: (Tensor
    grad_output, SymInt[R] output_size, SymInt[R+2] input_size,
    [bool align_corners,] float? scales..., [*, Tensor(a!) grad_input])."""
    comptime name = _up_name[MODE, RANK]() + "_backward"
    comptime interp = MODE >= LINEAR
    comptime first_scale = 4 if interp else 3
    var g = v_tensor(args[unsafe_offset=0])
    _require_mojo(g, name)
    var osize = IntList(args[unsafe_offset=1])
    var isize = IntList(args[unsafe_offset=2])
    var align = v_bool(args[unsafe_offset=3]) if interp else False
    var scales = List[Float64]()
    for k in range(RANK):
        scales.append(_opt_scale(args[unsafe_offset=first_scale + k]))
    var in_dims = isize.to_list()
    var out_dims = _up_check[RANK](in_dims, osize)
    if g.rank != RANK + 2:
        raise Error(
            "Expected grad_output to be a tensor of dimension ",
            RANK + 2,
            " but got: dimension ",
            g.rank,
        )
    for k in range(RANK + 2):
        if g.dim(k) != out_dims[k]:
            raise Error(
                "Expected grad_output to have the same shape as output;",
                " output.size(",
                k,
                ") = ",
                out_dims[k],
                " but got grad_output.size(",
                k,
                ") = ",
                g.dim(k),
            )
    if not _up_dtype_ok[MODE](g):
        unsupported(String(name, ": dtype ", g.dtype))
    var identity = _is_identity[MODE, RANK, True](in_dims, out_dims, scales)
    var shortcut = _has_copy_shortcut[MODE, RANK]() and _same_dims(
        in_dims, out_dims
    )
    comptime if OUT:
        var grad_input_numel = 1
        for k in range(len(in_dims)):
            grad_input_numel *= in_dims[k]
        var dst = _dest(
            args,
            first_scale + RANK,
            g,
            in_dims,
            [g.copy()],
            _copies_back[MODE, RANK, True](),
            shortcut and grad_input_numel > 0,
        )
        g = T(g.h)  # the resize may have moved a storage `g` shares
        if dst.numel == 0:
            ret_ref(rets, 0, dst)
            return
        # The CUDA template's literal sequence on the caller's tensor:
        # bilinear2d zeroes grad_input, then copies (same size) or reads a
        # `.contiguous()` of grad_output made after the zeroing; the other
        # interpolating backwards take that `.contiguous()` first, then
        # zero; the nearest ones never zero.
        comptime if MODE == LINEAR and RANK == 2:
            # CUDA zeroes grad_input before reading grad_output; the gather
            # kernel writes every element, so only an aliased grad_output can
            # tell the difference: zero only then.
            if shares_storage(dst, g):
                fill_value(dst, 0.0)
            if shortcut:
                if dst.impl() != g.impl():
                    copy_strided_into(dst, g)  # grad_input.copy_(grad_output)
            else:
                var src = own_if_new(contiguous(g), g)
                _into[MODE, RANK, True](
                    dst, src.t, identity, in_dims, out_dims, align, scales
                )
                _ = src^
        elif MODE >= LINEAR:
            var src = own_if_new(contiguous(g), g)
            # CUDA zeroes grad_input before reading grad_output; the gather
            # kernel writes every element, so only an aliased grad_output can
            # tell the difference: zero only then.
            if shares_storage(dst, g):
                fill_value(dst, 0.0)
            _into[MODE, RANK, True](
                dst, src.t, identity, in_dims, out_dims, align, scales
            )
            _ = src^
        else:
            if shortcut:
                if dst.impl() != g.impl():
                    copy_strided_into(dst, g)
            else:
                var src = own_if_new(contiguous(g), g)
                _into[MODE, RANK, True](
                    dst, src.t, identity, in_dims, out_dims, align, scales
                )
                _ = src^
        ret_ref(rets, 0, dst)
    else:
        var dst = new_tensor(_shape(in_dims), RANK + 2, g.stype, g.device)
        if dst.numel > 0:
            var src = own_if_new(contiguous(g), g)
            _into[MODE, RANK, True](
                dst, src.t, identity, in_dims, out_dims, align, scales
            )
            _ = src^
        var o = own(dst^)
        ret_owned(rets, 0, o)


# ---------------------------------------------------------------------------
# Reflection / replication padding
# ---------------------------------------------------------------------------


def _pad_name[REFLECT: Bool, RANK: Int]() -> String:
    comptime if REFLECT:
        return String("reflection_pad", RANK, "d")
    else:
        return String("replication_pad", RANK, "d")


def _reflect_check[RANK: Int](t: T, padding: IntList) raises:
    """Reflection's `pad < input size` check, per padded dim (the meta
    functions of the forward and of the backward)."""
    var lead = t.rank - RANK
    for k in range(RANK):
        # padding pairs run from the last dim: (left, right, top, ...)
        var axis = RANK - 1 - k
        var n = t.dim(lead + axis)
        var lo = padding[2 * k]
        var hi = padding[2 * k + 1]
        if lo >= n or hi >= n:
            var prefix = String()
            comptime if RANK != 2:
                prefix = String(
                    "Argument ",
                    _pick(3 - RANK + axis, "#8", "#6", "#4"),
                    ": ",
                )
            raise Error(
                prefix,
                (
                    "Padding size should be less than the corresponding"
                    " input dimension, but got: padding ("
                ),
                lo,
                ", ",
                hi,
                ") at dimension ",
                lead + axis,
                " of input ",
                _sizes_str(t),
            )


def _pad_geom[
    REFLECT: Bool, RANK: Int
](t: T, padding: IntList) raises -> List[Int]:
    """The meta checks of reflection_pad{1,2,3}d / replication_pad{1,2,3}d
    (Padding.h's check_valid_input, then the per-op size checks): the full
    output size."""
    if len(padding) != 2 * RANK:
        comptime if REFLECT:
            raise Error(
                "padding size is expected to be ",
                2 * RANK,
                ", but got: ",
                len(padding),
            )
        else:
            raise Error("padding size is expected to be ", 2 * RANK)
    var batch_mode = t.rank == RANK + 2
    var valid = batch_mode or t.rank == RANK + 1
    if valid:
        for k in range(1 if batch_mode else 0, t.rank):
            if t.dim(k) == 0:
                valid = False
    if not valid:
        raise Error(
            "Expected ",
            RANK + 1,
            "D or ",
            RANK + 2,
            (
                "D (batch mode) tensor with possibly 0 batch size and other"
                " non-zero dimensions for input, but got: "
            ),
            _sizes_str(t),
        )
    var lead = t.rank - RANK
    comptime if REFLECT:
        _reflect_check[RANK](t, padding)
    var dims = List[Int]()
    for k in range(lead):
        dims.append(t.dim(k))
    var any_pos = False
    var all_pos = True
    for axis in range(RANK):
        var k = RANK - 1 - axis
        var o = t.dim(lead + axis) + padding[2 * k] + padding[2 * k + 1]
        dims.append(o)
        if o >= 1:
            any_pos = True
        else:
            all_pos = False
    # Reflection 2-d/3-d only require one positive extent (CUDA's `||`).
    var ok = all_pos
    comptime if REFLECT and RANK >= 2:
        ok = any_pos
    if not ok:
        var ins = String()
        var outs = String()
        for axis in range(RANK):
            var label = _pick(3 - RANK + axis, "D", "H", "W")
            if axis:
                ins += ", "
                outs += " "
            ins += String(label, ": ", t.dim(lead + axis))
            outs += String(label, ": ", dims[lead + axis])
        comptime if RANK == 1:
            raise Error(
                "input (", ins, ") is too small. Calculated output ", outs
            )
        elif REFLECT:
            raise Error(
                "input (", ins, ") is too small. Calculated output ", outs
            )
        else:
            raise Error(
                "Calculated output ",
                outs,
                " must be >= 1 in every dimension (input ",
                ins,
                ")",
            )
    for k in range(len(dims)):
        if dims[k] < 0:
            raise Error(
                "Trying to create tensor with negative dimension ",
                dims[k],
                ": ",
                _list_str(dims),
            )
    return dims^


def _run_pad[
    REFLECT: Bool, RANK: Int, BACKWARD: Bool
](
    dst: T, src: T, in_dims: List[Int], out_dims: List[Int], padding: IntList
) raises:
    var lead = len(in_dims) - RANK
    var batch = 1
    for k in range(lead):
        batch *= in_dims[k]
    var geom = List[Int]()
    for k in range(3 - RANK):
        _ = k
        geom.append(1)
    for axis in range(RANK):
        geom.append(in_dims[lead + axis])
    for k in range(3 - RANK):
        _ = k
        geom.append(0)
    for axis in range(RANK):
        geom.append(padding[2 * (RANK - 1 - axis)])
    for k in range(3 - RANK):
        _ = k
        geom.append(1)
    for axis in range(RANK):
        geom.append(out_dims[lead + axis])
    var ctx = ctx_for(src.device)
    var call = KernelCall("resample", "PadBackward" if BACKWARD else "Pad")
    call.arg_dtype(0, src.dtype)
    call.flag("REFLECT", 1 if REFLECT else 0)
    call.int(dst.ptr)
    call.int(src.ptr)
    call.int(batch)
    call.tuple(geom)
    call.int(ctx_ptr(ctx))
    call.run()
    _ = ctx


def _pad_into[
    REFLECT: Bool, RANK: Int, BACKWARD: Bool
](
    dst: T, a: T, in_dims: List[Int], out_dims: List[Int], padding: IntList
) raises:
    """The pad kernel from a `.contiguous()` of `a` into `dst`, through a
    temporary copied back when `dst` is not contiguous."""
    if dst.numel == 0:
        return
    var src = own_if_new(contiguous(a), a)
    var target = Optional[Owned](None)
    var into = dst.copy()
    if not dst.contig:
        target = own(new_like(dst))
        into = target.value().t.copy()
    _run_pad[REFLECT, RANK, BACKWARD](into, src.t, in_dims, out_dims, padding)
    if target:
        copy_strided_into(dst, into)
    _ = target^
    _ = src^


def op_pad[
    REFLECT: Bool, RANK: Int, OUT: Bool
](args: Values, n_args: Int, rets: Values, n_rets: Int) raises:
    """aten::reflection_pad{1,2,3}d / replication_pad{1,2,3}d and `.out`:
    (Tensor self, SymInt[2R] padding, [*, Tensor(a!) out])."""
    comptime name = _pad_name[REFLECT, RANK]()
    var a = v_tensor(args[unsafe_offset=0])
    _require_mojo(a, name)
    var padding = IntList(args[unsafe_offset=1])
    var out_dims = _pad_geom[REFLECT, RANK](a, padding)
    if a.dtype == DType.bool:
        unsupported(String(name, ": bool"))
    var in_dims = List[Int]()
    for k in range(a.rank):
        in_dims.append(a.dim(k))
    comptime if OUT:
        # The CUDA kernel's sequence on the caller's tensor: resize, then
        # read a `.contiguous()` of the input and write the out.
        var dst = _dest(args, 2, a, out_dims, [a.copy()], False, False)
        a = T(a.h)  # the resize may have moved a storage `a` shares
        _pad_into[REFLECT, RANK, False](dst, a, in_dims, out_dims, padding)
        ret_ref(rets, 0, dst)
    else:
        var dst = new_tensor(_shape(out_dims), a.rank, a.stype, a.device)
        _pad_into[REFLECT, RANK, False](dst, a, in_dims, out_dims, padding)
        var o = own(dst^)
        ret_owned(rets, 0, o)


def op_pad_backward[
    REFLECT: Bool, RANK: Int, OUT: Bool
](args: Values, n_args: Int, rets: Values, n_rets: Int) raises:
    """The `_backward` / `_backward.grad_input` overloads: (Tensor
    grad_output, Tensor self, SymInt[2R] padding, [*, Tensor(a!)
    grad_input])."""
    comptime name = _pad_name[REFLECT, RANK]() + "_backward"
    var g = v_tensor(args[unsafe_offset=0])
    var a = v_tensor(args[unsafe_offset=1])
    _require_mojo(g, name)
    var padding = IntList(args[unsafe_offset=2])
    if len(padding) != 2 * RANK:
        raise Error("padding size is expected to be ", 2 * RANK)
    if a.rank != RANK + 1 and a.rank != RANK + 2:
        unsupported(String(name, ": input of rank ", a.rank))
    comptime if REFLECT:
        _reflect_check[RANK](a, padding)
    var lead = a.rank - RANK
    if g.rank != a.rank:
        raise Error(
            "grad_output rank unexpected. Expected: ",
            a.rank,
            ", Got: ",
            g.rank,
        )
    for k in range(lead):
        if g.dim(k) != a.dim(k):
            raise Error(
                "gradOutput channel unexpected. Expected: ",
                a.dim(k),
                ", Got: ",
                g.dim(k),
            )
    var in_dims = List[Int]()
    var out_dims = List[Int]()
    for k in range(lead):
        in_dims.append(a.dim(k))
        out_dims.append(a.dim(k))
    for axis in range(RANK):
        var k = RANK - 1 - axis
        var n = a.dim(lead + axis)
        in_dims.append(n)
        var o = n + padding[2 * k] + padding[2 * k + 1]
        if o != g.dim(lead + axis):
            raise Error(
                "grad_output ",
                _pick(3 - RANK + axis, "depth", "height", "width"),
                " unexpected. Expected: ",
                o,
                ", Got: ",
                g.dim(lead + axis),
            )
        out_dims.append(o)
    if not is_floating(g.stype):
        unsupported(String(name, ": dtype ", g.dtype))
    if a.stype != g.stype:
        unsupported(String(name, ": grad_output and self dtypes differ"))
    comptime if OUT:
        # `self` is only read for its shape, so it may alias grad_input.
        # The CUDA sequence: resize, zero grad_input, then read (a
        # `.contiguous()` of) grad_output and accumulate.
        var dst = _dest(args, 3, g, in_dims, [g.copy()], False, False)
        g = T(g.h)  # the resize may have moved a storage `g` shares
        if dst.numel > 0:
            # CUDA zeroes grad_input before reading grad_output; the gather
            # kernel writes every element, so only an aliased grad_output can
            # tell the difference: zero only then.
            if shares_storage(dst, g):
                fill_value(dst, 0.0)
        _pad_into[REFLECT, RANK, True](dst, g, in_dims, out_dims, padding)
        ret_ref(rets, 0, dst)
    else:
        var dst = new_tensor(_shape(in_dims), a.rank, g.stype, g.device)
        _pad_into[REFLECT, RANK, True](dst, g, in_dims, out_dims, padding)
        var o = own(dst^)
        ret_owned(rets, 0, o)


# ---------------------------------------------------------------------------
# grid_sampler_2d / grid_sampler_3d (F.grid_sample) and their backwards
# ---------------------------------------------------------------------------
#
# Kernels: tmb/kernels/grid_sample/entry.mojo (a port of GridSampler.cu).
# Checks and messages: GridSamplerUtils.h's check_grid_sampler_* at v2.14.0,
# then the CUDA launcher's dtype dispatch and data_ptr<scalar_t>() checks.
# Every operand is read through its own strides; the outputs are contiguous
# (grad_input: zeros_like(input, LEGACY_CONTIGUOUS), grad_grid: empty_like
# (grid, LEGACY_CONTIGUOUS)), as on CUDA. The `.out` overloads are torch's
# autogenerated ones: compute, then resize `out` and copy (casting) into it.


def _grid_float_ok(t: T) raises -> Bool:
    if t.dtype == DType.float64:
        return dev(t.device)[].api != "metal"
    return (
        t.dtype == DType.float32
        or t.dtype == DType.float16
        or t.dtype == DType.bfloat16
    )


def _grid_check[
    RANK: Int
](input: T, grid: T, interp: Int, pad: Int, name: String) raises:
    """check_grid_sampler_common + check_grid_sampler_{2,3}d, then the mode
    and dtype gates of the CUDA launcher."""
    if not input.on_mojo():
        unsupported(name + ": input is not on the mojo device")
    if input.device_type != grid.device_type or input.device != grid.device:
        raise Error(
            (
                "grid_sampler(): expected input and grid to be on same device,"
                " but input is on "
            ),
            device_str(input),
            " and grid is on ",
            device_str(grid),
        )
    if input.rank == 0 or grid.rank == 0:
        index_error("dimension specified as 0 but tensor has no dimensions")
    if input.dim(0) != grid.dim(0):
        raise Error(
            (
                "grid_sampler(): expected grid and input to have same batch"
                " size, but got input with sizes "
            ),
            _sizes_str(input),
            " and grid with sizes ",
            _sizes_str(grid),
        )
    if grid.dim(-1) != input.rank - 2:
        raise Error(
            "grid_sampler(): expected grid to have size ",
            input.rank - 2,
            " in last dimension, but got grid with sizes ",
            _sizes_str(grid),
        )
    for i in range(2, input.rank):
        if input.dim(i) <= 0:
            raise Error(
                (
                    "grid_sampler(): expected input to have non-empty spatial"
                    " dimensions, but input has sizes "
                ),
                _sizes_str(input),
                " with dimension ",
                i,
                " being empty",
            )
    if input.rank != RANK + 2 or grid.rank != input.rank:
        raise Error(
            "grid_sampler(): expected ",
            RANK + 2,
            (
                "D input and grid with same number of dimensions, but got input"
                " with sizes "
            ),
            _sizes_str(input),
            " and grid with sizes ",
            _sizes_str(grid),
        )
    comptime if RANK == 3:
        if interp == 2:
            raise Error(
                "grid_sampler(): bicubic interpolation only supports 4D input"
            )
    if interp < 0 or interp > 2:
        raise Error(
            name, ": unknown interpolation_mode ", interp, " (expected 0..2)"
        )
    if pad < 0 or pad > 2:
        raise Error(name, ": unknown padding_mode ", pad, " (expected 0..2)")


def _grid_dtype_check(
    input: T, others: List[T], count: Int, name: String
) raises:
    """AT_DISPATCH_FLOATING_TYPES_AND2(Half, BFloat16) over the input, then
    each operand's `data_ptr<scalar_t>()` (only reached when count > 0)."""
    if (
        input.dtype != DType.float32
        and input.dtype != DType.float16
        and input.dtype != DType.bfloat16
        and input.dtype != DType.float64
    ):
        unsupported(
            String(
                '"',
                name,
                "_cuda\" not implemented for '",
                _scalar_type_name(input.dtype),
                "'",
            )
        )
    if not _grid_float_ok(input):
        unsupported(name + ": float64 is unavailable on Apple GPUs")
    if count == 0:
        return
    for k in range(len(others)):
        if others[k].stype != input.stype:
            raise Error(
                "expected scalar type ",
                _scalar_type_name(input.dtype),
                " but found ",
                _scalar_type_name(others[k].dtype),
            )


def _grid_geometry[
    RANK: Int
](
    input: T,
    grid: T,
    strided: T,
    interp: Int,
    pad: Int,
    align: Bool,
    input_grad: Bool,
) -> List[Int]:
    """The kernel's geometry tuple (see G_* in the kernel family): sizes,
    the strides of input, grid and `strided` (output / grad_output), modes.
    A 2-d op has unit D extents and zero D strides."""
    var g = List[Int]()
    var count = input.dim(0)
    g.append(input.dim(0))
    g.append(input.dim(1))
    comptime if RANK == 2:
        g.append(1)
    for i in range(2, RANK + 2):
        g.append(input.dim(i))
    comptime if RANK == 2:
        g.append(1)
    for i in range(1, RANK + 1):
        g.append(grid.dim(i))
        count *= grid.dim(i)
    # input strides N, C, D, H, W
    g.append(input.stride(0))
    g.append(input.stride(1))
    comptime if RANK == 2:
        g.append(0)
    for i in range(2, RANK + 2):
        g.append(input.stride(i))
    # grid strides N, D, H, W, coordinate
    g.append(grid.stride(0))
    comptime if RANK == 2:
        g.append(0)
    for i in range(1, RANK + 2):
        g.append(grid.stride(i))
    # output / grad_output strides N, C, D, H, W
    g.append(strided.stride(0))
    g.append(strided.stride(1))
    comptime if RANK == 2:
        g.append(0)
    for i in range(2, RANK + 2):
        g.append(strided.stride(i))
    g.append(interp)
    g.append(pad)
    g.append(1 if align else 0)
    g.append(1 if input_grad else 0)
    g.append(count)
    return g^


def _grid_out_shape[RANK: Int](input: T, grid: T) -> List[Int]:
    var dims = List[Int]()
    dims.append(input.dim(0))
    dims.append(input.dim(1))
    for i in range(1, RANK + 1):
        dims.append(grid.dim(i))
    return dims^


def _copy_to_out(mut dst: T, src: T, dims: List[Int], name: String) raises:
    """torch's autogenerated `.out`: `resize_output(out, result.sizes())`
    then the generated `copy_arg`, which requires the result's exact dtype
    and device (no cast), into an `out` that does not overlap itself."""
    check_out_as(dst, src.stype, src)
    assert_no_internal_overlap(dst)
    resize_out(dst, _shape(dims), len(dims))
    if dst.numel == 0:
        return
    copy_strided_into(dst, src)


def op_grid_sampler[
    RANK: Int, OUT: Bool
](args: Values, n_args: Int, rets: Values, n_rets: Int) raises:
    """aten::grid_sampler_{2,3}d(Tensor input, Tensor grid,
    int interpolation_mode, int padding_mode, bool align_corners) -> Tensor,
    and `.out(..., *, Tensor(a!) out)`."""
    comptime name = String("grid_sampler_", RANK, "d")
    var input = v_tensor(args[unsafe_offset=0])
    var grid = v_tensor(args[unsafe_offset=1])
    var interp = v_int(args[unsafe_offset=2])
    var pad = v_int(args[unsafe_offset=3])
    var align = v_bool(args[unsafe_offset=4])
    _grid_check[RANK](input, grid, interp, pad, name)
    var dims = _grid_out_shape[RANK](input, grid)
    var count = input.dim(0)
    for i in range(1, RANK + 1):
        count *= grid.dim(i)
    _grid_dtype_check(input, [grid.copy()], count, name)
    var out = own(new_tensor(_shape(dims), RANK + 2, input.stype, input.device))
    if count > 0 and out.t.numel > 0:
        var geom = _grid_geometry[RANK](
            input, grid, out.t, interp, pad, align, False
        )
        var ctx = ctx_for(input.device)
        var call = KernelCall(
            "grid_sample", "GridSampler2d" if RANK == 2 else "GridSampler3d"
        )
        call.arg_dtype(0, input.dtype)
        call.int(out.t.ptr)
        call.int(input.ptr)
        call.int(grid.ptr)
        call.tuple(geom)
        call.int(ctx_ptr(ctx))
        call.run()
        _ = ctx
    comptime if OUT:
        var dst = v_tensor(args[unsafe_offset=5])
        _copy_to_out(dst, out.t, dims, name)
        _ = out^
        ret_ref(rets, 0, dst)
    else:
        ret_owned(rets, 0, out)


def _mask_list(v: Value) raises -> List[Bool]:
    """A borrowed `bool[2]` argument (uint8 per element in the call arena)."""
    if v.tag != TAG_BOOL_LIST:
        raise Error("expected a bool[] argument, got record tag ", v.tag)
    var out = List[Bool]()
    var p = Pointer[UInt8, MutUntrackedOrigin](unsafe_from_address=Int(v.a))
    for i in range(Int(v.len)):
        out.append(p[unsafe_offset=i] != 0)
    return out^


def op_grid_sampler_backward[
    RANK: Int, OUT: Bool
](args: Values, n_args: Int, rets: Values, n_rets: Int) raises:
    """aten::grid_sampler_{2,3}d_backward(Tensor grad_output, Tensor input,
    Tensor grid, int interpolation_mode, int padding_mode,
    bool align_corners, bool[2] output_mask) -> (Tensor, Tensor), and
    `.out(..., *, Tensor(a!) out0, Tensor(b!) out1)`. grad_input is computed
    only when output_mask[0] (else undefined); grad_grid always."""
    comptime name = String("grid_sampler_", RANK, "d_backward")
    var grad = v_tensor(args[unsafe_offset=0])
    var input = v_tensor(args[unsafe_offset=1])
    var grid = v_tensor(args[unsafe_offset=2])
    var interp = v_int(args[unsafe_offset=3])
    var pad = v_int(args[unsafe_offset=4])
    var align = v_bool(args[unsafe_offset=5])
    var mask = _mask_list(args[unsafe_offset=6])
    if len(mask) != 2:
        raise Error(name, ": output_mask must have 2 entries")
    var need_in = mask[0]
    _grid_check[RANK](input, grid, interp, pad, name)
    # check_grid_sampler_backward (a ValueError in torch).
    var expected = _grid_out_shape[RANK](input, grid)
    var shape_ok = grad.rank == len(expected)
    if shape_ok:
        for k in range(len(expected)):
            if grad.dim(k) != expected[k]:
                shape_ok = False
    if not shape_ok:
        raise Error(
            "grid_sampler(): expected grad_output to have sizes ",
            _list_str(expected),
            " but got grad_output with sizes ",
            _sizes_str(grad),
        )
    if not grad.on_mojo() or grad.device != input.device:
        unsupported(name + ": grad_output must be on the input's mojo device")
    # Nondeterministic because of atomicAdd usage (raised before the
    # launcher's count check, as on CUDA).
    alert_not_deterministic(name + "_cuda")
    var count = input.dim(0)
    for i in range(1, RANK + 1):
        count *= grid.dim(i)
    _grid_dtype_check(input, [grad.copy(), grid.copy()], count, name)
    var gi_dims = List[Int]()
    for i in range(input.rank):
        gi_dims.append(input.dim(i))
    var gg_dims = List[Int]()
    for i in range(grid.rank):
        gg_dims.append(grid.dim(i))
    var gi = own(
        new_tensor(
            _shape(gi_dims) if need_in else IndexList[MAX_RANK](0),
            RANK + 2 if need_in else 1,
            input.stype,
            input.device,
        )
    )
    var gg = own(new_tensor(_shape(gg_dims), RANK + 2, grid.stype, grid.device))
    if (need_in and gi.t.numel > 0) or gg.t.numel > 0:
        var geom = _grid_geometry[RANK](
            input, grid, grad, interp, pad, align, need_in
        )
        var ctx = ctx_for(input.device)
        var call = KernelCall(
            "grid_sample",
            "GridSampler2dBackward" if RANK == 2 else "GridSampler3dBackward",
        )
        call.arg_dtype(0, input.dtype)
        call.int(gi.t.ptr if need_in else 0)
        call.int(gg.t.ptr)
        call.int(grad.ptr)
        call.int(input.ptr)
        call.int(grid.ptr)
        call.tuple(geom)
        call.int(ctx_ptr(ctx))
        call.run()
        _ = ctx
    comptime if OUT:
        var d1 = v_tensor(args[unsafe_offset=8])
        if need_in:
            var d0 = v_tensor(args[unsafe_offset=7])
            _copy_to_out(d0, gi.t, gi_dims, name)
            ret_ref(rets, 0, d0)
        else:
            ret_ref(rets, 0, v_tensor(args[unsafe_offset=7]))
        _copy_to_out(d1, gg.t, gg_dims, name)
        ret_ref(rets, 1, d1)
        _ = gi^
        _ = gg^
    else:
        if need_in:
            ret_owned(rets, 0, gi)
        else:
            rets[unsafe_offset=0] = Value(TAG_NONE, 0, 0, 0)
        ret_owned(rets, 1, gg)


def register_resample(site: Site) raises:
    impl[op_upsample[NEAREST, 1, False], "upsample_nearest1d"](site)
    impl[op_upsample[NEAREST, 1, True], "upsample_nearest1d.out"](site)
    impl[
        op_upsample_backward[NEAREST, 1, False], "upsample_nearest1d_backward"
    ](site)
    impl[
        op_upsample_backward[NEAREST, 1, True],
        "upsample_nearest1d_backward.grad_input",
    ](site)
    impl[op_upsample[NEAREST, 2, False], "upsample_nearest2d"](site)
    impl[op_upsample[NEAREST, 2, True], "upsample_nearest2d.out"](site)
    impl[
        op_upsample_backward[NEAREST, 2, False], "upsample_nearest2d_backward"
    ](site)
    impl[
        op_upsample_backward[NEAREST, 2, True],
        "upsample_nearest2d_backward.grad_input",
    ](site)
    impl[op_upsample[NEAREST, 3, False], "upsample_nearest3d"](site)
    impl[op_upsample[NEAREST, 3, True], "upsample_nearest3d.out"](site)
    impl[
        op_upsample_backward[NEAREST, 3, False], "upsample_nearest3d_backward"
    ](site)
    impl[
        op_upsample_backward[NEAREST, 3, True],
        "upsample_nearest3d_backward.grad_input",
    ](site)
    impl[op_upsample[NEAREST_EXACT, 1, False], "_upsample_nearest_exact1d"](
        site
    )
    impl[op_upsample[NEAREST_EXACT, 1, True], "_upsample_nearest_exact1d.out"](
        site
    )
    impl[
        op_upsample_backward[NEAREST_EXACT, 1, False],
        "_upsample_nearest_exact1d_backward",
    ](site)
    impl[
        op_upsample_backward[NEAREST_EXACT, 1, True],
        "_upsample_nearest_exact1d_backward.grad_input",
    ](site)
    impl[op_upsample[NEAREST_EXACT, 2, False], "_upsample_nearest_exact2d"](
        site
    )
    impl[op_upsample[NEAREST_EXACT, 2, True], "_upsample_nearest_exact2d.out"](
        site
    )
    impl[
        op_upsample_backward[NEAREST_EXACT, 2, False],
        "_upsample_nearest_exact2d_backward",
    ](site)
    impl[
        op_upsample_backward[NEAREST_EXACT, 2, True],
        "_upsample_nearest_exact2d_backward.grad_input",
    ](site)
    impl[op_upsample[NEAREST_EXACT, 3, False], "_upsample_nearest_exact3d"](
        site
    )
    impl[op_upsample[NEAREST_EXACT, 3, True], "_upsample_nearest_exact3d.out"](
        site
    )
    impl[
        op_upsample_backward[NEAREST_EXACT, 3, False],
        "_upsample_nearest_exact3d_backward",
    ](site)
    impl[
        op_upsample_backward[NEAREST_EXACT, 3, True],
        "_upsample_nearest_exact3d_backward.grad_input",
    ](site)
    impl[op_upsample[LINEAR, 1, False], "upsample_linear1d"](site)
    impl[op_upsample[LINEAR, 1, True], "upsample_linear1d.out"](site)
    impl[op_upsample_backward[LINEAR, 1, False], "upsample_linear1d_backward"](
        site
    )
    impl[
        op_upsample_backward[LINEAR, 1, True],
        "upsample_linear1d_backward.grad_input",
    ](site)
    impl[op_upsample[LINEAR, 2, False], "upsample_bilinear2d"](site)
    impl[op_upsample[LINEAR, 2, True], "upsample_bilinear2d.out"](site)
    impl[
        op_upsample_backward[LINEAR, 2, False], "upsample_bilinear2d_backward"
    ](site)
    impl[
        op_upsample_backward[LINEAR, 2, True],
        "upsample_bilinear2d_backward.grad_input",
    ](site)
    impl[op_upsample[LINEAR, 3, False], "upsample_trilinear3d"](site)
    impl[op_upsample[LINEAR, 3, True], "upsample_trilinear3d.out"](site)
    impl[
        op_upsample_backward[LINEAR, 3, False], "upsample_trilinear3d_backward"
    ](site)
    impl[
        op_upsample_backward[LINEAR, 3, True],
        "upsample_trilinear3d_backward.grad_input",
    ](site)
    impl[op_upsample[CUBIC, 2, False], "upsample_bicubic2d"](site)
    impl[op_upsample[CUBIC, 2, True], "upsample_bicubic2d.out"](site)
    impl[op_upsample_backward[CUBIC, 2, False], "upsample_bicubic2d_backward"](
        site
    )
    impl[
        op_upsample_backward[CUBIC, 2, True],
        "upsample_bicubic2d_backward.grad_input",
    ](site)
    impl[op_upsample[BILINEAR_AA, 2, False], "_upsample_bilinear2d_aa"](site)
    impl[op_upsample[BILINEAR_AA, 2, True], "_upsample_bilinear2d_aa.out"](site)
    impl[
        op_upsample_backward[BILINEAR_AA, 2, False],
        "_upsample_bilinear2d_aa_backward",
    ](site)
    impl[
        op_upsample_backward[BILINEAR_AA, 2, True],
        "_upsample_bilinear2d_aa_backward.grad_input",
    ](site)
    impl[op_upsample[BICUBIC_AA, 2, False], "_upsample_bicubic2d_aa"](site)
    impl[op_upsample[BICUBIC_AA, 2, True], "_upsample_bicubic2d_aa.out"](site)
    impl[
        op_upsample_backward[BICUBIC_AA, 2, False],
        "_upsample_bicubic2d_aa_backward",
    ](site)
    impl[
        op_upsample_backward[BICUBIC_AA, 2, True],
        "_upsample_bicubic2d_aa_backward.grad_input",
    ](site)
    # Lanczos is in torch from 2.14: on an older torch these register
    # against a schema that never appears, and are never called.
    impl[op_upsample[LANCZOS_AA, 2, False], "_upsample_lanczos2d_aa"](site)
    impl[op_upsample[LANCZOS_AA, 2, True], "_upsample_lanczos2d_aa.out"](site)
    impl[
        op_upsample_backward[LANCZOS_AA, 2, False],
        "_upsample_lanczos2d_aa_backward",
    ](site)
    impl[
        op_upsample_backward[LANCZOS_AA, 2, True],
        "_upsample_lanczos2d_aa_backward.grad_input",
    ](site)
    impl[op_pad[True, 1, False], "reflection_pad1d"](site)
    impl[op_pad[True, 1, True], "reflection_pad1d.out"](site)
    impl[op_pad_backward[True, 1, False], "reflection_pad1d_backward"](site)
    impl[
        op_pad_backward[True, 1, True], "reflection_pad1d_backward.grad_input"
    ](site)
    impl[op_pad[True, 2, False], "reflection_pad2d"](site)
    impl[op_pad[True, 2, True], "reflection_pad2d.out"](site)
    impl[op_pad_backward[True, 2, False], "reflection_pad2d_backward"](site)
    impl[
        op_pad_backward[True, 2, True], "reflection_pad2d_backward.grad_input"
    ](site)
    impl[op_pad[True, 3, False], "reflection_pad3d"](site)
    impl[op_pad[True, 3, True], "reflection_pad3d.out"](site)
    impl[op_pad_backward[True, 3, False], "reflection_pad3d_backward"](site)
    impl[
        op_pad_backward[True, 3, True], "reflection_pad3d_backward.grad_input"
    ](site)
    impl[op_pad[False, 1, False], "replication_pad1d"](site)
    impl[op_pad[False, 1, True], "replication_pad1d.out"](site)
    impl[op_pad_backward[False, 1, False], "replication_pad1d_backward"](site)
    impl[
        op_pad_backward[False, 1, True], "replication_pad1d_backward.grad_input"
    ](site)
    impl[op_pad[False, 2, False], "replication_pad2d"](site)
    impl[op_pad[False, 2, True], "replication_pad2d.out"](site)
    impl[op_pad_backward[False, 2, False], "replication_pad2d_backward"](site)
    impl[
        op_pad_backward[False, 2, True], "replication_pad2d_backward.grad_input"
    ](site)
    impl[op_pad[False, 3, False], "replication_pad3d"](site)
    impl[op_pad[False, 3, True], "replication_pad3d.out"](site)
    impl[op_pad_backward[False, 3, False], "replication_pad3d_backward"](site)
    impl[
        op_pad_backward[False, 3, True], "replication_pad3d_backward.grad_input"
    ](site)
    impl[op_grid_sampler[2, False], "grid_sampler_2d"](site)
    impl[op_grid_sampler[2, True], "grid_sampler_2d.out"](site)
    impl[op_grid_sampler[3, False], "grid_sampler_3d"](site)
    impl[op_grid_sampler[3, True], "grid_sampler_3d.out"](site)
    impl[op_grid_sampler_backward[2, False], "grid_sampler_2d_backward"](site)
    impl[op_grid_sampler_backward[2, True], "grid_sampler_2d_backward.out"](
        site
    )
    impl[op_grid_sampler_backward[3, False], "grid_sampler_3d_backward"](site)
    impl[op_grid_sampler_backward[3, True], "grid_sampler_3d_backward.out"](
        site
    )
