"""ATen ops: pooling group (see agents_docs/native_backend.md).

Max / average / adaptive pooling in 2-D and 3-D with their backwards, max
unpooling, and im2col / col2im (nn.Unfold / nn.Fold). ATen's 1-D pooling
ops are composites over the 2-D ones, and `F.interpolate(mode="area")` over
the adaptive average.

Every op validates its arguments the way torch's meta / CUDA host code does
(aten/src/ATen/native/Pool.h, DilatedMaxPool2d.cpp, AveragePool2d.cpp,
AveragePool3d.cpp, AdaptiveMaxPooling{2,3}d.cpp, cuda/DilatedMaxPool3d.cu,
cuda/AdaptiveAveragePooling{,3d}.cu, cuda/MaxUnpooling.cu,
im2col_shape_check.h at v2.14.0), makes its input contiguous and runs one
generic kernel of the `pool` family (tmb/kernels/pool/entry.mojo): the 2-D
ops are the 3-D kernels with a unit depth. Dtypes follow CUDA: the float
dtypes (float64 off Metal); im2col / col2im also take bool.
"""
from std.utils import IndexList

from tmb.backend.abi import (
    IntList,
    Owned,
    ST_FLOAT32,
    ST_INT64,
    T,
    Values,
    alert_not_deterministic,
    cpu_empty,
    new_tensor,
    own,
    own_if_new,
    ret_owned,
    ret_ref,
    unsupported,
    v_bool,
    v_int,
    v_is_none,
    v_tensor,
)
from tmb.backend.device import copy_to_host, ctx_for, ctx_ptr, dev
from tmb.backend.kernel_call import KernelCall
from tmb.kernels.common.op_utils import MAX_RANK
from tmb.ops.common import (
    assert_no_internal_overlap,
    cast_into,
    check_out,
    check_out_as,
    contiguous,
    copy_strided_into,
    fill_value,
    resize_out,
)
from tmb.backend.registry import Site, impl


# ---------------------------------------------------------------------------
# Shared helpers
# ---------------------------------------------------------------------------


def _sizes(t: T) -> String:
    """`IntArrayRef` as torch streams it into a message: `[2, 3, 4]`."""
    var s = String("[")
    for i in range(t.rank):
        if i:
            s += ", "
        s += String(t.dim(i))
    return s + "]"


def _trunc_div(a: Int, b: Int) -> Int:
    """C++ integer division (toward zero)."""
    var q = a // b
    if q < 0 and q * b != a:
        q += 1
    return q


def _check_dtype(t: T, what: String, allow_bool: Bool = False) raises:
    if not t.on_mojo():
        unsupported(what + " expects a mojo tensor")
    var dt = t.dtype
    var ok = (
        dt == DType.float32
        or dt == DType.float16
        or dt == DType.bfloat16
        or dt == DType.float64
    )
    if allow_bool and dt == DType.bool:
        ok = True
    if not ok:
        # CUDA dispatches these ops over the floating types only (AT_DISPATCH
        # _FLOATING_TYPES_AND2(Half, BFloat16), plus bool for im2col/col2im).
        unsupported(what + " of dtype " + String(dt))
    if dt == DType.float64 and dev(t.device)[].api == "metal":
        unsupported(what + ": float64 is unavailable on Apple GPUs")


def _same_device(a: T, b: T, what: StaticString) raises:
    if not b.on_mojo() or a.device != b.device:
        raise Error(
            String(what), ": expected all tensors to be on the same device"
        )


def _spatial(
    l: IntList, n: Int, what: String, fill: Int
) raises -> IndexList[3]:
    """An `int[n]` pool argument as (D, H, W): one entry broadcasts, a 2-D
    op's depth is `fill` (the unit window)."""
    var out = IndexList[3](fill)
    if len(l) == 1:
        for i in range(3 - n, 3):
            out[i] = l[0]
    elif len(l) == n:
        for i in range(n):
            out[3 - n + i] = l[i]
    else:
        raise Error(what)
    return out


def _shape_with(
    t: T, n: Int, sp: IndexList[3]
) -> Tuple[IndexList[MAX_RANK], Int]:
    """`t`'s shape with its last `n` dims replaced by `sp`'s last `n`."""
    var shape = t.shape
    for i in range(n):
        shape[MAX_RANK - n + i] = sp[3 - n + i]
    return (shape, t.rank)


def _in_spatial(t: T, n: Int) -> IndexList[3]:
    var sp = IndexList[3](1)
    for i in range(n):
        sp[3 - n + i] = t.dim(t.rank - n + i)
    return sp


def _planes(t: T, n: Int) -> Int:
    var p = 1
    for i in range(t.rank - n):
        p *= t.dim(i)
    return p


def _append3(mut g: List[Int], v: IndexList[3]):
    for i in range(3):
        g.append(v[i])


def _geom(
    planes: Int,
    ins: IndexList[3],
    outs: IndexList[3],
    k: IndexList[3],
    s: IndexList[3],
    p: IndexList[3],
    d: IndexList[3],
    cip: Bool,
    divisor: Int,
    n: Int,
) -> List[Int]:
    """The kernels' geometry tuple (`G_*` slots of tmb/kernels/pool)."""
    var g = List[Int](capacity=22)
    g.append(planes)
    _append3(g, ins)
    _append3(g, outs)
    _append3(g, k)
    _append3(g, s)
    _append3(g, p)
    _append3(g, d)
    g.append(1 if cip else 0)
    g.append(divisor)
    g.append(1 if n == 3 else 0)
    return g^


def _numel(sp: IndexList[3]) -> Int:
    return sp[0] * sp[1] * sp[2]


def _alloc(
    dest: Optional[T],
    shape: IndexList[MAX_RANK],
    rank: Int,
    stype: Int32,
    device: Int,
    reads: List[T],
) raises -> Owned:
    """Where a kernel writes one result: a fresh tensor, or for an `out=`
    the caller's tensor itself.

    An `out` is resized to the result's shape first (`resize_output`) and
    must not repeat elements (`assert_no_internal_overlap`). The kernel
    then writes straight into it when it is contiguous and shares no
    storage with an input the kernel reads; otherwise into a temporary that
    `_store` copies across. The returned `Owned` is only live (released on
    drop) when it is a fresh allocation.
    """
    if not dest:
        return own(new_tensor(shape, rank, stype, device))
    var d = dest.value().copy()
    resize_out(d, shape, rank)
    assert_no_internal_overlap(d)
    var direct = d.contig and d.stype == stype
    if direct and d.numel:
        for r in reads:
            if r.numel and r.storage_ptr() == d.storage_ptr():
                direct = False
    if direct:
        var o = own(d^)
        _ = o.take()  # the caller's tensor: never released here
        return o^
    return own(new_tensor(shape, rank, stype, device))


def _store(rets: Values, i: Int, dest: T, var result: Owned) raises:
    """Finish an `out=` variant: hand the caller's tensor back, copying the
    result into it when `_alloc` had to compute into a temporary."""
    var dst = T(dest.h)  # `_alloc` may have resized it
    if result.t.h != dest.h and result.t.numel:
        copy_strided_into(dst, result.t)
    _ = result^
    ret_ref(rets, i, dst)


def _launch(
    op: StaticString,
    dtype: DType,
    a: Int,
    b: Int,
    c: Int,
    g: List[Int],
    device: Int,
) raises:
    """One `pool` family kernel over three pointers (0 = unused) and a
    geometry tuple."""
    var ctx = ctx_for(device)
    var call = KernelCall("pool", op)
    call.arg_dtype(0, dtype)
    call.int(a)
    call.int(b)
    if c != 0:
        call.int(c)
    call.tuple(g)
    call.int(ctx_ptr(ctx))
    call.run()
    _ = ctx


# ---------------------------------------------------------------------------
# Max pooling
# ---------------------------------------------------------------------------


def _pool_out_size(
    isz: Int, k: Int, pad: Int, s: Int, dil: Int, ceil_mode: Bool
) raises -> Int:
    """Pool.h `pooling_output_shape`."""
    if s == 0:
        raise Error("stride should not be zero")
    if pad < 0:
        raise Error("pad must be non-negative, but got pad: ", pad)
    if pad > _trunc_div((k - 1) * dil + 1, 2):
        raise Error(
            "pad should be at most half of effective kernel size, but got pad=",
            pad,
            ", kernel_size=",
            k,
            " and dilation=",
            dil,
        )
    var num = isz + 2 * pad - dil * (k - 1) - 1
    if ceil_mode:
        num += s - 1
    var o = num // s + 1  # div_rtn: floor
    if ceil_mode and (o - 1) * s >= isz + pad:
        o -= 1
    return o


struct Window(Copyable, Movable):
    """Parsed kernel / stride / padding / dilation of a windowed pool op."""

    var k: IndexList[3]
    var s: IndexList[3]
    var p: IndexList[3]
    var d: IndexList[3]
    var ceil_mode: Bool

    def __init__(
        out self,
        n: Int,
        name: StaticString,
        kernel: IntList,
        stride: IntList,
        padding: IntList,
        dilation: IntList,
        has_dilation: Bool,
        ceil_mode: Bool,
    ) raises:
        # avg_pool3d's meta words these messages without "either".
        var avg3d = name == "avg_pool3d"
        var either = "" if avg3d else "either "
        var many = "a tuple of two ints" if n == 2 else "a tuple of three ints"
        var lead = String(name) + ": "
        self.k = _spatial(
            kernel,
            n,
            lead + "kernel_size must " + either + "be a single int, or " + many,
            1,
        )
        if len(stride) == 0:
            self.s = self.k
        else:
            self.s = _spatial(
                stride,
                n,
                lead
                + "stride must "
                + either
                + "be omitted, a single int, or "
                + many,
                1,
            )
        self.p = _spatial(
            padding,
            n,
            lead + "padding must " + either + "be a single int, or " + many,
            0,
        )
        if has_dilation:
            self.d = _spatial(
                dilation,
                n,
                lead + "dilation must be either a single int, or " + many,
                1,
            )
        else:
            self.d = IndexList[3](1)
        self.ceil_mode = ceil_mode

    def out_spatial(self, ins: IndexList[3], n: Int) raises -> IndexList[3]:
        var o = IndexList[3](1)
        for i in range(3 - n, 3):
            o[i] = _pool_out_size(
                ins[i],
                self.k[i],
                self.p[i],
                self.s[i],
                self.d[i],
                self.ceil_mode,
            )
        return o

    def shape_check(
        self,
        x: T,
        n: Int,
        ins: IndexList[3],
        outs: IndexList[3],
        fn_name: StaticString,
        check_input_size: Bool,
    ) raises:
        """Pool.h `pool2d_shape_check` / `pool3d_shape_check`."""
        for i in range(3 - n, 3):
            if self.k[i] <= 0:
                raise Error(
                    "kernel size should be greater than zero, but got ",
                    self._fmt(self.k, n, "k"),
                )
        for i in range(3 - n, 3):
            if self.s[i] <= 0:
                raise Error(
                    "stride should be greater than zero, but got ",
                    self._fmt(self.s, n, "d"),
                )
        for i in range(3 - n, 3):
            if self.d[i] <= 0:
                raise Error(
                    "dilation should be greater than zero, but got ",
                    self._fmt(self.d, n, "dilation"),
                )
        if n == 2:
            var valid = x.dim(1) != 0 and x.dim(2) != 0
            if not (
                (x.rank == 3 and x.dim(0) != 0 and valid)
                or (x.rank == 4 and valid and x.dim(3) != 0)
            ):
                raise Error(
                    (
                        "Expected 3D or 4D (batch mode) tensor with optional 0"
                        " dim batch size for input, but got:"
                    ),
                    _sizes(x),
                )
        else:
            if x.rank != 4 and x.rank != 5:
                raise Error(
                    fn_name,
                    ": Expected 4D or 5D tensor for input, but got: ",
                    _sizes(x),
                )
            for i in range(x.rank):
                if x.rank == 5 and i == 0:
                    continue
                if x.dim(i) <= 0:
                    raise Error(
                        fn_name,
                        (
                            ": Expected input's non-batch dimensions to have"
                            " positive length, but input has a shape of "
                        ),
                        _sizes(x),
                        " and non-batch dimension ",
                        x.dim(i),
                        " has length zero!",
                    )
            if check_input_size:
                for i in range(3):
                    if ins[i] < self.k[i]:
                        raise Error(
                            "input image (T: ",
                            ins[0],
                            " H: ",
                            ins[1],
                            " W: ",
                            ins[2],
                            ") smaller than kernel size (kT: ",
                            self.k[0],
                            " kH: ",
                            self.k[1],
                            " kW: ",
                            self.k[2],
                            ")",
                        )
        for i in range(3 - n, 3):
            if self.k[i] // 2 < self.p[i]:
                raise Error(
                    (
                        "pad should be smaller than or equal to half of kernel"
                        " size, but got "
                    ),
                    self._fmt(self.p, n, "pad"),
                )
        for i in range(3 - n, 3):
            if outs[i] < 1:
                raise Error(
                    "Given input size: (",
                    _planes_str(x, n),
                    "). Calculated output size: (",
                    _out_str(x, n, outs),
                    "). Output size is too small",
                )

    def _fmt(self, v: IndexList[3], n: Int, tag: StaticString) -> String:
        var names = ["T", "H", "W"]
        var s = String()
        for i in range(3 - n, 3):
            if i > 3 - n:
                s += " "
            s += String(tag) + names[i] + ": " + String(v[i])
        return s


def _planes_str(x: T, n: Int) -> String:
    var s = String(x.dim(x.rank - n - 1))
    for i in range(n):
        s += "x" + String(x.dim(x.rank - n + i))
    return s


def _out_str(x: T, n: Int, outs: IndexList[3]) -> String:
    var s = String(x.dim(x.rank - n - 1))
    for i in range(3 - n, 3):
        s += "x" + String(outs[i])
    return s


def _max_window[
    n: Int
](x: T, args: Values, base: Int, name: StaticString) raises -> Window:
    return Window(
        n,
        name,
        IntList(args[unsafe_offset=base]),
        IntList(args[unsafe_offset=base + 1]),
        IntList(args[unsafe_offset=base + 2]),
        IntList(args[unsafe_offset=base + 3]),
        True,
        v_bool(args[unsafe_offset=base + 4]),
    )


def _max_pool_check[
    n: Int
](x: T, w: Window, name: StaticString) raises -> IndexList[3]:
    """Output spatial size of a max pool, after torch's checks."""
    if n == 2:
        if x.rank != 3 and x.rank != 4:
            raise Error(
                "non-empty 3D or 4D (batch mode) tensor expected for input"
            )
    elif x.rank != 4 and x.rank != 5:
        # pool3d_shape_check reports it; sizes below would misread dims.
        raise Error(
            name, ": Expected 4D or 5D tensor for input, but got: ", _sizes(x)
        )
    var ins = _in_spatial(x, n)
    var outs = w.out_spatial(ins, n)
    w.shape_check(x, n, ins, outs, name, False)
    return outs


def _outputs(
    x: T,
    n: Int,
    outs: IndexList[3],
    dest: Optional[T],
    dest_indices: Optional[T],
) raises -> List[Owned]:
    """The pooled output and its int64 indices of `x` (the caller's `out=`
    tensors when given)."""
    var sh = _shape_with(x, n, outs)
    var r = List[Owned]()
    r.append(_alloc(dest, sh[0], sh[1], x.stype, x.device, [x.copy()]))
    r.append(_alloc(dest_indices, sh[0], sh[1], ST_INT64, x.device, [x.copy()]))
    return r^


def _max_pool[
    n: Int
](
    x: T,
    w: Window,
    outs: IndexList[3],
    dest: Optional[T] = None,
    dest_indices: Optional[T] = None,
) raises -> List[Owned]:
    var r = _outputs(x, n, outs, dest, dest_indices)
    if r[0].t.numel:
        var xc = own_if_new(contiguous(x), x)
        _launch(
            "MaxPool",
            x.dtype,
            r[0].t.ptr,
            r[1].t.ptr,
            xc.t.ptr,
            _geom(
                _planes(x, n),
                _in_spatial(x, n),
                outs,
                w.k,
                w.s,
                w.p,
                w.d,
                False,
                0,
                n,
            ),
            x.device,
        )
        _ = xc^
    return r^


def _max_pool_name[n: Int]() -> StaticString:
    return (
        "max_pool2d" if n
        == 2 else "max_pool3d_with_indices_out_cuda_template()"
    )


# aten::max_pool{2,3}d_with_indices(Tensor self, int[N] kernel_size,
#   int[N] stride=[], int[N] padding=0, int[N] dilation=1,
#   bool ceil_mode=False) -> (Tensor, Tensor)
def op_max_pool_with_indices[
    n: Int
](args: Values, n_args: Int, rets: Values, n_rets: Int) raises:
    var x = v_tensor(args[unsafe_offset=0])
    comptime name: StaticString = "max_pool2d" if n == 2 else "max_pool3d"
    var w = _max_window[n](x, args, 1, name)
    var outs = _max_pool_check[n](x, w, _max_pool_name[n]())
    _check_dtype(x, name)
    var r = _max_pool[n](x, w, outs)
    ret_owned(rets, 0, r[0])
    ret_owned(rets, 1, r[1])
    _ = r^


# aten::max_pool{2,3}d_with_indices.out(..., *, Tensor(a!) out,
#   Tensor(b!) indices) -> (Tensor(a!), Tensor(b!))
def op_max_pool_with_indices_out[
    n: Int
](args: Values, n_args: Int, rets: Values, n_rets: Int) raises:
    var x = v_tensor(args[unsafe_offset=0])
    var out = v_tensor(args[unsafe_offset=6])
    var indices = v_tensor(args[unsafe_offset=7])
    comptime name: StaticString = "max_pool2d" if n == 2 else "max_pool3d"
    var w = _max_window[n](x, args, 1, name)
    var outs = _max_pool_check[n](x, w, _max_pool_name[n]())
    _check_dtype(x, name)
    check_out(out, x)
    check_out_as(indices, ST_INT64, x)
    var r = _max_pool[n](x, w, outs, out.copy(), indices.copy())
    _store(rets, 1, indices, r.pop())
    _store(rets, 0, out, r.pop())


def _grad_matches(
    grad: T, x: T, n: Int, outs: IndexList[3], what: StaticString
) raises:
    """`check_dim_size` over the channel and spatial dims (and the batch)."""
    if grad.rank != x.rank:
        raise Error(
            "Expected ",
            what,
            " to have ",
            x.rank,
            " dimensions, but got ",
            grad.rank,
        )
    for i in range(x.rank - n):
        if grad.dim(i) != x.dim(i):
            raise Error(
                "Need ",
                what,
                " of size ",
                x.dim(i),
                " at dimension ",
                i,
                " but got ",
                _sizes(grad),
            )
    for i in range(n):
        if grad.dim(x.rank - n + i) != outs[3 - n + i]:
            raise Error(
                "Need ",
                what,
                " of size ",
                outs[3 - n + i],
                " at dimension ",
                x.rank - n + i,
                " but got ",
                _sizes(grad),
            )


def _max_pool_backward[
    n: Int
](args: Values, dest: Optional[T] = None) raises -> Owned:
    var grad = v_tensor(args[unsafe_offset=0])
    var x = v_tensor(args[unsafe_offset=1])
    var indices = v_tensor(args[unsafe_offset=7])
    comptime name: StaticString = "max_pool2d" if n == 2 else "max_pool3d"
    var w = Window(
        n,
        name,
        IntList(args[unsafe_offset=2]),
        IntList(args[unsafe_offset=3]),
        IntList(args[unsafe_offset=4]),
        IntList(args[unsafe_offset=5]),
        True,
        v_bool(args[unsafe_offset=6]),
    )
    if grad.stype != x.stype:
        raise Error(
            "expected dtype ",
            x.dtype,
            " for `gradOutput` but got dtype ",
            grad.dtype,
        )
    var outs = _max_pool_check[n](x, w, _max_pool_name[n]())
    _grad_matches(grad, x, n, outs, "gradOutput")
    _grad_matches(indices, x, n, outs, "indices")
    _check_dtype(x, name)
    _same_device(x, grad, name)
    _same_device(x, indices, name)
    if indices.stype != ST_INT64:
        raise Error("indices must be an int64 tensor")
    var gin = _alloc(
        dest, x.shape, x.rank, x.stype, x.device, [grad.copy(), indices.copy()]
    )
    if n == 3:
        # CUDA's max_pool3d backward scatters (atomics).
        alert_not_deterministic(
            "max_pool3d_with_indices_backward_out_cuda" if dest else "max_pool3d_with_indices_backward_cuda"
        )
        _scatter_backward(gin.t, grad, indices, x, _numel(outs), n)
    elif gin.t.numel:
        var gc = own_if_new(contiguous(grad), grad)
        var ic = own_if_new(contiguous(indices), indices)
        _launch(
            "MaxPoolBackward",
            x.dtype,
            gin.t.ptr,
            gc.t.ptr,
            ic.t.ptr,
            _geom(
                _planes(x, n),
                _in_spatial(x, n),
                outs,
                w.k,
                w.s,
                w.p,
                w.d,
                False,
                0,
                n,
            ),
            x.device,
        )
        _ = gc^
        _ = ic^
    return gin^


# aten::max_pool{2,3}d_with_indices_backward(Tensor grad_output, Tensor self,
#   int[N] kernel_size, int[N] stride, int[N] padding, int[N] dilation,
#   bool ceil_mode, Tensor indices) -> Tensor
def op_max_pool_backward[
    n: Int
](args: Values, n_args: Int, rets: Values, n_rets: Int) raises:
    var gin = _max_pool_backward[n](args)
    ret_owned(rets, 0, gin)


# aten::max_pool{2,3}d_with_indices_backward.grad_input(..., *,
#   Tensor(a!) grad_input) -> Tensor(a!)
def op_max_pool_backward_out[
    n: Int
](args: Values, n_args: Int, rets: Values, n_rets: Int) raises:
    var dest = v_tensor(args[unsafe_offset=8])
    check_out(dest, v_tensor(args[unsafe_offset=1]))
    _store(rets, 0, dest, _max_pool_backward[n](args, dest.copy()))


def _scatter_backward(
    gin: T, grad: T, indices: T, x: T, out_plane: Int, n: Int
) raises:
    """grad_input = 0, then grad_input[plane][indices[i]] += grad[i] for every
    grad_output element: CUDA's atomic max-pool backward, so an index
    outside its output's window still receives its gradient. Accumulates in
    float32 (float64) and casts once into a half `gin`."""
    fill_value(gin, 0.0)
    if gin.numel == 0 or grad.numel == 0:
        return
    var gc = own_if_new(contiguous(grad), grad)
    var ic = own_if_new(contiguous(indices), indices)
    var wide = x.dtype == DType.float32 or x.dtype == DType.float64
    var ws = own(
        new_tensor(
            x.shape, x.rank, ST_FLOAT32, x.device
        ) if not wide else gin.copy()
    )
    if wide:
        _ = ws.take()  # `gin` itself: not ours to release
    else:
        fill_value(ws.t, 0.0)
    var ctx = ctx_for(x.device)
    var call = KernelCall("pool", "MaxPoolScatter")
    call.arg_dtype(0, x.dtype)
    call.int(ws.t.ptr)
    call.int(gc.t.ptr)
    call.int(ic.t.ptr)
    call.int(grad.numel)
    call.int(out_plane)
    call.int(_numel(_in_spatial(x, n)))
    call.int(ctx_ptr(ctx))
    call.run()
    _ = ctx
    _ = gc^
    _ = ic^
    if not wide:
        cast_into(gin, ws.t)
    _ = ws^


# ---------------------------------------------------------------------------
# Average pooling
# ---------------------------------------------------------------------------


def _avg_name[n: Int]() -> StaticString:
    return "avg_pool2d" if n == 2 else "avg_pool3d"


def _avg_check[
    n: Int
](x: T, w: Window, divisor: Optional[Int]) raises -> IndexList[3]:
    """AveragePool{2,3}d.cpp's meta checks; the output spatial size."""
    if n == 3 and x.rank != 4 and x.rank != 5:
        raise Error("non-empty 4D or 5D (batch mode) tensor expected for input")
    if divisor and divisor.value() == 0:
        raise Error("divisor must be not zero")
    if n == 2 and x.rank < 3:
        raise Error(
            (
                "Expected 3D or 4D (batch mode) tensor with optional 0 dim"
                " batch size for input, but got:"
            ),
            _sizes(x),
        )
    var ins = _in_spatial(x, n)
    var outs = w.out_spatial(ins, n)
    w.shape_check(x, n, ins, outs, "avg_pool3d()", True)
    return outs


def _avg_args[
    n: Int
](args: Values, base: Int) raises -> Tuple[Window, Bool, Optional[Int]]:
    """(window, count_include_pad, divisor_override) from `kernel_size` at
    `base`."""
    var w = Window(
        n,
        _avg_name[n](),
        IntList(args[unsafe_offset=base]),
        IntList(args[unsafe_offset=base + 1]),
        IntList(args[unsafe_offset=base + 2]),
        IntList(args[unsafe_offset=base]),
        False,
        v_bool(args[unsafe_offset=base + 3]),
    )
    var cip = v_bool(args[unsafe_offset=base + 4])
    var div = Optional[Int](None)
    if not v_is_none(args[unsafe_offset=base + 5]):
        div = v_int(args[unsafe_offset=base + 5])
    return (w^, cip, div)


def _avg_pool[n: Int](args: Values, dest: Optional[T] = None) raises -> Owned:
    var x = v_tensor(args[unsafe_offset=0])
    var a = _avg_args[n](args, 1)
    var outs = _avg_check[n](x, a[0], a[2])
    _check_dtype(x, _avg_name[n]())
    var sh = _shape_with(x, n, outs)
    var out = _alloc(dest, sh[0], sh[1], x.stype, x.device, [x.copy()])
    if out.t.numel:
        var xc = own_if_new(contiguous(x), x)
        _launch(
            "AvgPool",
            x.dtype,
            out.t.ptr,
            xc.t.ptr,
            0,
            _geom(
                _planes(x, n),
                _in_spatial(x, n),
                outs,
                a[0].k,
                a[0].s,
                a[0].p,
                a[0].d,
                a[1],
                a[2].value() if a[2] else 0,
                n,
            ),
            x.device,
        )
        _ = xc^
    return out^


# aten::avg_pool{2,3}d(Tensor self, int[N] kernel_size, int[N] stride=[],
#   int[N] padding=0, bool ceil_mode=False, bool count_include_pad=True,
#   int? divisor_override=None) -> Tensor
def op_avg_pool[
    n: Int
](args: Values, n_args: Int, rets: Values, n_rets: Int) raises:
    var out = _avg_pool[n](args)
    ret_owned(rets, 0, out)


# aten::avg_pool{2,3}d.out(..., *, Tensor(a!) out) -> Tensor(a!)
def op_avg_pool_out[
    n: Int
](args: Values, n_args: Int, rets: Values, n_rets: Int) raises:
    var dest = v_tensor(args[unsafe_offset=7])
    check_out(dest, v_tensor(args[unsafe_offset=0]))
    _store(rets, 0, dest, _avg_pool[n](args, dest.copy()))


def _avg_pool_backward[
    n: Int
](args: Values, dest: Optional[T] = None) raises -> Owned:
    var grad = v_tensor(args[unsafe_offset=0])
    var x = v_tensor(args[unsafe_offset=1])
    var a = _avg_args[n](args, 2)
    var outs = _avg_check[n](x, a[0], a[2])
    _grad_matches(grad, x, n, outs, "gradOutput")
    _check_dtype(x, _avg_name[n]())
    _same_device(x, grad, _avg_name[n]())
    if grad.stype != x.stype:
        raise Error(
            "expected dtype ",
            x.dtype,
            " for `gradOutput` but got dtype ",
            grad.dtype,
        )
    var gin = _alloc(dest, x.shape, x.rank, x.stype, x.device, [grad.copy()])
    if gin.t.numel:
        var gc = own_if_new(contiguous(grad), grad)
        _launch(
            "AvgPoolBackward",
            x.dtype,
            gin.t.ptr,
            gc.t.ptr,
            0,
            _geom(
                _planes(x, n),
                _in_spatial(x, n),
                outs,
                a[0].k,
                a[0].s,
                a[0].p,
                a[0].d,
                a[1],
                a[2].value() if a[2] else 0,
                n,
            ),
            x.device,
        )
        _ = gc^
    return gin^


# aten::avg_pool{2,3}d_backward(Tensor grad_output, Tensor self,
#   int[N] kernel_size, int[N] stride, int[N] padding, bool ceil_mode,
#   bool count_include_pad, int? divisor_override) -> Tensor
def op_avg_pool_backward[
    n: Int
](args: Values, n_args: Int, rets: Values, n_rets: Int) raises:
    var gin = _avg_pool_backward[n](args)
    ret_owned(rets, 0, gin)


# aten::avg_pool{2,3}d_backward.grad_input(..., *, Tensor(a!) grad_input)
def op_avg_pool_backward_out[
    n: Int
](args: Values, n_args: Int, rets: Values, n_rets: Int) raises:
    var dest = v_tensor(args[unsafe_offset=8])
    check_out(dest, v_tensor(args[unsafe_offset=1]))
    _store(rets, 0, dest, _avg_pool_backward[n](args, dest.copy()))


# ---------------------------------------------------------------------------
# Adaptive pooling
# ---------------------------------------------------------------------------


def _adaptive_size[
    n: Int
](x: T, l: IntList, name: StaticString, max_pool: Bool) raises -> IndexList[3]:
    """Checks of AdaptiveMaxPooling{2,3}d.cpp's meta / the CUDA adaptive
    average templates; the requested output spatial size."""
    var lo = n + 1
    if max_pool:
        if x.rank != lo and x.rank != lo + 1:
            raise Error(
                name,
                "(): Expected ",
                lo,
                "D or ",
                lo + 1,
                "D tensor, but got: ",
                _sizes(x),
            )
        for i in range(1, x.rank):
            if x.dim(i) <= 0:
                raise Error(
                    name,
                    (
                        "(): Expected input to have non-zero size for non-batch"
                        " dimensions, but input has sizes "
                    ),
                    _sizes(x),
                    " with dimension ",
                    i,
                    " being empty",
                )
        if len(l) != n:
            raise Error(
                name, "(): internal error: output_size.size() must be ", n
            )
    else:
        if len(l) != n:
            raise Error(name, ": output_size must be ", n)
        if x.rank != lo and x.rank != lo + 1:
            raise Error(
                name,
                "(): Expected ",
                lo,
                "D or ",
                lo + 1,
                "D tensor, but got ",
                _sizes(x),
            )
        # 2-D checks the spatial dims only; 3-D every non-batch dim, so a
        # zero-channel batch raises (CUDA's adaptive_avg_pool3d templates).
        for i in range(x.rank - n if n == 2 else 1, x.rank):
            if x.dim(i) <= 0:
                raise Error(
                    name,
                    (
                        "(): Expected input to have non-zero size for non-batch"
                        " dimensions, but input has sizes "
                    ),
                    _sizes(x),
                    " with dimension ",
                    i,
                    " being empty",
                )
    var o = IndexList[3](1)
    for i in range(n):
        if l[i] < 0:
            raise Error(
                "Trying to create tensor with negative dimension ", l[i]
            )
        o[3 - n + i] = l[i]
    return o


def _adaptive_geom(x: T, n: Int, outs: IndexList[3]) -> List[Int]:
    var one = IndexList[3](1)
    return _geom(
        _planes(x, n),
        _in_spatial(x, n),
        outs,
        one,
        one,
        IndexList[3](0),
        one,
        False,
        0,
        n,
    )


def _adaptive_avg[
    n: Int
](x: T, l: IntList, dest: Optional[T] = None) raises -> Owned:
    comptime name: StaticString = "adaptive_avg_pool2d" if n == 2 else "adaptive_avg_pool3d"
    var outs = _adaptive_size[n](x, l, name, False)
    _check_dtype(x, name)
    var sh = _shape_with(x, n, outs)
    var out = _alloc(dest, sh[0], sh[1], x.stype, x.device, [x.copy()])
    if out.t.numel:
        var xc = own_if_new(contiguous(x), x)
        _launch(
            "AdaptiveAvgPool",
            x.dtype,
            out.t.ptr,
            xc.t.ptr,
            0,
            _adaptive_geom(x, n, outs),
            x.device,
        )
        _ = xc^
    return out^


# aten::_adaptive_avg_pool{2,3}d(Tensor self, SymInt[N] output_size) -> Tensor
def op_adaptive_avg_pool[
    n: Int
](args: Values, n_args: Int, rets: Values, n_rets: Int) raises:
    var x = v_tensor(args[unsafe_offset=0])
    var out = _adaptive_avg[n](x, IntList(args[unsafe_offset=1]))
    ret_owned(rets, 0, out)


# aten::adaptive_avg_pool{2,3}d.out(Tensor self, SymInt[N] output_size, *,
#   Tensor(a!) out) -> Tensor(a!)
def op_adaptive_avg_pool_out[
    n: Int
](args: Values, n_args: Int, rets: Values, n_rets: Int) raises:
    var x = v_tensor(args[unsafe_offset=0])
    var dest = v_tensor(args[unsafe_offset=2])
    check_out(dest, x)
    _store(
        rets,
        0,
        dest,
        _adaptive_avg[n](x, IntList(args[unsafe_offset=1]), dest.copy()),
    )


def _adaptive_grad_check[
    n: Int
](grad: T, x: T, name: StaticString) raises -> IndexList[3]:
    """The grad_output's rank / non-empty checks; its spatial size."""
    if grad.rank != n + 1 and grad.rank != n + 2:
        raise Error(
            name,
            "(): Expected ",
            n + 1,
            "D or ",
            n + 2,
            "D grad_output, but got: ",
            _sizes(grad),
        )
    for i in range(1, grad.rank):
        if grad.dim(i) <= 0:
            raise Error(
                name,
                (
                    "(): Expected grad_output to have non-zero size for"
                    " non-batch dimensions, but grad_output has sizes "
                ),
                _sizes(grad),
                " with dimension ",
                i,
                " being empty",
            )
    if grad.rank != x.rank:
        raise Error(
            "expected dimensions ",
            x.rank,
            " for `grad_output` but got dimensions ",
            grad.rank,
        )
    for i in range(x.rank - n):
        if grad.dim(i) != x.dim(i):
            raise Error(
                name,
                "(): grad_output sizes ",
                _sizes(grad),
                " do not match input sizes ",
                _sizes(x),
            )
    if grad.stype != x.stype:
        raise Error(
            "expected dtype ",
            x.dtype,
            " for `grad_output` but got dtype ",
            grad.dtype,
        )
    _same_device(x, grad, name)
    return _in_spatial(grad, n)


def _adaptive_avg_backward[
    n: Int
](grad: T, x: T, dest: Optional[T] = None) raises -> Owned:
    comptime name: StaticString = "adaptive_avg_pool2d_backward" if n == 2 else "adaptive_avg_pool3d_backward"
    var outs = _adaptive_grad_check[n](grad, x, name)
    _check_dtype(x, name)
    var gin = _alloc(dest, x.shape, x.rank, x.stype, x.device, [grad.copy()])
    if gin.t.numel:
        var gc = own_if_new(contiguous(grad), grad)
        _launch(
            "AdaptiveAvgPoolBackward",
            x.dtype,
            gin.t.ptr,
            gc.t.ptr,
            0,
            _adaptive_geom(x, n, outs),
            x.device,
        )
        _ = gc^
    return gin^


# aten::_adaptive_avg_pool{2,3}d_backward(Tensor grad_output, Tensor self)
#   -> Tensor
def op_adaptive_avg_pool_backward[
    n: Int
](args: Values, n_args: Int, rets: Values, n_rets: Int) raises:
    var gin = _adaptive_avg_backward[n](
        v_tensor(args[unsafe_offset=0]), v_tensor(args[unsafe_offset=1])
    )
    ret_owned(rets, 0, gin)


# aten::adaptive_avg_pool3d_backward.grad_input(Tensor grad_output,
#   Tensor self, *, Tensor(a!) grad_input) -> Tensor(a!)
def op_adaptive_avg_pool_backward_out[
    n: Int
](args: Values, n_args: Int, rets: Values, n_rets: Int) raises:
    var x = v_tensor(args[unsafe_offset=1])
    var dest = v_tensor(args[unsafe_offset=2])
    check_out(dest, x)
    _store(
        rets,
        0,
        dest,
        _adaptive_avg_backward[n](
            v_tensor(args[unsafe_offset=0]), x, dest.copy()
        ),
    )


def _adaptive_max[
    n: Int
](
    x: T,
    l: IntList,
    dest: Optional[T] = None,
    dest_indices: Optional[T] = None,
) raises -> List[Owned]:
    comptime name: StaticString = "adaptive_max_pool2d" if n == 2 else "adaptive_max_pool3d"
    var outs = _adaptive_size[n](x, l, name, True)
    _check_dtype(x, name)
    var r = _outputs(x, n, outs, dest, dest_indices)
    if r[0].t.numel:
        var xc = own_if_new(contiguous(x), x)
        _launch(
            "AdaptiveMaxPool",
            x.dtype,
            r[0].t.ptr,
            r[1].t.ptr,
            xc.t.ptr,
            _adaptive_geom(x, n, outs),
            x.device,
        )
        _ = xc^
    return r^


# aten::adaptive_max_pool{2,3}d(Tensor self, int[N] output_size)
#   -> (Tensor, Tensor)
def op_adaptive_max_pool[
    n: Int
](args: Values, n_args: Int, rets: Values, n_rets: Int) raises:
    var r = _adaptive_max[n](
        v_tensor(args[unsafe_offset=0]), IntList(args[unsafe_offset=1])
    )
    ret_owned(rets, 0, r[0])
    ret_owned(rets, 1, r[1])
    _ = r^


# aten::adaptive_max_pool{2,3}d.out(Tensor self, int[N] output_size, *,
#   Tensor(a!) out, Tensor(b!) indices) -> (Tensor(a!), Tensor(b!))
def op_adaptive_max_pool_out[
    n: Int
](args: Values, n_args: Int, rets: Values, n_rets: Int) raises:
    var x = v_tensor(args[unsafe_offset=0])
    var out = v_tensor(args[unsafe_offset=2])
    var indices = v_tensor(args[unsafe_offset=3])
    check_out(out, x)
    check_out_as(indices, ST_INT64, x)
    var r = _adaptive_max[n](
        x, IntList(args[unsafe_offset=1]), out.copy(), indices.copy()
    )
    _store(rets, 1, indices, r.pop())
    _store(rets, 0, out, r.pop())


def _adaptive_max_backward[
    n: Int
](grad: T, x: T, indices: T, dest: Optional[T] = None) raises -> Owned:
    comptime name: StaticString = "adaptive_max_pool2d_backward" if n == 2 else "adaptive_max_pool3d_backward"
    if indices.rank != x.rank:
        raise Error(
            "expected dimensions ",
            x.rank,
            " for `indices` but got dimensions ",
            indices.rank,
        )
    var outs = _adaptive_grad_check[n](grad, x, name)
    if not indices.same_shape(grad):
        raise Error(
            "expected sizes ",
            _sizes(indices),
            " for `grad_output` but got sizes ",
            _sizes(grad),
        )
    if indices.stype != ST_INT64:
        raise Error("indices must be an int64 tensor")
    _same_device(x, indices, name)
    _check_dtype(x, name)
    var gin = _alloc(
        dest, x.shape, x.rank, x.stype, x.device, [grad.copy(), indices.copy()]
    )
    comptime if n == 2:
        # CUDA's adaptive max backwards scatter (the 2-D one with atomics).
        alert_not_deterministic("adaptive_max_pool2d_backward_cuda")
    _scatter_backward(gin.t, grad, indices, x, _numel(outs), n)
    return gin^


# aten::adaptive_max_pool{2,3}d_backward(Tensor grad_output, Tensor self,
#   Tensor indices) -> Tensor
def op_adaptive_max_pool_backward[
    n: Int
](args: Values, n_args: Int, rets: Values, n_rets: Int) raises:
    var gin = _adaptive_max_backward[n](
        v_tensor(args[unsafe_offset=0]),
        v_tensor(args[unsafe_offset=1]),
        v_tensor(args[unsafe_offset=2]),
    )
    ret_owned(rets, 0, gin)


# aten::adaptive_max_pool{2,3}d_backward.grad_input(..., *,
#   Tensor(a!) grad_input) -> Tensor(a!)
def op_adaptive_max_pool_backward_out[
    n: Int
](args: Values, n_args: Int, rets: Values, n_rets: Int) raises:
    var x = v_tensor(args[unsafe_offset=1])
    var dest = v_tensor(args[unsafe_offset=3])
    check_out(dest, x)
    _store(
        rets,
        0,
        dest,
        _adaptive_max_backward[n](
            v_tensor(args[unsafe_offset=0]),
            x,
            v_tensor(args[unsafe_offset=2]),
            dest.copy(),
        ),
    )


# ---------------------------------------------------------------------------
# Max unpooling
# ---------------------------------------------------------------------------


def _max_unpool[n: Int](args: Values, dest: Optional[T] = None) raises -> Owned:
    """MaxUnpooling.cu's forward: zero-filled output, then out[plane][idx]
    = input for every input element."""
    var x = v_tensor(args[unsafe_offset=0])
    var indices = v_tensor(args[unsafe_offset=1])
    var osize = IntList(args[unsafe_offset=2])
    comptime name: StaticString = "max_unpooling2d_forward_out_cuda()" if n == 2 else "max_unpooling3d_forward_out_cuda()"
    alert_not_deterministic(
        "max_unpooling2d_forward_out" if n
        == 2 else "max_unpooling3d_forward_out"
    )
    if indices.stype != ST_INT64:
        raise Error(
            "elements in indices should be type int64 but got: ",
            indices.dtype,
        )
    _same_device(x, indices, name)
    if n == 2:
        for i in range(1, x.rank):
            if x.dim(i) <= 0:
                raise Error(
                    (
                        "max_unpooling2d_forward_out_cuda(): Expected input to"
                        " have non-zero size for non-batch dimensions, but got "
                    ),
                    _sizes(x),
                    " with dimension ",
                    i,
                    " being empty.",
                )
        if x.rank != 3 and x.rank != 4:
            raise Error(
                (
                    "Input to max_unpooling2d should be a 3d or 4d Tensor, but"
                    " got tensor with dimension: "
                ),
                x.rank,
            )
        if not x.same_shape(indices):
            raise Error(
                "Expected shape of indices to be: ",
                _sizes(x),
                " but got: ",
                _sizes(indices),
            )
        if len(osize) != 2:
            raise Error(
                (
                    "There should be exactly two elements (height, width) in"
                    " output_size, but got "
                ),
                len(osize),
                " elements.",
            )
        if osize[0] < 0 or osize[1] < 0:
            raise Error(
                (
                    "max_unpooling2d(): output_size must contain non-negative"
                    " spatial dimensions, but got output_size=("
                ),
                osize[0],
                ", ",
                osize[1],
                ")",
            )
    else:
        var stride = IntList(args[unsafe_offset=3])
        var padding = IntList(args[unsafe_offset=4])
        if x.rank != 4 and x.rank != 5:
            raise Error(
                (
                    "Input to max_unpooling3d should be a 4d or 5d Tensor, but"
                    " got a tensor with dim "
                ),
                x.rank,
            )
        if len(osize) != 3:
            raise Error(
                (
                    "There should be exactly three elements (depth, height,"
                    " width) in output_size, but got "
                ),
                len(osize),
                " elements.",
            )
        if len(stride) != 3:
            raise Error(
                (
                    "There should be exactly three elements (depth, height,"
                    " width) in stride, but got: "
                ),
                len(stride),
                " elements.",
            )
        if len(padding) != 3:
            raise Error(
                (
                    "There should be exactly three elements (depth, height,"
                    " width) in padding, but got: "
                ),
                len(padding),
                " elements.",
            )
        if not x.same_shape(indices):
            raise Error(
                "Expected shape of indices to be: ",
                _sizes(x),
                " but got: ",
                _sizes(indices),
            )
        for i in range(1, x.rank):
            if x.dim(i) <= 0:
                raise Error(
                    (
                        "max_unpooling3d_forward_out_cuda(): Expected input to"
                        " have non-zero size for non-batch dimensions, but got "
                    ),
                    _sizes(x),
                    " with dimension ",
                    i,
                    " being empty.",
                )
        if stride[0] <= 0 or stride[1] <= 0 or stride[2] <= 0:
            raise Error("strides should be greater than zero, but got stride: ")
        for i in range(3):
            if osize[i] < 0:
                raise Error(
                    "max_unpooling3d(): output_size must contain non-negative"
                    " spatial dimensions"
                )
    _check_dtype(x, "max_unpool" + String(n) + "d")
    var outs = IndexList[3](1)
    for i in range(n):
        outs[3 - n + i] = osize[i]
    var sh = _shape_with(x, n, outs)
    var out = _alloc(
        dest, sh[0], sh[1], x.stype, x.device, [x.copy(), indices.copy()]
    )
    if out.t.numel:
        fill_value(out.t, 0.0)
    if x.numel and out.t.numel:
        var xc = own_if_new(contiguous(x), x)
        var ic = own_if_new(contiguous(indices), indices)
        var ctx = ctx_for(x.device)
        # [bad, index]: the kernel records an index outside the output
        # plane there instead of writing it (CUDA device-asserts).
        var two = IndexList[MAX_RANK](1)
        two[MAX_RANK - 1] = 2
        var flag = own(new_tensor(two, 1, ST_INT64, x.device))
        fill_value(flag.t, 0.0)
        var call = KernelCall("pool", "MaxUnpool")
        call.arg_dtype(0, x.dtype)
        call.int(out.t.ptr)
        call.int(xc.t.ptr)
        call.int(ic.t.ptr)
        call.int(x.numel)
        call.int(_numel(_in_spatial(x, n)))
        call.int(_numel(outs))
        call.int(flag.t.ptr)
        call.int(ctx_ptr(ctx))
        call.run()
        var host = own(cpu_empty(two, 1, ST_INT64))
        copy_to_host(ctx, flag.t.ptr, host.t.ptr, 16)
        var words = Pointer[Int64, MutUntrackedOrigin](
            unsafe_from_address=host.t.ptr
        )
        var bad = words[unsafe_offset=0] != 0
        var bad_index = Int(words[unsafe_offset=1])
        _ = host^
        _ = flag^
        _ = ctx
        _ = xc^
        _ = ic^
        if bad:
            # The CPU kernel's message (MaxUnpoolKernel.cpp).
            var size = String(outs[3 - n])
            for i in range(4 - n, 3):
                size += "x" + String(outs[i])
            raise Error(
                "Found an invalid max index: ",
                bad_index,
                " (output volumes are of size ",
                size,
                ")",
            )
    return out^


# aten::max_unpool2d(Tensor self, Tensor indices, SymInt[2] output_size)
# aten::max_unpool3d(Tensor self, Tensor indices, SymInt[3] output_size,
#   int[3] stride, int[3] padding) -> Tensor
def op_max_unpool[
    n: Int
](args: Values, n_args: Int, rets: Values, n_rets: Int) raises:
    var out = _max_unpool[n](args)
    ret_owned(rets, 0, out)


# aten::max_unpool{2,3}d.out(..., *, Tensor(a!) out) -> Tensor(a!)
def op_max_unpool_out[
    n: Int
](args: Values, n_args: Int, rets: Values, n_rets: Int) raises:
    var dest = v_tensor(args[unsafe_offset=3 if n == 2 else 5])
    check_out(dest, v_tensor(args[unsafe_offset=0]))
    _store(rets, 0, dest, _max_unpool[n](args, dest.copy()))


# ---------------------------------------------------------------------------
# im2col / col2im
# ---------------------------------------------------------------------------


def _pair(l: IntList, what: StaticString) raises -> Tuple[Int, Int]:
    if len(l) != 2:
        raise Error(
            "It is expected ",
            what,
            " equals to 2, but got size ",
            len(l),
        )
    return (l[0], l[1])


def _fold_params(
    channels: Int,
    in_h: Int,
    in_w: Int,
    out_h: Int,
    out_w: Int,
    k: Tuple[Int, Int],
    s: Tuple[Int, Int],
    p: Tuple[Int, Int],
    d: Tuple[Int, Int],
    batch: Int,
) -> List[Int]:
    return [
        channels,
        in_h,
        in_w,
        out_h,
        out_w,
        k[0],
        k[1],
        s[0],
        s[1],
        p[0],
        p[1],
        d[0],
        d[1],
        batch,
    ]


def _window_checks(
    k: Tuple[Int, Int],
    d: Tuple[Int, Int],
    p: Tuple[Int, Int],
    s: Tuple[Int, Int],
    stride_first: Bool,
) raises:
    if k[1] <= 0 or k[0] <= 0:
        raise Error(
            "kernel size should be greater than zero, but got kernel_height: ",
            k[0],
            " kernel_width: ",
            k[1],
        )
    if stride_first and (s[1] <= 0 or s[0] <= 0):
        raise Error(
            "stride should be greater than zero, but got stride_height: ",
            s[0],
            " stride_width: ",
            s[1],
        )
    if d[1] <= 0 or d[0] <= 0:
        raise Error(
            "dilation should be greater than zero, but got dilation_height: ",
            d[0],
            " dilation_width: ",
            d[1],
        )
    if p[1] < 0 or p[0] < 0:
        raise Error(
            "padding should be non-negative, but got pad_height: ",
            p[0],
            " pad_width: ",
            p[1],
        )
    if not stride_first and (s[1] <= 0 or s[0] <= 0):
        raise Error(
            "stride should be greater than zero, but got stride_height: ",
            s[0],
            " stride_width: ",
            s[1],
        )


def _blocks(isz: Int, pad: Int, dil: Int, k: Int, s: Int) -> Int:
    """im2col_shape_check.h `sliding_window_count` (div_rtn: floor)."""
    return (isz + 2 * pad - (dil * (k - 1) + 1)) // s + 1


def _im2col(args: Values, dest: Optional[T] = None) raises -> Owned:
    var x = v_tensor(args[unsafe_offset=0])
    var k = _pair(IntList(args[unsafe_offset=1]), "kernel_size")
    var d = _pair(IntList(args[unsafe_offset=2]), "dilation")
    var p = _pair(IntList(args[unsafe_offset=3]), "padding")
    var s = _pair(IntList(args[unsafe_offset=4]), "stride")
    _window_checks(k, d, p, s, False)
    var valid = x.rank >= 3 and x.dim(1) != 0 and x.dim(2) != 0
    if not (
        (x.rank == 3 and x.dim(0) != 0 and valid)
        or (x.rank == 4 and valid and x.dim(3) != 0)
    ):
        raise Error(
            (
                "Expected 3D or 4D (batch mode) tensor with possibly 0 batch"
                " size and other non-zero dimensions for input, but got: "
            ),
            _sizes(x),
        )
    var in_h = x.dim(-2)
    var in_w = x.dim(-1)
    var out_h = _blocks(in_h, p[0], d[0], k[0], s[0])
    var out_w = _blocks(in_w, p[1], d[1], k[1], s[1])
    if out_h < 1 or out_w < 1:
        raise Error(
            "Given input with spatial size (",
            in_h,
            ", ",
            in_w,
            "), kernel_size=(",
            k[0],
            ", ",
            k[1],
            "), dilation=(",
            d[0],
            ", ",
            d[1],
            "), padding=(",
            p[0],
            ", ",
            p[1],
            "), calculated shape of the array of sliding blocks as (",
            out_h,
            ", ",
            out_w,
            "), but its components must be at least one.",
        )
    _check_dtype(x, "im2col", True)
    var batch = x.dim(0) if x.rank == 4 else 1
    var channels = x.dim(-3)
    var shape = IndexList[MAX_RANK](1)
    shape[MAX_RANK - 2] = channels * k[0] * k[1]
    shape[MAX_RANK - 1] = out_h * out_w
    var rank = 2
    if x.rank == 4:
        shape[MAX_RANK - 3] = batch
        rank = 3
    var out = _alloc(dest, shape, rank, x.stype, x.device, [x.copy()])
    if out.t.numel:
        var xc = own_if_new(contiguous(x), x)
        _launch(
            "Im2col",
            x.dtype,
            out.t.ptr,
            xc.t.ptr,
            0,
            _fold_params(channels, in_h, in_w, out_h, out_w, k, s, p, d, batch),
            x.device,
        )
        _ = xc^
    return out^


# aten::im2col(Tensor self, int[2] kernel_size, int[2] dilation,
#   int[2] padding, int[2] stride) -> Tensor
def op_im2col(args: Values, n_args: Int, rets: Values, n_rets: Int) raises:
    var out = _im2col(args)
    ret_owned(rets, 0, out)


# aten::im2col.out(..., *, Tensor(a!) out) -> Tensor(a!)
def op_im2col_out(args: Values, n_args: Int, rets: Values, n_rets: Int) raises:
    var dest = v_tensor(args[unsafe_offset=5])
    check_out(dest, v_tensor(args[unsafe_offset=0]))
    _store(rets, 0, dest, _im2col(args, dest.copy()))


def _col2im(args: Values, dest: Optional[T] = None) raises -> Owned:
    var x = v_tensor(args[unsafe_offset=0])
    var o = _pair(IntList(args[unsafe_offset=1]), "output_size")
    var k = _pair(IntList(args[unsafe_offset=2]), "kernel_size")
    var d = _pair(IntList(args[unsafe_offset=3]), "dilation")
    var p = _pair(IntList(args[unsafe_offset=4]), "padding")
    var s = _pair(IntList(args[unsafe_offset=5]), "stride")
    _window_checks(k, d, p, s, True)
    if not (
        (x.rank == 2 and x.dim(0) != 0 and x.dim(1) != 0)
        or (x.rank == 3 and x.dim(1) != 0 and x.dim(2) != 0)
    ):
        raise Error(
            (
                "Expected 2D or 3D (batch mode) tensor for input with possibly"
                " 0 batch size and non-zero dimensions for input, but got: "
            ),
            _sizes(x),
        )
    var planes_in = x.dim(-2)
    var kk = k[0] * k[1]
    if planes_in % kk != 0:
        raise Error(
            (
                "Expected size of input's dimension 1 to be divisible by the"
                " product of kernel_size, but got input.size(1)="
            ),
            planes_in,
            " and kernel_size=(",
            k[0],
            ", ",
            k[1],
            ").",
        )
    var length = x.dim(-1)
    var bh = _blocks(o[0], p[0], d[0], k[0], s[0])
    var bw = _blocks(o[1], p[1], d[1], k[1], s[1])
    if length != bh * bw:
        raise Error(
            "Given output_size=(",
            o[0],
            ", ",
            o[1],
            "), kernel_size=(",
            k[0],
            ", ",
            k[1],
            "), dilation=(",
            d[0],
            ", ",
            d[1],
            "), padding=(",
            p[0],
            ", ",
            p[1],
            "), stride=(",
            s[0],
            ", ",
            s[1],
            (
                "), expected size of input's dimension 2 to match the"
                " calculated number of sliding blocks "
            ),
            bh,
            " * ",
            bw,
            " = ",
            bh * bw,
            ", but got input.size(2)=",
            length,
            ".",
        )
    if bh < 1 or bw < 1:
        raise Error(
            "Given output_size=(",
            o[0],
            ", ",
            o[1],
            "), calculated shape of the array of sliding blocks as (",
            bh,
            ", ",
            bw,
            "), which is too small (non-positive)",
        )
    if o[1] < 1 or o[0] < 1:
        raise Error(
            (
                "Expected output spatial size to be positive, but got:"
                " output_size=("
            ),
            o[0],
            ", ",
            o[1],
            ").",
        )
    _check_dtype(x, "col2im", True)
    var batch = x.dim(0) if x.rank == 3 else 1
    var channels = planes_in // kk
    var shape = IndexList[MAX_RANK](1)
    shape[MAX_RANK - 3] = channels
    shape[MAX_RANK - 2] = o[0]
    shape[MAX_RANK - 1] = o[1]
    var rank = 3
    if x.rank == 3:
        shape[MAX_RANK - 4] = batch
        rank = 4
    var out = _alloc(dest, shape, rank, x.stype, x.device, [x.copy()])
    if out.t.numel:
        var xc = own_if_new(contiguous(x), x)
        _launch(
            "Col2im",
            x.dtype,
            out.t.ptr,
            xc.t.ptr,
            0,
            _fold_params(channels, o[0], o[1], bh, bw, k, s, p, d, batch),
            x.device,
        )
        _ = xc^
    return out^


# aten::col2im(Tensor self, SymInt[2] output_size, int[2] kernel_size,
#   int[2] dilation, int[2] padding, int[2] stride) -> Tensor
def op_col2im(args: Values, n_args: Int, rets: Values, n_rets: Int) raises:
    var out = _col2im(args)
    ret_owned(rets, 0, out)


# aten::col2im.out(..., *, Tensor(a!) out) -> Tensor(a!)
def op_col2im_out(args: Values, n_args: Int, rets: Values, n_rets: Int) raises:
    var dest = v_tensor(args[unsafe_offset=6])
    check_out(dest, v_tensor(args[unsafe_offset=0]))
    _store(rets, 0, dest, _col2im(args, dest.copy()))


# Per-rank instantiations, one name per registered overload.
comptime op_adaptive_avg_pool2d = op_adaptive_avg_pool[2]
comptime op_adaptive_avg_pool2d_backward = op_adaptive_avg_pool_backward[2]
comptime op_adaptive_avg_pool3d = op_adaptive_avg_pool[3]
comptime op_adaptive_avg_pool3d_backward = op_adaptive_avg_pool_backward[3]
comptime op_adaptive_avg_pool2d_out = op_adaptive_avg_pool_out[2]
comptime op_adaptive_avg_pool3d_out = op_adaptive_avg_pool_out[3]
comptime op_adaptive_avg_pool3d_backward_grad_input = op_adaptive_avg_pool_backward_out[
    3
]
comptime op_adaptive_max_pool2d = op_adaptive_max_pool[2]
comptime op_adaptive_max_pool2d_out = op_adaptive_max_pool_out[2]
comptime op_adaptive_max_pool2d_backward = op_adaptive_max_pool_backward[2]
comptime op_adaptive_max_pool2d_backward_grad_input = op_adaptive_max_pool_backward_out[
    2
]
comptime op_adaptive_max_pool3d = op_adaptive_max_pool[3]
comptime op_adaptive_max_pool3d_out = op_adaptive_max_pool_out[3]
comptime op_adaptive_max_pool3d_backward = op_adaptive_max_pool_backward[3]
comptime op_adaptive_max_pool3d_backward_grad_input = op_adaptive_max_pool_backward_out[
    3
]
comptime op_avg_pool2d = op_avg_pool[2]
comptime op_avg_pool2d_out = op_avg_pool_out[2]
comptime op_avg_pool2d_backward = op_avg_pool_backward[2]
comptime op_avg_pool2d_backward_grad_input = op_avg_pool_backward_out[2]
comptime op_avg_pool3d = op_avg_pool[3]
comptime op_avg_pool3d_out = op_avg_pool_out[3]
comptime op_avg_pool3d_backward = op_avg_pool_backward[3]
comptime op_avg_pool3d_backward_grad_input = op_avg_pool_backward_out[3]
comptime op_max_pool2d_with_indices = op_max_pool_with_indices[2]
comptime op_max_pool2d_with_indices_out = op_max_pool_with_indices_out[2]
comptime op_max_pool2d_with_indices_backward = op_max_pool_backward[2]
comptime op_max_pool2d_with_indices_backward_grad_input = op_max_pool_backward_out[
    2
]
comptime op_max_pool3d_with_indices = op_max_pool_with_indices[3]
comptime op_max_pool3d_with_indices_out = op_max_pool_with_indices_out[3]
comptime op_max_pool3d_with_indices_backward = op_max_pool_backward[3]
comptime op_max_pool3d_with_indices_backward_grad_input = op_max_pool_backward_out[
    3
]
comptime op_max_unpool2d = op_max_unpool[2]
comptime op_max_unpool2d_out = op_max_unpool_out[2]
comptime op_max_unpool3d = op_max_unpool[3]
comptime op_max_unpool3d_out = op_max_unpool_out[3]


def register_pooling(site: Site) raises:
    impl[op_adaptive_avg_pool2d, "_adaptive_avg_pool2d"](site)
    impl[op_adaptive_avg_pool2d_backward, "_adaptive_avg_pool2d_backward"](site)
    impl[op_adaptive_avg_pool3d, "_adaptive_avg_pool3d"](site)
    impl[op_adaptive_avg_pool3d_backward, "_adaptive_avg_pool3d_backward"](site)
    impl[op_adaptive_avg_pool2d_out, "adaptive_avg_pool2d.out"](site)
    impl[op_adaptive_avg_pool3d_out, "adaptive_avg_pool3d.out"](site)
    impl[
        op_adaptive_avg_pool3d_backward_grad_input,
        "adaptive_avg_pool3d_backward.grad_input",
    ](site)
    impl[op_adaptive_max_pool2d, "adaptive_max_pool2d"](site)
    impl[op_adaptive_max_pool2d_out, "adaptive_max_pool2d.out"](site)
    impl[op_adaptive_max_pool2d_backward, "adaptive_max_pool2d_backward"](site)
    impl[
        op_adaptive_max_pool2d_backward_grad_input,
        "adaptive_max_pool2d_backward.grad_input",
    ](site)
    impl[op_adaptive_max_pool3d, "adaptive_max_pool3d"](site)
    impl[op_adaptive_max_pool3d_out, "adaptive_max_pool3d.out"](site)
    impl[op_adaptive_max_pool3d_backward, "adaptive_max_pool3d_backward"](site)
    impl[
        op_adaptive_max_pool3d_backward_grad_input,
        "adaptive_max_pool3d_backward.grad_input",
    ](site)
    impl[op_avg_pool2d, "avg_pool2d"](site)
    impl[op_avg_pool2d_out, "avg_pool2d.out"](site)
    impl[op_avg_pool2d_backward, "avg_pool2d_backward"](site)
    impl[op_avg_pool2d_backward_grad_input, "avg_pool2d_backward.grad_input"](
        site
    )
    impl[op_avg_pool3d, "avg_pool3d"](site)
    impl[op_avg_pool3d_out, "avg_pool3d.out"](site)
    impl[op_avg_pool3d_backward, "avg_pool3d_backward"](site)
    impl[op_avg_pool3d_backward_grad_input, "avg_pool3d_backward.grad_input"](
        site
    )
    impl[op_col2im, "col2im"](site)
    impl[op_col2im_out, "col2im.out"](site)
    impl[op_im2col, "im2col"](site)
    impl[op_im2col_out, "im2col.out"](site)
    impl[op_max_pool2d_with_indices, "max_pool2d_with_indices"](site)
    impl[op_max_pool2d_with_indices_out, "max_pool2d_with_indices.out"](site)
    impl[
        op_max_pool2d_with_indices_backward, "max_pool2d_with_indices_backward"
    ](site)
    impl[
        op_max_pool2d_with_indices_backward_grad_input,
        "max_pool2d_with_indices_backward.grad_input",
    ](site)
    impl[op_max_pool3d_with_indices, "max_pool3d_with_indices"](site)
    impl[op_max_pool3d_with_indices_out, "max_pool3d_with_indices.out"](site)
    impl[
        op_max_pool3d_with_indices_backward, "max_pool3d_with_indices_backward"
    ](site)
    impl[
        op_max_pool3d_with_indices_backward_grad_input,
        "max_pool3d_with_indices_backward.grad_input",
    ](site)
    impl[op_max_unpool2d, "max_unpool2d"](site)
    impl[op_max_unpool2d_out, "max_unpool2d.out"](site)
    impl[op_max_unpool3d, "max_unpool3d"](site)
    impl[op_max_unpool3d_out, "max_unpool3d.out"](site)
