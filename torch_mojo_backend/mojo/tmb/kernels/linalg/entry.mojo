# ===----------------------------------------------------------------------=== #
# Dense linear algebra factorizations, batched, one thread block per matrix.
#
#   Potrf    Cholesky A = L L^T of the lower triangle (LAPACK potrf, uplo L),
#            in place; info = j + 1 at the first non-positive pivot; the
#            strictly upper triangle is zeroed afterwards (torch's tril_).
#   Getrf    LU with partial pivoting (LAPACK getrf), in place: 1-based int32
#            pivots, info = j + 1 at the first exactly-zero pivot (the
#            factorization continues, as LAPACK's does). Optionally without
#            pivoting (cuSOLVER's getrf without ipiv).
#   Trsm     op(A) X = B for a triangular A, left side, in place on B
#            (LAPACK trsm). `forward` = the effective triangle is lower.
#   Laswp    the row interchanges of a Getrf, forwards or backwards.
#   Geqrf    Householder QR (LAPACK geqr2 with larfg's conventions): R on
#            and above the diagonal, the reflectors below it, tau.
#   Ormqr    C = Q C or Q^T C for the Q of a Geqrf (LAPACK orm2r, left side;
#            the right side is the left side of the transposed C).
#   Syevj    symmetric eigendecomposition by parallel cyclic two-sided Jacobi
#            (round-robin pairs), eigenvalues ascending as LAPACK's syevd.
#   Gesvdj   singular value decomposition by parallel one-sided (Hestenes)
#            Jacobi on m >= n, singular values descending.
#   Sytf2    Bunch-Kaufman LDL^T of the lower triangle (LAPACK sytf2, uplo L),
#            LAPACK's pivot encoding.
#
# Every matrix operand is addressed through (row stride, column stride, batch
# stride), so transposed and column-major views need no copy. Each block
# walks the batch with a grid stride. All arithmetic is in the element dtype
# (float32 or float64), as LAPACK's single and double routines.
# ===----------------------------------------------------------------------=== #

from std.math import fma, sqrt
from max.gpu import (
    MAX_THREADS_PER_BLOCK_METADATA,
    block_dim,
    block_idx,
    grid_dim,
    thread_idx,
)
from max.gpu.host import DeviceContext
from max.gpu.sync import barrier
from std.memory import AddressSpace, stack_allocation
from std.sys.info import has_accelerator
from std.utils.static_tuple import StaticTuple

from tmb.kernels.common.block_reduce import block_sum
from tmb.kernels.common.libdevice_port import nv_log, nv_logf
from tmb.kernels.common.op_utils import (
    Arg,
    Argv,
    _enqueue_cached,
    _make_ptr,
    _raw_ctx,
    _raw_dtype_int,
    _raw_int,
    _raw_tuple_int,
    _spec_dispatcher7,
)
from tmb.kernels.common.variant_gates import (
    ErrBuf,
    NO_OP_COMPILED,
    _dtype_arg_on,
    _op_on,
    _tmb_entry_error,
)

comptime LA_THREADS = 256
comptime LA_DTYPES = [DType.float32, DType.float64]
# Jacobi sweeps before giving up (each sweep visits every pair once);
# quadratic convergence needs well under 20 on any matrix seen in practice.
comptime JACOBI_MAX_SWEEPS = 60
# Grid cap for the per-matrix kernels: blocks walk the batch beyond it.
comptime LA_MAX_BLOCKS = 65535

comptime FPtr[dt: DType] = Pointer[Scalar[dt], MutAnyOrigin]
comptime IPtr = Pointer[Int32, MutAnyOrigin]


@always_inline
def _eps[dt: DType]() -> Scalar[dt]:
    """LAPACK's dlamch('E') * 2: the spacing of 1.0 (float32 / float64)."""
    comptime if dt == DType.float64:
        return Scalar[dt](2.220446049250313e-16)
    else:
        return Scalar[dt](1.1920928955078125e-07)


@always_inline
def _sfmin[dt: DType]() -> Scalar[dt]:
    """LAPACK's dlamch('S'): the smallest number whose reciprocal is finite."""
    comptime if dt == DType.float64:
        return Scalar[dt](2.2250738585072014e-308)
    else:
        return Scalar[dt](1.1754943508222875e-38)


@always_inline
def _abs[dt: DType](x: Scalar[dt]) -> Scalar[dt]:
    return -x if x < 0 else x


@always_inline
def _lapy2[dt: DType](x: Scalar[dt], y: Scalar[dt]) -> Scalar[dt]:
    """sqrt(x^2 + y^2) without overflow (LAPACK dlapy2)."""
    var xa = _abs(x)
    var ya = _abs(y)
    var w = max(xa, ya)
    var z = min(xa, ya)
    if z == 0 or w != w:
        return w
    var r = z / w
    return w * sqrt(1 + r * r)


@always_inline
def _blocks(n: Int) -> Int:
    return max(1, min(n, LA_MAX_BLOCKS))


# ---------------------------------------------------------------------------
# Cholesky
# ---------------------------------------------------------------------------


@__name(t"linalg_potrf_lower_{dt}_t{LA_THREADS}")
@__llvm_metadata(
    MAX_THREADS_PER_BLOCK_METADATA=StaticTuple[Int32, 1](Int32(LA_THREADS))
)
def _potrf_kernel[
    dt: DType
](
    a: FPtr[dt],
    info: IPtr,
    n64: Int64,
    rs64: Int64,
    cs64: Int64,
    bs64: Int64,
    batch64: Int64,
):
    var n = Int(n64)
    var rs = Int(rs64)
    var cs = Int(cs64)
    var bs = Int(bs64)
    var batch = Int(batch64)
    var tid = Int(thread_idx.x)
    var nt = Int(block_dim.x)
    var b = Int(block_idx.x)
    while b < batch:
        var base = b * bs
        var status = 0
        for j in range(n):
            var d = a[unsafe_offset=base + j * rs + j * cs]
            barrier()  # every thread has read the pivot before it is replaced
            if not (d > 0):  # also NaN, as LAPACK's `ajj <= 0 or isnan`
                status = j + 1
                break
            var s = sqrt(d)
            if tid == 0:
                a[unsafe_offset=base + j * rs + j * cs] = s
            var r = 1 / s
            var i = j + 1 + tid
            while i < n:
                a[unsafe_offset=base + i * rs + j * cs] *= r
                i += nt
            barrier()
            var rr = n - j - 1
            var idx = tid
            while idx < rr * rr:
                var ii = j + 1 + idx % rr
                var kk = j + 1 + idx // rr
                if ii >= kk:
                    a[unsafe_offset=base + ii * rs + kk * cs] -= (
                        a[unsafe_offset=base + ii * rs + j * cs]
                        * a[unsafe_offset=base + kk * rs + j * cs]
                    )
                idx += nt
            barrier()
        if tid == 0:
            info[unsafe_offset=b] = Int32(status)
        var idx = tid
        while idx < n * n:
            var i = idx % n
            var k = idx // n
            if i < k:
                a[unsafe_offset=base + i * rs + k * cs] = 0
            idx += nt
        barrier()
        b += Int(grid_dim.x)


# ---------------------------------------------------------------------------
# LU with partial pivoting
# ---------------------------------------------------------------------------


@__name(t"linalg_getrf_{dt}_t{LA_THREADS}")
@__llvm_metadata(
    MAX_THREADS_PER_BLOCK_METADATA=StaticTuple[Int32, 1](Int32(LA_THREADS))
)
def _getrf_kernel[
    dt: DType
](
    a: FPtr[dt],
    piv: IPtr,
    info: IPtr,
    m64: Int64,
    n64: Int64,
    rs64: Int64,
    cs64: Int64,
    bs64: Int64,
    batch64: Int64,
    pivot64: Int64,
):
    var m = Int(m64)
    var n = Int(n64)
    var rs = Int(rs64)
    var cs = Int(cs64)
    var bs = Int(bs64)
    var batch = Int(batch64)
    var pivot = pivot64 != 0
    var k = min(m, n)
    var tid = Int(thread_idx.x)
    var nt = Int(block_dim.x)
    var sv = stack_allocation[
        LA_THREADS, dt, address_space=AddressSpace.SHARED
    ]()
    var si = stack_allocation[
        LA_THREADS, DType.int32, address_space=AddressSpace.SHARED
    ]()
    var b = Int(block_idx.x)
    while b < batch:
        var base = b * bs
        var status = 0
        for j in range(k):
            var p = j
            if pivot:
                # idamax: the first row of largest magnitude (a NaN never
                # compares larger, so it only wins as the first candidate).
                var best = Scalar[dt](0)
                var bi = -1
                var i = j + tid
                while i < m:
                    var v = _abs(a[unsafe_offset=base + i * rs + j * cs])
                    if bi < 0 or v > best:
                        best = v
                        bi = i
                    i += nt
                sv[unsafe_offset=tid] = best
                si[unsafe_offset=tid] = Int32(bi)
                barrier()
                var stride = LA_THREADS // 2
                while stride > 0:
                    if tid < stride:
                        var oi = Int(si[unsafe_offset=tid + stride])
                        var mi = Int(si[unsafe_offset=tid])
                        var ov = sv[unsafe_offset=tid + stride]
                        var mv = sv[unsafe_offset=tid]
                        if oi >= 0 and (
                            mi < 0 or ov > mv or (ov == mv and oi < mi)
                        ):
                            sv[unsafe_offset=tid] = ov
                            si[unsafe_offset=tid] = Int32(oi)
                    barrier()
                    stride //= 2
                p = Int(si[unsafe_offset=0])
            var pv = a[unsafe_offset=base + p * rs + j * cs]
            barrier()  # pivot read everywhere before any swap
            if tid == 0:
                piv[unsafe_offset=b * k + j] = Int32(p + 1)
            if pv != 0:
                if p != j:
                    var c = tid
                    while c < n:
                        var x = a[unsafe_offset=base + j * rs + c * cs]
                        a[unsafe_offset=base + j * rs + c * cs] = a[
                            unsafe_offset=base + p * rs + c * cs
                        ]
                        a[unsafe_offset=base + p * rs + c * cs] = x
                        c += nt
                    barrier()
                var i = j + 1 + tid
                if _abs(pv) >= _sfmin[dt]():
                    var r = 1 / pv
                    while i < m:
                        a[unsafe_offset=base + i * rs + j * cs] *= r
                        i += nt
                else:
                    while i < m:
                        a[unsafe_offset=base + i * rs + j * cs] /= pv
                        i += nt
            elif status == 0:
                status = j + 1
            barrier()
            var rows = m - j - 1
            var cols = n - j - 1
            var idx = tid
            while idx < rows * cols:
                var ii = j + 1 + idx % rows
                var cc = j + 1 + idx // rows
                a[unsafe_offset=base + ii * rs + cc * cs] = fma(
                    -a[unsafe_offset=base + ii * rs + j * cs],
                    a[unsafe_offset=base + j * rs + cc * cs],
                    a[unsafe_offset=base + ii * rs + cc * cs],
                )
                idx += nt
            barrier()
        if tid == 0:
            info[unsafe_offset=b] = Int32(status)
        b += Int(grid_dim.x)


# ---------------------------------------------------------------------------
# Triangular solve and row interchanges
# ---------------------------------------------------------------------------

comptime TRSM_COLS = 64  # right-hand sides per block


@__name(t"linalg_trsm_left_{dt}_t{LA_THREADS}")
@__llvm_metadata(
    MAX_THREADS_PER_BLOCK_METADATA=StaticTuple[Int32, 1](Int32(LA_THREADS))
)
def _trsm_kernel[
    dt: DType
](
    a: FPtr[dt],
    bm: FPtr[dt],
    n64: Int64,
    k64: Int64,
    a_rs64: Int64,
    a_cs64: Int64,
    a_bs64: Int64,
    b_rs64: Int64,
    b_cs64: Int64,
    b_bs64: Int64,
    batch64: Int64,
    forward64: Int64,
    unit64: Int64,
):
    var n = Int(n64)
    var k = Int(k64)
    var a_rs = Int(a_rs64)
    var a_cs = Int(a_cs64)
    var a_bs = Int(a_bs64)
    var b_rs = Int(b_rs64)
    var b_cs = Int(b_cs64)
    var b_bs = Int(b_bs64)
    var batch = Int(batch64)
    var forward = forward64 != 0
    var unit = unit64 != 0
    var tid = Int(thread_idx.x)
    var nt = Int(block_dim.x)
    var chunks = (k + TRSM_COLS - 1) // TRSM_COLS
    var task = Int(block_idx.x)
    while task < batch * chunks:
        var bb = task // chunks
        var c0 = (task % chunks) * TRSM_COLS
        var w = min(k, c0 + TRSM_COLS) - c0
        var abase = bb * a_bs
        var bbase = bb * b_bs + c0 * b_cs
        for step in range(n):
            var i = step if forward else n - 1 - step
            if not unit:
                var d = a[unsafe_offset=abase + i * a_rs + i * a_cs]
                var c = tid
                while c < w:
                    bm[unsafe_offset=bbase + i * b_rs + c * b_cs] /= d
                    c += nt
                barrier()
            var r0 = i + 1 if forward else 0
            var rows = n - 1 - i if forward else i
            var idx = tid
            while idx < rows * w:
                var r = r0 + idx // w
                var c = idx % w
                bm[unsafe_offset=bbase + r * b_rs + c * b_cs] = fma(
                    -bm[unsafe_offset=bbase + i * b_rs + c * b_cs],
                    a[unsafe_offset=abase + r * a_rs + i * a_cs],
                    bm[unsafe_offset=bbase + r * b_rs + c * b_cs],
                )
                idx += nt
            barrier()
        task += Int(grid_dim.x)


@__name(t"linalg_laswp_{dt}_t{LA_THREADS}")
@__llvm_metadata(
    MAX_THREADS_PER_BLOCK_METADATA=StaticTuple[Int32, 1](Int32(LA_THREADS))
)
def _laswp_kernel[
    dt: DType
](
    bm: FPtr[dt],
    piv: IPtr,
    k64: Int64,
    ncols64: Int64,
    b_rs64: Int64,
    b_cs64: Int64,
    b_bs64: Int64,
    piv_bs64: Int64,
    batch64: Int64,
    forward64: Int64,
    nrows64: Int64,
):
    var k = Int(k64)
    var nrows = Int(nrows64)
    var ncols = Int(ncols64)
    var b_rs = Int(b_rs64)
    var b_cs = Int(b_cs64)
    var b_bs = Int(b_bs64)
    var piv_bs = Int(piv_bs64)
    var total = Int(batch64) * ncols
    var forward = forward64 != 0
    var t = Int(block_idx.x) * Int(block_dim.x) + Int(thread_idx.x)
    while t < total:
        var bb = t // ncols
        var c = t % ncols
        var base = bb * b_bs + c * b_cs
        for step in range(k):
            var i = step if forward else k - 1 - step
            var p = Int(piv[unsafe_offset=bb * piv_bs + i]) - 1
            # An out-of-range pivot (not one getrf wrote) is skipped rather
            # than followed out of the matrix.
            if p != i and p >= 0 and p < nrows and i < nrows:
                var x = bm[unsafe_offset=base + i * b_rs]
                bm[unsafe_offset=base + i * b_rs] = bm[
                    unsafe_offset=base + p * b_rs
                ]
                bm[unsafe_offset=base + p * b_rs] = x
        t += Int(grid_dim.x) * Int(block_dim.x)


# ---------------------------------------------------------------------------
# Householder QR and the application of its Q
# ---------------------------------------------------------------------------


@__name(t"linalg_geqr2_{dt}_t{LA_THREADS}")
@__llvm_metadata(
    MAX_THREADS_PER_BLOCK_METADATA=StaticTuple[Int32, 1](Int32(LA_THREADS))
)
def _geqrf_kernel[
    dt: DType
](
    a: FPtr[dt],
    tau: FPtr[dt],
    m64: Int64,
    n64: Int64,
    rs64: Int64,
    cs64: Int64,
    bs64: Int64,
    batch64: Int64,
):
    var m = Int(m64)
    var n = Int(n64)
    var rs = Int(rs64)
    var cs = Int(cs64)
    var bs = Int(bs64)
    var batch = Int(batch64)
    var k = min(m, n)
    var tid = Int(thread_idx.x)
    var nt = Int(block_dim.x)
    var b = Int(block_idx.x)
    while b < batch:
        var base = b * bs
        for j in range(k):
            var alpha = a[unsafe_offset=base + j * rs + j * cs]
            var part = Scalar[dt](0)
            var i = j + 1 + tid
            while i < m:
                var x = a[unsafe_offset=base + i * rs + j * cs]
                part += x * x
                i += nt
            var xnorm = sqrt(block_sum[dt, LA_THREADS](part))
            # larfg: H (alpha, x) = (beta, 0), H = I - tau v v^T, v = (1, x/(alpha-beta))
            var t = Scalar[dt](0)
            if xnorm != 0:
                var beta = _lapy2(alpha, xnorm)
                if alpha >= 0:
                    beta = -beta
                t = (beta - alpha) / beta
                var scal = 1 / (alpha - beta)
                i = j + 1 + tid
                while i < m:
                    a[unsafe_offset=base + i * rs + j * cs] *= scal
                    i += nt
                if tid == 0:
                    a[unsafe_offset=base + j * rs + j * cs] = beta
            if tid == 0:
                tau[unsafe_offset=b * k + j] = t
            barrier()
            if t != 0:
                var c = j + 1 + tid
                while c < n:
                    var w = a[unsafe_offset=base + j * rs + c * cs]
                    for r in range(j + 1, m):
                        w += (
                            a[unsafe_offset=base + r * rs + j * cs]
                            * a[unsafe_offset=base + r * rs + c * cs]
                        )
                    w *= t
                    a[unsafe_offset=base + j * rs + c * cs] -= w
                    for r in range(j + 1, m):
                        a[unsafe_offset=base + r * rs + c * cs] -= (
                            w * a[unsafe_offset=base + r * rs + j * cs]
                        )
                    c += nt
            barrier()
        b += Int(grid_dim.x)


@__name(t"linalg_orm2r_left_{dt}_t{LA_THREADS}")
@__llvm_metadata(
    MAX_THREADS_PER_BLOCK_METADATA=StaticTuple[Int32, 1](Int32(LA_THREADS))
)
def _ormqr_kernel[
    dt: DType
](
    a: FPtr[dt],
    tau: FPtr[dt],
    cm: FPtr[dt],
    nq64: Int64,
    ncols64: Int64,
    k64: Int64,
    a_rs64: Int64,
    a_cs64: Int64,
    a_bs64: Int64,
    tau_bs64: Int64,
    c_rs64: Int64,
    c_cs64: Int64,
    c_bs64: Int64,
    batch64: Int64,
    trans64: Int64,
):
    """One thread per column of C: every reflector touches that column
    alone, so no thread waits on another."""
    var nq = Int(nq64)
    var ncols = Int(ncols64)
    var k = Int(k64)
    var a_rs = Int(a_rs64)
    var a_cs = Int(a_cs64)
    var a_bs = Int(a_bs64)
    var tau_bs = Int(tau_bs64)
    var c_rs = Int(c_rs64)
    var c_cs = Int(c_cs64)
    var c_bs = Int(c_bs64)
    var total = Int(batch64) * ncols
    var trans = trans64 != 0
    var t = Int(block_idx.x) * Int(block_dim.x) + Int(thread_idx.x)
    while t < total:
        var bb = t // ncols
        var col = t % ncols
        var abase = bb * a_bs
        var cbase = bb * c_bs + col * c_cs
        for step in range(k):
            # Q = H(0) ... H(k-1): Q C applies H(k-1) first, Q^T C H(0).
            var i = step if trans else k - 1 - step
            var ti = tau[unsafe_offset=bb * tau_bs + i]
            if ti != 0:
                var w = cm[unsafe_offset=cbase + i * c_rs]
                for r in range(i + 1, nq):
                    w += (
                        a[unsafe_offset=abase + r * a_rs + i * a_cs]
                        * cm[unsafe_offset=cbase + r * c_rs]
                    )
                w *= ti
                cm[unsafe_offset=cbase + i * c_rs] -= w
                for r in range(i + 1, nq):
                    cm[unsafe_offset=cbase + r * c_rs] -= (
                        w * a[unsafe_offset=abase + r * a_rs + i * a_cs]
                    )
        t += Int(grid_dim.x) * Int(block_dim.x)


# ---------------------------------------------------------------------------
# Jacobi eigen and singular value decompositions
# ---------------------------------------------------------------------------


@always_inline
def _rr_pair(round: Int, t: Int, mm: Int) -> Tuple[Int, Int]:
    """Pair `t` of round `round` of the circle-method round robin over `mm`
    (even) players: every pair meets exactly once in mm - 1 rounds, and the
    pairs of one round are disjoint. Returned as (low, high)."""
    var p: Int
    var q: Int
    if t == 0:
        p = round
        q = mm - 1
    else:
        p = (round + t) % (mm - 1)
        q = (round - t + mm - 1) % (mm - 1)
    if p > q:
        return (q, p)
    return (p, q)


@always_inline
def _before[dt: DType](x: Scalar[dt], i: Int, y: Scalar[dt], j: Int) -> Bool:
    """Ascending total order with NaN last, ties by index."""
    var xn = x != x
    var yn = y != y
    if xn != yn:
        return yn
    if xn:
        return i < j
    return x < y or (x == y and i < j)


@__name(t"linalg_syevj_{dt}_t{LA_THREADS}")
@__llvm_metadata(
    MAX_THREADS_PER_BLOCK_METADATA=StaticTuple[Int32, 1](Int32(LA_THREADS))
)
def _syevj_kernel[
    dt: DType
](
    src: FPtr[dt],
    a: FPtr[dt],
    vw: FPtr[dt],
    ws: FPtr[dt],
    w_out: FPtr[dt],
    v_out: FPtr[dt],
    n64: Int64,
    batch64: Int64,
    want_v64: Int64,
    s_rs64: Int64,
    s_cs64: Int64,
    s_bs64: Int64,
    lower64: Int64,
):
    """The symmetric matrix whose `lower` (else upper) triangle `src`
    holds is copied whole into `a` (n x n, column-major); `vw` (n x n) and
    `ws` (2n + 2 per matrix) are workspaces too. The eigenvalues land
    ascending in `w_out` (n per matrix), their vectors in the columns of
    `v_out` (n x n, column-major)."""
    var n = Int(n64)
    var batch = Int(batch64)
    var want_v = want_v64 != 0
    var s_rs = Int(s_rs64)
    var s_cs = Int(s_cs64)
    var s_bs = Int(s_bs64)
    var lower = lower64 != 0
    var tid = Int(thread_idx.x)
    var nt = Int(block_dim.x)
    var flag = stack_allocation[
        1, DType.int32, address_space=AddressSpace.SHARED
    ]()
    var mm = n + (n % 2)
    var pairs = mm // 2
    var tol = _eps[dt]()
    var b = Int(block_idx.x)
    while b < batch:
        var ab = b * n * n
        var wsb = b * (2 * n + 2)
        var idx = tid
        while idx < n * n:
            var i = idx % n
            var j = idx // n
            var own_half = i >= j if lower else i <= j
            var r = i if own_half else j
            var c = j if own_half else i
            a[unsafe_offset=ab + idx] = src[
                unsafe_offset=b * s_bs + r * s_rs + c * s_cs
            ]
            vw[unsafe_offset=ab + idx] = Scalar[dt](1) if idx % (
                n + 1
            ) == 0 else Scalar[dt](0)
            idx += nt
        barrier()
        for _sweep in range(JACOBI_MAX_SWEEPS):
            if n < 2:
                break
            if tid == 0:
                flag[unsafe_offset=0] = 0
            barrier()
            for rnd in range(mm - 1):
                var t = tid
                while t < pairs:
                    var pq = _rr_pair(rnd, t, mm)
                    var p = pq[0]
                    var q = pq[1]
                    var c = Scalar[dt](1)
                    var s = Scalar[dt](0)
                    if q < n:
                        var apq = a[unsafe_offset=ab + p + q * n]
                        var app = a[unsafe_offset=ab + p + p * n]
                        var aqq = a[unsafe_offset=ab + q + q * n]
                        if apq != 0 and _abs(apq) > tol * sqrt(
                            _abs(app)
                        ) * sqrt(_abs(aqq)):
                            var theta = (aqq - app) / (2 * apq)
                            var tt: Scalar[dt]
                            if _abs(theta) > 1 / tol:
                                tt = 1 / (2 * theta)
                            else:
                                tt = 1 / (_abs(theta) + sqrt(1 + theta * theta))
                                if theta < 0:
                                    tt = -tt
                            c = 1 / sqrt(1 + tt * tt)
                            s = tt * c
                            flag[unsafe_offset=0] = 1
                    ws[unsafe_offset=wsb + 2 * t] = c
                    ws[unsafe_offset=wsb + 2 * t + 1] = s
                    t += nt
                barrier()
                # A J and V J: columns p, q of every row.
                idx = tid
                while idx < n * pairs:
                    var r = idx % n
                    var pt = idx // n
                    var s = ws[unsafe_offset=wsb + 2 * pt + 1]
                    if s != 0:
                        var c = ws[unsafe_offset=wsb + 2 * pt]
                        var pq = _rr_pair(rnd, pt, mm)
                        var p = pq[0]
                        var q = pq[1]
                        var x = a[unsafe_offset=ab + r + p * n]
                        var y = a[unsafe_offset=ab + r + q * n]
                        a[unsafe_offset=ab + r + p * n] = c * x - s * y
                        a[unsafe_offset=ab + r + q * n] = s * x + c * y
                        if want_v:
                            x = vw[unsafe_offset=ab + r + p * n]
                            y = vw[unsafe_offset=ab + r + q * n]
                            vw[unsafe_offset=ab + r + p * n] = c * x - s * y
                            vw[unsafe_offset=ab + r + q * n] = s * x + c * y
                    idx += nt
                barrier()
                # J^T (A J): rows p, q of every column.
                idx = tid
                while idx < n * pairs:
                    var r = idx % n
                    var pt = idx // n
                    var s = ws[unsafe_offset=wsb + 2 * pt + 1]
                    if s != 0:
                        var c = ws[unsafe_offset=wsb + 2 * pt]
                        var pq = _rr_pair(rnd, pt, mm)
                        var p = pq[0]
                        var q = pq[1]
                        var x = a[unsafe_offset=ab + p + r * n]
                        var y = a[unsafe_offset=ab + q + r * n]
                        a[unsafe_offset=ab + p + r * n] = c * x - s * y
                        a[unsafe_offset=ab + q + r * n] = s * x + c * y
                    idx += nt
                barrier()
            var rotated = flag[unsafe_offset=0]
            barrier()
            if rotated == 0:
                break
        # Rank each eigenvalue, then scatter values and vectors into place.
        idx = tid
        while idx < n:
            var x = a[unsafe_offset=ab + idx + idx * n]
            var rank = 0
            for j in range(n):
                if _before(a[unsafe_offset=ab + j + j * n], j, x, idx):
                    rank += 1
            w_out[unsafe_offset=b * n + rank] = x
            ws[unsafe_offset=wsb + idx] = Scalar[dt](rank)
            idx += nt
        barrier()
        if want_v:
            idx = tid
            while idx < n * n:
                var r = idx % n
                var j = idx // n
                var rank = Int(ws[unsafe_offset=wsb + j])
                v_out[unsafe_offset=ab + r + rank * n] = vw[
                    unsafe_offset=ab + idx
                ]
                idx += nt
        barrier()
        b += Int(grid_dim.x)


@__name(t"linalg_gesvj_{dt}_t{LA_THREADS}")
@__llvm_metadata(
    MAX_THREADS_PER_BLOCK_METADATA=StaticTuple[Int32, 1](Int32(LA_THREADS))
)
def _gesvdj_kernel[
    dt: DType
](
    src: FPtr[dt],
    u: FPtr[dt],
    vw: FPtr[dt],
    ws: FPtr[dt],
    s_out: FPtr[dt],
    u_out: FPtr[dt],
    v_out: FPtr[dt],
    m64: Int64,
    n64: Int64,
    batch64: Int64,
    want_uv64: Int64,
    s_rs64: Int64,
    s_cs64: Int64,
    s_bs64: Int64,
):
    """One-sided Jacobi on `u` (m x n, m >= n, column-major), which starts
    as a copy of `src` (read through its strides):
    rotates column pairs until every pair is orthogonal, accumulating the
    rotations in `vw` (n x n). The column norms are the singular values,
    descending in `s_out` (n); `u_out` (m x n) gets the normalized columns
    (zero for a zero singular value) and `v_out` (n x n) the matching
    columns of V. `ws` holds 3n + 2 per matrix."""
    var m = Int(m64)
    var n = Int(n64)
    var batch = Int(batch64)
    var want_uv = want_uv64 != 0
    var s_rs = Int(s_rs64)
    var s_cs = Int(s_cs64)
    var s_bs = Int(s_bs64)
    var tid = Int(thread_idx.x)
    var nt = Int(block_dim.x)
    var flag = stack_allocation[
        1, DType.int32, address_space=AddressSpace.SHARED
    ]()
    var mm = n + (n % 2)
    var pairs = mm // 2
    var tol = _eps[dt]() * sqrt(Scalar[dt](max(m, 1)))
    var b = Int(block_idx.x)
    while b < batch:
        var ub = b * m * n
        var vb = b * n * n
        var wsb = b * (3 * n + 2)
        var idx = tid
        while idx < m * n:
            u[unsafe_offset=ub + idx] = src[
                unsafe_offset=b * s_bs + (idx % m) * s_rs + (idx // m) * s_cs
            ]
            idx += nt
        idx = tid
        while idx < n * n:
            vw[unsafe_offset=vb + idx] = Scalar[dt](1) if idx % (
                n + 1
            ) == 0 else Scalar[dt](0)
            idx += nt
        barrier()
        for _sweep in range(JACOBI_MAX_SWEEPS):
            if n < 2:
                break
            if tid == 0:
                flag[unsafe_offset=0] = 0
            barrier()
            for rnd in range(mm - 1):
                var t = tid
                while t < pairs:
                    var pq = _rr_pair(rnd, t, mm)
                    var p = pq[0]
                    var q = pq[1]
                    var c = Scalar[dt](1)
                    var s = Scalar[dt](0)
                    if q < n:
                        var alpha = Scalar[dt](0)
                        var beta = Scalar[dt](0)
                        var gamma = Scalar[dt](0)
                        for r in range(m):
                            var x = u[unsafe_offset=ub + r + p * m]
                            var y = u[unsafe_offset=ub + r + q * m]
                            alpha += x * x
                            beta += y * y
                            gamma += x * y
                        if gamma != 0 and _abs(gamma) > tol * sqrt(
                            alpha
                        ) * sqrt(beta):
                            var zeta = (beta - alpha) / (2 * gamma)
                            var tt: Scalar[dt]
                            if _abs(zeta) > 1 / _eps[dt]():
                                tt = 1 / (2 * zeta)
                            else:
                                tt = 1 / (_abs(zeta) + sqrt(1 + zeta * zeta))
                                if zeta < 0:
                                    tt = -tt
                            c = 1 / sqrt(1 + tt * tt)
                            s = tt * c
                            flag[unsafe_offset=0] = 1
                    ws[unsafe_offset=wsb + 2 * t] = c
                    ws[unsafe_offset=wsb + 2 * t + 1] = s
                    t += nt
                barrier()
                idx = tid
                var rows = max(m, n)
                while idx < rows * pairs:
                    var r = idx % rows
                    var pt = idx // rows
                    var s = ws[unsafe_offset=wsb + 2 * pt + 1]
                    if s != 0:
                        var c = ws[unsafe_offset=wsb + 2 * pt]
                        var pq = _rr_pair(rnd, pt, mm)
                        var p = pq[0]
                        var q = pq[1]
                        if r < m:
                            var x = u[unsafe_offset=ub + r + p * m]
                            var y = u[unsafe_offset=ub + r + q * m]
                            u[unsafe_offset=ub + r + p * m] = c * x - s * y
                            u[unsafe_offset=ub + r + q * m] = s * x + c * y
                        if want_uv and r < n:
                            var x = vw[unsafe_offset=vb + r + p * n]
                            var y = vw[unsafe_offset=vb + r + q * n]
                            vw[unsafe_offset=vb + r + p * n] = c * x - s * y
                            vw[unsafe_offset=vb + r + q * n] = s * x + c * y
                    idx += nt
                barrier()
            var rotated = flag[unsafe_offset=0]
            barrier()
            if rotated == 0:
                break
        # Column norms, then descending ranks (NaN first, as the reversed
        # ascending order), then the scatter.
        idx = tid
        while idx < n:
            var acc = Scalar[dt](0)
            for r in range(m):
                var x = u[unsafe_offset=ub + r + idx * m]
                acc += x * x
            ws[unsafe_offset=wsb + n + 2 + idx] = sqrt(acc)
            idx += nt
        barrier()
        idx = tid
        while idx < n:
            var x = ws[unsafe_offset=wsb + n + 2 + idx]
            var rank = 0
            for j in range(n):
                if _before(x, idx, ws[unsafe_offset=wsb + n + 2 + j], j):
                    rank += 1
            # `rank` counts the values after x ascending: its slot descending.
            s_out[unsafe_offset=b * n + rank] = x
            ws[unsafe_offset=wsb + idx] = Scalar[dt](rank)
            idx += nt
        barrier()
        if want_uv:
            idx = tid
            while idx < m * n:
                var r = idx % m
                var j = idx // m
                var rank = Int(ws[unsafe_offset=wsb + j])
                var sig = ws[unsafe_offset=wsb + n + 2 + j]
                var x = u[unsafe_offset=ub + idx]
                u_out[
                    unsafe_offset=ub + r + rank * m
                ] = x / sig if sig != 0 else Scalar[dt](0)
                idx += nt
            idx = tid
            while idx < n * n:
                var r = idx % n
                var j = idx // n
                var rank = Int(ws[unsafe_offset=wsb + j])
                v_out[unsafe_offset=vb + r + rank * n] = vw[
                    unsafe_offset=vb + idx
                ]
                idx += nt
        barrier()
        b += Int(grid_dim.x)


# ---------------------------------------------------------------------------
# Bunch-Kaufman LDL^T (LAPACK dsytf2, lower)
# ---------------------------------------------------------------------------


@__name(t"linalg_sytf2_lower_{dt}_t{LA_THREADS}")
@__llvm_metadata(
    MAX_THREADS_PER_BLOCK_METADATA=StaticTuple[Int32, 1](Int32(LA_THREADS))
)
def _sytf2_kernel[
    dt: DType
](a: FPtr[dt], piv: IPtr, info: IPtr, n64: Int64, batch64: Int64,):
    """dsytf2 with uplo = 'L' on a column-major n x n `a`, in place. Pivot
    choices are made by every thread from the same values (each reads them
    after a barrier), so the control flow stays uniform; the updates are
    split across the block."""
    var n = Int(n64)
    var batch = Int(batch64)
    var tid = Int(thread_idx.x)
    var nt = Int(block_dim.x)
    # (1 + sqrt(17)) / 8, the Bunch-Kaufman growth bound.
    var alpha_bk = (1 + sqrt(Scalar[dt](17))) / 8
    var b = Int(block_idx.x)
    while b < batch:
        var ab = b * n * n
        var status = 0
        var k = 0
        while k < n:
            var kstep = 1
            var absakk = _abs(a[unsafe_offset=ab + k + k * n])
            # imax: the row of the largest off-diagonal magnitude in column k
            var imax = k
            var colmax = Scalar[dt](0)
            for i in range(k + 1, n):
                var v = _abs(a[unsafe_offset=ab + i + k * n])
                if i == k + 1 or v > colmax:  # idamax: the first largest
                    colmax = v
                    imax = i
            var kp = k
            if max(absakk, colmax) == 0 or absakk != absakk:
                if status == 0:
                    status = k + 1
                kp = k
            else:
                if absakk >= alpha_bk * colmax:
                    kp = k
                else:
                    # rowmax: the largest off-diagonal magnitude in row imax
                    var rowmax = Scalar[dt](0)
                    for j in range(k, imax):
                        var v = _abs(a[unsafe_offset=ab + imax + j * n])
                        if v > rowmax:
                            rowmax = v
                    for j in range(imax + 1, n):
                        var v = _abs(a[unsafe_offset=ab + j + imax * n])
                        if v > rowmax:
                            rowmax = v
                    if absakk >= alpha_bk * colmax * (colmax / rowmax):
                        kp = k
                    elif (
                        _abs(a[unsafe_offset=ab + imax + imax * n])
                        >= alpha_bk * rowmax
                    ):
                        kp = imax
                    else:
                        kp = imax
                        kstep = 2
                var kk = k + kstep - 1
                barrier()  # every thread has made the same choice
                if kp != kk:
                    # Interchange rows and columns kk and kp in the trailing
                    # submatrix A(k:n, k:n).
                    var i = kp + 1 + tid
                    while i < n:
                        var x = a[unsafe_offset=ab + i + kk * n]
                        a[unsafe_offset=ab + i + kk * n] = a[
                            unsafe_offset=ab + i + kp * n
                        ]
                        a[unsafe_offset=ab + i + kp * n] = x
                        i += nt
                    var j = kk + 1 + tid
                    while j < kp:
                        var x = a[unsafe_offset=ab + j + kk * n]
                        a[unsafe_offset=ab + j + kk * n] = a[
                            unsafe_offset=ab + kp + j * n
                        ]
                        a[unsafe_offset=ab + kp + j * n] = x
                        j += nt
                    if tid == 0:
                        var x = a[unsafe_offset=ab + kk + kk * n]
                        a[unsafe_offset=ab + kk + kk * n] = a[
                            unsafe_offset=ab + kp + kp * n
                        ]
                        a[unsafe_offset=ab + kp + kp * n] = x
                        if kstep == 2:
                            x = a[unsafe_offset=ab + k + 1 + k * n]
                            a[unsafe_offset=ab + k + 1 + k * n] = a[
                                unsafe_offset=ab + kp + k * n
                            ]
                            a[unsafe_offset=ab + kp + k * n] = x
                    barrier()
                if kstep == 1:
                    # D(k) = A(k,k); A(k+1:n, k+1:n) -= (1/D) x x^T; x /= D
                    var d11 = 1 / a[unsafe_offset=ab + k + k * n]
                    var rr = n - k - 1
                    var idx = tid
                    while idx < rr * rr:
                        var i = k + 1 + idx % rr
                        var j = k + 1 + idx // rr
                        if i >= j:
                            # dsyr: A(i,j) += x(i) * (-d11 * x(j))
                            a[unsafe_offset=ab + i + j * n] += a[
                                unsafe_offset=ab + i + k * n
                            ] * (-d11 * a[unsafe_offset=ab + j + k * n])
                        idx += nt
                    barrier()
                    var i = k + 1 + tid
                    while i < n:
                        a[unsafe_offset=ab + i + k * n] *= d11
                        i += nt
                    barrier()
                elif k < n - 2:
                    var d21 = a[unsafe_offset=ab + k + 1 + k * n]
                    var d11 = a[unsafe_offset=ab + k + 1 + (k + 1) * n] / d21
                    var d22 = a[unsafe_offset=ab + k + k * n] / d21
                    var tt = 1 / (d11 * d22 - 1)
                    d21 = tt / d21
                    var rr = n - k - 2
                    # wk, wkp1 per row j (computed before the update, from
                    # the unmodified columns k and k+1).
                    var idx = tid
                    while idx < rr * rr:
                        var i = k + 2 + idx % rr
                        var j = k + 2 + idx // rr
                        if i >= j:
                            var ajk = a[unsafe_offset=ab + j + k * n]
                            var ajk1 = a[unsafe_offset=ab + j + (k + 1) * n]
                            var wk = d21 * (d11 * ajk - ajk1)
                            var wkp1 = d21 * (d22 * ajk1 - ajk)
                            a[unsafe_offset=ab + i + j * n] = (
                                a[unsafe_offset=ab + i + j * n]
                                - a[unsafe_offset=ab + i + k * n] * wk
                                - a[unsafe_offset=ab + i + (k + 1) * n] * wkp1
                            )
                        idx += nt
                    barrier()
                    var j = k + 2 + tid
                    while j < n:
                        var ajk = a[unsafe_offset=ab + j + k * n]
                        var ajk1 = a[unsafe_offset=ab + j + (k + 1) * n]
                        a[unsafe_offset=ab + j + k * n] = d21 * (
                            d11 * ajk - ajk1
                        )
                        a[unsafe_offset=ab + j + (k + 1) * n] = d21 * (
                            d22 * ajk1 - ajk
                        )
                        j += nt
                    barrier()
            if tid == 0:
                if kstep == 1:
                    piv[unsafe_offset=b * n + k] = Int32(kp + 1)
                else:
                    piv[unsafe_offset=b * n + k] = Int32(-(kp + 1))
                    piv[unsafe_offset=b * n + k + 1] = Int32(-(kp + 1))
            barrier()
            k += kstep
        if tid == 0:
            info[unsafe_offset=b] = Int32(status)
        barrier()
        b += Int(grid_dim.x)


@__name(t"linalg_sytrs_lower_{dt}_t{LA_THREADS}")
@__llvm_metadata(
    MAX_THREADS_PER_BLOCK_METADATA=StaticTuple[Int32, 1](Int32(LA_THREADS))
)
def _sytrs_kernel[
    dt: DType
](
    a: FPtr[dt],
    piv: IPtr,
    bm: FPtr[dt],
    n64: Int64,
    ncols64: Int64,
    a_rs64: Int64,
    a_cs64: Int64,
    a_bs64: Int64,
    piv_bs64: Int64,
    b_rs64: Int64,
    b_cs64: Int64,
    b_bs64: Int64,
    batch64: Int64,
):
    """LAPACK dsytrs (uplo = 'L'): A X = B from the Sytf2 factorization,
    one thread per right-hand side column."""
    var n = Int(n64)
    var ncols = Int(ncols64)
    var a_rs = Int(a_rs64)
    var a_cs = Int(a_cs64)
    var a_bs = Int(a_bs64)
    var piv_bs = Int(piv_bs64)
    var b_rs = Int(b_rs64)
    var b_cs = Int(b_cs64)
    var b_bs = Int(b_bs64)
    var total = Int(batch64) * ncols
    var t = Int(block_idx.x) * Int(block_dim.x) + Int(thread_idx.x)
    while t < total:
        var bb = t // ncols
        var col = t % ncols
        var ab = bb * a_bs
        var pb = bb * piv_bs
        var cb = bb * b_bs + col * b_cs

        @always_inline
        @__parameter
        def A(i: Int, j: Int) -> Scalar[dt]:
            return a[unsafe_offset=ab + i * a_rs + j * a_cs]

        @always_inline
        @__parameter
        def swap(i: Int, j: Int):
            if i != j and j >= 0 and j < n:
                var x = bm[unsafe_offset=cb + i * b_rs]
                bm[unsafe_offset=cb + i * b_rs] = bm[
                    unsafe_offset=cb + j * b_rs
                ]
                bm[unsafe_offset=cb + j * b_rs] = x

        # L D Y = B
        var k = 0
        while k < n:
            var p = Int(piv[unsafe_offset=pb + k])
            if p > 0:
                swap(k, p - 1)
                var bk = bm[unsafe_offset=cb + k * b_rs]
                for i in range(k + 1, n):
                    bm[unsafe_offset=cb + i * b_rs] -= A(i, k) * bk
                bm[unsafe_offset=cb + k * b_rs] = bk * (1 / A(k, k))
                k += 1
            else:
                swap(k + 1, -p - 1)
                var bk = bm[unsafe_offset=cb + k * b_rs]
                var bk1 = bm[unsafe_offset=cb + (k + 1) * b_rs]
                for i in range(k + 2, n):
                    bm[unsafe_offset=cb + i * b_rs] -= A(i, k) * bk
                for i in range(k + 2, n):
                    bm[unsafe_offset=cb + i * b_rs] -= A(i, k + 1) * bk1
                var akm1k = A(k + 1, k)
                var akm1 = A(k, k) / akm1k
                var ak = A(k + 1, k + 1) / akm1k
                var denom = akm1 * ak - 1
                var bkm1 = bk / akm1k
                var bkk = bk1 / akm1k
                bm[unsafe_offset=cb + k * b_rs] = (ak * bkm1 - bkk) / denom
                bm[unsafe_offset=cb + (k + 1) * b_rs] = (
                    akm1 * bkk - bkm1
                ) / denom
                k += 2
        # L^T X = Y
        k = n - 1
        while k >= 0:
            var p = Int(piv[unsafe_offset=pb + k])
            # dgemv('T'): B(k) -= sum_i B(i) A(i, k), the sum taken first.
            var acc = Scalar[dt](0)
            for i in range(k + 1, n):
                acc += bm[unsafe_offset=cb + i * b_rs] * A(i, k)
            if p > 0:
                bm[unsafe_offset=cb + k * b_rs] -= acc
                swap(k, p - 1)
                k -= 1
            else:
                var acc1 = Scalar[dt](0)
                for i in range(k + 1, n):
                    acc1 += bm[unsafe_offset=cb + i * b_rs] * A(i, k - 1)
                bm[unsafe_offset=cb + k * b_rs] -= acc
                bm[unsafe_offset=cb + (k - 1) * b_rs] -= acc1
                swap(k, -p - 1)
                k -= 2
        t += Int(grid_dim.x) * Int(block_dim.x)


@__name(t"linalg_pivot_sign_{dt}_t{LA_THREADS}")
@__llvm_metadata(
    MAX_THREADS_PER_BLOCK_METADATA=StaticTuple[Int32, 1](Int32(LA_THREADS))
)
def _pivsign_kernel[
    dt: DType
](piv: IPtr, dst: FPtr[dt], k64: Int64, batch64: Int64):
    """det(P) of a getrf pivot sequence: -1 per row that was swapped."""
    var k = Int(k64)
    var batch = Int(batch64)
    var t = Int(block_idx.x) * Int(block_dim.x) + Int(thread_idx.x)
    while t < batch:
        var odd = False
        for i in range(k):
            if Int(piv[unsafe_offset=t * k + i]) != i + 1:
                odd = not odd
        dst[unsafe_offset=t] = Scalar[dt](-1) if odd else Scalar[dt](1)
        t += Int(grid_dim.x) * Int(block_dim.x)


@__name(t"linalg_lu_slogdet_{dt}_t{LA_THREADS}")
@__llvm_metadata(
    MAX_THREADS_PER_BLOCK_METADATA=StaticTuple[Int32, 1](Int32(LA_THREADS))
)
def _slogdet_kernel[
    dt: DType
](
    lu: FPtr[dt],
    piv: IPtr,
    sign: FPtr[dt],
    logabs: FPtr[dt],
    n64: Int64,
    rs64: Int64,
    cs64: Int64,
    bs64: Int64,
    batch64: Int64,
):
    """sign and log|det| of a getrf factorization: det(P) times the signs
    of U's diagonal (torch's `sgn`: 0 for 0 and NaN), and the sum of the
    logs of its magnitudes."""
    var n = Int(n64)
    var rs = Int(rs64)
    var cs = Int(cs64)
    var bs = Int(bs64)
    var batch = Int(batch64)
    var t = Int(block_idx.x) * Int(block_dim.x) + Int(thread_idx.x)
    while t < batch:
        var sg = Scalar[dt](1)
        var acc = Scalar[dt](0)
        for i in range(n):
            if Int(piv[unsafe_offset=t * n + i]) != i + 1:
                sg = -sg
            var d = lu[unsafe_offset=t * bs + i * rs + i * cs]
            var ds = Scalar[dt](1) if d > 0 else (
                Scalar[dt](-1) if d < 0 else Scalar[dt](0)
            )
            sg = sg * ds
            comptime if dt == DType.float64:
                acc += nv_log(_abs(d).cast[DType.float64]()).cast[dt]()
            else:
                acc += nv_logf(_abs(d).cast[DType.float32]()).cast[dt]()
        sign[unsafe_offset=t] = sg
        logabs[unsafe_offset=t] = acc
        t += Int(grid_dim.x) * Int(block_dim.x)


# ---------------------------------------------------------------------------
# Host side: one launcher per op, all behind one slot layout
#   (p0, p1, p2, p3, p4, ints, ctx), ints = [dtype code, op params...]
# ---------------------------------------------------------------------------


@always_inline
def _fp[dt: DType](addr: Int) -> FPtr[dt]:
    return _make_ptr[dt](addr).as_unsafe_any_origin()


@always_inline
def _ip(addr: Int) -> IPtr:
    return _make_ptr[DType.int32](addr).as_unsafe_any_origin()


@always_inline
def _ti(ints: Arg, i: Int) -> Int64:
    """Op parameter `i` (the tuple's slot 0 is the dtype code)."""
    return Int64(_raw_tuple_int(ints, i + 1))


def _launch[
    dt: DType
](
    p0: Int,
    p1: Int,
    p2: Int,
    p3: Int,
    p4: Int,
    ints: Arg,
    ctx: DeviceContext,
) raises:
    comptime if not has_accelerator():
        raise Error("no GPU accelerator available at compile time")
    else:
        comptime if _op_on["Potrf"]():
            # n, rs, cs, bs, batch
            _enqueue_cached[_potrf_kernel[dt]](
                ctx,
                _blocks(Int(_ti(ints, 4))),
                1,
                1,
                LA_THREADS,
                _fp[dt](p0),
                _ip(p1),
                _ti(ints, 0),
                _ti(ints, 1),
                _ti(ints, 2),
                _ti(ints, 3),
                _ti(ints, 4),
            )
        elif _op_on["Getrf"]():
            # m, n, rs, cs, bs, batch, pivot
            _enqueue_cached[_getrf_kernel[dt]](
                ctx,
                _blocks(Int(_ti(ints, 5))),
                1,
                1,
                LA_THREADS,
                _fp[dt](p0),
                _ip(p1),
                _ip(p2),
                _ti(ints, 0),
                _ti(ints, 1),
                _ti(ints, 2),
                _ti(ints, 3),
                _ti(ints, 4),
                _ti(ints, 5),
                _ti(ints, 6),
            )
        elif _op_on["Trsm"]():
            # n, k, a_rs, a_cs, a_bs, b_rs, b_cs, b_bs, batch, forward, unit
            var chunks = (Int(_ti(ints, 1)) + TRSM_COLS - 1) // TRSM_COLS
            _enqueue_cached[_trsm_kernel[dt]](
                ctx,
                _blocks(Int(_ti(ints, 8)) * chunks),
                1,
                1,
                LA_THREADS,
                _fp[dt](p0),
                _fp[dt](p1),
                _ti(ints, 0),
                _ti(ints, 1),
                _ti(ints, 2),
                _ti(ints, 3),
                _ti(ints, 4),
                _ti(ints, 5),
                _ti(ints, 6),
                _ti(ints, 7),
                _ti(ints, 8),
                _ti(ints, 9),
                _ti(ints, 10),
            )
        elif _op_on["Laswp"]():
            # k, ncols, b_rs, b_cs, b_bs, piv_bs, batch, forward, nrows
            var total = Int(_ti(ints, 1)) * Int(_ti(ints, 6))
            _enqueue_cached[_laswp_kernel[dt]](
                ctx,
                _blocks((total + LA_THREADS - 1) // LA_THREADS),
                1,
                1,
                LA_THREADS,
                _fp[dt](p0),
                _ip(p1),
                _ti(ints, 0),
                _ti(ints, 1),
                _ti(ints, 2),
                _ti(ints, 3),
                _ti(ints, 4),
                _ti(ints, 5),
                _ti(ints, 6),
                _ti(ints, 7),
                _ti(ints, 8),
            )
        elif _op_on["Geqrf"]():
            # m, n, rs, cs, bs, batch
            _enqueue_cached[_geqrf_kernel[dt]](
                ctx,
                _blocks(Int(_ti(ints, 5))),
                1,
                1,
                LA_THREADS,
                _fp[dt](p0),
                _fp[dt](p1),
                _ti(ints, 0),
                _ti(ints, 1),
                _ti(ints, 2),
                _ti(ints, 3),
                _ti(ints, 4),
                _ti(ints, 5),
            )
        elif _op_on["Ormqr"]():
            # nq, ncols, k, a_rs, a_cs, a_bs, tau_bs, c_rs, c_cs, c_bs,
            # batch, trans
            var total = Int(_ti(ints, 1)) * Int(_ti(ints, 10))
            _enqueue_cached[_ormqr_kernel[dt]](
                ctx,
                _blocks((total + LA_THREADS - 1) // LA_THREADS),
                1,
                1,
                LA_THREADS,
                _fp[dt](p0),
                _fp[dt](p1),
                _fp[dt](p2),
                _ti(ints, 0),
                _ti(ints, 1),
                _ti(ints, 2),
                _ti(ints, 3),
                _ti(ints, 4),
                _ti(ints, 5),
                _ti(ints, 6),
                _ti(ints, 7),
                _ti(ints, 8),
                _ti(ints, 9),
                _ti(ints, 10),
                _ti(ints, 11),
            )
        elif _op_on["Syevj"]():
            # p0 src, p1 a, p2 vw, p3 ws, p4 w; ints: v_out, n, batch,
            # want_v, s_rs, s_cs, s_bs, lower
            _enqueue_cached[_syevj_kernel[dt]](
                ctx,
                _blocks(Int(_ti(ints, 2))),
                1,
                1,
                LA_THREADS,
                _fp[dt](p0),
                _fp[dt](p1),
                _fp[dt](p2),
                _fp[dt](p3),
                _fp[dt](p4),
                _fp[dt](Int(_ti(ints, 0))),
                _ti(ints, 1),
                _ti(ints, 2),
                _ti(ints, 3),
                _ti(ints, 4),
                _ti(ints, 5),
                _ti(ints, 6),
                _ti(ints, 7),
            )
        elif _op_on["Gesvdj"]():
            # p0 src, p1 u, p2 vw, p3 ws, p4 s; ints: u_out, v_out, m, n,
            # batch, want_uv, s_rs, s_cs, s_bs
            _enqueue_cached[_gesvdj_kernel[dt]](
                ctx,
                _blocks(Int(_ti(ints, 4))),
                1,
                1,
                LA_THREADS,
                _fp[dt](p0),
                _fp[dt](p1),
                _fp[dt](p2),
                _fp[dt](p3),
                _fp[dt](p4),
                _fp[dt](Int(_ti(ints, 0))),
                _fp[dt](Int(_ti(ints, 1))),
                _ti(ints, 2),
                _ti(ints, 3),
                _ti(ints, 4),
                _ti(ints, 5),
                _ti(ints, 6),
                _ti(ints, 7),
                _ti(ints, 8),
            )
        elif _op_on["Sytrs"]():
            # n, ncols, a_rs, a_cs, a_bs, piv_bs, b_rs, b_cs, b_bs, batch
            var total = Int(_ti(ints, 1)) * Int(_ti(ints, 9))
            _enqueue_cached[_sytrs_kernel[dt]](
                ctx,
                _blocks((total + LA_THREADS - 1) // LA_THREADS),
                1,
                1,
                LA_THREADS,
                _fp[dt](p0),
                _ip(p1),
                _fp[dt](p2),
                _ti(ints, 0),
                _ti(ints, 1),
                _ti(ints, 2),
                _ti(ints, 3),
                _ti(ints, 4),
                _ti(ints, 5),
                _ti(ints, 6),
                _ti(ints, 7),
                _ti(ints, 8),
                _ti(ints, 9),
            )
        elif _op_on["Slogdet"]():
            # n, rs, cs, bs, batch; p0 lu, p1 piv, p2 sign, p3 logabs
            _enqueue_cached[_slogdet_kernel[dt]](
                ctx,
                _blocks((Int(_ti(ints, 4)) + LA_THREADS - 1) // LA_THREADS),
                1,
                1,
                LA_THREADS,
                _fp[dt](p0),
                _ip(p1),
                _fp[dt](p2),
                _fp[dt](p3),
                _ti(ints, 0),
                _ti(ints, 1),
                _ti(ints, 2),
                _ti(ints, 3),
                _ti(ints, 4),
            )
        elif _op_on["PivSign"]():
            # k, batch
            _enqueue_cached[_pivsign_kernel[dt]](
                ctx,
                _blocks((Int(_ti(ints, 1)) + LA_THREADS - 1) // LA_THREADS),
                1,
                1,
                LA_THREADS,
                _ip(p0),
                _fp[dt](p1),
                _ti(ints, 0),
                _ti(ints, 1),
            )
        elif _op_on["Sytf2"]():
            # n, batch
            _enqueue_cached[_sytf2_kernel[dt]](
                ctx,
                _blocks(Int(_ti(ints, 1))),
                1,
                1,
                LA_THREADS,
                _fp[dt](p0),
                _ip(p1),
                _ip(p2),
                _ti(ints, 0),
                _ti(ints, 1),
            )
        else:
            raise Error(NO_OP_COMPILED)


def _linalg_go(
    p0: Arg,
    p1: Arg,
    p2: Arg,
    p3: Arg,
    p4: Arg,
    ints: Arg,
    ctx_obj: Arg,
) raises:
    var dtype = _raw_dtype_int(_raw_tuple_int(ints, 0))
    var handled = False
    comptime for dt in LA_DTYPES:
        comptime if _dtype_arg_on[0, dt]():
            if dtype == dt:
                _launch[dt](
                    _raw_int(p0),
                    _raw_int(p1),
                    _raw_int(p2),
                    _raw_int(p3),
                    _raw_int(p4),
                    ints,
                    _raw_ctx(ctx_obj),
                )
                handled = True
    if not handled:
        raise Error("linalg: unsupported dtype specialization " + String(dtype))


@export
def tmb_call(argv: Argv, argc: Int, err: ErrBuf, errcap: Int) abi("C") -> Int32:
    """C entry of this family: one kernel per build (see `OP`).
    Slots are described in op_utils (`Arg`); errors come back as (rc=1, message).
    """
    try:
        comptime if (
            _op_on["Potrf"]()
            or _op_on["Getrf"]()
            or _op_on["Trsm"]()
            or _op_on["Laswp"]()
            or _op_on["Geqrf"]()
            or _op_on["Ormqr"]()
            or _op_on["Syevj"]()
            or _op_on["Gesvdj"]()
            or _op_on["Sytf2"]()
            or _op_on["Sytrs"]()
            or _op_on["PivSign"]()
            or _op_on["Slogdet"]()
        ):
            _spec_dispatcher7[_linalg_go, "linalg"](argv, argc)
            return 0
        raise Error(NO_OP_COMPILED)
    except e:
        return _tmb_entry_error(err, errcap, e)
