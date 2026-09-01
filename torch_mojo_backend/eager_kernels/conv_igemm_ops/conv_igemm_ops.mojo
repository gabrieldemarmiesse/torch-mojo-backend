"""Implicit-GEMM conv2d forward on sm_90a — no materialized im2col.

The patch matrix is never written to memory: the activation is read tile by
tile straight out of an NHWC buffer by the hardware im2col TMA
(`cp.async.bulk.tensor.4d...im2col`, `cta_group=1`).  On the body shape of a
ResNet (N32 C64 56x56 K64 3x3 s1, bf16) the materialized route spends 89% of
its device time writing and re-reading that matrix; this route spends none.

Operand assignment (the load-bearing design choice):

    D[oc, p] = sum_k  Wt[oc, k] * Act[p, k]          k = (r, s, c)
    A = repacked weight   (out_c x K_pad), k-major
    B = im2col activation (pixels x K_pad), k-major   <- TMA im2col box
    D = (out_c x pixels)                              <- ALREADY NCHW

The im2col box delivers (pixels_per_column x channels_per_pixel) with the
channels contiguous, i.e. exactly a k-major B operand, so wgmma consumes it
with `transpose_b=True` and no shuffle.  Putting `out_c` on the GEMM's M axis
makes the accumulator tile (out_c x pixels) = NCHW, so unlike cuDNN — whose
fprop kernels are NHWC-only and pay an `nhwcToNchwKernel` on the way out —
there is no output transpose at all.  One NCHW->NHWC pass on the input is
still needed: TMA requires the descriptor's innermost dimension to be
contiguous, and for im2col that dimension is C.

Body/mainloop/epilogue are the 16-bit GEMM family's "tiny" batched BMM body
(`gemm16_bmm_v5_kernels._b5_bmm_nn_tiny`): one 128-thread warp group per work
item, an mbarrier TMA pipeline, wgmma m64 x bn x k16, and — shared verbatim,
not copied — the C epilogue in `wgmma_c_epilogue.mojo`.  Only the B producer
is new.

Two pieces of hard-won knowledge about this instruction on sm_90a, both
established by experiment on an H100 PCIe (the modular repo only exercises
im2col TMA under B200 gating, so none of it was documented):

  * `layout.tma_async.TMATensorTileIm2col.async_copy` computes WRONG
    coordinates here — tap (0,0) is exact and every other tap returns pixels
    displaced in the pixel stream.  The descriptor itself is fine, so this
    file keeps `TMATensorTileIm2col` only as the descriptor carrier and
    issues `cp_async_bulk_tensor_shared_cluster_global_im2col` directly with
    locally computed coordinates.
  * `_im2col_desc_shape`'s 256-element box cap (`max_tma_box_elements`) is
    not a hardware limit.  One transaction per B tile (`pixels_per_column ==
    bn`, 16 KB) is 7.3x faster end to end than the 4-pixel box that cap
    implies, and is what this kernel issues.

Dtype: the source is width-generic and instantiates for bfloat16 or float16
(same operand width, same wgmma shapes); only bfloat16 is enabled at the
dispatch, because only bfloat16 has been measured.
"""

from std.gpu import (
    MAX_THREADS_PER_BLOCK_METADATA,
    block_dim,
    block_idx,
    thread_idx,
)
from max.gpu.host import DeviceBuffer, DeviceContext
from max.gpu.host._tensormap import SwizzleMode, create_tensormap_im2col
from max.gpu.host.nvidia.tma import (
    TMADescriptor,
    TensorMapSwizzle,
    create_tma_descriptor,
)
from max.gpu.memory import cp_async_bulk_tensor_shared_cluster_global_im2col
from max.gpu.sync import barrier
from std.memory import AddressSpace, stack_allocation
from std.os import abort
from std.python import PythonObject
from std.python._cpython import PyObjectPtr, Py_ssize_t
from std.python.bindings import PythonModuleBuilder
from std.sys import size_of
from std.sys.info import _is_sm_9x
from std.utils.index import Index, IndexList
from std.utils.static_tuple import StaticTuple

from layout import Layout, LayoutTensor
from layout.tensor_core_async import (
    TensorCoreAsync,
    tile_layout_k_major,
    warpgroup_fence,
)
from layout.tma_async import (
    SharedMemBarrier,
    TMATensorTile,
    TMATensorTileIm2col,
)

from op_utils import (
    _enqueue_cached,
    _make_ptr,
    _raw_ctx,
    _raw_dtype_int,
    _raw_int,
    _raw_ret_none,
    _raw_tuple_int,
    _spec_unsupported,
)
from variant_gates import _dtype_arg_on, _op_on, _register_call
from wgmma_c_epilogue import _wgmma_store_c_tile

comptime _CI_F32 = DType.float32
comptime _CI_SWZ = TensorMapSwizzle.SWIZZLE_128B
# 128 B swizzle / 2 B element: the im2col box's channels_per_pixel, and
# therefore the GEMM's k-tile.  K_pad is a multiple of it by construction, so
# one k-tile never straddles two (r, s) taps.
comptime _CI_BK = 64
comptime _CI_BM = 64
# The 16-bit dtypes this kernel is instantiated for.  float16 is byte-for-byte
# the same tensor-core path as bfloat16 (same operand width, same wgmma tile,
# same fp32 accumulator); it is listed here so the source stays honest about
# what it supports, while the Python dispatch admits bfloat16 only.
comptime _CI_DTYPES = [DType.bfloat16, DType.float16]


@always_inline
def _ci_tag[dtype: DType]() -> StaticString:
    """The dtype token the kernel names carry: what CUPTI and Nsight print."""
    comptime if dtype == DType.float16:
        return "f16"
    return "bf16"


@always_inline
def _ci_store_tag[tma_store: Bool]() -> StaticString:
    comptime if tma_store:
        return ""
    return "_scalar_c"


# ---------------------------------------------------------------------------
# NCHW -> NHWC with C padded up to a multiple of BK.  The pad channels are
# zeroed and the repacked weight's matching rows are zero too, so the padded
# reduction is exact.
#
# A thread owns a SUB x SUB (channel x pixel) sub-tile, loads it along the
# contiguous HW axis, transposes it IN REGISTERS (comptime index math, free)
# and moves it through shared memory in SUB-element vector accesses; the
# read-back puts one wavefront on one pixel's channels, so the global store is
# 128 B contiguous.
#
# SUB is 4, not 8, and that is counter-intuitive: the 8-wide (16 B) version
# measured 0.80 waves per SM on the body shape -- 64 elements per thread
# leaves too few warps resident to cover DRAM latency, and it ran at 13% of
# DRAM peak with nothing saturated.  Halving the tile edge quadrupled the warp
# count for the same bytes and took the body's transpose from ~38 us to
# ~18 us (1.4 TB/s, i.e. faster than the `nchwToNhwcKernel` cuDNN pays on the
# same shape).  Fitted on an H100 PCIe.
# ---------------------------------------------------------------------------
comptime _CI_T_SUB = 4
comptime _CI_T_PIX = 16 * _CI_T_SUB  # pixels per tile
comptime _CI_T_CH = 64  # channels per tile
comptime _CI_T_THREADS = (_CI_T_PIX // _CI_T_SUB) * (_CI_T_CH // _CI_T_SUB)
# Row stride of the transposed smem tile, in elements.  The write phase
# touches rows SUB apart (a thread owns SUB consecutive pixels) and any
# 16 B-aligned row stride is a multiple of 32 banks, so those rows collide;
# the channel offset therefore carries an XOR swizzle keyed on the row GROUP
# (p // SUB).  The read phase has every lane of a wavefront on ONE row, so the
# swizzle is constant there and costs it nothing.
comptime _CI_T_SPAD = _CI_T_CH + _CI_T_SUB


@always_inline
def _ci_swz(p: Int) -> Int:
    """XOR applied to the SUB-element-aligned channel offset of smem row p."""
    return ((p // _CI_T_SUB) % (_CI_T_CH // _CI_T_SUB)) * _CI_T_SUB


@__llvm_metadata(
    MAX_THREADS_PER_BLOCK_METADATA=StaticTuple[Int32, 1](Int32(_CI_T_THREADS))
)
@__name(t"conv_nchw_to_nhwc_padc_{_ci_tag[dtype]()}")
def _ci_nchw_to_nhwc_kernel[
    dtype: DType
](
    out_ptr: UnsafePointer[Scalar[dtype], MutAnyOrigin],
    in_ptr: UnsafePointer[Scalar[dtype], MutAnyOrigin],
    hw_arg: Int32,
    c_arg: Int32,
    c_pad_arg: Int32,
):
    var hw = Int(hw_arg)
    var c = Int(c_arg)
    var c_pad = Int(c_pad_arg)
    var p0 = Int(block_idx.x) * _CI_T_PIX
    var c0 = Int(block_idx.y) * _CI_T_CH
    var n = Int(block_idx.z)
    var tid = Int(thread_idx.x)

    var smem = LayoutTensor[
        dtype,
        Layout.row_major(_CI_T_PIX, _CI_T_SPAD),
        MutAnyOrigin,
        address_space=AddressSpace.SHARED,
        alignment=16,
    ].stack_allocation()

    comptime PG = _CI_T_PIX // _CI_T_SUB
    var cg = tid // PG
    var pg = tid % PG
    var lc = c0 + cg * _CI_T_SUB
    var lp = p0 + pg * _CI_T_SUB

    var frag = StaticTuple[SIMD[dtype, _CI_T_SUB], _CI_T_SUB]()
    comptime for j in range(_CI_T_SUB):
        var v = SIMD[dtype, _CI_T_SUB](0)
        # The global side stays at natural (2 B) alignment on purpose: the
        # NCHW plane pitch `hw` is arbitrary, so (n*c + lc + j)*hw + lp is not
        # a multiple of SUB in general and an alignment claim here would be a
        # lie -- a vector path over the input has to be gated on the RUNTIME
        # address, not on a dimension test.
        if lc + j < c and lp + _CI_T_SUB - 1 < hw:
            v = in_ptr.load[width=_CI_T_SUB]((n * c + lc + j) * hw + lp)
        elif lc + j < c:
            for i in range(_CI_T_SUB):
                if lp + i < hw:
                    v[i] = in_ptr[(n * c + lc + j) * hw + lp + i]
        frag[j] = v

    # register transpose: out row i (pixel) gathers channel j from frag[j][i]
    comptime for i in range(_CI_T_SUB):
        var row = SIMD[dtype, _CI_T_SUB](0)
        comptime for j in range(_CI_T_SUB):
            row[j] = frag[j][i]
        var pp = pg * _CI_T_SUB + i
        # 8 B shared stores.  The element offset is a multiple of SUB
        # (_CI_T_SPAD * 2 B = 136 B is a multiple of 8, and both the channel
        # offset and the XOR swizzle are multiples of SUB), so the alignment
        # is honest -- and without it Mojo defaults to align-4 and the store
        # reaches PTX as four scalar `st.shared.b16`.
        smem.ptr.store[alignment=_CI_T_SUB * 2](
            pp * _CI_T_SPAD + ((cg * _CI_T_SUB) ^ _ci_swz(pp)), row
        )
    barrier()

    # read-back: one wavefront covers one pixel's channels = one 128 B store
    comptime CHUNKS = _CI_T_CH // _CI_T_SUB
    comptime for m in range(_CI_T_PIX * CHUNKS // _CI_T_THREADS):
        var slot = m * _CI_T_THREADS + tid
        var sp = slot // CHUNKS
        var sc = (slot % CHUNKS) * _CI_T_SUB
        if p0 + sp < hw and c0 + sc < c_pad:
            # Both sides are 8 B here: c_pad is a multiple of BK = 64 elements
            # by construction (the caller pads C up to it), c0 is a multiple
            # of _CI_T_CH and sc of SUB, and the destination is a fresh
            # allocation.
            out_ptr.store[alignment=_CI_T_SUB * 2](
                (n * hw + p0 + sp) * c_pad + c0 + sc,
                smem.ptr.load[width=_CI_T_SUB, alignment=_CI_T_SUB * 2](
                    sp * _CI_T_SPAD + (sc ^ _ci_swz(sp))
                ),
            )


# ---------------------------------------------------------------------------
# Weight (out_c, C, R, S) -> A matrix (out_c, K_pad) with
# k = (r * S + s) * C_pad + c, matching the im2col K decomposition, and zeros
# wherever c >= C.  A is k-major, which is what wgmma wants for its A operand,
# so nothing further happens to it in the kernel.
# ---------------------------------------------------------------------------
@__llvm_metadata(
    MAX_THREADS_PER_BLOCK_METADATA=StaticTuple[Int32, 1](Int32(256))
)
@__name(t"conv_weight_repack_rsc_{_ci_tag[dtype]()}")
def _ci_weight_repack_kernel[
    dtype: DType
](
    out_ptr: UnsafePointer[Scalar[dtype], MutAnyOrigin],
    w_ptr: UnsafePointer[Scalar[dtype], MutAnyOrigin],
    out_c_arg: Int32,
    c_arg: Int32,
    c_pad_arg: Int32,
    r_arg: Int32,
    s_arg: Int32,
):
    var out_c = Int(out_c_arg)
    var c = Int(c_arg)
    var c_pad = Int(c_pad_arg)
    var r_f = Int(r_arg)
    var s_f = Int(s_arg)
    var k_pad = r_f * s_f * c_pad
    var idx = Int(block_idx.x) * Int(block_dim.x) + Int(thread_idx.x)
    if idx >= out_c * k_pad:
        return
    var oc = idx // k_pad
    var k = idx % k_pad
    var ch = k % c_pad
    var fi = k // c_pad
    var val = Scalar[dtype](0)
    if ch < c:
        var r = fi // s_f
        var s = fi % s_f
        val = w_ptr[((oc * c + ch) * r_f + r) * s_f + s]
    out_ptr[idx] = val


# ---------------------------------------------------------------------------
# The im2col B producer.
#
# One transaction delivers the whole `bn`-pixel B tile x BK channels
# (pixels_per_column == bn).  The hardware advances the base pixel inside the
# descriptor's pixel BOX, and with the CUTLASS fprop corner convention
# (lower = -pad, upper = pad - (filter - 1)) that box IS the (n, oh, ow)
# output grid, so a transaction that crosses a row -- or a sample -- boundary
# lands on the right pixels with no software help.
#
# Two DIFFERENT out-of-range behaviours matter here and must not be conflated:
#
#   * the BASE coordinate handed to the instruction must lie inside the pixel
#     box; an out-of-box base raises CUDA_ERROR_ILLEGAL_INSTRUCTION.  With one
#     transaction per B tile the base is (sample, n0) with n0 < OH*OW, so it
#     is in-box by construction: no clamp, and no guard sample.
#   * the pixels the transaction walks onto INTERNALLY (the pixels-per-column
#     tail) are generated by the hardware and follow ordinary im2col
#     out-of-bounds semantics -- past the end of the tensor they are
#     zero-filled, not faulted.  Every tail tile of the last sample exercises
#     this, down to a batch of 1 whose single transaction runs 79 pixels past
#     the only sample there is.
# ---------------------------------------------------------------------------
@always_inline
def _ci_im2col_load[
    dtype: DType, bn: Int
](
    tma: TMATensorTileIm2col[dtype, 2, Index(bn, _CI_BK), Index(bn, _CI_BK)],
    dst: UnsafePointer[
        Scalar[dtype], MutAnyOrigin, address_space=AddressSpace.SHARED
    ],
    ref[AddressSpace.SHARED] bar: SharedMemBarrier,
    ch: Int,  # channel base within C_pad
    tap_s: Int,
    tap_r: Int,
    pix_n: Int,
    pix_h: Int,
    pix_w: Int,
):
    cp_async_bulk_tensor_shared_cluster_global_im2col[cta_group=1](
        dst,
        UnsafePointer(to=tma.descriptor).bitcast[NoneType](),
        bar.unsafe_ptr(),
        Index(ch, pix_w, pix_h, pix_n),
        Index(tap_s, tap_r),
    )


@__llvm_arg_metadata(a_tma, `nvvm.grid_constant`)
@__llvm_arg_metadata(b_tma, `nvvm.grid_constant`)
@__llvm_arg_metadata(c_tma, `nvvm.grid_constant`)
@__llvm_metadata(
    MAX_THREADS_PER_BLOCK_METADATA=StaticTuple[Int32, 1](Int32(128))
)
@__name(
    t"conv_fprop_igemm_tma_im2col_{_ci_tag[dtype]()}_m64n{bn}{_ci_store_tag[tma_store]()}"
)
def _ci_conv_igemm_kernel[
    dtype: DType,
    stages: Int,
    bn: Int,
    tma_store: Bool,
](
    a_tma: TMATensorTile[
        dtype, 2, Index(_CI_BM, _CI_BK), Index(_CI_BM, _CI_BK)
    ],
    b_tma: TMATensorTileIm2col[dtype, 2, Index(bn, _CI_BK), Index(bn, _CI_BK)],
    c_tma: TMATensorTile[dtype, 3, Index(1, _CI_BM, 64), Index(1, _CI_BM, 64)],
    output: UnsafePointer[Scalar[dtype], MutAnyOrigin],
    out_c_arg: Int32,
    hw_arg: Int32,
    k_pad_arg: Int32,
    c_pad_arg: Int32,
    ow_arg: Int32,
    pad_h_arg: Int32,
    pad_w_arg: Int32,
    s_f_arg: Int32,
):
    # Legality of the tile parameters, at BUILD time.  An illegal combination
    # would be a runtime HANG rather than a wrong answer: the mbarrier is
    # armed for the whole (BM + bn) x BK tile, so a `bn` the single-
    # transaction B producer cannot fill leaves the consumer waiting forever.
    comptime assert stages >= 2, "conv igemm needs at least 2 pipeline stages"
    comptime assert (
        bn >= 64 and bn % 64 == 0
    ), "conv igemm needs bn a positive multiple of 64 (wgmma N granularity)"
    comptime assert bn * _CI_BK * 2 <= 32768, (
        "the im2col box is ONE transaction of bn x BK elements; keep it under"
        " the 32 KB TMA transaction limit"
    )
    comptime assert _CI_BK * 2 == 128, (
        "conv igemm assumes a 128 B swizzle, i.e. channels_per_pixel *"
        " sizeof(dtype) == 128"
    )
    comptime assert (
        size_of[dtype]() == 2
    ), "conv igemm is a 16-bit tensor-core kernel"
    comptime if _is_sm_9x():
        var out_c = Int(out_c_arg)
        var hw = Int(hw_arg)
        var k_pad = Int(k_pad_arg)
        var c_pad = Int(c_pad_arg)
        var ow = Int(ow_arg)
        var s_f = Int(s_f_arg)

        comptime A_LAYOUT = tile_layout_k_major[
            dtype, _CI_BM, _CI_BK, _CI_SWZ
        ]()
        comptime B_LAYOUT = tile_layout_k_major[dtype, bn, _CI_BK, _CI_SWZ]()

        var a_pipe = LayoutTensor[
            dtype,
            Layout.row_major(stages, _CI_BM * _CI_BK),
            MutAnyOrigin,
            address_space=AddressSpace.SHARED,
            alignment=128,
        ].stack_allocation()
        var b_pipe = LayoutTensor[
            dtype,
            Layout.row_major(stages, bn * _CI_BK),
            MutAnyOrigin,
            address_space=AddressSpace.SHARED,
            alignment=128,
        ].stack_allocation()
        comptime C_SMEM_ELEMS = _CI_BM * bn if tma_store else 32
        var c_smem = LayoutTensor[
            dtype,
            Layout.row_major(1, C_SMEM_ELEMS),
            MutAnyOrigin,
            address_space=AddressSpace.SHARED,
            alignment=1024,
        ].stack_allocation()
        var full_barriers = stack_allocation[
            stages,
            SharedMemBarrier,
            address_space=AddressSpace.SHARED,
            alignment=8,
        ]()
        if thread_idx.x == 0:
            comptime for stage in range(stages):
                full_barriers[stage].init()
            a_tma.prefetch_descriptor()
            b_tma.prefetch_descriptor()
            comptime if tma_store:
                c_tma.prefetch_descriptor()
        barrier()

        comptime CFRAG = 64 * bn // 128
        comptime TMA_BYTES = (_CI_BM + bn) * _CI_BK * 2

        var blocks_n = (hw + bn - 1) // bn
        var macro_rows = (out_c + _CI_BM - 1) // _CI_BM
        var works_per_item = macro_rows * blocks_n
        var num_tiles = (k_pad + _CI_BK - 1) // _CI_BK
        var w = Int(block_idx.x)
        var sample = w // works_per_item
        var rem = w % works_per_item
        var m0 = (rem // blocks_n) * _CI_BM
        var n0 = (rem % blocks_n) * bn

        # Base pixel of this tile's single im2col transaction: constant across
        # the whole k loop, so it is decoded once here instead of per k-tile.
        # n0 < hw by construction, so (sample, n0) is always inside the pixel
        # box -- this is what makes a guard sample unnecessary.
        var pix_n = sample
        var pix_h = n0 // ow - Int(pad_h_arg)
        var pix_w = n0 % ow - Int(pad_w_arg)

        @parameter
        @always_inline
        def issue_tile(t: Int):
            var stage = t % stages
            var k0 = t * _CI_BK
            if thread_idx.x == 0:
                full_barriers[stage].expect_bytes(Int32(TMA_BYTES))
                var a_tile = LayoutTensor[
                    dtype,
                    A_LAYOUT,
                    MutAnyOrigin,
                    address_space=AddressSpace.SHARED,
                    alignment=128,
                ](a_pipe.ptr + stage * _CI_BM * _CI_BK)
                a_tma.async_copy(a_tile, full_barriers[stage], (k0, m0))
                # One transaction per B tile, issued from the same lane as the
                # A copy.  BK divides C_pad, so one k-tile never straddles two
                # (r, s) taps and the tap indices are loop-invariant here.
                var ch = k0 % c_pad
                var fi = k0 // c_pad
                _ci_im2col_load[dtype, bn](
                    b_tma,
                    (b_pipe.ptr + stage * bn * _CI_BK).bitcast[Scalar[dtype]](),
                    full_barriers[stage],
                    ch,
                    fi % s_f,
                    fi // s_f,
                    pix_n,
                    pix_h,
                    pix_w,
                )

        var pre = min(stages, num_tiles)
        for t in range(pre):
            issue_tile(t)

        var accum = LayoutTensor[
            _CI_F32,
            Layout.row_major(1, CFRAG),
            MutAnyOrigin,
            address_space=AddressSpace.LOCAL,
        ].stack_allocation()
        _ = accum.fill(0.0)
        comptime wgmma = TensorCoreAsync[
            _CI_F32,
            dtype,
            dtype,
            Index(64, bn, 16),
            a_swizzle=_CI_SWZ,
            b_swizzle=_CI_SWZ,
            transpose_b=True,
        ]()

        var t = 0
        while t < num_tiles:
            var stage = t % stages
            var phase = UInt32((t // stages) % 2)
            full_barriers[stage].wait(phase)
            var a_tile = LayoutTensor[
                dtype,
                A_LAYOUT,
                MutAnyOrigin,
                address_space=AddressSpace.SHARED,
                alignment=128,
            ](a_pipe.ptr + stage * _CI_BM * _CI_BK)
            var b_tile = LayoutTensor[
                dtype,
                B_LAYOUT,
                MutAnyOrigin,
                address_space=AddressSpace.SHARED,
                alignment=128,
            ](b_pipe.ptr + stage * bn * _CI_BK)
            warpgroup_fence(accum)
            wgmma.arrive()
            wgmma.wgmma[1](a_tile, b_tile, accum, 0)
            wgmma.commit_group()
            warpgroup_fence(accum)
            wgmma.wait_group()
            if t + stages < num_tiles:
                issue_tile(t + stages)
            t += 1

        # (out_c x pixels) per sample IS the (batch, m, n) tile the GEMM
        # family's epilogue writes, so it is shared, not copied.
        _wgmma_store_c_tile[dtype, _CI_BM, bn, tma_store](
            c_tma,
            c_smem.ptr,
            accum.ptr,
            output,
            out_c,
            hw,
            out_c * hw,
            m0,
            n0,
            sample,
        )


# ---------------------------------------------------------------------------
# Host side
# ---------------------------------------------------------------------------


def conv_igemm_descriptor_legal(
    pad_h: Int, pad_w: Int, r_f: Int, s_f: Int
) -> Bool:
    """Whether the im2col descriptor and the taps this kernel issues are
    inside the hardware's documented domain.

    `cuTensorMapEncodeIm2col` takes the pixel-box corners as SIGNED CHAR, so
    both corners of both spatial dimensions must be in [-128, 127], and the
    per-transaction im2col offsets (the filter taps) are limited to [0, 255].
    A conv that violates either must decline BEFORE the descriptor is
    created: an illegal corner fails descriptor creation outright, and an
    illegal tap is undefined behaviour at the instruction -- it does not fail
    loudly.  E.g. R = 257 passes every other eligibility test and produces
    tap r = 256.

    The Python dispatch tests the same predicate before it allocates
    anything (`_conv_igemm_descriptor_legal` in aten_fast.py); this is the
    backstop that keeps an illegal descriptor from reaching the driver if a
    future caller forgets.
    """

    @always_inline
    def corner_ok(v: Int) -> Bool:
        return v >= -128 and v <= 127

    return (
        corner_ok(-pad_h)
        and corner_ok(-pad_w)
        and corner_ok(pad_h - (r_f - 1))
        and corner_ok(pad_w - (s_f - 1))
        and r_f >= 1
        and s_f >= 1
        and r_f - 1 <= 255
        and s_f - 1 <= 255
    )


def _ci_make_im2col_tma[
    dtype: DType, bn: Int
](
    ctx: DeviceContext,
    act: UnsafePointer[Scalar[dtype], MutAnyOrigin],
    n: Int,
    h: Int,
    w: Int,
    c_pad: Int,
    pad_h: Int,
    pad_w: Int,
    r_f: Int,
    s_f: Int,
    out_h: Int,
    out_w: Int,
) raises -> TMATensorTileIm2col[dtype, 2, Index(bn, _CI_BK), Index(bn, _CI_BK)]:
    """CUTLASS fprop corners: lower = -pad, upper = pad - (filter - 1), which
    makes the pixel box exactly the output grid (box_w == out_w,
    box_h == out_h) for stride = dilation = 1.  The dimensions are handed in
    NHWC order and reversed to CWHDN inside `create_tensormap_im2col`; the
    strides are element counts, not bytes."""
    if not conv_igemm_descriptor_legal(pad_h, pad_w, r_f, s_f):
        raise Error("conv igemm: im2col descriptor domain violated")
    var tensormap = create_tensormap_im2col[dtype, 4, 2](
        DeviceBuffer(
            ctx, act.address_space_cast[AddressSpace.GENERIC](), 1, owning=False
        ),
        IndexList[4](n, h, w, c_pad),
        IndexList[4](h * w * c_pad, w * c_pad, c_pad, 1),
        IndexList[2](-pad_h, -pad_w),
        IndexList[2](pad_h - (r_f - 1), pad_w - (s_f - 1)),
        _CI_BK,
        bn,
        SwizzleMode(Int32(Int(_CI_SWZ))),
    )
    var desc = TMADescriptor()
    desc.data = tensormap.data
    return TMATensorTileIm2col[dtype, 2, Index(bn, _CI_BK), Index(bn, _CI_BK)](
        desc,
        UInt32(out_h),
        UInt32(out_w),
        UInt32(r_f),
        UInt32(s_f),
        UInt32(c_pad),
        Int32(-pad_h),
        Int32(-pad_w),
    )


def _ci_enqueue_gemm[
    dtype: DType, stages: Int, bn: Int, tma_store: Bool
](
    out_ptr: UnsafePointer[Scalar[dtype], MutAnyOrigin],  # (N, out_c, OH*OW)
    act_nhwc: UnsafePointer[Scalar[dtype], MutAnyOrigin],  # (N, H, W, C_pad)
    wpack: UnsafePointer[Scalar[dtype], MutAnyOrigin],  # (out_c, K_pad)
    n: Int,
    c_pad: Int,
    h: Int,
    w: Int,
    out_c: Int,
    r_f: Int,
    s_f: Int,
    pad_h: Int,
    pad_w: Int,
    out_h: Int,
    out_w: Int,
    ctx: DeviceContext,
) raises:
    var k_pad = r_f * s_f * c_pad
    var hw = out_h * out_w
    var a_desc = create_tma_descriptor[dtype, 2, _CI_SWZ](
        DeviceBuffer(
            ctx,
            wpack.address_space_cast[AddressSpace.GENERIC](),
            1,
            owning=False,
        ),
        IndexList[2](out_c, k_pad),
        IndexList[2](k_pad, 1),
        IndexList[2](_CI_BM, _CI_BK),
    )
    # A TMA store needs a 16 B aligned output row pitch; a dummy descriptor
    # over the first tile keeps the kernel signature uniform when the shape
    # cannot have one (the scalar epilogue then owns the store).
    var c_dim0 = n if tma_store else 1
    var c_dim1 = out_c if tma_store else _CI_BM
    var c_dim2 = hw if tma_store else 64
    var c_str0 = out_c * hw if tma_store else _CI_BM * 64
    var c_str1 = hw if tma_store else 64
    var c_desc = create_tma_descriptor[dtype, 3, _CI_SWZ](
        DeviceBuffer(
            ctx,
            out_ptr.address_space_cast[AddressSpace.GENERIC](),
            1,
            owning=False,
        ),
        IndexList[3](c_dim0, c_dim1, c_dim2),
        IndexList[3](c_str0, c_str1, 1),
        IndexList[3](1, _CI_BM, 64),
    )
    var a_tma = TMATensorTile[
        dtype, 2, Index(_CI_BM, _CI_BK), Index(_CI_BM, _CI_BK)
    ](a_desc)
    var c_tma = TMATensorTile[
        dtype, 3, Index(1, _CI_BM, 64), Index(1, _CI_BM, 64)
    ](c_desc)
    var b_tma = _ci_make_im2col_tma[dtype, bn](
        ctx, act_nhwc, n, h, w, c_pad, pad_h, pad_w, r_f, s_f, out_h, out_w
    )
    var works = n * ((out_c + _CI_BM - 1) // _CI_BM) * ((hw + bn - 1) // bn)
    # Not `_enqueue_cached`: the TMA descriptors are per-call `grid_constant`
    # kernel arguments, which is also why every other TMA family in this
    # package (gemm16, flash attention) enqueues directly.  The three
    # `create_tma_descriptor` calls above dominate the host cost here anyway.
    ctx.enqueue_function[_ci_conv_igemm_kernel[dtype, stages, bn, tma_store]](
        a_tma,
        b_tma,
        c_tma,
        out_ptr,
        Int32(out_c),
        Int32(hw),
        Int32(k_pad),
        Int32(c_pad),
        Int32(out_w),
        Int32(pad_h),
        Int32(pad_w),
        Int32(s_f),
        grid_dim=(works,),
        block_dim=(128,),
    )


@always_inline
def _ci_conv_igemm[
    dtype: DType
](
    out_addr: Int,
    act_addr: Int,
    wpack_addr: Int,
    in_addr: Int,
    weight_addr: Int,
    n: Int,
    c: Int,
    c_pad: Int,
    h: Int,
    w: Int,
    out_c: Int,
    r_f: Int,
    s_f: Int,
    pad_h: Int,
    pad_w: Int,
    out_h: Int,
    out_w: Int,
    ctx: DeviceContext,
) raises:
    """The whole route: NHWC transpose, weight repack, implicit GEMM."""
    var out_ptr = _make_ptr[dtype](out_addr).as_unsafe_any_origin()
    var act_ptr = _make_ptr[dtype](act_addr).as_unsafe_any_origin()
    var wpack_ptr = _make_ptr[dtype](wpack_addr).as_unsafe_any_origin()
    var in_ptr = _make_ptr[dtype](in_addr).as_unsafe_any_origin()
    var weight_ptr = _make_ptr[dtype](weight_addr).as_unsafe_any_origin()
    var hw_in = h * w
    var hw = out_h * out_w

    _enqueue_cached[_ci_nchw_to_nhwc_kernel[dtype]](
        ctx,
        String(t"conv_nchw_to_nhwc_padc_{dtype}"),
        (hw_in + _CI_T_PIX - 1) // _CI_T_PIX,
        (c_pad + _CI_T_CH - 1) // _CI_T_CH,
        n,
        _CI_T_THREADS,
        act_ptr,
        in_ptr,
        Int32(hw_in),
        Int32(c),
        Int32(c_pad),
    )
    _enqueue_cached[_ci_weight_repack_kernel[dtype]](
        ctx,
        String(t"conv_weight_repack_rsc_{dtype}"),
        (out_c * r_f * s_f * c_pad + 255) // 256,
        1,
        1,
        256,
        wpack_ptr,
        weight_ptr,
        Int32(out_c),
        Int32(c),
        Int32(c_pad),
        Int32(r_f),
        Int32(s_f),
    )

    # A TMA-store descriptor needs a 16 B aligned output row pitch; the
    # scalar epilogue covers every other pitch.
    var tma_store = (hw * size_of[dtype]()) % 16 == 0
    # bn: 128 everywhere except small pixel counts, where the finer tile
    # wastes less of the last block and fills the grid better (measured on an
    # H100 PCIe: mid 32x256x14x14 hw=196 is 66.6 us at bn=64 vs 68.3 at
    # bn=128; the body's hw=3136 is 59.0 at bn=128 vs 59.5 at bn=64).
    var small_hw = hw < 512
    if tma_store:
        if small_hw:
            _ci_enqueue_gemm[dtype, 2, 64, True](
                out_ptr,
                act_ptr,
                wpack_ptr,
                n,
                c_pad,
                h,
                w,
                out_c,
                r_f,
                s_f,
                pad_h,
                pad_w,
                out_h,
                out_w,
                ctx,
            )
        else:
            _ci_enqueue_gemm[dtype, 2, 128, True](
                out_ptr,
                act_ptr,
                wpack_ptr,
                n,
                c_pad,
                h,
                w,
                out_c,
                r_f,
                s_f,
                pad_h,
                pad_w,
                out_h,
                out_w,
                ctx,
            )
    else:
        if small_hw:
            _ci_enqueue_gemm[dtype, 2, 64, False](
                out_ptr,
                act_ptr,
                wpack_ptr,
                n,
                c_pad,
                h,
                w,
                out_c,
                r_f,
                s_f,
                pad_h,
                pad_w,
                out_h,
                out_w,
                ctx,
            )
        else:
            _ci_enqueue_gemm[dtype, 2, 128, False](
                out_ptr,
                act_ptr,
                wpack_ptr,
                n,
                c_pad,
                h,
                w,
                out_c,
                r_f,
                s_f,
                pad_h,
                pad_w,
                out_h,
                out_w,
                ctx,
            )


def _conv_igemm_go(
    out_ptr: PyObjectPtr,
    act_ptr: PyObjectPtr,
    wpack_ptr: PyObjectPtr,
    in_ptr: PyObjectPtr,
    weight_ptr: PyObjectPtr,
    params: PyObjectPtr,
    dtype_obj: PyObjectPtr,
    device_context_ptr: PyObjectPtr,
) raises:
    var dtype = _raw_dtype_int(dtype_obj)
    var out_addr = _raw_int(out_ptr)
    var act_addr = _raw_int(act_ptr)
    var wpack_addr = _raw_int(wpack_ptr)
    var in_addr = _raw_int(in_ptr)
    var weight_addr = _raw_int(weight_ptr)
    var n = _raw_tuple_int(params, 0)
    var c = _raw_tuple_int(params, 1)
    var c_pad = _raw_tuple_int(params, 2)
    var h = _raw_tuple_int(params, 3)
    var w = _raw_tuple_int(params, 4)
    var out_c = _raw_tuple_int(params, 5)
    var r_f = _raw_tuple_int(params, 6)
    var s_f = _raw_tuple_int(params, 7)
    var pad_h = _raw_tuple_int(params, 8)
    var pad_w = _raw_tuple_int(params, 9)
    var out_h = _raw_tuple_int(params, 10)
    var out_w = _raw_tuple_int(params, 11)
    var ctx = _raw_ctx(device_context_ptr)

    var handled = False
    comptime for dt in _CI_DTYPES:
        comptime if _dtype_arg_on[0, dt]():
            if dtype == dt:
                _ci_conv_igemm[dt](
                    out_addr,
                    act_addr,
                    wpack_addr,
                    in_addr,
                    weight_addr,
                    n,
                    c,
                    c_pad,
                    h,
                    w,
                    out_c,
                    r_f,
                    s_f,
                    pad_h,
                    pad_w,
                    out_h,
                    out_w,
                    ctx,
                )
                handled = True
    if not handled:
        raise Error(
            "unsupported dtype for the implicit-GEMM conv: " + String(dtype)
        )


def _conv_igemm_dispatcher(
    py_self: PyObjectPtr,
    args_safe: Pointer[PyObjectPtr, MutUntrackedOrigin],
    nargs: Py_ssize_t,
) abi("C") -> PyObjectPtr:
    var args = UnsafePointer(args_safe)
    try:
        _conv_igemm_go(
            args[0],
            args[1],
            args[2],
            args[3],
            args[4],
            args[5],
            args[6],
            args[7],
        )
    except e:
        return _spec_unsupported(e)
    return _raw_ret_none()


@export
def PyInit_conv_igemm_ops() abi("C") -> PythonObject:
    try:
        var b = PythonModuleBuilder("conv_igemm_ops")
        comptime if _op_on["ConvIgemm"]():
            _register_call(
                b,
                _conv_igemm_dispatcher,
                docstring=(
                    "(out_ptr, act_nhwc_ptr, weight_pack_ptr, in_ptr,"
                    " weight_ptr, (n, c, c_pad, h, w, out_c, r, s, pad_h,"
                    " pad_w, out_h, out_w), dtype, context_ptr); sm_90a"
                    " implicit-GEMM conv2d forward with no materialized"
                    " im2col"
                ),
            )
        return b.finalize()
    except e:
        abort(t"failed to create conv_igemm_ops python module: {e}")
