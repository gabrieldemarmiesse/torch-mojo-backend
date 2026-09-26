"""GPU elementwise launcher: SAME API as `max.algorithm.elementwise` (MAX
26.6, max/mojo/max/algorithm/functional.mojo), so every eager call site
imports this instead:

    elementwise[func, simd_width=W, target="gpu", _trace_description="..."](
        Coord(shape), ctx)                       # capturing-closure form
    elementwise[W, target="gpu"](func, shape, ctx)  # unified-closure form

`func[width, alignment](idx: Coord)` is the caller's body; `_heavy=True`
(an extension MAX's API does not have, ignored off the fast path) marks a
body expensive enough to prefer smaller blocks (`_policy_block`). The NVIDIA GPU
rank-1 path is replaced; everything else (CPU, AMD/Apple GPUs, rank > 1)
forwards to MAX unchanged -- only NVIDIA (H100 PCIe, sm_90a) was measured,
so nothing here may regress those other targets.

It is also gated on the eager backend, not just on NVIDIA/rank-1
(`_EAGER_BUILD` below): a handful of eager kernels this launcher would
otherwise reach -- `_bias_add_row` (matmul/entry.mojo), `_gather0` and
other `op_utils._parallel_for` callers, the softmax-rows and
batch/layer-norm kernels -- are ALSO imported by `tmb/graph`'s custom ops,
which is how the torch.compile backend reaches them. torch.compile must
keep MAX's own `elementwise` there, so the fast path only activates when
`-D TMB_EAGER_ELEMENTWISE=1` is set, which `tmb/backend/loader.mojo`
appends to every eager kernel-family build and the graph package's `mojo
precompile` never does.

Same contract as MAX, and the same index partition, so results are
bit-identical: vectors at multiples of `simd_width` get `func[W, W]`, the
`length % W` tail elements get `func[1]`. What differs is the launch:
  - no grid-stride loop and no per-element bounds check: each block owns one
    tile of `block * unroll` vectors, full tiles run unchecked, only the last
    block checks (MAX checks every unrolled element and derives every index
    with a 64-bit divmod chain);
  - 32-bit index math up to 2**31 elements (MAX's own 32-bit gate never fires,
    see `_flat_launch`), 64-bit past that;
  - block size and unroll picked per width and size regime (policy below)
    instead of MAX's fixed 256 x 4-way grid-stride.
PDL (programmatic dependent launch) is kept exactly as MAX does it: wait on
entry, release on exit, launch attribute on.
"""
from max.algorithm import elementwise as _max_elementwise
from max.gpu import (
    MAX_THREADS_PER_BLOCK_METADATA,
    block_idx,
    grid_dim,
    thread_idx,
)
from max.gpu.host import DeviceContext
from max.gpu.primitives.grid_controls import (
    PDLLevel,
    launch_dependent_grids,
    pdl_launch_attributes,
    wait_on_dependent_grids,
)
from std.sys.defines import is_defined
from std.sys.info import has_nvidia_gpu_accelerator
from std.utils.coord import Coord
from std.utils.index import IndexList
from std.utils.static_tuple import StaticTuple

# Set on every eager kernel build (`tmb/backend/loader.mojo`'s `entry()`),
# never on the torch.compile graph package (`native.build_graph_package()`'s
# `mojo precompile` passes no `-D` at all, and MAX's in-process graph
# compiler that elaborates a custom op's body for a device sets none
# either). Several eager kernels this launcher's fast path serves are also
# imported by tmb/graph's custom ops (`_bias_add_row` in matmul/entry.mojo,
# `_gather0` / `op_utils._parallel_for`, softmax_rows_kernels,
# normalization_forward), so gating on this -- in addition to the NVIDIA /
# rank-1 gates below -- is what keeps the fast launcher out of
# torch.compile graphs while every eager call site still gets it through
# the plain `elementwise[...]` import swap.
comptime _EAGER_BUILD = is_defined["TMB_EAGER_ELEMENTWISE"]()

# Launch policy, per simd_width, fitted on H100 PCIe (sm_90a, 114 SMs, SM
# clock pinned at 1395 MHz) with a standalone harness (neg = HBM/L2 bound,
# acosh = issue bound; f16/bf16/f32; 4097 .. 16M elements). The launcher sees
# only `simd_width`, not the dtype, so each width gets what is best across
# the dtypes that call it. Three regimes by packed-vector count:
#   TINY  (<= `_tiny_waves` waves of resident threads): unroll 1 -- the grid is
#         too small for block dispatch to matter, so every vector gets its
#         own thread (unroll serializes: see below);
#   BIG   (>= _BIG_PACKED vectors, HBM-streaming sizes);
#   SMALL (the rest).
# What shapes the numbers:
#   - GPU-wide CTA dispatch is ~660 blocks/us on this part (measured: a kernel
#     of 131072 one-vector 128-thread blocks takes 193 us however little each
#     does), so a launch with too many tiny blocks is dispatch-bound: narrow
#     widths want unroll > 1 or bigger blocks;
#   - a body's load cannot be hoisted above the previous unrolled call's
#     store (the pointers may alias), so unroll > 1 serializes the memory
#     latency inside a thread; it only pays where it cuts a large block count;
#   - compute-bound bodies prefer smaller blocks (finer tail balancing).
comptime _BIG_PACKED = 1 << 20
comptime _TINY = 0
comptime _SMALL = 1
comptime _BIG = 2


def _policy_block[simd_width: Int, regime: Int, heavy: Bool]() -> Int:
    comptime if simd_width <= 2:
        return 512 if regime == _TINY else 256
    elif simd_width <= 4:
        # BIG: 1024 is the one block size that matches MAX on f32 16B HBM
        # streaming (MAX's 40-register kernel runs at ~60% occupancy; ours at
        # ~80% was 0.2-0.6% slower at 512 threads, same output buffer).
        return 1024 if regime == _BIG else 256
    elif simd_width <= 8:
        # A `heavy` body (the caller's call: transcendental or looping math)
        # takes 128-thread blocks in TINY, the finer balance of a grid that
        # does not fill the GPU; cheap bodies keep 256. benchmarks/, 2-byte
        # dtypes at 281673 elements, 256 -> 128: mish bf16 1.22 -> 1.10x
        # torch, hypot 1.20 -> 1.04, log_sigmoid 1.20 -> 1.09, atan2 1.06 ->
        # 0.88; but neg / abs 0.98 -> 1.09, floor / trunc 1.02 -> 1.09.
        return 128 if regime == _TINY and heavy else 256
    else:
        return 512 if regime == _BIG else 128


def _policy_unroll[simd_width: Int, regime: Int]() -> Int:
    comptime if regime == _TINY:
        return 1
    comptime if simd_width <= 2:
        return 4
    elif simd_width <= 4:
        return 2 if regime == _SMALL else 1
    else:
        return 1


def _tiny_waves[simd_width: Int]() -> Int:
    """How many full waves of resident threads count as TINY."""
    return 2 if simd_width <= 2 else 1


@fieldwise_init
struct _FlatKernel[
    FuncType: ImplicitlyCopyable
    & RegisterPassable
    & def[width: Int, alignment: Int = 1](Coord) -> None,
    //,
    idx_type: DType,
    simd_width: Int,
    block_size: Int,
    unroll: Int,
    trace_description: StaticString = "",
](ImplicitlyCopyable, RegisterPassable, def() -> None):
    """Block b covers packed vectors [b * tile, (b + 1) * tile), tile =
    block_size * unroll, thread t the vectors b * tile + u * block_size + t
    (coalesced per u). No grid-stride loop: the grid covers the range
    exactly. Full tiles run without bounds checks; the scalar tail
    (length % simd_width elements) goes to the first threads of the last
    block. Same width/index partition as MAX: vectors at multiples of
    simd_width with alignment simd_width, tail elements at width 1."""

    var func: Self.FuncType
    var num_packed: Scalar[Self.idx_type]
    var tail: Scalar[Self.idx_type]

    @__llvm_metadata(
        MAX_THREADS_PER_BLOCK_METADATA=StaticTuple[Int32, 1](
            Int32(Self.block_size)
        )
    )
    @__name(
        t"{Self.trace_description}_r1_w{Self.simd_width}_b{Self.block_size}_u{Self.unroll}.flat"
    )
    def __call__(self) capturing:
        comptime T = Scalar[Self.idx_type]
        comptime W = Self.simd_width
        comptime B = Self.block_size
        comptime tile = B * Self.unroll
        var tid = T(thread_idx.x)
        var tile_start = T(block_idx.x) * T(tile)
        var first = tile_start + tid

        # PDL exactly as MAX's `with PDL()`: wait on entry, release at exit.
        wait_on_dependent_grids()

        comptime if Self.unroll == 1:
            if first < self.num_packed:
                self.func[W, W](Coord(IndexList[1](Int(first * T(W)))))
        else:
            if tile_start + T(tile) <= self.num_packed:
                comptime for u in range(Self.unroll):
                    var p = first + T(u * B)
                    self.func[W, W](Coord(IndexList[1](Int(p * T(W)))))
            else:
                comptime for u in range(Self.unroll):
                    var p = first + T(u * B)
                    if p < self.num_packed:
                        self.func[W, W](Coord(IndexList[1](Int(p * T(W)))))

        comptime if W > 1:
            if T(block_idx.x) == T(grid_dim.x) - 1 and tid < self.tail:
                self.func[1](
                    Coord(IndexList[1](Int(self.num_packed * T(W) + tid)))
                )

        launch_dependent_grids()


@always_inline
def _launch_one[
    FuncType: ImplicitlyCopyable
    & RegisterPassable
    & def[width: Int, alignment: Int = 1](Coord) -> None,
    //,
    idx_type: DType,
    simd_width: Int,
    block_size: Int,
    unroll: Int,
    trace_description: StaticString,
](func: FuncType, num_packed: Int, tail: Int, ctx: DeviceContext) raises:
    comptime tile = block_size * unroll
    var blocks = max(1, (num_packed + tile - 1) // tile)
    var k = _FlatKernel[
        idx_type=idx_type,
        simd_width=simd_width,
        block_size=block_size,
        unroll=unroll,
        trace_description=trace_description,
    ](func, Scalar[idx_type](num_packed), Scalar[idx_type](tail))
    ctx.enqueue_function(
        k,
        grid_dim=blocks,
        block_dim=block_size,
        attributes=pdl_launch_attributes(PDLLevel.ON),
    )


@always_inline
def _flat_launch[
    FuncType: ImplicitlyCopyable
    & RegisterPassable
    & def[width: Int, alignment: Int = 1](Coord) -> None,
    //,
    simd_width: Int,
    trace_description: StaticString,
    heavy: Bool = False,
](func: FuncType, length: Int, ctx: DeviceContext) raises:
    if length <= 0:
        return
    var num_packed = length // simd_width
    var tail = length - num_packed * simd_width
    comptime Bt = _policy_block[simd_width, _TINY, heavy]()
    comptime Ut = _policy_unroll[simd_width, _TINY]()
    comptime Bs = _policy_block[simd_width, _SMALL, heavy]()
    comptime Us = _policy_unroll[simd_width, _SMALL]()
    comptime Bb = _policy_block[simd_width, _BIG, heavy]()
    comptime Ub = _policy_unroll[simd_width, _BIG]()
    comptime max_tile = max(Bt * Ut, max(Bs * Us, Bb * Ub))
    comptime tiny_max = (
        ctx.default_device_info.sm_count
        * ctx.default_device_info.threads_per_multiprocessor
        * _tiny_waves[simd_width]()
    )
    # Every index the kernel forms is < length + tile * simd_width. 32-bit
    # math whenever that stays below 2**31, not 2**32: a runtime UInt32
    # converts to Int correctly (zero-extends; measured: `Int` of a
    # non-comptime UInt32 holding 4294967295 is 4294967295), but Mojo 1.1
    # converts the comptime-folded named constant `UInt32.MAX` to -1
    # (`Int(UInt32.MAX) == -1`, a comptime quirk, not a property of
    # `Int(UInt32)` in general). MAX's own 32-bit gate reads
    # `length <= Int(UInt32.MAX)`, so it hits exactly that quirk and always
    # evaluates False, meaning MAX 26.6 always runs its 64-bit instantiation.
    if length + 2 * max_tile * simd_width > (1 << 31) - 1:
        _launch_one[DType.uint64, simd_width, Bb, Ub, trace_description](
            func, num_packed, tail, ctx
        )
        return
    # Regimes that share a configuration share one instantiation.
    comptime if Bt != Bs or Ut != Us:
        if num_packed <= tiny_max:
            _launch_one[DType.uint32, simd_width, Bt, Ut, trace_description](
                func, num_packed, tail, ctx
            )
            return
    comptime if Bb != Bs or Ub != Us:
        if num_packed >= _BIG_PACKED:
            _launch_one[DType.uint32, simd_width, Bb, Ub, trace_description](
                func, num_packed, tail, ctx
            )
            return
    _launch_one[DType.uint32, simd_width, Bs, Us, trace_description](
        func, num_packed, tail, ctx
    )


@always_inline
def elementwise[
    func: def[width: Int, alignment: Int = 1](Coord) capturing[_] -> None,
    simd_width: Int,
    *,
    target: StaticString = "cpu",
    _trace_description: StaticString = "elementwise",
    _heavy: Bool = False,
](shape: Coord, context: DeviceContext) raises:
    comptime if (
        _EAGER_BUILD
        and target == "gpu"
        and has_nvidia_gpu_accelerator()
        and shape.rank == 1
    ):

        def func_unified[width: Int, alignment: Int = 1](indices: Coord) {}:
            func[width, alignment](indices)

        _flat_launch[simd_width, _trace_description, _heavy](
            func_unified, Int(shape.product()), context
        )
    else:
        _max_elementwise[
            func,
            simd_width=simd_width,
            target=target,
            _trace_description=_trace_description,
        ](shape, context)


@always_inline
def elementwise[
    FuncType: ImplicitlyCopyable
    & RegisterPassable
    & def[width: Int, alignment: Int = 1](Coord) -> None,
    //,
    simd_width: Int,
    *,
    target: StaticString = "cpu",
    _trace_description: StaticString = "elementwise",
    _heavy: Bool = False,
](func: FuncType, shape: Coord, context: DeviceContext) raises:
    comptime if (
        _EAGER_BUILD
        and target == "gpu"
        and has_nvidia_gpu_accelerator()
        and shape.rank == 1
    ):
        _flat_launch[simd_width, _trace_description, _heavy](
            func, Int(shape.product()), context
        )
    else:
        _max_elementwise[
            simd_width,
            target=target,
            _trace_description=_trace_description,
        ](func, shape, context)
