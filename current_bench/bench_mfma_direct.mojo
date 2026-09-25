"""Fast standalone MI300X BF16 MFMA schedule microbenchmark.

Only one schedule is instantiated per build. Tensor extents are runtime command
line values; ``-D`` options select reusable tile/pipeline regimes, not model
dimensions. Example:

    uv run --no-sync mojo build bench_mfma_direct.mojo -o /tmp/bench_mfma \
        -D BM=32 -D BN=32 -D WM=32 -D WN=32 -D BK=32 \
        -D WARP_K=2 -D STAGES=3
    uv run --no-sync /tmp/bench_mfma --m 512 --n 768 --k 3072
"""

from std.builtin.sort import sort
from std.collections import List, Optional
from max.gpu import block_idx, grid_dim, thread_idx
from max.gpu.host import DeviceContext, FuncAttribute
from std.math import ceildiv
from std.time import perf_counter_ns
from std.sys import get_defined_bool, get_defined_int
from std.utils import Index, IndexList

from internal_utils import arg_parse
from layout import Coord, TileTensor, row_major
from layout.tensor_core import get_mma_shape
from linalg.matmul.gpu import multistage_gemm_kernel
from linalg.utils import elementwise_epilogue_type
from linalg.utils_gpu import MatmulConfig


comptime BM = get_defined_int["BM", 32]()
comptime BN = get_defined_int["BN", 32]()
comptime WM = get_defined_int["WM", 32]()
comptime WN = get_defined_int["WN", 32]()
comptime BK = get_defined_int["BK", 32]()
comptime WARP_K = get_defined_int["WARP_K", 2]()
comptime K_GROUP = get_defined_int["K_GROUP", 1]()
comptime STAGES = get_defined_int["STAGES", 2]()
comptime DUMP_ASM = get_defined_bool["DUMP_ASM", False]()
comptime THREADS = (BM // WM) * (BN // WN) * WARP_K * 64


@__name("bench_mfma_fill_bf16")
def _fill_bf16(
    ptr: Pointer[Scalar[DType.bfloat16], MutAnyOrigin],
    count_arg: Int64,
    value: Scalar[DType.bfloat16],
):
    var count = Int(count_arg)
    var i = (Int(block_idx.x) * 256 + Int(thread_idx.x)) * 4
    var stride = Int(grid_dim.x) * 256 * 4
    while i < count:
        if i + 4 <= count:
            ptr.unsafe_store[width=4](i, SIMD[DType.bfloat16, 4](value))
        else:
            for lane in range(4):
                if i + lane < count:
                    ptr[unsafe_offset=i + lane] = value
        i += stride


@always_inline
def _enqueue_fill(
    ptr: Pointer[Scalar[DType.bfloat16], MutAnyOrigin],
    count: Int,
    value: Scalar[DType.bfloat16],
    ctx: DeviceContext,
) raises:
    ctx.enqueue_function[_fill_bf16](
        ptr,
        Int64(count),
        value,
        grid_dim=(max(1, min(ceildiv(count, 1024), 512)),),
        block_dim=(256,),
    )


@always_inline
def _percentile(sorted_samples: List[Float64], numerator: Int) -> Float64:
    var scaled = numerator * (len(sorted_samples) - 1)
    var lower = scaled // 100
    var remainder = scaled % 100
    if remainder == 0:
        return sorted_samples[lower]
    var upper = lower + 1
    return (
        sorted_samples[lower]
        + (sorted_samples[upper] - sorted_samples[lower])
        * Float64(remainder)
        / 100.0
    )


def main() raises:
    var m = Int(arg_parse("m", 512))
    var n = Int(arg_parse("n", 768))
    var k = Int(arg_parse("k", 3072))
    var warmup = Int(arg_parse("warmup", 25))
    var iterations = Int(arg_parse("iterations", 100))
    if warmup < 25 or iterations < 100:
        raise Error("protocol requires >=25 warmups and >=100 iterations")
    if m % BM != 0 or n % BN != 0 or k % (BK * WARP_K) != 0:
        raise Error("runtime shape must be divisible by the selected schedule")

    with DeviceContext() as ctx:
        var a_buf = ctx.enqueue_create_buffer[DType.bfloat16](m * k)
        var b_buf = ctx.enqueue_create_buffer[DType.bfloat16](k * n)
        var bias_buf = ctx.enqueue_create_buffer[DType.bfloat16](n)
        var c_buf = ctx.enqueue_create_buffer[DType.bfloat16](m * n)

        var one = Float32(1.0).cast[DType.bfloat16]()
        var half = Float32(0.5).cast[DType.bfloat16]()
        _enqueue_fill(
            a_buf.unsafe_ptr().as_unsafe_any_origin(), m * k, one, ctx
        )
        _enqueue_fill(
            b_buf.unsafe_ptr().as_unsafe_any_origin(), k * n, half, ctx
        )
        _enqueue_fill(bias_buf.unsafe_ptr().as_unsafe_any_origin(), n, one, ctx)

        var a = TileTensor[mut=False](a_buf, row_major(Coord(m, k)))
        var b = TileTensor[mut=False](b_buf, row_major(Coord(k, n)))
        var c = TileTensor[mut=True](c_buf, row_major(Coord(m, n)))
        var c_ptr = c_buf.unsafe_ptr().as_unsafe_any_origin()
        var bias_ptr = bias_buf.unsafe_ptr().as_unsafe_any_origin().as_imm()

        @always_inline
        @__parameter
        @__copy_capture(c_ptr, bias_ptr, n)
        def _bias_store[
            value_dtype: DType, width: SIMDLength, *, alignment: Int = 1
        ](coords: IndexList[2], value: SIMD[value_dtype, width]):
            var row = Int(coords[0])
            var col = Int(coords[1])
            var off = row * n + col
            var result = (
                value.cast[DType.float32]()
                + bias_ptr.unsafe_load[width=width](col).cast[DType.float32]()
            )
            c_ptr.unsafe_store[width=width, alignment=4](
                off, result.cast[DType.bfloat16]()
            )

        comptime bias_epilogue = Optional[elementwise_epilogue_type](
            _bias_store
        )
        comptime config = MatmulConfig[
            DType.bfloat16,
            DType.bfloat16,
            DType.bfloat16,
            False,
        ](
            block_tile_shape=Index(BM, BN, BK),
            warp_tile_shape=Index(WM, WN, BK),
            mma_shape=get_mma_shape[DType.bfloat16, DType.float32](),
            num_pipeline_stages=STAGES,
            num_warp_k_partitions=WARP_K,
            k_group_size=K_GROUP,
        )
        comptime kernel = multistage_gemm_kernel[
            CLT=c.LayoutType,
            ALT=a.LayoutType,
            BLT=b.LayoutType,
            c_linear_idx_type=c.linear_idx_type,
            a_linear_idx_type=a.linear_idx_type,
            b_linear_idx_type=b.linear_idx_type,
            config=config,
            elementwise_lambda_fn=bias_epilogue,
        ]

        @always_inline
        @__parameter
        def _launch() raises:
            ctx.enqueue_function[kernel](
                c,
                a,
                b,
                grid_dim=(n // BN, m // BM),
                block_dim=config.block_dim(),
                shared_mem_bytes=config.shared_mem_usage(),
                func_attribute=FuncAttribute.MAX_DYNAMIC_SHARED_SIZE_BYTES(
                    UInt32(config.shared_mem_usage())
                ),
            )

        comptime if DUMP_ASM:
            ctx.enqueue_function[kernel, dump_asm=True](
                c,
                a,
                b,
                grid_dim=(n // BN, m // BM),
                block_dim=config.block_dim(),
                shared_mem_bytes=config.shared_mem_usage(),
                func_attribute=FuncAttribute.MAX_DYNAMIC_SHARED_SIZE_BYTES(
                    UInt32(config.shared_mem_usage())
                ),
            )
            ctx.synchronize()

        for _ in range(warmup):
            _launch()
        ctx.synchronize()

        var samples = List[Float64](capacity=iterations)
        for _ in range(iterations):
            ctx.synchronize()
            var start = perf_counter_ns()
            _launch()
            ctx.synchronize()
            var stop = perf_counter_ns()
            samples.append(Float64(stop - start) / 1000.0)
        sort(samples)

        var host = ctx.enqueue_create_host_buffer[DType.bfloat16](m * n)
        ctx.enqueue_copy(host, c_buf)
        ctx.synchronize()
        var expected = Float32(k) * 0.5 + 1.0
        for idx in [0, n - 1, (m // 2) * n + n // 2, m * n - 1]:
            var actual = host[idx].cast[DType.float32]()
            if abs(actual - expected) > 1.0:
                raise Error(
                    "correctness smoke failed at "
                    + String(idx)
                    + ": got "
                    + String(actual)
                    + ", expected "
                    + String(expected)
                )

        var median = _percentile(samples, 50)
        var p10 = _percentile(samples, 10)
        var p90 = _percentile(samples, 90)
        var flops = 2.0 * Float64(m) * Float64(n) * Float64(k)
        var tflops = flops / (median * 1.0e6)
        print(
            "shape=",
            m,
            "x",
            n,
            "x",
            k,
            " config=",
            BM,
            "x",
            BN,
            "x",
            BK,
            " warp=",
            WM,
            "x",
            WN,
            " warp_k=",
            WARP_K,
            " stages=",
            STAGES,
            " median_us=",
            median,
            " p10_us=",
            p10,
            " p90_us=",
            p90,
            " tflops=",
            tflops,
            " correctness=pass",
        )

        _ = a_buf^
        _ = b_buf^
        _ = bias_buf^
        _ = c_buf^
