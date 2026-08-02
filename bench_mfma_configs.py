"""Benchmark-only sweep of reusable MI300X BF16 MFMA configurations."""

import argparse
import csv
import math
import statistics
import subprocess
import time
from collections.abc import Callable
from pathlib import Path
from types import ModuleType

import torch

from torch_mojo_backend import get_accelerators, register_mojo_devices

CONFIGS = {
    0: "bm32_bn64_wm16_wn32_bk32",
    1: "bm32_bn128_wm16_wn64_bk32",
    2: "bm32_bn128_wm32_wn64_bk32",
    3: "bm64_bn64_wm32_wn32_bk32",
    4: "bm64_bn128_wm32_wn64_bk32",
    5: "bm96_bn64_wm48_wn32_bk64",
    6: "bm128_bn64_wm64_wn32_bk32",
    7: "bm32_bn32_wm32_wn32_bk64_kp2",
    8: "bm32_bn64_wm32_wn32_bk64_kp2",
    9: "bm64_bn32_wm32_wn32_bk64_kp2",
    15: "bm32_bn32_wm32_wn32_bk32_kp2",
    16: "bm32_bn32_wm32_wn32_bk64_kp4",
    17: "bm32_bn32_wm32_wn32_bk64_kp2_kg2",
    18: "bm16_bn32_wm16_wn32_bk32",
    19: "bm16_bn64_wm16_wn32_bk32",
    20: "bm64_bn32_wm32_wn32_bk32",
    21: "bm32_bn64_wm16_wn32_bk64_kg2",
    22: "bm128_bn32_wm64_wn32_bk32",
}

SHAPES = (
    (512, 768, 768),
    (512, 2304, 768),
    (512, 3072, 768),
    (512, 768, 3072),
    (4096, 768, 768),
    (4096, 2304, 768),
    (4096, 3072, 768),
    (4096, 768, 3072),
)


def percentile(samples: list[float], fraction: float) -> float:
    ordered = sorted(samples)
    position = fraction * (len(ordered) - 1)
    lower = math.floor(position)
    upper = math.ceil(position)
    if lower == upper:
        return ordered[lower]
    return ordered[lower] + (ordered[upper] - ordered[lower]) * (position - lower)


def load_tuning_extension() -> ModuleType:
    """The matmul_ops extension built with no OP define, for `AmdBf16Tune`.

    Operations are compiled one at a time: the loader builds a `.so` per
    `-D OP=<name>` and renames that one operation to `call()`, so
    `MatmulExtension.invoke("AmdBf16Tune", ...)` cannot reach this function --
    there is no `_op_on["AmdBf16Tune"]` gate for it to select, and the loader
    rejects a module without `call()`.  `MatmulTune` and `AmdBf16Tune` are
    registered under their own names outside every gate (see
    `PyInit_matmul_ops`), so the undefined build of the module carries exactly
    those two and none of the production kernels.  Building it here keeps this
    tuning harness off the dispatch path it is meant to measure.
    """
    from torch_mojo_backend.eager_kernels import (
        _build_extension,
        _load_extension,
        _resolve_mojo_file,
        _variant_module_name,
    )
    from torch_mojo_backend.eager_kernels.matmul_ops import MatmulExtension

    source = _resolve_mojo_file(MatmulExtension.MOJO_FILE)
    return _load_extension(
        _variant_module_name(source, None), _build_extension(source, None)
    )


def measure(
    fn: Callable[[], None],
    synchronize: Callable[[], None],
    warmup: int,
    iterations: int,
) -> tuple[float, float, float]:
    for _ in range(warmup):
        fn()
    synchronize()
    samples = []
    for _ in range(iterations):
        synchronize()
        start = time.perf_counter_ns()
        fn()
        synchronize()
        samples.append((time.perf_counter_ns() - start) / 1000)
    return (
        statistics.median(samples),
        percentile(samples, 0.1),
        percentile(samples, 0.9),
    )


def rocm_smi(label: str) -> None:
    result = subprocess.run(
        ["rocm-smi", "--showclocks", "--showtemp", "--showuse", "--showpower"],
        check=False,
        capture_output=True,
        text=True,
    )
    print(f"\n================ ROCm SMI: {label} ================")
    print(result.stdout)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--warmup", type=int, default=25)
    parser.add_argument("--iterations", type=int, default=100)
    parser.add_argument("--output", type=Path, default=Path("mfma_config_sweep.csv"))
    parser.add_argument("--config", type=int, action="append")
    parser.add_argument("--shape", action="append")
    args = parser.parse_args()
    if args.warmup < 25 or args.iterations < 100:
        raise ValueError("protocol requires >=25 warmups and >=100 iterations")

    register_mojo_devices()
    device = list(get_accelerators())[0]
    from max.dtype import DType

    from torch_mojo_backend.eager_kernels import _ctx_ptr, tensor_holder
    from torch_mojo_backend.mojo_device.torch_mojo_tensor import TorchMojoTensor

    ctx = _ctx_ptr(device)
    tuning = load_tuning_extension()

    def synchronize() -> None:
        tensor_holder.synchronize(ctx)

    configs = args.config or list(CONFIGS)
    shapes = SHAPES
    if args.shape:
        shapes = tuple(
            shape
            for shape in shapes
            if "x".join(str(dim) for dim in shape) in args.shape
        )
        if not shapes:
            raise ValueError("no requested shape matched")
    rows = []
    rocm_smi("before")
    for m, n, k in shapes:
        generator = torch.Generator(device="cpu")
        # Match bench_gemm.py exactly so the diagnostic and acceptance harnesses
        # exercise identical tensors.
        generator.manual_seed(17 + 3 * m + 5 * n + 7 * k)
        a_fp32 = torch.randn((m, k), generator=generator)
        b_fp32 = torch.randn((k, n), generator=generator)
        bias_fp32 = torch.randn((n,), generator=generator)
        reference = torch.addmm(
            bias_fp32.to("cuda"), a_fp32.to("cuda"), b_fp32.to("cuda")
        )
        a_bf16 = a_fp32.to(torch.bfloat16)
        b_bf16 = b_fp32.to(torch.bfloat16)
        bias_bf16 = bias_fp32.to(torch.bfloat16)
        native = torch.addmm(bias_bf16.to("cuda"), a_bf16.to("cuda"), b_bf16.to("cuda"))
        torch.cuda.synchronize()
        reference_cpu = reference.cpu()
        native_error = (native.cpu().float() - reference_cpu.float()).abs().max().item()
        a = a_bf16.to("mojo")
        b = b_bf16.to("mojo")
        bias = bias_bf16.to("mojo")
        out = TorchMojoTensor._alloc((m, n), DType.bfloat16, device)
        print(f"\nshape M={m} N={n} K={k}; native error={native_error:.8g}")
        for cfg in configs:
            if cfg not in CONFIGS:
                raise ValueError(f"unknown config {cfg}")

            def launch() -> None:
                tuning.AmdBf16Tune(
                    out._ptr, a._ptr, b._ptr, bias._ptr, (m, n, k, cfg), ctx
                )

            median, p10, p90 = measure(
                launch, synchronize, args.warmup, args.iterations
            )
            actual = out.cpu()
            error = (actual.float() - reference_cpu.float()).abs().max().item()
            passed = error <= 2 * native_error
            print(
                f"  cfg={cfg:2d} {CONFIGS[cfg]:34s} "
                f"median={median:9.3f} us p10={p10:9.3f} p90={p90:9.3f} "
                f"error={error:.8g} pass={passed}",
                flush=True,
            )
            rows.append(
                {
                    "m": m,
                    "n": n,
                    "k": k,
                    "config_id": cfg,
                    "config": CONFIGS[cfg],
                    "median_us": f"{median:.3f}",
                    "p10_us": f"{p10:.3f}",
                    "p90_us": f"{p90:.3f}",
                    "native_bf16_error": f"{native_error:.8g}",
                    "mojo_bf16_error": f"{error:.8g}",
                    "error_limit": f"{2 * native_error:.8g}",
                    "correctness_pass": passed,
                }
            )
        torch.cuda.empty_cache()
        torch.cuda.synchronize()
        synchronize()

    rocm_smi("after")
    with args.output.open("w", newline="") as file:
        writer = csv.DictWriter(file, fieldnames=rows[0].keys())
        writer.writeheader()
        writer.writerows(rows)
    if not all(row["correctness_pass"] for row in rows):
        raise RuntimeError("one or more MFMA configurations failed correctness")


if __name__ == "__main__":
    main()
