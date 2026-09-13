"""Exact-shape GPT-2 GEMM benchmark for ROCm and the eager Mojo device.

The default cases are extracted from the shape-grouped GPT-2 profiler CSVs.
Both backends run in one process on the same MI300X. Timings include an
explicit backend synchronization around every sample.
"""

import argparse
import ast
import csv
import math
import statistics
import subprocess
import time
from dataclasses import dataclass
from pathlib import Path

import torch

from torch_mojo_backend import get_accelerators, register_mojo_devices
from torch_mojo_backend.native import device_module

MI300X_BF16_FLOPS = 1.3e15
MI300X_HBM_BYTES = 5.3e12


@dataclass(frozen=True)
class GemmCase:
    phase: str
    op: str
    m: int
    n: int
    k: int
    calls: int
    profile_mojo_us: float
    profile_rocm_us: float
    transpose_b: bool
    bias: bool

    @property
    def name(self) -> str:
        suffix = "tb" if self.transpose_b else "nn"
        return f"{self.phase}_{self.op}_m{self.m}_n{self.n}_k{self.k}_{suffix}"


def percentile(samples: list[float], fraction: float) -> float:
    ordered = sorted(samples)
    position = fraction * (len(ordered) - 1)
    lower = math.floor(position)
    upper = math.ceil(position)
    if lower == upper:
        return ordered[lower]
    return ordered[lower] + (ordered[upper] - ordered[lower]) * (position - lower)


def read_shape_rows(path: Path) -> list[dict]:
    with path.open() as file:
        return list(csv.DictReader(file))


def extract_cases(rocm_dir: Path, mojo_dir: Path) -> list[GemmCase]:
    cases = []
    for phase in ("decode", "prefill"):
        rocm_rows = read_shape_rows(rocm_dir / f"aten_gpu_time_{phase}_by_shape.csv")
        mojo_rows = read_shape_rows(mojo_dir / f"aten_gpu_time_{phase}_by_shape.csv")

        def addmm_by_geometry(rows):
            result = {}
            for row in rows:
                if row["aten_op"] != "aten::addmm":
                    continue
                shapes = ast.literal_eval(row["input_shapes"])
                _, a_shape, b_shape = shapes[:3]
                geometry = (a_shape[0], b_shape[1], a_shape[1])
                result[geometry] = row
            return result

        rocm_addmm = addmm_by_geometry(rocm_rows)
        mojo_addmm = addmm_by_geometry(mojo_rows)
        if set(rocm_addmm) != set(mojo_addmm):
            raise RuntimeError(f"{phase} addmm shape sets differ")
        for geometry in sorted(rocm_addmm):
            m, n, k = geometry
            rocm_row = rocm_addmm[geometry]
            mojo_row = mojo_addmm[geometry]
            if int(rocm_row["calls"]) != int(mojo_row["calls"]):
                raise RuntimeError(f"{phase} call counts differ for {geometry}")
            cases.append(
                GemmCase(
                    phase=phase,
                    op="addmm",
                    m=m,
                    n=n,
                    k=k,
                    calls=int(mojo_row["calls"]),
                    profile_mojo_us=float(mojo_row["avg_us_per_call"]),
                    profile_rocm_us=float(rocm_row["avg_us_per_call"]),
                    transpose_b=False,
                    bias=True,
                )
            )

        rocm_lm = next(
            row
            for row in rocm_rows
            if row["aten_op"] == "aten::mm"
            and ast.literal_eval(row["input_shapes"])[1][-1] == 50257
        )
        mojo_lm = next(
            row
            for row in mojo_rows
            if row["aten_op"] == "aten::linear"
            and ast.literal_eval(row["input_shapes"])[1][0] == 50257
        )
        rocm_shapes = ast.literal_eval(rocm_lm["input_shapes"])
        m, k = rocm_shapes[0]
        n = rocm_shapes[1][1]
        cases.append(
            GemmCase(
                phase=phase,
                op="linear",
                m=m,
                n=n,
                k=k,
                calls=int(mojo_lm["calls"]),
                profile_mojo_us=float(mojo_lm["avg_us_per_call"]),
                profile_rocm_us=float(rocm_lm["avg_us_per_call"]),
                transpose_b=True,
                bias=False,
            )
        )
    return cases


def rocm_smi_snapshot(label: str):
    result = subprocess.run(
        [
            "rocm-smi",
            "--showclocks",
            "--showtemp",
            "--showuse",
            "--showmemuse",
            "--showpower",
        ],
        check=False,
        text=True,
        capture_output=True,
    )
    print(f"\n================ ROCm SMI: {label} ================")
    print(result.stdout)


def measure(fn, synchronize, warmup: int, iterations: int):
    output = None
    for _ in range(warmup):
        output = fn()
    synchronize()
    samples_us = []
    for _ in range(iterations):
        synchronize()
        start_ns = time.perf_counter_ns()
        output = fn()
        synchronize()
        samples_us.append((time.perf_counter_ns() - start_ns) / 1000)
    return {
        "median_us": statistics.median(samples_us),
        "p10_us": percentile(samples_us, 0.10),
        "p90_us": percentile(samples_us, 0.90),
        "output": output,
    }


def max_abs_error(actual: torch.Tensor, expected: torch.Tensor) -> float:
    return (actual.float() - expected.float()).abs().max().item()


def make_case_inputs(case: GemmCase):
    generator = torch.Generator(device="cpu")
    generator.manual_seed(
        17 + case.m * 3 + case.n * 5 + case.k * 7 + int(case.transpose_b)
    )
    a_cpu = torch.randn(case.m, case.k, generator=generator)
    if case.transpose_b:
        b_cpu = torch.randn(case.n, case.k, generator=generator)
    else:
        b_cpu = torch.randn(case.k, case.n, generator=generator)
    bias_cpu = torch.randn(case.n, generator=generator) if case.bias else None
    return a_cpu, b_cpu, bias_cpu


def make_op(case: GemmCase, a, b, bias):
    if case.op == "addmm":
        return lambda: torch.ops.aten.addmm.default(bias, a, b)
    if case.op == "linear":
        return lambda: torch.ops.aten.linear.default(a, b, None)
    if case.op == "mm":
        return lambda: torch.ops.aten.mm.default(a, b)
    raise ValueError(case.op)


def run_case(case: GemmCase, mojo_synchronize, warmup: int, iterations: int) -> dict:
    a_cpu, b_cpu, bias_cpu = make_case_inputs(case)

    a_fp32 = a_cpu.to("cuda")
    b_fp32 = b_cpu.to("cuda")
    bias_fp32 = bias_cpu.to("cuda") if bias_cpu is not None else None
    reference = make_op(case, a_fp32, b_fp32, bias_fp32)()
    torch.cuda.synchronize()

    a_bf16_cpu = a_cpu.to(torch.bfloat16)
    b_bf16_cpu = b_cpu.to(torch.bfloat16)
    bias_bf16_cpu = bias_cpu.to(torch.bfloat16) if bias_cpu is not None else None

    a_rocm = a_bf16_cpu.to("cuda")
    b_rocm = b_bf16_cpu.to("cuda")
    bias_rocm = bias_bf16_cpu.to("cuda") if bias_bf16_cpu is not None else None
    rocm = measure(
        make_op(case, a_rocm, b_rocm, bias_rocm),
        torch.cuda.synchronize,
        warmup,
        iterations,
    )
    rocm_output = rocm.pop("output").to("cpu")

    a_mojo = a_bf16_cpu.to("mojo")
    b_mojo = b_bf16_cpu.to("mojo")
    bias_mojo = bias_bf16_cpu.to("mojo") if bias_bf16_cpu is not None else None
    mojo = measure(
        make_op(case, a_mojo, b_mojo, bias_mojo), mojo_synchronize, warmup, iterations
    )
    mojo_output = mojo.pop("output").to("cpu")
    mojo_synchronize()

    reference_cpu = reference.to("cpu")
    rocm_error = max_abs_error(rocm_output, reference_cpu)
    mojo_error = max_abs_error(mojo_output, reference_cpu)
    error_limit = 2 * rocm_error
    correctness_pass = mojo_error <= error_limit

    flop = 2 * case.m * case.n * case.k
    moved_bytes = 2 * (case.m * case.k + case.k * case.n + case.m * case.n)
    compute_bound_us = flop / MI300X_BF16_FLOPS * 1e6
    bandwidth_bound_us = moved_bytes / MI300X_HBM_BYTES * 1e6
    roofline_us = max(compute_bound_us, bandwidth_bound_us)

    result = {
        "case": case,
        "mojo": mojo,
        "rocm": rocm,
        "ratio": mojo["median_us"] / rocm["median_us"],
        "mojo_tflops": flop / (mojo["median_us"] * 1e-6) / 1e12,
        "rocm_tflops": flop / (rocm["median_us"] * 1e-6) / 1e12,
        "mojo_gbps": moved_bytes / (mojo["median_us"] * 1e-6) / 1e9,
        "rocm_gbps": moved_bytes / (rocm["median_us"] * 1e-6) / 1e9,
        "compute_bound_us": compute_bound_us,
        "bandwidth_bound_us": bandwidth_bound_us,
        "roofline_us": roofline_us,
        "roofline_bound": (
            "compute" if compute_bound_us >= bandwidth_bound_us else "bandwidth"
        ),
        "rocm_error": rocm_error,
        "mojo_error": mojo_error,
        "error_limit": error_limit,
        "correctness_pass": correctness_pass,
    }
    del (
        a_cpu,
        b_cpu,
        bias_cpu,
        a_fp32,
        b_fp32,
        bias_fp32,
        reference,
        reference_cpu,
        a_rocm,
        b_rocm,
        bias_rocm,
        rocm_output,
        a_mojo,
        b_mojo,
        bias_mojo,
        mojo_output,
    )
    torch.cuda.empty_cache()
    torch.cuda.synchronize()
    mojo_synchronize()
    return result


def write_table(path: Path, results: list[dict]):
    columns = [
        "case",
        "phase",
        "op",
        "m",
        "n",
        "k",
        "calls",
        "a_strides",
        "b_strides",
        "transpose_b",
        "bias",
        "profile_mojo_us",
        "profile_rocm_us",
        "mojo_median_us",
        "mojo_p10_us",
        "mojo_p90_us",
        "rocm_median_us",
        "rocm_p10_us",
        "rocm_p90_us",
        "mojo_over_rocm",
        "mojo_tflops",
        "rocm_tflops",
        "mojo_hbm_gbps",
        "rocm_hbm_gbps",
        "roofline_bound",
        "roofline_us",
        "compute_bound_us",
        "bandwidth_bound_us",
        "rocm_bf16_max_abs_error",
        "mojo_bf16_max_abs_error",
        "mojo_error_limit",
        "correctness_pass",
        "acceptance_pass",
    ]
    with path.open("w", newline="") as file:
        writer = csv.DictWriter(file, fieldnames=columns)
        writer.writeheader()
        for result in results:
            case = result["case"]
            a_strides = (case.k, 1)
            b_strides = (case.k, 1) if case.transpose_b else (case.n, 1)
            writer.writerow(
                {
                    "case": case.name,
                    "phase": case.phase,
                    "op": case.op,
                    "m": case.m,
                    "n": case.n,
                    "k": case.k,
                    "calls": case.calls,
                    "a_strides": repr(a_strides),
                    "b_strides": repr(b_strides),
                    "transpose_b": case.transpose_b,
                    "bias": case.bias,
                    "profile_mojo_us": f"{case.profile_mojo_us:.2f}",
                    "profile_rocm_us": f"{case.profile_rocm_us:.2f}",
                    "mojo_median_us": f"{result['mojo']['median_us']:.3f}",
                    "mojo_p10_us": f"{result['mojo']['p10_us']:.3f}",
                    "mojo_p90_us": f"{result['mojo']['p90_us']:.3f}",
                    "rocm_median_us": f"{result['rocm']['median_us']:.3f}",
                    "rocm_p10_us": f"{result['rocm']['p10_us']:.3f}",
                    "rocm_p90_us": f"{result['rocm']['p90_us']:.3f}",
                    "mojo_over_rocm": f"{result['ratio']:.3f}",
                    "mojo_tflops": f"{result['mojo_tflops']:.3f}",
                    "rocm_tflops": f"{result['rocm_tflops']:.3f}",
                    "mojo_hbm_gbps": f"{result['mojo_gbps']:.3f}",
                    "rocm_hbm_gbps": f"{result['rocm_gbps']:.3f}",
                    "roofline_bound": result["roofline_bound"],
                    "roofline_us": f"{result['roofline_us']:.3f}",
                    "compute_bound_us": f"{result['compute_bound_us']:.3f}",
                    "bandwidth_bound_us": f"{result['bandwidth_bound_us']:.3f}",
                    "rocm_bf16_max_abs_error": f"{result['rocm_error']:.8g}",
                    "mojo_bf16_max_abs_error": f"{result['mojo_error']:.8g}",
                    "mojo_error_limit": f"{result['error_limit']:.8g}",
                    "correctness_pass": result["correctness_pass"],
                    "acceptance_pass": result["correctness_pass"]
                    and result["ratio"] <= 1.15,
                }
            )


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--rocm-profile-dir", type=Path, default=Path("."))
    parser.add_argument("--mojo-profile-dir", type=Path, default=Path("mojo_profile"))
    parser.add_argument("--output", type=Path, default=Path("gemm_target_table.csv"))
    parser.add_argument("--warmup", type=int, default=25)
    parser.add_argument("--iterations", type=int, default=100)
    parser.add_argument("--phase", choices=["decode", "prefill", "all"], default="all")
    parser.add_argument("--include-ceiling", action="store_true")
    parser.add_argument("--case", action="append", default=[])
    return parser.parse_args()


def main():
    args = parse_args()
    if args.warmup < 25 or args.iterations < 100:
        raise ValueError("protocol requires >=25 warmups and >=100 iterations")
    if torch.version.hip is None or not torch.cuda.is_available():
        raise RuntimeError("a working ROCm PyTorch build is required")
    if "MI300X" not in torch.cuda.get_device_name(0):
        raise RuntimeError(f"expected MI300X, found {torch.cuda.get_device_name(0)}")

    register_mojo_devices()
    max_device = list(get_accelerators())[0]

    def mojo_synchronize():
        device_module.synchronize(0)

    cases = extract_cases(args.rocm_profile_dir, args.mojo_profile_dir)
    if args.phase != "all":
        cases = [case for case in cases if case.phase == args.phase]
    if args.case:
        cases = [
            case for case in cases if any(pattern in case.name for pattern in args.case)
        ]
    if args.include_ceiling:
        cases.append(
            GemmCase(
                phase="ceiling",
                op="mm",
                m=8192,
                n=8192,
                k=8192,
                calls=1,
                profile_mojo_us=0.0,
                profile_rocm_us=0.0,
                transpose_b=False,
                bias=False,
            )
        )

    print(
        f"Environment: torch={torch.__version__}, hip={torch.version.hip}, "
        f"device={torch.cuda.get_device_name(0)}, MAX_device={max_device}, "
        f"warmup={args.warmup}, iterations={args.iterations}"
    )
    rocm_smi_snapshot("before")
    results = []
    for index, case in enumerate(cases, start=1):
        print(
            f"\n[{index}/{len(cases)}] {case.name}: "
            f"A=({case.m},{case.k}), N={case.n}, "
            f"transpose_b={case.transpose_b}, bias={case.bias}",
            flush=True,
        )
        result = run_case(case, mojo_synchronize, args.warmup, args.iterations)
        results.append(result)
        print(
            f"  ROCm {result['rocm']['median_us']:.3f} us "
            f"(p10 {result['rocm']['p10_us']:.3f}, "
            f"p90 {result['rocm']['p90_us']:.3f}) | "
            f"Mojo {result['mojo']['median_us']:.3f} us "
            f"(p10 {result['mojo']['p10_us']:.3f}, "
            f"p90 {result['mojo']['p90_us']:.3f}) | "
            f"ratio {result['ratio']:.3f}x"
        )
        print(
            f"  TFLOPS ROCm {result['rocm_tflops']:.3f}, "
            f"Mojo {result['mojo_tflops']:.3f}; "
            f"correctness ROCm error {result['rocm_error']:.8g}, "
            f"Mojo error {result['mojo_error']:.8g}, "
            f"limit {result['error_limit']:.8g}: "
            f"{'PASS' if result['correctness_pass'] else 'FAIL'}"
        )
        write_table(args.output, results)
    rocm_smi_snapshot("after")
    write_table(args.output, results)
    if not all(result["correctness_pass"] for result in results):
        raise SystemExit("one or more Mojo correctness gates failed")


if __name__ == "__main__":
    main()
