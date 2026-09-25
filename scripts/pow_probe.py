"""fp32 pow ulp probe: mojo device vs a float64 reference, plus candidate
formulas evaluated on the CPU to identify which one the kernel matches, and a
streamed 16M-element timing. Run on an idle GPU."""

import time
from collections.abc import Callable

import numpy as np
import torch
from torch.testing._internal.common_methods_invocations import op_db

import torch_mojo_backend.native as n
from torch_mojo_backend import register_mojo_devices


def _dev() -> torch.device:
    return torch.device("mojo:0")  # only after register_mojo_devices()


f32, f64 = np.float32, np.float64
LOG2E = f32(1.4426950408889634)


def ulps(v32: np.ndarray, ref64: np.ndarray) -> np.ndarray:
    """Signed error of v32 vs ref64 in units of the fp32 ulp at ref."""
    ulp = np.spacing(np.abs(ref64.astype(f32))).astype(f64)
    return (v32.astype(f64) - ref64) / ulp


def summarize(name: str, v32: np.ndarray, ref64: np.ndarray, mask: np.ndarray):
    e = np.abs(ulps(v32, ref64))[mask]
    bins = [0, 0.5, 1, 2, 4, 8, 16, np.inf]
    h, _ = np.histogram(e, bins=bins)
    pct = " ".join(
        f"{a:g}-{b:g}:{c / len(e) * 100:.3f}%"
        for a, b, c in zip(bins[:-1], bins[1:], h)
    )
    print(
        f"{name:36s} max {e.max():8.3f} ulp  mean {e.mean():.3f}  "
        f"p99 {np.percentile(e, 99):.3f} | {pct}",
        flush=True,
    )


def match(name: str, cand32: np.ndarray, ours32: np.ndarray, mask: np.ndarray):
    d = np.abs(ulps(cand32[mask], ours32[mask].astype(f64)))
    same = np.mean(cand32[mask] == ours32[mask]) * 100
    print(
        f"  vs ours: {name:32s} bit-identical {same:6.2f}%  "
        f"|diff| max {d.max():.2f} ulp  mean {d.mean():.3f}  within 1 ulp {np.mean(d <= 1) * 100:.2f}%",
        flush=True,
    )


def cephes_logf(x: np.ndarray, ft: type = f32) -> np.ndarray:
    """Mojo stdlib `_log_base[27]` (Cephes logf coefficients), emulated without
    fma, in the working dtype `ft` (the stdlib keeps the same float
    coefficients for float64)."""
    m, e = np.frexp(x.astype(ft))
    e = e.astype(ft)
    small = m < ft(0.70710678118654752440)
    e = np.where(small, e - 1, e).astype(ft)
    x1 = (np.where(small, m + m, m) - ft(1)).astype(ft)
    x2 = (x1 * x1).astype(ft)
    x3 = (x2 * x1).astype(ft)
    c = [
        ft(v)
        for v in (
            3.3333331174e-1,
            -2.4999993993e-1,
            2.0000714765e-1,
            -1.6668057665e-1,
            1.4249322787e-1,
            -1.2420140846e-1,
            1.1676998740e-1,
            -1.1514610310e-1,
            7.0376836292e-2,
        )
    ]
    acc = np.full_like(x1, c[-1])
    for cv in reversed(c[:-1]):
        acc = (acc * x1 + cv).astype(ft)
    y = (acc * x3).astype(ft)
    y = (x1 + (x2 * ft(-0.5) + y)).astype(ft)
    return (e * ft(0.69314718055994530942) + y).astype(ft)


def run_sample(
    tag: str, x: np.ndarray, y: np.ndarray
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    x = np.ascontiguousarray(x, dtype=f32)
    y = np.ascontiguousarray(y, dtype=f32)
    ref = np.power(x.astype(f64), y.astype(f64))
    ours = (
        torch.pow(torch.from_numpy(x).to(_dev()), torch.from_numpy(y).to(_dev()))
        .cpu()
        .numpy()
    )
    cpu = torch.pow(torch.from_numpy(x), torch.from_numpy(y)).numpy()
    finite = np.isfinite(ref) & np.isfinite(ours) & (ref != 0)
    print(f"--- {tag}: n={ref.size}, finite {finite.sum()}", flush=True)
    summarize("ours (mojo)", ours, ref, finite)
    summarize("torch cpu", cpu, ref, finite)
    xb0, yb0 = np.broadcast_arrays(x, y)
    e_all = np.where(finite, np.abs(ulps(ours, ref)), 0.0).ravel()
    for j in np.argsort(-e_all)[:3]:
        if e_all[j] > 0.5:
            print(
                f"  worst ours: x={xb0.ravel()[j]!r} y={yb0.ravel()[j]!r} ours={ours.ravel()[j]!r} "
                f"cpu={cpu.ravel()[j]!r} ref={ref.ravel()[j]!r} ({e_all[j]:.2f} ulp)",
                flush=True,
            )
    with np.errstate(all="ignore"):
        xb, yb = np.broadcast_arrays(x, y)
        cands = (
            ("A exp(f32 y*log x)", np.exp((yb * np.log(xb)).astype(f32)).astype(f32)),
            (
                "B exp2(f32 y*log2 x)",
                np.exp2((yb * np.log2(xb)).astype(f32)).astype(f32),
            ),
            (
                "C exp2(f32(y*cephes_logf x)*log2e)",
                np.exp2(
                    ((yb * cephes_logf(xb)).astype(f32) * LOG2E).astype(f32)
                ).astype(f32),
            ),
            (
                "D f32(exp64(y*log64 x))",
                np.exp(yb.astype(f64) * np.log(xb.astype(f64))).astype(f32),
            ),
            (
                "E f32(exp64(y*cephes_logf64 x)) [stdlib f64 pow]",
                np.exp(yb.astype(f64) * cephes_logf(xb.astype(f64), f64)).astype(f32),
            ),
        )
    for nm, c in cands:
        summarize("  cand " + nm, c, ref, finite)
        match(nm, c, ours, finite)
    return ours, cpu, ref


def time_us(
    fn: Callable[[], object], sync: Callable[[], object], iters: int = 20, reps: int = 5
) -> float:
    for _ in range(3):
        fn()
    sync()
    best = float("inf")
    for _ in range(reps):
        sync()
        t = time.perf_counter()
        for _ in range(iters):
            fn()
        sync()
        best = min(best, (time.perf_counter() - t) / iters * 1e6)
    return best


def main() -> int:
    register_mojo_devices()
    print("MOJO_ROOT", n._MOJO_ROOT, "torch", torch.__version__, flush=True)
    # 1. Random sample x in [0.5, 50], y in [0, 6].
    rng = np.random.default_rng(0)
    N = 1 << 20
    run_sample(
        "random x in [0.5,50], y in [0,6]",
        rng.uniform(0.5, 50, N),
        rng.uniform(0, 6, N),
    )
    run_sample(
        "stress x in [0.5,50], y in [-20,20]",
        rng.uniform(0.5, 50, N),
        rng.uniform(-20, 20, N),
    )
    run_sample(
        "stress x in [1,2], y in [0,120]", rng.uniform(1, 2, N), rng.uniform(0, 120, N)
    )
    run_sample(
        "stress x in [1e-3,1e3], y in [-5,5]",
        10 ** rng.uniform(-3, 3, N),
        rng.uniform(-5, 5, N),
    )

    # 2. __rpow__: scalar ** tensor
    xs = rng.uniform(0, 6, N).astype(f32)
    ref = np.power(f64(2.5), xs.astype(f64))
    ours = (
        (torch.tensor(2.5).to(_dev()) ** torch.from_numpy(xs).to(_dev())).cpu().numpy()
    )
    cpu = (2.5 ** torch.from_numpy(xs)).numpy()
    m = np.isfinite(ref)
    print(
        "--- __rpow__ tensor(2.5) ** x, x in [0, 6]  (pow.Scalar_out is not registered: 0-d tensor base)"
    )
    summarize("ours (mojo)", ours, ref, m)
    summarize("torch cpu", cpu, ref, m)

    # 3. Tensor_Scalar: x ** 2.5
    xs = rng.uniform(0.5, 50, N).astype(f32)
    ref = np.power(xs.astype(f64), 2.5)
    ours = (torch.from_numpy(xs).to(_dev()) ** 2.5).cpu().numpy()
    cpu = (torch.from_numpy(xs) ** 2.5).numpy()
    print("--- Tensor_Scalar x ** 2.5, x in [0.5, 50]")
    summarize("ours (mojo)", ours, ref, m)
    summarize("torch cpu", cpu, ref, m)

    # 4. The OpInfo samples the conformance node uses (pow, float32, tensor exponent).
    op = next(o for o in op_db if o.name == "pow" and o.variant_test_name == "")
    torch.manual_seed(0)
    for i, s in enumerate(op.sample_inputs("cpu", torch.float32, requires_grad=False)):
        if not (
            isinstance(s.input, torch.Tensor)
            and s.args
            and isinstance(s.args[0], torch.Tensor)
        ):
            continue
        if s.args[0].dtype != torch.float32 or s.input.dtype != torch.float32:
            continue
        x, y = s.input.numpy(), s.args[0].numpy()
        if x.size == 0 or y.size == 0:
            continue
        ours, cpu, ref = run_sample(
            f"opinfo sample {i} {tuple(x.shape)} ** {tuple(y.shape)}", x, y
        )
        fin = np.isfinite(ref) & (ref != 0)
        rel_o = np.abs(ours.astype(f64) - ref) / np.abs(ref)
        rel_c = np.abs(cpu.astype(f64) - ref) / np.abs(ref)
        print(
            f"  max rel vs f64: ours {rel_o[fin].max():.3e} ({(rel_o[fin] > 1.3e-6).sum()} > rtol 1.3e-6), "
            f"torch cpu {rel_c[fin].max():.3e}",
            flush=True,
        )

    # 5. Streamed device-time proxy on 16M elements (synchronize, burst, synchronize).
    sync = torch.mojo.synchronize  # ty: ignore[unresolved-attribute]

    M = 1 << 24
    xt = (torch.rand(M) * 49.5 + 0.5).to(_dev())
    yt = (torch.rand(M) * 6).to(_dev())
    base = torch.tensor(2.5).to(_dev())
    for name, fn, nbytes in (
        ("pow(T,T) fractional y", lambda: torch.pow(xt, yt), 12 * M),
        ("pow(T, 2.5)", lambda: torch.pow(xt, 2.5), 8 * M),
        ("pow(T, 2.0)", lambda: torch.pow(xt, 2.0), 8 * M),
        ("tensor(2.5) ** T", lambda: torch.pow(base, yt), 8 * M),
    ):
        us = time_us(fn, sync)
        print(
            f"timing 16M f32 {name:24s} {us:9.1f} us/iter  {nbytes / us / 1e3:7.0f} GB/s",
            flush=True,
        )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
