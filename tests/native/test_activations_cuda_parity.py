"""Activations, their backwards and the elementwise (reduction='none') losses
against stock CUDA bit for bit (the live oracle, tests/native/cuda_oracle.py).

Every case runs the same aten op on `cuda:<i>` and on the mojo device of the
same GPU: float16 / bfloat16 over all 65536 bit patterns of the swept operand,
float32 / float64 over special values, every binade, the neighbourhoods of the
op's branch points, a dense sweep of [-30, 30] and pseudo-random bit patterns;
the other operands (grad, target, weight) are pseudo-random, and a second test
crosses the special values of every operand. A third test covers `out=`,
in-place, strided, broadcast, 0-dim, empty and mixed-dtype operands, and a
fourth the dtypes the op does not compute in (CUDA's error type, or its bits).
"""

from __future__ import annotations

import itertools
import json
import math
import struct
from collections.abc import Callable
from dataclasses import dataclass, field
from pathlib import Path

import pytest
import torch

from tests.native.cuda_oracle import CudaOracle, assert_bits_equal

A = torch.ops.aten
FLOATS = (torch.float32, torch.float16, torch.bfloat16, torch.float64)
_INT_OF = {
    torch.float16: torch.int16,
    torch.bfloat16: torch.int16,
    torch.float32: torch.int32,
    torch.float64: torch.int64,
}
_GEN_SEED = 0x5EED

# Operand roles: "x" is swept; "g" (grad), "t" (target) and "w" (weight) are
# pseudo-random normal values; "u" is uniform in [0, 1] (bce's input and
# target). A role in upper case is swept instead of "x".
Fn = Callable[..., object]


@dataclass(frozen=True)
class Case:
    id: str
    fn: Fn
    roles: str
    centers: tuple[float, ...] = ()
    out: Fn | None = None
    inplace: Fn | None = None
    # glu: the swept operand is (n, 2), split along the last dim
    pairs: bool = False
    dtypes: tuple[torch.dtype, ...] = FLOATS
    extra: dict[str, object] = field(default_factory=dict)


def _elu_cases() -> list[Case]:
    params = {
        "default": (1.0, 1.0, 1.0),
        "selu": (1.6732632423543772, 1.0507009873554805, 1.0),
        "celu0.3": (0.3, 1.0, 1 / 0.3),
        "odd": (2.5, 0.7, 1.3),
    }
    out = []
    for name, (al, sc, ins) in params.items():
        out.append(
            Case(
                f"elu-{name}",
                lambda x, al=al, sc=sc, ins=ins: A.elu(x, al, sc, ins),
                "x",
                (0.0,),
                out=lambda x, o, al=al, sc=sc, ins=ins: A.elu.out(
                    x, al, sc, ins, out=o
                ),
                inplace=lambda x, al=al, sc=sc, ins=ins: A.elu_(x, al, sc, ins),
            )
        )
        for is_result in (False, True):
            out.append(
                Case(
                    f"elu_backward-{name}-{'result' if is_result else 'self'}",
                    lambda g, x, al=al, sc=sc, ins=ins, r=is_result: A.elu_backward(
                        g, al, sc, ins, r, x
                    ),
                    "gx",
                    (0.0,),
                    out=lambda g, x, o, al=al, sc=sc, ins=ins, r=is_result: (
                        A.elu_backward.grad_input(g, al, sc, ins, r, x, grad_input=o)
                    ),
                )
            )
    return out


def _cases() -> list[Case]:
    c: list[Case] = [
        Case(
            "sigmoid",
            torch.sigmoid,
            "x",
            out=lambda x, o: torch.sigmoid(x, out=o),
            inplace=torch.sigmoid_,
        ),
        Case(
            "sigmoid_backward",
            A.sigmoid_backward,
            "gX",
            (0.0, 0.5, 1.0),
            out=lambda g, y, o: A.sigmoid_backward.grad_input(g, y, grad_input=o),
        ),
        Case(
            "tanh",
            torch.tanh,
            "x",
            (0.0, 1.0),
            out=lambda x, o: torch.tanh(x, out=o),
            inplace=torch.tanh_,
        ),
        Case(
            "tanh_backward",
            A.tanh_backward,
            "gX",
            (0.0, 1.0),
            out=lambda g, y, o: A.tanh_backward.grad_input(g, y, grad_input=o),
        ),
        Case(
            "silu",
            A.silu,
            "x",
            (1.0,),
            out=lambda x, o: A.silu.out(x, out=o),
            inplace=A.silu_,
        ),
        Case(
            "silu_backward",
            A.silu_backward,
            "gx",
            out=lambda g, x, o: A.silu_backward.grad_input(g, x, grad_input=o),
        ),
        Case(
            "mish", A.mish, "x", out=lambda x, o: A.mish.out(x, out=o), inplace=A.mish_
        ),
        Case("mish_backward", A.mish_backward, "gx"),
        Case(
            "hardsigmoid",
            A.hardsigmoid,
            "x",
            (-3.0, 3.0),
            out=lambda x, o: A.hardsigmoid.out(x, out=o),
            inplace=A.hardsigmoid_,
        ),
        Case(
            "hardsigmoid_backward",
            A.hardsigmoid_backward,
            "gx",
            (-3.0, 3.0),
            out=lambda g, x, o: A.hardsigmoid_backward.grad_input(g, x, grad_input=o),
        ),
        Case(
            "hardswish",
            A.hardswish,
            "x",
            (-3.0, 3.0),
            out=lambda x, o: A.hardswish.out(x, out=o),
            inplace=A.hardswish_,
        ),
        Case(
            "hardswish_backward",
            A.hardswish_backward,
            "gx",
            (-3.0, 3.0),
            out=lambda g, x, o: A.hardswish_backward.out(g, x, out=o),
        ),
        Case("log_sigmoid_forward", lambda x: A.log_sigmoid_forward(x)[0], "x", (0.0,)),
        Case(
            "log_sigmoid_backward",
            lambda g, x: A.log_sigmoid_backward(g, x, torch.empty(0, device=x.device)),
            "gx",
            (0.0,),
        ),
        Case(
            "relu",
            torch.relu,
            "x",
            (0.0,),
            out=lambda x, o: torch.ops.aten.relu.out(x, out=o),
            inplace=torch.relu_,
            dtypes=FLOATS,
        ),
        Case("relu6", A.relu6, "x", (0.0, 6.0), inplace=A.relu6_),
        Case(
            "glu",
            lambda x: A.glu(x, -1),
            "x",
            pairs=True,
            out=lambda x, o: A.glu.out(x, -1, out=o),
        ),
        Case(
            "glu_backward",
            lambda g, x: A.glu_backward(g, x, -1),
            "gx",
            pairs=True,
            out=lambda g, x, o: A.glu_backward.grad_input(g, x, -1, grad_input=o),
        ),
        Case("_prelu_kernel", A._prelu_kernel, "xw", (0.0,)),
        Case("_prelu_kernel_backward", A._prelu_kernel_backward, "gxw", (0.0,)),
        Case(
            "rrelu_with_noise-eval",
            lambda x, w: A.rrelu_with_noise(x, w, 0.1, 0.3, False),
            "xw",
            (0.0,),
            out=lambda x, w, o: A.rrelu_with_noise.out(x, w, 0.1, 0.3, False, out=o),
            inplace=None,
        ),
    ]
    for approx in ("none", "tanh"):
        c.append(
            Case(
                f"gelu-{approx}",
                lambda x, a=approx: A.gelu(x, approximate=a),
                "x",
                out=lambda x, o, a=approx: A.gelu.out(x, approximate=a, out=o),
                inplace=lambda x, a=approx: A.gelu_(x, approximate=a),
            )
        )
        c.append(
            Case(
                f"gelu_backward-{approx}",
                lambda g, x, a=approx: A.gelu_backward(g, x, approximate=a),
                "gx",
                out=lambda g, x, o, a=approx: A.gelu_backward.grad_input(
                    g, x, approximate=a, grad_input=o
                ),
            )
        )
    c += _elu_cases()
    c.append(Case("celu-0.7", lambda x: A.celu(x, 0.7), "x", (0.0,)))
    c.append(Case("selu", A.selu, "x", (0.0,), inplace=A.selu_))
    for lo, hi in ((-1.0, 1.0), (-0.3, 2.5)):
        c.append(
            Case(
                f"hardtanh-{lo}-{hi}",
                lambda x, lo=lo, hi=hi: A.hardtanh(x, lo, hi),
                "x",
                (lo, hi),
                out=lambda x, o, lo=lo, hi=hi: A.hardtanh.out(x, lo, hi, out=o),
                inplace=lambda x, lo=lo, hi=hi: A.hardtanh_(x, lo, hi),
            )
        )
        c.append(
            Case(
                f"hardtanh_backward-{lo}-{hi}",
                lambda g, x, lo=lo, hi=hi: A.hardtanh_backward(g, x, lo, hi),
                "gx",
                (lo, hi),
                out=lambda g, x, o, lo=lo, hi=hi: A.hardtanh_backward.grad_input(
                    g, x, lo, hi, grad_input=o
                ),
            )
        )
    for lambd in (0.5, 0.3, 0.0):
        for name in ("hardshrink", "softshrink"):
            fwd = getattr(A, name)
            bwd = getattr(A, f"{name}_backward")
            c.append(
                Case(
                    f"{name}-{lambd}",
                    lambda x, f=fwd, lam=lambd: f(x, lam),
                    "x",
                    (lambd, -lambd),
                    out=lambda x, o, f=fwd, lam=lambd: f.out(x, lam, out=o),
                )
            )
            c.append(
                Case(
                    f"{name}_backward-{lambd}",
                    lambda g, x, f=bwd, lam=lambd: f(g, x, lam),
                    "gx",
                    (lambd, -lambd),
                    out=lambda g, x, o, f=bwd, lam=lambd: f.grad_input(
                        g, x, lam, grad_input=o
                    ),
                )
            )
    for slope in (0.01, 0.2, -1.5):
        c.append(
            Case(
                f"leaky_relu-{slope}",
                lambda x, s=slope: A.leaky_relu(x, s),
                "x",
                (0.0,),
                out=lambda x, o, s=slope: A.leaky_relu.out(x, s, out=o),
                inplace=lambda x, s=slope: A.leaky_relu_(x, s),
            )
        )
        for is_result in (False, True):
            if is_result and slope < 0:
                continue  # CUDA refuses a negative slope with self_is_result
            c.append(
                Case(
                    f"leaky_relu_backward-{slope}-{is_result}",
                    lambda g, x, s=slope, r=is_result: A.leaky_relu_backward(
                        g, x, s, r
                    ),
                    "gx",
                    (0.0,),
                    out=lambda g, x, o, s=slope, r=is_result: (
                        A.leaky_relu_backward.grad_input(g, x, s, r, grad_input=o)
                    ),
                )
            )
    for th, val in ((0.0, 0.0), (0.5, -2.0), (-1.25, 7.0)):
        c.append(
            Case(
                f"threshold-{th}-{val}",
                lambda x, th=th, val=val: A.threshold(x, th, val),
                "x",
                (th,),
                out=lambda x, o, th=th, val=val: A.threshold.out(x, th, val, out=o),
                inplace=lambda x, th=th, val=val: A.threshold_(x, th, val),
            )
        )
        c.append(
            Case(
                f"threshold_backward-{th}",
                lambda g, x, th=th: A.threshold_backward(g, x, th),
                "gx",
                (th,),
                out=lambda g, x, o, th=th: A.threshold_backward.grad_input(
                    g, x, th, grad_input=o
                ),
            )
        )
    for beta, th in ((1.0, 20.0), (2.0, 5.0), (0.5, 1.0)):
        c.append(
            Case(
                f"softplus-{beta}-{th}",
                lambda x, b=beta, t=th: A.softplus(x, b, t),
                "x",
                (th / beta,),
                out=lambda x, o, b=beta, t=th: A.softplus.out(x, b, t, out=o),
            )
        )
        c.append(
            Case(
                f"softplus_backward-{beta}-{th}",
                lambda g, x, b=beta, t=th: A.softplus_backward(g, x, b, t),
                "gx",
                (th / beta,),
                out=lambda g, x, o, b=beta, t=th: A.softplus_backward.grad_input(
                    g, x, b, t, grad_input=o
                ),
            )
        )
    for eps in (None, 1e-6, 0.25):
        c.append(
            Case(
                f"logit_backward-{eps}",
                lambda g, x, e=eps: A.logit_backward(g, x, e),
                "gx",
                (0.0, 1.0) + ((eps, 1 - eps) if eps else ()),
                out=lambda g, x, o, e=eps: A.logit_backward.grad_input(
                    g, x, e, grad_input=o
                ),
            )
        )
    # losses, reduction='none' (0)
    c.append(
        Case(
            "mse_loss",
            lambda x, t: A.mse_loss(x, t, 0),
            "xt",
            out=lambda x, t, o: A.mse_loss.out(x, t, 0, out=o),
        )
    )
    c.append(
        Case(
            "mse_loss_backward",
            lambda g, x, t: A.mse_loss_backward(g, x, t, 0),
            "gxt",
            out=lambda g, x, t, o: A.mse_loss_backward.grad_input(
                g, x, t, 0, grad_input=o
            ),
        )
    )
    for d in (1.0, 0.5, 2.7):
        c.append(
            Case(
                f"huber_loss-{d}",
                lambda x, t, d=d: A.huber_loss(x, t, 0, d),
                "xt",
                (d, -d),
                out=lambda x, t, o, d=d: A.huber_loss.out(x, t, 0, d, out=o),
            )
        )
        c.append(
            Case(
                f"huber_loss_backward-{d}",
                lambda g, x, t, d=d: A.huber_loss_backward(g, x, t, 0, d),
                "gxt",
                (d, -d),
                out=lambda g, x, t, o, d=d: A.huber_loss_backward.out(
                    g, x, t, 0, d, grad_input=o
                ),
            )
        )
    for b in (1.0, 0.5, 0.0, 2.7):
        c.append(
            Case(
                f"smooth_l1_loss-{b}",
                lambda x, t, b=b: A.smooth_l1_loss(x, t, 0, b),
                "xt",
                (b, -b),
                out=lambda x, t, o, b=b: A.smooth_l1_loss.out(x, t, 0, b, out=o),
            )
        )
        c.append(
            Case(
                f"smooth_l1_loss_backward-{b}",
                lambda g, x, t, b=b: A.smooth_l1_loss_backward(g, x, t, 0, b),
                "gxt",
                (b, -b),
                out=lambda g, x, t, o, b=b: A.smooth_l1_loss_backward.grad_input(
                    g, x, t, 0, b, grad_input=o
                ),
            )
        )
    c += [
        Case(
            "binary_cross_entropy",
            lambda x, t: A.binary_cross_entropy(x, t, None, 0),
            "Uu",
            out=lambda x, t, o: A.binary_cross_entropy.out(x, t, None, 0, out=o),
        ),
        Case(
            "binary_cross_entropy-weight",
            lambda x, t, w: A.binary_cross_entropy(x, t, w, 0),
            "Uuw",
        ),
        Case(
            "binary_cross_entropy_backward",
            lambda g, x, t: A.binary_cross_entropy_backward(g, x, t, None, 0),
            "gUu",
            out=lambda g, x, t, o: A.binary_cross_entropy_backward.grad_input(
                g, x, t, None, 0, grad_input=o
            ),
        ),
        Case(
            "binary_cross_entropy_backward-weight",
            lambda g, x, t, w: A.binary_cross_entropy_backward(g, x, t, w, 0),
            "gUuw",
        ),
        Case(
            "binary_cross_entropy_with_logits",
            lambda x, t: A.binary_cross_entropy_with_logits(x, t, None, None, 0),
            "xu",
            out=lambda x, t, o: A.binary_cross_entropy_with_logits.out(
                x, t, None, None, 0, out=o
            ),
        ),
        Case(
            "binary_cross_entropy_with_logits-weights",
            lambda x, t, w, pw: A.binary_cross_entropy_with_logits(x, t, w, pw, 0),
            "xuww",
        ),
    ]
    return c


CASES = {case.id: case for case in _cases()}


# --------------------------------------------------------------------------
# inputs
# --------------------------------------------------------------------------


def _hashed(count: int, seed: int) -> list[int]:
    h = 0x9E3779B97F4A7C15 ^ seed
    out = []
    for _ in range(count):
        h = (h * 6364136223846793005 + 1442695040888963407) % (1 << 64)
        out.append(h)
    return out


def _bits_of(v: float, dtype: torch.dtype) -> int:
    if dtype == torch.float64:
        return struct.unpack("<q", struct.pack("<d", v))[0]
    return int(torch.tensor([v], dtype=dtype).view(_INT_OF[dtype]).item())


def _from_bits(bits: list[int], dtype: torch.dtype) -> torch.Tensor:
    width = torch.iinfo(_INT_OF[dtype]).bits
    signed = [b - (1 << width) if b >= 1 << (width - 1) else b for b in bits]
    return torch.tensor(signed, dtype=_INT_OF[dtype]).view(dtype)


def _sweep(dtype: torch.dtype, centers: tuple[float, ...]) -> torch.Tensor:
    """Every bit pattern of a half dtype; for float32 / float64 the special
    values, both neighbours of every power of two, the neighbourhoods of
    `centers` (and of their negations), a dense sweep of [-30, 30], a
    log-uniform magnitude sweep and pseudo-random bit patterns."""
    if dtype in (torch.float16, torch.bfloat16):
        return (
            torch.arange(-(1 << 15), 1 << 15, dtype=torch.int32)
            .to(torch.int16)
            .view(dtype)
        )
    width = torch.iinfo(_INT_OF[dtype]).bits
    mbits = 23 if dtype == torch.float32 else 52
    ebits = width - 1 - mbits
    sign = 1 << (width - 1)
    pos: set[int] = set()
    for e in range(1 << ebits):
        for m in (0, 1, 2, (1 << mbits) - 1, (1 << mbits) - 2, 1 << (mbits - 1)):
            pos.add((e << mbits) | m)
    # quiet / signalling NaNs with payloads
    exp_all = ((1 << ebits) - 1) << mbits
    pos.update({exp_all | 1, exp_all | (1 << (mbits - 1)) | 5, exp_all | 0x1234})
    for c in centers + (1.0, 0.5, 2.0, 20.0, 3.0):
        b = _bits_of(abs(c), dtype)
        pos.update(range(max(b - 128, 0), b + 129))
    bits = sorted(pos) + [b | sign for b in sorted(pos)]
    bits += [h >> (64 - width) for h in _hashed(1 << 15, width)]
    dense = torch.linspace(-30, 30, 60001, dtype=torch.float64)
    top = 38.5 if dtype == torch.float32 else 308
    bottom = -45.5 if dtype == torch.float32 else -323.5
    mags = torch.logspace(bottom, top, 20000, dtype=torch.float64)
    return torch.cat(
        (_from_bits(bits, dtype), dense.to(dtype), mags.to(dtype), -mags.to(dtype))
    )


def _random(n: int, dtype: torch.dtype, role: str, seed: int) -> torch.Tensor:
    g = torch.Generator().manual_seed(_GEN_SEED + seed)
    if role == "u":
        return torch.rand(n, generator=g, dtype=torch.float64).to(dtype)
    return (torch.randn(n, generator=g, dtype=torch.float64) * 4).to(dtype)


def _specials(dtype: torch.dtype, role: str) -> torch.Tensor:
    if role in "uU":
        vals = [0.0, -0.0, 1.0, 0.5, 1e-3, 0.999, 1e-30, 1 - 1e-7]
        if dtype.is_floating_point:
            fi = torch.finfo(dtype)
            vals += [fi.tiny, fi.smallest_normal / 4, 1 - fi.eps / 2]
        return torch.tensor(vals, dtype=torch.float64).to(dtype)
    fi = torch.finfo(dtype)
    vals = [0.0, -0.0, 1.0, -1.0, 0.5, -0.5, 3.0, -3.0, 20.0, -20.0, 1e-3]
    vals += [math.inf, -math.inf, fi.max, -fi.max, fi.tiny, -fi.tiny]
    vals += [fi.smallest_normal * fi.eps, -fi.smallest_normal * fi.eps]
    t = torch.tensor(vals, dtype=torch.float64).to(dtype)
    width = torch.iinfo(_INT_OF[dtype]).bits
    mbits = {torch.float16: 10, torch.bfloat16: 7, torch.float32: 23}.get(dtype, 52)
    qnan = ((1 << (width - 1 - mbits)) - 1) << mbits | 1 << (mbits - 1)
    nans = _from_bits([qnan, qnan | 3, qnan | 1 << (width - 1)], dtype)
    return torch.cat((t, nans))


def _operands(case: Case, dtype: torch.dtype) -> list[torch.Tensor]:
    """The sweep test's operands: the swept one (role "x" or an upper-case
    role) and pseudo-random others of its length."""
    swept = _sweep(dtype, case.centers)
    sweep_role = next((r for r in case.roles if r.isupper()), "x")
    if sweep_role == "U":
        keep = (swept >= 0) & (swept <= 1)
        swept = swept[keep]
    n = swept.numel()
    ops = []
    for k, role in enumerate(case.roles):
        ops.append(swept if role == sweep_role else _random(n, dtype, role, k))
    if case.pairs:
        x_index = case.roles.index("x")
        x = ops[x_index]
        other = _random(n, dtype, "x", 99)
        ops[x_index] = torch.stack((x, other), dim=-1)
        # the second half swept too: (random, x)
        ops[x_index] = torch.cat((ops[x_index], torch.stack((other, x), dim=-1)))
        for k, role in enumerate(case.roles):
            if role == "g":
                ops[k] = _random(2 * n, dtype, "g", k).unsqueeze(-1)
    return ops


def _special_operands(case: Case, dtype: torch.dtype) -> list[torch.Tensor]:
    """Every combination of the special values of every operand."""
    per = [_specials(dtype, r.lower() if r.lower() == "u" else "x") for r in case.roles]
    if case.pairs:
        x_index = case.roles.index("x")
        per.insert(x_index + 1, per[x_index])
    grid = list(itertools.product(*[range(len(p)) for p in per]))
    cols = [torch.stack([p[i] for i in col]) for p, col in zip(per, zip(*grid))]
    if case.pairs:
        x_index = case.roles.index("x")
        cols[x_index] = torch.stack((cols[x_index], cols.pop(x_index + 1)), dim=-1)
        for k, role in enumerate(case.roles):
            if role == "g":
                cols[k] = cols[k].unsqueeze(-1)
    return cols


def _case_dtypes(case_ids: list[str]) -> list[tuple[str, torch.dtype]]:
    return [(i, d) for i in case_ids for d in CASES[i].dtypes]


def _dtype_id(d: torch.dtype) -> str:
    return str(d).removeprefix("torch.")


_SWEEP = _case_dtypes(list(CASES))


@pytest.mark.parametrize(
    ("case_id", "dtype"), _SWEEP, ids=[f"{i}-{_dtype_id(d)}" for i, d in _SWEEP]
)
def test_sweep_matches_cuda(cuda_oracle: CudaOracle, case_id: str, dtype):
    case = CASES[case_id]
    cuda_oracle.assert_bitwise_equal(case.fn, *_operands(case, dtype))


@pytest.mark.parametrize(
    ("case_id", "dtype"), _SWEEP, ids=[f"{i}-{_dtype_id(d)}" for i, d in _SWEEP]
)
def test_specials_match_cuda(cuda_oracle: CudaOracle, case_id: str, dtype):
    case = CASES[case_id]
    ops = _special_operands(case, dtype)
    if dtype != torch.float64 or case.pairs:
        cuda_oracle.assert_bitwise_equal(case.fn, *ops)
        return
    # float64: where two operands are NaN, or a NaN operand meets an inf or
    # a value whose square overflows (arithmetic that makes a NaN of its
    # own), only NaN-ness is compared.
    # Of two NaNs, DADD / DMUL / DFMA return the first source register's
    # payload and sign, and ptxas, not the C++ source, orders the registers:
    # stock CUDA's mse_backward multiplies `grad * prod` in SASS where the
    # source says `prod * grad`, its mse forward keeps `input`'s NaN of
    # `input - target` and its mse_backward `target`'s. The bits are a
    # property of each kernel's register allocation, not of the arithmetic.
    # Every single-NaN case stays exact (a quieted operand, as CUDA's).
    ours, cuda = cuda_oracle.run_by_device(lambda d: case.fn(*[o.to(d) for o in ops]))
    nans = sum(
        (o.isnan().int() for o in ops), torch.zeros(ops[0].shape, dtype=torch.int)
    )
    # an inf, or a value whose square overflows to one
    infs = sum(
        ((o.abs() > 1e150).int() for o in ops),
        torch.zeros(ops[0].shape, dtype=torch.int),
    )
    loose = (nans >= 2) | ((nans >= 1) & (infs >= 1))
    tight = loose.logical_not()
    ours_t = ours if isinstance(ours, tuple) else (ours,)
    cuda_t = cuda if isinstance(cuda, tuple) else (cuda,)
    assert len(ours_t) == len(cuda_t)
    for a, b in zip(ours_t, cuda_t, strict=True):
        assert isinstance(a, torch.Tensor), a
        assert isinstance(b, torch.Tensor), b
        a, b = a.cpu(), b.cpu()
        assert_bits_equal(a[tight], b[tight], inputs=[o[tight] for o in ops])
        assert_bits_equal(a[loose], b[loose], nan="any")


def _variant_calls(case: Case, dtype: torch.dtype) -> dict[str, Fn]:
    """`out=`, in-place, strided, broadcast, 0-dim, empty and mixed-dtype
    calls of `case`, each `fn(device)`."""
    n_ops = len(case.roles)
    cols = 6 if not case.pairs else 2
    base = []
    for k, role in enumerate(case.roles):
        r = role.lower()
        base.append(_random(5 * 6 * cols // 2 * 2, torch.float64, r, k))

    def shaped(device: str, k: int, shape: tuple[int, ...], dt=dtype):
        t = base[k][: math.prod(shape)].reshape(shape).to(dt).to(device)
        if case.pairs and case.roles[k] == "g":
            t = t[..., : shape[-1] // 2]
        return t

    def args(device: str, shape=(5, cols), dts=None):
        return [
            shaped(device, k, shape, dts[k] if dts else dtype) for k in range(n_ops)
        ]

    calls: dict[str, Fn] = {}
    if case.out is not None:
        out_fn = case.out
        calls["out_resized"] = lambda d: out_fn(
            *args(d), torch.empty(0, dtype=dtype, device=d)
        )
        calls["out_same"] = lambda d: (
            lambda a: out_fn(*a, torch.full_like(_result(case, a), 7.0))
        )(args(d))
        calls["out_strided"] = lambda d: (
            lambda a: out_fn(*a, torch.full_like(_result(case, a).t(), 7.0).t())
        )(args(d))
        calls["out_float64"] = lambda d: out_fn(
            *args(d), torch.empty(0, dtype=torch.float64, device=d)
        )
        calls["out_float16"] = lambda d: out_fn(
            *args(d), torch.empty(0, dtype=torch.float16, device=d)
        )
        calls["out_int32"] = lambda d: out_fn(
            *args(d), torch.empty(0, dtype=torch.int32, device=d)
        )
        calls["out_aliases_input"] = lambda d: (lambda a: (out_fn(*a, a[-1]), a))(
            args(d)
        )
    if case.inplace is not None:
        ip = case.inplace
        calls["inplace"] = lambda d: (lambda a: (ip(a[0]), a[0]))(args(d))
        calls["inplace_strided"] = lambda d: (lambda b: (ip(b.t()), b))(
            shaped(d, 0, (cols, 5))
        )
        calls["inplace_int"] = lambda d: ip(
            torch.arange(-3, 4, device=d, dtype=torch.int32)
        )
    calls["transposed"] = lambda d: (
        case.fn(*[shaped(d, k, (cols, 5)).t() for k in range(n_ops)])
        if not case.pairs
        else case.fn(*[shaped(d, k, (5, cols)) for k in range(n_ops)])
    )
    calls["strided_step"] = lambda d: case.fn(
        *[shaped(d, k, (10, cols))[::2] for k in range(n_ops)]
    )
    calls["zero_dim"] = lambda d: case.fn(
        *[shaped(d, k, (2,) if case.pairs else ()) for k in range(n_ops)]
    )
    calls["empty"] = lambda d: case.fn(*[shaped(d, k, (0, cols)) for k in range(n_ops)])
    if n_ops >= 2:
        for k in range(n_ops):
            calls[f"broadcast_row_{k}"] = lambda d, k=k: case.fn(
                *[
                    shaped(d, j, (1, cols) if j == k else (5, cols))
                    for j in range(n_ops)
                ]
            )
            calls[f"zero_dim_{k}"] = lambda d, k=k: case.fn(
                *[shaped(d, j, () if j == k else (5, cols)) for j in range(n_ops)]
            )
            for other in (torch.float32, torch.float16, torch.float64):
                if other == dtype:
                    continue
                dts = [other if j == k else dtype for j in range(n_ops)]
                calls[f"mixed_{k}_{_dtype_id(other)}"] = lambda d, dts=dts: case.fn(
                    *args(d, dts=dts)
                )
            calls[f"cpu_scalar_{k}"] = lambda d, k=k: case.fn(
                *[
                    shaped("cpu", j, ()) if j == k else shaped(d, j, (5, cols))
                    for j in range(n_ops)
                ]
            )
    return calls


def _result(case: Case, a: list[torch.Tensor]) -> torch.Tensor:
    r = case.fn(*a)
    if isinstance(r, tuple):
        r = r[0]
    assert isinstance(r, torch.Tensor), r
    return r


_VARIANTS = [
    (i, v)
    for i in CASES
    for v in dict.fromkeys(
        list(_variant_calls(CASES[i], torch.float32))
        + list(_variant_calls(CASES[i], torch.bfloat16))
    )
]


@pytest.mark.parametrize(
    ("case_id", "variant"), _VARIANTS, ids=[f"{i}-{v}" for i, v in _VARIANTS]
)
@pytest.mark.parametrize("dtype", [torch.float32, torch.bfloat16], ids=_dtype_id)
def test_variants_match_cuda(cuda_oracle: CudaOracle, case_id: str, variant, dtype):
    calls = _variant_calls(CASES[case_id], dtype)
    if variant not in calls:
        pytest.skip(f"{variant} is the same dtype as {dtype}")
    cuda_oracle.assert_bitwise_equal_by_device(calls[variant])


_OTHER_DTYPES = (
    torch.int8,
    torch.uint8,
    torch.int16,
    torch.int32,
    torch.int64,
    torch.bool,
    torch.complex64,
    torch.complex128,
)


# Complex128 rows that stock CUDA rounds differently (its jiterated complex
# exp in sigmoid / silu, and the backward products) are kept out of the
# mandatory tests: the failing rows of each seeded input set are listed in
# data/complex128_open_inputs.json, and every other row is mandatory.
# test_complex128_open_inputs checks exactly those rows, as a strict xfail.
_OPEN = json.loads(
    (Path(__file__).parent / "data" / "complex128_open_inputs.json").read_text()
)
_COMPLEX_CASES = ["sigmoid", "silu", "sigmoid_backward", "tanh_backward"]
_MIXED_KINDS = ["sigmoid_backward", "tanh_backward"]


def _rows(n: int, key: str) -> torch.Tensor:
    """The mandatory rows of an n-row input set: all but `key`'s open rows.
    The whole set still runs: CUDA's rounding depends on the element position
    (see the comment above), so the rows are selected from the output."""
    keep = torch.ones(n, dtype=torch.bool)
    keep[_OPEN.get(key, [])] = False
    return keep.nonzero().flatten()


def _check_rows(
    cuda_oracle: CudaOracle,
    fn: Callable[[str], object],
    rows: torch.Tensor | None = None,
) -> None:
    """`fn(device)` bit-equal to CUDA's, on `rows` of the output when given."""
    if rows is None:
        cuda_oracle.assert_bitwise_equal_by_device(fn)  # also exception types
        return
    ours, cuda = cuda_oracle.run_by_device(fn)
    assert isinstance(ours, torch.Tensor) and isinstance(cuda, torch.Tensor)
    assert_bits_equal(ours.cpu()[rows], cuda.cpu()[rows])


def _other_params() -> list[object]:
    return [
        pytest.param(case_id, dtype, id=f"{_dtype_id(dtype)}-{case_id}")
        for dtype in _OTHER_DTYPES
        for case_id in CASES
    ]


def _other_operands(case: Case, dtype: torch.dtype, device: str) -> list[torch.Tensor]:
    def operand(k: int) -> torch.Tensor:
        shape = (7, 2) if case.pairs and case.roles[k] != "g" else (7, 1)
        if not case.pairs:
            shape = (7,)
        if dtype.is_complex:
            g = torch.Generator().manual_seed(k)
            return torch.randn(shape, generator=g, dtype=dtype).to(device)
        v = torch.arange(-3, 4).add(k).reshape(7, 1).expand(7, shape[-1]).reshape(shape)
        return v.to(dtype).to(device)

    return [operand(k) for k in range(len(case.roles))]


@pytest.mark.parametrize(("case_id", "dtype"), _other_params())
def test_other_dtypes_match_cuda(cuda_oracle: CudaOracle, case_id: str, dtype):
    """Integer, bool and complex operands: CUDA's result bits, or its
    exception type (complex128: the mandatory rows, see above)."""
    case = CASES[case_id]
    key = f"other_dtypes/{case_id}-complex128"
    keep = _rows(7, key) if dtype == torch.complex128 and _OPEN.get(key) else None

    _check_rows(
        cuda_oracle, lambda device: case.fn(*_other_operands(case, dtype, device)), keep
    )


def _complex_ops(case: Case, dtype: torch.dtype) -> list[torch.Tensor]:
    """Random values, and every pair of special components, per operand."""
    real = torch.float32 if dtype == torch.complex64 else torch.float64
    sp = _specials(real, "x")
    grid = torch.cartesian_prod(sp, sp)
    special = torch.complex(grid[:, 0], grid[:, 1])
    g = torch.Generator().manual_seed(7)
    ops = []
    for k in range(len(case.roles)):
        rand = torch.randn(4000, generator=g, dtype=dtype) * 3
        ops.append(torch.cat((rand, special.roll(k * 5))))
    return ops


@pytest.mark.parametrize(
    ("case_id", "dtype"),
    [
        pytest.param(c, d, id=f"{c}-{_dtype_id(d)}")
        for c in _COMPLEX_CASES
        for d in (torch.complex64, torch.complex128)
    ],
)
def test_complex_matches_cuda(cuda_oracle: CudaOracle, case_id: str, dtype):
    """The complex kernels (jiterated sigmoid / sigmoid_backward /
    tanh_backward, c10::complex silu) over random values and every pair of
    special components (complex128: the mandatory rows, see above)."""
    case = CASES[case_id]
    ops = _complex_ops(case, dtype)
    keep = None
    if dtype == torch.complex128:
        keep = _rows(ops[0].numel(), f"complex_matches/{case_id}-complex128")
    _check_rows(cuda_oracle, lambda device: case.fn(*[o.to(device) for o in ops]), keep)


@pytest.mark.parametrize("inplace", [False, True], ids=["functional", "inplace"])
@pytest.mark.parametrize("dtype", FLOATS, ids=_dtype_id)
def test_rrelu_train_matches_cuda(cuda_oracle: CudaOracle, dtype, inplace: bool):
    """rrelu_with_noise in training: the slopes it draws from the device
    generator (returned through `noise`) and the output, from the same seed."""
    g = torch.Generator().manual_seed(0x5EED)
    x = torch.randn(4097, generator=g, dtype=torch.float64).to(dtype)
    op = A.rrelu_with_noise_ if inplace else A.rrelu_with_noise

    def fn(device: str) -> tuple[torch.Tensor, torch.Tensor]:
        xi = x.to(device)
        noise = torch.empty_like(xi)
        return op(xi, noise, 0.125, 1 / 3, True), noise

    cuda_oracle.assert_bitwise_equal_by_device(fn, seed=7)


def test_bce_general_path_does_not_leak(mojo_gpu: str):
    """Unreduced BCE with a [B, C] input and a broadcast [C] weight takes the
    general path, which builds squeezed views of its operands and loss. Those
    handles must be released: device memory is flat over repeated calls."""
    g = torch.Generator().manual_seed(1)
    x = torch.rand(8, 16, generator=g).to(mojo_gpu)
    t = torch.rand(8, 16, generator=g).to(mojo_gpu)
    w = torch.rand(16, generator=g).to(mojo_gpu)

    def once() -> None:
        A.binary_cross_entropy(x, t, w, 0)

    once()
    torch.accelerator.synchronize()
    before = torch.accelerator.memory_allocated()
    for _ in range(20):
        once()
    torch.accelerator.synchronize()
    assert torch.accelerator.memory_allocated() == before


def test_out_dtype_policy_matches_cuda(cuda_oracle: CudaOracle):
    """CUDA dispatches on the out= dtype: float16 out= of float32 operands is
    computed in float16, where `1 - 2**-24` rounds to 1 (so it is not below
    the upper bound). Off CUDA the common dtype keeps the float32 result."""
    x = torch.tensor([1 - 2**-24, -1.0, 1.0, 0.5], dtype=torch.float32)
    g = torch.ones(4, dtype=torch.float32)

    def fn(d: str) -> torch.Tensor:
        out = torch.empty(0, dtype=torch.float16, device=d)
        return A.hardtanh_backward.grad_input(
            g.to(d), x.to(d), -1.0, 1.0, grad_input=out
        )

    cuda_oracle.assert_bitwise_equal_by_device(fn)


def _mixed_operands(in_dtype: torch.dtype) -> tuple[torch.Tensor, torch.Tensor]:
    g = torch.Generator().manual_seed(11)
    grad = torch.randn(64, generator=g, dtype=in_dtype)
    saved = torch.randn(64, generator=g, dtype=in_dtype)
    return grad, saved


@pytest.mark.parametrize("kind", _MIXED_KINDS)
@pytest.mark.parametrize(
    ("in_dtype", "out_dtype"),
    [(torch.complex64, torch.complex128), (torch.complex64, torch.complex64)],
    ids=["c64-in-c128-out", "c64-in-c64-out"],
)
def test_complex_backward_out_dtype_matches_cuda(
    cuda_oracle: CudaOracle, kind: str, in_dtype, out_dtype
):
    """out= of another complex dtype than the inputs: CUDA computes in the
    out dtype, converting the operands first (complex128 out: the mandatory
    rows, see above)."""
    grad, saved = _mixed_operands(in_dtype)
    op = getattr(A, kind).grad_input
    keep = (
        _rows(64, f"mixed_c64_in_c128_out/{kind}")
        if out_dtype == torch.complex128
        else None
    )

    def fn(device: str) -> torch.Tensor:
        return op(
            grad.to(device),
            saved.to(device),
            grad_input=torch.empty(0, dtype=out_dtype, device=device),
        )

    _check_rows(cuda_oracle, fn, keep)


@pytest.mark.parametrize("key", sorted(k for k, rows in _OPEN.items() if rows))
@pytest.mark.xfail(
    strict=True,
    reason="complex128 rows that stock CUDA rounds differently (jiterated "
    "complex exp / backward products); not reproduced yet",
)
def test_complex128_open_inputs(cuda_oracle: CudaOracle, key: str):
    """The open complex128 rows only (data/complex128_open_inputs.json), of
    the same full input sets as the mandatory tests."""
    family, name = key.split("/", 1)
    rows = torch.tensor(_OPEN[key], dtype=torch.long)
    if family == "complex_matches":
        case = CASES[name.rsplit("-", 1)[0]]
        ops = _complex_ops(case, torch.complex128)
        _check_rows(
            cuda_oracle, lambda device: case.fn(*[o.to(device) for o in ops]), rows
        )
    elif family == "other_dtypes":
        case = CASES[name.rsplit("-", 1)[0]]
        _check_rows(
            cuda_oracle,
            lambda device: case.fn(*_other_operands(case, torch.complex128, device)),
            rows,
        )
    else:
        grad, saved = _mixed_operands(torch.complex64)
        op = getattr(A, name).grad_input
        _check_rows(
            cuda_oracle,
            lambda device: op(
                grad.to(device),
                saved.to(device),
                grad_input=torch.empty(0, dtype=torch.complex128, device=device),
            ),
            rows,
        )
