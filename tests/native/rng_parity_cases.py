"""The RNG parity cases: one function per draw, run on CUDA and on the mojo
device from the same seed, compared bit for bit (`test_random_parity.py`).

Below the grid cap (`sm_count * max_threads_per_sm / 256` blocks of 256
threads, 270 336 elements on an H100 SXM) ATen's launch is the same on every
card; above it (the 300 000+ element cases) the draw depends on the GPU
model, which the live oracle shares.
"""

import torch

SEED = 20260914

_F = torch.float32
_H = torch.float16
_B = torch.bfloat16
_D = torch.float64


def _view_storage(device, dtype=_F):
    """A [:, ::2] view and the storage it writes into."""
    storage = torch.zeros(4, 8, device=device, dtype=dtype)
    return storage[:, ::2], storage


def _fill_view(device, dtype, op):
    view, storage = _view_storage(device, dtype)
    op(view)
    return storage


def _bytes(*tensors):
    """Exact bytes of several tensors of any dtype, as one uint8 tensor."""
    parts = []
    for t in tensors:
        flat = t.detach().cpu().contiguous().reshape(-1)
        parts.append(
            flat.to(torch.uint8) if flat.dtype == torch.bool else flat.view(torch.uint8)
        )
    return torch.cat(parts)


def _dropout_input(shape, dtype, device):
    n = torch.Size(shape).numel()
    # Bounded so half/bfloat16 stay finite; exactly representable pattern.
    return (
        ((torch.arange(n, device=device, dtype=_F) % 251 - 125) / 8)
        .reshape(shape)
        .to(dtype)
    )


def _dropout(shape, dtype, p, transform=None):
    def fn(device):
        x = _dropout_input(shape, dtype, device)
        if transform is not None:
            x = transform(x)
        out, mask = torch.ops.aten.native_dropout.default(x, p, True)
        return _bytes(out, mask)

    return fn


def _dropout_backward(shape, dtype, p):
    def fn(device):
        x = _dropout_input(shape, dtype, device)
        _, mask = torch.ops.aten.native_dropout.default(x, p, True)
        grad = torch.ops.aten.native_dropout_backward.default(x, mask, 1.0 / (1.0 - p))
        return _bytes(grad)

    return fn


def _sequence(device):
    """Interleaved ops share one stream."""
    a = torch.rand(100, device=device)
    b = torch.randn(37, device=device)
    c = torch.randint(0, 10, (50,), device=device)
    e = torch.empty(64, device=device).bernoulli_(0.4)
    f, _ = torch.ops.aten.native_dropout.default(
        torch.ones(33, device=device), 0.5, True
    )
    g = torch.rand(9, device=device, dtype=_D)
    return torch.cat([a, b, c.float(), e, f, g.float()])


def _linspace(device, n, dtype=_F):
    return torch.linspace(0, 1, n).to(device).to(dtype)


def _gen(device, seed=777):
    return torch.Generator(device=device).manual_seed(seed)


CASES = {
    # uniform_ / rand
    "rand_f32_1": lambda d: torch.rand(1, device=d),
    "rand_f32_16": lambda d: torch.rand(16, device=d),
    "rand_f32_255": lambda d: torch.rand(255, device=d),
    "rand_f32_1000": lambda d: torch.rand(1000, device=d),
    "rand_f32_300000": lambda d: torch.rand(300_000, device=d),
    "rand_f32_2000000": lambda d: torch.rand(2_000_000, device=d),
    "rand_f16_1000": lambda d: torch.rand(1000, device=d, dtype=_H),
    "rand_bf16_1000": lambda d: torch.rand(1000, device=d, dtype=_B),
    "rand_f64_1000": lambda d: torch.rand(1000, device=d, dtype=_D),
    "rand_f16_2000000": lambda d: torch.rand(2_000_000, device=d, dtype=_H),
    "rand_f64_2000000": lambda d: torch.rand(2_000_000, device=d, dtype=_D),
    "uniform_f32_range": lambda d: torch.empty(1000, device=d).uniform_(-2.5, 7.25),
    "uniform_bf16_range": lambda d: torch.empty(4096, device=d, dtype=_B).uniform_(
        0.0, 1.0
    ),
    "uniform_f32_transposed": lambda d: (
        torch.zeros(8, 5, device=d).t().uniform_(-1.0, 2.0)
    ),
    "uniform_f32_strided_view": lambda d: _fill_view(d, _F, lambda v: v.uniform_()),
    "uniform_f32_offset_view": lambda d: (lambda s: (s[1:].uniform_(), s)[1])(
        torch.zeros(9, device=d)
    ),
    "uniform_f32_3d_permuted": lambda d: (
        torch.zeros(3, 4, 5, device=d).permute(2, 0, 1).uniform_()
    ),
    "uniform_f32_scalar": lambda d: torch.zeros((), device=d).uniform_(),
    "uniform_f32_generator": lambda d: torch.empty(1000, device=d).uniform_(
        generator=_gen(d)
    ),
    "uniform_f32_generator_twice": lambda d: (
        lambda g: torch.cat(
            [
                torch.empty(300, device=d).uniform_(generator=g),
                torch.empty(300, device=d).uniform_(generator=g),
            ]
        )
    )(_gen(d)),
    "uniform_f16_range": lambda d: torch.empty(1000, device=d, dtype=_H).uniform_(
        0.1, 1.0
    ),
    "uniform_bf16_wide_range": lambda d: torch.empty(1000, device=d, dtype=_B).uniform_(
        -2.5, 7.25
    ),
    "uniform_f32_from_equals_to": lambda d: torch.empty(64, device=d).uniform_(
        2.5, 2.5
    ),
    "uniform_f16_from_equals_to": lambda d: torch.empty(
        64, device=d, dtype=_H
    ).uniform_(0.3, 0.3),
    "uniform_f32_empty": lambda d: torch.empty(0, 5, device=d).uniform_(),
    # normal_ / randn
    "randn_f32_1": lambda d: torch.randn(1, device=d),
    "randn_f32_16": lambda d: torch.randn(16, device=d),
    "randn_f32_255": lambda d: torch.randn(255, device=d),
    "randn_f32_1000": lambda d: torch.randn(1000, device=d),
    "randn_f32_300000": lambda d: torch.randn(300_000, device=d),
    "randn_f32_2000000": lambda d: torch.randn(2_000_000, device=d),
    "randn_f16_1000": lambda d: torch.randn(1000, device=d, dtype=_H),
    "randn_bf16_1000": lambda d: torch.randn(1000, device=d, dtype=_B),
    "randn_f64_1000": lambda d: torch.randn(1000, device=d, dtype=_D),
    "normal_f32_mean_std": lambda d: torch.empty(1000, device=d).normal_(2.0, 3.0),
    "normal_f64_mean_std": lambda d: torch.empty(1000, device=d, dtype=_D).normal_(
        -1.0, 0.5
    ),
    "normal_f32_strided_view": lambda d: _fill_view(d, _F, lambda v: v.normal_()),
    "normal_f32_generator": lambda d: torch.empty(1000, device=d).normal_(
        generator=_gen(d)
    ),
    "torch_normal_tensor_mean": lambda d: torch.normal(torch.arange(10.0).to(d), 1.0),
    "torch_normal_float_tensor": lambda d: torch.normal(
        2.0, torch.arange(1.0, 9.0).to(d).to(_H)
    ),
    "torch_normal_tensor_tensor": lambda d: torch.normal(
        torch.arange(6.0).to(d).view(6, 1), torch.arange(1.0, 4.0).to(d)
    ),
    "randn_like_f16": lambda d: torch.randn_like(torch.zeros(7, 9, device=d, dtype=_H)),
    "normal_f32_std_zero": lambda d: torch.empty(100, device=d).normal_(1.5, 0.0),
    "normal_f32_generator_twice": lambda d: (
        lambda g: torch.cat(
            [
                torch.empty(300, device=d).normal_(generator=g),
                torch.empty(300, device=d).normal_(generator=g),
            ]
        )
    )(_gen(d)),
    "torch_normal_tensor_float_out": lambda d: torch.normal(
        torch.arange(10.0).to(d), 0.5, out=torch.empty(10, device=d)
    ),
    "torch_normal_float_tensor_out": lambda d: torch.normal(
        1.0, torch.arange(1.0, 9.0).to(d), out=torch.empty(8, device=d)
    ),
    "torch_normal_tensor_tensor_out": lambda d: torch.normal(
        torch.arange(6.0).to(d).view(6, 1),
        torch.arange(1.0, 4.0).to(d),
        out=torch.empty(0, device=d),
    ),
    "torch_normal_tensor_float_out_transposed": lambda d: torch.normal(
        torch.arange(12.0).to(d).view(3, 4), 1.0, out=torch.empty(4, 3, device=d).t()
    ),
    # random_ / randint
    "randint_i64_3_9": lambda d: torch.randint(3, 9, (200,), device=d),
    "randint_i64_large_range": lambda d: torch.randint(
        -(2**40), 2**40, (1000,), device=d
    ),
    "randint_i32_2p28": lambda d: torch.randint(
        0, 2**28, (1000,), device=d, dtype=torch.int32
    ),
    "randint_i32_2p28m1": lambda d: torch.randint(
        0, 2**28 - 1, (1000,), device=d, dtype=torch.int32
    ),
    "randint_f32": lambda d: torch.randint(0, 100, (1000,), device=d, dtype=_F),
    "randint_f16": lambda d: torch.randint(-50, 50, (1000,), device=d, dtype=_H),
    "randint_bf16": lambda d: torch.randint(0, 7, (1000,), device=d, dtype=_B),
    "randint_u8": lambda d: torch.randint(0, 256, (1000,), device=d, dtype=torch.uint8),
    "randint_i8": lambda d: torch.randint(
        -128, 128, (1000,), device=d, dtype=torch.int8
    ),
    "randint_i16": lambda d: torch.randint(
        -1000, 1000, (1000,), device=d, dtype=torch.int16
    ),
    "randint_bool": lambda d: torch.randint(0, 2, (1000,), device=d, dtype=torch.bool),
    "randint_f64_big": lambda d: torch.randint(0, 2**50, (1000,), device=d, dtype=_D),
    "random_to_i32": lambda d: torch.empty(64, dtype=torch.int32, device=d).random_(5),
    "random_from_none_i64": lambda d: torch.empty(
        64, dtype=torch.int64, device=d
    ).random_(-5, None),
    "random_from_none_f32": lambda d: torch.empty(64, dtype=_F, device=d).random_(
        3, None
    ),
    "random_full64_i64": lambda d: torch.empty(64, dtype=torch.int64, device=d).random_(
        -(2**63), None
    ),
    "random_full64_f32": lambda d: torch.empty(64, dtype=_F, device=d).random_(
        -(2**63), None
    ),
    "random_plain_u8": lambda d: torch.empty(
        1000, dtype=torch.uint8, device=d
    ).random_(),
    "random_plain_i64": lambda d: torch.empty(
        1000, dtype=torch.int64, device=d
    ).random_(),
    "random_plain_i32": lambda d: torch.empty(
        1000, dtype=torch.int32, device=d
    ).random_(),
    "random_plain_f32": lambda d: torch.empty(1000, dtype=_F, device=d).random_(),
    "random_plain_f64": lambda d: torch.empty(1000, dtype=_D, device=d).random_(),
    "random_plain_f16": lambda d: torch.empty(1000, dtype=_H, device=d).random_(),
    "random_plain_bf16": lambda d: torch.empty(1000, dtype=_B, device=d).random_(),
    "random_plain_bool": lambda d: torch.empty(
        1000, dtype=torch.bool, device=d
    ).random_(),
    "randint_strided": lambda d: (lambda s: (s.random_(1, 3), s)[1])(
        torch.zeros(4, 6, dtype=torch.int64, device=d).t()
    ),
    "randint_generator": lambda d: torch.randint(
        0, 1000, (1000,), device=d, generator=_gen(d, 5)
    ),
    "randint_f32_bound_adjust": lambda d: torch.randint(
        2**24 - 3, 2**24 + 10, (1000,), device=d, dtype=_F
    ),
    "randint_f16_bound_adjust": lambda d: torch.randint(
        2040, 2060, (1000,), device=d, dtype=_H
    ),
    "random_i64_full_endpoints": lambda d: torch.empty(
        64, dtype=torch.int64, device=d
    ).random_(-(2**63), 2**63 - 1),
    "random_i64_max_to_none": lambda d: torch.empty(
        64, dtype=torch.int64, device=d
    ).random_(2**63 - 2, None),
    "randint_u16": lambda d: torch.randint(
        0, 60000, (1000,), device=d, dtype=torch.uint16
    ),
    "randint_u32": lambda d: torch.randint(
        0, 2**32 - 1, (1000,), device=d, dtype=torch.uint32
    ),
    "randint_u64": lambda d: torch.randint(
        0, 2**62, (1000,), device=d, dtype=torch.uint64
    ),
    "randint_empty": lambda d: torch.randint(0, 10, (0,), device=d),
    # bernoulli
    "bernoulli_f32_p03": lambda d: torch.empty(1000, device=d).bernoulli_(0.3),
    "bernoulli_f64_p03": lambda d: torch.empty(1000, device=d, dtype=_D).bernoulli_(
        0.3
    ),
    "bernoulli_i64_p05": lambda d: torch.empty(
        1000, device=d, dtype=torch.int64
    ).bernoulli_(0.5),
    "bernoulli_bool_p07": lambda d: torch.empty(
        1000, device=d, dtype=torch.bool
    ).bernoulli_(0.7),
    "bernoulli_f16_p01": lambda d: torch.empty(1000, device=d, dtype=_H).bernoulli_(
        0.1
    ),
    "bernoulli_tensor_f32": lambda d: torch.bernoulli(_linspace(d, 1000)),
    "bernoulli_tensor_f64": lambda d: torch.bernoulli(_linspace(d, 1000, _D)),
    "bernoulli_tensor_broadcast": lambda d: torch.empty(5, 7, device=d).bernoulli_(
        _linspace(d, 7).view(1, 7)
    ),
    "bernoulli_tensor_p_f16_self_i32": lambda d: torch.empty(
        300, device=d, dtype=torch.int32
    ).bernoulli_(_linspace(d, 300, _H)),
    "bernoulli_tensor_p_transposed": lambda d: torch.empty(6, 4, device=d).bernoulli_(
        _linspace(d, 24).reshape(4, 6).t()
    ),
    "bernoulli_tensor_big": lambda d: torch.bernoulli(
        torch.full((300_000,), 0.5, device=d)
    ),
    "bernoulli_f32_p0": lambda d: torch.empty(1000, device=d).bernoulli_(0.0),
    "bernoulli_f32_p1": lambda d: torch.empty(1000, device=d).bernoulli_(1.0),
    "bernoulli_tensor_both_transposed_f16_p": lambda d: (
        torch.empty(4, 6, device=d)
        .t()
        .bernoulli_(_linspace(d, 24, _H).reshape(4, 6).t())
    ),
    "bernoulli_tensor_both_transposed_f32_p": lambda d: (
        torch.empty(4, 6, device=d).t().bernoulli_(_linspace(d, 24).reshape(4, 6).t())
    ),
    "bernoulli_tensor_overlapping_route": lambda d: torch.empty_strided(
        (3, 2), (2, 3), device=d
    ).bernoulli_(torch.full((), 0.5, device=d)),
    "bernoulli_tensor_scalar_self": lambda d: torch.empty((), device=d).bernoulli_(
        torch.full((), 0.5, device=d)
    ),
    "bernoulli_tensor_empty": lambda d: torch.empty(0, device=d).bernoulli_(
        torch.empty(0, device=d)
    ),
    # other continuous distributions
    "exponential_f32": lambda d: torch.empty(1000, device=d).exponential_(1.5),
    "exponential_f64": lambda d: torch.empty(1000, device=d, dtype=_D).exponential_(
        0.7
    ),
    "exponential_f16": lambda d: torch.empty(1000, device=d, dtype=_H).exponential_(),
    "geometric_f32": lambda d: torch.empty(1000, device=d).geometric_(0.3),
    "geometric_i64": lambda d: torch.empty(
        1000, device=d, dtype=torch.int64
    ).geometric_(0.05),
    "geometric_f64": lambda d: torch.empty(1000, device=d, dtype=_D).geometric_(0.5),
    "cauchy_f32": lambda d: torch.empty(1000, device=d).cauchy_(0.5, 2.0),
    "cauchy_f64": lambda d: torch.empty(1000, device=d, dtype=_D).cauchy_(),
    "log_normal_f32": lambda d: torch.empty(1000, device=d).log_normal_(1.0, 0.5),
    "log_normal_f64": lambda d: torch.empty(1000, device=d, dtype=_D).log_normal_(),
    # native_dropout
    "dropout_f32_1000": _dropout((1000,), _F, 0.2),
    "dropout_f32_1001_vec1": _dropout((1001,), _F, 0.2),
    "dropout_f32_1002_vec2": _dropout((1002,), _F, 0.2),
    "dropout_f16_1024_vec8": _dropout((1024,), _H, 0.3),
    "dropout_bf16_2052_vec4": _dropout((2052,), _B, 0.3),
    "dropout_f64_500": _dropout((500,), _D, 0.5),
    "dropout_f32_transposed": _dropout((16, 8), _F, 0.2, lambda x: x.t()),
    "dropout_f32_nondense": _dropout((16, 8), _F, 0.2, lambda x: x[:, ::2]),
    "dropout_f32_offset": _dropout((17,), _F, 0.2, lambda x: x[1:]),
    "dropout_f32_big": _dropout((3_000_000,), _F, 0.1),
    "dropout_f16_big": _dropout((3_000_000,), _H, 0.1),
    "dropout_f32_p0": _dropout((1000,), _F, 0.0),
    "dropout_f32_big_vec2": _dropout((300_002,), _F, 0.2),
    "dropout_f32_big_strided": _dropout((1000, 700), _F, 0.2, lambda x: x[:, ::2]),
    "dropout_backward_f32": _dropout_backward((1000,), _F, 0.2),
    "dropout_backward_f16": _dropout_backward((1024,), _H, 0.3),
    "dropout_backward_f64": _dropout_backward((500,), _D, 0.5),
    "sequence": _sequence,
}

# 2 GB + 4 bytes: crosses TensorIterator's 32-bit indexing limit and splits.
# Above the grid cap too; opt in with TMB_RNG_BIG=1 (~5 GB of device memory
# per side, ~10 GB on the one GPU).
BIG_CASES = {
    "rand_f32_split_2gb": lambda d: torch.rand(2**29 + 1, device=d),
    "randn_f32_split_2gb": lambda d: torch.randn(2**29 + 1, device=d),
}
