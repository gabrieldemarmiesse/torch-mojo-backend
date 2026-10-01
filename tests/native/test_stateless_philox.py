"""The stateless Philox ops on the mojo device: `_philox_key_split`,
`_philox_key_fold_in` (both overloads), `_philox_uniform_`, `_philox_normal_`
(torch >= 2.13, the keys behind `torch.func`'s functional random API).

Bit-exactness against stock CUDA is checked two ways: sha256 digests recorded
from stock CUDA (an H100, torch 2.14.0+cu130) and, when this process has a
CUDA build of torch that knows the ops, a live CUDA run. The key ops and the
uniform draw are integer / IEEE arithmetic and match on every GPU; the normal
draw uses CUDA's fast `__logf` / `__sincosf`, so it is only claimed on NVIDIA.

Torch before 2.13 has no schema for these ops; this module defines upstream's
(native_functions.yaml) so the kernels are exercised on every torch the
backend supports.
"""

import hashlib
from collections.abc import Callable

import pytest
import torch

from torch_mojo_backend import get_accelerators

_SCHEMAS = (
    "_philox_key_split(Tensor key, int num_splits) -> Tensor",
    "_philox_key_fold_in(Tensor key, int data) -> Tensor",
    "_philox_key_fold_in.Tensor(Tensor key, Tensor data) -> Tensor",
    "_philox_normal_(Tensor(a!) self, Tensor key, float mean=0, float std=1)"
    " -> Tensor(a!)",
    "_philox_uniform_(Tensor(a!) self, Tensor key, float low=0, float high=1)"
    " -> Tensor(a!)",
)
_UPSTREAM = hasattr(torch.ops.aten, "_philox_key_split")
_LIBRARY: list[torch.library.Library] = []
if not _UPSTREAM:
    _LIBRARY.append(torch.library.Library("aten", "FRAGMENT"))
    for _schema in _SCHEMAS:
        _LIBRARY[0].define(_schema)

A = torch.ops.aten


def _keys() -> torch.Tensor:
    """(3, 4, 2) uint64 keys with the high bits set somewhere."""
    k = torch.arange(24, dtype=torch.int64).reshape(3, 4, 2)
    k = k * 0x1E3779B97F4A7C15 + 0x0123456789ABCDEF  # wraps: every bit used
    k[0, 0, 0] = -1
    return k.view(torch.uint64)


def _draw(
    op: Callable[..., torch.Tensor],
    shape: tuple[int, ...],
    dtype: torch.dtype,
    key: torch.Tensor,
    *params: float,
) -> torch.Tensor:
    out = torch.empty(shape, dtype=dtype, device=key.device)
    op(out, key, *params)
    return out


def _cases(device: str) -> dict[str, torch.Tensor]:
    k = _keys().to(device)
    cases = {
        "split5": A._philox_key_split(k, 5),
        "split_one_key": A._philox_key_split(k[0, 0], 3),
        "fold_neg": A._philox_key_fold_in(k, -7),
        "fold_tensor": A._philox_key_fold_in.Tensor(
            k, torch.tensor([2**63 + 5], dtype=torch.uint64, device=device)
        ),
    }
    keyed = {
        "one_key_1000": ((1000,), k[0, 0]),
        "one_key_7": ((7,), k[0, 1]),
        "per_element": ((3, 4), k),
        "per_row_13": ((3, 13), k[:, :1]),
        "per_row_3d": ((3, 4, 5), k[:, :1, None]),
        "per_pair": ((3, 4, 6), k[:, :, None]),
        "broadcast_rows": ((3, 4, 5), k[:1, :, None]),
    }
    dtypes = [torch.float32, torch.float16, torch.bfloat16]
    if not device.startswith("mojo") or get_accelerators()[0].api != "metal":
        dtypes.append(torch.float64)
    for dtype in dtypes:
        for name, (shape, key) in keyed.items():
            tag = f"{name}_{str(dtype).removeprefix('torch.')}"
            cases[f"uniform_{tag}"] = _draw(
                A._philox_uniform_, shape, dtype, key, -2.3, 5.1
            )
            cases[f"normal_{tag}"] = _draw(
                A._philox_normal_, shape, dtype, key, 0.7, 3.3
            )
            cases[f"stdnormal_{tag}"] = _draw(A._philox_normal_, shape, dtype, key)
    return cases


def _digest(t: torch.Tensor) -> str:
    t = t.cpu().contiguous()
    h = hashlib.sha256(f"{t.dtype}{tuple(t.shape)}".encode())
    h.update(t.view(torch.uint8).numpy().tobytes())
    return h.hexdigest()[:16]


def _claimed(name: str) -> bool:
    """Normal draws are bit-exact on NVIDIA only (CUDA's fast intrinsics)."""
    return "normal" not in name or get_accelerators()[0].api == "cuda"


# sha256[:16] of each case on stock CUDA (H100, torch 2.14.0+cu130).
_GOLDEN = {
    "split5": "9ffdce0e676ff00b",
    "split_one_key": "9b525649e99f2298",
    "fold_neg": "7dceda1408759043",
    "fold_tensor": "e98672ad7d851119",
    "uniform_one_key_1000_float32": "93305a185911a754",
    "normal_one_key_1000_float32": "db58e2cd270caea4",
    "stdnormal_one_key_1000_float32": "80071b261ea9778d",
    "uniform_one_key_7_float32": "a99423ddf923d8c9",
    "normal_one_key_7_float32": "56807023ed6c498a",
    "stdnormal_one_key_7_float32": "7e91050a43de6822",
    "uniform_per_element_float32": "45fea07aa6e9ec04",
    "normal_per_element_float32": "8fbc8dfbfd2b2c8c",
    "stdnormal_per_element_float32": "46c2af189f13201d",
    "uniform_per_row_13_float32": "8ab6962ace47b2eb",
    "normal_per_row_13_float32": "504d7d85d7e199cd",
    "stdnormal_per_row_13_float32": "4a9b6a0253641165",
    "uniform_per_row_3d_float32": "2ad5cf749c9e513c",
    "normal_per_row_3d_float32": "959a96e35be8b321",
    "stdnormal_per_row_3d_float32": "22ae5dc4387d3b2c",
    "uniform_per_pair_float32": "f0a4cd3a512f4225",
    "normal_per_pair_float32": "4afc8a8e8448ce6d",
    "stdnormal_per_pair_float32": "4a9355095f2d97ac",
    "uniform_broadcast_rows_float32": "1e69ba0dff0c3822",
    "normal_broadcast_rows_float32": "4c5fce458841938e",
    "stdnormal_broadcast_rows_float32": "e431e4de8b0f7a8d",
    "uniform_one_key_1000_float16": "192aed151bfe821b",
    "normal_one_key_1000_float16": "c49bfea7924d9165",
    "stdnormal_one_key_1000_float16": "b12c3761a7bc16d5",
    "uniform_one_key_7_float16": "54ac64d44d94823e",
    "normal_one_key_7_float16": "6e967e7664003f0e",
    "stdnormal_one_key_7_float16": "9dc0852598c3b8ed",
    "uniform_per_element_float16": "07e1ce4e1d1426a9",
    "normal_per_element_float16": "6b24e9d3806bfcea",
    "stdnormal_per_element_float16": "3ccb03d2f07a5333",
    "uniform_per_row_13_float16": "378fb6ee501b9d14",
    "normal_per_row_13_float16": "70ea886f4113427f",
    "stdnormal_per_row_13_float16": "519f66a8f5bcf91c",
    "uniform_per_row_3d_float16": "5139f7045b79280f",
    "normal_per_row_3d_float16": "6833f3cead25c0d9",
    "stdnormal_per_row_3d_float16": "d55b29f7f6b71ca4",
    "uniform_per_pair_float16": "5f713fe528c7746f",
    "normal_per_pair_float16": "e52c2c6751ca0bc6",
    "stdnormal_per_pair_float16": "b33f40328be8fab3",
    "uniform_broadcast_rows_float16": "96c5fb9c8e5fb0f9",
    "normal_broadcast_rows_float16": "c67be68520b274b2",
    "stdnormal_broadcast_rows_float16": "f732eb8b29522fb9",
    "uniform_one_key_1000_bfloat16": "4ce45fdd03aff5c3",
    "normal_one_key_1000_bfloat16": "08c164b17f73fda8",
    "stdnormal_one_key_1000_bfloat16": "cc90a959d7517aba",
    "uniform_one_key_7_bfloat16": "3180bd1a18672398",
    "normal_one_key_7_bfloat16": "8262c48999c737ea",
    "stdnormal_one_key_7_bfloat16": "259a008e249ca8d1",
    "uniform_per_element_bfloat16": "7b963a178a1764f1",
    "normal_per_element_bfloat16": "b64c59c45dc5932c",
    "stdnormal_per_element_bfloat16": "74cc00a13f41267d",
    "uniform_per_row_13_bfloat16": "dba170b1f56dcf34",
    "normal_per_row_13_bfloat16": "ae09e53cfd586fc3",
    "stdnormal_per_row_13_bfloat16": "7a363f80a17b915f",
    "uniform_per_row_3d_bfloat16": "4c996670ed49538d",
    "normal_per_row_3d_bfloat16": "55e21bb07abec89c",
    "stdnormal_per_row_3d_bfloat16": "c5c431c7ebcfc50a",
    "uniform_per_pair_bfloat16": "f7d6bb2cba578ba9",
    "normal_per_pair_bfloat16": "39b38e20802a6dce",
    "stdnormal_per_pair_bfloat16": "666f17f205d4da30",
    "uniform_broadcast_rows_bfloat16": "24915e796b022fe9",
    "normal_broadcast_rows_bfloat16": "7d074806ac9e11b5",
    "stdnormal_broadcast_rows_bfloat16": "2f496126e8e1a343",
    "uniform_one_key_1000_float64": "d5e6cde87e4e6406",
    "normal_one_key_1000_float64": "aa8f7c7c936c8957",
    "stdnormal_one_key_1000_float64": "bce80bfc4da89605",
    "uniform_one_key_7_float64": "9e1e2c1a1f48f0d6",
    "normal_one_key_7_float64": "97359a4f1717fcd0",
    "stdnormal_one_key_7_float64": "1ae7766bd9f495f8",
    "uniform_per_element_float64": "eaa16efc1aa9bc83",
    "normal_per_element_float64": "b01581330dd0b37a",
    "stdnormal_per_element_float64": "7581d96eb7ab69bc",
    "uniform_per_row_13_float64": "ad03e2a1b67bb7ae",
    "normal_per_row_13_float64": "e251d6b4fe130001",
    "stdnormal_per_row_13_float64": "566e27ae706112a1",
    "uniform_per_row_3d_float64": "e270ffb97cf6bb6a",
    "normal_per_row_3d_float64": "dcc795d032e45c06",
    "stdnormal_per_row_3d_float64": "ec13019eca0da546",
    "uniform_per_pair_float64": "f9c0689b67bfd388",
    "normal_per_pair_float64": "9560a69c270d406e",
    "stdnormal_per_pair_float64": "d7b504b470eca1d6",
    "uniform_broadcast_rows_float64": "92bc1bdb30d32641",
    "normal_broadcast_rows_float64": "4e57aedac562bfa9",
    "stdnormal_broadcast_rows_float64": "562fa360c0280e44",
}


def test_matches_recorded_cuda(mojo_gpu):
    got = _cases(mojo_gpu)
    wrong = [
        name
        for name, value in got.items()
        if name in _GOLDEN and _claimed(name) and _digest(value) != _GOLDEN[name]
    ]
    assert not wrong
    assert set(_GOLDEN) >= set(got)


def test_matches_live_cuda(mojo_gpu):
    if not (_UPSTREAM and torch.cuda.is_available()):
        pytest.skip("needs a CUDA build of torch >= 2.13")
    got = _cases(mojo_gpu)
    want = _cases("cuda")
    wrong = [
        name
        for name in got
        if _claimed(name)
        and not torch.equal(
            got[name].cpu().view(torch.uint8), want[name].cpu().view(torch.uint8)
        )
    ]
    assert not wrong


@pytest.mark.parametrize("dtype", [torch.float32, torch.float16, torch.bfloat16])
def test_distributions_match_cpu(mojo_gpu, dtype):
    if not _UPSTREAM:
        pytest.skip("no CPU kernel before torch 2.13")
    k = _keys()
    for shape, key in [((1000,), k[0, 0]), ((3, 4, 5), k[:, :1, None])]:
        lo = _draw(A._philox_uniform_, shape, dtype, key, -2.3, 5.1)
        dev = _draw(A._philox_uniform_, shape, dtype, key.to(mojo_gpu), -2.3, 5.1)
        # CPU skips the fma contraction CUDA does: one rounding apart.
        torch.testing.assert_close(
            dev.cpu(), lo, rtol=0, atol=1e-5 if dtype == torch.float32 else 1e-2
        )
        assert ((dev.cpu() >= -2.3) & (dev.cpu() <= 5.1)).all()
        n = _draw(A._philox_normal_, shape, dtype, key, 0.5, 2.0)
        ndev = _draw(A._philox_normal_, shape, dtype, key.to(mojo_gpu), 0.5, 2.0)
        torch.testing.assert_close(ndev.cpu(), n, rtol=2e-3, atol=2e-3)


def test_keys_match_cpu(mojo_gpu):
    if not _UPSTREAM:
        pytest.skip("no CPU kernel before torch 2.13")
    k = _keys()
    km = k.to(mojo_gpu)
    for got, want in [
        (A._philox_key_split(km, 4), A._philox_key_split(k, 4)),
        (A._philox_key_fold_in(km, 123), A._philox_key_fold_in(k, 123)),
        (
            A._philox_key_fold_in(km.transpose(0, 1), 9),
            A._philox_key_fold_in(k.transpose(0, 1), 9),
        ),
    ]:
        assert torch.equal(got.cpu().view(torch.int64), want.view(torch.int64))


def test_noncontiguous_self_and_empty(mojo_gpu):
    k = _keys().to(mojo_gpu)
    dense = _draw(A._philox_normal_, (8, 6), torch.float32, k[0, 0])
    strided = torch.empty(6, 8, device=mojo_gpu).t()
    out = A._philox_normal_(strided, k[0, 0])
    assert out is strided
    torch.testing.assert_close(strided.cpu(), dense.cpu(), rtol=0, atol=0)
    empty = torch.empty(0, 3, device=mojo_gpu)
    assert A._philox_uniform_(empty, k[0, 0]) is empty
    assert A._philox_key_split(k[:0], 2).shape == (2, 0, 4, 2)


@pytest.mark.parametrize(
    ("call", "message"),
    [
        (lambda k: A._philox_key_split(k, 0), "num_splits must be positive, got 0"),
        (
            lambda k: A._philox_key_split(k[..., :1], 2),
            r"key must have shape \(\*batch, 2\), got shape \[3, 4, 1\]",
        ),
        (
            lambda k: A._philox_key_fold_in(k.view(torch.int64), 1),
            "key must have dtype uint64, got Long",
        ),
        (
            lambda k: A._philox_key_fold_in.Tensor(
                k, torch.zeros(2, dtype=torch.uint64, device=k.device)
            ),
            "data must be a single value, got 2 elements",
        ),
        (
            lambda k: A._philox_key_fold_in.Tensor(
                k, torch.zeros(1, dtype=torch.int64, device=k.device)
            ),
            "data must have dtype uint64, got Long",
        ),
        (
            lambda k: A._philox_normal_(
                torch.empty(3, dtype=torch.int64, device=k.device), k[0, 0]
            ),
            "self must be a floating point tensor, got Long",
        ),
        (
            lambda k: A._philox_normal_(torch.empty(3, 4, 5, device=k.device), k),
            r"batched key must have ndim == output ndim \+ 1, got key shape \[3, 4, 2\] with output shape \[3, 4, 5\]",
        ),
        (
            lambda k: A._philox_uniform_(torch.empty(3, 5, device=k.device), k),
            r"key batch shape \[3, 4\] is not broadcastable with output shape \[3, 5\]",
        ),
        (
            lambda k: A._philox_uniform_(
                torch.empty(3, device=k.device), k[0, 0].cpu()
            ),
            "self and key must be on the same device",
        ),
    ],
)
def test_errors(mojo_gpu, call, message):
    with pytest.raises(RuntimeError, match=message):
        call(_keys().to(mojo_gpu))
