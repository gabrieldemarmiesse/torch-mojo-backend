"""Bit-exactness of the device RNG against stock PyTorch CUDA.

Every case in rng_parity_cases runs from the same seed on the mojo GPU and on
`cuda` of the same GPU (the live oracle, tests/native/cuda_oracle.py): same
bits, same generator state after the draw.
"""

import os

import pytest
import torch

from tests.native import rng_parity_cases as cases
from tests.native.cuda_oracle import CudaOracle
from torch_mojo_backend.native import device_module


def _case_ids() -> list[str]:
    ids = list(cases.CASES)
    if os.environ.get("TMB_RNG_BIG"):
        ids += sorted(cases.BIG_CASES)
    return ids


@pytest.mark.parametrize("name", _case_ids())
def test_matches_cuda(cuda_oracle: CudaOracle, name: str):
    """Same seed, same bits, same counter as stock CUDA. Above the grid cap
    the draw depends on the GPU model, as on CUDA: the oracle is the same
    GPU."""
    fn = cases.CASES.get(name) or cases.BIG_CASES[name]
    cuda_oracle.assert_bitwise_equal_by_device(fn, seed=cases.SEED, what=name)


def test_state_layout_matches_cuda(mojo_gpu):
    """get_rng_state is CUDA's 16 bytes: seed then offset, little-endian."""
    torch.manual_seed(cases.SEED)
    torch.rand(1000, device=mojo_gpu)
    state = device_module.get_rng_state().tolist()
    assert int.from_bytes(bytes(state[:8]), "little") == cases.SEED
    assert int.from_bytes(bytes(state[8:]), "little") == 4
