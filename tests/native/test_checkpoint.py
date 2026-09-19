"""Checkpoint staging with CUDA and Mojo available in the same process."""

import pytest
import torch
from torch.distributed.checkpoint import filesystem


def test_overlapping_checkpoint_loader_stages_mixed_devices(mojo_gpu):
    if not torch.cuda.is_available():
        pytest.skip("needs CUDA torch and a Mojo GPU")
    expected = torch.arange(17, dtype=torch.float32)
    tensors = [expected.to(device) for device in (mojo_gpu, "cuda", "cpu")]
    pointers = [tensor.data_ptr() for tensor in tensors]
    versions = [tensor._version for tensor in tensors]
    loader = filesystem._OverlappingCpuLoader(
        lambda index: tensors[index], inflight_threshhold=1
    )
    for index, tensor in enumerate(tensors):
        loader.add(tensor.nbytes, index)
    results = list(loader.values())
    assert len(results) == len(tensors)
    for tensor, index in results:
        assert isinstance(index, int)
        assert tensor.device.type == "cpu"
        torch.testing.assert_close(tensor, expected, rtol=0, atol=0)
        torch.testing.assert_close(tensors[index].cpu(), expected, rtol=0, atol=0)
    assert [tensor.data_ptr() for tensor in tensors] == pointers
    assert [tensor._version for tensor in tensors] == versions
