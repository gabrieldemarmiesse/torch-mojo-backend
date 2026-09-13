"""Model conversion and portable checkpoint behavior for Mojo tensors."""

import io

import pytest
import torch

from torch_mojo_backend import register_mojo_devices


@pytest.fixture(autouse=True)
def setup_mojo_device():
    register_mojo_devices()


def test_module_to_mojo_preserves_tied_parameters(mojo_device):
    class TiedWeights(torch.nn.Module):
        def __init__(self):
            super().__init__()
            self.embedding = torch.nn.Embedding(16, 8)
            self.projection = torch.nn.Linear(8, 16, bias=False)
            self.projection.weight = self.embedding.weight

    module = TiedWeights()
    assert module.embedding.weight is module.projection.weight

    module.to(mojo_device)

    # `set_swap_module_params_on_conversion` (turned on by `register_mojo_devices`)
    # swaps a shared Parameter's storage once rather than duplicating it per
    # module, so identity -- and therefore the underlying allocation -- stays
    # shared after `.to()`. Real native tensors carry no extra attributes to
    # inspect directly; `data_ptr()` is the public way to confirm one
    # allocation is shared.
    assert module.embedding.weight is module.projection.weight
    embedding_weight = module.embedding.weight
    projection_weight = module.projection.weight
    assert embedding_weight.device.type == "mojo"
    assert projection_weight.device.type == "mojo"
    assert embedding_weight.data_ptr() == projection_weight.data_ptr()
    assert len(list(module.parameters())) == 1


@pytest.mark.xfail(
    strict=False,
    reason="op not ported yet: aten::set_.source_Storage (torch.load's tensor "
    "rebuild path needs it to restore the original mojo storage before "
    "map_location moves it to cpu)",
)
def test_mojo_tensor_checkpoint_loads_as_portable_cpu_tensor(mojo_device):
    expected = torch.arange(12, dtype=torch.float32).reshape(3, 4)
    value = expected.to(mojo_device)

    checkpoint = io.BytesIO()
    torch.save({"value": value}, checkpoint)
    checkpoint.seek(0)
    restored = torch.load(checkpoint, map_location="cpu")["value"]

    assert type(restored) is torch.Tensor
    assert restored.device.type == "cpu"
    torch.testing.assert_close(restored, expected)
