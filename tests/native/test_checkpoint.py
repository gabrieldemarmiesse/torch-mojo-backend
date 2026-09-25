"""Checkpoints of accelerator tensors: torch.save / torch.load (through
aten::set_.source_Storage and its storage-offset overload), and checkpoint
staging with CUDA and Mojo available in the same process."""

import io

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


def _roundtrip(obj, **load_kwargs):
    buffer = io.BytesIO()
    torch.save(obj, buffer)
    buffer.seek(0)
    return torch.load(buffer, **load_kwargs)


@pytest.mark.parametrize(
    "dtype", [torch.float32, torch.bfloat16, torch.float16, torch.int64, torch.bool]
)
@pytest.mark.parametrize("map_location", [None, "cpu", "device"])
def test_save_accelerator_tensors(mojo_gpu, dtype, map_location):
    """A saved accelerator tensor loads back where it was, or where
    map_location sends it, with views still sharing one storage."""
    base = (torch.arange(357 * 7) % 13).reshape(357, 7).to(dtype)
    t = base.to(mojo_gpu)
    location = mojo_gpu if map_location == "device" else map_location
    got = _roundtrip(
        {"t": t, "view": t[5:, 2:6].t(), "scalar": t[3, 4], "empty": t[:0]},
        map_location=location,
    )
    want_device = torch.device("cpu" if location == "cpu" else mojo_gpu)
    for key, want in (
        ("t", base),
        ("view", base[5:, 2:6].t()),
        ("scalar", base[3, 4]),
        ("empty", base[:0]),
    ):
        assert got[key].device == want_device, key
        assert got[key].shape == want.shape, key
        assert got[key].stride() == want.stride(), key
        torch.testing.assert_close(got[key].cpu(), want, rtol=0, atol=0)
    storage = got["t"].untyped_storage()
    assert got["view"].untyped_storage().data_ptr() == storage.data_ptr()
    assert got["view"].storage_offset() == 5 * 7 + 2


def test_load_cpu_checkpoint_onto_the_accelerator(mojo_gpu, tmp_path):
    expected = torch.randn(33, 17)
    path = tmp_path / "ckpt.pt"
    torch.save({"w": expected, "step": torch.tensor(3)}, path)
    device = torch.accelerator.current_accelerator()
    for location in (device, mojo_gpu, {"cpu": mojo_gpu}):
        loaded = torch.load(path, map_location=location)
        assert loaded["w"].device == torch.device(mojo_gpu)
        torch.testing.assert_close(loaded["w"].cpu(), expected, rtol=0, atol=0)
    mapped = torch.load(path, map_location=mojo_gpu, mmap=True)
    torch.testing.assert_close(mapped["w"].cpu(), expected, rtol=0, atol=0)


def test_state_dict_roundtrip_on_the_accelerator(mojo_gpu):
    model = torch.nn.Linear(9, 5).to(mojo_gpu)
    optimizer = torch.optim.AdamW(model.parameters(), lr=0.01)
    model(torch.randn(4, 9, device=mojo_gpu)).sum().backward()
    optimizer.step()
    checkpoint = _roundtrip(
        {"model": model.state_dict(), "optimizer": optimizer.state_dict()},
        map_location=mojo_gpu,
    )
    restored = torch.nn.Linear(9, 5).to(mojo_gpu)
    restored.load_state_dict(checkpoint["model"])
    restored_optimizer = torch.optim.AdamW(restored.parameters(), lr=0.01)
    restored_optimizer.load_state_dict(checkpoint["optimizer"])
    for mine, theirs in zip(restored.parameters(), model.parameters(), strict=True):
        torch.testing.assert_close(mine.cpu(), theirs.cpu(), rtol=0, atol=0)
    state = restored_optimizer.state[restored.weight]
    assert state["exp_avg"].device == torch.device(mojo_gpu)
    torch.testing.assert_close(
        state["exp_avg"].cpu(), optimizer.state[model.weight]["exp_avg"].cpu()
    )


def test_set_source_storage(mojo_gpu):
    """set_(storage) views the whole storage as 1-D in self's dtype."""
    src = torch.arange(10, dtype=torch.float32).to(mojo_gpu)
    t = torch.empty(0, dtype=torch.float32, device=mojo_gpu).set_(src.untyped_storage())
    assert t.shape == (10,)
    assert t.data_ptr() == src.data_ptr()
    torch.testing.assert_close(t.cpu(), torch.arange(10, dtype=torch.float32))
    halves = torch.empty(0, dtype=torch.float16, device=mojo_gpu)
    assert halves.set_(src.untyped_storage()).shape == (20,)


def test_set_source_storage_offset_and_growth(mojo_gpu):
    src = torch.arange(24, dtype=torch.float32).to(mojo_gpu)
    t = torch.empty(0, device=mojo_gpu)
    t.set_(src.untyped_storage(), 2, (3, 4), (6, 1))
    assert (t.shape, t.stride(), t.storage_offset()) == ((3, 4), (6, 1), 2)
    want = torch.arange(24, dtype=torch.float32).as_strided((3, 4), (6, 1), 2)
    torch.testing.assert_close(t.cpu(), want, rtol=0, atol=0)
    # No stride: contiguous. A view past the end grows the storage, as ATen's
    # resize_impl_ does, keeping the bytes already there.
    storage = torch.arange(4, dtype=torch.float32).to(mojo_gpu).untyped_storage()
    grown = torch.empty(0, device=mojo_gpu)
    torch.ops.aten.set_.source_Storage_storage_offset(grown, storage, 0, [2, 5])
    assert grown.stride() == (5, 1)
    assert storage.nbytes() == 40
    torch.testing.assert_close(grown.cpu().view(-1)[:4], torch.arange(4.0))


def test_set_source_storage_from_another_device_raises(mojo_gpu):
    t = torch.empty(0, device=mojo_gpu)
    with pytest.raises(RuntimeError, match="expected a storage on"):
        t.set_(torch.arange(4.0).untyped_storage())
