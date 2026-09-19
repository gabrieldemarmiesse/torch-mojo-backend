"""Two-rank FSDP2 correctness checks, launched by test_distributed.py."""

# ruff: noqa: E402 -- choose rank's GPU before importing torch/MAX

from torch_mojo_backend.distributed import use_local_rank_gpu

use_local_rank_gpu()

import copy
import datetime
import sys
import tempfile

import torch
import torch.distributed as dist
import torch.distributed.checkpoint as dcp
from torch.distributed.device_mesh import init_device_mesh
from torch.distributed.fsdp import fully_shard
from torch.distributed.tensor import DTensor

from torch_mojo_backend import register_mojo_devices
from torch_mojo_backend.native import device_module


def check_checkpoint(model: torch.nn.Module, optimizer: torch.optim.Optimizer):
    """Save/load sharded model and Adam state, then verify local values."""
    state = {"model": model.state_dict(), "optimizer": optimizer.state_dict()}
    expected = {k: v.to_local().cpu().clone() for k, v in state["model"].items()}
    with tempfile.TemporaryDirectory() as directory:
        paths = [directory]
        dist.broadcast_object_list(paths, src=0)
        dcp.save(state, checkpoint_id=paths[0])
        with torch.no_grad():
            for p in model.parameters():
                p.add_(1)
            for param_state in optimizer.state.values():
                for value in param_state.values():
                    if isinstance(value, DTensor):
                        value.zero_()
        dcp.load(state, checkpoint_id=paths[0])
        model.load_state_dict(state["model"])
        optimizer.load_state_dict(state["optimizer"])
        for name, value in model.state_dict().items():
            assert isinstance(value, DTensor)
            torch.testing.assert_close(
                value.to_local().cpu(), expected[name], rtol=0, atol=0
            )
        dist.barrier()
    print(f"rank={dist.get_rank()} sharded checkpoint round-trip OK", flush=True)


def check_reduce_scatter():
    rank, world = dist.get_rank(), dist.get_world_size()
    for dtype in (
        torch.float32,
        torch.float16,
        torch.bfloat16,
        torch.int32,
        torch.int64,
    ):
        # 700k elements is over mojoccl's staging arena at
        # MOJOCCL_REGION_MB=1, which is how test_distributed.py reaches the
        # chunk loop of a collective that is one launch at the default region.
        for count in (0, 1, 13, 357 * 789, 700_000):
            for op in (dist.ReduceOp.SUM, dist.ReduceOp.AVG):
                if not dtype.is_floating_point and op == dist.ReduceOp.AVG:
                    continue
                # Distinct rank and destination data; offset slices exercise
                # pointers that are not 16-byte aligned, including chunk tails.
                base = torch.arange(world * count + 1) % 17
                source = (base + rank).to(dtype).to("mojo")[1:]
                before = source.cpu().clone()
                storage = torch.full((count + 2,), -123, dtype=dtype, device="mojo")
                output = storage[1:-1]
                work = dist.reduce_scatter_tensor(output, source, op=op, async_op=True)
                side = torch.Stream(device="mojo")
                with device_module.stream(side):
                    work.wait()
                    observed = output.clone()
                side.synchronize()
                expected = (
                    base[1:].reshape(world, count)[rank] * world
                    + world * (world - 1) // 2
                ).to(torch.float32)
                if op == dist.ReduceOp.AVG:
                    expected /= world
                torch.testing.assert_close(
                    observed.cpu(), expected.to(dtype), rtol=0, atol=0
                )
                torch.testing.assert_close(source.cpu(), before, rtol=0, atol=0)
                assert storage[0].cpu().item() == storage[-1].cpu().item() == -123
    # AVG scales each contribution before summing (NCCL's PreMulSum), so the
    # average of values whose SUM overflows is still finite.
    hot = torch.full((world * 8,), 3.0e38, dtype=torch.float32, device="mojo")
    avg = torch.zeros(8, dtype=torch.float32, device="mojo")
    dist.reduce_scatter_tensor(avg, hot, op=dist.ReduceOp.AVG)
    torch.testing.assert_close(avg.cpu(), torch.full((8,), 3.0e38), rtol=0, atol=0)
    # NCCL also permits the result to alias this rank's input shard.
    source = (torch.arange(world * 13, dtype=torch.float32) + rank).to("mojo")
    output = source[rank * 13 : (rank + 1) * 13]
    dist.reduce_scatter_tensor(output, source)
    expected = (
        torch.arange(rank * 13, (rank + 1) * 13, dtype=torch.float32) * world
        + world * (world - 1) // 2
    )
    torch.testing.assert_close(output.cpu(), expected, rtol=0, atol=0)
    # Mix collective types and caller streams, including a zero-size scope.
    # A reduce-scatter must close its communicator ordering state so that
    # the next collective can submit, and Work.wait must order its reader.
    source = (torch.arange(world * 13, dtype=torch.float32) + rank).to("mojo")
    first, second = torch.Stream(device="mojo"), torch.Stream(device="mojo")
    first.wait_stream(torch.accelerator.current_stream())
    with device_module.stream(first):
        initial = dist.all_reduce(source, async_op=True)
    with device_module.stream(second):
        initial.wait()
        empty = source[:0]
        dist.reduce_scatter_tensor(empty, empty, async_op=True).wait()
        output = torch.empty(13, device="mojo")
        scattered = dist.reduce_scatter_tensor(output, source, async_op=True)
    with device_module.stream(first):
        scattered.wait()
        observed = output.clone()
        sentinel = torch.ones(13, device="mojo")
        dist.all_reduce(sentinel, async_op=True).wait()
    first.synchronize()
    expected = (
        torch.arange(rank * 13, (rank + 1) * 13, dtype=torch.float32) * world
        + world * (world - 1) // 2
    ) * world
    torch.testing.assert_close(observed.cpu(), expected, rtol=0, atol=0)
    torch.testing.assert_close(sentinel.cpu(), torch.full((13,), float(world)))
    print(f"rank={rank} reduce-scatter correctness OK", flush=True)


def main():
    register_mojo_devices()
    dist.init_process_group("mojo", timeout=datetime.timedelta(minutes=5))
    rank, world = dist.get_rank(), dist.get_world_size()
    if len(sys.argv) > 1 and sys.argv[1] == "reduce_scatter":
        check_reduce_scatter()
        dist.destroy_process_group()
        return
    torch.manual_seed(123)
    reference = torch.nn.Sequential(
        torch.nn.Linear(7, 13), torch.nn.GELU(), torch.nn.Linear(13, 5)
    )
    model = copy.deepcopy(reference).to("mojo")
    mesh = init_device_mesh("mojo", (world,))
    for layer in model:
        if isinstance(layer, torch.nn.Linear):
            fully_shard(layer, mesh=mesh)
    fully_shard(model, mesh=mesh)
    optimizer = torch.optim.AdamW(model.parameters(), lr=0.001, foreach=False)
    ref_optimizer = torch.optim.AdamW(reference.parameters(), lr=0.001, foreach=False)
    assert all(isinstance(p, DTensor) for p in model.parameters())
    for step in range(3):
        x = torch.randn(world * 3, 7)
        y = torch.randn(world * 3, 5)
        start, end = rank * 3, (rank + 1) * 3
        loss = (
            (model(x[start:end].to("mojo")) - y[start:end].to("mojo")).square().mean()
        )
        loss.backward()
        ref_loss = (reference(x) - y).square().mean()
        ref_loss.backward()
        for p, ref in zip(model.parameters(), reference.parameters()):
            assert isinstance(p.grad, DTensor)
            torch.testing.assert_close(
                p.grad.full_tensor().cpu(), ref.grad, atol=2e-5, rtol=2e-4
            )
        optimizer.step()
        ref_optimizer.step()
        for p, ref in zip(model.parameters(), reference.parameters()):
            assert isinstance(p, DTensor)
            torch.testing.assert_close(p.full_tensor().cpu(), ref, atol=2e-5, rtol=2e-4)
        optimizer.zero_grad(set_to_none=True)
        ref_optimizer.zero_grad(set_to_none=True)
        print(f"rank={rank} step={step} FSDP2 gradient/update parity OK", flush=True)
        if step == 1:
            check_checkpoint(model, optimizer)
    dist.destroy_process_group()


if __name__ == "__main__":
    main()
