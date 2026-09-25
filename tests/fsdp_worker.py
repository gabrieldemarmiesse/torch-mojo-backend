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


def _stress_indices(count: int) -> torch.Tensor:
    """Sample the whole shard, its ends, and both sides of regular boundaries."""
    if count <= 4096:
        return torch.arange(count)
    spread = torch.arange(1024) * count // 1024
    boundaries = torch.arange(16_384, count, 16_384)
    return torch.unique(
        torch.cat(
            (
                spread,
                torch.arange(16),
                torch.arange(count - 16, count),
                boundaries - 1,
                boundaries,
                boundaries + 1,
            )
        ).clamp_max(count - 1)
    )


def _gather_reference(
    pattern: torch.Tensor, world: int, step: int, dtype: torch.dtype
) -> torch.Tensor:
    """What every rank's all-gather output holds when peer p sent pattern+p+step."""
    peers = torch.arange(world, dtype=torch.float32)[:, None]
    return (pattern[None, :] + peers + step).reshape(-1).to(dtype)


def _check_inplace_all_gather(
    count: int, dtype: torch.dtype, offset: int, pattern: torch.Tensor
):
    """FSDP2's layout: the input shard is a view of this rank's output slot.

    `foreach_all_gather` copies the parameters into
    `all_gather_output.narrow(0, rank * numel, numel)` and gathers from that
    view, so the kernel's staging read and its write of the rank's own slot
    alias. Every element is checked, which also covers the chunk
    boundaries of whatever split the transport chose.
    """
    rank, world = dist.get_rank(), dist.get_world_size()
    storage = torch.full(
        (world * count + offset + 1,), -123, dtype=dtype, device="mojo"
    )
    output = storage[offset:-1]
    mine = output[rank * count : (rank + 1) * count]
    for generation in range(1, 4):
        step = 100 + generation  # disjoint from the out-of-place values
        output.fill_(-123)
        mine.copy_((pattern + rank + step).to(dtype))
        dist.all_gather_into_tensor(output, mine, async_op=True).wait()
        torch.testing.assert_close(
            output.cpu(), _gather_reference(pattern, world, step, dtype), rtol=0, atol=0
        )
    assert storage[offset - 1].cpu().item() == storage[-1].cpu().item() == -123


def check_fsdp_collectives_stress():
    """Reuse changing AG/RS payloads across generations and network inbox slots.

    Full FSDP2-sized transfers run on the device. Most generations send only
    deterministic samples to the CPU: four generations are queued before
    checking, so validation does not synchronize each collective. The first
    and last generation of each case are checked in full, every element of
    both outputs, so every chunk boundary of the transport's split is
    covered without this test having to know the split. Each case then runs
    an in-place all-gather, FSDP2's own layout. Values and SUM references
    are exact at any world size, AVG ones for a power-of-two world (see
    `avg_rtol`).
    """
    rank, world = dist.get_rank(), dist.get_world_size()
    # AVG scales each contribution by fp32(1/world) before summing (NCCL's
    # PreMulSum) while the reference divides the exact sum: identical only
    # when 1/world is exact, i.e. for a power-of-two world (as in
    # ddp_worker's stress).  Otherwise allow the fp32 rounding of `world`
    # scaled terms (well under 1e-4 at these magnitudes), far below the
    # >= 1/world by which one stale or wrong-shard contribution moves a value.
    exact_avg = world & (world - 1) == 0
    avg_rtol, avg_atol = (0.0, 0.0) if exact_avg else (1e-5, 1e-4)
    rounds = 12
    full_generations = (1, rounds)
    cases = (
        (1, torch.float32),
        (13, torch.bfloat16),
        (357 * 789 + 3, torch.float32),
        # 2,560,006 B per rank: above the balanced-split threshold at 4 local
        # ranks (PIPE_SPLIT_UNIT * 4 = 2.56 MB) and not a multiple of 16, so
        # the second chunk (1,279,990 B after 1,280,016) ends in a scalar
        # tail and every rank but 0 has its output slot off a 16-byte
        # boundary.
        (1_280_003, torch.bfloat16),
        (3_840_000, torch.bfloat16),  # XL block: 7.68 MB AG / 15.36 MB RS out.
        (10_254_200, torch.float32),  # XL root: 41.02 MB per rank.
    )
    for count, dtype in cases:
        # Small/awkward cases get misaligned bases; the large ones keep
        # FSDP2's 16-byte-aligned bases, so their misalignment (if any) is the
        # per-rank stride alone. Destination-dependent RS input catches
        # wrong-shard reads.
        offset = 1 if count < 1_000_000 else 8
        indices = _stress_indices(count)
        gather_indices = torch.cat([indices + peer * count for peer in range(world)])
        device_indices = indices.to("mojo")
        device_gather_indices = gather_indices.to("mojo")
        pattern = (torch.arange(count, dtype=torch.int32) % 17 - 8).float()
        ag_source_storage = torch.full(
            (count + offset + 1,), -123, dtype=dtype, device="mojo"
        )
        ag_source = ag_source_storage[offset:-1]
        ag_source.copy_((pattern + rank).to(dtype))
        rs_source_storage = torch.full(
            (world * count + offset + 1,), -123.0, device="mojo"
        )
        rs_source = rs_source_storage[offset:-1]
        rs_source.copy_(
            (pattern[None, :] + torch.arange(world)[:, None] + rank).reshape(-1)
        )
        ag_storage = torch.full(
            (world * count + offset + 1,), -123, dtype=dtype, device="mojo"
        )
        rs_storage = torch.full((count + offset + 1,), -123.0, device="mojo")
        ag_output, rs_output = ag_storage[offset:-1], rs_storage[offset:-1]
        pending = []
        for generation in range(1, rounds + 1):
            ag_source.add_(1)
            ag_output.fill_(-123)
            dist.all_gather_into_tensor(ag_output, ag_source, async_op=True).wait()
            if generation in full_generations:
                torch.testing.assert_close(
                    ag_output.cpu(),
                    _gather_reference(pattern, world, generation, dtype),
                    rtol=0,
                    atol=0,
                )
            ag_sample = ag_output.index_select(0, device_gather_indices)
            rs_source.add_(1)
            rs_output.fill_(-123)
            op = dist.ReduceOp.SUM if generation % 2 else dist.ReduceOp.AVG
            dist.reduce_scatter_tensor(
                rs_output, rs_source, op=op, async_op=True
            ).wait()
            if generation in full_generations:
                expected_full = (pattern + rank + generation) * world + world * (
                    world - 1
                ) // 2
                rtol = atol = 0.0
                if op == dist.ReduceOp.AVG:
                    expected_full /= world
                    rtol, atol = avg_rtol, avg_atol
                torch.testing.assert_close(
                    rs_output.cpu(), expected_full, rtol=rtol, atol=atol
                )
            rs_sample = rs_output.index_select(0, device_indices)
            pending.append((generation, op, ag_sample, rs_sample))
            if len(pending) < 4:
                continue
            for step, reduction, gathered, scattered in pending:
                expected_ag = torch.cat(
                    [pattern[indices] + peer + step for peer in range(world)]
                ).to(dtype)
                expected_rs = (pattern[indices] + rank + step) * world + world * (
                    world - 1
                ) // 2
                rtol = atol = 0.0
                if reduction == dist.ReduceOp.AVG:
                    expected_rs /= world
                    rtol, atol = avg_rtol, avg_atol
                torch.testing.assert_close(gathered.cpu(), expected_ag, rtol=0, atol=0)
                torch.testing.assert_close(
                    scattered.cpu(), expected_rs, rtol=rtol, atol=atol
                )
            pending.clear()
        # Source preservation is independent of output correctness.
        torch.testing.assert_close(
            ag_source.index_select(0, device_indices).cpu(),
            (pattern[indices] + rank + rounds).to(dtype),
            rtol=0,
            atol=0,
        )
        expected_source = torch.cat(
            [pattern[indices] + peer + rank + rounds for peer in range(world)]
        )
        torch.testing.assert_close(
            rs_source.index_select(0, device_gather_indices).cpu(),
            expected_source,
            rtol=0,
            atol=0,
        )
        for storage in (ag_source_storage, rs_source_storage, ag_storage, rs_storage):
            assert storage[offset - 1].cpu().item() == storage[-1].cpu().item() == -123
        _check_inplace_all_gather(count, dtype, offset, pattern)
        print(
            f"rank={rank} FSDP collectives stress count={count} dtype={dtype} "
            f"generations={rounds} in-place OK",
            flush=True,
        )


def main():
    register_mojo_devices()
    dist.init_process_group("mojo", timeout=datetime.timedelta(minutes=5))
    rank, world = dist.get_rank(), dist.get_world_size()
    if len(sys.argv) > 1 and sys.argv[1] == "reduce_scatter":
        check_reduce_scatter()
        dist.destroy_process_group()
        return
    if len(sys.argv) > 1 and sys.argv[1] == "fsdp_collectives_stress":
        check_fsdp_collectives_stress()
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
