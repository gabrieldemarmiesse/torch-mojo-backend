"""Standard DistributedDataParallel on the mojo device via MojoProcessGroup
(thread-per-rank, M2 of docs/multi_gpu_training_plan.md)."""

import pytest
import torch
import torch.distributed
import torch.nn as nn
from torch.nn.parallel import DistributedDataParallel as DDP

from torch_mojo_backend import distributed as mojo_dist
from torch_mojo_backend import register_mojo_devices
from torch_mojo_backend.mojo_device.torch_mojo_tensor import get_ordered_accelerators

pytestmark = pytest.mark.xdist_group(name="group1")


@pytest.fixture(autouse=True)
def setup_mojo_device():
    register_mojo_devices()


def gpu_count() -> int:
    return sum(acc.label == "gpu" for acc in get_ordered_accelerators())


def require_two_gpus():
    if gpu_count() < 2:
        pytest.skip("requires at least two MAX GPUs")


def make_model() -> nn.Sequential:
    torch.manual_seed(7)
    return nn.Sequential(nn.Linear(16, 32), nn.GELU(), nn.Linear(32, 4))


def test_process_group_allreduce_and_broadcast():
    require_two_gpus()
    world = 2
    reduced = {}
    broadcast_result = {}

    def worker(rank: int, world_size: int, pg: mojo_dist.MojoProcessGroup):
        assert pg.rank() == rank
        assert pg.size() == world_size
        t = torch.full((8,), float(rank + 1), device=f"mojo:{rank}")
        pg.allreduce([t]).wait()
        reduced[rank] = t.cpu()

        b = torch.full((4,), float(rank), device=f"mojo:{rank}")
        opts = torch.distributed.BroadcastOptions()
        opts.rootRank = 1
        opts.rootTensor = 0
        pg.broadcast([b], opts).wait()
        broadcast_result[rank] = b.cpu()

    mojo_dist.spawn(worker, world_size=world)

    expected_sum = torch.full((8,), 3.0)
    for rank in range(world):
        torch.testing.assert_close(reduced[rank], expected_sum)
        torch.testing.assert_close(broadcast_result[rank], torch.full((4,), 1.0))


def test_process_group_allgather():
    require_two_gpus()
    gathered = {}

    def worker(rank: int, world_size: int, pg: mojo_dist.MojoProcessGroup):
        inp = torch.full((2,), float(rank + 10), device=f"mojo:{rank}")
        outputs = [[torch.zeros(2, device=f"mojo:{rank}") for _ in range(world_size)]]
        pg.allgather(outputs, [inp]).wait()
        gathered[rank] = [t.cpu() for t in outputs[0]]

    mojo_dist.spawn(worker, world_size=2)

    for rank in range(2):
        torch.testing.assert_close(gathered[rank][0], torch.full((2,), 10.0))
        torch.testing.assert_close(gathered[rank][1], torch.full((2,), 11.0))


def test_ddp_ranks_stay_synchronized():
    """After DDP steps with different per-rank data, all ranks must hold
    identical parameters (grad averaging synchronizes them)."""
    require_two_gpus()
    world = 2
    states = {}

    def worker(rank: int, world_size: int, pg: mojo_dist.MojoProcessGroup):
        device = f"mojo:{rank}"
        model = make_model().to(device)
        ddp = DDP(model, process_group=pg, device_ids=None)
        optimizer = torch.optim.SGD(ddp.parameters(), lr=0.05)

        torch.manual_seed(100 + rank)  # different data per rank
        for _ in range(3):
            x = torch.randn(8, 16).to(device)
            y = torch.randn(8, 4).to(device)
            optimizer.zero_grad()
            loss = ((ddp(x) - y) ** 2).mean()
            loss.backward()
            optimizer.step()
        states[rank] = {
            name: param.detach().cpu() for name, param in model.state_dict().items()
        }

    mojo_dist.spawn(worker, world_size=world)

    assert states[0].keys() == states[1].keys()
    for name in states[0]:
        torch.testing.assert_close(states[0][name], states[1][name])


def test_ddp_ranks_stay_synchronized_async(monkeypatch):
    """The comm-stream (async) allreduce path keeps DDP ranks synchronized."""
    require_two_gpus()
    monkeypatch.setattr(mojo_dist, "_ASYNC_COMM_ENABLED", True)
    states = {}

    def worker(rank: int, world_size: int, pg: mojo_dist.MojoProcessGroup):
        device = f"mojo:{rank}"
        model = make_model().to(device)
        ddp = DDP(model, process_group=pg, device_ids=None)
        optimizer = torch.optim.SGD(ddp.parameters(), lr=0.05)
        torch.manual_seed(700 + rank)
        for _ in range(3):
            x = torch.randn(8, 16).to(device)
            y = torch.randn(8, 4).to(device)
            optimizer.zero_grad()
            ((ddp(x) - y) ** 2).mean().backward()
            optimizer.step()
        states[rank] = {n: p.detach().cpu() for n, p in model.state_dict().items()}

    mojo_dist.spawn(worker, world_size=2)

    for name in states[0]:
        torch.testing.assert_close(states[0][name], states[1][name])


def test_ddp_matches_single_process_large_batch():
    """DDP over shards must match one model trained on the full batch."""
    require_two_gpus()
    world = 2
    per_rank_batch = 8
    steps = 3
    lr = 0.05

    torch.manual_seed(55)
    xs = [torch.randn(world * per_rank_batch, 16) for _ in range(steps)]
    ys = [torch.randn(world * per_rank_batch, 4) for _ in range(steps)]

    # Single-process reference on the concatenated batch (mojo:0).
    reference = make_model().to("mojo:0")
    optimizer = torch.optim.SGD(reference.parameters(), lr=lr)
    for x, y in zip(xs, ys):
        optimizer.zero_grad()
        loss = ((reference(x.to("mojo:0")) - y.to("mojo:0")) ** 2).mean()
        loss.backward()
        optimizer.step()
    expected = {n: p.detach().cpu() for n, p in reference.state_dict().items()}

    states = {}

    def worker(rank: int, world_size: int, pg: mojo_dist.MojoProcessGroup):
        device = f"mojo:{rank}"
        model = make_model().to(device)
        ddp = DDP(model, process_group=pg, device_ids=None)
        opt = torch.optim.SGD(ddp.parameters(), lr=lr)
        lo, hi = rank * per_rank_batch, (rank + 1) * per_rank_batch
        for x, y in zip(xs, ys):
            opt.zero_grad()
            loss = ((ddp(x[lo:hi].to(device)) - y[lo:hi].to(device)) ** 2).mean()
            loss.backward()
            opt.step()
        states[rank] = {n: p.detach().cpu() for n, p in model.state_dict().items()}

    mojo_dist.spawn(worker, world_size=world)

    for name in expected:
        for rank in range(world):
            torch.testing.assert_close(
                states[rank][name], expected[name], rtol=1e-3, atol=1e-4
            )


def test_ddp_on_all_gpus():
    """Smoke test: DDP across every GPU on the box stays synchronized."""
    require_two_gpus()
    world = gpu_count()
    states = {}

    def worker(rank: int, world_size: int, pg: mojo_dist.MojoProcessGroup):
        device = f"mojo:{rank}"
        model = make_model().to(device)
        ddp = DDP(model, process_group=pg, device_ids=None)
        opt = torch.optim.SGD(ddp.parameters(), lr=0.05)
        torch.manual_seed(300 + rank)
        x = torch.randn(4, 16).to(device)
        y = torch.randn(4, 4).to(device)
        opt.zero_grad()
        ((ddp(x) - y) ** 2).mean().backward()
        opt.step()
        states[rank] = {n: p.detach().cpu() for n, p in model.state_dict().items()}

    mojo_dist.spawn(worker, world_size=world)

    for name in states[0]:
        for rank in range(1, world):
            torch.testing.assert_close(states[rank][name], states[0][name])


def test_async_allreduce_is_host_nonblocking(monkeypatch):
    """The comm-stream allreduce must return a pending Work immediately,
    without draining previously queued default-stream work (M3 overlap)."""
    require_two_gpus()
    monkeypatch.setattr(mojo_dist, "_ASYNC_COMM_ENABLED", True)
    import time

    measured = {}
    filler_repeats = 400

    def worker(rank: int, world_size: int, pg: mojo_dist.MojoProcessGroup):
        device = f"mojo:{rank}"
        # Allocation-free GPU filler: in-place adds on a big tensor. A
        # fresh-allocating filler (matmul chains) can block the HOST in the
        # stream-ordered allocator and skew the timing.
        filler = torch.zeros(1 << 24, device=device)
        t = torch.full((1 << 20,), float(rank + 1), device=device)
        # Warm every kernel (filler, allreduce path, copy-back).
        filler.add_(1.0)
        pg.allreduce([t]).wait()
        t.fill_(float(rank + 1))
        torch.mojo.synchronize(rank)
        pg.barrier().wait()

        # Calibrate the filler, synchronized.
        start = time.perf_counter()
        for _ in range(filler_repeats):
            filler.add_(1.0)
        torch.mojo.synchronize(rank)
        chain_seconds = time.perf_counter() - start
        pg.barrier().wait()

        # Queue the same filler WITHOUT syncing, then allreduce: the host
        # call must return long before the queued work could have drained.
        for _ in range(filler_repeats):
            filler.add_(1.0)
        start = time.perf_counter()
        work = pg.allreduce([t])
        call_seconds = time.perf_counter() - start
        work.wait()
        measured[rank] = (chain_seconds, call_seconds, t.cpu())

    mojo_dist.spawn(worker, world_size=2)

    expected = torch.full((1 << 20,), 3.0)
    for rank in range(2):
        chain_seconds, call_seconds, reduced = measured[rank]
        assert call_seconds < chain_seconds * 0.5, (
            f"rank {rank}: allreduce call took {call_seconds:.4f}s vs "
            f"queued-filler {chain_seconds:.4f}s — it drained the default stream"
        )
        torch.testing.assert_close(reduced, expected)


def test_async_allreduce_failure_poisons_future_not_process(monkeypatch):
    """A bad collective must surface from the future, not abort mid-backward."""
    require_two_gpus()
    monkeypatch.setattr(mojo_dist, "_ASYNC_COMM_ENABLED", True)

    def worker(rank: int, world_size: int, pg: mojo_dist.MojoProcessGroup):
        # Shape mismatch across ranks: rank 0 has 8 elements, rank 1 has 4.
        t = torch.ones(8 if rank == 0 else 4, device=f"mojo:{rank}")
        work = pg.allreduce([t])
        # The poisoned future completes with the exception. Upstream
        # semantics limit how it surfaces: Work.wait() never rethrows, and
        # a future re-wrapped through Work.get_future() loses the Python
        # wrapper's unwrap hook, so its wait() RETURNS the exception object
        # (DDP's C++ finalize path turns that into a loud parse error on
        # the rank thread). The essential contract: no hang, no abort.
        work.wait()
        outcome = work.get_future().wait()
        assert isinstance(outcome, ValueError)
        assert "shape mismatch" in str(outcome)

    mojo_dist.spawn(worker, world_size=2)


def test_async_allreduce_wait_inside_comm_hook_does_not_deadlock(monkeypatch):
    """Host-waiting the async Work INSIDE backward (the standard comm-hook
    idiom) must complete via the watcher, not deadlock: the engine callback
    that normally resolves the future cannot run while backward is blocked."""
    require_two_gpus()
    monkeypatch.setattr(mojo_dist, "_ASYNC_COMM_ENABLED", True)
    states = {}

    def worker(rank: int, world_size: int, pg: mojo_dist.MojoProcessGroup):
        device = f"mojo:{rank}"
        model = make_model().to(device)
        ddp = DDP(model, process_group=pg, device_ids=None)

        def hook(state, bucket):
            tensor = bucket.buffer()
            pg.allreduce([tensor]).wait()  # host-wait mid-backward
            fut = torch.futures.Future()
            fut.set_result(tensor)
            return fut

        ddp.register_comm_hook(None, hook)
        opt = torch.optim.SGD(ddp.parameters(), lr=0.05)
        torch.manual_seed(900 + rank)
        for _ in range(2):
            x = torch.randn(8, 16).to(device)
            y = torch.randn(8, 4).to(device)
            opt.zero_grad()
            ((ddp(x) - y) ** 2).mean().backward()
            opt.step()
        states[rank] = {n: p.detach().cpu() for n, p in model.state_dict().items()}

    mojo_dist.spawn(worker, world_size=2)

    for name in states[0]:
        torch.testing.assert_close(states[0][name], states[1][name])


def test_spawn_propagates_worker_exception():
    require_two_gpus()

    def worker(rank: int, world_size: int, pg: mojo_dist.MojoProcessGroup):
        if rank == 1:
            raise RuntimeError("boom from rank 1")

    with pytest.raises(RuntimeError, match="boom from rank 1"):
        mojo_dist.spawn(worker, world_size=2)


def test_spawn_does_not_hang_when_peer_dies_mid_collective():
    """A rank crashing while another waits in a collective must poison the
    rendezvous and surface the original exception, not hang forever."""
    require_two_gpus()

    def worker(rank: int, world_size: int, pg: mojo_dist.MojoProcessGroup):
        if rank == 1:
            raise RuntimeError("rank 1 died before the collective")
        # Rank 0 parks in a collective that rank 1 will never join.
        pg.barrier().wait()

    with pytest.raises(RuntimeError, match="rank 1 died before the collective"):
        mojo_dist.spawn(worker, world_size=2)


def test_spawn_fails_fast_after_peer_exit():
    """Collectives issued after a peer already exited raise immediately."""
    require_two_gpus()
    observed = {}

    def worker(rank: int, world_size: int, pg: mojo_dist.MojoProcessGroup):
        if rank == 1:
            raise RuntimeError("rank 1 exits early")
        try:
            pg.barrier().wait()
        except RuntimeError as exc:
            observed["first"] = str(exc)
        with pytest.raises(mojo_dist.RankExitedError):
            pg.barrier().wait()

    with pytest.raises(RuntimeError, match="rank 1 exits early"):
        mojo_dist.spawn(worker, world_size=2)
    assert "exited" in observed["first"]
