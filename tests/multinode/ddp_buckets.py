"""Print the gradient bucket sizes DDP actually issues for a nanoGPT model.

`bucket_loop.py` replays a DDP step's allreduce traffic, and it is only a
replay if its BUCKETS are the reducer's own. Those are not the ones
`reducer._get_zeros_like_grad_buckets()` reports before the first backward
(one bucket holding every parameter); the real ones are whatever the reducer
hands the process group, so this records them at the process group, on the
second step -- after DDP has rebuilt its buckets in autograd order::

    torchrun --nproc-per-node=1 tests/multinode/ddp_buckets.py \\
        --nanogpt-path ~/nanoGPT --n-layer 48 --n-head 25 --n-embd 1600 --bias

One rank is enough: bucketing does not depend on world size. It prints the
MiB list in issue order and the comma-separated form to paste into `BUCKETS=`.
"""

from torch_mojo_backend.distributed import use_local_rank_gpu

use_local_rank_gpu()

import argparse  # noqa: E402
import datetime  # noqa: E402
import sys  # noqa: E402
from pathlib import Path  # noqa: E402

import torch  # noqa: E402
import torch.distributed as dist  # noqa: E402
from torch.nn.parallel import DistributedDataParallel as DDP  # noqa: E402

from torch_mojo_backend import register_mojo_devices  # noqa: E402


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--nanogpt-path", type=Path, required=True)
    p.add_argument("--n-layer", type=int, default=48)
    p.add_argument("--n-head", type=int, default=25)
    p.add_argument("--n-embd", type=int, default=1600)
    p.add_argument("--block-size", type=int, default=1024)
    p.add_argument("--batch-size", type=int, default=1)
    p.add_argument("--vocab-size", type=int, default=50304)
    p.add_argument("--bias", action="store_true", default=False)
    args = p.parse_args()

    register_mojo_devices()
    dist.init_process_group(backend="mojo", timeout=datetime.timedelta(seconds=900))
    sys.path.insert(0, str(args.nanogpt_path))
    from model import (  # ty: ignore[unresolved-import] -- nanoGPT, from --nanogpt-path  # noqa: PLC0415 -- reachable only via the sys.path insert above
        GPT,
        GPTConfig,
    )

    torch.manual_seed(1337)
    device = torch.device("mojo", torch.accelerator.current_device_index())
    config = GPTConfig(
        block_size=args.block_size,
        vocab_size=args.vocab_size,
        n_layer=args.n_layer,
        n_head=args.n_head,
        n_embd=args.n_embd,
        dropout=0.0,
        bias=args.bias,
    )
    model = GPT(config).to(device)
    ddp_model = DDP(model, broadcast_buffers=False)

    # Patched on the CLASS, not the instance: the C++ reducer reaches a
    # Python backend through pybind's override lookup, which reads the type's
    # attribute rather than anything bound at registration.
    pg = dist.distributed_c10d._get_default_group()._get_backend(device)
    seen = []
    cls = type(pg)
    original = getattr(cls, "allreduce")

    def record(self, tensors, *a, **kw):
        t = tensors[0]
        seen.append(t.numel() * t.element_size())
        return original(self, tensors, *a, **kw)

    setattr(cls, "allreduce", record)

    idx = torch.randint(
        0, config.vocab_size, (args.batch_size, args.block_size + 1), device=device
    )
    x, y = idx[:, :-1].contiguous(), idx[:, 1:].contiguous()
    for step in range(2):
        seen.clear()
        with torch.autocast(device_type=device.type, dtype=torch.bfloat16):
            _, loss = ddp_model(x, y)
        loss.backward()
        torch.accelerator.synchronize()
        ddp_model.zero_grad(set_to_none=True)
        if dist.get_rank() == 0:
            mib = [b / 1024 / 1024 for b in seen]
            print(
                f"RESULT step={step} calls={len(mib)}"
                f" total_mib={sum(mib):.1f}"
                f" min_mib={min(mib) if mib else 0:.3f}"
                f" max_mib={max(mib) if mib else 0:.3f}",
                flush=True,
            )
            print(
                f"RESULT step={step} logged_bucket_sizes="
                + str(ddp_model._get_ddp_logging_data().get("bucket_sizes")),
                flush=True,
            )
            if step == 1 and mib:
                print("RESULT BUCKETS=" + ",".join(f"{m:.3f}" for m in mib), flush=True)
    dist.destroy_process_group()


if __name__ == "__main__":
    main()
