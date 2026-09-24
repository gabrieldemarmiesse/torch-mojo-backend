# Two-node mojoccl validation and bench

`run_two_node_checks.sbatch` is a 2-node x 8-GPU (16-rank) SLURM job that
exercises the mojo distributed backend (`torch_mojo_backend/distributed/`)
both with real NCCL (`TORCH_MOJO_BACKEND_CCL` unset, called "vendor" below)
and with `mojoccl` (`TORCH_MOJO_BACKEND_CCL=mojo`, the in-repo Mojo
re-implementation of NCCL's C ABI — see the "Mojo collectives" section of
`agents_docs/distributed.md`). Background reading: `AGENTS.md`, `agents_docs/distributed.md`,
`agents_docs/mojo_collectives_feasibility.md` (especially §2 traffic profile, §5.1
and §5.5 for the single- and two-node NCCL reference numbers, and §7/§8 for
the multi-node design this job is meant to validate once it lands).

## What it runs

One `sbatch` job, three phases, always in this order:

**A. `tests/ddp_worker.py`, 16 ranks.** For each `ccl` in `vendor`, `mojo`:
run modes `collectives`, `ddp_parity`, `stress` (one `torchrun
--nnodes=2 --nproc-per-node=8` launch per mode). `stress` is a no-op PASS
under vendor NCCL (it self-skips — see `run_stress` in `ddp_worker.py`); it
is mojoccl-specific regression coverage ported from the kernel harness.

**B. `ar_bench_gpt2.py` allreduce bench, ABBA order:** `vendor, mojo, mojo,
vendor` — the AGENTS.md-documented ordering that cancels a thermal/clock
ramp to first order. Prints one `RESULT ccl=... dtype=... size_mib=...
median_us=... min_us=... busbw_gbs=...` line per (ccl, dtype, size) at 1,
9, 27, 168, 512 MiB, fp32 and bf16. Only the vendor legs run with
`NCCL_DEBUG=INFO NCCL_DEBUG_SUBSYS=TUNING` (algo/protocol choices — the
mojo legs don't go through NCCL's tuner, and mojo's `TORCH_MOJO_BACKEND_CCL`
path talks to `libmojoccl.so`, not NCCL, so the flag can't tell us anything
about it).

**C. `demo_scripts/nanogpt_ddp.py --device mojo`, 40 steps, ABBA order,**
same shape as B: `vendor, mojo, mojo, vendor`. Fixed args:
`--nanogpt-path /home/gabriel/ddp_work/nanoGPT --data-dir
/home/gabriel/ddp_work/nanoGPT/data/shakespeare --log-interval 1
--eval-iters 10 --seed 1337 --max-iters 40 --eval-interval 40` (an eval,
hence a val-loss line, is printed once at step 40).

Every leg is a single `torchrun` launch spanning both nodes
(`--rdzv-backend=c10d`, `MASTER_ADDR` from `scontrol show hostnames`), run
through `srun bash -c "... flock /tmp/gpu_lock_0.lock uv run --no-sync
torchrun ..."` — `srun` puts one such shell on each node (`--ntasks-per-node=1`
from the `#SBATCH` header), so the flock is local to that node's `/tmp` and
guards that node's 8 GPUs against any other job touching them concurrently.
`PYTHONPATH` is pinned to this worktree
(`/home/gabriel/ddp_work/mojo_coll_mn`) so `ar_bench_gpt2.py`, which lives
outside the repo under `/home/gabriel/ddp_work/mojo_collectives/`, resolves
`torch_mojo_backend` from here rather than whatever else might be on the
path.

## RUN_MOJO

`RUN_MOJO=0` (env var, e.g. `sbatch --export=ALL,RUN_MOJO=0
run_two_node_checks.sbatch`) skips every mojo-ccl leg: phase A's `ccl=mojo`
iteration is skipped outright, and phases B and C run their vendor leg(s)
only (no ABBA — there is nothing to interleave against). Default
(`RUN_MOJO=1` or unset) attempts every leg.

`torch_mojo_backend/distributed/mojoccl/` supports multiple nodes:
`ncclGetUniqueId` encodes a TCP rendezvous (`{ipv4, port, magic}` of a
listening socket this library opens itself, not a `/dev/shm` path), and the
inter-node hop is GPUDirect RDMA written directly over the fabric —
`bootstrap.mojo` plus one transport-neutral engine (`internode.mojo`) on
either libibverbs (`ibverbs.mojo`, InfiniBand) or libfabric
(`libfabric.mojo`, HPE Slingshot / `cxi`), chosen at `ncclCommInitRank` time
or forced with `MOJOCCL_NET=verbs|fabric` — no vendor collective library at
any level. See the "Multi-node" subsection of
`agents_docs/distributed.md` for the design. `RUN_MOJO` therefore **defaults to
`1`**: leave it unset to exercise the mojo legs end to end at 16 ranks.
`RUN_MOJO=0` still exists to get a vendor-only NCCL reference run without
spending the job's time budget on the mojo legs (useful when only NCCL's
numbers are wanted, or while iterating on something unrelated to mojoccl).

## ring_pressure.py

`ring_pressure.py` is a two-node regression test for one specific way the
inter-node transport can be misused from above: issuing collectives faster
than the GPU consumes them until the calling thread laps `internode.mojo`'s
fixed work ring. DDP's `_sync_module_states` does exactly that, and on
Slingshot (where an exchange costs milliseconds rather than microseconds) it
killed `DDP(model)` at construction. `N` broadcasts, no synchronize until the
end:

```bash
N=1500 MIB=4 MOJOCCL_IB_TRACE=1 torchrun --nnodes=2 --nproc-per-node=4 \
    --rdzv-backend=c10d --rdzv-endpoint=$MASTER:29500 \
    tests/multinode/ring_pressure.py
```

It passes when every rank prints `payload=OK`; the trace line then says how
far the host ran ahead and how many times it waited for a ring slot. Needs
two nodes and takes a second.

## bucket_loop.py

`bucket_loop.py` issues DDP's gradient buckets as plain allreduces -- same
sizes, same order, one synchronize per step -- with optional concurrent
compute (`MM`) and rank skew (`SKEW`). It is the harness for separating "the
collectives are slow" from "something around the collectives is slow", which
a size sweep cannot do:

```bash
BUCKETS=1,25,25,25,25,25,90 STEPS=10 MM=1 SKEW=3 torchrun --nnodes=2 \
    --nproc-per-node=4 ... tests/multinode/bucket_loop.py
```

Measured on 2 nodes x 4 MI300A over cxi (216 MiB of buckets per step): 6.0
ms/step plain, 8.8 with one matmul per bucket, 36.8 with a rank-skewed three
-- against 2800 ms/step for a nanoGPT DDP step moving the same bytes.

## enqueue_bench.py, ddp_buckets.py, run_enqueue_probe.sbatch

`enqueue_bench.py` times the host side of `dist.all_reduce` -- the wall time
of the Python call, nothing synchronized, 100 calls per size, both CCLs --
because that, not the collective's device time, is what starves the compute
stream when DDP issues 146 allreduces per step from the autograd thread.
`ddp_buckets.py` prints the bucket sizes the reducer actually hands the
process group for a nanoGPT model (recorded at the process group on the
second step, after DDP has rebuilt its buckets in autograd order; GPT-2 XL:
144 × 39 MiB then 313 MiB), in the form `bucket_loop.py`'s `BUCKETS=` takes.
`deadline_probe.py` puts one rank to sleep past `MOJOCCL_IB_TIMEOUT_S` and
checks that every other rank raises within the deadline plus the fused
kernel's grid grace, under a watchdog; `small_region_probe.py` allreduces
20 and 129 MiB on a `MOJOCCL_REGION_MB=1` region, the second of which is
more chunks than the inter-node work ring holds and has to take the split
schedule. `run_enqueue_probe.sbatch` runs the first two, `bucket_loop.py` on that list, a
`MOJOCCL_IB_TRACE=1` GPT-2 XL step and `ring_pressure.py` in one two-node
job, with the enqueue legs in ABBA order; run it before and after a change
to the collectives' launch structure.

## GPU-free self-tests

`tests/multinode/selftest/` holds seven standalone Mojo programs that
exercise the TCP bootstrap (`bs_test.mojo`), the RDMA transport
(`ib_bringup.mojo`), the pipelined transport and its credit-based flow
control (`ib_pipeline.mojo`), the region geometry (`geometry_test.mojo`),
the `SCM_RIGHTS` fd transport (`fd_exchange.mojo`), the socket deadlines
(`sock_deadline.mojo`) and the libfabric ABI (`fabric_abi.mojo`, against
`fabric_abi.c` compiled by gcc). The two transport ones run against either
backend — `MOJOCCL_NET=verbs` between processes on any host with an ACTIVE
InfiniBand port (the SLURM login node included, where there is one),
`MOJOCCL_NET=fabric` between processes on any host with a libfabric RMA
provider (an Adastra **compute node**, which has four Slingshot NICs and no
InfiniBand at all) — and the rest need no NIC, so they all run in seconds
without a GPU. They caught six real bugs (bootstrap/QP wiring, resource
leaks on a failed `ib_setup`, a silently-misread port LID) before any GPU
time was spent chasing them; run them before and after any change to
`torch_mojo_backend/distributed/mojoccl/{bootstrap,ibverbs,libfabric,netutil,internode}.mojo`
or to the region layout in `mojoccl.mojo`. `fabric_hmem.mojo` is the one
self-test that does need a GPU: it is the only place the FI_HMEM_ROCR
registration of a `driver.alloc_region` allocation is exercised. See
`tests/multinode/selftest/README.md` for build and run commands.

## Output

`-o /home/gabriel/ddp_work/logs/mojoccl_2node_%j.log` is the **job log** —
node names, per-node SM clock, branch/commit, then, per leg, a `===
<phase> ccl=... ... ===` header, a filtered view of that leg's output
(`[rank N] OK/FAIL ...` lines for ddp_worker, `RESULT ...` lines for the
bench, `step ...`/`val loss ...` lines for training), and a `=== <phase>
ccl=... ... exit: N ===` trailer with that leg's `torchrun` exit code. This
is the file `summarize.py` reads.

Each leg's **full, unfiltered** output (including every per-step line and,
for the vendor bench legs, the full `NCCL_DEBUG=INFO` tuning log) is also
saved to its own file under `/home/gabriel/ddp_work/logs/`, e.g.
`mojoccl_2node_<jobid>_worker_vendor_collectives.log`,
`mojoccl_2node_<jobid>_arbench_vendor_leg1.log`,
`mojoccl_2node_<jobid>_nanogpt_mojo_leg2.log`.

## summarize.py

```bash
uv run --no-sync python tests/multinode/summarize.py \
    /home/gabriel/ddp_work/logs/mojoccl_2node_<jobid>.log
```

Stdlib only (no torch import — runs fine on the login node). Parses the job
log's `RESULT`/`=== ... ===`/`step ...` lines (regexes at the top of the
file) into one markdown report with three tables: allreduce bench (per
dtype/size, vendor and mojo median device time and the mojo/vendor ratio —
median-of-medians if a ccl has more than one ABBA leg at that size), the six
`ddp_worker` pass/fail results, and the training runs' step-40 loss, val
loss and tok/s with pass/fail. Pass `--out FILE` to write the report to a
file instead of stdout; multiple log paths may be given (e.g. to merge a
job log with one of the per-leg logs) and are concatenated before parsing.

## Vendor-only NCCL reference

The first run of this job should be `RUN_MOJO=0`, to establish the current
NCCL reference numbers at 16 ranks before the mojoccl transport exists —
see `/home/gabriel/ddp_work/mojo_collectives/mn/NCCL_REFERENCE_16.md` for
node names, SM clock, NCCL algo/protocol choices and the resulting numbers
from that run.

## e2e_three_stacks for other nanoGPT sizes

`e2e_three_stacks.sbatch` runs from any checkout (`MOJO_TREE`, default the submitting one) and
takes the nanoGPT config from `MODEL_ARGS`; the log-name prefix `E2E_TAG` keeps runs apart and
the summariser reads the same tag. GPT-2 XL, which fits batch 8 only on 80 GB under DDP:

```bash
sbatch --export=ALL,BATCHES=8,E2E_TAG=e2e_gpt2xl,MODEL_ARGS="--n-layer 48 --n-head 25 --n-embd 1600 --bias" \
    tests/multinode/e2e_three_stacks.sbatch
E2E_TAG=e2e_gpt2xl E2E_MODEL="nanoGPT GPT-2 XL (1.5B)" \
    uv run --no-sync python tests/multinode/summarize_three_stacks.py <jobid>
```

Each batch size now starts with one discarded warm-up per stack, which doubles as the fit gate
(a size whose warm-up fails on any stack is skipped), and a 1-rank mojo prewarm builds the
kernel cache before the 16-rank runs race for it.

## e2e_three_stacks on Adastra (2 x 4 MI300A, Slingshot)

`e2e_three_stacks_adastra.sh` is the same protocol as `e2e_three_stacks.sbatch`
(discarded warm-up, five interleaved rounds per batch size, the evidence lines
the summariser checks) for the CINES Adastra MI300A partition: 8 ranks on two
4-APU nodes, RCCL through the site's `aws-ofi-rccl` plugin over libfabric
`cxi`, mojoccl over its own libfabric transport. It runs as a batch script
(`sbatch --nodes=2 --exclusive ... e2e_three_stacks_adastra.sh`) or from a
login node against an existing allocation (`J=<jobid> ...`). `E2E_ROOT` points
at the scratch tree holding the checkout, the two venvs (`venv-cpu` with the
CPU torch wheel for the mojo stacks, `torch-rocm` for stock) and nanoGPT;
`ENDURANCE=1` adds one 1500-step run per stack at batch 32. Two AMD facts it
encodes: RCCL inside the mojo backend needs
`MODULAR_DEVICE_CONTEXT_MEMORY_MANAGER_VMM=1` on two MI300A nodes (without it
`ncclCommInitRank` fails in the OFI plugin's memory registration), and the
`srun` step must expose the whole node (`--cpus-per-task=192`) or
`rank_bind.py` cannot reach the NUMA nodes of GPUs 2 and 3.

Summarise with the log directory, world size and labels of that cluster:

```bash
E2E_STOCK_NAME="stock ROCm torch 2.9.1 + RCCL" E2E_SETUP="8 ranks on 2x4 MI300A" \
  uv run --no-sync python tests/multinode/summarize_three_stacks.py <jobid> $E2E_ROOT/logs 8
```
