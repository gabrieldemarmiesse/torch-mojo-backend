# Multi-GPU: dispatch-ceiling attribution and the next structural bets

Written 2026-08-07, after merging `main`'s kernel-call queue into
`multi-gpu-m0` and sharding it per device. Everything below was measured
on this cluster (8×H100 NVLink nodes, driver 570/CUDA 12.8, max nightly
26.5.0.dev2026072306) on the merged branch, or verified against the
pinned pytorch v2.11.0 / modular f807c5a3 (2026-07-23) sources.

## 1. The attribution experiment: where the dispatch ceiling actually is

`demo_scripts/multi_gpu_dispatch_attribution.py` runs the dispatch bench's
op mix three ways: one thread round-robining over N devices, one thread
per device (the DDP execution model), and N single-GPU *processes*
(`CUDA_VISIBLE_DEVICES=i`, common start barrier). Scaling efficiency
(baseline = that mode's own 1-device run), 50 steps, 7 ops/step:

| mode        | shapes            | 2 GPU | 4 GPU | 8 GPU |
|-------------|-------------------|------:|------:|------:|
| roundrobin  | batch 4096, d1024 | 1.00  | 0.99  | 0.89  |
| threads     | batch 4096, d1024 | 1.00  | 0.77  | 0.34  |
| processes   | batch 4096, d1024 | 0.81  | 0.53  | 0.37  |
| roundrobin  | tiny (32×64)      | 0.50  | 0.25  | 0.13  |
| threads     | tiny (32×64)      | 0.37  | 0.13  | 0.07  |
| processes   | tiny (32×64)      | 0.68  | 0.51  | 0.30  |

Readings:

- **A single Python thread can feed 8 GPUs at DDP-realistic shapes**
  (roundrobin 0.89 at 8; per-op budget ~44 µs). The runtime/driver side
  is not the binding constraint for one dispatcher at these shapes.
- **Thread-per-rank is the worst mode even with per-device queue locks.**
  The sharded queue removed every backend-side cross-rank mutex, and
  threads still collapse to 0.34 while roundrobin holds 0.89 *in the same
  process*. What remains between rank threads is the GIL and interpreter
  contention: the ceiling is interpreter-side.
- **Per-op host cost saturates one interpreter at small ops.** Tiny-shape
  roundrobin degrades exactly as 1/N with a flat ~58 µs/op: one
  interpreter issues ~17k ops/s total, however many devices wait.
- **Processes do not scale cleanly either** (0.37 at 8, and a fresh
  process pays a large fixed overhead we did not fully explain —
  its 1-device baseline is ~7× slower than in-process even after a
  warm-up round; treat the processes row as a lower bound). Some
  runtime/driver-side cross-process cost exists, but it is not what DDP
  hits first.

**Decision this buys:** levers that shard or release the interpreter win;
levers that only reduce runtime contention don't. Concretely: the
GIL-release/fan-out-in-Mojo direction (SPMD replicated tensors) and
fewer-launches directions (whole-step graphs, record/replay) pay;
free-threaded Python would attack the right layer but is blocked upstream
(§4), and more lock-sharding on our side is done — there are no shared
backend locks left between ranks.

## 1b. Head-to-head vs torch-cuda DDP (torchrun + NCCL, 1 process/GPU)

`demo_scripts/cuda_ddp_baseline.py` mirrors the mojo DDP demo exactly
(same 1024→4096→4096→1024 GELU MLP, fp32, AdamW, per-rank batch 2048,
20 timed steps, warmup excluded, wall = slowest rank) and runs it the
standard PyTorch way — one process per GPU, NCCL allreduce — from a
cu128 venv (`~/.cache/claude-cuda-baseline/venv`; the project venv's
cu130 torch cannot initialize this 12.8 driver). Same node, same H100s,
torch 2.11.0 on both sides, default matmul precision on both sides:

| stack                          | GPUs | wall (20 steps) | samples/s | per-GPU  |
|--------------------------------|-----:|----------------:|----------:|---------:|
| torch-cuda NCCL (1 proc/GPU)   |    1 |          0.135s |   302,346 |  302,346 |
| torch-cuda NCCL                |    2 |          0.147s |   558,048 |  279,024 |
| torch-cuda NCCL                |    8 |          0.150s | 2,188,090 |  273,511 |
| mojo DDP (thread-per-rank)     |    2 |          0.284s |   288,607 |  144,304 |
| mojo DDP, comm-stream async    |    2 |          0.286s |   286,582 |  143,291 |
| mojo DDP                       |    8 |          1.713s |   191,343 |   23,918 |
| mojo DDP, comm-stream async    |    8 |          1.887s |   173,627 |   21,703 |

Decomposition:

- **Per-GPU compute (the 2-GPU column, where our scaling is still ~1.0):
  ~2.1× slower than cuBLAS** on this fp32 MLP (14.2 ms/step vs
  6.8–7.4 ms). A kernel-level gap, not a dispatch gap — and fp32-specific
  context: the bf16 GEMM family is currently unbuildable on sm_90a (§6),
  so the dtype where the eager kernels are closest to parity is
  unavailable on H100 today.
- **Scaling: NCCL holds 0.90 at 8 GPUs; thread-per-rank collapses to
  ~0.17** (24k per-GPU at 8 vs 144k at 2). Worse than the raw dispatch
  bench's 0.34 because DDP adds per-bucket work the bench doesn't have:
  7 bucket rendezvous per step (park/wake 8 threads on condition
  variables under the GIL), the collective `_launch_lock`, and a
  drain-all before each launch.
- **Net: ~1.9× slower at 2 GPUs, ~11× at 8 GPUs.** The 8-GPU number is
  the interpreter ceiling made concrete: NCCL's processes each own a
  GIL; our eight rank threads share one. This is the gap the SPMD
  replicated-tensor direction (§3.1) exists to close — one thread
  issuing, fan-out in Mojo — with the per-GPU kernel gap as the second,
  independent term.

## 2. Priority comm streams: unblocked today (probe result)

The M3 overlap path uses a second `DeviceContext` per device because the
split launch path (`compile_function` + `DeviceStream.enqueue_function`)
cannot carry the collectives' capturing `output_lambda` under
`--emit shared-lib` (upstream_issues/modular-1). The probe refutes the
conclusion that this blocks priority streams:

- The pinned toolchain exposes `DeviceContext.create_stream(*, priority)`
  and `stream_priority_range()` (H100 range 0..-5), Mojo-side.
- The capture blocker is real — even the SUM variant substitutes a
  host-captured `default_output_lambda` — but a ~20-line **wrapper
  kernel** that builds the store epilogue *inside device code* from its
  runtime `result` argument has no capturing comptime parameters and
  compiles fine under `--emit shared-lib`.
- Verified on 2×H100: 1-stage and 2-stage allreduce launched via
  `compile_function` + priority-stream `enqueue_function`, correct
  results, and under a default-stream compute flood the priority=-5
  collective completes **34× sooner** than priority=0 (0.33 ms vs
  11.3 ms) with negligible impact on the compute.

Integration cost: add the wrapper kernel to `comm_ops.mojo` (the AVG
epilogue fits the same trick — pass `1/world` as a runtime arg),
replicate `_allreduce_p2p`'s 1-vs-2-stage dispatch host-side (~30 lines),
cache `compile_function` results per device. This removes the main reason
M3 is opt-in-because-slower, alongside persistent comm staging buffers.

## 3. Structural bets, ranked

1. **SPMD replicated tensors** (invert the execution model): one Python
   thread holds a tensor wrapping N per-device buffers; the N-way fan-out
   happens inside one Mojo extension call with the GIL released — the
   exact `comptime ngpus` + `GILReleased` + `_launch_device_collective`
   pattern `comm_ops.mojo` already uses for collectives, applied to
   compute. The attribution table is direct evidence for it: it turns the
   collapsing "threads" pattern into the scaling "roundrobin" pattern and
   divides per-op interpreter cost by N. Dispatch becomes O(1) in world
   size; collectives stop being a rendezvous (the one thread owns all
   replicas); the PrivateUse1 deviceCount()==1 autograd contortions
   become moot. Cost: a new tensor subclass + per-op N-buffer plumbing in
   aten_fast's hot path; start with the ~20 ops the DDP demo exercises.
2. **One multi-device MAX graph per step** (the torch.compile route): the
   compiler already opens the `InferenceSession` over *all* accelerators,
   `mojo_backend` already wraps with `aot_autograd` (backward compiles),
   and MAX exposes `ops.allreduce.sum` / `ops.allgather` at graph level
   (used by Modular's own multi-device serving pipelines) — none of which
   the graph builder uses yet. Replicating the joint graph N ways with
   collectives staged inline makes the MAX compiler, not Python, schedule
   comm against compute. Biggest single-lever win for compile-tolerant
   training; does nothing for eager-only users.
3. **Record/replay of a steady-state step from the kernel-call queue**:
   post-merge, a queued item is already `(compiled_unit, raw_args)` with
   no Python semantics left — a recording. The honest caveat found while
   sharding: raw args bake in *pointers*, and activations/temporaries get
   fresh allocations each step, so replay needs an address-stabilization
   story (a private allocation pool per recorded step, as CUDA graphs
   do) before it works. Params/grads are stable; a
   record-the-optimizer-step prototype (foreach ops, stable storage)
   is the cheap first slice. This is the only lever that also helps if
   any ceiling turns out driver-side, and it helps single-GPU decode.

## 4. Parallelism strategies beyond DDP

- **Tensor parallel / DTensor**: `_set_thread_isolation_mode(True)` is
  real in v2.11 — a rank-local c10d registry with autograd-thread
  forwarding (RankLocal keyed by the *forward* thread id), exactly what
  thread-per-rank TP needs. Caveats from reading the sources: it is
  test-harness API (only `MultiThreadedTestCase` uses it), the
  Python-side world state needs the `ThreadLocalWorld` half replicated,
  and DTensor has zero test coverage under thread isolation — expect to
  own bugs. Collective needs: vanilla Colwise+Rowwise TP needs only
  allreduce (which we have, kernel-backed); allgather enters with
  replicated-input reshuffles (kernel exists upstream, takes explicit
  `my_rank` already); reduce_scatter enters with SequenceParallel — and
  the pinned nightly's `reduce_scatter` gained `local_rank:
  Optional[Int]` (modular f807c5a3), so the "devices must be 0..n-1"
  blocker (upstream_issues/modular-3) is now stale for reduce_scatter
  and allgather; it still holds for allreduce and broadcast.
- **Pipeline parallel**: needs only pairwise send/recv — a (src, dst,
  tag) rendezvous over the existing `copy_d2d_peer`, no comm kernel, no
  PG splitting. It is also the one strategy that *reduces* per-rank
  dispatch (each rank runs 1/P of the layers), attacking the measured
  interpreter ceiling directly. Cheap to prototype on the current PG.
- **ZeRO-1**: works on the existing broadcast (copy_ loop) — no new
  kernels; per-rank optimizer states cut memory ~4× on Adam. Cheapest
  meaningful win.
- **Kernel-backed broadcast/allgather**: `comm/broadcast.mojo` and
  `comm/allgather.mojo` sit unused upstream; ours are copy_ loops
  (fine for DDP's one-time broadcast, wrong for ZeRO-3-style traffic).
  Adopting them inherits the default-stream restriction (modular-2) —
  or the §2 wrapper-kernel treatment.
- **FSDP2** stays deferred; before any shim, measure the per-collective
  cost of `_launch_lock` + drain under many small all-gathers — that
  serialization, not the Stream/Event surface, is the likely critical
  path.

## 5. Free-threaded Python: blocked one level down

`PythonModuleBuilder` uses single-phase init (`_PyModule_Create2`, with a
literal `# TODO: set gil stuff` in `_cpython.mojo`) and never declares
`Py_mod_gil`/`Py_MOD_GIL_NOT_USED`; slots cannot be attached to
single-phase modules at all, so supporting 3.13t means upstream moving to
multi-phase `PyModuleDef_Init`. Importing any Mojo extension under 3.13t
today silently re-enables the GIL. Worth filing as the next upstream
issue — but note the attribution result: even a working nogil would only
upgrade "threads" toward "roundrobin", which SPMD reaches without waiting
for CPython+Mojo to align.

## 6. Toolchain hazards found while validating (all fixed on the branch)

- The nightly bump broke every eager kernel load when `max.nn` is
  imported first (upstream_issues/modular-5, workaround shipped:
  tensor_holder is pinned before any `max.nn` import).
- `uv sync` had left the previous nightly's `max-mojo-mogg-libs`
  installed (the new nightly dropped it); mixed nightlies also segfault
  extension loads.
- `mojo.paths._build_mojo_source_package` (the issue-5495 workaround in
  conftest) rebuilds one fixed .mojoc path with no lock and no toolchain
  identity — pytest-xdist workers tear it, and after a nightly bump the
  losers of the race reuse a stale package. conftest now stamps and
  flocks it.
- Separately and more seriously: **graph mode is entirely broken in the
  Linux wheels of dev2026072306** — the nightly dropped
  `max-mojo-mogg-libs` (which shipped `builtin_kernels.mojoc` /
  `builtin_primitives.mojoc`) with no replacement, so every
  `max.engine` graph compile fails with `MAXG_addKernelPackage: failed
  to import kernels from ''` even with zero custom kernels
  (upstream_issues/modular-6). This is what the ~50
  `test_aten_functions` graph-mode failures on GPU nodes are; they
  reproduce identically on the pre-merge branch and pristine main.
  Eager mode (every mojo-device path, DDP included) is unaffected.
  Nothing actionable on our side except the upstream report and, if
  graph mode is needed sooner, pinning the toolchain back.

## 7. Ordered next steps

1. Land the §2 priority-stream comm path in `comm_ops.mojo` (+ persistent
   staging buffers) and re-measure M3 end-to-end; flip
   `TORCH_MOJO_BACKEND_COMM_STREAM` default if it wins.
2. Prototype SPMD replicated tensors over the DDP demo's op set; the
   attribution bench is the acceptance test (target: threads-mode work at
   roundrobin-mode efficiency).
3. ZeRO-1 + pipeline send/recv as cheap wins on the existing PG.
4. Multi-device-graph spike: N-replica joint graph with
   `ops.allreduce.sum` inline for the demo MLP; compare a full step
   against eager DDP.
5. File upstream: Py_mod_gil (multi-phase init) — plus the modular-5
   segfault already written up; update modular-3 (reduce_scatter/
   allgather now take explicit rank at the pinned nightly).
6. Record/replay only after 1–4: it needs the allocation-stabilization
   design first.
