# The kernel-call queue and the Into protocol

## Problem

Eager-mode kernels are Mojo CPython extensions compiled on demand, one op
per `.so`. A cold start therefore interleaves *discovering* which kernels a
workload needs with *building* them. If a compile miss blocks the
dispatching thread, discovery serializes behind every build and a zero-cache
GPT-2 training step pays for each unit's build latency in sequence.

## Design

The deferral boundary is the **extension call**: the point where an op is
nothing but raw pointers, shapes, dtypes and a `DeviceContext`. Everything
above it — autograd graph recording, version counters, view metadata,
layouts, output allocation — executes synchronously in Python at dispatch
time. Only the device kernel **launch** may lag, waiting in a FIFO
(`eager_kernels/call_queue.py`) while its unit builds in the background
pool. This is CUDA's async-stream model applied to kernel *compilation*:
the host runs ahead; reads synchronize.

Choosing this boundary is what keeps the machinery small. An earlier
prototype deferred whole aten ops and had to reconstruct torch-level
semantics at replay (meta-inferred placeholders, alias maps, version
fix-ups, backend-vs-meta layout mismatches); at the call boundary none of
those exist, because torch-level semantics were never deferred.

### Rules

1. **Strict FIFO.** Once any launch is queued (a compile miss), every
   subsequent launch queues behind it, warm or not: device work must land
   in program order. `pump()` — called at every dispatch entry — launches
   the longest ready prefix.
2. **Host reads drain.** D2H transfers, `read_scalar`, the `_SYNC_OPS` set
   (`item`, `equal`, `nonzero`, device-crossing copies …) and device
   `synchronize()` call `drain()` first. Out-of-dispatch payload readers
   (the AutogradPrivateUse1 sdpa impl) use `deferred_compile.wait_for`.
3. **Keep-alive.** Queued items hold raw pointers, not references. The
   dispatch layer brackets each aten op (`op_begin`/`op_end`) and retains
   its arguments, results and every allocation made inside
   (`TorchMojoTensor._alloc` reports into the bracket) while the queue is
   non-empty. The keep-alive list releases only from `drain()`: holder
   frees are stream-ordered per enqueuing thread, so they must follow the
   launches on the draining thread.
4. **Thread order.** Device work is ordered only within an enqueuing
   thread (verified empirically: cross-thread enqueues read stale data).
   `_direct` and the queue executor synchronize the device whenever the
   enqueuing thread changes — rare (main ↔ autograd engine).
5. **Launch-time dtype escalation.** A queued launch that raises
   "unsupported dtype" rebuilds its unit with the full dtype set and
   retries that one call. Errors of any other kind are held and re-raised
   at the next drain, CUDA-style.
6. **Ungated device calls ride the queue.** fa4 attention, `copy_d2d`,
   `StridedFill` and strided copies are always launch-ready but must hold
   their FIFO position (`external_call`), or they would read queued
   producers' outputs early.

## The Into protocol

Spec-tier entry points historically allocate their output inside Mojo and
return `(holder, spec, …)` — a value the caller consumes immediately, which
makes the call impossible to queue. The **Into** form removes the
allocation from Mojo: Python computes the output metadata (broadcast
shapes, dtype promotion, reduced shapes), allocates via
`tensor_holder.alloc` (kernel-free, never a drain), and passes the
out-spec as one extra trailing argument. The launch writes into it and
returns nothing.

Conventions:

- **Same registration name, `nargs`-branched dispatcher.** `AddSpec(a, b)`
  is the legacy allocate-and-return form; `AddSpec(a, b, out)` is Into.
  Adding a spec op means adding both forms (the hygiene of one entry name
  per op is preserved; nothing about gating or the demand profile changes).
- **Eligibility is replicated in Python, conservatively.** A queued launch
  cannot fall back, so `aten_fast` mirrors each family's Mojo prologue
  (dtype sets, dim validity, matmul contiguity/rank rules) and anything
  uncertain declines to the legacy synchronous call. The Into go-functions
  still validate (extent, dtype, device, contiguity) and raise loudly.
- Converted families: binary logic (all 20), the fused f32+bf16 add,
  `CastSpec`, `FillSpec`, elementwise unary/unary-bool/scalar/int-scalar,
  rowred/`Argmin`/`Var`/`Any`/`All`/`LogSoftmax` and the two-output
  `MinDim`, nn `Mean`/`Max`/`Argmax`/`Softmax`, and
  `Matmul`/`MatmulBias`/`Bmm`. Deliberately legacy-only: `Cumsum`,
  `BatchNorm`, `AttnDecode` (geometry gates not Python-mirrored; their
  calls drain and run synchronously).

## Modes

| Mode | How | Behavior |
|---|---|---|
| default | — | call queue on: misses queue, builds pool, reads drain |
| sequential | `TMB_NO_TRIGGER_DEFER=1` | every miss blocks inline |
| kill switch | `TMB_KERNEL_QUEUE=0` | same as sequential |
| test suite | `TORCH_MOJO_BACKEND_TESTING=1` | queue off (synchronous call-count/spy contracts); `TMB_FORCE_KERNEL_QUEUE=1` re-enables per-test (see `tests/test_call_queue.py`) |

## Measured (GPT-2 124M b32, one training step, zero kernel cache, H100)

| Transform cache | sequential | call queue + Into |
|---|---|---|
| warm | 74.1 s | **26.4 s** |
| cold (fresh `MODULAR_CACHE_DIR`) | 295.9 s | **129.9 s** |

With the `.so` files present no builds happen at all (~8 s total).

Known issue: the full nanoGPT loop's step-1 evaluation differs from the
sequential oracle by ±1e-4 on one of train/val loss.
