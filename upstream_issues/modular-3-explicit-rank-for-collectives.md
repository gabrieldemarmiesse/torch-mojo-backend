# [kernels][comm] `allreduce`/`reducescatter`/`broadcast` derive the rank from `ctx.id()` — GPU subsets that are not exactly {0..n-1} deadlock or corrupt

## Summary

The collective kernels take `my_rank` as a plain runtime argument, but the
public entry points hardwire it to the global device ordinal:

- `comm/allreduce.mojo`: `_allreduce_p2p` launches with `Int(ctx.id())`
  (lines ~1073 and ~1101), and the naive fallback indexes
  `dev_inputs[my_rank]` and creates `DeviceContext(device_id=i)` for
  `i in 0..ngpus-1`;
- `comm/broadcast.mojo:422`: `my_rank = Int(ctx.id())`;
- `comm/reducescatter.mojo` likewise;
- `comm/allgather.mojo:507` is the exception — it already takes `my_rank`
  as an argument.

`my_rank` indexes `rank_sigs[my_rank]` (the per-rank Signal), the 2-stage
partition (`rank_start(my_rank)`), and peer buffers. The implicit contract —
visible in `max/kernels/test/gpu/comm/test_allreduce.mojo`, where slot `i`
always belongs to device id `i` — is that the participating devices are
exactly ids `0..ngpus-1`, in order.

## Consequence

An allreduce over any other subset breaks. Example on an 8-GPU box: a
4-GPU data-parallel job pinned to devices `{4,5,6,7}` (say, the other four
are running something else). Each instance computes `my_rank = ctx.id() ∈
{4..7}` while `ngpus = 4`:

- `rank_sigs[my_rank]` reads Signal slots 4..7 — the caller filled 0..3
  (`rank_sigs` is `InlineArray[..., MAX_GPUS]` with only `ngpus` slots
  initialized in the canonical pattern), so the barrier spins on
  uninitialized pointers → deadlock or fault;
- the 2-stage partitioning indexes out of the `ngpus`-sized logical space →
  data corruption even if the pointers happen to be valid.

In torch-mojo-backend we currently validate and reject anything that is not
devices `0..n-1` in order, which means single-process DDP cannot run on a
subset of the box's GPUs — a real deployment shape (fractional-node jobs,
one process group per model on a shared box).

## Ask

Add `my_rank: Int` (and, where the naive fallback builds its own contexts,
an explicit device list) to the public `allreduce`, `reducescatter` and
`broadcast` signatures, defaulting to today's `Int(ctx.id())` for backward
compatibility — mirroring what `allgather` already does. The kernels need no
changes: the rank is already a runtime kernel argument; only the dispatch
layer pins it to `ctx.id()`.

## Environment

modular @ `12beae6f14` (nightly 26.5.0.dev2026061806), 8x H100.
Consumer: github.com/gabrieldemarmiesse/torch-mojo-backend (single-process
multi-GPU DDP on the `comm` kernels).
