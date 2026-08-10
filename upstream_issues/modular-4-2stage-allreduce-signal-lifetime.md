# [kernels][comm] 2-stage allreduce instances exit without an end barrier — the Signal-payload lifetime contract is undocumented

## Summary

`_allreduce_2stage_kernel` (`comm/allreduce.mojo:619+`) synchronizes at the
stage boundaries (barriers around the reduce-scatter → all-gather handoff,
lines ~694/~741) but has **no end barrier**: a device's instance can retire
while peer instances are still reading that device's Signal payload (the
staging bytes after the `Signal` header) through P2P during their all-gather
stage. (The 1-stage path raises a related question for *input* buffers: the
docstring already warns that outputs must not alias inputs because peers
keep reading inputs, but whether instance completion implies peers are done
reading that device's input is similarly undocumented.)

That is a perfectly good design for throughput — but the resulting lifetime
contract for the caller is nowhere documented, and it is easy to get wrong:

- Completion of device `i`'s instance (e.g. an event recorded behind it on
  device `i`'s stream) does **not** imply peers are done reading device
  `i`'s signal payload or input buffer.
- Therefore freeing or reusing a signal buffer (or an input buffer) after a
  per-device completion signal is a use-after-free through P2P; safety
  requires *all* participating devices' instances to have completed.
- Reuse for a *subsequent* collective on the same streams is safe (the next
  launch's start barrier orders it), which is presumably why the graph
  compiler never hits this — but eager callers managing their own signal
  buffers (as `max.nn.comm.allreduce.Signals` consumers or the
  `_distributed_ops` path do) need to know.

We found this while implementing signal-buffer pools with grow-on-demand for
eager-mode DDP: replacing a smaller signal buffer with a larger one is only
safe after host-synchronizing every participating device, not just the
buffer's owner.

## Ask

Either of:

1. Document the contract on `allreduce` (and `reducescatter`/`broadcast`,
   which share the payload-staging pattern): "signal buffers and input
   tensors may be freed/resized only after ALL participating devices'
   instances have completed; per-device completion is not sufficient"; or
2. add an optional `end_barrier: Bool = False` comptime parameter for
   callers that want per-device completion to be a global completion signal
   (costs one extra `_multi_gpu_barrier`).

Option 1 is zero-cost and probably enough.

## Environment

modular @ `12beae6f14` (nightly 26.5.0.dev2026061806), 8x H100.
Consumer: github.com/gabrieldemarmiesse/torch-mojo-backend.
