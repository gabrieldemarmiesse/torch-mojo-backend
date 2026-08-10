# [Mojo][GPU] Closure captures fail to lower when a kernel is compiled with `compile_function` and launched via `DeviceStream.enqueue_function` in a `--emit shared-lib` build

## Summary

A GPU kernel whose comptime parameter is a capturing `@parameter` closure
compiles and launches fine through the fused
`ctx.enqueue_function[kernel](...)` path, but the split path — compile once
with `ctx.compile_function[kernel]()`, launch with
`DeviceStream.enqueue_function(f, ...)` — fails to build under
`mojo build --emit shared-lib` with:

```
error: failed to legalize operation 'pop.compiler.global_load' that was
explicitly marked illegal: ... name = "<enclosing fn>[...]..._context_var_0"
...
error: failed to produce an archive for the module: failed to lower module
to LLVM IR for archive compilation, run LowerToLLVMPipeline failed
```

i.e. the closure's runtime captures are materialized as a compiler global
(`_context_var_0`) that cannot be lowered in the shared-library archive path.

This matters because `DeviceStream` is the only way to target a non-default
stream (e.g. `ctx.create_stream(priority=...)`), and `DeviceStream` only has
the `DeviceFunction`-taking `enqueue_function` overload
(`mojo/stdlib/std/gpu/host/device_context.mojo:2331`) — there is no fused
comptime-kernel overload on streams. So any kernel that takes a capturing
closure as a comptime parameter (for example every `comm` collective, whose
`output_lambda: elementwise_epilogue_type` epilogue captures the output
`TileTensor`) cannot currently be launched on a non-default stream from a
Python-extension build.

## Reproduction

Mojo 1.0.0b3.dev2026061806 / MAX 26.5.0.dev2026061806, H100, CUDA driver 570
(with `MODULAR_NVPTX_COMPILER_PATH` pointing at a system ptxas).

```mojo
from std.gpu.host import DeviceContext, DeviceStream
from std.memory import UnsafePointer
from layout import Coord, TileTensor, row_major
from comm.allreduce import _allreduce_1stage_kernel, elementwise_epilogue_type
# (Any kernel with a capturing-closure comptime param reproduces; the comm
# kernels are just the motivating case.)

def launch_on_stream[dtype: DType, ngpus: Int](
    stream: DeviceStream, out_ptr: UnsafePointer[Scalar[dtype], MutAnyOrigin],
    ..., ctx: DeviceContext,
) raises:
    var out_tile = TileTensor(out_ptr, row_major(numel))

    @always_inline
    @parameter
    @__copy_capture(out_tile)
    def epilogue[_dtype: DType, _width: SIMDSize, *, _alignment: Int](
        coords: Coord, val: SIMD[_dtype, _width]
    ) -> None:
        out_tile.store[width=_width, alignment=_alignment](coords, val.cast[dtype]())

    comptime kernel = _allreduce_1stage_kernel[
        dtype, ngpus, ..., output_lambda=epilogue, ...
    ]
    var f = ctx.compile_function[kernel]()          # <-- split path
    stream.enqueue_function(f, ..., grid_dim=..., block_dim=...)
```

`mojo build that_file.mojo --emit shared-lib -o out.so` fails with the
`pop.compiler.global_load` legalization error above. Replacing the last two
lines with `ctx.enqueue_function[kernel](...)` (fused path, default stream)
builds and runs correctly — which is how `comm/allreduce.mojo` itself
launches (`_allreduce_p2p`, lines 1068–1105).

## Expected

Either the split compile/launch path carries closure captures the same way
the fused path does (captures as launch-time arguments), or `DeviceStream`
gains the fused comptime-kernel `enqueue_function[func]` overload that
`DeviceContext` has (`device_context.mojo:5921`).

## Why we need it

torch-mojo-backend implements NCCL-style comm/compute overlap for eager
multi-GPU training: gradient allreduce should run on a raised-priority side
stream (`ctx.stream_priority_range().greatest`) so it preempts fat compute
kernels, exactly like `max/kernels/src/shmem/shmem_context.mojo:560-606`
does. The SHMEM precedent works because its kernels are capture-free; the
`comm` collectives are not (the output epilogue captures the output tensor).
Our current workaround is a second owning `DeviceContext(device_id=i)` per
GPU whose default stream acts as the comm stream — functional, but that
stream cannot be given a priority (`AsyncRT_DeviceContext_create` takes only
`(api, id)`), and NCCL's overlap quality depends on priority scheduling.
