# Priority-stream allreduce probe

Empirical answer (2026-08-07, 2xH100, max nightly dev2026072306) to
"can the pure-Mojo allreduce run on a raised-priority stream today?":
**yes** — see docs/multi_gpu_next_steps.md §2 for the result and the
integration plan into comm_ops.mojo.

The dodge for upstream_issues/modular-1 (closure captures cannot cross
`compile_function` + `DeviceStream.enqueue_function` under
--emit shared-lib): `_sum_allreduce_wrapper` in priority_probe.mojo has
no capturing comptime parameters — it builds the store epilogue INSIDE
device code from its runtime `result` argument and calls the internal
comm kernels, so the split compile/launch path lowers fine and the
kernel can be enqueued on `ctx.create_stream(priority=...)`.

Build (the .so must NOT live in eager_kernels — keep the kernel cache
clean):

    uv run mojo build demo_scripts/priority_stream_probe/priority_probe.mojo \
        --emit shared-lib --target-accelerator sm_90a \
        -I torch_mojo_backend/eager_kernels \
        -I demo_scripts/priority_stream_probe \
        -o /tmp/priority_probe.so

Run on a 2-GPU node (adjust the .so path inside run_probe.py if moved):

    MODULAR_NVPTX_COMPILER_PATH=/usr/local/cuda/bin/ptxas \
        uv run demo_scripts/priority_stream_probe/run_probe.py

Measured: correct 1-stage and 2-stage SUM allreduce from priority=-5
streams; under a default-stream compute flood the priority=-5 collective
completes 34x sooner than priority=0 (0.33 ms vs 11.3 ms).
