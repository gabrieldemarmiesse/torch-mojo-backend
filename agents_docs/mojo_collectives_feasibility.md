# Replacing NCCL/RCCL with Mojo collectives for GPT-2 DDP — feasibility study

Date 2026-09-08. Author: Fable 5.1 agent. Scope as briefed: nanoGPT-124M DDP on
(a) Kyutai 8×H100 SXM nodes (NVSwitch, IB 400 Gb/s, driver 570 / CUDA 12.8) and
(b) CINES Adastra 4×MI300A nodes (ROCm 6.4.3, RCCL 2.22.3, Slingshot/cxi), one
process per GPU under torchrun. Bar: NCCL/RCCL-class time on exactly this
traffic. Everything under `/home/gabriel/ddp_work/mojo_collectives/` (`proto/`
sources and binaries, `logs/` raw job output). Repo checkout untouched.

Software read: NCCL master 2.31.2 (`/home/gabriel/projects/nccl`, cited as
`nccl:path:line`), RCCL **2.30.7** develop (`/home/gabriel/projects/rocm-system/projects/rccl`,
cited `rccl:path:line`; the monorepo has no 2.22.3 tag — deltas to Adastra's
2.22.3 taken from `rccl:CHANGELOG.md`), modular checkout f807c5a (Jul 2026,
`modular:path:line`), installed MAX 26.5.0 / Mojo 1.0.0 (`.venv`, precompiled
`comm.mojoc` etc.). Stock reference torch 2.11.0+cu128 with NCCL 2.28.9
(`~/ddp_work/torch_cu128`); the repo's own PG uses nvidia-nccl-cu12 2.31.2.

## 1. Verdict (engineering estimate)

Feasible, and the intra-node half is already demonstrated: a 120-line
reduce-scatter + all-gather kernel written from scratch in Mojo, run as **8
separate processes** with buffers exchanged through `cuIpcGetMemHandle` /
`cuIpcOpenMemHandle` called from Mojo, allreduces the 27 MiB GPT-2 gradient
bucket in **163–165 µs (300–304 GB/s busbw)** against **175 µs (283 GB/s)** for
NCCL 2.28.9's NVLS/SIMPLE path on the same node class — i.e. at the size that
matters it is 6% faster than NCCL, and it reaches 75% of NCCL-NVLS (92% of
NCCL-ring) at 512 MiB, where NCCL's SHARP multicast wins. The same source
cross-compiles unchanged for gfx942 except a comptime gate around the NVLS
(multimem) variant, and the AMDGCN it emits (`sc0 sc1` system-scope loads/stores,
`buffer_wbl2 sc0 sc1` / `buffer_inv sc0 sc1`) is exactly what RCCL relies on.
Cost: **M1 intra-node NVIDIA in the ProcessGroup ≈ 3–4 weeks; M2 MI300A
validation ≈ 2–3 weeks (needs Adastra time; kernel untested there); M3
multi-node ≈ 2 weeks if the inter-node hop stays on NCCL/RCCL (hierarchical
RS → inter-node allreduce of the 1/8 shard → AG), or 8–16 weeks for a pure-Mojo
IB-verbs + libfabric/cxi transport, which is a host-proxy/RDMA project, not a
kernel one.** Two decisions move the estimate: (1) how user tensors reach peer
processes — MAX's allocator memory is **not** exportable with legacy IPC
(measured: `cuIpcGetMemHandle` → `CUDA_ERROR_INVALID_VALUE`), so either the
library owns `cuMemAlloc`/`hipMalloc`-ed staging buffers and copies in/out
(measured +27% at 27 MiB, +27% at 168 MiB, all hidden except the tail bucket) or
MAX's VMM arena becomes exportable with `cuMemExportToShareableHandle` — today
it is not (measured: the 256 MiB `cuMemCreate` chunks carry
`requestedHandleTypes = 0`, §5.6), so this needs a one-line change in MAX; (2) whether multi-node is in scope for the experiment at all.
GPT-2 DDP is not communication-bound on either topology (~2.8 ms of NVLink time
per ~45 ms step, 12 of 13 buckets overlapped), so any of the measured variants
keeps end-to-end throughput within ~1% of NCCL; the bar is met intra-node.

## 2. GPT-2 traffic profile (derived from source)

Model/config: `demo_scripts/nanogpt_ddp.py` — GPT-2 124M, `vocab 50304`,
`bias=False`, fp32 parameters, bf16 autocast, batch 12×1024 per rank, fused
AdamW, `DDP(model, broadcast_buffers=False)`, defaults otherwise
(`bucket_cap_mb=25`, `gradient_as_bucket_view=False`, no `find_unused_parameters`,
no `static_graph`). 75 parameter tensors (wte/lm_head tied), 124,373,760 elements
= **497,495,040 B (474.45 MiB) fp32 gradients per step**.

Reducer bucketing (torch `reducer.cpp:2289-2412`: push first, close when
`size >= limit`, limits `{1 MiB, 25 MiB}`, order = gradient-ready order, rebuilt
once at step 2): **13 allreduces per step**

| bucket | contents | bytes | MiB |
|---|---|---|---|
| 0 | ln_f, h.11.mlp.c_proj | 9,440,256 | 9.0 |
| 1–11 | 6 tensors each | 28,317,696 | 27.0 each |
| 12 | h.0.mlp.c_fc … wpe, **wte** (tied embedding, last to be ready) | 176,560,128 | **168.4, tail-exposed** |

Op: `ncclSum` on values pre-divided by world size on the host
(`reducer.cpp:384-392` `mul_out(bucket_view, grad, 1/div_factor)`; the default
hook has no division, `default_comm_hooks.cpp:44-58`). One contiguous fp32
tensor per bucket, in place. Step 1 is one 474 MiB allreduce (initial bucketing
uses `sys.maxsize`). Per run: 1 allgather (8 B), broadcasts of 2,000 B,
270,950,400 B, 226,544,640 B (parameters, `broadcast_bucket_size=250 MiB`),
304 B, 52 B (bucket indices); an `AVG` allreduce of 4 B every 10 steps;
`barrier()` once (routed to gloo/CPU). DDP itself only needs
`allreduce`, `broadcast`, `allgather` (+`barrier`) from the PG
(`torch_mojo_backend/distributed/process_group.py:393,456,522,937`).

Execution today: each bucket goes to `ncclAllReduce` on a per-device side
stream that first waits on the default stream (`process_group.py:303-335`,
`device_streams.py:87-91`); buckets serialize on that stream and overlap
backward; completion is the lazy fence in `mojo_device/comm_fence.py` (no
watcher thread). Budget: 8×H100 DDP step ≈ 45 ms at B=12 (memory notes); the 12
overlappable buckets need ≥ 17 GB/s busbw to hide — NVLink gives 300+.

## 3. What NCCL / RCCL do for this traffic

### 3.1 NCCL on 8×H100 NVSwitch (measured + source)

Measured choices (`NCCL_DEBUG_SUBSYS=TUNING`, stock 2.28.9, `logs/proto_233567.out`):
1 MiB → `RING/LL` 24 channels; 9–512 MiB → **`NVLS/SIMPLE`, 16 channels
(`NVLS_NCHANNELS_SM90`, `nccl:src/transport/nvls.cc:155`), 640 threads,
CGA cluster 4**. With `NCCL_NVLS_ENABLE=0`: 9–27 MiB → `RING/LL128` 24 ch,
≥128 MiB → `RING/SIMPLE` 24 ch. Cost model: `nccl:src/tuning/tuning_general.cc:64`
(`lat*latCount + nBytes/(1000*bw)`), tables `nccl:src/tuning/cost_model.cc:141-220`
(Hopper `llMaxBw` 141 GB/s single node, LL128 36.7 GB/s/channel, NVLS
efficiency 0.85 `nccl:src/tuning/nvls.cc:15`).

What NVLS needs: `cuMulticastCreate/AddDevice/BindMem`
(`nccl:src/transport/nvls.cc:60,344,372`), the shareable handle broadcast over the
bootstrap socket and **the POSIX fd passed with `SCM_RIGHTS` over an AF_UNIX
socket** (`nccl:src/proxy.cc:1583`, `nccl:src/os/linux_ipcsocket.cc:212`), device
`multimem.ld_reduce.relaxed.sys.global.add[.v4]` / `multimem.st.global`
(`nccl:src/device/reduce_kernel.h:1131-1198`, `nccl:src/device/op128.h:429-437`),
fp32 sum supported (`nccl:src/include/device.h:608-623`). Multi-node NVLS
proper is off (needs `collnetEnable`, `nccl:src/tuning/nvls.cc:38-41`); 2 nodes
pick `NVLS_TREE/SIMPLE` (measured, §5.5).

Intra-node P2P transport (what a ring/RS-AG port has to match): with driver ≥
12000 NCCL uses the **cuMem VMM path, not `cudaIpcGetMemHandle`**
(`nccl:src/misc/cudawrap.cc:16-51`; alloc `nccl:src/include/alloc.h:312-346`
`cuMemCreate(POSIX_FD)`; import `nccl:src/transport/p2p.cc:267-327` via
`cuMemImportFromShareableHandle` after the fd round-trip). H100 pairs run in
WRITE mode (READ only for compcap 80, `nccl:src/graph/paths.cc:441-445`);
per (peer, channel) 2 MiB send + 10 MiB recv FIFO (`nccl:src/transport/p2p.cc:410-413,489-494`),
8 steps of 512 KiB (`NCCL_STEPS`, `nccl:src/include/device.h:26`); head/tail
counters in **device** memory written remotely over NVLink
(`nccl:src/transport/p2p.cc:570-605`). Plain eager `ncclAllReduce` on
unregistered buffers never takes the direct (FIFO-bypass) path
(`nccl:src/register/coll_reg.cc:212-214,339-343`) — NCCL copies every byte
through its own buffers, so "staging" (option b below) is NCCL's own default
design, not a handicap.

Protocol primitives (minimum for SIMPLE + LL, `nccl:src/device/op128.h`):
`ld.volatile.global` (:342-345), `st.relaxed.sys.global.u64` (:380),
`fence.acq_rel.sys` (:395), 16 B `ld/st.global.v2.b64`, named
`barrier.sync`, `__syncwarp`; LL relies on 8-byte store atomicity of the fabric
(`nccl:src/include/device.h:75-88`), LL128 on 128-byte NVLink store atomicity
plus warp lockstep (`nccl:src/device/prims_ll128.h:194-207`). No `__nanosleep`,
no TMA/cluster sync in the general kernels; dynamic shmem ≈ 80 KiB
(`nccl:src/include/device.h:573-591`).

### 3.2 RCCL on 4×MI300A (source; not measurable from here)

- APU detection = `hipDeviceAttributeDirectManagedMemAccessFromHost` on device 0
  (`rccl:src/init.cc:1905-1920`); single node 4 ranks → channel multiplier
  `nc=4` (MI300X branch), multi-node APU → `nc=6`. Tuning model 5 for gfx942
  (`rccl:src/graph/tuning.cc:1637-1647`). Tree+Simple is removed on single-node
  gfx942 (`rccl:src/graph/tuning.cc:1232-1236`), so intra-node AllReduce is
  Ring/{LL,LL128,Simple}; LL128 is **off by default** on a generic 4-GPU node
  (`ll128Enabled` only via Rome-model options or `RCCL_LL128_FORCE_ENABLE=1`,
  `rccl:src/init.cc:1661,1946-1947`, gate `rccl:src/graph/tuning.cc:1469-1485`).
  Multi-node AllReduce protocol override: LL ≤ 512 KiB, else SIMPLE at 25 MiB
  (`rccl:src/rccl_wrap.cc:148-178`, ranges `rccl:src/graph/tuning.cc:556-573`).
- Memory: every cross-agent buffer is **uncached** `hipExtMallocWithFlags(hipDeviceMallocUncached)`
  (`rccl:src/transport/p2p.cc:282-287`), exported with legacy
  `hipIpcGetMemHandle` / `hipIpcOpenMemHandle(LazyEnablePeerAccess)`
  (`rccl:src/transport/p2p.cc:281-295,382-384`); fine-grain support is a hard
  P2P precondition (`rccl:src/transport/p2p.cc:138-146`). The cuMem/VMM path is
  auto-enabled only on gfx1250 and needs kernel ≥ 6.8 / HIP ≥ 7.2
  (`rccl:src/misc/rocmwrap.cc:121-150`) → **on Adastra RCCL itself uses
  hipMalloc-class staging buffers + legacy IPC**.
- Device ordering on gfx942: 16-byte `global_load/store_dwordx4` with system
  syncscope (`sc0 sc1`) (`rccl:src/device/op128.h:366-388`), `__hip_atomic_*`
  RELAXED/RELEASE at `__HIP_MEMORY_SCOPE_SYSTEM` for flags
  (`rccl:src/device/op128.h:428-468`), `fence_acq_rel_sys()` is a **no-op**
  (`rccl:src/device/op128.h:470-475`), producer ordering =
  `s_waitcnt lgkmcnt(0) vmcnt(0)` in the block barrier then a system-scope
  relaxed tail store (`rccl:src/device/prims_simple.h:193-210`, "cheap fence"
  on gfx942 `rccl:src/include/rccl_common.h:251-273`); no HDP flush on
  gfx90a/942/950 (`rccl:src/transport/p2p.cc:468-482`); wave 64, 256 threads
  max, 4 barrier groups (`rccl:src/include/device.h:146-157`); spin loops use
  `s_sleep(1)`/`s_wakeup` (`rccl:src/device/prims_simple.h:127-140`). LL line
  = two `u64` non-temporal loads / two plain `u64` stores
  (`rccl:src/device/prims_ll.h:170-190,276-296`); LL128 line is 64 B and its
  correctness is pinned on a single 128-bit vector store
  (`rccl:src/device/prims_ll128.h:44-60`).
- Slingshot: no libfabric/cxi code in RCCL; the cxi provider is an out-of-tree
  `ncclNet_v8+` plugin (`librccl-net.so`, `rccl:src/plugin/net.cc:259,102-187`;
  vtable `rccl:src/include/plugin/net/net_v8.h:27-80`), GPU registration via
  `hsa_amd_portable_export_dmabuf` + `regMrDmaBuf` (`rccl:src/include/rocmwrap.h:178-197`,
  gate `rccl:src/misc/rocmwrap.cc:292-334`), host proxy thread as in NCCL.

## 4. What Mojo/MAX provides and what is missing

| need | status (installed MAX 26.5 / Mojo 1.0) | evidence |
|---|---|---|
| kernel on GPU A reads/writes GPU B memory (same process) | yes; `enable_all_peer_access` (`modular:mojo/stdlib/std/gpu/host/device_context.mojo:7660`), raw peer pointers as kernel args | `comm/allreduce.mojo` does it; measured §5.2 |
| cross-process buffer sharing | **absent** from MAX (`cuMemImportFromShareableHandle`, `cuIpc*`, `hipIpc*` not in `libAsyncRTMojoBindings.so`); done here via `OwnedDLHandle("libcuda.so.1")` → `cuIpcGetMemHandle/cuIpcOpenMemHandle` (64-byte struct by value passed with a 4-dummy-register stack shim; `std.ffi` has no C-struct ABI yet, MOCO-3692) | `proto/ipc_probe.mojo`, `proto/ipc_allreduce.mojo`; §5.4 |
| exporting MAX-allocated buffers | **legacy IPC refused** (`cuIpcGetMemHandle` rc=1 on `enqueue_create_buffer` memory); VMM export: §5.6 | `logs/ipc_ar_233615.out`, `logs/vmm_233620.out` |
| system-scope release/acquire, volatile loads, fences | yes: `Atomic[..].store[ordering=RELEASE]` / `load[ACQUIRE]` with default (system) syncscope, `UnsafePointer.store[volatile=True]`, `std.atomic.fence`; lowers to `st.release.sys.global` / `ld.acquire.sys.global` / `st|ld.volatile.global` on sm_90a and `global_store/load … sc0 sc1`, `buffer_wbl2 sc0 sc1`, `buffer_inv sc0 sc1`, `flat_* sc0 sc1` on gfx942 | `proto/asm/rsag_sm_90a.asm`, `proto/asm/rsag_gfx942.asm` (dumped with `max.gpu.host.compile._compile_code`) |
| working in-kernel cross-GPU barrier | yes, `comm.sync._multi_gpu_barrier` + `Signal` (vLLM-style, portable NVIDIA/AMD) | `modular:max/kernels/src/comm/sync.mojo:330-479` |
| tuned intra-node allreduce | yes, `comm.allreduce` 1-stage/2-stage/Lamport/multimem, arch table sm_90a/sm_100a/CDNA3/CDNA4 (`modular:max/kernels/src/comm/allreduce.mojo:257-469`); single-process, rank = `ctx.id()`; no MI300A entry (falls to CDNA3: 32 blocks, 1-stage) | measured §5.2 |
| NVLS / multicast | `DeviceMulticastBuffer(List[DeviceContext])` + `multimem_ld_reduce/st` — **single process only**, no handle export | `device_context.mojo:7896`, `memory.mojo:3118` |
| side stream + priority, events, cross-device stream wait | Mojo `create_stream(priority=)`, `DeviceStream.enqueue_function(DeviceFunction)`; Python `DeviceStream` (no priority), `DeviceEvent.is_ready()`; closure-capturing kernels cannot be launched on a `DeviceStream` from a shared-lib build (m0 issue modular-1) — the prototype kernel is capture-free | repo `mojo_device/device_streams.py` |
| host-mapped flags / host callbacks (proxy designs) | `CompletionFlag`, `DeviceStream.wait_for_host_value`, `enqueue_host_func` (CUDA only), pinned `enqueue_create_host_buffer` | `device_context.mojo:2356,2389,2529,4300` |
| calling libcuda/libamdhip64 | `OwnedDLHandle` + `get_function[Ret](name)`; MAX already dlopened them | `modular:mojo/stdlib/std/ffi/__init__.mojo:316,380` |
| passing a `DeviceContext` into an extension | `_ctx_ptr` → `DeviceContext(OpaquePointer(unsafe_from_address=…))` | `eager_kernels/__init__.py:904-909`, `op_utils/__init__.mojo:865-869` |
| rendezvous | none in Mojo; the PG already has the c10d store (`process_group.py:352-365`) | file exchange stands in for it in the probes |
| pinned SM clock via SLURM comment | did not take (SM 1980 MHz under load in every job) | `logs/proto_233567.out` |

Inline `Signal` cost: `size_of[Signal]()` ≈ 24.75 MiB per rank (24 MiB Lamport
region unused by RS/AG); a slim counter-only struct would be 768 KiB.

## 5. Measurements

All on kyutai_debug 8×H100 80GB HBM3 nodes, SM clock 1980 MHz (unpinned), 5 reps
× 20 back-to-back calls per size, median; device time = CUDA events (stock
torch) / wall over 20 launches with host enqueue ≤ 16 µs per launch (Mojo;
`RESULT_DEV`/`kernel span` = in-kernel `globaltimer` per-rank span, within 3% of
wall everywhere). busbw = algbw × 2(N−1)/N. Raw logs in `logs/`.

### 5.1 NCCL reference, 8 ranks, 1 node

Job 233527 node `cl02s04dgx10`, job 233567 node `cl02s01dgx27`:
`torchrun --standalone --nproc-per-node=8 nccl_ref_stock.py|nccl_ref_stock2.py`
(cu128 venv, `NCCL_DEBUG=INFO NCCL_DEBUG_SUBSYS=TUNING`). fp32 (bf16 within 2%):

| MiB | NCCL default (NVLS on) | algo | NCCL `NCCL_NVLS_ENABLE=0` | algo |
|---|---|---|---|---|
| 1 | 0.026 ms · 71 GB/s | RING LL 24ch | 0.033 · 56 | RING LL |
| 9 | 0.087 · 190 | NVLS SIMPLE 16ch | 0.077 · 214 | RING LL128 24ch |
| 16 | 0.121 · 244 | NVLS | 0.116 · 253 | RING LL128 |
| 25 | 0.169 · 272 | NVLS | 0.174 · 263 | RING LL128 |
| 27 | **0.175 · 283** | NVLS | 0.182 (bf16; fp32 median 0.242 noisy, min 0.182) · 272 | RING LL128 |
| 128 | 0.585 · 401 | NVLS | 0.698 · 337 | RING SIMPLE |
| 168 | **0.751 · 410** | NVLS | 0.896 · 344 | RING SIMPLE |
| 512 | 2.146 · 438 | NVLS | 2.643 · 356 | RING SIMPLE |

Repo PG (`nccl_ref_mojo.py`, mojo backend over nvidia-nccl-cu12 2.31.2, same
algo choices): 25 MiB 0.177 ms, 512 MiB 2.156 ms; 1 MiB 0.107 ms wall — the
Python PG is host-bound below ~9 MiB (NCCL device time 0.026 ms).

### 5.2 Mojo intra-node prototypes, 1 process, 8 `DeviceContext`s (job 233567, node `cl02s01dgx27`)

`proto/ar_proto.mojo` (`ar_proto_<dt> <bytes> 20 5 <variant> [blocks]`, built
`uv run --no-sync mojo build --target-accelerator sm_90a -D dtype=… -D multimem=1`).
`own` = RS+AG written here (216×256 threads, 16 B vector peer loads, MAX's
`Signal`/`_multi_gpu_barrier` reused, output buffers double as the AG source,
no temp payload); `max` = `comm.allreduce` (2-stage on sm_90a for these sizes,
216 blocks, 8×msg payload); `maxmm` = `comm.allreduce[use_multimem=True]`
via `DeviceMulticastBuffer`. fp32 µs per allreduce (bf16 within 1%):

| MiB | own | MAX 2-stage | MAX multimem | NCCL NVLS | NCCL ring | own/NCCL-NVLS |
|---|---|---|---|---|---|---|
| 1 | 19.6 (94 GB/s) | 20.3 | 20.3 | 26 | 33 | 0.75 |
| 9 | 66 (249) | 74 | 72 | 87 | 77 | 0.76 |
| 16 | 107 (275) | 115 | 107 | 121 | 116 | 0.88 |
| 25 | 153 (299) | 172 | 166 | 169 | 174 | 0.91 |
| **27** | **164 (302)** | 174 (285) | 170 (291) | **175 (283)** | 182 | **0.94** |
| 128 | 723 (325) | 751 | 769 | 585 | 698 | 1.24 |
| **168** | **940 (328)** | 979 (315) | 1002 | **751 (410)** | 896 | **1.25** |
| 512 | 2857 (329) | 2971 (316) | 3018 | 2146 (438) | 2643 | 1.33 |

Grid sweep, own fp32 (µs): 27 MiB 64→187, 128→166, 216→164, 432→168, 864→180;
512 MiB 64→3356, 128→2926, 216→2857, 432→2845, 864→2848. The own kernel
saturates at ~330 GB/s busbw = 190 GB/s of NVLink reads per GPU; NCCL ring
reaches 356, NVLS 438. MAX's multimem path is not faster than its 2-stage path
here (unicast-bound copy phases).

### 5.3 Correctness/ABI probe, 2 processes, `cuIpc*` from Mojo (job 233615, node `cl02s01dgx15`; earlier 233572)

`proto/ipc_probe.mojo export 0 …` / `import 1 …`, 128 MiB uint32:
`cuIpcGetMemHandle` on MAX `enqueue_create_buffer` memory → **rc 1
(CUDA_ERROR_INVALID_VALUE)**; on `cuMemAlloc_v2` memory → rc 0;
`cuIpcOpenMemHandle` (struct-by-value via stack shim) → ok; importer read
pattern (0 wrong of 33,554,432), **peer read 369 GB/s**, wrote pattern+1, released
a flag with `Atomic.store[RELEASE]` (system scope); exporter observed the flag
and verified all 33,554,432 elements: **PASS** (also GPU 3→6). Lesson recorded
in the source: one legacy handle per allocation; opening the same allocation
twice in a process fails with CUDA_ERROR_ALREADY_MAPPED (208).

### 5.4 Mojo allreduce, 8 processes over IPC (job 233615, node `cl02s01dgx15`)

`proto/ipc_allreduce.mojo`: each rank `cuMemAlloc`s one region
[Signal | input | output], exports one handle to a file (stand-in for the
c10d store), imports the 7 peers' regions, runs the §5.2 `own` kernel. `direct`
= the collective operates on those library-owned buffers; `staging` = the user
tensor lives in a MAX buffer and every call copies it in and the result back
(option b), copies included in the time. µs per allreduce, max over ranks:

| MiB | direct fp32 | direct bf16 | staging fp32 | staging bf16 | NCCL NVLS |
|---|---|---|---|---|---|
| 9 | 66 (250 GB/s) | 68 | 82 | 85 | 87 |
| **27** | **165 (300)** | 163 | **210 (236)** | 210 | **175** |
| 168 | 946 (326) | 947 | 1200 | 1198 | 751 |
| 512 | 2861 (328) | 2858 | 3624 | 3622 | 2146 |

Cross-process costs nothing versus single-process (§5.2). `cuIpcOpenMemHandle`
took 10–17 ms per peer (70–118 ms per rank for 7 peers, one-off). Staging adds
one D2D round trip per direction (≈2.5 TB/s HBM): +45 µs at 27 MiB. Writing the
AG result straight into the user output and staging only the input would halve
that (not measured).

### 5.5 NCCL reference, 16 ranks, 2 nodes (job 233608, nodes `cl02s01dgx15`+`cl02s03dgx28`)

Stock cu128, IB with GPUDirect RDMA (12 HCAs seen, 10×400 Gb/s IB + 2×100 Gb/s
RoCE unmerged), algo `NVLS_TREE/SIMPLE` 16 ch for ≥9 MiB, `TREE/LL128` at 1 MiB:
fp32 1 MiB 0.082 ms (24 GB/s), 9 MiB 0.221 (80; min 0.165), 16 MiB 0.200 (158),
**27 MiB 0.259 (205)**, 128 MiB 0.764 (329), **168 MiB 0.946 (349)**,
512 MiB 2.397 (420). Reference busbw at large sizes agrees with the earlier
~392 GB/s measurement.

### 5.6 VMM export of MAX-owned memory (job 233620, `proto/vmm_probe.mojo`)

`cuPointerGetAttribute` dump (MEMPOOL_HANDLE, RANGE/MAPPING size,
ALLOWED_HANDLE_TYPES, IS_LEGACY_CUDA_IPC_CAPABLE) + `cuMemRetainAllocationHandle`
→ `cuMemGetAllocationPropertiesFromHandle` → `cuMemExportToShareableHandle`
(POSIX fd) → fd passed with `pidfd_getfd` → `cuMemImportFromShareableHandle`
+ `cuMemAddressReserve/Map/SetAccess` in the importer, with and without
`MODULAR_DEVICE_CONTEXT_MEMORY_MANAGER_VMM=1`.

Result (node `cl02s02dgx26`, kernel 5.15.0-1108, driver 570.211.01, identical
with and without the knob): MAX's `enqueue_create_buffer` memory is **VMM**, not
a mempool — `MEMPOOL_HANDLE = 0`, `IS_LEGACY_CUDA_IPC_CAPABLE = 0`,
`RANGE_START 0x420000000`, `RANGE_SIZE 307,090,161,664 B (286 GiB)` VA arena,
`MAPPING_SIZE 268,435,456 B` (256 MiB physical chunks), `ALLOWED_HANDLE_TYPES = 0`.
`cuMemRetainAllocationHandle` → rc 0 (a real `cuMemCreate` handle), but its
`CUmemAllocationProp.requestedHandleTypes = 0`, so
**`cuMemExportToShareableHandle` → rc 1 (CUDA_ERROR_INVALID_VALUE)**. Option (a)
is therefore not available on MAX 26.5 as shipped: the chunks were created
without `CU_MEM_HANDLE_TYPE_POSIX_FILE_DESCRIPTOR`. It is a one-line change on
Modular's side (`requestedHandleTypes` at `cuMemCreate`; NCCL sets it,
`nccl:src/include/alloc.h:312-346`) — the natural upstream ask — after which the
importer half of this probe (fd via `pidfd_getfd`/`SCM_RIGHTS`, import, map one
256 MiB chunk per peer, offset arithmetic) applies unchanged, and the mapping
granularity (256 MiB chunks, handle cached per chunk) makes the per-tensor
exchange cheap. Until then the library must own the buffers (option b, §5.4);
option (c) does not exist (no IPC/export symbol in MAX's Mojo or Python API,
§4). AMD side from source: RCCL on ROCm 6.4.3 uses legacy
`hipIpcGetMemHandle` on `hipExtMallocWithFlags(hipDeviceMallocUncached)` buffers
(§3.2) — i.e. also library-owned staging; `hipMemExportToShareableHandle` exists
but RCCL only enables the VMM path on gfx1250 with HIP ≥ 7.2
(`rccl:src/misc/rocmwrap.cc:121-150`), so option (b) is the expected path there
too, and whether MAX's HIP allocator memory accepts `hipIpcGetMemHandle` is
untested (it is `hipMalloc`-class by default per `agents_docs/distributed.md`, VMM with
`MODULAR_DEVICE_CONTEXT_MEMORY_MANAGER_VMM=1`).

## 6. One codebase for NVIDIA and AMD

Demonstrated by cross-compilation (`--target-accelerator gfx942`, no device):
`ar_proto.mojo` and `ipc_allreduce.mojo` build for gfx942 with **one** gate —
`comptime MM = get_defined_bool["multimem"]() and has_nvidia_gpu_accelerator()`
around the `DeviceMulticastBuffer`/`use_multimem` path (multimem is sm_90+ only,
`modular:mojo/stdlib/std/gpu/memory/memory.mojo:3085`); `ipc_probe.mojo` builds
with a comptime name switch (`hipIpcGetMemHandle`/`hipIpcOpenMemHandle`/`hipMalloc`/
`hipSetDevice` vs `cuIpc*`/`cuMemAlloc_v2`/`cuDevicePrimaryCtxRetain+cuCtxSetCurrent`;
identical signatures, same 64-byte handle struct). The kernel body is vendor-free:
`global_perf_counter_ns` → `s_memrealtime`, `Atomic` → `sc0 sc1` + `buffer_wbl2`/
`buffer_inv` (matches RCCL's `__hip_atomic_* SYSTEM` + `dwordx4 sc0 sc1`, §3.2),
16 B vectors → `global_load/store_dwordx4`, BLOCK=256 ≤ RCCL's gfx942 max.

Unavoidable per-vendor shims (all host-side, ~100 lines total): driver library
and function names; making the device current; **staging allocation flags** —
on MI300 the FIFO/flag buffers must be `hipExtMallocWithFlags(hipDeviceMallocUncached)`
(RCCL's precondition, §3.2), `hipMalloc` is not enough for polled flags; on
NVIDIA plain `cuMemAlloc`. Tuning: block count (216 on H100; MAX uses 32–64 on
CDNA; RCCL `nc=4` channels for 4 ranks), spin-loop `s_sleep`, 4 APUs on xGMI
(48 GB/s per link, 2 links per pair per `rome_model_80`, `rccl:src/graph/rome_models.cc:2214-2287`)
vs 8 on NVSwitch — expect ~100–150 GB/s busbw, measured only on Adastra.
Not verified anywhere here: MI300A runtime behaviour of the barrier, hipIpc of
MAX-allocated memory on ROCm (default MAX allocator differs from CUDA's; the
`…MEMORY_MANAGER_VMM=1` knob the repo documents for Adastra changes it again).

## 7. Multi-node

What the inter-node hop takes, from source: NCCL `net_ib` is ~13.5k lines that
are **not** a plugin (compiled into libnccl, `nccl:src/transport/CMakeLists.txt:15`);
a minimal re-implementation needs the verbs set (`ibv_open_device/alloc_pd/
create_cq/create_qp`, three `ibv_modify_qp` state masks `nccl:src/transport/net_ib/connect.cc:370-502`,
`ibv_reg_mr`/`ibv_reg_dmabuf_mr` with fd from `cuMemGetHandleForAddressRange`
`nccl:src/transport/net.cc:1014-1016`, `ibv_post_send` WRITE/WRITE_WITH_IMM,
`ibv_poll_cq`), a TCP/store exchange of QPN/LID/GID + CTS fifo address/rkeys,
the 8-slot credit ring with host-pinned GPU-mapped head/tail
(`nccl:src/transport/net.cc:958-976`), the CTS record protocol verbatim
(`nccl:src/transport/net_ib/common.h:268,540-546`; `p2p.cc:275-411`), a
non-blocking progress thread owning all verbs calls, and an MR cache
(`nccl:src/transport/net_ib/reg.cc:10-64`). Estimate 2–3k lines Mojo + a
host thread; 6–10 weeks to NCCL-class bandwidth on this fabric. Slingshot:
another libfabric/cxi backend (RCCL's is an external plugin with no in-tree
code; `fi_*` calls + `hsa_amd_portable_export_dmabuf`), 4–8 weeks, only testable
on Adastra. No GPU-side new features are needed for either (host proxy +
volatile flags), so this is not a language question; it is a transport port.

Honest first scope: **Mojo intra-node, NCCL/RCCL inter-node**. Hierarchical
allreduce per bucket: Mojo RS (my 1/8 shard reduced within the node) → one
`ncclAllReduce` per local rank over a communicator of size nNodes (3.4 MiB at
27 MiB; the PG already knows `ncclCommInitRank` with a store key) → Mojo AG.
Expected ≈ 165 µs + IB allreduce of 3.4 MiB (~60–90 µs at 400 Gb/s per HCA) ≈
230–260 µs vs NCCL's 259 µs NVLS_TREE at 2 nodes (§5.5) — parity within the
noise, and the dependency on NCCL drops to `ncclAllReduce` on a small
communicator. Alternative for a fully vendor-free run: stage the shard through
host memory and gloo (already instantiated in the PG) — ~3.4 MiB × 2 over PCIe
+ TCP, ~1–3 ms per bucket, hidden behind backward for buckets 0–11 but a
~5–10 ms exposed tail; acceptable for a demo, not for the bar.

## 8. Minimal plan

M1 — intra-node NVIDIA in the PG (3–4 weeks). Extension `eager_kernels/collectives_ops/`
(receives `_ctx_ptr` + raw pointers like every eager kernel); per-rank
library-owned region [Signal | in-stage | out-stage] sized to the largest
bucket (grow-on-demand needs an all-rank sync, m0 issue modular-4);
`cuIpcGetMemHandle` → 64 bytes through the c10d store key `mojo-ipc-<rank>-<gen>`
→ `cuIpcOpenMemHandle` at first collective; RS+AG kernel launched capture-free
on the existing comm `DeviceStream`; `allreduce` (+`broadcast`, `allgather` via
the same peer-load kernel) for same-node tensors, NCCL untouched for the rest;
completion via the existing `comm_fence`. Acceptance: 13-bucket step parity
with NCCL losses, 27 MiB ≤ 210 µs staged / ≤ 175 µs if §5.6 allows direct
export. Decision gate: ask Modular for `CU_MEM_HANDLE_TYPE_POSIX_FILE_DESCRIPTOR` on
the arena chunks (§5.6) — direct export removes the staging copies and the
size cap; the importer code is already written in `proto/vmm_probe.mojo`.

M2 — MI300A (2–3 weeks incl. cluster access). Swap names, uncached staging
allocation, `hipSetDevice`; tune blocks (32/64/128) and add `s_sleep` backoff;
validate barrier + IPC on 4 APUs; measure vs RCCL (`NCCL_DEBUG=INFO` for its
protocol) at 27/168 MiB.

M3 — multi-node (2 weeks): hierarchical RS → NCCL/RCCL shard allreduce → AG,
on both clusters; keep `TORCH_MOJO_BACKEND_COMM_STREAM`-style kill switch to
fall back to plain NCCL.

Optional M4 — NVLS in Mojo (2–3 weeks, NVIDIA only): `cuMulticastCreate`
+ `cuMemExportToShareableHandle` fd passing (needs an AF_UNIX `SCM_RIGHTS`
helper, ~100 lines) + MAX's existing `multimem_ld_reduce` kernel path; only
worth it for the 168 MiB tail bucket (+0.2 ms/step today).

## 9. Open risks

- Legacy IPC does not accept MAX's allocations (measured); the VMM export path
  depends on how MAX created its arena (§5.6). If neither works, staging is
  permanent: +45 µs per 27 MiB bucket, hidden, and the region size caps the
  largest single collective (step-1's 474 MiB allreduce → chunk it).
- Cross-process spin barriers have no abort: a dead rank hangs peers in-kernel
  (NCCL has abort flags, `nccl:src/device/primitives.h:154-164`). Needs a
  host watchdog + a poisoned generation counter.
- `Signal` lifetime contract: peers read my region after my kernel retires
  (m0 issue modular-4); the next start barrier orders reuse, so buffers must
  never be freed/resized without an all-rank host sync.
- MAX frees are stream-ordered on the owning stream only (memory note): every
  user tensor a side-stream collective touches must stay `record_use`-fenced,
  as the PG does today.
- `globaltimer`/`s_memrealtime` are per-GPU; never subtract stamps across GPUs.
- RCCL/MI300A numbers are extrapolated from source (2.30.7 read, 2.22.3 deployed);
  fine-grained/uncached memory semantics for polled flags on the APU are the
  first thing to test there.
- Mojo `std.ffi` has no C struct-by-value ABI; the 64-byte handle shim is
  x86-64 SysV specific (fine for both clusters, not portable to aarch64 hosts).
- ptxas 12.8 / driver 570 gate as for every kernel here; SLURM clock pin via
  `--comment` did not apply — all numbers at boost clock, NVLink-bound so low
  sensitivity.
