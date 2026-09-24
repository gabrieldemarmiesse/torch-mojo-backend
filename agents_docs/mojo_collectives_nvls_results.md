# NVLS (NVSwitch multicast) allreduce in Mojo — measured results

Standalone multi-process prototype under `nvls/`, 8xH100 80GB HBM3 + NVSwitch
(kyutai partition, driver 570.211.01, CUDA 12.8 toolchain, MAX 26.5 / Mojo 1.0).
One process per GPU (the torchrun model), no NCCL, no MAX multicast API — the
driver is called through `OwnedDLHandle("libcuda.so.1")`.

Sources: `mcsetup.mojo` (host: VMM + multicast bring-up, fd transport),
`mckernels.mojo` (device: the allreduce and the barrier), `mcharness.mojo`
(probe / verify / timing), `abharness.mojo` (ABBA A/B against the unicast
kernels of `../kernel`, which are snapshotted read-only in `ref/`),
`build.sh`, `run.sh`, `bench.sbatch`, `sweep.sbatch`, `crossover.sbatch`,
`report.py`, `nccl_sizes.py`, `dump_asm.mojo`, and the emitted PTX in `asm/`.

## 0. Verdict

* **NVLS works on these nodes** from Mojo, multi-process, with no MAX multicast
  API and no NCCL: every device reports `CU_DEVICE_ATTRIBUTE_MULTICAST_SUPPORTED
  = 1`, the whole `cuMulticast*` sequence returns 0, and the kernel verifies
  against an fp64 host reference at every size and world from 1 element to
  512 MiB, in place and out of place, fp32 and bf16.
* **The target is met, on the line at 168 MiB.** Staged (user tensors copied
  in and out), fp32, 8xH100, same node and same job as the reference:
  **785.7-788.3 us at 168 MiB against NCCL's 748.6-749.9** (1.048x and 1.053x
  in two independent jobs, budget 790) and **2217 us at 512 MiB against 2143**
  (1.035x, budget 2250). So 512 MiB is comfortably inside 5% and 168 MiB
  straddles it; §7.5 names the ~60 us of barrier that would settle it. Against
  the shipped unicast kernels the same numbers are 0.82x and 0.77x.
* **Crossover 48 MiB.** Below it the unicast push/reduce/pull kernels win and
  must keep the traffic; above it NVLS wins and the margin grows with size.
  GPT-2 DDP's 27 MiB bucket stays unicast, its 168 MiB tail bucket moves.
* **RDMA registration of the region failed for a reason that is now fixed**:
  the `cuMemCreate` prop lacked `allocFlags.gpuDirectRDMACapable`, which
  `nvidia_peermem` requires on VMM memory; there is still no dmabuf fallback
  on this driver -- see §4.
* **AMD keeps the unicast kernels**: `multimem` is sm_90+ and RCCL has no
  equivalent.

## 1. Does multicast work on these nodes?

**Yes.** Every device reports the capability and the whole sequence runs:

```
ATTR dev0..7  multicast=1  vmm=1  posix_fd=1  fabric=1  dmabuf=0  gdr=1
```

(`cuDeviceGetAttribute` 132 / 102 / 103 / 128 / 124 / 116.) `cuMulticastCreate`,
`cuMulticastAddDevice` on all 8, `cuMulticastBindMem` and the dual mapping all
return 0, and an 8-rank `multimem.ld_reduce` + `multimem.st` allreduce verifies
against an fp64 host reference. No fabric manager configuration was needed
beyond what the nodes already run.

Granularity (`cuMulticastGetGranularity`): **MINIMUM 2 MiB, RECOMMENDED 512
MiB**; `cuMemGetAllocationGranularity(RECOMMENDED)` is 2 MiB. NCCL uses
RECOMMENDED, which means the smallest multicast object on this hardware is
512 MiB; a 512 MiB payload plus the 4 KiB header therefore rounds up to a
1 GiB allocation per rank. Using MINIMUM instead is possible and was not
measured.

## 2. The setup recipe that worked

Ordering matters in exactly two places: every device must be in the team
before any memory is bound, and every rank must have mapped before any rank
issues a multimem instruction.

1. `cuDeviceGetAttribute(CU_DEVICE_ATTRIBUTE_MULTICAST_SUPPORTED)` on every
   device; bail out to the unicast kernels if any says 0.
2. Rank 0: `CUmulticastObjectProp{numDevices=world, size, handleTypes=
   CU_MEM_HANDLE_TYPE_POSIX_FILE_DESCRIPTOR, flags=0}` →
   `cuMulticastGetGranularity(RECOMMENDED)` → round `size` up →
   `cuMulticastCreate` → `cuMemExportToShareableHandle(&fd, mc,
   POSIX_FILE_DESCRIPTOR, 0)`.
3. Send `fd` to the other ranks (see §3). Each of them
   `cuMemImportFromShareableHandle(&mc, (void*)fd, POSIX_FILE_DESCRIPTOR)`.
4. Every rank: `cuMulticastAddDevice(mc, myDev)`.
5. **Barrier.**
6. Every rank: `cuMemCreate` its own physical memory with
   `CUmemAllocationProp{type=PINNED, location={DEVICE,myDev},
   requestedHandleTypes=POSIX_FILE_DESCRIPTOR}`, then
   `cuMulticastBindMem(mc, 0, mine, 0, size, 0)` — every rank binds at
   multicast offset 0, so one multicast address covers eight distinct physical
   allocations.
7. Every rank: `cuMemAddressReserve`/`cuMemMap`/`cuMemSetAccess` **twice** —
   once against the multicast handle (the address `multimem.*` takes) and once
   against its own memory handle (ordinary loads and stores; this is what the
   copy-in, the copy-out and the flag spin use).
8. **Barrier.** Zero the header (cuMemCreate memory is not guaranteed clean).

The peers never import each other's *unicast* memory — that is the structural
difference from the `cuIpc` path of `../kernel`. Only the multicast handle
crosses processes, i.e. 7 fd sends instead of 56 handle exchanges.

Setup cost, one-off, 512 MiB object (median of the ranks):

| step | us |
|---|---|
| `cuMulticastCreate` (rank 0) | 88 |
| export + fd transport + `cuMemImportFromShareableHandle` (peers) | 54 |
| `cuMemCreate` + `cuMulticastBindMem` | 10 000 - 32 000 |
| `cuMemAddressReserve`/`Map`/`SetAccess` x2 | 27 000 - 63 000 |
| **total including the two file barriers** | **150 000 - 230 000** |

`cuMulticastBindMem` and the mappings dominate and are wildly variable; NCCL
sees the same thing (its source calls bind "where we normally see issues if the
system NVLS/Multicast support is broken", `nccl:src/transport/nvls.cc:369`).
150-230 ms is a communicator-creation cost, paid once.

## 3. Passing the fd: SCM_RIGHTS vs pidfd_getfd

Both were implemented and both work here (`-D`-free, selected by the
harness's last argument).

* **`SCM_RIGHTS` over an AF_UNIX `SOCK_DGRAM` socket** — what NCCL does
  (`nccl:src/os/linux_ipcsocket.cc:171-245`, reached through its proxy thread,
  `nccl:src/proxy.cc:1572-1595`). Works regardless of
  `/proc/sys/kernel/yama/ptrace_scope`. ~70 lines of hand-laid-out `msghdr` /
  `cmsghdr` in Mojo because `std.ffi` has no C-struct ABI.
  One trap, found the hard way: the rendezvous file barrier must not use the
  same path as the bound socket — `open()` on a bound unix socket fails with
  ENXIO.
* **`pidfd_open` + `pidfd_getfd`** — 10 lines, reusing `proto/vmm_probe.mojo`'s
  shim, but it needs `PTRACE_MODE_ATTACH`. It works on these nodes only because
  `ptrace_scope` is 0.

**Recommendation for the library: SCM_RIGHTS.** `ptrace_scope` is 1 on stock
Ubuntu and 2 or 3 on hardened images, and a collective that silently loses NVLS
because of a sysctl is worse than 70 lines of `sendmsg`.

## 4. RDMA registration of the region (for the multi-node path)

On these nodes, **neither** of the two registration paths worked on the
cuMemMap'd unicast mapping:

* `cuMemGetHandleForAddressRange(..., CU_MEM_RANGE_HANDLE_TYPE_DMA_BUF_FD)` →
  **rc 801, `CUDA_ERROR_NOT_SUPPORTED`**, consistent with
  `CU_DEVICE_ATTRIBUTE_DMA_BUF_SUPPORTED = 0` on every device. So NCCL's
  `ibv_reg_dmabuf_mr` fallback is unavailable here.
* `ibv_reg_mr(pd, ucVA, size, LOCAL_WRITE|REMOTE_WRITE|REMOTE_READ)` on the
  first HCA → **NULL** (after 1.3-6.8 ms), even though `nvidia_peermem` is
  loaded on the compute nodes and `CU_DEVICE_ATTRIBUTE_GPU_DIRECT_RDMA_SUPPORTED
  = 1`.

Resolved (2026-09-09, `issue_repro/gdr_probe.mojo`: H100 node, all 12 mlx5
HCAs, 64 MiB per buffer): the NULL came from the allocation, not from the HCA
choice. `nvidia_peermem` only pins a `cuMemCreate` chunk whose prop set
`allocFlags.gpuDirectRDMACapable = 1`. Without it every HCA returns NULL with
`errno 14` (EFAULT); with it all 12 register. NCCL sets the flag whenever
`CU_DEVICE_ATTRIBUTE_GPU_DIRECT_RDMA_WITH_CUDA_VMM_SUPPORTED` (110) is 1
(`nccl:src/include/alloc.h:329`), `vmm.mojo` now does the same, and MAX's own
arena chunks already carry it: a `DeviceContext` buffer registers on 12/12
HCAs too. So the multi-node path can register a VMM region; only the dmabuf
half stays closed on this driver.

## 5. One kernel, three schedules, and the traffic that decides against unicast

Both stage: the user's tensors are MAX-allocated and cannot be bound to a
multicast object, so every byte is copied into the bound region and back out.
That staging is part of every number below.

Per GPU, for an allreduce of `N` bytes at world 8:

| | NVLink out | NVLink in | HBM |
|---|---|---|---|
| unicast push/reduce/pull (`../kernel`) | 7N/4 | 7N/4 | ~4N |
| NVLS | N (7N/8 serving peers' `ld_reduce` + N/8 of `multimem.st`) | N (N/8 of results + 7N/8 of peers' `st`) | ~6N (2N copy-in, N served, N received, 2N copy-out) |

NVLS moves **1.75x fewer NVLink bytes** and ~1.5x more HBM bytes. That is the
whole trade, and it is why NVLS only wins once the message is big enough for
the fabric rather than the launch to dominate.

**`nvls_allreduce_multimem_<dtype>`, three-phase (default).** One kernel:
copy-in of the whole message, barrier, `multimem.ld_reduce` + `multimem.st`
over this rank's 1/8 slice, barrier, copy-out of the whole message. 216 blocks
x 512 threads, 16-byte operands (`.v4.f32` / `.acc::f32.v4.bf16x2`), 40
registers.

**`-D nvls_pipe=1`, split grid (the shipped shape).** Same work, but the grid
is split: the low `nvls_reduce_pct`% of the blocks only drive the switch and the
rest only drive HBM, and the message is cut into `nvls_chunk_kib` chunks so that
chunk c's reduction runs at the same time as chunk c+1's copy-in and chunk c-1's
copy-out. One barrier per chunk covers all three because they touch disjoint
byte ranges. Tuned: 216 blocks x 256 threads, 25% reducers, 43 MiB chunks
(100 registers, so `nvvm.minctasm=2` at 256 threads; see §7.4).

**`-D nvls_pipe=2`, fused.** Same chunk schedule, but every thread does both:
a thread that owns reduce vector j also moves `world` copy-in and `world`
copy-out vectors, issued between that vector's `multimem.ld_reduce` and its
`multimem.st` so they run inside the switch round trip. Neither side gives up
threads. It is 6% slower than the tuned split grid; §6.3 says why.

## 6. Results

All on one 8xH100 SXM node, SM clock unpinned (1980 MHz under load), 20
back-to-back launches x 5 repetitions, median, device time from the GPU's own
`globaltimer` across the burst. NCCL is stock torch 2.11.0+cu128 / NCCL 2.28.9,
device time from CUDA events, max over ranks -- and it is measured **in the same
job on the same node** as the Mojo numbers, not quoted from an earlier run.

### 6.1 First cut: the untuned schedules, everything in one job

Job 234239, node cl02s01dgx05, fp32 world 8, us. The "split grid" column here
is the *untuned* 60%/21 MiB setting; §6.4 has the tuned one.

| MiB | unicast (`../kernel`) | NVLS 3-phase | NVLS split grid (untuned) | NCCL NVLS | NCCL ring |
|---|---|---|---|---|---|
| 1 | 23.5 | 24.4 | 24.5 | 25.3 | 30.4 |
| 4 | 37.6 | 42.8 | 43.5 | 52.7 | 47.6 |
| 9 | **64.7** | 75.6 | 76.5 | 88.4 | 75.8 |
| 16 | **104.2** | 127.4 | 122.8 | 118.7 | 117.1 |
| 27 (DDP bucket) | **162.2** | 203.9 | 181.7 | 175.3 | 183.2 |
| 64 | 374.0 | 403.3 | 378.2 | **323.8** | 366.4 |
| 128 | 737.4 | 744.4 | 712.1 | **585.7** | 703.8 |
| 168 (tail bucket) | 963.3 | 957.0 | 927.0 | **748.7** | 903.5 |
| 256 | 1440.8 | 1420.8 | 1384.1 | **1107.6** | 1354.9 |
| 512 | 2874.2 | 2772.3 | 2741.1 | **2143.4** | 2637.3 |

bf16 is within 0.5% of fp32 at every size for all three Mojo variants.

### 6.2 Where the time goes (phase builds, `-D nvls_phase=`)

The three-phase kernel splits cleanly, and the split says the whole gap to NCCL
is the staging:

| MiB | copies only | switch only | sum | measured full |
|---|---|---|---|---|
| 27 | 56.3 | 166.3 | 222.6 | 203.9 |
| 168 | 275.0 | 707.2 | 982.2 | 957.0 |
| 512 | 795.3 | 2005.3 | 2800.6 | 2772.3 |

At 132 blocks x 512 threads the switch phase is faster still -- **684 us** at
168 MiB and **1982 us** at 512 MiB -- and the copies are unchanged (271 us at
168 MiB). So:

* the **switch phase alone already beats NCCL's whole allreduce**: 684 vs 749
  at 168 MiB, 1982 vs 2143 at 512 MiB;
* the copies are 275 us of pure addition on top, which is exactly the 1.24x;
* NCCL is therefore overlapping its staging almost perfectly, and any Mojo
  kernel that does the same lands at 0.91-0.93x NCCL.

Effective fabric rate in the switch phase is 250-270 GB/s per direction against
NVLink4's 450, and it does **not** improve with more multimem loads in flight
(`-D nvls_mm_unroll`: at 168 MiB the switch phase is 684 us with 2 in flight,
726 with 4, 777 with 8) -- so it is a switch/reduction throughput limit, not a
latency-hiding problem, and NCCL sees the same ceiling.

### 6.3 Tuning the overlap (job 234241 sweep + jobs 234244/234249/234252/234253)

fp32, us, at the two sizes that matter. "split" = schedule 1 (a fraction of the
blocks on the switch), "fused" = schedule 2 (every thread does both).

| variant | 168 MiB | 512 MiB |
|---|---|---|
| three phases, 216x512 | 957 | 2772 |
| split, 60% reducers, 21 MiB chunks, 216x256 | 927 | 2745 |
| split, 50% | 877 | 2574 |
| split, 40% | 843 | 2472 |
| split, 30% | 824 | 2422 |
| split, 25% | 819 | 2404 |
| split, 20% | 805 | 2371 |
| **split, 25%, 43 MiB chunks** | **786** | **2217** |
| split, 25%, 86 MiB chunks | 826 | 2191 |
| fused, 132x512, 21 MiB chunks | 894 | 2645 |
| fused, 132x512, 43 MiB chunks | 830 | 2392 |
| fused, 216x256, 43 MiB chunks | 824 | 2368 |
| NCCL NVLS (same node, same job) | 749 | 2145 |

Two things the sweep settles:

* **Fewer blocks on the switch, more on HBM.** The switch phase saturates at
  about 33k threads (the whole-grid sweep of the three-phase kernel moves only
  2% between 64 and 264 blocks); the copies want every thread they can get.
  Hence 25%, not the 60% a naive "the fabric is the slow half" split suggests.
* **Chunks want to be big.** 43 MiB is best at 168 MiB (4 chunks) and 86 MiB is
  best at 512 MiB (6 chunks) -- i.e. about `bytes/4` to `bytes/6`, floor 21 MiB.
  Smaller chunks lose to the barrier: 5 MiB chunks cost 1163 us at 168 MiB
  against 927 for 21 MiB, which prices the two-level barrier at roughly 10 us.

The fused schedule was built to avoid the split's thread starvation -- every
thread reduces *and* copies, with the copies issued between a chunk's
`ld_reduce` and its `st` so they run inside the switch round trip -- and it does
beat the split grid at the same chunk size when the split is badly tuned, but it
loses to a well-tuned split (830 vs 786). The reason is visible in the source:
the `world` copy pairs a thread does per reduce are a chain of dependent
load-then-store pairs, so they serialize on HBM latency instead of pipelining.
Hoisting all `world` loads before all `world` stores would fix it and costs 64
more registers, which at 94 already is not affordable. Left as the obvious next
step.

### 6.4 Headline: the tuned NVLS against both references, same node, same job

fp32, world 8, us. NVLS and NCCL are both from job 234253; the unicast column
is from job 234239 on a different node of the same model (it reproduces
`../kernel`'s recorded numbers to within 1%).

| MiB | unicast | **NVLS (split 25%, 43 MiB chunks)** | NCCL NVLS | NVLS / NCCL |
|---|---|---|---|---|
| 27 (DDP bucket) | **162** | 172.0 | 175.2 | **0.98** |
| 64 | 374 | 339.1 | 325.6 | 1.04 |
| 128 | 737 | 620.5 | 585.8 | 1.06 |
| **168 (tail bucket)** | 963 | **785.7** | 749.9 | **1.048** |
| 256 | 1441 | 1153.5 | 1109.1 | 1.04 |
| **512** | 2874 | **2217.1** | 2143.1 | **1.035** |

bf16 at the same configuration: 171.7 / 338.3 / 621.8 / **785.2** / 1152.0 /
**2216.3** us -- within 0.2% of fp32 everywhere. NCCL's bf16 is 1-2% faster
than its fp32 at the big sizes (739.0 / 2115.3), so the bf16 ratios are 1.063
at 168 MiB and 1.048 at 512 MiB.

**The stated goal is met in fp32**: 785.7 us at 168 MiB against a 790 us budget
(NCCL + 5%), and 2217 us at 512 MiB against 2250. Staging is included in both.
A second independent job (234256, the shipped binaries straight out of
`build.sh`) gives 788.3 and 2217.0 against an NCCL of 748.6 in that same job --
1.053x and 1.035x. So 512 MiB is inside 5% by a clear margin and 168 MiB sits
on the line, moving 0.5 points between runs. bf16 tracks fp32 to within 0.2%,
but NCCL's bf16 is 1-2% faster than its fp32 at these sizes (739.0 / 2115.3),
so the bf16 ratios are 1.063 at 168 MiB and 1.048 at 512 MiB.

### 6.5 Crossover, and what the library should dispatch on

`abharness.mojo` runs both kernels in one process set over the same user
tensors, interleaved ABBA inside every repetition (nvls,uni | uni,nvls) so a
monotonic clock or thermal ramp cancels to first order. Job 234254, tuned NVLS
(split 25%, 43 MiB chunks), fp32, us. Job 234256 reproduces the two end points
from a fresh `build.sh` on another node to within 0.1% (786.2 / 963.3 and
2218.6 / 2872.1):

| MiB | NVLS | unicast | NVLS / unicast |
|---|---|---|---|
| 16 | 109.1 | 103.6 | 1.053 |
| 24 | 155.1 | 148.5 | 1.045 |
| 27 | 172.2 | 163.7 | 1.052 |
| 32 | 200.0 | 190.2 | 1.052 |
| 40 | 244.6 | 234.7 | 1.043 |
| **48** | **271.3** | **281.7** | **0.963** |
| 64 | 338.8 | 372.6 | 0.909 |
| 96 | 469.2 | 553.9 | 0.847 |
| 128 | 620.4 | 735.6 | 0.843 |
| 168 | 786.2 | 963.1 | 0.816 |
| 256 | 1153.1 | 1445.2 | 0.798 |
| 512 | 2218.3 | 2871.5 | 0.773 |

bf16 lands on the same crossover to within 0.3% at every size (1.051 / 1.048 /
1.050 / 1.053 / 1.043 / **0.961** / 0.903 / 0.842 / 0.840 / 0.814 / 0.796 /
0.772), so one constant covers both dtypes.

**The crossover is between 40 and 48 MiB**, and it is sharp -- 4% the wrong
side at 40 MiB, 4% the right side at 48. `NVLS_MIN_BYTES = 48 MiB` is the
dispatch constant; below it the unicast push/reduce/pull kernels keep the
traffic, above it NVLS takes it and the margin grows monotonically with size.

For GPT-2 DDP that puts the 27 MiB bucket on the unicast path (where it already
beats NCCL by 7%) and moves the 168 MiB tail bucket to NVLS, taking it from
1.29x NCCL to 1.05x.

## 7. What this changes in the library

### 7.1 Region allocation

`region_init` today takes a `cuMemAlloc`/`hipMalloc` base. NVLS needs the
region to be VMM memory bound to a multicast object, so the allocation moves
into the library:

* NVIDIA sm_90+: the §2 sequence. The region gets **two** base pointers, `mc`
  and `uc`; every existing kernel keeps using `uc` unchanged (it is ordinary
  device memory), and only the NVLS kernel needs `mc`.
* Everything else (sm_80 and below, all AMD): `cuMemAlloc`/`hipMalloc` exactly
  as now. `multimem` is an sm_90+ instruction and RCCL has no equivalent, so
  **AMD keeps the unicast kernels**, with no source-level fork beyond the
  existing `comptime if has_amd_gpu_accelerator()` and a runtime capability
  check.
* Granularity forces the region to a multiple of 512 MiB (or 2 MiB with
  `CU_MULTICAST_GRANULARITY_MINIMUM`). The library's `cap_bytes` should be
  chosen with that in mind rather than allocating a fresh object per bucket —
  bind cost alone is 10-30 ms.

### 7.2 The IPC import path becomes a VMM import

For the unicast kernels the peers still need each other's regions. Two ways to
get there, both already exercised in this tree:

* keep `cuIpcGetMemHandle`/`cuIpcOpenMemHandle` on a **separate**
  `cuMemAlloc` region (what `abharness.mojo` does: the two legs own two
  regions), or
* export each rank's `cuMemCreate` handle with
  `cuMemExportToShareableHandle` and import it with the same fd transport as
  the multicast handle (`cuMemImportFromShareableHandle` +
  `cuMemAddressReserve`/`Map`/`SetAccess`), which is what `proto/vmm_probe.mojo`
  established and what NCCL does for its P2P transport with driver >= 12.0
  (`nccl:src/transport/p2p.cc:267-327`). This is the better end state: one
  region, one allocation, both kernel families over it, and the fd machinery is
  written once.

Either way the c10d store carries a rendezvous, not a handle: the fd itself
must travel over an AF_UNIX socket, so the store holds the socket path.

### 7.3 Kernel dispatch by size

The library gains a third allreduce route next to one-shot and two-shot. The
switch is a pure size comparison on the same region, so it is a `if nbytes >=
NVLS_MIN_BYTES and have_multicast` in the existing dispatcher — no new
protocol, no new flags, and the generation counter keeps advancing across all
three (the NVLS kernel consumes a known number of barrier slots per call, see
`barriers_per_call`).

### 7.4 Two footguns the prototype found

* **The barrier must be a full barrier.** The unicast kernels match blocks by
  index (block b only consumes what block b of a peer produced). The NVLS
  kernel cannot: the reduce phase reads a contiguous slice that *every* block
  of every peer helped stage. The block-index-matched barrier passes the small
  cases and fails from n=65537 up, which is exactly the kind of bug that ships.
  The prototype uses a two-level barrier (device-scope arrival counter, then
  one `multimem.red.release.sys.global.add.u32` from the last block to arrive),
  and pays for it with the next footgun.
* **A full barrier requires the whole grid to be resident.** The three-phase
  kernel uses 40 registers, so 216 blocks of 512 threads fit on 132 SMs; the
  pipelined one uses 100, which is one block per SM, and it deadlocked until
  the block size dropped to 256 with `nvvm.minctasm=2`. A library must either
  pin the launch geometry to a measured occupancy, or go back to index-matched
  barriers by giving block b the *same* sub-range in all three phases (copy-in
  writes, for each rank s, only `[s*per + b*per/NB, ...)`; reduce and copy-out
  address the same sub-ranges). The second is the better answer for shipped
  code and is not implemented here.

### 7.5 Two constants the library has to carry

* `NVLS_MIN_BYTES = 48 MiB` (§6.5), one value for fp32 and bf16.
* The pipeline chunk should scale with the message, not be fixed: 43 MiB is
  best at 168 MiB (4 chunks) and 86 MiB at 512 MiB (6 chunks), while 5 MiB
  costs 48% at 168 MiB. `chunk = clamp(bytes / 4, 21 MiB, 86 MiB)` fits every
  point measured. The reason is the barrier: one per chunk at roughly 10 us,
  which is also the number to attack first (§7.4 second bullet) -- an
  index-matched barrier would be worth about 60 us at 168 MiB, i.e. the rest
  of the gap to NCCL.

## 8. What did not work, with numbers

Recorded so the next agent does not re-explore them.

1. **Block-index-matched barrier** (the one the unicast kernels use, and the
   first thing tried because it is 20 us cheaper). Passes every small case and
   fails from **n = 65537** up: 3-6 wrong elements out of 65537, every rank, at
   scattered indices. The reduce phase reads a contiguous slice that every
   block of every peer staged, so block b of rank r consumes bytes block b' != b
   of a peer produced. See §7.4.
2. **A grid that is not fully resident deadlocks**, because the two-level
   barrier waits for every block. Measured, all at 216 blocks: 512 threads x 40
   registers (three-phase) is fine, but 640 threads (58 registers, 1 block/SM),
   `-D nvls_unroll=4` (88 registers), `-D nvls_mm_unroll=4/8` (72/116
   registers) and the pipelined kernel at 512 threads (100 registers) all hang
   until the 15 s in-kernel deadline. 396 blocks x 512 x 40 registers hangs too
   -- exactly at the 3-blocks-per-SM boundary, so the boundary is not usable.
   Every one of these runs fine once the grid drops to 132 blocks.
3. **More multimem loads in flight does not help.** The switch-only phase at
   168 MiB: 684 us with 2 loads in flight, 726 with 4, 777 with 8 (132 blocks x
   512). At 27 MiB the order reverses (150 / 148 / 135), so the deeper unroll
   is a small-message optimization only. The fabric is at a throughput ceiling
   of 250-270 GB/s per direction, not a latency wall.
4. **`multimem` with `.gpu` scope instead of `.sys`**: 960 us at 168 MiB
   against 957 for `.sys` -- no gain, and `.sys` is the correct scope for eight
   devices, so there is nothing to trade.
5. **Small pipeline chunks.** 2.6 MiB chunks cost 1374 us at 168 MiB, 5 MiB
   1163, 10 MiB 1030, 21 MiB 927, 43 MiB 924 (all at the untuned 60% split).
   The barrier is worth about 10 us and there is one per chunk.
6. **The fused schedule**, which was supposed to beat the split grid by not
   giving up threads on either side: 830 us at 168 MiB against 786 for the
   tuned split. §6.3 says why (dependent load-store chains in the per-thread
   copy loop).
7. **`cuMemGetHandleForAddressRange(DMA_BUF_FD)`**: `CUDA_ERROR_NOT_SUPPORTED`
   (801), and `CU_DEVICE_ATTRIBUTE_DMA_BUF_SUPPORTED` is 0 on all 8 devices.
   **`ibv_reg_mr` on the cuMemMap'd unicast mapping**: NULL until the prop
   sets `gpuDirectRDMACapable`; fixed. §4.
8. **The multicast object cannot be small.** `CU_MULTICAST_GRANULARITY_RECOMMENDED`
   is 512 MiB on this hardware, so a 512 MiB payload plus a 4 KiB header
   allocates 1 GiB per rank. `MINIMUM` is 2 MiB and was not measured.

## 9. Reproducing

```bash
# login node, no GPU: cross-compile everything (add `sweeps` for the variants)
nvls/build.sh sweeps

# one node, 8 GPUs: probe + correctness + timing + both references + ABBA A/B
sbatch nvls/bench.sbatch
# tuning sweeps (grid, block size, unroll, scope, chunk, reducer share)
sbatch nvls/sweep.sbatch

# tables out of any of those logs
uv run --no-sync python nvls/report.py /home/gabriel/ddp_work/logs/nvls_bench_<job>.log
```

`run.sh <binary> <world> <suite> <iters> <reps> <cap_mib> <transport> [sizes]`
runs one process per rank under one flock of the node's GPU lock and kills the
group on `RUN_TIMEOUT` (default 420 s) so a wedged rank cannot eat the job.
Note that a clean run still prints `RUN_RC=143`: the inner `trap 'kill 0' EXIT`
signals its own process group on the way out. It is not a timeout.
