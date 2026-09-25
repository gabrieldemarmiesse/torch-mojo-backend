# Mojo intra-node collectives — measured results

> Historical record: the file names below predate the NCCL-layout split of
> `tmb/ccl/`; `agents_docs/distributed.md` ("File layout") maps them to
> today's files.

Deliverable: `collectives_kernels.mojo` (this directory). Harness: `harness.mojo`
+ `validate.sbatch` (correctness + the tables below, NCCL legs interleaved),
`ab.sbatch` (the NCCL A/B alone), `sweep.sbatch` / `tune.sbatch` /
`diag.sbatch` (the tuning sweeps), `bench.sbatch`, `smoke.sbatch`,
`split.sbatch` + `split_report.py` (the split allreduce of §10).
`./build.sh` builds the per-dtype harnesses and the assembly dumps;
`./build.sh sweeps` also builds the `-D`-tuned variants the sweep scripts
expect. Raw logs: `/home/gabriel/ddp_work/logs/ccl_*.log`.

All numbers: 8xH100 SXM (NVSwitch), one process per GPU, regions exchanged with
`cuIpcGetMemHandle`/`cuIpcOpenMemHandle`, **staged** (user tensors in
MAX-allocated memory; every copy the design needs is inside the measurement),
20 back-to-back launches x 5 reps, median, max over ranks. Device time = wall
over the 20 launches / 20, cross-checked against an in-kernel `globaltimer`
span recorded by stamp kernels on the same stream (they agree to <0.5%
everywhere; the CSV prints both). SM clock 1980 MHz (boost, unpinned — the
SLURM `--comment=gpu_clock` pin does not take on this cluster; checked per job).

Nodes: `cl02s01dgx18`, `cl02s01dgx04`, `cl02s01dgx16`, `cl02s04dgx01`. Node to
node spread on the same code is ~2%, so cross-job comparisons below are only
made within a job.

## 1. Headline: same-node A/B against NCCL (job 233776, shipped constants)

Legs interleaved within each round (NCCL NVLS, mojo fp32, mojo bf16, NCCL ring),
3 rounds, same node, same job. NCCL = stock torch 2.11.0+cu128 / NCCL 2.28.9,
device time from CUDA events, max over ranks — the same statistic the Mojo
harness prints.

fp32, world 8 (us):

| bytes | **mojo** | NCCL default (NVLS) | NCCL ring (`NCCL_NVLS_ENABLE=0`) | mojo / NVLS | target |
|---|---|---|---|---|---|
| 4 B | **9.4** | 18.4 | 17.5 | 0.51x | <= 25 ✔ |
| 256 KiB | **15.3** | 19.9 | 19.5 | 0.77x | — |
| 512 KiB | **21.4** | 19.6 | 19.6 | 1.09x | — |
| 1 MiB | **23.6** | 24.7 | 25.0 | 0.96x | — |
| 9 MiB | **64.6** | 85.8 | 76.4 | 0.75x | <= 91 ✔ |
| **27 MiB** (DDP bucket) | **163.8** | 174.9 | 181.8 | **0.94x** | <= 184 ✔ |
| 168 MiB (tail bucket) | **958** | 750 | 899 | 1.28x | "as close as unicast gets" |
| 512 MiB | **2860** | 2146 | 2636 | 1.33x | — |

bf16, world 8: 9.4 / 15.4 / 21.4 / 23.6 / 64.8 / **163.1** / 959 / 2856 us —
within 0.5% of fp32 at every size (target: 2%).

world 4 (fp32): 8.4 / 10.9 / 14.0 / 18.3 / 56.5 / 135.7 / 814 / 2437 us.
world 2 (fp32): 8.0 / 8.9 / 10.3 / 15.4 / 40.9 / 101.0 / 623 / 1856 us.

**The three stated targets are met with the staging included**, and the two
sizes that matter for GPT-2 DDP (9 and 27 MiB) beat NCCL's best algorithm on
the same node by 25% and 6%. The feasibility study's *direct* kernel (operating
on library-owned buffers, no user tensors involved) was 66 / 165 us at 9 /
27 MiB: **the staging now costs nothing measurable**, against +25% (82 /
210 us) for the naive copy-in/copy-out staging the study measured. The one
size where NCCL wins below 1 MiB is 512 KiB (21.4 vs 19.6), at the top of the
one-shot regime.

## 2. Why the staging became free

The user's tensors are MAX-allocated and cannot be exported by IPC, so peers can
only read the library's region. Instead of copying the input in and the result
out, the kernel makes the *transfers themselves* do the staging:

* **push** — each rank reads its own input straight from user memory and writes
  shard *s* into peer *s*'s slot. That write is the copy-in and it is the NVLink
  transfer the reduce-scatter needed anyway.
* **reduce** — each rank sums the `world` contributions to its own shard (its own
  from user memory, the peers' from its region) and writes the result both to
  its region and to its slice of the user output.
* **pull** — each rank reads the peers' reduced shards straight into the user
  output.

NVLink traffic per GPU is `2*(world-1)/world*bytes` — the unicast minimum, the
same as a direct reduce-scatter + all-gather — and no local byte is copied that
the direct kernel would not also copy. Three cross-GPU syncs: a start barrier
(the arena-reuse invariant, see §6) plus one per data dependency.

Messages at or below 512 KiB take a **one-shot** path instead — every rank
pushes its whole input to every peer and reduces locally, `(world-1)x` the
NVLink bytes but one data sync instead of two. Measured crossover (one-shot vs
two-shot, us): 128 KiB 9.3/19.0, 256 KiB 12.3/19.3, 512 KiB 18.1/19.8, 1 MiB
30.0/20.5 (job 233731, before the start barrier added ~2.5 us to both).

## 3. What tops out where: the unicast ceiling at 168 MiB

At 168 MiB the kernel is 1.07x NCCL ring and 1.28x NCCL NVLS. The decomposition
below says why, and it is measured, not modelled.

`allgather` with 21 MiB per rank (= 168 MiB output) moves exactly the same
154 MB of NVLink reads per GPU as the allreduce's pull phase, and nothing else
except a local 22 MB stage copy: **502-508 us**, i.e. **~305 GB/s per GPU per
direction**. The allreduce at 168 MiB is 958-990 us for a push (154 MB out)
plus a pull (154 MB in), i.e. **~315 GB/s per direction** — the same rate. So:

* both NVLink phases already run at the rate a pure peer-copy kernel achieves;
* the local reduce costs **~30 us of the ~980** (980 - 2x~480), not the ~110 us
  a naive HBM model predicts — the memory system absorbs it;
* therefore **pipelining the reduce behind the transfers can win at most ~3%**,
  and was not implemented (see §6, negative results).

The ceiling for this design is the fabric rate under all-to-all traffic:
**~310 GB/s per direction per GPU, 69% of NVLink4's 450 GB/s**. NCCL ring gets
343 GB/s (899 us) with neighbour-only traffic — ~10% better fabric efficiency
that a direct (all-to-all) reduce-scatter/all-gather does not reach by tuning
(every knob was swept, §5); closing it needs a different schedule, an
N-1-step ring, which is a different kernel and would give up the latency win
at 9-27 MiB that matters more for this workload.

**NCCL NVLS's 750 us is out of reach for any unicast kernel**, and not by a
small margin of tuning: `multimem.ld_reduce` has the NVSwitch do the reduction,
so each GPU moves `2*bytes/world` across the fabric instead of
`2*bytes*(world-1)/world` — 7x fewer bytes at world 8. To match it we would need
`cuMulticastCreate` + `cuMemExportToShareableHandle` + fd passing over an
AF_UNIX socket (`SCM_RIGHTS`), and — the part that reaches outside this file —
the region would have to be allocated with `cuMemCreate`/`cuMemMap` instead of
`cuMemAlloc` so it can be bound to the multicast object. MAX's
`DeviceMulticastBuffer` is single-process and cannot be reused across processes.
That is the exact gap; it was not attempted here because the region allocator
belongs to the plumbing layer. **It has since been closed exactly along those
lines** — `nvls_kernels.mojo` and `vmm.mojo`, dispatched above 48 MiB, taking
168 MiB from 990 to 790 us and 512 MiB from 2988 to 2226. Nothing in this file
changed: the six exported names below are untouched and the unicast kernels
still carry everything below the crossover. See the "NVLS" subsection of
agents_docs/distributed.md.

Consequences for GPT-2 DDP: buckets 0-11 (9 and 27 MiB) are faster than NCCL and
overlapped with backward anyway; the exposed 168 MiB tail bucket costs
+0.21 ms/step versus NCCL NVLS, on a ~45 ms step.

## 4. Broadcast and allgather

| collective | size | first design (us) | **shipped design (us)** |
|---|---|---|---|
| broadcast | 27 MiB | 552 (root stages, all read) | **154-157** |
| broadcast | 168 MiB | 3374 | **927-1035** |
| broadcast | 256 MiB | — | **1436** |
| allgather | 3.4 MiB/rank (27 MiB out) | — | **96-98** |
| allgather | 21 MiB/rank (168 MiB out) | — | **502-508** |
| allgather | 32 MiB/rank (256 MiB out) | — | **778** |

(ranges are across nodes, jobs 233776/233789/233794; world 3 also measured:
broadcast 27 MiB 118 us, allgather 9 MiB/rank 77 us.)

The first broadcast put `(world−1)·nbytes` on the root's single outbound link
(552 us at 27 MiB where the fabric can do ~150). The shipped one is **scatter +
all-gather**: the root scatters shard *p* into rank *p*'s stage (nbytes out of
the root, spread over the peers), then every non-root rank gathers the `world`
shards. 3.6× faster, and both phases are permutations, so no link carries more
than `nbytes`. It is out-of-place capable (ncclBroadcast semantics): only the
root reads `send_ptr`, everyone writes `recv_ptr`, and when the two differ on
the root it copies locally instead of gathering its own message back over
NVLink. Allgather is a local stage + a peer gather — already the unicast
minimum — and honours a `stride_bytes` that differs from `nbytes_per_rank` so
the ABI layer can chunk one rank's contribution across calls. DDP's two
~250 MiB parameter broadcasts cost ~1.4 ms each instead of ~5 ms.

## 5. Tuning sweeps

Grid (blocks of 256 threads), fp32 world 8, µs:

| blocks | 4 B | 1 MiB | 9 MiB | 27 MiB | 168 MiB | 512 MiB |
|---|---|---|---|---|---|---|
| 64 | — | — | — | — | 1000 | 2955 |
| 96 | 6.8 | 20.6 | 63.2 | 177.2 | 991 | 2982 |
| 128 | 6.7 | 20.6 | 65.7 | 166.9 | **977** | **2826** |
| 132 | 6.9 | 20.7 | 67.4 | 167.5 | 993 | 2934 |
| 160 | — | — | — | — | 1104 | 3458 |
| **216** | 6.8 | 20.5 | **61.1** | 167.4 | 1002 | 3030 |
| 264 | 6.8 | 20.6 | 66.6 | **164.7** | 951 | 3051 |
| 432 | 6.9 | 20.7 | 64.0 | 169.8 | 1075 | 3331 |
| 864 | 6.7 | 20.5 | 63.5 | 175.0 | 1000 | 3168 |

Two regimes, hence the two constants in the file (`_AR_MAX_BLOCKS` = 216 below
64 MiB, `_AR_BIG_BLOCKS` = 128 at and above it; both fitted on H100 and marked
as such):

* **small/medium**: more threads win because the transfer is latency-bound —
  216 blocks is the flat optimum at 9 MiB (61.1 vs 65.7 at 128).
* **large**: a grid that fits in **one wave** of the 132 SMs wins, because the
  barrier is per block index: with more blocks than SMs the second wave runs the
  *whole* collective after the first, on fewer SMs. 128 blocks is 7% faster than
  216 at 512 MiB (2826 vs 3030). 160 blocks (1.2 waves, a long tail wave) is the
  worst point measured — 3458 µs, 22% worse than 128.

Unroll (16-byte vectors in flight per thread in the copy loops), fp32 world 8:

| unroll | 9 MiB | 27 MiB | 168 MiB | 512 MiB |
|---|---|---|---|---|
| 1 | 62.2 | 168.1 | 1083 | 3222 |
| 2 | 61.3 | 163.9 | 1025 | 3162 |
| **4** | 61.5 | 167.1 | **994** | **3078** |
| 8 | 61.6 | 164.5 | 1026 | 3369 |

Memory-level parallelism matters only in the bandwidth-bound regime (+9% from
U=1 to U=4 at 168 MiB) and is flat below 27 MiB. U=8 regresses (register
pressure). `_UNROLL = 4`.

One-shot/two-shot threshold (job 233731, three interleaved rounds, measured
before the start barrier added ~2.5 us to every leg). Each column is the same
size run by two builds differing only in `-D ccl_oneshot_max`, so the pair
isolates the path, not the size:

| bytes | 64 KiB | 128 KiB | 256 KiB | 512 KiB | 1 MiB | 2 MiB | 4 MiB |
|---|---|---|---|---|---|---|---|
| one-shot | 7.5 | **9.3** | **12.2** | **18.1** | 30.0 | 51.9 | 97.0 |
| two-shot | 7.5 | 19.0 | 19.3 | 19.9 | **20.5** | **24.2** | **34.5** |

The crossover sits just above 512 KiB, so `_ONESHOT_MAX_BYTES = 512 KiB`. Below
256 KiB the one-shot path is 2x faster; above 1 MiB it degrades as
`(world-1)x` the bytes should.

## 6. Negative results (do not re-explore)

* **Pipelining the local reduce behind the transfers** (chunk the shard, give
  each chunk its own pair of flags, and run `reduce(c+1)` concurrently with
  `pull(c)` — half the block's warps on each): designed, costed, **not
  implemented**, because the measurement in §3 bounds the win. The reduce is
  worth ~30 µs of the 988 at 168 MiB, and K chunks recover at most
  `(K−1)/K · 30 µs` while adding `2K` syncs at ~2–3 µs each: K=4 nets ≈ +7 µs,
  i.e. nothing. The design is sound and written down here so nobody re-derives
  it: the matching invariant is per *block*, not per thread (block *b* must
  produce exactly the indices block *b* of its peers consumes, and with a
  grid-stride loop that set is fixed by the loop shape), so threads inside a
  block may be split by role as long as every phase keeps the same
  block→index mapping. Worth revisiting only if the reduce ever gets more
  expensive (a wider dtype conversion, a fused op).
* **A stack array of peer pointers in the reduce loop** (`InlineArray[Pointer,
  MAX_WORLD]`, the natural way to write it, and what MAX's `_multi_gpu_barrier`
  warns about as MOCO-1431): the array is demoted to **local memory**, so the
  PTX shows `ld.local.b64` per pointer per iteration and every payload load
  becomes a generic-address `ld.v4.b32` instead of `ld.global.v4.b32`. Forming
  the address arithmetically inside the unrolled loop fixes it: 18
  `ld.global.v4.b32`, zero local traffic.
* **More blocks**: 432 and 864 are worse everywhere (see the table); the
  intuition "more blocks fill NVLink better" is wrong once the grid exceeds one
  or two waves, because the per-block barrier serialises waves.
* **Broadcast by staging the whole message on the root**: 3.6× slower than
  scatter + all-gather (552 vs 154 us at 27 MiB). Do not "simplify" it back.
* **A byte copier that falls back to single bytes when a pointer is
  misaligned**: `_copy_bytes` chooses 16-byte vectors or a fallback per call,
  and for a byte collective the writer and the reader are different ranks
  looking at different pointer pairs (the root's `send` vs a peer's `recv`), so
  they can disagree. A byte-wise fallback would then give writer and reader
  different index -> block mappings, and the barrier is per block index: block 3
  would read a chunk block 0 wrote, with nothing ordering them. Both paths now
  walk the same 16-byte chunks in the same grid-stride order; only the
  instructions inside a chunk differ. Found by inspection, not by a failing
  test — it needs `send` and `recv` to have different 16-byte alignment.
* **Dropping the start barrier** (two syncs per allreduce instead of three,
  ordering the arena reuse out of the phase structure alone): measured 2.5–3 µs
  cheaper at every size, and **wrong**. That argument only holds for a run of
  identically shaped collectives. DDP interleaves a 4-byte one-shot allreduce
  and byte collectives with 27 MiB two-shot allreduces, whose staging layouts
  overlap; with only the data syncs, rank R's push for generation g+1 is
  concurrent with a slower rank's reads for generation g whenever the two
  generations lay the arena out differently. The start barrier reduces the
  whole question to one invariant (§ "Buffer-reuse invariant" in the source),
  and `harness.mojo mix` — 800 interleaved generations of exactly that pattern,
  both allreduces in place so a single corrupted generation survives to the end
  — is the regression test.

## 7. Correctness

`harness.mojo verify` (job 233776, 94 passing cases) checks every result against
a host reference recomputed from the same deterministic splitmix64 fill (no
RNG):

* dtypes float32, bfloat16, float16, int32, int64 — world 8; plus float32 world
  4, 2 and **1**, int64 world 3, bfloat16 world 5, int32 world 7 (odd worlds
  take the runtime-`world` kernel instantiation; world 1 degenerates to
  `out = scale * in` and is exercised, not special-cased away).
* sizes 1, 2, 3, 1003, one page, 65537 and `(cap−1)/element_size` elements — i.e.
  ragged, non-multiples of the 16-byte vector width, and the 4-byte case.
* every size run twice: out-of-place and **in-place** (`in_ptr == out_ptr`).
* float32 is checked **exactly** (the fill is k/128 with |k| ≤ 128, so the sum of
  8 is exact in fp32 and the ×1/8 scale is a power of two); bf16/fp16 to half a
  bf16 ulp; **int64 against an Int64 reference with samples of magnitude 2⁵⁶**,
  which no fp64 reference could verify.
* the region's error word is read back after every case (it stays 0).

`harness.mojo mix` is the interleaving regression test: 200 rounds x (4-byte
one-shot allreduce, 27 MiB two-shot allreduce, 2000-byte broadcast, 8-byte
allgather) = **800 generations** whose staging layouts overlap, both allreduces
in place with `scale = 1/world` so the value is idempotent once every rank holds
the mean and a single corrupted generation anywhere in the run is still visible
in the final buffer. Passes for float32 and bfloat16 at world 8, float32 at world 4 and bfloat16 at
world 5 (jobs 233776, 233789, 233794). This is the test that the start barrier
exists for.

Broadcast and allgather results are checked byte-for-byte against the same hash
(sampled every 997 bytes) in the `ops` suite.

## 8. Build

Both targets build from the one file, no vendor `#ifdef` in the device code:

```
uv run --no-sync mojo build harness.mojo -I . --target-accelerator sm_90a  -D dtype=float32 -o harness_float32
uv run --no-sync mojo build harness.mojo -I . --target-accelerator gfx942  -D dtype=float32 -o harness_f32_gfx942
```

`collectives_kernels.mojo` compiles warning-free on both. The emitted device
code is what NCCL/RCCL rely on (`asm/` holds the dumps, produced by
`dump_asm.mojo` with no GPU present):

| | sm_90a | gfx942 |
|---|---|---|
| flag publish | `st.release.sys.global.b64` | `global_store … sc0 sc1` + `buffer_wbl2 sc0 sc1` |
| flag wait | `ld.acquire.sys.global.b64` | `global_load … sc0 sc1` + `buffer_inv sc0 sc1` |
| payload | `ld/st.global.v4.b32` (18/12 in the allreduce) | `global_load/store_dwordx4` (18/12) |
| block barrier | `bar.sync` | `s_barrier` |
| deadline | `%globaltimer` | `s_memrealtime` |

Blocks are 256 threads (RCCL's gfx942 maximum) and every layout is wave-64
safe. The file has exactly **one** comptime vendor gate, in `_sync`: gfx942
emits its workgroup barrier as `s_waitcnt lgkmcnt(0); s_barrier`, which does
*not* wait on vector memory, so another wave's payload stores can still be in
flight when one thread publishes the flag. RCCL solves this by putting
`vmcnt(0)` inside its block barrier; the portable spelling is a release fence
in every thread, which lowers to `s_waitcnt vmcnt(0)` + `buffer_wbl2 sc0 sc1`.
NVIDIA needs nothing there (`bar.sync` is a CTA-scope fence and the release
store is cumulative over it — what NCCL's `postPeer` relies on), and the gate
is comptime, so the sm_90a instruction mix is byte-identical with and without
it (checked by diffing the dumps).

AMD numbers are unmeasured (no MI300A here). Two things the AMD host side will
need, from RCCL's source: the region must be allocated **uncached**
(`hipExtMallocWithFlags(hipDeviceMallocUncached)`) — RCCL's precondition for
polled flags on MI300 — and the tuning constants above are H100 fits and are
marked as such in the source.

## 9. What the ABI layer needs to know

(§10 adds three more names for the multi-node path; everything below applies to
them unchanged.)

Six exported names, matching the contract exactly: `signal_bytes`,
`error_offset`, `region_init(ctx, region)` (no stream — it blocks),
`allreduce[dtype](ctx, stream, ...)`, `broadcast(...)`,
`allgather(..., stride_bytes=-1)`. `MAX_WORLD = 8`. Everything is enqueued on
the stream passed in and returns right after the enqueue; `ctx` is only the
compile/cache handle (`DeviceFunction`s are cached process-globally per
`ctx.id()`, so the per-call cost is the enqueue, not a ~180 us
`compile_function`).

Preconditions the file enforces by raising, all of them cheap host-side checks:

* `numel * size_of[dtype]() <= cap_bytes`, `nbytes <= cap_bytes`,
  `nbytes_per_rank <= cap_bytes` — the caller chunks anything larger (it does).
* **`in_ptr` and `out_ptr` of `allreduce` must be 16-byte aligned.** The payload
  loops use 16-byte vectors, which fault on a misaligned address. Every
  allocator-returned pointer satisfies this and so does every chunk offset the
  ABI layer forms (they are multiples of `cap_bytes`); a mid-tensor *view* may
  not, and such a tensor must be staged into an aligned buffer by the caller
  rather than have the kernel guess. `broadcast`/`allgather` need no such rule —
  the byte copier checks alignment at run time and falls back to a scalar loop.
* `world` in `1..8` (world 1 works and degenerates to `out = scale * in`),
  `rank < world`, `generation >= 1`, `cap_bytes` a positive multiple of 4096.
* `scale` is applied on the final write for floating-point dtypes and ignored
  for integers (the caller passes 1.0 there, which is what NCCL's `ncclAvg`
  does anyway).

Two notes on the surrounding layer:

* **The error word is device memory.** `error_offset()` is a byte offset into
  the region, which is `cuMemAlloc`/`hipMalloc` memory: it cannot be read by
  dereferencing a host pointer (that faults or reads garbage). Copy it back with
  `cuMemcpyDtoH` / a one-thread kernel, as `harness.mojo`'s `check_error` does.
  A nonzero value is `code * 1_000_000 + phase` and means some block gave up
  waiting for a peer (60 s deadline, measured with the GPU's own timer, never
  compared across GPUs); the collective's result is undefined from that
  generation on.
* `generation` must be strictly increasing per communicator **across all
  collective kinds** — the flag values are `generation * 8 + phase` and the
  start barrier's whole job is to order one generation's writes after the
  previous generation's reads. A repeated or decreasing generation silently
  passes barriers it should not.

## 10. Split allreduce for the multi-node path (job 233937)

The hierarchical allreduce of §7 of the feasibility study needs the intra-node
allreduce cut in half so the vendor library can allreduce one shard across
nodes in between. Three names were added; the six existing exports are
untouched (their device code is **byte-identical** before and after -- checked
by diffing `asm/{ar2,ar1,bcast,ag}_{sm_90a,gfx942}.asm` against a build of the
previous file).

```
shard_range(numel, world, rank, elem_bytes) -> (offset_elems, count_elems)
reduce_scatter_stage[dtype](ctx, stream, rank, world, regions, in_ptr,
                            numel, cap_bytes, generation)
allgather_finish[dtype](ctx, stream, rank, world, regions, out_ptr,
                        numel, cap_bytes, scale, generation)
```

`rank`/`world`/`regions` are the **node-local** group. The sequence per bucket:

```
reduce_scatter_stage(g)        push + local reduce; my shard, SUM over the
                               node times `scale` (default 1), lands in MY
                               stage_out
<vendor allreduce, in place>   region + signal_bytes() + cap_bytes
                               + offset*elem_bytes, count elements, SAME stream
allgather_finish(g+1)          start barrier, then pull every rank's shard
                               (mine included) into out_ptr, times `scale`
```

Preconditions are `allreduce`'s (16-byte aligned buffer, `numel*elem_bytes <=
cap_bytes`, strictly increasing generation); the pair costs **two**
generations. `out_ptr` may be the same buffer as `in_ptr`.

### 10.1 Shard placement

`shard_range` is a different partition from the fused kernel's and
deliberately so: equal shards of `per` elements with `per` rounded up to the
16-byte vector width, the last non-empty shard short, ranks past the end
empty. Every offset is therefore 16-byte aligned, every rank derives the same
table from `(numel, world, elem_bytes)` alone, and the ABI layer can state the
inter-node collective's arguments in one line. Imbalance against the fused
kernel's balanced split is at most one 16-byte vector per rank.

The shard sits in stage_out **at its own element offset**, i.e. stage_out is an
image of the whole buffer of which only my shard is live. The push slots go at
the base of stage_in and are *compacted* to `world-1` (rank s never writes its
own slot): `world` uncompacted slots can be up to `16*world` bytes larger than
stage_in when `numel*elem_bytes == cap_bytes`, which would spill onto rank 0's
shard in stage_out. Checked exhaustively over every dtype width, world and cap.

### 10.2 The three ordering questions, and why no new protocol was needed

1. **A foreign library rewrites my stage_out shard between the two calls.**
   `allgather_finish`'s start barrier is the fence: a rank publishes its
   generation g+1 flags only from inside that kernel, which its stream starts
   only after its inter-node op completed. Seeing peer p's flag therefore
   implies p's shard is final.
2. **Peer p's shard was written by a previous kernel**, not by the thread that
   publishes the flag. Stream order puts that kernel's writes happen-before the
   release store, and the release/acquire pair is system-scoped and cumulative,
   so the acquiring reader sees them (on gfx942 `_sync`'s AMD-only release
   fence supplies the `buffer_wbl2 sc0 sc1` the workgroup barrier omits). The
   consequence is worth stating: block-index matching, which the fused kernel
   needs *within* one launch, is **not** needed across this boundary, so the
   two halves may be launched with different grids -- and they are.
3. **Arena reuse after the pulls.** Nothing new: the next collective of any
   kind opens with a start barrier and a rank reaches it only after its own
   `allgather_finish` retired, so no generation g+2 write can race a generation
   g+1 pull. The split pair just spends two generations. The existing
   buffer-reuse invariant covers it as written -- verified, not assumed, by the
   extended `mix` below.

The ABI layer now runs several such pairs CONCURRENTLY, to overlap the
inter-node hop with the intra-node halves of other chunks (see the
"Multi-node" subsection of `agents_docs/distributed.md`). It needs nothing from this
file to do it: each in-flight chunk is handed a different arena -- a shifted
region base and a smaller `cap_bytes`, carved so the arenas are disjoint --
so every rule above applies per arena, unchanged, and point 3 is what orders
one arena's reuse `PIPE_ARENAS` chunks later. Generations stay strictly
increasing globally (two are reserved per chunk, so a chunk's all-gather is
still its own reduce-scatter's plus one) and therefore per arena.

### 10.3 Split vs fused, 8xH100 SXM, 1980 MHz (job 233937)

`harness.mojo split` runs, per size, five legs back to back in one process:
the fused `allreduce`; each half alone (both are legal standalone collectives);
the stand-in inter-node step alone, so it can be subtracted; and the whole
pair. Device time from `%globaltimer` stamps on the stream, 20 back-to-back
launches x 5 reps, median, max over ranks, three interleaved rounds.

fp32, world 8 (us):

| bytes | fused | rs stage | stub | ag finish | **pair** | pair-fused | minus stub |
|---|---|---|---|---|---|---|---|
| 4 B | 9.5 | 8.8 | 2.1 | 7.1 | **17.6** | +8.1 | +5.9 |
| 9 MiB | 64.7 | 36.3 | 3.1 | 35.5 | **74.8** | +10.1 | +7.1 |
| **27 MiB** | 162.7 | 89.6 | 3.7 | 93.6 | **186.3** | +23.7 | +19.9 |
| 168 MiB | 935.4 | 507.2 | 9.8 | 492.4 | **1003.9** | +68.5 | +58.8 |
| 512 MiB | 2777.6 | 1513.2 | 60.7 | 1457.9 | **3025.8** | +248.2 | +187.5 |

bf16, world 8 (us):

| bytes | fused | rs stage | stub | ag finish | **pair** | pair-fused | minus stub |
|---|---|---|---|---|---|---|---|
| 4 B | 9.6 | 8.9 | 2.3 | 7.0 | **17.9** | +8.3 | +6.1 |
| 9 MiB | 64.7 | 36.3 | 3.3 | 35.5 | **74.9** | +10.2 | +6.9 |
| **27 MiB** | 162.4 | 89.7 | 4.2 | 93.6 | **187.9** | +25.4 | +21.3 |
| 168 MiB | 935.9 | 503.1 | 11.6 | 491.5 | **1012.2** | +76.3 | +64.7 |
| 512 MiB | 2778.8 | 1501.4 | 101.2 | 1458.8 | **3057.7** | +278.9 | +177.7 |

The stand-in is a one-pass read-modify-write of the shard on the same stream
(it adds a rank-independent constant, so the verify can tell "the inter-node
step ran" from "it was skipped"); a real `ncclAllReduce` over `nNodes` costs
much more, and these columns exist so it can be substituted rather than
guessed at.

**Net of the stand-in the pair costs +6 to +8 us up to 9 MiB and +6-7% at
168-512 MiB**, i.e. the promised "fused plus one launch" at small and medium
sizes, growing to a percentage at the bandwidth-bound end. Where it goes:

* one extra kernel launch and one extra 8-way start barrier (~6 us, and that
  is the whole story at 4 B and 9 MiB);
* `allgather_finish` pulls **`world`** shards, not `world-1`: my own shard has
  to come back out of my stage_out because the inter-node step rewrote it,
  where the fused kernel wrote its own shard straight to the user output during
  the reduce. That is `bytes/world` of extra local read at every size, and it
  is the term that grows.

Neither half is slow in itself: `ag finish` at 27 MiB (93.6 us) matches the
standalone `allgather` collective on the same per-rank size (96-98 us, §4), and
`rs stage` (89.6) is its mirror image. The split simply cannot amortise the
second launch the way one kernel does. Whether that matters is a question for
the ABI layer: at the 9 MiB and 27 MiB DDP buckets it is +7 and +20 us against
an inter-node leg of 60-90 us.

### 10.4 Correctness

`harness.mojo split` verifies against a host reference recomputed from the same
deterministic splitmix64 fill, with the stand-in's constant folded into the
expectation -- so a pair that silently skipped the inter-node step fails, and
does not merely look like a rounding difference. 116 passing cases:

* dtypes float32, bfloat16, float16, int32, int64 at world 8; float32 at world
  4, 2 and **1**; bfloat16 world 5, int64 world 3, int32 world 7 (odd worlds
  take the runtime-`world` kernel instantiation).
* eight sizes per configuration, chosen ragged: 1, 2, 3, 6, 1003, 65537,
  7079424 (27 MiB) and 16777215 elements at fp32, and the corresponding counts
  at the other widths -- i.e. non-multiples of the 16-byte vector width, sizes
  smaller than `world` (so most shards are empty), and `cap-4` bytes.
* each size also re-checked after the timing legs, whose last leg is the pair
  with a zero stand-in -- a plain allreduce, checked as one.
* float32 is checked exactly. bf16/fp16 get a wider band than the fused path
  and have to: the split stores the node-local sum in the *wire* dtype (the
  vendor library reduces that buffer, so it cannot stay in fp32) and rounds
  again on the scaled pull, where the fused kernel rounds once. That is a
  property of the hierarchical algorithm, not of this implementation -- NCCL's
  own hierarchical paths do the same.
* the region's error word is read back after every case (stays 0).

`harness.mojo mix` now rotates **eight** generations per round instead of four:
4-byte one-shot allreduce, 27 MiB two-shot allreduce, 2000-byte broadcast,
8-byte allgather, then a 4-byte split pair and a 27 MiB split pair, each with
the stand-in kernel running between its halves. The split pairs run in place
with a zero stand-in so they stay idempotent like the fused calls and a single
corrupted generation anywhere in the run still survives to the final check.
**200 rounds = 1600 generations**, passing for float32 and bfloat16 at world 8,
float32 at world 4 and bfloat16 at world 5. This is the test that the
buffer-reuse invariant of §10.2(3) actually holds with a foreign kernel writing
the arena mid-collective.

### 10.5 Build

Both new kernels build warning-free for both targets from the same source
(`./build.sh`, which now also dumps `asm/rs_*.asm` and `asm/agf_*.asm`):

| | sm_90a | gfx942 |
|---|---|---|
| `_rs_stage_kernel` payload | 13 `ld.global.v4.b32` / 6 `st.global.v4.b32` | 13 `global_load_dwordx4` / 6 `global_store_dwordx4` |
| `_ag_finish_kernel` payload | 10 / 10 | 10 / 10 |
| flags | 3 `st.release.sys.global` + 4 `ld.acquire.sys.global` | `buffer_wbl2 sc0 sc1` / `buffer_inv sc0 sc1` |
| local/scratch traffic | none | none |

Zero `ld.local` / `scratch_` in either kernel on either target, i.e. the
MOCO-1431 pointer-array trap of §6 was avoided here too (slot addresses are
formed arithmetically inside the unrolled loop).

Reproduce: `sbatch split.sbatch`, then
`python3 split_report.py /home/gabriel/ddp_work/logs/split_<jobid>`.
Raw logs: `/home/gabriel/ddp_work/logs/ccl_split_233937.log` and
`/home/gabriel/ddp_work/logs/split_233937/*.txt` (every rank's CSV).

---

# MI300A (gfx942), 4 ranks — 2026-09-09

One Adastra node, 4 × MI300A (gfx942, 228 CUs, ROCm 6.4.3), CPU torch 2.11,
MAX 26.5. Reference: RCCL 2.22.3 from the same ROCm install, through the same
process group. All device times are the streamed statistic `ar_bench.py`
prints: 20 back-to-back collectives on the comm stream, one synchronize,
wall/20, max over ranks of the median of 5 repetitions — the number comparable
to torch-profiler GPU time. Legs are interleaved RCCL, mojo, mojo, RCCL.

Clocks were **not** locked: `rocm-smi --setperfdeterminism` needs privileges a
job step does not have on this cluster, and the node is shared. The ABBA
ordering is what bounds the drift; run-to-run spread on repeated points was
under 4%.

## 1. The link-direction probe

`perf-work/linkbw.mojo` — one process, four `DeviceContext`s, raw
`hipExtMallocWithFlags` buffers, peer access enabled between every pair, timed
with HIP events on each device's own stream, deterministic hash fill and
host-verified samples. The copy loop is `_copy_vec`'s exact shape (16-byte
vectors, 4 in flight, 256-thread blocks, grid-stride). Per-GPU GB/s:

| mode | 4 MiB | 11733296 B | 27 MiB | 168 MiB |
|---|---|---|---|---|
| local HBM copy | 332 | 1028 | 1674 | 1453 |
| local copy, source = the **uncached** region | 293 | 928 | 1630 | 1434 |
| one link, write (GPU0 → GPU1) | 80 | 90 | 91 | 91 |
| one link, read (GPU0 ← GPU1) | 80 | 87 | 88 | 90 |
| ring of writers (each → successor) | 79 | 87 | 88 | 91 |
| ring of readers (each ← successor) | 46 | 49 | 52 | 56 |
| **all-to-all writes (3 peers)** | 208 | 222 | **238** | **233** |
| **all-to-all reads (3 peers)** | 88 | 83 | **81** | **93** |
| all-to-all writes + a second local store | 200 | 177 | 223 | 218 |

Three facts come out of it:

1. One xGMI link direction does ~91 GB/s either way.
2. **Writes scale across the three links; reads do not.** Three outbound
   streams reach 233–238 GB/s, three inbound reads 81–93, and a ring of
   simultaneous readers *falls* to 52–56. A GPU-initiated remote load is
   limited per GPU, not per link.
3. 233 GB/s is RCCL's number: 512 MiB in 3410 µs at world 4 is 236 GB/s of
   busbw. RCCL's P2P transport hard-wires `read = 0` on AMD
   (`rccl:src/graph/paths.cc:441`).

Reading the uncached region locally costs nothing (1434 vs 1453 GB/s), which
is what makes the redesign below affordable. Region memory type was swept
separately: `hipDeviceMallocUncached` beats plain `hipMalloc` at 4 MiB
(208 vs 178 GB/s of all-to-all write) and ties at 27 and 168 MiB.

## 2. The redesign: nothing crosses a link in the read direction

`_ar_twoshot_kernel` on AMD (NVIDIA is untouched, §5):

| | phase 1 | phase 2 | phase 3 |
|---|---|---|---|
| NVIDIA | push shard *s* into peer *s*'s slot | reduce my shard → my region + user output | **pull** the peers' reduced shards into the user output |
| AMD | same | reduce my shard → user output **and every peer's gather slot** | **local copy** of my own gather slots into the user output |

Cross-link traffic is identical — `2(world-1)/world × bytes` per GPU, the
unicast minimum — and only its direction changes. The user's output is a MAX
allocation and cannot be IPC-mapped, so a peer cannot write it directly; that
is the whole reason for the local copy, and at 0.75 × message and 1434 GB/s it
costs ~90 µs at 168 MiB against the ~1130 µs the wire needs.

The phase-2 stores are fused: one reduce, `world` stores (the user output slice
plus each peer's gather slot), which is NCCL's `MULTIDSTS` shape
(`rccl:src/device/common_kernel.h`, `reduceCopyPacks` stores to every
destination from one accumulator). gfx942 emits it as four
`global_store_dwordx4` from one `v[20:23]`. Doing it as a second pass over the
shard instead measured 1437 µs at 168 MiB against 1332 for the fused form
(128 blocks, both with the same fence).

Arena: the `world` push slots are unchanged and the area that held the single
reduced shard on NVIDIA holds `world-1` compacted gather slots on AMD, at the
same base offset. `world*slot` is already about `numel*elem` ≤ cap, so a full
`world` more slots would not fit a `2*cap` arena at the largest message; the
compaction (writer *w* uses slot `w if w < owner else w-1`, the same rule the
split allreduce already used for its push slots) bounds the total at
`cap*(2 - 1/world)`.

The all-gather and the broadcast's gather half became pushes for the same
reason. The all-gather needs `world-1` message-sized slots rather than one, so
`allgather_max_bytes` chunks at `2*cap/(world-1)` on AMD and stays the
identity on NVIDIA. The broadcast keeps its scatter and adds a push phase and
one more sync: root scatters shard *p* into rank *p*'s region, every rank
writes its shard into every other non-root rank's gather slot, everyone
assembles locally.

`_ag_finish_kernel` — the second half of the multi-node split allreduce — is
the one collective still pulling. Converting it needs an extra sync inside the
kernel (the vendor library rewrites the shard between the two halves, so the
push cannot happen in the first one) and this engagement had no second node to
measure or test it on, so it was left alone deliberately rather than changed
blind.

## 3. The barrier

On gfx942 `Atomic.store[RELEASE]` lowers to `buffer_wbl2 sc0 sc1` + the store
and `Atomic.load[ACQUIRE]` to the load + `buffer_inv sc0 sc1`. Both are
**whole-cache** operations: `wbl2` writes the device's L2 back to memory,
`inv` drops the CU's L1 and the device's L2. That makes the barrier expensive
in proportion to the grid -- the acquire in particular sits inside the spin
loop, so every polling thread invalidates the whole cache on every iteration
and throws the payload out of L2 for every block still working: 27 MiB
measured 243 microseconds at 128 blocks and 465 at 1024 with that spelling.

**Three cheaper spellings were tried in this engagement and all three were
given up; the barrier shipped on 2026-09-09 was the one that was there before
this work.** The two release-side ones stay given up. The acquire-side one,
a relaxed spin with one acquire fence after it, was re-adopted on gfx942 on
2026-09-24 (PR #545), its fence moved out of the polling branch after
review. Section 7 has why, and what the evidence does and
does not show. The spellings are recorded because the pattern was the same
every time and it was the lesson of the engagement: each looked provably
equivalent, and each broke only the *small* collectives, where a payload is
a few hundred bytes instead of megabytes and therefore does not drain out of
a cache on its own.

| spelling | 27 MiB allreduce | broadcast, 4 ranks | 1-element allreduce, 2 ranks |
|---|---|---|---|
| no release writeback at all (RCCL's `skip_fence`) | correct | **5 failures** | — |
| release writeback moved after the barrier, into the `world` publishing threads | correct | correct | **fails** |
| relaxed spin + one acquire fence after the wait (2026-09-09) | correct | correct | **2 failures in 13 runs** |
| release fence in every thread, acquire load per iteration (shipped 2026-09-09; still every non-gfx942 target) | correct | correct | 0 failures in 12 runs |
| gfx942 since PR #545: relaxed spin + `s_sleep(1)`, one acquire fence per wave after the polling branch | correct | correct | 0 failures; no directed repeat (section 7) |

RCCL's `skip_fence` (`rccl:src/include/rccl_common.h:262-273`, on for cudaArch
940 when the buffers are uncached) is sound for RCCL and not for us: our region
*is* `hipDeviceMallocUncached`, but the mapping a peer writes **through** comes
from `hipIpcOpenMemHandle` and does not carry that memory type, so a few bytes
can still be sitting in the writer's cache when the flag lands. The third row
is the one that hurt -- it is worth 75 microseconds at 168 MiB and its
failure evidence is weak (Fisher p about 0.5 against the fourth row) --
section 7 explains why it went on 2026-09-09 and why its fifth-row form came
back.

The one thing that did *not* have to be given up is the AMD `vmcnt(0)` drain
that was already there: `fence[RELEASE]()` in every thread before the block
barrier lowers to `s_waitcnt vmcnt(0)` + `buffer_wbl2 sc0 sc1`, which is what
makes a peer's payload visible before the flag it will be told about. RCCL
puts the same `vmcnt(0)` inside its block barrier
(`rccl:src/device/prims_simple.h:193-210`) for exactly this reason.

## 4. Grid caps

Both are fitted on this card and both are documented next to their
definitions. `_AR_MAX_BLOCKS` (messages < 64 MiB) and `_AR_BIG_BLOCKS` (above)
on AMD, fp32, 4 ranks, µs:

| blocks | 9 MiB | 27 MiB | 168 MiB | 512 MiB |
|---|---|---|---|---|
| 64 | 118 | 285 | 1987* | 6638* |
| **128** | **121** | **254** | 1330* | 4065* |
| 160 | — | — | 1302* | 4804* |
| 224 | 144 | 258 | 2224* | 6053* |
| 456 | — | — | 1509 | 7177 |
| **912** | — | — | **1221** | **3744** |

(`*` measured with the intermediate barrier; the 456/912 rows and the small
sizes are with the shipped one. The response is not monotonic — 224 is the
worst point at the large sizes and 64 starves the 27 MiB transfer — so do not
interpolate, re-sweep.) 912 is four waves of the 228 CUs. The best small-size
value moved from 224 down to 128 when the release fence went back to every
thread, because the grid cost there is the per-thread `buffer_wbl2`.

## 5. NVIDIA is untouched

Every behavioural difference is behind `has_amd_gpu_accelerator()` at compile
time, in the kernel and in the two host lines that size the arena and chunk
the all-gather. The arena layout and its capacity arithmetic are byte for byte
what they were on NVIDIA (the AMD gather slots sit at the same base offset the
single reduced shard used).

Proof: `mojo build mojoccl.mojo --emit asm --target-accelerator sm_90a -I .
-I ../../eager_kernels` in this tree and in the pre-MI300A tree, PTX sidecars
compared with the 8-hex-digit mangling hash masked in both the file name and
the body — **97 kernels before, 97 after, 0 only-before, 0 only-after, 0
differing** (`perf-work/asm90_diff.py`).

## 6. Headline: A/B against RCCL 2.22.3, 4 ranks

Interleaved RCCL, mojo, mojo, RCCL in one job on one node, shipped constants
and shipped barrier; each cell is the mean of that leg pair and both legs are
given so the spread is visible. Device time in microseconds. Both legs run
with `MODULAR_DEVICE_CONTEXT_MEMORY_MANAGER_VMM=1`: without it four ranks
reserve ~124 GB each on this APU, the node runs out of memory and every leg
reads hundreds of ms, which does not look like a measurement error.

| dtype | size | RCCL | mojo | ratio | legs (RCCL / mojo) |
|---|---|---|---|---|---|
| fp32 | 1 MiB | 53.4 | 51.1 | **0.96** | 53,54 / 52,51 |
| fp32 | 9 MiB | 105.4 | 113.3 | 1.08 | 104,106 / 114,113 |
| fp32 | 27 MiB (DDP bucket) | 226.9 | 226.7 | **1.00** | 226,228 / 227,226 |
| fp32 | 168 MiB | 1155.4 | 1335.4 | 1.16 | 1157,1154 / 1335,1336 |
| fp32 | 512 MiB | 3411.3 | 3790.2 | 1.11 | 3414,3408 / 3787,3793 |
| bf16 | 1 MiB | 57.8 | 65.3 | 1.13* | 50,65 / 76,55 |
| bf16 | 9 MiB | 110.0 | 113.8 | **1.04** | 110,110 / 114,114 |
| bf16 | 27 MiB | 233.0 | 226.8 | **0.97** | 233,233 / 228,226 |
| bf16 | 168 MiB | 1164.4 | 1335.2 | 1.15 | 1164,1164 / 1335,1335 |
| bf16 | 512 MiB | 3438.3 | 3774.5 | 1.10 | 3439,3438 / 3775,3774 |

\* 50 and 65 microseconds against 76 and 55, on a 1 MiB message whose kernel is
about 20 microseconds of work: launch noise, not a measurement of the kernel.

Where this started, on the same node: the shipped pull kernels with the peer
rotation and the first MI300A grid caps measured 147 / 345 / 1745 / 5300
microseconds at 9 / 27 / 168 / 512 MiB against RCCL's 108 / 230 / 1160 / 3410
-- ratios of 1.36 to 1.55. The size that matters for DDP is now at parity.

**168 and 512 MiB are the two that still miss.** The arithmetic says why, and
it is not the schedule. Per GPU the collective must move 1.5 x message across
the fabric, which at the 233 GB/s all-to-all write ceiling of section 1 is
1134 microseconds at 168 MiB -- RCCL's own 1155, i.e. RCCL is running at the
ceiling with no measurable overhead. On top of that this design pays the local
gather copy (0.75 x message read and written locally, ~90 microseconds
measured) and the barrier's whole-cache operations at the 912-block grid the
large sizes want (measured: the same allreduce is 1221 microseconds at 912
blocks with the relaxed-spin barrier and 1335 with the per-iteration acquire
load shipped at the time; section 7 says why the relaxed spin was given up
then and re-adopted since, and these 2026-09-09 tables were not re-measured
with it). Removing either needs a change this
engagement did not make: overlapping the gather copy with the second push
behind a sub-chunk pipeline, or a barrier that is both demonstrably sound here
and cheaper than one whole-cache operation per thread per sync.

## 6b. All-gather and broadcast

Same ABBA protocol, fp32, 4 ranks, per-rank contribution for the all-gather
and message size for the broadcast. Busbw is `(world-1)*n/t` and `n/t`
respectively, the same convention `ar_bench.py` prints.

| op | size | RCCL µs | mojo µs | ratio | mojo busbw | before this work |
|---|---|---|---|---|---|---|
| all_gather | 4 MiB | 101.8 | 134.7 | 1.32 | 93 GB/s | |
| all_gather | 16 MiB | 274.7 | 319.8 | 1.16 | 157 GB/s | |
| all_gather | 64 MiB | 947.8 | 1146.1 | 1.21 | **176 GB/s** | 80 GB/s |
| broadcast | 4 MiB | 53.2 | 89.8 | 1.69 | 47 GB/s | |
| broadcast | 16 MiB | 121.1 | 217.9 | 1.80 | 77 GB/s | |
| broadcast | 64 MiB | 342.9 | 680.0 | 1.98 | **99 GB/s** | 78 GB/s |

The all-gather is now within ~20% and the remaining gap is the same one the
allreduce has at the large sizes. The broadcast is not, and the reason is its
schedule rather than its direction: scatter-then-push-all-gather puts
`n + (world-1)/world*n` = 1.75 × message of outbound traffic on the root,
where a ring or a tree puts `n`. Converting the gather half from a pull to a
push took it from 78 to 99 GB/s; getting to RCCL's 195 needs the root to stop
being the only sender of the first phase, which is a different algorithm and
was out of scope here.

## 7. Correctness, and one unresolved flake

`tests/ddp_worker.py` through the process group, `TORCH_MOJO_BACKEND_CCL=mojo`,
`perf-work/suite.sh`, on the shipped tree:

| mode | 4 ranks | 2 ranks |
|---|---|---|
| `collectives` | 20 OK / 0 FAIL | 11 OK / **4 FAIL** (see below) |
| `ddp_parity` | 5 / 0 | 3 / 0 |
| `lazy_fence` | 11 / 0 | 6 / 0 |
| `stress` (`MOJOCCL_REGION_MB=4`, forces chunking) | 282 / 0 | 161 / 0 |
| `abort` | 16 / 0 | 10 / 0 |

`stress` with the default 256 MiB region also passes (300 OK / 0 FAIL). It
exits nonzero for reasons that have nothing to do with these kernels: without
`MODULAR_DEVICE_CONTEXT_MEMORY_MANAGER_VMM` four ranks reserving ~124 GB each
are OOM-killed on this APU, and with it the process segfaults in HIP's atexit
handler unless the script ends in `os._exit(0)` — both already in
`agents_docs/distributed.md`.

**A flake that cost the large messages 6%.** During the work the barrier's
acquire was cheapened: spin on a *relaxed* load (still `sc0 sc1`, so it cannot
read a stale flag) and issue one acquire fence after the wait, instead of an
acquire load — and therefore a whole-L1-and-L2 `buffer_inv` — on every
iteration. It looks exactly as strong and it is worth a lot: 243 µs against
465 at 27 MiB with a 1024-block grid, because the per-iteration invalidate
throws the payload out of L2 for every block still working.

With it, `allreduce.int64` — a *one-element* int64 allreduce, the smallest
collective in the suite: one-shot path, one block, one 8-byte store and one
8-byte load by a single thread — failed intermittently **at 2 ranks**:
2 failures in 13 runs of `ddp_worker.py collectives`. Every 4-rank run of
every mode passed. With the acquire load restored: **0 failures in 12 runs**
(`perf-work/flake.sh`, same node, same session).

Neither sample proves anything on its own — Fisher's exact on 2/13 against
0/12 is p ≈ 0.5. It was reverted anyway on 2026-09-09, and the reasoning is
worth recording because it was the lesson of the day rather than a number:

* the one-shot kernel is **unchanged** by this work, so the only thing that
  could have introduced a failure under it is the barrier;
* the argument for the cheap acquire was a symmetry ("the invalidate after the
  loop is at the same point as the invalidate in the last iteration"), not a
  guarantee — and that exact style of reasoning produced **two** unsound
  release spellings earlier the same day, both of which broke only the small
  collectives, which is precisely what this is;
* a rare silent wrong answer inside an allreduce costs incomparably more than
  75 µs on a 168 MiB message.

**What was not established**: whether the pre-MI300A tree flakes here too. The
A/B was attempted (`perf-work/flake.sh`, 8 runs per tree) and the before-tree
half is void — those runs die in `ncclCommInitRank` with `ncclResult_t=3` in
that harness, a harness problem rather than a result. So it is still unknown
whether this is a pre-existing race that the cheap acquire merely made more
likely. Settling it needs a few hundred runs of the 2-rank `collectives` mode
on both trees with the before-tree harness fixed; the shipped tree at
0/12 is not evidence of absence. A later A/B with a working harness ran the
pre-MI300A tree against the acquire-load tree: 0/20 on each. It shows the
redesign did not bring the flake in; it says nothing about the relaxed spin,
which neither tree had.

**Update, 2026-09-24: the relaxed spin is back on gfx942, fence moved.**
PR #545 re-adopted it for gfx942 only (every other target keeps the
acquire load), with RCCL 2.22.3's `s_sleep(1)` between failed polls. The
reason was a workload, not this benchmark: on 2 × 4 MI300A (GPT-2 XL FSDP2)
the per-iteration invalidate evicts the L2 of the compute kernels running
beside the collective, and the relaxed spin took the profiled communication
busy time from 209.5 to 188.5 ms per step and the compute kernel sum from
249 to 235 ms. End to end the gain was about 1%, inside the leg noise.
`agents_docs/distributed.md` ("gfx942 waits acquire payloads once") has the
ordering argument: a relaxed load that reads the release store, followed by
an acquire fence, is the atomic-to-fence rule of the C++ and LLVM memory
models. That is a proof, not the symmetry argument of 2026-09-09.

A later code review then found why the symmetry argument was worse than
unproven. In its first spelling the fence sat at the end of the `thread_idx.x < world`
branch, right after the divergent spin loop. The loop leaves EXEC holding
the lanes still waiting, which is 0 once every flag is seen, and LLVM's
SILowerControlFlow, which treats a fence as not reading EXEC, dropped the
loop's EXEC restore in front of it. So in the gfx942 assembly
`buffer_inv sc0 sc1` ran with EXEC = 0 on every successful exit, whether
the first poll succeeded or a later one did. It was an acquire only if the
hardware ignores EXEC for a cache invalidate, and nothing verified that for
CDNA3. If the 2026-09-09 spelling had its fence in the same place (its
assembly was not kept), a success path with no acquire at all is a
plausible cause of the 2/13, and would also explain why only one-element
payloads failed. That is unverified. The review fix moved the fence after
the branch closes. Every wave now issues it under its full mask, one
invalidate per wave per barrier and still none in the spin. The gfx942
assembly of the CCL entry shows all 309 barrier acquires after the EXEC
restore and directly before `s_barrier`, against 0 of 305 in the first
spelling's dump.

What the correctness evidence shows, then:

* **For the old spelling:** 2 failures in 13 runs of the 2-rank
  one-element allreduce.
* **For the first spelling's in-branch fence** (EXEC = 0): no failure in any
  validation round of the 2 × 4 and 1 × 4 suites, one per commit. Since
  the same PR the DDP `stress` mode runs one-element int64 allreduces in
  every generation, with changing data and an exact integer reference.
  None of this was a directed repeat of the 2-rank loop, so it bounds the
  rate only loosely. It does not show the in-branch fence was sound.
* **For the fence after the branch:** the soundness argument no longer
  rests on the hardware ignoring EXEC. Its success path is the old
  acquire load's last iteration, `global_load sc0 sc1; s_waitcnt;
  buffer_inv sc0 sc1`, executed by a wave with a non-empty mask. The
  2 × 4 and 1 × 4 suites passed with it, 20 of 20, one-element int64
  generations included. That was one round, not a directed repeat.

The combined samples settle nothing statistically. What changed is that
the barrier no longer needs the untested assumption.

