# GPT-2 MI300X optimization journal

Frozen workload: GPT-2, batch 512, prompt length 8, 200 new tokens, BF16,
greedy sampling, Hugging Face `DynamicCache`, eager execution, no graph capture
or `torch.compile`. A 200-token `generate()` consists of one prefill iteration
and 199 cached-decode iterations.

Acceptance rules used throughout:

- Correctness is measured against a PyTorch-ROCm FP32 result on identical
  inputs. A Mojo BF16 result passes only when its maximum absolute error is no
  more than twice the PyTorch-ROCm BF16 error against that FP32 result.
- Microbenchmarks use at least 25 warmups and 100 individually synchronized
  timed iterations. Median and p10/p90 are reported. ROCm and Mojo run in the
  same process/session; their ratio is the decision metric.
- The full frozen benchmark and top-shape regression checks gate every
  accepted kernel/dispatch change.
- PyTorch tensor dimensions remain runtime values. No model shape is compiled
  into the PyTorch-facing dispatch.

## Baseline verification — 2026-07-17

Environment: AMD Instinct MI300X VF (`gfx942`), PyTorch 2.11.0+rocm7.2,
MAX 26.5.0.dev2026061806, Mojo 1.0.0b3.dev2026061806.

| Decode group | Mojo GPU ms | ROCm GPU ms | Potential saving |
|---|---:|---:|---:|
| Transformer projection/MLP GEMMs | 3690.030 | 161.905 | 3528.125 |
| LM-head GEMM | 226.834 | 23.922 | 202.912 |
| Attention over cache | 411.805 | 308.977 | 102.828 |
| Elementwise GELU/residual | 222.204 | 150.345 | 71.859 |
| LayerNorm | 41.909 | 29.083 | 12.826 |
| Logits processing | 27.485 | 21.875 | 5.610 |
| Concat | 248.209 | 647.769 | Mojo advantage: 399.560 |

Baseline end-to-end: ROCm 1.489637 s (68,741.6 tok/s), Mojo 5.095026 s
(20,098.0 tok/s). Baseline decode self GPU time: ROCm 1418.300 ms, Mojo
4950.334 ms. Mojo trace idle is 2.57%; dispatch/launch optimization is out of
scope unless a later measurement disproves this baseline.

## Change 0 — Phase 0 measurement harness (infrastructure only)

**Hypothesis.** Exact-shape, exact-layout synchronized microbenchmarks will
reproduce the profiler's ordering: transformer `addmm` is the dominant gap,
with the `[512,3072] x [3072,768]` projection worst in absolute time, while
the LM head is the next GEMM target.

**Predicted effect.** No runtime change. The initial target table should show
Mojo/ROCm ratios far above the 1.15 acceptance threshold for every recorded
decode GEMM and should retain the same relative priority as the full profile.

**Measured effect.** The initial `bench_gemm.py` run used 25 warmups and 100
timed iterations per backend and produced `gemm_target_table.csv`. All ten
cases passed the correctness gate. No case passed the 1.15x performance gate:

| Phase / operation | M | N | K | Mojo median us | ROCm median us | Ratio |
|---|---:|---:|---:|---:|---:|---:|
| Decode addmm | 512 | 768 | 768 | 187.376 | 33.587 | 5.579x |
| Decode addmm | 512 | 768 | 3072 | 790.796 | 41.974 | 18.840x |
| Decode addmm | 512 | 2304 | 768 | 194.110 | 36.633 | 5.299x |
| Decode addmm | 512 | 3072 | 768 | 232.882 | 39.219 | 5.938x |
| Decode linear (LM head) | 512 | 50257 | 768 | 1162.959 | 136.255 | 8.535x |
| Prefill addmm | 4096 | 768 | 768 | 421.721 | 41.444 | 10.176x |
| Prefill addmm | 4096 | 768 | 3072 | 1662.814 | 71.651 | 23.207x |
| Prefill addmm | 4096 | 2304 | 768 | 486.248 | 57.986 | 8.386x |
| Prefill addmm | 4096 | 3072 | 768 | 648.770 | 70.246 | 9.236x |
| Prefill linear (LM head) | 512 | 50257 | 768 | 1164.510 | 135.911 | 8.568x |

The worst decode projection achieved 3.055 TFLOPS on Mojo versus 57.558
TFLOPS on ROCm. The LM head achieved 33.985 versus 290.073 TFLOPS. The
measured priority therefore matches the full profile. Temperatures remained
37 C or lower; the before/after SMI snapshots show no thermal throttling.

**Decision.** Accept the measurement harness and target table as Phase 0
infrastructure. It changes no runtime code. Proceed to the no-change Phase 1
diagnosis.

## Phase 1 diagnosis — 2026-07-17

**Hypothesis.** BF16 GEMMs are routed to a portable VALU tiled kernel instead
of MAX's AMD MFMA path; the worst shape also lacks enough workgroups, and
`addmm` applies bias separately.

**Predicted effect.** Diagnostic work changes no runtime. Evidence should show
near-zero practical MFMA throughput even on a grid-filled ceiling shape, and
the trace should show a portable GEMM kernel plus a bias kernel.

**Measured effect.** `gemm_diagnosis.md` records D1-D5. The worst addmm
launches `pure_gemm_tiled_bfloat16_64x64x16_tbFalse` (917.728 us in the
profile) plus a 5.331 us bias kernel; native ROCm uses one 26.258 us fused
MFMA kernel. The 8192^3 ceiling reaches only 41.322 TFLOPS on Mojo versus
639.339 TFLOPS on ROCm (15.472x slower). Source inspection proves BF16 is
excluded from the dynamic AMD MFMA and split-K branches. A direct stock MAX
graph BF16 matmul is not a usable alternate route on this pin: it aborts with
`Cannot select: llvm.amdgcn.fdot2.f32.bf16`.

Hardware counter collection was attempted with both launch and attach modes.
The profiler cannot share this MAX runtime's HSA initialization, so no counter
records were emitted. The kernel source, dispatcher gate, trace kernel name,
and grid-filled ceiling measurement all independently identify the VALU path.

**Decision.** Accept the diagnosis. Implement the cheapest fix first:
generalize the existing runtime-dimension gfx942 MFMA wrapper to BF16 and
route BF16 matmul/addmm/linear through it, folding bias into its epilogue. Do
not work on launch overhead, attention, or any lower-priority group.

## Change 1 — Route BF16 GEMM to the existing dynamic AMD MFMA core

**Hypothesis.** The dominant gap comes from the dtype gate that sends BF16 to
the portable VALU GEMM. Making the existing gfx942 multistage MFMA wrapper
dtype-generic and selecting it for BF16 will remove most of the 5.3x-23.2x
GEMM gap without compile-time PyTorch dimensions. This change deliberately
retains the separate BF16 bias-add kernel so bias fusion can be measured as a
separate hypothesis.

**Predicted effect.** Exact-shape Mojo medians should improve by at least 5x
on the worst decode projection and the 8192^3 ceiling, with all correctness
checks still passing. The transformer projection/MLP group should fall from
3690 ms to below 300 ms. `addmm` should still show two kernels until the next
change.

**Measured effect.** All ten exact-shape correctness checks passed. Decode
medians changed as follows:

| Shape / operation | Before us | After us | Speedup | After / ROCm |
|---|---:|---:|---:|---:|
| 512x768x768 addmm | 187.376 | 40.669 | 4.607x | 1.244x |
| 512x768x3072 addmm | 790.796 | 110.078 | 7.184x | 2.614x |
| 512x2304x768 addmm | 194.110 | 53.416 | 3.634x | 1.522x |
| 512x3072x768 addmm | 232.882 | 66.076 | 3.524x | 1.750x |
| 512x50257x768 linear | 1162.959 | 602.341 | 1.931x | 4.401x |

The same-session full gate reproduced ROCm at 1.492 s wall and 1417.999 ms
decode self GPU time. Mojo improved from the original 5.095 s to 1.871 s
wall and from 4950.334 to 1673.401 ms decode self GPU time. The transformer
projection/MLP group fell from 3690.030 to 533.059 ms (6.922x faster), while
the LM head fell from 226.834 to 117.801 ms. Prefill self GPU time fell from
51.350 to 23.319 ms. The prediction of projection/MLP below 300 ms was not
met, so shape selection still requires work.

Every original top-10 Mojo decode shape passed the regression gate. The only
increases were `mul` at 1.007x and unchanged `cat` at 1.000x; all other rows
improved. End-to-end wall time improved by 63.3%. Temperatures remained 36 C
or lower, and post-run clocks returned to idle without throttling evidence.

**Decision.** Accept as a major intermediate routing fix. It does not meet
the per-shape 1.15x or phase exit criteria, so continue within GEMM work. The
next separate change will test fused BF16 bias; shape-aware MFMA config tuning
remains necessary afterward.

## Change 2 — Fuse the BF16 addmm bias epilogue

**Hypothesis.** After Change 1, every BF16 `addmm` still launches the MFMA
kernel followed by a row-broadcast elementwise bias kernel. Routing the BF16
bias case through the MFMA wrapper's existing epilogue will remove one launch
and one complete output read/write without changing the GEMM core or its
shape selection.

**Predicted effect.** Exact-shape addmm medians should fall by roughly 5-10 us
per call, while LM-head `linear` remains unchanged. The full projection/MLP
group should fall from 533 ms to 450-480 ms. A profiled addmm must launch
exactly one MFMA kernel, all ten correctness checks must pass, and no top-10
row or end-to-end wall time may regress.

**Measured effect.** Kineto shows exactly one kernel for the worst addmm:
`multistage_gemm_kernel_bfloat16_bfloat16_bfloat16_False`, 92.650 us in the
single-op profiled trace, grid 24x16, block 256. The separate elementwise bias
kernel is gone. All ten exact-shape correctness checks passed. Decode addmm
medians improved by 2.639, 3.479, 5.024, and 8.015 us for the four shapes;
the unbiased LM head was unchanged within 0.1%. Prefill addmm medians improved
by 13.4-41.4 us because the eliminated output pass is larger.

The same-session full gate measured ROCm at 1.490 s wall / 1419.262 ms decode
and Mojo at 1.822 s wall / 1624.104 ms decode. Relative to Change 1, Mojo wall
improved 2.62%, projection/MLP GPU time fell from 533.059 to 483.287 ms, and
prefill GPU time fell from 23.319 to 22.126 ms. The group result is 3.0x ROCm,
so configuration work is still required. All prior top-10 rows passed: the
largest non-target increase was `cat` at 1.004x. Temperature stayed at or
below 36 C with no throttling evidence.

**Decision.** Accept. The one-kernel requirement and every correctness and
regression gate pass. Proceed to shape-aware MFMA configuration tuning; do
not move to lower-priority functional groups.

## Change 3 — BF16 shape-regime MFMA tile selection

**Hypothesis.** Change 2 still uses the FP32-oriented configurations with a
32-deep K tile for BF16. MAX's AMD BF16 MMA has K=16 and its stock schedule
uses BK=64; the current small 32x32/32x64 blocks sacrifice MFMA reuse on
grid-filled prefill and wide-N shapes. A runtime-shape heuristic using BK=64,
larger output tiles when the grid is sufficient, and four in-workgroup K
partitions for the K-dominant decode shape will improve core efficiency. All
M/N/K decisions remain runtime branches and no model dimension becomes a
comptime parameter.

**Predicted effect.** Prefill GEMMs should improve to <=1.5x ROCm, ordinary
decode projections to <=1.3x, and the K-dominant decode projection from 106.6
us to <=60 us. The LM head should improve by at least 2x. The selected regimes
are: 128x128/BK64 for M>=1024 or N>=8192, 64x64/BK64 for other decode GEMMs,
and 32x32/BK128 with four warp-K partitions when K>=2N.

**Measured effect.** Rejected at the first launch gate. The extension compiled,
but the K-dominant `MatmulBiasSpec` raised `hipErrorInvalidValue` at the
`ctx.enqueue_function` call for the 32x32/BK128/four-partition kernel. No
output was produced, so correctness and performance are unavailable.

**Decision.** Reject and roll back the entire heuristic without a full
profile. The unsupported four-partition schedule is not retained. Test the
BK64/larger-output-tile hypothesis separately without warp-K partitioning.

## Change 4 — Valid BK64 two-regime BF16 tile heuristic

**Hypothesis.** The reusable, stock-style BK64 depth and larger output tiles
can improve BF16 MFMA reuse without the invalid warp-K partition feature. A
64x64/BK64 tile should retain more decode grid fill than 128x128, while
128x128/BK64 should improve grid-filled prefill and wide-N work.

**Predicted effect.** Use 128x128/BK64 when M>=1024 or N>=8192 and
64x64/BK64 otherwise. Ordinary decode projections should be <=1.3x ROCm, the
K-dominant decode projection <=80 us, prefill shapes <=1.8x ROCm, and the LM
head should improve by at least 2x. Every launch and correctness gate must
pass before a full profile.

**Measured effect.** All correctness checks passed, but every target shape
regressed. The K-dominant decode shape rose from 106.599 to 150.997 us, the
other decode addmm shapes rose to 51.825-74.522 us, prefill rose to
154.212-480.747 us, and the LM head rose from 602.606 to 987.959 us.

**Decision.** Reject without a full profile because the target microbenchmarks
already violate the no-regression gate. Restore Change 2's BF16 configs. The
next hypothesis will isolate K partitioning to the K-dominant regime and use
two partitions with a valid BK64 schedule.

## Change 5 — Two-way warp-K partition for K-dominant BF16 GEMM only

**Hypothesis.** The `[M,3072] x [3072,768]` shape needs more K-parallel work,
but four partitions/BK128 is not launchable and large output tiles regress.
A 32x32 output tile with BK64 and two in-workgroup K partitions satisfies the
BF16 MMA grouping constraint while preserving the 384-workgroup decode grid.

**Predicted effect.** Only the runtime regime `K>=2048 and K>=2N` changes.
The decode shape should fall from 106.599 to <=80 us and its M=4096 prefill
counterpart from 415.160 to <=300 us. Every other target shape should remain
within measurement noise. Correctness and repeated-launch stability must pass.

**Measured effect.** All ten exact-shape correctness checks passed. The target
decode shape improved from 106.599 to 88.031 us (17.4%) and its M=4096
counterpart improved from 415.160 to 401.223 us (3.4%). The other four decode
medians were 38.124, 47.409, 58.123, and 601.741 us, all within 1.3% of Change
2. The prediction for decode was directionally correct but did not reach the
80 us target; the prefill prediction was not met.

The same-session full gate measured ROCm at 1.492 s wall / 1417.094 ms decode
and Mojo at 1.799 s wall / 1571.539 ms decode. Relative to Change 2, Mojo wall
improved 1.26%, projection/MLP GPU time fell from 483.287 to 430.159 ms, and
prefill GPU time improved from 22.126 to 21.961 ms. The K-dominant grouped
shape measured 174.761 ms across 2388 calls (73.18 us/call) in Kineto. Every
prior top-10 shape passed the regression gate; the largest non-target increase
was `mul` at 1.022x and `cat` was unchanged at 1.000x. Temperatures remained
at or below 36 C and the before/after clock readings showed no throttling.

**Decision.** Accept. Correctness, repeated launches, end-to-end wall time, and
the full top-10 regression gate all pass. The change is a runtime K/N regime,
not a model-shape specialization. Continue GEMM configuration work because the
2.10x micro ratio and 430.159 ms functional group remain above the Phase 2
targets.

## Diagnostic experiment A — Runtime MFMA configuration sweep

**Hypothesis.** The remaining addmm gap is primarily configuration selection,
and a single benchmark-only entry point that instantiates a small set of
reusable block/warp regimes will identify a faster schedule without repeatedly
changing production dispatch. In particular, wider N tiles should improve B
reuse for the wide-N projections, while a two-partition 32x64 or 64x32 tile may
improve the K-dominant shape beyond Change 5's 32x32 tile.

**Predicted effect.** At least one non-K-dominant candidate should beat the
current 32x64 schedule by 15% on N>=2304 without regressing N=768, and at least
one K-partition candidate should beat 88 us on the K-dominant decode shape.
The sweep interface is diagnostic only and does not alter normal eager routing.

**Measured effect.** The decode prediction was rejected. The current 32x64
schedule remained fastest on all three ordinary M=512 projections (34.0-54.2
us through the direct sweep entry), and Change 5's 32x32/two-partition schedule
remained fastest for the K-dominant shape at 84.2 us. Wider and larger tiles
regressed those decode cases by up to 60%.

The sweep did find a distinct M=4096 regime. A 32x128/two-MN-warp schedule
improved N=768 from 88.8 to 66.3 us and the K-dominant shape from 394.8 to
183.3 us; a 64x128 schedule improved N=2304 from 199.5 to 131.1 us and N=3072
from 255.9 to 186.5 us. All four winning schedules passed the canonical
`bench_gemm.py` input gate. An additional random seed made every schedule,
including the unchanged baseline, exceed the 2x bound on the M=4096/K=3072
case (2.079 vs a 1.877 limit); this was recorded but does not distinguish any
candidate. Temperatures were 36-38 C with stable active clocks.

**Decision.** Retain the sweep as benchmark-only infrastructure. Reject its
decode selection hypothesis; accept its evidence for a separate M>=1024
runtime-regime hypothesis. No production routing changed in this experiment.

## Change 6 — Larger MFMA tiles for runtime large-M addmm

**Hypothesis.** M>=1024 supplies enough independent output work that larger N
tiles improve reuse without starving MI300X. Select 32x128/two-MN-warp for
narrow N and 64x128/four-MN-warp for N>=1536, only for BF16 non-transposed
operands. M/N/K remain runtime values; the schedule applies to a broad large-M
regime and leaves all decode and LM-head paths unchanged.

**Predicted effect.** Canonical prefill medians should be approximately 70,
135, 190, and 190 us for the four addmm geometries, versus Change 5's 92.7,
203.1, 259.6, and 401.2 us. Decode shapes must remain within noise. Full
prefill self GPU time should fall by at least 20%, with no end-to-end or top-10
decode regression.

**Measured effect.** All five canonical prefill correctness checks passed. The
four addmm medians improved from 92.663, 401.223, 203.136, and 259.567 us to
71.593, 187.863, 135.368, and 190.600 us, respectively. The LM-head median was
unchanged at 601.681 us as required.

The same-session full gate measured ROCm at 1.490 s wall / 1417.793 ms decode
and Mojo at 1.786 s wall / 1570.680 ms decode. Relative to Change 5, Mojo wall
improved 0.72%, prefill self GPU time fell from 21.961 to 17.552 ms (20.1%),
and decode projection/MLP time was unchanged within noise at 429.637 ms. Every
prior top-10 decode row passed: the largest increase was LM head at 1.0054x.
The selected addmm/mm/linear test set passed (3 tests). Temperatures remained
35-36 C with no throttling evidence.

**Decision.** Accept. Correctness, full-profile, top-10, and wall-time gates all
pass. Continue projection/MLP work because the decode group remains 2.66x ROCm
and above the 200 ms phase exit target.

## Diagnostic experiment B — Runtime grid split-K for K-dominant decode

**Hypothesis.** The current two-way warp-K kernel reduces partials inside each
workgroup and still takes about 84.5 us through the direct entry. Splitting K
across the grid can expose 2-4x as many independent workgroups, remove the
in-workgroup reduction from the GEMM core, and use an fp32 workspace plus a
separate reduction/bias epilogue. The workspace is allocated normally through
MAX for exactly `splits*M*N` elements; all dimensions and partition geometry
remain runtime values.

**Predicted effect.** A two- or four-way split should reduce the K-dominant
M=512 median from 84.5 us to <=65 us after including reduction, while retaining
the strict fp32-reference correctness gate. Because the existing grid already
has 384 workgroups, this experiment is rejected if workspace/reduction traffic
outweighs the additional parallelism.

**Measured effect.** The unchanged Change 5 warp-K reference passed at 78.657
us on the canonical input. The first grid-split candidate then raised
`hipErrorIllegalAddress` at synchronization before producing a timed sample.
A device-synchronous diagnostic build was terminated after more than 11
minutes of CPU compilation without reaching a launch; no additional GPU state
or measurement was produced.

**Decision.** Reject and remove the experiment in full. The launch/correctness
gate fails, so no full profile is warranted. Normal eager routing was never
connected to this code and remains exactly at accepted Change 6.

## Diagnostic experiment C — One-stage LDS pipeline

**Hypothesis.** The dynamic wrapper currently uses two pipeline stages, while
MAX's current AMD block-shape builder defaults to one. For the K-dominant
32x32/BK64/two-warp-K configuration, two stages replicate A/B LDS storage for
each K partition and can limit resident workgroups. One stage may trade some
load overlap for materially higher occupancy; the same test on the ordinary
32x64 tile establishes whether this is specific to the K-dominant regime.

**Predicted effect.** The one-stage K-dominant schedule should beat 78-85 us by
at least 15%; the ordinary schedule should remain within 5% or improve. Both
must pass the exact canonical correctness gate. The new stage parameter is
comptime schedule metadata only; M/N/K remain runtime dynamic.

**Measured effect.** One stage was 3-8% faster, but every one-stage result
failed correctness. The ordinary schedule produced max errors of 39.95-47.67;
the warp-K schedule produced NaNs. The two-stage reference remained correct on
all four shapes. Temperature stayed at 35 C.

**Decision.** Reject and remove. The pipeline implementation requires its
two-stage prologue for correctness, so the small timing improvement is invalid.
Production dispatch remains unchanged at Change 6.

## Diagnostic experiment D — K-tile depth versus warp-K count

**Hypothesis.** Retain the required two-stage pipeline but change only the
K-dominant tile's K decomposition. BK32 with two warp-K partitions halves LDS
relative to Change 5; BK64 with four warp-K partitions exposes twice the K
parallelism and uses one full-LDS workgroup per CU. The existing 384-output-tile
grid is enough to place at least one workgroup on all 304 CUs.

**Predicted effect.** At least one configuration should improve the canonical
84 us direct median by >=15% while passing correctness and repeated launches.
If neither does, in-workgroup K partitioning is exhausted for this kernel.

**Measured effect.** BK32/two-warp-K improved the canonical direct median from
84.374 to 68.310 us (19.0%) and passed correctness with the identical 1.253998
max error. BK64/four-warp-K regressed to 105.650 us. Temperature stayed at 36 C
and active clocks showed no throttling.

**Decision.** Accept the BK32 evidence for a production change; reject the
four-warp alternative. Normal routing is still unchanged by this diagnostic.

## Change 7 — Halve BK in the K-dominant two-warp schedule

**Hypothesis.** Change only the existing runtime `K>=2048 and K>=2N` BF16
schedule from BK64 to BK32 while retaining two warp-K partitions, two pipeline
stages, the 32x32 output tile, and fused bias. This halves LDS per workgroup and
the direct sweep measured 19% lower latency without changing numerical output.

**Predicted effect.** The canonical aten micro median should fall from 88.0 to
about 72 us. Its full-profile shape should fall from 174.8 to about 142 ms,
bringing the projection/MLP group from 429.6 to roughly 397 ms. All other GEMM
shapes should remain within noise; correctness and top-10 gates must pass.

**Measured effect.** All ten canonical correctness checks passed. The target
aten median fell from 88.031 to 72.626 us (17.5%); every non-target median was
stable. In the first full gate, its profiled shape fell from 174.821 to 146.888
ms and projection/MLP fell from 429.637 to 400.742 ms. Decode self GPU time
fell from 1570.680 to 1541.058 ms. No prior top-10 row regressed by 5%.

The first wall sample was inconclusive: ROCm shifted 1.490->1.497 s and Mojo
1.786->1.798 s, moving the normalized ratio from 1.1990 to 1.2009. A mandatory
same-session repeat measured ROCm 1.495 s and Mojo 1.775 s, ratio 1.1874, so the
wall regression did not reproduce and the ratio improved 0.97% versus Change
6. The repeat reproduced projection/MLP at 400.988 ms and every top-10 gate;
the largest non-target ratio was `mul` at 1.0042x. The selected addmm/mm/linear
tests passed (3 tests). Temperatures remained <=36 C.

**Decision.** Accept. Correctness, repeat wall-time, GPU-time, and top-10 gates
pass. Continue projection/MLP work because the group remains 2.47x ROCm and
above the 200 ms exit target.

## Diagnostic experiment E — Group adjacent K MMAs

**Hypothesis.** `multistage_mma` supports grouping adjacent K MMAs in one
register-prefetch cycle, but the dynamic wrapper uses the default group size
one. BK64 has four BF16 K MMAs and can legally use group size two. The newer
structured AMD kernel derives the same two-MMA grouping from SIMD/fragment
width, so applying it to the existing schedules may reduce load/dispatch
overhead without changing tiles, shapes, or memory semantics.

**Predicted effect.** BK64/group-two should improve its matching group-one
schedule by at least 10% on both the K-dominant and ordinary geometries and
must pass canonical correctness. It must also beat Change 7's BK32/group-one
68.3 us K-dominant result before it can motivate a routing change.

**Measured effect.** Group-two improved the matching BK64/group-one
K-dominant schedule from 84.273 to 79.130 us (6.1%), below the 10% prediction,
but remained 15.8% slower than Change 7's BK32 result. On ordinary tiles it
regressed 4-28%. All outputs passed correctness; temperature stayed at 35 C.

**Decision.** Reject K grouping and retain group size one in production. The
sweep independently showed the already-validated Change 7 tile is faster on
two ordinary narrow-N geometries; treat that as a separate hypothesis.

## Change 8 — Reuse the BK32/two-warp tile for narrow-N decode

**Hypothesis.** For runtime BF16, non-transposed, M<1024 GEMMs with 512<=N<=2304
and K divisible by 64, the 32x32/BK32/two-warp-K schedule has more output-grid
parallelism and was 8-9% faster than the 32x64 schedule. Keep N=3072 on 32x64,
where the candidate regressed, and keep the accepted M>=1024 branch unchanged.

**Predicted effect.** The canonical N=768 and N=2304 aten medians should improve
by about 8%, saving roughly 12-13 ms from the full projection/MLP group. N=3072,
K-dominant, prefill, and LM head must remain within noise. The square decode
shape should reach <=1.15x ROCm.

**Measured effect.** All ten canonical checks passed. The N=768 median fell
from 38.326 to 35.618 us (1.073x ROCm, meeting the per-shape target) and N=2304
fell from 47.731 to 43.949 us. Excluded shapes remained stable.

The same-session full gate measured ROCm at 1.489 s wall / 1418.868 ms decode
and Mojo at 1.762 s wall / 1513.557 ms decode. Relative to Change 7's repeat,
the normalized wall ratio improved from 1.1874 to 1.1833 and projection/MLP
fell from 400.988 to 374.018 ms. The two intended profiled rows improved from
68.529 to 51.467 ms and 81.436 to 71.433 ms. No non-target top-10 row exceeded
the 5% regression threshold. Temperatures stayed <=36 C.

**Decision.** Accept. Correctness, wall-time, GPU-time, and regression gates
pass. Continue within projection/MLP because the group remains 2.31x ROCm.

## Diagnostic experiment F — Output-tile aspect ratio

**Hypothesis.** Basic tile size and K-decomposition are nearly exhausted, but
the two remaining expensive shapes have different reuse needs. A 16-row tile
can expose more independent output work for the K-dominant geometry without
an in-workgroup reduction; a 64- or 128-row by 32-column tile can increase B
reuse while maintaining grid fill for N=3072. All are generic runtime-shape
schedules with BK32 and the required two pipeline stages.

**Predicted effect.** At least one aspect-ratio candidate should beat the
accepted schedule by >=10% on its intended geometry and pass correctness. If
none does, static block/warp configuration selection is considered exhausted
for the current dynamic multistage core.

**Measured effect.** Every candidate passed correctness. All three smaller-M
K-dominant alternatives and both larger-M variants regressed; Change 8's
32x32/BK32/two-warp schedule remains best at 68.064 us. For N=3072, 64x32
improved 54.047 to 49.694 us (8.1%), below the 10% prediction but the only
validated win. Temperature stayed at 35 C.

**Decision.** The >=10% prediction was not met and basic configuration search
is now considered exhausted. Retain the 8.1% N=3072 evidence for one final
production selection change; K-dominant work now requires a new kernel core.

## Change 9 — 64x32 output tile for wide-N decode

**Hypothesis.** For runtime BF16, non-transposed, M<1024, 2304<N<8192 GEMMs,
the 64x32/BK32 tile improves weight reuse while its narrower N dimension keeps
enough grid work. It was 8.1% faster on the exact remaining wide-N projection.
All accepted large-M, narrow-N, K-dominant, and transposed paths take earlier
or disjoint branches.

**Predicted effect.** N=3072 should improve from 58.34 to about 54 us through
aten, saving 8-9 ms from the full projection/MLP group. Every other target row
must remain within noise, and all correctness, wall, and top-10 gates apply.

**Measured effect.** The canonical ten-shape table completed with every
correctness gate passing. The intended N=3072 decode median fell from 58.34
to 54.38 us (6.8%) and is 1.380x ROCm. All disjoint routes stayed within
normal run-to-run noise: square decode 35.98 us, narrow QKV 44.33 us,
K-dominant projection 72.50 us, and LM head 601.12 us. The full frozen
benchmark measured ROCm at 1.494 s wall / 1418.775 ms decode and Mojo at
1.766 s wall / 1503.337 ms decode. Projection/MLP fell from 374.018 to
363.305 ms. The intended profiled N=3072 row fell by 10.60 ms, from about
103.8 to 93.205 ms; K-dominant, QKV, and square rows remained stable at
147.234, 71.407, and 51.460 ms. The normalized wall ratio improved slightly
from 1.1833x to 1.1821x. No non-target top-10 row regressed by 5%, output
correctness passed, and temperature stayed at 36 C.

**Decision.** Accept. Correctness, target-shape, top-10, decode GPU-time, and
same-session normalized wall gates pass. Basic tile selection is now exhausted;
the remaining transformer GEMM gap requires a different kernel schedule.

## Diagnostic experiment G — Structured single-buffer register prefetch

**Hypothesis.** The legacy dynamic multistage core copies global memory
directly into two LDS stages and enables neither AMD's blocked LDS layout nor
its schedule-driven register prefetch. Modular's production AMD core instead
prefetches the next A/B K slab into registers while MFMA consumes the current
single LDS slab, then writes the prefetched slab to LDS after the read barrier.
Porting only this dataflow to a runtime-M/N/K, non-transposed B[K,N] diagnostic
kernel should hide VMEM latency and remove the duplicated LDS footprint without
hardcoding model dimensions. Static BM/BN/BK regimes remain launch-time
choices; all tensor extents and the K-loop bound remain runtime values.

**Predicted effect.** On exact canonical inputs, the new core should improve
at least two of the three remaining transformer shapes by >=15%: K-dominant
72.5 -> <=62 us, N=3072 54.4 -> <=46 us, and N=2304 44.3 -> <=38 us. The
8192-cubed ceiling should rise materially above 41 TFLOP/s if VMEM latency and
LDS duplication are principal causes. Every output must pass the unchanged
fp32-reference tolerance. A null result falsifies register prefetch as the main
missing ingredient and redirects the next isolated change to the structured
blocked/swizzled LDS fragment layout.

**Measured effect.** The runtime-dynamic diagnostic compiled and every tested
configuration matched the torch-ROCm bf16 result exactly (and therefore passed
the fp32-reference tolerance). Performance falsified the hypothesis. The best
K-dominant variant was 95.155 us versus the accepted direct 68.064 us. The
best N=3072 result was 50.039 us versus the accepted 49.694 us. The square
shape was 37.517 us versus the accepted/canonical 35-36 us. Only N=2304 showed
a small win, 42.174 us versus roughly 44 us, far below the predicted >=15%
improvement and insufficient to offset the other regressions. Temperature
stayed at 35 C.

**Decision.** Reject and remove the diagnostic kernel and bindings. A
single-LDS global-to-register prefetch schedule without Modular's blocked and
swizzled LDS fragment layout is not the missing core improvement. No
production route changed, so no full profile is warranted. The next core
hypothesis must isolate the LDS layout / fragment-load side rather than retain
this slower pipeline.

## D5 addendum — Direct pinned MAX kernel inventory

**Hypothesis.** The pinned Modular repository may contain a faster tuned BF16
matmul that the eager ATen lowering simply fails to select. Calling it directly
on the worst transformer shape would make routing the cheapest remaining fix.

**Predicted effect.** A usable existing kernel must run the runtime-equivalent
`M=512, N=768, K=3072` BF16 case within 1.15x of ROCm's 42.131 us (<=48.45 us)
and compile for gfx942. If it is slower than the accepted 68.064 us dynamic
kernel or requires CDNA4, D5 is closed and no production routing changes.

**Measured effect.** The exact-revision direct Modular benchmark's standard
structured `AMDMatmul` took 172.554 us over 100 cache-busting iterations
(14.001 TFLOP/s). Its fixed 256x256 output tile launches only 6 workgroups for
this shape. The newer ping-pong kernel failed to compile on gfx942 with
`MMA shape requires CDNA4 or newer`; both ping-pong and four-wave sources are
explicitly MI355X/CDNA4. The benchmark's vendor branch also attempted the
missing `cublasCreate_v2` symbol on the ROCm host, so it could not supply a
baseline; the canonical same-session ROCm measurement remains 42.131 us.

**Decision.** Reject stock-kernel rerouting. No compatible existing MAX kernel
meets the target, and the current dynamic MFMA kernel is already 2.54x faster
than stock `AMDMatmul` on the worst shape. Proceed to a separate blocked/
swizzled LDS hypothesis; do not change production routing.

## Diagnostic experiment H — Swizzle both MI300X LDS operands

**Hypothesis.** The accepted dynamic multistage core writes both BF16 operands
to plain row-major LDS and disables every LDS swizzle on AMD. Its MFMA fragment
loads therefore serialize on LDS banks, especially across the long K loop.
Applying one matching `make_ldmatrix_swizzle` permutation to each operand's
DRAM-to-LDS destination and LDS-to-register fragment load changes only the LDS
layout; runtime M/N/K, tile selection, fused bias, and the two-stage schedule
remain identical.

**Predicted effect.** The K-dominant canonical median should improve by at
least 15%, from 68.064 to <=57.85 us, because it performs four times as many
fragment loads as the K=768 projections. At least one of N=2304 or N=3072
should improve by >=8%, and every result must pass the unchanged fp32-reference
tolerance. If the effect is smaller or correctness fails, reject swizzling as
an isolated fix and restore the exact Change 9 linalg package.

**Measured effect.** All four decode addmm outputs failed correctness. Maximum
errors were 197.894 (square), 355.168 (K-dominant), 187.428 (N=2304), and
205.218 (N=3072), versus allowed limits of 0.954, 2.146, 0.931, and 1.050.
Raw medians were 33.219, 68.778, 40.144, and 47.040 us respectively. Thus the
K-dominant target did not improve at all; apparent wins on the other shapes
are invalid. Temperature remained 35 C, with no throttling evidence.

**Decision.** Reject at the correctness gate and restore the exact Change 9
package and backend source without a full profile. The generic
`make_ldmatrix_swizzle` permutation is not the blocked-product vector-space
mapping required by gfx942's MFMA lane layout. Any future LDS experiment must
use Modular's paired `RegTileWriterLDS.copy_blocked` + `MmaOp.load_frag`
mapping, not independently swizzle the legacy row-major multistage layout.

The restore gate passed on the K-dominant canonical case: Mojo max error
1.253998 <= 2.145691 and median 72.368 us (ROCm 41.821 us), consistent with
accepted Change 9 run-to-run behavior.

## Diagnostic experiment I — Matched blocked-product MFMA fragments

**Hypothesis.** Experiment H failed because it applied a row-major ldmatrix
swizzle to a gfx942 MFMA fragment layout. Modular's structured MI300X building
blocks instead pair `RegTileWriterLDS.copy_blocked` with `MmaOp.load_frag`
under one vector-space swizzle. A runtime-dimension diagnostic using exactly
that pair, plus a coalesced B[K,N] load and an explicit register-to-LDS
transpose, will remove LDS bank conflicts without changing the PyTorch operand
layout or making any tensor extent compile-time.

**Predicted effect.** Both generic 32x32 and 64x32/BK32 regimes must pass the
fp32-reference correctness gate. The best K-dominant result must beat the
accepted direct 68.064 us by >=15% (<=57.85 us), and at least one K=768
projection must improve by >=8%. The explicit B transpose is rejected if its
scalar LDS stores erase the fragment-load gain. No production route changes
until these thresholds pass.

**Measured effect.** Both generic regimes passed correctness exactly, with
Mojo max error equal to torch-ROCm BF16's 1.0728455. Performance failed the
primary gate: 32x32 took 92.669 us and 64x32 took 87.487 us, versus the
accepted direct 68.064 us. The explicit B[K,N] register-to-LDS transpose
requires scalar LDS stores and costs more than the blocked fragment layout
saves. Temperature remained 35 C with no throttling evidence.

**Decision.** Reject and remove the diagnostic kernel, bindings, and imports.
No production route changed and no full profile is warranted. For PyTorch
Conv1D's native [K,N] weight layout, the structured kernel needs a native
non-transposed-B DMA/writer primitive; adapting the current transpose-B
blocked layout inside each workgroup is not competitive.

## Diagnostic experiment J — gfx942 double-rate BF16 MFMA shape

**Hypothesis.** The accepted dynamic kernel asks MAX's generic
`get_mma_shape`, which selects 16x16x16 BF16 MFMA on CDNA. The pinned source
also implements gfx942's 32x32x16 double-rate BF16 instruction, which produces
four times the output per instruction at the same K depth and directly matches
the accepted 32x32 warp tiles. Selecting only that hardware MMA shape inside
the unchanged runtime-dimension multistage core should reduce instruction and
fragment-load overhead without changing global/LDS layout or dispatch.

**Predicted effect.** The existing 32x32/BK32/two-warp-K and
64x32/BK32 configurations must compile and pass the fp32-reference gate. The
K-dominant median should improve from 68.064 to <=55 us and K=768 projections
by >=10%. A target-constraint failure proves the instruction is unavailable on
gfx942 in this compiler; any correctness or performance failure rejects the
shape without a production route.

**Measured effect.** Both variants compiled for gfx942 and passed correctness.
The accepted 32x32/BK32/two-warp-K geometry measured 68.126 us with the
double-rate selection, statistically identical to its 68.064 us baseline.
The 64x32 variant regressed to 108.764 us. Temperature stayed at 36 C.

**Decision.** Reject and remove the MMA-shape selector and both benchmark call
sites. The generic TensorCore abstraction already lowers the 16x16 logical
tiling efficiently on gfx942; explicitly selecting 32x32x16 provides no
instruction-level gain in this multistage schedule.

## Diagnostic experiment K — 64-row reuse with two warp-K partitions

**Hypothesis.** The accepted K-dominant 32x32/BK32/two-warp-K route launches
384 workgroups x 2 waves = 768 waves. A 64x32/BK32/two-warp-K route launches
192 workgroups x 4 waves = the same 768 waves, while each B slab is reused
across twice as many output rows. The earlier 64x32/two-partition test used
BK64, doubling LDS and reducing occupancy; isolating BK32 may retain grid fill
and reduce weight traffic.

**Predicted effect.** The exact K-dominant case must pass correctness and
improve from 68.064 to <=58 us. If it does, test K=768 shapes before routing;
otherwise reject immediately. M/N/K stay runtime values and this remains a
generic tile-regime candidate.

**Measured effect.** The candidate passed correctness with max error 1.253998
against the 2.145691 limit, but measured 87.584 us (p10 83.904, p90 89.084),
a 28.7% regression from the accepted 68.064 us. Temperature remained 36 C.

**Decision.** Reject and remove the benchmark call site. Matching total wave
count is insufficient: the 256-thread workgroups and warp-K reduction lose
more occupancy/scheduling efficiency than 64-row B reuse saves. Static tile,
BK, grouping, aspect ratio, MMA shape, and in-workgroup K partition searches
are now exhausted for the legacy core.

## Diagnostic experiment L — gfx942 XCD/L2 block swizzle

**Hypothesis.** The accepted multistage kernel explicitly disables its block
swizzle on AMD, so monotonically increasing `(block_x, block_y)` can distribute
neighboring tiles poorly across MI300X's XCDs and L2 slices. Modular's newer
AMD kernels include chiplet/L2 swizzling as a first-class schedule component.
Enabling the existing bijective `block_swizzle` only for gfx942 BF16 changes
grid traversal—not tile math, global/LDS layout, MFMA, or runtime dimensions—
and may increase A/B cache reuse across adjacent output tiles.

**Predicted effect.** All four decode addmm shapes must pass correctness. The
K-dominant median must improve by >=10% (68.064 -> <=61.26 us), with no K=768
shape regressing >5%. A smaller effect is rejected because a dependency-level
schedule patch needs material benefit. No production source route changes
until the full micro table passes.

**Measured effect.** All four correctness gates passed, but every target was
flat or slower. Medians were 35.900 us (square), 74.593 us (K-dominant),
44.355 us (N=2304), and 54.797 us (N=3072). Relative to accepted direct
values, the K-dominant and wide-N shapes regressed materially. Temperature
remained 36 C with no throttling evidence.

**Decision.** Reject the dependency patch and restore the exact Change 9
linalg artifact and backend source without a full profile. The stock
NVIDIA-oriented block swizzle is a bijection on gfx942 but degrades MI300X
cache/XCD locality for these grids; a useful chiplet schedule would need an
AMD-specific mapping, not this existing function.

## Diagnostic experiment M — dynamic two-way split-K for K-dominant GEMM

**Hypothesis.** The accepted `(M,N,K)=(512,768,3072)` route has 384 small
32x32 output workgroups but assigns the entire K loop to each workgroup.
The matching hipBLASLt trace instead uses a `64x48x128` macro tile and launches
264 workgroups for only 128 logical output tiles; its `GSUAMBSK` kernel name
and 2.06x grid expansion identify a global split/stream-K schedule. Modular's
portable multistage kernel already has a single-launch workspace-based split-K
implementation. Making only its tensor extents runtime-dynamic and dispatching
a reusable K-dominant regime to two K partitions should expose the same
parallelism without hardcoding a PyTorch tensor dimension.

**Predicted effect.** First test a benchmark-only, two-partition
`64x48x128`-class schedule with a normal MAX-allocated fp32 workspace and a
single bias/reduction epilogue. It must pass the unchanged fp32-reference
tolerance and launch exactly one partial-GEMM kernel plus one reduction
kernel. Allocation, GEMM, and reduction are all included in the measured
operator time. The canonical median must improve from 68.064 us to <=55 us;
the eventual production target remains <=1.15x ROCm (48.45 us). If the stock
multistage split path cannot accept dynamic M/N/K, patch only that general
capability in the pinned Modular source and keep tile sizes—not tensor
extents—compile-time.

**Measured effect.** The direct non-split `64x48x128` control passed
correctness (max error 1.253998 <= 2.145691) but was slow at 110.419 us. The
dynamic split wrapper failed correctness for every control: two-way
`64x48x128` produced max error 326.964, its one-partition control produced the
same 326.964 error, and the already-proven `32x32x32` geometry produced
333.617. Replacing legacy `LayoutTensor` views with runtime-strided
`TileTensor` views did not change the failure. Matching the stock wrapper's
BF16 partial workspace instead of fp32 also failed identically. Invalid raw
medians ranged from 59.625 to 150.372 us and are not speedups. Junction
temperature stayed at 35-36 C.

**Decision.** Reject and remove the wrapper, workspace, bindings, and control
configs. The pinned multistage split implementation requires static operand
layouts for correct nested-kernel code generation on gfx942; compiling it with
dynamic extents silently produces incorrect partials. Hardcoding GPT-2 tensor
dimensions would violate the eager-backend contract, so this is not a viable
route. No production dispatch changed and no full profile is warranted.

## Diagnostic experiment N — four in-workgroup K partitions at BK32

**Hypothesis.** Global split-K is unavailable without static operand layouts,
but the accepted multistage core supports dynamic dimensions and K
partitioning among waves inside one workgroup. The prior four-partition test
used BK64 and measured 105.650 us; that doubles the accepted BK32 shared-memory
slab and changes two variables. A `32x32/BK32` tile with four warp-K partitions
uses 256 threads and cuts each wave's K loop in half relative to the accepted
two-partition route while retaining its best-performing slab depth.

**Predicted effect.** The canonical K-dominant output must pass the fp32
reference gate and improve from 68.064 to <=58 us. If it fails, the extra
in-workgroup reduction/occupancy cost outweighs the shorter K loop and the
entire warp-K decomposition family is closed. No production route changes
until the microbenchmark passes.

**Measured effect.** The candidate passed correctness with max error 1.253998
against the 2.145691 limit and improved the median from 68.064 to 63.555 us
(p10 63.145, p90 64.755), a 6.6% gain. It missed the predeclared <=58 us
acceptance gate. Junction temperature stayed at 35 C.

**Decision.** Reject and remove the benchmark config. Four BK32 warp-K
partitions shorten each wave's K loop but the extra waves and in-workgroup
reduction consume most of the gain. Because both BK32 and BK64 four-partition
variants now miss the target, close the in-workgroup K-decomposition family.
No production route changed and no full profile is warranted.

## Diagnostic experiment O — deeper multistage pipeline on accepted geometry

**Hypothesis.** Every accepted dynamic MFMA route uses two LDS pipeline stages,
while Modular's generic multistage default is four. On the K-dominant shape,
the accepted `32x32/BK32/two-warp-K` geometry executes a long K loop, so a
third or fourth stage may hide global-to-LDS latency without changing tile
math, operand layout, or grid fill. Stage one was already incorrect; stages
three and four have not been isolated.

**Predicted effect.** Both candidates must pass correctness. The best canonical
median must improve from 68.064 to <=58 us, and its LDS footprint must remain
within gfx942 limits. If neither passes, pipeline depth is closed and the next
work must replace the core data-movement structure rather than tune another
legacy parameter.

**Measured effect.** A new standalone Mojo harness compiled only the selected
specialization and kept M/N/K runtime-dynamic. In the same session, all three
variants passed the deterministic BF16 correctness smoke test:

| Stages | median us | p10 us | p90 us |
|---:|---:|---:|---:|
| 2 (control) | 63.5615 | 63.3353 | 64.0441 |
| 3 | 63.5340 | 63.3501 | 64.0712 |
| 4 | 62.7755 | 62.4695 | 63.3676 |

The best candidate improved only 1.24% over the same-harness control and
missed the <=58 us gate.

**Decision.** Reject and remove the production sweep configurations. More LDS
pipeline depth does not address the dominant cost on this geometry, so the
pipeline-depth family is closed.

## Iteration methodology — standalone Mojo specialization harness

`bench_mfma_direct.mojo` accepts M/N/K at runtime and tile/regime parameters
as build-time definitions. It compiles one selected implementation in about
one second instead of compiling the complete Python eager extension in roughly
six minutes. It is now the inner-loop screen: 25 warmups, 100 synchronized
measurements, median/p10/p90, and a deterministic correctness smoke test.
Candidates still have to pass the torch-ROCm fp32 random-input correctness
gate and the frozen Python full-profile regression gate before production
acceptance.

## Diagnostic experiment P — native-layout B data movement

**Hypothesis.** The accepted stock multistage kernel pays for transforming the
PyTorch `Conv1D` B operand from its native `[K,N]` layout into MFMA fragments.
A dynamic-shape kernel that loads B coalescently into LDS in native layout and
forms each gfx942 `32x32x8` MFMA fragment directly may remove that overhead.

**Predicted effect.** The standalone K-dominant shape must pass correctness and
improve from the accepted 68.064 us to <=57.85 us (15% below the accepted
baseline). A native-LDS candidate slower than the baseline closes this
data-movement design; it cannot justify integration or the fp32 random-input
gate.

**Measured effect.** Hardware `ds_read_tr16_b64` could not be selected on
gfx942 (the compiler reports that the corresponding intrinsic is unavailable),
so two gfx942-compatible layouts were measured. Direct coalesced global loads
of B fragments passed correctness but took 204.089 us. Coalesced vector loads
of native B into LDS followed by scalar strided fragment reads also passed and
took 90.539 us (p10 90.228, p90 91.019), versus the accepted 68.064 us.

**Decision.** Reject. On gfx942, scalar strided LDS fragment assembly costs
more than the stock kernel's transformation. No production route changed and
no full profile is warranted.

## Diagnostic experiment Q — ROCm-matched 64x48 MI16 macro-tile

**Hypothesis.** The ROCm trace identifies `MT64x48x128`, `MI16x16`, 256
threads, and roughly two global K partitions on the K-dominant shape. Mojo's
generic gfx942 core already selects 16x16x16 BF16 MFMA, but the accepted route
uses a 32x32 output tile. A runtime-shape 64x48 output regime may recover the
reuse and instruction-level parallelism missing from the accepted small tile.
Because the exact Tensile wave decomposition is not encoded explicitly in the
trace name, test only the small set of legal 64x48 decompositions with one,
two, or four output/K waves and BK in {32,64,128}.

**Predicted effect.** Every candidate must pass the standalone correctness
smoke. The best median must be <=58 us versus the accepted 68.064 us. A result
above the gate rejects the macro-tile hypothesis; no production source changes
until a candidate passes the random-input fp32 gate.

**Measured effect.** All launchable candidates passed the deterministic BF16
smoke. The one-wave output decompositions measured 119.466 us (BK32), 117.858
us (BK64), and 105.623 us (BK128). A two-wave output decomposition measured
118.055 us. The four-wave K decomposition measured 148.596 us, and a two-wave
BK64 K decomposition measured 249.173 us; BK128 exceeded the valid launch
resource limit. Junction and memory temperatures remained 35 C and 31 C.

**Decision.** Reject. The hipBLASLt macro-tile name does not describe its
stream-K scheduling and operand pipeline; reproducing only its 64x48 tile is
55% slower than the accepted 68.064 us route. No production source changed.

## Diagnostic experiment R — direct global-to-LDS copy in generic MFMA

**Hypothesis.** The accepted route's dumped gfx942 ISA uses register-bounce
copies (`global_load_dwordx4` followed by `ds_write_b128`) and allocates 82
VGPRs per wave. Modular's MI300X TileIO design instead calls out cooperative
`load_to_lds`, which emits `buffer_load_*_lds` without using VGPRs for the
payload. Replacing only the AMD global-to-LDS copy primitive in the generic
multistage core should reduce copy instructions/register pressure and improve
load/MFMA overlap while preserving the runtime tensor layouts.

**Predicted effect.** The unchanged `32x32/BK32/two-warp-K` standalone shape
must pass correctness, its ISA must contain `buffer_load_*_lds` instead of the
global-load/store pair, and its median must improve from 68.064 to <=57.85 us.
If it fails correctness or speed, restore the exact Change 9 linalg package;
no production dispatch change is allowed.

**Measured effect.** The pinned repository's dependency-aware Bazel build
succeeded, but elaborating the unchanged dynamic K-dominant specialization
then crashed the gfx942 AMDGPU instruction selector at the direct LDS load:
`LLVM ERROR: Do not know how to expand this operator's operand`. A standalone
raw-pointer `AMDBufferResource.load_to_lds` control failed at the same pass.
No timing exists because neither candidate produced an executable. The
accepted ISA control remained 63.512 us (p10 63.291, p90 63.902), correct, at
35 C before the attempted substitution.

**Decision.** Reject this implementation at the compile gate and restore the
exact Change 9 linalg package. The pinned compiler's direct-LDS lowering only
accepts the more constrained static-layout `TileLoaderLDS` operand form; it
cannot currently replace the generic runtime-layout copy. Hardcoding tensor
strides or extents to force that form is prohibited. No full profile is
warranted.

## Diagnostic experiment S — buffer-resource register-bounce loads

**Hypothesis.** Direct-to-LDS cannot be lowered for the generic dynamic layout,
but its buffer-resource addressing is independently useful. The accepted ISA
uses flat `global_load` operations and extensive per-lane 64-bit address
arithmetic. Loading the same SIMD8 BF16 vectors through
`AMDBufferResource.load`, then retaining the existing register-to-LDS store,
should reduce address instructions and VGPR pressure without changing the
pipeline, data layout, MFMA sequence, or synchronization.

**Predicted effect.** The dynamic K-dominant specialization must compile, pass
correctness, emit `buffer_load` in place of `global_load`, and improve the
accepted 68.064 us median to <=57.85 us. Any correctness failure or smaller
gain rejects the isolated addressing change and restores Change 9.

**Measured effect.** The candidate compiled and passed the deterministic BF16
smoke, but regressed to 92.241 us (p10 91.962, p90 92.778) versus the restored
63.572 us same-harness control. ISA confirmed eight `buffer_load` instructions
replaced the operand flat loads, while four unrelated flat loads remained.
VGPR allocation fell only from 82 to 80 and the twelve LDS stores remained.
Junction temperature was 35 C.

**Decision.** Reject and restore Change 9. Buffer descriptors save only two
VGPRs here and their load schedule is substantially slower than LLVM's flat
global-load addressing. The useful direct-LDS path cannot be decomposed into a
buffer-load plus ordinary LDS store on this compiler. No full profile is
warranted.

## Diagnostic experiment T — constrained dynamic-stride TileLoaderLDS

**Hypothesis.** Experiment R failed because its distributed generic fragments
presented arbitrary per-lane LDS pointers to the gfx942 selector. Modular's
working `TileLoaderLDS` presents one uniform, statically tiled LDS base per
wave. A small standalone loader can preserve that lowering pattern while
keeping the global row stride, M/N/K, and loop bound runtime values. This tests
whether static tensor extents are truly required or only the LDS tile geometry
must be compile-time.

**Predicted effect.** The native-layout 32x32 MFMA control must compile and pass
correctness. Direct DMA must improve its 90.539 us median to <=70 us to justify
porting the loader into a more complete scheduled core. Compiler failure or a
smaller gain rejects this loader form; it is never eligible for production
without the subsequent torch fp32 gate.

**Measured effect.** The constrained loader preserved a uniform static LDS tile
base and moved only the global row stride to runtime, but the pinned gfx942
instruction selector still crashed at `load_to_lds` with `LLVM ERROR: Do not
know how to expand this operator's operand`. Temperatures remained 35/30 C;
no executable or timing was produced.

**Decision.** Reject and remove. The pinned compiler requires a compile-time
global row stride for this intrinsic form. Making GPT-2 K/N compile-time would
violate the eager backend's dynamic-dimension contract, so direct LDS is closed
for this dependency version.

## Diagnostic experiment U — in-core dynamic global split-K

**Hypothesis.** The earlier split-K wrapper failed even with one partition
because constructing runtime split tensor views before entering the generic
kernel corrupted its iterator layout. The already-correct core can instead use
`block_idx.z` to offset its existing full-tensor iterators by a runtime K-slab
range and its output pointer by one workspace slice. This preserves the proven
dynamic A/B layouts and local two-wave K reduction while adding two global K
partitions, matching the hipBLASLt trace's roughly 2x launch expansion.

**Predicted effect.** A two-way global split with BF16 inputs, fp32 partial
workspace, and one reduction+bias kernel must pass correctness and launch
exactly two kernels. Its combined median must improve from 68.064 to <=55 us.
M/N/K and their strides remain runtime; only tile and split-count regimes are
compile-time. Failure restores the exact Change 9 dependency.

**Measured effect.** The two-way candidate compiled and launched, but failed
the deterministic correctness smoke: output element 393215 was 1 instead of
1537. The first three sampled positions passed, which narrows the defect to
the shifted fp32 workspace/output view near the final tile, but any uncovered
output is an unconditional correctness failure. No performance result is
eligible for consideration.

**Decision.** Reject and restore the exact Change 9 linalg package. The
in-core split arithmetic is not production-safe with the generic dynamic
output layout. Further debugging was stopped at the user's request; the
accepted runtime-dynamic Change 9 dispatch remains the best kernel.

# nanoGPT training-step MI300X journal (Linear GEMM target)

Second frozen workload, same machine and dependency pin: nanoGPT GPT-2 124M
(12 layers, 12 heads, 768 embedding, `bias=False`, `vocab_size=50304`), batch
48, block 1024, BF16 autocast, fused AdamW, gradient clipping, eager, no
`torch.compile`. The target is every Linear GEMM the step issues: 5 forward,
5 data-gradient and 5 weight-gradient shapes, weighted by calls per step.
Acceptance is mojo/rocm <= 1.02 on the per-step weighted total, with both of
`harness/nanogpt_train/bench_linear_gemm.mojo`'s exact-equality gates passing
(all-ones, and `--pattern-check=1`).

Protocol is unchanged: >= 25 warmups and >= 100 individually synchronized timed
iterations for every number, timing from `--kernel-trace` or the harness itself
and never from a `--pmc` run, and every tensor extent stays a runtime value.

## Baseline verification — 2026-07-25

`/tmp/bench_linear_gemm` at commit 0cab8db: per-step weighted total mojo
506.905 ms, rocm 71.639 ms, **ratio 7.076**, of which 140.883 ms is the
transposed-operand materialization that PyTorch-ROCm does not pay. Four of the
fifteen cases fail the all-ones gate: `attn_c_attn_fwd`, `attn_c_proj_fwd`,
`mlp_c_fc_fwd` (about 24.85% of outputs wrong each) and `lm_head_dgrad` (100%
wrong, every output short of K by exactly 128).

## Defect analysis D6 — the four wrong GEMMs are two unrelated defects

**Hypothesis.** The three forward failures share a selected configuration
(`BM=32 BN=64 WM=16 WN=32 BK=32`, four warps) and a fraction close to one
quarter, so they are one defect tied to that geometry. `lm_head_dgrad` is a
different one: it is the only case whose K is 50304 and the only one failing
100%.

**Predicted effect.** Diagnosis changes no runtime. If the forward defect is
geometry-bound it must reproduce at tiny shapes with the same tile/warp
decomposition and disappear when the decomposition changes, and if
`lm_head_dgrad` is an expectation error rather than a kernel error it must pass
the independent `--pattern-check` gate, which does not depend on K being
representable.

**Measured effect, `lm_head_dgrad`.** It passes `--pattern-check=1`
unchanged. BF16 has 8 significand bits; 50304 = 2^15 + 2^14 + 2^10 + 2^7 needs
9, and round-to-nearest-even gives 50176 — exactly the reported deficit of 128.
`torch.tensor(50304., dtype=float32).to(bfloat16)` returns 50176.0 on this
build. The kernel is correct; the harness's all-ones gate asserted an
unrepresentable expectation. Every other K in the table (768, 2304, 3072,
49152) is BF16-exact, which is why only this row failed.

**Measured effect, the three forward GEMMs.** Reproduced deterministically in a
standalone launcher at 64x64x32. With all-ones operands the wrong outputs are
exactly `K/2` (one of the two MMA k-steps missing), confined to one warp row,
one `n_mma` index and two of the four accumulator elements per lane. With a
per-k-slice weighted B (`B[.,kk] = 1 << (kk/16)`) at K = 32 the wrong value is
16, i.e. the *second* MMA's contribution is the one that vanishes. Which warp
and which block are affected changes with K and is not uniform across
identical blocks of the same launch.

A geometry sweep at (m,n,k) = (1024,1024,64), all with `transpose_b=True`,
`BK=32`, two pipeline stages:

| BM | BN | WM | WN | warps | result |
|---:|---:|---:|---:|---:|---|
| 32 | 64 | 16 | 32 | 2x2 | **wrong** |
| 32 | 64 | 32 | 16 | 1x4 | **wrong** |
| 16 | 64 | 16 | 32 | 1x2 | **wrong** |
| 16 | 128 | 16 | 32 | 1x4 | **wrong** |
| 96 | 128 | 32 | 64 | 3x2 | **wrong** |
| 32 | 32 | 16 | 32 | 2x1 | correct |
| 32 | 128 | 16 | 32 | 2x4 | correct |
| 32 | 256 | 16 | 32 | 2x8 | correct |
| 64 | 64 | 16 | 32 | 4x2 | correct |
| 32 | 64 | 16 | 16 | 2x4 | correct |
| 32 | 64 | 16 | 64 | 2x1 | correct |
| 32 | 64 | 32 | 32 | 1x2 | correct |
| 64 | 128 | 16 | 32 | 4x4 | correct |
| 64 | 128 | 32 | 32 | 2x4 | correct |
| 128 | 128 | 16 | 64 | 8x2 | correct |
| 32 | 128 | 16 | 64 | 2x2 | correct |
| 64 | 64 | 16 | 64 | 4x1 | correct |
| 64 | 64 | 32 | 32 | 2x2 | correct |

Every one of the eighteen is correct with `transpose_b=False`. The failing
`BM=32 BN=64 BK=32` block tile is correct when the *same* tile is launched with
128 or with 512 threads and wrong with 256. Raising `num_pipeline_stages` from
2 to 3 or to 4 makes every failing geometry correct.

Decisively: the defect reproduces at `K == BLOCK_K`, where `num_iters == 1` and
`multistage_mma`'s loop performs no global-to-LDS prefetch at all — the only
LDS writes are the prologue's, before its barrier. There is therefore no
possible write-after-read race, so this is a **code-generation defect in MAX's
two-stage `multistage_gemm_kernel` on gfx942 for transposed B**, not a
synchronization bug in the pipeline. `BM=96, BN=128` is a second, unrelated and
much simpler defect: the B-tile copy distributes `min(threads, BN*BK/simd)`
threads over 96 LDS rows while BN is 128, so rows of B are dropped.

No single structural predicate separates the wrong geometries from the right
ones (thread count, A/B copy coverage and warp counts all have
counter-examples on both sides), which is itself consistent with a scheduling
or register-allocation miscompilation rather than a logic error.

**Decision.** Accept the diagnosis. Fix `lm_head_dgrad` in the gate, not the
kernel. For the forward GEMMs, do not chase MAX's code generator: every
observed failure has a warp tile only one MMA wide in some dimension, so make
`WM, WN >= 2 * mma_dim` a compile-time precondition of the transposed-B route
and select only geometries that satisfy it. This is avoidance of a documented
dependency defect, not a workaround for a defect of ours.

## Change 10 — safe transposed-B warp geometries, and a representable gate

**Hypothesis.** Requiring a >= 2x2 MMA warp tile (and a B-copy row count that
divides BN) whenever `transpose_b=True` makes the miscompiled geometries
unreachable by construction. Comparing the all-ones gate against K rounded to
BF16 removes the false failure without weakening it: it stays an exact equality
over every output element.

**Predicted effect.** All fifteen cases pass both gates. The three forward
GEMMs change configuration and their timings change; nothing else moves.

**Measured effect.** `_amd_dynamic_mfma_gemm` now carries two comptime asserts
for `transpose_b`, and the two dispatch sites that violated them
(`32x64` with a `16x32` warp tile, `32x32` with a `16x16` warp tile) use
`32x32` warp tiles instead. Applied together with Change 11 (both touch the
same forward shapes, so their per-shape timings are not separable): all fifteen
cases pass the all-ones gate and all fifteen pass `--pattern-check=1`. Per-step
weighted total 506.905 -> 413.878 ms, ratio 7.076 -> 5.777.

**Decision.** Accept. This is priority (A): before it, 114.454 ms of the
per-step total was silently wrong, including every `nn.Linear` forward at these
shapes. Note the defect is not nanoGPT-specific — the miscompiled geometry was
the generic `else` fallback for BF16 `transpose_b`, so any Linear forward with
m >= 64, k % 32 == 0, k < 2048 and n < 8192 was affected.

## Change 11 — device-fill-driven macro tile for the many-rows regime

**Hypothesis.** The AMD dispatch never selects a macro tile larger than
128x128x64 and picks 32x64x32 or 32x128x32 for training shapes. At m = 49152
there are two orders of magnitude more output tiles than CUs, so the tile can
grow until the grid stops covering the device several times over. Make the
choice a runtime comparison between `ceildiv(m,128) * ceildiv(n,128)` and the
runtime CU count, and share the branch between both B layouts.

**Predicted effect.** A 128x128x32 tile with an 8-warp (32x64) decomposition
for grids that cover every CU at least twice, 64x128x32 for wide-N grids that
do not, 32x128x32 otherwise. The forward and data-gradient shapes should gain
40-80%; GPT-2 decode (m = 512) must not change route at all.

**Measured effect.** Standalone screen, BF16 TFLOP/s, 25 warmups and 100
synchronized iterations per point (`transpose_b=True` / `transpose_b=False`):

| shape (m,n,k) | 32x128 (32x64) | 64x128 (32x64) | 128x128 (32x64) | 128x128 (64x64) |
|---|---:|---:|---:|---:|
| 49152,2304,768 T | 110.8 | 115.0 | **123.1** | 78.8 |
| 49152,768,768 T | 103.9 | 106.4 | **113.9** | 74.6 |
| 49152,3072,768 T | 111.4 | 116.3 | **122.0** | 82.2 |
| 49152,768,2304 N | 142.2 | 151.2 | **169.2** | - |
| 49152,3072,768 N | 136.0 | 148.8 | **169.0** | - |
| 49152,768,50304 N | 146.9 | 160.4 | **177.4** | - |
| 4096,768,768 N | **80.8** | 71.8 | 71.0 | - |
| 4096,2304,768 N | 107.6 | 122.6 | **133.8** | - |
| 8192,768,768 N | 93.3 | **96.7** | 89.6 | - |
| 16384,768,768 N | 109.7 | 113.9 | **126.2** | - |

The threshold that selects the winner in every one of these rows is
`tiles >= 2*CUs`, relaxed to `tiles >= CUs` when there are at least 12 column
tiles. GPT-2 decode shapes keep their existing routes: (512,768,768) and
(512,2304,768) stay on 32x32/BK32/warp-K, (512,3072,768) stays on 64x32,
(512,50257,768) stays on 128x128x32, all because m < 1024. Prefill m = 4096
moves to 128x128 only for n >= 2304, where it is 9-12% faster.

Combined with Change 10: 506.905 -> 413.878 ms, ratio 5.777.

**Decision.** Accept. The rule is a runtime comparison of a runtime tile count
against the runtime CU count; no model dimension is compiled in.

## Change 12 — tiled LDS transpose for the strided 2-D materialization

**Hypothesis.** `_copy_strided`'s generic kernel walks a rank-8 index space one
element at a time. For a transposed 2-D read consecutive threads are `src_ld`
elements apart, so every 2-byte load pulls its own cache line: the measured
140.883 ms/step moves 26 GB, i.e. 185 GB/s on a 5.3 TB/s part. Detecting the
transposed-2-D case and staging a square tile through LDS makes both the read
and the write fully coalesced.

**Predicted effect.** The materialization should fall by an order of magnitude,
from 140.9 ms to roughly 10 ms/step. The generic kernel stays for every other
strided layout; the fast path triggers only when every leading extent is 1, the
destination is exactly row-major, and the source is that matrix read down its
columns.

**Measured effect.** The tile edge is one 128-byte line (64 BF16 elements) with
one padding column to remove the transposing access's bank conflict, and a
`TILE * 8`-thread block. Per-step materialization 141.032 -> 12.003 ms, i.e.
26 GB at 2.17 TB/s. `lm_head_wgrad`'s copy alone fell from 73.18 to 4.42 ms.
Per-step weighted total 413.878 -> 286.000 ms, ratio 5.777 -> 3.992. All
fifteen cases still pass both gates.

**Decision.** Accept. This is lead (1) from the harness README, addressed by
making the copy cheap rather than by removing it; a transposed-A GEMM would
remove the remaining 12 ms and is recorded below as not attempted.

## Change 13 — materialize B^T for the large-M transposed-B GEMMs

**Hypothesis.** With `transpose_b=True` the B tile is BN rows of BLOCK_K
elements, so each global load touches 64 bytes of a 128-byte line, and every
row block re-reads it. The `[K,N]` kernel reads BLOCK_K rows of BN elements
instead — 256-byte rows. The same screen measures the `[K,N]` route about 1.4x
faster on identical (m,n,k). Materializing B^T costs 4*n*k bytes of bandwidth,
which at these row counts is three orders of magnitude below the GEMM.

**Predicted effect.** For m >= 1024, transpose B with the Change 12 kernel and
run the `[K,N]` route. Forward shapes should rise from 113-127 to 153-177
TFLOP/s; the added copy should cost well under 1% of the GEMM. Below m = 1024
the GEMM is too small to amortize the copy and that regime keeps the `[N,K]`
kernel.

**Measured effect.** Forward shapes 123.0 -> 166.0, 113.6 -> 153.0,
122.0 -> 167.7, 118.7 -> 168.6 and 126.7 -> 174.5 TFLOP/s. The five B
transposes per step total about 0.05 ms (the harness's reported
materialization total moved 12.003 -> 12.054 ms). Per-step weighted total
286.000 -> 258.885 ms, ratio 3.992 -> 3.614. Both gates pass.

**Decision.** Accept. B^T is an ordinary per-call temporary, freed on the same
stream; nothing is cached between calls and no input buffer is written.

## Change 14 — four warp-K partitions in the deep-K regime

**Hypothesis.** The deep-K route (`k >= 2048 and k >= 2n`) uses a 32x32 tile
with two in-workgroup K partitions. At k = 49152 that grid leaves most SIMDs
with less than one resident wave, so the global-load latency of the
synchronous LDS fill is fully exposed; doubling to four partitions doubles the
waves per workgroup. Diagnostic experiment N measured this configuration 6.6%
faster on the *decode* K-dominant shape and rejected it only against a stricter
predeclared gate, so it should not regress decode either.

**Predicted effect.** (768,768,49152) and (2304,768,49152) improve by >= 10%;
the GPT-2 decode K-dominant shape (512,768,3072) must not regress.

**Measured effect.** TFLOP/s, two vs four partitions: (512,768,3072)
37.31 -> 39.98 (+7.2%), (768,768,49152) 66.63 -> 78.16 (+17.3%),
(2304,768,49152) 75.50 -> 82.30 (+9.0%). In the harness, `attn_c_proj_wgrad`
918.48 -> 790.40 us. Per-step weighted total 258.885 -> 257.371 ms, ratio
3.614 -> 3.593. Both gates pass.

**Decision.** Accept. It also improves, rather than degrades, the decode shape
the earlier journal tuned.

## Diagnostic experiment V — macro tiles above 128x128

**Hypothesis.** Tensile picks MT256x224x64, MT256x256x32, MT192x256x32,
MT128x512x32 and MT512x128x64 for these GEMMs, so 128x128 may still be leaving
arithmetic intensity on the table at m = 49152, where even a 256x256 tile
leaves thousands of workgroups.

**Predicted effect.** At least one tile above 128x128 beats 128x128x32 by
>= 10% on a grid-filled shape.

**Measured effect.** TFLOP/s at m = 49152, `transpose_b=False`:

| tile (warp) | n=3072,k=768 | n=768,k=3072 | n=50304,k=768 |
|---|---:|---:|---:|
| 128x128 (32x64) | **169.2** | **170.1** | **175.2** |
| 128x256 (32x64) | 147.3 | 157.1 | 142.6 |
| 256x128 (32x64) | 147.2 | 156.5 | 157.4 |
| 224x128 (32x64) | 140.5 | 139.1 | 150.7 |
| 160x128 (32x64) | 131.8 | 127.4 | 138.4 |
| 128x256 (64x64) | 68.1 | 71.3 | 68.4 |
| 128x128 (32x128) | 56.8 | 54.7 | 63.2 |

256x256x32 and 256x192x32 do not launch at all: two pipeline stages of a
256x256x32 tile request the entire 64 KB gfx942 LDS budget as dynamic shared
memory. Every candidate above 128x128 is 13-60% slower.

**Decision.** Reject. Tensile's macro tiles are not transferable to this
kernel: they come with a data-movement pipeline this one does not have (see the
barrier note below), and at two stages the LDS budget caps the tile at
128x128x32 anyway.

## Diagnostic experiment W — BLOCK_K = 64

**Hypothesis.** A 64-deep K tile halves the number of barriers and doubles the
bytes per global load row (128 bytes for a `[N,K]` operand), which is exactly
the coalescing problem Change 13 works around.

**Predicted effect.** BK=64 beats BK=32 on at least the `transpose_b=True`
shapes.

**Measured effect.** TFLOP/s at (49152, 2304, 768): BK=64 gives 60.0
(128x128), 53.7 (64x128), 48.9 (64x64), 45.7 (128x64), 31.5 (32x128) against
123.1 for 128x128 at BK=32. The whole BK=64 family measures 30-71 TFLOP/s on
every shape tried, `transpose_b` either way.

**Decision.** Reject. BK=64 doubles the LDS slab, which drops the resident
workgroups per CU from two to one; the halved barrier count does not come close
to paying for it.

## Diagnostic experiment X — deeper pipelines for the latency-bound shapes

**Hypothesis.** The weight-gradient shapes are latency-bound, not
bandwidth-bound: (2304,768,49152) launches 432 workgroups of 128 threads, i.e.
0.71 waves per SIMD, and spends about 1936 cycles per k-tile where the MFMA
work is 256. A third or fourth pipeline stage prefetches further ahead.

**Predicted effect.** >= 10% on the weight-gradient shapes.

**Measured effect.** TFLOP/s with 32x128x32 (32x64 warps) at stages 2/3/4/6:
(768,768,49152) 32.8/33.8/31.8/32.4; (2304,768,49152) 86.6/89.0/46.7/47.5;
(3072,768,49152) 114.6/117.0/61.9/63.0; (768,3072,49152)
113.3/115.3/56.5/57.3. Three stages gain 2.4-3.1%; four or six collapse.

**Decision.** Reject. Three stages is below the 10% bar and costs half again
as much LDS, which is the resource that caps the macro tile. Four stages puts
one workgroup per CU and halves throughput.

## Diagnostic experiment Y — MAX's own tuned AMD matmul entry points

**Hypothesis.** MAX ships AMD-specific matmul kernels (`AMDMatmul`,
`AMDPingPongMatmul`, `AMD4WaveMatmul`, `amd_4wave_split_k_matmul`,
`warp_specialized_matmul`) that are tuned far better than the generic
multistage core; one of them may be reachable from an eager backend that holds
only device pointers and runtime extents.

**Predicted effect.** If any of them accepts runtime N and K, route to it.

**Measured effect.** None does. `AMDMatmul` reads M at runtime but takes N and
K from the static shapes (`comptime K = type_of(a).static_shape[1]`,
`comptime N = type_of(b).static_shape[0]`, `comptime assert N > 0, "N must be
known at compile time"`), uses `comptime K` as a tile extent and a loop bound,
and its `RegTileLoader`/`RegTileWriterLDS` bake `static_stride` into the
instruction stream — a dynamic stride yields `UNKNOWN_VALUE` and silently wrong
addresses rather than a compile error. Ping-pong, 4-wave and 4-wave split-K are
CDNA4: their BF16 MMA shape is 16x16x32, which asserts `_cdna_4_or_newer()` on
gfx942, and their default configurations need 128 KB of LDS against gfx942's
64 KB; all three also use the `load_to_lds` path that experiments R and T
already showed cannot lower for dynamic layouts on this pin.
`warp_specialized_matmul` takes M, N and K as compile-time parameters.
`_matmul_gpu` itself gates the entire tensor-core path on
`has_static_NK` and otherwise falls back to hipBLASLt, which is prohibited.
`multistage_gemm_kernel` is the only runtime-extent tensor-core matmul in the
tree, and it is what this backend already uses.

**Decision.** Reject. Confirms and extends the D5 addendum: on this pin there
is no vendor-free, runtime-extent, better-tuned AMD matmul to route to.

## Remaining barrier — measured, and not addressed

At ratio 3.593 the gap is the data movement inside MAX's multistage core, and
it is quantified rather than guessed. Counters on the best configuration
(128x128x32, 32x64 warps, `transpose_b=False`, m=49152 n=768 k=3072, timed
separately from the counter run at 169.9 TFLOP/s):

    SQ_INSTS_MFMA          884736
    SQ_INSTS_LDS          2001024      2.26 LDS instructions per MFMA
    SQ_LDS_BANK_CONFLICT  9750528      4.87 conflict cycles per LDS instruction
    vgpr 120, lds 32768, block 512

The instruction mix is exactly what the source predicts. Per warp per k-tile
this geometry issues 16 MFMA, 4 A-fragment `ds_read_b64`, 2 LDS stores — and
**32 scalar LDS reads for its 8 B fragments**, because with `transpose_b=False`
the B LDS tile is `(BK, BN)`, so each MFMA B fragment is four elements strided
by BN and `_load_b_amd` lowers it to four 2-byte reads. Predicted 38 LDS
instructions per 16 MFMA, measured ratio 2.26.

The two operand layouts trade the two halves of the problem and the kernel
cannot have both: `(N,K)` gives k-contiguous LDS (one `ds_read_b64` per
fragment) but 64-byte global rows, and `(K,N)` gives 256-byte global rows but
the 4x scalar fragment reads. Change 13 buys the better of the two. Closing the
rest needs a kernel that transposes during the global-to-LDS store so both the
global rows and the LDS fragments are wide, plus a global split-K for the
weight-gradient shapes, whose 432-workgroup grids leave 0.71 waves per SIMD.
Global split-K is not expressible around the stock kernel: it derives its K
loop from B and offers no `block_idx.z` k-slab, and separate launches serialize
on the stream, so the extra workgroups would not be concurrent. Both are new
kernel work and neither was attempted here.
