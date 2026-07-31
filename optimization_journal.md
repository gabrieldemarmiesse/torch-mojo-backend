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

## End-to-end validation — 2026-07-25

Same session, same seed, `bench_nanogpt_train.py --warmup 3 --iters 10
--print-loss`:

| backend | step median | tokens/s | final loss |
|---|---:|---:|---:|
| PyTorch-ROCm | 156.698 ms | 313 673 | 10.621461 |
| eager Mojo, before | 921.39 ms | 53 346 | 10.644900 |
| eager Mojo, after | 661.369 ms | 74 319 | 10.636456 |

The step improved by 259.9 ms, which matches the harness's 249.5 ms of Linear
GEMM saving to within run-to-run noise, and the loss moved from 10.6449 to
10.6365 — toward PyTorch-ROCm's 10.6215, which is the independent evidence that
the correctness fix is real and not a re-routing that hid the defect. The
residual 0.015 is not this target's: the SDPA and copy kernels still differ in
accumulation order and are the next two entries in the gap table.

## Review finding R1 — four warp-K partitions on a K the kernel cannot split

**Hypothesis.** Change 14 raised the BF16 deep-K route from two in-workgroup K
partitions to four, but the only K precondition on that path is the dispatch's
`k % 32 != 0 -> return False`. MAX's `multistage_gemm_kernel` starts partition
`p` at K tile `(k / BLOCK_K) // parts * p` and gives every partition
`ceildiv(k / parts, BLOCK_K)` tiles, both floor divisions, so the partitions
cover K exactly only when `(k / BLOCK_K) % parts == 0`. At `BLOCK_K = 32` that
is `k % 128 == 0` for four partitions where two needed only `k % 64 == 0`. If
so, every `k` that is 64 mod 128 in this regime is silently wrong, and Change
14 only ever measured `k = 3072` and `k = 49152`, both multiples of 128.

**Predicted effect.** A BF16 GEMM with `m = 512`, `n = 1024`, `k = 2112`
reaches this route (`m >= 64`, `m < 1024`, `k % 32 == 0`, `k >= 2048`,
`k >= 2n`, `n < 2048`). Its four partitions should start at tiles 0/16/32/48
and run 17 tiles each, covering `[0,17) [16,33) [32,49) [48,65)` of 66 tiles:
tiles 16, 32 and 48 accumulated twice and tile 65 dropped. So the all-ones gate
must fail on every output, and the same shape must pass on the pre-Change-14
code. `k = 2048`, `2176` and `3072` (all 0 mod 128) must pass on both.

**Measured effect.** Exactly that. The same harness source built against the
pre-change kernels and against HEAD, so the only variable is the diff:

| case | k mod 128 | pre-change | HEAD before fix |
|---|---:|---|---|
| m512 n1024 k2048 | 0 | pass, 49.75 us | pass, 45.45 us |
| m512 n1024 k2112 | 64 | pass, 50.78 us | **FAIL 524288/524288** |
| m512 n1024 k2240 | 64 | pass, 53.29 us | **FAIL 524288/524288** |
| m512 n1024 k2880 | 64 | pass, 63.80 us | **FAIL 524288/524288** |
| m512 n1024 k3072 | 0 | pass, 65.43 us | pass, 59.48 us |
| m512 n1024 k2176 | 0 | pass, 51.24 us | pass, 46.91 us |

`k = 2880` is a production hidden size, so this is not a synthetic corner.

**Decision.** Accept the finding as a regression and fix it: the partition count
now steps down to whatever the runtime `k` admits (four at `k % 128`, two at
`k % 64`, otherwise one), and the same requirement is applied to the FP32
transposed-B deep-K route, which has `BLOCK_K = 16` and therefore needs
`k % 64` for four partitions. After the fix every case above passes, the
`k % 128 == 0` cases keep the four-partition speed (2048: 45.78 us against the
pre-change 49.75) and the `k = 64 mod 128` cases land on the two-partition
timings they had before (2112: 50.91 against 50.78). The training-shape table is
unchanged at 257.4 ms, ratio 3.593, both gates passing.

## Review finding R2 — the large-m gate swallowed the K-dominant regime

**Hypothesis.** Change 11/13's `if m >= 1024` branch is taken before the deep-K
branch is ever considered, so a K-dominant narrow-N shape with `m >= 1024` no
longer reaches the partitioned 32x32 route it used before. The large macro tile
has few output columns to work with when N is narrow, so it should lose on row
count alone until m is large enough, making `m >= 1024` the wrong boundary.

**Predicted effect.** Transposed-B shapes with `n = 768` and `k >= 2048` should
be slower than the pre-change code somewhere above `m = 1024`, and the loss
should shrink as m grows, since more rows favour the macro tile.

**Measured effect.** Confirmed, and larger than expected. Pre-change against
HEAD-before-fix, all correct in both, so this was pure speed:

| case (transpose_b) | pre-change | HEAD before fix | change |
|---|---:|---:|---:|
| m512 n768 k3072 | 64.18 us | 63.80 us | -0.6% |
| m1024 n768 k2048 | 63.07 us | 107.04 us | **+69.7%** |
| m1024 n768 k3072 | 85.77 us | 144.07 us | **+68.0%** |
| m1024 n768 k4096 | 137.94 us | 180.96 us | **+31.2%** |
| m1536 n768 k3072 | 107.56 us | 145.55 us | **+35.3%** |
| m2048 n768 k3072 | 154.37 us | 154.17 us | -0.1% |
| m4096 n768 k3072 | 263.71 us | 188.29 us | -28.6% |

The crossover is at `m = 2048`, not 1024.

**Decision.** Accept and fix in two parts. The large-m branch now yields to the
deep-K branch for `1024 <= m < 2048` when `k >= 2048 and k >= 2n`; and the
partition count in that branch became an occupancy decision using the runtime CU
count, the same instrument Change 11 uses, because four partitions win by 8% at
`m = 512` (384 tiles of 32x32 against 304 CUs) and lose by 6% at `m = 1024` (768
tiles). Together these bring the whole band back to within +-0.8% of the
pre-change timings while keeping the m = 4096 gain (-28.7%) and the m = 1024
n = 2304 case, which the pre-change code computed wrongly and is now both
correct and 34.6% faster. Every extent stays runtime-dynamic; only the regime
selection reads the runtime shape and the runtime CU count.

## Review finding R3 — the documented reach of the miscompile was too narrow

**Measured effect.** The journal recorded the miscompiled transposed-B geometry
as reachable for "m >= 64, k % 32 == 0, k < 2048, n < 8192". On the pre-change
kernels, `m = 1024, n = 2304, k = 3072` transposed-B is also wrong (589824 of
2359296 outputs), and `k = 3072` is outside that band. Controls confirm the
shape of the real predicate: `k = 3072` with `n = 768` passes, `n >= 8192`
passes, and `transpose_b = False` passes.

Also worth stating plainly, because the earlier entry did not: `m = 512,
n = 768, k = 768` transposed-B is GPT-2's own decode linear forward, and it was
wrong on the pre-change code. This defect silently corrupted GPT-2 *inference*
on gfx942, not only training.

**Decision.** Correct the record here and in the harness README. The precise
predicate is whichever `(BM, BN, WM, WN)` the old dispatch selected with a warp
tile one MMA wide in some dimension, which is a function of the branch taken
rather than of a single interval in k, so the honest statement is the branch
condition plus the geometry, not a k range.

# nanoGPT SDPA target, MI300X — 2026-07-25

Baseline for this target, from `current_bench_train/comparison_v2`: SDPA
backward mojo 239.28 ms/step against PyTorch-ROCm's 33.95 (7.05x, 40.2% of the
whole remaining gap) and SDPA forward 105.07 against 9.08 (11.57x, 18.8%).
Measured standalone, ROCm's fused flash attention is 8.158 ms/step forward and
28.966 backward.

## Diagnostic experiment Z — MAX's AMD flash attention on gfx942

**Hypothesis.** `max/kernels/src/nn/attention/gpu/mha.mojo` has a
`flash_attention` with real AMD CDNA support (`has_amd_gpu_accelerator()`
branches, "AMD bf16: 4 warps (256 threads) with 16x16 MMA", `BN = 128` for AMD).
If its `MHAConfig`/`q_layout` can carry a runtime sequence length and batch, and
if it reaches no vendor library, the forward half of this target is solved by
routing to it, leaving only a fused backward to write.

**Predicted effect.** Either a reachable forward at about ROCm's 8.2 ms/step, or
a specific reason it is unreachable.

**Measured effect.** Unreachable on this pin, for three independent reasons, all
in the source:

1. **Runtime extents: partly fine, partly not.** `batch`, `seq_len` and
   `num_keys` *are* runtime (`mha.mojo:1685-1688`, plain `Int` kernel arguments),
   but `num_heads` and `head_dim` must be comptime — `comptime assert depth ==
   Int(q.layout.shape[q.rank - 1])` and the same for `num_heads`
   (`mha.mojo:607-608`), and `head_depth_known` (`mha.mojo:1693`) silently
   degrades the whole call to `mha_gpu_naive` when they are not. Under AGENTS.md
   rule 4 head count and depth as compile-time regimes behind a runtime dispatch
   would have been acceptable. What is not: the dense overload discards the
   caller's strides and re-imposes contiguous BSHD (`mha.mojo:1710-1728`), and
   the AMD kernel bakes `num_heads * depth` into comptime tile strides
   (`amd_structured/attention.mojo:245-266`, `kv_buffer.mojo:86-91`).
2. **gfx942 is refused, hard.** Every AMD attention path is written for gfx950.
   `AMDStructuredConfig.get_mma_shape()` returns 32x32x16 for BF16 prefill and
   16x16x32 for decode (`amd_structured/config.mojo:100-121`), and both land in
   branches guarded by `comptime assert _cdna_4_or_newer()` in
   `mma_amd.mojo:196-212`. gfx942 is CDNA3, so `_cdna_4_or_newer()` is False. The
   V operand load uses `ds_read_tr16_b64`, itself
   `comptime assert _cdna_4_or_newer()` (`intrinsics.mojo:1231`), and the KV DMA
   emits a 16-byte `buffer_load_dwordx4 ... lds` that only exists on gfx950
   (`amd_tile_io.mojo:1675-1680`). The CDNA3-legal `32x32x8bf16.1k` and
   `16x16x16bf16.1k` shapes exist in the stdlib but no attention config selects
   them. LDS is *not* the obstacle: at head_dim 64 that path needs about 33 KB of
   gfx942's 64 KB.
3. **There is no backward.** `flash_attention_bwd`, `mha_bwd`,
   `attention_backward` and even the substring `_bwd` return zero hits over the
   entire `/tmp/modular` tree. MAX ships forward-only attention on every
   architecture.

No vendor library is involved on that path (rocBLAS/hipBLASLt/MIOpen are lazy
`dlopen` and unreachable from it), so the prohibition was not the blocker — the
architecture gate was.

**Decision.** Reject the fusion route. Porting `amd_structured` to CDNA3 means a
new MFMA fragment geometry, an explicit LDS transpose to replace
`ds_read_tr16_b64`, and a narrowed LDS DMA — a port, not a configuration — and it
would still leave the larger half (28.97 of 37.1 ms) to write from scratch. Take
the causal-decomposition route instead, whose floor the harness README puts near
25 ms/step against ROCm's 37.1.

## Change 15 — batched MFMA GEMM

**Hypothesis.** `_amd_dynamic_mfma_dispatch` opens with `if batch != 1 ... return
False`, so all six attention batched GEMMs fall through to the portable
scalar-FFMA `pure_gemm_tiled` kernel: 167.07 ms/step against `torch.bmm`'s 36.60,
at 32-36 TFLOP/s where Tensile reaches 122-178. `multistage_gemm_kernel` takes
its output-tile coordinates from `block_idx.x` and `.y` only, so `block_idx.z` is
free to carry the batch index — which is exactly how MAX's own
`batched_matmul_kernel_gpu` batches it on NVIDIA (`linalg/bmm.mojo`, the
`is_nvidia_gpu()` branch; its AMD branch routes to `AMDMatmul`, which
experiment Y already showed needs static N and K).

**Predicted effect.** The six GEMMs reach the same 73-178 TFLOP/s band the
unbatched route reaches, i.e. roughly 50 ms/step, and `batch == 1` does not move
at all.

**Measured effect.** A wrapper kernel offsets the three operand pointers by their
runtime batch strides and calls the same core; the trailing N columns get a
batched scalar cleanup kernel. Geometry swept over 27 candidates at the three
attention shapes (25 warmups, 100 synchronized iterations each), best per shape:

| shape (batch,m,n,k,tb) | before | 128x128 (32x64) | 128x64 (32x64) | 64x128 (32x64) | 64x64 (32x32) |
|---|---:|---:|---:|---:|---:|
| 576,1024,1024,64,T | 2166.98 | **894.61** | 1243.82 | 1205.76 | 979.59 |
| 576,1024,64,1024,N | 2373.52 | n/a (BN>n) | **648.67** | n/a | 699.98 |
| 576,64,1024,1024,N | 2419.20 | n/a (BM>m) | 1119.15 | **590.29** | 698.73 |

`n/a (BN>n)` is not a missing measurement: a tile wider than N leaves *every*
column to the scalar cleanup kernel and measures 22.2 ms, which is how the
dispatch rule below was found. Rejected along the way: 128x256, 256x128,
256x64, 64x256, 128x128 with 64x64 warps, BLOCK_K 64 and three pipeline stages
were all 13-390% slower (worst: 256x128 with 64x64 warps at 4659 us on the m=64
shape).

Production dispatch (`m >= 128` and `n >= 128` -> 128x128, else the largest tile
that still fits both extents) gives **167.069 -> 51.220 ms/step against ROCm's
36.604, ratio 4.564 -> 1.399**, at 86-131 TFLOP/s. All six cases pass the
all-ones gate and all six pass `--pattern-check=1`.

`batch == 1` is bit-identical by construction — the batched branch is entered
only for `batch != 1` and the old guard's other three clauses are unchanged.
Confirmed: `bench_linear_gemm` reports per-step 257.311 ms, ratio 3.592
(recorded: 257.371, 3.593; the difference is under run-to-run noise), fifteen of
fifteen cases passing both gates, transpose materialization 12.026 ms.

**Decision.** Accept.

## Change 16 — dtype-parameterized fused SDPA softmax backward

**Hypothesis.** `sdpa_dropout_softmax_backward_kernels.mojo` was written against
`UnsafePointer[Scalar[DType.float32]]` throughout and
`fast_sdpa_dropout_softmax_backward` gated on `probs._dtype == DType.float32`, so
this BF16 workload took the five-kernel fallback: `logic_ops__bin_bcast_kernel`
50.7 ms + `logic_ops__bin_flat_vec_kernel` 23.7 + `reduce_rows_block_bfloat16`
14.7 + `elementwise_r1_w1` 11.2, about 100 ms/step, against three passes over a
1.208 GB tensor if fused.

**Predicted effect.** Parameterizing on dtype while keeping the row reduction in
float32 should land near 8-10 ms/step (43.5 GB/step at realistic bandwidth).

**Measured effect.** One warp per row, four warps per 256-thread block, 16-byte
vector accesses, online float32 reduction, and a causal regime whose extent is
`min(cols, row % q_len + 1)`. New harness
`harness/nanogpt_train/bench_attention_softmax.mojo`, 25 warmups and 100
synchronized iterations: **778.27 us/layer, 9.34 ms/step** at the nanoGPT shape
(589824 rows x 1024 cols).

Correctness against a one-thread-per-row scalar reference in the harness that
shares no code with the kernel: maximum absolute error over every one of the
604 million elements is 1e-6 at the nanoGPT shape, 1e-6 at 1000 columns and 4e-6
at 1025 columns (the scalar regime), against a two-BF16-rounding budget of
3.9e-3.

**Decision.** Accept. This was the best value-per-line item in the target.

## Change 17 — causal row softmax for the forward

**Hypothesis.** The forward softmax delegates to MAX's `nn.softmax` with the
scale and the causal mask folded into an input lambda, so it reads,
exponentiates and writes the *whole* square and only then replaces masked lanes
with -inf: 41.9 ms/step for 43.5 GB, about 1.0 TB/s on a part whose spec is 5.3.
A kernel whose row extent is the live prefix does the same arithmetic over half
the bytes.

**Predicted effect.** Roughly 4x, i.e. about 10 ms/step.

**Measured effect.** Same warp-per-row shape as Change 16, with an online
single-read max+sum in float32: **833.14 us/layer, 9.98 ms/step**, a 4.2x
improvement. Maximum absolute error against the harness's scalar reference is
8e-6 at the nanoGPT shape, 1.5e-5 at 1000 columns and 1.22e-4 at 1025 columns.

The regime is selected only for causal rows with `cols <= 8192` and
`rows >= 256`: below that a warp per row cannot fill the device and above it one
warp per row becomes the bottleneck, and non-causal softmax — which has no
prefix to exploit — keeps MAX's kernel untouched, so no other op moves.

**Decision.** Accept.

## Change 18 — contraction-side causal regimes in the batched GEMM

**Hypothesis.** With `is_causal=True` the score matrix is zero above the
diagonal, and four of the six attention GEMMs contract over one of its axes. Two
of them (`P @ V`, `dS @ K`) can only see contraction indices below the last row
of their output row block; the other two (`dO^T @ P`, `Q^T @ dS`) can only see
indices from the first column of their output column block up. Skipping the rest
multiplies exact zeros away, so it is exact for any consumer and should remove
about 44% of the operand traffic.

**Predicted effect.** Roughly 40% on those four GEMMs, i.e. about 10 ms/step.

**Measured effect.** Reached through a new `matmul_ops.CausalBmm` bridge and
`aten_fast._try_sdpa_causal_bmm`; `multistage_gemm_kernel` takes K from B only,
so a shortened range is expressed as B's runtime extent plus a pointer offset on
both operands, and A keeps its true row stride. 25 warmups, 100 synchronized
iterations:

| case | dense | causal | change |
|---|---:|---:|---:|
| fwd_out_pv (`CAUSAL_A_ROWS`) | 649.06 | 612.82 | -5.6% |
| bwd_dq_ds_k (`CAUSAL_A_ROWS`) | 648.77 | 612.88 | -5.5% |
| bwd_dv_dotT_p (`CAUSAL_B_COLS`) | 590.28 | 588.13 | -0.4% |
| bwd_dk_qT_ds (`CAUSAL_B_COLS`) | 590.40 | 587.73 | -0.5% |

Far below the prediction; experiment AA below is why. All four pass a new
closed-form gate that inspects every output element: the operand the regime
claims is causal is filled with `1` on and below the diagonal and the other with
ones, so `CAUSAL_A_ROWS` must produce `min(k, row + 1)` and `CAUSAL_B_COLS` must
produce `max(0, k - col)` — closed forms that vary along exactly the axis the
regime restricts.

That gate initially failed on 1/8 of elements in all four cases, and the fault
was the gate, not the kernel: it reused `_bf16_round`, which rounds halves away
from zero, while the store rounds half to even. The expected values here run over
every integer up to k, so a quarter of them are exact BF16 ties and half of those
disagree — exactly one eighth. The gate now uses the same hardware conversion the
store uses, and all four pass.

**Decision.** Accept, at 0.92 ms/step rather than the predicted 10. It is exact,
it costs nothing at runtime, and it becomes worth much more if the barrier in
experiment AA is ever removed.

## Diagnostic experiment AA — causal output-tile skipping, and what really binds these GEMMs

**Hypothesis.** The two GEMMs whose *output* is the masked matrix (`Q @ K^T`,
`dO @ V^T`) can skip whole output tiles above the diagonal, neither computing nor
writing them. At `m = n = 1024` with a 128x128 tile that is 28 of 64 tiles, so
they should be about 44% faster.

**Predicted effect.** 895 us -> about 500 us on both.

**Measured effect.** 895.05 -> 901.91 us. A *regression*, and the tiles really
are being skipped: a deliberately skewed shape (`batch 64, m 128, n 8192, k 64`,
where 63 of 64 tiles are masked) goes 114.64 -> 48.50 us.

The control that explains it: a **dense** GEMM at `m = 1024, n = 512` has the
same 32 live tiles per batch as the causal one has (36), and runs in **488.42 us**
against the causal 901.91. So the skipped tiles are not free — they cost about
half of a working tile.

Across every measurement here the time is proportional to the **workgroup count**
and to nothing else:

| case | workgroups | us | WG/us |
|---|---:|---:|---:|
| m1024 n1024 dense | 36864 | 884.0 | 41.7 |
| m1024 n512 dense | 18432 | 488.4 | 37.8 |
| m1024 n1024 causal (28/64 skipped) | 36864 | 901.9 | 40.9 |
| skew dense | 4096 | 114.6 | 35.7 |
| skew causal (63/64 skipped) | 4096 | 48.5 | 84.5 |

35-42 workgroups per microsecond is a **workgroup launch rate**, not a bandwidth
or a FLOP rate: a 128x128x64 tile moves 64 KB, and 64 KB per 24 ns is 2.7 TB/s,
so at these shapes the launch rate and the bandwidth limit happen to coincide —
which is exactly why removing work from inside a workgroup buys nothing while
removing the workgroup itself buys everything (the skew row: an empty workgroup
launches about twice as fast as a full one, and no faster).

**Decision.** Reject `CAUSAL_OUT` for production. It stays implemented and
exercised by the harness (`--causal=1`) because it is the evidence for this
finding, and because it becomes valuable the moment the launch rate stops
binding. Production selects only the two contraction-side regimes, which need no
assumption about the consumer; that also removes the only place where an output's
masked half would have been left undefined, and with it the eligibility
plumbing that would have been needed to make that safe.

**The barrier this identifies, measured and not addressed.** These six GEMMs
issue 4608-36864 workgroups each and are bound by how fast the hardware can
launch them. The fix is a persistent kernel: launch about two workgroups per CU
and loop over output tiles inside one, so the launch rate stops mattering and a
causal skip becomes a pure work reduction. That is not expressible around MAX's
`multistage_gemm_kernel` as called here — it allocates its LDS with
`external_memory` and ends its K loop without a trailing barrier, so reusing the
same workgroup for a second output tile needs an explicit barrier and a review of
its LDS lifetime, i.e. new kernel work inside the core rather than around it.
Larger macro tiles are the cheap version of the same idea and were measured
worse: 128x256 and 256x128 halve the workgroup count but take 2.2-2.8x as long
per workgroup, because two pipeline stages of them leave one workgroup per CU.

## Review finding R4 — the causal softmax's -inf sentinel, and a NaN-blind gate

**Hypothesis.** `scripts/sdpa_correctness.py` (the whole-op gate against a
PyTorch-ROCm FP32 reference) reported `nan` for the Mojo forward output and all
three gradients on every causal case whose row count reaches the warp-softmax
regime, while `head1_small` (129 rows, below the regime) passed. So the new
causal row softmax produces NaN, and `bench_attention_softmax` passed it anyway.

**Predicted effect.** If the cause is the online rescaling, it must involve a
lane or vector slot with no live column, because rows whose live prefix covers
every lane cannot have an unused accumulator.

**Measured effect.** Exactly that, and it is one expression. The running maximum
was initialized to `Float32.MIN`, which in Mojo is `-inf`, not the most negative
finite float. A lane with no live column keeps the sentinel in both `run_max` and
`lane_max`, so the rescaling computes `exp(run_max - lane_max)` =
`exp(-inf - -inf)` = `exp(NaN)` = NaN, multiplies it by a zero partial sum to get
NaN rather than 0, and the warp reduction then poisons the whole row. Every row
with a live prefix shorter than `WARP_SIZE * VEC` = 512 columns was NaN, which at
`q_len = 1024` is half of them. `Float32.MIN_FINITE` makes the same expression
`exp(0) = 1` times a zero sum.

**And the harness gate could not see it.** `_max_abs_diff` kept a running
`if d > worst`, and `NaN > worst` is false, so a NaN difference was silently
dropped; worse, the harness's own scalar reference used the same `Float32.MIN`
sentinel, so on the rows that were NaN *both* sides were NaN. The gate now maps
any non-finite difference, and any non-finite operand on either side, to the
largest float, so it cannot be ignored. With that gate the reported forward error
at the nanoGPT shape moves 8e-6 -> 3.1e-5 — the 8e-6 was the error over the
subset of rows that were not NaN.

**Decision.** Accept both fixes. The lesson for this harness family is that a
maximum-absolute-difference gate is not a NaN detector unless it is written to
be one, and that a reference sharing a sentinel convention with the kernel under
test can share its bug.

## End-to-end validation — 2026-07-25, SDPA target

Same session, same seed, `bench_nanogpt_train.py --warmup 3 --iters 10
--print-loss`:

| backend | step median | tokens/s | final loss |
|---|---:|---:|---:|
| PyTorch-ROCm | 156.817 ms | 313 435 | 10.621461 |
| eager Mojo, before this target | 661.369 ms | 74 319 | 10.636456 |
| eager Mojo, after | **420.566 ms** | 116 871 | 10.628205 |

The step improved by 240.8 ms and the loss moved from 10.6365 to 10.6282, i.e.
*toward* PyTorch-ROCm's 10.6215 rather than away from it, which is the
independent evidence that the rewritten kernels are not trading accuracy for
speed. (Run-to-run loss noise on this backend is about +-0.002, so the 0.008
move is signal.)

Re-profiled with `profile_nanogpt_train_aten.py --device mojo --output-dir
current_bench_train/mojo_v3` and compared against `current_bench_train/rocm`:

| target | before | after | rocm | ratio before | ratio after |
|---|---:|---:|---:|---:|---:|
| SDPA (backward) | 239.279 | **66.393** | 33.953 | 7.05x | **1.96x** |
| SDPA (forward) | 105.068 | **36.997** | 9.083 | 11.57x | **4.07x** |
| whole step | 661.4 | 421.6 | 157.0 | 4.21x | 2.68x |

Neither reaches the 1.02 acceptance ratio. The two remaining SDPA items are both
measured:

1. **`data_movement_ops__permute_copy` is 25.2 ms/step and is now the largest
   single kernel in the SDPA backward** — larger than any of its GEMMs. It is the
   four transposes the decomposition needs (dO^T and Q^T going in, dV^T and dK^T
   coming out), each `[576, 1024, 64] <-> [576, 64, 1024]`, i.e. 75 MB per
   transpose, 600 MB of traffic per layer. 7.2 GB/step in 25.2 ms is **286 GB/s**
   on a part whose spec is 5.3 TB/s, so this is the same uncoalesced
   element-at-a-time transpose that Change 12 fixed for the *unbatched* 2-D case
   (141 -> 12 ms/step) with a tiled LDS staging pass. `_copy_strided`'s fast path
   requires every leading extent to be 1, so a batched transpose does not reach
   it and lands on `permute_copy` instead. Extending that fast path with a batch
   axis on `block_idx.z` should take this to roughly 4 ms/step, i.e. **about
   21 ms/step**, and would move the SDPA backward ratio from 1.96x to about
   1.35x. Not attempted here.
2. **The workgroup launch rate**, quantified in experiment AA. At 35-42
   workgroups per microsecond the six batched GEMMs cost 50.3 ms/step where
   `torch.bmm` costs 36.6, and no amount of causal skipping helps until the
   kernel becomes persistent.

Beyond those two the decomposition is at its floor: 36.997 ms of forward is
11.4 (scores GEMM) + 9.9 (row softmax) + 7.4 (PV GEMM) + copies, against ROCm's
single 9.1 ms fused kernel, which never materializes the 1.208 GB score matrix at
all. The README's floor estimate of about 25 ms/step for a causal decomposition
assumed causal skipping would work; experiment AA is why it does not, and with
`permute_copy` fixed the honest floor for this decomposition on this hardware is
about 75 ms/step for both directions against ROCm's 43.

## Test-visible changes and pre-existing failures — 2026-07-25

`tests/test_eager_kernels.py::test_fast_sdpa_partial_gradients_save_and_compute_only_dependencies`
counts the matmuls each requested input gradient costs, and it broke on the causal
work for a real reason: with `is_causal=True` the backward's batched GEMMs go
through `_try_sdpa_causal_bmm` instead of `fast_aten_bmm`, so the spy saw zero.
It now spies on both helpers and zeroes the counters between the forward and the
backward, because the causal forward issues a batched GEMM through the same
helper. The test's intent — one matmul per gradient, and never materializing an
`S x L` intermediate — is unchanged and passes.

Two failures on this branch are **not** from this work; both reproduce identically
at commit `4cd5f30`, before any of it:

- `tests/test_fa4_host_wiring.py::test_fa4_autograd_saves_public_inputs_without_persistent_physical_copies`
  patches `aten_fast` with a namespace that has no `_sdpa_math_forward_with_dropout`,
  which is fine on an H100 where the FA4 route is taken, and an `AttributeError`
  on gfx942 where it is not.
- Four `test_fast_log_softmax_*` tests. Unrelated to attention.

`scripts/sdpa_correctness.py` reports one honest failure: the `noncausal` case, at
2.3-3.1x the PyTorch-ROCm BF16 error on the output, dK and dV. That path does not
use any kernel written here — it takes MAX's `nn.softmax` with the scale folded
into an input lambda that rounds `scores * scale` back to BF16 before the
reduction, a second rounding PyTorch's math backend avoids by scaling Q before the
BMM (the lambda's own comment flags the round-trip as a known trade-off). The
contrast is the evidence: every *causal* case, which uses the rewritten kernel and
keeps the scaled value in float32, is at 0.76-1.84x. The forward output failing at
2.95x is what makes this conclusive — the noncausal forward's only changed
ingredient is the batched GEMM, which accumulates in float32 exactly as
`pure_gemm_tiled` did. Fixing it means selecting the warp kernel for non-causal
rows too, which is a change to the general `aten::_softmax` and was not attempted
here.

## Review finding R4 — the tiled transpose exceeded the gridDim.y cap

**Hypothesis.** Change 12's tiled-LDS transpose fast path launches
`(ceildiv(cols, TILE), ceildiv(rows, TILE), 1)` with one block per row tile.
`TILE = 128 / size_of[dtype]()`, so 64 for BF16 and 16 for 8-byte elements,
while HIP and CUDA both cap gridDim.y at 65535. Above `65535 * TILE` rows the
launch should be rejected outright. The generic `_copy_strided_kernel` this fast
path replaces launches `(_gs_blocks(total), 1, 1)`, which is clamped to 4096, so
this would be a regression on shapes that previously worked rather than a
pre-existing limit.

**Predicted effect.** A BF16 transposed copy of a `(3, rows)` source into a
`(rows, 3)` destination should fail for `rows = 4_200_000`
(`ceildiv(4200000, 64) = 65625 > 65535`) and succeed for `rows = 4_194_240`
(exactly 65535).

**Measured effect.** Confirmed with a standalone launcher calling `_copy_strided`
directly, built against the pre-fix source and against the fix, two seconds each:

| source | rows | y blocks | result |
|---|---:|---:|---|
| pre-fix (859d036) | 4 200 000 | 65 625 | **`HIP call failed: hipErrorInvalidValue`** |
| post-fix | 4 200 000 | 65 625 | launch OK |

Through PyTorch the same shapes now round-trip correctly:
`rows = 4_194_240` (65535 blocks), `4_200_000` (65625) and `5_000_000` (78125)
all match a CPU reference exactly.

**Decision.** Accept and fix. The kernel now grid-strides its row tiles and the
launch clamps the y dimension to 65535. The trip count depends only on
`block_idx.y`, `grid_dim.y` and `rows`, so it is uniform across the block and
both barriers stay outside divergent control flow; a second barrier was added
after the write phase so the next row tile cannot overwrite the LDS tile while a
lane is still reading it. The nanoGPT training table is unchanged — 257.35 ms,
ratio 3.592, both gates passing, transpose materialization 11.995 ms against the
12.03 ms recorded before the change.

## Review finding R5 — the transposed-B copy-row guard did not reproduce the kernel

**Hypothesis.** Change 10's `B_COPY_ROWS` compile-time assert is the entire
remedy chosen for the miscompiled transposed-B geometries, so it must compute the
row count MAX's B-tile copy actually uses. It appears to get both non-`BLOCK_K`
factors wrong. `multistage_mma` receives `num_threads_per_warp_k_part`, that is
`config.num_threads()` divided by the partition count
(`_multistage_gemm_gpu.mojo:757`), while the guard *multiplies* by
`WARP_K_PARTITIONS`; and the kernel's `simd_size` is `simd_width_of[a_type]()`
evaluated for the device (line 258), a 16-byte gfx942 vector, while the same call
inside this host `def` returns the build machine's vector width.

**Predicted effect.** Correcting both factors should change the computed value but
not the accept/reject verdict for any geometry the production dispatch can select,
so every harness must still build and every timing must be unchanged. If a
production geometry were newly rejected the build would fail at the assert, which
is the test.

**Measured effect.** Both harnesses build unchanged, so no reachable geometry
changes verdict, and the training table is unchanged. Worked example for the
`BM=128, BN=128, WM=32, WN=64, BLOCK_K=32` large-m tile: the true row count is
`min(4*2*64, 128*32/8) * 8 / 32 = min(512, 512) * 8 / 32 = 128`, which divides
BN; the old expression gave `min(512, 128) * 1 = 128` by a different route. The
guard was arriving at usable answers for the current set by coincidence rather
than by reproducing the kernel.

**Decision.** Accept the correction. The guard now uses
`(BM // WM) * (BN // WN) * 64` with no partition factor and a device SIMD width
of `16 // size_of[dtype]()`. Behaviour today is identical; the value of the change
is that the next geometry added will be judged against what the kernel does.

## Review finding R6 — the causal batched GEMM never declined

**Hypothesis.** `_try_sdpa_causal_bmm`'s docstring promises `None` when the
operands do not qualify, but its gates are device-agnostic and the Mojo bridge
never signals refusal: `_causal_bmm_dispatch` sets `eligible = False` on any
non-gfx942 target and then quietly calls `_gemm_dtype_dispatch`. If so the helper
returns a filled tensor on every GPU, `out is None` is never true, and the
fallback chain below it is dead code whenever `is_causal` holds -- which on sm_90a
means the tuned BF16/TF32 batched bridges are skipped in favour of the portable
kernel, at one forward and three backward call sites.

**Predicted effect.** Numerics are unaffected either way, since the substituted
dense GEMM is a correct superset of every causal regime, so this is invisible
except in wall time on non-gfx942 devices. Making the bridge raise instead must
leave gfx942 timings and correctness untouched, because every gfx942 shape the
SDPA path uses is eligible.

**Measured effect.** Confirmed by reading the chain: the bridge's only refusal
channel is a raised `Error` turned into `NotImplementedError`, and architecture
ineligibility never raised. After the change, gfx942 is unchanged --
`bench_attention_bmm` 51.218 ms default and 50.448 ms causal against the 51.237
and 50.500 recorded, all cases passing; the nanoGPT step is 420.60 ms against
420.94. The sm_90a improvement cannot be measured on this host and is not
claimed.

**Decision.** Accept. The bridge now raises when the causal MFMA route is not
selectable, so the caller's own chain runs. This is the difference between a
helper that declines and a helper that answers for everyone; only the first
composes with a fallback ladder.

## Review finding R7 — the batched GEMM's N remainder fell off a cliff

**Hypothesis.** `_amd_batched_mfma_gemm` covers `(n // BN) * BN` columns with the
MFMA core and hands the residual to `_amd_batched_mfma_edge_kernel`, one thread
per output element with a full runtime K loop. `_batched_mfma_tile` chose BN from
the magnitude of N only, so any N that is not a multiple of the selected BN sends
up to BN-1 columns down that path. The route's own table puts the scalar kernel
at about 34x the MFMA cost per column, and before the batched route existed every
batched GEMM used the edge-masked tiled kernel, so a non-multiple N should now be
*slower* than it was.

**Predicted effect.** At N = 96 with m >= 128 the old selector takes BN = 64,
leaving a third of every row to the scalar kernel; the estimate from the same
table is roughly 3.3x slower than the pre-batched path, and N = 127 about 4.8x.
Reachable from `torch.bmm` directly and from causal SDPA whenever `head_dim` or
`key_length` is not a multiple of the tile.

**Measured effect.** Not measured as a regression, because the fix is cheap
enough that measuring the defect was not worth a build cycle; the mechanism is
plain in the source and the per-column cost is already tabulated in this journal.
What is measured is that the fix costs nothing on shapes that already fitted:
`bench_attention_bmm` 51.205 ms against 51.218 recorded, `bench_linear_gemm`
257.59 ms ratio 3.596 against 257.41/3.593, both under both gates.

**Decision.** Accept the finding and fix it in two parts. `_batched_mfma_tile`
now picks the widest BN among 128, 64 and 32 that *divides* N rather than the
widest that fits, so the cleanup kernel only ever handles the M edge; and both
entry points require `n % 32 == 0`, declining to the edge-masked tiled kernel
otherwise. Declining is the right answer rather than a defeat: that kernel masks
every extent and keeps its shared-memory tiling, so it is the general path, not a
slow fallback bolted under a shape-specialized fast one.

## Change 15 — tiled transpose for batched permutes, and for tall shapes

**Hypothesis.** The eager SDPA backward transposes `[batch*heads, seq, head_dim]`
four times per layer. Those run through `data_movement_ops._permute_copy`, a
rank-4 gather with one thread per output element walking a column, which the
profile showed at 286 GB/s. `op_utils`'s tiled LDS transpose already stages a
TILE x TILE block so both reads and writes are contiguous, but it only accepted
rank-2 with trivial leading extents. Giving it a batch axis and calling it from
`_permute_copy` when the permutation is an innermost-two-dims transpose should
convert those four copies to something bandwidth-bound.

**Predicted effect.** A per-step saving in the low tens of milliseconds, with
every transposed shape still exact, including extents that do not divide the
64-wide BF16 tile.

**Measured effect.** The step went from 420.75 to **409.34 ms**, an 11.4 ms
saving. The new `transpose2d` kernel accounts for 15.15 ms/step in the profile
while `permute_copy` still accounts for 30.19: nanoGPT's own
`view(B, T, nh, hs).transpose(1, 2)` is a rank-4 dim-1/dim-2 permute, not an
innermost-two-dims transpose, so it keeps the generic kernel. Its innermost
extent is contiguous in both operands, so the remaining work is a vectorized
contiguous-run copy rather than a transpose, and it is left undone.

Correctness: batched and unbatched transposed copies match a CPU reference
exactly for batch 576/1/5/3/2/7 at rows x cols of 1024x64, 64x1024, 1000x63,
65x129, 4096x2 and 2x4096, in both BF16 and FP32. `scripts/sdpa_correctness.py`
is unchanged at 25 passing comparisons with the same three pre-existing
non-causal failures at identical ratios. GEMM and batched-GEMM harnesses
unchanged.

**Decision.** Accept. The same generalization also fixed R4's launch failure,
since batch and row tiles are now both grid-strided and both clamped.

# nanoGPT training step, MI300X — permute gather and GEMM split-K, 2026-07-25

Continues the two sections above. Entry state, independently re-verified at the
start of this work: PyTorch-ROCm 156.62 ms/step, eager Mojo 409.34 ms (2.61x).
`bench_linear_gemm` 257.543 ms against ROCm's 71.639, ratio 3.595, fifteen of
fifteen cases passing both gates. Protocol unchanged: >= 25 warmups and >= 100
synchronized iterations for every number, timing never from a `--pmc` run, every
tensor extent a runtime value.

## Change 19 — vectorized run gather for contiguous-innermost permutes

**Hypothesis.** `data_movement_ops._permute_copy` was still 30.19 ms/step after
Change 15 gave it a tiled path for innermost-two-dims transposes. Logging
`TorchMojoTensor._materialize_contiguous` for one step shows exactly five
distinct (shape, stride) pairs, and three of them -- 132 of the 180 calls,
19.93 GB of the 27.18 GB moved -- have `s3 == 1`:

    72x  out (48,12,1024,64)  src strides (2359296,64,2304,1)   q/k/v
    48x  out (48,1024,12,64)  src strides (786432,64,65536,1)    y and dq/dk/dv
    12x  out (48,12,1024,64)  src strides (786432,64,768,1)      dy
    24x  out (576,1024,64)    src strides (65536,1,1024)         dO^T, Q^T
    24x  out (576,64,1024)    src strides (65536,1,64)           dV^T, dK^T

The first three are nanoGPT's `view(B, T, nh, hs).transpose(1, 2)` and its
inverse. Their innermost extent is contiguous in *both* operands, so they are
not transposes at all: they are gathers of 64-element runs, which the tiled LDS
path correctly declines and the generic one-element-per-thread kernel then
serves at 2 bytes of a 16-byte access plus three integer divisions for those two
bytes. A whole run per thread group should be bandwidth-bound instead.

**Predicted effect.** Most of the 30.19 ms. PyTorch-ROCm's own `.contiguous()`
on the identical shapes and strides measures 55.05-55.77 us per call (2.74 TB/s)
against a 4.15 TB/s ceiling for a plain contiguous copy of the same 75.5 MB on
this GPU, so about 7.3 ms/step is the target.

**Measured effect.** New harness `harness/nanogpt_train/bench_permute_copy.mojo`
times all five permutes against the production entry point and against the
generic kernel it replaces:

| case | generic | now | rocm | now GB/s | ms/step before | ms/step now |
|---|---:|---:|---:|---:|---:|---:|
| qkv | 237.29 | **45.19** | 55.21 | 3341 | 17.08 | 3.25 |
| heads_out | 237.09 | **47.67** | 55.77 | 3168 | 11.38 | 2.29 |
| dy | 236.54 | **46.00** | 55.05 | 3283 | 2.84 | 0.55 |
| bmm_seq (tiled path) | 316.42 | 67.39 | 161.93 | 2241 | 7.59 | 1.62 |
| bmm_head (tiled path) | 295.13 | 63.44 | 256.14 | 2380 | 7.08 | 1.52 |

31.30 -> 6.09 ms/step on the three run-gather shapes, against ROCm's 7.32, and
the whole training step 409.34 -> **384.67 ms**. (The two tiled-transpose rows
are Change 15's, unchanged here; they are in the table because the harness covers
every permute the step issues, and they are the reason the profile's 15.15 ms of
`transpose2d` is 3.2 ms of these plus 12.0 ms of the GEMM operand copies.)

The vector width is a compile-time regime chosen at runtime: the widest of 32, 16
and 8 bytes whose extent, strides and base addresses are all admissible.
Measured on the q/k/v shape: 45.14 us at 32 bytes, 49.29 at 16, 70.94 at 8,
125.08 at 4. When none fits -- `d3 = 63`, or a source displaced by one element --
the copy declines to the general kernel, which masks every extent, and the
harness covers both of those cases.

**Decision.** Accept. This was the highest value-per-line item left: 25 ms/step
for one kernel and one dispatch clause.

## Change 20 — global split-K for the underfilled weight-gradient GEMMs

**Hypothesis.** The four small-output weight-gradient shapes are latency-bound on
grid fill, not on anything inside a workgroup: (2304, 768, 49152) is 108
workgroups of a 128x128 tile on a 304-CU part, i.e. 0.36 workgroups per CU, and
no tile choice fixes that because the shortage is output *area*. Partitioning K
across `parts` workgroups per output tile is the only axis left. The previous
agent recorded global split-K as inexpressible around the stock kernel; it is
expressible for the same reason the batch index is, and MAX's own
`multistage_gemm_split_k_kernel` is the proof of the shape of it (unreachable
itself: it reads N and K from *static* shapes).

**Predicted effect.** The four shapes reach the 153-177 TFLOP/s band the
well-filled forward and data-gradient shapes already reach, i.e. roughly
33 ms/step off the harness total. `lm_head_wgrad`, whose 2358 tiles already fill
the device, must not move.

**Measured effect.** A wrapper kernel puts the slab index on `block_idx.z`,
offsets A along the contraction axis and gives B the slab's extent, and calls the
same 2-D core with `c_type = float32` into that slab's own workspace slice; a
second pass sums the slices and rounds once. Geometry swept over five tiles x
four slab counts on all five shapes (25 warmups, 100 synchronized iterations,
all-ones gate on every element of every configuration, A already contiguous so
these numbers exclude the transposed-operand copy):

| shape (m,n,k) | production | 2 slabs | 4 | 8 | 16 |
|---|---:|---:|---:|---:|---:|
| 2304,768,49152 | 2003.1 | 1615.3 | 1392.0 | **1067.9** | 1063.0 |
| 768,768,49152 | 742.2 | 1529.4 | 785.7 | 419.3 | **380.2** |
| 3072,768,49152 | 2022.2 | 1619.8 | 1411.5 | **1394.2** | 1403.8 |
| 768,3072,49152 | 2046.5 | 1612.0 | 1417.2 | **1394.3** | 1395.0 |
| 50304,768,49152 | 21125.9 | 21069.0 | 21092.7 | 20989.1 | 21285.6 |

all at 128x128x32 with 32x64 warps, which won on every shape and every slab
count. Rejected geometries at their best slab count: 64x128 (1162/407/1532/1523),
128x64 (1179/420/1560/1550), 64x64 (1501/554/2007/1997), 128x128 with 64x64
warps (2813/1035/3440/3419) -- 9-146% slower.

The dispatch rule doubles the slab count while the grid stays under four
workgroups per CU, while `k % (parts * BLOCK_K) == 0`, and while each slab keeps
at least 2048 contraction indices. It selects 8, 16, 8, 8 and 1 slabs for the five
rows above, which is the measured optimum in all five.

The 2048-per-slab floor is conservative for shorter K than this workload has: at
(2304, 768, **8192**) the rule stops at four slabs (261.1 us, the same as the
unsplit route) where eight measured 214.4 and sixteen 232.7, so the floor that
matches that one measurement is 1024 rather than 2048. Left at 2048 because it is
the value every production shape here was measured under and because a single
point is not enough to move a regime boundary; noted so the next agent does not
have to rediscover it.

In the harness, with the transposed-operand copy included: `attn_c_attn_wgrad`
2204.65 -> 1267.92 us, `attn_c_proj_wgrad` 793.53 -> 442.70,
`mlp_c_fc_wgrad` 2307.98 -> 1678.53, `mlp_c_proj_wgrad` 2113.71 -> 1458.84,
`lm_head_wgrad` unchanged at 25581. **Per-step weighted total 257.543 -> 226.743
ms, ratio 3.595 -> 3.165**, fifteen of fifteen cases passing all three gates.

**The exactness requirement is real, and it is the same defect shape as R1.**
`k % (parts * BLOCK_K) == 0` is not decoration: forced to 8 slabs at
`k = 49280` (1540 K tiles, divisible by 4 but not 8) every one of the 1 769 472
outputs is wrong, while 4 slabs -- what the rule actually selects -- is correct.
Accumulation stays FP32 from the MFMA accumulators through the workspace to the
one rounding in the reduction, so this is not a precision trade; what changes is
the summation order of the weight gradient, and it moves *toward* PyTorch-ROCm
(13-step loss 10.631089 -> 10.621608 against ROCm's 10.621461, on a metric whose
run-to-run spread is about 0.04, so this is agreement rather than a claim of
improvement).

**Decision.** Accept.

## Diagnostic experiment AB — warp-tile shape against the LDS instruction count

**Hypothesis.** With `transpose_b=False` an A fragment is four contiguous k
elements (one `ds_read_b64`) and a B fragment is four elements strided by BN
(four 2-byte reads), so per warp per k-tile the counts are `num_m_mmas *
num_k_mmas` cheap A reads and `num_n_mmas * num_k_mmas * 4` expensive B reads for
`num_m_mmas * num_n_mmas * num_k_mmas` MFMA. At a fixed MFMA count and a fixed
warp count, a *taller, narrower* warp tile therefore buys cheap A reads with
expensive B ones: 128x128 with WM=32,WN=64 issues 4 + 32 = 36 LDS reads per 16
MFMA, and WM=64,WN=32 issues 8 + 16 = 24 for the same 16 MFMA and the same 8
warps. If the 2.26 LDS instructions per MFMA is what binds, that is a 33% cut.

**Predicted effect.** >= 10% on the forward, data-gradient and split-K
weight-gradient shapes.

**Measured effect.** The opposite, consistently. TFLOP/s (us in parentheses):

| geometry | 49152,2304,768 | 49152,768,3072 | 49152,768,768 | wgrad 2304,768 |
|---|---:|---:|---:|---:|
| 128x128 w32x64 | **167.9** (1036) | **169.9** (1365) | **156.6** (370) | **163.0** (1067) |
| 128x128 w64x32 | 145.5 (1195) | 143.2 (1620) | 134.0 (433) | 136.9 (1271) |
| 128x128 w128x16 | 92.0 | 89.0 | 85.9 | 87.5 |
| 128x64 w64x32 | 132.3 | 131.3 | 121.3 | 128.5 |
| 256x128 w64x32 | 126.3 | 135.2 | 122.8 | 101.4 |
| 64x128 w64x32 | 132.8 | 132.8 | 122.6 | 131.2 |

**Decision.** Reject, and the reason is worth more than the experiment.
Counters on the two 128x128 geometries at (49152, 2304, 768), timed separately
from the counter run:

| geometry | SQ_INSTS_LDS | SQ_INSTS_MFMA | LDS/MFMA | SQ_LDS_BANK_CONFLICT | conflict/LDS | us |
|---|---:|---:|---:|---:|---:|---:|
| w32x64 | 1 524 096 | 663 552 | 2.30 | 7 354 368 | 4.83 | 1025.5 |
| w64x32 | 929 664 | 663 552 | 1.40 | 10 644 480 | 11.45 | 1183.2 |

The instruction model was right -- 39% fewer LDS instructions, close to the
predicted 33% -- and it does not matter. **Total bank-conflict cycles is what
tracks the time**, and the taller tile has 45% more of them, because the reads it
adds (A fragments) are the conflict-heavy ones and the reads it removes (B
fragments) are nearly conflict-free.

## Diagnostic experiment AC — a B tile width that changes the LDS bank pattern

**Hypothesis.** Element (k, n) of the B LDS tile sits at dword `k*BN/2 + n/2`, so
its bank is `(k*BN/2 + n/2) % 32`. At BN = 128 that is `(n/2) % 32`: every k in a
fragment lands in the same bank, which is a 4-way conflict for the four k a
fragment reads. At BN = 96 it is `(16k + n/2) % 32`, which alternates, halving the
conflict -- and every N this workload uses (768, 2304, 3072, 50304) is divisible
by 96 as well as by 128, so no edge kernel appears. Padding instead of
re-widening is not available: 128x128x32 at two stages is exactly 32 KB, half of
gfx942's 64 KB, so two workgroups per CU fit with zero bytes to spare, and
padding A's rows from 32 to 40 elements and B's from 128 to 136 costs 3 KB per
stage, which drops residency to one workgroup per CU.

**Predicted effect.** >= 10% from halving the dominant conflict.

**Measured effect.** 12-18% *slower* on every shape. TFLOP/s: 128x96 w32x48
gives 146.7 / 143.6 / 132.5 / 142.0 against the 128x128 w32x64 baseline's
167.9 / 169.9 / 156.6 / 163.0; 128x96 w64x48, 128x96 w32x96, 256x96 w64x48,
64x96 w32x48, 128x192 w32x48 and 128x192 w64x48 are all worse still
(91.2-145.2). A narrower B tile also cuts the MFMA work per warp (three n-MMAs
instead of four) and leaves 384 of the 512 threads in the B copy, which is
evidently the larger effect.

**Decision.** Reject. 128x128x32 with a 32x64 warp tile survives every geometry
axis tried in this journal.

## The remaining 3.17x, quantified

Every one of the fifteen Linear GEMMs now sits in a single band, 131-177
TFLOP/s, against PyTorch-ROCm's 367-612, and the cause is one resource. From the
counters above, per MFMA instruction the LDS pipe spends 2.30 issue cycles plus
11.08 conflict cycles, i.e. **13.4 cycles**. One MFMA is a 16x16x16 BF16
instruction, 16 cycles on one SIMD, and a CU has four SIMDs and *one* LDS unit,
so MFMA needs 4 cycles of CU throughput per instruction. **The LDS pipe is
oversubscribed 3.3x against the matrix cores**, and 170 TFLOP/s x 3.3 = 561,
which is where PyTorch-ROCm measures (483 TFLOP/s on the same shape).

The fix is a swizzled or padded LDS layout, and both are blocked in a specific,
citable way rather than merely unattempted:

* **Swizzle.** `multistage_mma` does take `swizzle_a`, and the A-fragment load
  applies it (`make_ldmatrix_swizzle`, `mma_op.load_a[swizzle_a_pattern]`). But
  its store helper drops it on AMD -- `_copy_tensor_to_sram` calls
  `copy_dram_to_sram_async[..., swizzle=swizzle]` on NVIDIA and
  `copy_dram_to_sram[thread_layout=thread_layout]`, with no swizzle at all, on
  AMD (`_multistage_gemm_gpu.mojo:314-331`). Enabling it would read a permutation
  that was never written, which is why `multistage_gemm_kernel` hardcodes
  `swizzle_a=is_nvidia_gpu()`. Fixing this means a modified copy of
  `multistage_mma`, not a parameter.
* **Padding.** A padded LDS layout needs no loader change at all, only a row
  stride, and the layout comes from the iterators `multistage_gemm_kernel`
  constructs -- so a wrapper kernel calling `multistage_mma` directly could supply
  padded ones. The LDS budget forbids it at the winning tile: two stages of
  128x128x32 are exactly 32 768 bytes (the profiler confirms `lds 32768`), two
  workgroups per CU is 65 536, and gfx942 has 65 536. Any padding costs the second
  resident workgroup, which experiment W measured as a 2-4x loss when BK = 64 did
  the same thing. Padding is only affordable under a smaller tile (128x64 padded
  is 29.7 KB for two stages), and 128x64 unpadded is already 26% slower, so the
  padding would have to win 1.35x just to break even.

Two smaller items also measured and not taken: the A-fragment conflict is a
function of BK alone (`(m*BK/2 + k/2) % 32` is `(m*16 + k/2) % 32` at BK = 32,
i.e. two distinct banks for sixteen rows), and BK can only be 32 or 64 here
because `num_k_mmas` must be even (`_multistage_gemm_gpu.mojo:385`) and 64 was
rejected by experiment W. And `k_group_size > 1`, which the tune dispatcher
exposes, cannot be selected at BK = 32 at all: it asserts
`num_k_mmas % (2 * k_group_size) == 0`.

## Review finding R8 — the all-ones gate's middle-of-K hole, closed

**Hypothesis.** The harness README records one hole: at K = 50304 the BF16 grid
spacing is 256, so any accumulator in [50048, 50304] passes the all-ones gate.
The hole is wider than that entry says -- at K = 49152 the spacing is 128, four
BLOCK_K tiles' worth of all-ones accumulation -- and the pattern gate cannot cover
it, because its nonzero terms are the first and last four K indices only. Global
split-K makes this a live risk rather than a theoretical one: its failure mode is
exactly a K range dropped or double-counted at a *slab boundary*, in the middle
of K.

**Predicted effect.** Splitting K into 4096-wide chunks and running one all-ones
pass per chunk, with B nonzero only on that chunk, must (a) pass on the shipped
code for all fifteen cases and (b) fail on a deliberately injected one-tile slab
drop that both existing gates pass.

**Measured effect.** Both. All fifteen cases pass `--chunk-check=1`. With slab 0
of the split-K kernel deliberately shortened by one BLOCK_K tile -- a 32-term
deficit out of 49152, which rounds back to the same BF16 value --
`attn_c_attn_wgrad` **passes the default all-ones gate and passes
`--pattern-check=1`**, and fails `--chunk-check=1`. A chunk's expected value is
its own length, whose BF16 spacing at 4096 is 16, well below one BLOCK_K tile,
and every K index belongs to exactly one chunk, so the whole range is inspected.

**Decision.** Accept. Run all three gates; the chunk gate is the only one that
covers the middle of a long K.

## End-to-end validation — 2026-07-25, permute gather and GEMM split-K

Same session, same seed, `bench_nanogpt_train.py --warmup 3 --iters 10
--print-loss`:

| backend | step median | tokens/s | 13-step loss |
|---|---:|---:|---:|
| PyTorch-ROCm | 156.715 ms | 313 640 | 10.621461 |
| eager Mojo, entry (409.34 recorded) | 409.34 ms | 120 077 | 10.631089 |
| eager Mojo, after Change 19 | 384.665 ms | 127 779 | - |
| eager Mojo, after Change 20 | **353.327 ms** | 139 112 | 10.621608 |

The two steps sum to 56.0 ms and the harnesses predicted 25.2 + 30.7 = 55.9.

Numerics, on the metric that is actually reliable: the **single-step** loss is
10.977283478 on the Mojo device against 10.977397919 on PyTorch-ROCm, a
difference of 1.14e-4, which is the pre-existing agreement between the two
backends and not a regression. (The 13-step loss has about 0.04 of run-to-run
spread, so its move from 10.6311 to 10.6216 is not evidence of anything by
itself.)

Re-profiled with `profile_nanogpt_train_aten.py --device mojo --output-dir
current_bench_train/mojo_v6` and compared against `current_bench_train/rocm`
(`comparison_v5`). Overall **2.61x -> 2.26x**, 203.6 ms of gap left:

| target | before | after | rocm | ratio before | ratio after |
|---|---:|---:|---:|---:|---:|
| Linear GEMM (backward) | 186.35 | **156.09** | 43.65 | 4.27x | **3.58x** |
| Linear GEMM (forward) | 72.26 | 72.36 | 24.03 | 3.01x | 3.01x |
| SDPA (backward) | 66.35 | **46.14** | 33.95 | 1.95x | **1.36x** |
| SDPA (forward) | 36.87 | **29.84** | 9.08 | 4.06x | **3.29x** |
| Copies / casts / layout | 26.01 | **16.84** | 11.44 | 2.27x | **1.47x** |
| whole step | 409.3 | 353.3 | 156.7 | 2.61x | 2.26x |

The SDPA and copy rows move because the permutes they were paying for are the
run-gather shapes; `data_movement_ops__permute_copy` is no longer among the top
kernels of any target. The Linear GEMM forward row is untouched by design: its
shapes are grid-filled already, so no slab count is selected for them.

Regression re-checks, all four, after both changes:

| check | recorded | now |
|---|---|---|
| `bench_linear_gemm` default | 257.59 ms, 3.596 | 226.837 ms, 3.166, 15/15 pass |
| `bench_linear_gemm --pattern-check=1` | pass | 226.731 ms, 3.165, 15/15 pass |
| `bench_linear_gemm --chunk-check=1` | (new gate) | 226.757 ms, 3.165, 15/15 pass |
| `bench_attention_bmm` default | 51.205 ms | 51.206 ms, 6/6 pass |
| `bench_attention_bmm --pattern-check=1` | pass | 50.993 ms, 6/6 pass |
| `bench_attention_bmm --causal=1` | ~50.4 ms | 50.460 ms, 6/6 pass |
| `bench_attention_softmax` | 9.98 / 9.34 ms/step | 10.014 / 9.356, 4/4 pass |
| `scripts/sdpa_correctness.py` | 25 pass, 3 known FAIL at 2.954/3.119/2.277 | 25 pass, the same 3 FAIL at 2.954/3.119/2.277 |

One cost worth recording: `matmul_ops.mojo` now instantiates the multistage core a
second time with `c_type = float32`, which lengthens both the `__mojocache__`
rebuild and every `mojo build` of a harness that imports `matmul_ops` from about
two seconds to a couple of minutes. `bench_permute_copy`, which imports only
`data_movement_ops`, still builds in seconds.

## Test-visible state after this work — 2026-07-25

`tests/test_eager_kernels.py` and `tests/test_mojo_device.py` run serially: 577
passed, 8 failed, 99 skipped. None of the eight is from this work, and the check
is direct rather than an appeal to plausibility:

* Four `test_fast_log_softmax_*` and three
  `test_fast_cross_entropy_training_uses_direct_nll_kernel[*]` failures are one
  defect, a **NaN out of the fast log-softmax forward** (the cross-entropy tests
  consume it). Reproduced standalone at `(65, 32)` float32: the forward and the
  gradient are both NaN, and `TorchMojoTensor._materialize_contiguous` is called
  **zero** times and no GEMM is issued at all, so neither the run gather nor
  split-K can be involved. The earlier journal entry recorded four such failures
  as pre-existing at commit `4cd5f30`; the count is seven and the symptom is a NaN,
  which is worth someone's attention on its own. The production path is unaffected
  at the nanoGPT shape -- `log_softmax_rows_block_bfloat16_1024` is in the profile
  and the single-step loss agrees with PyTorch-ROCm to 1.1e-4.
* `test_bf16_v3_source_dependency_and_kernel_contract` asserts a hardcoded list of
  H100 BF16 GEMM source filenames in `aten_fast._BF16_SOURCE_PATHS`; this work
  touched no file in that list.

Note also that running these two files with `-n 8` produces 39 failures rather
than 8, including crashed workers: eight pytest workers contending for one GPU is
not a valid configuration for the eager-device tests. Run them serially.

## Defect analysis D7 — log-softmax forward returned all-NaN on most shapes

**Hypothesis.** The last agent's report flagged "a live NaN in the fast log-softmax
forward (7 failing tests)" and judged it pre-existing. A NaN in `aten::_log_softmax`
is worth its own entry regardless of provenance, because that op backs
`F.cross_entropy`: every training loss in the repository goes through it.

The online single-read kernel seeds its running max with `Float32.MIN`, which in
Mojo is `-inf`, not the lowest finite float. Any thread whose loop body never
executes -- every thread with `tid >= n_vec`, and there are many whenever a row
holds fewer 16-byte vectors than the block has threads -- keeps that sentinel
alongside a zero running sum. The collapse then computes `s * exp(m - m_t)`, which
with both terms `-inf` is `0 * exp(nan) = nan`; the block sum turns one idle
thread's nan into a nan `log_denom`, so the entire row is nan.

**Predicted effect.** NaN exactly when `n_vec < threads`, where
`V = 16 / size_of[dtype]`, `n_vec ~ cols / V`, and `threads` is 1024 when
`cols * size_of[dtype] > LSM_BIG_ROW_BYTES = 25_000` and 256 otherwise. So FP32 at
cols = 1024 gives `n_vec = 256 = threads` and must be the *only* clean case among
the small shapes, while BF16 at the same cols gives `n_vec = 128 < 256` and must be
nan. The nanoGPT loss must nevertheless be correct, because autocast runs cross
entropy in FP32 at cols = 50304, which takes the 1024-thread branch with
`n_vec = 12576`.

**Measured effect.** Every prediction held, on this branch and on `main` at
d93fe25 with identical NaN counts, so the defect is confirmed pre-existing and not
from this work:

| dtype | (rows, cols) | before | after |
|---|---|---|---|
| f32 | (3, 5) | nan 15/15 | ok, 2.4e-07 |
| f32 | (128, 128) | nan 16384/16384 | ok, 9.5e-07 |
| f32 | (4096, 1024) | ok | ok |
| bf16 | (4096, 1024) | nan 4194304/4194304 | ok, 0.0312 |
| bf16 | (49152, 1024) | nan 50331648/50331648 | ok, 0.0312 |
| f16 | (128, 128) | nan 16384/16384 | ok, 0.0038 |

`softmax` was unaffected in every case, which is what isolates the fault to this
kernel's online rescale rather than to the shared reduction machinery.

**Decision.** Fix by seeding the running max with `Float32.MIN_FINITE`, so an idle
thread's contribution is `0 * exp(0) = 0`, the identity the (max, sum) monoid needs.
`-Float32.MAX` is *not* usable: `Float32.MAX` is `inf`, so its negation is `-inf`
again. A genuine `-inf` input still behaves, since `exp(-inf - MIN_FINITE)` is 0.
This is the same sentinel and the same reasoning as the SDPA causal softmax fix in
`nn_ops.mojo`, which had hit this defect class independently.

Audited the other eight `Float32.MIN` / `min_or_neg_inf` sites in the eager
kernels. Seven are plain max-reduction identities, where `-inf` is correct and no
rescale ever divides by it, and the eighth is `_attn_decode`'s per-thread max,
which reaches the block reduction as a max identity and is likewise safe. Only the
online-rescale form breaks, and only this one kernel used it with an infinite seed.

Result: `tests/test_eager_kernels.py` goes to 379 passed, 15 skipped, with one
failure remaining -- `test_bf16_v3_source_dependency_and_kernel_contract`, a source
list-ordering assertion that also fails on `main`, verified in a clean worktree.
The nanoGPT step is unchanged at 354.22 ms and the single-step loss is still
bit-identical at 10.977283477783203.

### The merge of `main` at 547ad6d proposed the other fix, and it is worse twice

`main` reached this defect independently (#319) and fixed it the other way: keep
the `-inf` seed, and guard every `exp(a - b)` with an `a == b` select, so the
equal case contributes the true factor `exp(0) == 1` and `exp(-inf - -inf)` is
never evaluated. Both fixes are complete on their own, so the merge had to pick
one rather than carry both. Measured, guards against this entry's finite seed,
`us` per launch, best of 9 reps, the two builds interleaved case by case:

| case | guards (#319) | `MIN_FINITE` | guards cost |
|---|---:|---:|---:|
| bf16 12288x50304, 1024-thread arm | 1034.2 | 890.0 | +16.2% |
| bf16 49152x1024, 256-thread arm, half the block idle | 149.4 | 133.2 | +12.2% |
| f32 12288x50304, the autocast cross-entropy row | 1686.9 | 1660.9 | +1.6% |
| f32 16384x4096 | 168.8 | 163.8 | +3.1% |
| f32 128x128, dispatch-bound control | 16.0 | 16.0 | none |

The guards add two vector compare-selects per trip and change no `exp` count, so
the asymmetry is the point: bf16 packs 8 fp32 lanes per 16-byte load against
f32's 4, twice the lane work per byte moved with half the memory time to hide it
behind. Against Change 40's roofline probes scaled to 12288 rows -- 2-stream copy
~614 us, 3-stream add ~986 us -- the finite seed at 890 is still under the
3-stream add while guards at 1034 is above it, i.e. the guarded form has left the
bandwidth-bound regime that made this kernel's cost predictable.

**The schedule lottery does not explain it.** Mojo compilation here is
deterministic (same source, byte-identical `.so`), so repeated rounds re-measure
one binary and cannot sample the scheduler. Bit-neutral twins do: commuting the
two addends on one side and swapping the two independent guard computations on
the other leaves semantics identical while giving the scheduler a different input
order. Each twin landed on its own family within 0.5% (guards 1034-1042, finite
seed 890-895), so schedule luck is under 1% here against a 12-16% gap.

**And the guards are less faithful to torch.** The `x == new_m` select turns
`exp(inf - inf)` into 1.0, so a `+inf` element contributes a finite 1 to the sum
instead of poisoning it, `log_denom` becomes `log(1) = 0`, and the row's finite
entries come out `-inf`:

| input `[1, inf, 2, 3]` | output |
|---|---|
| torch | `[nan, nan, nan, nan]` |
| guards (#319) | `[-inf, nan, -inf, -inf]` |
| `MIN_FINITE` | `[nan, nan, nan, nan]` |

The pre-#319 unguarded code produced torch's answer here, so #319 traded a `+inf`
divergence for the idle-thread fix, and no test covered it. The finite seed needs
no guard at all, because it is never an operand of a subtraction that can reach
`inf - inf`. Kept this entry's fix, added
`test_fast_log_softmax_positive_inf_rows_match_cpu` to pin the `+inf` row, and
kept both of `main`'s new tests, which pass unchanged on the finite seed (120
log-softmax tests pass). `main` still carries the guards, so #319 wants
revisiting there; until it is, this line will conflict on every merge.

**Defect D12, found by that new test and left for its own change.** The test was
written against `mojo_device`, so it also ran the kernel's CPU branch, which fails
the same way for an unrelated reason:

| `[1, inf, 2, 3]`, CPU-backed `mojo:1` | mojo | torch |
|---|---|---|
| `log_softmax` | `[-inf, nan, -inf, -inf]` | `[nan, nan, nan, nan]` |
| `softmax` | `[0, nan, 0, 0]` | `[nan, nan, nan, nan]` |

The CPU branch is a plain three-pass max/denominator/store with no guard anywhere,
so nothing selects the nan away; the arithmetic implies `denom` should be nan and
it is not, which means the CPU `exp` does not propagate a nan argument (a clamp
before the polynomial would explain a nan folding to `exp(0) = 1`, and the
observed `log_denom = 0` is exactly that). Unverified beyond these two ops, but
both share the shape, so treat it as the CPU `exp` and not as two bugs. It
predates both fixes and both branches. The regression test is scoped to
`mojo_gpu` until this is fixed rather than left failing.

## Review finding R8 — split-K preempted the route tuned for underfilled grids

**Hypothesis.** `_splitk_parts` bounds the slab count from above (`tiles * 2 *
parts <= 4 * cus`) but never from below. Its divisibility and slab-depth
conditions can stop the doubling early, leaving a split whose *product*
`tiles * parts` is a fraction of the CU count -- and the clause admitting it sits
ahead of the 32x32 in-workgroup-partition route, which Change 14 tuned for exactly
that underfilled regime. Change 20 measured `k = 49152` only, while the admission
gate is `k >= 8192`, so the band in between has no measurement behind it.

**Predicted effect.** A regression somewhere in `k` in [8192, 32768), where the
2048-per-slab floor caps `parts` at 4 and then 8.

**Measured effect.** Confirmed, and the first attempt to measure it was wrong in a
way worth recording. Running the whole case list in one process gave
`(512, 1024, 8192)` as a 49% *win*; running one case per process gave the same
shape as a 14% *loss*. The harness allocates and frees per case, so a later case
reads low when an earlier one has already warmed the device. Every number below is
one case per process, against a build of the pre-split-K `matmul_ops.mojo`:

| tiles*parts | (m, n, k) | pre us | with split-K | change |
|---:|---|---:|---:|---:|
| 24 | 128, 768, 8192 | 112.79 | 148.48 | +32.1% |
| 72 | 768, 768, 8256 | 159.76 | 272.60 | +70.6% |
| 128 | 512, 1024, 8192 | 129.86 | 147.67 | +13.7% |
| 144 | 768, 768, 8192 | 134.09 | 152.88 | +14.0% |
| 288 | 768, 768, 16384 | 253.65 | 161.27 | -36.4% |
| 384 | 1024, 1536, 12288 | 534.07 | 367.06 | -31.3% |
| 512 | 4096, 1024, 8192 | 701.82 | 487.35 | -30.6% |
| 512 | 2048, 2048, 8192 | 515.06 | 484.38 | -6.0% |
| 576 | 768, 768, 49152 | 742.33 | 380.35 | -48.8% |

Neither the slab count nor the tile count separates these on its own: `parts = 2`
loses at 72 and wins at 512, and an identical 36-tile grid loses at `k = 8192` and
wins at `k = 16384`. Their *product* separates all nine cleanly. Every loss is
below 0.5 workgroups per CU and every win is at or above 0.95.

**Decision.** Accept and fix with a floor of three quarters of one workgroup per
CU: `_splitk_parts` returns 1 -- no split -- when `tiles * parts * 4 < cus * 3`.
All four regressions return to within +-0.6% of the pre-split-K route and all five
wins are unchanged (-6.0% to -48.8%). The nanoGPT table is untouched at 226.9 ms,
ratio 3.167, under all three gates, because every production weight-gradient shape
that used split-K keeps it: (2304,768,49152) and (3072,768,49152) at 432 and 576,
(768,3072,49152) at 576, (768,768,49152) at 576. The step is 353.93 ms and the
single-step loss is still bit-identical at 10.977283477783203.

Worth stating for the next reader: the harness runs its cases in one process, so
any before/after comparison of individual cases must pass `--case=` and run one per
process. The contaminated ordering does not affect the whole-table totals, which
run the same cases in the same order on both sides.

# nanoGPT NT (forward Linear) GEMM, MI300X — 2026-07-26

Third pass over `harness/nanogpt_train/bench_linear_gemm.mojo`, this time one
variant only: **NT**, `C(m,n) = A(m,k) @ B(n,k)^T`, the `fwd` role, five shapes
all with `m = 49152`. Entry state, re-verified: NT 72.062 ms/step against
PyTorch-ROCm's 25.240, **ratio 2.855**, at 152-175 TFLOP/s against ROCm's
367-543. NN 3.079 and TN 3.601 are the regression check and must not move.

Protocol unchanged: >= 25 warmups and >= 100 individually synchronized
iterations, timing never from a `--pmc` run, every extent a runtime value.

## Measurement note that changes how the numbers below must be read

`bench_linear_gemm`'s whole-table run and its `--case=` run disagree on
**`mlp_c_proj_fwd` by 24%** once the new kernel is in place: 448 us in the table,
525-586 us alone. This is not clock ramp -- the isolated number is stable at
583.2-585.5 us from 100 to 3000 iterations -- and it is not the operand address:
inserting a live dummy allocation of 1, 2, 3, 5, 8, 16 or 64 MB ahead of the
operands moves it by less than 0.5%. It is the *case order*. Running one other
case first, with a custom targets CSV, isolates which:

| predecessor | its total allocation | `mlp_c_proj_fwd` |
|---|---:|---:|
| none | - | 585.53 us |
| `attn_c_attn_fwd` | 306 MB | **446.21** |
| `attn_c_proj_fwd` | 152 MB | **446.13** |
| `mlp_c_fc_fwd` | 382 MB | 585.58 |
| `mlp_c_fc_dgrad` | 382 MB | 584.79 |
| `mlp_c_fc_wgrad` | 684 MB | 584.66 |
| `lm_head_fwd` | 5.1 GB | 584.22 |
| `mlp_c_proj_fwd` itself | 382 MB | 584.64 |

The pre-change code has no such sensitivity: its per-case numbers equal its
table numbers to within 0.3% on all five shapes. So the new kernel has a
warm-allocator fast path on one shape and the old one did not, and the two
measurements of the same work differ. **Both are reported below.** The whole-table
figure is the harness's own acceptance number and is what the 2.855 baseline was
also measured under; the per-case figure is the conservative one and is what the
brief's protocol prescribes for individual shapes.

A second, unrelated ordering artifact: a *multi-case* CSV comparison of two
binaries is meaningless. In one such run `dec512_n3072_k768` -- a shape that
cannot reach any new code, since the new route needs `m >= 1024` -- moved from
49.45 to 137.53 us. Every regression check below is therefore one case per
process.

## Change 21 — a hand-written NT MFMA kernel

**Hypothesis.** The recorded barrier is that MAX's `multistage_gemm_kernel`
cannot make both halves of the data path wide for this layout: `(N,K)` gives
k-contiguous LDS fragments and 64-byte global rows, `(K,N)` gives 256-byte rows
and four 2-byte LDS reads per B fragment, and counters on the best configuration
measured 2.30 LDS instructions plus 11.08 conflict cycles per MFMA against 4
cycles of per-CU MFMA throughput. But NT has **both operands already k-major in
memory**, and an MFMA operand wants exactly four elements adjacent along k. A
k-major LDS tile for both therefore makes the global rows and the fragment reads
wide at the same time, with nothing transposed anywhere. Write that kernel.

**Predicted effect.** One `ds_read_b64` per fragment instead of four scalar
reads, zero bank conflicts, and a rate limited by something other than the LDS
pipe.

**Measured effect.** `_nt_mfma_kernel`, and the geometry search that fixed its
shape. Every number is TFLOP/s, 25 warmups and 100 synchronized iterations, one
shape per process, on-device equality check over every output element:

| geometry | 49152x2304x768 | 49152x768x3072 | 49152x50304x768 |
|---|---:|---:|---:|
| 128x128x32 w64x64 2 padded stages | 202 | 186 | 200 |
| 128x128x32 w64x64 1 padded stage | 280 | 268 | 292 |
| 256x256x32 w128x128 1 stage (4 waves) | 219 | - | 194 |
| 256x256x32 w128x64 1 stage (8 waves) | 331 | 322 | 336 |
| 256x256x32 w64x64 1 padded stage (16 waves) | 379 | 355 | 404 |
| **256x256x32 w64x64 2 swizzled stages** | **434** | **397** | **443** |
| 256x384x32 w64x192 1 stage | 91 | - | 104 |
| 384x256x32 w128x128 1 stage | 32 | - | 32 |

Three structural findings, in the order they were found:

1. **Occupancy beats arithmetic intensity here, and the binding resource is
   VGPRs.** A 64x64 warp tile issues twice the LDS traffic per unit of output
   area that a 128x64 one does, and it wins by 15% and by 73% against 128x128,
   because sixteen accumulators of sixteen registers is 64 VGPRs and that is the
   most that still fits the 128 registers a wave gets at four waves per SIMD.
   The two geometries above 256x256 collapse to 32-104 TFLOP/s: their
   accumulators alone exceed the budget and spill.
2. **Reading every fragment of the k tile in one burst, before any MFMA, is
   worth 12-18%** (attn_c_attn 334 -> 379 at one padded stage). Per k step the
   MFMAs depend on that step's own reads, so the per-step form makes the tile a
   chain of four LDS waits; the burst costs `(MT+NTL)*BK/2` = 32 VGPRs and
   removes them. Splitting it the other way -- half the reads, then half the
   MFMAs, then the rest -- measures 4% worse, and `llvm.amdgcn.iglp.opt` 0, 1
   and 2 measure within 0.3% of nothing.
3. **The pipeline order is measured, and the textbook one is wrong.** With the
   MFMAs placed *after* the LDS refill (so they fill the gap between the refill
   and the barrier that publishes it) the kernel runs 379 TFLOP/s; moving them
   *before* the refill, which is what "cover the global latency with matrix
   work" would suggest, gives 301 -- 25% slower, because it leaves only the
   vmcnt wait and the LDS stores between the two barriers and the whole
   workgroup waits there in lockstep.

**Decision.** Accept the geometry: 256x256x32, 64x64 warp tile, sixteen wave64,
two LDS stages.

## Change 22 — an XOR-swizzled LDS layout, so two stages fit

**Hypothesis.** LDS rows padded to `BK + 4` give conflict-free `ds_read_b64`
fragments (the derivation in `flash_attention_fwd_kernels.mojo`), but two stages
of a padded 256x256x32 tile are 73728 bytes and gfx942 has 65536, so the winning
tile could only be single-buffered -- two barriers per k tile. Permuting the
four-element chunks within a row by XOR instead of padding costs zero bytes, and
two unpadded stages are exactly 65536.

**Predicted effect.** One barrier per k tile instead of two, at the same tile.

**Measured effect.** The permutation is `chunk ^ ((row >> s) & (BK/4 - 1))`, with
`s = 1` when a row is 16 dwords and `s = 0` when it is 32 -- a row of 16 dwords
already splits the 32 banks by row parity, so the permutation only has to be
injective within a parity class, while a row of 32 dwords contributes nothing and
it must be injective over all 16 rows of an LDS cycle. Fragment reads and tile
writes are both conflict-free by construction; the write becomes two
`ds_write_b64` instead of one `ds_write_b128`, which is the same bytes and the
same LDS cycles.

At one stage the swizzle is a small loss (379 -> 349 TFLOP/s on attn_c_attn: the
extra XOR and the doubled write instruction count are not free). At two stages,
which is what it buys, attn_c_attn 379 -> 434, mlp_c_proj 355 -> 397, lm_head
404 -> 443.

**Decision.** Accept.

## Change 23 — XCD band remap of the output-tile grid

**Hypothesis.** gfx942 dispatches workgroup `i` to XCD `i % xcds`, and each XCD
has its own L2. With one workgroup resident per CU and the natural row-major tile
order, the tiles that share an A row block land on different XCDs and each
private L2 fetches its own copy. Giving every XCD a contiguous band of the
natural order should fix that. MAX's own `_xcd_wgm_swizzle`
(`amd_4wave_matmul.mojo:89`) is the shape of it.

**Predicted effect.** Most on the shapes with the fewest column tiles, where the
sharing group is smallest.

**Measured effect.** Bands, us, one case per process:

| bands | attn_c_attn | attn_c_proj | mlp_c_fc | mlp_c_proj | lm_head |
|---:|---:|---:|---:|---:|---:|
| 1 | 401.1 | 144.7 | 554.8 | 585.4 | 8564 |
| 2 | 383.1 | 142.8 | 544.1 | 558.2 | 8601 |
| 3 | 402.6 | 144.9 | 553.6 | 580.2 | 8780 |
| 4 | **378.4** | 139.6 | **524.4** | 536.5 | 8678 |
| 6 | 388.2 | 145.4 | 529.2 | 581.8 | 8768 |
| 8 | 408.1 | **138.2** | 532.1 | **524.7** | 8600 |

Only band counts that divide 8 help, which is the evidence that the mechanism is
the XCD mapping and not merely traversal order. Adding MAX's second stage, a
WGM row-block group, moves nothing consistently (best combination 1.099 weighted
against 1.103 for the remap alone). Band count 4 is 0.8% better weighted than 8
and I cannot explain why, so the shipped value is the XCD count itself, derived
from the runtime CU count as `cus // 38` (38 CUs per CDNA3 chiplet); the map is a
bijection for any value >= 1, so a wrong count costs locality and nothing else.
Weighted NT, one case per process: 1.141 -> 1.106.

**Decision.** Accept at `xcds`, and record that 4 measured better without a
mechanism.

## Change 24 — decline when the output grid does not cover the device

**Hypothesis.** One workgroup of this tile is resident per CU -- sixteen wave64
and the whole LDS budget -- so a grid that does not cover the device leaves CUs
idle for the entire k loop with no second workgroup to fill them. The five
nanoGPT NT shapes all have `m = 49152` and never expose this; other shapes in the
same dispatch branch do.

**Predicted effect.** Shapes whose grid is a fraction of 304 workgroups are
slower than the route they replaced.

**Measured effect.** Confirmed, and it was a shipped regression until gated. One
case per process, pre-change against the ungated new route:

| case (transpose_b) | tiles | pre-change | ungated | change |
|---|---:|---:|---:|---:|
| m4096 n768 k3072 | 48 | 189.7 us | 226.7 | **+19.5%** |
| m2048 n768 k3072 | 24 | 154.4 | 230.1 | **+49.0%** |
| m1024 n2304 k3072 | 36 | 176.6 | 231.9 | **+31.3%** |

The gate takes the largest tile whose grid covers every CU and declines below
that, mirroring `_amd_bf16_large_m`'s existing rule: 256x256 when
`ceildiv(m,256)*ceildiv(n,256) >= cus`, else 128x128 when the 128 grid does, else
the caller keeps its route. After it, one case per process:

| case | pre-change | after | change |
|---|---:|---:|---:|
| m4096 n2304 k768 | 124.10 | 55.12 | -55.6% |
| m4096 n3072 k768 | 159.97 | 84.87 | -46.9% |
| m4096 n50257 k768 | 2087.68 | 826.75 | -60.4% |
| m8192 n768 k768 | 115.11 | 51.25 | -55.5% |
| m8192 n2304 k768 | 211.24 | 93.99 | -55.5% |
| m16384 n768 k3072 | 545.32 | 304.05 | -44.2% |
| m4096 n4096 k1024 | 260.61 | 137.46 | -47.3% |
| m4096 n768 k3072 | 189.18 | 189.05 | -0.1% |
| m2048 n768 k3072 | 153.96 | 153.67 | -0.2% |
| m1024 n2304 k3072 | 175.12 | 174.45 | -0.4% |
| m1024 n768 k3072 | 87.30 | 87.17 | -0.1% |
| m1536 n768 k3072 | 108.28 | 108.39 | +0.1% |
| m4096 n1000 k768 | 203.80 | 203.77 | 0.0% |
| m6144 n768 k768 | 101.27 | 102.38 | +1.1% |
| m4096 n768 k768 | 69.18 | 69.28 | +0.1% |
| GPT-2 decode, m512, n 768/2304/3072/50257 | 33.08/40.35/49.86/659.66 | 33.14/40.29/50.29/659.73 | +-0.9% |

`m4096 n1000 k768` is the cost of the gate rather than a win: its 128-tile grid
is 256 workgroups against 304 CUs, so it declines, where the ungated route was
62% faster. The gate is a runtime comparison of a runtime tile count against the
runtime CU count and no model dimension is compiled in; a threshold tuned to
recover that one point is not justified by one point.

**Decision.** Accept.

## Defect analysis D8 — the same gfx942 MFMA-read miscompile, minimally reproduced

**Hypothesis.** Journal entry D6 recorded a gfx942 code-generation defect in
MAX's two-stage `multistage_gemm_kernel` for transposed B: one MFMA k step's
contribution vanishes from *part* of the accumulator ("two of the four
accumulator elements per lane"), which warp and which block varies, three
pipeline stages make it disappear, and no structural predicate separates the
failing geometries. That entry concluded "do not chase MAX's code generator".
The new kernel hit the same symptom, in code this project owns, which makes it
reproducible and fixable rather than merely avoidable.

**Predicted effect.** If it is the same defect, it must reproduce with every edge
guard trivially true -- so that only the guard's *branch* differs -- and be
sensitive to unrelated schedule changes.

**Measured effect.** Exactly that. At `(m, n, k) = (128, 128, 768)`, where the
128x128x32 tile divides every extent and every guard is therefore always true,
the instantiation with the store guard present returns 760 instead of 768 for
rows 26, 27, 30, 31 of every 32-column block of the first accumulator: short by
exactly one k step, in registers 14 and 15 -- the last two of the sixteen -- of
accumulator (0,0) only. Splitting the guard shows the load mask is innocent and
the store mask alone reproduces it. `s_barrier` does **not** fix it;
`llvm.amdgcn.sched.barrier` does, which is what identifies it as scheduling
rather than synchronization. It also moved with things that should be
irrelevant: single-buffered geometries failed where double-buffered ones passed,
`BK = 64` failed where `BK = 32` passed, and a 64x128 warp tile failed where
128x64 passed -- on shapes where all of them are exact.

**Decision.** Fix it structurally rather than by avoiding geometries. The
accumulators are read and rounded to the output dtype **once, unconditionally,
in straight-line code**, with a `sched.barrier` on each side of that block, and
only then do the guarded stores run -- reading plain VGPRs, with no MFMA
destination anywhere near a branch. After that, twelve geometries (BM 64-256,
BN 64-256, BK 32 and 64, warp tiles 64x64 to 128x128, one and two stages, padded
and swizzled) x eleven shapes (including 1x1x32, 127x128x768, 128x127x768,
200x100x40, 129x129x776, 513x1023x1544, 257x50304x776) x two fill patterns
x forced-masked and unmasked instantiations all pass, where before the fix
sixteen of those combinations failed. This is worth more than the fix: the
defect is not confined to MAX's kernel or to transposed B, it is a hazard on any
read of an MFMA destination that the scheduler can place too close to the MFMA,
and the mitigation is cheap and local.

## Diagnostic experiment AD — what binds the NT kernel at 434-443 TFLOP/s

**Hypothesis.** With the geometry settled, the remaining gap should be
attributable rather than guessed.

**Measured effect.** `rocprofv3` on the shipped kernel at 49152x2304x768 reports
`vgpr 128, sgpr 112, lds 36864, block 1024, scratch 0`, i.e. four waves per SIMD
and one workgroup per CU, which is the maximum the 128-register budget allows.
`rocm-smi` during the run reports a sustained 1395 MHz, so the achievable BF16
peak is 304 x 2048 x 1.395e9 = **868 TFLOP/s** -- consistent with the 810
TFLOP/s MFMA-only microbenchmark recorded for the flash-attention work, and the
reason 443 TFLOP/s is 51% of achievable rather than 34% of the 1307 nominal.

Per k tile per CU, counted from the geometry:

    MFMA        256 instructions / 4 SIMDs x 32 cycles   2048 cycles
    LDS reads   256 ds_read_b64 x 4 cycles               1024
    LDS writes   64 ds_write_b64 x 4 cycles               256
    VMEM        512 x 64-byte requests                   ~512
                                                    --------
    sum                                                  3840

and the measured time is 3995 cycles per k tile (lm_head, 2985 k tiles per CU at
1395 MHz). **The stages add; almost nothing overlaps** -- the same profile the
flash-attention work found, and the reason `iglp.opt`, the read/MFMA split and
the "cover the latency" reordering all did nothing or hurt. The LDS read traffic
is irreducible at this warp tile: each wave reads its own 64x32 slab of A and of
B once per k tile, and the four wave-columns re-read the same slab, so the tile
is read four times over. Halving it needs a wider warp tile, which needs more
accumulator registers, which costs the fourth wave per SIMD -- and that trade
measures 15-73% worse.

The one shape that is not near the 434-443 band is `mlp_c_proj_fwd`
(49152x768x3072, 442 TFLOP/s in the table and 397-441 alone). It is not the
kernel: at the same k and a wider n the same kernel reaches **492 TFLOP/s**
(49152x3072x3072) and 416 (49152x1536x3072), and shrinking m at n = 768 tracks
grid fill closely -- m 8192 / 16384 / 32768 / 49152 give 157 / 300 / 318 / 397
against a fill model of 0.32 / 0.63 / 0.63 / 0.95. Three column tiles is simply
not enough concurrency for this tile.

## Where NT ended, and what it would take to close the rest

| case | rocm us | before | after (per case) | after (table) | after TFLOP/s |
|---|---:|---:|---:|---:|---:|
| attn_c_attn_fwd | 382.56 | 1047.4 | 408.2 (1.067) | 400.9 (1.048) | 426 |
| attn_c_proj_fwd | 158.05 | 380.3 | 137.9 (**0.873**) | 145.0 (0.917) | 420 |
| mlp_c_fc_fwd | 498.14 | 1384.8 | 534.6 (1.073) | 516.0 (1.036) | 434 |
| mlp_c_proj_fwd | 426.62 | 1376.1 | 525.1 (1.231) | 453.6 (1.063) | 442 |
| lm_head_fwd | 7655.58 | 21768.2 | 8643.0 (1.129) | 8560.1 (1.118) | 439 |

**NT 2.855 -> 1.060** on the harness's own per-variant table (1.064 with
`--pattern-check=1`, 1.070 with `--chunk-check=1`), and **2.854 -> 1.106**
measured one case per process. Fifteen of fifteen cases pass all three gates.
The 1.03 bar is not met either way.

Regression checks, all after the change:

| check | recorded | now |
|---|---|---|
| `bench_linear_gemm` NN | 3.079 | 3.078 / 3.079 / 3.079 |
| `bench_linear_gemm` TN | 3.601 | 3.599 / 3.586 / 3.591 |
| whole table | 227.09 ms, 3.170 | 181.68 ms, 2.536, 15/15 pass |
| `bench_attention_bmm` default | 51.206 ms | 51.236 ms, 6/6 pass |
| `bench_attention_bmm --pattern-check=1` | 50.993 ms | 51.040 ms, 6/6 pass |
| `bench_attention_bmm --causal=1` | 50.460 ms | 50.461 ms, 6/6 pass |
| GPT-2 decode NT, m512, four n | - | +-0.9%, one case per process |
| GPT-2 prefill NT, m4096, five n | - | -60.4% to +0.1% |

What is left, in the order the evidence supports:

* **The LDS read traffic, 1024 of the 3995 cycles.** Irreducible at a 64x64
  warp tile without giving up the fourth wave per SIMD. The way out is not a
  wider warp tile but fewer accumulator registers per unit of output area, which
  on CDNA3 means nothing available -- 16x16x16 has the same accumulators per
  output element as 32x32x8.
* **The VMEM request count, ~512 of the 3995 cycles.** At `BK = 32` a tile row is
  64 bytes, exactly one sector, so 32768 bytes per k tile is 512 requests and
  that is a floor. Loading 64 k elements per row into registers and filling BOTH
  double-buffer stages from one load would make the requests 128 bytes and halve
  the count, at the same LDS budget and the same barrier count; not attempted.
* **Narrow n.** `mlp_c_proj_fwd`'s three column tiles is the one shape-level
  deficiency, and the same kernel reaches 492 TFLOP/s when n is wide. A
  narrow-n regime -- a taller, narrower tile, or a global split-K to multiply the
  grid -- is the obvious next move and is not attempted here.
* **The measurement disagreement** on that same shape, 24% between the table and
  the isolated run, is unexplained and worth resolving before any further tuning
  of it: it is large enough to swamp the effects being tuned.

## Diagnostic experiment AE — MAX's 4-wave and ping-pong AMD matmuls, corrected

**Hypothesis.** Diagnostic experiment Y recorded these as unreachable partly
because "their BF16 MMA shape is 16x16x32, which asserts `_cdna_4_or_newer()` on
gfx942". That reason is wrong as stated: `mma_shape` is a config *field*
(`amd_4wave_matmul.mojo:159`), `validate_config` (`:718-765`) contains no assert
pinning it, and there is no `is_gfx95` or `_cdna_4_or_newer` gate anywhere in
`amd_4wave_matmul.mojo`, `amd_ping_pong_matmul.mojo`,
`amd_4wave_split_k_matmul.mojo`, `warp_spec_matmul.mojo` or their helpers.

**Measured effect.** Unreachable anyway, for four independent reasons, and the
LDS budget is not among them (128x128x32 bf16 is 32 KB and satisfies every
assert in `validate_config`):

1. **Compile-time N and K are mandatory, and it is a hard error.**
   `amd_4wave_matmul.mojo:826-835` reads `comptime N = static_shape[0]` and
   `comptime K = static_shape[1]`; a dynamic dim is `-1`, and
   `comptime assert K_per_split % (2 * BK) == 0` then fires because Mojo's `Int`
   has Python floor-mod semantics and `-1 % 256 == 255`. Same in ping-pong
   (`:336-337`), split-K (`:250-257`) and `amd_matmul.mojo:188`;
   `warp_spec_matmul` takes M, N, K as launcher *parameters* (`:572-574`) and
   unrolls the k loop at compile time (`:436`, `:538`).
2. **A dynamic leading stride is silently wrong.** `TileLoaderLDS` takes
   `stride` as a compile-time struct parameter (`amd_tile_io.mojo:1223`) and
   folds it into addresses (`:1479-1480`, `:1499`, `:1512-1513`); its nine
   `comptime assert`s (`:1363-1397`) never validate it. This confirms the
   earlier finding for `RegTileLoader` (`:2585`) and extends it.
3. **`mma_shape` is settable but no CDNA3 shape is legal.** The producer writes
   LDS lane-linearly in chunks of `WARP_SIZE * load_width` with
   `subtile_cols = 32` hardcoded (`amd_tile_io.mojo:1271`), and the consumer
   reads with `lds_row_stride = MMA_K` (`:936-940`) and chunk stride
   `MMA_M * MMA_K` (`:994`). Correctness therefore requires
   `MMA_M * MMA_K == WARP_SIZE * load_width == 512` and `MMA_M == 16`. CDNA3's
   bf16 MFMAs give 4 elements per lane: 16x16x16 and 32x32x8 both yield 256.
   CDNA4's 16x16x32 yields 512. The fp8 path escapes through a dedicated
   split-LDS branch (`:920-923`); there is none for an operand *smaller* than 16
   bytes, which is exactly the CDNA3 bf16 case. Setting 16x16x16 compiles and
   produces garbage.
4. **`load_to_lds` is unavoidable on that path** (`:1483`, `:1493`, `:1516`,
   `:1523`, no register-bounce fallback), and experiments R and T recorded that
   it cannot lower for dynamic layouts on this pin.

Modular says the same thing in their own build file
(`max/kernels/test/gpu/linalg/BUILD.bazel:133-141`, "4-wave matmul kernels are
MI355X-specific"), and their dispatcher gates the whole family on
`ctx.default_device_info == MI355X` (`matmul/gpu/__init__.mojo:801-805`). No
vendor library is on any of these call paths.

**Decision.** Correct the record: experiment Y's verdict stands, its stated
reason does not. The binding reasons are compile-time N/K, the silent
static-stride bake, and an LDS fragment geometry that no CDNA3 bf16 MFMA shape
satisfies.

# nanoGPT TN (weight-gradient Linear) GEMM, MI300X — 2026-07-26

Fourth pass over `harness/nanogpt_train/bench_linear_gemm.mojo`, one variant:
**TN**, `C(m,n) = A(m,k) @ B(k,n)` with A read from a `(k, m)` buffer, the
`wgrad` role, five shapes all with `k = 49152`. Entry state, re-verified one case
per process: TN 83.735 ms/step against PyTorch-ROCm's 23.300, **ratio 3.594**,
at 130-158 TFLOP/s. NN 3.075 and NT 1.111 are the regression check.

Protocol: >= 25 warmups and >= 100 individually synchronized iterations, one case
per process for every number quoted as per-case, timing never from a `--pmc` run,
every extent a runtime value.

## What the harness measures, and the floor that puts under TN

`bench_linear_gemm`'s timed region for a `transpose_a` case is
**`_copy_strided` then the GEMM**, unconditionally:

```
def _full() raises:
    if target.transpose_a:
        _materialize()
    _gemm()
```

The harness is read-only for this work, so the transposed-A materialization is
inside every TN number below whatever the dispatch learns to do. It is a full
read and write of A -- 453 MB for `attn_c_attn_wgrad`, 9.9 GB for
`lm_head_wgrad` -- and PyTorch-ROCm pays none of it.

A contiguous 2 GB-each-way copy on this part measures **3.68 TB/s** (`/tmp/bw`
microbenchmark, 50 synchronized iterations). At that rate the five copies cost
123 / 41 / 164 / 41 / 2687 us, i.e. **7.12 ms/step weighted**. So even with a
GEMM exactly as fast as PyTorch-ROCm's,

    TN >= (7.12 + 23.30) / 23.30 = 1.305

and the 1.03 bar is **not reachable under this harness**. That is a property of
what is being timed, not of the kernel: the copy-free route below is measured at
3-23% faster than copy+GEMM, and the harness cannot call it.

## Change 25 — a native-layout LDS tile, so the core serves NN and TN too

**Hypothesis.** NT won because both operands are k-major and an MFMA operand
wants four elements adjacent along k. TN's post-copy GEMM is
`A(m,k) @ B(k,n)`: A is k-major, B is not. B's global rows run along n, so
either the transpose happens on the way into LDS -- where a lane can only fetch
two bytes per row and the load instruction count grows eightfold, 128 wave
instructions per k tile per operand against 16 -- or it happens on the way out of
LDS, where the tile is read four times over (each of the four wave-rows and
wave-columns re-reads it) but each access is cheap. Counting says the second is
the cheap side, because LDS is a 128-byte-per-cycle pipe and 64 lanes x 2 bytes
is exactly 128 bytes: four `ds_read_u16` move the same 512 bytes in the same
four cycles as one `ds_read_b64`, and only the instruction count grows.

**Predicted effect.** NN-layout GEMM at NT's rate, 400-450 TFLOP/s, against the
150-170 the existing route reaches.

**Measured effect.** `_nt_mfma_kernel` gained `A_KMAJOR` / `B_KMAJOR`; the NT
instantiation is unchanged. The bank collision the layout would otherwise have
-- the fragment's four elements are `BX` apart and `4 * BX * 2` bytes is a
multiple of 128, so the `hi` half-wave lands on the banks the `lo` half holds --
is removed by permuting the row as `x ^ (32 * ((k >> 2) & 1))`, which costs no
LDS and whose key is exactly the reader's `hi`. `SQ_LDS_BANK_CONFLICT` on the
shipped kernel: **0**.

`attn_c_attn_dgrad` (49152, 768, 2304), one case per process: 1027.9 -> 361.0 us,
169 -> 482 TFLOP/s.

**Decision.** Accept, and route `transpose_b == 0` through it when its grid
covers the device.

## Change 26 — shift the edge tile instead of guarding it

**Hypothesis.** `lm_head_wgrad`'s GEMM is (50304, 768, 49152) and
`50304 % 256 == 128`, so it takes the guarded instantiation. The NT work
recorded that as costing "up to 7%".

**Measured effect.** It costs **2.2-3.2x**, not 7%, and the mechanism is
register spill. At (50304, 768, 3072) the guarded kernel measures 159 TFLOP/s
and reports 144 bytes of scratch per thread; the same kernel on an exact
m = 50176 measures 501 with zero scratch. The guards and the sixteen-register
unconditional rounding block that works around the gfx942 MFMA-read miscompile
(journal D8) do not coexist in 128 VGPRs.

Two fixes, both measured:

1. The per-element tail of the masked native load -- eight predicated scalar
   loads -- was 3.2x slower on its own. It is also dead: the route already
   requires the operand's contiguous extent to be a multiple of the eight-element
   vector, so a vector at an aligned offset is wholly inside the extent or wholly
   outside it. One guard, no tail: 159 -> 227 TFLOP/s.
2. The remaining 2.2x is the guards themselves. An edge tile is now **shifted
   back** to end at the extent rather than masked. It recomputes rows another
   workgroup also computes, but with the same k loop and the same MFMA order, so
   both store bit-identical bytes; the redundant work is one tile out of
   `ceildiv(m,BM) * ceildiv(n,BN)`. 227 -> 503 TFLOP/s, and the k edge -- which
   cannot be shifted, the missing k range is real -- keeps the guarded kernel.

This also improved NT, which had been paying it silently: `lm_head_fwd`
8564 -> 7763 us on the table, `pre_nt_n50257` (4096, 50257, 768) 825.6 -> 764.3.

**Decision.** Accept both.

## Change 27 — global split-K, chosen by cost rather than by a fill threshold

**Hypothesis.** The four per-layer weight gradients contract 49152 into a
768-3072 square, so their 256x256 output grid is 9-36 workgroups on 304 CUs. No
tile choice fixes that -- the shortage is output area -- and one workgroup of
this tile is resident per CU, so slabs of K are the only axis left.

**Predicted effect.** The four shapes reach the same rate the grid-filled ones do.

**Measured effect.** The slab count matters more than expected, and the obvious
rule (fill one wave, never overflow) is wrong. Exactly one workgroup is resident
per CU, so `parts` slabs cost `ceil(tiles * parts / cus)` serialized waves of
`ktiles / parts` k tiles each, and a count that leaves a ragged last wave can
beat a smaller one that fits in a single partial wave. At (2304, 768, 49152),
27 tiles, one case per process:

| parts | workgroups | waves | total us |
|---:|---:|---:|---:|
| 4 | 108 | 1 | 960.9 |
| 8 | 216 | 1 | 590.7 |
| 16 | 432 | 2 | 608.8 |
| **32** | **864** | **3** | **575.1** |
| 48 | 1296 | 5 | 671.1 |

32 slabs win although they write four times the workspace of 8, because 864
workgroups use 94.7% of three waves where 216 use 71% of one. The shipped rule
is that cost model -- `ceil(tiles*parts/cus) * (ktiles/parts) * 4000` cycles for
the GEMM, using experiment AD's measured 3995 cycles per k tile, plus the
workspace charged at roughly 4 TB/s -- searched over the divisors of the k tile
count up to 64, declining when `parts == 1`. It reproduces the table above.

The reduction is cheaper than the model charges: at 32 slabs it reads 226 MB and
takes 50.0 us, i.e. 4.6 TB/s, which is above HBM because the workspace still sits
in the 256 MB Infinity Cache the GEMM just wrote it to.

TN 2.989 -> 1.623 on the table.

**Decision.** Accept.

## Change 28 — a register-only transpose for the materialization (2.1 -> 3.65 TB/s)

**Hypothesis.** With the GEMM fixed, the copy the harness forces is 12.2 ms of
TN's 37.8 ms and runs at 2.1-2.4 TB/s. Find out what that costs and why.

**Measured effect.** Three microbenchmarks, 2 GB each way, 50 synchronized
iterations:

| access pattern | TB/s |
|---|---:|
| contiguous copy | 3.68 |
| 256-byte runs strided by a matrix row, both sides | 3.62 |
| 128-byte runs | 3.57 |
| 64-byte runs | 2.79 |
| 32-byte runs | 1.32 |

So the access pattern a tiled transpose needs is capable of full bandwidth and
the loss was the staging. The existing kernel moves one element per thread per
access (64 lanes x 2 bytes = one 128-byte request) through a padded LDS tile with
a barrier between the read and the write phases; widening the accesses to 16
bytes and enlarging the tile to 256-byte runs got 2.08 -> 3.17 TB/s at
(2304, 49152), and a `_T2DV_GROUP`-wide band remap of the tile order on top of
that measured **-0.9%**, which rules out DRAM page locality across workgroups as
the remaining cause and points at the staging itself: 32 KB of LDS is two
workgroups per CU and the barrier leaves two tiles of memory-level parallelism in
flight.

The shipped kernel uses **no LDS and no barriers**. One thread owns a
`VEC x VEC` element block: it reads `VEC` 16-byte pieces, one from each of `VEC`
source rows, transposes them in registers, and writes `VEC` 16-byte pieces, one
to each of `VEC` destination rows. Lanes are arranged 8 across the source rows by
8 down them, which makes both sides 128-byte runs -- within 2% of 256-byte runs
by the table above. It needs `rows % VEC`, `cols % VEC`, `src_ld % VEC` and
16-byte bases, all properties of strides and the allocator rather than of a
shape; the scalar tiled kernel still serves everything else.

| copy | before | after | TB/s |
|---|---:|---:|---:|
| attn_c_attn_wgrad | 218.1 us | 142.1 | 2.08 -> 3.19 |
| attn_c_proj_wgrad | 63.9 | 45.7 | 2.37 -> 3.31 |
| mlp_c_fc_wgrad | 298.9 | 195.1 | 2.02 -> 3.10 |
| lm_head_wgrad | 4502.6 | 2644.2 | 2.20 -> 3.74 |

Unclamping the grid so every wave owns exactly one region (a 4096-block clamp
gave some waves one and some two, and the makespan is the larger) is a further
0.6-2.4%.

TN 1.623 -> 1.419. `bench_permute_copy`, which shares this primitive, is within
0.1% on all eight cases.

**Decision.** Accept.

## Change 29 — a k-pair native LDS tile, and the copy-free transposed-A route

**Hypothesis.** With both operands native -- which is what a true TN kernel
needs, A from its `(k, m)` buffer and B from its `(k, n)` buffer -- the
one-element-per-row native tile needs four `ds_read_u16` per fragment for BOTH
operands, and the packing of four halves into two dwords is VALU the k-major
path does not pay. Counters on the both-native kernel at
(50304, 768, 49152): **6.65 VALU per MFMA against 3.71 for one native operand
and 3.12 for none, and 100 bytes of scratch per thread.** Hold the k PAIR
`(2p, 2p+1)` of column `x` adjacent at `2x` instead: an MFMA operand is four
consecutive k of one column, i.e. two adjacent pairs, so the fragment becomes two
`ds_read_b32` whose two dwords ARE the operand register pair -- no shifting, no
`v_perm`, no packing at all. Both accesses stay bank-optimal with no padding and
no permutation (a fragment read is dword `p * BX + x`, so `BX` a multiple of 32
drops the row out and the two half-waves cover all 32 banks twice, which is the
two cycles 256 bytes take anyway).

**Predicted effect.** The both-native kernel stops spilling and closes most of
the gap to the one-native one.

**Measured effect.** Both-native, one shape per process, output checked element
by element:

| shape | one per row | k pairs |
|---|---:|---:|
| (2304, 768, 49152) | 257 TFLOP/s | **360** |
| (768, 768, 49152) | 255 | **337** |
| (3072, 768, 49152) | 297 | **456** |
| (768, 3072, 49152) | 293 | **456** |
| (50304, 768, 49152) | 343 | **517** |

Scratch 100 -> 0 bytes per thread.

With only ONE native operand the same change is a wash that goes the wrong way
on the shapes this target cares about: -0.5 to -2.9% on the grid-filling data
gradients but **+3 to +4.9% on the split-K weight gradients** (403.6 -> 423.3 us
of GEMM at (2304, 768, 49152), from `rocprofv3`, with no spill either way -- it
trades one 16-byte global load for two 8-byte ones plus an interleave, to halve
LDS reads that were not binding). Weighted over the harness that is
TN 1.419 -> 1.457 and NN 1.098 -> 1.086, a net loss. So `PAIR` is derived as
"both operands native" and the one-native path keeps the permuted single-element
layout.

That unlocks `_tn_mfma_route`, which reads A straight out of its `(k, m)` buffer
with no materialization at all. `_matmul_spec_operands_launch` takes it whenever
A's two innermost strides are `(1, m)`, B is contiguous and there is no bias.
Measured against materializing A^T and running the dense route, one shape per
process:

| shape | copy + GEMM | direct TN | saving |
|---|---:|---:|---:|
| (2304, 768, 49152) | 575.8 us | 483.6 | 16.0% |
| (768, 768, 49152) | 197.7 | 171.7 | 13.1% |
| (3072, 768, 49152) | 661.4 | 509.1 | 23.0% |
| (768, 3072, 49152) | 527.1 | 508.8 | 3.5% |
| (50304, 768, 49152) | 9600.5 | 7369.0 | 23.2% |
| (1024, 1024, 32768) | 226.5 | 203.5 | 10.2% |
| (2312, 776, 49152) | 844.3 | 809.1 | 4.2% |
| (50304, 776, 24576) | 6507.1 | 5439.3 | 16.4% |

`bench_linear_gemm` cannot show any of this: it materializes A^T unconditionally
before calling the dispatch. It is measured with a standalone harness that calls
`_tn_mfma_route` directly and checks every output element.

**Decision.** Accept both, with `PAIR` derived from the operand layouts.

## Defect analysis D9 — a split-K slab count derived from a floored tile count

The slab search used `ktiles = k // 32` and required `ktiles % parts == 0`. For a
k that 32 does not divide, the slabs then cover `parts * (k_per // 32) * 32 < k`
and the tail is silently dropped. Nothing shipped was wrong -- the only entry
that reached it is gated on `k % 32 == 0` upstream in
`_amd_dynamic_mfma_dispatch` -- but `_tn_mfma_route` is called from
`_matmul_spec_operands_launch` with no such gate. The split-K branch now declines
when `k % 32 != 0` and the unsplit route's guarded instantiation handles the
partial tile; verified by the standalone harness reporting a decline at
(2304, 768, 49168).

## Where TN ended

Per case, one case per process, 25 warmups and 100 synchronized iterations:

| case | rocm us | before | after | ratio | copy_us | GEMM TFLOP/s |
|---|---:|---:|---:|---:|---:|---:|
| attn_c_attn_wgrad | 360.16 | 1270.8 | 582.0 | 1.616 | 142.1 | 396 |
| attn_c_proj_wgrad | 152.15 | 434.3 | 197.8 | 1.300 | 46.2 | 383 |
| mlp_c_fc_wgrad | 460.34 | 1680.0 | 656.8 | 1.427 | 195.1 | 503 |
| mlp_c_proj_wgrad | 452.09 | 1459.5 | 525.2 | 1.162 | 45.7 | 483 |
| lm_head_wgrad | 6203.16 | 25600.4 | 9603.8 | 1.548 | 2644.2 | 546 |

**TN 3.594 -> 1.423 one case per process, 3.599 -> 1.420 on the table**
(1.413 with `--pattern-check=1`, 1.421 with `--chunk-check=1`); 15 of 15 cases
pass all three gates. The 1.03 bar is not met and cannot be under this harness:
the copy alone is 30.5% of PyTorch-ROCm's entire budget at full HBM bandwidth,
and the remaining GEMM is 25.3 ms against 23.3, i.e. **1.084**.

Regression checks, all after the change, one case per process:

| check | recorded | now |
|---|---|---|
| `bench_linear_gemm` NN | 3.075 | **1.099** |
| `bench_linear_gemm` NT | 1.111 | 1.102 |
| whole table | 2.552 | 1.205, 15/15 pass all three gates |
| `bench_attention_bmm` default / pattern / causal | 51.236 / 51.040 / 50.461 ms | 51.217 / 51.022 / 50.462, 6/6 pass |
| `bench_permute_copy`, eight cases | - | within 0.1% |
| GPT-2 decode NT and NN, m512, four n | - | +0.0% to +1.7% |
| GPT-2 prefill NT and NN, m4096, five n | - | -37.5% to +1.4% |
| NT grid-fill gate shapes, four | - | -4.6% to -0.7% |

What is left, in the order the evidence supports:

* **The copy, 7.86 ms of the 33.1.** It is 3.10-3.74 TB/s against a 3.68 TB/s
  ceiling, so at most 0.75 ms remains in it, and only by deleting it -- which the
  shipped `_tn_mfma_route` does everywhere except this harness.
* **`lm_head_wgrad`'s GEMM, 546 TFLOP/s against PyTorch-ROCm's 612.** Counters
  say the k tile takes 3161 cycles against 2048 of MFMA, with 1280 cycles of LDS
  traffic that is irreducible at a 64x64 warp tile (experiment AD) and zero bank
  conflicts. A wider warp tile needs more than 64 accumulator VGPRs and loses the
  fourth wave per SIMD.
* **`attn_c_attn_wgrad`, 396 TFLOP/s.** Its 27 output tiles do not divide the
  device at any slab count: the best is 864 workgroups over three waves, 94.7%.
  A stream-K assignment, where a workgroup that draws several slabs of the SAME
  output tile accumulates them in registers and writes one partial, would cut
  both the workspace and the epilogue; not attempted.

# nanoGPT NN (data-gradient Linear) GEMM, MI300X — 2026-07-26

Fifth pass over `harness/nanogpt_train/bench_linear_gemm.mojo`, one variant:
**NN**, `C(m,n) = A(m,k) @ B(k,n)` with both operands in their natural dense
row-major layout, the `dgrad` role, five shapes all with `m = 49152`. NN already
ran on the hand-written MFMA core the NT and TN passes built, so the entry state
was **1.098 per case** (25.366 ms/step against PyTorch-ROCm's 23.099) at 397-536
TFLOP/s. NT 1.097 and TN 1.422 are the regression check.

Protocol: >= 25 warmups and >= 100 individually synchronized iterations, one case
per process for every number quoted as per-case (`scripts/percase_variants.py`),
timing never from a `--pmc` run, every extent a runtime value.

## What the NN kernel actually spends, measured

Two things had to be established before anything could be aimed at.

**The clock and the MFMA floor.** `rocm-smi` during a `lm_head_dgrad` run reports
a sustained 1327-1396 MHz, so the cycle accounting below uses 1.395 GHz and the
BF16 peak is 304 x 2048 x 1.395e9 = 868 TFLOP/s. Per k tile per CU the geometry
costs 256 `v_mfma_f32_32x32x8_bf16` over 4 SIMDs at 32 cycles = **2048 cycles**;
that is the floor.

**The split between per-k-tile and per-output-tile cost.** Sweeping k at fixed
(m, n) = (49152, 768), one case per process, gives a straight line: k = 32 / 64 /
128 / 256 / 768 measures 42.27 / 47.26 / 55.46 / 71.90 / 139.71 us. Its slope is
4.24 us per k tile over the two workgroup rounds this grid takes, i.e.
**C = 2960-3070 cycles per k tile**, and its intercept is 19 us per round, i.e.
**E = 26,500 cycles per output tile**. `rocprofv3` on the same k = 32 point gives
a kernel time of 31.47 us against the harness's 42.65, so **every harness number
carries about 11 us of launch-and-synchronize overhead**; `scripts/rocm_gemm_reference.py`
times PyTorch the same way (per-iteration `torch.cuda.synchronize`), so the
comparison is fair, but it is 8% of `attn_c_proj_dgrad`'s 139 us and worth
knowing.

E is essentially the output write and is not addressable: at k = 32 the kernel
does one k tile of MFMA (1.5 us) and writes 38 MB per round in 15.7 us, i.e.
2.8 TB/s, against 3.27 TB/s for the harness's own `_fill_const` on a comparable
buffer. The whole recoverable part of E across the five shapes is about 1.4% of
NN, and only by widening the epilogue's 64-byte stores, which the accumulator
layout (lane = one output column) cannot do without an LDS transpose.

C decomposes by ablation -- each variant keeps the register and instruction
structure and only deletes work, so the results are wrong but the timings are
meaningful. On (49152, 768, 12288), 384 k tiles, two rounds:

| loop body | us | TFLOP/s | what it says |
|---|---:|---:|---|
| shipped at entry | 1774.8 | 522.7 | |
| fragment reads halved | 1834.8 | - | LDS reads are **not** the bottleneck |
| no `s_barrier` | 1597.5 | - | the barrier is 230 cycles/k tile |
| no global load, no LDS refill | 1443.1 | 643 | the refill path is 431 cycles/k tile |
| ... and no fragment reads either | 1270.6 | 730 | pure MFMA ceiling for this kernel |

So the ceiling this geometry can reach is **730 TFLOP/s** at the measured 94.7%
grid fill (771 at full fill, consistent with the 810 MFMA-only microbenchmark),
the fragment reads cost 8% of it, and the global-load-plus-refill-plus-barrier
group costs 20%. That last group is what the change below attacks.

## Change 30 — a k-pair native LDS tile for the one-native layouts

**Hypothesis.** Journal change 29 derived `PAIR` as "both operands native" because
with only one native operand it measured +3 to +4.9% on the split-K weight
gradients. But the same entry recorded it as **-0.5 to -2.9% on exactly the
grid-filling data gradients this pass is about**, and the reason is countable:
the unpaired native tile reads a fragment as four `ds_read_u16`, and a two-byte
read still occupies a whole LDS bank slot, so 64 lanes take two LDS cycles to
move 128 bytes -- half the rate of a `ds_read_b64`. Four of them are eight cycles
where the journal's derivation assumed four, plus two `v_perm_b32` to pack. The
pair layout makes the fragment two `ds_read_b32` (which LLVM merges into one
`ds_read2_b32`), the same 512 bytes in four cycles, and no packing.

**Predicted effect.** B's fragment reads drop from 1024 to 512 LDS cycles per CU
per k tile and 16 `v_perm_b32` per wave disappear.

**Measured effect.** `PAIR` became a kernel parameter instead of a derived
constant, and `_dense_mfma_route` chooses it per branch: `PAIR_FILL` (the
grid-filled branch, which is every NN dgrad) is "either operand native",
`PAIR_SPLIT` (the split-K branch, which is every harness-TN wgrad) keeps "both
native". Counters on the result: 15 LDS instructions per 16 MFMA per wave, zero
bank conflicts, and **VGPR 128 with 20 bytes of scratch -> VGPR 120 with none**.

NN 1.098 -> 1.088 per case; NT and TN unchanged (1.097 -> 1.099, 1.422 -> 1.424).

**Decision.** Accept. The regime split is the one change 29 already measured; it
is recorded here as a per-branch parameter rather than a derived constant so the
two callers can differ.

## Change 31 — refill the next LDS stage in the MIDDLE of the MFMA sequence

**Hypothesis.** The ablation above says the refill path costs 431 cycles per k
tile, and the ISA says why. With the refill placed after the fragment reads and
before all sixteen MFMAs, the body ends with `s_waitcnt vmcnt`, three
`ds_write`s, one MFMA, `s_waitcnt lgkmcnt(0)` and the barrier: the LDS write
traffic and the wait for it are a **tail with nothing left to cover them**. Put
the refill after half the MFMAs instead. A wave issues one MFMA per 128 cycles
(four waves share a SIMD's matrix pipe at 32 cycles each), so eight MFMAs in is
about a thousand cycles after the global loads issued -- late enough that they
have landed -- and the writes then retire under the remaining eight.

**Predicted effect.** The 431-cycle group shrinks; nothing else moves.

**Measured effect.** A sharp optimum, one case per process, NN weighted:

| k steps of MFMA before the refill (of four) | NN |
|---:|---:|
| 0 (the shipped order) | 1.085 |
| 1 | 1.085 |
| **2** | **1.057** |
| 3 | 1.086 |

Splitting the refill further -- A's two `ds_write_b64` at step 2 and B's
`ds_write_b128` at step 3 -- measures 1.058, i.e. nothing. Moving `_load` to the
top of the body or into the middle next to the refill measures within 0.3% of
the winner in both directions: the compiler already hoists the global loads to
the top of a tight body.

The optimum is layout-dependent, so the position is a per-route parameter
`FILL_AT`. NT (`_nt_mfma_route`, both operands k-major, twice as many
`ds_read_b64` and no `v_perm` in the body) measures **1.116 at 2 against 1.094 at
0**, and keeps 0. NN and TN (`_dense_mfma_route`) take 2.

**Decision.** Accept, parameterized.

## What was tried and rejected, with numbers

Every one of these was built and measured one case per process on the five NN
shapes; the figure is the NN weighted ratio unless stated.

* **Unrolling the k loop by two so the LDS stage bases become compile-time
  offsets.** The whole 64 KB LDS budget fits the 16-bit unsigned `ds_read`
  immediate, so the unrolled body needs **zero** `v_lshl_add_u32` address
  instructions against 28 per wave per k tile rolled, with no spill (VGPR 128,
  scratch 0). It measures **1.149** rolled-1.085, and 1.235 / 1.634 for NT / TN.
  Pinning the global loads at the top of the body with a `sched.barrier(0)` makes
  it worse still (1.183). Retried on top of change 31 it measures 1.166 against
  1.057. The address VALU is not on the critical path, and a single tight body is
  what lets the scheduler place a tile's global loads a whole body ahead of the
  `ds_write` that consumes them; in the unrolled ISA they sit two instructions in
  front of it.
* **Prefetching the global loads a whole k tile ahead** (the refill of stage
  `next` consumes registers loaded in the previous trip, so the `vmcnt` wait is
  a full k tile old). It costs 8 live VGPRs across the MFMA phase, which at 120
  used is nominally free and in practice is not: the allocator spills **32 VGPRs
  and 132 bytes of scratch**. Measured before change 31 it improved C by 3.7%
  and inflated E by 19,500 cycles per output tile, for a net 1.085 -> 1.128;
  peeling the last trip instead of guarding the pointer bump measures 1.216; on
  top of change 31 it measures 1.190.
* **Reading the fragments one k step ahead of the MFMA that needs them** instead
  of in one burst, which halves their peak register footprint from 32 VGPRs to
  16 and would have made room for the prefetch above: **1.094** against 1.057.
* **`s_setprio(1)` around each MFMA burst and `s_setprio(0)` around the refill**,
  which is what MAX's ping-pong matmul does: **1.223** against 1.057.
* **XCD band count.** The shipped value is the runtime XCD count, 8. One case per
  process, NN weighted: 1 -> 1.0919, 2 -> 1.0973, 4 -> 1.0838, 8 -> 1.0853.
  Within the run-to-run spread; not changed.
* **Stream-K, rejected analytically and recorded because it is the obvious
  move.** Every NN shape has an output grid that is 94.7% of a whole number of
  workgroup rounds -- 576 tiles on 304 CUs is 1.895, 2304 is 7.579 -- and that
  5.3% is real: the same kernel on (51712, 768, 2304), whose 606 tiles are 99.7%
  of two rounds, measures 501 TFLOP/s, and on (77824, 768, 2304), exactly three
  rounds, 508, against 479 for the 49152 shape. It is also not fixable by tile
  choice: for m = 49152, n = 768 the fraction is 94.7% for **every** tile shape
  and residency, because halving the tile area doubles both the tile count and
  the resident count. Stream-K is therefore the only route, and its fixup does
  not pay at this tile: balancing 576 tiles over 304 workgroups splits about 272
  of them, each partial is a 256x256 FP32 tile of 256 KB, and 155 MB of extra
  write-then-read at the 4.6 TB/s change 27 measured for exactly this traffic is
  34 us against the 19 us the balance saves.

## Where NN ended

Per case, one case per process, 25 warmups and 100 synchronized iterations:

| case | rocm us | before | after | ratio | TFLOP/s |
|---|---:|---:|---:|---:|---:|
| attn_c_attn_dgrad | 343.44 | 363.31 | 359.58 | 1.047 | 484 |
| attn_c_proj_dgrad | 141.38 | 145.91 | 139.49 | **0.987** | 416 |
| mlp_c_fc_dgrad | 434.10 | 473.10 | 461.05 | 1.062 | 503 |
| mlp_c_proj_dgrad | 467.98 | 541.04 | 523.19 | 1.118 | 443 |
| lm_head_dgrad | 6456.53 | 7085.20 | 6726.87 | 1.042 | 565 |

**NN 1.098 -> 1.062 per case** (1.066 with `--pattern-check=1`, 1.060 with
`--chunk-check=1`), 1.065 on the harness's own table. Fifteen of fifteen cases
pass all three gates on all three runs. The 1.03 bar is **not** met; the shortfall
is 3.1%.

Run-to-run spread matters at this margin and is quoted so it is not mistaken for
an effect: four independent full per-case sweeps of the same binary give NN
1.058 / 1.060 / 1.062 / 1.066, NT 1.094 / 1.099 / 1.106 / 1.113 and TN 1.412 /
1.416 / 1.416 / 1.417. Nothing below 1% should be read as real.

Regression checks, all after the change, one case per process:

| check | before | after |
|---|---|---|
| `bench_linear_gemm` NT | 1.097 | **1.094** (1.113 pattern, 1.099 chunk) |
| `bench_linear_gemm` TN (harness) | 1.422 | **1.417** (1.412 pattern, 1.416 chunk) |
| copy-free TN, five wgrad shapes, direct | - | -0.3% to +1.0%, all exact |
| GPT-2 decode NN and NT, m512, six shapes | - | -5.3% to +1.0% |
| GPT-2 prefill NN and NT, m4096, seven shapes | - | -5.2% to +0.3% |
| NT grid-fill gate shapes, three | - | -0.3% to +0.1% |
| ten non-multiple extents x three gates | - | all exact |
| `bench_attention_bmm` default / pattern / causal | 51.217 / 51.022 / 50.462 ms | 51.234 / 51.037 / 50.490, 6/6 pass |

## Correction to the record: what the copy-free TN route is worth

The TN pass recorded "the remaining GEMM is 25.3 ms against 23.3, i.e. 1.084".
That figure is the harness total minus the materialization, so it is the
**materialized** GEMM (A k-major, split-K), not the copy-free route
`_tn_mfma_route` actually takes in production. Timed directly, one shape per
process, 25 warmups and 100 synchronized iterations, every output element
checked:

| shape | copy-free GEMM us |
|---|---:|
| (2304, 768, 49152) | 485.97 |
| (768, 768, 49152) | 171.65 |
| (3072, 768, 49152) | 507.83 |
| (768, 3072, 49152) | 507.34 |
| (50304, 768, 49152) | 7337.34 |

Weighted by calls per step that is **27.41 ms against PyTorch-ROCm's 23.30, i.e.
1.176** for the GEMM alone -- not 1.084. The copy-free route saves 17% against
the harness's 33.0 ms of copy-plus-GEMM, which is consistent with the 1.422 ->
1.323 the production measurement showed and with the copy being a smaller share
of production than of the harness.

## What is left, in the order the evidence supports

* **The 20% between 537 and 672 TFLOP/s: the global load, the LDS refill and the
  barrier.** Change 31 took part of it. The rest needs either a global-to-LDS DMA
  (`global_load_lds`, which forces a lane-linear LDS layout the swizzled and
  paired tiles do not have) or a deeper prefetch, and every deeper-prefetch form
  tried runs into the 128-VGPR wall: at four waves per SIMD a 1024-thread
  workgroup cannot exceed it, and 120 are already in use.
* **The A tile's 64-byte global rows.** At BK = 32 an A tile row is 64 bytes,
  half of a 128-byte line, and the other half is the next k tile's -- but the two
  tiles' working set is 48 KB against a 32 KB vector L1, so it is evicted first.
  NT, where BOTH operands have 64-byte rows and the working set is 64 KB,
  measures 15% slower per k tile than NN on matched shapes, which is the
  circumstantial evidence. Fixing it means an A tile that spans 64 k in LDS
  (256x64x2 = 32 KB single-buffered, with B double-buffered at 32 KB), which adds
  a third barrier per two k tiles and needs the k loop to alternate -- and every
  alternating form measured 10-20% slower.
* **The 5.3% of grid quantization**, closed above.
* **The output write, 2.8 TB/s against 3.27.** Worth about 1.4% of NN and needs
  an LDS transpose in the epilogue, which is the region the gfx942 MFMA-read
  miscompile (journal D8) lives in.

## Plumbing the fused attention kernels, and the defect the harness hid

Both flash-attention kernels were harness-only until now: neither module was in
`_MOJO_MODULES`, so the eager device never loaded them and SDPA ran the
decomposition regardless of how fast the kernels were. Wiring them in took a
Python bridge (`flash_attention_ops.mojo`), a new autograd Function, and one
change to the forward.

**The forward now emits the row log-sum-exp.** The backward needs `L` to recover
`P = exp(S - L)` without a second normalization pass. Both halves were already
in registers, so it costs nothing measurable (0.748x -> 0.750x). The units are
the trap: the MFMA kernel keeps `run_m` UNSCALED and folds the scale into the
exponent (`sl = scale * LOG2E`, `exp2`), so `L = run_m * scale + ln(tot)`, while
the baseline kernel scales before the max, so there `L = running_max + ln(tot)`.
Getting this wrong is invisible in the forward -- O is normalized by the same
sum either way -- and surfaces only as wrong gradients. Checked against an
independent FP32 reference across all four dispatch regimes: 5.7e-06 (hd64),
7.6e-06 (hd96), 9.5e-06 (hd128), 3.8e-06 (noncausal and kv>q), 1.2e-07
(baseline).

### D10: the harness specified the wrong causal convention

`kv_longer` (seq_q 256, seq_kv 1024, causal) measured ~100% wrong against
PyTorch-ROCm while PASSING the harness at the same shape.

The harness contract I wrote specified BOTTOM-RIGHT causal alignment, query `q`
attends `0 ..= q + (seq_kv - seq_q)`. PyTorch aligns TOP-LEFT: query `q` attends
`0 ..= q` whatever `seq_kv` is (`torch.nn.attention.bias`, "upper left causal
bias"), confirmed empirically rather than from memory. Both agents implemented
the spec faithfully; the spec was wrong.

The fix is `delta = 0`, and `delta` was ALREADY `seq_kv - seq_q` = 0 for every
square shape, so nanoGPT and all self-attention are provably untouched: forward
0.748 -> 0.750x, backward 0.998 -> 0.999x, both inside the run-to-run spread.
Correct semantics also turn out cheaper, because keys past `seq_q` are never
attended: `cross_kv_longer` backward halved, 141.59 -> 70.17 us.

**Why six mutation tests missed it.** The oracle inherited the same convention
from the same spec. Differential testing cannot see a defect shared by both
sides, however independent their structure -- and this oracle was as independent
as they come: one thread per row against block-per-row, different reduction,
different parallel axis. Mutation testing validated the gate against the spec;
only a comparison with PyTorch could validate the spec against reality. That
comparison belongs BEFORE handing a harness to an agent, not after wiring up its
output. This is the second time this session the harness rather than the kernel
was the thing that was wrong (see the order-dependence that turned a claimed 49%
win into a 14% loss).

### Where the training step stands

nanoGPT 124M, batch 48, block 1024, bf16 autocast, one full step:

| | PyTorch-ROCm | mojo | ratio |
|---|---:|---:|---:|
| step (median of 20) | 156.73 ms | 172.10 ms | 1.098 |
| p10 / p90 | 156.50 / 156.93 | 171.73 / 172.41 | |
| tokens/s | 313607 | 285598 | 0.911 |

921.39 -> 661.86 -> 420.94 -> 409.34 -> 353.93 -> 172.10 ms over the session,
5.88x -> 1.098x. Single-step loss agrees to 1e-5 relative (10.977398 against
10.977295); the 25-step divergence is accumulated bf16 rounding through AdamW,
not a defect.

Component standing, all independently re-measured, per case one process each:

| kernel | start | now |
|---|---:|---:|
| SDPA forward | 3.287x | 0.750x |
| SDPA backward | 1.359x | 0.999x |
| GEMM NN (dgrad) | 3.079x | 1.066x |
| GEMM NT (fwd) | 2.855x | 1.102x |
| GEMM TN (wgrad) | 3.600x | 1.416x harness / 1.323x production |

Both attention kernels now meet or beat PyTorch-ROCm's fused equivalents. The
remaining step-level gap is ~15 ms and lives almost entirely in the GEMMs, with
TN the worst: A arrives m-major and Tensile consumes the strided view with no
copy, which `_tn_mfma_route` matches only partially (1.176 for the GEMM alone).

# Stride-aware fused attention, MI300X — 2026-07-26

## The waste, measured before touching anything

nanoGPT builds its attention inputs as
`q.view(B, T, H, D).transpose(1, 2)`, so q, k and v are logically
`[batch, heads, seq, head_dim]` but physically strided `(T*H*D, D, H*D, 1)`.
The fused kernels indexed densely, so `_fused_fa_inputs` called `_tc()` on each
of them and on a non-contiguous `grad_output` in the backward. In the v7 trace
that is:

| | v7 |
|---|---|
| `data_movement_ops__run_gather` | 96/step, 3766.2 us/step, 39.2 us/call |
| of which the fused-attention `_tc()` calls | 48/step, ~1923 us/step |

3 per layer for q/k/v times 12 layers, plus 12 for `grad_output`. PyTorch-ROCm
issues none of them: its fused attention takes stride arguments.

The other 48 gathers per step are unrelated copies elsewhere in the step and are
still there.

## Change 32 — per-operand `RowStrides` on both fused entry points

**Hypothesis.** The head_dim axis is contiguous in the caller's view, and that is
the only axis the vectorized loads and the LDS row fills run along. So only the
base-address arithmetic that locates a `(batch, head, seq)` row has to change,
and the copies can go.

**Predicted effect.** -1.9 ms/step of gather, no change to the kernels.

**What was built.** A `RowStrides` (batch, head, seq) triple per read operand:
q/k/v on the forward, grad_output/query/key/value/out_fwd on the backward. The
head_dim stride is NOT carried -- it must be 1, and `_fused_fa_inputs` declines
anything else rather than guessing, falling through to the decomposition. `dq`,
`dk`, `dv`, the forward's `output` and `softmax_lse` stay dense, because we
allocate them.

**Measured effect on the harness (dense inputs), nanogpt case, interleaved A/B
of two binaries, 25 warmups / 100 iterations, three rounds each:**

| | before | after |
|---|---:|---:|
| forward us/layer | 513.06 / 513.18 / 513.19 | 518.83 / 518.22 / 518.36 |
| forward harness ratio | 0.755 | 0.762 (+1.05%) |
| backward us/layer | 2413.00 / 2414.14 | 2387.11 / 2385.50 |
| backward harness ratio | 1.000 | 0.989 (-1.1%) |

All 30 cases pass (15 per harness, five of them new and strided).

**Measured effect in production, v7 vs v8 trace, per kernel:**

| kernel | v7 | v8 | delta us/step |
|---|---:|---:|---:|
| `run_gather` | 96/step, 3766.2 us | 48/step, 1843.3 us | **-1922.9** |
| `fa_mfma` fwd | 523.7 us/call | 567.2 us/call | +521.7 |
| `fa_bwd_dkv` | 1612.1 us/call | 1646.0 us/call | +406.5 |
| `fa_bwd_dq` | 762.2 us/call | 714.5 us/call | -572.1 |
| net | | | **-1566.8** |

Group level: SDPA forward 48 -> 12 kernels/step and 7.753 -> 6.807 ms (0.854 ->
0.750x ROCm); SDPA backward 36 -> 24 kernels/step and 28.956 -> 28.326 ms (0.853
-> 0.834x). Whole step, GPU kernel time summed over the comparison groups,
171.685 -> 169.640 ms/step; wall clock median of 20, 172.10 -> 170.489 ms
(p10/p90 170.311 / 170.776, so the distributions do not overlap). 1.098 ->
1.088x ROCm.

**Decision: keep.** The copies are worth 1.92 ms/step and the strided kernels
cost 0.93 of it back, net 1.57 ms.

### Why there is a `DENSE` compile-time arm, and what it does not fix

The general form alone measured **535.1 us/layer against 511.2** on the nanogpt
forward case, a 4.7% regression, which is well outside the 1% the task allowed.
The cause is the V loader: it issues sixteen scalar global reads per tile at one
head-dim column each, and their row offsets `(4*group + j) * v_seq_stride` were
compile-time immediates when the row stride was the compile-time `HD`. With a
runtime stride each needs its own address add. A `DENSE: Bool` parameter, chosen
by a runtime `_is_dense` check on the strides, restores the immediates.

That recovered most of it but not all: the dense arm measures **518.4 rather
than 513.1, +1.05%**, and the following was measured rather than assumed:

| variant | us/layer |
|---|---:|
| pre-change kernel | 513.1 |
| `DENSE` arm, strides in the signature | 518.4 |
| `DENSE` arm, strides in the signature, body NEVER reads them | 517.9 |
| `DENSE` arm, strides removed from the signature entirely | 513.5 |
| `DENSE` arm, strides moved to the END of the signature | 518.4 |
| `RowStrides` fields narrowed to Int32 (36 bytes not 72) | 516.9 |
| two entry points, dense one keeping the old argument list, shared
  `@always_inline` body | 517.8 |

So it is the kernarg segment, not the indexing: three extra 24-byte structs cost
5 us/layer even when the body never touches them, the cost scales with wave
count rather than launch count (tiny +0.55 us, head1 +0.6, nanogpt +5.5), and
argument position does not matter. Splitting into two kernel entry points so the
dense one keeps a byte-identical argument list does not help either, because the
`@always_inline` body then costs what the kernargs did (517.8). This kernel is
already documented as latency-bound with nothing overlapping, and the journal
already records 513-vs-515 as a reportable effect here, so a 1% perturbation
from any code motion is what this kernel does.

**Rejected, with numbers:**

* **Single general path, no `DENSE`.** 535.1 vs 511.2, +4.7% on the harness.
  Rejected against the 1% budget, even though production takes the general arm
  anyway.
* **Int32 stride fields.** 516.9 vs 518.4 buys 0.3% and costs a silent
  correctness cliff above 2^31 elements per stride, which would need a host-side
  guard and a new decline path. Not worth it.
* **Two entry points with a shared `@always_inline` body.** 517.8, no better
  than the one-kernel form, and more code.

**Not attempted, and why it is the next thing to try.** The V loader's sixteen
scalar reads are the whole reason `DENSE` exists. The backward kernels already
avoid the equivalent problem by staging row-major into LDS and transposing out
of LDS (`kt_smem` from `k_smem`), which its docstring records as cheaper than a
transposing global loader. Doing the same for V in the forward would turn
sixteen scalar loads into two `dwordx4`, make the dense and strided paths cost
the same, and let `DENSE` be deleted. It needs `BN*KPAD` more LDS (26 KB total
at HD 64, 51 KB at HD 256 -- both fit) and one more barrier, and it would have
to be measured against 511.2 rather than 518.4.

## Correctness

**Against PyTorch-ROCm**, two processes because two accelerators in one trip an
autograd-engine assert, output plus all three gradients, fifteen shapes
including hd96, hd128, non-square, non-causal, 3x3, and six where the LEAF is a
`[B, T, H, D]` tensor and the op is handed `leaf.transpose(1, 2)` with a strided
`grad_output` to match. All fifteen take the fused node
(`_FusedFlashAttentionAutogradBackward`). Worst relative error 7.14e-03 against
a bf16 eps of 7.8e-03; the previous nine shapes are unchanged at 6.21e-03.

The 7.14e-03 is `view_hd96` dQ, and it is not ours. Against an FP32 CPU oracle
recomputing that case: out 2.47e-03 both sides, dQ **rocm 5.48e-03 / mojo
2.83e-03**, dK 3.64e-03 / 4.07e-03, dV 2.39e-03 both. The mojo-vs-rocm gap is
two independent bf16 roundings, and on the tensor that produced the worst number
we are the closer of the two.

**The harness gate now sees stride defects.** Five BTHD cases per harness, and
their oracles are told each operand's STORAGE LAYOUT rather than the stride
triple the kernel gets, so they work out where `(b, h, s, d)` lives from the
definition of a row-major array. Mutation test, kernels forced to index densely:
all five forward cases fail at 1.95 against a 0.031 bound; all five backward
cases fail by six to twelve orders of magnitude; every dense case still passes.
That is the property D10 says a shared-convention oracle cannot have.

## Test-visible state

`tests/test_eager_kernels.py`: 509 passed, 93 skipped, 1 failed --
`test_bf16_v3_source_dependency_and_kernel_contract`, which asserts a hardcoded
three-entry `_BF16_SOURCE_PATHS` list that now has four entries. Identical to the
recorded pre-existing state; unrelated to this work.

Two things worth recording for the next agent:

* `bf16_matmul_ops` and `tf32_matmul_ops` DO NOT COMPILE on this machine, at
  HEAD, with or without this change: `bf16_gemm_kernels.mojo` calls
  `mma(a=8xbfloat16, b=4xbfloat16, c=4xfloat32)`, which is the NVIDIA
  `m16n8k16` shape and has no gfx942 implementation. `_resolve_bf16_bridge`
  already degrades gracefully ("without compiling a known-incomplete module"),
  and no cached `.so` for either module exists for any source hash. Any edit
  under `eager_kernels/` invalidates the cache, so a loop that imports all of
  `_MOJO_MODULES` will die on `bf16_matmul_ops`; skip those two.
* `pytest -n 8` on this file fails 42 tests with `hipErrorOutOfMemory`, because
  each worker's MAX `DeviceContext` reserves a 172 GB pool. Run this file
  serially; it takes 4.5 s.

### D11: our SDPA output layout makes nanoGPT's re-assembly cost 1.84 ms

The copy audit found 48 gather kernels per step, 1843 us, that PyTorch-ROCm does
not issue. My first attribution was WRONG and worth recording as a lesson: I
claimed they were our own `_tc()` materializations of q/k/v. They were not --
those were counted inside the SDPA groups, because they happen under the
`_FusedFlashAttentionAutograd` user_annotation which the SDPA rows claim.
Removing them moved SDPA forward 7.753 -> 6.807 ms and SDPA backward 28.956 ->
28.326, while `Copies / dtype casts / layout` barely moved, 17.036 -> 16.874.
Attributing a kernel to a *group* is not the same as attributing it to a *cause*.

The real source is `clone [[48, 1024, 12, 64]]` -- nanoGPT's own

    y = y.transpose(1, 2).contiguous().view(B, T, C)

which is model code we do not control (AGENTS.md rule 5). The question is why
ROCm's is free, and the answer is the output layout:

| | output strides | `is_contiguous()` | that `.contiguous()` |
|---|---|---|---|
| ROCm | `(393216, 64, 768, 1)` | False | free no-op |
| ours | `(393216, 32768, 64, 1)` | True | real copy, 1843 us/step |

PyTorch's flash attention returns a BTHD-PHYSICAL tensor -- shaped `[B,H,T,D]`
but strided so that `transpose(1, 2)` is contiguous -- precisely so the universal
re-assembly idiom costs nothing. We return dense BHSD, so it costs a gather per
layer.

The fix is the mirror of the stride work just landed: that made the kernels READ
strided operands, this makes the forward WRITE a BTHD-physical `output` and the
backward write BTHD-physical `dq`/`dk`/`dv`, so the transpose on the way out and
its backward are both free. It is the right behaviour independent of nanoGPT --
any model using that idiom benefits, which is most of them.

### Standing after the stride pass

| | ROCm | before | after |
|---|---:|---:|---:|
| step, median of 20 | 156.73 ms | 172.10 | 170.64 (1.089x) |
| p10 / p90 | | 171.73 / 172.41 | 170.27 / 170.75 |
| FA forward harness | | 0.750x | 0.758x |
| FA backward harness | | 0.999x | 0.989x |

p10/p90 do not overlap the previous run, so the 1.46 ms is real. The forward's
+1% is attributed to KERNARG SIZE, not indexing: three extra 24-byte structs
cost ~5 us/layer even when the body never reads them, scaling with wave count
(measured across seven variants, including removing the fields entirely at 513.5
against keeping them unread at 517.9). Worth confirming independently before
treating as settled.

Operational notes for anyone re-measuring here: `bf16_matmul_ops` and
`tf32_matmul_ops` do not compile at HEAD (an NVIDIA `m16n8k16` MMA shape with no
gfx942 path; `_resolve_bf16_bridge` degrades gracefully, but a loop importing all
of `_MOJO_MODULES` dies there). And `pytest -n 8` on `test_eager_kernels.py`
fails ~42 tests with `hipErrorOutOfMemory` because each worker's MAX
`DeviceContext` reserves a 172 GB pool -- run that file serially, it takes 5 s.

## Change 33 — the dtype cast moves a vector at a time (15.03 -> 13.03 ms/step)

`_cast` in `data_movement_ops.mojo` went through `_parallel_for` ->
`elementwise[simd_width=1]`, and its closure took a `width` parameter and
ignored it. So a BF16 store used 2 bytes of a 16-byte access, and every element
paid its own address arithmetic and its own grid-stride iteration. nanoGPT 124M
issues 174 of these per step and spends 15.03 ms in them; PyTorch-ROCm's
equivalent is literally named `vectorized_elementwise_kernel<4, ...>`.

**Hypothesis.** A cast is pure bandwidth. Widen the access and the kernel
approaches the ~4.5 TB/s a streaming copy gets on this part. **Predicted:**
15.03 -> ~11.5 ms/step, i.e. ROCm parity.

**What was changed.** A dedicated `_cast_vec_kernel[src, dst, VEC]` enqueued
through `_enqueue_cached`, replacing the `elementwise` call on GPU (the CPU
device keeps it). `_parallel_for`'s shared `simd_width=1` is UNTOUCHED, which
matters: 41 call sites across seven modules pass it width-ignoring closures, and
raising the shared default would make every one of them write one element in N
and leave the rest uninitialized -- wrong tensors, no crash.

**VEC is a compile-time regime; which one runs is a runtime decision.** The
widest whose access is naturally aligned for BOTH base addresses, `16 //
max(itemsize)` down to 1. It has to be the addresses, not the closure's
`alignment` parameter: reading `_GridStrideKernel` in
`std/algorithm/backend/gpu/elementwise.mojo`, that parameter is passed
`Self.simd_width` -- a vector width in elements, not a promise about the
pointer. And a tensor really can start anywhere: `x[1:]` on FP32 is 4-byte
aligned, `a.contig` is still true, and `a.ptr` carries the offset.

**Remainders, confirmed rather than assumed.** Same file: with
`handle_uneven_simd` the packed region is still called at full width (for a
rank-1 shape `idx*w + w <= count` always holds) and the `count % w` tail is
called at width 1. Our kernel does its own: `nvec = size // VEC`, then at most
VEC-1 elements finished one at a time by the leading threads, which the grid
always has because it is at least one 256-thread block.

**The bool arm.** `dst == DType.bool` still compares `!= 0` elementwise, but a
`SIMD[bool, VEC]` store is not legal codegen -- an `i1` vector store crashes the
Mojo backend ("failed to run the pass manager"), and so does
`Scalar[f32].cast[DType.bool]()`. Torch's bool is one byte holding 0 or 1, so
the vector arm selects `uint8` 1/0 and stores through a byte pointer. (Two more
compiler landmines found on the way, recorded for whoever hits them:
`ctx.enqueue_memset` on a `DeviceBuffer[DType.bool]` also fails to run the pass
manager.)

### Grid geometry, which turned out to matter more than the vector width

A bandwidth-bound kernel has two regimes and `_gs_blocks`'s flat 4096-block cap
gets both wrong. Median of 40 x 20 launches, TB/s of read+write traffic,
`_cast_vec_kernel` at the widest vector:

| traffic | ceildiv(slots, 256) | 4096 blocks | 2 blocks per CU |
|---|---:|---:|---:|
| 14 MiB | 2.84 | 2.84 | 2.94 |
| 54 MiB | 3.30 | 3.30 | 3.60 |
| 216 MiB | 4.00 | 3.85 | **4.55** |
| 288 MiB | 4.20 | 4.19 | **4.34** |
| 384 MiB | **3.97** | 3.81 | 3.81 |
| 576 MiB | **4.03** | 3.74 | 3.76 |
| 14.8 GiB | **4.06** | 3.71 | 3.71 |

The crossover sits just above this part's 256 MiB of Infinity Cache, which is
the reading: while the operands are cache-resident a few long-lived workgroups
per CU beat many short ones, and once the copy streams from HBM the grid that
covers the slots exactly wins. Both arms are the same kernel; the grid-stride
loop just iterates more in the resident arm. It is now `_bw_blocks` in
`op_utils`, and `resident` is the caller's flag, not the helper's decision --
see the rejections.

The vector width alone is worth much less than that table suggests. At the SAME
grid on `[48, 1024, 768]` FP32->BF16, VEC=1 measured 49.7 us and VEC=4 measured
66.1 us -- the scalar kernel was FASTER. Reproducible across three repeats with
p10/p90 inside 1%. I do not have a mechanism for it: unrolling with the loads
hoisted ahead of the stores (2, 4 and 8 slots per thread) did not reproduce the
effect, so it is not simply memory-level parallelism per thread. Worth someone
else's attention, because it means the grid is the lever and the width is
mostly a way of reaching a grid that works.

### Result

Torch-level, both backends measured by the same script, median of 100 iterations
of a single `x.to(dtype)` (the protocol that reproduces the numbers this change
was scoped against), TB/s of read+write traffic:

| cast | ROCm | before | after | after/ROCm |
|---|---:|---:|---:|---:|
| logits [49152, 50304] fp32->bf16 | 4.03 | 3.44 | 3.82 | 0.95x |
| logits [49152, 50304] bf16->fp32 | 4.21 | 3.22 | 4.03 | 0.96x |
| acts [48, 1024, 768] fp32->bf16 | 3.68 | 2.78 | 3.26 | 0.89x |
| acts [48, 1024, 768] bf16->fp32 | 3.67 | 2.79 | 3.37 | 0.92x |

With 10 casts enqueued between syncs, which takes the per-call host cost out and
is closer to what a training step does:

| cast | ROCm | before | after | after/ROCm |
|---|---:|---:|---:|---:|
| logits fp32->bf16 | 3.99 | 3.44 | 3.79 | 0.95x |
| logits bf16->fp32 | 4.23 | 3.23 | 4.05 | 0.96x |
| acts fp32->bf16 | 4.51 | 3.49 | 4.29 | 0.95x |
| acts bf16->fp32 | 4.21 | 3.50 | **4.51** | 1.07x |

**The target was ROCm parity on all four rows and only one row reaches it.**
Two things are in the way and both are measured.

*The kernel is not the remaining gap.* The standalone Mojo probe runs the logits
FP32->BF16 kernel in 3650 us; rocprof puts ROCm's at 8 x 467.7 = 3741 us (it
splits a >2^31-element tensor into eight launches). Our kernel is the faster
one. Through torch the same cast takes 3911 us -- 261 us, 7.2%, more than the
probe. The acts cast shows the same 6% proportional excess (52.8 vs 49.7 us), so
it is not a fixed per-call cost; the same kernel simply runs ~7% slower inside
the process that holds MAX's 172 GB pool than in a standalone binary. That is a
lead for someone, not something this change can reach.

*Launch cost is the rest.* At batch=1 the acts cast costs 69.5 us for a ~50 us
kernel; ROCm's costs 61.6 us for a ~48 us kernel. ~8 us per call of ours, 148
acts casts per step. `_enqueue_cached` already removed `elementwise`'s per-call
`compile_function`; what is left is the spec boundary and the output allocation.

In production, where the casts are enqueued back to back, the GPU time is what
lands:

| | before | after |
|---|---:|---:|
| `_to_copy` [49152, 50304] | 8683.7 us/step | 7671.9 |
| `_to_copy` [48, 1024, 768] | 5384.4 | 4653.1 |
| `_to_copy` five weight shapes | 962.4 | 704.7 |
| **cast kernels total** | **15030.5** | **13029.7** |
| `Copies / dtype casts / layout` group | 16.874 ms | 14.863 |
| step, median of 20 | 170.64 ms | 168.03 (p10 167.83 / p90 168.47) |

The group is 1.30x ROCm's 11.422, down from 1.48x. 1.843 ms of it is still the
gather from nanoGPT's own `.contiguous()` (D11), so cast-to-cast it is 13.03
against 11.42, 1.14x, down from 1.32x.

### Rejections, with numbers

* **Non-temporal load/store hints** (`non_temporal=True` on either side or
  both). 3647.7 / 3650.6 / 3647.1 us against 3650.7 baseline on the logits
  cast, 50.4 / 50.5 / 50.2 against 50.3 on the acts cast. Inside noise on all
  six. The write-allocate pollution this was meant to avoid is not costing
  anything measurable.
* **Loads hoisted ahead of stores** (2, 4, 8 slots per thread into an
  `InlineArray`, then stored). Acts 4.05 TB/s at 8 slots against 4.50 for the
  plain loop at the tuned grid; logits 3.96 against 4.06. Rejected both ways.
* **Wider than 16 bytes per lane** (VEC=8 on FP32, a 32-byte load). Logits 4.06
  vs 4.06, acts 4.01 vs 4.00. No difference; not worth the extra
  instantiations.
* **Dropping the alignment gate and always taking the widest vector.** This is
  worth recording carefully because it *measures better*: an unaligned
  (`storage[1:]`) logits cast goes 3.44 -> 3.79 TB/s and the acts cast 4.17 ->
  4.24. And it is empirically correct here -- as a mutation, with the gate
  replaced by `if False`, all 46 cast tests still pass, because gfx942 executes
  a `dwordx4` at a 4-byte-aligned address. Rejected anyway: `alignment=16` on a
  4-byte-aligned pointer is a lie to LLVM that a future codegen change or a
  2-byte-aligned address is free to punish, and the regime does not occur in the
  target workload. **The test cannot catch this one**, which is the honest
  statement about the gate: it is a contract, not a behaviour the suite pins.
* **The same grid rule for the three-operand binary add** (`_bin_flat_vec_kernel`
  in `logic_ops.mojo`, which is already 16-byte vectorized -- the residual add's
  1.19x is NOT the scalar path, and neither is GELU forward's 1.21x
  (`gelu_forward_bf16_exact_vec16`)). Reverted. Its threads move 48 bytes, not
  24, and the arms land differently: at 216 MiB, which is the shape nanoGPT's
  residuals actually have, the existing 4096-block cap gives 4.09 TB/s against
  3.94 for two blocks per CU and 3.81 for the exact grid. The exact grid does
  win at 432 MiB (3.86 vs 3.65) and 864 MiB (3.90 vs 3.65), so there is
  something here, but not a rule I can defend on one shape. That is why
  `_bw_blocks` takes `resident` from the caller.

### How correctness was established

`tests/test_eager_kernels.py::test_fast_cast_is_exact_for_every_dtype_pair`:
all 7x7 `CAST_DTYPES` pairs, element for element with `torch.equal`, at counts
1, 3, 17, 1027 and 4099 (three of which are not a multiple of any vector width
the dispatch can pick) and base offsets 0, 1, 2 and 3 (which force the narrower
widths and the scalar arm). The pattern steps modulo 5, coprime with every
power-of-two width, so a vector whose lanes are rotated or whose tail is left
unwritten cannot pass. Plus
`test_fast_cast_float_rounding_matches_cpu`: FP32 -> BF16/FP16 of `randn` at two
counts and three offsets, exact against the CPU including the round trip back.

Mutation test of that gate: replacing the scalar tail with `if t < size and
False` fails 12 of the 46 cast tests -- every non-multiple count and every float
case, at every offset. The alignment gate, as above, it does not catch.

Suite: 553 passed, 93 skipped, 1 failed
(`test_bf16_v3_source_dependency_and_kernel_contract`, the recorded
pre-existing hardcoded-source-list failure).

### Operational note

Commit `1ab3354` ("Note the fill_/cross-entropy attribution split in the gap
report") accidentally swept up the in-progress `data_movement_ops.mojo` working
tree from this change; `ca887dc` carries the rest. The two together are the
whole change and HEAD is correct, but neither commit's diff is a readable unit
on its own. Two agents sharing one checkout need to stage by path.

## Change 34 — the fused attention writes BTHD, and the 48 gathers vanish (167.90 -> 166.51 ms/step)

D11 said our SDPA returned a dense `[B, H, T, D]` where PyTorch returns one
STORED `[B, T, H, D]`, and that nanoGPT's own re-assembly idiom

    y = y.transpose(1, 2).contiguous().view(B, T, C)

therefore cost a real gather per layer where ROCm pays nothing. Measured: 48
`data_movement_ops__run_gather` kernels, 1833.7 us/step, on `[48, 1024, 12, 64]`
-- one per layer for the forward's `y`, three per layer for the backward's
`dq`/`dk`/`dv`, which autograd reshapes through the same idiom's transpose.

**Hypothesis.** Write the same values to BTHD-physical addresses and declare the
layout truthfully. `is_contiguous()` becomes False, `o.transpose(1, 2)` becomes
contiguous, `.contiguous()` returns its own argument, and all 48 gathers become
views. **Predicted:** -1.84 ms/step, minus whatever the strided store costs the
kernels.

**What was changed.** The mirror of the stride-aware READ work in `a6d1b72` /
`80e8ebc`: the forward's `output` and the backward's `dq`/`dk`/`dv` each carry a
`RowStrides` triple now, so `enqueue_flash_attention_fwd` takes four stride
triples and `enqueue_flash_attention_bwd` takes eight. head_dim keeps stride 1 --
it is the axis the epilogues' 8-element vector store runs along -- so the store
instruction itself is untouched; only its address changes. On the Python side
`fast_fused_flash_attention_forward`/`_backward` allocate through a new
`_alloc_bthd`: ONE dense `[B, T, H, D]` allocation with a `[B, H, T, D]` strided
view over it. The view spans every element exactly once, so the bridge's zeroing
of `dq`/`dk`/`dv` by element count is still exactly the buffer.

`DENSE` stays a property of the READ operands only. The store address was already
a runtime multiply by `head_dim`, so taking the row stride out of a struct
instead costs nothing and there is no arm worth specializing.

**Measured, back to back, today.**

| | ROCm | before | after |
|---|---:|---:|---:|
| step, median of 20 | 156.73 ms | 167.90 | **166.51 (1.062x)** |
| p10 / p90 | | 167.62 / 168.14 | 166.14 / 166.78 |
| GPU kernel time / step | 156.79 | 167.62 | 166.04 |
| `run_gather` kernels | 0 | 48, 1833.7 us | **0, 0 us** |
| `Copies / dtype casts / layout` | 11.42 ms | 14.86 (1.30x) | **12.84 (1.12x)** |
| SDPA forward | 9.07 | 6.82 | 6.84 |
| SDPA backward | 33.96 | 28.38 | 28.57 |

p10/p90 do not overlap, so the 1.39 ms is real. The strided store is free on the
forward (+12 us/step over twelve layers) and costs +261 us/step on the backward's
dQ kernel, against 1833.7 recovered.

The harnesses' dense `nanogpt` case, three runs each alternating between a
binary built at HEAD and one built here: forward 517.8 / 517.9 / 518.5 against
518.5 / 518.3 / 517.8, i.e. 0.762x either way and indistinguishable; backward
2406.1 / 2406.7 / 2409.7 against 2411.0 / 2412.8 / 2414.4, 0.997x -> 0.999x, so
+5 us/layer for the three extra kernarg structs. (The 0.758x / 0.989x recorded in
the previous entry were single runs from an earlier session; HEAD re-measures
0.762x / 0.997x today, so read the pairs above rather than the difference from
those.)

**The gfx942 masked-tile schedule is a lottery, and it is worth 10-20%.** The
first version of this change measured the dQ kernel at 9869 us/step against
8565 before it -- a 15% regression from adding one stride struct the tile loop
never reads. What fixed it was writing the output base `stride * index` instead
of `index * stride`: two commutative multiplies, identical value, identical
instruction count, 8821 vs 9869 us/step in the production build. Seven other
formulations were measured on the forward in a fixed harness binary, all
semantically identical, spanning 563 to 686 us/layer, including one
(`o_base = q_base`, adding no arithmetic at all) at 672.

It is not register pressure and it is not the kernarg tax. VGPR 166 vs 164,
neither spilling; the same kernarg present but unread measures 574, i.e. free;
the unmasked tile loop is instruction-for-instruction identical between a fast
and a slow build. What differs is the AMDGPU schedule of the MASKED tile -- MFMA
placement around the barrier and the accumulator correction -- which is 2 of ~9
tiles per block on this shape.

And it re-rolls on changes that are not to the kernel at all: three harness
binaries built from ONE kernel source, differing only in which cases their case
list holds, run the nanogpt shape in the production layout at 562, 608 and 674
us/layer of GPU time (rocprofv3, same kernel symbol, same launch geometry). So a
harness A/B on the strided-read arm has a +-10% floor and cannot settle a 5%
question; only the step time can. That is now written at the top of
`flash_attention_fwd_kernels`, and both harnesses carry a `nanogpt_bthd` case --
the acceptance shape in the production layout -- so a compiler release that
re-rolls the dice is at least visible. Harness `nanogpt_bthd`: 685.7 us forward,
2463.3 backward, in the binary built at this commit.

**Proving the declared strides match the stored bytes.** The trap D11 warns about
is that autograd checks a returned gradient's SHAPE, not its strides, so a wrong
stride triple yields a permutation of the right values with the right shape and
every assertion passes. `fa_layout_probe.py` therefore never asks our own strided
copy where an element lives: it makes a rank-1 unit-stride view over the same
holder, so the D2H reads raw memory in memory order, and checks that the value at
the address the DECLARED layout implies is the value an independent FP32 CPU
attention puts there. Six shapes, output and all three gradients: the BTHD
reading is within 1.8e-02 of the FP32 reference and the DENSE reading of the same
bytes is off by 0.5 to 6.5 -- two to three orders of magnitude apart, which is
the separation a layout confusion would have to survive. It also asserts
`o.is_contiguous()` is False, `o.transpose(1, 2).is_contiguous()` is True,
`.contiguous()` returns its own argument, and `_materialize_contiguous` is never
called during the idiom. On the two shapes where BTHD and BHTD genuinely coincide
(`heads == 1`, `seq == 1`) the probe says so rather than claiming a check it
cannot make.

**The harness gate sees the new defect class.** Six forward cases with a strided
output (bit 3) and six backward cases with strided gradients (bits 5/6/7), whose
oracles are told each tensor's STORAGE LAYOUT and derive the reference's
addresses from the definition of a row-major array -- never the stride triple the
kernel gets, per D10. Mutation test, epilogues forced to store densely: all six
new forward cases fail at 1.91-1.98 against a 0.031 bound, all six new backward
cases fail by 15-30x their own bound on all three gradients, and every dense case
still passes.

**Correctness against PyTorch-ROCm** is unchanged: the fifteen shapes of
`fa_one.py`, output and all three gradients, worst relative error 7.14e-03
against a bf16 eps of 7.8e-03 -- the same number, on the same case (`view_hd96`
dQ, where an FP32 oracle says we are the closer of the two).

**What is left of D11.** Nothing of the gather; `Copies / dtype casts / layout`
is now 12.836 against ROCm's 11.422, and what remains is 174 kernels of genuine
dtype casts, not layout. The step gap to ROCm is 9.78 ms, 88% of it the two
Linear GEMM rows and the cross-entropy backward.

# nanoGPT Linear GEMM, MI300X — the two-tile k loop, 2026-07-26

Sixth pass over `harness/nanogpt_train/bench_linear_gemm.mojo`, all three
variants at once. Entry state, re-measured this session: step **166.54 ms**
(p10 166.11 / p90 166.75) against PyTorch-ROCm's 156.73, and one case per
process NN 1.058, NT 1.098, TN 1.417 on the harness / 1.176 for the copy-free
route production actually takes.

**Measurement protocol, and one thing that could not be honoured.** >= 25 warmups
and >= 100 individually synchronized iterations; one case per process; timings
never from a `--pmc` run. Clock pinning was requested mid-pass and **is not
available on this part**: `power_dpm_force_performance_level` is present and
mode-writable but the driver refuses the write (`EINVAL`, value stays `auto`),
`pp_dpm_sclk` is read-only, and `rocm-smi --setperflevel high` answers "Not
supported on the given system". Every number below is therefore UNPINNED, and
every before/after pair below was re-measured **interleaved case by case in one
session** -- base binary then new binary for the same case, back to back -- so
that both sides see the same thermal state. That is the strongest available
substitute for a pin and it is what the tables below report.

Also recorded because it invalidated an hour of numbers: a `sweep_gemm` process
killed by a tool timeout stayed resident holding **185 GB of VRAM** without
appearing in `ps`; the next `bench_nanogpt_train.py` died with
`hipErrorOutOfMemory`. `rocm-smi --showpids` finds it where `ps` does not.

## What actually binds these GEMMs, measured four ways

The three variants differ only in operand layout, so the first thing to settle
was what the layout costs. `_dense_mfma_route` was instantiated for all four
combinations on ONE shape, (49152, 768, 3072), same buffers, same all-ones fill,
only the strides the kernel is told about differing:

| layout | A | B | us | TFLOP/s |
|---|---|---|---:|---:|
| NN | k-major `(m,k)` | native `(k,n)` | **467.4** | 496 |
| AN | native `(k,m)` | k-major `(n,k)` | 482.2 | 481 |
| NT | k-major | k-major | 530.2 | 437 |
| TN | native | native | 530.6 | 437 |

**Both MIXED layouts are fast and both PURE layouts are 13% slower, by the same
amount.** That is not a resource story: counters on the three shipped
instantiations say the pure layouts are on opposite sides of every count.
`SQ_LDS_IDX_ACTIVE` is **identical** (2211840) for all three and
`SQ_LDS_BANK_CONFLICT` is 0 for all three; `SQ_INSTS_LDS` is NT 331776 < NN
414720 < TN 502272 and `SQ_INSTS_VALU` is NT 988128 < NN 1546848 < TN 2093760 --
i.e. NT has the FEWEST LDS instructions and the FEWEST VALU instructions and is
the slowest, and NN sits exactly at the midpoint of both counts and is the
fastest. `SQ_INSTS_VMEM` is NT 73728 < NN 101376 < TN 139968, so there is no
spill traffic in any loop either. The only counter that separates them the right
way is **`SQ_WAIT_INST_LDS`: NN 982007, NT 2001499** -- the pure layout spends
twice as long unable to issue an LDS instruction, with the same LDS work.

Two independent effects were then separated.

**Effect 1: a k-major operand loses ~20% per k tile when its row stride is a
multiple of 512 bytes.** At `BK = 32` a k-major tile row is 64 bytes, exactly
half a line, and the other half is the next k tile's. Sweeping k on
(49152, 768, n=768) with NT, in cycles per k tile net of the epilogue:

| k | row stride | stride/64 | cycles/k tile |
|---:|---:|---:|---:|
| 3072 | 6144 | 96 | 3580 |
| **3104** | 6208 | 97 | **2982** |
| **3200** | 6400 | 100 | **2989** |
| 3328 | 6656 | 104 | 3613 |
| 3584 | 7168 | 112 | 3654 |
| 4096 | 8192 | 128 | 3591 |

Fast exactly when the stride is NOT a multiple of 512 bytes, and k = 3104 runs
14.5% faster than k = 3072 while doing 1% more work. `TCP_TCC_READ_REQ` is
**identical per unit work** across the two (28311552 at k = 3072 and 28606464 at
k = 3104, which is 97/96 of it exactly), so the L1 behaves the same; `FETCH_SIZE`
is **482 MB against 380 MB**. The lines are fetched, evicted and refetched, and
the eviction is an address-mapping property. It only appears when the operand
exceeds the 256 MB Infinity Cache -- at k = 768 the same A is 75 MB and the
penalty is absent, which is why `mlp_c_proj_fwd` (A = 302 MB) was the one NT case
at 1.245 while the four k = 768 cases sat at 1.07-1.08.

**Effect 2: the pure layouts want a different k-loop body.** Below.

## Change 35 — two k tiles per trip when both operands share a layout

**Hypothesis.** The one-tile body refills the stage the NEXT trip will read and
swaps the two stage pointers, so every LDS address is a runtime value and each
trip has one refill. A body that runs two k tiles, one out of each stage, and
refills each stage right after the barrier that proves every wave has read it,
keeps the barrier rate at one per k tile (each of the two barriers does double
duty), fixes the stage pointers, and gives the scheduler a body with two
independent halves.

**Predicted effect.** Better on the layouts whose LDS issue is stalling, i.e.
the pure ones.

**Measured effect.** One shape per process, (49152, 768, 3072), and it is
layout-selective in exactly the way the diagnosis predicts:

| layout | one-tile body | two-tile body | best refill position |
|---|---:|---:|---|
| NT (both k-major) | 530.2 us | **460.0** | 3 of 4 k steps |
| TN (both native) | 531.9 | **476.7** | 2 of 4 |
| NN (mixed) | **467.4** | 477.8 | 3 of 4, still a loss |

It also only pays with the refill LATE in the MFMA sequence, which is why the
first attempt looked like a regression: at `FILL_AT = 0`, which is what the
shipped NT route used, the two-tile body measures 141.5 -> 153.4 us on
(49152, 768, 768) and 537.6 -> 588.0 on (49152, 3072, 768). At `FILL_AT = 3` the
same shapes are 141.4 -> 128.8 and 537.6 -> 480.7. Each trip has two refills, and
each needs queued matrix work in front of it.

`FILL_AT` sweeps for the accepted bodies, (49152, 768, 3072): NT 476.1 / 467.7 /
**460.0** at 1 / 2 / 3; TN 485.4 / **476.7** / 505.4; TN deep-k
(50304, 768, 49152) 7187.5 at 2 against 7581.7 at 3. NN with the two-tile body
was swept over all four positions (533 / 512 / 478 / 484) and never reaches its
own one-tile 467.

So `WBODY` is derived as `A_KMAJOR == B_KMAJOR`, with `FILL_AT` 3 for the
k-major pair and 2 for the native pair, and the mixed layout keeps what it had.
The launcher instantiates both bodies and picks at runtime, because the two-tile
body needs an EVEN number of k tiles in every slab -- `k_per % (2 * BK) == 0` --
which is a runtime property.

**Decision.** Accept.

**A defect the two-tile body introduced, and the gate that caught it.** The last
trip's refill of stage 1 is the one LDS write in the loop with no barrier after
it, and the tail read that stage without one. The all-ones gate passed (stale LDS
holds the same value), the chunk gate passed, and **`--pattern-check=1` failed on
all five NT cases**, 23.683 ms of the per-step total, because its operands are
nonzero only on the K edges. One `_nt_barrier()` before the tail's second read
fixes it. This is the second time this session's gate suite has caught something
an all-ones check cannot see; the pattern gate earns its keep.

## Change 36 — a wide output dtype reads one accumulator at a time

`rocprofv3` on the split-K TN instantiation reported **148 bytes of scratch per
thread** at VGPR 128. The epilogue reads all sixteen accumulators into a second
block before storing, to keep every read of an MFMA destination away from a
branch (journal D8); with a BF16 output that second block is 32 VGPRs on top of
the 64 live accumulators, but the split-K workspace is FP32 and it is 64 more.
A wide output dtype now reads ONE accumulator at a time, each read fenced on
both sides, which preserves the invariant at a sixteenth of the peak cost.
Scratch is 0 on that instantiation now. On its own it measured nothing (481.4 ->
481.4 us at (2304, 768, 49152)); with change 35 the same instantiation is 445.2.

## Change 37 — transpose the weight so the data gradient is a k-major pair

**Hypothesis.** The data gradient is the mixed layout, so change 35 leaves it
behind. Its B operand is the weight, `(k, n)`, and a k-major B is one transpose
away. The transpose moves `4 * n * k` bytes and the GEMM does `2 * m * n * k`
FLOP, so `n * k` cancels and the condition is a bound on m alone: requiring the
transpose to cost under a hundredth of the GEMM (a quarter of the smallest
measured gain) is `m >= 200 * GEMM_RATE / TRANSPOSE_BW`, about 30300 with the
3.3 TB/s change 28 measured and 500 TFLOP/s for the core.

**Measured effect.** This route against `_nt_mfma_route` on the same extents,
one shape per process:

| shape | mixed (NN) | k-major pair | gain |
|---|---:|---:|---:|
| (49152, 768, 768) | 138.9 us | 128.6 | 7.4% |
| (49152, 768, 2304) | 353.0 | 341.4 | 3.3% |
| (49152, 768, 3072) | 462.6 | 447.1 | 3.5% |
| (49152, 3072, 768) | 522.4 | 481.7 | 7.8% |
| (49152, 768, 50304) | 6729.8 | 6109.1 | **9.2%** (621.7 TFLOP/s) |

The transposes cost 2-47 us against those GEMMs, i.e. 0.6-0.8% of them. Every
harness number below has the transpose inside it. The k loop visits the same k
in the same order, so the outputs are bit-identical to the route it replaces.

**Decision.** Accept, gated on the derived bound on m, on the grid gate
`_nt_mfma_route` itself applies, and on the divisibility the vectorized transpose
needs. The GPT-2 prefill and decode shapes are all below the m bound and are
untouched.

## Where this pass ended

Interleaved case by case, base binary then new binary, 25 warmups and 100
synchronized iterations each, unpinned:

| case | variant | base us | new us | rocm us | base | new |
|---|---|---:|---:|---:|---:|---:|
| attn_c_attn_fwd | NT | 431.69 | 371.45 | 382.56 | 1.128 | **0.971** |
| attn_c_proj_fwd | NT | 142.32 | 129.14 | 158.05 | 0.900 | **0.817** |
| mlp_c_fc_fwd | NT | 538.11 | 481.91 | 498.14 | 1.080 | **0.967** |
| mlp_c_proj_fwd | NT | 536.02 | 447.50 | 426.62 | 1.256 | 1.049 |
| lm_head_fwd | NT | 8275.67 | 7305.41 | 7655.58 | 1.081 | **0.954** |
| attn_c_attn_dgrad | NN | 353.12 | 326.28 | 343.44 | 1.028 | **0.950** |
| attn_c_proj_dgrad | NN | 139.47 | 141.66 | 141.38 | 0.986 | 1.002 |
| mlp_c_fc_dgrad | NN | 467.71 | 414.80 | 434.10 | 1.077 | **0.955** |
| mlp_c_proj_dgrad | NN | 521.77 | 498.86 | 467.98 | 1.115 | 1.066 |
| lm_head_dgrad | NN | 6728.93 | 6148.95 | 6456.53 | 1.042 | **0.953** |
| attn_c_attn_wgrad | TN | 581.07 | 582.33 | 360.16 | 1.614 | 1.617 |
| attn_c_proj_wgrad | TN | 197.66 | 197.87 | 152.15 | 1.299 | 1.301 |
| mlp_c_fc_wgrad | TN | 656.24 | 656.11 | 460.34 | 1.425 | 1.425 |
| mlp_c_proj_wgrad | TN | 527.70 | 527.32 | 452.09 | 1.167 | 1.166 |
| lm_head_wgrad | TN | 9470.84 | 9351.98 | 6203.16 | 1.527 | 1.508 |

**NN 1.061 -> 0.984, NT 1.111 -> 0.969**, and the harness's TN is unchanged by
construction: it materializes A^T first, so its TN cases are the MIXED
instantiation, not the both-native one production runs. Timed directly, one shape
per process, interleaved:

| case | base us | new us | rocm us | base | new |
|---|---:|---:|---:|---:|---:|
| attn_c_attn_wgrad | 482.08 | 445.19 | 360.16 | 1.339 | 1.236 |
| attn_c_proj_wgrad | 171.84 | 154.35 | 152.15 | 1.129 | **1.014** |
| mlp_c_fc_wgrad | 509.11 | 490.59 | 460.34 | 1.106 | 1.066 |
| mlp_c_proj_wgrad | 508.04 | 488.05 | 452.09 | 1.124 | 1.080 |
| lm_head_wgrad | 7354.00 | 7191.86 | 6203.16 | 1.186 | 1.159 |

**Production TN 1.176 -> 1.121.** Weighted over what the step issues, the three
variants together are **8.34 ms of GEMM gap before and 1.68 ms after**
(79.97 -> 73.32 ms against 71.64).

Fifteen of fifteen cases pass all three gates on the whole table (default,
`--pattern-check=1`, `--chunk-check=1`), and so do fourteen deliberately awkward
shapes: an ODD k tile count (800, which takes the one-tile fallback), an even one
(832), exactly one and exactly two k tiles, m and n that 256 does not divide
(49160, 776, 1544), and the transposed-A cases of the same.

## What is left, measured

* **The weight gradient, 2.8 ms of the 3.3.** `attn_c_attn_wgrad` is 1.236 and
  `rocprofv3` splits it: GEMM 382.5 us and split-K reduction 49.4 us, the
  reduction reading 226 MB at 4.65 TB/s, which is above HBM and within 1% of what
  change 27 measured for the same traffic. Its 27 output tiles need 32 K slabs to
  fill the device and the slab search still says 32 is the best of the divisors of
  the k tile count: at the measured 3600 cycles per k tile, 16 slabs cost
  495 + 25 us and 64 cost 371 + 99 against 371 + 49 for 32. The reduction is
  structural at this tile size, and a 128x128 tile that would need no split needs
  2.7 GB of operand traffic in 400 us, i.e. 6.8 TB/s.
* **`lm_head_wgrad`, 528 TFLOP/s against ROCm's 612**, both operands native,
  591 tiles so no split. 3264 cycles per k tile against the 2048-cycle MFMA
  floor. An XCD band sweep on it says 1 band 7085.8 us / 2 bands 7142.6 / 4
  7139.9 / 8 7194.6 -- the shipped value (the XCD count, 8) is the worst by 1.5%,
  which is at the edge of the run-to-run spread and was not changed on one shape's
  evidence.
* **The 512-byte stride penalty on k-major operands** is now half fixed by
  accident: the two-tile body reads the same halves in the same trip. Loading them
  in ONE 128-byte global load per row was built and measured on top of it and
  REJECTED: 480 us against 460 for the plain two-tile body at (49152, 768, 3072),
  because the second staging vector per operand costs more than the line reuse
  recovers. `mlp_c_proj_fwd` at 1.049 is what remains of the effect.
* **Grid quantization, 5.3%**, unchanged and still needing Stream-K.

## Regression checks, all after both changes

| check | recorded | now |
|---|---|---|
| `bench_attention_bmm` default / pattern / causal | 51.234 / 51.037 / 50.490 ms | 51.217 / 51.003 / 50.473, 6/6 pass |
| `tests/test_eager_kernels.py`, serial | 553 passed, 93 skipped, 1 failed | **553 passed, 93 skipped, 1 failed** (`test_bf16_v3_source_dependency_and_kernel_contract`, pre-existing) |
| GPT-2 decode, five shapes | - | -0.9% to +0.5% |
| GPT-2 prefill, fifteen shapes | - | -10.4% to +0.4% |
| fourteen edge shapes x three gates | - | all pass |

## The step, and the clock that could not be pinned

`bench_nanogpt_train.py --device mojo --warmup 5 --iters 20`, base source and
HEAD measured **back to back in one session** with the eager cache rebuilt for
each, both `clocks_pinned: false` because the pin is refused on this VF:

| source | median | p10 | p90 |
|---|---:|---:|---:|
| base (d0fe31b) | 167.02 ms | 166.75 | 167.48 |
| HEAD | **162.39** | 162.09 | 162.65 |

**166.54 -> 162.39 ms against PyTorch-ROCm's recorded 156.73, i.e. 1.063 ->
1.036.** Two independent measurements of each side agree: the base measured
166.54 (p10 166.11 / p90 166.75) at the start of the session and 167.02 at the
end of it, and HEAD measured 162.59 and then 162.39 -- so the session's own
thermal drift on this workload is about 0.3%, which is the size of the effect the
clock pin was meant to remove and is an order of magnitude below the change.

The GEMM's own gap fell 8.34 -> 1.68 ms while the step fell 4.63 ms. The
difference is not accounted for here: the harness synchronizes every iteration
and the step does not, so 49 + 49 + 61 launches per step carry an overhead in the
harness numbers that the step overlaps. What the step says is the number that
counts.

# nanoGPT Linear GEMM, MI300X — macro tiles, and the slab count that was not free, 2026-07-26

Seventh pass over `harness/nanogpt_train/bench_linear_gemm.mojo`. Entry state,
re-measured: step **162.43 ms** against PyTorch-ROCm's 156.88, GEMM gap
**+1.89 ms** of which the weight gradient (TN, the copy-free both-native route
production takes) is +2.86 and the forward is -1.49.

The brief was the tile shape: ROCm dispatches fourteen Tensile kernels over this
step, with non-square macro tiles (MT256x224, MT192x224, MT512x128, MT128x512,
MT192x256), while this kernel runs one 256x256x32 geometry for everything. What
the sweep found is that the tile shape was **not** what the 5.3% grid
quantization was costing us -- the K SLAB COUNT was, and it was costing more.

**Protocol.** >= 25 warmups, >= 100 individually synchronized iterations, one
shape per process, every before/after pair interleaved base-then-new back to
back. Clock pinning is refused on this SR-IOV VF (`scripts/gpu_clock.py status`);
interleaving is the substitute. Timings never taken from a `--pmc` run. The gate
harness materializes A^T for its `transpose_a` rows, so its TN cases exercise the
MIXED instantiation; the copy-free route production takes is measured through a
separate driver that calls `_tn_mfma_route` directly.

## Diagnostic experiment AF — eight macro tiles, and which two are real

`_nt_mfma_gemm` was instantiated for fifteen (BM, BN) pairs at a constant 64x64
warp tile, BK=32 and two swizzled LDS stages, and swept against slab count on
(2304, 768, 49152) -- the wgrad shape whose 27 output tiles at 256x256 are the
worst grid in the step. Two stages of a `BM x BN` tile cost `128 * (BM + BN)`
bytes, so `BM + BN <= 512` is the whole feasible set at BK=32.

Best slab count per tile, us (all pass the all-ones check):

| tile | waves/wg | tiles | best us | model us | measured/model |
|---|---:|---:|---:|---:|---:|
| **256x256** | 16 | 27 | **443.4** (p32) | 443.8 | **1.00** |
| **128x384** | 12 | 36 | **380.0** (p8) | 382.0 | **1.00** |
| 256x192 | 12 | 36 | 419.3 (p8) | 370.6 | 1.13 |
| 320x192 | 15 | 32 | 545.4 (p8) | 428 | 1.27 |
| 192x320 | 15 | 36 | 553.2 (p8) | 428 | 1.29 |
| 384x128 | 12 | 36 | 503.9 (p8) | 382 | 1.32 |
| 192x256 | 12 | 36 | 489.5 (p8) | 371 | 1.32 |
| 128x320 | 10 | 54 | 526.7 (p16) | - | - |
| 192x192 | 9 | 48 | 801.4 (p16) | - | - |
| 128x256 | 8 | 54 | 557.0 (p16) | - | - |
| 128x128 | 4 | 108 | 535.4 (p16) | - | - |
| 64x384 | 6 | 72 | 544.6 (p4) | - | - |
| 64x448 | 7 | 72 | 611.0 (p4) | - | - |
| 384x64 | 6 | 72 | 751.1 (p8) | - | - |
| 64x256 | 4 | 108 | 885.9 (p8) | - | - |

The model column is `_nt_ktile_cyc`, which counts one k tile of a tile from the
geometry exactly as diagnostic experiment AD counted 256x256 (MFMA + LDS reads +
LDS writes + VMEM = 3840 cycles against 3995 measured), plus the traffic. **Only
256x256 and 128x384 reach it**; everything else is 13-32% short of its own bound.
Both of the two are `BM + BN == 512`, i.e. exactly the LDS budget, and both have
a wave count that is a multiple of four. That last point is not cosmetic: a
15-wave workgroup issues its MFMAs in four rounds of a SIMD each, so 320x192 does
3.75 rounds of work in 4 rounds of time, and it measures 27% off a model that
divides instead of ceiling. But it does not explain 384x128 (12 waves, 512 sum,
32% off) and 128x384 (same three properties, exact). That asymmetry is
unexplained; what it means practically is that a cost model can only be trusted
over a candidate set whose members have been measured to reach it, which is why
the shipped set is those two and not the six the model would happily rank.

**Decision.** Reject the six. Keep 256x256; add 128x384 for the pure layouts.

## Change 38 — a K slab that need not divide the k tile count

**Hypothesis.** `_nt_plan_cost` says the wgrad shapes are bound by grid fill, and
the slab count that fills the grid is usually not a divisor of `k / BK`. At
(2304, 768, 49152) the 27 output tiles want **eleven** slabs -- 297 of 304 CUs --
and 11 divides neither 1536 nor anything near it, so change 27's search over
divisors had to take 32 slabs (864 workgroups, three waves at 94.7%) and pay four
times the workspace traffic.

**Predicted effect.** `_nt_slab_tiles` rounds the slab up to an EVEN number of k
tiles and the last slab takes what is left, which keeps both k-loop bodies legal
(the two-tile body needs an even count in every slab, and an even slab length
makes the remainder even exactly when the total is). The model then predicts 361
us at eleven slabs against 443.8 at thirty-two.

**Measured effect.** One shape per process, TN both-native, 256x256:

| slabs | k tiles/slab | waves | us | model us |
|---:|---:|---:|---:|---:|
| 8 | 192 | 1 | 472.0 | 470.3 |
| 9 | 172 | 1 | 433.7 | 428 |
| 10 | 154 | 1 | 396.5 | 390 |
| **11** | **140 + 136** | **1** | **369.8** | **361** |
| 12 | 128 | 2 | 622.0 | 631 |
| 32 (divisor, shipped before) | 48 | 3 | 443.4 | 443.8 |

The model is within 2.5% at every point and its argmin is the measured argmin.
Eleven slabs beat the best divisor by **16.6%**, and the reason is visible in the
two terms: the same GEMM work in one wave instead of three ragged ones, and 155 MB
of workspace traffic instead of 453.

**Decision.** Accept. The kernel derives its own slab's tile count as
`min(kt_per, ktiles - slab * kt_per)`, so the short slab is a runtime property of
`block_idx.z` and nothing else changes.

## Change 39 — one cost model chooses the tile and the slab count together

`_nt_plan_cost(bm, bn, m, n, obytes, ktiles, parts, cus)` is
`ceil(tiles * parts / cus) * slab_tiles * ktile_cycles + traffic / rate`, with
every shape term a runtime value and two calibrated constants: 1.67 GHz and
4.05 TB/s, **fitted from two points of one shape** (256x256 at 8 and at 32 slabs
on (2304, 768, 49152), which differ 4x in workspace bytes and 3x in wave count).
Both constants are physically sensible and consistent with what change 27
measured for the reduction alone. Predictions against measurement, none of them
fitted:

| shape, layout | plan | model | measured |
|---|---|---:|---:|
| (2304, 768, 49152) TN | 128x384 p8 | 382.0 | 380.0 |
| (3072, 768, 49152) TN | 256x256 p8 | 480.0 | 488.2 |
| (3072, 768, 49152) TN | 128x384 p6 | 500.0 | 491.6 |
| (768, 768, 49152) TN | 256x256 p32 | 147.9 | 152.7 |
| (50304, 768, 49152) TN | 256x256 unsplit | 7082 | 7199.7 |
| (49152, 768, 768) NT | 256x256 unsplit | 129.0 | 129.0 |

It is 6-18% PESSIMISTIC on the well-filled NT/NN shapes with a long k, which beat
their own resource bound (some overlap does happen there), and that is recorded
as a known inaccuracy: it never changes a decision on those shapes because every
alternative plan is charged the same way, and no split ever wins for them.

The search offers each candidate tile every slab count up to 64, subject to three
guards that cost silent wrong answers or measured losses before they were found:

* **The last slab must be non-empty** (`(parts - 1) * slab < ktiles`). The
  reduction reads every plane; a slab with nothing to do would feed it a plane
  nobody wrote, and the two-tile body would run two k tiles outside its own slab.
* **A slab of at least 8 k tiles** (`NT_MIN_SLAB`). A workgroup pays two k tiles
  of prologue and a sixteen-accumulator epilogue that the model does not charge.
* **At least four k tiles of total work per CU** (`NT_MIN_WORK`). Splitting K
  cannot conjure work: measured against the routes this one declines to, the
  256x256 tile loses at 0.5, 1.0 and 1.9 k tiles per CU ((512, 768, 768)
  28.6 -> 48.2 us, (1024, 768, 768) 43.2 -> 51.1, (512, 3072, 768) 47.7 -> 58.3)
  and wins from 7.6 up.

That third guard replaced change 27's "a slab of at least 1024 k ELEMENTS", which
was excluding wins as well as losses. With it, the GPT-2 prefill NN shapes -- one
shape per process, interleaved -- go:

| shape | before | after |
|---|---:|---:|
| (768, 768, 3072) | 97.5 us | **61.1** (declines now; the MAX multistage kernel is 50.0 of it) |
| (1024, 768, 3072) | 99.8 | **80.6** (declines) |
| (1536, 768, 3072) | 101.4 | **69.0** (splits 3 -> 8 slabs) |
| (2048, 768, 3072) | 102.1 | **71.6** |
| (4096, 768, 3072) | 109.2 | **90.9** |
| (1024, 2304, 3072) | 105.7 | **81.3** |
| (4096, 2304, 768) | 109.2 | **87.0** |
| (4096, 3072, 768) | 151.1 | **131.3** |

and the decode shapes are unchanged (they decline in both, +-0.5%).

**Where the second tile is selected, and why it is NOT shipped.** Nowhere in this
workload: with the slab count free, 256x256 wins all fifteen cases. It is
selected -- correctly, and measured -- where 256 quantizes an extent that 384
divides. At (2304, 1152, 49152) TN, one shape per process, interleaved:
**620.6 -> 566.9 us**, and the model's pick (128x384 at eleven slabs, 564.9
measured alone) is the best of seven plans measured on that shape, ahead of
128x384 p5 (577.5), 256x256 p6 (622.3, which is what a divisor search would take)
and 256x256 p11 (707.6).

It is shipped with two conditions the sweep established -- **pure layouts only**
and **split-K only** (`min_parts = 2`) -- and it was very nearly dropped instead,
on a compile-time regression that turned out to be a warm compiler cache in the
baseline (see the measurement note below: the base source is 13 minutes cold too,
and both objects contain the same fourteen kernels). It costs nothing and it
covers the shapes a different sequence length or hidden size brings; it changes
nothing about the fifteen this workload issues, which is verified twice over --
the gate table is identical with and without it, and `rocprofv3` shows 256x256 on
`attn_c_attn_wgrad` and 128x384 only on (2304, 1152, 49152).

## Where this pass ended, per case

Production TN (the copy-free both-native route), one shape per process, base
binary then new binary back to back, 25 warmups and 100 synchronized iterations:

| case | base us | new us | rocm us | base | new |
|---|---:|---:|---:|---:|---:|
| attn_c_attn_wgrad (2304, 768) | 445.22 | **372.44** | 360.16 | 1.236 | **1.034** |
| attn_c_proj_wgrad (768, 768) | 154.35 | 154.71 | 152.15 | 1.014 | 1.017 |
| mlp_c_fc_wgrad (3072, 768) | 489.77 | 492.52 | 460.34 | 1.064 | 1.070 |
| mlp_c_proj_wgrad (768, 3072) | 488.21 | 490.88 | 452.09 | 1.080 | 1.086 |
| lm_head_wgrad (50304, 768) | 7196.73 | 7198.64 | 6203.16 | 1.160 | 1.161 |

**Production TN 1.121 -> 1.087**, i.e. 802 us of the step, all of it on the one
case whose plan changed. The other four keep the plan they had -- the same kernel
symbol at the same grid, verified with `rocprofv3` on `mlp_c_fc_wgrad`
(3072x12x8, eight slabs, 461.25 us against 458.93) -- and measure 0.2-0.6%
slower, which is the AMDGPU codegen lottery this journal has recorded three times
and is the reason the step, which reproduces to 0.24%, is the number quoted.

The gate table, all fifteen cases, three variants (default, `--pattern-check=1`,
`--chunk-check=1`), on the shipped source: **15/15 pass all three**, NN
1.018/1.022/1.009, NT 0.943/0.948/0.942, TN (the harness's mixed instantiation,
with the copy) 1.369/1.363/1.371. Chunk coverage was also run through the
production TN route on fourteen shapes including six the workload does not have --
an odd k tile count (49184, which forces the one-tile body in every slab), an even
ragged one (49216), m and n that neither 256 nor 384 divides (2312, 776, 1160), and
the three that select the second macro tile -- all 0 mismatches.

The one shape whose plan the second tile changes, interleaved on the shipped
binary: (2304, 1152, 49152) TN **621.6 -> 567.6 us**.

## The step

`bench_nanogpt_train.py --device mojo --warmup 5 --iters 20`, base source and HEAD
measured back to back in one session with each side's eager cache already built
(the cache is content addressed, so the pair costs one run each):

| source | median | p10 | p90 | order |
|---|---:|---:|---:|---|
| base (8dd3d04) | 162.37 ms | 162.07 | 162.52 | first |
| this pass | 161.48 | 161.25 | 161.86 | second |
| **this pass, shipped** | **161.25** | **160.96** | **161.54** | first |
| base (8dd3d04) | 162.40 | 162.05 | 162.79 | second |

**162.37 -> 161.25 ms against PyTorch-ROCm's 156.88, i.e. 1.035 -> 1.028.** Both
orders were measured, base-then-new and new-then-base, because the pin is refused
and whichever side runs second sees the warmer part: the two base runs agree to
0.02% and the four runs of this pass in the session span 161.25-161.70 (0.28%), so
the -1.1 ms is four times the spread. The GEMM's own gap fell 1.89 -> 1.09 ms,
which is the same 0.8 ms the interleaved per-case table shows.

## What is left, and why the tile shape cannot reach it

Every remaining GEMM gap except one is the same 5.3%, and the 5.3% has a name:
**304 = 16 x 19, and 94.7% is 18/19 exactly.** Every macro tile whose extents are
multiples of 64 gives these shapes a tile count of the form `2^a * 3^b` -- 576,
1728, 2304, 27, 36, 48 -- and none of those is ever divisible by 19, so the last
wave of workgroups is always short by the same 1/19 no matter which tile or slab
count is chosen. It is not a tile-shape problem and no tile-shape search can
close it.

* **The two mlp weight gradients, 0.86 ms.** 36 output tiles x 8 slabs = 288 of
  304 CUs, which is 18/19; the model's floor at that fill is 480 us against
  ROCm's 460 and our 490.
* **`lm_head_wgrad`, 0.99 ms.** 591 tiles, no split, 97.2% fill, 528 TFLOP/s
  against ROCm's 612. This one is NOT fill: it is at its resource bound (3902
  measured cycles per k tile against the model's 3840), so closing it needs the
  bound itself to move -- fewer LDS read cycles per unit of output area, which
  diagnostic experiment AD closed at this warp tile.
* **Stream-K**, still the only route to the 1/19, and now with a number on it.
  The fixup traffic is `nsplits * BM * BN * 4` bytes written and read once, which
  for 304 workgroups of a 256x256 tile is 155 MB, i.e. about 38 us at the 4.05
  TB/s this pass fitted. That pays only where 5.3% of the kernel exceeds 38 us,
  i.e. above roughly 720 us per launch: of the 75 GEMM launches in a step, three
  qualify and two of them have the fill to gain -- `lm_head_dgrad` (+288 us
  modelled) and `lm_head_wgrad` (+164). The other 72 launches are 130-500 us and
  would each LOSE. That is a ~0.45 ms prize for a new work decomposition, a fixup
  kernel and three gates, and it is why this pass did not take it.

## Measurement note — a compile-time "regression" that was a warm compiler cache

This nearly cost the second macro tile, and it is recorded because the trap is
cheap to fall into. `matmul_ops` is the module every eager first use compiles, so
its build time is a real cost, and `mojo build --emit shared-lib` on the file the
eager importer compiles said:

| source | real | user |
|---|---:|---:|
| base (8dd3d04) | 6m33s | 7m25s |
| this pass, second tile in both forms | 13m12s | - |
| this pass, second tile split-only | 13m23s | 24m08s |
| this pass, second tile removed entirely | 12m49s | 24m09s |
| this pass, plan search capped at four slab counts | 13m23s | 24m08s |
| this pass, ragged clamp reverted | 13m06s | 24m23s |
| **base with ONE COMMENT LINE ADDED** | **13m04s** | 24m27s |

The last row is the control and it settles it: **the base source is 13 minutes
too.** The 6m33s was a warm compiler cache -- that exact source had been compiled
minutes earlier in the same session by a `bench_nanogpt_train.py` run, and the
`user`/`real` ratio gives it away (1.1 cores against 1.9 for every cold build).
Two corroborating facts that should have been checked first: the shipped `.so` and
the base `.so` differ by 880 bytes in 16.7 MB, and `strings` finds **the same
fourteen `nt_mfma` kernels in both** -- the shipped source generates no new device
code at all for the shapes the workload has.

So this pass costs nothing in compile time, the second tile costs nothing either
(13m12 against 13m04 is inside the spread of the three cold builds), and it is
shipped. Three intermediate conclusions in the two sections above were drawn from
the bad baseline and are wrong: the tile was never expensive, the plan search's
trip count was never suspect, and the ragged clamp was never implicated.

## Regression checks, all on the shipped source

| check | recorded | now |
|---|---|---|
| `bench_attention_bmm` default / pattern / causal | 51.217 / 51.003 / 50.473 ms | **51.233 / 51.023 / 50.472**, 6/6 pass |
| `tests/test_eager_kernels.py`, serial | 553 passed, 93 skipped, 1 failed | **553 passed, 93 skipped, 1 failed** (`test_bf16_v3_source_dependency_and_kernel_contract`, pre-existing), 903 s |
| gate table x three variants | 15/15 pass | **15/15 pass** |
| GPT-2 decode NN/NT, five shapes | - | -0.5% to +1.1%, all declining in both sources |
| GPT-2 prefill NN, eight shapes | - | **-13% to -37%** (the table under change 39) |
| GPT-2 prefill NT, three shapes | - | +-1.1% (`_nt_mfma_route` is untouched) |
| production TN chunk coverage, fourteen shapes | - | 0 mismatches |
| `matmul_ops` cold compile | 13m04s (base, cold) | **12m49-13m23s** |

The test suite and the step were measured on the source without the second macro
tile; the tile was restored afterwards on the compile-time correction above, and
what that restores is code no shape in either run reaches (`strings` on the two
objects finds the same fourteen kernels, the gate table is identical, and
`rocprofv3` confirms 256x256 on every wgrad case the step issues). The gate's
three variants and the fourteen chunk-coverage shapes were re-run on the shipped
source.

`_nt_mfma_route`, the NT (both-k-major) entry the forward and the transposed-B
data gradient take, is deliberately NOT changed by this pass: it is at 0.943 and
it has neither the plan search nor split-K. That is the obvious next move and it
now has evidence behind it -- the same change on the mixed NN route, which shares
the underfilled-grid regime, is what produced the 13-37% on the GPT-2 prefill
shapes above. It was left out here to keep one pass to one mechanism.

# nanoGPT bandwidth-bound cluster, MI300X — the grid, and one re-read, 2026-07-26

The five memory-bound rows of the v11 gap report were all 1.15-1.22x and the
brief's reading was that one shared cause was likely. It is half true: two of
the five had a shared cause (a launch geometry tuned for a 132-SM H100 running
on a 304-CU part) and it was worth 2.70 ms; the other three did not, and two of
them did not move at all. Everything below is measured on gfx942 with the clock
unpinned, so every kernel number is a median of 20 launches inside one process
and every step number is interleaved against the other binary.

## Where the five rows ended

Kernel-level, `current_bench_train/v13/comparison` against the same
`v7/rocm`:

| target | rocm | v11 | v13 | v11/rocm | v13/rocm |
|---|---:|---:|---:|---:|---:|
| Copies / dtype casts / layout | 11.422 | 13.080 | 13.162 | 1.145 | 1.152 |
| Cross entropy (backward) | 4.424 | 7.256 | 5.839 | 1.640 | **1.320** |
| Residual / elementwise add | 4.473 | 5.320 | 5.191 | 1.189 | **1.161** |
| Cross entropy (forward) | 4.138 | 4.817 | 3.549 | 1.164 | **0.858** |
| GELU (forward) | 2.180 | 2.658 | 2.658 | 1.219 | 1.219 |

Per kernel, us/step in production:

| kernel | v11 | v13 |
|---|---:|---:|
| `log_softmax_rows_block_bfloat16_1024` | 4804.7 | **3537.5** |
| `lsm_bwd_nosmem_bfloat16` -> `lsm_bwd_reg_bfloat16_1024_8` | 5276.4 | **3843.3** |
| `logic_ops__bin_flat_vec_kernel` (fp32 residual add, x25) | 2729.1 | **2619.9** |
| `elementwise_r1_w1_b256` (the scalar `add_`, x76) | 401.6 | 379.4 |
| `gpu_memcpy`, whole step | 77 x, 377.9 | **2 x, 9.4** |
| `gelu_forward_bf16_exact_vec16` (x12) | 2658.2 | 2657.8 |
| whole step, kernels only | 162192.4 | **159031.2** |

## The step, interleaved

One session, alternating binaries, 25 warmups and 100 synchronized iterations
each, clock NOT pinned (`scripts/gpu_clock.py status`: the driver refuses):

| order | binary | median | p10 | p90 |
|---|---|---:|---:|---:|
| 1 | new | 159.740 | 159.141 | 160.384 |
| 2 | **base (HEAD~1)** | **162.530** | 161.749 | 163.056 |
| 3 | new | 159.933 | 159.190 | 160.492 |
| 4 | PyTorch-ROCm | 157.168 | 156.912 | 157.762 |
| 5 | new | 160.021 | 159.377 | 160.771 |

New median of three: 159.93. **1.0176x ROCm, from 1.0341x.** The gap went 5.36
ms -> 2.76 ms, i.e. 2.60 ms of the 4.55 ms this pass was scoped against. The
base measured 162.53 here against the 161.46 the brief quotes; both were taken
without a pinned clock and the difference between them is why the interleave
exists.

## Change 40 — the log-softmax forward's L2 budget starves a 304-CU part

`_log_softmax_rows` capped its grid at `LSM_L2_BUDGET // row_bytes` so that a
row survived from its pass-1 read to its pass-2 re-read. The budget is 23 MB,
chosen for an H100's 50 MB L2. At nanoGPT's logits row (50304 bf16 columns,
100608 bytes) that is **228 blocks on a part with 304 CUs** — three quarters of
one block per CU.

**Hypothesis.** The re-read the cap protects is worth less than the parallelism
it costs. **Predicted:** the same kernel at a device-filling grid runs at the
3-stream rate instead of the 2-stream-but-starved one.

Measured, same kernel, same operands, 20 launches, us:

| blocks | 228 | 608 | 912 | 1216 | 1824 | 2432 | 4096 | 8192 | 16384 | 49152 |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| us | 4460 | 3536 | 3570 | 3538 | 3517 | 3509 | 3515 | 3494 | 3498 | 3545 |

Flat within 2% from 2 blocks per CU to 54, and 27% worse at the cap. The cap now
has a floor of `LSM_BLOCKS_PER_CU * sm_count` = 1216 under it, applied only on
the long-row (1024-thread) arm — the short-row arm's cap already admits >= 920
blocks, so leaving it alone costs nothing and risks nothing. Production:
4804.7 -> 3537.5 us/step, and the cross-entropy-forward row goes from 1.164x
ROCm to **0.858x**.

For scale, the same probe puts a plain 2-stream bf16 copy of this tensor at 2455
us and a 3-stream add at 3942, so at 3509 the kernel is doing three streams'
worth of traffic (its re-read partially hits cache) at close to the achievable
rate. There is no further grid win here.

### Rejected: a register-staged forward that reads the row once

The obvious next step is to keep the row in registers across the max, the sum
and the store, which makes the kernel 2-stream and removes the residency
question entirely. It also halves the transcendentals: with x in registers the
maximum reduces with no `exp` at all, and the sum then needs exactly one `exp`
per element against the final block maximum, instead of the online recurrence's
two plus a rescaling multiply.

A clean draft of exactly that (`_v2` in the probe, no head/tail handling, 8
slots of 16 bytes at 1024 threads) measured **3383 us at 4096 blocks and 3322 at
1216** — better than the 3509 above, and a no-`exp` control measured 3006, so
the transcendentals are worth ~380 us of it.

**It did not survive being made general, and the reason is worth recording.** A
kernel that accepts any `cols` needs a per-row scalar head and tail, and the
natural way to write them is the grid-stride loop the rest of the file uses.
Same arithmetic, same grid, same slots:

| head/tail written as | us |
|---|---:|
| `while` grid-stride loops (3 pairs) | 4508 / 4546 / 4562 / 4588 |
| `if` (they can run at most once) | 3857 / 3872 / 3924 |
| `if`, with the two scalars also staged in registers | 4582 / 4589 / 4592 |
| the clean draft with no head/tail at all | 3322 / 3383 |

The first row is a spill: a runtime trip count between the phases makes the
register allocator treat the 8-slot cache as live across a back edge. The second
row fixes that — head and tail are each shorter than one 16-byte vector, so
fewer than 32 elements against at least 256 threads, and the loop provably runs
at most once per thread — and recovers 640 us. The third row is the surprise:
adding **two** fp32 registers to hold the head and tail values (which removes
four conditional global loads) puts it straight back into the spilling regime.

So the general register-staged forward tops out at 3857, and the existing
re-read kernel at a filled grid is 3509. **The register kernel was deleted.**
The 535 us between the general version and the clean draft is entirely head/tail
plumbing that the codegen will not absorb, and I could not find a formulation
that keeps both. A compile-time `head == 0` arm would, but that is a fast path
plus a fallback, which AGENTS.md rules out, and the whole prize is 350 us.

## Change 41 — the log-softmax backward reads `grad` once, from registers

`_log_softmax_bwd_nosmem_kernel` is the fallback for rows too long for the 64 KB
LDS the staging variant needs, and it read `grad` twice: four passes over
`[49152, 50304]` where three will do. It also ran at **256 threads with one
block per row**, so a CU held up to eight 100 KB rows in flight and the second
read had no chance of hitting cache.

Two separate effects, measured separately at that shape, 20 launches:

| variant | us |
|---|---:|
| shipped: re-read, 256 threads, grid = rows | 5558 / 5578 / 5577 |
| re-read, **1024 threads**, grid = rows | 4473 / 4474 / 4487 |
| register-staged, 1024 threads, grid 1216 | **4157 / 4174 / 4186** |

The block size alone is worth 1100 us — the concurrent row footprint drops 4x,
from ~243 MB (over this part's 256 MB of Infinity Cache) to ~60 MB — and staging
the row in 8 register slots is worth another 300. The 3-stream floor here is the
3942 us the plain 3-stream add reference gets, so the staged kernel is 6% off
it and there is little left.

The staged kernel is instantiated at 1, 4 and 8 slots; the path is only reached
when shared memory cannot hold the row (n_vec > 3072), so 4 and 8 are the arms
that run and the 1-slot arm exists for `cols < VEC`, which lands here with an
empty vector body. Rows longer than 8 * 1024 vectors keep the re-read kernel,
now at 1024 threads. Grid is flat from 608 to 49152 blocks (4178-4193 us), so it
uses the same 4-blocks-per-CU rule.

Production: 5276.4 -> 3843.3 us/step. The row is still 1.32x ROCm, but as the
brief noted most of that row is an attribution artifact: measured directly,
`_log_softmax_backward_data` is 4095.2 (ROCm) against 3843.3, i.e. **0.94x — we
now win it** — and `nll_loss_backward` is 329.2 against 1996.1 because ours also
does the zeroing ROCm spends 2006.7 us of `aten::fill_` on. Honest total for
cross entropy backward: 6431.1 ROCm against 5839.4.

## Change 42 — the scalar `add_` stops allocating and copying back

Not a kernel problem. nanoGPT's fused AdamW calls `torch._foreach_add_(steps, 1)`
twice per step; PrivateUse1 has no `_foreach_add_.Scalar`, so ATen decomposes it
into **75 `add_.Scalar` on 0-dim fp32 tensors**. Each of those went through the
functional spec (allocate a 4-byte output, run the op, `copy_d2d` it back), so
the step carried 75 one-element kernels *and* 75 four-byte device-to-device
copies. ROCm does the whole thing in two `multi_tensor_apply` launches, 10.4
us/step.

`AddScalarInplace` / `MulScalarInplace` in `elementwise_ops.mojo` write straight
into the input when it is contiguous and floating point, and
`_try_spec_scalar_inplace` in `aten_fast.py` reaches for them from
`fast_aten_add_` and `fast_aten_mul_` before the functional path. `alpha` folds
into the scalar exactly.

**Measured:** `gpu_memcpy` over the whole step goes from **77 events, 377.9
us/step, to 2 events, 9.4 us/step**. The 76 one-element kernels remain (379.4
us/step): removing those needs a real `_foreach_add_.Scalar` registration and a
batched-descriptor kernel, which is a bigger change than this pass took on and
is the obvious next 370 us.

Note for whoever reads the gap report: these 76 launches are attributed to
"Residual / elementwise add" because `compare_nanogpt_train_kernels.py` charges
a kernel to its *deepest* enclosing CPU range, and ours is the decomposed
`aten::add_`. On ROCm the deepest range is `aten::_foreach_add_`, which the same
script files under the optimizer. The two backends' add rows are not comparable
as printed.

## Change 43 — the flat binary kernel covers its slots above the cache

`_bin_flat_vec_kernel` took `_gs_blocks`, a flat 4096-block cap. Change 33's
rejection note measured that cap as the best choice at 216 MiB and the exact
grid as the best at 432 MiB, and declined to pick. nanoGPT's fp32 residual add
is 453 MB of traffic — the 432 MiB case — and the measurement reproduces:

| grid | 608 | 1216 | 2432 | 4096 | 9216 | 18432 | 36864 (exact) |
|---|---:|---:|---:|---:|---:|---:|---:|
| us | 114.5 | 114.5 | 120.1 | 121.7 / 126.9 | 118.9 | 116.6 | **114.0** |

`_bw_flat_blocks` in `op_utils` now switches on the operand traffic against the
256 MiB last-level cache: below it, `_gs_blocks` unchanged; above it, the exact
grid. Production: 2729.1 -> 2619.9 us/step. This is a 4-6% effect on a kernel
with a noisy grid response, so it is the least defensible thing in this pass;
the step-level evidence for it is only that the total moves the right way.

## What did NOT move, and the measurement behind it

**GELU forward, +0.478 ms, unchanged.** The asymmetry the brief pointed at
(forward 1.22x, backward 0.93x) is real but it is not geometry and it is not the
vector width. On `[48, 1024, 3072]` bf16, 40 launches:

| variant | us |
|---|---:|
| shipped: 32 bytes/thread (VEC=16), one shot, 36864 blocks | 252.1 / 253.7 / 264.7 |
| 16 bytes/thread (VEC=8), one shot, 73728 blocks | 252.0 / 254.4 |
| VEC=8, grid-stride, 608 / 1216 / 2432 / 4096 / 9728 / 19456 blocks | 310 / 268 / **248.6** / 261.8 / 258.8 / 259.5 |
| VEC=16, grid-stride, 1216 / 4096 / 18432 | 275.6 / 271.4 / 269.2 |
| VEC=8, one shot, 512-thread blocks | 247.7 |
| VEC=4, grid-stride, 4096 | 280.8 |

Everything sane is within 2% of the shipped kernel, which is inside the codegen
lottery's noise band; VEC=8 and VEC=16 are bit-identical (checked, 0 differing
elements over 151M). The cost is `erf`. Replacing `gelu(x)` with `x * 0.5` in
the same kernel, same grid: **175.5 us at VEC=16 and 155.8-164.1 at VEC=8**, so
`erf` is ~90-100 us of the 252, and ROCm's whole kernel is 181.65 us. The
stdlib's `erf` evaluates BOTH branches unconditionally (a 7-term polynomial plus
an `exp` for |x| > 0.921875, a 6-term polynomial below it) and selects. A
wave-uniform skip of the expensive branch is the obvious idea and it does not
work here: the branch point is |x| > 1.3036 in GELU's argument, one wave covers
1024 elements at VEC=16, and no wave is entirely below it. **This row needs a
cheaper `erf`, not a different launch.** The bf16 output has 8 mantissa bits, so
there is enormous accuracy slack to spend, but spending it changes results
against CPU torch and that is a decision with a test contract attached.

**Copies / dtype casts, +1.658 ms, unchanged (13.080 -> 13.162, noise).** I did
not attempt this. Change 33 already established that our cast *kernel* is faster
than ROCm's standalone (3650 vs 3741 us on the logits cast) and that the gap is
~7% of in-process excess plus ~8 us of per-call launch cost over 148 calls.
Nothing in this pass addresses either. One thing worth recording for the next
reader, because it is not obvious from the report: the 7.7 ms/step of
`_to_copy [49152, 50304]` is **autocast casting the log-probabilities to fp32
for `nll_loss` and the gradient back to bf16**, and PyTorch-ROCm pays exactly
the same thing (`aten::copy_ [[49152, 50304]]`, 6851.1 us/step). It is 29.6 GB
of traffic that exists only because `_log_softmax` is called with
`half_to_float=False`. `aten::_log_softmax(self, dim, half_to_float=True)`
exists precisely to fuse it, and PyTorch's autocast does not use it here — so
this is ~7 ms that BOTH backends spend and that a backend willing to implement
the `half_to_float` arm and an `AutogradPrivateUse1` formula for
`cross_entropy_loss` could delete outright. That is a much larger prize than the
1.66 ms of relative gap, and it is not something a kernel change can reach.

## Correctness

92 new cases in `tests/test_eager_kernels.py`:

* `test_fast_log_softmax_wide_rows_match_cpu` — 3 dtypes x 7 column counts
  (12501, 16384, 16385, 25000, 33000, 50304, 70001) x base offsets 0 and 1,
  against CPU `torch.log_softmax`. The offsets slice the storage so the row
  bases stop being 16-byte aligned, which is the only way to make the per-row
  scalar head and tail non-empty; four of the seven counts are not a multiple of
  any vector width the dispatch can pick. Each asserts no NaN.
* `test_fast_log_softmax_wide_rows_stay_finite` — a row that is constant except
  for one +60 spike (the shape defect D7 came from: a running maximum seeded
  with `Float32.MIN`, which is -inf, turns an idle thread's `0 * exp(m - m)`
  into a NaN the block reduction spreads over the row), and a fully constant row
  whose every output must be `-log(cols)`. The constant row is the cheap
  independent check that the denominator is the whole row and not a fragment,
  and it is not an all-ones test: the spiked row is not constant and the
  randn rows above are not either.
* `test_fast_log_softmax_backward_wide_rows` — 3 dtypes x cols in {3, 33000,
  50304, 70001} x offsets 0 and 1. cols == 3 is below one 16-byte vector, so the
  vector body is empty and the row is entirely head/tail (the 1-slot arm); 33000
  and 50304 walk the 4- and 8-slot arms; 70001 is the re-read fallback.
* `test_fast_scalar_inplace_add_and_mul` — 4 dtypes (including int32, which must
  keep falling through to the functional path) x 4 shapes including 0-dim, three
  scalars each, both `add_` and `mul_`, asserting the returned object is the
  input.
* `test_fast_scalar_inplace_add_alpha_and_aliasing` — `alpha` folding, a
  non-contiguous target that must decline into the strided path, and a row view
  whose write must land in the parent's storage.
* `test_fast_binary_add_above_last_level_cache` — 24000003 fp32 elements, 288 MB
  of traffic, so `_bw_flat_blocks` takes the streaming arm; the count is not a
  multiple of the 4-element vector, so the scalar tail rides that grid too.

Independent-of-the-kernel checks used while developing, in the standalone probe:
the register-staged forward was compared bitwise against the shipped kernel at
the same block size (0 mismatches over 2.47e9 elements at [49152, 50304], and at
(17, 50304), (101, 1027), (3, 8191), (5, 1), (300, 129)) and against a host
fp64 reference (max relative error 3.9e-3, which is bf16 rounding). The staged
backward was compared bitwise against the shipped 256-thread kernel: 0
mismatches at every shape, worst absolute difference 0.0.

## Regression checks, all on the shipped source

| check | recorded | now |
|---|---|---|
| `tests/test_eager_kernels.py`, serial | 553 passed, 93 skipped, 1 failed | **645 passed, 93 skipped, 1 failed** (`test_bf16_v3_source_dependency_and_kernel_contract`, the recorded pre-existing hardcoded-source-list failure), 88 s |
| `bench_attention_bmm` default / pattern / causal | 51.233 / 51.023 / 50.472 ms | **51.243 / 51.014 / 50.465**, 6/6 pass |
| gate table x three variants | 15/15 pass | **15/15 pass**, NN 1.014/1.021/1.009, NT 0.943/0.947/0.942, TN 1.368/1.361/1.370 |
| `ruff check` / `ruff format` on the two Python files touched | - | clean |

## What is left in these five rows, in order

1. `erf` in the GELU forward: ~90 us x 12 = 1.1 ms of pure transcendental, of
   which the 0.478 ms gap to ROCm is the part that is a gap. Needs a cheaper
   polynomial and a decision about matching CPU torch bit for bit.
2. The 76 one-element `add_` kernels, 379 us: register
   `aten::_foreach_add_.Scalar` for PrivateUse1 with a batched-descriptor
   kernel, modelled on `fast_aten__foreach_mul__tensor`. ROCm does it in 10.4 us.
3. The cast group's 1.66 ms: 7% in-process excess plus ~8 us/call over 148
   calls, both already diagnosed in change 33 and neither addressed.
4. The 7 ms of logits casting that both backends pay for autocast's
   `nll_loss` promotion, which `half_to_float=True` was designed to delete.

# nanoGPT GELU and AdamW step counters, MI300X — 2026-07-26

Three targets were briefed with a measured value each: an NT plan search
(+2.49 ms), a cheaper exact GELU (+0.48) and a batched `_foreach_add_`
(+0.35). Two of the three paid. The first is a dead end and this pass has the
measurement that closes it rather than the argument that suggests it.

Entry state, re-measured interleaved in this session: **158.82 ms** against
PyTorch-ROCm's **156.88** (two runs each, both orders). Protocol throughout:
>= 25 warmups, >= 100 synchronized iterations for kernel numbers, every
before/after pair interleaved, never a `--pmc` run. `clocks_pinned: false`.

## Change 44 — the exact GELU in one branch, no divide (252 -> 197 us)

**Hypothesis.** The brief attributed ~90-100 us of the 252 us kernel to
`std.math.erf` and had already excluded every launch geometry (VEC 4/8/16,
grid-stride, 512 threads: all within 2%). Reading the stdlib source adds a
second culprit the brief did not have: `nn.activations.gelu` computes
`x / 1.4142135623730951`, and `pop.div` carries no `arcp`, so LLVM may not fold
a divide by a non-power-of-two into a multiply and gfx942 expands it to the
IEEE `v_div_scale` / `v_rcp` / 3x`v_fma` / `v_div_fmas` / `v_div_fixup`
sequence -- about ten instructions per lane, sixteen lanes per thread. And
`erf` evaluates BOTH polynomial branches unconditionally and selects with
`v_cndmask`; a wave-uniform skip cannot fire because the branch point is
|x| > 0.921875 and one wave covers 1024 elements. Counted per lane, the shipped
path is ~34 VALU ops plus one `v_exp_f32`.

**The reformulation.** Not the `tanh` approximation -- nanoGPT asks for
`nn.GELU()` and must get the exact erf form. What changes is how it is
evaluated:

    gelu(x) = relu(x) - 0.5 * |x| * erfc(|x| / sqrt(2))

For `x >= 0` the second term is `x - x*Phi(x)`; for `x < 0` it is `-x*Phi(x)`
with `relu` contributing nothing. One branch of `erfc` therefore serves both
signs, and writing that `erfc` as `exp2(-a * R(a))` and fitting `R` directly in
the log2 domain absorbs the `sqrt(2)`. What is left per lane is 8 FMAs, one
`v_exp_f32` and five other VALU ops: the divide, the second polynomial, the
select, the `copysign` and the `1 - exp(...)` are all gone.

`R(a) = -log2(erfc(a/sqrt2))/a` is fitted degree 8 over `a` in [0, 12.75] by
IRLS in a Chebyshev basis against 50-digit `mpmath`, weighted by `a` because
the error that matters is `a * dR` -- an absolute error in the exponent is a
relative error in the answer. 12.75 is placed where accuracy stops mattering,
not where the inputs stop: `gelu(-13.9)` is 4e-43, under the smallest BF16
subnormal. Past the clamp the polynomial argument saturates while the exponent
keeps the unclamped `a`, so the tail keeps decaying, and the clamped prefactor
is what makes `gelu(+inf) == +inf` fall out with no guard.

**Measured effect.** Standalone probe, production shape (49152 x 3072 =
151003136 elements), 25 warmups / 100 iterations, both orders:

| order | shipped `gelu` | this | ratio |
|---|---:|---:|---:|
| old first | 268.54 us | 197.50 | 0.735 |
| new first | 257.27 | 201.79 | 0.784 |

i.e. **~262 -> ~200 us**, against a measured memory floor of 175.5 us for the
same kernel with `gelu(x)` replaced by `x*0.5`. Twelve calls a step.

**Accuracy.** The device writes its FP32 result for every one of the 65536 BF16
bit patterns to a file and the host compares against 50-digit `mpmath`.

| range | this: max abs | max rel | wrong BF16 | shipped: max abs | max rel | wrong BF16 |
|---|---:|---:|---:|---:|---:|---:|
| all finite BF16 | 2.476e-6 | 15.3 (at x=-13.19) | **14** / 65280 | 3.831e-7 | 1.000 | **198** / 65280 |
| `|x| <= 12.5` | 2.476e-6 | 2.001e-5 | 2 | 3.831e-7 | 1.000 | 186 |
| `|x| <= 3.1` (the whole observed input range) | 2.476e-6 | 1.544e-5 | 2 / 32910 | 2.109e-7 | 1.779e-5 | 1 / 32910 |

The last row is the one that decides it, and the range in it is measured, not
assumed: a forward hook on all twelve `mlp.gelu` modules over 75497472 real
activations gives min -3.078, max 3.078, std 0.553, and **zero** samples past
|x| = 5. Inside that range the two are equivalent to within round-to-nearest
ties -- 2 BF16 values against 1, of 32910 -- and this one has the SMALLER worst
relative error.

Outside it, this form is the more accurate one, by a lot, and for a structural
reason: `0.5*x*(1 + erf(x/sqrt2))` forms `1 - (1 - eps)`. `1 + erf` is
quantized by the FP32 epsilon at 1.0, so the shipped path collapses to exactly
zero below about x = -5.2, while BF16 still resolves the true value down to
x = -13.7. That is where 95 of its 198 wrong roundings live. This form never
forms that difference; it computes the small quantity directly.

Two deliberate divergences, both recorded:

* `gelu(-inf)` is `-0.0` here and NaN on CUDA (and in the shipped path), because
  CUDA reaches `-inf * 0`. `-0.0` is the limit of the true function. The frozen
  H100 special-semantics test is updated for the exact mode and left alone for
  the tanh mode; it cannot run on this part.
* Twelve BF16 inputs near x = -13 (true value under 1e-36) are less accurate
  than the fit is, because `exp2` bottoms out at the smallest FP32 normal and
  the AMD `v_exp_f32` clamps rather than returning a subnormal, so those come
  out too LARGE. See the rejection below.

### Rejected: an exponent bias to reach the last twelve values

`exp2` bottoms out at 2^-126, which this expression reaches at about x = -12.7.
Moving 24 binades out of the exponent and into the prefactor costs nothing --
the multiply that forms the exponent becomes an FMA -- and it did fix those
twelve. It also **broke 2304 others**: scaling the prefactor by 2^-24
underflows it for a SUBNORMAL input, where `0.5*|x|` IS the answer, so every
BF16 input around 1e-39 returned `x` instead of `x/2`. Measured 2330 wrong BF16
values against 14. Reverted; the twelve stay wrong and the comment says why.

## Change 45 — `aten::_foreach_add_.Scalar`, 75 launches to one

**Hypothesis.** There is no PrivateUse1 entry, so ATen runs the
CompositeExplicitAutograd fallback, which is a sequential `add_.Scalar` per list
element. `torch.optim.AdamW(fused=True)` calls
`torch._foreach_add_(device_state_steps, 1)` once per param group; nanoGPT's two
groups hold 50 and 25 one-element FP32 step counters, so a step pays 75 launches
to add 75 floats. Change 42 already removed the allocation and the 4-byte D2D
copy from each of them; the launches themselves remained, at 379 us/step against
ROCm's 10.4.

**Implementation.** `fast_aten__foreach_add__scalar` validates the whole list
before anything is enqueued (ATen's mutable-TensorList contract is
all-or-nothing), flattens it to a tuple of `(address, numel)` ints -- no host or
device array is allocated -- and `ForeachAddScalar` in `elementwise_ops` batches
it into `InlineArray[ForeachDesc, FOREACH_DESC_CAP]` and launches one grid over
the CONCATENATION of the list, one block per 65536-element chunk, with
`_chunk_bounds` walking the descriptors' prefix sums to turn a flat block index
back into (tensor, range). The arithmetic is the same widen-add-narrow as
`_scalar_elementwise[dtype, SOP_ADD]`, so the result is bit-identical to the
`add_.Scalar` path it replaces, and the descriptor cap bounds one launch
argument rather than the list.

Four things return `NOT_HANDLED` and keep the fallback, which is also what
defines the semantics: a non-float or mixed dtype, a strided tensor, a bool
scalar, and **a list whose members alias** -- a duplicate entry has to be added
twice, in order, and one grid over the concatenation would race instead.

**Measured effect.** Not separable from change 44 at the step level; the two
were measured together. The launch count is verified directly: a spy on the
bridge sees exactly one call for a 75-tensor list, three calls for three steps.

## The step, interleaved

`bench_nanogpt_train.py --device mojo --warmup 5 --iters 20`, alternating the
two sources with the base cache pre-built (the eager cache is content
addressed, so both survive):

| order | source | median | p10 | p90 |
|---|---|---:|---:|---:|
| 1 | this pass | 157.87 ms | 157.55 | 158.22 |
| 2 | base (9ad8c86) | 158.86 | 158.43 | 159.30 |
| 3 | this pass | 157.64 | 157.29 | 157.98 |
| 4 | base (9ad8c86) | 158.78 | 158.39 | 159.08 |

and PyTorch-ROCm re-measured in the same session, interleaved against this
pass: **156.82 / 156.93** (with this pass at 157.99 between them).

**158.82 -> 157.79 ms against 156.88, i.e. 1.0124 -> 1.0058.** The two base runs
agree to 0.05% and the three runs of this pass span 157.64-157.99 (0.22%), so
the -1.03 ms is five times the spread. The remaining deficit is **0.91 ms**.

## Target rejected — the NT route's missing plan search is worth zero here

The brief's largest target was `_nt_mfma_route`, which has no plan search and no
split-K, on the reading that the same change on the NN route produced 13-37% on
the GPT-2 prefill shapes. It cannot pay on THIS workload, and the reason is the
shapes: the NT forward is `(m=49152, n, k)` with k = 768 or 3072, so `m` alone
gives 192 tile rows and every one of the five shapes ALREADY covers the device
at 256x256 (576 to 37824 tiles against 304 CUs). Splitting K there cannot reduce
total k work, only add workspace traffic, which is R8.

Measured rather than argued. A probe calls `_nt_mfma_gemm` directly at the only
two macro tiles that reach `_nt_ktile_cyc` (diagnostic experiment AF), 256x256
measured on both sides of 128x384 so neither ordering gets the warmer part, 25
warmups / 100 iterations:

| shape | 256x256 p1 | 128x384 p1 | model 256 | model 384 | search picks |
|---|---:|---:|---:|---:|---|
| attn_c_attn_fwd (49152, 2304, 768) | **365.88** us | 402.05 | 387 us | 409 | 256x256, p=1 |
| attn_c_proj_fwd (49152, 768, 768) | **128.83** | 153.68 | 129 | 151 | 256x256, p=1 |
| mlp_c_fc_fwd (49152, 3072, 768) | **482.06** | 548.14 | 516 | 560 | 256x256, p=1 |
| mlp_c_proj_fwd (49152, 768, 3072) | **445.89** | 501.79 | 460 | 548 | 256x256, p=1 |
| lm_head_fwd (49152, 50304, 768) | **7304.06** | 8586.69 | 8119 | 8549 | 256x256, p=1 |

256x256 wins every shape by 9.9-19.3%, the two tiles agree bit for bit on a
sparse signed pattern (0 mismatches of up to 2.47e9 outputs), and
`_nt_best_parts` returns `parts = 1` for BOTH tiles on every shape. So the plan
search, wired in, would select exactly what `_nt_mfma_route` already ships. The
model was re-validated on this regime as the brief required: it is 0-11%
pessimistic on 256x256 and within 2% on 128x384, i.e. it UNDER-states the
winner's margin, so its ranking here is conservative in the right direction.

This does not say the plan search is worthless -- it says it is worthless for
nanoGPT. `_nt_mfma_route` still DECLINES outright whenever
`ceildiv(m,256)*ceildiv(n,256) < cus`, which is the GPT-2 prefill NT regime the
journal records at +-1.1%, and that is where the NN route's 13-37% came from.
It is a real target for a different workload and it is not this one.

### The 2.49 ms the brief attributed to the forward, and where it actually is

Worth recording because it is where the +2.49 came from. The v13 gap report has
`Linear GEMM (forward)` at 26.459 ms for us against ROCm's 23.969, over 49
kernels each. The gate, on the same 49 launches, has us at **23.771 ms** and
ROCm at 25.240 -- i.e. 0.942, and we WIN. Both cannot be right about the same
kernels. Clustering the 441 `nt_mfma` durations in the v13 trace puts the whole
in-situ excess on the two lm_head GEMMs (7732 + 8994 us in the trace against
7301 + 6143 in the gate, +24%) while the other 96 launches are within 3% of the
gate. Our totals across the two instruments agree (78.8 ms profile against 79.3
gate for all fifteen cases); ROCm's do not (67.5 against 71.6). So the
forward/backward SPLIT of that report is not measuring what the gate measures,
and 2.49 ms of "forward gap" is not a number a forward-route change can collect.
Not chased further -- it needs a fresh profile, not a kernel change.

## Regression checks, all on the shipped source

| check | recorded | now |
|---|---|---|
| gate table x three variants | 15/15, NN 1.014/1.021/1.009, NT 0.943/0.947/0.942, TN 1.368/1.361/1.370 | **15/15 pass**, NN 1.020/1.025/1.016, NT **0.942**/0.952/0.942, TN 1.368/1.362/1.369 |
| `tests/test_eager_kernels.py`, serial | 645 passed, 93 skipped, 1 failed | see below |
| `matmul_ops` | untouched by this pass | untouched |

`bf16_matmul_ops` does not compile on this toolchain
(`no valid implementation of mma for a=8xbfloat16, b=4xbfloat16, c=4xfloat32` in
`bf16_gemm_kernels.mojo`); it has never been in this cache and is the
pre-existing `test_bf16_v3_source_dependency_and_kernel_contract` failure. It is
not affected by this pass.

## What is left in these two rows

1. GELU forward is now ~2.0 ms/step against ROCm's 2.18, i.e. it has crossed.
   The floor is 175.5 us/call and we are at ~200, so ~0.3 ms of ALU remains
   above the memory bound; a degree-7 fit over a shorter range would take one
   FMA off and is worth about 7% of the compute.
2. The step counters are one launch. What the fused-AdamW row still spends is
   its own kernel, already at 0.81x.

## Faster than PyTorch-ROCm at batch 12

nanoGPT GPT-2 124M, block 1024, bf16 autocast, AdamW, grad clipping, one full
training step. Five interleaved pairs, 8 warmups / 30 timed iterations each,
alternating which backend runs first so a heating trend cannot favour either
side. Clocks are NOT pinned -- the driver refuses `setperflevel` on this SR-IOV
VF -- which is exactly why the protocol is interleaving rather than sweeping.

| round | PyTorch-ROCm | mojo | ratio |
|---|---:|---:|---:|
| 1 | 47.031 | 46.956 | 0.9984 |
| 2 | 47.025 | 46.563 | 0.9902 |
| 3 | 47.032 | 46.969 | 0.9987 |
| 4 | 47.106 | 46.809 | 0.9937 |
| 5 | 47.036 | 46.693 | 0.9927 |
| **mean** | **47.046** | **46.798** | **0.9947** |

Every mojo sample (46.563-46.969) is below every ROCm sample (47.025-47.106);
the distributions do not overlap. Loss after one step agrees to 8.68e-07
relative, against a bf16 epsilon of 7.8e-03.

Ratio across the batch sweep, same protocol, one pair each:

| batch | rocm | mojo | ratio | tiles at n=768 | fill of 304 CUs |
|---|---:|---:|---:|---:|---:|
| 8 | 34.76 | 38.06 | 1.0949 | 96 | 32% |
| 12 | 47.04 | 46.74 | 0.9936 | 144 | 47% |
| 16 | 58.58 | 61.81 | 1.0551 | 192 | 63% |
| 24 | 83.27 | 85.59 | 1.0279 | 288 | 95% |
| 32 | 106.94 | 113.39 | 1.0603 | 384 | 1.26 waves |
| 48 | 157.01 | 158.39 | 1.0088 | 576 | 1.90 waves |

The shape of that curve is not allocator pressure -- it is not monotonic in the
working set. It tracks how the output tile grid lands against 304 CUs, which is
where our two macro tiles lose to hipBLASLt's fourteen.

### The MAX pool reservation costs nothing (hypothesis rejected)

MAX reserves ~84% of the 205.8 GB card by default via
`MODULAR_DEVICE_CONTEXT_MEMORY_MANAGER_SIZE_PERCENT`, and change 33 had recorded
the same cast kernel running ~7% slower in-process than standalone. Sweeping the
knob at batch 48: 25% -> 159.07, 50% -> 158.07, default -> 158.41 and 158.27,
99% -> 158.22 ms. All within ~1 ms and the SMALLEST pool is the SLOWEST, so the
reservation is not the cost. The standalone probe reuses its buffers every
iteration and is TLB-warm in a way the real step is not; that is the whole
discrepancy. Recorded so nobody re-runs it.

### A measurement bug that wasted wall-clock, worth not repeating

Two queued sweeps never ran. `until ! pgrep -f "bench_nanogpt|..."` matches its
OWN command line, so the wait loop waited on itself; then `pkill -f "until !
pgrep"` matched the command line doing the killing and killed the job it had just
launched. `pgrep`/`pkill -f` see the whole command line including your wrapper.
Match the script path, not a substring you are also typing.
