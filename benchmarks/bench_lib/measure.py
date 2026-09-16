"""Device-time measurement for one benchmark case.

Rules this module enforces (they were learned painfully):

* GPU/DEVICE time only, never wall time.  On CUDA/ROCm both legs are timed
  from torch.profiler kernel events (CUPTI / roctracer): the sum of all
  device-kernel durations in a burst, memcpy/memset excluded, divided by
  the burst's iteration count.  On Apple there is no public per-kernel GPU
  counter, so device time is a saturated-queue proxy: synchronize, enqueue
  the whole burst, synchronize, divide — valid because the queue is deep
  enough that the GPU, not the dispatching CPU, is the bottleneck.  The
  hardware key records which method produced a ratio.
* Every leg is pre-warmed before anything is timed, so JIT kernel builds
  and clock ramp never land inside a timed region (an under-warmed probe
  once read 1.43x where the truth was 1.02x).
* The two legs are interleaved burst by burst, and the order FLIPS every
  pair — ref, ours | ours, ref | ref, ours ... — so consecutive pairs are
  an ABBA quartet.  The reported ratio is the MEDIAN OF PER-PAIR RATIOS
  (ours burst / adjacent ref burst): throttle drift moves both bursts of
  a pair together, so pair ratios stay stable even when absolute per-leg
  times swing hard (the S7 lm_head shapes throttle the card by ~30%
  across a run).

  The flip is what makes a MONOTONIC ramp cancel.  Under fixed ABAB order
  the "ours" burst always sits one burst later than the ref it is divided
  by, so a steady drift of rate r biases every pair by about r * (settle
  + burst) in the SAME direction; a median over pairs cannot remove a
  bias that every pair shares.  Worse, the bias is invisible to this
  module's own stopping rule, which watches scatter — a constant offset
  contributes none, so the sampler happily reports 0.3% uncertainty on a
  number carrying a systematic tilt.  Flipping the order puts the "ours"
  burst before its ref in every other pair, so the two lags are equal and
  opposite and the first-order term cancels across the quartet.  Higher
  orders (curvature of the ramp) survive; the core-clock pin in clock.py
  is what keeps those small.
* Sampling is spread-aware: burst pairs keep being added until the
  uncertainty of the median pair ratio (robust scatter / sqrt(n)) drops
  under TARGET_UNCERTAINTY_PCT, capped at MAX_BURSTS so the suite always
  terminates.  A quiet case stops at MIN_BURSTS; a throttling case buys
  itself more samples, longer bursts (MIN_BURST_US escalation) and a
  settle lead-in per burst.  If the cap is reached with the uncertainty
  still above RATIO_NOISE_LIMIT_PCT the caller must refuse the number
  (check.py retries after a cooldown, then fails the node) instead of
  silently recording a value that cannot resolve the regression
  threshold.
* Within-run sampling cannot see BETWEEN-process variance: device clock
  policy is sampled once per process and is then stable, so determinism
  across processes is owned by the core-clock pin in bench_lib/clock.py,
  applied by check.py around this module's timed region.
* A handful of cases (large-batch bmm NT/TT, some tf32 addmm) are not just
  noisy, they are BIMODAL: the stock leg's kernel-selection heuristic settles
  on one of two algorithms and a run can sample bursts from either or both.
  A short run that happens to land only in one mode reports a tight,
  low-uncertainty ratio that still disagrees with an equally confident run
  that landed in the other mode (see gemm-baselines-are-unreproducible).  The
  ordinary noise/uncertainty machinery above cannot see this: it is built for
  scatter around ONE center and, once enough bursts pile up in a single
  mode, "converges" early on whichever mode got sampled first.  Detecting a
  two-cluster (bimodal) split in the pair ratios and refusing to stop until
  enough bursts have crossed it (BIMODAL_MAX_BURSTS, well past the ordinary
  MAX_BURSTS) reports the population median across both modes instead of a
  coin flip — a number that reproduces across process launches because the
  long-run mode-mix is a property of the hardware/driver, not of when the
  measurement happened to stop.  This costs extra bursts ONLY for cases that
  actually show the pattern; the common unimodal case is unaffected.
"""

from __future__ import annotations

import contextlib
import dataclasses
import fcntl
import math
import statistics
import time
from collections.abc import Callable, Iterator

from torch.profiler import ProfilerActivity, profile

WARMUP_ITERS = 6
# Burst pairs, always sampled in ABBA quartets: both bounds are EVEN and the
# sampler only stops on an even count, because it takes two oppositely
# ordered pairs to cancel a monotonic ramp.  Stopping at 3 would leave a
# third of the drift bias in the reported number.
MIN_BURSTS = 4  # a quiet case stops here
MAX_BURSTS = 12  # hard cap so the suite terminates on a noisy case

# Convergence is judged on the sampling uncertainty of the REPORTED number
# (the median of the pair ratios), i.e. robust pair scatter / sqrt(n): that
# is what must resolve the regression threshold.  Individual pairs may
# scatter a few percent on throttle-heavy cases while the median is still
# pinned well below 1%.
TARGET_UNCERTAINTY_PCT = 0.7  # stop adding bursts once the median is this tight
RATIO_NOISE_LIMIT_PCT = 1.5  # above this at the cap, the number is untrustworthy

# When a case is still noisy after MIN_BURSTS pairs, the bursts are aliasing
# against clock/power-throttle oscillation (the S7 lm_head shapes throttle an
# H100 by ~30%; a 24 ms burst samples one clock state, a long burst averages
# the cycle).  The sampler then lengthens the bursts — at least MIN_BURST_US
# on the faster leg, doubling on every further escalation — and restarts
# sampling, until either cap below stops it.  150 ms was fitted on an H100
# PCIe: S7 f32 (327 ms bursts) measured 0.9% spread while S7 bf16 (24 ms
# bursts) measured 10%+; S7 TN-bf16 needed several doublings beyond that.
MIN_BURST_US = 150_000.0
MAX_BURST_US = 1_500_000.0  # do not grow the slower leg's burst past this
ITERS_ESCALATION_CAP = 64  # bounds kineto event volume for fast kernels

# Each timed burst gets an untimed lead-in of the SAME leg (settle // of the
# burst's iterations) so it starts from that leg's own steady power state.
# Interleaving two legs with unequal power draw otherwise alternates the
# card's clock state, and each burst inherits the throttle the OTHER leg
# left behind (observed: 26.7% stock-leg spread on S7 TN-bf16 without it).
SETTLE_FRACTION = 2  # settle iters = burst iters // SETTLE_FRACTION, min 1

# A bimodal split is a single dominant gap in the sorted pair ratios with
# real samples (>= 2) settled on each side, distinguishing "two clusters"
# from "one Gaussian-ish cluster with a long tail". Both conditions must
# hold so ordinary noise is never misclassified as bimodal:
#  - the gap must dwarf the spread WITHIN either side (a wide single cluster
#    has gaps comparable to its own internal spacing, a real split does not)
#  - the gap must be a real fraction of the median, so float jitter on an
#    already-tiny number can never trip this
BIMODAL_GAP_RATIO = 2.5
BIMODAL_MIN_GAP_PCT = 5.0
# Extended cap used only while a case keeps showing the bimodal signature;
# ordinary noisy-but-unimodal cases still stop at MAX_BURSTS. Bounded so a
# persistently bimodal case still terminates the suite in finite time.
BIMODAL_MAX_BURSTS = 32

# Device-event types torch.profiler may report kernels under: cuBLAS/rocBLAS
# kernels land on DeviceType.CUDA; the mojo-device kernels are reported by
# CUPTI under the PrivateUse1 backend.
_DEVICE_EVENT_TYPES = ("DeviceType.CUDA", "DeviceType.PrivateUse1")


class NoDeviceKernels(RuntimeError):
    """The profiler saw no device kernel events for a timed burst."""


@dataclasses.dataclass(frozen=True)
class LegTiming:
    per_iter_us: float  # median over bursts
    spread_pct: float  # (max - min) / median over bursts, in percent


@dataclasses.dataclass(frozen=True)
class Measurement:
    ratio: float  # median of per-pair ours/ref ratios; > 1 means we are slower
    ratio_spread_pct: float  # robust scatter of individual pair ratios, percent
    ratio_uncertainty_pct: float  # scatter / sqrt(n): uncertainty of the median
    bursts: int  # burst pairs actually sampled (adaptive, MIN..MAX)
    ours: LegTiming
    ref: LegTiming
    bimodal: bool = False  # a two-cluster split was seen in the pair ratios


@contextlib.contextmanager
def gpu_lock(index: int = 0) -> Iterator[None]:
    """AGENTS.md rule: hold flock /tmp/gpu_lock_{index}.lock around GPU work."""
    with open(f"/tmp/gpu_lock_{index}.lock", "w") as handle:
        fcntl.flock(handle, fcntl.LOCK_EX)
        try:
            yield
        finally:
            fcntl.flock(handle, fcntl.LOCK_UN)


def _burst_us_kineto(
    fn: Callable[[], object], iters: int, sync: Callable[[], None]
) -> float:
    with profile(activities=[ProfilerActivity.CPU, ProfilerActivity.CUDA]) as prof:
        for _ in range(iters):
            fn()
        sync()
    total_us = 0.0
    for evt in prof.events():
        if str(evt.device_type) not in _DEVICE_EVENT_TYPES:
            continue
        low = evt.name.lower()
        if "memcpy" in low or "memset" in low:
            continue
        dur = evt.device_time if hasattr(evt, "device_time") else evt.cuda_time
        total_us += float(dur)
    if total_us <= 0.0:
        raise NoDeviceKernels("profiler saw no device kernel events in the burst")
    return total_us / iters


def _burst_us_queue_proxy(
    fn: Callable[[], object], iters: int, sync: Callable[[], None]
) -> float:
    sync()
    start = time.perf_counter()
    for _ in range(iters):
        fn()
    sync()
    return (time.perf_counter() - start) / iters * 1e6


def _burst_fn(
    method: str,
) -> Callable[[Callable[[], object], int, Callable[[], None]], float]:
    if method == "kineto-device-time":
        return _burst_us_kineto
    return _burst_us_queue_proxy  # queue-proxy-device-time and perf-counter


def _leg(per_burst: list[float]) -> LegTiming:
    med = statistics.median(per_burst)
    return LegTiming(med, (max(per_burst) - min(per_burst)) / med * 100.0)


def _robust_spread_pct(values: list[float]) -> float:
    """Spread of the central mass relative to the median, in percent.

    Interquartile-based once there are enough samples, so one throttled
    outlier burst cannot keep the adaptive loop from ever converging;
    plain (max - min) below four samples where quartiles are meaningless.
    """
    med = statistics.median(values)
    if len(values) < 4:
        lo, hi = min(values), max(values)
    else:
        quarts = statistics.quantiles(values, n=4, method="inclusive")
        lo, hi = quarts[0], quarts[2]
    return (hi - lo) / med * 100.0


def _bimodal_split(values: list[float]) -> bool:
    """True if `values` look like two tight clusters, not one noisy one.

    Finds the single largest gap in the sorted samples and requires real
    samples (>= 2) on each side, so one outlier burst cannot trip this. The
    gap must also dominate the spread WITHIN either side (BIMODAL_GAP_RATIO)
    and be a real fraction of the median (BIMODAL_MIN_GAP_PCT) — see the
    module docstring and the constants' comments for why both are needed.
    """
    if len(values) < 4:
        return False
    s = sorted(values)
    gaps = [b - a for a, b in zip(s, s[1:])]
    i = max(range(len(gaps)), key=gaps.__getitem__)
    left, right = s[: i + 1], s[i + 1 :]
    if len(left) < 2 or len(right) < 2:
        return False
    gap = gaps[i]
    within = max(left[-1] - left[0], right[-1] - right[0], 1e-12)
    med = statistics.median(s)
    return (
        gap >= BIMODAL_GAP_RATIO * within and gap / med * 100.0 >= BIMODAL_MIN_GAP_PCT
    )


def _longer_iters(
    iters_now: int, ref_us: list[float], our_us: list[float]
) -> int | None:
    """Next escalation of the burst length, or None once the caps bind.

    Grows to MIN_BURST_US on the faster leg first, then doubles, so a case
    whose throttle oscillation is slower than one burst keeps buying longer
    bursts instead of accumulating equally-aliased samples.
    """
    if iters_now >= ITERS_ESCALATION_CAP:
        return None
    fastest = min(statistics.median(ref_us), statistics.median(our_us))
    slowest = max(statistics.median(ref_us), statistics.median(our_us))
    wanted = max(iters_now * 2, math.ceil(MIN_BURST_US / fastest))
    wanted = min(wanted, ITERS_ESCALATION_CAP, math.floor(MAX_BURST_US / slowest))
    return wanted if wanted > iters_now else None


def measure(
    ref_fn: Callable[[], object],
    our_fn: Callable[[], object],
    ref_sync: Callable[[], None],
    our_sync: Callable[[], None],
    method: str,
    iters: int,
    min_bursts: int = MIN_BURSTS,
    max_bursts: int = MAX_BURSTS,
    target_uncertainty_pct: float = TARGET_UNCERTAINTY_PCT,
) -> Measurement:
    """Interleaved two-leg device-time measurement; see module docstring."""
    burst = _burst_fn(method)

    def settled_burst(
        fn: Callable[[], object], iters_now: int, sync: Callable[[], None]
    ) -> float:
        for _ in range(max(1, iters_now // SETTLE_FRACTION)):
            fn()
        sync()  # the settle lead-in must fully retire before timing starts
        return burst(fn, iters_now, sync)

    # Warmup, both legs, before any timed region: the first mojo call JIT
    # compiles its kernel variant, the first cuBLAS call settles heuristics,
    # and the extra iterations start the clock ramp.
    for _ in range(WARMUP_ITERS):
        ref_fn()
    ref_sync()
    for _ in range(WARMUP_ITERS):
        our_fn()
    our_sync()

    ref_us: list[float] = []
    our_us: list[float] = []
    pair_ratios: list[float] = []
    spread = float("inf")
    uncertainty = float("inf")
    iters_now = iters
    # Sticky latch, not a per-iteration recomputation: a two-cluster split is
    # only detectable at a complete quartet (see _bimodal_split's >= 4
    # requirement), so on the odd sample mid-quartet this must hold its last
    # value rather than momentarily reporting False — otherwise the extended
    # cap below collapses back to max_bursts one sample after being earned
    # and the loop stops right past the ordinary cap regardless.
    bimodal = False
    while True:
        # ABBA: flip which leg leads on every pair, so the lag between a
        # pair's two bursts alternates sign and a monotonic ramp cancels
        # between consecutive pairs (see the module docstring).
        if len(pair_ratios) % 2:
            our_us.append(settled_burst(our_fn, iters_now, our_sync))
            ref_us.append(settled_burst(ref_fn, iters_now, ref_sync))
        else:
            ref_us.append(settled_burst(ref_fn, iters_now, ref_sync))
            our_us.append(settled_burst(our_fn, iters_now, our_sync))
        pair_ratios.append(our_us[-1] / ref_us[-1])
        spread = _robust_spread_pct(pair_ratios)
        uncertainty = spread / math.sqrt(len(pair_ratios))
        complete_quartets = len(pair_ratios) % 2 == 0
        if complete_quartets:
            bimodal = bimodal or _bimodal_split(pair_ratios)
        if (
            len(pair_ratios) >= min_bursts
            and complete_quartets
            and uncertainty <= target_uncertainty_pct
            and not bimodal
        ):
            break
        if len(pair_ratios) >= min_bursts and complete_quartets:
            longer = _longer_iters(iters_now, ref_us, our_us)
            if longer is not None:
                # Short noisy bursts: lengthen them so each one averages a
                # whole throttle cycle, and restart sampling — the short
                # bursts sampled single clock states and would poison the
                # median. Fresh burst length, fresh bimodality read.
                iters_now = longer
                ref_us.clear()
                our_us.clear()
                pair_ratios.clear()
                bimodal = False
                continue
        # A confirmed bimodal split gets a much bigger burst budget than
        # ordinary noise: it needs enough draws to cross BOTH clusters
        # before the median means anything (see the module docstring). Every
        # other case is unaffected and still stops at max_bursts.
        cap = BIMODAL_MAX_BURSTS if bimodal else max_bursts
        if len(pair_ratios) >= cap:
            break
    return Measurement(
        statistics.median(pair_ratios),
        spread,
        uncertainty,
        len(pair_ratios),
        _leg(our_us),
        _leg(ref_us),
        bimodal,
    )


def iters_for_flops(flops: float, budget_flops: float = 1.0e12) -> int:
    """Burst length: enough iterations for a stable median, capped so the
    huge cases (S7 is 3.8 TFLOP per matmul) stay affordable."""
    return max(3, min(32, round(budget_flops / max(flops, 1.0))))
