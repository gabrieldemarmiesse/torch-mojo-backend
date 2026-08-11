"""The per-test measure -> compare-to-baseline -> stage-update logic.

Pass/fail rules (per test node, for this machine's hardware config):

* no baseline entry            -> PASS (recorded when updating is enabled)
* ratio within +/-4% of base   -> PASS, nothing written (dead band, no churn)
* ratio  > baseline * 1.04     -> FAIL: this kernel regime regressed
* ratio  < baseline * 0.96     -> PASS (recorded when updating is enabled)

Read-only by default; writing needs --update-baselines (improvements and
new entries) or --update-baselines=force (also accepts regressions after
an intentional trade-off).  Writes happen per test through
baselines.merge_write, so a partial run can never clobber entries it did
not measure.
"""

from __future__ import annotations

import os
import time
from collections.abc import Callable

import pytest
import torch

from bench_lib import baselines
from bench_lib.clock import pin_intact, pinned_clock
from bench_lib.hw import Hardware
from bench_lib.measure import (
    RATIO_NOISE_LIMIT_PCT,
    Measurement,
    NoDeviceKernels,
    gpu_lock,
    iters_for_flops,
    measure,
)

REGRESSION_THRESHOLD = 0.04  # fail when the ratio worsens by more than 4%

# A too-noisy measurement is re-attempted after a cooldown before the node
# goes red: the observed noisy S7 episodes are transient throttle states
# (S7-NN-bf16: one attempt at 3.1% uncertainty, the rerun clean at 0.3%),
# and an intermittently red node trains people to ignore red.  The cooldown
# happens OUTSIDE the GPU flock, so the card idles toward a steady state
# and co-tenants are not starved while we wait.  Refusing after the
# retries is still correct — better no number than a noisy baseline.
NOISE_RETRIES = 2
NOISE_COOLDOWN_S = 30.0


def update_mode(config: pytest.Config) -> str | None:
    mode = config.getoption("--update-baselines")
    if mode is not None:
        return mode
    env = os.environ.get("TORCH_MOJO_BACKEND_BENCH_UPDATE", "")
    if env in ("1", "improve"):
        return "improve"
    if env == "force":
        return "force"
    return None


def _sync_for(device: str) -> Callable[[], None]:
    if device == "cuda":
        return torch.cuda.synchronize
    if device == "mps":
        return torch.mps.synchronize
    if device == "mojo":
        return torch.mojo.synchronize
    return lambda: None


def bench_key(node: pytest.Function) -> baselines.BenchKey:
    """The baseline tree path of this test node: op/dtype/shape/layout.

    Derived from the test's own axes, never parsed out of the node id:
    the op token is the test function name minus "test_" (override with
    @pytest.mark.bench_op("add.Tensor") when the aten name cannot be a
    Python identifier), dtype and shape come from the parametrize ids,
    and a test without a layout axis gets the fixed sentinel "contig" so
    every path has the same rank.

    Takes the node rather than the request so that collection can derive
    the same keys without running anything: benchmarks/test_coverage.py
    reconciles the recorded baselines against the suite through this very
    function, which is what keeps the check honest — it cannot drift from
    the key a real run would write.
    """
    marker = node.get_closest_marker("bench_op")
    op = marker.args[0] if marker else node.originalname.removeprefix("test_")
    callspec = getattr(node, "callspec", None)
    params = callspec.params if callspec is not None else {}
    missing = [axis for axis in ("dtype_id", "shape_id") if axis not in params]
    if missing:
        pytest.fail(
            f"benchmark {node.nodeid} lacks the {missing} parametrize "
            "axes: every benchmark must be parametrized with dtype_id and "
            "shape_id (fold any extra axis into the shape token), plus an "
            "optional layout axis, so its baseline path "
            "op/dtype/shape/layout is derivable.",
            pytrace=False,
        )
    return baselines.BenchKey(
        dtype=params["dtype_id"],
        op=op,
        shape=params["shape_id"],
        layout=params.get("layout", "contig"),
    )


class Bench:
    """Measures one case, checks it against the baseline, stages updates."""

    def __init__(
        self, request: pytest.FixtureRequest, hw: Hardware, mojo: torch.device
    ) -> None:
        self._request = request
        self._hw = hw
        self._mojo = mojo

    def run(
        self, ref_fn: Callable[[], object], our_fn: Callable[[], object], flops: float
    ) -> None:
        hw = self._hw
        entry_key = bench_key(self._request.node)
        iters = iters_for_flops(flops)
        try:
            for attempt in range(1 + NOISE_RETRIES):
                if attempt:
                    # Cooldown with the flock RELEASED (see NOISE_COOLDOWN_S).
                    time.sleep(NOISE_COOLDOWN_S)
                if hw.is_accelerator:
                    # Pin the core clock inside the flock: ambient clock
                    # policy left by co-tenants is what made the stock leg
                    # bimodal between processes (see bench_lib/clock.py).
                    with gpu_lock(), pinned_clock(hw.pinned_clock_mhz):
                        result = measure(
                            ref_fn,
                            our_fn,
                            _sync_for(hw.stock_device),
                            _sync_for("mojo"),
                            hw.method,
                            iters,
                        )
                        held = pin_intact(hw.pinned_clock_mhz)
                else:
                    result = measure(
                        ref_fn, our_fn, lambda: None, lambda: None, hw.method, iters
                    )
                    held = True
                if held and result.ratio_uncertainty_pct <= RATIO_NOISE_LIMIT_PCT:
                    break
        except NotImplementedError as exc:
            pytest.skip(f"not supported on the mojo device: {exc}")
        except NoDeviceKernels as exc:
            pytest.skip(f"cannot observe device time here: {exc}")
        if not held:
            pytest.fail(
                "the GPU core-clock pin did not hold during measurement on "
                f"any of {attempt + 1} attempts: something outside the GPU "
                "flock is changing device clock policy mid-run (a co-tenant "
                "engagement?); nothing was compared or recorded.",
                pytrace=False,
            )
        self._check(entry_key, result, attempts=attempt + 1)

    def _check(
        self, entry_key: baselines.BenchKey, result: Measurement, attempts: int = 1
    ) -> None:
        mode = update_mode(self._request.config)
        data = baselines.load()
        base = baselines.lookup(data, self._hw.key, entry_key)
        ratio = result.ratio
        detail = (
            f"ratio {ratio:.3f} +/- {result.ratio_uncertainty_pct:.1f}% "
            f"(pair scatter {result.ratio_spread_pct:.1f}% over "
            f"{result.bursts} burst pairs; ours/stock device time: "
            f"{result.ours.per_iter_us:.1f}us / {result.ref.per_iter_us:.1f}us), "
            f"leg spread ours {result.ours.spread_pct:.1f}% "
            f"stock {result.ref.spread_pct:.1f}%"
        )

        if result.ratio_uncertainty_pct > RATIO_NOISE_LIMIT_PCT:
            # Never compare or record a number the sampler could not pin
            # down: a noisy baseline would poison every later comparison.
            pytest.fail(
                f"measurement too noisy to trust: {detail}. The uncertainty of "
                f"the median ratio stayed above {RATIO_NOISE_LIMIT_PCT:.1f}% "
                f"after {result.bursts} burst pairs (the cap) on each of "
                f"{attempts} attempts ({NOISE_COOLDOWN_S:.0f}s cooldown "
                f"between attempts), so a "
                f"{REGRESSION_THRESHOLD:.0%} regression cannot be resolved; "
                "nothing was compared or recorded. If this persists, this "
                "case needs a larger iteration budget in "
                "bench_lib/measure.py, not a noisy baseline.",
                pytrace=False,
            )

        if base is None:
            self._record_or_hint(mode, entry_key, ratio, f"new entry: {detail}")
            return

        if ratio > base * (1.0 + REGRESSION_THRESHOLD):
            if mode == "force":
                self._record(entry_key, ratio)
                self._note(
                    f"ACCEPTED regression {entry_key}: "
                    f"baseline {base:.3f} -> {ratio:.3f}"
                )
                return
            pytest.fail(
                f"performance regression: {detail}; baseline {base:.3f}, "
                f"{(ratio / base - 1) * 100:+.1f}% vs the 4% threshold on "
                f"[{self._hw.key}]. If this trade-off is intentional, rerun "
                "this node with --update-baselines=force.",
                pytrace=False,
            )
        if ratio < base * (1.0 - REGRESSION_THRESHOLD):
            self._record_or_hint(
                mode,
                entry_key,
                ratio,
                f"improvement: baseline {base:.3f} -> {ratio:.3f} ({detail})",
            )
        # else: dead band, nothing to write, no churn.

    def _record_or_hint(
        self,
        mode: str | None,
        entry_key: baselines.BenchKey,
        ratio: float,
        message: str,
    ) -> None:
        if mode is not None:
            self._record(entry_key, ratio)
            self._note(f"recorded {entry_key}: {message}")
        else:
            self._note(
                f"NOT recorded (rerun with --update-baselines) {entry_key}: {message}"
            )

    def _record(self, entry_key: baselines.BenchKey, ratio: float) -> None:
        baselines.merge_write(self._hw.key, {entry_key: ratio})

    def _note(self, message: str) -> None:
        notes = getattr(self._request.config, "_bench_notes", None)
        if notes is None:
            notes = []
            self._request.config._bench_notes = notes
        notes.append(message)
