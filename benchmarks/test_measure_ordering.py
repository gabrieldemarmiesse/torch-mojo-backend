"""The sampler's ABBA ordering, checked against a known ramp.

No GPU and no kernels: the two legs are fakes on a simulated card that
slows down monotonically, so the true ratio is known exactly and the
error of the reported ratio can be measured instead of argued about.

What this pins down is the one failure the sampler cannot see in itself.
Its stopping rule watches SCATTER, and a monotonic clock ramp produces
none — it produces a constant tilt, the same size and sign in every pair,
which a median over pairs preserves and `sqrt(n)` averaging never
shrinks.  A run can therefore report 0.3% uncertainty on a number that is
1% wrong.  Ordering is the only defense, so it gets a test.
"""

from __future__ import annotations

import sys
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parent))

from bench_lib import measure as measure_mod

TRUE_RATIO = 2.0
RAMP_PER_BURST = 0.02  # the card gets 2% slower for every burst it has run

# iters at the escalation cap: the sampler cannot lengthen bursts further,
# so it samples to max_bursts and these tests measure the ordering rather
# than the burst-length escalation, which has its own reasons.
PINNED_ITERS = measure_mod.ITERS_ESCALATION_CAP


def _ramped_card(ramp: float) -> tuple[object, object, object]:
    """Fake (ref_fn, our_fn, burst) on a card whose clock decays steadily.

    Drift advances once per TIMED BURST, which is what the real thing
    does: the untimed settle lead-in is part of the same burst's cost.
    Whichever leg runs second in a pair is therefore charged one extra
    tick — precisely the ABAB bias, in its purest form.
    """
    state = {"tick": 0}
    ours = object()

    def ref_fn() -> object:
        return None

    def our_fn() -> object:
        return ours

    def burst(fn: object, iters: int, sync: object) -> float:
        base = TRUE_RATIO if fn() is ours else 1.0
        cost = base * (1.0 + ramp) ** state["tick"]
        state["tick"] += 1
        return cost

    return ref_fn, our_fn, burst


def _reported_ratio(ramp: float, monkeypatch: pytest.MonkeyPatch, **kwargs) -> object:
    """Run the REAL sampler over the fake card and return its Measurement.

    Only the timing primitive is replaced, so the loop under test is the
    production one: its own ordering, settle lead-ins and convergence.
    """
    ref_fn, our_fn, burst = _ramped_card(ramp)
    monkeypatch.setattr(measure_mod, "_burst_fn", lambda method: burst)
    return measure_mod.measure(
        ref_fn,
        our_fn,
        lambda: None,
        lambda: None,
        "kineto-device-time",
        iters=PINNED_ITERS,
        **kwargs,
    )


def test_reported_ratio_survives_a_monotonic_ramp(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    result = _reported_ratio(RAMP_PER_BURST, monkeypatch)
    error_pct = abs(result.ratio / TRUE_RATIO - 1.0) * 100.0

    # Fixed ABAB order reports TRUE_RATIO * (1 + RAMP_PER_BURST) — 2% high
    # here — because "ours" always pays for the ref burst that preceded it.
    # ABBA sets each pair's lag against the next pair's, leaving only the
    # ramp's curvature: second order, and ~100x smaller at this ramp.
    abab_error_pct = RAMP_PER_BURST * 100.0
    assert error_pct < abab_error_pct / 10.0, (
        f"reported ratio {result.ratio:.6f} is {error_pct:.3f}% off the true "
        f"{TRUE_RATIO} under a {RAMP_PER_BURST:.0%}-per-burst ramp; fixed "
        f"ABAB ordering is about {abab_error_pct:.1f}% off, so the order flip "
        "in bench_lib/measure.py has stopped cancelling the first-order term."
    )

    # And the reason this test exists: the run's own uncertainty estimate
    # says nothing about that error, because a shared tilt is not scatter.
    assert result.ratio_uncertainty_pct > 0.0


def test_no_ramp_is_exact(monkeypatch: pytest.MonkeyPatch) -> None:
    # Sanity on the fixture: with no drift the sampler must return the true
    # ratio outright, or the test above is measuring its own bugs.
    assert _reported_ratio(0.0, monkeypatch).ratio == TRUE_RATIO


@pytest.mark.parametrize("max_bursts", (4, 6, 12))
def test_sampling_stops_on_whole_quartets(
    max_bursts: int, monkeypatch: pytest.MonkeyPatch
) -> None:
    # An odd number of pairs leaves one pair's lag uncancelled, so both the
    # convergence exit and the hard cap must land on an even count.
    result = _reported_ratio(RAMP_PER_BURST, monkeypatch, max_bursts=max_bursts)
    assert result.bursts % 2 == 0, (
        f"sampler stopped on {result.bursts} burst pairs (max_bursts="
        f"{max_bursts}); an odd count leaves one pair's drift lag uncancelled"
    )
