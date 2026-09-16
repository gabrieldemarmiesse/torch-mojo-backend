"""Pure-Python unit tests for bench_lib.measure's bimodal-detection path.

No GPU needed, and no `bench`/`hw`/`mojo_device` fixtures used: measure()'s
burst timing is driven entirely by injected callables, and `_longer_iters`
(the separate, pre-existing throttle-aliasing escalation) is monkeypatched
off so these tests exercise the NEW bimodal-budget-extension path in
isolation. See the bench_lib/measure.py module docstring for the mechanism
this is checking: a two-cluster split in the pair ratios must earn a much
larger burst budget (BIMODAL_MAX_BURSTS) than ordinary noise, while an
ordinary noisy-but-unimodal case must stay on the fast common path.
"""

from __future__ import annotations

import itertools
from collections.abc import Callable

import pytest
from bench_lib import measure as measure_mod


def _alternating_fn(
    low: float, high: float, calls_per_reading: int = 2
) -> Callable[[], float]:
    """A deterministic "device leg" whose recorded reading flips every call.

    Each settled_burst() call makes `calls_per_reading` calls to this
    function (the settle lead-in plus the timed call); grouping by that
    stride makes one reading come out of each settled_burst regardless of
    the exact settle count, so the returned sequence alternates cleanly
    between `low` and `high` once per burst.
    """
    counter = itertools.count()

    def fn() -> float:
        n = next(counter)
        return high if (n // calls_per_reading) % 2 else low

    return fn


@pytest.mark.parametrize(
    ("values", "expected"),
    [
        ([1.0, 1.01, 0.99, 1.02], False),  # tight single cluster, tiny jitter
        ([1.0, 1.0, 1.5, 1.5], True),  # two tight clusters, 50% apart
        ([1.0, 1.5, 1.0, 1.5, 1.0, 1.5], True),  # same, interleaved order
        ([1.0, 1.01, 1.02, 1.03], False),  # monotonic drift, no real gap
        ([1.0, 1.0, 1.0], False),  # below the 4-sample floor
        ([1.0, 1.0, 1.01, 1.5], False),  # gap real, but only 1 sample on one side
    ],
)
def test_bimodal_split(values: list[float], expected: bool) -> None:
    assert measure_mod._bimodal_split(values) is expected


def _use_scripted_bursts(monkeypatch: pytest.MonkeyPatch) -> None:
    """Make measure() read a burst's "time" straight from fn()'s return value.

    The real burst backends (_burst_us_kineto / _burst_us_queue_proxy)
    measure actual elapsed time and ignore what `fn` returns, which is
    right for production but useless for scripting a GPU-free scenario.
    Swapping in `lambda fn, iters, sync: fn()` makes settled_burst() report
    exactly the scripted value, while the settle lead-in's extra calls to
    `fn` still advance its internal state exactly as in production.
    """
    monkeypatch.setattr(
        measure_mod, "_burst_fn", lambda method: lambda fn, iters, sync: fn()
    )
    monkeypatch.setattr(measure_mod, "_longer_iters", lambda *a, **k: None)


def test_unimodal_noise_stays_on_the_fast_path(monkeypatch: pytest.MonkeyPatch) -> None:
    """An ordinary noisy-but-single-mode case must not pay the bimodal budget."""
    _use_scripted_bursts(monkeypatch)
    ref_fn = _alternating_fn(100.0, 101.0)  # ~1% jitter: well under the 5% gap floor
    result = measure_mod.measure(
        ref_fn=ref_fn,
        our_fn=lambda: 250.0,
        ref_sync=lambda: None,
        our_sync=lambda: None,
        method="queue-proxy-device-time",
        iters=1,
    )
    assert result.bimodal is False
    # Converges quickly on the ordinary path; the exact stopping point can
    # shift by one quartet depending on the quantile method's rounding, but
    # it must stay far short of paying the bimodal budget.
    assert result.bursts < measure_mod.MAX_BURSTS


def test_bimodal_case_extends_past_max_bursts(monkeypatch: pytest.MonkeyPatch) -> None:
    """A genuinely bimodal ref leg must sample well past the ordinary cap."""
    _use_scripted_bursts(monkeypatch)
    ref_fn = _alternating_fn(100.0, 160.0)  # 60% apart: a real split, not jitter
    result = measure_mod.measure(
        ref_fn=ref_fn,
        our_fn=lambda: 250.0,
        ref_sync=lambda: None,
        our_sync=lambda: None,
        method="queue-proxy-device-time",
        iters=1,
    )
    assert result.bimodal is True
    assert result.bursts > measure_mod.MAX_BURSTS
    assert result.bursts <= measure_mod.BIMODAL_MAX_BURSTS


def test_bimodal_budget_is_bounded(monkeypatch: pytest.MonkeyPatch) -> None:
    """A persistently bimodal case still terminates, at BIMODAL_MAX_BURSTS."""
    _use_scripted_bursts(monkeypatch)
    ref_fn = _alternating_fn(100.0, 160.0)
    result = measure_mod.measure(
        ref_fn=ref_fn,
        our_fn=lambda: 250.0,
        ref_sync=lambda: None,
        our_sync=lambda: None,
        method="queue-proxy-device-time",
        iters=1,
    )
    assert result.bursts == measure_mod.BIMODAL_MAX_BURSTS
