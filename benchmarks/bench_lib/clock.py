"""GPU core-clock pinning: the between-process determinism of the suite.

Why this module exists (measured 2026-08-10 on an H100 PCIe, torch 2.11):

The stock cuBLAS leg of the tf32 GEMM cases was bimodal ACROSS processes
(S1-NN stock leg ~614us in some processes, ~787us in others) while our leg
barely moved, so seeded ratios flapped against the 4% threshold.  It is NOT
cuBLAS kernel selection: the profiler shows the byte-identical kernel
(sm90_xmma_..._tilesize128x128x32...) in both modes, so pinning knobs like
CUBLAS_WORKSPACE_CONFIG or extra warmup cannot help.  The driver is the
DEVICE CLOCK POLICY the process happens to inherit: on a shared box, kernel
engagements lock the core clock for long stretches (per AGENTS.md) and
reset it later.  Reproduced exactly: an ambient 900MHz core-clock lock
gives 787.6us, free-run gives ~612us — the two observed modes.  The GPU
flock serializes device WORK, but clock policy is device-global state that
outlives the flock, so each benchmark process samples whatever policy the
co-tenant left behind, keeps it for its whole (short) life, and the
within-run uncertainty estimate is blind to it by construction.

The fix: pin the core clock ourselves, inside the flock, for every timed
region.  Pinned at 1395MHz the same leg reads 567-569us across fresh
processes under deliberately hostile ambient policy (900MHz lock, max-boost
lock, free), i.e. 0.4% peak-to-peak versus 25% unpinned.  Two non-obvious
properties of the pinned regime, both measured:

* The pin target sits on a PLATEAU: 1275, 1365 and 1395MHz all give
  ~568us, so the number is robust to the exact step chosen.
* A pinned sub-max clock is FASTER and cleaner than free-running boost
  (568us vs 612us at "1755"): at max boost the tensor-dense kernels
  oscillate against the electrical (EDP) current limit — the driver even
  reports 1620MHz under load with an ambient max-boost lock.  Sub-max
  pinning is therefore not a compromise; it is the only stationary
  operating point.  It also keeps sustained draw far under every
  power-limit setting (86W vs a 310-350W cap on the S1 case), so ambient
  power-limit changes stop mattering too, and the S7 lm_head shapes lose
  their free-run ~30% power-throttle oscillation.

PIN_FRACTION = 0.8 (of max graphics clock, snapped DOWN to a supported
step -> 1395MHz on H100 PCIe) was validated on that card only, per the
repo rule of naming where tuning constants were fitted; the plateau above
suggests it travels, but a new card should get a quick probe-and-look.

Co-tenancy contract: the pin is applied after taking the GPU flock and
RESET before releasing it, so it can never distort a co-tenant's timed
region — but the reset does mean any engagement-long ambient lock is gone
afterwards.  That is the fleet convention working as intended: the fleet's
own rule is "lock the clocks before taking reference numbers", i.e. per
timed run, exactly what this module does.  Anyone relying on an ambient
lock surviving someone else's benchmark was already gambling.

Pinning needs admin rights (root or CAP_SYS_ADMIN).  Where it is
unavailable the suite still runs, unpinned, under the OLD method key —
hw.py puts the pinned frequency in the config key precisely because a
ratio measured at a pinned clock and one at ambient policy are different
measurements (the denominator is clock-sensitive; the numerator hardly
is).  ROCm and Apple pinning are not implemented; those configs stay
unpinned and keyed accordingly.
"""

from __future__ import annotations

import contextlib
import subprocess
import time
from collections.abc import Iterator

PIN_FRACTION = 0.8  # of max supported graphics clock; fitted on H100 PCIe

# The suite, like gpu_lock(), addresses GPU 0. If the suite ever grows a
# device axis, the index must travel with the flock path AND this -i.
_GPU_INDEX = 0


def _smi(*args: str) -> subprocess.CompletedProcess[str]:
    return subprocess.run(
        ["nvidia-smi", "-i", str(_GPU_INDEX), *args],
        capture_output=True,
        text=True,
        timeout=30,
    )


def _supported_graphics_mhz() -> list[int]:
    proc = _smi("--query-supported-clocks=graphics", "--format=csv,noheader,nounits")
    if proc.returncode != 0:
        return []
    clocks = []
    for token in proc.stdout.split():
        with contextlib.suppress(ValueError):
            clocks.append(int(token))
    return sorted(clocks, reverse=True)


def pin_target_mhz() -> int | None:
    """The canonical pin: highest supported step <= PIN_FRACTION * max."""
    clocks = _supported_graphics_mhz()
    if not clocks:
        return None
    below = [c for c in clocks if c <= clocks[0] * PIN_FRACTION]
    return below[0] if below else clocks[-1]


def probe_pinning() -> int | None:
    """The pin frequency if this process may pin the core clock, else None.

    Actually applies and reverts a lock: permission cannot be inferred (root
    inside a container may still lack the capability).  Called once per
    session by hw.detect(); the transient lock is harmless because the probe
    runs no GPU work.
    """
    target = pin_target_mhz()
    if target is None:
        return None
    if _smi("-lgc", str(target)).returncode != 0:
        return None
    _smi("-rgc")
    return target


# A locked clock is a CEILING, not a floor: hardware power/EDP throttle can
# hold the instantaneous graphics clock far below an intact lock.  Measured
# 2026-08-10 on the H100 PCIe, S7 mm bf16 NT/TN: sustained ~344W (the power
# cap) drags clocks.gr to 825-930MHz under a healthy 1395MHz pin, and the
# clock returns to EXACTLY the pin within ~150ms of the last kernel
# retiring.  pin_intact therefore polls briefly instead of reading once —
# an immediate read after a power-dense burst sees the throttle tail and
# would deterministically fail nodes whose last burst ends deep in
# throttle (exactly how S7-NT/TN-bf16 went red on every attempt).
_PIN_VERIFY_TIMEOUT_S = 2.0
_PIN_VERIFY_INTERVAL_S = 0.1


def pin_intact(mhz: int | None) -> bool:
    """True when the core-clock LOCK is still in place after a timed region.

    Call INSIDE pinned_clock, after the timed region.  The clock is polled
    for up to _PIN_VERIFY_TIMEOUT_S: with the lock intact it settles at
    exactly the locked value as soon as the post-burst throttle tail decays
    (measured <=150ms; see above), while a lock that was removed or
    retargeted mid-measurement settles at the idle/boost/other-lock value
    instead (co-tenants run nvidia-smi without taking the GPU flock —
    observed once as a 4.3% stock leg excursion that survived every
    within-run check).  On a genuine policy change the caller must discard
    the measurement and retry.  When verification itself is impossible the
    answer is True: do not invent failures.
    """
    if mhz is None:
        return True
    deadline = time.monotonic() + _PIN_VERIFY_TIMEOUT_S
    while True:
        proc = _smi("--query-gpu=clocks.gr", "--format=csv,noheader,nounits")
        if proc.returncode != 0:
            return True
        try:
            if int(proc.stdout.strip()) == mhz:
                return True
        except ValueError:
            return True
        if time.monotonic() >= deadline:
            return False
        time.sleep(_PIN_VERIFY_INTERVAL_S)


@contextlib.contextmanager
def pinned_clock(mhz: int | None) -> Iterator[None]:
    """Pin the core clock around a timed region; no-op when mhz is None.

    Use INSIDE gpu_lock(): the pin is device-global state, and holding the
    flock is what makes set-measure-reset atomic against co-tenants.
    """
    if mhz is None:
        yield
        return
    proc = _smi("-lgc", str(mhz))
    if proc.returncode != 0:
        # Rights were probed at detect() time, so this is unexpected: fail
        # loudly rather than silently measure under ambient clock policy
        # with a config key that promises a pinned measurement.
        raise RuntimeError(
            f"could not pin the GPU core clock to {mhz}MHz: "
            f"{proc.stderr.strip() or proc.stdout.strip()}"
        )
    try:
        yield
    finally:
        _smi("-rgc")
