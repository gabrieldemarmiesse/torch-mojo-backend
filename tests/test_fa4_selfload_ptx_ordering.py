"""PTX *and SASS* ordering regression check for the self-loading FA4 fwd kernel.

``fa4_fwd_selfload_kernel.mojo`` deleted the empty[] mbarriers a phase-2b
producer/consumer split used to prove a ring slot's smem read had retired
before the next TMA refill into that same slot. The substitute proof is
program order plus ``wgmma.wait_group`` (see the PREFETCH invariant comment
next to its definition in that file): the refill's `cp.async.bulk.tensor` /
`mbarrier.arrive.expect_tx` must be emitted AFTER the `wgmma.wait_group`
that proves the slot is free.

That proof has two compilers in it, and this file checks both:

* **LLVM** decides what the emitted PTX order is. The kernel pins it with
  ``wgmma_wait_group_ordered`` (`wgmma.wait_group.sync.aligned` carrying a
  `~{memory}` clobber, which the stdlib's `wgmma_wait_group_sync` does
  not), and ``test_selfload_refills_follow_their_wait_group`` checks the
  result in the PTX.
* **ptxas** decides what the SASS order is, and is reached by nothing we
  can write in Mojo. ``test_selfload_refills_follow_their_wait_group_sass``
  assembles the PTX and checks that each ``UTMALDG`` refill still sits
  after a ``WARPGROUP.DEPBAR``. Without it this file would assert a
  property one compiler down from the one that matters -- the PTX can be
  in perfect order while the scheduled machine code is not.

Neither test is a check on the kernel's own logic (the soaks in
``test_fa4_selfload_soak.py`` cover that at runtime); both are guards
against toolchain drift.

Cross-compiles with ``mojo build --emit asm --target-accelerator sm_90a``
(needs no GPU, never touches ``/tmp/gpu_lock_0.lock``) the same way
``scripts/compare_kernel_asm.py`` does, and greps the emitted PTX. The SASS
leg needs ``ptxas`` and ``nvdisasm`` -- also host-only tools, needing no
device -- and skips when neither PATH nor the triton wheel provides them.

Builds land in a STABLE directory (``tests/__mojocache__/...``, matching the
already-gitignored ``__mojocache__/`` pattern), not a fresh temp dir per run:
Mojo's own transform cache is keyed by kernel content hash independent of
``-o``, and replays a previously built kernel's sidecar to the path it was
FIRST built at -- a fresh ``tempfile.TemporaryDirectory()`` each run means
that path is already gone by the second run in one session (see
``compare_kernel_asm.py``'s own docstring: "Both trees are built into one
shared directory per specialization, which is load-bearing").
"""

from __future__ import annotations

import re
import shutil
import subprocess
from pathlib import Path

import pytest

from scripts.compare_kernel_asm import build_env, mojo_cli

_REPO_ROOT = Path(__file__).resolve().parents[1]
_PROBE = Path(__file__).resolve().parent / "fa4_selfload_ptx_probe.mojo"
_D128_GUARD_PROBE = (
    Path(__file__).resolve().parent / "fa4_selfload_d128_guard_probe.mojo"
)
_FA4_DIR = _REPO_ROOT / "torch_mojo_backend" / "eager_flash_attention"
_BUILD_DIR = Path(__file__).resolve().parent / "__mojocache__" / "fa4_selfload_ptx"
_D128_GUARD_BUILD_DIR = (
    Path(__file__).resolve().parent / "__mojocache__" / "fa4_selfload_d128_guard"
)

_SASS_BUILD_DIR = (
    Path(__file__).resolve().parent / "__mojocache__" / "fa4_selfload_sass"
)

_WAIT_GROUP_RE = re.compile(r"\bwgmma\.wait_group\.sync\.aligned\b")
_REFILL_COPY_RE = re.compile(r"\bcp\.async\.bulk\.tensor\b.*mbarrier::complete_tx")
_EXPECT_TX_RE = re.compile(r"\bmbarrier\.arrive\.expect_tx\b")

# SASS spellings on sm_90a: the wgmma wait is WARPGROUP.DEPBAR.LE gsb0, N
# (the plain DEPBAR.LE SB0 in the epilogue is the TMA *store* wait, a
# different scoreboard), and a TMA tile load is UTMALDG.
_SASS_ADDR_RE = re.compile(r"/\*([0-9a-f]{4,})\*/")
_SASS_WAIT_RE = re.compile(r"\bWARPGROUP\.DEPBAR\b")
_SASS_TMA_LOAD_RE = re.compile(r"\bUTMALDG\b")


def _build_probe_ptx(out_dir: Path) -> str:
    try:
        mojo = mojo_cli()
    except FileNotFoundError:
        pytest.skip("mojo compiler not found")
    out_dir.mkdir(parents=True, exist_ok=True)
    # Stale sidecars from a previous source edit would otherwise accumulate
    # (the directory is stable across runs -- see the module docstring) and
    # break the "exactly one .ptx" sanity check below.
    for stale in list(out_dir.glob("*.ptx")) + list(out_dir.glob("*.s")):
        stale.unlink()
    out_stem = out_dir / "fa4_selfload_ptx_probe"
    command = [
        str(mojo),
        "build",
        str(_PROBE),
        "-I",
        str(_FA4_DIR),
        "--emit",
        "asm",
        "--target-accelerator",
        "sm_90a",
        "-o",
        str(out_stem) + ".s",
    ]
    result = subprocess.run(
        command,
        cwd=str(_REPO_ROOT),
        env=build_env(),
        capture_output=True,
        text=True,
        timeout=600,
    )
    assert result.returncode == 0, (
        "mojo build --emit asm failed for the self-load fwd probe:\n"
        f"{result.stderr or result.stdout}"
    )
    ptx_files = sorted(out_dir.glob("*.ptx"))
    assert ptx_files, (
        "mojo build --emit asm produced no .ptx sidecar -- the probe did"
        " not reach a GPU kernel instantiation"
    )
    selfload_ptx = [p for p in ptx_files if "selfload" in p.name]
    assert selfload_ptx, (
        f"no PTX sidecar named for the self-load kernel among {ptx_files}"
        " -- check fwd_fa4_selfload_kernel's naming"
    )
    assert len(selfload_ptx) == 1, (
        f"expected exactly one self-load kernel instantiation, got {selfload_ptx}"
    )
    return selfload_ptx[0].read_text()


@pytest.fixture(scope="module")
def selfload_ptx() -> str:
    return _build_probe_ptx(_BUILD_DIR)


def _cuda_host_tool(name: str) -> Path | None:
    """``ptxas``/``nvdisasm`` from PATH, else from the triton wheel.

    Both are host tools that need no GPU. A CUDA-toolkit install puts them
    on PATH; a torch install without one still ships them inside triton
    (``triton/backends/nvidia/bin``), which is where this repo's own venv
    has them.
    """
    found = shutil.which(name)
    if found is not None:
        return Path(found)
    try:
        import triton
    except ImportError:
        return None
    candidate = (
        Path(triton.__file__).resolve().parent / "backends" / "nvidia" / "bin" / name
    )
    return candidate if candidate.exists() else None


@pytest.fixture(scope="module")
def selfload_sass(selfload_ptx: str) -> str:
    """The kernel's SASS: ptxas-assembled from the same PTX, then disassembled."""
    ptxas = _cuda_host_tool("ptxas")
    nvdisasm = _cuda_host_tool("nvdisasm")
    if ptxas is None or nvdisasm is None:
        pytest.skip("ptxas/nvdisasm not found (no CUDA toolkit and no triton wheel)")
    _SASS_BUILD_DIR.mkdir(parents=True, exist_ok=True)
    ptx_path = _SASS_BUILD_DIR / "fa4_selfload.ptx"
    ptx_path.write_text(selfload_ptx)
    cubin_path = _SASS_BUILD_DIR / "fa4_selfload.cubin"
    assembled = subprocess.run(
        [str(ptxas), "-arch=sm_90a", "-O3", str(ptx_path), "-o", str(cubin_path)],
        capture_output=True,
        text=True,
        timeout=600,
    )
    assert assembled.returncode == 0, (
        f"ptxas failed on the self-load kernel PTX:\n"
        f"{assembled.stderr or assembled.stdout}"
    )
    disassembled = subprocess.run(
        [str(nvdisasm), "-c", str(cubin_path)],
        capture_output=True,
        text=True,
        timeout=600,
    )
    assert disassembled.returncode == 0, (
        f"nvdisasm failed on the self-load kernel cubin:\n"
        f"{disassembled.stderr or disassembled.stdout}"
    )
    return disassembled.stdout


def _sass_events(
    sass: str, pattern: re.Pattern[str], kind: str
) -> list[tuple[int, str]]:
    """(instruction address, kind) for every SASS line matching `pattern`.

    Keyed on the address nvdisasm prints in each line's ``/*01a0*/``
    comment rather than on line order: the listing is grouped into basic
    blocks, and the address is what actually orders the instructions.
    """
    events = []
    for line in sass.splitlines():
        if not pattern.search(line):
            continue
        address = _SASS_ADDR_RE.search(line)
        if address is None:
            continue
        events.append((int(address.group(1), 16), kind))
    return events


def _line_numbers(pattern: re.Pattern[str], text: str) -> list[int]:
    return [i for i, line in enumerate(text.splitlines()) if pattern.search(line)]


def _assert_every_event_follows_a_wait(
    events: list[tuple[int, str]], event_label: str
) -> None:
    """Walk (position, kind) events in position order: every 'event_label'
    kind must have a 'wait' kind since the previous 'event_label' (or since
    the start of the region, for the first one). `position` is a PTX line
    number or a SASS instruction address depending on the caller."""
    waits_since_last_event = 0
    seen_event = 0
    for position, kind in sorted(events):
        if kind == "wait":
            waits_since_last_event += 1
        else:
            assert waits_since_last_event >= 1, (
                f"a {event_label} at {position} has no preceding"
                " wgmma.wait_group since the previous one (or region start)"
                " -- the refill was not ordered after the wait"
                " that proves its slot is free"
            )
            waits_since_last_event = 0
            seen_event += 1
    assert seen_event >= 3, (
        f"only {seen_event} {event_label} event(s) found after the first"
        " wgmma.wait_group -- expected at least 3 (1 prologue refill + 2"
        " per loop trip), the check would otherwise be nearly vacuous"
    )


def test_selfload_refills_follow_their_wait_group(selfload_ptx: str):
    """Every K/V refill TMA copy is textually ordered after a wgmma.wait_group.

    The prologue's initial fill (Q plus the first PREFETCH tile-pairs) has
    no preceding wait -- there is nothing to wait for, no read has happened
    yet -- so this only checks copies/expects that occur AFTER the first
    wgmma.wait_group in the file, which excludes exactly that initial fill.
    """
    wait_lines = _line_numbers(_WAIT_GROUP_RE, selfload_ptx)
    assert len(wait_lines) >= 3, (
        f"expected at least 3 wgmma.wait_group sites (prologue + 2 per loop"
        f" trip), found {len(wait_lines)} -- the probe may not have reached"
        " the kernel body"
    )
    first_wait = wait_lines[0]

    refill_lines = _line_numbers(_REFILL_COPY_RE, selfload_ptx)
    expect_lines = _line_numbers(_EXPECT_TX_RE, selfload_ptx)

    wait_events = [(line, "wait") for line in wait_lines if line >= first_wait]
    refill_events = wait_events + [
        (line, "refill") for line in refill_lines if line > first_wait
    ]
    _assert_every_event_follows_a_wait(refill_events, "cp.async.bulk.tensor refill")

    expect_events = wait_events + [
        (line, "expect_tx") for line in expect_lines if line > first_wait
    ]
    _assert_every_event_follows_a_wait(expect_events, "mbarrier.arrive.expect_tx")


def test_selfload_refills_follow_their_wait_group_sass(selfload_sass: str) -> None:
    """Same ordering, one compiler further down: in ptxas-scheduled SASS.

    The PTX test above constrains LLVM only. ptxas re-schedules that PTX
    into SASS and is not reachable from Mojo source at all -- no clobber,
    intrinsic or fence we can write is *addressed* to it -- so the only
    way to know it kept the refill behind its wait is to look. Every
    ``UTMALDG`` after the first ``WARPGROUP.DEPBAR`` (i.e. every refill;
    the prologue's initial fill precedes all waits and is excluded, as in
    the PTX test) must follow a ``WARPGROUP.DEPBAR``.
    """
    waits = _sass_events(selfload_sass, _SASS_WAIT_RE, "wait")
    loads = _sass_events(selfload_sass, _SASS_TMA_LOAD_RE, "refill")
    assert len(waits) >= 3, (
        f"expected at least 3 WARPGROUP.DEPBAR sites in the SASS, found"
        f" {len(waits)} -- did the disassembly reach the kernel body?"
    )
    assert len(loads) >= 6, (
        f"expected at least 6 UTMALDG sites in the SASS (prologue fill plus"
        f" refills), found {len(loads)}"
    )
    first_wait = waits[0][0]
    events = waits + [
        (address, kind) for address, kind in loads if address > first_wait
    ]
    _assert_every_event_follows_a_wait(events, "UTMALDG refill")


def test_selfload_rejects_d128_at_compile_time() -> None:
    """Scope enforcement, not just documentation: instantiating the
    self-loading kernel at head_dim=128 must fail the BUILD (ported from
    agent A2's review artifact, d128_guard_v7.mojo)."""
    try:
        mojo = mojo_cli()
    except FileNotFoundError:
        pytest.skip("mojo compiler not found")
    _D128_GUARD_BUILD_DIR.mkdir(parents=True, exist_ok=True)
    out_path = _D128_GUARD_BUILD_DIR / "d128_guard.s"
    command = [
        str(mojo),
        "build",
        str(_D128_GUARD_PROBE),
        "-I",
        str(_FA4_DIR),
        "--emit",
        "asm",
        "--target-accelerator",
        "sm_90a",
        "-o",
        str(out_path),
    ]
    result = subprocess.run(
        command,
        cwd=str(_REPO_ROOT),
        env=build_env(),
        capture_output=True,
        text=True,
        timeout=600,
    )
    output = result.stderr + result.stdout
    assert result.returncode != 0, (
        "instantiating the self-load kernel at head_dim=128 built"
        " successfully -- the d64-only comptime assert did not fire"
    )
    assert "d64-only" in output, (
        f"build failed for an unexpected reason (not the d64-only scope"
        f" guard):\n{output}"
    )
