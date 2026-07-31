"""Run a command with the Apple GPU's performance state pinned.

macOS has no public knob for GPU DVFS; Instruments' internal "Induced GPU
Performance State" is the only handle. `apple_gpu_clock_lock.py` builds a
patched Metal System Trace template with that state baked in, and this
wrapper runs the given command under `xctrace record --launch` with it —
the pin applies for the whole recording, i.e. the child's whole lifetime.

    uv run python bench_gpu_locked.py Maximum -- env \
        MODULAR_DEBUG=device-sync-mode /tmp/bench_apple_gemm --m=2048 ...

States: Automatic, Minimum, Medium, Maximum. Use Maximum for stable kernel
timing (no thermal/DVFS ramp noise between iterations); Minimum is useful
to check a kernel's behavior when bandwidth-starved. The throwaway .trace
bundle goes to a temp dir and is deleted afterwards. `xctrace` sometimes
crashes while finalizing the bundle — that is AFTER the child has run and
printed, so the child's output and exit are unaffected; we ignore it.

The child's stdout/stderr pass through to this terminal unchanged.
"""

from __future__ import annotations

import shutil
import subprocess
import sys
import tempfile
from pathlib import Path

from apple_gpu_clock_lock import STATE_VALUES, locked_template_path


def main() -> int:
    argv = sys.argv[1:]
    if "--" not in argv or len(argv) < 3:
        print(
            "usage: bench_gpu_locked.py <Automatic|Minimum|Medium|Maximum>"
            " -- <command...>",
            file=sys.stderr,
        )
        return 2
    split = argv.index("--")
    state = argv[0] if split == 1 else ""
    command = argv[split + 1 :]
    if state not in STATE_VALUES or not command:
        print(
            f"unknown state {state!r} (choose from {sorted(STATE_VALUES)})"
            if state not in STATE_VALUES
            else "empty command",
            file=sys.stderr,
        )
        return 2

    # xctrace --launch rejects bare command names ("Path not found 'env'").
    resolved = shutil.which(command[0])
    if resolved is None:
        print(f"command not found: {command[0]!r}", file=sys.stderr)
        return 2
    command[0] = resolved

    template = locked_template_path(state)
    if template is None:
        print(
            "could not build a clock-locked template (missing Xcode or"
            " changed Instruments internals); running WITHOUT the pin",
            file=sys.stderr,
        )
        return subprocess.run(command).returncode

    with tempfile.TemporaryDirectory(prefix="tmb_xctrace_") as tmp:
        trace = Path(tmp) / "bench.trace"
        result = subprocess.run(
            [
                "xctrace",
                "record",
                "--template",
                str(template),
                "--output",
                str(trace),
                "--target-stdout",
                "-",
                "--launch",
                "--",
                *command,
            ]
        )
    # xctrace's own exit code also reflects bundle-finalize crashes that
    # happen after the child already ran; don't let those look like child
    # failures. The child's output has already streamed through.
    return result.returncode


if __name__ == "__main__":
    raise SystemExit(main())
