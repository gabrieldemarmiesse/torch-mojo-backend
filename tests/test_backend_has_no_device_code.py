"""The Mojo base library must contain no accelerator-specific code.

The wheel ships `libtmb_backend` prebuilt, one file per platform, and that
file drives whatever GPU the user has: an H100, an MI300A, an Apple GPU, or
no accelerator at all (the wheel job builds it on a runner with none). That
only holds while the library carries no device code and makes no
compile-time accelerator choice -- every kernel is compiled later, per
(op, dtype, GPU), by the loader. `toolchain_identity()` in
torch_mojo_backend/native/__init__.py states this; here it is checked.

The check is the strongest one available: build the library with the exact
production command for four targets -- no `--target-accelerator` (what the
wheel job does), Apple M4, MI300A and H100 -- and require the four shared
libraries (or their lowered LLVM programs) to be byte-identical, with no
`.ptx` / `.amdgcn` / `.ll` sidecar
written beside any of them. A library holding one GPU kernel fails this
(the control test below proves the compiler honors the flag: a one-kernel
module differs across two targets), and so does a `comptime if
_accelerator_arch() == ...` reaching the base library from a shared helper.

Cross-compiling needs no accelerator, only the mojo compiler, so this runs
wherever the unit tests run. Four builds of the base library take about
50 s (12 s each) plus the compiler's module-cache fill on a cold machine.
"""

from __future__ import annotations

import hashlib
import subprocess
from pathlib import Path

from torch_mojo_backend import native

# None is production: the compiler picks, and on a machine with a GPU it
# picks that GPU, so the comparison covers the host's accelerator too.
TARGETS = (None, "apple-m4", "mi300a", "sm_90a")
SIDECAR_SUFFIXES = (".ptx", ".amdgcn", ".ll")

# The control: a module that does hold one GPU kernel, launched the way the
# eager kernels launch theirs. Built for two targets it must differ, or the
# equality above would be vacuous (a compiler that ignored the flag).
ONE_KERNEL = """\
from max.gpu.host import DeviceContext
from max.gpu import thread_idx


def fill_kernel(p: Pointer[Float32, MutAnyOrigin]):
    p[unsafe_offset=Int(thread_idx.x)] = Float32(thread_idx.x)


@export
def run_fill(n: Int) abi("C"):
    try:
        var ctx = DeviceContext()
        var buf = ctx.enqueue_create_buffer[DType.float32](n)
        ctx.enqueue_function[fill_kernel](
            buf.unsafe_ptr(), grid_dim=1, block_dim=n
        )
        ctx.synchronize()
    except:
        pass
"""


def _target_id(accelerator: str | None) -> str:
    return accelerator or "no-accelerator"


def _build(cmd: list[str]):
    proc = subprocess.run(
        cmd, capture_output=True, text=True, env=native.compiler_env()
    )
    assert proc.returncode == 0, " ".join(cmd) + "\n" + proc.stdout + proc.stderr


def _sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _sidecars(directory: Path) -> list[str]:
    return sorted(p.name for p in directory.iterdir() if p.suffix in SIDECAR_SUFFIXES)


def test_base_library_is_the_same_bytes_for_every_accelerator(tmp_path: Path):
    digests = {}
    for accelerator in TARGETS:
        # Mach-O embeds the output name in its install name and signature.
        # Hold it fixed so only the accelerator flag varies.
        out = tmp_path / "backend.so"
        _build(native.backend_build_command(out, accelerator))
        digests[_target_id(accelerator)] = _sha256(out)

    assert _sidecars(tmp_path) == [], "the base library emitted device code"
    if len(set(digests.values())) == 1:
        return

    # Mojo 26.5's Darwin host optimizer emits different machine code for
    # sm_90a even when the lowered LLVM program is byte-identical. Compare
    # that program too: target-dependent dispatch/device code has already
    # been lowered here, while host optimization/code signing has not.
    ir_digests = {}
    for accelerator in TARGETS:
        out = tmp_path / "backend.ir"
        command = native.backend_build_command(out, accelerator)
        command[command.index("shared-lib")] = "llvm"
        _build(command)
        ir_digests[_target_id(accelerator)] = _sha256(out)
    assert _sidecars(tmp_path) == [], "the base library emitted device code"
    assert len(set(ir_digests.values())) == 1, (
        "the Mojo base library contains accelerator-dependent LLVM code: "
        f"{ir_digests}; shared-library digests: {digests}"
    )


def test_a_module_with_one_kernel_does_differ(tmp_path: Path):
    """Control for the test above: the flag is honored, so equality means
    something."""
    source = tmp_path / "one_kernel.mojo"
    source.write_text(ONE_KERNEL)
    for emission, suffix in (("shared-lib", "so"), ("llvm", "ir")):
        digests = {}
        for accelerator in ("sm_90a", "mi300a"):
            out = tmp_path / f"one_kernel.{suffix}"
            _build(
                [
                    native._find_mojo(),
                    "build",
                    str(source),
                    "--emit",
                    emission,
                    "--target-cpu",
                    native.portable_target_cpu(),
                    "--target-accelerator",
                    accelerator,
                    "-o",
                    str(out),
                ]
            )
            digests[accelerator] = _sha256(out)
        assert digests["sm_90a"] != digests["mi300a"], (
            f"a GPU kernel emitted the same {emission} for NVIDIA and AMD; "
            "the base-library comparison would prove nothing"
        )
