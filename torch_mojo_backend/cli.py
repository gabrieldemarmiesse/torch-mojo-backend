"""The `torch-mojo-backend` command line: `cache dir` prints the native
kernel cache directory and `cache clean` removes it (see
agents_docs/native_backend.md, "Three builds": nothing else ever reaps it).
`ptxas` prints which assembler the kernels are built with and why."""

from __future__ import annotations

import argparse
import shutil
import sys

from torch_mojo_backend import _ptxas, native


def _build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(prog="torch-mojo-backend")
    commands = parser.add_subparsers(dest="command", required=True)
    cache = commands.add_parser(
        "cache", help="the compiled native libraries (shim, backend, kernels)"
    )
    cache_commands = cache.add_subparsers(dest="cache_command", required=True)
    cache_commands.add_parser("dir", help="print the cache directory")
    cache_commands.add_parser(
        "clean", help="remove the cache directory; every build recompiles at next use"
    )
    commands.add_parser(
        "ptxas", help="the assembler the NVIDIA kernels are built with, and why"
    )
    return parser


def main(argv: list[str] | None = None) -> int:
    args = _build_parser().parse_args(argv)
    if args.command == "ptxas":
        if _ptxas.driver_cuda_version() is None:
            # Which ptxas fits is decided by the driver's CUDA version, and a
            # machine without an NVIDIA driver has none to decide by -- the
            # answer would belong to whatever machine the cubins end up on.
            print(
                "no NVIDIA driver on this machine (libcuda.so.1): which ptxas "
                "can be used depends on the driver's CUDA version, so there is "
                "nothing to report here. Run this on the machine with the GPU.",
                file=sys.stderr,
            )
            return 1
        print(_ptxas.report())
        return 0
    cache_dir = native.cache_dir()
    if args.cache_command == "dir":
        print(cache_dir)
    elif args.cache_command == "clean":
        if cache_dir.exists():
            shutil.rmtree(cache_dir)
            print(f"removed {cache_dir}", file=sys.stderr)
        else:
            print(f"nothing to remove: {cache_dir} does not exist", file=sys.stderr)
    return 0


if __name__ == "__main__":
    sys.exit(main())
