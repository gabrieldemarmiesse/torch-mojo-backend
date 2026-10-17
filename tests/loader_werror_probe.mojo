"""Probe for tests/test_build_warning_replay.py: the loader's side of the
warnings record, with no GPU and no torch.

`filter <file>` prints `warning_diagnostics` of the file's text, for the
parity check against the Python filter. `entry <root> <cache> <mojo>
<family>` asks a Loader rooted at `<root>` (a fake source tree holding
`tmb/kernels/<family>/entry.mojo`) for the family's entry and prints `OK`,
or `FAILED: ` and the error; the loader's own `[TRACE] built` line says
whether it compiled."""
from std.pathlib import Path
from std.sys import argv

from tmb.backend.loader import Loader, warning_diagnostics


def main() raises:
    var args = argv()
    var mode = String(args[1])
    if mode == "filter":
        print(warning_diagnostics(Path(String(args[2])).read_text()), end="")
        return
    var loader = Loader(
        String(args[2]), String(args[3]), String(args[4]), "probe", True
    )
    try:
        _ = loader.entry(String(args[5]), List[String]())
        print("OK")
    except e:
        print("FAILED: ", e, sep="")
