"""torchrun entry for run_worker_suites_adastra.sh: run one worker script, then
leave without the C exit handlers when MAX's VMM allocator is on.

Usage: torchrun ... tests/multinode/vmm_exit_entry.py <script.py> [args...]

On MI300A the suites need MODULAR_DEVICE_CONTEXT_MEMORY_MANAGER_VMM=1 (without
it four ranks reserve ~115 GB each of the APU's memory and the node runs out),
and with it the process segfaults in HIP's atexit handlers. That is a MAX bug,
reproduced without this repo, and it turns a passing worker into a failing exit
code. demo_scripts/gpt2_fsdp2.py and ring_pressure.py leave through os._exit
for the same reason. The worker's own exit code is kept either way.
"""

import os
import runpy
import sys
from pathlib import Path

script = str(Path(sys.argv[1]).resolve())
sys.argv = sys.argv[1:]
sys.path.insert(0, str(Path(script).parent))
code = 0
try:
    runpy.run_path(script, run_name="__main__")
except SystemExit as exit_request:
    code = (
        exit_request.code
        if isinstance(exit_request.code, int)
        else int(exit_request.code is not None)
    )
sys.stdout.flush()
sys.stderr.flush()
if os.environ.get("MODULAR_DEVICE_CONTEXT_MEMORY_MANAGER_VMM") == "1":
    os._exit(code)
sys.exit(code)
