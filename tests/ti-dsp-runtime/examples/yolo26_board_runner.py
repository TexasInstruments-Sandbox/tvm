#!/usr/bin/env python3
"""
YOLO26 board-side inference runner.

This script runs *on* the board (BeagleY-AI/AM67A), never on the dev host.
`run_yolo26_detection.py` (the host-side driver) scp's everything this
script needs into one directory before invoking it over SSH:

  - the compiled model (``lib0.out``, a TVM c_static DLOAD module)
  - one ``input_<name>.npy`` per image to run
  - a copy of ``c7x_runtime.py`` (the ctypes wrapper behind
    ``tvm.contrib.c7x.C7xVirtualMachine``), imported directly here instead of
    ``from tvm.contrib.c7x import ...`` so this script has no dependency
    beyond numpy + the deployed ``libc7x_arm_runtime.so`` -- it does not
    require a full TVM install on the board.

Everything above is just setup to get here. The part that actually matters
-- the Python offload API itself -- is just this:

    with C7xVirtualMachine(module_path, so_path=so_path) as vm:
        output = vm.run_nocopy(input_array)

identical in spirit to running a ``relax.VirtualMachine`` on CPU, except the
inference actually executes on the C7x DSP via the on-device rpmsg IPC
service (see ``python/tvm/contrib/c7x/c7x_runtime.py`` and
``src/runtime/ti_dsp/firmware/c7x/arm/``).

Usage (normally invoked by run_yolo26_detection.py over SSH; can also be run
by hand on the board for debugging):

    python3 yolo26_board_runner.py --module lib0.out
"""

import argparse
import sys
import time
from pathlib import Path

import numpy as np

# c7x_runtime.py is copied into this same directory by the host-side driver.
sys.path.insert(0, str(Path(__file__).parent))
from c7x_runtime import C7xVirtualMachine  # noqa: E402  # pyright: ignore[reportMissingImports]


def main() -> int:
    parser = argparse.ArgumentParser(description="Run YOLO26 inference on the C7x DSP")
    parser.add_argument(
        "--module", default="lib0.out", help="Path to the compiled DLOAD module (lib0.out)"
    )
    parser.add_argument(
        "--so-path",
        default="/usr/local/lib/libc7x_arm_runtime.so",
        help="Path to libc7x_arm_runtime.so (installed by "
        "src/runtime/ti_dsp/firmware/c7x/arm/build.sh deploy)",
    )
    parser.add_argument(
        "--input-dir",
        default=".",
        help="Directory containing input_<name>.npy files to run inference on",
    )
    args = parser.parse_args()

    input_paths = sorted(Path(args.input_dir).glob("input_*.npy"))
    if not input_paths:
        print(f"No input_*.npy files found in {args.input_dir}", file=sys.stderr)
        return 1

    # Load the module once; run every image through that same loaded module
    # (each call to Infer() re-runs the graph, not a reload).
    with C7xVirtualMachine(args.module, so_path=args.so_path) as vm:
        for input_path in input_paths:
            input_array = np.load(input_path)

            # run_nocopy() returns a numpy view straight into the DSP's
            # result DDR buffer (no copy) -- safe here because we save it to
            # disk with np.save() before the next run_nocopy() call
            # overwrites that buffer. Use vm["main"](...) instead if you need
            # the result to survive past the next inference call.
            #
            # Timed around just this call (not np.load/np.save) so it isolates
            # the actual offload round-trip -- ctypes call -> rpmsg IPC -> DSP
            # -> back -- from local disk I/O. Complements vm.last_cycles: that
            # is the DSP's own TSC count (pure on-chip execution), while this
            # wall-clock delta also includes IPC/ctypes overhead around it.
            t0 = time.perf_counter()
            output = vm.run_nocopy(input_array)
            elapsed_ms = (time.perf_counter() - t0) * 1000

            output_path = Path(str(input_path).replace("input_", "output_", 1))
            np.save(output_path, output)
            print(
                f"{input_path.name}: {vm.last_cycles:,} DSP cycles, "
                f"{elapsed_ms:.2f} ms wall-clock -> {output_path.name}"
            )

    return 0


if __name__ == "__main__":
    sys.exit(main())
