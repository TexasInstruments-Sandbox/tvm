#!/usr/bin/env python
"""Check whether a TIDL model is stable across repeated inferences on one load.

Loads a module once and infers three times through ``C7xVirtualMachine``, the
way a deployed application does, then diffs the results.  This is the pattern
that matters for correctness: the c7x_compute CLI cannot stand in for it,
because its standalone ``infer`` verb reserves IO per connection and a fresh CLI
process reports capacity 0, while its ``run`` verb reloads the module each time
and so resets any per-load state.

Use it when a change touches TIDL output handling, buffer binding, or anything
else that could carry state between ``algProcess`` calls.  A model whose output
is small will not exercise size-related faults -- prefer an output of a few KiB
or more.

Usage:
    python diag_tidl_persistent_load.py --board j722s-evm [--channels 8]

Requires TI_CGT_C7000_PATH, a staged TIDL tools directory, a board running c7x_compute
firmware, and python3 + numpy on the board.
"""

import argparse
import os
import subprocess
import sys
import tempfile
from pathlib import Path

import numpy as np

_TESTS_DIR = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(_TESTS_DIR / "dsp-cpp"))
sys.path.insert(0, str(Path(__file__).resolve().parent))

from dsp_utils import (  # noqa: E402
    add_board_arg,
    get_board_hostname,
    get_current_board,
    tidl_tools_config,
)

_HW = 16
_REMOTE_DIR = "/tmp/c7x_persistent_load"


def _ssh(host: str, cmd: str) -> subprocess.CompletedProcess:
    return subprocess.run(["ssh", f"root@{host}", cmd],
                          capture_output=True, text=True, check=False)


def _build(channels: int, work_dir: Path):
    import tvm
    from tvm.relax.backend.tidl import TIDLOffloadCompiler
    from tvm.relax.frontend import nn

    class ConvReluModel(nn.Module):
        def __init__(self):
            super().__init__()
            self.conv = nn.Conv2D(3, channels, 3, 1, 1, bias=False)

        def main(self, x):
            return nn.relu(self.conv(x))

    input_spec = {"x": nn.spec.Tensor((1, 3, _HW, _HW), "float32")}
    mod, param_spec = ConvReluModel().export_tvm(spec={"main": input_spec})
    np.random.seed(42)
    params = [
        tvm.runtime.tensor(np.random.rand(*p.shape).astype("float32"),
                           device=tvm.cpu())
        for _, p in param_spec
    ]
    param_dict = dict(zip(mod["main"].params[1:], params))
    input_data = np.random.randn(1, 3, _HW, _HW).astype("float32")
    calib = [np.random.randn(1, 3, _HW, _HW).astype("float32") for _ in range(2)]

    compiler = TIDLOffloadCompiler(
        config={
            "board": get_current_board(),
            "artifacts_dir": str(work_dir / "tidl_artifacts"),
            **tidl_tools_config(),
            "num_calibration_frames": len(calib),
            "calibration_inputs": calib,
        }
    )
    result = compiler.build(mod, params=param_dict, exec_mode="c7x_dload",
                            build_dir=str(work_dir / "build"))
    return result.module_path, input_data


def main() -> int:
    parser = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    add_board_arg(parser)
    parser.add_argument("--channels", type=int, default=8)
    args = parser.parse_args()

    if args.board is None:
        parser.error("--board is required (this runs on hardware)")
    for var in ("TI_CGT_C7000_PATH",):
        if os.environ.get(var) is None:
            parser.error(f"{var} is not set")

    host = get_board_hostname(args.board)
    work_dir = Path(tempfile.mkdtemp(prefix=f"diag_persistent_c{args.channels}_"))
    module_path, input_data = _build(args.channels, work_dir)

    local_in = work_dir / "input.bin"
    local_in.write_bytes(np.ascontiguousarray(input_data).tobytes())

    # c7x_runtime.py needs only ctypes + numpy on the board (TVM import is lazy),
    # matching how dsp-tests/test_c7x_vm_dsp.py drives the board.
    c7x_runtime_py = (Path(__file__).resolve().parents[3] / "python" / "tvm" /
                      "contrib" / "c7x" / "c7x_runtime.py")

    _ssh(host, f"mkdir -p {_REMOTE_DIR}")
    for src, dst in ((module_path, "lib0.out"), (local_in, "input.bin"),
                     (c7x_runtime_py, "c7x_runtime.py")):
        subprocess.run(["scp", "-q", str(src), f"root@{host}:{_REMOTE_DIR}/{dst}"],
                       check=True, capture_output=True)

    # One VM, one connection, three inferences: exactly what a deployed app does.
    # The CLI's standalone `infer` verb cannot be used here -- the IO reservation
    # is per-connection, so a fresh CLI process reports capacity 0.
    script = rf"""
import sys
import numpy as np
sys.path.insert(0, '{_REMOTE_DIR}')
from c7x_runtime import C7xVirtualMachine

inp = np.frombuffer(open('{_REMOTE_DIR}/input.bin','rb').read(),
                    dtype='float32').reshape(1, 3, {_HW}, {_HW})
vm = C7xVirtualMachine('{_REMOTE_DIR}/lib0.out',
                       so_path='/usr/local/lib/libc7x_arm_runtime.so')
for i in (1, 2, 3):
    out = vm.run_nocopy(inp).copy()
    out.tofile('{_REMOTE_DIR}/vmout%d.bin' % i)
    print('INFER %d done shape=%s' % (i, out.shape))
vm.close()
"""
    remote_script = f"{_REMOTE_DIR}/persistent_load.py"
    subprocess.run(["ssh", f"root@{host}", f"cat > {remote_script}"],
                   input=script, text=True, check=True, capture_output=True)
    r = _ssh(host, f"cd {_REMOTE_DIR} && python3 {remote_script} 2>&1")
    log = r.stdout + r.stderr
    print("--- board run ---")
    print(log.strip()[:2500])

    outs = []
    for i in (1, 2, 3):
        local_out = work_dir / f"vmout{i}.bin"
        subprocess.run(["scp", "-q", f"root@{host}:{_REMOTE_DIR}/vmout{i}.bin",
                        str(local_out)], check=False, capture_output=True)
        if not local_out.exists() or local_out.stat().st_size == 0:
            print(f"\ninference #{i} produced no output -- see the board log above")
            return 1
        outs.append(np.fromfile(local_out, dtype=np.float32))

    print("\n--- per-call TIDL lines (one loaded module, 3 inferences) ---")
    for line in log.splitlines():
        if "process_tidl_subgraph" in line or "INFER" in line:
            print(f"  {line.strip()}")

    a, b, c = outs
    print(f"\ninference 1: {a.shape} first4={a[:4]}")
    print(f"inference 2: {b.shape} first4={b[:4]}")
    print(f"inference 3: {c.shape} first4={c[:4]}")

    nd = int((a != b).sum())
    nd3 = int((a != c).sum())
    print(f"inference 3 vs 1: {nd3} of {len(a)} elements differ")
    print(f"inference 2 vs 1: {nd} of {len(a)} elements differ")
    if nd == 0 and nd3 == 0:
        print("\n=> CLEAN: all three inferences agree. Note a small output "
              "cannot exercise\n   size-related faults -- raise --channels to "
              "widen the output if that is\n   what you are testing.")
        return 0

    n = len(a)
    shift = next((s for s in range(1, n) if np.array_equal(b[s:], a[: n - s])), None)
    if shift is not None:
        print(f"\n=> DIVERGES: out2[{shift}:] == out1[:{n - shift}] "
              f"(shift {shift} floats = {shift * 4} bytes).\n   A tail that "
              "repeats the head means something wrote past the output\n   "
              "allocation; check its size against TIDL's requirement with\n   "
              "diag_tidl_output_sizing.py.")
    else:
        print(f"\n=> DIVERGES in {nd} elements, but not as a head-copy; "
              "inspect the board log above.")
    return 0


if __name__ == "__main__":
    sys.exit(main())
