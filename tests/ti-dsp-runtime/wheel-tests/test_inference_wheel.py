"""Tests for the tvm-ti-c7x-inference (aarch64) wheel.

These tests verify the inference wheel's bundled artifacts and path
resolution.  Tests that load libc7x_arm_runtime.so or run inference
are skipped on x86 (the .so is aarch64-only).  Artifact tests are
skipped when running from a source tree.

The TestInferenceWheelE2E class compiles a model on the host, installs
the inference wheel on the AM67A board via SSH, and runs inference
using C7xVirtualMachine from the wheel-installed package.
"""

import os
import platform
import subprocess
import tempfile
from pathlib import Path

import numpy as np
import pytest

pytestmark = [pytest.mark.quick]

_IS_AARCH64 = platform.machine() in ("aarch64", "arm64")


def _has_bundled_firmware():
    """True when running from an installed wheel with firmware."""
    try:
        from tvm.data.ti_dsp.paths import get_ti_dsp_path
        return (get_ti_dsp_path() / "firmware" / "c7x_compute.out").exists()
    except ImportError:
        return False


_skip_source_tree = pytest.mark.skipif(
    not _has_bundled_firmware(),
    reason="Firmware artifacts only available in installed wheel",
)


class TestInferenceWheelContents:
    """Verify bundled artifacts in the inference wheel."""

    def test_c7x_runtime_imports(self):
        """C7xVirtualMachine must be importable without libtvm.so."""
        from tvm.contrib.c7x.c7x_runtime import C7xVirtualMachine
        assert C7xVirtualMachine is not None

    def test_paths_module_imports(self):
        """Path resolution module must be importable."""
        from tvm.data.ti_dsp.paths import get_ti_dsp_path
        assert get_ti_dsp_path().is_dir()

    @_skip_source_tree
    def test_firmware_bundled(self):
        """Firmware and ARM client must be in the data directory."""
        from tvm.data.ti_dsp.paths import get_ti_dsp_path
        fw_dir = get_ti_dsp_path() / "firmware"
        assert (fw_dir / "c7x_compute.out").exists(), "DSP firmware missing"
        assert (fw_dir / "c7x_compute").exists(), "ARM client missing"
        assert (fw_dir / "libc7x_arm_runtime.so").exists(), "ARM runtime missing"

    def test_arm_runtime_so_path_resolution(self):
        """find_c7x_arm_runtime_so returns path only on aarch64."""
        from tvm.data.ti_dsp.paths import find_c7x_arm_runtime_so
        result = find_c7x_arm_runtime_so()
        if _IS_AARCH64:
            assert result is not None
        else:
            assert result is None

    @_skip_source_tree
    def test_firmware_sizes_reasonable(self):
        """Bundled binaries must have non-trivial file sizes."""
        from tvm.data.ti_dsp.paths import get_ti_dsp_path
        fw_dir = get_ti_dsp_path() / "firmware"
        assert (fw_dir / "c7x_compute.out").stat().st_size > 1_000_000, \
            "Firmware too small — likely corrupt"
        assert (fw_dir / "libc7x_arm_runtime.so").stat().st_size > 10_000, \
            "ARM runtime too small — likely corrupt"


# ---------------------------------------------------------------------------
# E2E inference: compile on host, run on board via inference wheel
# ---------------------------------------------------------------------------

def _find_inference_wheel():
    """Find the aarch64 inference wheel in the build artifacts."""
    search_dirs = [
        Path.cwd() / "artifacts",
        Path.cwd() / "packaging" / "ti_dsp" / "staging-arm64" / "dist",
    ]
    tvm_home = os.environ.get("TVM_HOME")
    if tvm_home:
        search_dirs.append(Path(tvm_home) / "packaging" / "ti_dsp" / "staging-arm64" / "dist")
        search_dirs.append(Path(tvm_home) / "artifacts")
    for d in search_dirs:
        wheels = list(d.glob("tvm_ti_c7x_inference-*.whl")) if d.is_dir() else []
        if wheels:
            return wheels[0]
    return None


def _board_reachable(board):
    """Check if the board is reachable via SSH."""
    try:
        result = subprocess.run(
            ["ssh", "-o", "ConnectTimeout=5", "-o", "StrictHostKeyChecking=no",
             f"root@{board}", "echo ok"],
            capture_output=True, text=True, timeout=10,
        )
        return result.returncode == 0
    except (subprocess.TimeoutExpired, OSError):
        return False


_BOARD = os.environ.get("AM67A_TARGET", "am67a")

_skip_no_wheel = pytest.mark.skipif(
    _find_inference_wheel() is None,
    reason="aarch64 inference wheel not found (build with: build_wheel.sh --target arm64)",
)
_skip_no_board = pytest.mark.skipif(
    not _board_reachable(_BOARD),
    reason=f"AM67A board ({_BOARD}) not reachable via SSH",
)


class TestInferenceWheelE2E:
    """Compile on host, install inference wheel on board, run inference."""

    @_skip_no_wheel
    @_skip_no_board
    def test_mlp_inference_via_wheel(self):
        """Compile MLP on host, run on AM67A using the inference wheel."""
        import sys
        sys.path.insert(0, str(Path(__file__).parent.parent / "dsp-tests"))
        sys.path.insert(0, str(Path(__file__).parent.parent / "dsp-cpp"))
        from dsp_utils import build_dsp_dynmod, compile_for_dsp
        from model_utils import create_mlp_model

        import tvm
        from tvm import relax

        # --- Compile on host ---
        tvm_mod, _, _ = create_mlp_model(input_size=64, hidden_size=32, output_size=8)

        # CPU reference
        ex = relax.build(tvm_mod, target=tvm.target.Target("llvm"))
        vm = relax.VirtualMachine(ex, tvm.cpu())
        np.random.seed(42)
        inp = np.random.randn(1, 64).astype(np.float32)
        ref_out = vm["main"](inp)
        if hasattr(ref_out, "__getitem__") and not hasattr(ref_out, "numpy"):
            ref_out = ref_out[0]
        ref = ref_out.numpy()

        # DSP module
        gen_dir = Path(tempfile.mkdtemp(prefix="wheel_e2e_gen_"))
        compile_for_dsp(tvm_mod, "c_static -mcpu=c7x -use-cpp-api=1", output_dir=gen_dir)
        build_dir = Path(tempfile.mkdtemp(prefix="wheel_e2e_build_"))
        weights = gen_dir / "weights.bin"
        lib0 = build_dsp_dynmod(
            generated_dir=gen_dir,
            build_dir=build_dir,
            weights_file=weights if weights.exists() else None,
        )

        # --- Deploy to board ---
        wheel = _find_inference_wheel()
        remote = f"root@{_BOARD}"
        remote_dir = "/tmp/_wheel_e2e_test"

        subprocess.run(
            ["ssh", "-o", "StrictHostKeyChecking=no", remote,
             f"rm -rf {remote_dir} && mkdir -p {remote_dir}"],
            check=True, timeout=10,
        )

        # SCP artifacts
        inp_path = gen_dir / "inp.bin"
        ref_path = gen_dir / "ref.bin"
        inp_path.write_bytes(np.ascontiguousarray(inp).tobytes())
        ref_path.write_bytes(np.ascontiguousarray(ref).tobytes())

        for local in [str(lib0), str(inp_path), str(ref_path), str(wheel)]:
            subprocess.run(
                ["scp", "-q", local, f"{remote}:{remote_dir}/"],
                check=True, timeout=120,
            )

        # --- Run on board ---
        wheel_name = wheel.name
        script = f"""
import subprocess, sys, json
import numpy as np

# Install inference wheel (--no-deps: board has no internet for PyPI)
subprocess.run(
    [sys.executable, '-m', 'pip', 'install', '--quiet', '--force-reinstall',
     '--no-deps', '{remote_dir}/{wheel_name}'],
    check=True,
)

from tvm.contrib.c7x import C7xVirtualMachine

vm = C7xVirtualMachine('{remote_dir}/lib0.out')
inp = np.frombuffer(open('{remote_dir}/inp.bin','rb').read(), dtype='float32').reshape(1,64)
ref = np.frombuffer(open('{remote_dir}/ref.bin','rb').read(), dtype='float32').reshape(1,8)

out = vm.run_nocopy(inp)
max_diff = float(np.max(np.abs(out - ref)))
cycles = vm.last_cycles
vm.close()

r = {{'max_diff': max_diff, 'cycles': cycles, 'shape': list(out.shape)}}
print(json.dumps(r))
"""
        import shlex
        result = subprocess.run(
            ["ssh", "-o", "ConnectTimeout=10", remote,
             f"python3 -c {shlex.quote(script)}"],
            capture_output=True, text=True, timeout=300,
        )

        assert result.returncode == 0, (
            f"Remote inference failed (rc={result.returncode}):\n"
            f"stdout: {result.stdout}\nstderr: {result.stderr}"
        )

        import json
        for line in reversed(result.stdout.strip().splitlines()):
            if line.strip().startswith("{"):
                r = json.loads(line)
                break
        else:
            pytest.fail(f"No JSON output:\nstdout: {result.stdout}\nstderr: {result.stderr}")

        assert r["shape"] == [1, 8], f"Wrong shape: {r['shape']}"
        assert r["max_diff"] < 1e-3, f"max diff {r['max_diff']:.6f} exceeds 1e-3"
        assert r["cycles"] > 0, "No cycle count reported"
