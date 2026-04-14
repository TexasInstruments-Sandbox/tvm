"""
C7xVirtualMachine unit and integration tests.

Tests the Python Arm-side wrapper (C7xVirtualMachine) that provides a
relax.VirtualMachine-compatible interface for DSP inference.

Test layers
-----------
1. **Unit tests** (no hardware, run anywhere):
   - Struct layout: _C7xTensorDesc must be exactly 80 bytes
   - Dtype mapping: _DLTYPE_TO_NUMPY covers all required DLPack types
   - API contract: bad .so path raises RuntimeError
   - API contract: bad module path raises FileNotFoundError

2. **Integration tests** (require libc7x_arm_runtime.so and AM67A DSP):
   - Standard path: vm["main"](tvm.nd.array(x)) matches CPU reference
   - Zero-copy output: run_nocopy() returns correct numpy views
   - Zero-copy input: create_input() → vm["main"](pre_staged) correct results
   - Context manager: __enter__/__exit__ lifecycle works
   - Multi-inference: repeated calls return consistent results
   - last_cycles property: returns positive cycle count after inference

Run unit tests anywhere:
    pytest test_c7x_vm_dsp.py -v -k "unit"

Run on AM67A board (requires libc7x_arm_runtime.so installed):
    pytest test_c7x_vm_dsp.py -v
"""

import ctypes
import shutil
import subprocess
import sys
import tempfile
from pathlib import Path

import numpy as np
import pytest

_THIS_DIR = Path(__file__).parent
_DSP_CPP_DIR = _THIS_DIR.parent / "dsp-cpp"
sys.path.insert(0, str(_DSP_CPP_DIR))

# ruff: noqa: E402
from dsp_utils import build_dsp_dynmod, compile_for_dsp  # pyright: ignore[reportMissingImports]
from model_utils import create_mlp_model  # pyright: ignore[reportMissingImports]

import tvm
from tvm import relax
from tvm.contrib.c7x import C7xVirtualMachine  # pyright: ignore[reportMissingImports]
from tvm.contrib.c7x.c7x_runtime import (  # pyright: ignore[reportMissingImports]
    _C7xTensorDesc,
    _DLTYPE_TO_NUMPY,
    _DL_FLOAT,
    _DL_INT,
    _DL_UINT,
)
from tvm.relax.backend.tidl import TIDLBuildResult  # pyright: ignore[reportMissingImports]

pytestmark = [pytest.mark.c7x_only, pytest.mark.core]

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

_LIB_NAME = "libc7x_arm_runtime.so"


def _has_c7x_vm_lib() -> bool:
    """True if libc7x_arm_runtime.so is loadable on this system."""
    try:
        ctypes.CDLL(_LIB_NAME)
        return True
    except OSError:
        return False


def _has_c7x_firmware() -> bool:
    """True if the C7x DSP firmware is running (only on AM67A).

    Uses a fast sysfs check (no IPC) to avoid hanging test collection
    when the firmware is slow to respond.  Checks for the rpmsg endpoint
    that the c7x_compute service announces on startup.
    """
    if not _has_c7x_vm_lib():
        return False
    import glob
    # The firmware announces "rpmsg_chrdev" on the C7x DSP endpoint.
    # Presence of any rpmsg channel for the 7e000000.dsp device indicates
    # the firmware is loaded and the service is active.
    for dev_link in glob.glob("/sys/class/rpmsg/rpmsg_ctrl*/device"):
        try:
            target = Path(dev_link).resolve()
            if "7e000000.dsp" in str(target):
                return True
        except OSError:
            pass
    return False


def _compile_mlp_lib0() -> Path:
    """Compile a small MLP to lib0.out and return its path."""
    tvm_mod, _, _ = create_mlp_model(input_size=64, hidden_size=32, output_size=8)
    target = "c_static -mcpu=c7x -use-cpp-api=1"
    gen_dir = Path(tempfile.mkdtemp(prefix="c7x_vm_test_"))
    compile_for_dsp(tvm_mod, target_string=target, output_dir=gen_dir)
    build_dir = Path(tempfile.mkdtemp(prefix="c7x_vm_build_"))
    weights = gen_dir / "weights.bin"
    module_path = build_dsp_dynmod(
        generated_dir=gen_dir,
        build_dir=build_dir,
        weights_file=weights if weights.exists() else None,
    )
    return module_path


def _cpu_reference_mlp(input_data: np.ndarray) -> np.ndarray:
    """Run the same MLP on CPU via TVM RelaxVM for reference."""
    tvm_mod, _, _ = create_mlp_model(input_size=64, hidden_size=32, output_size=8)
    ex = relax.build(tvm_mod, target=tvm.target.Target("llvm"))
    vm = relax.VirtualMachine(ex, tvm.cpu())
    out = vm["main"](tvm.nd.array(input_data))
    return out.numpy()


def _find_c7x_runtime_binary() -> Path | None:
    """Return path to test_c7x_runtime binary, or None if not found."""
    path = shutil.which("test_c7x_runtime")
    if path:
        return Path(path)
    # Common install location after ./build.sh deploy
    candidate = Path("/usr/local/bin/test_c7x_runtime")
    if candidate.exists():
        return candidate
    return None


def _has_c7x_runtime_binary() -> bool:
    """True if test_c7x_runtime binary is available."""
    return _find_c7x_runtime_binary() is not None


# ---------------------------------------------------------------------------
# Markers
# ---------------------------------------------------------------------------

requires_lib = pytest.mark.skipif(
    not _has_c7x_vm_lib(),
    reason="libc7x_arm_runtime.so not found — install on AM67A board",
)
requires_firmware = pytest.mark.skipif(
    not _has_c7x_firmware(),
    reason="c7x_compute firmware not reachable — run on AM67A board",
)

# ---------------------------------------------------------------------------
# Unit tests (no hardware required)
# ---------------------------------------------------------------------------


class TestStructLayout:
    """Verify the ctypes struct matches the C layout exactly."""

    @pytest.mark.core
    def test_c7x_tensor_desc_size(self):
        """_C7xTensorDesc must be 80 bytes to match c7x_tensor_desc_t in C."""
        assert ctypes.sizeof(_C7xTensorDesc) == 80, (
            f"Expected 80 bytes, got {ctypes.sizeof(_C7xTensorDesc)}. "
            "Padding mismatch with c7x_tensor_desc_t in c7x_compute_client.h."
        )

    @pytest.mark.core
    def test_c7x_tensor_desc_field_offsets(self):
        """Field offsets must match the C struct layout."""
        # data: offset 0
        assert _C7xTensorDesc.data.offset == 0
        # data_size: offset 8 (after 8-byte void*)
        assert _C7xTensorDesc.data_size.offset == 8
        # ndim: offset 16 (after 8-byte size_t)
        assert _C7xTensorDesc.ndim.offset == 16
        # dtype_code: offset 20
        assert _C7xTensorDesc.dtype_code.offset == 20
        # dtype_bits: offset 24
        assert _C7xTensorDesc.dtype_bits.offset == 24
        # _pad: offset 28
        assert _C7xTensorDesc._pad.offset == 28  # noqa: SLF001
        # shape: offset 32 (8-byte aligned after 4-byte _pad)
        assert _C7xTensorDesc.shape.offset == 32

    @pytest.mark.core
    def test_dtype_mapping_coverage(self):
        """_DLTYPE_TO_NUMPY must cover the dtypes used in inference."""
        required = [
            (_DL_FLOAT, 32),  # float32
            (_DL_FLOAT, 16),  # float16
            (_DL_INT, 8),     # int8
            (_DL_INT, 32),    # int32
            (_DL_INT, 64),    # int64
            (_DL_UINT, 8),    # uint8
        ]
        for key in required:
            assert key in _DLTYPE_TO_NUMPY, (
                f"Missing dtype ({key[0]}, {key[1]}) in _DLTYPE_TO_NUMPY"
            )

    @pytest.mark.core
    def test_dtype_mapping_numpy_types(self):
        """Mapped numpy dtypes must be correct numpy scalar types."""
        assert _DLTYPE_TO_NUMPY[(_DL_FLOAT, 32)] is np.float32
        assert _DLTYPE_TO_NUMPY[(_DL_FLOAT, 16)] is np.float16
        assert _DLTYPE_TO_NUMPY[(_DL_INT, 8)] is np.int8
        assert _DLTYPE_TO_NUMPY[(_DL_INT, 32)] is np.int32
        assert _DLTYPE_TO_NUMPY[(_DL_UINT, 8)] is np.uint8


class TestAPIContract:
    """Verify the public API contract without hardware."""

    @pytest.mark.core
    def test_init_bad_so_raises(self, tmp_path):
        """C7xVirtualMachine must raise RuntimeError for a missing .so."""
        with pytest.raises(RuntimeError, match="Cannot load"):
            C7xVirtualMachine("/tmp/fake_lib0.out",
                              so_path=str(tmp_path / "nonexistent.so"))

    @pytest.mark.core
    @requires_lib
    def test_init_bad_module_lazy(self):
        """FileNotFoundError raised on first inference if module path is invalid."""
        vm = C7xVirtualMachine("/tmp/this_does_not_exist.out")
        with pytest.raises(FileNotFoundError):
            vm["main"](np.zeros((1, 64), dtype=np.float32))

    @pytest.mark.core
    @requires_lib
    def test_is_loaded_before_first_call(self):
        """is_loaded must be False before first inference call."""
        vm = C7xVirtualMachine("/tmp/fake.out")
        assert not vm.is_loaded

    @pytest.mark.core
    @requires_lib
    def test_close_is_idempotent(self):
        """close() on an unloaded VM must not raise."""
        vm = C7xVirtualMachine("/tmp/fake.out")
        vm.close()
        vm.close()  # second close must be a no-op

    @pytest.mark.core
    def test_getitem_returns_callable(self, tmp_path):
        """vm["main"] must return a callable without triggering IO."""
        # We can't call __getitem__ without the .so, but we can verify it's
        # a function attribute look-up that doesn't touch filesystem at
        # construction time.
        try:
            vm = C7xVirtualMachine("/tmp/fake.out", so_path=str(tmp_path / "x.so"))
        except RuntimeError:
            pytest.skip("Library not found (expected on dev PC)")
        fn = vm["main"]
        assert callable(fn)

    @pytest.mark.core
    def test_context_manager_cleanup(self, tmp_path):
        """Context manager __exit__ must call close without error."""
        try:
            with C7xVirtualMachine("/tmp/fake.out",
                                   so_path=str(tmp_path / "x.so")) as vm:
                assert not vm.is_loaded
        except RuntimeError:
            pytest.skip("Library not found")


# ---------------------------------------------------------------------------
# Integration tests (require libc7x_arm_runtime.so + firmware)
# ---------------------------------------------------------------------------


@pytest.fixture(scope="module")
def mlp_lib0():
    """Compile a small MLP to lib0.out (once per module)."""
    return _compile_mlp_lib0()


@pytest.fixture(scope="module")
def mlp_input():
    """Fixed input for reproducible inference comparisons."""
    np.random.seed(0)
    return np.random.randn(1, 64).astype(np.float32)


@pytest.fixture(scope="module")
def mlp_cpu_ref(mlp_input):
    """CPU reference output for the MLP."""
    return _cpu_reference_mlp(mlp_input)


@requires_firmware
class TestC7xVMInference:
    """Integration tests: C7xVirtualMachine running against live firmware."""

    @pytest.mark.core
    def test_standard_inference_matches_cpu(self, mlp_lib0, mlp_input, mlp_cpu_ref):
        """vm["main"](tvm.nd.array(x)) result must match CPU reference."""
        vm = C7xVirtualMachine(mlp_lib0)
        inp = tvm.nd.array(mlp_input)
        out = vm["main"](inp)
        vm.close()

        dsp_out = out.numpy()
        np.testing.assert_allclose(dsp_out, mlp_cpu_ref, rtol=1e-3, atol=1e-3,
                                   err_msg="DSP output diverges from CPU reference")

    @pytest.mark.core
    def test_is_loaded_after_inference(self, mlp_lib0, mlp_input):
        """is_loaded must be True after the first successful inference."""
        vm = C7xVirtualMachine(mlp_lib0)
        assert not vm.is_loaded
        vm["main"](mlp_input)
        assert vm.is_loaded
        vm.close()
        assert not vm.is_loaded

    @pytest.mark.core
    def test_last_cycles_positive(self, mlp_lib0, mlp_input):
        """last_cycles must be a positive integer after inference."""
        with C7xVirtualMachine(mlp_lib0) as vm:
            vm["main"](mlp_input)
            assert vm.last_cycles > 0, "Expected positive cycle count from DSP"

    @pytest.mark.core
    def test_context_manager_closes_on_exit(self, mlp_lib0, mlp_input):
        """Context manager must unload the module on __exit__."""
        vm_ref = None
        with C7xVirtualMachine(mlp_lib0) as vm:
            vm_ref = vm
            vm["main"](mlp_input)
            assert vm.is_loaded
        assert not vm_ref.is_loaded

    def test_repeated_inference_consistent(self, mlp_lib0, mlp_input, mlp_cpu_ref):
        """Three consecutive inferences must return the same result."""
        with C7xVirtualMachine(mlp_lib0) as vm:
            results = [vm["main"](mlp_input).numpy() for _ in range(3)]

        for i, r in enumerate(results):
            np.testing.assert_allclose(
                r, mlp_cpu_ref, rtol=1e-3, atol=1e-3,
                err_msg=f"Inference #{i} diverges from reference"
            )
        # Ensure runs are bit-exact with each other
        np.testing.assert_array_equal(results[0], results[1])
        np.testing.assert_array_equal(results[1], results[2])

    @pytest.mark.core
    def test_as_vm_factory(self, mlp_lib0, mlp_input, mlp_cpu_ref):
        """TIDLBuildResult.as_vm() must return a working C7xVirtualMachine.

        Manually constructs a TIDLBuildResult from the mlp_lib0 fixture
        (avoids the TIDL import .so dependency) and calls as_vm().
        """
        # Build a minimal TIDLBuildResult that wraps the pre-compiled lib0.out.
        # weights.bin may or may not exist alongside lib0.out; pass lib0 dir.
        lib0_path = Path(mlp_lib0)
        result = TIDLBuildResult(
            module_path=lib0_path,
            weights_path=lib0_path.parent / "weights.bin",
            gen_dir=lib0_path.parent,
        )
        vm = result.as_vm()
        assert isinstance(vm, C7xVirtualMachine)

        out = vm["main"](mlp_input)
        vm.close()
        np.testing.assert_allclose(
            out.numpy(), mlp_cpu_ref, rtol=1e-3, atol=1e-3,
            err_msg="as_vm() inference diverges from CPU reference"
        )

    @pytest.mark.core
    def test_numpy_input_accepted(self, mlp_lib0, mlp_input, mlp_cpu_ref):
        """C7xVirtualMachine must accept plain numpy arrays (not just tvm.nd)."""
        with C7xVirtualMachine(mlp_lib0) as vm:
            out = vm["main"](mlp_input)  # plain numpy array
        np.testing.assert_allclose(out.numpy(), mlp_cpu_ref, rtol=1e-3, atol=1e-3)

    @pytest.mark.core
    def test_output_is_tvm_ndarray(self, mlp_lib0, mlp_input):
        """vm["main"] must return a tvm.nd.NDArray."""
        with C7xVirtualMachine(mlp_lib0) as vm:
            out = vm["main"](mlp_input)
        assert isinstance(out, tvm.nd.NDArray), (
            f"Expected tvm.nd.NDArray, got {type(out)}"
        )
        assert out.shape == (1, 8)
        assert str(out.dtype) == "float32"


@requires_firmware
class TestRunNocopy:
    """Zero-copy output via run_nocopy()."""

    @pytest.mark.core
    def test_run_nocopy_returns_numpy(self, mlp_lib0, mlp_input, mlp_cpu_ref):
        """run_nocopy() must return a numpy array backed by result DDR."""
        with C7xVirtualMachine(mlp_lib0) as vm:
            out = vm.run_nocopy(mlp_input)

        assert isinstance(out, np.ndarray), (
            f"Expected np.ndarray from run_nocopy, got {type(out)}"
        )
        np.testing.assert_allclose(out, mlp_cpu_ref, rtol=1e-3, atol=1e-3)

    def test_run_nocopy_matches_standard_run(self, mlp_lib0, mlp_input):
        """run_nocopy() and vm["main"]() must produce identical results."""
        with C7xVirtualMachine(mlp_lib0) as vm:
            out_copy = vm["main"](mlp_input).numpy()
            out_view = vm.run_nocopy(mlp_input)

        np.testing.assert_array_equal(
            out_copy, out_view,
            err_msg="run_nocopy() result differs from standard vm['main'] result"
        )

    @pytest.mark.core
    def test_last_nocopy_outputs_list(self, mlp_lib0, mlp_input):
        """_last_nocopy_outputs must be populated after run_nocopy()."""
        with C7xVirtualMachine(mlp_lib0) as vm:
            vm.run_nocopy(mlp_input)
            # _last_nocopy_outputs holds the view; it should be non-empty
            assert len(vm._last_nocopy_outputs) > 0  # noqa: SLF001


@requires_firmware
class TestCreateInput:
    """Zero-copy input via create_input()."""

    @pytest.mark.core
    def test_create_input_shape_dtype(self, mlp_lib0):
        """create_input() must return a tensor with the correct shape and dtype."""
        with C7xVirtualMachine(mlp_lib0) as vm:
            inp = vm.create_input((1, 64), "float32")
        assert inp.shape == (1, 64)
        assert str(inp.dtype) == "float32"

    @pytest.mark.core
    def test_create_input_inference_correct(self, mlp_lib0, mlp_input, mlp_cpu_ref):
        """Inference via pre-staged create_input() tensor must match CPU reference."""
        with C7xVirtualMachine(mlp_lib0) as vm:
            inp = vm.create_input((1, 64), "float32")
            inp.copyfrom(mlp_input)  # write to staging buffer in-place
            out = vm["main"](inp)

        np.testing.assert_allclose(
            out.numpy(), mlp_cpu_ref, rtol=1e-3, atol=1e-3,
            err_msg="create_input() inference diverges from CPU reference"
        )

    def test_create_input_matches_standard_path(self, mlp_lib0, mlp_input):
        """create_input() and standard numpy input must give identical results."""
        with C7xVirtualMachine(mlp_lib0) as vm:
            # Standard path
            out_standard = vm["main"](mlp_input).numpy()
            # Zero-copy input path
            inp = vm.create_input((1, 64), "float32")
            inp.copyfrom(mlp_input)
            out_prestaged = vm["main"](inp).numpy()

        np.testing.assert_array_equal(
            out_standard, out_prestaged,
            err_msg="create_input() and standard path give different results"
        )

    @pytest.mark.core
    def test_create_input_registered_in_slots(self, mlp_lib0):
        """create_input() must register the tensor in _input_slots dict."""
        with C7xVirtualMachine(mlp_lib0) as vm:
            before = len(vm._input_slots)  # noqa: SLF001
            vm.create_input((1, 64), "float32")
            after = len(vm._input_slots)  # noqa: SLF001

        assert after == before + 1, "create_input() did not register slot"

    def test_staging_offset_advances_after_create_input(self, mlp_lib0):
        """Staging alloc offset must advance by at least tensor size after create_input()."""
        with C7xVirtualMachine(mlp_lib0) as vm:
            offset_before = vm._staging_alloc_offset  # noqa: SLF001
            vm.create_input((1, 64), "float32")
            offset_after = vm._staging_alloc_offset  # noqa: SLF001

        # 1×64×4 bytes = 256 bytes; advance must be ≥ 256
        assert offset_after >= offset_before + 256

    @pytest.mark.core
    def test_staging_offset_from_elf_not_hardcoded(self, mlp_lib0):
        """Staging alloc offset must come from the real ELF size, not 256 MB."""
        with C7xVirtualMachine(mlp_lib0) as vm:
            _ = vm["main"](np.zeros((1, 64), dtype=np.float32))  # triggers load
            offset = vm._staging_alloc_offset  # noqa: SLF001

        # 256 MB = 268_435_456.  A real ELF is << 256 MB (typically 1-50 MB).
        # The offset should be the actual ELF size, NOT the hardcoded sentinel.
        assert offset < 256 * 1024 * 1024, (
            f"Staging offset {offset} looks like the hardcoded 256 MB sentinel; "
            "expected the actual ELF file size from c7x_client_get_input_data_offset()"
        )

    def test_close_resets_staging_slots(self, mlp_lib0):
        """close() must clear _input_slots and reset staging offset."""
        vm = C7xVirtualMachine(mlp_lib0)
        vm["main"](np.zeros((1, 64), dtype=np.float32))  # triggers load
        vm.create_input((1, 64), "float32")
        assert len(vm._input_slots) == 1  # noqa: SLF001
        vm.close()
        assert len(vm._input_slots) == 0  # noqa: SLF001
        assert vm._staging_alloc_offset == 0  # noqa: SLF001


# ---------------------------------------------------------------------------
# C++ test binary wrapper (test_c7x_runtime)
# ---------------------------------------------------------------------------

requires_c7x_cpp = pytest.mark.skipif(
    not _has_c7x_firmware() or not _has_c7x_runtime_binary(),
    reason="test_c7x_runtime binary or firmware not available — deploy and run on AM67A",
)


@requires_c7x_cpp
class TestC7xCpp:
    """Pytest wrapper around the test_c7x_runtime C++ binary.

    Exercises the c7x::Module C++ API end-to-end on AM67A.
    Requires: test_c7x_runtime in PATH or /usr/local/bin/, and firmware running.
    Build and deploy: cd src/runtime/ti_dsp/firmware/c7x/arm && ./build.sh deploy
    """

    @pytest.mark.core
    def test_cpp_standard_inference(self, mlp_lib0, mlp_input, mlp_cpu_ref):
        """C++ test binary: standard inference, reference comparison, create_input."""
        binary = _find_c7x_runtime_binary()
        assert binary is not None

        # Write input.bin and ref.bin to a temp directory
        tmp = Path(tempfile.mkdtemp(prefix="c7x_cpp_test_"))
        input_bin = tmp / "input.bin"
        ref_bin = tmp / "ref.bin"
        input_bin.write_bytes(np.ascontiguousarray(mlp_input).tobytes())
        ref_bin.write_bytes(np.ascontiguousarray(mlp_cpu_ref).tobytes())

        cmd = [
            str(binary),
            str(mlp_lib0),
            str(input_bin),
            "--shape", "1,64",
            "--dtype", "float32",
            "--ref", str(ref_bin),
            "--atol", "1e-3",
        ]
        result = subprocess.run(cmd, capture_output=True, text=True, timeout=120)

        if result.stdout:
            print(result.stdout)
        if result.stderr:
            print(result.stderr)

        assert result.returncode == 0, (
            f"test_c7x_runtime failed (rc={result.returncode}):\n"
            f"stdout: {result.stdout}\nstderr: {result.stderr}"
        )


# ---------------------------------------------------------------------------
# Standalone script mode
# ---------------------------------------------------------------------------


def main():
    """Run all tests in standalone mode (no pytest)."""
    import traceback

    passed = failed = skipped = 0

    def run(fn, name):
        nonlocal passed, failed, skipped
        try:
            fn()
            print(f"  PASS  {name}")
            passed += 1
        except pytest.skip.Exception as e:
            print(f"  SKIP  {name}: {e}")
            skipped += 1
        except Exception:
            print(f"  FAIL  {name}")
            traceback.print_exc()
            failed += 1

    suite = TestStructLayout()
    run(suite.test_c7x_tensor_desc_size, "struct_size")
    run(suite.test_c7x_tensor_desc_field_offsets, "struct_offsets")
    run(suite.test_dtype_mapping_coverage, "dtype_coverage")
    run(suite.test_dtype_mapping_numpy_types, "dtype_values")

    if _has_c7x_firmware():
        lib0 = _compile_mlp_lib0()
        inp = np.random.randn(1, 64).astype(np.float32)
        ref = _cpu_reference_mlp(inp)

        inf_suite = TestC7xVMInference()
        run(lambda: inf_suite.test_standard_inference_matches_cpu(lib0, inp, ref),
            "standard_inference")
        run(lambda: inf_suite.test_last_cycles_positive(lib0, inp),
            "last_cycles")
        run(lambda: inf_suite.test_numpy_input_accepted(lib0, inp, ref),
            "numpy_input")
        run(lambda: inf_suite.test_as_vm_factory(lib0, inp, ref),
            "as_vm_factory")

        nc_suite = TestRunNocopy()
        run(lambda: nc_suite.test_run_nocopy_returns_numpy(lib0, inp, ref),
            "run_nocopy")

        ci_suite = TestCreateInput()
        run(lambda: ci_suite.test_create_input_inference_correct(lib0, inp, ref),
            "create_input_inference")
        run(lambda: ci_suite.test_staging_offset_from_elf_not_hardcoded(lib0),
            "staging_offset_elf")

        if _has_c7x_runtime_binary():
            cpp_suite = TestC7xCpp()
            run(lambda: cpp_suite.test_cpp_standard_inference(lib0, inp, ref),
                "cpp_standard_inference")
        else:
            print("  SKIP  C++ tests (test_c7x_runtime binary not found)")
            skipped += 1
    else:
        print("  SKIP  Integration tests (no firmware)")
        skipped += 8

    print(f"\nResults: {passed} passed, {failed} failed, {skipped} skipped")
    return failed


if __name__ == "__main__":
    sys.exit(main())
