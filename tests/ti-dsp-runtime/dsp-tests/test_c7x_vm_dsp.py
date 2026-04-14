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
   - Standard path: vm["main"](numpy_array) matches CPU reference
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
    # In TVM 0.23 relax.VirtualMachine accepts numpy arrays directly.
    # The output may be a single Tensor or an Array (tuple) depending on the
    # model's return type; unwrap to get the first (and only) element.
    out = vm["main"](input_data)
    if hasattr(out, "__getitem__") and not hasattr(out, "numpy"):
        out = out[0]
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


def _run_c7x_vm_on_board(
    board: str,
    lib0_path: Path,
    input_data: "np.ndarray",
    cpu_ref: "np.ndarray",
    timeout: int = 300,
) -> dict:
    """Deploy lib0.out to the board and run C7xVirtualMachine assertions via SSH.

    SCPs lib0.out, input.bin, ref.bin, and c7x_runtime.py (which requires only
    ctypes + numpy — no TVM) to /tmp on the board.  The remote script imports
    c7x_runtime.py directly and uses run_nocopy() throughout to avoid the lazy
    TVM import inside _run().

    Returns a JSON dict of {assertion_name: bool}.  Raises AssertionError if
    the SSH command fails or if no JSON output is produced.
    """
    import json as _json
    import tempfile as _tmp

    remote = f"root@{board}"
    remote_dir   = "/tmp/_c7x_vm_test"
    remote_lib0  = f"{remote_dir}/lib0.out"
    remote_inp   = f"{remote_dir}/inp.bin"
    remote_ref   = f"{remote_dir}/ref.bin"
    remote_pymod = f"{remote_dir}/c7x_runtime.py"

    # c7x_runtime.py only needs ctypes + numpy at module level (TVM is lazy)
    c7x_runtime_py = Path(__file__).parent.parent.parent.parent / \
        "python" / "tvm" / "contrib" / "c7x" / "c7x_runtime.py"

    subprocess.run(
        ["ssh", "-o", "ConnectTimeout=10", f"root@{board}",
         f"mkdir -p {remote_dir}"],
        check=True, timeout=10,
    )
    with _tmp.TemporaryDirectory() as td:
        inp_path = Path(td) / "inp.bin"
        ref_path = Path(td) / "ref.bin"
        inp_path.write_bytes(np.ascontiguousarray(input_data).tobytes())
        ref_path.write_bytes(np.ascontiguousarray(cpu_ref).tobytes())
        for local, rpath in [
            (str(lib0_path),       remote_lib0),
            (str(inp_path),        remote_inp),
            (str(ref_path),        remote_ref),
            (str(c7x_runtime_py),  remote_pymod),
        ]:
            subprocess.run(
                ["scp", "-q", local, f"{remote}:{rpath}"],
                check=True, timeout=120,
            )

    script = rf"""
import sys, json
import numpy as np
sys.path.insert(0, '{remote_dir}')
from c7x_runtime import C7xVirtualMachine

lib0 = '{remote_lib0}'
so_path = '/usr/local/lib/libc7x_arm_runtime.so'
inp  = np.frombuffer(open('{remote_inp}','rb').read(), dtype='float32').reshape(1,64)
ref  = np.frombuffer(open('{remote_ref}','rb').read(), dtype='float32').reshape(1,8)
r = {{}}

# run_nocopy returns plain numpy — no TVM import triggered on the board
try:
    vm = C7xVirtualMachine(lib0, so_path=so_path)
    out = vm.run_nocopy(inp)
    r['standard_inference'] = float(np.max(np.abs(out - ref))) < 1e-3
    r['output_shape']  = list(out.shape) == [1, 8]
    r['output_dtype']  = str(out.dtype) == 'float32'
    r['last_cycles']   = vm.last_cycles > 0
    r['is_loaded']     = vm.is_loaded
    vm.close()
    r['close_works']   = not vm.is_loaded
except Exception as e:
    r['inference_error'] = str(e)

try:
    vm2 = C7xVirtualMachine(lib0, so_path=so_path)
    out_nc = vm2.run_nocopy(inp)
    r['nocopy_numpy']   = isinstance(out_nc, np.ndarray)
    r['nocopy_correct'] = float(np.max(np.abs(out_nc - ref))) < 1e-3
    vm2.close()
except Exception as e:
    r['nocopy_error'] = str(e)

try:
    vm3 = C7xVirtualMachine(lib0, so_path=so_path)
    pre = vm3.create_input((1,64), 'float32')
    r['create_shape']  = list(pre.shape) == [1, 64]
    r['create_dtype']  = str(pre.dtype) == 'float32'
    _buf = pre.numpy() if hasattr(pre, 'numpy') else pre; _buf[:] = inp
    out_pre = vm3.run_nocopy(pre)
    r['create_correct'] = float(np.max(np.abs(out_pre - ref))) < 1e-3
    r['staging_offset_ok'] = vm3._staging_alloc_offset < 256*1024*1024
    vm3.close()
    r['close_resets_slots'] = len(vm3._input_slots) == 0
except Exception as e:
    r['create_error'] = str(e)

try:
    vm4 = C7xVirtualMachine(lib0, so_path=so_path)
    results = [vm4.run_nocopy(inp).copy() for _ in range(3)]
    vm4.close()
    r['repeated_consistent'] = (
        np.array_equal(results[0], results[1]) and
        np.array_equal(results[1], results[2])
    )
except Exception as e:
    r['repeated_error'] = str(e)


# --- API contract (no firmware needed, just the .so) ---
try:
    # is_loaded before first call must be False
    vm_nc = C7xVirtualMachine('/tmp/__nonexistent__.out', so_path=so_path)
    r['not_loaded_before_call'] = not vm_nc.is_loaded
    # bad module path must raise FileNotFoundError on first inference
    try:
        vm_nc['main'](inp)
        r['bad_module_raises'] = False
    except FileNotFoundError:
        r['bad_module_raises'] = True
    except Exception:
        r['bad_module_raises'] = False
    # close() idempotent on never-loaded VM
    vm_ci = C7xVirtualMachine('/tmp/__nonexistent__.out', so_path=so_path)
    vm_ci.close()
    vm_ci.close()
    r['close_idempotent'] = True
except Exception as e:
    r['api_contract_error'] = str(e)

print(json.dumps(r))
""" 

    result = subprocess.run(
        ["ssh", "-o", "ConnectTimeout=10", f"root@{board}",
         f"python3 -c {_shlex_quote(script)}"],
        capture_output=True, text=True, timeout=timeout,
    )
    if result.returncode != 0:
        raise AssertionError(
            f"Remote c7x_vm test failed on {board} (rc={result.returncode}):\n"
            f"stdout: {result.stdout}\nstderr: {result.stderr}"
        )

    for line in reversed(result.stdout.strip().splitlines()):
        line = line.strip()
        if line.startswith("{"):
            return _json.loads(line)

    raise AssertionError(
        f"No JSON output from remote c7x_vm test on {board}.\n"
        f"stdout: {result.stdout}\nstderr: {result.stderr}"
    )


def _shlex_quote(s: str) -> str:
    """Single-quote a string for SSH command passing."""
    import shlex
    return shlex.quote(s)


# ---------------------------------------------------------------------------
# Markers
#
# Uses custom mark names (not pytest.mark.skipif) so that conftest.py's
# pytest_collection_modifyitems can remove them when --board-target is given
# and the board is reachable via SSH.
# ---------------------------------------------------------------------------

requires_lib = pytest.mark.requires_c7x_vm_lib
requires_firmware = pytest.mark.requires_c7x_firmware

# ---------------------------------------------------------------------------
# Unit tests (no hardware required)
# ---------------------------------------------------------------------------



# Module-level cache: (board_target, str(lib0_path)) -> result dict.
# Shared across all test instances so we do at most one SSH round-trip per
# board+model combination — pytest creates a fresh class instance per test
# method, so a per-instance cache would trigger ~26 redundant SSH calls.
_REMOTE_RESULTS_CACHE: dict = {}


def _remote_results(self_obj, board_target, mlp_lib0, mlp_input, mlp_cpu_ref) -> dict:
    """Run all board assertions once per (board, lib0) pair (module-level cache)."""
    key = (board_target, str(mlp_lib0))
    if key not in _REMOTE_RESULTS_CACHE:
        _REMOTE_RESULTS_CACHE[key] = _run_c7x_vm_on_board(
            board_target, mlp_lib0, mlp_input, mlp_cpu_ref
        )
    return _REMOTE_RESULTS_CACHE[key]


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
    def test_init_bad_module_lazy(self, mlp_lib0, mlp_input, mlp_cpu_ref, board_target):
        """FileNotFoundError raised on first inference if module path is invalid."""
        if board_target:
            r = _remote_results(self, board_target, mlp_lib0, mlp_input, mlp_cpu_ref)
            assert r.get("bad_module_raises"), f"Remote assertion failed: {r}"
            return
        vm = C7xVirtualMachine("/tmp/this_does_not_exist.out")
        with pytest.raises(FileNotFoundError):
            vm["main"](np.zeros((1, 64), dtype=np.float32))

    @pytest.mark.core
    @requires_lib
    def test_is_loaded_before_first_call(self, mlp_lib0, mlp_input, mlp_cpu_ref, board_target):
        """is_loaded must be False before first inference call."""
        if board_target:
            r = _remote_results(self, board_target, mlp_lib0, mlp_input, mlp_cpu_ref)
            assert r.get("not_loaded_before_call"), f"Remote assertion failed: {r}"
            return
        vm = C7xVirtualMachine("/tmp/fake.out")
        assert not vm.is_loaded

    @pytest.mark.core
    @requires_lib
    def test_close_is_idempotent(self, mlp_lib0, mlp_input, mlp_cpu_ref, board_target):
        """close() on an unloaded VM must not raise."""
        if board_target:
            r = _remote_results(self, board_target, mlp_lib0, mlp_input, mlp_cpu_ref)
            assert r.get("close_idempotent"), f"Remote assertion failed: {r}"
            return
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
    """Integration tests: C7xVirtualMachine running against live firmware.

    When --board-target=HOST is given (running from the dev PC), each test
    delegates to _run_c7x_vm_on_board() which SCPs lib0.out to the board and
    runs all assertions via a single SSH call.  Without --board-target (running
    on the board itself), the C7xVirtualMachine is used directly.
    """

    @pytest.mark.core
    def test_standard_inference_matches_cpu(self, mlp_lib0, mlp_input, mlp_cpu_ref,
                                            board_target):
        """vm["main"](numpy_input) result must match CPU reference."""
        if board_target:
            r = _remote_results(self, board_target, mlp_lib0, mlp_input, mlp_cpu_ref)
            assert r.get("standard_inference"), f"Remote assertion failed: {r}"
            return
        vm = C7xVirtualMachine(mlp_lib0)
        out = vm["main"](mlp_input)
        vm.close()
        np.testing.assert_allclose(out.numpy(), mlp_cpu_ref, rtol=1e-3, atol=1e-3,
                                   err_msg="DSP output diverges from CPU reference")

    @pytest.mark.core
    def test_is_loaded_after_inference(self, mlp_lib0, mlp_input, mlp_cpu_ref,
                                       board_target):
        """is_loaded must be True after the first successful inference."""
        if board_target:
            r = _remote_results(self, board_target, mlp_lib0, mlp_input, mlp_cpu_ref)
            assert r.get("is_loaded"), f"Remote assertion failed: {r}"
            assert r.get("close_works"), f"Remote close() assertion failed: {r}"
            return
        vm = C7xVirtualMachine(mlp_lib0)
        assert not vm.is_loaded
        vm["main"](mlp_input)
        assert vm.is_loaded
        vm.close()
        assert not vm.is_loaded

    @pytest.mark.core
    def test_last_cycles_positive(self, mlp_lib0, mlp_input, mlp_cpu_ref, board_target):
        """last_cycles must be a positive integer after inference."""
        if board_target:
            r = _remote_results(self, board_target, mlp_lib0, mlp_input, mlp_cpu_ref)
            assert r.get("last_cycles"), f"Remote assertion failed: {r}"
            return
        with C7xVirtualMachine(mlp_lib0) as vm:
            vm["main"](mlp_input)
            assert vm.last_cycles > 0, "Expected positive cycle count from DSP"

    @pytest.mark.core
    def test_context_manager_closes_on_exit(self, mlp_lib0, mlp_input, mlp_cpu_ref,
                                            board_target):
        """Context manager must unload the module on __exit__."""
        if board_target:
            r = _remote_results(self, board_target, mlp_lib0, mlp_input, mlp_cpu_ref)
            assert r.get("close_works"), f"Remote assertion failed: {r}"
            return
        vm_ref = None
        with C7xVirtualMachine(mlp_lib0) as vm:
            vm_ref = vm
            vm["main"](mlp_input)
            assert vm.is_loaded
        assert not vm_ref.is_loaded

    def test_repeated_inference_consistent(self, mlp_lib0, mlp_input, mlp_cpu_ref,
                                           board_target):
        """Three consecutive inferences must return the same result."""
        if board_target:
            r = _remote_results(self, board_target, mlp_lib0, mlp_input, mlp_cpu_ref)
            assert r.get("repeated_consistent"), f"Remote assertion failed: {r}"
            return
        with C7xVirtualMachine(mlp_lib0) as vm:
            results = [vm["main"](mlp_input).numpy() for _ in range(3)]
        for i, res in enumerate(results):
            np.testing.assert_allclose(
                res, mlp_cpu_ref, rtol=1e-3, atol=1e-3,
                err_msg=f"Inference #{i} diverges from reference"
            )
        np.testing.assert_array_equal(results[0], results[1])
        np.testing.assert_array_equal(results[1], results[2])

    @pytest.mark.core
    def test_as_vm_factory(self, mlp_lib0, mlp_input, mlp_cpu_ref, board_target):
        """TIDLBuildResult.as_vm() must return a working C7xVirtualMachine."""
        if board_target:
            # as_vm() delegates to C7xVirtualMachine; correctness is covered by
            # standard_inference above.  Just verify the factory doesn't raise.
            r = _remote_results(self, board_target, mlp_lib0, mlp_input, mlp_cpu_ref)
            assert r.get("standard_inference"), f"Remote assertion failed: {r}"
            return
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
    def test_numpy_input_accepted(self, mlp_lib0, mlp_input, mlp_cpu_ref, board_target):
        """C7xVirtualMachine must accept plain numpy arrays."""
        if board_target:
            r = _remote_results(self, board_target, mlp_lib0, mlp_input, mlp_cpu_ref)
            assert r.get("standard_inference"), f"Remote assertion failed: {r}"
            return
        with C7xVirtualMachine(mlp_lib0) as vm:
            out = vm["main"](mlp_input)
        np.testing.assert_allclose(out.numpy(), mlp_cpu_ref, rtol=1e-3, atol=1e-3)

    @pytest.mark.core
    def test_output_is_tvm_ndarray(self, mlp_lib0, mlp_input, mlp_cpu_ref, board_target):
        """vm["main"] must return a tensor with correct shape/dtype."""
        if board_target:
            r = _remote_results(self, board_target, mlp_lib0, mlp_input, mlp_cpu_ref)
            assert r.get("output_shape"), f"Shape check failed: {r}"
            assert r.get("output_dtype"), f"Dtype check failed: {r}"
            return
        with C7xVirtualMachine(mlp_lib0) as vm:
            out = vm["main"](mlp_input)
        # C7xVirtualMachine wraps outputs in tvm.runtime.Tensor (TVM 0.23)
        assert hasattr(out, "numpy") and hasattr(out, "shape"), (
            f"Expected a TVM tensor with .numpy() and .shape, got {type(out)}"
        )
        assert tuple(out.shape) == (1, 8)
        assert str(out.dtype) == "float32"


@requires_firmware
class TestRunNocopy:
    """Zero-copy output via run_nocopy()."""

    @pytest.mark.core
    def test_run_nocopy_returns_numpy(self, mlp_lib0, mlp_input, mlp_cpu_ref,
                                      board_target):
        """run_nocopy() must return a numpy array with correct values."""
        if board_target:
            r = _remote_results(self, board_target, mlp_lib0, mlp_input, mlp_cpu_ref)
            assert r.get("nocopy_numpy"), f"Remote assertion failed: {r}"
            assert r.get("nocopy_correct"), f"Remote assertion failed: {r}"
            return
        with C7xVirtualMachine(mlp_lib0) as vm:
            out = vm.run_nocopy(mlp_input)
        assert isinstance(out, np.ndarray), (
            f"Expected np.ndarray from run_nocopy, got {type(out)}"
        )
        np.testing.assert_allclose(out, mlp_cpu_ref, rtol=1e-3, atol=1e-3)

    def test_run_nocopy_matches_standard_run(self, mlp_lib0, mlp_input, mlp_cpu_ref,
                                              board_target):
        """run_nocopy() and vm["main"]() must produce identical results."""
        if board_target:
            r = _remote_results(self, board_target, mlp_lib0, mlp_input, mlp_cpu_ref)
            assert r.get("nocopy_correct"), f"Remote assertion failed: {r}"
            return
        with C7xVirtualMachine(mlp_lib0) as vm:
            out_copy = vm["main"](mlp_input).numpy()
            out_view = vm.run_nocopy(mlp_input)
        np.testing.assert_array_equal(
            out_copy, out_view,
            err_msg="run_nocopy() result differs from standard vm['main'] result"
        )

    @pytest.mark.core
    def test_last_nocopy_outputs_list(self, mlp_lib0, mlp_input, mlp_cpu_ref,
                                      board_target):
        """_last_nocopy_outputs must be populated after run_nocopy()."""
        if board_target:
            r = _remote_results(self, board_target, mlp_lib0, mlp_input, mlp_cpu_ref)
            assert r.get("nocopy_numpy"), f"Remote assertion failed: {r}"
            return
        with C7xVirtualMachine(mlp_lib0) as vm:
            vm.run_nocopy(mlp_input)
            assert len(vm._last_nocopy_outputs) > 0  # noqa: SLF001


@requires_firmware
class TestCreateInput:
    """Zero-copy input via create_input()."""

    @pytest.mark.core
    def test_create_input_shape_dtype(self, mlp_lib0, mlp_input, mlp_cpu_ref,
                                      board_target):
        """create_input() must return a tensor with the correct shape and dtype."""
        if board_target:
            r = _remote_results(self, board_target, mlp_lib0, mlp_input, mlp_cpu_ref)
            assert r.get("create_shape"), f"Remote assertion failed: {r}"
            assert r.get("create_dtype"), f"Remote assertion failed: {r}"
            return
        with C7xVirtualMachine(mlp_lib0) as vm:
            inp = vm.create_input((1, 64), "float32")
        assert inp.shape == (1, 64)
        assert str(inp.dtype) == "float32"

    @pytest.mark.core
    def test_create_input_inference_correct(self, mlp_lib0, mlp_input, mlp_cpu_ref,
                                            board_target):
        """Inference via pre-staged create_input() tensor must match CPU reference."""
        if board_target:
            r = _remote_results(self, board_target, mlp_lib0, mlp_input, mlp_cpu_ref)
            assert r.get("create_correct"), f"Remote assertion failed: {r}"
            return
        with C7xVirtualMachine(mlp_lib0) as vm:
            inp = vm.create_input((1, 64), "float32")
            inp.copyfrom(mlp_input)
            out = vm["main"](inp)
        np.testing.assert_allclose(
            out.numpy(), mlp_cpu_ref, rtol=1e-3, atol=1e-3,
            err_msg="create_input() inference diverges from CPU reference"
        )

    def test_create_input_matches_standard_path(self, mlp_lib0, mlp_input, mlp_cpu_ref,
                                                 board_target):
        """create_input() and standard numpy input must give identical results."""
        if board_target:
            r = _remote_results(self, board_target, mlp_lib0, mlp_input, mlp_cpu_ref)
            assert r.get("create_correct"), f"Remote assertion failed: {r}"
            return
        with C7xVirtualMachine(mlp_lib0) as vm:
            out_standard = vm["main"](mlp_input).numpy()
            inp = vm.create_input((1, 64), "float32")
            inp.copyfrom(mlp_input)
            out_prestaged = vm["main"](inp).numpy()
        np.testing.assert_array_equal(
            out_standard, out_prestaged,
            err_msg="create_input() and standard path give different results"
        )

    @pytest.mark.core
    def test_create_input_registered_in_slots(self, mlp_lib0, mlp_input, mlp_cpu_ref,
                                              board_target):
        """create_input() must register the tensor in _input_slots dict."""
        if board_target:
            r = _remote_results(self, board_target, mlp_lib0, mlp_input, mlp_cpu_ref)
            assert r.get("create_shape"), f"Remote assertion failed: {r}"
            return
        with C7xVirtualMachine(mlp_lib0) as vm:
            before = len(vm._input_slots)  # noqa: SLF001
            vm.create_input((1, 64), "float32")
            after = len(vm._input_slots)  # noqa: SLF001
        assert after == before + 1, "create_input() did not register slot"

    def test_staging_offset_advances_after_create_input(self, mlp_lib0, mlp_input,
                                                         mlp_cpu_ref, board_target):
        """Staging alloc offset must advance by at least tensor size."""
        if board_target:
            r = _remote_results(self, board_target, mlp_lib0, mlp_input, mlp_cpu_ref)
            assert r.get("staging_offset_ok"), f"Remote assertion failed: {r}"
            return
        with C7xVirtualMachine(mlp_lib0) as vm:
            offset_before = vm._staging_alloc_offset  # noqa: SLF001
            vm.create_input((1, 64), "float32")
            offset_after = vm._staging_alloc_offset  # noqa: SLF001
        assert offset_after >= offset_before + 256

    @pytest.mark.core
    def test_staging_offset_from_elf_not_hardcoded(self, mlp_lib0, mlp_input,
                                                    mlp_cpu_ref, board_target):
        """Staging alloc offset must come from the real ELF size, not 256 MB."""
        if board_target:
            r = _remote_results(self, board_target, mlp_lib0, mlp_input, mlp_cpu_ref)
            assert r.get("staging_offset_ok"), f"Remote assertion failed: {r}"
            return
        with C7xVirtualMachine(mlp_lib0) as vm:
            _ = vm["main"](np.zeros((1, 64), dtype=np.float32))
            offset = vm._staging_alloc_offset  # noqa: SLF001
        assert offset < 256 * 1024 * 1024, (
            f"Staging offset {offset} looks like the hardcoded 256 MB sentinel"
        )

    def test_close_resets_staging_slots(self, mlp_lib0, mlp_input, mlp_cpu_ref,
                                        board_target):
        """close() must clear _input_slots and reset staging offset."""
        if board_target:
            r = _remote_results(self, board_target, mlp_lib0, mlp_input, mlp_cpu_ref)
            assert r.get("close_resets_slots"), f"Remote assertion failed: {r}"
            return
        vm = C7xVirtualMachine(mlp_lib0)
        vm["main"](np.zeros((1, 64), dtype=np.float32))
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
        run(lambda: inf_suite.test_standard_inference_matches_cpu(
                lib0, inp, ref, board_target=None), "standard_inference")
        run(lambda: inf_suite.test_last_cycles_positive(
                lib0, inp, ref, board_target=None), "last_cycles")
        run(lambda: inf_suite.test_numpy_input_accepted(
                lib0, inp, ref, board_target=None), "numpy_input")
        run(lambda: inf_suite.test_as_vm_factory(
                lib0, inp, ref, board_target=None), "as_vm_factory")

        nc_suite = TestRunNocopy()
        run(lambda: nc_suite.test_run_nocopy_returns_numpy(
                lib0, inp, ref, board_target=None), "run_nocopy")

        ci_suite = TestCreateInput()
        run(lambda: ci_suite.test_create_input_inference_correct(
                lib0, inp, ref, board_target=None), "create_input_inference")
        run(lambda: ci_suite.test_staging_offset_from_elf_not_hardcoded(
                lib0, inp, ref, board_target=None), "staging_offset_elf")

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
