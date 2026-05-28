"""Pytest configuration for TIDL tests."""

import os
import shutil
import sys
from pathlib import Path

import pytest

_THIS_DIR = Path(__file__).parent
_DSP_CPP_DIR = _THIS_DIR.parent / "dsp-cpp"
sys.path.insert(0, str(_DSP_CPP_DIR))


@pytest.fixture(autouse=True)
def _set_dsp_test_name(request):
    """Set the current test name in dsp_utils for workspace naming."""
    from dsp_utils import set_current_test_name

    set_current_test_name(request.node.name)
    yield
    set_current_test_name(None)


def pytest_addoption(parser):
    parser.addoption(
        "--dsp-mode",
        choices=["c7x_host", "c7x_dload"],
        default=None,
        help="DSP execution mode: c7x_host (host emulation) or c7x_dload (AM67A hardware)",
    )
    parser.addoption(
        "--dsp-timeout",
        default=60000,
        type=int,
        help="Timeout for DSP execution in milliseconds (default: 60000)",
    )
    parser.addoption(
        "--profile-layers",
        action="store_true",
        default=False,
        help="Compile with per-layer cycle counters",
    )
    parser.addoption(
        "--use-cpp-api",
        action="store_true",
        default=False,
        help="Enable direct VM builtin calls (bypass FFI dispatch)",
    )


@pytest.fixture
def dsp_mode(request):
    """DSP execution mode selected via --dsp-mode CLI option."""
    return request.config.getoption("--dsp-mode")


@pytest.fixture
def dsp_timeout(request):
    """Timeout for DSP execution in milliseconds."""
    return request.config.getoption("--dsp-timeout")


@pytest.fixture
def profile_layers(request):
    """Enable per-layer cycle counters."""
    return request.config.getoption("--profile-layers")


@pytest.fixture
def use_cpp_api(request):
    """Enable direct VM builtin calls."""
    return request.config.getoption("--use-cpp-api")


@pytest.fixture
def record_cycles():
    """No-op cycle recorder (cycle CSV is a dsp-tests feature)."""

    def _record(name: str, cycles: int):
        pass

    return _record


def has_c7x_host_env():
    """True if the C7x host emulation build environment is available."""
    return os.environ.get("TI_CGT_C7000_PATH") is not None


def has_tidl_pc_libs():
    """True if PC (x86-64) TIDL algo libs are present."""
    c7x = os.environ.get("C7X_MMA_TIDL_PATH", os.path.expanduser("~/ml/c7x-mma-tidl"))
    return os.path.isfile(os.path.join(c7x, "ti_dl/lib/J722S/PC/algo/release/libtidl_algo.a"))


def tidl_build_and_run(compiler, mod, params, input_data, tmp_path, dsp_mode):
    """Build a TIDL model and run inference, dispatching on dsp_mode.

    Calls ``compiler.build(exec_mode=dsp_mode)`` then runs the result
    via ``run_dsp_host`` (c7x_host) or ``run_dsp_dload`` (c7x_dload).

    Parameters
    ----------
    compiler : TIDLOffloadCompiler
        Pre-configured compiler (with file-level SO path / tools path).
    mod : IRModule
        Module to compile (raw, before preparation).
    params : dict
        Named parameters for ``compiler.build``.
    input_data : np.ndarray
        Single input tensor for inference.
    tmp_path : Path
        Pytest tmp_path fixture value; build_dir is placed here.
    dsp_mode : str or None
        "c7x_host" or "c7x_dload".  None or other values cause a skip.

    Returns
    -------
    output : np.ndarray
        Inference output tensor.
    n_artifacts : int
        Number of TIDL subgraphs compiled.
    """
    if dsp_mode not in ("c7x_host", "c7x_dload"):
        pytest.skip(f"requires --dsp-mode=c7x_host or c7x_dload, got {dsp_mode!r}")
    if dsp_mode == "c7x_host" and not has_c7x_host_env():
        pytest.skip("TI_CGT_C7000_PATH not set (required for c7x_host)")
    if dsp_mode == "c7x_host" and not has_tidl_pc_libs():
        pytest.skip("PC TIDL algo libs not found (required for c7x_host)")

    result = compiler.build(
        mod,
        params=params,
        exec_mode=dsp_mode,
        build_dir=str(tmp_path / "build"),
    )

    try:
        if dsp_mode == "c7x_host":
            from dsp_utils import INPUT_BIN_FILE, run_dsp_host, write_tensors_to_file

            write_tensors_to_file([input_data], str(result.build_dir / INPUT_BIN_FILE))
            output = run_dsp_host(result.module_path)
        else:
            from dsp_utils import run_dsp_dload

            output, _stdout, _cycles = run_dsp_dload(
                result.module_path,
                result.weights_path,
                [input_data],
                embedded_weights=True,
            )

        return output, len(result.artifacts)

    finally:
        if not os.environ.get("DSP_KEEP_TEMP"):
            shutil.rmtree(str(result.gen_dir), ignore_errors=True)


def pytest_configure(config):
    """Register custom markers."""
    config.addinivalue_line(
        "markers",
        "quick: mark test as quick (no hardware, no .so, fast execution)",
    )
    config.addinivalue_line(
        "markers",
        "core: post-merge gate — dependency-free tests plus c7x_host stub "
        "pipeline; excludes tests requiring tidl_model_import_relax.so or AM67A",
    )
    config.addinivalue_line(
        "markers",
        "c7x_only: tests requiring TI C7x toolchain or hardware",
    )
