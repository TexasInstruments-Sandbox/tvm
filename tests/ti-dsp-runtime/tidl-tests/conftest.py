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
    from dsp_utils import set_current_test_name  # pyright: ignore[reportMissingImports]

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
    from dsp_utils import get_tidl_soc_dir

    soc = get_tidl_soc_dir()
    return os.path.isfile(os.path.join(c7x, f"ti_dl/lib/{soc}/PC/algo/release/libtidl_algo.a"))


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
            from dsp_utils import (  # pyright: ignore[reportMissingImports]
                INPUT_BIN_FILE,
                run_dsp_host,
                write_tensors_to_file,
            )

            write_tensors_to_file([input_data], str(result.build_dir / INPUT_BIN_FILE))
            output = run_dsp_host(result.module_path)
        else:
            from dsp_utils import run_dsp_dload  # pyright: ignore[reportMissingImports]

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
        "quick: mark test as quick (no hardware, fast execution). Partitioning "
        "needs tidl_model_import_relax.so, so this tier is no longer .so-free",
    )
    config.addinivalue_line(
        "markers",
        "core: post-merge gate — no hardware, plus the c7x_host stub pipeline; "
        "excludes tests requiring AM67A. Needs tidl_model_import_relax.so",
    )
    config.addinivalue_line(
        "markers",
        "c7x_only: tests requiring TI C7x toolchain or hardware",
    )
    config.addinivalue_line(
        "markers",
        "needs_tidl_so: test partitions through the eligibility gate and is "
        "skipped when tidl_model_import_relax.so is absent",
    )


# ---------------------------------------------------------------------------
# Offload eligibility is decided by tidl_model_import_relax.so, so every
# partition needs it. Modules that gate themselves on the .so (a module-level
# skipif or an autouse skip fixture) are not listed below; these are the ones
# that partition without such a gate, so collection has to skip them.
#
# A test that only wants to exercise pattern *structure* can pass
# ``allow_fn=lambda call: 1`` in the compiler config and drop off this list.
# ---------------------------------------------------------------------------

_C7X_MMA_TIDL_PATH = os.environ.setdefault(
    "C7X_MMA_TIDL_PATH", os.path.expanduser("~/ml/c7x-mma-tidl")
)
RELAX_SO_PATH = os.path.join(
    _C7X_MMA_TIDL_PATH, "ti_dl/utils/tidlModelImport/out/tidl_model_import_relax.so"
)

# Keep sorted. See tidl-tests/README.md ("Library-gated tests").
_NEEDS_SO_MODULES = (
    "test_tidl_cast_attrs.py",
    "test_tidl_codegen.py",
    "test_tidl_constraint_coverage.py",
    "test_tidl_e2e.py",
    "test_tidl_layer_offload.py",
    "test_tidl_multioutput_metadata.py",
    "test_tidl_negative_axis.py",
    "test_tidl_partition.py",
    "test_tidl_pattern_checks.py",
    "test_tidl_pseudo3d_conv.py",
    "test_tidl_split_single_index.py",
    "test_tidl_var_input_dtype.py",
)


def pytest_collection_modifyitems(config, items):  # noqa: ARG001
    """Skip (not fail) tests that partition through the .so when it isn't built."""
    if os.path.isfile(RELAX_SO_PATH):
        return
    skip = pytest.mark.skip(reason="tidl_model_import_relax.so not built (partition needs it)")
    for item in items:
        # Match on the file name: nodeid is relative to rootdir, which differs
        # between `pytest tidl-tests/...` and `pytest --rootdir=... tests/...`.
        if (
            item.get_closest_marker("needs_tidl_so")
            or os.path.basename(str(item.path)) in _NEEDS_SO_MODULES
        ):
            item.add_marker(skip)
