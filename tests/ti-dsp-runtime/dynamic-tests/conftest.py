"""
Pytest configuration for dynamic shape / control flow tests.

These tests validate Relax IR features (If expressions, dynamic shapes,
tail-recursive loops) with the c_static backend on C7x targets.

Usage:
    # C7x host emulation (x86, fast)
    pytest tests/ti-dsp-runtime/dynamic-tests/ -m quick --dsp-mode=c7x_host -v

    # C7x hardware via DLOAD
    pytest tests/ti-dsp-runtime/dynamic-tests/ -m quick --dsp-mode=c7x_dload -v

Run from the repo root with PYTHONPATH set:
    export PYTHONPATH=$TVM_HOME/python:$PYTHONPATH
    export TI_CGT_C7000_PATH=~/ti/ccs2041/ccs/tools/compiler/ti-cgt-c7000_5.0.1.LTS
"""

import sys
from pathlib import Path

import pytest

# Add dsp-cpp to path so dsp_utils is importable
_THIS_DIR = Path(__file__).parent
_DSP_CPP_DIR = _THIS_DIR.parent / "dsp-cpp"
sys.path.insert(0, str(_DSP_CPP_DIR))


def pytest_configure(config):
    config.addinivalue_line(
        "markers",
        "quick: mark test as quick (small model, fast compile and run)",
    )
    config.addinivalue_line(
        "markers",
        "c7x_only: test only valid for c7x targets (feature is c7x-specific)",
    )


def pytest_addoption(parser):
    """Add DSP-specific command-line options (guarded for shared-session use)."""
    try:
        parser.addoption(
            "--dsp-mode",
            action="store",
            default=None,
            choices=["c66x_host", "c66x", "c7x_host", "c7x_dload"],
            help="DSP execution mode: c7x_host (C7x emulation) or c7x_dload (C7x DLOAD)",
        )
    except ValueError:
        pass  # Already registered by a parent conftest (e.g. dsp-tests/conftest.py)

    try:
        parser.addoption(
            "--dsp-timeout",
            action="store",
            default=60000,
            type=int,
            help="Timeout for DSP execution in milliseconds (default: 60000)",
        )
    except ValueError:
        pass

    try:
        parser.addoption(
            "--dsp-verbose",
            action="store_true",
            default=False,
            help="Enable verbose DSP logging",
        )
    except ValueError:
        pass

    try:
        parser.addoption(
            "--use-cpp-api",
            action="store_true",
            default=False,
            help="Enable direct VM builtin calls (bypass FFI dispatch)",
        )
    except ValueError:
        pass


@pytest.fixture(autouse=True)
def _set_dsp_test_name(request):
    """Set the current test name in dsp_utils for workspace naming."""
    from dsp_utils import set_current_test_name

    set_current_test_name(request.node.name)
    yield
    set_current_test_name(None)


@pytest.fixture
def dsp_mode(request):
    """Fixture providing the DSP execution mode."""
    return request.config.getoption("--dsp-mode")


@pytest.fixture
def dsp_timeout(request):
    """Fixture providing the DSP execution timeout in milliseconds."""
    return request.config.getoption("--dsp-timeout")


@pytest.fixture
def dsp_verbose(request):
    """Fixture providing verbose logging flag."""
    return request.config.getoption("--dsp-verbose")


@pytest.fixture
def use_cpp_api(request):
    """Fixture providing direct VM calls flag."""
    return request.config.getoption("--use-cpp-api")
