"""
Pytest configuration for DSP tests.

This module provides fixtures and configuration for running TVM models on DSP.
Tests can be run with different execution modes controlled via --dsp-mode.

Usage:
    # Run with C66x host emulation
    pytest test_conv2d_dsp.py --dsp-mode=c66x_host

    # Run on C66x hardware
    pytest test_conv2d_dsp.py --dsp-mode=c66x

    # Run with C7x host emulation
    pytest test_conv2d_dsp.py --dsp-mode=c7x_host

    # Run via C7x DLOAD pipeline (c7x_compute)
    pytest test_conv2d_dsp.py --dsp-mode=c7x_dload

Markers:
    @pytest.mark.dsp_host_only - Test can only run on host emulation (too large for C66x)
"""

import sys
from pathlib import Path

import pytest


def pytest_configure(config):
    """Register custom markers and suppress noisy third-party warnings."""
    config.addinivalue_line(
        "markers",
        "dsp_host_only: mark test as host-only (too large for C66x hardware)",
    )
    config.addinivalue_line(
        "markers",
        "quick: mark test as quick (small model, fast compile and run)",
    )
    # Suppress torch.ao.quantization deprecation warnings.
    # torch 2.10 deprecates these in favor of torchao, but torchao 0.16
    # doesn't ship XNNPACKQuantizer yet.  Suppress until migration.
    config.addinivalue_line(
        "filterwarnings",
        "ignore:.*torch.ao.quantization is deprecated.*:DeprecationWarning",
    )
    config.addinivalue_line(
        "filterwarnings",
        "ignore:.*XNNPACKQuantizer is deprecated.*:DeprecationWarning",
    )
    config.addinivalue_line(
        "filterwarnings",
        "ignore:.*erase_node.*:UserWarning",
    )


# Add dsp-cpp directory to path for dsp_utils
_THIS_DIR = Path(__file__).parent
_DSP_CPP_DIR = _THIS_DIR.parent / "dsp-cpp"
sys.path.insert(0, str(_DSP_CPP_DIR))


def pytest_addoption(parser):
    """Add DSP-specific command-line options."""
    parser.addoption(
        "--dsp-mode",
        action="store",
        default=None,
        choices=["c66x_host", "c66x", "c7x_host", "c7x_dload"],
        help="DSP execution mode: c66x_host (C66x emulation), c66x (C66x hardware), "
        "c7x_host (C7x emulation), or c7x_dload (C7x DLOAD pipeline)",
    )
    parser.addoption(
        "--dsp-timeout",
        action="store",
        default=60000,
        type=int,
        help="Timeout for C66x execution in milliseconds (default: 60000)",
    )
    parser.addoption(
        "--dsp-verbose",
        action="store_true",
        default=False,
        help="Enable verbose DSP logging",
    )
    parser.addoption(
        "--save-artifacts",
        action="store",
        default=None,
        metavar="DIR",
        help="Copy build artifacts (lib0.c, weights.bin, devc.c) to specified directory",
    )
    parser.addoption(
        "--profile",
        action="store_true",
        default=False,
        help="Enable profiling: compile with per-layer cycle counters "
        "and run with repeat=2 for init/steady-state separation "
        "(c7x_dload only)",
    )
    parser.addoption(
        "--profile-layers",
        action="store_true",
        default=False,
        help="(Deprecated, use --profile) Alias for --profile",
    )
    parser.addoption(
        "--use-cpp-api",
        action="store_true",
        default=False,
        help="Enable direct VM builtin calls (bypass FFI dispatch)",
    )


@pytest.fixture
def dsp_mode(request):
    """Fixture providing the DSP execution mode."""
    return request.config.getoption("--dsp-mode")


@pytest.fixture
def dsp_timeout(request):
    """Fixture providing the C66x timeout value."""
    return request.config.getoption("--dsp-timeout")


@pytest.fixture
def dsp_verbose(request):
    """Fixture providing verbose logging flag."""
    return request.config.getoption("--dsp-verbose")


@pytest.fixture
def save_artifacts(request):
    """Fixture providing the directory to save build artifacts."""
    return request.config.getoption("--save-artifacts")


@pytest.fixture
def profile(request):
    """Fixture: enable profiling (layer counters + repeat=2)."""
    return (
        request.config.getoption("--profile")
        or request.config.getoption("--profile-layers")
    )


@pytest.fixture
def profile_layers(request):
    """Fixture: compile with per-layer cycle counters."""
    return (
        request.config.getoption("--profile")
        or request.config.getoption("--profile-layers")
    )


@pytest.fixture
def use_cpp_api(request):
    """Fixture providing direct VM calls flag."""
    return request.config.getoption("--use-cpp-api")


@pytest.fixture
def dsp_config(
    dsp_mode, dsp_timeout, dsp_verbose, save_artifacts,
    profile_layers, profile, use_cpp_api,
):
    """Combined fixture providing all DSP configuration."""
    return {
        "mode": dsp_mode,
        "timeout_ms": dsp_timeout,
        "verbose": dsp_verbose,
        "save_artifacts": save_artifacts,
        "profile_layers": profile_layers,
        "profile": profile,
        "use_cpp_api": use_cpp_api,
    }


def pytest_collection_modifyitems(config, items):
    """Skip dsp_host_only tests when running on C66x hardware only."""
    dsp_mode = config.getoption("--dsp-mode")

    if dsp_mode == "c66x":
        skip_host_only = pytest.mark.skip(
            reason="Test marked as dsp_host_only (too large for C66x hardware)"
        )
        for item in items:
            if "dsp_host_only" in item.keywords:
                item.add_marker(skip_host_only)
