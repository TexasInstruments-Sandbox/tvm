"""
Pytest configuration for quantized model DSP tests.

Provides fixtures and CLI options for running quantized TorchVision models
on the C7x DSP. The --use-cpp-api flag defaults to True (always enabled).

Usage:
    pytest quantized/ -v --dsp-mode=c7x_host
    pytest quantized/ -v --dsp-mode=c7x_dload
    pytest quantized/ -v --dsp-mode=c7x_dload --mmalib
"""

import os
import sys
from pathlib import Path

import pytest

_THIS_DIR = Path(__file__).parent
_DSP_CPP_DIR = _THIS_DIR.parent / "dsp-cpp"
sys.path.insert(0, str(_THIS_DIR))
sys.path.insert(0, str(_DSP_CPP_DIR))


def pytest_configure(config):
    """Register custom markers and suppress noisy third-party warnings."""
    config.addinivalue_line(
        "markers",
        "c7x_only: test only valid for c7x targets",
    )
    config.addinivalue_line(
        "markers",
        "core: post-merge gate — all core ops and classification models",
    )
    config.addinivalue_line(
        "filterwarnings",
        "ignore:.*erase_node.*:UserWarning",
    )


def pytest_addoption(parser):
    """Add DSP-specific command-line options."""
    parser.addoption(
        "--dsp-mode",
        action="store",
        default=None,
        choices=["c7x_host", "c7x_dload"],
        help="DSP execution mode: c7x_host (emulation) or c7x_dload (hardware)",
    )
    parser.addoption(
        "--dsp-timeout",
        action="store",
        default=120000,
        type=int,
        help="Execution timeout in milliseconds (default: 120000)",
    )
    parser.addoption(
        "--profile",
        action="store_true",
        default=False,
        help="Enable profiling: per-layer cycle counters + repeat=2",
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
        default=True,
        help="Enable direct VM builtin calls (default: True)",
    )
    parser.addoption(
        "--mmalib",
        action="store_true",
        default=False,
        help="Enable MMALIB acceleration for eligible conv2d/matmul ops",
    )


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
    """Fixture providing execution timeout."""
    return request.config.getoption("--dsp-timeout")


@pytest.fixture
def profile(request):
    """Fixture: enable profiling (layer counters + repeat=2)."""
    return request.config.getoption("--profile") or request.config.getoption("--profile-layers")


@pytest.fixture
def profile_layers(request):
    """Fixture: compile with per-layer cycle counters."""
    return request.config.getoption("--profile") or request.config.getoption("--profile-layers")


@pytest.fixture
def use_cpp_api(request):
    """Fixture providing direct VM calls flag (default True)."""
    return request.config.getoption("--use-cpp-api")


@pytest.fixture
def mmalib(request):
    """Fixture providing MMALIB acceleration flag."""
    return request.config.getoption("--mmalib")


# ---------------------------------------------------------------------------
# Cycle performance tracking
# ---------------------------------------------------------------------------


@pytest.fixture
def record_cycles():
    """Record cycle count for a model, written to cycles.csv immediately.

    Writes per-call (not batched at session end) so the count survives even
    if this test runs in a forked subprocess (pytest --forked / --isolate),
    where in-memory state can't cross the fork boundary back to the parent.
    Safe without locking: --forked runs tests one at a time, never in
    parallel (that would require -n/xdist, which combining with --forked
    would violate the single-DSP-session hardware rule anyway).
    """

    def _record(name: str, cycles: int):
        if cycles <= 0:
            return
        out_dir = Path(os.environ.get("WORKSPACE", ".")) / "results"
        out_dir.mkdir(parents=True, exist_ok=True)
        out_file = out_dir / "cycles.csv"
        existing: dict = {}
        if out_file.exists():
            lines = out_file.read_text().splitlines()
            if len(lines) >= 2:
                headers = lines[0].split(",")
                values = lines[1].split(",")
                existing = dict(zip(headers, values))
        existing[name] = str(cycles)
        sorted_keys = sorted(existing.keys())
        with open(out_file, "w") as f:
            f.write(",".join(sorted_keys) + "\n")
            f.write(",".join(existing[k] for k in sorted_keys) + "\n")

    return _record
