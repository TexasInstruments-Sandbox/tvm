"""Pytest configuration for unit tests."""

import os
import sys
from pathlib import Path

import pytest

_THIS_DIR = Path(__file__).parent
_DSP_CPP_DIR = _THIS_DIR.parent / "dsp-cpp"
sys.path.insert(0, str(_DSP_CPP_DIR))


def pytest_addoption(parser):
    parser.addoption(
        "--dsp-mode",
        action="store",
        default=None,
        choices=["c7x_host", "c7x_dload"],
        help="DSP execution mode",
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
    return request.config.getoption("--dsp-mode")


_cycle_data: dict = {}


@pytest.fixture(scope="session", autouse=True)
def _cycle_writer():
    """Write collected cycle data to results/cycles.csv at session end."""
    yield
    non_zero = {k: v for k, v in _cycle_data.items() if v > 0}
    if not non_zero:
        return
    out_dir = Path(os.environ.get("WORKSPACE", str(_THIS_DIR.parent))) / "results"
    out_dir.mkdir(parents=True, exist_ok=True)
    out_file = out_dir / "cycles.csv"
    existing: dict = {}
    if out_file.exists():
        lines = out_file.read_text().splitlines()
        if len(lines) >= 2:
            headers = lines[0].split(",")
            values = lines[1].split(",")
            for h, v in zip(headers, values):
                existing[h] = v
    existing.update({k: str(v) for k, v in non_zero.items()})
    sorted_keys = sorted(existing.keys())
    with open(out_file, "w") as f:
        f.write(",".join(sorted_keys) + "\n")
        f.write(",".join(existing[k] for k in sorted_keys) + "\n")


@pytest.fixture
def record_cycles():
    """Record cycle count for a test. Written to cycles.csv at session end."""

    def _record(name: str, cycles: int):
        _cycle_data[name] = cycles

    return _record
