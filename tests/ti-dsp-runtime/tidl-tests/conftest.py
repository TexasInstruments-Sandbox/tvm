"""Pytest configuration for TIDL tests.

This file is intentionally minimal.  TIDL tests do not use the
``--dsp-mode`` fixture from ``dsp-tests/conftest.py`` because they
manage their own build and execution flows (stub bridge via c7x_host,
or full hardware via run_dsp_dload).

The file must exist so pytest treats ``tidl-tests/`` as a separate
test root and does not inherit the ``dsp-tests/`` conftest, which
would require ``--dsp-mode`` for every invocation.
"""

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
