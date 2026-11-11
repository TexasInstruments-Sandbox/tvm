"""
Top-level pytest configuration for tests/ti-dsp-runtime/.

Registers --board here (rather than in each leaf conftest.py) so it's
picked up automatically by every subdir-scoped `pytest --rootdir=. <dir>/`
invocation, and so a second registration doesn't collide with any leaf
conftest's own pytest_addoption.
"""

import sys
from pathlib import Path

_THIS_DIR = Path(__file__).parent
sys.path.insert(0, str(_THIS_DIR / "dsp-cpp"))


def pytest_addoption(parser):
    parser.addoption(
        "--board",
        action="store",
        default=None,
        choices=["j722s-evm", "beagley-ai", "am62d"],
        help="Target board for c7x_dload (hardware) runs. Required whenever "
        "--dsp-mode=c7x_dload is used; ignored for c7x_host emulation.",
    )
    parser.addoption(
        "--tidl-tools-dir",
        action="store",
        default=None,
        help="TIDL tools directory for TIDL tests (passed as config['tidl_tools_dir']). "
        "Default: the one in the tvm package, tvm/data/ti_dsp/tidl.",
    )


def pytest_configure(config):
    from dsp_utils import set_current_board, set_tidl_tools_dir

    set_current_board(config.getoption("--board"))
    set_tidl_tools_dir(config.getoption("--tidl-tools-dir"))
