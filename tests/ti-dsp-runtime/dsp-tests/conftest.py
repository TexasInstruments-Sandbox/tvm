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

import os
import subprocess
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
    config.addinivalue_line(
        "markers",
        "core: post-merge gate — all core ops, classification, accuracy, "
        "and small detection; excludes large YOLO/segmentation and benchmarks",
    )
    config.addinivalue_line(
        "markers",
        "c66x_only: test only valid for c66x targets "
        "(e.g. c66x pragma generation, c66x-specific codegen)",
    )
    config.addinivalue_line(
        "markers",
        "c7x_only: test only valid for c7x targets "
        "(model too large for C66x, or feature is c7x-specific)",
    )
    config.addinivalue_line(
        "markers",
        "requires_c7x_vm_lib: test requires libc7x_arm_runtime.so "
        "(installed on AM67A via ./build.sh deploy, or reachable via --board-target)",
    )
    config.addinivalue_line(
        "markers",
        "requires_c7x_firmware: test requires c7x_compute firmware running "
        "(on AM67A locally, or reachable via --board-target)",
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
    parser.addoption(
        "--mmalib",
        action="store_true",
        default=False,
        help="Enable MMALIB acceleration for eligible conv2d/matmul ops",
    )
    parser.addoption(
        "--board-target",
        action="store",
        default=None,
        metavar="HOST",
        help="AM67A board hostname for remote c7x_vm tests via SSH "
        "(e.g. 'am67a').  When set with --dsp-mode=c7x_dload, "
        "test_c7x_vm_dsp integration tests deploy lib0.out to the "
        "board and run C7xVirtualMachine assertions there via SSH.",
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
    return request.config.getoption("--profile") or request.config.getoption("--profile-layers")


@pytest.fixture
def profile_layers(request):
    """Fixture: compile with per-layer cycle counters."""
    return request.config.getoption("--profile") or request.config.getoption("--profile-layers")


@pytest.fixture
def use_cpp_api(request):
    """Fixture providing direct VM calls flag."""
    return request.config.getoption("--use-cpp-api")


@pytest.fixture
def mmalib(request):
    """Fixture providing MMALIB acceleration flag."""
    return request.config.getoption("--mmalib")


@pytest.fixture
def board_target(request):
    """Hostname of the AM67A board for remote c7x_vm tests.

    None when running locally (on the board itself or without --board-target).
    Set to e.g. 'am67a' to run C7xVirtualMachine integration tests by SSHing
    into the board from the dev PC.
    """
    return request.config.getoption("--board-target", default=None)


@pytest.fixture
def dsp_config(
    dsp_mode,
    dsp_timeout,
    dsp_verbose,
    save_artifacts,
    profile_layers,
    profile,
    use_cpp_api,
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


def _board_ssh_reachable(board: str) -> bool:
    """True if the board is reachable via SSH and has the required files."""
    try:
        rc = subprocess.run(
            [
                "ssh",
                "-o",
                "ConnectTimeout=5",
                "-o",
                "BatchMode=yes",
                f"root@{board}",
                "test -f /usr/local/lib/libc7x_arm_runtime.so "
                "&& test -f /usr/local/bin/c7x_compute",
            ],
            capture_output=True,
            timeout=10,
        ).returncode
        return rc == 0
    except Exception:  # noqa: BLE001
        return False


def pytest_collection_modifyitems(config, items):
    """Manage mode-specific skip conditions."""
    dsp_mode = config.getoption("--dsp-mode")
    board_target = config.getoption("--board-target", default=None)

    if dsp_mode == "c66x":
        skip_host_only = pytest.mark.skip(
            reason="Test marked as dsp_host_only (too large for C66x hardware)"
        )
        for item in items:
            if "dsp_host_only" in item.keywords:
                item.add_marker(skip_host_only)

    # Apply local skip conditions for c7x_vm custom marks.
    # requires_c7x_vm_lib: always evaluated locally (tests need local .so).
    # requires_c7x_firmware: only evaluated when no board_target; with
    # board_target the marks are removed so tests run remotely via SSH.
    if True:
        # Check for local .so and firmware without importing the test module
        # (importing test_c7x_vm_dsp at collection time would partially init TVM,
        # breaking tvm.nd in subsequent fixture setups).
        import ctypes as _ctypes
        import glob as _glob

        try:
            _ctypes.CDLL("libc7x_arm_runtime.so")
            lib_ok = True
        except OSError:
            lib_ok = False
        fw_ok = lib_ok and any(
            "7e000000.dsp" in str(Path(p).resolve())
            for p in _glob.glob("/sys/class/rpmsg/rpmsg_ctrl*/device")
        )

        skip_no_lib = pytest.mark.skip(
            reason="libc7x_arm_runtime.so not found — "
            "install on AM67A board or pass --board-target=HOST"
        )
        skip_no_fw = pytest.mark.skip(
            reason="c7x_compute firmware not reachable — "
            "run on AM67A board or pass --board-target=HOST"
        )
        for item in items:
            # requires_c7x_vm_lib: skip locally when no board_target (the .so
            # is aarch64-only and won't load on x86).  When board_target is
            # set, the "board reachable" branch below removes the mark so
            # tests run via SSH instead.
            if "requires_c7x_vm_lib" in item.keywords and not lib_ok and not board_target:
                item.add_marker(skip_no_lib)
            # requires_c7x_firmware: skip locally only when no board_target;
            # with board_target the marks are removed and tests run via SSH.
            if "requires_c7x_firmware" in item.keywords and not fw_ok and not board_target:
                item.add_marker(skip_no_fw)

    # When --board-target is given with c7x_dload, un-skip c7x_vm integration
    # tests if the board is reachable via SSH.  The tests will run remotely.
    if board_target and dsp_mode == "c7x_dload":
        if _board_ssh_reachable(board_target):
            for item in items:
                if "test_c7x_vm_dsp" in str(item.fspath):
                    # Remove both requires_c7x_firmware and requires_c7x_vm_lib:
                    # all c7x_vm integration and API-contract tests run via the
                    # board_target SSH path (the board has the .so installed).
                    item.own_markers = [
                        m
                        for m in item.own_markers
                        if m.name not in ("requires_c7x_firmware", "requires_c7x_vm_lib")
                    ]
        else:
            # Board specified but unreachable — add a clear failure marker
            skip_unreachable = pytest.mark.skip(
                reason=f"--board-target={board_target} is not reachable via SSH "
                f"or missing /usr/local/lib/libc7x_arm_runtime.so"
            )
            for item in items:
                if "test_c7x_vm_dsp" in str(item.fspath):
                    if any(
                        m.name in ("requires_c7x_vm_lib", "requires_c7x_firmware")
                        for m in item.own_markers
                    ):
                        item.add_marker(skip_unreachable)


# ---------------------------------------------------------------------------
# Cycle performance tracking
# ---------------------------------------------------------------------------

_cycle_data: dict = {}


@pytest.fixture(scope="session", autouse=True)
def _cycle_writer(request):
    """Write collected cycle data to cycles.csv at session end.

    Uses read-then-write to merge results from multiple pytest sessions
    within one Jenkins build (e.g. c7x_dload Quick + c7x_dload Full).
    The Init stage clears results/ at the start of each build.
    Only writes entries with non-zero cycles (skips host emulation runs).
    """
    yield
    non_zero = {k: v for k, v in _cycle_data.items() if v > 0}
    if not non_zero:
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
            for h, v in zip(headers, values):
                existing[h] = v
    existing.update({k: str(v) for k, v in non_zero.items()})
    sorted_keys = sorted(existing.keys())
    with open(out_file, "w") as f:
        f.write(",".join(sorted_keys) + "\n")
        f.write(",".join(existing[k] for k in sorted_keys) + "\n")


@pytest.fixture
def record_cycles():
    """Record cycle count for a model. Written to cycles.csv at session end."""

    def _record(name: str, cycles: int):
        _cycle_data[name] = cycles

    return _record
