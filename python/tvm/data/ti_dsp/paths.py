"""Path resolution for bundled TI DSP artifacts.

When tvm-ti-c7x-dsp is installed from a wheel, DSP binaries live
under tvm/data/ti_dsp/ in the installed package.  Each find_*()
function checks the bundled location first and returns None if the
artifact is not present (the caller is expected to fall back to
environment variables or other discovery mechanisms).
"""

import os
import platform
from pathlib import Path
from typing import Optional

_DATA_DIR = Path(__file__).resolve().parent
_IS_AARCH64 = platform.machine() in ("aarch64", "arm64")


def get_ti_dsp_path() -> Path:
    """Return the root of the bundled TI DSP data directory."""
    return _DATA_DIR


def find_c7x_arm_runtime_so() -> Optional[str]:
    """Find libc7x_arm_runtime.so in the bundled package.

    Returns None on x86 hosts because the bundled .so is an aarch64
    binary that cannot be loaded via ctypes on x86.  On aarch64
    (i.e. running directly on the AM67A board) the bundled path is
    returned.
    """
    if not _IS_AARCH64:
        return None
    bundled = _DATA_DIR / "firmware" / "libc7x_arm_runtime.so"
    if bundled.exists():
        return str(bundled)
    return None


def find_c7x_compute_binary() -> Optional[str]:
    """Find the c7x_compute ARM CLI binary in the bundled package.

    Returns None on x86 hosts because the bundled binary is an aarch64
    ELF executable that cannot be exec'd there (same gating as
    find_c7x_arm_runtime_so()).
    """
    if not _IS_AARCH64:
        return None
    bundled = _DATA_DIR / "firmware" / "c7x_compute"
    if bundled.exists():
        return str(bundled)
    return None


def find_c7x_include_dir() -> Optional[Path]:
    """Find the include directory for the C7x C++ API.

    Contains c7x_runtime.h and its DLPack dependency. Unlike
    find_c7x_arm_runtime_so(), this is architecture-independent (headers
    are text) so it is returned regardless of host arch. Suitable for
    passing to a C++ compiler as -I.
    """
    bundled = _DATA_DIR / "include"
    if (bundled / "c7x_runtime.h").exists():
        return bundled
    return None


def find_tidl_relax_so() -> Optional[str]:
    """Find tidl_model_import_relax.so in the bundled package."""
    bundled = _DATA_DIR / "tidl" / "tidl_model_import_relax.so"
    if bundled.exists():
        return str(bundled)
    return None


def find_dsp_runtime_dir() -> Optional[Path]:
    """Find the DSP runtime directory for dynmod builds.

    Returns the bundled data directory if it contains the dynmod
    CMakeLists.txt, otherwise returns the source tree path if
    TVM_HOME is set.
    """
    if (_DATA_DIR / "dynmod" / "CMakeLists.txt").exists():
        return _DATA_DIR
    tvm_home = os.environ.get("TVM_HOME")
    if tvm_home:
        src = Path(tvm_home) / "src" / "runtime" / "ti_dsp"
        if src.is_dir():
            return src
    return None


def find_tidl_iobufdesc_layout() -> Optional[str]:
    """Find the sTIDL_IOBufDesc_t layout table bundled by build_wheel.sh.

    Generated from itidl_io.h when the wheel is built, so a wheel install can
    read TIDL's params_1.bin without the (non-redistributable) header.
    """
    bundled = _DATA_DIR / "tidl" / "sTIDL_IOBufDesc_t.layout.json"
    if bundled.exists():
        return str(bundled)
    return None


def find_tidl_tools_dir() -> Optional[str]:
    """Find the tidl_tools directory (perfsim, PC algo, device config) bundled
    for tidl_model_import_relax.so."""
    bundled = _DATA_DIR / "tidl" / "tidl_tools"
    if (bundled / "device_config.cfg").exists():
        return str(bundled)
    return None


def find_host_harness_dir() -> Optional[Path]:
    """Find the bundled c7x_host build harness (dsp-cpp: main_dsp.cpp, io/,
    CMakeLists.txt), staged by build_wheel.sh."""
    bundled = _DATA_DIR / "host"
    if (bundled / "CMakeLists.txt").exists():
        return bundled
    return None
