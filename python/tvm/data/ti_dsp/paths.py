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
