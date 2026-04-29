"""Pytest configuration for unit tests."""

import sys
from pathlib import Path

_THIS_DIR = Path(__file__).parent
_PARENT_DIR = _THIS_DIR.parent

# Add parent directory to sys.path so we can import tvm_utils
sys.path.insert(0, str(_PARENT_DIR))

# Ensure symlinks exist for shared directories
for _dirname in ("cpp", "test_images"):
    _link = _THIS_DIR / _dirname
    _target = _PARENT_DIR / _dirname
    if not _link.exists() and _target.exists():
        _link.symlink_to(f"../{_dirname}")
