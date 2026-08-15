"""On-board deploy helper for the tvm-ti-c7x-inference wheel.

Copies the wheel's bundled DSP firmware, ARM CLI binary, and runtime
shared library to the fixed system paths the scp-based deploy flow
(deploy-c7x.sh, firmware/c7x/arm/build.sh's `deploy` subcommand) has
always used, so every existing consumer that shells out to
/usr/local/bin/c7x_compute or expects /lib/firmware/j722s-c71_0-fw
keeps working unmodified regardless of how the files got there.

Run as root on the target board, after `pip install`ing this wheel:

    python3 -m tvm.data.ti_dsp.deploy
"""

import os
import platform
import subprocess
import sys
import tempfile
from typing import Optional

from tvm.data.ti_dsp.paths import (
    find_c7x_arm_runtime_so,
    find_c7x_compute_binary,
    get_ti_dsp_path,
)

# Fixed on-board targets -- must match deploy-c7x.sh's FIRMWARE_PATH and
# firmware/c7x/arm/build.sh's `deploy` subcommand exactly, since existing
# consumers (dsp_utils.py, Jenkinsfile, conftest.py health checks, ...)
# hardcode these same paths regardless of how the files got there.
FIRMWARE_PATH = "/lib/firmware/j722s-c71_0-fw"
CLI_PATH = "/usr/local/bin/c7x_compute"
SO_PATH = "/usr/local/lib/libc7x_arm_runtime.so"
SO_SONAME_PATH = SO_PATH + ".1"
LDCONFIG_CONF_PATH = "/etc/ld.so.conf.d/c7x_arm_runtime.conf"


def _atomic_copy(src: str, dst: str, mode: Optional[int] = None) -> None:
    """Copy src to dst via a same-directory temp file + os.replace().

    Never truncates dst in place -- a process with dst's old inode
    still mapped (e.g. a long-lived `c7x_compute session-run`) keeps
    running against the old bytes until it exits, instead of reading a
    half-written file.
    """
    dst_dir = os.path.dirname(dst)
    fd, tmp_path = tempfile.mkstemp(dir=dst_dir, prefix=".deploy-")
    try:
        with os.fdopen(fd, "wb") as tmp_f, open(src, "rb") as src_f:
            tmp_f.write(src_f.read())
        if mode is not None:
            os.chmod(tmp_path, mode)
        os.replace(tmp_path, dst)
    except Exception:
        os.unlink(tmp_path)
        raise


def _atomic_symlink(target: str, link_path: str) -> None:
    """Create/replace a symlink atomically (temp name + os.replace())."""
    tmp_path = f"{link_path}.tmp{os.getpid()}"
    if os.path.lexists(tmp_path):
        os.unlink(tmp_path)
    os.symlink(target, tmp_path)
    os.replace(tmp_path, link_path)


def main() -> int:
    if os.geteuid() != 0:
        print(
            "error: must run as root (writes to /usr/local, /lib/firmware)",
            file=sys.stderr,
        )
        return 1
    if platform.machine() not in ("aarch64", "arm64"):
        print(
            "error: this deploy helper only runs on the board (aarch64), "
            f"not {platform.machine()!r}",
            file=sys.stderr,
        )
        return 1

    firmware_src = get_ti_dsp_path() / "firmware" / "c7x_compute.out"
    cli_src = find_c7x_compute_binary()
    so_src = find_c7x_arm_runtime_so()

    missing = []
    if not firmware_src.exists():
        missing.append("c7x_compute.out")
    if cli_src is None:
        missing.append("c7x_compute")
    if so_src is None:
        missing.append("libc7x_arm_runtime.so")
    if missing:
        print(
            f"error: wheel is missing bundled artifact(s): {', '.join(missing)}",
            file=sys.stderr,
        )
        return 1
    # Narrows cli_src/so_src from Optional[str] to str for the type checker;
    # both are guaranteed non-None here since the missing-artifact check
    # above already returned if either was None.
    assert cli_src is not None and so_src is not None

    # A long-lived `c7x_compute session-run` process keeps these files'
    # text pages mapped; kill it before overwriting so nothing observes
    # a mid-write state even transiently. Best-effort: nothing running
    # is not an error.
    subprocess.run(["pkill", "-x", "c7x_compute"], check=False)

    print(f"Copying firmware: {firmware_src} -> {FIRMWARE_PATH}")
    # mkstemp's default mode is 0600 (owner-only) -- explicit mode on every
    # copy below, not just the CLI binary, so the .so stays readable by
    # non-root callers (e.g. C7xVirtualMachine) and the firmware file stays
    # readable by whatever loads it, matching the old scp flow's perms.
    _atomic_copy(str(firmware_src), FIRMWARE_PATH, mode=0o644)
    local_size = os.path.getsize(firmware_src)
    remote_size = os.path.getsize(FIRMWARE_PATH)
    if local_size != remote_size:
        print(
            f"error: firmware size mismatch after copy: src={local_size} dst={remote_size}",
            file=sys.stderr,
        )
        return 1

    print(f"Copying CLI: {cli_src} -> {CLI_PATH}")
    _atomic_copy(cli_src, CLI_PATH, mode=0o755)

    print(f"Copying runtime library: {so_src} -> {SO_PATH}")
    _atomic_copy(so_src, SO_PATH, mode=0o755)

    os.makedirs(os.path.dirname(LDCONFIG_CONF_PATH), exist_ok=True)
    with open(LDCONFIG_CONF_PATH, "w") as f:
        f.write("/usr/local/lib\n")
    _atomic_symlink(SO_PATH, SO_SONAME_PATH)
    subprocess.run(["ldconfig"], check=True)

    print("Firmware staged -- reboot the board to activate")
    return 0


if __name__ == "__main__":
    sys.exit(main())
