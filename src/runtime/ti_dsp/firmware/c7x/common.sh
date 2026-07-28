#!/bin/bash
#
# Shared helpers for C7x firmware scripts (deploy, test, etc.)
#
# Source this file from other scripts:
#   SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
#   source "$SCRIPT_DIR/common.sh"       # from c7x-firmware/
#   source "$SCRIPT_DIR/../common.sh"    # from c7x-firmware/test/
#

# --- Configuration (can be overridden before sourcing) ---
TARGET="${TARGET:-${BOARD_HOSTNAME:-am67a}}"
DSP_DEVICE="${DSP_DEVICE:-7e000000.dsp}"

# Resolved by find_rproc()
RPROC_ID=""
RPROC=""
RPROC_DEBUG=""

# --- Colors ---
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
CYAN='\033[0;36m'
BOLD='\033[1m'
NC='\033[0m'

# --- SSH helper ---
# StrictHostKeyChecking=no is appropriate for dev-board targets
# whose host keys change across reflash cycles.
ssh_cmd() {
    ssh -o ConnectTimeout=5 -o StrictHostKeyChecking=no \
        "root@${TARGET}" "$@"
}

# --- Remoteproc discovery ---
# Find the remoteproc instance for our DSP by matching the device
# tree address.  The remoteproc index can change across reboots,
# but the device tree address (e.g. 7e000000.dsp) is stable.
#
# Sets: RPROC_ID  (e.g. "remoteproc0")
#       RPROC     (/sys/class/remoteproc/remoteproc0)
#       RPROC_DEBUG (/sys/kernel/debug/remoteproc/remoteproc0)
find_rproc() {
    RPROC_ID=$(ssh_cmd "
        for rp in /sys/class/remoteproc/remoteproc*; do
            if [ \"\$(basename \$(readlink -f \$rp/device) 2>/dev/null)\" = \"${DSP_DEVICE}\" ]; then
                basename \$rp
                exit 0
            fi
        done
        exit 1
    " 2>/dev/null) || {
        echo -e "${RED}[ERROR]${NC} Could not find remoteproc for device ${DSP_DEVICE}" >&2
        echo -e "${RED}[ERROR]${NC} Available remoteprocs:" >&2
        ssh_cmd "for rp in /sys/class/remoteproc/remoteproc*; do \
            echo \"  \$(basename \$rp): device=\$(basename \$(readlink -f \$rp/device) 2>/dev/null) \
firmware=\$(cat \$rp/firmware 2>/dev/null)\"; done" 2>/dev/null >&2 || true
        return 1
    }
    RPROC="/sys/class/remoteproc/${RPROC_ID}"
    RPROC_DEBUG="/sys/kernel/debug/remoteproc/${RPROC_ID}"
}
