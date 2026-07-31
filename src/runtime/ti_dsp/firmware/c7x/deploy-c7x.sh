#!/bin/bash
#
# Deploy C7x firmware to AM67A via Linux remoteproc
#
# Usage:
#   ./deploy-c7x.sh <firmware.out>           # Deploy and restart
#   ./deploy-c7x.sh <firmware.out> --trace   # Deploy, restart, and show trace
#   ./deploy-c7x.sh --stop                   # Stop firmware
#   ./deploy-c7x.sh --start                  # Start firmware
#   ./deploy-c7x.sh --status                 # Show status
#   ./deploy-c7x.sh --trace                  # Show trace buffer
#

set -e

SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"

# --- Parse --board before sourcing common.sh, which resolves TARGET.
# --board is required and TARGET is a deterministic function of it -- no
# env-var override; use an SSH-config alias if your board is reachable
# under a different name.
TVM_BOARD=""
ARGS=()
while [ $# -gt 0 ]; do
    case "$1" in
        --board) TVM_BOARD="$2"; shift 2 ;;
        --board=*) TVM_BOARD="${1#*=}"; shift ;;
        *) ARGS+=("$1"); shift ;;
    esac
done
set -- "${ARGS[@]}"

if [ -z "$TVM_BOARD" ]; then
    echo "Error: --board <j722s-evm|beagley-ai> is required" >&2
    exit 1
fi
case "$TVM_BOARD" in
    beagley-ai) export TARGET="beagley-ai" ;;
    *)          export TARGET="am67a" ;;
esac

# Load shared helpers (ssh_cmd, find_rproc, colors)
source "$SCRIPT_DIR/common.sh"

# Configuration
FIRMWARE_NAME="j722s-c71_0-fw"
FIRMWARE_PATH="/lib/firmware/${FIRMWARE_NAME}"

info() { echo -e "${GREEN}[INFO]${NC} $*"; }
warn() { echo -e "${YELLOW}[WARN]${NC} $*"; }
error() { echo -e "${RED}[ERROR]${NC} $*" >&2; }

check_connection() {
    if ! ssh_cmd "true" 2>/dev/null; then
        error "Cannot connect to ${TARGET}"
        error "If your board is reachable under a different name, add an SSH-config alias for '${TARGET}'"
        exit 1
    fi
    find_rproc || exit 1
    info "Using ${RPROC_ID} (device ${DSP_DEVICE})"
}

get_state() {
    ssh_cmd "cat ${RPROC}/state 2>/dev/null" || echo "unknown"
}

stop_firmware() {
    local state=$(get_state)
    if [ "$state" = "running" ]; then
        info "Stopping C7x firmware..."
        # Retry on EBUSY: the kernel rejects "stop" while virtio vdev
        # negotiation is still in progress after autostart, even though
        # state already reads "running".
        local i
        for i in $(seq 1 10); do
            if ssh_cmd "echo stop > ${RPROC}/state" 2>/dev/null; then
                info "Stopped"
                return 0
            fi
            warn "stop returned EBUSY, retrying ($i/10)..."
            sleep 2
        done
        error "Failed to stop C7x firmware after 10 retries"
        return 1
    else
        info "C7x already stopped (state: $state)"
    fi
}

start_firmware() {
    local state=$(get_state)
    if [ "$state" = "offline" ]; then
        info "Starting C7x firmware..."
        ssh_cmd "echo start > ${RPROC}/state"
        info "Started"
    else
        info "C7x already running (state: $state)"
    fi
}

show_status() {
    echo "=== C7x Remoteproc Status ==="
    echo "Target: ${TARGET}"
    echo "State: $(get_state)"
    echo ""
    echo "=== Remoteproc Info ==="
    ssh_cmd "cat ${RPROC}/name 2>/dev/null || true"
    ssh_cmd "ls -la ${FIRMWARE_PATH} 2>/dev/null || echo 'Firmware not found'"
    echo ""
    echo "=== Recent dmesg ==="
    ssh_cmd "dmesg | grep -i '${DSP_DEVICE}\|remoteproc.*$(basename ${RPROC})' | tail -5"
}

show_trace() {
    info "Trace buffer (${RPROC_DEBUG}/trace0):"
    echo "----------------------------------------"
    ssh_cmd "cat ${RPROC_DEBUG}/trace0 2>/dev/null" || warn "Could not read trace buffer"
    echo "----------------------------------------"
}

deploy_firmware() {
    local firmware="$1"

    if [ ! -f "$firmware" ]; then
        error "Firmware file not found: $firmware"
        exit 1
    fi

    info "Deploying: $firmware"
    info "Target: ${TARGET}:${FIRMWARE_PATH}"

    # Just copy — caller is responsible for rebooting so that remoteproc
    # autostart loads the new firmware cleanly.  Avoid stop/start here:
    # the kernel returns EBUSY while virtio vdev negotiation is still in
    # progress after the board's own autostart, and there is no reliable
    # way to know when that window has passed.
    info "Copying firmware..."
    ssh_cmd "rm -f ${FIRMWARE_PATH}"
    scp -o ConnectTimeout=5 "$firmware" "root@${TARGET}:${FIRMWARE_PATH}"

    # Verify (use -L to follow symlinks, though we removed it)
    local remote_size=$(ssh_cmd "stat -L -c%s ${FIRMWARE_PATH} 2>/dev/null" || echo "0")
    local local_size=$(stat -c%s "$firmware")

    if [ "$remote_size" != "$local_size" ]; then
        error "Size mismatch: local=$local_size remote=$remote_size"
        exit 1
    fi

    info "Copied $(numfmt --to=iec $local_size)"
    info "Firmware staged -- reboot the board to activate"
}

usage() {
    echo "Deploy C7x firmware to AM67A via Linux remoteproc"
    echo ""
    echo "Usage:"
    echo "  $0 <firmware.out> [--trace]   Deploy firmware and optionally show trace"
    echo "  $0 --stop                     Stop C7x firmware"
    echo "  $0 --start                    Start C7x firmware"
    echo "  $0 --restart                  Restart C7x firmware"
    echo "  $0 --status                   Show remoteproc status"
    echo "  $0 --trace                    Show trace buffer"
    echo ""
    echo "  --board <j722s-evm|beagley-ai>  Target board (required)"
    echo "                                  Selects the deploy host:"
    echo "                                  beagley-ai -> beagley-ai, else am67a."
    echo "                                  If reachable under a different name,"
    echo "                                  add an SSH-config alias instead."
    echo ""
    echo "Examples:"
    echo "  $0 --board j722s-evm hello_world_rproc/build/hello_world_rproc.out"
    echo "  $0 --board j722s-evm hello_world_rproc/build/hello_world_rproc.out --trace"
    echo "  $0 --board beagley-ai dsp/build-beagley-ai-4gb/c7x_compute.out"
    echo "  $0 --board j722s-evm --status"
}

# Main
if [ $# -eq 0 ]; then
    usage
    exit 1
fi

check_connection

case "$1" in
    --help|-h)
        usage
        exit 0
        ;;
    --stop)
        stop_firmware
        ;;
    --start)
        start_firmware
        ;;
    --restart)
        stop_firmware
        sleep 1
        start_firmware
        ;;
    --status)
        show_status
        ;;
    --trace)
        show_trace
        ;;
    *)
        # Deploy firmware
        firmware="$1"
        shift
        deploy_firmware "$firmware"

        # Check for --trace flag
        if [ "${1:-}" = "--trace" ]; then
            sleep 1
            show_trace
        fi
        ;;
esac
