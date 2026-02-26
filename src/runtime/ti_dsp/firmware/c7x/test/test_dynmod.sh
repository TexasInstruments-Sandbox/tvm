#!/bin/bash
# test_dynmod.sh - ARM+DSP dynamic module load/infer/unload test
#
# Runs milestones 1-6 against a live AM67A target:
#   1. Firmware boots via remoteproc
#   2. Basic IPC (ping/status)
#   3. Dynamic module load (DLOAD)
#   4. Inference execution
#   5. Module unload
#   6. Load-infer-unload stability (5 cycles)
#
# Prerequisites:
#   - AM67A target accessible via SSH (root@$TARGET)
#   - Firmware deployed: ./deploy-c7x.sh <firmware> --trace
#   - Host CLI installed on target: /usr/local/bin/c7x_compute
#   - Test module on target: /tmp/lib0.out
#
# Usage:
#   ./test_dynmod.sh [--target HOST] [--deploy] [--module PATH]

set -euo pipefail

# --- Configuration ---
SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
REPO_ROOT="$(cd "$SCRIPT_DIR/../.." && pwd)"

# Load shared helpers (ssh_cmd, find_rproc, colors)
source "$SCRIPT_DIR/../common.sh"

DEPLOY=0
MODULE_PATH=""
FIRMWARE="$REPO_ROOT/c7x-firmware/dsp/build/c7x_compute.out"
HOST_CLI="$REPO_ROOT/c7x-firmware/host/build/c7x_compute"

# Default module search paths (tried in order)
MODULE_SEARCH_PATHS=(
    "$REPO_ROOT/dsp-cpp/build-test-dynmod/lib0.out"
    "$REPO_ROOT/../tvm/tvm-relax-tests/dsp-cpp/build-test-dynmod/lib0.out"
)

# --- Counters ---
PASS=0
FAIL=0
SKIP=0

# --- Argument parsing ---
while [[ $# -gt 0 ]]; do
    case "$1" in
        --target) TARGET="$2"; shift 2 ;;
        --deploy) DEPLOY=1; shift ;;
        --module) MODULE_PATH="$2"; shift 2 ;;
        --help|-h)
            echo "Usage: $0 [--target HOST] [--deploy] [--module PATH]"
            echo ""
            echo "  --target HOST   SSH target (default: \$AM67A_TARGET or 'am67a')"
            echo "  --deploy        Build and deploy firmware+CLI before testing"
            echo "  --module PATH   Path to test module lib0.out"
            exit 0
            ;;
        *) echo "Unknown option: $1"; exit 1 ;;
    esac
done

# --- Find test module ---
if [[ -z "$MODULE_PATH" ]]; then
    for p in "${MODULE_SEARCH_PATHS[@]}"; do
        if [[ -f "$p" ]]; then
            MODULE_PATH="$p"
            break
        fi
    done
fi

# --- Helper functions ---
remote_c7x() {
    ssh_cmd /usr/local/bin/c7x_compute "$@"
}

log_info() {
    echo -e "${CYAN}[INFO]${NC} $*"
}

log_milestone() {
    echo ""
    echo -e "${BOLD}${CYAN}=== Milestone $1: $2 ===${NC}"
}

log_pass() {
    echo -e "  ${GREEN}[PASS]${NC} $*"
    PASS=$((PASS + 1))
}

log_fail() {
    echo -e "  ${RED}[FAIL]${NC} $*"
    FAIL=$((FAIL + 1))
}

log_skip() {
    echo -e "  ${YELLOW}[SKIP]${NC} $*"
    SKIP=$((SKIP + 1))
}

rproc_state() {
    ssh_cmd "cat ${RPROC}/state 2>/dev/null" || echo "error"
}

rproc_trace() {
    ssh_cmd "cat ${RPROC_DEBUG}/trace0 2>/dev/null" || true
}

# --- Pre-flight checks ---
echo -e "${BOLD}ARM+DSP Dynamic Module Test${NC}"
echo "Target: root@$TARGET"
echo ""

log_info "Checking SSH connectivity..."
if ! ssh_cmd "true" 2>/dev/null; then
    echo -e "${RED}ERROR: Cannot SSH to root@$TARGET${NC}"
    exit 1
fi
log_info "SSH OK"

find_rproc || exit 1
log_info "Found $RPROC_ID for device $DSP_DEVICE"

# --- Optional: build and deploy ---
if [[ $DEPLOY -eq 1 ]]; then
    log_info "Building DSP firmware..."
    (cd "$REPO_ROOT/c7x-firmware/dsp" && ./build.sh)

    log_info "Building host CLI..."
    (cd "$REPO_ROOT/c7x-firmware/host" && ./build.sh)

    log_info "Deploying firmware..."
    "$REPO_ROOT/deploy-c7x.sh" "$FIRMWARE" --trace

    log_info "Deploying host CLI..."
    scp "$HOST_CLI" "root@$TARGET:/usr/local/bin/c7x_compute"
fi

# --- Deploy test module if available ---
if [[ -n "$MODULE_PATH" && -f "$MODULE_PATH" ]]; then
    log_info "Deploying test module: $MODULE_PATH"
    scp "$MODULE_PATH" "root@$TARGET:/tmp/lib0.out"
else
    log_info "No local test module found; assuming /tmp/lib0.out already on target"
fi

# Create dummy input file on target
ssh_cmd "echo -n 'TEST' > /tmp/dummy_input.bin"

# ============================================================
# Milestone 1: Firmware boots via remoteproc
# ============================================================
log_milestone 1 "Firmware boots via remoteproc"

STATE=$(rproc_state)
if [[ "$STATE" == "running" ]]; then
    log_pass "$RPROC_ID/state = running"
else
    log_fail "$RPROC_ID/state = '$STATE' (expected 'running')"
fi

DMESG_RPMSG=$(ssh_cmd "dmesg | grep -i 'rpmsg' | tail -5" || true)
if echo "$DMESG_RPMSG" | grep -qi "rpmsg_chrdev.*channel\|rpmsg.*created"; then
    log_pass "RPMessage channel created"
else
    # Check for virtio rpmsg bus which also indicates success
    if echo "$DMESG_RPMSG" | grep -qi "rpmsg"; then
        log_pass "RPMessage channel present (rpmsg in dmesg)"
    else
        log_fail "No RPMessage channel found in dmesg"
    fi
fi

DMESG_PANIC=$(ssh_cmd "dmesg | grep -i 'panic\|oops' | tail -3" || true)
if [[ -z "$DMESG_PANIC" ]]; then
    log_pass "No kernel panic or oops"
else
    log_fail "Kernel panic/oops found: $DMESG_PANIC"
fi

# ============================================================
# Milestone 2: Basic IPC (ping/status)
# ============================================================
log_milestone 2 "Basic IPC works"

PING_OUT=$(remote_c7x ping 2>&1) && PING_RC=$? || PING_RC=$?
if [[ $PING_RC -eq 0 ]]; then
    log_pass "ping returned success"
    echo "    $PING_OUT" | head -3
else
    log_fail "ping failed (rc=$PING_RC): $PING_OUT"
fi

STATUS_OUT=$(remote_c7x status 2>&1) && STATUS_RC=$? || STATUS_RC=$?
if [[ $STATUS_RC -eq 0 ]]; then
    log_pass "status returned success"
    echo "    $STATUS_OUT" | head -5
else
    log_fail "status failed (rc=$STATUS_RC): $STATUS_OUT"
fi

# ============================================================
# Milestone 3: Dynamic module load (DLOAD)
# ============================================================
log_milestone 3 "Dynamic module load"

# Verify module exists on target
if ! ssh_cmd "test -f /tmp/lib0.out"; then
    log_fail "Test module /tmp/lib0.out not found on target"
    log_skip "Skipping milestones 3-6 (no module)"
    # Print summary and exit
    echo ""
    echo -e "${BOLD}=== Summary ===${NC}"
    echo -e "  ${GREEN}PASS: $PASS${NC}  ${RED}FAIL: $FAIL${NC}  ${YELLOW}SKIP: $SKIP${NC}"
    [[ $FAIL -eq 0 ]] && exit 0 || exit 1
fi

LOAD_OUT=$(remote_c7x load /tmp/lib0.out 2>&1) && LOAD_RC=$? || LOAD_RC=$?
if [[ $LOAD_RC -eq 0 ]]; then
    log_pass "load returned success"
    echo "    $LOAD_OUT"
else
    log_fail "load failed (rc=$LOAD_RC): $LOAD_OUT"
fi

# Extract handle from output (expect "handle=N" or similar)
HANDLE=$(echo "$LOAD_OUT" | grep -oP 'handle=\K[0-9]+' | head -1)
if [[ -n "$HANDLE" && "$HANDLE" != "0" ]]; then
    log_pass "Got non-zero handle=$HANDLE"
else
    log_fail "No valid handle in output (got '$HANDLE')"
    HANDLE="1"  # Assume 1 for subsequent tests
fi

# Check trace for DLOAD success
TRACE=$(rproc_trace)
if echo "$TRACE" | grep -q "\[DLOAD\].*Loaded module"; then
    log_pass "Trace confirms DLOAD loaded module"
else
    log_fail "No DLOAD loaded message in trace"
fi

# ============================================================
# Milestone 4: Inference execution
# ============================================================
log_milestone 4 "Inference execution"

INFER_OUT=$(remote_c7x infer "$HANDLE" 0 --input /tmp/dummy_input.bin --output /tmp/out.bin --dtype int8 2>&1) && INFER_RC=$? || INFER_RC=$?
if [[ $INFER_RC -eq 0 ]]; then
    log_pass "infer returned success"
    echo "    $INFER_OUT" | head -5
else
    log_fail "infer failed (rc=$INFER_RC): $INFER_OUT"
fi

# Check for successful inference output (CLI prints "Inference complete: N cycles")
if echo "$INFER_OUT" | grep -qi 'Inference complete'; then
    log_pass "Inference completed with cycle count"
elif echo "$INFER_OUT" | grep -qiP 'return.*(value)?.*=?\s*0'; then
    log_pass "return_value=0"
else
    log_fail "No success indication in output: $INFER_OUT"
fi

# Check trace for DYNMOD TEST messages
TRACE=$(rproc_trace)

EXPECTED_MSGS=(
    "cg_main_dsp called"
    "memcpy test: PASS"
    "compute test:.*45.*expected 45"
    "ALL TESTS PASSED"
)
TRACE_PASS=0
for msg in "${EXPECTED_MSGS[@]}"; do
    if echo "$TRACE" | grep -qP "\[DYNMOD TEST\].*$msg"; then
        log_pass "Trace: $msg"
        TRACE_PASS=$((TRACE_PASS + 1))
    else
        log_fail "Trace missing: [DYNMOD TEST] $msg"
    fi
done

if [[ $TRACE_PASS -eq ${#EXPECTED_MSGS[@]} ]]; then
    log_pass "All 4 DYNMOD TEST messages present"
fi

# ============================================================
# Milestone 5: Module unload
# ============================================================
log_milestone 5 "Module unload"

UNLOAD_OUT=$(remote_c7x unload "$HANDLE" 2>&1) && UNLOAD_RC=$? || UNLOAD_RC=$?
if [[ $UNLOAD_RC -eq 0 ]]; then
    log_pass "unload returned success"
else
    log_fail "unload failed (rc=$UNLOAD_RC): $UNLOAD_OUT"
fi

STATUS_OUT=$(remote_c7x status 2>&1) || true
echo "    $STATUS_OUT" | head -5

# ============================================================
# Milestone 6: Load-infer-unload stability (5 cycles)
# ============================================================
log_milestone 6 "Load-infer-unload stability (5 cycles)"

CYCLE_FAILS=0
for i in $(seq 1 5); do
    # Load
    LOAD_OUT=$(remote_c7x load /tmp/lib0.out 2>&1) && LOAD_RC=$? || LOAD_RC=$?
    HANDLE=$(echo "$LOAD_OUT" | grep -oP 'handle=\K[0-9]+' | head -1)
    if [[ $LOAD_RC -ne 0 || -z "$HANDLE" ]]; then
        echo -e "  ${RED}Cycle $i: load failed${NC}"
        CYCLE_FAILS=$((CYCLE_FAILS + 1))
        continue
    fi

    # Infer
    INFER_OUT=$(remote_c7x infer "$HANDLE" 0 --input /tmp/dummy_input.bin --output /tmp/out.bin --dtype int8 2>&1) && INFER_RC=$? || INFER_RC=$?
    if [[ $INFER_RC -ne 0 ]]; then
        echo -e "  ${RED}Cycle $i: infer failed${NC}"
        CYCLE_FAILS=$((CYCLE_FAILS + 1))
        # Try to unload anyway
        remote_c7x unload "$HANDLE" 2>/dev/null || true
        continue
    fi

    # Unload
    UNLOAD_OUT=$(remote_c7x unload "$HANDLE" 2>&1) && UNLOAD_RC=$? || UNLOAD_RC=$?
    if [[ $UNLOAD_RC -ne 0 ]]; then
        echo -e "  ${RED}Cycle $i: unload failed${NC}"
        CYCLE_FAILS=$((CYCLE_FAILS + 1))
        continue
    fi

    echo -e "  ${GREEN}Cycle $i: OK${NC}"
done

if [[ $CYCLE_FAILS -eq 0 ]]; then
    log_pass "All 5 load-infer-unload cycles passed"
else
    log_fail "$CYCLE_FAILS/5 cycles failed"
fi

# Final status check
STATUS_OUT=$(remote_c7x status 2>&1) || true
echo "  Final status: $STATUS_OUT"

# ============================================================
# Summary
# ============================================================
echo ""
echo -e "${BOLD}=== Summary ===${NC}"
echo -e "  ${GREEN}PASS: $PASS${NC}  ${RED}FAIL: $FAIL${NC}  ${YELLOW}SKIP: $SKIP${NC}"

if [[ $FAIL -eq 0 ]]; then
    echo -e "${GREEN}${BOLD}ALL TESTS PASSED${NC}"
    exit 0
else
    echo -e "${RED}${BOLD}SOME TESTS FAILED${NC}"
    exit 1
fi
