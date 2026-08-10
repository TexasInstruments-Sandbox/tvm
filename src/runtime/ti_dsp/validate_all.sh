#!/bin/bash
# Deploy + hardware-validate the C7x DSP stack for one target board --
# firmware/ARM-client deploy, board health check with recovery, and the
# quantized-model test suite on real c7x_dload hardware.
#
# Meant to be handed to `docker/bash.sh <ci-image> -- bash
# src/runtime/ti_dsp/validate_all.sh ...` so it runs against the
# bind-mounted repo inside the ci_c7x container, pairing with
# build_all.sh -- but nothing here is docker-specific, it runs the same way
# directly on a host with the TI toolchain/SDKs already on PATH/env,
# matching this project's own Build instructions. Assumes build_all.sh
# --board <same board> --wheels already ran against this same checkout:
# this script only deploys and tests, it doesn't build. Tests run against
# the packaged tvm-ti-c7x-compile wheel (the artifact actually shipped to
# users), installed from packaging/ti_dsp/staging-x86/dist/ -- not the
# source tree via PYTHONPATH.
#
# Usage:
#   src/runtime/ti_dsp/validate_all.sh --board <j722s-evm|beagley-ai>
#       [--ddr <4gb|8gb>] [--skip-deploy]
#
# --skip-deploy assumes the board already has the firmware/ARM client this
# checkout just built (e.g. a prior invocation already deployed it) -- the
# board health check and quantized test run still happen.

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
export TVM_HOME="$(cd "$SCRIPT_DIR/../../.." && pwd)"
source "$SCRIPT_DIR/board_build_dir.sh"

TVM_BOARD=""
TVM_DDR=""
SKIP_DEPLOY=0

while [ $# -gt 0 ]; do
    case "$1" in
        --board) TVM_BOARD="$2"; shift 2 ;;
        --board=*) TVM_BOARD="${1#*=}"; shift ;;
        --ddr) TVM_DDR="$2"; shift 2 ;;
        --ddr=*) TVM_DDR="${1#*=}"; shift ;;
        --skip-deploy) SKIP_DEPLOY=1; shift ;;
        -h|--help) sed -n '2,20p' "$0"; exit 0 ;;
        *) echo "Unknown argument: $1" >&2; exit 1 ;;
    esac
done

if [ -z "$TVM_BOARD" ]; then
    echo "Error: --board <j722s-evm|beagley-ai> is required" >&2
    exit 1
fi

case "$TVM_BOARD" in
    beagley-ai) BOARD_TARGET_HOST="beagley-ai" ;;
    *)          BOARD_TARGET_HOST="am67a" ;;
esac

# Firmware output dir -- mirrors build_all.sh's own --wheels-branch
# derivation of FW_SUFFIX, so this can never drift from what actually got
# built (beagley-ai always builds --tidl OFF --mmalib ON; see build_all.sh).
[ "$TVM_BOARD" = "beagley-ai" ] && { TVM_TIDL=OFF; TVM_MMALIB=ON; }
resolve_board_build_dir
FW_SUFFIX="$BUILD_SUFFIX"

echo "=== [1/4] Python test environment ==="
# $HOME inside a docker/bash.sh container is the repo mount point, not the
# host's real home -- default TORCH_HOME to a repo-relative path so a
# containerized caller just needs to bind-mount the host's real torch
# cache to that path rather than fight $HOME semantics.
export TORCH_HOME="${TORCH_HOME:-$TVM_HOME/.cache/torch}"
export VENV_DIR=.venv-ci-c7x
cd "$TVM_HOME"
source tests/ti-dsp-runtime/setup_test_env.sh

# Install the packaged wheel -- the artifact actually shipped to users --
# rather than pointing PYTHONPATH at the source tree, so this validates
# what ships, not just what's on disk in this checkout. Same wheel and
# same --force-reinstall --no-deps flags as tests/ti-dsp-runtime/Jenkinsfile's
# own "Build & Install Wheels" stage.
shopt -s nullglob
X86_WHEELS=("$TVM_HOME"/packaging/ti_dsp/staging-x86/dist/tvm_ti_c7x_compile-*.whl)
shopt -u nullglob
if [ "${#X86_WHEELS[@]}" -eq 0 ]; then
    echo "ERROR: x86 wheel not found under packaging/ti_dsp/staging-x86/dist/" >&2
    echo "  Build it first: bash src/runtime/ti_dsp/build_all.sh --board $TVM_BOARD --wheels" >&2
    exit 1
fi
# build_wheel.sh removes the whole staging dir before each build, so under
# normal operation exactly one wheel matches; if more than one is present
# (e.g. a stale leftover from a manually-copied or interrupted prior
# build), pick the most recently built one rather than whichever sorts
# first lexicographically.
X86_WHEEL="$(ls -t "${X86_WHEELS[@]}" | head -1)"
uv pip install --force-reinstall --no-deps "$X86_WHEEL"

# docker/bash.sh injects PYTHONPATH=<repo>/python into any *ci*-named
# image (see "Set TVM import path inside the docker image" there), which
# takes precedence over the venv's site-packages and silently shadows the
# wheel just installed above with the source tree. Clear it so `import
# tvm` actually resolves to the wheel.
unset PYTHONPATH

wait_for_ssh() {
    local tries=12
    for i in $(seq 1 "$tries"); do
        ssh -o ConnectTimeout=5 -o StrictHostKeyChecking=no \
            "root@${BOARD_TARGET_HOST}" true 2>/dev/null && return 0
        echo "Waiting for ${BOARD_TARGET_HOST} to come back up... ($i/$tries)"
        sleep 10
    done
    ssh -o ConnectTimeout=10 -o StrictHostKeyChecking=no "root@${BOARD_TARGET_HOST}" true
}

reboot_and_wait() {
    ssh -o ConnectTimeout=10 -o StrictHostKeyChecking=no "root@${BOARD_TARGET_HOST}" reboot || true
    sleep 30
    wait_for_ssh
}

if [ "$SKIP_DEPLOY" -eq 0 ]; then
    echo "=== [2/4] Deploy firmware + ARM client (board=$TVM_BOARD) ==="
    reboot_and_wait
    ( cd "$TVM_HOME/src/runtime/ti_dsp/firmware/c7x" && \
      ./deploy-c7x.sh --board "$TVM_BOARD" "dsp/build${FW_SUFFIX}/c7x_compute.out" )
    ( cd "$TVM_HOME/src/runtime/ti_dsp/firmware/c7x/arm" && \
      ./build.sh --board "$TVM_BOARD" deploy )
    # Reboot so remoteproc autostart loads the newly-copied firmware --
    # copying while the board is running avoids the EBUSY error from
    # stopping remoteproc mid-vdev-negotiation (see deploy-c7x.sh).
    reboot_and_wait
else
    echo "=== [2/4] Deploy firmware + ARM client -- skipped (--skip-deploy) ==="
fi

echo "=== [3/4] Board health check ==="
if ! ssh -o ConnectTimeout=10 -o StrictHostKeyChecking=no \
        "root@${BOARD_TARGET_HOST}" "/usr/local/bin/c7x_compute status"; then
    echo "Board unresponsive -- attempting stop/start recovery"
    ( cd "$TVM_HOME/src/runtime/ti_dsp/firmware/c7x" && ./deploy-c7x.sh --board "$TVM_BOARD" --stop ) || true
    sleep 3
    ( cd "$TVM_HOME/src/runtime/ti_dsp/firmware/c7x" && ./deploy-c7x.sh --board "$TVM_BOARD" --start )
    sleep 3
    if ! ssh -o ConnectTimeout=10 -o StrictHostKeyChecking=no \
            "root@${BOARD_TARGET_HOST}" "/usr/local/bin/c7x_compute status"; then
        echo "Stop/start failed -- rebooting board"
        reboot_and_wait
        ssh -o ConnectTimeout=10 -o StrictHostKeyChecking=no \
            "root@${BOARD_TARGET_HOST}" "/usr/local/bin/c7x_compute status"
    fi
fi

echo "=== [4/4] Quantized MMALIB model tests (board=$TVM_BOARD) ==="
# Deliberately MMALIB-only, not both suites -- the plain (non-MMALIB)
# c_static path is exercised by the native Jenkinsfile's own nightly run
# instead; this Docker-based flow's job is validating the shipped wheel +
# MMALIB offload against real hardware, not full test-suite duplication.
mkdir -p "$TVM_HOME/results"
( cd "$TVM_HOME/tests/ti-dsp-runtime" && \
  pytest -p no:tvm.testing.plugin --rootdir=. quantized/ \
      --dsp-mode=c7x_dload \
      --board "$TVM_BOARD" \
      --mmalib \
      --isolate \
      -v \
      --junit-xml="$TVM_HOME/results/quantized_mmalib_dload.xml" )
