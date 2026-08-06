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
# --board <same board> already ran against this same checkout: this script
# only deploys and tests, it doesn't build.
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
# Must match build_all.sh's own TVM_BUILD_DIR -- this script assumes that
# script already built into it.
TVM_BUILD_DIR="build-ci-c7x"

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

# TVM core builds into build-ci-c7x/ (build_all.sh), not the plain build/
# name python/tvm/libinfo.py's default search assumes.
export TVM_LIBRARY_PATH="$TVM_HOME/$TVM_BUILD_DIR"
export PYTHONPATH="$TVM_HOME/python${PYTHONPATH:+:$PYTHONPATH}"

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

echo "=== [4/4] Quantized model tests (board=$TVM_BOARD) ==="
mkdir -p "$TVM_HOME/results"
# InceptionV3/ResNeXt101 excluded from the non-mmalib run only: without
# MMALIB's grouped-conv path they fall back to the plain scalar c_static
# path, which fails on real c7x_dload hardware. See the equivalent
# Jenkinsfile stages (Quantized c7x_dload / Quantized MMALIB c7x_dload).
#
# Both suites always run, and to completion, even if one has test
# failures -- unlike the Jenkinsfile's two separate sequential stages
# (where a failure in the first skips the second), a failure here is
# surfaced via the script's own exit code at the end so both JUnit XMLs
# are always available for debugging.
set +e
( cd "$TVM_HOME/tests/ti-dsp-runtime" && \
  pytest -p no:tvm.testing.plugin --rootdir=. quantized/ \
      --dsp-mode=c7x_dload \
      --board "$TVM_BOARD" \
      --ignore=quantized/test_quantized_inception_v3.py \
      --ignore=quantized/test_quantized_resnext101.py \
      -v \
      --junit-xml="$TVM_HOME/results/quantized_dload.xml" )
RC_PLAIN=$?
( cd "$TVM_HOME/tests/ti-dsp-runtime" && \
  pytest -p no:tvm.testing.plugin --rootdir=. quantized/ \
      --dsp-mode=c7x_dload \
      --board "$TVM_BOARD" \
      --mmalib \
      --isolate \
      -v \
      --junit-xml="$TVM_HOME/results/quantized_mmalib_dload.xml" )
RC_MMALIB=$?
set -e

echo "Done. (plain rc=$RC_PLAIN, mmalib rc=$RC_MMALIB)"
if [ "$RC_PLAIN" -ne 0 ] || [ "$RC_MMALIB" -ne 0 ]; then
    exit 1
fi
