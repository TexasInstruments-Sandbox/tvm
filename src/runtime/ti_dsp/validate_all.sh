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
#       [--x86-wheel <path>] [--arm64-wheel <path>]
#
# --skip-deploy assumes the board already has the firmware/ARM client this
# checkout just built (e.g. a prior invocation already deployed it) -- the
# board health check and quantized test run still happen.
#
# --x86-wheel/--arm64-wheel each accept either an exact .whl file or a
# directory to glob (same discovery rule as the default
# packaging/ti_dsp/staging-{x86,arm64}/dist/ lookup) -- this is what lets a
# wheel pair built elsewhere (e.g. a c7x-build-wheels.yml GitHub Actions
# artifact) be validated against real hardware without a local build_all.sh
# run first.

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
export TVM_HOME="$(cd "$SCRIPT_DIR/../../.." && pwd)"
source "$SCRIPT_DIR/board_build_dir.sh"

TVM_BOARD=""
TVM_DDR=""
SKIP_DEPLOY=0
X86_WHEEL_OVERRIDE=""
ARM64_WHEEL_OVERRIDE=""

while [ $# -gt 0 ]; do
    case "$1" in
        --board) TVM_BOARD="$2"; shift 2 ;;
        --board=*) TVM_BOARD="${1#*=}"; shift ;;
        --ddr) TVM_DDR="$2"; shift 2 ;;
        --ddr=*) TVM_DDR="${1#*=}"; shift ;;
        --skip-deploy) SKIP_DEPLOY=1; shift ;;
        --x86-wheel) X86_WHEEL_OVERRIDE="$2"; shift 2 ;;
        --x86-wheel=*) X86_WHEEL_OVERRIDE="${1#*=}"; shift ;;
        --arm64-wheel) ARM64_WHEEL_OVERRIDE="$2"; shift 2 ;;
        --arm64-wheel=*) ARM64_WHEEL_OVERRIDE="${1#*=}"; shift ;;
        -h|--help) sed -n '2,32p' "$0"; exit 0 ;;
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

# Resolves to a single wheel path: an explicit override (either an exact
# .whl file or a directory to glob) if given, else $default_dir. Picking
# the newest match (`ls -t | head -1`) handles both the normal case
# (exactly one wheel, since build_wheel.sh clears its staging dir before
# each build) and a stale leftover from an interrupted/manually-copied
# prior build.
resolve_wheel() {
    local override="$1" glob_pattern="$2" default_dir="$3" label="$4"
    local dir="${override:-$default_dir}"
    if [ -n "$override" ] && [ -f "$override" ]; then
        echo "$override"
        return 0
    fi
    shopt -s nullglob
    local matches=("$dir"/$glob_pattern)
    shopt -u nullglob
    if [ "${#matches[@]}" -eq 0 ]; then
        echo "ERROR: no $label wheel found under $dir" >&2
        echo "  Build it first: bash src/runtime/ti_dsp/build_all.sh --board $TVM_BOARD --wheels" >&2
        echo "  Or pass an explicit --$label-wheel <path-to-.whl-or-dir>" >&2
        exit 1
    fi
    ls -t "${matches[@]}" | head -1
}

echo "=== [1/5] Python test environment ==="
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
X86_WHEEL="$(resolve_wheel "$X86_WHEEL_OVERRIDE" "tvm_ti_c7x_compile-*.whl" \
    "$TVM_HOME/packaging/ti_dsp/staging-x86/dist" "x86")"
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
    echo "=== [2/5] Deploy firmware + ARM client (board=$TVM_BOARD) ==="
    reboot_and_wait
    if [ "$TVM_BOARD" = "beagley-ai" ]; then
        # BeagleY-AI deploys from the packaged arm64 inference wheel rather
        # than scp'ing raw build output: tvm.data.ti_dsp.deploy (bundled in
        # the wheel) copies the same firmware/CLI/runtime-lib files to the
        # same system paths deploy-c7x.sh/arm/build.sh's `deploy` subcommand
        # have always used, so nothing downstream (health check, quantized
        # tests) needs to know which path produced them. This is also what
        # lets --arm64-wheel point at a GitHub-Actions-built artifact instead
        # of a local build.
        ARM64_WHEEL="$(resolve_wheel "$ARM64_WHEEL_OVERRIDE" "tvm_ti_c7x_inference-*.whl" \
            "$TVM_HOME/packaging/ti_dsp/staging-arm64/dist" "arm64")"
        REMOTE_WHEEL="/tmp/$(basename "$ARM64_WHEEL")"
        scp -o ConnectTimeout=10 "$ARM64_WHEEL" "root@${BOARD_TARGET_HOST}:${REMOTE_WHEEL}"
        # One ssh session so `python3` resolves identically for the install
        # and the deploy helper. A stock BeagleY-AI image has no pip at all
        # (no python3-pip, no ensurepip) -- install it via apt on demand,
        # forwarding the proxy vars the same way docker/README_c7x.md
        # already documents for the numpy prerequisite. Once pip exists,
        # probe for --break-system-packages support statically (grep on
        # `pip --help`) rather than guessing or retrying after a failed
        # install: this board's pip enforces PEP 668
        # externally-managed-environment restrictions, and this way there's
        # no double-install either way.
        ssh -o ConnectTimeout=10 -o StrictHostKeyChecking=no "root@${BOARD_TARGET_HOST}" \
            bash -s -- "$REMOTE_WHEEL" "${http_proxy:-}" "${https_proxy:-}" <<'REMOTE_EOF'
set -e
remote_wheel="$1"
export http_proxy="$2"
export https_proxy="$3"
if ! python3 -m pip --version >/dev/null 2>&1; then
    export DEBIAN_FRONTEND=noninteractive
    apt-get update -qq
    apt-get install -y -qq python3-pip
fi
break_flag=""
python3 -m pip install --help 2>/dev/null | grep -q -- --break-system-packages && \
    break_flag="--break-system-packages"
# $break_flag is deliberately unquoted: empty must vanish via word
# splitting, not be passed to pip as a literal empty-string argument.
python3 -m pip install --quiet --force-reinstall --no-deps $break_flag "$remote_wheel"
python3 -m tvm.data.ti_dsp.deploy
rm -f "$remote_wheel"
REMOTE_EOF
    else
        ( cd "$TVM_HOME/src/runtime/ti_dsp/firmware/c7x" && \
          ./deploy-c7x.sh --board "$TVM_BOARD" "dsp/build${FW_SUFFIX}/c7x_compute.out" )
        ( cd "$TVM_HOME/src/runtime/ti_dsp/firmware/c7x/arm" && \
          ./build.sh --board "$TVM_BOARD" deploy )
    fi
    # Reboot so remoteproc autostart loads the newly-copied firmware --
    # copying while the board is running avoids the EBUSY error from
    # stopping remoteproc mid-vdev-negotiation (see deploy-c7x.sh).
    reboot_and_wait
else
    echo "=== [2/5] Deploy firmware + ARM client -- skipped (--skip-deploy) ==="
fi

echo "=== [3/5] Board health check ==="
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

echo "=== [4/5] Quantized MMALIB model tests (board=$TVM_BOARD) ==="
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

echo "=== [5/5] Standalone example smoke tests (board=$TVM_BOARD) ==="
# tests/ti-dsp-runtime/examples/ -- hand-run demos of the public offload
# APIs (Python C7xVirtualMachine, C++ c7x::Module) that aren't otherwise
# covered by any pytest suite. Reuses the firmware/ARM client this script
# already deployed above. Full defaults (no --image): YOLO26 runs all 3
# bundled test images, ResNet-18 its 1 default image.
( cd "$TVM_HOME/tests/ti-dsp-runtime" && \
  python examples/run_yolo26_detection.py \
      --board "$TVM_BOARD" \
      2>&1 | tee "$TVM_HOME/results/example_yolo26.log" )
( cd "$TVM_HOME/tests/ti-dsp-runtime" && \
  python examples/run_resnet18_classification.py \
      --board "$TVM_BOARD" \
      2>&1 | tee "$TVM_HOME/results/example_resnet18.log" )
