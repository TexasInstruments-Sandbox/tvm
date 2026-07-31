#!/bin/bash
# Build the full C7x DSP stack -- TVM core, DSP runtime, firmware, and
# ARM client -- for one target board in a single invocation.
#
# Meant to be handed to `docker/bash.sh <ci-image> -- bash
# src/runtime/ti_dsp/build_all.sh ...` so it runs against the
# bind-mounted repo inside the ci_c7x container, but nothing here is
# docker-specific -- it runs the same way directly on a host with the TI
# toolchain/SDKs already on PATH/env, matching CLAUDE.md's Build section.
#
# Usage:
#   src/runtime/ti_dsp/build_all.sh [--board <j722s-evm|beagley-ai>]
#       [--ddr <4gb|8gb>] [--wheels]
#
# --wheels additionally builds the x86/arm64 packaging wheels. Off by
# default: packaging/ti_dsp/build_wheel.sh predates --board support and
# hardcodes unsuffixed build/ directory names, so --wheels has to bridge
# to it via symlinks (see below) -- more moving parts than the plain
# artifact build, so it's opt-in rather than the default path.
#
# Board-specific TIDL/MMALIB linkage follows the project convention
# documented in CLAUDE.md: beagley-ai always builds --tidl OFF --mmalib
# ON (no TIDL subgraph offload on that board); j722s-evm/am67a uses
# firmware/c7x/dsp/build.sh's own default (--tidl ON, which forces
# --mmalib ON too).

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
export TVM_HOME="$(cd "$SCRIPT_DIR/../../.." && pwd)"
source "$SCRIPT_DIR/board_build_dir.sh"

TVM_BOARD="j722s-evm"
TVM_DDR=""
BUILD_WHEELS=0
# Deliberately not the plain `build/` name, so this script never
# collides with a developer's own native (non-container) TVM build in
# the same bind-mounted repo.
TVM_BUILD_DIR="build-ci-c7x"

while [ $# -gt 0 ]; do
    case "$1" in
        --board) TVM_BOARD="$2"; shift 2 ;;
        --board=*) TVM_BOARD="${1#*=}"; shift ;;
        --ddr) TVM_DDR="$2"; shift 2 ;;
        --ddr=*) TVM_DDR="${1#*=}"; shift ;;
        --wheels) BUILD_WHEELS=1; shift ;;
        -h|--help) sed -n '2,20p' "$0"; exit 0 ;;
        *) echo "Unknown argument: $1" >&2; exit 1 ;;
    esac
done

DDR_ARGS=()
[ -n "$TVM_DDR" ] && DDR_ARGS=(--ddr "$TVM_DDR")

# Only beagley-ai needs an explicit TIDL/MMALIB override -- see CLAUDE.md's
# Board support section.
FW_TIDL_ARGS=()
[ "$TVM_BOARD" = "beagley-ai" ] && FW_TIDL_ARGS=(--tidl OFF --mmalib ON)

echo "=== [1/4] TVM core ==="
mkdir -p "$TVM_BUILD_DIR"
cp cmake/config.cmake "$TVM_BUILD_DIR/"
( cd "$TVM_BUILD_DIR" && cmake -G Ninja .. && ninja )

echo "=== [2/4] DSP runtime (c7x_host) ==="
( cd src/runtime/ti_dsp && bash build_runtime.sh c7x_host )

echo "=== [3/4] DSP runtime (c7x, board=$TVM_BOARD) ==="
( cd src/runtime/ti_dsp && bash build_runtime.sh c7x --board "$TVM_BOARD" "${DDR_ARGS[@]}" )

echo "=== [4/4] Firmware + ARM client (board=$TVM_BOARD) ==="
( cd src/runtime/ti_dsp/firmware/c7x/dsp && \
  ./build.sh --board "$TVM_BOARD" "${DDR_ARGS[@]}" "${FW_TIDL_ARGS[@]}" )
( cd src/runtime/ti_dsp/firmware/c7x/arm && ./build.sh --board "$TVM_BOARD" )

if [ "$BUILD_WHEELS" -eq 0 ]; then
    echo "Done."
    exit 0
fi

echo "=== Wheels ==="
# packaging/ti_dsp/build_wheel.sh predates --board support and hardcodes
# unsuffixed build/ directory names (build/, firmware/c7x/dsp/build/,
# firmware/c7x/arm/build/). Bridge via symlinks -- computed by calling
# the SAME resolve_board_build_dir() the actual builds above used, not
# a re-derived copy, so this can't drift from what really got built.
resolve_board_build_dir
RT_SUFFIX="$BUILD_SUFFIX"           # runtime/arm dirs (no tidl/mmalib)
[ "$TVM_BOARD" = "beagley-ai" ] && { TVM_TIDL=OFF; TVM_MMALIB=ON; }
resolve_board_build_dir
FW_SUFFIX="$BUILD_SUFFIX"           # firmware dir (includes tidl/mmalib)

ln -sfn "$TVM_BUILD_DIR" build
ln -sfn "build-c7x${RT_SUFFIX}" src/runtime/ti_dsp/build-c7x
ln -sfn "build${FW_SUFFIX}" src/runtime/ti_dsp/firmware/c7x/dsp/build
ln -sfn "build${RT_SUFFIX}" src/runtime/ti_dsp/firmware/c7x/arm/build

# `build` (for `python -m build`) lives in a venv baked into the image
# at /opt/venv-build -- not created here in the mounted repo, so there's
# no leftover-venv state to clean up between runs and no risk of it
# ending up inside the repo checkout, and nothing here needs network
# access. See docker/Dockerfile.ci_c7x.
export VIRTUAL_ENV=/opt/venv-build
export PATH="$VIRTUAL_ENV/bin:$PATH"

# build_wheel.sh's own --tidl mirrors firmware/c7x/dsp/build.sh's: the x86
# wheel bundles a TIDL .so by default, which beagley-ai never builds.
WHEEL_TIDL_ARGS=()
[ "$TVM_BOARD" = "beagley-ai" ] && WHEEL_TIDL_ARGS=(--tidl OFF)

bash packaging/ti_dsp/build_wheel.sh --target x86 "${WHEEL_TIDL_ARGS[@]}"
bash packaging/ti_dsp/build_wheel.sh --target arm64

echo "Done. Wheels in packaging/ti_dsp/staging-{x86,arm64}/dist/"
