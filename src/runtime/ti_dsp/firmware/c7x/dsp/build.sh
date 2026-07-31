#!/bin/bash
#
# Build script for C7x Compute Service DSP Firmware
#
# Usage:
#   ./build.sh          - Build firmware
#   ./build.sh clean    - Clean build directory
#   ./build.sh deploy   - Deploy and start firmware
#
#   --board <j722s-evm|beagley-ai>      - Target board (required)
#   --ddr <4gb|8gb>                     - Shared-DMA DDR size (default: per-board)
#   --tidl <ON|OFF>                     - Link TIDL algo libs (default: ON)
#   --mmalib <ON|OFF>                   - Link MMALIB direct-integration libs
#                                          (default: OFF; forced ON if --tidl
#                                          is ON, since TIDL requires MMALIB
#                                          at link time)
#
# Board/ddr resolve to SDK paths and the shared-DMA physical base entirely
# in cmake/boards.cmake -- this script only forwards the flags and picks a
# build-dir name so switching --ddr/--tidl/--mmalib never reuses a stale
# build/sysconfig.
#

set -e

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
FIRMWARE_NAME="c7x_compute.out"

# Use the common deploy script
DEPLOY_SCRIPT="${SCRIPT_DIR}/../../deploy-c7x.sh"

SUBCOMMAND=""
TVM_BOARD=""
TVM_DDR=""
TVM_TIDL=""
TVM_MMALIB=""
while [ $# -gt 0 ]; do
    case "$1" in
        --board) TVM_BOARD="$2"; shift 2 ;;
        --board=*) TVM_BOARD="${1#*=}"; shift ;;
        --ddr) TVM_DDR="$2"; shift 2 ;;
        --ddr=*) TVM_DDR="${1#*=}"; shift ;;
        --tidl) TVM_TIDL="$2"; shift 2 ;;
        --tidl=*) TVM_TIDL="${1#*=}"; shift ;;
        --mmalib) TVM_MMALIB="$2"; shift 2 ;;
        --mmalib=*) TVM_MMALIB="${1#*=}"; shift ;;
        *) SUBCOMMAND="$1"; shift ;;
    esac
done

# Backward-compat: TIDL defaulted ON unconditionally before --tidl existed.
TVM_TIDL="${TVM_TIDL:-ON}"

if [ -z "$TVM_BOARD" ]; then
    echo "Error: --board <j722s-evm|beagley-ai> is required" >&2
    exit 1
fi

# Board/ddr/tidl/mmalib -> build-dir suffix. Shared with build_runtime.sh
# and firmware/c7x/arm/build.sh so all three always agree on where a
# given board's artifacts live; cmake/boards.cmake remains the sole
# source of truth for what actually gets built. Also sets MMALIB (the
# CMakeLists force-enables MMALIB whenever TIDL is ON, so this mirrors
# that here for the build-dir key and the banner below).
source "${SCRIPT_DIR}/../../../board_build_dir.sh"
resolve_board_build_dir
BUILD_DIR="${SCRIPT_DIR}/build${BUILD_SUFFIX}"

# Plain strings, not arrays: values are always simple enum tokens (no
# spaces), and an empty array expanded under `set -u` is an unbound-variable
# error on bash < 4.4.
CMAKE_BOARD_ARGS=""
[ -n "$TVM_BOARD" ] && CMAKE_BOARD_ARGS="$CMAKE_BOARD_ARGS -DTVM_BOARD=$TVM_BOARD"
[ -n "$TVM_DDR" ] && CMAKE_BOARD_ARGS="$CMAKE_BOARD_ARGS -DTVM_DDR=$TVM_DDR"

DEPLOY_BOARD_ARGS=""
[ -n "$TVM_BOARD" ] && DEPLOY_BOARD_ARGS="--board $TVM_BOARD"

case "$SUBCOMMAND" in
    clean)
        echo "Cleaning build directory..."
        rm -rf "${BUILD_DIR}"
        echo "Done."
        ;;
    deploy)
        if [ ! -f "${BUILD_DIR}/${FIRMWARE_NAME}" ]; then
            echo "Error: Firmware not found. Run './build.sh' first."
            exit 1
        fi
        if [ ! -x "${DEPLOY_SCRIPT}" ]; then
            echo "Error: Deploy script not found at ${DEPLOY_SCRIPT}"
            exit 1
        fi
        "${DEPLOY_SCRIPT}" $DEPLOY_BOARD_ARGS "${BUILD_DIR}/${FIRMWARE_NAME}" --trace
        ;;
    *)
        echo "Building C7x Compute Service firmware (board=$BOARD ddr=$DDR tidl=$TVM_TIDL mmalib=$MMALIB)..."
        echo ""
        mkdir -p "${BUILD_DIR}"
        cd "${BUILD_DIR}"
        cmake -DUSE_TIDL_RUNTIME=$TVM_TIDL -DUSE_TI_MMALIB=${TVM_MMALIB:-OFF} $CMAKE_BOARD_ARGS ..
        make VERBOSE=1
        echo ""
        echo "Build complete: ${BUILD_DIR}/${FIRMWARE_NAME}"
        if [ "$TVM_TIDL" != "ON" ]; then
            echo ""
            echo "NOTE: This firmware has no TIDL kernels. Compile models with"
            echo "      '-tidl-kernels=0' in the c_static target so max_pool2d"
            echo "      lowers to c7x_int8_max_pool; the default emits"
            echo "      c7x_int8_max_pool_tidl, which is absent here and fails"
            echo "      to resolve at DLOAD load time."
        fi
        echo ""
        echo "Next steps:"
        echo "  ./build.sh deploy  - Deploy to target and show trace"
        ;;
esac
