#!/bin/bash
#
# Build script for C7x Compute Service DSP Firmware
#
# Usage:
#   ./build.sh          - Build firmware
#   ./build.sh clean    - Clean build directory
#   ./build.sh deploy   - Deploy and start firmware
#
#   --board <j722s-evm|beagley-ai>      - Target board (default: j722s-evm)
#   --ddr <4gb|8gb>                     - Shared-DMA DDR size (default: per-board)
#
# Board/ddr resolve to SDK paths and the shared-DMA physical base entirely
# in cmake/boards.cmake -- this script only forwards the flags and picks a
# build-dir name so switching --ddr never reuses a stale build/sysconfig.
#

set -e

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
FIRMWARE_NAME="c7x_compute.out"

# Use the common deploy script
DEPLOY_SCRIPT="${SCRIPT_DIR}/../../deploy-c7x.sh"

SUBCOMMAND=""
TVM_BOARD=""
TVM_DDR=""
while [ $# -gt 0 ]; do
    case "$1" in
        --board) TVM_BOARD="$2"; shift 2 ;;
        --board=*) TVM_BOARD="${1#*=}"; shift ;;
        --ddr) TVM_DDR="$2"; shift 2 ;;
        --ddr=*) TVM_DDR="${1#*=}"; shift ;;
        *) SUBCOMMAND="$1"; shift ;;
    esac
done

# Board/ddr -> build-dir suffix (naming only; cmake/boards.cmake is the
# sole source of truth for what actually gets built).
BOARD="${TVM_BOARD:-j722s-evm}"
if [ -n "$TVM_DDR" ]; then
    DDR="$TVM_DDR"
elif [ "$BOARD" = "beagley-ai" ]; then
    DDR="4gb"
else
    DDR="8gb"
fi
BUILD_SUFFIX=""
if [ "$BOARD" != "j722s-evm" ] || [ "$DDR" != "8gb" ]; then
    BUILD_SUFFIX="-${BOARD}-${DDR}"
fi
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
        echo "Building C7x Compute Service firmware (board=$BOARD ddr=$DDR)..."
        echo ""
        mkdir -p "${BUILD_DIR}"
        cd "${BUILD_DIR}"
        cmake -DUSE_TIDL_RUNTIME=ON $CMAKE_BOARD_ARGS ..
        make VERBOSE=1
        echo ""
        echo "Build complete: ${BUILD_DIR}/${FIRMWARE_NAME}"
        echo ""
        echo "Next steps:"
        echo "  ./build.sh deploy  - Deploy to target and show trace"
        ;;
esac
