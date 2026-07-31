#!/bin/bash
#
# Build script for C7x Compute Service - ARM Client
#
# Usage:
#   ./build.sh          - Cross-compile for ARM64 (default)
#   ./build.sh native   - Native compile (run on target)
#   ./build.sh clean    - Clean build directory
#   ./build.sh deploy   - Deploy to target
#
#   --board <j722s-evm|beagley-ai>      - Target board (default: j722s-evm)
#   --ddr <4gb|8gb>                     - Shared-DMA DDR size (default: per-board)
#
# --board/--ddr only affect the cosmetic C7X_SHARED_PHYS_BASE sanity check
# in c7x_compute_client.cpp (see cmake/boards.cmake); forwarded here purely
# for build-dir naming consistency with build_runtime.sh/dsp/build.sh.
#

set -e

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
TARGET_HOST="${BOARD_HOSTNAME:-am67a}"

# ARM64 cross-compiler (Ubuntu packages: gcc-aarch64-linux-gnu, g++-aarch64-linux-gnu)
CROSS_COMPILE="${CROSS_COMPILE:-aarch64-linux-gnu-}"

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

# Board/ddr -> build-dir suffix. Shared with build_runtime.sh and
# firmware/c7x/dsp/build.sh so all three always agree on where a given
# board's artifacts live; cmake/boards.cmake remains the sole source of
# truth for what actually gets built.
source "${SCRIPT_DIR}/../../../board_build_dir.sh"
resolve_board_build_dir
BUILD_DIR="${SCRIPT_DIR}/build${BUILD_SUFFIX}"

# Plain string, not an array: values are always simple enum tokens (no
# spaces), and an empty array expanded under `set -u` is an unbound-variable
# error on bash < 4.4.
CMAKE_BOARD_ARGS=""
[ -n "$TVM_BOARD" ] && CMAKE_BOARD_ARGS="$CMAKE_BOARD_ARGS -DTVM_BOARD=$TVM_BOARD"
[ -n "$TVM_DDR" ] && CMAKE_BOARD_ARGS="$CMAKE_BOARD_ARGS -DTVM_DDR=$TVM_DDR"

case "$SUBCOMMAND" in
    clean)
        echo "Cleaning build directory..."
        rm -rf "${BUILD_DIR}"
        echo "Done."
        ;;
    native)
        echo "Building natively (must run on ARM64 target)..."
        mkdir -p "${BUILD_DIR}"
        cd "${BUILD_DIR}"
        cmake $CMAKE_BOARD_ARGS ..
        make ${VERBOSE:+VERBOSE=1}
        echo ""
        echo "Build complete:"
        echo "  ${BUILD_DIR}/c7x_compute"
        echo "  ${BUILD_DIR}/libc7x_arm_runtime.so"
        if [ -f "${BUILD_DIR}/test_c7x_runtime" ]; then
            echo "  ${BUILD_DIR}/test_c7x_runtime"
        fi
        ;;
    deploy)
        if [ ! -f "${BUILD_DIR}/c7x_compute" ] || [ ! -f "${BUILD_DIR}/libc7x_arm_runtime.so" ]; then
            echo "Error: Build artifacts not found. Run './build.sh' first."
            exit 1
        fi
        echo "Deploying to ${TARGET_HOST}..."
        scp "${BUILD_DIR}/c7x_compute" "${TARGET_HOST}:/usr/local/bin/"
        scp "${BUILD_DIR}/libc7x_arm_runtime.so" "${TARGET_HOST}:/usr/local/lib/"
        scp "${SCRIPT_DIR}/include/c7x_runtime.h" "${TARGET_HOST}:/usr/local/include/"
        # Deploy test binary if it was built
        if [ -f "${BUILD_DIR}/test_c7x_runtime" ]; then
            scp "${BUILD_DIR}/test_c7x_runtime" "${TARGET_HOST}:/usr/local/bin/"
            echo "Deployed test_c7x_runtime"
        fi
        # Ensure /usr/local/lib is in the dynamic linker search path and create
        # the SONAME symlink (.so.1) that the binary's DT_NEEDED entry requires.
        ssh "${TARGET_HOST}" \
            "mkdir -p /etc/ld.so.conf.d && \
             echo /usr/local/lib > /etc/ld.so.conf.d/c7x_arm_runtime.conf && \
             ln -sf /usr/local/lib/libc7x_arm_runtime.so \
                    /usr/local/lib/libc7x_arm_runtime.so.1 && \
             ldconfig"
        echo "Deployed to ${TARGET_HOST}:"
        echo "  /usr/local/bin/c7x_compute"
        echo "  /usr/local/lib/libc7x_arm_runtime.so"
        echo "  /usr/local/include/c7x_runtime.h"
        ;;
    *)
        echo "Cross-compiling for ARM64 (board=$BOARD ddr=$DDR)..."
        mkdir -p "${BUILD_DIR}"
        cd "${BUILD_DIR}"
        cmake -DCMAKE_CXX_COMPILER="${CROSS_COMPILE}g++" \
              $CMAKE_BOARD_ARGS \
              ..
        make ${VERBOSE:+VERBOSE=1}
        echo ""
        echo "Build complete:"
        echo "  ${BUILD_DIR}/c7x_compute"
        echo "  ${BUILD_DIR}/libc7x_arm_runtime.so"
        if [ -f "${BUILD_DIR}/test_c7x_runtime" ]; then
            echo "  ${BUILD_DIR}/test_c7x_runtime"
        fi
        echo ""
        echo "Next steps:"
        echo "  ./build.sh deploy  - Deploy to target (${TARGET_HOST})"
        echo ""
        echo "Or copy manually:"
        echo "  scp ${BUILD_DIR}/c7x_compute ${TARGET_HOST}:/usr/local/bin/"
        echo "  scp ${BUILD_DIR}/libc7x_arm_runtime.so ${TARGET_HOST}:/usr/local/lib/"
        echo "  scp ${SCRIPT_DIR}/include/c7x_runtime.h ${TARGET_HOST}:/usr/local/include/"
        echo "  ssh ${TARGET_HOST} ldconfig"
        ;;
esac
