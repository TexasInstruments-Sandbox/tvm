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

set -e

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
BUILD_DIR="${SCRIPT_DIR}/build"
TARGET_HOST="${AM67A_TARGET:-am67a}"

# ARM64 cross-compiler (Ubuntu packages: gcc-aarch64-linux-gnu, g++-aarch64-linux-gnu)
CROSS_COMPILE="${CROSS_COMPILE:-aarch64-linux-gnu-}"

case "${1:-}" in
    clean)
        echo "Cleaning build directory..."
        rm -rf "${BUILD_DIR}"
        echo "Done."
        ;;
    native)
        echo "Building natively (must run on ARM64 target)..."
        mkdir -p "${BUILD_DIR}"
        cd "${BUILD_DIR}"
        cmake ..
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
        ssh "${TARGET_HOST}" ldconfig
        echo "Deployed to ${TARGET_HOST}:"
        echo "  /usr/local/bin/c7x_compute"
        echo "  /usr/local/lib/libc7x_arm_runtime.so"
        echo "  /usr/local/include/c7x_runtime.h"
        ;;
    *)
        echo "Cross-compiling for ARM64..."
        mkdir -p "${BUILD_DIR}"
        cd "${BUILD_DIR}"
        cmake -DCMAKE_CXX_COMPILER="${CROSS_COMPILE}g++" \
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
