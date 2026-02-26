#!/bin/bash
#
# Build script for C7x Compute Service - Host Application
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
        make VERBOSE=1
        echo ""
        echo "Build complete: ${BUILD_DIR}/c7x_compute"
        ;;
    deploy)
        if [ ! -f "${BUILD_DIR}/c7x_compute" ]; then
            echo "Error: Binary not found. Run './build.sh' first."
            exit 1
        fi
        echo "Deploying to ${TARGET_HOST}..."
        scp "${BUILD_DIR}/c7x_compute" "${TARGET_HOST}:/usr/local/bin/"
        echo "Deployed to ${TARGET_HOST}:/usr/local/bin/c7x_compute"
        ;;
    *)
        echo "Cross-compiling for ARM64..."
        mkdir -p "${BUILD_DIR}"
        cd "${BUILD_DIR}"
        cmake -DCMAKE_CXX_COMPILER="${CROSS_COMPILE}g++" \
              ..
        make VERBOSE=1
        echo ""
        echo "Build complete: ${BUILD_DIR}/c7x_compute"
        echo ""
        echo "Next steps:"
        echo "  ./build.sh deploy  - Deploy to target"
        echo ""
        echo "Or copy manually:"
        echo "  scp ${BUILD_DIR}/c7x_compute ${TARGET_HOST}:/usr/local/bin/"
        ;;
esac
