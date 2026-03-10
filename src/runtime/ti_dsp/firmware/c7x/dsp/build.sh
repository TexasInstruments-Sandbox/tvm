#!/bin/bash
#
# Build script for C7x Compute Service DSP Firmware
#
# Usage:
#   ./build.sh          - Build firmware
#   ./build.sh clean    - Clean build directory
#   ./build.sh deploy   - Deploy and start firmware
#

set -e

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
BUILD_DIR="${SCRIPT_DIR}/build"
FIRMWARE_NAME="c7x_compute.out"

# Use the common deploy script
DEPLOY_SCRIPT="${SCRIPT_DIR}/../../deploy-c7x.sh"

case "${1:-}" in
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
        "${DEPLOY_SCRIPT}" "${BUILD_DIR}/${FIRMWARE_NAME}" --trace
        ;;
    *)
        echo "Building C7x Compute Service firmware..."
        echo ""
        mkdir -p "${BUILD_DIR}"
        cd "${BUILD_DIR}"
        cmake -DUSE_TIDL_RUNTIME=ON ..
        make VERBOSE=1
        echo ""
        echo "Build complete: ${BUILD_DIR}/${FIRMWARE_NAME}"
        echo ""
        echo "Next steps:"
        echo "  ./build.sh deploy  - Deploy to target and show trace"
        ;;
esac
