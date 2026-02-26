#!/bin/bash
# Build TVM DSP runtime libraries for all C7x targets.
#
# Usage:
#   ./build_runtime.sh              # Build all targets (c7x + c7x_host)
#   ./build_runtime.sh c7x          # Build C7x cross-compiled only
#   ./build_runtime.sh c7x_host     # Build C7x host emulation only
#   ./build_runtime.sh clean        # Remove all build directories
#
# Environment variables (auto-detected if not set):
#   TI_CGT_C7000_PATH   - TI C7000 compiler (required for both targets)
#   MCU_PLUS_SDK_PATH    - MCU+ SDK for J722S (required for c7x DMA)

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
cd "$SCRIPT_DIR"

# --- Auto-detect TI C7000 compiler ---
if [ -z "${TI_CGT_C7000_PATH:-}" ]; then
    for p in \
        "$HOME/ti/ccs2041/ccs/tools/compiler/ti-cgt-c7000_5.0.1.LTS" \
        "$HOME/ti/ccs2040/ccs/tools/compiler/ti-cgt-c7000_5.0.1.LTS" \
        "$HOME/ti/ti-cgt-c7000_5.0.1.LTS"; do
        if [ -x "$p/bin/cl7x" ]; then
            export TI_CGT_C7000_PATH="$p"
            break
        fi
    done
fi
if [ -z "${TI_CGT_C7000_PATH:-}" ]; then
    echo "ERROR: TI_CGT_C7000_PATH not set and compiler not found"
    echo "  Set TI_CGT_C7000_PATH to the TI C7000 CGT installation path"
    exit 1
fi
echo "TI C7000 compiler: $TI_CGT_C7000_PATH"

# --- Auto-detect MCU+ SDK ---
if [ -z "${MCU_PLUS_SDK_PATH:-}" ]; then
    for p in \
        "$HOME/ml/am67a/ti-processor-sdk-rtos-j722s-evm-11_00_00_06/mcu_plus_sdk_j722s_11_00_00_12" \
        "$HOME/ti/mcu_plus_sdk_j722s_11_01_00_07" \
        "$HOME/ti/MCU_PLUS_SDK_J722S_11_01"; do
        if [ -d "$p/source/drivers" ]; then
            export MCU_PLUS_SDK_PATH="$p"
            break
        fi
    done
fi
if [ -n "${MCU_PLUS_SDK_PATH:-}" ]; then
    echo "MCU+ SDK: $MCU_PLUS_SDK_PATH"
else
    echo "MCU+ SDK: not found (c7x DMA will build without SDK drivers)"
fi

# --- Build functions ---
build_c7x() {
    echo ""
    echo "=== Building C7x cross-compiled runtime ==="
    rm -rf build-c7x
    mkdir build-c7x && cd build-c7x
    cmake -DCMAKE_TOOLCHAIN_FILE=../cmake/toolchain-j722s-c7x.cmake \
          ${MCU_PLUS_SDK_PATH:+-DMCU_PLUS_SDK_PATH="$MCU_PLUS_SDK_PATH"} \
          ..
    cmake --build .
    echo "Output: $SCRIPT_DIR/build-c7x/libtvm_dsp_runtime_c7x.a"
    cd "$SCRIPT_DIR"
}

build_c7x_host() {
    echo ""
    echo "=== Building C7x host emulation runtime ==="
    rm -rf build-c7x-host
    mkdir build-c7x-host && cd build-c7x-host
    cmake -DTVM_DSP_TARGET=c7x_host ..
    cmake --build .
    echo "Output: $SCRIPT_DIR/build-c7x-host/libtvm_dsp_runtime_c7x_host.a"
    cd "$SCRIPT_DIR"
}

do_clean() {
    echo "Cleaning build directories..."
    rm -rf build-c7x build-c7x-host
    echo "Done"
}

# --- Main ---
TARGET="${1:-all}"
case "$TARGET" in
    c7x)       build_c7x ;;
    c7x_host)  build_c7x_host ;;
    clean)     do_clean ;;
    all)       build_c7x; build_c7x_host ;;
    *)
        echo "Usage: $0 [c7x|c7x_host|clean|all]"
        exit 1
        ;;
esac

echo ""
echo "Build complete."
