#!/bin/bash
# Build TVM DSP runtime libraries.
#
# Usage:
#   ./build_runtime.sh                  # Build all targets (c7x + c7x_host)
#   ./build_runtime.sh c66x_host        # Build C66x host emulation only
#   ./build_runtime.sh c66x             # Build C66x cross-compiled only
#   ./build_runtime.sh c7x              # Build C7x cross-compiled only
#   ./build_runtime.sh c7x_host         # Build C7x host emulation only
#   ./build_runtime.sh clean            # Remove all build directories
#
#   --board <j722s-evm|beagley-ai>      # Target board (required for c7x)
#   --ddr <4gb|8gb>                     # Shared-DMA DDR size (default: per-board)
#
# Board/ddr resolve to SDK paths and the shared-DMA physical base entirely
# in cmake/boards.cmake -- this script only forwards the flags and picks a
# build-dir name so switching --ddr never reuses a stale build. --board is
# only required for the c7x (hardware) target -- c66x_host/c66x/c7x_host
# emulation targets don't depend on board identity at all.
#
# Environment variables (auto-detected if not set):
#   TI_CGT_C6000_PATH   - TI C6000 compiler (required for c66x target)
#   TI_CGT_C7000_PATH   - TI C7000 compiler (required for c7x targets)
#   MCU_PLUS_SDK_PATH    - MCU+ SDK for J722S (required for c7x DMA; default
#                          is board-dependent, see cmake/boards.cmake)

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
cd "$SCRIPT_DIR"

TARGET="all"
TVM_BOARD=""
TVM_DDR=""
while [ $# -gt 0 ]; do
    case "$1" in
        --board) TVM_BOARD="$2"; shift 2 ;;
        --board=*) TVM_BOARD="${1#*=}"; shift ;;
        --ddr) TVM_DDR="$2"; shift 2 ;;
        --ddr=*) TVM_DDR="${1#*=}"; shift ;;
        *) TARGET="$1"; shift ;;
    esac
done

# --- Board/ddr -> build-dir suffix ---
# Shared with firmware/c7x/dsp/build.sh and firmware/c7x/arm/build.sh so
# all three always agree on where a given board's artifacts live;
# cmake/boards.cmake remains the sole source of truth for what is built.
source "${SCRIPT_DIR}/board_build_dir.sh"
resolve_board_build_dir

# Plain string, not an array: values are always simple enum tokens (no
# spaces), and an empty array expanded under `set -u` is an unbound-variable
# error on bash < 4.4.
CMAKE_BOARD_ARGS=""
[ -n "$TVM_BOARD" ] && CMAKE_BOARD_ARGS="$CMAKE_BOARD_ARGS -DTVM_BOARD=$TVM_BOARD"
[ -n "$TVM_DDR" ] && CMAKE_BOARD_ARGS="$CMAKE_BOARD_ARGS -DTVM_DDR=$TVM_DDR"

# --- Auto-detect TI C6000 compiler ---
if [ -z "${TI_CGT_C6000_PATH:-}" ]; then
    for p in \
        "$HOME/ti/ccs2041/ccs/tools/compiler/ti-cgt-c6000_8.5.0.LTS" \
        "$HOME/ti/ccs2050/ccs/tools/compiler/ti-cgt-c6000_8.5.0.LTS" \
        "$HOME/ti/ccs2040/ccs/tools/compiler/ti-cgt-c6000_8.5.0.LTS" \
        "$HOME/ti/ti-cgt-c6000_8.5.0.LTS"; do
        if [ -x "$p/bin/cl6x" ]; then
            export TI_CGT_C6000_PATH="$p"
            break
        fi
    done
fi

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

# --- Validate required compilers for chosen target ---
if [[ "$TARGET" == "c66x" || "$TARGET" == "all" ]]; then
    if [ -z "${TI_CGT_C6000_PATH:-}" ]; then
        echo "ERROR: TI_CGT_C6000_PATH not set and compiler not found"
        echo "  Set TI_CGT_C6000_PATH to the TI C6000 CGT installation path"
        exit 1
    fi
    echo "TI C6000 compiler: $TI_CGT_C6000_PATH"
fi

if [[ "$TARGET" == "c7x" || "$TARGET" == "c7x_host" || "$TARGET" == "all" ]]; then
    if [ -z "${TI_CGT_C7000_PATH:-}" ]; then
        echo "ERROR: TI_CGT_C7000_PATH not set and compiler not found"
        echo "  Set TI_CGT_C7000_PATH to the TI C7000 CGT installation path"
        exit 1
    fi
    echo "TI C7000 compiler: $TI_CGT_C7000_PATH"
fi

# MCU+ SDK path: resolved by cmake/boards.cmake (board default, or
# MCU_PLUS_SDK_PATH env var override) -- not duplicated here.

# --- Build functions ---
build_c66x_host() {
    echo ""
    echo "=== Building C66x host emulation runtime ==="
    rm -rf "build-c66x-host${BUILD_SUFFIX}"
    mkdir "build-c66x-host${BUILD_SUFFIX}" && cd "build-c66x-host${BUILD_SUFFIX}"
    cmake $CMAKE_BOARD_ARGS ..
    cmake --build .
    echo "Output: $SCRIPT_DIR/build-c66x-host${BUILD_SUFFIX}/libtvm_dsp_runtime_host.a"
    cd "$SCRIPT_DIR"
}

build_c66x() {
    echo ""
    echo "=== Building C66x cross-compiled runtime ==="
    rm -rf "build-c66x${BUILD_SUFFIX}"
    mkdir "build-c66x${BUILD_SUFFIX}" && cd "build-c66x${BUILD_SUFFIX}"
    cmake -DCMAKE_TOOLCHAIN_FILE=../cmake/toolchain-awrl6844.cmake \
          -DTVM_DSP_DEVICE=awrl6844 \
          $CMAKE_BOARD_ARGS \
          ..
    cmake --build .
    echo "Output: $SCRIPT_DIR/build-c66x${BUILD_SUFFIX}/libtvm_dsp_runtime_c66x.a"
    cd "$SCRIPT_DIR"
}

build_c7x() {
    if [ -z "$TVM_BOARD" ]; then
        echo "Error: --board <j722s-evm|beagley-ai> is required for the c7x target" >&2
        exit 1
    fi
    echo ""
    echo "=== Building C7x cross-compiled runtime (board=$BOARD ddr=$DDR) ==="
    rm -rf "build-c7x${BUILD_SUFFIX}"
    mkdir "build-c7x${BUILD_SUFFIX}" && cd "build-c7x${BUILD_SUFFIX}"
    cmake -DCMAKE_TOOLCHAIN_FILE=../cmake/toolchain-j722s-c7x.cmake \
          $CMAKE_BOARD_ARGS \
          ..
    cmake --build .
    echo "Output: $SCRIPT_DIR/build-c7x${BUILD_SUFFIX}/libtvm_dsp_runtime_c7x.a"
    cd "$SCRIPT_DIR"
}

build_c7x_host() {
    echo ""
    echo "=== Building C7x host emulation runtime ==="
    rm -rf "build-c7x-host${BUILD_SUFFIX}"
    mkdir "build-c7x-host${BUILD_SUFFIX}" && cd "build-c7x-host${BUILD_SUFFIX}"
    cmake -DTVM_DSP_TARGET=c7x_host $CMAKE_BOARD_ARGS ..
    cmake --build .
    echo "Output: $SCRIPT_DIR/build-c7x-host${BUILD_SUFFIX}/libtvm_dsp_runtime_c7x_host.a"
    cd "$SCRIPT_DIR"
}

do_clean() {
    echo "Cleaning build directories..."
    rm -rf "build-c66x-host${BUILD_SUFFIX}" "build-c66x${BUILD_SUFFIX}" \
           "build-c7x${BUILD_SUFFIX}" "build-c7x-host${BUILD_SUFFIX}"
    echo "Done"
}

# --- Main ---
case "$TARGET" in
    c66x_host) build_c66x_host ;;
    c66x)      build_c66x ;;
    c7x)       build_c7x ;;
    c7x_host)  build_c7x_host ;;
    clean)     do_clean ;;
    all)       build_c66x; build_c7x; build_c7x_host ;;
    *)
        echo "Usage: $0 [c66x_host|c66x|c7x|c7x_host|clean|all] --board <j722s-evm|beagley-ai> [--ddr <4gb|8gb>]"
        echo "  (--board is required only for the c7x target)"
        exit 1
        ;;
esac

echo ""
echo "Build complete."
