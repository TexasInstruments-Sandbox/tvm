#!/bin/bash
# Build TVM TI C7x DSP wheels from pre-built artifacts.
#
# Two wheel variants:
#   x86   (default)  tvm-ti-c7x-compile   — full compilation toolchain
#   arm64            tvm-ti-c7x-inference  — minimal on-board inference runtime
#
# Prerequisites:
#   - TVM built (cmake + ninja) with libtvm.so in $TVM_HOME/build/
#   - DSP runtime built: build_runtime.sh c7x_host && build_runtime.sh c7x
#   - Firmware built: firmware/c7x/dsp/build.sh && firmware/c7x/arm/build.sh
#   - TIDL .so built (x86 wheel only, and only when --tidl ON, the default --
#     matches firmware/c7x/dsp/build.sh's own --tidl convention. beagley-ai
#     firmware is always built --tidl OFF, so its x86 wheel needs --tidl OFF
#     here too; no TIDL bridge/source files get bundled in that case)
#   - Python packages: pip install build
#
# Usage:
#   bash packaging/ti_dsp/build_wheel.sh                       # x86 compile wheel
#   bash packaging/ti_dsp/build_wheel.sh --target arm64         # aarch64 inference wheel
#   bash packaging/ti_dsp/build_wheel.sh --tidl OFF              # x86 wheel, no TIDL
#   bash packaging/ti_dsp/build_wheel.sh --target arm64 \
#        --board beagley-ai --tidl OFF --mmalib ON              # beagley-ai inference wheel
#
# --board/--ddr (default: j722s-evm/8gb, same as the underlying build
# scripts) select which board's pre-built artifacts get packaged -- see
# board_build_dir.sh for the build-dir naming this mirrors. beagley-ai
# firmware is always built --tidl OFF --mmalib ON (see
# firmware/c7x/dsp/build.sh), so pass the same two flags here too when
# packaging for that board, or the DSP firmware check_file below looks in
# the wrong build-<board>-<ddr>[-tidl-*-mmalib-*] dir.
#
# Environment variables:
#   TVM_HOME           - TVM repo root (default: script's grandparent dir)
#   DSP_BUILD_NUM      - Post-release build number (default: 1)
#   TIDL_RELAX_SO      - Path to tidl_model_import_relax.so
#   C7X_MMA_TIDL_PATH  - c7x-mma-tidl repo root (fallback for TIDL .so)
#   STRIP_LIBS         - Strip debug symbols from .so files (default: 1)

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
TVM_HOME="${TVM_HOME:-$(cd "$SCRIPT_DIR/../.." && pwd)}"
DSP_BUILD_NUM="${DSP_BUILD_NUM:-1}"
STRIP_LIBS="${STRIP_LIBS:-1}"
DSP_RT="$TVM_HOME/src/runtime/ti_dsp"

# --- Parse arguments ---
TARGET="x86"
TIDL="ON"
TVM_BOARD=""
TVM_DDR=""
TVM_MMALIB=""
while [ $# -gt 0 ]; do
    case "$1" in
        --target) TARGET="$2"; shift 2 ;;
        --tidl) TIDL="$2"; shift 2 ;;
        --board) TVM_BOARD="$2"; shift 2 ;;
        --ddr) TVM_DDR="$2"; shift 2 ;;
        --mmalib) TVM_MMALIB="$2"; shift 2 ;;
        *) echo "Unknown argument: $1"; exit 1 ;;
    esac
done

if [[ "$TARGET" != "x86" && "$TARGET" != "arm64" ]]; then
    echo "ERROR: --target must be x86 or arm64 (got: $TARGET)"
    exit 1
fi

if [[ "$TIDL" != "ON" && "$TIDL" != "OFF" ]]; then
    echo "ERROR: --tidl must be ON or OFF (got: $TIDL)"
    exit 1
fi

if [[ -n "$TVM_BOARD" && "$TVM_BOARD" != "j722s-evm" && "$TVM_BOARD" != "beagley-ai" ]]; then
    echo "ERROR: --board must be j722s-evm or beagley-ai (got: $TVM_BOARD)"
    exit 1
fi

# Board/ddr -> build-dir suffix, shared with build_runtime.sh and the
# firmware/c7x/{dsp,arm} build scripts so this always agrees on where a
# given board's artifacts live. RUNTIME_SUFFIX (board+ddr only) matches
# the DSP runtime lib and ARM client build dirs; FW_SUFFIX additionally
# folds in --tidl/mmalib, matching the DSP firmware build dir -- the two
# differ whenever --tidl is OFF.
source "$DSP_RT/board_build_dir.sh"
resolve_board_build_dir
RUNTIME_SUFFIX="$BUILD_SUFFIX"
BOARD_DDR_DESC="board=$BOARD ddr=$DDR"
TVM_TIDL="$TIDL"
resolve_board_build_dir
FW_SUFFIX="$BUILD_SUFFIX"

# Use separate staging dirs so both can coexist
STAGING="$SCRIPT_DIR/staging-${TARGET}"

check_file() {
    if [ ! -f "$1" ]; then
        echo "ERROR: Required file not found: $1"
        echo "  $2"
        exit 1
    fi
}

get_version() {
    python3 -c "
import re, pathlib
text = pathlib.Path('$TVM_HOME/python/tvm/libinfo.py').read_text()
m = re.search(r'__version__\s*=\s*\"([^\"]+)\"', text)
print(m.group(1))
"
}

# =====================================================================
# x86 compile wheel
# =====================================================================
build_x86() {
    echo "=== tvm-ti-c7x-compile wheel (x86) ==="
    echo "  TVM_HOME:      $TVM_HOME"
    echo "  DSP_BUILD_NUM: $DSP_BUILD_NUM"
    echo "  TIDL:          $TIDL"
    echo "  $BOARD_DDR_DESC"
    echo ""

    # --- Validate ---
    check_file "$TVM_HOME/build/libtvm.so" \
        "Build TVM first: cd build && cmake -G Ninja .. && ninja"
    check_file "$DSP_RT/build-c7x-host${RUNTIME_SUFFIX}/libtvm_dsp_runtime_c7x_host.a" \
        "Build DSP runtime: bash build_runtime.sh c7x_host --board $BOARD --ddr $DDR"
    check_file "$DSP_RT/build-c7x${RUNTIME_SUFFIX}/libtvm_dsp_runtime_c7x.a" \
        "Build DSP runtime: bash build_runtime.sh c7x --board $BOARD --ddr $DDR"
    check_file "$DSP_RT/firmware/c7x/dsp/build${FW_SUFFIX}/c7x_compute.out" \
        "Build firmware: cd firmware/c7x/dsp && ./build.sh --board $BOARD --ddr $DDR --tidl $TIDL"
    check_file "$DSP_RT/firmware/c7x/arm/build${RUNTIME_SUFFIX}/c7x_compute" \
        "Build ARM client: cd firmware/c7x/arm && ./build.sh --board $BOARD --ddr $DDR"

    TIDL_SO=""
    if [ "$TIDL" = "ON" ]; then
        TIDL_SO="${TIDL_RELAX_SO:-}"
        if [ -z "$TIDL_SO" ]; then
            C7X_TIDL="${C7X_MMA_TIDL_PATH:-$HOME/ml/c7x-mma-tidl}"
            TIDL_SO="$C7X_TIDL/ti_dl/utils/tidlModelImport/out/tidl_model_import_relax.so"
        fi
        check_file "$TIDL_SO" \
            "Build TIDL .so or set TIDL_RELAX_SO / C7X_MMA_TIDL_PATH (or pass --tidl OFF for a beagley-ai-style no-TIDL wheel)"
    fi

    # --- Clean ---
    rm -rf "$STAGING"
    mkdir -p "$STAGING"

    # --- TVM Python source ---
    echo ">>> Copying TVM Python source ..."
    cp -a "$TVM_HOME/python/tvm" "$STAGING/tvm"
    find "$STAGING/tvm" -type d -name __pycache__ -exec rm -rf {} + 2>/dev/null || true
    find "$STAGING/tvm" -name "*.pyc" -delete 2>/dev/null || true

    # --- Native libraries ---
    echo ">>> Copying native libraries ..."
    if [ "$STRIP_LIBS" = "1" ]; then
        strip --strip-debug -o "$STAGING/tvm/libtvm.so" "$TVM_HOME/build/libtvm.so"
        strip --strip-debug -o "$STAGING/tvm/libtvm_runtime.so" "$TVM_HOME/build/libtvm_runtime.so"
    else
        cp "$TVM_HOME/build/libtvm.so" "$STAGING/tvm/"
        cp "$TVM_HOME/build/libtvm_runtime.so" "$STAGING/tvm/"
    fi
    if [ -f "$TVM_HOME/python/tvm/ffi/core.abi3.so" ]; then
        cp "$TVM_HOME/python/tvm/ffi/core.abi3.so" "$STAGING/tvm/ffi/"
    fi

    # --- DSP data directory ---
    echo ">>> Staging DSP artifacts ..."
    DATA="$STAGING/tvm/data/ti_dsp"
    mkdir -p "$DATA/firmware" "$DATA/lib" "$DATA/tidl" \
             "$DATA/cmake" "$DATA/dynmod" "$DATA/include/dlpack"
    touch "$STAGING/tvm/data/__init__.py"
    touch "$DATA/__init__.py"

    # DSP runtime libraries
    cp "$DSP_RT/build-c7x-host${RUNTIME_SUFFIX}/libtvm_dsp_runtime_c7x_host.a" "$DATA/lib/"
    cp "$DSP_RT/build-c7x${RUNTIME_SUFFIX}/libtvm_dsp_runtime_c7x.a" "$DATA/lib/"

    # Firmware
    cp "$DSP_RT/firmware/c7x/dsp/build${FW_SUFFIX}/c7x_compute.out" "$DATA/firmware/"
    cp "$DSP_RT/firmware/c7x/arm/build${RUNTIME_SUFFIX}/c7x_compute" "$DATA/firmware/"
    chmod +x "$DATA/firmware/c7x_compute"
    ARM_SO="$DSP_RT/firmware/c7x/arm/build${RUNTIME_SUFFIX}"
    if [ -f "$ARM_SO/libc7x_arm_runtime.so.1" ]; then
        cp "$ARM_SO/libc7x_arm_runtime.so.1" "$DATA/firmware/libc7x_arm_runtime.so.1"
        cp "$ARM_SO/libc7x_arm_runtime.so.1" "$DATA/firmware/libc7x_arm_runtime.so"
    elif [ -f "$ARM_SO/libc7x_arm_runtime.so" ]; then
        cp "$ARM_SO/libc7x_arm_runtime.so" "$DATA/firmware/libc7x_arm_runtime.so"
    fi

    # TIDL — copy both the TVM bridge and its runtime dependency so the
    # dynamic linker can find tidl_model_custom_import.so when loading
    # tidl_model_import_relax.so from the installed wheel path. Skipped
    # entirely for --tidl OFF (e.g. beagley-ai, which never builds TIDL).
    if [ "$TIDL" = "ON" ]; then
        cp "$TIDL_SO" "$DATA/tidl/"
        TIDL_CUSTOM="$(dirname "$TIDL_SO")/tidl_model_custom_import.so"
        if [ -f "$TIDL_CUSTOM" ]; then
            cp "$TIDL_CUSTOM" "$DATA/tidl/"
        fi
    fi
    # TIDL API C sources compiled into lib0.out when USE_TIDL=ON.
    # dynmod/CMakeLists.txt references ${DSP_RUNTIME_DIR}/tidl/*.c directly.
    cp "$DSP_RT/tidl/"*.c "$DATA/tidl/"

    # Build infrastructure
    cp "$DSP_RT/cmake/toolchain-j722s-c7x.cmake" "$DATA/cmake/"
    cp -a "$DSP_RT/dynmod/." "$DATA/dynmod/"
    rm -rf "$DATA/dynmod/build-dynmod" "$DATA/dynmod/build" 2>/dev/null || true
    mkdir -p "$DATA/scripts"
    cp "$DSP_RT/scripts/bin_to_asm.py" "$DATA/scripts/"
    DLPACK="$TVM_HOME/3rdparty/tvm-ffi/3rdparty/dlpack/include/dlpack"
    if [ -d "$DLPACK" ]; then
        cp "$DLPACK"/*.h "$DATA/include/dlpack/"
    fi
    if [ -d "$DSP_RT/include" ]; then
        cp -a "$DSP_RT/include/." "$DATA/include/"
    fi
    # Copy all DSP runtime headers needed by generated lib0.c at compile time.
    # The dynmod CMakeLists adds -I${DSP_RUNTIME_DIR} so every header
    # subdirectory reachable via #include "subdir/header.h" must be present.
    # Firmware internals (dload, compute_service) are excluded — they are only
    # needed to build the firmware binary, not to compile generated code.
    find "$DSP_RT" -name "*.h" \
        ! -path "*/build*" \
        ! -path "*/firmware/c7x/dsp/src/*" \
        ! -path "*/firmware/c7x/arm/*" \
        | while read -r f; do
            rel="${f#$DSP_RT/}"
            mkdir -p "$DATA/$(dirname "$rel")"
            cp "$f" "$DATA/$rel"
        done

    # --- pyproject.toml + version ---
    cp "$SCRIPT_DIR/pyproject.toml" "$STAGING/"
    FULL_VERSION="$(get_version).post${DSP_BUILD_NUM}"
    sed -i "s/^version = .*/version = \"${FULL_VERSION}\"/" "$STAGING/pyproject.toml"
    echo "  Version: $FULL_VERSION"

    # --- Build wheel ---
    PLAT_TAG="linux_$(uname -m)"
    echo ">>> Building wheel (platform: $PLAT_TAG) ..."
    cd "$STAGING"
    python -m build --wheel --no-isolation \
        -C--build-option=--plat-name="$PLAT_TAG" 2>&1 | tail -5
}

# =====================================================================
# aarch64 inference wheel
# =====================================================================
build_arm64() {
    echo "=== tvm-ti-c7x-inference wheel (aarch64) ==="
    echo "  TVM_HOME:      $TVM_HOME"
    echo "  DSP_BUILD_NUM: $DSP_BUILD_NUM"
    echo "  $BOARD_DDR_DESC"
    echo ""

    # --- Validate ---
    check_file "$DSP_RT/firmware/c7x/dsp/build${FW_SUFFIX}/c7x_compute.out" \
        "Build firmware: cd firmware/c7x/dsp && ./build.sh --board $BOARD --ddr $DDR --tidl $TIDL"
    check_file "$DSP_RT/firmware/c7x/arm/build${RUNTIME_SUFFIX}/c7x_compute" \
        "Build ARM client: cd firmware/c7x/arm && ./build.sh --board $BOARD --ddr $DDR"

    # --- Clean ---
    rm -rf "$STAGING"
    mkdir -p "$STAGING"

    # --- Minimal Python package ---
    echo ">>> Creating minimal tvm package ..."
    mkdir -p "$STAGING/tvm/contrib/c7x"
    mkdir -p "$STAGING/tvm/data/ti_dsp"

    # Stub tvm/__init__.py — does not load libtvm.so
    cat > "$STAGING/tvm/__init__.py" << 'PYEOF'
"""TVM C7x inference runtime (minimal aarch64 package)."""
PYEOF

    # Empty package markers
    touch "$STAGING/tvm/contrib/__init__.py"
    cp "$TVM_HOME/python/tvm/contrib/c7x/__init__.py" "$STAGING/tvm/contrib/c7x/"
    cp "$TVM_HOME/python/tvm/contrib/c7x/c7x_runtime.py" "$STAGING/tvm/contrib/c7x/"
    touch "$STAGING/tvm/data/__init__.py"
    cp "$TVM_HOME/python/tvm/data/ti_dsp/__init__.py" "$STAGING/tvm/data/ti_dsp/"
    cp "$TVM_HOME/python/tvm/data/ti_dsp/paths.py" "$STAGING/tvm/data/ti_dsp/"

    # --- Firmware and ARM runtime ---
    echo ">>> Staging firmware artifacts ..."
    DATA="$STAGING/tvm/data/ti_dsp"
    mkdir -p "$DATA/firmware"

    cp "$DSP_RT/firmware/c7x/dsp/build${FW_SUFFIX}/c7x_compute.out" "$DATA/firmware/"
    cp "$DSP_RT/firmware/c7x/arm/build${RUNTIME_SUFFIX}/c7x_compute" "$DATA/firmware/"
    chmod +x "$DATA/firmware/c7x_compute"

    ARM_SO="$DSP_RT/firmware/c7x/arm/build${RUNTIME_SUFFIX}"
    if [ -f "$ARM_SO/libc7x_arm_runtime.so.1" ]; then
        cp "$ARM_SO/libc7x_arm_runtime.so.1" "$DATA/firmware/libc7x_arm_runtime.so.1"
        cp "$ARM_SO/libc7x_arm_runtime.so.1" "$DATA/firmware/libc7x_arm_runtime.so"
    elif [ -f "$ARM_SO/libc7x_arm_runtime.so" ]; then
        cp "$ARM_SO/libc7x_arm_runtime.so" "$DATA/firmware/libc7x_arm_runtime.so"
    fi

    # Public header for the C++ API (c7x::Module in c7x_runtime.h), plus its
    # only dependency (DLPack), so a C++ consumer can build against the wheel
    # with a single -I flag without checking out the TVM source tree.
    echo ">>> Staging C++ API header ..."
    mkdir -p "$DATA/include/dlpack"
    cp "$DSP_RT/firmware/c7x/arm/include/c7x_runtime.h" "$DATA/include/"
    DLPACK="$TVM_HOME/3rdparty/tvm-ffi/3rdparty/dlpack/include/dlpack"
    if [ -d "$DLPACK" ]; then
        cp "$DLPACK"/*.h "$DATA/include/dlpack/"
    fi

    # --- pyproject.toml + version ---
    cp "$SCRIPT_DIR/pyproject_arm64.toml" "$STAGING/pyproject.toml"
    FULL_VERSION="$(get_version).post${DSP_BUILD_NUM}"
    sed -i "s/^version = .*/version = \"${FULL_VERSION}\"/" "$STAGING/pyproject.toml"
    echo "  Version: $FULL_VERSION"

    # --- Build wheel ---
    echo ">>> Building wheel (platform: linux_aarch64) ..."
    cd "$STAGING"
    python -m build --wheel --no-isolation \
        -C--build-option=--plat-name=linux_aarch64 2>&1 | tail -5
}

# =====================================================================
# Main
# =====================================================================
case "$TARGET" in
    x86)   build_x86 ;;
    arm64) build_arm64 ;;
esac

echo ""
echo "=== Wheel built ==="
ls -lh "$STAGING/dist/"*.whl
