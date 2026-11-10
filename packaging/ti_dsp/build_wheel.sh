#!/bin/bash
# Build TVM TI C7x DSP wheels from pre-built artifacts.
#
# Two wheel variants:
#   x86   (default)  tvm-ti-c7x-compile   — full compilation toolchain
#   arm64            tvm-ti-c7x-inference  — minimal on-board inference runtime
#
# Prerequisites:
#   - TVM built (cmake + ninja) with libtvm_runtime.so and libtvm_compiler.so
#     in $TVM_HOME/build/lib/
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
#   C7X_MMA_TIDL_PATH  - c7x-mma-tidl repo root (TIDL .so fallback; source of
#                        the IOBufDesc layout table and tidl_tools)
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
    check_file "$TVM_HOME/build/lib/libtvm_runtime.so" \
        "Build TVM first: cd build && cmake -G Ninja .. && ninja"
    check_file "$TVM_HOME/build/lib/libtvm_runtime_extra.so" \
        "Build TVM first with USE_SORT=ON: cd build && cmake -G Ninja .. && ninja"
    check_file "$TVM_HOME/build/lib/libtvm_compiler.so" \
        "Build TVM first: cd build && cmake -G Ninja .. && ninja"
    check_file "$DSP_RT/build-c7x-host/libtvm_dsp_runtime_c7x_host.a" \
        "Build DSP runtime: bash build_runtime.sh c7x_host"
    check_file "$DSP_RT/build-c7x${RUNTIME_SUFFIX}/libtvm_dsp_runtime_c7x.a" \
        "Build DSP runtime: bash build_runtime.sh c7x --board $BOARD --ddr $DDR"
    check_file "$DSP_RT/firmware/c7x/dsp/build${FW_SUFFIX}/c7x_compute.out" \
        "Build firmware: cd firmware/c7x/dsp && ./build.sh --board $BOARD --ddr $DDR --tidl $TIDL"
    check_file "$DSP_RT/firmware/c7x/arm/build${RUNTIME_SUFFIX}/c7x_compute" \
        "Build ARM client: cd firmware/c7x/arm && ./build.sh --board $BOARD --ddr $DDR"

    TIDL_SO=""
    if [ "$TIDL" = "ON" ]; then
        C7X_TIDL="${C7X_MMA_TIDL_PATH:-$HOME/ml/c7x-mma-tidl}"
        TIDL_SO="${TIDL_RELAX_SO:-$C7X_TIDL/ti_dl/utils/tidlModelImport/out/tidl_model_import_relax.so}"
        check_file "$TIDL_SO" \
            "Build TIDL .so or set TIDL_RELAX_SO / C7X_MMA_TIDL_PATH (or pass --tidl OFF for a beagley-ai-style no-TIDL wheel)"
        # The IOBufDesc layout and tidl_tools are taken from $C7X_TIDL. Keep
        # TIDL_RELAX_SO (if set) pointing into the same tree: a layout or
        # tool that differs from the .so's build misreads its blobs.
        TIDL_IO_H="$C7X_TIDL/ti_dl/inc/itidl_io.h"
        check_file "$TIDL_IO_H" \
            "Set C7X_MMA_TIDL_PATH to the c7x-mma-tidl tree the TIDL .so was built in"
        # Only what tidl_model_import_relax.so runs or reads, not the whole
        # directory (it also holds TI headers and ONNX/TFLite runtime plugins).
        TIDL_TOOLS_FILES="PC_dsp_test_dl_algo.out ti_cnnperfsim.out device_config.cfg"
        for f in $TIDL_TOOLS_FILES; do
            check_file "$C7X_TIDL/tidl_tools/$f" \
                "Build c7x-mma-tidl (build_j722s.sh) so tidl_tools/$f exists"
        done
        # c7x_host TIDL: the J722S x86 TIDL algo libs (same tree as the .so)
        # and the C7524 MMALIB x86 libs they call, defaulting to the MMALIB
        # the dsp-cpp harness links in source mode. Names and order match
        # dsp-cpp's TIDL_PC_LIB_NAMES / MMALIB_HOST_LIB_NAMES.
        TIDL_PC_LIB_DIR="$C7X_TIDL/ti_dl/lib/J722S/PC/algo/release"
        HOST_TIDL_LIBS="tidl_algo tidl_ref tidl_priv tidl_kernels tidl_avx_kernels tidl_custom"
        HOST_MMALIB_ROOT="${MMALIB_PATH:-/opt/ti/am67a/ti-processor-sdk-rtos-j722s-evm-11_00_00_06/mmalib_11_02_00_06}"
        HOST_MMALIB_DIR="$HOST_MMALIB_ROOT/lib/C7524/release"
        HOST_MMALIB_LIBS="mmalib_x86_64 mmalib_cn_x86_64 common_x86_64"
        for l in $HOST_TIDL_LIBS; do
            check_file "$TIDL_PC_LIB_DIR/lib$l.a" "Build the c7x-mma-tidl J722S PC libs"
        done
        for l in $HOST_MMALIB_LIBS; do
            check_file "$HOST_MMALIB_DIR/lib$l.a" \
                "Set MMALIB_PATH to the MMALIB the c7x_host harness links"
        done
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
    # TVM 0.27 splits libtvm.so into libtvm_runtime.so and libtvm_compiler.so.
    # libtvm_runtime_extra.so holds the sort kernels (tvm.contrib.sort.topk
    # etc.) that host-side calibration runs need; the backend autoloader
    # ignores it silently when missing, so it must be bundled.
    # python/tvm/libinfo.py searches <package>/lib/ first for the wheel layout.
    mkdir -p "$STAGING/tvm/lib"
    for lib in libtvm_runtime.so libtvm_runtime_extra.so libtvm_compiler.so; do
        if [ "$STRIP_LIBS" = "1" ]; then
            strip --strip-debug -o "$STAGING/tvm/lib/$lib" "$TVM_HOME/build/lib/$lib"
        else
            cp "$TVM_HOME/build/lib/$lib" "$STAGING/tvm/lib/"
        fi
    done

    # --- DSP data directory ---
    echo ">>> Staging DSP artifacts ..."
    DATA="$STAGING/tvm/data/ti_dsp"
    mkdir -p "$DATA/firmware" "$DATA/lib" "$DATA/tidl" \
             "$DATA/cmake" "$DATA/dynmod" "$DATA/include/dlpack"
    touch "$STAGING/tvm/data/__init__.py"
    touch "$DATA/__init__.py"

    # DSP runtime libraries
    cp "$DSP_RT/build-c7x-host/libtvm_dsp_runtime_c7x_host.a" "$DATA/lib/"
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
        # sTIDL_IOBufDesc_t layout, for reading params_1.bin without the
        # header (itidl_io.h is not redistributable). The script has no
        # module-level tvm imports, so it runs without loading libtvm.
        python3 "$TVM_HOME/python/tvm/relax/backend/tidl/io_buf_desc.py" \
            --header "$TIDL_IO_H" --out "$DATA/tidl/sTIDL_IOBufDesc_t.layout.json"
        # Every tidl_tools entry is a symlink into ti_dl/ or arm-tidl/, hence -L.
        mkdir -p "$DATA/tidl/tidl_tools"
        for f in $TIDL_TOOLS_FILES; do
            cp -L "$C7X_TIDL/tidl_tools/$f" "$DATA/tidl/tidl_tools/"
        done

        # c7x_host TIDL without SDKs: build the TIDL glue (tidl_api.c and
        # friends) with the harness's own tidl_api_host target, so the flags
        # are exactly those of a source-mode build, and against the same
        # itidl_ti.h as the shipped libtidl_*.a. The harness links lib/host
        # instead of compiling the glue when it finds this bundle.
        echo ">>> Building TIDL host-emulation glue ..."
        HOST_GLUE_BUILD="$(mktemp -d)"
        (cd "$HOST_GLUE_BUILD" && \
            C7X_MMA_TIDL_PATH="$C7X_TIDL" MMALIB_PATH="$HOST_MMALIB_ROOT" \
            cmake -DTVM_DSP_TARGET=c7x_host -DTVM_HOME="$TVM_HOME" \
                  -DCMAKE_BUILD_TYPE=Release -DUSE_TIDL=ON -DTIDL_PREBUILT=OFF \
                  -DGENERATED_CODE_DIR="$HOST_GLUE_BUILD" \
                  "$TVM_HOME/tests/ti-dsp-runtime/dsp-cpp" > cmake.log 2>&1 && \
            cmake --build . --target tidl_api_host >> cmake.log 2>&1) || {
            echo "ERROR: TIDL host glue build failed, see $HOST_GLUE_BUILD/cmake.log"
            exit 1
        }
        mkdir -p "$DATA/lib/host"
        cp "$HOST_GLUE_BUILD/libtidl_api_host.a" "$DATA/lib/host/"
        rm -rf "$HOST_GLUE_BUILD"
        for l in $HOST_TIDL_LIBS; do
            cp "$TIDL_PC_LIB_DIR/lib$l.a" "$DATA/lib/host/"
        done
        for l in $HOST_MMALIB_LIBS; do
            cp "$HOST_MMALIB_DIR/lib$l.a" "$DATA/lib/host/"
        done
    fi

    # Build infrastructure
    cp "$DSP_RT/cmake/toolchain-c7x.cmake" "$DSP_RT/cmake/WeightEmbedding.cmake" "$DATA/cmake/"
    # c7x_host harness (tidl.py _build_c7x_host): dsp-cpp's main, io/ and
    # CMakeLists, plus the two runtime sources it compiles in. It finds the
    # headers and libtvm_dsp_runtime_c7x_host.a through DSP_RUNTIME_DIR.
    DSP_CPP="$TVM_HOME/tests/ti-dsp-runtime/dsp-cpp"
    mkdir -p "$DATA/host/io" "$DATA/dma" "$DATA/mmalib"
    cp "$DSP_CPP/CMakeLists.txt" "$DSP_CPP/main_dsp.cpp" "$DATA/host/"
    cp "$DSP_CPP/io/"*.cpp "$DSP_CPP/io/"*.h "$DATA/host/io/"
    cp "$DSP_RT/dma/tvm_dsp_dma_host.c" "$DATA/dma/"
    cp "$DSP_RT/mmalib/tidl_maxpool_wrapper.cpp" "$DATA/mmalib/"
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

    # Stub tvm/__init__.py — does not load libtvm_runtime.so/libtvm_compiler.so
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
    cp "$TVM_HOME/python/tvm/data/ti_dsp/deploy.py" "$STAGING/tvm/data/ti_dsp/"

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
