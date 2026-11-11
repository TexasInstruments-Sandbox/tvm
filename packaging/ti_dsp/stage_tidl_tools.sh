#!/bin/bash
# Stage the TIDL tools directory that TVM reads during TIDL offload.
#
# TVM's TIDL flow (partition, import, output sizing, c7x_host link) reads
# every TIDL file from one directory, by default tvm/data/ti_dsp/tidl/ in
# the Python package, or config["tidl_tools_dir"]. The files come from five
# places in a c7x-mma-tidl build plus the Processor SDK; this script is the
# only thing that knows those places. The wheel build and a source-tree build
# (build_runtime.sh tidl-tools) both call it, so the two get the same tree.
#
# Layout written to <dest>:
#   tidl_model_import_relax.so, tidl_model_custom_import.so   import library
#   tidl_tools/        PC_dsp_test_dl_algo.out, ti_cnnperfsim.out,
#                      device_config.cfg (run by the import library)
#   host/              libtidl_api_host.a, TIDL and MMALIB x86 archives
#                      (c7x_host link)
#
# Everything comes from one c7x-mma-tidl tree, so the import library, the
# tools and the host algo libraries match.
#
# Usage:
#   stage_tidl_tools.sh --c7x-mma-tidl <tree> --psdk <psdk> [--mmalib <mmalib>] <dest>
#
#   --c7x-mma-tidl  c7x-mma-tidl tree, already built (build_j722s.sh)
#   --psdk          TI Processor SDK RTOS for J722S (for ivision.h)
#   --mmalib        MMALIB root (default: <psdk>/mmalib_11_02_00_06)
#
# Needs TVM_HOME (the TVM source tree) and TI_CGT_C7000_PATH (the host
# emulation headers the glue compiles against). J722S only.

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
TVM_HOME="${TVM_HOME:-$(cd "$SCRIPT_DIR/../.." && pwd)}"

C7X_TIDL=""
PSDK=""
MMALIB=""
DEST=""
while [ $# -gt 0 ]; do
    case "$1" in
        --c7x-mma-tidl) C7X_TIDL="$2"; shift 2 ;;
        --psdk) PSDK="$2"; shift 2 ;;
        --mmalib) MMALIB="$2"; shift 2 ;;
        -*) echo "Unknown argument: $1" >&2; exit 1 ;;
        *) DEST="$1"; shift ;;
    esac
done
if [ -z "$C7X_TIDL" ] || [ -z "$PSDK" ] || [ -z "$DEST" ]; then
    echo "Usage: $0 --c7x-mma-tidl <tree> --psdk <psdk> [--mmalib <mmalib>] <dest>" >&2
    exit 1
fi
MMALIB="${MMALIB:-$PSDK/mmalib_11_02_00_06}"

check_file() {
    if [ ! -f "$1" ]; then
        echo "ERROR: Required file not found: $1" >&2
        echo "  $2" >&2
        exit 1
    fi
}

IMPORT_OUT="$C7X_TIDL/ti_dl/utils/tidlModelImport/out"
IMPORT_SOS="tidl_model_import_relax.so tidl_model_custom_import.so"
TOOLS="$C7X_TIDL/ti_dl/test/PC_dsp_test_dl_algo.out
$C7X_TIDL/ti_dl/utils/perfsim/ti_cnnperfsim.out
$C7X_TIDL/ti_dl/test/testvecs/config/import/device_config.cfg"
TIDL_PC_LIB_DIR="$C7X_TIDL/ti_dl/lib/J722S/PC/algo/release"
# Names and order match TIDL_PC_LIB_NAMES / MMALIB_HOST_LIB_NAMES in
# tests/ti-dsp-runtime/dsp-cpp/CMakeLists.txt.
TIDL_PC_LIBS="tidl_algo tidl_ref tidl_priv tidl_kernels tidl_avx_kernels tidl_custom"
MMALIB_DIR="$MMALIB/lib/C7524/release"
MMALIB_LIBS="mmalib_x86_64 mmalib_cn_x86_64 common_x86_64"

for f in $IMPORT_SOS; do
    check_file "$IMPORT_OUT/$f" "Build c7x-mma-tidl first (build_j722s.sh)"
done
for f in $TOOLS; do
    check_file "$f" "Build c7x-mma-tidl first (build_j722s.sh)"
done
for l in $TIDL_PC_LIBS; do
    check_file "$TIDL_PC_LIB_DIR/lib$l.a" "Build the c7x-mma-tidl J722S PC libs (build_j722s.sh)"
done
for l in $MMALIB_LIBS; do
    check_file "$MMALIB_DIR/lib$l.a" "Pass --mmalib <MMALIB root>"
done
check_file "$PSDK/ivision/ivision.h" "Pass --psdk <TI Processor SDK RTOS for J722S>"

echo ">>> Staging TIDL tools into $DEST"
# Remove only what this script writes: in a wheel, <dest> is also the DSP
# runtime's tidl/ directory and holds tidl_api.h, which the bridge includes.
mkdir -p "$DEST"
# sTIDL_IOBufDesc_t.layout.json: written by earlier versions, no longer used.
for f in $IMPORT_SOS sTIDL_IOBufDesc_t.layout.json; do
    rm -f "$DEST/$f"
done
rm -rf "$DEST/tidl_tools" "$DEST/host"
mkdir -p "$DEST/tidl_tools" "$DEST/host"

# Import library. The relax .so loads the custom one by RUNPATH $ORIGIN, so
# the two must stay side by side.
for f in $IMPORT_SOS; do
    cp "$IMPORT_OUT/$f" "$DEST/"
done
# Only what the import library runs or reads. cp -L: c7x-mma-tidl keeps
# some of these as symlinks.
for f in $TOOLS; do
    cp -L "$f" "$DEST/tidl_tools/"
done

# c7x_host glue: build tidl_api.c and friends with the harness's own
# tidl_api_host target, so the flags are those of a c7x_host build, and
# against the itidl_ti.h of the same tree as the libtidl_*.a staged below.
# A glue library built against a different itidl_ti.h shifts struct
# layouts without any error.
echo ">>> Building TIDL host-emulation glue ..."
GLUE_BUILD="$(mktemp -d)"
trap 'rm -rf "$GLUE_BUILD"' EXIT
if ! (cd "$GLUE_BUILD" && \
      cmake -DTVM_DSP_TARGET=c7x_host -DTVM_HOME="$TVM_HOME" \
            -DCMAKE_BUILD_TYPE=Release -DUSE_TIDL=ON \
            -DTIDL_GLUE_ONLY=ON -DTIDL_GLUE_C7X_MMA_TIDL="$C7X_TIDL" \
            -DTIDL_GLUE_IVISION="$PSDK/ivision" \
            -DGENERATED_CODE_DIR="$GLUE_BUILD" \
            "$TVM_HOME/tests/ti-dsp-runtime/dsp-cpp" > cmake.log 2>&1 && \
      cmake --build . --target tidl_api_host >> cmake.log 2>&1); then
    cat "$GLUE_BUILD/cmake.log" >&2
    echo "ERROR: TIDL host glue build failed" >&2
    exit 1
fi
cp "$GLUE_BUILD/libtidl_api_host.a" "$DEST/host/"
for l in $TIDL_PC_LIBS; do
    cp "$TIDL_PC_LIB_DIR/lib$l.a" "$DEST/host/"
done
for l in $MMALIB_LIBS; do
    cp "$MMALIB_DIR/lib$l.a" "$DEST/host/"
done

echo ">>> TIDL tools staged: $DEST"
