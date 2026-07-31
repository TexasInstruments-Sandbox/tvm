# Shared board/ddr (+ optional tidl/mmalib) -> build-directory-suffix
# naming, so build_runtime.sh, firmware/c7x/dsp/build.sh, and
# firmware/c7x/arm/build.sh always agree on where a given board's
# artifacts live. cmake/boards.cmake remains the sole source of truth
# for what actually gets built; this only picks the directory name, so
# nothing here duplicates it independently -- it just needs to keep
# agreeing with it.
#
# Usage: source this file, set TVM_BOARD/TVM_DDR (and, only for the
# firmware dsp build, TVM_TIDL/TVM_MMALIB) as the caller already does
# for its own --board/--ddr/--tidl/--mmalib flags, then call
# `resolve_board_build_dir`. Sets (in the caller's scope, not a
# subshell): BOARD, DDR, BUILD_SUFFIX, and -- only when TVM_TIDL is
# set -- MMALIB.

resolve_board_build_dir() {
    BOARD="${TVM_BOARD:-j722s-evm}"
    if [ -n "${TVM_DDR:-}" ]; then
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

    # TIDL/MMALIB suffix only applies to the firmware dsp build; callers
    # that never set TVM_TIDL (build_runtime.sh, arm/build.sh) skip this
    # entirely, same as before this was factored out.
    if [ -n "${TVM_TIDL:-}" ]; then
        MMALIB="${TVM_MMALIB:-OFF}"
        [ "$TVM_TIDL" = "ON" ] && MMALIB="ON"
        if [ "$TVM_TIDL" != "ON" ]; then
            BUILD_SUFFIX="${BUILD_SUFFIX}-tidl-${TVM_TIDL}-mmalib-${MMALIB}"
        fi
    fi
}
