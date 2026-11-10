# C7x Compute Service - DSP Firmware
#
# Provides RPMessage-based compute service for host applications.
#
# Build: ./build.sh
# Output: c7x_compute.out (ELF firmware)

cmake_minimum_required(VERSION 3.16)

# Board/DDR -> SDK roots + shared-DMA phys base (single source of truth).
# The firmware always cross-compiles against the board's TI SDK, so require it:
# boards.cmake errors (no default) if PSDK_INSTALL_PATH/MCU_PLUS_SDK_PATH
# resolve to nothing or a nonexistent path.
set(TVM_REQUIRE_SDK TRUE)
include("${CMAKE_CURRENT_SOURCE_DIR}/../../../cmake/boards.cmake")

#
# Tool paths - update these if your installation differs
#
# Tool paths: prefer environment variables, fall back to developer defaults
if(DEFINED ENV{TI_SYSCONFIG_ROOT})
    set(TI_SYSCONFIG_ROOT "$ENV{TI_SYSCONFIG_ROOT}")
else()
    # 1.28 is required for the beagley-ai (SDK 11.02.01) sysconfig run --
    # 1.26.0 errors with "requires at least version 1.26.2 of SysConfig".
    # Verified byte-identical MMU region generation for the j722s-evm
    # (SDK 11.00.00) path too, so this is the default for both boards.
    set(TI_SYSCONFIG_ROOT "/opt/ti/sysconfig-1.28")
endif()

if(DEFINED ENV{TI_CGT_C7000_PATH})
    set(TI_CGT_ROOT "$ENV{TI_CGT_C7000_PATH}")
elseif(DEFINED ENV{TI_CGT_ROOT})
    set(TI_CGT_ROOT "$ENV{TI_CGT_ROOT}")
else()
    set(TI_CGT_ROOT "$ENV{HOME}/ti/ccs2040/ccs/tools/compiler/ti-cgt-c7000_5.0.1.LTS")
endif()

# Resolved by boards.cmake from TVM_BOARD/TVM_DDR (env override still wins).
set(MCUPSDK_ROOT "${MCU_PLUS_SDK_PATH}")

# Device and core settings.  TVM_SOC comes from boards.cmake (j722s for
# j722s-evm/beagley-ai, am62dx for am62d); the j722s values are unchanged.
if(TVM_SOC STREQUAL "am62dx")
    set(DEVICE "am62dx")
    set(CORE "c75x")
    set(TVM_FW_SOC_DEFINE SOC_AM62DX)
    set(TVM_FW_DEVICE_DEFINE TVM_DSP_DEVICE_AM62DX)
else()
    set(DEVICE "j722s")
    set(CORE "c75ss0-0")
    set(TVM_FW_SOC_DEFINE SOC_J722S)
    set(TVM_FW_DEVICE_DEFINE TVM_DSP_DEVICE_J722S)
endif()
set(PROFILE "release")

# Set toolchain before project()
set(CMAKE_TOOLCHAIN_FILE "${CMAKE_CURRENT_SOURCE_DIR}/cmake/toolchain-c7000.cmake")

project(c7x_compute C CXX ASM)

# Set proper library path and library flags for TI linker
set(CMAKE_LIBRARY_PATH_FLAG "-i")
set(CMAKE_LINK_LIBRARY_FLAG "-l")

# Verify tool paths exist
if(NOT EXISTS "${TI_SYSCONFIG_ROOT}")
    message(FATAL_ERROR "SysConfig not found at: ${TI_SYSCONFIG_ROOT}")
endif()
if(NOT EXISTS "${TI_CGT_ROOT}")
    message(FATAL_ERROR "TI CGT C7000 not found at: ${TI_CGT_ROOT}")
endif()
if(NOT EXISTS "${MCUPSDK_ROOT}")
    message(FATAL_ERROR "MCU+ SDK not found at: ${MCUPSDK_ROOT}")
endif()

message(STATUS "SysConfig: ${TI_SYSCONFIG_ROOT}")
message(STATUS "TI CGT: ${TI_CGT_ROOT}")
message(STATUS "MCU+ SDK: ${MCUPSDK_ROOT}")

# Find SysConfig CLI tool
find_program(SYSCONFIG_TOOL
    NAMES sysconfig_cli.sh
    HINTS "${TI_SYSCONFIG_ROOT}"
    NO_DEFAULT_PATH
)
if(NOT SYSCONFIG_TOOL)
    message(FATAL_ERROR "SysConfig CLI not found in ${TI_SYSCONFIG_ROOT}")
endif()
message(STATUS "SysConfig CLI: ${SYSCONFIG_TOOL}")

# SysConfig input file.  Region 9's pAddr (shared DMA carveout) is
# parameterized by TVM_DDR via boards.cmake's C7X_SHARED_PHYS_BASE;
# configure_file runs at CMake configure time, so a --ddr switch (which
# reconfigures into a fresh, board+ddr-keyed build dir) always regenerates
# this with the correct base -- no stale MMU map is ever reused.
# am62d uses a fixed, plain syscfg (single shared carveout, nothing to
# parameterize).
if(TVM_SOC STREQUAL "am62dx")
    set(SYSCFG_FILE "${CMAKE_CURRENT_SOURCE_DIR}/configs/am62d_c75ss0.syscfg")
else()
    configure_file(
        "${CMAKE_CURRENT_SOURCE_DIR}/configs/c75ss0.syscfg.in"
        "${CMAKE_CURRENT_BINARY_DIR}/c75ss0.syscfg"
        @ONLY
    )
    set(SYSCFG_FILE "${CMAKE_CURRENT_BINARY_DIR}/c75ss0.syscfg")
endif()

# Generated files from SysConfig
set(SYSCFG_OUTPUT_DIR "${CMAKE_CURRENT_BINARY_DIR}/syscfg")
set(SYSCFG_C_FILES
    ${SYSCFG_OUTPUT_DIR}/ti_drivers_config.c
    ${SYSCFG_OUTPUT_DIR}/ti_drivers_open_close.c
    ${SYSCFG_OUTPUT_DIR}/ti_board_config.c
    ${SYSCFG_OUTPUT_DIR}/ti_board_open_close.c
    ${SYSCFG_OUTPUT_DIR}/ti_dpl_config.c
    ${SYSCFG_OUTPUT_DIR}/ti_pinmux_config.c
    ${SYSCFG_OUTPUT_DIR}/ti_power_clock_config.c
)
set(SYSCFG_H_FILES
    ${SYSCFG_OUTPUT_DIR}/ti_drivers_config.h
    ${SYSCFG_OUTPUT_DIR}/ti_drivers_open_close.h
    ${SYSCFG_OUTPUT_DIR}/ti_board_config.h
    ${SYSCFG_OUTPUT_DIR}/ti_board_open_close.h
    ${SYSCFG_OUTPUT_DIR}/ti_dpl_config.h
    ${SYSCFG_OUTPUT_DIR}/ti_pinmux_config.h
    ${SYSCFG_OUTPUT_DIR}/ti_power_clock_config.h
)

# Run SysConfig to generate driver configuration
add_custom_command(
    OUTPUT ${SYSCFG_C_FILES} ${SYSCFG_H_FILES}
    COMMAND ${CMAKE_COMMAND} -E make_directory ${SYSCFG_OUTPUT_DIR}
    COMMAND ${SYSCONFIG_TOOL}
        --script ${SYSCFG_FILE}
        --output ${SYSCFG_OUTPUT_DIR}
        --compiler ticlang
        --part Default
        --package Default
        --product ${MCUPSDK_ROOT}/.metadata/product.json
    DEPENDS ${SYSCFG_FILE}
    COMMENT "Running SysConfig to generate driver configuration"
    VERBATIM
)

# Create library from generated SysConfig files
add_library(syscfg_generated OBJECT ${SYSCFG_C_FILES})
target_include_directories(syscfg_generated PRIVATE
    ${SYSCFG_OUTPUT_DIR}
    ${MCUPSDK_ROOT}/source
    ${MCUPSDK_ROOT}/source/kernel/freertos/FreeRTOS-Kernel/include
    ${MCUPSDK_ROOT}/source/kernel/freertos/portable/TI_CGT/DSP_C75X
    ${MCUPSDK_ROOT}/source/kernel/freertos/config/${DEVICE}/c75x
    ${MCUPSDK_ROOT}/source/drivers
    ${MCUPSDK_ROOT}/source/drivers/hw_include
    ${MCUPSDK_ROOT}/source/drivers/hw_include/${DEVICE}
)
target_compile_definitions(syscfg_generated PRIVATE
    ${TVM_FW_SOC_DEFINE}
    BUILD_C7X
    BUILD_C7X_1
    OS_FREERTOS
    Hwi_bootToNonSecure__D=1
    # Increase remoteproc trace buffer from 4KB to 256KB.
    # Requires #ifndef guard patch in SDK DebugP.h (see commit log).
    # Must fit within DDR_C7x_1_IPC_TRACE linker region (0xFFC00 bytes).
    DebugP_MEM_LOG_SIZE=\(256U*1024U\)
)

# Linker command file
if(TVM_SOC STREQUAL "am62dx")
    set(LINKER_CMD "${CMAKE_CURRENT_SOURCE_DIR}/configs/linker_am62d_c75_freertos.cmd")
else()
    set(LINKER_CMD "${CMAKE_CURRENT_SOURCE_DIR}/configs/linker_c75_freertos.cmd")
endif()

# TVM DSP Runtime (relative to firmware location within ti_dsp/)
set(TVM_DSP_RUNTIME_DIR "${CMAKE_CURRENT_SOURCE_DIR}/../../..")
# build_runtime.sh names the runtime build dir build-c7x${TVM_BUILD_SUFFIX}
# (empty for the default board+ddr); boards.cmake resolves the same
# suffix here so a non-default --board/--ddr firmware build finds the
# matching runtime lib instead of the EVM/8gb default.
set(TVM_DSP_RUNTIME_LIB "${TVM_DSP_RUNTIME_DIR}/build-c7x${TVM_BUILD_SUFFIX}/libtvm_dsp_runtime_c7x.a")
# TVM repo root is 6 levels up: firmware/c7x/dsp -> ti_dsp -> runtime -> src -> tvm
set(TVM_ROOT "${CMAKE_CURRENT_SOURCE_DIR}/../../../../../../")
set(TVM_DLPACK_DIR "${TVM_ROOT}/3rdparty/tvm-ffi/3rdparty/dlpack/include")

if(NOT EXISTS "${TVM_DSP_RUNTIME_LIB}")
    message(WARNING "TVM DSP runtime library not found at: ${TVM_DSP_RUNTIME_LIB}")
    message(WARNING "Build it first: cd ${TVM_DSP_RUNTIME_DIR}/build-c7x && cmake --build .")
endif()

# DLOAD library sources
set(DLOAD_SOURCES
    src/dload/DLOAD/dload.c
    src/dload/DLOAD/elf32.c
    src/dload/DLOAD/dload_endian.c
    src/dload/DLOAD/ArrayList.c
    src/dload/DLOAD_SYM/symtab.c
    src/dload/C70_DLOAD_DYN/c70_dynamic.c
    src/dload/C70_DLOAD_REL/c70_reloc.c
)

# DmaUtils compiled from source with standalone UDMA path.
# The prebuilt dmautils.lib uses the full UDMA driver which accesses
# NAVSS registers not available in remoteproc mode.  Compiling from
# source with DMA_UTILS_STANDALONE uses the standalone UDMA driver
# which only accesses the DRU directly.  The two .c files under
# src/dmautils/ are vendored copies with include paths fixed to use
# MCU+ SDK-native paths (no PDK shim required).
set(DMAUTILS_SRC_DIR ${MCUPSDK_ROOT}/source/drivers/dmautils)
set(DMAUTILS_SOURCES
    src/dmautils/dmautils_autoincrement_3d.c
    src/dmautils/udma.c
)

# Build dmautils+standalone UDMA as a static library so the linker
# only pulls needed symbols (avoids conflicts with drivers.lib).
add_library(dmautils_standalone STATIC ${DMAUTILS_SOURCES})
target_include_directories(dmautils_standalone PRIVATE
    ${DMAUTILS_SRC_DIR}/include
    ${DMAUTILS_SRC_DIR}/csl
    ${DMAUTILS_SRC_DIR}/udma_standalone
    ${DMAUTILS_SRC_DIR}/udma_standalone/include
    ${MCUPSDK_ROOT}/source
    ${MCUPSDK_ROOT}/source/kernel/nortos/dpl/c75
    ${MCUPSDK_ROOT}/source/drivers/hw_include
    ${MCUPSDK_ROOT}/source/drivers/hw_include/${DEVICE}
    ${TI_CGT_ROOT}/include
)
target_compile_definitions(dmautils_standalone PRIVATE
    DMA_UTILS_STANDALONE ${TVM_FW_SOC_DEFINE} BUILD_C7X_1 BUILD_C7X MCU_PLUS_SDK
)

# Main executable
add_executable(${PROJECT_NAME}.out
    src/main.c
    src/compute_service.c
    src/dyn_loader.c
    src/tvm_model.c
    src/shm_printf.c
    src/tidl_support.c
    ${DLOAD_SOURCES}
)

# Add linker command file
target_sources(${PROJECT_NAME}.out PRIVATE ${LINKER_CMD})

# Include directories
target_include_directories(${PROJECT_NAME}.out PRIVATE
    ${SYSCFG_OUTPUT_DIR}
    ${CMAKE_CURRENT_SOURCE_DIR}/../common
    ${CMAKE_CURRENT_SOURCE_DIR}/src
    ${MCUPSDK_ROOT}/source
    ${MCUPSDK_ROOT}/source/kernel/freertos/FreeRTOS-Kernel/include
    ${MCUPSDK_ROOT}/source/kernel/freertos/portable/TI_CGT/DSP_C75X
    ${MCUPSDK_ROOT}/source/kernel/freertos/config/${DEVICE}/c75x
    ${MCUPSDK_ROOT}/source/drivers
    ${MCUPSDK_ROOT}/source/drivers/hw_include
    ${MCUPSDK_ROOT}/source/drivers/hw_include/${DEVICE}
    ${MCUPSDK_ROOT}/source/kernel/nortos/dpl/c75
    ${TI_CGT_ROOT}/include
    ${CMAKE_CURRENT_SOURCE_DIR}/configs
    # DLOAD includes
    ${CMAKE_CURRENT_SOURCE_DIR}/src/dload/DLOAD
    ${CMAKE_CURRENT_SOURCE_DIR}/src/dload/DLOAD_API
    ${CMAKE_CURRENT_SOURCE_DIR}/src/dload/C70_DLOAD_DYN
    ${CMAKE_CURRENT_SOURCE_DIR}/src/dload/C70_DLOAD_REL
    # TVM DSP Runtime includes
    ${TVM_DSP_RUNTIME_DIR}
    ${TVM_DSP_RUNTIME_DIR}/include
    ${TVM_DSP_RUNTIME_DIR}/ffi
    ${TVM_DSP_RUNTIME_DIR}/container
    ${TVM_DSP_RUNTIME_DIR}/core
    ${TVM_DLPACK_DIR}
    # DmaUtils for async DMA (DRU direct TR mode)
    ${MCUPSDK_ROOT}/source/drivers/dmautils/include
    ${MCUPSDK_ROOT}/source/drivers/dmautils/csl
    ${MCUPSDK_ROOT}/source/drivers/dmautils/udma_standalone
    ${MCUPSDK_ROOT}/source/drivers/dmautils/udma_standalone/include
)

# Compile definitions
target_compile_definitions(${PROJECT_NAME}.out PRIVATE
    ${TVM_FW_SOC_DEFINE}
    BUILD_C7X
    BUILD_C7X_1
    OS_FREERTOS
    Hwi_bootToNonSecure__D=1
    # DLOAD configuration
    C70_TARGET
    ELF64=1
    _SYS_BIOS
    # TVM DSP runtime configuration
    TVM_DSP_TARGET_C7X
    ${TVM_FW_DEVICE_DEFINE}
    TVM_DSP_NO_EXCEPTIONS=1
    TVM_DSP_NO_THREADS=1
    # DmaUtils headers need MCU_PLUS_SDK for correct include paths
    MCU_PLUS_SDK
)
if(DEFINED C7X_SHARED_PHYS_BASE)
    target_compile_definitions(${PROJECT_NAME}.out PRIVATE
        C7X_SHARED_PHYS_BASE=${C7X_SHARED_PHYS_BASE}ULL
    )
endif()

# Link directories
target_link_directories(${PROJECT_NAME}.out PRIVATE
    ${TI_CGT_ROOT}/lib
)

# --- TIDL Runtime (optional) ---
# When enabled, links TIDL algo libraries into the firmware. TIDL's own algo
# library (tidl_algo.lib) has unresolved MMALIB_CNN_*/MMALIB_LINALG_*
# symbol references, so USE_TIDL_RUNTIME=ON requires USE_TI_MMALIB=ON too --
# enforced below rather than left as a user error that fails at link time.
option(USE_TIDL_RUNTIME "Link TIDL algo libs into firmware" OFF)
# --- MMALIB direct integration (optional) ---
# When enabled, links MMALIB conv2d/matmul kernels into the firmware for
# direct offload (independent of TIDL subgraph offload).
option(USE_TI_MMALIB "Link MMALIB direct-integration libs into firmware" OFF)
# c7x-mma-tidl ti_dl/lib/<SOC> directory and MMALIB silicon directory.  am62d
# is built with TARGET_SOC=AM62A (see c7x-mma-tidl/build_am62d.sh).
if(TVM_SOC STREQUAL "am62dx")
    set(_TIDL_SOC_DIR "AM62A")
    set(_MMALIB_SILICON "C7504")
else()
    set(_TIDL_SOC_DIR "J722S")
    set(_MMALIB_SILICON "C7524")
endif()
if(USE_TIDL_RUNTIME AND NOT USE_TI_MMALIB)
    message(STATUS "USE_TIDL_RUNTIME=ON requires MMALIB (tidl_algo.lib "
                    "has unresolved MMALIB_CNN_*/MMALIB_LINALG_* symbols) "
                    "-- forcing USE_TI_MMALIB=ON")
    set(USE_TI_MMALIB ON CACHE BOOL "Link MMALIB direct-integration libs into firmware" FORCE)
endif()

set(TIDL_RUNTIME_LIBS "")
if(USE_TIDL_RUNTIME)
    # PSDK_INSTALL_PATH resolved by boards.cmake (env override still wins).
    # C7X_MMA_TIDL_PATH is not board-keyed -- same override semantics as
    # before, just falling back to boards.cmake's SDK root.
    if(DEFINED ENV{C7X_MMA_TIDL_PATH})
        set(TIDL_LIB_PATH "$ENV{C7X_MMA_TIDL_PATH}/ti_dl/lib/${_TIDL_SOC_DIR}/dsp/algo/release")
    else()
        set(TIDL_LIB_PATH "${PSDK_INSTALL_PATH}/c7x-mma-tidl/ti_dl/lib/${_TIDL_SOC_DIR}/dsp/algo/release")
    endif()

    set(_TIDL_LIB_NAMES tidl_algo tidl_ref tidl_priv tidl_kernels tidl_custom)
    foreach(_lib IN LISTS _TIDL_LIB_NAMES)
        if(NOT EXISTS "${TIDL_LIB_PATH}/${_lib}.lib")
            message(FATAL_ERROR "TIDL library ${_lib}.lib not found in: ${TIDL_LIB_PATH} "
                                "(an older c7x-mma-tidl checkout may predate the "
                                "tidl_ref/tidl_priv/tidl_kernels split)")
        endif()
        list(APPEND TIDL_RUNTIME_LIBS ${TIDL_LIB_PATH}/${_lib}.lib)
    endforeach()
    # Force the linker to retain TIDL_VISION_FXNS (IALG function table) and
    # the TIDL maxpool entry point that DLOAD modules reference. Without
    # --retain, the linker eliminates them as "unused" since nothing in the
    # firmware calls them directly.
    target_link_options(${PROJECT_NAME}.out PRIVATE
        --retain=TIDL_VISION_FXNS
        --retain=c7x_int8_max_pool_tidl
    )

    # TIDL max pool wrapper: compiled into firmware, exported to DLOAD
    # modules. Lives in mmalib/ (alongside the MMALIB wrappers) since it
    # requires MMALIB at link (see force logic above).
    # Resolve TIDL source root for include headers.  TIDL_LIB_PATH goes deep
    # into the lib tree so we derive it from C7X_MMA_TIDL_PATH directly.
    if(DEFINED ENV{C7X_MMA_TIDL_PATH})
        set(_TIDL_SRC_ROOT "$ENV{C7X_MMA_TIDL_PATH}/ti_dl")
        set(_TIDL_KERNELS_INC "$ENV{C7X_MMA_TIDL_PATH}/tidl-kernels/include")
    else()
        set(_TIDL_SRC_ROOT "${PSDK_INSTALL_PATH}/c7x-mma-tidl/ti_dl")
        set(_TIDL_KERNELS_INC "${PSDK_INSTALL_PATH}/c7x-mma-tidl/tidl-kernels/include")
    endif()
    if(NOT EXISTS "${_TIDL_KERNELS_INC}/tidl_spatialMaxPool_ixX_oxX.h")
        message(FATAL_ERROR "TIDL kernel headers not found at: ${_TIDL_KERNELS_INC} "
                             "(tidl-kernels submodule not checked out/built?)")
    endif()
    target_sources(${PROJECT_NAME}.out PRIVATE
        ${TVM_DSP_RUNTIME_DIR}/mmalib/tidl_maxpool_wrapper.cpp
        # TIDL IALG wrapper (init/process/free_tidl_subgraph), exported to
        # DLOAD modules so a module build needs no TIDL or PSDK headers.
        ${TVM_DSP_RUNTIME_DIR}/tidl/tidl_api.c
        ${TVM_DSP_RUNTIME_DIR}/tidl/tidl_api_mem.c
        ${TVM_DSP_RUNTIME_DIR}/tidl/ti_mem_manager.c
    )
    # Compile these the way the module build did, or TIDL inference gets
    # slower:
    #  - --c99: the firmware's default C dialect rejects their C99 loops.
    #  - -O3 --auto_inline=500: the dynmod Release flags, not the firmware's -O2.
    #  - printf -> shm_printf: in a module, printf is resolved to the
    #    firmware's fast shm_printf (dyn_loader.c); compiled into the
    #    firmware it would bind to the RTS printf and its add_device stdout
    #    path instead, which is slow for the per-layer table printed after
    #    every inference.
    # tidl_api.c #undefs TIDL_MAX_NUM_CORES and sets 4 before itidl_ti.h,
    # which overrides the AM62D define below; on J722S itidl_ti.h then
    # redefines it to 2 from SOC_J722S.  That is exactly what the module
    # build produced, so leave both alone.
    set_source_files_properties(
        ${TVM_DSP_RUNTIME_DIR}/tidl/tidl_api.c
        ${TVM_DSP_RUNTIME_DIR}/tidl/tidl_api_mem.c
        ${TVM_DSP_RUNTIME_DIR}/tidl/ti_mem_manager.c
        PROPERTIES COMPILE_OPTIONS
            "--c99;-O3;--auto_inline=500;--preinclude=${CMAKE_CURRENT_SOURCE_DIR}/src/shm_printf.h;-Dprintf=shm_printf"
    )
    target_include_directories(${PROJECT_NAME}.out PRIVATE
        ${TVM_DSP_RUNTIME_DIR}/mmalib
        ${TVM_DSP_RUNTIME_DIR}/tidl
        # TIDL headers needed by tidl_maxpool_wrapper.cpp and tidl_api.c
        ${_TIDL_SRC_ROOT}/inc
        ${_TIDL_SRC_ROOT}/algo/inc
        # tidl_spatialMaxPool_ixX_oxX.h lives in the tidl-kernels submodule
        # (a sibling of ti_dl/, not nested under it) as of the AM62D tag's
        # library restructure -- it used to be at algo/src/tidsp/inc, which
        # no longer exists.
        ${_TIDL_KERNELS_INC}
        # ivision.h is included by itidl_ti.h (TI IVISION interface)
        ${PSDK_INSTALL_PATH}/ivision
    )
    target_compile_definitions(${PROJECT_NAME}.out PRIVATE USE_TIDL_RUNTIME=1)
    if(TVM_SOC STREQUAL "am62dx")
        # itidl_ti.h sets TIDL_MAX_NUM_CORES from SOC_AM62A / SOC_J722S, neither
        # of which the firmware defines for AM62D.  Use the value the AM62A
        # algo libs were built with (SOC_AM62A => 1) so the headers agree with
        # the linked libraries.
        target_compile_definitions(${PROJECT_NAME}.out PRIVATE TIDL_MAX_NUM_CORES=1U)
    endif()

    message(STATUS "TIDL Runtime: ENABLED")
    message(STATUS "  TIDL libs: ${TIDL_LIB_PATH}")
endif()

if(USE_TI_MMALIB)
    # PSDK_INSTALL_PATH / MMALIB_PATH resolved by boards.cmake (env override
    # still wins).
    set(MMALIB_LIB_PATH "${MMALIB_PATH}/lib/${_MMALIB_SILICON}/release")

    if(NOT EXISTS "${MMALIB_LIB_PATH}/mmalib_${_MMALIB_SILICON}.lib")
        message(FATAL_ERROR "MMALIB libraries not found at: ${MMALIB_LIB_PATH} "
                             "(MMALIB_PATH='${MMALIB_PATH}'). MMALIB_PATH is "
                             "derived from PSDK_INSTALL_PATH='${PSDK_INSTALL_PATH}' "
                             "unless set directly; set PSDK_INSTALL_PATH (or "
                             "MMALIB_PATH) via env/-D so it points at this board's "
                             "SDK.")
    endif()

    list(APPEND TIDL_RUNTIME_LIBS
        ${MMALIB_LIB_PATH}/mmalib_${_MMALIB_SILICON}.lib
        ${MMALIB_LIB_PATH}/mmalib_cn_${_MMALIB_SILICON}.lib
        ${MMALIB_LIB_PATH}/common_${_MMALIB_SILICON}.lib
    )
    if(TVM_SOC STREQUAL "am62dx")
        # MMALIB 12.x references *_exec_perf / *_exec_perfEst symbols that live
        # in perfEst_<silicon>.lib; the J722S (11.x) libs do not need it.
        list(APPEND TIDL_RUNTIME_LIBS ${MMALIB_LIB_PATH}/perfEst_${_MMALIB_SILICON}.lib)
    endif()
    # Force the linker to retain key MMALIB entry points that DLOAD modules
    # reference. Without --retain, the linker eliminates them as "unused"
    # since nothing in the firmware calls them directly.
    target_link_options(${PROJECT_NAME}.out PRIVATE
        --retain=mmalib_conv2d_i8
        --retain=mmalib_conv2d_i8_grouped_loop
        --retain=mmalib_conv2d_i16
        --retain=mmalib_matmul_i8
        --retain=mmalib_matmul_i16
    )

    # MMALIB wrappers: compiled into firmware, exported to DLOAD modules.
    # Resolve MMALIB include path from MMALIB_LIB_PATH (go up to root).
    get_filename_component(_MMALIB_ROOT "${MMALIB_LIB_PATH}/../../../" ABSOLUTE)
    target_sources(${PROJECT_NAME}.out PRIVATE
        ${TVM_DSP_RUNTIME_DIR}/mmalib/mmalib_wrappers.cpp
    )
    target_include_directories(${PROJECT_NAME}.out PRIVATE
        ${TVM_DSP_RUNTIME_DIR}/mmalib
        ${_MMALIB_ROOT}/ti/mmalib/src
        ${_MMALIB_ROOT}/ti/mmalib/src/common
    )
    target_compile_definitions(${PROJECT_NAME}.out PRIVATE USE_TI_MMALIB=1)

    message(STATUS "MMALIB: ENABLED")
    message(STATUS "  MMALIB libs: ${MMALIB_LIB_PATH}")
endif()

# Link libraries
# dmautils_standalone (from source, standalone UDMA) must come BEFORE
# drivers.lib so its UDMA symbols satisfy the runtime's references first,
# preventing drivers.lib from pulling in the full UDMA driver.
# c7x_int8_channel_scale_multiply/c7x_int8_concat_rescale are always-compiled
# native kernels (kernels/c7x_activation.cpp, kernels/c7x_concat.cpp) with no
# TIDL/MMALIB dependency -- retained unconditionally so they survive linker GC
# regardless of USE_TIDL_RUNTIME/USE_TI_MMALIB.
target_link_options(${PROJECT_NAME}.out PRIVATE
    --reread_libs
    --retain=c7x_int8_channel_scale_multiply
    --retain=c7x_int8_concat_rescale
)
target_link_libraries(${PROJECT_NAME}.out
    syscfg_generated
    ${MCUPSDK_ROOT}/source/kernel/freertos/lib/freertos.${DEVICE}.c75x.ti-c7000.${PROFILE}.lib
    ${TVM_DSP_RUNTIME_LIB}
    dmautils_standalone
    ${TIDL_RUNTIME_LIBS}
    ${MCUPSDK_ROOT}/source/drivers/lib/drivers.${DEVICE}.${CORE}.ti-c7000.${PROFILE}.lib
    ${TI_CGT_ROOT}/lib/libc.a
    ${LINKER_CMD}
)

# Post-build message
add_custom_command(TARGET ${PROJECT_NAME}.out POST_BUILD
    COMMAND ${CMAKE_COMMAND} -E echo "================================================"
    COMMAND ${CMAKE_COMMAND} -E echo "C7x Compute Service built: ${PROJECT_NAME}.out"
    COMMAND ${CMAKE_COMMAND} -E echo "Deploy with: ../../deploy-c7x.sh build/${PROJECT_NAME}.out"
    COMMAND ${CMAKE_COMMAND} -E echo "================================================"
)
