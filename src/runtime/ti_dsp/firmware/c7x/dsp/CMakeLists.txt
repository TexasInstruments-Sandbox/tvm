# C7x Compute Service - DSP Firmware
#
# Provides RPMessage-based compute service for host applications.
#
# Build: ./build.sh
# Output: c7x_compute.out (ELF firmware)

cmake_minimum_required(VERSION 3.16)

#
# Tool paths - update these if your installation differs
#
# Tool paths: prefer environment variables, fall back to developer defaults
if(DEFINED ENV{TI_SYSCONFIG_ROOT})
    set(TI_SYSCONFIG_ROOT "$ENV{TI_SYSCONFIG_ROOT}")
else()
    set(TI_SYSCONFIG_ROOT "$ENV{HOME}/ti/ccs2041/ccs/utils/sysconfig_1.26.0")
endif()

if(DEFINED ENV{TI_CGT_C7000_PATH})
    set(TI_CGT_ROOT "$ENV{TI_CGT_C7000_PATH}")
elseif(DEFINED ENV{TI_CGT_ROOT})
    set(TI_CGT_ROOT "$ENV{TI_CGT_ROOT}")
else()
    set(TI_CGT_ROOT "$ENV{HOME}/ti/ccs2040/ccs/tools/compiler/ti-cgt-c7000_5.0.1.LTS")
endif()

# Use clean SDK (11_00_00_06) which enables MMU without TLB_INV issues
if(DEFINED ENV{MCU_PLUS_SDK_PATH})
    set(MCUPSDK_ROOT "$ENV{MCU_PLUS_SDK_PATH}")
elseif(DEFINED ENV{PSDK_INSTALL_PATH})
    set(MCUPSDK_ROOT "$ENV{PSDK_INSTALL_PATH}/mcu_plus_sdk_j722s_11_00_00_12")
else()
    set(MCUPSDK_ROOT "$ENV{HOME}/ml/am67a/ti-processor-sdk-rtos-j722s-evm-11_00_00_06/mcu_plus_sdk_j722s_11_00_00_12")
endif()

# Device and core settings
set(DEVICE "j722s")
set(CORE "c75ss0-0")
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

# SysConfig input file
set(SYSCFG_FILE "${CMAKE_CURRENT_SOURCE_DIR}/configs/c75ss0.syscfg")

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
    SOC_J722S
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
set(LINKER_CMD "${CMAKE_CURRENT_SOURCE_DIR}/configs/linker_c75_freertos.cmd")

# TVM DSP Runtime (relative to firmware location within ti_dsp/)
set(TVM_DSP_RUNTIME_DIR "${CMAKE_CURRENT_SOURCE_DIR}/../../..")
set(TVM_DSP_RUNTIME_LIB "${TVM_DSP_RUNTIME_DIR}/build-c7x/libtvm_dsp_runtime_c7x.a")
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
    DMA_UTILS_STANDALONE SOC_J722S BUILD_C7X_1 BUILD_C7X MCU_PLUS_SDK
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
    SOC_J722S
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
    TVM_DSP_DEVICE_J722S
    TVM_DSP_NO_EXCEPTIONS=1
    TVM_DSP_NO_THREADS=1
    # DmaUtils headers need MCU_PLUS_SDK for correct include paths
    MCU_PLUS_SDK
)

# Link directories
target_link_directories(${PROJECT_NAME}.out PRIVATE
    ${TI_CGT_ROOT}/lib
)

# --- TIDL Runtime (optional) ---
# When enabled, links TIDL algo libraries and MMALIB into the firmware.
# The DLOAD model module then only needs tidl_api.lib (thin IALG wrapper).
option(USE_TIDL_RUNTIME "Link TIDL algo libs into firmware" OFF)
set(TIDL_RUNTIME_LIBS "")
if(USE_TIDL_RUNTIME)
    if(DEFINED ENV{PSDK_INSTALL_PATH})
        set(PSDK_PATH "$ENV{PSDK_INSTALL_PATH}" CACHE PATH "TI Processor SDK root")
    else()
        set(PSDK_PATH "$ENV{HOME}/ml/am67a/ti-processor-sdk-rtos-j722s-evm-11_00_00_06"
            CACHE PATH "TI Processor SDK root")
    endif()
    if(DEFINED ENV{C7X_MMA_TIDL_PATH})
        set(TIDL_LIB_PATH "$ENV{C7X_MMA_TIDL_PATH}/ti_dl/lib/J722S/dsp/algo/release")
    else()
        set(TIDL_LIB_PATH "${PSDK_PATH}/c7x-mma-tidl/ti_dl/lib/J722S/dsp/algo/release")
    endif()
    # MMALIB: prefer env var, then try versioned name under PSDK
    if(DEFINED ENV{MMALIB_PATH})
        set(MMALIB_LIB_PATH "$ENV{MMALIB_PATH}/lib/C7524/release")
    else()
        file(GLOB _MMALIB_DIRS "${PSDK_PATH}/mmalib*/lib/C7524/release")
        if(_MMALIB_DIRS)
            list(SORT _MMALIB_DIRS)
            list(GET _MMALIB_DIRS -1 MMALIB_LIB_PATH)
        else()
            set(MMALIB_LIB_PATH "${PSDK_PATH}/mmalib/lib/C7524/release")
        endif()
    endif()

    if(NOT EXISTS "${TIDL_LIB_PATH}/tidl_algo.lib")
        message(FATAL_ERROR "TIDL libraries not found at: ${TIDL_LIB_PATH}")
    endif()

    set(TIDL_RUNTIME_LIBS
        ${TIDL_LIB_PATH}/tidl_algo.lib
        ${TIDL_LIB_PATH}/tidl_obj_algo.lib
        ${TIDL_LIB_PATH}/tidl_priv_algo.lib
        ${TIDL_LIB_PATH}/tidl_custom.lib
        ${MMALIB_LIB_PATH}/mmalib_C7524.lib
        ${MMALIB_LIB_PATH}/mmalib_cn_C7524.lib
        ${MMALIB_LIB_PATH}/common_C7524.lib
    )
    # Force the linker to retain TIDL_VISION_FXNS (IALG function table)
    # and key TIDL algo entry points that DLOAD modules reference.
    # Without --retain, the linker eliminates them as "unused" since
    # nothing in the firmware calls them directly.
    target_link_options(${PROJECT_NAME}.out PRIVATE
        --retain=TIDL_VISION_FXNS
        --retain=mmalib_conv2d_i8
        --retain=mmalib_conv2d_i16
        --retain=mmalib_matmul_i8
        --retain=mmalib_matmul_i16
    )

    # MMALIB wrappers + TIDL max pool wrapper: compiled into firmware, exported
    # to DLOAD modules.  Both live in mmalib/ and require TIDL + MMALIB at link.
    # Resolve MMALIB include path from MMALIB_LIB_PATH (go up to root).
    get_filename_component(_MMALIB_ROOT "${MMALIB_LIB_PATH}/../../../" ABSOLUTE)
    # Resolve TIDL source root for include headers.  TIDL_LIB_PATH goes deep
    # into the lib tree so we derive it from C7X_MMA_TIDL_PATH directly.
    if(DEFINED ENV{C7X_MMA_TIDL_PATH})
        set(_TIDL_SRC_ROOT "$ENV{C7X_MMA_TIDL_PATH}/ti_dl")
    else()
        set(_TIDL_SRC_ROOT "${PSDK_PATH}/c7x-mma-tidl/ti_dl")
    endif()
    target_sources(${PROJECT_NAME}.out PRIVATE
        ${TVM_DSP_RUNTIME_DIR}/mmalib/mmalib_wrappers.cpp
        ${TVM_DSP_RUNTIME_DIR}/mmalib/tidl_maxpool_wrapper.cpp
    )
    target_include_directories(${PROJECT_NAME}.out PRIVATE
        ${TVM_DSP_RUNTIME_DIR}/mmalib
        ${_MMALIB_ROOT}/ti/mmalib/src
        ${_MMALIB_ROOT}/ti/mmalib/src/common
        # TIDL headers needed by tidl_maxpool_wrapper.cpp
        ${_TIDL_SRC_ROOT}/inc
        ${_TIDL_SRC_ROOT}/algo/inc
        ${_TIDL_SRC_ROOT}/algo/src/tidsp/inc
        # ivision.h is included by itidl_ti.h (TI IVISION interface)
        ${PSDK_PATH}/ivision
    )
    target_link_options(${PROJECT_NAME}.out PRIVATE
        --retain=c7x_int8_max_pool_tidl
    )

    message(STATUS "TIDL Runtime: ENABLED")
    message(STATUS "  TIDL libs: ${TIDL_LIB_PATH}")
    message(STATUS "  MMALIB: ${MMALIB_LIB_PATH}")
endif()

# Link libraries
# dmautils_standalone (from source, standalone UDMA) must come BEFORE
# drivers.lib so its UDMA symbols satisfy the runtime's references first,
# preventing drivers.lib from pulling in the full UDMA driver.
target_link_options(${PROJECT_NAME}.out PRIVATE --reread_libs)
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
