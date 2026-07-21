# Licensed to the Apache Software Foundation (ASF) under one
# or more contributor license agreements.  See the NOTICE file
# distributed with this work for additional information
# regarding copyright ownership.  The ASF licenses this file
# to you under the Apache License, Version 2.0 (the
# "License"); you may not use this file except in compliance
# with the License.  You may obtain a copy of the License at
#
#   http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing,
# software distributed under the License is distributed on an
# "AS IS" BASIS, WITHOUT WARRANTIES OR CONDITIONS OF ANY
# KIND, either express or implied.  See the License for the
# specific language governing permissions and limitations
# under the License.

#
# CMakeLists.txt for TVM C Static DSP Integration
#
# Builds TVM-generated code against the DSP runtime.
# Supports PC host emulation, AWRL6844 C66x, C7x host emulation, and
# C7x DLOAD dynamic modules.
#
# Usage:
#   Host emulation (Debug - default):
#     mkdir build && cd build
#     cmake ..
#     cmake --build .
#     ./cg_dsp
#
#   Host emulation (Release with -O3):
#     mkdir build-release && cd build-release
#     cmake -DCMAKE_BUILD_TYPE=Release ..
#     cmake --build .
#
#   C66x cross-compilation (Debug - default):
#     mkdir build-awrl6844 && cd build-awrl6844
#     cmake -DCMAKE_TOOLCHAIN_FILE=$TVM_HOME/src/runtime/ti_dsp/cmake/toolchain-awrl6844.cmake ..
#     cmake --build .
#
#   C66x cross-compilation (Release with -O3):
#     mkdir build-awrl6844-release && cd build-awrl6844-release
#     cmake -DCMAKE_TOOLCHAIN_FILE=$TVM_HOME/src/runtime/ti_dsp/cmake/toolchain-awrl6844.cmake \
#           -DCMAKE_BUILD_TYPE=Release ..
#     cmake --build .
#

cmake_minimum_required(VERSION 3.16)

project(CStaticDSP LANGUAGES C CXX)

# === Build Type Configuration ===
# Set default build type to Debug if not specified
if(NOT CMAKE_BUILD_TYPE)
    set(CMAKE_BUILD_TYPE "Debug" CACHE STRING "Build type (Debug/Release/RelWithDebInfo)" FORCE)
endif()
message(STATUS "Build type: ${CMAKE_BUILD_TYPE}")

# === Target Selection ===
set(TVM_DSP_TARGET "host" CACHE STRING "DSP target: host, c66x, c7x_host, c7x-dynmod")
set(TVM_DSP_DEVICE "" CACHE STRING "Device variant (e.g., awrl6844, j722s)")

# Allow overriding target via C7X_DYNMOD option (since toolchain forces TVM_DSP_TARGET)
option(C7X_DYNMOD "Build as DLOAD dynamic module" OFF)
if(C7X_DYNMOD)
    set(TVM_DSP_TARGET "c7x-dynmod")
endif()

# === Model Configuration ===
set(MODEL_ENTRY_FUNCTION "main" CACHE STRING "Model entry function name")
set(MODEL_NUM_INPUTS "1" CACHE STRING "Number of model inputs")
option(MODEL_RETURNS_TUPLE "Model returns tuple" ON)

# === TIDL Bridge (optional) ===
# Extra C source files for TIDL bridge functions (generated per model)
set(TIDL_BRIDGE_SOURCES "" CACHE STRING "Extra .c files for TIDL bridge")

# === TIDL Integration (optional) ===
# When enabled, links TIDL algo libraries and TIDL API source files into
# the model DLOAD module.  Only used with c7x-dynmod target.
option(USE_TIDL "Link TIDL libraries into model module" OFF)
set(PSDK_PATH "/opt/ti/am67a/ti-processor-sdk-rtos-j722s-evm-11_00_00_06"
    CACHE PATH "Path to TI Processor SDK (for TIDL/MMALIB libs and headers)")
# === Tilde expansion helper ===
# CMake doesn't expand ~ in paths read from environment variables.
macro(expand_tilde VAR)
    if("${${VAR}}" MATCHES "^~")
        string(REGEX REPLACE "^~" "$ENV{HOME}" ${VAR} "${${VAR}}")
    endif()
endmacro()

# === Path Configuration ===
# TVM and DSP runtime paths
if(APPLE)
    set(TVM_HOME "$ENV{HOME}/work/tvm" CACHE PATH "Path to TVM")
elseif(UNIX)
    set(TVM_HOME "$ENV{HOME}/ml/tvm" CACHE PATH "Path to TVM")
endif()

# Allow environment override
if(DEFINED ENV{TVM_HOME})
    set(TVM_HOME "$ENV{TVM_HOME}" CACHE PATH "Path to TVM" FORCE)
    expand_tilde(TVM_HOME)
endif()

set(DSP_RUNTIME_DIR "${TVM_HOME}/src/runtime/ti_dsp")
set(TIDL_API_DIR "${DSP_RUNTIME_DIR}/tidl")

# Generated code directory (parent directory by default)
set(GENERATED_CODE_DIR "${CMAKE_CURRENT_SOURCE_DIR}/../cstatic-tests" CACHE PATH "Generated code directory")

message(STATUS "TVM DSP Integration Configuration:")
message(STATUS "  Target: ${TVM_DSP_TARGET}")
message(STATUS "  Device: ${TVM_DSP_DEVICE}")
message(STATUS "  TVM_HOME: ${TVM_HOME}")
message(STATUS "  Generated code: ${GENERATED_CODE_DIR}")

# === Build Configuration ===
set(CMAKE_C_STANDARD 99)
set(CMAKE_CXX_STANDARD 11)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_EXPORT_COMPILE_COMMANDS ON)

# === Compile Definitions ===
add_compile_definitions(
    MODEL_ENTRY_FUNCTION=${MODEL_ENTRY_FUNCTION}
    MODEL_NUM_INPUTS=${MODEL_NUM_INPUTS}
)
if(MODEL_RETURNS_TUPLE)
    add_compile_definitions(MODEL_RETURNS_TUPLE)
endif()

# === Include Directories ===
set(INCLUDE_DIRS
    ${CMAKE_CURRENT_SOURCE_DIR}  # Local io/ directory
    ${DSP_RUNTIME_DIR}           # Runtime root: include/, platform/, container/, etc.
    ${TVM_HOME}/3rdparty/tvm-ffi/3rdparty/dlpack/include
)

# === Weights file path (used by both standalone and dynmod builds) ===
set(WEIGHTS_FILE "${GENERATED_CODE_DIR}/weights.bin" CACHE FILEPATH "Path to weights.bin")

# === Weight Embedding Support ===
if(NOT C7X_DYNMOD)
    list(APPEND CMAKE_MODULE_PATH "${DSP_RUNTIME_DIR}/cmake")
    include(WeightEmbedding)

    # === Source Files ===
    # Note: Constants loader (TVMDSPParseConstants, TVMGetConstants) is now
    # provided by the TVM DSP runtime library (constants/constants_loader.cpp)
    # weights_loader.cpp provides the weights data source (filesystem or embedded)
    set(DSP_SOURCES
        main_dsp.cpp
        io/tensor_file.cpp
        io/weights_loader.cpp
    )

    # Generated code (lib*.c + optional TIDL bridge)
    file(GLOB GENERATED_SOURCES "${GENERATED_CODE_DIR}/lib*.c")
    if(NOT GENERATED_SOURCES)
        message(WARNING "No generated code (lib*.c) found in ${GENERATED_CODE_DIR}")
        message(WARNING "Run TVM compilation first to generate lib0.c")
    endif()
    list(SORT GENERATED_SOURCES)
    if(TIDL_BRIDGE_SOURCES)
        list(APPEND GENERATED_SOURCES ${TIDL_BRIDGE_SOURCES})
        # Generate -include flag for the bridge header so lib*.c can
        # resolve tidl_subgraph_N_process() forward declarations.
        # When TIDL_BRIDGE_SOURCES is a list (bridge.c + artifacts.c),
        # only the first element has a matching .h header.
        list(GET TIDL_BRIDGE_SOURCES 0 _FIRST_BRIDGE_SRC)
        string(REPLACE ".c" ".h" TIDL_BRIDGE_HEADER "${_FIRST_BRIDGE_SRC}")
        if(EXISTS "${TIDL_BRIDGE_HEADER}")
            add_compile_options("-include" "${TIDL_BRIDGE_HEADER}")
        endif()
        message(STATUS "  TIDL bridge: ${TIDL_BRIDGE_SOURCES}")
    endif()
endif()

# === Target-Specific Configuration ===
if(TVM_DSP_TARGET STREQUAL "host")
    # =========================================
    # Host Emulation Build
    # =========================================
    add_compile_definitions(TVM_DSP_TARGET_HOST)

    # Use same code path as C66x to debug alignment/format issues
    add_compile_definitions(TVM_DSP_SKIP_CG_MAIN)

    # Compiler flags based on build type
    if(CMAKE_BUILD_TYPE STREQUAL "Release")
        set(CMAKE_C_FLAGS "${CMAKE_C_FLAGS} -O3 -DNDEBUG")
        set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -O3 -DNDEBUG")
        message(STATUS "  Optimization: Release (-O3)")
    else()
        set(CMAKE_C_FLAGS "${CMAKE_C_FLAGS} -g -O0")
        set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -g -O0")
        message(STATUS "  Optimization: Debug (-g -O0)")
    endif()

    # Suppress warnings for generated code
    set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -w")

    # DSP runtime library
    set(DSP_RUNTIME_BUILD_DIR "${DSP_RUNTIME_DIR}/build-c66x-host")
    set(DSP_RUNTIME_LIB "${DSP_RUNTIME_BUILD_DIR}/libtvm_dsp_runtime_host.a")

    if(NOT EXISTS "${DSP_RUNTIME_LIB}")
        message(FATAL_ERROR "DSP runtime library not found: ${DSP_RUNTIME_LIB}\n"
                "Build it first: cd ${DSP_RUNTIME_DIR}/build && cmake .. && make")
    endif()

    # Create executable
    add_executable(cg_dsp ${DSP_SOURCES} ${GENERATED_SOURCES})

    # Force .c files to compile as C++
    set_source_files_properties(${GENERATED_SOURCES} PROPERTIES LANGUAGE CXX)

    target_include_directories(cg_dsp PRIVATE ${INCLUDE_DIRS})
    target_link_libraries(cg_dsp PRIVATE ${DSP_RUNTIME_LIB})

    # Configure weights loading - use filesystem (same parser as C66x)
    if(EXISTS "${WEIGHTS_FILE}")
        target_compile_definitions(cg_dsp PRIVATE
            TVM_DSP_WEIGHTS_PATH="${WEIGHTS_FILE}"
        )
        message(STATUS "  Weights: ${WEIGHTS_FILE} (filesystem)")
    else()
        message(WARNING "Weights file not found: ${WEIGHTS_FILE}")
    endif()

    message(STATUS "Host emulation build configured")
    message(STATUS "  Executable: cg_dsp")
    message(STATUS "  Runtime: ${DSP_RUNTIME_LIB}")

elseif(TVM_DSP_TARGET STREQUAL "c66x")
    # =========================================
    # C66x Cross-Compilation Build
    # =========================================
    add_compile_definitions(TVM_DSP_TARGET_C66X)

    if(TVM_DSP_DEVICE STREQUAL "awrl6844")
        add_compile_definitions(TVM_DSP_DEVICE_AWRL6844)
    endif()

    # C66x compiler flags - optimization flags are set in toolchain file per build type:
    #   Debug:   -g --opt_level=0
    #   Release: -O3 --opt_for_speed=5
    # Keep assembly output for both Debug and Release builds
    # --debug_software_pipeline emits SW pipelining info to .asm files
    set(CMAKE_C_FLAGS "${CMAKE_C_FLAGS} -k --debug_software_pipeline")
    set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -k --debug_software_pipeline")

    if(CMAKE_BUILD_TYPE STREQUAL "Release")
        message(STATUS "  Optimization: Release (-O3 --opt_for_speed=5)")
    else()
        message(STATUS "  Optimization: Debug (-g --opt_level=0)")
    endif()

    # DSP runtime library - select debug or release build
    if(CMAKE_BUILD_TYPE STREQUAL "Release")
        set(DSP_RUNTIME_BUILD_DIR "${DSP_RUNTIME_DIR}/build-c66x-release")
        # Fall back to regular build if release doesn't exist
        if(NOT EXISTS "${DSP_RUNTIME_BUILD_DIR}")
            set(DSP_RUNTIME_BUILD_DIR "${DSP_RUNTIME_DIR}/build-c66x")
        endif()
    else()
        set(DSP_RUNTIME_BUILD_DIR "${DSP_RUNTIME_DIR}/build-c66x")
    endif()
    set(DSP_RUNTIME_LIB "${DSP_RUNTIME_BUILD_DIR}/libtvm_dsp_runtime_c66x.a")

    if(NOT EXISTS "${DSP_RUNTIME_LIB}")
        message(FATAL_ERROR "DSP runtime library not found: ${DSP_RUNTIME_LIB}\n"
                "Build it first with C66x toolchain")
    endif()

    # Board support (local awrl6844 directory)
    set(BOARD_SUPPORT_DIR "${CMAKE_CURRENT_SOURCE_DIR}/awrl6844")
    if(EXISTS "${BOARD_SUPPORT_DIR}")
        file(GLOB AWRL6844_SYSCFG_SRCS "${BOARD_SUPPORT_DIR}/syscfg/*.c")
        set(BOARD_SOURCES
            ${BOARD_SUPPORT_DIR}/board_init.c
            ${AWRL6844_SYSCFG_SRCS}
        )
        list(APPEND INCLUDE_DIRS
            ${BOARD_SUPPORT_DIR}
            ${BOARD_SUPPORT_DIR}/syscfg
        )
    endif()

    # MCU+ SDK includes
    if(DEFINED MMWAVE_SDK_PATH AND EXISTS "${MMWAVE_SDK_PATH}")
        list(APPEND INCLUDE_DIRS
            ${MMWAVE_SDK_PATH}/source
            ${MMWAVE_SDK_PATH}/source/kernel/nortos
            ${MMWAVE_SDK_PATH}/source/kernel/nortos/dpl/c66
            ${MMWAVE_SDK_PATH}/source/drivers
            ${MMWAVE_SDK_PATH}/source/board
        )
    endif()

    # C66x defines
    add_compile_definitions(
        CORE_DSP
        _TMS320C6600
        TMS320C66X
        TARGET_C66X_AWRL6844
        SOC_XWRL684X
    )

    # Create executable
    add_executable(cg_dsp_c66x
        ${DSP_SOURCES}
        ${GENERATED_SOURCES}
        ${BOARD_SOURCES}
    )

    # Force .c files to compile as C++
    set_source_files_properties(${GENERATED_SOURCES} PROPERTIES LANGUAGE CXX)

    target_include_directories(cg_dsp_c66x PRIVATE ${INCLUDE_DIRS})
    target_link_libraries(cg_dsp_c66x PRIVATE ${DSP_RUNTIME_LIB})

    # Linker configuration
    set(LINKER_CMD_FILE "${BOARD_SUPPORT_DIR}/linker_c66x.cmd")
    set(C66X_LINK_FLAGS "-cr -x")
    set(C66X_LINK_FLAGS "${C66X_LINK_FLAGS} -m ${CMAKE_CURRENT_BINARY_DIR}/cg_dsp_c66x.map")
    set(C66X_LINK_FLAGS "${C66X_LINK_FLAGS} ${LINKER_CMD_FILE}")
    set(C66X_LINK_FLAGS "${C66X_LINK_FLAGS} -l${TI_CGT_C6000_PATH}/lib/rts6600_elf.lib")
    set(C66X_LINK_FLAGS "${C66X_LINK_FLAGS} -l${TI_CGT_C6000_PATH}/lib/libc.a")

    # SDK libraries - select debug or release based on build type
    if(DEFINED MMWAVE_SDK_PATH AND EXISTS "${MMWAVE_SDK_PATH}")
        if(CMAKE_BUILD_TYPE STREQUAL "Release")
            set(SDK_PROFILE "release")
        else()
            set(SDK_PROFILE "debug")
        endif()

        set(SDK_NORTOS_LIB "${MMWAVE_SDK_PATH}/source/kernel/nortos/lib")
        set(SDK_DRIVERS_LIB "${MMWAVE_SDK_PATH}/source/drivers/lib")
        set(SDK_BOARD_LIB "${MMWAVE_SDK_PATH}/source/board/lib")

        set(C66X_LINK_FLAGS "${C66X_LINK_FLAGS} -i${SDK_NORTOS_LIB}")
        set(C66X_LINK_FLAGS "${C66X_LINK_FLAGS} -i${SDK_DRIVERS_LIB}")
        set(C66X_LINK_FLAGS "${C66X_LINK_FLAGS} -i${SDK_BOARD_LIB}")
        set(C66X_LINK_FLAGS "${C66X_LINK_FLAGS} -lnortos.xwrL684x.c66.ti-c6000.${SDK_PROFILE}.lib")
        set(C66X_LINK_FLAGS "${C66X_LINK_FLAGS} -ldrivers.xwrL684x.c66.ti-c6000.${SDK_PROFILE}.lib")
        set(C66X_LINK_FLAGS "${C66X_LINK_FLAGS} -lboard.xwrL684x.c66.ti-c6000.${SDK_PROFILE}.lib")

        message(STATUS "  SDK libraries: ${SDK_PROFILE}")
    endif()

    set_target_properties(cg_dsp_c66x PROPERTIES
        LINK_FLAGS "${C66X_LINK_FLAGS}"
        SUFFIX ".out"
    )

    # Embed weights for C66x
    if(EXISTS "${WEIGHTS_FILE}")
        tvm_dsp_embed_weights(cg_dsp_c66x "${WEIGHTS_FILE}")
        target_compile_definitions(cg_dsp_c66x PRIVATE TVM_DSP_WEIGHTS_EMBEDDED)
        message(STATUS "  Weights: ${WEIGHTS_FILE} (embedded)")
    else()
        message(WARNING "Weights file not found: ${WEIGHTS_FILE}")
    endif()

    message(STATUS "C66x cross-compilation build configured")
    message(STATUS "  Executable: cg_dsp_c66x.out")
    message(STATUS "  Runtime: ${DSP_RUNTIME_LIB}")

elseif(TVM_DSP_TARGET STREQUAL "c7x_host")
    # =========================================
    # C7x Host Emulation Build
    # =========================================
    # Uses TI C7000 Host Emulation library to compile C7x-targeted code
    # with system g++ and run on x86-64 Linux. This enables local testing
    # of c_static -mcpu=c7x codegen without C7x hardware.
    add_compile_definitions(TVM_DSP_TARGET_C7X)
    add_compile_definitions(TVM_DSP_DEVICE_J722S)

    # Use same code path as hardware C7x (skip cg_main pattern)
    add_compile_definitions(TVM_DSP_SKIP_CG_MAIN)

    # Distinguish host emulation from real C7x hardware in generated code.
    # TI Host Emu headers define __C7000__ and __TI_COMPILER_VERSION__,
    # so generated code checks !defined(C7X_HOST_EMULATION) to enable
    # hardware-specific paths (e.g., hardcoded L2 SRAM base address).
    add_compile_definitions(C7X_HOST_EMULATION)

    # TI C7000 Host Emulation library
    if(DEFINED ENV{TI_CGT_C7000_PATH})
        set(TI_CGT_C7000_PATH "$ENV{TI_CGT_C7000_PATH}")
        expand_tilde(TI_CGT_C7000_PATH)
    endif()
    if(NOT DEFINED TI_CGT_C7000_PATH)
        message(FATAL_ERROR "TI_CGT_C7000_PATH is not set. "
                "Set via environment variable or -DTI_CGT_C7000_PATH=...")
    endif()
    set(C7X_HOST_EMU_DIR "${TI_CGT_C7000_PATH}/host_emulation")
    set(C7X_HOST_EMU_INCLUDE "${C7X_HOST_EMU_DIR}/include/C7524-MMA2_256")
    set(C7X_HOST_EMU_LIB "${C7X_HOST_EMU_DIR}/libC7524-MMA2_256-host-emulation.a")

    if(NOT EXISTS "${C7X_HOST_EMU_INCLUDE}/c7x.h")
        message(FATAL_ERROR "TI Host Emulation headers not found: ${C7X_HOST_EMU_INCLUDE}/c7x.h")
    endif()
    if(NOT EXISTS "${C7X_HOST_EMU_LIB}")
        message(FATAL_ERROR "TI Host Emulation library not found: ${C7X_HOST_EMU_LIB}")
    endif()

    # C++14 required by TI Host Emulation headers
    set(CMAKE_CXX_STANDARD 14)
    set(CMAKE_CXX_STANDARD_REQUIRED ON)

    # Compiler flags
    if(CMAKE_BUILD_TYPE STREQUAL "Release")
        set(CMAKE_C_FLAGS "${CMAKE_C_FLAGS} -O3 -DNDEBUG")
        set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -O3 -DNDEBUG")
        message(STATUS "  Optimization: Release (-O3)")
    else()
        set(CMAKE_C_FLAGS "${CMAKE_C_FLAGS} -g -O0")
        set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -g -O0")
        message(STATUS "  Optimization: Debug (-g -O0)")
    endif()

    # -fno-strict-aliasing required by TI Host Emulation
    # -include c7x.h: force-include the Host Emulation c7x.h so that
    # TI C7x intrinsics (__max, __min, etc.) generated by c_static -mcpu=c7x
    # are available when compiling with g++.
    # The Host Emulation transitively defines __TI_COMPILER_VERSION__ (via
    # ti_he_impl/version.h), which causes the generated code to enable
    # __declspec(dllexport) -- a TI linker feature not supported by g++.
    # We neutralize it with -D'__declspec(x)=' so it becomes a no-op.
    set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -fno-strict-aliasing -w -include c7x.h \"-D__declspec(x)=\"")

    # DSP runtime library (c7x_host build)
    set(DSP_RUNTIME_BUILD_DIR "${DSP_RUNTIME_DIR}/build-c7x-host")
    set(DSP_RUNTIME_LIB "${DSP_RUNTIME_BUILD_DIR}/libtvm_dsp_runtime_c7x_host.a")

    if(NOT EXISTS "${DSP_RUNTIME_LIB}")
        message(FATAL_ERROR "DSP runtime library not found: ${DSP_RUNTIME_LIB}\n"
                "Build it first:\n"
                "  cd ${DSP_RUNTIME_DIR}\n"
                "  mkdir build-c7x-host && cd build-c7x-host\n"
                "  cmake -DTVM_DSP_TARGET=c7x_host ..\n"
                "  cmake --build .")
    endif()

    # Include directories (add Host Emulation headers + DMA)
    list(APPEND INCLUDE_DIRS ${C7X_HOST_EMU_INCLUDE})
    list(APPEND INCLUDE_DIRS ${DSP_RUNTIME_DIR}/dma)

    # DMA runtime (host emulation stub: memcpy)
    list(APPEND DSP_SOURCES ${DSP_RUNTIME_DIR}/dma/tvm_dsp_dma_host.c)

    # TIDL max pool wrapper (c7x_int8_max_pool_tidl). Needed whenever a model
    # has a max_pool2d that FuseQDQToTIDLMaxPool lowers, independent of
    # --mmalib/-DUSE_TIDL -- unlike those, this is not optional: on real
    # firmware this file is always compiled in (see its own file comment).
    # Under C7X_HOST_EMULATION (added above), the file itself falls back to
    # the portable scalar c7x_int8_max_pool -- see tidl_maxpool_wrapper.cpp's
    # TIDL_MAXPOOL_USE_TIDL_KERNEL guard -- so no TIDL headers/libs are
    # needed here, just the source file and its own include directory.
    list(APPEND DSP_SOURCES ${DSP_RUNTIME_DIR}/mmalib/tidl_maxpool_wrapper.cpp)
    list(APPEND INCLUDE_DIRS ${DSP_RUNTIME_DIR}/mmalib)

    # MMALIB integration (optional).
    # libmmalib_x86_64.a provides the x86-64 AVX reference implementation of
    # the MMA kernels used by both TVM's MMALIB direct integration and TIDL's
    # internal convolution path.  Always link when present.
    # mmalib_wrappers.cpp is the TVM-side call layer (MMALIB ops invoked from
    # TVM generated code).  Skip it under USE_TIDL: in the TIDL path TVM code
    # never calls MMALIB directly — TIDL calls it internally — so the wrapper
    # would introduce a duplicate symbol conflict.
    set(MMALIB_ROOT "/opt/ti/am67a/ti-processor-sdk-rtos-j722s-evm-11_00_00_06/mmalib_11_02_00_06")
    if(DEFINED ENV{MMALIB_PATH})
        set(MMALIB_ROOT "$ENV{MMALIB_PATH}")
    endif()
    if(EXISTS "${MMALIB_ROOT}/ti/mmalib/src/mmalib.h")
        message(STATUS "  MMALIB: ${MMALIB_ROOT}")
        list(APPEND INCLUDE_DIRS "${MMALIB_ROOT}/ti/mmalib/src")
        list(APPEND INCLUDE_DIRS "${MMALIB_ROOT}/ti/mmalib/src/common")
        list(APPEND INCLUDE_DIRS "${DSP_RUNTIME_DIR}/mmalib")
        if(NOT USE_TIDL)
            list(APPEND DSP_SOURCES "${DSP_RUNTIME_DIR}/mmalib/mmalib_wrappers.cpp")
        endif()
        set(MMALIB_LIBS
            "${MMALIB_ROOT}/lib/C7524/release/libmmalib_x86_64.a"
            "${MMALIB_ROOT}/lib/C7524/release/libmmalib_cn_x86_64.a"
            "${MMALIB_ROOT}/lib/C7524/release/libcommon_x86_64.a"
        )
        add_compile_definitions(_HOST_BUILD __C7524__ __C7X_MMA_2_256__)
        if(NOT USE_TIDL)
            set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -include mmalib_wrappers.h")
        endif()
    else()
        message(STATUS "  MMALIB: not found (skipping)")
        set(MMALIB_LIBS "")
    endif()

    # TIDL integration for c7x_host (optional, enabled by -DUSE_TIDL=ON).
    # Links the PC x86-64 TIDL algo libs (AVX reference path) and compiles
    # tidl_api.c with -DHOST_EMULATION so TIDL uses the REF_ONLY+AVX_REF
    # flowCtrl and skips hardware-specific code paths (DMA, MMA co-processor).
    # Net/io artifacts are embedded via generate_artifacts_c() in Python and
    # passed as extra .c files through TIDL_BRIDGE_SOURCES, not here.
    set(TIDL_HOST_LIBS "")
    if(USE_TIDL)
        set(C7X_MMA_TIDL_PATH "$ENV{C7X_MMA_TIDL_PATH}")
        if(NOT C7X_MMA_TIDL_PATH)
            set(C7X_MMA_TIDL_PATH "$ENV{HOME}/ml/c7x-mma-tidl")
        endif()
        set(TIDL_PC_LIB_DIR "${C7X_MMA_TIDL_PATH}/ti_dl/lib/J722S/PC/algo/release")
        set(TIDL_INC_PATH   "${C7X_MMA_TIDL_PATH}/ti_dl/inc")
        if(EXISTS "${TIDL_PC_LIB_DIR}/libtidl_algo.a" AND EXISTS "${TIDL_INC_PATH}/itidl_ti.h")
            message(STATUS "  TIDL (c7x_host): PC libs from ${TIDL_PC_LIB_DIR}")
            list(APPEND DSP_SOURCES ${TIDL_API_DIR}/tidl_api.c)
            list(APPEND DSP_SOURCES ${TIDL_API_DIR}/tidl_api_mem.c)
            list(APPEND DSP_SOURCES ${TIDL_API_DIR}/ti_mem_manager.c)
            list(APPEND INCLUDE_DIRS
                ${TIDL_API_DIR}
                ${TIDL_INC_PATH}
                ${DSP_RUNTIME_DIR}/firmware/c7x/common
            )
            # ivision.h: TI VISION algorithm interface header required by TIDL
            # algo libs.  Distributed with PSDK but not with c7x-mma-tidl.
            # MMALIB and ivision are co-installed under the same PSDK root, so
            # derive the path from MMALIB_ROOT's parent rather than requiring a
            # separate env variable.  PSDK_INSTALL_PATH overrides if set.
            if(NOT "$ENV{PSDK_INSTALL_PATH}" STREQUAL "")
                set(_IVISION_INC "$ENV{PSDK_INSTALL_PATH}/ivision")
            else()
                get_filename_component(_PSDK_ROOT "${MMALIB_ROOT}" DIRECTORY)
                set(_IVISION_INC "${_PSDK_ROOT}/ivision")
            endif()
            if(EXISTS "${_IVISION_INC}/ivision.h")
                list(APPEND INCLUDE_DIRS ${_IVISION_INC})
            else()
                message(FATAL_ERROR "ivision.h not found at ${_IVISION_INC}\n"
                        "Set PSDK_INSTALL_PATH to the PSDK root directory.")
            endif()
            # TIDL_MAX_NUM_CORES=4: the c7x-mma-tidl import tool writes
            # sTIDL_IOBufDesc_t arrays sized with TIDL_IO_MAX_NUM_CORES=4.
            # The PSDK header uses SOC-specific TIDL_MAX_NUM_CORES (2 for
            # J722S), causing a struct layout mismatch when reading io.bin.
            # Force 4 here to match the import tool's layout.
            add_compile_definitions(TIDL_MAX_NUM_CORES=4 HOST_EMULATION)
            # Host stubs for firmware-provided symbols that TIDL algo expects
            # at runtime (appMemAlloc/Free, TVM_cacheWbInvRegion, interrupt
            # lock/unlock, DMA no-ops).  Equivalent to tidl_firmware_syms.c
            # on the DLOAD path, but without __declspec(dllexport) syntax.
            # appUdmaGetObj is intentionally absent: under HOST_EMULATION +
            # REF_ONLY, TIDL's algInit skips the NULL UDMA check, so no stub
            # is needed.  The bridge passes NULL for the UDMA handle.
            list(APPEND DSP_SOURCES ${TIDL_API_DIR}/tidl_host_stubs.c)
            set(TIDL_HOST_LIBS
                "${TIDL_PC_LIB_DIR}/libtidl_algo.a"
                "${TIDL_PC_LIB_DIR}/libtidl_priv_algo.a"
                "${TIDL_PC_LIB_DIR}/libtidl_avx_kernels.a"
                "${TIDL_PC_LIB_DIR}/libtidl_custom.a"
                "${TIDL_PC_LIB_DIR}/libtidl_obj_algo.a"
            )
            # circularBlockSizeTbl is defined in both libtidl_algo.a and
            # libcommon_x86_64.a.  Both libs are required (TIDL algo calls
            # MMALIB internally), and the definitions are identical, so
            # --allow-multiple-definition is safe.
            set(TIDL_LINK_FLAGS "-Wl,--allow-multiple-definition")
        else()
            message(WARNING "TIDL PC libs not found at ${TIDL_PC_LIB_DIR} -- USE_TIDL ignored")
        endif()
    endif()

    # Create executable
    add_executable(cg_dsp ${DSP_SOURCES} ${GENERATED_SOURCES})

    # Force .c files to compile as C++ (required for Host Emulation headers)
    set_source_files_properties(${GENERATED_SOURCES} PROPERTIES LANGUAGE CXX)

    target_include_directories(cg_dsp PRIVATE ${INCLUDE_DIRS})
    # TIDL_HOST_LIBS before MMALIB_LIBS: TIDL algo references MMALIB symbols,
    # so MMALIB must appear after TIDL in the link order.
    target_link_libraries(cg_dsp PRIVATE ${DSP_RUNTIME_LIB} ${TIDL_HOST_LIBS} ${MMALIB_LIBS} ${C7X_HOST_EMU_LIB})
    target_link_libraries(cg_dsp PRIVATE pthread)
    if(TIDL_LINK_FLAGS)
        target_link_options(cg_dsp PRIVATE ${TIDL_LINK_FLAGS})
    endif()

    # Configure weights loading - use filesystem
    if(EXISTS "${WEIGHTS_FILE}")
        target_compile_definitions(cg_dsp PRIVATE
            TVM_DSP_WEIGHTS_PATH="${WEIGHTS_FILE}"
        )
        message(STATUS "  Weights: ${WEIGHTS_FILE} (filesystem)")
    else()
        message(WARNING "Weights file not found: ${WEIGHTS_FILE}")
    endif()

    message(STATUS "C7x host emulation build configured")
    message(STATUS "  Executable: cg_dsp")
    message(STATUS "  Runtime: ${DSP_RUNTIME_LIB}")
    message(STATUS "  Host Emulation: ${C7X_HOST_EMU_LIB}")
    if(MMALIB_LIBS)
        message(STATUS "  MMALIB: linked")
    endif()

elseif(TVM_DSP_TARGET STREQUAL "c7x-dynmod")
    # =========================================
    # C7x Dynamic Module Build (DLOAD-compatible)
    # =========================================
    # Builds lib0.c as a relocatable ELF module that can be loaded at runtime
    # by the c7x_compute firmware's DLOAD dynamic loader.
    #
    # Two-stage link (same pattern as neo-tvm TIDL deployment):
    #   Stage 1: dsp_syms.c → dsp_syms.out (pseudo-firmware symbol table)
    #   Stage 2: lib0.c + dsp_syms.out → lib0.out (relocatable module)
    #
    # Usage:
    #   cmake -DCMAKE_TOOLCHAIN_FILE=<path>/toolchain-j722s-c7x.cmake \
    #         -DTVM_DSP_TARGET=c7x-dynmod ..
    #   cmake --build .

    # Dynmod infrastructure directory (canonical location in src/runtime/ti_dsp/dynmod)
    set(DYNMOD_DIR "${DSP_RUNTIME_DIR}/dynmod/c7x_dynmod")

    # Compiler flags — same as standalone c7x but no -k (no asm output needed)
    if(CMAKE_BUILD_TYPE STREQUAL "Release")
        message(STATUS "  Optimization: Release (-O2 --auto_inline=500)")
    else()
        message(STATUS "  Optimization: Debug (-g --opt_level=0)")
    endif()

    # Check for generated code
    file(GLOB LIB_SRCS "${GENERATED_CODE_DIR}/lib*.c")
    if(NOT LIB_SRCS)
        message(FATAL_ERROR "No generated code (lib*.c) found in ${GENERATED_CODE_DIR}\n"
                "Run TVM compilation first to generate lib0.c")
    endif()
    list(SORT LIB_SRCS)

    # Split flag strings into lists for use in add_custom_command.
    # Must include both base flags and build-type-specific flags since
    # add_custom_command doesn't automatically append CMAKE_*_FLAGS_<CONFIG>.
    string(TOUPPER "${CMAKE_BUILD_TYPE}" _BUILD_TYPE_UPPER)
    set(_C_ALL_FLAGS "${CMAKE_C_FLAGS} ${CMAKE_C_FLAGS_${_BUILD_TYPE_UPPER}}")
    set(_CXX_ALL_FLAGS "${CMAKE_CXX_FLAGS} ${CMAKE_CXX_FLAGS_${_BUILD_TYPE_UPPER}}")
    separate_arguments(C_FLAGS_LIST NATIVE_COMMAND "${_C_ALL_FLAGS}")
    separate_arguments(CXX_FLAGS_LIST NATIVE_COMMAND "${_CXX_ALL_FLAGS}")

    # --- Stage 1: Build dsp_syms.out (pseudo-firmware symbol table) ---

    # Compile dsp_syms.c → dsp_syms.obj
    set(DSP_SYMS_SRC "${DYNMOD_DIR}/dsp_syms.c")
    set(DSP_SYMS_OBJ "${CMAKE_CURRENT_BINARY_DIR}/dsp_syms.obj")
    set(DSP_SYMS_OUT "${CMAKE_CURRENT_BINARY_DIR}/dsp_syms.out")

    add_custom_command(
        OUTPUT "${DSP_SYMS_OBJ}"
        COMMAND ${CMAKE_C_COMPILER}
            ${C_FLAGS_LIST}
            --compile_only
            --output_file="${DSP_SYMS_OBJ}"
            "${DSP_SYMS_SRC}"
        DEPENDS "${DSP_SYMS_SRC}"
        COMMENT "Stage 1a: Compiling dsp_syms.c"
    )

    # Link dsp_syms.obj → dsp_syms.out (dynamic executable with exported symbols)
    # Note: lnk7x is the standalone linker; don't pass -z (that's for cl7x driver mode)
    add_custom_command(
        OUTPUT "${DSP_SYMS_OUT}"
        COMMAND ${CMAKE_LINKER}
            "${DSP_SYMS_OBJ}"
            --dynamic=exe
            --no_entry_point
            --soname=dsp_syms.out
            --output_file="${DSP_SYMS_OUT}"
        COMMAND ${TI_CGT_C7000_PATH}/bin/strip7x "${DSP_SYMS_OUT}"
        DEPENDS "${DSP_SYMS_OBJ}"
        COMMENT "Stage 1b: Linking dsp_syms.out (pseudo-firmware)"
    )

    # --- Stage 1.5: Embed dsp_syms.out as .dsp_syms_out section ---

    set(DSP_SYMS_ASM "${CMAKE_CURRENT_BINARY_DIR}/dsp_syms.out.asm")
    set(DSP_SYMS_SECT_OBJ "${CMAKE_CURRENT_BINARY_DIR}/dsp_syms.out.obj")

    add_custom_command(
        OUTPUT "${DSP_SYMS_ASM}"
        COMMAND python3
            "${DYNMOD_DIR}/gen_dsp_syms_sect.py"
            "${DSP_SYMS_OUT}"
            "${DSP_SYMS_ASM}"
        DEPENDS "${DSP_SYMS_OUT}" "${DYNMOD_DIR}/gen_dsp_syms_sect.py"
        COMMENT "Stage 1.5a: Embedding dsp_syms.out as assembly section"
    )

    add_custom_command(
        OUTPUT "${DSP_SYMS_SECT_OBJ}"
        COMMAND ${CMAKE_C_COMPILER}
            -mv7524 --abi=eabi
            --compile_only
            --output_file="${DSP_SYMS_SECT_OBJ}"
            "${DSP_SYMS_ASM}"
        DEPENDS "${DSP_SYMS_ASM}"
        COMMENT "Stage 1.5b: Assembling dsp_syms.out.asm"
    )

    # --- Stage 1.75: Embed weights.bin as .rodata.weights ---
    set(BIN_TO_ASM_SCRIPT "${DSP_RUNTIME_DIR}/scripts/bin_to_asm.py")
    set(WEIGHTS_OBJ_FILE "")
    set(WEIGHTS_EXPORT_FLAGS "")
    if(EXISTS "${WEIGHTS_FILE}")
        set(WEIGHTS_ASM "${CMAKE_CURRENT_BINARY_DIR}/weights_bin_data.asm")
        set(WEIGHTS_OBJ_FILE "${CMAKE_CURRENT_BINARY_DIR}/weights_bin_data.obj")

        add_custom_command(
            OUTPUT "${WEIGHTS_ASM}"
            COMMAND python3 "${BIN_TO_ASM_SCRIPT}"
                "${WEIGHTS_FILE}" "${WEIGHTS_ASM}" "weights_bin"
            DEPENDS "${WEIGHTS_FILE}" "${BIN_TO_ASM_SCRIPT}"
            COMMENT "Stage 1.75a: Generating weights assembly"
        )
        add_custom_command(
            OUTPUT "${WEIGHTS_OBJ_FILE}"
            COMMAND ${CMAKE_C_COMPILER}
                -mv7524 --abi=eabi
                --compile_only
                --output_file="${WEIGHTS_OBJ_FILE}"
                "${WEIGHTS_ASM}"
            DEPENDS "${WEIGHTS_ASM}"
            COMMENT "Stage 1.75b: Assembling weights object"
        )
        # Export weight symbols so DLOAD can query them at runtime.
        # Without --export, the linker strips unreferenced globals from --dynamic=lib.
        set(WEIGHTS_EXPORT_FLAGS
            --export=_binary_weights_bin_start
            --export=_binary_weights_bin_end
            --export=_binary_weights_bin_size
        )
        message(STATUS "  Weights: ${WEIGHTS_FILE} (embedded in dynmod)")
    endif()

    # --- Stage 2: Compile and link the TVM module ---

    set(LIB0_OUT "${CMAKE_CURRENT_BINARY_DIR}/lib0.out")
    set(LIB0_MAP "${CMAKE_CURRENT_BINARY_DIR}/lib0.out.map")
    set(DYNMOD_CMD "${DYNMOD_DIR}/c7x_dynmod.cmd")

    # Build include path and define lists for custom compile commands
    set(DYNMOD_INCLUDES
        -I${CMAKE_CURRENT_SOURCE_DIR}
    )
    if(DSP_RUNTIME_DIR AND EXISTS "${DSP_RUNTIME_DIR}")
        list(APPEND DYNMOD_INCLUDES -I${DSP_RUNTIME_DIR})
        list(APPEND DYNMOD_INCLUDES -I${DSP_RUNTIME_DIR}/mmalib)
    endif()
    if(TVM_HOME AND EXISTS "${TVM_HOME}/3rdparty/tvm-ffi/3rdparty/dlpack/include")
        list(APPEND DYNMOD_INCLUDES -I${TVM_HOME}/3rdparty/tvm-ffi/3rdparty/dlpack/include)
    endif()
    # TIDL_IO_MAX_NUM_CORES=4 is hardcoded in the c7x-mma-tidl source header
    # (itidl_io.h) for the IOBufDesc struct layout.  The PSDK header uses
    # SOC-dependent TIDL_MAX_NUM_CORES (2 for J722S), causing a mismatch.
    # Override to 4 so tidl_api.c reads IOBufDesc at the correct offsets.
    set(DYNMOD_DEFINES -DTVM_DSP_TARGET_C7X -DSOC_J722S -DTIDL_MAX_NUM_CORES=4)
    # lib*.c contains C++ code (namespaces, cstdint) - force C++ compilation
    set(DYNMOD_LANG_FLAGS --cpp_default)

    # TIDL bridge header for extern declarations (if provided)
    # TI cl7x uses --preinclude= to force-include a header
    if(TIDL_BRIDGE_SOURCES)
        string(REPLACE ".c" ".h" _DYNMOD_BRIDGE_HDR "${TIDL_BRIDGE_SOURCES}")
        if(EXISTS "${_DYNMOD_BRIDGE_HDR}")
            list(APPEND DYNMOD_LANG_FLAGS --preinclude=${_DYNMOD_BRIDGE_HDR})
        endif()
    endif()

    # --- TIDL integration (optional) ---
    set(TIDL_OBJ_FILES "")
    set(TIDL_LIB_FLAGS "")
    set(TIDL_EXPORT_FLAGS "")
    if(USE_TIDL)
        # TIDL headers from c7x-mma-tidl source tree.
        # The source header (itidl_io.h) uses TIDL_IO_MAX_NUM_CORES=4 for
        # sTIDL_IOBufDesc_t, matching the import tool's struct layout.
        # The PSDK header uses SOC-dependent TIDL_MAX_NUM_CORES (2 for J722S),
        # causing a struct size mismatch with the io.bin artifacts.
        set(C7X_MMA_TIDL_PATH "$ENV{C7X_MMA_TIDL_PATH}")
        if(NOT C7X_MMA_TIDL_PATH)
            set(C7X_MMA_TIDL_PATH "$ENV{HOME}/ml/c7x-mma-tidl")
        endif()
        if(EXISTS "${C7X_MMA_TIDL_PATH}/ti_dl/inc/itidl_ti.h")
            set(TIDL_INC_PATH "${C7X_MMA_TIDL_PATH}/ti_dl/inc")
        else()
            set(TIDL_INC_PATH "${PSDK_PATH}/c7x-mma-tidl/ti_dl/inc")
        endif()
        set(IVISION_INC_PATH "${PSDK_PATH}/ivision")
        if(NOT EXISTS "${TIDL_INC_PATH}/itidl_ti.h")
            message(FATAL_ERROR "TIDL headers not found at: ${TIDL_INC_PATH}")
        endif()
        list(APPEND DYNMOD_INCLUDES -I${TIDL_API_DIR} -I${TIDL_INC_PATH} -I${IVISION_INC_PATH})

        # NOTE: TIDL algo libraries (tidl_algo.lib, MMALIB) are linked
        # into the FIRMWARE, not the model module.  The model only needs
        # tidl_api sources compiled below.  Firmware symbols (TIDL_VISION_FXNS,
        # appMemAlloc, DmaUtils, etc.) are resolved by DLOAD at load time.

        # Compile TIDL API source files
        set(TIDL_SRCS
            ${TIDL_API_DIR}/tidl_api.c
            ${TIDL_API_DIR}/tidl_api_mem.c
            ${TIDL_API_DIR}/ti_mem_manager.c
        )
        foreach(TIDL_SRC ${TIDL_SRCS})
            get_filename_component(TIDL_SRC_NAME ${TIDL_SRC} NAME_WE)
            set(TIDL_OBJ "${CMAKE_CURRENT_BINARY_DIR}/${TIDL_SRC_NAME}.obj")
            add_custom_command(
                OUTPUT "${TIDL_OBJ}"
                COMMAND ${CMAKE_C_COMPILER}
                    ${C_FLAGS_LIST}
                    ${DYNMOD_INCLUDES}
                    ${DYNMOD_DEFINES}
                    --compile_only
                    --output_file="${TIDL_OBJ}"
                    "${TIDL_SRC}"
                DEPENDS "${TIDL_SRC}"
                COMMENT "TIDL: Compiling ${TIDL_SRC_NAME}.c"
            )
            list(APPEND TIDL_OBJ_FILES "${TIDL_OBJ}")
        endforeach()

        # Export TIDL API symbols for DLOAD
        set(TIDL_EXPORT_FLAGS
            --export=init_tidl_subgraph
            --export=process_tidl_subgraph
            --export=free_tidl_subgraph
        )

        # Embed TIDL artifacts (multi-subgraph: glob for net.bin + io.bin)
        set(TIDL_ARTIFACTS_DIR "" CACHE PATH "Directory with TIDL artifacts")
        if(TIDL_ARTIFACTS_DIR AND EXISTS "${TIDL_ARTIFACTS_DIR}")
            file(GLOB TIDL_NET_BINS "${TIDL_ARTIFACTS_DIR}/subgraph*_net.bin")
            file(GLOB TIDL_IO_BINS "${TIDL_ARTIFACTS_DIR}/subgraph*_params_1.bin")
            list(SORT TIDL_NET_BINS)
            list(SORT TIDL_IO_BINS)

            foreach(NET_BIN ${TIDL_NET_BINS})
                get_filename_component(NET_NAME ${NET_BIN} NAME)
                string(REGEX MATCH "subgraph([0-9]+)_net\\.bin" _MATCH "${NET_NAME}")
                set(SG_ID "${CMAKE_MATCH_1}")

                set(IO_BIN "${TIDL_ARTIFACTS_DIR}/subgraph${SG_ID}_params_1.bin")
                if(NOT EXISTS "${IO_BIN}")
                    message(WARNING "TIDL io.bin not found for subgraph ${SG_ID}: ${IO_BIN}")
                    continue()
                endif()

                foreach(ARTIFACT_PAIR "tidl_net_${SG_ID};${NET_BIN}" "tidl_io_${SG_ID};${IO_BIN}")
                    list(GET ARTIFACT_PAIR 0 ARTIFACT_NAME)
                    list(GET ARTIFACT_PAIR 1 ARTIFACT_BIN)

                    set(ARTIFACT_ASM "${CMAKE_CURRENT_BINARY_DIR}/${ARTIFACT_NAME}_data.asm")
                    set(ARTIFACT_OBJ "${CMAKE_CURRENT_BINARY_DIR}/${ARTIFACT_NAME}_data.obj")

                    add_custom_command(
                        OUTPUT "${ARTIFACT_ASM}"
                        COMMAND python3 "${BIN_TO_ASM_SCRIPT}"
                            "${ARTIFACT_BIN}" "${ARTIFACT_ASM}" "${ARTIFACT_NAME}"
                        DEPENDS "${ARTIFACT_BIN}" "${BIN_TO_ASM_SCRIPT}"
                        COMMENT "TIDL: Embedding ${ARTIFACT_NAME}"
                    )
                    add_custom_command(
                        OUTPUT "${ARTIFACT_OBJ}"
                        COMMAND ${CMAKE_C_COMPILER}
                            -mv7524 --abi=eabi
                            --compile_only
                            --output_file="${ARTIFACT_OBJ}"
                            "${ARTIFACT_ASM}"
                        DEPENDS "${ARTIFACT_ASM}"
                        COMMENT "TIDL: Assembling ${ARTIFACT_NAME}"
                    )
                    list(APPEND TIDL_OBJ_FILES "${ARTIFACT_OBJ}")
                    list(APPEND TIDL_EXPORT_FLAGS
                        --export=_binary_${ARTIFACT_NAME}_start
                        --export=_binary_${ARTIFACT_NAME}_end
                        --export=_binary_${ARTIFACT_NAME}_size
                    )
                endforeach()
                message(STATUS "  TIDL subgraph ${SG_ID}: ${NET_BIN}")
            endforeach()
            message(STATUS "  TIDL artifacts: ${TIDL_ARTIFACTS_DIR}")
        endif()

        # Compile TIDL firmware symbol stubs (separate from dsp_syms to avoid
        # symbol redefinition errors with malloc/free from rts*.lib)
        set(TIDL_FW_SYMS_SRC "${DYNMOD_DIR}/tidl_firmware_syms.c")
        set(TIDL_FW_SYMS_OBJ "${CMAKE_CURRENT_BINARY_DIR}/tidl_firmware_syms.obj")
        add_custom_command(
            OUTPUT "${TIDL_FW_SYMS_OBJ}"
            COMMAND ${CMAKE_C_COMPILER}
                ${C_FLAGS_LIST}
                --compile_only
                --output_file="${TIDL_FW_SYMS_OBJ}"
                "${TIDL_FW_SYMS_SRC}"
            DEPENDS "${TIDL_FW_SYMS_SRC}"
            COMMENT "TIDL: Compiling firmware symbol stubs"
        )
        list(APPEND TIDL_OBJ_FILES "${TIDL_FW_SYMS_OBJ}")

        message(STATUS "  TIDL: enabled (libs from ${TIDL_LIB_PATH})")
    endif()

    # Compile lib*.c → lib*.obj
    set(LIB_OBJS "")
    foreach(LIB_SRC ${LIB_SRCS})
        get_filename_component(LIB_NAME_WE ${LIB_SRC} NAME_WE)
        set(LIB_OBJ "${CMAKE_CURRENT_BINARY_DIR}/${LIB_NAME_WE}.obj")
        add_custom_command(
            OUTPUT "${LIB_OBJ}"
            COMMAND ${CMAKE_CXX_COMPILER}
                ${CXX_FLAGS_LIST}
                ${DYNMOD_LANG_FLAGS}
                ${DYNMOD_INCLUDES}
                ${DYNMOD_DEFINES}
                --compile_only
                --output_file="${LIB_OBJ}"
                "${LIB_SRC}"
            DEPENDS "${LIB_SRC}"
            COMMENT "Stage 2a: Compiling ${LIB_NAME_WE}.c"
        )
        list(APPEND LIB_OBJS "${LIB_OBJ}")
    endforeach()

    # Compile TIDL bridge (if provided)
    set(BRIDGE_OBJ_FILE "")
    if(TIDL_BRIDGE_SOURCES AND EXISTS "${TIDL_BRIDGE_SOURCES}")
        set(BRIDGE_OBJ_FILE "${CMAKE_CURRENT_BINARY_DIR}/tidl_bridge.obj")
        add_custom_command(
            OUTPUT "${BRIDGE_OBJ_FILE}"
            COMMAND ${CMAKE_C_COMPILER}
                ${C_FLAGS_LIST}
                ${DYNMOD_INCLUDES}
                ${DYNMOD_DEFINES}
                --compile_only
                --output_file="${BRIDGE_OBJ_FILE}"
                "${TIDL_BRIDGE_SOURCES}"
            DEPENDS "${TIDL_BRIDGE_SOURCES}"
            COMMENT "Stage 2a: Compiling TIDL bridge"
        )
    endif()

    # Link lib*.obj + dsp_syms.out.obj [+ TIDL objs] → lib0.out
    # Note: lnk7x is the standalone linker; don't pass -z
    add_custom_command(
        OUTPUT "${LIB0_OUT}"
        COMMAND ${CMAKE_LINKER}
            ${LIB_OBJS}
            ${BRIDGE_OBJ_FILE}
            ${WEIGHTS_OBJ_FILE}
            ${TIDL_OBJ_FILES}
            "${DYNMOD_CMD}"
            ${WEIGHTS_EXPORT_FLAGS}
            ${TIDL_EXPORT_FLAGS}
            ${TIDL_LIB_FLAGS}
            "${DSP_SYMS_SECT_OBJ}"
            -l${TI_CGT_C7000_PATH}/lib/rts7524_le.lib
            -l${TI_CGT_C7000_PATH}/lib/libc.a
            -m "${LIB0_MAP}"
            --output_file="${LIB0_OUT}"
        DEPENDS ${LIB_OBJS} ${BRIDGE_OBJ_FILE} "${DSP_SYMS_SECT_OBJ}" "${DSP_SYMS_OUT}" "${DYNMOD_CMD}"
                ${WEIGHTS_OBJ_FILE} ${TIDL_OBJ_FILES}
        WORKING_DIRECTORY "${CMAKE_CURRENT_BINARY_DIR}"
        COMMENT "Stage 2b: Linking lib0.out (DLOAD relocatable module)"
    )

    # Top-level target
    add_custom_target(c7x_dynmod ALL DEPENDS "${LIB0_OUT}")

    message(STATUS "C7x dynamic module build configured")
    message(STATUS "  Module: lib0.out")
    message(STATUS "  Linker script: ${DYNMOD_CMD}")
    message(STATUS "  Pseudo-firmware: dsp_syms.out")

else()
    message(FATAL_ERROR "Invalid TVM_DSP_TARGET: ${TVM_DSP_TARGET}. Use: host, c66x, c7x_host, c7x-dynmod")
endif()

# === Info Target ===
add_custom_target(info
    COMMAND ${CMAKE_COMMAND} -E echo "Build type: ${CMAKE_BUILD_TYPE}"
    COMMAND ${CMAKE_COMMAND} -E echo "Target: ${TVM_DSP_TARGET}"
    COMMAND ${CMAKE_COMMAND} -E echo "TVM_HOME: ${TVM_HOME}"
    COMMAND ${CMAKE_COMMAND} -E echo "Generated code: ${GENERATED_CODE_DIR}"
    COMMENT "Build configuration info"
)
