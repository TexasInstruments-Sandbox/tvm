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
# CMakeLists.txt for TVM DSP Runtime
#
# Build configurations:
#   1. Host emulation (default): Standard PC build for development/testing
#   2. C66x library: Cross-compile static library for C66x DSP
#   3. C7x library: Cross-compile static library for C7x DSP (future)
#
# Usage:
#   Host emulation (PC build):
#     mkdir build && cd build
#     cmake ..
#     cmake --build .
#
#   AWRL6844 C66x library (cross-compile):
#     mkdir build-c66x && cd build-c66x
#     cmake -DTVM_DSP_TARGET=c66x -DTVM_DSP_DEVICE=awrl6844 \
#           -DCMAKE_TOOLCHAIN_FILE=<path-to-toolchain-awrl6844.cmake> ..
#     cmake --build .
#

cmake_minimum_required(VERSION 3.16)
project(tvm_dsp_runtime LANGUAGES C CXX)

# === Default Build Type ===
# Use Release build by default for optimal performance (especially on C66x)
if(NOT CMAKE_BUILD_TYPE AND NOT CMAKE_CONFIGURATION_TYPES)
    set(CMAKE_BUILD_TYPE "Release" CACHE STRING "Build type (Release/Debug/RelWithDebInfo)" FORCE)
    message(STATUS "Build type not specified, defaulting to Release")
endif()

# === Target Selection ===
set(TVM_DSP_TARGET "host" CACHE STRING "DSP target: host, c66x, c7x, c7x_host")
set(TVM_DSP_DEVICE "" CACHE STRING "Device variant (e.g., awrl6844)")

# Validate target
if(NOT TVM_DSP_TARGET MATCHES "^(host|c66x|c7x|c7x_host)$")
    message(FATAL_ERROR "Invalid TVM_DSP_TARGET: ${TVM_DSP_TARGET}. Use: host, c66x, c7x, c7x_host")
endif()

message(STATUS "TVM DSP Runtime Configuration:")
message(STATUS "  Target: ${TVM_DSP_TARGET}")
if(TVM_DSP_DEVICE)
    message(STATUS "  Device: ${TVM_DSP_DEVICE}")
endif()

# === Build Configuration ===
set(CMAKE_C_STANDARD 99)
set(CMAKE_C_STANDARD_REQUIRED ON)
set(CMAKE_CXX_STANDARD 14)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_EXPORT_COMPILE_COMMANDS ON)

# === Target-Specific Configuration ===
if(TVM_DSP_TARGET STREQUAL "host")
    # Host emulation build
    add_compile_definitions(TVM_DSP_TARGET_HOST)
    # Enable aligned weights parsing (weights.bin generated with alignment=4)
    add_compile_definitions(TVM_DSP_ALIGNED_WEIGHTS)
    set(PLATFORM_SOURCES
        platform/host/host_platform.c
        platform/common/memory_pool.c
        dma/tvm_dsp_dma_host.c
        kernels/tvm_int8_residual_add.cpp
        kernels/tidl_activation_wrappers.c
        kernels/tidl_avgpool_wrappers.c
        kernels/tidl_norm_wrappers.c
        kernels/c7x_pool_relu_wrappers.cpp
        kernels/c7x_quantize.cpp
        kernels/tvm_dequantize_vecmatmul.cpp
        kernels/tvm_sdpa_decode.cpp
    )
    set(CMAKE_C_FLAGS "${CMAKE_C_FLAGS} -Wall -Wextra -Wno-unused-parameter")

elseif(TVM_DSP_TARGET STREQUAL "c66x")
    # C66x cross-compilation
    add_compile_definitions(TVM_DSP_TARGET_C66X)
    # Enable aligned weights parsing (weights.bin generated with alignment=4)
    add_compile_definitions(TVM_DSP_ALIGNED_WEIGHTS)
    if(TVM_DSP_DEVICE STREQUAL "awrl6844")
        add_compile_definitions(TVM_DSP_DEVICE_AWRL6844)
    endif()
    set(PLATFORM_SOURCES
        platform/c66x/c66x_platform.c
        platform/common/memory_pool.c
    )

elseif(TVM_DSP_TARGET STREQUAL "c7x")
    # C7x cross-compilation
    add_compile_definitions(TVM_DSP_TARGET_C7X)
    # Enable aligned weights parsing (weights.bin generated with alignment=4)
    add_compile_definitions(TVM_DSP_ALIGNED_WEIGHTS)
    if(TVM_DSP_DEVICE STREQUAL "j722s")
        add_compile_definitions(TVM_DSP_DEVICE_J722S)
    endif()
    # Code in DDR mode: application code runs from DDR (cached via MMU)
    # This maximizes L2 SRAM available for TVM data heap
    # MMU init code stays in L2 (marked with .text:l2_init section)
    option(TVM_DSP_CODE_IN_DDR "Place application code in DDR (requires MMU)" ON)
    if(TVM_DSP_CODE_IN_DDR)
        add_compile_definitions(TVM_DSP_CODE_IN_DDR)
        message(STATUS "C7x: Code in DDR mode enabled (L2 maximized for data)")
    endif()
    # Note: MMU initialization is the application's responsibility.
    # The application must configure the MMU before calling TVM runtime functions.
    # See tests/ti-dsp-runtime/dsp-cpp/j722s/mmu.c for a reference implementation.
    set(PLATFORM_SOURCES
        platform/c7x/c7x_platform.c
        platform/c7x/c7x_cxm.asm
        platform/common/memory_pool.c
        dma/tvm_dsp_dma.c
        kernels/tvm_int8_residual_add.cpp
        kernels/tidl_activation_wrappers.c
        kernels/tidl_avgpool_wrappers.c
        kernels/tidl_norm_wrappers.c
        kernels/c7x_pool_relu_wrappers.cpp
        kernels/c7x_quantize.cpp
        kernels/tvm_dequantize_vecmatmul.cpp
        kernels/tvm_sdpa_decode.cpp
    )
    # Enable assembly language for C7x
    enable_language(ASM)
    # DmaUtils headers require MCU_PLUS_SDK define for correct include paths
    add_compile_definitions(MCU_PLUS_SDK)

elseif(TVM_DSP_TARGET STREQUAL "c7x_host")
    # C7x host emulation build using TI C7000 Host Emulation library
    # Compiles with system g++ and the TI Host Emulation package, which
    # provides C7x vector types, intrinsics, streaming engine, and MMA
    # as C++ emulation on x86-64 Linux.
    add_compile_definitions(TVM_DSP_TARGET_C7X)
    add_compile_definitions(TVM_DSP_DEVICE_J722S)
    add_compile_definitions(TVM_DSP_ALIGNED_WEIGHTS)
    # Distinguish host emulation from real C7x hardware.
    # TI Host Emu headers define __C7000__ and __TI_COMPILER_VERSION__,
    # so platform code checks C7X_HOST_EMULATION to use malloc-based
    # L2 instead of hardware addresses.
    add_compile_definitions(C7X_HOST_EMULATION)
    # Override fallback pool sizes for host emulation (larger than
    # hardware defaults to accommodate models during development)
    add_compile_definitions(
        TVM_DSP_L2_SIZE_FALLBACK=4*1024*1024
        TVM_DSP_DDR_SIZE_FALLBACK=256*1024*1024
    )
    set(PLATFORM_SOURCES
        platform/c7x/c7x_platform.c
        platform/common/memory_pool.c
        dma/tvm_dsp_dma_host.c
        kernels/tvm_int8_residual_add.cpp
        kernels/tidl_activation_wrappers.c
        kernels/tidl_avgpool_wrappers.c
        kernels/tidl_norm_wrappers.c
        kernels/c7x_pool_relu_wrappers.cpp
        kernels/c7x_quantize.cpp
        kernels/tvm_dequantize_vecmatmul.cpp
        kernels/tvm_sdpa_decode.cpp
    )
    # TI C7000 Host Emulation library
    if(DEFINED ENV{TI_CGT_C7000_PATH})
        set(TI_CGT_C7000_PATH "$ENV{TI_CGT_C7000_PATH}")
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
    set(CMAKE_C_FLAGS "${CMAKE_C_FLAGS} -Wall -Wextra -Wno-unused-parameter")
    set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -fno-strict-aliasing -Wall -Wextra -Wno-unused-parameter")
endif()

# === Core Source Files ===
# These are added as the implementation progresses
set(CORE_SOURCES
    # Phase 2: FFI types
    ffi/ffi_types.c
    ffi/object.c

    # Phase 3: Containers
    container/array.c
    container/ndarray.c
    container/shape.c

    # Phase 4: VM and Backend API (C++ implementation using ScopeGuard, TypedHandle, Span)
    vm/vm_builtins.cpp
    vm/storage.cpp
    vm/c_backend_api.cpp

    # Phase 5: Registry (C++ implementation using FixedVector, TypedHandle)
    registry/registry.cpp

    # Phase 7: Constants loader (weights.bin parser)
    constants/stream.c
    constants/constants.c
    constants/constants_c_api.c
    constants/constants_loader.cpp

    # C++14 Model API
    cpp/model.cpp
)

# === Include Directories ===
set(INCLUDE_DIRS
    ${CMAKE_CURRENT_SOURCE_DIR}/include
    ${CMAKE_CURRENT_SOURCE_DIR}/core
    ${CMAKE_CURRENT_SOURCE_DIR}/platform
    ${CMAKE_CURRENT_SOURCE_DIR}/ffi
    ${CMAKE_CURRENT_SOURCE_DIR}/container
    ${CMAKE_CURRENT_SOURCE_DIR}/vm
    ${CMAKE_CURRENT_SOURCE_DIR}/registry
    ${CMAKE_CURRENT_SOURCE_DIR}/constants
    ${CMAKE_CURRENT_SOURCE_DIR}/cpp
    ${CMAKE_CURRENT_SOURCE_DIR}/dma
    # DLPack headers from TVM 3rdparty
    ${CMAKE_CURRENT_SOURCE_DIR}/../../../3rdparty/tvm-ffi/3rdparty/dlpack/include
)

# === Weight Embedding Module ===
# Include the weight embedding CMake module for applications
list(APPEND CMAKE_MODULE_PATH ${CMAKE_CURRENT_SOURCE_DIR}/cmake)

# === Build Library ===
add_library(tvm_dsp_runtime STATIC
    ${PLATFORM_SOURCES}
    ${CORE_SOURCES}
)

target_include_directories(tvm_dsp_runtime PUBLIC ${INCLUDE_DIRS})

# Add TI Host Emulation includes for c7x_host target
if(TVM_DSP_TARGET STREQUAL "c7x_host")
    target_include_directories(tvm_dsp_runtime PUBLIC ${C7X_HOST_EMU_INCLUDE})
    target_link_libraries(tvm_dsp_runtime PUBLIC ${C7X_HOST_EMU_LIB})
endif()

# Add DmaUtils include paths for c7x target (async DMA via EDMA/DRU).
# The runtime .a only needs the headers; the application (firmware) links
# the prebuilt dmautils .lib at final link time.
if(TVM_DSP_TARGET STREQUAL "c7x" AND DEFINED MCU_PLUS_SDK_PATH AND EXISTS "${MCU_PLUS_SDK_PATH}")
    target_include_directories(tvm_dsp_runtime PRIVATE
        ${MCU_PLUS_SDK_PATH}/source/drivers/dmautils/include
        ${MCU_PLUS_SDK_PATH}/source/drivers/dmautils/csl
        ${MCU_PLUS_SDK_PATH}/source/drivers/dmautils/udma_standalone
        ${MCU_PLUS_SDK_PATH}/source/drivers/dmautils/udma_standalone/include
    )
endif()

# Set output name based on target
if(TVM_DSP_TARGET STREQUAL "host")
    set_target_properties(tvm_dsp_runtime PROPERTIES
        OUTPUT_NAME "tvm_dsp_runtime_host"
    )
elseif(TVM_DSP_TARGET STREQUAL "c66x")
    set_target_properties(tvm_dsp_runtime PROPERTIES
        OUTPUT_NAME "tvm_dsp_runtime_c66x"
    )
elseif(TVM_DSP_TARGET STREQUAL "c7x")
    set_target_properties(tvm_dsp_runtime PROPERTIES
        OUTPUT_NAME "tvm_dsp_runtime_c7x"
    )
elseif(TVM_DSP_TARGET STREQUAL "c7x_host")
    set_target_properties(tvm_dsp_runtime PROPERTIES
        OUTPUT_NAME "tvm_dsp_runtime_c7x_host"
    )
endif()

# === Tests ===
option(TVM_DSP_BUILD_TESTS "Build tests" ON)

if(TVM_DSP_BUILD_TESTS AND EXISTS ${CMAKE_CURRENT_SOURCE_DIR}/tests)
    if(TVM_DSP_TARGET STREQUAL "host")
        # Host tests - simple executable
        enable_testing()

        if(EXISTS ${CMAKE_CURRENT_SOURCE_DIR}/tests/test_memory.c)
            add_executable(test_memory tests/test_memory.c)
            target_link_libraries(test_memory tvm_dsp_runtime)
            add_test(NAME test_memory COMMAND test_memory)
        endif()

        if(EXISTS ${CMAKE_CURRENT_SOURCE_DIR}/tests/test_ndarray.c)
            add_executable(test_ndarray tests/test_ndarray.c)
            target_link_libraries(test_ndarray tvm_dsp_runtime)
            add_test(NAME test_ndarray COMMAND test_ndarray)
        endif()

        if(EXISTS ${CMAKE_CURRENT_SOURCE_DIR}/tests/test_builtins.c)
            add_executable(test_builtins tests/test_builtins.c)
            target_link_libraries(test_builtins tvm_dsp_runtime)
            add_test(NAME test_builtins COMMAND test_builtins)
        endif()

        # Phase 2: FFI types test
        if(EXISTS ${CMAKE_CURRENT_SOURCE_DIR}/tests/test_ffi_types.c)
            add_executable(test_ffi_types tests/test_ffi_types.c)
            target_link_libraries(test_ffi_types tvm_dsp_runtime m)
            add_test(NAME test_ffi_types COMMAND test_ffi_types)
        endif()

        # Phase 4: VM builtins test
        if(EXISTS ${CMAKE_CURRENT_SOURCE_DIR}/tests/test_vm_builtins.c)
            add_executable(test_vm_builtins tests/test_vm_builtins.c)
            target_link_libraries(test_vm_builtins tvm_dsp_runtime)
            add_test(NAME test_vm_builtins COMMAND test_vm_builtins)
        endif()

        # Phase 5: Registry test
        if(EXISTS ${CMAKE_CURRENT_SOURCE_DIR}/tests/test_registry.c)
            add_executable(test_registry tests/test_registry.c)
            target_link_libraries(test_registry tvm_dsp_runtime)
            add_test(NAME test_registry COMMAND test_registry)
        endif()

        # Phase 6: C++ wrappers test
        if(EXISTS ${CMAKE_CURRENT_SOURCE_DIR}/tests/test_cpp_wrappers.cpp)
            add_executable(test_cpp_wrappers tests/test_cpp_wrappers.cpp)
            target_link_libraries(test_cpp_wrappers tvm_dsp_runtime)
            add_test(NAME test_cpp_wrappers COMMAND test_cpp_wrappers)
        endif()

        # C++14 migration: ScopeGuard test
        if(EXISTS ${CMAKE_CURRENT_SOURCE_DIR}/tests/test_scope_guard.cpp)
            add_executable(test_scope_guard tests/test_scope_guard.cpp)
            target_link_libraries(test_scope_guard tvm_dsp_runtime)
            add_test(NAME test_scope_guard COMMAND test_scope_guard)
        endif()

        # C++14 migration: Span test
        if(EXISTS ${CMAKE_CURRENT_SOURCE_DIR}/tests/test_span.cpp)
            add_executable(test_span tests/test_span.cpp)
            target_link_libraries(test_span tvm_dsp_runtime)
            add_test(NAME test_span COMMAND test_span)
        endif()

        # C++14 migration: FixedVector test
        if(EXISTS ${CMAKE_CURRENT_SOURCE_DIR}/tests/test_fixed_vector.cpp)
            add_executable(test_fixed_vector tests/test_fixed_vector.cpp)
            target_link_libraries(test_fixed_vector tvm_dsp_runtime)
            add_test(NAME test_fixed_vector COMMAND test_fixed_vector)
        endif()

        # C++14 migration: Result test
        if(EXISTS ${CMAKE_CURRENT_SOURCE_DIR}/tests/test_result.cpp)
            add_executable(test_result tests/test_result.cpp)
            target_link_libraries(test_result tvm_dsp_runtime)
            add_test(NAME test_result COMMAND test_result)
        endif()

        # C++14 migration: TypedHandle test
        if(EXISTS ${CMAKE_CURRENT_SOURCE_DIR}/tests/test_typed_handle.cpp)
            add_executable(test_typed_handle tests/test_typed_handle.cpp)
            target_link_libraries(test_typed_handle tvm_dsp_runtime)
            add_test(NAME test_typed_handle COMMAND test_typed_handle)
        endif()

        # Constants parser test
        if(EXISTS ${CMAKE_CURRENT_SOURCE_DIR}/tests/test_constants.c)
            add_executable(test_constants tests/test_constants.c)
            target_link_libraries(test_constants tvm_dsp_runtime)
            # Define path to test weights file (local copy in tests dir)
            target_compile_definitions(test_constants PRIVATE
                TEST_WEIGHTS_PATH="${CMAKE_CURRENT_SOURCE_DIR}/tests/test_weights.bin"
            )
            add_test(NAME test_constants COMMAND test_constants)
        endif()

    elseif(TVM_DSP_TARGET STREQUAL "c66x")
        # C66x tests - need board support and linker script
        # Use board support from radar project
        set(RADAR_CPP_DIR "${CMAKE_CURRENT_SOURCE_DIR}/../../../tests/ti-dsp-runtime/dsp-cpp")

        if(EXISTS ${RADAR_CPP_DIR}/awrl6844)
            message(STATUS "Using AWRL6844 board support from: ${RADAR_CPP_DIR}/awrl6844")

            # Board support sources
            file(GLOB AWRL6844_SYSCFG_SRCS "${RADAR_CPP_DIR}/awrl6844/syscfg/*.c")
            set(C66X_BOARD_SRCS
                ${RADAR_CPP_DIR}/awrl6844/board_init.c
                ${AWRL6844_SYSCFG_SRCS}
            )

            # Include directories for board support
            target_include_directories(tvm_dsp_runtime PUBLIC
                ${RADAR_CPP_DIR}/awrl6844
                ${RADAR_CPP_DIR}/awrl6844/syscfg
            )

            # MCU+ SDK includes
            if(DEFINED MMWAVE_SDK_PATH AND EXISTS "${MMWAVE_SDK_PATH}")
                target_include_directories(tvm_dsp_runtime PUBLIC
                    ${MMWAVE_SDK_PATH}/source
                    ${MMWAVE_SDK_PATH}/source/kernel/nortos
                    ${MMWAVE_SDK_PATH}/source/kernel/nortos/dpl/c66
                    ${MMWAVE_SDK_PATH}/source/drivers
                    ${MMWAVE_SDK_PATH}/source/board
                )
            endif()

            # C66x target defines
            target_compile_definitions(tvm_dsp_runtime PUBLIC
                CORE_DSP
                _TMS320C6600
                TMS320C66X
                TARGET_C66X_AWRL6844
                SOC_XWRL684X
                ENABLE_PRINTFS=1
                _TI_STD_TYPES=1
                TISTDTYPES_H_=1
                TRUE=1
                FALSE=0
            )

            # === Common C66x linker configuration ===
            # Clear any host linker flags before linking
            set(CMAKE_EXE_LINKER_FLAGS "" CACHE STRING "" FORCE)
            set(CMAKE_EXE_LINKER_FLAGS_RELEASE "" CACHE STRING "" FORCE)
            set(CMAKE_EXE_LINKER_FLAGS_DEBUG "" CACHE STRING "" FORCE)

            # Linker command file
            set(C66X_LINKER_CMD_FILE "${RADAR_CPP_DIR}/awrl6844/linker_c66x.cmd")

            # Base linker flags (without map file - each test adds its own)
            set(C66X_BASE_LINK_FLAGS "-cr -x")
            set(C66X_BASE_LINK_FLAGS "${C66X_BASE_LINK_FLAGS} ${C66X_LINKER_CMD_FILE}")
            set(C66X_BASE_LINK_FLAGS "${C66X_BASE_LINK_FLAGS} -l${TI_CGT_C6000_PATH}/lib/rts6600_elf.lib")
            set(C66X_BASE_LINK_FLAGS "${C66X_BASE_LINK_FLAGS} -l${TI_CGT_C6000_PATH}/lib/libc.a")

            # SDK libraries
            if(DEFINED MMWAVE_SDK_PATH AND EXISTS "${MMWAVE_SDK_PATH}")
                if(CMAKE_BUILD_TYPE STREQUAL "Debug" OR NOT CMAKE_BUILD_TYPE)
                    set(SDK_PROFILE "debug")
                else()
                    set(SDK_PROFILE "release")
                endif()

                set(SDK_NORTOS_LIB "${MMWAVE_SDK_PATH}/source/kernel/nortos/lib")
                set(SDK_DRIVERS_LIB "${MMWAVE_SDK_PATH}/source/drivers/lib")
                set(SDK_BOARD_LIB "${MMWAVE_SDK_PATH}/source/board/lib")

                set(C66X_BASE_LINK_FLAGS "${C66X_BASE_LINK_FLAGS} -i${SDK_NORTOS_LIB}")
                set(C66X_BASE_LINK_FLAGS "${C66X_BASE_LINK_FLAGS} -i${SDK_DRIVERS_LIB}")
                set(C66X_BASE_LINK_FLAGS "${C66X_BASE_LINK_FLAGS} -i${SDK_BOARD_LIB}")
                set(C66X_BASE_LINK_FLAGS "${C66X_BASE_LINK_FLAGS} -lnortos.xwrL684x.c66.ti-c6000.${SDK_PROFILE}.lib")
                set(C66X_BASE_LINK_FLAGS "${C66X_BASE_LINK_FLAGS} -ldrivers.xwrL684x.c66.ti-c6000.${SDK_PROFILE}.lib")
                set(C66X_BASE_LINK_FLAGS "${C66X_BASE_LINK_FLAGS} -lboard.xwrL684x.c66.ti-c6000.${SDK_PROFILE}.lib")
            endif()

            # Function to add a C66x test executable
            function(add_c66x_test TEST_NAME TEST_SOURCE)
                if(EXISTS ${CMAKE_CURRENT_SOURCE_DIR}/${TEST_SOURCE})
                    add_executable(${TEST_NAME}_c66x
                        ${TEST_SOURCE}
                        ${C66X_BOARD_SRCS}
                    )
                    target_link_libraries(${TEST_NAME}_c66x tvm_dsp_runtime)
                    set(TEST_LINK_FLAGS "-m ${CMAKE_CURRENT_BINARY_DIR}/${TEST_NAME}_c66x.map ${C66X_BASE_LINK_FLAGS}")
                    set_target_properties(${TEST_NAME}_c66x PROPERTIES
                        LINK_FLAGS "${TEST_LINK_FLAGS}"
                        SUFFIX ".out"
                    )
                    message(STATUS "C66x test executable: ${TEST_NAME}_c66x.out")
                endif()
            endfunction()

            # === C66x Test Executables ===
            add_c66x_test(test_memory tests/test_memory.c)
            add_c66x_test(test_ffi_types tests/test_ffi_types.c)
            add_c66x_test(test_ndarray tests/test_ndarray.c)
            add_c66x_test(test_vm_builtins tests/test_vm_builtins.c)
            add_c66x_test(test_registry tests/test_registry.c)
            add_c66x_test(test_scope_guard tests/test_scope_guard.cpp)
            add_c66x_test(test_span tests/test_span.cpp)
            add_c66x_test(test_fixed_vector tests/test_fixed_vector.cpp)
            add_c66x_test(test_result tests/test_result.cpp)
            add_c66x_test(test_typed_handle tests/test_typed_handle.cpp)
        else()
            message(WARNING "AWRL6844 board support not found at ${RADAR_CPP_DIR}/awrl6844, skipping C66x tests")
        endif()

    elseif(TVM_DSP_TARGET STREQUAL "c7x_host")
        # C7x host emulation tests - same as host tests (x86 executable)
        enable_testing()

        if(EXISTS ${CMAKE_CURRENT_SOURCE_DIR}/tests/test_memory.c)
            add_executable(test_memory tests/test_memory.c)
            target_link_libraries(test_memory tvm_dsp_runtime)
            add_test(NAME test_memory COMMAND test_memory)
        endif()

        if(EXISTS ${CMAKE_CURRENT_SOURCE_DIR}/tests/test_ffi_types.c)
            add_executable(test_ffi_types tests/test_ffi_types.c)
            target_link_libraries(test_ffi_types tvm_dsp_runtime m)
            add_test(NAME test_ffi_types COMMAND test_ffi_types)
        endif()

        if(EXISTS ${CMAKE_CURRENT_SOURCE_DIR}/tests/test_vm_builtins.c)
            add_executable(test_vm_builtins tests/test_vm_builtins.c)
            target_link_libraries(test_vm_builtins tvm_dsp_runtime)
            add_test(NAME test_vm_builtins COMMAND test_vm_builtins)
        endif()

        if(EXISTS ${CMAKE_CURRENT_SOURCE_DIR}/tests/test_registry.c)
            add_executable(test_registry tests/test_registry.c)
            target_link_libraries(test_registry tvm_dsp_runtime)
            add_test(NAME test_registry COMMAND test_registry)
        endif()

        if(EXISTS ${CMAKE_CURRENT_SOURCE_DIR}/tests/test_constants.c)
            add_executable(test_constants tests/test_constants.c)
            target_link_libraries(test_constants tvm_dsp_runtime)
            target_compile_definitions(test_constants PRIVATE
                TEST_WEIGHTS_PATH="${CMAKE_CURRENT_SOURCE_DIR}/tests/test_weights.bin"
            )
            add_test(NAME test_constants COMMAND test_constants)
        endif()

    elseif(TVM_DSP_TARGET STREQUAL "c7x")
        # C7x tests - need board support and linker script
        # Note: C7x test infrastructure requires MCU+ SDK and J722S board support
        # For now, we just configure the library build; tests can be added when
        # board support files are available.

        # MCU+ SDK includes
        if(DEFINED MCU_PLUS_SDK_PATH AND EXISTS "${MCU_PLUS_SDK_PATH}")
            target_include_directories(tvm_dsp_runtime PUBLIC
                ${MCU_PLUS_SDK_PATH}/source
                ${MCU_PLUS_SDK_PATH}/source/kernel/nortos
                ${MCU_PLUS_SDK_PATH}/source/kernel/nortos/dpl/c75
                ${MCU_PLUS_SDK_PATH}/source/drivers
                ${MCU_PLUS_SDK_PATH}/source/board
            )

            # C7x target defines
            target_compile_definitions(tvm_dsp_runtime PUBLIC
                SOC_J722S
                CORE_C75
                _TMS320C7X
                TARGET_C7X_J722S
                ENABLE_PRINTFS=1
                _TI_STD_TYPES=1
                TISTDTYPES_H_=1
                TRUE=1
                FALSE=0
            )
        endif()

        # C7x linker configuration
        set(CMAKE_EXE_LINKER_FLAGS "" CACHE STRING "" FORCE)
        set(CMAKE_EXE_LINKER_FLAGS_RELEASE "" CACHE STRING "" FORCE)
        set(CMAKE_EXE_LINKER_FLAGS_DEBUG "" CACHE STRING "" FORCE)

        # Linker command file
        set(C7X_LINKER_CMD_FILE "${CMAKE_CURRENT_SOURCE_DIR}/platform/c7x/j722s/linker_c7x.cmd")

        # Base linker flags (C7x requires -z flag)
        set(C7X_BASE_LINK_FLAGS "--rom_model --reread_libs --warn_sections")
        set(C7X_BASE_LINK_FLAGS "${C7X_BASE_LINK_FLAGS} ${C7X_LINKER_CMD_FILE}")
        if(DEFINED TI_CGT_C7000_PATH)
            set(C7X_BASE_LINK_FLAGS "${C7X_BASE_LINK_FLAGS} -l${TI_CGT_C7000_PATH}/lib/rts7524_le.lib")
            set(C7X_BASE_LINK_FLAGS "${C7X_BASE_LINK_FLAGS} -l${TI_CGT_C7000_PATH}/lib/libc.a")
        endif()

        # SDK libraries
        if(DEFINED MCU_PLUS_SDK_PATH AND EXISTS "${MCU_PLUS_SDK_PATH}")
            if(CMAKE_BUILD_TYPE STREQUAL "Debug" OR NOT CMAKE_BUILD_TYPE)
                set(SDK_PROFILE "debug")
            else()
                set(SDK_PROFILE "release")
            endif()

            set(SDK_NORTOS_LIB "${MCU_PLUS_SDK_PATH}/source/kernel/nortos/lib")
            set(SDK_DRIVERS_LIB "${MCU_PLUS_SDK_PATH}/source/drivers/lib")
            set(SDK_BOARD_LIB "${MCU_PLUS_SDK_PATH}/source/board/lib")

            set(C7X_BASE_LINK_FLAGS "${C7X_BASE_LINK_FLAGS} -i${SDK_NORTOS_LIB}")
            set(C7X_BASE_LINK_FLAGS "${C7X_BASE_LINK_FLAGS} -i${SDK_DRIVERS_LIB}")
            set(C7X_BASE_LINK_FLAGS "${C7X_BASE_LINK_FLAGS} -i${SDK_BOARD_LIB}")
        endif()

        # Function to add a C7x test executable
        function(add_c7x_test TEST_NAME TEST_SOURCE)
            if(EXISTS ${CMAKE_CURRENT_SOURCE_DIR}/${TEST_SOURCE})
                add_executable(${TEST_NAME}_c7x ${TEST_SOURCE})
                target_link_libraries(${TEST_NAME}_c7x tvm_dsp_runtime)
                set(TEST_LINK_FLAGS "-m ${CMAKE_CURRENT_BINARY_DIR}/${TEST_NAME}_c7x.map ${C7X_BASE_LINK_FLAGS}")
                set_target_properties(${TEST_NAME}_c7x PROPERTIES
                    LINK_FLAGS "${TEST_LINK_FLAGS}"
                    SUFFIX ".out"
                )
                message(STATUS "C7x test executable: ${TEST_NAME}_c7x.out")
            endif()
        endfunction()

        message(STATUS "C7x build configured. Test executables require MCU+ SDK board support files.")
    endif()
endif()

# === Install ===
install(TARGETS tvm_dsp_runtime ARCHIVE DESTINATION lib)
install(DIRECTORY include/ DESTINATION include/tvm_dsp_runtime)
install(FILES
    platform/dsp_platform.h
    platform/dsp_memory.h
    DESTINATION include/tvm_dsp_runtime/platform
)
install(FILES
    ffi/ffi_types.h
    ffi/object.h
    ffi/any_view.h
    DESTINATION include/tvm_dsp_runtime/ffi
)
install(FILES
    container/array.h
    container/ndarray.h
    container/shape.h
    DESTINATION include/tvm_dsp_runtime/container
)
install(FILES
    vm/storage.h
    vm/vm_builtins.h
    DESTINATION include/tvm_dsp_runtime/vm
)
install(FILES
    registry/registry.h
    registry/packed_func.h
    DESTINATION include/tvm_dsp_runtime/registry
)
install(FILES
    cpp/object_ref.h
    cpp/ndarray.h
    cpp/any.h
    cpp/packed_func.h
    cpp/c_runtime_api.h
    cpp/scope_guard.h
    cpp/span.h
    cpp/fixed_vector.h
    cpp/result.h
    cpp/typed_handle.h
    DESTINATION include/tvm_dsp_runtime/cpp
)
install(FILES
    constants/constants.h
    constants/constants_c_api.h
    constants/stream.h
    DESTINATION include/tvm_dsp_runtime/constants
)
install(FILES
    cmake/WeightEmbedding.cmake
    DESTINATION share/tvm_dsp_runtime/cmake
)

# === Print Summary ===
message(STATUS "TVM DSP Runtime Build Summary:")
message(STATUS "  Library: tvm_dsp_runtime")
message(STATUS "  Target: ${TVM_DSP_TARGET}")
message(STATUS "  Device: ${TVM_DSP_DEVICE}")
message(STATUS "  Build type: ${CMAKE_BUILD_TYPE}")
message(STATUS "  Platform sources: ${PLATFORM_SOURCES}")
message(STATUS "  Core sources: ${CORE_SOURCES}")
