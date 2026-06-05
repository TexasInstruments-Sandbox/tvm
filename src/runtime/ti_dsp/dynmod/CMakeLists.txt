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
# CMakeLists.txt for C7x DLOAD dynamic module builds.
#
# Builds TVM-generated lib0.c as a relocatable ELF module loadable by
# the c7x_compute firmware's DLOAD dynamic loader.
#
# Two-stage link:
#   Stage 1: dsp_syms.c -> dsp_syms.out (pseudo-firmware symbol table)
#   Stage 2: lib0.c + dsp_syms.out -> lib0.out (relocatable module)
#
# Usage (standalone):
#   cmake -DCMAKE_TOOLCHAIN_FILE=<path>/toolchain-j722s-c7x.cmake \
#         -DGENERATED_CODE_DIR=<path_to_lib0> ..
#   cmake --build .
#
# Usage (from Python build pipeline):
#   Invoked by TIDLOffloadCompiler._build_dynmod()
#

cmake_minimum_required(VERSION 3.16)

project(C7xDynmod LANGUAGES C CXX)

# === Build Type Configuration ===
if(NOT CMAKE_BUILD_TYPE)
    set(CMAKE_BUILD_TYPE "Release" CACHE STRING "Build type" FORCE)
endif()
message(STATUS "Build type: ${CMAKE_BUILD_TYPE}")

# === Path Configuration ===
if(DEFINED ENV{TVM_HOME})
    # FORCE so ENV{TVM_HOME} wins even when -DTVM_HOME=<wrong-path> is passed
    # by a pip-installed caller whose parents[5] points into .venv/lib/pythonX.Y/.
    # Mirrors the FORCE already present in dsp-cpp/CMakeLists.txt.
    set(TVM_HOME "$ENV{TVM_HOME}" CACHE PATH "Path to TVM" FORCE)
elseif(UNIX)
    set(TVM_HOME "$ENV{HOME}/ml/tvm" CACHE PATH "Path to TVM")
endif()

# DSP_RUNTIME_DIR can be set explicitly for pip-installed packages
# where the source tree is not available and build infrastructure
# is bundled under tvm/data/ti_dsp/.
if(NOT DSP_RUNTIME_DIR)
    set(DSP_RUNTIME_DIR "${TVM_HOME}/src/runtime/ti_dsp" CACHE PATH "DSP runtime dir")
endif()
set(TIDL_API_DIR "${DSP_RUNTIME_DIR}/tidl")

# Generated code directory (must contain lib*.c files)
set(GENERATED_CODE_DIR "" CACHE PATH "Directory containing lib*.c files")
if(NOT GENERATED_CODE_DIR)
    message(FATAL_ERROR "GENERATED_CODE_DIR must be set")
endif()

# === TIDL Bridge (optional) ===
set(TIDL_BRIDGE_SOURCES "" CACHE STRING "Extra .c files for TIDL bridge")

# === TIDL Integration (optional) ===
option(USE_TIDL "Link TIDL libraries into model module" OFF)
set(PSDK_PATH "/opt/ti/am67a/ti-processor-sdk-rtos-j722s-evm-11_00_00_06"
    CACHE PATH "Path to TI Processor SDK")

# === Weights ===
set(WEIGHTS_FILE "${GENERATED_CODE_DIR}/weights.bin" CACHE FILEPATH "Path to weights.bin")

message(STATUS "C7x Dynmod Configuration:")
message(STATUS "  TVM_HOME: ${TVM_HOME}")
message(STATUS "  Generated code: ${GENERATED_CODE_DIR}")

# === Dynmod infrastructure ===
set(DYNMOD_DIR "${CMAKE_CURRENT_SOURCE_DIR}/c7x_dynmod")

# Compiler flags
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

# Split flag strings into lists for add_custom_command
string(TOUPPER "${CMAKE_BUILD_TYPE}" _BUILD_TYPE_UPPER)
set(_C_ALL_FLAGS "${CMAKE_C_FLAGS} ${CMAKE_C_FLAGS_${_BUILD_TYPE_UPPER}}")
set(_CXX_ALL_FLAGS "${CMAKE_CXX_FLAGS} ${CMAKE_CXX_FLAGS_${_BUILD_TYPE_UPPER}}")
separate_arguments(C_FLAGS_LIST NATIVE_COMMAND "${_C_ALL_FLAGS}")
separate_arguments(CXX_FLAGS_LIST NATIVE_COMMAND "${_CXX_ALL_FLAGS}")

# --- Stage 1: Build dsp_syms.out (pseudo-firmware symbol table) ---

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

# Build include path and define lists
set(DYNMOD_INCLUDES
    -I${GENERATED_CODE_DIR}
)
if(DSP_RUNTIME_DIR AND EXISTS "${DSP_RUNTIME_DIR}")
    list(APPEND DYNMOD_INCLUDES -I${DSP_RUNTIME_DIR})
endif()
if(TVM_HOME AND EXISTS "${TVM_HOME}/3rdparty/tvm-ffi/3rdparty/dlpack/include")
    list(APPEND DYNMOD_INCLUDES -I${TVM_HOME}/3rdparty/tvm-ffi/3rdparty/dlpack/include)
elseif(DSP_RUNTIME_DIR AND EXISTS "${DSP_RUNTIME_DIR}/include/dlpack")
    # Bundled dlpack headers in pip-installed package
    list(APPEND DYNMOD_INCLUDES -I${DSP_RUNTIME_DIR}/include)
endif()
set(DYNMOD_DEFINES -DTVM_DSP_TARGET_C7X -DSOC_J722S -DTIDL_MAX_NUM_CORES=4 -DRPROC_TRACE=0)

# Common C7x headers (rproc_trace.h etc.)
set(C7X_COMMON_DIR "${TVM_HOME}/src/runtime/ti_dsp/firmware/c7x/common")
if(EXISTS "${C7X_COMMON_DIR}")
    list(APPEND DYNMOD_INCLUDES -I${C7X_COMMON_DIR})
endif()
set(DYNMOD_LANG_FLAGS --cpp_default)

# TIDL bridge header for extern declarations
if(TIDL_BRIDGE_SOURCES)
    string(REPLACE ".c" ".h" _DYNMOD_BRIDGE_HDR "${TIDL_BRIDGE_SOURCES}")
    if(EXISTS "${_DYNMOD_BRIDGE_HDR}")
        list(APPEND DYNMOD_LANG_FLAGS --preinclude=${_DYNMOD_BRIDGE_HDR})
    endif()
endif()

# MMALIB function declarations (symbols resolved at DLOAD time from firmware)
if(EXISTS "${DSP_RUNTIME_DIR}/mmalib/mmalib_wrappers.h")
    list(APPEND DYNMOD_LANG_FLAGS --preinclude=${DSP_RUNTIME_DIR}/mmalib/mmalib_wrappers.h)
endif()

# --- TIDL integration (optional) ---
set(TIDL_OBJ_FILES "")
set(TIDL_LIB_FLAGS "")
set(TIDL_EXPORT_FLAGS "")
if(USE_TIDL)
    # TIDL headers
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
        --export=tidl_bridge_cleanup
    )

    # Embed TIDL artifacts (multi-subgraph support via glob)
    set(TIDL_ARTIFACTS_DIR "" CACHE PATH "Directory with TIDL artifacts")
    if(TIDL_ARTIFACTS_DIR AND EXISTS "${TIDL_ARTIFACTS_DIR}")
        file(GLOB TIDL_NET_BINS "${TIDL_ARTIFACTS_DIR}/subgraph*_net.bin")
        file(GLOB TIDL_IO_BINS "${TIDL_ARTIFACTS_DIR}/subgraph*_params_1.bin")
        list(SORT TIDL_NET_BINS)
        list(SORT TIDL_IO_BINS)

        # Extract subgraph IDs and embed each pair
        foreach(NET_BIN ${TIDL_NET_BINS})
            get_filename_component(NET_NAME ${NET_BIN} NAME)
            # Extract ID from "subgraphN_net.bin"
            string(REGEX MATCH "subgraph([0-9]+)_net\\.bin" _MATCH "${NET_NAME}")
            set(SG_ID "${CMAKE_MATCH_1}")

            set(IO_BIN "${TIDL_ARTIFACTS_DIR}/subgraph${SG_ID}_params_1.bin")
            if(NOT EXISTS "${IO_BIN}")
                message(WARNING "TIDL io.bin not found for subgraph ${SG_ID}: ${IO_BIN}")
                continue()
            endif()

            # Use unique symbol names per subgraph: tidl_net_N, tidl_io_N
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

    # Compile TIDL firmware symbol stubs
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

    message(STATUS "  TIDL: enabled")
endif()

# Optional: disable floating-point reassociation for lib0.c.
# The cl7x -O2 optimizer reorders float additions in matmul inner loops,
# which produces large numerical divergence in models with ill-conditioned
# weight matrices (e.g. LLM attention Q_proj cond=7M, lm_head sigma_max=626).
# --fp_reassoc=off prevents this reordering at ~27% cycle cost.
# Pass -DFP_REASSOC_OFF=ON from build_dsp_dynmod to enable.
set(LIB0_EXTRA_FLAGS "")
if(FP_REASSOC_OFF)
    set(LIB0_EXTRA_FLAGS "--fp_reassoc=off")
    message(STATUS "  FP reassociation disabled for lib0.c (--fp_reassoc=off)")
endif()
# User-specified flags appended to lib0.c compilation only.
# Usage: cmake -DLIB0_USER_FLAGS="-O3 --opt_for_speed=5 -k" ...
# Useful for testing alternative optimization levels or enabling -k (keep asm).
if(LIB0_USER_FLAGS)
    separate_arguments(LIB0_USER_FLAGS_LIST NATIVE_COMMAND "${LIB0_USER_FLAGS}")
    list(APPEND LIB0_EXTRA_FLAGS ${LIB0_USER_FLAGS_LIST})
    message(STATUS "  lib0.c user flags: ${LIB0_USER_FLAGS}")
endif()

# Compile lib*.c -> lib*.obj
set(LIB_OBJS "")
foreach(LIB_SRC ${LIB_SRCS})
    get_filename_component(LIB_NAME_WE ${LIB_SRC} NAME_WE)
    set(LIB_OBJ "${CMAKE_CURRENT_BINARY_DIR}/${LIB_NAME_WE}.obj")
    add_custom_command(
        OUTPUT "${LIB_OBJ}"
        COMMAND ${CMAKE_CXX_COMPILER}
            ${CXX_FLAGS_LIST}
            ${LIB0_EXTRA_FLAGS}
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

# Link lib*.obj + dsp_syms.out.obj [+ TIDL objs] -> lib0.out
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
