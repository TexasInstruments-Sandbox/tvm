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

#[=======================================================================[.rst:
WeightEmbedding
---------------

CMake module for embedding TVM model weights into executables.

This module provides functions to embed weights.bin files into executables
using platform-specific methods:

- Linux (ELF): objcopy to create object file
- macOS (Mach-O): ld with -sectcreate
- C66x DSP (COFF): hex6x or assembly embedding

Functions:
^^^^^^^^^^

.. command:: tvm_dsp_embed_weights

  Embed weights.bin into a target executable::

    tvm_dsp_embed_weights(<target> <weights_file>
      [SYMBOL_PREFIX <prefix>]      # Default: "weights_bin"
      [SECTION_NAME <name>])        # Default: ".rodata" or "__DATA,__weights"

Example:
^^^^^^^^

.. code-block:: cmake

  add_executable(my_model main.cpp lib0.c devc.c)

  # Embed weights.bin into executable
  tvm_dsp_embed_weights(my_model ${CMAKE_CURRENT_SOURCE_DIR}/weights.bin)

  # Enable linker embedding mode
  target_compile_definitions(my_model PRIVATE TVM_DSP_WEIGHTS_LINKER_EMBEDDED)

#]=======================================================================]

include(CMakeParseArguments)

# Store path to scripts at include time (CMAKE_CURRENT_LIST_DIR changes in functions)
set(_TVM_DSP_SCRIPTS_DIR "${CMAKE_CURRENT_LIST_DIR}/../scripts" CACHE INTERNAL "")

#[=======================================================================[.rst:
.. command:: tvm_dsp_embed_weights

  Embed a binary weights file into a target executable.

  .. code-block:: cmake

    tvm_dsp_embed_weights(<target> <weights_file>
      [SYMBOL_PREFIX <prefix>]
      [SECTION_NAME <name>])

  ``<target>``
    Name of the target executable to embed weights into.

  ``<weights_file>``
    Path to the weights.bin file to embed.

  ``SYMBOL_PREFIX``
    Optional prefix for the generated symbols. Default is "weights_bin".
    Generated symbols will be: _binary_<prefix>_start, _binary_<prefix>_end

  ``SECTION_NAME``
    Optional section name for the embedded data.
    Default varies by platform.

#]=======================================================================]
function(tvm_dsp_embed_weights TARGET WEIGHTS_FILE)
  cmake_parse_arguments(EMBED
    ""                              # Options
    "SYMBOL_PREFIX;SECTION_NAME"    # Single-value args
    ""                              # Multi-value args
    ${ARGN}
  )

  # Default symbol prefix
  if(NOT EMBED_SYMBOL_PREFIX)
    set(EMBED_SYMBOL_PREFIX "weights_bin")
  endif()

  # Check if weights file exists
  if(NOT EXISTS "${WEIGHTS_FILE}")
    message(WARNING "Weights file not found: ${WEIGHTS_FILE}")
    message(STATUS "Weight embedding will be skipped - no weights.bin")
    return()
  endif()

  # Get file size for logging
  file(SIZE "${WEIGHTS_FILE}" WEIGHTS_SIZE)
  math(EXPR WEIGHTS_SIZE_KB "${WEIGHTS_SIZE} / 1024")
  message(STATUS "Embedding weights: ${WEIGHTS_FILE} (${WEIGHTS_SIZE_KB} KB)")

  # Generate output object file path
  get_filename_component(WEIGHTS_BASENAME "${WEIGHTS_FILE}" NAME_WE)
  set(WEIGHTS_OBJ "${CMAKE_CURRENT_BINARY_DIR}/${WEIGHTS_BASENAME}.o")

  # Platform-specific embedding
  if(CMAKE_SYSTEM_NAME STREQUAL "Generic" AND CMAKE_SYSTEM_PROCESSOR MATCHES "c6.*|c7.*")
    # TI C66x/C7x DSP - use assembly embedding (works for cross-compilation)
    _tvm_dsp_embed_weights_ti_asm(
      "${WEIGHTS_FILE}" "${WEIGHTS_OBJ}" "${EMBED_SYMBOL_PREFIX}")
    target_sources(${TARGET} PRIVATE "${WEIGHTS_OBJ}")

  elseif(CMAKE_SYSTEM_NAME STREQUAL "Linux")
    # Linux/ELF
    _tvm_dsp_embed_weights_elf(
      "${WEIGHTS_FILE}" "${WEIGHTS_OBJ}" "${EMBED_SYMBOL_PREFIX}")
    target_sources(${TARGET} PRIVATE "${WEIGHTS_OBJ}")

  elseif(CMAKE_SYSTEM_NAME STREQUAL "Darwin")
    # macOS/Mach-O
    _tvm_dsp_embed_weights_macho(
      "${WEIGHTS_FILE}" "${WEIGHTS_OBJ}" "${EMBED_SYMBOL_PREFIX}")
    target_sources(${TARGET} PRIVATE "${WEIGHTS_OBJ}")

  elseif(CMAKE_SYSTEM_NAME STREQUAL "Windows")
    # Windows/PE - use rc file
    _tvm_dsp_embed_weights_windows(
      "${WEIGHTS_FILE}" "${WEIGHTS_OBJ}" "${EMBED_SYMBOL_PREFIX}")
    target_sources(${TARGET} PRIVATE "${WEIGHTS_OBJ}")

  else()
    message(WARNING "Platform ${CMAKE_SYSTEM_NAME} not supported for weight embedding")
    message(STATUS "Falling back to filesystem weight loading")
  endif()

  # Add compile definition to enable linker embedding
  target_compile_definitions(${TARGET} PRIVATE TVM_DSP_WEIGHTS_LINKER_EMBEDDED)
endfunction()

#[=======================================================================[
Internal function: TI C66x/C7x assembly-based weight embedding
Uses .byte directives which assemble much faster than compiling C arrays
#]=======================================================================]
function(_tvm_dsp_embed_weights_ti_asm WEIGHTS_FILE WEIGHTS_OBJ SYMBOL_PREFIX)
  get_filename_component(WEIGHTS_NAME "${WEIGHTS_FILE}" NAME)
  set(ASM_FILE "${CMAKE_CURRENT_BINARY_DIR}/${SYMBOL_PREFIX}_data.asm")

  # Find Python for binary to assembly conversion
  find_package(Python3 COMPONENTS Interpreter REQUIRED)

  # Path to bin_to_asm.py script (use cached path from include time)
  set(BIN_TO_ASM_SCRIPT "${_TVM_DSP_SCRIPTS_DIR}/bin_to_asm.py")
  if(NOT EXISTS "${BIN_TO_ASM_SCRIPT}")
    message(FATAL_ERROR "bin_to_asm.py not found at ${BIN_TO_ASM_SCRIPT}")
  endif()

  # Create custom command to generate assembly file
  add_custom_command(
    OUTPUT "${ASM_FILE}"
    COMMAND ${Python3_EXECUTABLE} "${BIN_TO_ASM_SCRIPT}"
      "${WEIGHTS_FILE}" "${ASM_FILE}" "${SYMBOL_PREFIX}"
    DEPENDS "${WEIGHTS_FILE}" "${BIN_TO_ASM_SCRIPT}"
    COMMENT "Generating TI assembly for ${WEIGHTS_NAME}"
    VERBATIM
  )

  # Determine silicon version based on processor
  # C66x: --silicon_version=6600
  # C7x:  -mv7524 (J722S) or appropriate variant
  if(CMAKE_SYSTEM_PROCESSOR MATCHES "c7.*")
    # C7x uses -mv flag instead of --silicon_version
    set(TI_TARGET_NAME "TI C7x")
    set(TI_ASM_CMD
      ${CMAKE_C_COMPILER}
      -mv7524
      --abi=eabi
      -c "${ASM_FILE}"
      --output_file="${WEIGHTS_OBJ}"
    )
  else()
    # C66x
    set(TI_TARGET_NAME "TI C66x")
    set(TI_ASM_CMD
      ${CMAKE_C_COMPILER}
      --silicon_version=6600
      --abi=eabi
      -c "${ASM_FILE}"
      --output_file="${WEIGHTS_OBJ}"
    )
  endif()

  # Assemble to object using compiler - produces ELF format with --abi=eabi
  get_filename_component(ASM_FILE_NAME "${ASM_FILE}" NAME_WE)
  set(TI_OBJ_FILE "${CMAKE_CURRENT_BINARY_DIR}/${ASM_FILE_NAME}.obj")

  add_custom_command(
    OUTPUT "${WEIGHTS_OBJ}"
    COMMAND ${TI_ASM_CMD}
    DEPENDS "${ASM_FILE}"
    WORKING_DIRECTORY "${CMAKE_CURRENT_BINARY_DIR}"
    COMMENT "Assembling ${SYMBOL_PREFIX}_data.asm for ${TI_TARGET_NAME}"
  )

  set_source_files_properties("${WEIGHTS_OBJ}" PROPERTIES
    GENERATED TRUE
    EXTERNAL_OBJECT TRUE
  )
endfunction()

#[=======================================================================[
Internal function: ELF-based weight embedding using objcopy
#]=======================================================================]
function(_tvm_dsp_embed_weights_elf WEIGHTS_FILE WEIGHTS_OBJ SYMBOL_PREFIX)
  # Find objcopy
  if(CMAKE_OBJCOPY)
    set(OBJCOPY_CMD "${CMAKE_OBJCOPY}")
  else()
    find_program(OBJCOPY_CMD NAMES
      objcopy
      llvm-objcopy
      ${CMAKE_C_COMPILER_TARGET}-objcopy
    )
  endif()

  if(NOT OBJCOPY_CMD)
    message(FATAL_ERROR "objcopy not found - required for weight embedding")
  endif()

  # Determine output format based on target architecture
  if(CMAKE_SIZEOF_VOID_P EQUAL 8)
    set(OBJCOPY_FORMAT "elf64-x86-64")
    set(OBJCOPY_ARCH "i386:x86-64")
  elseif(CMAKE_SYSTEM_PROCESSOR MATCHES "arm|aarch64")
    if(CMAKE_SIZEOF_VOID_P EQUAL 8)
      set(OBJCOPY_FORMAT "elf64-littleaarch64")
      set(OBJCOPY_ARCH "aarch64")
    else()
      set(OBJCOPY_FORMAT "elf32-littlearm")
      set(OBJCOPY_ARCH "arm")
    endif()
  elseif(CMAKE_SYSTEM_PROCESSOR MATCHES "c6.*|c7.*")
    # TI C66x/C7x DSP
    set(OBJCOPY_FORMAT "elf32-tic6x-elf-le")
    set(OBJCOPY_ARCH "tic6x")
  else()
    set(OBJCOPY_FORMAT "elf64-x86-64")
    set(OBJCOPY_ARCH "i386:x86-64")
  endif()

  # Get filename for symbol generation
  get_filename_component(WEIGHTS_NAME "${WEIGHTS_FILE}" NAME)
  string(REGEX REPLACE "[^a-zA-Z0-9_]" "_" WEIGHTS_SYMBOL "${WEIGHTS_NAME}")

  # Create custom command to convert binary to object
  add_custom_command(
    OUTPUT "${WEIGHTS_OBJ}"
    COMMAND ${OBJCOPY_CMD}
      -I binary
      -O ${OBJCOPY_FORMAT}
      -B ${OBJCOPY_ARCH}
      --rename-section .data=.rodata,alloc,load,readonly,data,contents
      --redefine-sym _binary_${WEIGHTS_SYMBOL}_start=_binary_${SYMBOL_PREFIX}_start
      --redefine-sym _binary_${WEIGHTS_SYMBOL}_end=_binary_${SYMBOL_PREFIX}_end
      --redefine-sym _binary_${WEIGHTS_SYMBOL}_size=_binary_${SYMBOL_PREFIX}_size
      "${WEIGHTS_FILE}"
      "${WEIGHTS_OBJ}"
    DEPENDS "${WEIGHTS_FILE}"
    COMMENT "Embedding ${WEIGHTS_NAME} as ELF object (${OBJCOPY_FORMAT})"
    VERBATIM
  )

  # Set property so parent knows about generated file
  set_source_files_properties("${WEIGHTS_OBJ}" PROPERTIES
    GENERATED TRUE
    EXTERNAL_OBJECT TRUE
  )
endfunction()

#[=======================================================================[
Internal function: Mach-O weight embedding for macOS
#]=======================================================================]
function(_tvm_dsp_embed_weights_macho WEIGHTS_FILE WEIGHTS_OBJ SYMBOL_PREFIX)
  # For macOS, we create an assembly file that includes the binary data
  # This avoids issues with ld -sectcreate symbol naming

  get_filename_component(WEIGHTS_NAME "${WEIGHTS_FILE}" NAME)
  set(ASM_FILE "${CMAKE_CURRENT_BINARY_DIR}/${SYMBOL_PREFIX}.s")

  # Detect architecture
  if(CMAKE_OSX_ARCHITECTURES)
    list(GET CMAKE_OSX_ARCHITECTURES 0 ARCH)
  elseif(CMAKE_SYSTEM_PROCESSOR STREQUAL "arm64")
    set(ARCH "arm64")
  else()
    set(ARCH "x86_64")
  endif()

  # Create assembly source file with binary inclusion
  file(WRITE "${ASM_FILE}"
"# Auto-generated assembly for embedding ${WEIGHTS_NAME}
.section __DATA,__weights
.global _binary_${SYMBOL_PREFIX}_start
.global _binary_${SYMBOL_PREFIX}_end

.balign 16
_binary_${SYMBOL_PREFIX}_start:
.incbin \"${WEIGHTS_FILE}\"
_binary_${SYMBOL_PREFIX}_end:
.balign 16
")

  # Create object from assembly
  add_custom_command(
    OUTPUT "${WEIGHTS_OBJ}"
    COMMAND ${CMAKE_C_COMPILER} -c -arch ${ARCH} "${ASM_FILE}" -o "${WEIGHTS_OBJ}"
    DEPENDS "${WEIGHTS_FILE}" "${ASM_FILE}"
    COMMENT "Embedding ${WEIGHTS_NAME} as Mach-O object"
    VERBATIM
  )

  set_source_files_properties("${WEIGHTS_OBJ}" PROPERTIES
    GENERATED TRUE
    EXTERNAL_OBJECT TRUE
  )
endfunction()

#[=======================================================================[
Internal function: Windows PE weight embedding
#]=======================================================================]
function(_tvm_dsp_embed_weights_windows WEIGHTS_FILE WEIGHTS_OBJ SYMBOL_PREFIX)
  # For Windows, create an RC file with RCDATA resource
  get_filename_component(WEIGHTS_NAME "${WEIGHTS_FILE}" NAME)
  set(RC_FILE "${CMAKE_CURRENT_BINARY_DIR}/${SYMBOL_PREFIX}.rc")

  file(WRITE "${RC_FILE}"
"// Auto-generated resource file for embedding ${WEIGHTS_NAME}
#define WEIGHTS_RESOURCE 1000
WEIGHTS_RESOURCE RCDATA \"${WEIGHTS_FILE}\"
")

  # Use resource compiler
  if(MSVC)
    add_custom_command(
      OUTPUT "${WEIGHTS_OBJ}"
      COMMAND rc /fo "${WEIGHTS_OBJ}" "${RC_FILE}"
      DEPENDS "${WEIGHTS_FILE}" "${RC_FILE}"
      COMMENT "Embedding ${WEIGHTS_NAME} as Windows resource"
      VERBATIM
    )
  else()
    # MinGW
    add_custom_command(
      OUTPUT "${WEIGHTS_OBJ}"
      COMMAND windres "${RC_FILE}" "${WEIGHTS_OBJ}"
      DEPENDS "${WEIGHTS_FILE}" "${RC_FILE}"
      COMMENT "Embedding ${WEIGHTS_NAME} as Windows resource (MinGW)"
      VERBATIM
    )
  endif()

  set_source_files_properties("${WEIGHTS_OBJ}" PROPERTIES
    GENERATED TRUE
    EXTERNAL_OBJECT TRUE
  )
endfunction()

#[=======================================================================[.rst:
.. command:: tvm_dsp_configure_weights

  Configure weight loading mode for a target.

  .. code-block:: cmake

    tvm_dsp_configure_weights(<target>
      MODE <EMBEDDED|FILESYSTEM|NONE>
      [WEIGHTS_FILE <path>]
      [WEIGHTS_PATH <runtime_path>])

  ``MODE``
    Weight loading mode:
    - EMBEDDED: Embed weights in binary (recommended for production)
    - FILESYSTEM: Load from file at runtime (for development)
    - NONE: No weights (for models without constants)

  ``WEIGHTS_FILE``
    Path to weights.bin file (required for EMBEDDED mode)

  ``WEIGHTS_PATH``
    Runtime path to weights.bin (for FILESYSTEM mode)
    Default: "weights.bin"

#]=======================================================================]
function(tvm_dsp_configure_weights TARGET)
  cmake_parse_arguments(CFG
    ""
    "MODE;WEIGHTS_FILE;WEIGHTS_PATH"
    ""
    ${ARGN}
  )

  if(NOT CFG_MODE)
    set(CFG_MODE "NONE")
  endif()

  if(CFG_MODE STREQUAL "EMBEDDED")
    if(NOT CFG_WEIGHTS_FILE)
      message(FATAL_ERROR "WEIGHTS_FILE required for EMBEDDED mode")
    endif()
    tvm_dsp_embed_weights(${TARGET} "${CFG_WEIGHTS_FILE}")

  elseif(CFG_MODE STREQUAL "FILESYSTEM")
    target_compile_definitions(${TARGET} PRIVATE TVM_DSP_WEIGHTS_FILESYSTEM)
    if(CFG_WEIGHTS_PATH)
      target_compile_definitions(${TARGET} PRIVATE
        TVM_DSP_WEIGHTS_PATH="${CFG_WEIGHTS_PATH}")
    endif()

  elseif(CFG_MODE STREQUAL "NONE")
    # No weights - nothing to configure
    message(STATUS "Weight loading disabled for ${TARGET}")

  else()
    message(FATAL_ERROR "Unknown weight mode: ${CFG_MODE}")
  endif()
endfunction()
