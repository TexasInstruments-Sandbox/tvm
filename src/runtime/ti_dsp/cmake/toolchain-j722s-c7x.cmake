#
# CMake toolchain file for J722S C7x DSP cross-compilation
#
# This toolchain targets the TI J722S SoC's C75 DSP core (C7x architecture).
# It requires TI C7000 compiler and MCU+ SDK.
#
# Usage:
#   mkdir build-c7x && cd build-c7x
#   cmake -DCMAKE_TOOLCHAIN_FILE=../cmake/toolchain-j722s-c7x.cmake ..
#   cmake --build .
#
#   For Release build with -O3:
#   cmake -DCMAKE_TOOLCHAIN_FILE=../cmake/toolchain-j722s-c7x.cmake -DCMAKE_BUILD_TYPE=Release ..
#

# Set the system name to Generic (bare-metal/embedded)
set(CMAKE_SYSTEM_NAME Generic)
set(CMAKE_SYSTEM_PROCESSOR c7x)

# Set TVM DSP target for CMakeLists.txt
set(TVM_DSP_TARGET "c7x" CACHE STRING "DSP target" FORCE)
set(TVM_DSP_DEVICE "j722s" CACHE STRING "Device variant" FORCE)

# TI C7000 compiler path - check environment or use default
set(TI_CGT_C7000_PATH "$ENV{TI_CGT_C7000_PATH}")
# Expand ~ to $HOME (CMake doesn't do shell-style tilde expansion)
if(TI_CGT_C7000_PATH MATCHES "^~")
    string(REGEX REPLACE "^~" "$ENV{HOME}" TI_CGT_C7000_PATH "${TI_CGT_C7000_PATH}")
endif()
if(NOT TI_CGT_C7000_PATH)
    # Try common installation paths
    if(EXISTS "$ENV{HOME}/ti/ccs2040/ccs/tools/compiler/ti-cgt-c7000_5.0.1.LTS")
        set(TI_CGT_C7000_PATH "$ENV{HOME}/ti/ccs2040/ccs/tools/compiler/ti-cgt-c7000_5.0.1.LTS")
    elseif(EXISTS "$ENV{HOME}/ti/ccs2041/ccs/tools/compiler/ti-cgt-c7000_5.0.1.LTS")
        set(TI_CGT_C7000_PATH "$ENV{HOME}/ti/ccs2041/ccs/tools/compiler/ti-cgt-c7000_5.0.1.LTS")
    elseif(EXISTS "$ENV{HOME}/ti/ti-cgt-c7000_5.0.1.LTS")
        set(TI_CGT_C7000_PATH "$ENV{HOME}/ti/ti-cgt-c7000_5.0.1.LTS")
    endif()
endif()

# Verify compiler exists
if(NOT EXISTS "${TI_CGT_C7000_PATH}/bin/cl7x")
    message(FATAL_ERROR "TI C7000 compiler not found at ${TI_CGT_C7000_PATH}/bin/cl7x\n"
                        "Set TI_CGT_C7000_PATH environment variable or edit toolchain file.\n"
                        "Expected: ~/ti/ccs2040/ccs/tools/compiler/ti-cgt-c7000_5.0.1.LTS")
endif()

# MCU+ SDK path (optional for standalone builds)
# For standalone JTAG operation, the SDK is NOT required - standalone startup
# files in j722s/ directory provide self-contained boot/MMU/cache support.
# The SDK is only needed if using SDK drivers (UART, EDMA, etc.) or RTOS.
if(NOT DEFINED MCU_PLUS_SDK_PATH OR NOT EXISTS "${MCU_PLUS_SDK_PATH}")
    set(MCU_PLUS_SDK_PATH "$ENV{MCU_PLUS_SDK_PATH}")
endif()

# Try default installation location if not set
if(NOT MCU_PLUS_SDK_PATH OR NOT EXISTS "${MCU_PLUS_SDK_PATH}")
    if(EXISTS "$ENV{HOME}/ti/mcu_plus_sdk_j722s_11_01_00_07")
        set(MCU_PLUS_SDK_PATH "$ENV{HOME}/ti/mcu_plus_sdk_j722s_11_01_00_07")
    elseif(EXISTS "$ENV{HOME}/ti/MCU_PLUS_SDK_J722S_11_01")
        set(MCU_PLUS_SDK_PATH "$ENV{HOME}/ti/MCU_PLUS_SDK_J722S_11_01")
    endif()
endif()

if(MCU_PLUS_SDK_PATH AND EXISTS "${MCU_PLUS_SDK_PATH}")
    message(STATUS "MCU+ SDK Path: ${MCU_PLUS_SDK_PATH}")
    set(TVM_DSP_C7X_HAS_SDK TRUE CACHE BOOL "MCU+ SDK available" FORCE)
else()
    message(STATUS "MCU+ SDK not found - using standalone mode (no SDK drivers)")
    message(STATUS "  To use SDK, set MCU_PLUS_SDK_PATH environment variable")
    set(TVM_DSP_C7X_HAS_SDK FALSE CACHE BOOL "MCU+ SDK available" FORCE)
endif()

# Set compiler and linker
set(CMAKE_C_COMPILER "${TI_CGT_C7000_PATH}/bin/cl7x")
set(CMAKE_CXX_COMPILER "${TI_CGT_C7000_PATH}/bin/cl7x")
set(CMAKE_AR "${TI_CGT_C7000_PATH}/bin/ar7x")
set(CMAKE_LINKER "${TI_CGT_C7000_PATH}/bin/lnk7x")

# Tell CMake not to test the compiler (cross-compilation)
set(CMAKE_C_COMPILER_WORKS 1)
set(CMAKE_CXX_COMPILER_WORKS 1)
set(CMAKE_TRY_COMPILE_TARGET_TYPE STATIC_LIBRARY)

# Compiler identification
set(CMAKE_C_COMPILER_ID TI)
set(CMAKE_CXX_COMPILER_ID TI)

# C7x (C7524 core on J722S) specific flags
# -mv7524 targets the C7524 core variant (C75 family)
set(C7X_COMMON_FLAGS "-mv7524 --abi=eabi --endian=little --display_error_number")
# Suppress common warnings that don't affect correctness
set(C7X_COMMON_FLAGS "${C7X_COMMON_FLAGS} --diag_suppress=496 --diag_suppress=1311 --diag_suppress=303")
set(C7X_COMMON_FLAGS "${C7X_COMMON_FLAGS} -I${TI_CGT_C7000_PATH}/include")

# J722S-specific flags for FP optimization
# Note: C7x doesn't use --mem_model options (that's C66x specific)
set(C7X_COMMON_FLAGS "${C7X_COMMON_FLAGS} --fp_mode=relaxed")
# Define NULL and SOC
set(C7X_COMMON_FLAGS "${C7X_COMMON_FLAGS} -DNULL=0 -DSOC_J722S")

# C99 mode for C files
set(C7X_C_FLAGS "--c99")

# C flags
set(CMAKE_C_FLAGS_INIT "${C7X_COMMON_FLAGS} ${C7X_C_FLAGS}")
set(CMAKE_C_FLAGS_DEBUG_INIT "-g --opt_level=0")
set(CMAKE_C_FLAGS_RELEASE_INIT "-O3 --auto_inline=500")
set(CMAKE_C_FLAGS_RELWITHDEBINFO_INIT "-g -O2 --symdebug:dwarf")

# C++ flags (no exceptions for bare-metal)
set(CMAKE_CXX_FLAGS_INIT "${C7X_COMMON_FLAGS} --rtti")
set(CMAKE_CXX_FLAGS_DEBUG_INIT "-g --opt_level=0")
set(CMAKE_CXX_FLAGS_RELEASE_INIT "-O3 --auto_inline=500")
set(CMAKE_CXX_FLAGS_RELWITHDEBINFO_INIT "-g -O2 --symdebug:dwarf")

# Assembler
set(CMAKE_ASM_COMPILER "${TI_CGT_C7000_PATH}/bin/cl7x")
set(CMAKE_ASM_FLAGS_INIT "${C7X_COMMON_FLAGS}")

# Archive (static library) command
set(CMAKE_C_CREATE_STATIC_LIBRARY "<CMAKE_AR> r <TARGET> <OBJECTS>")
set(CMAKE_CXX_CREATE_STATIC_LIBRARY "<CMAKE_AR> r <TARGET> <OBJECTS>")

# Compile rules - TI compiler uses different output flag syntax
set(CMAKE_C_COMPILE_OBJECT "<CMAKE_C_COMPILER> <FLAGS> <DEFINES> <INCLUDES> --compile_only --output_file=<OBJECT> <SOURCE>")
set(CMAKE_CXX_COMPILE_OBJECT "<CMAKE_CXX_COMPILER> <FLAGS> <DEFINES> <INCLUDES> --compile_only --output_file=<OBJECT> <SOURCE>")

# Link rules for executable
# C7x linker requires -z flag for linking
set(CMAKE_C_LINK_EXECUTABLE "<CMAKE_LINKER> -z <LINK_FLAGS> --output_file=<TARGET> <OBJECTS> <LINK_LIBRARIES>")
set(CMAKE_CXX_LINK_EXECUTABLE "<CMAKE_LINKER> -z <LINK_FLAGS> --output_file=<TARGET> <OBJECTS> <LINK_LIBRARIES>")

# File extensions
set(CMAKE_C_OUTPUT_EXTENSION ".obj")
set(CMAKE_CXX_OUTPUT_EXTENSION ".obj")
set(CMAKE_STATIC_LIBRARY_PREFIX "")
set(CMAKE_STATIC_LIBRARY_SUFFIX ".lib")
set(CMAKE_EXECUTABLE_SUFFIX ".out")

# Include/define flags (same as gcc)
set(CMAKE_INCLUDE_FLAG_C "-I")
set(CMAKE_INCLUDE_FLAG_CXX "-I")
set(CMAKE_C_DEFINE_FLAG "-D")
set(CMAKE_CXX_DEFINE_FLAG "-D")

# Search paths - don't search host system paths
set(CMAKE_FIND_ROOT_PATH "${TI_CGT_C7000_PATH}")
set(CMAKE_FIND_ROOT_PATH_MODE_PROGRAM NEVER)
set(CMAKE_FIND_ROOT_PATH_MODE_LIBRARY ONLY)
set(CMAKE_FIND_ROOT_PATH_MODE_INCLUDE ONLY)

# Clear host linker flags - force override any inherited values
set(CMAKE_EXE_LINKER_FLAGS "" CACHE STRING "Flags for executable linking" FORCE)
set(CMAKE_SHARED_LINKER_FLAGS "" CACHE STRING "Flags for shared library linking" FORCE)
set(CMAKE_MODULE_LINKER_FLAGS "" CACHE STRING "Flags for module linking" FORCE)
set(CMAKE_EXE_LINKER_FLAGS_INIT "")
set(CMAKE_SHARED_LINKER_FLAGS_INIT "")
set(CMAKE_MODULE_LINKER_FLAGS_INIT "")

# Runtime library path (export for CMakeLists.txt)
set(TI_CGT_C7000_LIB_PATH "${TI_CGT_C7000_PATH}/lib")

message(STATUS "TI C7000 Compiler: ${CMAKE_C_COMPILER}")
message(STATUS "TI C7000 Include: ${TI_CGT_C7000_PATH}/include")
message(STATUS "TI C7000 Lib: ${TI_CGT_C7000_LIB_PATH}")
