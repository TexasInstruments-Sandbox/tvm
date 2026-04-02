#
# CMake toolchain file for AWRL6844 C66x DSP cross-compilation
#
# This toolchain targets the TI AWRL6844 mmWave radar sensor's C66x DSP core.
# It requires TI C6000 compiler and MMWAVE-L-SDK-6.
#
# Usage:
#   mkdir build-awrl6844 && cd build-awrl6844
#   cmake -DCMAKE_TOOLCHAIN_FILE=../cmake/toolchain-awrl6844.cmake ..
#   cmake --build .
#
#   For Release build with -O3:
#   cmake -DCMAKE_TOOLCHAIN_FILE=../cmake/toolchain-awrl6844.cmake -DCMAKE_BUILD_TYPE=Release ..
#

# Set the system name to Generic (bare-metal/embedded)
set(CMAKE_SYSTEM_NAME Generic)
set(CMAKE_SYSTEM_PROCESSOR c6600)

# Set TVM DSP target for CMakeLists.txt
set(TVM_DSP_TARGET "c66x" CACHE STRING "DSP target" FORCE)
set(TVM_DSP_DEVICE "awrl6844" CACHE STRING "Device variant" FORCE)

# TI C6000 compiler path - check environment or use default
set(TI_CGT_C6000_PATH "$ENV{TI_CGT_C6000_PATH}")
# Expand ~ to $HOME (CMake doesn't do shell-style tilde expansion)
if(TI_CGT_C6000_PATH MATCHES "^~")
    string(REGEX REPLACE "^~" "$ENV{HOME}" TI_CGT_C6000_PATH "${TI_CGT_C6000_PATH}")
endif()
if(NOT TI_CGT_C6000_PATH)
    # Try common installation paths
    if(EXISTS "$ENV{HOME}/ti/ccs2041/ccs/tools/compiler/ti-cgt-c6000_8.5.0.LTS")
        set(TI_CGT_C6000_PATH "$ENV{HOME}/ti/ccs2041/ccs/tools/compiler/ti-cgt-c6000_8.5.0.LTS")
    elseif(EXISTS "$ENV{HOME}/ti/ccs2050/ccs/tools/compiler/ti-cgt-c6000_8.5.0.LTS")
        set(TI_CGT_C6000_PATH "$ENV{HOME}/ti/ccs2050/ccs/tools/compiler/ti-cgt-c6000_8.5.0.LTS")
    elseif(EXISTS "$ENV{HOME}/ti/ccs2040/ccs/tools/compiler/ti-cgt-c6000_8.5.0.LTS")
        set(TI_CGT_C6000_PATH "$ENV{HOME}/ti/ccs2040/ccs/tools/compiler/ti-cgt-c6000_8.5.0.LTS")
    elseif(EXISTS "$ENV{HOME}/ti/ti-cgt-c6000_8.5.0.LTS")
        set(TI_CGT_C6000_PATH "$ENV{HOME}/ti/ti-cgt-c6000_8.5.0.LTS")
    endif()
endif()

# Verify compiler exists
if(NOT EXISTS "${TI_CGT_C6000_PATH}/bin/cl6x")
    message(FATAL_ERROR "TI C6000 compiler not found at ${TI_CGT_C6000_PATH}/bin/cl6x\n"
                        "Set TI_CGT_C6000_PATH environment variable or edit toolchain file.\n"
                        "Expected: ~/ti/ccs2050/ccs/tools/compiler/ti-cgt-c6000_8.5.0.LTS")
endif()

# MMWAVE-L-SDK-6 path (required for AWRL6844)
if(NOT DEFINED MMWAVE_SDK_PATH OR NOT EXISTS "${MMWAVE_SDK_PATH}")
    set(MMWAVE_SDK_PATH "$ENV{MMWAVE_SDK_PATH}")
endif()

# Try default installation location if not set
if(NOT MMWAVE_SDK_PATH OR NOT EXISTS "${MMWAVE_SDK_PATH}")
    if(EXISTS "$ENV{HOME}/ti/MMWAVE_L_SDK_06_01_00_05")
        set(MMWAVE_SDK_PATH "$ENV{HOME}/ti/MMWAVE_L_SDK_06_01_00_05")
    elseif(EXISTS "$ENV{HOME}/ti/mmwave_l_sdk_06_01_00_05")
        set(MMWAVE_SDK_PATH "$ENV{HOME}/ti/mmwave_l_sdk_06_01_00_05")
    endif()
endif()

if(MMWAVE_SDK_PATH AND EXISTS "${MMWAVE_SDK_PATH}")
    message(STATUS "MMWAVE-L-SDK-6 Path: ${MMWAVE_SDK_PATH}")
else()
    message(WARNING "MMWAVE-L-SDK-6 not found - AWRL6844 build will fail without SDK libraries\n"
                    "Set MMWAVE_SDK_PATH environment variable to your SDK installation path.")
endif()

# Set compiler and linker
set(CMAKE_C_COMPILER "${TI_CGT_C6000_PATH}/bin/cl6x")
set(CMAKE_CXX_COMPILER "${TI_CGT_C6000_PATH}/bin/cl6x")
set(CMAKE_AR "${TI_CGT_C6000_PATH}/bin/ar6x")
set(CMAKE_LINKER "${TI_CGT_C6000_PATH}/bin/lnk6x")

# Tell CMake not to test the compiler (cross-compilation)
set(CMAKE_C_COMPILER_WORKS 1)
set(CMAKE_CXX_COMPILER_WORKS 1)
set(CMAKE_TRY_COMPILE_TARGET_TYPE STATIC_LIBRARY)

# Compiler identification
set(CMAKE_C_COMPILER_ID TI)
set(CMAKE_CXX_COMPILER_ID TI)

# C66x specific flags
set(C66X_COMMON_FLAGS "--silicon_version=6600 --abi=eabi -ml3 -mo --display_error_number")
# Suppress common warnings that don't affect correctness
set(C66X_COMMON_FLAGS "${C66X_COMMON_FLAGS} --diag_suppress=496 --diag_suppress=1311 --diag_suppress=303")
set(C66X_COMMON_FLAGS "${C66X_COMMON_FLAGS} -I${TI_CGT_C6000_PATH}/include")

# AWRL6844-specific flags for memory model and FP optimization
# Use far data model to avoid relocation overflow for large embedded data
set(C66X_COMMON_FLAGS "${C66X_COMMON_FLAGS} --mem_model:const=data --mem_model:data=far")
set(C66X_COMMON_FLAGS "${C66X_COMMON_FLAGS} --fp_mode=relaxed")
# Define NULL for SDK headers
set(C66X_COMMON_FLAGS "${C66X_COMMON_FLAGS} -DNULL=0")

# C99 mode for C files
set(C66X_C_FLAGS "--c99")

# C flags
set(CMAKE_C_FLAGS_INIT "${C66X_COMMON_FLAGS} ${C66X_C_FLAGS}")
set(CMAKE_C_FLAGS_DEBUG_INIT "-g --opt_level=0")
set(CMAKE_C_FLAGS_RELEASE_INIT "-O3 --opt_for_speed=5 --auto_inline=1000")
set(CMAKE_C_FLAGS_RELWITHDEBINFO_INIT "-g -O2")

# C++ flags (no exceptions for bare-metal)
set(CMAKE_CXX_FLAGS_INIT "${C66X_COMMON_FLAGS} --rtti")
set(CMAKE_CXX_FLAGS_DEBUG_INIT "-g --opt_level=0")
set(CMAKE_CXX_FLAGS_RELEASE_INIT "-O3 --opt_for_speed=5 --auto_inline=1000")
set(CMAKE_CXX_FLAGS_RELWITHDEBINFO_INIT "-g -O2")

# Assembler
set(CMAKE_ASM_COMPILER "${TI_CGT_C6000_PATH}/bin/cl6x")
set(CMAKE_ASM_FLAGS_INIT "${C66X_COMMON_FLAGS}")

# Archive (static library) command
set(CMAKE_C_CREATE_STATIC_LIBRARY "<CMAKE_AR> r <TARGET> <OBJECTS>")
set(CMAKE_CXX_CREATE_STATIC_LIBRARY "<CMAKE_AR> r <TARGET> <OBJECTS>")

# Compile rules - TI compiler uses different output flag syntax
set(CMAKE_C_COMPILE_OBJECT "<CMAKE_C_COMPILER> <FLAGS> <DEFINES> <INCLUDES> --compile_only --output_file=<OBJECT> <SOURCE>")
set(CMAKE_CXX_COMPILE_OBJECT "<CMAKE_CXX_COMPILER> <FLAGS> <DEFINES> <INCLUDES> --compile_only --output_file=<OBJECT> <SOURCE>")

# Link rules for executable
set(CMAKE_C_LINK_EXECUTABLE "<CMAKE_LINKER> <LINK_FLAGS> --output_file=<TARGET> <OBJECTS> <LINK_LIBRARIES>")
set(CMAKE_CXX_LINK_EXECUTABLE "<CMAKE_LINKER> <LINK_FLAGS> --output_file=<TARGET> <OBJECTS> <LINK_LIBRARIES>")

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
set(CMAKE_FIND_ROOT_PATH "${TI_CGT_C6000_PATH}")
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
set(TI_CGT_C6000_LIB_PATH "${TI_CGT_C6000_PATH}/lib")

message(STATUS "TI C6000 Compiler: ${CMAKE_C_COMPILER}")
message(STATUS "TI C6000 Include: ${TI_CGT_C6000_PATH}/include")
message(STATUS "TI C6000 Lib: ${TI_CGT_C6000_LIB_PATH}")
