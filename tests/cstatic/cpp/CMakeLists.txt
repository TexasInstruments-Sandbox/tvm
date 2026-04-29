cmake_minimum_required(VERSION 3.18)
project(CStaticTestExecutable CXX C)

# Set C++ standard
set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)

# Enable compile commands database for IDE support
set(CMAKE_EXPORT_COMPILE_COMMANDS ON)

# TVM_HOME: use environment variable, or default to repo root (../../..)
if(DEFINED ENV{TVM_HOME})
    set(TVM_HOME "$ENV{TVM_HOME}" CACHE PATH "Path to TVM installation" FORCE)
else()
    get_filename_component(TVM_HOME "${CMAKE_CURRENT_SOURCE_DIR}/../../.." ABSOLUTE)
    set(TVM_HOME "${TVM_HOME}" CACHE PATH "Path to TVM installation")
endif()

# Verify TVM_HOME exists
if(NOT EXISTS "${TVM_HOME}")
    message(FATAL_ERROR "TVM_HOME directory not found: ${TVM_HOME}")
endif()

# Set build type to Release by default
if(NOT CMAKE_BUILD_TYPE)
    set(CMAKE_BUILD_TYPE Release CACHE STRING "Choose the type of build: Debug Release RelWithDebInfo MinSizeRel" FORCE)
endif()

# Set compiler flags for different build modes
set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -w -fpermissive")

# Release mode: Maximum optimization
set(CMAKE_CXX_FLAGS_RELEASE "-O3 -DNDEBUG -march=native -mtune=native")
set(CMAKE_C_FLAGS_RELEASE "-O3 -DNDEBUG -march=native -mtune=native")

# Debug mode: Debug symbols, minimal optimization, debugging support
set(CMAKE_CXX_FLAGS_DEBUG "-O0 -g3 -DDEBUG -fno-omit-frame-pointer")
set(CMAKE_C_FLAGS_DEBUG "-O0 -g3 -DDEBUG -fno-omit-frame-pointer")

# RelWithDebInfo mode: Optimized with debug info (good for profiling)
set(CMAKE_CXX_FLAGS_RELWITHDEBINFO "-O2 -g -DNDEBUG -fno-omit-frame-pointer")
set(CMAKE_C_FLAGS_RELWITHDEBINFO "-O2 -g -DNDEBUG -fno-omit-frame-pointer")

# MinSizeRel mode: Optimize for size
set(CMAKE_CXX_FLAGS_MINSIZEREL "-Os -DNDEBUG")
set(CMAKE_C_FLAGS_MINSIZEREL "-Os -DNDEBUG")

# Add conditional compilation flag for tuple returns
option(MODEL_RETURNS_TUPLE "Enable tuple return support" OFF)
if(MODEL_RETURNS_TUPLE)
    add_compile_definitions(MODEL_RETURNS_TUPLE)
endif()

# Add option for model entry function name
set(MODEL_ENTRY_FUNCTION "main" CACHE STRING "Name of the model entry function")
add_compile_definitions(MODEL_ENTRY_FUNCTION=${MODEL_ENTRY_FUNCTION})

# Add option for number of model inputs
set(MODEL_NUM_INPUTS "1" CACHE STRING "Number of model inputs")
add_compile_definitions(MODEL_NUM_INPUTS=${MODEL_NUM_INPUTS})

# Include directories
include_directories(
    ${TVM_HOME}/include
    ${TVM_HOME}/3rdparty/tvm-ffi/include
    ${TVM_HOME}/3rdparty/tvm-ffi/3rdparty/dlpack/include
    ${TVM_HOME}/3rdparty/dmlc-core/include
    ${TVM_HOME}/3rdparty/cnpy
)

# Define source files - compile .c files as C++
set(CXX_SRC_FILES
    main.cpp
)

set(C_AS_CXX_SRC_FILES
    lib0.c
    devc.c
)

# Create executable
add_executable(cg_static ${CXX_SRC_FILES} ${C_AS_CXX_SRC_FILES})

# Force .c files to be compiled as C++ (similar to g++ -x c++ in original Makefile)
set_source_files_properties(${C_AS_CXX_SRC_FILES} PROPERTIES LANGUAGE CXX)

# Find required libraries
set(TVM_RUNTIME_LIB "${TVM_HOME}/build/libtvm_runtime.a")
set(BACKTRACE_LIB "${TVM_HOME}/build/3rdparty/tvm-ffi/libbacktrace/lib/libbacktrace.a")
set(TVM_FFI_LIB "${TVM_HOME}/build/lib/libtvm_ffi_static.a")
set(CNPY_LIB "${TVM_HOME}/3rdparty/cnpy/build/libcnpy.a")

# Verify required libraries exist
foreach(lib ${TVM_RUNTIME_LIB} ${TVM_FFI_LIB} ${BACKTRACE_LIB} ${CNPY_LIB})
    if(NOT EXISTS "${lib}")
        message(FATAL_ERROR "Required library not found: ${lib}")
    endif()
endforeach()

# Platform-specific linking flags for whole-archive
if(APPLE)
    # macOS: Use -all_load to include all symbols from static libraries
    target_link_options(cg_static PRIVATE
        -Wl,-all_load ${TVM_RUNTIME_LIB}
        -Wl,-all_load
    )
    target_link_libraries(cg_static PRIVATE
        ${BACKTRACE_LIB}
        ${CNPY_LIB}
        z
    )
elseif(UNIX AND NOT APPLE)
    # Linux: Use --whole-archive to include all symbols from static libraries
    target_link_options(cg_static PRIVATE
        -Wl,--whole-archive ${TVM_RUNTIME_LIB} ${TVM_FFI_LIB} -Wl,--no-whole-archive
    )
    target_link_libraries(cg_static PRIVATE
        ${BACKTRACE_LIB}
        ${CNPY_LIB}
        z
    )
endif()

# Add custom targets for convenience
add_custom_target(run
    COMMAND cg_static
    DEPENDS cg_static
    WORKING_DIRECTORY ${CMAKE_CURRENT_BINARY_DIR}
    COMMENT "Running cg_static executable"
)

# Clean target (CMake provides this automatically, but we can add custom behavior)
set_directory_properties(PROPERTIES
    ADDITIONAL_MAKE_CLEAN_FILES "cg_static;../cg_static;../input.npy;../weights.bin;../cmake.log;../devc.c;../lib0.c"
)


# Print configuration information
message(STATUS "TVM_HOME: ${TVM_HOME}")
message(STATUS "BUILD_TYPE: ${CMAKE_BUILD_TYPE}")
message(STATUS "MODEL_RETURNS_TUPLE: ${MODEL_RETURNS_TUPLE}")
message(STATUS "MODEL_ENTRY_FUNCTION: ${MODEL_ENTRY_FUNCTION}")
message(STATUS "MODEL_NUM_INPUTS: ${MODEL_NUM_INPUTS}")
message(STATUS "CMAKE_SYSTEM_NAME: ${CMAKE_SYSTEM_NAME}")
message(STATUS "CMAKE_CXX_COMPILER: ${CMAKE_CXX_COMPILER}")
message(STATUS "CXX_FLAGS: ${CMAKE_CXX_FLAGS}")
if(CMAKE_BUILD_TYPE STREQUAL "Release")
    message(STATUS "CXX_FLAGS_RELEASE: ${CMAKE_CXX_FLAGS_RELEASE}")
elseif(CMAKE_BUILD_TYPE STREQUAL "Debug")
    message(STATUS "CXX_FLAGS_DEBUG: ${CMAKE_CXX_FLAGS_DEBUG}")
elseif(CMAKE_BUILD_TYPE STREQUAL "RelWithDebInfo")
    message(STATUS "CXX_FLAGS_RELWITHDEBINFO: ${CMAKE_CXX_FLAGS_RELWITHDEBINFO}")
elseif(CMAKE_BUILD_TYPE STREQUAL "MinSizeRel")
    message(STATUS "CXX_FLAGS_MINSIZEREL: ${CMAKE_CXX_FLAGS_MINSIZEREL}")
endif()

add_custom_target(copy_binary
    COMMAND ${CMAKE_COMMAND} -E copy $<TARGET_FILE:cg_static> ${CMAKE_CURRENT_SOURCE_DIR}/cg_static
    DEPENDS cg_static
    COMMENT "Copying cg_static to parent directory"
)
