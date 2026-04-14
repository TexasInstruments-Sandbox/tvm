# C7x Compute Service - ARM Client
#
# Build:
#   ./build.sh          - Cross-compile for ARM64
#   ./build.sh native   - Native compile (on target)

cmake_minimum_required(VERSION 3.10)
project(c7x_compute_arm CXX)

set(CMAKE_CXX_STANDARD 14)
set(CMAKE_CXX_STANDARD_REQUIRED ON)

# ---------------------------------------------------------------------------
# Shared library: IPC client + C++ Module runtime
# ---------------------------------------------------------------------------
# Sources for the shared library (no CLI code)
set(LIB_SOURCES
    src/c7x_compute_client.cpp
    src/rpmsg_wrapper.cpp
    src/c7x_runtime.cc
)

add_library(c7x_arm_runtime SHARED ${LIB_SOURCES})
target_include_directories(c7x_arm_runtime PUBLIC
    ${CMAKE_CURRENT_SOURCE_DIR}/include
)
target_include_directories(c7x_arm_runtime PRIVATE
    ${CMAKE_CURRENT_SOURCE_DIR}/src
    ${CMAKE_CURRENT_SOURCE_DIR}/../common
)
target_link_libraries(c7x_arm_runtime PRIVATE pthread)
set_target_properties(c7x_arm_runtime PROPERTIES SOVERSION 1)

# ---------------------------------------------------------------------------
# CLI executable — links the shared library (no logic change)
# ---------------------------------------------------------------------------
add_executable(c7x_compute src/c7x_compute_cli.cpp)
target_include_directories(c7x_compute PRIVATE
    ${CMAKE_CURRENT_SOURCE_DIR}/include
    ${CMAKE_CURRENT_SOURCE_DIR}/../common
)
target_link_libraries(c7x_compute PRIVATE c7x_arm_runtime)

# ---------------------------------------------------------------------------
# Test binary — exercises c7x::Module C++ API on ARM
# ---------------------------------------------------------------------------
if(EXISTS "${CMAKE_CURRENT_SOURCE_DIR}/test/test_c7x_runtime.cpp")
    add_executable(test_c7x_runtime test/test_c7x_runtime.cpp)
    target_link_libraries(test_c7x_runtime PRIVATE c7x_arm_runtime)
    install(TARGETS test_c7x_runtime RUNTIME DESTINATION bin)
endif()

# ---------------------------------------------------------------------------
# Installation
# ---------------------------------------------------------------------------
install(TARGETS c7x_arm_runtime LIBRARY DESTINATION lib)
install(FILES include/c7x_runtime.h DESTINATION include)
install(TARGETS c7x_compute RUNTIME DESTINATION bin)
