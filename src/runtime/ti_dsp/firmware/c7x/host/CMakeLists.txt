# C7x Compute Service - Host Application
#
# Build:
#   ./build.sh          - Cross-compile for ARM64
#   ./build.sh native   - Native compile (on target)

cmake_minimum_required(VERSION 3.10)
project(c7x_compute_host CXX)

set(CMAKE_CXX_STANDARD 14)
set(CMAKE_CXX_STANDARD_REQUIRED ON)

# Include directories
include_directories(
    ${CMAKE_CURRENT_SOURCE_DIR}/include
    ${CMAKE_CURRENT_SOURCE_DIR}/../common
)

# Source files
set(SOURCES
    src/c7x_compute_client.cpp
    src/rpmsg_wrapper.cpp
    src/c7x_compute_cli.cpp
)

# Build executable
add_executable(c7x_compute ${SOURCES})

# Link libraries
target_link_libraries(c7x_compute pthread)

# Installation
install(TARGETS c7x_compute DESTINATION bin)
