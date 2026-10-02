# cmake/toolchain-c7000.cmake
# Toolchain file for TI C7000 Code Generation Tools (CGT)

# Set the target system name (generic is usually fine for baremetal/RTOS)
set(CMAKE_SYSTEM_NAME Generic)
set(CMAKE_SYSTEM_PROCESSOR c7000) # Or specific core if needed, e.g., c7100

# Specify the compiler
set(CMAKE_C_COMPILER "${TI_CGT_ROOT}/bin/cl7x")
set(CMAKE_ASM_COMPILER "${TI_CGT_ROOT}/bin/cl7x") # Use same compiler for assembly
set(CMAKE_CXX_COMPILER "${TI_CGT_ROOT}/bin/cl7x") # C++ needed for TVM runtime

# Specify the archiver
set(CMAKE_AR "${TI_CGT_ROOT}/bin/ar7x" CACHE FILEPATH "Archiver")

# CPU variant: C7504 on am62dx (AM62D), C7524 on j722s (default).  TVM_SOC is
# set by cmake/boards.cmake, which the firmware CMakeLists includes before
# project().
if(TVM_SOC STREQUAL "am62dx")
    set(_C7X_CPU_FLAG "-mv7504")
else()
    set(_C7X_CPU_FLAG "-mv7524")
endif()

# Set default compile flags (adjust as needed for C7x and your project)
# Consult TI documentation for recommended flags
set(CMAKE_C_FLAGS_INIT "${_C7X_CPU_FLAG} --abi=eabi --include_path=\"${TI_CGT_ROOT}/include\" -O2 -g" CACHE STRING "" FORCE)
set(CMAKE_CXX_FLAGS_INIT "${_C7X_CPU_FLAG} --abi=eabi --include_path=\"${TI_CGT_ROOT}/include\" -O2 -g --rtti --c++14" CACHE STRING "" FORCE)
set(CMAKE_ASM_FLAGS_INIT "${_C7X_CPU_FLAG} --abi=eabi --include_path=\"${TI_CGT_ROOT}/include\"" CACHE STRING "" FORCE)

# Set the define flag for TI compiler (uses -D like GCC)
set(CMAKE_C_COMPILE_OPTIONS_PIC "")
set(CMAKE_C_COMPILE_OPTIONS_PIE "")

# Force override of TI link command after CMake loads TI module
# The TI CMake module sets this in a macro, so we need to override it after project() is called
# This will be set again in the main CMakeLists.txt after project()

# Set the linker to use the same compiler but without --run_linker
set(CMAKE_LINKER "${TI_CGT_ROOT}/bin/cl7x" CACHE FILEPATH "Linker")

# Disable the --run_linker flag that CMake might add automatically
set(CMAKE_C_COMPILER_FORCED TRUE)
set(CMAKE_CXX_COMPILER_FORCED TRUE)

# Where to look for find_program(), find_library(), find_path(), find_file()
set(CMAKE_FIND_ROOT_PATH ${TI_CGT_ROOT})
set(CMAKE_FIND_ROOT_PATH_MODE_PROGRAM NEVER)
set(CMAKE_FIND_ROOT_PATH_MODE_LIBRARY ONLY)
set(CMAKE_FIND_ROOT_PATH_MODE_INCLUDE ONLY)
set(CMAKE_FIND_ROOT_PATH_MODE_PACKAGE ONLY)