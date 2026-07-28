# Board/DDR configuration - single source of truth
#
# Resolves the two top-level build parameters, TVM_BOARD and TVM_DDR, into
# the SDK roots and shared-DMA physical base that vary between boards.
# include()'d from the three CMake projects that need any of these facts:
#   - src/runtime/ti_dsp/CMakeLists.txt          (runtime library)
#   - firmware/c7x/dsp/CMakeLists.txt            (DSP firmware)
#   - firmware/c7x/arm/CMakeLists.txt            (ARM client, cosmetic)
#
# SDK roots have no hardcoded defaults: a build that consumes the SDK must
# supply PSDK_INSTALL_PATH (or MCU_PLUS_SDK_PATH) and sets TVM_REQUIRE_SDK
# before include() so a missing/wrong root is a hard error, not a silent
# guess.  See docs/dsp/beagley_ai_enablement.md for the design.

set(TVM_BOARD "j722s-evm" CACHE STRING "Target board: j722s-evm, beagley-ai")
set_property(CACHE TVM_BOARD PROPERTY STRINGS j722s-evm beagley-ai)
if(NOT TVM_BOARD MATCHES "^(j722s-evm|beagley-ai)$")
    message(FATAL_ERROR "Invalid TVM_BOARD: ${TVM_BOARD}. Use: j722s-evm, beagley-ai")
endif()

# Per-board default DDR size (any explicitly-set TVM_DDR wins).
if(TVM_BOARD STREQUAL "beagley-ai")
    set(_TVM_DDR_DEFAULT "4gb")
else()
    set(_TVM_DDR_DEFAULT "8gb")
endif()
set(TVM_DDR "${_TVM_DDR_DEFAULT}" CACHE STRING "Shared-DMA DDR size: 4gb, 8gb")
set_property(CACHE TVM_DDR PROPERTY STRINGS 4gb 8gb)
if(NOT TVM_DDR MATCHES "^(4gb|8gb)$")
    message(FATAL_ERROR "Invalid TVM_DDR: ${TVM_DDR}. Use: 4gb, 8gb")
endif()

if(TVM_BOARD STREQUAL "beagley-ai" AND TVM_DDR STREQUAL "8gb")
    message(WARNING "TVM_BOARD=beagley-ai with TVM_DDR=8gb: BeagleY-AI has "
                     "only 4gb of DDR; this combination builds but the "
                     "carveout will point past physical memory at runtime.")
endif()

# Shared DMA carveout physical base.  This is the *only* memory-map
# difference between 4gb and 8gb boards (confirmed against the upstream
# armbian j722s-4gb-edgeai-memory-map.patch).
if(TVM_DDR STREQUAL "8gb")
    set(C7X_SHARED_PHYS_BASE "0x900000000")
else()
    set(C7X_SHARED_PHYS_BASE "0x8a0000000")
endif()

# SDK roots.  PSDK_INSTALL_PATH is the single required root for any build
# that consumes the TI SDK (the c7x runtime lib and the DSP firmware);
# MCU_PLUS_SDK_PATH and MMALIB_PATH derive from it.  There are deliberately
# NO hardcoded install-root defaults: a silently-wrong SDK path only
# surfaces as corruption at runtime, so an SDK build with nothing set is a
# hard error (see the TVM_REQUIRE_SDK block below) rather than a guess.
#
# Precedence per variable: an explicitly-set env var / -D wins; otherwise it
# is derived from PSDK_INSTALL_PATH.  The per-board SDK version subdirectory
# names are the only SDK strings this module hardcodes -- they are board
# facts (the 11_00 vs 11_02 SDKs), not machine paths.
if(TVM_BOARD STREQUAL "beagley-ai")
    set(_TVM_MCU_PLUS_SDK_SUBDIR "mcu_plus_sdk_j722s_11_02_01_05")
    set(_TVM_MMALIB_SUBDIR       "mmalib_11_02_00_11")
else()
    set(_TVM_MCU_PLUS_SDK_SUBDIR "mcu_plus_sdk_j722s_11_00_00_12")
    set(_TVM_MMALIB_SUBDIR       "mmalib_11_02_00_06")
endif()

# PSDK_INSTALL_PATH: env / -D only, no default.
if(DEFINED ENV{PSDK_INSTALL_PATH} AND NOT "$ENV{PSDK_INSTALL_PATH}" STREQUAL "")
    set(PSDK_INSTALL_PATH "$ENV{PSDK_INSTALL_PATH}")
endif()

# MCU_PLUS_SDK_PATH: env / -D wins (the workflow documented in CLAUDE.md
# exports this directly); else derive from PSDK_INSTALL_PATH.
if(DEFINED ENV{MCU_PLUS_SDK_PATH} AND NOT "$ENV{MCU_PLUS_SDK_PATH}" STREQUAL "")
    set(MCU_PLUS_SDK_PATH "$ENV{MCU_PLUS_SDK_PATH}")
elseif(NOT MCU_PLUS_SDK_PATH AND PSDK_INSTALL_PATH)
    set(MCU_PLUS_SDK_PATH "${PSDK_INSTALL_PATH}/${_TVM_MCU_PLUS_SDK_SUBDIR}")
endif()

# MMALIB_PATH: env / -D wins; else derive from the *resolved* PSDK root, so
# it can never drift from the chosen SDK the way a separate default could.
if(DEFINED ENV{MMALIB_PATH} AND NOT "$ENV{MMALIB_PATH}" STREQUAL "")
    set(MMALIB_PATH "$ENV{MMALIB_PATH}")
elseif(NOT MMALIB_PATH AND PSDK_INSTALL_PATH)
    set(MMALIB_PATH "${PSDK_INSTALL_PATH}/${_TVM_MMALIB_SUBDIR}")
endif()

# Strict validation for builds that actually consume the SDK.  The including
# project sets TVM_REQUIRE_SDK; host/c66x emulation and the ARM client do
# not (they include this module only for C7X_SHARED_PHYS_BASE /
# TVM_BUILD_SUFFIX), so they never trip these errors.
if(TVM_REQUIRE_SDK)
    if(NOT MCU_PLUS_SDK_PATH)
        message(FATAL_ERROR
            "No TI SDK configured for TVM_BOARD=${TVM_BOARD}. Set "
            "PSDK_INSTALL_PATH (recommended -- MCU_PLUS_SDK_PATH and "
            "MMALIB_PATH derive from it) or MCU_PLUS_SDK_PATH directly, via "
            "an environment variable or -D. No default is assumed, because a "
            "wrong SDK path corrupts the build silently at runtime.")
    endif()
    if(NOT EXISTS "${MCU_PLUS_SDK_PATH}")
        message(FATAL_ERROR
            "MCU_PLUS_SDK_PATH does not exist: ${MCU_PLUS_SDK_PATH} "
            "(PSDK_INSTALL_PATH=${PSDK_INSTALL_PATH}). Check the path, or set "
            "MCU_PLUS_SDK_PATH / PSDK_INSTALL_PATH explicitly.")
    endif()
endif()

# Build-dir naming suffix, mirrored by the *.sh wrappers (which must
# create/locate the same directories before cmake/boards.cmake ever runs).
# Empty for the default board+ddr so existing build-c7x/build-c7x-host
# paths are unchanged when no flags are passed.
set(TVM_BUILD_SUFFIX "")
if(NOT TVM_BOARD STREQUAL "j722s-evm" OR NOT TVM_DDR STREQUAL "8gb")
    set(TVM_BUILD_SUFFIX "-${TVM_BOARD}-${TVM_DDR}")
endif()

message(STATUS "TVM board config: TVM_BOARD=${TVM_BOARD} TVM_DDR=${TVM_DDR} "
                "C7X_SHARED_PHYS_BASE=${C7X_SHARED_PHYS_BASE}")
message(STATUS "  MCU_PLUS_SDK_PATH=${MCU_PLUS_SDK_PATH}")
message(STATUS "  PSDK_INSTALL_PATH=${PSDK_INSTALL_PATH}")
message(STATUS "  MMALIB_PATH=${MMALIB_PATH}")
