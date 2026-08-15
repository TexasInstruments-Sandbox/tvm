# DSP Runtime Internals

Minimal runtime for executing TVM-compiled models on TI DSP processors.
Located at `src/runtime/ti_dsp/`. For the public C++ API this runtime
exposes to applications, see `src/runtime/ti_dsp/MODEL_API.md` in the
source tree.

## Overview

The TVM DSP Runtime provides a lightweight C++14 API for running TVM-generated
models on resource-constrained embedded DSP environments. Key features:

- **C++14 Model API**: Clean, RAII-based interface (`model.h`)
- **Zero-copy design**: No copies of input/output data
- **Static memory pools**: No dynamic allocation in hot path
- **Cross-platform**: PC host emulation, TI C66x, and TI C7x hardware

## Supported Platforms

| Build variant | Device | Output library | Toolchain |
|---------------|--------|----------------|-----------|
| `c66x_host` | PC (C66x host emulation) | `libtvm_dsp_runtime_host.a` | GCC/Clang |
| `c66x` | AWRL6844 C66x DSP | `libtvm_dsp_runtime_c66x.a` | ti-cgt-c6000 v8.5+ |
| `c7x_host` | PC (C7x host emulation) | `libtvm_dsp_runtime_c7x_host.a` | GCC + TI Host Emu |
| `c7x` | J722S/AM67A C7x DSP | `libtvm_dsp_runtime_c7x.a` | ti-cgt-c7000 v5.0+ |

## Directory Structure

```
ti_dsp/
├── include/                  # Public API
│   └── model.h               # C++14 Model API (only header users need)
├── cpp/                      # C++14 implementation and utilities
│   ├── model.cpp             # Model class implementation
│   ├── scope_guard.h         # RAII scope guard
│   ├── fixed_vector.h        # Fixed-capacity vector
│   ├── span.h                # Non-owning array view
│   ├── result.h              # Error handling without exceptions
│   ├── typed_handle.h        # Type-safe handle wrapper
│   └── ...                   # Other internal utilities
├── cmake/                    # CMake modules
│   ├── toolchain-awrl6844.cmake  # C66x cross-compilation toolchain
│   ├── toolchain-j722s-c7x.cmake # C7x cross-compilation toolchain
│   └── WeightEmbedding.cmake     # Binary weight embedding support
├── constants/                # weights.bin parser (internal)
├── container/                # NDArray, Shape, Array containers (internal)
├── dma/                      # DMA abstraction (tiling, host stub)
├── dynmod/                   # C7x DLOAD dynamic module infrastructure
├── ffi/                      # FFI types (internal)
├── firmware/                 # C7x remoteproc firmware (J722S/AM67A)
├── platform/                 # Platform abstraction (internal)
│   ├── host/                 # C66x host emulation backend (GCC)
│   ├── c66x/                 # TI C66x backend (AWRL6844)
│   ├── c7x/                  # TI C7x backend (J722S)
│   │   ├── c7x_platform.c    # Platform init, memory pools
│   │   └── c7x_cxm.asm       # Security mode detection
│   └── common/               # Shared memory pool manager
├── registry/                 # Function registry (internal)
├── scripts/                  # JTAG deployment scripts
│   ├── run_on_c66x.sh        # C66x hardware deployment (AWRL6844)
│   └── run_on_c75x.sh        # C7x hardware deployment (J722S)
├── tidl/                     # TIDL offload API (internal)
├── vm/                       # VM builtins (internal)
├── tests/                    # Unit tests
├── build_runtime.sh          # Build script for all runtime variants
└── CMakeLists.txt            # Build configuration
```

## Building

Use `build_runtime.sh` to build any runtime variant. The script
auto-detects TI compiler installations.

```bash
cd src/runtime/ti_dsp

bash build_runtime.sh c66x_host   # C66x host emulation (no TI compiler needed)
bash build_runtime.sh c66x        # C66x cross-compilation (needs TI_CGT_C6000_PATH)
bash build_runtime.sh c7x_host    # C7x host emulation (needs TI_CGT_C7000_PATH)
bash build_runtime.sh c7x         # C7x cross-compilation (needs TI_CGT_C7000_PATH)
bash build_runtime.sh all         # Build c66x + c7x + c7x_host
bash build_runtime.sh clean       # Remove all build directories
```

### Prerequisites

**`c66x_host`** (C66x host emulation on PC):
- CMake 3.16+
- C99/C++14 compiler (GCC, Clang)

**`c66x`** (C66x cross-compilation for AWRL6844):
- TI C6000 Compiler v8.5.0+ (`ti-cgt-c6000`) — set `TI_CGT_C6000_PATH`
- MMWAVE-L-SDK-6 v6.1.0.05 (auto-detected or set `MMWAVE_SDK_PATH`)

**`c7x_host`** (C7x host emulation on PC):
- TI C7000 Compiler v5.0.1+ (`ti-cgt-c7000`) — set `TI_CGT_C7000_PATH`
  (provides the Host Emulation library; system GCC is used to compile)

**`c7x`** (C7x cross-compilation for J722S/AM67A):
- TI C7000 Compiler v5.0.1+ (`ti-cgt-c7000`) — set `TI_CGT_C7000_PATH`
- MCU+ SDK for J722S (auto-detected or set `MCU_PLUS_SDK_PATH`)

### Build Outputs

| Variant | Build directory | Library |
|---------|-----------------|---------|
| `c66x_host` | `build-c66x-host/` | `libtvm_dsp_runtime_host.a` |
| `c66x` | `build-c66x/` | `libtvm_dsp_runtime_c66x.a` |
| `c7x_host` | `build-c7x-host/` | `libtvm_dsp_runtime_c7x_host.a` |
| `c7x` | `build-c7x/` | `libtvm_dsp_runtime_c7x.a` |

Each build also produces unit test executables in its build directory.

### Running Unit Tests

```bash
# C66x host emulation (runs on PC)
cd build-c66x-host
ctest --output-on-failure

# C7x host emulation (runs on PC)
cd build-c7x-host
ctest --output-on-failure

# C66x hardware (requires AWRL6844 + XDS110 probe)
$TVM_HOME/src/runtime/ti_dsp/scripts/run_on_c66x.sh \
    build-c66x/test_vm_builtins_c66x.out

# C7x hardware (requires J722S/AM67A board)
$TVM_HOME/src/runtime/ti_dsp/scripts/run_on_c75x.sh \
    build-c7x/test_vm_builtins_c7x.out
```

## Memory Configuration

Pool sizes are defined in the platform headers:

**C66x host emulation** (`platform/host/host_platform.h`):
```c
#define TVM_DSP_L2_SIZE  (4 * 1024 * 1024)   // 4MB emulated fast pool
#define TVM_DSP_L3_SIZE  (64 * 1024 * 1024)  // 64MB emulated main pool
```

**C66x AWRL6844** (`platform/c66x/c66x_platform.h`):
```c
#define TVM_DSP_L2_SIZE  (256 * 1024)  // 256KB L2 SRAM (default)
#define TVM_DSP_L3_SIZE  (512 * 1024)  // 512KB L3 SRAM (default)
```

**C7x J722S** (`platform/c7x/c7x_platform.h`):
```c
#define TVM_DSP_L2_SIZE_FALLBACK   (512 * 1024)       // 512KB L2 SRAM fallback
#define TVM_DSP_DDR_SIZE_FALLBACK  (64 * 1024 * 1024) // 64MB DDR fallback
```
C7x pool sizes are queried at runtime from the firmware via
`tvm_dsp_get_l2_base()` / `tvm_dsp_get_l2_size()`; the fallback values
are used only when firmware getters are unavailable.

## Memory Architecture

The DSP runtime uses **static pool allocation** - no `malloc()` calls at runtime.
All memory is pre-allocated at link time for deterministic behavior.

### Memory Pools

**C66x (AWRL6844):**
| Pool | Size | Use Case | Access Speed |
|------|------|----------|--------------|
| L2 (Fast) | 256KB | Storage ≤32KB, hot data | Fastest (L2 SRAM) |
| L3 (Main) | 512KB | Storage >32KB, constants | Slower (L3 SRAM) |

**C7x (J722S):**
| Pool | Size | Use Case | Access Speed |
|------|------|----------|--------------|
| L2 (Fast) | 512KB | Storage ≤32KB, hot data | Fastest (L2 SRAM) |
| DDR (Main) | 64MB | Storage >32KB, constants | Slower (DDR) |

The storage allocation threshold balances L2 utilization against
capacity. Tensors exceeding this threshold are placed in L3/DDR to avoid
exhausting the limited L2 space.

### Allocation Strategy

The allocator uses a **bump-pointer + free-list** approach:

1. **First allocation**: Bumps the pool pointer forward
2. **Free**: Adds block to a size-segregated free-list
3. **Subsequent allocations**: Checks free-list first for reuse
4. **Fallback**: Uses bump-pointer if no suitable free block

This design provides:
- O(1) allocation in common case
- Memory reuse without fragmentation
- No dynamic memory at runtime

### Backing Memory Setup

**Host emulation**: Pools are allocated via `malloc()` during platform init.

**C66x hardware**: Pools are placed via linker script sections:

```
SECTIONS {
    .tvm_l2_heap > L2SRAM_C66x  /* 256KB in L2 SRAM */
    .tvm_l3_heap > L3_MEM       /* 512KB in L3 memory */
}
```

**C7x hardware**: Pools are placed via linker script sections:

```
SECTIONS {
    .tvm_l2_heap > L2_HEAP       /* 512KB in L2 SRAM */
    .tvm_ddr_heap > DDR_C7X_MAIN /* 64MB in DDR */
}
```

The linker allocates these sections at build time. At runtime,
`tvm_dsp_memory_pool_init()` initializes the pool structures using
linker-provided symbols.

Note: these pool sizes describe the *standard runtime build* config
(`platform/c7x/c7x_platform.h` fallbacks). The actual deployed firmware's
unified DDR pool is larger (352 MiB) -- see
[Firmware Design Deep-Dive](../firmware/design-deep-dive.md) -- and the
standalone JTAG test harness uses yet another, independent memory map --
see [C7x Memory Map Reference](memory-map.md). These are three distinct
build targets; don't cross-check pool sizes between them.

## C7x MMU Configuration

For C7x platforms (J722S), the application is responsible for initializing the
MMU before calling any TVM runtime functions. The TVM DSP runtime assumes the
MMU is already configured when `tvm_dsp_platform_init()` is called.

### Why Application-Managed MMU?

1. **Linker-MMU Coupling**: The application manages the linker command file,
   which defines memory regions. MMU page tables must match the linker's
   memory layout exactly.

2. **Single Source of Truth**: Having MMU configuration in one place (the
   application) avoids conflicts between runtime and application page tables.

3. **Boot Sequence**: MMU init must happen early in boot (before DDR is
   accessible), which is naturally handled by application boot code.

See [C7x Memory Map Reference](memory-map.md) for the detailed
register-level configuration (ECR registers, MAIR table, page table
structure) used by the standalone JTAG test harness's reference
implementation (`tests/ti-dsp-runtime/dsp-cpp/j722s/mmu.c`).

## Tensor File I/O (Host Testing Only)

For host testing, the `io/` module provides binary tensor file I/O:

```c
#include "io/tensor_file.h"

// Read tensors from file
int num_tensors;
TVMDSPNDArray** tensors = TVMDSPReadTensorsFromFile("input.bin", &num_tensors);

// Write tensors to file
TVMDSPWriteTensorsToFile("output.bin", tensors, num_tensors);

// Free tensor array
TVMDSPFreeTensorArray(tensors, num_tensors);
```

Python utilities for creating/reading these files are in
`tests/ti-dsp-runtime/dsp-cpp/dsp_utils.py`.

## Differences from Full TVM Runtime

| Feature | Full TVM | DSP Runtime |
|---------|----------|-------------|
| Memory management | Dynamic (malloc/free) | Static pools |
| Error handling | Exceptions | ModelError enum |
| C++ standard | C++17 | C++14 |
| RTTI | Yes | No |
| VirtualMachine | Full VM class | Direct TIR calls |
| Binary size | ~10MB+ | ~100KB |
