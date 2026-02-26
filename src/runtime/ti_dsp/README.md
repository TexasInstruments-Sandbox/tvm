# TVM DSP Runtime

Minimal runtime for executing TVM-compiled models on TI DSP processors.

## Overview

The TVM DSP Runtime provides a lightweight C++14 API for running TVM-generated
models on resource-constrained embedded DSP environments. Key features:

- **C++14 Model API**: Clean, RAII-based interface (`model.h`)
- **Zero-copy design**: No copies of input/output data
- **Static memory pools**: No dynamic allocation in hot path
- **Cross-platform**: PC host emulation, TI C66x, and TI C7x hardware

## Supported Platforms

| Platform | Device | Memory | Toolchain |
|----------|--------|--------|-----------|
| Host | PC | malloc | GCC/Clang |
| C66x | AWRL6844 | L2 (64KB) + L3 (1MB) | ti-cgt-c6000 v8.5+ |
| C7x | J722S | L2 (128KB) + DDR (58MB) | ti-cgt-c7000 v5.0+ |

## Quick Start

```cpp
#include "model.h"

int main() {
  using namespace tvm::dsp;

  // Load model (handles platform init + constants parsing)
  Model model;
  if (model.Load() != ModelError::kSuccess) {
    return 1;
  }

  // Create input tensor (caller provides buffer)
  static float input_buffer[1 * 2 * 16];
  static int64_t input_shape[] = {1, 2, 16};
  auto input = NDArray::Float32(input_buffer, input_shape, 3);

  // Run inference
  NDArray* output;
  if (model.Infer(&input, &output) != ModelError::kSuccess) {
    return 1;
  }

  // Use output
  float* result = output->DataAs<float>();
  printf("Output[0]: %f\n", result[0]);
  printf("Cycles: %llu\n", model.LastInferenceCycles());

  return 0;
}  // Automatic cleanup via RAII
```

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
├── ffi/                      # FFI types (internal)
├── io/                       # Tensor file I/O (host testing only)
├── platform/                 # Platform abstraction (internal)
│   ├── host/                 # PC emulation backend
│   ├── c66x/                 # TI C66x backend (AWRL6844)
│   ├── c7x/                  # TI C7x backend (J722S)
│   │   ├── c7x_platform.c    # Platform init, memory pools
│   │   └── c7x_cxm.asm       # Security mode detection
│   └── common/               # Shared memory pool manager
├── registry/                 # Function registry (internal)
├── vm/                       # VM builtins (internal)
├── tests/                    # Unit tests
└── CMakeLists.txt            # Build configuration
```

## Building

### Prerequisites

**Host Emulation (PC)**:
- CMake 3.16+
- C99/C++14 compiler (GCC, Clang)

**C66x Cross-Compilation (AWRL6844)**:
- TI C6000 Compiler v8.5.0+ (`ti-cgt-c6000`)
- MMWAVE-L-SDK-6 v6.1.0.05
- Code Composer Studio 12.0+ (optional, for debugging)

**C7x Cross-Compilation (J722S)**:
- TI C7000 Compiler v5.0.1+ (`ti-cgt-c7000`)
- Code Composer Studio 12.0+ with XDS110 debug probe
- MCU+ SDK 11.01 (optional, for SDK integration)

### Host Emulation Build

```bash
cd src/runtime/ti_dsp
mkdir build && cd build
cmake ..
cmake --build .
```

Produces:
- `libtvm_dsp_runtime_host.a` - Static library
- `test_*` - Unit test executables

### C66x Cross-Compilation Build (AWRL6844)

```bash
cd src/runtime/ti_dsp
mkdir build-c66x && cd build-c66x
cmake -DCMAKE_TOOLCHAIN_FILE=../cmake/toolchain-awrl6844.cmake ..
cmake --build .
```

Produces:
- `libtvm_dsp_runtime_c66x.a` - Static library for C66x DSP

### C7x Cross-Compilation Build (J722S)

```bash
cd src/runtime/ti_dsp
mkdir build-c7x && cd build-c7x
cmake -DCMAKE_TOOLCHAIN_FILE=../cmake/toolchain-j722s-c7x.cmake ..
cmake --build .
```

Produces:
- `libtvm_dsp_runtime_c7x.a` - Static library for C7x DSP

### Running Tests

```bash
# Host tests
cd build
ctest --output-on-failure

# C66x tests (requires AWRL6844 hardware)
$TVM_HOME/src/runtime/ti_dsp/scripts/run_on_c66x.sh \
    build-c66x/test_vm_builtins_c66x.out

# C7x tests (requires J722S hardware)
$TVM_HOME/src/runtime/ti_dsp/scripts/run_on_c7x.sh \
    build-c7x/test_vm_builtins_c7x.out
```

## API Reference

### Public Header

Users only need to include `model.h`:

```cpp
#include "model.h"
```

This provides the complete `tvm::dsp` namespace with all public types.

### Model Class

```cpp
namespace tvm {
namespace dsp {

class Model {
 public:
  // Load model (initializes platform, parses constants)
  // weights_data: nullptr to use configured source (filesystem/embedded)
  ModelError Load(const void* weights_data = nullptr,
                  size_t weights_size = 0);

  // Run inference (single output)
  // input: caller-owned NDArray
  // output: pointer to internal result (valid until next Infer or ~Model)
  // For multi-output models, returns first output only
  ModelError Infer(NDArray* input, NDArray** output);

  // Run inference (multi-output)
  // outputs: array of NDArray pointers (max 8)
  // num_outputs: set to number of outputs on success
  ModelError InferMulti(NDArray* input, NDArray** outputs, int* num_outputs);

  // Diagnostics
  uint64_t LastInferenceCycles() const;
  MemoryStats GetMemoryStats(MemoryPool pool) const;
  int ConstantCount() const;
  int OutputCount() const;  // Number of outputs from last inference
  bool IsLoaded() const;

  // Lifecycle (move-only, no copy)
  Model();
  ~Model();  // Automatic cleanup
  Model(Model&&) noexcept;
  Model& operator=(Model&&) noexcept;
};

}}  // namespace tvm::dsp
```

### NDArray Structure

```cpp
namespace tvm {
namespace dsp {

struct NDArray {
  void* data;           // Pointer to tensor data (caller owns)
  int64_t* shape;       // Shape array (caller owns)
  int32_t ndim;         // Number of dimensions
  DLDataType dtype;     // Data type {code, bits, lanes}
  int32_t ref_counter;  // Reference count

  // Constructors
  NDArray();  // Zero initialization
  NDArray(void* data, int64_t* shape, int32_t ndim, DLDataType dtype);

  // Factory methods
  static NDArray Float32(float* data, int64_t* shape, int32_t ndim);
  static NDArray Float16(void* data, int64_t* shape, int32_t ndim);
  static NDArray Int32(int32_t* data, int64_t* shape, int32_t ndim);
  static NDArray Int8(int8_t* data, int64_t* shape, int32_t ndim);
  static NDArray UInt8(uint8_t* data, int64_t* shape, int32_t ndim);

  // Utilities
  bool IsValid() const;
  int64_t NumElements() const;
  size_t SizeBytes() const;
  template <typename T> T* DataAs() const;
};

}}
```

### Error Handling

```cpp
namespace tvm {
namespace dsp {

enum class ModelError {
  kSuccess = 0,           // Operation succeeded
  kPlatformInitFailed,    // Platform initialization failed
  kConstantsParseFailed,  // Constants parsing failed
  kNullInput,             // Null input pointer
  kNotLoaded,             // Model not loaded
  kInferenceFailed,       // Inference execution failed
  kInvalidOutputType      // Output type not supported
};

}}
```

### Memory Statistics

```cpp
namespace tvm {
namespace dsp {

enum class MemoryPool {
  kFast,  // L2 SRAM (64KB on C66x)
  kMain   // L3/DDR (larger, slower)
};

struct MemoryStats {
  size_t total_size;     // Total pool size in bytes
  size_t used_size;      // Currently used bytes
  size_t peak_used;      // Peak usage during lifetime
  uint32_t alloc_count;  // Number of allocations
  uint32_t free_count;   // Number of frees
};

}}
```

## Multi-Output Models

Models returning tuples of tensors (up to 8 outputs) are supported:

```cpp
using namespace tvm::dsp;

Model model;
model.Load();

// Get all outputs
NDArray* outputs[8];
int num_outputs;
if (model.InferMulti(&input, outputs, &num_outputs) == ModelError::kSuccess) {
  printf("Model returned %d outputs\n", num_outputs);
  for (int i = 0; i < num_outputs; i++) {
    printf("Output %d: %lld elements\n", i, outputs[i]->NumElements());
  }
}
```

**Limitations:**
- Maximum 8 outputs (enforced at compile time with `LOG(FATAL)`)
- `Infer()` returns only the first output for backward compatibility
- Use `InferMulti()` to retrieve all outputs

## Memory Ownership

| Data | Allocated By | Freed By | Lifetime |
|------|--------------|----------|----------|
| Input NDArray struct | Caller | Caller | Caller controls |
| Input data buffer | Caller | Caller | Caller controls |
| Output NDArray | Model | Model destructor | Until next Infer() or ~Model() |
| Constants | Model | Model destructor | Model lifetime |

**Key points:**
- Caller provides input buffer (stack, static, or hardware buffer)
- Output pointer is to internal register file - copy if persistence needed
- All cleanup is automatic via RAII

## Usage Examples

### Embedded System (No File I/O)

```cpp
#include "model.h"

// Linker-embedded weights
extern const char _binary_weights_bin_start[];
extern const unsigned int _binary_weights_bin_size;

// Hardware buffers
volatile float* adc_buffer = (float*)0x80000000;
volatile float* dac_buffer = (float*)0x80010000;

int main() {
  using namespace tvm::dsp;

  Model model;
  if (model.Load(_binary_weights_bin_start,
                 _binary_weights_bin_size) != ModelError::kSuccess) {
    return 1;
  }

  static int64_t shape[] = {1, 2, 16};
  auto input = NDArray::Float32(const_cast<float*>(adc_buffer), shape, 3);

  while (true) {
    NDArray* output;
    if (model.Infer(&input, &output) == ModelError::kSuccess) {
      float* out = output->DataAs<float>();
      for (int i = 0; i < 128; i++) {
        dac_buffer[i] = out[i];
      }
    }
  }
}
```

### Multiple Inferences with Timing

```cpp
using namespace tvm::dsp;

Model model;
model.Load();

uint64_t total_cycles = 0;
const int NUM_RUNS = 100;

for (int i = 0; i < NUM_RUNS; i++) {
  NDArray* output;
  model.Infer(&input, &output);
  total_cycles += model.LastInferenceCycles();
}

printf("Average: %llu cycles\n", total_cycles / NUM_RUNS);
```

### Memory Statistics

```cpp
using namespace tvm::dsp;

Model model;
model.Load();

MemoryStats l2 = model.GetMemoryStats(MemoryPool::kFast);
MemoryStats l3 = model.GetMemoryStats(MemoryPool::kMain);

printf("L2: %zu/%zu bytes (peak: %zu)\n",
       l2.used_size, l2.total_size, l2.peak_used);
printf("L3: %zu/%zu bytes (peak: %zu)\n",
       l3.used_size, l3.total_size, l3.peak_used);
```

## Memory Configuration

Configure pool sizes in platform headers:

**Host** (`platform/host/host_platform.h`):
```c
#define TVM_DSP_HOST_FAST_POOL_SIZE  (4 * 1024 * 1024)   // 4MB
#define TVM_DSP_HOST_MAIN_POOL_SIZE  (64 * 1024 * 1024)  // 64MB
```

**C66x AWRL6844** (`platform/c66x/awrl6844/awrl6844_config.h`):
```c
#define TVM_DSP_L2_HEAP_SIZE   (64 * 1024)   // 64KB L2 SRAM
#define TVM_DSP_L3_HEAP_SIZE   (1024 * 1024) // 1MB L3 SRAM
```

**C7x J722S** (`platform/c7x/c7x_platform.h`):
```c
#define TVM_DSP_L2_SIZE   (128 * 1024)       // 128KB L2 SRAM
#define TVM_DSP_DDR_SIZE  (58 * 1024 * 1024) // 58MB DDR
```

## Memory Architecture

The DSP runtime uses **static pool allocation** - no `malloc()` calls at runtime.
All memory is pre-allocated at link time for deterministic behavior.

### Memory Pools

**C66x (AWRL6844):**
| Pool | Size | Use Case | Access Speed |
|------|------|----------|--------------|
| L2 (Fast) | 64KB | Storage ≤32KB, hot data | Fastest (L2 SRAM) |
| L3 (Main) | 1MB | Storage >32KB, constants | Slower (L3 SRAM) |

**C7x (J722S):**
| Pool | Size | Use Case | Access Speed |
|------|------|----------|--------------|
| L2 (Fast) | 128KB | Storage ≤64KB, hot data | Fastest (L2 SRAM) |
| DDR (Main) | 58MB | Storage >64KB, constants | Slower (DDR) |

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
    .tvm_l2_heap > L2SRAM_C66x  /* 64KB in L2 SRAM */
    .tvm_l3_heap > L3_MEM       /* 1MB in L3 memory */
}
```

**C7x hardware**: Pools are placed via linker script sections:

```
SECTIONS {
    .tvm_l2_heap > L2_HEAP      /* 128KB in L2 SRAM */
    .tvm_ddr_heap > DDR_C7X_MAIN /* 58MB in DDR */
}
```

The linker allocates these sections at build time. At runtime,
`tvm_dsp_memory_pool_init()` initializes the pool structures using
linker-provided symbols.

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

### Required Memory Mappings

The application's MMU configuration must include mappings for:

- **L2 SRAM** (0x7E000000-0x7E1FFFFF): Cached, Non-Shareable
  - Used for TVM's fast memory pool
- **DDR** (0x80000000-0xFFFFFFFF): Cached, Outer Shareable
  - Used for TVM's main memory pool, code, and weights

See `tvm-relax-tests/dsp-cpp/j722s/mmu.c` for a reference implementation with
detailed documentation of MMU registers and page table structure.

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
`tvm-relax-tests/dsp-cpp/dsp_utils.py`.

## Differences from Full TVM Runtime

| Feature | Full TVM | DSP Runtime |
|---------|----------|-------------|
| Memory management | Dynamic (malloc/free) | Static pools |
| Error handling | Exceptions | ModelError enum |
| C++ standard | C++17 | C++14 |
| RTTI | Yes | No |
| VirtualMachine | Full VM class | Direct TIR calls |
| Binary size | ~10MB+ | ~100KB |

## Documentation

- [MODEL_API.md](MODEL_API.md) - Full API documentation and design rationale

## License

Apache License 2.0 (same as TVM)
