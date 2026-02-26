# DSP Weights Parser Design

## Overview

This document describes the design for a minimal weights.bin parser for the TVM DSP
runtime that works within the constraints of embedded devices (limited memory, no
filesystem in many cases, no dynamic allocation).

## Requirements

### Functional Requirements
1. Parse TVM's weights.bin serialization format
2. Support NDArray, Shape, Int, Float, String, and DLDataType constants
3. Create TVMFFIAny objects that the generated code can use
4. Support zero-copy access for embedded weight data

### Non-Functional Requirements
1. **Zero-copy**: Weight tensor data should point directly to embedded memory
2. **No malloc**: Use static pools for runtime structures
3. **No file I/O**: Weights embedded in binary via linker
4. **Minimal code size**: No C++ stdlib dependencies (no std::vector, std::string)
5. **Portable**: C99 compatible for maximum toolchain support

## TVM Binary Format

The weights.bin file uses this structure:

```
[uint64_t] num_constants
[Constant_0]
[Constant_1]
...
[Constant_n-1]
```

Each constant has a type discriminator followed by type-specific data:

### NDArray Constant (type = 15)
```
[int32_t]  type_index (15 = kTVMFFITensor)
[uint64_t] magic (0xDD5E40F096B4A13F)
[uint64_t] reserved (0)
[int32_t]  device_type (always kDLCPU = 1)
[int32_t]  device_id (always 0)
[int32_t]  ndim (number of dimensions)
[DLDataType] dtype
  [uint8_t]  code (0=int, 1=uint, 2=float)
  [uint8_t]  bits (8, 16, 32, 64)
  [uint16_t] lanes (usually 1)
[int64_t]  shape[ndim]
[int64_t]  data_byte_size
[uint8_t]  data[data_byte_size]  <- Points to tensor data
```

### Shape Constant (type = 17)
```
[int32_t]  type_index (17 = kTVMFFIShape)
[uint64_t] num_dims
[int64_t]  dims[num_dims]
```

### String Constant (type = 16)
```
[int32_t]  type_index (16 = kTVMFFIStr)
[uint64_t] length
[char]     chars[length]
```

### Int Constant (type = 18)
```
[int32_t]  type_index (18 = kTVMFFIInt)
[int64_t]  value
```

### Float Constant (type = 19)
```
[int32_t]  type_index (19 = kTVMFFIFloat)
[double]   value
```

### DLDataType Constant (type = 20)
```
[int32_t]  type_index (20 = kTVMFFIDataType)
[DLDataType] dtype (4 bytes)
```

## Design

### Architecture

```
+------------------+      +------------------+      +------------------+
| weights.bin data |----->| DSP Parser       |----->| TVMFFIAny[]      |
| (embedded)       |      | (zero-copy)      |      | constants array  |
+------------------+      +------------------+      +------------------+
                                  |
                                  v
                          +------------------+
                          | Static Pools     |
                          | - NDArray pool   |
                          | - Shape pool     |
                          | - String pool    |
                          +------------------+
```

### Key Components

1. **Stream Reader**: Sequential reader over embedded binary data
2. **Constant Parser**: Parses each constant based on type discriminator
3. **NDArray Pool**: Static pool for DLTensor/NDArray metadata
4. **Shape Pool**: Static pool for shape arrays
5. **Constants Array**: Output array of TVMFFIAny values

### Static Memory Pools

Since DSP has no heap, we use static pools:

```c
/* Maximum constants supported */
#define TVM_DSP_MAX_CONSTANTS  256

/* Maximum total shape elements across all tensors */
#define TVM_DSP_MAX_SHAPE_ELEMENTS  1024

/* Maximum string constant bytes */
#define TVM_DSP_MAX_STRING_BYTES  4096

/* Static pools */
static TVMDSPNDArray g_ndarray_pool[TVM_DSP_MAX_CONSTANTS];
static int64_t g_shape_pool[TVM_DSP_MAX_SHAPE_ELEMENTS];
static char g_string_pool[TVM_DSP_MAX_STRING_BYTES];
static TVMFFIAny g_constants[TVM_DSP_MAX_CONSTANTS];
```

### Zero-Copy NDArray

The NDArray's data pointer points directly into the embedded weights:

```c
typedef struct {
  TVMFFIObject base;    /* Object header for ref counting */
  DLTensor dl_tensor;   /* DLPack tensor descriptor */
  /* Note: dl_tensor.data points into embedded weights - no copy! */
} TVMDSPNDArray;
```

### API

```c
/**
 * @brief Initialize the constants system
 * Called once at startup before parsing
 */
void TVMDSPConstantsInit(void);

/**
 * @brief Parse weights from embedded binary data
 * @param data Pointer to weights.bin data (embedded in binary)
 * @param size Size of weights.bin in bytes
 * @return Number of constants parsed, or -1 on error
 */
int TVMDSPConstantsParse(const void* data, size_t size);

/**
 * @brief Get parsed constants array
 * @param count Output: number of constants
 * @return Pointer to constants array, or NULL if not initialized
 */
TVMFFIAny* TVMDSPConstantsGet(int* count);

/**
 * @brief Get a single constant by index
 * @param index Constant index (0 to num_constants-1)
 * @return Pointer to constant, or NULL if out of bounds
 */
TVMFFIAny* TVMDSPConstantGetByIndex(int index);
```

## Implementation Plan

### Phase 1: Stream Reader

Create a simple sequential reader:

```c
typedef struct {
  const uint8_t* data;  /* Base pointer */
  size_t size;          /* Total size */
  size_t pos;           /* Current position */
} TVMDSPStream;

void stream_init(TVMDSPStream* s, const void* data, size_t size);
int stream_read(TVMDSPStream* s, void* buf, size_t size);
int stream_skip(TVMDSPStream* s, size_t size);
const void* stream_peek(TVMDSPStream* s, size_t size);  /* Zero-copy read */
```

### Phase 2: NDArray Parser

Parse NDArray metadata, point data to embedded memory:

```c
int parse_ndarray(TVMDSPStream* s, TVMDSPNDArray* arr) {
  uint64_t magic, reserved;
  int32_t device_type, device_id, ndim;

  /* Read header */
  stream_read(s, &magic, 8);
  if (magic != TVM_NDARRAY_MAGIC) return -1;

  stream_read(s, &reserved, 8);
  stream_read(s, &device_type, 4);
  stream_read(s, &device_id, 4);
  stream_read(s, &ndim, 4);
  stream_read(s, &arr->dl_tensor.dtype, sizeof(DLDataType));

  /* Allocate shape from pool */
  arr->dl_tensor.shape = alloc_shape(ndim);
  stream_read(s, arr->dl_tensor.shape, ndim * sizeof(int64_t));

  /* Read data size */
  int64_t data_size;
  stream_read(s, &data_size, 8);

  /* ZERO-COPY: Point directly to embedded data */
  arr->dl_tensor.data = (void*)stream_peek(s, data_size);
  stream_skip(s, data_size);

  /* Set remaining fields */
  arr->dl_tensor.ndim = ndim;
  arr->dl_tensor.device.device_type = kDLCPU;  /* DSP sees it as "CPU" */
  arr->dl_tensor.device.device_id = 0;
  arr->dl_tensor.strides = NULL;
  arr->dl_tensor.byte_offset = 0;

  return 0;
}
```

### Phase 3: Constants Parser

Parse all constants:

```c
int TVMDSPConstantsParse(const void* data, size_t size) {
  TVMDSPStream stream;
  stream_init(&stream, data, size);

  /* Read constant count */
  uint64_t num_constants;
  stream_read(&stream, &num_constants, 8);

  if (num_constants > TVM_DSP_MAX_CONSTANTS) {
    tvm_dsp_log("ERROR: Too many constants (%lu > %d)\n",
                num_constants, TVM_DSP_MAX_CONSTANTS);
    return -1;
  }

  for (size_t i = 0; i < num_constants; i++) {
    int32_t type_index;
    stream_read(&stream, &type_index, 4);

    switch (type_index) {
      case kTVMFFITensor:
        parse_ndarray_constant(&stream, &g_constants[i]);
        break;
      case kTVMFFIShape:
        parse_shape_constant(&stream, &g_constants[i]);
        break;
      case kTVMFFIInt:
        parse_int_constant(&stream, &g_constants[i]);
        break;
      case kTVMFFIFloat:
        parse_float_constant(&stream, &g_constants[i]);
        break;
      /* ... other types ... */
      default:
        tvm_dsp_log("ERROR: Unknown constant type %d\n", type_index);
        return -1;
    }
  }

  g_num_constants = (int)num_constants;
  return g_num_constants;
}
```

### Phase 4: Integration with Generated Code

The generated code calls `TVMGetConstants()` which maps to our DSP implementation:

```c
/* In devc_dsp.cpp or constants_loader.c */

/* Linker symbols for embedded weights */
extern const char _binary_weights_bin_start[];
extern const char _binary_weights_bin_end[];

/* Cached constants (converted from TVMFFIAny* to std::vector wrapper) */
static bool g_constants_loaded = false;

std::vector<tvm::ffi::Any> TVMGetConstants() {
  if (!g_constants_loaded) {
    /* Initialize DSP constants system */
    TVMDSPConstantsInit();

    /* Parse embedded weights - zero copy! */
    size_t size = _binary_weights_bin_end - _binary_weights_bin_start;
    int count = TVMDSPConstantsParse(_binary_weights_bin_start, size);

    if (count < 0) {
      tvm_dsp_log("ERROR: Failed to parse weights\n");
      /* Return empty - will fail later with more info */
    }

    g_constants_loaded = true;
  }

  /* Return wrapper around static constants array */
  int count;
  TVMFFIAny* arr = TVMDSPConstantsGet(&count);

  /* Create C++ wrapper (no copy, just pointer wrapper) */
  return std::vector<tvm::ffi::Any>(
    reinterpret_cast<tvm::ffi::Any*>(arr),
    reinterpret_cast<tvm::ffi::Any*>(arr + count)
  );
}
```

## Memory Layout

For a model with 10 weight tensors totaling 1MB:

```
+------------------------------------------+
| Embedded .rodata section (from linker)   |
| - weights.bin data                       |
| - Tensor data bytes pointed to directly  |
+------------------------------------------+
| Static BSS                               |
| - g_ndarray_pool[256] (~8KB)             |
| - g_shape_pool[1024] (~8KB)              |
| - g_constants[256] (~4KB)                |
| - g_string_pool[4KB]                     |
+------------------------------------------+
| Total static overhead: ~24KB             |
| Tensor data: 0 bytes (zero-copy!)        |
+------------------------------------------+
```

## Error Handling

Since DSP has no exceptions:

```c
/* Error codes */
#define TVM_DSP_SUCCESS           0
#define TVM_DSP_ERR_INVALID_MAGIC -1
#define TVM_DSP_ERR_TOO_MANY_CONST -2
#define TVM_DSP_ERR_SHAPE_POOL_FULL -3
#define TVM_DSP_ERR_UNKNOWN_TYPE   -4
#define TVM_DSP_ERR_BUFFER_OVERFLOW -5

/* All functions return error code or count */
/* Use tvm_dsp_log() for error messages */
```

## Testing Strategy

1. **Unit Tests (Host Emulation)**:
   - Parse known weights.bin files
   - Verify NDArray shapes, dtypes, data pointers
   - Test boundary conditions (max constants, shape overflow)

2. **Integration Tests**:
   - Load CLISTA model weights
   - Run inference with parsed constants
   - Compare output with full TVM runtime

3. **Memory Tests**:
   - Verify no malloc calls
   - Check static pool usage
   - Confirm zero-copy (data pointers in .rodata)

## File Organization

```
src/runtime/ti_dsp/constants/
├── DESIGN.md           # This document
├── constants.h         # Public API header
├── constants.c         # Main parser implementation
├── stream.h            # Stream reader header
├── stream.c            # Stream reader implementation
└── pools.c             # Static pool management
```

## Compatibility Notes

### Endianness
- TVM serializes in host endian (little-endian on x86/ARM)
- C66x DSP is little-endian - no byte swapping needed
- If targeting big-endian DSP, add swap on read

### Alignment
- NDArray data may not be aligned for SIMD
- For performance, consider adding alignment padding
- Or use memcpy to aligned buffer for SIMD operations

### Type Mapping
- TVM's NDArray -> TVMDSPNDArray (same DLTensor layout)
- TVM's Shape -> int64_t* with count
- TVM's String -> const char* (null-terminated)
- TVM's int/float -> stored directly in TVMFFIAny

## Weight Embedding Options

The constants loader supports multiple methods for providing weight data:

### Option 1: Linker Embedding (Recommended for Production)

Embed weights.bin directly into the executable using objcopy or linker scripts.
Zero-copy access to weight data from .rodata section.

**Linux (ELF):**
```bash
objcopy -I binary -O elf64-x86-64 \
  --rename-section .data=.rodata,alloc,load,readonly,data,contents \
  weights.bin weights.o
```

**macOS (Mach-O):**
Use assembly with .incbin directive or ld -sectcreate.

**C66x DSP:**
Use hex6x or assembly .byte directives to embed data.

**CMake Integration:**
```cmake
include(WeightEmbedding)
tvm_dsp_embed_weights(my_target ${CMAKE_CURRENT_SOURCE_DIR}/weights.bin)
```

### Option 2: Filesystem Loading (Development Only)

Load weights from file at runtime. Not recommended for embedded deployment.

```cmake
target_compile_definitions(my_target PRIVATE
  TVM_DSP_WEIGHTS_FILESYSTEM
  TVM_DSP_WEIGHTS_PATH="path/to/weights.bin"
)
```

### Option 3: C Array (Small Models Only)

Convert weights.bin to a C array header. Only practical for small models (<10MB).

```bash
xxd -i weights.bin > weights_data.h
```

```cmake
target_compile_definitions(my_target PRIVATE TVM_DSP_WEIGHTS_C_ARRAY)
```

## File Organization

```
src/runtime/ti_dsp/constants/
├── DESIGN.md           # This document
├── constants.h         # Main parser API
├── constants.c         # Parser implementation
├── constants_c_api.h   # Pure C API for weight data and constant access
├── constants_c_api.c   # C API implementation
├── constants_loader.h  # High-level loader API header
├── constants_loader.cpp # Loader with TVMDSPParseConstants() and TVMGetConstants()
├── stream.h            # Stream reader header
└── stream.c            # Stream reader implementation

src/runtime/ti_dsp/cmake/
└── WeightEmbedding.cmake  # CMake module for weight embedding
```

The constants_loader module provides a high-level API that:
1. Handles weight data source initialization (embedded, filesystem, or external)
2. Provides `TVMDSPParseConstants()` as the single entry point
3. Provides `TVMGetConstants()` C++ API for TVM-generated code compatibility

## Usage Example

### Application Code (C)

```c
#include "constants/constants_c_api.h"

int main() {
    /* Set weights data (for linker embedded, this happens automatically) */
    TVMDSPSetWeightsData(weights_start, weights_size);

    /* Load and parse constants */
    int count = TVMDSPLoadConstants();
    if (count < 0) {
        printf("Failed to load constants: %s\n", TVMDSPConstantsErrorString(count));
        return -1;
    }

    /* Access individual constants */
    TVMFFIAny* const0 = TVMDSPGetConstant(0);
    if (const0 && const0->type_index == kTVMFFITensor) {
        TVMDSPNDArray* arr = (TVMDSPNDArray*)const0->v_obj;
        DLTensor* tensor = &arr->dl_tensor;
        /* Use tensor... */
    }

    return 0;
}
```

### CMakeLists.txt Integration

```cmake
# Include DSP runtime CMake modules
list(APPEND CMAKE_MODULE_PATH "${TVM_DSP_RUNTIME_DIR}/cmake")
include(WeightEmbedding)

# Create executable
add_executable(my_dsp_app main.c lib0.c)

# Link DSP runtime
target_link_libraries(my_dsp_app PRIVATE tvm_dsp_runtime)

# Embed weights (creates weights.o and links it)
tvm_dsp_embed_weights(my_dsp_app ${CMAKE_CURRENT_SOURCE_DIR}/weights.bin)
```

## Testing

1. **Unit Tests (Host Emulation)**:
   - Parse known weights.bin files
   - Verify NDArray shapes, dtypes, data pointers
   - Test boundary conditions (max constants, shape overflow)

2. **Integration Tests**:
   - Build with TVM-generated code
   - Run inference with parsed constants
   - Compare output with full TVM runtime

## Status

- [x] Design document
- [x] Stream reader (stream.h/c)
- [x] Constants parser (constants.h/c)
- [x] C API (constants_c_api.h/c)
- [x] High-level loader API (constants_loader.h)
- [x] C++ loader implementation (constants_loader.cpp)
- [x] TVMDSPParseConstants() - unified entry point
- [x] TVMGetConstants() - C++ API for host emulation
- [x] CMake weight embedding module
- [x] Integration with DSP CMakeLists.txt
- [x] Integration with tvm_dsp_runtime.h unified header
- [x] Integration with CLISTA model (via dsp-cpp)
