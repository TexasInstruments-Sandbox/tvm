# TVM DSP C++14 Model API Design

**Document Version:** 1.0
**Date:** 2026-01-23
**Status:** Implementation In Progress

---

## Overview

This document describes the C++14 Model API for the TVM DSP Runtime. The API
provides a clean, RAII-based interface for running TVM-generated models on
TI DSP processors (C66x, C7x) and PC host emulation.

### Design Goals

1. **Minimal API surface** - Single `Model` class with few methods
2. **RAII-based** - Automatic cleanup, no manual free calls
3. **Zero-copy** - No copies of input or output data
4. **No file I/O** - Pointer-based API for embedded systems
5. **Type-safe** - Factory methods prevent common errors
6. **C++14 compatible** - Works with TI C6000 compiler

### Naming Conventions

Following Google C++ Style Guide:
- Classes/Structs: `PascalCase` (Model, NDArray, MemoryStats)
- Methods: `PascalCase` (Load, Infer, GetMemoryStats)
- Variables: `snake_case` (weights_data, const_count_)
- Constants: `kPascalCase` (kFast, kMain)
- Namespaces: `lowercase` (tvm::dsp)

---

## API Reference

### Namespace

```cpp
namespace tvm {
namespace dsp {
  // All API types and classes
}}
```

### NDArray Structure

```cpp
namespace tvm {
namespace dsp {

struct NDArray {
  void* data;           // Pointer to tensor data
  int64_t* shape;       // Shape array
  int32_t ndim;         // Number of dimensions
  DLDataType dtype;     // Data type {code, bits, lanes}
  int32_t ref_counter;  // Reference count (internal use)

  // Default constructor - safe zero initialization
  NDArray();

  // Main constructor - ref_counter automatically set to 1
  NDArray(void* data, int64_t* shape, int32_t ndim, DLDataType dtype);

  // Factory methods for common types
  static NDArray Float32(float* data, int64_t* shape, int32_t ndim);
  static NDArray Float16(void* data, int64_t* shape, int32_t ndim);
  static NDArray Int32(int32_t* data, int64_t* shape, int32_t ndim);
  static NDArray Int8(int8_t* data, int64_t* shape, int32_t ndim);
};

}}  // namespace tvm::dsp
```

### MemoryPool Enum

```cpp
namespace tvm {
namespace dsp {

enum class MemoryPool {
  kFast,  // L2 SRAM - fast, limited (64KB on C66x)
  kMain   // L3/DDR - slower, larger
};

}}
```

### MemoryStats Structure

```cpp
namespace tvm {
namespace dsp {

struct MemoryStats {
  size_t total_size;     // Total pool size in bytes
  size_t used_size;      // Currently used bytes
  size_t peak_used;      // Peak usage during lifetime
  uint32_t alloc_count;  // Number of allocations
  uint32_t free_count;   // Number of frees
};

}}
```

### ModelError Enum

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

### Model Class

```cpp
namespace tvm {
namespace dsp {

class Model {
 public:
  //----------------------------------------------------------
  // Loading
  //----------------------------------------------------------

  /*!
   * \brief Load a model and initialize the runtime
   *
   * Performs all initialization:
   * 1. Platform init (memory pools, hardware)
   * 2. Constants parsing from weights data
   *
   * \param weights_data Pointer to weights.bin data, or nullptr for embedded
   * \param weights_size Size of weights data in bytes (0 if embedded)
   * \return ModelError::kSuccess on success, error code on failure
   */
  ModelError Load(const void* weights_data = nullptr,
                  size_t weights_size = 0);

  //----------------------------------------------------------
  // Inference
  //----------------------------------------------------------

  /*!
   * \brief Run inference on input tensor
   *
   * \param input Pointer to input NDArray (caller retains ownership)
   * \param output Output pointer - set to output NDArray on success
   * \return ModelError::kSuccess on success, error code on failure
   *
   * Memory ownership:
   * - Input: Caller owns, Model borrows during Infer()
   * - Output: Model owns, valid until next Infer() or ~Model()
   */
  ModelError Infer(NDArray* input, NDArray** output);

  //----------------------------------------------------------
  // Diagnostics
  //----------------------------------------------------------

  /*!
   * \brief Get cycle count from last inference
   * \return CPU cycles, or 0 if Infer() never called
   */
  uint64_t LastInferenceCycles() const;

  /*!
   * \brief Get memory statistics for a pool
   * \param pool Which memory pool to query
   * \return MemoryStats structure
   */
  MemoryStats GetMemoryStats(MemoryPool pool) const;

  /*!
   * \brief Get number of constants loaded
   * \return Number of constants
   */
  int ConstantCount() const;

  /*!
   * \brief Check if model is loaded and ready
   * \return true if ready for inference
   */
  bool IsLoaded() const;

  //----------------------------------------------------------
  // Lifecycle
  //----------------------------------------------------------

  // Default constructor - creates unloaded model
  Model();

  // Destructor - handles all cleanup automatically
  ~Model();

  // Move-only (no copy)
  Model(Model&& other) noexcept;
  Model& operator=(Model&& other) noexcept;
  Model(const Model&) = delete;
  Model& operator=(const Model&) = delete;
};

}}  // namespace tvm::dsp
```

---

## Memory Ownership Model

| Data | Allocated By | Freed By | Lifetime |
|------|--------------|----------|----------|
| Input NDArray struct | Caller | Caller | Caller controls |
| Input data buffer | Caller | Caller | Caller controls |
| Output NDArray struct | Model | Model destructor | Until next Infer() or ~Model() |
| Output data buffer | Model | Model destructor | Until next Infer() or ~Model() |
| Constants | Model | Model destructor | Model lifetime |

**Key Points:**
- Caller provides input buffer (can be stack, static, or hardware buffer)
- Model provides output pointer to internal register file
- Output is invalidated by next Infer() call - copy if persistence needed
- All cleanup is automatic via RAII

---

## Usage Examples

### Basic Inference

```cpp
#include "model.h"

int main() {
  using namespace tvm::dsp;

  // Create and load model
  Model model;
  if (model.Load(weights_data, weights_size) != ModelError::kSuccess) {
    printf("Load failed\n");
    return 1;
  }

  // Prepare input (caller allocates)
  static float input_buffer[1 * 2 * 16];
  static int64_t input_shape[] = {1, 2, 16};

  // Fill from sensor/hardware
  fill_from_sensor(input_buffer);

  // Create NDArray (one-liner, no manual ref_counter)
  auto input = NDArray::Float32(input_buffer, input_shape, 3);

  // Run inference
  NDArray* output;
  if (model.Infer(&input, &output) != ModelError::kSuccess) {
    return 1;
  }

  // Use output (valid until next Infer or destructor)
  float* out_data = output->DataAs<float>();

  printf("Output[0]: %f\n", out_data[0]);
  printf("Cycles: %llu\n", model.LastInferenceCycles());

  return 0;
}  // Automatic cleanup
```

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

  // Load from embedded weights
  Model model;
  if (model.Load(_binary_weights_bin_start,
                 _binary_weights_bin_size) != ModelError::kSuccess) {
    return 1;
  }

  // Input from ADC buffer (no copy)
  static int64_t shape[] = {1, 2, 16};
  auto input = NDArray::Float32(
    const_cast<float*>(adc_buffer), shape, 3
  );

  // Continuous inference loop
  while (true) {
    NDArray* output;
    if (model.Infer(&input, &output) == ModelError::kSuccess) {
      // Copy output to DAC buffer
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
model.Load(weights, size);

uint64_t total_cycles = 0;
const int NUM_RUNS = 100;

for (int i = 0; i < NUM_RUNS; i++) {
  NDArray* output;
  model.Infer(&input, &output);
  total_cycles += model.LastInferenceCycles();
}

printf("Average: %llu cycles\n", total_cycles / NUM_RUNS);
```

---

## Error Handling

The API uses `ModelError` enum for error handling (no exceptions on DSP):

```cpp
using namespace tvm::dsp;

Model model;
ModelError err = model.Load(data, size);

// Check success
if (err == ModelError::kSuccess) {
  // Model loaded successfully
}

// Or use switch for handling specific errors
switch (err) {
  case ModelError::kSuccess:
    break;
  case ModelError::kPlatformInitFailed:
    printf("Platform init failed\n");
    break;
  case ModelError::kConstantsParseFailed:
    printf("Constants parse failed\n");
    break;
  default:
    printf("Unknown error\n");
    break;
}
```

### Error Codes

| Error | Meaning |
|-------|---------|
| kSuccess | Operation succeeded |
| kPlatformInitFailed | Platform initialization failed |
| kConstantsParseFailed | Constants parsing failed |
| kNullInput | Null input pointer |
| kNotLoaded | Model not loaded |
| kInferenceFailed | Inference execution failed |
| kInvalidOutputType | Output type not supported |

---

## Comparison with Previous API

### Before (C-style)

```cpp
// 15+ API calls, manual cleanup
tvm_dsp_platform_init();
TVMDSPLoadWeightsFromSource();
TVMDSPParseConstants();
TVMFFIAny* constants = TVMDSPConstantsGet(&count);
TVMDSPNDArray** inputs = TVMDSPReadTensorsFromFile(...);

TVMFFIAny input_any;
input_any.type_index = kTVMFFITensor;
input_any.zero_padding = 0;
input_any.v_obj = (TVMFFIObject*)inputs[0];
inputs[0]->ref_counter++;

TVMFFIAny output_any;
cg_main_dsp(&input_any, 1, constants, &output_any);

TVMDSPRegFileCleanup();
TVMDSPConstantsCleanup();
tvm_dsp_platform_shutdown();
```

### After (C++14)

```cpp
// 4 API calls, automatic cleanup
using namespace tvm::dsp;
Model model;
model.Load(weights, size);
auto input = NDArray::Float32(buffer, shape, 3);
NDArray* output;
model.Infer(&input, &output);
```

| Metric | Before | After |
|--------|--------|-------|
| API calls | 15+ | 4 |
| Lines of code | ~50 | ~6 |
| Manual cleanup | 4 calls | 0 |
| Error-prone setup | TVMFFIAny, ref_counter | Factory methods |

---

## Implementation Files

| File | Description |
|------|-------------|
| `include/model.h` | Public API: Model class, NDArray struct, ModelError enum |
| `cpp/model.cpp` | Model class implementation |

---

## Thread Safety

The Model class is **NOT thread-safe**:
- Single inference at a time
- Static register file shared across calls
- Designed for single-threaded DSP execution

---

## Compatibility

- **C++14** required (TI C6000 compiler supports this)
- **No exceptions** - uses ModelError enum pattern
- **No dynamic allocation** in hot path - uses static pools
- Works on C66x, C7x, and host emulation
