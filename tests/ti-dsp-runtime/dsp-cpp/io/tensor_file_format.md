# TVM Tensor File Format Specification

## Overview

This document specifies a binary file format for storing one or more tensors,
used for file-based input/output between Python test orchestration and DSP
inference executables.

## Motivation

Previously, `main_dsp.cpp` generated hardcoded test inputs and compared against
compiled-in golden data. This approach had several limitations:

- Required recompilation to change test data
- Golden data comparison logic duplicated between Python and C++
- Difficult to test with arbitrary inputs

The new approach uses file-based I/O:

```
compile_clista.py                    main_dsp.cpp
      │                                   │
      ├─► Compile model                   │
      ├─► Generate input tensor           │
      ├─► Write input.bin                 │
      ├─► Build DSP executable            │
      ├─► Run DSP ──────────────────────► │
      │                                   ├─► Read input.bin
      │                                   ├─► Run inference
      │                                   ├─► Write output.bin
      │   ◄────────────────────────────── │
      ├─► Read output.bin                 │
      └─► Compare vs PyTorch              │
```

## File Structure

```
┌─────────────────────────────────────────────────────────┐
│ File Header (12 bytes)                                  │
├─────────────────────────────────────────────────────────┤
│   magic       : uint32 = 0x54564D54 ("TVMT")           │
│   version     : uint32 = 1                              │
│   num_tensors : uint32                                  │
├─────────────────────────────────────────────────────────┤
│ Tensor 0                                                │
├─────────────────────────────────────────────────────────┤
│   ndim        : int32                                   │
│   shape[ndim] : int64[ndim]                             │
│   dtype_code  : int32  (0=int, 1=uint, 2=float)        │
│   dtype_bits  : int32  (8, 16, 32, 64)                 │
│   data_size   : int64  (bytes)                          │
│   data        : uint8[data_size]                        │
├─────────────────────────────────────────────────────────┤
│ Tensor 1 ...                                            │
├─────────────────────────────────────────────────────────┤
│ Tensor N-1                                              │
└─────────────────────────────────────────────────────────┘
```

### Field Descriptions

#### File Header (12 bytes, fixed)

| Field | Type | Description |
|-------|------|-------------|
| magic | uint32 | Magic number `0x54564D54` ("TVMT" in ASCII) |
| version | uint32 | Format version, currently `1` |
| num_tensors | uint32 | Number of tensors in the file |

#### Per-Tensor Header (variable size)

| Field | Type | Description |
|-------|------|-------------|
| ndim | int32 | Number of dimensions |
| shape | int64[ndim] | Shape array |
| dtype_code | int32 | DLPack type code: 0=int, 1=uint, 2=float |
| dtype_bits | int32 | Bits per element: 8, 16, 32, or 64 |
| data_size | int64 | Size of data in bytes |
| data | uint8[data_size] | Raw tensor data, row-major order |

### Data Layout

- All integers are little-endian (native on x86 and C66x)
- Tensor data is stored in row-major (C-contiguous) order
- No padding between fields or tensors

## API

### Python API (dsp_utils.py)

```python
def write_tensors_to_file(tensors: List[np.ndarray], filename: str) -> None:
    """Write list of numpy arrays to binary tensor file.

    Args:
        tensors: List of numpy arrays to write
        filename: Output file path
    """

def read_tensors_from_file(filename: str) -> List[np.ndarray]:
    """Read list of numpy arrays from binary tensor file.

    Args:
        filename: Input file path

    Returns:
        List of numpy arrays
    """
```

### C API (tensor_file.h)

```c
#define TVM_TENSOR_FILE_MAGIC   0x54564D54  /* "TVMT" */
#define TVM_TENSOR_FILE_VERSION 1

/**
 * Read tensors from binary file.
 *
 * @param filename Path to input file
 * @param num_tensors Output: number of tensors read
 * @return Array of NDArray pointers, or NULL on error
 */
TVMDSPNDArray** TVMDSPReadTensorsFromFile(const char* filename,
                                           int* num_tensors);

/**
 * Write tensors to binary file.
 *
 * @param filename Path to output file
 * @param tensors Array of NDArray pointers
 * @param num_tensors Number of tensors to write
 * @return 0 on success, -1 on error
 */
int TVMDSPWriteTensorsToFile(const char* filename,
                              TVMDSPNDArray** tensors,
                              int num_tensors);

/**
 * Free tensor array returned by TVMDSPReadTensorsFromFile.
 *
 * @param tensors Array of NDArray pointers
 * @param num_tensors Number of tensors
 */
void TVMDSPFreeTensorArray(TVMDSPNDArray** tensors, int num_tensors);
```

## Example Usage

### Python: Write inputs, read outputs

```python
import numpy as np
from dsp_utils import write_tensors_to_file, read_tensors_from_file

# Prepare inputs
input_tensor = np.random.randn(1, 2, 16).astype(np.float32)
write_tensors_to_file([input_tensor], "input.bin")

# ... run DSP executable ...

# Read and validate outputs
outputs = read_tensors_from_file("output.bin")
pytorch_output = model(torch.from_numpy(input_tensor)).numpy()
max_diff = np.max(np.abs(outputs[0] - pytorch_output))
print(f"Max difference: {max_diff:.2e}")
```

### C: Read inputs, write outputs

```c
int num_inputs;
TVMDSPNDArray** inputs = TVMDSPReadTensorsFromFile("input.bin", &num_inputs);
if (inputs == NULL) {
    printf("ERROR: Failed to read input file\n");
    return 1;
}

/* Set up register file with input */
reg_file[0].type_index = kTVMFFINDArray;
reg_file[0].v_obj = (TVMFFIObject*)inputs[0];

/* Run inference... */

/* Write output */
TVMDSPNDArray* outputs[] = { (TVMDSPNDArray*)reg_file[1].v_obj };
TVMDSPWriteTensorsToFile("output.bin", outputs, 1);

/* Cleanup */
TVMDSPFreeTensorArray(inputs, num_inputs);
```

## Benefits

1. **Flexibility**: Test with any input without recompilation
2. **Multi-tensor support**: Handle models with multiple inputs/outputs
3. **Self-describing**: Magic number and version enable validation
4. **Simplicity**: Sequential layout, easy to parse on embedded systems
5. **Separation of concerns**: DSP does inference; Python does validation
