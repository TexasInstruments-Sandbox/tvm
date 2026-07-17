# TVM DSP Integration

This directory contains the infrastructure for running TVM-compiled models on
TI DSP hardware (C66x and C7x) using the minimal DSP runtime.

## Overview

The integration demonstrates:
- Loading TVM-generated code (`lib0.c`) with the DSP runtime
- Parsing model weights from `weights.bin`
- Running inference on PC (host emulation), C66x, and C7x hardware
- Using the RAII-based Model API (`model.h`) to call the generated
  `cg_main_dsp` entry point directly, instead of using the full TVM VM

## Supported Platforms

| Platform | Device | L2 SRAM | Main Memory | Weights | Status |
|----------|--------|---------|-------------|---------|--------|
| Host | PC | 4MB (emulated) | 64MB (emulated) | N/A | ✅ Tested |
| C66x | AWRL6844 | 64KB | 1MB L3 | Embedded | ✅ Tested |
| C7x | J722S | 1.59MB | 128MB DDR | 55.7MB DDR | ✅ Tested |

## Quick Start (Python)

For most users, the `dsp_utils.py` module provides a simple Python interface:

```python
import sys
sys.path.insert(0, "path/to/dsp-cpp")

from dsp_utils import compile_and_run_dsp, compare_results

# Compile and run on DSP (mod is a TVM IRModule with parameters bound)
results = compile_and_run_dsp(
    mod=mod,
    input_data=input_data,  # numpy array or tuple of arrays
    target_string="c_static -mcpu=c66x",
    execution_mode="both",  # "host", "c66x", or "both"
    build_type="Release",   # "Release" or "Debug"
)
```

The workflow uses file-based I/O:
1. Python writes input tensors to `input.bin`
2. DSP executable reads `input.bin`, runs inference, writes `output.bin`
3. Python reads `output.bin` and compares against reference

See `../dsp-tests/test_clista_dsp.py` for a complete example.

## Files

| File | Description |
|------|-------------|
| `main_dsp.cpp` | Main entry point - reads input.bin, runs inference, writes output.bin |
| `CMakeLists.txt` | Build configuration for host, C66x, and C7x targets |
| `dsp_utils.py` | Python utilities for DSP compilation and execution |
| `io/tensor_file.cpp`, `io/tensor_file.h` | Tensor file I/O (`input.bin`/`output.bin`) |
| `io/weights_loader.cpp`, `io/weights_loader.h` | Weights source (filesystem or linker-embedded) |
| `io/tensor_file_format.md` | Binary tensor file format specification |

Note: Constants loading (`TVMDSPParseConstants`, `TVMGetConstants`) is now provided
by the TVM DSP runtime library (`constants/constants_loader.cpp`).

## Prerequisites

### For Host Emulation
- CMake 3.16+
- C++11 compiler (GCC, Clang, or AppleClang)
- TVM DSP runtime library (`libtvm_dsp_runtime_host.a`)

### For C66x Hardware
- TI C6000 Compiler v8.5.0+ (part of CCS)
- MMWAVE-L-SDK-6 v6.1.0.05
- AWRL6844 evaluation board with XDS110 debug probe
- TVM DSP runtime library (`libtvm_dsp_runtime_c66x.a`)

### For C7x Hardware
- TI C7000 Compiler v5.0.0+ (part of CCS)
- J722S evaluation board with XDS110 debug probe
- Code Composer Studio 12.0+
- TVM DSP runtime library (`libtvm_dsp_runtime_c7x.a`)

## Building

### Step 1: Build the DSP Runtime

```bash
# Host emulation runtime
cd $TVM_HOME/src/runtime/ti_dsp
mkdir -p build && cd build
cmake ..
cmake --build .

# C66x runtime (optional, for hardware deployment)
cd $TVM_HOME/src/runtime/ti_dsp
mkdir -p build-c66x && cd build-c66x
cmake -DCMAKE_TOOLCHAIN_FILE=../cmake/toolchain-awrl6844.cmake ..
cmake --build .

# C7x runtime (optional, for J722S hardware deployment)
cd $TVM_HOME/src/runtime/ti_dsp
mkdir -p build-c7x && cd build-c7x
cmake -DCMAKE_TOOLCHAIN_FILE=../cmake/toolchain-j722s-c7x.cmake ..
cmake --build .
```

### Step 2: Generate TVM Model Code

Use TVM's C static backend to compile your model:

```python
import tvm
from tvm import relax

# Load/compile your model to get an IRModule
mod = ...

# Build with c_static target for C66x
target = tvm.target.Target("c_static -mcpu=c66x")
with tvm.transform.PassContext(opt_level=3):
    ex = relax.build(mod, target, exec_mode="compiled", system_lib=True)

# Export generated code to a directory
ex.export_library("model_dir/model_library.tar", target=target)
# Extract to get lib0.c and weights.bin
```

Or use `dsp_utils.py` which handles this automatically:

```python
from dsp_utils import compile_for_dsp
generated_dir = compile_for_dsp(mod, "c_static -mcpu=c66x")
```

### Step 3: Build Host Emulation

```bash
cd tests/ti-dsp-runtime/dsp-cpp

# Debug build (default)
mkdir -p build-debug && cd build-debug
cmake -DGENERATED_CODE_DIR=/path/to/model_dir ..
cmake --build .

# Release build
mkdir -p build && cd build
cmake -DCMAKE_BUILD_TYPE=Release -DGENERATED_CODE_DIR=/path/to/model_dir ..
cmake --build .
```

### Step 4: Build for C66x Hardware

```bash
cd tests/ti-dsp-runtime/dsp-cpp

# Debug build (default)
mkdir -p build-awrl6844-debug && cd build-awrl6844-debug
cmake \
  -DCMAKE_TOOLCHAIN_FILE=$TVM_HOME/src/runtime/ti_dsp/cmake/toolchain-awrl6844.cmake \
  -DGENERATED_CODE_DIR=/path/to/model_dir \
  ..
cmake --build .

# Release build
mkdir -p build-awrl6844 && cd build-awrl6844
cmake \
  -DCMAKE_TOOLCHAIN_FILE=$TVM_HOME/src/runtime/ti_dsp/cmake/toolchain-awrl6844.cmake \
  -DCMAKE_BUILD_TYPE=Release \
  -DGENERATED_CODE_DIR=/path/to/model_dir \
  ..
cmake --build .
```

### Step 5: Build for C7x Hardware (J722S)

```bash
cd tests/ti-dsp-runtime/dsp-cpp

# Debug build (default)
mkdir -p build-j722s-debug && cd build-j722s-debug
cmake \
  -DCMAKE_TOOLCHAIN_FILE=$TVM_HOME/src/runtime/ti_dsp/cmake/toolchain-j722s-c7x.cmake \
  -DGENERATED_CODE_DIR=/path/to/model_dir \
  ..
cmake --build .

# Release build
mkdir -p build-j722s && cd build-j722s
cmake \
  -DCMAKE_TOOLCHAIN_FILE=$TVM_HOME/src/runtime/ti_dsp/cmake/toolchain-j722s-c7x.cmake \
  -DCMAKE_BUILD_TYPE=Release \
  -DGENERATED_CODE_DIR=/path/to/model_dir \
  ..
cmake --build .
```

## Running Inference

The DSP executable uses file-based I/O:
- **Input**: Reads tensor(s) from `input.bin` (must exist in working directory)
- **Output**: Writes tensor(s) to `output.bin`

The binary tensor file format is self-describing with magic number validation.
See `io/tensor_file_format.md` for the full specification.

### Host Emulation

```bash
cd build
# First, create input.bin (typically done by Python)
./cg_dsp
```

Expected output:
```
INFO: Loaded weights from /path/to/model_dir/weights.bin (123456 bytes)
TVM DSP Runtime initialized on Host (PC Emulation)
  Fast pool: 4096 KB
  Main pool: 65536 KB
Loaded 185 constants
Input[0]: shape=[1,2,16], dtype=2.32
Cycles: 42000
Num outputs: 1
Output[0]: shape=[1,128,1], dtype=2.32
Memory: L2 peak=12345, L3 peak=0
Done
```

### C66x Hardware

```bash
# Using the deployment script
$TVM_HOME/src/runtime/ti_dsp/scripts/run_on_c66x.sh build-awrl6844/cg_dsp_c66x.out

# Or with extended timeout
$TVM_HOME/src/runtime/ti_dsp/scripts/run_on_c66x.sh build-awrl6844/cg_dsp_c66x.out --timeout 120000
```

Expected output (similar to host, with C66x memory configuration):
```
INFO: Using embedded weights (123456 bytes)
TVM DSP Runtime initialized on C66x (AWRL6844)
  L2 pool: 0x00850000 - 0x00860000 (64 KB)
  L3 pool: 0x88050400 - 0x88150400 (1024 KB)
...
```

### C7x Hardware (J722S)

```bash
# Using the deployment script
$TVM_HOME/src/runtime/ti_dsp/scripts/run_on_c75x.sh build-j722s/cg_dsp_c7x.out

# Or with extended timeout
$TVM_HOME/src/runtime/ti_dsp/scripts/run_on_c75x.sh build-j722s/cg_dsp_c7x.out --timeout 120000
```

Expected output:
```
INFO: Using embedded weights (340 bytes)
C7x security mode: CXM=3 (RootSupervisor)
TVM DSP Runtime initialized on C7x (J722S_C75)
  L2 pool: 0x7e069000 - 0x7e200000 (1628 KB)
  DDR pool: 0x108000000 - 0x110000000 (128 MB)
...
```

## Configuration Options

### CMake Variables

| Variable | Default | Description |
|----------|---------|-------------|
| `CMAKE_BUILD_TYPE` | `Debug` | Build type: `Debug` or `Release` |
| `TVM_DSP_TARGET` | `host` | Target: `host`, `c66x`, `c7x_host`, or `c7x-dynmod` |
| `TVM_DSP_DEVICE` | (none) | Device variant: `awrl6844` or `j722s` |
| `GENERATED_CODE_DIR` | `../cstatic-tests` | Directory containing lib0.c and weights.bin |
| `TVM_HOME` | Auto-detect | Path to TVM repository |
| `WEIGHTS_FILE` | `${GENERATED_CODE_DIR}/weights.bin` | Path to weights file |

### Build Types

| Build Type | Host Flags | C66x Flags |
|------------|------------|------------|
| Debug | `-g -O0` | `-g --opt_level=0` |
| Release | `-O3 -DNDEBUG` | `-O3 --opt_for_speed=5` |

### Model Configuration

Set via CMake, compiled in as preprocessor definitions consumed by the
Model API:

```bash
cmake -DMODEL_ENTRY_FUNCTION=main -DMODEL_NUM_INPUTS=1 \
      -DMODEL_RETURNS_TUPLE=ON -DGENERATED_CODE_DIR=/path/to/model_dir ..
```

Input shape and data are provided via `input.bin` at runtime (no recompilation needed).

## Generated Code Notes

When using `-mcpu=c66x` target, the generated `lib0.c`:
- Includes DSP-specific headers directly (`ffi_types.h`, etc.)
- Uses `TVM_DSP_SKIP_CG_MAIN` to exclude exception-based wrapper
- Is compiled as C++ to support the DSP runtime API

The CMakeLists.txt handles this automatically:
```cmake
set_source_files_properties(${GENERATED_SOURCES} PROPERTIES LANGUAGE CXX)
```

## Troubleshooting

### "Failed to load weights"
- Verify `weights.bin` exists at the configured path
- Check file permissions
- For C66x: ensure weights are properly embedded

### "TVMFFIFunctionCall with NULL function"
- Ensure `InitVMBuiltins()` is called before `__vmtir__main`
- Verify VM builtins are registered with `TVMDSPRegisterVMBuiltins()`

### Inference timeout on C66x
- Check memory usage in the `.map` file
- Verify L2/L3 pool sizes are sufficient
- Try increasing timeout: `--timeout 300000`

### Linker errors about LLVM paths
- The toolchain file should clear host linker flags
- If not, manually set: `-DCMAKE_EXE_LINKER_FLAGS=""`

## Architecture

```
┌─────────────────────────────────────────────────────────────┐
│                    Python (dsp_utils.py)                    │
│  - write_tensors_to_file() -> input.bin                     │
│  - Run DSP executable                                       │
│  - read_tensors_from_file() <- output.bin                   │
│  - Compare against PyTorch reference                        │
└─────────────────────────────────────────────────────────────┘
                              │
                              ▼
┌─────────────────────────────────────────────────────────────┐
│                       main_dsp.cpp                          │
│  - GetWeightsData() - loads weights.bin                     │
│  - model.Load() - parses constants                          │
│  - Read input tensor(s) from input.bin                      │
│  - model.InferMulti() - calls generated cg_main_dsp()       │
│  - Write output tensor(s) to output.bin                     │
└─────────────────────────────────────────────────────────────┘
                              │
                              ▼
┌─────────────────────────────────────────────────────────────┐
│                         lib0.c                              │
│  - Generated TIR code                                       │
│  - cg_main_dsp(inputs, num_inputs, outputs, num_outputs)    │
│  - Operator implementations (conv1d, add, etc.)             │
└─────────────────────────────────────────────────────────────┘
                              │
                              ▼
┌─────────────────────────────────────────────────────────────┐
│                    TVM DSP Runtime                          │
│  - TVMDSPParseConstants() / TVMGetConstants()               │
│  - vm.builtin.alloc_storage, alloc_tensor, reshape, copy    │
│  - Tensor file I/O (input.bin, output.bin)                  │
│  - Memory pools (L2/L3 SRAM)                                │
│  - NDArray management                                       │
│  - weights.bin parsing (zero-copy)                          │
└─────────────────────────────────────────────────────────────┘
```

## J722S C7x Memory Layout

The J722S C75 DSP has 2MB of L2 SRAM local to the core plus access to DDR memory.
The linker command file (`j722s/linker_c7x.cmd`) defines the memory layout for
standalone JTAG execution.

### Code-in-DDR Mode (Default)

The default configuration places application code in DDR (cached via MMU) to
maximize L2 SRAM for the TVM data heap. Only boot code and MMU init remain in L2.

### L2 SRAM Regions (2MB at 0x7E000000)

| Region | Address Range | Size | Purpose |
|--------|---------------|------|---------|
| L2_VECS | 0x7E000000 - 0x7E003FFF | 16KB | Interrupt/exception vectors |
| L2_SECVECS | 0x7E004000 - 0x7E007FFF | 16KB | Secure vectors |
| L2_BOOT | 0x7E008000 - 0x7E008FFF | 4KB | Boot code (`_c_int00_secure`) |
| L2_INIT | 0x7E009000 - 0x7E018FFF | 64KB | Pre-MMU init code (`.text:l2_init`) |
| L2_DATA | 0x7E019000 - 0x7E038FFF | 128KB | Data sections (`.data`, `.bss`, `.cio`) |
| L2_STACK | 0x7E039000 - 0x7E068FFF | **192KB** | Stack (`.stack`) - expanded |
| L2_SCRATCH | 0x7E069000 - 0x7E1FFFFF | **1.59MB** | **TVM L2 pool** (`.tvm_l2_heap`) |
| L2_AUX | 0x7F000000 - 0x7F03BFFF | 240KB | Auxiliary storage (`.l2aux`) |

**Note:** The standard malloc heap (`.sysmem`) has been moved to DDR to maximize L2 for
the TVM runtime. This allows a larger stack (192KB vs 128KB) and keeps all of L2_SCRATCH
available for TVM tensor allocations.

**Key difference from code-in-L2 mode**: Application code (`.text`) goes to DDR,
freeing ~1.3MB additional L2 space for the TVM heap.

### DDR Regions

| Region | Address Range | Size | Purpose |
|--------|---------------|------|---------|
| DDR_C7X_BOOT | 0xAD200000 | 1KB | DDR boot code |
| DDR_C7X_VECS | 0xAD400000 | 16KB | DDR vectors |
| DDR_C7X_SECVECS | 0xAD600000 | 16KB | DDR secure vectors |
| DDR_C7X_CODE | 0xAD604000 - 0xAD803FFF | 2MB | Application code (cached via MMU) |
| DDR_SYSMEM | 0xAD804000 - 0xAD843FFF | 256KB | Standard malloc heap (`.sysmem`) |
| DDR_C7X_MAIN | 0xAD844000 - 0xB0FFFFFF | ~55.7MB | **Model weights** (`.rodata.weights`) |
| DDR_C7X_EXTENDED | 0x108000000 - 0x10FFFFFFF | 128MB | **TVM DDR heap** (runtime tensors) |

**Note:** DDR_C7X_EXTENDED uses high DDR above 4GB (virtual addresses 0x108000000+
map to physical 0x888000000+ per TI SDK memory map). This requires 8GB LPDDR4 and
is verified by the TI RTOS SDK `app_mem_map.h`.

### TVM Runtime Memory Pools

The TVM DSP runtime uses two memory pools configured via linker symbols:

| Pool | Region | Address Range | Size | Usage |
|------|--------|---------------|------|-------|
| L2 (Fast) | L2_SCRATCH | 0x7E069000 - 0x7E1FFFFF | **1.59MB** | Intermediate tensors, frequently accessed data |
| DDR (Main) | DDR_C7X_EXTENDED | 0x108000000 - 0x110000000 | **128MB** | Runtime tensor allocations, large outputs |

**Model weights** are placed in DDR_C7X_MAIN (~55.7MB) via the `.rodata.weights` section,
separate from the runtime heap.

**Linker symbols** (read by TVM runtime at initialization):
```
__TVM_DSP_L2_HEAP_START  = 0x7E069000
__TVM_DSP_L2_HEAP_END    = 0x7E200000
__TVM_DSP_DDR_HEAP_START = 0x108000000
__TVM_DSP_DDR_HEAP_END   = 0x110000000
```

### Important: Memory Pool Separation

The TVM heaps **must not overlap** with `.sysmem` (standard malloc heap used by
`printf`, `fopen`, etc.). Earlier versions had overlap issues causing memory
corruption when printf's internal buffers overwrote TVM tensor data.

**Current layout (correct, code-in-DDR mode):**
- `.sysmem` (malloc) → DDR_SYSMEM (0xAD804000, 256KB) - in low DDR
- `.stack` → L2_STACK (0x7E039000, 192KB) - expanded
- `.rodata.weights` → DDR_C7X_MAIN (0xAD844000, ~55.7MB) - model weights
- TVM L2 pool → L2_SCRATCH (0x7E069000, 1.59MB)
- TVM DDR pool → DDR_C7X_EXTENDED (0x108000000, 128MB) - in high DDR (>4GB)

**Why use extended DDR for TVM heap:** The J722S has 8GB LPDDR4 with high DDR
addresses (0x108000000+) mapped via MMU. Using extended DDR for the TVM runtime
heap frees DDR_C7X_MAIN for model weights (up to ~55.7MB), enabling larger models.

To customize the TVM pool locations, modify the linker symbols in `linker_c7x.cmd`.
The runtime reads these symbols at initialization, so no library rebuild is needed.

### Section Placement Summary

| Section | Region | Description |
|---------|--------|-------------|
| `.vecs` | L2_VECS | Interrupt vector table |
| `.text` | DDR_C7X_CODE | Application code (cached via MMU) |
| `.text:l2_init` | L2_INIT | Pre-MMU init code (runs before DDR cached) |
| `.const` | DDR_C7X_CODE | Read-only constants (small) |
| `.rodata.weights` | DDR_C7X_MAIN | **Model weights** (up to ~55.7MB) |
| `.data` | L2_DATA | Initialized global data |
| `.bss` | L2_DATA | Zero-initialized data |
| `.cio` | L2_DATA | Console I/O buffer (printf) |
| `.stack` | L2_STACK | Program stack (192KB) |
| `.sysmem` | DDR_SYSMEM | Standard heap (malloc/free) - 256KB in DDR |
| `.tvm_l2_heap` | L2_SCRATCH | TVM fast memory pool (1.59MB) |
| `.tvm_ddr_heap` | DDR_C7X_EXTENDED | TVM main memory pool (128MB in high DDR) |
| `.fardata` | DDR_C7X_MAIN | Large data arrays |

## C7x MMU Configuration

The C7x DSP on J722S requires MMU (Memory Management Unit) configuration for
cached and executable DDR access. The application is responsible for
initializing the MMU before calling any TVM runtime functions.

### Why MMU is Needed

Without proper MMU configuration, DDR memory accesses are:
- Uncached (slow, every access goes to external memory)
- Potentially non-executable (can't run code from DDR)

The MMU enables:
- Cached DDR access for model weights and tensors
- Executable DDR regions for code-in-DDR mode (maximizes L2 for data)
- Proper memory attributes for L2 SRAM and peripheral regions

### MMU Register Configuration

The MMU uses direct ECR (Extended Control Register) access, adapted from
TI's edgeai-tidl-kernels approach:

| Register | ECR | Value | Description |
|----------|-----|-------|-------------|
| SCR | ECR784 | 0x80000000000000C1 | System Control: MMU + caches enabled |
| TCR0 | ECR785 | 0x0000000000002A21 | Translation Control: 4KB granule, 2GB space |
| TBR0 | ECR787 | (page table addr) | Translation Base: points to level 1 table |
| MAR | ECR789 | 0x3D3D3D2915032A00 | Memory Attributes: cacheable/device types |

### MAIR (Memory Attribute Indirection Register)

The MAR value packs 8 memory attribute configurations (MAIR0-7):

| Index | Value | Description |
|-------|-------|-------------|
| MAIR0 | 0x00 | Device-nGnRnE (strongly ordered device memory) |
| MAIR1 | 0x2A | Write-Through No-Allocate |
| MAIR2 | 0x03 | Device-nGnRE |
| MAIR3 | 0x15 | Write-Through Allocate |
| MAIR4 | 0x29 | Non-cacheable |
| MAIR5 | 0x3D | Write-Back Read-Allocate Write-Allocate (normal cached) |
| MAIR6 | 0x3D | Write-Back Read-Allocate Write-Allocate |
| MAIR7 | 0x3D | Write-Back Read-Allocate Write-Allocate |

### Page Table Structure

The J722S implementation uses ARMv8-style 2-level page tables with 1GB L1 blocks
and 2MB L2 blocks:

```
Level 1 (512 entries)         Level 2 (512 entries)
┌─────────────────────┐       ┌─────────────────────┐
│ [0] 0x00-0x3F: Dev  │ Block │                     │
│ [1] 0x40-0x7F: Tbl  │──────>│ MSMC (0x70): Cached │
│ [2] 0x80-0xBF: DDR  │ Block │ L2   (0x7E): Cached │
│ [3] 0xC0-0xFF: DDR  │ Block │ L2AUX(0x7F): Cached │
│ [4] 1.0-1.3G: DDR   │ Block │                     │
│ [5] 1.4-1.7G: DDR   │ Block └─────────────────────┘
└─────────────────────┘
```

### Memory Regions

| Region | Address Range | Size | Attributes |
|--------|---------------|------|------------|
| Peripherals | 0x00000000-0x3FFFFFFF | 1GB | Device, Non-Shareable |
| Secure Proxy | 0x48000000-0x4FFFFFFF | 128MB | Device (for DMSC comm) |
| MSMC | 0x70000000-0x703FFFFF | 4MB | Cached, Outer Shareable |
| L2 SRAM | 0x7E000000-0x7E1FFFFF | 2MB | Cached, Non-Shareable |
| L2 AUX | 0x7F000000-0x7F03FFFF | 256KB | Cached, Non-Shareable |
| DDR | 0x80000000-0xFFFFFFFF | 2GB | Cached, Outer Shareable |
| DDR (ext) | 0x100000000-0x17FFFFFFF | 2GB | Cached, Outer Shareable (TVM heap at 0x108000000) |

### Block Descriptor Attributes

| Attribute | Bits | Values |
|-----------|------|--------|
| Type | [1:0] | 0b01 = Block descriptor |
| AttrIndx | [4:2] | MAIR index (0-7) |
| NS | [5] | Non-Secure |
| AP | [7:6] | Access permissions (0b00 = RW) |
| SH | [9:8] | Shareability (0=NSH, 2=OSH, 3=ISH) |
| AF | [10] | Access Flag (must be 1) |
| Address | [47:21] | 2MB-aligned physical address (L2) |

### Shareability Considerations

- **Non-Shareable (NSH)**: Use for L2 SRAM which is local to each C75 core
- **Outer Shareable (OSH)**: Use for DDR and MSMC for multi-core coherency

### MMU Source Files

The MMU implementation is in `j722s/`:

| File | Description |
|------|-------------|
| `mmu.c` | MMU initialization with detailed comments |
| `c75_asm.S` | Assembly functions for ECR register access |
| `boot_c75.c` | Boot code that calls `MmuP_init()` before `main()` |
| `linker_c7x.cmd` | Linker script with page table sections |

### Security Mode

The C7x runs in CXM=3 (RootSupervisor) mode when loaded via JTAG. This mode:
- Has full access to MMU configuration registers
- Can modify page tables and enable/disable MMU
- Requires direct ECR register access (SDK MmuP functions may not work)

The runtime detects the security mode and reports it during initialization:
```
C7x security mode: CXM=3 (RootSupervisor)
```

## License

Apache License 2.0
