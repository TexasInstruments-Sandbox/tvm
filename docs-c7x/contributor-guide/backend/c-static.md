# C Static Backend

The `c_static` backend is a specialized C code generator for TVM that produces
standalone static binaries for Relax VM execution. It generates portable C/C++
code suitable for embedded deployment, including optimized support for TI C66x
and C7x DSP processors.

## Table of Contents

- [Overview](#overview)
- [Target Configuration](#target-configuration)
- [Quick Start](#quick-start)
- [C66x DSP Quick Start](#c66x-dsp-quick-start)
- [C7x DSP and DLOAD Deployment](#c7x-dsp-and-dload-deployment)
- [C7x DMA Tiling](#c7x-dma-tiling)
- [Architecture](#architecture)
- [C++ API for VM Operations](#c-api-for-vm-operations)
- [Building and Testing](#building-and-testing)

## Overview

### When to Use c_static

Use the `c_static` backend when you need:

- **Static binary deployment**: Self-contained executables without shared library
  dependencies
- **Embedded systems**: Resource-constrained environments (microcontrollers, DSPs)
- **Cross-compilation**: Portable C code that compiles on any target toolchain
- **DSP deployment**: Optimized code generation for TI C66x/C7x processors

### Key Features

- Complete C code generation for TVM Relax VM
- Automatic wrapper function generation for easy integration
- Parameter serialization in binary or C source formats
- Multi-input/multi-output model support
- TI DSP support with cycle-accurate profiling
- C++ API mode for reduced FFI overhead (12% faster on DSP)

## Target Configuration

### Basic Usage

```python
import tvm
from tvm import relax

# Basic c_static target
target = tvm.target.Target("c_static")

# Target TI C66x DSP
target = tvm.target.Target("c_static -mcpu=c66x")

# Target TI C7x DSP (AM67A/J722S via DLOAD)
target = tvm.target.Target("c_static -mcpu=c7x")

# Compile model
mod = relax.transform.LegalizeOps()(mod)
ex = relax.build(mod, target=target)
```

### Target Attributes Reference

| Attribute | Type | Default | Description |
|-----------|------|---------|-------------|
| `mcpu` | String | `generic` | Target CPU: `c66x`, `c7x` |
| `use-cpp-api` | Bool | `true` | Use C++ API for VM operations (bypass FFI) |
| `skip-runtime-checks` | Bool | `true` | Skip tensor shape/type validation |
| `profile-layers` | Bool | `false` | Enable per-layer cycle profiling |
| `debug-alloc` | Bool | `false` | Enable diagnostic allocation tracing |
| `constants-byte-alignment` | Int | `64` (DSP) | Byte alignment for constant arrays |
| `l1d-cache-size` | Int | `32768` | L1D cache size in bytes (32KB) |
| `l2-sram-size` | Int | `1310720` | L2 SRAM size in bytes (1.25MB, J722S C7x) |
| `vector-width` | Int | `128` | SIMD vector width in bits |
| `mmalib` | Bool | `false` | Use MMALIB kernels for eligible matmul/conv2d ops (requires `mcpu=c7x`) |

### TI DSP Runtime

When `-mcpu=c66x` or `-mcpu=c7x` is specified, the generated code targets
the TI DSP runtime (`src/runtime/ti_dsp/`) instead of the standard TVM VM
runtime. The code generator handles this switch automatically: it emits
DSP-specific headers, `TVM_DSP_EXPORT` on entry points, TI compiler
pragmas for loop optimization, and calls to `TVMDSPBuiltin*` functions
instead of the standard VM FFI dispatch.

The DSP runtime is a lightweight, self-contained C++14 library (~100 KB)
designed for bare-metal and RTOS environments. Key differences from the
standard TVM runtime:

- **No VM class** -- c_static emits direct function calls to builtin
  stubs instead of bytecode interpretation
- **Static memory pools** -- pre-allocated L2 SRAM (fast) and L3/DDR
  (main) pools with bump-pointer allocation; no `malloc()` at runtime
- **No exceptions or RTTI** -- error handling via `ModelError` enum
  return codes, compatible with TI CGT compiler constraints
- **C struct FFI** -- 16-byte `TVMFFIAny` with manual ref counting
  instead of C++ `tvm::ffi::Any` with smart pointers
- **Platform abstraction** -- implementations for host emulation (PC),
  C66x (AWRL6844), and C7x (AM67A/J722S)

See [DSP Runtime Internals](../dsp-runtime/internals.md) for build
instructions, memory architecture, and platform configuration. The full
`model.h` C++ API reference is in `src/runtime/ti_dsp/MODEL_API.md` in
the source tree.

## Quick Start

### 1. Compile a Model

```python
import tvm
from tvm import relax
import numpy as np

# Load or create your model
# Example: Simple MLP
from tvm.script import ir as I, relax as R

@I.ir_module
class MLPModule:
    @R.function
    def main(x: R.Tensor((1, 784), "float32")):
        # ... model definition
        pass

# Compile with c_static backend
target = tvm.target.Target("c_static")
mod = relax.transform.LegalizeOps()(MLPModule)
ex = relax.build(mod, target=target)

# Export artifacts
ex.export_library("model_lib.so")
```

### 2. Generated Files

After compilation, you get:
- `lib0.c` - Main computation kernels
- `lib0.h` - Header with function declarations
- `weights.bin` - Model parameters (if binary mode)

### 3. Integration Example

```c
#include "lib0.h"
#include <tvm/runtime/crt/ndarray.h>

int main() {
    // Allocate input/output tensors
    NDArray input = /* ... */;
    NDArray output = cg_main(input);
    // Process output
    return 0;
}
```

## C66x DSP Quick Start

### Target Configuration for C66x

```python
import tvm

# C66x with default optimizations (recommended)
target = tvm.target.Target("c_static -mcpu=c66x")
# Defaults enabled: -use-cpp-api=1, -skip-runtime-checks=1

# With layer profiling
target = tvm.target.Target("c_static -mcpu=c66x -profile-layers=1")

# Disable optimizations for debugging
target = tvm.target.Target(
    "c_static -mcpu=c66x "
    "-skip-runtime-checks=0 "
    "-use-cpp-api=0"
)
```

### Build Environment Setup

```bash
# Required environment variables
export TI_CGT_C6000_PATH=/path/to/ti-cgt-c6000_8.5.0.LTS
export CCS_ROOT=/path/to/ccs

# Build DSP runtime
cd $TVM_HOME/src/runtime/ti_dsp
mkdir build-c66x && cd build-c66x
cmake -DCMAKE_TOOLCHAIN_FILE=../cmake/toolchain-awrl6844.cmake ..
cmake --build .
```

### Run on C66x Hardware

```bash
# Deploy and run
$TVM_HOME/src/runtime/ti_dsp/scripts/run_on_c66x.sh program.out

# With extended timeout (milliseconds)
$TVM_HOME/src/runtime/ti_dsp/scripts/run_on_c66x.sh program.out --timeout 120000
```

### DSP Test Framework

```bash
cd $TVM_HOME
export PYTHONPATH=$TVM_HOME/python:$PYTHONPATH

# Run on C66x host emulation
pytest tests/ti-dsp-runtime/dsp-tests/ -v --dsp-mode=c66x_host

# Run on C66x hardware
pytest tests/ti-dsp-runtime/dsp-tests/ -v --dsp-mode=c66x

# With layer profiling
pytest tests/ti-dsp-runtime/dsp-tests/ -v --dsp-mode=c66x --profile-layers
```

## C7x DSP and DLOAD Deployment

The c_static backend supports deploying TVM-compiled models to the TI C7x
DSP on the AM67A (J722S) SoC via runtime dynamic loading. Instead of
linking the model into the DSP firmware at build time, the generated code is
compiled into a relocatable C7x ELF module that the firmware's DLOAD
dynamic linker loads at runtime over RPMessage IPC from Linux.

### End-to-End Flow

```
  Dev Host                                   AM67A / J722S Board
 ─────────                                  ─────────────────────

 ┌────────────────────────┐
 │    TVM Compiler        │
 │    (Python)            │
 │                        │
 │  target = "c_static    │
 │           -mcpu=c7x"   │
 └──────────┬─────────────┘
            │
            │  relax.build()
            ▼
 ┌──────────┴─────────────┐
 │                        ├───▶  lib0.c       (computation kernels)
 │   c_static codegen     │
 │                        ├───▶  weights.bin  (model parameters)
 └──────────┬─────────────┘
            │
            │  TI CGT C7000 compiler
            │  + DLOAD linker script
            ▼
 ┌──────────┴─────────────┐     Relocatable C7x ELF:
 │                        │      .text           compiled lib0.c
 │   lib0.out             │      .rodata.weights embedded weights.bin
 │   (DLOAD module)       │      --dynamic=lib   relocatable
 │                        │      --import=        symbols from firmware
 └──────────┬─────────────┘
            │
            │  scp to AM67A
            │
  ══════════╪═════════════════════════════════════════════════════
            │
            ▼
 ┌──────────┴─────────────┐          ┌────────────────────────────┐
 │                        │  RPMsg   │                            │
 │  c7x_compute CLI       ├─────────▶│  C7x DSP Firmware          │
 │  (ARM Linux)           │  IPC     │  (FreeRTOS)                │
 │                        │          │                            │
 │  1. load lib0.out      ├─────────▶│  DLOAD: parse ELF,         │
 │                        │          │    allocate in DDR heap,   │
 │                        │          │    resolve exported syms,  │
 │                        │          │    apply relocations       │
 │                        │          │                            │
 │  2. infer               ├─────────▶│  Call cg_main_dsp():       │
 │     --input X.bin      │          │    build DLTensors from    │
 │                        │◀─────────┤    shared DDR, run model,  │
 │                        │          │    write output to DDR     │
 │                        │          │                            │
 │  3. unload             ├─────────▶│  Free module segments      │
 │                        │          │                            │
 └────────────────────────┘          └────────────────────────────┘
```

### Target Configuration for C7x

```python
import tvm

# C7x with default optimizations (recommended)
target = tvm.target.Target("c_static -mcpu=c7x")

# With layer profiling
target = tvm.target.Target("c_static -mcpu=c7x -profile-layers=1")
```

When `mcpu=c7x` is set, the following defaults apply:

| Attribute | Default | Effect |
|-----------|---------|--------|
| `use-cpp-api` | `true` | Direct C++ calls instead of FFI dispatch (~12% faster) |
| `skip-runtime-checks` | `true` | Skip tensor shape/type validation (~5% faster) |
| `constants-byte-alignment` | `64` | 64-byte alignment for cache-line efficiency |

The code generator emits `#include <c7x.h>` (vs `<c6x.h>` for C66x),
`TVM_DSP_EXPORT __declspec(dllexport)` on wrapper functions for DLOAD
symbol visibility, TI `#pragma MUST_ITERATE` / `#pragma UNROLL` for loop
optimization, and a `cg_main_dsp` entry point that the firmware resolves
by name at load time.

### Weight Handling

TVM serializes model parameters (convolution filters, biases, batch norm
statistics, etc.) into `weights.bin` using its binary parameter format.
For C7x DLOAD deployment, these weights are embedded directly into the
ELF module:

```
 weights.bin                         lib0.out (ELF)
 ┌───────────┐                       ┌────────────────────────┐
 │ TVM param │    bin_to_asm.py      │ .text                  │
 │ format    ├──────────────────────▶│ .rodata.weights:       │
 │ (binary)  │    converts to TI     │   _binary_weights_..   │
 └───────────┘    assembly with      │   _binary_weights_..   │
                  .sect directive    │   _binary_weights_..   │
                                     └────────────────────────┘
```

The `bin_to_asm.py` script converts `weights.bin` to a TI assembly file
with a `.rodata.weights` section directive, exposing three symbols:
- `_binary_weights_bin_start` -- pointer to weights data
- `_binary_weights_bin_end` -- end marker
- `_binary_weights_bin_size` -- total size in bytes

The linker script places `.rodata.weights` alongside other read-only data.
At load time, DLOAD allocates space for the entire ELF in the TVM DDR
heap (352 MiB), and the firmware's TVM model manager locates the weights
via the exported symbols and constructs DLTensor descriptors from the TVM
binary parameter format.

Embedding weights avoids a separate `model-load` IPC step and keeps the
deployment as a single file (`lib0.out`). For ResNet-18, this produces a
~47 MB ELF containing ~1 MB of code and ~46 MB of weights.

### Building a DLOAD Module

The build process uses a two-stage link with TI CGT C7000:

**Stage 1** -- Build a pseudo-firmware (`dsp_syms.out`) containing stub
`__declspec(dllexport)` declarations of the symbols the firmware
exports (127 in the current symbol list, spanning the C library, TVM
runtime, VM builtins, math, and MMALIB wrappers). This provides
link-time symbol definitions so the TI linker can resolve references in
`lib0.c` without the actual firmware binary.

**Stage 2** -- Compile `lib0.c` with the TI C++ compiler and link it
against `dsp_syms.out` using the DLOAD linker script (`c7x_dynmod.cmd`):
- `--dynamic=lib` -- produce a relocatable shared library ELF
- `--relocatable` -- emit C7x dynamic relocation entries
- `--import=<symbol>` -- declare symbols resolved by DLOAD at runtime
- `.rodata.weights` section with embedded `weights.bin`

The output `lib0.out` is a standard C7x ELF that DLOAD can parse and
relocate into DSP memory at runtime.

The dynmod build infrastructure lives in the `tvm` repo, under
`src/runtime/ti_dsp/dynmod/`:
- `dynmod/c7x_dynmod/c7x_dynmod.cmd` -- DLOAD linker script
- `dynmod/c7x_dynmod/dsp_syms.c` -- pseudo-firmware symbol stubs
- `dynmod/CMakeLists.txt` -- standalone cmake project, target `c7x_dynmod`
- `src/runtime/ti_dsp/scripts/bin_to_asm.py` -- weights embedder

The `tests/ti-dsp-runtime` repository's `dsp-cpp/CMakeLists.txt` can also
drive this build (via `-DC7X_DYNMOD=ON`), referencing the same files in
the `tvm` repo rather than a local copy.

### Running on AM67A Hardware

The C7x firmware and host CLI are in `src/runtime/ti_dsp/firmware/c7x/`.
See [Deploying Firmware](../../user-guide/deploying-firmware.md) for
build, deploy, and usage instructions, and
[Firmware Design Deep-Dive](../firmware/design-deep-dive.md) for
architecture and DLOAD internals.

```bash
# On dev host: compile model and build DLOAD module
target = tvm.target.Target("c_static -mcpu=c7x")
# ... (produces lib0.c + weights.bin)
# ... (TI CGT C7000 build produces lib0.out)

# On dev host: copy to board
scp lib0.out root@am67a:/tmp/

# On AM67A: load, infer, unload
c7x_compute load /tmp/lib0.out
c7x_compute infer <handle> 0 --input in.bin --output out.bin --dtype float32
c7x_compute unload <handle>
```

### Automated Testing (pytest)

The `tests/ti-dsp-runtime` repository provides end-to-end pytest tests that
automate the full pipeline: TVM compilation, C7x ELF build, SCP to board,
and inference verification.

```bash
cd $TVM_HOME
export PYTHONPATH=$TVM_HOME/python:$PYTHONPATH

# Conv2D on C7x via DLOAD
pytest tests/ti-dsp-runtime/dsp-tests/test_conv2d_dsp.py -v --dsp-mode=c7x_dload

# ResNet-18 on C7x via DLOAD
pytest tests/ti-dsp-runtime/dsp-tests/test_resnet_dsp.py -v --dsp-mode=c7x_dload --use-cpp-api
```

### Cross-Repository Structure

The C7x DLOAD flow spans two repositories:

| Component | Repository | Path |
|-----------|-----------|------|
| c_static code generator | `tvm` | `src/target/c_static/` |
| TI DSP runtime | `tvm` | `src/runtime/ti_dsp/` |
| `bin_to_asm.py` (weights embedder) | `tvm` | `src/runtime/ti_dsp/scripts/` |
| DLOAD linker script + stubs | `tvm` | `src/runtime/ti_dsp/dynmod/c7x_dynmod/` |
| C7x DLOAD build scripts (test harness) | `tests/ti-dsp-runtime` | `dsp-cpp/` |
| DSP firmware + host CLI | `tvm` | `src/runtime/ti_dsp/firmware/c7x/` |
| pytest integration tests | `tests/ti-dsp-runtime` | `dsp-tests/` |

## C7x DMA Tiling

When targeting `c_static -mcpu=c7x`, the compiler automatically applies
DMA-based double-buffered tiling to conv2d layers whose working set
exceeds the L2 SRAM budget.  This overlaps data movement (DDR to L2)
with computation using the C7x DMA engine.

### How It Works

The DMA tiling pipeline has three stages:

**Stage 1 -- Scheduling (Relax pipeline)**

`ScheduleC7xDMATiling` runs after `FuseTIR`.  For each PrimFunc it
tries three strategies in order, applying the first one that matches:

1. **NHWC H-tiling** (preferred, `conv2d_nhwc` blocks) -- splits the
   output height (H) loop so each tile's input activation strip
   (double-buffered) fits in the L2 SRAM budget (default 384 KB). Works
   with fused quantized kernels because per-channel post-conv ops
   (requantize, bias, relu, clip, cast) are independent of H. Also
   caches weights into L2 (invariant across H-tiles) when they fit
   alongside the double-buffered input strip.
2. **NCHW OC-tiling** (legacy, `conv2d_nchw` blocks) -- splits the
   output-channel loop into `oc_outer` / `oc_inner` so input + weight
   tiles fit double-buffered in the L2 budget. Only applied to
   standalone conv2d (skipped if the PrimFunc has fused post-conv
   blocks, since OC-tiling breaks per-output-channel fused ops).
3. **N-tiling** (`dequantize_matmul_acc` blocks) -- caches the weight
   matrix into L2 if it fits; SW-pipelined N-tiling for weights that
   don't fit is not yet implemented.

Whichever strategy applies, matching `cache_read`s are inserted into
`global.l2sram` scope, copy loops are fused into a single flat loop per
cache block, and the outer loop is annotated with software pipeline
metadata (e.g. for OC-tiling: `software_pipeline_stage = [0, 0, 1]`
(DMA, DMA, compute), `software_pipeline_order = [0, 1, 2]`,
`software_pipeline_async_stages = [0]`).

**Stage 2 -- TIR lowering (custom C7x TIR pipeline)**

`_c7x_dma_tir_pipeline()` is a stripped-down fork of the default TIR
pipeline tailored for a single-core CPU DSP (no GPU, no threads, no
shared memory).  The key difference: `LowerAsyncDMA` and
`LowerDMAToExtern` run immediately after `FlattenBuffer`, before
`NarrowDataType(32)`.

`NarrowDataType(32)` converts int64 loop indices to int32, which
changes the index arithmetic enough that `IdentifyMemCpy` (used by
`LowerAsyncDMA`) can no longer prove contiguity of the copy regions.
Running DMA lowering while indices are still int64 avoids this.

The pass ordering:

```
InjectSoftwarePipeline   -- expand annotations into async prologue/body/epilogue
LowerOpaqueBlock
FlattenBuffer            -- flatten multi-dim buffers to 1D
LowerAsyncDMA            -- convert copy loops to dma_copy/dma_wait intrinsics
LowerDMAToExtern         -- convert intrinsics to call_extern("tvm_dsp_dma_copy", ...)
NarrowDataType(32)       -- safe now, DMA calls are opaque externs
StorageRewrite
MakePackedAPI
```

**Stage 3 -- C codegen and linking**

The c_static codegen emits `tvm_dsp_dma_copy()` and
`tvm_dsp_dma_wait()` as regular C function calls.  The DSP runtime
header (`dma/tvm_dsp_dma.h`) is included via `kDSPHeaders` in the code
template.  At link time, the calls resolve to:

- **Host emulation** (`tvm_dsp_dma_host.c`): synchronous `memcpy`
- **C7x hardware** (`tvm_dsp_dma.c`): synchronous `memcpy` (Phase 1);
  TI DmaUtilsAutoInc3d async DMA planned for Phase 3

### Generated Code Structure

For a conv2d with OC=128 tiled at oc_tile=64 (2 tiles):

```c
// Prologue: DMA tile 0 (input + weight) as a group
tvm_dsp_dma_copy(0, &l2_buf[0],      &ddr_input[0],  460800, 0);
tvm_dsp_dma_copy(0, &l2_buf[921600], &ddr_weight[0],  73728, 0);

// Body: DMA tile 1, overlapped with compute on tile 0
tvm_dsp_dma_copy(0, &l2_buf[115200],  &ddr_input[0],  460800, 0);
tvm_dsp_dma_copy(0, &l2_buf[995328],  &ddr_weight[73728], 73728, 0);
tvm_dsp_dma_wait(0, 1);   // wait for tile 0 DMA (allow 1 group in flight)
// ... compute on tile 0 using l2_buf[0..] ...

// Epilogue: compute on tile 1
tvm_dsp_dma_wait(0, 0);   // wait for tile 1 DMA (all complete)
// ... compute on tile 1 using l2_buf[115200..] ...
```

The `dma_wait(queue, max_inflight)` calls synchronize at the group
level.  Each async commit group contains 2 copies (input + weight).
`max_inflight=1` means "at most 1 group still in flight", so the
prologue group must be done before compute starts, while the body
group can overlap with tile 0 computation.

### Tiling Decisions

The pass skips tiling when the full working set fits in L2.  For a
standalone (non-fused) 4-layer conv2d stack -- the scenario the legacy
NCHW OC-tiling strategy targets:

| Layer | IC  | OC  | KH | Working set | Tiled? |
|-------|-----|-----|----|-------------|--------|
| conv0 | 3   | 64  | 3  | 12 KB       | no     |
| conv1 | 64  | 64  | 3  | 246 KB      | no     |
| conv2 | 64  | 128 | 3  | 282 KB      | no     |
| conv3 | 128 | 128 | 3  | 256 KB      | yes (oc_tile=64) |

Only conv3 exceeds the 192 KB half-budget (384 KB / 2 for
double-buffering) and gets tiled.

Note: fused quantized conv2d stacks (requantize/bias/relu/clip/cast
fused into the same PrimFunc) are skipped by NCHW OC-tiling and instead
go through NHWC H-tiling (see "How It Works" above), which tiles on
output height rather than output channel.

### DMA Runtime API

```c
// Initiate async 1D DMA transfer (DDR <-> L2 SRAM)
int tvm_dsp_dma_copy(int queue_id, void* dst, const void* src,
                     int size, int bypass_cache);

// Wait until in-flight transfers on queue <= max_inflight
int tvm_dsp_dma_wait(int queue_id, int max_inflight);
```

The API is defined in `src/runtime/ti_dsp/dma/tvm_dsp_dma.h`.
Phase 1 provides synchronous (memcpy) implementations for both host
emulation and C7x hardware.

### File Inventory

| File | Role |
|------|------|
| `python/tvm/relax/transform/schedule_c7x_dma.py` | ScheduleC7xDMATiling pass |
| `python/tvm/tir/pipeline.py` | `_c7x_dma_tir_pipeline` with reordered DMA lowering |
| `python/tvm/tir/transform/lower_dma_to_extern.py` | `LowerDMAToExtern` pass (TIR intrinsics to call_extern) |
| `src/target/c_static/codegen_c_static_templates.h` | `kDSPHeaders` includes `dma/tvm_dsp_dma.h` |
| `src/runtime/ti_dsp/dma/tvm_dsp_dma.h` | DMA API header |
| `src/runtime/ti_dsp/dma/tvm_dsp_dma.c` | C7x hardware stub (memcpy, Phase 1) |
| `src/runtime/ti_dsp/dma/tvm_dsp_dma_host.c` | Host emulation stub (memcpy) |
| `src/tir/transforms/lower_async_dma.cc` | `LowerAsyncDMA` (upstream TVM, uses IdentifyMemCpy) |
| `src/tir/analysis/identify_memcpy.cc` | `IdentifyMemCpy` (upstream TVM, contiguity proof) |

## Architecture

### Directory Structure

```
src/target/c_static/
|-- codegen_c_static.h           # Core code generator class
|-- codegen_c_static.cc          # Main implementation (~1800 lines)
|-- codegen_c_static_dsp.h       # DSP extension class
|-- codegen_c_static_dsp.cc      # TI DSP pragmas, profiling
|-- codegen_c_static_wrapper.h   # Wrapper generator class
|-- codegen_c_static_wrapper.cc  # C++ wrapper generation
|-- codegen_c_static_templates.h # Code templates (headers, helpers)
|-- weight_packer.cc             # Weight/constant serialization to weights.bin
```

### Modular Components

| Class | Responsibility |
|-------|----------------|
| `CodeGenCStatic` | Core TIR-to-C code generation, inherits from CodeGenC |
| `DSPCodeGenExtension` | Emit TI DSP pragmas, headers, profiling infrastructure |
| `WrapperGenerator` | Generate C++ wrapper functions for exported functions |

### Code Generation Flow

1. **IR Analysis**: Examine TVM IR to detect function signatures and return types
2. **VM Builtin Emission**: `EmitAnylistVMBuiltinCall` converts compact anylist intrinsics to C++ API
3. **Register Allocation**: Calculate register file requirements per function
4. **Parameter Processing**: Handle serialization (binary or source format)
5. **DSP Optimization**: `DSPCodeGenExtension` emits TI-specific pragmas
6. **Wrapper Generation**: `WrapperGenerator` creates C++ wrapper functions
7. **Output**: Produce compilation units suitable for static binary generation

### Key Data Structures

```cpp
// Function metadata
struct CGFunctionInfo {
    int64_t max_register_index = -1;  // Maximum register usage
    int64_t num_args = 0;             // Number of input arguments
    bool returns_tuple = false;       // Multi-output detection
    int64_t num_outputs = 1;          // Number of outputs (N for tuple)
    uint64_t total_params = 0;        // Parameter count
    bool was_private = false;         // Visibility control
};

// DSP configuration
struct DSPConfig {
    bool enabled = false;             // Targeting TI DSP
    std::string mcpu;                 // Target CPU (c66x, c7x)
    bool profile_layers = false;      // Per-layer profiling
    bool tidl_runtime = false;        // Emit tidl_bridge_init_all() in cg_main_dsp
    std::vector<std::string> profiled_layer_names;
};
```

## C++ API for VM Operations

The c_static backend includes an optimized C++ API mode (`-use-cpp-api=1`) that
bypasses the FFI layer for VM operations, providing significant performance gains
on embedded targets.

### Performance Benefits

On CLISTA-DoA model (C66x DSP @ 450 MHz):

| Metric | Improvement |
|--------|-------------|
| Cycle reduction | 12% (62K cycles) |
| Memory reduction | 9% (1.8KB L2 peak) |
| Code size | 22% reduction |

### How It Works

The C++ API replaces verbose FFI dispatch sequences with direct function calls:

**Before (FFI mode)**:
```c
TVMBackendAnyListSetPackedArg(r, 2, stack_ffi_any, 0);
SetFFIAnyInt(&((stack_ffi_any)[1]), (long)0);
TVMBackendAnyListSetPackedArg(c, 5, stack_ffi_any, 2);
// ... 4 more lines
TVMBackendAnyListMoveFromPackedReturn(r, 3, stack_ffi_any, 4);
```

**After (C++ API mode)**:
```c
_r.SetNDArray(3, vm::AllocTensor(_r.GetStorage(2), 0, _c.GetShape(5), _c.GetDType(6)));
```

### Optimization Attributes

| Attribute | Default | Effect |
|-----------|---------|--------|
| `use-cpp-api` | `true` | Direct C++ calls (~12% faster) |
| `skip-runtime-checks` | `true` | Skip tensor validation (~5% faster) |

Both optimizations are **enabled by default** for all c_static targets.

## Building and Testing

### Build TVM with c_static Backend

```bash
# Configure build
mkdir -p build
cp cmake/cstatic_config.cmake build/config.cmake
cd build

# Build (requires LLVM 15+)
cmake -G Ninja ..
ninja

# Set up Python environment
cd ..
export TVM_HOME=$(pwd)
export PYTHONPATH=$TVM_HOME/python:$PYTHONPATH
```

### Build Static Runtime (Optional)

```bash
# Enable static runtime in config
# Set BUILD_STATIC_RUNTIME=ON in build/config.cmake
cd build
cmake -G Ninja ..
ninja
```

### Run Tests

```bash
cd $TVM_HOME
export PYTHONPATH=$TVM_HOME/python:$PYTHONPATH

# C static backend tests
pytest tests/cstatic/unit-tests/test_conv2d.py -v
pytest tests/cstatic/unit-tests/test_resnet.py -v
pytest tests/cstatic/unit-tests/test_matmul.py -v
pytest tests/cstatic/unit-tests/test_mlp.py -v

# DSP tests (host emulation)
pytest tests/ti-dsp-runtime/dsp-tests/ -v --dsp-mode=c66x_host

# DSP tests (C66x hardware)
pytest tests/ti-dsp-runtime/dsp-tests/ -v --dsp-mode=c66x
```

### Build DSP Runtime

**Host emulation**:
```bash
cd $TVM_HOME/src/runtime/ti_dsp
mkdir build && cd build
cmake ..
cmake --build .
```

**C66x hardware (AWRL6844)**:
```bash
cd $TVM_HOME/src/runtime/ti_dsp
mkdir build-c66x && cd build-c66x
cmake -DCMAKE_TOOLCHAIN_FILE=../cmake/toolchain-awrl6844.cmake ..
cmake --build .
```

**C7x hardware (J722S/AM67A)**:
```bash
cd $TVM_HOME/src/runtime/ti_dsp
mkdir build-c7x && cd build-c7x
cmake -DCMAKE_TOOLCHAIN_FILE=../cmake/toolchain-j722s-c7x.cmake ..
cmake --build .
```

### Verify Installation

```bash
# Run VM builtins test (host)
./src/runtime/ti_dsp/build/test_vm_builtins

# Run VM builtins test (C66x)
$TVM_HOME/src/runtime/ti_dsp/scripts/run_on_c66x.sh \
    src/runtime/ti_dsp/build-c66x/test_vm_builtins_c66x.out
```
