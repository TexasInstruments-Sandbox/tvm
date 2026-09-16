---
name: dsp-runtime
description: "TI DSP runtime library for TVM. Use when working on: runtime library code (model.h, Model API), build_runtime.sh targets (c66x_host/c7x_host/c7x/c66x), memory pools (L2 SRAM/DDR), bump-pointer allocator, NDArray/DLTensor handling, platform init, weight parsing, KV cache (resident/non-resident), or persistent sessions. NOT for firmware build/deploy (see firmware), test authoring (see testing), or build environment setup (see build)."
---

# TI DSP Runtime

Lightweight C++14 library (~100 KB) providing the execution environment for TVM-generated code on bare-metal/RTOS DSP targets. Replaces the full TVM runtime with a minimal implementation.

## Location

`src/runtime/ti_dsp/`

## Build Targets

```bash
cd src/runtime/ti_dsp
bash build_runtime.sh c66x_host  # PC host emulation (g++, TVM_DSP_TARGET=host)
bash build_runtime.sh c7x_host   # C7x host emulation (x86 + TI Host Emu)
bash build_runtime.sh c7x        # C7x cross-compilation (TI CGT)
bash build_runtime.sh c66x       # C66x cross-compilation
```

`build_runtime.sh` only accepts `c66x_host|c66x|c7x|c7x_host|clean|all` — there
is no bare `host` target on the CLI (the underlying CMake `TVM_DSP_TARGET=host`
default is what `c66x_host` builds).

Requires: `TI_CGT_C7000_PATH` for c7x/c7x_host targets.

## Memory Architecture (C7x / J722S)

| Pool | Backing | Size | Threshold | Use Case |
|------|---------|------|-----------|----------|
| L2 (Fast) | L2 SRAM | 128 KB | ≤32 KB tensors | Intermediate activations, hot data |
| DDR (Main) | DDR | 352 MB (as of 2026-07-27; has drifted before — verify against `linker_c75_freertos.cmd`, see `relax-c7x:firmware` Memory Layout) | >32 KB tensors | Constants, large activations, DLOAD modules |

Threshold is `TVM_DSP_L2_ALLOC_THRESHOLD` (`src/runtime/ti_dsp/core/config.h`), 32 KB.

Allocator: bump-pointer with size-segregated free-list for O(1) allocation. Memory pools backed by linker sections (`.tvm_l2_heap`, `.tvm_ddr_heap`) on hardware, `malloc()` on host emulation.

For full DDR memory map (DLOAD heap, KV region, I/O buffers), see `relax-c7x:firmware` Memory Layout.

## Model API

```cpp
#include "model.h"
tvm::dsp::Model model;
model.Load(weights_data, weights_size);        // parse weights.bin, resolve symbols
NDArray* output;
model.Infer(&input, &output);                  // single input/output
model.InferMulti(inputs, num_inputs, &outputs, &num_outputs);  // multi-output (up to 128)
```

Key characteristics:
- No exceptions/RTTI (TI CGT compatible)
- RAII lifecycle (platform init → weight parse → infer → cleanup)
- Zero-copy I/O (inputs/outputs reference caller-owned buffers)
- Cross-platform: same `model.h` API for host, C66x, C7x

## Runtime Symbols (117 exports)

DLOAD modules import these from the firmware (`dsp_syms` table in
`src/runtime/ti_dsp/firmware/c7x/dsp/src/dyn_loader.c`):
- **VM builtins**: tensor allocation, storage management, shape manipulation, tuple construction
- **Memory pools**: `tvm_dsp_alloc`, `tvm_dsp_free`, L2 bump allocator getters
- **Platform services**: cycle counting, printf redirection, DMA handles, cache ops
- **Kernels / TIDL / MMALIB**: hand-written `c7x_*` kernels, TIDL support hooks, MMALIB wrapper functions

## KV Cache Modes

| Mode | Description |
|------|-------------|
| `C7X_INFER_FLAG_KV_RESIDENT` | KV cache in fixed 12 MB DSP DDR region (0xDD400000). Only logits cross host↔DSP boundary. Default for SmolLM. |
| Legacy (non-resident) | KV cache on ARM, passed as 60 input tensors per call. DSP returns 60 updated outputs. |

Both modes use persistent sessions (decode ELF stays loaded between calls).

## Testing

```bash
bash build_runtime.sh c7x_host && pytest --rootdir=. dsp-tests/ -m quick --dsp-mode=c7x_host -v
```

For full test infrastructure (fixtures, profiling, debugging), see `relax-c7x:testing`.

## Key Files

| File | Purpose |
|------|---------|
| `src/runtime/ti_dsp/README.md` | Full runtime documentation |
| `src/runtime/ti_dsp/MODEL_API.md` | API reference |
| `src/runtime/ti_dsp/include/model.h` | Public API header |
| `src/runtime/ti_dsp/CMakeLists.txt` | Build system |
| `src/runtime/ti_dsp/build_runtime.sh` | Build script |
| `src/runtime/ti_dsp/platform/c7x/` | C7x platform-specific code |
| `src/runtime/ti_dsp/cmake/` | Toolchain files |

## Related Skills

- `relax-c7x:firmware` — Firmware that hosts this runtime (full DDR memory map)
- `relax-c7x:build` — Build targets and environment variables
- `relax-c7x:testing` — Pytest options, fixtures, and debugging
- `relax-c7x:cstatic` — Code generator that produces code linked against this runtime
