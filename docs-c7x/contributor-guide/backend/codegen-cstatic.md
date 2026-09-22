# CodeGenCStaticLib Internals

The `CodeGenCStaticLib` class and its helpers handle C/C++ code generation for the
`c_static_lib` backend, with TI DSP-specific customizations for C7x deployment.

## Class Hierarchy

`CodeGenCStaticLib` is `final` — it has no subclasses. `DSPCodeGenExtension`
and `WrapperGenerator` are separate, stateless helper classes with static
methods; `CodeGenCStaticLib` delegates to them by composition, not
inheritance:

```
CodeGenC (upstream TVM, src/target/source/codegen_c.h)
    └── CodeGenCStaticLib final (src/target/c_static_lib/codegen_c_static_lib.{h,cc})
          ├── delegates to → DSPCodeGenExtension  (src/target/c_static_lib/codegen_c_static_lib_dsp.{h,cc})
          └── delegates to → WrapperGenerator     (src/target/c_static_lib/codegen_c_static_lib_wrapper.{h,cc})
```

## CodeGenCStaticLib

Core TIR-to-C translation. Inherits from `CodeGenC` and overrides
key methods for static library generation.

### Key Responsibilities

1. **Function emission** — Translates TIR `PrimFunc` to C functions
2. **VM builtin handling** — `EmitAnylistVMBuiltinCall` converts compact anylist
   intrinsics to C++ API calls
3. **Register allocation** — Calculates register file requirements per function
4. **Parameter processing** — Handles serialization (binary or source format)
5. **DSP extension delegation** — Calls `DSPCodeGenExtension` for TI-specific emission

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
    bool enabled = false;             // Targeting TI DSP (C66x or C7x)
    std::string mcpu;                 // Target CPU (e.g., "c66", "c7x")
    std::string device_name;          // Device identifier for CCXML generation
    bool profile_layers = false;      // Per-layer profiling
    bool debug_alloc = false;         // Diagnostic allocation tracing
    bool tidl_runtime = false;        // Emit tidl_bridge_init_all() in cg_main_dsp
    int layer_call_index = 0;         // Counter for profiled layer calls
    int alloc_storage_index = 0;      // Counter for traced AllocStorage calls
    std::vector<std::string> profiled_layer_names;
};
```

### Code Generation Flow

1. **IR Analysis** — Examine TVM IR to detect function signatures and return types
2. **VM Builtin Emission** — `EmitAnylistVMBuiltinCall` converts compact anylist intrinsics to C++ API
3. **Register Allocation** — Calculate register file requirements per function
4. **Parameter Processing** — Handle serialization (binary or source format)
5. **DSP Optimization** — `DSPCodeGenExtension` emits TI-specific pragmas
6. **Wrapper Generation** — `WrapperGenerator` creates C++ wrapper functions
7. **Output** — Produce compilation units suitable for static binary generation

### Weight Serialization

`WeightPacker` (`src/target/c_static_lib/weight_packer.cc`) serializes model
parameters (convolution filters, biases, batch norm statistics) into
`weights.bin` using TVM's binary parameter format.

For C7x DLOAD deployment, weights are embedded into the ELF via
`bin_to_asm.py` → `.rodata.weights` section with symbols:
- `_binary_weights_bin_start` — pointer to weights data
- `_binary_weights_bin_end` — end marker
- `_binary_weights_bin_size` — total size in bytes

## DSPCodeGenExtension

Emits TI DSP-specific code: compiler pragmas, headers, profiling infrastructure.

### TI Compiler Pragmas

```cpp
// Loop optimization hints for C7x VLIW/SIMD
#pragma MUST_ITERATE(1, , 1)   // Minimum trip count, known trip count, multiple
#pragma UNROLL(4)              // Unroll factor
```

Emitted automatically on loops meeting criteria (known trip count, no
complex control flow).

### Headers Included

Via `kDSPHeaders` in `codegen_c_static_lib_templates.h`:
- `c7x.h` / `c6x.h` — TI intrinsics
- `tvm/runtime/crt/...` — CRT headers
- `dma/tvm_dsp_dma.h` — DMA API (for DMA tiling)
- `mmalib_wrappers.h` — MMALIB wrapper prototypes (via `--preinclude`)

### Profiling Infrastructure

When `-profile-layers=1`:
- Inserts cycle counter reads (`__TSC`) around each kernel
- Emits `printf` with layer name, cycles, timestamp
- Output redirected to shared memory buffer (last 64 KB of output buffer)
- Host CLI reads and displays after inference

### C++ API Mode (`-use-cpp-api=1`)

Replaces verbose FFI dispatch with direct C++ calls:

**Before (FFI mode):**
```c
TVMBackendAnyListSetPackedArg(r, 2, stack_ffi_any, 0);
SetFFIAnyInt(&((stack_ffi_any)[1]), (long)0);
TVMBackendAnyListSetPackedArg(c, 5, stack_ffi_any, 2);
// ... 4 more lines
TVMBackendAnyListMoveFromPackedReturn(r, 3, stack_ffi_any, 4);
```

**After (C++ API mode):**
```c
_r.SetNDArray(3, vm::AllocTensor(_r.GetStorage(2), 0, _c.GetShape(5), _c.GetDType(6)));
```

~12% cycle reduction, ~22% code size reduction on C66x.

## WrapperGenerator

Generates `cg_main_dsp` entry point and I/O marshalling.

### cg_main_dsp

The firmware calls this function by name at load time (resolved via DLOAD
symbol table). Signature:

```c
TVM_DSP_EXPORT int32_t cg_main_dsp(
    void* args,       // Packed input tensors
    int32_t num_args, // Number of inputs
    void* outputs,    // Output tensor pointers
    int32_t num_outs  // Number of outputs
);
```

### Responsibilities

1. **Unpack inputs** — Convert packed args to `DLTensor*`
2. **Allocate intermediates** — Bump-pointer from L2/DDR pools
3. **Call kernels in dependency order** — Topological sort of PrimFuncs
4. **Pack outputs** — Return `DLTensor*` to caller
5. **Handle multi-output** — Tuple return via `outputs` array

## Symbol Resolution with DLOAD

The generated C code calls external symbols (MMALIB wrappers, DMA runtime,
VM builtins) via `tir.call_extern`. `CodeGenC` base class prints these as
literal C function calls:

```cpp
void CodeGenC::PrintCallExtern(Type ret_type, String global_symbol,
                                const Array<PrimExpr>& args, bool skip_first_arg,
                                std::ostream& os) {
  os << global_symbol << "(";
  for (size_t i = skip_first_arg; i < args.size(); ++i) {
    this->PrintExpr(args[i], os);
    if (i < args.size() - 1) os << ", ";
  }
  os << ")";
}
```

So `tir.call_extern("int32", "mmalib_conv2d_i8", a0, a1, ...)` becomes
literally `mmalib_conv2d_i8(a0, a1, ...)` in the generated C.

### Header Visibility: --preinclude

The generated file **does not** include wrapper headers directly.
Instead, the DSP-side build passes `--preinclude=mmalib_wrappers.h` to the
TI compiler for every translation unit (`src/runtime/ti_dsp/dynmod/CMakeLists.txt`).
Every generated `lib*.c` sees the wrapper prototypes without its own
`#include` or `extern` line.

### DLOAD Export Table

Firmware exports its symbol table in
`src/runtime/ti_dsp/firmware/c7x/dsp/src/dyn_loader.c` (see [Firmware
Architecture -- Dynamic Module Loading
(DLOAD)](../firmware/architecture.md#dynamic-module-loading-dload) for the
current count):

```c
/* TVM DSP Runtime - C backend API */
SYM(TVMBackendAllocWorkspace), SYM(TVMBackendFreeWorkspace),
SYM(TVMBackendAnyListSetPackedArg), SYM(TVMBackendAnyListMoveFromPackedReturn),
// ... grouped by comment into runtime API, kernels, TIDL-guarded, and
// MMALIB-guarded blocks

#ifdef USE_TI_MMALIB
SYM(mmalib_conv2d_i8), SYM(mmalib_conv2d_i8_sliced),
SYM(mmalib_conv2d_i8_grouped_loop), SYM(mmalib_conv2d_i16),
SYM(mmalib_matmul_i8), SYM(mmalib_matmul_i16),
SYM(mmalib_depthwise_conv2d_i8), SYM(mmalib_depthwise_conv2d_i16),
SYM(mmalib_matmul_bias_i8), SYM(mmalib_matmul_bias_i16),
#endif
```

The DLOAD module (`lib0.out`) is a separately-built relocatable ELF that
calls these as *unresolved* external symbols. TVM's DLOAD dynamic loader
resolves each against the firmware's export table at load time. No MMALIB
code is statically linked into the module — only header declarations (via
`--preinclude`) are needed to compile it.

## Two-Stage DLOAD Build

**Stage 1 — Pseudo-firmware (`dsp_syms.out`)**
- Builds stub `__declspec(dllexport)` declarations for all exported symbols
- Provides link-time definitions so TI linker resolves references in `lib0.c`
- Stubs are empty functions — exist only to satisfy the linker

**Stage 2 — DLOAD module (`lib0.out`)**
- Compiles `lib0.c` (and `lib1.c` if split) with `cl7x`
- Links against `dsp_syms.out` using `c7x_dynmod.cmd` linker script
- Key flags: `--dynamic=lib`, `--relocatable`, `--import=<symbol>`
- Embeds `.rodata.weights` with `weights.bin`

## Output File Structure

For large models, codegen splits output:
- `lib0.c` — `cg_main_dsp` entry point, `__vmtir__main` orchestration
- `lib1.c` — All kernel function bodies
- Enables parallel compilation (`make -j2`), ~1.6x wall time reduction

## Target Options Reference

| Option | Default | Purpose |
|--------|---------|---------|
| `-mcpu=c7x` | — | Target C7x DSP |
| `-use-cpp-api` | `true` | Direct C++ calls instead of FFI dispatch |
| `-skip-runtime-checks` | `true` | Skip tensor shape/type validation |
| `-profile-layers` | `false` | Per-layer cycle profiling via DSP printf |
| `-constants-byte-alignment` | `64` | Cache-line aligned constant arrays |
| `-params-in-binary` | `true` | Weights in binary file vs C source |
| `-mmalib` | `false` | Enable MMALIB kernel offload |
| `-tidl-kernels` | `true` | Use TIDL-backed kernels (max_pool) |

## Related Files

| File | Purpose |
|------|---------|
| `src/target/c_static_lib/codegen_c_static_lib.h/cc` | Core code generator |
| `src/target/c_static_lib/codegen_c_static_lib_dsp.h/cc` | DSP extension (pragmas, profiling) |
| `src/target/c_static_lib/codegen_c_static_lib_wrapper.h/cc` | Wrapper generator |
| `src/target/c_static_lib/codegen_c_static_lib_templates.h` | Code templates, headers |
| `src/target/c_static_lib/weight_packer.cc` | Weight serialization |
| `src/runtime/ti_dsp/scripts/bin_to_asm.py` | Weights embedder |
| `src/runtime/ti_dsp/dynmod/c7x_dynmod/` | DLOAD linker script + stubs |
| `src/runtime/ti_dsp/firmware/c7x/dsp/src/dyn_loader.c` | DLOAD symbol export table |
