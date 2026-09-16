---
name: cstatic
description: "TVM c_static_lib backend for C7x DSP. Use when working on: code generation (Relax→TIR→C), target options (-mcpu=c7x, -use-cpp-api, -profile-layers, -mmalib), CodeGenCStatic/DSPCodeGenExtension/WrapperGenerator classes, cg_main_dsp entry point, weight serialization (weights.bin, bin_to_asm.py), DLOAD ELF build (cl7x, lnk7x --dynamic=lib), TI CGT compiler integration, or the c_static_lib compilation pipeline pass order. NOT for runtime library code (see dsp-runtime), firmware deploy (see firmware), or operator kernel implementation (see dsp-ops)."
---

# TVM c_static_lib Backend for C7x

## End-to-End Compilation Flow

```
Relax IR → optimization passes → TIR → CodeGenCStatic → lib0.c + weights.bin
  → cl7x compile → lnk7x --dynamic=lib --relocatable → lib0.out (DLOAD ELF)
```

Entry point: `relax.build(mod, target="c_static_lib -mcpu=c7x")`

## Target Options

| Option | Default | Purpose |
|--------|---------|---------|
| `-mcpu=c7x` | — | Target C7x DSP |
| `-use-cpp-api` | `true` | Direct C++ calls instead of FFI dispatch (~12% faster, ~22% smaller) |
| `-skip-runtime-checks` | `true` | Skip tensor shape/type validation |
| `-profile-layers` | `false` | Per-layer cycle profiling via DSP printf |
| `-constants-byte-alignment` | `64` | Cache-line aligned constant arrays |
| `-mmalib` | `false` | Enable MMALIB integration (NCHW layout, MMA offload; see `relax-c7x:mmalib-offload` for pass details) |

## Code Generator Classes

| Class | Location | Role |
|-------|----------|------|
| `CodeGenCStatic` | `src/target/c_static_lib/codegen_c_static_lib.cc` | Core TIR→C translation (inherits CodeGenCHost) |
| `DSPCodeGenExtension` | `src/target/c_static_lib/codegen_c_static_lib_dsp.cc` | TI-specific pragmas, `#include <c7x.h>`, profiling |
| `WrapperGenerator` | `src/target/c_static_lib/codegen_c_static_lib_wrapper.cc` | `cg_main_dsp` entry point, I/O marshalling |

## C7x-Specific Customizations

When `-mcpu=c7x`:
- Emits `#include <c7x.h>` and TI compiler pragmas (`MUST_ITERATE`, `UNROLL`)
- Marks entry points with `__declspec(dllexport)` for DLOAD
- Generates `cg_main_dsp` wrapper matching firmware calling convention
- C++ API mode replaces multi-line FFI dispatch with direct function calls
- Serializes weights into `weights.bin` (TVM binary parameter format)

## Weight Embedding

1. TVM compiles parameters → `weights.bin`
2. `bin_to_asm.py` → `.rodata.weights` section in TI assembly
3. Assembler produces object with `_binary_weights_bin_{start,end,size}` symbols
4. Linker places `.rodata.weights` in ELF alongside code

## DLOAD ELF Build (Two-Stage Link)

**Stage 1:** Build `dsp_syms.out` — pseudo-firmware with stub `__declspec(dllexport)` declarations for all 127 runtime symbols (provides linker definitions without actual firmware).

**Stage 2:** Compile `lib0.c` + link against `dsp_syms.out`:
```
cl7x lib0.c + weights.asm
lnk7x --dynamic=lib --relocatable --import=<symbols> → lib0.out
```

Key linker script: `src/runtime/ti_dsp/dynmod/c7x_dynmod/c7x_dynmod.cmd`

## Compilation Pipeline (Pass Order)

Standard pipeline (`python/tvm/relax/backend/cpu_generic/pipeline.py`):

```python
passes = [
    # MMALIB passes (only with -mmalib=1)
    FuseMMALIBQDQConv2d(),
    FuseMMALIBQDQDwConv2d(),
    FuseMMALIBQDQFC(),
    FuseInt8ResidualAdd(),
    # Standard passes
    FuseQDQToInt8Conv2D(),
    EliminateQDQRoundTrip(),
    FuseDequantizeMatmul(),
    LegalizeOps(mmalib_map),       # when -mmalib=1
    AnnotateTIROpPattern(),
    FoldConstant(),
    FuseOps(),
    FuseTIR(),
    ScheduleC7xDMATiling(l2),      # DMA tiling: NHWC H-tiling / NCHW OC-tiling
                                   # (conv2d) + N-tiling (matmul)
]
```

`InjectMMALIBDMA()` is *not* in this Relax-level list — it runs later, in the
separate TIR-level pipeline (`_c7x_dma_tir_pipeline()` in
`python/tvm/tir/pipeline.py`), inserted after `StorageRewrite()` and before
`LowerL2SramAlloc()`, when `-mmalib=1`.

## Key Files

| File | Purpose |
|------|---------|
| `src/target/c_static_lib/` | Code generator source |
| `src/target/c_static_lib/README.md` | Detailed architecture doc |
| `python/tvm/relax/backend/cpu_generic/pipeline.py` | Pipeline wiring |
| `src/runtime/ti_dsp/scripts/bin_to_asm.py` | Weight embedder |
| `src/runtime/ti_dsp/dynmod/` | DLOAD linker scripts + stubs |
| `tests/ti-dsp-runtime/dsp-tests/` | Integration tests |

## Testing

```bash
pytest tests/cstatic/unit-tests/ -v                                    # codegen unit tests
pytest --rootdir=. dsp-tests/ -m quick --dsp-mode=c7x_host -v         # quick regression
```

For full test options, fixtures, and profiling, see `relax-c7x:testing`.

## Related Skills

- `relax-c7x:mmalib-offload` — MMALIB pass details (positions 1-4 in pipeline)
- `relax-c7x:dsp-ops` — Operator kernel implementation and DMA tiling
- `relax-c7x:build` — Building TVM core and DLOAD modules
- `relax-c7x:testing` — Test authoring and profiling
