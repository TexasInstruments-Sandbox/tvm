---
name: dsp-ops
description: "Writing and modifying DSP operator implementations for C7x. Use when: implementing a new vectorized kernel (hand-written C7x asm/C), adding DMA tiling to a new op (ScheduleC7xDMATiling strategies), writing L2 prefetch logic, implementing quantization math (rescale fusion, QDQ parameter conversion), modifying the DMA runtime (tvm_dsp_dma_copy, tvm_l2_alloc), or adding a new call_extern kernel to the firmware. NOT for pipeline/pass configuration (see cstatic) or MMALIB wrapper integration (see mmalib-offload)."
---

# DSP Operator Implementation

Writing and optimizing operator kernels for C7x DSP execution.

## Operator Categories

### 1. Hand-Written Vectorized Kernels

Custom C7x implementations called via `call_extern` from TIR:

| Kernel | File | Description |
|--------|------|-------------|
| `c7x_dequantize_vecmatmul` | `src/runtime/ti_dsp/kernels/c7x_dequantize_vecmatmul.cpp` | SE + PROMOTE_4X int8→fp32 dequant fused with vector matmul |
| `c7x_sdpa_decode` | `src/runtime/ti_dsp/kernels/c7x_sdpa_decode.cpp` | Scaled dot-product attention for decode (seq_len=1) |

All hand-written kernels use the `c7x_` prefix (the older `tvm_`-prefixed,
`_wrappers`-suffixed names were standardized away); see
`src/runtime/ti_dsp/kernels/` for the full set (activation, pooling,
quantize, residual add, norm, concat).

These are linked into firmware and exported via DLOAD symbol table.

### 2. DMA-Tiled Operators

`ScheduleC7xDMATiling` (`python/tvm/relax/transform/schedule_c7x_dma.py`) tries three tiling strategies in order, applying the first one whose block pattern matches:

**1. NHWC H-tiling (preferred):**
- Splits output height loop (`conv2d_nhwc` blocks)
- Each tile: async DMA prefetch of input activation strip into L2
- Factor-based tile sizes → even division, no boundary conditionals
- Double-buffered: DMA overlaps compute via software pipeline
- Works with fused quantized kernels (per-channel ops independent of H)
- Weight caching: loaded once at batch level if fits in L2 budget

**2. NCHW OC-tiling (legacy fallback):**
- Splits output-channel loop (`conv2d_nchw` blocks)
- Copies full input + weight slice into L2
- Only safe for standalone conv2d (no fused post-conv blocks)

**3. N-tiling for matmul:**
- Splits the output-channel (N) loop of `dequantize_matmul_acc` blocks
- Prefetches weight tiles from DDR into L2 with double-buffering
- If the whole weight matrix fits in the L2 budget, it is cached once
  before the M loop instead (no software pipeline needed)

### 3. Quantization Math

**Rescale fusion** (`docs/dsp/quantized_model_compilation.md`):
Replace `int8 → float32 → round → int8` roundtrip between layers with direct `int32_accum → rescale → round → int8`:
```
combined_scale = scale_input * scale_weight / scale_output
combined_offset = zero_point_output
```

**QDQ parameter conversion** (compile-time, in MMALIB passes):
```
weight_sum[ch] = sum(weight_int8[ch, :, :, :])
zp_correction[ch] = -d_zp * weight_sum[ch]
bias_i32[ch] = round(float_bias / (d_scale * w_scale[ch])) + zp_correction
(scale_u8[ch], shift_u8[ch]) = best_uint8_approx(d_scale * w_scale[ch] / o_scale)
```

## DMA Runtime API

| Function | Purpose |
|----------|---------|
| `tvm_dsp_dma_copy(queue_id, dst, src, size, bypass_cache)` | Async 1D DMA transfer (DRU direct TR); `LowerDMAToExtern` lowers `tir.dma_copy` to this |
| `tvm_dsp_dma_wait(queue_id, max_inflight)` | Block until in-flight transfers on `queue_id` drop to `max_inflight`; lowered from `tir.dma_wait` |
| `tvm_l2_alloc(nbytes)` | Inline bump allocator into L2 SRAM (falls back to DDR via `TVMBackendAllocWorkspace` if exhausted) — emitted in generated code, `src/target/c_static_lib/codegen_c_static_lib_templates.h` |
| `tvm_l2_reset()` | Reset the L2 bump pointer to the base (no per-allocation free); called once per inference in the generated wrapper |

Wraps TI's `DmaUtilsAutoInc3d`. Source: `src/runtime/ti_dsp/dma/tvm_dsp_dma.c` (2D/3D transfers happen by issuing multiple 1D `tvm_dsp_dma_copy` calls from the generated copy loop, not via a single 2D primitive).

## TIR Lowering Pipeline (DMA passes)

After `FuseTIR`, the TIR lowering chain handles DMA:
```
InjectSoftwarePipeline → LowerOpaqueBlock → FlattenBuffer
  → LowerAsyncDMA → LowerDMAToExtern → NarrowDataType(32)
  → StorageRewrite → LowerL2SramAlloc → MakePackedAPI
```

For the full Relax compilation pipeline pass order, see `relax-c7x:cstatic`.

## Adding a New Kernel

1. Implement in `src/runtime/ti_dsp/kernels/` (C++) or `.asm`
2. Add `__declspec(dllexport)` for DLOAD visibility
3. Add export to `src/runtime/ti_dsp/firmware/c7x/dsp/src/dyn_loader.c`
4. Add link-time stub to `src/runtime/ti_dsp/dynmod/c7x_dynmod/dsp_syms.c`
5. Add `--import=<symbol>` to `c7x_dynmod.cmd`
6. Reference via `call_extern("my_kernel", ...)` in TIR
7. Rebuild firmware (see `relax-c7x:build` steps 4-5)

## Performance Reference

Single conv2d (int8, 64ch 56x56, 3x3, stride=1):

| Path | Cycles | Speedup |
|------|--------|---------|
| MMALIB + L2 DMA prefetch | 477K | 96x |
| MMALIB (DDR-resident) | 1.67M | 27x |
| TVM loop-based (scalar) | 45.2M | baseline |

## Key Files

| File | Purpose |
|------|---------|
| `python/tvm/relax/transform/schedule_c7x_dma.py` | DMA tiling pass |
| `src/runtime/ti_dsp/dma/tvm_dsp_dma.c` | DMA runtime |
| `src/runtime/ti_dsp/kernels/` | Hand-written kernels |
| `docs/dsp/c7x_dma.md` | Full DMA design/debug record |
| `docs/dsp/quantized_model_compilation.md` | Rescale fusion design |

## Related Skills

- `relax-c7x:cstatic` — Full pipeline pass order, target options
- `relax-c7x:mmalib-offload` — MMALIB wrapper functions and L2 prefetch injection
- `relax-c7x:firmware` — Adding new exports to firmware symbol table
- `c7x-optimizer:c7x-patterns` — Low-level vector kernel patterns (atomic ops, data-flow)
