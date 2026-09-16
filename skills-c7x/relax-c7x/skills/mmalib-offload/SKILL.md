---
name: mmalib-offload
description: "MMALIB direct integration for TVM c_static_lib backend. Use when working on: MMALIB QDQ partitioning passes (int8: FuseMMALIBQDQConv2d/DwConv2d/FC/FuseInt8ResidualAdd; int16: FuseMMALIBQDQConv2dI16/DwConv2dI16/FCI16/FuseInt16ResidualAdd), pass registry (ti_mmalib_passes.py, get_mmalib_qdq_passes), MMALIB wrapper functions (8 total: mmalib_conv2d_i8/i16, mmalib_matmul_i8/i16, mmalib_depthwise_conv2d_i8/i16, mmalib_matmul_bias_i8/i16), int16 LLM offload (LegalizeMLPToMMALIBInt16), L2 DMA prefetch injection (InjectMMALIBDMA), MMA vector width portability, compile-time QDQ parameter conversion (scale_u8/shift_u8/bias_i32/bias_i64), residual add fusion (ti_residual_add.py), or MMALIB kernel API. NOT for TIDL subgraph offload (see tidl-offload) or pipeline pass ordering (see cstatic)."
---

# MMALIB Direct Integration

TVM's c_static_lib backend calls MMALIB functions directly for compute-intensive ops (matmul, conv2d) on the C7x MMA accelerator, bypassing TIDL. TVM controls the schedule; MMALIB handles MMA hardware programming.

Target string: `c_static_lib -mcpu=c7x -mmalib=1`

## When to Use MMALIB (vs TIDL)

Use MMALIB when you need fine-grained control over scheduling, support custom ops alongside MMA acceleration, or need int16 precision for LLMs. Handles conv2d, matmul, depthwise individually — TVM controls the overall execution order. Does not require TIDL tools or calibration data (for int16 path).

## Architecture

```
Relax IR
  ├─ Int8 QDQ path: FuseMMALIBQDQ{Conv2d,DwConv2d,FC} + FuseInt8ResidualAdd
  │    (runs BEFORE FuseQDQToInt8Conv2D)
  │    → Matches PT2E pattern: dequant(data)→op(_, dequant(w))→[bias]→[relu]→quantize
  │    → Extracts quant params, folds zero-point and bias into int32 at compile time
  │    → TIR with call_extern("mmalib_*_i8", ..., bias_i32, scale_u8, shift_u8)
  │
  ├─ Int16 QDQ path: FuseMMALIBQDQ{Conv2dI16,DwConv2dI16,FCI16} + FuseInt16ResidualAdd
  │    (runs AFTER int8 QDQ passes, BEFORE FuseQDQToInt8Conv2D)
  │    → Same PT2E pattern, dtype=int16; d_zp and o_zp must be 0 (symmetric only)
  │    → Bias int64 (wider accumulator); same uint8 scale_u8/shift_u8 requant
  │    → TIR with call_extern("mmalib_*_i16", ..., bias_i64, scale_u8, shift_u8)
  │
  ├─ Int16 legalize path: LegalizeMLPToMMALIBInt16
  │    → Dynamic per-tensor activation quantization at runtime (LLM weight-only path)
  │    → TIR with call_extern("mmalib_matmul_i16")
  │
  ▼  CodeGenCStatic → link against firmware MMALIB symbols
```

All MMALIB passes are registered in `ti_mmalib_passes.py` via `get_mmalib_qdq_passes()`.
`pipeline.py` calls that function — never edit pass lists in `pipeline.py` directly.

## Supported Operations

| Op | Dtype | Path | Constraints |
|----|-------|------|-------------|
| conv2d (QDQ) | int8 | QDQ | N=1, symmetric stride, dilation=1, groups=1, C_out%64==0 |
| conv2d (QDQ) | int16 | QDQ | N=1, symmetric stride, dilation=1, groups=1, C_out%32==0; d_zp=o_zp=0 |
| depthwise conv2d (QDQ) | int8 | QDQ | N=1, groups=C_in, kernel 3x3/5x5/7x7, stride 1-2 |
| depthwise conv2d (QDQ) | int16 | QDQ | N=1, groups=C_in, **3x3 only** (MMALIB-882), stride 1-2; d_zp=o_zp=0 |
| matmul_bias / FC (QDQ) | int8 | QDQ | K%64==0, N%64==0, weight [N,K] transposed internally |
| matmul_bias / FC (QDQ) | int16 | QDQ | K%32==0, N%32==0; d_zp=o_zp=0; bias int64 |
| residual add (QDQ) | int8 | QDQ | Both add(x,skip) and add(skip,x) operand orders; with/without relu |
| residual add (QDQ) | int16 | QDQ | Same as int8; d_zp=skip_zp=o_zp=0 required |
| matmul | int16 | legalize | 2D, dims multiples of 32 (int16) or 64 (int8) |
| conv2d | int16 | legalize | N=1, symmetric stride, dilation=1, groups=1, C_out%32==0 |

## Int8 QDQ Pattern Matching

`FuseMMALIBQDQConv2d` matches the PT2E pattern BEFORE `FuseQDQToInt8Conv2D`:

```
dequantize(data_int8, d_scale, d_zp)
  → conv2d(float, dequantize(weight_int8, w_scale, w_zp=0))
  → [add(float_bias)] → [relu]
  → quantize(output, o_scale, o_zp)
```

4 variants (priority order): conv2d_i8_qdq_bias_relu, _bias, _relu, plain.

### Compile-Time Parameter Conversion

```
weight_sum[ch] = sum(weight_int8[ch, :, :, :])
zp_correction[ch] = -d_zp * weight_sum[ch]
bias_i32[ch] = round(float_bias / (d_scale * w_scale[ch])) + zp_correction
combined_rescale[ch] = d_scale * w_scale[ch] / o_scale
(scale_u8[ch], shift_u8[ch]) = best_uint8_approx(combined_rescale[ch])
```

## FC / Matmul_Bias QDQ (`FuseMMALIBQDQFC`)

Four DFPatterns matched in priority order (longest/most-specific first):

| Composite name | Pattern |
|---|---|
| `mmalib.fc_i8_qdq_reshape_bias` | dequant → reshape → matmul → reshape → add → quantize |
| `mmalib.fc_i8_qdq_reshape` | dequant → reshape → matmul → reshape → quantize |
| `mmalib.fc_i8_qdq_bias` | dequant → matmul → add → quantize |
| `mmalib.fc_i8_qdq` | dequant → matmul → quantize |

Reshape variants handle 3D inputs from `aten.linear` which decomposes as
reshape + matmul + reshape before quantize.

### Eligibility (`_check_mmalib_qdq_fc`)

Rejected if any of:

- `w_zp` is not a zero constant — MMALIB requires zero-point-free weights
- weight dtype is not `int8`, or data is a constant (weights-only path not supported)
- weight shape is not 2D `[N_out, K]`
- `K % 64 != 0` or `N_out % 64 != 0` (MMA tile alignment)
- `permute_dims` is not a standard last-two-dim transpose
- bias (if present) is not a compile-time constant

### Compile-Time Parameter Folding

The data zero-point correction is absorbed directly into the bias tensor,
so nothing is computed at runtime:

```
weight_sum[n]    = sum(weight_int8[n, :])                        # [N_out]
zp_correction[n] = -d_zp * weight_sum[n]
bias_accum[n]    = round(float_bias[n] / (d_scale * w_scale[n]))
bias_i32[n]      = bias_accum[n] + zp_correction[n]             # int32

# output zero-point also absorbed into bias (if o_zp != 0)
bias_i32[n]     += round(o_zp / combined_rescale[n])

combined_rescale[n]          = d_scale * w_scale[n] / o_scale
(scale_u8[n], shift_u8[n])   = best_uint8_approx(combined_rescale[n])
```

### Weight Layout

Weights are passed in natural `[N_out, K]` layout without reordering.
`mmalib_matmul_bias_i8` uses `bTranspose=1` internally, so no runtime
weight reorder step is needed (unlike depthwise conv which reorders at
runtime).

## L2 DMA Prefetch (`InjectMMALIBDMA`)

`InjectMMALIBDMA` is a TIR PrimFunc pass that runs after `StorageRewrite`
and before `LowerL2SramAlloc`. It finds each MMALIB `call_extern` and
wraps it with L2 `Allocate` nodes, async `tvm_dsp_dma_copy` transfers,
and a `tvm_dsp_dma_wait` before the kernel call.

Supported kernels: `mmalib_conv2d_i8`, `mmalib_conv2d_i8_grouped_loop`,
`mmalib_conv2d_i16`, `mmalib_depthwise_conv2d_i8/i16`, and
`mmalib_matmul_bias_i8/i16` — i.e. 6 of the 8 wrapper entry points (the
plain `mmalib_matmul_i8/i16`, which have no bias/requant args, are not
handled) plus the `_grouped_loop` conv2d variant. Residual add
(`c7x_int8_residual_add_relu` / `c7x_int16_residual_add_relu`) runs on the
C7x scalar pipeline, not the MMA, and is **not** covered by this pass.
Guard bytes: `pad_top * W_in * elem_bytes` (read from args[15] for conv2d
i8/i16/grouped_loop; falls back to 128 when pad_top == 0).

### Injected TIR structure

```
Allocate l2_guard[int8, guard_bytes]        # SE backward-prefetch guard
  Allocate l2_input[int8, input_bytes]
    Allocate l2_weight[int8, weight_bytes]  # omitted if weights don't fit
      Evaluate(tvm_dsp_dma_copy(0, l2_input,  ddr_input,  input_bytes,  0))
      Evaluate(tvm_dsp_dma_copy(0, l2_weight, ddr_weight, weight_bytes, 0))
      Evaluate(tvm_dsp_dma_wait(0, 0))
      Evaluate(mmalib_*(l2_input, l2_weight, ...))
```

The guard allocation (`max(pad_top * W * elem_bytes, 128)` bytes) bumps
the L2 bump pointer so the input buffer does not start at address 0,
preventing streaming-engine backward-prefetch underflow. `LowerL2SramAlloc`
later converts each `Allocate(scope="global.l2sram")` to a
`tvm_l2_alloc(nbytes)` call.

### FC DMA: M-dependent firing

For FC the pass maps `[M, K, N]` matmul dims onto the shared conv-shaped
slots: `c_in=K`, `h_in=M`, `w_in=1`, `c_out=N`, `kh=kw=1`.

```
input_bytes  = K * M   (activations)
weight_bytes = N * K   (weights)
```

DMA is skipped entirely when `input_bytes > l2_budget` (default 384 KB):

| Scenario | M | K | N | input KB | weight KB | DMA fires? |
|---|---|---|---|---|---|---|
| Decode (seq=1) | 1 | 256 | 512 | 0.25 | 128 | Yes — both |
| Prefill seq=64 | 64 | 256 | 512 | 16 | 128 | Yes — both |
| Prefill seq=512 | 512 | 256 | 512 | 128 | 128 | Yes — both |
| Prefill seq=2048 | 2048 | 256 | 512 | 512 | 128 | No (>384 KB) |

Weights are only prefetched when `input_bytes + weight_bytes <= l2_budget`.
Large prefill batches silently fall back to DDR access with no DMA.

## Int16 MMALIB (LLM Path)

For LLMs (SmolLM-135M), int8 requantization error (24-61 per layer) destroys accuracy across 30 layers. Int16 reduces per-layer error to ±1 LSB but **still fails at depth**: 30 layers (211 quantize/dequant cycles) compounds to 4.7% top-1 accuracy (vs 93.8% float baseline). The fundamental limitation is that MMA hardware only accepts integer input — every layer requires a `float32→int16→int64_accum→>>shift→int16→float32` roundtrip, and repeated truncation to 16-bit accumulates.

Current status: int16 MMALIB provides 11x speedup over scalar but is **not accurate enough** for production LLM inference. The current production path uses weight-only INT8 with scalar `FuseDequantizeMatmul` (float activations × int8 weights).

Implementation:
- `MMALIB_LINALG_matrixMatrixMultiply` (non-bias) with int16 inputs
- Global shift parameter prevents overflow (computed from weight L1-norms)
- No lossy per-channel uint8 scale/shift requantization
- Output dequantized: `out_float = out_i16 * (2^shift) * x_scale * w_scale`

Pass: `LegalizeMLPToMMALIBInt16` (`python/tvm/relax/transform/ti_mmalib_i16_fc.py`)

## Pipeline Position

MMALIB passes run BEFORE `FuseQDQToInt8Conv2D` to capture the intact PT2E QDQ graph.
`get_mmalib_qdq_passes()` returns them in the correct order — add new passes there only.
For the full pipeline, see `relax-c7x:cstatic`.

## MMALIB Wrapper Functions (8 exports)

| Symbol | Purpose |
|--------|---------|
| `mmalib_conv2d_i8` | Int8 conv2d with per-channel requant |
| `mmalib_conv2d_i16` | Int16 conv2d with per-channel bias(int64)/scale/shift |
| `mmalib_matmul_i8` | Int8 matmul (no bias/requant) |
| `mmalib_matmul_i16` | Int16 matmul (no requant; used by LLM weight-only path) |
| `mmalib_depthwise_conv2d_i8` | Int8 depthwise with runtime weight reorder |
| `mmalib_depthwise_conv2d_i16` | Int16 depthwise; 3×3 only (MMALIB-882) |
| `mmalib_matmul_bias_i8` | Int8 FC with bias, per-channel scale/shift |
| `mmalib_matmul_bias_i16` | Int16 FC with bias(int64), per-channel scale/shift |

Located: `src/runtime/ti_dsp/mmalib/mmalib_wrappers.{h,cpp}`
Exported in firmware via DLOAD symbol table (dyn_loader.c).
Also in `dynmod/c7x_dynmod/dsp_syms.c` (link-time stubs) and `c7x_dynmod.cmd` (--import).

**int16 wrapper notes:**
- All NULL bias/scale/shift args apply identity values (zero bias, scale=1, shift=0)
- `mmalib_conv2d_i16`: `inChOffset` and `groupOffset` in InitArgs are in **elements** (confirmed from MMALIB `_d.c`); `blockFeaturePitch` is in **bytes**
- `mmalib_depthwise_conv2d_i16`: `src2_addr.stride_y = 1` (element count, not bytes — confirmed from real-world TIDL usage)
- `te.extern(name=...)` must NOT match the call_extern function name — use `name="mmalib_conv2d"` not `"mmalib_conv2d_i16"` to avoid C variable shadowing the function declaration

## MMA Vector Width

C7x MMA vector width varies by device:
- C7120 (AM67A hardware): 64 bytes
- C7504 (host emulation): 32 bytes

Use `MMALIB_MMA_SIZE_8_BIT` from headers, never hardcode. Affects `outPairOffset`, `inPairOffset`, `columnOffset`.

## MMALIB Kernel API Reference

No standalone API reference doc exists in this repo; see the inline
init/exec pattern (bufParams, InitArgs, data types) in
`src/runtime/ti_dsp/mmalib/mmalib_wrappers.cpp`.

Key kernels used:
- `MMALIB_CNN_convolveBias_row_ixX_ixX_oxX` — Dense conv2d
- `MMALIB_CNN_convolve_col_smallNo_ixX_ixX_oxX` — Depthwise conv2d
- `MMALIB_LINALG_matrixMatrixMultiply_ixX_ixX_oxX` — Matmul (int16)
- `MMALIB_LINALG_matrixMatrixMultiplyBias_ixX_ixX_oxX` — Matmul + bias (int8 FC)

## Performance (AM67A C7x @ 1 GHz)

| Model | MMALIB Layers | Time | vs Scalar |
|-------|--------------|------|-----------|
| ResNet-18 | 20 conv2d + 8 res_add + 1 FC | 100ms | 47x |
| ShuffleNet V2 | 1 conv + 19 dwconv + 1 FC | 315ms | — |
| MobileNet V2 | 19 conv + 17 dwconv + 10 res_add + 1 FC | 2.4s | — |

L2 DMA prefetch: 3.5x per conv2d layer (477K vs 1.67M cycles).

## Data Layout

All ops use NCHW when `-mmalib=1`. Pipeline skips `ConvertLayoutNHWC` pass. Layout conversion at network I/O boundaries only.

## Testing

```bash
# MMALIB kernel unit tests (int8 + int16)
pytest --rootdir=. mmalib-tests/ -m quick --dsp-mode=c7x_host -v

# PT2E quantizer + int16 pipeline (pure Python, no DSP)
pytest --rootdir=. pt2e-tests/test_c7x_mma_quantizer_i16.py -m quick -v

# PT2E e2e (int8 + int16 on DSP)
pytest --rootdir=. pt2e-tests/ -m quick --dsp-mode=c7x_host -v

# Full model
pytest --rootdir=. quantized/test_quantized_resnet.py -v \
    --dsp-mode=c7x_dload --mmalib --profile
```

For fixtures, markers, and debugging, see `relax-c7x:testing`.

## Key Files

| File | Purpose |
|------|---------|
| `python/tvm/relax/transform/ti_mmalib_passes.py` | **Central registry**: `get_mmalib_qdq_passes()` |
| `python/tvm/relax/transform/ti_mmalib_qdq_fusion.py` | QDQ conv2d pattern + lowering (int8) |
| `python/tvm/relax/transform/ti_mmalib_qdq_i16_conv.py` | QDQ conv2d pattern + lowering (int16) |
| `python/tvm/relax/transform/ti_mmalib_qdq_dwconv.py` | QDQ depthwise (int8); `_check_dwconv2d_geometry` shared helper |
| `python/tvm/relax/transform/ti_mmalib_qdq_i16_dwconv.py` | QDQ depthwise (int16) |
| `python/tvm/relax/transform/ti_mmalib_qdq_fc.py` | QDQ FC/matmul_bias (int8 + int16) |
| `python/tvm/relax/transform/ti_mmalib_i16_fc.py` | Int16 matmul (LLM weight-only path) |
| `python/tvm/relax/transform/ti_mmalib_inject_dma.py` | L2 DMA prefetch (6 of 8 MMALIB kernels + `_grouped_loop`; excludes plain matmul and residual add) |
| `python/tvm/relax/transform/ti_residual_add.py` | Residual add fusion — int8 + int16, both operand orders |
| `python/tvm/relax/transform/ti_mmalib_constants.py` | MMA_SIZE constants |
| `src/runtime/ti_dsp/mmalib/mmalib_wrappers.{h,cpp}` | C wrappers (8 entry points) |
| `src/runtime/ti_dsp/kernels/c7x_residual_add.{h,cpp}` | Fixed-point residual add (int8 + int16), scalar pipeline |
| `docs/dsp/mmalib_subgraph_offloading.md` | Full integration design doc |
| `docs/dsp/smollm_mmalib_fixes.md` | SmolLM int16 offload accuracy analysis |

## Limitations

- Weight quantization must be symmetric (w_zp=0)
- Int16 QDQ: activation quantization is symmetric only (d_zp=0, o_zp=0 required); asymmetric patterns are rejected and fall through to float
- Int16 depthwise: 3×3 only (MMALIB-882 blocks 5×5/7×7 for `convolve_col_smallNo_highPrecision`)
- Int8 requantization error scales with sqrt(K) — unusable for deep networks (30+ layers)
- L2 DMA guard: `pad_top * W_in * elem_bytes` bytes; 128-byte fallback when pad_top=0
- Weights exceeding L2 budget remain in DDR, except `mmalib_conv2d_i8`, which OC-tiles oversized weights into L2 chunk-by-chunk via `mmalib_conv2d_i8_sliced`
- Guard with `isinstance(arg, tir.IntImm)` before reading `.value` for symbolic pad_top
- Depthwise: 64-byte aligned output row stride; wrapper pads and compacts if W_out not aligned

## Related Skills

- `relax-c7x:cstatic` — Full pipeline pass order, `-mmalib` target option
- `relax-c7x:tidl-offload` — Alternative offload strategy (mutually exclusive)
- `relax-c7x:dsp-ops` — DMA tiling and L2 prefetch implementation details
- `relax-c7x:firmware` — Firmware rebuild when MMALIB wrappers change
- `relax-c7x:model-workflow` — Choosing MMALIB vs TIDL for a new model
