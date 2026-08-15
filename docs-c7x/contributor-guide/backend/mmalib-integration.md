# MMALIB Integration

C wrappers that let TVM's `c_static` backend offload compute-intensive
Relax ops to the C7x MMA (Matrix Multiply Accelerator) coprocessor on
AM67A (J722S) via TI's MMALIB library, plus the glue that lets the
firmware and codegen link MMALIB without also requiring TIDL. Located
at `src/runtime/ti_dsp/mmalib/`.

Target string: `c_static -mcpu=c7x -mmalib=1`

## Files

| File | Purpose |
|------|---------|
| `mmalib_wrappers.{h,cpp}` | C wrappers for the 8 MMALIB kernels (conv2d/depthwise-conv2d/matmul/matmul_bias × int8/int16), linked into the C7x firmware and exported via DLOAD |
| `tidl_maxpool_wrapper.{h,cpp}` | `max_pool2d` wrapper — TIDL-backed (`c7x_int8_max_pool_tidl`) when the firmware links TIDL, with a scalar fallback (`c7x_int8_max_pool`) for `--tidl OFF` builds (e.g. BeagleY-AI) |

## Why MMALIB

A single 64ch 56×56 int8 conv2d layer takes ~45M cycles on the C7x scalar
pipeline. The same layer takes ~1.67M cycles via the MMA coprocessor
through MMALIB — a 27× speedup — and drops to ~477K cycles when input
data is staged into L2 SRAM via DMA before the MMA call (96× speedup).

## Compiler-Side: QDQ Offload Pipeline

TVM generates a single C source file; each eligible Relax op is replaced
by a `call_extern` to an MMALIB wrapper, with quantization scale/shift/
bias folded in at compile time. There are three code paths, distinguished
by quantization scheme and dtype:

```
Relax IR (R.matmul, R.nn.conv2d)
  │
  ├─ Int8 QDQ path: FuseMMALIBQDQ{Conv2d,DwConv2d,FC} + FuseInt8ResidualAdd
  │    (runs BEFORE FuseQDQToInt8Conv2D)
  │    → Matches PT2E pattern: dequant(data)→op(_, dequant(w))→[bias]→[relu]→quantize
  │    → All quant params folded into integer bias/scale/shift at compile time
  │    → TIR: call_extern("mmalib_conv2d_i8", ..., bias_i32, scale_u8, shift_u8)
  │
  ├─ Int16 QDQ path: FuseMMALIBQDQ{Conv2dI16,DwConv2dI16,FCI16} + FuseInt16ResidualAdd
  │    (runs AFTER int8 QDQ passes, BEFORE FuseQDQToInt8Conv2D)
  │    → Same PT2E pattern structure; d_zp and o_zp must be 0 (symmetric only)
  │    → Bias int64 (wider accumulator); same uint8 scale_u8/shift_u8 requant
  │    → TIR: call_extern("mmalib_conv2d_i16" / "mmalib_matmul_bias_i16" / ...)
  │
  ├─ Int16 legalize path: LegalizeOps(customize_legalize_map)
  │    → Float32 ops with no quantization → call_extern("mmalib_conv2d_i16" / "mmalib_matmul_i16")
  │    → Used for weight-only-quantized LLM inference (LegalizeMLPToMMALIBInt16)
  │
  ▼  CodeGenCStatic
Generated C code calling wrapper functions
  │
  ▼  Link against MMALIB
Executable (c7x_host or c7x_dload)
```

All ops use NCHW (planar channel-first) when MMALIB is enabled — the
pipeline skips `ConvertLayoutNHWC` when `-mmalib=1` is set, since
MMALIB's conv kernel (`convolveBias_row`) expects each input channel as
a contiguous H×W block, which maps directly to NCHW storage. Layout
conversion happens at network I/O boundaries only.

The QDQ fusion passes must run **before** `FuseQDQToInt8Conv2D` and
`EliminateQDQRoundTrip` — those elimination passes remove intermediate
`quantize→dequantize` pairs between consecutive quantized layers,
destroying the pattern the MMALIB passes need to match. Running first
means the passes see the intact PT2E graph where every layer boundary
has explicit QDQ nodes. All pass instantiation is centralized in
`get_mmalib_qdq_passes()` in `ti_mmalib_passes.py`; `pipeline.py` calls
this function and is never edited when new MMALIB passes are added.

### Entry point: C7xMMAQuantizer

```python
from tvm.relax.frontend.torch import C7xMMAQuantizer, from_exported_program

quantizer = C7xMMAQuantizer(dtype="int8")   # or "int16"
prepared  = prepare_pt2e(model, quantizer)
# ... calibrate ...
quantized = convert_pt2e(prepared)
ep  = torch.export.export(quantized, example_inputs)
mod = from_exported_program(ep, keep_params_as_input=True)
# → mod now contains the QDQ pattern that the MMALIB passes will match
```

All QDQ fusion passes match this PT2E pattern (4 variants per op:
with/without bias × with/without relu):

```
dequantize(data, d_scale, d_zp)
  → op(float, dequantize(weight, w_scale, w_zp=0))
  → [add(float_bias)] → [relu]
  → quantize(output, o_scale, o_zp)
```

The entire chain is replaced by a single `call_extern` with compile-time
integer parameters — scale, shift, and bias are all computed once during
TVM compilation, nothing at inference time.

## Supported Operations

| Op | Dtype | Path | Constraints |
|----|-------|------|-------------|
| matmul | int8, int16 | legalize | 2D, dims multiples of 64 (i8) or 32 (i16) |
| conv2d | int16 | legalize | N=1, symmetric stride, dilation=1, groups=1, C_out%32==0 |
| conv2d (QDQ fused) | int8 | QDQ | N=1, symmetric stride, dilation=1, groups=1, C_out%64==0 |
| conv2d (QDQ fused) | int16 | QDQ | N=1, symmetric stride, dilation=1, groups=1, C_out%32==0 |
| depthwise conv2d (QDQ fused) | int8 | QDQ | N=1, groups=C_in, kernel 3x3/5x5/7x7, stride 1-2, dilation=1 |
| depthwise conv2d (QDQ fused) | int16 | QDQ | N=1, groups=C_in, **3×3 only** (MMALIB-882), stride 1-2, dilation=1 |
| matmul_bias (FC, QDQ fused) | int8 | QDQ | K%64==0, N%64==0, weight [N,K] transposed internally |
| matmul_bias (FC, QDQ fused) | int16 | QDQ | K%32==0, N%32==0, d_zp/o_zp must be 0, bias int64 |
| residual add (QDQ fused) | int8 | QDQ | Both add(x,skip) and add(skip,x) operand orders; with/without relu |
| residual add (QDQ fused) | int16 | QDQ | Same as int8; d_zp/skip_zp/o_zp must all be 0 (symmetric only) |

Non-eligible ops fall through to the default loop-based legalization.

## Firmware: Decoupling MMALIB from TIDL

BeagleY-AI firmware is built `--tidl OFF --mmalib ON`:

- **`firmware/c7x/dsp/CMakeLists.txt`** has two independent CMake options,
  `USE_TIDL_RUNTIME` and `USE_TI_MMALIB`. `--tidl <ON|OFF>` and
  `--mmalib <ON|OFF>` on `build.sh` forward to them. `--tidl ON` still
  forces `--mmalib ON` (TIDL's own algo lib has unresolved
  `MMALIB_CNN_*`/`MMALIB_LINALG_*` symbols at link time), but the reverse
  isn't true: `--tidl OFF --mmalib ON` links MMALIB without TIDL.
- **`dyn_loader.c`** guards TIDL-only symbols (`c7x_int8_max_pool_tidl`,
  `TIDL_VISION_FXNS`) behind `#ifdef USE_TIDL_RUNTIME` and MMALIB symbols
  behind `#ifdef USE_TI_MMALIB`, so a no-TIDL firmware's export table
  never references a TIDL symbol that isn't linked.
- **Codegen**: `FuseQDQToTIDLMaxPool` — the one pass that unconditionally
  emitted a TIDL-backed kernel even outside TIDL-offload paths — now reads
  a `tidl-kernels` target attr (default `true`, preserving prior
  behavior). Passing `-tidl-kernels=0` in the `c_static` target string
  makes it emit `call_extern("c7x_int8_max_pool", ...)` (the scalar
  fallback in `tidl_maxpool_wrapper.cpp`) instead of
  `c7x_int8_max_pool_tidl`, so a model with `max_pool2d` still links
  against a no-TIDL firmware. **Every BeagleY-AI compile must pass
  `-tidl-kernels=0` explicitly** — codegen has no way to detect what the
  firmware actually linked, so without this flag `FuseQDQToTIDLMaxPool`
  emits a call to a symbol that doesn't exist there, which fails only at
  DLOAD load time on the board, not at compile time.

## Key Files (Full Pipeline)

| File | Purpose |
|------|---------|
| `python/tvm/relax/transform/ti_mmalib_passes.py` | **Central registry**: `get_mmalib_qdq_passes()`, all MMALIB pass config |
| `python/tvm/relax/transform/ti_mmalib_qdq_fusion.py` | QDQ conv2d pattern matching + lowering (int8) |
| `python/tvm/relax/transform/ti_mmalib_qdq_i16_conv.py` | QDQ conv2d pattern matching + lowering (int16) |
| `python/tvm/relax/transform/ti_mmalib_qdq_dwconv.py` | QDQ depthwise conv2d — int8 check + shared `_check_dwconv2d_geometry` |
| `python/tvm/relax/transform/ti_mmalib_qdq_i16_dwconv.py` | QDQ depthwise conv2d pattern matching + lowering (int16) |
| `python/tvm/relax/transform/ti_mmalib_qdq_fc.py` | QDQ FC/matmul_bias pattern matching + lowering (int8 + int16) |
| `python/tvm/relax/transform/ti_mmalib_legalize.py` | Float→int16 legalization + `_float_to_scale_shift` helper |
| `python/tvm/relax/transform/ti_mmalib_inject_dma.py` | L2 DMA prefetch injection (TIR pass, all MMALIB kernels) |
| `python/tvm/relax/transform/ti_residual_add.py` | Residual add fusion — int8 + int16, both operand orders |
| `python/tvm/relax/backend/cpu_generic/pipeline.py` | Pipeline wiring |
| `src/runtime/ti_dsp/mmalib/mmalib_wrappers.{h,cpp}` | C wrappers (8 entry points) |
| `src/runtime/ti_dsp/mmalib/tidl_maxpool_wrapper.{h,cpp}` | `max_pool2d` wrapper (TIDL-backed + scalar fallback) |
| `src/runtime/ti_dsp/kernels/c7x_residual_add.{cpp,h}` | Fixed-point residual add kernels (int8 + int16) |
| `src/runtime/ti_dsp/dma/tvm_dsp_dma.c` | EDMA runtime (virt_to_phys for staging buffer) |
| `src/target/target_kind.cc` | `mmalib` and `tidl-kernels` target attributes |
| `src/runtime/ti_dsp/firmware/c7x/dsp/CMakeLists.txt` | `USE_TIDL_RUNTIME` / `USE_TI_MMALIB` build options |
| `src/runtime/ti_dsp/firmware/c7x/dsp/src/dyn_loader.c` | Exports 8 MMALIB symbols + guards TIDL-only symbols |
| `tests/ti-dsp-runtime/mmalib-tests/` | Unit tests (conv2d, dwconv2d, FC, residual add — i8 and i16) |
| `tests/ti-dsp-runtime/quantized/` | Full model tests (ResNet, MobileNet, GoogLeNet, ShuffleNet) |

## Testing

```bash
cd tests/ti-dsp-runtime

# Quick unit tests — all MMALIB kernels, host emulation
pytest --rootdir=. mmalib-tests/ -m quick --dsp-mode=c7x_host -v

# Full unit test suite (includes non-quick tests)
pytest --rootdir=. mmalib-tests/ -v --dsp-mode=c7x_host

# ResNet-18 int8 with MMALIB (host emulation)
pytest --rootdir=. dsp-tests/test_quantized_resnet_dsp.py -v --dsp-mode=c7x_host --mmalib

# ResNet-18 int8 with MMALIB (AM67A hardware)
pytest --rootdir=. dsp-tests/test_quantized_resnet_dsp.py -v \
    --dsp-mode=c7x_dload --use-cpp-api --mmalib --profile

# PT2E int8 models (MobileNet V2)
pytest --rootdir=. pt2e-tests/test_mobilenet_v2_pt2e_dsp.py -v \
    --dsp-mode=c7x_host --mmalib

# PT2E int8 + int16 single-layer e2e (C7xMMAQuantizer → DSP)
pytest --rootdir=. pt2e-tests/ -m quick --dsp-mode=c7x_host -v
```

## Performance (AM67A C7x @ 1 GHz)

Single conv2d layer (int8, 64ch 56x56, 3x3 kernel, stride=1):

| Path | Cycles | Time | Speedup |
|------|--------|------|---------|
| MMALIB + L2 DMA prefetch | 477K | 0.48ms | 96x |
| MMALIB (MMA, DDR-resident data) | 1.67M | 1.67ms | 27x |
| TVM loop-based (C7x scalar) | 45.2M | 45.2ms | baseline |

Full model end-to-end (all layers, int8 quantized):

| Model | MMALIB Offloaded Layers | Cycles | Time |
|-------|------------------------|--------|------|
| ResNet-18 | 20 conv2d, 8 res_add, 1 FC | 100M | 100ms |
| ShuffleNet V2 | 1 conv2d, 19 dwconv2d, 1 FC | 315M | 315ms |
| MobileNet V2 | 19 conv2d, 17 dwconv2d, 10 res_add, 1 FC | 2,386M | 2.4s |
| MobileNet V3 | 6 conv2d, 15 dwconv2d, 1 FC, 10 res_add | 2,369M | 2.4s |
| GoogLeNet | 33 conv2d, 3 res_add, 1 FC | 6,790M | 6.8s |

ResNet-18 baseline (no MMALIB, scalar loops): 4,705M cycles (4.7s) → **47x speedup**.

L2 DMA prefetch provides 3.5x speedup per conv2d layer by staging input
(and weights when they fit) into L2 SRAM before the MMALIB call — the MMA
coprocessor reads from fast L2 scratchpad instead of slow DDR.

## Limitations

- Weight quantization must be symmetric (w_zp=0)
- Output-channel tiling for stride>1 adds init/exec overhead per chunk
- Intermediate activation alignment: 128-byte (exceeds MMA's 64-byte requirement)
- L2 DMA requires guard allocation (128 bytes) to prevent SE prefetch page fault
- Weights exceeding L2 budget remain in DDR (512ch layers: 2.3 MB > 1.25 MB L2)
- Depthwise conv2d requires 64-byte aligned output row stride; wrapper allocates
  a padded buffer and compacts if W_out is not 64-aligned
- MobileNet V2/V3 dominated by non-MMALIB layers (dequantize, hardswish, clip,
  adaptive_avg_pool) which still run as scalar loops
- INT16 depthwise conv2d: only 3×3 kernels supported (`mmalib_depthwise_conv2d_i16`);
  5×5 and 7×7 return `MMALIB_ERR_NOT_IMPLEMENTED` (tracked as MMALIB-882)
- INT16 QDQ activation quantization: always symmetric — d_zp and o_zp must be 0.
  Asymmetric patterns are rejected by the i16 check functions and fall through to float
