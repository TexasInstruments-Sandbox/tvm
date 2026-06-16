# MMALIB Integration Tests

End-to-end tests for TVM c_static backend calling MMALIB functions
directly on the C7x MMA accelerator (AM67A / J722S).

## Running

```bash
cd tests/ti-dsp-runtime
export TI_CGT_C7000_PATH=/opt/ti/c7x/ti-cgt-c7000_5.0.1.LTS

# Quick smoke tests (~2 min host emulation)
pytest --rootdir=. mmalib-tests/ -m quick --dsp-mode=c7x_host -v

# Full suite (all markers)
pytest --rootdir=. mmalib-tests/ -v --dsp-mode=c7x_host

# Hardware (AM67A board)
pytest --rootdir=. mmalib-tests/ -m quick --dsp-mode=c7x_dload -v
```

## Test Files

### Kernel unit tests (execution required)

| File | Op | Dtype | Path | Description |
|------|----|-------|------|-------------|
| `test_mmalib_matmul_dsp.py` | matmul | int8 | legalize | Direct legalization via `LegalizeOps`, exact match |
| `test_mmalib_matmul_i16_dsp.py` | matmul | int16 | legalize | Float→int16 dynamic quant + shift-based overflow prevention; used by SmolLM MLP offload |
| `test_mmalib_conv2d_dsp.py` | conv2d | int16 | legalize | Direct int16 conv2d legalization, exact match |
| `test_mmalib_conv2d_i8_dsp.py` | conv2d | int8 | QDQ | `FuseMMALIBQDQConv2d` — PT2E pattern with per-channel bias/scale/shift, ±2 tolerance |
| `test_mmalib_conv2d_i16_dsp.py` | conv2d | int16 | QDQ | `FuseMMALIBQDQConv2dI16` — same PT2E pattern but int16, ±10 tolerance (Phase 2b) |
| `test_mmalib_dwconv2d_i8_dsp.py` | depthwise conv2d | int8 | QDQ | `FuseMMALIBQDQDwConv2d` — depthwise (groups=C), 3×3/5×5/7×7, ±2 tolerance |
| `test_mmalib_dwconv_i16_dsp.py` | depthwise conv2d | int16 | QDQ | `FuseMMALIBQDQDwConv2dI16` — int16 depthwise, 3×3 only (MMALIB-882), ±5 tolerance (Phase 2c) |
| `test_mmalib_fc_i8_dsp.py` | FC / linear | int8 | QDQ | `FuseMMALIBQDQFC` — matmul_bias_i8, per-channel scale/shift; 2D and 3D reshape variants |
| `test_mmalib_fc_i16_dsp.py` | FC / linear | int16 | QDQ + direct | Direct `mmalib_matmul_bias_i16` wrapper tests (SmolLM dims) plus `FuseMMALIBQDQFCI16` PT2E QDQ fusion (Phase 2b) |
| `test_mmalib_residual_add_i8_dsp.py` | residual add | int8 | QDQ | `FuseInt8ResidualAdd` — both `add(x,skip)` and `add(skip,x)` operand orders (Phase 2a) |
| `test_mmalib_residual_add_i16_dsp.py` | residual add | int16 | QDQ | `FuseInt16ResidualAdd` — symmetric only (zp=0), both operand orders (Phase 2c) |

### Pass-level unit tests (pure Python, no DSP required)

| File | What it tests |
|------|---------------|
| `test_mmalib_inject_dma.py` | `InjectMMALIBDMA` guard bytes: verifies `pad_top` (not `stride_h`) is read from args[15] for i8 and i16 conv2d; fallback to 128 bytes when `pad_top == 0` |
| `test_mmalib_fc_i16_dsp.py` *(guard test)* | `test_fuse_fc_i16_rejects_nonzero_o_zp` — verifies the i16 FC check function rejects patterns with non-zero output zero-point |

## How it works

Each DSP test creates a Relax IRModule, compiles with
`c_static -mcpu=c7x -mmalib=1`, builds an executable (c7x_host or c7x_dload),
runs it, and compares against a numpy float reference.

### Two code paths

**Legalize path** (`test_mmalib_matmul_dsp.py`, `test_mmalib_conv2d_dsp.py`,
`test_mmalib_matmul_i16_dsp.py`): The `LegalizeOps` pass with a custom
`legalize_map` replaces eligible float ops with `call_extern` to MMALIB
wrappers. No quantization nodes in the graph.

**QDQ fusion path** (all other tests): The `FuseMMALIBQDQ*` passes run
*before* `FuseQDQToInt8Conv2D` and match the intact PT2E QDQ pattern:
```
dequantize(data_int8/16) → op(_, dequantize(weight)) → [bias] → [relu] → quantize
```
The fused kernel receives compile-time-computed integer bias/scale/shift
derived from the quantization parameters.

### Tolerances

| Dtype | Tolerance | Reason |
|-------|-----------|--------|
| int8 | ≤ 2 | uint8 scale/shift approximation; small K |
| int16 | ≤ 5–10 | wider uint8 scale/shift approximation error for larger K |
| int16 direct | ≤ 1 | per-row L1-norm shift, no requantization |

## Data layout

All ops use NCHW (planar channel-first). The pipeline skips NHWC conversion
when `-mmalib=1` is set.

## Known limitations

- **INT16 depthwise**: only 3×3 kernels supported (`mmalib_depthwise_conv2d_i16`);
  5×5 and 7×7 return `MMALIB_ERR_NOT_IMPLEMENTED` (tracked as MMALIB-882).
- **INT16 QDQ activation quantization**: always symmetric (d_zp = 0 required).
  Asymmetric activation quant (`d_zp ≠ 0`) is rejected by the i16 check
  functions and falls through to float computation.

## Prerequisites

- TVM built with c_static backend
- TI C7000 CGT with host emulation (`TI_CGT_C7000_PATH`)
- MMALIB SDK at `/opt/ti/am67a/.../mmalib_11_02_00_06` (auto-detected)
- DSP runtime built for host emulation:
  ```bash
  cd src/runtime/ti_dsp && bash build_runtime.sh c7x_host
  ```
- For `c7x_dload` tests: firmware rebuilt and deployed — see
  `docs/dsp/operations.md` and the firmware skill for the full procedure
