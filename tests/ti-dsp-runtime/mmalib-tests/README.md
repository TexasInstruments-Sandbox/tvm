# MMALIB Integration Tests

End-to-end tests for TVM c_static backend calling MMALIB functions
directly on the C7x MMA accelerator (AM67A / J722S).

## Running

```bash
cd tests/ti-dsp-runtime
export TI_CGT_C7000_PATH=/opt/ti/c7x/ti-cgt-c7000_5.0.1.LTS
pytest --rootdir=. mmalib-tests/ -v --dsp-mode=c7x_host
```

## Tests

| Test | Op | Dtype | Description |
|------|----|-------|-------------|
| `test_mmalib_matmul_dsp.py` | matmul | int16 | Direct legalization, exact match |
| `test_mmalib_conv2d_dsp.py` | conv2d | int16 | Direct legalization, exact match |
| `test_mmalib_conv2d_i8_dsp.py` | conv2d | int8 | QDQ fusion (bias+rescale), ±2 tolerance |

## How it works

Each test creates a Relax model, compiles with `c_static -mcpu=c7x -mmalib=1`,
builds a c7x_host binary (x86 host emulation linked against MMALIB), runs it,
and compares against a numpy reference.

- **Int16 tests**: The `customize_legalize_map` in `LegalizeOps` replaces
  eligible ops with `call_extern` to MMALIB wrappers.
- **Int8 QDQ test**: `FuseConv2dToMMALIB` matches the quantized pattern
  (conv2d→rescale→int8), converts float bias/rescale to integer scale/shift,
  and emits a single `call_extern("mmalib_conv2d_i8", ...)`.

## Data layout

All ops use NCHW (planar channel-first). The pipeline skips NHWC conversion
when `-mmalib=1` is set.

## Prerequisites

- TVM built with c_static backend
- TI C7000 CGT with host emulation (`TI_CGT_C7000_PATH`)
- MMALIB SDK at `/opt/ti/am67a/.../mmalib_11_02_00_06` (auto-detected)
- DSP runtime built: `cd src/runtime/ti_dsp && bash build_runtime.sh c7x_host`
