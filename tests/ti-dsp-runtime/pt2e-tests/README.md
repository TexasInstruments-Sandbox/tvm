# PT2E Quantizer Tests

End-to-end tests for the `C7xMMAQuantizer` → TVM c_static → MMALIB pipeline.
`C7xMMAQuantizer` is a `torchao` `Quantizer` subclass that annotates a PyTorch
exported graph for int8 or int16 quantization targeting TI C7x MMALIB kernels.

## Running

```bash
cd tests/ti-dsp-runtime
export TI_CGT_C7000_PATH=/opt/ti/c7x/ti-cgt-c7000_5.0.1.LTS

# Pure-Python tests (no DSP or toolchain required, ~3s)
pytest --rootdir=. pt2e-tests/ -m quick -v

# Host emulation (requires TI CGT, ~1 min)
pytest --rootdir=. pt2e-tests/ -m quick --dsp-mode=c7x_host -v

# AM67A hardware
pytest --rootdir=. pt2e-tests/ -m quick --dsp-mode=c7x_dload -v
```

## What is being tested

The tests validate the full quantization pipeline:

```
float PyTorch model
  → C7xMMAQuantizer (prepare_pt2e / calibrate / convert_pt2e)
  → from_exported_program  →  Relax IRModule
  → c_static -mcpu=c7x -mmalib=1  (FuseMMALIBQDQ* passes)
  → MMALIB kernel on c7x_host / c7x_dload
```

There are three layers of testing, from fastest to slowest:

| Layer | Requires DSP? | What it checks |
|-------|--------------|----------------|
| Annotation / PyTorch-level | No | C7xMMAQuantizer annotates the right ops with correct dtypes and zero_points |
| TVM import + fusion | No | `from_exported_program` imports correctly; MMALIB fusion passes fire on the right patterns |
| End-to-end on DSP | Yes | Compiled DSP output matches PyTorch quantized reference within tolerance |

## Test files

### `test_c7x_mma_quantizer.py` — annotator unit tests (pure Python)

Tests `C7xMMAQuantizer` at the PyTorch graph level, before TVM is involved.

| Test | What it checks |
|------|----------------|
| `test_conv_int8_sym_produces_qdq` | Conv2d gets Q/DQ nodes with int8 symmetric spec |
| `test_conv_int8_affine_produces_qdq` | Conv2d with affine (asymmetric) activation also works |
| `test_linear_int8_produces_qdq` | Linear layer gets Q/DQ nodes |
| `test_depthwise_conv_int8_produces_qdq` | Depthwise conv2d annotated via `conv2d.default` |
| `test_conv_weight_zero_point_is_zero` | Per-channel weight zero_points are all 0 (MMALIB requirement) |
| `test_linear_weight_zero_point_is_zero` | Same for linear weights |
| `test_int16_force_symmetric_warning` | `symmetric_activations=False` with int16 emits UserWarning and is ignored |
| `test_int16_produces_int16_quantize_nodes` | int16 quantizer emits `torch.int16` quantize nodes |
| `test_no_double_annotation` | Running annotate() twice does not overwrite existing annotations |
| `test_invalid_dtype_raises` | `C7xMMAQuantizer(dtype="int4")` raises ValueError |
| `test_mm_both_inputs_use_act_spec` | `aten.mm` annotates both matrix inputs as activations |
| `test_addmm_bias_not_annotated` | Bias arg of `aten.addmm` is not annotated (stays float32) |
| `test_add_tensor_both_inputs_annotated` | Residual add annotates both inputs |
| `test_add_tensor_produces_qdq` | Residual add model produces Q/DQ nodes after convert |
| `test_transparent_ops_get_annotated` | max_pool2d, view, permute, flatten are annotated by `_TRANSPARENT_OPS` |
| `test_transparent_ops_output_uses_shared_spec` | Transparent op output uses `SharedQuantizationSpec` → matching scales → EliminateQDQTransparent fires |
| `test_transparent_ops_produce_qdq` | Full convert produces Q/DQ wrappers that EliminateQDQTransparent can remove |

### `test_c7x_mma_quantizer_i16.py` — int16 pipeline unit tests (pure Python)

Tests the int16 PT2E flow through TVM import and MMALIB fusion, without DSP execution.

**Section 1 — Annotation:**

| Test | What it checks |
|------|----------------|
| `test_int16_produces_int16_quantize_on_activations` | Activations quantized with `torch.int16` dtype |
| `test_int16_weight_zero_point_is_zero` | Per-channel weight zero_points are 0 |
| `test_int16_activation_zero_point_is_zero` | Activation zero_points are 0 (symmetric only) |

**Section 2 — TVM import:**

| Test | What it checks |
|------|----------------|
| `test_int16_tvm_import_succeeds` | `from_exported_program` does not crash on int16 graphs |
| `test_int16_tvm_import_zero_point_is_int8` | Zero_points in Relax IR are `int8` (regression: previously crashed with "got int16") |

**Section 3 — Fusion:**

| Test | What it checks |
|------|----------------|
| `test_int16_conv2d_fusion_fires` | `FuseMMALIBQDQConv2dI16` produces a `mmalib_conv2d` function |
| `test_int16_linear_fusion_fires` | `FuseMMALIBQDQFCI16` produces a `mmalib_fc_i16` function |
| `test_int16_residual_add_fusion_fires` | `FuseInt16ResidualAdd` produces an `i16_residual_add` function |
| `test_int16_dwconv2d_fusion_fires` | `FuseMMALIBQDQDwConv2dI16` produces a `mmalib_dwconv2d` function |
| `test_int16_5x5_dwconv2d_not_fused` | 5×5 depthwise is **not** fused (MMALIB-882: unsupported kernel size) |
| `test_int16_i8_passes_not_triggered_on_int16_graph` | Int8 conv2d pass does not match int16 inputs |

### `test_c7x_mma_quantizer_e2e_dsp.py` — end-to-end DSP tests

Runs single-layer models through the full pipeline on DSP hardware or host emulation.
Correctness is asserted against the PyTorch quantized reference output.

**Int8 tests** — `max_diff ≤ 2`:

| Test | Model | MMALIB kernel |
|------|-------|---------------|
| `test_e2e_conv2d_i8` | `Conv2d(3,8,3)` | `mmalib_conv2d_i8` |
| `test_e2e_depthwise_conv2d_i8` | `Conv2d(8,8,3,groups=8)` | `mmalib_depthwise_conv2d_i8` |
| `test_e2e_linear_i8` | `Linear(64,128)` | `mmalib_matmul_bias_i8` |
| `test_e2e_linear_i8_no_bias` | `Linear(64,128,bias=False)` | generic int8 fallback |
| `test_e2e_residual_add_i8` | `Conv2d + skip` | `tvm_int8_residual_add_relu` |
| `test_e2e_linear_3d_i8` | `Linear(64,128)` on 3D input | `mmalib_matmul_bias_i8` |

**Int16 tests** — `max_diff ≤ 10` (higher tolerance: uint8 scale/shift requantization
error scales with √K; observed max ≤ 6 in practice):

| Test | Model | MMALIB kernel |
|------|-------|---------------|
| `test_e2e_conv2d_i16` | `Conv2d(32,32,3)` | `mmalib_conv2d_i16` |
| `test_e2e_depthwise_conv2d_i16` | `Conv2d(32,32,3,groups=32)` | `mmalib_depthwise_conv2d_i16` |
| `test_e2e_linear_i16` | `Linear(64,64)` | `mmalib_matmul_bias_i16` |
| `test_e2e_residual_add_i16` | `Conv2d(32,32,3) + skip` | `tvm_int16_residual_add_relu` |

### `test_c7x_tidl_activation.py` — TIDL activation fusion unit tests (pure Python)

Tests `FuseQDQToTIDLActivation`, `FuseQDQToTIDLAvgPool`, and
`FuseQDQToTIDLLayerNorm` at the Relax IR level without DSP execution.

**Annotation tests:**

| Test | What it checks |
|------|----------------|
| `test_gelu_gets_annotated` | `aten.gelu` is in `_TIDL_ACT_OPS` and gets annotated |
| `test_silu_gets_annotated` | `aten.silu` gets annotated |
| `test_hardsigmoid_gets_annotated` | `aten.hardsigmoid` gets annotated |
| `test_hardswish_gets_annotated` | `aten.hardswish` gets annotated |

**Fusion tests:**

| Test | What it checks |
|------|----------------|
| `test_activation_fusion_fires[GeluModel-tidl_int8_gelu]` | `FuseQDQToTIDLActivation` emits `call_tir(tidl_int8_gelu, ...)` |
| `test_activation_fusion_fires[SiluModel-tidl_int8_silu]` | silu fused to `tidl_int8_silu` |
| `test_activation_fusion_fires[HardsigmoidModel-tidl_int8_hardsigmoid]` | hardsigmoid fused |
| `test_activation_fusion_fires[HardswishModel-tidl_int8_hardswish]` | hardswish fused |
| `test_gelu_output_is_int8` | Fused gelu kernel output dtype is int8 |
| `test_i8_passes_do_not_trigger_on_int16` | `tidl_int8_gelu` does not fire on int16 graphs |

### `test_c7x_tidl_activation_e2e_dsp.py` — TIDL activation end-to-end DSP tests

Runs models through the full pipeline on DSP hardware or host emulation.
Validates `FuseQDQToTIDLActivation`, `FuseQDQToTIDLAvgPool`, and
`FuseQDQToTIDLLayerNorm` produce correct output.

All tests: `max_diff ≤ 2` vs PyTorch quantized reference.

**Activation tests** — model: `Linear(32,32) → activation → Linear(32,16)`:

| Test | Kernel |
|------|--------|
| `test_e2e_gelu_i8` | `tidl_int8_gelu` |
| `test_e2e_silu_i8` | `tidl_int8_silu` |
| `test_e2e_hardsigmoid_i8` | `tidl_int8_hardsigmoid` |
| `test_e2e_hardswish_i8` | `tidl_int8_hardswish` |

**Pooling tests** — model: `Conv2d(8,8,3) → pool`:

| Test | Model | Kernel |
|------|-------|--------|
| `test_e2e_global_avg_pool_i8` | `AdaptiveAvgPool2d(1,1)` on `[1,8,16,16]` | `tidl_int8_global_avg_pool` |
| `test_e2e_avg_pool2d_i8` | `AvgPool2d(3,stride=1,padding=1)` on `[1,8,16,16]` | `tidl_int8_avg_pool` |

**Normalization test** — model: `Linear(32,32) → LayerNorm(32) → Linear(32,16)`:

| Test | Kernel |
|------|--------|
| `test_e2e_layer_norm_i8` | `tidl_int8_layer_norm` |

### `test_mobilenet_v2_pt2e_dsp.py` — MobileNetV2 integration test

Exercises all four int8 MMALIB kernels together on a real classification model.
Two assertions: (1) top-1 prediction matches without MMALIB (validates import pipeline),
(2) `max_diff ≤ 20` with MMALIB (validates all kernels produce plausible output).

## Shared helpers (`pt2e_utils.py`)

| Function | Purpose |
|----------|---------|
| `quantize_pt2e(model, inputs, quantizer)` | Export → prepare → calibrate → convert; returns Q/DQ GraphModule |
| `e2e_quantize_and_import(model, inputs, dtype)` | Full pipeline to Relax IRModule with params bound |
| `run_and_check(mod, input, ref, dsp_mode, ...)` | Compile with `-mmalib=1`, run on DSP, assert `max_diff` |

## Prerequisites

- TVM built with c_static backend (`TVM_HOME` set, `PYTHONPATH` includes `python/`)
- For DSP tests: TI C7000 CGT (`TI_CGT_C7000_PATH`)
- For `c7x_dload` tests: firmware deployed on AM67A board (`deploy-c7x.sh`)
- DSP runtime built: `cd src/runtime/ti_dsp && bash build_runtime.sh c7x_host`
