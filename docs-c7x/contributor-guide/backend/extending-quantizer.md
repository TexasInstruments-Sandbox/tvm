# Extending the C7xMMAQuantizer

The `C7xMMAQuantizer` is a `torchao.quantization.pt2e.Quantizer` subclass that
annotates PyTorch-exported graphs for quantization targeting the C7x DSP.
This guide covers its structure, constraints, and how to add new operators.

## Overview

**Quantization** converts a float32 model to use integer arithmetic (int8 or int16).
On C7x, quantized ops map to MMALIB (MMA hardware accelerator), TIDL optimized
kernels, or TVM vectorized implementations — all significantly faster than float32.

**PT2E** (PyTorch 2 Export) is the modern PyTorch quantization pipeline operating
on exported FX graphs. The pipeline has four stages:

1. **Annotate** — `Quantizer` subclass marks each op to quantize, specifying
   which inputs are activations/weights/bias and the numerical format
2. **Prepare** — `prepare_pt2e` inserts observer modules next to marked tensors
3. **Calibrate** — Run representative inputs; observers record min/max ranges
4. **Convert** — `convert_pt2e` replaces observers with explicit quantize/dequantize
   (Q/DQ) nodes

After conversion, a `conv2d` becomes:
```
x_int8 → dequantize(x_int8, scale=x_s, zp=x_zp) ─┐
weight_int8 → dequantize(weight_int8, scale=w_s, zp=0) ─┤→ conv2d(bias_float) → quantize(scale=y_s, zp=y_zp) → y_int8
```

TVM fusion passes then collapse `dequantize → conv2d → quantize` into a single
integer kernel call (e.g., `mmalib_conv2d_i8`), eliminating intermediate
float32 conversions at runtime.

**File:** `python/tvm/relax/frontend/torch/c7x_mma_quantizer.py`
**Export:** `from tvm.relax.frontend.torch import C7xMMAQuantizer`

## Full Pipeline

```
float PyTorch model
  → torch.export.export
  → prepare_pt2e(model, C7xMMAQuantizer)
  → calibrate (run representative inputs)
  → convert_pt2e  →  Q/DQ graph (GraphModule)
  → torch.export.export  →  ExportedProgram
  → from_exported_program  →  Relax IRModule
  → c_static -mcpu=c7x -mmalib=1
      Int8:  FuseMMALIBQDQConv2d / FuseMMALIBQDQDwConv2d / FuseMMALIBQDQFC / FuseInt8ResidualAdd
      Int16: FuseMMALIBQDQConv2dI16 / FuseMMALIBQDQDwConv2dI16 / FuseMMALIBQDQFCI16 / FuseInt16ResidualAdd
  → MMALIB kernels on C7x
```

## Quantization Constraints

| Tensor | dtype | scheme | zero_point |
|--------|-------|--------|------------|
| Activations (int8) | `torch.int8` [-128, 127] | per-tensor symmetric **or** per-tensor affine | 0 or any |
| Activations (int16) | `torch.int16` [-32768, 32767] | per-tensor symmetric only | 0 (required) |
| Weights (int8) | `torch.int8` [-128, 127] | **per-channel symmetric** | 0 (required) |
| Weights (int16) | `torch.int16` [-32768, 32767] | per-channel symmetric | 0 |
| Bias | `torch.float32` (not quantized) | — | — |
| Output | same as activations | per-tensor | same as activations |

Observers used:
- Activations: `torchao.quantization.pt2e.MinMaxObserver`
- Weights: `torchao.quantization.pt2e.PerChannelMinMaxObserver`

**Note:** The `torch.ao` observer hierarchy is incompatible — `convert_pt2e` would
silently leave observers unreplaced as `call_module` nodes.

## Annotated Operator Groups

`annotate()` walks the exported FX graph and records which tensors to quantize.
`prepare_pt2e` reads those records and inserts observers; `convert_pt2e` later
replaces observers with Q/DQ nodes. **Any op not in the lists below remains float32.**

The annotated ops fall into six groups, each producing a different Q/DQ pattern
handled differently by the TVM backend.

### Group 1 — MMALIB Compute Ops (`_WEIGHT_OPS`)

Both activation input and weight are quantized; bias stays float32.
TVM fuses the resulting Q/DQ pattern with an MMALIB kernel call.

| `node.target` | annotated inputs | int8 fusion pass | int16 fusion pass |
|---|---|---|---|
| `aten.conv2d.default` | `args[0]` act, `args[1]` weight (per-channel) | `FuseMMALIBQDQConv2d` / `FuseMMALIBQDQDwConv2d` | `FuseMMALIBQDQConv2dI16` / `FuseMMALIBQDQDwConv2dI16`¹ |
| `aten.linear.default` | `args[0]` act, `args[1]` weight (per-channel) | `FuseMMALIBQDQFC` | `FuseMMALIBQDQFCI16` |

- `aten.conv2d.default` covers both regular and depthwise; TVM distinguishes by `groups`.
- ¹ int16 depthwise only supports **3×3 kernels** (MMALIB-882 blocks 5×5/7×7); 5×5 layers are rejected and fall through to float32.

### Group 2 — Activation-Only Compute Ops (`_ACT_ONLY_OPS`)

Both inputs are activation tensors; no weight with per-channel spec.
Bias (`args[0]` of `addmm`) stays float32.

| `node.target` | annotated inputs | int8 fusion pass | int16 fusion pass |
|---|---|---|---|
| `aten.mm.default` | `args[0]`, `args[1]` | `FuseMMALIBQDQFC` | `FuseMMALIBQDQFCI16` |
| `aten.addmm.default` | `args[1]`, `args[2]` | `FuseMMALIBQDQFC` | `FuseMMALIBQDQFCI16` |
| `aten.add.Tensor` | `args[0]`, `args[1]` | `FuseInt8ResidualAdd` | `FuseInt16ResidualAdd` |

- `add.Tensor` only annotated when both arguments are tensors (scalar args skipped).
- int16 activations always symmetric; `UserWarning` emitted if `symmetric_activations=False`.

### Group 3 — C7x Activation Ops (`_TIDL_ACT_OPS`)

Single activation input; output gets independent quantization.
TVM fuses the Q/DQ pattern with a native C7x int8 kernel (no TIDL library
call — despite the internal `_TIDL_ACT_OPS` set name).

| `node.target` | C7x kernel |
|---|---|
| `aten.gelu.default` | `c7x_int8_gelu` |
| `aten.silu.default` | `c7x_int8_silu` |
| `aten.hardsigmoid.default` | `c7x_int8_hardsigmoid` |
| `aten.hardswish.default` | `c7x_int8_hardswish` |

### Group 4 — Average Pooling Ops (`_AVG_POOL_OPS`)

Single activation input; output gets independent quantization.
Unlike max pooling, averaging changes numerical range so Q/DQ wrappers
cannot be eliminated and must fuse into a kernel. Like Group 3, these are
native C7x kernels, not TIDL library calls.

| `node.target` | C7x kernel |
|---|---|
| `aten.adaptive_avg_pool2d.default` | `c7x_int8_global_avg_pool` (1×1) or `c7x_int8_avg_pool` |
| `aten.avg_pool2d.default` | `c7x_int8_avg_pool` |

### Group 5 — Normalization Ops (`_NORM_OPS`)

Single activation input; weight and bias stay float32.
Normalization arithmetic runs in float32 internally (dequant → normalize → requant),
fused into a native C7x kernel.

| `node.target` | C7x kernel |
|---|---|
| `aten.layer_norm.default` | `c7x_int8_layer_norm` |

### Group 6 — Transparent Ops (`_TRANSPARENT_OPS`)

Single activation input. Output forced to share **same** scale as input via
`SharedQuantizationSpec`. `EliminateQDQTransparent` then removes Q/DQ wrappers
entirely, leaving the op to receive and produce raw int8 data.

| `node.target` | why scale unchanged |
|---|---|
| `aten.max_pool2d.default` | `max` is monotonic — preserves rank order |
| `aten.view.default` | pure shape op (reshape) |
| `aten.permute.default` | pure shape op (axis reorder) |
| `aten.flatten.using_ints` | pure shape op (collapse dims) |

### Not Annotated — Stays Float32

| Op | Reason |
|---|---|
| `aten.cat.default` | Inputs may have different per-tensor scales; concat requantization not yet implemented |
| `aten.softmax.default` | `exp` requires float precision; no int8 benefit |
| `aten.group_norm.default` | No TIDL kernel available; needs custom C7x implementation |

## Usage

```python
import torch
from torchao.quantization.pt2e.quantize_pt2e import prepare_pt2e, convert_pt2e
from tvm.relax.frontend.torch import C7xMMAQuantizer, from_exported_program
from tvm import relax

# 1. Export float model
exported = torch.export.export(model, example_inputs).module()

# 2. Prepare + calibrate
quantizer = C7xMMAQuantizer(dtype="int8", symmetric_activations=True)
prepared = prepare_pt2e(exported, quantizer)
with torch.no_grad():
    for batch in calibration_loader:
        prepared(batch)

# 3. Convert → Q/DQ graph
quantized_pt = convert_pt2e(prepared)

# 4. Re-export and import into TVM
quantized_ep = torch.export.export(quantized_pt, example_inputs)
mod = from_exported_program(quantized_ep, keep_params_as_input=True)
mod, params = relax.frontend.detach_params(mod)
func_params_dict = dict(zip(mod["main"].params[1:], params["main"]))
mod = relax.transform.BindParams("main", func_params_dict)(mod)

# 5. Compile with MMALIB fusion
# FuseMMALIBQDQConv2d / FC / DwConv2d / FuseInt8ResidualAdd fire automatically
```

## Tests

### Unit Tests (no hardware)

```bash
cd tests/ti-dsp-runtime
# int8 annotator tests
pytest --rootdir=. pt2e-tests/test_c7x_mma_quantizer.py -m quick -v
# int16 pipeline tests (import fix, fusion pass firing)
pytest --rootdir=. pt2e-tests/test_c7x_mma_quantizer_i16.py -m quick -v
```

**`test_c7x_mma_quantizer.py`** covers: Q/DQ presence for all annotated ops,
weight zero_point=0, int16 symmetric forcing, no-double-annotation guard,
`aten.add.Tensor` annotation structure, invalid dtype error.

**`test_c7x_mma_quantizer_i16.py`** covers:

| Test group | What it checks |
|---|---|
| Annotation | int16 activations get `torch.int16` dtype; weight and activation zero_points are 0 |
| TVM import | `from_exported_program` succeeds on int16 graphs; zero_points in Relax IR are `int8` (regression for zero_point dtype fix) |
| Fusion | Conv2d / depthwise / linear / residual add fire correct i16 kernels; 5×5 depthwise not fused (MMALIB-882); int8 passes don't trigger on int16 graphs |

### End-to-End Tests (c7x_host / c7x_dload)

```bash
cd tests/ti-dsp-runtime
export TI_CGT_C7000_PATH=/opt/ti/c7x/ti-cgt-c7000_5.0.1.LTS

# Host emulation (no board)
pytest --rootdir=. pt2e-tests/ -m quick --dsp-mode=c7x_host -v

# AM67A hardware
pytest --rootdir=. pt2e-tests/ -m quick --dsp-mode=c7x_dload -v
```

#### Kernel-level tests — `test_c7x_mma_quantizer_e2e_dsp.py`

Single-layer models compared against PyTorch quantized reference.

**Int8 tests** — `max_diff ≤ 2`:

| Test | Model | MMALIB kernel | Input shape |
|---|---|---|---|
| `test_e2e_conv2d_i8` | `Conv2d(3,8,3,p=1)` | `mmalib_conv2d_i8` | 1×3×56×56 |
| `test_e2e_depthwise_conv2d_i8` | `Conv2d(8,8,3,p=1,g=8)` | `mmalib_depthwise_conv2d_i8` | 1×8×56×56 |
| `test_e2e_linear_i8` | `Linear(64,128)` | `mmalib_matmul_bias_i8` | 1×64 |
| `test_e2e_linear_i8_no_bias` | `Linear(64,128,bias=False)` | generic int8 fallback¹ | 1×64 |
| `test_e2e_residual_add_i8` | `Conv2d(8,8,3,p=1) + x` | `c7x_int8_residual_add_relu` | 1×8×16×16 |
| `test_e2e_linear_3d_i8` | `Linear(64,128)` | `mmalib_matmul_bias_i8` | 1×4×64 |

**Int16 tests** — `max_diff ≤ 10` (higher tolerance: uint8 scale/shift error scales with √K):

| Test | Model | MMALIB kernel | Input shape |
|---|---|---|---|
| `test_e2e_conv2d_i16` | `Conv2d(32,32,3,p=1)` | `mmalib_conv2d_i16` | 1×32×28×28 |
| `test_e2e_depthwise_conv2d_i16` | `Conv2d(32,32,3,p=1,g=32)` | `mmalib_depthwise_conv2d_i16` | 1×32×28×28 |
| `test_e2e_linear_i16` | `Linear(64,64)` | `mmalib_matmul_bias_i16` | 1×64 |
| `test_e2e_residual_add_i16` | `Conv2d(32,32,3,p=1) + x` | `c7x_int16_residual_add_relu` | 1×32×16×16 |

¹ `mmalib_matmul_i8` (no-bias) is defined but no TVM fusion pass emits it; no-bias linear compiles correctly via generic codegen.

#### Classification model test — `test_mobilenet_v2_pt2e_dsp.py`

MobileNetV2 exercises all four int8 MMALIB kernels together. Input: `dog.jpg`
from `tests/cstatic/test_images/`, loaded via `cl_torchvision.load_model_with_preprocessing`.
10 calibration batches (same count as `quantized/model_utils.py`).

Two assertions:
1. **No-MMALIB, top-1 match** — validates C7xMMAQuantizer + TVM import pipeline
2. **MMALIB, `max_diff ≤ 20.0`** — validates all four MMALIB kernels compile and execute. Top-1 not asserted: MMALIB integer arithmetic diverges from float-simulated int8 across 17+ bottleneck blocks. Consistent with `quantized/test_quantized_mobilenet_v2.py` using `atol=25.0` for MMALIB with XNNPACKQuantizer — the gap is inherent to actual int8 arithmetic vs float simulation.

## Adding a New Operator

To add a new op to `C7xMMAQuantizer`:

### 1. Determine the Group

Based on the op's characteristics:
- Has weight tensor with per-channel quantization? → Group 1 (`_WEIGHT_OPS`)
- Only activation inputs (matmul-like)? → Group 2 (`_ACT_ONLY_OPS`)
- Single activation input, TIDL kernel exists? → Group 3 (`_TIDL_ACT_OPS`)
- Average pooling? → Group 4 (`_AVG_POOL_OPS`)
- Normalization with float weight/bias? → Group 5 (`_NORM_OPS`)
- Pure shape op or monotonic? → Group 6 (`_TRANSPARENT_OPS`)
- None of the above → don't annotate (stays float32)

### 2. Add to the Appropriate Frozenset

In `c7x_mma_quantizer.py`, the groups are defined as `frozenset` at module level:

```python
_WEIGHT_OPS = frozenset([
    "aten.conv2d.default",
    "aten.linear.default",
])

_ACT_ONLY_OPS = frozenset([
    "aten.mm.default",
    "aten.addmm.default",
    "aten.add.Tensor",
])

_TIDL_ACT_OPS = frozenset([
    "aten.gelu.default",
    "aten.silu.default",
    "aten.hardsigmoid.default",
    "aten.hardswish.default",
])
# ... etc
```

Add the `node.target` string (as shown in `torch.export` output) to the appropriate set.

### 3. Add Quantization Spec

In `annotate()`, the quantizer creates `QuantizationSpec` objects for each
input/output. Follow the existing patterns:

```python
# For weights (per-channel symmetric)
weight_spec = QuantizationSpec(
    dtype=torch.int8,  # or torch.int16
    observer_or_fake_quant_ctr=PerChannelMinMaxObserver,
    quant_min=-128, quant_max=127,  # or -32768, 32767 for int16
    qscheme=torch.per_channel_symmetric,
    ch_axis=0,  # output channel axis
)

# For activations (per-tensor)
act_spec = QuantizationSpec(
    dtype=torch.int8,  # or torch.int16
    observer_or_fake_quant_ctr=MinMaxObserver,
    quant_min=-128, quant_max=127,
    qscheme=torch.per_tensor_symmetric,  # or affine for int8
)
```

For int16, always use symmetric (`qscheme=torch.per_tensor_symmetric`,
`quant_min=-32768`, `quant_max=32767`). The quantizer enforces this and
emits a `UserWarning` if `symmetric_activations=False` is passed.

### 4. Ensure TVM Fusion Pass Exists

The quantizer annotation is only half the pipeline. A corresponding TVM
Relax pass must exist to fuse the resulting Q/DQ pattern:

- Group 1 → MMALIB QDQ fusion passes (`ti_mmalib_qdq_*.py`)
- Group 2 → MMALIB QDQ FC / residual add passes
- Group 3 → `FuseQDQToC7xActivation` (`ti_fuse_qdq_c7x_activation.py`)
- Group 4 → `FuseQDQToC7xAvgPool` (`ti_fuse_qdq_c7x_avgpool.py`)
- Group 5 → `FuseQDQToC7xLayerNorm` (`ti_fuse_qdq_c7x_layernorm.py`)
- Group 6 → `EliminateQDQTransparent` (`ti_eliminate_qdq_transparent.py`)

If no pass exists, the Q/DQ nodes survive to `LegalizeOps` and are lowered
to scalar element-wise float conversions — slower than the original float model.

### 5. Add Tests

- Unit test in `pt2e-tests/test_c7x_mma_quantizer.py` (annotation presence)
- E2E test in `pt2e-tests/test_c7x_mma_quantizer_e2e_dsp.py` (accuracy vs reference)

## Future Work

### `aten.cat.default` — Concat Annotation

Concat is the only common op not yet annotated. Adding it requires:

1. **Quantizer** — `aten.cat.default` takes a Python list as `args[0]`; annotation
   loop must iterate the list and assign `act_spec` to each element. Output gets
   independent `act_spec` (concat is not scale-transparent when inputs differ).

2. **`EliminateQDQTransparent`** — A concat pattern (`_make_qdq_concat_pattern`)
   is already defined in `ti_eliminate_qdq_transparent.py` but not wired into
   `transform_module()`. Handles common case where all inputs share calibrated scale.

Without this, concat between two int8 tensors forces float32 round-trip.

### `aten.group_norm.default` — No Kernel Available

Used in MobileNetV3 and transformers. No TIDL kernel exists (`TIDL_layerNorm`
handles only layer norm). Options:

- Write custom C7x kernel (similar to `c7x_int8_layer_norm` with group partitioning)
- Defer — not in current test model set (ResNet, MobileNetV2, SmolLM)

Until a kernel exists, `group_norm` should remain unannotated so it stays
float32 without QDQ overhead.

### HistogramObserver (Optional)

`C7xMMAQuantizer` uses `MinMaxObserver` for activations. With large calibration
datasets, `HistogramObserver` produces tighter scales (clips outliers) at
higher memory cost. With current 10-batch calibration, difference is negligible.
To switch, replace `MinMaxObserver` in `_act_spec()` in `c7x_mma_quantizer.py`.

## Known Constraints

- **Int16 activation quantization is symmetric only.** Input zero-point (`d_zp`)
  and output zero-point (`o_zp`) must both be 0; i16 fusion passes enforce this.
- **Int16 depthwise: 3×3 kernels only.** 5×5 and 7×7 not implemented in MMALIB
  for int16 (MMALIB-882). Check function rejects them; layers fall through to float32.
- **Int16 quantization requires calibration.** Unlike weight-only int8 (where only
  constant weights are quantized), int16 PT2E quantizes activations too, so
  calibration data must run through observers.
- **No-bias matmul MMALIB fusion not implemented.** `mmalib_matmul_i8` is defined
  but no TVM pass emits it; no-bias linear falls through to generic codegen.

## Related Files

| File | Purpose |
|---|---|
| `python/tvm/relax/frontend/torch/c7x_mma_quantizer.py` | Quantizer implementation |
| `python/tvm/relax/transform/ti_fuse_qdq_c7x_activation.py` | Group 3 fusion (gelu, silu, hardsigmoid, hardswish) |
| `python/tvm/relax/transform/ti_fuse_qdq_c7x_avgpool.py` | Group 4 fusion (avg pool) |
| `python/tvm/relax/transform/ti_fuse_qdq_c7x_layernorm.py` | Group 5 fusion (layer_norm) |
| `python/tvm/relax/transform/ti_eliminate_qdq_transparent.py` | Group 6 elimination |
| `python/tvm/relax/transform/ti_mmalib_qdq_fusion.py` | Group 1 int8 conv2d |
| `python/tvm/relax/transform/ti_mmalib_qdq_dwconv.py` | Group 1 int8 depthwise |
| `python/tvm/relax/transform/ti_mmalib_qdq_fc.py` | Group 1/2 int8 + int16 FC |
| `python/tvm/relax/transform/ti_mmalib_qdq_i16_conv.py` | Group 1 int16 conv2d |
| `python/tvm/relax/transform/ti_mmalib_qdq_i16_dwconv.py` | Group 1 int16 depthwise |
| `python/tvm/relax/transform/ti_residual_add.py` | Group 2 residual add |
| `tests/ti-dsp-runtime/pt2e-tests/` | Quantizer and E2E tests |
