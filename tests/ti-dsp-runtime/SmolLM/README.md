# SmolLM-135M on C7x DSP

End-to-end flow for compiling HuggingFace SmolLM-135M-Instruct through
TVM's c_static backend and running inference on TI C7x DSP — first in
host emulation (c7x_host), then on AM67A hardware (c7x_dload).

## Status

| Variant | c7x_host | c7x_dload |
|---------|----------|-----------|
| Float32 (~621 MB weights) | PASS (max diff 0.21) | not tested |
| INT8 weight-only (~333 MB weights) | PASS (max diff 0.19) | not tested |

## Model Overview

SmolLM-135M-Instruct is a 135M-parameter LLaMA-style causal language
model (30 transformer layers, 576 hidden dim, 49152 vocab).  The float32
weights are ~621 MB which exceeds the AM67A DDR budget.  INT8
weight-only quantization reduces weights to ~333 MB (int8 linear
weights + float32 embedding/lm_head + scales + layernorm constants).

## Pipeline

### 1. Model Preparation

**SmolLMWrapper** provides explicit `position_ids` and `attention_mask`
as model inputs.  HuggingFace internally computes position IDs using
`cumsum`, which TVM cannot lower to TIR.  Passing them explicitly
eliminates cumsum from the exported graph.

### 2. Manual Per-Channel INT8 Weight Quantization

`QuantizedLinear` replaces `nn.Linear`:
- Stores `weight_int8` (int8 buffer) and `scale` (float32 buffer)
- Forward: `F.linear(x, weight_int8.float() * scale, bias)`
- Per-channel: one scale per output channel (axis=0)
- `lm_head` is kept in float32 (shared with input embedding)

The `quantize_linears()` function replaces all 210 `nn.Linear` layers
(7 per transformer layer x 30 layers) with `QuantizedLinear`.

### 3. torch.export + TVM Import

```
torch.export(wrapper, (input_ids, position_ids, attention_mask))
  -> from_exported_program(ep, keep_params_as_input=True)
  -> detach_params + BindParams
```

After BindParams, the graph contains the pattern:
```
Constant(int8) -> astype(float32) -> multiply(Constant(float32_scale))
```

### 4. RewriteDequantize Pass

**Problem:** TVM's `FoldConstant` evaluates `cast(int8) * scale` at
compile time, producing float32 constants in `weights.bin` (~621 MB
again), defeating the purpose of INT8.

**Solution:** `relax.transform.RewriteDequantize` rewrites:
```
Constant(int8) -> astype(float32) -> multiply(Constant(scale))
```
into:
```
R.dequantize(int8_const, scale_const, zero_point=0, axis=0)
```

### 5. FuseDequantizeMatmul Pass

After RewriteDequantize, the graph has:
```
R.dequantize(w_int8, scale, zp=0) -> R.permute_dims -> R.matmul(act, w_T)
  [+ R.add(bias)]
```

Without fusion, `FoldConstant` would evaluate the all-constant
`dequantize -> permute_dims` chain, expanding int8 weights back to
float32 in `weights.bin`.

`relax.transform.FuseDequantizeMatmul` fuses the entire chain into a
single TIR kernel that takes `(activation, w_int8, scale)` as inputs:

```
output[..., n] = sum_k(act[..., k] * float(w_int8[n, k])) * scale[n]
```

Since the activation is non-constant, `FoldConstant` naturally skips
the fused op.  The int8 weights and float32 scales remain as separate
constants in `weights.bin`, and the scale multiplication is factored
out of the reduction (one float mul per output element, not per MAC).

The pass runs in the standard `cpu_generic` pipeline (after
`EliminateQDQRoundTrip`, before `LegalizeOps`), so no custom pipeline
is needed.

### 6. Compilation

```python
compile_and_run_dsp(mod, input_data, execution_mode="c7x_host")
```

Uses the standard `cpu_generic` pipeline:
FuseQDQToInt8Conv2D -> EliminateQDQRoundTrip -> FuseDequantizeMatmul ->
LegalizeOps -> FoldConstant -> FuseOps -> FuseTIR ->
StaticPlanBlockMemory -> ...

## Bugs Fixed During Development

1. **strided_slice negative axis** (`legalize_ops/index.py`):
   Normalize negative axes before passing to TOPI.

2. **QDQ per-tensor axis validation** (`src/relax/op/tensor/qdq.cc`):
   Skip axis range check when scale/zp are scalars.

3. **PyTorch importer per-tensor QDQ axis** (`exported_program_translator.py`):
   Changed `axis=-1` to `axis=0` for `_quantize_per_tensor` and
   `_dequantize_per_tensor` (axis is irrelevant for scalar scale/zp).

4. **Multi-input DSP runtime** (`model.cpp`, `model.h`, `main_dsp.cpp`):
   `Model::Infer` and `Model::InferMulti` now accept arrays of
   input tensors.  SmolLM needs 3 inputs (input_ids, position_ids,
   attention_mask).

5. **DDR pool malloc** (`c7x_platform.c`):
   Changed c7x_host DDR pool from static BSS array to `malloc` to
   support larger models without bloating the binary.

## Prerequisites

```bash
export TI_CGT_C7000_PATH=~/ti/ccs2041/ccs/tools/compiler/ti-cgt-c7000_5.0.1.LTS
export PYTHONPATH=$TVM_HOME/python:$PYTHONPATH
```

Download model weights once:
```bash
python -c "
from transformers import AutoModelForCausalLM, AutoTokenizer
import torch
m = AutoModelForCausalLM.from_pretrained(
    'HuggingFaceTB/SmolLM-135M-Instruct', dtype=torch.float32)
t = AutoTokenizer.from_pretrained('HuggingFaceTB/SmolLM-135M-Instruct')
m.save_pretrained('tests/ti-dsp-runtime/SmolLM/model')
t.save_pretrained('tests/ti-dsp-runtime/SmolLM/model')
"
```

## Usage

```bash
# Float32 on c7x_host (~621 MB weights)
python smollm_c7x_host.py

# INT8 weight-only on c7x_host (~333 MB weights)
python smollm_c7x_host.py --quantize

# INT8 on AM67A hardware (not yet tested)
python smollm_c7x_host.py --quantize --dsp-mode c7x_dload

# Options
python smollm_c7x_host.py --seq-len 32   # longer sequence
python smollm_c7x_host.py -v             # verbose logging
```
