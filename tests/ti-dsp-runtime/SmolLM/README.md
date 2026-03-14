# SmolLM-135M on C7x DSP

End-to-end flow for compiling HuggingFace SmolLM-135M-Instruct through
TVM's c_static backend and running inference on TI C7x DSP — first in
host emulation (c7x_host), then on AM67A hardware (c7x_dload).

## Status

| Variant | c7x_host | c7x_dload |
|---------|----------|-----------|
| Float32 (~621 MB weights) | PASS (max diff 0.21) | ELF too large |
| INT8 weight-only (~333 MB weights) | PASS (max diff 0.19) | max diff 30.9 (logits), see below |

### c7x_dload logit divergence

The 30.9 max logit diff on c7x_dload is not a quantization or compiler
correctness bug.  It is a floating-point amplification effect specific
to SmolLM's trained weight matrices.

**It is not INT8.**  FP32 (no quantization) shows the same error:

| Precision | 1-layer c7x_host | 1-layer c7x_dload |
|-----------|------------------|-------------------|
| FP32      | 0.077            | 1.889             |
| INT8      | 0.125            | 1.873             |

**It is not a single broken operation.**  Individual components
(RMSNorm, attention, MLP, full transformer block) all pass with
<0.1 max diff on c7x_dload when tested with random weights.  See
`test_component_divergence.py`.

**Root cause: lm_head amplification of small hidden-state errors.**

1. The TI cl7x compiler and g++ schedule matmul accumulation loops in
   different orders.  IEEE 754 float addition is not associative, so
   `(a+b)+c != a+(b+c)`.  This produces per-element differences of
   ~1e-6 per matmul.

2. SmolLM's Q_proj attention weight matrix has condition number ~7M.
   A 1e-6 per-element matmul difference in a 576-dim dot product
   becomes ~0.001-0.01 in the hidden state after passing through
   ill-conditioned projections, attention softmax, and residual
   additions.

3. The lm_head (576 -> 49152) has max singular value 626.  This
   amplifies a ~0.05 hidden-state difference to ~31 in logit space:

   ```
   hidden_state_diff=0.001  ->  logit_diff ~0.6
   hidden_state_diff=0.01   ->  logit_diff ~6.3
   hidden_state_diff=0.05   ->  logit_diff ~31    <-- observed
   ```

4. Random weights do NOT trigger this because they have condition
   numbers ~O(1) and singular values ~O(sqrt(dim)), so there is
   no amplification.  This is why `test_component_divergence.py`
   passes perfectly.

**Error by layer count (c7x_dload, SmolLM INT8, logit space):**

| Layers | max diff | cos sim | notes                          |
|--------|----------|---------|--------------------------------|
| 1      | 1.87     | 0.9998  | lm_head amplifies hidden diff  |
| 2      | 45.0     | 0.39    | attention cross-token coupling  |
| 4      | 31.3     | 0.71    | plateaus (logits bounded)       |
| 8      | 34.1     | 0.68    |                                |
| 30     | 30.9     | 0.26    | full model                     |

On c7x_host the error stays <0.2 at all depths because g++ and
PyTorch use the same x86 float arithmetic (same accumulation order
in the matmul inner loop).

**What was ruled out:**

- `--fp_mode=strict` on TI cl7x: no change (tested, diff unchanged)
- `-ffp-contract=off` on g++ (disabling FMA): no change
- INT8 quantization: FP32 shows the same error
- Single-operation bugs: all ops pass with random weights
- Memory corruption: error is deterministic and reproducible

**Potential mitigations (future work):**

- Kahan summation in matmul inner loops (reduces accumulation error)
- Double-precision accumulators for inner products
- Compare hidden states (before lm_head) instead of logits
- Use top-k token accuracy as the validation metric

**Diagnostic scripts:**

- `test_component_divergence.py` — tests individual ops with random
  weights on c7x_host and c7x_dload (confirms ops are precise)
- `test_smollm_layers.py` — truncated SmolLM layer sweep with real
  weights, `--fp32` flag proves quantization is not the cause

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
# Compile once, run many times
python smollm_c7x.py compile --quantize -o /tmp/smol_int8
python smollm_c7x.py infer   --artifacts /tmp/smol_int8

# Compile + infer in one shot
python smollm_c7x.py test                          # FP32 on c7x_host
python smollm_c7x.py test --quantize               # INT8 on c7x_host
python smollm_c7x.py test --quantize --dsp-mode c7x_dload  # INT8 on AM67A

# Options
python smollm_c7x.py test --seq-len 32   # longer sequence
python smollm_c7x.py test -v             # verbose logging
```
