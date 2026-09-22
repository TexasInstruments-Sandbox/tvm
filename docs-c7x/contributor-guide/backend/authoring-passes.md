# Authoring Relax Passes for C7x

This guide covers writing compiler passes for the TVM C7x backend — from Relax graph-level
transformations to TIR lowering and the C codegen interface.

## Overview

The C7x backend extends TVM's compilation pipeline with passes that:
1. **Match quantized patterns** in Relax IR (dequantize → op → quantize)
2. **Replace with integer kernel calls** via `te.extern` → `tir.call_extern`
3. **Inject DMA staging** at the TIR level for L2 SRAM prefetch
4. **Generate C code** that links against firmware-exported symbols

All MMALIB and TIDL integration passes live in `python/tvm/relax/transform/ti_*.py`.

## Pass Types and When to Use Each

| Pass Type | IR Level | Use For |
|-----------|----------|---------|
| `FuseOpsByPattern` | Relax | Pattern matching + composite function creation |
| `PyExprMutator` subclass | Relax | Lowering composites to `te.extern` / `call_extern` |
| TIR pass (`tir.transform.Pass`) | TIR | DMA injection, buffer allocation, loop transformation |
| `LegalizeOps` customization | Relax → TIR | Custom lowering for specific ops |

## FuseOpsByPattern: Pattern Matching

`FuseOpsByPattern` is the primary mechanism for identifying fusable subgraphs.
It uses TVM's `DFPattern` API to describe dataflow patterns.

### Basic Pattern Structure

```python
from tvm.relax.transform import FuseOpsByPattern
from tvm.relax.dpl.pattern import is_op, wildcard, is_constant

# Define pattern: dequantize → conv2d → quantize
pattern = is_op("relax.nn.conv2d")(
    is_op("relax.dequantize")(wildcard(), is_constant(), is_constant()),  # data
    is_op("relax.dequantize")(wildcard(), is_constant(), is_constant())   # weight
)
pattern = is_op("relax.quantize")(pattern, is_constant(), is_constant())
```

### Key DFPattern Constructors

| Function | Purpose |
|----------|---------|
| `wildcard()` | Matches any expression |
| `is_op("op_name")` | Matches a specific op; can nest children |
| `is_constant()` | Matches a constant (folded or symbolic) |
| `is_tuple_get_item(idx)` | Matches `TupleGetItem` at index |
| `is_var()` | Matches a variable (function parameter) |

### Annotations for Parameter Extraction

Add `annotation` to capture matched sub-expressions for later use:

```python
from tvm.relax.dpl.pattern import wildcard

data = wildcard()
weight = wildcard()
pattern = is_op("relax.nn.conv2d")(
    is_op("relax.dequantize")(data.annotate("data"), ...),
    is_op("relax.dequantize")(weight.annotate("weight"), ...)
)
```

In the callback, access via `matched["data"]`, `matched["weight"]`.

### Composite Function Creation

When a pattern matches, `FuseOpsByPattern` creates a `Composite` function
containing the matched subgraph. The composite has a name (e.g., `"mmalib.conv2d_i8_qdq"`)
and is registered in the module.

### The `bind_constants` Parameter

`bind_constants` controls where a matched constant ends up, per
`fuse_ops.cc` (`lift_constants = !bind_constants`):

- `bind_constants=True` (default): constants stay bound **inside** the
  composite body — the mutator sees them directly as `relax.Constant` nodes
  when it extracts matched sub-expressions.
- `bind_constants=False`: constants are **lifted out** to become parameters
  of the composite function, passed in as arguments at the call site — the
  composite body itself normally holds no `relax.Constant`.

All of the C7x QDQ-fusion passes (`FuseMMALIBQDQConv2d`, `FuseQDQToC7xMovement`,
`FuseQDQToC7xRelu`, etc.) use `bind_constants=False`, then recover the
constant values from the *call-site arguments* (via `_resolve_constant_tensor`,
which unwraps `reshape`/`expand_dims`/`squeeze`/`astype` chains PT2E wraps
around them) for compile-time parameter folding.

**The decline-branch case**: if your check function returns `False` (decline),
the matched call is not automatically consumed — it remains as a `Call` to the
composite `GlobalVar`. `FuseTIR`'s `TIRFuseMutator` fuses *every*
`Primitive`-tagged `GlobalVar` still in the module regardless of whether
anything still calls it, so a composite left un-consumed must legalize and
fuse cleanly entirely on its own. The shared `inline_declined_composite`
helper (`ti_c7x_composite_inline.py`) re-emits the composite's own body back
into the caller on decline, so the ungrouped ops go through ordinary
`LegalizeOps`/`FuseOps`/`FuseTIR` instead; the caller must still run
`relax.transform.DeadCodeElimination()(mod)` afterward so the now-orphaned
composite function is actually deleted (inlining only rewrites the call
site).

This landed in `ti_fuse_qdq_c7x_movement.py` and `ti_fuse_qdq_c7x_relu.py`
(commit `b64d0fa3bf`). It's defense-in-depth for the `TIRFuseMutator`
invariant above, not a fix for a demonstrated crash: `bind_constants=False`
lifts constants to parameters rather than embedding them, so no model on this
tree has been shown to require it (see the corrected rationale in
`ti_c7x_composite_inline.py`, commit `d1c0f41f27`). The distinct, *actually*
demonstrated decline-related hazard is `ConstReachability`'s — an all-constant
match makes `FoldConstant` host-JIT a C7x-only extern symbol and segfault
(`be39717c39`); see [C Static Lib Backend](c-static-lib.md#constreachability-and-the-decline-branch-helper).

## PyExprMutator: Lowering Composites to call_extern

After `FuseOpsByPattern` creates a composite, a `PyExprMutator` subclass
rewrites the composite call into a `te.extern` → `tir.call_extern` sequence.

### Skeleton Lowerer

```python
from tvm.relax.expr import PyExprMutator
from tvm import te, tir

class MyLowerer(PyExprMutator):
    def __init__(self, mod, builder):
        super().__init__(mod)
        self.builder_ = builder
        
    def visit_call_(self, call):
        if not isinstance(call.op, relax.GlobalVar):
            return super().visit_call_(call)
            
        gv = call.op
        composite = self.builder_.get()[gv]
        if not isinstance(composite, relax.Function) or \
           "Composite" not in composite.attrs:
            return super().visit_call_(call)
            
        composite_name = composite.attrs["Composite"]
        if composite_name != "my.target.composite":
            return super().visit_call_(call)
            
        return self._lower_composite(call, composite)
        
    def _lower_composite(self, call, composite):
        # 1. Extract matched sub-expressions via annotations
        matched = self._extract_matched(composite)
        
        # 2. Resolve constants (handle reshape/expand_dims/squeeze wrappers)
        weight = self._resolve_constant_tensor(matched["weight"])
        scale = self._resolve_constant_tensor(matched["scale"])
        # ...
        
        # 3. Compile-time parameter folding
        bias_i32, scale_u8, shift_u8 = self._fold_quant_params(...)
        
        # 4. Emit te.extern → call_extern
        return self.builder_.call_te(
            self._make_te_extern_func(bias_i32, scale_u8, shift_u8),
            matched["data"], weight, ...
        )
```

### Compile-Time Parameter Folding

Quantization parameters (scale, zero_point, bias) are folded at compile time
into integer kernel parameters. This avoids all float arithmetic at inference.

```python
def _fold_quant_params(self, d_scale, d_zp, w_scale, w_zp, o_scale, o_zp, bias):
    # Weight zero-point must be 0 (MMALIB requirement)
    assert w_zp == 0
    
    # Zero-point correction absorbed into bias
    weight_sum = weight_np.astype(np.int32).sum(axis=(1,2,3))  # per output channel
    zp_correction = -d_zp * weight_sum
    
    # Bias converted to accumulator integer scale
    dw_scale = d_scale * w_scale          # per-channel
    bias_accum = np.round(bias / dw_scale).astype(np.int32)
    bias_i32 = bias_accum + zp_correction
    
    # Output zero-point absorbed into bias
    if o_zp != 0:
        bias_i32 += np.round(o_zp / (dw_scale / o_scale)).astype(np.int32)
    
    # Requantization as uint8 fixed-point multiply-shift
    combined_rescale = d_scale * w_scale / o_scale
    scale_u8, shift_u8 = _float_to_scale_shift(combined_rescale)
    
    return bias_i32, scale_u8, shift_u8
```

### Shared Constant Resolution Helper

Use `_resolve_constant_tensor` (in `ti_mmalib_legalize.py`) to unwrap
`reshape` / `expand_dims` / `squeeze` / `astype` chains around constants:

```python
def _resolve_constant_tensor(expr):
    """Unwrap reshape/expand_dims/squeeze/astype around a constant."""
    while isinstance(expr, relax.Call):
        if expr.op.name in ("reshape", "expand_dims", "squeeze", "astype"):
            expr = expr.args[0]
        else:
            break
    if isinstance(expr, relax.Constant):
        return expr.data.numpy()
    return None
```

If resolution fails, **reject the match** rather than silently dropping bias.

### te.extern Emission

```python
def _make_te_extern_func(self, bias_i32, scale_u8, shift_u8):
    def te_kernel(data_t, weight_t, bias_t, scale_t, shift_t):
        def fcompute(ins, outs):
            return tir.call_extern(
                "int32",                    # return type
                "kernel_name",              # C symbol (must match firmware export)
                ins[0].data,                # input pointer
                ins[1].data,                # weight pointer
                ins[2].data,                # bias pointer (int32/int64)
                ins[3].data,                # scale pointer (uint8)
                ins[4].data,                # shift pointer (uint8)
                outs[0].data,               # output pointer
                *shape_args                 # M, K, N or H, W, C, KH, KW
            )
        return te.extern(
            output_shape,
            [data_t, weight_t, bias_t, scale_t, shift_t],
            fcompute,
            name="kernel_hint",             # TIR PrimFunc name (different from extern!)
            dtype="int8"
        )
    return te_kernel
```

**Important**: The `te.extern` `name` hint (`"kernel_hint"`) must be **different**
from the `call_extern` symbol (`"kernel_name"`). Reusing the extern name for
`te.extern`'s `name=` creates a local variable that shadows the function
declaration and breaks the build.

## Registering Passes in the Pipeline

All C7x-specific passes are wired in `python/tvm/relax/backend/cpu_generic/pipeline.py`.

### MMALIB Pass Registration

```python
# In pipeline.py
from tvm.relax.transform.ti_mmalib_passes import get_mmalib_qdq_passes

def _get_legalize_passes(target):
    passes = []
    
    if is_c7x and target.attrs.get("mmalib", False):
        # MMALIB QDQ fusion runs FIRST (before generic QDQ passes)
        passes += get_mmalib_qdq_passes()
    
    passes += [
        # ... generic QDQ lowering passes ...
        FuseQDQToInt8Conv2D(),
        EliminateQDQRoundTrip(),
        RewriteDequantize(),
    ]
    
    if is_c7x and target.attrs.get("mmalib", False):
        # LLM int16 path
        passes += get_mmalib_i16_fc_pass()   # LegalizeMLPToMMALIBInt16
    
    passes += [FuseDequantizeMatmul()]
    
    if is_c7x and target.attrs.get("mmalib", False):
        # Custom legalize map for float→int16 path
        passes += [LegalizeOps(customize_legalize_map=get_mmalib_legalize_map())]
    else:
        passes += [ConvertLayoutNHWC(), LegalizeOps()]
    
    passes += [AnnotateTIROpPattern(), FoldConstant(), FuseOps(), FuseTIR()]
    
    if is_c7x:
        passes += [ScheduleC7xDMATiling(l2_budget)]
    
    return passes
```

### Centralized Pass Registry

All MMALIB pass construction lives in `ti_mmalib_passes.py` via three factories:

```python
# ti_mmalib_passes.py
def get_mmalib_qdq_passes():
    """8 passes: int8 + int16 QDQ fusion for conv2d, dwconv, FC, residual add."""
    return [
        FuseMMALIBQDQConv2d(),
        FuseMMALIBQDQDwConv2d(),
        FuseMMALIBQDQFC(),
        FuseInt8ResidualAdd(),
        FuseMMALIBQDQConv2dI16(),
        FuseMMALIBQDQDwConv2dI16(),
        FuseMMALIBQDQFCI16(),
        FuseInt16ResidualAdd(),
    ]

def get_mmalib_i16_fc_pass():
    """LegalizeMLPToMMALIBInt16 — weight-only LLM path."""
    return LegalizeMLPToMMALIBInt16()

def get_mmalib_legalize_map():
    """Custom legalize map for float→int16 direct offload."""
    return {
        "nn.conv2d": _mmalib_conv2d_legalize,
        "nn.matmul": _mmalib_matmul_legalize,
    }
```

`pipeline.py` calls these factories and is never edited when new MMALIB passes
are added — this avoids scattering `target.attrs.get("mmalib")` checks.

## TIR Passes: DMA Injection

`InjectMMALIBDMA` runs after `FuseTIR` (Relax → TIR) and wraps `call_extern`
with L2 SRAM staging. See `te-extern-lowering.md` for the full TIR pass order.

## Best Practices

1. **Match longest patterns first** — register most-specific DFPattern variants
   (with bias + relu) before shorter ones (no bias, no relu).

2. **Validate aggressively in check function** — reject early if constraints
   aren't met (stride, dilation, groups, alignment, zero-points).

3. **Fold all quantization math at compile time** — nothing should compute
   scale/shift/bias at inference.

4. **Handle both operand orders** for commutative ops (`add(x, y)` vs `add(y, x)`)
   since `DFPattern` is not commutative.

5. **Use the shared constant resolution helper** — PT2E wraps bias in reshape.

6. **Run DCE after any pass that can decline** — prevents orphaned composite calls.

7. **Keep `te.extern` name and `call_extern` symbol distinct** — avoids shadowing.

## Related Files

| File | Purpose |
|------|---------|
| `python/tvm/relax/transform/ti_mmalib_passes.py` | Central pass registry |
| `python/tvm/relax/transform/ti_mmalib_qdq_fusion.py` | Int8 conv2d pattern + lowering |
| `python/tvm/relax/transform/ti_mmalib_qdq_dwconv.py` | Int8 depthwise + shared geometry check |
| `python/tvm/relax/transform/ti_mmalib_qdq_fc.py` | Int8 + int16 FC/matmul_bias |
| `python/tvm/relax/transform/ti_mmalib_qdq_i16_conv.py` | Int16 conv2d |
| `python/tvm/relax/transform/ti_mmalib_qdq_i16_dwconv.py` | Int16 depthwise |
| `python/tvm/relax/transform/ti_residual_add.py` | Residual add fusion (int8 + int16) |
| `python/tvm/relax/transform/ti_mmalib_legalize.py` | Float→int16 legalize, `_float_to_scale_shift`, `_resolve_constant_tensor` |
| `python/tvm/relax/transform/ti_mmalib_inject_dma.py` | L2 DMA prefetch (TIR) |
| `python/tvm/relax/backend/cpu_generic/pipeline.py` | Pipeline wiring |
