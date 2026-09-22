# te.extern Pattern for C7x Kernel Dispatch

How TVM maps Relax ops to optimized C7x kernels using `te.extern` and
`tir.call_extern`. Covers every dispatch site, the full compilation
chain, and the DMA injection layer.

## The Core Pattern

Every C7x kernel dispatch follows the same four-step chain:

```
Relax op(s)
  → FuseOpsByPattern (composite)
  → PyExprMutator → builder_.call_te(te_fn, ...)
  → te.extern → tir.call_extern
  → CodeGenCStaticLib → C function call
```

`te.extern` replaces what would otherwise be a loop nest with a single
`fcompute` callback that emits one `tir.call_extern`. TVM is responsible
for tiling, DMA transfers, and double buffering via the
`InjectMMALIBDMA` and `ScheduleC7xDMATiling` passes; the external kernel
handles the inner compute (MMA coprocessor programming) for each tile.

```python
te.extern(
    output_shape,
    [input_tensors],
    lambda ins, outs: tir.call_extern("int32", "kernel_name", *args),
    name="...",
    dtype="...",
)
```

## All te.extern Dispatch Sites

| File | Line | Extern name | Inputs | Output dtype |
|------|------|-------------|--------|-------------|
| `ti_mmalib_legalize.py` | ~123 | `mmalib_matmul_i16` | `[a, b]` | int16 |
| `ti_mmalib_legalize.py` | ~386 | `mmalib_conv2d_i16` | `[data, weight]` | int16 |
| `ti_mmalib_qdq_fusion.py` | ~582 | `mmalib_conv2d_i8` | `[data, weight, bias, scale, shift]` | int8 |
| `ti_mmalib_qdq_dwconv.py` | ~523 | `mmalib_depthwise_conv2d_i8` | `[data, weight, bias, scale, shift]` | int8 |
| `ti_mmalib_qdq_fc.py` | ~425 | `mmalib_matmul_bias_i8` | `[data, weight, bias, scale, shift]` | int8 |
| `ti_mmalib_i16_fc.py` | ~251 | `mmalib_matmul_bias_i16` | `[data, weight, bias, scale, shift]` | int16 |
| `ti_residual_add.py` | ~459 | `c7x_int8_residual_add_relu` | `[x, skip, params]` | int8 |
| `fuse_dequantize_matmul.py` | ~343 | `c7x_dequantize_vecmatmul` | `[act, w, scale]` | float32 |
| `ti_fuse_sdpa_decode.py` | ~248 | `c7x_sdpa_decode` | `[q, k, v, mask]` | float32 |

All files are under `python/tvm/relax/transform/`.

## Full Compilation Chain

```
Relax IR
  ↓ FuseMMALIBQDQFC / FuseMMALIBQDQConv2d / ...
    FuseOpsByPattern marks composite: "mmalib.fc_i8_qdq_bias"
  ↓ _MMALIBQDQFCLowerer (PyExprMutator)
    - Extracts constants: weight_int8, bias_float, scales, zero_points
    - Folds quantization math at compile time → bias_i32, scale_u8, shift_u8
    - Calls builder_.call_te(te_mmalib_fc_i8, data, weight, bias, ...)
  ↓ te.extern → ExternOp → TIR PrimFunc
    T.evaluate(tir.call_extern("int32", "mmalib_matmul_bias_i8", ...))
  ↓ InjectMMALIBDMA (TIR pass)
    Wraps call_extern with L2 Allocate + DMA copy/wait if input fits
  ↓ LowerL2SramAlloc
    Allocate(scope="global.l2sram") → tvm_l2_alloc(nbytes)
  ↓ LowerDMAToExtern
    tir.dma_copy / tir.dma_wait → tir.call_extern("tvm_dsp_dma_copy", ...)
  ↓ CodeGenCStaticLib
    call_extern → C: mmalib_matmul_bias_i8(data->data, weight->data, ...)
  ↓ cl7x + lnk7x --dynamic=lib
    Resolves symbol from firmware DLOAD export table
```

## Deep Dive: FC Lowering (ti_mmalib_qdq_fc.py)

### Matched Patterns

Four DFPatterns are registered, matched in order (longest/most-specific first):

| Composite name | Pattern |
|---|---|
| `mmalib.fc_i8_qdq_reshape_bias` | dequant → reshape → matmul → reshape → add → quantize |
| `mmalib.fc_i8_qdq_reshape` | dequant → reshape → matmul → reshape → quantize |
| `mmalib.fc_i8_qdq_bias` | dequant → matmul → add → quantize |
| `mmalib.fc_i8_qdq` | dequant → matmul → quantize |

The reshape variants handle 3D inputs from `aten.linear` which
decomposes as reshape + matmul + reshape before quantize.

### Eligibility Check (`_check_mmalib_qdq_fc`)

A pattern match is rejected if any of these fail:

- `w_zp` is not a zero constant (requires zero-point-free weights)
- weight dtype is not `int8`
- data dtype is not `int8` or is a constant (weights-only not supported here)
- weight shape is not 2D `[N_out, K]`
- `K % 64 != 0` or `N_out % 64 != 0` (MMA tile alignment)
- `permute_dims` is not a standard last-two-dim transpose
- bias (if present) is not a compile-time constant

### Compile-Time Parameter Folding

The lowerer extracts all constants and folds the quantization math before
emitting `te.extern`. Nothing is computed at runtime.

```python
# Zero-point correction absorbed into bias
weight_sum = w_int8_np.astype(np.int32).sum(axis=1)      # [N_out]
zp_correction = -d_zp_val * weight_sum                    # [N_out]

# Bias converted to accumulator integer scale
dw_scale = d_scale_val * w_scale_np                       # per-channel
bias_accum = round(bias_float / dw_scale)                 # [N_out] int32
bias_i32 = bias_accum + zp_correction                     # [N_out] int32

# Output zero-point absorbed into bias too
if o_zp_val != 0:
    bias_i32 += round(o_zp_val / (dw_scale / o_scale_val))

# Requantization represented as uint8 fixed-point multiply-shift
combined_rescale = d_scale_val * w_scale_np / o_scale_val
scale_u8, shift_u8 = _float_to_scale_shift(combined_rescale)
# → kernel does: out = clip((acc * scale_u8) >> shift_u8, -128, 127)
```

### te.extern Emission

```python
def te_mmalib_fc_i8(data_t, weight_t, bias_t, scale_t, shift_t):
    def fcompute(ins, outs):
        return tir.call_extern(
            "int32",
            "mmalib_matmul_bias_i8",
            ins[0].data,   # int8 activations  [M, K]
            ins[1].data,   # int8 weights       [N_out, K]
            ins[2].data,   # int32 bias         [N_out]
            ins[3].data,   # uint8 scale        [N_out]
            ins[4].data,   # uint8 shift        [N_out]
            outs[0].data,  # int8 output        [..., N_out]
            M, K, N_out,
        )
    return te.extern(
        data_shape[:-1] + [N_out],
        [data_t, weight_t, bias_t, scale_t, shift_t],
        fcompute,
        name="mmalib_fc",           # TIR PrimFunc name (different from extern!)
        dtype="int8",
    )
```

Weight is passed in natural `[N_out, K]` layout. The wrapper accepts this
directly without a reorder step (unlike depthwise conv which reorders at runtime).

## DMA Injection for FC Layers

`InjectMMALIBDMA` (`ti_mmalib_inject_dma.py`) explicitly supports
`mmalib_matmul_bias_i8` in its `_SUPPORTED` set.

### Dimension Mapping

FC has no spatial dimensions, so `_extract_dims_fc_i8` maps
the flat `[M, K, N]` args into the shared conv-shaped slot names:

```python
# call_extern args: (name, input, weights, bias, scale, shift, output, M, K, N)
{
    "input_arg_idx":  1,   # activations pointer
    "weight_arg_idx": 2,   # weight pointer
    "c_in":  K,            # "channels in" = K
    "h_in":  M,            # "height"      = M (batch)
    "w_in":  1,
    "c_out": N,
    "kh": 1, "kw": 1,
}
```

Buffer sizes:
```
input_bytes  = K * M bytes   (activations)
weight_bytes = N * K bytes   (weights)
```

### When DMA Fires

The pass skips DMA entirely if `input_bytes > l2_budget` (default 384 KB):

| Scenario | M | K | N | input KB | weight KB | DMA fires? |
|---|---|---|---|---|---|---|
| Decode (single token) | 1 | 256 | 512 | 0.25 | 128 | Yes — both |
| Prefill seq=64 | 64 | 256 | 512 | 16 | 128 | Yes — both |
| Prefill seq=512 | 512 | 256 | 512 | 128 | 128 | Yes — both |
| Prefill seq=2048 | 2048 | 256 | 512 | 512 | 128 | **No** (>384 KB) |

Weights are only cached if `input_bytes + weight_bytes <= l2_budget`.

### Injected TIR Structure

When DMA fires, the single `call_extern` is wrapped:

```
Allocate l2_guard[int8, guard_bytes]        # SE backward-prefetch guard
  Allocate l2_input[int8, input_bytes]
    Allocate l2_weight[int8, weight_bytes]  # omitted if weights don't fit
      Evaluate(tvm_dsp_dma_copy(0, l2_input,  ddr_input,  input_bytes,  0))
      Evaluate(tvm_dsp_dma_copy(0, l2_weight, ddr_weight, weight_bytes, 0))
      Evaluate(tvm_dsp_dma_wait(0, 0))      # block until complete
      Evaluate(mmalib_matmul_bias_i8(l2_input, l2_weight, bias, scale, shift, ...))
```

The guard allocation (`pad_top * w_in * elem_bytes`, min 128 bytes)
bumps the L2 bump pointer so the input buffer does not start at address 0
of L2, preventing streaming-engine backward-prefetch underflow.

`LowerL2SramAlloc` later converts each `Allocate(scope="global.l2sram")`
to a `tvm_l2_alloc(nbytes)` call.

## TIR Pass Order (c_static -mcpu=c7x -mmalib=1)

```
FuseMMALIBQDQFC / FuseMMALIBQDQConv2d / FuseMMALIBQDQDwConv2d
FuseInt8ResidualAdd
FuseQDQToInt8Conv2D
EliminateQDQRoundTrip
FuseDequantizeMatmul
LegalizeOps
FuseOps → FuseTIR
InjectMMALIBDMA         ← wraps call_extern with L2 DMA
ScheduleC7xDMATiling
InjectSoftwarePipeline
LowerOpaqueBlock
FlattenBuffer
LowerAsyncDMA
LowerDMAToExtern        ← tir.dma_copy/wait → call_extern
NarrowDataType(32)
StorageRewrite
LowerL2SramAlloc        ← Allocate(l2sram) → tvm_l2_alloc
MakePackedAPI
CodeGenCStaticLib
```

MMALIB fusion passes run before standard QDQ passes to capture the full
`dequantize → op → quantize` graph before `FuseQDQToInt8Conv2D`
collapses it.

## Key Implementation Details

### Shared Helpers (ti_mmalib_legalize.py)

- `_float_to_scale_shift(val)` — finds `s ∈ [0,255]`, `sh ∈ [0,31]` such that
  `s × 2^(-sh) ≈ val`. Used for per-channel requantization.
- `_resolve_constant_tensor(expr)` — unwraps `reshape`/`expand_dims`/`squeeze`/`astype`
  chains around constants so bias/scale/zp are recognized even when PT2E wraps them.

### Two-Names Convention

Every lowering pass uses two distinct names:
- `extern_name`: The literal C symbol (e.g., `"mmalib_conv2d_i8"`) — must match
  firmware DLOAD export table exactly.
- `name_hint`: The `te.extern` `name=` (e.g., `"mmalib_conv2d"`) — only names
  the generated TIR PrimFunc and loop variable.

**Never reuse `extern_name` for `name_hint`** — it creates a local variable
that shadows the function declaration and breaks the build.

### Constant Folding with bind_constants

Per `fuse_ops.cc` (`lift_constants = !bind_constants`), `bind_constants=False`
**lifts** matched `relax.Constant` leaves out to become parameters of the
composite function, passed in as call-site arguments — the composite body
itself normally holds no `relax.Constant`. `bind_constants=True` (default)
leaves them bound inside the composite body instead.

Both the C7x activation passes (`FuseQDQToC7xActivation`, `FuseQDQToC7xAvgPool`,
`FuseQDQToC7xLayerNorm`) and the MMALIB QDQ passes use `bind_constants=False`,
then recover the constant values from the composite call's arguments via
`_resolve_constant_tensor` (dereferencing through `PyExprMutator.lookup_binding`
as needed) for compile-time folding — nothing here uses the `bind_constants=True`
default.

## Related Files

| File | Purpose |
|------|---------|
| `python/tvm/relax/transform/ti_mmalib_qdq_fusion.py` | Int8 conv2d lowering |
| `python/tvm/relax/transform/ti_mmalib_qdq_dwconv.py` | Int8 depthwise lowering |
| `python/tvm/relax/transform/ti_mmalib_qdq_fc.py` | Int8 + int16 FC lowering |
| `python/tvm/relax/transform/ti_mmalib_qdq_i16_conv.py` | Int16 conv2d lowering |
| `python/tvm/relax/transform/ti_mmalib_qdq_i16_dwconv.py` | Int16 depthwise lowering |
| `python/tvm/relax/transform/ti_mmalib_i16_fc.py` | Weight-only LLM int16 FC lowering |
| `python/tvm/relax/transform/ti_mmalib_legalize.py` | Shared helpers: `_float_to_scale_shift`, `_resolve_constant_tensor` |
| `python/tvm/relax/transform/ti_mmalib_inject_dma.py` | L2 DMA prefetch (TIR pass) |
| `python/tvm/relax/transform/ti_residual_add.py` | Residual add lowering |
| `python/tvm/tir/pipeline.py` | TIR pipeline wiring |
| `src/target/source/codegen_c.cc` | Base `PrintCallExtern` — turns `call_extern` into C call |
| `src/target/c_static_lib/codegen_c_static_lib.cc` | `CodeGenCStaticLib::PrintCallExtern` override, delegates to the base |
