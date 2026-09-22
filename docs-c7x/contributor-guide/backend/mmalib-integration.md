# MMALIB Integration

C wrappers that let TVM's `c_static_lib` backend offload compute-intensive
Relax ops to the C7x MMA (Matrix Multiply Accelerator) coprocessor on
AM67A (J722S) via TI's MMALIB library, plus the glue that lets the
firmware and codegen link MMALIB without also requiring TIDL. Located
at `src/runtime/ti_dsp/mmalib/`.

Target string: `c_static_lib -mcpu=c7x -mmalib=1`

## Files

| File | Purpose |
|------|---------|
| `mmalib_wrappers.{h,cpp}` | C wrappers for the 8 MMALIB kernels (conv2d/depthwise-conv2d/matmul/matmul_bias × int8/int16), linked into the C7x firmware and exported via DLOAD |
| `tidl_maxpool_wrapper.{h,cpp}` | `max_pool2d` wrapper — TIDL-backed (`c7x_int8_max_pool_tidl`) when the firmware links TIDL, falling back to the native vectorized C7x kernel (`c7x_int8_max_pool` in `kernels/c7x_pool_relu.cpp`) for `--tidl OFF` builds (e.g. BeagleY-AI) |

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
  ▼  CodeGenCStaticLib
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
  behavior). Passing `-tidl-kernels=0` in the `c_static_lib` target string
  makes it emit `call_extern("c7x_int8_max_pool", ...)` (the native
  vectorized C7x kernel in `kernels/c7x_pool_relu.cpp` -- not a scalar
  fallback; it has its own SE-based fast path for the 3x3/2x2 shapes
  models actually use, falling back to scalar only for borders and
  shapes outside that table) instead of
  `c7x_int8_max_pool_tidl`, so a model with `max_pool2d` still links
  against a no-TIDL firmware. **Every BeagleY-AI compile must pass
  `-tidl-kernels=0` explicitly** — codegen has no way to detect what the
  firmware actually linked, so without this flag `FuseQDQToTIDLMaxPool`
  emits a call to a symbol that doesn't exist there, which fails only at
  DLOAD load time on the board, not at compile time.

## QDQ Fusion Design

### Matched Pattern

Every QDQ fusion pass matches this structure (4 pattern variants per op:
with/without bias × with/without relu):

```
dequantize(data, d_scale, d_zp)
  → op(float, dequantize(weight, w_scale, w_zp=0))
  → [add(float_bias)] → [relu]
  → quantize(output, o_scale, o_zp)
```

The whole chain is replaced by one `call_extern` carrying compile-time
integer parameters — nothing is computed at inference time.

### Compile-Time Parameter Folding

```
# Int8:
weight_sum[ch]        = sum(weight[ch, :, :, :])
zp_correction[ch]      = -d_zp * weight_sum[ch]        # absorbs the input zero-point
bias_i32[ch]           = round(float_bias[ch] / (d_scale * w_scale[ch])) + zp_correction[ch]
combined_rescale[ch]   = d_scale * w_scale[ch] / o_scale
(scale_u8[ch], shift_u8[ch]) = best_uint8_approx(combined_rescale[ch])

# Int16 (symmetric only, d_zp = o_zp = 0 enforced by the check function):
bias_i64[ch]           = round(float_bias[ch] / (d_scale * w_scale[ch]))   # wider accumulator
combined_rescale[ch]   = d_scale * w_scale[ch] / o_scale
(scale_u8[ch], shift_u8[ch]) = best_uint8_approx(combined_rescale[ch])
```

`best_uint8_approx` (`_float_to_scale_shift` in `ti_mmalib_legalize.py`)
finds integers `s ∈ [0,255]`, `sh ∈ [0,31]` such that `s × 2^(-sh) ≈
combined_rescale[ch]` — this `(scale, shift)` pair is the fixed-point
requantization format MMALIB expects natively. Precision degrades with
larger reduction dimension K (error scales roughly as `√K`); this is why
int16's accumulator is widened to `int64`.

### Eligibility Constraints

A pass's check function rejects a pattern match — falling back to scalar
codegen — whenever any of the following hold:

- Weight zero-point (`w_zp`) is non-zero. MMALIB requires symmetric weight
  quantization for every op.
- Bias, if present, doesn't resolve to a compile-time constant. PT2E often
  wraps a 1D bias in `reshape(bias, (1, C, 1, 1))` before the `add`; a
  shared helper (`_resolve_constant_tensor`) unwraps `reshape` /
  `expand_dims` / `squeeze` / `astype` chains around a constant so this is
  still recognized. If it still can't resolve, the match is rejected
  outright rather than silently lowering with a dropped bias.
- Any spatial or batch dimension isn't a static, compile-time-known
  integer, or batch size N isn't 1.
- For depthwise conv2d: `groups != C_in`, kernel isn't square, kernel size
  isn't in the allowed set for that dtype, or stride/dilation are outside
  the supported range (see the [Supported Operations](#supported-operations)
  table).
- For matmul/FC: weight isn't a 2D constant tensor, or K/N aren't multiples
  of the MMA alignment for that dtype.

## Pipeline Pass Order (with `-mmalib=1`)

The MMALIB-specific stages are inserted into the general `c_static`
Relax legalization pipeline (`cpu_generic/pipeline.py`); everything else in
this sequence (relu/activation/concat/movement/pooling/layernorm QDQ
lowering) is shared with non-MMALIB compiles and unaffected by the
`mmalib` flag:

```python
if is_c7x and target.attrs.get("mmalib", False):
    passes += get_mmalib_qdq_passes()      # 8 passes: int8+int16 QDQ fusion

passes += [
    # ... general c_static QDQ-lowering passes, shared with non-MMALIB ...
    FuseQDQToInt8Conv2D(),
    EliminateQDQRoundTrip(),
    RewriteDequantize(),
]

if is_c7x and target.attrs.get("mmalib", False):
    passes += get_mmalib_i16_fc_pass()     # LegalizeMLPToMMALIBInt16 (LLM path)

passes += [FuseDequantizeMatmul()]

if is_c7x and target.attrs.get("mmalib", False):
    # No NHWC conversion — MMALIB wants NCHW throughout
    passes += [LegalizeOps(customize_legalize_map=get_mmalib_legalize_map())]
else:
    passes += [ConvertLayoutNHWC(), LegalizeOps(...)]

passes += [AnnotateTIROpPattern(), FoldConstant(), FuseOps(), FuseTIR()]

if is_c7x:
    passes += [ScheduleC7xDMATiling(l2_budget)]   # DMA tiling for remaining, non-MMALIB conv2d
```

Two further passes run later, once the graph has been lowered from Relax
into TIR (`tvm/tir/pipeline.py`, not the Relax pipeline above):
`InjectMMALIBDMA` (always active — see next section) and, after it,
`LowerL2SramAlloc`, which turns its `Allocate(scope="global.l2sram")` nodes
into concrete `tvm_l2_alloc` calls.

## L2 DMA Prefetch (`InjectMMALIBDMA`)

`InjectMMALIBDMA` is a TIR-level pass — it runs on the loop/buffer IR,
after Relax has already been lowered — invoked unconditionally for every
`c_static` compile (it's a no-op if the function contains no MMALIB
`call_extern`). It wraps each supported MMALIB call with DMA that stages
its input (and weights, when they fit) into on-chip L2 SRAM first, trading
~200-cycle DDR latency for ~32-cycle L2 latency.

**Supported kernels** (7): `mmalib_conv2d_i8`, `mmalib_conv2d_i8_grouped_loop`,
`mmalib_conv2d_i16`, `mmalib_depthwise_conv2d_i8`,
`mmalib_depthwise_conv2d_i16`, `mmalib_matmul_bias_i8`,
`mmalib_matmul_bias_i16`. The bare, bias-less `mmalib_matmul_i8`/`_i16`
calls (used by the no-bias FC variant and the weight-only LLM path) are
**not** covered — those always read from DDR directly. Residual add
(`c7x_int8_residual_add_relu` / `c7x_int16_residual_add_relu`) is a native
C7x kernel, not an MMALIB call, and is also outside this pass's scope — see
[Residual Add](#residual-add).

Injected structure, conceptually:

```
Allocate l2_guard[guard_bytes]              # prevents SE prefetch underflow
  Allocate l2_input[input_bytes]
    Allocate l2_weight[weight_bytes]        # omitted if weights don't fit L2
      tvm_dsp_dma_copy(l2_input,  ddr_input,  input_bytes)
      tvm_dsp_dma_copy(l2_weight, ddr_weight, weight_bytes)
      tvm_dsp_dma_wait(...)
      mmalib_*(l2_input, l2_weight, ...)
```

The guard allocation (`pad_top × W_in × elem_bytes` bytes, or a fixed
128-byte fallback when there's no padding) bumps the L2 bump-pointer
allocator so the input buffer doesn't start at address 0 — otherwise the
C7x streaming engine's backward-prefetch can underflow past the start of
L2. For FC/matmul_bias, the pass maps `[M, K, N]` matmul dimensions onto
the same conv-shaped DMA logic (`K→C_in`, `M→H_in`, `N→C_out`).

**Output-channel tiling for oversized weights.** When a plain, ungrouped
int8 conv2d's weight tensor doesn't fit the L2 budget but its input does,
`InjectMMALIBDMA` tiles the weight by output channel instead of skipping
L2 staging altogether: it emits a loop that DMAs one output-channel chunk
of the weight into a shared L2 buffer at a time and calls a dedicated
wrapper, `mmalib_conv2d_i8_sliced`, once per chunk. This is currently
int8-only and only for the ungrouped conv2d kernel — int16, depthwise, and
FC would each need their own sliced wrapper to get the same treatment.

## Residual Add

Residual (skip-connection) adds are handled by a separate pass,
`ti_residual_add.py`, and run on the C7x **scalar** pipeline, not the MMA
coprocessor — there's no matrix multiply to accelerate here, just a
fixed-point weighted sum. The pass matches:

```
dequantize(x, x_scale, x_zp) + dequantize(skip, skip_scale, skip_zp)
  → [relu] → quantize(out, out_scale, out_zp)
```

and replaces it with one call to `c7x_int8_residual_add_relu` or
`c7x_int16_residual_add_relu`, computing:

```
out[i] = sat(((x[i]-x_zp)*M_x + (skip[i]-skip_zp)*M_skip) >> shift + out_zp)
```

with `M_x`, `M_skip`, and `shift` derived from the quantization scales at
compile time. Because TVM's pattern matcher isn't commutative
(`add(a, b)` doesn't match `add(b, a)`), there are 4 pattern variants per
dtype — {relu, no-relu} × {operand order `add(x, skip)`, order
`add(skip, x)`} — 8 total across int8 and int16. Int16 additionally
requires every zero-point involved to be 0. This eliminates float32
intermediate computation and cuts per-layer cost from roughly 5–11M cycles
down to 100–500K cycles.

## Runtime: MMALIB Wrappers and Code Generation

Everything above describes how a Relax/TIR pass *decides* to call a
wrapper and what integer parameters it computes at compile time. This
section covers the other two ends of that decision: how it becomes literal
C source calling the wrapper by name, and what the wrapper actually does
when the DSP executes it.

### From pass to `call_extern`

Each QDQ lowering pass builds the extern call with TVM's `te.extern` /
`tir.call_extern`, inside a helper reused for every variant of that op. For
int8 conv2d (`ti_mmalib_qdq_fusion.py`), the call site picks the extern
function name based on `groups`, then hands it to a shared emitter:

```python
def _emit_conv2d(self, extern_name, name_hint, data_arg, kernel_relax, ...):
    def fcompute(ins, outs):
        args = ["int32", extern_name, ins[0].data, ins[1].data, ...,
                outs[0].data, C_in, H_in, W_in, C_out, KH, KW, ...]
        return tir.call_extern(*args)

    def te_mmalib_conv2d(data_t, weight_t, bias_t, scale_t, shift_t):
        return te.extern((1, C_out, H_out, W_out),
                          [data_t, weight_t, bias_t, scale_t, shift_t],
                          fcompute, name=name_hint, dtype="int8")

    return self.builder_.call_te(te_mmalib_conv2d, data_arg, kernel_relax, ...)
```

`extern_name` (`"mmalib_conv2d_i8"` or `"mmalib_conv2d_i8_grouped_loop"`) is
the literal C symbol that must resolve at DLOAD load time; `name_hint`
(`"mmalib_conv2d"` / `"mmalib_conv2d_grouped_loop"`) only names the
generated TIR loop variable and PrimFunc — it's deliberately a *different*
string. Reusing the extern name for `te.extern`'s own `name=` would make
the C codegen emit a local variable named `mmalib_conv2d_i8` holding the
output buffer pointer, which shadows the real function declaration and
breaks the build. Every op file (`ti_mmalib_qdq_fc.py`,
`ti_mmalib_qdq_dwconv.py`, and their int16 counterparts) follows the same
two-names split.

### `CodeGenCStaticLib`: no indirection

There's no function-pointer table or dispatch layer — `call_extern` is
printed as a literal function call by name. The base C codegen
(`src/target/source/codegen_c.cc`) handles `tir::builtin::call_extern()` by
printing the function name followed by its arguments (`PrintCallExtern`):

```cpp
void CodeGenC::PrintCallExtern(Type ret_type, String global_symbol,
                                const Array<PrimExpr>& args, bool skip_first_arg,
                                std::ostream& os) {
  os << global_symbol << "(";
  for (size_t i = skip_first_arg; i < args.size(); ++i) {
    this->PrintExpr(args[i], os);
    if (i < args.size() - 1) os << ", ";
  }
  os << ")";
}
```

So `tir.call_extern("int32", "mmalib_conv2d_i8", a0, a1, ...)` becomes
literally `mmalib_conv2d_i8(a0, a1, ...)` in the generated C.
`CodeGenCStaticLib` (`src/target/c_static_lib/codegen_c_static_lib.cc`) overrides this
method only to track VM register usage for a few unrelated AnyList
builtins, then delegates to the base implementation for everything else —
MMALIB calls pass through unmodified.

### Header visibility: `--preinclude`, not a codegen declaration

The generated file never gets an inline forward declaration for
`mmalib_conv2d_i8` from the codegen itself — `CodeGenCStaticLib` doesn't
override the (empty-by-default) `GenerateForwardFunctionDeclarations` hook
that `CodeGenCHost` uses for its own host-side extern calls. Instead, the
DSP-side build passes `--preinclude=mmalib_wrappers.h` to the TI compiler
for every generated translation unit
(`src/runtime/ti_dsp/dynmod/CMakeLists.txt`), so every generated `lib*.c`
file sees the wrapper prototypes without needing its own `#include` or
`extern` line.

### Wrapper internals: alloc → init → exec → free, every call

MMALIB kernels follow an IALG-style init/exec lifecycle, and the C++
wrappers in `mmalib_wrappers.cpp` re-run all of it — allocate, init, exec,
free — on **every single call**; there's no handle cached across calls
(despite an older doc comment in the same file suggesting otherwise — worth
treating that comment as stale). A small RAII struct keeps this safe:

```cpp
struct Workspace {
    void* ptr = nullptr;
    ~Workspace() { if (ptr) TVMBackendFreeWorkspace(1, 0, ptr); }
    void* alloc(int32_t n) { ptr = TVMBackendAllocWorkspace(1, 0, (uint64_t)n, 0, 8); return ptr; }
};
```

`mmalib_conv2d_i8`, for example, builds MMALIB's own `bufParams`/`InitArgs`
structs describing the input/weight/bias/output buffers, sizes a handle with
`MMALIB_CNN_convolveBias_row_ixX_ixX_oxX_getHandleSize()`, allocates it from
the DDR heap through `Workspace`, calls `..._init(...)`, then `..._exec(...)`
— and the handle frees itself when the wrapper returns. For strided
convolutions this alloc/init/exec cycle repeats once per output-channel
chunk. `mmalib_matmul_bias_i8` follows the same shape with a single
init/exec pair and no chunking. Every wrapper checks its pointer arguments
for NULL and returns `-1` before calling into MMALIB at all; MMALIB's own
`_init`/`_exec` status codes are then propagated straight through as the
wrapper's return value.

### Temporary buffers and NULL defaults

All scratch memory comes from `TVMBackendAllocWorkspace` (64-byte-aligned
DDR), never the stack — the file's own header calls this out as a
deliberate choice ("no stack arrays"). Two wrappers need real scratch space
beyond the MMA handle itself: `mmalib_depthwise_conv2d_i8`/`_i16` reorder
the weight tensor from its natural `[C, 1, KH, KW]` layout into MMALIB's
internal column-interleaved format into a temporary buffer, and — when the
output row stride isn't aligned to the hardware's 64-byte (i8) / 32-byte
(i16) boundary — write into a padded temporary output buffer and copy it
back row-by-row afterward.

Every wrapper that takes optional bias/scale/shift substitutes an identity
value when the caller passes `NULL` — a zeroed bias buffer, a scale of 1, a
shift of 0 — rather than branching on every call site inside MMALIB
itself. This is what lets the plain float→int16 legalize path (no
quantization parameters at all) and the QDQ paths (real per-channel
scale/shift) share the same wrapper functions.

### DLOAD symbol resolution

The 10 MMALIB wrapper functions are compiled into the C7x firmware and
exported into a static symbol table
(`src/runtime/ti_dsp/firmware/c7x/dsp/src/dyn_loader.c`):

```c
SYM(mmalib_conv2d_i8), SYM(mmalib_conv2d_i8_sliced),
SYM(mmalib_conv2d_i8_grouped_loop), SYM(mmalib_conv2d_i16),
SYM(mmalib_matmul_i8), SYM(mmalib_matmul_i16),
SYM(mmalib_depthwise_conv2d_i8), SYM(mmalib_depthwise_conv2d_i16),
SYM(mmalib_matmul_bias_i8), SYM(mmalib_matmul_bias_i16),
```

A compiled inference module (`lib0.out`) is a separately-built DLOAD ELF
that calls these as *unresolved* external symbols; TVM's DLOAD dynamic
loader resolves each one against this table at load time, against the
already-linked firmware. No MMALIB code is statically linked into the
module itself — only the header declarations (via `--preinclude`, above)
are needed to compile it. Firmware changes require a rebuild; see
`src/runtime/ti_dsp/mmalib/README.md` and the `firmware` skill for the
build procedure.

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
| `src/runtime/ti_dsp/mmalib/mmalib_wrappers.{h,cpp}` | C wrappers (10 entry points) |
| `src/runtime/ti_dsp/mmalib/tidl_maxpool_wrapper.{h,cpp}` | `max_pool2d` wrapper (TIDL-backed) |
| `src/runtime/ti_dsp/kernels/c7x_pool_relu.cpp` | Native vectorized C7x `max_pool2d` kernel (`c7x_int8_max_pool`), used for `--tidl OFF` builds |
| `src/runtime/ti_dsp/kernels/c7x_residual_add.{cpp,h}` | Fixed-point residual add kernels (int8 + int16) |
| `src/runtime/ti_dsp/dma/tvm_dsp_dma.c` | EDMA runtime (virt_to_phys for staging buffer) |
| `src/target/target_kind.cc` | `mmalib` and `tidl-kernels` target attributes |
| `src/runtime/ti_dsp/firmware/c7x/dsp/CMakeLists.txt` | `USE_TIDL_RUNTIME` / `USE_TI_MMALIB` build options |
| `src/runtime/ti_dsp/firmware/c7x/dsp/src/dyn_loader.c` | Exports 10 MMALIB symbols + guards TIDL-only symbols |
| `tests/ti-dsp-runtime/mmalib-tests/` | Unit tests (conv2d, dwconv2d, FC, residual add — i8 and i16) |
| `tests/ti-dsp-runtime/quantized/` | Full model tests (ResNet, MobileNet, GoogLeNet, ShuffleNet) |

## Testing

```bash
cd tests/ti-dsp-runtime
export TI_CGT_C7000_PATH=/opt/ti/c7x/ti-cgt-c7000_5.0.1.LTS

# Quick unit tests — all MMALIB kernels, host emulation
pytest --rootdir=. mmalib-tests/ -m quick --dsp-mode=c7x_host -v

# Full unit test suite (includes non-quick tests)
pytest --rootdir=. mmalib-tests/ -v --dsp-mode=c7x_host

# ResNet-18 int8 with MMALIB (host emulation)
pytest --rootdir=. quantized/test_quantized_resnet.py -v --dsp-mode=c7x_host --mmalib

# ResNet-18 int8 with MMALIB (AM67A hardware)
pytest --rootdir=. quantized/test_quantized_resnet.py -v \
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

## Related Documentation

- [C Static Lib Backend](c-static-lib.md) — Target config, pass ordering, DMA tiling
- [te.extern Lowering](te-extern-lowering.md) — Full compilation chain, dispatch sites
- [Authoring Passes](authoring-passes.md) — Writing new MMALIB fusion passes
- [Extending Quantizer](extending-quantizer.md) — Adding ops to C7xMMAQuantizer
- [Debugging Passes](debugging-passes.md) — --dump-ir, TVM_LOG_DEBUG, composite inspection
