# Debugging C7x Compiler Passes

This guide covers tools and techniques for debugging Relax and TIR passes
in the C7x backend — from IR inspection to runtime verification.

## IR Dumping Flags

### `--dump-ir` (Relax)

Prints the Relax IRModule at each pass. Usage:

```python
import tvm
from tvm import relax

target = tvm.target.Target("c_static_lib -mcpu=c7x -mmalib=1")
mod = relax.transform.LegalizeOps()(mod)

# Enable IR dumping
with tvm.transform.PassContext(opt_level=3, config={"relay.dump_ir": True}):
    ex = relax.build(mod, target=target)
```

Or via environment variable:
```bash
TVM_LOG_DEBUG=1 python compile_model.py 2>&1 | grep -A 50 "After FuseMMALIBQDQConv2d"
```

### Pass-Specific Dumping

Individual passes can dump their input/output IR by adding logging:

```python
# In a pass's transform_module() or apply() method
import logging
logger = logging.getLogger("TVM.pass.my_pass")
logger.setLevel(logging.DEBUG)

logger.debug("Input module:\n%s", mod.script())
# ... pass logic ...
logger.debug("Output module:\n%s", mod.script())
```

Then run with:
```bash
TVM_LOG_DEBUG=pass.my_pass python compile_model.py
```

### TIR Dumping

After `FuseTIR` lowers to TIR, dump TIR PrimFuncs:

```python
# In TIR pass
func = ...  # tir.PrimFunc
print(func.script())  # Human-readable TIR
# Or with annotations
print(tvm.script.printer.ir_printer.IRPrinter()(func))
```

## TVM_LOG_DEBUG Categories

Key debug categories for C7x development:

| Category | What It Logs |
|----------|--------------|
| `TVM.pass.fuse_ops_by_pattern` | Pattern matches, composites created, decline reasons |
| `TVM.pass.mmalib_qdq_fusion` | QDQ pattern matches, eligibility check results, folded params |
| `TVM.pass.mmalib_inject_dma` | DMA injection decisions, buffer sizes, L2 budget checks |
| `TVM.pass.schedule_c7x_dma` | Tiling strategy chosen, tile sizes, software pipeline annotations |
| `TVM.tir.lower_dma_to_extern` | DMA intrinsic → call_extern conversions |
| `TVM.codegen.c_static` | Function emission, wrapper generation, symbol resolution |

Enable multiple:
```bash
TVM_LOG_DEBUG="TVM.pass.mmalib_qdq_fusion,TVM.pass.mmalib_inject_dma" python compile_model.py
```

## Inspecting Composite Functions

### After FuseOpsByPattern

Composites appear as `GlobalVar` with `"Composite"` attribute:

```python
# In a PyExprMutator lowerer
def visit_call_(self, call):
    if isinstance(call.op, relax.GlobalVar):
        gv = call.op
        func = self.builder_.get()[gv]
        if "Composite" in func.attrs:
            composite_name = func.attrs["Composite"]
            print(f"Found composite: {composite_name}")
            print(f"Composite body:\n{func.script()}")
```

### After FuseTIR

Composites are lowered to `PrimFunc` with `global_symbol` matching the
composite name. Check `mod.functions` for TIR functions.

### Decline Branches: inline_declined_composite

If a pass's check function returns `False`, the matched call is not
automatically consumed — it remains as a `Call` to the composite `GlobalVar`.
`FuseTIR`'s `TIRFuseMutator` fuses *every* `Primitive`-tagged `GlobalVar`
still in the module regardless of whether anything still calls it, so a
composite left un-consumed must legalize and fuse cleanly entirely on its own.

**Defense**: call the shared `inline_declined_composite` helper
(`ti_c7x_composite_inline.py`) from the decline branch, then run
`relax.transform.DeadCodeElimination()(mod)` afterward so the now-orphaned
composite function is actually deleted. Landed in `ti_fuse_qdq_c7x_movement.py`
and `ti_fuse_qdq_c7x_relu.py` (commit `b64d0fa3bf`).

**Not a demonstrated crash on this tree**: `bind_constants=False` lifts
matched constants to composite *parameters* rather than embedding them in the
body, and a declined composite left in place has been measured to compile
cleanly through `LegalizeOps`/`FoldConstant`/`FuseOps`/`FuseTIR` (corrected
rationale in `ti_c7x_composite_inline.py`, commit `d1c0f41f27`). If you hit
`FuseTIR` error `"Relax.Constant is not supported in primitive functions"`,
capture the stack trace first — the one demonstrated cause on this tree was
an unrelated stale parameter index in `fuse_ops.cc`'s tuple-parameter
splicing (`6fe9a33f09`), not a decline branch. See
[C Static Lib Backend](c-static-lib.md#constreachability-and-the-decline-branch-helper)
for the distinct, actually-demonstrated decline-related hazard
(`ConstReachability`, `be39717c39`).

## DSP_KEEP_TEMP: Preserving Generated Artifacts

Set `DSP_KEEP_TEMP=1` to preserve the generated `lib0.c`, `lib1.c`,
`weights.bin`, and intermediate build artifacts:

```bash
DSP_KEEP_TEMP=1 python compile_model.py
# Artifacts in /tmp/tvm_<pid>/
```

### Finding Unfused QDQ Nodes

Search for unfused quantize/dequantize in generated C:

```bash
grep -n "quantize_per_tensor\|dequantize" /tmp/tvm_*/lib0.c
```

Each call site is an unfused QDQ boundary — overhead that a fusion pass
should have eliminated.

### Finding DMA Injection

```bash
grep -n "tvm_dsp_dma_copy\|tvm_dsp_dma_wait" /tmp/tvm_*/lib0.c
```

Shows which kernels got L2 SRAM staging.

## Layer Profiling

Compile with `-profile-layers=1` and run on hardware/emulation:

```python
target = tvm.target.Target("c_static_lib -mcpu=c7x -profile-layers=1")
```

Output (via DSP printf to shared memory buffer):

```
[LAYER] conv2d_0: 1245832 cycles
[LAYER] relu_0: 12345 cycles
[LAYER] mmalib_conv2d_i8: 477234 cycles
[LAYER] quantize_per_tensor: 89234 cycles  <-- UNFUSED QDQ OVERHEAD
```

Look for `quantize_per_tensor` / `dequantize` entries — their cycle counts
are pure overhead that a fusion pass would eliminate.

## Common Debugging Scenarios

### Pass Not Firing

1. **Check pattern specificity** — Is the DFPattern too strict/loose?
   Add logging in check function to see what's matched vs rejected.

2. **Check pass ordering** — MMALIB passes must run BEFORE `FuseQDQToInt8Conv2D`
   and `EliminateQDQRoundTrip`. Verify in `pipeline.py`.

3. **Check bind_constants** — `bind_constants=False` lifts matched constants
   out to composite *parameters* (call-site arguments); `bind_constants=True`
   (default) leaves them bound inside the composite body. The C7x/MMALIB QDQ
   passes all use `False` and recover values via `_resolve_constant_tensor`.
   Separately, a 0-d scalar constant (e.g. a per-tensor scale) represented as
   a TE tensor inside a `te.compute`/`te.extern` callback is indexed as
   `scale[()]`, not `scale[indices[axis]]` — see `fuse_dequantize_matmul.py`.

4. **Check constant resolution** — PT2E wraps bias in `reshape(bias, (1,C,1,1))`.
   Use `_resolve_constant_tensor` to unwrap.

### Incorrect Codegen

1. **Check te.extern name vs call_extern symbol** — Must be different!
   Same name creates local variable shadowing function declaration.

2. **Check extern symbol matches firmware export** — Symbol must be in
   `dyn_loader.c` export table exactly.

3. **Check parameter order** — `call_extern` args must match wrapper signature.

### Runtime Failures

1. **DLOAD symbol resolution error** — Missing symbol in firmware export table.
   Check `dyn_loader.c` for the symbol.

2. **MMALIB error codes** — Wrapper returns MMALIB status. Check:
   - `-1` = NULL pointer argument
   - `MMALIB_ERR_NOT_IMPLEMENTED` = unsupported geometry (e.g., int16 5×5 dwconv)

3. **Numerical mismatch** — Compare against PyTorch reference:
   ```bash
   pytest --rootdir=. pt2e-tests/test_c7x_mma_quantizer_e2e_dsp.py::test_e2e_conv2d_i8 -v --dsp-mode=c7x_dload
   ```

## Debugging TIR Passes

### Print TIR Before/After Pass

```python
# In TIR pass transform function
def transform(mod, ctx):
    for gv, func in mod.functions.items():
        if isinstance(func, tir.PrimFunc):
            print(f"=== BEFORE {pass_name} ===")
            print(func.script())
    
    # ... pass logic ...
    
    for gv, func in mod.functions.items():
        if isinstance(func, tir.PrimFunc):
            print(f"=== AFTER {pass_name} ===")
            print(func.script())
    return mod
```

### Key TIR Passes to Inspect

| Pass | What to Check |
|------|---------------|
| `InjectMMALIBDMA` | L2 Allocate nodes added, DMA copy/wait inserted, guard allocation present |
| `LowerL2SramAlloc` | `Allocate(scope="global.l2sram")` → `tvm_l2_alloc` calls |
| `LowerDMAToExtern` | `tir.dma_copy`/`tir.dma_wait` → `call_extern("tvm_dsp_dma_copy", ...)` |
| `LowerAsyncDMA` | Copy loops → `tir.dma_copy` intrinsics (requires int64 indices) |
| `ScheduleC7xDMATiling` | `cache_read` into `global.l2sram`, software pipeline annotations |

### Software Pipeline Annotations

Check for these attributes on the outer loop:

```python
# Expected annotations on tiled loop
attrs = {
    "software_pipeline_stage": [0, 0, 1],  # DMA, DMA, compute
    "software_pipeline_order": [0, 1, 2],
    "software_pipeline_async_stages": [0],
}
```

## Debugging Tips

1. **Start with unit tests** — `pt2e-tests/test_c7x_mma_quantizer.py` runs without hardware

2. **Use host emulation first** — `--dsp-mode=c7x_host` runs on x86 with memcpy DMA

3. **Compare IR at each stage** — Dump after each pass in the pipeline

4. **Check pass dependencies** — `CanonicalizeBindings` + `DCE` must run before
   QDQ passes to clean up PT2E tuple artifacts

5. **Verify constant folding** — All scale/shift/bias math should be compile-time.
   No float ops should reach the generated C for fused kernels.

6. **Check `bind_constants` is `False`** — If constants aren't folding, verify
   the pattern uses `bind_constants=False` so matched constants are lifted to
   composite call-site arguments (where `_resolve_constant_tensor` expects to
   find them), not left bound inside the composite body.

## Related Files

| File | Purpose |
|------|---------|
| `python/tvm/relax/transform/ti_mmalib_passes.py` | Central registry, pass ordering |
| `python/tvm/relax/backend/cpu_generic/pipeline.py` | Full pipeline wiring |
| `python/tvm/tir/pipeline.py` | TIR pipeline (InjectMMALIBDMA, ScheduleC7xDMATiling) |
| `python/tvm/relax/transform/ti_mmalib_inject_dma.py` | DMA injection logic |
| `python/tvm/relax/transform/ti_mmalib_qdq_fusion.py` | Example pass with logging |
| `tests/ti-dsp-runtime/pt2e-tests/` | Quantizer and E2E test patterns |
