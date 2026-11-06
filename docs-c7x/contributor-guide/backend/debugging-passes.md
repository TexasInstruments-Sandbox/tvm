# Debugging C7x Compiler Passes

This guide covers tools and techniques for debugging Relax and TIR passes
in the C7x backend — from IR inspection to runtime verification.

## Inspecting IR Between Passes

TVM's pass infrastructure accepts *instruments* (`tvm.ir.instrument`) on the
`PassContext` that wraps a build. They run before and after every pass
executed inside that context, including the passes inside a `Sequential` or a
pipeline such as `get_default_pipeline`.

```python
import tvm
from tvm import relax
from tvm.ir.instrument import PassPrintingInstrument
from tvm.relax.backend.cpu_generic.pipeline import get_default_pipeline

target = tvm.target.Target("c_static_lib -mcpu=c7x -mmalib=1")
pipeline = get_default_pipeline(target)

printer = PassPrintingInstrument(
    print_before_pass_names=[],
    print_after_pass_names=["FuseMMALIBQDQConv2d", "FuseTIR"],
)

with target, tvm.transform.PassContext(opt_level=3, instruments=[printer]):
    ex = relax.build(mod, target, relax_pipeline=pipeline)
```

| Instrument | Module | What it does |
|------------|--------|--------------|
| `PassPrintingInstrument(before, after)` | `tvm.ir.instrument` | Prints the module before/after passes whose name is in the given lists |
| `PrintBeforeAll()` / `PrintAfterAll()` | `tvm.ir.instrument` | Prints the module around every pass (very verbose) |
| `DumpIR(dump_dir, refresh=False)` | `tvm.ir.instrument` | Writes the module after each pass to `dump_dir/NNN_<pass>.py`, numbered in execution order |
| `PassTimingInstrument()` | `tvm.ir.instrument` | Per-pass compile time; call `.render()` before the `PassContext` exits |
| `WellFormedInstrument()` | `tvm.relax.ir.instrument` | Checks the module is well formed before and after each pass and shows the module that broke it |

Pass names are the names TVM reports for each pass, for example the
`name="FuseMMALIBQDQConv2d"` given to `tvm.transform.module_pass`. Run once
with `PrintAfterAll()` or `DumpIR(...)` to see the exact names in your
pipeline.

`DumpIR` is the easiest way to find the pass that changed something: dump a
good and a bad build to two directories and `diff -r` them.

### Printing a module by hand

```python
print(mod.script())   # TVMScript text
mod.show()            # same, with syntax highlighting where available
```

To bisect a single pass, call it directly on the module between two pipeline
stages and print the result, or check it with
`relax.analysis.well_formed(mod)` (raises on the first violation;
`relax.analysis.check_well_formed(mod)` returns a bool).

## Python Logging from the C7x Passes

Each C7x pass module creates `logger = logging.getLogger(__name__)` and emits
`logger.debug` / `logger.info` records (fusion counts, DMA injection choices,
reasons for declining). They are silent until the application configures
Python logging:

```python
import logging

logging.basicConfig(level=logging.INFO)

# or one module at debug level
logging.getLogger("tvm.relax.transform.ti_mmalib_inject_dma").setLevel(logging.DEBUG)
```

The logger name is the module path, for example:

- `tvm.relax.transform.ti_mmalib_qdq_fusion`
- `tvm.relax.transform.ti_mmalib_inject_dma`
- `tvm.relax.transform.schedule_c7x_dma`
- `tvm.tirx.transform.lower_l2sram_alloc`

New passes should follow the same `getLogger(__name__)` pattern;
`ti_mmalib_qdq_fusion.py` is an example.

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
composite name. Check `mod.functions` for TIR functions, and print one with
`func.script()`.

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

The test harness (`tests/ti-dsp-runtime/dsp-cpp/dsp_utils.py`) deletes its
temporary build directory after a test. Set `DSP_KEEP_TEMP=1` when running a
test to keep the generated `lib0.c`, `lib1.c`, `weights.bin` and build
artifacts:

```bash
DSP_KEEP_TEMP=1 pytest --rootdir=. dsp-tests/test_conv2d_dsp.py --dsp-mode=c7x_host
# Artifacts in /tmp/dsp_test_<name>_<timestamp>/
```

The variable is read by the test helpers, not by `relax.build`, so it has no
effect on a standalone compile script.

### Finding Unfused QDQ Nodes

Search for unfused quantize/dequantize in generated C:

```bash
grep -n "quantize_per_tensor\|dequantize" /tmp/dsp_test_*/lib0.c
```

Each call site is an unfused QDQ boundary — overhead that a fusion pass
should have eliminated.

### Finding DMA Injection

```bash
grep -n "tvm_dsp_dma_copy\|tvm_dsp_dma_wait" /tmp/dsp_test_*/lib0.c
```

Shows which kernels got L2 SRAM staging.

## Layer Profiling

Compile with `-profile-layers=1` and run on hardware or emulation:

```python
target = tvm.target.Target("c_static_lib -mcpu=c7x -profile-layers=1")
```

The generated code records a cycle count per layer and prints a table through
the DSP printf (shared-memory trace buffer):

```
===== TVM Layer Profile =====
[  0] <layer name>                               1245832 cycles
[  1] <layer name>                                 12345 cycles
...
Total: <N> cycles (<M> layers)
```

A layer that was running when a call failed is printed as
`[FAILED, in flight] <layer name>`. Look for `quantize_per_tensor` /
`dequantize` layers — their cycle counts are pure overhead that a fusion pass
would eliminate.

## Common Debugging Scenarios

### Pass Not Firing

1. **Check pattern specificity** — Is the DFPattern too strict/loose?
   Add logging in the check function to see what's matched vs rejected.

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
   - `-1` = NULL pointer argument or failed handle allocation
   - `MMALIB_ERR_NOT_IMPLEMENTED` = unsupported geometry (e.g., int16 5×5 dwconv)

3. **Numerical mismatch** — Compare against PyTorch reference:
   ```bash
   pytest --rootdir=. pt2e-tests/test_c7x_mma_quantizer_e2e_dsp.py::test_e2e_conv2d_i8 -v --dsp-mode=c7x_dload
   ```

## Debugging TIR Passes

The C7x TIR pipeline is `_c7x_dma_tir_pipeline` in
`python/tvm/tirx/compilation_pipeline.py`. The instruments above work on it
the same way: pass `PassPrintingInstrument` or `DumpIR` in the `PassContext`
around `relax.build`, using the TIR pass names from the table below.

To print a single PrimFunc from inside a pass or a test:

```python
for gv, func in mod.functions_items():
    if isinstance(func, tvm.tirx.PrimFunc):
        print(f"=== {gv.name_hint} ===")
        print(func.script())
```

### Key TIR Passes to Inspect

| Pass | What to Check |
|------|---------------|
| `InjectMMALIBDMA` | L2 Allocate nodes added, DMA copy/wait inserted, guard allocation present |
| `LowerL2SramAlloc` | `AllocBuffer(scope="global.l2sram")` → `tvm_l2_alloc` calls |
| `LowerDMAToExtern` | `tirx.dma_copy`/`tirx.dma_wait` → `call_extern("tvm_dsp_dma_copy", ...)` |
| `LowerAsyncDMA` | Copy loops → `tirx.dma_copy` intrinsics (requires int64 indices) |
| `ScheduleC7xDMATiling` | `cache_read` into `global.l2sram`, software pipeline annotations |

`ScheduleC7xDMATiling` and `InjectMMALIBDMA` are Relax-side transforms
(`relax.transform`); `LowerL2SramAlloc` and `LowerDMAToExtern` are in
`tirx.transform`; `LowerAsyncDMA` is in `s_tir.transform`.

### Software Pipeline Annotations

`ScheduleC7xDMATiling` annotates the outer loop with these attributes
(`schedule_c7x_dma.py`); check they are present on the tiled loop:

```python
# Conv2d H-tiling: stage 0 = DMA (async), stage 1 = compute
attrs = {
    "software_pipeline_stage": [0, 1],
    "software_pipeline_order": [0, 1],
    "software_pipeline_async_stages": [0],
}
# The output-channel tiling variant uses stage [0, 0, 1] and order [0, 1, 2]
```

## Debugging Tips

1. **Start with unit tests** — `pt2e-tests/test_c7x_mma_quantizer.py` runs without hardware

2. **Use host emulation first** — `--dsp-mode=c7x_host` runs on x86 with memcpy DMA

3. **Compare IR at each stage** — Use `DumpIR` and `diff -r` two runs

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
| `python/tvm/tirx/compilation_pipeline.py` | TIR pipeline (InjectMMALIBDMA wiring) |
| `python/tvm/relax/transform/ti_mmalib_inject_dma.py` | DMA injection logic |
| `python/tvm/relax/transform/ti_mmalib_qdq_fusion.py` | Example pass with logging |
| `tests/ti-dsp-runtime/pt2e-tests/` | Quantizer and E2E test patterns |
