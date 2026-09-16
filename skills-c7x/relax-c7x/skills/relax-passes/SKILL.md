---
name: relax-passes
description: "Writing TVM Relax and TIR passes for C7x. Use when: implementing a new compiler pass (pattern matching, fusion, legalization, lowering), using DFPattern API (wildcard, is_op, annotations, constraint callbacks), writing PyExprMutator-based lowering, emitting call_extern via te.extern/call_te, constructing TIR PrimFuncs, injecting DMA/buffer operations in TIR, wiring passes into pipeline.py, or debugging pass ordering. NOT for using existing passes (see cstatic) or operator kernel implementation (see dsp-ops)."
---

# Writing Relax and TIR Passes

Patterns and API for implementing compiler passes in this TVM codebase.

## Before Writing a Custom Pass

Prefer leveraging existing TVM/Relax infrastructure and optimization passes to the maximum extent. Custom passes add maintenance burden and can break when TVM upstream changes. Only write a custom pass when:

1. **No existing pass handles the pattern** — check `relax.transform.*` (FuseOps, FuseTIR, LegalizeOps, FoldConstant, DeadCodeElimination, etc.) first
2. **LegalizeOps custom map is insufficient** — if a single op needs a different lowering, a legalization function is cheaper than a full pass
3. **The transformation requires cross-op context** — e.g. QDQ parameter extraction across multiple nodes, or fusing a specific multi-op pattern into one extern call

If the goal is just scheduling or tiling an existing TIR kernel, use the TE schedule API (`cache_read`, `split`, `reorder`, `compute_at`) rather than a TIR pass.

## Pass Types

| Type | Decorator | Signature | Use Case |
|------|-----------|-----------|----------|
| Module pass (Relax) | `@tvm.transform.module_pass(opt_level=0, name="X")` | `transform_module(self, mod, ctx) -> IRModule` | Pattern matching, fusion, lowering |
| PrimFunc pass (TIR) | `@tvm.tir.transform.prim_func_pass(opt_level=0, name="X")` | `fn(func, mod, ctx) -> PrimFunc` | Buffer manipulation, DMA injection |

## Pattern 1: FuseOpsByPattern + PyExprMutator Lowering

The most common pattern in this codebase. Two phases: match → lower.

### Phase 1: Pattern Matching

```python
from tvm.relax.dpl import wildcard, is_op

def _my_pattern():
    data = wildcard()
    weight = wildcard()
    scale = wildcard()
    dq = is_op("relax.dequantize")(data, scale, wildcard())
    matmul = is_op("relax.matmul")(dq, weight)
    quant = is_op("relax.quantize")(matmul, wildcard(), wildcard())

    annotations = {"data": data, "weight": weight, "scale": scale, "matmul": matmul}
    return quant, annotations, _check_constraints

def _check_constraints(ctx) -> bool:
    """Constraint callback — return False to reject match."""
    w = ctx.annotated_expr["weight"]
    if not hasattr(w, "struct_info"):
        return False
    shape = w.struct_info.shape
    return int(shape[-1]) % 64 == 0  # dimension alignment
```

Apply patterns:
```python
from tvm import relax

patterns = [
    ("mypkg.op_variant_a", *_variant_a_pattern()),
    ("mypkg.op_variant_b", *_variant_b_pattern()),
]
# Order: longest/most-specific first (greedy matching)
mod = relax.transform.FuseOpsByPattern(patterns, bind_constants=False)(mod)
```

### Phase 2: Lowering with PyExprMutator

```python
from tvm.relax.expr_functor import mutator, PyExprMutator

@mutator
class _MyLowerer(PyExprMutator):
    def __init__(self, mod):
        super().__init__(mod)
        self.mod = mod
        self.count = 0

    def visit_call_(self, call: relax.Call):
        # Only intercept calls to composite functions
        if not isinstance(call.op, relax.GlobalVar):
            return super().visit_call_(call)

        func = self.mod[call.op]
        composite = func.attrs.get("Composite", "") if func.attrs else ""
        if not composite.startswith("mypkg."):
            return super().visit_call_(call)

        # Map params to args
        arg_map = dict(zip(func.params, call.args))

        # Walk bindings to extract named values
        data = self._resolve(func, arg_map, "data")
        weight = self._resolve(func, arg_map, "weight")

        # Compute constants (numpy)
        w_np = weight.data.numpy()
        bias_np = compute_bias(w_np)
        bias_const = relax.Constant(tvm.nd.array(bias_np))

        # Emit replacement via te.extern
        result = self.builder_.call_te(
            _te_mmalib_call, data, weight, bias_const,
            primfunc_name_hint="mmalib_my_op",
        )
        self.count += 1
        return result
```

### te.extern for call_extern

```python
from tvm import te, tir

def _te_mmalib_call(data, weight, bias):
    N, K = data.shape[0], weight.shape[1]
    return te.extern(
        [N, K],  # output shape
        [data, weight, bias],  # inputs
        lambda ins, outs: tir.call_extern(
            "int32",  # return type
            "mmalib_my_op",  # extern function name
            ins[0].data, ins[1].data, ins[2].data,
            outs[0].data,
            N, K,  # dimension args
        ),
        name="mmalib_my_op",
        dtype="int8",  # output dtype
    )
```

### Putting it together (module pass)

```python
@tvm.transform.module_pass(opt_level=0, name="FuseMyOp")
class FuseMyOp:
    def transform_module(self, mod: IRModule, _ctx: PassContext) -> IRModule:
        # Phase 1: Pattern match
        patterns = [("mypkg.variant_a", *_variant_a_pattern())]
        mod = relax.transform.FuseOpsByPattern(patterns, bind_constants=False)(mod)

        # Phase 2: Lower composites
        lowerer = _MyLowerer(mod)
        for gv, func in mod.functions_items():
            if isinstance(func, relax.Function):
                new_func = lowerer.visit_expr(func)
                lowerer.builder_.update_func(gv, new_func)
        mod = lowerer.builder_.get()

        # Cleanup
        if lowerer.count > 0:
            mod = relax.transform.DeadCodeElimination()(mod)
        return mod
```

## Pattern 1b: FuseOpsByPattern + MergeCompositeFunctions (Subgraph Offload)

`FuseOpsByPattern` is the general mechanism for identifying a sequence of ops to offload — whether to an optimized C7x custom kernel, MMALIB, or TIDL. When multiple adjacent matched composites should be handled as a single unit, `MergeCompositeFunctions` groups them into a subgraph function:

```python
@tvm.transform.module_pass(opt_level=0, name="PartitionForBackend")
class PartitionForBackend:
    def transform_module(self, mod, _ctx):
        # Phase 1: Mark individual ops/sequences as composites
        patterns = [
            ("mybackend.conv2d_bias_relu", *_conv2d_bias_relu_pattern()),
            ("mybackend.relu", *_relu_pattern()),
            ...
        ]
        mod = relax.transform.FuseOpsByPattern(patterns, bind_constants=False)(mod)

        # Phase 2: Merge adjacent composites into subgraph functions.
        # MergeCompositeFunctions takes no arguments — it infers the codegen
        # name from each composite's pattern-name prefix (e.g.
        # "mybackend.conv2d_bias_relu" -> "mybackend").
        mod = relax.transform.MergeCompositeFunctions()(mod)
        # Result: functions with attrs["Codegen"] = "mybackend" containing
        # multiple composites that form a connected subgraph
        return mod
```

**When to use Pattern 1 vs 1b:**
- **Pattern 1** (FuseOpsByPattern + Mutator): Each matched pattern is independently lowered to a `call_extern`. Use when TVM controls scheduling and each fused op is a separate extern call (e.g. MMALIB conv2d, custom vectorized kernels).
- **Pattern 1b** (FuseOpsByPattern + MergeCompositeFunctions): Multiple matched patterns are grouped into subgraph functions for external compilation. Use when an external runtime handles the entire subgraph — scheduling, memory, multi-op fusion (e.g. TIDL, or any custom backend that processes a connected graph rather than individual ops).

After `MergeCompositeFunctions`, the subgraph functions are typically processed by a custom compiler (e.g. `TIDLOffloadCompiler.tidl_import()`) that replaces them with `call_extern` stubs pointing to the external runtime.

### Non-composite bridge and cyclic subgraph merging (fixed)

`MergeCompositeFunctions` had a cycle-detection gap: if a non-composite op B sat between
two composite groups (G1 → B → G2, where G2 also had a direct skip-connection arg from G1),
the algorithm could merge G1 and G2 into one subgraph.  The local merge check only looked
at G1's current `group_deps_` set — which was empty when no composite had yet consumed G1's
output — so the skip-connection arg passed unchecked.  The result was a cyclic module that
crashed `relax.build` with SSA ordering violations or a stale DataflowVar reference.

**Fix (implemented):** `UpdateGroupDependencies` in
`src/relax/transform/merge_composite_functions.cc` now injects a self-dep into the upstream
composite group whenever a non-composite group reads from it:

```cpp
// When a non-composite group consumes a composite group's output, mark the
// composite as "closed" so future skip-connection merges are blocked.
if (!GetCodegenName(group_root) && GetCodegenName(arg_group_root)) {
    group_deps_[arg_group_root].insert(arg_group_root);
}
```

This makes the upstream composite appear in `parent_dependencies` for any later composite
that takes it as a direct arg, causing `GetGroupsToMerge` to reject the merge.

Regression tests in `tests/ti-dsp-runtime/tidl-tests/test_tidl_partition.py`
(`TestNonCompositeBridgeCycle`).  Full analysis in
`docs/dsp/merge_composites_avoid_cycles.md`.

## Pattern 2: Direct Mutator (No DFPattern)

When the pattern is a simple linear chain, skip `FuseOpsByPattern` and match manually in `visit_call_`:

```python
@mutator
class _DirectMutator(PyExprMutator):
    def __init__(self, mod):
        super().__init__(mod)
        self._bindings = {}  # var -> value map

    def visit_call_(self, call: relax.Call):
        if call.op.name != "relax.matmul":
            return super().visit_call_(call)

        # Trace backward through bindings
        lhs = self._resolve_var(call.args[0])
        if not (isinstance(lhs, relax.Call) and lhs.op.name == "relax.dequantize"):
            return super().visit_call_(call)

        # ... validate and emit replacement
```

Pre-scan bindings (required for backward tracing):
```python
def _pre_scan_bindings(func):
    """Build var -> value map from function body."""
    var_map = {}
    for block in func.body.blocks:
        for binding in block.bindings:
            if isinstance(binding, relax.VarBinding):
                var_map[binding.var] = binding.value
    return var_map
```

## Pattern 3: TIR PrimFunc Pass

For post-lowering manipulation (buffer injection, DMA):

```python
def InjectMyTransform(param=128):
    @tvm.tir.transform.prim_func_pass(opt_level=0, name="InjectMyTransform")
    def _pass(func: tir.PrimFunc, mod, ctx):
        return _transform(func, param)
    return _pass

def _transform(func, param):
    body = func.body

    # Find target statement via post_order_visit
    target = None
    def _find(stmt):
        nonlocal target
        if isinstance(stmt, tir.Evaluate) and _is_target_call(stmt.value):
            target = stmt
    tir.stmt_functor.post_order_visit(body, _find)

    if target is None:
        return func

    # Build new statements
    l2_var = tir.Var("l2_buf", PointerType(PrimType("int8"), "global.l2sram"))
    dma_copy = tir.Evaluate(tir.call_extern("int32", "tvm_dsp_dma_copy", ...))
    dma_wait = tir.Evaluate(tir.call_extern("int32", "tvm_dma_wait", ...))

    # Compose: alloc -> dma_copy -> dma_wait -> original_call
    new_body = tir.SeqStmt([dma_copy, dma_wait, target])
    new_body = tir.Allocate(l2_var, "int8", [extent], tir.const(1), new_body)

    return func.with_body(new_body)
```

## Pattern 4: Custom Legalization

Hook into `LegalizeOps` with a custom map:

```python
def _legalize_matmul(bb: relax.BlockBuilder, call: relax.Call):
    """Replace relax.matmul with custom implementation."""
    if not _is_eligible(call):
        return call  # fall through to default
    # ... emit replacement
    return bb.call_te(_te_impl, *call.args)

# In pipeline:
relax.transform.LegalizeOps(customize_legalize_map={
    "relax.matmul": _legalize_matmul,
    "relax.nn.conv2d": _legalize_conv2d,
})
```

## Wiring into pipeline.py

Location: `python/tvm/relax/backend/cpu_generic/pipeline.py`

```python
def legalize_passes(target):
    passes = []
    is_c7x = target.kind.name == "c_static_lib" and str(target.attrs.get("mcpu", "")) == "c7x"
    use_mmalib = target.attrs.get("mmalib", False)

    if is_c7x and use_mmalib:
        passes.append(FuseMyOp())  # insert before or after existing passes

    passes.append(relax.transform.FuseOps())
    passes.append(relax.transform.FuseTIR())
    return passes
```

Rules:
- MMALIB QDQ passes run BEFORE `FuseQDQToInt8Conv2D` (to see intact QDQ graph)
- Pattern passes ordered longest-match-first
- `FoldConstant()` after any pass that creates new constants
- `DeadCodeElimination()` is implicit in most passes (each pass cleans up after itself)

## Key API Reference

| API | Purpose |
|-----|---------|
| `wildcard()` | Match any expression |
| `is_op("relax.nn.conv2d")(arg1, arg2)` | Match specific op with args |
| `ctx.annotated_expr["name"]` | Access matched sub-expression in check callback |
| `relax.transform.FuseOpsByPattern(patterns, bind_constants=False)` | Phase 1: match and group into composites |
| `relax.transform.MergeCompositeFunctions()` | Merge adjacent composites into subgraph functions (codegen name inferred from each composite's pattern-name prefix) |
| `@mutator` + `PyExprMutator` | Phase 2: rewrite matched composites |
| `self.builder_.call_te(fn, *args, primfunc_name_hint="X")` | Emit te function as TIR |
| `self.builder_.emit(relax.op.abs(x))` | Emit Relax op into current block |
| `te.extern(shape, inputs, fcompute, name, dtype)` | Create call_extern tensor expression |
| `tir.call_extern("int32", "func_name", *args)` | Direct extern call in TIR |
| `relax.Constant(tvm.nd.array(np_array))` | Wrap numpy as Relax constant |
| `func.with_body(new_body)` | Replace PrimFunc body |
| `tir.stmt_functor.post_order_visit(body, fn)` | Walk TIR tree |
| `var.same_as(other)` | Compare TVM Var identity (NOT `==` or `id()`) |

## Common Pitfalls

1. **Var comparison**: Use `var.same_as(other)`, never Python `==` or `id()`
2. **Constants**: Always `np.ascontiguousarray()` before wrapping in `relax.Constant`
3. **Pattern order**: Longest/most-specific patterns first in the list (greedy matching)
4. **bind_constants=False**: Required when patterns reference constants that should remain as function params (not inlined)
5. **Pre-scan bindings**: `PyExprMutator.visit_call_` sees the call but not necessarily earlier bindings — build a var→value map if you need to trace backward
6. **Cleanup**: Always run `DeadCodeElimination()` after fusion to remove orphaned composites

## Key Files (Examples)

| File | Pass Type | Technique |
|------|-----------|-----------|
| `ti_mmalib_qdq_fusion.py` | FuseOpsByPattern + Mutator | QDQ pattern → call_extern (688 lines) |
| `ti_mmalib_qdq_dwconv.py` | FuseOpsByPattern + Mutator | Depthwise variant |
| `ti_mmalib_qdq_fc.py` | FuseOpsByPattern + Mutator | FC/matmul variant with reshape |
| `ti_mmalib_i16_fc.py` | Direct Mutator | Manual matmul matching + runtime quant (332 lines) |
| `ti_mmalib_inject_dma.py` | TIR PrimFunc pass | Buffer injection (545 lines) |
| `ti_residual_add.py` | FuseOpsByPattern + Mutator | Residual pattern, int8 + int16 (557 lines) |
| `schedule_c7x_dma.py` | TIR scheduling pass | DMA tiling via TE schedule API |
| `cpu_generic/pipeline.py` | Pipeline wiring | Pass ordering and target gating (230 lines) |

All in `python/tvm/relax/transform/` (passes) or `python/tvm/relax/backend/cpu_generic/` (pipeline).

## Related Skills

- `relax-c7x:cstatic` — Full pipeline pass order (where your pass fits)
- `relax-c7x:mmalib-offload` — MMALIB-specific pass details and constraints
- `relax-c7x:dsp-ops` — Operator implementations called via call_extern
