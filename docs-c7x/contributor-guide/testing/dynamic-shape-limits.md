# Dynamic Shape Limitations

Tests for Relax IR dynamic features on the c_static backend with the
C7x DSP runtime. These validate that models with runtime-determined
shapes and conditional logic compile and execute correctly. Located at
`tests/ti-dsp-runtime/dynamic-tests/`.

## Tests

| Test file | Model | What it exercises |
|-----------|-------|-------------------|
| `test_if_dsp.py` | `IfSelectModule` | Relax `If` expression: `astype(float32, bool)` condition, add vs multiply branch selection |
| `test_dynamic_batch_dsp.py` | `DynBatchAdd` | Element-wise add with symbolic batch dim (`R.Tensor(("batch", 4))`) |
| `test_dynamic_batch_dsp.py` | `DynBatchMatmul` | Matrix multiply `x[batch,8] @ w[8,4]` with symbolic batch and non-trivial `shape_func` |

### Quick tests (8 total)

```
test_if_true_branch          If cond=1.0 -> add(x,x)
test_if_false_branch         If cond=0.0 -> mul(x,x)
test_dynamic_batch_1         batch=1  add
test_dynamic_batch_4         batch=4  add
test_dynamic_batch_8         batch=8  add
test_dynamic_matmul_batch_1  batch=1  matmul
test_dynamic_matmul_batch_4  batch=4  matmul
test_dynamic_matmul_batch_16 batch=16 matmul
```

Plus `test_if_both_branches` (not marked quick, runs both branches).

## Running

```bash
cd $TVM_HOME
export PYTHONPATH=$TVM_HOME/python:$PYTHONPATH
export TI_CGT_C7000_PATH=/opt/ti/c7x/ti-cgt-c7000_5.0.1.LTS

# Quick tests on C7x host emulation (~13s)
pytest tests/ti-dsp-runtime/dynamic-tests/ \
    --rootdir=tests/ti-dsp-runtime/dynamic-tests \
    -m quick --dsp-mode=c7x_host -v

# On AM67A hardware via DLOAD
pytest tests/ti-dsp-runtime/dynamic-tests/ \
    --rootdir=tests/ti-dsp-runtime/dynamic-tests \
    -m quick --dsp-mode=c7x_dload -v

# Standalone (no pytest)
python tests/ti-dsp-runtime/dynamic-tests/test_if_dsp.py --dsp-mode c7x_host
python tests/ti-dsp-runtime/dynamic-tests/test_dynamic_batch_dsp.py --dsp-mode c7x_host
```

## How dynamic shapes work in c_static

The shape heap pipeline handles runtime dimension values:

1. `alloc_shape_heap(N)` -- allocate N-slot int64 array
2. `match_shape(input, heap, codes, vals)` -- extract runtime dims
   from input tensor shapes into heap slots (StoreToHeap)
3. `shape_func(heap)` -- TIR function that computes derived values
   (e.g. storage sizes) from heap entries
4. `make_shape(heap, codes, vals)` -- construct shape objects from
   heap values (dynamic) and immediates (static)
5. `alloc_storage` / `alloc_tensor` -- allocate output using the
   constructed shapes

All five are generated as direct API calls (no FFI dispatch).

## Limitations: no loops

Relax represents loops as tail-recursive functions: a function calls
itself with updated arguments until a termination condition is met.
This requires `vm.builtin.invoke_closure` with a runtime function
dispatch table.

The c_static backend generates standalone C functions with no
inter-function call mechanism. Each Relax function becomes a single
C function, and there is no function table or dispatch loop. This
means tail-recursive patterns cannot be compiled to c_static.

This is a fundamental architectural constraint of the static C
codegen, not a bug. Models that need iterative computation
(autoregressive LLMs, RNNs, iterative refinement) must unroll the
loop at the graph level or use the VM backend instead.
