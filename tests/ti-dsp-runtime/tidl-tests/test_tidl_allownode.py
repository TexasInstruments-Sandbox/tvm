#!/usr/bin/env python
"""Track 1 gate tests: offload eligibility decided by ``TIDL_relaxAllowNode``.

Requires ``tidl_model_import_relax.so`` (skipped otherwise). No hardware and no
TIDL import/calibration are needed: only the parse + allowlisting entry point
is exercised.

* ``TestCompositeFidelity`` - the composite the gate synthesises for a match is
  structurally equal to the one ``FuseOpsByPattern`` creates, so the library
  judges exactly what would later be imported.
* ``TestDescribeNode`` / ``TestDriverDeclines`` / ``TestNodeNameLists`` /
  ``TestGateFailsClosed`` - behaviour of the library driver (dtype and shape
  metadata, declines, name lists) and of the Python seam.
"""

import inspect
import json
import os
import subprocess
import sys
import textwrap

import numpy as np
import pytest
import test_tidl_layer_offload as layer_models

import tvm
from tvm import relax
from tvm.relax.backend.tidl import _gate, get_tidl_patterns, partition_for_tidl
from tvm.script import ir as I
from tvm.script import relax as R

pytestmark = [pytest.mark.quick]

C7X_MMA_TIDL_PATH = os.environ.get("C7X_MMA_TIDL_PATH", os.path.expanduser("~/ml/c7x-mma-tidl"))
RELAX_SO_PATH = os.path.join(
    C7X_MMA_TIDL_PATH, "ti_dl/utils/tidlModelImport/out/tidl_model_import_relax.so"
)
TIDL_TOOLS_PATH = os.path.join(C7X_MMA_TIDL_PATH, "tidl_tools")

requires_so = pytest.mark.skipif(
    not os.path.isfile(RELAX_SO_PATH), reason="tidl_model_import_relax.so not built"
)

op = relax.op


def _single_op_module(fn):
    x = relax.Var("x", relax.TensorStructInfo((1, 8, 16, 16), "float32"))
    bb = relax.BlockBuilder()
    with bb.function("main", [x]):
        with bb.dataflow():
            out = bb.emit_output(fn(x))
        bb.emit_func_output(out)
    return bb.get()


@pytest.fixture(scope="module")
def tidl_so():
    """Make sure the library is loadable (partition initialises it itself)."""
    _gate.load_relax_so(tidl_tools_path=TIDL_TOOLS_PATH)


def _layer_model_builders():
    """All ``_build_*_model`` factories from the per-layer offload tests."""
    return sorted(
        (name, fn)
        for name, fn in inspect.getmembers(layer_models, inspect.isfunction)
        if name.startswith("_build_")
        and name.endswith("_model")
        and not inspect.signature(fn).parameters
    )


def _composites_in(mod):
    """Every function with a ``Composite`` attr in *mod*, nested ones included."""
    found = []

    def walk_fn(fn):
        if fn.attrs and "Composite" in fn.attrs:
            found.append(fn)
        body = fn.body
        if isinstance(body, relax.SeqExpr):
            for block in body.blocks:
                for b in block.bindings:
                    v = b.value
                    if isinstance(v, relax.Call) and isinstance(v.op, relax.Function):
                        walk_fn(v.op)
                    elif isinstance(v, relax.Function):
                        walk_fn(v)

    for _, fn in mod.functions.items():
        if isinstance(fn, relax.Function):
            walk_fn(fn)
    return found


def _flatten_blocks(fn, attrs):
    """*fn* re-emitted with all bindings as plain (non-dataflow) vars in one block."""
    bb = relax.BlockBuilder()
    with bb.function("canonical", list(fn.params), private=True):
        vmap = {}
        for block in fn.body.blocks:
            for b in block.bindings:
                vmap[b.var] = bb.emit(_gate._subst(b.value, vmap), b.var.name_hint)
        bb.emit_func_output(_gate._subst(fn.body.body, vmap))
    return bb.get()["canonical"].with_attrs(dict(attrs.items()))


def _equal_ignoring_attrs(synth, real):
    """Structural equality of params/bindings/ret, ignoring attrs and block kind.

    ``MergeCompositeFunctions`` drops ``Primitive`` from the composites, and on
    the per-pattern-merge fallback path it also turns their dataflow block into
    a plain one; ``tidl_node_name`` is gate-only. The parsers walk blocks and
    bindings, so none of those change what the library sees. ``Composite`` is
    checked separately by the caller.
    """
    return tvm.ir.structural_equal(
        _flatten_blocks(synth, real.attrs), _flatten_blocks(real, real.attrs)
    )


def _two_branch_elu(relu_branch_first):
    """``add(multiply(c, relu(sub(c, exp(x)))), relu(x))`` -- the tidl.elu pattern.

    Both of the root ``add``'s operands are computed inside the match, so the
    composite's binding order depends on the order the block defines the two
    branches in. ``relu_branch_first`` emits the right-hand operand's branch
    first, which is the opposite of a left-operand-first dependency walk.
    """
    x = relax.Var("x", relax.TensorStructInfo((1, 8, 16, 16), "float32"))
    one = relax.const(np.ones((1, 8, 16, 16), "float32"))
    alpha = relax.const(np.full((1, 8, 16, 16), -1.0, "float32"))
    bb = relax.BlockBuilder()
    with bb.function("main", [x]):
        with bb.dataflow():

            def left():
                e = bb.emit(relax.op.exp(x), "e")
                s = bb.emit(relax.op.subtract(one, e), "s")
                rn = bb.emit(relax.op.nn.relu(s), "rn")
                return bb.emit(relax.op.multiply(alpha, rn), "m")

            if relu_branch_first:
                r = bb.emit(relax.op.nn.relu(x), "rx")
                m = left()
            else:
                m = left()
                r = bb.emit(relax.op.nn.relu(x), "rx")
            out = bb.emit_output(relax.op.add(m, r), "o")
        bb.emit_func_output(out)
    return bb.get()


@requires_so
@pytest.mark.usefixtures("tidl_so")
class TestCompositeFidelity:
    @pytest.mark.parametrize(
        "name,builder", _layer_model_builders(), ids=lambda x: x if isinstance(x, str) else ""
    )
    def test_synthesised_composite_equals_fused(self, name, builder):
        mod = builder()
        with _gate.recording() as records:
            out = partition_for_tidl(mod, {})
        reals = _composites_in(out)
        for res in records:
            if not res.verdict:
                continue  # declined matches are never fused
            assert any(
                str(r.attrs["Composite"]) == res.pattern and _equal_ignoring_attrs(res.composite, r)
                for r in reals
            ), f"{name}: gate composite for {res.node_name} matches no fused composite"

    @pytest.mark.parametrize("relu_branch_first", [False, True])
    def test_multi_branch_match_follows_program_order(self, relu_branch_first):
        """A two-branch match must synthesise in the block's order, either way.

        The single-op models above cannot reach this: with one binding there is
        no order to get wrong. Here a dependency walk would always emit the
        left operand's branch first, so with ``relu_branch_first`` it produces a
        composite whose bindings, parameters and inline-constant order are not
        the ones fusion builds -- and the library would be judging a function
        that is never imported.
        """
        mod = _two_branch_elu(relu_branch_first)
        with _gate.recording() as records:
            out = partition_for_tidl(mod, {})
        elu = [r for r in _composites_in(out) if str(r.attrs["Composite"]) == "tidl.elu"]
        assert elu, "tidl.elu did not fuse; the fixture no longer matches the pattern"
        gate = [r for r in records if r.pattern == "tidl.elu" and r.verdict]
        assert gate, "the gate did not accept tidl.elu"
        assert _equal_ignoring_attrs(gate[0].composite, elu[0])


# ---------------------------------------------------------------------------
# Library driver behaviour (C1-C6, C8, C9)
# ---------------------------------------------------------------------------

TIDL_FLOAT32 = 6  # eTIDL_ElementType TIDL_SinglePrecFloat


def _run_script(tmp_path, code):
    """Run *code* as a script file (TVMScript needs real source) in a subprocess."""
    script = tmp_path / "case.py"
    script.write_text(code)
    return subprocess.run(
        [sys.executable, str(script)], capture_output=True, text=True, timeout=300
    )


def _gate_results(mod, config=None):
    """Partition *mod* through the real gate; return the recorded GateResults."""
    with _gate.recording() as records:
        partition_for_tidl(mod, config or {})
    return list(records)


def _describe(result):
    return json.loads(str(tvm.get_global_func("TIDL_relaxDescribeNode")(result.call)))


def _by_pattern(results, pattern):
    return next(r for r in results if r.pattern == pattern)


@requires_so
@pytest.mark.usefixtures("tidl_so")
class TestDescribeNode:
    """C1/C4/C9: parse-level introspection of what the library saw."""

    def test_conv_metadata_and_dtypes(self):
        results = _gate_results(layer_models._build_sigmoid_model())
        d = _describe(_by_pattern(results, "tidl.nn.conv2d_relu"))
        # the weight was bound to its own var in the fixture; partition inlines it
        assert d["varTensorsDims"] == [[1, 3, 32, 32]]
        assert d["constTensorsDims"] == [[16, 3, 3, 3]]
        assert d["outputTensorDims"] == [[1, 16, 32, 32]]
        assert d["inputDataTypes"] == [TIDL_FLOAT32, TIDL_FLOAT32]
        assert d["outputDataTypes"] == [TIDL_FLOAT32]

    def test_inline_constant_is_a_const_input_and_follows_variables(self):
        @I.ir_module
        class Model:
            @R.function
            def main(x: R.Tensor((1, 3, 32, 32), "float32")):
                with R.dataflow():
                    y = R.nn.conv2d(
                        x, R.const(np.ones((8, 3, 3, 3), "float32")), padding=[1, 1, 1, 1]
                    )
                    R.output(y)
                return y

        res = _by_pattern(_gate_results(Model), "tidl.nn.conv2d")
        d = _describe(res)
        assert res.verdict is True
        assert d["varTensorsDims"] == [[1, 3, 32, 32]]
        assert d["constTensorsDims"] == [[8, 3, 3, 3]]
        # variables first, then constants
        assert d["inputDataTypes"] == [TIDL_FLOAT32, TIDL_FLOAT32]

    def test_tuple_output_has_one_entry_per_field(self):
        d = _describe(_by_pattern(_gate_results(layer_models._build_split_model()), "tidl.split"))
        assert d["numOutputs"] >= 2
        assert len(d["outputTensorDims"]) == d["numOutputs"]
        assert len(d["outputDataTypes"]) == d["numOutputs"]
        assert all(len(dims) == 4 for dims in d["outputTensorDims"])


@requires_so
@pytest.mark.usefixtures("tidl_so")
class TestDriverDeclines:
    def test_unknown_dtype_is_declined(self):
        @I.ir_module
        class Model:
            @R.function
            def main(x: R.Tensor((1, 8, 16, 16), "float16")):
                with R.dataflow():
                    y = R.sigmoid(x)
                    R.output(y)
                return y

        res = _by_pattern(_gate_results(Model), "tidl.sigmoid")
        assert res.verdict is False

    def test_unsupported_int64_on_non_exempt_layer_is_declined(self):
        @I.ir_module
        class Model:
            @R.function
            def main(x: R.Tensor((1, 8, 16, 16), "int64")):
                with R.dataflow():
                    y = R.abs(x)
                    R.output(y)
                return y

        assert _by_pattern(_gate_results(Model), "tidl.abs").verdict is False

    def test_symbolic_dimension_declines_instead_of_crashing(self, tmp_path):
        # A non-constant dim used to dereference a null IntImmNode in the
        # library; run in a subprocess so a crash fails the test, not pytest.
        code = textwrap.dedent(
            """
            import tvm
            from tvm import relax
            from tvm.script import ir as I, relax as R
            from tvm.relax.backend.tidl import _gate, partition_for_tidl

            
            @I.ir_module
            class Model:
                @R.function
                def main(x: R.Tensor(("n", 8, 16, 16), "float32")):
                    with R.dataflow():
                        y = R.sigmoid(x)
                        R.output(y)
                    return y

            with _gate.recording() as recs:
                partition_for_tidl(Model, {})
            assert recs and all(r.verdict is False for r in recs), [r.verdict for r in recs]
            print("DECLINED")
            """
        )
        proc = _run_script(tmp_path, code)
        assert proc.returncode == 0, proc.stderr[-2000:]
        assert "DECLINED" in proc.stdout

    def test_rejection_reason_is_available(self):
        # TIDL ScatterElements only supports the width axis (3 in NCHW)
        data = relax.Var("data", relax.TensorStructInfo((1, 8, 16, 16), "float32"))
        indices = relax.Var("indices", relax.TensorStructInfo((1, 2, 16, 16), "int32"))
        updates = relax.Var("updates", relax.TensorStructInfo((1, 2, 16, 16), "float32"))
        bb = relax.BlockBuilder()
        with bb.function("main", [data, indices, updates]):
            with bb.dataflow():
                out = bb.emit(relax.op.scatter_elements(data, indices, updates, axis=1))
                gv = bb.emit_output(out)
            bb.emit_func_output(gv)
        res = _by_pattern(_gate_results(bb.get()), "tidl.scatter_elements")
        assert res.verdict is False
        assert res.reason.startswith(res.node_name)
        assert "width axis" in res.reason


@requires_so
class TestNodeNameLists:
    """C6: deny lists match the ``tidl_node_name`` attr on synthetic calls."""

    def test_deny_list_by_name_rejects_exactly_that_node(self):
        mod = layer_models._build_sigmoid_model()
        base_cfg = {"tidl_tools_path": TIDL_TOOLS_PATH}
        baseline = {r.node_name: r.verdict for r in _gate_results(mod, base_cfg)}
        sig = next(n for n in baseline if n.startswith("tidl.sigmoid:"))
        assert baseline[sig]

        deny_cfg = dict(base_cfg, tidl_options={"deny_list:layer_name": sig})
        denied = {r.node_name: r.verdict for r in _gate_results(mod, deny_cfg)}
        assert denied[sig] is False
        assert all(denied[n] == baseline[n] for n in baseline if n != sig)
        # a later partition with default options starts from clean state
        again = {r.node_name: r.verdict for r in _gate_results(mod, base_cfg)}
        assert again == baseline


class TestGateFailsClosed:
    def test_missing_library_raises_through_fuse_ops_callback(self, tmp_path):
        # Fail closed: with no .so and no allow_fn the error must surface from
        # inside the C++ FuseOpsByPattern callback with its message intact
        # (not be swallowed or mistaken for a cyclic-dependency fallback).
        code = textwrap.dedent(
            """
            import os
            os.environ.pop("TIDL_RELAX_SO_PATH", None)
            os.environ["C7X_MMA_TIDL_PATH"] = "/nonexistent"
            from tvm.script import ir as I, relax as R
            import tvm
            from tvm import relax
            # A wheel install bundles the .so, which load_relax_so prefers
            # over C7X_MMA_TIDL_PATH; hide it too.
            import tvm.data.ti_dsp.paths as _paths
            _paths.find_tidl_relax_so = lambda: None
            from tvm.relax.backend.tidl import _gate
            from tvm.relax.dpl.pattern import is_op, wildcard
            from tvm.relax.transform import FusionPattern

            @I.ir_module
            class Model:
                @R.function
                def main(x: R.Tensor((1, 8, 16, 16), "float32")):
                    with R.dataflow():
                        y = R.sigmoid(x)
                        R.output(y)
                    return y

            inp = wildcard()
            pat = FusionPattern("tidl.sigmoid", is_op("relax.sigmoid")(inp), {"i": inp},
                                _gate.tidl_gate("tidl.sigmoid"))
            try:
                relax.transform.FuseOpsByPattern([pat], bind_constants=True)(Model)
            except Exception as e:
                assert "tidl_model_import_relax.so" in str(e), str(e)
                print("RAISED")
            """
        )
        proc = _run_script(tmp_path, code)
        assert proc.returncode == 0, proc.stderr[-2000:]
        assert "RAISED" in proc.stdout


def _fuse_directly(mod, pattern_name):
    """FuseOpsByPattern with one TIDL pattern, outside partition(), accepting all."""
    pat = next(p for p in get_tidl_patterns() if p.name == pattern_name)
    with _gate.active_allow_fn(lambda call: 1):
        return relax.transform.FuseOpsByPattern([pat], bind_constants=True)(mod)


class TestDirectPatternUse:
    """Patterns driven without partition(), so no binding order is installed."""

    def test_multi_branch_match_fails_closed(self):
        # The dependency walk cannot tell which branch the block defines first.
        with pytest.raises(Exception, match="binding_order"):
            _fuse_directly(_two_branch_elu(relu_branch_first=True), "tidl.elu")

    @pytest.mark.parametrize("relu_branch_first", [False, True])
    def test_multi_branch_match_with_binding_order(self, relu_branch_first):
        mod = _two_branch_elu(relu_branch_first)
        with _gate.binding_order(mod):
            out = _fuse_directly(mod, "tidl.elu")
        assert any(str(f.attrs["Composite"]) == "tidl.elu" for f in _composites_in(out))

    def test_single_chain_match_needs_no_binding_order(self):
        # One valid order, so the fallback is exact and must not raise.
        out = _fuse_directly(_single_op_module(relax.op.sigmoid), "tidl.sigmoid")
        assert any(str(f.attrs["Composite"]) == "tidl.sigmoid" for f in _composites_in(out))


class TestDefaultArtifactsDir:
    def test_unique_per_process_and_removed_at_exit(self, tmp_path):
        code = textwrap.dedent(
            """
            import os
            from tvm.relax.backend.tidl import _gate, TIDLOffloadCompiler
            d = _gate.default_artifacts_dir()
            assert os.path.isdir(d)
            assert _gate.default_artifacts_dir() == d
            assert TIDLOffloadCompiler()._artifacts_dir == d
            print(d)
            """
        )
        dirs = []
        for _ in range(2):
            proc = _run_script(tmp_path, code)
            assert proc.returncode == 0, proc.stderr[-2000:]
            dirs.append(proc.stdout.strip().splitlines()[-1])
        assert dirs[0] != dirs[1]
        assert not any(os.path.exists(d) for d in dirs)


@requires_so
class TestFirmwareGates:
    """C7: ops the Relay parsers gate on the c7x firmware version."""

    def _verdicts(self, firmware=None):
        cfg = {"tidl_tools_path": TIDL_TOOLS_PATH}
        if firmware:
            cfg["tidl_options"] = {"advanced_options:c7x_firmware_version": firmware}
        mod = _single_op_module(op.log)
        return {r.pattern: r for r in _gate_results(mod, cfg)}

    def test_current_firmware_accepts(self):
        assert self._verdicts()["tidl.log"].verdict is True

    def test_old_firmware_declines(self):
        res = self._verdicts("09_00_00_00")["tidl.log"]
        assert res.verdict is False
        assert "firmware" in res.reason

    def test_declining_does_not_leak_into_the_next_partition(self):
        self._verdicts("09_00_00_00")
        assert self._verdicts()["tidl.log"].verdict is True
