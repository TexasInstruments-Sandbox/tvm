"""Unit tests for ti_c7x_composite_inline.inline_declined_composite.

Exercises the shared helper directly: build a small Composite(+Primitive)
function via BlockBuilder whose data operand is a genuine tensor-shaped
relax.Constant embedded directly in the body, then call it from main. A
throwaway mutator declines the match (the same thing FuseQDQToC7xMovement/
FuseQDQToC7xRelu's _lower_* methods do on a decline path) and calls the helper
instead of leaving the call in place.

Scope caveat -- this module is hand-built, NOT what the passes produce.
FuseOpsByPattern(bind_constants=False) sets lift_constant_=true, so it lifts
matched Constants to composite *parameters*; the composite body it emits holds
no relax.Constant (measured on the dq-reshape-q and concat-tuple-field shapes
alike). This test therefore validates that the helper correctly inlines and
substitutes a body containing an embedded Constant, and that such a body would
indeed crash FuseTIR if one ever reached it -- it does not demonstrate that any
pass on this tree emits one.

A *scalar* embedded Constant (e.g. a quantization scale/zero-point) would not
be enough even then: LegalizeOps bakes scalar operands into the generated TIR
as literals, so no relax.Constant survives to reach FuseTIR. Only a
tensor-shaped Constant (passed through as an actual call_tir buffer argument)
does.

Confirms the leftover Composite/Primitive function is actually removable via
DeadCodeElimination, and that the real pipeline's own
LegalizeOps -> FoldConstant -> FuseOps -> FuseTIR sequence (see
legalize_passes in python/tvm/relax/backend/cpu_generic/pipeline.py) then
succeeds.
"""

import numpy as np

import tvm
from tvm import relax
from tvm.relax import TensorStructInfo
from tvm.relax.expr_functor import PyExprMutator, mutator
from tvm.relax.transform.ti_c7x_composite_inline import inline_declined_composite

_CONST_TENSOR = relax.const(np.arange(4, dtype="int8"))


def _build_module_with_composite():
    bb = relax.BlockBuilder()
    # No formal params: the data operand is the embedded tensor Constant
    # itself, matching a matched-but-already-constant tensor leaf under
    # bind_constants=False.
    with bb.function(
        "composite_fn", [], attrs={"Composite": "test.inline", "Primitive": 1}, private=True
    ):
        with bb.dataflow():
            v1 = bb.emit(
                relax.op.dequantize(
                    _CONST_TENSOR, relax.const(0.03, "float32"), relax.const(0, "int8")
                )
            )
            v2 = bb.emit(
                relax.op.quantize(
                    v1, relax.const(0.05, "float32"), relax.const(1, "int8"), out_dtype="int8"
                )
            )
            out = bb.emit_output(v2)
        bb.emit_func_output(out)
    composite_gv = bb.get().get_global_var("composite_fn")

    x = relax.Var("x", TensorStructInfo((4,), "int8"))
    with bb.function("main", [x], attrs={"num_input": 1}):
        with bb.dataflow():
            # x is unused by the composite (it takes no params) -- present
            # only so main has a realistic signature; the composite call
            # itself takes no arguments.
            call = bb.emit(relax.Call(composite_gv, []))
            out = bb.emit_output(call)
        bb.emit_func_output(out)
    return bb.finalize(), composite_gv


@mutator
class _DeclineAndInline(PyExprMutator):
    """Mirrors FuseQDQToC7x*'s own visit_call_ dispatch, but always declines
    (inlines) any match to the test composite -- exactly what those passes'
    _lower_* methods do on a decline path."""

    def visit_call_(self, call):
        if not isinstance(call.op, relax.GlobalVar):
            return super().visit_call_(call)
        func = self.builder_.get()[call.op]
        if not isinstance(func, relax.Function) or "Composite" not in (func.attrs or {}):
            return super().visit_call_(call)
        return inline_declined_composite(self.builder_, call, func)


def _run_decline(mod):
    inliner = _DeclineAndInline(mod)
    for gv, func in mod.functions_items():
        if isinstance(func, relax.Function) and "Composite" not in (func.attrs or {}):
            new_func = inliner.visit_expr(func)
            inliner.builder_.update_func(gv, new_func)
    return inliner.builder_.get()


def _calls_gv(func, gv):
    for block in func.body.blocks:
        for binding in block.bindings:
            val = binding.value
            if isinstance(val, relax.Call) and val.op == gv:
                return True
    return False


def _op_names(func):
    names = []
    for block in func.body.blocks:
        for binding in block.bindings:
            val = binding.value
            if isinstance(val, relax.Call) and hasattr(val.op, "name"):
                names.append(str(val.op.name))
    return names


class TestInlineDeclinedComposite:
    def test_inlines_ops_and_drops_composite_call(self):
        mod, composite_gv = _build_module_with_composite()
        new_mod = _run_decline(mod)
        main = new_mod["main"]
        assert not _calls_gv(main, composite_gv)
        assert "relax.dequantize" in _op_names(main)
        assert "relax.quantize" in _op_names(main)

    def test_dead_code_elimination_removes_orphaned_composite(self):
        mod, _composite_gv = _build_module_with_composite()
        new_mod = _run_decline(mod)
        new_mod = relax.transform.DeadCodeElimination()(new_mod)
        gvar_names = [gv.name_hint for gv in new_mod.get_global_vars()]
        assert "composite_fn" not in gvar_names

    def test_legalize_fuse_fuse_tir_succeeds(self):
        """End-to-end reproduction of the original crash mode: a declined
        composite merely left un-consumed (or inlined without deleting the
        orphan) would make FuseTIR raise "Relax.Constant is not supported
        in primitive functions" here. With inlining + DeadCodeElimination,
        the real pipeline's LegalizeOps -> FoldConstant -> FuseOps -> FuseTIR
        sequence must all succeed."""
        mod, _composite_gv = _build_module_with_composite()
        new_mod = _run_decline(mod)
        new_mod = relax.transform.DeadCodeElimination()(new_mod)
        new_mod = relax.transform.LegalizeOps()(new_mod)
        new_mod = relax.transform.FoldConstant()(new_mod)
        new_mod = relax.transform.FuseOps()(new_mod)
        new_mod = relax.transform.FuseTIR()(new_mod)  # must not raise

    def test_orphan_left_behind_reproduces_the_original_crash(self):
        """Negative control: if the declined composite is left un-consumed
        (today's pre-fix behavior) rather than inlined, FuseTIR must raise
        -- confirms this test actually exercises the bug, not something
        FuseTIR would have tolerated anyway."""
        mod, _composite_gv = _build_module_with_composite()
        # No decline/inline pass runs at all -- main still calls the
        # Composite/Primitive function with its embedded tensor Constant.
        mod = relax.transform.LegalizeOps()(mod)
        mod = relax.transform.FoldConstant()(mod)
        mod = relax.transform.FuseOps()(mod)
        try:
            relax.transform.FuseTIR()(mod)
        except tvm.error.TVMError as e:
            assert "Relax.Constant is not supported in primitive functions" in str(e)
        else:
            raise AssertionError("expected FuseTIR to raise on the un-inlined composite")
