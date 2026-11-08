"""Unit tests for c7x_topk_legalize (relax.topk -> c7x_topk call_extern).

Pure Relax/TIR-level tests, no hardware or DSP build needed.
"""

import pytest
import tvm_ffi

import tvm
from tvm import relax, tirx
from tvm.relax import TensorType as TensorStructInfo
from tvm.relax.transform.ti_c7x_topk_legalize import c7x_topk_legalize

pytestmark = pytest.mark.quick


def _legalized_topk(n=100, k=5):
    bb = relax.BlockBuilder()
    x = relax.Var("x", TensorStructInfo((1, n), "float32"))
    with bb.function("main", [x]):
        with bb.dataflow():
            out = bb.emit_output(relax.op.topk(x, k=k, axis=-1, ret_type="both", dtype="int64"))
        bb.emit_func_output(out)
    legalize = relax.transform.LegalizeOps(customize_legalize_map={"relax.topk": c7x_topk_legalize})
    return legalize(bb.finalize())


def test_topk_call_is_status_checked():
    """c7x_topk returns -1 for a bad k or a failed scratch allocation; the call
    must be status-checked (a Bind on its result, as _call_extern_checked
    emits), or the outputs are left unwritten with no error."""
    mod = _legalized_topk()
    op = tvm.ir.Op.get("tirx.call_extern")
    found = {"total": 0, "checked": 0}

    def _is_topk(node):
        return (
            isinstance(node, tvm.ir.Call)
            and node.op.same_as(op)
            and isinstance(node.args[0], tvm.ir.StringImm)
            and node.args[0].value == "c7x_topk"
        )

    def _visit_call(node):
        found["total"] += _is_topk(node)

    def _visit_bind(node):
        value = node.value.value if isinstance(node.value, tirx.Cast) else node.value
        found["checked"] += _is_topk(value)

    for func in mod.functions.values():
        if isinstance(func, tirx.PrimFunc):
            tvm_ffi.structural_walk(
                func.body, [(tvm.ir.Call, _visit_call), (tirx.Bind, _visit_bind)], order="post"
            )
    assert found["total"] == 1, f"expected one c7x_topk call, found {found['total']}"
    assert found["checked"] == 1, "c7x_topk is emitted without a status check"
