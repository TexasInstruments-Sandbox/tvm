"""Unit tests for LowerDMAToExtern (tirx.dma_copy/dma_wait -> tvm_dsp_dma_*).

Pure TIR-level tests, no hardware or DSP build needed.
"""

import pytest
import tvm_ffi

import tvm
from tvm import tirx
from tvm.tirx.transform import LowerDMAToExtern

pytestmark = pytest.mark.quick


def _dma_primfunc():
    dst = tirx.Var("dst", "handle")
    src = tirx.Var("src", "handle")
    queue, nbytes, bypass, inflight = (tirx.const(v, "int32") for v in (0, 4096, 0, 0))
    body = tirx.SeqStmt(
        [
            tirx.Evaluate(tirx.call_intrin("int32", "tirx.dma_start_group", queue)),
            tirx.Evaluate(
                tirx.call_intrin("int32", "tirx.dma_copy", queue, dst, src, nbytes, bypass)
            ),
            tirx.Evaluate(tirx.call_intrin("int32", "tirx.dma_end_group", queue)),
            tirx.Evaluate(tirx.call_intrin("int32", "tirx.dma_wait", queue, inflight)),
        ]
    )
    return tirx.PrimFunc([dst, src], body)


def test_dma_copy_and_wait_are_status_checked():
    """tvm_dsp_dma_copy can reject a transfer; unchecked, the scheduled loop
    would compute on whatever the L2 tile already held. Every lowered copy
    and wait must be the value of a Bind, the form _call_extern_checked
    emits, and the error path must report through tvm_dsp_report_error."""
    mod = LowerDMAToExtern()(tvm.IRModule({"main": _dma_primfunc()}))
    op = tvm.ir.Op.get("tirx.call_extern")
    names = ("tvm_dsp_dma_copy", "tvm_dsp_dma_wait")
    seen = {"total": 0, "checked": 0, "report_error": 0, "dma_builtin": 0}

    def _extern_name(node):
        if isinstance(node, tvm.ir.Call) and node.op.same_as(op):
            return node.args[0].value
        return None

    def _visit_call(node):
        name = _extern_name(node)
        seen["total"] += name in names
        seen["report_error"] += name == "tvm_dsp_report_error"
        if isinstance(node.op, tvm.ir.Op) and node.op.name.startswith("tirx.dma_"):
            seen["dma_builtin"] += 1

    def _visit_bind(node):
        value = node.value.value if isinstance(node.value, tirx.Cast) else node.value
        seen["checked"] += _extern_name(value) in names

    tvm_ffi.structural_walk(
        mod["main"].body, [(tvm.ir.Call, _visit_call), (tirx.Bind, _visit_bind)], order="post"
    )
    assert seen["dma_builtin"] == 0, "a tirx.dma_* builtin was left unlowered"
    assert seen["total"] == 2, f"expected one copy and one wait, found {seen['total']}"
    assert seen["checked"] == 2, f"{2 - seen['checked']} of 2 DMA calls are not status-checked"
    assert seen["report_error"] == 2
