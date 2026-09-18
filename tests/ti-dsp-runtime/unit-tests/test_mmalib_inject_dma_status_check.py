"""Regression test: InjectMMALIBDMA must still fire on status-checked calls.

MMALIB legalization wraps its call_extern in a status check
(``_call_extern_checked``): ``LetStmt(status, Cast(int64, call), IfThenElse(
status != 0, report_error, None))``, instead of a bare
``Evaluate(call_extern(...))``. InjectMMALIBDMA's pattern matcher only
recognized the bare form, so it silently stopped finding any MMALIB call
once every lowerer switched to the checked form -- L2 SRAM DMA prefetch and
OC-tiling were dropped for every MMALIB kernel on c7x, with no error and no
existing test catching it (the old regression test in mmalib-tests/ builds
bare-Evaluate PrimFuncs directly, bypassing _call_extern_checked entirely).

Pure Python TIR-level tests -- no hardware / DSP build.
"""

import pytest

import tvm
from tvm import tir
from tvm.relax.transform.ti_mmalib_inject_dma import InjectMMALIBDMA
from tvm.relax.transform.ti_mmalib_legalize import _call_extern_checked

pytestmark = pytest.mark.quick

_L2_BUDGET = 2 * 1024 * 1024  # large enough to cache both input and weights


def _make_checked_conv2d_primfunc(kernel_name: str):
    """Build a PrimFunc whose body matches what MMALIB legalization emits:
    a single status-checked call_extern, not a bare Evaluate."""
    h_input = tir.Var("input", "handle")
    h_kernel = tir.Var("kernel", "handle")
    h_bias = tir.Var("bias", "handle")
    h_scale = tir.Var("scale", "handle")
    h_shift = tir.Var("shift", "handle")
    h_output = tir.Var("output", "handle")

    body = _call_extern_checked(
        "int32",
        kernel_name,
        h_input,  # args[1]
        h_kernel,  # args[2]
        h_bias,  # args[3]
        h_scale,  # args[4]
        h_shift,  # args[5]
        h_output,  # args[6]
        tir.const(64, "int32"),  # C_in
        tir.const(28, "int32"),  # H_in
        tir.const(28, "int32"),  # W_in
        tir.const(64, "int32"),  # C_out
        tir.const(3, "int32"),  # KH
        tir.const(3, "int32"),  # KW
        tir.const(1, "int32"),  # stride_h
        tir.const(1, "int32"),  # stride_w
        tir.const(1, "int32"),  # pad_top
        tir.const(1, "int32"),  # pad_bottom
        tir.const(0, "int32"),  # pad_left
        tir.const(0, "int32"),  # pad_right
    )
    return tir.PrimFunc([h_input, h_kernel, h_bias, h_scale, h_shift, h_output], body)


def _count_dma_copies(func):
    count = 0

    def _visit(node):
        nonlocal count
        if (
            isinstance(node, tir.Call)
            and node.op.same_as(tvm.ir.Op.get("tir.call_extern"))
            and isinstance(node.args[0], tir.StringImm)
            and node.args[0].value == "tvm_dsp_dma_copy"
        ):
            count += 1

    tir.stmt_functor.post_order_visit(func.body, _visit)
    return count


def _has_report_error(func):
    found = False

    def _visit(node):
        nonlocal found
        if (
            isinstance(node, tir.Call)
            and node.op.same_as(tvm.ir.Op.get("tir.call_extern"))
            and isinstance(node.args[0], tir.StringImm)
            and node.args[0].value == "tvm_dsp_report_error"
        ):
            found = True

    tir.stmt_functor.post_order_visit(func.body, _visit)
    return found


def test_dma_injected_for_status_checked_call():
    """InjectMMALIBDMA must fire even when the call is status-checked.

    Before the fix, the pattern matcher only recognized a bare
    Evaluate(call_extern(...)), so it silently no-op'd on the checked form
    and no DMA copy/wait was ever emitted.
    """
    func = _make_checked_conv2d_primfunc("mmalib_conv2d_i8")
    mod = tvm.IRModule({"mmalib_conv2d": func})
    mod_after = InjectMMALIBDMA(_L2_BUDGET)(mod)

    assert _count_dma_copies(mod_after["mmalib_conv2d"]) == 2, (
        "Expected input + weight DMA copies to be injected for a "
        "status-checked MMALIB call; InjectMMALIBDMA silently no-op'd."
    )


def test_status_check_preserved_after_dma_injection():
    """Rewriting the call's args for L2 pointers must not drop the status
    check -- otherwise a kernel failure silently produces stale output
    again, the exact bug the status check was added to prevent."""
    func = _make_checked_conv2d_primfunc("mmalib_conv2d_i8")
    mod = tvm.IRModule({"mmalib_conv2d": func})
    mod_after = InjectMMALIBDMA(_L2_BUDGET)(mod)

    assert _has_report_error(mod_after["mmalib_conv2d"]), (
        "Status-check error reporting (tvm_dsp_report_error) was dropped "
        "when InjectMMALIBDMA rewrote the call's args for L2 pointers."
    )
