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

import tvm_ffi

import tvm
from tvm import tirx
from tvm.relax.transform.ti_mmalib_inject_dma import InjectMMALIBDMA
from tvm.relax.transform.ti_mmalib_legalize import _call_extern_checked

pytestmark = pytest.mark.quick

_L2_BUDGET = 2 * 1024 * 1024  # large enough to cache both input and weights


def _make_checked_conv2d_primfunc(kernel_name: str, c_out: int = 64):
    """Build a PrimFunc whose body matches what MMALIB legalization emits:
    a single status-checked call_extern, not a bare Evaluate."""
    h_input = tirx.Var("input", "handle")
    h_kernel = tirx.Var("kernel", "handle")
    h_bias = tirx.Var("bias", "handle")
    h_scale = tirx.Var("scale", "handle")
    h_shift = tirx.Var("shift", "handle")
    h_output = tirx.Var("output", "handle")

    body = _call_extern_checked(
        "int32",
        kernel_name,
        h_input,  # args[1]
        h_kernel,  # args[2]
        h_bias,  # args[3]
        h_scale,  # args[4]
        h_shift,  # args[5]
        h_output,  # args[6]
        tirx.const(64, "int32"),  # C_in
        tirx.const(28, "int32"),  # H_in
        tirx.const(28, "int32"),  # W_in
        tirx.const(c_out, "int32"),  # C_out
        tirx.const(3, "int32"),  # KH
        tirx.const(3, "int32"),  # KW
        tirx.const(1, "int32"),  # stride_h
        tirx.const(1, "int32"),  # stride_w
        tirx.const(1, "int32"),  # pad_top
        tirx.const(1, "int32"),  # pad_bottom
        tirx.const(0, "int32"),  # pad_left
        tirx.const(0, "int32"),  # pad_right
    )
    return tirx.PrimFunc([h_input, h_kernel, h_bias, h_scale, h_shift, h_output], body)


def _count_dma_copies(func):
    count = 0

    def _visit(node):
        nonlocal count
        if (
            isinstance(node, tvm.ir.Call)
            and node.op.same_as(tvm.ir.Op.get("tirx.call_extern"))
            and isinstance(node.args[0], tvm.ir.StringImm)
            and node.args[0].value == "tvm_dsp_dma_copy"
        ):
            count += 1

    tvm_ffi.structural_walk(func.body, (tvm.ir.Call, _visit), order="post")
    return count


def _has_report_error(func):
    found = False

    def _visit(node):
        nonlocal found
        if (
            isinstance(node, tvm.ir.Call)
            and node.op.same_as(tvm.ir.Op.get("tirx.call_extern"))
            and isinstance(node.args[0], tvm.ir.StringImm)
            and node.args[0].value == "tvm_dsp_report_error"
        ):
            found = True

    tvm_ffi.structural_walk(func.body, (tvm.ir.Call, _visit), order="post")
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


def _dma_calls(func):
    """Return (total, status_checked) counts of tvm_dsp_dma_copy/_wait calls.

    A call counts as status-checked when it is the value of a Bind, the
    form _call_extern_checked emits (Bind(status, Cast(int64, call)) then
    an IfThenElse on status)."""
    names = ("tvm_dsp_dma_copy", "tvm_dsp_dma_wait")
    op = tvm.ir.Op.get("tirx.call_extern")
    total = 0
    checked = 0

    def _is_dma(node):
        return (
            isinstance(node, tvm.ir.Call)
            and node.op.same_as(op)
            and isinstance(node.args[0], tvm.ir.StringImm)
            and node.args[0].value in names
        )

    def _visit_call(node):
        nonlocal total
        if _is_dma(node):
            total += 1

    def _visit_bind(node):
        nonlocal checked
        value = node.value.value if isinstance(node.value, tirx.Cast) else node.value
        if _is_dma(value):
            checked += 1

    tvm_ffi.structural_walk(
        func.body, [(tvm.ir.Call, _visit_call), (tirx.Bind, _visit_bind)], order="post"
    )
    return total, checked


@pytest.mark.parametrize(
    "c_out, l2_budget",
    [
        # Input and all weights fit: one input copy, one weight copy, one wait.
        (64, _L2_BUDGET),
        # Weights don't fit: OC-tiled path, input copy + wait once, then a
        # weight-tile copy + wait per tile.  Budget = input (64x28x28) plus
        # one 64-channel weight tile (64 x 64x3x3), so C_out=256 tiles by 64.
        (256, 64 * 28 * 28 + 64 * 64 * 3 * 3),
    ],
    ids=["whole_weights", "oc_tiled"],
)
def test_dma_copies_and_waits_are_status_checked(c_out, l2_budget):
    """Every DMA copy and wait InjectMMALIBDMA emits must be status-checked.

    tvm_dsp_dma_copy can reject a transfer, and the copies used to be bare
    Evaluates whose status was dropped: the MMALIB call then read whatever
    the L2 buffer already held and produced wrong output with no error."""
    func = _make_checked_conv2d_primfunc("mmalib_conv2d_i8", c_out=c_out)
    mod_after = InjectMMALIBDMA(l2_budget)(tvm.IRModule({"mmalib_conv2d": func}))

    total, checked = _dma_calls(mod_after["mmalib_conv2d"])
    assert total >= 3, f"expected DMA copies and a wait to be injected, found {total} calls"
    assert checked == total, f"{total - checked} of {total} DMA calls are not status-checked"
