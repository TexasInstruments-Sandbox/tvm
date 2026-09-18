"""Regression test: _call_extern_checked must abort, not just log, on failure.

_call_extern_checked wraps an offload kernel's call_extern in a status
check that reports the failure via tvm_dsp_report_error. Reporting alone
is not enough: the wrapper PrimFunc this compiles into has its own
unconditional `return 0;` on the line right after the check, so without
an explicit abort, the wrapper reports the error and then still tells its
caller the call succeeded -- execution proceeds to consume whatever stale
or partial output the failed kernel left.

Confirmed by inspecting the actual generated C code (compiled through the
real c_static_lib/c7x_host pipeline) that _call_extern_checked's error
branch must include tir.tvm_throw_last_error() -- the same builtin
lower_l2sram_alloc.py uses for its own tvm_l2_alloc null-check -- which
compiles to a plain early `return -1;`. The generated driver (lib0.c)
already checks and propagates that non-zero return
(`if (__call_ret != 0) return __call_ret;`), the same convention used for
the L2-alloc failure paths, so this is sufficient to turn a kernel failure
into a real, surfaced ModelError instead of a log line next to garbage
output. tir.ret() was tried first and rejected: under this pipeline's
packed-call convention, it sets the *logical return value* of the
function (as if returning a real result), not a C-level failure code, so
the wrapper still reported success to its caller.

Pure TIR-level test (no C compile, no hardware / DSP build).
"""

import pytest

from tvm import tir
from tvm.relax.transform.ti_mmalib_legalize import _call_extern_checked

pytestmark = pytest.mark.quick

_THROW_LAST_ERROR = "tir.tvm_throw_last_error"
_REPORT_ERROR = "tvm_dsp_report_error"


def _find_error_branch(stmt):
    """Return the LetStmt's IfThenElse.then_case (the status != 0 branch)."""
    assert isinstance(stmt, tir.LetStmt)
    if_stmt = stmt.body
    assert isinstance(if_stmt, tir.IfThenElse)
    return if_stmt.then_case


def _collect_call_names(stmt):
    names = []

    def _visit(node):
        if not isinstance(node, tir.Call):
            return
        op_name = node.op.name if hasattr(node.op, "name") else None
        if op_name == "tir.call_extern" and node.args and isinstance(node.args[0], tir.StringImm):
            names.append(node.args[0].value)
        elif op_name is not None:
            names.append(op_name)

    tir.stmt_functor.post_order_visit(stmt, _visit)
    return names


def test_error_branch_aborts_via_throw_last_error():
    """The failure branch must call tir.tvm_throw_last_error(), not just log."""
    stmt = _call_extern_checked("int32", "mmalib_conv2d_i8", tir.Var("x", "handle"))
    error_branch = _find_error_branch(stmt)
    names = _collect_call_names(error_branch)
    assert _THROW_LAST_ERROR in names, (
        f"Error branch does not abort via {_THROW_LAST_ERROR}; found calls: {names}. "
        "A kernel failure would be logged but execution would continue on stale output."
    )


def test_error_branch_still_reports_before_aborting():
    """The failure branch must still report the specific op + status."""
    stmt = _call_extern_checked("int32", "mmalib_conv2d_i8", tir.Var("x", "handle"))
    error_branch = _find_error_branch(stmt)
    names = _collect_call_names(error_branch)
    assert _REPORT_ERROR in names, (
        f"Error branch dropped {_REPORT_ERROR}; found calls: {names}. "
        "A kernel failure would abort silently with no diagnostic."
    )


def test_success_branch_is_unchanged():
    """The status == 0 case must not report or abort."""
    stmt = _call_extern_checked("int32", "mmalib_conv2d_i8", tir.Var("x", "handle"))
    assert isinstance(stmt, tir.LetStmt)
    if_stmt = stmt.body
    assert isinstance(if_stmt, tir.IfThenElse)
    assert if_stmt.else_case is None
