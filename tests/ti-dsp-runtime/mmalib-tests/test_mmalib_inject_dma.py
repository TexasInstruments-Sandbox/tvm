"""Unit tests for InjectMMALIBDMA — specifically the guard bytes computation.

The guard allocation at the start of the L2 SRAM region prevents the
streaming-engine backward prefetch from underflowing the L2 base address.
Its size is `pad_top * W_in * elem_bytes` when pad_top > 0, falling back
to 128 bytes when pad_top == 0.

Before the fix (Phase 2c code review), args[13] (stride_h) was read instead
of args[15] (pad_top) for both i8 and i16 conv2d.  With stride_h=1 and
pad_top=2 the guard would be 1*W_in instead of the correct 2*W_in.

These are pure-Python pass-level tests — no DSP hardware required.

Usage:
    pytest test_mmalib_inject_dma.py -v
"""

import tvm
from tvm import tir
from tvm.relax.transform import InjectMMALIBDMA

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _make_conv2d_primfunc(
    kernel_name: str,
    C_in: int,
    H_in: int,
    W_in: int,
    C_out: int,
    KH: int,
    KW: int,
    stride_h: int,
    stride_w: int,
    pad_top: int,
    pad_bottom: int,
    pad_left: int,
    pad_right: int,
):
    """Build the minimal TIR PrimFunc that InjectMMALIBDMA expects.

    The function body is a single:
        Evaluate(call_extern("int32", kernel_name, input, kernel, bias,
                             scale, shift, output, C_in, H_in, W_in, C_out,
                             KH, KW, stride_h, stride_w,
                             pad_top, pad_bottom, pad_left, pad_right))

    The pass inspects args[15] for pad_top (after the fix); before the fix
    it incorrectly read args[13] (stride_h).
    """
    # Buffer handles — the pass treats these as opaque pointers and does
    # not inspect their types, only passes them through to DMA calls.
    h_input = tir.Var("input", "handle")
    h_kernel = tir.Var("kernel", "handle")
    h_bias = tir.Var("bias", "handle")
    h_scale = tir.Var("scale", "handle")
    h_shift = tir.Var("shift", "handle")
    h_output = tir.Var("output", "handle")

    call = tir.call_extern(
        "int32",
        kernel_name,
        h_input,          # args[1]
        h_kernel,         # args[2]
        h_bias,           # args[3]
        h_scale,          # args[4]
        h_shift,          # args[5]
        h_output,         # args[6]
        tir.const(C_in,    "int32"),   # args[7]
        tir.const(H_in,    "int32"),   # args[8]
        tir.const(W_in,    "int32"),   # args[9]
        tir.const(C_out,   "int32"),   # args[10]
        tir.const(KH,      "int32"),   # args[11]
        tir.const(KW,      "int32"),   # args[12]
        tir.const(stride_h,"int32"),   # args[13]  ← was incorrectly used
        tir.const(stride_w,"int32"),   # args[14]
        tir.const(pad_top, "int32"),   # args[15]  ← correct guard source
        tir.const(pad_bottom,"int32"), # args[16]
        tir.const(pad_left,"int32"),   # args[17]
        tir.const(pad_right,"int32"),  # args[18]
    )
    body = tir.Evaluate(call)
    return tir.PrimFunc(
        [h_input, h_kernel, h_bias, h_scale, h_shift, h_output],
        body,
    )


def _collect_allocate_sizes(func):
    """Return a list of all Allocate extents (as ints) in the PrimFunc body."""
    sizes = []

    def _visit(node):
        if isinstance(node, tir.Allocate):
            for ext in node.extents:
                if isinstance(ext, tir.IntImm):
                    sizes.append(int(ext.value))

    tir.stmt_functor.post_order_visit(func.body, _visit)
    return sizes


_L2_BUDGET = 2 * 1024 * 1024  # 2 MB — large enough to cache both input and weights


# ---------------------------------------------------------------------------
# Tests
# ---------------------------------------------------------------------------


def test_guard_uses_pad_top_not_stride_h():
    """Guard size must be pad_top * W_in * 1, not stride_h * W_in * 1.

    With stride_h=1 and pad_top=2, the correct guard is 2*W_in=56 bytes.
    The old (broken) code would give 1*W_in=28 (stride_h) or fall back to
    128 bytes — neither equals the expected 56 bytes.
    """
    C_in, H_in, W_in = 64, 28, 28
    C_out, KH, KW = 64, 3, 3
    stride_h = 1   # args[13] — was the source of the wrong guard
    pad_top = 2    # args[15] — the correct source; differs from stride_h

    func = _make_conv2d_primfunc(
        "mmalib_conv2d_i8",
        C_in, H_in, W_in, C_out, KH, KW,
        stride_h, stride_w=1, pad_top=pad_top, pad_bottom=2,
        pad_left=0, pad_right=0,
    )
    mod = tvm.IRModule({"mmalib_conv2d": func})
    mod_after = InjectMMALIBDMA(_L2_BUDGET)(mod)
    sizes = _collect_allocate_sizes(mod_after["mmalib_conv2d"])

    expected_guard = pad_top * W_in  # elem_bytes=1 for int8
    assert expected_guard in sizes, (
        f"Guard not found. Expected {expected_guard} bytes ({pad_top}*{W_in}*1). "
        f"Got allocations: {sorted(sizes)}. "
        f"If stride_h={stride_h} was used instead, guard would be {stride_h * W_in}."
    )


def test_guard_i16_conv2d_with_padding():
    """Guard size for i16 conv2d is pad_top * W_in * 2 (int16 = 2 bytes/elem)."""
    C_in, H_in, W_in = 32, 28, 28
    C_out, KH, KW = 32, 3, 3
    pad_top = 1

    func = _make_conv2d_primfunc(
        "mmalib_conv2d_i16",
        C_in, H_in, W_in, C_out, KH, KW,
        stride_h=1, stride_w=1,
        pad_top=pad_top, pad_bottom=1, pad_left=1, pad_right=1,
    )
    mod = tvm.IRModule({"mmalib_conv2d_i16_fn": func})
    mod_after = InjectMMALIBDMA(_L2_BUDGET)(mod)
    sizes = _collect_allocate_sizes(mod_after["mmalib_conv2d_i16_fn"])

    expected_guard = pad_top * W_in * 2  # elem_bytes=2 for int16
    assert expected_guard in sizes, (
        f"i16 guard not found. Expected {expected_guard} ({pad_top}*{W_in}*2). "
        f"Got allocations: {sorted(sizes)}."
    )


def test_guard_fallback_128_when_no_padding():
    """Guard falls back to 128 bytes when pad_top == 0."""
    func = _make_conv2d_primfunc(
        "mmalib_conv2d_i8",
        64, 28, 28, 64, 3, 3,
        stride_h=1, stride_w=1,
        pad_top=0, pad_bottom=0, pad_left=0, pad_right=0,
    )
    mod = tvm.IRModule({"mmalib_conv2d_nopad": func})
    mod_after = InjectMMALIBDMA(_L2_BUDGET)(mod)
    sizes = _collect_allocate_sizes(mod_after["mmalib_conv2d_nopad"])

    assert 128 in sizes, (
        f"Fallback 128-byte guard not found in allocations: {sorted(sizes)}"
    )


def test_guard_i8_unpadded_stride2():
    """Guard is 128 bytes (fallback) for stride-2 conv with no padding.

    This also verifies stride_h != pad_top does not cause a false-positive guard:
    stride_h=2 > 0 but pad_top=0, so the guard should be 128 not 2*W_in.
    """
    func = _make_conv2d_primfunc(
        "mmalib_conv2d_i8",
        64, 56, 56, 64, 3, 3,
        stride_h=2, stride_w=2,     # stride > 1, but no padding
        pad_top=0, pad_bottom=0, pad_left=0, pad_right=0,
    )
    mod = tvm.IRModule({"mmalib_conv2d_s2": func})
    mod_after = InjectMMALIBDMA(_L2_BUDGET)(mod)
    sizes = _collect_allocate_sizes(mod_after["mmalib_conv2d_s2"])

    # stride_h=2 and W_in=56 → if stride was misread as pad_top, guard=112
    wrong_guard = 2 * 56  # what the old code would compute if stride was used
    assert wrong_guard not in sizes or 128 in sizes, (
        f"Guard appears to be using stride_h ({wrong_guard}) instead of pad_top "
        f"(0→128). Allocations: {sorted(sizes)}"
    )
    assert 128 in sizes, (
        f"Expected 128-byte fallback guard. Got: {sorted(sizes)}"
    )
