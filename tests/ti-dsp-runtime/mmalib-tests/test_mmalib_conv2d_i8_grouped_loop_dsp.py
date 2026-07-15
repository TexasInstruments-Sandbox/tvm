"""Correctness test for mmalib_conv2d_i8_grouped_loop.

Direct call_extern test comparing against a numpy grouped-conv reference,
for the four distinct stage shapes that actually occur in ResNeXt101-32x8d
(see docs/dsp/quantized_model_optimization.md Step 13). This is the sole
grouped-conv path: a single call whose C++ implementation loops over
groups internally via the already-proven conv2d_impl (see
mmalib_wrappers.cpp), never touching MMALIB's untested numGroupsPerKernel
field.

Usage:
    pytest test_mmalib_conv2d_i8_grouped_loop_dsp.py -v --dsp-mode=c7x_host
    pytest test_mmalib_conv2d_i8_grouped_loop_dsp.py -v --dsp-mode=c7x_dload
"""

import sys
from pathlib import Path

import numpy as np
import pytest

from tvm import relax, te, tir

_THIS_DIR = Path(__file__).parent
_DSP_CPP_DIR = _THIS_DIR.parent / "dsp-cpp"
sys.path.insert(0, str(_DSP_CPP_DIR))
sys.path.insert(0, str(_THIS_DIR))

from _grouped_conv_ref import numpy_grouped_conv2d_i8 as _numpy_grouped_conv2d_i8  # noqa: E402
from dsp_utils import compile_and_run_dsp, get_target_string  # noqa: E402

_KERNEL = "mmalib_conv2d_i8_grouped_loop"


def _build_grouped_conv_module(
    kernel_np, bias_np, scale_np, shift_np, C_in, H_in, W_in, C_out, KH, KW, stride, pad, groups
):
    H_out = (H_in + 2 * pad - KH) // stride + 1
    W_out = (W_in + 2 * pad - KW) // stride + 1

    kernel_c = relax.Constant(kernel_np)
    bias_c = relax.Constant(bias_np)
    scale_c = relax.Constant(scale_np)
    shift_c = relax.Constant(shift_np)

    def te_conv(x_t, k_t, b_t, s_t, sh_t):
        def fcompute(ins, outs):
            return tir.call_extern(
                "int32",
                _KERNEL,
                ins[0].data,
                ins[1].data,
                ins[2].data,
                ins[3].data,
                ins[4].data,
                outs[0].data,
                tir.IntImm("int32", C_in),
                tir.IntImm("int32", H_in),
                tir.IntImm("int32", W_in),
                tir.IntImm("int32", C_out),
                tir.IntImm("int32", KH),
                tir.IntImm("int32", KW),
                tir.IntImm("int32", stride),
                tir.IntImm("int32", stride),
                tir.IntImm("int32", pad),
                tir.IntImm("int32", pad),
                tir.IntImm("int32", pad),
                tir.IntImm("int32", pad),
                tir.IntImm("int32", groups),
            )

        return te.extern(
            [1, C_out, H_out, W_out],
            [x_t, k_t, b_t, s_t, sh_t],
            fcompute,
            name="grouped_conv_loop_out",
            dtype="int8",
        )

    bb = relax.BlockBuilder()
    x_var = relax.Var("x", relax.TensorStructInfo([1, C_in, H_in, W_in], "int8"))
    with bb.function("main", [x_var], attrs={"num_input": 1}):
        with bb.dataflow():
            out = bb.emit_te(
                te_conv, x_var, kernel_c, bias_c, scale_c, shift_c, primfunc_name_hint=_KERNEL
            )
            result = bb.emit_output(out)
        bb.emit_func_output(result)
    return bb.finalize()


def _run_grouped_conv(
    dsp_mode,
    input_np,
    kernel_np,
    bias_np,
    scale_np,
    shift_np,
    C_in,
    H_in,
    W_in,
    C_out,
    KH,
    KW,
    stride,
    pad,
    groups,
):
    mod = _build_grouped_conv_module(
        kernel_np, bias_np, scale_np, shift_np, C_in, H_in, W_in, C_out, KH, KW, stride, pad, groups
    )
    target = get_target_string(dsp_mode, use_cpp_api=True)
    results = compile_and_run_dsp(
        mod=mod, input_data=input_np, target_string=target, execution_mode=dsp_mode, profile=False
    )
    H_out = (H_in + 2 * pad - KH) // stride + 1
    W_out = (W_in + 2 * pad - KW) // stride + 1
    return results[f"{dsp_mode}_result"].reshape(1, C_out, H_out, W_out)


def _check_grouped_conv(dsp_mode, C_in, H_in, W_in, C_out, KH, KW, stride, pad, groups, seed):
    if dsp_mode not in ("c7x_host", "c7x_dload"):
        pytest.skip("requires c7x_host or c7x_dload")

    rng = np.random.default_rng(seed)
    input_np = rng.integers(-16, 16, (1, C_in, H_in, W_in), dtype=np.int8)
    kernel_np = rng.integers(-8, 8, (C_out, C_in // groups, KH, KW), dtype=np.int8)
    bias_np = rng.integers(-200, 200, C_out, dtype=np.int32)
    scale_np = rng.integers(1, 4, C_out, dtype=np.uint8)
    shift_np = rng.integers(0, 4, C_out, dtype=np.uint8)

    ref = _numpy_grouped_conv2d_i8(
        input_np,
        kernel_np,
        bias_np,
        scale_np,
        shift_np,
        C_in,
        H_in,
        W_in,
        C_out,
        KH,
        KW,
        stride,
        pad,
        groups,
    )
    out = _run_grouped_conv(
        dsp_mode,
        input_np,
        kernel_np,
        bias_np,
        scale_np,
        shift_np,
        C_in,
        H_in,
        W_in,
        C_out,
        KH,
        KW,
        stride,
        pad,
        groups,
    )
    assert np.array_equal(out, ref), (
        f"mmalib_conv2d_i8_grouped_loop mismatch (C_in={C_in} C_out={C_out} "
        f"groups={groups} stride={stride}): "
        f"max_err={np.abs(out.astype(int) - ref.astype(int)).max()}"
    )


# ResNeXt101-32x8d's four distinct stage shapes, each with both stride=1
# (most blocks) and stride=2 (stage-transition block 0).


@pytest.mark.quick
def test_grouped_conv_loop_layer1(dsp_mode):
    """layer1: C=256, groups=32, No=8 (small No)."""
    _check_grouped_conv(dsp_mode, 256, 14, 14, 256, 3, 3, 1, 1, 32, seed=0)


@pytest.mark.quick
def test_grouped_conv_loop_layer2_stride1(dsp_mode):
    """layer2 (stride=1 blocks): C=512, groups=32, No=16."""
    _check_grouped_conv(dsp_mode, 512, 10, 10, 512, 3, 3, 1, 1, 32, seed=1)


@pytest.mark.quick
def test_grouped_conv_loop_layer2_stride2(dsp_mode):
    """layer2 block 0 (stage transition): C_in=256, C_out=512, stride=2."""
    _check_grouped_conv(dsp_mode, 256, 20, 20, 512, 3, 3, 2, 1, 32, seed=2)


@pytest.mark.quick
def test_grouped_conv_loop_layer3(dsp_mode):
    """layer3: C=1024, groups=32, No=32 -- exactly MMA_SIZE boundary."""
    _check_grouped_conv(dsp_mode, 1024, 7, 7, 1024, 3, 3, 1, 1, 32, seed=3)


@pytest.mark.quick
def test_grouped_conv_loop_layer3_stride2(dsp_mode):
    """layer3.0.conv2 (real ResNeXt101-32x8d shape): C_in=C_out=1024,
    groups=32, No=32 (exactly MMA_SIZE boundary) AND stride=2 -- exercises
    conv2d_impl's internal OC-tiling loop (triggered by stride>1) nested
    inside the outer group loop, at the exact boundary case."""
    _check_grouped_conv(dsp_mode, 1024, 14, 14, 1024, 3, 3, 2, 1, 32, seed=5)


@pytest.mark.quick
def test_grouped_conv_loop_layer4(dsp_mode):
    """layer4: C=2048, groups=32, No=64 -- exceeds MMA_SIZE."""
    _check_grouped_conv(dsp_mode, 2048, 7, 7, 2048, 3, 3, 1, 1, 32, seed=4)


@pytest.mark.quick
def test_grouped_conv_loop_layer4_stride2(dsp_mode):
    """layer4.0.conv2 (real ResNeXt101-32x8d shape): C_in=C_out=2048,
    groups=32, No=64 (exceeds MMA_SIZE) AND stride=2 -- the specific
    shape/stride combination never covered by a dedicated test before
    (see docs/dsp/quantized_model_optimization.md Step 13's own
    investigation, which found exactly this class of untested combination
    was where the abandoned native numGroupsPerKernel path silently
    corrupted data on real hardware)."""
    _check_grouped_conv(dsp_mode, 2048, 14, 14, 2048, 3, 3, 2, 1, 32, seed=6)
