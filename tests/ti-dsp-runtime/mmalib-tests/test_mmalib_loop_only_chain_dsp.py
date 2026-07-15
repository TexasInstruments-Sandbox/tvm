"""Regression test: chain of mmalib_conv2d_i8_grouped_loop calls within a
single inference, matching how grouped conv2d layers actually run in a
real multi-layer model (e.g. ResNeXt101's 33 grouped conv2 layers).

This is the sole grouped-conv path for Step 13. An earlier native
single-call path (MMALIB's numGroupsPerKernel field on the bias-fused
convolveBias_row kernel) was implemented and hardware-tested, but abandoned
after it produced intermittent silent corruption and hangs on real
hardware when chained within one inference -- code-archaeology of TIDL
confirmed that exact combination (bias-fused kernel + grouped mode) has no
validation coverage anywhere in TI's own software; TIDL always pairs
grouped mode with the non-bias kernel instead. mmalib_conv2d_i8_grouped_loop
avoids that risk entirely: it is a single call_extern whose C++
implementation loops over groups internally via the already-proven
conv2d_impl, never touching numGroupsPerKernel. See
docs/dsp/quantized_model_optimization.md Step 13 for the full
investigation. This test guards against that class of failure (or a new
one introduced by this wrapper) recurring when chained within one
inference.

Usage:
    pytest test_mmalib_loop_only_chain_dsp.py -v --dsp-mode=c7x_dload
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


def _emit_grouped_loop_call(
    bb, x, C_in, C_out, H_in, W_in, groups, stride, kernel_np, bias_np, scale_np, shift_np
):
    H_out = (H_in + 2 - 3) // stride + 1
    W_out = (W_in + 2 - 3) // stride + 1
    kernel_c = relax.Constant(kernel_np)
    bias_c = relax.Constant(bias_np)
    scale_c = relax.Constant(scale_np)
    shift_c = relax.Constant(shift_np)

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
            C_in,
            H_in,
            W_in,
            C_out,
            3,
            3,
            stride,
            stride,
            1,
            1,
            1,
            1,
            groups,
        )

    def te_conv(data_t, weight_t, bias_t, scale_t, shift_t):
        return te.extern(
            (1, C_out, H_out, W_out),
            [data_t, weight_t, bias_t, scale_t, shift_t],
            fcompute,
            name="grouped_loop_out",
            dtype="int8",
        )

    out = bb.emit_te(te_conv, x, kernel_c, bias_c, scale_c, shift_c, primfunc_name_hint=_KERNEL)
    return out, H_out, W_out


def _build_loop_chain(rng, n_calls, C, H, W, groups, stride):
    consts = []
    bb = relax.BlockBuilder()
    x_var = relax.Var("x", relax.TensorStructInfo([1, C, H, W], "int8"))
    with bb.function("main", [x_var], attrs={"num_input": 1}):
        with bb.dataflow():
            cur, cur_H, cur_W = x_var, H, W
            for _ in range(n_calls):
                kernel_np = rng.integers(-4, 4, (C, C // groups, 3, 3), dtype=np.int8)
                bias_np = rng.integers(-100, 100, C, dtype=np.int32)
                scale_np = rng.integers(1, 3, C, dtype=np.uint8)
                shift_np = rng.integers(2, 5, C, dtype=np.uint8)
                consts.append((kernel_np, bias_np, scale_np, shift_np))
                cur, cur_H, cur_W = _emit_grouped_loop_call(
                    bb,
                    cur,
                    C,
                    C,
                    cur_H,
                    cur_W,
                    groups,
                    stride,
                    kernel_np,
                    bias_np,
                    scale_np,
                    shift_np,
                )
                stride = 1  # only the first call downsamples, matches finite spatial size
            result = bb.emit_output(cur)
        bb.emit_func_output(result)
    return bb.finalize(), consts, cur_H, cur_W


def _numpy_loop_chain_reference(input_np, consts, C, H, W, groups, stride):
    cur, cur_H, cur_W = input_np, H, W
    for kernel_np, bias_np, scale_np, shift_np in consts:
        cur = _numpy_grouped_conv2d_i8(
            cur, kernel_np, bias_np, scale_np, shift_np, C, cur_H, cur_W, C, 3, 3, stride, 1, groups
        )
        cur_H = (cur_H + 2 - 3) // stride + 1
        cur_W = (cur_W + 2 - 3) // stride + 1
        stride = 1
    return cur


def _check_loop_chain(dsp_mode, n_calls):
    if dsp_mode not in ("c7x_host", "c7x_dload"):
        pytest.skip("requires c7x_host or c7x_dload")
    C, H, W, groups, stride = 256, 14, 14, 32, 2
    rng = np.random.default_rng(42)
    mod, consts, H_out, W_out = _build_loop_chain(rng, n_calls, C, H, W, groups, stride)
    input_np = rng.integers(-8, 8, (1, C, H, W), dtype=np.int8)
    ref = _numpy_loop_chain_reference(input_np, consts, C, H, W, groups, stride)

    target = get_target_string(dsp_mode, use_cpp_api=True)
    results = compile_and_run_dsp(
        mod=mod, input_data=input_np, target_string=target, execution_mode=dsp_mode, profile=False
    )
    out = results[f"{dsp_mode}_result"].reshape(1, C, H_out, W_out)
    assert np.array_equal(out, ref), (
        f"grouped_loop chain mismatch ({n_calls} calls): max_err="
        f"{np.abs(out.astype(int) - ref.astype(int)).max()}"
    )


@pytest.mark.quick
@pytest.mark.parametrize("n_calls", [2, 3])
def test_loop_only_chain(dsp_mode, n_calls):
    """Chain of N mmalib_conv2d_i8_grouped_loop calls within one inference."""
    _check_loop_chain(dsp_mode, n_calls)


def test_loop_only_chain_stress(dsp_mode):
    """16-call chain (16 x 32 groups = 512 group-level MMALIB calls in one
    inference) -- much closer to ResNeXt101's real ~33 layers x 32 groups
    per inference than the 2-3 call smoke tests above. Not marked @quick:
    the numpy reference is O(n_calls) slower to compute in pure Python.
    See docs/dsp/quantized_model_optimization.md Step 13: this class of
    test (many chained calls within one inference) is what caught the
    abandoned native path's intermittent corruption; the smoke tests
    alone reach only 64-96 group-level calls, far short of where that bug
    actually manifested."""
    _check_loop_chain(dsp_mode, 16)
