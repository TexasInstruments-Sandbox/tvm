"""
Verifies the mmalib_conv2d_i8 C_out>1024 tiling fix: conv2d_impl in
mmalib_wrappers.cpp used to silently return -1 (output never written)
for C_out > 1024, a stale guard left over from when default bias/scale/
shift used fixed [1024]-sized stack buffers -- those were long since
converted to dynamic Workspace allocation, but the guard wasn't removed.
"""
import numpy as np
import pytest

from test_mmalib_conv2d_i8_dsp import _create_qdq_conv2d_model, _numpy_qdq_conv2d_i8

import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent.parent / "dsp-cpp"))
from dsp_utils import compile_and_run_dsp, get_target_string  # noqa: E402


@pytest.mark.parametrize("c_out", [512, 1024, 1280, 2048])
def test_conv2d_cout_boundary(dsp_mode, record_cycles, c_out):
    if dsp_mode not in ("c7x_host", "c7x_dload"):
        pytest.skip("requires c7x_host or c7x_dload")
    c_in, h, w, kh, kw, padding = 64, 4, 4, 1, 1, 0

    mod, input_data, kernel_data, d_scale, w_scale, o_scale, bias_data = _create_qdq_conv2d_model(
        c_in, h, w, c_out, kh, kw, padding
    )
    ref_output = _numpy_qdq_conv2d_i8(
        input_data, kernel_data, d_scale, w_scale, o_scale, bias_data, padding
    )

    result_key = "c7x_host_result" if dsp_mode == "c7x_host" else "c7x_dload_result"
    target_mmalib = get_target_string(dsp_mode, use_cpp_api=True) + " -mmalib=1"
    results = compile_and_run_dsp(mod=mod, input_data=input_data, target_string=target_mmalib,
                                   execution_mode=dsp_mode, profile=False)
    dsp_output = results.get(result_key)
    assert dsp_output is not None
    dsp_output_i8 = dsp_output.astype(np.int8).reshape(ref_output.shape)

    diff = np.abs(dsp_output_i8.astype(np.int32) - ref_output.astype(np.int32))
    frac_zero_dsp = (dsp_output_i8 == 0).mean()
    frac_zero_ref = (ref_output == 0).mean()
    print(f"\nc_out={c_out}: max_diff={diff.max()}, dsp_zero_frac={frac_zero_dsp:.3f}, ref_zero_frac={frac_zero_ref:.3f}")
    assert diff.max() <= 2, (
        f"c_out={c_out}: max_diff={diff.max()} "
        f"(dsp_zero_frac={frac_zero_dsp:.3f} vs ref_zero_frac={frac_zero_ref:.3f})"
    )
