"""Shared helpers for PT2E quantizer end-to-end DSP tests."""

import warnings

import numpy as np
import torch
import torch.nn as nn
from torchao.quantization.pt2e.quantize_pt2e import convert_pt2e, prepare_pt2e

import tvm
from tvm import relax
from tvm.relax.frontend.torch import C7xMMAQuantizer, from_exported_program

from dsp_utils import compile_and_run_dsp, get_target_string  # noqa: E402


def e2e_quantize_and_import(
    model: nn.Module,
    example_inputs: tuple,
    dtype: str = "int8",
    n_calibration_batches: int = 1,
) -> tuple[tvm.IRModule, np.ndarray, np.ndarray]:
    """Run the full PT2E → TVM import pipeline.

    Args:
        model:                float PyTorch model (eval mode set internally)
        example_inputs:       tuple of example tensors defining input shapes
        dtype:                "int8" or "int16"
        n_calibration_batches: number of random batches to run through the
                              observer.  1 suffices for small models; use 10+
                              for full classification networks.

    Returns:
        mod:      Relax IRModule with params bound, ready for compile_and_run_dsp
        input_np: numpy array matching example_inputs[0]
        ref_np:   PyTorch quantized model CPU output (correctness reference)
    """
    model.eval()
    exported = torch.export.export(model, example_inputs).module()

    quantizer = C7xMMAQuantizer(dtype=dtype)
    prepared = prepare_pt2e(exported, quantizer)
    with torch.no_grad():
        for _ in range(n_calibration_batches):
            prepared(torch.randn_like(example_inputs[0]))

    with warnings.catch_warnings():
        warnings.filterwarnings("ignore", message="erase_node")
        quantized_pt = convert_pt2e(prepared)

    with torch.no_grad():
        ref_np = quantized_pt(*example_inputs).numpy()

    # convert_pt2e returns a GraphModule, not an ExportedProgram.
    # Re-export to get the ExportedProgram that from_exported_program requires.
    quantized_ep = torch.export.export(quantized_pt, example_inputs)
    mod = from_exported_program(quantized_ep, keep_params_as_input=True)
    mod, params = relax.frontend.detach_params(mod)
    func_params_dict = dict(zip(mod["main"].params[1:], params["main"]))
    mod = relax.transform.BindParams(func_name="main", params=func_params_dict)(mod)  # pyright: ignore[reportArgumentType]

    input_np = example_inputs[0].numpy()
    return mod, input_np, ref_np


def run_and_check(
    mod: tvm.IRModule,
    input_np: np.ndarray,
    ref_np: np.ndarray,
    dsp_mode: str,
    record_cycles,
    cycles_key: str,
    max_diff: int = 2,
) -> None:
    """Compile with MMALIB, run, assert max_diff correctness, record cycles."""
    target = get_target_string(dsp_mode, use_cpp_api=True) + " -mmalib=1"
    results = compile_and_run_dsp(
        mod=mod,
        input_data=input_np,
        target_string=target,
        execution_mode=dsp_mode,
    )
    result_key = "c7x_host_result" if dsp_mode == "c7x_host" else "c7x_dload_result"
    dsp_out = results.get(result_key)
    assert dsp_out is not None, f"No {result_key} in results"

    dsp_i8 = dsp_out.reshape(ref_np.shape).astype(np.int32)
    ref_i32 = ref_np.astype(np.int32)
    diff = np.abs(dsp_i8 - ref_i32)
    actual_max = int(diff.max())
    assert actual_max <= max_diff, (
        f"max_diff {actual_max} > {max_diff}: "
        f"DSP output diverges from PyTorch quantized reference"
    )

    cycles = results.get("c7x_dload_cycles", 0)
    record_cycles(cycles_key, cycles)
