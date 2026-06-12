"""Shared helpers for PT2E quantizer end-to-end DSP tests.

Three public functions covering the full quantization and compilation pipeline:

  quantize_pt2e            –  float PyTorch model → calibrated Q/DQ GraphModule
  e2e_quantize_and_import  –  Q/DQ GraphModule → Relax IRModule with weights bound
  run_and_check            –  Relax IRModule → compile → run on DSP
                               → assert correctness vs. PyTorch reference

See docs/dsp/c7x_mma_quantizer.md for the full pipeline description.
"""

import warnings

import numpy as np
import torch
import torch.nn as nn
from dsp_utils import compile_and_run_dsp, get_target_string  # noqa: E402
from torch.fx import GraphModule
from torchao.quantization.pt2e.quantize_pt2e import convert_pt2e, prepare_pt2e

import tvm
from tvm import relax
from tvm.relax.frontend.torch import C7xMMAQuantizer, from_exported_program


def quantize_pt2e(
    model: nn.Module,
    example_inputs: tuple,
    quantizer: C7xMMAQuantizer,
    n_calibration_batches: int = 1,
) -> GraphModule:
    """Export → prepare → calibrate → convert. Returns the converted GraphModule.

    This is the pure-PyTorch half of the pipeline: it produces a Q/DQ graph but
    does not import into TVM.  Use e2e_quantize_and_import for the full pipeline.

    Args:
        model:                float PyTorch model (eval mode set internally)
        example_inputs:       tuple of example tensors; used for export and calibration
        quantizer:            configured C7xMMAQuantizer instance
        n_calibration_batches: number of times to run example_inputs through the
                              observers.  More than 1 is only useful when the caller
                              supplies different inputs on each iteration.

    Returns:
        GraphModule with Q/DQ nodes replacing the observer modules.
    """
    model.eval()
    # torch.export.export() returns an ExportedProgram; .module() unwraps it to a
    # GraphModule, which is what prepare_pt2e expects.
    exported = torch.export.export(model, example_inputs).module()

    # prepare_pt2e calls quantizer.annotate() to read the quantization specs, then
    # inserts MinMaxObserver / PerChannelMinMaxObserver modules at each annotated
    # tensor edge.  Running inputs through the prepared model fills in the observed
    # min/max ranges that will become the Q/DQ scale factors.
    prepared = prepare_pt2e(exported, quantizer)
    with torch.no_grad():
        for _ in range(n_calibration_batches):
            prepared(*example_inputs)

    with warnings.catch_warnings():
        # convert_pt2e removes observer nodes from the graph; torch emits a
        # harmless UserWarning("erase_node") for each one.
        warnings.filterwarnings("ignore", message="erase_node")
        # convert_pt2e replaces observer modules with explicit quantize/dequantize
        # (Q/DQ) nodes, producing a graph of the form q → dq → op → q → dq → ...
        return convert_pt2e(prepared)


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
        n_calibration_batches: passed through to quantize_pt2e

    Returns:
        mod:      Relax IRModule with params bound, ready for compile_and_run_dsp
        input_np: numpy array matching example_inputs[0]
        ref_np:   PyTorch quantized model CPU output (correctness reference)
    """
    quantized_pt = quantize_pt2e(
        model, example_inputs, C7xMMAQuantizer(dtype=dtype), n_calibration_batches
    )

    with torch.no_grad():
        # Capture PyTorch's simulated-quantization output now, before re-export,
        # as the correctness reference for the DSP result.
        ref_np = quantized_pt(*example_inputs).numpy()

    # convert_pt2e returns a GraphModule, not an ExportedProgram.
    # Re-export to get the ExportedProgram that from_exported_program requires.
    quantized_ep = torch.export.export(quantized_pt, example_inputs)

    # keep_params_as_input=True imports weights and biases as extra function
    # arguments rather than embedding them as constants in the IR.
    mod = from_exported_program(quantized_ep, keep_params_as_input=True)

    # detach_params separates those extra arguments into a Python dict; BindParams
    # then folds them back as constants inside the function.  This is the standard
    # Relax pattern for binding learned weights into a compiled module.
    mod, params = relax.frontend.detach_params(mod)

    # params[0] of "main" is the runtime input tensor; [1:] are the weight/bias params.
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
    """Compile with MMALIB, run on DSP, and assert output correctness.

    The compiled target includes ``-mmalib=1`` which activates the
    FuseMMALIBQDQConv2d / FuseMMALIBQDQDwConv2d / FuseMMALIBQDQFC /
    FuseInt8ResidualAdd fusion passes in the c_static backend.

    Correctness is checked as ``max(|dsp_output - ref|) <= max_diff``.
    The default ``max_diff=2`` allows ±1 LSB of int8 arithmetic rounding
    difference between PyTorch's float-simulated quantization and actual
    hardware integer arithmetic, with one count of extra headroom.

    Args:
        mod:          Relax IRModule with weights bound (from e2e_quantize_and_import)
        input_np:     Runtime input as a numpy array
        ref_np:       PyTorch quantized reference output (from e2e_quantize_and_import)
        dsp_mode:     "c7x_host" (host emulation) or "c7x_dload" (AM67A hardware)
        record_cycles: pytest fixture (conftest.py) that logs cycle counts to cycles.csv
        cycles_key:   name under which to store the cycle count
        max_diff:     maximum tolerated per-element absolute difference
    """
    # -mmalib=1 activates the FuseMMALIBQDQ* passes in the c_static backend.
    target = get_target_string(dsp_mode, use_cpp_api=True) + " -mmalib=1"
    results = compile_and_run_dsp(
        mod=mod,
        input_data=input_np,
        target_string=target,
        execution_mode=dsp_mode,
    )
    # compile_and_run_dsp tags each mode's output under its own key in the results dict.
    result_key = "c7x_host_result" if dsp_mode == "c7x_host" else "c7x_dload_result"
    dsp_out = results.get(result_key)
    assert dsp_out is not None, f"No {result_key} in results"

    # Cast to int32 before subtraction to avoid int8 overflow in the element-wise diff.
    dsp_i8 = dsp_out.reshape(ref_np.shape).astype(np.int32)
    ref_i32 = ref_np.astype(np.int32)
    diff = np.abs(dsp_i8 - ref_i32)
    actual_max = int(diff.max())
    assert actual_max <= max_diff, (
        f"max_diff {actual_max} > {max_diff}: "
        f"DSP output diverges from PyTorch quantized reference"
    )
    # c7x_dload_cycles is only populated for hardware runs; host emulation returns 0.
    cycles = results.get("c7x_dload_cycles", 0)
    record_cycles(cycles_key, cycles)
