"""Reduction accumulators are printed as scalars, not one-element arrays.

PromoteReductionAccumulator rewrites ``out[i] = out[i] + x[...]`` over a reduction
loop to accumulate in a one-element "local" buffer and store the result once.
c_static_lib prints that buffer as a scalar (``float out_acc_0;``). Printed as
``float out_acc_0[1];``, cl7x 6.2.1 keeps it in memory, and the loop-carried
dependency through it raises the loop's initiation interval several times over.

The source check needs no DSP. The numeric check runs on c7x_host and c7x_dload.

Usage:
    pytest test_reduction_accumulator.py -v --dsp-mode=c7x_host
    pytest test_reduction_accumulator.py -v --dsp-mode=c7x_dload --board beagley-ai
"""

import re
import sys
import tempfile
from pathlib import Path

import numpy as np
import pytest

from tvm import relax

_THIS_DIR = Path(__file__).parent
_DSP_CPP_DIR = _THIS_DIR.parent / "dsp-cpp"
sys.path.insert(0, str(_DSP_CPP_DIR))

from dsp_utils import compile_and_run_dsp, compile_for_dsp, get_target_string  # noqa: E402

_M, _K, _N = 8, 64, 16


def _build_module():
    """main(a, b) = concat([matmul(a, b), sum(a, axis=1)], axis=1)."""
    bb = relax.BlockBuilder()
    a = relax.Var("a", relax.TensorType([_M, _K], "float32"))
    b = relax.Var("b", relax.TensorType([_K, _N], "float32"))
    with bb.function("main", [a, b], attrs={"num_input": 2}):
        with bb.dataflow():
            mm = bb.emit(relax.op.matmul(a, b))
            s = bb.emit(relax.op.sum(a, axis=1, keepdims=True))
            out = bb.emit_output(relax.op.concat([mm, s], axis=1))
        bb.emit_func_output(out)
    return bb.finalize()


@pytest.mark.quick
def test_accumulator_printed_as_scalar():
    """Promoted accumulators are declared as scalars and never as x_acc_N[1]."""
    with tempfile.TemporaryDirectory() as tmp:
        gen = compile_for_dsp(
            _build_module(), "c_static_lib -mcpu=c7x -use-cpp-api=1", output_dir=Path(tmp)
        )
        src = "\n".join(p.read_text() for p in sorted(gen.glob("lib*.c")))
    arrays = re.findall(r"\b\w+_acc_\d+\w*\s*\[\s*1\s*\]\s*;", src)
    assert not arrays, f"accumulators still printed as arrays: {arrays[:3]}"
    scalars = re.findall(r"\bfloat\s+\w+_acc_\d+\s*;", src)
    assert scalars, "no scalar accumulator declared; did PromoteReductionAccumulator run?"


@pytest.mark.quick
def test_accumulator_reduction(dsp_mode):
    """Matmul and row sum through scalar accumulators match numpy."""
    if dsp_mode not in ("c7x_host", "c7x_dload"):
        pytest.skip("requires c7x_host or c7x_dload")
    rng = np.random.default_rng(0)
    a = rng.standard_normal((_M, _K)).astype(np.float32)
    b = rng.standard_normal((_K, _N)).astype(np.float32)
    ref = np.concatenate([a @ b, a.sum(axis=1, keepdims=True)], axis=1)
    results = compile_and_run_dsp(
        mod=_build_module(),
        input_data=(a, b),
        target_string=get_target_string(dsp_mode, use_cpp_api=True),
        execution_mode=dsp_mode,
    )
    out = results[f"{dsp_mode}_result"].reshape(ref.shape)
    np.testing.assert_allclose(out, ref, rtol=1e-4, atol=1e-4)
