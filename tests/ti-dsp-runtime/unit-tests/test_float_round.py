"""Float rounding on C7x is inline and rounds half to even.

The code generator prints nearbyintf(x), which TIR's round lowers to (every
quantize and every requantize in generated code), as tvm_c7x_nearbyint(x):
__float_to_int (VSPINT) for magnitudes below 2^23, x itself above. cl7x
compiles nearbyintf to an out-of-line call, about 140 cycles a value, which
also keeps the loop from software pipelining.

The source check needs no DSP. The numeric check runs on c7x_host and c7x_dload.

Usage:
    pytest test_float_round.py -v --dsp-mode=c7x_host
    pytest test_float_round.py -v --dsp-mode=c7x_dload --board beagley-ai
"""

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

_N = 4096


def _build_module():
    """main(x) = round(x)."""
    bb = relax.BlockBuilder()
    x = relax.Var("x", relax.TensorType([_N], "float32"))
    with bb.function("main", [x], attrs={"num_input": 1}):
        with bb.dataflow():
            out = bb.emit_output(relax.op.round(x))
        bb.emit_func_output(out)
    return bb.finalize()


def _inputs():
    """Ties, near-ties, quantize-range values and values beyond 2^23."""
    rng = np.random.default_rng(0)
    x = np.empty(_N, dtype=np.float32)
    ties = np.arange(-512, 512, dtype=np.float32) + 0.5
    x[:1024] = ties
    x[1024:2048] = np.nextafter(ties, np.float32(np.inf))
    x[2048:3072] = rng.uniform(-200, 200, 1024).astype(np.float32)
    big = np.exp2(rng.integers(23, 60, 1024)).astype(np.float32) * rng.choice([-1, 1], 1024)
    x[3072:] = big * rng.uniform(1, 2, 1024).astype(np.float32)
    return x


@pytest.mark.quick
def test_round_printed_inline():
    """No call to nearbyintf in generated code."""
    with tempfile.TemporaryDirectory() as tmp:
        gen = compile_for_dsp(
            _build_module(), "c_static_lib -mcpu=c7x -use-cpp-api=1", output_dir=Path(tmp)
        )
        src = "\n".join(p.read_text() for p in sorted(gen.glob("lib*.c")))
    calls = [line.strip() for line in src.splitlines() if "nearbyintf(" in line]
    assert not calls, f"nearbyintf still called: {calls[:3]}"
    assert "tvm_c7x_nearbyint((" in src


@pytest.mark.quick
def test_round_half_to_even(dsp_mode):
    """round(x) on the DSP equals numpy's (half to even) exactly."""
    if dsp_mode not in ("c7x_host", "c7x_dload"):
        pytest.skip("requires c7x_host or c7x_dload")
    x = _inputs()
    results = compile_and_run_dsp(
        mod=_build_module(),
        input_data=x,
        target_string=get_target_string(dsp_mode, use_cpp_api=True),
        execution_mode=dsp_mode,
    )
    out = results[f"{dsp_mode}_result"].reshape(x.shape)
    np.testing.assert_array_equal(out, np.round(x))
