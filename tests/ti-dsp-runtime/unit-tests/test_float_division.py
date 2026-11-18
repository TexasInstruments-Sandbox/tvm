"""Float division, rsqrt and sqrt on C7x are accurate to float32 precision.

The code generator prints scalar float division as a * tvm_c7x_recip(b),
1 / sqrtf(x) as tvm_c7x_rsqrt(x) and sqrtf(x) as x * tvm_c7x_rsqrt(x). The
helpers refine the __recip / __recip_sqrt estimates, which on their own are
good to about 8 bits (relative error up to 0.46%): enough to move a 15-block
QuartzNet's log-probabilities by up to 12 through its runtime BatchNorms.

The source check needs no DSP. The numeric check runs on c7x_host and c7x_dload.

Usage:
    pytest test_float_division.py -v --dsp-mode=c7x_host
    pytest test_float_division.py -v --dsp-mode=c7x_dload --board beagley-ai
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

_N = 4096


def _build_module():
    """main(a, b, c) = concat([a / b, rsqrt(c), sqrt(c)])."""
    bb = relax.BlockBuilder()
    a = relax.Var("a", relax.TensorType([_N], "float32"))
    b = relax.Var("b", relax.TensorType([_N], "float32"))
    c = relax.Var("c", relax.TensorType([_N], "float32"))
    with bb.function("main", [a, b, c], attrs={"num_input": 3}):
        with bb.dataflow():
            q = bb.emit(relax.op.divide(a, b))
            r = bb.emit(relax.op.rsqrt(c))
            s = bb.emit(relax.op.sqrt(c))
            out = bb.emit_output(relax.op.concat([q, r, s], axis=0))
        bb.emit_func_output(out)
    return bb.finalize()


def _inputs():
    """Values spread over many binades, so every estimate's error shows."""
    rng = np.random.default_rng(0)
    a = (rng.standard_normal(_N) * np.exp2(rng.integers(-20, 20, _N))).astype(np.float32)
    b = (rng.uniform(1, 2, _N) * np.exp2(rng.integers(-30, 30, _N))).astype(np.float32)
    b[::2] *= -1
    c = (rng.uniform(1, 4, _N) * np.exp2(rng.integers(-30, 30, _N))).astype(np.float32)
    return a, b, c


@pytest.mark.quick
def test_division_uses_refined_helpers():
    """No bare __recip / __recip_sqrt outside the helpers' definitions."""
    with tempfile.TemporaryDirectory() as tmp:
        gen = compile_for_dsp(
            _build_module(), "c_static_lib -mcpu=c7x -use-cpp-api=1", output_dir=Path(tmp)
        )
        src = "\n".join(p.read_text() for p in sorted(gen.glob("lib*.c")))
    body = re.sub(r"static inline float tvm_c7x_r\w+\(float \w\) \{.*?\n\}", "", src, flags=re.S)
    bare = [line.strip() for line in body.splitlines() if re.search(r"__recip(_sqrt)?\(", line)]
    assert not bare, f"unrefined estimates in generated code: {bare[:3]}"
    assert "tvm_c7x_recip((" in body and "tvm_c7x_rsqrt((" in body


@pytest.mark.quick
def test_division_precision(dsp_mode):
    """a / b, rsqrt(c) and sqrt(c) within a few ulp of float32 numpy."""
    if dsp_mode not in ("c7x_host", "c7x_dload"):
        pytest.skip("requires c7x_host or c7x_dload")
    a, b, c = _inputs()
    ref = np.concatenate([a / b, 1 / np.sqrt(c), np.sqrt(c)])
    results = compile_and_run_dsp(
        mod=_build_module(),
        input_data=(a, b, c),
        target_string=get_target_string(dsp_mode, use_cpp_api=True),
        execution_mode=dsp_mode,
    )
    out = results[f"{dsp_mode}_result"].reshape(ref.shape)
    # The estimates alone are off by up to 0.46%; refined, by about 2 ulp.
    np.testing.assert_allclose(out, ref, rtol=4 * float(np.finfo(np.float32).eps), atol=0)
