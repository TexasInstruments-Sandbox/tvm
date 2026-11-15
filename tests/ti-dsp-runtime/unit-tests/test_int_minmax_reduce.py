"""Integer max/min reductions must not use std::max/std::min.

A max or min reduction lowers to an accumulator in the output buffer,
``red[r] = max(red[r], x[...])``. std::max returns a reference to its larger
argument, so this becomes a load through ``cond ? &x[...] : &red[r]``. With
``red`` restrict-qualified, cl7x 5.0.1.LTS at -O2/-O3 moves the stores to
red[r] below those loads, and whenever the running value doesn't change the
result is whatever red[r] held before the call. c_static_lib prints integer
max/min through by-value helpers (``tvm_max``/``tvm_min``) instead.

The source check needs no DSP. The reduction runs on c7x_host and c7x_dload;
the bug only shows on hardware, where the rows below that keep their first
element as the running max or min come back as stale memory.

Usage:
    pytest test_int_minmax_reduce.py -v --dsp-mode=c7x_host
    pytest test_int_minmax_reduce.py -v --dsp-mode=c7x_dload --board beagley-ai
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

_ROWS, _COLS = 16, 50


def _build_minmax_module(dtype):
    """main(x) = concat([max(x, axis=1), min(x, axis=1)], axis=1)."""
    bb = relax.BlockBuilder()
    x = relax.Var("x", relax.TensorType([_ROWS, _COLS], dtype))
    with bb.function("main", [x], attrs={"num_input": 1}):
        with bb.dataflow():
            mx = bb.emit(relax.op.max(x, axis=1, keepdims=True))
            mn = bb.emit(relax.op.min(x, axis=1, keepdims=True))
            out = bb.emit_output(relax.op.concat([mx, mn], axis=1))
        bb.emit_func_output(out)
    return bb.finalize()


def _rows(dtype):
    """Rows whose max or min is their first element, plus random rows."""
    info = np.iinfo(dtype)
    x = np.empty((_ROWS, _COLS), dtype=dtype)
    for r, v in enumerate((37, -5, info.max, info.min)):
        x[r] = v  # all equal: no element after the first changes max or min
    x[4] = np.linspace(100, -100, _COLS)  # descending: max is first
    x[5] = np.linspace(-100, 100, _COLS)  # ascending: min is first
    x[6] = 7
    x[6, 0] = 9  # max is first, then constant
    x[7] = 7
    x[7, 0] = 5  # min is first, then constant
    rng = np.random.default_rng(0)
    x[8:] = rng.integers(-100, 100, (_ROWS - 8, _COLS), dtype=dtype)
    return x


@pytest.mark.quick
@pytest.mark.parametrize("dtype", ["int8", "int32"])
def test_int_minmax_printed_by_value(dtype):
    """Integer max/min are printed as tvm_max<T>/tvm_min<T>, not max/min."""
    with tempfile.TemporaryDirectory() as tmp:
        gen = compile_for_dsp(
            _build_minmax_module(dtype),
            "c_static_lib -mcpu=c7x -use-cpp-api=1",
            output_dir=Path(tmp),
        )
        src = "\n".join(p.read_text() for p in sorted(gen.glob("lib*.c")))
    ctype = f"{dtype}_t"
    assert f"tvm_max<{ctype}>(" in src
    assert f"tvm_min<{ctype}>(" in src
    code = re.sub(r"//[^\n]*|/\*.*?\*/", "", src, flags=re.S)
    bare = re.findall(r"(?<![\w:])(?:max|min)\(", code)
    assert not bare, f"std::max/std::min still printed: {len(bare)} call(s)"


@pytest.mark.quick
@pytest.mark.parametrize("dtype", ["int8", "int32"])
def test_int_minmax_reduce(dsp_mode, dtype):
    """Row max and min match numpy, including rows the bug corrupts."""
    if dsp_mode not in ("c7x_host", "c7x_dload"):
        pytest.skip("requires c7x_host or c7x_dload")
    x = _rows(dtype)
    ref = np.concatenate([x.max(axis=1, keepdims=True), x.min(axis=1, keepdims=True)], axis=1)
    results = compile_and_run_dsp(
        mod=_build_minmax_module(dtype),
        input_data=x,
        target_string=get_target_string(dsp_mode, use_cpp_api=True),
        execution_mode=dsp_mode,
    )
    out = results[f"{dsp_mode}_result"].reshape(ref.shape)
    bad = np.argwhere(out != ref)
    assert bad.size == 0, (
        f"{len(bad)} of {ref.size} wrong; first (row, col, got, expected): "
        f"{[(int(r), int(c), int(out[r, c]), int(ref[r, c])) for r, c in bad[:5]]}"
    )
