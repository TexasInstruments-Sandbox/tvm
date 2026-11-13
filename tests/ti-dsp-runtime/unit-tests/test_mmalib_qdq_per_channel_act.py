"""Per-channel activation scales through the import and the MMALIB QDQ fusions.

A depthwise conv keeps its channels apart, so a per-channel scale on its input folds
into its per-channel requantize, and so does a per-channel scale on any conv's
output. A regular conv's input mixes channels and must stay per tensor. Relax
IR-level tests, plus one c7x_host run of a stem conv -> hardswish -> depthwise model
quantized per channel, against PyTorch.

Usage:
    pytest test_mmalib_qdq_per_channel_act.py -v --dsp-mode=c7x_host
"""

import re
import sys
from pathlib import Path

import numpy as np
import pytest
import torch
import torch.nn as nn

import tvm
from tvm import relax
from tvm.relax import TensorType as TensorStructInfo
from tvm.relax.transform import FuseMMALIBQDQConv2d, FuseMMALIBQDQDwConv2d
from tvm.relax.transform.ti_const_utils import ti_tensor_const
from tvm.relax.transform.ti_mmalib_legalize import _scale_shift_or_none

sys.path.insert(0, str(Path(__file__).parent.parent / "dsp-cpp"))

pytestmark = pytest.mark.quick

C = 8
_RNG = np.random.default_rng(7)
_W_SCALE = _RNG.uniform(0.002, 0.02, C).astype(np.float32)


def _const(v, dtype):
    return ti_tensor_const(np.asarray(v, dtype=dtype))


def _conv_model(groups, d_scale, o_scale, c_in=C):
    """dequant(x, d_scale) -> conv2d(dequant(w)) -> quantize(o_scale); scales may be vectors."""
    per_out = 1 if groups > 1 else c_in
    kernel = _RNG.integers(-100, 100, size=(C, per_out, 3, 3), dtype=np.int8)
    d_axis = 1 if np.ndim(d_scale) else -1
    o_axis = 1 if np.ndim(o_scale) else -1
    bb = relax.BlockBuilder()
    x = relax.Var("x", TensorStructInfo((1, c_in, 8, 8), "int8"))
    with bb.function("main", [x], attrs={"num_input": 1}):
        with bb.dataflow():
            w = bb.emit(
                relax.op.dequantize(
                    ti_tensor_const(kernel),
                    _const(_W_SCALE, "float32"),
                    _const(np.zeros(C), "int8"),
                    axis=0,
                )
            )
            d = bb.emit(
                relax.op.dequantize(
                    x,
                    _const(d_scale, "float32"),
                    _const(np.zeros_like(d_scale), "int8"),
                    axis=d_axis,
                )
            )
            conv = bb.emit(relax.op.nn.conv2d(d, w, padding=(1, 1), groups=groups))
            q = bb.emit(
                relax.op.quantize(
                    conv,
                    _const(o_scale, "float32"),
                    _const(np.zeros_like(o_scale), "int8"),
                    axis=o_axis,
                    out_dtype="int8",
                )
            )
            out = bb.emit_output(q)
        bb.emit_func_output(out)
    return bb.finalize()


def _mmalib_scale_shift(mod, kernel):
    """The scale and shift constants of the fused MMALIB call, or None if it didn't fuse."""
    found = []

    def visit(e):
        if isinstance(e, relax.Call) and getattr(e.op, "name", "") == "relax.call_tir":
            if kernel in e.args[0].name_hint:
                found.append(e.args[1].fields)

    relax.analysis.post_order_visit(mod["main"], visit)
    if not found:
        return None
    # (data, kernel, bias, scale, shift)
    fields = found[0]
    return fields[3].value.numpy(), fields[4].value.numpy()


@pytest.mark.parametrize("per_channel", ["output"])
def test_conv_folds_a_per_channel_output_scale(per_channel):
    o_scale = _RNG.uniform(0.02, 0.2, C).astype(np.float32)
    got = _mmalib_scale_shift(
        FuseMMALIBQDQConv2d()(_conv_model(1, np.float32(0.05), o_scale)), "mmalib_conv2d"
    )
    assert got is not None
    want = _scale_shift_or_none(
        np.float32(0.05) * _W_SCALE.astype(np.float64) / o_scale.astype(np.float64)
    )
    np.testing.assert_array_equal(got[0], want[0])
    np.testing.assert_array_equal(got[1], want[1])


def test_conv_declines_a_per_channel_input_scale():
    """A regular conv sums over its input channels: a per-channel input scale can't fold."""
    d_scale = _RNG.uniform(0.02, 0.2, C).astype(np.float32)
    assert (
        _mmalib_scale_shift(
            FuseMMALIBQDQConv2d()(_conv_model(1, d_scale, np.float32(0.1))), "mmalib_conv2d"
        )
        is None
    )


@pytest.mark.parametrize("which", ["input", "output", "both"])
def test_depthwise_folds_per_channel_scales(which):
    d = _RNG.uniform(0.02, 0.2, C).astype(np.float32) if which != "output" else np.float32(0.05)
    o = _RNG.uniform(0.02, 0.2, C).astype(np.float32) if which != "input" else np.float32(0.1)
    got = _mmalib_scale_shift(FuseMMALIBQDQDwConv2d()(_conv_model(C, d, o)), "mmalib_dwconv2d")
    assert got is not None
    want = _scale_shift_or_none(
        np.asarray(d, np.float64) * _W_SCALE.astype(np.float64) / np.asarray(o, np.float64)
    )
    np.testing.assert_array_equal(got[0], want[0])
    np.testing.assert_array_equal(got[1], want[1])


def _per_channel_activation(act):
    """dq(x, per-channel) -> act -> q(per-channel), act "silu" or "gelu"."""
    s = _const(_RNG.uniform(0.02, 0.2, C), "float32")
    z = _const(np.zeros(C), "int8")
    bb = relax.BlockBuilder()
    x = relax.Var("x", TensorStructInfo((1, C, 4, 4), "int8"))
    with bb.function("main", [x], attrs={"num_input": 1}):
        with bb.dataflow():
            d = bb.emit(relax.op.dequantize(x, s, z, axis=1))
            if act == "silu":
                a = bb.emit(relax.op.multiply(d, bb.emit(relax.op.sigmoid(d))))
            else:
                a = bb.emit(relax.op.nn.gelu(d))
            out = bb.emit_output(bb.emit(relax.op.quantize(a, s, z, axis=1, out_dtype="int8")))
        bb.emit_func_output(out)
    return bb.finalize(), s.value.numpy()


def test_silu_quantized_per_channel_lowers_to_its_kernel():
    mod, scales = _per_channel_activation("silu")
    out = tvm.relax.transform.FuseQDQToC7xActivation()(mod)
    text = out.script()
    assert "c7x_int8_silu_per_channel" in text and "R.sigmoid" not in text
    consts = []
    relax.analysis.post_order_visit(
        out["main"],
        lambda e: (
            consts.append(e.value.numpy())
            if isinstance(e, tvm.ir.GenericConst) and e.value.numpy().shape == (C,)
            else None
        ),
    )
    assert len(consts) == 2 and all(np.allclose(c, scales) for c in consts)


def test_activation_without_a_per_channel_kernel_stays_in_float():
    """gelu has no per-channel kernel; its per-channel scales leave it in float."""
    mod, _ = _per_channel_activation("gelu")
    text = tvm.relax.transform.FuseQDQToC7xActivation()(mod).script()
    assert "c7x_int8_gelu" not in text and "R.nn.gelu" in text


class _StemBlock(nn.Module):
    """MobileNetV3-Small's stem: conv -> hardswish -> depthwise -> project."""

    def __init__(self):
        super().__init__()
        self.stem = nn.Conv2d(3, 16, 3, padding=1)
        self.dw = nn.Conv2d(16, 16, 3, padding=1, groups=16)
        self.project = nn.Conv2d(16, 32, 1)

    def forward(self, x):
        return self.project(self.dw(nn.functional.hardswish(self.stem(x))))


def _quantize_stem_per_channel():
    """PT2E with the stem conv output and hardswish output quantized per channel."""
    from torchao.quantization.pt2e import PerChannelMinMaxObserver
    from torchao.quantization.pt2e.quantize_pt2e import convert_pt2e, prepare_pt2e
    from torchao.quantization.pt2e.quantizer import QuantizationSpec

    from tvm.relax.frontend.torch import C7xMMAQuantizer

    pc = QuantizationSpec(
        dtype=torch.int8,
        quant_min=-128,
        quant_max=127,
        qscheme=torch.per_channel_symmetric,
        ch_axis=1,
        observer_or_fake_quant_ctr=PerChannelMinMaxObserver.with_args(ch_axis=1),
    )

    class PerChannelStem(C7xMMAQuantizer):
        def annotate(self, model):
            super().annotate(model)
            for n in model.graph.nodes:
                ann = n.meta.get("quantization_annotation")
                if ann is None:
                    continue
                if n.target == torch.ops.aten.hardswish.default:
                    ann.input_qspec_map = {k: pc for k in ann.input_qspec_map}
                    ann.output_qspec = pc
                elif n.target == torch.ops.aten.conv2d.default:
                    if any(u.target == torch.ops.aten.hardswish.default for u in n.users):
                        ann.output_qspec = pc
                    if n.args[0].target == torch.ops.aten.hardswish.default:
                        ann.input_qspec_map[n.args[0]] = pc
            return model

    torch.manual_seed(0)
    x = torch.randn(1, 3, 16, 16)
    prepared = prepare_pt2e(
        torch.export.export(_StemBlock().eval(), (x,)).module(), PerChannelStem()
    )
    for _ in range(4):
        prepared(torch.randn(1, 3, 16, 16) * 2)
    return convert_pt2e(prepared), x


def _import(q, x):
    from tvm.relax.frontend.torch import from_exported_program

    mod = from_exported_program(torch.export.export(q, (x,)), keep_params_as_input=True)
    mod, params = relax.frontend.detach_params(mod)
    return relax.transform.BindParams("main", dict(zip(mod["main"].params[1:], params["main"])))(
        mod
    )


def test_import_keeps_per_channel_activation_quantize():
    q, x = _quantize_stem_per_channel()
    text = _import(q, x)["main"].script(show_meta=False)
    assert len(re.findall(r"R\.quantize\([^\n]*axis=1", text)) == 2


def test_per_channel_stem_on_dsp_matches_pytorch(dsp_mode):
    if dsp_mode not in ("c7x_host", "c7x_dload"):
        pytest.skip("requires c7x_host or c7x_dload")
    from dsp_utils import compile_and_run_dsp, get_target_string

    q, x = _quantize_stem_per_channel()
    with torch.no_grad():
        ref = q(x).numpy()
    results = compile_and_run_dsp(
        mod=_import(q, x),
        input_data=[x.numpy()],
        target_string=get_target_string(dsp_mode, use_cpp_api=True) + " -mmalib=1",
        execution_mode=dsp_mode,
    )
    dsp = results[f"{dsp_mode}_result"].reshape(ref.shape)
    # The output's quantization step: the last quantize's scale.
    (out_q,) = [
        n
        for n in q.graph.nodes
        if "quantize_per_tensor" in str(n.target) and "dequantize" not in str(n.target)
    ][-1:]
    step = float(out_q.args[1])
    # MMALIB's 8-bit scale/shift requantize is within a code or two of the
    # reference's float requantize, as on the per-tensor path.
    assert np.abs(dsp - ref).max() <= 3 * step
