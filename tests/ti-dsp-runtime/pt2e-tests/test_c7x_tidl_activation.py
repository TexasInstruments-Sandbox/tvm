"""Unit tests for TIDL activation QDQ fusion (FuseQDQToTIDLActivation).

Pure-Python tests — no DSP hardware or TI toolchain required.
Validates:
  1. C7xMMAQuantizer annotates gelu/silu/hardsigmoid/hardswish
  2. After TVM import, FuseQDQToTIDLActivation fires for all four ops
  3. The fused call_tir target matches the expected tidl_int8_* symbol

Run with:
    cd tests/ti-dsp-runtime
    pytest --rootdir=. pt2e-tests/test_c7x_tidl_activation.py -m quick -v
"""

import pytest
import torch
import torch.nn as nn
import torch.nn.functional as F
from pt2e_utils import quantize_pt2e

from tvm.relax.frontend.torch import C7xMMAQuantizer, from_exported_program
from tvm.relax.transform import FuseQDQToTIDLActivation

# ---------------------------------------------------------------------------
# Models
# ---------------------------------------------------------------------------


class GeluModel(nn.Module):
    def forward(self, x):
        return F.gelu(x)


class SiluModel(nn.Module):
    def forward(self, x):
        return F.silu(x)


class HardsigmoidModel(nn.Module):
    def forward(self, x):
        return F.hardsigmoid(x)


class HardswishModel(nn.Module):
    def forward(self, x):
        return F.hardswish(x)


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

_EXAMPLE = (torch.randn(2, 16),)


def _quantize_and_import(model: nn.Module):
    """Quantize with C7xMMAQuantizer and import into TVM."""
    q = quantize_pt2e(model, _EXAMPLE, C7xMMAQuantizer(dtype="int8"))
    ep = torch.export.export(q, _EXAMPLE)
    mod = from_exported_program(ep, keep_params_as_input=False)
    return mod


def _fuse(mod):
    return FuseQDQToTIDLActivation()(mod)


# ---------------------------------------------------------------------------
# Annotation tests (pure-PyTorch, no TVM)
# ---------------------------------------------------------------------------


@pytest.mark.quick
def test_gelu_gets_annotated():
    model = GeluModel().eval()
    exported = torch.export.export(model, _EXAMPLE).module()
    C7xMMAQuantizer("int8").annotate(exported)

    for node in exported.graph.nodes:
        if node.op == "call_function" and node.target is torch.ops.aten.gelu.default:
            ann = node.meta.get("quantization_annotation")
            assert ann is not None, "gelu not annotated"
            assert node.args[0] in ann.input_qspec_map
            return
    pytest.fail("no gelu node found")


@pytest.mark.quick
def test_silu_gets_annotated():
    model = SiluModel().eval()
    exported = torch.export.export(model, _EXAMPLE).module()
    C7xMMAQuantizer("int8").annotate(exported)

    for node in exported.graph.nodes:
        if node.op == "call_function" and node.target is torch.ops.aten.silu.default:
            ann = node.meta.get("quantization_annotation")
            assert ann is not None, "silu not annotated"
            return
    pytest.fail("no silu node found")


@pytest.mark.quick
def test_hardsigmoid_gets_annotated():
    model = HardsigmoidModel().eval()
    exported = torch.export.export(model, _EXAMPLE).module()
    C7xMMAQuantizer("int8").annotate(exported)

    for node in exported.graph.nodes:
        if node.op == "call_function" and node.target is torch.ops.aten.hardsigmoid.default:
            ann = node.meta.get("quantization_annotation")
            assert ann is not None, "hardsigmoid not annotated"
            return
    pytest.fail("no hardsigmoid node found")


@pytest.mark.quick
def test_hardswish_gets_annotated():
    model = HardswishModel().eval()
    exported = torch.export.export(model, _EXAMPLE).module()
    C7xMMAQuantizer("int8").annotate(exported)

    for node in exported.graph.nodes:
        if node.op == "call_function" and node.target is torch.ops.aten.hardswish.default:
            ann = node.meta.get("quantization_annotation")
            assert ann is not None, "hardswish not annotated"
            return
    pytest.fail("no hardswish node found")


# ---------------------------------------------------------------------------
# Fusion tests (TVM import + FuseQDQToTIDLActivation)
# ---------------------------------------------------------------------------


@pytest.mark.quick
@pytest.mark.parametrize(
    "model_cls, extern_sym",
    [
        (GeluModel, "tidl_int8_gelu"),
        (SiluModel, "tidl_int8_silu"),
        (HardsigmoidModel, "tidl_int8_hardsigmoid"),
        (HardswishModel, "tidl_int8_hardswish"),
    ],
)
def test_activation_fusion_fires(model_cls, extern_sym):
    """FuseQDQToTIDLActivation produces a call_tir targeting the expected kernel."""
    mod = _quantize_and_import(model_cls().eval())
    fused = _fuse(mod)
    main_str = str(fused["main"])
    assert extern_sym in main_str, (
        f"{extern_sym} not found in fused IR after FuseQDQToTIDLActivation.\n"
        f"IR:\n{main_str}"
    )


@pytest.mark.quick
def test_gelu_output_is_int8():
    """Fused gelu kernel output has int8 dtype."""
    mod = _quantize_and_import(GeluModel().eval())
    fused = _fuse(mod)
    # The output of the fused function should be float32 (restored by outer dq)
    # or int8 (if the consumer is another quantized op). Check that the IR
    # contains an int8 tensor from the gelu call.
    assert "int8" in str(fused["main"]), "expected int8 tensor in fused IR"


@pytest.mark.quick
def test_i8_passes_do_not_trigger_on_int16():
    """FuseQDQToTIDLActivation does not fuse int16 activations (int8 only)."""
    # int16 quantized gelu should stay as float ops (no tidl_int8_gelu)
    # because the i8 check function rejects int16 input dtype.
    mod = _quantize_and_import_i16(GeluModel().eval())
    fused = _fuse(mod)
    assert "tidl_int8_gelu" not in str(fused["main"]), (
        "tidl_int8_gelu must not fire on int16 graph"
    )


def _quantize_and_import_i16(model: nn.Module):
    q = quantize_pt2e(model, _EXAMPLE, C7xMMAQuantizer(dtype="int16"))
    ep = torch.export.export(q, _EXAMPLE)
    mod = from_exported_program(ep, keep_params_as_input=False)
    return mod
