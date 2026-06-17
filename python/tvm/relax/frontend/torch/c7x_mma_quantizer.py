# Licensed to the Apache Software Foundation (ASF) under one
# or more contributor license agreements.  See the NOTICE file
# distributed with this work for additional information
# regarding copyright ownership.  The ASF licenses this file
# to you under the Apache License, Version 2.0 (the
# "License"); you may not use this file except in compliance
# with the License.  You may obtain a copy of the License at
#
#   http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing,
# software distributed under the License is distributed on an
# "AS IS" BASIS, WITHOUT WARRANTIES OR CONDITIONS OF ANY
# KIND, either express or implied.  See the License for the
# specific language governing permissions and limitations
# under the License.
"""C7x MMA quantizer for the torchao PT2E quantization flow.

PT2E ("Post-Training Export 2E") is PyTorch's graph-level quantization pipeline:
a Quantizer subclass annotates an FX graph with quantization specs; prepare_pt2e
inserts calibration observers at those points; after calibration, convert_pt2e
replaces the observers with explicit Q/DQ (quantize/dequantize) nodes.  TVM then
imports the Q/DQ graph and its c_static backend fuses the Q/DQ sequences with
MMALIB kernel calls.

See docs/dsp/c7x_mma_quantizer.md for the full pipeline and usage examples.
"""

import logging
import warnings

import torch
from torch.fx import GraphModule, Node
from torchao.quantization.pt2e import MinMaxObserver, PerChannelMinMaxObserver
from torchao.quantization.pt2e.quantizer import (
    QuantizationAnnotation,
    QuantizationSpec,
    Quantizer,
)

logger = logging.getLogger(__name__)

_INT8_MIN, _INT8_MAX = -128, 127
_INT16_MIN, _INT16_MAX = -32768, 32767

# torch.export produces conv2d.default, not convolution.default.
# These ops have a weight tensor at args[1]; the quantizer assigns a per-channel
# spec to that arg (args[0] gets the activation spec, args[2] bias stays float32).
_WEIGHT_OPS = frozenset(
    [torch.ops.aten.conv2d.default, torch.ops.aten.linear.default]
)

# Both inputs are activations; no per-channel weight spec applies.
# add.Tensor produces the dq(x)+dq(skip)->q pattern consumed by FuseInt8ResidualAdd.
_ACT_ONLY_OPS = frozenset(
    [
        torch.ops.aten.mm.default,
        torch.ops.aten.addmm.default,
        torch.ops.aten.add.Tensor,
    ]
)


class C7xMMAQuantizer(Quantizer):
    """torchao PT2E quantizer targeting TI C7x MMALIB.

    Lifecycle (called in this order by the PT2E helpers):

    1. ``prepare_pt2e(model, quantizer)`` calls ``annotate()``, which writes a
       ``QuantizationAnnotation`` into each relevant node's ``node.meta`` dict.
       The annotation specifies *what kind* of quantization to apply to each
       input tensor and the output — but inserts nothing into the graph yet.
    2. ``prepare_pt2e`` then reads those annotations and inserts observer modules
       (``MinMaxObserver`` / ``PerChannelMinMaxObserver``) next to each annotated
       tensor.  Run calibration data through the prepared model to fill in the
       observed min/max ranges.
    3. ``convert_pt2e`` replaces the observers with explicit Q/DQ (quantize /
       dequantize) nodes, producing a graph of the form
       ``q → dq → op → q → dq → ...``.
    4. TVM imports the Q/DQ graph; its c_static backend fuses Q/DQ sequences with
       MMALIB kernel calls via passes: FuseMMALIBQDQConv2d, FuseMMALIBQDQDwConv2d,
       FuseMMALIBQDQFC, FuseInt8ResidualAdd.

    Args:
        dtype: "int8" or "int16".
        symmetric_activations: Use per-tensor symmetric for activations.
            Forced True for int16 (no asymmetric lowering path exists).
    """

    def __init__(self, dtype: str = "int8", symmetric_activations: bool = True):
        super().__init__()
        if dtype not in ("int8", "int16"):
            raise ValueError(f"dtype must be 'int8' or 'int16', got {dtype!r}")
        self.dtype = dtype
        if dtype == "int16" and not symmetric_activations:
            warnings.warn(
                "int16 activations require symmetric quantization; "
                "forcing symmetric_activations=True",
                stacklevel=2,
            )
            symmetric_activations = True
        self.symmetric_activations = symmetric_activations

    def _act_spec(self) -> QuantizationSpec:
        """Return the QuantizationSpec for activation tensors (per-tensor, MinMaxObserver).

        QuantizationSpec bundles the observer class, dtype, numeric range, and
        quantization scheme into one object.  prepare_pt2e reads these fields to
        decide which observer module to insert and how to configure it.

        MinMaxObserver must come from torchao.quantization.pt2e, not torch.ao —
        torchao's convert_pt2e only recognises torchao observer classes; torch.ao
        observers would be left as opaque call_module nodes after conversion.
        """
        if self.dtype == "int8":
            torch_dtype, quant_min, quant_max = torch.int8, _INT8_MIN, _INT8_MAX
        else:
            torch_dtype, quant_min, quant_max = torch.int16, _INT16_MIN, _INT16_MAX
        qscheme = (
            torch.per_tensor_symmetric
            if self.symmetric_activations
            else torch.per_tensor_affine
        )
        return QuantizationSpec(
            dtype=torch_dtype,
            observer_or_fake_quant_ctr=MinMaxObserver,
            quant_min=quant_min,
            quant_max=quant_max,
            qscheme=qscheme,
        )

    def _weight_spec(self) -> QuantizationSpec:
        """Return the QuantizationSpec for weight tensors (per-channel symmetric).

        MMALIB requires one scale per output channel and zero_point=0 for weights.
        ch_axis=0 selects the output-channel dimension, which is axis 0 for all
        standard weight layouts (conv2d: [out_ch, in_ch, kH, kW]; linear: [out, in]).
        Per-channel quantization produces tighter ranges than per-tensor, which
        reduces clipping error on weights whose per-channel distributions differ.
        """
        if self.dtype == "int8":
            torch_dtype, quant_min, quant_max = torch.int8, _INT8_MIN, _INT8_MAX
        else:
            torch_dtype, quant_min, quant_max = torch.int16, _INT16_MIN, _INT16_MAX
        return QuantizationSpec(
            dtype=torch_dtype,
            observer_or_fake_quant_ctr=PerChannelMinMaxObserver,
            quant_min=quant_min,
            quant_max=quant_max,
            qscheme=torch.per_channel_symmetric,
            ch_axis=0,
        )

    def annotate(self, model: GraphModule) -> GraphModule:
        """Annotate FX graph nodes with quantization specs.

        "Annotation" means writing a QuantizationAnnotation object into
        node.meta["quantization_annotation"] for each supported op.  The
        annotation holds an input_qspec_map (Node → QuantizationSpec for each
        input that should be quantized) and an output_qspec for the op's output.

        Nothing is inserted into the graph here.  prepare_pt2e reads the
        annotations after this method returns and inserts the actual observer
        modules.
        """
        act_spec = self._act_spec()
        weight_spec = self._weight_spec()

        for node in model.graph.nodes:
            if node.op != "call_function":
                continue
            if node.target not in _WEIGHT_OPS and node.target not in _ACT_ONLY_OPS:
                continue
            if node.meta.get("quantization_annotation") is not None:
                continue  # already annotated by a composed quantizer

            if node.target in _WEIGHT_OPS:
                # bias (args[2]) stays float32 — not annotated
                input_qspec_map: dict[Node, QuantizationSpec] = {
                    node.args[0]: act_spec,  # type: ignore[index]
                    node.args[1]: weight_spec,  # type: ignore[index]
                }
            elif node.target in (
                torch.ops.aten.mm.default,
                torch.ops.aten.add.Tensor,
            ):
                # Skip if either argument is a scalar (not an FX Node).
                # This happens for ops like `x + 0` in attention masking;
                # only Tensor + Tensor residual adds should be quantized.
                if not isinstance(node.args[0], Node) or not isinstance(node.args[1], Node):
                    continue
                input_qspec_map = {
                    node.args[0]: act_spec,  # type: ignore[index]
                    node.args[1]: act_spec,  # type: ignore[index]
                }
            else:
                # addmm args: (bias, input, weight) — bias (args[0]) stays float32
                input_qspec_map = {
                    node.args[1]: act_spec,  # type: ignore[index]
                    node.args[2]: act_spec,  # type: ignore[index]
                }

            node.meta["quantization_annotation"] = QuantizationAnnotation(
                input_qspec_map=input_qspec_map,  # type: ignore[arg-type]
                output_qspec=act_spec,
                _annotated=True,  # prevents double-annotation when composed with another quantizer
            )

        return model

    def validate(self, model: GraphModule) -> None:
        """Warn if any weight was annotated with a per-tensor spec.

        prepare_pt2e calls validate() after annotate().  If a composed quantizer
        replaced the per-channel weight spec set by annotate() with a per-tensor
        one, MMALIB would receive weights quantized incorrectly and silently
        produce wrong results.  This check makes that misconfiguration visible.
        """
        for node in model.graph.nodes:
            if node.op != "call_function" or node.target not in _WEIGHT_OPS:
                continue
            ann = node.meta.get("quantization_annotation")
            if ann is None:
                continue
            weight_node = node.args[1]
            spec = ann.input_qspec_map.get(weight_node)
            if spec is not None and hasattr(spec, "qscheme"):
                if spec.qscheme in (torch.per_tensor_symmetric, torch.per_tensor_affine):
                    logger.warning(
                        "Node %s: weight uses per-tensor quantization; "
                        "MMALIB requires per-channel symmetric (zero_point=0)",
                        node.name,
                    )
