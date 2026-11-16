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
imports the Q/DQ graph and its c_static_lib backend fuses the Q/DQ sequences with
MMALIB kernel calls.

See docs-c7x/contributor-guide/backend/extending-quantizer.md for the full
pipeline and usage examples.
"""

import dataclasses
import logging
import operator
import warnings

import torch
from torch.fx import GraphModule, Node
from torchao.quantization.pt2e import HistogramObserver, MinMaxObserver, PerChannelMinMaxObserver
from torchao.quantization.pt2e.quantizer import (
    FixedQParamsQuantizationSpec,
    QuantizationAnnotation,
    QuantizationSpec,
    QuantizationSpecBase,
    Quantizer,
    SharedQuantizationSpec,
)

logger = logging.getLogger(__name__)

_INT8_MIN, _INT8_MAX = -128, 127
_INT16_MIN, _INT16_MAX = -32768, 32767

# torch.export produces conv2d.default, not convolution.default.
# These ops have a weight tensor at args[1]; the quantizer assigns a per-channel
# spec to that arg (args[0] gets the activation spec, args[2] bias stays float32).
_WEIGHT_OPS = frozenset([torch.ops.aten.conv2d.default, torch.ops.aten.linear.default])

# Both inputs are activations; no per-channel weight spec applies.
# add.Tensor / add_.Tensor produce the dq(x)+dq(skip)->q pattern consumed by
# FuseInt8ResidualAdd. add_ is the in-place variant (used by ResNet shortcuts).
_ACT_ONLY_OPS = frozenset(
    [
        torch.ops.aten.mm.default,
        torch.ops.aten.addmm.default,
        torch.ops.aten.add.Tensor,
        torch.ops.aten.add_.Tensor,
    ]
)

# Single-input activation ops that require independent output quantization.
# These map to c7x_int8_* kernels via FuseQDQToC7xActivation.
# NOTE: group_norm is excluded — no TIDL kernel available; needs a custom
# C7x implementation.
_TIDL_ACT_OPS = frozenset(
    [
        torch.ops.aten.gelu.default,
        torch.ops.aten.silu.default,
        torch.ops.aten.hardsigmoid.default,
        torch.ops.aten.hardswish.default,
    ]
)

# Normalization ops: input is annotated as int8; weight/bias stay float32.
# The normalization itself runs in float32 internally. Handled by FuseQDQToC7xLayerNorm.
# NOTE: group_norm excluded — no TIDL kernel; needs a custom C7x implementation.
_NORM_OPS = frozenset(
    [
        torch.ops.aten.layer_norm.default,
    ]
)

# Average pooling ops: not scale-transparent (averaging changes the range),
# so need full requantization. Handled by FuseQDQToC7xAvgPool.
_AVG_POOL_OPS = frozenset(
    [
        torch.ops.aten.adaptive_avg_pool2d.default,
        torch.ops.aten.avg_pool2d.default,
    ]
)

# Single-input ops that are quantization-transparent: the output range equals
# the input range, so EliminateQDQTransparent can remove the QDQ wrappers.
# Output uses SharedQuantizationSpec pointing to the input to guarantee
# matching scales after calibration.
# NOTE: aten.cat.default is intentionally excluded — concatenating tensors with
# different per-input scales requires requantization, which EliminateQDQTransparent
# does not currently implement.
# relu/relu_ clip negatives but the post-relu range is still covered by the
# same observer that sees the conv2d output (calibration learns [0, max]).
# Including them here lets SharedQuantizationSpec chain correctly through
# conv2d → relu_ → max_pool2d without a KeyError in prepare_pt2e.
_TRANSPARENT_OPS = frozenset(
    [
        torch.ops.aten.relu.default,
        torch.ops.aten.relu_.default,
        torch.ops.aten.max_pool2d.default,
        torch.ops.aten.view.default,
        torch.ops.aten.permute.default,
        torch.ops.aten.flatten.using_ints,
    ]
)


_STRUCTURAL_OPS = frozenset(
    [
        operator.getitem,
        torch.ops.aten.split_with_sizes.default,
        torch.ops.aten.split.Tensor,
    ]
)

# Ops that don't need their own annotation (not quantizable ops in the first
# place) but must not block _feeds_topk's search -- e.g. a detection head's
# `scores.max(dim=-1)[0].topk(...)` puts max.dim directly between a
# transparent/structural op and the topk that makes it selection-sensitive.
_REACHABILITY_PASSTHROUGH_OPS = frozenset(
    [
        torch.ops.aten.max.dim,
        torch.ops.aten.min.dim,
    ]
)


# hardtanh clamps to [min_val, max_val]; ReLU6 is hardtanh(0, 6). Not annotated
# itself: its output is observed by its consumer's input spec.
_HARDTANH_OPS = frozenset([torch.ops.aten.hardtanh.default, torch.ops.aten.hardtanh_.default])


def _hardtanh_bound(node: Node) -> float | None:
    """max(|min_val|, |max_val|) if node's only user is a hardtanh, else None.

    Quantizing node's output to [-bound, bound] loses nothing the clamp keeps,
    so a conv or linear feeding only a clamp needs no wider range.
    """
    if len(node.users) != 1:
        return None
    user = next(iter(node.users))
    if user.op != "call_function" or user.target not in _HARDTANH_OPS:
        return None
    lo = user.args[1] if len(user.args) > 1 else user.kwargs.get("min_val", -1.0)
    hi = user.args[2] if len(user.args) > 2 else user.kwargs.get("max_val", 1.0)
    return max(abs(float(lo)), abs(float(hi)))  # type: ignore[arg-type]


def _feeds_topk(node: Node) -> bool:
    """True if `node`'s output reaches an `aten.topk.default` call through a
    chain of transparent/structural/passthrough ops only (permute, view,
    flatten, getitem, split, max.dim/min.dim) -- i.e. it is part of a
    detection-style postprocess selection region, not a genuine compute path.

    This only follows a bounded set of edges, not arbitrary graph paths, so
    it won't over-reach into an unrelated conv/linear branch that happens to
    share an ancestor with a topk elsewhere in the model.
    """
    seen = set()
    frontier = list(node.users)
    while frontier:
        user = frontier.pop()
        if user in seen:
            continue
        seen.add(user)
        target = getattr(user, "target", None)
        if target is torch.ops.aten.topk.default:
            return True
        if (
            target in _TRANSPARENT_OPS
            or target in _STRUCTURAL_OPS
            or target in _REACHABILITY_PASSTHROUGH_OPS
        ):
            frontier.extend(user.users)
    return False


def _is_float_tensor(node: Node) -> bool:
    """True if node's traced FakeTensor value is a floating-point tensor.

    view/permute/flatten/add/mm are generic ops that can also match on
    non-quantizable integer buffers -- e.g. torchvision MaxViT's
    relative_position_index, reshaped via .view(-1) before being used as a
    gather index. Annotating an int64 tensor with a float QuantizationSpec
    makes convert_pt2e emit quantize_per_tensor on it, which crashes.
    """
    val = node.meta.get("val")
    return isinstance(val, torch.Tensor) and val.is_floating_point()


# Observers for activation tensors, chosen by C7xMMAQuantizer(activation_observer=).
_ACTIVATION_OBSERVERS = {"minmax": MinMaxObserver, "histogram": HistogramObserver}

# Smooth activations with no upper bound (and hardsigmoid, which shares hardswish's
# kernel input). The tensors around them get a percentile range; see
# C7xMMAQuantizer(smooth_activation_percentile=).
_SMOOTH_ACTIVATIONS = frozenset(
    [
        torch.ops.aten.silu.default,
        torch.ops.aten.silu_.default,
        torch.ops.aten.hardswish.default,
        torch.ops.aten.hardswish_.default,
        torch.ops.aten.hardsigmoid.default,
        torch.ops.aten.hardsigmoid_.default,
    ]
)
# Squeeze-excite gates read a conv output through a plain sigmoid.
_SMOOTH_ACTIVATION_INPUTS = _SMOOTH_ACTIVATIONS | {torch.ops.aten.sigmoid.default}


def _around_smooth_activation(node: Node) -> bool:
    """True if node is a smooth activation's output, or feeds only smooth activations."""
    if node.op != "call_function":
        return False
    if node.target in _SMOOTH_ACTIVATIONS:
        return True
    return bool(node.users) and all(
        user.op == "call_function" and user.target in _SMOOTH_ACTIVATION_INPUTS
        for user in node.users
    )


# Squeeze-excite gates: the excitation after its activation.
_GATES = frozenset(
    [
        torch.ops.aten.sigmoid.default,
        torch.ops.aten.hardsigmoid.default,
        torch.ops.aten.hardsigmoid_.default,
    ]
)
_RELUS = frozenset([torch.ops.aten.relu.default, torch.ops.aten.relu_.default])


def _is_depthwise_conv(node: Node) -> bool:
    if node.op != "call_function" or node.target != torch.ops.aten.conv2d.default:
        return False
    args = tuple(node.args)
    groups = args[6] if len(args) > 6 else node.kwargs.get("groups", 1)
    src = args[0] if args else None
    val = src.meta.get("val") if isinstance(src, Node) else None
    if not isinstance(groups, int) or not isinstance(val, torch.Tensor) or val.dim() != 4:
        return False
    return groups > 1 and groups == int(val.shape[1])


def _in_depthwise_block(node: Node) -> bool:
    """True for the tensors of a depthwise block that its outliers set the range of.

    A squeeze-excite output: a SiLU or hardswish output times its gate. And a conv
    output read only by ReLUs next to a depthwise conv: the conv is depthwise, or
    the ReLU feeds one. (BN is already folded into the conv when annotate runs.)
    """
    if node.op != "call_function":
        return False
    if node.target == torch.ops.aten.mul.Tensor:
        args = [a for a in node.args if isinstance(a, Node) and a.op == "call_function"]
        return (
            len(args) == 2
            and any(a.target in _SMOOTH_ACTIVATIONS for a in args)
            and any(a.target in _GATES for a in args)
        )
    if node.target != torch.ops.aten.conv2d.default or not node.users:
        return False
    if not all(u.op == "call_function" and u.target in _RELUS for u in node.users):
        return False
    return _is_depthwise_conv(node) or any(
        _is_depthwise_conv(v) for u in node.users for v in u.users
    )


# Activations between a conv and a depthwise conv that a per-channel scale can pass
# through: c7x_int8_{silu,hardswish}_per_channel, and a ReLU fused into the conv.
_PER_CHANNEL_ACTIVATIONS = frozenset(
    [
        torch.ops.aten.silu.default,
        torch.ops.aten.silu_.default,
        torch.ops.aten.hardswish.default,
        torch.ops.aten.hardswish_.default,
    ]
)


def _depthwise_input_chains(model: GraphModule) -> list[tuple[Node, Node]]:
    """(conv, activation) pairs whose tensors depthwise_input_headroom makes per channel.

    A depthwise conv reads the activation, which reads the conv. A depthwise conv
    keeps its channels apart, so a per-channel input scale folds into its
    per-channel requantize; the conv's own per-channel requantize produces the
    scales; and the activation runs per channel. Other readers of either tensor
    (a residual add reading a block's input) keep a per-tensor spec.
    """
    pairs = []
    for dw in model.graph.nodes:
        if not _is_depthwise_conv(dw):
            continue
        act = dw.args[0]
        if not (
            isinstance(act, Node)
            and act.op == "call_function"
            and act.target in (_PER_CHANNEL_ACTIVATIONS | _RELUS)
        ):
            continue
        conv = act.args[0] if act.args else None
        pair = (conv, act)
        if (
            isinstance(conv, Node)
            and conv.op == "call_function"
            and conv.target == torch.ops.aten.conv2d.default
            and pair not in pairs
        ):
            pairs.append(pair)
    return pairs


def _reads_per_channel(reader: Node, tensor: Node) -> bool:
    """True if reader can take tensor quantized per channel along its channel axis: a
    depthwise conv's data input, or the activation (a per-channel kernel, or a ReLU
    fused into the conv) of a _depthwise_input_chains pair."""
    if reader.op != "call_function" or not reader.args or reader.args[0] is not tensor:
        return False
    return _is_depthwise_conv(reader) or reader.target in (_PER_CHANNEL_ACTIVATIONS | _RELUS)


_ADDS = frozenset([torch.ops.aten.add.Tensor, torch.ops.aten.add_.Tensor])

# Ops that pass a tensor's values on unchanged: a tensor read through them keeps the
# reader that matters (a layer_norm, a residual add) in view.
_PASS_THROUGH_OPS = frozenset(
    [
        torch.ops.aten.dropout.default,
        torch.ops.aten.clone.default,
        torch.ops.aten.contiguous.default,
        torch.ops.aten.alias.default,
        torch.ops.aten.view.default,
        torch.ops.aten.reshape.default,
        torch.ops.aten.permute.default,
        torch.ops.aten.transpose.int,
        torch.ops.aten.flatten.using_ints,
        torch.ops.aten.unsqueeze.default,
        torch.ops.aten.squeeze.dim,
    ]
)


def _through_pass_through(node: Node) -> Node:
    """The tensor node's values come from, past any pass-through ops."""
    while (
        node.op == "call_function"
        and node.target in _PASS_THROUGH_OPS
        and isinstance(node.args[0], Node)
    ):
        node = node.args[0]
    return node


def _is_layernorm_residual_add(node: Node) -> bool:
    """True for add(skip, branch) where the branch normalizes skip with a layer_norm.

    That is the residual add of a pre-norm transformer block (attention or MLP) or of
    a ConvNeXt block: the branch is computed from skip, and a layer_norm lies on the
    way. An add inside the branch, such as Swin's attention bias, doesn't qualify:
    its other operand is not where the branch starts.
    """
    if node.op != "call_function" or node.target not in _ADDS:
        return False
    operands = [a for a in node.args[:2] if isinstance(a, Node)]
    if len(operands) != 2:
        return False
    for skip, branch in (operands, operands[::-1]):
        start = _through_pass_through(skip)
        # What the branch is computed from, back to skip.
        cone: set[Node] = set()
        reached_skip = False
        frontier = [branch]
        while frontier:
            m = frontier.pop()
            if m is start:
                reached_skip = True
            elif m not in cone:
                cone.add(m)
                frontier.extend(m.all_input_nodes)
        if not reached_skip:
            continue
        # The part of it computed from skip: a layer_norm there normalizes skip.
        seen: set[Node] = set()
        frontier = list(start.users)
        while frontier:
            m = frontier.pop()
            if m in seen or m not in cone:
                continue
            seen.add(m)
            if m.op == "call_function" and m.target in _NORM_OPS:
                return True
            frontier.extend(m.users)
    return False


def _readers(node: Node):
    """The nodes reading node's values, past pass-through ops, with the node each
    reads them from (node itself or a pass-through op after it)."""
    frontier = [node]
    while frontier:
        source = frontier.pop()
        for user in source.users:
            if user.op == "call_function" and user.target in _PASS_THROUGH_OPS:
                frontier.append(user)
            yield user, source


class _PercentileObserver(HistogramObserver):
    """Range between the p-th and (100 - p)-th percentiles of the values seen.

    Accumulates the same histogram as HistogramObserver, but instead of searching
    for the range with the least squared error it clips `percentile` percent of
    the values from each tail. The approach follows AdaptiveRangeShrinkObserver in
    TI's edgeai-torchmodelopt (xmodelopt/quantization/v3/observer_utils.py,
    BSD-3-Clause); this implementation is written on torchao's base class.
    """

    def __init__(self, *args, percentile: float = 0.1, **kwargs):
        super().__init__(*args, **kwargs)
        self.percentile = percentile

    def _non_linear_param_search(self) -> tuple[torch.Tensor, torch.Tensor]:
        hist = self.histogram.float()
        total = hist.sum()
        if total == 0 or not bool(self.max_val > self.min_val):
            return self.min_val, self.max_val
        cdf = torch.cumsum(hist, dim=0) / total
        tail = self.percentile / 100.0
        start = torch.searchsorted(cdf, torch.tensor([tail]), side="left")[0]
        end = torch.searchsorted(cdf, torch.tensor([1.0 - tail]), side="right")[0]
        width = (self.max_val - self.min_val) / self.bins
        new_min = self.min_val + width * start
        new_max = torch.minimum(self.min_val + width * (end + 1), self.max_val)
        return new_min, new_max


class _CappedPerChannelObserver(PerChannelMinMaxObserver):
    """Per-channel symmetric scales, each channel's peak times headroom, capped at
    the scale per_tensor would give the whole tensor.

    So no channel gets a coarser grid than one scale for the tensor would give it,
    and small channels get a finer one. The headroom covers peaks the calibration
    data didn't show, which a small calibration set understates channel by channel.
    """

    def __init__(self, *args, headroom: float = 2.0, per_tensor=MinMaxObserver, **kwargs):
        kwargs.update(ch_axis=1, qscheme=torch.per_channel_symmetric)
        super().__init__(*args, **kwargs)
        self.headroom = headroom
        self.per_tensor = per_tensor(
            dtype=self.dtype,
            quant_min=self.quant_min,
            quant_max=self.quant_max,
            qscheme=torch.per_tensor_symmetric,
        )

    def forward(self, x_orig):
        self.per_tensor(x_orig)
        return super().forward(x_orig)

    @torch.jit.export
    def calculate_qparams(self):
        scale, zero_point = super().calculate_qparams()
        tensor_scale, _ = self.per_tensor.calculate_qparams()
        return torch.minimum(scale * self.headroom, tensor_scale.reshape(-1)[:1]), zero_point


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
    4. TVM imports the Q/DQ graph; its c_static_lib backend fuses Q/DQ sequences with
       MMALIB kernel calls via passes: FuseMMALIBQDQConv2d, FuseMMALIBQDQDwConv2d,
       FuseMMALIBQDQFC, FuseInt8ResidualAdd.

    Args:
        dtype: "int8" or "int16".
        symmetric_activations: Use per-tensor symmetric for activations.
            Forced True for int16 (no asymmetric lowering path exists).
        activation_observer: How calibration sets each activation's range.
            "minmax" uses the extremes seen, so a single outlier widens the
            range for every value, and the range keeps growing with more
            calibration data. "histogram" picks the range with the least
            total squared error over the values seen, so it saturates an
            outlier once the precision it costs the other values outweighs
            clipping it; a tensor with few values keeps behaving like min/max.
            It calibrates more slowly. Weights always use min/max.
        smooth_activation_percentile: Percent of values clipped from each tail
            of the tensors around SiLU, hardswish and hardsigmoid (their inputs
            and outputs, and a sigmoid's input), whatever activation_observer
            is. None uses activation_observer for these tensors too. int8 only:
            int16 has the resolution to spare. A few channels of these tensors
            carry far larger values than the rest, and a range that covers them
            leaves the other channels few codes; ReLU networks have no such
            tensors and are unaffected.
        depthwise_block_percentile: The same, for two kinds of tensor in
            depthwise (inverted-residual) blocks, whose per-channel outliers set
            their range in the same way: a conv output read only by ReLUs next
            to a depthwise conv (the conv is depthwise, or the ReLU feeds one),
            and a squeeze-excite output (a SiLU or hardswish output times its
            gate). None leaves them to activation_observer. int8 only. Networks
            without depthwise convs or such squeeze-excite blocks are
            unaffected; clipping the ReLU tensors of a ResNet instead costs it.
        depthwise_input_headroom: Quantize a depthwise conv's input per channel,
            with the conv output feeding it through a SiLU, hardswish or ReLU: one
            scale per channel, its calibrated peak times this headroom, never coarser
            than the per-tensor scale the tensor would otherwise get. A depthwise
            conv keeps its channels apart, so the scales fold into its per-channel
            requantize, and the producing conv's requantize makes them. Its
            channels' ranges can differ by over 100x (MobileNetV3-Small's stem), more
            than one scale can serve. Any other reader of these tensors (a residual
            add) keeps reading them per tensor. None keeps them per tensor. int8 and
            symmetric activations only.
        float_layernorm_residuals: Leave the residual stream of LayerNorm blocks in
            float: the residual add of a block whose branch normalizes its input
            with a layer_norm (a transformer's attention and MLP blocks, ConvNeXt's
            blocks), and any add that a layer_norm or such a residual add reads (a
            ViT's position embedding add). A layer_norm or residual add reads it as
            float; any other reader, such as a conv, still reads it in int8. A
            layer_norm normalizes each token by its own statistics, so it turns a
            token's rounding into the token's whole content wherever the token is
            small next to the tensor's range: ViT-B/32's class token got about 4
            int8 codes after the position embedding add. False quantizes these adds
            like any other. int8 only. Networks without layer_norm are unaffected.

    A conv or linear whose only consumer is a hardtanh, such as ReLU6, gets the
    clamp's range for its output instead of a calibrated one (symmetric
    activations only). Calibration would record the extremes before the clamp,
    which in MobileNetV2 are 1.4 to 11 times the clamp's bound (about 3 times
    typically), so the values the clamp keeps got that much less resolution.
    """

    def __init__(
        self,
        dtype: str = "int8",
        symmetric_activations: bool = True,
        activation_observer: str = "minmax",
        smooth_activation_percentile: float | None = 0.1,
        depthwise_block_percentile: float | None = 0.01,
        depthwise_input_headroom: float | None = 2.0,
        float_layernorm_residuals: bool = True,
    ):
        super().__init__()
        if dtype not in ("int8", "int16"):
            raise ValueError(f"dtype must be 'int8' or 'int16', got {dtype!r}")
        if activation_observer not in _ACTIVATION_OBSERVERS:
            raise ValueError(
                f"activation_observer must be one of {sorted(_ACTIVATION_OBSERVERS)}, "
                f"got {activation_observer!r}"
            )
        for name, value in (
            ("smooth_activation_percentile", smooth_activation_percentile),
            ("depthwise_block_percentile", depthwise_block_percentile),
        ):
            if value is not None and not 0 < value < 50:
                raise ValueError(f"{name} must be in (0, 50) or None, got {value!r}")
        if depthwise_input_headroom is not None and not depthwise_input_headroom >= 1:
            raise ValueError(
                f"depthwise_input_headroom must be >= 1 or None, got {depthwise_input_headroom!r}"
            )
        self.depthwise_input_headroom = depthwise_input_headroom
        self.float_layernorm_residuals = float_layernorm_residuals
        self.dtype = dtype
        self.activation_observer = activation_observer
        self.smooth_activation_percentile = smooth_activation_percentile
        self.depthwise_block_percentile = depthwise_block_percentile
        if dtype == "int16" and not symmetric_activations:
            warnings.warn(
                "int16 activations require symmetric quantization; "
                "forcing symmetric_activations=True",
                stacklevel=2,
            )
            symmetric_activations = True
        self.symmetric_activations = symmetric_activations

    def _act_spec(self) -> QuantizationSpec:
        """Return the QuantizationSpec for activation tensors (per-tensor).

        QuantizationSpec bundles the observer class, dtype, numeric range, and
        quantization scheme into one object.  prepare_pt2e reads these fields to
        decide which observer module to insert and how to configure it.

        Observers must come from torchao.quantization.pt2e, not torch.ao —
        torchao's convert_pt2e only recognises torchao observer classes; torch.ao
        observers would be left as opaque call_module nodes after conversion.
        """
        if self.dtype == "int8":
            torch_dtype, quant_min, quant_max = torch.int8, _INT8_MIN, _INT8_MAX
        else:
            torch_dtype, quant_min, quant_max = torch.int16, _INT16_MIN, _INT16_MAX
        qscheme = (
            torch.per_tensor_symmetric if self.symmetric_activations else torch.per_tensor_affine
        )
        return QuantizationSpec(
            dtype=torch_dtype,
            observer_or_fake_quant_ctr=_ACTIVATION_OBSERVERS[self.activation_observer],
            quant_min=quant_min,
            quant_max=quant_max,
            qscheme=qscheme,
        )

    def _clamp_range_spec(self, bound: float) -> FixedQParamsQuantizationSpec:
        """Return a fixed per-tensor symmetric spec covering [-bound, bound]."""
        if self.dtype == "int8":
            torch_dtype, quant_min, quant_max = torch.int8, _INT8_MIN, _INT8_MAX
        else:
            torch_dtype, quant_min, quant_max = torch.int16, _INT16_MIN, _INT16_MAX
        # The scale MinMaxObserver gives a tensor whose extreme is bound. A clamp
        # that saturates is observed with the same scale at its output, so its
        # input and output scales match and the backend lowers it as an integer
        # clamp with no requantize.
        return FixedQParamsQuantizationSpec(
            dtype=torch_dtype,
            scale=bound / ((quant_max - quant_min) / 2),
            zero_point=0,
            quant_min=quant_min,
            quant_max=quant_max,
            qscheme=torch.per_tensor_symmetric,
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
        annotated: list[Node] = []

        for node in model.graph.nodes:
            if node.op != "call_function":
                continue
            all_ops = (
                _WEIGHT_OPS
                | _ACT_ONLY_OPS
                | _TRANSPARENT_OPS
                | _TIDL_ACT_OPS
                | _AVG_POOL_OPS
                | _NORM_OPS
            )
            if node.target not in all_ops:
                continue
            if node.meta.get("quantization_annotation") is not None:
                continue  # already annotated by a composed quantizer

            output_qspec: QuantizationSpecBase = act_spec

            if node.target in _WEIGHT_OPS:
                # bias (args[2]) stays float32 — not annotated
                input_qspec_map: dict[Node, QuantizationSpec] = {
                    node.args[0]: act_spec,  # type: ignore[index]
                    node.args[1]: weight_spec,  # type: ignore[index]
                }
                bound = _hardtanh_bound(node)
                if bound is not None and self.symmetric_activations:
                    output_qspec = self._clamp_range_spec(bound)
            elif node.target in (_TIDL_ACT_OPS | _AVG_POOL_OPS | _NORM_OPS):
                # Single activation input; weight/bias (if any) stay float32.
                if not isinstance(node.args[0], Node) or not _is_float_tensor(node.args[0]):
                    continue
                input_qspec_map = {node.args[0]: act_spec}  # type: ignore[index]
            elif node.target in _TRANSPARENT_OPS:
                # Single activation input; output shares scale with input so
                # EliminateQDQTransparent can remove the QDQ wrappers.
                # SharedQuantizationSpec(n) requires n to already be annotated;
                # fall back to independent act_spec when it isn't (e.g. relu_
                # after cat, or relu_ after an unannotated in-place op).
                if not isinstance(node.args[0], Node) or not _is_float_tensor(node.args[0]):
                    continue
                if _feeds_topk(node):
                    # A transparent op feeding a topk -- directly, or through
                    # a chain of further transparent/structural ops (e.g. the
                    # permute right after a detection head's box+score concat,
                    # which flows through split/getitem/max into topk in
                    # yolo26's postprocess) -- must stay float. topk's ranking
                    # is sensitive to the relative ordering of values, and
                    # quantizing this region with a scale calibrated for a
                    # different value range (e.g. box coordinates up to ~320
                    # sharing a tensor with [0,1] sigmoid probabilities)
                    # corrupts that ordering -- in the worst case crushing
                    # every score to the same quantized value, so topk's
                    # selection degenerates to picking by tie-break order
                    # instead of by score.
                    continue
                input_qspec_map = {node.args[0]: act_spec}  # type: ignore[index]
                input_node = node.args[0]
                if input_node.meta.get("quantization_annotation") is not None:
                    output_qspec = SharedQuantizationSpec(input_node)  # type: ignore[arg-type]
                # else: output_qspec stays act_spec (independent observer)
            elif node.target in (
                torch.ops.aten.mm.default,
                torch.ops.aten.add.Tensor,
                torch.ops.aten.add_.Tensor,
            ):
                # Skip if either argument is a scalar (not an FX Node) or not
                # a floating-point tensor. This happens for ops like `x + 0`
                # in attention masking; only Tensor + Tensor residual adds
                # between float activations should be quantized.
                if (
                    not isinstance(node.args[0], Node)
                    or not isinstance(node.args[1], Node)
                    or not _is_float_tensor(node.args[0])
                    or not _is_float_tensor(node.args[1])
                ):
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
                output_qspec=output_qspec,
                _annotated=True,  # prevents double-annotation when composed with another quantizer
            )
            annotated.append(node)

        if self.dtype == "int8":
            self._clip_ranges(annotated, act_spec)
            if self.depthwise_input_headroom is not None and self.symmetric_activations:
                self._per_channel_depthwise_inputs(model)
            if self.float_layernorm_residuals:
                self._float_layernorm_residuals(annotated)
        return model

    def _float_layernorm_residuals(self, annotated: list[Node]) -> None:
        """Drop the annotations that quantize the residual stream of LayerNorm blocks.

        The stream's adds lose theirs, and so do the edges a layer_norm or another of
        these adds reads them on, and any annotated pass-through op (view, permute) in
        between, whose output spec would share the add's. Every other reader keeps its
        input spec, so PT2E quantizes the stream for it alone.
        """
        if not any(n.target in _NORM_OPS for n in annotated):
            return
        residual = {n for n in annotated if _is_layernorm_residual_add(n)}
        stream = residual | {
            n
            for n in annotated
            if n.target in _ADDS
            and any(
                reader.target in _NORM_OPS or reader in residual
                for reader, _ in _readers(n)
                if reader.op == "call_function"
            )
        }
        for node in stream:
            node.meta.pop("quantization_annotation", None)
        for node in stream:
            for reader, source in _readers(node):
                annotation = reader.meta.get("quantization_annotation")
                if annotation is None:
                    continue
                if reader.target in _PASS_THROUGH_OPS:
                    reader.meta.pop("quantization_annotation")
                elif reader.target in _NORM_OPS:
                    annotation.input_qspec_map.pop(source, None)
                elif (
                    isinstance(annotation.output_qspec, SharedQuantizationSpec)
                    and annotation.output_qspec.edge_or_node is source
                    and source in annotation.input_qspec_map
                ):
                    # A transparent op (relu, max_pool2d) shared the add's output
                    # spec; share the edge it still quantizes on instead.
                    annotation.output_qspec = SharedQuantizationSpec((source, reader))

    def _per_channel_depthwise_inputs(self, model: GraphModule) -> None:
        """Quantize each _depthwise_input_chains tensor per channel.

        Its output spec and the edges of the readers that can take per-channel scales
        (_reads_per_channel) get one spec, capped at the per-tensor spec it had. Any
        other reader keeps that per-tensor spec, so PT2E gives it a second observer
        and a requantize: a residual add reads the block's input per tensor, as
        before. A ReLU's output shares its conv's spec, so the depthwise convs
        reading the ReLU take the conv's.
        """

        def readers(node, per_channel_only=True):
            for user in node.users:
                annotation = user.meta.get("quantization_annotation")
                if annotation is None or node not in annotation.input_qspec_map:
                    continue
                if not per_channel_only or _reads_per_channel(user, node):
                    yield annotation

        def per_channel(node) -> QuantizationSpec | None:
            annotation = node.meta.get("quantization_annotation")
            own = annotation.output_qspec if annotation is not None else None
            edges = list(readers(node))
            base = own if isinstance(own, QuantizationSpec) else None
            if base is None:
                every = list(readers(node, per_channel_only=False))
                base = every[0].input_qspec_map[node] if every else None
            if not isinstance(base, QuantizationSpec) or base.qscheme != torch.per_tensor_symmetric:
                return None
            spec = QuantizationSpec(
                dtype=base.dtype,
                quant_min=base.quant_min,
                quant_max=base.quant_max,
                qscheme=torch.per_channel_symmetric,
                ch_axis=1,
                observer_or_fake_quant_ctr=_CappedPerChannelObserver.with_args(
                    headroom=self.depthwise_input_headroom,
                    per_tensor=base.observer_or_fake_quant_ctr,
                ),
            )
            if isinstance(own, QuantizationSpec):
                annotation.output_qspec = spec  # type: ignore[union-attr]
            for edge in edges:
                edge.input_qspec_map[node] = spec
            return spec

        for conv, act in _depthwise_input_chains(model):
            spec = per_channel(conv)
            if spec is None:
                continue
            if act.target in _RELUS:
                for edge in readers(act):
                    edge.input_qspec_map[act] = spec
            else:
                per_channel(act)

    def _percentile_for(self, node: Node) -> float | None:
        """Percent clipped from each tail of node's output, or None for act_spec's range."""
        if self.smooth_activation_percentile is not None and _around_smooth_activation(node):
            return self.smooth_activation_percentile
        if self.depthwise_block_percentile is not None and _in_depthwise_block(node):
            return self.depthwise_block_percentile
        return None

    def _clip_ranges(self, annotated: list[Node], act_spec) -> None:
        """Give the tensors the percentile options name a percentile range.

        The choice is per tensor: its output spec and every input edge reading it
        switch together. An edge whose spec differed from its producer's output
        spec would get a second observer, and so a requantize between them.
        """
        specs: dict[float, QuantizationSpec] = {}

        def clip_spec(percentile: float) -> QuantizationSpec:
            if percentile not in specs:
                specs[percentile] = dataclasses.replace(
                    act_spec,
                    observer_or_fake_quant_ctr=_PercentileObserver.with_args(percentile=percentile),
                )
            return specs[percentile]

        for node in annotated:
            annotation = node.meta["quantization_annotation"]
            for arg, spec in list(annotation.input_qspec_map.items()):
                percentile = self._percentile_for(arg) if spec == act_spec else None
                if percentile is None:
                    continue
                annotation.input_qspec_map[arg] = clip_spec(percentile)
                # A transparent op after an unannotated producer (max_pool2d after
                # an in-place silu_) has an output observer of its own; make it
                # follow the clipped edge, or the op would requantize.
                if node.target in _TRANSPARENT_OPS and annotation.output_qspec == act_spec:
                    annotation.output_qspec = SharedQuantizationSpec((arg, node))
            if annotation.output_qspec == act_spec:
                percentile = self._percentile_for(node)
                if percentile is not None:
                    annotation.output_qspec = clip_spec(percentile)

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
