#!/usr/bin/env python
"""(Re)generate constraint_coverage.json from the decline-table cases.

Runs every case in test_tidl_declines.py, records which TIDL constraint texts
appear in the library's rejection reasons, and writes the map. Review the diff:
a constraint that drops from ``covered`` to ``untested`` lost its test.

    python diag_tidl_constraint_coverage.py          # print a summary
    python diag_tidl_constraint_coverage.py --write  # update the JSON
"""

import argparse
import json
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import constraint_coverage as cc  # noqa: E402
import test_tidl_declines as D  # noqa: E402

from tvm.relax.backend.tidl import _gate  # noqa: E402


def case_reasons():
    """``{case_id: reason}`` for every rejecting case in the decline table."""
    _gate.load_relax_so(tidl_tools_path=os.path.join(cc.C7X_MMA_TIDL_PATH, "tidl_tools"))
    out = {}
    for p in D.CASES:
        pattern, mod, verdict, _, config = p.values
        if verdict:
            continue
        out[p.id] = D._run(mod, pattern, config).reason
    return out


_OTHER_FRONTEND = (
    "Caffe",
    "ONNX",
    "NHWC",
    "Gemm",
    "proto",
    "allowzero",
    "initializer",
    "firmware version",
    "Training mode",
    "TfLite",
    "TFLite",
    "bool, float32 and signed/unsigned",
    "round mode",
    "saturate mode",
    "antialias",
    "exclude_outside",
    "keep_aspect_ratio",
    "DCR",
)


def classify(text):
    """Why a constraint has no case yet (reviewed by hand; edit the JSON to refine)."""
    if any(k in text for k in _OTHER_FRONTEND):
        return "not reachable from Relax: another frontend's attribute or a firmware gate"
    if text.startswith(("Number of non-singleton", "Number of variable input")):
        return "generic rank limit; see the per-op 7-D cases or the stricter rank bound in declines"
    return "needs a decline-table case"


# Constraints that a stricter Relax-side decline (tidl_parse_relax_declines.cpp or a
# parser) rejects first, so TIDL's own text is never printed for the case. Value is
# the decline-table case that proves the configuration is rejected.
SUBSUMED = {
    "argop::Only axis = -3 is supported": "argmax-height",
    "argop::Only keepdims = 1 (default) is supported": "argmin-no-keepdims",
    "concat::Not supported along the batch dimension": "concat-batch-4d",
    "conv::Stride must be the same along both horizontal and vertical dimensions": (
        "conv2d-unequal-strides"
    ),
    "conv::Kernel size greater than 7 with stride 2 is not supported": "conv2d-kernel-9",
    "pad::Only constant pad mode with constant_value = 0 is supported": "pad-nonzero-value",
    "pad::Padding is only supported for width/height axes": "pad-channel",
    "pad::Pad layer is expected to provide 8 pad values": "pad-rank3",
    "reduce::Reduction is only supported along height axis": "max-width",
    "resize::Only 'nearest' and 'linear' resize mode are supported": "resize-cubic",
    "gather::Input data dimensions must be greater than 1D": "take-rank1-data",
    "gather::Only 1D indices are supported": "take-2d-indices",
    "conv::Only one variable input is allowed": "conv2d-variable-weight",
    "deconv::Only one variable input is allowed": "deconv-variable-weight",
    "depthToSpace::Input should be four-dimensional (4D)": "depth-to-space-rank5",
    "expand::Shape tensor cannot be a variable input": "expand-variable-shape",
    "innerProduct::Filter tensor input should have atleast 2 dimensions": "matmul-1d-weight",
    "layernorm::Dimension of scale and bias vector can either be [1, N] or [N]": (
        "layer-norm-height-width"
    ),
    "pad::Maximum number of input dimension supported is 6": "pad-rank7",
    "pooling::Only 1D and 2D GlobalAveragePool is supported": "mean-rank5-three-axes",
    "reshape::Output dimensions must be less than ": "reshape-target-rank7",
    "reshape::Variable shape is not supported": "reshape-variable-shape",
    "resize::'pytorch_half_pixel' coordinate_transformation_mode is not supported for Resize "
    "output shapes <= 1": "resize-pytorch-half-pixel-1x1",
    "resize::Only 'half_pixel', 'pytorch_half_pixel' and 'asymmetric' "
    "coordinate_transformation_mode is supported": "resize-tf-half-pixel-for-nn",
    "resize::Resize is only supported along width and height axis": "resize-nhwc",
    "scatterElements::The constant input 'data' must be a zero tensor": (
        "scatter-elements-const-data"
    ),
    "slice::Slice with stride greater than 0 is supported": "slice-negative-stride",
    "topK::TopK axis other than width and height is not supported": "topk-channel-axis",
    "unsqueeze::Output dimensions after unsqueeze must be <= 6": "expand-dims-to-rank7",
    "cast::Only bool, float32 and signed/unsigned 8/16/32/64-bit integers are supported as cast "
    "destination types": "cast-to-int32",
    "eltwise::Number of non-singleton variable input dimensions in Add/Mul/Sub/Div/Max/Min/Sum/Mod "
    "must be <= 6": "add-rank5",
    "instanceNormalization::Number of variable input dimensions must be >= 3": (
        "instance-norm-rank3"
    ),
    "pooling::Input should be variable and not constant/initializer": "maxpool-constant-input",
}

_UNREACHABLE = "not reachable from Relax: "
_TFLITE_ONLY = _UNREACHABLE + "the check applies only to the TFLite model types"
_TWO_OPERANDS = _UNREACHABLE + "Relax binary elementwise ops always have exactly two operands"
_DEAD = _UNREACHABLE + "dead constraint, every path returns true"

# Constraints no Relax module can trip, each with the reason (reviewed by hand
# against the constraint's code, not its text).
UNREACHABLE = {
    "batchNorm::Scale, Bias, input_mean and input_var should be constant 1-D tensor of size "
    "same as input channel dimension": _UNREACHABLE
    + "no tidl.nn.batch_norm composite exists; prepare() folds batch_norm into conv2d",
    "conv::Padding greater than input width is not supported for AM62A and AM67A": _UNREACHABLE
    + "the check is skipped for the TVM_RELAY model type the Relax importer uses",
    "deconv::Change to Upsample/Resize if possible. Upsample/Resize will be more efficient": (
        _UNREACHABLE + "warning only, never declines"
    ),
    "depthToSpace::Input depth (channel dimension) should be multiple of (blocksize * "
    "blocksize)": _UNREACHABLE + "pixel_shuffle struct inference rejects such a module",
    "eltwise::Only 2 inputs are supported": _TWO_OPERANDS,
    "eltwise::Only 2 inputs are supported in Add/Mul/Sub/Div/Max/Min/Sum/Mod layers": _TWO_OPERANDS,
    "eltwise::Too many inputs for elementwise operator": _UNREACHABLE
    + "compiled out (TIDL_EN_MULTI_TENSOR_ELT_WISE is not defined)",
    "eltwise::The variable inputs in Add/Mul/Div/Sub/Max/Min/Sum/Mod layer must of be same "
    "dimensions or broadcast-able": _DEAD,
    "eltwise::Eltwise operator(Add/Mul/Div/Sub/Max/Min/Sum/Mod layer) is supported only with "
    "operands of similar dimensions or broadcast supported patterns of both inputs": _DEAD,
    "innerProduct::Filter and input must be of same dimensions or broadcast-able": _UNREACHABLE
    + "needs two different non-1 channel dims, which matmul broadcasting rejects",
    "innerProduct::MatMul with signed inputs & unsigned output is not supported ": _UNREACHABLE
    + "checked only on optimised layers (layer->optimized == 1), never at allowlisting",
    "instanceNormalization::Scale and Bias should be 1-D tensor of size same as input channel "
    "dimension": _UNREACHABLE
    + "instance_norm struct inference requires 1-D gamma/beta of the channel size",
    "pooling::Number of outputs should be 1": _UNREACHABLE + "Relax pooling ops have one output",
    "reshape::Input volume should be equal to output volume": _UNREACHABLE
    + "reshape struct inference rejects a volume change",
    "slice::Must have 4 inputs": _TFLITE_ONLY,
    "slice::Number of dimensions for output tensor must be 4": _TFLITE_ONLY,
    "slice::Number of dimensions for variable inputs must be 4": _TFLITE_ONLY,
    "slice::Only batch size = 1 is supported": _TFLITE_ONLY,
    "slice::Only one input should be variable": _TFLITE_ONLY,
    "argop::select_last_index = 1 is only supported in firmware version >= 11_00_07_00": (
        _UNREACHABLE + "relax.argmax/argmin have no select_last_index; the parser sets 0"
    ),
    "innerProduct::MatMul with signed & unsigned input combination is not supported in TDA4VM & "
    "is only supported in firmware version >= 10_00_07_00": _UNREACHABLE
    + "relax.matmul requires both operands to have the same dtype",
    "topK::TopK is not supported with 'sorted' attribute is set to 0": _UNREACHABLE
    + "relax.topk has no 'sorted' attribute; results are always sorted",
}


def build_map():
    reasons = case_reasons()
    prev = cc.load_map() if os.path.exists(cc.COVERAGE_PATH) else {}
    result = {}
    for key in sorted(cc.inventory()):
        text = key.split("::", 1)[1]
        cases = sorted(cid for cid, reason in reasons.items() if f"[constraint: {text}" in reason)
        if cases:
            result[key] = {"status": "covered", "cases": cases}
        elif key in SUBSUMED:
            result[key] = {"status": "subsumed", "cases": [SUBSUMED[key]]}
        else:
            note = UNREACHABLE.get(key) or prev.get(key, {}).get("note") or classify(text)
            result[key] = {"status": "untested", "note": note}
    return result


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--write", action="store_true")
    args = ap.parse_args()
    m = build_map()
    counts = {
        st: sum(v["status"] == st for v in m.values()) for st in ("covered", "subsumed", "untested")
    }
    print(f"{counts} of {len(m)} in-scope constraints")
    if args.write:
        with open(cc.COVERAGE_PATH, "w") as f:
            json.dump(m, f, indent=1, sort_keys=True)
            f.write("\n")
        print("wrote", cc.COVERAGE_PATH)
