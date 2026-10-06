"""Constraint inventory for the coverage gate (plan section 5.6, test layer 1).

TIDL's own allowlisting constraints are the spec for what the library may
decline: the first argument of each ``TIDL_CSTR(...)`` in
``tidlAllowlistingConstraints/tidl_constraint_*.h``. This module extracts them
for the layer types that have a Relax parser; ``constraint_coverage.json`` maps
each to the decline-table cases (``test_tidl_declines.py``) that make the
library emit it, or marks it ``untested``. The coverage test fails when a
constraint appears, disappears or changes text without the map being updated,
and when the number of untested constraints grows.
"""

import json
import os
import re

C7X_MMA_TIDL_PATH = os.environ.get("C7X_MMA_TIDL_PATH", os.path.expanduser("~/ml/c7x-mma-tidl"))
CONSTRAINT_DIR = os.path.join(
    C7X_MMA_TIDL_PATH, "ti_dl/utils/tidlModelImport/tidlAllowlistingConstraints"
)
COVERAGE_PATH = os.path.join(os.path.dirname(os.path.abspath(__file__)), "constraint_coverage.json")

# Constraint header (without tidl_constraint_/.h) -> the Relax composites it governs.
# Headers for layer types with no Relax parser are out of scope; Track 2 adds
# entries here as it adds parsers.
HEADER_TO_PATTERNS = {
    "abs": ["tidl.abs"],
    "acos": ["tidl.acos"],
    "argop": ["tidl.argmax", "tidl.argmin"],
    "asin": ["tidl.asin"],
    "asinh": ["tidl.asinh"],
    "atan": ["tidl.atan"],
    "batchNorm": ["tidl.nn.batch_norm"],
    "cast": ["tidl.cast"],
    "clip": ["tidl.clip"],
    "concat": ["tidl.concat"],
    "conv": ["tidl.nn.conv2d"],
    "cos": ["tidl.cos"],
    "cosh": ["tidl.cosh"],
    "deconv": ["tidl.nn.conv2d_transpose"],
    "depthToSpace": ["tidl.nn.depth_to_space"],
    "eltwise": ["tidl.add", "tidl.multiply", "tidl.divide", "tidl.subtract"],
    "elu": ["tidl.elu"],
    "erf": ["tidl.erf"],
    "exp": ["tidl.exp"],
    "expand": ["tidl.expand"],
    "flatten": ["tidl.flatten"],
    "floor": ["tidl.floor"],
    "gather": ["tidl.take"],
    "gridsample": ["tidl.image.grid_sample"],
    "hardSigmoid": ["tidl.hard_sigmoid"],
    "hardswish": ["tidl.hard_swish"],
    "innerProduct": ["tidl.matmul", "tidl.matmul_bias"],
    "instanceNormalization": ["tidl.nn.instance_norm"],
    "layernorm": ["tidl.nn.layer_norm"],
    "leakyRelu": ["tidl.nn.leakyrelu"],
    "log": ["tidl.log"],
    "mish": ["tidl.mish"],
    "neg": ["tidl.negative"],
    "pad": ["tidl.nn.pad"],
    "pooling": ["tidl.nn.max_pool2d", "tidl.nn.avg_pool2d"],
    "pow": ["tidl.power"],
    "prelu": ["tidl.nn.prelu"],
    "reduce": ["tidl.reduce_max", "tidl.reduce_min"],
    "reduceMean": ["tidl.mean"],
    "reduceSum": ["tidl.sum"],
    "relu": ["tidl.nn.relu"],
    "reshape": ["tidl.reshape"],
    "resize": ["tidl.image.resize2d"],
    "scatterElements": ["tidl.scatter_elements", "tidl.scatter_nd"],
    "sigmoid": ["tidl.sigmoid"],
    "sin": ["tidl.sin"],
    "sinh": ["tidl.sinh"],
    "slice": ["tidl.strided_slice", "tidl.split"],
    "softmax": ["tidl.nn.softmax"],
    "sqrt": ["tidl.sqrt"],
    "tan": ["tidl.tan"],
    "tanh": ["tidl.tanh"],
    "topK": ["tidl.topk"],
    "transpose": ["tidl.permute_dims"],
    "unsqueeze": ["tidl.expand_dims"],
}

# The first argument is the message TIDL prints when the constraint fails.
# Constraints with an empty first argument print nothing, so a rejection cannot
# be attributed to them; they are left out of the inventory.
_CSTR = re.compile(r'TIDL_CSTR\(\s*"((?:[^"\\]|\\.)+)"', re.S)


def inventory():
    """``{"<header>::<text>": [patterns]}`` for every in-scope constraint."""
    out = {}
    for header, patterns in HEADER_TO_PATTERNS.items():
        path = os.path.join(CONSTRAINT_DIR, f"tidl_constraint_{header}.h")
        if not os.path.isfile(path):
            continue
        with open(path) as f:
            src = f.read()
        for text in _CSTR.findall(src):
            out[f"{header}::{text}"] = patterns
    return out


def load_map():
    with open(COVERAGE_PATH) as f:
        return json.load(f)
