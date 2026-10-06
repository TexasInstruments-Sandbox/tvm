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
        pattern, mod, verdict, expect = p.values
        if verdict:
            continue
        out[p.id] = D._run(mod, pattern).reason
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
            note = prev.get(key, {}).get("note") or classify(text)
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
