#!/usr/bin/env python
"""Write or diff the Track 1 partition snapshots.

python diag_tidl_partition_snapshot.py --write      # (re)generate the JSON
python diag_tidl_partition_snapshot.py              # diff against the JSON
python diag_tidl_partition_snapshot.py --models resnet18 yolov8n
"""

import argparse
import json
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import partition_snapshot as ps  # noqa: E402
from dsp_utils import tidl_tools_config  # noqa: E402


def collect(only=None):
    out = {}
    for name, builder in sorted(ps.layer_model_builders().items()):
        if only and name not in only:
            continue
        out[f"layer/{name}"] = ps.snapshot_module(ps.partition_layer_model(builder))
    for name in ps.REAL_MODELS:
        if only and name not in only:
            continue
        try:
            out[f"model/{name}"] = ps.snapshot_module(
                ps.partition_real_model(name, tidl_tools_config())
            )
        except Exception as e:  # noqa: BLE001
            print(f"SKIPPED {name}: {type(e).__name__}: {str(e)[:150]}")
    return out


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--write", action="store_true")
    ap.add_argument("--models", nargs="*")
    args = ap.parse_args()
    cur = collect(args.models)
    if args.write:
        old = ps.load_snapshots() if os.path.exists(ps.SNAPSHOT_PATH) else {}
        old.update(cur)
        with open(ps.SNAPSHOT_PATH, "w") as f:
            json.dump(old, f, indent=1, sort_keys=True)
            f.write("\n")
        print(f"wrote {len(cur)} snapshots to {ps.SNAPSHOT_PATH}")
    else:
        base = ps.load_snapshots()
        bad = 0
        for k, v in sorted(cur.items()):
            if base.get(k) != v:
                bad += 1
                print(f"DIFF {k}")
        print(f"{bad} of {len(cur)} differ")
        sys.exit(1 if bad else 0)
