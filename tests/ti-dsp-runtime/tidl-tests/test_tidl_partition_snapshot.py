#!/usr/bin/env python
"""Partition must match the committed snapshots (Track 1 regression guard).

A difference means an op started or stopped being offloaded. Classify it
(plan section 5.7), then regenerate with
``python diag_tidl_partition_snapshot.py --write`` and explain it in the PR.
"""

import os

import partition_snapshot as ps
import pytest

pytestmark = [pytest.mark.quick]

C7X_MMA_TIDL_PATH = os.environ.get("C7X_MMA_TIDL_PATH", os.path.expanduser("~/ml/c7x-mma-tidl"))
TIDL_TOOLS_PATH = os.path.join(C7X_MMA_TIDL_PATH, "tidl_tools")
RELAX_SO = os.path.join(
    C7X_MMA_TIDL_PATH, "ti_dl/utils/tidlModelImport/out/tidl_model_import_relax.so"
)

_SNAPSHOTS = ps.load_snapshots() if os.path.exists(ps.SNAPSHOT_PATH) else {}


def test_snapshot_file_is_present():
    """A missing baseline would make every test below vanish, silently.

    The parametrized tests take their ids from the snapshot file, so without it
    they collect zero cases and the whole guard reports green. Fail here
    instead. The file is committed (see ``.gitignore``'s exception for it).
    """
    assert _SNAPSHOTS, f"no partition snapshots at {ps.SNAPSHOT_PATH}"


@pytest.mark.parametrize("name", sorted(k for k in _SNAPSHOTS if k.startswith("layer/")))
def test_layer_partition_matches_snapshot(name):
    if not os.path.isfile(RELAX_SO):
        pytest.skip("tidl_model_import_relax.so not built")
    builder = ps.layer_model_builders()[name.split("/", 1)[1]]
    assert ps.snapshot_module(ps.partition_layer_model(builder)) == _SNAPSHOTS[name]


@pytest.mark.parametrize("name", sorted(k for k in _SNAPSHOTS if k.startswith("model/")))
def test_model_partition_matches_snapshot(name):
    if not os.path.isfile(RELAX_SO):
        pytest.skip("tidl_model_import_relax.so not built")
    try:
        mod = ps.partition_real_model(name.split("/", 1)[1], TIDL_TOOLS_PATH)
    except (ImportError, OSError) as e:
        # Only a missing model source/weights file is a skip. RuntimeError is
        # deliberately not caught: that is what a failed partition and a
        # missing library raise, and both are the regression this guards.
        pytest.skip(f"model unavailable: {e}")
    assert ps.snapshot_module(mod) == _SNAPSHOTS[name]
