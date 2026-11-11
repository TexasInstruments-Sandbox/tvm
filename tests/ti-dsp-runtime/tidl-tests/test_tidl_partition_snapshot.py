#!/usr/bin/env python
"""Partition must match the committed snapshots (Track 1 regression guard).

A difference means an op started or stopped being offloaded. Classify it
(plan section 5.7), then regenerate with
``python diag_tidl_partition_snapshot.py --write`` and explain it in the PR.
"""

import os

import partition_snapshot as ps
import pytest
from dsp_utils import has_tidl_tools, tidl_tools_config

pytestmark = [pytest.mark.quick]


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
    if not has_tidl_tools():
        pytest.skip("TIDL tools directory not staged")
    builder = ps.layer_model_builders()[name.split("/", 1)[1]]
    assert ps.snapshot_module(ps.partition_layer_model(builder)) == _SNAPSHOTS[name]


@pytest.mark.parametrize("name", sorted(k for k in _SNAPSHOTS if k.startswith("model/")))
def test_model_partition_matches_snapshot(name):
    if not has_tidl_tools():
        pytest.skip("TIDL tools directory not staged")
    try:
        mod = ps.partition_real_model(name.split("/", 1)[1], tidl_tools_config())
    except (ImportError, OSError) as e:
        # Only a missing model source/weights file is a skip. RuntimeError is
        # deliberately not caught: that is what a failed partition and a
        # missing library raise, and both are the regression this guards.
        pytest.skip(f"model unavailable: {e}")
    assert ps.snapshot_module(mod) == _SNAPSHOTS[name]
