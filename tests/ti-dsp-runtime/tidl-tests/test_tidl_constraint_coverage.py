#!/usr/bin/env python
"""Constraint-coverage gate (plan section 5.6, test layer 1).

TIDL's own allowlisting constraints are the spec for what the library may
decline. ``constraint_coverage.json`` classifies each one for a Relax layer type
that has a parser:

* ``covered``   - a decline-table case makes the library print it;
* ``subsumed``  - a stricter Relax-side decline rejects the configuration first;
* ``untested``  - no case yet (``note`` says whether it is unreachable from
  Relax: another frontend's attribute or a firmware gate).

The test fails when a constraint is added, removed or re-worded upstream without
the map being updated, when a ``covered``/``subsumed`` entry stops holding, and
when the number of constraints that still need a case grows.
Regenerate with ``python diag_tidl_constraint_coverage.py --write`` and review.
"""

import os

import constraint_coverage as cc
import pytest
import test_tidl_declines as D

from tvm.relax.backend.tidl import _gate

pytestmark = [pytest.mark.quick]

NEEDS_CASE = "needs a decline-table case"


@pytest.fixture(scope="module")
def coverage():
    if not os.path.isdir(cc.CONSTRAINT_DIR):
        pytest.skip("c7x-mma-tidl constraint headers not found")
    _gate.load_relax_so(tidl_tools_path=os.path.join(cc.C7X_MMA_TIDL_PATH, "tidl_tools"))
    return cc.load_map()


@pytest.fixture(scope="module")
def cases():
    return {p.id: p for p in D.CASES}


def test_inventory_matches_map(coverage):
    """A new, removed or changed TIDL constraint must be classified in the map."""
    inv = set(cc.inventory())
    mapped = set(coverage)
    assert inv - mapped == set(), f"constraints with no classification: {sorted(inv - mapped)}"
    assert mapped - inv == set(), (
        f"stale map entries (constraint gone or re-worded): {sorted(mapped - inv)}"
    )


def test_covered_constraints_are_still_triggered(coverage, cases):
    for key, entry in coverage.items():
        if entry["status"] != "covered":
            continue
        text = key.split("::", 1)[1]
        hit = []
        for cid in entry["cases"]:
            pattern, mod, verdict, _ = cases[cid].values
            assert not verdict, f"{cid} is listed as rejecting"
            if f"[constraint: {text}" in D._run(mod, pattern).reason:
                hit.append(cid)
        assert hit, f"no listed case still triggers {key!r}: {entry['cases']}"


def test_subsumed_constraints_are_still_rejected(coverage, cases):
    for key, entry in coverage.items():
        if entry["status"] != "subsumed":
            continue
        for cid in entry["cases"]:
            pattern, mod, verdict, _ = cases[cid].values
            assert not verdict and not D._run(mod, pattern).verdict, (key, cid)


def test_untested_count_does_not_grow(coverage):
    """Ratchet: lower this number as cases are added; it may never rise."""
    needs_case = [
        k for k, v in coverage.items() if v["status"] == "untested" and v["note"] == NEEDS_CASE
    ]
    assert len(needs_case) <= NEEDS_CASE_BASELINE, (
        f"{len(needs_case)} constraints need a case (baseline {NEEDS_CASE_BASELINE})"
    )


def test_untested_entries_explain_themselves(coverage):
    for key, entry in coverage.items():
        if entry["status"] == "untested":
            assert entry.get("note"), key


# Constraints that still need a decline-table case (see the JSON). Lower it as
# cases are added.
NEEDS_CASE_BASELINE = 63
