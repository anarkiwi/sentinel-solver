"""Outcome classification for a fired plan step: the tree-absorb vs Sentinel-
discharge divergence must resync (diverge), not retry (best_effort_miss)."""

import os
import sys
import types

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from driver import sentinel_execute as sx  # noqa: E402
from sentinel import memmap as mm  # noqa: E402


def _objs(n):
    return types.SimpleNamespace(objects=list(range(n)), player=None)


def test_tree_absorb_with_discharge_verifies_false_on_dtot():
    """A tree absorb that removed its on-tile object but coincided with a Sentinel
    discharge (a fresh tree elsewhere) leaves the global count unchanged: verify()
    rejects on dtot, though the primary on-tile absorb succeeded."""
    before, after = _objs(5), _objs(5)  # -1 absorbed + 1 discharged == net 0
    ok, msg = sx.verify("absorb", mm.T_TREE, (5, 30), before, after, 1, 0, 62, 62, 6, 7)
    assert ok is False
    assert "global object count by 0" in msg


def test_discharge_divergence_classifies_diverge_not_miss():
    """The regression: with the on-tile object removed (primary_ok) the outcome is
    diverge (resync + replan), NOT best_effort_miss (which the caller would retry)."""
    assert (
        sx.classify_outcome("absorb", mm.T_TREE, ok=False, primary_ok=True) == "diverge"
    )


def test_genuine_fuel_miss_is_best_effort():
    """A non-Sentinel absorb that removed nothing (primary_ok False) is a survivable
    fuel-recovery miss."""
    for otype in (mm.T_TREE, mm.T_BOULDER, mm.T_ROBOT):
        assert (
            sx.classify_outcome("absorb", otype, ok=False, primary_ok=False)
            == "best_effort_miss"
        )


def test_sentinel_absorb_miss_is_fatal():
    """Absorbing the Sentinel (otype 5) is not fuel recovery: a miss stays fail."""
    assert (
        sx.classify_outcome("absorb", mm.T_SENTINEL, ok=False, primary_ok=False)
        == "fail"
    )


def test_create_and_transfer_ordering():
    assert sx.classify_outcome("create", mm.T_BOULDER, True, False) == "ok"
    assert sx.classify_outcome("create", mm.T_BOULDER, False, True) == "diverge"
    assert sx.classify_outcome("create", mm.T_BOULDER, False, False) == "fail"
    assert sx.classify_outcome("transfer", mm.T_ROBOT, False, True) == "diverge"
    assert sx.classify_outcome("transfer", mm.T_ROBOT, False, False) == "fail"
