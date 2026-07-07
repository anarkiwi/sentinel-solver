#!/usr/bin/env python3
"""Tests for the visibility-deferred refuel macro (T2.2): it stays silent while
energy is unconstrained and, when energy-blocked, absorbs reachable below-eye fuel
for a survivable, energy-raising child."""

import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from solver import cost, macros  # noqa: E402
from solver.plan_game import PlanGame  # noqa: E402
from solver.search_node import Node, energy  # noqa: E402

NEED = cost.NEXT_COST_FLOOR + 2


def _node(g):
    return Node(g=g, t=0, vh=0x50, vv=0xF5, cost=0.0)


def test_refuel_deferred_when_unconstrained():
    g = PlanGame(0)  # fresh ls0 starts with ample energy
    assert g.energy >= NEED
    kids = macros.expand_refuel(_node(g), None)
    assert kids == []  # pylint: disable=use-implicit-booleaness-not-comparison


def test_refuel_emits_energy_raising_child_when_blocked():
    g = PlanGame(0)
    g.energy = cost.NEXT_COST_FLOOR  # below need -> refuel is enabled
    n = _node(g)
    kids = macros.expand_refuel(n, None)
    assert kids, "a reachable below-eye tree should yield at least one refuel child"
    assert all(energy(k) > energy(n) for k in kids)
