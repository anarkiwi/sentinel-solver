#!/usr/bin/env python3
"""Gate for the T2.1 climb macro (solver/macros.py).

Parity assertion against the validated ``climb_search._apply`` while it still exists: for
a concrete reachable ls0 foothold (8,16), the child ``_apply_climb`` / ``expand_climb``
produces reproduces the SAME final eye and energy the ported ``_apply`` yields for the
same foothold/use_boulder/view (the window is hidden at t=0, so the survivability step
drains nothing and energy stays at parity).  Also checks the expander emits ``Node``s
tagged ``macro["kind"]=="climb"`` with non-regressing eye.
"""

import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from solver import macros, climb_search as CS  # noqa: E402
from solver.plan_game import PlanGame  # noqa: E402
from solver.search_node import Node  # noqa: E402
from sentinel import los  # noqa: E402
from sentinel import memmap as mm  # noqa: E402

FOOTHOLD = (8, 16)  # a bare-terrain foothold in the ls0 initial keyboard LOS sweep


def _root():
    g = PlanGame(0)
    ph = g.player
    vh = int(g.mem[mm.OBJECTS_H_ANGLE + ph])
    vv = int(g.mem[mm.OBJECTS_V_ANGLE + ph])
    return g, Node(g=g, t=0, vh=vh, vv=vv, cost=0.0)


def _foothold_view(g):
    views, _c = los.landable_sweep_with_centres(
        g.state, g.player, int(g.eye), max_steps=6000
    )
    assert FOOTHOLD in views, "chosen foothold not in the ls0 initial sweep"
    return views[FOOTHOLD]


def test_apply_climb_matches_climb_search_apply():
    """_apply_climb reproduces _apply's eye + energy for the hop and the boulder-step."""
    g0, root = _root()
    view = _foothold_view(g0)
    for use_b in (False, True):
        n_b = macros._boulder_batch(g0, FOOTHOLD)[0] if use_b else 0
        n_b = min(n_b, macros.MAX_BATCH)
        child = macros._apply_climb(root, FOOTHOLD, use_b, n_b, view, None)
        assert child is not None, f"foothold rejected (use_b={use_b})"
        assert isinstance(child, Node)
        assert child.macro["kind"] == "climb"
        assert child.macro["tile"] == list(FOOTHOLD)
        assert child.g.eye >= g0.eye - 1e-9  # non-regression

        gref = g0.clone()
        CS._apply(gref, FOOTHOLD, use_b, view)
        assert abs(child.g.eye - gref.eye) < 1e-9
        assert child.g.energy == gref.energy


def test_expand_climb_emits_climb_nodes():
    """expand_climb yields Node children, all tagged climb and non-regressing, including
    the chosen foothold."""
    g0, root = _root()
    children = macros.expand_climb(root, None)
    assert children, "no climb children on the ls0 initial state"
    for c in children:
        assert isinstance(c, Node)
        assert c.macro["kind"] == "climb"
        assert c.g.eye >= g0.eye - 1e-9
    tiles = {tuple(c.macro["tile"]) for c in children}
    assert FOOTHOLD in tiles
