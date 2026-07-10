#!/usr/bin/env python3
"""Gate for the T2.1 climb macro (solver/macros.py).

For a concrete reachable ls0 foothold (8,16), the climb expander emits ``Node``s tagged
``macro["kind"]=="climb"`` with non-regressing eye, including the chosen foothold.
"""

import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from solver import macros  # noqa: E402
from solver.plan_game import PlanGame  # noqa: E402
from solver.search_node import Node  # noqa: E402
from sentinel import memmap as mm  # noqa: E402

FOOTHOLD = (8, 16)  # a bare-terrain foothold in the ls0 initial keyboard LOS sweep


def _root():
    g = PlanGame(0)
    ph = g.player
    vh = int(g.mem[mm.OBJECTS_H_ANGLE + ph])
    vv = int(g.mem[mm.OBJECTS_V_ANGLE + ph])
    return g, Node(g=g, t=0, vh=vh, vv=vv, cost=0.0)


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
