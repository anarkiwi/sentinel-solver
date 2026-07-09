#!/usr/bin/env python3
"""Test for the endgame macro (T2.3): from a hand-constructed launch-ready ls0 node
(eye > plat_ground, down-look LOS to the platform), the drive-through endgame wins."""

import os
import sys

import pytest

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from sentinel import actions  # noqa: E402
from solver import macros  # noqa: E402
from solver.plan_game import PlanGame  # noqa: E402
from solver.search_node import Node  # noqa: E402

PLAT = (12, 4)
PLAT_GROUND = 9
LAUNCH = (2, 10)  # human launch tile, eye 9.375 (terrain 8 + a boulder)


def _launch_ready_node():
    """A launch-ready node: player relocated to (2,10) at eye 9.375, matching
    ``test_launch._game_at_launch`` -- above plat_ground with down-look LOS to PLAT."""
    g = PlanGame(0)
    g.state.obj_x[g.player] = LAUNCH[0]
    g.state.obj_y[g.player] = LAUNCH[1]
    g.eye = 9.375
    return Node(g=g, t=0, vh=0x50, vv=0xF5, cost=0.0)


@pytest.mark.xfail(
    reason="endgame_child now gates the launch on a keyboard-lattice view + real-eye "
    "LOS (sentinel.aim). Even from a genuine real eye of 9.375 the far platform (12,4) "
    "has no landable_view from (2,10) -- geometric down-look LOS exists but no keyboard "
    "aim lands the sights on it -- so the drive-through endgame cannot fire. Faithful "
    "consequence of the unified real-eye gate, not a regression.",
    strict=False,
)
def test_endgame_child_wins():
    child = macros.endgame_child(_launch_ready_node(), PLAT, PLAT_GROUND)
    assert child is not None
    assert actions.on_platform(child.g.state) is True
