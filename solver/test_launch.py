#!/usr/bin/env python3
"""Tests for solver.launch -- down-look launch enumeration (T1.1).

The endgame is the player looking DOWN at the platform.  These tests pin the
asymmetry the ROM's looking-up waiver ($1D2E) creates: a tile can down-shot the
platform even when the reverse platform-vantage up-shot to that tile is blocked.

Discrepancy vs the planner's pinned "human ls0 win" numbers (reported in the T1.1
PR): the pinned launch tile (2,10) @ eye 9.375 does NOT have down-look line of
sight to the platform (12,4) in the bit-exact simulator (the ray is blocked by the
terrain ridge; the furthest tile toward the platform it reaches is (9,3)), and the
platform (12,4) sits at the maximum ls0 terrain height (object z-height 8), so no
bare-terrain tile can down-look it -- launch needs BUILT height.  These tests
therefore assert the true, verified geometry while preserving the down-look
asymmetry the task is about: a concrete launch tile whose down-shot lands and whose
reverse up-shot is blocked.
"""

import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from solver.plan_game import PlanGame  # noqa: E402
from solver import launch  # noqa: E402
from sentinel import los, memmap as mm  # noqa: E402

PLAT = (12, 4)
# A verified down-look launch tile: the down-shot to the platform lands, but the
# symmetric platform-vantage up-shot to it is blocked (the $1D2E asymmetry).
LAUNCH_TILE = (14, 4)
# A tallest-terrain (z8) far dead-end corner -- the kind of tile the old
# platform-vantage launch-readiness signal stranded the climb on
# (climb_search._launch_tiles docstring): it can never down-shot the platform.
DEAD_END_CORNER = (28, 18)


def _plat_ground(g):
    """The real ls0 platform ground (object z-height 8), not the planner's pinned 9
    (which no ls0 terrain tile reaches -- see the module discrepancy note)."""
    return g.plat_ground


def test_launch_tiles_contains_verified_launch_tile():
    g = PlanGame(0)
    tiles = launch.launch_tiles(g.state, PLAT, _plat_ground(g))
    assert LAUNCH_TILE in tiles
    assert PLAT not in tiles  # the platform tile itself is excluded


def test_launch_tiles_excludes_dead_end_corner():
    g = PlanGame(0)
    tiles = launch.launch_tiles(g.state, PLAT, _plat_ground(g))
    assert DEAD_END_CORNER not in tiles


def test_down_look_los_from_launch_tile():
    """A player standing on LAUNCH_TILE at a launch eye (8.875, a synthoid on the
    z8 terrain -- eye strictly above the z8 platform) down-looks the platform."""
    g = PlanGame(0)
    pslot = g.state.player
    assert launch._place_phantom(g.state, LAUNCH_TILE, pslot) is not None
    g.state.obj_z_height[pslot] = 8
    g.state.obj_z_frac[pslot] = 0xE0  # eye 8.875
    g.eye = 8.875
    assert g.player_xy() == LAUNCH_TILE
    assert launch.down_look_los(g, PLAT) is True
    assert launch.endgame_ready(g, PLAT, _plat_ground(g)) is True


def test_reverse_up_shot_is_blocked():
    """The symmetric platform-vantage reverse sweep does NOT see LAUNCH_TILE: aiming
    FROM the platform tile UP does not reach it, even though the down-shot the other
    way does (test above).  This is the down-look asymmetry the endgame relies on."""
    g = PlanGame(0)
    plat_slot = g.state.slot_of_type(mm.T_PLATFORM)
    assert plat_slot is not None
    up_shot = los.sees_tile(
        g.state, LAUNCH_TILE, plat_slot, eye_z=_plat_ground(g) + 1, max_steps=2000
    )
    assert up_shot is False


def test_pinned_human_launch_tile_cannot_downshot():
    """Documents the discrepancy: the pinned human-win launch tile (2,10) @ 9.375
    does NOT down-look the platform in the bit-exact sim (ray blocked), so it is not
    a launch tile.  Kept as a regression guard on the reported finding."""
    g = PlanGame(0)
    tiles = launch.launch_tiles(g.state, PLAT, _plat_ground(g))
    assert (2, 10) not in tiles
    pslot = g.state.player
    assert launch._place_phantom(g.state, (2, 10), pslot) is not None
    g.state.obj_z_height[pslot] = 9
    g.state.obj_z_frac[pslot] = 0x60  # eye 9.375, standing on a boulder
    g.eye = 9.375
    assert launch.down_look_los(g, PLAT) is False
