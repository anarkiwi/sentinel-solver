#!/usr/bin/env python3
"""Tests for PlanGame -- the sentinel-backed native_game.Game adapter."""

import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from solver.plan_game import PlanGame, cheb, terrain_z, N  # noqa: E402
from sentinel import memmap as mm  # noqa: E402


def _adjacent_bare_tile(g):
    """A bare-terrain tile next to the player that isn't the player's own tile."""
    px, py = g.player_xy()
    for dx, dy in ((1, 0), (-1, 0), (0, 1), (0, -1), (1, 1), (-1, -1)):
        t = (px + dx, py + dy)
        if 0 <= t[0] < N and 0 <= t[1] < N and terrain_z(g.mem, *t) is not None:
            return t
    raise AssertionError("no adjacent bare tile")


def test_start_state():
    g = PlanGame(0)
    assert g.player_xy() == (8, 17)
    assert g.energy == 10
    assert isinstance(g.plat, tuple) and len(g.plat) == 2
    assert g.plat_ground is None or isinstance(g.plat_ground, int)
    assert isinstance(g.eye, float)
    # fresh-board start eye: plain terrain height, or (start tile is object-
    # occupied by the player robot) the player's z_height with the $E0 render
    # fraction dropped -- byte-for-byte the native_game baseline.
    px, py = g.player_xy()
    tz = terrain_z(g.mem, px, py)
    expected_eye = float(tz if tz is not None else g.state.obj_z_height[g.state.player])
    assert g.eye == expected_eye
    assert g.sentinel_slot is not None
    assert isinstance(g.free, list) and g.free == sorted(g.free)
    assert len(g.steps) == 0 and g.native_won is False


def test_top_of_offboard():
    g = PlanGame(0)
    assert g.top_of((-1, 0)) is None
    assert g.top_of((N, N)) is None
    # a bare-terrain tile resolves to its terrain height (a float)
    assert isinstance(g.top_of(_adjacent_bare_tile(g)), float)


def test_feasible_bool():
    g = PlanGame(0)
    assert isinstance(g.feasible(0, g.plat), bool)
    assert g.feasible(3, g.player_xy()) is False  # own tile


def test_module_helpers():
    assert cheb((0, 0), (3, 5)) == 5
    g = PlanGame(0)
    px, py = g.player_xy()
    assert terrain_z(g.mem, px, py) == terrain_z(g.state, px, py)
    assert mm.tidx(px, py) == ((px & 3) * 256 + ((px >> 2) & 7) * 32 + py)


def test_create_transfer_raises_eye_and_records_step():
    g = PlanGame(0)
    g.energy = 30
    tile = _adjacent_bare_tile(g)
    start_eye = g.eye
    n0 = len(g.steps)
    slot = g.create(3, tile, None, "boulder")  # boulder on adjacent bare terrain
    assert slot is not None
    assert tile in g.col
    syn = g.create(0, tile, None, "synthoid on boulder")
    assert syn is not None
    g.transfer(syn, "step up")
    assert g.eye > start_eye
    assert g.player_xy() == tile
    verbs = [s["verb"] for s in g.steps[n0:]]
    assert verbs == ["create", "create", "transfer"]


def test_clone_independent():
    g = PlanGame(0)
    g.energy = 30
    tile = _adjacent_bare_tile(g)
    g2 = g.clone()
    g2.create(3, tile, None, "on clone only")
    assert tile in g2.col
    assert tile not in g.col
    assert len(g2.steps) == 1 and len(g.steps) == 0
    assert g2.energy != g.energy  # clone spent energy, original untouched


def test_absorb_repairs_col():
    g = PlanGame(0)
    g.energy = 30
    tile = _adjacent_bare_tile(g)
    slot = g.create(3, tile, None, "boulder")
    assert tile in g.col
    g.absorb(slot, None, "remove boulder")
    assert tile not in g.col  # bare-terrain absorb removes the column entry
