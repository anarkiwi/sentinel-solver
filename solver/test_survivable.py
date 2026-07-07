#!/usr/bin/env python3
"""Tests for the managed-exposure survivability test (solver/cost.py:survivable).

Headline gate: over a window the Sentinel actually drains, ``survivable``'s returned
``energy_after`` equals ``energy_before - threat.drain_over_window(state, W)`` EXACTLY --
both drive the identical ``enemies.step`` loop, so the timed-race feasibility test is the
true transition, not an approximation. A state whose buffer cannot absorb the drain
returns ``ok=False``.
"""

import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from solver import cost  # noqa: E402
from solver.plan_game import PlanGame, terrain_z  # noqa: E402
from sentinel import memmap as mm, terrain, threat  # noqa: E402

# A high, bare ls0 tile the Sentinel has terrain line-of-sight to at t=0 (ticks_until_seen
# == 0); the drain cooldown reaches its fire point well inside this window.
SEEN_TILE = (13, 2)
WINDOW = 450


def _seen_ls0():
    """ls0 with the player object relocated onto a Sentinel-seen tile so the true
    transition drains the player over the window. Placement mirrors
    ``threat._place_phantom``'s bare-terrain math (z from the terrain nibble, z_frac
    0xE0, flags 0), with the departed tile restored to bare terrain."""
    g = PlanGame(0)
    s = g.state
    pslot = s.player
    px, py = g.player_xy()
    old_z = terrain_z(g.mem, px, py)
    terrain.set_tile_byte(
        s, px, py, (old_z if old_z is not None else s.obj_z_height[pslot]) << 4
    )
    tx, ty = SEEN_TILE
    s.obj_x[pslot] = tx
    s.obj_y[pslot] = ty
    s.obj_z_height[pslot] = terrain_z(g.mem, tx, ty)
    s.obj_z_frac[pslot] = 0xE0
    s.obj_flags[pslot] = 0x00
    terrain.set_tile_byte(s, tx, ty, mm.OBJECT_TILE | pslot)
    return g


def test_seen_state_actually_drains():
    """The construction genuinely exposes the player: the window drains energy."""
    g = _seen_ls0()
    assert g.player_xy() == SEEN_TILE
    assert threat.drain_over_window(g.state, WINDOW) > 0


def test_survivable_matches_drain_over_window():
    """energy_after == energy_before - drain_over_window(W), exactly."""
    g = _seen_ls0()
    e_before = g.energy
    # drain_over_window clones internally, leaving g.state pristine for survivable, which
    # then steps the very same start state -- identical starting conditions.
    ref_drain = threat.drain_over_window(g.state, WINDOW)
    assert ref_drain > 0
    ok, energy_after = cost.survivable(g, SEEN_TILE, WINDOW)
    assert energy_after == e_before - ref_drain
    # a comfortable buffer over the floor survives.
    assert energy_after >= cost.NEXT_COST_FLOOR
    assert ok is True


def test_survivable_insufficient_buffer_fails():
    """A buffer the drain pushes below NEXT_COST_FLOOR is not survivable."""
    g = _seen_ls0()
    g.energy = (
        cost.NEXT_COST_FLOOR
    )  # the drain over the window drops this under the floor
    ok, energy_after = cost.survivable(g, SEEN_TILE, WINDOW)
    assert energy_after < cost.NEXT_COST_FLOOR
    assert ok is False
