#!/usr/bin/env python3
"""Tests for the move cost model (solver/cost.py).

The headline gate: ``move_rounds``'s settle contribution equals the sum of
``actioncost.action_rounds`` that ``run_plan_simulated.execute_step`` advances the world
by for the fired verbs -- create(s) -> transfer -> look-back absorb -- for both a hop and
a 1-boulder climb step on landscape 0. ``move_rounds`` charges the flat per-verb SETTLE
floors (no scene-redraw edge term), so the equivalence is exact up to
``STEPS_PER_EDGE * visible_edges`` per view, which the test strips out explicitly.
"""

import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from solver import cost  # noqa: E402
from solver.plan_game import PlanGame, terrain_z, N  # noqa: E402
from sentinel import aimcost as ac, actioncost  # noqa: E402


def _adjacent_bare_tile(g):
    """A bare-terrain tile next to the player that isn't the player's own tile."""
    px, py = g.player_xy()
    for dx, dy in ((1, 0), (-1, 0), (0, 1), (0, -1), (1, 1), (-1, -1)):
        t = (px + dx, py + dy)
        if 0 <= t[0] < N and 0 <= t[1] < N and terrain_z(g.mem, *t) is not None:
            return t
    raise AssertionError("no adjacent bare tile")


def _edges(mem, view):
    return actioncost.STEPS_PER_EDGE * actioncost.visible_edges(mem, view)


def test_aim_rounds_zero_when_no_view():
    assert cost.aim_rounds(0, 0, None) == 0.0
    assert cost.aim_rounds(0, 0, {}) == 0.0
    assert cost.aim_rounds(0, 0, {"h_angle": None, "v_angle": 0x20}) == 0.0


def test_aim_rounds_known_nonzero():
    # bearing 0 -> 64 = 8 lattice steps (no U-turn) * 16 = 128; pitch 0x10 -> 0x20 = 4
    # steps * 8 = 32; total 160.
    view = {"h_angle": 64, "v_angle": 0x20}
    assert cost.aim_rounds(0, 0x10, view) == 160.0
    # pitch ignored when the start pitch is unknown.
    assert cost.aim_rounds(0, None, view) == 128.0


def _move_vs_execute(use_boulder):
    """move_rounds' total == aim pan + the action_rounds settle execute_step applies for
    the create(s) -> transfer -> absorb sequence (edge term stripped)."""
    g = PlanGame(0)
    g.energy = 30
    t2 = _adjacent_bare_tile(g)
    px, py = g.player_xy()
    # A view-shaped aim at the build tile: a real aim centres on this bearing.
    view = {"h_angle": ac.bearing_to(px, py, t2[0], t2[1]), "v_angle": 0x00}
    vh, vv = 0, 0
    n_boulders = 1

    total, end_h, end_v = cost.move_rounds(g, t2, use_boulder, n_boulders, view, vh, vv)

    # Independently reconstruct the aim pan the model charges.
    aim = cost.aim_rounds(vh, vv, view)
    if use_boulder:
        aim += cost.ROUNDS_PER_H_STEP + cost.ROUNDS_PER_V_STEP
    back_h = ac.bearing_to(t2[0], t2[1], px, py)
    aim += ac.bearing_rounds(
        view["h_angle"], back_h, cost.ROUNDS_PER_H_STEP, cost.ROUNDS_PER_UTURN
    )

    # The verbs execute_step fires, with the exact (view, stacked) each action_rounds
    # call uses: a boulder-step lays boulder (bare, not stacked) then a synthoid on the
    # boulder (stacked); a hop lays a lone synthoid; both transfer then reabsorb the
    # departed shell (a coarse absorb resolved with no carried view).
    if use_boulder:
        scripted = [
            ("create", view, False),
            ("create", view, True),
            ("transfer", view, False),
            ("absorb", None, False),
        ]
    else:
        scripted = [
            ("create", view, actioncost.is_stacked(g.mem, t2)),
            ("transfer", view, False),
            ("absorb", None, False),
        ]
    exec_settle = sum(
        actioncost.action_rounds(g.mem, verb, v, stacked=s) for verb, v, s in scripted
    )
    edges = sum(_edges(g.mem, v) for _verb, v, _s in scripted)

    # move_rounds omits the scene-redraw edge term; strip it from the execute_step sum.
    assert abs(total - (aim + exec_settle - edges)) <= 1.0
    assert end_h == back_h
    assert end_v == view["v_angle"]


def test_move_rounds_hop_matches_execute():
    _move_vs_execute(use_boulder=False)


def test_move_rounds_boulder_step_matches_execute():
    _move_vs_execute(use_boulder=True)
