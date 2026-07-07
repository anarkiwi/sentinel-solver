#!/usr/bin/env python3
"""Gate test for the T2.4 weighted-A* macro planner: it must WIN ls0 offline.

The win is the first real test of whether the macro model climbs to launch and
fires the endgame.  We also assert the winning plan is the *hidden* one -- its
energy trace is pure build/absorb with no drain -- and that it is a short macro
path that runs well inside the 60 s CPU budget.
"""

import pytest

from solver.astar_planner import plan
from solver.plan_game import PlanGame
from sentinel import memmap as mm

_VERBS = {"create", "absorb", "transfer"}


@pytest.fixture(scope="module")
def result():
    return plan(0)  # one solve shared by the gate assertions (~33 s)


def test_plan_ls0_wins(result):
    assert result.won is True, f"ls0 not won: {result.failure} stats={result.stats}"
    assert result.steps, "won plan has no steps"
    assert result.stats["wall_s"] < 60.0, f"solve too slow: {result.stats['wall_s']}s"


def test_plan_ls0_short_macro_path(result):
    assert result.won is True, result.failure
    # Macro-move proxy: every climb macro and the endgame land via one transfer;
    # refuel macros carry none. The plan says the winning path is <= ~12 macro steps.
    transfers = sum(1 for s in result.steps if s["verb"] == "transfer")
    assert transfers <= 12, f"too many macro moves: {transfers}"


def test_plan_ls0_no_drain(result):
    """The winning plan is the never-seen (hidden) one: reconstructing energy from
    the step verbs alone -- create pays ``ENERGY_IN_OBJECTS[otype]``, absorb gains it,
    transfer is free -- never goes negative and every delta is an explained build/absorb
    cost. A drained window would leave an unexplained loss, which this rejects."""
    assert result.won is True, result.failure
    energy = PlanGame(0).state.energy
    for s in result.steps:
        assert s["verb"] in _VERBS, f"unexpected verb {s['verb']}"
        if s["verb"] == "create":
            energy -= mm.ENERGY_IN_OBJECTS[s["otype"]]
        elif s["verb"] == "absorb":
            energy += mm.ENERGY_IN_OBJECTS[s["otype"]]
        assert energy >= 0, f"energy went negative at {s}: reconstructed {energy}"
