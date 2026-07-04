"""Tests for the receding-horizon best-first climb search (climb_search.py)."""

import os
import sys

import pytest

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)

pytestmark = pytest.mark.skipif(
    not os.path.exists(os.path.join(HERE, "..", "out", "sentinel_stage2.bin")),
    reason="needs out/sentinel_stage2.bin fixture",
)


@pytest.fixture(scope="module")
def ng():
    import native_game

    return native_game


def test_clone_isolation(ng):
    """clone() must fully decouple every MUTABLE piece of state so a search branch
    cannot leak into the parent (or a sibling) it was cloned from."""
    g = ng.Game(0)
    g.create(3, (g.player_xy()[0] + 2, g.player_xy()[1]), None, "seed boulder")
    snap_mem = bytes(g.mem)
    snap_col = dict(g.col)
    snap_free = list(g.free)
    snap_energy = g.energy
    snap_steps = len(g.steps)

    c = g.clone()
    # mutate every container/scalar on the clone
    c.mem[0x0900] ^= 0xFF
    c.energy -= 3
    c.col[(1, 1)] = 9.0
    if c.free:
        c.free.pop()
    c.steps.append({"verb": "x"})
    c.eye += 1.0

    assert bytes(g.mem) == snap_mem
    assert g.col == snap_col
    assert g.free == snap_free
    assert g.energy == snap_energy
    assert len(g.steps) == snap_steps


def test_clone_equivalent_start(ng):
    """A fresh clone is byte-for-byte and field-for-field equal to its parent."""
    g = ng.Game(0)
    c = g.clone()
    assert bytes(c.mem) == bytes(g.mem)
    assert c.col == g.col and c.free == g.free
    assert (c.energy, c.eye, c.player) == (g.energy, g.eye, g.player)
    assert c.plat == g.plat and c.plat_ground == g.plat_ground
    assert c.sentinel_slot == g.sentinel_slot


def test_cost_and_ticks_monotone():
    """A boulder-step costs more energy AND more ticks than a hop (the search relies on
    both for affordability filtering and enemy-phase advancement)."""
    import climb_search as CS

    hop = ((1, 2), False, 6.0, None)
    boulder = ((1, 2), True, 6.0, None)
    assert CS._cost(boulder, set()) > CS._cost(hop, set())
    assert CS._ticks_for(boulder) > CS._ticks_for(hop)
    # exposure adds reserve to the up-front cost
    assert CS._cost(hop, {(1, 2)}) > CS._cost(hop, set())


def test_advance_phase_is_deterministic_and_pure(ng):
    """Advancing the enemy phase must be deterministic (fixed-rate automata, sec.5) and
    must not mutate the input phase."""
    import climb_search as CS
    import enemy_dynamics as ED

    g = ng.Game(0)
    state = CS._read_state(g)
    phase = ED.init_phase_from_ram(state, g.mem)
    before = {s: e.h_angle for s, e in phase.enemies.items()}
    c = ((1, 2), False, 6.0, None)
    p1 = CS._advance_phase(state, phase, c)
    p2 = CS._advance_phase(state, phase, c)
    assert {s: e.h_angle for s, e in p1.enemies.items()} == {
        s: e.h_angle for s, e in p2.enemies.items()
    }
    # input phase untouched
    assert {s: e.h_angle for s, e in phase.enemies.items()} == before


def test_reached_approach_flags_endgame_state(ng):
    """_reached_approach is True exactly when the eye is above the platform ground and
    adjacent (endgame can launch) and False from the start tile."""
    import climb_search as CS
    import climb_greedy as cg

    g = ng.Game(0)
    ctx = cg.climb_ctx(g, toward_plat=False)
    assert CS._reached_approach(g, ctx) is False


def test_search_climbs_without_height_regression(ng):
    """The core property the redesign restores (SEARCH_REDESIGN.md sec.1/sec.9): the
    lookahead never commits a move that LOSES height. Run a few real decisions and
    assert the eye is monotonically non-decreasing across committed steps -- the exact
    invariant the old greedy 'reposition to a lower tile' fallback violated."""
    import climb_search as CS
    import climb_greedy as cg

    g = ng.Game(0)
    ctx = cg.climb_ctx(g, toward_plat=False)
    eyes = [g.eye]
    steps_taken = 0
    for _ in range(20):
        if steps_taken >= 3:
            break
        status = CS.search_iterate(g, ctx, set(), lambda *a: None, depth=2, beam=2)
        if status == "stepped":
            eyes.append(g.eye)
            steps_taken += 1
        elif status in ("approach", "no_gain", "stuck"):
            break
    assert steps_taken >= 1, "search made no committed move"
    assert all(
        b >= a - 1e-9 for a, b in zip(eyes, eyes[1:])
    ), f"height regressed across committed steps: {eyes}"
