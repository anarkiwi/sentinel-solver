"""Tests for the receding-horizon best-first climb search (climb_search.py)."""

import os
import sys

import pytest

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.dirname(HERE))


def test_clone_isolation():
    """clone() must fully decouple every MUTABLE piece of state so a search branch
    cannot leak into the parent (or a sibling) it was cloned from."""
    from solver import plan_game

    g = plan_game.PlanGame(0)
    g.create(3, (g.player_xy()[0] + 2, g.player_xy()[1]), None, "seed boulder")
    snap_mem = bytes(g.mem)
    snap_col = dict(g.col)
    snap_energy = g.energy
    snap_steps = len(g.steps)
    snap_eye = g.eye

    c = g.clone()
    # mutate every container/scalar on the clone
    c.mem[0x0900] ^= 0xFF
    c.energy -= 3
    c.col[(1, 1)] = 9.0
    c.steps.append({"verb": "x"})
    c.eye += 1.0

    assert bytes(g.mem) == snap_mem
    assert g.col == snap_col
    assert g.energy == snap_energy
    assert len(g.steps) == snap_steps
    assert g.eye == snap_eye


def test_clone_equivalent_start():
    """A fresh clone is byte-for-byte and field-for-field equal to its parent."""
    from solver import plan_game

    g = plan_game.PlanGame(0)
    c = g.clone()
    assert bytes(c.mem) == bytes(g.mem)
    assert c.col == g.col
    assert (c.energy, c.eye, c.player) == (g.energy, g.eye, g.player)
    assert c.plat == g.plat and c.plat_ground == g.plat_ground
    assert c.sentinel_slot == g.sentinel_slot


def test_cost_and_ticks_monotone():
    """A boulder-step costs more energy AND more ticks than a hop (the search relies on
    both for affordability filtering and enemy-state advancement)."""
    from solver import climb_search as CS
    from solver import plan_game

    g = plan_game.PlanGame(0)
    hop = ((1, 2), False, 6.0, None)
    boulder = ((1, 2), True, 6.0, None)
    assert CS._cost(g, boulder, set()) > CS._cost(g, hop, set())
    hop_ticks, _, _ = CS._move_cost(g, hop, None, None)
    boulder_ticks, _, _ = CS._move_cost(g, boulder, None, None)
    assert boulder_ticks > hop_ticks
    # exposure adds reserve to the up-front cost
    assert CS._cost(g, hop, {(1, 2)}) > CS._cost(g, hop, set())


def test_move_cost_prices_return_pan_and_geometry():
    """The move cost prices the WHOLE keyboard sequence -- not a flat per-move constant.
    (a) It includes the return-pan to reabsorb the departed tile's shell, so even a move
    whose build-aim needs no pan still costs pan keystrokes to swing the view back;
    (b) the tick cost varies with the foothold's bearing geometry (the thing the flat
    constant ignored); (c) the ending heading it reports is the return-pan bearing, so the
    next move chains from where the view actually ends."""
    from solver import climb_search as CS
    from solver import plan_game

    g = plan_game.PlanGame(0)
    px, py = g.player_xy()
    east = ((px + 3, py), False, 6.0, {"h_angle": 0x00, "v_angle": 0xF5})
    south = ((px, py + 3), False, 6.0, {"h_angle": 0x40, "v_angle": 0xF5})
    # (a) build-aim is already on-heading (view h == cur h) yet the move still costs pan
    # keystrokes -- the return swing back to the departed tile behind the new foothold.
    fires_only = 2 * CS.ROUNDS_PER_ACTION  # hop = synthoid + transfer, no reabsorb pan
    east_ticks, end_h, _ = CS._move_cost(g, east, 0x00, 0xF5)
    assert east_ticks > fires_only
    # (b) a different bearing -> a different combined aim+return pan -> a different cost.
    south_ticks, _, _ = CS._move_cost(g, south, 0x00, 0xF5)
    assert south_ticks != east_ticks
    # (c) the ending heading is the bearing from the new foothold back to the departed tile.
    from sentinel import aimcost as ac

    assert end_h == ac.bearing_to(px + 3, py, px, py)


def test_read_state_returns_sentinel_state():
    """_read_state hands back the node's live sentinel State (enemy timing lives there)."""
    from solver import climb_search as CS
    from solver import plan_game

    g = plan_game.PlanGame(0)
    assert CS._read_state(g) is g.state


def test_advance_enemies_rotation_forecast_pure_on_energy():
    """With seen-drain off, _advance_enemies advances enemy rotation IN PLACE but restores
    the player energy (rotation forecast only, the ROM-validated default accounting)."""
    from solver import climb_search as CS
    from solver import plan_game

    g = plan_game.PlanGame(0)
    e0 = g.state.energy
    CS._advance_enemies(g.state, 40, apply_drain=False)
    assert g.state.energy == e0


def test_reached_approach_flags_endgame_state():
    """_reached_approach is False from the start tile (eye not yet above the platform)."""
    from solver import climb_search as CS
    from solver import plan_game

    g = plan_game.PlanGame(0)
    ctx = CS.climb_ctx(g, toward_plat=False)
    assert CS._reached_approach(g, ctx) is False


def test_search_climbs_without_height_regression():
    """The core property the redesign restores (SEARCH_REDESIGN.md sec.1/sec.9): the
    lookahead never commits a move that LOSES height. Run a few real decisions and
    assert the eye is monotonically non-decreasing across committed steps -- the exact
    invariant the old greedy 'reposition to a lower tile' fallback violated."""
    from solver import climb_search as CS
    from solver import plan_game

    g = plan_game.PlanGame(0)
    ctx = CS.climb_ctx(g, toward_plat=False)
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


def test_refuel_drain_charges_full_absorb_cost():
    """A drain-aware refuel must advance the world by the WHOLE cost of each absorb --
    the aim pan PLUS the execute/settle (sentinel.actioncost) -- not the pan alone.
    Charging only the pan let a nearby tree's tiny pan skip the drain cooldown, so every
    tree booked a phantom +1 even while the player was being drained (the fruitless
    seen-tile refuel the fix removes). On a SAFE start tile no drain fires over that
    window, so the absorb still nets its full +1 (legitimate refuel preserved)."""
    from solver import climb_search as CS
    from solver import plan_game
    from sentinel import actioncost, actions

    prev = CS._REFUEL_DRAIN
    CS._REFUEL_DRAIN = True
    try:
        g = plan_game.PlanGame(0)
        ticks = {"n": 0}
        real = CS._advance_enemies

        def spy(state, t, apply_drain):
            ticks["n"] += t
            return real(state, t, apply_drain)

        CS._advance_enemies = spy
        try:
            e0 = g.energy
            gained = CS._refuel(g, lambda *a: None)
        finally:
            CS._advance_enemies = real
    finally:
        CS._REFUEL_DRAIN = prev
    # at least one absorb happened on the ls0 start refuel...
    assert gained > 0 and g.energy > e0
    # ...and the world was advanced by at least the absorb SETTLE floor per absorb --
    # proof the execute time is charged (pan-only would be a small fraction of this).
    assert ticks["n"] >= actioncost.SETTLE["absorb"]
    assert not actions.player_dead(g.state)  # safe tile: refuel does not kill


def test_refuel_drain_stops_when_drained_to_death():
    """If the drain-honest window kills the player mid-refuel (seen tile, drained faster
    than trees bank), the refuel must STOP crediting absorbs and search_iterate must end
    the climb as 'stuck' -- never loop retries or hand a dead state to the endgame (a
    false win). Modelled by forcing the death flag inside the world advance."""
    from solver import climb_search as CS
    from solver import plan_game
    from sentinel import actions
    from sentinel import memmap as mm

    prev = CS._REFUEL_DRAIN
    CS._REFUEL_DRAIN = True
    try:
        g = plan_game.PlanGame(0)
        real = CS._advance_enemies

        def kill(state, t, apply_drain):
            state.mem[mm.PLAYER_DIED_BY_DRAINING] |= 0x80  # drained to death mid-window

        CS._advance_enemies = kill
        try:
            e_before = g.energy
            CS._refuel(g, lambda *a: None)
            assert actions.player_dead(g.state)
            assert g.energy == e_before  # no absorb credited after death

            g2 = plan_game.PlanGame(0)
            ctx = CS.climb_ctx(g2, toward_plat=False)
            status = CS.search_iterate(g2, ctx, set(), lambda *a: None, depth=2, beam=2)
            assert status == "stuck"
        finally:
            CS._advance_enemies = real
    finally:
        CS._REFUEL_DRAIN = prev


@pytest.mark.skipif(
    os.environ.get("RUN_SLOW_WIN") != "1",
    reason="full offline climb ~85s (over the 60s dev budget); "
    "run with RUN_SLOW_WIN=1, or reproduce via `python3 solver/climb_search.py 0 2`",
)
def test_plan_search_wins_ls0():
    """A full depth-2 offline climb of ls0 reaches the win (native_won True).

    Gated behind RUN_SLOW_WIN because the bit-exact sentinel line-of-sight makes a
    full multi-decision plan take ~85s; the fast tests above cover the per-decision
    search behaviour (progress, no height regression, clone isolation)."""
    from solver import climb_search as CS

    g = CS.plan_search(0, verbose=False, depth=2)
    assert g.native_won is True
