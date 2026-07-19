"""The A* planner prices each action with the executor's real aim+settle cost.

The (13,27) death (docs/player.md) was a search under-charging an action's aim, so
it planned a body onto a tile a further-rotated cone in fact sees.  A full solve
exceeds the CI budget, so this locks the cost-parity and placement guarantees.
"""

import math

import pytest

from sentinel import actions, enemies, memmap as mm, projector, terrain
from sentinel.game import Game
from sentinel.astar_player import AStarPlayer
from sentinel.playerbase import SIGHTS_CENTRE

_LANDSCAPE = 42  # player starts at (14,27), a down-look hollow adjacent to (13,27)


def _reset_stance(player, st):
    """Seed the player at the live start stance the search seeds ``start`` from."""
    player.st = st
    player.last_bearing = None
    player.cursor = list(SIGHTS_CENTRE)


def _feasible_tiles(player, st, want=3):
    """A few keyboard-landable tiles with a real view."""
    out = []
    for tile in sorted(player._landset(st)):
        _reset_stance(player, st)
        view = player._view_for(tile)
        if view is not None:
            out.append((tile, view))
        if len(out) >= want:
            break
    return out


def test_charge_matches_executor_cost_and_advances_enemies():
    """Each search charge == executor ``_step_aim_frames + _settle`` over the same
    view, advances enemies by ``int(cost)``, and mirrors the post-aim stance."""
    game = Game.new(_LANDSCAPE)
    player = AStarPlayer(game)
    st = game.state
    tiles = _feasible_tiles(player, st)
    assert tiles, "no landable tile to price"
    for tile, view in tiles:
        for verb in ("boulder", "robot", "transfer", "absorb"):
            _reset_stance(player, st)
            expected = player._step_aim_frames(verb, view) + player._settle(verb, view)
            clone = st.clone()
            reference = st.clone()
            enemies.advance_frames(reference, int(expected))
            _reset_stance(player, st)
            cost = player._charge(clone, verb, tile)
            assert cost == expected, f"{verb}@{tile}: {cost} != executor {expected}"
            for e in enemies.enemy_slots(st):
                assert clone.obj_h_angle[e] == reference.obj_h_angle[e]
            me = clone.player
            assert clone.obj_h_angle[me] == view["h_angle"]
            assert clone.obj_v_angle[me] == view["v_angle"]
            assert player.cursor == list(view["cursor"])
            if verb == "transfer":
                assert player.last_bearing is None
            else:
                assert player.last_bearing == (view["h_angle"], view["v_angle"])


def test_transfer_charges_zero_aim_only_on_a_reused_bearing():
    """A transfer over the bearing the preceding same-tile create left committed fires on
    the object under the parked cursor (no aim keys), so its charge is 0; on a STALE
    bearing ``live_player._drive_transfer_aim`` drives the full view, which must cost the
    same as any other verb's aim -- charging 0 there is a silent free pan."""
    game = Game.new(_LANDSCAPE)
    player = AStarPlayer(game)
    st = game.state
    tiles = _feasible_tiles(player, st)
    assert tiles, "no landable tile to price"
    for _, view in tiles:
        _reset_stance(player, st)  # last_bearing None: nothing committed
        assert player._step_aim_frames("robot", view) == player._aim_frames(view) > 0.0
        assert player._step_aim_frames("transfer", view) == player._aim_frames(view)
        player.last_bearing = (view["h_angle"], view["v_angle"])
        assert player._step_aim_frames("transfer", view) == 0.0


def test_downlook_charge_is_faithful_not_flat():
    """A below-eye build charges the real pitched view, far above the level floor --
    the under-charge that killed (13,27)."""
    game = Game.new(_LANDSCAPE)
    player = AStarPlayer(game)
    st = game.state
    tile = (13, 27)
    _reset_stance(player, st)
    view = player._view_for(tile)
    assert view is not None and view["v_angle"] > 128  # pitched down, not level
    floor = player._settle("boulder", view)
    _reset_stance(player, st)
    cost = player._charge(st.clone(), "boulder", tile)
    assert cost > floor + 100  # pitched-aim terms dominate the flat floor


def test_transfer_settle_is_priced_from_the_post_transfer_eye():
    """The eye moves into the target ($0C63) BEFORE the $35C3/$35C6 replot passes, so
    the settle is the NEW body's scene at its OWN bearing (creator ^ $80, $1BE0) --
    not the aim view, which belongs to the abandoned eye and prices a different scene.
    """
    game = Game.new(_LANDSCAPE)
    player = AStarPlayer(game)
    st = game.state
    differed = 0
    for tile, view in _feasible_tiles(player, st, want=8):
        clone = st.clone()
        me = clone.player
        clone.obj_h_angle[me] = view["h_angle"]  # the executor aims, then creates
        clone.obj_v_angle[me] = view["v_angle"]
        if actions.create(clone, mm.T_ROBOT, tile) is None:
            continue
        top = terrain.top_object(clone, *tile)
        assert clone.obj_h_angle[top] == view["h_angle"] ^ 0x80
        player.st = clone
        eye_view = {
            "h_angle": int(clone.obj_h_angle[top]),
            "v_angle": int(clone.obj_v_angle[top]),
        }
        got = player._settle("transfer", view, player._settle_eye("transfer", tile))
        assert got == projector.viewpoint_replot_frames(clone, eye_view, top)
        if got != projector.viewpoint_replot_frames(clone, view):
            differed += 1
    assert differed, "no tile where the two eyes see different scenes"


def test_bounded_run_leaves_no_body_in_a_live_cone():
    """The audited executor never leaves a create/transfer in a live enemy cone."""
    game = Game.new(_LANDSCAPE)
    player = AStarPlayer(game, audit=True, time_budget=1.0, node_budget=400)
    player.run(max_actions=6)
    assert not player.breaches, f"body left in a live cone: {player.breaches}"
    assert mm.NUM_SLOTS  # memmap loaded


def test_macro_search_wins_landscape0_without_breaches():
    """The hierarchical macro search (directed pursuit + endgame) reaches a real
    win: every enemy absorbed, hyperspaced off the platform, no body ever left
    in a live cone.  Guards the depth collapse -- primitive-hop search never got
    here in budget."""
    game = Game.new(0)
    player = AStarPlayer(game, audit=True, time_budget=90.0)
    won = player.run(max_actions=80)
    assert won, "macro search failed to win landscape 0"
    assert not player.breaches, f"body left in a live cone: {player.breaches}"
    verbs = {rec[1] for rec in player.trace}
    assert "absorb" in verbs and "hyperspace" in verbs


def test_margin_rejects_a_step_inside_the_cost_interval():
    """A step whose predicted window clears the raw budget but not the step-cost
    interval's pessimistic end is rejected; the margin widens with plan depth
    (random walk over the measured zero-mean per-step frame error)."""
    game = Game.new(_LANDSCAPE)
    player = AStarPlayer(game)
    budget = 100.0
    player._margin_k = 1.0
    player._depth = 0
    m0 = player._margin()
    assert m0 > 0
    assert player._hot(budget, budget + m0 - 1.0)  # inside the interval: rejected
    assert not player._hot(budget, budget + m0 + 1.0)
    player._depth = 8
    assert player._margin() > m0
    assert player._margin() == pytest.approx(m0 * math.sqrt(9.0))
    player._margin_k = 0.0
    assert player._margin() == 0.0  # relaxed re-search restores the raw gate
    assert player._hot(budget, budget - 1.0)  # raw gate still binds


def test_stale_step_prefers_a_survivable_replan_over_escape_hyperspace():
    """When the next step's premise is stale, ``_restale`` re-plans, then defends,
    and never concedes an escape hyperspace itself."""
    game = Game.new(_LANDSCAPE)
    player = AStarPlayer(game, time_budget=0.01, node_budget=1)
    calls = []
    player._hyperspace = lambda: calls.append("hyperspace")
    player._wait = lambda: calls.append("wait")

    player._search = lambda margin_k=None: [("absorb", (1, 1))]
    player._defend = lambda: calls.append("defend") or True
    player._restale()
    assert player.plan == [("absorb", (1, 1))] and player._pi == 0 and not calls

    searches = []
    player._search = lambda margin_k=None: searches.append(margin_k) or None
    player._restale()
    assert calls == ["defend"] and searches == [None]  # defended before relaxing

    calls.clear()
    searches.clear()
    player._defend = lambda: False
    player._restale()
    assert searches == [None, 0.0]  # relaxed last-chance line, not a hyperspace
    assert calls == ["wait"] and "hyperspace" not in calls
