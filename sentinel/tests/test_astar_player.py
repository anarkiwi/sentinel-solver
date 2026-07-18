"""The A* planner prices each action with the executor's real aim+settle cost.

The (13,27) death (docs/player.md) was a search under-charging an action's aim, so
it planned a body onto a tile a further-rotated cone in fact sees.  A full solve
exceeds the CI budget, so this locks the cost-parity and placement guarantees.
"""

from sentinel import enemies, memmap as mm
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
    """Each search charge == executor ``_aim_frames + _settle`` over the same view,
    advances enemies by ``int(cost)``, and mirrors the post-aim stance."""
    game = Game.new(_LANDSCAPE)
    player = AStarPlayer(game)
    st = game.state
    tiles = _feasible_tiles(player, st)
    assert tiles, "no landable tile to price"
    for tile, view in tiles:
        for verb in ("boulder", "robot", "transfer", "absorb"):
            _reset_stance(player, st)
            expected = player._aim_frames(view) + player._settle(verb, view)
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
