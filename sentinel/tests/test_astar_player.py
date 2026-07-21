"""The A* planner prices each action with the executor's real aim+settle cost.

The (13,27) death (docs/player.md) was a search under-charging an action's aim, so
it planned a body onto a tile a further-rotated cone in fact sees.  A full solve
exceeds the CI budget, so this locks the cost-parity and placement guarantees.
"""

import math

import pytest

from sentinel import actions, enemies, memmap as mm, projector, terrain
from sentinel.game import Game
from sentinel.astar_player import AStarPlayer, GATE_BODY, GATE_TILE, PlanStep, _Node
from sentinel.playerbase import SIGHTS_CENTRE

_LANDSCAPE = 42  # player starts at (14,27), a down-look hollow adjacent to (13,27)
_LS42 = 66  # what typing "0042" seeds: landscape_from_digits reads the code as HEX


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
    view, advances enemies by ``int(cost)`` split at the u-turn unfreeze, and mirrors
    the post-aim stance."""
    game = Game.new(_LANDSCAPE)
    player = AStarPlayer(game)
    st = game.state
    tiles = _feasible_tiles(player, st)
    assert tiles, "no landable tile to price"
    for tile, view in tiles:
        for verb in ("boulder", "robot", "transfer", "absorb"):
            _reset_stance(player, st)
            aim_f = player._step_aim_frames(verb, view)
            expected = aim_f + player._settle(verb, view)
            split = player._aim_unfreeze_split(view)
            clone = st.clone()
            reference = st.clone()
            if split is None:
                enemies.advance_frames(reference, int(expected))
            else:
                # $12E1: keying the u-turn unfreezes the world part-way through the aim
                pre = int(min(aim_f, split))
                enemies.advance_frames(reference, pre)
                reference.mem[mm.PLAYER_NOT_ACTED] = 0x00
                enemies.advance_frames(reference, int(expected) - pre)
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


def _root(player):
    root = _Node(player.st.clone(), 0.0, (), None, player.last_bearing, player.cursor)
    root.key = player._key(root.state)
    player._deadline = math.inf
    return root


def test_climb_child_is_one_hop():
    """A climb child is ONE hop -- k boulders, a robot, the transfer up, plus the
    inchworm recycle of the stack it just left -- not a chain to an enemy.  Depth
    therefore counts hops, and a climb child never absorbs an enemy (``_c_absorb``
    owns the terminal strike)."""
    player = AStarPlayer(Game.new(_LS42), time_budget=30.0)
    root = _root(player)
    start_eye = root.state.eye_z()
    climbs = [c for c in player._expand(root) if c.state.eye_z() > start_eye]
    assert climbs, "no child raises the eye"
    for c in climbs:
        verbs = [s.verb for s in c.path]
        assert verbs.count("transfer") == 1  # exactly one hop, not a chain
        assert verbs.index("transfer") == len(
            [v for v in verbs if v in ("boulder", "robot")]
        )  # builds, then the transfer; anything after is the recycle
        assert all(v == "absorb" for v in verbs[verbs.index("transfer") + 1 :])
    assert not any(actions.won(c.state) for c in climbs)  # partial, not a solve


def test_expansion_branches_over_hop_tiles():
    """Each ranked ``_pick_hop`` candidate becomes its own child, so a node offers
    several DISTINCT stances the search can arbitrate between with ``g`` -- the greedy
    chain collapsed them to one rollout per enemy."""
    player = AStarPlayer(Game.new(_LS42), time_budget=30.0)
    root = _root(player)
    cands = player._hop_candidates(root)
    assert len(cands) > 1, "one hop candidate: the expansion cannot compare stances"
    assert len({(t, k) for t, k, _w in cands}) == len(cands)  # deduped across targets
    kids = [player._c_hop(root, t, k, w) for t, k, w in cands]
    kids = [c for c in kids if c is not None]
    assert len({c.key for c in kids}) > 1, "children collapse to one frontier key"


def test_macro_search_wins_ls42_internal_66():
    """The board the live driver plays when `0042` is typed (hex -> internal 66).  It is
    NOT ``_LANDSCAPE`` (42), which is a different board with no slot overlap."""
    game = Game.new(_LS42)
    # generous budget on purpose: the search's ONLY nondeterminism is its wall-clock
    # deadline, and this is a "does the planner find the line" assertion, not a
    # "in N seconds" one -- it wins in ~25 s idle, several times that under -n auto.
    player = AStarPlayer(game, audit=True, time_budget=600.0)
    won = player.run(max_actions=80)
    assert won, f"lost ls42 in {len(player.trace)} actions, energy {game.energy}"
    assert not player.breaches, f"body left in a live cone: {player.breaches}"


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

    found = [PlanStep("absorb", (1, 1), 300.0, GATE_BODY, math.inf, math.inf)]
    player._search = lambda margin_k=None: found
    player._defend = lambda: calls.append("defend") or True
    player._restale()
    assert player.plan == found and player._pi == 0 and not calls

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


def test_react_takes_the_plans_own_transfer_before_conceding_a_hyperspace():
    """Hot, with the plan's next step being the transfer off this tile: the pedestal the
    pursuit just built IS the escape, so ``_react`` takes it and KEEPS the plan.  Ranking
    escapes by window alone rejects it (the pedestal's window is no wider than here) and
    the ladder falls through to a hyperspace -- one keystroke short of the climb, the
    ls42 live loss."""
    player = AStarPlayer(Game.new(_LS42), time_budget=0.01, node_budget=1)
    player.plan = [
        PlanStep("transfer", (9, 26), 300.0, GATE_TILE, math.inf, math.inf),
        PlanStep("absorb", (13, 29), 300.0, GATE_BODY, math.inf, math.inf),
    ]
    player._pi = 0
    player._player_window = lambda exclude=None: 0.0  # hot: react must deviate
    player._defend = lambda: False  # no counterattack, no window-ranked escape
    player._view_for = lambda tile: {
        "h_angle": 0x60,
        "v_angle": 0x35,
        "cursor": [80, 95],
    }
    fired, hs = [], []
    player._fire = lambda verb, tile, view: fired.append((verb, tuple(tile))) or True
    player._hyperspace = lambda: hs.append(1)
    assert player._react() is True
    assert fired == [("transfer", (9, 26))] and not hs
    assert player.plan is not None and player._pi == 1
    assert player._on_plan  # _tick must NOT throw the plan away
