"""``StancePlayer`` only ADDS the route child, so it is pinned as a strict superset.

With a graph that can produce no route its expansion is ``AStarPlayer``'s exactly; the
route child itself compiles hop by hop through the ranked hop's own gates, truncating at
the first hop the live board refuses.
"""

import math

import numpy as np
import pytest

from sentinel import actions, enemies, memmap as mm
from sentinel.astar_player import (
    AStarPlayer,
    GATE_TILE,
    PlanStep,
    _ABSORB_EST,
    _ENDGAME_EST,
    _HOP_EST,
    _Node,
)
from sentinel.game import Game
from sentinel.stancegraph import Route, StanceGraph, build_frames, flat_tiles
from sentinel.stance_player import ROUTE_TARGETS, StancePlayer, main

_LS = 335  # 7 enemies: more pursue targets than ROUTE_TARGETS, so the cap bites
_NCELL = mm.N * mm.N


def _empty_graph():
    """A graph with no stance at all: every ``route`` call answers None."""
    return StanceGraph(
        [],
        np.zeros((0, _NCELL), dtype=bool),
        np.zeros(0, dtype=np.int32),
        np.zeros(0, dtype=np.int32),
        np.zeros(0, dtype=np.float32),
        1,
        "empty",
    )


def _one_stance_graph(tile, eye, strikes):
    """A single stance at ``eye`` that strikes either every tile on the board or none."""
    return StanceGraph(
        [(tile, 1)],
        np.full((1, _NCELL), strikes, dtype=bool),
        np.zeros(1, dtype=np.int32),
        np.zeros(1, dtype=np.int32),
        np.array([eye], dtype=np.float32),
        1,
        "one",
    )


def _stub_graph(tiles):
    """Stances on real tiles that see nothing, at rising eyes: hops the live board judges."""
    n = len(tiles)
    return StanceGraph(
        [(t, i) for i, t in enumerate(tiles)],
        np.zeros((n, _NCELL), dtype=bool),
        np.zeros(n, dtype=np.int32),
        np.zeros(n, dtype=np.int32),
        np.arange(1.0, n + 1.0, dtype=np.float32),
        1,
        "stub",
    )


def _small_graph(state, want=8):
    """A real but partial sweep over the flat tiles nearest the body."""
    px, py = state.player_xy()
    tiles = sorted(flat_tiles(state), key=lambda t: (t[0] - px) ** 2 + (t[1] - py) ** 2)
    return StanceGraph.build(state, kmax=1, tiles=tiles[:want])


def _root(player):
    """The search's own start node, as ``_search`` seeds it."""
    node = _Node(player.st.clone(), 0.0, (), None, player.last_bearing, player.cursor)
    node.key = player._key(node.state)
    player._deadline = math.inf
    return node


def _shape(children):
    return [(round(c.g, 6), tuple((s.verb, s.tile) for s in c.path)) for c in children]


def test_expansion_is_a_strict_superset_of_the_astar_one():
    """With no route available the route child adds nothing, so the children match."""
    base = AStarPlayer(Game.typed(_LS))
    want = _shape(base._expand(_root(base)))
    player = StancePlayer(Game.typed(_LS), graph=_empty_graph())
    node = _root(player)
    assert player._route_targets(node.state)  # the generator did run
    assert _shape(player._expand(node)) == want
    assert want


def test_route_targets_are_the_living_unlandable_enemies_capped():
    """Sentinel alive: at most ``ROUTE_TARGETS`` living enemies, none of them landable."""
    player = StancePlayer(Game.typed(_LS), graph=_empty_graph())
    st = player.st
    targets = player._route_targets(st)
    assert 0 < len(targets) == ROUTE_TARGETS < len(player._pursue_targets(st)) + 1
    foes = {tuple(st.tile_of(e)) for e in enemies.enemy_slots(st)}
    for tile in targets:
        assert tile in foes
        assert not player._landable(st, tile)
    assert len(set(targets)) == len(targets)


def test_route_targets_are_the_platform_once_the_sentinel_is_gone():
    """Sentinel slot empty: route to the platform, and only while it is out of reach."""
    player = StancePlayer(Game.typed(_LS), graph=_empty_graph())
    st = player.st.clone()
    actions.remove_object(st, actions.SENTINEL_SLOT)
    assert st.is_empty(actions.SENTINEL_SLOT)
    player._landable = lambda state, tile: False
    assert player._route_targets(st) == [tuple(st.platform_xy)]
    player._landable = lambda state, tile: True
    assert not player._route_targets(st)


def test_route_mask_is_the_landset():
    """The mask the route starts from is the player's own memoized landset, bit for bit."""
    player = StancePlayer(Game.typed(_LS), graph=_empty_graph())
    st = player.st
    mask = player._route_mask(st)
    assert mask.shape == (_NCELL,) and mask.dtype == np.bool_
    got = {(int(f) // mm.N, int(f) % mm.N) for f in np.flatnonzero(mask)}
    assert got == set(player._landset(st))
    assert player._route_mask(st) is mask  # memoized per stance signature


def test_route_child_compiles_the_graphs_own_hops():
    """A real partial graph yields a route child whose steps build the routed stance."""
    game = Game.typed(_LS)
    graph = _small_graph(game.state)
    player = StancePlayer(game, graph=graph)
    node = _root(player)
    order = np.argsort(-graph.seen.sum(axis=0))[:8].tolist()
    child = target = route = None
    for flat in order:
        target = (flat // mm.N, flat % mm.N)
        route = player._route_for(node.state, target)
        child = player._c_route(node, target)
        if child is not None:
            break
    assert child is not None, "no routed target compiled a single hop"
    assert child.path and all(isinstance(s, PlanStep) for s in child.path)
    assert child.g > node.g
    assert child.path[0].tile == graph.stances[route.path[0]][0]
    assert any(s.verb == "transfer" for s in child.path)  # the hop completed
    assert child.path[-1].verb in ("transfer", "absorb")  # _harvest may bank fuel after
    assert child.state.eye_z() > node.state.eye_z()


def test_route_child_is_none_without_a_route():
    """No strike stance in the partial graph: no route, and so no child."""
    game = Game.typed(_LS)
    player = StancePlayer(game, graph=_small_graph(game.state, want=4))
    node = _root(player)
    unseen = [f for f in range(_NCELL) if not player.graph.seen[:, f].any()]
    assert unseen
    target = (unseen[0] // mm.N, unseen[0] % mm.N)
    assert player._route_for(node.state, target) is None
    assert player._c_route(node, target) is None


def test_a_refused_hop_truncates_the_route_child():
    """A hop the live board no longer permits ends the child at the prefix it made.

    The graph is a SNAPSHOT, so a stale hop must fail to compile, never raise."""
    game = Game.typed(_LS)
    player = StancePlayer(game, graph=_empty_graph())
    node = _root(player)
    tiles = flat_tiles(node.state)[:2]
    player._graph = _stub_graph(tiles)
    target = tiles[1]
    player._route_for = lambda st, tgt, blocked=(): Route([0, 1], 9, tgt)
    player._repair = (
        lambda st, tgt, blocked, node_i, tile, k: None
    )  # no re-route: pin truncation
    player._landable = lambda st, tile: False
    player._hop_funds = lambda st, cost=None: (True, 0.0)
    player._hop_ok = lambda st, tile, k, wait_ok=True: (900.0, 0.0)
    player._harvest = (
        lambda st, steps: 0.0
    )  # isolate the hop chain from the fuel top-up
    made = []
    step = PlanStep("robot", tiles[0], 7.0, GATE_TILE, 900.0, math.inf)
    player._hop_exec = lambda tile, k, window: (
        None if made else (made.append(tile) or (7.0, [step]))
    )
    child = player._c_route(node, target)
    assert made == [tiles[0]]
    assert child is not None and list(child.path) == [step]
    assert child.g == node.g + 7.0

    player._hop_ok = lambda st, tile, k, wait_ok=True: None
    assert player._c_route(node, target) is None  # refused at the first hop: no child


def test_a_landing_that_cannot_be_left_is_refused_and_re_routed():
    """A hop the board permits but whose landing traps the body is not taken.

    ls335's route opens onto (10,17) k=3 -- one full-sight seer for a body on the bare
    tile, four for one on three boulders, which then hold it -- and the executor must
    hand that verdict to ``_repair`` rather than land and die there."""
    game = Game.typed(_LS)
    player = StancePlayer(game, graph=_empty_graph())
    node = _root(player)
    tiles = flat_tiles(node.state)[:2]
    player._graph = _stub_graph(tiles)
    player._route_for = lambda st, tgt, blocked=(): Route([0, 1], 9, tgt)
    player._landable = lambda st, tile: False
    player._hop_funds = lambda st, cost=None: (True, 0.0)
    player._hop_ok = lambda st, tile, k, wait_ok=True: (900.0, 0.0)
    player._harvest = lambda st, steps: 0.0
    step = PlanStep("robot", tiles[0], 7.0, GATE_TILE, 900.0, math.inf)
    player._hop_exec = lambda tile, k, window: (7.0, [step])
    player._can_leave = lambda st, target: False
    blocked = []
    player._repair = lambda st, tgt, seen, node_i, tile, k: blocked.append(node_i)
    assert player._c_route(node, tiles[1]) is None  # nothing committed
    assert blocked == [0]  # and the trap was handed to the router to route around


def test_repair_both_vetoes_the_stance_and_keeps_what_it_measured():
    """The ban is for THIS query; the measured price is for every later one.

    ``build_frames`` knows only k, so the router keeps proposing aim-expensive landings.
    Vetoing alone throws away the number that would stop it; re-pricing alone never
    removes the stance just refused, because the measurement always beats the floor."""
    game = Game.typed(_LS)
    player = StancePlayer(game, graph=_empty_graph())
    tiles = flat_tiles(player.st.clone())[:2]
    player._graph = _stub_graph(tiles)
    seen = []
    player._route_for = lambda st, tgt, blocked=(): seen.append(blocked) or None
    floor = build_frames(1)

    player._hop_price = lambda st, tile, k: (floor * 3, 0.0)  # dearer than the floor
    assert player._repair(player.st, tiles[1], set(), 0, tiles[0], 1) is None
    assert player._edge_frames[0] == floor * 3  # re-priced...
    assert seen[-1] == frozenset({0})  # ...AND out of this query

    player._hop_price = lambda st, tile, k: (floor, 0.0)  # a cheaper later reading
    player._repair(player.st, tiles[1], set(), 0, tiles[0], 1)
    assert player._edge_frames[0] == floor * 3  # weights only ever rise

    player._hop_price = lambda st, tile, k: None  # cannot be built at all
    assert player._repair(player.st, tiles[1], set(), 1, tiles[0], 1) is None
    assert 1 not in player._edge_frames  # nothing to learn, but still vetoed
    assert seen[-1] == frozenset({1})


def test_the_graph_is_built_once_from_the_board_captured_at_construction(
    tmp_path, monkeypatch
):
    """The lazy property sweeps once, off the constructor's snapshot, into ``cache_dir``."""
    graph = _empty_graph()
    calls = []

    def fake_cached(_cls, state, _kmax=None, _log=None, cache_dir=None):
        calls.append((state, cache_dir))
        return graph

    monkeypatch.setattr(StanceGraph, "cached", classmethod(fake_cached))
    player = StancePlayer(Game.typed(_LS), cache_dir=str(tmp_path))
    assert player.graph is graph and player.graph is graph
    assert calls == [(player._graph_state, str(tmp_path))]
    assert player._graph_state is not player.st


def test_hop_ok_schedules_a_hop_rather_than_refusing_it():
    """An unaimable, unschedulable or unaffordable hop answers None; a late one WAITS.

    A cone that cannot cover the hop NOW is a delay.  A sweep that never covers it is a
    refusal, not a price: the tile window bounds how long the STACK survives, and a
    sentry that sees the boulder absorbs it and leaves a tree."""
    player = StancePlayer(Game.typed(_LS), graph=_empty_graph())
    st = player.st.clone()
    tile = flat_tiles(st)[0]
    player._exposing_enemies = lambda t: ()
    player._gaze_window = lambda t, exposed=None: 900.0
    player._drain_gate = lambda verb, t, exposed=None, budget=0.0: True
    player._hop_price = lambda state, t, k: None
    assert player._hop_ok(st, tile, 1) is None  # the stack cannot be built or aimed
    player._hop_price = lambda state, t, k: (10.0, 5000.0)
    player._earliest_start = lambda t, need, exposed=None: math.inf
    player._affords = lambda cost, budget, window=None: True
    assert player._hop_ok(st, tile, 1) is None  # no phase holds the stack: never built
    del player._earliest_start, player._affords
    player._hop_price = lambda state, t, k: (10.0, 5.0)
    player._affords = lambda cost, budget, window=None: False
    assert player._hop_ok(st, tile, 1) is None  # the source side cannot fund it
    player._affords = lambda cost, budget, window=None: True
    assert player._hop_ok(st, tile, 1) == (900.0, 0.0)  # clear now: no wait needed

    player._earliest_start = lambda t, need, exposed=None: 773.0
    player._drain_gate = lambda verb, t, exposed=None, budget=0.0: False
    window, wait = player._hop_ok(st, tile, 1)
    assert wait == 773.0 and window == 5.0 + player._margin()


def test_a_route_child_reclaims_when_it_cannot_fund_the_next_hop():
    """An unaffordable hop takes the inchworm reclaim, and ends when the reclaim dries up."""
    player = StancePlayer(Game.typed(_LS), graph=_empty_graph())
    node = _root(player)
    tiles = flat_tiles(node.state)[:2]
    player._graph = _stub_graph(tiles)
    step = PlanStep("absorb", tiles[0], 5.0, GATE_TILE, 900.0, math.inf)
    player._route_for = lambda st, tgt, blocked=(): Route([0, 1], 9, tgt)
    player._landable = lambda st, tile: False
    player._hop_funds = lambda st, cost=None: (False, 0.0)
    left = [(4.0, step)]
    player._reclaim_one = lambda st: left.pop() if left else None
    child = player._c_route(node, tiles[1])
    assert child is not None and list(child.path) == [step]
    assert child.g == node.g + 4.0
    assert not left


def test_a_route_child_stops_once_the_target_is_already_landable():
    """Nothing to climb for: the loop breaks before the first hop, so there is no child."""
    player = StancePlayer(Game.typed(_LS), graph=_empty_graph())
    node = _root(player)
    player._route_for = lambda st, tgt, blocked=(): Route([0], 9, tgt)
    player._landable = lambda st, tile: True
    player._hop_funds = lambda st, cost=None: pytest.fail(
        "a landable target must not be climbed to"
    )
    assert player._c_route(node, (3, 4)) is None


def test_routes_are_a_fallback_not_an_extra_child():
    """A node with ranked children keeps them alone: an extra child surfaces a worse goal.

    A weighted search returns the first goal popped, so offering routes unconditionally
    cost ls42 2 actions and ls110 6 on boards the ranked climb already solves."""
    player = StancePlayer(Game.typed(_LS), graph=_empty_graph())
    node = _root(player)
    base = _shape(player._expand(node))
    assert base  # the ranked generators do produce children here
    marker = _Node(node.state, 0.0, (), None, player.last_bearing, player.cursor)
    player._c_route = lambda n, target: marker
    assert _shape(player._expand(node)) == base
    assert marker not in player._expand(node)


def test_routes_are_offered_when_the_node_has_none_and_when_forced(monkeypatch):
    """Routes append exactly where the frontier would otherwise die, or on demand."""
    player = StancePlayer(Game.typed(_LS), graph=_empty_graph())
    node = _root(player)
    base = _shape(player._expand(node))
    marker = _Node(node.state, 0.0, (), None, player.last_bearing, player.cursor)
    player._c_route = lambda n, target: marker
    monkeypatch.setattr(AStarPlayer, "_expand", lambda self, n: [])
    got = player._expand(node)
    assert len(got) == ROUTE_TARGETS and all(c is marker for c in got)
    monkeypatch.undo()

    forced = StancePlayer(Game.typed(_LS), graph=_empty_graph(), route_policy="always")
    forced._c_route = lambda n, target: marker
    out = forced._expand(_root(forced))
    assert len(out) == len(base) + ROUTE_TARGETS
    assert _shape(out[: len(base)]) == base


def test_order_for_is_memoized_per_board_and_live_enemy_set():
    """One DP per (stance signature, living enemies); absorbing one re-solves the order."""
    player = StancePlayer(Game.typed(_LS), graph=_empty_graph())
    st = player.st.clone()
    player._graph = _stub_graph(flat_tiles(st)[:3])
    got = player._order_for(st)
    assert (
        player._order_for(st) is got
    )  # the DP builds a fresh list, so identity pins it
    key = (player._sig(st), tuple(sorted(enemies.enemy_slots(st))))
    assert list(player._order_memo) == [key]
    fewer = st.clone()
    actions.remove_object(fewer, max(enemies.enemy_slots(fewer)))
    assert player._order_for(fewer) is not got
    assert len(player._order_memo) == 2


def test_route_targets_come_from_the_ordering_dp_once_the_graph_has_stances():
    """A blind graph bridges nothing, so the DP's forced tail IS the order it returns.

    The Sentinel STANDS on its platform, so the one locked tile holds both goals back
    until it is the last enemy alive, and the order dedupes to a single route."""
    player = StancePlayer(Game.typed(_LS), graph=_empty_graph())
    st = player.st
    player._graph = _stub_graph(flat_tiles(st)[:3])
    plat = tuple(st.platform_xy)
    assert tuple(int(v) for v in st.tile_of(actions.SENTINEL_SLOT)) == plat
    assert player._order_for(st) == [plat, plat]
    assert not player._route_targets(st)
    lone = st.clone()
    for slot in enemies.enemy_slots(lone):
        if slot != actions.SENTINEL_SLOT:
            actions.remove_object(lone, slot)
    player._landable = lambda state, tile: False
    assert player._route_targets(lone) == [plat]


def test_route_targets_skip_the_landable_and_stop_at_the_cap():
    """The DP's order, minus anything already reachable and any repeat, capped."""
    player = StancePlayer(Game.typed(_LS), graph=_empty_graph())
    st = player.st
    player._graph = _stub_graph(flat_tiles(st)[:3])
    goals = [(1, 1), (2, 2), (1, 1), (3, 3), (4, 4), (5, 5)]
    player._order_for = lambda state: goals
    player._landable = lambda state, tile: tile == (2, 2)
    assert player._route_targets(st) == [(1, 1), (3, 3), (4, 4)]
    assert ROUTE_TARGETS == 3


def test_route_targets_hold_the_platform_until_the_sentinel_stands_alone():
    """``$1B8E``: the platform is not a goal while more than one enemy still lives."""
    player = StancePlayer(Game.typed(_LS), graph=_empty_graph())
    st = player.st
    player._graph = _stub_graph(flat_tiles(st)[:3])
    plat = tuple(st.platform_xy)
    player._order_for = lambda state: [(1, 1), plat]
    player._landable = lambda state, tile: False
    assert player._route_targets(st) == [(1, 1)]
    lone = st.clone()
    for slot in enemies.enemy_slots(lone):
        if slot != actions.SENTINEL_SLOT:
            actions.remove_object(lone, slot)
    assert player._route_targets(lone) == [(1, 1), plat]


def test_graph_hops_answers_only_when_the_graph_reaches_a_striker():
    """No target means zero hops only when the goals really are landable, else no answer.

    "All goals in reach" and "the DP bridged nothing" both empty ``_route_targets``, and
    scoring the second as 0.0 made ``h`` optimistic exactly where the search is stuck.
    """
    player = StancePlayer(Game.typed(_LS), graph=_empty_graph())
    st = player.st
    tile = tuple(sorted(player._landset(st))[0])
    high, low = st.eye_z() + 1.0, st.eye_z() - 1.0
    player._graph = _one_stance_graph(tile, high, True)
    player._route_targets = lambda state: []
    assert not player._goals_landable(st)  # 7 live enemies, none of them landable
    assert player._graph_hops(st) is None
    player._goals_landable = lambda state: True
    assert player._graph_hops(st) == 0.0
    del player._goals_landable
    player._route_targets = lambda state: [(3, 4), (5, 6)]
    assert player._graph_hops(st) == 1.0  # one hop onto a stance that strikes both
    player._graph = _one_stance_graph(tile, low, True)
    assert player._graph_hops(st) is None  # nothing higher is landable from here
    player._graph = _one_stance_graph(tile, high, False)
    assert player._graph_hops(st) is None  # what it reaches strikes nothing
    player._graph = _empty_graph()
    assert player._graph_hops(st) is None


def test_h_prices_the_graphs_hops_and_falls_back_without_them(monkeypatch):
    """The graph's hop count replaces the eye deficit; a won board is still zero."""
    player = StancePlayer(Game.typed(_LS), graph=_empty_graph())
    st = player.st
    assert player._h(st) == AStarPlayer._h(player, st)  # empty graph: no answer
    player._graph_hops = lambda state: 2.0
    want = len(enemies.enemy_slots(st)) * _ABSORB_EST + 2.0 * _HOP_EST + _ENDGAME_EST
    assert player._h(st) == want
    monkeypatch.setattr(actions, "won", lambda state: True)
    assert player._h(st) == 0.0


def test_main_runs_a_budgeted_search_and_reports_the_result(
    tmp_path, monkeypatch, capsys
):
    """The CLI drives the player over a graph offering no route, and exits on the loss."""
    monkeypatch.setattr(
        StanceGraph, "cached", classmethod(lambda cls, *a, **kw: _empty_graph())
    )
    argv = [
        str(_LS),
        "--quiet",
        "--max-actions",
        "2",
        "--node-budget",
        "1",
        "--time-budget",
        "2",
        "--cache-dir",
        str(tmp_path),
    ]
    assert main(argv) == 1
    assert f"landscape {_LS}: lost" in capsys.readouterr().out
    assert not list(tmp_path.iterdir())
