"""The stance graph is exact geometry, so it is pinned against independent derivations.

``strike_stances`` against the transpose of the sweep, ``route`` against a brute-force
BFS over the same energy rules, and one real board's swept rows against
``landtable.landable_set`` itself.
"""

import numpy as np
import pytest

from sentinel import actions, landtable, memmap as mm
from sentinel.game import Game
from sentinel.stancegraph import (
    KMAX,
    Route,
    StanceGraph,
    _probe,
    flat_tiles,
    hop_energy,
    live_seen,
    main,
    route_to,
    signature,
    stance_eye,
)

_LS = 335  # the board with the 13-of-1192 Sentinel strike set
_CAP = int(mm.ENERGY_MASK)
_NCELL = mm.N * mm.N
_TREE_GAIN = mm.ENERGY_IN_OBJECTS[mm.T_TREE]


def _synth(seed=0, n=12, kmax=2):
    """A hand-built graph: random landability over distinct tiles, distinct rising eyes.

    Sweeping a real board costs ~40 s, so every algorithmic invariant is pinned here
    and only the geometry itself is checked against a real sweep."""
    rng = np.random.default_rng(seed)
    stances = [((i, 0), int(i % (kmax + 1))) for i in range(n)]
    seen = rng.random((n, _NCELL)) < 0.02
    cols = np.array([x * mm.N + y for (x, y), _k in stances], dtype=np.int32)
    seen[:, cols] = rng.random((n, n)) < 0.45
    eye = np.arange(n, dtype=np.float32) * 0.5 + rng.random(n).astype(np.float32) * 0.01
    return StanceGraph(
        stances,
        seen,
        rng.integers(0, 3, n).astype(np.int32),
        rng.integers(0, 12, n).astype(np.int32),
        eye,
        kmax,
        f"synth{seed}",
    )


def _tree_tile(state):
    """A tile carrying a tree: nothing stacks on it, so no probe stands there."""
    slot = state.slot_of_type(mm.T_TREE)
    assert slot is not None
    return tuple(int(v) for v in state.tile_of(slot))


def _reachable_graph(state):
    """One stance, on a flat tile the live body can already land, that strikes every tile.

    The cheapest graph a whole ``route_to``/``main`` run can be driven through without
    sweeping a board."""
    flat = set(flat_tiles(state))
    tile = next(
        t
        for t in (
            (int(f) // mm.N, int(f) % mm.N) for f in np.flatnonzero(live_seen(state))
        )
        if t in flat
    )
    return StanceGraph(
        [(tile, 1)],
        np.ones((1, _NCELL), dtype=bool),
        np.zeros(1, dtype=np.int32),
        np.zeros(1, dtype=np.int32),
        np.array([state.eye_z() + 1.0], dtype=np.float32),
        1,
        "reachable",
    )


def _bfs_hops(graph, target, energy, start, monotone=True):
    """Fewest hops to a strike stance, by exhaustive layered BFS over ``(stance, energy)``.

    Independent of ``route``: no domination pruning, no parent chain -- just every
    reachable state, layer by layer, so the first layer holding a goal IS the optimum.
    """
    goals = set(graph.strike_stances(target).tolist())
    if not goals:
        return None
    spend = graph._spend()
    gain = graph.fuel * _TREE_GAIN
    frontier = {(start, min(_CAP, int(energy)))}
    reached = set(frontier)
    hops = 0
    while frontier:
        if any(node in goals for node, _e in frontier):
            return hops
        nxt = set()
        for node, have in frontier:
            here, here_k = graph.stances[node]
            row = graph.adj[node] if monotone else graph.seen[node, graph._cols]
            for j in np.flatnonzero(row).tolist():
                cost = int(spend[j])
                if have < cost:
                    continue
                got = have - cost + int(gain[j])
                if graph.lands(j, here):
                    got += hop_energy(here_k)
                state = (j, min(_CAP, got))
                if state not in reached:
                    reached.add(state)
                    nxt.add(state)
        frontier = nxt
        hops += 1
    return None


def _bfs_hops_to(graph, tile, monotone=True):
    """Hops from every stance to the strike set, by a naive forward BFS per stance."""
    edges = graph.adj if monotone else graph.seen[:, graph._cols]
    goals = set(graph.strike_stances(tile).tolist())
    want = np.full(len(graph), -1, dtype=np.int32)
    for i in range(len(graph)):
        frontier, reached, depth = [i], {i}, 0
        while frontier:
            if goals & set(frontier):
                want[i] = depth
                break
            nxt = [
                j
                for f in frontier
                for j in np.flatnonzero(edges[f]).tolist()
                if j not in reached
            ]
            reached.update(nxt)
            frontier, depth = nxt, depth + 1
    return want


def _replay(graph, route, start, energy):
    """Walk ``route.path`` from ``start``, mirroring ``route``'s own accounting."""
    spend = graph._spend()
    gain = graph.fuel * _TREE_GAIN
    have = min(_CAP, int(energy))
    prev = start
    tile, k = graph.stances[start]
    eye = float(graph.eye[start])
    for j in route.path:
        assert graph.lands(prev, graph.stances[j][0])
        assert float(graph.eye[j]) > eye
        have -= int(spend[j])
        assert have >= 0
        have += int(gain[j])
        if graph.lands(j, tile):
            have += hop_energy(k)
        have = min(_CAP, have)
        prev, (tile, k), eye = j, graph.stances[j], float(graph.eye[j])
    assert have == route.energy
    assert graph.lands(prev, route.goal)  # the arrival is a strike stance


def _pick_target(graph, start):
    """A target with a strike stance that is not ``start`` itself."""
    for j in range(len(graph)):
        tile = graph.stances[j][0]
        strikers = set(graph.strike_stances(tile).tolist())
        if strikers and start not in strikers:
            return tile
    raise AssertionError("no reachable target in the synthetic graph")


def test_strike_stances_is_the_transpose_of_the_sweep():
    """The column read inverts the rows exactly: no tile gains or loses a striker."""
    graph = _synth()
    rows = [
        {(f // mm.N, f % mm.N) for f in np.flatnonzero(graph.seen[i])}
        for i in range(len(graph))
    ]
    for flat in range(_NCELL):
        tile = (flat // mm.N, flat % mm.N)
        want = [i for i, row in enumerate(rows) if tile in row]
        assert graph.strike_stances(tile).tolist() == want
        for i, row in enumerate(rows):
            assert graph.lands(i, tile) == (tile in row)


def test_adj_is_landability_and_a_strictly_rising_eye():
    """``adj[i, j]`` is exactly "i lands j's tile and j is higher", which makes it a DAG."""
    graph = _synth()
    adj = graph.adj
    for i in range(len(graph)):
        for j in range(len(graph)):
            want = graph.lands(i, graph.stances[j][0]) and graph.eye[j] > graph.eye[i]
            assert bool(adj[i, j]) == want, (i, j)
    order = np.argsort(graph.eye, kind="stable")
    assert not np.tril(adj[np.ix_(order, order)]).any()  # acyclic: eye-ordered upper


def test_route_takes_the_fewest_hops_a_brute_force_search_can_find():
    """``len(route.path)`` equals the BFS optimum, and the walk replays feasibly.

    A parent chain keyed by ``(node, energy)`` alone lets a longer push overwrite a
    shorter one's, so ``_unwind`` walks further than Dijkstra settled."""
    checked = 0
    for seed in range(6):
        graph = _synth(seed=seed)
        for energy in (0, 6, 12, 30, _CAP):
            for j in range(len(graph)):
                target = graph.stances[j][0]
                for start in (0, 1, 2):
                    got = graph.route(target, energy, start=start)
                    want = _bfs_hops(graph, target, energy, start)
                    assert (got is None) == (want is None), (seed, energy, j, start)
                    if got is None:
                        continue
                    assert len(got.path) == len(got) == want
                    assert got.goal == target
                    _replay(graph, got, start, energy)
                    checked += 1
    assert checked > 500


def test_free_climb_is_never_longer_than_the_monotone_one():
    """``monotone=False`` drops the eye half, so it can only match or beat the DAG route."""
    for seed in range(4):
        graph = _synth(seed=seed)
        for energy in (6, 20, _CAP):
            for j in range(len(graph)):
                target = graph.stances[j][0]
                free = graph.route(target, energy, start=0, monotone=False)
                want = _bfs_hops(graph, target, energy, 0, monotone=False)
                assert (free is None) == (want is None)
                if free is not None:
                    assert len(free.path) == want
                tight = graph.route(target, energy, start=0)
                if tight is not None:
                    assert free is not None and len(free) <= len(tight)


def test_a_live_body_start_matches_the_same_stance_by_index():
    """``start_seen``/``start_eye`` route the live body exactly as ``start=`` routes it."""
    graph = _synth()
    start = 0
    for energy in (6, 20, _CAP):
        for j in range(len(graph)):
            target = graph.stances[j][0]
            if start in set(graph.strike_stances(target).tolist()):
                continue  # start= returns the empty route; the virtual origin cannot
            indexed = graph.route(target, energy, start=start)
            live = graph.route(
                target,
                energy,
                start_seen=graph.seen[start],
                start_tile=graph.stances[start][0],
                start_k=graph.stances[start][1],
                start_eye=float(graph.eye[start]),
            )
            assert (indexed is None) == (live is None)
            if indexed is not None:
                assert live.path == indexed.path and live.energy == indexed.energy


def test_route_is_none_without_a_striker_or_without_the_energy_to_start():
    """No stance sees the target, or none of the first hops is affordable."""
    graph = _synth()
    unseen = [f for f in range(_NCELL) if not graph.seen[:, f].any()]
    assert unseen
    tile = (unseen[0] // mm.N, unseen[0] % mm.N)
    assert graph.strike_stances(tile).size == 0
    assert graph.route(tile, _CAP, start=0) is None
    target = _pick_target(graph, 0)
    assert graph.route(target, _CAP, start=0) is not None
    assert graph.route(target, 0, start=0) is None  # every hop spends at least a robot


def test_route_demands_a_start():
    """Neither a stance index nor a live body: the call is unanswerable, not empty."""
    graph = _synth()
    target = _pick_target(graph, 0)
    with pytest.raises(ValueError):
        graph.route(target, _CAP)
    with pytest.raises(ValueError):
        graph.route(target, _CAP, start_seen=graph.seen[0])
    with pytest.raises(ValueError):
        graph.route(target, _CAP, start_eye=0.0)


def test_save_load_round_trips_every_field(tmp_path):
    """``seen`` is bit-packed on the way out; nothing may be lost on the way back."""
    graph = _synth(seed=3)
    path = str(tmp_path / "graph.npz")
    assert graph.save(path) == path
    got = StanceGraph.load(path)
    assert got.stances == graph.stances
    assert got.seen.dtype == np.bool_ and np.array_equal(got.seen, graph.seen)
    assert np.array_equal(got.fuel, graph.fuel)
    assert np.array_equal(got.hot, graph.hot)
    assert np.array_equal(got.eye, graph.eye)
    assert got.kmax == graph.kmax and got.sig == graph.sig
    assert np.array_equal(got.adj, graph.adj)


def test_cached_sweeps_once_and_reloads_from_disk(tmp_path, monkeypatch, new_state):
    """A second call on an unchanged tile map must not re-enter the sweep at all."""
    state = new_state(0)
    graph = _synth(seed=5)
    graph.sig = signature(state)
    calls = []

    def fake_build(cls, _st, _kmax=KMAX, _log=None, _tiles=None):
        calls.append(cls)
        return graph

    monkeypatch.setattr(StanceGraph, "build", classmethod(fake_build))
    first = StanceGraph.cached(state, cache_dir=str(tmp_path))
    assert len(calls) == 1
    assert [p.name for p in tmp_path.iterdir()] == [f"{signature(state)}.npz"]

    def boom(cls, *_a, **_kw):
        raise AssertionError(f"{cls.__name__} re-swept a board already on disk")

    monkeypatch.setattr(StanceGraph, "build", classmethod(boom))
    second = StanceGraph.cached(state, cache_dir=str(tmp_path))
    assert second.stances == first.stances
    assert np.array_equal(second.seen, first.seen)
    assert second.sig == first.sig and second.kmax == first.kmax


def test_signature_is_a_pure_function_of_the_tile_map(new_state):
    """Clones of a board agree; one created object, and the cache key must move."""
    state = new_state(0)
    assert signature(state) == signature(state.clone())
    assert signature(state, kmax=1) != signature(state, kmax=2)
    built = state.clone()
    built.energy = 60
    assert actions.create(built, mm.T_BOULDER, flat_tiles(built)[0]) is not None
    assert signature(built) != signature(state)


def test_swept_rows_are_landtables_own_geometry(new_state):
    """Each swept row IS ``landable_set`` on that probe stance, not a re-derivation.

    ``tiles=`` keeps the sweep to a handful of stances; the whole board costs ~40 s."""
    state = new_state(_LS)
    tiles = flat_tiles(state)[:8]
    graph = StanceGraph.build(state, kmax=1, tiles=tiles)
    assert len(graph) == 2 * len(tiles)
    grids = landtable.lattice(coarse=True)
    for i, (tile, k) in enumerate(graph.stances):
        probe = _probe(state, tile, k)
        assert probe is not None
        want = np.zeros(_NCELL, dtype=bool)
        for tx, ty in landtable.landable_set(probe[0], probe[1], grids=grids):
            want[tx * mm.N + ty] = True
        assert np.array_equal(graph.seen[i], want), (tile, k)
        assert graph.eye[i] == pytest.approx(stance_eye(state, tile, k))


def test_hops_to_is_the_backward_distance_to_the_strike_set():
    """One level BFS answers for all stances what a per-stance forward BFS answers one at
    a time, under both edge sets, and -1 wherever the strike set is unreachable."""
    for seed in range(3):
        graph = _synth(seed=seed)
        for j in range(len(graph)):
            tile = graph.stances[j][0]
            for monotone in (True, False):
                got = graph.hops_to(tile, monotone=monotone)
                assert np.array_equal(got, _bfs_hops_to(graph, tile, monotone)), (
                    seed,
                    j,
                    monotone,
                )
    graph = _synth()
    unseen = [f for f in range(_NCELL) if not graph.seen[:, f].any()]
    tile = (unseen[0] // mm.N, unseen[0] % mm.N)
    assert (graph.hops_to(tile) == -1).all()
    assert graph.hops_to(tile) is graph.hops_to(tile)  # memoized per (tile, monotone)


def test_route_reports_its_hops_and_its_arrival():
    """A route is its hop list: its length, and a repr naming the energy and the goal."""
    route = Route([4, 9], 12, (3, 5))
    assert len(route) == 2 and route.path == [4, 9]
    assert repr(route) == "Route(hops=2, energy=12, goal=(3, 5))"


def test_probe_refuses_a_tile_no_stack_can_stand_on(new_state):
    """A tree is unstackable, so neither the boulder arm nor the robot arm returns."""
    state = new_state(_LS)
    tile = _tree_tile(state)
    assert _probe(state, tile, 1) is None  # the boulder never lands
    assert _probe(state, tile, 0) is None  # nor does the robot alone
    assert _probe(state, flat_tiles(state)[0], 0) is not None


def test_build_skips_unprobeable_tiles_and_logs_the_sweep(new_state):
    """An unstackable tile contributes no stance, and ``log`` gets one summary line."""
    state = new_state(_LS)
    good = flat_tiles(state)[0]
    lines = []
    graph = StanceGraph.build(
        state, kmax=0, tiles=[_tree_tile(state), good], log=lines.append
    )
    assert graph.stances == [(good, 0)]
    assert len(lines) == 1 and "stances swept" in lines[0]


def test_cached_re_sweeps_a_corrupt_cache_file(tmp_path, monkeypatch, new_state):
    """A file that is not an npz is not a graph: the sweep runs again and overwrites it."""
    state = new_state(0)
    graph = _synth(seed=7)
    graph.sig = signature(state)
    path = tmp_path / f"{signature(state)}.npz"
    path.write_bytes(b"not an npz at all")
    calls = []

    def fake_build(cls, _st, _kmax=KMAX, _log=None, _tiles=None):
        calls.append(cls)
        return graph

    monkeypatch.setattr(StanceGraph, "build", classmethod(fake_build))
    got = StanceGraph.cached(state, cache_dir=str(tmp_path))
    assert len(calls) == 1
    assert got.stances == graph.stances
    assert StanceGraph.load(str(path)).stances == graph.stances


def test_describe_walks_the_route_eye_by_eye(new_state):
    """Each row is the stance hopped onto, and its eye is the next row's ``before``."""
    state = new_state(0)
    graph = _synth(seed=2)
    route = Route([1, 3, 5], 7, graph.stances[5][0])
    rows = list(graph.describe(route, state))
    assert [r[0] for r in rows] == [1, 2, 3]
    eye = state.eye_z()
    for (_i, tile, k, before, after), node in zip(rows, route.path):
        assert (tile, k) == graph.stances[node]
        assert before == eye and after == float(graph.eye[node])
        eye = after


def test_live_seen_is_the_bodys_own_landable_set(new_state):
    """The start row is ``landable_set`` on the live body, flattened to a board mask."""
    state = new_state(_LS)
    mask = live_seen(state)
    assert mask.shape == (_NCELL,) and mask.dtype == np.bool_
    grids = landtable.lattice(coarse=True)
    want = {
        (int(tx), int(ty))
        for tx, ty in landtable.landable_set(state, state.player, grids=grids)
    }
    assert {(int(f) // mm.N, int(f) % mm.N) for f in np.flatnonzero(mask)} == want
    assert np.array_equal(live_seen(state, grids=grids), mask)


def test_route_to_climbs_the_live_body_into_the_cached_graph(
    tmp_path, monkeypatch, new_state
):
    """``route_to`` hands back the cached graph and the live body's route through it."""
    state = new_state(_LS)
    graph = _reachable_graph(state)
    monkeypatch.setattr(StanceGraph, "cached", classmethod(lambda cls, *a, **kw: graph))
    got, route = route_to(state, (0, 0), cache_dir=str(tmp_path))
    assert got is graph
    assert route is not None and route.path == [0] and route.goal == (0, 0)
    assert not list(tmp_path.iterdir())  # the sweep never ran, so nothing was written


def test_main_prints_every_hop_of_the_route_it_found(tmp_path, monkeypatch, capsys):
    """The CLI routes the typed board's body to the Sentinel and describes the climb."""
    graph = _reachable_graph(Game.typed(_LS).state)
    monkeypatch.setattr(StanceGraph, "cached", classmethod(lambda cls, *a, **kw: graph))
    assert main([str(_LS), "--cache-dir", str(tmp_path)]) == 0
    out = capsys.readouterr().out
    assert "1 strike stances on 1 tiles" in out
    assert "route: 1 hops" in out
    assert str(graph.stances[0][0]) in out and "k=1" in out


def test_main_reports_no_route_on_an_open_board(tmp_path, monkeypatch, capsys):
    """``--open`` strips the trees and sentries; a stanceless graph still strikes nothing."""
    monkeypatch.setattr(
        StanceGraph, "cached", classmethod(lambda cls, *a, **kw: _synth(n=0))
    )
    argv = [str(_LS), "--open", "--target", "1,1", "--cache-dir", str(tmp_path)]
    assert main(argv) == 1
    out = capsys.readouterr().out
    assert "target=(1, 1)" in out
    assert "0 strike stances on 0 tiles" in out and "NO ROUTE" in out
