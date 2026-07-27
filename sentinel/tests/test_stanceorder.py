"""The goal ordering is an exact DP, so it is pinned against exhaustive search.

``leg_matrix`` and ``start_legs`` are checked against an independent per-stance forward
BFS, and ``order`` against ``itertools.permutations`` over the sentries.  Sweeping a real
board costs ~40 s, so every invariant is pinned on synthetic graphs.
"""

from itertools import permutations

import numpy as np

from sentinel import actions, enemies, memmap as mm
from sentinel.stancegraph import StanceGraph
from sentinel.stanceorder import (
    UNREACHABLE,
    _tail_only,
    _unwind,
    goal_tiles,
    leg_matrix,
    order,
    start_legs,
)
from sentinel.tests.test_stance_player import _empty_graph
from sentinel.tests.test_stancegraph import _bfs_hops_to, _synth

_LS = 335  # the board with 7 enemies, so the sentry subset DP is non-trivial
_NCELL = mm.N * mm.N


class _Board:
    """The whole board interface ``goal_tiles`` reads: enemy slots, tiles, the platform."""

    def __init__(self, sentries, sentinel, platform):
        self._tiles = {} if sentinel is None else {actions.SENTINEL_SLOT: sentinel}
        self._tiles.update(dict(enumerate(sentries, 1)))
        self.platform_xy = platform
        self.obj_type = [mm.T_SENTRY] * mm.NUM_SLOTS
        self.obj_type[actions.SENTINEL_SLOT] = mm.T_SENTINEL

    def is_empty(self, slot):
        return slot not in self._tiles

    def tile_of(self, slot):
        return self._tiles[slot]


def _mask(tiles):
    """``tiles`` as a flat board mask."""
    out = np.zeros(_NCELL, dtype=bool)
    for x, y in tiles:
        out[x * mm.N + y] = True
    return out


def _tiny(stances, strikes, eye):
    """A hand-built graph where ``strikes[i]`` is exactly what stance ``i`` can land."""
    return StanceGraph(
        stances,
        np.array([_mask(s) for s in strikes], dtype=bool).reshape(len(stances), _NCELL),
        np.zeros(len(stances), dtype=np.int32),
        np.zeros(len(stances), dtype=np.int32),
        np.array(eye, dtype=np.float32),
        1,
        "tiny",
    )


def _blind_tile(graph):
    """A tile no stance in ``graph`` can strike."""
    unseen = [f for f in range(_NCELL) if not graph.seen[:, f].any()]
    assert unseen
    return (unseen[0] // mm.N, unseen[0] % mm.N)


def _want_start(graph, tiles, start_seen, start_eye):
    """``start_legs`` re-derived off the per-stance forward BFS, not the backward tables."""
    reach = np.flatnonzero(start_seen[graph._cols] & (graph.eye > start_eye))
    out = []
    for tile in tiles:
        if start_seen[tile[0] * mm.N + tile[1]]:
            out.append(0)
            continue
        dist = _bfs_hops_to(graph, tile)[reach]
        live = dist[dist >= 0]
        out.append(1 + int(live.min()) if live.size else UNREACHABLE)
    return out


def _cost(legs, start, perm, m, n):
    """Total legs of taking the sentries in ``perm``, then the tail in its forced order."""
    here = perm[0]
    total = int(start[here])
    for j in list(perm[1:]) + list(range(m, n)):
        total += int(legs[here, j])
        here = j
    return total


def test_goal_tiles_splits_the_sentries_from_the_sentinel_and_the_platform(new_state):
    """The Sentinel is its own goal and never a sentry; the platform comes off the map."""
    st = new_state(_LS)
    sentries, sentinel, platform = goal_tiles(st)
    slots = enemies.enemy_slots(st)
    assert sentinel == tuple(int(v) for v in st.tile_of(actions.SENTINEL_SLOT))
    assert sentries == [
        tuple(int(v) for v in st.tile_of(s))
        for s in slots
        if s != actions.SENTINEL_SLOT
    ]
    assert len(sentries) == len(slots) - 1 and sentinel not in sentries
    assert platform == tuple(st.platform_xy)
    bare = _Board([(1, 0)], None, (2, 3))
    assert goal_tiles(bare) == ([(1, 0)], None, (2, 3))


def test_leg_matrix_is_the_least_backward_hop_between_the_strike_sets():
    """``legs[i][j]`` is the min over ``i``'s strikers of the hops to ``j``'s strike set."""
    for seed in range(4):
        graph = _synth(seed=seed)
        blind = _blind_tile(graph)
        tiles = [graph.stances[i][0] for i in range(0, len(graph), 2)] + [blind]
        legs = leg_matrix(graph, tiles)
        for i, tile in enumerate(tiles):
            src = graph.strike_stances(tile)
            for j, other in enumerate(tiles):
                dist = _bfs_hops_to(graph, other)[src]
                live = dist[dist >= 0]
                if not src.size:
                    want = UNREACHABLE
                elif i == j:
                    want = 0
                else:
                    want = int(live.min()) if live.size else UNREACHABLE
                assert legs[i, j] == want, (seed, i, j)
        assert (legs[-1] == UNREACHABLE).all()  # a goal with no striker bridges nothing
        assert (legs[:, -1] == UNREACHABLE).all()
        assert (np.diag(legs)[:-1] == 0).all()
        assert (legs == UNREACHABLE).any()


def test_start_legs_is_zero_when_landable_and_a_hop_past_the_reachable_stances():
    """The live body pays nothing for what it already lands, else one hop plus the climb."""
    landed = climbed = 0
    for seed in range(4):
        graph = _synth(seed=seed)
        tiles = [graph.stances[i][0] for i in range(0, len(graph), 3)]
        tiles.append(_blind_tile(graph))
        for row, eye in ((graph.seen[1], 0.0), (np.zeros(_NCELL, dtype=bool), 0.0)):
            got = start_legs(graph, tiles, row, eye).tolist()
            assert got == _want_start(graph, tiles, row, eye), seed
            landed += got.count(0)
            climbed += sum(0 < d < UNREACHABLE for d in got)
        assert start_legs(graph, tiles, np.zeros(_NCELL, dtype=bool), 0.0).tolist() == [
            UNREACHABLE
        ] * len(tiles)
    assert landed and climbed


def test_order_is_the_exhaustive_optimum_over_the_sentry_permutations():
    """The Held-Karp order costs exactly what the best of all ``m!`` orders costs.

    The tail is forced -- ``$1B8E`` puts the Sentinel after every sentry and the platform
    after it -- so only the sentry prefix is searched, here against every permutation.
    """
    graded = fell_back = 0
    for seed in range(6):
        graph = _synth(seed=seed)
        reach = [graph.stances[i][0] for i in (0, 2, 4, 6, 8)]
        sentinel, platform = graph.stances[10][0], graph.stances[11][0]
        for sentries in (reach, reach[:4] + [_blind_tile(graph)]):
            st = _Board(sentries, sentinel, platform)
            tiles = sentries + [sentinel, platform]
            m, n = len(sentries), len(tiles)
            legs = leg_matrix(graph, tiles)
            for k in (1, 5, 9):
                seen, eye = graph.seen[k], float(graph.eye[k]) - 0.6
                start = start_legs(graph, tiles, seen, eye)
                costs = [_cost(legs, start, p, m, n) for p in permutations(range(m))]
                got = order(graph, st, seen, eye)
                if min(costs) >= UNREACHABLE:
                    assert got == _tail_only(graph, [sentinel, platform], seen, eye)
                    fell_back += 1
                    continue
                assert sorted(got[:m]) == sorted(sentries)  # each sentry taken once
                assert got[m:] == [sentinel, platform]
                perm = [tiles.index(t) for t in got[:m]]
                assert _cost(legs, start, perm, m, n) == min(costs), (seed, k)
                graded += len(set(costs)) > 1
    assert graded >= 4 and fell_back


def test_order_without_sentries_keeps_only_the_reachable_tail():
    """Nothing left to absorb: the order is whichever of Sentinel/platform is reachable."""
    sentinel, platform, stance = (3, 4), (5, 6), ((7, 8), 0)
    st = _Board([], sentinel, platform)
    seen = _mask([stance[0]])
    assert order(_tiny([stance], [[sentinel]], [2.0]), st, seen, 0.0) == [sentinel]
    blind = _tiny([stance], [[]], [2.0])
    assert order(blind, st, seen, 0.0) == [
        sentinel,
        platform,
    ]  # neither: the whole tail
    assert order(blind, _Board([], None, platform), seen, 0.0) == [platform]


def test_order_falls_back_to_the_tail_when_no_stance_bridges_the_sentries():
    """Every leg unreachable, so the DP has no finish and the tail is all that is left."""
    sentinel, platform, stance = (3, 4), (5, 6), ((7, 8), 0)
    st = _Board([(1, 1), (2, 2)], sentinel, platform)
    graph = _tiny([stance], [[]], [2.0])
    assert order(graph, st, _mask([stance[0]]), 0.0) == [sentinel, platform]


def test_an_empty_graph_answers_with_the_tail_and_no_tail_answers_nothing():
    """No stance at all: every leg is unreachable, and an empty tail is an empty order."""
    graph = _empty_graph()
    seen = np.zeros(_NCELL, dtype=bool)
    st = _Board([(1, 0)], (3, 4), (5, 6))
    assert order(graph, st, seen, 0.0) == [(3, 4), (5, 6)]
    assert _tail_only(graph, [], seen, 0.0) == []


def test_unwind_recovers_the_chain_as_a_permutation():
    """Walking the predecessor table back drops each visited sentry from the mask once."""
    prev = np.full((8, 3), -1, dtype=np.int64)
    prev[0b111, 1] = 2
    prev[0b101, 2] = 0
    got = _unwind(prev, 0b111, 1)
    assert got == [0, 2, 1] and sorted(got) == [0, 1, 2]
