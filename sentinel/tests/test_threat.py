"""Type and sanity checks for the threat helpers, plus a no-mutation guarantee
for the visibility queries. No ROM required."""

import pytest

from sentinel import memmap as mm, threat
from sentinel.game import Game


@pytest.fixture(params=[0, 42])
def game(request):
    return Game.new(request.param)


def _all_tiles():
    return [(x, y) for x in range(mm.N) for y in range(mm.N)]


def test_is_exposed_returns_bool(game):
    px, py = game.player_xy()
    assert isinstance(threat.is_exposed(game.state, px, py), bool)


def test_is_exposed_does_not_mutate(game):
    px, py = game.player_xy()
    before = bytes(game.state.mem)
    threat.is_exposed(game.state, px, py)
    assert bytes(game.state.mem) == before


def test_exposed_tiles_is_subset(game):
    tiles = _all_tiles()
    before = bytes(game.state.mem)
    out = threat.exposed_tiles(game.state, tiles)
    assert isinstance(out, set)
    assert out <= set(tiles)
    assert bytes(game.state.mem) == before


def test_exposed_tiles_matches_is_exposed(game):
    sample = [(x, y) for x in range(0, mm.N, 4) for y in range(0, mm.N, 4)]
    batched = threat.exposed_tiles(game.state, sample)
    for t in sample:
        assert (t in batched) == threat.is_exposed(game.state, t[0], t[1])


def test_gaze_distance(game):
    tiles = [(x, y) for x in range(0, mm.N, 4) for y in range(0, mm.N, 4)]
    out = threat.gaze_distance(game.state, tiles)
    assert isinstance(out, dict)
    assert set(out) == set(tiles)
    for v in out.values():
        assert isinstance(v, int)
        assert 0 <= v <= 128


def test_ticks_until_seen(game):
    px, py = game.player_xy()
    before = bytes(game.state.mem)
    horizon = 64
    t = threat.ticks_until_seen(game.state, px, py, horizon=horizon)
    assert isinstance(t, int)
    assert 0 <= t <= horizon
    assert bytes(game.state.mem) == before


def test_meanie_safe(game):
    px, py = game.player_xy()
    before = bytes(game.state.mem)
    out = threat.meanie_safe(game.state, (px, py))
    assert isinstance(out, bool)
    assert bytes(game.state.mem) == before


def test_drain_over_window(game):
    before = bytes(game.state.mem)
    d = threat.drain_over_window(game.state, 20)
    assert isinstance(d, int)
    assert d >= 0
    assert bytes(game.state.mem) == before
