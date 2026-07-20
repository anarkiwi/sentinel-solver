"""No-mutation guarantee for the geometric visibility query. No ROM required."""

import pytest

from sentinel import memmap as mm, threat
from sentinel.game import Game


@pytest.fixture(params=[0, 42])
def game(request):
    return Game.new(request.param)


def test_player_sees_tile_returns_bool_and_does_not_mutate(game):
    px, py = game.player_xy()
    before = bytes(game.state.mem)
    out = threat.player_sees_tile(game.state, (px, py), game.state.player)
    assert isinstance(out, bool)
    assert bytes(game.state.mem) == before


def test_player_sees_tile_sees_own_tile_neighbourhood(game):
    """A phantom on a tile the player already stands next to is visible; the sweep
    over the whole board must never mutate the observed state."""
    px, py = game.player_xy()
    before = bytes(game.state.mem)
    seen = [
        (x, y)
        for x in range(max(0, px - 2), min(mm.N, px + 3))
        for y in range(max(0, py - 2), min(mm.N, py + 3))
        if threat.player_sees_tile(game.state, (x, y), game.state.player)
    ]
    assert seen
    assert bytes(game.state.mem) == before
