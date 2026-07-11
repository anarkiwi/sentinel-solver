"""Type and sanity checks for the threat helpers, plus a no-mutation guarantee
for the visibility queries. No ROM required."""

import pytest

from sentinel import memmap as mm, threat, relative
from sentinel.game import Game
from sentinel.state import State


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


def test_meanie_safe_flags_tree_in_range_enemy_sees_tree(monkeypatch):
    """attempt_to_create_meanie $19A1 arms a meanie with NO tree->player LOS test:
    a partially-seen player plus a tree in range the enemy fully sees is enough, so
    `meanie_safe` must report UNSAFE.  (Before the fix its spurious tree->player
    condition wrongly called this tile safe.)"""
    state = State.from_mem(bytearray(0x10000))
    for s in range(mm.NUM_SLOTS):
        state.obj_flags[s] = 0x80
    enemy = 0
    state.obj_flags[enemy] = 0x00
    state.obj_type[enemy] = mm.T_SENTRY
    state.obj_x[enemy], state.obj_y[enemy] = 5, 5
    tree = 10
    state.obj_flags[tree] = 0x00
    state.obj_type[tree] = mm.T_TREE
    state.obj_x[tree], state.obj_y[tree] = 8, 8  # within 10 tiles of the query tile

    def fake_can_see(_st, observer, _target, expected_type, _fov, max_steps=20000):
        if expected_type == mm.T_TREE:  # (C) the enemy fully sees the tree
            return {"full": True, "in_fov": True, "in_slot": True, "probes": [True]}
        if observer == enemy:  # (A) enemy sees the phantom player partially (head only)
            return {
                "full": False,
                "in_fov": True,
                "in_slot": True,
                "probes": [True, False],
            }
        return {"full": False, "in_fov": False, "in_slot": True, "probes": [False]}

    monkeypatch.setattr(relative, "can_see_object", fake_can_see)
    assert threat.meanie_safe(state, (7, 7)) is False


def test_drain_over_window(game):
    before = bytes(game.state.mem)
    d = threat.drain_over_window(game.state, 20)
    assert isinstance(d, int)
    assert d >= 0
    assert bytes(game.state.mem) == before
