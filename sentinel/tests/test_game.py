"""The Game facade drives the whole simulator on one state, with no emulator."""

from sentinel.game import Game
from sentinel import memmap as mm


def test_new_builds_a_board():
    g = Game.new(42)
    assert g.player_xy() == (14, 27)
    assert g.energy == 10
    assert g.won() is False
    assert len(g.enemy_slots()) >= 1


def test_landscape_0_is_fixed():
    g = Game.new(0)
    assert g.player_xy() == (8, 17)


def test_clone_is_independent():
    g = Game.new(42)
    h = g.clone()
    h.step_enemies()
    # advancing the clone must not touch the original board.
    assert g.state.mem != h.state.mem


def test_objects_lists_occupied_slots():
    g = Game.new(42)
    objs = g.objects()
    types = {t for _s, t, _x, _y in objs}
    assert mm.T_SENTINEL in types
    assert mm.T_PLATFORM in types
    # the player robot is present
    assert any(t == mm.T_ROBOT for _s, t, _x, _y in objs)


def test_step_enemies_advances_cursor_and_rotates():
    g = Game.new(42)
    enemy = g.enemy_slots()[0]
    before = g.state.obj_h_angle[enemy]
    for _ in range(300):
        g.step_enemies()
    after = g.state.obj_h_angle[enemy]
    assert after != before  # the enemy has rotated over 300 rounds


def test_player_sees_returns_bool():
    g = Game.new(42)
    assert isinstance(g.player_sees(g.platform_xy()), bool)


def test_enemy_sees_and_meanie_threat():
    g = Game.new(42)
    enemy = g.enemy_slots()[0]
    sentinel_slot = g.state.slot_of_type(mm.T_SENTINEL)
    assert isinstance(g.enemy_sees(enemy, sentinel_slot), bool)
    assert g.meanie_threat(enemy) in (None, g.state.mem[mm.PLAYER_OBJECT])
