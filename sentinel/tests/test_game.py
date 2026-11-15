"""The Game facade builds a board as the game enters it, with no emulator."""

from sentinel import memmap as mm
from sentinel.game import Game


def test_new_builds_a_board_at_entry():
    g = Game.new(42)
    assert g.player_xy() == (14, 27) and g.energy == 10
    assert len(g.enemy_slots()) >= 1
    assert g.state.mem[mm.PLAYER_NOT_ACTED] == 0x80 and g.state.mem[mm.CURSOR] == 7


def test_landscape_0_is_fixed():
    assert Game.typed(0).player_xy() == (8, 17)


def test_clone_is_independent():
    g = Game.new(42)
    h = g.clone()
    h.state.energy = 3
    assert g.energy == 10 and g.state.mem != h.state.mem


def test_enemy_slots_are_the_occupied_sentries_and_sentinel():
    st = Game.typed(335).state
    want = [
        s
        for s in range(mm.NUM_SLOTS)
        if not st.obj_flags[s] & 0x80 and st.obj_type[s] in mm.ENEMY_TYPES
    ]
    assert st.enemy_slots() == want and mm.T_SENTINEL in {st.obj_type[s] for s in want}
    st.obj_flags[want[0]] = 0x80
    assert st.enemy_slots() == want[1:]
