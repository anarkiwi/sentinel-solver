"""The exposure scan spec matches the pure-Python scan, results and memory side effects."""

import pytest

from sentinel import enemies, memmap as mm, relative
from sentinel.game import Game
from sentinel.playerbase import BasePlayer

LANDSCAPES = [0, 42, 335]


def _reference(st, slot):
    out = []
    for e in enemies.enemy_slots(st):
        see = relative.can_see_object(st, e, slot, mm.T_ROBOT, 0x100)
        if see["exposure"]:
            out.append(
                (
                    e,
                    relative.relative_angles(st, e, slot)["angle_hi"],
                    bool(see["full"]),
                )
            )
    return out


def _bodies(st):
    return [st.player] + [
        s
        for s in range(mm.NUM_SLOTS)
        if not st.is_empty(s) and st.obj_type[s] in (mm.T_ROBOT, mm.T_TREE)
    ]


@pytest.mark.parametrize("landscape", LANDSCAPES)
def test_matches_python_over_time(landscape):
    game = Game.typed(landscape)
    st = game.state
    st.mem[mm.PLAYER_NOT_ACTED] = 0
    seen_any = False
    for _ in range(6):
        for slot in _bodies(st):
            ref, got = st.clone(), st.clone()
            want = _reference(ref, slot)
            assert BasePlayer._exposures(got, slot) == want
            assert bytes(got.mem) == bytes(ref.mem)
            seen_any |= bool(want)
        enemies.advance_frames(st, 400)
    assert seen_any or landscape != 0


def test_empty_when_no_enemy_sees():
    st = Game.typed(335).state
    for e in enemies.enemy_slots(st):
        st.obj_flags[e] = 0x80
    assert BasePlayer._exposures(st, st.player) == []
