"""The exposure scan spec matches the pure-Python scan, results and memory side effects."""

import numpy as np
import pytest

from sentinel import exposure_jit, memmap as mm, relative
from sentinel.game import Game
from sentinel.tests import aimref

LANDSCAPES = [0, 42, 335]


def _exposures(st, slot):
    rows = exposure_jit.exposures_of(np.frombuffer(st.mem, dtype=np.uint8), slot)
    return [(int(e), int(ah), bool(full)) for e, ah, full in rows.tolist()]


def _reference(st, slot):
    out = []
    for e in st.enemy_slots():
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
@pytest.mark.parametrize("boulders", [0, 4])
def test_matches_python_on_every_body(landscape, boulders):
    st = aimref.board(landscape, boulders)
    seen_any = False
    for turn in range(0, 256, 64):
        for e in st.enemy_slots():
            st.obj_h_angle[e] = (st.obj_h_angle[e] + turn) & 0xFF
        for slot in _bodies(st):
            ref, got = st.clone(), st.clone()
            want = _reference(ref, slot)
            assert _exposures(got, slot) == want
            assert bytes(got.mem) == bytes(ref.mem)
            seen_any |= bool(want)
    assert seen_any or landscape != 0


def test_empty_when_no_enemy_sees():
    st = Game.typed(335).state
    for e in st.enemy_slots():
        st.obj_flags[e] = 0x80
    assert _exposures(st, st.player) == []
