"""The object-relative geometry reproduces the game's fixed-point trig exactly.

``golden_relative.json`` holds, per landscape, every enemy->object pair with the
ROM's $8401 outputs (FOV byte, horizontal angle, relative z, distance) and the
$1887 full-visibility bit. ``test_relative_matches_golden`` replays them with no
emulator; the coefficient-table and self-consistency tests need no fixture at all.
"""

import json
import math
import os

from sentinel import landscape, relative

GOLDEN = os.path.join(os.path.dirname(__file__), "golden_relative.json")
FOV = 0x14


def test_arctan_table_matches_closed_form():
    for y in range(257):
        want = round(math.atan(y / 256.0) / (2 * math.pi) * 65536.0)
        assert relative._ARCTAN_LO[y] == (want & 0xFF)
        assert relative._ARCTAN_HI[y] == ((want >> 8) & 0xFF)


def test_hypotenuse_table_endpoints():
    assert relative._HYP[0] == 0
    assert len(relative._HYP) == 129


def test_divide_and_arctan_is_pure():
    # a pure function of its inputs (no shared state between calls).
    a = relative._divide_and_arctan(0x40, 0, 0x80, 0)
    b = relative._divide_and_arctan(0x40, 0, 0x80, 0)
    assert a == b
    assert len(a) == 3 and all(0 <= v <= 0xFF for v in a)


def test_relative_angles_matches_golden():
    with open(GOLDEN) as f:
        data = json.load(f)
    keys = ["c57", "angle_lo", "angle_hi", "z_lo", "z_hi", "hyp_lo", "hyp_hi"]
    checked = 0
    for ls, pairs in data.items():
        state = landscape.generate(int(ls))
        for obs, tgt, etype, rec in pairs:
            ra = relative.relative_angles(state, obs, tgt)
            got = [ra[k] for k in keys]
            assert got == rec[:7], f"ls{ls} {obs}->{tgt} angles {got} != {rec[:7]}"
            cs = relative.can_see_object(state, obs, tgt, etype, FOV)
            assert int(cs["full"]) == rec[7], f"ls{ls} {obs}->{tgt} vis"
            checked += 1
    assert checked > 100


def test_can_see_rejects_wrong_type_and_empty():
    state = landscape.generate(42)
    enemy = state.slot_of_type(2)  # a tree slot as a stand-in target
    sentinel_slot = state.slot_of_type(5)
    # wrong expected type -> not in slot
    res = relative.can_see_object(state, sentinel_slot, enemy, 0, FOV)
    assert res["in_slot"] is False
    # empty slot -> not in slot
    empty = next(s for s in range(64) if state.is_empty(s))
    res = relative.can_see_object(state, sentinel_slot, empty, 2, FOV)
    assert res["in_slot"] is False
