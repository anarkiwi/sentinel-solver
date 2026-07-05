"""The player actions reproduce the real ROM, byte for byte.

``golden_actions.json`` holds, per operation, the game-state regions before and
after the real 6502 routine ran (create / stack / absorb / transfer), captured
via py65.  This test rebuilds a State from the "before" regions, applies the same
op through :mod:`sentinel.actions`, and asserts the state matches the ROM's
"after" -- with no emulator.
"""

import json
import os

from sentinel.state import State
from sentinel import actions, energy, memmap as mm

GOLDEN = os.path.join(os.path.dirname(__file__), "golden_actions.json")

# the regions the golden fixture stores (must match the generator's SPANS).
SPANS = [
    (0x000B, 1),
    (0x0100, 0x80),
    (0x0400, 0x400),
    (0x0900, 0x180),
    (0x0C0A, 1),
    (0x0C7B, 5),
]


def _golden():
    with open(GOLDEN) as f:
        return json.load(f)


def _rebuild(regions):
    mem = bytearray(0x10000)
    for addr, hx in regions:
        blob = bytes.fromhex(hx)
        mem[addr : addr + len(blob)] = blob
    return State(mem)


def _assert_regions(state, post, skip=()):
    exp = bytearray(0x10000)
    for addr, hx in post:
        blob = bytes.fromhex(hx)
        exp[addr : addr + len(blob)] = blob
    for base, n in SPANS:
        for i in range(n):
            a = base + i
            if a in skip:
                continue
            assert state.mem[a] == exp[a], f"byte {hex(a)} diverged from the ROM"


def test_actions_match_rom():
    recs = _golden()
    assert len(recs) >= 15
    for rec in recs:
        state = _rebuild(rec["pre"])
        op = rec["op"]
        skip = ()
        if op == "create":
            actions.create(state, rec["otype"], tuple(rec["tile"]))
        elif op == "stack":
            tile = tuple(rec["tile"])
            actions.create(state, mm.T_BOULDER, tile)
            actions.create(state, mm.T_BOULDER, tile)
        elif op == "absorb":
            actions.absorb(state, rec["slot"])
        elif op == "transfer":
            slot = rec["robot_slot"]
            actions.create(state, mm.T_ROBOT, tuple(rec["tile"]))
            actions.transfer(state, slot)
            # the ungated ROM create leaves the robot facing random; actions applies
            # the try_to_create_object face-player override -- an intentional 1-byte
            # difference validated separately (test_robot_faces_player).
            skip = (mm.OBJECTS_H_ANGLE + slot,)
        else:
            raise AssertionError(op)
        _assert_regions(state, rec["post"], skip=skip)


def test_energy_gain_wraps_mod_64():
    state = State.from_mem(bytearray(0x10000))
    state.energy = 62
    energy.gain(state, mm.T_SENTINEL)  # +4
    assert state.energy == 2  # 66 & 0x3F


def test_energy_lose_fails_on_underflow():
    state = State.from_mem(bytearray(0x10000))
    state.energy = 1
    assert energy.lose(state, mm.T_BOULDER) is False  # cost 2 > 1
    assert state.energy == 1  # unchanged
    assert energy.lose(state, mm.T_TREE) is True  # cost 1
    assert state.energy == 0


def test_create_fails_without_energy():
    state = State.from_mem(bytearray(0x10000))
    for s in range(mm.NUM_SLOTS):
        state.obj_flags[s] = 0x80  # all empty
    state.energy = 0
    assert actions.create(state, mm.T_BOULDER, (5, 5)) is None


def test_robot_faces_player():
    # try_to_create_object $1BE0: a created synthoid faces the player (h ^ $80).
    state = State.from_mem(bytearray(0x10000))
    for s in range(mm.NUM_SLOTS):
        state.obj_flags[s] = 0x80
    state.energy = 30
    player = 10
    state.obj_flags[player] = 0x00
    state.obj_type[player] = mm.T_ROBOT
    state.obj_h_angle[player] = 0x40
    state.obj_x[player] = 3
    state.obj_y[player] = 3
    state.player = player
    slot = actions.create(state, mm.T_ROBOT, (6, 6))
    assert slot is not None
    assert state.obj_h_angle[slot] == (0x40 ^ 0x80)


def test_on_platform_detects_win():
    state = State.from_mem(bytearray(0x10000))
    for s in range(mm.NUM_SLOTS):
        state.obj_flags[s] = 0x80
    # platform is always slot $3F; player stacked on it.
    state.obj_flags[0x3F] = 0x00
    state.obj_type[0x3F] = mm.T_PLATFORM
    player = 5
    state.obj_flags[player] = 0x40 | 0x3F
    state.obj_type[player] = mm.T_ROBOT
    state.player = player
    assert actions.on_platform(state) is True
    assert actions.won(state) is True
