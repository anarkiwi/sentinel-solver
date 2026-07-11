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
from sentinel.terrain import set_tile_byte

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


def _empty_state():
    state = State.from_mem(bytearray(0x10000))
    for s in range(mm.NUM_SLOTS):
        state.obj_flags[s] = 0x80  # all slots empty; terrain is bare flat height 0
    state.mem[mm.PRND_STATE] = 0x01  # non-degenerate PRNG for hyperspace placement
    return state


def _player_on_platform(state, px=8, py=8):
    """Slot $3F platform at (px, py) with a player robot (slot 5) stacked on it."""
    state.obj_flags[0x3F] = 0x00
    state.obj_type[0x3F] = mm.T_PLATFORM
    state.obj_x[0x3F], state.obj_y[0x3F] = px, py
    state.obj_z_height[0x3F], state.obj_z_frac[0x3F] = 0, 0xE0
    set_tile_byte(state, px, py, mm.OBJECT_TILE | 0x3F)
    state.mem[mm.PLATFORM_X], state.mem[mm.PLATFORM_Y] = px, py
    player = 5
    state.obj_flags[player] = 0x40 | 0x3F
    state.obj_type[player] = mm.T_ROBOT
    state.obj_x[player], state.obj_y[player] = px, py
    state.obj_z_height[player], state.obj_z_frac[player] = 1, 0xE0
    state.player = player
    return player


def test_on_platform_is_not_by_itself_a_win():
    state = _empty_state()
    _player_on_platform(state)
    state.energy = 10
    # Standing on the platform is on_platform, but NOT a win: the landscape-complete
    # flag ($0CDE bit6) is set only by hyperspacing from the platform ($217F).
    assert actions.on_platform(state) is True
    assert actions.won(state) is False


def test_won_only_after_hyperspace_from_platform():
    state = _empty_state()
    _player_on_platform(state)
    state.energy = 10
    assert actions.won(state) is False
    assert actions.hyperspace(state) is True  # survived
    assert actions.won(state) is True  # $0CDE == $C0 (bit7 + bit6)
    assert actions.player_dead(state) is False


def test_hyperspace_off_platform_is_not_a_win():
    state = _empty_state()
    player = 5
    state.obj_flags[player] = 0x00
    state.obj_type[player] = mm.T_ROBOT
    state.obj_x[player], state.obj_y[player] = 3, 3  # not the platform tile
    state.obj_z_height[player], state.obj_z_frac[player] = 0, 0xE0
    set_tile_byte(state, 3, 3, mm.OBJECT_TILE | player)
    state.mem[mm.PLATFORM_X], state.mem[mm.PLATFORM_Y] = 20, 20
    state.player = player
    state.energy = 10
    assert actions.hyperspace(state) is True  # survived
    assert actions.won(state) is False  # not on the platform -> no complete flag


def test_hyperspace_underfunded_kills():
    state = _empty_state()
    player = 5
    state.obj_flags[player] = 0x00
    state.obj_type[player] = mm.T_ROBOT
    state.obj_x[player], state.obj_y[player] = 3, 3
    state.obj_z_height[player], state.obj_z_frac[player] = 0, 0xE0
    set_tile_byte(state, 3, 3, mm.OBJECT_TILE | player)
    state.player = player
    state.energy = 2  # < 3 (robot value) -> the hyperspace kills
    assert actions.hyperspace(state) is False
    assert actions.player_dead(state) is True
    assert state.player == player  # the ROM does NOT relocate on a hyperspace death


def test_win_absorbs_hyperspaces_and_completes():
    state = _empty_state()
    px, py = 8, 8
    # a Sentinel (slot 0) standing on its platform (slot $3F).
    state.obj_flags[0x3F] = 0x00
    state.obj_type[0x3F] = mm.T_PLATFORM
    state.obj_x[0x3F], state.obj_y[0x3F] = px, py
    state.obj_z_height[0x3F], state.obj_z_frac[0x3F] = 0, 0xE0
    set_tile_byte(state, px, py, mm.OBJECT_TILE | 0x3F)
    state.mem[mm.PLATFORM_X], state.mem[mm.PLATFORM_Y] = px, py
    state.obj_flags[0] = 0x40 | 0x3F  # Sentinel stacked on the platform
    state.obj_type[0] = mm.T_SENTINEL
    state.obj_x[0], state.obj_y[0] = px, py
    state.obj_z_height[0], state.obj_z_frac[0] = 1, 0xE0
    # a spare player robot somewhere so `state.player` is valid pre-win.
    player = 5
    state.obj_flags[player] = 0x00
    state.obj_type[player] = mm.T_ROBOT
    state.obj_x[player], state.obj_y[player] = 2, 2
    state.obj_z_height[player], state.obj_z_frac[player] = 0, 0xE0
    set_tile_byte(state, 2, 2, mm.OBJECT_TILE | player)
    state.player = player
    state.energy = 10
    assert actions.win(state) is True
    assert actions.won(state) is True
    assert state.slot_of_type(mm.T_SENTINEL) is None  # Sentinel absorbed


def test_absorb_locked_after_sentinel_absorbed():
    state = _empty_state()
    # Sentinel in slot 0, a free-standing tree in slot 10.
    state.obj_flags[0] = 0x00
    state.obj_type[0] = mm.T_SENTINEL
    state.obj_x[0], state.obj_y[0] = 1, 1
    state.obj_z_height[0] = 0
    tree = 10
    state.obj_flags[tree] = 0x00
    state.obj_type[tree] = mm.T_TREE
    state.obj_x[tree], state.obj_y[tree] = 2, 2
    state.obj_z_height[tree] = 0
    state.energy = 0
    # both are absorbable while the Sentinel lives.
    assert actions.can_absorb(state, tree) is True
    assert actions.can_absorb(state, 0) is True
    # absorb the Sentinel; slot 0 becomes SLOT_EMPTY.
    assert actions.absorb(state, 0) is True
    # now $1B8E's absolute check on slot 0 rejects EVERY absorb.
    assert actions.can_absorb(state, tree) is False
    assert actions.absorb(state, tree) is False
    assert state.obj_type[tree] == mm.T_TREE  # unchanged; still present


def test_player_dead_detects_both_death_paths():
    state = State.from_mem(bytearray(0x10000))
    assert actions.player_dead(state) is False
    # drain-death: kill_player $1A00 sets $0C4E bit7.
    state.mem[mm.PLAYER_DIED_BY_DRAINING] = 0x80
    assert actions.player_dead(state) is True
    state.mem[mm.PLAYER_DIED_BY_DRAINING] = 0
    # meanie forced-hyperspace death: $0CDE bit7 set, complete bit6 clear.
    state.mem[mm.PLAYER_HAS_HYPERSPACED] = 0x80
    assert actions.player_dead(state) is True
    # a win also touches $0CDE (bit6): 0xC0 is NOT death.
    state.mem[mm.PLAYER_HAS_HYPERSPACED] = 0xC0
    assert actions.player_dead(state) is False
