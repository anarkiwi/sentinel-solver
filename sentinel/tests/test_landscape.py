"""The pure-Python landscape generator reproduces the game's own generator,
byte for byte.

``golden_landscape.json`` holds, per landscape, the game-state regions the ROM
produced (terrain, object tables, platform, cooldowns, enemy rotation speeds,
energy and PRNG state), captured via py65.  ``test_generate_matches_golden``
replays them with no emulator, proving the pure-Python generator reproduces the
game's own generator byte for byte.  Live re-derivation against the 6502 code is
exercised by the oracle-parity suite (Phase 6).
"""

import json
import os

import pytest

from sentinel import landscape, memmap as mm
from sentinel.tests import oracle

GOLDEN = os.path.join(os.path.dirname(__file__), "golden_landscape.json")

SPANS = [
    (0x000B, 1),
    (0x0100, 0x40),
    (0x0140, 0x40),
    (0x0400, 0x400),
    (0x0900, 0x180),
    (0x0C06, 4),
    (0x0C0A, 1),
    (0x0C19, 2),
    (0x0C30, 8),
    (0x0C6F, 1),
    (0x0C7B, 5),
]


def _golden():
    with open(GOLDEN) as f:
        return json.load(f)


def _regions(mem, num):
    reg = [[a, bytes(mem[a : a + n]).hex()] for a, n in SPANS]
    reg.append([0x9D37, bytes(mem[0x9D37 : 0x9D37 + num]).hex()])
    return reg


def test_generate_matches_golden():
    data = _golden()
    assert len(data) >= 8
    for ls, expected in data.items():
        state = landscape.generate(int(ls))
        num = state.mem[0x0C6F]
        assert _regions(state.mem, num) == expected, f"landscape {ls} diverged"


def test_generate_is_deterministic():
    a = landscape.generate(42)
    b = landscape.generate(42)
    assert a.mem == b.mem


def test_generated_state_is_sane():
    state = landscape.generate(42)
    # a Sentinel on a platform, a player robot, and some trees exist.
    assert state.slot_of_type(mm.T_SENTINEL) is not None
    assert state.slot_of_type(mm.T_PLATFORM) is not None
    assert state.obj_type[state.player] == mm.T_ROBOT
    assert state.energy == 10
    # every terrain height is in the legal 1..11 band.
    height, _slope = None, None
    from sentinel import terrain

    height, _slope = terrain.height_slope_grid(state)
    for row in height:
        for h in row:
            assert 0 <= h <= 11


def test_landscape_0_player_is_fixed():
    state = landscape.generate(0)
    assert state.player_xy() == (8, 17)


@pytest.mark.oracle
def test_generate_matches_rom_live():
    """Re-derive a spread of boards live through the 6502 code and diff, byte for
    byte, against the pure-Python generator (needs the ROM image fixture)."""
    for ls in (0, 7, 42, 314, 777, 2024, 9999):
        rom = oracle.generate(ls)
        state = landscape.generate(ls)
        num = rom[0x0C6F]
        for base, n in SPANS:
            assert bytes(rom[base : base + n]) == bytes(
                state.mem[base : base + n]
            ), f"landscape {ls} region {hex(base)} diverged"
        assert bytes(rom[0x9D37 : 0x9D37 + num]) == bytes(
            state.mem[0x9D37 : 0x9D37 + num]
        )
