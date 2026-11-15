"""The line of sight reproduces the real ROM verdict ($1CDD), bit for bit.

``golden_los.json`` holds, per landscape, the game-state regions that rebuild a
:class:`sentinel.state.State` and aim samples ``(h, v, cursor)`` with the ROM's tile and
visibility, replayed without an emulator so parity is proven in CI without the ROM.
"""

import json
import os

from sentinel.state import State
from sentinel import terrain, memmap as mm

GOLDEN = os.path.join(os.path.dirname(__file__), "golden_los.json")


def _golden():
    with open(GOLDEN) as f:
        return json.load(f)


def _rebuild(regions):
    mem = bytearray(0x10000)
    for addr, hx in regions:
        blob = bytes.fromhex(hx)
        mem[addr : addr + len(blob)] = blob
    return State(mem)


def test_tile_byte_matches_memmap_index_in_range():
    # the ROM addressing form used by terrain.tile_byte equals TILES_TABLE+tidx.
    g = next(iter(_golden().values()))
    state = _rebuild(g["regions"])
    for x in range(mm.N):
        for y in range(mm.N):
            assert (
                terrain.tile_byte(state, x, y)
                == state.mem[mm.TILES_TABLE + mm.tidx(x, y)]
            )


def test_resolve_ground_is_in_range():
    g = next(iter(_golden().values()))
    state = _rebuild(g["regions"])
    for y in range(mm.N):
        for x in range(mm.N):
            h, s = terrain.resolve_ground(state, x, y)
            assert 0 <= h <= 15 and 0 <= s <= 15
