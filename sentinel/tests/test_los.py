"""The line of sight reproduces the real ROM verdict ($1CDD), bit for bit.

``golden_los.json`` holds, per landscape, the minimal game-state regions needed
to rebuild a :class:`sentinel.state.State` plus a set of aim samples
``(h, v, cursor)`` with the tile + visibility the real 6502 code returned.  This
test replays them with no emulator, so it proves parity in CI without the ROM.
"""

import json
import os

from sentinel.state import State
from sentinel import los, terrain, memmap as mm

GOLDEN = os.path.join(os.path.dirname(__file__), "golden_los.json")

# every mem span the golden fixture stores (matches the LOS reads).
_SPANS = None


def _golden():
    with open(GOLDEN) as f:
        return json.load(f)


def _rebuild(regions):
    mem = bytearray(0x10000)
    for addr, hx in regions:
        blob = bytes.fromhex(hx)
        mem[addr : addr + len(blob)] = blob
    return State(mem)


def test_aim_target_matches_rom():
    total = 0
    for ls, g in _golden().items():
        state = _rebuild(g["regions"])
        ps = g["player_slot"]
        for h, v, cx, cy, tx, ty, vis in g["samples"]:
            got_tx, got_ty, got_los = los.aim_target(state, h, v, cx, cy, ps)
            assert (got_tx, got_ty, int(got_los)) == (
                tx,
                ty,
                vis,
            ), f"landscape {ls} aim h={h} v={v} cursor=({cx},{cy})"
            total += 1
    assert total > 100  # the fixture actually exercised the engine


def test_visible_tiles_are_all_los_positive():
    for _ls, g in _golden().items():
        state = _rebuild(g["regions"])
        ps = g["player_slot"]
        seen = los.visible_tiles(state, ps, max_steps=2000)
        assert seen, "the observer should see at least some tiles"
        # every returned view really does land on its tile with LOS.
        for (tx, ty), view in list(seen.items())[:40]:
            rtx, rty, rlos = los.aim_target(
                state, view["h_angle"], view["v_angle"], *view["cursor"], ps
            )
            assert (rtx, rty) == (tx, ty) and rlos


def test_can_see_agrees_with_centre_view():
    for _ls, g in _golden().items():
        state = _rebuild(g["regions"])
        ps = g["player_slot"]
        seen = los.visible_tiles(state, ps, max_steps=2000)
        for tile in list(seen)[:10]:
            assert los.can_see(state, tile, ps)


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


def test_height_slope_grid_shape():
    g = next(iter(_golden().values()))
    state = _rebuild(g["regions"])
    height, slope = terrain.height_slope_grid(state)
    assert len(height) == mm.N and len(height[0]) == mm.N
    assert all(0 <= height[y][x] <= 15 for y in range(mm.N) for x in range(mm.N))
    assert len(slope) == mm.N
