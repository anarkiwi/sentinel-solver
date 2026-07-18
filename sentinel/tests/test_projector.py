"""The terrain render-projector reproduces plot_world's per-tile projection ($2845).

``golden_projector.json`` holds, per landscape and view, the ROM's per-grid-point
$2845 outputs captured via py65 with occlusion disabled; the non-oracle test replays
them with :mod:`sentinel.projector` and asserts byte-exact equality.
"""

import json
import os

import pytest

from sentinel import landscape, projector
from sentinel.tests import oracle

GOLDEN = os.path.join(os.path.dirname(__file__), "golden_projector.json")

# (landscape, h_angle, v_angle) views: a spread of bearings/pitches and boards.
VIEWS = [
    (0, 0x00, 0x00),
    (0, 0x30, 0x08),
    (0, 0x88, 0xF8),
    (42, 0x00, 0x00),
    (42, 0x50, 0x04),
    (42, 0xC0, 0x00),
    (42, 0xA0, 0xF0),
    (777, 0x20, 0x00),
    (777, 0x70, 0x0C),
    (2024, 0x10, 0x00),
    (2024, 0xB8, 0xF4),
]

# $2845 output tables (indexed by column, buffer half $0005 = 0).
_SX_LO = 0x0BA0  # plottables_relative_h_angle_low
_SX_HI = 0xA800
_SY_HI = 0x0AE0  # plottables_screen_y_high
_SY_LO = 0x0A80  # plottables_screen_y_low
_VIS = 0x0180  # tile_raytrace_visibility_table


def _drive_rom_view(cpu, mem, state, h_angle, v_angle):
    """Run plot_world's setup live, then $2845 over every (col,row); return the
    captured per-tile outputs and the setup ZP the ROM produced."""
    player = mem[0x000B]
    mem[0x006E] = player  # observer slot
    mem[0x09C0 + player] = h_angle  # objects_h_angle
    mem[0x0140 + player] = v_angle  # objects_v_angle
    mem[0x001F] = 0
    mem[0x005E] = 0
    mem[0x0C78] = 0
    oracle.call(cpu, mem, 0x2993, a=0, state=state)  # initialise_buffer_variables
    for a in range(0x3E80, 0x3F00):
        mem[a] = 0xFF  # disable the occlusion bitmap
    state["stop"] = False
    oracle.call(
        cpu, mem, 0x2625, state=state, stop_pc=0x26D6
    )  # setup, stop at row loop
    setup_zp = {str(a): mem[a] for a in (0x0003, 0x001D, 0x001C, 0x0020, 0x001F)}
    setup_zp["quadrant"] = (mem[0x001C] >> 6) & 3
    rows = {}
    for row in range(32):
        mem[0x0026] = row
        mem[0x0005] = 0
        for col in range(32):
            for a in range(0x3E80, 0x3F00):
                mem[a] = 0xFF
            mem[0x0078] = 0  # $0D4A divide result_low, pure per relative.py
            state["stop"] = False
            oracle.call(cpu, mem, 0x2845, y=col, state=state)
            rows[f"{col},{row}"] = [
                mem[_SX_LO + col],
                mem[_SX_HI + col],
                mem[_SY_LO + col],
                mem[_SY_HI + col],
                mem[_VIS + col],
                cpu.a,
            ]
    return {"setup": setup_zp, "tiles": rows}


def _build_golden():
    """{ 'ls,h,v': {setup, tiles} } across VIEWS, driven through the real 6502."""
    out = {}
    for ls, h, v in VIEWS:
        cpu, mem, state = oracle.generate_machine(ls)
        out[f"{ls},{h},{v}"] = _drive_rom_view(cpu, mem, state, h, v)
    return out


def _replay(state, setup, rec, key):
    """Assert the pure-Python setup + $2845 match one golden record; return count."""
    assert setup["c3"] == rec["setup"]["3"], f"{key} setup c3"
    assert setup["c1d"] == rec["setup"]["29"], f"{key} setup c1d"
    assert setup["ref_hi"] == rec["setup"]["32"], f"{key} setup ref_hi"
    assert setup["quadrant"] == rec["setup"]["quadrant"], f"{key} setup quadrant"
    n = 0
    for tk, want in rec["tiles"].items():
        col, row = (int(x) for x in tk.split(","))
        got = list(projector._project(state, setup, col, row))
        assert got == want, f"{key} tile {tk}: {got} != {want}"
        n += 1
    return n


@pytest.mark.oracle
def test_regenerate_and_match_rom_live():
    """Drive the real $2845 across VIEWS, dump the golden, and confirm the pure
    Python projector matches the ROM live, byte for byte."""
    data = _build_golden()
    with open(GOLDEN, "w") as f:
        json.dump(data, f, separators=(",", ":"), sort_keys=True)
    for key, rec in data.items():
        ls, h, v = (int(x) for x in key.split(","))
        state = landscape.generate(ls)
        _replay(state, projector._setup(state, h, v, state.player), rec, key)


def test_projector_matches_golden():
    with open(GOLDEN) as f:
        data = json.load(f)
    assert data
    checked = 0
    for key, rec in data.items():
        ls, h, v = (int(x) for x in key.split(","))
        state = landscape.generate(ls)
        setup = projector._setup(state, h, v, state.player)
        checked += _replay(state, setup, rec, key)
    assert checked > 1000


def test_render_cost_is_scene_dependent():
    s0 = landscape.generate(0)
    s42 = landscape.generate(42)
    c0 = projector.render_cost(s0, {"h_angle": 0, "v_angle": 0})
    c42 = projector.render_cost(s42, {"h_angle": 0, "v_angle": 0})
    assert c0 > 0 and c42 > 0
    assert projector.render_cost(s0, None) == 0.0
    assert projector.render_cost(s0, {"h_angle": None}) == 0.0


def test_visible_tiles_and_replot():
    s = landscape.generate(0)
    p = s.player
    h, v = s.obj_h_angle[p], s.obj_v_angle[p]
    tiles = projector.visible_tiles(s, h, v)
    # plot_tile ($2A24) draws every nonzero-$0180 tile, incl off-screen ones that clip in the rasteriser.
    assert tiles and all(t["tile_byte"] for t in tiles)
    assert all(t["h"] >= 0 and t["w"] >= 0 for t in tiles)
    settle = projector.viewpoint_replot_frames(s, {"h_angle": h, "v_angle": v})
    single = projector.render_cost(s, {"h_angle": h, "v_angle": v})
    base = projector.TUNE_TRANSFER_FRAMES + projector.SETTLE_FIXED_FRAMES
    assert settle == base + projector.REPLOT_PASSES * single
    assert (
        base <= settle <= 700
    )  # tune+fixed base .. live 259-460f (docs/render_cost.md)
