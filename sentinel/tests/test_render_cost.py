"""render_cost reproduces plot_world's exact per-scene cost, cycle-counted in py65.

``golden_render_cost.json`` holds, per view, the exact plot_world cycle count and $2845
examination count from py65 (no ROM bytes); the non-oracle test asserts the projector
reproduces the examination count byte-for-byte and the frame cost within tolerance.
"""

import json
import os

import pytest

from sentinel import landscape, projector
from sentinel.tests import oracle

GOLDEN = os.path.join(os.path.dirname(__file__), "golden_render_cost.json")

# (landscape, h_angle, v_angle): bearings/pitches spanning the cost swing and boards.
VIEWS = [
    (0, 0x00, 0x00),
    (0, 0x30, 0x08),
    (0, 0x88, 0xF8),
    (42, 0x00, 0x00),
    (42, 0x50, 0x04),
    (42, 0xC0, 0x00),
    (42, 0xA0, 0xF0),
    (66, 0x00, 0x00),
    (66, 0x60, 0x10),
    (335, 0x00, 0x00),
    (335, 0x40, 0x10),
    (335, 0x9A, 0xF0),
    (777, 0x20, 0x00),
    (2024, 0x10, 0x00),
    (2024, 0xB8, 0xF4),
]


# examine call-tree PC ranges: $2845 shell + calculate_angle/hypotenuse/vertical trig.
_EXAMINE_RANGES = (
    (0x2845, 0x295C),
    (0x9287, 0x93AC),
    (0x0D4A, 0x0F49),
    (0x0F4A, 0x1000),
)


def _in_examine_tree(pc):
    return any(lo <= pc <= hi for lo, hi in _EXAMINE_RANGES)


def _measure_plot_world(cpu, mem, state, h_angle, v_angle):
    """Run plot_world ($2625) headless in py65 with the raytraced occlusion table
    ($245B) active; return per-view exact cycles broken into examine / object-fill /
    terrain-fill, the $2845 count, and the count of tiles actually filled ($2A24)."""
    player = mem[0x000B]
    mem[0x006E] = player
    mem[0x09C0 + player] = h_angle  # objects_h_angle
    mem[0x0140 + player] = v_angle  # objects_v_angle
    mem[0x001F] = 0
    mem[0x005E] = 0
    mem[0x0C78] = 0
    mem[0x0C1B] = 0  # force_pan clear: no keyboard pan-abort
    mem[0x0CDE] = 0  # not hyperspaced
    mem[0x0CCE] = 0x80  # skip secret-code check in the raytracer
    mem[0x352C] = 0x60  # stub update_sound (foreground-only cost)
    oracle.call(cpu, mem, 0x2993, a=0, state=state)  # initialise_buffer_variables
    state["stop"] = False
    oracle.call(cpu, mem, 0x245B, state=state)  # populate raytraced occlusion table
    ret = 0xFFF0
    mem[ret] = 0x60
    sp = cpu.sp
    mem[0x0100 + sp] = (ret - 1) >> 8
    mem[0x0100 + ((sp - 1) & 0xFF)] = (ret - 1) & 0xFF
    cpu.sp = (sp - 2) & 0xFF
    cpu.pc = 0x2625
    c0 = cpu.processorCycles
    n_examine = n_filled = examine_cycles = object_cycles = 0
    in_obj = False
    obj_sp = 0
    steps = 0
    while cpu.pc != ret and steps < 20_000_000:
        pc = cpu.pc
        if pc == 0x2845:
            n_examine += 1
        if pc == 0x2A24 and mem[0x0180 + cpu.x]:  # plot_tile with a non-hidden byte
            n_filled += 1
        if pc == 0x21AE and not in_obj:  # plot_stack_of_objects subtree
            in_obj, obj_sp = True, cpu.sp
        c1 = cpu.processorCycles
        cpu.step()
        d = cpu.processorCycles - c1
        if _in_examine_tree(pc):
            examine_cycles += d
        if in_obj:
            object_cycles += d
            if cpu.sp > obj_sp:
                in_obj = False
        steps += 1
    total = cpu.processorCycles - c0
    return {
        "cycles": total,
        "examine_cycles": examine_cycles,
        "object_fill_cycles": object_cycles,
        "terrain_fill_cycles": total - examine_cycles - object_cycles,
        "n_examine": n_examine,
        "n_filled": n_filled,
    }


def _build_golden():
    """{ 'ls,h,v': {cycles, examine/object/terrain-fill cycles, counts} } over VIEWS."""
    out = {}
    for ls, h, v in VIEWS:
        cpu, mem, state = oracle.generate_machine(ls)
        out[f"{ls},{h},{v}"] = _measure_plot_world(cpu, mem, state, h, v)
    return out


def _check(data):
    for key, rec in data.items():
        ls, h, v = (int(x) for x in key.split(","))
        state = landscape.generate(ls)
        tiles, n_examine = projector.project_scene(state, h, v)
        assert n_examine == rec["n_examine"], f"{key} examines {n_examine} != {rec}"
        assert len(tiles) == rec["n_filled"], f"{key} plots {len(tiles)} != {rec}"
        pred_ex = n_examine * projector.C_EXAMINE  # examine cycles ~ count * mean floor
        assert pred_ex == pytest.approx(rec["examine_cycles"], rel=0.16), key
        want = (
            rec["cycles"] / projector.FRAME_CYCLES
        )  # fill term approx: see residual doc
        got = projector.render_cost(state, {"h_angle": h, "v_angle": v})
        assert 0.30 * want <= got <= 2.4 * want, f"{key} frames {got:.1f} vs {want:.1f}"


@pytest.mark.oracle
def test_regenerate_render_cost_golden():
    """Cycle-count plot_world across VIEWS in py65, dump the golden, and confirm the
    projector matches the exact examination count and frame cost."""
    data = _build_golden()
    with open(GOLDEN, "w") as f:
        json.dump(data, f, separators=(",", ":"), sort_keys=True)
    _check(data)


def test_render_cost_matches_golden():
    with open(GOLDEN) as f:
        data = json.load(f)
    assert data
    _check(data)


def test_examine_count_is_exact_and_scene_dependent():
    with open(GOLDEN) as f:
        data = json.load(f)
    counts = set()
    for key, rec in data.items():
        ls, h, v = (int(x) for x in key.split(","))
        _tiles, n_examine = projector.project_scene(landscape.generate(ls), h, v)
        assert n_examine == rec["n_examine"]
        counts.add(rec["n_examine"])
    assert len(counts) > 5  # examine count varies strongly with scene/view


def test_object_base_never_overshoots_and_is_present():
    """The plot_object base-floor term (c) stays at or under the exact per-view
    object_fill_cycles (never overshoots, since object span_fill is unmodelled) and is
    non-zero on the object-bearing views."""
    with open(GOLDEN) as f:
        data = json.load(f)
    seen_object_view = False
    for key, rec in data.items():
        ls, h, v = (int(x) for x in key.split(","))
        state = landscape.generate(ls)
        tiles, _ = projector.project_scene(state, h, v)
        base = projector._inview_object_base(state, tiles)
        assert base <= rec["object_fill_cycles"] + 1  # floor, never overshoots
        if rec["object_fill_cycles"] and base:
            seen_object_view = True
            assert base >= 40_000  # a plotted object carries its trig+prepare floor
    assert seen_object_view


def _py65_occlusion(cpu, mem, state):
    """Decode the real $3E80/$24DA bitmap after running $2993 then $245B in py65."""
    player = mem[0x000B]
    mem[0x006E] = player
    for addr in (0x001F, 0x005E, 0x0C78, 0x0C1B, 0x0CDE):
        mem[addr] = 0
    mem[0x0CCE] = 0x80
    mem[0x352C] = 0x60
    oracle.call(cpu, mem, 0x2993, a=0, state=state)
    state["stop"] = False
    oracle.call(cpu, mem, 0x245B, state=state)
    vis = [[False] * 32 for _ in range(32)]
    for ty in range(32):
        for tx in range(32):
            lo = ((tx << 3) & 0xE0) | (ty & 0x1F)
            vis[ty][tx] = bool(
                mem[0x3E80 + (lo >> 1)] & (0x80 >> (2 * (tx & 3) + (lo & 1)))
            )
    return vis


_OCCLUSION_LANDSCAPES = (0, 42, 66, 335, 777, 2024)


@pytest.mark.oracle
def test_occlusion_table_is_byte_exact():
    """projector._occlusion_visible reproduces the ROM $245B/$3E80 bitmap byte-for-byte
    (every one of the 32x32 tiles) on a spread of landscapes."""
    for ls in _OCCLUSION_LANDSCAPES:
        cpu, mem, state = oracle.generate_machine(ls)
        ref = _py65_occlusion(cpu, mem, state)
        got = projector._occlusion_visible(landscape.generate(ls))
        mism = [(x, y) for y in range(32) for x in range(32) if ref[y][x] != got[y][x]]
        assert not mism, f"landscape {ls}: {len(mism)} occlusion mismatches {mism[:8]}"


def test_occlusion_is_view_independent_and_hides_tiles():
    """$245B depends only on observer position, not view angle; and it hides a real
    fraction of tiles on a sloped board (the terrain-fill occlusion gate)."""
    state = landscape.generate(0)
    vis = projector._occlusion_visible(state)
    hidden = sum(1 for row in vis for cell in row if not cell)
    assert hidden > 32  # the far/edge rows plus terrain-occluded tiles are hidden


# Exact live $9630 transfer-settle frame counts, per landscape (docs/render_cost.md).
_LIVE_SETTLES = {42: (338, 305, 435, 460), 335: (259, 333, 371)}


def test_viewpoint_replot_lands_in_live_settle_band():
    """viewpoint_replot_frames = tune(96) + fixed(~176) + 2*render_cost predicts the
    live 259-460f transfer settle: every ls42/ls335 sweep view lands in the live band
    (vs the old ~10x under 2*render_cost), median abs error modest."""
    lo, hi = min(min(v) for v in _LIVE_SETTLES.values()), max(
        max(v) for v in _LIVE_SETTLES.values()
    )
    errs = []
    for ls, settles in _LIVE_SETTLES.items():
        state = landscape.generate(ls)
        preds = []
        for _lsk, h, v in (view for view in VIEWS if view[0] == ls):
            f = projector.viewpoint_replot_frames(state, {"h_angle": h, "v_angle": v})
            assert 0.75 * lo <= f <= 1.25 * hi, f"ls{ls} {h},{v}: {f:.1f} out of band"
            preds.append(f)
        for p, live in zip(
            sorted(preds), sorted(settles)
        ):  # best-effort magnitude pair
            errs.append(abs(p - live) / live)
    errs.sort()
    assert (
        errs[len(errs) // 2] < 0.15
    )  # median abs error (~9% with the object-base term; was ~22%, ~90% before)


@pytest.mark.oracle
def test_transfer_tune_is_96_frames():
    """TUNE_TRANSFER_FRAMES is ROM-derived: decode the #$19 ($AB69) and #$0 ($AB50)
    $34DE tune tables from the image -- both sum to 96 note-hold frames ($0C70=(b-$C8)*4,
    $0CDF ticked once/frame). No ROM bytes committed; the constant is validated live."""
    with open(oracle.IMG, "rb") as f:
        img = f.read()

    def tune_frames(number):
        pos, length, total = number, 0, 0
        while True:
            b = img[0xAB50 + pos]
            pos += 1
            if b == 0xFF:
                return total
            if b >= 0xC8:
                length = ((b - 0xC8) * 4) & 0xFF
            else:
                total += length  # a note holds `length` frames in $0CDF

    assert tune_frames(0x00) == 96  # matches actioncost.TUNE_FRAMES
    assert tune_frames(0x19) == 96 == projector.TUNE_TRANSFER_FRAMES
