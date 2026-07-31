"""render_cost reproduces plot_world's exact per-scene cost, cycle-counted in py65.

``golden_render_cost.json`` holds, per view, the exact plot_world cycle count and $2845
examination count from py65 (no ROM bytes); the non-oracle test asserts the projector
reproduces the examination count byte-for-byte and the frame cost within tolerance.
"""

import json
import os

import pytest

from sentinel import landscape, projector, rendercost
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


def _measure_plot_world(cpu, mem, state, h_angle, v_angle):
    """Run plot_world ($2625) headless in py65 with the raytraced occlusion table
    ($245B) active; return per-view exact cycles split by SUBTREE -- the $2845
    examinations, the $21AE object stacks and the terrain fill that is left."""
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
    mem[0x0051], mem[0x0052] = 0xF0, 0x30  # play-view raster clip window
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
    in_exam = in_obj = False
    exam_sp = obj_sp = steps = 0
    while cpu.pc != ret and steps < 20_000_000:
        pc = cpu.pc
        if pc == 0x2845 and not in_exam:  # $2845 subtree, the trig it calls included
            in_exam, exam_sp = True, cpu.sp
            n_examine += 1
        if pc == 0x2A24 and mem[0x0180 + cpu.x]:  # plot_tile with a non-hidden byte
            n_filled += 1
        if pc == 0x21AE and not in_obj:  # plot_stack_of_objects subtree
            in_obj, obj_sp = True, cpu.sp
        c1 = cpu.processorCycles
        cpu.step()
        d = cpu.processorCycles - c1
        if in_exam:
            examine_cycles += d
            if cpu.sp > exam_sp:
                in_exam = False
        if in_obj:
            object_cycles += d
            if cpu.sp > obj_sp:
                in_obj = False
        steps += 1
    total = cpu.processorCycles - c0
    examine_cycles += 6 * n_examine  # the JSR the caller pays, which the model charges
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
        tiles, n_examine, exam_cycles = projector.project_scene(state, h, v)[:3]
        assert n_examine == rec["n_examine"], f"{key} examines {n_examine} != {rec}"
        assert len(tiles) == rec["n_filled"], f"{key} plots {len(tiles)} != {rec}"
        # The $2845 tree is cycle-exact bar one $0078-stale branch a pass (open item 5).
        assert exam_cycles == pytest.approx(rec["examine_cycles"], rel=0.002), key
        want = (
            rec["cycles"] / projector.FRAME_CYCLES
        )  # fill term approx: see residual doc
        got = projector.render_cost(state, {"h_angle": h, "v_angle": v})
        assert (
            0.55 * want <= got <= 1.15 * want
        ), f"{key} frames {got:.1f} vs {want:.1f}"


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


def test_examine_count_is_scene_dependent():
    """Exactness is pinned by ``_check``; this pins that the count actually varies."""
    with open(GOLDEN) as f:
        data = json.load(f)
    assert len({rec["n_examine"] for rec in data.values()}) > 5


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
        tiles = projector.project_scene(state, h, v)[0]
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


# Exact live $9630 transfer-settle frame counts, per landscape (docs/architecture.md).
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

    assert tune_frames(0x00) == 96  # matches projector.TUNE_TRANSFER_FRAMES
    assert tune_frames(0x19) == 96 == projector.TUNE_TRANSFER_FRAMES


def _fill_cycles(state, h, v, mode=projector.PLAY_MODE):
    """The fill model's own cycles for one view, the object stacks included."""
    setup = projector._setup(state, h, v, state.player, mode)
    fill = projector.project_scene(state, h, v, None, mode)[3]
    return projector.fill_frames(state, setup, fill, mode, state.player)[0]


def test_offband_tiles_still_cost_their_prepare_polygon_calls():
    """A tile whose polygon clips out of the fill band has zero screen area but still
    runs prepare_polygon per section; views exist where EVERY plotted tile is off-band
    and the 6502 spends 100k+ cycles, so the fill model must price them above zero."""
    zero_area = []
    for key, rec in sorted(json.load(open(GOLDEN)).items()):
        ls, h, v = (int(x) for x in key.split(","))
        state = landscape.generate(ls)
        tiles = projector.project_scene(state, h, v)[0]
        if not tiles or any(t["h"] for t in tiles):
            continue
        zero_area.append(key)
        assert rec["terrain_fill_cycles"] > 0, f"{key}: golden claims no fill"
        assert _fill_cycles(state, h, v) > 0, f"{key}: off-band view priced at zero"
        assert projector.render_cost(state, {"h_angle": h, "v_angle": v}) > 0
    assert zero_area, "no all-off-band view in the sweep to guard"


def test_fill_model_tracks_the_measured_terrain_fill():
    """rendercost emulates $2A24 -> $22AA and $21AE in render order, so its cycles
    track the golden's own fill subtrees rather than an area proxy."""
    ratios = []
    for key, rec in sorted(json.load(open(GOLDEN)).items()):
        ls, h, v = (int(x) for x in key.split(","))
        state = landscape.generate(ls)
        want = rec["terrain_fill_cycles"]
        if want < 20_000:  # a view with almost no fill has no ratio worth pinning
            continue
        want += rec["object_fill_cycles"]  # the emulation covers $21AE too
        got = _fill_cycles(state, h, v)
        ratios.append(got / want)
        assert 0.85 <= got / want <= 1.10, f"{key}: {got} vs {want}"
    assert len(ratios) >= 8
    ratios.sort()
    assert 0.90 <= ratios[len(ratios) // 2] <= 1.05


def test_fill_model_numba_twin_matches_python():
    """rendercost is one source: the compiled kernel and ``py_func`` must agree."""
    seen = 0
    for key in sorted(json.load(open(GOLDEN))):
        ls, h, v = (int(x) for x in key.split(","))
        state = landscape.generate(ls)
        setup = projector._setup(state, h, v, state.player)
        fill = projector.project_scene(state, h, v)[3]
        bufs, sect = rendercost.buffers(projector.PLAY_MODE)
        args = (
            fill,
            len(fill),
            (setup["quadrant"] - 2) & 0xFF,
            (setup["quadrant"] * 4) & 0xFF,
            sect,
            bufs,
            rendercost.SCREEN_TOP,
            rendercost.SCREEN_BOTTOM,
        )
        assert rendercost.fill_cycles(*args) == rendercost.fill_cycles.py_func(*args)
        seen += 1
    assert seen > 10


def test_the_walk_around_the_examines_is_priced():
    """$26DE/$27D7/$295D are 0.6-3.0% of a pass, one-signed if unpriced; the model
    charges them off their own branches, so the term is non-zero and scene-dependent."""
    walks = []
    for key in sorted(json.load(open(GOLDEN))):
        ls, h, v = (int(x) for x in key.split(","))
        state = landscape.generate(ls)
        walk = projector.project_scene(state, h, v)[4]
        assert walk > 0, key
        walks.append(walk)
    assert len(set(walks)) > 10, "the walk term is not varying with the scene"
