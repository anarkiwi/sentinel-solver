"""render_cost reproduces plot_world's exact per-scene cost, cycle-counted in py65.

``golden_render_cost.json`` holds, per view, the exact plot_world cycle count and $283D
examination count from py65 under play conditions (no ROM bytes); the non-oracle test
asserts the projector reproduces the examination count byte-for-byte and the frame cost
within tolerance.
"""

import json
import os

import pytest

import numpy as np

from sentinel import landscape, objmodel, projector, rendercost, settlecost
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


_MARKS = frozenset((0x283D, 0x2A24, 0x21AE))  # the examine, plot_tile and object stack


def _measure_plot_world(cpu, mem, h_angle, v_angle):
    """Run plot_world ($2625) headless in py65 on a machine :func:`_prepared` has already
    put in the $357D render context; return per-view exact cycles split by SUBTREE --
    the $283D examinations, the $21AE object stacks and the terrain fill left over."""
    player = mem[0x000B]
    mem[0x006E] = player
    mem[0x09C0 + player] = h_angle  # objects_h_angle
    mem[0x0140 + player] = v_angle  # objects_v_angle
    ret = 0xFFF0
    mem[ret] = 0x60
    sp = cpu.sp
    mem[0x0100 + sp] = (ret - 1) >> 8
    mem[0x0100 + ((sp - 1) & 0xFF)] = (ret - 1) & 0xFF
    cpu.sp = (sp - 2) & 0xFF
    cpu.pc = 0x2625
    c0 = cpu.processorCycles
    n_examine = n_filled = examine_cycles = object_cycles = 0
    exam_sp = obj_sp = exam_c0 = obj_c0 = -1
    hits = [0] * 0x10000
    step = cpu.step
    for _ in range(20_000_000):
        pc = cpu.pc
        if pc == ret:
            break
        hits[pc] += 1
        if pc in _MARKS:
            if pc == 0x283D and exam_sp < 0:  # the examine, either branch of $2840
                exam_sp, exam_c0 = cpu.sp, cpu.processorCycles
                n_examine += 1
            elif pc == 0x2A24 and mem[0x0180 + cpu.x]:  # plot_tile, byte not hidden
                n_filled += 1
            elif pc == 0x21AE and obj_sp < 0:  # plot_stack_of_objects subtree
                obj_sp, obj_c0 = cpu.sp, cpu.processorCycles
        step()
        if 0 <= exam_sp < cpu.sp:
            examine_cycles += cpu.processorCycles - exam_c0
            exam_sp = -1
        if 0 <= obj_sp < cpu.sp:
            object_cycles += cpu.processorCycles - obj_c0
            obj_sp = -1
    total = cpu.processorCycles - c0
    examine_cycles += 6 * n_examine  # the JSR the caller pays, which the model charges
    return {
        "cycles": total,
        "examine_cycles": examine_cycles,
        "object_fill_cycles": object_cycles,
        "terrain_fill_cycles": total - examine_cycles - object_cycles,
        "n_examine": n_examine,
        "n_filled": n_filled,
        "geometry": {  # the loops whose iteration counts the fill model must reproduce
            "span_rows": hits[0x2377],
            "span_bytes": sum(hits[a] for a in range(0x23DE, 0x2458, 4)),
            "steep": hits[0x2F58],
            "shallow": hits[0x2FA1],
            "wide_steep": hits[0x3113],
            "wide_shallow": hits[0x316D],
            "edges": hits[0x2EE4],
            "sections": hits[0x3002],
        },
    }


_PREPARED = {}


def _prepared(landscape_number):
    """A board generated and put in the $357D render context, on a fresh machine.

    $245B, $3700 and $2993 read no view angle -- two prepares of one board at different
    (h, v) differ only in the angle bytes -- so one prepare serves all of its views."""
    image = _PREPARED.get(landscape_number)
    if image is None:
        cpu, mem, state = oracle.generate_machine(landscape_number)
        mem[0x006E] = mem[0x000B]
        oracle.prepare_render_context(cpu, mem, state)
        image = _PREPARED[landscape_number] = bytes(mem)
    return oracle.wrap_image(image)[:2]


def _build_golden():
    """{ 'ls,h,v': {cycles, examine/object/terrain-fill cycles, counts} } over VIEWS."""
    out = {}
    for ls, h, v in VIEWS:
        cpu, mem = _prepared(ls)
        out[f"{ls},{h},{v}"] = _measure_plot_world(cpu, mem, h, v)
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
    """Decode the real $3E80/$24DA bitmap after the $357D render prologue in py65."""
    mem[0x006E] = mem[0x000B]
    oracle.prepare_render_context(cpu, mem, state)
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


def test_viewpoint_settle_lands_in_live_settle_band():
    """settlecost.viewpoint_settle_frames on the sweep views lands in the live
    259-460f transfer band.  These sweep views are NOT the live transfers' own
    scenes, so only the band is claimed; the per-transfer accuracy against
    settle_oracle_ls42.json is test_settle_accuracy.py's."""
    lo, hi = min(min(v) for v in _LIVE_SETTLES.values()), max(
        max(v) for v in _LIVE_SETTLES.values()
    )
    floor = settlecost.BRACKET_FRAMES + settlecost.TUNE_FRAMES[0x19]
    for ls in _LIVE_SETTLES:
        state = landscape.generate(ls)
        for _lsk, h, v in (view for view in VIEWS if view[0] == ls):
            f = settlecost.viewpoint_settle_frames(
                state, eye_view={"h_angle": h, "v_angle": v}
            )
            assert f >= floor  # the $35D5 tune wait is a hard lower bound
            assert 0.75 * lo <= f <= 1.25 * hi, f"ls{ls} {h},{v}: {f:.1f} out of band"


@pytest.mark.oracle
def test_transfer_tune_is_96_frames():
    """TUNE_TRANSFER_FRAMES is ROM-derived: decode the #$19 ($AB69) and #$0 ($AB50)
    $34DE tune tables from the image -- both sum to 96 note-hold frames ($0C70=(b-$C8)*4,
    $0CDF ticked once/frame); the u-turn's #$28 tune decodes to 24, settlecost's floor.
    """
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
    assert settlecost.TUNE_FRAMES[0x19] == 96.0
    assert tune_frames(0x28) == 24 == settlecost.TUNE_FRAMES[0x28]


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


@pytest.mark.skipif(
    not objmodel.available(), reason="the object stacks need the game image"
)
def test_fill_model_tracks_the_measured_terrain_fill():
    """rendercost emulates $2A24 -> $22AA and $21AE in render order, so its cycles
    track the golden's own fill subtrees rather than an area proxy.  The golden's
    object_fill_cycles are part of the claim, so the $21AE emulation -- and with it the
    model geometry out of the game image -- has to be there to make it."""
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


def _geometry(state, h, v):
    """The fill model's own loop counts for one view, the object stacks included."""
    from sentinel import objectcost

    setup = projector._setup(state, h, v, state.player, projector.PLAY_MODE)
    fill = projector.project_scene(state, h, v)[3]
    bufs, sect = rendercost.buffers(projector.PLAY_MODE)
    tile_work, obj_work = rendercost.tile_workspace(), objectcost.workspace()
    left, right = rendercost.edge_tables()
    objectcost.pass_cycles(
        np.frombuffer(state.mem, dtype=np.uint8),
        objectcost.scratch_zp(),
        objectcost.model_arrays(),
        fill,
        len(fill),
        (setup["quadrant"] - 2) & 0xFF,
        (setup["quadrant"] * 4) & 0xFF,
        sect,
        bufs,
        rendercost.SCREEN_TOP,
        rendercost.SCREEN_BOTTOM,
        state.player,
        state.player,
        h,
        v,
        tile_work,
        obj_work,
        left,
        right,
    )
    rows = tile_work[4] + obj_work[4]
    flags = tile_work[5] + obj_work[5]
    return {
        "span_rows": int(flags[rendercost.F_SPAN_ROWS]),
        "span_bytes": int(flags[rendercost.F_SPAN_BYTES]),
        "steep": int(rows[rendercost.R_STEEP]),
        "shallow": int(rows[rendercost.R_SHALLOW]),
        "wide_steep": int(rows[rendercost.R_WIDE_STEEP]),
        "wide_shallow": int(rows[rendercost.R_WIDE_SHALLOW]),
        "edges": int(rows[rendercost.R_N_EDGE]),
        "sections": int(rows[rendercost.R_N_SECT]),
    }


@pytest.mark.skipif(
    not objmodel.available(), reason="the object stacks need the game image"
)
def test_fill_geometry_matches_the_rom_loop_counts():
    """The residual is a geometry question, so pin the geometry: the model's span rows,
    filled bytes and DDA iterations against the ROM's own $2377/$23DC/$2F58/$2FA1/
    $3113/$316D/$2EE4/$3002 hit counts. Exact on 14 of 15; 2024,16,0 is the object
    vertex angle ([5](../../docs/open_items.md)), not the fill.
    """
    exact = 0
    for key, rec in sorted(json.load(open(GOLDEN)).items()):
        ls, h, v = (int(x) for x in key.split(","))
        want = rec["geometry"]
        got = _geometry(landscape.generate(ls), h, v)
        exact += got == want
        for name, w in want.items():
            assert abs(got[name] - w) <= max(
                2, 0.01 * w
            ), f"{key} {name}: {got[name]} != {w}"
    assert exact >= 14, f"only {exact} views reproduce the ROM's loop counts"
