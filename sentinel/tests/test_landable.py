"""Acceptance tests for the keyboard aim-landability oracle (:mod:`sentinel.los`
``landable_views`` / ``landable_view`` / ``landable_sweep_with_centres``).

Buildability == AIM-landability: a real keyboard aim (body h in 8-notches, sights
cursor at 1px resolution, body v_angle) lands the sights ray on the tile with LOS.
This is a STRICT subset of geometric visibility (``threat.player_sees_tile``
over-reports far/low tiles). Validated against ground-truth
human-win telemetry (``out/play_*.jsonl``, base64 ``mem`` per record): from every
exact pre-FIRE state (the LAST record while standing on the from-tile, after the
player has absorbed/built and is about to transfer), the tile actually built on
must be aim-landable.

Logs are gitignored fixtures; each test skips cleanly when they are absent.
"""

import os

import pytest

from sentinel import astar_player, los, memmap as mm, threat
from sentinel.astar_player import AStarPlayer
from sentinel.tests.telemetry import log_path, records, state_from_record, tile_ladder

LS0 = log_path("play_20260707_193356.jsonl")
LS335 = log_path("play_20260707_203210.jsonl")

# The exact aim-landable set from the real ls0 start (8,17,eye5) -- a regression lock on the
# oracle. The v-swept set (body v_angle over PITCH_BAND x the 1px sights-cursor window): 50
# tiles. The sights cursor is now enumerated at the ROM's true 1px resolution ($9965/$9994 step
# +/-1px; each 1px step a distinct ray via prepare_vector_from_player_sights $1C10), swept over a
# 64px window per axis that is bit-equivalent to the full cx[16,143] x cy[32,159] range. This
# faithful resolution adds (22,12) -- a far tile the old 9px notch grid false-negatived. A
# geometric-visibility test reports far more still.
LS0_START_LANDABLE = {
    (7, 10),
    (7, 11),
    (7, 12),
    (7, 13),
    (7, 14),
    (8, 9),
    (8, 10),
    (8, 11),
    (8, 12),
    (8, 16),
    (9, 7),
    (9, 8),
    (9, 9),
    (9, 10),
    (9, 11),
    (9, 12),
    (9, 14),
    (10, 12),
    (10, 14),
    (10, 15),
    (10, 16),
    (10, 17),
    (10, 19),
    (11, 12),
    (11, 14),
    (11, 15),
    (11, 16),
    (11, 17),
    (11, 19),
    (12, 12),
    (12, 14),
    (12, 19),
    (13, 14),
    (13, 16),
    (14, 13),
    (14, 14),
    (14, 16),
    (15, 13),
    (15, 14),
    (15, 16),
    (16, 13),
    (16, 14),
    (16, 16),
    (17, 13),
    (17, 14),
    (18, 13),
    (18, 14),
    (19, 14),
    (20, 14),
    (22, 12),
}


def _tile_is_bare(state, tile):
    """A fresh build lands on a BARE tile; a move onto an already-occupied tile is a
    transfer to a standing synthoid (a backtrack), for which aim-landability from the
    previous tile does not apply."""
    from sentinel import terrain, memmap as mm

    return terrain.tile_byte(state, tile[0], tile[1]) < mm.OBJECT_TILE


def test_generated_landscape_oracle_invariants(new_state):
    """CI-safe (no external fixture): on the deterministic landscape 0 board, the aim
    lattice sweep and the single-tile query agree, and every landable tile is also
    geometrically visible (aim-landability is a STRICT subset of the plotted scene)."""
    st = new_state(0)
    views = los.landable_views(st, st.player)
    assert views, "expected some aim-landable tiles from the start"
    # every aim-landable tile is also geometrically visible: aim-landability is a
    # STRICT subset of the ROM observer->tile march (threat.player_sees_tile).
    # views/view agreement: every swept tile resolves via the single-tile query too,
    # and the sweep's recorded aim actually lands on its tile with LOS.
    for tile, view in list(views.items())[:8]:
        assert los.landable_view(st, tile, st.player, v_band=True) is not None
        tx, ty, ok = los.aim_target(
            st,
            view["h_angle"],
            view["v_angle"],
            view["cursor"][0],
            view["cursor"][1],
            st.player,
        )
        assert ok and (tx, ty) == tile
    # strict subset of geometric visibility: every landable tile passes the ROM
    # observer->tile march that threat.player_sees_tile distils.
    for tile in list(views)[:8]:
        assert threat.player_sees_tile(
            st, tile, st.player
        ), f"landable tile {tile} is not geometrically visible"
    # sweep-with-centres returns the same view keys.
    sviews, centres = los.landable_sweep_with_centres(st, st.player)
    assert set(sviews) == set(views)
    assert all(0 <= c <= 0xFF for c in centres.values())


def test_targeted_view_matches_full_board_sweep(new_state):
    """The heading-cone single-tile band march (:func:`los.landable_view_targeted`, the A*
    planner's fallback) is bit-identical to ``landable_views(st).get(tile)`` for every tile,
    and returns None off the landable set -- the g-invariance the node-cost win relies on.
    """
    st = new_state(0)
    full = los.landable_views(st, st.player)
    for tile, view in full.items():
        assert los.landable_view_targeted(st, tile) == view, tile
    for tile in ((0, 0), (30, 30), (2, 2), (28, 5)):
        if tile not in full:
            assert los.landable_view_targeted(st, tile) is None, tile


def test_targeted_view_matches_coarse_lattice_sweep(new_state):
    """On the A* planner's SUBSAMPLED cursor lattice, the per-tile heading cone agrees with
    that lattice's own full sweep on every tile of the board -- the membership equality
    ``AStarPlayer._landable`` substitutes for ``tile in _landset``."""
    st = new_state(0)
    coarse = AStarPlayer._coarse_landable(st)
    cxs, cys = astar_player._COARSE_CX, astar_player._COARSE_CY
    for x in range(mm.N):
        for y in range(mm.N):
            got = los.landable_view_targeted(st, (x, y), cxs=cxs, cys=cys) is not None
            assert got == ((x, y) in coarse), (x, y)


@pytest.mark.skipif(not os.path.exists(LS0), reason="ls0 human-win log absent")
def test_ls0_start_landable_set_exact():
    st = state_from_record(records(LS0)[0])
    got = set(los.landable_views(st, st.player))
    assert got == LS0_START_LANDABLE


@pytest.mark.skipif(not os.path.exists(LS0), reason="ls0 human-win log absent")
def test_ls0_forward_builds_all_aim_landable():
    """Every fresh forward build in the ls0 human win is aim-landable (v=$F5) from its
    exact pre-fire state -- 0 false-negatives, the property builds rely on."""
    for frm, to, rec in tile_ladder(LS0):
        st = state_from_record(rec)
        if not _tile_is_bare(st, to):  # skip transfer-to-existing (backtrack)
            continue
        views = los.landable_views(st, st.player)
        assert to in views, f"ls0 build {frm}->{to} not aim-landable"


@pytest.mark.skipif(not os.path.exists(LS335), reason="ls335 human-win log absent")
def test_ls335_adjacent_build_now_landable():
    """The ls335 opening (11,17)->(11,18) is an ADJACENT (below) build the human fired with
    the body pitched DOWN (v=225). It is NOT landable with v fixed at $F5, but IS once
    landable_views sweeps the body v_angle DOF -- so the v-complete oracle now returns it
    directly (it used to need the landable_view(..., v_band=True) fallback)."""
    for frm, to, rec in tile_ladder(LS335):
        if (frm, to) != ((11, 17), (11, 18)):
            continue
        st = state_from_record(rec)
        views = los.landable_views(st, st.player)
        assert (
            to in views
        ), "v-complete landable_views must contain the pitched-down build"
        assert views[to]["v_angle"] != los.KBD_V_ANGLE  # reached via a pitched body v
        return
    pytest.skip("ls335 opening (11,17)->(11,18) not present in log")


def test_prep_vec_matches_python():
    """The numba ray-vector builder (los_jit._prep_vec / build_lattice, used by both
    aim_target and the batched sweep) is BIT-for-bit identical to the pure-Python reference
    prepare_vector_from_player_sights $1C10 over the keyboard lattice."""
    if not los._HAVE_JIT:
        pytest.skip("numba not available")
    bad = 0
    for h in range(0, 256, los.AZIMUTH_STEP):
        for v in (los.KBD_V_ANGLE, 0xCD, 0x35, 0x11, 0xE1):
            for cx in (16, 48, 80, 111, 143):
                for cy in (32, 63, 95, 126, 159):
                    vec = los.prepare_vector_from_player_sights(None, h, v, cx, cy, 0)
                    a = tuple(int(x) for x in los.los_jit._prep_vec(h, v, cx, cy))
                    exp = (
                        vec.vx_lo,
                        vec.vx_hi,
                        vec.vz_lo,
                        vec.vz_hi,
                        vec.vy_lo,
                        vec.vy_hi,
                        vec.s30,
                    )
                    if exp != a:
                        bad += 1
    assert bad == 0


def test_batched_sweep_matches_per_probe_aim_target(new_state):
    """CI-safe bit-exactness lock: the batched numba lattice march used by landable_views /
    landable_sweep_with_centres returns IDENTICAL (tx, ty, los) and tile-centre fraction to
    calling aim_target once per aim, over a representative slice of the v-complete lattice.
    """
    if not los._HAVE_JIT:
        pytest.skip("numba not available -- batched march path not exercised")
    st = new_state(0)
    slot = st.player
    # A representative slice (a few h notches x the whole v band x the 1px cursor window) --
    # the full lattice is ~3.5M aims; this keeps the per-probe reference loop fast while still
    # exercising flat->(h,v,cx,cy) reconstruction, all pitches and the cursor edges.
    hgrid = [0, 64, 128]
    status, tx, ty, centre, grids = los._landable_batch(
        st, slot, None, 6000, hgrid, los._V_PRIORITY, los.CURSOR_CX, los.CURSOR_CY
    )
    verdict_bad = 0
    centre_bad = 0
    for i in range(status.shape[0]):
        h, v, cx, cy = los._meta_at(i, *grids)
        atx, aty, alos, acen = los.aim_target(
            st, h, v, cx, cy, slot, max_steps=6000, return_centre=True
        )
        blos = status[i] == los.los_jit.LOS_CLEAR
        if (int(tx[i]), int(ty[i]), blos) != (atx, aty, alos):
            verdict_bad += 1
        if blos and int(centre[i]) != acen:
            centre_bad += 1
    assert verdict_bad == 0 and centre_bad == 0, (verdict_bad, centre_bad)


def test_window_equals_full_1px_cursor(new_state):
    """The 64px step-1 cursor WINDOW (los.CURSOR_CX/CY) is BIT-EQUIVALENT to the full 1px ROM
    cursor range (cx[16,143] x cy[32,159], los.CURSOR_CX_FULL/CY_FULL): the full v-band sweep
    returns the EXACT same landable tile set AND per-tile min tile-centre fraction.  Body-h
    (step 8) + cx>>3 over 64px tiles the h-integer, and body-v (step 4) + (cy-5)>>4 over 64px
    tiles the v-integer, so the wider cursor range only re-reaches rays a different body notch
    already produced.  (Proven on the generated board; the recorded-win states match too.)
    """
    if not los._HAVE_JIT:
        pytest.skip("numba not available")
    st = new_state(0)
    slot = st.player
    hgrid = list(range(0, 256, los.AZIMUTH_STEP))

    def sweep(vgrid, cxs, cys):
        status, tx, ty, centre, _ = los._landable_batch(
            st, slot, None, 6000, hgrid, vgrid, cxs, cys
        )
        best = {}
        for i in range(status.shape[0]):
            if status[i] != los.los_jit.LOS_CLEAR:
                continue
            tile = (int(tx[i]), int(ty[i]))
            c = int(centre[i])
            if tile not in best or c < best[tile]:
                best[tile] = c
        return best

    def diff(a, b):
        return (
            set(b) - set(a),
            set(a) - set(b),
            {t: (a.get(t), b.get(t)) for t in set(a) & set(b) if a[t] != b[t]},
        )

    # Full v-band: cx WINDOW x cy WINDOW == full 1px cursor (body-v step 4 fills the cy gaps).
    win = sweep(los._V_PRIORITY, los.CURSOR_CX, los.CURSOR_CY)
    full = sweep(los._V_PRIORITY, los.CURSOR_CX_FULL, los.CURSOR_CY_FULL)
    assert win == full, diff(win, full)

    # Single $F5 plane (v_primary): NO body-v, so cy must be FULL; cx WINDOW still suffices.
    v1 = [los.KBD_V_ANGLE]
    p_win = sweep(v1, los.CURSOR_CX, los.CURSOR_CY_FULL)
    p_full = sweep(v1, los.CURSOR_CX_FULL, los.CURSOR_CY_FULL)
    assert p_win == p_full, diff(p_win, p_full)
