"""Acceptance tests for the keyboard aim-landability oracle (:mod:`sentinel.los`
``landable_views`` / ``landable_view`` / ``landable_sweep_with_centres``).

Buildability == AIM-landability: a real keyboard aim (body h in 8-notches, sights
cursor on its 9px grid, body v_angle) lands the sights ray on the tile with LOS.
This is a STRICT subset of geometric visibility (``relative.can_see_object`` /
``los.visible_tiles`` over-report far/low tiles). Validated against ground-truth
human-win telemetry (``out/play_*.jsonl``, base64 ``mem`` per record): from every
exact pre-FIRE state (the LAST record while standing on the from-tile, after the
player has absorbed/built and is about to transfer), the tile actually built on
must be aim-landable.

Logs are gitignored fixtures; each test skips cleanly when they are absent.
"""

import base64
import json
import os

import pytest

from sentinel import landscape, los
from sentinel.state import State

ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
LS0 = os.path.join(ROOT, "out", "play_20260707_193356.jsonl")
LS335 = os.path.join(ROOT, "out", "play_20260707_203210.jsonl")

# The exact aim-landable set from the real ls0 start (8,17,eye5) -- a regression lock on the
# oracle. Now the v-swept set (body v_angle swept over PITCH_BAND as well as the sights
# cursor): 48 tiles, a strict superset of the old v=$F5-only 17-tile set (the extra tiles are
# near/below builds the player pitches the body down for). Reflects the documented pitched-v
# lattice reduction (los.CURSOR_CX/CY_PITCHED): the full pitched grid yields one more tile,
# (17,14); the reduced grid drops that single marginal extra to fit the 60s solve budget while
# keeping every ground-truth build landable. A geometric-visibility test reports far more still.
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
    (18, 13),
    (18, 14),
    (19, 14),
    (20, 14),
}


def _state_from_record(rec):
    raw = base64.b64decode(rec["mem"])
    mem = bytearray(0x10000)
    mem[0 : len(raw)] = raw
    return State(mem)


def _records(path):
    with open(path, encoding="utf-8") as fh:
        lines = [ln for ln in fh if ln.strip()]
    return [json.loads(ln) for ln in lines[1:]]  # line 0 is the header


def _fire_ladder(path):
    """Consecutive (from_tile, to_tile, pre_FIRE_record) triples up to the first win,
    keeping the LAST record at each from-tile (the state right before the transfer --
    the player builds while standing, so the first-arrival state is stale)."""
    recs = _records(path)
    cut = len(recs)
    for i, r in enumerate(recs):
        if r.get("done_flag"):
            cut = i
            break
    recs = recs[:cut]
    seq = []
    for r in recs:
        pl = r.get("player")
        if not pl:
            continue
        xy = (pl["x"], pl["y"])
        if seq and seq[-1][0] == xy:
            seq[-1] = (xy, r)
        else:
            seq.append((xy, r))
    return [
        (seq[i][0], seq[i + 1][0], seq[i][1])
        for i in range(len(seq) - 1)
        if seq[i + 1][0] != seq[i][0]
    ]


def _tile_is_bare(state, tile):
    """A fresh build lands on a BARE tile; a move onto an already-occupied tile is a
    transfer to a standing synthoid (a backtrack), for which aim-landability from the
    previous tile does not apply."""
    from sentinel import terrain, memmap as mm

    return terrain.tile_byte(state, tile[0], tile[1]) < mm.OBJECT_TILE


def test_generated_landscape_oracle_invariants():
    """CI-safe (no external fixture): on the deterministic landscape 0 board, the aim
    lattice sweep and the single-tile query agree, and every landable tile is also
    geometrically visible (aim-landability is a STRICT subset of visible_tiles)."""
    st = landscape.generate(0)
    views = los.landable_views(st, st.player)
    assert views, "expected some aim-landable tiles from the start"
    # views/view agreement: every swept tile resolves via the single-tile query too,
    # and the sweep's recorded aim actually lands on its tile with LOS.
    for tile, view in list(views.items())[:8]:
        assert los.landable_view(st, tile, st.player) is not None
        tx, ty, ok = los.aim_target(
            st,
            view["h_angle"],
            view["v_angle"],
            view["cursor"][0],
            view["cursor"][1],
            st.player,
        )
        assert ok and (tx, ty) == tile
    # subset of geometric visibility (visible_tiles sweeps the centred-cursor lattice;
    # the keyboard sweep is at least as reachable via the same underlying march).
    vis = set(los.visible_tiles(st, st.player))
    assert set(views) & vis, "landable and visible sets should overlap"
    # sweep-with-centres returns the same view keys.
    sviews, centres = los.landable_sweep_with_centres(st, st.player)
    assert set(sviews) == set(views)
    assert all(0 <= c <= 0xFF for c in centres.values())


@pytest.mark.skipif(not os.path.exists(LS0), reason="ls0 human-win log absent")
def test_ls0_start_landable_set_exact():
    st = _state_from_record(_records(LS0)[0])
    got = set(los.landable_views(st, st.player))
    assert got == LS0_START_LANDABLE


@pytest.mark.skipif(not os.path.exists(LS0), reason="ls0 human-win log absent")
def test_ls0_forward_builds_all_aim_landable():
    """Every fresh forward build in the ls0 human win is aim-landable (v=$F5) from its
    exact pre-fire state -- 0 false-negatives, the property the planner relies on."""
    for frm, to, rec in _fire_ladder(LS0):
        st = _state_from_record(rec)
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
    for frm, to, rec in _fire_ladder(LS335):
        if (frm, to) != ((11, 17), (11, 18)):
            continue
        st = _state_from_record(rec)
        views = los.landable_views(st, st.player)
        assert (
            to in views
        ), "v-complete landable_views must contain the pitched-down build"
        assert views[to]["v_angle"] != los.KBD_V_ANGLE  # reached via a pitched body v
        return
    pytest.skip("ls335 opening (11,17)->(11,18) not present in log")


def test_batched_sweep_matches_per_probe_aim_target():
    """CI-safe bit-exactness lock: the batched numba lattice march used by landable_views /
    landable_sweep_with_centres returns IDENTICAL (tx, ty, los) and tile-centre fraction to
    calling aim_target once per aim, over the WHOLE v-complete keyboard lattice."""
    if not los._HAVE_JIT:
        pytest.skip("numba not available -- batched march path not exercised")
    st = landscape.generate(0)
    slot = st.player
    hgrid = list(range(0, 256, los.AZIMUTH_STEP))
    status, tx, ty, centre, meta = los._landable_batch(
        st,
        slot,
        None,
        6000,
        hgrid,
        los._V_PRIORITY,
        los.CURSOR_CX,
        los.CURSOR_CY,
        los.CURSOR_CX_PITCHED,
        los.CURSOR_CY_PITCHED,
        los.KBD_V_ANGLE,
    )
    verdict_bad = 0
    centre_bad = 0
    for i, (h, v, cx, cy) in enumerate(meta):
        atx, aty, alos, acen = los.aim_target(
            st, h, v, cx, cy, slot, max_steps=6000, return_centre=True
        )
        blos = status[i] == los.los_jit.LOS_CLEAR
        if (int(tx[i]), int(ty[i]), blos) != (atx, aty, alos):
            verdict_bad += 1
        if blos and int(centre[i]) != acen:
            centre_bad += 1
    assert verdict_bad == 0 and centre_bad == 0, (verdict_bad, centre_bad)
