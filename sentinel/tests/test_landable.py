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

# The exact aim-landable set from the real ls0 start (8,17,eye5) -- a regression lock
# on the oracle (17 tiles; a geometric-visibility test reports far more).
LS0_START_LANDABLE = {
    (7, 10),
    (7, 11),
    (8, 9),
    (8, 10),
    (8, 11),
    (9, 7),
    (9, 8),
    (9, 9),
    (9, 10),
    (9, 11),
    (10, 12),
    (11, 12),
    (12, 12),
    (14, 16),
    (15, 16),
    (16, 16),
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
def test_ls335_adjacent_build_needs_v_band():
    """The ls335 opening (11,17)->(11,18) is an ADJACENT (below) build the human fired
    with the body pitched DOWN (v=225). It is NOT landable with v fixed at $F5, but IS
    with the v_angle DOF -- locking in that v_angle is a real aim degree of freedom."""
    for frm, to, rec in _fire_ladder(LS335):
        if (frm, to) != ((11, 17), (11, 18)):
            continue
        st = _state_from_record(rec)
        assert to not in los.landable_views(st, st.player)  # v=$F5 misses it
        assert los.landable_view(st, to, st.player, v_band=True) is not None
        return
    pytest.skip("ls335 opening (11,17)->(11,18) not present in log")
