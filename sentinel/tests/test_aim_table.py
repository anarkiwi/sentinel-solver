"""The whole-board landable-view sweep against the per-tile reference path.

``_ref_cheapest_ray`` is that path in miniature: a scalar march of every lattice ray and a
branch-and-bound over :func:`aimref.aim_frames`.  Every tile must get the same view AND
winning ray, since ``_Views`` readers see the dict in that ray order.
"""

import functools
import math

import numpy as np
import pytest

from sentinel import landtable as lt, los, memmap as mm, views
from sentinel.tests import aimref

pytestmark = pytest.mark.skipif(
    not los._HAVE_JIT, reason="numba absent: the sweep defers to the exact path"
)


def _ref_cheapest_ray(st, rays, grids, aim_from, observer):
    cell, nu, ns, dv, cur = views._lattice_terms(rays, grids, aim_from)
    order = np.lexsort((rays, cur, cell))
    cs = cell[order]
    head = order[np.concatenate(([True], cs[1:] != cs[:-1]))]
    reps = rays[head]
    bound = views._aim_bound(nu[head], ns[head], dv[head], cur[head])
    best_frames, best = math.inf, -1
    for j in np.argsort(bound, kind="stable"):
        if bound[j] > best_frames:
            break
        ray = int(reps[j])
        frames = aimref.aim_frames(st, views._view_at(ray, grids), aim_from, observer)
        if frames < best_frames or (frames == best_frames and ray < best):
            best_frames, best = frames, ray
    return best


def _ref_cheap_views(st, grids, aim_from):
    """``{tile: (view, ray)}``: every lattice ray marched on the scalar path, then each
    tile's cheapest landing ray."""
    slot = st.player
    idx = np.arange(np.prod([len(g) for g in grids]), dtype=np.int64)
    status, tx, ty = lt._march(st, slot, None, grids, idx, lt.MAX_STEPS)
    clear = status == los.los_jit.LOS_CLEAR
    out = {}
    for tile in {(int(x), int(y)) for x, y in zip(tx[clear], ty[clear])}:
        hit = clear & (tx == tile[0]) & (ty == tile[1])
        ray = _ref_cheapest_ray(st, idx[hit], grids, aim_from, slot)
        out[tile] = (views._view_at(ray, grids), ray)
    return out


@functools.cache
def _boards(number, boulders):
    return aimref.board(number, boulders)


def _board(number, boulders):
    """Landscape ``number`` with ``boulders`` boulders placed (:func:`aimref.board`)."""
    return _boards(number, boulders).clone()


BOARDS = [(0, 0), (42, 0), (335, 0), (1442, 0), (42, 3), (335, 2), (1442, 1)]


@pytest.mark.parametrize("v_primary", [True, False])
@pytest.mark.parametrize("number,boulders", BOARDS)
def test_sweep_matches_the_per_tile_path_on_every_tile(number, boulders, v_primary):
    st = _board(number, boulders)
    slot = st.player
    aim_from = (st.obj_h_angle[slot], st.obj_v_angle[slot])
    grids = lt.lattice(v_primary=v_primary)
    got = views._sweep(st, v_primary, aim_from).views()
    want = _ref_cheap_views(st, grids, aim_from)
    assert got == {t: v for t, (v, _r) in want.items()}
    assert list(got) == sorted(want, key=lambda t: want[t][1])


@pytest.mark.parametrize("v_primary", [True, False])
@pytest.mark.parametrize("number,boulders", [(335, 0), (42, 3)])
def test_eager_lattice_pricing_picks_the_branch_and_bound_winners(
    number, boulders, v_primary
):
    st = _board(number, boulders)
    me = st.player
    aim_from = (st.obj_h_angle[me], st.obj_v_angle[me])
    lazy = views._sweep(st, v_primary, aim_from)
    eager = views._sweep(st, v_primary, aim_from)
    eager._price_cells(eager.cell)
    assert eager.known[eager.cell].all()
    assert list(eager.views().items()) == list(lazy.views().items())


@pytest.mark.parametrize("v_primary", [True, False])
@pytest.mark.parametrize("skew", [(0, 0), (3, 0), (-5, 0x14)])
def test_aim_table_prices_every_cell_as_aim_frames(skew, v_primary):
    """Every (bearing, pitch) cell, from aligned and unaligned facings: the table plus the
    cursor term IS :func:`aimref.aim_frames`, bit for bit."""
    st = _board(42, 3)
    slot = st.player
    aim_from = (
        (int(st.obj_h_angle[slot]) + skew[0]) & 0xFF,
        (int(st.obj_v_angle[slot]) + skew[1]) & 0xFF,
    )
    grids = lt.lattice(v_primary=v_primary)
    hgrid, vgrid, cxs, cys = grids
    cells = np.arange(len(hgrid) * len(vgrid))
    table = views.aim_table(st, cells, grids, aim_from, slot)
    per_cell = len(cxs) * len(cys)
    rng = np.random.default_rng([len(vgrid), skew[0] & 0xFF])
    rays = cells * per_cell + rng.integers(0, per_cell, cells.size)
    cur = views._lattice_terms(rays, grids, aim_from)[4]
    got = views.aim_total(table, np.where(cur > 0, cur + views.CURSOR_RAMP, 0.0))
    want = [
        aimref.aim_frames(st, views._view_at(r, grids), aim_from, slot) for r in rays
    ]
    assert got.tolist() == want


def test_views_pin_the_board_of_their_first_query():
    """A caller acts while holding ``_Views``: lookups after it moves on still read the
    board each lattice was first asked on."""
    st = _board(42, 0)
    held = views._Views(st)
    plane = dict(held.primary())
    band = dict(held.band())
    moved = _board(42, 3)
    st.mem[:] = moved.mem
    assert held.primary() == plane and held.band() == band
    for tile in ((x, y) for x in range(mm.N) for y in range(mm.N)):
        assert held.get(tile, band=True) == plane.get(tile, band.get(tile))


def test_group_min_matches_sorted_reps():
    """The dense-table reduction keeps, per (tile, cell), the first key of the sorted run."""
    rng = np.random.default_rng(7)
    ncell, dsz = 32, 40
    drive_of = rng.integers(0, 60, dsz)
    rays = np.sort(rng.choice(ncell * dsz, 400, replace=False))
    tiles = rng.integers(0, 1024, rays.size)
    rb = (ncell * dsz - 1).bit_length()
    db = int(drive_of.max()).bit_length()
    cell, off = np.divmod(rays, dsz)
    key = np.sort(
        (((tiles * ncell + cell) << db | drive_of[off]) << rb) | rays.astype(np.int64)
    )
    want = key[views._starts(key >> (rb + db))]
    for _ in range(2):
        got = views._group_min(rays, tiles, drive_of, ncell, rb, db)
        assert np.array_equal(got, want)
