"""The whole-board landable-view sweep against the per-tile path it replaced.

``_ref_cheapest_ray``/``_ref_cheap_view`` are that path in miniature: one targeted march
per tile and a branch-and-bound over :func:`playerbase.aim_frames`.  Every tile must get
the same view AND winning ray, since ``_Views`` readers see the dict in that ray order.
"""

import functools
import math

import numpy as np
import pytest

from sentinel import landtable as lt, los, playerbase
from sentinel.game import Game
from sentinel.phase_player import PhasePlayer

pytestmark = pytest.mark.skipif(
    not los._HAVE_JIT, reason="numba absent: the sweep defers to the exact path"
)


def _ref_cheapest_ray(st, rays, grids, aim_from, observer):
    cell, nu, ns, dv, cur = playerbase._lattice_terms(rays, grids, aim_from)
    order = np.lexsort((rays, cur, cell))
    cs = cell[order]
    head = order[np.concatenate(([True], cs[1:] != cs[:-1]))]
    reps = rays[head]
    bound = playerbase._aim_bound(nu[head], ns[head], dv[head], cur[head])
    best_frames, best = math.inf, -1
    for j in np.argsort(bound, kind="stable"):
        if bound[j] > best_frames:
            break
        ray = int(reps[j])
        frames = playerbase.aim_frames(
            st, playerbase._view_at(ray, grids), aim_from, observer
        )
        if frames < best_frames or (frames == best_frames and ray < best):
            best_frames, best = frames, ray
    return best


def _ref_cheap_view(st, tile, grids, aim_from):
    slot = st.player
    if lt.never_lands(st, tile, slot):
        return None, -1
    idx = lt.candidates(st, tile, slot, None, grids, lt.MAX_STEPS)
    if not idx.size:
        return None, -1
    status, tx, ty = lt._march(st, slot, None, grids, idx, lt.MAX_STEPS)
    hit = np.flatnonzero(
        (status == los.los_jit.LOS_CLEAR) & (tx == tile[0]) & (ty == tile[1])
    )
    if not hit.size:
        return None, -1
    ray = _ref_cheapest_ray(st, idx[hit], grids, aim_from, slot)
    return playerbase._view_at(ray, grids), ray


@functools.cache
def _played(number, actions):
    player = PhasePlayer(Game.typed(number), verbose=False)
    if actions:
        player.run(max_actions=actions)
    return player.st


def _board(number, actions):
    """Landscape ``number`` after the phase player's first ``actions`` decision ticks."""
    return _played(number, actions).clone()


BOARDS = [(0, 0), (42, 0), (335, 0), (1442, 0), (42, 6), (335, 10), (1442, 8)]


@pytest.mark.parametrize("v_primary", [True, False])
@pytest.mark.parametrize("number,actions", BOARDS)
def test_sweep_matches_the_per_tile_path_on_every_tile(number, actions, v_primary):
    st = _board(number, actions)
    slot = st.player
    aim_from = (st.obj_h_angle[slot], st.obj_v_angle[slot])
    grids = lt.lattice(v_primary=v_primary)
    got = playerbase._cheap_views(st, v_primary, aim_from)
    want = {}
    for tile in playerbase.BOARD_TILES:
        view, ray = _ref_cheap_view(st, tile, grids, aim_from)
        assert got.get(tile) == view, tile
        if view is not None:
            want[tile] = ray
    assert list(got) == sorted(want, key=want.get)


@pytest.mark.parametrize("v_primary", [True, False])
@pytest.mark.parametrize("skew", [(0, 0), (3, 0), (-5, 0x14)])
def test_aim_table_prices_every_cell_as_aim_frames(skew, v_primary):
    """Every (bearing, pitch) cell, from aligned and unaligned facings: the table plus the
    cursor term IS :func:`playerbase.aim_frames`, bit for bit."""
    st = _board(42, 6)
    slot = st.player
    aim_from = (
        (int(st.obj_h_angle[slot]) + skew[0]) & 0xFF,
        (int(st.obj_v_angle[slot]) + skew[1]) & 0xFF,
    )
    grids = lt.lattice(v_primary=v_primary)
    hgrid, vgrid, cxs, cys = grids
    cells = np.arange(len(hgrid) * len(vgrid))
    table = playerbase.aim_table(st, cells, grids, aim_from, slot)
    per_cell = len(cxs) * len(cys)
    rng = np.random.default_rng([len(vgrid), skew[0] & 0xFF])
    rays = cells * per_cell + rng.integers(0, per_cell, cells.size)
    cur = playerbase._lattice_terms(rays, grids, aim_from)[4]
    got = playerbase.aim_total(
        table, np.where(cur > 0, cur + playerbase.CURSOR_RAMP, 0.0)
    )
    want = [
        playerbase.aim_frames(st, playerbase._view_at(r, grids), aim_from, slot)
        for r in rays
    ]
    assert got.tolist() == want


def test_views_pin_the_board_of_their_first_query():
    """A caller acts while holding ``_Views``: lookups after it moves on still read the
    board each lattice was first asked on."""
    player = PhasePlayer(Game.typed(42), verbose=False)
    views = playerbase._Views(player.st)
    plane = dict(views.primary())
    band = dict(views.band())
    player.run(max_actions=6)
    assert views.primary() == plane and views.band() == band
    for tile in playerbase.BOARD_TILES:
        assert views.get(tile, band=True) == plane.get(tile, band.get(tile))
    assert views.band_ordered(lambda t: t[0] % 2) == [
        (t, v) for t, v in band.items() if t[0] % 2
    ]
