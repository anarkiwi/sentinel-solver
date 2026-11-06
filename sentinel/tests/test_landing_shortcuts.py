"""The landing sweep's exact short-circuits: a climbing ray lands only on a boulder tile,
and a claim-ordered sweep marches only the rays a consumer keeps."""

import numpy as np
import pytest
from numba import njit, prange

from sentinel import actions, batch, landtable as lt, los, memmap as mm, playerbase
from sentinel import terrain
from sentinel.tests.test_landtable import _landings, _stack_and_transfer

pytestmark = pytest.mark.skipif(not los._HAVE_JIT, reason="numba absent")


@njit(cache=True, parallel=True)
def _old_stop(vx, vy, vz, lo, hi, slope, ex, ey, zseed, max_steps, edge):
    """The pre-short-circuit :func:`landtable._stop_one` over every ray."""
    n = vx.shape[0]
    first = np.full(n, -1, dtype=np.int32)
    ncand = np.zeros(n, dtype=np.int32)
    for j in prange(n):  # pylint: disable=not-an-iterable
        ax, ay, az = np.int64(vx[j]), np.int64(vy[j]), np.int64(vz[j])
        i0 = 1
        while i0 <= max_steps:
            dx = (0x8000 + i0 * ax) >> 16
            dy = (0x8000 + i0 * ay) >> 16
            x = ex + dx
            y = ey + dy
            if x < 0 or x > edge or y < 0 or y > edge:
                break
            _fx, lx = lt._cell_span(ax, dx, max_steps)
            _fy, ly = lt._cell_span(ay, dy, max_steps)
            i1 = min(lx, ly, max_steps)
            zin = (i0 * az) >> 8
            zout = (i1 * az) >> 8
            c = x * 32 + y
            if (x != ex or y != ey) and slope[c] == 0:
                tlo = lo[c] - zseed
                thi = hi[c] - zseed
                zmin = zout if az < 0 else zin
                zmax = zin if az < 0 else zout
                alias = zmax >= lt.WRAP_D + tlo
                if az > 0:
                    cand = alias or (tlo - lt.LAND_BAND < zin <= thi)
                elif az < 0:
                    cand = alias or (zin > tlo - lt.LAND_BAND and zout <= thi)
                else:
                    zmin = 0
                    cand = tlo - lt.LAND_BAND < 0 <= thi
                if cand:
                    if ncand[j] == 0:
                        first[j] = c
                    ncand[j] += 1
                if not alias and zmin <= tlo:
                    break
            i0 = i1 + 1
    return first, ncand


def _tower(st, count=3):
    """``count`` boulders stacked on a bare tile near the player, no transfer."""
    px, py = st.player_xy()
    near = sorted(
        ((px + i, py + j) for i in range(-6, 7) for j in range(-6, 7)),
        key=lambda t: abs(t[0] - px) + abs(t[1] - py),
    )
    tile = next(
        t
        for t in near[5:]
        if 0 <= t[0] < lt.RADIUS
        and 0 <= t[1] < lt.RADIUS
        and terrain.tile_byte(st, *t) < mm.OBJECT_TILE
        and not terrain.tile_byte(st, *t) & 0x0F
    )
    for _ in range(count):
        actions.create(st, mm.T_BOULDER, tile)
    return tile


def _boards(new_state):
    """``(name, state, slot, eye_z)``: boulder-free, stacked-and-transferred, and boulder
    towers seen from a lower eye."""
    bare = new_state(42)
    stacked = new_state(0)
    _stack_and_transfer(stacked)
    low, high = new_state(0), new_state(42)
    _tower(low, 4)
    _tower(high, 2)
    raised = (high.obj_z_height[high.player] + 3) & 0xFF
    return [
        ("bare", bare, bare.player, None),
        ("stacked", stacked, stacked.player, None),
        ("tower", low, low.player, None),
        ("tower-eyez", high, high.player, raised),
    ]


def _stop_args(st, slot, eye_z):
    return (
        int(st.obj_x[slot]),
        int(st.obj_y[slot]),
        lt.seed_z(st, slot, eye_z),
        lt.MAX_STEPS,
        lt.RADIUS,
    )


@pytest.mark.parametrize("coarse", (True, False))
def test_stop_cells_restrict_only_climbing_rays(new_state, coarse):
    grids = lt.lattice(coarse=coarse, v_primary=not coarse)
    comps = lt.cached_components(grids)
    climbing = comps[2] >= 0
    seen_climber = False
    for name, st, slot, eye_z in _boards(new_state):
        lo, hi, flags = lt.surface_maps(st)
        args = _stop_args(st, slot, eye_z)
        new = lt.stop_cells(*comps, lo, hi, flags, *args)
        old = _old_stop(*comps, lo, hi, flags & lt.NEVER_LANDS, *args)
        assert np.array_equal(new[0][~climbing], old[0][~climbing]), name
        assert np.array_equal(new[1][~climbing], old[1][~climbing]), name
        assert (new[1] <= old[1]).all(), name
        boulders = bool(flags[0] & lt.BOARD_BOULDER)
        assert boulders == (name != "bare"), name
        if not boulders:
            assert (new[1][climbing] == 0).all() and (new[0][climbing] == -1).all()
        else:
            kept = climbing & (new[1] > 0)
            assert ((flags[new[0][kept]] & lt.HAS_BOULDER) > 0).all(), name
            seen_climber |= bool(kept.any())
            assert (old[1][climbing] > new[1][climbing]).any(), name
    assert seen_climber


@pytest.mark.parametrize("kind", ("coarse", "plane", "full"))
def test_landings_equal_full_sweep_and_old_prefilter(new_state, kind):
    grids = lt.lattice(v_primary=kind == "plane", coarse=kind == "coarse")
    climbing = lt.cached_components(grids)[2] >= 0
    landed_climber = False
    for name, st, slot, eye_z in _boards(new_state):
        truth = _landings(st, slot, grids, eye_z)
        rays = np.concatenate([np.empty(0, np.int64), *truth.values()])
        tiles = np.repeat(
            np.array([t[0] * mm.N + t[1] for t in truth], dtype=np.int64),
            [len(v) for v in truth.values()],
        )
        by_ray = np.argsort(rays)
        got = lt.landing_pairs(st, slot, eye_z, grids, lt.MAX_STEPS)
        assert np.array_equal(got[0], rays[by_ray]), name
        assert np.array_equal(got[1], tiles[by_ray]), name
        lo, hi, flags = lt.surface_maps(st)
        old = _old_stop(
            *lt.cached_components(grids),
            lo,
            hi,
            flags & lt.NEVER_LANDS,
            *_stop_args(st, slot, eye_z),
        )
        assert np.isin(got[0], np.flatnonzero(old[1] > 0)).all(), name
        landed_climber |= bool(climbing[got[0]].any())
    assert landed_climber or kind == "plane"


def _minima(rays, tiles, keep, drive, n):
    """Per claim group the least-key ray: ``{group: ray}`` of ``keep``'s consumer."""
    if keep == "first":
        group, key = tiles, rays
    else:
        nd = len(drive)
        group = tiles * (n // nd) + rays // nd
        key = drive[rays % nd] * n + rays
    order = np.lexsort((key, group))
    start = np.ones(order.size, dtype=bool)
    start[1:] = group[order][1:] != group[order][:-1]
    return dict(zip(group[order][start].tolist(), rays[order][start].tolist()))


@pytest.mark.parametrize("keep", ("first", "cell"))
@pytest.mark.parametrize("path", ("single", "many", "batch"))
def test_keep_holds_every_group_minimum(new_state, keep, path, monkeypatch):
    grids = lt.lattice(v_primary=True)
    n = lt.cached_components(grids)[0].shape[0]
    drive = playerbase._cursor_drive(grids)
    boards = _boards(new_state)
    sts = [b[1] for b in boards]
    slots = [b[2] for b in boards]
    if path == "single":
        got = [
            lt.landing_pairs(st, sl, None, grids, lt.MAX_STEPS, keep, drive)
            for st, sl in zip(sts, slots)
        ]
    elif path == "many":
        got = lt.landing_pairs_many(sts, slots, None, grids, lt.MAX_STEPS, keep, drive)
    else:
        monkeypatch.setattr(batch, "width", lambda: 1)
        got = batch.landings(sts, grids, slots, None, lt.MAX_STEPS, keep, drive)
    pruned = False
    for (rays, tiles), st, sl in zip(got, sts, slots):
        full = lt.landing_pairs(st, sl, None, grids, lt.MAX_STEPS)
        assert np.isin(rays, full[0]).all()
        assert np.array_equal(tiles, full[1][np.searchsorted(full[0], rays)])
        assert np.all(np.diff(rays) > 0)
        assert _minima(rays, tiles, keep, drive, n) == _minima(*full, keep, drive, n)
        pruned |= rays.size < full[0].size
    assert pruned or (path != "many" and lt._gpu() is not None)  # pylint: disable=W0212


def test_keep_all_is_default_and_plan_validates():
    grids = lt.lattice(coarse=True)
    order, group = lt.claim_plan(grids)
    assert np.array_equal(order, np.arange(order.size)) and (group == -1).all()
    drive = playerbase._cursor_drive(grids)
    order, group = lt.claim_plan(grids, "cell", drive)
    assert np.array_equal(np.sort(order), np.arange(order.size))
    assert np.all(np.diff(group[order]) >= 0)
    key = drive[order % len(drive)] * order.size + order
    assert np.all(np.diff(key.reshape(-1, len(drive)), axis=1) > 0)
    with pytest.raises(ValueError):
        lt.claim_plan(grids, "some")
