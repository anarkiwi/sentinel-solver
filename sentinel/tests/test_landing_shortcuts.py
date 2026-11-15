"""The landing sweep's exact short-circuits: a climbing ray lands only on a boulder tile,
and a claim-ordered sweep marches only the rays a consumer keeps."""

import numpy as np
import pytest

from sentinel import actions, batch, landtable as lt, los, memmap as mm
from sentinel import views
from sentinel import terrain
from sentinel.tests.test_landtable import _stack_and_transfer

pytestmark = pytest.mark.skipif(not los._HAVE_JIT, reason="numba absent")


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
    drive = views._cursor_drive(grids)
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
    drive = views._cursor_drive(grids)
    order, group = lt.claim_plan(grids, "cell", drive)
    assert np.array_equal(np.sort(order), np.arange(order.size))
    assert np.all(np.diff(group[order]) >= 0)
    key = drive[order % len(drive)] * order.size + order
    assert np.all(np.diff(key.reshape(-1, len(drive)), axis=1) > 0)
    with pytest.raises(ValueError):
        lt.claim_plan(grids, "some")
