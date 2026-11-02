"""Keyboard-aim pricing for many boards at once, over arrays.

:func:`aim_parts` is :func:`playerbase.aim_table` for a batch of boards: every pan notch
and u-turn redraw is one pricing job, deduplicated on scene, observer and notch.
"""

import numpy as np

from sentinel import aimcost, memmap as mm, pancost, projector, settlecost
from sentinel.playerbase import CURSOR_RAMP, SCROLL, TAP_FRAMES, _sum_step

_PAN_DELTA = np.array(pancost.PAN_DELTA, dtype=np.int64)
UTURN_KIND = 4
_PLANS = np.stack(
    [
        projector.window_plan(pancost.PAN_MODE[d], None, pancost.PAN_ROWS[d])
        for d in range(UTURN_KIND)
    ]
    + [projector.window_plan(projector.PLAY_MODE, None, None)]
)
_CLEAR = np.array([pancost.CLEAR_FRAMES[0 if m else 1] for m in pancost.PAN_MODE])
_SCROLL = np.array([SCROLL[0 if d < pancost.V_UP else 1] for d in range(4)])


def _distance(a, b):
    """:func:`aimcost.angle_dist` over arrays."""
    return np.abs(((b - a) + 128) % 256 - 128)


def press_plan(h0, h1):
    """:func:`aimcost.h_press_count` over arrays: ``(u-turn presses, step presses)``."""
    d = _distance(h0, h1) // aimcost.AZIMUTH_STEP
    turn = (1 + (aimcost.UTURN_STEP - d)) < d
    return turn.astype(np.int64), np.where(turn, aimcost.UTURN_STEP - d, d)


def notch_plots(h0, v0, h1, v1):
    """:func:`pancost.notch_plots` over arrays: ``(valid, direction, plot h, plot v,
    bearing steps)``, the first four ``(n, K)`` with row ``i`` listing the notches of the
    aim ``(h0[i], v0[i])`` to ``(h1[i], v1[i])`` in order and ``valid`` masking padding.
    """
    turn, ns = press_plan(h0, h1)
    nv = _distance(v0, v1) // aimcost.PITCH_STEP
    j = np.arange(int((ns + nv).max(initial=0)))[None, :]
    start = np.where(turn > 0, h0 ^ 0x80, h0)
    step = np.where(((h1 - start) & 0xFF) < 0x80, 1, -1)
    along = (start[:, None] + step[:, None] * aimcost.AZIMUTH_STEP * j) & 0xFF
    bearing = j < ns[:, None]
    d_h = np.where(step > 0, pancost.H_RIGHT, pancost.H_LEFT)[:, None]
    after = (start + step * aimcost.AZIMUTH_STEP * ns) & 0xFF
    vstep = np.where(((v1 - v0) & 0xFF) < 0x80, 1, -1)
    d_v = np.where(vstep > 0, pancost.V_UP, pancost.V_DOWN)[:, None]
    tilt = v0[:, None] + vstep[:, None] * aimcost.PITCH_STEP * (j - ns[:, None])
    return (
        j < (ns + nv)[:, None],
        np.where(bearing, d_h, d_v),
        np.where(bearing, (along + _PAN_DELTA[d_h]) & 0xFF, after[:, None]),
        np.where(bearing, v0[:, None], ((tilt & 0xFF) + _PAN_DELTA[d_v]) & 0xFF),
        ns,
    )


def _grid_table(grids):
    """``(h, v)`` padded ``(len(grids), widest)`` int64 tables of the lattices' bearing and
    pitch grids, and each lattice's bearing count."""
    hgrid = np.zeros((len(grids), max(len(g[0]) for g in grids)), dtype=np.int64)
    vgrid = np.zeros((len(grids), max(len(g[1]) for g in grids)), dtype=np.int64)
    for i, g in enumerate(grids):
        hgrid[i, : len(g[0])], vgrid[i, : len(g[1])] = g[0], g[1]
    return hgrid, vgrid, np.array([len(g[0]) for g in grids], dtype=np.int64)


def _notch_costs(mems, rows, job_row, job_obs, direction, plot_h, plot_v):
    """``(notch frames, u-turn frames)``: every notch job's frames, and for each of
    ``rows`` its u-turn redraw's, from one launch over the distinct jobs."""
    obs = mems[rows, mm.PLAYER_OBJECT].astype(np.int64)
    flip_h = (mems[rows, mm.OBJECTS_H_ANGLE + obs].astype(np.int64) ^ 0x80) & 0xFF
    flip_v = mems[rows, mm.OBJECTS_V_ANGLE + obs].astype(np.int64)
    scene = np.zeros(len(mems), dtype=np.int64)
    scene[rows] = np.unique(projector.scene_keys(mems[rows]), return_inverse=True)[1]
    ident = np.concatenate(
        [
            np.column_stack([scene[job_row], job_obs, direction, plot_h, plot_v]),
            np.column_stack(
                [scene[rows], obs, np.full(len(rows), UTURN_KIND), flip_h, flip_v]
            ),
        ]
    )
    uniq, first, back = np.unique(ident, axis=0, return_index=True, return_inverse=True)
    cycles = projector.price_columns(
        mems,
        np.concatenate([job_row, rows])[first],
        uniq[:, 1],
        uniq[:, 3],
        uniq[:, 4],
        _PLANS[uniq[:, 2]],
    )
    cost = projector._cost_of(cycles)[back.reshape(-1)]  # pylint: disable=W0212
    notch = (cost[: len(job_row)] + _CLEAR[direction]) + _SCROLL[direction]
    return notch, settlecost.uturn_from_cost(cost[len(job_row) :])


def aim_parts(mems, row, lat, h0, v0, cells, grids):
    """``(pan, turn, uturn)`` per request ``q``: the pan frames and u-turn term
    :func:`playerbase.aim_table` prices for ``cells[q]`` of lattice ``grids[lat[q]]`` from
    facing ``(h0[q], v0[q])`` on board image ``mems[row[q]]``, and that board's u-turn
    frames.  ``pan`` and ``turn`` are aligned with ``cells[q]``."""
    count = np.fromiter(map(len, cells), np.int64, len(row))
    req = np.repeat(np.arange(len(row)), count)
    cell = np.concatenate(cells).astype(np.int64)
    hgrid, vgrid, nh = _grid_table(grids)
    vi, hi = np.divmod(cell, nh[lat[req]])
    h1, v1 = hgrid[lat[req], hi], vgrid[lat[req], vi]
    start_h, start_v = h0[req], v0[req]
    vs = _distance(start_v, v1) // aimcost.PITCH_STEP
    up = (v1 - start_v) % 256 < 0x80
    turn_c, ns_c = press_plan(start_h, h1)

    group, back = np.unique((req * hgrid.shape[1] + hi) * 2 + up, return_inverse=True)
    back = back.reshape(-1)
    far = np.zeros(group.size, dtype=np.int64)
    np.maximum.at(far, back, vs)
    far_v = np.zeros(group.size, dtype=np.int64)
    top = vs == far[back]
    far_v[back[top]] = v1[top]
    one = np.zeros(group.size, dtype=np.int64)
    one[back] = np.arange(cell.size)
    valid, direction, plot_h, plot_v, _ns = notch_plots(
        start_h[one], start_v[one], h1[one], far_v
    )

    rows = np.unique(row)
    g_at, j_at = np.nonzero(valid)
    job_row = row[req[one]][g_at]
    notch, uturn = _notch_costs(
        mems,
        rows,
        job_row,
        mems[job_row, mm.PLAYER_OBJECT].astype(np.int64),
        direction[g_at, j_at],
        plot_h[g_at, j_at],
        plot_v[g_at, j_at],
    )
    grid = np.zeros(valid.shape)
    grid[g_at, j_at] = notch
    total, comp = np.zeros(group.size), np.zeros(group.size)
    sums = np.zeros((group.size, valid.shape[1] + 1))
    for k in range(valid.shape[1]):
        total, comp = _sum_step(total, comp, grid[:, k])
        sums[:, k + 1] = total + comp
    at_row = np.searchsorted(rows, row)
    ends = np.cumsum(count)[:-1]
    turn = np.where(turn_c > 0, uturn[at_row[req]], 0.0)
    return (
        np.split(sums[back, ns_c + vs], ends),
        np.split(turn, ends),
        uturn[at_row],
    )


def aim_total_columns(toggle, turn, pan, cur):
    """:func:`playerbase.aim_frames` over columns: the compensated sum of the toggle,
    u-turn, pan and cursor-drive terms (``cur`` the drive distance, 0 when parked)."""
    total, comp = np.zeros(len(pan)), np.zeros(len(pan))
    drive = np.where(cur > 0, cur + CURSOR_RAMP, 0.0)
    for term in (toggle, turn, pan, drive + TAP_FRAMES):
        total, comp = _sum_step(total, comp, term)
    return total + comp
