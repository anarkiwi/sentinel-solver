"""Keyboard views: per landable tile, the frame-cheapest lattice aim from a facing.

A lattice ray is a (bearing, pitch, cursor) aim; :class:`_Landings` keeps, per tile, the
ray whose aim prices cheapest under :func:`aim_table` (toggle, u-turn, pan notches) plus
the cursor drive, and :class:`_Views` answers per-tile view queries on one board.
"""

import os

import numpy as np
from numba import njit

from sentinel import (
    aimcost,
    batch,
    kern,
    landtable,
    los,
    pancost,
    projector,
    settlecost,
)
from sentinel import memmap as mm

H_SCROLL = 16  # $10EE: 16-step horizontal scroll per +-8 bearing notch
V_SCROLL = 8  # $1135: 8-step vertical scroll per +-4 pitch notch
SCROLL = (H_SCROLL, V_SCROLL)  # indexed by notch axis
_VIEW_CACHE = {}
_VIEW_CACHE_MAX = int(os.environ.get("VIEW_CACHE_MAX", "512"))
CURSOR_REPEAT_MASK = 0x6B  # $11E0: move_sights auto-repeat mask, reloaded per idle scan
CURSOR_RAMP = float(
    bin(CURSOR_REPEAT_MASK).count("1")
)  # $11F6 ASL $0CC8 / BCS: one gated scan skipped per set bit before the mask empties
SIGHTS_CENTRE = (80, 95)  # $134C: a sights-ON toggle re-centres the cursor
TOGGLE_FRAMES = 3.5  # sights OFF + ON ($134C recentre + plot_sights)
TAP_FRAMES = 3  # tap_action: idle full scan + press scan ($9678) + latch
UTURN_FLOOR_FRAMES = (
    27.0  # settlecost.BRACKET_FRAMES + the #$28 tune floor: a lower bound
)
ROBOT_EYE = 0.875  # a robot's eye above its foot tile ($E0 fraction)
BOULDER_H = 0.5  # a boulder raises a stack half a unit
EYE_EPS = 0.1  # minimum eye-height progress for a climb move


def _lattice_terms(idx, grids, aim_from):
    """Keyboard terms of lattice rays ``idx`` from facing ``aim_from``: ``(bearing/pitch
    cell, u-turn presses, bearing notches, pitch notches, cursor drive)``."""
    hgrid, vgrid, cxs, cys = grids
    per_h, per_v = len(cxs) * len(cys), len(hgrid) * len(cxs) * len(cys)
    vi, rem = np.divmod(idx, per_v)
    hi, rem2 = np.divmod(rem, per_h)
    cxi, cyi = np.divmod(rem2, len(cys))
    h0, v0 = aim_from
    dh = (
        np.abs(((np.asarray(hgrid)[hi] - h0) + 128) % 256 - 128) // aimcost.AZIMUTH_STEP
    )
    dv = np.abs(((np.asarray(vgrid)[vi] - v0) + 128) % 256 - 128) // aimcost.PITCH_STEP
    nu = (1 + (aimcost.UTURN_STEP - dh) < dh).astype(np.int64)  # aimcost.h_press_count
    ns = np.where(nu > 0, aimcost.UTURN_STEP - dh, dh)
    cur = np.maximum(
        np.abs(np.asarray(cxs)[cxi] - SIGHTS_CENTRE[0]),
        np.abs(np.asarray(cys)[cyi] - SIGHTS_CENTRE[1]),
    )
    return vi.astype(np.int64) * len(hgrid) + hi, nu, ns, dv, cur


def _aim_bound(nu, ns, dv, cur):
    """An aim's frames with every notch's ``projector.render_cost`` dropped: a notch
    costs at least its strip clear plus its queued scroll and a render is never negative,
    so this is an ADMISSIBLE lower bound."""
    drive = np.where(cur > 0, cur + CURSOR_RAMP, 0.0)
    return (
        TOGGLE_FRAMES
        + TAP_FRAMES
        + nu * UTURN_FLOOR_FRAMES
        + ns * (pancost.CLEAR_FRAMES[0] + H_SCROLL)
        + dv * (pancost.CLEAR_FRAMES[1] + V_SCROLL)
        + drive
    )


def _sum_step(t, c, x):
    """One term of CPython's float ``sum`` (Neumaier-compensated), elementwise: running
    total ``t`` and compensation ``c``; the sum is ``t + c``."""
    s = t + x
    return s, c + np.where(np.abs(t) >= np.abs(x), (t - s) + x, (x - s) + t)


def aim_table(st, cells, grids, aim_from, observer):
    """``(t, c)``: the compensated sum of an aim's toggle, u-turn and pan frames, per
    lattice (bearing, pitch) ``cells`` entry; :func:`aim_total` adds the cursor.

    :func:`pancost.notch_plots` pans along ``v0`` first, so per (bearing, pitch direction)
    every pan's notches are a prefix of the farthest one's."""
    hgrid, vgrid = grids[0], grids[1]
    h0, v0 = int(aim_from[0]), int(aim_from[1])
    vi, hi = np.divmod(np.asarray(cells, dtype=np.int64), len(hgrid))
    v1 = np.asarray(vgrid, dtype=np.int64)[vi]
    vs = np.abs((v1 - v0 + 128) % 256 - 128) // aimcost.PITCH_STEP
    up = (v1 - v0) % 256 < 0x80
    pan = np.empty(vi.size)
    turn = np.zeros(vi.size)
    skey = projector.scene_key(st)
    uturn = None
    plans = []
    for h in np.unique(hi):
        h1 = int(hgrid[h])
        nu, ns = aimcost.h_press_count(h0, h1)
        col = hi == h
        if nu:
            if uturn is None:
                uturn = settlecost.uturn_settle_frames(st, observer)
            turn[col] = uturn
        for d in (True, False):
            sel = np.flatnonzero(col & (up == d))
            if sel.size:
                far = sel[np.argmax(vs[sel])]
                plots = pancost.notch_plots(h0, v0, h1, int(v1[far]))
                plans.append((sel, plots, ns))
    pancost.prime_notches(
        st, [p for _sel, plots, _ns in plans for p in plots], observer
    )
    for sel, plots, ns in plans:
        notches = [
            pancost.notch_frames(st, dn, ph, pv, observer, skey)
            + SCROLL[0 if dn < pancost.V_UP else 1]
            for dn, ph, pv in plots
        ]
        sums = np.array([sum(notches[:k]) for k in range(len(notches) + 1)], float)
        pan[sel] = sums[ns + vs[sel]]
    return _sum_step(*_sum_step(np.full(vi.size, TOGGLE_FRAMES), 0.0, turn), pan)


def aim_total(table, drive):
    """An aim's frames from an :func:`aim_table` entry and the cursor ``drive`` term."""
    t, c = _sum_step(*table, drive + TAP_FRAMES)
    return t + c


_DRIVE_CACHE = {}


def _cursor_drive(grids):
    """Cursor drive distance from :data:`SIGHTS_CENTRE` per in-cell lattice offset: a
    lattice index ``i`` lies in cell ``i // len(table)`` at drive ``table[i % len(table)]``.
    """
    key = (tuple(grids[2]), tuple(grids[3]))
    got = _DRIVE_CACHE.get(key)
    if got is None:
        cx = np.abs(np.asarray(grids[2], dtype=np.int64) - SIGHTS_CENTRE[0])
        cy = np.abs(np.asarray(grids[3], dtype=np.int64) - SIGHTS_CENTRE[1])
        got = _DRIVE_CACHE[key] = np.maximum.outer(cx, cy).ravel()
    return got


def _starts(a):
    """Mask of each run's first element in ``a`` (empty-safe)."""
    m = np.ones(a.size, dtype=bool)
    m[1:] = a[1:] != a[:-1]
    return m


def _first_per(group, order):
    """``order`` kept at each run start of ``group[order]`` (``group`` sorted by it)."""
    return order[_starts(group[order])]


_EMPTY = np.iinfo(np.int64).max
_GROUP_TABLES = {}


def _group_min(rays, tiles, drive_of, ncell, rb, db):
    """Per (tile, cell) of the landing ``rays`` the least ``(drive, ray)`` key, packed as
    ``((tile * ncell + cell) << db | drive) << rb | ray`` and ascending: the first of each
    group of the sorted keys, found by a min over a dense table that is cleared after.
    """
    table = _GROUP_TABLES.get(ncell)
    if table is None:
        table = _GROUP_TABLES[ncell] = np.full(mm.N * mm.N * ncell, _EMPTY)
    return _group_min_jit(rays, tiles, drive_of, ncell, rb, db, table)


@njit(cache=True)
def _group_min_jit(rays, tiles, drive_of, ncell, rb, db, table):
    """:func:`_group_min` over the cleared dense ``table``."""
    slots = np.empty(min(rays.size, table.size), dtype=np.int64)
    n = 0
    for i in range(rays.size):
        ray = np.int64(rays[i])
        slot = np.int64(tiles[i]) * ncell + ray // drive_of.size
        key = (((slot << db) | drive_of[ray % drive_of.size]) << rb) | ray
        if table[slot] == _EMPTY:
            slots[n] = slot
            n += 1
        if key < table[slot]:
            table[slot] = key
    slots = np.sort(slots[:n])
    keys = table[slots]
    table[slots] = _EMPTY
    return keys


class _Landings:
    """Per tile, the frame-cheapest of its landing ``rays`` (lowest ray on ties), priced on ask.

    ``tiles`` are packed ``x * N + y`` keys, ``st`` the board aims price on.  The pan prices
    by (h, v) and the drive is monotone, so the least-drive, lowest-ray ray per (tile, cell)
    holds each minimum; asked tiles share one branch-and-bound over :func:`_aim_bound`.
    """

    def __init__(self, st, rays, tiles, grids, aim_from):
        self.st, self.grids, self.aim_from = st, grids, aim_from
        drive_of = _cursor_drive(grids)
        ncell = len(grids[0]) * len(grids[1])
        rb = (ncell * drive_of.size - 1).bit_length()
        db = int(drive_of.max()).bit_length()
        key = _group_min(rays, tiles, drive_of, ncell, rb, db)
        self.ray = key & ((1 << rb) - 1)
        tile, self.cell = np.divmod(key >> (rb + db), ncell)
        cur = (key >> rb) & ((1 << db) - 1)
        _cell, nu, ns, dv, _cur = _lattice_terms(self.ray, grids, aim_from)
        self.bound = _aim_bound(nu, ns, dv, cur)
        self.drive = np.where(cur > 0, cur + CURSOR_RAMP, 0.0)
        start = _starts(tile)
        self.tid = np.cumsum(start) - 1
        self.tiles = tile[start]
        self.index = {
            (int(t) // mm.N, int(t) % mm.N): i for i, t in enumerate(self.tiles)
        }
        self.by_bound = np.lexsort((self.bound, self.tid))
        self.known = np.zeros(ncell, dtype=bool)
        self.table = np.zeros((2, ncell))
        self.pan = np.zeros(ncell)
        self.turn = np.zeros(ncell)
        self.frames = np.full(self.ray.size, np.inf)
        self.won = np.full(self.tiles.size, -1, dtype=np.int64)

    def _price_cells(self, cells):
        """Fill the table of the ``cells`` not yet priced."""
        new = np.unique(cells[~self.known[cells]])
        if new.size:
            self.table[:, new] = aim_table(
                self.st, new, self.grids, self.aim_from, self.st.player
            )
            self.known[new] = True

    def fill(self, cells, pan, turn):
        """Record ``cells`` priced elsewhere: their ``pan`` and ``turn`` terms and the
        :func:`aim_table` sum state they give."""
        self.table[:, cells] = _sum_step(
            *_sum_step(np.full(len(cells), TOGGLE_FRAMES), 0.0, turn), pan
        )
        self.pan[cells], self.turn[cells] = pan, turn
        self.known[cells] = True

    def winners(self, tids):
        """Winning rays of tile ids ``tids``, every rep's cell already priced: the least
        ``(frames, ray)`` of each tile's reps, the answer :meth:`price` bounds its way to.
        """
        want = np.zeros(self.tiles.size, dtype=bool)
        want[tids] = True
        reps = np.flatnonzero(want[self.tid])
        frames = aim_total(self.table[:, self.cell[reps]], self.drive[reps])
        order = np.lexsort((self.ray[reps], frames, self.tid[reps]))
        won = _first_per(self.tid, reps[order])
        self.won[self.tid[won]] = self.ray[won]
        return self.won[tids]

    def _fill(self, sel):
        """Frames of the ``sel`` reps whose cells are priced; ``sel`` returned unpriced."""
        hit = sel[self.known[self.cell[sel]] & np.isinf(self.frames[sel])]
        self.frames[hit] = aim_total(self.table[:, self.cell[hit]], self.drive[hit])
        return hit

    def price(self, tids):
        """Winning rays (``self.won``) of tile ids ``tids``, priced if not yet."""
        want = np.zeros(self.tiles.size, dtype=bool)
        want[tids] = True
        want &= self.won < 0
        if not want.any():
            return self.won[tids]
        order = self.by_bound[want[self.tid[self.by_bound]]]
        if kern.backend() == "cuda":  # the whole lattice in one launch beats its rounds
            self._price_cells(self.cell)
        tid, bound, frames = self.tid, self.bound, self.frames
        best = np.full(self.tiles.size, np.inf)
        self._fill(order)
        np.minimum.at(best, tid[order], frames[order])
        while True:
            live = order[np.isinf(frames[order]) & (bound[order] <= best[tid[order]])]
            if not live.size:
                break
            self._price_cells(self.cell[_first_per(tid, live)])
            hit = self._fill(order)
            np.minimum.at(best, tid[hit], frames[hit])
        won = _first_per(
            tid, order[np.lexsort((self.ray[order], frames[order], tid[order]))]
        )
        self.won[tid[won]] = self.ray[won]
        return self.won[tids]

    def views(self, pick=None):
        """``{tile: view}`` in winning-ray order over the landed tiles ``pick`` takes."""
        tids = np.array(
            [i for t, i in self.index.items() if pick is None or pick(t)],
            dtype=np.int64,
        )
        rays = self.price(tids)
        by_ray = np.argsort(rays)
        return {
            (int(t) // mm.N, int(t) % mm.N): _view_at(r, self.grids)
            for t, r in zip(self.tiles[tids[by_ray]], rays[by_ray])
        }

    def view(self, tile):
        """``tile``'s view, or None if no lattice ray lands on it."""
        i = self.index.get((int(tile[0]), int(tile[1])))
        return None if i is None else _view_at(self.price([i])[0], self.grids)


def _view_at(idx, grids):
    """The build view dict for one flat lattice index."""
    h, v, cx, cy = los._meta_at(int(idx), *grids)
    return {"h_angle": h, "v_angle": v, "cursor": [cx, cy]}


def _sweeps(sts, v_primary, aim_from):
    """:class:`_Landings` of one lattice for each of ``sts`` from the facing ``aim_from``
    (one per state), each on a pinned copy of its board; the whole-board landings come
    from one :func:`batch.landings` call."""
    grids = landtable.lattice(v_primary=v_primary)
    pairs = batch.landings(sts, grids, keep="cell", drive=_cursor_drive(grids))
    return [
        _Landings(st.clone(), rays, tiles, grids, aim_from)
        for st, (rays, tiles) in zip(sts, pairs)
    ]


def _sweep(st, v_primary, aim_from):
    """:class:`_Landings` of one lattice from facing ``aim_from``, on a pinned copy of
    ``st``: only the landing rays are marched, once, for the whole board."""
    return _sweeps([st], v_primary, aim_from)[0]


class _Views:
    """Per-tick lazy cache of the keyboard-aim landable views.

    Each lattice is marched whole at its first query and every answer is read off that
    sweep: the caller keeps acting -- builds, transfers, advanced frames -- while holding
    the instance, and the sweep pinned the board it was taken on."""

    def __init__(self, st):
        self.st = st
        me = st.player
        self.aim_from = (st.obj_h_angle[me], st.obj_v_angle[me])
        self._lat = {}
        self._dict = {}

    def _landings(self, v_primary):
        """The lattice's :class:`_Landings`, memoized across ticks: a pure function of
        the board and the facing it costs aims from, so a tick that gates out or a fork
        of an unchanged board re-enters it, priced tiles and all."""
        got = self._lat.get(v_primary)
        if got is None:
            st = self.st
            key = (projector.scene_key(st), st.player, self.aim_from, v_primary)
            got = self._lat[v_primary] = projector.memo(
                _VIEW_CACHE,
                key,
                _VIEW_CACHE_MAX,
                lambda: _sweep(st, v_primary, self.aim_from),
            )
        return got

    def _views(self, v_primary):
        got = self._dict.get(v_primary)
        if got is None:
            if los._HAVE_JIT:
                got = self._landings(v_primary).views()
            else:
                got = los.landable_sweep_with_centres(self.st, v_primary=v_primary)[0]
            self._dict[v_primary] = got
        return got

    def _one(self, tile, v_primary):
        if v_primary in self._dict or not los._HAVE_JIT:
            return self._views(v_primary).get(tuple(tile))
        return self._landings(v_primary).view(tile)

    def landed(self):
        """``(N, N)`` bool: the tiles some keyboard view lands on, primary plane or band --
        the set :meth:`get` answers for."""
        mask = np.zeros(mm.N * mm.N, dtype=bool)
        for primary in (True, False):
            mask[self._landings(primary).tiles] = True
        return mask.reshape(mm.N, mm.N)

    def primary(self):
        return self._views(True)

    def band(self):
        """The full pitch-band landable views (down-looks included)."""
        return self._views(False)

    def get(self, tile, band=False):
        """The landable view for `tile`, or None; `band` falls back to the
        full pitch-band sweep (down-looks at reclaim/endgame targets)."""
        view = self._one(tile, True)
        if view is not None or not band:
            return view
        return self._one(tile, False)
