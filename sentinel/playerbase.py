"""Shared machinery for players over the sentinel model: world clock, geometry,
threat/gaze windows, aim cost, action firing, and the run loop. Player
subclasses implement _tick()."""

import math
import os

import numpy as np

from sentinel import actioncost, actions, aim, aimcost, enemies, landtable, los
from sentinel import memmap as mm
from sentinel import pancost, projector, relative, settlecost, terrain, threat

H_SCROLL = 16  # $10EE: 16-step horizontal scroll per +-8 bearing notch
V_SCROLL = 8  # $1135: 8-step vertical scroll per +-4 pitch notch
SCROLL = (H_SCROLL, V_SCROLL)  # sentinel.pancost.pan_frames, indexed by notch axis
_VIEW_CACHE = {}
_VIEW_CACHE_MAX = int(os.environ.get("VIEW_CACHE_MAX", "512"))
CURSOR_REPEAT_MASK = 0x6B  # $11E0: move_sights auto-repeat mask, reloaded on every scan with no direction key down
CURSOR_RAMP = float(
    bin(CURSOR_REPEAT_MASK).count("1")
)  # $11F6 ASL $0CC8 / BCS: one gated scan skipped per set bit before the mask empties
SIGHTS_CENTRE = (80, 95)  # $134C: a sights-ON toggle re-centres the cursor
TOGGLE_FRAMES = 3.5  # sights OFF (~0) + ON ($134C recentre + plot_sights); the old 12 folded in the last pan notch's 8-frame trailing V scroll, isolated by the toggle-leak split (dv>0 rows measure 11-12, dv==0 rows 19-20, exactly 8 apart)
TAP_FRAMES = 3  # tap_action: idle full scan + press scan ($9678) + latch
UTURN_FLOOR_FRAMES = 27.0  # settlecost.BRACKET_FRAMES + the 24-frame #$28 tune floor: a u-turn takes the $357D redraw less $245B/$3700 ($1B2F -> $1B84 -> $0C63, $0C51 skipping $35B5), so this is an admissible lower bound; the real cost is settlecost.uturn_settle_frames per scene
UNIT_FRAMES = 3 * 256.0 / mm.COOLDOWN_BRESENHAM_STEP  # cooldown unit in frames
ROT_PERIOD_FRAMES = enemies.ROTATION_COOLDOWN_RELOAD * UNIT_FRAMES
REVOLUTION_STEPS = 14  # rotations covering a full 256-bearing turn at the +-20 step
REVOLUTION_FRAMES = REVOLUTION_STEPS * ROT_PERIOD_FRAMES  # the threat calendar's period
FOV_HALF = enemies.FOV_SCAN // 2  # +-10 units of the enemy scan cone
FOV_MARGIN = 4  # safety margin on top of the cone half-width
EDGE = (
    landtable.RADIUS
)  # highest in-board tile coord: LOS quits at $1F, so 31 never lands
BOARD_TILES = [(x, y) for x in range(EDGE + 1) for y in range(EDGE + 1)]
ROBOT_EYE = 0.875  # a robot's eye above its foot tile ($E0 fraction)
BOULDER_H = 0.5  # a boulder raises a stack half a unit
HOP_COST = 5  # boulder (2) + robot (3)
HOP_FRAMES = 700  # gaze window a full hop (2 creates + transfer + aims) needs; live ls42 hops measured 745 and 879 f whole-step (out/live_ls42_uturnfix.json p2-p4, p6-p8), so this is -6%/-20%, NOT the 2-3x under-charge an offline replay suggested
SAFE_FRAMES = 250  # window below which the current tile is "urgent"
WAIT_FRAMES = 60  # idle advance when no action is available
EYE_EPS = 0.1  # minimum eye-height progress for a climb move
DRAIN_DELAY = 120.0 * UNIT_FRAMES  # $0C20: first-seen -> first drain countdown
MEANIE_SPAWN_FRAMES = enemies.UPDATE_COOLDOWN_MEANIE_MADE * UNIT_FRAMES  # $1869 hold
MEANIE_ARM_FRAMES = (
    (128 // enemies.MEANIE_ROTATE_STEP)
    * enemies.UPDATE_COOLDOWN_MEANIE_ROTATE
    * UNIT_FRAMES
)  # $171B worst-case meanie rotate-to-face the player


def _signed(b):
    return b - 256 if b >= 128 else b


def aim_phases(st, view, aim_from, observer, cursor_from=None):
    """A keyboard aim to ``view`` from facing ``aim_from``, as ordered
    ``(frames, plotting)`` segments.  ``cursor_from`` is the live cursor of a
    same-bearing REUSE, which keeps sights on; None toggles them and $134C re-centres.
    Pure in ``st``: the executor and the view SELECTION price through this one call."""
    h0, v0 = int(aim_from[0]), int(aim_from[1])
    nu = aimcost.h_press_count(h0, view["h_angle"])[0]
    toggles = 0 if cursor_from is not None else TOGGLE_FRAMES
    cur_from = SIGHTS_CENTRE if cursor_from is None else cursor_from
    # move_sights ($9958) steps cx and cy in ONE call: a diagonal drive costs max(|dx|,|dy|) gated scans, plus the $0CC8 ramp, and nothing at all when parked.
    cur = max(
        abs(view["cursor"][0] - cur_from[0]),
        abs(view["cursor"][1] - cur_from[1]),
    )
    cur = cur + CURSOR_RAMP if cur else 0.0
    # Each notch scrolls then replots the strip at its own intermediate angle; the u-turn ($1B2F) flips the bearing, starts tune #$28 and takes the full $357D redraw.
    pan = pancost.pan_frames(
        st, h0, v0, view["h_angle"], view["v_angle"], SCROLL, observer
    )
    uturn = settlecost.uturn_settle_frames(st, observer) if nu else 0.0
    return (
        (toggles, True),
        (nu * uturn, True),
        (pan, True),
        (cur + TAP_FRAMES, False),
    )


def aim_frames(st, view, aim_from, observer, cursor_from=None):
    """Total frames :func:`aim_phases` costs."""
    return sum(f for f, _ in aim_phases(st, view, aim_from, observer, cursor_from))


def _lattice_terms(idx, grids, aim_from):
    """Keyboard terms of lattice rays ``idx`` from facing ``aim_from``: ``(bearing/pitch
    cell, u-turn presses, bearing notches, pitch notches, cursor drive)`` -- the pieces
    :func:`aim_frames` is built from."""
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
    """:func:`aim_frames` with every notch's ``projector.render_cost`` dropped: a notch
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
    """``(t, c)``: :func:`aim_frames`'s ``sum`` state after its toggle, u-turn and pan
    terms, per lattice (bearing, pitch) ``cells`` entry; :func:`aim_total` adds the cursor.

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
            if not sel.size:
                continue
            far = sel[np.argmax(vs[sel])]
            notches = [
                pancost.notch_frames(st, dn, ph, pv, observer, skey)
                + SCROLL[0 if dn < pancost.V_UP else 1]
                for dn, ph, pv in pancost.notch_plots(h0, v0, h1, int(v1[far]))
            ]
            sums = np.array([sum(notches[:k]) for k in range(len(notches) + 1)], float)
            pan[sel] = sums[ns + vs[sel]]
    return _sum_step(*_sum_step(np.full(vi.size, TOGGLE_FRAMES), 0.0, turn), pan)


def aim_total(table, drive):
    """:func:`aim_frames` from an :func:`aim_table` entry and the cursor ``drive`` term."""
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


class _Landings:
    """Per tile, the frame-cheapest of its landing ``rays`` (lowest ray on ties), priced
    on first ask; ``tiles`` are packed ``x * N + y`` keys, ``st`` the board aims price on.

    The pan prices by (h, v) and the cursor drive is monotone, so one representative per
    (tile, cell) -- least drive, then lowest ray -- holds each minimum.  The asked tiles run
    one branch-and-bound over :func:`_aim_bound` together, a priced cell pricing every rep.
    """

    def __init__(self, st, rays, tiles, grids, aim_from):
        self.st, self.grids, self.aim_from = st, grids, aim_from
        drive_of = _cursor_drive(grids)
        ncell = len(grids[0]) * len(grids[1])
        rb = (ncell * drive_of.size - 1).bit_length()
        db = int(drive_of.max()).bit_length()
        cell, off = np.divmod(rays, drive_of.size)
        key = np.sort(
            (((tiles * ncell + cell) << db | drive_of[off]) << rb)
            | rays.astype(np.int64)
        )
        key = key[_starts(key >> (rb + db))]
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
        self.frames = np.full(self.ray.size, np.inf)
        self.won = np.full(self.tiles.size, -1, dtype=np.int64)

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
        tid, bound, frames = self.tid, self.bound, self.frames
        best = np.full(self.tiles.size, np.inf)
        self._fill(order)
        np.minimum.at(best, tid[order], frames[order])
        while True:
            live = order[np.isinf(frames[order]) & (bound[order] <= best[tid[order]])]
            if not live.size:
                break
            new = np.unique(self.cell[_first_per(tid, live)])
            self.table[:, new] = aim_table(
                self.st, new, self.grids, self.aim_from, self.st.player
            )
            self.known[new] = True
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


def _cheapest_rays(st, rays, tiles, grids, aim_from):
    """``(tiles, rays)``: every distinct ``tiles`` key's :class:`_Landings` winner."""
    got = _Landings(st, rays, tiles, grids, aim_from)
    return got.tiles, got.price(np.arange(got.tiles.size))


def _view_at(idx, grids):
    """The build view dict for one flat lattice index."""
    h, v, cx, cy = los._meta_at(int(idx), *grids)
    return {"h_angle": h, "v_angle": v, "cursor": [cx, cy]}


def _sweep(st, v_primary, aim_from):
    """:class:`_Landings` of one lattice from facing ``aim_from``, on a pinned copy of
    ``st``: only :func:`landtable.landing_rays` is marched, once, for the whole board.
    """
    grids = landtable.lattice(v_primary=v_primary)
    slot = st.player
    gpu = landtable._gpu()
    if gpu is not None:
        lit, key, _first = gpu.landing_table(st, slot, None, grids, landtable.MAX_STEPS)
        return _Landings(st.clone(), lit, key, grids, aim_from)
    idx = landtable.landing_rays(st, slot, None, grids, landtable.MAX_STEPS)
    status, tx, ty = landtable._march(st, slot, None, grids, idx, landtable.MAX_STEPS)
    hit = np.flatnonzero(status == los.los_jit.LOS_CLEAR)
    key = tx[hit].astype(np.int64) * mm.N + ty[hit].astype(np.int64)
    return _Landings(st.clone(), idx[hit], key, grids, aim_from)


def _cheap_views(st, v_primary, aim_from):
    """Landable views choosing, per tile, the FRAME-CHEAPEST keyboard view from facing
    ``aim_from`` -- the tile membership of ``los.landable_sweep_with_centres``, in
    winning-ray order."""
    if not los._HAVE_JIT:
        return los.landable_sweep_with_centres(st, v_primary=v_primary)[0]
    return _sweep(st, v_primary, aim_from).views()


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

    def band_ordered(self, pick):
        """``[(tile, view)]`` in the band sweep's own (winning-ray) order for every tile
        ``pick`` takes -- what a positional reader of :meth:`band` sees."""
        if False in self._dict or not los._HAVE_JIT:
            return [(t, v) for t, v in self._views(False).items() if pick(t)]
        return list(self._landings(False).views(pick).items())

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

    def band_get(self, tile):
        """The full pitch-band view for `tile`; the lazy fallback lookup passed
        to :meth:`BasePlayer._view_with_band`."""
        return self._one(tile, False)


class BasePlayer:
    """Shared machinery for a player bound to a live :class:`Game`: the sights
    cursor and committed bearing are the only cross-tick state here."""

    def __init__(self, game, verbose=False, audit=False):
        self.g = game
        self.st = game.state
        self.cursor = list(SIGHTS_CENTRE)
        self.last_bearing = None  # committed (h, v): a same-bearing aim reuses
        self._stale = None  # (plan step key, consecutive stale verdicts on it)
        self.frames = 0
        self.verbose = verbose
        self.audit = audit  # strict post-settle invariant accounting (below)
        self.breaches = []
        self.fire_reason = None  # clause that refused the last _fire, or None
        self.trace = []

    # ------------------------------------------------------------------ clock
    def _advance(self, frames, plotting=False):
        """Advance `frames`; `plotting` marks foreground the ROM spends inside
        plot_world/dither/scroll, where $16B5 is never reached and only the raster
        cooldown clock runs ($130C).  Measured against the recorded enemy clock in
        ``sentinel/tests/test_human_clock.py``."""
        frames = int(round(frames))
        enemies.advance_frames(self.st, frames, plotting=plotting)
        self.frames += frames

    def _advance_phases(self, phases):
        """Advance a sequence of ``(frames, plotting)`` segments in order."""
        for frames, plotting in phases:
            if frames > 0:
                self._advance(frames, plotting=plotting)

    # ------------------------------------------------------------- geometry
    def _my_eye(self):
        return self.st.eye_z()

    def _top(self, tile):
        return terrain.top_object(self.st, *tile)

    def _base_z(self, slot):
        return self.st.obj_z_height[slot] + self.st.obj_z_frac[slot] / 256.0

    def _robot_eye_after_boulder(self, tile):
        """Eye height of a robot on `tile` after adding one more boulder."""
        top = self._top(tile)
        if top is None:
            b = terrain.tile_byte(self.st, *tile)
            return (b >> 4) + ROBOT_EYE + BOULDER_H
        return self._base_z(top) + 2 * BOULDER_H

    def _sees_tile(self, tile):
        """Cheap geometric pre-filter (one ray march) before a full-band
        landability sweep: can the player's eye see `tile` at all?"""
        return threat.player_sees_tile(self.st, tile, self.st.player)

    def _view_with_band(self, tile, primary, band):
        """The `primary`-plane view for `tile`, else `band(tile)` -- the wider
        down-look lookup.  This IS `aim.propose`: the ROM's own action gate ($1B46)
        asks only whether SOME keyboard view reaches `tile` with clear line of sight
        at the true eye.  A `_sees_tile` pre-filter here is a DIFFERENT and stricter
        question (can an observer see a body standing on the tile) and vetoed views
        the ROM grants, so the band lookup is consulted unconditionally."""
        tile = tuple(tile)
        view = primary.get(tile)
        if view is not None:
            return view
        return band(tile)

    # ------------------------------------------------------------ candidates
    def _robot_bodies(self):
        """(slot, tile) for every foreign robot body topping its own tile: the
        transfer/escape candidate set.  Callers rank and gate it themselves."""
        st = self.st
        for slot in range(mm.NUM_SLOTS):
            if st.is_empty(slot) or slot == st.player:
                continue
            if st.obj_type[slot] != mm.T_ROBOT:
                continue
            tile = st.tile_of(slot)
            if self._top(tile) != slot:
                continue
            yield slot, tile

    def _reclaim_targets(self, st, want_trees, skip=None):
        """(energy value, tile) for every absorbable spent investment on `st`:
        shells and pedestals at or below the eye (a taller one is a live
        pedestal, not a spent one) and, when `want_trees`, trees.  Never the
        player's own tile nor `skip` (the hop in progress).  Slot order; callers
        rank and gate it themselves."""
        my_eye = st.eye_z()
        here = st.player_xy()
        for slot in range(mm.NUM_SLOTS):
            if st.is_empty(slot) or slot == st.player:
                continue
            otype = st.obj_type[slot]
            tile = st.tile_of(slot)
            if tile == skip or tile == here:
                continue
            if terrain.top_object(st, *tile) != slot:
                continue
            if otype in (mm.T_ROBOT, mm.T_BOULDER):
                base = st.obj_z_height[slot] + st.obj_z_frac[slot] / 256.0
                if base > my_eye + EYE_EPS:
                    continue
            elif not (otype == mm.T_TREE and want_trees):
                continue
            yield mm.ENERGY_IN_OBJECTS[otype], tile

    # ---------------------------------------------------------------- threat
    @staticmethod
    def _in_cone(angle_hi, facing, half):
        """The ROM's FOV gate ($18B8) on its own bearing: byte test
        (angle_hi - facing + fov/2) & $FF < fov, with fov == 2*half."""
        return ((angle_hi - facing + half) & 0xFF) < 2 * half

    def _exposing_enemies(self, tile):
        """(enemy, angle_hi, full) for every enemy with ANY sight of a robot on
        `tile` at ANY facing; angle_hi is the ROM bearing ($8401) the real FOV
        gate compares the facing against (NOT an analytic atan2 -- the game's
        compass is rotated/mirrored relative to it).  A tile already topped by
        a robot (a transfer target) is evaluated on THAT robot -- a phantom
        cannot stand there, and treating it as unexposed hid every transfer
        destination from the invariant."""
        st = self.st
        top = self._top(tile)
        if top is not None and st.obj_type[top] == mm.T_ROBOT:
            return self._exposures(st, top)
        clone = st.clone()
        slot = threat._free_slot(clone)
        if slot is None:
            return []
        old = terrain.tile_byte(clone, *tile)
        if not threat._place_phantom(clone, tile, slot):
            return []
        out = self._exposures(clone, slot)
        threat._restore_tile(clone, tile, old, slot)
        return out

    @staticmethod
    def _exposures(st, slot):
        out = []
        for e in enemies.enemy_slots(st):
            see = relative.can_see_object(st, e, slot, mm.T_ROBOT, threat.FOV_FULL)
            if see["exposure"]:
                ah = relative.relative_angles(st, e, slot)["angle_hi"]
                out.append((e, ah, bool(see["full"])))
        return out

    def _tile_sees_target(self, tile, target):
        """Whether a robot standing on `tile` would see `target`'s tile (a
        phantom-observer $1CDD march) -- the LOS-progress term of a hop."""
        clone = self.st.clone()
        slot = threat._free_slot(clone)
        if slot is None or not threat._place_phantom(clone, tile, slot):
            return False
        return threat.player_sees_tile(clone, target, slot)

    def _tree_near(self, tile):
        """The meanie precondition ($19C3/$19D5): some tree within 10 tiles of
        `tile` in BOTH axes -- only then is PARTIAL visibility dangerous."""
        st = self.st
        for s in range(mm.NUM_SLOTS):
            if st.is_empty(s) or st.obj_type[s] != mm.T_TREE:
                continue
            if abs(st.obj_x[s] - tile[0]) < 10 and abs(st.obj_y[s] - tile[1]) < 10:
                return True
        return False

    def _seen_now(self, exposed):
        """Whether an exposing enemy has the spot in its live cone right now --
        the never-place-in-enemy-view test."""
        st = self.st
        half = FOV_HALF + FOV_MARGIN
        return any(
            self._in_cone(ah, st.obj_h_angle[e], half) for e, ah, _full in exposed
        )

    def _cone_onset(self, e, angle_hi, half):
        """Frames until enemy `e`'s rotating scan cone ($1805, fixed +-step /
        200-round reload) first covers `angle_hi`: 0 if it does now, inf if its
        step never brings it round.  The bare cone-arrival clock, which is NOT
        when a drain lands (`_drain_clock`).  Deterministic -- no PRNG."""
        st = self.st
        facing = st.obj_h_angle[e]
        if self._in_cone(angle_hi, facing, half):
            return 0.0
        step = _signed(st.mem[mm.ROTATION_SPEED_TABLE + e])
        if step == 0:
            return math.inf
        first = st.mem[mm.ENEMIES_ROTATION_COOLDOWN + e] * UNIT_FRAMES
        for k in range(1, 256 // abs(step) + 2):
            facing = (facing + step) & 0xFF
            if self._in_cone(angle_hi, facing, half):
                return first + (k - 1) * ROT_PERIOD_FRAMES
        return math.inf

    def _drain_clock(self, e, angle_hi, half, target):
        """Frames until enemy `e` can take energy off (or arm a meanie against) a
        body on bearing `angle_hi`: the cone onset PLUS the draining countdown still
        owed.  Sight alone never drains -- $1825 ARMS $0C20 to 120 rounds on first
        sight and only acts as it expires, and $1A31 re-zeroes it after every drain
        -- so a cone pass costs its first `DRAIN_DELAY` frames nothing.  The owed
        countdown is the live cooldown byte when `e` already holds `target` under
        its live cone ($178C keeps a still-visible target counting down), else the
        full $1835 reload a fresh sighting loads."""
        onset = self._cone_onset(e, angle_hi, half)
        if onset == math.inf:
            return math.inf
        cd = int(self.st.mem[mm.ENEMIES_DRAINING_COOLDOWN + e])
        held = int(self.st.mem[mm.ENEMIES_TARGETED_OBJECT + e])
        if onset == 0.0 and cd and target is not None and held == target:
            return cd * UNIT_FRAMES
        return onset + DRAIN_DELAY

    def _meanie_window(self, tile, exposed):
        """Frames until a meanie armed from PARTIAL sight of a body on `tile`
        could force a hyperspace ($1986): a partially-seeing enemy must rotate its
        cone on, run the drain countdown to the meanie branch ($183D/$1852), spawn
        the meanie ($1869) and rotate it to face ($171B).  Needs a tree within 10
        tiles ($19C3); inf otherwise.  Always far slower than a drain -- this clock
        is NEVER 0."""
        if not self._tree_near(tile):
            return math.inf
        half = FOV_HALF + FOV_MARGIN
        target = self._top(tile)
        armed = min(
            (
                self._drain_clock(e, ah, half, target)
                for e, ah, full in exposed
                if not full
            ),
            default=math.inf,
        )
        if armed == math.inf:
            return math.inf
        return armed + MEANIE_SPAWN_FRAMES + MEANIE_ARM_FRAMES

    def _gaze_window(self, tile, exposed=None):
        """Frames until an enemy can take ENERGY off a robot body on `tile` -- cone
        onset plus the draining countdown ($0C20), NOT cone onset alone: arrival in
        a cone is free, the drain is what costs.  Only FULL sight drains ($1838;
        $16E6 step 3 skips a partially-seen robot), so the drain clock keys on
        full-sight enemies.  The partial+tree path is the MEANIE arm ($19C3) -- a
        SEPARATE, far slower clock (`_meanie_window`).  inf == never; `_cone_onset`
        is the bare cone-arrival accessor for callers wanting that instead.
        Deterministic -- no PRNG."""
        if exposed is None:
            exposed = self._exposing_enemies(tile)
        half = FOV_HALF + FOV_MARGIN
        target = self._top(tile)
        best = self._meanie_window(tile, exposed)
        for e, angle_hi, full in exposed:
            if full:
                best = min(best, self._drain_clock(e, angle_hi, half, target))
        return best

    @staticmethod
    def _drains_in(window, budget):
        """Energy units an exposure lasting `budget` frames costs a body whose
        time-to-first-drain is `window`.  Exposure is a RATE, not a deadline: $1825
        arms the 120-round countdown and $1A31 re-zeroes it after every drain, so
        drains recur every `DRAIN_DELAY` frames of continuous sight."""
        if budget <= window or window == math.inf:
            return 0
        return int(math.ceil((budget - window) / DRAIN_DELAY))

    def _drain_gate(self, verb, tile, exposed=None, budget=0.0):
        """Whether placing `verb` on `tile` is drain-safe: its time-to-first-drain
        must outlast `budget` (the aim+settle the body stands exposed).  A boulder is
        exempt ($16E6 drains robots only, never a boulder body); partial sight is not
        a drain either -- its slower meanie arm is priced into `_gaze_window`.

        A live FULL-sight cone is not a separate refusal.  Sight only ARMS the
        draining countdown ($1825), so a body under a cone still has its residual
        before it costs anything, and `_gaze_window` prices exactly that: refusing
        cone presence outright was the onset premise's degenerate case (window 0),
        not a rule of its own."""
        if verb == "boulder":
            return True
        return self._gaze_window(tile, exposed=exposed) >= budget

    def _frozen(self):
        """World frozen until the player's first action ($0CE5 bit7, $3682)."""
        return bool(self.st.mem[mm.PLAYER_NOT_ACTED] & 0x80)

    def _reserve(self):
        """Energy the stance owes unconditionally -- 0 unless a meanie is ALIVE.

        Reaching 0 is not death ($1A08 debits; $1A00 kills only on a drain arriving at
        0) and $1BBF lets a create spend to empty, so survival is
        `energy - cost - drains >= reserve` -- `_affords`, exposure billed as the
        $0C20 RATE it is.  A live meanie owes its forced hyperspace's robot cost
        ($2156 spends 3, $2170 kills on underflow); an ARMABLE one owes nothing here,
        `_gaze_window` already folds `_meanie_window` into every gate's window."""
        st = self.st
        floor = 0
        if any(
            not st.is_empty(s) and st.obj_type[s] == mm.T_MEANIE
            for s in range(mm.NUM_SLOTS)
        ):
            floor += mm.ENERGY_IN_OBJECTS[mm.T_ROBOT]
        return floor

    def _affords(self, cost, budget):
        """Whether spending `cost` energy survives the drains `budget` frames of
        exposure bill at the $0C20/$1A31 rate, over `_reserve`.

        `budget` is the span the body stands exposed for -- a create's settle, a hop's
        whole build.  Executor and search both gate on this, at the same instant, so a
        plan cannot carry a step the executor then refuses."""
        drains = self._drains_in(self._player_window(), budget)
        return self.st.energy - cost - drains >= self._reserve()

    def _player_window(self):
        """Frames until the player's OWN body is drainable (inf if never; no phantom)."""
        st = self.st
        if self._frozen():
            return math.inf  # no drain/meanie clock runs before the first action
        exposed = self._exposures(st, st.player)
        if not exposed:
            return math.inf
        return self._gaze_window(st.player_xy(), exposed=exposed)

    # ------------------------------------------------------------------- aim
    def _aim_phases(self, view):
        """:func:`aim_phases` bound to this player's facing, committed bearing and live
        cursor: a same-bearing REUSE keeps sights on and drives from where the cursor
        is, otherwise they toggle and $134C re-centres."""
        st = self.st
        me = st.player
        reuse = self.last_bearing == (view["h_angle"], view["v_angle"])
        return aim_phases(
            st,
            view,
            (st.obj_h_angle[me], st.obj_v_angle[me]),
            me,
            cursor_from=self.cursor if reuse else None,
        )

    def _aim_frames(self, view):
        """Total frames the executor's aim method costs."""
        return sum(f for f, _ in self._aim_phases(view))

    def _aim_unfreeze_split(self, view):
        """Frames of aim elapsing BEFORE a u-turn unfreezes the world, else None.

        A u-turn is dispatched as an ACTION ($23) and $12D5 `CMP #$22 / BCS $12DE` lets
        codes >= $22 skip the sights-on check and fall into $12E1 `LSR $0CE5`, so keying
        one STARTS the enemy clock mid-aim, before any real action fires."""
        if not self._frozen():
            return None
        nu = aimcost.h_press_count(
            int(self.st.obj_h_angle[self.st.player]), view["h_angle"]
        )[0]
        if not nu:
            return None
        reuse = self.last_bearing == (view["h_angle"], view["v_angle"])
        uturn = settlecost.uturn_settle_frames(self.st)
        return (0.0 if reuse else TOGGLE_FRAMES) + nu * uturn

    def _step_aim_phases(self, verb, view):
        """`_aim_phases` for `verb`, empty when a transfer rides a reused bearing."""
        if verb == "transfer" and self.last_bearing == (
            view["h_angle"],
            view["v_angle"],
        ):
            return ()
        return self._aim_phases(view)

    def _aim_head_tail(self, verb, view):
        """The aim split at the u-turn unfreeze: ``(before, after)`` phase segments.

        A non-empty head means keying the u-turn starts the enemy clock part-way
        through the aim ($12E1), so the caller clears $0CE5 between the two.
        """
        phases = self._step_aim_phases(verb, view)
        if self._aim_unfreeze_split(view) is None:
            return (), phases
        return phases[:2], phases[2:]

    def _step_aim_frames(self, verb, view):
        """Aim frames the executor spends before `verb` fires.  A transfer over a REUSED
        committed bearing sends no aim keys ($21 fires on the object under the cursor the
        preceding same-tile create/absorb parked there), so its aim is 0 (measured: every
        recorded transfer); on a MISMATCHED bearing the executor drives the full view
        (``live_player._drive_transfer_aim``), the same aim every other verb pays."""
        if verb == "transfer" and self.last_bearing == (
            view["h_angle"],
            view["v_angle"],
        ):
            return 0.0
        return self._aim_frames(view)

    def _fire(self, verb, tile, view):
        """Aim (world advances), re-gate, apply `verb` on `tile`, settle.
        Returns False if the gate fails after the aim (the world changed under
        us) -- the caller just re-plans next tick.

        ``fire_reason`` names the clause that refused, so a refusal measured against a
        human WIN can be attributed instead of merely counted."""
        st = self.st
        self.fire_reason = None
        head, tail = self._aim_head_tail(verb, view)
        self._advance_phases(head)
        if head:
            st.mem[mm.PLAYER_NOT_ACTED] = 0x00  # $12E1: the u-turn unfroze the world
        self._advance_phases(tail)
        if actions.player_dead(st):
            self.fire_reason = "dead_during_aim"
            return False
        me = st.player
        st.obj_h_angle[me] = view["h_angle"]
        st.obj_v_angle[me] = view["v_angle"]
        self.cursor = list(view["cursor"])
        self.last_bearing = (view["h_angle"], view["v_angle"])
        if not aim.gate(st, view, tile):
            view = aim.propose(st, tile)
            if view is None or not aim.gate(st, view, tile):
                self.fire_reason = "aim_gate"
                return False
        settle = self._settle(verb, view, tile=tile)  # pre-action: the clone phantom
        if verb in ("boulder", "robot"):
            cost = mm.ENERGY_IN_OBJECTS[
                mm.T_BOULDER if verb == "boulder" else mm.T_ROBOT
            ]
            if not self._affords(cost, settle):
                self.fire_reason = "affords"
                return False  # the drains this create's own settle bills take the body
        ok = False
        if verb == "boulder":
            ok = actions.create(st, mm.T_BOULDER, tile) is not None
        elif verb == "robot":
            ok = actions.create(st, mm.T_ROBOT, tile) is not None
        elif verb == "absorb":
            top = self._top(tile)
            ok = top is not None and actions.absorb(st, top)
        elif verb == "transfer":
            top = self._top(tile)
            ok = top is not None and actions.transfer(st, top)
        if ok:
            if verb == "transfer":
                self.last_bearing = None  # new body: committed bearing is stale
            # dither/replot ($1FA4/$86A5) or the viewpoint redraw ($357D): all foreground
            self._advance(settle, plotting=True)
            if self.audit and verb in ("boulder", "robot", "transfer"):
                self._account(verb, tile)
            self._log(verb, tile)
        else:
            self.fire_reason = "rom_refused"  # $1B46/$1B9B: the action itself would not
        return ok

    def _settle_eye(self, verb, tile):
        """Eye the post-action settle is seen from: for a transfer the topmost
        object of `tile` (the body $0C63 moves into BEFORE the $35C3/$35C6 replot
        passes), else None -- the current viewpoint, unmoved."""
        return self._top(tile) if verb == "transfer" else None

    def _settle(self, verb, view=None, observer=None, tile=None):
        """World frames the ROM advances AFTER `verb` fires; the placed object is
        on the board and exposable for this whole settle, so a danger window must
        cover the aim plus this before an enemy's cone can rotate in.

        A transfer moves the eye ($0C63) then settles through the $357D redraw as
        the NEW body sees it (settlecost.viewpoint_settle_frames); a create/absorb
        settles through the $12EC dither replot of the acted slot, priced on a
        clone carrying the mid-animation posture ($1F10 flags $80 for an absorb)."""
        st = self.st
        if verb == "transfer":
            eye = observer
            if eye is None:
                eye = self._top(tile) if tile is not None else st.player
            return actioncost.FRAME_TICKS * settlecost.viewpoint_settle_frames(
                st, eye, view
            )
        key = {"boulder": "create", "robot": "create"}.get(verb, verb)
        if tile is None:
            return actioncost.SETTLE.get(key, 60)
        clone = st.clone()
        if view is not None:  # the settle runs with the player facing the aim
            clone.obj_h_angle[clone.player] = view["h_angle"]
            clone.obj_v_angle[clone.player] = view["v_angle"]
        plot_state = None
        if key == "create":
            otype = mm.T_BOULDER if verb == "boulder" else mm.T_ROBOT
            slot = actions.create(clone, otype, tile)
        else:
            slot = terrain.top_object(clone, *tile)
            if slot is not None:
                plot_state = clone.clone()
                actions.absorb(plot_state, slot)  # the strip plots the object erased
                clone.obj_flags[slot] = 0x80
        if slot is None:
            return actioncost.SETTLE.get(key, 60)
        return actioncost.FRAME_TICKS * settlecost.action_settle_frames(
            clone, slot, view, plot_state=plot_state
        )

    def _account(self, verb, tile):
        """Strict post-settle invariant on the ACTUAL placed object via the ROM's
        own scan cone: record only a robot body left in a live FULL-sight cone
        ($1838 drain) -- the sole immediate breach.  A boulder is not a drainable
        body ($16E6 drains robots only), and a partially-seen robot cannot be
        drained either (it only arms a slower meanie, `_meanie_window`), so
        neither is an immediate breach the plan-time gate must have caught."""
        if verb == "boulder":
            return
        st = self.st
        top = terrain.top_object(st, *tile)
        if top is None:
            return
        dangerous = [
            (int(e), True)
            for e in enemies.enemy_slots(st)
            if relative.can_see_object(st, e, top, st.obj_type[top], enemies.FOV_SCAN)[
                "full"
            ]
        ]
        if dangerous:
            self.breaches.append((self.frames, verb, tuple(tile), dangerous))
            if self.verbose:
                print(f"  BREACH {verb} {tuple(tile)} seen_by={dangerous}")

    def _log(self, verb, tile):
        st = self.st
        rec = (self.frames, verb, tuple(tile), st.energy, round(self._my_eye(), 3))
        self.trace.append(rec)
        if self.verbose:
            faces = [st.obj_h_angle[e] for e in enemies.enemy_slots(st)]
            print(
                f"f={rec[0]:6d} {verb:10s} {rec[2]} E={rec[3]:2d} "
                f"eye={rec[4]:6.3f} enemy_h={faces}"
            )

    def _hyperspace(self):
        """Fire hyperspace (the win move from the platform / last-resort escape);
        the live driver overrides this with the real keystroke."""
        actions.hyperspace(self.st)
        self._log("hyperspace", self.st.player_xy())

    def _observe(self):
        """Refresh the observed state at tick start (the live driver re-reads
        game memory here; the simulator's state is already live)."""

    def _dead(self):
        """Whether the player is dead; the live driver refines the ambiguous
        $0CDE bit7 (set by ANY survived hyperspace, not only a meanie kill)."""
        return actions.player_dead(self.st)

    # ------------------------------------------------------------- decisions
    def run(self, max_actions=300):
        """Play until won, dead, or `max_actions` decision ticks."""
        for _ in range(max_actions):
            self._observe()
            if actions.won(self.st):
                return True
            if self._dead():
                return False
            self._tick()
        self._observe()
        return actions.won(self.st)

    def _tick(self):
        """Subclasses pick one action per decision tick."""
        raise NotImplementedError
