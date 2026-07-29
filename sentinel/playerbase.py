"""Shared machinery for players over the sentinel model: world clock, geometry,
threat/gaze windows, aim cost, action firing, and the run loop. Player
subclasses implement _tick()."""

import math
import os

import numpy as np

from sentinel import actioncost, actions, aim, aimcost, enemies, landtable, los
from sentinel import memmap as mm
from sentinel import pancost, projector, relative, terrain, threat

H_SCROLL = 16  # $10EE: 16-step horizontal scroll per +-8 bearing notch
V_SCROLL = 8  # $1135: 8-step vertical scroll per +-4 pitch notch
SCROLL = (H_SCROLL, V_SCROLL)  # sentinel.pancost.pan_frames, indexed by notch axis
_VIEW_CACHE = {}
_VIEW_CACHE_MAX = int(os.environ.get("VIEW_CACHE_MAX", "512"))
_TILE_VIEW_CACHE = {}  # per-(sweep key, tile) targeted views; entries are 1-tuples
_TILE_VIEW_CACHE_MAX = 1 << 14
_EPOCH_CACHE = {}  # measured first-rotation frames, keyed by the enemy clock
_EPOCH_CACHE_MAX = 256
CURSOR_REPEAT_MASK = 0x6B  # $11E0: move_sights auto-repeat mask, reloaded on every scan with no direction key down
CURSOR_RAMP = float(
    bin(CURSOR_REPEAT_MASK).count("1")
)  # $11F6 ASL $0CC8 / BCS: one gated scan skipped per set bit before the mask empties
SIGHTS_CENTRE = (80, 95)  # $134C: a sights-ON toggle re-centres the cursor
TOGGLE_FRAMES = 12  # sights OFF (~0) + ON (~10: $134C recentre + plot_sights), measured
TAP_FRAMES = 3  # tap_action: idle full scan + press scan ($9678) + latch
UTURN_FRAMES = 74  # a u-turn is a full action tap (want-flag $23, kbd_aim._uturn), not a bare keystroke: idle+press scans plus the action's own consumption. Live ls42 p1, n=1.
UNIT_FRAMES = 3 * 256.0 / mm.COOLDOWN_BRESENHAM_STEP  # cooldown unit in frames
ROT_PERIOD_FRAMES = enemies.ROTATION_COOLDOWN_RELOAD * UNIT_FRAMES
REVOLUTION_STEPS = 14  # rotations covering a full 256-bearing turn at the +-20 step
REVOLUTION_FRAMES = REVOLUTION_STEPS * ROT_PERIOD_FRAMES  # the threat calendar's period
FOV_HALF = enemies.FOV_SCAN // 2  # +-10 units of the enemy scan cone
FOV_MARGIN = 4  # safety margin on top of the cone half-width
ROBOT_EYE = 0.875  # a robot's eye above its foot tile ($E0 fraction)
BOULDER_H = 0.5  # a boulder raises a stack half a unit
HOP_COST = 5  # boulder (2) + robot (3)
HOP_FRAMES = 700  # gaze window a full hop (2 creates + transfer + aims) needs; live ls42 hops measured 745 and 879 f whole-step (out/live_ls42_uturnfix.json p2-p4, p6-p8), so this is -6%/-20%, NOT the 2-3x under-charge an offline replay suggested
SAFE_FRAMES = 250  # window below which the current tile is "urgent"
WAIT_FRAMES = 60  # idle advance when no action is available
EYE_EPS = 0.1  # minimum eye-height progress for a climb move
DRAIN_DELAY = 120.0 * UNIT_FRAMES  # $0C20: first-seen -> first drain countdown
DRAIN_STEP = (
    enemies.UPDATE_COOLDOWN_DRAIN * UNIT_FRAMES
)  # $17F1 re-arm: gap between drains on a HELD target, measured 125 f
MEANIE_SPAWN_FRAMES = enemies.UPDATE_COOLDOWN_MEANIE_MADE * UNIT_FRAMES  # $1869 hold
MEANIE_ARM_FRAMES = (
    (128 // enemies.MEANIE_ROTATE_STEP)
    * enemies.UPDATE_COOLDOWN_MEANIE_ROTATE
    * UNIT_FRAMES
)  # $171B worst-case meanie rotate-to-face the player


def _signed(b):
    return b - 256 if b >= 128 else b


def _aim_cost(idx, grids, aim_from):
    """Keyboard aim cost of lattice rays ``idx`` from facing ``aim_from``: u-turn-aware
    bearing notches, then pitch notches, then cursor distance from the $134C recentre
    point."""
    hgrid, vgrid, cxs, cys = grids
    per_h, per_v = len(cxs) * len(cys), len(hgrid) * len(cxs) * len(cys)
    vi, rem = np.divmod(idx, per_v)
    hi, rem2 = np.divmod(rem, per_h)
    cxi, cyi = np.divmod(rem2, len(cys))
    h0, v0 = aim_from
    h = np.asarray(hgrid)[hi]
    v = np.asarray(vgrid)[vi]
    dh = np.abs(((h - h0) + 128) % 256 - 128) // aimcost.AZIMUTH_STEP
    dv = np.abs(((v - v0) + 128) % 256 - 128) // aimcost.PITCH_STEP
    cur = np.maximum(
        np.abs(np.asarray(cxs)[cxi] - los.SIGHTS_CX),
        np.abs(np.asarray(cys)[cyi] - los.SIGHTS_CY),
    )
    return np.minimum(dh, 17 - dh) * 1000 + dv * 100 + cur


def _view_at(idx, grids):
    """The build view dict for one flat lattice index."""
    h, v, cx, cy = los._meta_at(int(idx), *grids)
    return {"h_angle": h, "v_angle": v, "cursor": [cx, cy]}


def _cheap_views(st, v_primary, aim_from):
    """Landable views choosing, per tile, the MIN-AIM-COST keyboard view from facing
    ``aim_from`` (:func:`_aim_cost`, ties broken by lattice index) -- same tile
    membership as ``los.landable_sweep_with_centres``, cheapest representative view."""
    if not los._HAVE_JIT:
        return los.landable_sweep_with_centres(st, v_primary=v_primary)[0]
    grids = landtable.lattice(v_primary=v_primary)
    status, tx, ty, _, grids = los._landable_batch(st, st.player, None, 6000, *grids)
    views = {}
    clear = np.flatnonzero(status == los.los_jit.LOS_CLEAR)
    if not clear.size:
        return views
    key = (tx[clear].astype(np.int64) << 16) | ty[clear].astype(np.int64)
    cost = _aim_cost(clear, grids, aim_from)
    order = np.lexsort((clear, cost, key))
    ks = key[order]
    head = order[np.concatenate(([True], ks[1:] != ks[:-1]))]
    for i in head[np.argsort(clear[head], kind="stable")]:
        views[(int(key[i] >> 16), int(key[i] & 0xFFFF))] = _view_at(clear[i], grids)
    return views


def _cheap_view(st, tile, v_primary, aim_from):
    """The :func:`_cheap_views` entry for ONE tile, bit-identical, without the sweep.

    Only :func:`landtable.candidates` is marched -- a proven superset of the rays that
    can land ``tile`` -- and the same min-(cost, index) rule picks the winner, so a
    candidate set that lands nothing is a proven "no view"."""
    tile = (int(tile[0]), int(tile[1]))
    if not los._HAVE_JIT:  # pragma: no cover - numba absent -> the whole-board sweep
        return _cheap_views(st, v_primary, aim_from).get(tile)
    slot = st.player
    if landtable.never_lands(st, tile, slot):
        return None
    grids = landtable.lattice(v_primary=v_primary)
    idx = landtable.candidates(st, tile, slot, None, grids, landtable.MAX_STEPS)
    if not idx.size:
        return None
    status, tx, ty = landtable._march(st, slot, None, grids, idx, landtable.MAX_STEPS)
    hit = np.flatnonzero(
        (status == los.los_jit.LOS_CLEAR) & (tx == tile[0]) & (ty == tile[1])
    )
    if not hit.size:
        return None
    sel = idx[hit]  # ascending: argmin takes the lowest index among min-cost rays
    return _view_at(sel[np.argmin(_aim_cost(sel, grids, aim_from))], grids)


class _Views:
    """Per-tick lazy cache of the keyboard-aim landable views.

    A lattice is PINNED (board snapshot + memo key) at its first query and every later
    answer read off that pin: the caller keeps acting -- builds, transfers, advanced
    frames -- while holding the instance, and a built sweep dict froze that board too.
    """

    def __init__(self, st):
        self.st = st
        me = st.player
        self.aim_from = (st.obj_h_angle[me], st.obj_v_angle[me])
        self._primary = None
        self._full = None
        self._pin = {}

    def _pinned(self, v_primary):
        """``(state, memo key)`` for one lattice, fixed at its first query."""
        got = self._pin.get(v_primary)
        if got is None:
            st = self.st
            key = (projector.scene_key(st), st.player, self.aim_from, v_primary)
            got = self._pin[v_primary] = (st.clone(), key)
        return got

    def _sweep(self, v_primary):
        """The lattice sweep, memoized across ticks: it is a pure function of the board
        and the facing it costs aims from, and a tick that gates out (a wait, a rejected
        step) re-enters with both unchanged -- the sweep is ~90% of a player's runtime.
        """
        st, key = self._pinned(v_primary)
        return projector.memo(
            _VIEW_CACHE,
            key,
            _VIEW_CACHE_MAX,
            lambda: _cheap_views(st, v_primary, self.aim_from),
        )

    def _one(self, tile, v_primary):
        """One tile's sweep entry: read off a sweep already in hand, else marched
        targeted (:func:`_cheap_view`) -- the same answer for a fraction of the rays."""
        built = self._primary if v_primary else self._full
        if built is not None:
            return built.get(tuple(tile))
        st, key = self._pinned(v_primary)
        built = _VIEW_CACHE.get(key)
        if built is not None:
            return built.get(tuple(tile))
        return projector.memo(
            _TILE_VIEW_CACHE,
            key + (tuple(tile),),
            _TILE_VIEW_CACHE_MAX,
            lambda: (_cheap_view(st, tile, v_primary, self.aim_from),),
        )[0]

    def primary(self):
        if self._primary is None:
            self._primary = self._sweep(True)
        return self._primary

    def band(self):
        """The full pitch-band landable views (down-looks included)."""
        if self._full is None:
            self._full = self._sweep(False)
        return self._full

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

    @staticmethod
    def advance_phases(st, phases):
        """Advance `st` through ``(frames, plotting)`` segments; the search prices a
        step with this so it evolves the world exactly as ``_fire`` will."""
        for frames, plotting in phases:
            n = int(round(frames))
            if n > 0:
                enemies.advance_frames(st, n, plotting=plotting)

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

    def _seen_now(self, exposed, full_only=False):
        """Whether an exposing enemy has the spot in its live cone right now --
        the never-place-in-enemy-view test.  `full_only` restricts to enemies
        with FULL sight (the drainers): the middle relaxation tier when no
        unseen tile exists at all (partial-without-tree cannot be damaged)."""
        st = self.st
        half = FOV_HALF + FOV_MARGIN
        return any(
            self._in_cone(ah, st.obj_h_angle[e], half)
            for e, ah, full in exposed
            if full or not full_only
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

    def _rotation_epochs(self):
        """Frame of each enemy's FIRST rotation, MEASURED on a clone of the live clock.

        ``rotation_cooldown * UNIT_FRAMES`` is not that instant -- the $1335 bresenham
        accumulator and the enemy's own $16E9 update gate both offset it, and a cd of 0
        read as t=0 put every interval a rotation early (ls335 (13,27): predicted clear at
        749, actually covered until ~800).  Every later gap is exactly one period."""
        st = self.st
        key = (
            bytes(
                st.mem[mm.ENEMIES_ROTATION_COOLDOWN : mm.ENEMIES_ROTATION_COOLDOWN + 8]
            )
            + bytes(st.mem[mm.ENEMIES_UPDATE_COOLDOWN : mm.ENEMIES_UPDATE_COOLDOWN + 8])
            + bytes([st.mem[mm.COOLDOWN_BRESENHAM], st.mem[mm.COOLDOWN_GATE]])
            + bytes(int(st.obj_h_angle[e]) for e in range(8))
        )
        got = _EPOCH_CACHE.get(key)
        if got is not None:
            return got
        clone = st.clone()
        clone.mem[mm.PLAYER_NOT_ACTED] = 0
        slots = list(enemies.enemy_slots(clone))
        was = {e: int(clone.obj_h_angle[e]) for e in slots}
        seen = {e: [] for e in slots}
        for frame in range(1, int(2.2 * ROT_PERIOD_FRAMES) + 2):
            enemies.advance_frame(clone)
            for e in slots:
                now = int(clone.obj_h_angle[e])
                if now != was[e]:
                    if len(seen[e]) < 2:
                        seen[e].append(float(frame))
                    was[e] = now
            if all(len(v) >= 2 for v in seen.values()):
                break
        epochs = {}
        for e in slots:  # a stalled or non-rotating enemy never sets one
            got = seen[e]
            first = got[0] if got else ROT_PERIOD_FRAMES
            period = got[1] - got[0] if len(got) > 1 else ROT_PERIOD_FRAMES
            epochs[e] = (first, period)
        if len(_EPOCH_CACHE) > _EPOCH_CACHE_MAX:
            _EPOCH_CACHE.clear()
        _EPOCH_CACHE[key] = epochs
        return epochs

    def _cover_intervals(self, e, angle_hi, half, horizon, epochs=None):
        """``[start, end)`` frame spans inside ``horizon`` where ``e``'s cone holds
        ``angle_hi``.  Facing only changes at a rotation boundary, so the calendar is a
        step function: facing after ``n`` rotations is ``facing0 + n*step``."""
        st = self.st
        facing = int(st.obj_h_angle[e])
        step = _signed(st.mem[mm.ROTATION_SPEED_TABLE + e])
        if step == 0:  # never rotates: its bearing decides once, for all time
            return [(0.0, horizon)] if self._in_cone(angle_hi, facing, half) else []
        epochs = self._rotation_epochs() if epochs is None else epochs
        first, period = epochs.get(e, (ROT_PERIOD_FRAMES, ROT_PERIOD_FRAMES))
        out = []
        n = 0
        while True:
            lo = 0.0 if n == 0 else first + (n - 1) * period
            if lo >= horizon:
                return out
            hi = first if n == 0 else first + n * period
            if self._in_cone(angle_hi, (facing + step * n) & 0xFF, half):
                out.append((lo, min(hi, horizon)))
            n += 1

    def _busy_spans(self, exposed, horizon):
        """The exposing enemies' cover intervals, unioned into disjoint busy spans."""
        half = FOV_HALF + FOV_MARGIN
        spans = []
        for e, angle_hi, full in exposed:
            if full:
                spans.extend(self._cover_intervals(e, angle_hi, half, horizon))
        if not spans:
            return []
        spans.sort(key=lambda s: (s[0], s[1]))
        merged = [list(spans[0])]
        for lo, hi in spans[1:]:
            if lo <= merged[-1][1]:
                merged[-1][1] = max(merged[-1][1], hi)
            else:
                merged.append([lo, hi])
        return merged

    def _earliest_start(self, tile, duration, exposed=None, horizon=None):
        """Frames to wait before a body on ``tile`` gets ``duration`` clear of drains.

        The scheduling primitive the gates lacked: a cone that cannot cover an action NOW
        is a DELAY, not a refusal.  ls335 (13,27) offers 450 f at once and 3304 f after
        ~1000 f of waiting, against a hop needing 1141 f.  ``inf`` if no phase ever fits.
        """
        return self._verify_starts(
            tile, duration, self._gap_starts(tile, duration, exposed, horizon)
        )

    def _gap_starts(self, tile, duration, exposed=None, horizon=None):
        """Candidate waits: the start of every calendar gap long enough to hold
        ``duration``.  A gap is longest at its own start, so only starts can be answers.
        """
        if exposed is None:
            exposed = self._exposing_enemies(tile)
        horizon = REVOLUTION_FRAMES if horizon is None else horizon
        out = []
        at = 0.0
        for lo, hi in self._busy_spans(exposed, horizon):
            if lo - at + DRAIN_DELAY >= duration:
                out.append(at)
            at = hi
        if horizon - at + DRAIN_DELAY >= duration:
            out.append(at)
        return out

    def _verify_starts(self, tile, duration, cands):
        """The first candidate the BIT-EXACT clock agrees on, else ``inf``.

        The calendar proposes and the clock disposes: ``_cone_onset`` assumes
        uninterrupted rotation, but a draining enemy stalls ($178C returns before the
        $17F9 rotate) and its cone then holds, so an unverified gap reads far too long.
        """
        real = self.st
        try:
            for at in cands:
                probe = real.clone()
                probe.mem[mm.PLAYER_NOT_ACTED] = 0
                if at:
                    enemies.advance_frames(probe, int(at))
                self.st = probe
                if self._gaze_window(tile) >= duration:
                    return at
        finally:
            self.st = real
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

    def _drain_units(self, budget, tile=None, exposed=None):
        """Energy `budget` frames of exposure costs a body that STAYS there.

        `DRAIN_DELAY` is only the FIRST drain.  A holder keeps its target while it stays
        visible ($178C returns before the $17F9 rotate) and re-arms to
        `UPDATE_COOLDOWN_DRAIN`, so the body is re-drained every `DRAIN_STEP` until it
        leaves -- SEER-INDEPENDENT: one holder is enough and more do not speed it up.

        Measured at the ls335 4-enemy stall, one seer and two seers alike: drains at
        150, 275, 375, 500, 625, 750, 875 f, gap 125 against DRAIN_STEP 112.4, energy
        13 -> 0 in 875 f.
        """
        if tile is None:
            tile = self.st.player_xy()
        if exposed is None:
            exposed = self._exposing_enemies(tile)
        window = self._gaze_window(tile, exposed=exposed)
        meanie = self._drains_in(self._meanie_window(tile, exposed), budget)
        if not any(full for _, _, full in exposed) or budget <= window:
            return meanie
        if window == math.inf:
            return meanie
        return max(meanie, int(math.floor((budget - window) / DRAIN_STEP)) + 1)

    def _affords_drains(self, n_drains, cost=0):
        """Whether the player can hand `n_drains` energy to the enemies on top of an
        action costing `cost` and stay alive above the survival floor (`_reserve`).
        `cost` is signed: an absorb RETURNS the object's energy ($1B9B), so it is
        priced as a negative cost -- the refuelling reclaim is exactly the move a
        drained body needs, and judging it on the energy it has not banked yet
        refuses it at the moment it matters.  Unexposed steps (`n_drains` 0) are
        never gated here."""
        if not n_drains:
            return True
        return self.st.energy - cost - n_drains >= self._reserve()

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

    def _margin(self, depth=None):  # pylint: disable=unused-argument
        """Frames of enemy-phase uncertainty a gate holds back.  Zero for a player
        that acts on the board in front of it; a lookahead planner overrides it with
        the accumulated per-step error at its own plan depth."""
        return 0.0

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

    def _affords(self, cost, budget, window=None):
        """Whether spending `cost` energy survives the drains `budget` frames of
        exposure bill at the $0C20/$1A31 rate, over `_reserve`.

        `budget` is the span the body stands exposed for -- a create's settle, a hop's
        whole build.  Executor and search both gate on this, at the same instant, so a
        plan cannot carry a step the executor then refuses."""
        if window is None:
            window = self._player_window()
        drains = self._drains_in(window, budget + self._margin())
        return self.st.energy - cost - drains >= self._reserve()

    def _player_window(self, exclude=None):
        """Frames until the player's OWN body is drainable (inf if never; no
        phantom), ignoring enemy `exclude` -- the one an absorb is about to
        remove, so absorbing an attacker still counts safe."""
        st = self.st
        if self._frozen():
            return math.inf  # no drain/meanie clock runs before the first action
        exposed = [x for x in self._exposures(st, st.player) if x[0] != exclude]
        if not exposed:
            return math.inf
        return self._gaze_window(st.player_xy(), exposed=exposed)

    # ------------------------------------------------------------------- aim
    def _aim_phases(self, view):
        """The aim as ordered ``(frames, plotting)`` segments, mechanism for mechanism.

        A same-bearing REUSE keeps sights on and drives the cursor from where it is;
        otherwise sights toggle off/on ($134C re-centres the cursor) before the coarse
        pan and a from-centre drive.  The toggle's replot and the pan's per-notch
        scroll+replot are foreground ($16B5 unreached); the u-turn taps, the cursor
        drive and the firing tap are gated main-loop scans, where it runs.
        """
        st = self.st
        me = st.player
        want = (view["h_angle"], view["v_angle"])
        h0, v0 = int(st.obj_h_angle[me]), int(st.obj_v_angle[me])
        nu = aimcost.h_press_count(h0, view["h_angle"])[0]
        if self.last_bearing == want:
            cur_from = self.cursor
            toggles = 0
        else:
            cur_from = SIGHTS_CENTRE
            toggles = TOGGLE_FRAMES
        # move_sights ($9958) steps cx and cy in ONE call: a diagonal drive costs max(|dx|,|dy|) gated scans, plus the $0CC8 ramp, and nothing at all when parked.
        cur = max(
            abs(view["cursor"][0] - cur_from[0]),
            abs(view["cursor"][1] - cur_from[1]),
        )
        cur = cur + CURSOR_RAMP if cur else 0.0
        # Each notch scrolls then replots the strip at its own intermediate angle; the u-turn ($1B2F EOR $80) is one keystroke with no scroll and no replot.
        pan = pancost.pan_frames(
            st, h0, v0, view["h_angle"], view["v_angle"], SCROLL, me
        )
        return (
            (toggles, True),
            (nu * UTURN_FRAMES, False),
            (pan, True),
            (cur + TAP_FRAMES, False),
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
        return (0.0 if reuse else TOGGLE_FRAMES) + nu * UTURN_FRAMES

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
            view = aim.propose(st, tile, v_band=True)
            if view is None or not aim.gate(st, view, tile):
                self.fire_reason = "aim_gate"
                return False
        if verb in ("boulder", "robot"):
            cost = mm.ENERGY_IN_OBJECTS[
                mm.T_BOULDER if verb == "boulder" else mm.T_ROBOT
            ]
            if not self._affords(cost, self._settle(verb, view)):
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
            self._advance(self._settle(verb, view), plotting=True)
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

    def _settle(self, verb, view=None, observer=None):
        """World frames the ROM advances AFTER `verb` fires; the placed object is
        on the board and exposable for this whole settle, so a danger window must
        cover the aim plus this before an enemy's cone can rotate in.

        A transfer moves the eye ($0C63) BEFORE the viewpoint full-redraw path
        ($357D) runs its two plot_world passes ($35C3/$35C6), so its settle is the
        projector cost of the scene as the NEW body sees it: observer `observer`
        (the transferred-into slot; default the already-moved ``st.player``) at
        THAT body's own bearing -- a created robot faces creator ^ $80 ($1BE0), not
        the aim `view`, which belongs to the abandoned eye.  `view` is unused for a
        transfer and prices nothing else here."""
        del view
        st = self.st
        if verb == "transfer":
            eye = st.player if observer is None else observer
            eye_view = {
                "h_angle": int(st.obj_h_angle[eye]),
                "v_angle": int(st.obj_v_angle[eye]),
            }
            return actioncost.FRAME_TICKS * projector.viewpoint_replot_frames(
                st, eye_view, eye
            )
        key = {"boulder": "create", "robot": "create"}.get(verb, verb)
        return actioncost.SETTLE.get(key, 60)

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
