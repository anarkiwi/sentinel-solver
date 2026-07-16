"""A reactive tick-by-tick greedy player: no search tree, no PRNG reads.

Each tick observes the live State, picks one action by fixed priorities (win
move > dissolve meanie > absorb Sentinel > transfer up > reclaim > climb >
wait), gates it through the ROM aim oracle, then advances the world in frames.
"""

import argparse
import math
import os

import numpy as np

from sentinel import actioncost, actions, aim, aimcost, enemies, los, memmap as mm
from sentinel import relative, terrain, threat
from sentinel.game import Game

H_SCROLL = 16  # $10EE: 16-step horizontal scroll per +-8 bearing notch
V_SCROLL = 8  # $1135: 8-step vertical scroll per +-4 pitch notch
# Per-notch plot_world ($2625): per-pitch terrain base (~34, measured) + STEPS_PER_EDGE/edge.
REDRAW_BASE = float(os.environ.get("REDRAW_BASE", "34"))
CURSOR_PER_PX = 1.24  # sights-cursor rounds/pixel (gated move_sights, measured)
SIGHTS_CENTRE = (80, 95)  # $134C: a sights-ON toggle re-centres the cursor
TOGGLE_FRAMES = 12  # sights OFF (~0) + ON (~10: $134C recentre + plot_sights), measured
TAP_FRAMES = 3  # tap_action: idle full scan + press scan ($9678) + latch
UNIT_FRAMES = 3 * 256.0 / mm.COOLDOWN_BRESENHAM_STEP  # cooldown unit in frames
ROT_PERIOD_FRAMES = enemies.ROTATION_COOLDOWN_RELOAD * UNIT_FRAMES
FOV_HALF = enemies.FOV_SCAN // 2  # +-10 units of the enemy scan cone
FOV_MARGIN = 4  # safety margin on top of the cone half-width
ROBOT_EYE = 0.875  # a robot's eye above its foot tile ($E0 fraction)
BOULDER_H = 0.5  # a boulder raises a stack half a unit
HOP_COST = 5  # boulder (2) + robot (3)
HOP_FRAMES = 700  # gaze window a full hop (2 creates + transfer + aims) needs
SAFE_FRAMES = 250  # window below which the current tile is "urgent"
WAIT_FRAMES = 60  # idle advance when no action is available
EYE_EPS = 0.1  # minimum eye-height progress for a climb move


def _signed(b):
    return b - 256 if b >= 128 else b


def _cheap_views(st, v_primary, aim_from):
    """Landable views choosing, per tile, the MIN-AIM-COST keyboard view from
    facing ``aim_from`` (u-turn-aware bearing steps, then pitch steps, then
    cursor distance from the $134C recentre point) -- same tile membership as
    ``los.landable_sweep_with_centres``, cheapest representative view."""
    if not los._HAVE_JIT:
        return los.landable_sweep_with_centres(st, v_primary=v_primary)[0]
    hgrid = list(range(0, 256, los.AZIMUTH_STEP))
    vgrid = [los.KBD_V_ANGLE] if v_primary else los._V_PRIORITY
    cxs = los.CURSOR_CX
    cys = los.CURSOR_CY_FULL if v_primary else los.CURSOR_CY
    status, tx, ty, _, grids = los._landable_batch(
        st, st.player, None, 6000, hgrid, vgrid, cxs, cys
    )
    views = {}
    clear = np.flatnonzero(status == los.los_jit.LOS_CLEAR)
    if not clear.size:
        return views
    key = (tx[clear].astype(np.int64) << 16) | ty[clear].astype(np.int64)
    per_h, per_v = len(cxs) * len(cys), len(hgrid) * len(cxs) * len(cys)
    vi, rem = np.divmod(clear, per_v)
    hi, rem2 = np.divmod(rem, per_h)
    cxi, cyi = np.divmod(rem2, len(cys))
    h = np.asarray(hgrid)[hi]
    v = np.asarray(vgrid)[vi]
    cx = np.asarray(cxs)[cxi]
    cy = np.asarray(cys)[cyi]
    h0, v0 = aim_from
    dh = np.abs(((h - h0) + 128) % 256 - 128) // aimcost.AZIMUTH_STEP
    dv = np.abs(((v - v0) + 128) % 256 - 128) // aimcost.PITCH_STEP
    cur = np.maximum(np.abs(cx - los.SIGHTS_CX), np.abs(cy - los.SIGHTS_CY))
    cost = np.minimum(dh, 17 - dh) * 1000 + dv * 100 + cur
    order = np.lexsort((clear, cost, key))
    ks = key[order]
    head = order[np.concatenate(([True], ks[1:] != ks[:-1]))]
    for i in head[np.argsort(clear[head], kind="stable")]:
        hh, vv, cxx, cyy = los._meta_at(int(clear[i]), *grids)
        views[(int(key[i] >> 16), int(key[i] & 0xFFFF))] = {
            "h_angle": hh,
            "v_angle": vv,
            "cursor": [cxx, cyy],
        }
    return views


class _Views:
    """Per-tick lazy cache of the keyboard-aim landable views.

    One primary ($F5-plane) sweep and at most one full pitch-band sweep per
    tick, replacing a per-candidate ``aim.propose`` full sweep each; each
    tile's view is the cheapest-to-aim one from the player's current facing.
    """

    def __init__(self, st):
        self.st = st
        me = st.player
        self.aim_from = (st.obj_h_angle[me], st.obj_v_angle[me])
        self._primary = None
        self._full = None

    def primary(self):
        if self._primary is None:
            self._primary = _cheap_views(self.st, True, self.aim_from)
        return self._primary

    def band(self):
        """The full pitch-band landable views (down-looks included)."""
        if self._full is None:
            self._full = _cheap_views(self.st, False, self.aim_from)
        return self._full

    def get(self, tile, band=False):
        """The landable view for `tile`, or None; `band` falls back to the
        full pitch-band sweep (down-looks at reclaim/endgame targets)."""
        view = self.primary().get(tuple(tile))
        if view is not None or not band:
            return view
        return self.band().get(tuple(tile))


class Player:
    """One greedy reactive player bound to a live :class:`Game`.

    The only cross-tick memory is the tile of the hop in progress (so a
    half-built pedestal is not reclaimed) and the sights-cursor position.
    """

    def __init__(self, game, verbose=False):
        self.g = game
        self.st = game.state
        self.cursor = list(SIGHTS_CENTRE)
        self.last_bearing = None  # committed (h, v): a same-bearing aim reuses
        self.hop_tile = None
        self.endgame_waits = 0
        self.tick_window = math.inf  # own-tile gaze window, refreshed per tick
        self.frames = 0
        self.verbose = verbose
        self.trace = []

    # ------------------------------------------------------------------ clock
    def _advance(self, frames):
        frames = int(round(frames))
        enemies.advance_frames(self.st, frames)
        self.frames += frames

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

    def _gaze_window(self, tile, exposed=None):
        """Frames until an enemy that can DAMAGE a robot on `tile` has it in
        its rotating cone: full visibility drains ($1838); partial visibility
        counts only with a tree within 10 tiles (the meanie arm, $19C3).  0 ==
        in such a cone now; inf == never.  Deterministic -- no PRNG."""
        st = self.st
        if exposed is None:
            exposed = self._exposing_enemies(tile)
        dangerous = [x for x in exposed if x[2]]
        if len(dangerous) < len(exposed) and self._tree_near(tile):
            dangerous = exposed
        best = math.inf
        half = FOV_HALF + FOV_MARGIN
        for e, angle_hi, _ in dangerous:
            facing = st.obj_h_angle[e]
            if self._in_cone(angle_hi, facing, half):
                return 0.0
            step = _signed(st.mem[mm.ROTATION_SPEED_TABLE + e])
            if step == 0:
                continue
            first = st.mem[mm.ENEMIES_ROTATION_COOLDOWN + e] * UNIT_FRAMES
            for k in range(1, 256 // abs(step) + 2):
                facing = (facing + step) & 0xFF
                if self._in_cone(angle_hi, facing, half):
                    best = min(best, first + (k - 1) * ROT_PERIOD_FRAMES)
                    break
        return best

    def _frozen(self):
        """World frozen until the player's first action ($0CE5 bit7, $3682)."""
        return bool(self.st.mem[mm.PLAYER_NOT_ACTED] & 0x80)

    def _reserve(self):
        """Survival floor under any live threat: a forced (meanie) hyperspace
        spends the 3-energy robot cost and KILLS below it ($215F), so while any
        enemy or meanie exists a create must never leave energy under 3."""
        st = self.st
        if enemies.enemy_slots(st):
            return mm.ENERGY_IN_OBJECTS[mm.T_ROBOT]
        for s in range(mm.NUM_SLOTS):
            if not st.is_empty(s) and st.obj_type[s] == mm.T_MEANIE:
                return mm.ENERGY_IN_OBJECTS[mm.T_ROBOT]
        return 0

    def _player_window(self):
        """Gaze window of the player's own current body (no phantom)."""
        st = self.st
        if self._frozen():
            return math.inf  # no drain/meanie clock runs before the first action
        me = st.player
        exposed = self._exposures(st, me)
        if not exposed:
            return math.inf
        return self._gaze_window(st.player_xy(), exposed=exposed)

    # ------------------------------------------------------------------- aim
    def _aim_frames(self, view):
        """Frames the executor's aim method costs, mechanism for mechanism: a
        same-bearing REUSE keeps sights on and drives the cursor from where it
        is; otherwise sights toggle off/on (gated scans + replots, and $134C
        re-centres the cursor) before the coarse pan and a from-centre drive."""
        st = self.st
        me = st.player
        want = (view["h_angle"], view["v_angle"])
        nu, ns = aimcost.h_press_count(st.obj_h_angle[me], view["h_angle"])
        nv = aimcost.v_steps(st.obj_v_angle[me], view["v_angle"])
        if self.last_bearing == want:
            cur_from = self.cursor
            toggles = 0
        else:
            cur_from = SIGHTS_CENTRE
            toggles = TOGGLE_FRAMES
        cur = max(
            abs(view["cursor"][0] - cur_from[0]),
            abs(view["cursor"][1] - cur_from[1]),
        )
        # per-notch plot_world cost, geometric in the in-view object edges ($2625 scene patch)
        redraw = REDRAW_BASE + actioncost.STEPS_PER_EDGE * actioncost.visible_edges(
            st.mem, view
        )
        return (
            toggles
            + nu * redraw
            + ns * (H_SCROLL + redraw)
            + nv * (V_SCROLL + redraw)
            + cur * CURSOR_PER_PX
            + TAP_FRAMES
        )

    def _fire(self, verb, tile, view):
        """Aim (world advances), re-gate, apply `verb` on `tile`, settle.
        Returns False if the gate fails after the aim (the world changed under
        us) -- the caller just re-plans next tick."""
        st = self.st
        self._advance(self._aim_frames(view))
        if actions.player_dead(st):
            return False
        me = st.player
        st.obj_h_angle[me] = view["h_angle"]
        st.obj_v_angle[me] = view["v_angle"]
        self.cursor = list(view["cursor"])
        self.last_bearing = (view["h_angle"], view["v_angle"])
        if not aim.gate(st, view, tile):
            view = aim.propose(st, tile, v_band=True)
            if view is None or not aim.gate(st, view, tile):
                return False
        if verb in ("boulder", "robot"):
            cost = mm.ENERGY_IN_OBJECTS[
                mm.T_BOULDER if verb == "boulder" else mm.T_ROBOT
            ]
            if st.energy - cost < self._reserve():
                return False  # drained during the aim: creating now breaches the floor
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
            settle_verb = {"boulder": "create", "robot": "create"}.get(verb, verb)
            self._advance(actioncost.SETTLE.get(settle_verb, 60))
            self._log(verb, tile)
        return ok

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
        st = self.st
        views = _Views(st)
        if st.is_empty(actions.SENTINEL_SLOT):
            if self._endgame(views):
                return
            self._wait()
            return
        self.tick_window = self._player_window()
        urgent = self.tick_window <= SAFE_FRAMES
        if self._meanie_response(views):
            return
        if urgent and self._counterattack(views):
            return
        if not urgent and self._hunt_enemies(views):
            return
        if not urgent and self._absorb_sentinel(views):
            return
        if self._transfer_up(views, urgent=urgent):
            return
        if self.hop_tile is not None and self._climb(
            views, urgent=urgent, only_tile=self.hop_tile
        ):
            return
        if not urgent and self._reclaim(views):
            return
        if self._climb(views, urgent=urgent):
            return
        if urgent and st.energy < HOP_COST and self._reclaim(views, urgent=True):
            return  # cornered and poor: any cheap absorb is survival energy
        if urgent and self._escape(views):
            return
        if self._frozen() and self._climb(views, urgent=True):
            return  # waiting cannot change a frozen world: take the least-bad hop
        self._wait()

    def _wait(self):
        """Idle one beat and re-observe; the live driver overrides with real time."""
        self._advance(WAIT_FRAMES)

    def _endgame(self, views):
        """Sentinel absorbed: robot on the platform, transfer in, hyperspace.
        The platform strike obeys the placement invariant too: wait for the
        surviving sentries' cones to rotate off the platform, striking exposed
        only once a full rotation cycle has shown no window (no other choice)."""
        st = self.st
        ptile = st.platform_xy
        if actions.on_platform(st):
            self._hyperspace()
            return True
        top = self._top(ptile)
        if top is not None and st.obj_type[top] in (mm.T_ROBOT, mm.T_PLATFORM):
            verb = "transfer" if st.obj_type[top] == mm.T_ROBOT else "robot"
            if verb == "robot" and st.energy < 6:
                return False
            if self._gaze_window(ptile) < SAFE_FRAMES:
                self.endgame_waits += 1
                if self.endgame_waits * WAIT_FRAMES <= ROT_PERIOD_FRAMES:
                    return False  # let the cone rotate off the platform first
            view = views.get(ptile)
            if view is None and self._sees_tile(ptile):
                view = views.get(ptile, band=True)
            if view is not None:
                return self._fire(verb, ptile, view)
        return self._climb(views, urgent=False, need_progress=True)

    def _meanie_faces_window(self, meanie):
        """Frames until the meanie rotates to face the player ($16F2 turns it
        +-8 units toward us per ~10-unit update reload) -- the budget a cheap
        absorb of it must fit inside."""
        st = self.st
        ra = relative.relative_angles(st, meanie, st.player)
        gap = aimcost.angle_dist(ra["angle_hi"], st.obj_h_angle[meanie]) - FOV_HALF
        if gap <= 0:
            return 0.0
        steps = -(-gap // enemies.MEANIE_ROTATE_STEP)
        return steps * enemies.UPDATE_COOLDOWN_MEANIE_ROTATE * UNIT_FRAMES

    def _meanie_response(self, views):
        """Absorb a live meanie if the aim is CHEAP -- it must land before the
        meanie rotates to face us; otherwise the transfer-out dissolve (the
        normal climb priorities) outruns it instead."""
        st = self.st
        for e in enemies.enemy_slots(st):
            meanie = st.mem[mm.ENEMIES_MEANIE_OBJECT + e]
            if meanie & 0x80:
                continue
            tile = st.tile_of(meanie)
            view = views.get(tile)
            if view is None and self._sees_tile(tile):
                view = views.get(tile, band=True)
            if view is None:
                continue
            if self._aim_frames(view) >= self._meanie_faces_window(meanie):
                continue
            if self._fire("absorb", tile, view):
                return True
        return False

    def _counterattack(self, views):
        """Seen by an absorbable enemy: absorb IT instead of fleeing.  The
        budget is the enemy's own drain countdown ($0C20, ~120 units of grace
        before the first drain, $1838); absorbs have no facing requirement and
        a u-turn aim costs one keystroke.  The Sentinel qualifies only as the
        last enemy standing (the $1B8E lock) with the endgame affordable."""
        st = self.st
        me = st.player
        half = FOV_HALF + FOV_MARGIN
        others = len(enemies.enemy_slots(st)) > 1
        for e in enemies.enemy_slots(st):
            see = relative.can_see_object(st, e, me, mm.T_ROBOT, threat.FOV_FULL)
            if not see["exposure"]:
                continue
            ah = relative.relative_angles(st, e, me)["angle_hi"]
            if not self._in_cone(ah, st.obj_h_angle[e], half):
                continue
            if e == actions.SENTINEL_SLOT and (others or st.energy < 2):
                continue
            tile = st.tile_of(e)
            view = views.get(tile)
            if view is None and self._sees_tile(tile):
                view = views.get(tile, band=True)
            if view is None:
                continue
            budget = st.mem[mm.ENEMIES_DRAINING_COOLDOWN + e] * UNIT_FRAMES
            if budget and self._aim_frames(view) >= budget:
                continue  # cannot land the absorb before its drain fires
            if self._fire("absorb", tile, view):
                return True
        return False

    def _hunt_enemies(self, views):
        """Absorb sentries whenever landable within our own safety window,
        cheapest aim first: each one PERMANENTLY deletes a rotating gaze
        (worth far more than its +3), and the $1B8E absorb-lock forces every
        enemy absorb to precede the Sentinel's anyway."""
        st = self.st
        cands = []
        for e in enemies.enemy_slots(st):
            if e == actions.SENTINEL_SLOT:
                continue
            tile = st.tile_of(e)
            if self._top(tile) != e:
                continue
            view = views.get(tile)
            if view is None and self._sees_tile(tile):
                view = views.get(tile, band=True)
            if view is None:
                continue
            aimf = self._aim_frames(view)
            if aimf + SAFE_FRAMES >= self.tick_window:
                continue  # a hunt aim must not consume the escape margin
            cands.append((aimf, tile, view))
        cands.sort(key=lambda c: c[0])
        for _, tile, view in cands:
            if self._fire("absorb", tile, view):
                return True
        return False

    def _absorb_sentinel(self, views):
        """Absorb the Sentinel dead last ($1B8E lock): only once no other
        enemy or meanie remains absorbable, with the endgame affordable
        (robot 3 + hyperspace 3 - Sentinel's +4 => energy >= 2)."""
        st = self.st
        if st.energy < 6 - mm.ENERGY_IN_OBJECTS[mm.T_SENTINEL]:
            return False
        if len(enemies.enemy_slots(st)) > 1:
            return False  # sentries first: the lock would strand them forever
        if any(
            not st.mem[mm.ENEMIES_MEANIE_OBJECT + e] & 0x80
            for e in enemies.enemy_slots(st)
        ):
            return False
        stile = st.tile_of(actions.SENTINEL_SLOT)
        view = views.get(stile)
        if view is None and self._my_eye() > self._base_z(actions.SENTINEL_SLOT):
            view = views.get(stile, band=True)  # down-look at the Sentinel only
        if view is None:
            return False
        return self._fire("absorb", stile, view)

    def _transfer_up(self, views, urgent=False):
        """Transfer into the highest landable robot that raises the eye --
        never into a gaze, and only with a safe window unless urgent."""
        st = self.st
        my_eye = self._my_eye()
        best = None
        for slot in range(mm.NUM_SLOTS):
            if st.is_empty(slot) or slot == st.player:
                continue
            if st.obj_type[slot] != mm.T_ROBOT:
                continue
            tile = st.tile_of(slot)
            if self._top(tile) != slot:
                continue
            eye = self._base_z(slot)
            if eye <= my_eye + EYE_EPS and not urgent:
                continue
            exposed = self._exposing_enemies(tile)
            window = self._gaze_window(tile, exposed=exposed)
            if window <= 0.0:
                continue  # dangerous live view: same basis as the build gate
            view = views.get(tile, band=True)
            if view is None:
                continue
            aimf = self._aim_frames(view)
            if not urgent and (aimf >= self.tick_window or window < aimf + SAFE_FRAMES):
                continue  # the destination clock starts at ARRIVAL, after the aim
            key = (eye, -aimf, window)
            if best is None or key > best[0]:
                best = (key, tile, view)
        if best is None:
            return False
        _, tile, view = best
        if self._fire("transfer", tile, view):
            self.hop_tile = None
            return True
        return False

    def _reclaim(self, views, urgent=False):
        """Absorb invested/loose energy: old shells and spent pedestals (never
        the hop in progress), and trees while energy has headroom.  Urgent
        reclaims drop the follow-up-hop reservation: the energy IS the move."""
        st = self.st
        my_eye = self._my_eye()
        want_trees = st.energy < HOP_COST + 6
        cands = []
        for slot in range(mm.NUM_SLOTS):
            if st.is_empty(slot) or slot == st.player:
                continue
            otype = st.obj_type[slot]
            tile = st.tile_of(slot)
            if tile == self.hop_tile or tile == st.player_xy():
                continue
            if self._top(tile) != slot:
                continue
            if otype in (mm.T_ROBOT, mm.T_BOULDER):
                if self._base_z(slot) > my_eye + EYE_EPS:
                    continue  # a live pedestal above us, not a spent one
            elif otype != mm.T_TREE or not want_trees:
                continue
            cands.append((mm.ENERGY_IN_OBJECTS[otype], tile))
        landable = []
        for value, tile in cands:
            view = views.get(tile)
            if view is None and self._sees_tile(tile):
                view = views.get(tile, band=True)  # near/below targets pitch down
            if view is None:
                continue
            aimf = self._aim_frames(view)
            if not urgent and aimf + HOP_FRAMES >= self.tick_window:
                continue  # optional: must leave room for the hop that follows
            landable.append((aimf / value, tile, view))
        landable.sort(key=lambda c: c[0])  # cheapest aim frames per energy unit
        for _, tile, view in landable:
            if self._fire("absorb", tile, view):
                return True
        return False

    def _climb(self, views, urgent=False, need_progress=True, only_tile=None):
        """One hop step toward height: robot on a tall-enough pedestal, another
        boulder on a short one, or a new boulder on the best safe tile.
        `only_tile` restricts to the hop in progress (finish before roaming).
        A primary-plane scan with no candidate falls back to the full pitch
        band -- from a hollow every buildable tile needs a down-pitch aim."""
        best_robot, best_boulder = self._climb_scan(
            views.primary(), urgent, need_progress, only_tile
        )
        if best_robot is None and best_boulder is None:
            best_robot, best_boulder = self._climb_scan(
                views.band(), urgent, need_progress, only_tile
            )
        if best_robot is None and best_boulder is None:
            best_robot, best_boulder = self._climb_scan(
                views.band(), urgent, need_progress, only_tile, seen_tier=1
            )
        if best_robot is None and best_boulder is None and (urgent or self._frozen()):
            best_robot, best_boulder = self._climb_scan(
                views.band(), urgent, need_progress, only_tile, seen_tier=2
            )
        if best_robot is not None and not self._no_strand(best_robot[1]):
            best_robot = None  # completing this pedestal would strand our reclaims
        if best_robot is not None:
            _, tile, view = best_robot
            if self._fire("robot", tile, view):
                self.hop_tile = tile
                return True
        if best_boulder is not None:
            _, tile, view = best_boulder
            if self._fire("boulder", tile, view):
                self.hop_tile = tile
                return True
        return False

    def _no_strand(self, tile):
        """Whether completing the pedestal at `tile` is energy-safe: either the
        next hop stays affordable without reclaims, or the shell we abandon is
        KEYBOARD-LANDABLE (not merely geometrically visible) from up there."""
        st = self.st
        if st.energy - mm.ENERGY_IN_OBJECTS[mm.T_ROBOT] >= HOP_COST + self._reserve():
            return True
        clone = st.clone()
        slot = threat._free_slot(clone)
        if slot is None or not threat._place_phantom(clone, tile, slot):
            return False
        views, _ = los.landable_sweep_with_centres(clone, slot=slot, v_primary=False)
        return tuple(st.player_xy()) in views

    def _hunt_target(self):
        """The tile the climb works toward LOS on: the NEAREST sentry first
        (the hunt order -- each one absorbed deletes a gaze), the Sentinel
        once it stands alone, the platform for the endgame strike."""
        st = self.st
        px, py = st.player_xy()
        sentries = [e for e in enemies.enemy_slots(st) if e != actions.SENTINEL_SLOT]
        if sentries:
            near = min(
                sentries,
                key=lambda e: (st.obj_x[e] - px) ** 2 + (st.obj_y[e] - py) ** 2,
            )
            return st.tile_of(near)
        if not st.is_empty(actions.SENTINEL_SLOT):
            return st.tile_of(actions.SENTINEL_SLOT)
        return st.platform_xy

    def _tile_sees_target(self, tile, target):
        """Whether a robot standing on `tile` would see `target`'s tile (a
        phantom-observer $1CDD march) -- the LOS-progress term of a hop."""
        clone = self.st.clone()
        slot = threat._free_slot(clone)
        if slot is None or not threat._place_phantom(clone, tile, slot):
            return False
        return threat.player_sees_tile(clone, target, slot)

    def _climb_scan(self, cand_views, urgent, need_progress, only_tile, seen_tier=0):
        """Scan `cand_views` for the best pedestal-robot and boulder builds.
        Progress = gaining LOS on the hunt target first, then eye height (an
        equal/lower hop is legal exactly when it buys the target sight line).
        Budget: hops start only when boulder + robot + reserve are affordable.
        `seen_tier`: 0 = unseen tiles only; 1 = tolerate partial sight with a
        safe horizon (undrainable); 2 = no-other-choice (least-exposed)."""
        st = self.st
        my_eye = self._my_eye()
        need = 0 if urgent else HOP_FRAMES
        reserve = self._reserve()
        robot_min = mm.ENERGY_IN_OBJECTS[mm.T_ROBOT] + reserve
        hop_min = HOP_COST + reserve
        target = self._hunt_target()
        here = st.player_xy()
        best_robot = None
        best_boulder = None
        for tile, view in cand_views.items():
            if tile == here:
                continue
            if only_tile is not None and tile != only_tile:
                continue
            top = self._top(tile)
            if top is not None and st.obj_type[top] == mm.T_BOULDER:
                robot_eye = self._base_z(top) + BOULDER_H
                aimf = self._aim_frames(view)
                exposed = self._exposing_enemies(tile)
                if self._seen_now(exposed, full_only=seen_tier >= 1) and seen_tier < 2:
                    continue  # never build under a live view, even a partial one
                window = self._gaze_window(tile, exposed=exposed)
                if window < aimf + need and seen_tier < 2:
                    continue  # the destination clock starts at ARRIVAL
                if not urgent and aimf >= self.tick_window:
                    continue  # the aim itself outlasts our own tile's safety
                sees = self._tile_sees_target(tile, target)
                grows = robot_eye > my_eye + EYE_EPS or sees
                if (
                    grows or (urgent and window > SAFE_FRAMES)
                ) and st.energy >= robot_min:
                    key = (sees, robot_eye, -aimf, window)
                    if best_robot is None or key > best_robot[0]:
                        best_robot = (key, tile, view)
                if not grows and tile == self.hop_tile and st.energy >= hop_min:
                    key = (sees, robot_eye, -aimf, window)
                    if best_boulder is None or key > best_boulder[0]:
                        best_boulder = (key, tile, view)
            elif top is None:
                if st.energy < hop_min:
                    continue
                if not actions.can_create(st, mm.T_BOULDER, tile):
                    continue
                aimf = self._aim_frames(view)
                if not urgent and aimf >= self.tick_window:
                    continue
                exposed = self._exposing_enemies(tile)
                if self._seen_now(exposed, full_only=seen_tier >= 1) and seen_tier < 2:
                    continue
                window = self._gaze_window(tile, exposed=exposed)
                if window < aimf + need and seen_tier < 2:
                    continue
                robot_eye = self._robot_eye_after_boulder(tile)
                sees = self._tile_sees_target(tile, target)
                if (
                    need_progress
                    and robot_eye <= my_eye + EYE_EPS
                    and not sees
                    and not urgent
                ):
                    continue
                key = (sees, robot_eye, -aimf, window)
                if best_boulder is None or key > best_boulder[0]:
                    best_boulder = (key, tile, view)
        return best_robot, best_boulder

    def _escape(self, views):
        """In the gaze with no safe option: a transfer that STRICTLY improves
        the danger window (a lateral swap just resets nothing and burns time),
        else the truly last resort, an unsteerable hyperspace."""
        st = self.st
        best = None
        for slot in range(mm.NUM_SLOTS):
            if st.is_empty(slot) or slot == st.player:
                continue
            if st.obj_type[slot] != mm.T_ROBOT:
                continue
            tile = st.tile_of(slot)
            if self._top(tile) != slot:
                continue
            view = views.get(tile, band=True)
            if view is None:
                continue
            window = self._gaze_window(tile)
            if window <= self.tick_window:
                continue  # no improvement: not an escape
            if best is None or window > best[0]:
                best = (window, tile, view)
        if best is not None and self._fire("transfer", best[1], best[2]):
            self.hop_tile = None
            return True
        if st.energy >= mm.ENERGY_IN_OBJECTS[mm.T_ROBOT]:
            self._hyperspace()
            return True
        return False


def main():
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("landscape", nargs="?", type=int, default=0)
    parser.add_argument("--max-actions", type=int, default=300)
    parser.add_argument("--quiet", action="store_true")
    args = parser.parse_args()
    game = Game.new(args.landscape)
    player = Player(game, verbose=not args.quiet)
    won = player.run(max_actions=args.max_actions)
    print(
        f"landscape {args.landscape}: {'WON' if won else 'lost'} "
        f"in {len(player.trace)} actions / {player.frames} frames, "
        f"energy {game.energy}, dead={actions.player_dead(game.state)}"
    )
    return 0 if won else 1


if __name__ == "__main__":
    raise SystemExit(main())
