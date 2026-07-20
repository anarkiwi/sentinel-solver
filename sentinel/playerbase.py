"""Shared machinery for players over the sentinel model: world clock, geometry,
threat/gaze windows, aim cost, action firing, and the run loop. Player
subclasses implement _tick()."""

import math
import os

import numpy as np

from sentinel import actioncost, actions, aim, aimcost, enemies, los, memmap as mm
from sentinel import pancost, projector, relative, terrain, threat

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
DRAIN_DELAY = 120.0 * UNIT_FRAMES  # $0C20: first-seen -> first drain countdown
MEANIE_SPAWN_FRAMES = enemies.UPDATE_COOLDOWN_MEANIE_MADE * UNIT_FRAMES  # $1869 hold
MEANIE_ARM_FRAMES = (
    (128 // enemies.MEANIE_ROTATE_STEP)
    * enemies.UPDATE_COOLDOWN_MEANIE_ROTATE
    * UNIT_FRAMES
)  # $171B worst-case meanie rotate-to-face the player


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

    def _sweep(self, v_primary):
        """The lattice sweep, memoized across ticks: it is a pure function of the board
        and the facing it costs aims from, and a tick that gates out (a wait, a rejected
        step) re-enters with both unchanged -- the sweep is ~90% of a player's runtime.
        """
        st = self.st
        return projector.memo(
            _VIEW_CACHE,
            (projector.scene_key(st), st.player, self.aim_from, v_primary),
            _VIEW_CACHE_MAX,
            lambda: _cheap_views(st, v_primary, self.aim_from),
        )

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
        view = self.primary().get(tuple(tile))
        if view is not None or not band:
            return view
        return self.band().get(tuple(tile))


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
        step never brings it round.  Deterministic -- no PRNG."""
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

    def _meanie_window(self, tile, exposed):
        """Frames until a meanie armed from PARTIAL sight of a body on `tile`
        could force a hyperspace ($1986): a partially-seeing enemy must rotate its
        cone on, run the ~120-round drain countdown to the meanie branch
        ($183D/$1852), spawn the meanie ($1869) and rotate it to face ($171B).
        Needs a tree within 10 tiles ($19C3); inf otherwise.  Always far slower
        than a drain -- this clock is NEVER 0."""
        if not self._tree_near(tile):
            return math.inf
        half = FOV_HALF + FOV_MARGIN
        onset = min(
            (self._cone_onset(e, ah, half) for e, ah, full in exposed if not full),
            default=math.inf,
        )
        if onset == math.inf:
            return math.inf
        return onset + DRAIN_DELAY + MEANIE_SPAWN_FRAMES + MEANIE_ARM_FRAMES

    def _gaze_window(self, tile, exposed=None):
        """Frames until an enemy can DRAIN a robot body on `tile`.  Only FULL
        sight drains ($1838; $16E6 step 3 skips a partially-seen robot), so the
        drain clock keys on full-sight enemies rotating their cone on.  The
        partial+tree path is the MEANIE arm ($19C3) -- a SEPARATE, far slower
        clock (`_meanie_window`), never an immediate drain.  0 == drainable now;
        inf == never.  Deterministic -- no PRNG."""
        if exposed is None:
            exposed = self._exposing_enemies(tile)
        half = FOV_HALF + FOV_MARGIN
        best = self._meanie_window(tile, exposed)
        for e, angle_hi, full in exposed:
            if not full:
                continue
            best = min(best, self._cone_onset(e, angle_hi, half))
            if best == 0.0:
                break
        return best

    def _drain_gate(self, verb, tile, exposed=None, budget=0.0):
        """Whether placing `verb` on `tile` is drain-safe.  A boulder is exempt
        ($16E6 drains robots only, never a boulder body); a robot/transfer must be
        clear of every live FULL-sight cone now and keep its full-sight drain
        window past `budget` (the aim+settle it stands exposed).  Partial sight is
        not a drain -- its slower meanie arm is priced into `_gaze_window`."""
        if verb == "boulder":
            return True
        if exposed is None:
            exposed = self._exposing_enemies(tile)
        if self._seen_now(exposed, full_only=True):
            return False
        return self._gaze_window(tile, exposed=exposed) >= budget

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
        return toggles + nu * TAP_FRAMES + pan + cur + TAP_FRAMES

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
        us) -- the caller just re-plans next tick."""
        st = self.st
        self._advance(self._step_aim_frames(verb, view))
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
            self._advance(self._settle(verb, view))
            if self.audit and verb in ("boulder", "robot", "transfer"):
                self._account(verb, tile)
            self._log(verb, tile)
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
