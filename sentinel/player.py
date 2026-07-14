"""A reactive tick-by-tick greedy player: no search tree, no PRNG reads.

Each tick observes the live State, picks one action by fixed priorities (win
move > dissolve meanie > absorb Sentinel > transfer up > reclaim > climb >
wait), gates it through the ROM aim oracle, then advances the world in frames.
"""

import argparse
import math

from sentinel import actioncost, actions, aim, aimcost, enemies, los, memmap as mm
from sentinel import relative, terrain, threat
from sentinel.game import Game

REDRAW = actioncost.REDRAW_FRAMES  # one plot_world follows every pan notch
H_NOTCH_FRAMES = 16 + REDRAW  # $10EE: 16-step scroll per +-8 bearing notch
V_NOTCH_FRAMES = 8 + REDRAW  # $1135: 8-step scroll per +-4 pitch notch
UTURN_FRAMES = REDRAW  # $1B2F: instant EOR $80 flip + replot
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


class _Views:
    """Per-tick lazy cache of the keyboard-aim landable views.

    One primary ($F5-plane) sweep and at most one full pitch-band sweep per
    tick, replacing a per-candidate ``aim.propose`` full sweep each.
    """

    def __init__(self, st):
        self.st = st
        self._primary = None
        self._full = None

    def primary(self):
        if self._primary is None:
            self._primary, _ = los.landable_sweep_with_centres(self.st, v_primary=True)
        return self._primary

    def get(self, tile, band=False):
        """The landable view for `tile`, or None; `band` falls back to the
        full pitch-band sweep (down-looks at reclaim/endgame targets)."""
        view = self.primary().get(tuple(tile))
        if view is not None or not band:
            return view
        if self._full is None:
            self._full, _ = los.landable_sweep_with_centres(self.st, v_primary=False)
        return self._full.get(tuple(tile))


class Player:
    """One greedy reactive player bound to a live :class:`Game`.

    The only cross-tick memory is the tile of the hop in progress (so a
    half-built pedestal is not reclaimed) and the sights-cursor position.
    """

    def __init__(self, game, verbose=False):
        self.g = game
        self.st = game.state
        self.cursor = [80, 95]  # $134C sights-centre reset position
        self.hop_tile = None
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
    def _exposing_enemies(self, tile):
        """Enemy slots that could FULLY see a robot on `tile` at ANY facing."""
        clone = self.st.clone()
        slot = threat._free_slot(clone)
        if slot is None:
            return []
        old = terrain.tile_byte(clone, *tile)
        if not threat._place_phantom(clone, tile, slot):
            return []
        out = [
            e
            for e in enemies.enemy_slots(clone)
            if relative.can_see_object(clone, e, slot, mm.T_ROBOT, threat.FOV_FULL)[
                "full"
            ]
        ]
        threat._restore_tile(clone, tile, old, slot)
        return out

    def _gaze_window(self, tile, exposed=None):
        """Frames until some enemy's rotating scan cone covers a robot on
        `tile` while it has full line of sight there (0 == in a gaze now, inf
        == blocked from every facing).  Deterministic: facing, fixed rotation
        step and cooldown cadence only -- no PRNG."""
        st = self.st
        if exposed is None:
            exposed = self._exposing_enemies(tile)
        best = math.inf
        half = FOV_HALF + FOV_MARGIN
        for e in exposed:
            bearing = aimcost.bearing_to(st.obj_x[e], st.obj_y[e], tile[0], tile[1])
            if bearing is None:
                continue
            facing = st.obj_h_angle[e]
            if aimcost.angle_dist(bearing, facing) <= half:
                return 0.0
            step = _signed(st.mem[mm.ROTATION_SPEED_TABLE + e])
            if step == 0:
                continue
            first = st.mem[mm.ENEMIES_ROTATION_COOLDOWN + e] * UNIT_FRAMES
            for k in range(1, 256 // abs(step) + 2):
                facing = (facing + step) & 0xFF
                if aimcost.angle_dist(bearing, facing) <= half:
                    best = min(best, first + (k - 1) * ROT_PERIOD_FRAMES)
                    break
        return best

    def _player_window(self):
        """Gaze window of the player's own current body (no phantom)."""
        st = self.st
        me = st.player
        exposed = [
            e
            for e in enemies.enemy_slots(st)
            if relative.can_see_object(st, e, me, mm.T_ROBOT, threat.FOV_FULL)["full"]
        ]
        if not exposed:
            return math.inf
        return self._gaze_window(st.player_xy(), exposed=exposed)

    # ------------------------------------------------------------------- aim
    def _aim_frames(self, view):
        """Frames the keyboard pan from the current facing/cursor to `view`
        costs: u-turn-aware bearing notches, pitch notches, 1px/frame cursor."""
        st = self.st
        me = st.player
        nu, ns = aimcost.h_press_count(st.obj_h_angle[me], view["h_angle"])
        nv = aimcost.v_steps(st.obj_v_angle[me], view["v_angle"])
        cur = max(
            abs(view["cursor"][0] - self.cursor[0]),
            abs(view["cursor"][1] - self.cursor[1]),
        )
        return nu * UTURN_FRAMES + ns * H_NOTCH_FRAMES + nv * V_NOTCH_FRAMES + cur

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
        if not aim.gate(st, view, tile):
            view = aim.propose(st, tile, v_band=True)
            if view is None or not aim.gate(st, view, tile):
                return False
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

    # ------------------------------------------------------------- decisions
    def run(self, max_actions=300):
        """Play until won, dead, or `max_actions` decision ticks."""
        for _ in range(max_actions):
            self._observe()
            if actions.won(self.st):
                return True
            if actions.player_dead(self.st):
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
            self._advance(WAIT_FRAMES)
            return
        urgent = self._player_window() <= SAFE_FRAMES
        if self._meanie_response(views):
            return
        if not urgent and self._absorb_sentinel(views):
            return
        if self._transfer_up(views, urgent=urgent):
            return
        if not urgent and self._reclaim(views):
            return
        if self._climb(views, urgent=urgent):
            return
        if urgent and self._escape(views):
            return
        self._advance(WAIT_FRAMES)

    def _endgame(self, views):
        """Sentinel absorbed: robot on the platform, transfer in, hyperspace."""
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
            view = views.get(ptile)
            if view is None and self._sees_tile(ptile):
                view = views.get(ptile, band=True)
            if view is not None:
                return self._fire(verb, ptile, view)
        return self._climb(views, urgent=False, need_progress=True)

    def _meanie_response(self, views):
        """Absorb a live meanie before it lines up its forced hyperspace; if it
        is not landable, the normal transfer priorities dissolve it instead."""
        st = self.st
        for e in enemies.enemy_slots(st):
            meanie = st.mem[mm.ENEMIES_MEANIE_OBJECT + e]
            if meanie & 0x80:
                continue
            tile = st.tile_of(meanie)
            view = views.get(tile, band=True)
            if view is not None and self._fire("absorb", tile, view):
                return True
        return False

    def _absorb_sentinel(self, views):
        """Absorb the Sentinel dead last: only with the endgame affordable
        (robot 3 + hyperspace 3 - Sentinel's +4 => energy >= 2) and no live
        meanie (the absorb-lock would make it permanent)."""
        st = self.st
        if st.energy < 6 - mm.ENERGY_IN_OBJECTS[mm.T_SENTINEL]:
            return False
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
            window = self._gaze_window(tile)
            if window <= 0.0:
                continue
            if not urgent and window < SAFE_FRAMES:
                continue
            view = views.get(tile, band=True)
            if view is None:
                continue
            key = (eye, window)
            if best is None or key > best[0]:
                best = (key, tile, view)
        if best is None:
            return False
        _, tile, view = best
        if self._fire("transfer", tile, view):
            self.hop_tile = None
            return True
        return False

    def _reclaim(self, views):
        """Absorb invested/loose energy: old shells and spent pedestals (never
        the hop in progress), and trees while energy has headroom."""
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
            if otype == mm.T_ROBOT and self._base_z(slot) <= my_eye + EYE_EPS:
                prio = 0  # a shell we climbed out of: 3 energy, look-down aim
            elif otype == mm.T_BOULDER and self._base_z(slot) <= my_eye + EYE_EPS:
                prio = 1  # a spent pedestal: 2 energy
            elif otype == mm.T_TREE and want_trees:
                prio = 2
            else:
                continue
            cands.append((prio, slot, tile))
        cands.sort()
        for prio, _, tile in cands:
            view = views.get(tile)
            if view is None and prio < 2 and self._sees_tile(tile):
                view = views.get(tile, band=True)
            if view is None:
                continue
            if self._fire("absorb", tile, view):
                return True
        return False

    def _climb(self, views, urgent=False, need_progress=True):
        """One hop step toward height: robot on a tall-enough pedestal, another
        boulder on a short one, or a new boulder on the best safe tile."""
        st = self.st
        my_eye = self._my_eye()
        need = 0 if urgent else HOP_FRAMES
        best_robot = None
        best_boulder = None
        for tile, view in views.primary().items():
            if tile == st.player_xy():
                continue
            top = self._top(tile)
            if top is not None and st.obj_type[top] == mm.T_BOULDER:
                robot_eye = self._base_z(top) + BOULDER_H
                window = self._gaze_window(tile)
                if window <= 0.0 or window < need:
                    continue
                grows = robot_eye > my_eye + EYE_EPS
                if (grows or (urgent and window > SAFE_FRAMES)) and st.energy >= 3:
                    key = (robot_eye, window)
                    if best_robot is None or key > best_robot[0]:
                        best_robot = (key, tile, view)
                if not grows and tile == self.hop_tile and st.energy >= HOP_COST:
                    key = (robot_eye, window)
                    if best_boulder is None or key > best_boulder[0]:
                        best_boulder = (key, tile, view)
            elif top is None:
                if st.energy < HOP_COST:
                    continue
                if not actions.can_create(st, mm.T_BOULDER, tile):
                    continue
                robot_eye = self._robot_eye_after_boulder(tile)
                if need_progress and robot_eye <= my_eye + EYE_EPS and not urgent:
                    continue
                window = self._gaze_window(tile)
                if window <= 0.0 or window < need:
                    continue
                key = (robot_eye, window)
                if best_boulder is None or key > best_boulder[0]:
                    best_boulder = (key, tile, view)
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

    def _escape(self, views):
        """In the gaze with no safe option: least-bad transfer, else the truly
        last resort, an unsteerable hyperspace."""
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
