"""A reactive tick-by-tick greedy player: no search tree, no PRNG reads.

Each tick observes the live State, picks one action by fixed priorities (win
move > dissolve meanie > absorb Sentinel > transfer up > reclaim > climb >
wait), gates it through the ROM aim oracle, then advances the world in frames.
"""

import argparse
import math

from sentinel import actions, aimcost, enemies, los, memmap as mm, relative, threat
from sentinel.game import Game
from sentinel.playerbase import (
    BasePlayer,
    BOULDER_H,
    EYE_EPS,
    FOV_HALF,
    FOV_MARGIN,
    HOP_COST,
    HOP_FRAMES,
    ROT_PERIOD_FRAMES,
    SAFE_FRAMES,
    UNIT_FRAMES,
    WAIT_FRAMES,
    _Views,
)


class Player(BasePlayer):
    """One greedy reactive player: each tick picks one action by fixed
    priorities (win move > dissolve meanie > absorb Sentinel > transfer up >
    reclaim > climb > wait), gated through the ROM aim oracle."""

    def __init__(self, game, verbose=False, audit=False):
        super().__init__(game, verbose=verbose, audit=audit)
        self.hop_tile = None
        self.endgame_waits = 0
        self.tick_window = math.inf  # own-tile gaze window, refreshed per tick

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
            view = views.get(tile, band=True)
            if view is None:
                continue
            aimf = self._aim_frames(view)
            settle = self._settle("transfer", view)
            if not self._drain_gate("transfer", tile, exposed, aimf + settle):
                continue  # a body drainable during the aim or the post-transfer settle
            window = self._gaze_window(tile, exposed=exposed)
            arrival = window - aimf - settle
            if not urgent and (aimf >= self.tick_window or arrival < SAFE_FRAMES):
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

    def _climb_scan(self, cand_views, urgent, need_progress, only_tile, seen_tier=0):
        """Scan `cand_views` for the best pedestal-robot and boulder builds.
        Progress = gaining LOS on the hunt target first, then eye height (an
        equal/lower hop is legal exactly when it buys the target sight line).
        Budget: hops start only when boulder + robot + reserve are affordable.
        `seen_tier`: 0 = unseen tiles only; 1 = tolerate partial sight with a
        safe horizon (undrainable); 2 = no-other-choice (least-exposed)."""
        st = self.st
        my_eye = self._my_eye()
        # even an urgent build must not be seen DURING its own create settle
        need = self._settle("robot") if urgent else HOP_FRAMES
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
                if not urgent and aimf >= self.tick_window:
                    continue  # the aim itself outlasts our own tile's safety
                exposed = self._exposing_enemies(tile)
                window = self._gaze_window(tile, exposed=exposed)
                # a robot body is never left in a live FULL-sight cone; a boulder-raise is drain-exempt
                robot_ok = seen_tier >= 2 or self._drain_gate(
                    "robot", tile, exposed, aimf + need
                )
                sees = self._tile_sees_target(tile, target)
                grows = robot_eye > my_eye + EYE_EPS or sees
                if (
                    robot_ok
                    and (grows or (urgent and window > SAFE_FRAMES))
                    and st.energy >= robot_min
                ):
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
                # a bare-tile boulder starts a hop: gate on the FUTURE robot's full-sight safety, not a partial glimpse
                if seen_tier < 2 and not self._drain_gate(
                    "robot", tile, exposed, aimf + need
                ):
                    continue
                window = self._gaze_window(tile, exposed=exposed)
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
            arrival = (
                self._gaze_window(tile)
                - self._aim_frames(view)
                - self._settle("transfer", view)
            )
            if arrival <= self.tick_window:
                continue  # seen during the hop, or no improvement: not an escape
            if best is None or arrival > best[0]:
                best = (arrival, tile, view)
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
    parser.add_argument(
        "--audit",
        action="store_true",
        help="strict post-settle invariant accounting: flag any create/transfer "
        "that ended in a live enemy cone",
    )
    args = parser.parse_args()
    game = Game.new(args.landscape)
    player = Player(game, verbose=not args.quiet, audit=args.audit)
    won = player.run(max_actions=args.max_actions)
    print(
        f"landscape {args.landscape}: {'WON' if won else 'lost'} "
        f"in {len(player.trace)} actions / {player.frames} frames, "
        f"energy {game.energy}, dead={actions.player_dead(game.state)}"
    )
    if args.audit:
        print(f"invariant breaches: {len(player.breaches)}")
        for f, verb, tile, seen in player.breaches:
            print(f"  f={f} {verb} {tile} seen_by={seen}")
    return 0 if won else 1


if __name__ == "__main__":
    raise SystemExit(main())
