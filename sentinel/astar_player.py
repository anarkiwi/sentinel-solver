"""An A* / best-first planning player over the sentinel model.

Enemies only rotate, so each tile has a closed-form window (frames until a cone
rotates onto it); the search carries the cheap enemy phase, gates moves on
window >= aim+settle, and defers the keyboard-aim sweep to execution.
"""

import argparse
import heapq
import math
import time

import numpy as np

from sentinel import (
    actioncost,
    actions,
    enemies,
    los,
    memmap as mm,
    terrain,
    threat,
)
from sentinel.game import Game
from sentinel.playerbase import (
    BasePlayer,
    BOULDER_H,
    EYE_EPS,
    FOV_HALF,
    FOV_MARGIN,
    HOP_COST,
    HOP_FRAMES,
    SAFE_FRAMES,
    SIGHTS_CENTRE,
    TAP_FRAMES,
    UNIT_FRAMES,
)

_ROBOT_EYE = 0.875
_MARGIN = SAFE_FRAMES  # window slack a landed body needs beyond the action
_DRAIN_DELAY = 120.0 * UNIT_FRAMES  # $0C20: frames from first-seen to first drain
_RADIUS = 16  # tile scan radius for build candidates around the player
# Per-op h floors from charged primitives: min aim == tap_action latch + per-verb settle floor.
_AIM_FLOOR = float(TAP_FRAMES)  # minimal aim (_aim_frames with nu=ns=nv=cur=0)
_OP_FLOOR = {
    v: _AIM_FLOOR + s for v, s in actioncost.SETTLE.items()
}  # absorb/create/transfer
_HYPERSPACE_FLOOR = _AIM_FLOOR + actioncost.SETTLE.get(
    "hyperspace", 60
)  # _settle default
_ABSORB_EST = _OP_FLOOR["absorb"]  # 1 absorb per remaining enemy
_HOP_EST = (
    2 * _OP_FLOOR["create"] + _OP_FLOOR["transfer"]
)  # >=1 boulder + robot + transfer
_ENDGAME_EST = (
    _OP_FLOOR["create"] + _OP_FLOOR["transfer"] + _HYPERSPACE_FLOOR
)  # robot+xfer+hs
_EYE_PER_HOP = 0.9
_TARGET_EYE = 9.0
_MAX_BOULDERS = 12
_HOP_BOULDERS = 2  # human-win k distribution is {1:27,2:3} (ls42.json et al): never >2
_TOP_TARGETS = 4  # enemies a node may branch a directed pursuit toward
_TOP_HOPS = 8  # ranked pedestal candidates a pursuit tries per climb step
_MAX_PURSUE = 40  # inner hop/reclaim steps one pursuit macro may chain


class _Node:
    """A search node: a state, its estimated cost g, and the path that made it."""

    __slots__ = ("state", "g", "path", "key", "last_bearing", "cursor")

    def __init__(self, state, g, path, key, last_bearing=None, cursor=None):
        self.state = state
        self.g = g
        self.path = path
        self.key = key
        self.last_bearing = last_bearing
        self.cursor = list(cursor) if cursor is not None else list(SIGHTS_CENTRE)


class AStarPlayer(BasePlayer):
    """Search a winning line once, then execute it."""

    def __init__(
        self,
        game,
        verbose=False,
        audit=False,
        node_budget=200000,
        time_budget=30.0,
        weight=1.4,
    ):
        super().__init__(game, verbose=verbose, audit=audit)
        self.node_budget = node_budget
        self.time_budget = time_budget
        self.weight = weight
        self.plan = None
        self._pi = 0
        self.expansions = 0
        self._deadline = None  # run-wide wall-clock deadline (set on first search)
        self._land_memo = {}  # search: coarse landable tile-sets
        self._view_memo = {}  # per-sig $F5-plane view dicts (band via targeted march)
        self._hs_streak = 0  # consecutive last-resort hyperspaces (spiral guard)

    # ---------------------------------------------------------------- execute
    def _tick(self):
        if self.plan is None:
            self.plan = self._search()
            self._pi = 0
            if self.verbose:
                print(f"  plan ({self.expansions} nodes): {self.plan}")
        if not self._frozen() and self._react():
            self.plan = None  # deviated for survival: re-plan from the new state
            return
        if not self.plan or self._pi >= len(self.plan):
            self._wait()
            return
        verb, tile = self.plan[self._pi]
        if verb == "hyperspace":
            self._hyperspace()
            self._pi += 1
            return
        view = self._view_for(tile)
        if view is None or not self._fire(verb, tile, view):
            self.plan = self._search()  # live/plan divergence: re-plan
            self._pi = 0
            if not self.plan:
                self._wait()
            return
        self._pi += 1

    def _wait(self):
        self._advance(60)

    def _react(self):
        """Survival override: if a drainer has the current body, counterattack
        (absorb a landable seer), else escape (hyperspace as last resort).  A
        precomputed plan cannot predict enemy phase exactly, so execution guards
        it.  Returns True if it deviated."""
        st = self.st
        if self._player_window() > SAFE_FRAMES:
            return False  # not under threat: follow the plan
        seers = self._dangerous_seers()
        if not seers:
            return False
        foes = enemies.enemy_slots(st)
        cands = []
        for e in seers:
            if e == actions.SENTINEL_SLOT and len(foes) > 1:
                continue  # the $1B8E lock forbids the Sentinel before the rest
            etile = st.tile_of(e)
            if self._top(etile) != e:
                continue
            view = self._view_for(etile)
            if view is not None:
                cands.append((self._aim_frames(view), etile, view))
        cands.sort()
        for _, etile, view in cands:
            if self._fire("absorb", etile, view):
                self._hs_streak = 0
                return True  # counterattack: the seer is gone
        if self._escape_transfer():
            self._hs_streak = 0
            return True
        drain_now = self._player_window() <= 0
        if (
            drain_now
            and self._hs_streak == 0
            and st.energy > mm.ENERGY_IN_OBJECTS[mm.T_ROBOT]
        ):
            self._hs_streak += 1
            self._hyperspace()  # last resort: flee an unabsorbable drainer
            return True
        return False

    def _escape_transfer(self):
        """Transfer to the landable robot with the widest rotation window, if it
        is strictly safer than staying put."""
        st = self.st
        here = self._player_window()
        best = None
        for s in range(mm.NUM_SLOTS):
            if st.is_empty(s) or s == st.player or st.obj_type[s] != mm.T_ROBOT:
                continue
            tile = st.tile_of(s)
            if self._top(tile) != s:
                continue
            view = self._view_for(tile)
            if view is None:
                continue
            window = self._gaze_window(tile)
            if window > here and (best is None or window > best[0]):
                best = (window, tile, view)
        if best is None:
            return False
        return self._fire("transfer", best[1], best[2])

    def _dangerous_seers(self):
        """Living enemies whose cone is on the player NOW and can damage it (full
        sight, or partial with a tree within 10 tiles)."""
        st = self.st
        tree = self._tree_near(st.player_xy())
        half = FOV_HALF + FOV_MARGIN
        out = []
        for e, ah, full in self._exposures(st, st.player):
            if (full or tree) and self._in_cone(ah, st.obj_h_angle[e], half):
                out.append(e)
        return out

    def _views_for_sig(self):
        """Memoized $F5-plane view dict for the current stance, keyed like
        ``_land_memo``.  ``los.landable_view(st, tile, v_band=False)`` IS a lookup
        into exactly this dict, so one sweep per distinct sig serves every tile at
        that stance; below-eye tiles miss it and take the targeted band fallback."""
        sig = bytes(self.st.mem[0x0400:0x0800]) + bytes([self.st.player])
        primary = self._view_memo.get(sig)
        if primary is None:
            primary = los._landable_sweep(
                self.st, self.st.player, None, 6000, want_centres=False, v_primary=True
            )[0]
            self._view_memo[sig] = primary
        return primary

    def _view_for(self, tile):
        """Cheapest keyboard view landing ``tile`` (execution only): memoized
        F5-plane lookup, targeted single-tile band march as fallback.  The band
        fallback marches only the narrow cone of rays that can land on ``tile``
        (:func:`los.landable_view_targeted`) -- bit-identical to the full-board
        ``landable_views`` sweep, so ``_c_reclaim``'s per-iteration terrain sigs
        each cost one cone instead of a whole-board re-sweep."""
        tile = tuple(tile)
        view = self._views_for_sig().get(tile)
        if view is None and self._sees_tile(tile):
            view = los.landable_view_targeted(self.st, tile)
        return view

    # ----------------------------------------------------------------- search
    def _search(self):
        """Best-first search for a frame-cheap winning line; the list of
        ``(verb, tile)`` steps, or ``None`` if none found in budget."""
        real = self.st  # the cheap executors rebind self.st to clones; restore after
        real_bearing = self.last_bearing
        real_cursor = list(self.cursor)
        try:
            start = _Node(
                self.st.clone(), 0.0, (), None, self.last_bearing, self.cursor
            )
            start.key = self._key(start.state)
            heap = [(self._h(start.state), 0, start)]
            best_g = {start.key: 0.0}
            counter = 0
            if self._deadline is None:
                self._deadline = time.time() + self.time_budget
            self.expansions = 0
            while heap:
                if self.expansions >= self.node_budget:
                    break
                if time.time() >= self._deadline:
                    break
                _, _, node = heapq.heappop(heap)
                if node.g > best_g.get(node.key, math.inf) + 1e-6:
                    continue
                if actions.won(node.state):
                    return list(node.path)
                self.expansions += 1
                if self.verbose and self.expansions % 40 == 0:
                    ns = node.state
                    print(
                        f"    exp={self.expansions} depth={len(node.path)} "
                        f"eye={round(ns.eye_z(),2)} foes={len(enemies.enemy_slots(ns))} "
                        f"E={ns.energy} tile={ns.player_xy()} g={round(node.g)}"
                    )
                for child in self._expand(node):
                    prev = best_g.get(child.key)
                    if prev is not None and child.g >= prev - 1e-6:
                        continue
                    best_g[child.key] = child.g
                    counter += 1
                    f = child.g + self.weight * self._h(child.state)
                    heapq.heappush(heap, (f, counter, child))
            return None
        finally:
            self.st = real
            self.last_bearing = real_bearing
            self.cursor = real_cursor

    def _key(self, st):
        """Dedup key: player tile+eye, energy, remaining enemies (bucketed
        facing), and built stacks."""
        p = st.player
        objs = []
        foes = []
        for s in range(mm.NUM_SLOTS):
            if st.is_empty(s):
                continue
            t = st.obj_type[s]
            if t in (mm.T_SENTINEL, mm.T_SENTRY):
                foes.append((s, st.obj_h_angle[s] >> 3))
            elif t in (mm.T_BOULDER, mm.T_ROBOT) and s != p:
                objs.append((st.obj_x[s], st.obj_y[s], round(self._base_z(s) * 2)))
        return (
            st.obj_x[p],
            st.obj_y[p],
            round(st.eye_z() * 2),
            st.energy,
            tuple(sorted(foes)),
            tuple(sorted(objs)),
        )

    def _h(self, st):
        if actions.won(st):
            return 0.0
        remaining = len(enemies.enemy_slots(st))
        hops = max(0.0, (_TARGET_EYE - st.eye_z()) / _EYE_PER_HOP)
        return remaining * _ABSORB_EST + hops * _HOP_EST + _ENDGAME_EST

    # -------------------------------------------------------------- expansion
    def _expand(self, node):
        """Macro-actions: endgame (Sentinel gone), the terminal absorb of any
        already-landable enemy, a reclaim, and a DIRECTED pursuit of each
        not-yet-landable enemy (a multi-hop climb + absorb as one child).  The
        branching factor is "which enemy to pursue next", not "which tile"."""
        st = node.state
        if st.is_empty(actions.SENTINEL_SLOT):
            child = self._c_endgame(node)
            return [child] if child is not None else []
        children = []
        for tile, e in self._absorb_enemy_targets(st):
            child = self._c_absorb(node, tile, e)
            if child is not None:
                children.append(child)
        child = self._c_reclaim(node)
        if child is not None:
            children.append(child)
        for e in self._pursue_targets(st):
            child = self._c_pursue(node, e)
            if child is not None:
                children.append(child)
        return children

    def _node(self, node, st, g, steps):
        return _Node(
            st,
            g,
            node.path + tuple(steps),
            self._key(st),
            self.last_bearing,
            self.cursor,
        )

    def _charge(self, st, verb, tile):
        """Advance the enemies by this action's REAL aim+settle cost (the same
        faithful ``_aim_frames``/``_settle`` the executor prices with, over the
        same ``_view_for`` selector), then mirror the post-aim stance update so
        an intra-hop follow-up on the same tile reuses the bearing.  Returns the
        frames spent."""
        self.st = st
        view = self._view_for(tile)
        if view is None:
            cost = self._settle(verb)  # infeasible guard: the gates reject these
            enemies.advance_frames(st, int(cost))
            return cost
        cost = self._aim_frames(view) + self._settle(verb, view)
        enemies.advance_frames(st, int(cost))
        me = st.player
        st.obj_h_angle[me] = view["h_angle"]
        st.obj_v_angle[me] = view["v_angle"]
        self.cursor = list(view["cursor"])
        self.last_bearing = (view["h_angle"], view["v_angle"])
        if verb == "transfer":
            self.last_bearing = None  # new body: committed bearing is stale
        return cost

    def _landset(self, st):
        """Keyboard-landable tiles from the current stance, memoized by
        terrain-map + observer.  A coarse-cursor batch gives the exact tile set
        ~16x faster than the full sweep (the fine cursor only refines the view,
        recovered at execution)."""
        sig = bytes(st.mem[0x0400:0x0800]) + bytes([st.player])
        tiles = self._land_memo.get(sig)
        if tiles is None:
            tiles = self._coarse_landable(st)
            self._land_memo[sig] = tiles
        return tiles

    @staticmethod
    def _coarse_landable(st, cstep=2):
        if not los._HAVE_JIT:
            return set(los.landable_views(st))
        hgrid = list(range(0, 256, los.AZIMUTH_STEP))
        cxs = list(range(48, 112, cstep))
        cys = list(range(63, 127, cstep))
        status, tx, ty, _, _ = los._landable_batch(
            st, st.player, None, 6000, hgrid, los._V_PRIORITY, cxs, cys
        )
        clear = np.flatnonzero(status == los.los_jit.LOS_CLEAR)
        return set(zip(tx[clear].tolist(), ty[clear].tolist()))

    def _tile_base(self, st, tile):
        """Foot height a stack on ``tile`` builds from, or ``None`` if the top is
        not stackable (bare ground / boulder / platform only)."""
        top = self._top(tile)
        if top is None:
            return terrain.tile_byte(st, *tile) >> 4
        if st.obj_type[top] in (mm.T_BOULDER, mm.T_PLATFORM):
            return self._base_z(top) + (
                1.0 if st.obj_type[top] == mm.T_PLATFORM else BOULDER_H
            )
        return None

    def _pursue_targets(self, st):
        """Living enemies not landable from the current stance, nearest first
        (small branching); the Sentinel only once it stands alone ($1B8E lock)."""
        foes = enemies.enemy_slots(st)
        land = self._landset(st)
        px, py = st.player_xy()
        out = []
        for e in foes:
            if e == actions.SENTINEL_SLOT and len(foes) > 1:
                continue
            tile = st.tile_of(e)
            if terrain.top_object(st, *tile) == e and tile in land:
                continue  # already landable: _c_absorb handles the terminal strike
            out.append(((st.obj_x[e] - px) ** 2 + (st.obj_y[e] - py) ** 2, e))
        out.sort()
        return [e for _, e in out[:_TOP_TARGETS]]

    def _c_pursue(self, node, e):
        """Directed climb to enemy ``e``: chain minimal pedestal hops (reclaiming
        spent ones for energy) until ``e`` is landable, then absorb it -- ONE
        child node.  Each sub-action is charged the real aim+settle via
        ``_charge``; ``None`` if the climb cannot safely reach ``e``."""
        st = node.state.clone()
        self.st = st
        self.last_bearing = node.last_bearing
        self.cursor = list(node.cursor)
        g = node.g
        steps = []
        target = st.tile_of(e)
        for _ in range(_MAX_PURSUE):
            self.st = st
            if target in self._landset(st) and terrain.top_object(st, *target) == e:
                if not actions.can_absorb(st, e):
                    return None
                g += self._charge(st, "absorb", target)
                if not actions.absorb(st, e):
                    return None
                steps.append(("absorb", target))
                return self._node(node, st, g, steps)
            if st.energy < HOP_COST + self._reserve():
                got = self._reclaim_one(st)
                if got is not None:
                    g += got[0]
                    steps.append(got[1])
                    continue
            bearing, cursor = self.last_bearing, list(self.cursor)
            advanced = False
            for tile, k in self._pick_hop(target):
                trial = st.clone()
                self.st = trial
                self.last_bearing = bearing
                self.cursor = list(cursor)
                res = self._hop_exec(tile, k)
                if res is not None:
                    st, advanced = trial, True
                    g += res[0]
                    steps.extend(res[1])
                    break
            if not advanced:
                return None
            # inchworm: recycle now-below shells/pedestals into energy after the
            # transfer up (base_z <= new eye, not the current support/player tile),
            # keeping the climb near the reserve floor -- the human's ls42 line.
            self.st = st
            for _ in range(_HOP_BOULDERS + 1):
                got = self._reclaim_one(st, pedestal_only=True)
                if got is None:
                    break
                g += got[0]
                steps.append(got[1])
        return None

    def _pick_hop(self, target):
        """Ranked pedestal builds directed at ``target``: prefer tiles that gain
        LOS on it, then raise the eye, then the widest window -- gated on the
        rotation window (>= a full hop) and never under a live cone (the search's
        placement safety).  The pursuit tries them in order until one hop lands."""
        st = self.st
        my_eye = st.eye_z()
        reserve = self._reserve()
        cands = []
        for tile in self._landset(st):
            base = self._tile_base(st, tile)
            if base is None:
                continue
            k = max(1, math.ceil((my_eye + EYE_EPS - _ROBOT_EYE - base) / BOULDER_H))
            if k > _HOP_BOULDERS:
                continue
            if st.energy - reserve < 2 * k + mm.ENERGY_IN_OBJECTS[mm.T_ROBOT]:
                continue
            robot_eye = base + BOULDER_H * k + _ROBOT_EYE
            if robot_eye <= my_eye + EYE_EPS:
                continue
            exposed = self._exposing_enemies(tile)
            if self._seen_now(exposed):
                continue
            window = self._gaze_window(tile, exposed=exposed)
            if window < HOP_FRAMES:
                continue
            sees = self._tile_sees_target(tile, target)
            cands.append(((sees, robot_eye, window), tile, k))
        cands.sort(key=lambda c: c[0], reverse=True)
        return [(t, k) for _, t, k in cands[:_TOP_HOPS]]

    def _hop_exec(self, tile, k):
        """Build ``k`` boulders + a robot on ``tile`` and transfer up (``self.st``
        is the working clone); ``(g_delta, steps)`` or ``None`` if unsafe or
        infeasible.  The lone-hop builder shared by every pursuit macro."""
        st = self.st
        g = 0.0
        steps = []
        for _ in range(k):
            if not self._can_build(st, tile, mm.T_BOULDER):
                return None
            g += self._charge(st, "boulder", tile)
            actions.create(st, mm.T_BOULDER, tile)
            steps.append(("boulder", tile))
        if not self._can_build(st, tile, mm.T_ROBOT):
            return None
        g += self._charge(st, "robot", tile)
        if actions.create(st, mm.T_ROBOT, tile) is None:
            return None
        steps.append(("robot", tile))
        top = terrain.top_object(st, *tile)
        if not threat.player_sees_tile(st, tile, st.player):
            return None
        g += self._charge(st, "transfer", tile)
        if not actions.transfer(st, top) or actions.player_dead(st):
            return None
        steps.append(("transfer", tile))
        # a seen destination is a trap only if no seer is absorbable from it
        if self._gaze_window(
            tile
        ) + _DRAIN_DELAY < _MARGIN and not self._absorbable_here(st):
            return None
        return g, steps

    def _absorbable_here(self, st):
        """Whether a living enemy is landable from the current stance (the
        counterattack is available)."""
        foes = enemies.enemy_slots(st)
        land = self._landset(st)
        for e in foes:
            if e == actions.SENTINEL_SLOT and len(foes) > 1:
                continue
            tile = st.tile_of(e)
            if terrain.top_object(st, *tile) == e and tile in land:
                return True
        return False

    def _can_build(self, st, tile, otype):
        return threat.player_sees_tile(st, tile, st.player) and actions.can_create(
            st, otype, tile
        )

    def _absorb_enemy_targets(self, st):
        foes = enemies.enemy_slots(st)
        landset = self._landset(st)
        out = []
        for e in foes:
            if e == actions.SENTINEL_SLOT and len(foes) > 1:
                continue  # Sentinel dead last ($1B8E lock)
            tile = st.tile_of(e)
            if terrain.top_object(st, *tile) == e and tile in landset:
                out.append((tile, e))
        return out

    def _c_absorb(self, node, tile, e):
        st = node.state.clone()
        self.st = st
        self.last_bearing = node.last_bearing
        self.cursor = list(node.cursor)
        if not actions.can_absorb(st, e):
            return None
        g = node.g + self._charge(st, "absorb", tile)
        if not actions.absorb(st, e):
            return None
        return self._node(node, st, g, [("absorb", tile)])

    def _reclaim_one(self, st, pedestal_only=False):
        """Absorb ONE landable spent pedestal/shell (base <= eye), or a tree when
        short; the player stays put so its own window bounds the aim.  Returns
        ``(g_delta, step)`` or ``None``.  ``pedestal_only`` skips the tree sweep so
        the inchworm recycle grabs only the player's own spent boulders/shells."""
        my_eye = st.eye_z()
        want_trees = (not pedestal_only) and st.energy < HOP_COST + 6
        for s in range(mm.NUM_SLOTS):
            if st.is_empty(s) or s == st.player:
                continue
            otype = st.obj_type[s]
            tile = st.tile_of(s)
            if terrain.top_object(st, *tile) != s or tile == st.player_xy():
                continue
            if otype in (mm.T_ROBOT, mm.T_BOULDER):
                if self._base_z(s) > my_eye + EYE_EPS:
                    continue
            elif not (otype == mm.T_TREE and want_trees):
                continue
            if not threat.player_sees_tile(st, tile, st.player):
                continue
            g = self._charge(st, "absorb", tile)
            if not actions.absorb(st, s):
                return None
            return g, ("absorb", tile)
        return None

    def _c_reclaim(self, node):
        """Absorb landable spent pedestals, shells and, when short, trees, up to
        eight in one macro; the player stays put throughout."""
        st = node.state.clone()
        self.st = st
        self.last_bearing = node.last_bearing
        self.cursor = list(node.cursor)
        g = node.g
        steps = []
        for _ in range(8):
            got = self._reclaim_one(st)
            if got is None:
                break
            g += got[0]
            steps.append(got[1])
        if not steps:
            return None
        return self._node(node, st, g, steps)

    def _c_endgame(self, node):
        """Sentinel gone (no enemy remains): robot on the platform, transfer,
        hyperspace -- the win."""
        st = node.state.clone()
        self.st = st
        self.last_bearing = node.last_bearing
        self.cursor = list(node.cursor)
        ptile = st.platform_xy
        g = node.g
        if not actions.on_platform(st):
            if ptile not in self._landset(st):
                return None
            g += self._charge(st, "robot", ptile)
            slot = actions.create(st, mm.T_ROBOT, ptile)
            if slot is None:
                return None
            g += self._charge(st, "transfer", ptile)
            if not actions.transfer(st, slot):
                return None
        g += self._charge(st, "hyperspace", ptile)
        actions.hyperspace(st)
        if not actions.won(st):
            return None
        steps = [("robot", ptile), ("transfer", ptile), ("hyperspace", ptile)]
        return self._node(node, st, g, steps)


def main():
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("landscape", nargs="?", type=int, default=66)
    parser.add_argument("--max-actions", type=int, default=400)
    parser.add_argument("--time-budget", type=float, default=30.0)
    parser.add_argument("--quiet", action="store_true")
    parser.add_argument(
        "--audit",
        action="store_true",
        help="strict post-settle invariant accounting: flag any create/transfer "
        "that ended in a live enemy cone",
    )
    args = parser.parse_args()
    game = Game.new(args.landscape)
    player = AStarPlayer(
        game,
        verbose=not args.quiet,
        audit=args.audit,
        time_budget=args.time_budget,
    )
    won = player.run(max_actions=args.max_actions)
    print(
        f"landscape {args.landscape}: {'WON' if won else 'lost'} in "
        f"{len(player.trace)} actions / {player.frames} frames, "
        f"energy {game.energy}, dead={actions.player_dead(game.state)}"
    )
    if args.audit:
        print(f"invariant breaches: {len(player.breaches)}")
        for f, verb, tile, seen in player.breaches:
            print(f"  f={f} {verb} {tile} seen_by={seen}")
    return 0 if won else 1


if __name__ == "__main__":
    raise SystemExit(main())
