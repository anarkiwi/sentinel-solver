"""An A* / best-first planning player over the sentinel model.

Enemies only rotate, so each tile has a closed-form window (frames until a cone
rotates onto it); the search carries the cheap enemy phase, gates moves on
window >= aim+settle, and defers the keyboard-aim sweep to execution.
"""

import argparse
import heapq
import math
import os
import time
import typing

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
    ROBOT_EYE,
    SAFE_FRAMES,
    SIGHTS_CENTRE,
    TAP_FRAMES,
)

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
_TAIL_FLOOR = _OP_FLOOR["create"] + _OP_FLOOR["transfer"]  # robot+xfer: drainable span
_EYE_PER_HOP = 0.9
_TARGET_EYE = 9.0
_TOP_TARGETS = 4  # enemies a node may rank pedestal candidates toward
_TOP_HOPS = 8  # ranked pedestal candidates _pick_hop returns per target
_MAX_HOP_KIDS = int(
    os.environ.get("SENTINEL_MAX_HOP_KIDS", "12")
)  # distinct (tile, k) hop children one node may emit, over all targets
_STRAND_PRUNE = (
    os.environ.get("SENTINEL_STRAND_PRUNE", "0") != "0"
)  # OFF: measured on ls335/2000, the probe costs a 1-ply lookahead per child (2.56 s/expansion vs 1.40) and reaches the SAME best node (eye 6.38, 0 of 7 killed) in 184 expansions instead of 343 -- with per-hop children A* backtracks out of a stranded landing for the price of one expansion, which is cheaper than proving it stranded
_MAX_RECLAIM = 8  # reclaims one macro (or one strand probe) may chain
_TOP_CLEARS = 4  # tree-blocked pedestal sites a node may branch a clearing child on
_STEP_SIGMA = float(
    os.environ.get("SENTINEL_STEP_SIGMA", "24.1")
)  # measured whole-step rms, live ls42 (live_ls42_hops.json); see _margin
_MARGIN_K = float(os.environ.get("SENTINEL_MARGIN_K", "1.0"))  # sigmas of headroom
_NO_VIEW = object()  # cone-memo miss sentinel (a cached view may legitimately be None)
_COARSE_CX = list(range(48, 112, 2))  # landset sights-cursor grid: the 1px window 2:1
_COARSE_CY = list(range(63, 127, 2))  # subsampled; _landable queries the SAME lattice

GATE_BODY = "body"  # gated on the PLAYER'S body window (_hot): absorbs
GATE_TILE = "tile"  # gated on the TARGET TILE's window (_drain_gate): builds/transfers


class PlanStep(typing.NamedTuple):
    """One executable plan step, carrying what the search knew when it made it, so
    execution re-validates the SAME premise instead of re-deriving one.

    ``budget`` is what ``_charge`` charged; ``gate`` is which gaze window the
    generator gated the step on, and ``window`` that window's predicted value;
    ``pbody`` is the player body window at plan time (== ``window`` under
    ``GATE_BODY``), which the live audit compares against reality."""

    verb: str
    tile: tuple
    budget: float
    gate: str
    window: float
    pbody: float


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
    """Search a winning line once, then execute it.

    ``audit_pred`` records ``PlanStep.pbody``: a ``_player_window`` per charged step,
    so per speculative branch.  Only ``driver.plan_audit`` reads it, never execution."""

    audit_pred = False

    def __init__(
        self,
        game,
        verbose=False,
        audit=False,
        node_budget=200000,
        time_budget=None,  # wall-clock cut: OFF by default, see _search
        weight=1.4,
    ):
        super().__init__(game, verbose=verbose, audit=audit)
        self.node_budget = node_budget
        self.time_budget = time_budget
        self.weight = weight
        self.plan = None
        self._pi = 0
        self.expansions = 0
        self._deadline = None  # per-search wall-clock deadline (set at each _search)
        self._land_memo = {}  # search: coarse landable tile-sets
        self._tile_memo = {}  # per-(sig, tile) single-tile landability (targeted cone)
        self._view_memo = {}  # per-sig $F5-plane view dicts (band via targeted march)
        self._cone_memo = {}  # per-(sig, tile) targeted band march results
        self._hop_price_memo = {}  # per-(stance, tile, k) exact hop cost
        self._hs_streak = 0  # consecutive last-resort hyperspaces (spiral guard)
        self._depth = 0  # steps charged ahead of the live board (margin scale)
        self._margin_k = _MARGIN_K  # 0 in a relaxed (last-chance) re-search
        self._hop_audit = None  # list => shadow-record the body-window hop gate
        self._on_plan = False  # last _react deviation WAS the plan's next step
        self._last_pbody = math.inf  # body window at the last _charge's pre-step state
        self._hop_obs = None  # cheapest whole hop charged this search (scales _h)

    def _plan_step(self, verb, tile, budget, gate, window=None):
        """Record the step just charged: ``window`` defaults to the body window
        ``_charge`` measured, which IS the gated value under ``GATE_BODY``."""
        pbody = self._last_pbody
        return PlanStep(
            verb, tuple(tile), budget, gate, pbody if window is None else window, pbody
        )

    def _margin(self, depth=None):
        """Frames of enemy-phase uncertainty a gate must hold back at plan depth
        ``depth``.  Per-step charged-vs-measured frame error (frame_audit in
        out/play_player_0042.json, n=15) is zero-mean (+1f) with rms sigma=68f and
        does not cancel, so it accumulates as a random walk: k*sigma*sqrt(depth+1).
        """
        d = self._depth if depth is None else depth
        return self._margin_k * _STEP_SIGMA * math.sqrt(d + 1.0)

    def _hot(self, budget, window=None):
        """Whether standing exposed for ``budget`` frames breaches the pessimistic
        end of the step-cost interval (``budget + _margin()``)."""
        if window is None:
            window = self._player_window()
        return window < budget + self._margin()

    # ---------------------------------------------------------------- execute
    def _tick(self):
        if self.plan is None:
            self.plan = self._search()
            self._pi = 0
            if self.verbose:
                line = self.plan and [(s.verb, tuple(s.tile)) for s in self.plan]
                print(f"  plan ({self.expansions} nodes): {line}")
        if not self._frozen() and self._react():
            if not self._on_plan:  # deviated for survival: re-plan from the new state
                self.plan = None
            self._on_plan = False
            return
        if not self.plan or self._pi >= len(self.plan):
            self._wait()
            return
        step = self.plan[self._pi]
        verb, tile = step.verb, step.tile
        if verb == "hyperspace":
            self._hyperspace()
            self._pi += 1
            return
        view = self._view_for(tile)
        if view is not None and self._plan_step_stale(step, view):
            self._restale((verb, tuple(tile)))
            return
        self._stale = None
        if view is None or not self._fire(verb, tile, view):
            self.plan = self._search()  # live/plan divergence: re-plan
            self._pi = 0
            if not self.plan:
                self._wait()
            return
        self._pi += 1

    def _restale(self, key=None):
        """Ladder taken when the next planned step's premise is stale on the live
        board: re-plan under the normal margin, else take a survivable defensive
        move, else a last-chance zero-margin line, else wait.  Conceding an escape
        hyperspace is left to ``_react``, after every non-conceding option.

        ``key`` is the stale ``(verb, tile)``.  A REPEAT of the same verdict cannot
        be re-planned away -- ``_search`` is a pure function of the board and does
        not advance it, so it re-derives the same head and the gate re-fires on an
        identical enemy phase -- so a repeat WAITS instead: the world moves, and
        ``_plan_step_stale`` may then clear a step only the margin still blocks."""
        repeat = key is not None and self._stale is not None and self._stale[0] == key
        self._stale = (key, self._stale[1] + 1 if repeat else 1) if key else None
        self._pi = 0
        if repeat:
            self._wait()
            return
        self.plan = self._search()
        if self.plan:
            return
        if self._defend():
            self._hs_streak = 0
            return
        self.plan = self._search(margin_k=0.0)
        if not self.plan:
            self._wait()  # let the enemy cone rotate (react acts once it is on us)

    def _plan_step_stale(self, step, view):
        """Whether the next planned ``PlanStep`` needs a fresh search before firing. The
        offline plan is deterministic and already drain-gated, so it never goes
        stale; the live player overrides this to re-check the real enemy phase."""
        return False

    def _wait(self):
        self._advance(60)

    def _defend(self):
        """Non-conceding survival ladder on the observed board: counterattack a
        landable dangerous seer, else flee to the widest-window body.  Returns True
        if it acted."""
        st = self.st
        foes = enemies.enemy_slots(st)
        cands = []
        for e in self._dangerous_seers():
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
                return True  # counterattack: the seer is gone
        return self._escape_transfer()

    def _react(self):
        """Survival override: if a drainer has the current body, defend, else
        hyperspace as the last resort.  A precomputed plan cannot predict enemy
        phase exactly, so execution guards it.  Returns True if it deviated."""
        st = self.st
        if self._player_window() > SAFE_FRAMES:
            return False  # not under threat: follow the plan
        if self._defend():
            self._hs_streak = 0
            return True
        if self._plan_escape_transfer():
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

    def _plan_escape_transfer(self):
        """Take the plan's next step when it is the transfer OFF this tile: the body
        the pursuit has just finished building IS the escape, drain-gated at plan time,
        and it leaves the eye higher.  ``_escape_transfer`` ranks bodies by window and
        rejects it whenever the pedestal's window is no wider than the one the player
        is standing in, which on ls42 live conceded a hyperspace one keystroke short of
        the climb it had already paid for."""
        if not self.plan or self._pi >= len(self.plan):
            return False
        step = self.plan[self._pi]
        tile = step.tile
        if step.verb != "transfer":
            return False
        view = self._view_for(tile)
        if view is None or not self._fire("transfer", tile, view):
            return False
        self._pi += 1
        self._on_plan = True
        return True

    def _escape_transfer(self):
        """Transfer to the safer landable robot the player can actually REACH: cheapest
        aim first, widest window breaking ties, and only bodies whose aim+settle fits
        inside the window the current body has left. Ranking on window alone picks a
        wide-window body half a pan away and is drained mid-aim -- while escaping, the
        aim IS the exposure, which is why the counterattack above sorts the same way."""
        here = self._player_window()
        cands = []
        for s, tile in self._robot_bodies():
            view = self._view_for(tile)
            if view is None:
                continue
            window = self._gaze_window(tile)
            if window <= here:
                continue
            cost = self._step_aim_frames("transfer", view) + self._settle(
                "transfer", view, s
            )
            if cost > here:
                continue  # drained mid-aim: a safer perch it cannot live to reach
            cands.append((cost, -window, tile, view))
        cands.sort()
        for _cost, _w, tile, view in cands:
            if self._fire("transfer", tile, view):
                return True
        return False

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
        sig = self._sig()
        primary = self._view_memo.get(sig)
        if primary is None:
            primary = los._landable_sweep(
                self.st, self.st.player, None, 6000, want_centres=False, v_primary=True
            )[0]
            self._view_memo[sig] = primary
        return primary

    def _sig(self, st=None):
        """Stance signature: object/terrain map + observer -- what every landability
        answer is a pure function of (enemy facings never enter it).  ``st`` defaults
        to the working stance, but the expansion prologue prices candidate sets on a
        node's own state before rebinding it."""
        st = self.st if st is None else st
        return bytes(st.mem[0x0400:0x0800]) + bytes([st.player])

    def _view_for(self, tile):
        """Cheapest keyboard view landing ``tile`` (execution only): memoized
        F5-plane lookup, targeted single-tile band march as fallback.  The band
        fallback marches only the narrow cone of rays that can land on ``tile``
        (:func:`los.landable_view_targeted`) -- bit-identical to the full-board
        ``landable_views`` sweep, so ``_c_reclaim``'s per-iteration terrain sigs
        each cost one cone instead of a whole-board re-sweep.  The cone is memoized
        per (sig, tile) too: the same below-eye tile is re-priced by every trial hop,
        probe and re-search at a stance, and the march was the search's top cost."""
        return self._view_with_band(tile, self._views_for_sig(), self._band_march)

    def _band_march(self, tile):
        """Targeted single-tile band march for ``tile``, memoized per (sig, tile)."""
        key = (self._sig(), tile)
        view = self._cone_memo.get(key, _NO_VIEW)
        if view is _NO_VIEW:
            view = los.landable_view_targeted(self.st, tile)
            self._cone_memo[key] = view
        return view

    # ----------------------------------------------------------------- search
    def _search(self, margin_k=None):
        """Best-first search for a frame-cheap winning line; the list of
        ``PlanStep`` records, or ``None`` if none found in budget.  ``margin_k``
        overrides the drain-gate headroom (0 == the old zero-margin search).

        The search is a PURE FUNCTION OF THE BOARD: ``node_budget`` bounds it, and
        ``time_budget`` (a wall-clock cut) is off by default.  With one set, a loaded
        host truncates the search sooner and plays a DIFFERENT line -- which is not a
        cheaper plan, just a worse one, and it makes a live run unreproducible."""
        real = self.st  # the cheap executors rebind self.st to clones; restore after
        real_bearing = self.last_bearing
        real_cursor = list(self.cursor)
        self._margin_k = _MARGIN_K if margin_k is None else margin_k
        try:
            start = _Node(
                self.st.clone(), 0.0, (), None, self.last_bearing, self.cursor
            )
            start.key = self._key(start.state)
            heap = [(self._h(start.state), 0, start)]
            best_g = {start.key: 0.0}
            counter = 0
            # per-search budget: each replan gets its own window (run-wide starves replans)
            self._deadline = (
                time.time() + self.time_budget if self.time_budget else None
            )
            self.expansions = 0
            self._hop_obs = None
            while heap:
                if self.expansions >= self.node_budget:
                    break
                if self._deadline is not None and time.time() >= self._deadline:
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
            self._margin_k = _MARGIN_K
            self._depth = 0

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
        """Frames to go: one absorb per living enemy, one hop per ``_EYE_PER_HOP`` of
        eye still owed, plus the endgame.  ``hops`` counts CLIMB STEPS, which is what a
        child now is, so the term reads directly as plies * cost-per-ply.

        The per-hop cost is the CHEAPEST WHOLE HOP this search has actually charged,
        floored at the op-floor bound ``_HOP_EST``.  A real hop costs 745 f (ls42) to
        1294 f (ls335) against a 468 f floor, so the floor alone under-rewards climbing
        by 2-3x per ply: a hop raises g by its real cost while dropping h by the floor,
        so f RISES on every climb and the frontier prefers to dither among reclaims.
        The chain generator hid this by folding a whole climb into one child.  The
        observed minimum is still a lower bound over the hops this board offers, and it
        self-calibrates to the board instead of being a constant that is simultaneously
        too strict and too lax."""
        if actions.won(st):
            return 0.0
        remaining = len(enemies.enemy_slots(st))
        hops = max(0.0, (_TARGET_EYE - st.eye_z()) / _EYE_PER_HOP)
        est = _HOP_EST if self._hop_obs is None else max(_HOP_EST, self._hop_obs)
        return remaining * _ABSORB_EST + hops * est + _ENDGAME_EST

    # -------------------------------------------------------------- expansion
    def _expand(self, node):
        """Macro-actions: endgame (Sentinel gone), the terminal absorb of any
        already-landable enemy, a reclaim, a tree clearing, and ONE CLIMB STEP per
        ranked pedestal candidate.  The branching factor is "which tile to climb
        on", and depth counts hops rather than enemies, so two stances that differ
        only in where the climb went are both on the frontier and A* can back out
        of a dead end instead of dead-ending with the frontier empty."""
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
        children.extend(self._c_clear(node))
        for tile, k, window in self._hop_candidates(node):
            child = self._c_hop(node, tile, k, window)
            if child is not None:
                children.append(child)
        return children

    def _hop_candidates(self, node):
        """``(tile, k, window)`` climb steps this node may take, ranked by
        ``_pick_hop`` toward each pursuit target in turn and deduped on ``(tile, k)``
        -- the same pedestal serves whichever enemy it gains LOS on, so the nearest
        target's ranking leads and the others only append tiles it did not offer.
        Capped at ``_MAX_HOP_KIDS``."""
        st = self._begin(node)  # ranking is read-only: one clone serves every target
        out = {}
        for e in self._pursue_targets(st):
            for tile, k, window in self._pick_hop(st.tile_of(e)):
                out.setdefault((tuple(tile), k), window)
                if len(out) == _MAX_HOP_KIDS:
                    return [(t, k, w) for (t, k), w in out.items()]
        return [(t, k, w) for (t, k), w in out.items()]

    def _c_hop(self, node, tile, k, window):
        """ONE climb step as a child: build ``k`` boulders + a robot on ``tile``,
        transfer up, then inchworm-recycle the stack the transfer left below the new
        eye.  ``None`` if the build/transfer is infeasible or unsurvivable, or -- under
        ``_STRAND_PRUNE`` -- if the landing has no move at all.

        The recycle stays ATTACHED rather than becoming its own child because it is
        only defined against the stack this very hop just left: as a separate ply it
        would duplicate ``_c_reclaim``, which already offers "reclaim from here" as an
        alternative to any node.  Attached, one child == one hop and the ``hops`` term
        of ``_h`` counts plies."""
        self._begin(node)
        st = self.st
        res = self._hop_exec(tile, k, window)
        if res is None:
            return None
        g, steps = res
        self._hop_obs = g if self._hop_obs is None else min(self._hop_obs, g)
        for _ in range(k + 1):
            got = self._reclaim_one(st, pedestal_only=True)
            if got is None:
                break
            g += got[0]
            steps.append(got[1])
        if _STRAND_PRUNE and self._stranded(st):
            return None
        return self._node(node, st, node.g + g, steps)

    def _stranded(self, st):
        """Whether a landing on ``st`` can do NOTHING next: no enemy landable, no
        blocking tree worth clearing, no affordable hop toward any pursuit target,
        and -- energy short -- no reclaim chain that makes one affordable.

        With per-hop children a stranded landing is just a node whose expansion
        returns no children, and A* backtracks by itself; this is purely a PRUNER,
        keeping the frontier from filling with stances that cost an expansion to
        discover are dead.  The reclaim arm re-ranks against the landing's own frozen
        tile set rather than re-sweeping per absorbed object (that sweep dominates the
        expansion profile); absorbing a spent pedestal below the eye only uncovers
        tiles, so the frozen set is the conservative side."""
        self.st = st
        if self._absorbable_here(st) or self._blocking_trees(st):
            return False
        targets = [st.tile_of(e) for e in self._pursue_targets(st)]
        if not targets:
            return False
        if any(self._pick_hop(t) for t in targets):
            return False
        if st.energy >= HOP_COST + self._reserve():
            return True  # affordable already: no reclaim can add a hop
        landset = self._landset(st)
        bearing, cursor, depth = self.last_bearing, list(self.cursor), self._depth
        probe = st.clone()
        self.st = probe
        ok = False
        for _ in range(_MAX_RECLAIM):
            if self._reclaim_one(probe) is None:
                break  # nothing left to recycle from here
            if any(self._pick_hop(t, landset=landset) for t in targets):
                ok = True
                break
            if probe.energy >= HOP_COST + self._reserve():
                break  # energy is the only budget filter: more of it adds nothing
        self.st = st
        self.last_bearing, self.cursor, self._depth = bearing, cursor, depth
        return not ok

    def _begin(self, node):
        """Rebind the working stance (a clone of ``node``'s state) and the plan
        depth the margin scales with; the shared child-builder prologue."""
        self.st = node.state.clone()
        self.last_bearing = node.last_bearing
        self.cursor = list(node.cursor)
        self._depth = len(node.path)
        return self.st

    def _node(self, node, st, g, steps):
        return _Node(
            st,
            g,
            node.path + tuple(steps),
            self._key(st),
            self.last_bearing,
            self.cursor,
        )

    def _price(self, st, verb, tile):
        """``(cost, aim, view)`` for ``verb`` on ``tile`` from stance ``st``, WITHOUT
        advancing the enemies: the ONE cost expression, shared by ``_charge`` (which
        advances the world by it) and by the hop gates (which must know what a hop
        costs before committing to it).  ``view is None`` is the infeasible guard --
        settle only, which the gates then reject."""
        self.st = st
        eye = self._settle_eye(verb, tile)
        view = self._view_for(tile)
        if view is None:
            return self._settle(verb, None, eye), 0.0, None
        aim = self._step_aim_frames(verb, view)
        return aim + self._settle(verb, view, eye), aim, view

    def _commit_view(self, view, verb):
        """Mirror the post-aim stance update so an intra-hop follow-up on the same
        tile reuses the bearing."""
        st = self.st
        me = st.player
        st.obj_h_angle[me] = view["h_angle"]
        st.obj_v_angle[me] = view["v_angle"]
        self.cursor = list(view["cursor"])
        # a transfer makes a new body: the committed bearing is stale
        self.last_bearing = (
            None if verb == "transfer" else (view["h_angle"], view["v_angle"])
        )

    def _charge(self, st, verb, tile):
        """Advance the enemies by this action's REAL ``_price``, then commit the
        stance the aim left.  Returns the frames spent, and stashes the PRE-step
        body window ``_plan_step`` records."""
        self.st = st
        self._depth += 1
        if self.audit_pred:
            self._last_pbody = self._player_window()
        cost, aim, view = self._price(st, verb, tile)
        if view is None:
            enemies.advance_frames(st, int(cost))
            return cost
        split = self._aim_unfreeze_split(view)
        if split is None:
            enemies.advance_frames(st, int(cost))
        else:  # $12E1: keying the u-turn started the enemy clock mid-aim
            pre = int(min(aim, split))
            enemies.advance_frames(st, pre)
            st.mem[mm.PLAYER_NOT_ACTED] = 0x00
            enemies.advance_frames(st, int(cost) - pre)
        self._commit_view(view, verb)
        return cost

    def _landset(self, st):
        """Keyboard-landable tiles from the current stance, memoized by
        terrain-map + observer.  A coarse-cursor batch gives the exact tile set
        ~16x faster than the full sweep (the fine cursor only refines the view,
        recovered at execution)."""
        sig = self._sig(st)
        tiles = self._land_memo.get(sig)
        if tiles is None:
            tiles = self._coarse_landable(st)
            self._land_memo[sig] = tiles
        return tiles

    def _landable(self, st, tile):
        """Whether ``tile`` alone is in :meth:`_landset` -- answered, when the whole set is
        not already memoized, by a targeted cone march over the SAME coarse lattice
        (:func:`los.landable_view_targeted`), which is bit-identical to the full sweep but
        marches only the rays whose heading points at the cell."""
        sig = self._sig(st)
        tiles = self._land_memo.get(sig)
        if tiles is not None:
            return tile in tiles
        if not los._HAVE_JIT:
            return tile in self._landset(st)
        key = (sig, tile)
        hit = self._tile_memo.get(key)
        if hit is None:
            hit = (
                los.landable_view_targeted(st, tile, cxs=_COARSE_CX, cys=_COARSE_CY)
                is not None
            )
            self._tile_memo[key] = hit
        return hit

    @staticmethod
    def _coarse_landable(st):
        if not los._HAVE_JIT:
            return set(los.landable_views(st))
        hgrid = list(range(0, 256, los.AZIMUTH_STEP))
        status, tx, ty, _, _ = los._landable_batch(
            st, st.player, None, 6000, hgrid, los._V_PRIORITY, _COARSE_CX, _COARSE_CY
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
        px, py = st.player_xy()
        out = []
        for e in foes:
            if e == actions.SENTINEL_SLOT and len(foes) > 1:
                continue
            tile = st.tile_of(e)
            if terrain.top_object(st, *tile) == e and self._landable(st, tile):
                continue  # already landable: _c_absorb handles the terminal strike
            out.append(((st.obj_x[e] - px) ** 2 + (st.obj_y[e] - py) ** 2, e))
        out.sort()
        return [e for _, e in out[:_TOP_TARGETS]]

    def _record_hop_gate(self, tile, k, exposed, tile_ok, priced):
        """Record both hop gates for a candidate: the ENFORCED destination gate (the
        tile's window against the drainable ``tail``) and the SHADOW source gate (the
        player's own body window against the whole-hop ``total`` it stands there for),
        which decides nothing.

        The source gate is the half the destination gate cannot see -- the player
        stands on its current tile for every frame of the build -- and it is the one
        that would have refused ls335's fatal (8,21) hop (total 1294 f against a 120 f
        body window).  Enforcing it is measured UNAFFORDABLE: ls42/internal 66 hops
        cost 891-1572 f from body windows of 120-892 f, so the search drops to 6
        expansions and no plan on a board it wins, the same collapse enforcing it live
        produced (docs/plan_fidelity.md).  Exposure onset is not death -- a drain
        costs energy over frames and the transfer moves the body off -- so the
        condition needs a cost, not a deadline.  Recorded until it has one."""
        total, tail = (math.inf, math.inf) if priced is None else priced
        margin = self._margin()
        body = self._player_window()
        self._hop_audit.append(
            {
                "depth": self._depth,
                "tile": tuple(tile),
                "k": k,
                "tile_window": self._gaze_window(tile, exposed=exposed),
                "body_window": body,
                "budget": total,
                "tail": tail,
                "margin": margin,
                "tile_ok": bool(tile_ok),
                "body_ok": bool(body >= total + margin),
                "body_ok_raw": bool(body >= total),
                "frozen": bool(self._frozen()),
            }
        )

    def _hop_price(self, st, tile, k):
        """``(total, tail)`` frames for ``k`` boulders + a robot + the transfer up on
        ``tile`` -- exactly what ``_hop_exec`` will charge -- priced on a clone, or
        ``None`` if the stack cannot be built or aimed.  ``tail`` is the robot-create
        plus transfer: the only interval a DRAINABLE body stands on the tile (a
        boulder body is not drainable, $16E6).

        The clone carries the creates, so each sub-action is priced against the stack
        the ones before it left, with the bearing reuse they commit; it does NOT carry
        the enemy phase forward (``_price`` never advances), which prices intra-hop
        render cost off the pre-hop facings -- second order against the aim terms."""
        key = (
            self._sig(st),
            tuple(tile),
            k,
            self.last_bearing,
            tuple(self.cursor),
            int(st.obj_h_angle[st.player]),
            int(st.obj_v_angle[st.player]),
        )
        hit = self._hop_price_memo.get(key, _NO_VIEW)
        if hit is not _NO_VIEW:
            return hit
        real, bearing, cursor = self.st, self.last_bearing, list(self.cursor)
        trial = st.clone()
        out = None
        total = 0.0
        try:
            for verb in ["boulder"] * k + ["robot"]:
                otype = mm.T_BOULDER if verb == "boulder" else mm.T_ROBOT
                if not self._can_build(trial, tile, otype):
                    break
                cost, _, view = self._price(trial, verb, tile)
                if view is None:
                    break
                total += cost
                self._commit_view(view, verb)
                if actions.create(trial, otype, tile) is None:
                    break
                if verb == "robot":
                    tail = cost
                    if not threat.player_sees_tile(trial, tile, trial.player):
                        break
                    cost, _, view = self._price(trial, "transfer", tile)
                    if view is None:
                        break
                    out = (total + cost, tail + cost)
        finally:
            self.st, self.last_bearing, self.cursor = real, bearing, cursor
        self._hop_price_memo[key] = out
        return out

    def _pick_hop(self, target, landset=None):
        """Ranked pedestal builds directed at ``target``: prefer tiles that gain LOS
        on it, then raise the eye, then the widest window.  The DESTINATION gate is
        what that hop actually costs (``_hop_price``), not a flat constant -- hop cost
        swings 745 f (ls42) to 1294 f (ls335) with how expensive the aims are from the
        eye, so a constant is simultaneously too strict and too lax.  The tile must
        stay drain-clear for ``tail``: the robot create plus the transfer, the only
        span a drainable body stands there (the boulders under it are not drainable,
        $16E6).  Charging the whole hop here over-gated -- it rejected tiles clear for
        every frame the robot existed.  The SOURCE half (the player stands on its
        current tile for the WHOLE build) is shadow-recorded only, see
        ``_record_hop_gate``.

        Ranking is cheap and pricing is not, so candidates are pre-filtered on
        ``_TAIL_FLOOR`` (a lower bound on the drainable span) and priced exactly only
        in rank order, until ``_TOP_HOPS`` survive -- the ones the pursuit would try.
        ``landset`` overrides the tile set (the frozen one ``_stranded``
        probes a refuelled stance against).

        A tile whose bare base already clears the eye (``base + ROBOT_EYE >
        my_eye``) is offered at k=0 AS WELL AS at k=1: the boulder is 2 energy, one
        create and a wider window the gate must find, none of which that tile needs,
        but it is also half a unit of eye.  Both forms are candidates and the eye
        ranking decides.  Neither shortcut works: REPLACING k=1 with k=0 costs ls0
        11 actions / 8283 f and pushes ls42 past a 900 s cut, and ranking k=0 behind
        every k>=1 candidate loses ls42 outright."""
        st = self.st
        my_eye = st.eye_z()
        reserve = self._reserve()
        margin = self._margin()
        cands = []
        for tile in self._landset(st) if landset is None else landset:
            base = self._tile_base(st, tile)
            if base is None:
                continue
            need = math.ceil((my_eye + EYE_EPS - ROBOT_EYE - base) / BOULDER_H)
            ks = [
                k
                for k in ((0, 1) if need <= 0 else (need,))
                # k is bounded here: a taller stack costs energy it lacks
                if st.energy - reserve >= 2 * k + mm.ENERGY_IN_OBJECTS[mm.T_ROBOT]
                and base + BOULDER_H * k + ROBOT_EYE > my_eye + EYE_EPS
            ]
            if not ks:
                continue
            exposed = self._exposing_enemies(tile)
            if not self._drain_gate("robot", tile, exposed, _TAIL_FLOOR + margin):
                continue
            window = self._gaze_window(tile, exposed=exposed)
            sees = self._tile_sees_target(tile, target)
            for k in ks:
                robot_eye = base + BOULDER_H * k + ROBOT_EYE
                cands.append(((sees, robot_eye, window), tile, k, window, exposed))
        cands.sort(key=lambda c: c[0], reverse=True)
        out = []
        for _, tile, k, window, exposed in cands:
            priced = self._hop_price(st, tile, k)
            tile_ok = priced is not None and window >= priced[1] + margin
            if self._hop_audit is not None:
                self._record_hop_gate(tile, k, exposed, tile_ok, priced)
            if tile_ok:
                out.append((tile, k, window))
                if len(out) == _TOP_HOPS:
                    break
        return out

    def _hop_exec(self, tile, k, window):
        """Build ``k`` boulders + a robot on ``tile`` and transfer up (``self.st``
        is the working clone); ``(g_delta, steps)`` or ``None`` if unsafe or
        infeasible.  The lone-hop builder shared by every pursuit macro.  ``window``
        is the tile window ``_pick_hop`` gated the hop on, recorded on each step."""
        st = self.st
        g = 0.0
        steps = []
        for _ in range(k):
            if not self._can_build(st, tile, mm.T_BOULDER):
                return None
            cost = self._charge(st, "boulder", tile)
            g += cost
            actions.create(st, mm.T_BOULDER, tile)
            steps.append(self._plan_step("boulder", tile, cost, GATE_TILE, window))
        if not self._can_build(st, tile, mm.T_ROBOT):
            return None
        cost = self._charge(st, "robot", tile)
        g += cost
        if actions.create(st, mm.T_ROBOT, tile) is None:
            return None
        steps.append(self._plan_step("robot", tile, cost, GATE_TILE, window))
        top = terrain.top_object(st, *tile)
        if not threat.player_sees_tile(st, tile, st.player):
            return None
        cost = self._charge(st, "transfer", tile)
        g += cost
        if not actions.transfer(st, top) or actions.player_dead(st):
            return None
        # a landing with no time before its first drain is a trap unless a seer is absorbable from here (_drain_gate("transfer", ...) inlined so the step records the window it gated on)
        exposed = self._exposing_enemies(tile)
        landed = self._gaze_window(tile, exposed=exposed)
        steps.append(self._plan_step("transfer", tile, cost, GATE_TILE, landed))
        gate_ok = landed >= self._margin()
        if not gate_ok and not self._absorbable_here(st):
            return None
        return g, steps

    def _absorbable_here(self, st):
        """Whether a living enemy is landable from the current stance (the
        counterattack is available)."""
        foes = enemies.enemy_slots(st)
        for e in foes:
            if e == actions.SENTINEL_SLOT and len(foes) > 1:
                continue
            tile = st.tile_of(e)
            if terrain.top_object(st, *tile) == e and self._landable(st, tile):
                return True
        return False

    def _can_build(self, st, tile, otype):
        return threat.player_sees_tile(st, tile, st.player) and actions.can_create(
            st, otype, tile
        )

    def _absorb_enemy_targets(self, st):
        foes = enemies.enemy_slots(st)
        out = []
        for e in foes:
            if e == actions.SENTINEL_SLOT and len(foes) > 1:
                continue  # Sentinel dead last ($1B8E lock)
            tile = st.tile_of(e)
            if terrain.top_object(st, *tile) == e and self._landable(st, tile):
                out.append((tile, e))
        return out

    def _c_absorb(self, node, tile, e):
        st = self._begin(node)
        if not actions.can_absorb(st, e):
            return None
        view = self._view_for(tile)
        if view is None:
            return None
        budget = self._aim_frames(view) + self._settle("absorb", view)
        window = self._player_window(exclude=e)
        if self._hot(budget, window):
            return None  # the player's body would be drained before the absorb fires
        cost = self._charge(st, "absorb", tile)
        g = node.g + cost
        if not actions.absorb(st, e):
            return None
        step = self._plan_step("absorb", tile, cost, GATE_BODY, window)
        return self._node(node, st, g, [step])

    def _reclaim_one(self, st, pedestal_only=False):
        """Absorb ONE landable spent pedestal/shell (base <= eye), or a tree when
        short; the player stays put so its own window bounds the aim.  Returns
        ``(g_delta, step)`` or ``None``.  ``pedestal_only`` skips the tree sweep so
        the inchworm recycle grabs only the player's own spent boulders/shells."""
        want_trees = (not pedestal_only) and st.energy < HOP_COST + 6
        for _value, tile in self._reclaim_targets(st, want_trees):
            if not threat.player_sees_tile(st, tile, st.player):
                continue
            view = self._view_for(tile)
            if view is None or self._hot(
                self._aim_frames(view) + self._settle("absorb", view)
            ):
                continue  # would be drained mid-reclaim: try a safer object
            g = self._charge(st, "absorb", tile)
            if not actions.absorb(st, terrain.top_object(st, *tile)):
                return None
            return g, self._plan_step("absorb", tile, g, GATE_BODY)
        return None

    def _c_reclaim(self, node):
        """Absorb landable spent pedestals, shells and, when short, trees, up to
        eight in one macro; the player stays put throughout."""
        st = self._begin(node)
        g = node.g
        steps = []
        for _ in range(_MAX_RECLAIM):
            got = self._reclaim_one(st)
            if got is None:
                break
            g += got[0]
            steps.append(got[1])
        if not steps:
            return None
        return self._node(node, st, g, steps)

    def _blocking_trees(self, st):
        """Landable tiles whose ONLY disqualification as a pedestal site is the tree
        on top: ``_tile_base`` is None there, but the surface the tree stands on
        (``_base_z`` of the tree) would carry a stack that raises the eye, within the
        energy left once the absorb has paid its own +1.  Tallest site first."""
        my_eye = st.eye_z()
        reserve = self._reserve()
        energy = st.energy + mm.ENERGY_IN_OBJECTS[mm.T_TREE]
        out = []
        for tile in self._landset(st):
            top = self._top(tile)
            if top is None or st.obj_type[top] != mm.T_TREE:
                continue
            base = self._base_z(top)
            k = max(0, math.ceil((my_eye + EYE_EPS - ROBOT_EYE - base) / BOULDER_H))
            if energy - reserve < 2 * k + mm.ENERGY_IN_OBJECTS[mm.T_ROBOT]:
                continue
            if base + BOULDER_H * k + ROBOT_EYE <= my_eye + EYE_EPS:
                continue
            out.append((base, tile))
        out.sort(reverse=True)
        return [tile for _base, tile in out]

    def _c_clear(self, node):
        """Absorb a tree BECAUSE it blocks a wanted pedestal site -- one child per
        blocked site, the player staying put so its own window bounds the aim.

        ``_reclaim_one``'s tree arm cannot do this: it is a FUEL heuristic gated on
        ``st.energy < HOP_COST + 6``, so it takes one tree, the energy it returns
        switches the gate off, and every further tree is permanently unreachable.
        At the ls335 root that is the whole board -- 4 landable tiles, 2 of them
        tree-topped at base 3.875 against an eye of 3.875, i.e. k=0 pedestals the
        moment the tree is gone -- and the search never built on either."""
        st0 = self._begin(node)
        children = []
        for tile in self._blocking_trees(st0)[:_TOP_CLEARS]:
            st = self._begin(node)
            slot = terrain.top_object(st, *tile)
            if slot is None or not actions.can_absorb(st, slot):
                continue
            if not threat.player_sees_tile(st, tile, st.player):
                continue
            view = self._view_for(tile)
            if view is None:
                continue
            if self._hot(self._aim_frames(view) + self._settle("absorb", view)):
                continue  # the player would be drained mid-clear
            cost = self._charge(st, "absorb", tile)
            if not actions.absorb(st, slot):
                continue
            step = self._plan_step("absorb", tile, cost, GATE_BODY)
            children.append(self._node(node, st, node.g + cost, [step]))
        return children

    def _c_endgame(self, node):
        """Sentinel gone (no enemy remains): robot on the platform, transfer,
        hyperspace -- the win."""
        st = self._begin(node)
        ptile = st.platform_xy
        g = node.g
        steps = []
        if not actions.on_platform(st):
            if not self._landable(st, ptile):
                return None
            cost = self._charge(st, "robot", ptile)
            g += cost
            slot = actions.create(st, mm.T_ROBOT, ptile)
            if slot is None:
                return None
            steps.append(
                self._plan_step(
                    "robot", ptile, cost, GATE_TILE, self._gaze_window(ptile)
                )
            )
            cost = self._charge(st, "transfer", ptile)
            g += cost
            if not actions.transfer(st, slot):
                return None
            steps.append(
                self._plan_step(
                    "transfer", ptile, cost, GATE_TILE, self._gaze_window(ptile)
                )
            )
        cost = self._charge(st, "hyperspace", ptile)
        g += cost
        actions.hyperspace(st)
        if not actions.won(st):
            return None
        steps.append(self._plan_step("hyperspace", ptile, cost, GATE_TILE, math.inf))
        return self._node(node, st, g, steps)


def main():
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("landscape", nargs="?", type=int, default=66)
    parser.add_argument("--max-actions", type=int, default=400)
    parser.add_argument(
        "--time-budget",
        type=float,
        default=None,
        help="wall-clock search cut (s); off by default -- setting it makes the plan depend on host load",
    )
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
