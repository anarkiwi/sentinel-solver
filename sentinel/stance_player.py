"""A* over the same macros, plus one child that follows a STANCE-GRAPH route.

``_c_route`` asks :mod:`sentinel.stancegraph` for a whole energy-feasible climb to a
stance that can strike the goal, then compiles it hop by hop through ``_hop_exec`` --
the ls335 Sentinel is landable from 13 of 1192 stances, which no ranker proposes.
"""

import argparse
import math

import numpy as np

from sentinel import actions, enemies, memmap as mm, stancegraph, stanceorder
from sentinel.astar_player import (
    AStarPlayer,
    GATE_BODY,
    GATE_TILE,
    _ABSORB_EST,
    _ENDGAME_EST,
    _HOP_EST,
    _MAX_PURSUE,
    _MAX_RECLAIM,
    _Node,
    _TAIL_FLOOR,
)
from sentinel.game import Game
from sentinel.playerbase import DRAIN_DELAY, HOP_COST, HOP_FRAMES, REVOLUTION_FRAMES

FUEL_RESERVE = 6  # energy above a hop's cost that _harvest tops the stance back up to
MAX_WAIT = 4000.0  # frames a planned wait may spend; beyond this the hop is not worth scheduling
ROUTE_REPAIRS = 6  # stances one route child may blacklist before it gives up
SUBGOAL = False  # stop the search at the DP's next enemy rather than at the win; OFF because it leaves ls335's opening unchanged while costing ls110 417 f and 8 energy

ROUTE_TARGETS = 3  # strategic goals routed to per expansion (nearest enemies first)
ROUTE_HOPS = 8  # hops one route child compiles before it hands back to the search
ROUTE_POLICY = "stuck"  # when to offer route children: stuck | root | always; measured equal-best and cheapest, and h0 loses identically under all three
ROUTE_METRIC = "frames"  # price legs in TIME including waits, not hop count: hops biased the climb toward tall stacks
ROUTE_FOLLOW = True  # a search that found nothing COMMITS to the route rather than stalling; it can only fire where the frontier already returned None
_MISS = object()


class StancePlayer(AStarPlayer):
    """:class:`AStarPlayer` with a stance-graph route generator bolted on.

    The graph is built from the board this player starts on and is a SNAPSHOT: later
    creates and absorbs change LOS, so a stale hop simply fails to compile and the route
    child returns the prefix it managed, as ``_c_reach`` does."""

    def __init__(
        self,
        game,
        graph=None,
        cache_dir=stancegraph.CACHE_DIR,
        route_policy=ROUTE_POLICY,
        subgoal=SUBGOAL,
        route_follow=ROUTE_FOLLOW,
        **kwargs,
    ):
        super().__init__(game, **kwargs)
        self.route_policy = route_policy
        self.route_follow = route_follow
        self.subgoal = subgoal
        self._goal_slot = None
        self._graph = graph
        self._graph_state = self.st.clone()
        self._cache_dir = cache_dir
        self._route_memo = {}
        self._edge_frames = {}  # stance -> the build frames the executor MEASURED there
        self._cost_epoch = 0  # bumped per correction; part of the route memo key
        self._mask_memo = {}
        self._order_memo = {}
        self._span_memo = {}

    @property
    def graph(self):
        """The stance graph for this board, built on first use and cached on disk.

        Built from the board captured at construction, never from ``self.st``: mid-search
        that is a speculative clone, which would make the graph -- and so the plan --
        depend on expansion order."""
        if self._graph is None:
            self._graph = stancegraph.StanceGraph.cached(
                self._graph_state, cache_dir=self._cache_dir
            )
        return self._graph

    def _route_mask(self, st):
        """This stance's landable set as a board mask, off the player's own memoized
        landset rather than a second sweep."""
        sig = self._sig(st)
        mask = self._mask_memo.get(sig)
        if mask is None:
            mask = np.zeros(mm.N * mm.N, dtype=bool)
            for tx, ty in self._landset(st):
                mask[tx * mm.N + ty] = True
            self._mask_memo[sig] = mask
        return mask

    def _route_for(self, st, target, blocked=()):
        """The stance route from this stance to a strike stance for ``target``.

        Memoized against ``_cost_epoch`` as well as the board: every correction
        ``_repair`` learns invalidates the routes priced before it."""
        key = (
            self._sig(st),
            int(st.energy),
            tuple(target),
            tuple(sorted(blocked)),
            self._cost_epoch,
        )
        got = self._route_memo.get(key, _MISS)
        if got is _MISS:
            got = self.graph.route(
                tuple(target),
                int(st.energy),
                start_seen=self._route_mask(st),
                start_tile=tuple(st.player_xy()),
                start_eye=st.eye_z(),
                blocked=blocked,
                metric=ROUTE_METRIC,
                wait=self._route_wait(st) if ROUTE_METRIC == "frames" else None,
                priced=self._edge_frames,
            )
            self._route_memo[key] = got
        return got

    def _stance_spans(self, st, tile):
        """The busy spans of ``tile`` on this board's clock, memoized per (stance, tile).

        The calendar is ABSOLUTE, so one computation answers every elapsed time: no
        re-simulation per edge, which is what makes waiting affordable as a route cost.
        """
        key = (self._sig(st), tuple(tile))
        got = self._span_memo.get(key)
        if got is None:
            self.st = st
            got = self._busy_spans(self._exposing_enemies(tile), REVOLUTION_FRAMES)
            self._span_memo[key] = got
        return got

    def _route_wait(self, st, need=_TAIL_FLOOR):
        """``wait(stance, elapsed)`` for ``route(metric="frames")``.

        Frames the hop onto that stance must sit out, read off the absolute calendar at
        the time the route reaches it.  Analytic and optimistic on purpose -- ``_hop_ok``
        re-asks the bit-exact clock before any of it is committed."""

        def wait(j, spent):
            tile = self.graph.stances[j][0]
            at = float(spent)
            for lo, hi in self._stance_spans(st, tile):
                if hi <= at:
                    continue
                if lo - at + DRAIN_DELAY >= need:
                    return 0.0
                at = hi
            return min(at - spent, MAX_WAIT)

        return wait

    def _hop_funds(self, st, cost=HOP_COST):
        """``(ok, wait)``: whether this stance funds a hop costing ``cost``, and the
        frames to wait until it does.

        A route NAMES its next stance, so the spend is exactly ``2k + robot`` rather than
        ``HOP_COST``, the k=1 price of a hop nobody has picked yet: ls335's second hop is
        a k=0 landing costing 3, held against 5 at the E=3 the router itself predicted.

        Exposure is the other half of funding, and it is asked FIRST: ``_can_hop`` is a
        drain-RATE test -- can the body pay the hop and the drains its span bills -- which
        a stance can pass while no span of the sweep is long enough to build in at all.
        ls335 (10,18) affords a hop at E=7 and has a 367 f window ceiling (400 f is
        already ``inf``) against destinations priced at 1250 f+, so affordability alone
        called a dead end fundable."""
        self.st = st
        if st.energy - self._reserve() < cost:
            return False, 0.0  # genuinely broke: no phase of the sweep funds it
        wait = self._earliest_start(tuple(st.player_xy()), HOP_FRAMES)
        if wait == math.inf or wait > MAX_WAIT:
            return False, 0.0
        return (True, 0.0) if self._can_hop(st, cost) else (True, wait)

    def _can_leave(self, st, target):
        """Whether a landing can be LEFT again, or is a trap that has to be refused.

        ``_hop_ok`` gates ARRIVAL -- can this tile hold the build long enough -- and
        nothing gated departure, so ls335's route spends 9 of the opening 10 energy
        climbing a stack it can never step off.  A body on the bare tile has ONE
        full-sight seer; one on three boulders has FOUR, and a draining enemy stops
        rotating ($178C returns before the $17F9 rotate), so the cone never clears: the
        window ceiling collapses below any hop's aim tail and no reclaim is aimable.

        A landing that strikes the target needs no departure -- the absorb is the exit.
        """
        self.st = st
        return self._landable(st, target) or self._hop_funds(st)[0]

    def _hop_ok(self, st, tile, k, wait_ok=True):
        """``(window, wait)`` for this hop, or ``None`` if no phase of the sweep fits it.

        A cone that cannot cover the hop NOW is a delay, not a refusal: ``_earliest_start``
        returns the first verified moment the tile holds ``priced`` tail clear.  Without
        it ls335's (13,27) reads 450 f against an 1141 f hop and is refused outright,
        when waiting 773 f opens 1236 f.

        ``wait_ok=False`` forbids scheduling and asks only whether the hop stands RIGHT
        NOW.  That is the re-gate ``_try_hop`` runs after it has actually waited: the
        price was taken on the pre-wait board, and the wait is exactly what invalidates
        it.

        The wait is scheduled on the DESTINATION's calendar but served by the body on the
        SOURCE tile, so it is billed with the build: 45 of the 48 hops ls335 tries at
        (8,21) are waits, and the 890 f one that opens (12,27) from 450 f to 1203 f also
        takes the body from E=12 to E=10 and its window down to 4 f.  Charging only the
        build made standing still look free."""
        self.st = st
        margin = self._margin()
        exposed = self._exposing_enemies(tile)
        priced = self._hop_price(st, tile, k)
        if priced is None:
            return None
        need = priced[1] + margin
        window = self._gaze_window(tile, exposed=exposed)
        wait = 0.0
        if window < need or not self._drain_gate(
            "robot", tile, exposed, _TAIL_FLOOR + margin
        ):
            if not wait_ok:
                return None
            wait = self._earliest_start(tile, need, exposed=exposed)
            if wait == math.inf or wait > MAX_WAIT:
                return None
            window = need  # the verified span the clock agreed to at `wait`
        if not self._affords(
            2 * k + mm.ENERGY_IN_OBJECTS[mm.T_ROBOT],
            wait + priced[0],
            window=self._player_window(),
        ):
            return None
        return window, wait

    def _c_route(self, node, target):
        """Follow the stance route toward ``target`` until it is landable.

        Interleaves the inchworm reclaim when the next hop is unaffordable, and stops at
        the first hop the live board no longer permits, returning the prefix it made."""
        st = self._begin(node)
        route = self._route_for(st, target)
        if route is None:
            return None
        g = node.g
        steps = []
        blocked = set()
        i = 0
        for _ in range(_MAX_PURSUE):
            if i >= min(len(route.path), ROUTE_HOPS):
                break
            self.st = st
            if self._landable(st, target):
                break
            node_i = route.path[i]
            tile, k = self.graph.stances[node_i]
            fundable, source_wait = self._hop_funds(
                st, 2 * k + mm.ENERGY_IN_OBJECTS[mm.T_ROBOT]
            )
            if not fundable:
                got = self._reclaim_one(st)
                if got is None:
                    break
                g += got[0]
                steps.append(got[1])
                continue
            if source_wait:
                here = tuple(st.player_xy())
                enemies.advance_frames(st, int(source_wait))
                g += source_wait
                steps.append(
                    self._plan_step("wait", here, source_wait, GATE_BODY, HOP_FRAMES)
                )
            got = self._try_hop(st, tile, k, target)
            if got is None:
                route = self._repair(st, target, blocked, node_i, tile, k)
                if route is None:
                    break
                i = 0
                continue
            st, delta, hop_steps = got
            g += delta
            steps.extend(hop_steps)
            i += 1
        return self._node(node, st, g, steps) if steps else None

    def _try_hop(self, st, tile, k, target):
        """One routed hop on a TRIAL clone: ``(landed, g, steps)``, or ``None`` when the
        board refuses it or the landing cannot be left.

        Trial rather than in place because a refusal has to leave ``st`` intact for
        ``_repair`` to re-route from -- the discipline ``_advance_hop`` already keeps for
        the ranked climb, and what makes ``_can_leave`` a gate rather than an epitaph.
        Harvest runs BEFORE that gate: the reclaims are the landing's own way out."""
        ok = self._hop_ok(st, tile, k)
        if ok is None:
            return None
        window, wait = ok
        trial = st.clone()
        self.st = trial
        steps = []
        if wait:
            enemies.advance_frames(trial, int(wait))
            again = self._hop_ok(trial, tile, k, wait_ok=False)
            if again is None:
                return None  # the wait moved the board the price was taken on
            window = again[0]
            steps.append(self._plan_step("wait", tile, wait, GATE_TILE, window))
        got = self._hop_exec(tile, k, window)
        if got is None:
            return None
        g = wait + got[0] + self._harvest(trial, got[1])
        steps.extend(got[1])
        return (trial, g, steps) if self._can_leave(trial, target) else None

    def _repair(self, st, target, blocked, node_i, tile, k):
        """Correct the router's price for the stance it just failed on, and re-route.

        LazySP: ``route`` prices an edge at ``build_frames(k)``, a function of k ALONE
        (372/469/565/662 f), while ``_hop_price`` -- what the executor actually charges --
        runs 981-2261 f and spreads ~790 f across tiles at fixed k.  The router therefore
        cannot tell a cheap landing from an unbuildable one, and blacklisting one stance
        per refusal threw away the exact number needed to fix that.  Measuring the truth
        only on edges a proposed route USES pays for exactness where it changes the answer
        and nowhere else.

        The veto and the correction are SEPARATE verdicts, and both are recorded.  The
        stance was refused here and now, so it leaves the current query whatever the
        reason; the measured price is what the NEXT query, and every later search, ranks
        with.  Vetoing only when the price looked uninformative meant never vetoing at
        all -- the measurement always exceeds the floor.

        A refusal the TILE owns condemns its whole k column.  The graph is 298 tiles x 4
        stacks on ls335, the gaze window is a property of the tile, and banning one stance
        at a time spent the entire budget re-asking (12,27) at k=0,1,2,3.  The budget is
        counted in TILES for the same reason."""
        if len({self.graph.stances[j][0] for j in blocked}) >= ROUTE_REPAIRS:
            return None
        self.st = st
        priced = self._hop_price(st, tile, k)
        if priced is not None:
            was = self._edge_frames.get(node_i, stancegraph.build_frames(k))
            self._edge_frames[node_i] = max(was, priced[0])  # weights only ever rise
            self._cost_epoch += 1
        if self._tile_refused(st, tile, priced):
            blocked.update(self.graph.stances_on(tile).tolist())
        else:
            blocked.add(node_i)
        return self._route_for(st, target, blocked=frozenset(blocked))

    def _tile_refused(self, st, tile, priced):
        """Whether the refusal belongs to the TILE rather than to this stack height.

        The tail a hop stands drainable for is the robot create plus the transfer -- no
        boulder is drainable ($16E6) -- so it barely moves with k, and a tile whose sweep
        never holds it refuses every stance on it.  An unbuildable stack or an energy
        shortfall is the stance's own and bans only that one."""
        if priced is None:
            return False
        self.st = st
        return self._earliest_start(tile, priced[1] + self._margin()) == math.inf

    def _harvest(self, st, steps):
        """Absorb reachable fuel from the landing until the stance can fund a hop again.

        ``route`` credits each landing's cheap-aim trees, but reclaiming only when the
        NEXT hop is unaffordable never banks them, so the executor arrived at E=1 where
        the graph promised 14 and every node lost its children."""
        g = 0.0
        for _ in range(_MAX_RECLAIM):
            if st.energy >= HOP_COST + FUEL_RESERVE:
                break
            got = self._reclaim_one(st)
            if got is None:
                break
            g += got[0]
            steps.append(got[1])
        return g

    def _order_for(self, st):
        """The subset DP's goal order for this board, memoized per live enemy set."""
        key = (self._sig(st), tuple(sorted(enemies.enemy_slots(st))))
        got = self._order_memo.get(key)
        if got is None:
            got = stanceorder.order(self.graph, st, self._route_mask(st), st.eye_z())
            self._order_memo[key] = got
        return got

    def _route_targets(self, st):
        """The next goals in the DP's order, skipping any already landable.

        Nearest-first is what the A\\* player guesses with; the order is what a human
        plays and what the ordering DP solves for."""
        if not self.graph:
            return self._nearest_targets(st)
        foes = enemies.enemy_slots(st)
        locked = tuple(st.platform_xy) if len(foes) > 1 else None
        out = []
        for tile in self._order_for(st):
            if tile == locked or self._landable(st, tile):
                continue  # the Sentinel waits on the $1B8E lock; landable needs no route
            if tile not in out:
                out.append(tile)
        return out[:ROUTE_TARGETS]

    def _subgoal_slot(self, st):
        """The enemy slot the DP order says to take next, or ``None`` for the win itself.

        The Sentinel is skipped while any sentry lives ($1B8E), so the subgoal is always
        an absorb the ROM will actually permit."""
        if not self.subgoal or st.is_empty(actions.SENTINEL_SLOT):
            return None
        foes = enemies.enemy_slots(st)
        if len(foes) <= 1:
            return None  # only the Sentinel left: search straight for the win
        by_tile = {}
        for e in foes:
            by_tile.setdefault(tuple(st.tile_of(e)), e)
        for tile in self._order_for(st) if len(self.graph) else ():
            e = by_tile.get(tile)
            if e is not None and e != actions.SENTINEL_SLOT:
                return e
        got = self._pursue_targets(st)
        return got[0] if got else None

    def _is_goal(self, st):
        """The win, or the DP's current subgoal once that enemy is gone."""
        if actions.won(st):
            return True
        slot = self._goal_slot
        return slot is not None and st.is_empty(slot)

    def _search(self, margin_k=None):
        """Fix the subgoal from the LIVE board, then search for it rather than the win.

        A complete winning line is ~120 actions on ls335 and no single search spans it,
        so the run loop got ``None`` and was drained to death having acted zero times.
        """
        self._goal_slot = self._subgoal_slot(self.st)
        try:
            plan = super()._search(margin_k)
        finally:
            self._goal_slot = None
        if plan is not None or not self.route_follow:
            return plan
        return self._route_plan()

    def _route_plan(self):
        """Compile the DP's next goal into a plan directly, bypassing the frontier.

        A route child bundles ~9 steps, so its ``g`` is thousands of frames and
        ``f = g + weight*h`` pops cheap shallow children first: it never expands.  A
        search returning ``None`` costs the run loop everything, so a route is worth
        committing to on its own terms rather than entering a race it cannot win."""
        if not self.graph:
            return None
        real, bearing, cursor = self.st, self.last_bearing, list(self.cursor)
        try:
            node = _Node(real.clone(), 0.0, (), None, bearing, cursor)
            node.key = self._key(node.state)
            for target in self._route_targets(real):
                child = self._c_route(node, target)
                if child is not None and child.path:
                    return list(child.path)
            return None
        finally:
            self.st, self.last_bearing, self.cursor = real, bearing, cursor

    def _hops_to_tile(self, st, tile):
        """Hops from this stance to one that strikes ``tile``, or ``None`` if unreached."""
        graph = self.graph
        if not graph:
            return None
        if self._landable(st, tile):
            return 0.0
        reach = self._route_mask(st)[graph._cols] & (graph.eye > st.eye_z())
        if not reach.any():
            return None
        got = graph.hops_to(tuple(tile))[reach]
        live = got[got >= 0]
        return 1.0 + float(live.min()) if live.size else None

    def _goals_landable(self, st):
        """Whether every goal left on the board is already landable from this stance.

        Distinguishes "nothing to route to because it is all in reach" from "the DP
        bridged nothing", which ``_route_targets`` returns as the same empty list."""
        goals = [tuple(st.tile_of(e)) for e in enemies.enemy_slots(st)]
        if st.is_empty(actions.SENTINEL_SLOT):
            goals.append(tuple(st.platform_xy))
        return all(self._landable(st, tile) for tile in goals)

    def _nearest_targets(self, st):
        """Target selection without a graph: the fallback the injected-empty-graph and
        no-sweep paths take."""
        if st.is_empty(actions.SENTINEL_SLOT):
            plat = tuple(st.platform_xy)
            return [] if self._landable(st, plat) else [plat]
        return [tuple(st.tile_of(e)) for e in self._pursue_targets(st)[:ROUTE_TARGETS]]

    def _graph_hops(self, st):
        """Hops from this stance to one that strikes the nearest goal, or ``None`` when
        the graph cannot answer -- the board-derived replacement for the eye deficit."""
        graph = self.graph
        if not graph:
            return None
        targets = self._route_targets(st)
        if not targets:
            return (
                0.0 if self._goals_landable(st) else None
            )  # 0 hops to go, vs no answer
        mask = self._route_mask(st)
        reach = mask[graph._cols] & (graph.eye > st.eye_z())
        if not reach.any():
            return None
        best = None
        for target in targets:
            dist = graph.hops_to(target)[reach]
            live = dist[dist >= 0]
            if live.size:
                got = 1.0 + float(live.min())
                best = got if best is None else min(best, got)
        return best

    def _h(self, st):
        """``AStarPlayer._h`` with the eye-deficit hop estimate replaced by the graph's
        own backward hop count, so a bundled route and a single ranked hop are scored on
        the same scale.  Under a subgoal it prices that one absorb, not the whole win.
        """
        if self._is_goal(st):
            return 0.0
        slot = self._goal_slot
        if slot is not None and not st.is_empty(slot):
            hops = self._hops_to_tile(st, tuple(st.tile_of(slot)))
            if hops is not None:
                return hops * _HOP_EST + _ABSORB_EST
        hops = self._graph_hops(st)
        if hops is None:
            return super()._h(st)
        remaining = len(enemies.enemy_slots(st))
        return remaining * _ABSORB_EST + hops * _HOP_EST + _ENDGAME_EST

    def _wants_route(self, node, children):
        """Whether this node should be offered route children under ``route_policy``.

        A weighted search returns the FIRST goal it pops, so an extra child can surface a
        worse goal sooner -- offering routes at every node cost ls42 2 actions and ls110 6.
        Offering them only when a node is childless never fires at the ls335 root, which
        has 11 ranked children and still loses.  ``root`` is the measured middle."""
        if not children:
            return True
        if self.route_policy == "always":
            return True
        return self.route_policy == "root" and not node.path

    def _expand(self, node):
        """The ranked children, plus route children as ``route_policy`` allows."""
        children = super()._expand(node)
        if not self._wants_route(node, children):
            return children
        for target in self._route_targets(node.state):
            child = self._c_route(node, target)
            if child is not None:
                children.append(child)
        return children


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("landscape", nargs="?", type=int, default=335)
    parser.add_argument("--max-actions", type=int, default=400)
    parser.add_argument("--time-budget", type=float, default=None)
    parser.add_argument("--node-budget", type=int, default=200000)
    parser.add_argument("--cache-dir", default=stancegraph.CACHE_DIR)
    parser.add_argument("--quiet", action="store_true")
    args = parser.parse_args(argv)
    game = Game.typed(args.landscape)
    player = StancePlayer(
        game,
        cache_dir=args.cache_dir,
        verbose=not args.quiet,
        node_budget=args.node_budget,
        time_budget=args.time_budget,
    )
    won = player.run(max_actions=args.max_actions)
    print(
        f"landscape {args.landscape}: {'WON' if won else 'lost'} in "
        f"{len(player.trace)} actions / {player.frames} frames, "
        f"energy {game.energy}, dead={actions.player_dead(game.state)}"
    )
    return 0 if won else 1


if __name__ == "__main__":
    raise SystemExit(main())
