"""The board's GEOMETRY as an explicit graph, solved exactly instead of searched.

A stance is ``(tile, k)`` -- a body on ``k`` boulders -- and an edge is "that stance can
keyboard-land this one".  ~1200 stances on a 32x32 board, so a route is a
resource-constrained shortest path and "what can strike this tile" is a table lookup.
"""

import argparse
import hashlib
import heapq
import os
import time

import numpy as np

from sentinel import actions, landtable, memmap as mm, terrain
from sentinel.game import Game
from sentinel.playerbase import (
    BOULDER_H,
    DRAIN_DELAY,
    FOV_MARGIN,
    ROBOT_EYE,
    SIGHTS_CENTRE,
    TAP_FRAMES,
)
from sentinel import actioncost, enemies

KMAX = 3  # stack heights a stance may carry; k=4+ adds no strike tile on ls335
HOP_ENERGY = mm.ENERGY_IN_OBJECTS[mm.T_ROBOT]  # plus 2 per boulder
CHEAP_AIM = 240.0  # human ls335 tree absorbs: median 105 f, max 240 -- "easy reach"
CONE_DUTY = (enemies.FOV_SCAN + 2 * FOV_MARGIN) / 256.0  # duty cycle of a rotating cone
CACHE_DIR = os.path.join("out", "stancegraph")
CACHE_VERSION = 2  # bump when the on-disk layout or the sweep's meaning changes
_CREATE_F = TAP_FRAMES + actioncost.SETTLE["create"]  # per-create floor
_TRANSFER_F = TAP_FRAMES + actioncost.SETTLE["transfer"]
_NCELL = mm.N * mm.N


def flat_tiles(state):
    """Bare flat tiles -- the only ones a stack builds on (a sloped tile never lands)."""
    out = []
    for x in range(mm.N):
        for y in range(mm.N):
            b = terrain.tile_byte(state, x, y)
            if b < mm.OBJECT_TILE and not (b & 0x0F):
                out.append((x, y))
    return out


def stance_eye(state, tile, k):
    """Eye height of a body standing on ``k`` boulders on ``tile``."""
    return (terrain.tile_byte(state, *tile) >> 4) + BOULDER_H * k + ROBOT_EYE


def build_frames(k):
    """Frames a body stands still building ``k`` boulders + a robot, then transferring."""
    return _CREATE_F * (k + 1) + _TRANSFER_F


def hop_energy(k):
    """Energy a hop onto a ``k``-boulder stance spends: the stack plus the robot."""
    return 2 * k + HOP_ENERGY


def signature(state, kmax=KMAX):
    """Cache key: the tile map every LOS march reads, plus the stack heights swept."""
    digest = hashlib.sha1(bytes(state.mem[0x0400:0x0800])).hexdigest()[:16]
    return f"v{CACHE_VERSION}_k{kmax}_{digest}"


def _probe(state, tile, k):
    """A clone with ``k`` boulders + a robot on ``tile``, the player inside it."""
    st = state.clone()
    st.energy = 60  # geometry only; energy is a separate layer
    for _ in range(k):
        if actions.create(st, mm.T_BOULDER, tile) is None:
            return None
    slot = actions.create(st, mm.T_ROBOT, tile)
    if slot is None:
        return None
    st.player = slot
    return st, slot


def _cheap_fuel(player, probe, seen):
    """Trees a stance can absorb at an aim inside ``CHEAP_AIM``.

    Counting trees a stance merely SEES routed the climb through landings whose nearest
    tree costs 410-700 f to aim at, against the human's observed 105 f median -- fuel you
    cannot reach cheaply is not fuel, it is a detour under a drain clock."""
    player.st = probe
    player.last_bearing = None
    player.cursor = list(SIGHTS_CENTRE)
    n = 0
    for tile in seen:
        top = terrain.top_object(probe, *tile)
        if top is None or probe.obj_type[top] != mm.T_TREE:
            continue
        view = player._view_for(tile)
        if view is not None and player._aim_frames(view) <= CHEAP_AIM:
            n += 1
    return n


class Route:
    """A stance route: the hops taken, the energy left on arrival, and the goal tile."""

    __slots__ = ("path", "energy", "goal")

    def __init__(self, path, energy, goal):
        self.path = path  # stance indices in order, excluding the start
        self.energy = energy
        self.goal = goal

    def __len__(self):
        return len(self.path)

    def __repr__(self):
        return f"Route(hops={len(self.path)}, energy={self.energy}, goal={self.goal})"


class StanceGraph:
    """Every ``(tile, k)`` stance on a board, what each can land, and the routes between.

    ``seen`` is a bool matrix ``[stance, tile]``.  A ROW is "what can this stance land",
    the forward question; a COLUMN is "which stances can strike this tile", the backward
    question a forward search cannot ask -- and on ls335 the Sentinel answers 13 of 1192.
    """

    __slots__ = (
        "stances",
        "seen",
        "fuel",
        "hot",
        "eye",
        "kmax",
        "sig",
        "_adj",
        "_cols",
        "_hops",
        "seers",
    )

    def __init__(self, stances, seen, fuel, hot, eye, kmax, sig, seers=None):
        self.stances = stances
        self.seen = seen
        self.fuel = fuel
        self.hot = hot
        self.eye = eye
        self.kmax = kmax
        self.sig = sig
        self.seers = (
            seers  # bool[stance, slot]: WHICH enemies see it, so a dead one can go
        )
        self._adj = None
        self._hops = {}
        self._cols = np.array([t[0] * mm.N + t[1] for t, _k in stances], dtype=np.int32)

    def __len__(self):
        return len(self.stances)

    @classmethod
    def build(cls, state, kmax=KMAX, log=None, tiles=None):
        """Sweep every stance's landable set; ~40 s a board, so prefer :meth:`cached`.

        ``tiles`` restricts the sweep to a subset of :func:`flat_tiles` -- a partial
        graph, sound for every stance it holds, which is what a test can afford."""
        from sentinel.astar_player import AStarPlayer  # circular at module scope

        grids = landtable.lattice(coarse=True)
        player = AStarPlayer(Game(state.clone()))
        stances, rows, fuel, sees, eye = [], [], [], [], []
        t0 = time.time()
        for tile in flat_tiles(state) if tiles is None else tiles:
            for k in range(kmax + 1):
                pr = _probe(state, tile, k)
                if pr is None:
                    continue
                st, slot = pr
                seen = landtable.landable_set(st, slot, grids=grids)
                row = np.zeros(_NCELL, dtype=bool)
                for tx, ty in seen:
                    row[tx * mm.N + ty] = True
                watch = np.zeros(mm.NUM_SLOTS, dtype=bool)
                for e, _ah, _full in player._exposures(st, slot):
                    watch[e] = True
                stances.append((tile, k))
                rows.append(row)
                fuel.append(_cheap_fuel(player, st, seen))
                sees.append(watch)
                eye.append(stance_eye(state, tile, k))
        seers = np.array(sees, dtype=bool).reshape(len(sees), mm.NUM_SLOTS)
        graph = cls(
            stances,
            np.array(rows, dtype=bool).reshape(len(rows), _NCELL),
            np.array(fuel, dtype=np.int32),
            seers.sum(axis=1).astype(np.int32),
            np.array(eye, dtype=np.float32),
            kmax,
            signature(state, kmax),
            seers,
        )
        if log:
            log(
                f"{len(graph)} stances swept in {time.time() - t0:.0f}s; "
                f"{int((graph.fuel > 0).sum())} have cheap-aim fuel, "
                f"{int((graph.hot > 0).sum())} lie in some enemy's sightline"
            )
        return graph

    @classmethod
    def cached(cls, state, kmax=KMAX, log=None, cache_dir=CACHE_DIR):
        """:meth:`build` memoized on disk: an unchanged tile map reloads in ms, not 40 s."""
        sig = signature(state, kmax)
        path = os.path.join(cache_dir, f"{sig}.npz")
        if os.path.exists(path):
            try:
                return cls.load(path)
            except (OSError, ValueError, KeyError):  # truncated or stale
                pass
        graph = cls.build(state, kmax, log)
        os.makedirs(cache_dir, exist_ok=True)
        graph.save(path)
        return graph

    def save(self, path):
        """Write the graph, ``seen`` bit-packed (1192 x 1024 bits: 152 KB, ~18 KB npz)."""
        tmp = f"{path}.{os.getpid()}.tmp"
        np.savez_compressed(
            tmp,
            tiles=np.array([t for t, _k in self.stances], dtype=np.int16),
            ks=np.array([k for _t, k in self.stances], dtype=np.int8),
            seen=np.packbits(self.seen, axis=1),
            fuel=self.fuel,
            hot=self.hot,
            eye=self.eye,
            meta=np.array([self.kmax], dtype=np.int32),
            sig=np.array([self.sig]),
            seers=np.packbits(
                (
                    np.zeros((len(self.stances), mm.NUM_SLOTS), dtype=bool)
                    if self.seers is None
                    else self.seers
                ),
                axis=1,
            ),
        )
        os.replace(tmp + ".npz", path)
        return path

    @classmethod
    def load(cls, path):
        """Read back a :meth:`save`d graph."""
        with np.load(path, allow_pickle=False) as z:
            got = {k: np.asarray(z[k]) for k in z.files}
        stances = [
            ((int(t[0]), int(t[1])), int(k)) for t, k in zip(got["tiles"], got["ks"])
        ]
        seen = np.unpackbits(got["seen"], axis=1, count=_NCELL).astype(bool)
        return cls(
            stances,
            seen,
            got["fuel"],
            got["hot"],
            got["eye"],
            int(got["meta"].ravel()[0]),
            str(got["sig"].ravel()[0]),
            np.unpackbits(got["seers"], axis=1, count=mm.NUM_SLOTS).astype(bool),
        )

    def strike_stances(self, tile):
        """Indices of every stance that can keyboard-land ``tile`` -- the COLUMN read.

        The query a forward search cannot make: it routes the endgame to a platform not
        yet in sight, and lets a target be chosen on whether it is reachable at all."""
        return np.flatnonzero(self.seen[:, tile[0] * mm.N + tile[1]])

    def lands(self, index, tile):
        """Whether stance ``index`` can land ``tile`` -- the single-cell read."""
        return bool(self.seen[index, tile[0] * mm.N + tile[1]])

    def hops_to(self, tile, monotone=True):
        """Hops from every stance to the nearest one that can strike ``tile``; -1 never.

        A backward level BFS from the strike set over the reversed graph, so it answers
        for all 1192 stances at once.  This is the distance-to-goal a forward search
        cannot derive, and the term an eye-deficit proxy stands in for."""
        key = (tuple(tile), bool(monotone))
        got = self._hops.get(key)
        if got is None:
            got = self._backward(tile, monotone)
            self._hops[key] = got
        return got

    def _backward(self, tile, monotone):
        n = len(self.stances)
        dist = np.full(n, -1, dtype=np.int32)
        edges = self.adj if monotone else self.seen[:, self._cols]
        frontier = np.zeros(n, dtype=bool)
        frontier[self.strike_stances(tile)] = True
        dist[frontier] = 0
        depth = 0
        while frontier.any():
            depth += 1
            pred = edges[:, frontier].any(axis=1) & (dist < 0)
            if not pred.any():
                break
            dist[pred] = depth
            frontier = pred
        return dist

    @property
    def adj(self):
        """``adj[i, j]``: stance ``i`` can land ``j``'s tile AND ``j`` is higher.

        The eye-monotone half makes the graph a DAG.  It is an APPROXIMATION -- the human
        ls335 line descends twice to cross the board -- kept because it is what keeps the
        route search linear; ``route(monotone=False)`` drops it."""
        if self._adj is None:
            adj = self.seen[:, self._cols].copy()
            adj &= self.eye[None, :] > self.eye[:, None]
            self._adj = adj
        return self._adj

    def _successors(self, node, monotone):
        if monotone:
            return np.flatnonzero(self.adj[node])
        return np.flatnonzero(self.seen[node, self._cols])

    def watchers(self, dead=()):
        """Enemies still watching each stance, with ``dead`` slots discounted.

        An absorbed enemy's cone is gone, so the board opens up as the hunt proceeds --
        the graph mutation the ordering DP reasons over."""
        if self.seers is None or not dead:
            return self.hot
        live = self.seers.copy()
        live[:, list(dead)] = False
        return live.sum(axis=1).astype(np.int32)

    def _spend(self, dead=()):
        """Energy each stance costs to hop onto: its stack, plus the drains the build is
        billed under a sightline at the rotating cone's duty cycle."""
        ks = np.array([k for _t, k in self.stances], dtype=np.int32)
        toll = np.array(
            [
                round(h * CONE_DUTY * build_frames(k) / DRAIN_DELAY)
                for h, k in zip(self.watchers(dead).tolist(), ks.tolist())
            ],
            dtype=np.int32,
        )
        return hop_energy(ks) + toll

    def route(
        self,
        target,
        energy,
        start=None,
        start_seen=None,
        start_tile=None,
        start_k=0,
        start_eye=None,
        monotone=True,
        dead=(),
        blocked=(),
    ):
        """Fewest-hop ENERGY-FEASIBLE climb to a stance that can land ``target``.

        A resource-constrained shortest path over ``(stance, energy)``: a hop spends its
        stack plus its exposure toll, and the arrival pays back the pedestal climbed off
        plus the cheap-aim trees it reaches.  Start at a stance or at a live body.
        """
        goals = set(self.strike_stances(target).tolist()) - set(blocked)
        if not goals:
            return None
        veto = set(blocked)
        n = len(self.stances)
        cap = int(mm.ENERGY_MASK)
        if start is not None:
            start_tile, start_k = self.stances[start]
            start_eye = float(self.eye[start])
        elif start_seen is None or start_eye is None:
            raise ValueError("route needs start=, or start_seen= with start_eye=")

        spend = self._spend(dead)
        gain = self.fuel * mm.ENERGY_IN_OBJECTS[mm.T_TREE]
        # best[node, e]: fewest hops to node holding exactly e; a state is dominated when a settled one there holds >= energy in <= hops
        best = np.full((n + 1, cap + 1), 1 << 30, dtype=np.int32)
        parent = (
            {}
        )  # keyed by hops too: (node, energy) alone lets a longer push overwrite a shorter one's chain
        origin = n if start is None else start
        heap = [(0, -min(cap, int(energy)), origin)]
        while heap:
            hops, nege, node = heapq.heappop(heap)
            have = -nege
            if node in goals:
                path = self._unwind(parent, (node, have, hops))
                return Route(path, have, target)
            if best[node, have:].min() <= hops:
                continue
            best[node, have] = hops
            if node == n:
                here, here_k = start_tile, start_k
                succ = np.flatnonzero(start_seen[self._cols])
                if monotone:
                    succ = succ[self.eye[succ] > start_eye]
            else:
                here, here_k = self.stances[node]
                succ = self._successors(node, monotone)
            for j in succ.tolist():
                if j in veto:
                    continue  # the executor's exact gates already refused this stance
                cost = int(spend[j])
                if have < cost:
                    continue  # not affordable: the geometric optimum is often infeasible
                got = have - cost + int(gain[j])
                if here is not None and self.lands(j, here):
                    got += hop_energy(here_k)  # the inchworm reclaim of the pedestal
                got = min(cap, got)
                if best[j, got:].min() <= hops + 1:
                    continue
                nxt = (j, got, hops + 1)
                if nxt in parent:
                    continue
                parent[nxt] = (node, have, hops)
                heapq.heappush(heap, (hops + 1, -got, j))
        return None

    @staticmethod
    def _unwind(parent, state):
        """The stance indices of the chain ending at ``state``, start excluded."""
        path = []
        while state in parent:
            path.append(state[0])
            state = parent[state]
        return list(reversed(path))

    def describe(self, route, state):
        """The route as ``(index, tile, k, eye_before, eye_after)`` rows."""
        eye = state.eye_z()
        for i, node in enumerate(route.path, 1):
            tile, k = self.stances[node]
            after = float(self.eye[node])
            yield i, tile, k, eye, after
            eye = after


def live_seen(state, grids=None):
    """The live body's landable set as a board mask -- the route's start row."""
    grids = landtable.lattice(coarse=True) if grids is None else grids
    seen = np.zeros(_NCELL, dtype=bool)
    for tx, ty in landtable.landable_set(state, state.player, grids=grids):
        seen[tx * mm.N + ty] = True
    return seen


def route_to(state, target, kmax=KMAX, log=None, energy=None, cache_dir=CACHE_DIR):
    """Build (or load) the graph for ``state`` and route the live body to ``target``."""
    graph = StanceGraph.cached(state, kmax, log, cache_dir)
    return graph, graph.route(
        target,
        state.energy if energy is None else energy,
        start_seen=live_seen(state),
        start_tile=tuple(state.player_xy()),
        start_eye=state.eye_z(),
    )


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("landscape", type=int, default=335, nargs="?")
    ap.add_argument("--kmax", type=int, default=KMAX)
    ap.add_argument("--target", default=None, help="tile 'x,y'; default the Sentinel")
    ap.add_argument("--cache-dir", default=CACHE_DIR)
    ap.add_argument(
        "--open",
        action="store_true",
        help="strip trees/sentries: the optimistic upper-bound geometry",
    )
    args = ap.parse_args(argv)

    def log(msg):
        print(f"[{time.strftime('%H:%M:%S')}] {msg}", flush=True)

    game = Game.typed(args.landscape)
    st = game.state
    if args.open:
        for slot, otype, _x, _y in game.objects():
            if otype in (mm.T_TREE, mm.T_SENTRY, mm.T_MEANIE):
                actions.remove_object(st, slot)
    target = (
        tuple(int(v) for v in args.target.split(","))
        if args.target
        else st.tile_of(actions.SENTINEL_SLOT)
    )
    log(f"ls{args.landscape}: player {st.player_xy()} eye {st.eye_z():.2f}, {target=}")
    graph, got = route_to(st, target, args.kmax, log, cache_dir=args.cache_dir)
    strike = graph.strike_stances(target)
    tiles = {graph.stances[i][0] for i in strike.tolist()}
    log(f"{len(strike)} strike stances on {len(tiles)} tiles")
    if got is None:
        log("NO ROUTE -- no reachable stance in the graph can land the target")
        return 1
    log(f"route: {len(got)} hops, {got.energy} energy left on arrival")
    for i, tile, k, before, after in graph.describe(got, st):
        print(f"  {i:2d}. {str(tile):9s} k={k}  eye {before:5.2f} -> {after:5.2f}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
