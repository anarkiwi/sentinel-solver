"""A wide-search player: a beam over macro-actions on :class:`engine.Pool`.

Every ply expands each beam row by its legal moves (candidate generation is numpy over the
row's object and tile arrays), runs all children's enemy clocks in one ``Pool.advance``,
reads them with one ``Pool.evaluate``, drops duplicate rows through the row hash and keeps
the best ``width`` under a pluggable objective.  The moves are the phase player's own
verbs: absorb a reachable object, hop (stack boulders, cap with a robot, transfer), mount a
robot already owned, or wait one quantum; each is priced by the phase player's aim and settle
spans, batched across the beam through the fleet scheduler.
"""

import argparse
import json

import numpy as np
from tqdm import tqdm

from sentinel import (
    actions,
    engine,
    fleet,
    memmap as mm,
    projector,
    rows_jit,
)
from sentinel.game import Game
from sentinel.phase_player import WAIT_QUANTUM, PhasePlayer
from sentinel.playerbase import (
    BOULDER_H,
    EYE_EPS,
    ROBOT_EYE,
    SIGHTS_CENTRE,
    _Views,
)

ABSORB, HOP, MOUNT, WAIT = range(4)
KINDS = ("absorb", "hop", "mount", "wait")
MAX_BOULDERS = 3
OP_LEN = 3 + MAX_BOULDERS
REACH = 10
ROBOT_E = mm.ENERGY_IN_OBJECTS[mm.T_ROBOT]
VALUE = np.zeros(8, dtype=np.int64)
for _otype, _energy in mm.ENERGY_IN_OBJECTS.items():
    VALUE[_otype] = _energy
_X, _Y = np.meshgrid(np.arange(mm.N), np.arange(mm.N), indexing="ij")
TILE_ADDR = ((_X & 3) + 4) * 256 + (((_X << 3) & 0xE0) | (_Y & 0x1F))
SLOTS = np.arange(mm.NUM_SLOTS)
NO_BEARING = -1


def lexicographic(fields):
    """The default objective ``(alive, keys)``: a row lives while the player does; among
    the living the order is the phase player's score on facts -- won, fewer enemies, higher
    eye, more energy -- the first key most significant, larger better."""
    return fields["alive"] == 1, (
        fields["won"],
        -fields["enemies"],
        fields["eye"],
        fields["energy"],
    )


def keep_best(identity, keys, frames, width):
    """Positions of at most ``width`` rows to keep, best first.

    ``identity`` is ``(n, c)`` int64: rows with equal identity are one entry and the best
    of them stands for it.  ``keys`` are ``(n,)`` arrays, the first most significant,
    larger better; ties fall to fewer ``frames``, then to the lower position, so the order
    is total and every backend picks the same rows."""
    n = len(frames)
    cols = [np.arange(n), np.asarray(frames)] + [-np.asarray(k) for k in keys[::-1]]
    order = np.lexsort(cols)
    if not n:
        return order
    _, first = np.unique(identity[order], axis=0, return_index=True)
    return order[np.sort(first)][:width]


class Board:
    """The decision-relevant arrays of one state, as numpy columns over its slots."""

    def __init__(self, st):
        mem = np.frombuffer(st.mem, dtype=np.uint8)

        def col(base):
            return mem[base : base + mm.NUM_SLOTS].astype(np.int64)

        self.flags, self.otype = col(mm.OBJECTS_FLAGS), col(mm.OBJECTS_TYPE)
        self.x, self.y = col(mm.OBJECTS_X), col(mm.OBJECTS_Y)
        self.z = col(mm.OBJECTS_Z_HEIGHT) + col(mm.OBJECTS_Z_FRACTION) / 256.0
        self.tile = mem[TILE_ADDR].astype(np.int64)
        self.player = int(mem[mm.PLAYER_OBJECT])
        self.energy = int(mem[mm.PLAYER_ENERGY])
        self.occupied = (self.flags & 0x80) == 0
        xs, ys = np.where(self.occupied, self.x, 0), np.where(self.occupied, self.y, 0)
        at = self.tile[xs, ys]
        self.tops = self.occupied & (at >= mm.OBJECT_TILE) & ((at & 0x3F) == SLOTS)
        self.eye = self.z[self.player]

    def fuel(self):
        """``(N, N)`` int: absorbable objects within :data:`REACH` of each tile in both axes
        (what a landing there can live on): each stamps its own box."""
        take = self.occupied & (SLOTS != self.player)
        take &= ~np.isin(self.otype, mm.ENEMY_TYPES + (mm.T_PLATFORM,))
        grid = np.zeros((mm.N, mm.N), dtype=np.int64)
        np.add.at(grid, (self.x[take], self.y[take]), 1)
        total = np.zeros((mm.N + 1, mm.N + 1), dtype=np.int64)
        total[1:, 1:] = grid.cumsum(0).cumsum(1)
        idx = np.arange(mm.N)
        lo, hi = np.clip(idx - REACH, 0, mm.N), np.clip(idx + REACH + 1, 0, mm.N)
        return total[hi][:, hi] - total[lo][:, hi] - total[hi][:, lo] + total[lo][:, lo]


def _top_ranked(mask, keys, branch):
    """The first ``branch`` indices of ``mask`` under the lexicographic order of ``keys``
    (the first most significant, smaller first), the lower index first among equals."""
    at = np.flatnonzero(mask)
    order = np.lexsort([at] + [np.asarray(k)[at] for k in keys[::-1]])
    return at[order][:branch]


def moves(board, landed, branch):
    """``(kind, tile, k, slot)`` int64 arrays of a board's candidate moves: absorbs, hops,
    mounts and one wait, each class capped at ``branch`` and gated as the phase player gates
    it.  ``landed`` masks the tiles a keyboard view reaches; ``tile`` is packed x * N + y.
    """
    me, energy = board.player, board.energy
    reach = landed[
        np.where(board.occupied, board.x, 0), np.where(board.occupied, board.y, 0)
    ]
    notme = (SLOTS != me) & ~((board.x == board.x[me]) & (board.y == board.y[me]))
    foe = board.occupied & np.isin(board.otype, mm.ENEMY_TYPES)
    sentinel_ok = (foe.sum() == 1) & (
        energy + mm.ENERGY_IN_OBJECTS[mm.T_SENTINEL] >= 2 * ROBOT_E
    )
    low = board.z <= board.eye + EYE_EPS
    kind = board.otype
    spent = (
        (kind == mm.T_TREE)
        | (((kind == mm.T_ROBOT) | (kind == mm.T_BOULDER)) & low)
        | (kind == mm.T_SENTRY)
        | ((kind == mm.T_SENTINEL) & sentinel_ok)
    )
    gone = bool(board.flags[actions.SENTINEL_SLOT] & 0x80)
    eat = (
        board.tops & reach & notme & spent & (not gone) & (energy < mm.ENERGY_MASK - 4)
    )
    pick_eat = _top_ranked(eat, (~foe, -VALUE[kind]), branch)

    covered = board.tile >= mm.OBJECT_TILE
    under = board.tile & 0x3F
    utype = board.otype[under]
    base = np.where(
        covered,
        board.z[under] + np.where(utype == mm.T_PLATFORM, 1.0, BOULDER_H),
        (board.tile >> 4) + ROBOT_EYE,
    )
    stack = ~covered | (utype == mm.T_BOULDER) | (utype == mm.T_PLATFORM)
    need = np.maximum(0, np.ceil((board.eye + EYE_EPS - base) / BOULDER_H)).astype(
        np.int64
    )
    gain = base + BOULDER_H * need - board.eye
    cost = 2 * need + ROBOT_E
    fuel = board.fuel()
    hop = landed & stack & (gain > 0) & (need <= MAX_BOULDERS) & (cost <= energy)
    hop &= (energy - cost > 0) & ((energy - cost >= ROBOT_E) | (fuel > 0))
    flat = hop.ravel()
    pick_hop = _top_ranked(flat, (-(gain / cost).ravel(), -fuel.ravel()), branch)

    ours = (
        board.tops
        & reach
        & notme
        & (kind == mm.T_ROBOT)
        & (board.z > board.eye + EYE_EPS)
    )
    near = fuel[
        np.where(board.occupied, board.x, 0), np.where(board.occupied, board.y, 0)
    ]
    ours &= (energy >= ROBOT_E) | (near > 0)
    pick_mount = _top_ranked(ours, (-board.z,), branch)

    here = board.x[me] * mm.N + board.y[me]
    tiles = np.concatenate(
        [
            board.x[pick_eat] * mm.N + board.y[pick_eat],
            pick_hop,
            board.x[pick_mount] * mm.N + board.y[pick_mount],
            [here],
        ]
    ).astype(np.int64)
    kinds = np.repeat(
        [ABSORB, HOP, MOUNT, WAIT],
        [len(pick_eat), len(pick_hop), len(pick_mount), 1],
    )
    ks = np.concatenate(
        [
            np.zeros(len(pick_eat), dtype=np.int64),
            need.ravel()[pick_hop],
            np.zeros(len(pick_mount) + 1, dtype=np.int64),
        ]
    )
    slots = np.concatenate(
        [pick_eat, np.full(len(pick_hop), -1), pick_mount, [-1]]
    ).astype(np.int64)
    return kinds.astype(np.int64), tiles, ks, slots


def build_ops(kind, tile, k, slot, view):
    """``(n, OP_LEN, 3)`` engine ops of ``n`` moves: face the aim, then absorb ``slot``, or
    stack ``k`` boulders on ``tile``, cap with a robot and transfer into it, or transfer
    into ``slot``; a wait is all padding.  ``view`` is ``(n, 2)`` bearing and pitch."""
    n = len(kind)
    ops = np.tile(engine.NOOP_ROW, (n, OP_LEN, 1))
    aim = (kind == ABSORB) | (kind == HOP)
    ops[aim, 0] = np.stack(
        [np.full(aim.sum(), rows_jit.FACE), view[aim, 0], view[aim, 1]], 1
    )
    ab = kind == ABSORB
    ops[ab, 1, 0], ops[ab, 1, 1], ops[ab, 1, 2] = rows_jit.ABSORB, slot[ab], 0
    mt = kind == MOUNT
    ops[mt, 0, 0], ops[mt, 0, 1], ops[mt, 0, 2] = rows_jit.TRANSFER, slot[mt], 0
    hop = np.flatnonzero(kind == HOP)
    for j in range(MAX_BOULDERS):
        rows = hop[k[hop] > j]
        ops[rows, 1 + j] = np.stack(
            [
                np.full(len(rows), rows_jit.CREATE),
                np.full(len(rows), mm.T_BOULDER),
                tile[rows],
            ],
            1,
        )
    ops[hop, 1 + k[hop]] = np.stack(
        [np.full(len(hop), rows_jit.CREATE), np.full(len(hop), mm.T_ROBOT), tile[hop]],
        1,
    )
    ops[hop, 2 + k[hop], 0] = rows_jit.TRANSFER
    ops[hop, 2 + k[hop], 1] = rows_jit.LAST
    ops[hop, 2 + k[hop], 2] = 0
    return ops


VERBS = {ABSORB: ("absorb",), MOUNT: ("transfer",)}


def _verbs(kind, k):
    if kind == HOP:
        return (("boulder",) if k else ()) + ("robot", "transfer")
    return VERBS.get(kind, ())


def _span_of(kind, k, tile, span):
    """Frames of a move from the priced ``span[(tile, verb)]``."""
    if kind == HOP:
        return (
            k * span.get((tile, "boulder"), 0.0)
            + span[(tile, "robot")]
            + span[(tile, "transfer")]
        )
    if kind == WAIT:
        return float(WAIT_QUANTUM)
    return span[(tile, _verbs(kind, k)[0])]


def plan(player, branch):
    """The priced moves of ``player``'s board, as a dict of columns: ``kind``, ``tile``,
    ``k``, ``slot``, ``span`` (frames, int64) and ``view`` (bearing, pitch, cursor x, y)
    with -1 where a move has no aim.  Runs as a fleet task, so its landing sweeps and
    span pricings are batched with the other rows'."""
    st = player.st
    views = _Views(st)
    kind, tile, k, slot = moves(Board(st), views.landed(), branch)
    xy = [(int(p) // mm.N, int(p) % mm.N) for p in tile]
    aimed = np.flatnonzero(kind != WAIT)
    seen, calls, found = {}, [], {}
    for i in aimed:
        t = xy[i]
        if t not in found:
            found[t] = views.get(t, band=True)
        if found[t] is None:
            continue
        for verb in _verbs(kind[i], k[i]):
            if (int(tile[i]), verb) not in seen:
                seen[(int(tile[i]), verb)] = (t, verb)
                calls += player._span_calls(  # pylint: disable=protected-access
                    t, (verb,), found[t]
                )
    projector.prefetch(calls)
    span = {
        key: player._span(verb, found[t], t)  # pylint: disable=protected-access
        for key, (t, verb) in seen.items()
    }
    view = np.full((len(kind), 4), -1, dtype=np.int64)
    frames = np.zeros(len(kind), dtype=np.int64)
    live = np.ones(len(kind), dtype=bool)
    for i, t in enumerate(xy):
        if kind[i] != WAIT and found.get(t) is None:
            live[i] = False
            continue
        if kind[i] != WAIT:
            v = found[t]
            view[i] = (v["h_angle"], v["v_angle"], *v["cursor"])
        frames[i] = int(round(_span_of(kind[i], int(k[i]), int(tile[i]), span)))
    return {
        "kind": kind[live],
        "tile": tile[live],
        "k": k[live],
        "slot": slot[live],
        "span": frames[live],
        "view": view[live],
    }


class BeamPlayer:
    """Beam search over :func:`moves` on one pool; ``run`` plays the best line found."""

    def __init__(
        self, game, width=16, depth=60, branch=3, objective=lexicographic, verbose=False
    ):
        self.g = game
        self.width, self.depth, self.branch = width, depth, branch
        self.objective = objective
        self.verbose = verbose
        self.trace = []
        self.frames = 0

    def _players(self, states, cursor, bearing, frames):
        out = []
        for st, cur, bear, fr in zip(states, cursor, bearing, frames):
            player = PhasePlayer(Game(st), verbose=False, rollout=True)
            player.cursor = [int(c) for c in cur]
            player.last_bearing = (
                None if bear[0] == NO_BEARING else tuple(map(int, bear))
            )
            player.frames = int(fr)
            out.append(player)
        return out

    @staticmethod
    def _finishable(st):
        """Standing on the platform with the Sentinel gone and the toll in hand."""
        return (
            actions.on_platform(st)
            and bool(st.obj_flags[actions.SENTINEL_SLOT] & 0x80)
            and st.energy >= ROBOT_E
            and not actions.player_dead(st)
        )

    def run(self, max_actions=None):
        """Search up to ``max_actions`` plies (default: the depth); True when the best line
        wins.  ``trace`` is the line's macro-actions, ``self.g.state`` its final state.
        """
        depth = self.depth if max_actions is None else max_actions
        cap = self.width * (3 * self.branch + 1)
        pool = engine.Pool.from_states(
            [self.g.state], device=engine.default_device(cap)
        )
        beam = {
            "row": np.zeros(1, dtype=np.int64),
            "cursor": np.array([SIGHTS_CENTRE], dtype=np.int64),
            "bearing": np.full((1, 2), NO_BEARING, dtype=np.int64),
            "frames": np.zeros(1, dtype=np.int64),
        }
        history = []
        final = None
        with tqdm(range(depth + 1), unit="ply", desc="beam") as progress:
            for ply in progress:
                states = pool.states(beam["row"])
                done = [self._finishable(st) for st in states]
                if any(done) or ply == depth:
                    final = int(np.flatnonzero(done)[0]) if any(done) else 0
                    break
                players = self._players(
                    states, beam["cursor"], beam["bearing"], beam["frames"]
                )
                parts = fleet.run([lambda p=p: plan(p, self.branch) for p in players])
                bad = [p for p in parts if isinstance(p, fleet.Failure)]
                if bad:
                    raise RuntimeError(f"a beam row failed: {bad[0].error!r}")
                nxt = self._ply(pool, beam, parts)
                if nxt is None:
                    final = 0
                    break
                beam, record = nxt
                history.append(record)
                progress.set_postfix(rows=len(pool), best=int(record["eye"][0]))
        return self._conclude(pool, beam, history, final)

    def _ply(self, pool, beam, parts):
        """``(next beam, record)`` after one ply, or None when no child survives."""
        count = [len(p["kind"]) for p in parts]
        cols = {k: np.concatenate([p[k] for p in parts]) for k in parts[0]}
        parent = np.repeat(np.arange(len(parts)), count)
        ops = build_ops(
            cols["kind"], cols["tile"], cols["k"], cols["slot"], cols["view"][:, :2]
        )
        kids, built = pool.expand(beam["row"][parent], ops)
        run = np.flatnonzero(built)
        pool.advance(kids[run], cols["span"][run])
        fields = pool.evaluate(kids[run])
        alive, keys = self.objective(fields)
        move = cols["kind"][run]
        cursor = np.where(
            (move == WAIT)[:, None], beam["cursor"][parent[run]], cols["view"][run, 2:]
        )
        bearing = np.where(
            (move == WAIT)[:, None],
            beam["bearing"][parent[run]],
            np.where((move == ABSORB)[:, None], cols["view"][run, :2], NO_BEARING),
        )
        frames = beam["frames"][parent[run]] + cols["span"][run]
        identity = np.column_stack([pool.keys(kids[run]), cursor, bearing])
        live = np.flatnonzero(alive)
        if not live.size:
            return None
        pick = live[
            keep_best(
                identity[live],
                [np.asarray(k)[live] for k in keys],
                frames[live],
                self.width,
            )
        ]
        at = run[pick]
        nxt = {
            "row": kids[at],
            "cursor": cursor[pick],
            "bearing": bearing[pick],
            "frames": frames[pick],
        }
        record = {
            "parent": parent[at],
            "kind": cols["kind"][at],
            "tile": cols["tile"][at],
            "k": cols["k"][at],
            "frames": frames[pick],
            "energy": fields["energy"][pick],
            "eye": fields["eye"][pick],
        }
        return nxt, record

    def _conclude(self, pool, beam, history, final):
        """Trace the line ending at beam row ``final``, play the hyperspace if it can, and
        leave the final state in the game."""
        states = pool.states(beam["row"])
        st, at = states[final], final
        line = []
        for record in reversed(history):
            line.append((record, at))
            at = int(record["parent"][at])
        trace = []
        for record, at in reversed(line):
            tile = int(record["tile"][at])
            trace.append(
                [
                    int(record["frames"][at]),
                    KINDS[int(record["kind"][at])],
                    [tile // mm.N, tile % mm.N],
                    int(record["energy"][at]),
                    round(int(record["eye"][at]) / 256, 3),
                ]
            )
        self.frames = int(beam["frames"][final])
        won = False
        if self._finishable(st):
            actions.hyperspace(st)
            won = actions.won(st)
            trace.append(
                [
                    self.frames,
                    "hyperspace",
                    list(map(int, st.player_xy())),
                    st.energy,
                    round(st.eye_z(), 3),
                ]
            )
        self.trace = trace
        self.g.state = st
        return won


def main(argv=None):
    """Play one landscape offline with the beam player."""
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("landscape", type=int)
    parser.add_argument("--width", type=int, default=16)
    parser.add_argument("--depth", type=int, default=60)
    parser.add_argument("--branch", type=int, default=3)
    args = parser.parse_args(argv)
    game = Game.typed(args.landscape)
    player = BeamPlayer(game, args.width, args.depth, args.branch)
    won = player.run()
    print(json.dumps(player.trace))
    print(
        f"landscape {args.landscape}: {'WON' if won else 'lost'} in "
        f"{len(player.trace)} actions / {player.frames} frames, energy {game.energy}"
    )
    return 0 if won else 1


if __name__ == "__main__":
    raise SystemExit(main())
