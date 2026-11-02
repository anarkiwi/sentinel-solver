"""A wide-search player: a beam over macro-actions on :class:`engine.Pool`.

Every ply expands each beam row by its legal moves (candidate generation is numpy over the
row's object and tile arrays), runs all children's enemy clocks in one ``Pool.advance``,
reads them with one ``Pool.evaluate``, drops duplicate rows through the row hash and keeps
the best ``width`` under a pluggable objective.  The moves are the phase player's own
verbs: absorb a reachable object, hop (stack boulders, cap with a robot, transfer), mount a
robot already owned, or wait one quantum; each is priced by the phase player's aim and settle
spans, the aims over arrays for the whole beam at once.
"""

import argparse
import json
from functools import partial

import numpy as np
from tqdm import tqdm

from sentinel import (
    actions,
    aimbatch,
    batch,
    engine,
    landtable,
    memmap as mm,
    projector,
    rows_jit,
    terrain,
)
from sentinel.game import Game
from sentinel.phase_player import WAIT_QUANTUM
from sentinel.playerbase import (
    BOULDER_H,
    EYE_EPS,
    ROBOT_EYE,
    SIGHTS_CENTRE,
    TOGGLE_FRAMES,
    _Landings,
    _view_at,
    settle_frames,
)

ABSORB, HOP, MOUNT, WAIT, RELOCATE = range(5)
KINDS = ("absorb", "hop", "mount", "wait", "hyperspace")
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
    """The decision-relevant arrays of ``n`` states, as ``(n, slots)`` numpy columns.

    ``read(addrs)`` gives the ``(n, len(addrs))`` uint8 bytes at image ``addrs`` of every
    state: a pool's mutable region, or the images themselves."""

    def __init__(self, read):
        def col(base):
            return read(base + SLOTS).astype(np.int64)

        self.flags, self.otype = col(mm.OBJECTS_FLAGS), col(mm.OBJECTS_TYPE)
        self.x, self.y = col(mm.OBJECTS_X), col(mm.OBJECTS_Y)
        self.h, self.v = col(mm.OBJECTS_H_ANGLE), col(mm.OBJECTS_V_ANGLE)
        self.z = col(mm.OBJECTS_Z_HEIGHT) + col(mm.OBJECTS_Z_FRACTION) / 256.0
        self.n = len(self.flags)
        self.rows = np.arange(self.n)
        shape = (self.n, mm.N, mm.N)
        self.tile = read(TILE_ADDR.ravel()).astype(np.int64).reshape(shape)
        self.player = read([mm.PLAYER_OBJECT])[:, 0].astype(np.int64)
        self.energy = read([mm.PLAYER_ENERGY])[:, 0].astype(np.int64)
        self.occupied = (self.flags & 0x80) == 0
        self.xs, self.ys = (np.where(self.occupied, a, 0) for a in (self.x, self.y))
        at = self.tile[self.rows[:, None], self.xs, self.ys]
        self.tops = self.occupied & (at >= mm.OBJECT_TILE) & ((at & 0x3F) == SLOTS)
        self.eye = self.z[self.rows, self.player]

    @classmethod
    def of_states(cls, states):
        """The board of each of ``states``."""
        mems = np.stack([np.frombuffer(st.mem, dtype=np.uint8) for st in states])
        return cls(lambda addrs: mems[:, addrs])

    def aim_from(self):
        """``(n, 2)`` bearing and pitch each player faces."""
        return np.stack([a[self.rows, self.player] for a in (self.h, self.v)], 1)

    def fuel(self):
        """``(n, N, N)`` int: absorbable objects within :data:`REACH` of each tile in both
        axes (what a landing there can live on): each stamps its own box."""
        take = self.occupied & (SLOTS != self.player[:, None])
        take &= ~np.isin(self.otype, mm.ENEMY_TYPES + (mm.T_PLATFORM,))
        grid = np.zeros((self.n, mm.N, mm.N), dtype=np.int64)
        row = np.broadcast_to(self.rows[:, None], take.shape)
        np.add.at(grid, (row[take], self.x[take], self.y[take]), 1)
        total = np.zeros((self.n, mm.N + 1, mm.N + 1), dtype=np.int64)
        total[:, 1:, 1:] = grid.cumsum(1).cumsum(2)
        idx = np.arange(mm.N)
        lo, hi = np.clip(idx - REACH, 0, mm.N), np.clip(idx + REACH + 1, 0, mm.N)
        return (
            total[:, hi][:, :, hi]
            - total[:, lo][:, :, hi]
            - total[:, hi][:, :, lo]
            + total[:, lo][:, :, lo]
        )


def _top_ranked(mask, keys, branch):
    """``(row, index)`` of the first ``branch`` true entries of each row of ``mask`` under
    the lexicographic order of ``keys`` (the first most significant, smaller first), the
    lower index first among equals; rows ascending."""
    row, at = np.nonzero(mask)
    order = np.lexsort([at] + [np.asarray(k)[row, at] for k in keys[::-1]] + [row])
    row, at = row[order], at[order]
    keep = np.arange(row.size) - np.searchsorted(row, row) < branch
    return row[keep], at[keep]


def moves(board, landed, branch):
    """The candidate moves of every board, as int64 columns ``parent``, ``kind``, ``tile``,
    ``k`` and ``slot``: absorbs, hops, mounts, one wait per board and, where the board is
    barren, one relocation, each class capped at ``branch`` and gated as the phase player
    gates it, a board's moves together and in that order.  A board is barren when no
    absorb, hop or mount reaches it; the relocation also needs the toll in hand and the
    Sentinel still standing (:meth:`PhasePlayer._relocate`).  ``landed`` is ``(n, N, N)``: the tiles a keyboard view reaches; ``tile``
    is packed x * N + y."""
    me, ar = board.player[:, None], board.rows[:, None]
    energy = board.energy[:, None]
    reach = landed[ar, board.xs, board.ys]
    at_me = [np.take_along_axis(a, me, 1) for a in (board.x, board.y)]
    notme = (SLOTS != me) & ~((board.x == at_me[0]) & (board.y == at_me[1]))
    foe = board.occupied & np.isin(board.otype, mm.ENEMY_TYPES)
    sentinel_ok = (foe.sum(1) == 1) & (
        board.energy + mm.ENERGY_IN_OBJECTS[mm.T_SENTINEL] >= 2 * ROBOT_E
    )
    low = board.z <= board.eye[:, None] + EYE_EPS
    kind = board.otype
    spent = (
        (kind == mm.T_TREE)
        | (((kind == mm.T_ROBOT) | (kind == mm.T_BOULDER)) & low)
        | (kind == mm.T_SENTRY)
        | ((kind == mm.T_SENTINEL) & sentinel_ok[:, None])
    )
    gone = (board.flags[:, actions.SENTINEL_SLOT] & 0x80) != 0
    eat = board.tops & reach & notme & spent & ~gone[:, None]
    eat &= energy < mm.ENERGY_MASK - 4
    eat_row, eat_at = _top_ranked(eat, (~foe, -VALUE[kind]), branch)

    covered = board.tile >= mm.OBJECT_TILE
    under = (board.tile & 0x3F).reshape(board.n, -1)
    utype = np.take_along_axis(board.otype, under, 1).reshape(board.tile.shape)
    uz = np.take_along_axis(board.z, under, 1).reshape(board.tile.shape)
    base = np.where(
        covered,
        uz + np.where(utype == mm.T_PLATFORM, 1.0, BOULDER_H),
        (board.tile >> 4) + ROBOT_EYE,
    )
    stack = ~covered | (utype == mm.T_BOULDER) | (utype == mm.T_PLATFORM)
    eye = board.eye[:, None, None]
    need = np.maximum(0, np.ceil((eye + EYE_EPS - base) / BOULDER_H)).astype(np.int64)
    gain = base + BOULDER_H * need - eye
    cost = 2 * need + ROBOT_E
    fuel = board.fuel()
    purse = energy[:, :, None]
    hop = landed & stack & (gain > 0) & (need <= MAX_BOULDERS) & (cost <= purse)
    hop &= (purse - cost > 0) & ((purse - cost >= ROBOT_E) | (fuel > 0))
    flat = (board.n, -1)
    hop_row, hop_at = _top_ranked(
        hop.reshape(flat), (-(gain / cost).reshape(flat), -fuel.reshape(flat)), branch
    )

    ours = (
        board.tops
        & reach
        & notme
        & (kind == mm.T_ROBOT)
        & (board.z > board.eye[:, None] + EYE_EPS)
    )
    near = fuel[ar, board.xs, board.ys]
    ours &= (energy >= ROBOT_E) | (near > 0)
    mount_row, mount_at = _top_ranked(ours, (-board.z,), branch)

    own = board.rows
    barren = ~(eat.any(1) | hop.reshape(flat).any(1) | ours.any(1))
    away = np.flatnonzero(barren & ~gone & (board.energy >= ROBOT_E))
    sizes = [eat_row.size, hop_row.size, mount_row.size, own.size, away.size]
    parent = np.concatenate([eat_row, hop_row, mount_row, own, away])
    cls = np.repeat([ABSORB, HOP, MOUNT, WAIT, RELOCATE], sizes)
    pack = lambda row, at: board.x[row, at] * mm.N + board.y[row, at]
    tile = np.concatenate(
        [
            pack(eat_row, eat_at),
            hop_at,
            pack(mount_row, mount_at),
            pack(own, board.player),
            pack(away, board.player[away]),
        ]
    )
    k = np.concatenate(
        [
            np.zeros(eat_row.size, dtype=np.int64),
            need.reshape(flat)[hop_row, hop_at],
            np.zeros(mount_row.size + own.size + away.size, dtype=np.int64),
        ]
    )
    slot = np.concatenate(
        [
            eat_at,
            np.full(hop_row.size, -1),
            mount_at,
            np.full(own.size + away.size, -1),
        ]
    )
    order = np.lexsort([cls, parent])
    return {
        "parent": parent[order].astype(np.int64),
        "kind": cls[order].astype(np.int64),
        "tile": tile[order].astype(np.int64),
        "k": k[order].astype(np.int64),
        "slot": slot[order].astype(np.int64),
    }


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


def _span_of(kind, k, span):
    """Frames of a move from the priced ``span[verb]`` of its tile."""
    if kind == HOP:
        return k * span.get("boulder", 0.0) + span["robot"] + span["transfer"]
    if kind == WAIT:
        return float(WAIT_QUANTUM)
    return span[_verbs(kind, k)[0]]


def lattice_axes(ray, grids):
    """``(h, v, cx, cy, cell)`` of lattice ``ray`` of ``grids``: the aim's four axes and
    its (bearing, pitch) cell index."""
    hgrid, vgrid, cxs, cys = (np.asarray(g, dtype=np.int64) for g in grids)
    per_h = len(cxs) * len(cys)
    vi, rem = np.divmod(ray, len(hgrid) * per_h)
    hi, rem = np.divmod(rem, per_h)
    cxi, cyi = np.divmod(rem, len(cys))
    return hgrid[hi], vgrid[vi], cxs[cxi], cys[cyi], vi * len(hgrid) + hi


def _landings(states, grids, aim_from):
    """:class:`_Landings` of every state on lattice ``grids``, one launch for the sweeps."""
    return [
        _Landings(st, rays, tiles, grids, (int(a[0]), int(a[1])))
        for st, (rays, tiles), a in zip(states, batch.landings(states, grids), aim_from)
    ]


def plan(states, imgs, board, beam, branch):
    """The priced moves of every board as a dict of columns: :func:`moves`' ``parent``,
    ``kind``, ``tile``, ``k`` and ``slot`` for the moves some view reaches, with ``span``
    (frames, int64) and ``view`` (bearing, pitch, cursor x, y; -1 where a move has no aim).

    Both lattices are swept for every board in one launch each, the (bearing, pitch) cells
    of the aimed tiles are priced together and their aims summed over arrays; only the
    settles run on the scalar models, sharing one pricing pass."""
    n, tiles = len(states), mm.N * mm.N
    aim_from = board.aim_from()
    grids = [landtable.lattice(v_primary=p) for p in (True, False)]
    lands = [_landings(states, g, aim_from) for g in grids]
    tid_of = np.full((2, n, tiles), -1, dtype=np.int64)
    for lat in range(2):
        for r, land in enumerate(lands[lat]):
            tid_of[lat, r, land.tiles] = np.arange(land.tiles.size)
    reach = (tid_of >= 0).any(axis=0).reshape(n, mm.N, mm.N)
    mv = moves(board, reach, branch)
    parent, kind, tile = mv["parent"], mv["kind"], mv["tile"]
    in_primary, in_band = (tid_of[g, parent, tile] >= 0 for g in range(2))
    lat = np.where(in_primary, 0, np.where(in_band, 1, -1))
    aimed = (kind != WAIT) & (lat >= 0)
    tid = tid_of[np.maximum(lat, 0), parent, tile]
    want = np.unique(np.stack([parent, lat, tid])[:, aimed], axis=1)
    pair, start = np.unique(want[0] * 2 + want[1], return_index=True)
    ends = np.append(start[1:], want.shape[1])
    ask_tids = [want[2, a:b] for a, b in zip(start, ends)]
    row, la = pair // 2, pair % 2
    asks = [
        np.unique(lands[g][r].cell[np.isin(lands[g][r].tid, t)])
        for r, g, t in zip(row, la, ask_tids)
    ]
    pan, turn, uturn = aimbatch.aim_parts(
        imgs, row, la, aim_from[row, 0], aim_from[row, 1], asks, grids
    )
    ray_of = np.full((2, n, tiles), -1, dtype=np.int64)
    pan_of = np.zeros((2, n, max(len(g[0]) * len(g[1]) for g in grids)))
    ut_of = np.zeros(n)
    for q, (r, g) in enumerate(zip(row, la)):
        land = lands[g][r]
        land.fill(asks[q], pan[q], turn[q])
        ray_of[g, r, land.tiles[ask_tids[q]]] = land.winners(ask_tids[q])
        pan_of[g, r, asks[q]] = pan[q]
        ut_of[r] = uturn[q]

    ray = ray_of[np.maximum(lat, 0), parent, tile]
    view = np.full((len(kind), 4), -1, dtype=np.int64)
    cell = np.zeros(len(kind), dtype=np.int64)
    for g in range(2):
        pick = aimed & (lat == g)
        *axes, cell[pick] = lattice_axes(ray[pick], grids[g])
        view[pick] = np.stack(axes, 1)
    bear = beam["bearing"][parent]
    reuse = (bear[:, 0] == view[:, 0]) & (bear[:, 1] == view[:, 1])
    cursor = np.where(reuse[:, None], beam["cursor"][parent], SIGHTS_CENTRE)
    drive = np.maximum(
        np.abs(view[:, 2] - cursor[:, 0]), np.abs(view[:, 3] - cursor[:, 1])
    )
    turns = aimbatch.press_plan(aim_from[parent, 0], view[:, 0])[0]
    aim = aimbatch.aim_total_columns(
        np.where(reuse, 0.0, TOGGLE_FRAMES),
        np.where(turns > 0, ut_of[parent], 0.0),
        pan_of[np.maximum(lat, 0), parent, cell],
        drive,
    )

    verbs = {}
    for i in np.flatnonzero(aimed):
        for verb in _verbs(kind[i], int(mv["k"][i])):
            verbs.setdefault((int(parent[i]), int(tile[i]), verb), i)
    calls = []
    for (r, t, verb), i in verbs.items():
        xy = (t // mm.N, t % mm.N)
        eye = terrain.top_object(states[r], *xy) if verb == "transfer" else None
        view_i = _view_at(ray[i], grids[lat[i]])
        calls.append(partial(settle_frames, states[r], verb, view_i, eye, tile=xy))
    projector.prefetch(calls)
    span = {}
    for (key, i), call in zip(verbs.items(), calls):
        same = key[2] == "transfer" and reuse[i]
        span[key] = (0.0 if same else aim[i]) + call()
    frames = np.zeros(len(kind), dtype=np.int64)
    live = (kind == WAIT) | (kind == RELOCATE) | aimed
    for i in np.flatnonzero(live & (kind != RELOCATE)):
        k = int(mv["k"][i])
        per = {v: span[(int(parent[i]), int(tile[i]), v)] for v in _verbs(kind[i], k)}
        frames[i] = int(round(float(_span_of(kind[i], k, per))))
    mv["span"], mv["view"] = frames, view
    return {name: col[live] for name, col in mv.items()}


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
                mut = pool.mutable(beam["row"])
                imgs = pool.overlay(mut)
                states = pool.as_states(imgs, pool.clocks(beam["row"]))
                done = [self._finishable(st) for st in states]
                if any(done) or ply == depth:
                    final = int(np.flatnonzero(done)[0]) if any(done) else 0
                    break
                board = Board(lambda addrs, m=mut: pool.read(m, addrs))
                cols = plan(states, imgs, board, beam, self.branch)
                nxt = self._ply(pool, beam, cols)
                if nxt is None:
                    final = 0
                    break
                beam, record = nxt
                history.append(record)
                progress.set_postfix(rows=len(pool), best=int(record["eye"][0]))
        return self._conclude(pool, beam, history, final)

    def _children(self, pool, rows, kind, ops):
        """``(children, ok, tile)`` of pool ``rows`` under ``ops``: the engine builds every
        move but a relocation, whose hyperspace is PRNG-driven and runs on the host state;
        ``tile`` is the packed player tile of each relocation child, 0 elsewhere."""
        away = kind == RELOCATE
        kids = np.zeros(len(rows), dtype=np.int64)
        built = np.ones(len(rows), dtype=bool)
        tile = np.zeros(len(rows), dtype=np.int64)
        if (~away).any():
            kids[~away], built[~away] = pool.expand(rows[~away], ops[~away])
        if away.any():
            states = pool.states(rows[away])
            for st in states:
                actions.hyperspace(st)
            kids[away] = pool.add(states)
            tile[away] = [x * mm.N + y for x, y in (st.player_xy() for st in states)]
        return kids, built, tile

    def _ply(self, pool, beam, cols):
        """``(next beam, record)`` after one ply, or None when no child survives."""
        parent = cols["parent"]
        ops = build_ops(
            cols["kind"], cols["tile"], cols["k"], cols["slot"], cols["view"][:, :2]
        )
        kids, built, landed = self._children(
            pool, beam["row"][parent], cols["kind"], ops
        )
        run = np.flatnonzero(built)
        pool.advance(kids[run], cols["span"][run])
        fields = pool.evaluate(kids[run])
        alive, keys = self.objective(fields)
        move = cols["kind"][run]
        stay = ((move == WAIT) | (move == RELOCATE))[:, None]
        cursor = np.where(stay, beam["cursor"][parent[run]], cols["view"][run, 2:])
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
        tile = np.where(cols["kind"] == RELOCATE, landed, cols["tile"])
        record = {
            "parent": parent[at],
            "kind": cols["kind"][at],
            "tile": tile[at],
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
