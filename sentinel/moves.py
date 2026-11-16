"""Candidate moves of many boards at once, over numpy arrays.

:func:`views` lists, per board, the absorbs, hops, mounts, the wait and the relocation
the search expands, each with the frame-cheapest lattice view of its tile; :func:`action_views`
the per-action views and :func:`keep_best` the order the search keeps children in.
"""

import numpy as np

from sentinel import actions, aimbatch, batch, landtable, memmap as mm
from sentinel.views import BOULDER_H, EYE_EPS, ROBOT_EYE, _Landings, _Views

ABSORB, HOP, MOUNT, WAIT, RELOCATE = range(5)
KINDS = ("absorb", "hop", "mount", "wait", "hyperspace")
WAIT_QUANTUM = 60  # frames one wait move idles
MAX_BOULDERS = 3
REACH = 10
ROBOT_E = mm.ENERGY_IN_OBJECTS[mm.T_ROBOT]
VALUE = np.zeros(8, dtype=np.int64)
for _otype, _energy in mm.ENERGY_IN_OBJECTS.items():
    VALUE[_otype] = _energy
_X, _Y = np.meshgrid(np.arange(mm.N), np.arange(mm.N), indexing="ij")
TILE_ADDR = ((_X & 3) + 4) * 256 + (((_X << 3) & 0xE0) | (_Y & 0x1F))
SLOTS = np.arange(mm.NUM_SLOTS)


def lexicographic(fields):
    """The search objective ``(alive, keys)``: a row lives while the player does; among
    the living the order is won, fewer enemies, fewer enemies summed over the line's moves
    (taken sooner), on the platform, unseen by any enemy, higher eye, more energy --
    the first key most significant, larger better."""
    return fields["alive"] == 1, (
        fields["won"],
        -fields["enemies"],
        -fields["area"],
        fields["home"],
        -fields["seen"],
        fields["eye"],
        fields["energy"],
    )


def best_case(st, name, tile, k=0):
    """An upper bound on :func:`lexicographic`'s keys ``(won, -enemies, home, -seen, eye,
    energy)`` once ``name`` on ``tile`` (``k`` boulders for a hop) is played from ``st``,
    or None where none is known; a drain only lowers energy, any move may end unseen."""
    me = st.player
    eye = st.obj_z_height[me] * 256 + st.obj_z_frac[me]
    foes = sum(st.obj_type[s] in mm.ENEMY_TYPES for s in st.occupied_slots())
    energy = st.energy
    home = int(tuple(st.player_xy()) == tuple(st.platform_xy))
    there = int(tile is not None and tuple(tile) == tuple(st.platform_xy))
    if name == "wait":
        return 0, -foes, home, 0, eye, energy
    if name == "relocate":
        toll = mm.ENERGY_IN_OBJECTS[mm.T_ROBOT]
        top = (st.obj_z_height[me] + 1) * 256 + 0xE0
        return None if energy < toll else (0, -foes, 1, 0, top, energy - toll)
    if name not in ("absorb", "mount", "hop"):
        return None
    stack = _stack(st, tile)
    if name == "absorb":
        if not stack:
            return None
        gain = max(mm.ENERGY_IN_OBJECTS[st.obj_type[s]] for s in stack)
        foes -= any(st.obj_type[s] in mm.ENEMY_TYPES for s in stack)
        return 0, -foes, home, 0, eye, min(mm.ENERGY_MASK, energy + gain)
    if name == "mount":
        if not stack or st.obj_type[stack[0]] != mm.T_ROBOT:
            return None
        top = stack[0]
        return (
            0,
            -foes,
            there,
            0,
            st.obj_z_height[top] * 256 + st.obj_z_frac[top],
            energy,
        )
    twin = st.clone()
    slot = None
    for otype in (mm.T_BOULDER,) * int(k) + (mm.T_ROBOT,):
        slot = actions.create(twin, otype, tile)
        if slot is None:
            return None
    return (
        0,
        -foes,
        there,
        0,
        twin.obj_z_height[slot] * 256 + twin.obj_z_frac[slot],
        twin.energy,
    )


def _stack(st, tile):
    """Slots stacked on ``tile``, top first."""
    b = st.mem[TILE_ADDR[tile]]
    out = []
    while b >= mm.OBJECT_TILE:
        out.append(b & 0x3F)
        flags = st.obj_flags[out[-1]]
        b = mm.OBJECT_TILE | (flags & 0x3F) if flags & 0x40 else 0
    return out


def keep_best(identity, keys, frames, width):
    """Positions of at most ``width`` rows to keep, best first; equal ``identity`` rows are one.

    ``identity`` is ``(n, c)`` int64, the best row standing for its entry.  ``keys`` are ``(n,)`` arrays, the first most significant,
    larger better; ties fall to fewer ``frames``, then the lower position, so every backend
    picks the same rows."""
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
    """Int64 columns ``parent, kind, tile, k, slot`` of every board's candidate moves.

    Absorbs, hops, mounts (each capped at ``branch``), a wait, and a relocation where the
    toll is in hand; ``landed`` ``(n, N, N)`` marks keyboard-reachable tiles.  With the
    Sentinel gone the platform takes any hop or mount and relocating from it is the win.
    """
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
    plat = np.zeros_like(landed)
    end = gone
    if end.any():
        pslot = board.occupied & (board.otype == mm.T_PLATFORM)
        prow, pat = np.nonzero(pslot & end[:, None])
        plat[prow, board.x[prow, pat], board.y[prow, pat]] = True
    need = np.maximum(0, np.ceil((eye + EYE_EPS - base) / BOULDER_H)).astype(np.int64)
    need = np.where(plat, 0, need)
    gain = base + BOULDER_H * need - eye
    cost = 2 * need + ROBOT_E
    fuel = np.where(end[:, None, None], 0, board.fuel())
    purse = energy[:, :, None]
    hop = (
        landed & stack & ((gain > 0) | plat) & (need <= MAX_BOULDERS) & (cost <= purse)
    )
    hop &= (purse - cost > 0) & ((purse - cost >= ROBOT_E) | (fuel > 0))
    flat = (board.n, -1)
    rank = (~plat, -(gain / cost), -fuel)
    hop_row, hop_at = _top_ranked(
        hop.reshape(flat), [r.reshape(flat) for r in rank], branch
    )

    on_plat = plat[ar, board.xs, board.ys]
    ours = (
        board.tops
        & reach
        & notme
        & (kind == mm.T_ROBOT)
        & ((board.z > board.eye[:, None] + EYE_EPS) | on_plat)
    )
    near = fuel[ar, board.xs, board.ys]
    ours &= (energy >= ROBOT_E) | (near > 0)
    mount_row, mount_at = _top_ranked(ours, (-board.z,), branch)

    own = board.rows
    leave = ~gone | plat[board.rows, at_me[0][:, 0], at_me[1][:, 0]]
    away = np.flatnonzero(leave & (board.energy >= ROBOT_E))
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


def lattice_axes(ray, grids):
    """``(h, v, cx, cy)``: the four aim axes of lattice ``ray`` of ``grids``."""
    hgrid, vgrid, cxs, cys = (np.asarray(g, dtype=np.int64) for g in grids)
    per_h = len(cxs) * len(cys)
    vi, rem = np.divmod(ray, len(hgrid) * per_h)
    hi, rem = np.divmod(rem, per_h)
    cxi, cyi = np.divmod(rem, len(cys))
    return hgrid[hi], vgrid[vi], cxs[cxi], cys[cyi]


def _landings(states, grids, aim_from):
    """:class:`_Landings` of every state on lattice ``grids``, one launch for the sweeps."""
    return [
        _Landings(st, rays, tiles, grids, (int(a[0]), int(a[1])))
        for st, (rays, tiles), a in zip(states, batch.landings(states, grids), aim_from)
    ]


def views(states, imgs, board, branch):
    """:func:`moves` with ``view``: each move tile's frame-cheapest lattice aim.

    ``view`` is (bearing, pitch, cursor x, y), -1 where a move needs no aim; moves no
    view reaches are dropped.  Both lattices sweep every board in one launch each and
    the aimed (bearing, pitch) cells are priced together over arrays."""
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
    pan, turn, _uturn = aimbatch.aim_parts(
        imgs, row, la, aim_from[row, 0], aim_from[row, 1], asks, grids
    )
    ray_of = np.full((2, n, tiles), -1, dtype=np.int64)
    for q, (r, g) in enumerate(zip(row, la)):
        land = lands[g][r]
        land.fill(asks[q], pan[q], turn[q])
        ray_of[g, r, land.tiles[ask_tids[q]]] = land.winners(ask_tids[q])

    ray = ray_of[np.maximum(lat, 0), parent, tile]
    view = np.full((len(kind), 4), -1, dtype=np.int64)
    for g in range(2):
        pick = aimed & (lat == g)
        view[pick] = np.stack(lattice_axes(ray[pick], grids[g]), 1)
    mv["view"] = view
    live = (kind == WAIT) | (kind == RELOCATE) | aimed
    return {name: col[live] for name, col in mv.items()}


def action_views(states, cols):
    """Per move of ``cols`` (``parent`` indexing ``states``), each action's ``(h, v, cx, cy)``.

    None where an action has no landing view.  A hop's later actions aim fresh at the stack
    its earlier creates leave, from the facing the previous aim left; every other move is
    its one view."""
    out = []
    for parent, kind, tile, k, view in zip(
        cols["parent"], cols["kind"], cols["tile"], cols["k"], cols["view"]
    ):
        first = tuple(map(int, view))
        if kind != HOP:
            out.append((first,))
            continue
        xy = (int(tile) // mm.N, int(tile) % mm.N)
        st, got = states[parent].clone(), [first]
        for otype in (mm.T_BOULDER,) * int(k) + (mm.T_ROBOT,):
            if actions.create(st, otype, xy) is None:
                got = None
                break
            st.obj_h_angle[st.player], st.obj_v_angle[st.player] = got[-1][:2]
            nxt = _Views(st).get(xy, band=True)
            if nxt is None:
                got = None
                break
            got.append((nxt["h_angle"], nxt["v_angle"], *map(int, nxt["cursor"])))
        out.append(None if got is None else tuple(got))
    return out
