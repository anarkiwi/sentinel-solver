"""WHICH enemy next: a subset DP over the goals, on the stance graph's own distances.

The ordering is the part the A\\* player pays search depth for and never solves -- 7
enemies is 2^7 x 7 = 896 DP cells, and the leg costs come free from the graph's backward
hop tables, so the whole decision is milliseconds rather than a 120-deep search.
"""

import numpy as np

from sentinel import actions, enemies, memmap as mm

UNREACHABLE = 1 << 20  # leg cost standing in for "no stance bridges these two goals"


def goal_tiles(state):
    """``(sentries, sentinel, platform)`` tiles -- the goals, in absorb-lock order.

    The Sentinel is last because ``$1B8E`` disables absorption once it falls, and the
    platform is reached only after it."""
    sentries = []
    sentinel = None
    for slot in enemies.enemy_slots(state):
        tile = tuple(int(v) for v in state.tile_of(slot))
        if slot == actions.SENTINEL_SLOT:
            sentinel = tile
        else:
            sentries.append(tile)
    return sentries, sentinel, tuple(state.platform_xy)


def leg_matrix(graph, tiles):
    """``legs[i][j]``: fewest hops from a stance striking ``tiles[i]`` to one striking
    ``tiles[j]``, read off the backward hop tables rather than searched."""
    n = len(tiles)
    dist = [graph.hops_to(t) for t in tiles]
    legs = np.full((n, n), UNREACHABLE, dtype=np.int64)
    for i, tile in enumerate(tiles):
        src = graph.strike_stances(tile)
        if not src.size:
            continue
        for j in range(n):
            if i == j:
                legs[i, j] = 0
                continue
            got = dist[j][src]
            live = got[got >= 0]
            if live.size:
                legs[i, j] = int(live.min())
    return legs


def start_legs(graph, tiles, start_seen, start_eye):
    """Hops from the live body to a stance striking each tile (0 when already landable)."""
    cols = graph._cols
    reach = start_seen[cols] & (graph.eye > start_eye)
    out = np.full(len(tiles), UNREACHABLE, dtype=np.int64)
    for j, tile in enumerate(tiles):
        if start_seen[tile[0] * mm.N + tile[1]]:
            out[j] = 0
            continue
        got = graph.hops_to(tile)[reach]
        live = got[got >= 0]
        if live.size:
            out[j] = 1 + int(live.min())
    return out


def order(graph, state, start_seen, start_eye):
    """The cheapest order to take the goals: sentries in any order, then the Sentinel,
    then the platform.  Held-Karp over the sentry subsets, so 2^m x m cells, not m!."""
    sentries, sentinel, platform = goal_tiles(state)
    tail = [t for t in (sentinel, platform) if t is not None]
    if not sentries:
        return _tail_only(graph, tail, start_seen, start_eye)
    tiles = sentries + tail
    legs = leg_matrix(graph, tiles)
    start = start_legs(graph, tiles, start_seen, start_eye)
    m = len(sentries)
    full = (1 << m) - 1
    best = np.full((1 << m, m), UNREACHABLE, dtype=np.int64)
    prev = np.full((1 << m, m), -1, dtype=np.int64)
    for i in range(m):
        best[1 << i, i] = start[i]
    for mask in range(1 << m):
        for i in range(m):
            if not mask & (1 << i) or best[mask, i] >= UNREACHABLE:
                continue
            for j in range(m):
                if mask & (1 << j):
                    continue
                cost = best[mask, i] + legs[i, j]
                nxt = mask | (1 << j)
                if cost < best[nxt, j]:
                    best[nxt, j] = cost
                    prev[nxt, j] = i
    finish = np.array(
        [best[full, i] + _tail_cost(legs, m, i, len(tiles)) for i in range(m)],
        dtype=np.int64,
    )
    last = int(np.argmin(finish))
    if finish[last] >= UNREACHABLE:
        return _tail_only(graph, tail, start_seen, start_eye)
    return [sentries[i] for i in _unwind(prev, full, last)] + tail


def _tail_cost(legs, m, i, n):
    """Hops from sentry ``i`` through the Sentinel and on to the platform."""
    cost = 0
    here = i
    for j in range(m, n):
        cost += legs[here, j]
        here = j
    return cost


def _unwind(prev, mask, last):
    out = []
    while last >= 0:
        out.append(last)
        nxt = int(prev[mask, last])
        mask &= ~(1 << last)
        last = nxt
    return list(reversed(out))


def _tail_only(graph, tail, start_seen, start_eye):
    """No sentries left: the order is whatever of Sentinel/platform remains reachable."""
    if not tail:
        return []
    reach = start_legs(graph, tail, start_seen, start_eye)
    return [t for t, d in zip(tail, reach.tolist()) if d < UNREACHABLE] or tail
