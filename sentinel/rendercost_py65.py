"""Exact plot_world render cost by running the real 6502 code in jennings, memoized.

jennings executes the machine code byte- and cycle-exact (it generates
``golden_render_cost.json``) on its native core with the GIL released, so misses
computed through :func:`render_costs_exact` run in parallel threads. ROM-gated.
"""

import collections
import concurrent.futures
import hashlib
import os
import threading

# Render-relevant board (memmap): flags/v_angle, terrain grid, object arrays; the fingerprint adds view (h,v) + player index.
_REGIONS = ((0x0100, 0x0180), (0x0400, 0x0800), (0x0900, 0x0A80))

_CACHE = collections.OrderedDict()
_PREPARED = collections.OrderedDict()
_MAX = int(os.environ.get("RENDER_CACHE_MAX", "8192"))
_STATS = {"hits": 0, "misses": 0}
_PREPARED_STATS = {"hits": 0, "misses": 0}
_LOCK = threading.Lock()


def _key(state, h_angle, v_angle):
    mem = state.mem
    dig = hashlib.blake2b(digest_size=16)
    for a, b in _REGIONS:
        dig.update(mem[a:b])
    dig.update(bytes([mem[0x000B], h_angle & 0xFF, v_angle & 0xFF]))
    return dig.digest()


def _memo(cache, key, compute, stats=_STATS):
    with _LOCK:
        hit = cache.get(key)
        if hit is not None:
            cache.move_to_end(key)
            stats["hits"] += 1
            return hit
        stats["misses"] += 1
    value = compute()
    with _LOCK:
        cache[key] = value
        if len(cache) > _MAX:
            cache.popitem(last=False)
    return value


def _board_key(state):
    dig = hashlib.blake2b(digest_size=16)
    for a, b in _REGIONS:
        dig.update(state.mem[a:b])
    dig.update(bytes([state.mem[0x000B]]))
    return dig.digest()


def _prepared(state):
    """The board after the view-independent render prologue, memoized per board."""
    from sentinel.tests import oracle

    image = bytes(state.mem)
    return _memo(
        _PREPARED,
        _board_key(state),
        lambda: oracle.prepared_image(image),
        _PREPARED_STATS,
    )


def render_cost_exact(state, h_angle, v_angle):
    """Exact plot_world frame cost for ``state`` viewed at (h_angle, v_angle),
    memoized on the render-relevant board bytes + view. Requires the ROM fixture."""
    return render_costs_exact(state, ((h_angle, v_angle),), workers=1)[0]


def render_costs_exact(state, views, workers=None):
    """:func:`render_cost_exact` for each (h_angle, v_angle) of ``views``; the misses
    run concurrently on ``workers`` threads (default: one per CPU)."""
    from sentinel.tests import oracle

    def one(h, v):
        return _memo(
            _CACHE,
            _key(state, h, v),
            lambda: oracle.plot_world_frames(_prepared(state), h, v),
        )

    if workers == 1 or len(views) == 1:
        return [one(h, v) for h, v in views]
    _prepared(state)
    with concurrent.futures.ThreadPoolExecutor(workers or os.cpu_count()) as pool:
        return list(pool.map(lambda hv: one(*hv), views))


def strip_replot_cost_exact(state, target):
    """Exact $1FA4..$1F9E strip-replot frame cost for ``target`` on ``state``, memoized.

    $1F9F re-points the camera itself ($1FC2), so the key is the board plus the
    player's own view and the object being redrawn. $209B is the caller's to price."""
    from sentinel.tests import oracle

    mem = state.mem
    player = mem[0x000B]
    key = _key(state, mem[0x09C0 + player], mem[0x0140 + player]) + bytes([target])

    def compute():
        cpu, m, mstate = oracle.machine_from_image(mem)
        return oracle.update_object_cost(
            cpu, m, mstate, target, from_pc=oracle.REPLOT_LINE
        )

    return _memo(_CACHE, key, compute)


def reset():
    """Clear the cache and counters (tests)."""
    with _LOCK:
        _CACHE.clear()
        _PREPARED.clear()
        _STATS.update(hits=0, misses=0)
        _PREPARED_STATS.update(hits=0, misses=0)
