"""Exact render costs by running the real 6502 code on jennings' native core, memoized.

jennings is byte- and cycle-exact (it generates ``golden_render_cost.json``) and runs
with the GIL released, so :func:`render_costs_exact` misses run in parallel threads.
Selected by ``RENDER_COST_BACKEND=exact``; ROM-gated (:func:`backend`).
"""

import collections
import concurrent.futures
import hashlib
import os
import sys
import threading

# Render-relevant board (memmap): flags/v_angle, terrain grid, object arrays; the fingerprint adds view (h,v) + player index.
_REGIONS = ((0x0100, 0x0180), (0x0400, 0x0800), (0x0900, 0x0A80))

_CACHE = collections.OrderedDict()
_PREPARED = collections.OrderedDict()
_MAX = int(os.environ.get("RENDER_CACHE_MAX", "8192"))
_STATS = {"hits": 0, "misses": 0}
_PREPARED_STATS = {"hits": 0, "misses": 0}
_LOCK = threading.Lock()
_WARNED = [False]


def backend():
    """This module when ``RENDER_COST_BACKEND=exact`` selects it and the ROM image is
    present, else None (warning once when selected but unavailable)."""
    if os.environ.get("RENDER_COST_BACKEND", "proxy").lower() != "exact":
        return None
    from sentinel.tests import oracle

    if oracle.available():
        return sys.modules[__name__]
    if not _WARNED[0]:
        _WARNED[0] = True
        print(
            f"RENDER_COST_BACKEND=exact unavailable ({oracle.IMG} absent); using proxy"
        )
    return None


def viewer(state, observer):
    """``state`` as the ROM redraws it from ``observer``: ``$000B`` holds the viewing
    slot (a transfer sets it before its $357D redraw), so any slot's view is exact."""
    if observer is None or observer == state.player:
        return state
    view = state.clone()
    view.player = observer
    return view


def _board_key(state):
    dig = hashlib.blake2b(digest_size=16)
    for a, b in _REGIONS:
        dig.update(state.mem[a:b])
    dig.update(bytes([state.mem[0x000B]]))
    return dig.digest()


def _key(state, h_angle, v_angle):
    return _board_key(state) + bytes([h_angle & 0xFF, v_angle & 0xFF])


def _memo(cache, key, compute, stats):
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


def _prepared(state):
    """The board after the view-independent $357D prologue, memoized per board."""
    from sentinel.tests import oracle

    image = bytes(state.mem)
    return _memo(
        _PREPARED,
        _board_key(state),
        lambda: oracle.prepared_image(image),
        _PREPARED_STATS,
    )


def prologue_cycles(state):
    """``(occlusion, tables)``: the CPU cycles of the player's $245B and $3700."""
    prep = _prepared(state)
    return prep.occlusion_cycles, prep.table_cycles


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
            _STATS,
        )

    if workers == 1 or len(views) == 1:
        return [one(h, v) for h, v in views]
    _prepared(state)
    with concurrent.futures.ThreadPoolExecutor(workers or os.cpu_count()) as pool:
        return list(pool.map(lambda hv: one(*hv), views))


def strip_replot_cost_exact(state, target):
    """Exact $1FA4..$1F9E frames of update_object_on_screen redrawing ``target``.

    $1F9F re-points the camera itself ($1FC2), so the key is the board plus the
    player's own view and the object being redrawn. $209B is the caller's to price."""
    from sentinel.tests import oracle

    mem = state.mem
    player = mem[0x000B]
    key = _key(state, mem[0x09C0 + player], mem[0x0140 + player]) + bytes([target])
    return _memo(
        _CACHE, key, lambda: oracle.strip_frames(_prepared(state), target), _STATS
    )


def reset():
    """Clear the cache and counters (tests)."""
    with _LOCK:
        _CACHE.clear()
        _PREPARED.clear()
        _STATS.update(hits=0, misses=0)
        _PREPARED_STATS.update(hits=0, misses=0)
