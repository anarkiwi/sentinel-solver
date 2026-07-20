"""Exact plot_world render cost by running the real 6502 code in py65, memoized.

py65 executes the machine code byte-exact (it generates ``golden_render_cost.json``), so
the truly-exact frame cost is that emulated pass, not a reimplementation of the proxy in
:mod:`sentinel.projector`. Optional and ROM-gated: importing this never imports py65.
"""

import collections
import hashlib
import os

# Render-relevant board (memmap): flags/v_angle, terrain grid, object arrays; the fingerprint adds view (h,v) + player index.
_REGIONS = ((0x0100, 0x0180), (0x0400, 0x0800), (0x0900, 0x0A80))

_CACHE = collections.OrderedDict()
_MAX = int(os.environ.get("RENDER_CACHE_MAX", "8192"))
_STATS = {"hits": 0, "misses": 0}


def _key(state, h_angle, v_angle):
    mem = state.mem
    dig = hashlib.blake2b(digest_size=16)
    for a, b in _REGIONS:
        dig.update(mem[a:b])
    dig.update(bytes([mem[0x000B], h_angle & 0xFF, v_angle & 0xFF]))
    return dig.digest()


def render_cost_exact(state, h_angle, v_angle):
    """Exact plot_world frame cost for ``state`` viewed at (h_angle, v_angle),
    memoized on the render-relevant board bytes + view. Requires the ROM fixture."""
    key = _key(state, h_angle, v_angle)
    hit = _CACHE.get(key)
    if hit is not None:
        _CACHE.move_to_end(key)
        _STATS["hits"] += 1
        return hit
    _STATS["misses"] += 1
    from sentinel.tests import oracle

    cpu, mem, mstate = oracle.machine_from_image(state.mem)
    frames = oracle.render_frame_cost(cpu, mem, mstate, h_angle, v_angle)
    _CACHE[key] = frames
    if len(_CACHE) > _MAX:
        _CACHE.popitem(last=False)
    return frames


def reset():
    """Clear the cache and counters (tests)."""
    _CACHE.clear()
    _STATS.update(hits=0, misses=0)
