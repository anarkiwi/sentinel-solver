"""CUDA launch layer: one thread per (state, lattice ray) over the CUDA builds of the specs.

Each thread writes its own cell of a dense per-ray result (the packed landing tile, -1 for
a ray that does not end LOS_CLEAR), so results never depend on thread scheduling and need
no sort: the host compacts them in lattice order.  The state-independent lattice vectors
stay resident on the device between calls, and B board images march in one launch.
"""

# pylint: disable=no-value-for-parameter,too-many-function-args,comparison-with-callable

import functools
import hashlib
import os

import numpy as np
from numba import cuda

from sentinel import kern, landtable, los_jit, memmap as mm

BLOCK = 128
LOS_CLEAR = los_jit.LOS_CLEAR
_march_one = kern.build(los_jit._march_one, "cuda")
_stop_one = kern.build(landtable._stop_one, "cuda")
_MEMO = {}
_MEMO_MAX = 8


@cuda.jit(cache=True)
def _clear_kernel(
    mems, ax_lo, ax_hi, az_lo, az_hi, ay_lo, ay_hi, s30, seeds, obss, out
):
    """``out[b, i]`` = packed tile of ray ``i`` marched against image ``b``, -1 unless
    the ray ends LOS_CLEAR."""
    i, b = cuda.grid(2)
    if i < ax_lo.shape[0] and b < mems.shape[0]:
        r = _march_one(
            mems[b], i, ax_lo, ax_hi, az_lo, az_hi, ay_lo, ay_hi, s30, seeds[b], obss[b]
        )
        out[b, i] = r[1] * mm.N + r[2] if r[0] == LOS_CLEAR else -1


@cuda.jit(cache=True)
def _stop_kernel(vx, vy, vz, lo, hi, slope, ex, ey, zseed, max_steps, edge, bits):
    """Set bit ``j`` of ``bits`` for every ray with a candidate landing tile."""
    j = cuda.grid(1)
    if j < vx.shape[0]:
        _first, ncand = _stop_one(
            np.int64(vx[j]),
            np.int64(vy[j]),
            np.int64(vz[j]),
            lo,
            hi,
            slope,
            ex,
            ey,
            zseed,
            max_steps,
            edge,
        )
        if ncand > 0:
            cuda.atomic.or_(bits, j >> 5, np.uint32(1 << (j & 31)))


def _blocks(n):
    return (n + BLOCK - 1) // BLOCK


def resident_threads():
    """Threads the device keeps in flight at once (0 under the simulator, which has no
    occupancy to fill)."""
    if os.environ.get("NUMBA_ENABLE_CUDASIM") == "1":
        return 0
    dev = cuda.get_current_device()
    return dev.MULTIPROCESSOR_COUNT * dev.MAX_THREADS_PER_MULTIPROCESSOR


def landing_dense(mems, vecs, params):
    """``(B, rays)`` int16 on the device for the ``(B, 65536)`` image stack ``mems`` and
    ``params`` rows (seed then observer, as :func:`landtable._march_params`): the packed
    landing tile of every ray of ``vecs`` against each image, -1 where it does not land.
    One launch; the result stays a device array."""
    n = vecs[0].shape[0]
    rows = np.asarray(params, dtype=np.int64)
    dense = cuda.device_array((len(mems), n), dtype=np.int16)
    _clear_kernel[(_blocks(n), len(mems)), BLOCK](
        cuda.to_device(np.ascontiguousarray(mems, dtype=np.uint8)),
        *vecs,
        cuda.to_device(np.ascontiguousarray(rows[:, :9])),
        cuda.to_device(np.ascontiguousarray(rows[:, 9:])),
        dense,
    )
    return dense


def landing_many(mems, vecs, params):
    """``[(index, tile)]`` per image of ``mems``: every CLEAR ray of ``vecs`` marched
    against that image from its ``params``, ascending.  Images go in launches of as many
    as :data:`landtable.RESULT_BYTES` of dense results hold."""
    step = max(1, landtable.RESULT_BYTES // (2 * vecs[0].shape[0]))
    out = []
    for lo in range(0, len(mems), step):
        part = slice(lo, lo + step)
        images = np.stack([np.asarray(m, dtype=np.uint8) for m in mems[part]])
        out += landtable._compact(
            landing_dense(images, vecs, params[part]).copy_to_host()
        )
    return out


def landing(mem, vecs, params):
    """:func:`landing_many` for one image."""
    return landing_many([mem], vecs, [params])[0]


def stop_mask(comps, lo, hi, slope, ex, ey, zseed, max_steps, edge):
    """Boolean per ray: :func:`landtable.stop_cells` reports a candidate landing tile."""
    n = comps[0].shape[0]
    bits = cuda.to_device(np.zeros((n + 31) // 32, dtype=np.uint32))
    surf = (cuda.to_device(a) for a in (lo, hi, slope))
    _stop_kernel[_blocks(n), BLOCK](*comps, *surf, ex, ey, zseed, max_steps, edge, bits)
    raw = bits.copy_to_host().view(np.uint8)
    return np.unpackbits(raw, bitorder="little")[:n].astype(bool)


@functools.lru_cache(maxsize=4)
def device_vectors(key):
    """The seven int16 lattice component arrays of grids ``key``, resident on the device."""
    return tuple(
        cuda.to_device(np.ascontiguousarray(a)) for a in landtable.vectors(key)
    )


@functools.lru_cache(maxsize=4)
def device_components(key):
    """The int32 ``(vx, vy, vz)`` of grids ``key``, resident on the device."""
    comps = landtable.cached_components(key)
    return tuple(cuda.to_device(np.ascontiguousarray(a)) for a in comps)


def _tile_first(idx, tile):
    first = np.full(mm.N * mm.N, -1, dtype=np.int64)
    tiles, at = np.unique(tile, return_index=True)
    first[tiles] = idx[at]
    return first


def landing_tables(states, slots, eye_z, grids, max_steps):
    """``(index, tile, first)`` per state: ``grids``' whole-lattice sweep against each
    ``states[i]`` from ``slots[i]`` -- ascending CLEAR ray indices, their packed tiles,
    and per tile the lowest index (-1 none).  Memoized on the board bytes, which
    determine the march; boards repeated within the call march once, the rest in one
    launch."""
    key = tuple(tuple(g) for g in grids)
    memos = [
        (
            hashlib.blake2b(st.mem, digest_size=16).digest(),
            slot,
            eye_z,
            max_steps,
            key,
        )
        for st, slot in zip(states, slots)
    ]
    have = {m: _MEMO[m] for m in memos if m in _MEMO}
    todo = {}
    for i, memo in enumerate(memos):
        if memo not in have:
            todo.setdefault(memo, i)
    if todo:
        at = list(todo.values())
        got = landing_many(
            [np.frombuffer(states[i].mem, dtype=np.uint8) for i in at],
            device_vectors(key),
            [
                landtable._march_params(states[i], slots[i], eye_z, max_steps)
                for i in at
            ],
        )
        for memo, (idx, tile) in zip(todo, got):
            if len(_MEMO) >= _MEMO_MAX:
                _MEMO.pop(next(iter(_MEMO)))
            have[memo] = _MEMO[memo] = (idx, tile, _tile_first(idx, tile))
    return [have[memo] for memo in memos]


def landing_table(state, slot, eye_z, grids, max_steps):
    """:func:`landing_tables` for one state."""
    return landing_tables([state], [slot], eye_z, grids, max_steps)[0]
