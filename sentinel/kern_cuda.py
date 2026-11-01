"""CUDA launch layer: one thread per lattice ray over the CUDA builds of the specs.

Results are order-independent integer reductions -- an atomic count that compacts the
landing rays (sorted on the host by lattice index) and an atomic OR into a bitmask -- so
they equal the CPU kernels' regardless of thread scheduling.  The state-independent
lattice vectors stay resident on the device between calls.
"""

# pylint: disable=no-value-for-parameter,too-many-function-args

import functools
import hashlib

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
def _clear_kernel(mem, ax_lo, ax_hi, az_lo, az_hi, ay_lo, ay_hi, s30, seed, obs, out):
    """Compact ``(index, tile)`` of every ray that ends LOS_CLEAR into ``out``; the
    count is ``out[2, 0]`` (rows: index, packed tile, counter)."""
    i = cuda.grid(1)
    if i < ax_lo.shape[0]:
        r = _march_one(mem, i, ax_lo, ax_hi, az_lo, az_hi, ay_lo, ay_hi, s30, seed, obs)
        if r[0] == LOS_CLEAR:
            slot = cuda.atomic.add(out, (2, 0), 1)
            out[0, slot] = i
            out[1, slot] = r[1] * mm.N + r[2]


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


def landing(mem, vecs, params):
    """``(index, tile)`` of every CLEAR ray of ``vecs`` marched against ``mem`` from
    ``params`` (seed then observer, as :func:`landtable._march_params`), ascending."""
    n = vecs[0].shape[0]
    out = cuda.to_device(np.zeros((3, n + 1), dtype=np.int32))
    kern_args = (
        cuda.to_device(np.asarray(mem, dtype=np.uint8)),
        *vecs,
        cuda.to_device(np.asarray(params[:9], dtype=np.int64)),
        cuda.to_device(np.asarray(params[9:], dtype=np.int64)),
        out,
    )
    _clear_kernel[_blocks(n), BLOCK](*kern_args)
    count = int(out[2, 0:1].copy_to_host()[0])
    got = out[:2, :count].copy_to_host().astype(np.int64)
    order = np.argsort(got[0])
    return got[0][order], got[1][order]


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


def landing_table(state, slot, eye_z, grids, max_steps):
    """``(index, tile, first)`` of ``grids``' whole-lattice sweep against ``state``:
    ascending CLEAR ray indices, their packed tiles, and per tile the lowest index (-1
    none).  Memoized on the board bytes, which determine the march."""
    key = tuple(tuple(g) for g in grids)
    memo = (
        hashlib.blake2b(state.mem, digest_size=16).digest(),
        slot,
        eye_z,
        max_steps,
        key,
    )
    got = _MEMO.get(memo)
    if got is None:
        idx, tile = landing(
            np.frombuffer(state.mem, dtype=np.uint8),
            device_vectors(key),
            landtable._march_params(state, slot, eye_z, max_steps),
        )
        first = np.full(mm.N * mm.N, -1, dtype=np.int64)
        tiles, at = np.unique(tile, return_index=True)
        first[tiles] = idx[at]
        if len(_MEMO) >= _MEMO_MAX:
            _MEMO.pop(next(iter(_MEMO)))
        got = _MEMO[memo] = (idx, tile, first)
    return got
