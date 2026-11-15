"""Precomputed landability candidates: a sound superset filter over the aim lattice.

The ray seed is state-independent (get_object_details $1ECC always centres x/y), so a
ray's track of (tile offset, ray z) is a pure function of its aim; terrain only decides
where the march stops.  Derivation, soundness and sizing: docs/architecture.md.
"""

import math

import numpy as np

from sentinel import los, los_jit, memmap as mm
from sentinel import kern
from sentinel.kern import spec

try:
    from numba import njit, prange

    _HAVE_JIT = True
except Exception:  # pragma: no cover - numba absent -> pure-Python (slow) build
    _HAVE_JIT = False
    prange = range

    def njit(*args, **kwargs):
        if len(args) == 1 and callable(args[0]) and not kwargs:
            return args[0]

        def wrap(fn):
            return fn

        return wrap


RADIUS = 30  # max in-board tile offset (LOS quits at coord $1F, so 0..30 is the board)
# the landset grid (pinned by tests): CURSOR_CX/CY subsampled 2:1
COARSE_CX = list(range(48, 112, 2))
COARSE_CY = list(range(63, 127, 2))
MAX_STEPS = 6000  # the los sweep's march cap
RESULT_BYTES = (
    1 << 28
)  # device or host bytes one launch may spend on dense per-ray results
MARCH_CHUNK = 1 << 13  # first early-exit march chunk: ~20us/call fixed cost, <1% here
LAND_BAND = 0x80  # check_flat_tile $1D1C lands only for surface-z in [0, $80)
WRAP_D = 0x10000 - LAND_BAND  # z distance at which the 8-bit high-byte compare aliases
NEVER_LANDS = 1  # tile flag: sloping, check_sloping_tile $1D46 only loops or blocks
HAS_BOULDER = 2  # tile flag: the object stack holds a boulder
BOARD_BOULDER = 4  # set on every tile when any tile has one
BOULDER = 3  # object type of a boulder


def lattice(v_primary=False, coarse=False):
    """The (hgrid, vgrid, cxs, cys) a query runs on: the grids
    :func:`los._landable_sweep` sweeps (``v_primary`` = the cheap $F5 plane), or the 2:1
    subsampled landset grid :func:`landable_set` uses (``coarse``)."""
    if coarse:
        return los.kbd_grids(COARSE_CX, COARSE_CY)
    return los.kbd_grids(v_primary=v_primary)


def vectors(grids):
    """The seven int16 component arrays of a lattice (:func:`los._lattice_vectors`)."""
    return los._lattice_vectors(*grids)


def components(grids):
    """``(vx, vy, vz)`` signed int32 per-sub-step increments for a lattice, from the
    cached :func:`los._lattice_vectors` bytes."""
    vx_lo, vx_hi, vz_lo, vz_hi, vy_lo, vy_hi, _s30 = los._lattice_vectors(*grids)

    def s16(hi, lo):
        v = ((hi.astype(np.int32) & 0xFF) << 8) | (lo.astype(np.int32) & 0xFF)
        return np.where(v >= 0x8000, v - 0x10000, v).astype(np.int32)

    return s16(vx_hi, vx_lo), s16(vy_hi, vy_lo), s16(vz_hi, vz_lo)


_COMP_CACHE = {}


def cached_components(grids):
    """:func:`components`, memoized per lattice."""
    key = tuple(tuple(g) for g in grids)
    got = _COMP_CACHE.get(key)
    if got is None:
        got = components(grids)
        _COMP_CACHE[key] = got
    return got


@spec(inline=True)
def _cell_last(v, d, max_steps):
    """The last sub-step of :func:`_cell_span`'s interval."""
    if v == 0:
        return max_steps if d == 0 else 0
    if v > 0:
        return int(math.ceil((d * 65536 + 0x8000) / v)) - 1
    return int(math.floor((d * 65536 - 0x8000) / v))


def seed_z(state, slot, eye_z=None):
    """The ray's starting 16-bit z, ``eye_z*256 + obj_z_frac`` (get_object_details
    $1ECC): the z fraction is a pure additive offset, never a table axis."""
    z = state.obj_z_height[slot] if eye_z is None else (eye_z & 0xFF)
    return z * 256 + state.obj_z_frac[slot]


@njit(cache=True)
def _surface_maps_mem(mem):
    """:func:`surface_maps` over a whole memory image (:func:`surface_bounds` and the
    per-tile flags, the object stack walked in place)."""
    lo = np.zeros(mm.N * mm.N, dtype=np.int32)
    hi = np.zeros(mm.N * mm.N, dtype=np.int32)
    flags = np.zeros(mm.N * mm.N, dtype=np.uint8)
    boulders = False
    for x in range(mm.N):
        for y in range(mm.N):
            b = np.int64(mem[mm.TILES_TABLE + (x & 3) * 256 + (x >> 2) * 32 + y])
            c = x * mm.N + y
            if b < 0xC0:
                lo[c] = hi[c] = ((b >> 4) & 0x0F) * 256
                flags[c] = NEVER_LANDS if b & 0x0F else 0
                continue
            lo[c] = 1 << 30
            slot = b & 0x3F
            for _ in range(64):
                z = np.int64(mem[mm.OBJECTS_Z_HEIGHT + slot]) * 256
                zf = np.int64(mem[mm.OBJECTS_Z_FRACTION + slot])
                lo[c] = min(lo[c], z + min(0, zf - 0x60))
                hi[c] = max(hi[c], z + zf + 0x20)
                if mem[mm.OBJECTS_TYPE + slot] == BOULDER:
                    flags[c] |= HAS_BOULDER
                    boulders = True
                link = np.int64(mem[mm.OBJECTS_FLAGS + slot])
                if link < 0x40:
                    break
                slot = link & 0x3F
    if boulders:
        for c in range(mm.N * mm.N):
            flags[c] |= BOARD_BOULDER
    return lo, hi, flags


def surface_maps(state):
    """``(lo, hi, flags)`` over the whole board, packed ``x*32 + y`` -- :func:`surface_bounds`
    for every tile plus its flag bits (:data:`NEVER_LANDS`, :data:`HAS_BOULDER` and, on
    every tile, :data:`BOARD_BOULDER`), hoisted out of the per-ray walk."""
    return _surface_maps_mem(np.frombuffer(state.mem, dtype=np.uint8))


@spec
def _stop_one(ax, ay, az, surf_lo, surf_hi, flags, ex, ey, zseed, max_steps, edge):
    """One ray's ``(first candidate tile, count)`` for :func:`stop_cells`.

    The ROM rejects a look-up (``az >= 0``) at a flat or object tile unless the object walk
    set ``$0C67`` bit 7 (only a boulder does; landing sweeps keep ``$0C6E`` bit 7 clear), so
    a climbing ray lands only on a :data:`HAS_BOULDER` tile.
    """
    climb = az >= 0
    if climb and (flags[ex * 32 + ey] & BOARD_BOULDER) == 0:
        return -1, 0
    first = -1
    ncand = 0
    i0 = 1
    while i0 <= max_steps:
        dx = (0x8000 + i0 * ax) >> 16
        dy = (0x8000 + i0 * ay) >> 16
        x = ex + dx
        y = ey + dy
        if x < 0 or x > edge or y < 0 or y > edge:
            break
        lx = _cell_last(ax, dx, max_steps)
        ly = _cell_last(ay, dy, max_steps)
        i1 = min(lx, ly, max_steps)
        zin = (i0 * az) >> 8
        zout = (i1 * az) >> 8
        c = x * 32 + y
        if (x != ex or y != ey) and (flags[c] & NEVER_LANDS) == 0:
            tlo = surf_lo[c] - zseed
            thi = surf_hi[c] - zseed
            zmin = zout if az < 0 else zin
            zmax = zin if az < 0 else zout
            alias = zmax >= WRAP_D + tlo
            if az > 0:
                cand = alias or (tlo - LAND_BAND < zin <= thi)
            elif az < 0:
                cand = alias or (zin > tlo - LAND_BAND and zout <= thi)
            else:
                zmin = 0
                cand = tlo - LAND_BAND < 0 <= thi
            if cand and (not climb or (flags[c] & HAS_BOULDER) != 0):
                if ncand == 0:
                    first = c
                ncand += 1
            if not alias and zmin <= tlo:
                break  # the ray reaches this surface: the ROM stops in this cell
        i0 = i1 + 1
    return first, ncand


@njit(cache=True, parallel=True)
def stop_cells(vx, vy, vz, surf_lo, surf_hi, flags, ex, ey, zseed, max_steps, edge):
    """Per ray: its first candidate landing tile (packed ``x*32+y``, -1 none) and count.

    The track walk partitions the lattice by landing tile, so a board sweep marches each ray
    once at most; the count stops at a cell the ray PROVABLY stops in, and sloping tiles and
    the observer's tile never land nor terminate (``flags`` as :func:`surface_maps`).
    """
    n = vx.shape[0]
    first = np.full(n, -1, dtype=np.int32)
    ncand = np.zeros(n, dtype=np.int32)
    for j in prange(n):  # pylint: disable=not-an-iterable
        first[j], ncand[j] = _stop_one(
            vx[j], vy[j], vz[j], surf_lo, surf_hi, flags, ex, ey, zseed, max_steps, edge
        )
    return first, ncand


def _march_params(state, slot, eye_z, max_steps):
    """The march's scalar inputs: the 9 seed bytes then the 7 observer values."""
    return (
        *(int(v) for v in los._seed_position(state, slot, eye_z)),
        int(state.obj_x[slot] & 0xFF),
        int(state.obj_y[slot] & 0xFF),
        int(state.mem[0x0C6E] & 0x7F),
        int(state.mem[0x0C58] & 0xFF),
        int((state.mem[0x0C56] >> 1) & 0xFF),
        int((state.mem[0x0CDD] >> 1) & 0xFF),
        max_steps,
    )


def _march(state, slot, eye_z, grids, idx, max_steps):
    """March the given lattice indices against ``state``; ``(status, tx, ty)``."""
    sub = [np.ascontiguousarray(a[idx]) for a in vectors(grids)]
    status, tx, ty, _centre = los.los_jit.march_batch(
        np.frombuffer(state.mem, dtype=np.uint8),
        *sub,
        *_march_params(state, slot, eye_z, max_steps),
    )
    return status, tx, ty


def _gpu():
    """The CUDA launch layer when the backend is cuda, else None."""
    if kern.backend() != "cuda":
        return None
    from sentinel import kern_cuda  # pylint: disable=import-outside-toplevel

    return kern_cuda


@njit(cache=True, parallel=True)
def _landing_states(mems, vecs, comps, params, zseed, max_steps, edge, order, group):
    """Per image (one thread each) and lattice ray, the LOS_CLEAR landing tile, else -1.

    Rays are visited in ``order``, claiming ``(tile, group[ray])`` on landing; a ray whose
    only candidate tile its group claimed is left -1 unmarched.  Groups are contiguous in
    ``order``; a ``group`` of -1 turns the pruning off."""
    ax_lo, ax_hi, az_lo, az_hi, ay_lo, ay_hi, s30 = vecs
    vx, vy, vz = comps
    n = vx.shape[0]
    out = np.full((mems.shape[0], n), -1, dtype=np.int16)
    for b in prange(mems.shape[0]):  # pylint: disable=not-an-iterable
        lo, hi, flags = _surface_maps_mem(mems[b])
        seed, obs = params[b, :9], params[b, 9:]
        claimed = np.full(mm.N * mm.N, -2, dtype=np.int32)
        for k in range(n):
            j = order[k]
            first, ncand = _stop_one(
                vx[j],
                vy[j],
                vz[j],
                lo,
                hi,
                flags,
                obs[0],
                obs[1],
                zseed[b],
                max_steps,
                edge,
            )
            if ncand == 0 or (ncand == 1 and claimed[first] == group[j] >= 0):
                continue
            r = los_jit._march_one(
                mems[b], j, ax_lo, ax_hi, az_lo, az_hi, ay_lo, ay_hi, s30, seed, obs
            )
            if r[0] == los_jit.LOS_CLEAR:
                out[b, j] = r[1] * mm.N + r[2]
                claimed[r[1] * mm.N + r[2]] = group[j]
    return out


KEEPS = ("all", "first", "cell")


def claim_plan(grids, keep="all", drive=None):
    """``(order, group)`` for :func:`_landing_states`: ray visit order and claim groups.

    ``"first"`` claims per tile in ray order (the lowest ray per tile).  ``"cell"`` claims
    per (tile, cell) in ``(drive, ray)`` order, where lattice index ``i`` lies in cell
    ``i // len(drive)`` at drive ``drive[i % len(drive)]``.  ``"all"`` prunes nothing.
    """
    n = cached_components(grids)[0].shape[0]
    ray = np.arange(n, dtype=np.int64)
    if keep == "all":
        return ray, np.full(n, -1, dtype=np.int32)
    if keep == "first":
        return ray, np.zeros(n, dtype=np.int32)
    if keep != "cell":
        raise ValueError(f"keep must be one of {KEEPS}: {keep!r}")
    nd = len(drive)
    within = np.lexsort((np.arange(nd), drive))
    order = (np.arange(n // nd)[:, None] * nd + within).ravel()
    return order, (ray // nd).astype(np.int32)


def _compact(dense):
    """``[(rays, tiles)]`` per row of a dense per-ray tile matrix (-1 = no landing)."""
    out = []
    for row in dense:
        idx = np.flatnonzero(row >= 0)
        out.append((idx, row[idx].astype(np.int64)))
    return out


def landing_dense(mems, grids, params, zseed, max_steps, keep="all", drive=None):
    """``(B, rays)`` int16 on the CPU, one image per thread: each lattice ray's landing tile.

    Against image ``b`` (``params[b]`` as :func:`_march_params`, ``zseed[b]`` as
    :func:`seed_z`), -1 where it does not land; ``keep``/``drive`` (:func:`claim_plan`) may
    also leave -1 a landing ray that is not the minimum of its claim group.
    """
    return _landing_states(
        mems,
        vectors(grids),
        cached_components(grids),
        params,
        zseed,
        max_steps,
        RADIUS,
        *claim_plan(grids, keep, drive),
    )


def landing_pairs_many(states, slots, eye_z, grids, max_steps, keep="all", drive=None):
    """:func:`landing_pairs` of every state, one state per thread: for a batch that fills
    the threads this beats the ray-parallel loops, which stop scaling well within one
    state.  Images go in groups of as many as :data:`RESULT_BYTES` of dense results hold.
    ``keep``/``drive`` as :func:`claim_plan`.
    """
    n = cached_components(grids)[0].shape[0]
    step = max(1, RESULT_BYTES // (2 * n))
    out = []
    for lo in range(0, len(states), step):
        part = range(lo, min(lo + step, len(states)))
        out += _compact(
            landing_dense(
                np.stack([np.frombuffer(states[i].mem, dtype=np.uint8) for i in part]),
                grids,
                np.array(
                    [
                        _march_params(states[i], slots[i], eye_z, max_steps)
                        for i in part
                    ],
                    dtype=np.int64,
                ),
                np.array(
                    [seed_z(states[i], slots[i], eye_z) for i in part], dtype=np.int64
                ),
                max_steps,
                keep,
                drive,
            )
        )
    return out


def landing_pairs(state, slot, eye_z, grids, max_steps, keep="all", drive=None):
    """``(rays, tiles)`` of every CLEAR ray of ``grids`` against ``state`` on the CPU.

    Ascending lattice indices and packed ``x*32+y`` tiles; only :func:`landing_rays` march.
    ``keep``/``drive`` (:func:`claim_plan`) march candidates in claim order, in doubling
    chunks, dropping a ray whose only candidate tile its group claimed."""
    if keep == "all" or _gpu() is not None:
        idx = landing_rays(state, slot, eye_z, grids, max_steps)
        status, tx, ty = _march(state, slot, eye_z, grids, idx, max_steps)
        hit = np.flatnonzero(status == los.los_jit.LOS_CLEAR)
        return idx[hit], tx[hit].astype(np.int64) * mm.N + ty[hit].astype(np.int64)
    first, ncand = _partition(state, slot, eye_z, grids, max_steps)
    order, group = claim_plan(grids, keep, drive)
    order = order[ncand[order] > 0]
    claimed = np.full(mm.N * mm.N, -2, dtype=np.int32)
    rays, tiles = [np.empty(0, np.int64)], [np.empty(0, np.int64)]
    lo, step = 0, MARCH_CHUNK
    while lo < order.size:
        sel = order[lo : lo + step]
        lo += step
        step += step
        sel = sel[(ncand[sel] > 1) | (claimed[first[sel]] != group[sel])]
        status, tx, ty = _march(state, slot, eye_z, grids, sel, max_steps)
        hit = np.flatnonzero(status == los.los_jit.LOS_CLEAR)
        tile = tx[hit].astype(np.int64) * mm.N + ty[hit].astype(np.int64)
        claimed[tile] = group[sel[hit]]
        rays.append(sel[hit])
        tiles.append(tile)
    rays, tiles = np.concatenate(rays), np.concatenate(tiles)
    by_ray = np.argsort(rays)
    return rays[by_ray], tiles[by_ray]


def _partition(state, slot, eye_z, grids, max_steps):
    """:func:`stop_cells` over a whole lattice: ``(first candidate tile, count)`` per ray."""
    lo, hi, slope = surface_maps(state)
    return stop_cells(
        *cached_components(grids),
        lo,
        hi,
        slope,
        int(state.obj_x[slot]),
        int(state.obj_y[slot]),
        seed_z(state, slot, eye_z),
        max_steps,
        RADIUS,
    )


def landing_rays(state, slot=None, eye_z=None, grids=None, max_steps=MAX_STEPS):
    """Ascending lattice indices holding EVERY ray that can land -- :func:`candidates`
    for the whole board at once.

    :func:`stop_cells` walks each ray's track closed-form against the real surfaces, so a
    ray naming no candidate landing tile provably never lands and is dropped unmarched.
    """
    if slot is None:
        slot = state.player
    grids = lattice() if grids is None else grids
    gpu = _gpu()
    if gpu is not None:
        lo, hi, slope = surface_maps(state)
        mask = gpu.stop_mask(
            gpu.device_components(tuple(tuple(g) for g in grids)),
            lo,
            hi,
            slope,
            int(state.obj_x[slot]),
            int(state.obj_y[slot]),
            seed_z(state, slot, eye_z),
            max_steps,
            RADIUS,
        )
        return np.flatnonzero(mask)
    _first, ncand = _partition(state, slot, eye_z, grids, max_steps)
    return np.flatnonzero(ncand > 0)
