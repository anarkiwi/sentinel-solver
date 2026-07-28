"""Precomputed landability candidates: a sound superset filter over the aim lattice.

The ray seed is state-independent (get_object_details $1ECC always centres x/y), so a
ray's track of (tile offset, ray z) is a pure function of its aim; terrain only decides
where the march stops.  Derivation, soundness and sizing: docs/landtable.md.
"""

import os

import numpy as np

from sentinel import los, memmap as mm, terrain

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
BUCKET_LO = -24  # census T buckets, whole z units; real play spans -5..+6
BUCKET_HI = 23
MAX_STEPS = 6000  # the los sweep's march cap
MARCH_CHUNK = 1 << 13  # first early-exit march chunk: ~20us/call fixed cost, <1% here
LAND_BAND = 0x80  # check_flat_tile $1D1C lands only for surface-z in [0, $80)
WRAP_D = 0x10000 - LAND_BAND  # z distance at which the 8-bit high-byte compare aliases
_WILD = -(1 << 40)  # zmin sentinel: unbounded (wrap-aliasing) candidate


def lattice(v_primary=False, coarse=False):
    """The (hgrid, vgrid, cxs, cys) a query runs on: the grids
    :func:`los._landable_sweep` sweeps (``v_primary`` = the cheap $F5 plane), or the 2:1
    subsampled landset grid :func:`landable_set` uses (``coarse``)."""
    if coarse:
        return los.kbd_grids(COARSE_CX, COARSE_CY)
    return los.kbd_grids(v_primary=v_primary)


def _los_kwargs(grids):
    """``(cxs, cys, v_primary)`` -- the same lattice expressed as
    :func:`los.landable_view_targeted` arguments, for the numba-absent fallback."""
    _hgrid, vgrid, cxs, cys = grids
    return cxs, cys, list(vgrid) == [los.KBD_V_ANGLE]


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


@njit(cache=True, inline="always")
def _ceil_div(p, q):  # q > 0
    return -((-p) // q)


@njit(cache=True, inline="always")
def _cell_span(v, d, max_steps):
    """The sub-step interval ``[first, last]`` in which the ray's tile offset on one axis
    equals ``d`` (empty when ``last < first``)."""
    if v > 0:
        return (
            _ceil_div(d * 65536 - 0x8000, v),
            _ceil_div((d + 1) * 65536 - 0x8000, v) - 1,
        )
    if v < 0:
        return (
            _ceil_div(0x8000 + 1 - (d + 1) * 65536, -v),
            _ceil_div(0x8000 + 1 - d * 65536, -v) - 1,
        )
    if d == 0:
        return 1, max_steps
    return 1, 0


@njit(cache=True, parallel=True)
def crossing_mask(vx, vy, vz, sel, dx, dy, tlo, thi, max_steps):
    """Which rays of ``sel`` can land on cell ``(dx, dy)`` with surface z in [tlo, thi].

    Bounds are relative to the ray seed z, 1/256 units.  Landing needs ``surface-ray_z``
    in ``[0, $80)`` inside the cell, so the ray must ENTER above ``surface - $80`` and
    reach ``surface`` before leaving; 8-bit-alias risks are kept unconditionally.
    """
    n = sel.shape[0]
    out = np.zeros(n, dtype=np.bool_)
    wrap_z = WRAP_D + tlo
    for e in prange(n):  # pylint: disable=not-an-iterable
        j = sel[e]
        az = vz[j]
        fx, lx = _cell_span(vx[j], dx, max_steps)
        fy, ly = _cell_span(vy[j], dy, max_steps)
        i0 = fx if fx > fy else fy
        if i0 < 1:
            i0 = 1
        i1 = min(lx, ly, max_steps)
        if i0 > i1:
            continue
        zin = (i0 * az) >> 8
        zout = (i1 * az) >> 8
        if az > 0:  # climbing: z only leaves the band, so the entry must be inside it
            out[e] = zout >= wrap_z or (tlo - LAND_BAND < zin <= thi)
        elif az < 0:  # descending: enter above the band's floor, reach the surface
            out[e] = zin >= wrap_z or (zin > tlo - LAND_BAND and zout <= thi)
        else:
            out[e] = tlo - LAND_BAND < 0 <= thi
    return out


@njit(cache=True)
def _track(ax, ay, az, max_steps, radius, wrap_z, cells, zmins):
    """One entry per distinct cell offset visited, with the min ray z over that cell.

    Tile offsets are monotone in the sub-step, so each cell is one closed-form interval.
    ``zmins`` is 1/256 height units relative to the seed z; :data:`_WILD` marks a visit
    whose ray z could alias the ROM's 8-bit compare.  Returns the entry count.
    """
    span = 2 * radius + 1
    n = 0
    i0 = 1
    while i0 <= max_steps:
        cx = (0x8000 + i0 * ax) >> 16
        cy = (0x8000 + i0 * ay) >> 16
        if cx < -radius or cx > radius or cy < -radius or cy > radius:
            break
        _f, lx = _cell_span(ax, cx, max_steps)
        _g, ly = _cell_span(ay, cy, max_steps)
        i1 = min(lx, ly, max_steps)
        if az > 0:  # z monotone: extremes at the interval ends
            zlo = (i0 * az) >> 8
            zhi = (i1 * az) >> 8
        elif az < 0:
            zlo = (i1 * az) >> 8
            zhi = (i0 * az) >> 8
        else:
            zlo = 0
            zhi = 0
        cells[n] = (cx + radius) * span + (cy + radius)
        zmins[n] = _WILD if zhi >= wrap_z else zlo
        n += 1
        i0 = i1 + 1
    return n


@njit(cache=True, parallel=True)
def census(vx, vy, vz, max_steps, radius, blo, bhi, wrap_z, nthread):
    """``(counts[ncell, nb], wild[ncell], cells_per_ray[n])``: candidates per cell per
    EXACT T bucket (cumulate for a prefix), plus the unbounded-z wildcards per cell."""
    n = vx.shape[0]
    span = 2 * radius + 1
    ncell = span * span
    nb = bhi - blo + 1
    part = np.zeros((nthread, ncell, nb), dtype=np.int32)
    pwild = np.zeros((nthread, ncell), dtype=np.int32)
    per_ray = np.zeros(n, dtype=np.int32)
    for t in prange(nthread):  # pylint: disable=not-an-iterable
        cells = np.empty(4 * radius + 8, dtype=np.int64)
        zmins = np.empty(4 * radius + 8, dtype=np.int64)
        for j in range(t, n, nthread):
            m = _track(vx[j], vy[j], vz[j], max_steps, radius, wrap_z, cells, zmins)
            per_ray[j] = m
            for e in range(m):
                c = cells[e]
                z = zmins[e]
                if z == _WILD:
                    pwild[t, c] += 1
                    continue
                b = z >> 8
                if b > bhi:
                    continue
                if b < blo:
                    b = blo
                part[t, c, b - blo] += 1
    counts = np.zeros((ncell, nb), dtype=np.int64)
    wild = np.zeros(ncell, dtype=np.int64)
    for t in range(nthread):
        counts += part[t]
        wild += pwild[t]
    return counts, wild, per_ray


def surface_bounds(state, x, y):
    """``(lo, hi)`` bracketing the ``surface16`` (``tile_z*256 + $0079``) the LOS compare
    can see at ``(x, y)``: exact for bare terrain, else over the object stack -- the
    platform form $1E30 (``+$20``) on top, the boulder form $1E5A (``-$60``) at the
    bottom.  A wider bracket only widens the candidate set."""
    b = terrain.tile_byte(state, x, y)
    if b < 0xC0:
        z = ((b >> 4) & 0x0F) * 256
        return z, z
    lo = 1 << 30
    hi = 0
    slot = b & 0x3F
    for _ in range(64):
        z = state.obj_z_height[slot] * 256
        zf = state.obj_z_frac[slot]
        lo = min(lo, z + min(0, zf - 0x60))
        hi = max(hi, z + zf + 0x20)
        flags = state.obj_flags[slot]
        if flags < 0x40:
            break
        slot = flags & 0x3F
    return lo, hi


def seed_z(state, slot, eye_z=None):
    """The ray's starting 16-bit z, ``eye_z*256 + obj_z_frac`` (get_object_details
    $1ECC): the z fraction is a pure additive offset, never a table axis."""
    z = state.obj_z_height[slot] if eye_z is None else (eye_z & 0xFF)
    return z * 256 + state.obj_z_frac[slot]


def never_lands(state, tile, slot):
    """Whether no keyboard aim can land on ``tile``: the observer's own tile ($1D32
    keeps marching) or a sloping tile (check_sloping_tile $1D46 only loops or blocks).
    """
    if tile[0] == state.obj_x[slot] and tile[1] == state.obj_y[slot]:
        return True
    b = terrain.tile_byte(state, tile[0], tile[1])
    return b < 0xC0 and (b & 0x0F) != 0


def surface_maps(state):
    """``(lo, hi, slope)`` over the whole board, packed ``x*32 + y`` -- :func:`surface_bounds`
    for every tile plus the never-lands slope flag, hoisted out of the per-ray walk."""
    lo = np.zeros(mm.N * mm.N, dtype=np.int32)
    hi = np.zeros(mm.N * mm.N, dtype=np.int32)
    slope = np.zeros(mm.N * mm.N, dtype=np.uint8)
    for x in range(mm.N):
        for y in range(mm.N):
            a, b = surface_bounds(state, x, y)
            lo[x * mm.N + y] = a
            hi[x * mm.N + y] = b
            byte = terrain.tile_byte(state, x, y)
            slope[x * mm.N + y] = 1 if byte < 0xC0 and (byte & 0x0F) else 0
    return lo, hi, slope


@njit(cache=True, parallel=True)
def stop_cells(vx, vy, vz, surf_lo, surf_hi, slope, ex, ey, zseed, max_steps, edge):
    """Per ray: its first candidate landing tile (packed ``x*32+y``, -1 none) and count.

    Walking the track against the real surfaces partitions the lattice by landing tile, so
    a whole-board sweep marches each ray once at most.  The count stops at a cell the ray
    PROVABLY stops in; sloping tiles and the observer's own tile never land nor terminate.
    """
    n = vx.shape[0]
    first = np.full(n, -1, dtype=np.int32)
    ncand = np.zeros(n, dtype=np.int32)
    for j in prange(n):  # pylint: disable=not-an-iterable
        ax = vx[j]
        ay = vy[j]
        az = vz[j]
        i0 = 1
        while i0 <= max_steps:
            dx = (0x8000 + i0 * ax) >> 16
            dy = (0x8000 + i0 * ay) >> 16
            x = ex + dx
            y = ey + dy
            if x < 0 or x > edge or y < 0 or y > edge:
                break
            _fx, lx = _cell_span(ax, dx, max_steps)
            _fy, ly = _cell_span(ay, dy, max_steps)
            i1 = min(lx, ly, max_steps)
            zin = (i0 * az) >> 8
            zout = (i1 * az) >> 8
            c = x * 32 + y
            if (x != ex or y != ey) and slope[c] == 0:
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
                if cand:
                    if ncand[j] == 0:
                        first[j] = c
                    ncand[j] += 1
                if not alias and zmin <= tlo:
                    break  # the ray reaches this surface: the ROM stops in this cell
            i0 = i1 + 1
    return first, ncand


def candidates(state, tile, slot, eye_z=None, grids=None, max_steps=MAX_STEPS):
    """Ascending lattice indices that can land on ``tile`` -- a proven superset of the
    rays the sweep finds, from the heading arc (:func:`los._tile_arc_indices`) narrowed
    by :func:`crossing_mask`.  Sorting the survivors (~3x fewer) rather than the arc is
    the same set for a third of the sort."""
    grids = lattice() if grids is None else grids
    vx, vy, vz = cached_components(grids)
    ex, ey = int(state.obj_x[slot]), int(state.obj_y[slot])
    idx = los._tile_arc_indices(
        *los._lattice_headings(*grids), ex, ey, tile[0], tile[1]
    )
    if idx is None:
        idx = np.arange(vx.size)
    zs = seed_z(state, slot, eye_z)
    lo, hi = surface_bounds(state, tile[0], tile[1])
    keep = crossing_mask(
        vx, vy, vz, idx, tile[0] - ex, tile[1] - ey, lo - zs, hi - zs, max_steps
    )
    return np.sort(idx[keep])


def _march(state, slot, eye_z, grids, idx, max_steps):
    """March the given lattice indices against ``state``; ``(status, tx, ty)``."""
    sub = [np.ascontiguousarray(a[idx]) for a in los._lattice_vectors(*grids)]
    seed = los._seed_position(state, slot, eye_z)
    status, tx, ty, _centre = los.los_jit.march_batch(
        np.frombuffer(state.mem, dtype=np.uint8),
        *sub,
        *seed,
        state.obj_x[slot] & 0xFF,
        state.obj_y[slot] & 0xFF,
        state.mem[0x0C6E] & 0x7F,
        state.mem[0x0C58] & 0xFF,
        (state.mem[0x0C56] >> 1) & 0xFF,
        (state.mem[0x0CDD] >> 1) & 0xFF,
        max_steps,
    )
    return status, tx, ty


def landable_view(
    state, tile, slot=None, eye_z=None, grids=None, max_steps=MAX_STEPS, stats=None
):
    """The build view for ``tile``, identical to that lattice's full sweep entry.

    :func:`los.landable_view_targeted` delegates here.  Only :func:`candidates` is
    marched -- a proven superset of the landing rays, so a set that lands nothing is a
    proven "no view".  The answer is the LOWEST landing lattice index, so the ascending
    candidates are marched in doubling chunks and the first chunk that lands ends it:
    the same index, marching under 2x the prefix before it instead of the whole set.
    ``stats`` counts the rays actually marched.
    """
    if slot is None:
        slot = state.player
    grids = lattice() if grids is None else grids
    if stats is not None:
        stats["queries"] = stats.get("queries", 0) + 1
    if not los._HAVE_JIT:  # pragma: no cover - numba absent -> the pure-Python probe
        cxs, cys, v_primary = _los_kwargs(grids)
        return los.landable_view_targeted(
            state, tile, slot, eye_z, max_steps, cxs=cxs, cys=cys, v_primary=v_primary
        )
    if never_lands(state, tile, slot):
        return None
    idx = candidates(state, tile, slot, eye_z, grids, max_steps)
    lo, step = 0, MARCH_CHUNK
    while lo < idx.size:
        sel = idx[lo : lo + step]
        if stats is not None:
            stats["marched"] = stats.get("marched", 0) + int(sel.size)
        status, tx, ty = _march(state, slot, eye_z, grids, sel, max_steps)
        hit = np.flatnonzero(
            (status == los.los_jit.LOS_CLEAR) & (tx == tile[0]) & (ty == tile[1])
        )
        if hit.size:
            h, v, cx, cy = los._meta_at(int(sel[hit[0]]), *grids)
            return {"h_angle": h, "v_angle": v, "cursor": [cx, cy]}
        lo += step
        step += step
    if stats is not None:
        stats["proven_none"] = stats.get("proven_none", 0) + 1
    return None


def landable_set(
    state,
    slot=None,
    eye_z=None,
    grids=None,
    max_steps=MAX_STEPS,
    chunk=1 << 16,
    stats=None,
):
    """The whole-board landable tile set, as ``_coarse_landable`` returns it (ring incl).

    :func:`stop_cells` partitions the rays by candidate landing tile, so each ray is
    marched at most once and only while its tile is unresolved; rays with no candidate
    tile are never marched.  The partition is a superset, so the set is exact.
    """
    if slot is None:
        slot = state.player
    grids = lattice(coarse=True) if grids is None else grids
    if not los._HAVE_JIT:  # pragma: no cover - numba absent
        return set(los.landable_views(state, slot, eye_z, max_steps))
    vx, vy, vz = cached_components(grids)
    lo, hi, slope = surface_maps(state)
    first, ncand = stop_cells(
        vx,
        vy,
        vz,
        lo,
        hi,
        slope,
        int(state.obj_x[slot]),
        int(state.obj_y[slot]),
        seed_z(state, slot, eye_z),
        max_steps,
        RADIUS,
    )
    todo = np.flatnonzero(ncand > 0)
    cell = first[todo]
    multi = ncand[todo] > 1
    landed = np.zeros(mm.N * mm.N, dtype=bool)
    out = set()
    for s in range(0, todo.size, chunk):
        sel = todo[s : s + chunk]
        keep = sel[multi[s : s + chunk] | ~landed[cell[s : s + chunk]]]
        if keep.size == 0:
            continue
        if stats is not None:
            stats["marched"] = stats.get("marched", 0) + int(keep.size)
        status, tx, ty = _march(state, slot, eye_z, grids, keep, max_steps)
        clear = np.flatnonzero(status == los.los_jit.LOS_CLEAR)
        if clear.size:
            landed[tx[clear] * mm.N + ty[clear]] = True
            out.update(zip(tx[clear].tolist(), ty[clear].tolist()))
    return out


def main(argv=None):
    """Print the per-lattice track census: candidates per cell per threshold bucket."""
    import argparse

    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.parse_args(argv)
    nthread = max(1, min(32, os.cpu_count() or 1))
    for name, grids in (
        ("F5-plane", lattice(True)),
        ("full-band", lattice()),
        ("coarse", lattice(coarse=True)),
    ):
        vx, vy, vz = components(grids)
        counts, wild, per_ray = census(
            vx,
            vy,
            vz,
            MAX_STEPS,
            RADIUS,
            BUCKET_LO,
            BUCKET_HI,
            WRAP_D + BUCKET_LO * 256,
            nthread,
        )
        cum = np.cumsum(counts, axis=1)
        print(
            "lattice %s: rays %d cells/ray %.1f visits %d wildcards %d"
            % (
                name,
                int(vx.size),
                float(np.mean(per_ray)),
                int(counts.sum() + wild.sum()),
                int(wild.sum()),
            )
        )
        for b in (-8, -4, -2, -1, 0, 1, 2, 4):
            col = cum[:, b - BUCKET_LO] + wild
            print(
                "  candidates/cell at T<=%+d: mean %8.0f p50 %7d p90 %8d max %8d"
                % (
                    b,
                    float(np.mean(col)),
                    float(np.percentile(col, 50)),
                    float(np.percentile(col, 90)),
                    int(col.max()),
                )
            )
    return 0


if __name__ == "__main__":  # pragma: no cover
    raise SystemExit(main())
