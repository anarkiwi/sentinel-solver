"""Superset validation for :mod:`sentinel.landtable`.

Everything rests on one property: the candidate set for a cell CONTAINS every lattice
ray the exact sweep lands there.  Outer-ring tiles are asserted too -- soundness here is
geometric, not read off a flat board -- see docs/architecture.md.
"""

import numpy as np
import pytest

from sentinel import actions, landtable as lt, los, memmap as mm, terrain

pytestmark = pytest.mark.skipif(
    not los._HAVE_JIT, reason="numba absent: landtable queries defer to the exact path"
)


def _landings(st, slot, grids, eye_z=None):
    """``{tile: ascending lattice indices landing on it}`` from one full-lattice sweep."""
    vecs = los._lattice_vectors(*grids)
    seed = los._seed_position(st, slot, eye_z)
    status, tx, ty, _c = los.los_jit.march_batch(
        np.frombuffer(st.mem, dtype=np.uint8),
        *vecs,
        *seed,
        st.obj_x[slot] & 0xFF,
        st.obj_y[slot] & 0xFF,
        st.mem[0x0C6E] & 0x7F,
        st.mem[0x0C58] & 0xFF,
        (st.mem[0x0C56] >> 1) & 0xFF,
        (st.mem[0x0CDD] >> 1) & 0xFF,
        lt.MAX_STEPS,
    )
    clear = np.flatnonzero(status == los.los_jit.LOS_CLEAR)
    if clear.size == 0:
        return {}
    key = (tx[clear] << 16) | ty[clear]
    order = np.argsort(key, kind="stable")
    key = key[order]
    rays = clear[order]
    bnd = np.concatenate(([0], np.flatnonzero(key[1:] != key[:-1]) + 1, [key.size]))
    return {
        (int(key[a] >> 16), int(key[a] & 0xFFFF)): rays[a:b]
        for a, b in zip(bnd[:-1], bnd[1:])
    }


def test_coarse_lattice_is_the_landset_lattice():
    """The coarse grid's rays are a subset of the full band lattice's -- one candidate
    rule serves both paths."""
    assert set(lt.COARSE_CX) <= set(los.CURSOR_CX)
    assert set(lt.COARSE_CY) <= set(los.CURSOR_CY)
    coarse, band = lt.lattice(coarse=True), lt.lattice()
    assert coarse[0] == band[0] and coarse[1] == band[1]
    cvecs = np.stack(los._lattice_vectors(*coarse))
    bvecs = np.stack(los._lattice_vectors(*band))
    rng = np.random.default_rng(3)
    for i in rng.choice(cvecs.shape[1], 64, replace=False):
        h, v, cx, cy = los._meta_at(int(i), *coarse)
        nh, ncx, ncy = len(band[0]), len(band[2]), len(band[3])
        j = (
            (band[1].index(v) * nh + band[0].index(h)) * ncx * ncy
            + band[2].index(cx) * ncy
            + band[3].index(cy)
        )
        assert (cvecs[:, i] == bvecs[:, j]).all()


def test_closed_form_track_matches_add_vector():
    """The closed-form track equals the ROM add_vector $1CBB accumulation, and the seed z
    fraction is a pure additive offset (so it is never a table axis)."""
    grids = lt.lattice(False)
    vx, vy, vz = lt.components(grids)
    raw = los._lattice_vectors(*grids)
    rng = np.random.default_rng(7)
    for j in rng.choice(vx.size, 8, replace=False):
        for eye_z, zf in ((5, 0x60), (9, 0xE0), (0, 0x00)):
            vec = los.Vector()
            for attr, arr in zip(
                ("vx_lo", "vx_hi", "vz_lo", "vz_hi", "vy_lo", "vy_hi", "s30"), raw
            ):
                setattr(vec, attr, int(arr[j]) & 0xFF)
            vec.px_frac = vec.py_frac = vec.pz_frac = 0
            vec.px_sub = vec.py_sub = 0x80
            vec.pz_sub = zf
            vec.px_whole, vec.py_whole, vec.pz_whole = 13, 17, eye_z
            for i in range(1, 300):
                los._add_vector(vec)
                assert vec.px_whole == (13 + ((0x8000 + i * int(vx[j])) >> 16)) & 0xFF
                assert vec.py_whole == (17 + ((0x8000 + i * int(vy[j])) >> 16)) & 0xFF
                z16 = (vec.pz_whole << 8) | vec.pz_sub
                assert z16 == (eye_z * 256 + zf + ((i * int(vz[j])) >> 8)) & 0xFFFF


def _arc_ref(head, order, ex, ey, tx, ty):
    """Lattice indices whose heading lies within cell ``(tx, ty)``'s angular span."""
    dx = np.array([tx - ex - 0.5, tx - ex + 0.5])
    dy = np.array([ty - ey - 0.5, ty - ey + 0.5])
    if dx[0] <= 0 <= dx[1] and dy[0] <= 0 <= dy[1]:
        return order
    angs = np.arctan2(*np.meshgrid(dy, dx)).ravel()
    rel = (angs - angs[0] + np.pi) % (2 * np.pi) - np.pi
    a_lo = (angs[0] + rel.min() + np.pi) % (2 * np.pi) - np.pi - 1e-9
    a_hi = a_lo + np.ptp(rel) + 2e-9
    lo = np.searchsorted(head, a_lo, side="left")
    if a_hi <= np.pi:
        return order[lo : np.searchsorted(head, a_hi, side="right")]
    return np.concatenate(
        (order[lo:], order[: np.searchsorted(head, a_hi - 2 * np.pi, side="right")])
    )


def _span_ref(v, d):
    """Vectorised ``[first, last]`` sub-steps with tile offset ``d`` on one axis."""
    q = np.where(v == 0, 1, np.abs(v))
    pos = (-((0x8000 - d * 65536) // q), -((0x8000 - (d + 1) * 65536) // q) - 1)
    neg = (
        -((-(0x8001 - (d + 1) * 65536)) // q),
        -((-(0x8001 - d * 65536)) // q) - 1,
    )
    zero = (1, lt.MAX_STEPS) if d == 0 else (1, 0)
    return tuple(
        np.where(v > 0, p, np.where(v < 0, n, z)) for p, n, z in zip(pos, neg, zero)
    )


def _stack_and_transfer(st):
    """Build boulder+boulder+robot on a bare neighbour and take the robot as the body:
    object tiles, a raised eye, a different observer slot."""
    px, py = st.player_xy()
    tile = next(
        (px + dx, py + dy)
        for dx in (0, 1, -1)
        for dy in (1, -1, 0)
        if (dx or dy)
        and terrain.tile_byte(st, px + dx, py + dy) < mm.OBJECT_TILE
        and not (terrain.tile_byte(st, px + dx, py + dy) & 0x0F)
    )
    slot = None
    for otype in (mm.T_BOULDER, mm.T_BOULDER, mm.T_ROBOT):
        slot = actions.create(st, otype, tile)
    assert slot is not None
    st.player = slot
    return tile


def test_stop_cell_partition_holds_every_landing(new_state):
    """The invariant :func:`landtable.landable_set` rests on: a ray that lands on a tile
    either names that tile as its first candidate or is marked multi-candidate."""
    st = new_state(0)
    _stack_and_transfer(st)
    slot = st.player
    grids = lt.lattice(coarse=True)
    lo, hi, slope = lt.surface_maps(st)
    first, ncand = lt.stop_cells(
        *lt.cached_components(grids),
        lo,
        hi,
        slope,
        int(st.obj_x[slot]),
        int(st.obj_y[slot]),
        lt.seed_z(st, slot),
        lt.MAX_STEPS,
        lt.RADIUS,
    )
    for tile, rays in _landings(st, slot, grids).items():
        packed = tile[0] * mm.N + tile[1]
        assert ((first[rays] == packed) | (ncand[rays] > 1)).all(), tile
