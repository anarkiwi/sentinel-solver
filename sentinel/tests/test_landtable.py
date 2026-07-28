"""Superset validation for :mod:`sentinel.landtable`.

Everything rests on one property: the candidate set for a cell CONTAINS every lattice
ray the exact sweep lands there.  Outer-ring tiles are asserted too -- soundness here is
geometric, not read off a flat board -- see docs/architecture.md.
"""

import numpy as np
import pytest

from sentinel import actions, landtable as lt, los, memmap as mm, playerbase, terrain

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


def _stack_and_transfer(st):
    """Build boulder+boulder+robot on a bare neighbour and transfer: object tiles, a
    raised eye, a different observer slot."""
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
    assert slot is not None and actions.transfer(st, slot)
    return tile


def _variants(new_state):
    """(name, state, slot, eye_z) observers: fresh boards, a stacked/transferred eye and
    an eye_z override (the search's speculative stance)."""
    out = []
    for ls in (0, 42):
        st = new_state(ls)
        out.append(("ls%d" % ls, st, st.player, None))
    st = new_state(0)
    _stack_and_transfer(st)
    out.append(("ls0-stacked", st, st.player, None))
    st = new_state(42)
    out.append(("ls42-eyez", st, st.player, (st.obj_z_height[st.player] + 3) & 0xFF))
    return out


def _assert_superset(st, slot, eye_z, grids):
    """Every ray the sweep lands on a tile is in that tile's candidate set."""
    land = _landings(st, slot, grids, eye_z)
    ring = 0
    for tile, rays in land.items():
        cand = lt.candidates(st, tile, slot, eye_z, grids)
        missing = np.setdiff1d(rays, cand, assume_unique=False)
        assert missing.size == 0, "%s dropped %d/%d rays" % (
            tile,
            missing.size,
            rays.size,
        )
        assert int(rays.min()) == int(cand[np.isin(cand, rays)].min())
        ring += tile[0] in (0, 30) or tile[1] in (0, 30)
    return land, ring


@pytest.mark.parametrize("name", ["ls0", "ls42", "ls0-stacked", "ls42-eyez"])
def test_candidates_superset_plane(new_state, name):
    """$F5 plane: the candidate set contains every landing ray, over bare boards, an
    object-tile/raised-eye stance and an eye_z override."""
    _name, st, slot, eye_z = next(v for v in _variants(new_state) if v[0] == name)
    _assert_superset(st, slot, eye_z, lt.lattice(True))


@pytest.mark.parametrize("name", ["ls0", "ls0-stacked", "ls42-eyez"])
def test_candidates_superset_band(new_state, name):
    """Full pitch band (3.5M rays, the completeness-critical lattice): below-eye and
    object tiles included, outer-ring tiles asserted like any other."""
    _name, st, slot, eye_z = next(v for v in _variants(new_state) if v[0] == name)
    land, _ring = _assert_superset(st, slot, eye_z, lt.lattice())
    assert land, "expected some landable tiles in the band sweep"
    seed = lt.seed_z(st, slot, eye_z)
    assert [t for t in land if lt.surface_bounds(st, *t)[1] < seed], "no below-eye tile"


@pytest.mark.parametrize("kind", ["plane", "band", "coarse"])
def test_landable_view_matches_exact(new_state, kind):
    """The candidate-driven view is bit-identical to the lattice's OWN FULL SWEEP -- the
    first landing ray's aim -- on landable, non-landable, object and outer-ring tiles.
    """
    st = new_state(0)
    _stack_and_transfer(st)
    slot = st.player
    grids = lt.lattice(kind == "plane", coarse=(kind == "coarse"))
    land = _landings(st, slot, grids)
    tiles = list(land)[:12]
    tiles += [(0, 0), (30, 30), (0, 15), (15, 30), (2, 3), (28, 27)]
    tiles += [(int(st.obj_x[slot]), int(st.obj_y[slot]))]
    for tile in tiles:
        rays = land.get(tuple(tile))
        want = None
        if rays is not None:
            h, v, cx, cy = los._meta_at(int(rays.min()), *grids)
            want = {"h_angle": h, "v_angle": v, "cursor": [cx, cy]}
        assert lt.landable_view(st, tile, slot, grids=grids) == want, tile


def _buildable(st, want):
    """The ``want`` bare, non-sloping tiles nearest the player -- where a create lands."""
    px, py = st.player_xy()
    near = []
    for x in range(mm.N):
        for y in range(mm.N):
            b = terrain.tile_byte(st, x, y)
            if b < mm.OBJECT_TILE and not b & 0x0F:
                near.append((abs(x - px) + abs(y - py), (x, y)))
    near.sort()
    return [t for _d, t in near[:want]]


def _hop(st):
    """Stack boulder+boulder+robot on the nearest buildable tile and transfer onto it."""
    st.energy = mm.ENERGY_MASK
    tile = _buildable(st, 1)[0]
    slot = None
    for otype in (mm.T_BOULDER, mm.T_BOULDER, mm.T_ROBOT):
        slot = actions.create(st, otype, tile)
    assert slot is not None and actions.transfer(st, slot)


def _midgame(new_state, number=321):
    """A board partway through a game: three hops (a raised eye on an object tile, a
    player slot that is not the starting one) plus loose boulders around it."""
    st = new_state(number)
    for _ in range(3):
        _hop(st)
    st.energy = mm.ENERGY_MASK
    for tile in _buildable(st, 8):
        actions.create(st, mm.T_BOULDER, tile)
    return st


@pytest.mark.parametrize("kind", ["plane", "band"])
def test_landable_view_matches_sweep_every_tile_midgame(new_state, kind):
    """EVERY tile of a MID-GAME board agrees with that lattice's own full sweep.

    Start-state sampling never reaches this case: the surface bracket the candidate
    filter rests on is exact on bare terrain and only approximate over object stacks,
    which exist only after the player has built.
    """
    st = _midgame(new_state)
    slot = st.player
    grids = lt.lattice(kind == "plane")
    land = _landings(st, slot, grids)
    for x in range(mm.N):
        for y in range(mm.N):
            rays = land.get((x, y))
            want = None
            if rays is not None:
                h, v, cx, cy = los._meta_at(int(rays.min()), *grids)
                want = {"h_angle": h, "v_angle": v, "cursor": [cx, cy]}
            assert lt.landable_view(st, (x, y), slot, grids=grids) == want, (x, y)


def test_views_answer_from_the_board_pinned_at_the_first_query(new_state):
    """``_Views`` answers every later query from the board pinned at its first one.

    A caller builds and transfers while holding the instance, so a per-tile march on the
    LIVE state would disagree with the sweep dict the same instance hands out -- both
    ``get`` and the whole-sweep readers must see one board.
    """
    st = _midgame(new_state)
    views = playerbase._Views(st)
    tiles = list(_landings(st, st.player, lt.lattice(True)))[:24]
    tiles += [(0, 0), (30, 30), (2, 3)]
    views.band_get(tiles[0])  # touch both lattices: each pins at ITS first query
    before = {t: views.get(t, band=True) for t in tiles}
    assert any(v is not None for v in before.values())
    _hop(st)  # the caller acts: new object tiles, a raised eye, a new observer slot
    assert {t: views.get(t, band=True) for t in tiles} == before
    plane, band = views.primary(), views.band()
    assert {t: plane.get(t, band.get(t)) for t in tiles} == before


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


@pytest.mark.parametrize("name", ["ls0", "ls42", "ls0-stacked", "ls42-eyez"])
def test_landable_set_matches_coarse_sweep(new_state, name):
    """The whole-board set is EXACTLY the coarse lattice's own sweep -- every tile
    including the outer ring, over bare boards, an object-tile stance and an eye_z
    override."""
    _name, st, slot, eye_z = next(v for v in _variants(new_state) if v[0] == name)
    got = lt.landable_set(st, slot, eye_z)
    assert got == set(_landings(st, slot, lt.lattice(coarse=True), eye_z))


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


def test_never_lands_shortcuts(new_state):
    """No ray in the full sweep lands on a sloping tile or the observer's own tile --
    the premise of :func:`landtable.never_lands`."""
    st = new_state(0)
    slot = st.player
    for tile in _landings(st, slot, lt.lattice(False)):
        assert not lt.never_lands(st, tile, slot), tile


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


def test_surface_bounds_bracket_the_stack(new_state):
    """Bare tiles bound the surface exactly; an object stack's bracket contains every
    object's own height."""
    st = new_state(0)
    for x in range(0, 31, 7):
        for y in range(0, 31, 7):
            b = terrain.tile_byte(st, x, y)
            lo, hi = lt.surface_bounds(st, x, y)
            if b < mm.OBJECT_TILE:
                assert lo == hi == ((b >> 4) & 0x0F) * 256
    tile = _stack_and_transfer(st)
    lo, hi = lt.surface_bounds(st, *tile)
    slot = terrain.top_object(st, *tile)
    seen = 0
    for _ in range(8):
        assert lo <= st.obj_z_height[slot] * 256 <= hi
        seen += 1
        flags = st.obj_flags[slot]
        if flags < 0x40:
            break
        slot = flags & 0x3F
    assert seen >= 3
