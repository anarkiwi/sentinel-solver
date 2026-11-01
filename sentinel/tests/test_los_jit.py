"""The numba fast-march (:func:`sentinel.los._march_jit`) is bit-for-bit identical
to the reference pure-Python march (:func:`sentinel.los._march_python`), and the
closed-form :func:`sentinel.los_jit.march` to the per-sub-step
:func:`_march_reference` kept here in all sixteen outputs.

The numba path now resolves object tiles (the recursive object-stack walk
$1E3F) entirely in numba, so these tests deliberately place rich object stacks --
lone boulders, boulder+synthoid stacks, trees, the platform tile, deep
multi-boulder stacks -- in the ray's path and sweep the full aim lattice, at
ground and elevated eye height, asserting the two marches never diverge.

If numba is absent the JIT path is never taken, so these tests skip.
"""

import time

import numba
import numpy as np
import pytest
from numba import njit

from sentinel import los, los_jit, landtable, actions, memmap as mm
from sentinel.game import Game
from sentinel.los import (
    _march_python,
    _march_jit,
    prepare_vector_from_player_sights,
    PITCH_BAND,
    SIGHTS_CX,
    SIGHTS_CY,
)
from sentinel.los_jit import (
    BLOCKED,
    LOS_CLEAR,
    _ADD_VECTOR,
    _ADD_VECTOR_NEG,
    _FLAT_ABOVE,
    _FLAT_ANGLE,
    _FLAT_ANGLE_SKIP,
    _FLAT_BELOW,
    _FLAT_BIT60,
    _FLAT_BIT60_HIT,
    _FLAT_BRANCH,
    _FLAT_DIFF,
    _FLAT_LOOKING_UP,
    _FLAT_LOOKUP,
    _FLAT_SAME,
    _FLAT_SAME_HIT,
    _FLAT_SAME_X_DIFF,
    _FLAT_SAME_Y,
    _FLAT_SAME_Y_DIFF,
    _FLAT_TOL,
    _FLAT_TOL_HIT,
    _LEAVE_SET,
    _MUL8_BIT,
    _OBJ_BT_BOULDER,
    _OBJ_BT_SKIP,
    _OBJ_BT_TREE,
    _OBJ_BT_TYPE,
    _OBJ_GHOL_LOOP,
    _OBJ_GHOL_RTS,
    _OBJ_HEAD_GHOL,
    _OBJ_HEAD_LOS,
    _OBJ_PLAT_RTS,
    _OBJ_PLAT_SKIP,
    _OBJ_SKIP_OTHER,
    _OBJ_SKIP_TREE,
    _OBJ_TARGET_HIT,
    _OBJ_TARGET_MISS,
    _OBJ_TREE_BELOW,
    _OBJ_TREE_HIGH,
    _OBJ_TREE_NEAR,
    _OBJ_TREE_SEEN,
    _OBJ_TREE_TARGETED,
    _OBJ_TYPE_BOULDER,
    _OBJ_TYPE_OTHER,
    _OBJ_TYPE_PLATFORM,
    _OBJ_TYPE_TREE,
    _OFLAGS,
    _OTYPE,
    _OZFRAC,
    _OZHEIGHT,
    _SLOPE_BRANCH,
    _SLOPE_EDGE_BLOCK,
    _SLOPE_EDGE_HIT,
    _SLOPE_EDGE_LDA,
    _SLOPE_EDGE_MISS,
    _SLOPE_HEAD,
    _SLOPE_NIB_12,
    _SLOPE_NIB_4,
    _SLOPE_NIB_QUAD,
    _SLOPE_Q_ABS,
    _SLOPE_Q_BLOCK,
    _SLOPE_Q_C1,
    _SLOPE_Q_C2,
    _SLOPE_Q_CORNER,
    _SLOPE_Q_CORNER_EOR,
    _SLOPE_Q_EDGE,
    _SLOPE_Q_INVERT,
    _SLOPE_Q_NEG,
    _SLOPE_Q_TAIL,
    _SLOPE_Q_USE_Y,
    _STEP_EDGE,
    _STEP_EDGE_EXIT,
    _STEP_SETUP,
    _TILE_ADDR,
    _TILE_Z_CALL,
    _TILE_Z_FLAT,
    _TILE_Z_OBJ,
    _TILE_Z_READ,
    _bits,
    _corner_at,
    _corner_z,
    _edge,
    _min_xy,
    _tile_byte,
)

pytestmark = pytest.mark.skipif(not los._HAVE_JIT, reason="numba not available")

SEEDS = [0, 1, 7, 42, 66, 999, 9999]


def _sweep_mismatches(state, slot, eye_z, max_steps):
    """Every (h, v) lattice aim where the two marches disagree on (tx, ty, los).

    Returns ``(bad, n)`` -- the list of disagreements and the comparison count."""
    do_los = state.mem[0x0C6E] & 0x7F
    bad = []
    n = 0
    for h in range(0, 256, 8):
        for v in PITCH_BAND:
            v1 = prepare_vector_from_player_sights(
                state, h, v, SIGHTS_CX, SIGHTS_CY, slot
            )
            r1 = _march_python(
                v1, state, slot, do_los_checks=do_los, eye_z=eye_z, max_steps=max_steps
            )
            v2 = prepare_vector_from_player_sights(
                state, h, v, SIGHTS_CX, SIGHTS_CY, slot
            )
            r2 = _march_jit(
                v2, state, slot, do_los_checks=do_los, eye_z=eye_z, max_steps=max_steps
            )
            n += 1
            if r1 != r2:
                bad.append((h, v, r1, r2))
    return bad, n


@pytest.mark.parametrize("seed", SEEDS)
@pytest.mark.parametrize("max_steps", [2000, 20000])
def test_jit_matches_python_bare(seed, max_steps):
    g = Game.new(seed)
    bad, _ = _sweep_mismatches(g.state, g.state.player, None, max_steps)
    assert not bad, f"seed {seed} ms {max_steps}: {bad[:3]}"


@pytest.mark.parametrize("seed", SEEDS)
def test_jit_matches_python_centre(seed):
    """The return_centre path (final px_sub/py_sub) matches too."""
    g = Game.new(seed)
    st, slot = g.state, g.state.player
    for h in range(0, 256, 8):
        for v in PITCH_BAND:
            los._HAVE_JIT = False
            try:
                a = los.aim_target(
                    st,
                    h,
                    v,
                    SIGHTS_CX,
                    SIGHTS_CY,
                    slot,
                    max_steps=2000,
                    return_centre=True,
                )
            finally:
                los._HAVE_JIT = True
            b = los.aim_target(
                st, h, v, SIGHTS_CX, SIGHTS_CY, slot, max_steps=2000, return_centre=True
            )
            assert a == b, f"seed {seed} aim h={h} v={v}: py={a} jit={b}"


# ---------------------------------------------------------------------------
# rich object-stack scenarios
# ---------------------------------------------------------------------------
RING8 = [(1, 0), (1, 1), (0, 1), (-1, 1), (-1, 0), (-1, -1), (0, -1), (1, -1)]


def _ring(typ, r):
    return [(dx * r, dy * r, typ) for dx, dy in RING8]


def _place(st, px, py, specs):
    """create each (dx, dy, type) around (px, py); creates silently no-op if the
    tile is off-board or the stack can't take it."""
    for dx, dy, typ in specs:
        x, y = px + dx, py + dy
        if 0 <= x < mm.N and 0 <= y < mm.N:
            actions.create(st, typ, (x, y))


def _scen_boulders(st, px, py):
    """Lone boulders in concentric rings."""
    _place(
        st,
        px,
        py,
        _ring(mm.T_BOULDER, 1) + _ring(mm.T_BOULDER, 2) + _ring(mm.T_BOULDER, 3),
    )


def _scen_stacks(st, px, py):
    """Boulder + synthoid stacks, plus trees and a platform tile."""
    for dx, dy in RING8:
        x, y = px + dx, py + dy
        if 0 <= x < mm.N and 0 <= y < mm.N:
            actions.create(st, mm.T_BOULDER, (x, y))
            actions.create(st, mm.T_ROBOT, (x, y))  # stacks on the boulder
    _place(
        st,
        px,
        py,
        [
            (2, 0, mm.T_TREE),
            (0, 2, mm.T_TREE),
            (2, 2, mm.T_PLATFORM),
            (3, 0, mm.T_TREE),
        ],
    )


def _scen_deep(st, px, py):
    """Deep multi-boulder stacks with a synthoid on top."""
    for dx, dy in [(1, 0), (0, 1), (2, 0), (0, 2), (1, 1), (-1, 0), (0, -1)]:
        x, y = px + dx, py + dy
        if 0 <= x < mm.N and 0 <= y < mm.N:
            actions.create(st, mm.T_BOULDER, (x, y))
            actions.create(st, mm.T_BOULDER, (x, y))
            actions.create(st, mm.T_BOULDER, (x, y))
            actions.create(st, mm.T_ROBOT, (x, y))


def _scen_elevated(st, px, py):
    """Objects all around, viewed from an elevated eye (as after a transfer up)."""
    _place(
        st,
        px,
        py,
        _ring(mm.T_BOULDER, 1) + _ring(mm.T_TREE, 2) + _ring(mm.T_BOULDER, 3),
    )
    return (st.obj_z_height[st.player] + 2) & 0xFF


SCENARIOS = [_scen_boulders, _scen_stacks, _scen_deep, _scen_elevated]


@pytest.mark.parametrize("seed", SEEDS)
@pytest.mark.parametrize("scen", SCENARIOS)
@pytest.mark.parametrize("max_steps", [2000, 20000])
def test_jit_matches_python_object_stacks(seed, scen, max_steps):
    """Rich object stacks in the ray path: the numba object-stack resolution stays
    bit-for-bit identical to the pure-Python oracle across the full aim lattice."""
    g = Game.new(seed)
    st = g.state
    st.energy = 9999  # guarantee every create lands, regardless of energy
    px, py = st.player_xy()
    eye_z = scen(st, px, py)
    bad, n = _sweep_mismatches(st, st.player, eye_z, max_steps)
    assert n == len(range(0, 256, 8)) * len(PITCH_BAND)
    assert not bad, f"seed {seed} {scen.__name__} ms {max_steps}: {bad[:5]}"


@njit
def _is_tree_cdd_reference(mem, Y, pz_sub, pz_whole, minxy, c56, cdd):
    """los._is_tree $1E69: the enemy-can-see-a-tree marker ($0CDD).  Works in a
    scratch byte, not $0079, so it only (maybe) sets $0CDD; the $1E8C compare reuses
    the $0074 the $1E48 call already left.  Returns (cdd, cycles)."""
    zf = int(mem[_OZFRAC + Y])
    t = zf - (pz_sub & 0xFF)
    s75 = t & 0xFF
    borrow = 1 if t < 0 else 0
    saved_hi = (int(mem[_OZHEIGHT + Y]) - (pz_whole & 0xFF) - borrow) & 0xFF
    t2 = s75 + 0xE0
    s75 = t2 & 0xFF
    carry = 1 if t2 > 0xFF else 0
    a = (saved_hi + carry) & 0xFF
    if a & 0x80:
        return cdd, _OBJ_TREE_BELOW
    c = a & 1
    a >>= 1
    s75 = ((s75 >> 1) | (c << 7)) & 0xFF
    c = a & 1
    a >>= 1
    if a != 0:
        return cdd, _OBJ_TREE_HIGH
    a = ((s75 >> 1) | (c << 7)) & 0xFF
    if a < minxy:
        return cdd, _OBJ_TREE_NEAR
    if c56 & 0x80:
        return cdd, _OBJ_TREE_TARGETED
    return ((cdd >> 1) | 0x80) & 0xFF, _OBJ_TREE_SEEN


@njit
def _object_surface_reference(
    mem, raw0, px_sub, py_sub, pz_sub, pz_whole, c58, c56, cdd
):
    """los._get_tile_z_from_object $1E3F and its helpers, flattened into one bounded
    iterative walk of the object stack.  Returns the 8-tuple
    ``(z, s79, c0c, c67, c56, cdd, s60, cycles)`` -- the object surface the flat-tile
    check compares against, the $0C56/$0CDD trackers, and the walk's own 6502 cost."""
    s60 = 0x80
    s79 = 0
    c0c = 0x80
    c67 = 0
    raw = raw0 & 0xFF
    cyc = np.int64(0)
    for _ in range(80):
        Y = raw & 0x3F
        do_ghol = False
        if (s60 & 0x80) == 0:
            # $1E44 BPL get_height_of_lowest_object
            cyc += _OBJ_HEAD_GHOL
            do_ghol = True
        else:
            # get_tile_z_for_line_of_sight $1E0E
            cyc += _OBJ_HEAD_LOS
            if Y == (c58 & 0xFF):
                c56 = ((c56 >> 1) | 0x80) & 0xFF
                cyc += _OBJ_TARGET_HIT
            else:
                cyc += _OBJ_TARGET_MISS
            otype = int(mem[_OTYPE + Y])
            if otype == 3 or otype == 2:
                cyc += _OBJ_TYPE_BOULDER if otype == 3 else _OBJ_TYPE_TREE
                # get_boulder_or_tree_z_for_line_of_sight $1E48
                go_skip = False
                frac, fcyc = _min_xy(px_sub, py_sub)
                cyc += fcyc
                if frac >= 0x40:
                    cyc += _OBJ_BT_SKIP
                    go_skip = True
                elif otype == 2:  # is_tree $1E69
                    cyc += _OBJ_BT_TYPE + _OBJ_BT_TREE
                    cdd, tcyc = _is_tree_cdd_reference(
                        mem, Y, pz_sub, pz_whole, frac, c56, cdd
                    )
                    cyc += tcyc
                    go_skip = True
                else:
                    # boulder near-centre $1E56: targetable, RTS with z
                    cyc += _OBJ_BT_TYPE + _OBJ_BT_BOULDER
                    c67 = ((c67 >> 1) | 0x80) & 0xFF
                    t = int(mem[_OZFRAC + Y]) - 0x60
                    s79 = t & 0xFF
                    borrow = 1 if t < 0 else 0
                    z = (int(mem[_OZHEIGHT + Y]) - borrow) & 0xFF
                    return (z, s79, c0c, c67, c56, cdd, s60, cyc)
                if go_skip:
                    # skip_targeting_object $1E99, then fall into ghol
                    if int(mem[_OTYPE + Y]) != 2:
                        s60 = 0xC0
                        cyc += _OBJ_SKIP_OTHER
                    else:
                        cyc += _OBJ_SKIP_TREE
                    do_ghol = True
            elif otype != 6:
                # $1E23 BNE ghol (robot/sentry/enemy)
                cyc += _OBJ_TYPE_OTHER
                do_ghol = True
            else:
                # platform (type 6) $1E25
                cyc += _OBJ_TYPE_PLATFORM
                frac, fcyc = _min_xy(px_sub, py_sub)
                cyc += fcyc
                if frac >= 0x64:
                    cyc += _OBJ_PLAT_SKIP
                    if int(mem[_OTYPE + Y]) != 2:
                        s60 = 0xC0
                        cyc += _OBJ_SKIP_OTHER
                    else:
                        cyc += _OBJ_SKIP_TREE
                    do_ghol = True
                else:
                    cyc += _OBJ_PLAT_RTS
                    c0c = 0x10
                    t = int(mem[_OZFRAC + Y]) + 0x20
                    s79 = t & 0xFF
                    carry = 1 if t > 0xFF else 0
                    z = (int(mem[_OZHEIGHT + Y]) + carry) & 0xFF
                    return (z, s79, c0c, c67, c56, cdd, s60, cyc)
        if do_ghol:
            # get_height_of_lowest_object $1EA4: stacked -> recurse on the object
            # beneath (raw = flags); else the bottom object's z_height.
            flags = int(mem[_OFLAGS + Y])
            if flags >= 0x40:
                cyc += _OBJ_GHOL_LOOP
                raw = flags
                continue
            cyc += _OBJ_GHOL_RTS
            return (int(mem[_OZHEIGHT + Y]), s79, c0c, c67, c56, cdd, s60, cyc)
    # safety: corrupt/deep stack
    return (int(mem[_OZHEIGHT + (raw & 0x3F)]), s79, c0c, c67, c56, cdd, s60, cyc)


@njit
def _slope_quad_reference(nib, p73, p74, p75, p76, px_sub, py_sub, pz_sub, pz_whole):
    """los._slope_corner_or_quad $1D8A-$1DEE, opcode-faithful.  Returns (verdict,
    cycles): 1 blocked (ray below the slope) or 0 loop (above it)."""
    cyc = np.int64(0)
    A = nib & 0xFF
    C = A & 1
    A >>= 1
    if C == 0:
        cyc += np.int64(_SLOPE_Q_C2)
        C = A & 1
        A >>= 1
        s78 = A
        C = A & 1
        A >>= 1
        A = px_sub & 0xFF
        if C != 0:
            A ^= 0xFF
            cyc += np.int64(_SLOPE_Q_CORNER_EOR)
        cyc += np.int64(_SLOPE_Q_CORNER)
        C = 1 if A >= (py_sub & 0xFF) else 0
        A = ((s78 << 1) | C) & 0xFF
        A = _edge(A)
    else:
        C = A & 1
        A >>= 1
        if C == 1:
            cyc += np.int64(_SLOPE_Q_C1)
            A = (A + 1 + C) & 0xFF
            A &= 0x03
            s78 = A
            C = A & 1
            A >>= 1
            A = px_sub & 0xFF
            if C != 0:
                A ^= 0xFF
                cyc += np.int64(_SLOPE_Q_CORNER_EOR)
            cyc += np.int64(_SLOPE_Q_CORNER)
            C = 1 if A >= (py_sub & 0xFF) else 0
            A = ((s78 << 1) | C) & 0xFF
            A = _edge(A)
        else:
            cyc += np.int64(_SLOPE_Q_EDGE)
            A = A & 1

    # use_edge_for_slope $1DAF
    cyc += np.int64(_SLOPE_Q_TAIL)
    X = A & 0xFF
    C = A & 1
    A >>= 1
    yreg = px_sub & 0xFF
    if C == 0:
        yreg = py_sub & 0xFF
        cyc += np.int64(_SLOPE_Q_USE_Y)
    C = A & 1
    A >>= 1
    A = yreg
    if C != 0:
        A ^= 0xFF
        cyc += np.int64(_SLOPE_Q_INVERT)
    s02 = A & 0xFF

    s78 = _corner_at(X & 3, p73, p74, p75, p76)
    a = (
        _corner_at((X & 3) + 1, p73, p74, p75, p76)
        - _corner_at(X & 3, p73, p74, p75, p76)
    ) & 0x1FF
    res = a & 0xFF
    neg = (res & 0x80) != 0
    if neg:
        res = ((res ^ 0xFF) + 1) & 0xFF
        cyc += np.int64(_SLOPE_Q_ABS)
    s75 = res
    cyc += np.int64(_MUL8_BIT) * _bits(s02)
    prod = (s02 & 0xFF) * (s75 & 0xFF)
    prod_h = (prod >> 8) & 0xFF
    prod_lo = prod & 0xFF
    if neg:
        val = ((prod_h & 0xFF) << 8) | (prod_lo & 0xFF)
        negv = (-val) & 0xFFFF
        prod_h = (negv >> 8) & 0xFF
        prod_lo = negv & 0xFF
        cyc += np.int64(_SLOPE_Q_NEG)
    s75b = (prod_h + s78) & 0xFF
    lo = (pz_sub & 0xFF) - prod_lo
    borrow = 1 if lo < 0 else 0
    hi8 = ((pz_whole & 0xFF) - s75b - borrow) & 0xFF
    if hi8 & 0x80:
        return 1, cyc + np.int64(_SLOPE_Q_BLOCK)  # blocked
    return 0, cyc  # loop


@njit
def _march_reference(
    mem,
    ax_lo,
    ax_hi,
    az_lo,
    az_hi,
    ay_lo,
    ay_hi,
    s30,
    px_frac,
    px_sub,
    px_whole,
    pz_frac,
    pz_sub,
    pz_whole,
    py_frac,
    py_sub,
    py_whole,
    ox,
    oy,
    c6e,
    c58,
    c56,
    cdd,
    max_steps,
):
    """The per-sub-step march :func:`los_jit.march` must reproduce exactly."""
    ty = 0
    tx = 0
    steps = 0
    cycles = np.int64(0)
    status = BLOCKED
    # Signed 16-bit per-axis vectors: each march sub-step adds this (sign-extended)
    # to the 24-bit (whole:sub:frac) position accumulator.  Used by the flat-tile
    # intra-tile fast-forward below.
    vx16 = ((ax_hi & 0xFF) << 8) | (ax_lo & 0xFF)
    if vx16 >= 0x8000:
        vx16 -= 0x10000
    vy16 = ((ay_hi & 0xFF) << 8) | (ay_lo & 0xFF)
    if vy16 >= 0x8000:
        vy16 -= 0x10000
    vz16 = ((az_hi & 0xFF) << 8) | (az_lo & 0xFF)
    if vz16 >= 0x8000:
        vz16 -= 0x10000
    # Per-tile cache: the tile byte and (for a sloping tile) the four corner
    # heights depend only on (tx,ty), not on the sub-step position, so they are
    # computed once per tile and reused across its ~tens of sub-steps.
    cur_tx = -1
    cur_ty = -1
    cb = 0
    cnib = 0
    cz = 0
    cp73 = 0
    cp74 = 0
    cp75 = 0
    cp76 = 0
    ccorner = np.int64(0)
    neg = 0
    if ax_hi & 0x80:
        neg += 1
    if az_hi & 0x80:
        neg += 1
    if ay_hi & 0x80:
        neg += 1
    step_fixed = np.int64(_ADD_VECTOR + _ADD_VECTOR_NEG * neg + 2 * _STEP_EDGE)
    # $1CFB..$1DFE: only reached once both edge tests fall through.
    tile_fixed = np.int64(_STEP_SETUP + _TILE_Z_CALL + _TILE_ADDR + _TILE_Z_READ)
    flat_step = (
        step_fixed
        + tile_fixed
        + np.int64(_TILE_Z_FLAT + _FLAT_BRANCH + _FLAT_DIFF + _FLAT_BELOW)
    )
    while steps < max_steps:
        steps += 1
        cycles += step_fixed  # $1CE8 JSR $1CBB: add_vector, then the edge tests
        # add_vector $1CBB: signed 24-bit step on x, z, y (order irrelevant --
        # the axes are independent).
        t = (px_frac & 0xFF) + (ax_lo & 0xFF)
        px_frac = t & 0xFF
        carry = 1 if t > 0xFF else 0
        ext = -1 if (ax_hi & 0x80) else 0
        t2 = (px_sub & 0xFF) + (ax_hi & 0xFF) + carry
        px_sub = t2 & 0xFF
        px_whole = (px_whole + ext + (1 if t2 > 0xFF else 0)) & 0xFF

        t = (pz_frac & 0xFF) + (az_lo & 0xFF)
        pz_frac = t & 0xFF
        carry = 1 if t > 0xFF else 0
        ext = -1 if (az_hi & 0x80) else 0
        t2 = (pz_sub & 0xFF) + (az_hi & 0xFF) + carry
        pz_sub = t2 & 0xFF
        pz_whole = (pz_whole + ext + (1 if t2 > 0xFF else 0)) & 0xFF

        t = (py_frac & 0xFF) + (ay_lo & 0xFF)
        py_frac = t & 0xFF
        carry = 1 if t > 0xFF else 0
        ext = -1 if (ay_hi & 0x80) else 0
        t2 = (py_sub & 0xFF) + (ay_hi & 0xFF) + carry
        py_sub = t2 & 0xFF
        py_whole = (py_whole + ext + (1 if t2 > 0xFF else 0)) & 0xFF

        tx = px_whole & 0xFF
        if tx >= 0x1F:
            cycles += np.int64(_STEP_EDGE_EXIT - _STEP_EDGE + _LEAVE_SET)
            status = BLOCKED
            break
        ty = py_whole & 0xFF
        if ty >= 0x1F:
            cycles += np.int64(_STEP_EDGE_EXIT + _LEAVE_SET)
            status = BLOCKED
            break

        cycles += tile_fixed
        if tx != cur_tx or ty != cur_ty:
            cur_tx = tx
            cur_ty = ty
            cb = _tile_byte(mem, tx, ty)
            cnib = cb & 0x0F
            cz = (cb >> 4) & 0x0F
            if cb < 0xC0 and cnib != 0:
                cp73 = cz
                cp76, c1 = _corner_z(mem, tx + 1, ty)
                cp75, c2 = _corner_z(mem, tx + 1, ty + 1)
                cp74, c3 = _corner_z(mem, tx, ty + 1)
                ccorner = c1 + c2 + c3
        b = cb
        if b >= 0xC0:
            cycles += np.int64(_TILE_Z_OBJ + _FLAT_BRANCH + _FLAT_DIFF)
            # object tile: stack surface $1E3F, then object-aware check_flat_tile $1D0D
            z, s79, c0c, c67, c56, cdd, s60, ocyc = _object_surface_reference(
                mem, b, px_sub, py_sub, pz_sub, pz_whole, c58, c56, cdd
            )
            cycles += ocyc
            t = (s79 & 0xFF) - (pz_sub & 0xFF)
            borrow = 1 if t < 0 else 0
            s79 = t & 0xFF
            d = ((z & 0xFF) - (pz_whole & 0xFF) - borrow) & 0xFF
            if d & 0x80:
                cycles += np.int64(_FLAT_BELOW)
                continue
            if d != 0:
                cycles += np.int64(_FLAT_ABOVE + _LEAVE_SET)
                status = BLOCKED
                break
            cycles += np.int64(_FLAT_TOL)
            if s79 >= (c0c & 0xFF):
                cycles += np.int64(_FLAT_TOL_HIT + _LEAVE_SET)
                status = BLOCKED
                break
            cycles += np.int64(_FLAT_BIT60)
            if s60 & 0x40:
                cycles += np.int64(_FLAT_BIT60_HIT + _LEAVE_SET)
                status = BLOCKED
                break
            cycles += np.int64(_FLAT_ANGLE)
            if ((c6e | c67) & 0x80) == 0:
                cycles += np.int64(_FLAT_LOOKUP)
                if (s30 & 0x80) == 0:  # looking up -> rejected
                    cycles += np.int64(_FLAT_LOOKING_UP + _LEAVE_SET)
                    status = BLOCKED
                    break
            else:
                cycles += np.int64(_FLAT_ANGLE_SKIP)
            cycles += np.int64(_FLAT_SAME)
            if tx != (ox & 0xFF):
                cycles += np.int64(_FLAT_SAME_X_DIFF)
                status = LOS_CLEAR
                break
            cycles += np.int64(_FLAT_SAME_Y)
            if ty == (oy & 0xFF):
                cycles += np.int64(_FLAT_SAME_HIT)
                continue
            cycles += np.int64(_FLAT_SAME_Y_DIFF)
            status = LOS_CLEAR
            break

        slope = cnib
        z = cz
        if slope == 0:
            cycles += np.int64(_TILE_Z_FLAT + _FLAT_BRANCH + _FLAT_DIFF)
            # check_flat_tile $1D0D, fast path (s79=0, tolerance $000C=$80,
            # $0060 bit6 clear, $0C67 clear).
            s79 = (0 - (pz_sub & 0xFF)) & 0xFF
            borrow = 1 if pz_sub & 0xFF else 0
            d = ((z & 0xFF) - (pz_whole & 0xFF) - borrow) & 0xFF
            if d & 0x80:
                # Tile below the ray -> keep marching.  The tile (tx,ty) is fixed
                # for a closed-form run of sub-steps (position accumulators are
                # linear); fast-forward that run, replaying ONLY the z surface
                # comparison per sub-step and skipping the redundant x/y add_vector,
                # edge tests and tile_byte re-read.  Bit-identical to per-sub-step.
                if tx == (ox & 0xFF) and ty == (oy & 0xFF):
                    continue
                xacc = ((px_sub & 0xFF) << 8) | (px_frac & 0xFF)
                yacc = ((py_sub & 0xFF) << 8) | (py_frac & 0xFF)
                if vx16 > 0:
                    nx = (0xFFFF - xacc) // vx16
                elif vx16 < 0:
                    nx = xacc // (-vx16)
                else:
                    nx = max_steps
                if vy16 > 0:
                    ny = (0xFFFF - yacc) // vy16
                elif vy16 < 0:
                    ny = yacc // (-vy16)
                else:
                    ny = max_steps
                n_tile = min(nx, ny, max_steps - steps)
                zacc = (
                    ((pz_whole & 0xFF) << 16)
                    | ((pz_sub & 0xFF) << 8)
                    | (pz_frac & 0xFF)
                )
                m = 0
                hit = 0
                while m < n_tile:
                    m += 1
                    zacc = (zacc + vz16) & 0xFFFFFF
                    pzw = (zacc >> 16) & 0xFF
                    pzs = (zacc >> 8) & 0xFF
                    dd = (z - pzw - (1 if pzs else 0)) & 0xFF
                    if dd & 0x80:
                        continue
                    hit = 1
                    break
                steps += m
                cycles += np.int64(_FLAT_BELOW) + m * flat_step  # replayed flat steps
                xacc = (xacc + m * vx16) & 0xFFFF
                yacc = (yacc + m * vy16) & 0xFFFF
                px_frac = xacc & 0xFF
                px_sub = (xacc >> 8) & 0xFF
                py_frac = yacc & 0xFF
                py_sub = (yacc >> 8) & 0xFF
                pz_frac = zacc & 0xFF
                pz_sub = (zacc >> 8) & 0xFF
                pz_whole = (zacc >> 16) & 0xFF
                if hit == 0:
                    continue  # left the tile (or hit budget) -> resume march
                # terminated inside the tile: flat verdict at this sub-step
                cycles -= np.int64(_FLAT_BELOW)
                s79 = (0 - (pz_sub & 0xFF)) & 0xFF
                if (
                    ((z & 0xFF) - (pz_whole & 0xFF) - (1 if pz_sub & 0xFF else 0))
                    & 0xFF
                ) != 0:
                    cycles += np.int64(_FLAT_ABOVE + _LEAVE_SET)
                    status = BLOCKED
                    break
                cycles += np.int64(_FLAT_TOL)
                if s79 >= 0x80:
                    cycles += np.int64(_FLAT_TOL_HIT + _LEAVE_SET)
                    status = BLOCKED
                    break
                cycles += np.int64(_FLAT_BIT60 + _FLAT_ANGLE)
                if (c6e & 0x80) == 0:
                    cycles += np.int64(_FLAT_LOOKUP)
                    if (s30 & 0x80) == 0:
                        cycles += np.int64(_FLAT_LOOKING_UP + _LEAVE_SET)
                        status = BLOCKED
                        break
                else:
                    cycles += np.int64(_FLAT_ANGLE_SKIP)
                cycles += np.int64(_FLAT_SAME)
                if tx != (ox & 0xFF):
                    cycles += np.int64(_FLAT_SAME_X_DIFF)
                else:
                    cycles += np.int64(_FLAT_SAME_Y + _FLAT_SAME_Y_DIFF)
                status = LOS_CLEAR
                break
            if d != 0:
                cycles += np.int64(_FLAT_ABOVE + _LEAVE_SET)
                status = BLOCKED
                break
            cycles += np.int64(_FLAT_TOL)
            if s79 >= 0x80:
                cycles += np.int64(_FLAT_TOL_HIT + _LEAVE_SET)
                status = BLOCKED
                break
            cycles += np.int64(_FLAT_BIT60 + _FLAT_ANGLE)
            if (c6e & 0x80) == 0:
                cycles += np.int64(_FLAT_LOOKUP)
                if (s30 & 0x80) == 0:  # looking up -> rejected
                    cycles += np.int64(_FLAT_LOOKING_UP + _LEAVE_SET)
                    status = BLOCKED
                    break
            else:
                cycles += np.int64(_FLAT_ANGLE_SKIP)
            cycles += np.int64(_FLAT_SAME)
            if tx != (ox & 0xFF):
                cycles += np.int64(_FLAT_SAME_X_DIFF)
                status = LOS_CLEAR
                break
            cycles += np.int64(_FLAT_SAME_Y)
            if ty == (oy & 0xFF):
                cycles += np.int64(_FLAT_SAME_HIT)
                continue  # same tile as the observer -> keep going
            cycles += np.int64(_FLAT_SAME_Y_DIFF)
            status = LOS_CLEAR
            break
        else:
            # check_sloping_tile $1D46, corner heights hoisted out of the sub-step loop
            cycles += np.int64(_TILE_Z_FLAT + _SLOPE_BRANCH + _SLOPE_HEAD) + ccorner
            if cnib == 0x04 or cnib == 0x0C:
                cycles += np.int64(
                    _SLOPE_NIB_4 if cnib == 0x04 else _SLOPE_NIB_12
                ) + np.int64(_SLOPE_EDGE_LDA)
                b8 = pz_whole & 0xFF
                hit = 0
                for corner in (cp73, cp74, cp75, cp76):
                    if b8 >= corner:
                        hit = 1
                        break
                    cycles += np.int64(_SLOPE_EDGE_MISS)
                if hit:
                    cycles += np.int64(_SLOPE_EDGE_HIT)
                    continue  # ray above the slope -> keep marching
                cycles += np.int64(_SLOPE_EDGE_BLOCK + _LEAVE_SET)
                status = BLOCKED
                break
            cycles += np.int64(_SLOPE_NIB_QUAD)
            verdict, qcyc = _slope_quad_reference(
                cnib, cp73, cp74, cp75, cp76, px_sub, py_sub, pz_sub, pz_whole
            )
            cycles += qcyc
            if verdict == 0:
                continue  # ray above the slope -> keep marching
            cycles += np.int64(_LEAVE_SET)  # $1D44 SEC + RTS
            status = BLOCKED
            break
    return (
        status,
        tx,
        ty,
        px_frac,
        px_sub,
        px_whole,
        pz_frac,
        pz_sub,
        pz_whole,
        py_frac,
        py_sub,
        py_whole,
        c56,
        cdd,
        steps,
        cycles,
    )


@njit
def _march_both(mem, vecs, seeds, obs):
    """Every ray through :func:`los_jit.march` and :func:`_march_reference`: two
    ``(n, 16)`` arrays of the full return tuples."""
    ax_lo, ax_hi, az_lo, az_hi, ay_lo, ay_hi, s30 = vecs
    n = ax_lo.shape[0]
    out = np.empty((2, n, 16), dtype=np.int64)
    for i in range(n):
        args = (
            ax_lo[i],
            ax_hi[i],
            az_lo[i],
            az_hi[i],
            ay_lo[i],
            ay_hi[i],
            s30[i],
            seeds[i, 0],
            seeds[i, 1],
            seeds[i, 2],
            seeds[i, 3],
            seeds[i, 4],
            seeds[i, 5],
            seeds[i, 6],
            seeds[i, 7],
            seeds[i, 8],
            obs[i, 0],
            obs[i, 1],
            obs[i, 2],
            obs[i, 3],
            obs[i, 4],
            obs[i, 5],
            obs[i, 6],
        )
        (
            out[0, i, 0],
            out[0, i, 1],
            out[0, i, 2],
            out[0, i, 3],
            out[0, i, 4],
            out[0, i, 5],
            out[0, i, 6],
            out[0, i, 7],
            out[0, i, 8],
            out[0, i, 9],
            out[0, i, 10],
            out[0, i, 11],
            out[0, i, 12],
            out[0, i, 13],
            out[0, i, 14],
            out[0, i, 15],
        ) = los_jit.march(mem, *args)
        (
            out[1, i, 0],
            out[1, i, 1],
            out[1, i, 2],
            out[1, i, 3],
            out[1, i, 4],
            out[1, i, 5],
            out[1, i, 6],
            out[1, i, 7],
            out[1, i, 8],
            out[1, i, 9],
            out[1, i, 10],
            out[1, i, 11],
            out[1, i, 12],
            out[1, i, 13],
            out[1, i, 14],
            out[1, i, 15],
        ) = _march_reference(mem, *args)
    return out


def _assert_same(mem, vecs, seeds, obs):
    """The closed-form and per-sub-step marches agree on every output of every ray;
    returns the per-sub-step results."""
    out = _march_both(mem, tuple(np.asarray(v) for v in vecs), seeds, obs)
    bad = np.flatnonzero((out[0] != out[1]).any(axis=1))
    assert bad.size == 0, (
        f"{bad.size} rays differ, first #{bad[0]}: "
        f"vec={[int(v[bad[0]]) for v in vecs]} seed={seeds[bad[0]].tolist()} "
        f"obs={obs[bad[0]].tolist()}\nnew={out[0, bad[0]].tolist()}\n"
        f"ref={out[1, bad[0]].tolist()}"
    )
    return out[1]


def _state_args(state, slot, eye_z, max_steps):
    seed = los._seed_position(state, slot, eye_z)
    obs = (
        state.obj_x[slot] & 0xFF,
        state.obj_y[slot] & 0xFF,
        state.mem[0x0C6E] & 0x7F,
        state.mem[0x0C58] & 0xFF,
        (state.mem[0x0C56] >> 1) & 0xFF,
        (state.mem[0x0CDD] >> 1) & 0xFF,
        max_steps,
    )
    return seed, obs


LATTICE_STRIDE = 29


@pytest.mark.parametrize("landscape", [0, 42, 335, 1442])
@pytest.mark.parametrize("eye_dz", [None, 0, 3])
def test_closed_form_matches_reference_lattice(landscape, eye_dz):
    """The keyboard lattice on real boards, from the eye and raised to it: all
    sixteen outputs, and the serial and threaded batch verdicts, match the reference."""
    st = Game.typed(landscape).state
    slot = st.player
    eye_z = None if eye_dz is None else int(st.obj_z_height[slot]) + eye_dz
    vecs = los._lattice_vectors(*landtable.lattice())
    idx = np.arange(landscape % LATTICE_STRIDE, vecs[0].size, LATTICE_STRIDE)
    sub = tuple(np.ascontiguousarray(v[idx]) for v in vecs)
    mem = np.frombuffer(st.mem, dtype=np.uint8)
    for max_steps in (landtable.MAX_STEPS, 97):
        seed, obs = _state_args(st, slot, eye_z, max_steps)
        n = idx.size
        ref = _assert_same(
            mem,
            sub,
            np.tile(np.array(seed, dtype=np.int64), (n, 1)),
            np.tile(np.array(obs, dtype=np.int64), (n, 1)),
        )
        centre = [los_jit._min_xy(a, b)[0] for a, b in ref[:, [4, 10]]]
        want = np.stack([ref[:, 0], ref[:, 1], ref[:, 2], np.array(centre)])
        seed_t = tuple(int(v) for v in seed)
        obs_t = tuple(int(v) for v in obs)
        for kernel in (los_jit._march_batch_serial, los_jit._march_batch_parallel):
            assert np.array_equal(kernel(mem, *sub, seed_t, obs_t), want)
        assert np.array_equal(
            np.stack(los_jit.march_batch(mem, *sub, *seed, *obs)), want
        )


def _random_component(rng, n):
    """Signed 16-bit components over every magnitude, with the extremes and zero."""
    v = rng.integers(-0x8000, 0x8000, n) >> rng.integers(0, 15, n)
    special = np.array([0, 1, -1, 0x7FFF, -0x8000, 0x100, -0x100, 0x80, -0x80])
    pick = rng.random(n) < 0.08
    v[pick] = rng.choice(special, int(pick.sum()))
    return v


def _random_rays(rng, n, z_hi):
    vx, vy, vz = (_random_component(rng, n) for _ in range(3))
    s30 = np.where(rng.random(n) < 0.5, (vz >> 8) & 0xFF, rng.integers(0, 256, n))
    vecs = (vx & 0xFF, (vx >> 8) & 0xFF, vz & 0xFF, (vz >> 8) & 0xFF)
    vecs = vecs + (vy & 0xFF, (vy >> 8) & 0xFF, s30)
    seeds = rng.integers(0, 256, (n, 9))
    seeds[:, 2] = rng.integers(0, 0x1F, n)
    seeds[:, 8] = rng.integers(0, 0x1F, n)
    seeds[:, 5] = rng.integers(0, z_hi, n)
    obs = rng.integers(0, 256, (n, 7))
    obs[:, 0] = np.where(rng.random(n) < 0.5, seeds[:, 2], rng.integers(0, 0x20, n))
    obs[:, 1] = np.where(rng.random(n) < 0.5, seeds[:, 8], rng.integers(0, 0x20, n))
    obs[:, 3] = rng.integers(0, 0x40, n)
    obs[:, 6] = rng.integers(1, 6001, n)
    return tuple(np.asarray(v, dtype=np.int64) for v in vecs), seeds, obs


@pytest.mark.parametrize("seed", range(6))
@pytest.mark.parametrize("z_hi", [18, 256])
def test_closed_form_matches_reference_random(seed, z_hi):
    """Random memory (object stacks, corner heights to $FF so slope deltas alias past
    +-127) and random rays of every magnitude, climbing and diving.  Odd seeds draw
    real object types and low heights, so stacks of trees, boulders and platforms."""
    rng = np.random.default_rng(seed)
    mem = rng.integers(0, 256, 0x10000).astype(np.uint8)
    if seed & 1:
        mem[los_jit._OTYPE : los_jit._OTYPE + 0x40] = rng.integers(0, 7, 0x40)
        mem[los_jit._OZHEIGHT : los_jit._OZHEIGHT + 0x40] = rng.integers(0, 18, 0x40)
        flags = rng.integers(0, 0x40, 0x40) | (rng.integers(0, 2, 0x40) * 0xC0)
        mem[los_jit._OFLAGS : los_jit._OFLAGS + 0x40] = flags
    vecs, seeds, obs = _random_rays(rng, 20000, z_hi)
    ref = _assert_same(mem, vecs, seeds, obs)
    assert len(np.unique(ref[:, 0])) == 2


@pytest.mark.parametrize("landscape", [7, 335])
def test_closed_form_matches_reference_board_random_rays(landscape):
    """Random rays (every heading and pitch, grazing included) over a real board,
    where long flat and sloping runs dominate."""
    rng = np.random.default_rng(landscape)
    mem = np.frombuffer(Game.typed(landscape).state.mem, dtype=np.uint8)
    vecs, seeds, obs = _random_rays(rng, 40000, 16)
    _assert_same(mem, vecs, seeds, obs)


def _flat_run_brute(z, za, vz, n):
    for m in range(1, n + 1):
        zz = (za + m * vz) & 0xFFFFFF
        if ((z - (zz >> 16) - (1 if (zz >> 8) & 0xFF else 0)) & 0x80) == 0:
            return m - 1
    return n


def test_flat_run_closed_form():
    """:func:`los_jit._flat_run` stops at the first non-below sub-step from any below
    start, across both the 0x8000 alias and the 0 boundary."""
    rng = np.random.default_rng(1)
    hits = 0
    for _ in range(4000):
        z = int(rng.integers(0, 16))
        za = int(rng.integers(0, 1 << 24))
        if (z - (za >> 16) - (1 if (za >> 8) & 0xFF else 0)) & 0x80 == 0:
            continue
        vz = int(_random_component(rng, 1)[0])
        n = int(rng.integers(0, 600))
        want = _flat_run_brute(z, za, vz, n)
        assert los_jit._flat_run(z, za, vz, n) == want, (z, za, vz, n)
        hits += want < n
    assert hits > 100


def test_popcount_sum():
    rng = np.random.default_rng(2)
    for _ in range(2000):
        r = int(rng.integers(1, 300))
        v = int(_random_component(rng, 1)[0])
        lo, hi = max(0, -v * r), min(0xFFFF, 0xFFFF - v * r)
        if lo > hi:
            continue
        c = int(rng.integers(lo, hi + 1))
        want = sum(bin((c + j * v) >> 8).count("1") for j in range(1, r + 1))
        assert los_jit._popcount_sum(c, v, r) == want


def test_march_batch_restores_threads():
    st = Game.typed(335).state
    vecs = los._lattice_vectors(*landtable.lattice())
    n = 4 * los_jit.BATCH_GRAIN
    sub = tuple(np.ascontiguousarray(v[:n]) for v in vecs)
    seed, obs = _state_args(st, st.player, None, landtable.MAX_STEPS)
    before = numba.get_num_threads()
    los_jit.march_batch(np.frombuffer(st.mem, dtype=np.uint8), *sub, *seed, *obs)
    assert numba.get_num_threads() == before


def _best(fn, args, reps):
    fn(*args)
    t = []
    for _ in range(reps):
        t0 = time.perf_counter()
        fn(*args)
        t.append(time.perf_counter() - t0)
    return min(t)


def batch_grain(landscape=335, reps=7):
    """Rays per thread at the fastest thread count, the median over the lattice batches
    of 2^6..2^17 rays too small to use every thread (``python -m
    sentinel.tests.test_los_jit``; sets ``los_jit.BATCH_GRAIN``)."""
    st = Game.typed(landscape).state
    mem = np.frombuffer(st.mem, dtype=np.uint8)
    vecs = los._lattice_vectors(*landtable.lattice())
    seed, obs = _state_args(st, st.player, None, landtable.MAX_STEPS)
    tail = (tuple(int(v) for v in seed), tuple(int(v) for v in obs))
    rng = np.random.default_rng(0)
    top = numba.get_num_threads()
    ladder = sorted({1 << k for k in range(top.bit_length())} | {top})
    grains = []
    for k in range(6, 18):
        idx = rng.choice(vecs[0].size, 1 << k, replace=False)
        args = (mem,) + tuple(np.ascontiguousarray(v[idx]) for v in vecs) + tail
        times = []
        for p in ladder:
            numba.set_num_threads(p)
            try:
                kernel = (
                    los_jit._march_batch_serial
                    if p == 1
                    else los_jit._march_batch_parallel
                )
                times.append(_best(kernel, args, reps))
            finally:
                numba.set_num_threads(top)
        best = ladder[int(np.argmin(times))]
        if best < top:
            grains.append((1 << k) // best)
        row = " ".join(f"{p}:{t * 1e3:.2f}" for p, t in zip(ladder, times))
        print(f"{1 << k:7d} rays, best {best:3d} threads, ms by threads {row}")
    grain = int(np.median(grains)) if grains else 1
    print(f"BATCH_GRAIN = {grain}")
    return grain


if __name__ == "__main__":
    batch_grain()
