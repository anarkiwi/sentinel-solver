"""Numba fast-march for the line-of-sight ray (:func:`sentinel.los.march`).

A bit-exact numba transcription of the HOT part of the ROM ray-march
(check_for_line_of_sight_to_tile $1CDD): the per-sub-step ``add_vector``, the
board-edge test, and the flat-tile / sloping-tile surface comparison, all over a
raw ``numpy.uint8`` view of the 64 KB memory image.

It reproduces exactly the pure-Python march in :mod:`sentinel.los` for every tile
EXCEPT a *primary object tile* (the ray's current tile byte >= $C0): that case
walks the recursive object stack (get_tile_z_from_object $1E3F), which stays in
tested Python.

Return status:
  * ``LOS_CLEAR`` (1) -- the ray reached a visible tile (ROM carry clear).
  * ``BLOCKED``   (0) -- tile above the ray / board edge / max_steps (no LOS).

Every integer is kept in a Python-int-width (int64) register and masked with
``& 0xFF`` exactly where the 6502 truncates, so uint8 wrap never corrupts a byte.
"""

import numpy as np

from sentinel import jitcache

jitcache.install()  # must precede numba: the cache key carries the cost constants

from numba import njit, prange  # noqa: E402  pylint: disable=wrong-import-position

from sentinel import passcost  # noqa: E402

LOS_CLEAR = 1
BLOCKED = 0

# Per-sub-step 6502 cost, bound as njit-visible globals (sentinel.passcost).
_ADD_VECTOR = passcost.ADD_VECTOR
_ADD_VECTOR_NEG = passcost.ADD_VECTOR_NEG
_STEP_EDGE = passcost.STEP_EDGE
_STEP_EDGE_EXIT = passcost.STEP_EDGE_EXIT
_STEP_SETUP = passcost.STEP_SETUP
_TILE_Z_CALL = passcost.TILE_Z_CALL
_TILE_ADDR = passcost.TILE_ADDR
_TILE_Z_READ = passcost.TILE_Z_READ
_TILE_Z_FLAT = passcost.TILE_Z_FLAT
_TILE_Z_OBJ = passcost.TILE_Z_OBJ
_LEAVE_SET = passcost.LEAVE_SET
_FLAT_BRANCH = passcost.FLAT_BRANCH
_FLAT_DIFF = passcost.FLAT_DIFF
_FLAT_BELOW = passcost.FLAT_BELOW
_FLAT_ABOVE = passcost.FLAT_ABOVE
_FLAT_TOL = passcost.FLAT_TOL
_FLAT_TOL_HIT = passcost.FLAT_TOL_HIT
_FLAT_BIT60 = passcost.FLAT_BIT60
_FLAT_BIT60_HIT = passcost.FLAT_BIT60_HIT
_FLAT_ANGLE = passcost.FLAT_ANGLE
_FLAT_ANGLE_SKIP = passcost.FLAT_ANGLE_SKIP
_FLAT_LOOKUP = passcost.FLAT_LOOKUP
_FLAT_LOOKING_UP = passcost.FLAT_LOOKING_UP
_FLAT_SAME = passcost.FLAT_SAME
_FLAT_SAME_X_DIFF = passcost.FLAT_SAME_X_DIFF
_FLAT_SAME_Y = passcost.FLAT_SAME_Y
_FLAT_SAME_HIT = passcost.FLAT_SAME_HIT
_FLAT_SAME_Y_DIFF = passcost.FLAT_SAME_Y_DIFF
_SLOPE_BRANCH = passcost.SLOPE_BRANCH
_SLOPE_HEAD = passcost.SLOPE_HEAD
_SLOPE_NIB_4 = passcost.SLOPE_NIB_4
_SLOPE_NIB_12 = passcost.SLOPE_NIB_12
_SLOPE_NIB_QUAD = passcost.SLOPE_NIB_QUAD
_SLOPE_EDGE_LDA = passcost.SLOPE_EDGE_LDA
_SLOPE_EDGE_MISS = passcost.SLOPE_EDGE_MISS
_SLOPE_EDGE_HIT = passcost.SLOPE_EDGE_HIT
_SLOPE_EDGE_BLOCK = passcost.SLOPE_EDGE_BLOCK
_SLOPE_Q_C2 = passcost.SLOPE_Q_C2
_SLOPE_Q_C1 = passcost.SLOPE_Q_C1
_SLOPE_Q_EDGE = passcost.SLOPE_Q_EDGE
_SLOPE_Q_CORNER = passcost.SLOPE_Q_CORNER
_SLOPE_Q_CORNER_EOR = passcost.SLOPE_Q_CORNER_EOR
_SLOPE_Q_TAIL = passcost.SLOPE_Q_TAIL
_SLOPE_Q_USE_Y = passcost.SLOPE_Q_USE_Y
_SLOPE_Q_INVERT = passcost.SLOPE_Q_INVERT
_SLOPE_Q_ABS = passcost.SLOPE_Q_ABS
_SLOPE_Q_NEG = passcost.SLOPE_Q_NEG
_SLOPE_Q_BLOCK = passcost.SLOPE_Q_BLOCK
_OBJ_HEAD_GHOL = passcost.OBJ_HEAD_GHOL
_OBJ_HEAD_LOS = passcost.OBJ_HEAD_LOS
_OBJ_TARGET_HIT = passcost.OBJ_TARGET_HIT
_OBJ_TARGET_MISS = passcost.OBJ_TARGET_MISS
_OBJ_TYPE_BOULDER = passcost.OBJ_TYPE_BOULDER
_OBJ_TYPE_TREE = passcost.OBJ_TYPE_TREE
_OBJ_TYPE_OTHER = passcost.OBJ_TYPE_OTHER
_OBJ_TYPE_PLATFORM = passcost.OBJ_TYPE_PLATFORM
_MINXY = passcost.MINXY
_MINXY_ABS = passcost.MINXY_ABS
_MINXY_Y_WINS = passcost.MINXY_Y_WINS
_OBJ_BT_SKIP = passcost.OBJ_BT_SKIP
_OBJ_BT_TYPE = passcost.OBJ_BT_TYPE
_OBJ_BT_TREE = passcost.OBJ_BT_TREE
_OBJ_BT_BOULDER = passcost.OBJ_BT_BOULDER
_OBJ_PLAT_SKIP = passcost.OBJ_PLAT_SKIP
_OBJ_PLAT_RTS = passcost.OBJ_PLAT_RTS
_OBJ_TREE_BELOW = passcost.OBJ_TREE_BELOW
_OBJ_TREE_HIGH = passcost.OBJ_TREE_HIGH
_OBJ_TREE_NEAR = passcost.OBJ_TREE_NEAR
_OBJ_TREE_TARGETED = passcost.OBJ_TREE_TARGETED
_OBJ_TREE_SEEN = passcost.OBJ_TREE_SEEN
_OBJ_SKIP_TREE = passcost.OBJ_SKIP_TREE
_OBJ_SKIP_OTHER = passcost.OBJ_SKIP_OTHER
_OBJ_GHOL_LOOP = passcost.OBJ_GHOL_LOOP
_OBJ_GHOL_RTS = passcost.OBJ_GHOL_RTS
_MUL8 = passcost.MUL8
_MUL8_LOW = passcost.MUL8_LOW
_MUL8_BIT = passcost.MUL8_BIT
_MUL_DBL_BYTE = passcost.MUL_DBL_BYTE
_MUL_DBL_BYTE_CARRY = passcost.MUL_DBL_BYTE_CARRY
_MUL_DBL_BYTE_CALL = passcost.MUL_DBL_BYTE_CALL
_MUL_PI = passcost.MUL_PI
_SIN_COS = passcost.SIN_COS
_SIN_COS_PI_CALL = passcost.SIN_COS_PI_CALL
_SIN_COS_QUAD = passcost.SIN_COS_QUAD
_SIN_COS_LOW = passcost.SIN_COS_LOW
_SIN_COS_HIGH = passcost.SIN_COS_HIGH
_SIN_COS_ONE = passcost.SIN_COS_ONE
_SIN_COS_TWO = passcost.SIN_COS_TWO
_SIN_COS_TWO_CLAMP = passcost.SIN_COS_TWO_CLAMP
_SIN_COS_NEXT = passcost.SIN_COS_NEXT
_SIN_COS_DONE = passcost.SIN_COS_DONE
_SIN_COS_SIGNS = passcost.SIN_COS_SIGNS
_SIN_COS_SIGN = passcost.SIN_COS_SIGN
_PROC_SC = passcost.PROC_SC
_PROC_SC_INVERT = passcost.PROC_SC_INVERT
_MUL_DBL_DBL = passcost.MUL_DBL_DBL
_MUL_DBL_DBL_NEG_Y = passcost.MUL_DBL_DBL_NEG_Y
_MUL_DBL_DBL_ODD = passcost.MUL_DBL_DBL_ODD
_MUL_DBL_DBL_CARRY = passcost.MUL_DBL_DBL_CARRY
_MUL_DBL_DBL_NEG = passcost.MUL_DBL_DBL_NEG
_PREP_VEC = passcost.PREP_VEC

# Object-array bases in the 64 KB image (sentinel.memmap), inlined so the njit
# code needs no Python object: OBJECTS_FLAGS $0100, OBJECTS_Z_HEIGHT $0940,
# OBJECTS_Z_FRACTION $0A00, OBJECTS_TYPE $0A40.
_OFLAGS = 0x0100
_OZHEIGHT = 0x0940
_OZFRAC = 0x0A00
_OTYPE = 0x0A40


@njit(cache=True, inline="always")
def _tile_byte(mem, x, y):
    """terrain.tile_byte: the raw tiles_table byte via the ROM address arithmetic
    ``page=(x&3)+4``, ``lo=((x<<3)&0xE0)|(y&0x1F)``."""
    xx = x & 0xFF
    yy = y & 0xFF
    lo = ((xx << 3) & 0xE0) | (yy & 0x1F)
    page = (xx & 3) + 4
    return int(mem[page * 256 + lo])


@njit(cache=True)
def _corner_z(mem, x, y):
    """los._slope_corner_z: one $1D4E/$1D55/$1D5C corner read.  Returns (z, cycles)."""
    cyc = np.int64(_TILE_Z_CALL + _TILE_ADDR + _TILE_Z_READ)
    b = _tile_byte(mem, x, y)
    if b < 0xC0:
        return (b >> 4) & 0x0F, cyc + np.int64(_TILE_Z_FLAT)
    cyc += np.int64(_TILE_Z_OBJ)
    o = b & 0x3F
    for _ in range(64):
        cyc += np.int64(_OBJ_HEAD_GHOL)
        f = int(mem[_OFLAGS + o])
        if f < 0x40:
            break
        cyc += np.int64(_OBJ_GHOL_LOOP)
        o = f & 0x3F
    return int(mem[_OZHEIGHT + o]), cyc + np.int64(_OBJ_GHOL_RTS)


@njit(cache=True, inline="always")
def _edge(y):
    """The $1DF1-$1DF8 edge table."""
    if y == 0:
        return 0x00
    elif y == 1:
        return 0x03
    elif y == 2:
        return 0x01
    elif y == 3:
        return 0x00
    elif y == 4:
        return 0x01
    elif y == 5:
        return 0x02
    elif y == 6:
        return 0x02
    return 0x03  # y == 7


@njit(cache=True, inline="always")
def _corner_at(i, p73, p74, p75, p76):
    """The zero-page corner square $73..$77 == [p73,p74,p75,p76,p73], indexed."""
    if i == 0:
        return p73
    elif i == 1:
        return p74
    elif i == 2:
        return p75
    elif i == 3:
        return p76
    return p73  # i == 4


@njit(cache=True)
def _slope_quad(nib, p73, p74, p75, p76, px_sub, py_sub, pz_sub, pz_whole):
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


@njit(cache=True, inline="always")
def _min_xy(px_sub, py_sub):
    """los._get_min_xy_fraction $1EAF: min tile-centre fraction of x/y (the exact
    6502 form, not plain abs).  Returns (fraction, cycles)."""
    cyc = _MINXY
    ax = (px_sub - 0x80) & 0xFF
    if ax & 0x80:
        ax ^= 0xFF
        cyc += _MINXY_ABS
    t74 = ax & 0xFF
    ay = (py_sub - 0x80) & 0xFF
    if ay & 0x80:
        ay ^= 0xFF
        cyc += _MINXY_ABS
    if ay >= t74:
        t74 = ay
    else:
        cyc += _MINXY_Y_WINS
    return t74 & 0xFF, cyc


@njit(cache=True)
def _is_tree_cdd(mem, Y, pz_sub, pz_whole, minxy, c56, cdd):
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


@njit(cache=True)
def _object_surface(mem, raw0, px_sub, py_sub, pz_sub, pz_whole, c58, c56, cdd):
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
                    cdd, tcyc = _is_tree_cdd(mem, Y, pz_sub, pz_whole, frac, c56, cdd)
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


@njit(cache=True)
def march(
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
    """March the ray from the given position for at most ``max_steps`` sub-steps.

    Fully self-contained: flat, sloping AND object tiles are resolved in numba, so
    the march never bails back to Python.  Returns the 16-tuple::

        (status, tx, ty,
         px_frac, px_sub, px_whole, pz_frac, pz_sub, pz_whole,
         py_frac, py_sub, py_whole, c56, cdd, steps_used, cycles)

    ``cycles`` is the march's 6502 cost: ``MARCH_STEP`` per sub-step plus
    ``MARCH_OBJECT``/``MARCH_SLOPE_EDGE``/``MARCH_SLOPE_QUAD`` for the branch taken.

    ``s30`` is the ray's vector_z high byte (the looking-up sign).  ``c6e`` is the
    do_line_of_sight_checks byte ($0C6E); bit7 waives the looking-up rejection.
    ``c58`` is the targeted-object slot; ``c56``/``cdd`` are the $0C56/$0CDD
    trackers (seeded/LSR'd by the caller, returned so the caller can persist them)."""
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
    step_fixed = np.int64(
        _ADD_VECTOR
        + _ADD_VECTOR_NEG * neg
        + 2 * _STEP_EDGE
        + _STEP_SETUP
        + _TILE_Z_CALL
        + _TILE_ADDR
        + _TILE_Z_READ
    )
    flat_step = step_fixed + np.int64(
        _TILE_Z_FLAT + _FLAT_BRANCH + _FLAT_DIFF + _FLAT_BELOW
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
            z, s79, c0c, c67, c56, cdd, s60, ocyc = _object_surface(
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
                cycles += np.int64(_SLOPE_EDGE_BLOCK)
                status = BLOCKED
                break
            cycles += np.int64(_SLOPE_NIB_QUAD)
            verdict, qcyc = _slope_quad(
                cnib, cp73, cp74, cp75, cp76, px_sub, py_sub, pz_sub, pz_whole
            )
            cycles += qcyc
            if verdict == 0:
                continue  # ray above the slope -> keep marching
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


@njit(cache=True, inline="always")
def _balance_stride(n):
    """A step coprime to ``n``, so ``k -> (k * stride) % n`` permutes the lattice and every
    prange chunk draws a spread of headings instead of one contiguous bundle.  Near n/phi,
    the classic low-discrepancy choice; walks up until gcd is 1, which is always reachable
    (a stride of 1 is the degenerate fallback for tiny n)."""
    if n < 3:
        return 1
    stride = int(n * 0.6180339887498949) | 1
    while stride > 1:
        a, b = stride, n
        while b != 0:
            a, b = b, a % b
        if a == 1:
            return stride
        stride -= 2
    return 1


@njit(cache=True, parallel=True)
def march_batch(
    mem,
    ax_lo,
    ax_hi,
    az_lo,
    az_hi,
    ay_lo,
    ay_hi,
    s30,
    px_frac0,
    px_sub0,
    px_whole0,
    pz_frac0,
    pz_sub0,
    pz_whole0,
    py_frac0,
    py_sub0,
    py_whole0,
    ox,
    oy,
    c6e,
    c58,
    c56,
    cdd,
    max_steps,
):
    """March a whole PRECOMPUTED lattice of ray vectors against one memory snapshot in
    a single numba call (the v-complete keyboard-aim sweep).  ``ax_lo..s30`` are equal-
    length arrays of per-aim vector components (state-INDEPENDENT -- built once from the
    aim params, see :func:`sentinel.los._lattice_vectors`); every ray starts from the
    SAME seed position (``px_frac0`` .. ``py_whole0``, from get_object_details $1ECC) and
    is marched by :func:`march`.  Returns per-ray ``(status, tx, ty, centre)`` where
    ``centre`` is the final tile-centre fraction ($1EAF).

    Bit-for-bit identical to calling :func:`march` per aim: the $0C56/$0CDD trackers are
    seeded fresh (not threaded through memory between rays), which never changes the
    ``(status, tx, ty)`` verdict -- within a single march they feed only the $0CDD tree
    marker, not the LOS result."""
    n = ax_lo.shape[0]
    status = np.empty(n, dtype=np.int64)
    tx = np.empty(n, dtype=np.int64)
    ty = np.empty(n, dtype=np.int64)
    centre = np.empty(n, dtype=np.int64)
    stride = _balance_stride(n)  # interleave: contiguous chunks stall the join
    for k in prange(n):  # pylint: disable=not-an-iterable
        i = (k * stride) % n
        (
            st,
            txi,
            tyi,
            _pxf,
            pxs,
            _pxw,
            _pzf,
            _pzs,
            _pzw,
            _pyf,
            pys,
            _pyw,
            _c56,
            _cdd,
            _steps,
            _cycles,
        ) = march(
            mem,
            ax_lo[i],
            ax_hi[i],
            az_lo[i],
            az_hi[i],
            ay_lo[i],
            ay_hi[i],
            s30[i],
            px_frac0,
            px_sub0,
            px_whole0,
            pz_frac0,
            pz_sub0,
            pz_whole0,
            py_frac0,
            py_sub0,
            py_whole0,
            ox,
            oy,
            c6e,
            c58,
            c56,
            cdd,
            max_steps,
        )
        status[i] = st
        tx[i] = txi
        ty[i] = tyi
        centre[i] = _min_xy(pxs, pys)[0]
    return status, tx, ty, centre


# ============================================================================
# lattice ray-vector builder (numba twin of sentinel.los.prepare_vector_from_
# player_sights).  A batched, prange builder so the full 1px keyboard-cursor
# lattice can be constructed in a fraction of a second instead of ~1min of pure
# Python.  Bit-for-bit identical to the pure-Python path (locked by
# tests/test_landable.py::test_lattice_vectors_match_python).
# ============================================================================
@njit(cache=True, inline="always")
def _vmul8(a, b):
    p = (a & 0xFF) * (b & 0xFF)
    return (p >> 8) & 0xFF, p & 0xFF


@njit(cache=True, inline="always")
def _vinvert16(high, frac):
    val = ((high & 0xFF) << 8) | (frac & 0xFF)
    neg = (-val) & 0xFFFF
    return (neg >> 8) & 0xFF, neg & 0xFF


@njit(cache=True, inline="always")
def _bits(v):
    """Set bits of a $0D03 multiplier: each costs MUL8_BIT more than a clear one."""
    n = np.int64(0)
    x = v & 0xFF
    while x:
        n += x & 1
        x >>= 1
    return n


@njit(cache=True)
def _vmul_dbl_by_byte(low74, high75, byte76):
    cyc = np.int64(_MUL_DBL_BYTE) + np.int64(_MUL8_BIT) * (_bits(low74) + _bits(byte76))
    r1h, _r1l = _vmul8(low74, high75)
    r2h, r2l = _vmul8(byte76, high75)
    res75 = r2h
    total = r1h + r2l
    res74 = total & 0xFF
    if total > 0xFF:
        res75 = (res75 + 1) & 0xFF
        cyc += _MUL_DBL_BYTE_CARRY
    return res74, res75, cyc


@njit(cache=True)
def _vmul_dbl_A_by_pi(A, frac74):
    frac = frac74 & 0xFF
    a = A & 0xFF
    for _ in range(2):
        c = (frac >> 7) & 1
        frac = (frac << 1) & 0xFF
        a = ((a << 1) | c) & 0xFF
    m76 = a
    cyc = np.int64(_MUL_PI + _MUL_DBL_BYTE) + np.int64(_MUL8_BIT) * (
        _bits(frac) + _bits(m76)
    )
    r1h, _r1l = _vmul8(frac, 0xC9)
    r77 = r1h
    r2h, _r2l = _vmul8(m76, 0xC9)
    r75 = r2h
    total = r77 + _r2l
    r74 = total & 0xFF
    if total > 0xFF:
        r75 = (r75 + 1) & 0xFF
        cyc += _MUL_DBL_BYTE_CARRY
    return r74, r75, cyc


@njit(cache=True)
def _vsin_cos(angle, frac74):
    cyc = np.int64(_SIN_COS + _SIN_COS_PI_CALL)
    angle &= 0xFF
    c0c = angle
    _apl, aPI_hi, mcyc = _vmul_dbl_A_by_pi(angle, frac74)
    cyc += np.int64(mcyc)
    c53 = _apl
    c54 = aPI_hi
    sixty = 1
    X = 0
    if c0c & 0x40:
        X = 1
        sixty = 0
        cyc += np.int64(_SIN_COS_QUAD)
    A_cmp = c54
    cur_c53 = c53
    cur_c54 = c54
    cur_75 = aPI_hi
    sc_low0 = 0
    sc_low1 = 0
    sc_high0 = 0
    sc_high1 = 0
    for _pass in range(2):  # X and sixty are always 0/1, so the $0EFD loop runs twice
        if (A_cmp & 0xFF) >= 0x7A:
            cyc += np.int64(_SIN_COS_HIGH + _SIN_COS_TWO + _MUL_DBL_BYTE_CALL)
            t74 = (0 - cur_c53) & 0xFF
            borrow1 = 1 if (0 - cur_c53) < 0 else 0
            v = 0xC9 - cur_c54 - borrow1
            t75 = v & 0xFF
            t76 = t75
            r74, r75, bcyc = _vmul_dbl_by_byte(t74, t75, t76)
            cyc += bcyc
            c = (r74 >> 7) & 1
            r74 = (r74 << 1) & 0xFF
            r75 = ((r75 << 1) | c) & 0xFF
            sub_lo = 0 - r74
            low = (sub_lo & 0xFF) & 0xFE
            borrow2 = 1 if sub_lo < 0 else 0
            cur_low = low
            sub_hi = 0 - r75 - borrow2
            if sub_hi < 0:
                cur_high = sub_hi & 0xFF
            else:
                cur_low = 0xFE
                cur_high = 0xFF
                cyc += np.int64(_SIN_COS_TWO_CLAMP)
        else:
            r_high1, _r1 = _vmul8(0xAB, cur_75)
            r_high2, r_low2 = _vmul8(r_high1, cur_75)
            t76 = r_high2
            cyc += np.int64(
                _SIN_COS_LOW + _SIN_COS_ONE + _MUL_DBL_BYTE_CALL + 2 * _MUL8
            ) + np.int64(_MUL8_BIT) * (
                5 + _bits(r_high1)
            )  # popcount($AB) = 5
            r74, r75, bcyc = _vmul_dbl_by_byte(r_low2, cur_75, t76)
            cyc += bcyc
            t74b = (cur_c53 - r74) & 0xFF
            borrow1 = 1 if cur_c53 < r74 else 0
            hv = (cur_c54 - r75 - borrow1) & 0xFF
            c = (t74b >> 7) & 1
            hi = ((hv << 1) | c) & 0xFF
            cur_high = hi
            cur_low = ((t74b << 1) & 0xFF) & 0xFE
        if X == 0:
            sc_low0 = cur_low
            sc_high0 = cur_high
        else:
            sc_low1 = cur_low
            sc_high1 = cur_high
        if X == sixty:
            cyc += np.int64(_SIN_COS_DONE)
            break
        cyc += np.int64(_SIN_COS_NEXT)
        X = sixty
        new_c53 = (0 - cur_c53) & 0xFF
        borrow = 1 if (0 - cur_c53) < 0 else 0
        new_c54 = (0xC9 - cur_c54 - borrow) & 0xFF
        cur_c53 = new_c53
        cur_c54 = new_c54
        cur_75 = new_c54
        A_cmp = new_c54
    sin_lo = sc_low0
    cos_lo = sc_low1
    sin_hi = sc_high0
    cos_hi = sc_high1
    cyc += np.int64(_SIN_COS_SIGNS)
    if c0c & 0x80:
        sin_lo |= 1
        cyc += np.int64(_SIN_COS_SIGN)
    t = ((c0c << 1) & 0xFF) ^ c0c
    if t & 0x80:
        cos_lo |= 1
        cyc += np.int64(_SIN_COS_SIGN)
    return sin_lo & 0xFF, cos_lo & 0xFF, sin_hi & 0xFF, cos_hi & 0xFF, cyc


@njit(cache=True, inline="always")
def _vproc_sc(low, high):
    t74 = low & 0xFF
    A = high & 0xFF
    # LSR A ; ROR $0074, four times; carry from the FIRST is the sign.
    c = A & 1
    A >>= 1
    saved = t74 & 1
    t74 = (t74 >> 1) | (c << 7)
    for _ in range(3):
        c = A & 1
        A >>= 1
        t74 = (t74 >> 1) | (c << 7)
    cyc = np.int64(_PROC_SC)
    if saved:
        A, t74 = _vinvert16(A, t74)
        cyc += _PROC_SC_INVERT
    return A & 0xFF, t74 & 0xFF, cyc


@njit(cache=True, inline="always")
def _vmul_dbl_dbl(x_lo, x_hi, y_lo, y_hi):
    s67 = 0
    s6a = y_lo & 0xFF
    s6b = y_hi & 0xFF
    s68 = x_lo & 0xFF
    s69 = x_hi & 0xFF
    cyc = np.int64(_MUL_DBL_DBL) + np.int64(_MUL8_BIT) * (_bits(s68) + 2 * _bits(s69))
    if s6b & 0x80:
        neg = (-(((s6b << 8) | s6a))) & 0xFFFF
        s6a = neg & 0xFF
        s6b = (neg >> 8) & 0xFF
        s67 ^= 0x80
        cyc += _MUL_DBL_DBL_NEG_Y
    if s68 & 1:
        s67 ^= 0x80
        cyc += _MUL_DBL_DBL_ODD
    r1h, xl_yh_low = _vmul8(s68, s6b)
    r77 = r1h
    rounded = xl_yh_low + 0x80
    r76 = rounded & 0xFF
    if rounded > 0xFF:
        r77 = (r77 + 1) & 0xFF
        cyc += _MUL_DBL_DBL_CARRY
    r2h, r2l = _vmul8(s69, s6b)
    r78 = r2h
    total = r2l + r77
    r77 = total & 0xFF
    if total > 0xFF:
        r78 = (r78 + 1) & 0xFF
        cyc += _MUL_DBL_DBL_CARRY
    r3h, r3l = _vmul8(s69, s6a)
    t = r3l + r76
    carry = 1 if t > 0xFF else 0
    t2 = r3h + r77 + carry
    res74 = t2 & 0xFF
    if t2 > 0xFF:
        r78 = (r78 + 1) & 0xFF
        cyc += _MUL_DBL_DBL_CARRY
    A = r78 & 0xFF
    frac = res74 & 0xFF
    if s67 & 0x80:
        A, frac = _vinvert16(A, frac)
        cyc += _MUL_DBL_DBL_NEG
    return A & 0xFF, frac & 0xFF, cyc


@njit(cache=True)
def _prep_vec(h_angle, v_angle, cur_x, cur_y):
    """Numba twin of prepare_vector_from_player_sights $1C10 + prepare_vector_from_
    angle $1C54.  Returns (vx_lo, vx_hi, vz_lo, vz_hi, vy_lo, vy_hi, s30)."""
    cc6 = cur_x & 0xFF
    s75 = cc6
    A = 0
    for _ in range(3):
        c = s75 & 1
        s75 >>= 1
        A = (A >> 1) | (c << 7)
    h_frac = A & 0xFF
    val = s75 + (h_angle & 0xFF)
    h_angle_v = (val - 0x0A) & 0xFF
    s75 = (cur_y - 0x05) & 0xFF
    A = 0
    for _ in range(4):
        c = s75 & 1
        s75 >>= 1
        A = (A >> 1) | (c << 7)
    v_frac = (A + 0x20) & 0xFF
    s74 = v_frac
    carry_in = 1 if (A + 0x20) > 0xFF else 0
    val2 = s75 + (v_angle & 0xFF) + carry_in
    v_angle_v = ((val2 & 0xFF) + 0x03) & 0xFF
    # prepare_vector_from_angle $1C54
    sin_lo_v, cos_lo_v, sin_hi_v, cos_hi_v, _c = _vsin_cos(v_angle_v, s74)
    _s33, s32, _c = _vproc_sc(cos_lo_v, cos_hi_v)
    s30, s2d, _c = _vproc_sc(sin_lo_v, sin_hi_v)
    h_sin_lo, h_cos_lo, h_sin_hi, h_cos_hi, _c = _vsin_cos(h_angle_v, h_frac)
    vy_hi, vy_lo, _c = _vmul_dbl_dbl(h_cos_lo, h_cos_hi, s32, _s33)
    vx_hi, vx_lo, _c = _vmul_dbl_dbl(h_sin_lo, h_sin_hi, s32, _s33)
    return vx_lo, vx_hi, s2d, s30, vy_lo, vy_hi, s30


@njit(cache=True, parallel=True)
def build_lattice(hgrid, vgrid, cxs, cys):
    """Build every keyboard-lattice ray vector, order ``for v: for h: for cx: for cy``
    (uniform cursor grid at all pitches).  Returns the six int16 component arrays plus
    s30, matching :func:`_prep_vec` per index; the grids reconstruct (h, v, cx, cy)."""
    nv = vgrid.shape[0]
    nh = hgrid.shape[0]
    ncx = cxs.shape[0]
    ncy = cys.shape[0]
    per_h = ncx * ncy
    per_v = nh * per_h
    n = nv * per_v
    vx_lo = np.empty(n, dtype=np.int16)
    vx_hi = np.empty(n, dtype=np.int16)
    vz_lo = np.empty(n, dtype=np.int16)
    vz_hi = np.empty(n, dtype=np.int16)
    vy_lo = np.empty(n, dtype=np.int16)
    vy_hi = np.empty(n, dtype=np.int16)
    s30 = np.empty(n, dtype=np.int16)
    for idx in prange(n):  # pylint: disable=not-an-iterable
        vi = idx // per_v
        rem = idx - vi * per_v
        hi = rem // per_h
        rem2 = rem - hi * per_h
        cxi = rem2 // ncy
        cyi = rem2 - cxi * ncy
        a, b, cl, ch, dl, dh, s = _prep_vec(hgrid[hi], vgrid[vi], cxs[cxi], cys[cyi])
        vx_lo[idx] = a
        vx_hi[idx] = b
        vz_lo[idx] = cl
        vz_hi[idx] = ch
        vy_lo[idx] = dl
        vy_hi[idx] = dh
        s30[idx] = s
    return vx_lo, vx_hi, vz_lo, vz_hi, vy_lo, vy_hi, s30
