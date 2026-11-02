"""Numba fast-march for the line-of-sight ray (:func:`sentinel.los.march`).

A bit-exact numba transcription of the HOT part of the ROM ray-march
(check_for_line_of_sight_to_tile $1CDD): the per-sub-step ``add_vector``, the
board-edge test, and the flat-tile / sloping-tile surface comparison, all over a
raw ``numpy.uint8`` view of the whole memory image.

Flat, sloping and object tiles are all resolved here.  Within a tile the sub-steps
that provably keep marching on one branch path are taken in closed form with their
exact summed cost, so a march costs O(tiles crossed); the per-sub-step reference it
matches bit for bit lives in ``tests/test_los_jit.py``.

Return status:
  * ``LOS_CLEAR`` (1) -- the ray reached a visible tile (ROM carry clear).
  * ``BLOCKED``   (0) -- tile above the ray / board edge / max_steps (no LOS).

Every integer is kept in a Python-int-width (int64) register and masked with
``& 0xFF`` exactly where the 6502 truncates, so uint8 wrap never corrupts a byte.
"""

import numpy as np

from sentinel import jitcache

jitcache.install()  # must precede numba: the cache key carries the cost constants

import numba  # noqa: E402  pylint: disable=wrong-import-position
from numba import njit, prange  # noqa: E402  pylint: disable=wrong-import-position

from sentinel import passcost  # noqa: E402
from sentinel.kern import spec  # noqa: E402
from sentinel.writeweight import MARCH as wcost  # noqa: E402

LOS_CLEAR = 1
BLOCKED = 0

# march_batch runs batch // BATCH_GRAIN threads (serially below two): the rays per
# thread at the measured fastest thread count (sentinel.tests.test_los_jit.batch_grain).
BATCH_GRAIN = 8

_RUN_QUAD = 0
_RUN_EDGE = 1
_RUN_OBJECT = 2
# object-walk position dependences: the $1EAF fraction against $40 / $64, is_tree
_USE_40 = 1
_USE_64 = 2
_USE_TREE = 4
_USE_MULTI = 8
_POPCOUNT = np.array([bin(i).count("1") for i in range(256)], dtype=np.int64)

# Per-sub-step 6502 cost, bound as njit-visible globals (sentinel.passcost).
_ADD_VECTOR = wcost.ADD_VECTOR
_ADD_VECTOR_NEG = wcost.ADD_VECTOR_NEG
_STEP_EDGE = wcost.STEP_EDGE
_STEP_EDGE_EXIT = wcost.STEP_EDGE_EXIT
_STEP_SETUP = wcost.STEP_SETUP
_TILE_Z_CALL = wcost.TILE_Z_CALL
_TILE_ADDR = wcost.TILE_ADDR
_TILE_Z_READ = wcost.TILE_Z_READ
_TILE_Z_FLAT = wcost.TILE_Z_FLAT
_TILE_Z_OBJ = wcost.TILE_Z_OBJ
_LEAVE_SET = wcost.LEAVE_SET
_FLAT_BRANCH = wcost.FLAT_BRANCH
_FLAT_DIFF = wcost.FLAT_DIFF
_FLAT_BELOW = wcost.FLAT_BELOW
_FLAT_ABOVE = wcost.FLAT_ABOVE
_FLAT_TOL = wcost.FLAT_TOL
_FLAT_TOL_HIT = wcost.FLAT_TOL_HIT
_FLAT_BIT60 = wcost.FLAT_BIT60
_FLAT_BIT60_HIT = wcost.FLAT_BIT60_HIT
_FLAT_ANGLE = wcost.FLAT_ANGLE
_FLAT_ANGLE_SKIP = wcost.FLAT_ANGLE_SKIP
_FLAT_LOOKUP = wcost.FLAT_LOOKUP
_FLAT_LOOKING_UP = wcost.FLAT_LOOKING_UP
_FLAT_SAME = wcost.FLAT_SAME
_FLAT_SAME_X_DIFF = wcost.FLAT_SAME_X_DIFF
_FLAT_SAME_Y = wcost.FLAT_SAME_Y
_FLAT_SAME_HIT = wcost.FLAT_SAME_HIT
_FLAT_SAME_Y_DIFF = wcost.FLAT_SAME_Y_DIFF
_SLOPE_BRANCH = wcost.SLOPE_BRANCH
_SLOPE_HEAD = wcost.SLOPE_HEAD
_SLOPE_NIB_4 = wcost.SLOPE_NIB_4
_SLOPE_NIB_12 = wcost.SLOPE_NIB_12
_SLOPE_NIB_QUAD = wcost.SLOPE_NIB_QUAD
_SLOPE_EDGE_LDA = wcost.SLOPE_EDGE_LDA
_SLOPE_EDGE_MISS = wcost.SLOPE_EDGE_MISS
_SLOPE_EDGE_HIT = wcost.SLOPE_EDGE_HIT
_SLOPE_EDGE_BLOCK = wcost.SLOPE_EDGE_BLOCK
_SLOPE_Q_C2 = wcost.SLOPE_Q_C2
_SLOPE_Q_C1 = wcost.SLOPE_Q_C1
_SLOPE_Q_EDGE = wcost.SLOPE_Q_EDGE
_SLOPE_Q_CORNER = wcost.SLOPE_Q_CORNER
_SLOPE_Q_CORNER_EOR = wcost.SLOPE_Q_CORNER_EOR
_SLOPE_Q_TAIL = wcost.SLOPE_Q_TAIL
_SLOPE_Q_USE_Y = wcost.SLOPE_Q_USE_Y
_SLOPE_Q_INVERT = wcost.SLOPE_Q_INVERT
_SLOPE_Q_ABS = wcost.SLOPE_Q_ABS
_SLOPE_Q_NEG = wcost.SLOPE_Q_NEG
_SLOPE_Q_BLOCK = wcost.SLOPE_Q_BLOCK
_OBJ_HEAD_GHOL = wcost.OBJ_HEAD_GHOL
_OBJ_HEAD_LOS = wcost.OBJ_HEAD_LOS
_OBJ_TARGET_HIT = wcost.OBJ_TARGET_HIT
_OBJ_TARGET_MISS = wcost.OBJ_TARGET_MISS
_OBJ_TYPE_BOULDER = wcost.OBJ_TYPE_BOULDER
_OBJ_TYPE_TREE = wcost.OBJ_TYPE_TREE
_OBJ_TYPE_OTHER = wcost.OBJ_TYPE_OTHER
_OBJ_TYPE_PLATFORM = wcost.OBJ_TYPE_PLATFORM
_MINXY = wcost.MINXY
_MINXY_ABS = wcost.MINXY_ABS
_MINXY_Y_WINS = wcost.MINXY_Y_WINS
_OBJ_BT_SKIP = wcost.OBJ_BT_SKIP
_OBJ_BT_TYPE = wcost.OBJ_BT_TYPE
_OBJ_BT_TREE = wcost.OBJ_BT_TREE
_OBJ_BT_BOULDER = wcost.OBJ_BT_BOULDER
_OBJ_PLAT_SKIP = wcost.OBJ_PLAT_SKIP
_OBJ_PLAT_RTS = wcost.OBJ_PLAT_RTS
_OBJ_TREE_BELOW = wcost.OBJ_TREE_BELOW
_OBJ_TREE_HIGH = wcost.OBJ_TREE_HIGH
_OBJ_TREE_NEAR = wcost.OBJ_TREE_NEAR
_OBJ_TREE_TARGETED = wcost.OBJ_TREE_TARGETED
_OBJ_TREE_SEEN = wcost.OBJ_TREE_SEEN
_OBJ_SKIP_TREE = wcost.OBJ_SKIP_TREE
_OBJ_SKIP_OTHER = wcost.OBJ_SKIP_OTHER
_OBJ_GHOL_LOOP = wcost.OBJ_GHOL_LOOP
_OBJ_GHOL_RTS = wcost.OBJ_GHOL_RTS
_MUL8 = passcost.MUL8
_MUL8_LOW = passcost.MUL8_LOW
_MUL8_BIT = passcost.MUL8_BIT
_MUL_DBL_BYTE = passcost.MUL_DBL_BYTE
_MUL_DBL_BYTE_CARRY = passcost.MUL_DBL_BYTE_CARRY
_MUL_DBL_BYTE_CALL = passcost.MUL_DBL_BYTE_CALL
_MUL_PI = passcost.MUL_PI
_SIN_COS = passcost.SIN_COS
_SIN_COS_NOQUAD = passcost.SIN_COS_NOQUAD
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

# Object-array bases in the memory image (sentinel.memmap), inlined so the njit
# code needs no Python object: OBJECTS_FLAGS $0100, OBJECTS_Z_HEIGHT $0940,
# OBJECTS_Z_FRACTION $0A00, OBJECTS_TYPE $0A40.
_OFLAGS = 0x0100
_OZHEIGHT = 0x0940
_OZFRAC = 0x0A00
_OTYPE = 0x0A40


@spec(inline=True)
def _tile_byte(mem, x, y):
    """terrain.tile_byte: the raw tiles_table byte via the ROM address arithmetic
    ``page=(x&3)+4``, ``lo=((x<<3)&0xE0)|(y&0x1F)``."""
    xx = x & 0xFF
    yy = y & 0xFF
    lo = ((xx << 3) & 0xE0) | (yy & 0x1F)
    page = (xx & 3) + 4
    return int(mem[page * 256 + lo])


@spec
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


@spec(inline=True)
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


@spec(inline=True)
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


@spec(inline=True)
def _quad_head(nib):
    """The position-independent half of the $1D8A-$1DAE edge pick.  Returns ``(flip,
    x0, x1, cycles)``: the edge is ``x1`` when ``(px_sub ^ flip) >= py_sub``, else
    ``x0`` (the same edge for the fixed-edge nibbles)."""
    if (nib & 1) == 0:
        s78 = (nib >> 2) & 3
        cyc = np.int64(_SLOPE_Q_C2 + _SLOPE_Q_CORNER)
    elif nib & 2:
        s78 = ((nib >> 2) + 2) & 3
        cyc = np.int64(_SLOPE_Q_C1 + _SLOPE_Q_CORNER)
    else:
        x = (nib >> 2) & 1
        return 0, x, x, np.int64(_SLOPE_Q_EDGE)
    flip = 0xFF if s78 & 1 else 0
    if flip:
        cyc += np.int64(_SLOPE_Q_CORNER_EOR)
    return flip, _edge(s78 << 1), _edge((s78 << 1) | 1), cyc


@spec(inline=True)
def _quad_edge(x, p73, p74, p75, p76):
    """use_edge_for_slope $1DAF-$1DEE for edge ``x``, as ``(axis, invert, base, slope,
    cycles)``: the ray is blocked iff ``(pz16 - base - s02 * slope) & 0x8000`` (mod
    2^16), ``s02`` being the sub-tile byte of x (``axis`` 1) or y (0) XOR ``invert``.
    ``cycles`` excludes the per-set-bit multiply cost and the blocked exit."""
    cyc = np.int64(_SLOPE_Q_TAIL)
    axis = x & 1
    if axis == 0:
        cyc += np.int64(_SLOPE_Q_USE_Y)
    inv = 0xFF if x & 2 else 0
    if inv:
        cyc += np.int64(_SLOPE_Q_INVERT)
    c0 = _corner_at(x & 3, p73, p74, p75, p76)
    res = (_corner_at((x & 3) + 1, p73, p74, p75, p76) - c0) & 0xFF
    slope = res - 0x100 if res & 0x80 else res
    if slope < 0:
        cyc += np.int64(_SLOPE_Q_ABS + _SLOPE_Q_NEG)
    return (
        np.int64(axis),
        np.int64(inv),
        np.int64(c0 << 8),
        np.int64(slope),
        cyc,
    )


@spec(inline=True)
def _edge_miss(b8, p73, p74, p75, p76):
    """The nibble 4/$C edge test: index of the first corner the ray byte clears (the
    $1D66 miss count), 4 when it clears none (blocked).  Non-increasing in ``b8``."""
    if b8 >= p73:
        return 0
    if b8 >= p74:
        return 1
    if b8 >= p75:
        return 2
    if b8 >= p76:
        return 3
    return 4


@spec(inline=True)
def _tile_steps(a, v):
    """Further sub-steps a 16-bit in-tile accumulator ``a`` takes before carrying out."""
    if v > 0:
        return (0xFFFF - a) // v
    if v < 0:
        return a // (-v)
    return np.int64(1) << 62


@spec(inline=True)
def _flat_run(z, za, vz, n):
    """Leading sub-steps of ``1..n`` that stay below flat height ``z``.

    Below means ``((z << 8) - pz16) & 0x8000`` with ``pz16 = za_j >> 8``; that is
    ``ceil(w_j / 256) mod 2^16 >= 0x8000`` for ``w_j = ((z << 16) - za) mod 2^24 - j*vz``.
    Step 0 is below, so ``ceil(w_0/256)`` is in [0x8000, 0xFFFF]; it moves by at most
    0x80 per step, so it leaves that band only through 0x7FFF (climbing) or 0x10000
    (descending), at the first ``j`` solving the matching linear inequality."""
    if vz == 0:
        return n
    w = ((z << 16) - za) & 0xFFFFFF
    if vz > 0:
        m = -(((0x7FFF << 8) - w) // vz)
    else:
        m = ((0xFFFF << 8) - w) // (-vz) + 1
    return min(m - 1, n)


@spec
def _popcount_sum(c, v, r):
    """``sum(popcount((c + j*v) >> 8) for j in 1..r)``: the run's summed $0D03
    multiplier set bits, by table over its s02 bytes."""
    tot = np.int64(0)
    for _ in range(r):
        c += v
        tot += _POPCOUNT[c >> 8]
    return tot


@spec(inline=True)
def _floor_ge(a1, a2, b1, b2):
    """Whether ``floor(La/256) >= floor(Lb/256)`` over a run on which the linear ``La``,
    ``Lb`` go from ``a1``/``b1`` to ``a2``/``b2``: 1 always, 0 never, -1 undecided."""
    if min(a1, a2) >> 8 >= max(b1, b2) >> 8 or min(a1 - b1, a2 - b2) >= 0:
        return 1
    if max(a1, a2) >> 8 < min(b1, b2) >> 8 or max(a1 - b1, a2 - b2) <= -0x100:
        return 0
    return -1


@spec(inline=True)
def _centred(a, side):
    """The linear accumulator whose high byte is the $1EAF centre distance (_min_xy)."""
    return a - 0x8000 if side else 0x7FFF - a


@spec
def _object_ok(z0, z2, x0, x2, y0, y2, zs, uses, tk):
    """:func:`_run_ok` for an object tile: sub-steps ``0..L`` (accumulators from ``*0``
    to ``*2``) share sub-step 0's walk path and stay below its surface ``zs``.
    A tree walk also holds its is_tree class (``tk`` = its $1E69 height + $E0)."""
    if uses:
        sx = x0 >> 15
        sy = y0 >> 15
        if sx != x2 >> 15 or sy != y2 >> 15:
            return False
        u0, u2 = _centred(x0, sx), _centred(x2, sx)
        w0, w2 = _centred(y0, sy), _centred(y2, sy)
        ywins = _floor_ge(w0, w2, u0, u2)
        if ywins < 0:
            return False
        f0, f2 = (w0, w2) if ywins else (u0, u2)
        fa = f0 >> 8
        fb = f2 >> 8
        if uses & _USE_40 and (fa >= 0x40) != (fb >= 0x40):
            return False
        if uses & _USE_64 and (fa >= 0x64) != (fb >= 0x64):
            return False
        if uses & _USE_TREE:
            # V = (tk - pz16) mod 2^16: >= $8000 below, >= $400 high, else near iff
            # V >> 2 < frac, i.e. 256*tk' - Z - 4*Lf < -(Z mod 256) - 4*(Lf mod 256)
            vlo = tk - (max(z0, z2) >> 8)
            vhi = tk - (min(z0, z2) >> 8)
            if vlo >> 16 != vhi >> 16:
                return False
            lo = vlo & 0xFFFF
            hi = vhi & 0xFFFF
            if (lo >= 0x8000) != (hi >= 0x8000) or (lo >= 0x400) != (hi >= 0x400):
                return False
            if hi < 0x400:
                base = (tk - (vlo >> 16 << 16)) << 8
                h0 = base - z0 - 4 * f0
                h2 = base - z2 - 4 * f2
                near = hi >> 2 < min(fa, fb) or max(h0, h2) < -0x4FB
                if not near and not (lo >> 2 >= max(fa, fb) or min(h0, h2) >= 0):
                    return False
    qlo = zs - (max(z0, z2) >> 8)
    qhi = zs - (min(z0, z2) >> 8)
    return (qlo >> 16) == (qhi >> 16) and (qlo & 0xFFFF) >= 0x8000


@spec
def _run_ok(L, mode, za, vz, xa, vx, ya, vy, flip, cfix, e0, e1, corners, zs, uses, tk):
    """Whether sub-steps ``1..L`` of an in-tile run all keep marching along one branch
    path, as ``(ok, class)``.  Every input is monotone or linear-plus-floor over the run
    (x/y never carry), so endpoints bound it, and ``ok`` is monotone in ``L``."""
    za1 = za + vz
    za2 = za + L * vz
    if mode == _RUN_EDGE:
        wa = za1 >> 16
        wb = za2 >> 16
        if (wa >> 8) != (wb >> 8):
            return False, 0
        ka = _edge_miss(wa & 0xFF, corners[0], corners[1], corners[2], corners[3])
        kb = _edge_miss(wb & 0xFF, corners[0], corners[1], corners[2], corners[3])
        return ka == kb < 4, ka
    x2 = xa + L * vx
    y2 = ya + L * vy
    if mode == _RUN_OBJECT:
        return _object_ok(za, za2, xa, x2, ya, y2, zs, uses, tk), 0
    x1 = xa + vx
    y1 = ya + vy
    pa = za1 >> 8
    pb = za2 >> 8
    c = 0
    if not cfix:
        if flip:
            c = _floor_ge(0xFFFF - x1, 0xFFFF - x2, y1, y2)
        else:
            c = _floor_ge(x1, x2, y1, y2)
        if c < 0:
            return False, 0
    e = e1 if c else e0
    if e[0]:
        l1, l2 = x1, x2
    else:
        l1, l2 = y1, y2
    if e[1]:
        l1, l2 = 0xFFFF - l1, 0xFFFF - l2
    ds = e[3]
    sa = e[2] + (l1 >> 8) * ds
    sb = e[2] + (l2 >> 8) * ds
    # 256*(pz16 - S) = Z - ds*Ls - 256*base - (Z mod 256) + ds*(Ls mod 256)
    g1 = za1 - ds * l1 - (e[2] << 8)
    g2 = za2 - ds * l2 - (e[2] << 8)
    qlo = max(
        min(pa, pb) - max(sa, sb),
        -((0xFF + 0xFF * max(0, -ds) - min(g1, g2)) >> 8),
    )
    qhi = min(max(pa, pb) - min(sa, sb), (max(g1, g2) + 0xFF * max(0, ds)) >> 8)
    return (qlo >> 16) == (qhi >> 16) and (qhi & 0xFFFF) < 0x8000, c


@spec
def _run(n, mode, za, vz, xa, vx, ya, vy, flip, cfix, e0, e1, corners, zs, uses, tk):
    """The longest prefix ``r <= n`` :func:`_run_ok` proves, with its class, by
    galloping then bisection on the monotone predicate."""
    if n <= 0:
        return 0, 0
    args = (mode, za, vz, xa, vx, ya, vy, flip, cfix, e0, e1, corners, zs, uses, tk)
    ok, cls = _run_ok(n, *args)
    if ok:
        return n, cls
    lo = 0
    lo_cls = 0
    hi = 1
    while hi < n:
        ok, cls = _run_ok(hi, *args)
        if not ok:
            break
        lo = hi
        lo_cls = cls
        hi *= 2
    hi = min(hi, n)
    while hi - lo > 1:
        mid = (lo + hi) >> 1
        ok, cls = _run_ok(mid, *args)
        if ok:
            lo = mid
            lo_cls = cls
        else:
            hi = mid
    return lo, lo_cls


@spec(inline=True)
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


@spec
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
        return cdd, _OBJ_TREE_BELOW, 0
    c = a & 1
    a >>= 1
    s75 = ((s75 >> 1) | (c << 7)) & 0xFF
    c = a & 1
    a >>= 1
    if a != 0:
        return cdd, _OBJ_TREE_HIGH, 0
    a = ((s75 >> 1) | (c << 7)) & 0xFF
    if a < minxy:
        return cdd, _OBJ_TREE_NEAR, 0
    if c56 & 0x80:
        return cdd, _OBJ_TREE_TARGETED, 0
    return ((cdd >> 1) | 0x80) & 0xFF, _OBJ_TREE_SEEN, 1


@spec
def _object_surface(mem, raw0, px_sub, py_sub, pz_sub, pz_whole, c58, c56, cdd):
    """los._get_tile_z_from_object $1E3F and its helpers, flattened into one bounded
    iterative walk of the object stack.  Returns ``(z, s79, c0c, c67, c56, cdd, s60,
    cycles, hits, uses, tk, seen)``: the surface, the $0C56/$0CDD trackers, the walk's
    6502 cost, the targeted-slot visits, its ``_USE_*`` position dependences, and the
    is_tree height + $E0 and $0CDD marks of its tree."""
    s60 = 0x80
    hits = 0
    uses = 0
    tk = 0
    seen = 0
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
                hits += 1
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
                uses |= _USE_40
                if frac >= 0x40:
                    cyc += _OBJ_BT_SKIP
                    go_skip = True
                elif otype == 2:  # is_tree $1E69
                    cyc += _OBJ_BT_TYPE + _OBJ_BT_TREE
                    uses |= _USE_MULTI if uses & _USE_TREE else _USE_TREE
                    tk = (int(mem[_OZHEIGHT + Y]) << 8) + int(mem[_OZFRAC + Y]) + 0xE0
                    cdd, tcyc, tseen = _is_tree_cdd(
                        mem, Y, pz_sub, pz_whole, frac, c56, cdd
                    )
                    seen += tseen
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
                    return (z, s79, c0c, c67, c56, cdd, s60, cyc, hits, uses, tk, seen)
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
                uses |= _USE_64
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
                    return (z, s79, c0c, c67, c56, cdd, s60, cyc, hits, uses, tk, seen)
        if do_ghol:
            # get_height_of_lowest_object $1EA4: stacked -> recurse on the object
            # beneath (raw = flags); else the bottom object's z_height.
            flags = int(mem[_OFLAGS + Y])
            if flags >= 0x40:
                cyc += _OBJ_GHOL_LOOP
                raw = flags
                continue
            cyc += _OBJ_GHOL_RTS
            z = int(mem[_OZHEIGHT + Y])
            return (z, s79, c0c, c67, c56, cdd, s60, cyc, hits, uses, tk, seen)
    # safety: corrupt/deep stack
    z = int(mem[_OZHEIGHT + (raw & 0x3F)])
    return (z, s79, c0c, c67, c56, cdd, s60, cyc, hits, _USE_MULTI, tk, seen)


@spec(inline=True)
def _s16(hi, lo):
    v = ((hi & 0xFF) << 8) | (lo & 0xFF)
    return v - 0x10000 if v >= 0x8000 else v


@spec(inline=True)
def _acc(whole, sub, frac):
    return ((whole & 0xFF) << 16) | ((sub & 0xFF) << 8) | (frac & 0xFF)


@spec
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

    ``cycles`` is the march's 6502 cost, per sub-step by the branch taken, with the
    laps' own write weight packed above it (:mod:`sentinel.writeweight`).

    ``s30`` is the ray's vector_z high byte (the looking-up sign).  ``c6e`` is the
    do_line_of_sight_checks byte ($0C6E); bit7 waives the looking-up rejection.
    ``c58`` is the targeted-object slot; ``c56``/``cdd`` are the $0C56/$0CDD
    trackers (seeded/LSR'd by the caller, returned so the caller can persist them).

    add_vector $1CBB is a signed 16-bit step on each 24-bit (whole:sub:frac)
    accumulator.  Once a flat or sloping tile's sub-step keeps marching, the run of
    further sub-steps inside that tile that provably keep marching (:func:`_flat_run`,
    :func:`_slope_run`) is taken in closed form with its exact summed cost, so the
    march costs O(tiles crossed), not O(sub-steps); object tiles go sub-step by sub-step.
    """
    vx = _s16(ax_hi, ax_lo)
    vy = _s16(ay_hi, ay_lo)
    vz = _s16(az_hi, az_lo)
    xa = _acc(px_whole, px_sub, px_frac)
    ya = _acc(py_whole, py_sub, py_frac)
    za = _acc(pz_whole, pz_sub, pz_frac)
    ty = 0
    tx = 0
    steps = 0
    cycles = np.int64(0)
    status = BLOCKED
    cur_tx = -1
    cur_ty = -1
    cb = 0
    cnib = 0
    cz = 0
    corners = (0, 0, 0, 0)
    ccorner = np.int64(0)
    flip = 0
    cfix = True
    hcyc = np.int64(0)
    e0 = (np.int64(0), np.int64(0), np.int64(0), np.int64(0), np.int64(0))
    e1 = e0
    neg = (vx < 0) + (vy < 0) + (vz < 0)
    step_fixed = np.int64(_ADD_VECTOR + _ADD_VECTOR_NEG * neg + 2 * _STEP_EDGE)
    # $1CFB..$1DFE: only reached once both edge tests fall through.
    tile_fixed = np.int64(_STEP_SETUP + _TILE_Z_CALL + _TILE_ADDR + _TILE_Z_READ)
    flat_step = (
        step_fixed
        + tile_fixed
        + np.int64(_TILE_Z_FLAT + _FLAT_BRANCH + _FLAT_DIFF + _FLAT_BELOW)
    )
    obj_step = step_fixed + tile_fixed + np.int64(_TILE_Z_OBJ + _FLAT_BRANCH)
    obj_step += np.int64(_FLAT_DIFF + _FLAT_BELOW)
    slope_fixed = step_fixed + tile_fixed + np.int64(_TILE_Z_FLAT + _SLOPE_BRANCH)
    slope_fixed += np.int64(_SLOPE_HEAD)
    while steps < max_steps:
        steps += 1
        cycles += step_fixed
        xa = (xa + vx) & 0xFFFFFF
        za = (za + vz) & 0xFFFFFF
        ya = (ya + vy) & 0xFFFFFF
        tx = xa >> 16
        if tx >= 0x1F:
            cycles += np.int64(_STEP_EDGE_EXIT - _STEP_EDGE + _LEAVE_SET)
            status = BLOCKED
            break
        ty = ya >> 16
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
                p76, c1 = _corner_z(mem, tx + 1, ty)
                p75, c2 = _corner_z(mem, tx + 1, ty + 1)
                p74, c3 = _corner_z(mem, tx, ty + 1)
                corners = (cz, p74, p75, p76)
                ccorner = c1 + c2 + c3
                flip, x0, x1, hcyc = _quad_head(cnib)
                cfix = x0 == x1
                e0 = _quad_edge(x0, cz, p74, p75, p76)
                e1 = _quad_edge(x1, cz, p74, p75, p76)
        px_sub = (xa >> 8) & 0xFF
        py_sub = (ya >> 8) & 0xFF
        pz_sub = (za >> 8) & 0xFF
        pz_whole = za >> 16
        budget = min(
            _tile_steps(xa & 0xFFFF, vx),
            _tile_steps(ya & 0xFFFF, vy),
            max_steps - steps,
        )
        if cb >= 0xC0:
            cycles += np.int64(_TILE_Z_OBJ + _FLAT_BRANCH + _FLAT_DIFF)
            # object tile: stack surface $1E3F, then object-aware check_flat_tile $1D0D
            c56_0 = c56
            z, s79, c0c, c67, c56, cdd, s60, ocyc, hits, uses, tk, seen = (
                _object_surface(
                    mem, cb, px_sub, py_sub, pz_sub, pz_whole, c58, c56, cdd
                )
            )
            cycles += ocyc
            zs = ((z & 0xFF) << 8) + (s79 & 0xFF)
            t = (s79 & 0xFF) - pz_sub
            borrow = 1 if t < 0 else 0
            s79 = t & 0xFF
            d = ((z & 0xFF) - pz_whole - borrow) & 0xFF
            if d & 0x80:
                cycles += np.int64(_FLAT_BELOW)
                if budget == 0 or uses & _USE_MULTI:
                    continue
                if uses & _USE_TREE and hits and not c56_0 & 0x80:
                    continue  # the targeted test of is_tree flips once $0C56 bit7 sets
                r, _cls = _run(
                    budget,
                    _RUN_OBJECT,
                    za,
                    vz,
                    xa & 0xFFFF,
                    vx,
                    ya & 0xFFFF,
                    vy,
                    flip,
                    cfix,
                    e0,
                    e1,
                    corners,
                    zs,
                    uses,
                    tk,
                )
                steps += r
                cycles += r * (obj_step + ocyc)
                k = min(r * hits, 8)
                c56 = ((c56 >> k) | (0xFF00 >> k)) & 0xFF
                k = min(r * seen, 8)
                cdd = ((cdd >> k) | (0xFF00 >> k)) & 0xFF
                xa += r * vx
                ya += r * vy
                za = (za + r * vz) & 0xFFFFFF
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

        if cnib == 0:
            cycles += np.int64(_TILE_Z_FLAT + _FLAT_BRANCH + _FLAT_DIFF)
            # check_flat_tile $1D0D, fast path (s79=0, tolerance $000C=$80,
            # $0060 bit6 clear, $0C67 clear).
            s79 = (-pz_sub) & 0xFF
            d = (cz - pz_whole - (1 if pz_sub else 0)) & 0xFF
            if d & 0x80:
                # the observer's own tile is charged no FLAT_BELOW
                below = np.int64(_FLAT_BELOW)
                if tx == (ox & 0xFF) and ty == (oy & 0xFF):
                    below = np.int64(0)
                cycles += below
                r = _flat_run(cz, za, vz, budget)
                steps += r
                cycles += r * (flat_step - _FLAT_BELOW + below)
                xa += r * vx
                ya += r * vy
                za = (za + r * vz) & 0xFFFFFF
                continue
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

        # check_sloping_tile $1D46, corner heights hoisted out of the sub-step loop
        cycles += slope_fixed - step_fixed - tile_fixed + ccorner
        edge_mode = cnib == 0x04 or cnib == 0x0C
        if edge_mode:
            per = np.int64(_SLOPE_NIB_4 if cnib == 0x04 else _SLOPE_NIB_12)
            per += np.int64(_SLOPE_EDGE_LDA)
            k = _edge_miss(pz_whole, corners[0], corners[1], corners[2], corners[3])
            cycles += per + k * np.int64(_SLOPE_EDGE_MISS)
            if k == 4:
                cycles += np.int64(_SLOPE_EDGE_BLOCK + _LEAVE_SET)
                status = BLOCKED
                break
            per += np.int64(_SLOPE_EDGE_HIT)
            cycles += np.int64(_SLOPE_EDGE_HIT)
        else:
            e = e1 if (px_sub ^ flip) >= py_sub else e0
            s02 = (px_sub if e[0] else py_sub) ^ e[1]
            per = np.int64(_SLOPE_NIB_QUAD) + hcyc + e[4]
            cycles += per + np.int64(_MUL8_BIT) * _bits(s02)
            if (((za >> 8) & 0xFFFF) - e[2] - s02 * e[3]) & 0x8000:
                cycles += np.int64(_SLOPE_Q_BLOCK + _LEAVE_SET)  # $1D44 SEC + RTS
                status = BLOCKED
                break
        r, cls = _run(
            budget,
            _RUN_EDGE if edge_mode else _RUN_QUAD,
            za,
            vz,
            xa & 0xFFFF,
            vx,
            ya & 0xFFFF,
            vy,
            flip,
            cfix,
            e0,
            e1,
            corners,
            0,
            0,
            0,
        )
        if r == 0:
            continue
        if edge_mode:
            cycles += r * (slope_fixed + ccorner + per + cls * _SLOPE_EDGE_MISS)
        else:
            e = e1 if cls else e0
            if e[0]:
                c, v = xa & 0xFFFF, vx
            else:
                c, v = ya & 0xFFFF, vy
            if e[1]:
                c, v = 0xFFFF - c, -v
            cycles += r * (slope_fixed + ccorner + np.int64(_SLOPE_NIB_QUAD) + hcyc)
            cycles += r * e[4] + np.int64(_MUL8_BIT) * _popcount_sum(c, v, r)
        steps += r
        xa += r * vx
        ya += r * vy
        za = (za + r * vz) & 0xFFFFFF
    return (
        status,
        tx,
        ty,
        xa & 0xFF,
        (xa >> 8) & 0xFF,
        xa >> 16,
        za & 0xFF,
        (za >> 8) & 0xFF,
        za >> 16,
        ya & 0xFF,
        (ya >> 8) & 0xFF,
        ya >> 16,
        c56,
        cdd,
        steps,
        cycles,
    )


@spec(inline=True)
def _march_one(mem, i, ax_lo, ax_hi, az_lo, az_hi, ay_lo, ay_hi, s30, seed, obs):
    """One lattice ray of :func:`march_batch`: ``(status, tx, ty, centre)``."""
    r = march(
        mem,
        np.int64(ax_lo[i]),
        np.int64(ax_hi[i]),
        np.int64(az_lo[i]),
        np.int64(az_hi[i]),
        np.int64(ay_lo[i]),
        np.int64(ay_hi[i]),
        np.int64(s30[i]),
        seed[0],
        seed[1],
        seed[2],
        seed[3],
        seed[4],
        seed[5],
        seed[6],
        seed[7],
        seed[8],
        obs[0],
        obs[1],
        obs[2],
        obs[3],
        obs[4],
        obs[5],
        obs[6],
    )
    return r[0], r[1], r[2], _min_xy(r[4], r[10])[0]


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
def _march_batch_parallel(
    mem, ax_lo, ax_hi, az_lo, az_hi, ay_lo, ay_hi, s30, seed, obs
):
    n = ax_lo.shape[0]
    out = np.empty((4, n), dtype=np.int64)
    stride = _balance_stride(n)  # interleave: contiguous chunks stall the join
    for k in prange(n):  # pylint: disable=not-an-iterable
        i = (k * stride) % n
        out[0, i], out[1, i], out[2, i], out[3, i] = _march_one(
            mem, i, ax_lo, ax_hi, az_lo, az_hi, ay_lo, ay_hi, s30, seed, obs
        )
    return out


@njit(cache=True)
def _march_batch_serial(mem, ax_lo, ax_hi, az_lo, az_hi, ay_lo, ay_hi, s30, seed, obs):
    n = ax_lo.shape[0]
    out = np.empty((4, n), dtype=np.int64)
    for i in range(n):
        out[0, i], out[1, i], out[2, i], out[3, i] = _march_one(
            mem, i, ax_lo, ax_hi, az_lo, az_hi, ay_lo, ay_hi, s30, seed, obs
        )
    return out


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
    seed = tuple(
        int(v)
        for v in (
            px_frac0,
            px_sub0,
            px_whole0,
            pz_frac0,
            pz_sub0,
            pz_whole0,
            py_frac0,
            py_sub0,
            py_whole0,
        )
    )
    obs = tuple(int(v) for v in (ox, oy, c6e, c58, c56, cdd, max_steps))
    vecs = (ax_lo, ax_hi, az_lo, az_hi, ay_lo, ay_hi, s30)
    threads = min(numba.get_num_threads(), ax_lo.shape[0] // BATCH_GRAIN)
    if threads <= 1:
        out = _march_batch_serial(mem, *vecs, seed, obs)
    else:
        prev = numba.get_num_threads()
        numba.set_num_threads(threads)
        try:
            out = _march_batch_parallel(mem, *vecs, seed, obs)
        finally:
            numba.set_num_threads(prev)
    return out[0], out[1], out[2], out[3]


# ============================================================================
# lattice ray-vector builder (numba twin of sentinel.los.prepare_vector_from_
# player_sights).  A batched, prange builder so the full 1px keyboard-cursor
# lattice can be constructed in a fraction of a second instead of ~1min of pure
# Python.  Bit-for-bit identical to the pure-Python path (locked by
# tests/test_landable.py::test_lattice_vectors_match_python).
# ============================================================================
@spec(inline=True)
def _vmul8(a, b):
    p = (a & 0xFF) * (b & 0xFF)
    return (p >> 8) & 0xFF, p & 0xFF


@spec(inline=True)
def _vinvert16(high, frac):
    val = ((high & 0xFF) << 8) | (frac & 0xFF)
    neg = (-val) & 0xFFFF
    return (neg >> 8) & 0xFF, neg & 0xFF


@spec(inline=True)
def _bits(v):
    """Set bits of a $0D03 multiplier: each costs MUL8_BIT more than a clear one."""
    n = np.int64(0)
    x = v & 0xFF
    while x:
        n += x & 1
        x >>= 1
    return n


@spec
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


@spec
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


@spec
def _vsin_cos(angle, frac74):
    cyc = np.int64(_SIN_COS)
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
    else:
        cyc += np.int64(_SIN_COS_NOQUAD)
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


@spec(inline=True)
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


@spec(inline=True)
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


@spec
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
