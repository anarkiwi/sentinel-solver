"""Numba specs of the ROM's object sight: relative angles ($8401) and
``check_if_enemy_can_see_object`` ($1887), with the cycles each costs.

:mod:`sentinel.relative` is the pure-Python reference they match bit for bit.
"""

import numpy as np

from sentinel import jitcache

jitcache.install()  # must precede numba: the cache key carries the cost constants

from sentinel.kern import spec  # noqa: E402

from sentinel import memmap as mm, passcost, writeweight  # noqa: E402
from sentinel.relative import _ARCTAN_LO, _ARCTAN_HI, _HYP
from sentinel.los_jit import (  # noqa: E402
    march,
    LOS_CLEAR,
    _vsin_cos,
    _vproc_sc,
    _vmul_dbl_dbl,
    _vmul_dbl_by_byte,
    _vinvert16,
)

# Coefficient tables, shared with the reference so they cannot drift.
ARCTAN_LO = np.array(_ARCTAN_LO, dtype=np.int64)
ARCTAN_HI = np.array(_ARCTAN_HI, dtype=np.int64)
HYP = np.array(_HYP, dtype=np.int64)

# Addresses and tuning constants, inlined as njit-visible globals.
_OFLAGS = mm.OBJECTS_FLAGS
_OVANGLE = mm.OBJECTS_V_ANGLE
_OX = mm.OBJECTS_X
_OZH = mm.OBJECTS_Z_HEIGHT
_OY = mm.OBJECTS_Y
_OHANG = mm.OBJECTS_H_ANGLE
_OZF = mm.OBJECTS_Z_FRACTION
_OTYPE = mm.OBJECTS_TYPE


_TARGETED_SLOT = mm.TARGETED_OBJECT_SLOT

_T_ROBOT = mm.T_ROBOT


_W_SHIFT = writeweight.SHIFT
_W_HALF = 1 << (writeweight.SHIFT - 1)


_SEE_SLOT_EMPTY = passcost.SEE_SLOT_EMPTY
_SEE_SLOT_WRONG_TYPE = passcost.SEE_SLOT_WRONG_TYPE
_SEE_PROLOGUE = passcost.SEE_PROLOGUE
_SEE_FOV = passcost.SEE_FOV
_SEE_FOV_REJECT = passcost.SEE_FOV_REJECT
_SEE_FOV_PASS = passcost.SEE_FOV_PASS
_SEE_ROBOT = passcost.SEE_ROBOT
_SEE_NOT_ROBOT = passcost.SEE_NOT_ROBOT
_SEE_PROBE_FIXED = passcost.SEE_PROBE_FIXED
_SEE_BASE = passcost.SEE_BASE
_SEE_BASE_AGAIN = passcost.SEE_BASE_AGAIN
_SEE_BASE_LAST = passcost.SEE_BASE_LAST
_SEE_TAIL = passcost.SEE_TAIL
_MARCH_ENTRY = passcost.MARCH_ENTRY
_PREP_VEC = passcost.PREP_VEC

_REL_XY = writeweight.pack("REL_XY")
_REL_XY_ABS = writeweight.pack("REL_XY_ABS")
_REL_Z = writeweight.pack("REL_Z")
_REL_ANGLES = writeweight.pack("REL_ANGLES")
_ANG_CMP = writeweight.pack("ANG_CMP")
_ANG_X_LARGER = writeweight.pack("ANG_X_LARGER")
_ANG_Y_LARGER = writeweight.pack("ANG_Y_LARGER")
_ANG_EQ = writeweight.pack("ANG_EQ")
_ANG_EQ_Y = writeweight.pack("ANG_EQ_Y")
_ANG_EQ_X = writeweight.pack("ANG_EQ_X")
_ANG_MIN_Y = writeweight.pack("ANG_MIN_Y")
_ANG_MIN_X = writeweight.pack("ANG_MIN_X")
_ANG_ZERO = writeweight.pack("ANG_ZERO")
_ANG_NONZERO = writeweight.pack("ANG_NONZERO")
_SCALE_SHIFT = writeweight.pack("SCALE_SHIFT")
_SCALE_LOOP = writeweight.pack("SCALE_LOOP")
_SCALE_LOOP_Y = writeweight.pack("SCALE_LOOP_Y")
_SCALE_TAIL = writeweight.pack("SCALE_TAIL")
_ANG_SIGN = writeweight.pack("ANG_SIGN")
_ANG_SIGN_KEEP = writeweight.pack("ANG_SIGN_KEEP")
_ANG_SIGN_NEGATE = writeweight.pack("ANG_SIGN_NEGATE")
_ANG_QUAD = writeweight.pack("ANG_QUAD")
_ANG_QUAD_LOW = writeweight.pack("ANG_QUAD_LOW")
_ANG_QUAD_HIGH = writeweight.pack("ANG_QUAD_HIGH")
_ANG_QUAD_TAIL = writeweight.pack("ANG_QUAD_TAIL")

_DIV_FULL_SHIFT = writeweight.pack("DIV_FULL_SHIFT")
_DIV_FULL_CARRY = writeweight.pack("DIV_FULL_CARRY")
_DIV_FULL_UNDER = writeweight.pack("DIV_FULL_UNDER")
_DIV_FULL_OVER = writeweight.pack("DIV_FULL_OVER")
_DIV_FULL_EQ = writeweight.pack("DIV_FULL_EQ")
_DIV_FULL_EQ_UNDER = writeweight.pack("DIV_FULL_EQ_UNDER")
_DIV_FULL_EQ_OVER = writeweight.pack("DIV_FULL_EQ_OVER")
_DIV_SUB = writeweight.pack("DIV_SUB")
_DIV_SKIP_TEST = writeweight.pack("DIV_SKIP_TEST")
_DIV_SKIP = writeweight.pack("DIV_SKIP")
_DIV_NO_SKIP = writeweight.pack("DIV_NO_SKIP")
_DIV_BYTE_SHIFT = writeweight.pack("DIV_BYTE_SHIFT")
_DIV_BYTE_CARRY = writeweight.pack("DIV_BYTE_CARRY")
_DIV_BYTE_UNDER = writeweight.pack("DIV_BYTE_UNDER")
_DIV_BYTE_OVER = writeweight.pack("DIV_BYTE_OVER")
_DIV_LAST_SHIFT = writeweight.pack("DIV_LAST_SHIFT")
_DIV_LAST_CARRY = writeweight.pack("DIV_LAST_CARRY")
_DIV_LAST_CMP = writeweight.pack("DIV_LAST_CMP")
_DIV_LAST_TAIL = writeweight.pack("DIV_LAST_TAIL")
_DIV_PLP = writeweight.pack("DIV_PLP")
_DIV_ROUND3_UNDER = writeweight.pack("DIV_ROUND3_UNDER")
_DIV_ROUND3_OVER = writeweight.pack("DIV_ROUND3_OVER")
_DIV_NO_OVERFLOW = writeweight.pack("DIV_NO_OVERFLOW")
_DIV_OVERFLOW = writeweight.pack("DIV_OVERFLOW")
_DIV_ARCTAN = writeweight.pack("DIV_ARCTAN")
_DIV_ARCTAN_DONE = writeweight.pack("DIV_ARCTAN_DONE")
_DIV_DELTA_BR = writeweight.pack("DIV_DELTA_BR")
_DIV_HALF_BR = writeweight.pack("DIV_HALF_BR")
_DIV_DELTA = writeweight.pack("DIV_DELTA")
_DIV_DELTA_KEEP = writeweight.pack("DIV_DELTA_KEEP")
_DIV_DELTA_INVERT = writeweight.pack("DIV_DELTA_INVERT")
_DIV_DELTA_HALVE = writeweight.pack("DIV_DELTA_HALVE")
_DIV_NEXT = writeweight.pack("DIV_NEXT")
_DIV_ADD_SKIP = writeweight.pack("DIV_ADD_SKIP")
_DIV_ADD_HALF = writeweight.pack("DIV_ADD_HALF")
_DIV_AVERAGE = writeweight.pack("DIV_AVERAGE")
_DIV_TABLE_CROSS = writeweight.pack("DIV_TABLE_CROSS")

_VANG_HEAD = writeweight.pack("VANG_HEAD")
_VANG_POS = writeweight.pack("VANG_POS")
_VANG_NEG = writeweight.pack("VANG_NEG")
_VANG_SETUP = writeweight.pack("VANG_SETUP")
_VANG_SHIFT = writeweight.pack("VANG_SHIFT")
_VANG_SIGN_POS = writeweight.pack("VANG_SIGN_POS")
_VANG_SIGN_NEG = writeweight.pack("VANG_SIGN_NEG")
_VANG_TAIL = writeweight.pack("VANG_TAIL")
_HYP_HEAD = writeweight.pack("HYP_HEAD")
_HYP_TAIL = writeweight.pack("HYP_TAIL")


_MAX_STEPS = 20000  # can_see_object's march bound (the ROM's board-edge exit)
ZP_LO = 0x50  # the zero-page window the geometry touches ($0050..$008B)
ZP_HI = 0x8C


@spec(inline=True)
def _wweight(total):
    """The write weight packed above a cost total (:mod:`sentinel.writeweight`)."""
    return (total + np.int64(_W_HALF)) >> np.int64(_W_SHIFT)


@spec(inline=True)
def _wcycles(total):
    """The 6502 cycles of a packed cost total."""
    return total - (_wweight(total) << np.int64(_W_SHIFT))


@spec(inline=True)
def _rd(mem, addr):
    return np.int64(mem[addr])


@spec(inline=True)
def _asl(v):
    return (v << 1) & 0xFF, (v >> 7) & 1


@spec(inline=True)
def _rol(v, c):
    return ((v << 1) | c) & 0xFF, (v >> 7) & 1


@spec(inline=True)
def _ror(v, c):
    return ((c << 7) | (v >> 1)) & 0xFF, v & 1


@spec(inline=True)
def _full_over(ca, a, t74, t76, t77):
    """Rounds 1-3 of $0D4A: a >= b as a 16-bit compare, and its branches' cost."""
    if ca:
        return True, np.int64(_DIV_FULL_CARRY)
    if a > t76:
        return True, np.int64(_DIV_FULL_OVER)
    if a < t76:
        return False, np.int64(_DIV_FULL_UNDER)
    if t74 >= t77:
        return True, np.int64(_DIV_FULL_EQ + _DIV_FULL_EQ_OVER)
    return False, np.int64(_DIV_FULL_EQ + _DIV_FULL_EQ_UNDER)


@spec
def _finish_overflow(a, php_carry, t78, cyc):
    """consider_overflow $0DFC: the round-3 bit, the 45-degree clamp, then the
    arctan lookup + rounds-9/10 interpolation."""
    carry = 0
    cyc += np.int64(_DIV_PLP)
    if php_carry:
        s = a + 0x1F + 1
        a = s & 0xFF
        carry = 1 if s > 0xFF else 0
        cyc += np.int64(_DIV_ROUND3_OVER)
    else:
        cyc += np.int64(_DIV_ROUND3_UNDER)
    if carry:
        return np.int64(0x00), np.int64(0x20), np.int64(0xFF), cyc + _DIV_OVERFLOW
    if php_carry:
        cyc += np.int64(_DIV_NO_OVERFLOW)

    y = a & 0xFF
    ratio = y
    ang_lo = ARCTAN_LO[y]
    ang_hi = ARCTAN_HI[y]
    cyc += np.int64(_DIV_ARCTAN)
    if y >= 0xFF:
        cyc += np.int64(_DIV_TABLE_CROSS)
    b78_7 = (t78 >> 7) & 1
    b78_6 = (t78 >> 6) & 1
    if b78_7 == 0 and b78_6 == 0:
        return (
            np.int64(ang_lo),
            np.int64(ang_hi),
            np.int64(ratio),
            cyc + _DIV_ARCTAN_DONE,
        )
    # One $3B01,Y and one $3C02,Y read per interpolation block, a cycle dearer off-page
    cross = np.int64(0)
    if y >= 0xFF:
        cross += np.int64(_DIV_TABLE_CROSS)
    if y >= 0xFE:
        cross += np.int64(_DIV_TABLE_CROSS)

    nxt_lo = ARCTAN_LO[y + 1]
    nxt_hi = ARCTAN_HI[y + 1]
    d_lo = (ang_lo - nxt_lo) & 0xFF
    borrow = 1 if ang_lo < nxt_lo else 0
    d_hi = (ang_hi - nxt_hi - borrow) & 0xFF
    if b78_7:
        cyc += np.int64(_DIV_DELTA_BR + _DIV_DELTA) + cross
        if b78_6:  # round 9 over -> invert the delta
            d_hi, d_lo = _vinvert16(d_hi, d_lo)
            cyc += np.int64(_DIV_DELTA_INVERT)
        else:
            cyc += np.int64(_DIV_DELTA_KEEP)
        cyc += np.int64(_DIV_DELTA_HALVE)
    else:
        cyc += np.int64(_DIV_HALF_BR)  # $0E30 BVS $0E50: only the average is wanted
    # ROL A ; ROR $0075 ; ROR $0074 -- arithmetic >>1 keeping the sign.
    t75, c2 = _ror(d_hi, (d_hi >> 7) & 1)
    t74b, _c = _ror(d_lo, c2)
    s = ang_lo + nxt_lo
    ang_lo = s & 0xFF
    cc = 1 if s > 0xFF else 0
    ang_hi = (ang_hi + nxt_hi + cc) & 0xFF
    cyc += np.int64(_DIV_NEXT) + cross
    if b78_7:  # round 10 over -> add half-delta
        s = ang_lo + t74b
        ang_lo = s & 0xFF
        cc = 1 if s > 0xFF else 0
        ang_hi = (ang_hi + t75 + cc) & 0xFF
        cyc += np.int64(_DIV_ADD_HALF)
    else:
        cyc += np.int64(_DIV_ADD_SKIP)
    ang_hi, c3 = _ror(ang_hi, 0)
    ang_lo, _c2 = _ror(ang_lo, c3)
    return np.int64(ang_lo), np.int64(ang_hi), np.int64(ratio), cyc + _DIV_AVERAGE


@spec
def _divide_and_arctan(a_lo, a_hi, b_lo, b_hi):
    """$0D4A: shift/subtract divide of 16-bit a by 16-bit b, then arctan(a/b)."""
    t74 = a_lo & 0xFF
    a = a_hi & 0xFF
    t76 = b_hi & 0xFF
    t77 = b_lo & 0xFF
    t78 = np.int64(0)
    cyc = np.int64(0)

    c = 0
    php_carry = 0
    for rnd in range(1, 4):
        if rnd == 1:
            t74, c0 = _asl(t74)
        else:
            t74, c0 = _rol(t74, c)
        a, ca = _rol(a, c0)
        over, dcyc = _full_over(ca, a, t74, t76, t77)
        cyc += np.int64(_DIV_FULL_SHIFT) + dcyc
        if over:
            v = t74 - t77
            t74 = v & 0xFF
            a = (a - t76 - (1 if v < 0 else 0)) & 0xFF
            c = 1
            cyc += np.int64(_DIV_SUB)
        else:
            c = 0
        if rnd == 3:
            php_carry = c  # PHP after round 3

    cyc += np.int64(_DIV_SKIP_TEST)
    if a == t76:  # skip_further_division $0E10
        cyc += np.int64(_DIV_SKIP)
        a = np.int64(0)
        cc = 1
        t74, cc = _ror(t74, cc)
        a, cc = _ror(a, cc)
        t74, cc = _ror(t74, cc)
        a, cc = _ror(a, cc)
        return _finish_overflow(a | 0x20, php_carry, np.int64(0), cyc)
    cyc += np.int64(_DIV_NO_SKIP)

    for rnd in range(4, 10):
        if rnd == 4:
            t74, c0 = _asl(t74)
        else:
            t74, c0 = _rol(t74, c)
        a, ca = _rol(a, c0)
        cyc += np.int64(_DIV_BYTE_SHIFT)
        if ca:
            cyc += np.int64(_DIV_BYTE_CARRY)
        elif a >= t76:
            cyc += np.int64(_DIV_BYTE_OVER)
        else:
            cyc += np.int64(_DIV_BYTE_UNDER)
        if ca or a >= t76:
            a = (a - t76) & 0xFF
            c = 1
        else:
            c = 0
    # round 10 $0DF1: ROR $78 (round-9 bit), ROL A, compare, ROR $78
    t78, c0 = _ror(t78, c)
    a, ca = _rol(a, c0)
    cyc += np.int64(_DIV_LAST_SHIFT + _DIV_LAST_TAIL)
    if ca:
        c = 1
        cyc += np.int64(_DIV_LAST_CARRY)
    else:
        c = 1 if a >= t76 else 0
        cyc += np.int64(_DIV_LAST_CMP)
    t78, _c = _ror(t78, c)
    return _finish_overflow(t74, php_carry, t78, cyc)


@spec
def _normalise(zp, max_lo_a, max_hi_a, min_lo_a, min_hi_a, lap):
    """scale_using_x/_y $92C1/$92FF: shift max left until it overflows, min in
    lock-step, then back the max off by one.  ``lap`` is one loop-back's cost.
    Returns (b_lo, b_hi, a_lo, a_hi, cycles)."""
    max_lo = zp[max_lo_a]
    min_lo = zp[min_lo_a]
    min_hi = zp[min_hi_a]
    a = zp[max_hi_a]
    laps = 0
    while True:
        max_lo, c = _asl(max_lo)
        a, ca = _rol(a, c)
        if ca:  # max overflowed
            break
        min_lo, c2 = _asl(min_lo)
        min_hi, _c = _rol(min_hi, c2)
        laps += 1
    a, c = _ror(a, 1)
    max_lo, _c2 = _ror(max_lo, c)
    cyc = (
        np.int64(_SCALE_SHIFT) + np.int64(laps) * np.int64(lap) + np.int64(_SCALE_TAIL)
    )
    return max_lo & 0xFC, a, min_lo, min_hi, cyc


@spec
def _calc_angle(zp):
    """calculate_angle $9287: quadrant-folded arctan of (x, y), and its cycles."""
    x_lo = zp[0x80]
    x_hi = zp[0x83]
    y_lo = zp[0x82]
    y_hi = zp[0x85]
    sx = zp[0x86]
    sy = zp[0x88]
    cyc = np.int64(_ANG_CMP)
    if y_hi < x_hi:
        x_larger = True
        cyc += np.int64(_ANG_X_LARGER)
    elif y_hi > x_hi:
        x_larger = False
        cyc += np.int64(_ANG_Y_LARGER)
    else:
        x_larger = y_lo < x_lo
        cyc += np.int64(_ANG_EQ)
        cyc += np.int64(_ANG_EQ_X if x_larger else _ANG_EQ_Y)
    if x_larger:
        cyc += np.int64(_ANG_MIN_Y)
        zp[0x5D] = y_hi
        zp[0x5C] = y_lo
        zp[0x7A] = x_lo
        zp[0x7B] = x_hi
        b_lo, b_hi, a_lo, a_hi, ncyc = _normalise(
            zp, 0x80, 0x83, 0x82, 0x85, _SCALE_LOOP
        )
        ang_lo, ang_hi, ratio, dcyc = _divide_and_arctan(a_lo, a_hi, b_lo, b_hi)
        cyc += ncyc + dcyc + np.int64(_ANG_SIGN)
        if ((sx ^ sy) & 0x80) == 0:  # same sign -> negate angle
            ang_hi, ang_lo = _vinvert16(ang_hi, ang_lo)
            cyc += np.int64(_ANG_SIGN_NEGATE)
        else:
            cyc += np.int64(_ANG_SIGN_KEEP)
        base = 0x40 if (sx & 0x80) == 0 else 0xC0
        cyc += np.int64(_ANG_QUAD)
        cyc += np.int64(_ANG_QUAD_LOW if base == 0x40 else _ANG_QUAD_HIGH)
        ang_hi = (ang_hi + base) & 0xFF
    else:
        cyc += np.int64(_ANG_MIN_X)
        if (y_hi | y_lo) == 0:  # both zero
            zp[0x7E] = 0
            zp[0x8A] = 0
            zp[0x8B] = 0
            return cyc + np.int64(_ANG_ZERO)
        cyc += np.int64(_ANG_NONZERO)
        zp[0x5D] = x_hi
        zp[0x5C] = x_lo
        zp[0x7A] = y_lo
        zp[0x7B] = y_hi
        b_lo, b_hi, a_lo, a_hi, ncyc = _normalise(
            zp, 0x82, 0x85, 0x80, 0x83, _SCALE_LOOP_Y
        )
        ang_lo, ang_hi, ratio, dcyc = _divide_and_arctan(a_lo, a_hi, b_lo, b_hi)
        cyc += ncyc + dcyc + np.int64(_ANG_SIGN)
        if (sx ^ sy) & 0x80:  # opposite sign -> negate angle
            ang_hi, ang_lo = _vinvert16(ang_hi, ang_lo)
            cyc += np.int64(_ANG_SIGN_NEGATE)
        else:
            cyc += np.int64(_ANG_SIGN_KEEP)
        base = 0x00 if (sy & 0x80) == 0 else 0x80
        cyc += np.int64(_ANG_QUAD)
        cyc += np.int64(_ANG_QUAD_LOW if base == 0x00 else _ANG_QUAD_HIGH)
        ang_hi = (ang_hi + base) & 0xFF
    zp[0x8A] = ang_lo
    zp[0x8B] = ang_hi
    zp[0x7E] = ratio
    return cyc + np.int64(_ANG_QUAD_TAIL)


@spec
def _calc_hypotenuse(zp):
    """calculate_hypotenuse $937F: distance = max + f*min/512."""
    ratio = zp[0x7E]
    f = HYP[(ratio >> 1) + (ratio & 1)]
    res_lo, res_hi, mcyc = _vmul_dbl_by_byte(zp[0x5C], f, zp[0x5D])
    new_hi = res_hi >> 1
    new_lo = ((res_hi & 1) << 7) | (res_lo >> 1)
    s = new_lo + zp[0x7A]
    zp[0x7C] = s & 0xFF
    cc = 1 if s > 0xFF else 0
    zp[0x7D] = (new_hi + zp[0x7B] + cc) & 0xFF
    return np.int64(_HYP_HEAD + _HYP_TAIL) + np.int64(mcyc)


@spec
def _vertical_angle(zp, z_hi, v_angle):
    """calculate_object_relative_vertical_angle $933D."""
    sx = z_hi & 0xFF
    cyc = np.int64(_VANG_HEAD)
    if sx & 0x80:  # negative -> make positive
        zlo = (-zp[0x80]) & 0xFF
        borrow = 1 if zp[0x80] != 0 else 0
        sx_abs = (-sx - borrow) & 0xFF
        zp[0x80] = zlo
        cyc += np.int64(_VANG_NEG)
    else:
        sx_abs = sx
        cyc += np.int64(_VANG_POS)
    zp[0x83] = sx_abs
    zp[0x82] = zp[0x7C]
    zp[0x85] = zp[0x7D]
    zp[0x88] = 0
    zp[0x86] = sx
    cyc += np.int64(_VANG_SETUP) + _calc_angle(zp) + np.int64(_VANG_SHIFT)
    lo = zp[0x8A] - 0x20
    t50 = lo & 0xFF
    a = (zp[0x8B] - v_angle - (1 if lo < 0 else 0)) & 0xFF
    neg = a & 0x80
    for _ in range(4):
        c = a & 1
        a = a >> 1
        t50 = ((c << 7) | (t50 >> 1)) & 0xFF
    if neg:
        a = a | 0xF0
        cyc += np.int64(_VANG_SIGN_NEG)
    else:
        cyc += np.int64(_VANG_SIGN_POS)
    zp[0x50] = t50
    return a & 0xFF, cyc + np.int64(_VANG_TAIL)


@spec
def _relative_angles(mem, zp, observer, target):
    """calculate_object_relative_angles_and_distance $8401, play-mode path.

    The zero-page window is zeroed here, where the reference allocates its scratch.
    Returns (c57, angle_lo, angle_hi, z_lo, z_hi, cycles)."""
    for i in range(ZP_LO, ZP_HI):
        zp[i] = 0
    dx = (_rd(mem, _OX + target) - _rd(mem, _OX + observer)) & 0xFF
    zp[0x86] = dx
    zp[0x80] = 0
    zp[0x83] = (-dx) & 0xFF if dx & 0x80 else dx
    dy = (_rd(mem, _OY + target) - _rd(mem, _OY + observer)) & 0xFF
    zp[0x88] = dy
    zp[0x82] = 0
    zp[0x85] = (-dy) & 0xFF if dy & 0x80 else dy
    v = _rd(mem, _OZF + target) - _rd(mem, _OZF + observer)
    zp[0x81] = v & 0xFF
    zp[0x84] = (
        _rd(mem, _OZH + target) - _rd(mem, _OZH + observer) - (1 if v < 0 else 0)
    ) & 0xFF
    cyc = np.int64(_REL_ANGLES + _REL_XY + _REL_Z)
    cyc += np.int64(_REL_XY_ABS) * np.int64((dx >> 7) + (dy >> 7))
    cyc += _calc_angle(zp)
    c57 = (zp[0x8B] - _rd(mem, _OHANG + observer) + 0x0A) & 0xFF
    cyc += _calc_hypotenuse(zp)
    return c57, zp[0x8A], zp[0x8B], zp[0x81], zp[0x84], cyc


@spec
def _prep_vec_angle(h_angle, h_frac, v_angle, v_frac):
    """prepare_vector_from_angle $1C54, the standalone entry the enemy probes use.

    Returns (vx_lo, vx_hi, vz_lo, vz_hi, vy_lo, vy_hi, s30, cycles)."""
    sin_lo_v, cos_lo_v, sin_hi_v, cos_hi_v, cyc = _vsin_cos(v_angle, v_frac)
    cyc += _PREP_VEC
    s33, s32, c = _vproc_sc(cos_lo_v, cos_hi_v)
    cyc += c
    s30, s2d, c = _vproc_sc(sin_lo_v, sin_hi_v)
    cyc += c
    h_sin_lo, h_cos_lo, h_sin_hi, h_cos_hi, c = _vsin_cos(h_angle, h_frac)
    cyc += c
    vy_hi, vy_lo, c = _vmul_dbl_dbl(h_cos_lo, h_cos_hi, s32, s33)
    cyc += c
    vx_hi, vx_lo, c = _vmul_dbl_dbl(h_sin_lo, h_sin_hi, s32, s33)
    cyc += c
    return vx_lo, vx_hi, s2d, s30, vy_lo, vy_hi, s30, cyc


@spec
def _can_see_object(mem, zp, observer, target, expected_type, fov_width):
    """check_if_enemy_can_see_object $1887: a robot at its upper point, then every object's base.

    The upper probe ($18DC) sets $0C6E bit7, waiving the looking-up rejection.

    Returns (in_slot, in_fov, exposure, full, tree_in_los_head, cycles, weight),
    ``weight`` being the write weight the marches inside it carry."""
    mem[0x0014] = 0
    if mem[_OFLAGS + target] & 0x80:  # empty slot
        return 0, 0, np.int64(0), 0, 0, np.int64(_SEE_SLOT_EMPTY), np.int64(0)
    if mem[_OTYPE + target] != expected_type:
        return 0, 0, np.int64(0), 0, 0, np.int64(_SEE_SLOT_WRONG_TYPE), np.int64(0)

    c57, angle_lo, angle_hi, z_lo, z_hi, rcyc = _relative_angles(
        mem, zp, observer, target
    )
    cyc = np.int64(_SEE_PROLOGUE) + rcyc + np.int64(_SEE_FOV)
    a = (c57 - 0x0A + (fov_width >> 1)) & 0xFF
    if a >= fov_width:  # $18B8 FOV gate
        cyc += np.int64(_SEE_FOV_REJECT)
        return 1, 0, np.int64(0), 0, 0, _wcycles(cyc), _wweight(cyc)
    cyc += np.int64(_SEE_FOV_PASS)

    v_angle_obs = _rd(mem, _OVANGLE + observer)
    mem[_TARGETED_SLOT] = np.uint8(target)
    ox = _rd(mem, _OX + observer)
    oy = _rd(mem, _OY + observer)
    obs_zf = _rd(mem, _OZF + observer)
    obs_zh = _rd(mem, _OZH + observer)
    n_probes = 2 if expected_type == _T_ROBOT else 1
    if n_probes == 2:
        cyc += np.int64(_SEE_ROBOT)
    else:
        cyc += np.int64(_SEE_NOT_ROBOT)
    # $1904..$1914 runs twice whatever the type: the $1E counter always reaches 0.
    cyc += np.int64(2 * _SEE_BASE + _SEE_BASE_AGAIN + _SEE_BASE_LAST)
    for probe in range(n_probes):
        if n_probes == 2 and probe == 0:
            plo = z_lo
            phi = z_hi
            do_los = 0x80
        else:
            plo = (z_lo - 0xE0) & 0xFF
            phi = (z_hi - (1 if z_lo < 0xE0 else 0)) & 0xFF
            do_los = 0x00
        zp[0x80] = plo
        _va, vcyc = _vertical_angle(zp, phi, v_angle_obs)
        cyc += np.int64(_SEE_PROBE_FIXED) + vcyc + np.int64(_MARCH_ENTRY)
        vx_lo, vx_hi, vz_lo, vz_hi, vy_lo, vy_hi, s30, pcyc = _prep_vec_angle(
            angle_hi, angle_lo, zp[0x8B], zp[0x8A]
        )
        cyc += pcyc  # $1C54, priced from its own shift-adds
        c56 = (_rd(mem, 0x0C56) >> 1) & 0xFF  # $1CDF LSR $0C56
        cdd = (_rd(mem, 0x0CDD) >> 1) & 0xFF  # $1CE2 LSR $0CDD
        mem[0x0C56] = np.uint8(c56)
        mem[0x0CDD] = np.uint8(cdd)
        res = march(
            mem,
            vx_lo,
            vx_hi,
            vz_lo,
            vz_hi,
            vy_lo,
            vy_hi,
            s30,
            0,
            0x80,
            ox,
            0,
            obs_zf,
            obs_zh,
            0,
            0x80,
            oy,
            ox,
            oy,
            do_los,
            _rd(mem, _TARGETED_SLOT),
            c56,
            cdd,
            _MAX_STEPS,
        )
        los_ok = res[0] == LOS_CLEAR
        cyc += np.int64(res[15])
        mem[0x0C56] = np.uint8(res[12] & 0xFF)
        mem[0x0CDD] = np.uint8(res[13] & 0xFF)
        # $18F9-$1901: the four-rotate chained-carry cascade.
        c56v = _rd(mem, 0x0C56)
        reached = (c56v >> 7) & 1
        mem[0x0C56] = np.uint8(((c56v << 1) | (0 if los_ok else 1)) & 0xFF)
        c14 = _rd(mem, 0x0014)
        c14_out = c14 & 1
        mem[0x0014] = np.uint8(((reached << 7) | (c14 >> 1)) & 0xFF)
        cddv = _rd(mem, 0x0CDD)
        tree_flag = (cddv >> 7) & 1
        mem[0x0CDD] = np.uint8(((cddv << 1) | c14_out) & 0xFF)
        c76 = _rd(mem, 0x0C76)
        mem[0x0C76] = np.uint8(((tree_flag << 7) | (c76 >> 1)) & 0xFF)
    exposure = _rd(mem, 0x0014)
    full = 1 if exposure & 0x80 else 0
    tree_head = 1 if _rd(mem, 0x0C76) & 0x40 else 0
    cyc += np.int64(_SEE_TAIL)  # the marches inside it carry their write weight
    return 1, 1, exposure, full, tree_head, _wcycles(cyc), _wweight(cyc)
