"""Numba twin of the enemy frame clock -- :func:`sentinel.enemies.advance_frames`.

:mod:`sentinel.enemies` remains the bit-exact reference and the numba-absent fallback,
mirroring :mod:`sentinel.los`/:mod:`sentinel.los_jit`; ``tests/test_enemies_jit.py``
asserts byte-identical 64 KB images between the two over hundreds of frames.
"""

import numpy as np
from numba import njit

from sentinel import memmap as mm
from sentinel.relative import _ARCTAN_LO, _ARCTAN_HI, _HYP
from sentinel.los_jit import (
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

_PLAYER = mm.PLAYER_OBJECT
_ENERGY = mm.PLAYER_ENERGY
_CURSOR = mm.CURSOR
_FOV_WIDTH = mm.FOV_WIDTH
_PLATFORM_X = mm.PLATFORM_X
_PLATFORM_Y = mm.PLATFORM_Y
_DIED_DRAINING = mm.PLAYER_DIED_BY_DRAINING
_HYPERSPACED = mm.PLAYER_HAS_HYPERSPACED
_COMPLETE = mm.LANDSCAPE_COMPLETE
_BELOW_Z = mm.ENEMY_BELOW_Z
_NOT_ACTED = mm.PLAYER_NOT_ACTED
_BRESENHAM = mm.COOLDOWN_BRESENHAM
_BRESENHAM_STEP = mm.COOLDOWN_BRESENHAM_STEP
_GATE = mm.COOLDOWN_GATE
_PRND = mm.PRND_STATE

_DRAIN_CD = mm.ENEMIES_DRAINING_COOLDOWN
_ROT_CD = mm.ENEMIES_ROTATION_COOLDOWN
_UPD_CD = mm.ENEMIES_UPDATE_COOLDOWN
_M_SEARCH = mm.ENEMIES_MEANIE_SEARCH_OBJECT
_DISCHARGE = mm.ENEMIES_ENERGY_TO_DISCHARGE
_M_FAILED = mm.ENEMIES_FAILED_MEANIE_MEMORY
_M_SCANS = mm.ENEMIES_MEANIE_ATTEMPT_SCANS
_M_OBJECT = mm.ENEMIES_MEANIE_OBJECT
_TARGET = mm.ENEMIES_TARGETED_OBJECT
_TARGET_EXP = mm.ENEMIES_TARGETED_OBJECT_EXPOSURE
_CONSIDERING = mm.ENEMIES_CONSIDERING_MEANIE
_ROT_SPEED = mm.ROTATION_SPEED_TABLE
_TARGETED_SLOT = mm.TARGETED_OBJECT_SLOT

_T_ROBOT = mm.T_ROBOT
_T_TREE = mm.T_TREE
_T_BOULDER = mm.T_BOULDER
_T_MEANIE = mm.T_MEANIE
_T_SENTRY = mm.T_SENTRY
_T_SENTINEL = mm.T_SENTINEL
_OBJECT_TILE = mm.OBJECT_TILE
_NUM_SLOTS = mm.NUM_SLOTS
_ENERGY_MASK = mm.ENERGY_MASK
_ROBOT_ENERGY = mm.ENERGY_IN_OBJECTS[mm.T_ROBOT]

_FOV_SCAN = 0x14
_FOV_CREATE_MEANIE = 0x28
_UPD_CD_SCAN = 0x04
_UPD_CD_DRAIN = 0x1E
_UPD_CD_MEANIE_ROTATE = 0x0A
_UPD_CD_MEANIE_MADE = 0x32
_ROT_CD_RELOAD = 0xC8
_DRAIN_CD_RELOAD = 0x78
_COOLDOWN_STICK = 0x02
_MEANIE_ROTATE_STEP = 0x08
_MEANIE_MAX_ATTEMPTS = 0x02

_MAX_STEPS = 20000  # can_see_object's march bound (the ROM's board-edge exit)
ZP_LO = 0x50  # the zero-page window the geometry touches ($0050..$008B)
ZP_HI = 0x8C


@njit(cache=True, inline="always")
def _rd(mem, addr):
    return np.int64(mem[addr])


@njit(cache=True, inline="always")
def _wr(mem, addr, val):
    mem[addr] = np.uint8(val & 0xFF)


@njit(cache=True, inline="always")
def _tile_byte(mem, x, y):
    """calculate_tile_address $2BA8, masked-8-bit form (edge reads wrap as on 6502)."""
    lo = (((x & 0xFF) << 3) & 0xE0) | (y & 0x1F)
    return np.int64(mem[((x & 3) + 4) * 256 + lo])


@njit(cache=True, inline="always")
def _set_tile_byte(mem, x, y, val):
    """Write a tiles_table byte through the same addressing as :func:`_tile_byte`."""
    lo = (((x & 0xFF) << 3) & 0xE0) | (y & 0x1F)
    mem[((x & 3) + 4) * 256 + lo] = np.uint8(val & 0xFF)


@njit(cache=True)
def _prng_next(mem):
    """prnd $31CA: 8 shuffles of the 5-byte LFSR at $0C7B, returning $0C7F."""
    s0 = np.int64(mem[_PRND])
    s1 = np.int64(mem[_PRND + 1])
    s2 = np.int64(mem[_PRND + 2])
    s3 = np.int64(mem[_PRND + 3])
    s4 = np.int64(mem[_PRND + 4])
    for _ in range(8):
        carry = ((s2 >> 3) ^ s4) & 1
        v = (s0 << 1) | carry
        carry = v >> 8
        s0 = v & 0xFF
        v = (s1 << 1) | carry
        carry = v >> 8
        s1 = v & 0xFF
        v = (s2 << 1) | carry
        carry = v >> 8
        s2 = v & 0xFF
        v = (s3 << 1) | carry
        carry = v >> 8
        s3 = v & 0xFF
        s4 = ((s4 << 1) | carry) & 0xFF
    mem[_PRND] = np.uint8(s0)
    mem[_PRND + 1] = np.uint8(s1)
    mem[_PRND + 2] = np.uint8(s2)
    mem[_PRND + 3] = np.uint8(s3)
    mem[_PRND + 4] = np.uint8(s4)
    return s4


@njit(cache=True, inline="always")
def _asl(v):
    return (v << 1) & 0xFF, (v >> 7) & 1


@njit(cache=True, inline="always")
def _rol(v, c):
    return ((v << 1) | c) & 0xFF, (v >> 7) & 1


@njit(cache=True, inline="always")
def _ror(v, c):
    return ((c << 7) | (v >> 1)) & 0xFF, v & 1


@njit(cache=True, inline="always")
def _full_over(ca, a, t74, t76, t77):
    """Rounds 1-3 of $0D4A: a >= b as a 16-bit compare (A:$74 vs $76:$77)."""
    if ca:
        return True
    if a > t76:
        return True
    if a == t76 and t74 >= t77:
        return True
    return False


@njit(cache=True)
def _finish_overflow(a, php_carry, t78):
    """consider_overflow $0DFC: the round-3 bit, the 45-degree clamp, then the
    arctan lookup + rounds-9/10 interpolation."""
    carry = 0
    if php_carry:
        s = a + 0x1F + 1
        a = s & 0xFF
        carry = 1 if s > 0xFF else 0
    if carry:
        return np.int64(0x00), np.int64(0x20), np.int64(0xFF)

    y = a & 0xFF
    ratio = y
    ang_lo = ARCTAN_LO[y]
    ang_hi = ARCTAN_HI[y]
    b78_7 = (t78 >> 7) & 1
    b78_6 = (t78 >> 6) & 1
    if b78_7 == 0 and b78_6 == 0:
        return np.int64(ang_lo), np.int64(ang_hi), np.int64(ratio)

    nxt_lo = ARCTAN_LO[y + 1]
    nxt_hi = ARCTAN_HI[y + 1]
    d_lo = (ang_lo - nxt_lo) & 0xFF
    borrow = 1 if ang_lo < nxt_lo else 0
    d_hi = (ang_hi - nxt_hi - borrow) & 0xFF
    if b78_6:  # round 9 over -> invert the delta
        d_hi, d_lo = _vinvert16(d_hi, d_lo)
    # ROL A ; ROR $0075 ; ROR $0074 -- arithmetic >>1 keeping the sign.
    t75, c2 = _ror(d_hi, (d_hi >> 7) & 1)
    t74b, _c = _ror(d_lo, c2)
    s = ang_lo + nxt_lo
    ang_lo = s & 0xFF
    cc = 1 if s > 0xFF else 0
    ang_hi = (ang_hi + nxt_hi + cc) & 0xFF
    if b78_7:  # round 10 over -> add half-delta
        s = ang_lo + t74b
        ang_lo = s & 0xFF
        cc = 1 if s > 0xFF else 0
        ang_hi = (ang_hi + t75 + cc) & 0xFF
    ang_hi, c3 = _ror(ang_hi, 0)
    ang_lo, _c2 = _ror(ang_lo, c3)
    return np.int64(ang_lo), np.int64(ang_hi), np.int64(ratio)


@njit(cache=True)
def _divide_and_arctan(a_lo, a_hi, b_lo, b_hi):
    """$0D4A: shift/subtract divide of 16-bit a by 16-bit b, then arctan(a/b)."""
    t74 = a_lo & 0xFF
    a = a_hi & 0xFF
    t76 = b_hi & 0xFF
    t77 = b_lo & 0xFF
    t78 = np.int64(0)

    t74, c = _asl(t74)
    a, ca = _rol(a, c)
    if _full_over(ca, a, t74, t76, t77):
        v = t74 - t77
        t74 = v & 0xFF
        a = (a - t76 - (1 if v < 0 else 0)) & 0xFF
        c = 1
    else:
        c = 0
    php_carry = 0
    for rnd in range(2, 4):
        t74, c0 = _rol(t74, c)
        a, ca = _rol(a, c0)
        if _full_over(ca, a, t74, t76, t77):
            v = t74 - t77
            t74 = v & 0xFF
            a = (a - t76 - (1 if v < 0 else 0)) & 0xFF
            c = 1
        else:
            c = 0
        if rnd == 3:
            php_carry = c  # PHP after round 3

    if a == t76:  # skip_further_division $0E10
        a = np.int64(0)
        cc = 1
        t74, cc = _ror(t74, cc)
        a, cc = _ror(a, cc)
        t74, cc = _ror(t74, cc)
        a, cc = _ror(a, cc)
        return _finish_overflow(a | 0x20, php_carry, np.int64(0))

    t74, c0 = _asl(t74)
    a, ca = _rol(a, c0)
    if ca or a >= t76:
        a = (a - t76) & 0xFF
        c = 1
    else:
        c = 0
    for _rnd in range(5, 10):
        t74, c0 = _rol(t74, c)
        a, ca = _rol(a, c0)
        if ca or a >= t76:
            a = (a - t76) & 0xFF
            c = 1
        else:
            c = 0
    # round 10 $0DF1: ROR $78 (round-9 bit), ROL A, compare, ROR $78
    t78, c0 = _ror(t78, c)
    a, ca = _rol(a, c0)
    if ca:
        c = 1
    else:
        c = 1 if a >= t76 else 0
    t78, _c = _ror(t78, c)
    return _finish_overflow(t74, php_carry, t78)


@njit(cache=True)
def _normalise(zp, max_lo_a, max_hi_a, min_lo_a, min_hi_a):
    """scale_using_x/_y $92C1/$92FF: shift max left until it overflows, min in
    lock-step, then back the max off by one.  Returns (b_lo, b_hi, a_lo, a_hi)."""
    max_lo = zp[max_lo_a]
    min_lo = zp[min_lo_a]
    min_hi = zp[min_hi_a]
    a = zp[max_hi_a]
    while True:
        max_lo, c = _asl(max_lo)
        a, ca = _rol(a, c)
        if ca:  # max overflowed
            break
        min_lo, c2 = _asl(min_lo)
        min_hi, _c = _rol(min_hi, c2)
    a, c = _ror(a, 1)
    max_lo, _c2 = _ror(max_lo, c)
    return max_lo & 0xFC, a, min_lo, min_hi


@njit(cache=True)
def _calc_angle(zp):
    """calculate_angle $9287: quadrant-folded arctan of (x, y)."""
    x_lo = zp[0x80]
    x_hi = zp[0x83]
    y_lo = zp[0x82]
    y_hi = zp[0x85]
    sx = zp[0x86]
    sy = zp[0x88]
    if (y_hi < x_hi) or (y_hi == x_hi and y_lo < x_lo):  # x is the larger
        zp[0x5D] = y_hi
        zp[0x5C] = y_lo
        zp[0x7A] = x_lo
        zp[0x7B] = x_hi
        b_lo, b_hi, a_lo, a_hi = _normalise(zp, 0x80, 0x83, 0x82, 0x85)
        ang_lo, ang_hi, ratio = _divide_and_arctan(a_lo, a_hi, b_lo, b_hi)
        if ((sx ^ sy) & 0x80) == 0:  # same sign -> negate angle
            ang_hi, ang_lo = _vinvert16(ang_hi, ang_lo)
        base = 0x40 if (sx & 0x80) == 0 else 0xC0
        ang_hi = (ang_hi + base) & 0xFF
    else:
        if (y_hi | y_lo) == 0:  # both zero
            zp[0x7E] = 0
            zp[0x8A] = 0
            zp[0x8B] = 0
            return
        zp[0x5D] = x_hi
        zp[0x5C] = x_lo
        zp[0x7A] = y_lo
        zp[0x7B] = y_hi
        b_lo, b_hi, a_lo, a_hi = _normalise(zp, 0x82, 0x85, 0x80, 0x83)
        ang_lo, ang_hi, ratio = _divide_and_arctan(a_lo, a_hi, b_lo, b_hi)
        if (sx ^ sy) & 0x80:  # opposite sign -> negate angle
            ang_hi, ang_lo = _vinvert16(ang_hi, ang_lo)
        base = 0x00 if (sy & 0x80) == 0 else 0x80
        ang_hi = (ang_hi + base) & 0xFF
    zp[0x8A] = ang_lo
    zp[0x8B] = ang_hi
    zp[0x7E] = ratio


@njit(cache=True)
def _calc_hypotenuse(zp):
    """calculate_hypotenuse $937F: distance = max + f*min/512."""
    ratio = zp[0x7E]
    f = HYP[(ratio >> 1) + (ratio & 1)]
    res_lo, res_hi = _vmul_dbl_by_byte(zp[0x5C], f, zp[0x5D])
    new_hi = res_hi >> 1
    new_lo = ((res_hi & 1) << 7) | (res_lo >> 1)
    s = new_lo + zp[0x7A]
    zp[0x7C] = s & 0xFF
    cc = 1 if s > 0xFF else 0
    zp[0x7D] = (new_hi + zp[0x7B] + cc) & 0xFF


@njit(cache=True)
def _vertical_angle(zp, z_hi, v_angle):
    """calculate_object_relative_vertical_angle $933D."""
    sx = z_hi & 0xFF
    if sx & 0x80:  # negative -> make positive
        zlo = (-zp[0x80]) & 0xFF
        borrow = 1 if zp[0x80] != 0 else 0
        sx_abs = (-sx - borrow) & 0xFF
        zp[0x80] = zlo
    else:
        sx_abs = sx
    zp[0x83] = sx_abs
    zp[0x82] = zp[0x7C]
    zp[0x85] = zp[0x7D]
    zp[0x88] = 0
    zp[0x86] = sx
    _calc_angle(zp)
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
    zp[0x50] = t50
    return a & 0xFF


@njit(cache=True)
def _relative_angles(mem, zp, observer, target):
    """calculate_object_relative_angles_and_distance $8401, play-mode path.

    The zero-page window is zeroed here, where the reference allocates its scratch.
    Returns (c57, angle_lo, angle_hi, z_lo, z_hi)."""
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
    _calc_angle(zp)
    c57 = (zp[0x8B] - _rd(mem, _OHANG + observer) + 0x0A) & 0xFF
    _calc_hypotenuse(zp)
    return c57, zp[0x8A], zp[0x8B], zp[0x81], zp[0x84]


@njit(cache=True)
def _prep_vec_angle(h_angle, h_frac, v_angle, v_frac):
    """prepare_vector_from_angle $1C54, the standalone entry the enemy probes use.

    Returns (vx_lo, vx_hi, vz_lo, vz_hi, vy_lo, vy_hi, s30)."""
    sin_lo_v, cos_lo_v, sin_hi_v, cos_hi_v = _vsin_cos(v_angle, v_frac)
    s33, s32 = _vproc_sc(cos_lo_v, cos_hi_v)
    s30, s2d = _vproc_sc(sin_lo_v, sin_hi_v)
    h_sin_lo, h_cos_lo, h_sin_hi, h_cos_hi = _vsin_cos(h_angle, h_frac)
    vy_hi, vy_lo = _vmul_dbl_dbl(h_cos_lo, h_cos_hi, s32, s33)
    vx_hi, vx_lo = _vmul_dbl_dbl(h_sin_lo, h_sin_hi, s32, s33)
    return vx_lo, vx_hi, s2d, s30, vy_lo, vy_hi, s30


@njit(cache=True)
def _can_see_object(mem, zp, observer, target, expected_type, fov_width):
    """check_if_enemy_can_see_object $1887.  A robot is probed at its upper point
    first ($18DC, $0C6E bit7 set so the looking-up rejection is waived) then at its
    base; every other object only at its base.

    Returns (in_slot, in_fov, exposure, full, tree_in_los_head)."""
    mem[0x0014] = 0
    if mem[_OFLAGS + target] & 0x80:  # empty slot
        return 0, 0, np.int64(0), 0, 0
    if mem[_OTYPE + target] != expected_type:
        return 0, 0, np.int64(0), 0, 0

    c57, angle_lo, angle_hi, z_lo, z_hi = _relative_angles(mem, zp, observer, target)
    a = (c57 - 0x0A + (fov_width >> 1)) & 0xFF
    if a >= fov_width:  # $18B8 FOV gate
        return 1, 0, np.int64(0), 0, 0

    v_angle_obs = _rd(mem, _OVANGLE + observer)
    mem[_TARGETED_SLOT] = np.uint8(target)
    ox = _rd(mem, _OX + observer)
    oy = _rd(mem, _OY + observer)
    obs_zf = _rd(mem, _OZF + observer)
    obs_zh = _rd(mem, _OZH + observer)
    n_probes = 2 if expected_type == _T_ROBOT else 1
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
        _vertical_angle(zp, phi, v_angle_obs)
        vx_lo, vx_hi, vz_lo, vz_hi, vy_lo, vy_hi, s30 = _prep_vec_angle(
            angle_hi, angle_lo, zp[0x8B], zp[0x8A]
        )
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
    return 1, 1, exposure, full, tree_head


@njit(cache=True, inline="always")
def _exposure_byte(in_slot, in_fov, exposure):
    """The ROM's object_exposure ($0014) from a can-see check."""
    if in_slot == 0 or in_fov == 0:
        return np.int64(0)
    return exposure


@njit(cache=True)
def _remove_object(mem, slot):
    """remove_object $1EEF: unlink the object and repair the tile it stood on."""
    tx = _rd(mem, _OX + slot)
    ty = _rd(mem, _OY + slot)
    flags = _rd(mem, _OFLAGS + slot)
    if flags >= 0x40:
        _set_tile_byte(mem, tx, ty, _OBJECT_TILE | (flags & 0x3F))
    else:
        _set_tile_byte(mem, tx, ty, (_rd(mem, _OZH + slot) << 4) & 0xFF)
    _wr(mem, _OFLAGS + slot, 0x80)


@njit(cache=True)
def _create_object(mem, otype):
    """create_object $210E: the highest empty slot, typed `otype`, or -1."""
    for slot in range(_NUM_SLOTS - 1, -1, -1):
        if mem[_OFLAGS + slot] & 0x80:
            _wr(mem, _OTYPE + slot, otype)
            return slot
    return -1


@njit(cache=True)
def _random_tile_coord(mem):
    """get_random_tile_coordinate $125A: a prnd draw masked to 0..31, rejecting 31."""
    while True:
        v = _prng_next(mem) & 0x1F
        if v != 0x1F:
            return v


@njit(cache=True)
def _put_object_in_tile(mem, slot, tx, ty):
    """put_object_in_tile $1EFF for a bare flat tile."""
    b = _tile_byte(mem, tx, ty)
    _wr(mem, _OX + slot, tx)
    _wr(mem, _OY + slot, ty)
    _wr(mem, _OFLAGS + slot, 0x00)
    _wr(mem, _OZF + slot, 0xE0)
    _wr(mem, _OZH + slot, (b >> 4) & 0xFF)
    _set_tile_byte(mem, tx, ty, _OBJECT_TILE | slot)
    _wr(mem, _OVANGLE + slot, 0xF5)
    _wr(mem, _OHANG + slot, (_prng_next(mem) & 0xF8) + 0x60)


@njit(cache=True)
def _put_object_in_random_tile_below_z(mem, slot, z):
    """put_object_in_random_tile_below_z $1224: a random flat, empty tile no higher
    than `z`; after 256 misses the ceiling rises, and it fails once it reaches 12."""
    attempts = 0
    while True:
        attempts = (attempts - 1) & 0xFF
        if attempts == 0:  # $122E: 256 misses -> relax the height ceiling
            z = (z + 1) & 0xFF
            if z >= 0x0C:
                return False
        tx = _random_tile_coord(mem)
        ty = _random_tile_coord(mem)
        b = _tile_byte(mem, tx, ty)
        if b >= _OBJECT_TILE:  # tile already holds an object
            continue
        if b & 0x0F:  # not flat
            continue
        if (b >> 4) >= z:  # too high
            continue
        _put_object_in_tile(mem, slot, tx, ty)
        return True


@njit(cache=True, inline="always")
def _discharge_bank(mem, enemy):
    """increase_enemy_energy_to_discharge $1A4F."""
    _wr(mem, _DISCHARGE + enemy, _rd(mem, _DISCHARGE + enemy) + 1)


@njit(cache=True)
def _reduce_object_energy(mem, target, enemy):
    """reduce_object_energy $1A08: drain `target`, banking a unit on `enemy`."""
    if target == _rd(mem, _PLAYER):
        if _rd(mem, _ENERGY) == 0:  # kill_player $1A00
            _wr(mem, _DIED_DRAINING, _rd(mem, _DIED_DRAINING) | 0x80)
            return True
        _wr(mem, _ENERGY, (_rd(mem, _ENERGY) - 1) & _ENERGY_MASK)
        _discharge_bank(mem, enemy)
        return True
    otype = _rd(mem, _OTYPE + target)
    if otype == _T_ROBOT:
        _wr(mem, _DRAIN_CD + enemy, 0)  # $1A31
        _wr(mem, _OTYPE + target, _T_BOULDER)
    elif otype == _T_TREE:
        _remove_object(mem, target)
    else:  # boulder -> tree
        _wr(mem, _OTYPE + target, _T_TREE)
    _discharge_bank(mem, enemy)
    return False


@njit(cache=True)
def _consider_discharging_enemy_energy(mem, enemy):
    """consider_discharging_enemy_energy $1A5D: return one banked unit to the
    landscape as a tree on a random flat tile."""
    if mem[_DISCHARGE + enemy] == 0:  # $1A63: nothing to discharge
        return False
    slot = _create_object(mem, _T_TREE)  # $1A65
    if slot < 0:
        return False
    if not _put_object_in_random_tile_below_z(mem, slot, _rd(mem, _BELOW_Z)):
        return False  # $1A70: no tile found -> abandon
    _wr(mem, _DISCHARGE + enemy, _rd(mem, _DISCHARGE + enemy) - 1)  # $1A7A
    return True


@njit(cache=True)
def _do_hyperspace(mem):
    """do_hyperspace $2147: a synthoid on a random low tile, energy spent, player
    transferred; too little energy kills, and doing it from the platform wins."""
    slot = _create_object(mem, _T_ROBOT)
    if slot < 0:
        return
    player = _rd(mem, _PLAYER)
    z = (_rd(mem, _OZH + player) + 1) & 0xFF
    if not _put_object_in_random_tile_below_z(mem, slot, z):
        _wr(mem, _OFLAGS + slot, _rd(mem, _OFLAGS + slot) | 0x80)  # $2159
        return
    if _rd(mem, _ENERGY) < _ROBOT_ENERGY:  # $215F: out of energy -> death
        _remove_object(mem, slot)
        _wr(mem, _HYPERSPACED, 0x80)
        return
    _wr(mem, _ENERGY, (_rd(mem, _ENERGY) - _ROBOT_ENERGY) & _ENERGY_MASK)
    on_platform = _rd(mem, _OX + player) == _rd(mem, _PLATFORM_X) and _rd(
        mem, _OY + player
    ) == _rd(mem, _PLATFORM_Y)
    if on_platform:  # $2187: hyperspacing from the platform completes the landscape
        _wr(mem, _COMPLETE, 0xC0)
    _wr(mem, _PLAYER, slot)


@njit(cache=True)
def _find_drainable_boulder_or_tree(mem, zp, enemy):
    """find_drainable_boulder_or_tree_on_stack $1AB0; -1 when nothing is drainable."""
    for x in range(_NUM_SLOTS - 1, -1, -1):
        flags = _rd(mem, _OFLAGS + x)
        if flags & 0x80:  # empty slot
            continue
        if not (flags >= 0x40 or _rd(mem, _OTYPE + x) == _T_BOULDER):
            continue
        tb = _tile_byte(mem, _rd(mem, _OX + x), _rd(mem, _OY + x))
        if tb < _OBJECT_TILE:
            continue
        y = tb & 0x3F  # topmost object of the tile
        otype = _rd(mem, _OTYPE + y)
        if otype != _T_TREE and otype != _T_BOULDER:
            continue
        _in_slot, _in_fov, _exp, full, _th = _can_see_object(
            mem, zp, enemy, y, otype, _FOV_SCAN
        )
        if full:
            _wr(mem, _TARGETED_SLOT, y)
            return y
    return -1


@njit(cache=True)
def _initialise_enemy_meanie_variables(mem, enemy):
    """initialise_enemy_meanie_variables $196A: (re)arm an enemy's meanie hunt."""
    _wr(mem, _M_OBJECT + enemy, 0x80)
    _wr(mem, _M_FAILED + enemy, 0x80)
    _wr(mem, _M_SCANS + enemy, 0)
    _wr(mem, _M_SEARCH + enemy, 0x40)


@njit(cache=True)
def _consider_creating_meanie(mem, zp, enemy):
    """consider_creating_meanie $197D: the first fully-visible tree within 10 tiles
    of the targeted player, in both axes, becomes a meanie owned by `enemy`."""
    player = _rd(mem, _TARGET + enemy)
    while True:
        sc = _rd(mem, _M_SEARCH + enemy)
        if sc == 0:  # $198D: scanned everything -> no meanie this pass
            _wr(mem, _M_SCANS + enemy, _rd(mem, _M_SCANS + enemy) + 1)
            _wr(mem, _M_FAILED + enemy, player)
            return False
        _wr(mem, _M_SEARCH + enemy, sc - 1)
        slot = sc - 1  # $199B DEY
        if mem[_OFLAGS + slot] & 0x80:
            continue
        if mem[_OTYPE + slot] != _T_TREE:
            continue
        dx = (_rd(mem, _OX + player) - _rd(mem, _OX + slot)) & 0xFF
        if dx >= 0x80:
            dx = 0x100 - dx  # $19B5 abs
        if dx >= 0x0A:
            continue
        dy = (_rd(mem, _OY + player) - _rd(mem, _OY + slot)) & 0xFF
        if dy >= 0x80:
            dy = 0x100 - dy
        if dy >= 0x0A:
            continue
        _in_slot, _in_fov, _exp, full, _th = _can_see_object(
            mem, zp, enemy, slot, _T_TREE, _FOV_CREATE_MEANIE
        )
        if not full:
            continue
        _wr(mem, _M_OBJECT + enemy, slot)  # $19E1
        _wr(mem, _OTYPE + slot, _T_MEANIE)
        return True


@njit(cache=True)
def _remove_meanie(mem, enemy):
    """remove_meanie $1754: turn the meanie back into a tree."""
    meanie = _rd(mem, _M_OBJECT + enemy)
    _wr(mem, _M_OBJECT + enemy, 0x80)
    _wr(mem, _OTYPE + meanie, _T_TREE)


@njit(cache=True)
def _remove_meanie_and_reset_enemy(mem, enemy):
    """remove_meanie_and_reset_enemy $174F: also clear the draining cooldown."""
    _wr(mem, _DRAIN_CD + enemy, 0)
    _remove_meanie(mem, enemy)


@njit(cache=True)
def _update_meanie(mem, zp, enemy):
    """update_meanie $16F2: rotate toward the player, then force a hyperspace."""
    meanie = _rd(mem, _M_OBJECT + enemy)
    target = _rd(mem, _TARGET + enemy)
    if mem[_OFLAGS + target] & 0x80:  # $16F7: the object the player was in is gone
        _remove_meanie_and_reset_enemy(mem, enemy)
        return
    in_slot, in_fov, exposure, _full, _th = _can_see_object(
        mem, zp, meanie, target, _T_ROBOT, _FOV_SCAN
    )
    if in_fov == 0:  # $1706: not yet looking at the player -> rotate
        c57 = _relative_angles(mem, zp, meanie, target)[0]
        step = _MEANIE_ROTATE_STEP
        if c57 & 0x80:
            step = 0x100 - _MEANIE_ROTATE_STEP
        _wr(mem, _OHANG + meanie, _rd(mem, _OHANG + meanie) + step)
        _wr(mem, _UPD_CD + enemy, _UPD_CD_MEANIE_ROTATE)
        return
    if target != _rd(mem, _PLAYER):  # $1708: player transferred out of the object
        _remove_meanie_and_reset_enemy(mem, enemy)
        return
    if _exposure_byte(in_slot, in_fov, exposure) == 0:  # $170E
        _remove_meanie(mem, enemy)
        return
    _do_hyperspace(mem)  # $1710: forced hyperspace


@njit(cache=True)
def _target_object(mem, zp, enemy, target, exposure):
    """target_object $1825: record the target and drain it when the timer expires."""
    _wr(mem, _TARGET + enemy, target)
    _wr(mem, _TARGET_EXP + enemy, exposure)
    cd = _rd(mem, _DRAIN_CD + enemy)
    if cd < 0x01:  # first sight -> arm the drain timer
        _wr(mem, _DRAIN_CD + enemy, _DRAIN_CD_RELOAD)
        return
    if cd != 0x01:  # still counting down
        return
    if exposure & 0x80:  # fully visible -> drain
        _wr(mem, _TARGETED_SLOT, target)
        killed = target == _rd(mem, _PLAYER) and _rd(mem, _ENERGY) == 0
        _reduce_object_energy(mem, target, enemy)
        if killed:  # kill_player $1A00 unwinds the stack
            return
        _wr(mem, _UPD_CD + enemy, _UPD_CD_DRAIN)
        return
    if _consider_creating_meanie(mem, zp, enemy):  # $184D
        _wr(mem, _UPD_CD + enemy, _UPD_CD_MEANIE_MADE)
        return
    if _rd(mem, _M_SCANS + enemy) >= _MEANIE_MAX_ATTEMPTS:
        _wr(mem, _DRAIN_CD + enemy, 0)  # give up on this player
    else:
        _wr(mem, _CONSIDERING + enemy, 0x80)  # keep trying next time


@njit(cache=True)
def _rotate_enemy(mem, enemy):
    """rotate_enemy $1805: add the per-enemy step to the facing, reload $C8."""
    _wr(mem, _OHANG + enemy, _rd(mem, _OHANG + enemy) + _rd(mem, _ROT_SPEED + enemy))
    _wr(mem, _ROT_CD + enemy, _ROT_CD_RELOAD)
    _initialise_enemy_meanie_variables(mem, enemy)  # $1818


@njit(cache=True)
def _consider_enemy_state(mem, zp, enemy):
    """consider_enemy_state $16E6: the meanie/discharge/drain/rotate decision."""
    if mem[_UPD_CD + enemy] >= _COOLDOWN_STICK:
        return
    _wr(mem, _UPD_CD + enemy, _UPD_CD_SCAN)
    _wr(mem, _FOV_WIDTH, _FOV_SCAN)

    if not (mem[_M_OBJECT + enemy] & 0x80):  # $16EA: already owns a meanie
        _update_meanie(mem, zp, enemy)
        return

    if _consider_discharging_enemy_energy(mem, enemy):  # $1773
        return

    if mem[_CONSIDERING + enemy] & 0x80:  # $177F: mid meanie-hunt
        tb = _find_drainable_boulder_or_tree(mem, zp, enemy)
        if tb >= 0:
            _wr(mem, _M_SEARCH + enemy, 0x40)  # $178B
            _reduce_object_energy(mem, tb, enemy)
            _wr(mem, _UPD_CD + enemy, _UPD_CD_DRAIN)
            return
        _wr(mem, _CONSIDERING + enemy, _rd(mem, _CONSIDERING + enemy) >> 1)

    if mem[_DRAIN_CD + enemy] != 0:  # $178C: re-check a held target
        held = _rd(mem, _TARGET + enemy)
        in_slot, in_fov, exp_raw, _full, _th = _can_see_object(
            mem, zp, enemy, held, _T_ROBOT, _FOV_SCAN
        )
        exposure = _exposure_byte(in_slot, in_fov, exp_raw)
        if exposure != 0:
            _target_object(mem, zp, enemy, held, exposure)
            return
        _wr(mem, _DRAIN_CD + enemy, 0)  # target lost

    player = _rd(mem, _PLAYER)  # find_drainable_robot_loop $17B2
    partial_player = -1
    for y in range(_NUM_SLOTS - 1, -1, -1):
        in_slot, in_fov, exp_raw, _full, tree_head = _can_see_object(
            mem, zp, enemy, y, _T_ROBOT, _FOV_SCAN
        )
        if tree_head:  # $17B7: a tree hides this robot's head
            continue
        exposure = _exposure_byte(in_slot, in_fov, exp_raw)
        if exposure == 0:  # $17BE: not visible at all
            continue
        if exposure & 0x80:  # $17BA: fully visible -> drain target
            _target_object(mem, zp, enemy, y, exposure)
            return
        if y == player:  # $17C0: head only -> meanie candidate
            partial_player = y
    if partial_player >= 0 and partial_player != _rd(mem, _M_FAILED + enemy):  # $17C4
        _initialise_enemy_meanie_variables(mem, enemy)
        _target_object(mem, zp, enemy, partial_player, 0x40)
        return

    _wr(mem, _DRAIN_CD + enemy, 0)  # $17E0
    tb = _find_drainable_boulder_or_tree(mem, zp, enemy)
    if tb >= 0:
        _wr(mem, _TARGETED_SLOT, tb)
        _reduce_object_energy(mem, tb, enemy)
        _wr(mem, _UPD_CD + enemy, _UPD_CD_DRAIN)
        return

    if mem[_ROT_CD + enemy] < _COOLDOWN_STICK:  # $17F9 no_drain
        _rotate_enemy(mem, enemy)


@njit(cache=True)
def _tick_cooldowns(mem):
    """update_enemy_cooldowns $1317: the 1-in-3 gate, then every cooldown >= 2."""
    if mem[_GATE] != 0:
        _wr(mem, _GATE, _rd(mem, _GATE) - 1)
        return
    for addr in range(_DRAIN_CD, _UPD_CD + 8):
        if mem[addr] >= _COOLDOWN_STICK:
            mem[addr] = np.uint8(mem[addr] - 1)
    _wr(mem, _GATE, 2)


@njit(cache=True)
def _update_enemies(mem, zp):
    """update_enemies $16B5: consider the enemy at the cursor, advance prnd and the
    cursor (7->0 wrap)."""
    x = _rd(mem, _CURSOR)
    otype = _rd(mem, _OTYPE + x)
    if otype == _T_SENTRY or otype == _T_SENTINEL:  # $16BB
        if not (mem[_OFLAGS + x] & 0x80):  # $16CC: not absorbed
            _consider_enemy_state(mem, zp, x)
        else:  # $16CE: an absorbed slot still discharges its bank
            _consider_discharging_enemy_energy(mem, x)
    _prng_next(mem)  # $16D6
    c = _rd(mem, _CURSOR)  # $16D9
    _wr(mem, _CURSOR, (c - 1) if c > 0 else 7)


@njit(cache=True)
def _cooldown_frame(mem):
    """$130C: the per-frame Bresenham gate on update_enemy_cooldowns."""
    if mem[_NOT_ACTED] & 0x80:  # player has not yet acted
        return
    acc = _rd(mem, _BRESENHAM) + _BRESENHAM_STEP
    _wr(mem, _BRESENHAM, acc)
    if acc > 0xFF:  # $1315 BCC skip
        _tick_cooldowns(mem)


@njit(cache=True)
def _advance(mem, zp, n_frames, plotting, updates_per_frame):
    """The frame loop: the raster cooldown tick, then the foreground sweep."""
    for _ in range(n_frames):
        _cooldown_frame(mem)
        if not plotting:
            for _u in range(updates_per_frame):
                _update_enemies(mem, zp)


def advance_frames(mem, n_frames, plotting, updates_per_frame):
    """Advance ``n_frames`` video frames on the caller's 64 KB ``bytearray``.

    The numpy view shares the caller's buffer, so every mutation lands in the state
    the caller holds -- exactly as :func:`sentinel.los._march_jit` does."""
    view = np.frombuffer(mem, dtype=np.uint8)
    zp = np.zeros(ZP_HI, dtype=np.int64)
    _advance(view, zp, int(n_frames), bool(plotting), int(updates_per_frame))
