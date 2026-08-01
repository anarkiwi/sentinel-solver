"""Object-relative bearing, distance and vertical angle -- the geometry an enemy
uses to decide whether it can see a target.

This is the bit-exact port of the game's fixed-point trig:

  * ``_divide_and_arctan``  ($0D4A) -- shift/subtract 16-bit divide of a/b,
    arctan lookup + rounds-9/10 interpolation.
  * ``calculate_angle``     ($9287) -- quadrant-folded arctan of (x, y).
  * ``calculate_hypotenuse``($937F) -- distance from the divide's residue.
  * ``relative_angles``     ($8401) -- relative x/y ($85C4), z ($85F5), then the
    horizontal bearing and distance of a target from an observer, as the game
    stores them in $0C57/$0C59/$0C5B..$0C5E.
  * ``vertical_angle``      ($933D) -- the target's vertical angle from distance.
  * ``can_see_object``      ($1887) -- the enemy's field-of-view gate plus the
    two-height terrain line-of-sight probe (reusing :mod:`sentinel.los`).

The arctan/hypotenuse coefficient tables the ROM keeps at $3B00/$3C01 and $3D02
are reproduced here from closed-form expressions (verified to match the ROM byte
for byte), so no game data is embedded.
"""

import collections
import math

from sentinel import memmap as mm, los, passcost

# ---------------------------------------------------------------------------
# Coefficient tables, reproduced from closed form (byte-exact vs ROM).
# ---------------------------------------------------------------------------
# arctan: angle (16-bit, full circle = $10000) of ratio Y/256, for Y in 0..256.
_ARCTAN = [round(math.atan(y / 256.0) / (2 * math.pi) * 65536.0) for y in range(257)]
_ARCTAN_LO = [a & 0xFF for a in _ARCTAN]  # $3B00,Y
_ARCTAN_HI = [(a >> 8) & 0xFF for a in _ARCTAN]  # $3C01,Y

# hypotenuse coefficient f(Y): hyp = max + f*min/512, r = Y/128, Y in 0..128.
_HYP = [0] + [
    round(512.0 * (math.sqrt(1 + (y / 128.0) ** 2) - 1) / (y / 128.0)) & 0xFF
    for y in range(1, 129)
]  # $3D02,Y


def _asl(v):
    return (v << 1) & 0xFF, (v >> 7) & 1


def _rol(v, c):
    return ((v << 1) | c) & 0xFF, (v >> 7) & 1


def _ror(v, c):
    return ((c << 7) | (v >> 1)) & 0xFF, v & 1


def _invert16(high, frac):
    """invert_A_and_a_fraction $1009: two's-complement negate 16-bit (high.frac)."""
    frac = (-frac) & 0xFF
    borrow = 1 if frac != 0 else 0
    high = (-high - borrow) & 0xFF
    return high, frac


# ---------------------------------------------------------------------------
# divide_and_arctan $0D4A
# ---------------------------------------------------------------------------
def _divide_and_arctan(a_lo, a_hi, b_lo, b_hi):
    """$0D4A: divide 16-bit a by 16-bit b (a <= b). Returns
    (angle_lo=$008A, angle_hi=$008B, ratio=$007E, cycles) with angle = arctan(a/b)
    as a 16-bit value (full circle = $10000)."""
    t74 = a_lo & 0xFF  # a low / quotient accumulator
    A = a_hi & 0xFF  # a high / working remainder
    t76 = b_hi & 0xFF
    t77 = b_lo & 0xFF
    t78 = 0  # result_low (rounds 9/10 bits)
    cyc = 0

    def full_over(cA):
        # rounds 1-3, a >= b as a 16-bit compare (A:$74 vs $76:$77), and its branches
        if cA:
            return True, passcost.DIV_FULL_CARRY
        if A > t76:
            return True, passcost.DIV_FULL_OVER
        if A < t76:
            return False, passcost.DIV_FULL_UNDER
        if t74 >= t77:
            return True, passcost.DIV_FULL_EQ + passcost.DIV_FULL_EQ_OVER
        return False, passcost.DIV_FULL_EQ + passcost.DIV_FULL_EQ_UNDER

    # rounds 1..3: ASL $74 then ROL $74, each with a full 16-bit compare.
    c = 0
    php_carry = 0
    for rnd in (1, 2, 3):
        t74, c0 = _asl(t74) if rnd == 1 else _rol(t74, c)
        A, cA = _rol(A, c0)
        over, dcyc = full_over(cA)
        cyc += passcost.DIV_FULL_SHIFT + dcyc
        if over:
            v = t74 - t77
            t74 = v & 0xFF
            A = (A - t76 - (1 if v < 0 else 0)) & 0xFF
            c = 1
            cyc += passcost.DIV_SUB
        else:
            c = 0
        if rnd == 3:
            php_carry = c  # PHP after round 3

    # skip_further_division ($0E10): remainder == b_hi after round 3.
    cyc += passcost.DIV_SKIP_TEST
    if A == t76:
        cyc += passcost.DIV_SKIP
        t78 = 0  # LDA #$0 ; STA $0078
        A = 0  # LDA #$0 -> the RORs work on A=0, not the remainder
        cc = 1  # CMP equal -> carry set
        t74, cc = _ror(t74, cc)
        A, cc = _ror(A, cc)
        t74, cc = _ror(t74, cc)
        A, cc = _ror(A, cc)
        A = A | 0x20
        return _finish_overflow(A, php_carry, t78, cyc)
    cyc += passcost.DIV_NO_SKIP

    # rounds 4..9: ASL $74 then ROL $74, each with an 8-bit compare.
    for rnd in range(4, 10):
        t74, c0 = _asl(t74) if rnd == 4 else _rol(t74, c)
        A, cA = _rol(A, c0)
        cyc += passcost.DIV_BYTE_SHIFT
        if cA:
            cyc += passcost.DIV_BYTE_CARRY
        elif A >= t76:
            cyc += passcost.DIV_BYTE_OVER
        else:
            cyc += passcost.DIV_BYTE_UNDER
        if cA or A >= t76:
            A = (A - t76) & 0xFF
            c = 1
        else:
            c = 0
    # round 10 ($0DF1): ROR $78 (round-9 bit), ROL A, compare, ROR $78.
    t78, c0 = _ror(t78, c)  # ROR $0078: shift in round-9 quotient bit
    A, cA = _rol(A, c0)  # ROL A
    cyc += passcost.DIV_LAST_SHIFT + passcost.DIV_LAST_TAIL
    if cA:
        c = 1
        cyc += passcost.DIV_LAST_CARRY
    else:
        c = 1 if A >= t76 else 0  # CMP $0076
        cyc += passcost.DIV_LAST_CMP
    t78, _ = _ror(t78, c)  # ROR $0078: shift in round-10 bit

    return _finish_overflow(t74, php_carry, t78, cyc)


def _finish_overflow(A, php_carry, t78, cyc):
    """consider_overflow ($0DFC): apply the round-3 bit ($20), detect the 45-degree
    overflow, then the arctan lookup + rounds-9/10 interpolation."""
    cyc += passcost.DIV_PLP
    if php_carry:  # round 3 was over
        s = A + 0x1F + 1
        A = s & 0xFF
        carry = 1 if s > 0xFF else 0
        cyc += passcost.DIV_ROUND3_OVER
    else:
        carry = 0  # BCC round_3_under with carry from prior op == 0
        cyc += passcost.DIV_ROUND3_UNDER
    if carry:  # overflow: clamp to 45 degrees
        return 0x00, 0x20, 0xFF, cyc + passcost.DIV_OVERFLOW
    if php_carry:
        cyc += passcost.DIV_NO_OVERFLOW

    # no_overflow ($0E1F): arctan lookup + rounds 9/10 interpolation.
    Y = A & 0xFF
    ratio = Y
    ang_lo = _ARCTAN_LO[Y]
    ang_hi = _ARCTAN_HI[Y]
    cyc += passcost.DIV_ARCTAN + (passcost.DIV_TABLE_CROSS if Y >= 0xFF else 0)
    b78_7 = (t78 >> 7) & 1  # round 10 over
    b78_6 = (t78 >> 6) & 1  # round 9 over
    if not (b78_7 or b78_6):
        cyc += passcost.DIV_ARCTAN_DONE
        return ang_lo, ang_hi, ratio, cyc  # BMI/BVS both false -> leave
    # An interpolation block reads $3B01,Y and $3C02,Y, a cycle dearer off-page each
    cross = (passcost.DIV_TABLE_CROSS if Y >= 0xFF else 0) + (
        passcost.DIV_TABLE_CROSS if Y >= 0xFE else 0
    )

    # calculate_delta ($0E35): delta = arctan[Y] - arctan[Y+1]
    d_lo = (ang_lo - _ARCTAN_LO[Y + 1]) & 0xFF
    borrow = 1 if ang_lo < _ARCTAN_LO[Y + 1] else 0
    d_hi = (ang_hi - _ARCTAN_HI[Y + 1] - borrow) & 0xFF
    if b78_7:
        cyc += passcost.DIV_DELTA_BR + passcost.DIV_DELTA + cross
        if b78_6:  # round 9 over ($0E44 BVC skip; BVS -> invert)
            d_hi, d_lo = _invert16(d_hi, d_lo)
            cyc += passcost.DIV_DELTA_INVERT
        else:
            cyc += passcost.DIV_DELTA_KEEP
        cyc += passcost.DIV_DELTA_HALVE
    else:
        cyc += passcost.DIV_HALF_BR  # $0E30 BVS $0E50: only the average is wanted
    # ROL A puts bit7(d_hi) in the carry, ROR $0075/$0074 shift it back: a signed >>1
    t75, c2 = _ror(d_hi, (d_hi >> 7) & 1)
    t74b, _ = _ror(d_lo, c2)
    d_hi2, d_lo2 = t75, t74b
    # angle += arctan[Y+1]  ($0E50)
    s = ang_lo + _ARCTAN_LO[Y + 1]
    ang_lo = s & 0xFF
    cc = 1 if s > 0xFF else 0
    ang_hi = (ang_hi + _ARCTAN_HI[Y + 1] + cc) & 0xFF
    cyc += passcost.DIV_NEXT + cross
    if b78_7:  # round 10 over ($0E61 BPL skip): add half-delta
        s = ang_lo + d_lo2
        ang_lo = s & 0xFF
        cc = 1 if s > 0xFF else 0
        ang_hi = (ang_hi + d_hi2 + cc) & 0xFF
        cyc += passcost.DIV_ADD_HALF
    else:
        cyc += passcost.DIV_ADD_SKIP
    # LSR $008B ; ROR $008A  -> average
    ang_hi, c3 = _ror(ang_hi, 0)
    ang_lo, _ = _ror(ang_lo, c3)
    return ang_lo, ang_hi, ratio, cyc + passcost.DIV_AVERAGE


# ---------------------------------------------------------------------------
# calculate_angle $9287 (quadrant-folded arctan of x, y)
# ---------------------------------------------------------------------------
def _calc_angle(zp):
    """$9287: horizontal angle of the vector (x, y), and the cycles it cost.

    x = (zp[$80], zp[$83]), y = (zp[$82], zp[$85]), signs zp[$86]/zp[$88]. Writes
    the angle to zp[$8A]/zp[$8B], the ratio to zp[$7E] and the min/max magnitudes
    to zp[$5C..$5D]/zp[$7A..$7B] for calculate_hypotenuse."""
    x_lo, x_hi = zp[0x80], zp[0x83]
    y_lo, y_hi = zp[0x82], zp[0x85]
    sx, sy = zp[0x86], zp[0x88]

    cyc = passcost.ANG_CMP
    if y_hi < x_hi:
        x_larger = True
        cyc += passcost.ANG_X_LARGER
    elif y_hi > x_hi:
        x_larger = False
        cyc += passcost.ANG_Y_LARGER
    else:
        x_larger = y_lo < x_lo
        cyc += passcost.ANG_EQ + (passcost.ANG_EQ_X if x_larger else passcost.ANG_EQ_Y)

    if x_larger:
        cyc += passcost.ANG_MIN_Y
        zp[0x5D], zp[0x5C] = y_hi, y_lo  # min = y
        zp[0x7A], zp[0x7B] = x_lo, x_hi  # max = x
        b_lo, b_hi, a_lo, a_hi, ncyc = _normalise(
            zp, 0x80, 0x83, 0x82, 0x85, passcost.SCALE_LOOP
        )
        ang_lo, ang_hi, ratio, dcyc = _divide_and_arctan(a_lo, a_hi, b_lo, b_hi)
        cyc += ncyc + dcyc + passcost.ANG_SIGN
        if not ((sx ^ sy) & 0x80):  # same sign -> negate angle
            ang_hi, ang_lo = _invert16(ang_hi, ang_lo)
            cyc += passcost.ANG_SIGN_NEGATE
        else:
            cyc += passcost.ANG_SIGN_KEEP
        base = 0x40 if not (sx & 0x80) else 0xC0  # +90 or +270 degrees
        cyc += passcost.ANG_QUAD + (
            passcost.ANG_QUAD_LOW if base == 0x40 else passcost.ANG_QUAD_HIGH
        )
        ang_hi = (ang_hi + base) & 0xFF
    else:
        cyc += passcost.ANG_MIN_X
        if (y_hi | y_lo) == 0:  # both zero
            zp[0x7E] = zp[0x8A] = zp[0x8B] = 0
            return cyc + passcost.ANG_ZERO
        cyc += passcost.ANG_NONZERO
        zp[0x5D], zp[0x5C] = x_hi, x_lo  # min = x
        zp[0x7A], zp[0x7B] = y_lo, y_hi  # max = y
        b_lo, b_hi, a_lo, a_hi, ncyc = _normalise(
            zp, 0x82, 0x85, 0x80, 0x83, passcost.SCALE_LOOP_Y
        )
        ang_lo, ang_hi, ratio, dcyc = _divide_and_arctan(a_lo, a_hi, b_lo, b_hi)
        cyc += ncyc + dcyc + passcost.ANG_SIGN
        if (sx ^ sy) & 0x80:  # opposite sign -> negate angle
            ang_hi, ang_lo = _invert16(ang_hi, ang_lo)
            cyc += passcost.ANG_SIGN_NEGATE
        else:
            cyc += passcost.ANG_SIGN_KEEP
        base = 0x00 if not (sy & 0x80) else 0x80  # +0 or +180 degrees
        cyc += passcost.ANG_QUAD + (
            passcost.ANG_QUAD_LOW if base == 0x00 else passcost.ANG_QUAD_HIGH
        )
        ang_hi = (ang_hi + base) & 0xFF

    zp[0x8A], zp[0x8B], zp[0x7E] = ang_lo, ang_hi, ratio
    return cyc + passcost.ANG_QUAD_TAIL


def _normalise(zp, max_lo_a, max_hi_a, min_lo_a, min_hi_a, lap):
    """The scale_using_x/_y normalisation ($92C1/$92FF): shift the larger vector
    (max) left until it overflows, shifting the smaller (min) in lock-step, then
    back the max off by one. Returns the divide inputs (b_lo, b_hi, a_lo, a_hi)
    and the variable-length loop's own cycles, `lap` being one loop-back's cost."""
    max_lo, max_hi = zp[max_lo_a], zp[max_hi_a]
    min_lo, min_hi = zp[min_lo_a], zp[min_hi_a]
    A = max_hi
    laps = 0
    while True:
        max_lo, c = _asl(max_lo)
        A, cA = _rol(A, c)
        if cA:  # max overflowed
            break
        min_lo, c2 = _asl(min_lo)
        min_hi, _ = _rol(min_hi, c2)
        laps += 1
    # ROR A ; ROR max_lo  (back off the overflow shift)
    A, c = _ror(A, 1)  # carry in = 1 (the overflow bit)
    max_lo, _ = _ror(max_lo, c)
    b_hi = A
    a_lo = min_lo
    b_lo = max_lo & 0xFC  # AND #$fc: drop scaling noise
    a_hi = min_hi
    cyc = passcost.SCALE_SHIFT + laps * lap + passcost.SCALE_TAIL
    return b_lo, b_hi, a_lo, a_hi, cyc


# ---------------------------------------------------------------------------
# calculate_hypotenuse $937F
# ---------------------------------------------------------------------------
def _calc_hypotenuse(zp):
    """$937F: distance = max + f*min/512, with f from the $3D02 coefficient table
    indexed by round(ratio/2). Reads zp[$7E] (ratio), zp[$5C..$5D] (min),
    zp[$7A..$7B] (max); writes zp[$7C]/zp[$7D]. Returns its cycles."""
    ratio = zp[0x7E]
    y = (ratio >> 1) + (ratio & 1)  # LSR A ; ADC #$0  -> round(ratio/2)
    f = _HYP[y]
    res_lo, res_hi, mcyc = los.multiply_double_by_byte(zp[0x5C], f, zp[0x5D])
    # LSR $0075 ; ROR $0074  -> halve the 16-bit result
    new_hi = res_hi >> 1
    new_lo = ((res_hi & 1) << 7) | (res_lo >> 1)
    # + max
    s = new_lo + zp[0x7A]
    zp[0x7C] = s & 0xFF
    cc = 1 if s > 0xFF else 0
    zp[0x7D] = (new_hi + zp[0x7B] + cc) & 0xFF
    return passcost.HYP_HEAD + mcyc + passcost.HYP_TAIL


# ---------------------------------------------------------------------------
# calculate_object_relative_vertical_angle $933D
# ---------------------------------------------------------------------------
def _vertical_angle(zp, z_hi, v_angle):
    """$933D: the target's vertical angle, and the cycles it cost.

    z_hi is the (signed) relative-z high byte to test; zp[$80] holds its low byte;
    zp[$7C]/zp[$7D] hold the horizontal distance; v_angle is the observer's
    objects_v_angle. Returns (signed vertical angle byte zp[$8D], cycles)."""
    sx = z_hi & 0xFF
    cyc = passcost.VANG_HEAD
    if sx & 0x80:  # negative -> make positive
        zlo = (-zp[0x80]) & 0xFF
        borrow = 1 if zp[0x80] != 0 else 0
        sx_abs = (-sx - borrow) & 0xFF
        zp[0x80] = zlo
        cyc += passcost.VANG_NEG
    else:
        sx_abs = sx
        cyc += passcost.VANG_POS
    zp[0x83] = sx_abs  # x_hi = |rel z hi|
    zp[0x82] = zp[0x7C]  # y_lo = hyp_lo
    zp[0x85] = zp[0x7D]  # y_hi = hyp_hi
    zp[0x88] = 0  # signed_y = 0
    zp[0x86] = sx  # signed_x = original rel z hi
    cyc += passcost.VANG_SETUP + _calc_angle(zp) + passcost.VANG_SHIFT
    # $50 = angle_lo - $20 ; A = angle_hi - v_angle ; PHP ; >>4 (with sign)
    lo = zp[0x8A] - 0x20
    t50 = lo & 0xFF
    hi = (zp[0x8B] - v_angle - (1 if lo < 0 else 0)) & 0xFF
    A = hi
    neg = A & 0x80
    for _ in range(4):
        c = A & 1
        A = A >> 1
        t50 = ((c << 7) | (t50 >> 1)) & 0xFF
    if neg:
        A = A | 0xF0
        cyc += passcost.VANG_SIGN_NEG
    else:
        cyc += passcost.VANG_SIGN_POS
    zp[0x50] = t50  # $0050: the fine (low) byte of the vertical angle (screen y low)
    return A & 0xFF, cyc + passcost.VANG_TAIL


# ---------------------------------------------------------------------------
# relative x/y/z ($85C4 / $85F5) and the $8401 glue
# ---------------------------------------------------------------------------
def _relative_xyz(zp, obj, obs_x, obs_y, obs_zf, obs_zh, tgt_x, tgt_y, tgt_zf, tgt_zh):
    """calculate_object_relative_x_and_y ($85C4) + _z ($85F5): the signed and
    absolute component distances of a target from an observer, and their cycles."""
    cyc = passcost.REL_XY + passcost.REL_Z
    dx = (tgt_x - obs_x) & 0xFF  # $0C78 == 0 outside title screen
    zp[0x86] = dx
    zp[0x80] = 0
    zp[0x83] = (-dx) & 0xFF if dx & 0x80 else dx
    dy = (tgt_y - obs_y) & 0xFF
    zp[0x88] = dy
    zp[0x82] = 0
    zp[0x85] = (-dy) & 0xFF if dy & 0x80 else dy
    v = tgt_zf - obs_zf
    zp[0x81] = v & 0xFF
    zp[0x84] = (tgt_zh - obs_zh - (1 if v < 0 else 0)) & 0xFF
    del obj
    return cyc + passcost.REL_XY_ABS * ((dx >> 7) + (dy >> 7))


def relative_angles(state, observer, target):
    """calculate_object_relative_angles_and_distance ($8401), play-mode path
    ($001F=$0C78=0, not preview). Returns a dict with the FOV byte ($0C57), the
    horizontal angle ($008A/$008B), the relative z ($0081/$0084), the horizontal
    distance ($007C/$007D) and the target type ($004C)."""
    # A zero-page scratch: only a handful of addresses are ever touched, so a
    # defaultdict(int) (unset reads == 0, exactly like a pre-filled dict) is far
    # cheaper to allocate than a 256-entry comprehension on this hot enemy-scan path.
    zp = collections.defaultdict(int)
    otype = state.obj_type[target]
    cyc = passcost.REL_ANGLES + _relative_xyz(
        zp,
        target,
        state.obj_x[observer],
        state.obj_y[observer],
        state.obj_z_frac[observer],
        state.obj_z_height[observer],
        state.obj_x[target],
        state.obj_y[target],
        state.obj_z_frac[target],
        state.obj_z_height[target],
    )
    cyc += _calc_angle(zp)
    # $0C57 = angle_hi - h_angle[observer] + $0A ; $0C59 = angle_lo - $001F(=0)
    h_angle = state.obj_h_angle[observer]
    c59 = zp[0x8A]  # SBC $001F(0), carry stays set -> no borrow into the high byte
    c57 = (zp[0x8B] - h_angle + 0x0A) & 0xFF
    cyc += _calc_hypotenuse(zp)
    return {
        "cycles": cyc,
        "c57": c57,
        "c59": c59,
        "angle_lo": zp[0x8A],
        "angle_hi": zp[0x8B],
        "z_lo": zp[0x81],
        "z_hi": zp[0x84],
        "hyp_lo": zp[0x7C],
        "hyp_hi": zp[0x7D],
        "type": otype,
        "_zp": zp,
    }


# ---------------------------------------------------------------------------
# check_if_enemy_can_see_object $1887
# ---------------------------------------------------------------------------
def can_see_object(state, observer, target, expected_type, fov_width, max_steps=20000):
    """$1887: can `observer` see `target` (which must be `expected_type`) within
    the horizontal field of view `fov_width`?  Returns a dict with:
      in_slot   -- target occupies its slot and is the expected type
      in_fov    -- target bearing lies within the horizontal cone
      exposure  -- the ROM's object_exposure byte ($0014): $80 fully visible (base
                   reached), $40 partial (only the upper point reached), 0 unseen
      full      -- exposure top bit ($0014 bit7): the base was reached un-occluded
      probes    -- per-probe reached-the-target booleans (upper first for robots,
                   then base)

    The horizontal/vertical bearings come from the bit-exact geometry above; the
    terrain ray-march is :func:`sentinel.los.check_for_line_of_sight_to_tile`."""
    out = {
        "in_slot": False,
        "in_fov": False,
        "full": False,
        "exposure": 0,
        "tree_in_los_head": False,
        "probes": [],
        "cycles": passcost.SEE_SLOT_EMPTY,
        "wrong_type": False,
    }
    # $188F LDA #0 ; $1891 STA $0014 -- object_exposure is cleared before any early
    # exit, so even a rejected slot leaves $0014 == 0 (the robot scan then skips it).
    state.mem[0x0014] = 0
    if state.obj_flags[target] & 0x80:  # empty slot
        return out
    if state.obj_type[target] != expected_type:  # wrong type
        out["wrong_type"] = True
        out["cycles"] = passcost.SEE_SLOT_WRONG_TYPE
        return out
    out["in_slot"] = True

    ra = relative_angles(state, observer, target)
    zp = ra["_zp"]
    cyc = passcost.SEE_PROLOGUE + ra["cycles"] + passcost.SEE_FOV
    # FOV gate ($18B8): A = c57 - $0A + (fov>>1) ; out of view if A >= fov.
    a = (ra["c57"] - 0x0A + (fov_width >> 1)) & 0xFF
    if a >= fov_width:
        out["cycles"] = cyc + passcost.SEE_FOV_REJECT
        return out
    out["in_fov"] = True
    cyc += passcost.SEE_FOV_PASS

    h_lo, h_hi = ra["angle_lo"], ra["angle_hi"]  # horizontal bearing ($003D/$003E)
    z_lo, z_hi = ra["z_lo"], ra["z_hi"]  # relative z ($0081/$0084)
    v_angle_obs = state.obj_v_angle[observer]
    state.mem[0x0C58] = target  # the ray-march recognises the targeted object

    # A robot is probed at its upper point first ($18DC is_robot) then its base
    # ($1904 not_robot); every other object only at its base.  The upper-point
    # probe runs with enemy_is_considering_robot ($0C6E bit7) SET, which waives the
    # "can't look up" rejection ($1D26) so an enemy can see a robot's head above its
    # own eye; the base probe runs with the flag clear.
    probes = []  # (rel_z_lo, rel_z_hi, do_los_checks)
    if expected_type == mm.T_ROBOT:  # robots probe the upper point first
        probes.append((z_lo, z_hi, 0x80))
        cyc += passcost.SEE_ROBOT
    else:
        cyc += passcost.SEE_NOT_ROBOT
    base_lo = (z_lo - 0xE0) & 0xFF
    base_hi = (z_hi - (1 if z_lo < 0xE0 else 0)) & 0xFF
    probes.append((base_lo, base_hi, 0x00))
    # $1904..$1914 runs twice whatever the type: the $1E counter always reaches 0.
    cyc += 2 * passcost.SEE_BASE + passcost.SEE_BASE_AGAIN + passcost.SEE_BASE_LAST

    # object_exposure ($0014) was cleared above; each probe rotates one bit in via the
    # $18F9-$1901 four-rotate chained-carry cascade:
    #   $18F9 ROL $0C56   carry_in = no-LOS ;          carry_out = old $0C56 bit7
    #   $18FC ROR $0014   carry_in = old $0C56 bit7 ;  carry_out = old $0014 bit0
    #   $18FE ROL $0CDD   carry_in = old $0014 bit0 ;  carry_out = old $0CDD bit7
    #   $1901 ROR $0C76   carry_in = old $0CDD bit7 ;  carry_out discarded
    # The march writes $0C56 bit7 when the ray reaches the targeted object's own tile
    # un-occluded ($1E13, gated on $0C58) and $0CDD bit7 when a non-target tree lay in
    # the sightline ($1E96 is_tree).  So the target's tree-in-sightline flag rotates
    # through $0CDD into $0C76: over the two probes (head first, base second) $0C76
    # ends with bit7 = base-probe tree flag, bit6 = HEAD-probe tree flag -- the byte
    # find_drainable_robot_loop reads at $17B7 to skip a robot with a tree in front of
    # its head.
    for plo, phi, do_los in probes:
        zp[0x80] = plo
        # $933D sets zp[$8A]/zp[$8B] to the vertical bearing.
        _va, vcyc = _vertical_angle(zp, phi, v_angle_obs)
        cyc += passcost.SEE_PROBE_FIXED + vcyc + passcost.MARCH_ENTRY
        v_lo, v_hi = zp[0x8A], zp[0x8B]
        vec = los.prepare_vector_from_angle(h_hi, h_lo, v_hi, v_lo, v_lo)
        cyc += vec.cycles  # $1C54, priced from its own shift-adds
        _tx, _ty, los_ok, march_cycles = los.check_for_line_of_sight_to_tile(
            vec, state, observer, do_los_checks=do_los, max_steps=max_steps
        )
        cyc += march_cycles
        # The march writes the marched-out $0C56/$0CDD trackers back to memory, so read
        # them here exactly as the cascade does.
        c56 = state.mem[0x0C56]  # $18F9 ROL $0C56
        reached = (c56 >> 7) & 1  # carry out == old $0C56 bit7
        state.mem[0x0C56] = ((c56 << 1) | (0 if los_ok else 1)) & 0xFF
        c14 = state.mem[0x0014]  # $18FC ROR $0014
        c14_out = c14 & 1  # carry out == old $0014 bit0
        state.mem[0x0014] = ((reached << 7) | (c14 >> 1)) & 0xFF
        cdd = state.mem[0x0CDD]  # $18FE ROL $0CDD
        tree_flag = (cdd >> 7) & 1  # carry out == old $0CDD bit7 (tree in sightline)
        state.mem[0x0CDD] = ((cdd << 1) | c14_out) & 0xFF
        c76 = state.mem[0x0C76]  # $1901 ROR $0C76
        state.mem[0x0C76] = ((tree_flag << 7) | (c76 >> 1)) & 0xFF
        out["probes"].append(bool(reached))
    exposure = state.mem[0x0014]
    out["exposure"] = exposure
    # $0014 top bit == fully visible (base reached for a robot, the sole probe else).
    out["full"] = bool(exposure & 0x80)
    # $0C76 bit6 == a non-target tree lay in the enemy's sightline to the (robot) head.
    out["tree_in_los_head"] = bool(state.mem[0x0C76] & 0x40)
    out["cycles"] = cyc + passcost.SEE_TAIL
    return out
