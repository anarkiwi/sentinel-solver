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

from sentinel import memmap as mm, los

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
    (angle_lo=$008A, angle_hi=$008B, ratio=$007E) with angle = arctan(a/b) as a
    16-bit value (full circle = $10000)."""
    t74 = a_lo & 0xFF  # a low / quotient accumulator
    A = a_hi & 0xFF  # a high / working remainder
    t76 = b_hi & 0xFF
    t77 = b_lo & 0xFF
    t78 = 0  # result_low (rounds 9/10 bits)

    def full_over(cA):
        # rounds 1-3: a >= b as a 16-bit compare (A:$74 vs $76:$77)
        if cA:
            return True
        if A > t76:
            return True
        if A == t76 and t74 >= t77:
            return True
        return False

    # --- round 1 (ASL $74) ---
    t74, c = _asl(t74)
    A, cA = _rol(A, c)
    if full_over(cA):
        v = t74 - t77
        t74 = v & 0xFF
        A = (A - t76 - (1 if v < 0 else 0)) & 0xFF
        c = 1
    else:
        c = 0
    # --- rounds 2, 3 (ROL $74), full 16-bit ---
    php_carry = 0
    for rnd in (2, 3):
        t74, c0 = _rol(t74, c)
        A, cA = _rol(A, c0)
        if full_over(cA):
            v = t74 - t77
            t74 = v & 0xFF
            A = (A - t76 - (1 if v < 0 else 0)) & 0xFF
            c = 1
        else:
            c = 0
        if rnd == 3:
            php_carry = c  # PHP after round 3

    # --- skip_further_division ($0E10): remainder == b_hi after round 3 ---
    if A == t76:
        t78 = 0  # LDA #$0 ; STA $0078
        A = 0  # LDA #$0 -> the RORs work on A=0, not the remainder
        cc = 1  # CMP equal -> carry set
        t74, cc = _ror(t74, cc)
        A, cc = _ror(A, cc)
        t74, cc = _ror(t74, cc)
        A, cc = _ror(A, cc)
        A = A | 0x20
        return _finish_overflow(A, php_carry, t78)

    # --- round 4 (ASL $74), 8-bit compare ---
    t74, c0 = _asl(t74)
    A, cA = _rol(A, c0)
    if cA or A >= t76:
        A = (A - t76) & 0xFF
        c = 1
    else:
        c = 0
    # --- rounds 5..9 (ROL $74), 8-bit compare ---
    for _rnd in range(5, 10):
        t74, c0 = _rol(t74, c)
        A, cA = _rol(A, c0)
        if cA or A >= t76:
            A = (A - t76) & 0xFF
            c = 1
        else:
            c = 0
    # --- round 10 ($0DF1): ROR $78 (round-9 bit), ROL A, compare, ROR $78 ---
    t78, c0 = _ror(t78, c)  # ROR $0078: shift in round-9 quotient bit
    A, cA = _rol(A, c0)  # ROL A
    if cA:
        c = 1
    else:
        c = 1 if A >= t76 else 0  # CMP $0076
    t78, _ = _ror(t78, c)  # ROR $0078: shift in round-10 bit

    return _finish_overflow(t74, php_carry, t78)


def _finish_overflow(A, php_carry, t78):
    """consider_overflow ($0DFC): apply the round-3 bit ($20), detect the 45-degree
    overflow, then the arctan lookup + rounds-9/10 interpolation."""
    if php_carry:  # round 3 was over
        s = A + 0x1F + 1
        A = s & 0xFF
        carry = 1 if s > 0xFF else 0
    else:
        carry = 0  # BCC round_3_under with carry from prior op == 0
    if carry:  # overflow: clamp to 45 degrees
        return 0x00, 0x20, 0xFF

    # --- no_overflow ($0E1F): arctan lookup + rounds 9/10 interpolation ---
    Y = A & 0xFF
    ratio = Y
    ang_lo = _ARCTAN_LO[Y]
    ang_hi = _ARCTAN_HI[Y]
    b78_7 = (t78 >> 7) & 1  # round 10 over
    b78_6 = (t78 >> 6) & 1  # round 9 over
    if not (b78_7 or b78_6):
        return ang_lo, ang_hi, ratio  # BMI/BVS both false -> leave

    # calculate_delta ($0E35): delta = arctan[Y] - arctan[Y+1]
    d_lo = (ang_lo - _ARCTAN_LO[Y + 1]) & 0xFF
    borrow = 1 if ang_lo < _ARCTAN_LO[Y + 1] else 0
    d_hi = (ang_hi - _ARCTAN_HI[Y + 1] - borrow) & 0xFF
    if b78_6:  # round 9 over ($0E44 BVC skip; BVS -> invert)
        d_hi, d_lo = _invert16(d_hi, d_lo)
    # STA $0075 ; ROL A ; ROR $0075 ; ROR $0074  -> arithmetic >>1 keeping sign.
    # ROL A sets carry = bit7(d_hi) (the sign); ROR $0075 shifts it back in.
    t75, c2 = _ror(d_hi, (d_hi >> 7) & 1)
    t74b, _ = _ror(d_lo, c2)
    d_hi2, d_lo2 = t75, t74b
    # angle += arctan[Y+1]  ($0E50)
    s = ang_lo + _ARCTAN_LO[Y + 1]
    ang_lo = s & 0xFF
    cc = 1 if s > 0xFF else 0
    ang_hi = (ang_hi + _ARCTAN_HI[Y + 1] + cc) & 0xFF
    if b78_7:  # round 10 over ($0E61 BPL skip): add half-delta
        s = ang_lo + d_lo2
        ang_lo = s & 0xFF
        cc = 1 if s > 0xFF else 0
        ang_hi = (ang_hi + d_hi2 + cc) & 0xFF
    # LSR $008B ; ROR $008A  -> average
    ang_hi, c3 = _ror(ang_hi, 0)
    ang_lo, _ = _ror(ang_lo, c3)
    return ang_lo, ang_hi, ratio


# ---------------------------------------------------------------------------
# calculate_angle $9287 (quadrant-folded arctan of x, y)
# ---------------------------------------------------------------------------
def _calc_angle(zp):
    """$9287: horizontal angle of the vector (x, y). x = (zp[$80], zp[$83]),
    y = (zp[$82], zp[$85]), with signs zp[$86] (x) and zp[$88] (y). Writes the
    16-bit angle to zp[$8A]/zp[$8B], the divide ratio to zp[$7E], and the raw
    min/max magnitudes to zp[$5C..$5D]/zp[$7A..$7B] for calculate_hypotenuse."""
    x_lo, x_hi = zp[0x80], zp[0x83]
    y_lo, y_hi = zp[0x82], zp[0x85]
    sx, sy = zp[0x86], zp[0x88]

    x_larger = (y_hi < x_hi) or (y_hi == x_hi and y_lo < x_lo)

    if x_larger:
        zp[0x5D], zp[0x5C] = y_hi, y_lo  # min = y
        zp[0x7A], zp[0x7B] = x_lo, x_hi  # max = x
        b_lo, b_hi, a_lo, a_hi = _normalise(zp, 0x80, 0x83, 0x82, 0x85)
        ang_lo, ang_hi, ratio = _divide_and_arctan(a_lo, a_hi, b_lo, b_hi)
        if not ((sx ^ sy) & 0x80):  # same sign -> negate angle
            ang_hi, ang_lo = _invert16(ang_hi, ang_lo)
        base = 0x40 if not (sx & 0x80) else 0xC0  # +90 or +270 degrees
        ang_hi = (ang_hi + base) & 0xFF
    else:
        if (y_hi | y_lo) == 0:  # both zero
            zp[0x7E] = zp[0x8A] = zp[0x8B] = 0
            return
        zp[0x5D], zp[0x5C] = x_hi, x_lo  # min = x
        zp[0x7A], zp[0x7B] = y_lo, y_hi  # max = y
        b_lo, b_hi, a_lo, a_hi = _normalise(zp, 0x82, 0x85, 0x80, 0x83)
        ang_lo, ang_hi, ratio = _divide_and_arctan(a_lo, a_hi, b_lo, b_hi)
        if (sx ^ sy) & 0x80:  # opposite sign -> negate angle
            ang_hi, ang_lo = _invert16(ang_hi, ang_lo)
        base = 0x00 if not (sy & 0x80) else 0x80  # +0 or +180 degrees
        ang_hi = (ang_hi + base) & 0xFF

    zp[0x8A], zp[0x8B], zp[0x7E] = ang_lo, ang_hi, ratio


def _normalise(zp, max_lo_a, max_hi_a, min_lo_a, min_hi_a):
    """The scale_using_x/_y normalisation ($92C1/$92FF): shift the larger vector
    (max) left until it overflows, shifting the smaller (min) in lock-step, then
    back the max off by one. Returns the divide inputs (b_lo, b_hi, a_lo, a_hi)
    where b = normalised max and a = normalised min. Mutates only local copies."""
    max_lo, max_hi = zp[max_lo_a], zp[max_hi_a]
    min_lo, min_hi = zp[min_lo_a], zp[min_hi_a]
    A = max_hi
    while True:
        max_lo, c = _asl(max_lo)
        A, cA = _rol(A, c)
        if cA:  # max overflowed
            break
        min_lo, c2 = _asl(min_lo)
        min_hi, _ = _rol(min_hi, c2)
    # ROR A ; ROR max_lo  (back off the overflow shift)
    A, c = _ror(A, 1)  # carry in = 1 (the overflow bit)
    max_lo, _ = _ror(max_lo, c)
    b_hi = A
    a_lo = min_lo
    b_lo = max_lo & 0xFC  # AND #$fc: drop scaling noise
    a_hi = min_hi
    return b_lo, b_hi, a_lo, a_hi


# ---------------------------------------------------------------------------
# calculate_hypotenuse $937F
# ---------------------------------------------------------------------------
def _calc_hypotenuse(zp):
    """$937F: distance = max + f*min/512, with f from the $3D02 coefficient table
    indexed by round(ratio/2). Reads zp[$7E] (ratio), zp[$5C..$5D] (min),
    zp[$7A..$7B] (max); writes zp[$7C]/zp[$7D]."""
    ratio = zp[0x7E]
    y = (ratio >> 1) + (ratio & 1)  # LSR A ; ADC #$0  -> round(ratio/2)
    f = _HYP[y]
    res_lo, res_hi = los.multiply_double_by_byte(zp[0x5C], f, zp[0x5D])
    # LSR $0075 ; ROR $0074  -> halve the 16-bit result
    new_hi = res_hi >> 1
    new_lo = ((res_hi & 1) << 7) | (res_lo >> 1)
    # + max
    s = new_lo + zp[0x7A]
    zp[0x7C] = s & 0xFF
    cc = 1 if s > 0xFF else 0
    zp[0x7D] = (new_hi + zp[0x7B] + cc) & 0xFF


# ---------------------------------------------------------------------------
# calculate_object_relative_vertical_angle $933D
# ---------------------------------------------------------------------------
def _vertical_angle(zp, z_hi, v_angle):
    """$933D: the target's vertical angle. z_hi is the (signed) relative-z high
    byte to test; zp[$80] holds the relative-z low byte; zp[$7C]/zp[$7D] hold the
    horizontal distance; v_angle is the observer's objects_v_angle. Returns the
    signed vertical angle byte (zp[$8D])."""
    sx = z_hi & 0xFF
    if sx & 0x80:  # negative -> make positive
        zlo = (-zp[0x80]) & 0xFF
        borrow = 1 if zp[0x80] != 0 else 0
        sx_abs = (-sx - borrow) & 0xFF
        zp[0x80] = zlo
    else:
        sx_abs = sx
    zp[0x83] = sx_abs  # x_hi = |rel z hi|
    zp[0x82] = zp[0x7C]  # y_lo = hyp_lo
    zp[0x85] = zp[0x7D]  # y_hi = hyp_hi
    zp[0x88] = 0  # signed_y = 0
    zp[0x86] = sx  # signed_x = original rel z hi
    _calc_angle(zp)
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
    return A & 0xFF


# ---------------------------------------------------------------------------
# relative x/y/z ($85C4 / $85F5) and the $8401 glue
# ---------------------------------------------------------------------------
def _relative_xyz(zp, obj, obs_x, obs_y, obs_zf, obs_zh, tgt_x, tgt_y, tgt_zf, tgt_zh):
    """calculate_object_relative_x_and_y ($85C4) + _z ($85F5): the signed and
    absolute component distances of a target from an observer."""
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
    _relative_xyz(
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
    _calc_angle(zp)
    # $0C57 = angle_hi - h_angle[observer] + $0A ; $0C59 = angle_lo - $001F(=0)
    h_angle = state.obj_h_angle[observer]
    c59 = zp[0x8A]  # SBC $001F(0), carry stays set -> no borrow into the high byte
    c57 = (zp[0x8B] - h_angle + 0x0A) & 0xFF
    _calc_hypotenuse(zp)
    return {
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
        "probes": [],
    }
    if state.obj_flags[target] & 0x80:  # empty slot
        return out
    if state.obj_type[target] != expected_type:  # wrong type
        return out
    out["in_slot"] = True

    ra = relative_angles(state, observer, target)
    zp = ra["_zp"]
    # FOV gate ($18B8): A = c57 - $0A + (fov>>1) ; out of view if A >= fov.
    a = (ra["c57"] - 0x0A + (fov_width >> 1)) & 0xFF
    if a >= fov_width:
        return out
    out["in_fov"] = True

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
    base_lo = (z_lo - 0xE0) & 0xFF
    base_hi = (z_hi - (1 if z_lo < 0xE0 else 0)) & 0xFF
    probes.append((base_lo, base_hi, 0x00))

    # object_exposure ($0014) starts at 0 ($188A) and each probe shifts one bit in.
    exposure = 0
    for plo, phi, do_los in probes:
        zp[0x80] = plo
        _vertical_angle(zp, phi, v_angle_obs)  # sets zp[$8A]/zp[$8B] = vertical bearing
        v_lo, v_hi = zp[0x8A], zp[0x8B]
        vec = los.prepare_vector_from_angle(h_hi, h_lo, v_hi, v_lo, v_lo)
        _tx, _ty, los_ok = los.check_for_line_of_sight_to_tile(
            vec, state, observer, do_los_checks=do_los, max_steps=max_steps
        )
        # $18F9 ROL $0C56 (carry in = no-LOS) then $18FC ROR $0014 (carry out = the
        # OLD $0C56 bit7).  The march sets $0C56 bit7 when it steps onto the targeted
        # object's own tile un-occluded ($1E13, gated on $0C58) -- so "reached", not
        # "the ray ended on the tile", is what marks the object visible.  A ray that
        # clears intervening terrain and passes over the target's head still reaches
        # its tile, which is how an enemy sees a robot standing above its eye.
        c56 = state.mem[0x0C56]
        reached = (c56 >> 7) & 1
        state.mem[0x0C56] = ((c56 << 1) | (0 if los_ok else 1)) & 0xFF
        exposure = ((exposure >> 1) | (reached << 7)) & 0xFF
        out["probes"].append(bool(reached))
    state.mem[0x0014] = exposure
    out["exposure"] = exposure
    # $0014 top bit == fully visible (base reached for a robot, the sole probe else).
    out["full"] = bool(exposure & 0x80)
    return out
