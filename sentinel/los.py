"""Bit-exact transcription of the ROM's player aim and line-of-sight ray march.

Answers "can the observer at this tile, aimed this way, see that tile?" in integer math with
no emulator.  The aim uses the sin_cos_lookup $0E75 polynomial, not the $AC80 table.
:func:`landable_sweep_with_centres` is the keyboard-aim buildability oracle.
"""

import numpy as np

from sentinel import passcost
from sentinel.writeweight import MARCH as wcost
from sentinel.terrain import tile_byte

from sentinel import los_jit

_HAVE_JIT = True  # tests set this False to drive the pure-Python reference bodies


# math core (multiply / invert) -- $0D03-$1016
def _bits(v):
    """Set bits of a $0D03 multiplier: each costs MUL8_BIT more than a clear one."""
    return bin(v & 0xFF).count("1")


def _mul8(a, b):
    """multiply_byte_by_byte $0D03/$0D05: the 8x8->16 product as (a*b >> 8, a*b & 0xFF)."""
    p = (a & 0xFF) * (b & 0xFF)
    return (p >> 8) & 0xFF, p & 0xFF


def invert16(high, frac):
    """invert_A_and_a_fraction $1009: negate the 16-bit (high:frac)."""
    val = ((high & 0xFF) << 8) | (frac & 0xFF)
    neg = (-val) & 0xFFFF
    return (neg >> 8) & 0xFF, neg & 0xFF


def multiply_double_by_byte(low74, high75, byte76):
    """multiply_double_by_byte $0F4A: the 16-bit ($0076:$0074) times $0075.

    The shared tail of multiply_double_A_by_pi: r1 = $0074*$0075, r2 = $0076*$0075, then
    $0074 = r1h + r2l and $0075 = r2h plus the carry.  Returns (low, high, cycles).
    """
    cyc = passcost.MUL_DBL_BYTE + passcost.MUL8_BIT * (_bits(low74) + _bits(byte76))
    r1h, _r1l = _mul8(low74, high75)
    r2h, r2l = _mul8(byte76, high75)
    res75 = r2h
    total = r1h + r2l
    res74 = total & 0xFF
    if total > 0xFF:
        res75 = (res75 + 1) & 0xFF
        cyc += passcost.MUL_DBL_BYTE_CARRY
    return res74, res75, cyc


def multiply_double_A_by_pi(A, frac74):
    """$0F3E: multiply (A.frac74) by ~PI (constant $C9 used as multiplicand, the
    value being *4 first). Returns (result_low, result_high)."""
    frac = frac74 & 0xFF
    a = A & 0xFF
    # ASL $0074/ROL A twice -> *4
    for _ in range(2):
        c = (frac >> 7) & 1
        frac = (frac << 1) & 0xFF
        a = ((a << 1) | c) & 0xFF
    m76 = a  # STA $0076
    # LDA #$c9 ; STA $0075 ; JSR multiply_byte_by_byte (entry $0D05).
    cyc = (
        passcost.MUL_PI
        + passcost.MUL_DBL_BYTE
        + passcost.MUL8_BIT * (_bits(frac) + _bits(m76))
    )
    r1h, _r1l = _mul8(frac, 0xC9)
    r77 = r1h  # STA $0077 (a_fraction discarded)
    # LDA $0076 ; JSR $0D03 ($0F51): STA $0074(=m76) ; mult m76*$0075($c9)
    r2h, _r2l = _mul8(m76, 0xC9)
    r75 = r2h  # STA $0075
    # LDA $0077 ; CLC ADC $0074(=_r2l) -> $0074 ; BCC/INC $0075
    total = r77 + _r2l
    r74 = total & 0xFF
    if total > 0xFF:
        r75 = (r75 + 1) & 0xFF
        cyc += passcost.MUL_DBL_BYTE_CARRY
    return r74, r75, cyc  # (result_low, result_high, cycles)


def sin_cos_lookup(angle, frac74=0):
    """$0E75: sine & cosine of `angle` (byte; full circle=256), with incoming
    a_fraction $0074=frac74. Returns (sin_lo, cos_lo, sin_hi, cos_hi) -- the four
    bytes at &0C00..&0C03; bit0 of the low byte is the sign flag (set => negate)."""
    angle &= 0xFF
    c0c = angle  # $0C0C
    aPI_lo, aPI_hi, cyc = multiply_double_A_by_pi(angle, frac74)
    cyc += passcost.SIN_COS
    c53 = aPI_lo  # $0C53
    c54 = aPI_hi  # $0C54
    # X=1; $0060=1; X=0; BIT $0C0C: V(bit6) set => INX, DEC $0060
    sixty = 1
    X = 0
    if c0c & 0x40:
        X = 1
        sixty = 0
        cyc += passcost.SIN_COS_QUAD
    else:
        cyc += passcost.SIN_COS_NOQUAD
    # the loop-top CMP #$7a tests A: c54 ($0E81 STA $0C54) on pass 1, recomputed on pass 2
    A_cmp = c54
    cur_c53 = c53
    cur_c54 = c54
    cur_75 = aPI_hi
    sc_low = [0, 0]
    sc_high = [0, 0]

    while True:
        if (A_cmp & 0xFF) >= 0x7A:
            cyc += passcost.SIN_COS_HIGH + passcost.SIN_COS_TWO
            cyc += passcost.MUL_DBL_BYTE_CALL
            # method two $0ECB (A>=$7a): sin a =~ 1 - 2*(PI/4 - aPI)^2 ; 0 - $0C53 -> $0074
            t74 = (0 - cur_c53) & 0xFF
            borrow1 = 1 if (0 - cur_c53) < 0 else 0
            # LDA #$c9; SBC $0C54 -> $0075,$0076 ; (PI/4 - aPI)
            v = 0xC9 - cur_c54 - borrow1
            t75 = v & 0xFF
            t76 = t75
            # multiply_double_by_byte: ($0074:$0076)*$0075 -> (PI/4-aPI)^2
            r74, r75, mcyc = multiply_double_by_byte(t74, t75, t76)
            cyc += mcyc
            # ASL $0074 ; ROL $0075 (double)
            c = (r74 >> 7) & 1
            r74 = (r74 << 1) & 0xFF
            r75 = ((r75 << 1) | c) & 0xFF
            # A=#0; SEC; SBC $0074 ; AND #$fe -> low
            sub_lo = 0 - r74
            low = (sub_lo & 0xFF) & 0xFE
            borrow2 = 1 if sub_lo < 0 else 0
            sc_low[X] = low
            # A=#0; SBC $0075: BCC (borrow) stores the negative result, carry set clamps
            sub_hi = 0 - r75 - borrow2
            if sub_hi < 0:
                sc_high[X] = sub_hi & 0xFF  # skip_overflow: store result_high
            else:
                # carry set: clamp to &fffe (lowest bit = sign)
                sc_low[X] = 0xFE
                sc_high[X] = 0xFF
                cyc += passcost.SIN_COS_TWO_CLAMP
        else:
            # method one $0EA1 (A<$7a)
            r_high1, _ = _mul8(0xAB, cur_75)
            r_high2, r_low2 = _mul8(r_high1, cur_75)
            t76 = r_high2  # STA $0076
            cyc += (
                passcost.SIN_COS_LOW
                + passcost.SIN_COS_ONE
                + passcost.MUL_DBL_BYTE_CALL
                + 2 * passcost.MUL8
                + passcost.MUL8_BIT * (_bits(0xAB) + _bits(r_high1))
            )
            r74, r75, mcyc = multiply_double_by_byte(r_low2, cur_75, t76)
            cyc += mcyc
            # LDA $0C53; SEC; SBC $0074 -> $0074
            t74b = (cur_c53 - r74) & 0xFF
            borrow1 = 1 if cur_c53 < r74 else 0
            # LDA $0C54; SBC $0075 ; ASL $0074 ; ROL A -> high
            hv = (cur_c54 - r75 - borrow1) & 0xFF
            c = (t74b >> 7) & 1
            t74b2 = (t74b << 1) & 0xFF
            hi = ((hv << 1) | c) & 0xFF
            sc_high[X] = hi
            sc_low[X] = t74b2 & 0xFE

        # calculate_next_value $0EFD
        if X == sixty:
            cyc += passcost.SIN_COS_DONE
            break
        cyc += passcost.SIN_COS_NEXT
        X = sixty
        # A=#0; SEC; SBC $0C53 -> $0C53 ; LDA #$c9; SBC $0C54 -> $0C54 ; $0075=$0C54
        new_c53 = (0 - cur_c53) & 0xFF
        borrow = 1 if (0 - cur_c53) < 0 else 0
        new_c54 = (0xC9 - cur_c54 - borrow) & 0xFF
        cur_c53 = new_c53
        cur_c54 = new_c54
        cur_75 = new_c54
        A_cmp = new_c54  # JMP loop; A = $0075 = new_c54

    # set_signs $0F19
    sin_lo, cos_lo = sc_low[0], sc_low[1]
    sin_hi, cos_hi = sc_high[0], sc_high[1]
    # slot 0 is always sine, slot 1 cosine: pass 1 writes sc[X], pass 2 sc[sixty]
    cyc += passcost.SIN_COS_SIGNS
    if c0c & 0x80:  # $0F19: $0C0C bit7 set (sin negative) -> ORA #1 into sin_lo
        sin_lo |= 1
        cyc += passcost.SIN_COS_SIGN
    # cos sign: A=$0C0C; ASL A; EOR $0C0C; if result bit7 set -> cos negative
    t = ((c0c << 1) & 0xFF) ^ c0c
    if t & 0x80:
        cos_lo |= 1
        cyc += passcost.SIN_COS_SIGN
    return sin_lo & 0xFF, cos_lo & 0xFF, sin_hi & 0xFF, cos_hi & 0xFF, cyc


def process_sine_or_cosine(low, high):
    """process_sine_or_cosine $1C9D: take (low,high) of sin or cos (low bit0=sign),
    divide by 16, invert if sign bit set. Returns (A=result_high, X=result_low)."""
    t74 = low & 0xFF  # STA $0074
    A = high & 0xFF

    # LSR A ; ROR $0074 ; PHP saves the sign (bit0 of low) ; 3 more divide by 16 ; PLP
    def lsr_a_ror74(A, t74):
        c = A & 1
        A >>= 1
        new_c = t74 & 1
        t74 = (t74 >> 1) | (c << 7)
        return A, t74, new_c

    A, t74, carry1 = lsr_a_ror74(A, t74)  # 0F.. first
    saved_carry = carry1  # PHP
    A, t74, _ = lsr_a_ror74(A, t74)
    A, t74, _ = lsr_a_ror74(A, t74)
    A, t74, _ = lsr_a_ror74(A, t74)  # divided by 16
    cyc = passcost.PROC_SC
    if saved_carry:  # PLP; BCC skip_inversion
        A, t74 = invert16(A, t74)
        cyc += passcost.PROC_SC_INVERT
    X = t74 & 0xFF  # LDX $0074
    return A & 0xFF, X, cyc


def multiply_double_by_double(x_lo, x_hi, y_lo, y_hi):
    """multiply_double_by_double $0F9E: signed 16x16 fixed-point multiply used by
    set_vector. Inputs: $0068=x_lo,$0069=x_hi (the sin/cos), $006A=y_lo,$006B=y_hi
    (cos(v)). $0067 starts 0 (invert flag). Returns (A=result_high, frac=result_low)
    after the invert_A_and_a_fraction_if_negative at the end ($1005 BIT $0067)."""
    s67 = 0
    s6a = y_lo & 0xFF
    s6b = y_hi & 0xFF
    s68 = x_lo & 0xFF
    s69 = x_hi & 0xFF
    cyc = passcost.MUL_DBL_DBL + passcost.MUL8_BIT * (_bits(s68) + 2 * _bits(s69))
    # if $006B negative: negate y (16-bit), flip $0067 bit7
    if s6b & 0x80:
        neg = (-(((s6b << 8) | s6a))) & 0xFFFF
        s6a = neg & 0xFF
        s6b = (neg >> 8) & 0xFF
        s67 ^= 0x80
        cyc += passcost.MUL_DBL_DBL_NEG_Y
    # if $0068 bit0 set: flip sign
    if s68 & 1:
        s67 ^= 0x80
        cyc += passcost.MUL_DBL_DBL_ODD
    # $0075 = $006B ; A=$0068 ; multiply_byte_by_byte ($0D03): $0074=$0068 * $0075
    r1h, xlow_yhigh_low = _mul8(s68, s6b)  # x_low * y_high (high, low)
    r77 = r1h  # STA $0077
    # $0074 = result_low of that mult; LDA $0074 ; CLC ADC #$80 (round) -> $0076
    rounded = xlow_yhigh_low + 0x80
    r76 = rounded & 0xFF
    if rounded > 0xFF:
        r77 = (r77 + 1) & 0xFF
        cyc += passcost.MUL_DBL_DBL_CARRY
    # $0075=$006B ; LDA $0069 ; multiply_byte_by_byte: $0069 * $006B -> $0078=high
    r2h, r2l = _mul8(s69, s6b)  # x_high * y_high
    r78 = r2h
    # LDA $0074(=r2l) ; CLC ADC $0077 -> $0077 ; BCC/INC $0078
    total = r2l + r77
    r77 = total & 0xFF
    if total > 0xFF:
        r78 = (r78 + 1) & 0xFF
        cyc += passcost.MUL_DBL_DBL_CARRY
    # $0075=$006A ; LDA $0069 ; multiply_byte_by_byte: $0069 * $006A -> high,low
    r3h, r3l = _mul8(s69, s6a)  # x_high * y_low
    # LDA $0074(=r3l) ; CLC ADC $0076 ; LDA $0075(=r3h) ; ADC $0077 -> $0074 ; BCC/INC $0078
    t = r3l + r76
    carry = 1 if t > 0xFF else 0
    t2 = r3h + r77 + carry
    res74 = t2 & 0xFF
    if t2 > 0xFF:
        r78 = (r78 + 1) & 0xFF
        cyc += passcost.MUL_DBL_DBL_CARRY
    # LDA $0078 ; BIT $0067 ; invert_if_negative
    A = r78 & 0xFF
    frac = res74 & 0xFF
    if s67 & 0x80:
        A, frac = invert16(A, frac)
        cyc += passcost.MUL_DBL_DBL_NEG
    return A & 0xFF, frac & 0xFF, cyc


# vector construction -- prepare_vector_from_player_sights / _from_angle
class Vector:
    """The marching ray state: vector components (signed 16-bit, lo/hi) and the
    position triples (frac/sub/whole). Mirrors zero-page $2C-$3C."""

    __slots__ = (
        "vx_lo",
        "vx_hi",
        "vz_lo",
        "vz_hi",
        "vy_lo",
        "vy_hi",
        "s30",
        "s33",
        "s32",
        "cycles",
        "px_frac",
        "px_sub",
        "px_whole",
        "pz_frac",
        "pz_sub",
        "pz_whole",
        "py_frac",
        "py_sub",
        "py_whole",
    )


def prepare_vector_from_angle(h_angle, h_frac, v_angle, v_frac, s74):
    """$1C54: compute the unit direction vector from horizontal/vertical angles.
    h_frac=$003D, h_angle=$003E, v_frac/s74 used as a_fraction for v's sin_cos.
    Returns a Vector."""
    vec = Vector()
    # sin_cos_lookup of v_angle, a_fraction $0074 = v_frac ($1C48 STA $0074 on the player path)
    sin_lo_v, cos_lo_v, sin_hi_v, cos_hi_v, cyc = sin_cos_lookup(v_angle, v_frac)
    cyc += passcost.PREP_VEC
    # process cosine (Y=1) -> $0033=A, $0032=X
    A_cos, X_cos, pcyc = process_sine_or_cosine(cos_lo_v, cos_hi_v)
    cyc += pcyc
    s33 = A_cos
    s32 = X_cos
    # process sine (Y=0) -> $0030=A, $002D=X
    A_sin, X_sin, pcyc = process_sine_or_cosine(sin_lo_v, sin_hi_v)
    cyc += pcyc
    s30 = A_sin
    s2d = X_sin
    # $1C69 LDA $003D -> $0074 ; LDA $003E -> sin_cos_lookup (horizontal)
    h_sin_lo, h_cos_lo, h_sin_hi, h_cos_hi, hcyc = sin_cos_lookup(h_angle, h_frac)
    cyc += hcyc
    # set_vector: cos(v) [$0032/$0033] times cos(h) (Y=1) -> vector_y ($2E/$31) [X=2]
    vy_hi, vy_lo, mcyc = multiply_double_by_double(h_cos_lo, h_cos_hi, s32, s33)
    cyc += mcyc
    # Y=0 -> uses sine(h); store to vector_x ($2C/$2F) [X=0]
    vx_hi, vx_lo, mcyc = multiply_double_by_double(h_sin_lo, h_sin_hi, s32, s33)
    cyc += mcyc
    # set_vector stores $2F+X=high, $2C+X=low; vector_z ($2D/$30) is the v-sine directly
    vec.vx_lo = vx_lo
    vec.vx_hi = vx_hi
    vec.vz_lo = s2d
    vec.vz_hi = s30
    vec.vy_lo = vy_lo
    vec.vy_hi = vy_hi
    vec.s30 = s30
    vec.s33 = s33
    vec.s32 = s32
    vec.cycles = cyc
    return vec


# line-of-sight march -- get_object_details + add_vector + check_for_los_to_tile
def _get_object_details(vec, state, slot, eye_z=None):
    """$1ECC: seed the ray position from observer object `slot`."""
    vec.px_frac = 0
    vec.pz_frac = 0
    vec.py_frac = 0
    vec.px_sub = 0x80  # $0037
    vec.py_sub = 0x80  # $0039
    vec.pz_sub = state.obj_z_frac[slot]  # $0038
    vec.px_whole = state.obj_x[slot]  # $003A
    z = state.obj_z_height[slot] if eye_z is None else (eye_z & 0xFF)
    vec.pz_whole = z  # $003B
    vec.py_whole = state.obj_y[slot]  # $003C


def _add_vector(vec):
    """$1CBB: 24-bit signed step of the ray for x, z, y."""
    for axis in (2, 1, 0):  # X=2(y),1(z),0(x)
        if axis == 0:
            frac, sub, whole = vec.px_frac, vec.px_sub, vec.px_whole
            vlo, vhi = vec.vx_lo, vec.vx_hi
        elif axis == 1:
            frac, sub, whole = vec.pz_frac, vec.pz_sub, vec.pz_whole
            vlo, vhi = vec.vz_lo, vec.vz_hi
        else:
            frac, sub, whole = vec.py_frac, vec.py_sub, vec.py_whole
            vlo, vhi = vec.vy_lo, vec.vy_hi
        ext = 0
        # frac += vlo (CLC)
        t = frac + vlo
        frac = t & 0xFF
        carry = 1 if t > 0xFF else 0
        # if vhi (sign byte) negative: $0074-- (ext=-1)
        if vhi & 0x80:
            ext = (ext - 1) & 0xFF  # $0074 = $FF
        # sub = sub + vhi + carry(from frac add, which is the ADC carry chain)
        t2 = sub + vhi + carry
        sub = t2 & 0xFF
        carry2 = 1 if t2 > 0xFF else 0
        # whole = whole + ext + carry2
        ext_signed = -1 if ext == 0xFF else 0
        whole = (whole + ext_signed + carry2) & 0xFF
        if axis == 0:
            vec.px_frac, vec.px_sub, vec.px_whole = frac, sub, whole
        elif axis == 1:
            vec.pz_frac, vec.pz_sub, vec.pz_whole = frac, sub, whole
        else:
            vec.py_frac, vec.py_sub, vec.py_whole = frac, sub, whole


def _calc_tile_z_and_slope(state, x, y):
    """calculate_tile_address_z_and_slope $1DF9: (z, slope, carry_set, is_object, byte).

    An object tile (byte >= $C0) returns is_object with the other fields None, for the caller's
    object-stack walk.  carry_set is True for a sloping tile."""
    b = tile_byte(state, x, y)
    if b >= 0xC0:
        return None, None, None, True, b
    slope = b & 0x0F
    z = (b >> 4) & 0x0F
    carry_set = slope >= 1  # CPY #$1 -> carry set if slope!=0
    return z, slope, carry_set, False, b


def _get_min_xy_fraction(vec):
    """get_minimum_x_or_y_fraction_from_tile_centre $1EAF, literal $1EAF..$1ECB.

    (sub-$80), EOR #$FF when bit7 set, per axis, then the smaller; the exact 6502
    form, off by one against a true abs at the low side.  Returns (value, cycles)."""
    cyc = wcost.MINXY
    ax = (vec.px_sub - 0x80) & 0xFF  # $1EAF LDA $0037 ; SEC ; SBC #$80
    if ax & 0x80:  # $1EB4 BPL skip ; $1EB6 EOR #$ff
        ax ^= 0xFF
        cyc += wcost.MINXY_ABS
    t74 = ax & 0xFF  # $1EB8 STA $0074
    ay = (vec.py_sub - 0x80) & 0xFF  # $1EBA LDA $0039 ; SEC ; SBC #$80
    if ay & 0x80:  # $1EBF BPL skip ; $1EC1 EOR #$ff
        ay ^= 0xFF
        cyc += wcost.MINXY_ABS
    if ay >= (t74 & 0xFF):  # $1EC3 CMP $0074 ; $1EC5 BCS skip_min
        t74 = ay  # keep A=ay
    else:  # $1EC7 LDA $0074 -> t74 unchanged
        cyc += wcost.MINXY_Y_WINS
    return t74 & 0xFF, cyc


def _is_tree(vec, state, Y, minxy, c56, cdd):
    """is_tree $1E69: the enemy-can-see-a-tree marker ($0CDD).

    It works in $0075, not $0079, so it cannot change the surface height the LOS
    compare uses; the $1E8C compare reuses the $0074 the $1E48 call left.  Returns
    (cdd, cycles)."""
    zf = state.obj_z_frac[Y] & 0xFF
    t = zf - vec.pz_sub  # $1E69 LDA ; SEC ; SBC $0038
    s75 = t & 0xFF  # $1E6F STA $0075
    borrow = 1 if t < 0 else 0
    hi = (state.obj_z_height[Y] - vec.pz_whole - borrow) & 0xFF  # $1E71 LDA ; SBC $003B
    saved_hi = hi  # $1E76 PHA
    t2 = s75 + 0xE0  # $1E77 LDA $0075 ; CLC ; ADC #$e0
    s75 = t2 & 0xFF  # $1E7C STA $0075
    carry = 1 if t2 > 0xFF else 0
    a = (saved_hi + carry) & 0xFF  # $1E7E PLA ; $1E7F ADC #$0
    if a & 0x80:  # $1E81 BMI skip_targeting_object
        return cdd, wcost.OBJ_TREE_BELOW
    c = a & 1  # $1E83 LSR A
    a >>= 1
    s75 = ((s75 >> 1) | (c << 7)) & 0xFF  # $1E84 ROR $0075 (C from LSR A)
    c = a & 1  # $1E86 LSR A
    a >>= 1
    if a != 0:  # $1E87 BNE skip_targeting_object
        return cdd, wcost.OBJ_TREE_HIGH
    a = ((s75 >> 1) | (c << 7)) & 0xFF  # $1E89 LDA $0075 ; $1E8B ROR A
    if a < minxy:  # $1E8C CMP $0074 ; $1E8E BCC skip
        return cdd, wcost.OBJ_TREE_NEAR
    if c56 & 0x80:  # $1E90 BIT $0C56 ; $1E93 BMI skip
        return cdd, wcost.OBJ_TREE_TARGETED
    return (cdd >> 1) | 0x80, wcost.OBJ_TREE_SEEN  # $1E95 SEC ; $1E96 ROR $0CDD


def _get_tile_z_from_object(vec, state, raw, c56, cdd, c58):
    """get_tile_z_from_object $1E3F with its helpers ($1E0E, $1E48, $1E69, $1E99, $1EA4).

    One bounded object-stack walk: the object-tile branch of $1DF9 ($1E00 BCS).  Every exit
    leaves carry clear, so the caller always takes check_flat_tile.
    Returns (z, s79, c0c, c67, c56, cdd, s60, cycles)."""
    s60, s79, c0c, c67 = 0x80, 0, 0x80, 0
    raw &= 0xFF
    cyc = 0
    for _ in range(80):  # bound the stack walk
        Y = raw & 0x3F  # $1E3F AND #$3F ; TAY
        do_ghol = False
        if not (s60 & 0x80):  # $1E42 BIT $0060 ; $1E44 BPL ghol
            cyc += wcost.OBJ_HEAD_GHOL
            do_ghol = True
        else:
            cyc += wcost.OBJ_HEAD_LOS  # $1E46 BMI $1E0E (always)
            if Y == (c58 & 0xFF):  # $1E0E CPY $0C58
                c56 = (c56 >> 1) | 0x80  # $1E13 ROR $0C56
                cyc += wcost.OBJ_TARGET_HIT
            else:
                cyc += wcost.OBJ_TARGET_MISS
            otype = state.obj_type[Y] & 0xFF  # $1E16 LDA objects_type,Y
            if otype in (2, 3):  # $1E19/$1E1D boulder|tree -> $1E48
                cyc += wcost.OBJ_TYPE_BOULDER if otype == 3 else wcost.OBJ_TYPE_TREE
                frac, fcyc = _get_min_xy_fraction(vec)  # $1E48 JSR $1EAF
                cyc += fcyc
                if frac >= 0x40:  # $1E4B CMP #$40 ; $1E4D BCS skip
                    cyc += wcost.OBJ_BT_SKIP
                elif otype == 2:  # $1E52 CMP #$2 ; $1E54 BEQ is_tree
                    cyc += wcost.OBJ_BT_TYPE + wcost.OBJ_BT_TREE
                    cdd, tcyc = _is_tree(vec, state, Y, frac, c56, cdd)
                    cyc += tcyc
                else:  # boulder near-centre $1E56: targetable, RTS with z
                    cyc += wcost.OBJ_BT_TYPE + wcost.OBJ_BT_BOULDER
                    c67 = (c67 >> 1) | 0x80  # $1E56 SEC ; $1E57 ROR $0C67
                    t = (state.obj_z_frac[Y] & 0xFF) - 0x60  # $1E5A ; SBC #$60
                    s79 = t & 0xFF
                    borrow = 1 if t < 0 else 0
                    z = (state.obj_z_height[Y] - borrow) & 0xFF  # $1E62 ; SBC #$0
                    return (z, s79, c0c, c67, c56, cdd, s60, cyc)
                if state.obj_type[Y] != 2:  # skip_targeting_object $1E99
                    s60 = 0xC0
                    cyc += wcost.OBJ_SKIP_OTHER
                else:
                    cyc += wcost.OBJ_SKIP_TREE
                do_ghol = True
            elif otype != 6:  # $1E23 BNE ghol (robot/sentry/enemy)
                cyc += wcost.OBJ_TYPE_OTHER
                do_ghol = True
            else:  # platform (type 6) $1E25
                cyc += wcost.OBJ_TYPE_PLATFORM
                frac, fcyc = _get_min_xy_fraction(vec)  # $1E25 JSR $1EAF
                cyc += fcyc
                if frac >= 0x64:  # $1E28 CMP #$64 ; $1E2A BCS skip
                    cyc += wcost.OBJ_PLAT_SKIP
                    if state.obj_type[Y] != 2:  # $1E99
                        s60 = 0xC0
                        cyc += wcost.OBJ_SKIP_OTHER
                    else:
                        cyc += wcost.OBJ_SKIP_TREE
                    do_ghol = True
                else:
                    cyc += wcost.OBJ_PLAT_RTS
                    c0c = 0x10  # $1E2C LDA #$10 ; STA $000C
                    t = (state.obj_z_frac[Y] & 0xFF) + 0x20  # $1E30 ; ADC #$20
                    s79 = t & 0xFF
                    carry = 1 if t > 0xFF else 0
                    z = (state.obj_z_height[Y] + carry) & 0xFF  # $1E38 ; ADC #$0
                    return (z, s79, c0c, c67, c56, cdd, s60, cyc)
        if do_ghol:  # get_height_of_lowest_object $1EA4
            flags = state.obj_flags[Y] & 0xFF
            if flags >= 0x40:  # $1EA7 CMP #$40 ; $1EA9 BCS gtzfo
                cyc += wcost.OBJ_GHOL_LOOP
                raw = flags
                continue
            cyc += wcost.OBJ_GHOL_RTS
            z = state.obj_z_height[Y] & 0xFF  # $1EAB ; RTS
            return (z, s79, c0c, c67, c56, cdd, s60, cyc)
    Y = raw & 0x3F  # safety: corrupt/deep stack
    return (state.obj_z_height[Y] & 0xFF, s79, c0c, c67, c56, cdd, s60, cyc)


def check_for_line_of_sight_to_tile(
    vec, state, slot, do_los_checks=0x00, eye_z=None, max_steps=20000
):
    """$1CDD: march the ray; return (tx, ty, los_ok, cycles), los_ok True == carry clear.

    ``cycles`` is the ROM cost per sub-step by tile branch, with write weight packed above it.
    Runs :func:`_march_jit`, or the bit-identical :func:`_march_python` with ``_HAVE_JIT`` off.
    ``do_los_checks`` is $0C6E (0 for the player aim); a small ``max_steps`` cuts miss rays short.
    """
    if _HAVE_JIT:
        return _march_jit(vec, state, slot, do_los_checks, eye_z, max_steps)
    return _march_python(vec, state, slot, do_los_checks, eye_z, max_steps)


def _march_python(vec, state, slot, do_los_checks=0x00, eye_z=None, max_steps=20000):
    """Reference pure-Python march (the bit-exact 6502 transcription)."""
    _get_object_details(vec, state, slot, eye_z=eye_z)
    # $1CDF LSR $0C56 ; $1CE2 LSR $0CDD: clear the targeted-object trackers' top bits
    c56 = (state.mem[0x0C56] >> 1) & 0xFF
    cdd = (state.mem[0x0CDD] >> 1) & 0xFF
    # write the trackers back so a post-march reader sees the ROM value on edge returns too
    writable = not isinstance(state.mem, bytes)
    if writable:
        state.mem[0x0C56] = c56
        state.mem[0x0CDD] = cdd
    c58 = state.mem[0x0C58] & 0xFF  # $0C58 targeted object slot ($FF = none)
    c67 = 0  # $0C67 boulder-consider flag
    # $0C6E bit7 (read at $1D26) waives the looking-up rejection; the player path clears it
    c6e = do_los_checks & 0xFF
    tx = ty = 0
    cycles = 0
    # $1CBB costs more per negative high byte ($1CCC DEC $0074); the signs are fixed per march
    neg = sum(1 for h in (vec.vy_hi, vec.vz_hi, vec.vx_hi) if h & 0x80)
    step_fixed = wcost.ADD_VECTOR + wcost.ADD_VECTOR_NEG * neg + 2 * wcost.STEP_EDGE
    # $1CFB..$1DFE: only reached once both edge tests fall through.
    tile_fixed = (
        wcost.STEP_SETUP + wcost.TILE_Z_CALL + wcost.TILE_ADDR + wcost.TILE_Z_READ
    )
    # the ROM exits at the board edge; a ray still marching at the cap is treated as no-LOS
    for _step in range(max_steps):
        cycles += step_fixed  # $1CE8 JSR $1CBB: add_vector, then the edge tests
        _add_vector(vec)
        # $1CEB LDA $003A -> $0024 ; CMP #$1f ; BCS leave_with_carry_set
        tx = vec.px_whole
        if tx >= 0x1F:
            cycles += wcost.STEP_EDGE_EXIT - wcost.STEP_EDGE + wcost.LEAVE_SET
            return tx, ty, False, cycles
        ty = vec.py_whole  # $003C -> $0026
        if ty >= 0x1F:
            cycles += wcost.STEP_EDGE_EXIT + wcost.LEAVE_SET
            return tx, ty, False, cycles
        # $1CFB..$1D05: $0060 = $000C = $80, $0079 = 0, clear $0C67 (boulder-consider flag)
        cycles += tile_fixed
        s60 = 0x80
        c0c_var = 0x80  # $000C tolerance (object path may lower it)
        s79 = 0
        c67 = 0
        z, slope, carry_set, is_obj, raw = _calc_tile_z_and_slope(state, tx, ty)
        cycles += wcost.TILE_Z_OBJ if is_obj else wcost.TILE_Z_FLAT
        cycles += wcost.SLOPE_BRANCH if carry_set else wcost.FLAT_BRANCH
        if is_obj:
            # $1E00 BCS get_tile_z_from_object: z, $0079, $000C, $0C67, trackers; carry clear
            z, s79, c0c_var, c67, c56, cdd, s60, ocyc = _get_tile_z_from_object(
                vec, state, raw, c56, cdd, c58
            )
            cycles += ocyc
            # persist the $0C56/$0CDD tracker writes ($1E13 ROR) for post-march readers
            if writable:
                state.mem[0x0C56] = c56
                state.mem[0x0CDD] = cdd
            carry_set = False
            slope = 0
        if not carry_set:
            # check_flat_tile $1D0D: X=z ; A=$0079 ; SEC SBC $0038(pz_sub) -> $0079
            cycles += wcost.FLAT_DIFF
            tX = z & 0xFF
            t = s79 - vec.pz_sub
            s79 = t & 0xFF
            borrow = 1 if t < 0 else 0
            # TXA ; SBC $003B ; BMI on the 8-bit N flag, not the true sign
            d = (tX - vec.pz_whole - borrow) & 0xFF
            if d & 0x80:
                # BMI: 8-bit result negative -> tile below position, keep marching
                cycles += wcost.FLAT_BELOW
                continue
            if d != 0:
                # BNE leave_with_carry_set: tile above position
                cycles += wcost.FLAT_ABOVE + wcost.LEAVE_SET
                return tx, ty, False, cycles
            cycles += wcost.FLAT_TOL
            # $1D1C LDA $0079 ; CMP $000C ; BCS leave_set ($000C is $10 near a platform centre)
            if (s79 & 0xFF) >= c0c_var:
                cycles += wcost.FLAT_TOL_HIT + wcost.LEAVE_SET
                return tx, ty, False, cycles
            cycles += wcost.FLAT_BIT60
            # $1D22 BIT $0060 ; BVS leave_set: local $0060 bit6 marks a checked slope or object
            if s60 & 0x40:
                cycles += wcost.FLAT_BIT60_HIT + wcost.LEAVE_SET
                return tx, ty, False, cycles
            cycles += wcost.FLAT_ANGLE
            # $1D26 LDA $0C6E ; ORA $0C67 ; BMI: a targetable boulder skips the looking-up check
            if (c6e | c67) & 0x80:
                cycles += wcost.FLAT_ANGLE_SKIP
            else:
                cycles += wcost.FLAT_LOOKUP
                # LDA $0030 (vector_z high = vec.s30) ; BPL leave_set (looking up)
                if not (vec.s30 & 0x80):
                    cycles += wcost.FLAT_LOOKING_UP + wcost.LEAVE_SET
                    return tx, ty, False, cycles
            # skip_angle_check $1D32: same tile as observer? keep marching, else clear
            ox = state.obj_x[slot]
            oy = state.obj_y[slot]
            cycles += wcost.FLAT_SAME
            if (tx & 0xFF) != (ox & 0xFF):  # $1D39 BNE leave_with_carry_clear
                cycles += wcost.FLAT_SAME_X_DIFF
                return tx, ty, True, cycles
            cycles += wcost.FLAT_SAME_Y
            if (ty & 0xFF) == (oy & 0xFF):
                cycles += wcost.FLAT_SAME_HIT
                continue  # same tile as observer -> keep going
            cycles += wcost.FLAT_SAME_Y_DIFF
            return tx, ty, True, cycles  # leave_with_carry_clear (LOS!)
        else:
            # check_sloping_tile $1D46
            res, scyc = _check_sloping_tile(vec, state, tx, ty, z, slope)
            cycles += scyc
            if res == "loop":
                continue
            return tx, ty, False, cycles + wcost.LEAVE_SET  # $1D44 SEC + RTS
    return tx, ty, False, cycles


def _march_jit(vec, state, slot, do_los_checks=0x00, eye_z=None, max_steps=20000):
    """Numba-accelerated march: :func:`sentinel.los_jit.march` resolves flat, slope
    AND object tiles entirely in numba (the object-stack walk $1E3F is ported), so
    the march is a single numba call with no Python bail-out.  Bit-for-bit identical
    to :func:`_march_python`."""
    _get_object_details(vec, state, slot, eye_z=eye_z)
    # $1CDF LSR $0C56 ; $1CE2 LSR $0CDD -- clear top bits (targeted-object trackers).
    c56 = (state.mem[0x0C56] >> 1) & 0xFF
    cdd = (state.mem[0x0CDD] >> 1) & 0xFF
    writable = not isinstance(state.mem, bytes)
    if writable:
        state.mem[0x0C56] = c56
        state.mem[0x0CDD] = cdd
    c58 = state.mem[0x0C58] & 0xFF
    c6e = do_los_checks & 0xFF
    ox = state.obj_x[slot]
    oy = state.obj_y[slot]

    mem_np = np.frombuffer(state.mem, dtype=np.uint8)
    (
        status,
        tx,
        ty,
        vec.px_frac,
        vec.px_sub,
        vec.px_whole,
        vec.pz_frac,
        vec.pz_sub,
        vec.pz_whole,
        vec.py_frac,
        vec.py_sub,
        vec.py_whole,
        c56,
        cdd,
        _used,
        cycles,
    ) = los_jit.march(
        mem_np,
        vec.vx_lo,
        vec.vx_hi,
        vec.vz_lo,
        vec.vz_hi,
        vec.vy_lo,
        vec.vy_hi,
        vec.s30,
        vec.px_frac,
        vec.px_sub,
        vec.px_whole,
        vec.pz_frac,
        vec.pz_sub,
        vec.pz_whole,
        vec.py_frac,
        vec.py_sub,
        vec.py_whole,
        ox,
        oy,
        c6e,
        c58,
        c56,
        cdd,
        max_steps,
    )
    # persist the marched-out trackers; only the final value is observable
    if writable:
        state.mem[0x0C56] = c56 & 0xFF
        state.mem[0x0CDD] = cdd & 0xFF
    return tx, ty, status == los_jit.LOS_CLEAR, cycles


def _check_sloping_tile(vec, state, x, y, z00, slope):
    """check_sloping_tile $1D46. Returns (verdict, cycles): 'loop' (above the slope) or 'blocked'.

    Corner heights: $0073=$0077 = z00 at (x,y), $0076 at (x+1,y), $0075 at (x+1,y+1),
    $0074 at (x,y+1).
    """
    cyc = wcost.SLOPE_HEAD
    p73 = z00 & 0xFF
    _p77 = z00 & 0xFF
    # INC $0024 -> z at (x+1,y) -> $0076
    p76, c = _slope_corner_z(state, x + 1, y)
    cyc += c
    # INC $0026 -> z at (x+1,y+1) -> $0075
    p75, c = _slope_corner_z(state, x + 1, y + 1)
    cyc += c
    # DEC $0024 -> z at (x,y+1) -> $0074
    p74, c = _slope_corner_z(state, x, y + 1)
    cyc += c
    # DEC $0026 ; calculate_tile_address ; read slope nibble
    nib = tile_byte(state, x, y) & 0x0F
    # CMP #$4 BEQ ... CMP #$c BNE tile_is_corner_or_quad
    if nib in (0x04, 0x0C):
        cyc += wcost.SLOPE_NIB_4 if nib == 0x04 else wcost.SLOPE_NIB_12
        # tile_has_one_flat_and_one_sloping_edge: compare $003B against all 4 corners
        cyc += wcost.SLOPE_EDGE_LDA
        b = vec.pz_whole & 0xFF
        for corner in (p73, p74, p75, p76):
            if b >= corner:  # BCS to_loop
                return "loop", cyc + wcost.SLOPE_EDGE_HIT
            cyc += wcost.SLOPE_EDGE_MISS
        return "blocked", cyc + wcost.SLOPE_EDGE_BLOCK
    # tile_is_corner_or_quadrilateral $1D8A
    verdict, qcyc = _slope_corner_or_quad(vec, state, nib, p73, p74, p75, p76)
    return verdict, cyc + wcost.SLOPE_NIB_QUAD + qcyc


def _slope_corner_z(state, x, y):
    """One $1D4E/$1D55/$1D5C corner read inside check_sloping_tile.

    $1D4A LSR $0060 clears bit7, so an object tile takes $1E44 BPL straight to
    get_height_of_lowest_object and leaves no trace.  Returns (z, cycles)."""
    cyc = wcost.TILE_Z_CALL + wcost.TILE_ADDR + wcost.TILE_Z_READ
    z, _slope, _carry_set, is_obj, raw = _calc_tile_z_and_slope(
        state, x & 0xFF, y & 0xFF
    )
    if not is_obj:
        return z & 0xFF, cyc + wcost.TILE_Z_FLAT
    cyc += wcost.TILE_Z_OBJ
    o = raw & 0x3F
    for _ in range(64):
        cyc += wcost.OBJ_HEAD_GHOL
        if state.obj_flags[o] < 0x40:
            break
        cyc += wcost.OBJ_GHOL_LOOP
        o = state.obj_flags[o] & 0x3F
    return state.obj_z_height[o] & 0xFF, cyc + wcost.OBJ_GHOL_RTS


_EDGES = (0x00, 0x03, 0x01, 0x00, 0x01, 0x02, 0x02, 0x03)  # the $1DF1-$1DF8 ROM table


def _slope_corner_or_quad(vec, state, nib, p73, p74, p75, p76):
    """$1D8A-$1DEE corner/quadrilateral slope, opcode-by-opcode with explicit A and carry.

    The $1D8B BCC lands on $1D9C, an extra LSR before use_corner_for_slope.
    Returns (verdict, cycles): 'loop' (ray above the slope) or 'blocked' (tile hit)."""
    # zero-page square: $73=p73, $74=p74, $75=p75, $76=p76, $77=p73 (first==last)
    corners = [p73 & 0xFF, p74 & 0xFF, p75 & 0xFF, p76 & 0xFF, p73 & 0xFF]  # $73..$77

    cyc = 0
    A = nib & 0xFF
    # $1D8A LSR A
    C = A & 1
    A >>= 1
    if C == 0:
        cyc += wcost.SLOPE_Q_C2
        # $1D8B BCC tile_is_corner_type_two lands on $1D9C LSR A (the extra LSR)
        C = A & 1
        A >>= 1
        # falls into use_corner_for_slope $1D9D
        s78 = A  # STA $0078
        # $1D9F LSR A
        C = A & 1
        A >>= 1
        # $1DA0 LDA $0037 (px_sub)
        A = vec.px_sub & 0xFF
        if C:
            A ^= 0xFF  # $1DA4 EOR #$ff
            cyc += wcost.SLOPE_Q_CORNER_EOR
        cyc += wcost.SLOPE_Q_CORNER
        # $1DA6 CMP $0039 (py_sub): carry = A >= py_sub
        C = 1 if A >= (vec.py_sub & 0xFF) else 0
        # $1DA8 LDA $0078 ; $1DAA ROL A ; $1DAB TAY ; $1DAC LDA edges,Y
        A = s78
        A = ((A << 1) | C) & 0xFF  # ROL A
        Y = A
        A = _EDGES[Y] if Y < len(_EDGES) else 0
    else:
        # $1D8D LSR A
        C = A & 1
        A >>= 1
        if C == 1:
            cyc += wcost.SLOPE_Q_C1
            # $1D8E BCS -> $1D95 ADC #$1 (no extra LSR): A + 1 + C, C set by $1D8D
            A = (A + 1 + C) & 0xFF
            A &= 0x03  # $1D97 AND #$3
            # $1D99 JMP use_corner_for_slope $1D9D
            s78 = A  # $1D9D STA $0078
            C = A & 1
            A >>= 1  # $1D9F LSR A
            A = vec.px_sub & 0xFF  # $1DA0 LDA $0037
            if C:
                A ^= 0xFF  # $1DA4 EOR #$ff
                cyc += wcost.SLOPE_Q_CORNER_EOR
            cyc += wcost.SLOPE_Q_CORNER
            C = 1 if A >= (vec.py_sub & 0xFF) else 0  # $1DA6 CMP $0039
            A = ((s78 << 1) | C) & 0xFF  # $1DA8 LDA $0078 ; $1DAA ROL A
            Y = A
            A = _EDGES[Y] if Y < len(_EDGES) else 0
        else:
            # $1D90 AND #$1 ; $1D92 JMP use_edge_for_slope
            cyc += wcost.SLOPE_Q_EDGE
            A = A & 1

    # use_edge_for_slope $1DAF: TAX ; LSR A ; LDY $0037 ; BCS use_x ; LDY $0039
    X = A & 0xFF
    C = A & 1
    A >>= 1  # $1DB0 LSR A
    Yreg = vec.px_sub & 0xFF  # $1DB1 LDY $0037
    cyc += wcost.SLOPE_Q_TAIL
    if not C:
        Yreg = vec.py_sub & 0xFF  # $1DB5 use_y_for_slope
        cyc += wcost.SLOPE_Q_USE_Y
    C = A & 1
    A >>= 1  # $1DB7 LSR A
    A = Yreg  # $1DB8 TYA
    if C:
        A ^= 0xFF  # $1DBB EOR #$ff
        cyc += wcost.SLOPE_Q_INVERT
    s02 = A & 0xFF  # $1DBD STA $0002

    # $1DBF LDA $73,X ; STA $0078
    s78 = corners[X & 3]
    # $1DC3 LDA $74,X ; SEC ; SBC $73,X
    a = (corners[(X & 3) + 1] - corners[X & 3]) & 0x1FF
    res = a & 0xFF
    neg = bool(res & 0x80)  # PHP: N flag of the subtraction
    if neg:
        res = ((res ^ 0xFF) + 1) & 0xFF  # abs
        cyc += wcost.SLOPE_Q_ABS
    s75 = res  # $1DD0 STA $0075
    # $1DD2 LDA $0002 ; $1DD4 JSR multiply_byte_by_byte ($0D03): $0074=$0002 * $0075
    prod_h, prod_lo = _mul8(s02, s75)  # A=high, $0074=low
    cyc += passcost.MUL8_BIT * _bits(s02)
    # $1DD7 PLP ; $1DD8 invert (A:$0074) if the saved slope subtraction was negative
    if neg:
        prod_h, prod_lo = invert16(prod_h, prod_lo)
        cyc += wcost.SLOPE_Q_NEG
    # $1DDB CLC ; ADC $0078 -> $0075
    s75b = (prod_h + s78) & 0xFF
    # $1DE0 LDA $0038 ; SEC ; SBC $0074(=prod_lo) ; LDA $003B ; SBC $0075(=s75b) ; BPL
    lo = vec.pz_sub - prod_lo
    borrow = 1 if lo < 0 else 0
    hi8 = (vec.pz_whole - s75b - borrow) & 0xFF
    if hi8 & 0x80:  # BPL false -> below slope -> hit
        return "blocked", cyc + wcost.SLOPE_Q_BLOCK
    return "loop", cyc


# keyboard pan lattice: h in AZIMUTH_STEP notches, pitch every 4 in the $10FF/$1149 clamp
AZIMUTH_STEP = 8  # h ≡ 0 (mod 8): ±8 pans / u-turn EOR $80 from any body's facing
PITCH_BAND = [
    v & 0xFF for v in list(range(0xCD, 0x100, 4)) + list(range(0x01, 0x36, 4))
]

# each 1px cursor step ($9965/$9994) is a distinct $1C10 sub-angle; bodies face v $F5 ($1F7E)
KBD_V_ANGLE = 0xF5
_CURSOR_CY_LO, _CURSOR_CY_HI = 32, 159  # ROM cursor y clamp $20..$9F ($9994)
CURSOR_CX = list(
    range(48, 112)
)  # spans one AZIMUTH_STEP of cx>>3, centre 80 (== SIGHTS_CX)
CURSOR_CY = list(
    range(63, 127)
)  # one PITCH_BAND step of (cy-5)>>4, centre 95 (== SIGHTS_CY)
CURSOR_CY_FULL = list(range(_CURSOR_CY_LO, _CURSOR_CY_HI + 1))

# body pitch is a further aim DOF for near/below builds; $F5 first keeps its views
_V_PRIORITY = [KBD_V_ANGLE] + [v for v in PITCH_BAND if v != KBD_V_ANGLE]

# a ray vector depends only on the aim, so los_jit.build_lattice builds each lattice once
_VEC_CACHE = {}


def kbd_grids(cxs=None, cys=None, v_primary=False):
    """The ``(hgrid, vgrid, cxs, cys)`` keyboard-aim lattice every sweep and query runs on.

    ``v_primary`` = the $F5 plane alone, whose sole pitch control is the cursor cy, so it
    must span :data:`CURSOR_CY_FULL` (see :func:`_landable_sweep`); ``cxs``/``cys`` override
    the cursor grid (:mod:`sentinel.landtable`'s 2:1 landset subsample)."""
    return (
        list(range(0, 256, AZIMUTH_STEP)),
        [KBD_V_ANGLE] if v_primary else _V_PRIORITY,
        CURSOR_CX if cxs is None else cxs,
        (CURSOR_CY_FULL if v_primary else CURSOR_CY) if cys is None else cys,
    )


def _lattice_vectors(hgrid, vgrid, cxs, cys):
    """Build (or fetch from :data:`_VEC_CACHE`) the keyboard lattice's ray-vector arrays.

    Order is ``for v: for h: for cx: for cy``; returns ``(vx_lo, vx_hi, vz_lo, vz_hi, vy_lo,
    vy_hi, s30)`` for :func:`los_jit.march_batch`, each aim recovered by :func:`_meta_at`.
    """
    key = (tuple(hgrid), tuple(vgrid), tuple(cxs), tuple(cys))
    cached = _VEC_CACHE.get(key)
    if cached is not None:
        return cached
    cached = los_jit.build_lattice(
        np.asarray(hgrid, dtype=np.int16),
        np.asarray(vgrid, dtype=np.int16),
        np.asarray(cxs, dtype=np.int16),
        np.asarray(cys, dtype=np.int16),
    )
    _VEC_CACHE[key] = cached
    return cached


def _meta_at(i, hgrid, vgrid, cxs, cys):
    """Reconstruct the ``(h, v, cx, cy)`` aim for flat lattice index ``i`` (order
    ``for v: for h: for cx: for cy``) -- the inverse of :func:`los_jit.build_lattice`'s
    indexing, so no per-aim meta list is stored."""
    nh, ncx, ncy = len(hgrid), len(cxs), len(cys)
    per_h = ncx * ncy
    per_v = nh * per_h
    vi, rem = divmod(i, per_v)
    hi, rem2 = divmod(rem, per_h)
    cxi, cyi = divmod(rem2, ncy)
    return hgrid[hi], vgrid[vi], cxs[cxi], cys[cyi]


def _seed_position(state, slot, eye_z):
    """The ray seed (get_object_details $1ECC) shared by every aim in a sweep:
    ``(px_frac, px_sub, px_whole, pz_frac, pz_sub, pz_whole, py_frac, py_sub, py_whole)``.
    """
    z = state.obj_z_height[slot] if eye_z is None else (eye_z & 0xFF)
    return (
        0,
        0x80,
        state.obj_x[slot] & 0xFF,
        0,
        state.obj_z_frac[slot] & 0xFF,
        z & 0xFF,
        0,
        0x80,
        state.obj_y[slot] & 0xFF,
    )


def _landable_batch(state, slot, eye_z, max_steps, hgrid, vgrid, cxs, cys):
    """Run the whole keyboard-aim lattice against `state` in ONE numba call and return
    ``(status, tx, ty, centre, grids)`` where ``grids = (hgrid, vgrid, cxs, cys)`` -- the
    per-aim ``(h, v, cx, cy)`` is reconstructed from the flat index via :func:`_meta_at`.
    First-hit order = lattice order (``for v: for h: for cx: for cy``).
    """
    vx_lo, vx_hi, vz_lo, vz_hi, vy_lo, vy_hi, s30 = _lattice_vectors(
        hgrid, vgrid, cxs, cys
    )
    px_f, px_s, px_w, pz_f, pz_s, pz_w, py_f, py_s, py_w = _seed_position(
        state, slot, eye_z
    )
    mem_np = np.frombuffer(state.mem, dtype=np.uint8)
    c56 = (state.mem[0x0C56] >> 1) & 0xFF
    cdd = (state.mem[0x0CDD] >> 1) & 0xFF
    status, tx, ty, centre = los_jit.march_batch(
        mem_np,
        vx_lo,
        vx_hi,
        vz_lo,
        vz_hi,
        vy_lo,
        vy_hi,
        s30,
        px_f,
        px_s,
        px_w,
        pz_f,
        pz_s,
        pz_w,
        py_f,
        py_s,
        py_w,
        state.obj_x[slot] & 0xFF,
        state.obj_y[slot] & 0xFF,
        state.mem[0x0C6E] & 0x7F,
        state.mem[0x0C58] & 0xFF,
        c56,
        cdd,
        max_steps,
    )
    return status, tx, ty, centre, (hgrid, vgrid, cxs, cys)


def _landable_sweep(state, slot, eye_z, max_steps, want_centres, v_primary=False):
    """Sweep the keyboard aim lattice: ``(views, centres)``, first-LOS view and min centre per tile.

    ``v_primary`` keeps only the $F5 body plane (a climb foothold never needs a pitched-down
    body); cursor cy is then the sole pitch control, so it spans :data:`CURSOR_CY_FULL`.  The cy
    window matches the full range only beside the body pitch band; the cx window always does.
    """
    grids = kbd_grids(v_primary=v_primary)
    status, tx, ty, centre, grids = _landable_batch(
        state, slot, eye_z, max_steps, *grids
    )
    views = {}
    centres = {}
    clear = np.flatnonzero(status == los_jit.LOS_CLEAR)
    if clear.size:
        # `clear` ascends, so np.unique's first indices, re-sorted, keep discovery order
        key = (tx[clear].astype(np.int64) << 16) | ty[clear].astype(np.int64)
        uniq, first = np.unique(key, return_index=True)
        order_ins = np.argsort(first, kind="stable")
        uniq = uniq[order_ins]
        first = first[order_ins]
        cmin = None
        if want_centres:
            # min tile-centre per tile: stable-sort into key groups, reduceat each minimum
            order = np.argsort(key, kind="stable")
            ks = key[order]
            cs = centre[clear][order].astype(np.int64)
            bnd = np.concatenate(([0], np.flatnonzero(ks[1:] != ks[:-1]) + 1))
            mins = np.minimum.reduceat(cs, bnd)
            cmin = dict(zip(ks[bnd].tolist(), mins.tolist()))
        for k, fi in zip(uniq.tolist(), first.tolist()):
            h, v, cx, cy = _meta_at(int(clear[fi]), *grids)
            tile = (int(k >> 16), int(k & 0xFFFF))
            views[tile] = {"h_angle": h, "v_angle": v, "cursor": [cx, cy]}
            if want_centres:
                centres[tile] = int(cmin[k])
    return views, centres


def landable_sweep_with_centres(
    state, slot=None, eye_z=None, max_steps=6000, v_primary=False
):
    """The buildability oracle: one sweep of the keyboard aim lattice returning (views, centres).

    ``views`` maps (tx,ty) to the first-LOS {"h_angle","v_angle","cursor"}; ``centres`` to the
    min tile-centre fraction (the $1E48 boulder gate needs < $40).  ``v_primary=True`` sweeps
    only the $F5 plane, for the hot climb loop.
    """
    if slot is None:
        slot = state.player
    return _landable_sweep(
        state, slot, eye_z, max_steps, want_centres=True, v_primary=v_primary
    )
