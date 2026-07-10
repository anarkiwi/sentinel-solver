"""The line of sight -- a bit-exact, pure-Python transcription of the game's
action-time aim + LOS path.

This reproduces the C64 ROM's player aim and ray-march in integer math exactly
(bit-for-bit), so the simulator can answer "can the observer at this tile, aimed
this way, see that tile?" in microseconds with no emulator.

Faithful transcription of these routines:

  prepare_vector_from_player_sights  $1C10
  prepare_vector_from_angle          $1C54
  sin_cos_lookup                     $0E75  (multiply_double_A_by_pi $0F3E,
      method one $0ECB, method two $0EA1, sign fix $0F19)
  multiply_byte_by_byte              $0D03/$0D05
  multiply_double_by_double          $0F9E
  multiply_double_by_byte            $0F4A
  process_sine_or_cosine             $1C9D
  set_vector                         $1C7D
  invert_A_and_a_fraction[_if_negative] $1009/$1007
  get_object_details                 $1ECC
  add_vector_to_object_position      $1CBB
  check_for_line_of_sight_to_tile    $1CDD
      (calculate_tile_address_z_and_slope $1DF9, calculate_tile_address $2BA8,
       check_sloping_tile $1D46, looking-up rejection $1D2E)

The player-aim path uses sin_cos_lookup $0E75 (a polynomial); it does NOT use the
$AC80 table (only the enemy/relative-angle path $848F reaches that), so no table
is needed here.

Public entry points:
  aim_target(state, h_angle, v_angle, cur_x, cur_y, player_slot, eye_z=None)
      -> (tx, ty, los): where the sights, at that pan, land and whether visible.
  landable_views / landable_view / landable_sweep_with_centres: the ROM-faithful
      keyboard-aim buildability oracle (which tiles a real keyboard aim can land on).

`state` is a :class:`sentinel.state.State`.
"""

import math

import numpy as np

from sentinel.terrain import tile_byte

try:
    from sentinel import los_jit

    _HAVE_JIT = True
except Exception:  # pragma: no cover - numba absent -> pure-Python fallback
    los_jit = None
    _HAVE_JIT = False


# ============================================================================
# math core (multiply / invert) -- $0D03-$1016
# ============================================================================
def _mul8(a, b):
    """multiply_byte_by_byte ($0D03 entry, A=0): returns (high, low) of a*b using
    the ROM's shift-add. Result is exactly (a*b) >> ... well it is the standard
    8x8->16 product: high = (a*b)>>8, low = (a*b)&0xFF."""
    p = (a & 0xFF) * (b & 0xFF)
    return (p >> 8) & 0xFF, p & 0xFF


def invert16(high, frac):
    """invert_A_and_a_fraction $1009: negate the 16-bit (high:frac)."""
    val = ((high & 0xFF) << 8) | (frac & 0xFF)
    neg = (-val) & 0xFFFF
    return (neg >> 8) & 0xFF, neg & 0xFF


def invert16_if_negative(high, frac):
    """invert_A_and_a_fraction_if_negative $1007."""
    if high & 0x80:
        return invert16(high, frac)
    return high & 0xFF, frac & 0xFF


def multiply_double_by_byte(low74, high75, byte76):
    """multiply_double_by_byte $0F4A: ($0075:$0074) * $0076, the double (16-bit)
    times byte. Returns (low=$0074, high=$0075).

    ROM body ($0F4A..$0F61):
      $0075=high75 already; $0F4A JSR multiply_byte_by_byte ($0D05) with $0074=low74,
      $0075=$0076(byte)? -- trace carefully below.
    """
    # $0F4A is reached as 'multiply_double_by_byte'. Trace from $0F4A:
    #   (the label multiply_double_by_byte == $0F4A). Body:
    #   $0F4A JSR multiply_byte_by_byte (entry $0D05) -- multiplier_low=$0074(low74),
    #        multiplicand=$0075. But what is $0075 here? Callers set $0075=high75 and
    #        $0076=byte. The $0D05 entry computes $0074 * $0075.
    # Re-reading the actual ROM: multiply_double_by_byte begins at $0F4A which is
    # INSIDE multiply_double_A_by_pi ($0F3E..$0F61). The shared tail at $0F4A:
    #   $0F4A JSR $0D05 (mult $0074*$0075 -> A=high) ; $0F4D STA $0077 ;
    #   $0F4F LDA $0076 ; $0F51 JSR $0D03 ($0074=$0076 ; *$0075 -> A=high) ;
    #   $0F54 STA $0075 ; $0F56 LDA $0077 ; CLC ADC $0074 -> $0074 ; BCC/INC $0075.
    # So it computes ($0074:$0076) interpreted as a 16-bit value times $0075:
    #   low part:  $0074 * $0075  -> r1 (high=r1h, low=r1l); $0077 = r1h
    #   high part: $0076 * $0075  -> r2 (high=r2h, low=r2l); $0075 = r2h
    #   $0074 = $0077(r1h) + r2l ; carry -> $0075++
    # Returns A=$0074(result_low), a=$0075(result_high).
    r1h, _r1l = _mul8(low74, high75)
    r2h, r2l = _mul8(byte76, high75)
    res75 = r2h
    total = r1h + r2l
    res74 = total & 0xFF
    if total > 0xFF:
        res75 = (res75 + 1) & 0xFF
    return res74, res75


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
    r1h, _r1l = _mul8(frac, 0xC9)
    r77 = r1h  # STA $0077 (a_fraction discarded)
    # LDA $0076 ; JSR multiply_byte_by_byte ($0F51 `20 03 0d` entry $0D03):
    #   $0D03 STA $0074(=m76) ; $0D05 mult m76*$0075($c9)
    r2h, _r2l = _mul8(m76, 0xC9)
    r75 = r2h  # STA $0075
    # LDA $0077 ; CLC ADC $0074(=_r2l) -> $0074 ; BCC/INC $0075
    total = r77 + _r2l
    r74 = total & 0xFF
    if total > 0xFF:
        r75 = (r75 + 1) & 0xFF
    return r74, r75  # (result_low, result_high)


def sin_cos_lookup(angle, frac74=0):
    """$0E75: sine & cosine of `angle` (byte; full circle=256), with incoming
    a_fraction $0074=frac74. Returns (sin_lo, cos_lo, sin_hi, cos_hi) -- the four
    bytes at &0C00..&0C03; bit0 of the low byte is the sign flag (set => negate)."""
    angle &= 0xFF
    c0c = angle  # $0C0C
    aPI_lo, aPI_hi = multiply_double_A_by_pi(angle, frac74)
    c53 = aPI_lo  # $0C53
    c54 = aPI_hi  # $0C54
    # X=1; $0060=1; X=0; BIT $0C0C: V(bit6) set => INX, DEC $0060
    sixty = 1
    X = 0
    if c0c & 0x40:
        X = 1
        sixty = 0
    # After $0E81 LDA $0075 / STA $0C54, A = c54. The loop-top CMP #$7a tests A,
    # which on pass 1 is c54 and on pass 2 is the recomputed c54.
    A_cmp = c54
    cur_c53 = c53
    cur_c54 = c54
    cur_75 = aPI_hi
    sc_low = [0, 0]
    sc_high = [0, 0]

    while True:
        if (A_cmp & 0xFF) >= 0x7A:
            # method two $0ECB (A>=$7a): sin a =~ 1 - 2*(PI/4 - aPI)^2
            # A=#0; SEC; SBC $0C53 -> $0074
            t74 = (0 - cur_c53) & 0xFF
            borrow1 = 1 if (0 - cur_c53) < 0 else 0
            # LDA #$c9; SBC $0C54 -> $0075,$0076 ; (PI/4 - aPI)
            v = 0xC9 - cur_c54 - borrow1
            t75 = v & 0xFF
            t76 = t75
            # multiply_double_by_byte: ($0074:$0076)*$0075 -> (PI/4-aPI)^2
            r74, r75 = multiply_double_by_byte(t74, t75, t76)
            # ASL $0074 ; ROL $0075 (double)
            c = (r74 >> 7) & 1
            r74 = (r74 << 1) & 0xFF
            r75 = ((r75 << 1) | c) & 0xFF
            # A=#0; SEC; SBC $0074 ; AND #$fe -> low
            sub_lo = 0 - r74
            low = (sub_lo & 0xFF) & 0xFE
            borrow2 = 1 if sub_lo < 0 else 0
            sc_low[X] = low
            # A=#0; SBC $0075 ; BCC skip_overflow.
            # SBC borrows (carry CLEAR) when 0 - r75 - borrow2 < 0 -> BCC taken ->
            # store the (negative) result. Carry SET (>=0) -> the overflow clamp.
            sub_hi = 0 - r75 - borrow2
            if sub_hi < 0:
                sc_high[X] = sub_hi & 0xFF  # skip_overflow: store result_high
            else:
                # carry set: clamp to &fffe (lowest bit = sign)
                sc_low[X] = 0xFE
                sc_high[X] = 0xFF
        else:
            # method one $0EA1 (A<$7a)
            r_high1, _ = _mul8(0xAB, cur_75)
            r_high2, r_low2 = _mul8(r_high1, cur_75)
            t76 = r_high2  # STA $0076
            r74, r75 = multiply_double_by_byte(r_low2, cur_75, t76)
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
            break
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
    # NB sc index 0 is the slot computed first. With X start 0 (or 1 if opposite),
    # index 0 = sine, index 1 = cosine ALWAYS (X selects which is computed first but
    # both end in their fixed slots: pass1 writes sc[X], pass2 writes sc[sixty];
    # {X,sixty}={0,1}, so slot0 and slot1 always get filled -- slot0=sine, slot1=cos.)
    # $0F19: if $0C0C bit7 set (sin negative) -> ORA #1 into sin_lo
    if c0c & 0x80:
        sin_lo |= 1
    # cos sign: A=$0C0C; ASL A; EOR $0C0C; if result bit7 set -> cos negative
    t = ((c0c << 1) & 0xFF) ^ c0c
    if t & 0x80:
        cos_lo |= 1
    return sin_lo & 0xFF, cos_lo & 0xFF, sin_hi & 0xFF, cos_hi & 0xFF


def process_sine_or_cosine(low, high):
    """process_sine_or_cosine $1C9D: take (low,high) of sin or cos (low bit0=sign),
    divide by 16, invert if sign bit set. Returns (A=result_high, X=result_low)."""
    t74 = low & 0xFF  # STA $0074
    A = high & 0xFF

    # LSR A ; ROR $0074 ; PHP (save the carry = bit0 of low = the sign) ; then three
    # more (LSR A ; ROR $0074) to divide by 16 ; PLP -> carry = sign ; invert if set.
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
    if saved_carry:  # PLP; BCC skip_inversion
        A, t74 = invert16(A, t74)
    X = t74 & 0xFF  # LDX $0074
    return A & 0xFF, X


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
    # if $006B negative: negate y (16-bit), flip $0067 bit7
    if s6b & 0x80:
        neg = (-(((s6b << 8) | s6a))) & 0xFFFF
        s6a = neg & 0xFF
        s6b = (neg >> 8) & 0xFF
        s67 ^= 0x80
    # if $0068 bit0 set: flip sign
    if s68 & 1:
        s67 ^= 0x80
    # $0075 = $006B ; A=$0068 ; multiply_byte_by_byte ($0D03): $0074=$0068 * $0075
    r1h, xlow_yhigh_low = _mul8(s68, s6b)  # x_low * y_high (high, low)
    r77 = r1h  # STA $0077
    # $0074 = result_low of that mult; LDA $0074 ; CLC ADC #$80 (round) -> $0076
    rounded = xlow_yhigh_low + 0x80
    r76 = rounded & 0xFF
    if rounded > 0xFF:
        r77 = (r77 + 1) & 0xFF
    # $0075=$006B ; LDA $0069 ; multiply_byte_by_byte: $0069 * $006B -> $0078=high
    r2h, r2l = _mul8(s69, s6b)  # x_high * y_high
    r78 = r2h
    # LDA $0074(=r2l) ; CLC ADC $0077 -> $0077 ; BCC/INC $0078
    total = r2l + r77
    r77 = total & 0xFF
    if total > 0xFF:
        r78 = (r78 + 1) & 0xFF
    # $0075=$006A ; LDA $0069 ; multiply_byte_by_byte: $0069 * $006A -> high,low
    r3h, r3l = _mul8(s69, s6a)  # x_high * y_low
    # $0075=r3h, $0074=r3l ; LDA $0074(=r3l); CLC ADC $0076 -> A ; LDA $0075(=r3h)
    #   ADC $0077 -> $0074 ; BCC/INC $0078
    t = r3l + r76
    carry = 1 if t > 0xFF else 0
    t2 = r3h + r77 + carry
    res74 = t2 & 0xFF
    if t2 > 0xFF:
        r78 = (r78 + 1) & 0xFF
    # LDA $0078 ; BIT $0067 ; invert_if_negative
    A = r78 & 0xFF
    frac = res74 & 0xFF
    if s67 & 0x80:
        A, frac = invert16(A, frac)
    return A & 0xFF, frac & 0xFF


# ============================================================================
# vector construction -- prepare_vector_from_player_sights / _from_angle
# ============================================================================
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


def prepare_vector_from_player_sights(state, h_angle, v_angle, cur_x, cur_y, slot):
    """$1C10: build vector h_angle ($003E) + h_frac ($003D) and v_angle ($0040) +
    v_frac ($003F,$0074) from the sights cursor + objects_h_angle/objects_v_angle.
    Then call prepare_vector_from_angle $1C54.
    Returns a Vector with vx/vz/vy and the $30/$33/$32 (sin/cos of v) set."""
    cc6 = cur_x & 0xFF
    # $1C13 STA $0075 ; A=#0 ; LSR $75/ROR A x3 -> A = (cc6 >> ... ) the bottom 3 bits
    # of cc6 shifted into top of A: effectively A = (cc6 << 5) & 0xE0 ; $0075=cc6>>3
    s75 = cc6
    A = 0
    for _ in range(3):
        c = s75 & 1
        s75 >>= 1
        A = (A >> 1) | (c << 7)
    # CLC ; STA $003D (h_frac)
    h_frac = A & 0xFF
    # LDA $0075 ; ADC objects_h_angle,X ; SEC ; SBC #$0a -> $003E
    # NB the player aim sets objects_h_angle[player] = h_angle (the param), so we use
    # h_angle directly as objects_h_angle[slot] (matches aim_oracle.aim_target).
    obj_h = h_angle & 0xFF
    val = s75 + obj_h  # ADC (carry clear from CLC at $1C20)
    # SEC SBC #$0a
    h_angle_v = (val - 0x0A) & 0xFF
    # $1C2D LDA $0CC7 ; SEC SBC #$05 -> $0075
    s75 = (cur_y - 0x05) & 0xFF
    # A=#0 ; LSR $75/ROR A x4 -> A = bottom 4 bits of s75 into top of A; $0075 = s75>>4
    A = 0
    for _ in range(4):
        c = s75 & 1
        s75 >>= 1
        A = (A >> 1) | (c << 7)
    # CLC ADC #$20 -> $003F (v_frac) ; STA $0074
    v_frac = (A + 0x20) & 0xFF
    s74 = v_frac
    # LDA $0075 ; ADC objects_v_angle,X ; CLC ADC #$03 -> $0040 (v_angle)
    obj_v = v_angle & 0xFF
    # ADC objects_v_angle,X with carry = carry out of the (A+$20) above
    carry_in = 1 if (A + 0x20) > 0xFF else 0
    val2 = s75 + obj_v + carry_in
    # CLC ADC #$03
    v_angle_v = ((val2 & 0xFF) + 0x03) & 0xFF
    return prepare_vector_from_angle(h_angle_v, h_frac, v_angle_v, v_frac, s74)


def prepare_vector_from_angle(h_angle, h_frac, v_angle, v_frac, s74):
    """$1C54: compute the unit direction vector from horizontal/vertical angles.
    h_frac=$003D, h_angle=$003E, v_frac/s74 used as a_fraction for v's sin_cos.
    Returns a Vector."""
    vec = Vector()
    # $1C54 sin_cos_lookup of vertical angle (A=$0040=v_angle, $0074=s74... actually
    # at $1C54 entry $0074 holds v_frac stored at $1C48 STA $0074). Wait prepare_
    # vector_from_angle is also entered standalone with $0074 set by caller. For the
    # player path $1C48 STA $0074 = v_frac. So a_fraction = v_frac.
    sin_lo_v, cos_lo_v, sin_hi_v, cos_hi_v = sin_cos_lookup(v_angle, v_frac)
    # process cosine (Y=1) -> $0033=A, $0032=X
    A_cos, X_cos = process_sine_or_cosine(cos_lo_v, cos_hi_v)
    s33 = A_cos
    s32 = X_cos
    # process sine (Y=0) -> $0030=A, $002D=X
    A_sin, X_sin = process_sine_or_cosine(sin_lo_v, sin_hi_v)
    s30 = A_sin
    s2d = X_sin
    # $1C69 LDA $003D -> $0074 ; LDA $003E -> sin_cos_lookup (horizontal)
    h_sin_lo, h_cos_lo, h_sin_hi, h_cos_hi = sin_cos_lookup(h_angle, h_frac)
    # set_vector for Y=1 (cos h) X=2 -> vector_y ; Y=0 (sin h) X=0 -> vector_x
    # set_vector multiplies cos(v) [$0032/$0033] by sin/cos(h) [$0C00+Y]:
    #   multiply_double_by_double(x_lo=h_sincos_low, x_hi=h_sincos_high,
    #                             y_lo=$006A=$0032=cos(v)_low, y_hi=$006B=$0033=cos(v)_high)
    # Y=1 -> uses cosine(h); store to vector_y ($2E/$31) [X=2]
    vy_hi, vy_lo = multiply_double_by_double(h_cos_lo, h_cos_hi, s32, s33)
    # Y=0 -> uses sine(h); store to vector_x ($2C/$2F) [X=0]
    vx_hi, vx_lo = multiply_double_by_double(h_sin_lo, h_sin_hi, s32, s33)
    # set_vector stores: $2F+X = result_high(A) ; $2C+X = $0074(result_low/frac)
    # vector_z ($2D/$30) = sin(v): from process sine, $0030=high, $002D=low
    # (the ROM sets vector_z directly as the v-sine result -- $30/$2D)
    vec.vx_lo = vx_lo
    vec.vx_hi = vx_hi
    vec.vz_lo = s2d
    vec.vz_hi = s30
    vec.vy_lo = vy_lo
    vec.vy_hi = vy_hi
    vec.s30 = s30
    vec.s33 = s33
    vec.s32 = s32
    return vec


# ============================================================================
# line-of-sight march -- get_object_details + add_vector + check_for_los_to_tile
# ============================================================================
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
    """calculate_tile_address_z_and_slope $1DF9 for the LOS (flat) case: returns
    (z, slope_nibble, carry_set, is_object). For an object tile (byte>=$C0) the ROM
    branches to get_tile_z_from_object -- but in the player-aim LOS for a bare tile
    target we handle only terrain here; object tiles are resolved by the caller's
    height-field semantics (object base z). Returns carry_set True if not flat."""
    b = tile_byte(state, x, y)
    if b >= 0xC0:
        return None, None, None, True, b
    slope = b & 0x0F
    z = (b >> 4) & 0x0F
    carry_set = slope >= 1  # CPY #$1 -> carry set if slope!=0
    return z, slope, carry_set, False, b


def _get_min_xy_fraction(vec):
    """get_minimum_x_or_y_fraction_from_tile_centre $1EAF, literal $1EAF..$1ECB.
    For each of x ($0037=px_sub) and y ($0039=py_sub) compute (sub-$80) and, if the
    8-bit result has bit7 set, EOR #$FF (so sub<$80 -> 127-sub, sub>=$80 -> sub-128;
    note the off-by-one vs true abs at the low side -- this is the exact 6502 form).
    $1EC3 CMP $0074 ; BCS skip_minimum keeps A=ay when ay>=t74, else LDA $0074 (t74).
    Result is the value left in $0074 (returned)."""
    ax = (vec.px_sub - 0x80) & 0xFF  # $1EAF LDA $0037 ; SEC ; SBC #$80
    if ax & 0x80:  # $1EB4 BPL skip ; $1EB6 EOR #$ff
        ax ^= 0xFF
    t74 = ax & 0xFF  # $1EB8 STA $0074
    ay = (vec.py_sub - 0x80) & 0xFF  # $1EBA LDA $0039 ; SEC ; SBC #$80
    if ay & 0x80:  # $1EBF BPL skip ; $1EC1 EOR #$ff
        ay ^= 0xFF
    if ay >= (t74 & 0xFF):  # $1EC3 CMP $0074 ; $1EC5 BCS skip_min
        t74 = ay  # keep A=ay
    # else $1EC7 LDA $0074 -> t74 unchanged
    return t74 & 0xFF


def _get_tile_z_from_object(vec, state, raw, s60, s79, c0c, c67, c56, cdd, c58):
    """get_tile_z_from_object $1E3F + get_tile_z_for_line_of_sight $1E0E +
    get_boulder_or_tree_z_for_line_of_sight $1E48 + is_tree $1E69 +
    get_height_of_lowest_object $1EA4 -- the OBJECT-tile branch of
    calculate_tile_address_z_and_slope $1DF9 ($1E00 BCS get_tile_z_from_object).

    `raw` is the tile byte (>=$C0) on first entry, or an objects_flags value (>=$40)
    on a recursive entry (a stacked object). The zero-page state ($0060,$0079,$000C,
    $0C67,$0C56,$0CDD) is threaded in/out so check_flat_tile sees the same surface
    the ROM does.

    Returns (z, s79, c0c, c67, c56, cdd, s60). ALL object exits leave carry CLEAR
    (get_height_of_lowest_object falls through with carry clear from CMP #$40<; the
    boulder/platform RTS do CLC) so the caller always takes the check_flat_tile path.
    """
    for _ in range(80):  # bound the stack walk
        Y = raw & 0x3F  # $1E3F AND #$3F ; TAY
        if not (s60 & 0x80):  # $1E42 BIT $0060 ; $1E44 BPL ghol
            return _get_height_of_lowest_object(
                vec, state, Y, s60, s79, c0c, c67, c56, cdd, c58
            )
        # $1E46 BMI get_tile_z_for_line_of_sight (always) -- $1E0E:
        if Y == (c58 & 0xFF):  # $1E0E CPY $0C58
            c56 = (c56 >> 1) | 0x80  # $1E13 ROR $0C56 (top bit set)
        otype = state.obj_type[Y] & 0xFF  # $1E16 LDA objects_type,Y
        if otype == 3 or otype == 2:  # $1E19/$1E1D boulder|tree -> $1E48
            tag, *rest = _boulder_or_tree_z(
                vec, state, Y, s60, s79, c0c, c67, c56, cdd, c58
            )
            if tag == "rts":  # near-centre boulder: RTS with z
                return tuple(rest)
            # tag == "cont": skip_targeting_object continuation
            s60, s79, c0c, c67, c56, cdd, raw, done, z = rest
            if done:
                return (z, s79, c0c, c67, c56, cdd, s60)
            continue  # recurse down the stack
        if otype != 6:  # $1E23 BNE ghol (robot/sentry/enemy)
            return _get_height_of_lowest_object(
                vec, state, Y, s60, s79, c0c, c67, c56, cdd, c58
            )
        # ---- platform (type 6) $1E25 ----
        frac = _get_min_xy_fraction(vec)  # $1E25 JSR ; A in $0074
        if frac >= 0x64:  # $1E28 CMP #$64 ; $1E2A BCS skip
            s60, s79, c0c, c67, c56, cdd, raw, done, z = _skip_targeting_object(
                state, Y, s60, s79, c0c, c67, c56, cdd
            )
            if done:
                return (z, s79, c0c, c67, c56, cdd, s60)
            continue
        c0c = 0x10  # $1E2C LDA #$10 ; STA $000C
        zf = state.obj_z_frac[Y] & 0xFF
        t = zf + 0x20  # $1E30 LDA z_frac ; CLC ; ADC #$20
        s79 = t & 0xFF  # $1E36 STA $0079
        carry = 1 if t > 0xFF else 0
        z = (state.obj_z_height[Y] + carry) & 0xFF  # $1E38 LDA z_height ; ADC #$0
        return (z, s79, c0c, c67, c56, cdd, s60)  # $1E3D CLC ; RTS
    Y = raw & 0x3F  # safety: corrupt/deep stack
    return (state.obj_z_height[Y] & 0xFF, s79, c0c, c67, c56, cdd, s60)


def _boulder_or_tree_z(vec, state, Y, s60, s79, c0c, c67, c56, cdd, c58):
    """get_boulder_or_tree_z_for_line_of_sight $1E48 + is_tree $1E69. Returns a tagged
    tuple: ("rts", z, s79, c0c, c67, c56, cdd, s60) when a near-centre BOULDER RTSes
    ($1E68); else ("cont", s60, s79, c0c, c67, c56, cdd, raw, done, z) describing the
    skip_targeting_object / get_height_of_lowest_object continuation."""
    frac = _get_min_xy_fraction(vec)  # $1E48 JSR ; $0074
    if frac >= 0x40:  # $1E4B CMP #$40 ; $1E4D BCS skip
        return ("cont",) + _skip_targeting_object(
            state, Y, s60, s79, c0c, c67, c56, cdd
        )
    otype = state.obj_type[Y] & 0xFF  # $1E4F LDA objects_type,Y
    if otype == 2:  # $1E52 CMP #$2 ; $1E54 BEQ is_tree
        s79, c0c, c67, c56, cdd = _is_tree(vec, state, Y, s79, c0c, c67, c56, cdd)
        return ("cont",) + _skip_targeting_object(
            state, Y, s60, s79, c0c, c67, c56, cdd
        )
    # ---- boulder near-centre $1E56 ----
    c67 = (c67 >> 1) | 0x80  # $1E56 SEC ; $1E57 ROR $0C67
    zf = state.obj_z_frac[Y] & 0xFF
    t = zf - 0x60  # $1E5A LDA ; SEC ; SBC #$60
    s79 = t & 0xFF  # $1E60 STA $0079
    borrow = 1 if t < 0 else 0
    z = (state.obj_z_height[Y] - 0 - borrow) & 0xFF  # $1E62 LDA ; SBC #$0
    return ("rts", z, s79, c0c, c67, c56, cdd, s60)  # $1E67 CLC ; RTS


def _is_tree(vec, state, Y, s79, c0c, c67, c56, cdd):
    """is_tree $1E69: the enemy-can-see-a-tree marker ($0CDD). Does NOT change the
    surface height used by the LOS comparison (it works in $0075, not $0079), so for
    the (tx,ty,los) verdict it only (maybe) sets $0CDD. Ported faithfully."""
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
        return s79, c0c, c67, c56, cdd
    c = a & 1  # $1E83 LSR A
    a >>= 1
    s75 = ((s75 >> 1) | (c << 7)) & 0xFF  # $1E84 ROR $0075 (C from LSR A)
    c = a & 1  # $1E86 LSR A
    a >>= 1
    if a != 0:  # $1E87 BNE skip_targeting_object
        return s79, c0c, c67, c56, cdd
    a = ((s75 >> 1) | (c << 7)) & 0xFF  # $1E89 LDA $0075 ; $1E8B ROR A (C from $1E86)
    if a < _get_min_xy_fraction(vec):  # $1E8C CMP $0074 ; $1E8E BCC skip
        return s79, c0c, c67, c56, cdd
    if c56 & 0x80:  # $1E90 BIT $0C56 ; $1E93 BMI skip
        return s79, c0c, c67, c56, cdd
    cdd = (cdd >> 1) | 0x80  # $1E95 SEC ; $1E96 ROR $0CDD
    return s79, c0c, c67, c56, cdd


def _skip_targeting_object(state, Y, s60, s79, c0c, c67, c56, cdd):
    """skip_targeting_object $1E99: tree -> straight to get_height_of_lowest_object;
    else set $0060=$C0 (bit7 keeps doing get_tile_z_for_line_of_sight, bit6 marks an
    object checked) then fall into get_height_of_lowest_object. Returns the loop
    continuation tuple (s60,s79,c0c,c67,c56,cdd,raw,done,z) where on `done` the (z,...)
    is the final return, else `raw` continues the stack walk."""
    otype = state.obj_type[Y] & 0xFF  # $1E99 LDA objects_type,Y
    if otype != 2:  # $1E9C CMP #$2 ; $1E9E BEQ ghol (tree)
        s60 = 0xC0  # $1EA0 LDA #$c0 ; $1EA2 STA $0060
    # fall into get_height_of_lowest_object $1EA4
    flags = state.obj_flags[Y] & 0xFF  # $1EA4 LDA objects_flags,Y
    if flags >= 0x40:  # $1EA7 CMP #$40 ; $1EA9 BCS gtzfo
        # recurse: raw = flags (>=$40), continue the for-loop in caller
        return (s60, s79, c0c, c67, c56, cdd, flags, False, 0)
    z = state.obj_z_height[Y] & 0xFF  # $1EAB LDA objects_z_height,Y ; RTS
    return (s60, s79, c0c, c67, c56, cdd, 0, True, z)


def _get_height_of_lowest_object(vec, state, Y, s60, s79, c0c, c67, c56, cdd, c58):
    """get_height_of_lowest_object $1EA4 entered directly ($1E44/$1E23/$1E9E): walk
    down the flags chain to the bottom object's z_height. A stacked object (flags>=
    $40) re-enters get_tile_z_from_object $1E3F (recursion), so a boulder under a
    synthoid still gets the near-centre treatment."""
    for _ in range(80):
        flags = state.obj_flags[Y] & 0xFF  # $1EA4 LDA objects_flags,Y
        if flags >= 0x40:  # $1EA7 CMP #$40 ; $1EA9 BCS gtzfo
            return _get_tile_z_from_object(
                vec, state, flags, s60, s79, c0c, c67, c56, cdd, c58
            )
        z = state.obj_z_height[Y] & 0xFF  # $1EAB ; RTS (carry clear)
        return (z, s79, c0c, c67, c56, cdd, s60)
    return (state.obj_z_height[Y] & 0xFF, s79, c0c, c67, c56, cdd, s60)


def check_for_line_of_sight_to_tile(
    vec, state, slot, do_los_checks=0x00, eye_z=None, max_steps=20000
):
    """$1CDD: march the ray; return (tx, ty, los_ok). los_ok True == carry clear.

    Dispatches to the numba fast-march (:func:`_march_jit`) when numba is present,
    else the reference pure-Python march (:func:`_march_python`).  The two are
    bit-for-bit identical (see ``tests/test_los_jit.py``); the JIT path is the hot
    one every LOS sweep runs.

    do_los_checks = the $0C6E byte (its top bit cleared by the player path at $1B40);
    we pass 0 (top bit clear), matching handle_player_actions for the player aim.

    max_steps bounds the march. The ROM exits naturally at the board edge; the
    default 20000 reproduces that. PLANNING callers may pass a small cap so that
    near-horizontal "miss" rays (which march thousands of sub-steps to the edge)
    return no-LOS cheaply -- a sound approximation for nearby down-looking aims.
    """
    if _HAVE_JIT:
        return _march_jit(vec, state, slot, do_los_checks, eye_z, max_steps)
    return _march_python(vec, state, slot, do_los_checks, eye_z, max_steps)


def _march_python(vec, state, slot, do_los_checks=0x00, eye_z=None, max_steps=20000):
    """Reference pure-Python march (the bit-exact 6502 transcription)."""
    _get_object_details(vec, state, slot, eye_z=eye_z)
    # $1CDF LSR $0C56 ; $1CE2 LSR $0CDD -- clear top bits (targeted-object trackers).
    # We seed them from live memory and clear bit7 exactly as the ROM does.
    c56 = (state.mem[0x0C56] >> 1) & 0xFF
    cdd = (state.mem[0x0CDD] >> 1) & 0xFF
    # $0C56/$0CDD live in memory: seed them (LSR) back so a caller reading them
    # after the march (the enemy visibility plumbing) sees the ROM's value on the
    # edge-return paths too.  They only change again in the object branch below.
    # The read-only player-aim path (bytes-backed state) never reads them back.
    writable = not isinstance(state.mem, bytes)
    if writable:
        state.mem[0x0C56] = c56
        state.mem[0x0CDD] = cdd
    c58 = state.mem[0x0C58] & 0xFF  # $0C58 targeted object slot ($FF = none)
    c67 = 0  # $0C67 boulder-consider flag
    # $0C6E as the caller left it: the player path LSRs bit7 clear ($1B40), while
    # the enemy's robot upper-point probe sets it ($18DD is_robot) so bit7 waives
    # the looking-up rejection below.  Only bit7 is read (at the $1D26 check).
    c6e = do_los_checks & 0xFF
    tx = ty = 0
    # bound the march like the ROM's natural board-edge exit; an off-board ray
    # eventually trips the $1F edge test. Cap to avoid an infinite near-horizontal
    # ray (treated as no-LOS, walked off board).
    for _step in range(max_steps):
        _add_vector(vec)
        # $1CEB LDA $003A -> $0024 ; CMP #$1f ; BCS leave_with_carry_set
        tx = vec.px_whole
        if tx >= 0x1F:
            return tx, ty, False
        ty = vec.py_whole  # $003C -> $0026
        if ty >= 0x1F:
            return tx, ty, False
        # $1CFB LDA #$80 ; STA $0060 ; STA $000C ; $1D01 LDA #0 ; STA $0079 ;
        # $1D05 STA $0C67 (clear boulder-consider flag).
        s60 = 0x80
        c0c_var = 0x80  # $000C tolerance (object path may lower it)
        s79 = 0
        c67 = 0
        z, slope, carry_set, is_obj, raw = _calc_tile_z_and_slope(state, tx, ty)
        if is_obj:
            # $1E00 BCS get_tile_z_from_object: faithfully walk the object stack. This
            # sets z (surface high byte), $0079 (surface fraction), $000C (tolerance),
            # $0C67 (boulder targetable) and $0C56/$0CDD trackers; carry is always
            # clear on return -> check_flat_tile path. (calculate_tile_address_z_and_
            # slope $1DF9 -> $1E3F.)
            z, s79, c0c_var, c67, c56, cdd, s60 = _get_tile_z_from_object(
                vec, state, raw, s60, s79, c0c_var, c67, c56, cdd, c58
            )
            # $1E13 ROR $0C56 / the $0CDD tree tracker are memory writes in the ROM;
            # persist them so the caller's post-march plumbing reads the right byte.
            if writable:
                state.mem[0x0C56] = c56
                state.mem[0x0CDD] = cdd
            carry_set = False
            slope = 0
        if not carry_set:
            # check_flat_tile $1D0D
            # X=A(=z) ; A=$0079(=s79) ; SEC SBC $0038(pz_sub) -> $0079
            tX = z & 0xFF
            t = s79 - vec.pz_sub
            s79 = t & 0xFF
            borrow = 1 if t < 0 else 0
            # TXA ; SBC $003B(pz_whole) ; BMI loop (tile below) ; BNE leave_set (above)
            # BMI tests bit7 of the 8-BIT result (the 6502 N flag), NOT the true sign:
            # e.g. 4 - 132 - 1 = -129 -> &0xFF = 0x7F (bit7 clear) -> NOT BMI.
            d = (tX - vec.pz_whole - borrow) & 0xFF
            if d & 0x80:
                # BMI: 8-bit result negative -> tile below position, keep marching
                continue
            if d != 0:
                # BNE leave_with_carry_set: tile above position
                return tx, ty, False
            # equal high byte: $1D1C LDA $0079 ; CMP $000C; BCS leave_set
            # ($000C defaults to $80 but the object path lowers it to $10 for a
            # near-centre platform -- a tight vertical tolerance.)
            if (s79 & 0xFF) >= c0c_var:
                return tx, ty, False
            # $1D22 BIT $0060 ; $1D24 BVS leave_set -- $0060 bit6 ($40) is set if a
            # SLOPE or an OBJECT has been checked (partial-obscure rejection). Note
            # this is the LOCAL $0060 (set $C0 by the object path / $40 by a slope),
            # NOT $0C6E.
            if s60 & 0x40:
                return tx, ty, False
            # $1D26 LDA $0C6E ; $1D29 ORA $0C67 ; $1D2C BMI skip_angle_check
            # ($0C67 top bit = a boulder on this tile is TARGETABLE -> SKIP the
            # looking-up rejection, so a centre-aimed boulder above the eye is
            # visible/buildable.)
            if (c6e | c67) & 0x80:
                pass  # skip angle check
            else:
                # LDA $0030 (vector_z high = vec.s30) ; BPL leave_set (looking up)
                if not (vec.s30 & 0x80):
                    return tx, ty, False
            # skip_angle_check $1D32: same tile as observer? keep marching, else clear
            ox = state.obj_x[slot]
            oy = state.obj_y[slot]
            if (tx & 0xFF) == (ox & 0xFF) and (ty & 0xFF) == (oy & 0xFF):
                continue  # same tile as observer -> keep going
            return tx, ty, True  # leave_with_carry_clear (LOS!)
        else:
            # check_sloping_tile $1D46
            res = _check_sloping_tile(vec, state, tx, ty, z, slope)
            if res == "loop":
                continue
            if res == "blocked":
                return tx, ty, False
            # res == "clear" doesn't happen from slope (it only loops or blocks)
            return tx, ty, False
    return tx, ty, False


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
    # $1E13 ROR $0C56 / the $0CDD tree tracker are memory writes in the ROM; persist
    # the marched-out values so the caller's post-march plumbing reads the right byte
    # (only the final value is observable, so one write-back is bit-exact).
    if writable:
        state.mem[0x0C56] = c56 & 0xFF
        state.mem[0x0CDD] = cdd & 0xFF
    return tx, ty, status == los_jit.LOS_CLEAR


def _check_sloping_tile(vec, state, x, y, z00, slope):
    """check_sloping_tile $1D46. Returns 'loop' (vector above the slope, keep
    marching) or 'blocked' (vector below the slope -> tile hit, no LOS).

    The four corner heights:
      $0073=$0077 = z at (x,y)        [the calculate_tile_address_z_and_slope value
                                       that produced the slope; passed in as z00]
      $0076 = z at (x+1, y)
      $0075 = z at (x+1, y+1)
      $0074 = z at (x,   y+1)
    """
    p73 = z00 & 0xFF
    _p77 = z00 & 0xFF
    # INC $0024 -> z at (x+1,y) -> $0076
    z76 = _slope_corner_z(state, x + 1, y)
    p76 = z76
    # INC $0026 -> z at (x+1,y+1) -> $0075
    z75 = _slope_corner_z(state, x + 1, y + 1)
    p75 = z75
    # DEC $0024 -> z at (x,y+1) -> $0074
    z74 = _slope_corner_z(state, x, y + 1)
    p74 = z74
    # DEC $0026 ; calculate_tile_address ; read slope nibble
    nib = tile_byte(state, x, y) & 0x0F
    # CMP #$4 BEQ ... CMP #$c BNE tile_is_corner_or_quad
    if nib == 0x04 or nib == 0x0C:
        # tile_has_one_flat_and_one_sloping_edge: compare $003B against all 4 corners
        b = vec.pz_whole & 0xFF
        for corner in (p73, p74, p75, p76):
            if b >= corner:  # BCS to_loop
                return "loop"
        return "blocked"  # vector below all four -> hit
    # tile_is_corner_or_quadrilateral $1D8A
    return _slope_corner_or_quad(vec, state, nib, p73, p74, p75, p76)


def _slope_corner_z(state, x, y):
    """The corner-height read inside check_sloping_tile: calculate_tile_address_z_
    and_slope returns A = tile z (high nibble) for a flat read; for object tiles it
    resolves the object base z. Here it is used purely as a corner height."""
    z, _slope, _carry_set, is_obj, raw = _calc_tile_z_and_slope(
        state, x & 0xFF, y & 0xFF
    )
    if is_obj:
        o = raw & 0x3F
        for _ in range(64):
            if state.obj_flags[o] < 0x40:
                break
            o = state.obj_flags[o] & 0x3F
        return state.obj_z_height[o] & 0xFF
    return z & 0xFF


def _slope_corner_or_quad(vec, state, nib, p73, p74, p75, p76):
    """$1D8A-$1DEE, literal instruction-faithful port (corner/quadrilateral slope).
    The 6502 control flow here is bit-intricate (the $1D8B BCC lands on $1D9C, an
    EXTRA LSR before use_corner_for_slope), so we transcribe it opcode-by-opcode with
    explicit A and the carry flag. Returns 'loop' (ray above slope -> keep marching)
    or 'blocked' (ray below slope -> tile hit)."""
    # zero-page square: $73=p73, $74=p74, $75=p75, $76=p76, $77=p73 (first==last)
    corners = [p73 & 0xFF, p74 & 0xFF, p75 & 0xFF, p76 & 0xFF, p73 & 0xFF]  # $73..$77

    A = nib & 0xFF
    # $1D8A LSR A
    C = A & 1
    A >>= 1
    if C == 0:
        # $1D8B BCC tile_is_corner_type_two -> target is $1D9C (an extra LSR)
        # $1D9C LSR A
        C = A & 1
        A >>= 1
        # falls into use_corner_for_slope $1D9D
        s78 = A  # STA $0078
        # $1D9F LSR A
        C = A & 1
        A >>= 1
        # $1DA0 LDA $0037 (px_sub)
        A = vec.px_sub & 0xFF
        if C == 0:
            pass  # $1DA2 BCC skip_inversion
        else:
            A ^= 0xFF  # $1DA4 EOR #$ff
        # $1DA6 CMP $0039 (py_sub): carry = A >= py_sub
        C = 1 if A >= (vec.py_sub & 0xFF) else 0
        # $1DA8 LDA $0078 ; $1DAA ROL A ; $1DAB TAY ; $1DAC LDA edges,Y
        A = s78
        A = ((A << 1) | C) & 0xFF  # ROL A
        Y = A
        edges = [0x00, 0x03, 0x01, 0x00, 0x01, 0x02, 0x02, 0x03]  # $1DF1-$1DF8
        A = edges[Y] if Y < len(edges) else 0
    else:
        # $1D8D LSR A
        C = A & 1
        A >>= 1
        if C == 1:
            # $1D8E BCS tile_is_corner_type_two -> $1D95 (NOTE: no extra LSR here)
            # $1D95 ADC #$1 : A = A + 1 + C (C is the carry from $1D8D LSR, =1)
            A = (A + 1 + C) & 0xFF
            A &= 0x03  # $1D97 AND #$3
            # $1D99 JMP use_corner_for_slope $1D9D
            s78 = A  # $1D9D STA $0078
            C = A & 1
            A >>= 1  # $1D9F LSR A
            A = vec.px_sub & 0xFF  # $1DA0 LDA $0037
            if C == 0:
                pass  # $1DA2 BCC skip
            else:
                A ^= 0xFF  # $1DA4 EOR #$ff
            C = 1 if A >= (vec.py_sub & 0xFF) else 0  # $1DA6 CMP $0039
            A = ((s78 << 1) | C) & 0xFF  # $1DA8 LDA $0078 ; $1DAA ROL A
            Y = A
            edges = [0x00, 0x03, 0x01, 0x00, 0x01, 0x02, 0x02, 0x03]
            A = edges[Y] if Y < len(edges) else 0
        else:
            # $1D90 AND #$1 ; $1D92 JMP use_edge_for_slope
            A = A & 1

    # use_edge_for_slope $1DAF: TAX ; LSR A ; LDY $0037 ; BCS use_x ; LDY $0039
    X = A & 0xFF
    C = A & 1
    A >>= 1  # $1DB0 LSR A
    Yreg = vec.px_sub & 0xFF  # $1DB1 LDY $0037
    if C:
        pass  # $1DB3 BCS use_x_for_slope
    else:
        Yreg = vec.py_sub & 0xFF  # $1DB5 use_y_for_slope
    C = A & 1
    A >>= 1  # $1DB7 LSR A
    A = Yreg  # $1DB8 TYA
    if C == 0:
        pass  # $1DB9 BCC skip_inversion
    else:
        A ^= 0xFF  # $1DBB EOR #$ff
    s02 = A & 0xFF  # $1DBD STA $0002

    # $1DBF LDA $73,X ; STA $0078
    s78 = corners[X & 3]
    # $1DC3 LDA $74,X ; SEC ; SBC $73,X
    a = (corners[(X & 3) + 1] - corners[X & 3]) & 0x1FF
    res = a & 0xFF
    neg = bool(res & 0x80)  # PHP: N flag of the subtraction
    if not neg:
        pass  # $1DC9 BPL skip_inversion
    else:
        res = ((res ^ 0xFF) + 1) & 0xFF  # abs
    s75 = res  # $1DD0 STA $0075
    # $1DD2 LDA $0002 ; $1DD4 JSR multiply_byte_by_byte ($0D03): $0074=$0002 * $0075
    prod_h, prod_lo = _mul8(s02, s75)  # A=high, $0074=low
    # $1DD7 PLP ; $1DD8 invert_A_and_a_fraction_if_negative: invert (A:$0074) if the
    # slope subtraction (N flag we saved) was negative.
    if neg:
        prod_h, prod_lo = invert16(prod_h, prod_lo)
    # $1DDB CLC ; ADC $0078 -> $0075
    s75b = (prod_h + s78) & 0xFF
    # $1DE0 LDA $0038 ; SEC ; SBC $0074(=prod_lo) ; LDA $003B ; SBC $0075(=s75b) ; BPL
    lo = vec.pz_sub - prod_lo
    borrow = 1 if lo < 0 else 0
    hi8 = (vec.pz_whole - s75b - borrow) & 0xFF
    if hi8 & 0x80:  # BPL false -> below slope -> hit
        return "blocked"
    return "loop"


# ============================================================================
# public entry points
# ============================================================================
# The sights pan on a keyboard-reachable lattice: azimuth in 8-unit steps, pitch
# on a 4-unit lattice inside the pan clamp [$CD..$35] ($10FF/$1149; every body's
# v_angle == $F5 per put_object_in_tile $1F7E).  Views off this lattice cannot be
# reached by the keyboard.
AZIMUTH_STEP = 8  # h ≡ 0 (mod 8): ±8 pans / u-turn EOR $80 from any body's facing
PITCH_BAND = [
    v & 0xFF for v in list(range(0xCD, 0x100, 4)) + list(range(0x01, 0x36, 4))
]
SIGHTS_CX = 0x50  # centred sights cursor ($1356); cx range midpoint 80
SIGHTS_CY = 0x5F  # 95; cy range midpoint

# The REAL keyboard aim lattice (the buildability oracle). The ROM keyboard pitches/pans
# by MOVING the sights cursor ONE PIXEL AT A TIME (cursor-move $9965/$9994), each 1px step
# a distinct ray sub-angle ($1C10) -- so the lattice samples the cursor at 1px resolution.
# The cursor move routines $9965 (x) / $9994 (y) step by +/-1 pixel, clamped to
# cx $10..$8F (16..143) and cy $20..$9F (32..159) ($9965/$9994 range checks); the ROM
# centres are cx=$50=80, cy=$5F=95.  prepare_vector_from_player_sights $1C10 then makes
# EACH 1px cursor step a DISTINCT ray sub-angle: h_frac=(cx&7)<<5 / h_int+=cx>>3, and
# v_frac=((cy-5)&0xF)<<4 / v_int+=(cy-5)>>4.  So the faithful oracle enumerates the cursor
# at 1px resolution, NOT the old 9px notch grid (which false-negatived far/adjacent tiles
# the human actually builds on -- e.g. ls0 build (8,24) lands at cy=115).
#
# PERF (bit-equivalence proof, tests/test_landable.py::test_window_equals_full_1px_cursor):
# a 64px-wide, step-1 cursor WINDOW per axis is BIT-EQUIVALENT to the full 1px range.
# Body-h steps by AZIMUTH_STEP=8 and cx>>3 over 64px spans 8 consecutive h-int offsets,
# so (h_int, h_frac) is tiled identically to the full cx range (h is cyclic -- position-
# independent).  Body-v steps by 4 (PITCH_BAND) and (cy-5)>>4 over 64px spans the 4 gap
# offsets, with (cy-5)&0xF covering all 16 v-fracs, so the full-band sweep's ray SET (and
# thus every marched tile + tile-centre fraction) is identical to a full cx[16,143] x
# cy[32,159] 1px sweep -- verified exactly (0 diff) on the recorded human-win states.
# The window is centred on the ROM centres 80/95 so they sit inside it.
KBD_V_ANGLE = 0xF5
_CURSOR_CX_LO, _CURSOR_CX_HI = 16, 143  # ROM cursor x clamp $10..$8F ($9965)
_CURSOR_CY_LO, _CURSOR_CY_HI = 32, 159  # ROM cursor y clamp $20..$9F ($9994)
CURSOR_CX = list(range(48, 112))  # 64px step-1 window, centre 80 (== SIGHTS_CX) inside
CURSOR_CY = list(range(63, 127))  # 64px step-1 window, centre 95 (== SIGHTS_CY) inside
# The full 1px ROM cursor range -- the ground-truth the window is proven equal to.
CURSOR_CX_FULL = list(range(_CURSOR_CX_LO, _CURSOR_CX_HI + 1))
CURSOR_CY_FULL = list(range(_CURSOR_CY_LO, _CURSOR_CY_HI + 1))

# The COMPLETE keyboard aim has THREE pitch controls: the body v_angle (over PITCH_BAND)
# AND the sights cursor cy.  Fixing v_angle at $F5 is SOUND but INCOMPLETE -- the player
# also pitches the BODY down to aim at near/below tiles (the ls335 opening (11,17)->(11,18)
# adjacent build was fired at v=225).  landable_views/_sweep_with_centres therefore sweep
# the body v_angle too.  KBD_V_ANGLE is kept FIRST in the sweep order so every tile already
# landable at v=$F5 keeps its exact (h, v, cursor) view; only tiles reachable ONLY via a
# pitched body get a pitched view.
_V_PRIORITY = [KBD_V_ANGLE] + [v for v in PITCH_BAND if v != KBD_V_ANGLE]

# The pitched-body angles sweep the SAME 1px cursor window as $F5 (the full 4-DOF brute
# sweep = the ROM via aim_target).  The whole lattice vector set is built ONCE by the numba
# batch builder (:func:`sentinel.los_jit.build_lattice`) and cached (:data:`_VEC_CACHE`);
# each state sweep is a single batched numba march.
CURSOR_CX_PITCHED = CURSOR_CX
CURSOR_CY_PITCHED = CURSOR_CY

# Precomputed lattice ray vectors, keyed by the (h, v, cx, cy) grid.  The ray vector for an
# aim depends ONLY on the aim params (prepare_vector_from_player_sights reads neither `state`
# nor `slot`), so the whole lattice is built ONCE and reused for every state/sweep -- the
# only per-state work left is the march itself (batched in los_jit.march_batch).
_VEC_CACHE = {}


def _lattice_vectors(hgrid, vgrid, cxs, cys):
    """Build (or fetch cached) the per-aim ray-vector arrays for the keyboard lattice, nested
    ``for v in vgrid: for h in hgrid: for cx: for cy`` (one uniform 1px cursor window at every
    pitch).  Returns ``(vx_lo, vx_hi, vz_lo, vz_hi, vy_lo, vy_hi, s30)`` -- the six int16
    component arrays + s30 that feed :func:`los_jit.march_batch`.  Each landing's (h, v, cx, cy)
    build view is reconstructed from the flat index by :func:`_meta_at` (no millions-of-tuples
    meta list).  The build runs in :func:`los_jit.build_lattice` (numba, prange) so the full 1px
    lattice is assembled in a fraction of a second; result cached in :data:`_VEC_CACHE`.
    """
    key = (tuple(hgrid), tuple(vgrid), tuple(cxs), tuple(cys))
    cached = _VEC_CACHE.get(key)
    if cached is not None:
        return cached
    if _HAVE_JIT:
        cached = los_jit.build_lattice(
            np.asarray(hgrid, dtype=np.int16),
            np.asarray(vgrid, dtype=np.int16),
            np.asarray(cxs, dtype=np.int16),
            np.asarray(cys, dtype=np.int16),
        )
    else:  # pragma: no cover - numba absent -> pure-Python builder
        n = len(vgrid) * len(hgrid) * len(cxs) * len(cys)
        arrs = [np.empty(n, dtype=np.int16) for _ in range(7)]
        i = 0
        for v in vgrid:
            for h in hgrid:
                for cx in cxs:
                    for cy in cys:
                        vec = prepare_vector_from_player_sights(None, h, v, cx, cy, 0)
                        vals = (
                            vec.vx_lo,
                            vec.vx_hi,
                            vec.vz_lo,
                            vec.vz_hi,
                            vec.vy_lo,
                            vec.vy_hi,
                            vec.s30,
                        )
                        for a, val in zip(arrs, vals):
                            a[i] = val
                        i += 1
        cached = tuple(arrs)
    _VEC_CACHE[key] = cached
    return cached


def _meta_at(i, hgrid, vgrid, cxs, cys):
    """Reconstruct the ``(h, v, cx, cy)`` aim for flat lattice index ``i`` (order
    ``for v: for h: for cx: for cy``) -- the inverse of :func:`los_jit.build_lattice`'s
    indexing, so no parallel meta list of millions of tuples is stored."""
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


def landable_views(state, slot=None, eye_z=None, max_steps=6000):
    """Every tile a real KEYBOARD aim can land the sights on with line of sight from the
    observer at `slot` (default player), in ONE forward sweep of the keyboard input
    lattice: body h in :data:`AZIMUTH_STEP` notches x sights cursor (cx, cy) on the 1px
    window, v_angle over :data:`_V_PRIORITY`. Returns
    ``{(tx, ty): {"h_angle", "v_angle", "cursor"}}`` -- the keystroke target that BUILDS
    on each tile (first LOS hit per tile).

    This is the planner's buildability oracle: AIM-landability, not mere geometric
    visibility.  The sweep now covers the body v_angle DOF too (over :data:`_V_PRIORITY`,
    $F5 first) so it returns every tile ANY keyboard aim lands on -- including near/below
    tiles the player pitches the body down for (the ls335 (11,18) adjacent build).  One
    batched numba march over the precomputed lattice yields the whole landable set at once.
    """
    if slot is None:
        slot = state.player
    return _landable_sweep(state, slot, eye_z, max_steps, want_centres=False)[0]


def _landable_sweep(state, slot, eye_z, max_steps, want_centres, v_primary=False):
    """Shared body of :func:`landable_views` / :func:`landable_sweep_with_centres`: sweep the
    full keyboard aim lattice (h notches x body v_angle over :data:`_V_PRIORITY` x sights
    cursor 1px window) and return ``(views, centres)`` -- first-LOS view per tile (lattice
    order) and the min tile-centre fraction per tile.  Uses the batched numba march when
    numba is present, else a per-aim :func:`aim_target` fallback (bit-identical, slower).

    ``v_primary`` restricts the body v_angle to :data:`KBD_V_ANGLE` ($F5) alone -- the
    up/level pitch of every non-regressing climb build (a foothold whose top is at or above
    the eye is never reached by pitching the body DOWN).  Dropping the pitch band is much
    cheaper (one v-plane vs the whole band) and loses only the down/adjacent tiles a climb
    discards anyway; the full band stays the default for the completeness-critical endgame
    launch gate.

    Cursor grid: the 64px cy WINDOW is bit-equivalent to the full cy range ONLY when the
    body v_angle sweeps the pitch band (body-v step 4 fills the cy-integer gaps).  With
    ``v_primary`` there is no body-v, so the cursor cy is the sole pitch control and MUST
    span its full ROM range (:data:`CURSOR_CY_FULL`) or the $F5 plane under-reports.  The cx
    window stays faithful either way (body-h step 8 fills the h-integer gaps).
    """
    hgrid = list(range(0, 256, AZIMUTH_STEP))
    vgrid = [KBD_V_ANGLE] if v_primary else _V_PRIORITY
    cys = CURSOR_CY_FULL if v_primary else CURSOR_CY
    if not _HAVE_JIT:
        return _landable_sweep_py(
            state, slot, eye_z, max_steps, hgrid, vgrid, CURSOR_CX, cys, want_centres
        )
    status, tx, ty, centre, grids = _landable_batch(
        state, slot, eye_z, max_steps, hgrid, vgrid, CURSOR_CX, cys
    )
    views = {}
    centres = {}
    for i in range(status.shape[0]):
        if status[i] != los_jit.LOS_CLEAR:
            continue
        tile = (int(tx[i]), int(ty[i]))
        if tile not in views:
            h, v, cx, cy = _meta_at(i, *grids)
            views[tile] = {"h_angle": h, "v_angle": v, "cursor": [cx, cy]}
        if want_centres:
            c = int(centre[i])
            if tile not in centres or c < centres[tile]:
                centres[tile] = c
    return views, centres


def _landable_sweep_py(
    state, slot, eye_z, max_steps, hgrid, vgrid, cxs, cys, want_centres
):
    """Numba-absent fallback for :func:`_landable_sweep`: the same lattice via per-aim
    :func:`aim_target` (bit-identical results, no batched march)."""
    views = {}
    centres = {}
    for v in vgrid:
        for h in hgrid:
            for cx in cxs:
                for cy in cys:
                    tx, ty, los, centre = aim_target(
                        state,
                        h,
                        v,
                        cx,
                        cy,
                        slot,
                        eye_z=eye_z,
                        max_steps=max_steps,
                        return_centre=True,
                    )
                    if not los:
                        continue
                    tile = (tx, ty)
                    if tile not in views:
                        views[tile] = {"h_angle": h, "v_angle": v, "cursor": [cx, cy]}
                    if want_centres and (tile not in centres or centre < centres[tile]):
                        centres[tile] = centre
    return views, centres


def landable_sweep_with_centres(
    state, slot=None, eye_z=None, max_steps=6000, v_primary=False
):
    """One forward sweep of the REAL keyboard aim lattice (h notches x 1px cursor window x
    body v_angle) returning (views, centres):
      views:   {(tx,ty): {"h_angle","v_angle","cursor"}}  first LOS landing per tile
      centres: {(tx,ty): min tile-centre fraction seen}    for the on-boulder centre gate
                                                           ($1E48 needs fraction < $40)
    The buildability oracle for the planner -- aim-landability (which tiles a real keyboard
    aim lands the sights on), not mere geometric visibility.  Sweeps the body v_angle DOF too
    (:data:`_V_PRIORITY`, $F5 first) so near/below builds are covered.

    ``v_primary=True`` restricts the sweep to :data:`KBD_V_ANGLE` -- the up/level pitch of
    every non-regressing climb foothold -- for a ~100x cheaper sweep in the hot climb loop.
    """
    if slot is None:
        slot = state.player
    return _landable_sweep(
        state, slot, eye_z, max_steps, want_centres=True, v_primary=v_primary
    )


def landable_view(state, tile, slot=None, eye_z=None, max_steps=6000, v_band=False):
    """The build view for a SINGLE `tile`, or None if no keyboard aim lands the sights on
    it with line of sight.  Returns ``{"h_angle","v_angle","cursor"}``.

    Targeted + cheap-first: the CHEAP primary ($F5) plane is swept first (one batched numba
    march, ~131k rays) and, on a hit, returned immediately -- most (up/level) climb builds
    land here.  Only when the tile is NOT on the $F5 plane AND ``v_band`` is set does it fall
    to the full pitch band (the player pitches the body DOWN to aim at near/below tiles -- the
    ls335 v=225 (11,18) build, the endgame down-look).  So a per-tile query costs one cheap
    march in the common case and is bounded by the full band otherwise -- no ~3.5M pure-Python
    probe scan.  Bit-identical to membership in :func:`landable_views`.
    """
    if slot is None:
        slot = state.player
    key = (tile[0], tile[1])
    if _HAVE_JIT:
        views, _ = _landable_sweep(
            state, slot, eye_z, max_steps, want_centres=False, v_primary=True
        )
        view = views.get(key)
        if view is not None or not v_band:
            return view
        views, _ = _landable_sweep(
            state, slot, eye_z, max_steps, want_centres=False, v_primary=False
        )
        return views.get(key)
    return _landable_view_py(state, key, slot, eye_z, max_steps, v_band)


def _landable_view_py(state, key, slot, eye_z, max_steps, v_band):
    """Numba-absent fallback for :func:`landable_view`: pure-Python probes ordered from the
    analytic bearing / cursor centre outward, short-circuiting on the first LOS landing.
    """
    tx0, ty0 = key
    ex, ey = state.obj_x[slot], state.obj_y[slot]
    h0 = _bearing_notch(ex, ey, tx0, ty0)
    hgrid = sorted(range(0, 256, AZIMUTH_STEP), key=lambda h: _angle_dist(h0, h))
    cxs = sorted(CURSOR_CX, key=lambda c: abs(c - SIGHTS_CX))
    # $F5-plane probes have no body-v gap-fill -> the cursor cy must span the full ROM range;
    # the pitched band (v_band) refills the cy gaps, so the 64px window suffices there.
    cy_src = CURSOR_CY if v_band else CURSOR_CY_FULL
    cys = sorted(cy_src, key=lambda c: abs(c - SIGHTS_CY))
    vgrid = (
        sorted(PITCH_BAND, key=lambda v: _angle_dist(KBD_V_ANGLE, v))
        if v_band
        else [KBD_V_ANGLE]
    )
    for v in vgrid:
        for h in hgrid:
            for cx in cxs:
                for cy in cys:
                    tx, ty, los = aim_target(
                        state, h, v, cx, cy, slot, eye_z=eye_z, max_steps=max_steps
                    )
                    if los and tx == tx0 and ty == ty0:
                        return {"h_angle": h, "v_angle": v, "cursor": [cx, cy]}
    return None


def _bearing_notch(ex, ey, tx, ty):
    """The body-angle notch (0..255, multiple of AZIMUTH_STEP) nearest the bearing from
    (ex,ey) to (tx,ty); 0 when the target is the observer's own tile."""
    dx, dy = tx - ex, ty - ey
    if dx == 0 and dy == 0:
        return 0
    ang = int(round(math.atan2(dy, dx) * 128.0 / math.pi)) & 0xFF
    return (ang + AZIMUTH_STEP // 2) // AZIMUTH_STEP * AZIMUTH_STEP & 0xFF


def _angle_dist(a, b):
    """Shortest angular distance (0..128) between two 8-bit angles."""
    d = (a - b) & 0xFF
    return min(d, 256 - d)


def _prepare_vector(state, h_angle, v_angle, cur_x, cur_y, player_slot):
    """The action-time ray vector for one aim.  When numba is present this is the njit
    :func:`sentinel.los_jit._prep_vec` (bit-identical to, but ~16x cheaper than, the pure
    Python :func:`prepare_vector_from_player_sights`); otherwise the pure-Python path.  The
    player aim reads neither `state` nor `player_slot` (obj_h/obj_v are the aim params), so
    both accept them only for signature parity."""
    if not _HAVE_JIT:
        return prepare_vector_from_player_sights(
            state, h_angle, v_angle, cur_x, cur_y, player_slot
        )
    vx_lo, vx_hi, vz_lo, vz_hi, vy_lo, vy_hi, s30 = los_jit._prep_vec(
        h_angle & 0xFF, v_angle & 0xFF, cur_x & 0xFF, cur_y & 0xFF
    )
    vec = Vector()
    vec.vx_lo = int(vx_lo)
    vec.vx_hi = int(vx_hi)
    vec.vz_lo = int(vz_lo)
    vec.vz_hi = int(vz_hi)
    vec.vy_lo = int(vy_lo)
    vec.vy_hi = int(vy_hi)
    vec.s30 = int(s30)
    return vec


def aim_target(
    state,
    h_angle,
    v_angle,
    cur_x,
    cur_y,
    player_slot,
    eye_z=None,
    max_steps=20000,
    return_centre=False,
):
    """Native port of the action-time aim (handle_player_actions $1B40-$1B46):
    prepare_vector_from_player_sights $1C10 then check_for_line_of_sight_to_tile
    $1CDD. Returns (tx, ty, los) where los True == ROM carry clear (visible).

    The ray vector is built by the numba :func:`sentinel.los_jit._prep_vec` when numba is
    present -- bit-for-bit identical to :func:`prepare_vector_from_player_sights` (locked by
    tests/test_landable.py::test_prep_vec_matches_python) but ~16x cheaper per probe, so the
    1px single-tile short-circuit stays fast."""
    vec = _prepare_vector(state, h_angle, v_angle, cur_x, cur_y, player_slot)
    # do_line_of_sight_checks $0C6E: the player aim path clears its top bit ($1B40
    # LSR $0C6E). Read it from live memory and mask, matching the ROM exactly.
    do_los = state.mem[0x0C6E] & 0x7F
    tx, ty, los = check_for_line_of_sight_to_tile(
        vec, state, player_slot, do_los_checks=do_los, eye_z=eye_z, max_steps=max_steps
    )
    if return_centre:
        # get_minimum_x_or_y_fraction_from_tile_centre $1EAF (the EXACT 6502 form, not
        # plain abs): < $40 => the boulder on this tile is targetable ($1E4B) and the
        # looking-up rejection is skipped -> you can build on it. Uses the FINAL marched
        # px_sub/py_sub ($0037/$0039) -- the same inputs the ROM's $1EAF reads when the
        # ray reaches the object tile. This lets the NATIVE planner compute a centre-
        # aimed view without any emulation.
        centre = _get_min_xy_fraction(vec)
        return tx, ty, los, centre
    return tx, ty, los
