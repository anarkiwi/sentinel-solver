"""The from-scratch landscape generator -- a pure-Python port of the game's own
terrain + object placement, so a board can be built from a landscape number with
no emulator.

Sequence (play-setup $1A97):
  reset ($0C7D <- 1) -> seed ($33ED) ->
  generate_landscape ($2ACC):
     process_landscape($80)  random heights (prnd per tile)
     smooth_landscape($00)   average neighbours (2 passes, rows then columns)
     process_landscape($01)  scale by landscape_vertical_scale, clamp 1..11
     smooth_landscape($40)   level spikes (2 passes)
     set_tile_slopes         per-tile slope from the 4 corner heights
     process_landscape($02)  swap nibbles -> (height<<4) | slope
  initialise_enemies ($14FB) place the Sentinel + platform + sentries
  initialise_player_and_trees ($1450) place the player robot + trees

Heights live in the LOW nibble until the final swap; the slope is computed from
the four neighbouring corner heights (calculate_tile_slope $2C7C).
"""

from sentinel import memmap as mm
from sentinel.prng import Prng
from sentinel.state import State


# ---- tiles_table access (the ROM addressing, calculate_tile_address $2BA8) ---
def _addr(x, y):
    x &= 0xFF
    y &= 0xFF
    return ((x & 3) + 4) * 256 + (((x << 3) & 0xE0) | (y & 0x1F))


def _get(mem, x, y):
    return mem[_addr(x, y)]


def _set(mem, x, y, value):
    mem[_addr(x, y)] = value & 0xFF


# ---- math helpers (multiply_byte_by_byte $0D03, invert $1009) ----------------
def _invert16(high, low):
    neg = (-(((high & 0xFF) << 8) | (low & 0xFF))) & 0xFFFF
    return (neg >> 8) & 0xFF, neg & 0xFF


# ---- process_landscape $2B22 -------------------------------------------------
def _process_landscape(mem, mode, scale):
    """mode $80 randomise, $01 scale (clamp 1..11), $02 swap nibbles.  prnd is only
    drawn in the randomise pass; `scale` is landscape_vertical_scale."""
    prng = Prng().load(mem)
    for y in range(31, -1, -1):
        for x in range(31, -1, -1):
            if mode == 0x00:
                _set(mem, x, y, 0)
            elif mode & 0x80:  # randomise ($2B75)
                _set(mem, x, y, prng.next())
            elif mode & 0x01:  # scale ($2B4B)
                _set(mem, x, y, _scale_tile(_get(mem, x, y), scale))
            else:  # swap nibbles ($2B3A)
                b = _get(mem, x, y)
                _set(mem, x, y, ((b << 4) & 0xF0) | ((b >> 4) & 0x0F))
    prng.store(mem)


def _scale_tile(tile, scale):
    """scale_tile_height $2B4B: |tile-$80| * scale, high byte, signed, +6 floor 0,
    +1 clamp 11."""
    diff = (tile - 0x80) & 0xFF  # SEC SBC #$80
    negative = bool(diff & 0x80)  # PHP: N flag
    mag = ((diff ^ 0xFF) + 1) & 0xFF if negative else diff  # abs
    prod = (scale & 0xFF) * mag
    high, low = (prod >> 8) & 0xFF, prod & 0xFF
    if negative:  # invert_A_and_a_fraction_if_negative $1007
        high, low = _invert16(high, low)
    a = (high + 0x06) & 0xFF  # CLC ADC #$06
    if a & 0x80:  # BPL skip_floor
        a = 0
    a = (a + 0x01) & 0xFF  # CLC ADC #$01
    if a >= 0x0C:  # CMP #$0C BCC skip_ceiling
        a = 0x0B
    return a


# ---- smooth_landscape $2B83 --------------------------------------------------
def _smooth_landscape(mem, mode):
    """mode $00 average, $40 level spikes.  Two passes, each smoothing every row
    then every column."""
    for _pass in range(2):
        for y in range(31, -1, -1):
            _smooth_line(mem, mode, is_col=False, fixed=y)
        for x in range(31, -1, -1):
            _smooth_line(mem, mode, is_col=True, fixed=x)


def _smooth_line(mem, mode, is_col, fixed):
    """smooth_row_or_column $2BBC: copy 35 heights (with wraparound) into a temp
    table, average or level, then copy 32 back."""

    def coord(i):
        return (fixed, i) if is_col else (i, fixed)

    # copy_heights_to_temporary_table: X=34..0, temp[X] = tile((X & $1F))
    temp = [0] * 35
    for x in range(34, -1, -1):
        temp[x] = _get(mem, *coord(x & 0x1F))
    if mode & 0x40:  # level_spikes $2BDF
        for x in range(0x1F, -1, -1):
            z0, z1, z2 = temp[x], temp[x + 1], temp[x + 2]
            temp[x + 1] = _spike(z0, z1, z2)
    else:  # average_tile_heights $2C2C
        for x in range(0x20):
            temp[x] = (temp[x] + temp[x + 1] + temp[x + 2] + temp[x + 3]) >> 2
    # copy_heights_from_temporary_table: X=$1F..0, tile((X)) = temp[X]
    for x in range(0x1F, -1, -1):
        _set(mem, *coord(x), temp[x])


def _spike(z0, z1, z2):
    """One level_spikes step ($2BE1): decide the middle height z1 from its
    neighbours z0 (previous) and z2 (next).  Plateaus and monotone slopes are
    kept; a lone spike is clipped toward the lower of z0/z2."""
    if z1 == z2:  # keep plateau
        return z1
    if z1 > z2:  # middle_is_higher_than_last $2BFB
        if z1 <= z0:  # $2BFE/$2C00: keep plateau / downward slope
            return z1
        return z2 if z0 < z2 else z0  # $2C05/$2C08
    # z1 < z2
    if z1 >= z0:  # $2BEE/$2BF0: keep plateau / upward slope
        return z1
    return z2 if z2 < z0 else z0  # $2BF5/$2C08


def _generate_terrain(mem, landscape):
    """generate_landscape $2ACC (terrain only, up to the RTS $2B21)."""
    prng = Prng().load(mem)
    # randomise_row_or_column_tile_z_table $2ACE: 0x51 prnd draws into a scratch
    # table (kept only so the stream stays aligned).
    for _ in range(0x51):
        prng.next()
    prng.store(mem)
    if landscape == 0:  # $2ADC: fixed vertical scale for landscape 0000
        scale = 0x18
    else:
        scale = (_rnd_0_22(mem) + 0x0E) & 0xFF
    mem[mm.VERTICAL_SCALE] = scale
    _process_landscape(mem, 0x80, scale)
    _smooth_landscape(mem, 0x00)
    _process_landscape(mem, 0x01, scale)
    _smooth_landscape(mem, 0x40)
    _set_tile_slopes(mem)
    _process_landscape(mem, 0x02, scale)


def _rnd_0_22(mem):
    """get_random_number_between_0_and_22 $3451."""
    prng = Prng().load(mem)
    r = prng.next()
    prng.store(mem)
    lo3 = r & 0x07
    return (lo3 + (((r >> 2) & 0x1E) >> 1)) & 0xFF


# ---- set_tile_slopes $2AFD / calculate_tile_slope $2C7C ----------------------
def _set_tile_slopes(mem):
    for y in range(0x1E, -1, -1):
        for x in range(0x1E, -1, -1):
            slope = _tile_slope(mem, x, y)
            b = _get(mem, x, y)
            _set(mem, x, y, ((slope << 4) & 0xF0) | b)  # $2B0B ASL x4 ORA


# ---- object placement --------------------------------------------------------
_MASK_TABLE = [0xFF, 0x7F, 0x3F, 0x1F, 0x0F, 0x07, 0x03, 0x01]  # $15C4


def _put_object(mem, prng, slot, x, y, otype):
    """put_object_in_tile $1F16 with an explicit slot (no energy): stacking, tile
    byte, flags, z/zf, v_angle=$F5 and the prnd-driven random facing.  Returns
    True, or False if the tile can't hold an object ($1F38)."""
    b = _get(mem, x, y)
    if b >= mm.OBJECT_TILE:  # stacking
        below = b & 0x3F
        btype = mem[mm.OBJECTS_TYPE + below]
        if btype == mm.T_PLATFORM:
            zf = mem[mm.OBJECTS_Z_FRACTION + below]
            z = mem[mm.OBJECTS_Z_HEIGHT + below] + 1
        elif btype == mm.T_BOULDER:
            t = mem[mm.OBJECTS_Z_FRACTION + below] + 0x80
            zf = t & 0xFF
            z = mem[mm.OBJECTS_Z_HEIGHT + below] + (t >> 8)
        else:
            return False
        mem[mm.OBJECTS_FLAGS + slot] = 0x40 | below
    else:  # bare terrain
        mem[mm.OBJECTS_FLAGS + slot] = 0x00
        zf = 0xE0
        z = b >> 4
    mem[mm.OBJECTS_X + slot] = x & 0xFF
    mem[mm.OBJECTS_Y + slot] = y & 0xFF
    mem[mm.OBJECTS_Z_HEIGHT + slot] = z & 0xFF
    mem[mm.OBJECTS_Z_FRACTION + slot] = zf & 0xFF
    _set(mem, x, y, mm.OBJECT_TILE | slot)
    mem[mm.OBJECTS_V_ANGLE + slot] = 0xF5
    rot = prng.next()
    mem[mm.OBJECTS_H_ANGLE + slot] = ((rot & 0xF8) + 0x60) & 0xFF
    return True


def _create_object(mem, otype):
    """create_object_from_action $2120: highest empty slot, set its type.  No prnd."""
    for slot in range(mm.NUM_SLOTS - 1, -1, -1):
        if mem[mm.OBJECTS_FLAGS + slot] & 0x80:
            mem[mm.OBJECTS_TYPE + slot] = otype & 0xFF
            return slot
    return None


def _random_tile_coord(prng):
    """get_random_tile_coordinate $1272: prnd & $1F, reject $1F (so 0..30)."""
    while True:
        v = prng.next() & 0x1F
        if v != 0x1F:
            return v


def _put_in_random_tile_below_z(mem, prng, slot, zlimit):
    """put_object_in_random_tile_below_z $1238: place `slot` on a random flat tile
    whose height < zlimit, raising the limit every 256 tries.  Returns True/False."""
    limit = zlimit & 0xFF
    counter = 0
    while True:
        counter = (counter - 1) & 0xFF
        if counter == 0:  # every 256 tries, raise the height limit
            limit = (limit + 1) & 0xFF
            if limit >= 0x0C:
                return False
        x = _random_tile_coord(prng)
        y = _random_tile_coord(prng)
        b = _get(mem, x, y)
        if b >= mm.OBJECT_TILE:  # occupied
            continue
        if b & 0x0F:  # not flat
            continue
        if (b >> 4) >= limit:  # too high
            continue
        _put_object(mem, prng, slot, x, y, mem[mm.OBJECTS_TYPE + slot])
        return True


def _find_highest_tiles(mem):
    """find_highest_tiles_in_grid $15CC: for each of the 64 grid sections (8x8 of
    4x4 tiles), the highest flat tile.  Returns (max_h, sec_h, sec_x, sec_y) with
    heights in the byte&$F0 form."""
    sec_h = [0] * 64
    sec_x = [0] * 64
    sec_y = [0] * 64
    max_h = 0
    for s in range(64):
        bx = (s & 0x07) << 2  # $0018
        by = (s & 0x38) >> 1  # $001A
        ny = 3 if by >= 0x1C else 4
        nx0 = 3 if bx >= 0x1C else 4
        for dy in range(ny):
            y = by + dy
            for dx in range(nx0):
                x = bx + dx
                b = _get(mem, x, y)
                if b & 0x0F:  # only flat tiles
                    continue
                h = b & 0xF0
                if h < sec_h[s]:
                    continue
                sec_h[s] = h
                if h >= max_h:
                    max_h = h
                sec_x[s] = x
                sec_y[s] = y
    return max_h, sec_h, sec_x, sec_y


def _initialise_enemies(mem, prng, num):
    """initialise_enemies $14FB: place the Sentinel (+ platform) and sentries on
    the highest flat tiles of distinct grid sections.  Returns the enemy count
    actually placed and $0C06 (the height baseline for player/trees)."""
    max_h, sec_h, sec_x, sec_y = _find_highest_tiles(mem)
    z = max_h  # $0006
    placed = 0
    while placed < num:
        mem[mm.OBJECTS_TYPE + placed] = mm.T_SENTRY  # $1502 (sentinel overrides below)
        # find_grid_sections_at_given_z $159D: sections whose max height == z.
        found = [s for s in range(0x3F, -1, -1) if sec_h[s] == z]
        if not found:
            z = (z - 0x10) & 0xFF  # drop one height unit and retry
            if z == 0:
                break
            continue
        # $AD40 holds them in descending-index order.
        ad40 = found
        count = len(ad40)
        # calculate_mask $15B5: mask for prnd selection.
        a = count & 0xFF
        y = 0xFF
        while True:
            carry = (a >> 7) & 1
            a = (a << 1) & 0xFF
            y = (y + 1) & 0xFF
            if carry:
                break
        mask = _MASK_TABLE[y]
        # choose_a_random_grid_section $151B: prnd & mask, reject >= count.
        while True:
            idx = prng.next() & mask
            if idx < count:
                break
        sec = ad40[idx]
        # mark this section and its 8 neighbours used (height -> 0).
        for off in (-9, -8, -7, -1, 0, 1, 7, 8, 9):
            j = sec + off
            if 0 <= j < 64:
                sec_h[j] = 0
        x, y2 = sec_x[sec], sec_y[sec]
        if placed == 0:  # the Sentinel + its platform
            mem[mm.PLATFORM_Y] = y2 & 0xFF
            mem[mm.PLATFORM_X] = x & 0xFF
            mem[mm.OBJECTS_TYPE + 0] = mm.T_SENTINEL
            plat = _create_object(mem, mm.T_PLATFORM)
            _put_object(mem, prng, plat, x, y2, mm.T_PLATFORM)
            mem[mm.OBJECTS_H_ANGLE + plat] = 0x00
        _put_object(mem, prng, placed, x, y2, mem[mm.OBJECTS_TYPE + placed])
        _init_meanie_vars(mem, placed)
        # $1575 prnd; LSR A (bit0 -> carry); AND #$3F; ORA #$05 -> update_cooldown.
        rot = prng.next()
        mem[mm.ENEMIES_UPDATE_COOLDOWN + placed] = ((rot >> 1) & 0x3F) | 0x05
        # rotation clockwise ($14) if bit0 clear, else anticlockwise ($EC).
        mem[mm.ROTATION_SPEED_TABLE + placed] = 0x14 if (rot & 0x01) == 0 else 0xEC
        placed += 1
    mem[0x0C06] = (z >> 4) & 0xFF
    return placed, mem[0x0C06]


def _init_meanie_vars(mem, slot):
    """initialise_enemy_meanie_variables $1973."""
    mem[0x0CA0 + slot] = 0x80
    mem[0x0C90 + slot] = 0x80
    mem[0x0C98 + slot] = 0x00
    mem[0x0C80 + slot] = 0x40


def _max_enemies_second_cap(mem, prng):
    """get_maximum_number_of_enemies $3426: base = top digit of the landscape + 2,
    plus a bit-weighted 0..7 draw, clamped to 1..8 (retrying on >= 8)."""
    hi = mem[0x0CFE]
    base = ((hi >> 4) + 0x02) & 0xFF
    while True:
        r = prng.next()
        sign = (r >> 7) & 1  # ASL A -> carry saved
        a = (r << 1) & 0xFF
        if a == 0:
            count = 7
        else:
            count = 0xFF
            while True:
                carry = (a >> 7) & 1
                a = (a << 1) & 0xFF
                count = (count + 1) & 0xFF
                if carry:
                    break
        if sign:
            count ^= 0xFF
        val = (count + base) & 0xFF
        if val >= 0x08:
            continue
        return (val + 1) & 0xFF


def _initialise_player_and_trees(mem, prng, landscape, num_enemies, z_base):
    """initialise_player_and_trees $1450."""
    player = _create_object(mem, mm.T_ROBOT)
    mem[mm.PLAYER_OBJECT] = player
    mem[mm.PLAYER_ENERGY] = 0x0A
    if landscape == 0:
        _put_object(mem, prng, player, 8, 0x11, mm.T_ROBOT)
    else:
        z = 0x06 if z_base >= 0x06 else z_base
        while not _put_in_random_tile_below_z(mem, prng, player, z):
            pass
    # initialise_trees: energy budget 48 - 3*num_enemies, count 10..32 capped.
    max_energy = (0x30 - 3 * num_enemies) & 0xFF
    count = min((_rnd_0_22_prng(prng) + 0x0A) & 0xFF, max_energy)
    for _ in range(count):
        tree = _create_object(mem, mm.T_TREE)
        if tree is None:
            break
        if not _put_in_random_tile_below_z(mem, prng, tree, z_base):
            mem[mm.OBJECTS_FLAGS + tree] = 0x80  # unplaced -> leave empty
            break
    # generate_secret_code_validation_table $14AA: in the setup (preview) path it
    # calls get_random_two_digit_bcd_number for X = $AA..$80 (43 draws).  The result
    # only fills an anti-piracy table (no board effect), but it advances the PRNG that
    # play-time hyperspace/meanie draws consume, so it must be replayed.
    for _ in range(0xAA - 0x80 + 1):
        prng.next()  # each get_random_two_digit_bcd_number $339A draws one prnd


def _rnd_0_22_prng(prng):
    r = prng.next()
    return ((r & 0x07) + (((r >> 2) & 0x1E) >> 1)) & 0xFF


def seed_for(landscape_number):
    """The PRNG seed for the landscape number a PLAYER TYPES -- the canonical id.

    Decimal digits are numerically identical to hex nibbles, so the packed-BCD seed is
    the typed code read as hex: 42 -> 0x0042 = 66, 335 -> 0x0335 = 821, 0 -> 0. Pass
    typed numbers through here; only :func:`generate` takes the raw seed.
    """
    return int(f"{int(landscape_number):04d}", 16)


def generate(landscape):
    """Build landscape `landscape` from scratch -- terrain + Sentinel/platform +
    sentries + player + trees -- and return a :class:`sentinel.state.State`.
    Byte-for-byte identical to running the game's own generator.

    Input contract (seed_prnd_from_landscape_number $33ED): `landscape` is used
    exactly as the ROM's seed routine uses its X/Y registers -- its raw little-endian
    bytes seed prnd_state ($0C7B <- landscape & $FF, $0C7C <- landscape >> 8).  It is
    therefore the game's PACKED-BCD landscape number, NOT a plain decimal to be
    re-encoded: the ROM reads those same bytes back as BCD digits (`>> 4` for the
    tens/thousands digit) when it derives the enemy cap (1 + landscape DIV 10, $3403;
    top digit + 2, $3426).  The number a player enters on the keypad as "NNNN" is thus
    the integer whose hex nibbles are NNNN -- e.g. the human landscape 0042 is
    ``generate(0x0042)`` == ``generate(66)``, and 2024 is ``generate(0x2024)``.  A
    plain decimal >= 10 (``generate(42)``) seeds a different, legal-but-non-BCD board.
    This matches the oracle (``oracle.generate`` seeds X=n&$FF, Y=n>>8 identically),
    which is why the byte-exact ``golden_landscape`` fixtures validate it directly."""
    mem = bytearray(mm.MEM_SIZE)
    for slot in range(mm.NUM_SLOTS):  # reset_game_state: all object slots empty
        mem[mm.OBJECTS_FLAGS + slot] = 0x80
    mem[0x0C7D] = 1  # reset_game_state INC $0C7D
    mem[mm.PRND_STATE] = landscape & 0xFF  # seed $33ED
    mem[mm.PRND_STATE + 1] = (landscape >> 8) & 0xFF
    mem[0x0CFE] = (landscape >> 8) & 0xFF  # $33F3 STY $0CFE
    mem[0x0C52] = 0 if landscape == 0 else 1  # $3400
    mem[mm.MAX_ENEMIES] = _seed_max_enemies(landscape)  # $0C07
    _generate_terrain(mem, landscape)
    prng = Prng().load(mem)
    if landscape == 0:
        num = 1  # $1425: only the Sentinel on landscape 0000
    else:
        cap2 = _max_enemies_second_cap(mem, prng)
        num = min(cap2, mem[mm.MAX_ENEMIES])
    mem[0x0C6F] = num
    placed, z_base = _initialise_enemies(mem, prng, num)
    mem[0x0C6F] = placed
    _initialise_player_and_trees(mem, prng, landscape, placed, z_base)
    prng.store(mem)
    return State(mem)


def _seed_max_enemies(landscape):
    """The first enemy cap from seed_prnd_from_landscape_number $33FD: 1 +
    (landscape // 10) for 0..99, else 8 (a further limit applies at $1429)."""
    hi = (landscape >> 8) & 0xFF
    if hi != 0:
        return 8
    lo = landscape & 0xFF
    val = (lo >> 4) + 1
    return 8 if val >= 9 else val


def _tile_slope(mem, x, y):
    """calculate_tile_slope $2C7C: the slope nibble from the four corner heights
    a=(x,y), b=(x+1,y), c=(x+1,y+1), d=(x,y+1)."""
    a = _get(mem, x, y) & 0x0F
    b = _get(mem, x + 1, y) & 0x0F
    c = _get(mem, x + 1, y + 1) & 0x0F
    d = _get(mem, x, y + 1) & 0x0F
    if a == b:  # is_1_3_6_9_a_c_or_f
        if a == d:  # is_0_3_or_a
            if a == c:
                return 0x0
            return 0x0A if a < c else 0x03
        if c == d:  # is_1_or_9
            return 0x01 if c < b else 0x09
        if c != b:
            return 0x0C
        return 0x06 if c < d else 0x0F  # is_6_or_f
    # is_2_4_5_7_b_c_d_or_e
    if a == d:  # is_4_5_7_d_or_e
        if c == b:  # is_5_or_d
            return 0x05 if c < d else 0x0D
        if c == d:  # is_7_or_e
            return 0x0E if c < b else 0x07
        return 0x04
    if c == b:  # is_2_4_or_b
        if c != d:
            return 0x04
        return 0x02 if c >= a else 0x0B
    return 0x0C
