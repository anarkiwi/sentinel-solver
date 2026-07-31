"""Numba twins of $245B (tile visibility) and of plot_world's projection walk.

$2845 + $27D7 + $26DE + the $2A24 tile selection; :mod:`sentinel.projector` stays the
bit-exact reference and the numba-absent fallback, as :mod:`sentinel.los_jit` does for
:mod:`sentinel.los`.  The trig core is reused from :mod:`sentinel.enemies_jit`.
"""

import numpy as np
from numba import njit

from sentinel import memmap as mm, passcost
from sentinel.enemies_jit import (
    ZP_LO,
    ZP_HI,
    _calc_angle,
    _calc_hypotenuse,
    _vertical_angle,
)

# Object-array bases in the 64 KB image, inlined so the njit code needs no Python object.
_OFLAGS = mm.OBJECTS_FLAGS
_OX = mm.OBJECTS_X
_OY = mm.OBJECTS_Y
_OZHEIGHT = mm.OBJECTS_Z_HEIGHT
_OZFRAC = mm.OBJECTS_Z_FRACTION

_N = mm.N
_NUM_SLOTS = mm.NUM_SLOTS
_OBJECT_TILE = mm.OBJECT_TILE
_LAST = mm.N - 2  # $1E: last tile with all four corners on the board
_FILL_COLS = (
    18  # rendercost.TILE_COLS: tile byte, parity, four corners of xlo/xhi/ylo/yhi
)

# $2845's own branch costs, inlined as njit globals (see passcost.EXAM_*).
_EX_CALL = passcost.EXAM_CALL
_EX_HEAD = passcost.EXAM_HEAD
_EX_COL_POS = passcost.EXAM_COL_POS
_EX_COL_NEG = passcost.EXAM_COL_NEG
_EX_ROW = passcost.EXAM_ROW
_EX_ROW_POS = passcost.EXAM_ROW_POS
_EX_ROW_NEG = passcost.EXAM_ROW_NEG
_EX_ANGLE = passcost.EXAM_ANGLE
_EX_STORE = passcost.EXAM_STORE
_EX_QUAD = np.array(
    (
        passcost.EXAM_QUAD_NORTH,
        passcost.EXAM_QUAD_EAST,
        passcost.EXAM_QUAD_SOUTH,
        passcost.EXAM_QUAD_WEST,
    ),
    dtype=np.int64,
)
_EX_ADDR = passcost.EXAM_ADDR
_EX_OBJECT = passcost.EXAM_OBJECT
_EX_OBJECT_STEP = passcost.EXAM_OBJECT_STEP
_EX_OBJECT_BOTTOM = passcost.EXAM_OBJECT_BOTTOM
_EX_GROUND = passcost.EXAM_GROUND
_EX_VISIBLE = passcost.EXAM_VISIBLE
_EX_HIDDEN = passcost.EXAM_HIDDEN
_EX_VANGLE = passcost.EXAM_VANGLE
_EX_OFF_LEFT = passcost.EXAM_OFF_LEFT
_EX_ON_LEFT = passcost.EXAM_ON_LEFT
_EX_NO_FRACTION = passcost.EXAM_NO_FRACTION
_EX_FRACTION = passcost.EXAM_FRACTION + passcost.EXAM_FRACTION_OK
_EX_RIGHT = passcost.EXAM_RIGHT
_EX_OFF_RIGHT = passcost.EXAM_OFF_RIGHT
_EX_ON_RIGHT = passcost.EXAM_ON_RIGHT
_EX_TAIL = passcost.EXAM_TAIL

# The $2625 walk's own branch costs (see passcost); cnt[2] accumulates them.
_W_SETUP = passcost.WALK_SETUP
_W_QUAD = np.array(passcost.WALK_SETUP_QUAD, dtype=np.int64)
_W_PITCH = passcost.WALK_SETUP_PITCH
_W_ROWS = passcost.WALK_ROWS
_W_HINT = passcost.WALK_HINT
_W_SOUND = passcost.TILE_SOUND
_R_HEAD = passcost.ROW_HEAD
_R_LAST = passcost.ROW_LAST
_R_MORE = passcost.ROW_MORE
_R_NEAR = passcost.ROW_NEAR
_R_OBS = passcost.ROW_OBSERVER
_R_SCAN = passcost.ROW_SCAN
_R_START_SAME = passcost.ROW_START_SAME
_R_START_AFTER = passcost.ROW_START_AFTER + passcost.ROW_START_AFTER_TAIL
_R_START_BEFORE = passcost.ROW_START_BEFORE + passcost.ROW_START_BEFORE_TAIL
_R_EXTRA = passcost.ROW_EXTRA
_R_EXTRA_LAST = passcost.ROW_EXTRA_LAST
_R_END_SAME = passcost.ROW_END_SAME
_R_END_BEFORE = passcost.ROW_END_BEFORE + passcost.ROW_END_BEFORE_TAIL
_R_END_AFTER = passcost.ROW_END_AFTER + passcost.ROW_END_AFTER_TAIL
_R_PLOT = passcost.ROW_PLOT
_O_HEAD = passcost.OBS_ROW_HEAD
_O_START = passcost.OBS_ROW_START
_O_TEST = passcost.OBS_ROW_TEST
_O_END = passcost.OBS_ROW_END
_O_SKIP = passcost.OBS_ROW_SKIP
_O_PLOT = passcost.OBS_ROW_PLOT
_O_TILE = passcost.OBS_TILE + passcost.OBS_TILE_TEST
_O_TILE_ON = passcost.OBS_TILE_ON + passcost.OBS_TILE_TAIL
_O_TILE_OFF = passcost.OBS_TILE_OFF
_S_HEAD = passcost.SCAN_HEAD
_S_OFF = passcost.SCAN_OFF
_S_VISIBLE = passcost.SCAN_VISIBLE
_S_CROPPED = passcost.SCAN_CROPPED
_S_WHOLE = passcost.SCAN_WHOLE
_S_INC = passcost.SCAN_INC
_S_INC_END = passcost.SCAN_INC_END
_S_DEC = passcost.SCAN_DEC
_S_DEC_END = passcost.SCAN_DEC_END
_S_END_HEAD = passcost.SCAN_END_HEAD
_S_END_WHOLE = passcost.SCAN_END_WHOLE
_S_END_CROP = passcost.SCAN_END_CROP
_S_END_GAP = passcost.SCAN_END_GAP
_S_END_STOP = passcost.SCAN_END_STOP
_S_GAP_LOOP = passcost.SCAN_GAP_LOOP
_S_GAP_HIT = passcost.SCAN_GAP_HIT
_S_GAP_STOP = passcost.SCAN_GAP_STOP
_S_CROP_HEAD = passcost.SCAN_CROP_HEAD
_S_CROP_MORE = passcost.SCAN_CROP_MORE
_S_CROP_AGAIN = passcost.SCAN_CROP_AGAIN
_S_CROP_EXIT = passcost.SCAN_CROP_EXIT
_S_CROP_LEFT = passcost.SCAN_CROP_LEFT
_S_CROP_STOP = passcost.SCAN_CROP_STOP
_S_START_LOOP = passcost.SCAN_START_LOOP
_S_START_HIT = passcost.SCAN_START_HIT
_S_START_STOP = passcost.SCAN_START_STOP
_S_START_SETUP = passcost.SCAN_START_SETUP
_S_LEFT_LOOP = passcost.SCAN_LEFT_LOOP
_S_LEFT_HIT = passcost.SCAN_LEFT_HIT
_S_LEFT_STOP = passcost.SCAN_LEFT_STOP
_P_ROW_HEAD = passcost.PLOT_ROW_HEAD
_P_ROW_FRONT = passcost.PLOT_ROW_FRONT
_P_ROW_BACK = passcost.PLOT_ROW_BACK
_P_ROW_TURN = passcost.PLOT_ROW_TURN
_P_ROW_DONE = passcost.PLOT_ROW_DONE
_P_ROW_BACK_END = passcost.PLOT_ROW_BACK_END
_P_ROW_BACK_LOW = passcost.PLOT_ROW_BACK_LOW
_P_ROW_BACK_EMPTY = passcost.PLOT_ROW_BACK_EMPTY


@njit(cache=True, inline="always")
def _tile_byte(mem, x, y):
    """terrain.tile_byte: the raw tiles_table byte via the ROM address arithmetic
    ``page=(x&3)+4``, ``lo=((x<<3)&0xE0)|(y&0x1F)``."""
    xx = x & 0xFF
    yy = y & 0xFF
    return int(mem[((xx & 3) + 4) * 256 + (((xx << 3) & 0xE0) | (yy & 0x1F))])


@njit(cache=True)
def _ground(mem, x, y):
    """terrain.resolve_ground: (height, slope) at (x,y).  An object tile resolves to the
    bottommost stacked object's z_height with slope 0."""
    b = _tile_byte(mem, x, y)
    if b < _OBJECT_TILE:
        return b >> 4, b & 0x0F
    slot = b & 0x3F
    for _ in range(_NUM_SLOTS):
        flags = int(mem[_OFLAGS + slot])
        if flags < 0x40:  # on the ground
            break
        slot = flags & 0x3F
    return int(mem[_OZHEIGHT + slot]), 0


@njit(cache=True, inline="always")
def _maxz(mz, row, xi):
    """($72),Y horizon lookup; off-table reads hit zeroed RAM."""
    if 0 <= row <= _LAST and 0 <= xi <= _LAST:
        return mz[row, xi]
    return 0


@njit(cache=True, inline="always")
def _axis(tile_k, olo_k, ohi_k):
    """$2503 per-axis signed delta: returns (lo, hi, sign_extension, |hi|)."""
    lo = (0 - olo_k) & 0xFF
    cin = 1 if olo_k == 0 else 0
    h = (tile_k - ohi_k - (1 - cin)) & 0xFF
    if h & 0x80:
        nb = 1 if lo != 0 else 0
        return lo, h, 0xFF, (0 - h - nb) & 0xFF
    return lo, h, 0, h


@njit(cache=True)
def _trace(tz, mz, objx, objy, objz, zfrac, ty, tx):
    """$24E2: ray-march observer->tile, 1 if unobstructed."""
    d0, h0, e0, a0 = _axis(tx, 0x80, objx)
    d1, h1, e1, a1 = _axis(tz[ty, tx] >> 1, zfrac, objz)
    d2, h2, e2, a2 = _axis(ty, 0x80, objy)
    maxd = a2
    if a1 >= maxd:
        maxd = a1
    if a0 >= maxd:
        maxd = a0
    if ((maxd << 2) & 0xFF) < 6:  # $252A: within ~1 tile => visible
        return 1
    step = 0xFF
    a = (maxd << 2) & 0xFF  # $2532 scale: ~2-4 substeps per tile
    while True:
        c = (d0 >> 7) & 1
        d0 = (d0 << 1) & 0xFF
        h0 = ((h0 << 1) | c) & 0xFF
        c = (d1 >> 7) & 1
        d1 = (d1 << 1) & 0xFF
        h1 = ((h1 << 1) | c) & 0xFF
        c = (d2 >> 7) & 1
        d2 = (d2 << 1) & 0xFF
        h2 = ((h2 << 1) | c) & 0xFF
        step >>= 1
        carry = (a >> 7) & 1
        a = (a << 1) & 0xFF
        if carry:
            break
    ax_lo = 0x80  # $37
    ax_int = objx  # $3A
    ay_lo = 0x80  # $39
    ay_row = (objy + 0x40) & 0xFF  # $73
    az_lo = 0  # $35
    az_mid = zfrac  # $38
    az_hi = objz  # $3B
    for _ in range(step):  # $2576 march; blocked when the ray dips below the horizon
        ax_lo += h0
        ax_int = (ax_int + e0 + (ax_lo >> 8)) & 0xFF
        ax_lo &= 0xFF
        ay_lo += h2
        ay_row = (ay_row + e2 + (ay_lo >> 8)) & 0xFF
        ay_lo &= 0xFF
        az_lo += d1
        az_mid += h1 + (az_lo >> 8)
        az_lo &= 0xFF
        az_hi = (az_hi + e1 + (az_mid >> 8)) & 0xFF
        az_mid &= 0xFF
        if az_hi < _maxz(mz, (ay_row - 0x40) & 0xFF, ax_int):
            return 0
    return 1


@njit(cache=True)
def occlusion_table(mem, observer):
    """populate_tile_visibility_bit_table ($245B) over the 64 KB image ``mem``, rays cast
    from object slot ``observer`` -> uint8[32,32], 1 where the tile is unoccluded.
    ``out[ty, tx]`` is the $3E80/$24DA bit for tile (tx,ty)."""
    objx = int(mem[_OX + observer])
    objy = int(mem[_OY + observer])
    objz = int(mem[_OZHEIGHT + observer])
    zfrac = int(mem[_OZFRAC + observer])
    tz = np.zeros((_N, _N), dtype=np.int64)  # $25C4: (z<<1)|not_flat per tile
    for y in range(_N):
        for x in range(_N):
            z, slope = _ground(mem, x, y)
            tz[y, x] = ((z << 1) | (1 if slope else 0)) & 0xFF
    mz = np.zeros((_N, _N), dtype=np.int64)  # $25ED: min of the 4 corner bytes, >>1
    for y in range(_LAST, -1, -1):
        for x in range(_LAST, -1, -1):
            b = tz[y, x]
            if not b & 1:
                mz[y, x] = b >> 1
            else:
                m = b
                if tz[y, x + 1] < m:
                    m = tz[y, x + 1]
                if tz[y + 1, x + 1] < m:
                    m = tz[y + 1, x + 1]
                if tz[y + 1, x] < m:
                    m = tz[y + 1, x]
                mz[y, x] = m >> 1
    raw = np.zeros((_N, _N), dtype=np.uint8)
    for ty in range(_N):
        for tx in range(_N):
            raw[ty, tx] = _trace(tz, mz, objx, objy, objz, zfrac, ty, tx)
    vis = np.zeros((_N, _N), dtype=np.uint8)
    for y in range(_LAST, -1, -1):  # $248A combine: 2x2 raytrace dilation AND height
        for x in range(_LAST, -1, -1):
            b = tz[y, x]
            height_ok = (b & 1) != 0 or (b >> 1) <= objz  # flat and above eye => hidden
            block = raw[y, x] or raw[y, x + 1] or raw[y + 1, x] or raw[y + 1, x + 1]
            if block and height_ok:
                vis[y, x] = 1
    return vis


_LAST_TILE = _N - 1  # $1F: highest column/row index the walk addresses
_OFFSET_TO_TILE = np.array(
    (0x00, 0x01, 0x21, 0x20), dtype=np.int64
)  # $27D3, by quadrant

# ``setup`` ($2625) packed into one int64 vector so the njit signatures stay short.
S_QUAD, S_C3, S_C1D, S_REF_LO, S_REF_HI = 0, 1, 2, 3, 4
S_VANGLE, S_OBS_ZH, S_OBS_ZF, S_LEFT, S_RIGHT = 5, 6, 7, 8, 9
S_N = 10


@njit(cache=True, inline="always")
def _neg16(hi, lo):
    """Two's-complement negate the 16-bit (hi:lo), as $2865/$286C do."""
    borrow = 1 if lo != 0 else 0
    return (-hi - borrow) & 0xFF, (-lo) & 0xFF


@njit(cache=True, inline="always")
def _tile_xy(quadrant, col, row):
    """($28A3) map (column,row) to (tile_x,tile_y) for the view orientation."""
    if quadrant == 0:
        return col, row
    if quadrant == 1:
        return row, (_LAST_TILE - col) & 0xFF
    if quadrant == 2:
        return (_LAST_TILE - col) & 0xFF, (_LAST_TILE - row) & 0xFF
    return (_LAST_TILE - row) & 0xFF, col


@njit(cache=True, inline="always")
def _signed16(hi, lo):
    """A corner's signed 16-bit projected coordinate ($2F02/$2DD2)."""
    return (hi - 256 if hi & 0x80 else hi) * 256 + lo


@njit(cache=True, inline="always")
def _clamp(v, lo, hi):
    """Clamp ``v`` into ``[lo, hi]``."""
    if v < lo:
        return lo
    if v > hi:
        return hi
    return v


@njit(cache=True)
def _tile_height(mem, x, y):
    """projector._tile_height: ($28E9) height, tile byte and object levels walked."""
    b = _tile_byte(mem, x, y)
    if b < _OBJECT_TILE:
        return b >> 4, b, 0
    slot = b & 0x3F
    levels = 1
    for _ in range(_NUM_SLOTS):
        flags = int(mem[_OFLAGS + slot])
        if flags < 0x40:  # $28FA CMP #$40: this level is on the ground
            break
        slot = flags & 0x3F
        levels += 1
    return int(mem[_OZHEIGHT + slot]), b, levels


@njit(cache=True)
def _project(mem, zp, su, vis, col, row, out):
    """$2845 for grid point (col,row) -> out = sx_lo, sx_hi, sy_lo, sy_hi, tile_byte,
    onscreen, cycles.  ``zp`` is zeroed first because the reference allocates a fresh
    ``defaultdict(int)`` per call and _calc_angle's both-zero exit leaves $5C/$5D/$7A/$7B
    unwritten, so a stale window would leak into _calc_hypotenuse."""
    for i in range(ZP_LO, ZP_HI):  # nothing below $50 is touched
        zp[i] = 0
    cyc = _EX_CALL + _EX_HEAD
    sx = (col - su[S_C3] - 1) & 0xFF  # signed column ($2858)
    zp[0x86] = sx
    zp[0x80] = 0x80
    if sx & 0x80:
        zp[0x83], zp[0x80] = _neg16(sx, 0x80)
        cyc += _EX_COL_NEG
    else:
        zp[0x83] = sx
        cyc += _EX_COL_POS
    sr = (row - su[S_C1D] - 1) & 0xFF  # signed row ($2876)
    zp[0x88] = sr
    zp[0x82] = 0x80
    cyc += _EX_ROW
    if sr & 0x80:
        zp[0x85], zp[0x82] = _neg16(sr, 0x80)
        cyc += _EX_ROW_NEG
    else:
        zp[0x85] = sr
        cyc += _EX_ROW_POS
    cyc += _EX_ANGLE + _calc_angle(zp)  # $9287
    out[0] = (zp[0x8A] - su[S_REF_LO]) & 0xFF  # screen x ($2891), carry stays set
    out[1] = (zp[0x8B] - su[S_REF_HI]) & 0xFF
    cyc += _EX_STORE + _calc_hypotenuse(zp)  # $937F
    cyc += _EX_QUAD[su[S_QUAD]] + _EX_ADDR
    tx, ty = _tile_xy(su[S_QUAD], col, row)
    height, tile_byte, levels = _tile_height(mem, tx, ty)
    if levels:  # $28F2: an object tile walks its stack and skips the raytrace bit
        cyc += _EX_OBJECT + _EX_OBJECT_BOTTOM + (levels - 1) * _EX_OBJECT_STEP
    else:
        cyc += _EX_GROUND + (_EX_VISIBLE if vis[ty, tx] else _EX_HIDDEN)
    zf = su[S_OBS_ZF]  # tile height relative to eye ($291E)
    zp[0x80] = (-zf) & 0xFF
    rel_z_hi = (height - su[S_OBS_ZH] - (1 if zf else 0)) & 0xFF
    out[3], vcyc = _vertical_angle(zp, rel_z_hi, su[S_VANGLE])  # $933D
    out[2] = zp[0x50]
    out[4] = tile_byte
    cyc += _EX_VANGLE + vcyc + _EX_TAIL
    if out[1] < su[S_LEFT]:  # $293C on-screen test against $0007/$0012
        out[5] = 0x00
        cyc += _EX_OFF_LEFT
    else:
        cyc += _EX_ON_LEFT
        cyc += _EX_FRACTION if out[1] == su[S_LEFT] else _EX_NO_FRACTION
        cyc += _EX_RIGHT
        if out[1] < su[S_RIGHT]:
            out[5] = 0x80
            cyc += _EX_OFF_RIGHT
        else:
            out[5] = 0x81
            cyc += _EX_ON_RIGHT
    out[6] = cyc


@njit(cache=True)
def _cached(mem, zp, su, vis, cres, seen, col, row):
    """Fill the projection cache slot for (col & $FF, row) and return its column index."""
    c = col & 0xFF
    if seen[c, row] == 0:
        _project(mem, zp, su, vis, c, row, cres[c, row])
        seen[c, row] = 1
    return c


@njit(cache=True)
def _probe(mem, zp, su, vis, cres, seen, cnt, col, row):
    """One $2845 examination: counted on every call, cache hits included."""
    c = _cached(mem, zp, su, vis, cres, seen, col, row)
    cnt[0] += 1
    cnt[1] += cres[c, row, 6]
    return cres[c, row, 5]


@njit(cache=True)
def _find_end(mem, zp, su, vis, cres, seen, cnt, row, col):
    """find_end_of_row_loop $27E2."""
    while True:
        start = col
        cnt[2] += _S_END_HEAD
        if col == _LAST_TILE:
            cnt[2] += _S_INC_END + _S_END_STOP
            return start, _LAST_TILE
        col += 1
        cnt[2] += _S_INC
        a = _probe(mem, zp, su, vis, cres, seen, cnt, col, row)
        if a == 0x81:
            cnt[2] += _S_END_WHOLE
            continue
        if a == 0x80:
            cnt[2] += _S_END_CROP
            return start, col
        cnt[2] += _S_END_GAP
        while True:  # find_first_visible_tile_at_end_loop $27F3
            if col == _LAST_TILE:
                cnt[2] += _S_INC_END + _S_GAP_STOP
                return start, col
            col += 1
            cnt[2] += _S_INC
            a = _probe(mem, zp, su, vis, cres, seen, cnt, col, row)
            if a == 0:
                cnt[2] += _S_GAP_LOOP
                continue
            cnt[2] += _S_GAP_HIT
            return start, col


@njit(cache=True)
def _start_left(mem, zp, su, vis, cres, seen, cnt, row, end, col):
    """find_first_visible_tile_at_start_of_row_loop $2820."""
    while True:
        if col == 0:
            cnt[2] += _S_DEC_END + _S_LEFT_STOP
            return 0, end
        col -= 1
        cnt[2] += _S_DEC
        if _probe(mem, zp, su, vis, cres, seen, cnt, col, row) == 0:
            cnt[2] += _S_LEFT_LOOP
            continue
        cnt[2] += _S_LEFT_HIT
        return col, end


@njit(cache=True)
def _crop_right(mem, zp, su, vis, cres, seen, cnt, row, col):
    """tile_is_cropped_to_right $27FF."""
    while True:
        end = col
        cnt[2] += _S_CROP_HEAD
        if col == 0:
            cnt[2] += _S_DEC_END + _S_CROP_STOP
            return 0, end
        col -= 1
        cnt[2] += _S_DEC
        a = _probe(mem, zp, su, vis, cres, seen, cnt, col, row)
        cnt[2] += _S_CROP_MORE
        if a == 0x80:
            cnt[2] += _S_CROP_AGAIN
            continue
        if a != 0:  # into_find_first_visible_tile_at_start_of_row_loop $2825
            cnt[2] += _S_CROP_EXIT
            return col, end
        cnt[2] += _S_CROP_LEFT
        return _start_left(mem, zp, su, vis, cres, seen, cnt, row, end, col)


@njit(cache=True)
def _find_extent(mem, zp, su, vis, cres, seen, cnt, row, hint):
    """find_visible_extent $27D7."""
    col = hint & 0xFF
    cnt[2] += _S_HEAD
    a = _probe(mem, zp, su, vis, cres, seen, cnt, col, row)
    if a == 0x80:
        cnt[2] += _S_VISIBLE + _S_CROPPED
        return _crop_right(mem, zp, su, vis, cres, seen, cnt, row, col)
    if a != 0:
        cnt[2] += _S_VISIBLE + _S_WHOLE
        return _find_end(mem, zp, su, vis, cres, seen, cnt, row, col)
    cnt[2] += _S_OFF
    while True:  # find_first_visible_tile_at_start_loop $2811
        if col == _LAST_TILE:  # endRow2 $2818
            cnt[2] += _S_INC_END + _S_START_STOP + _S_START_SETUP
            return _start_left(
                mem, zp, su, vis, cres, seen, cnt, row, _LAST_TILE, hint & 0xFF
            )
        col += 1
        cnt[2] += _S_INC
        a = _probe(mem, zp, su, vis, cres, seen, cnt, col, row)
        if a == 0:
            cnt[2] += _S_START_LOOP
            continue
        cnt[2] += _S_START_HIT + _S_START_SETUP
        return _start_left(mem, zp, su, vis, cres, seen, cnt, row, col, hint & 0xFF)


@njit(cache=True)
def project_scene(mem, su, vis, row_hint, screen_h, w_scale):
    """plot_world's tile selection: the $27D7/$26DE walk then the $2A24 plot range.

    Returns (ints[n,11], span[n], n, n_examine) with ints = col, row, tx, ty, sx_lo,
    sx_hi, sy_lo, sy_hi, tile_byte, onscreen, h; the caller clamps the unclamped span
    to the screen width.  ``vis`` is the memoized $245B table."""
    zp = np.zeros(ZP_HI, dtype=np.int64)
    # The walk masks col to $FF and steps y down through 0, and the plot range reaches
    # row $20, so the cache spans 256 columns by 33 rows -- not 32 by 32.
    cres = np.zeros((256, _N + 1, 7), dtype=np.int64)
    seen = np.zeros((256, _N + 1), dtype=np.uint8)
    cnt = np.zeros(3, dtype=np.int64)
    c3 = su[S_C3]
    c1d = su[S_C1D]
    rrow = np.zeros(_N + 1, dtype=np.int64)
    rlo = np.zeros(_N + 1, dtype=np.int64)
    rhi = np.zeros(_N + 1, dtype=np.int64)
    nrows = 0
    row = _LAST_TILE
    cnt[2] += (
        _W_SETUP
        + _W_QUAD[su[S_QUAD]]
        + (_W_PITCH if su[S_VANGLE] & 0x80 else 0)
        + _W_ROWS
        + _R_SCAN
    )
    start, end = _find_extent(mem, zp, su, vis, cres, seen, cnt, row, row_hint)
    cnt[2] += _W_HINT
    while True:
        row -= 1
        cnt[2] += _R_HEAD + _W_SOUND
        if row < 0:
            cnt[2] += _R_LAST
            break
        cnt[2] += _R_MORE
        if row == c1d:  # consider_plotting_observer_row $276F: last, observer row
            cnt[2] += _R_OBS + _O_HEAD
            y = (start + 1) & 0xFF
            if y == c3:  # plot_observer_row $2786: plots the single tile $0037
                cnt[2] += _O_START + _O_PLOT
                _probe(mem, zp, su, vis, cres, seen, cnt, start, row)
                _probe(mem, zp, su, vis, cres, seen, cnt, y, row)
                _probe(mem, zp, su, vis, cres, seen, cnt, c3, row)
                rrow[nrows], rlo[nrows], rhi[nrows] = row, start, y
                nrows += 1
            elif (end - 2) & 0xFF == c3:  # $277B: plots the single tile $0038-1
                cnt[2] += _O_TEST + _O_END + _O_PLOT
                _probe(mem, zp, su, vis, cres, seen, cnt, (end - 1) & 0xFF, row)
                _probe(mem, zp, su, vis, cres, seen, cnt, end, row)
                _probe(mem, zp, su, vis, cres, seen, cnt, c3, row)
                rrow[nrows], rlo[nrows], rhi[nrows] = row, (end - 1) & 0xFF, end
                nrows += 1
            else:  # skip_plotting_observer_row $2793: only the observer tile ($27CE)
                cnt[2] += _O_TEST + _O_SKIP
                _probe(mem, zp, su, vis, cres, seen, cnt, c3, row)
            # $279E: the observer's own tile is plotted only when it is on screen.
            cnt[2] += _O_TILE
            ic = _cached(mem, zp, su, vis, cres, seen, c3, row)
            cnt[2] += _O_TILE_ON if cres[ic, row, 3] < 2 else _O_TILE_OFF
            break
        cnt[2] += _R_NEAR + _R_SCAN
        p_start, p_end = start, end
        start, end = _find_extent(mem, zp, su, vis, cres, seen, cnt, row, p_start)
        if start < p_start:  # this_row_starts_before $2713
            cnt[2] += _R_START_BEFORE
            y = (p_start - 1) & 0xFF
            _probe(mem, zp, su, vis, cres, seen, cnt, y, row)
            while y != start:
                y = (y - 1) & 0xFF
                cnt[2] += _R_EXTRA
                _probe(mem, zp, su, vis, cres, seen, cnt, y, row)
            cnt[2] += _R_EXTRA_LAST
        elif start > p_start:  # calculate_this_row_new_first_tiles $2709
            cnt[2] += _R_START_AFTER
            y = (start - 1) & 0xFF
            _probe(mem, zp, su, vis, cres, seen, cnt, y, row)
            while y != p_start:
                y = (y - 1) & 0xFF
                cnt[2] += _R_EXTRA
                _probe(mem, zp, su, vis, cres, seen, cnt, y, row)
            cnt[2] += _R_EXTRA_LAST
        else:
            cnt[2] += _R_START_SAME
        if end > p_end:  # this_row_ends_after $2741
            cnt[2] += _R_END_AFTER
            y = p_end
            while True:
                y = (y + 1) & 0xFF
                _probe(mem, zp, su, vis, cres, seen, cnt, y, row)
                if y == end:
                    break
                cnt[2] += _R_EXTRA
            cnt[2] += _R_EXTRA_LAST
        elif end < p_end:  # calculate_this_row_new_last_tiles $2737
            cnt[2] += _R_END_BEFORE
            y = end
            while True:
                y = (y + 1) & 0xFF
                _probe(mem, zp, su, vis, cres, seen, cnt, y, row)
                if y == p_end:
                    break
                cnt[2] += _R_EXTRA
            cnt[2] += _R_EXTRA_LAST
        else:
            cnt[2] += _R_END_SAME
        cnt[2] += _R_PLOT
        rrow[nrows] = row
        rlo[nrows] = min(start, p_start)
        rhi[nrows] = max(end, p_end)
        nrows += 1

    # plot_tile ($2A24): the drawn tile is examine (col+offc, row+offr), $001B=$27D3[quad].
    s1b = _OFFSET_TO_TILE[su[S_QUAD]]
    offc = s1b & 1
    offr = (s1b >> 5) & 1
    cap = (_N + 1) * (_N + 1)
    out = np.zeros((cap, 11), dtype=np.int64)
    ws = np.zeros(cap, dtype=np.float64)
    fill = np.zeros((cap, _FILL_COLS), dtype=np.int64)
    n = 0
    nfill = 0
    c3 = su[S_C3]
    for i in range(nrows):
        re = (rrow[i] + offr) & 0xFF
        lo, hi = rlo[i], rhi[i]  # $295D plots up to $0003, then back down from $0038
        nfront = max(0, min(hi, c3) - lo)
        nback = hi - max(lo, c3) if c3 < hi else 0
        cnt[2] += _P_ROW_HEAD + nfront * _P_ROW_FRONT + nback * _P_ROW_BACK
        if nfront == hi - lo:  # $2963: the front half ran to $0038 and left
            cnt[2] += _P_ROW_DONE
        else:
            cnt[2] += _P_ROW_TURN
            if max(lo, c3) == 0:  # $2978 BMI: the descent ran off the bottom
                cnt[2] += _P_ROW_BACK_EMPTY
            elif c3 <= lo:  # $297E BCC: it reached $0037 before $0003
                cnt[2] += _P_ROW_BACK_LOW
            else:
                cnt[2] += _P_ROW_BACK_END
        for k in range(hi - lo):
            col = lo + k if k < nfront else hi - 1 - (k - nfront)
            if k >= nfront and col < c3:
                break
            ce = (col + offc) & 0xFF
            i0 = _cached(mem, zp, su, vis, cres, seen, ce, re)
            tile_byte = cres[i0, re, 4]
            tx, ty = _tile_xy(su[S_QUAD], ce, re)
            if tile_byte != 0 and tile_byte < _OBJECT_TILE and vis[ty, tx] == 0:
                tile_byte = 0  # $291B zeroes $0180 for hidden non-object tiles
            fill[nfill, 0] = tile_byte
            fill[nfill, 1] = (col ^ rrow[i]) & 1
            for k4 in range(4):
                cc = col + (1 if k4 >= 2 else 0)
                rr = rrow[i] + (1 if 1 <= k4 <= 2 else 0)
                ic = _cached(mem, zp, su, vis, cres, seen, cc, rr)
                b = 2 + 4 * k4
                fill[nfill, b] = cres[ic, rr, 0]
                fill[nfill, b + 1] = cres[ic, rr, 1]
                fill[nfill, b + 2] = cres[ic, rr, 2]
                fill[nfill, b + 3] = cres[ic, rr, 3]
            nfill += 1
            if tile_byte == 0:  # $0180 slot zero: nothing to plot ($2A27 BEQ)
                continue
            c1 = min(ce + 1, _LAST_TILE)
            r1 = min(re + 1, _LAST_TILE)
            i1 = _cached(mem, zp, su, vis, cres, seen, c1, re)
            i2 = _cached(mem, zp, su, vis, cres, seen, ce, r1)
            i3 = _cached(mem, zp, su, vis, cres, seen, c1, r1)
            y0 = _signed16(cres[i0, re, 3], cres[i0, re, 2])
            y1 = _signed16(cres[i1, re, 3], cres[i1, re, 2])
            y2 = _signed16(cres[i2, r1, 3], cres[i2, r1, 2])
            y3 = _signed16(cres[i3, r1, 3], cres[i3, r1, 2])
            x0 = _signed16(cres[i0, re, 1], cres[i0, re, 0])
            x1 = _signed16(cres[i1, re, 1], cres[i1, re, 0])
            x2 = _signed16(cres[i2, r1, 1], cres[i2, r1, 0])
            x3 = _signed16(cres[i3, r1, 1], cres[i3, r1, 0])
            top = _clamp(min(y0, y1, y2, y3), 0, screen_h)
            bot = _clamp(max(y0, y1, y2, y3), 0, screen_h)
            span = (max(x0, x1, x2, x3) - min(x0, x1, x2, x3)) / w_scale
            out[n, 0] = ce
            out[n, 1] = re
            out[n, 2] = tx
            out[n, 3] = ty
            out[n, 4] = cres[i0, re, 0]
            out[n, 5] = cres[i0, re, 1]
            out[n, 6] = cres[i0, re, 2]
            out[n, 7] = cres[i0, re, 3]
            out[n, 8] = tile_byte
            out[n, 9] = cres[i0, re, 5]
            out[n, 10] = bot - top
            ws[n] = span
            n += 1
    return out, ws, n, cnt[0], cnt[1], fill[:nfill], cnt[2]
