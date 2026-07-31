"""Numba twins of $245B (tile visibility) and of plot_world's projection walk.

$2845 + $27D7 + $26DE + the $2A24 tile selection; :mod:`sentinel.projector` stays the
bit-exact reference and the numba-absent fallback, as :mod:`sentinel.los_jit` does for
:mod:`sentinel.los`.  The trig core is reused from :mod:`sentinel.enemies_jit`.
"""

import numpy as np
from numba import njit

from sentinel import memmap as mm
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
def _project(mem, zp, su, col, row, out):
    """$2845 for grid point (col,row) -> out = sx_lo, sx_hi, sy_lo, sy_hi, tile_byte,
    onscreen.  ``zp`` is zeroed first because the reference allocates a fresh
    ``defaultdict(int)`` per call and _calc_angle's both-zero exit leaves $5C/$5D/$7A/$7B
    unwritten, so a stale window would leak into _calc_hypotenuse."""
    for i in range(ZP_LO, ZP_HI):  # nothing below $50 is touched
        zp[i] = 0
    sx = (col - su[S_C3] - 1) & 0xFF  # signed column ($2858)
    zp[0x86] = sx
    zp[0x80] = 0x80
    if sx & 0x80:
        zp[0x83], zp[0x80] = _neg16(sx, 0x80)
    else:
        zp[0x83] = sx
    sr = (row - su[S_C1D] - 1) & 0xFF  # signed row ($2876)
    zp[0x88] = sr
    zp[0x82] = 0x80
    if sr & 0x80:
        zp[0x85], zp[0x82] = _neg16(sr, 0x80)
    else:
        zp[0x85] = sr
    _calc_angle(zp)  # $9287
    out[0] = (zp[0x8A] - su[S_REF_LO]) & 0xFF  # screen x ($2891), carry stays set
    out[1] = (zp[0x8B] - su[S_REF_HI]) & 0xFF
    _calc_hypotenuse(zp)  # $937F
    tx, ty = _tile_xy(su[S_QUAD], col, row)
    tile_byte = _tile_byte(mem, tx, ty)
    height = _ground(mem, tx, ty)[0]  # $28E9: object tiles resolve down the flags chain
    zf = su[S_OBS_ZF]  # tile height relative to eye ($291E)
    zp[0x80] = (-zf) & 0xFF
    rel_z_hi = (height - su[S_OBS_ZH] - (1 if zf else 0)) & 0xFF
    out[3], _vcyc = _vertical_angle(zp, rel_z_hi, su[S_VANGLE])  # $933D
    out[2] = zp[0x50]
    out[4] = tile_byte
    if out[1] < su[S_LEFT]:  # $293C on-screen test against $0007/$0012
        out[5] = 0x00
    elif out[1] < su[S_RIGHT]:
        out[5] = 0x80
    else:
        out[5] = 0x81


@njit(cache=True)
def _cached(mem, zp, su, cres, seen, col, row):
    """Fill the projection cache slot for (col & $FF, row) and return its column index."""
    c = col & 0xFF
    if seen[c, row] == 0:
        _project(mem, zp, su, c, row, cres[c, row])
        seen[c, row] = 1
    return c


@njit(cache=True)
def _probe(mem, zp, su, cres, seen, cnt, col, row):
    """One $2845 examination: counted on every call, cache hits included."""
    cnt[0] += 1
    return cres[_cached(mem, zp, su, cres, seen, col, row), row, 5]


@njit(cache=True)
def _find_end(mem, zp, su, cres, seen, cnt, row, col):
    """find_end_of_row_loop $27E2."""
    while True:
        start = col
        if col == _LAST_TILE:
            return start, _LAST_TILE
        col += 1
        a = _probe(mem, zp, su, cres, seen, cnt, col, row)
        if a == 0x81:
            continue
        if a == 0x80:
            return start, col
        while True:  # find_first_visible_tile_at_end_loop $27F3
            if col == _LAST_TILE:
                return start, col
            col += 1
            a = _probe(mem, zp, su, cres, seen, cnt, col, row)
            if a == 0:
                continue
            return start, col


@njit(cache=True)
def _start_left(mem, zp, su, cres, seen, cnt, row, end, col):
    """find_first_visible_tile_at_start_of_row_loop $2820."""
    while True:
        if col == 0:
            return 0, end
        col -= 1
        if _probe(mem, zp, su, cres, seen, cnt, col, row) == 0:
            continue
        return col, end


@njit(cache=True)
def _crop_right(mem, zp, su, cres, seen, cnt, row, col):
    """tile_is_cropped_to_right $27FF."""
    while True:
        end = col
        if col == 0:
            return 0, end
        col -= 1
        a = _probe(mem, zp, su, cres, seen, cnt, col, row)
        if a == 0x80:
            continue
        if a != 0:  # into_find_first_visible_tile_at_start_of_row_loop $2825
            return col, end
        return _start_left(mem, zp, su, cres, seen, cnt, row, end, col)


@njit(cache=True)
def _find_extent(mem, zp, su, cres, seen, cnt, row, hint):
    """find_visible_extent $27D7."""
    col = hint & 0xFF
    a = _probe(mem, zp, su, cres, seen, cnt, col, row)
    if a == 0x80:
        return _crop_right(mem, zp, su, cres, seen, cnt, row, col)
    if a != 0:
        return _find_end(mem, zp, su, cres, seen, cnt, row, col)
    while True:  # find_first_visible_tile_at_start_loop $2811
        if col == _LAST_TILE:  # endRow2 $2818
            return _start_left(
                mem, zp, su, cres, seen, cnt, row, _LAST_TILE, hint & 0xFF
            )
        col += 1
        a = _probe(mem, zp, su, cres, seen, cnt, col, row)
        if a == 0:
            continue
        return _start_left(mem, zp, su, cres, seen, cnt, row, col, hint & 0xFF)


@njit(cache=True)
def project_scene(mem, su, vis, row_hint, screen_h, w_scale):
    """plot_world's tile selection: the $27D7/$26DE walk then the $2A24 plot range.

    Returns (ints[n,11], span[n], n, n_examine) with ints = col, row, tx, ty, sx_lo,
    sx_hi, sy_lo, sy_hi, tile_byte, onscreen, h; the caller clamps the unclamped span
    to the screen width.  ``vis`` is the memoized $245B table."""
    zp = np.zeros(ZP_HI, dtype=np.int64)
    # The walk masks col to $FF and steps y down through 0, and the plot range reaches
    # row $20, so the cache spans 256 columns by 33 rows -- not 32 by 32.
    cres = np.zeros((256, _N + 1, 6), dtype=np.int64)
    seen = np.zeros((256, _N + 1), dtype=np.uint8)
    cnt = np.zeros(1, dtype=np.int64)
    c3 = su[S_C3]
    c1d = su[S_C1D]
    rrow = np.zeros(_N + 1, dtype=np.int64)
    rlo = np.zeros(_N + 1, dtype=np.int64)
    rhi = np.zeros(_N + 1, dtype=np.int64)
    nrows = 0
    row = _LAST_TILE
    start, end = _find_extent(mem, zp, su, cres, seen, cnt, row, row_hint)
    while True:
        row -= 1
        if row < 0:
            break
        if row == c1d:  # consider_plotting_observer_row $276F: last, observer row
            y = (start + 1) & 0xFF
            if y == c3:  # plot_observer_row $2786: plots the single tile $0037
                _probe(mem, zp, su, cres, seen, cnt, start, row)
                _probe(mem, zp, su, cres, seen, cnt, y, row)
                _probe(mem, zp, su, cres, seen, cnt, c3, row)
                rrow[nrows], rlo[nrows], rhi[nrows] = row, start, y
                nrows += 1
            elif (end - 2) & 0xFF == c3:  # $277B: plots the single tile $0038-1
                _probe(mem, zp, su, cres, seen, cnt, (end - 1) & 0xFF, row)
                _probe(mem, zp, su, cres, seen, cnt, end, row)
                _probe(mem, zp, su, cres, seen, cnt, c3, row)
                rrow[nrows], rlo[nrows], rhi[nrows] = row, (end - 1) & 0xFF, end
                nrows += 1
            else:  # skip_plotting_observer_row $2793: only the observer tile ($27CE)
                _probe(mem, zp, su, cres, seen, cnt, c3, row)
            break
        p_start, p_end = start, end
        start, end = _find_extent(mem, zp, su, cres, seen, cnt, row, p_start)
        if start < p_start:  # this_row_starts_before $2713
            y = (p_start - 1) & 0xFF
            _probe(mem, zp, su, cres, seen, cnt, y, row)
            while y != start:
                y = (y - 1) & 0xFF
                _probe(mem, zp, su, cres, seen, cnt, y, row)
        elif start > p_start:  # calculate_this_row_new_first_tiles $2709
            y = (start - 1) & 0xFF
            _probe(mem, zp, su, cres, seen, cnt, y, row)
            while y != p_start:
                y = (y - 1) & 0xFF
                _probe(mem, zp, su, cres, seen, cnt, y, row)
        if end > p_end:  # this_row_ends_after $2741
            y = p_end
            while True:
                y = (y + 1) & 0xFF
                _probe(mem, zp, su, cres, seen, cnt, y, row)
                if y == end:
                    break
        elif end < p_end:  # calculate_this_row_new_last_tiles $2737
            y = end
            while True:
                y = (y + 1) & 0xFF
                _probe(mem, zp, su, cres, seen, cnt, y, row)
                if y == p_end:
                    break
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
    n = 0
    for i in range(nrows):
        re = (rrow[i] + offr) & 0xFF
        for col in range(rlo[i], rhi[i]):  # plot range [$0037, $0038)
            ce = (col + offc) & 0xFF
            i0 = _cached(mem, zp, su, cres, seen, ce, re)
            tile_byte = cres[i0, re, 4]
            if tile_byte == 0:  # $0180 slot zero: nothing to plot ($2A27 BEQ)
                continue
            tx, ty = _tile_xy(su[S_QUAD], ce, re)
            if tile_byte < _OBJECT_TILE and vis[ty, tx] == 0:
                continue  # $291B zeroes $0180 for hidden non-object tiles
            c1 = min(ce + 1, _LAST_TILE)
            r1 = min(re + 1, _LAST_TILE)
            i1 = _cached(mem, zp, su, cres, seen, c1, re)
            i2 = _cached(mem, zp, su, cres, seen, ce, r1)
            i3 = _cached(mem, zp, su, cres, seen, c1, r1)
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
    return out, ws, n, cnt[0]
