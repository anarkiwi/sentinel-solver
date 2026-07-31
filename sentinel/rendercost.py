"""plot_world's fill sequence in render order, emulated statefully and cycle-counted.

`$2A24` picks a tile's polygons, `$2D6C` projects their vertices into the current
vertical buffer, `$2EE4`/`$3002`/`$30BD` rasterise the edges into `$AD00`/`$AE00`, and
`$22AA` fills the rows between them. Per-block cycles: :mod:`sentinel.passcost`.
"""

import numpy as np

from sentinel import passcost

try:  # one source, compiled when numba is present; ``f.py_func`` is the same code
    from numba import njit
except ImportError:  # pragma: no cover - numba absent

    def njit(*_args, **_kwargs):
        """No-op stand-in for numba's decorator."""
        return lambda fn: fn


# $2D15 triangle_corner_tile_delta_table as the corner it picks, indexed [triangle two][$0045]; a tile's corners are 0 = (col,row), 1 = (col,row+1), 2 = (col+1,row+1), 3 = (col+1,row).
TRI_THIRD = np.array(((2, 3), (0, 1)), dtype=np.int64)
# $2D39, the $0045 flag a one-point-different slope nibble sets.
SLOPE_FLAG = np.array((0, 0, 0, 0, 0, 0, 1, 1, 0, 0, 0, 0, 0, 0, 1, 1), dtype=np.int64)
# $29BB/$29BE/$29C1 per vertical buffer: ($0011 angle offset, $0035 left, $0061 width).
PLAY_BUFFERS = np.array(((0x0A, 0x50, 0x70), (0x02, 0x40, 0x70)), dtype=np.int64)
PAN_BUFFER = np.array(((0x0C, 0x60, 0x40),), dtype=np.int64)  # $2993 mode 2
# $994B/$994D, the raster window the caller of $2993 sets: a vertical pan plots into a
# 64-row strip, the play view and a horizontal pan into the whole 192.
SCREEN_TOP = 0xF0
SCREEN_BOTTOM = 0x30
STRIP_BOTTOM = 0xB0


def buffers(mode, window_columns=None):
    """The ($0011, $0035, $0061) of every vertical buffer and the $0010 to start in.

    A play or vertical-pan plot alternates the two halves of the wide buffer; a
    horizontal pan ($2993 mode 2) and a strip replot ($29C7) each own one buffer.
    """
    out = np.zeros((3, 3), dtype=np.int64)
    out[0] = PLAY_BUFFERS[0]
    out[1] = PLAY_BUFFERS[1]
    if window_columns is not None:  # $29C7 initialise_buffer_variables_for_updating
        b61 = (window_columns * 4) & 0xFF
        b36 = ((b61 >> 1) & 0xFC) | 0x80
        b35 = (b36 - b61) & 0xFF
        out[2] = (b35 >> 3, b35, b61)
        return out, 2
    if mode == 2:
        out[2] = PAN_BUFFER[0]
        return out, 2
    return out, mode


LINES_LEAVE_FLAG = 6 + passcost.LINES_NOTHING  # $2EBE LDA/BNE taken + $2ECA SEC/RTS
LINES_LEAVE_ROWS = 14 + passcost.LINES_NOTHING  # ... the $2EC2 row compare instead
LINES_OK = passcost.LINES_DONE + passcost.LINES_RTS  # carry clear: something to fill

# rows[] carries $0004/$0006/$007F and then the counts the ROM's own loops make, so a
# scene's geometry can be pinned against $2377/$23DC/$2F58/$2FA1/$3113/$316D hit counts.
R_BOTTOM, R_TOP, R_NOTHING = 0, 1, 2
R_STEEP, R_SHALLOW, R_WIDE_STEEP, R_WIDE_SHALLOW = 3, 4, 5, 6
R_N_STEEP, R_N_SHALLOW, R_N_WSTEEP, R_N_WSHALLOW, R_N_EDGE, R_N_SECT = (
    7,
    8,
    9,
    10,
    11,
    12,
)
R_N = 13
F_CLIP_RIGHT, F_CLIP_LEFT, F_SPAN_ROWS, F_SPAN_BYTES = 0, 1, 2, 3
F_N = 4

QUAD, TRI_ONE, TRI_TWO = 0, 1, 2
T_BYTE, T_PARITY, T_CORNER = 0, 1, 2  # tiles[] columns: 4 corners x (xlo,xhi,ylo,yhi)
TILE_COLS = T_CORNER + 16


@njit(cache=True)
def _edge(
    left,
    right,
    tab,
    y0hi,
    y0lo,
    y1hi,
    y1lo,
    x0,
    x1,
    xh0,
    xh1,
    dlo,
    rows,
    top,
    bot,
):
    """rasterise_polygon_edge $2EE4: clip the line to the buffer, then walk its DDA into
    the left ($AD00) or right ($AE00) edge table. ``rows`` carries ($0004, $0006, $007F).
    """
    cyc = passcost.EDGE_CALL + passcost.EDGE_HEAD
    rows[R_N_EDGE] += 1
    if y1hi & 0x80:  # $2EF8: the line ends below the inner area
        cyc += passcost.EDGE_BOTTOM_LOW
        rows[0] = bot
    elif y1hi:
        return cyc + passcost.EDGE_ABOVE
    else:
        cyc += passcost.EDGE_BOTTOM_TEST
        if y1lo >= top:
            return cyc + passcost.EDGE_BOTTOM_OFF - 6
        if y1lo >= rows[0]:
            cyc += passcost.EDGE_BOTTOM_KEEP
        elif y1lo >= bot:
            cyc += passcost.EDGE_BOTTOM_SET
            rows[0] = y1lo
        else:
            cyc += passcost.EDGE_BOTTOM_CLIP
            rows[0] = bot
    if y0hi & 0x80:  # $2EFE: the line starts below the bottom of the inner area
        return cyc + passcost.EDGE_TOP_OFF
    if y0hi:
        cyc += passcost.EDGE_TOP_HIGH
        rows[1] = (top - 1) & 0xFF
    else:
        cyc += passcost.EDGE_TOP_TEST
        if y0lo < bot:
            return cyc + passcost.EDGE_TOP_BELOW
        if y0lo < rows[1]:
            cyc += passcost.EDGE_TOP_KEEP
        elif y0lo < top:
            cyc += passcost.EDGE_TOP_SET
            rows[1] = y0lo
        else:
            cyc += passcost.EDGE_TOP_CLIP
            rows[1] = (top - 1) & 0xFF
    cyc += passcost.EDGE_WIDE_TEST
    if xh0 | xh1:
        return (
            cyc
            + passcost.EDGE_WIDE
            + _wide_edge(left, right, tab, y0hi, y0lo, x0, x1, xh0, xh1, dlo, rows)
        )
    if x0 >= x1:  # $2F2F: the line runs right to left
        dx = (x0 - x1) & 0xFF
        step = -1
        cyc += passcost.EDGE_DX
    else:
        dx = (x1 - x0) & 0xFF
        step = 1
        cyc += passcost.EDGE_DX_NEG
    cyc += passcost.EDGE_SLOPE
    tabv = right if tab else left
    row = y0lo
    x = x0
    if dx < dlo:  # steep: at most one column step per row
        rows[R_N_STEEP] += 1
        cyc += passcost.STEEP_SETUP
        n = (dlo + 1) & 0xFF
        acc = ((dlo >> 1) ^ 0xFF) & 0xFF
        carry = 0
        if y0hi:  # $2FB6: run the rows above the inner area without storing
            cyc += passcost.OUTSIDE_ENTRY
            row = (row + 1) & 0xFF
            while True:
                row = (row - 1) & 0xFF
                cyc += passcost.OUTSIDE_STEEP_ROW
                if row == 0:
                    cyc += passcost.OUTSIDE_STEEP_BACK
                    row = 0xFF
                    break
                n -= 1
                if n == 0:
                    return cyc + passcost.OUTSIDE_STEEP_END
                s = acc + dx + carry
                carry = 1 if s > 0xFF else 0
                acc = s & 0xFF
                if carry:
                    s = acc - dlo
                    carry = 1 if s >= 0 else 0
                    acc = s & 0xFF
                    x = (x + step) & 0xFF
        while True:
            tabv[row] = x
            row = (row - 1) & 0xFF
            cyc += passcost.STEEP_ROW
            if row == 0:
                cyc += passcost.STEEP_BOTTOM
                rows[2] = 0
                break
            n -= 1
            if n == 0:
                cyc += passcost.STEEP_LAST
                rows[2] = 0
                break
            rows[R_STEEP] += 1  # $2F58 process_narrow_steep_line_loop
            s = acc + dx + carry
            carry = 1 if s > 0xFF else 0
            acc = s & 0xFF
            if carry:
                cyc += passcost.STEEP_COLUMN
                s = acc - dlo
                carry = 1 if s >= 0 else 0
                acc = s & 0xFF
                x = (x + step) & 0xFF
        return cyc
    rows[R_N_SHALLOW] += 1
    cyc += passcost.SHALLOW_SETUP  # shallow: at most one row step per column
    store_flat = (step < 0) if tab == 0 else (step > 0)
    n = (dx + 1) & 0xFF
    acc = ((dx >> 1) ^ 0xFF) & 0xFF
    carry = 0
    tabv[row] = x
    cyc += passcost.SHALLOW_STORE
    while True:
        n -= 1
        if n == 0:
            cyc += passcost.SHALLOW_LAST
            rows[2] = 0
            break
        cyc += passcost.SHALLOW_STEP  # $2FA1 the shallow loop's own column step
        rows[R_SHALLOW] += 1
        x = (x + step) & 0xFF
        s = acc + dlo + carry
        carry = 1 if s > 0xFF else 0
        acc = s & 0xFF
        if carry:
            cyc += passcost.SHALLOW_ROW
            s = acc - dx
            carry = 1 if s >= 0 else 0
            acc = s & 0xFF
            row = (row - 1) & 0xFF
            if row == 0:
                cyc += passcost.SHALLOW_BOTTOM
                rows[2] = 0
                break
            tabv[row] = x
            cyc += passcost.SHALLOW_STORE
        elif store_flat:
            tabv[row] = x
            cyc += passcost.SHALLOW_STORE
    return cyc


@njit(cache=True)
def _wide_area(area_v, area_h):
    """set_wide_line_store_ops_and_edge $31A4: what this area stores, and its cycles.

    Returns (mode, cycles): -1 nothing, 0 the line's own column, 1 the area's edge.
    """
    if area_v:
        return -1, passcost.WIDE_AREA_CALL + passcost.WIDE_AREA_OUT
    if area_h == 0:
        return 0, passcost.WIDE_AREA_CALL + passcost.WIDE_AREA
    if area_h & 0x80:
        return 1, passcost.WIDE_AREA_CALL + passcost.WIDE_AREA_LEFT
    return 2, passcost.WIDE_AREA_CALL + passcost.WIDE_AREA_RIGHT


@njit(cache=True)
def _wide_edge(left, right, tab, y0hi, y0lo, x0, x1, xh0, xh1, dlo, rows):
    """process_wide_line $30BD: the DDA for a line that leaves the inner area, whose
    store is self-modified per area to skip the row, plot the line, or plot the edge."""
    cyc = passcost.WIDE_HEAD + passcost.WIDE_DIR + passcost.WIDE_SLOPE
    dxl = (x0 - x1) & 0xFF
    dxh = (xh0 - xh1 - (1 if x0 < x1 else 0)) & 0xFF
    if dxh & 0x80:  # $30CA: ensure the delta is positive and step right
        dx = (-((dxh << 8) | dxl)) & 0xFF
        step, wrap = 1, 0x00
    else:
        dx = dxl
        step, wrap = -1, 0xFF
    tabv = right if tab else left
    area_v = y0hi
    area_h = xh0
    mode, acyc = _wide_area(area_v, area_h)
    cyc += acyc
    row = (y0lo + (1 if area_v else 0)) & 0xFF
    x = x0
    steep = dx < dlo
    if steep:
        rows[R_N_WSTEEP] += 1
        cyc += passcost.WIDE_STEEP_SETUP
        n = (dlo + 1) & 0xFF
        acc = ((dlo >> 1) ^ 0xFF) & 0xFF
    else:
        rows[R_N_WSHALLOW] += 1
        cyc += passcost.WIDE_SHALLOW_SETUP
        n = (dx + 1) & 0xFF
        acc = ((dx >> 1) ^ 0xFF) & 0xFF
    guard = 0
    while guard < 4096:
        guard += 1
        if mode == 0:  # $311F/$317E STX: the line's own column
            tabv[row] = x
            rows[2] = 0
        elif mode > 0:  # $8C STY: the area edge, 0 to the left and $FF to the right
            tabv[row] = 0 if mode == 1 else 0xFF
            rows[2] = 0
        if steep:
            cyc += passcost.WIDE_STEEP_ROW
            row = (row - 1) & 0xFF
            if row == 0:  # $3125 into move_to_next_area_of_wide_steep_line
                cyc += passcost.WIDE_STEEP_AREA_V
                area_v = (area_v - 1) & 0xFF
                if area_v & 0x80:
                    cyc += passcost.WIDE_LEAVE
                    break
                if area_v == 0:
                    cyc += passcost.WIDE_STEEP_AREA_BACK
                    row = 0xFF
                    mode, acyc = _wide_area(area_v, area_h)
                    cyc += acyc
            n -= 1
            if n == 0:
                cyc += passcost.WIDE_LEAVE
                break
            cyc += passcost.WIDE_STEEP_STEP  # $3113 process_wide_steep_line_loop
            rows[R_WIDE_STEEP] += 1
            s = acc + dx  # $311C CLCs every lap, so no carry comes in
            acc = s & 0xFF
            if s > 0xFF:
                cyc += passcost.WIDE_STEEP_COLUMN
                acc = (acc - dlo) & 0xFF
                x = (x + step) & 0xFF
                if x == wrap:  # $311A CPX $0076: the line crossed into the next area
                    cyc += passcost.WIDE_STEEP_AREA_H
                    area_h = (area_h + step) & 0xFF
                    mode, acyc = _wide_area(area_v, area_h)
                    cyc += acyc
            continue
        n -= 1
        if n == 0:  # $3183 BNE not taken: the line is done, $316D never runs again
            cyc += passcost.WIDE_LEAVE
            break
        cyc += passcost.WIDE_SHALLOW_STEP  # $316D the shallow loop's own column step
        rows[R_WIDE_SHALLOW] += 1
        x = (x + step) & 0xFF
        if x == wrap:
            cyc += passcost.WIDE_SHALLOW_AREA_H
            area_h = (area_h + step) & 0xFF
            mode, acyc = _wide_area(area_v, area_h)
            cyc += acyc
        s = acc + dlo  # $3170 CLCs every lap, so no carry comes in
        acc = s & 0xFF
        if s > 0xFF:
            cyc += passcost.WIDE_SHALLOW_ROW
            acc = (acc - dx) & 0xFF
            row = (row - 1) & 0xFF
            if row == 0:
                cyc += passcost.WIDE_SHALLOW_AREA_V
                area_v = (area_v - 1) & 0xFF
                if area_v & 0x80:
                    cyc += passcost.WIDE_LEAVE
                    break
                if area_v == 0:
                    cyc += passcost.WIDE_SHALLOW_AREA_BACK
                    row = 0xFF
                    mode, acyc = _wide_area(area_v, area_h)
                    cyc += acyc
    return cyc


@njit(cache=True)
def _section_line(vxy, left, right, sxb, sxh, vx, vy, dlo, dhi, tab, rows, top, bot):
    """process_line $3002: halve the line until both deltas fit a byte, then rasterise
    each section in turn."""
    rows[R_N_SECT] += 1
    cyc = passcost.SECTION_HEAD + passcost.SECTION_INVERT + passcost.SECTION_TEST
    dxl = (sxb[vy] - sxb[vx]) & 0xFF
    dxh = (sxh[vy] - sxh[vx] - (1 if sxb[vy] < sxb[vx] else 0)) & 0xFF
    dx16 = dxh * 256 + dxl
    negative = dxh & 0x80
    if negative:  # $3019 invert_A_and_a_fraction_if_negative
        dx16 = (0x10000 - dx16) & 0xFFFF
    dy16 = dhi * 256 + dlo
    nsec = 0
    if dx16 or dy16:
        while max(dx16 >> 8, dy16 >> 8):  # $3022: halve until both fit in a byte
            cyc += passcost.SECTION_HALVE
            dx16 >>= 1
            dy16 >>= 1
            nsec = (nsec * 2 + 1) & 0xFF
        while (dy16 & 0xFF) == 0xFF or (dx16 & 0xFF) == 0xFF:  # $3030 overflow guard
            cyc += passcost.SECTION_HALVE
            dx16 >>= 1
            dy16 >>= 1
            nsec = (nsec * 2 + 1) & 0xFF
    cyc += passcost.SECTION_SCALE + passcost.SECTION_SEED
    y_hi, y_lo = vxy[vy, 3], vxy[vy, 2]
    x, xh = sxb[vy], sxh[vy]
    step_lo, step_hi = dy16 & 0xFF, (dy16 >> 8) & 0xFF
    step = (0x10000 - dx16) & 0xFFFF if negative else dx16  # $303C restores the sign
    sx_lo, sx_hi = step & 0xFF, (step >> 8) & 0xFF
    for _ in range(nsec):
        cyc += passcost.SECTION_STEP + passcost.SECTION_LOOP
        e_lo = (y_lo - step_lo) & 0xFF
        e_hi = (y_hi - step_hi - (1 if y_lo < step_lo else 0)) & 0xFF
        nx = (x - sx_lo) & 0xFF
        nxh = (xh - sx_hi - (1 if x < sx_lo else 0)) & 0xFF
        cyc += (
            _edge(
                left,
                right,
                tab,
                y_hi,
                y_lo,
                e_hi,
                e_lo,
                x,
                nx,
                xh,
                nxh,
                dy16 & 0xFF,
                rows,
                top,
                bot,
            )
            - passcost.EDGE_CALL
        )
        y_hi, y_lo, x, xh = e_hi, e_lo, nx, nxh
    cyc += passcost.SECTION_END + passcost.SECTION_TAIL
    return (
        cyc
        + _edge(
            left,
            right,
            tab,
            y_hi,
            y_lo,
            vxy[vx, 3],
            vxy[vx, 2],
            x,
            sxb[vx],
            xh,
            sxh[vx],
            (y_lo - vxy[vx, 2]) & 0xFF,
            rows,
            top,
            bot,
        )
        - passcost.EDGE_CALL
    )


@njit(cache=True)
def _span_fill(left, right, p04, p06, b35, b61, flags, suppress):
    """span_fill $22AA: walk the polygon's rows $0006 down to $0004, plotting each row's
    two edge bytes and every whole byte between them. ``flags`` carries $002C/$002D."""
    cyc = passcost.SPAN_CALL + passcost.SPAN_ENTRY
    total = p06 + p04
    mid = ((total >> 1) | (0x80 if total > 0xFF else 0)) & 0xFF
    flags[0] = 1
    flags[1] = 1
    if right[mid] < left[mid]:  # the vertices run clockwise: the face points away
        return cyc + passcost.SPAN_BACKFACE
    cyc += passcost.SPAN_ADDRESS + passcost.SPAN_START
    # $22E2 BIT suppress_lines: $8538 sets bit 7 while a distant object is plotted.
    cyc += passcost.SPAN_SUPPRESSED if suppress else passcost.SPAN_COLOURS
    if p06 < p04:
        return cyc + passcost.SPAN_START_EMPTY
    b36 = (b35 + b61) & 0xFF
    row = p06
    while True:
        cyc += passcost.SPAN_ROW_TEST
        flags[F_SPAN_ROWS] += 1
        r, ln = right[row], left[row]
        if r < ln:
            return cyc + passcost.SPAN_ROW_EMPTY
        r_off = 0
        middle = False
        if r < b35:  # $2382: the right edge is left of the buffer
            cyc += passcost.SPAN_OFF_LEFT + passcost.SPAN_EXTENDS
            flags[1] = 0
        else:
            if ((r - b35) & 0xFF) >= b61:  # $2384: right of the buffer, so clip
                cyc += passcost.SPAN_OFF_RIGHT_ROW + passcost.SPAN_CLIP_RIGHT
                r_off = (b61 * 2) & 0xFF
            else:
                cyc += passcost.SPAN_LEFT_EDGE + passcost.SPAN_RIGHT_PIXEL
                cyc += passcost.SPAN_PLOT_RIGHT
                r_off = (((r - b35) & 0xFF) * 2) & 0xF8
            cyc += passcost.SPAN_READ_LEFT
            if ln >= b36:  # $23A6: the left edge is right of the buffer
                cyc += passcost.SPAN_EXTENDS
                flags[0] = 0
            elif ln < b35:  # $2340: clip the row to the left edge of the buffer
                cyc += passcost.SPAN_CLIP_LEFT
                flags[1] = 0
                l_off = 0xF8
                middle = True
            else:
                cyc += passcost.SPAN_LEFT_TEST
                l_off = (((ln - b35) & 0xFF) * 2) & 0xF8
                if l_off >= r_off:  # both edges land in the same byte
                    cyc += passcost.SPAN_SAME_BYTE
                else:
                    cyc += passcost.SPAN_PLOT_LEFT
                    middle = True
            if middle:
                cyc += passcost.SPAN_MIDDLE
                nbytes = ((r_off - l_off) & 0xFF) // 8 - 1
                flags[F_SPAN_BYTES] += nbytes
                cyc += passcost.SPAN_BYTE * nbytes
        if row == p04:
            cyc += passcost.SPAN_LAST
            break
        cyc += passcost.SPAN_NEXT
        cyc += passcost.SPAN_ROW_STEP if row & 7 else passcost.SPAN_NEXT_GROUP
        row -= 1
    return cyc


@njit(cache=True)
def _plot_polygon(
    vxy,
    vlist,
    nv,
    prep_cyc,
    sect,
    bufs,
    top,
    bot,
    left,
    right,
    sxb,
    sxh,
    rows,
    flags,
    suppress,
):
    """plot_polygon $2AA9 for one polygon: prepare, rasterise its lines, span_fill.

    ``vxy[v]`` is vertex ``v``'s (screen_x lo, screen_x hi, screen_y lo, screen_y hi);
    ``vlist[0..nv]`` its $003C vertex list, first and last the same. Returns the cycles
    and the $0010 the pass leaves behind, which the next polygon inherits.
    """
    cyc = 0
    cyc += passcost.PP_HEAD  # plot_polygon $2AA9
    passes = 2 if sect < 2 else 1
    cyc += passcost.PP_TALL if passes == 1 else passcost.PP_WIDE
    for p in range(passes):
        cyc += prep_cyc  # prepare_polygon $2D6C builds the vertex list
        o_hi = bufs[sect, 0]
        cyc += passcost.PREP_DATA_HEAD
        outside = False
        for j in range(nv, -1, -1):
            v = vlist[j]
            hi8 = (vxy[v, 1] + o_hi) & 0xFF
            if hi8 >= 0x20:  # $2DDF: this polygon leaves the inner area
                cyc += passcost.PREP_VERTEX_OUT
                outside = True
                break
            cyc += passcost.PREP_VERTEX_LAST if j == 0 else passcost.PREP_VERTEX
            sxb[v] = (((hi8 * 256 + vxy[v, 0]) << 3) >> 8) & 0xFF
            sxh[v] = 0
        if outside:
            cyc += passcost.PREP_WIDE_HEAD + passcost.PREP_WIDE_JMP
            for j in range(nv, -1, -1):
                v = vlist[j]
                acc = (vxy[v, 1] + o_hi) & 0xFF
                lo = vxy[v, 0]
                carry = 0
                for r in range(3):  # $2DA9: a circular 16-bit rotate left
                    t = lo >> 7
                    lo = ((lo << 1) | (carry if r else 0)) & 0xFF
                    carry = acc >> 7
                    acc = ((acc << 1) | t) & 0xFF
                sxb[v] = acc
                h = (((lo << 1) | carry) & 0xFF) & 7
                if h >= 4:
                    h |= 0xF8
                    cyc += passcost.PREP_WIDE_NEG
                sxh[v] = h
                cyc += passcost.PREP_WIDE_LAST if j == 0 else passcost.PREP_WIDE_VERTEX
        cyc += passcost.LINES_HEAD  # process_lines $2DF2
        rows[R_BOTTOM], rows[R_TOP], rows[R_NOTHING] = 0xFF, 0x00, 0xFF
        p30, p31 = 0xFF, 0x00
        npoint = 0
        lastx = vlist[0]
        for j in range(nv):
            vx, vy = vlist[j], vlist[j + 1]
            lastx = vx
            dlo = (vxy[vy, 2] - vxy[vx, 2]) & 0xFF
            borrow = 1 if vxy[vy, 2] < vxy[vx, 2] else 0
            dhi = (vxy[vy, 3] - vxy[vx, 3] - borrow) & 0xFF
            tab = 0
            cyc += passcost.LINE_HEAD
            if dhi & 0x80:  # $2E20: the line slopes upwards, so swap its ends
                cyc += passcost.LINE_UP
                tab = 1
                vx, vy = vy, vx
                dhi = (-dhi - (1 if dlo else 0)) & 0xFF
                dlo = (-dlo) & 0xFF
            else:
                cyc += passcost.LINE_DOWN
            cyc += passcost.LINE_MID
            wide = False
            if outside and (sxh[vx] | sxh[vy]):
                cyc += passcost.LINE_OUTER
                if dhi:
                    cyc += passcost.LINE_OUTER_STEEP
                elif dlo:
                    cyc += passcost.LINE_OUTER_FLAT
                else:
                    cyc += passcost.LINE_OUTER_POINT + passcost.LINE_NEXT
                    continue
                wide = True
            else:
                cyc += passcost.LINE_OUTER_INNER if outside else passcost.LINE_INNER
                if dhi:  # $2E56: over 255 rows, so section the line
                    cyc += passcost.LINE_STEEP
                    sxh[vx] = 0
                    sxh[vy] = 0
                    wide = True
                else:
                    cyc += passcost.LINE_SHALLOW
                    if not dlo:  # $2ECC line_is_a_point
                        cyc += passcost.LINE_IS_POINT + passcost.POINT_LINE
                        a = sxb[vx]
                        if a >= p31:
                            p31 = a
                            cyc += passcost.POINT_LINE_FLOOR
                        if a < p30:
                            p30 = a
                            cyc += passcost.POINT_LINE_CEIL
                        npoint += 1
                        cyc += passcost.LINE_NEXT
                        continue
                    cyc += passcost.LINE_RASTERISE
            if wide:
                cyc += _section_line(
                    vxy,
                    left,
                    right,
                    sxb,
                    sxh,
                    vx,
                    vy,
                    dlo,
                    dhi,
                    tab,
                    rows,
                    top,
                    bot,
                )
            else:
                cyc += _edge(
                    left,
                    right,
                    tab,
                    vxy[vy, 3],
                    vxy[vy, 2],
                    vxy[vx, 3],
                    vxy[vx, 2],
                    sxb[vy],
                    sxb[vx],
                    0,
                    0,
                    dlo,
                    rows,
                    top,
                    bot,
                )
            cyc += passcost.LINE_NEXT
        cyc += passcost.LINE_LAST - passcost.LINE_NEXT + passcost.LINES_TAIL
        if npoint != nv:
            cyc += passcost.LINES_NOT_POINT
        else:
            cyc += passcost.LINES_ALL_POINTS
            y = vxy[lastx, 2]
            if vxy[lastx, 3] or y < bot or y >= top:
                cyc += passcost.LINES_POINT_OFF
            else:
                cyc += passcost.LINES_POINT_ROW
                rows[0], rows[1], rows[2] = y, y, 0
                left[y], right[y] = p30, p31
        if rows[2]:
            cyc += LINES_LEAVE_FLAG
            nothing = True
        elif rows[1] < rows[0]:
            cyc += LINES_LEAVE_ROWS
            nothing = True
        else:
            cyc += LINES_OK
            nothing = False
        if nothing:
            cyc += passcost.PP_NOTHING
            flags[0], flags[1] = 1, 1
        else:
            cyc += passcost.PP_SPAN
            cyc += _span_fill(
                left,
                right,
                rows[0],
                rows[1],
                bufs[sect, 1],
                bufs[sect, 2],
                flags,
                suppress,
            )
        if passes == 1 or p == 1:
            cyc += passcost.PP_RTS
            break
        cyc += passcost.PP_CLIP_TEST
        if not nothing and flags[sect]:
            cyc += passcost.PP_RTS
            break
        cyc += passcost.PP_CLIPPED + passcost.BUFVARS
        sect ^= 1
    return cyc, sect


@njit(cache=True)
def tile_cycles(tiles, i, s4b, s66, sect, bufs, top, bot, left, right, work):
    """plot_tile $2A24 for one tile of the plot range: its polygons, in render order.

    Returns (cycles, $0010).  An object tile's own stack is $21AE's, not this.
    """
    vxy, sxb, sxh, vlist, rows, flags = work
    tb = tiles[i, T_BYTE]
    cyc = passcost.TILE_SOUND + passcost.TILE_ENTRY
    if tb == 0:
        return cyc + passcost.TILE_EMPTY, sect
    cyc += passcost.TILE_KEPT
    s45 = 0
    kinds = 1
    checker = False
    if tb >= 0xC0:  # $2A2D an object tile: a checkerboard, then $21AE's own stack
        cyc += passcost.TILE_OBJECT
        checker = True
    else:
        cyc += passcost.TILE_GROUND
        nib = tb & 0x0F
        if nib == 0:
            cyc += passcost.TILE_FLAT
            checker = True
        elif nib == 0x0C or nib == 0x04:  # $2A41 a flat and a sloping edge
            cyc += passcost.TILE_EDGE_NIB
            s45 = s4b & 1
            kinds = 2
        else:
            cyc += passcost.TILE_SLOPE_CALC
            yidx = (nib - s66) & 0x0F
            if (yidx & 3) == 1:  # $2A74 a quadrilateral after all
                cyc += passcost.TILE_SLOPE_QUAD
                checker = True
            else:
                cyc += passcost.TILE_SLOPE_TRI
                s45 = SLOPE_FLAG[yidx]
                kinds = 2
                if (((nib & 4) >> 2) + s4b) & 0xFF < 2:
                    cyc += passcost.TILE_SLOPE_TRI_ALT
    if checker:
        cyc += passcost.CHECKER_HEAD + passcost.QUAD_TILE
        cyc += passcost.CHECKER_ODD if tiles[i, T_PARITY] else passcost.CHECKER_EVEN
    for v in range(4):
        b = T_CORNER + 4 * v
        vxy[v, 0] = tiles[i, b]
        vxy[v, 1] = tiles[i, b + 1]
        vxy[v, 2] = tiles[i, b + 2]
        vxy[v, 3] = tiles[i, b + 3]
    for k in range(kinds):
        kind = QUAD
        if not checker:
            kind = TRI_ONE if k == 0 else TRI_TWO
            cyc += passcost.TWO_TRI_FIRST if k == 0 else passcost.TWO_TRI_SECOND
        if kind == QUAD:
            vlist[0], vlist[1], vlist[2], vlist[3], vlist[4] = 0, 1, 2, 3, 0
            nv = 4
            prep = passcost.PREP_HEAD + passcost.PREP_QUAD
        else:
            prep = passcost.PREP_HEAD + passcost.PREP_TRI_HEAD + passcost.PREP_TRI_TAIL
            if kind == TRI_ONE:
                prep += passcost.PREP_TRI_ONE
                vlist[0], vlist[1] = 0, 1
                vlist[2] = TRI_THIRD[0, s45]
            else:
                prep += passcost.PREP_TRI_TWO
                vlist[0], vlist[1] = 2, 3
                vlist[2] = TRI_THIRD[1, s45]
            vlist[3] = vlist[0]
            nv = 3
        c, sect = _plot_polygon(
            vxy,
            vlist,
            nv,
            prep,
            sect,
            bufs,
            top,
            bot,
            left,
            right,
            sxb,
            sxh,
            rows,
            flags,
            0,
        )
        cyc += c
    return cyc, sect


def edge_tables():
    """The never-cleared $AD00/$AE00 pair a whole plot_world pass shares."""
    return np.zeros(256, dtype=np.int64), np.zeros(256, dtype=np.int64)


def tile_workspace():
    """Scratch for the terrain side of one pass: four corners and the $2D6C list."""
    return (
        np.zeros((4, 4), dtype=np.int64),
        np.zeros(4, dtype=np.int64),
        np.zeros(4, dtype=np.int64),
        np.zeros(5, dtype=np.int64),
        np.zeros(R_N, dtype=np.int64),
        np.zeros(F_N, dtype=np.int64),
    )


@njit(cache=True)
def fill_cycles(tiles, ntiles, s4b, s66, sect0, bufs, top, bot):
    """Cycles plot_world spends between $2A24 and $22AA over ``tiles`` in render order.

    ``tiles[i]`` is (tile byte, tile parity, four corners of xlo/xhi/ylo/yhi); ``bufs[s]``
    is buffer ``s``'s ($0011, $0035, $0061); ``sect0`` is $0010 on entry, and a section
    below 2 is the wide play buffer, which plots each polygon in two passes.
    """
    left = np.zeros(256, dtype=np.int64)  # polygon_left_edge_table $AD00
    right = np.zeros(256, dtype=np.int64)  # polygon_right_edge_table $AE00
    work = (
        np.zeros((4, 4), dtype=np.int64),
        np.zeros(4, dtype=np.int64),
        np.zeros(4, dtype=np.int64),
        np.zeros(5, dtype=np.int64),
        np.zeros(R_N, dtype=np.int64),
        np.zeros(F_N, dtype=np.int64),
    )
    cyc = 0
    sect = sect0
    for i in range(ntiles):
        c, sect = tile_cycles(
            tiles, i, s4b, s66, sect, bufs, top, bot, left, right, work
        )
        cyc += c
    return cyc
