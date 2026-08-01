"""plot_world ($2625) terrain render-projector, ported bit-exactly from the ROM.

Walks the 32x32 tile grid furthest->nearest, projecting each grid point ($2845) to a
horizontal angle (screen x) and vertical angle (screen y) with :mod:`sentinel.relative`'s
trig cores; the projected corners feed :func:`render_cost`.
"""

import collections
import os

import numpy as np

from sentinel import passcost, relative, rendercost, terrain, memmap as mm
from sentinel.writeweight import cycles as _plain  # the $8401 chain rides packed

try:
    from sentinel import projector_jit

    _HAVE_JIT = True
except Exception:  # pragma: no cover - numba absent -> pure-Python fallback
    projector_jit = None
    _HAVE_JIT = False

# initialise_buffer_variables ($2993) mode -> ($0007, $0012=($0007>>1)^$80, $0028), table $29C4; mode 0 = play buffer and vertical-pan strip ($9939), mode 2 = horizontal-pan strip ($994F).  $29B4 zeroes $0028 in every mode; only $29C7's odd strip sets it.
BUF_WINDOW = {0: (0x14, 0x8A, 0x00), 1: (0x14, 0x8A, 0x00), 2: (0x08, 0x84, 0x00)}
PLAY_MODE = 0
_BUF_LEFT, _BUF_RIGHT = BUF_WINDOW[PLAY_MODE][:2]
# $0C48 furthest-row extent hint ($26CD): plot_world's own carry between passes, 0 in a
# fresh play state and up to $1F once the game has plotted. RENDER_ROW_HINT overrides it.
_ROW_HINT_ENV = os.environ.get("RENDER_ROW_HINT")


def _neg16(hi, lo):
    """Two's-complement negate the 16-bit (hi:lo), as $2865/$286C do."""
    borrow = 1 if lo != 0 else 0
    return ((-hi - borrow) & 0xFF, (-lo) & 0xFF)


def _setup(state, h_angle, v_angle, observer, mode=PLAY_MODE, window=None, ref_lo=0x00):
    """plot_world setup ($2625-$26D6): the view-orientation case and screen-x
    reference angle from the observer's h_angle, against ``mode``'s $2993 window,
    or ``window`` when a caller sets its own ($29C7, as $1F9F does for a strip).
    ``ref_lo`` is $001F, 0 for a play redraw and $80 for a strip on an odd column."""
    view_angle = (h_angle + 0x20) & 0xFF  # $001C ($265A)
    quadrant = view_angle >> 6  # $2665: top two bits
    folded = ((view_angle & 0x3F) - 0x20) & 0xFF  # $0074 ($265E)
    ox, oy = state.obj_x[observer], state.obj_y[observer]
    if quadrant == 0:  # north ($268A)
        c3, c1d = ox, oy
    elif quadrant == 1:  # east ($2697): CLC before SBC -> extra -1
        c3, c1d = (0x1E - oy) & 0xFF, ox
    elif quadrant == 2:  # south ($26A9)
        c3, c1d = (0x1E - ox) & 0xFF, (0x1E - oy) & 0xFF
    else:  # west ($26BC)
        c3, c1d = oy, (0x1E - ox) & 0xFF
    left, right, frac = window if window is not None else BUF_WINDOW[mode]
    return {
        "observer": observer,
        "quadrant": quadrant,
        "buf_left": left,  # $0007
        "buf_right": right,  # $0012
        "buf_frac": frac,  # $0028
        "c3": c3,  # $0003
        "c1d": c1d,  # $001D
        "ref_lo": ref_lo & 0xFF,  # $001F
        "ref_hi": (folded - 0x0A) & 0xFF,  # $0020 ($267D)
        "h_angle": h_angle,  # objects_h_angle, the bearing being priced
        "v_angle": v_angle,  # objects_v_angle used by $933D
        # $283D BIT $9AF6: bit7 set (all of play) sends every examine to $37F2.
        "table_examine": bool(state.mem[mm.PLAY_DISPLAY_FLAG] & 0x80),
        "row_hint": (
            state.mem[mm.FURTHEST_ROW_HINT]
            if _ROW_HINT_ENV is None
            else int(_ROW_HINT_ENV)
        )
        & 0xFF,
    }


def _tile_xy(quadrant, col, row):
    """($28A3) map (column,row) to (tile_x,tile_y) for the view orientation."""
    if quadrant == 0:
        return col, row
    if quadrant == 1:
        return row, (0x1F - col) & 0xFF
    if quadrant == 2:
        return (0x1F - col) & 0xFF, (0x1F - row) & 0xFF
    return (0x1F - row) & 0xFF, col


def _tile_height(state, tx, ty):
    """($28E9) ground height of tile (tx,ty); object tiles resolve to the
    bottommost object's z_height. Returns (height, tile_byte, stack levels walked)."""
    tb = terrain.tile_byte(state, tx, ty)
    if tb < mm.OBJECT_TILE:
        return tb >> 4, tb, 0
    slot, levels = tb & 0x3F, 1
    for _ in range(mm.NUM_SLOTS):
        flags = state.obj_flags[slot]
        if flags < 0x40:  # $28FA CMP #$40: this level is on the ground
            break
        slot, levels = flags & 0x3F, levels + 1
    return state.obj_z_height[slot], tb, levels


_EXAM_QUAD = (
    passcost.EXAM_QUAD_NORTH,
    passcost.EXAM_QUAD_EAST,
    passcost.EXAM_QUAD_SOUTH,
    passcost.EXAM_QUAD_WEST,
)
_TAB_QUAD = (
    passcost.TAB_QUAD_NORTH,
    passcost.TAB_QUAD_EAST,
    passcost.TAB_QUAD_SOUTH,
    passcost.TAB_QUAD_WEST,
)


def _project(state, setup, col, row, vis=None):
    """$283D: project grid point (col,row) through whichever examine $9AF6 selects.

    Returns (sx_lo, sx_hi, sy_lo, sy_hi, tile_byte, onscreen, cycles). The two examines
    produce identical plottables -- $37F2 reads back what $3700 computed with the same
    trig -- so only the cycles differ, and ``cyc``/``tab`` price them side by side.
    """
    observer = setup["observer"]
    zp = collections.defaultdict(int)
    cyc = passcost.EXAM_CALL + passcost.EXAM_ENTRY_TRIG + passcost.EXAM_HEAD
    tab = passcost.EXAM_CALL + passcost.EXAM_ENTRY_TABLE + passcost.TAB_HEAD
    sx = (col - setup["c3"] - 1) & 0xFF  # signed column ($2858, CLC-before-SBC -1)
    zp[0x86], zp[0x80] = sx, 0x80
    if sx & 0x80:
        zp[0x83], zp[0x80] = _neg16(sx, 0x80)
        cyc += passcost.EXAM_COL_NEG
    else:
        zp[0x83] = sx
        cyc += passcost.EXAM_COL_POS
    sr = (row - setup["c1d"] - 1) & 0xFF  # signed row ($2876)
    zp[0x88], zp[0x82] = sr, 0x80
    cyc += passcost.EXAM_ROW
    if sr & 0x80:
        zp[0x85], zp[0x82] = _neg16(sr, 0x80)
        cyc += passcost.EXAM_ROW_NEG
    else:
        zp[0x85] = sr
        cyc += passcost.EXAM_ROW_POS
    cyc += passcost.EXAM_ANGLE + _plain(relative._calc_angle(zp))  # $9287 -> $8A/$8B
    sx_lo = (zp[0x8A] - setup["ref_lo"]) & 0xFF  # screen x ($2891), carry stays set
    sx_hi = (zp[0x8B] - setup["ref_hi"]) & 0xFF
    cyc += passcost.EXAM_STORE + _plain(
        relative._calc_hypotenuse(zp)
    )  # $937F -> $7C/$7D
    quadrant = setup["quadrant"]
    cyc += _EXAM_QUAD[quadrant] + passcost.EXAM_ADDR
    tab += _TAB_QUAD[quadrant] + passcost.TAB_ADDR
    tx, ty = _tile_xy(quadrant, col, row)
    height, tile_byte, levels = _tile_height(state, tx, ty)
    if levels:  # $28F2/$3876: an object tile skips the raytrace bit
        cyc += passcost.EXAM_OBJECT + passcost.EXAM_OBJECT_BOTTOM
        cyc += (levels - 1) * passcost.EXAM_OBJECT_STEP  # $37F2 walks no stack
        tab += passcost.TAB_OBJECT
    else:
        if vis is None:
            vis = occlusion_visible(state, observer)
        seen = vis[ty][tx]
        cyc += passcost.EXAM_GROUND
        cyc += passcost.EXAM_VISIBLE if seen else passcost.EXAM_HIDDEN
        tab += passcost.TAB_GROUND
        tab += passcost.TAB_VISIBLE if seen else passcost.TAB_HIDDEN
    zf = state.obj_z_frac[observer]  # tile height relative to eye ($291E)
    zp[0x80] = (-zf) & 0xFF
    rel_z_hi = (height - state.obj_z_height[observer] - (1 if zf else 0)) & 0xFF
    sy_hi, vcyc = relative._vertical_angle(zp, rel_z_hi, setup["v_angle"])  # $933D
    sy_lo = zp[0x50]
    cyc += passcost.EXAM_VANGLE + _plain(vcyc) + passcost.EXAM_TAIL
    tab += passcost.TAB_SCREEN + passcost.TAB_TAIL
    onscreen = 0x00
    # $293C/$388F on-screen test against the mode's $0007/$0012/$0028.
    if sx_hi < setup["buf_left"]:
        cyc += passcost.EXAM_OFF_LEFT
        tab += passcost.TAB_OFF_LEFT
    else:
        cyc += passcost.EXAM_ON_LEFT
        tab += passcost.TAB_ON_LEFT
        if sx_hi != setup["buf_left"]:
            cyc += passcost.EXAM_NO_FRACTION
            tab += passcost.TAB_NO_FRACTION
            fraction_left = False
        else:  # $2945/$3895: $29C7's odd strip drops the left column's near half
            cyc += passcost.EXAM_FRACTION
            tab += passcost.TAB_FRACTION
            fraction_left = sx_lo < setup["buf_frac"]
            cyc += (
                passcost.EXAM_FRACTION_LEFT
                if fraction_left
                else passcost.EXAM_FRACTION_OK
            )
            tab += (
                passcost.TAB_FRACTION_LEFT
                if fraction_left
                else passcost.TAB_FRACTION_OK
            )
        if not fraction_left:
            cyc += passcost.EXAM_RIGHT
            tab += passcost.TAB_RIGHT
            if sx_hi < setup["buf_right"]:
                onscreen = 0x80
                cyc += passcost.EXAM_OFF_RIGHT
                tab += passcost.TAB_OFF_RIGHT
            else:
                onscreen = 0x81
                cyc += passcost.EXAM_ON_RIGHT
                tab += passcost.TAB_ON_RIGHT
    spent = tab if setup["table_examine"] else cyc
    return sx_lo, sx_hi, sy_lo, sy_hi, tile_byte, onscreen, spent


_SCREEN_H = int(os.environ.get("RENDER_SCREEN_H", "240"))  # $F0 fillable scanlines
_W_SCALE = int(os.environ.get("RENDER_W_SCALE", "32"))  # angle16 -> screen pixels
_W_SCREEN = int(os.environ.get("RENDER_W_SCREEN", "160"))  # screen width, pixels


def _signed16(hi, lo):
    """A corner's signed 16-bit projected coordinate (prepare_polygon $2F02/$2DD2):
    high byte selects the region/band, low byte the scanline/fine position."""
    return (hi - 256 if hi & 0x80 else hi) * 256 + lo


def _clamp(v, lo, hi):
    """Clamp ``v`` into ``[lo, hi]``."""
    return lo if v < lo else hi if v > hi else v


_LAST = mm.N - 1  # 0x1F
_OFFSET_TO_TILE = (
    0x00,
    0x01,
    0x21,
    0x20,
)  # offset_to_tile_table $27D3, indexed by quadrant


def _scan_visible(state, setup, vis=None, schedule=None):
    """Exact port of find_visible_extent ($27D7) + plot_rows_in_front_of_observer_loop
    ($26DE), the furthest->nearest walk that probes tiles via $2845.

    Returns (n_examine, exam_cycles, walk_cycles, rows, cache), and fills ``schedule``
    with one entry per $295D call carrying the scan cycles that preceded it.
    """
    cache = {}
    exam = [0, 0, 0]
    mark = [0]

    def stage(row):
        """Close the current schedule entry: everything walked since the last one."""
        if schedule is not None:
            spent = exam[1] + exam[2]
            schedule.append({"row": row, "scan": spent - mark[0]})
            mark[0] = spent

    def probe(col, row):
        col &= 0xFF
        r = cache.get((col, row))
        if r is None:
            r = _project(state, setup, col, row, vis)
            cache[(col, row)] = r
        exam[0] += 1
        exam[1] += r[6]
        return r[5]

    def walk(n):
        exam[2] += n

    def find_end(row, col):  # find_end_of_row_loop $27E2
        while True:
            start = col
            walk(passcost.SCAN_END_HEAD)
            if col == _LAST:
                walk(passcost.SCAN_INC_END + passcost.SCAN_END_STOP)
                return start, _LAST
            col += 1
            walk(passcost.SCAN_INC)
            a = probe(col, row)
            if a == 0x81:
                walk(passcost.SCAN_END_WHOLE)
                continue
            if a == 0x80:
                walk(passcost.SCAN_END_CROP)
                return start, col
            walk(passcost.SCAN_END_GAP)
            while True:  # find_first_visible_tile_at_end_loop $27F3
                if col == _LAST:
                    walk(passcost.SCAN_INC_END + passcost.SCAN_GAP_STOP)
                    return start, col
                col += 1
                walk(passcost.SCAN_INC)
                a = probe(col, row)
                if a == 0:
                    walk(passcost.SCAN_GAP_LOOP)
                    continue
                walk(passcost.SCAN_GAP_HIT)
                return start, col

    def start_left(row, end, col):  # find_first_visible_tile_at_start_of_row_loop $2820
        while True:
            if col == 0:
                walk(passcost.SCAN_DEC_END + passcost.SCAN_LEFT_STOP)
                return 0, end
            col -= 1
            walk(passcost.SCAN_DEC)
            if probe(col, row) == 0:
                walk(passcost.SCAN_LEFT_LOOP)
                continue
            walk(passcost.SCAN_LEFT_HIT)
            return col, end

    def crop_right(row, col):  # tile_is_cropped_to_right $27FF
        while True:
            end = col
            walk(passcost.SCAN_CROP_HEAD)
            if col == 0:
                walk(passcost.SCAN_DEC_END + passcost.SCAN_CROP_STOP)
                return 0, end
            col -= 1
            walk(passcost.SCAN_DEC)
            a = probe(col, row)
            walk(passcost.SCAN_CROP_MORE)
            if a == 0x80:
                walk(passcost.SCAN_CROP_AGAIN)
                continue
            if a != 0:  # into_find_first_visible_tile_at_start_of_row_loop $2825
                walk(passcost.SCAN_CROP_EXIT)
                return col, end
            walk(passcost.SCAN_CROP_LEFT)
            return start_left(row, end, col)

    def find_extent(row, hint):  # find_visible_extent $27D7
        col = hint & 0xFF
        walk(passcost.SCAN_HEAD)
        a = probe(col, row)
        if a == 0x80:
            walk(passcost.SCAN_VISIBLE + passcost.SCAN_CROPPED)
            return crop_right(row, col)
        if a != 0:
            walk(passcost.SCAN_VISIBLE + passcost.SCAN_WHOLE)
            return find_end(row, col)
        walk(passcost.SCAN_OFF)
        while True:  # find_first_visible_tile_at_start_loop $2811
            if col == _LAST:  # endRow2 $2818
                walk(passcost.SCAN_INC_END + passcost.SCAN_START_STOP)
                walk(passcost.SCAN_START_SETUP)
                return start_left(row, _LAST, hint & 0xFF)
            col += 1
            walk(passcost.SCAN_INC)
            a = probe(col, row)
            if a == 0:
                walk(passcost.SCAN_START_LOOP)
                continue
            walk(passcost.SCAN_START_HIT + passcost.SCAN_START_SETUP)
            return start_left(row, col, hint & 0xFF)

    c1d, c3 = setup["c1d"], setup["c3"]
    rows = []
    row = _LAST
    walk(
        passcost.WALK_SETUP
        + passcost.WALK_SETUP_QUAD[setup["quadrant"]]
        + (passcost.WALK_SETUP_PITCH if setup["v_angle"] & 0x80 else 0)
        + passcost.WALK_ROWS
        + passcost.ROW_SCAN
    )
    start, end = find_extent(row, setup["row_hint"])
    walk(passcost.WALK_HINT)
    stage(None)  # $2625..$26DD: the setup and the furthest row's own scan
    while True:
        row -= 1
        walk(passcost.ROW_HEAD + passcost.TILE_SOUND)
        if row < 0:
            walk(passcost.ROW_LAST)
            break
        walk(passcost.ROW_MORE)
        if row == c1d:  # consider_plotting_observer_row $276F: last, observer row
            walk(passcost.ROW_OBSERVER + passcost.OBS_ROW_HEAD)
            y = (start + 1) & 0xFF
            if y == c3:  # plot_observer_row $2786: plots the single tile $0037
                walk(passcost.OBS_ROW_START + passcost.OBS_ROW_PLOT)
                probe(start, row)
                probe(y, row)
                rows.append((row, start, (start + 1) & 0xFF))
                stage(row)
            elif (end - 2) & 0xFF == c3:  # $277B: plots the single tile $0038-1
                walk(
                    passcost.OBS_ROW_TEST + passcost.OBS_ROW_END + passcost.OBS_ROW_PLOT
                )
                probe((end - 1) & 0xFF, row)
                probe(end, row)
                rows.append((row, (end - 1) & 0xFF, end))
                stage(row)
            else:  # skip_plotting_observer_row $2793: only the observer tile ($27CE)
                walk(passcost.OBS_ROW_TEST + passcost.OBS_ROW_SKIP)
            # All three branches fall into $2793, whose $2797 INC $0026 probes the observer tile one row NEARER than the loop reached; $279E then plots it only when on screen.
            obs_row = (row + 1) & 0xFF
            probe(c3, obs_row)
            walk(passcost.OBS_TILE + passcost.OBS_TILE_TEST)
            walk(
                passcost.OBS_TILE_ON + passcost.OBS_TILE_TAIL
                if cache[(c3, obs_row)][3] < 2
                else passcost.OBS_TILE_OFF
            )
            break
        walk(passcost.ROW_NEAR + passcost.ROW_SCAN)
        p_start, p_end = start, end
        start, end = find_extent(row, p_start)
        if start < p_start:  # this_row_starts_before $2713
            walk(passcost.ROW_START_BEFORE + passcost.ROW_START_BEFORE_TAIL)
            y = (p_start - 1) & 0xFF
            probe(y, row)
            while y != start:
                y = (y - 1) & 0xFF
                walk(passcost.ROW_EXTRA)
                probe(y, row)
            walk(passcost.ROW_EXTRA_LAST)
        elif start > p_start:  # calculate_this_row_new_first_tiles $2709
            walk(passcost.ROW_START_AFTER + passcost.ROW_START_AFTER_TAIL)
            y = (start - 1) & 0xFF
            probe(y, row)
            while y != p_start:
                y = (y - 1) & 0xFF
                walk(passcost.ROW_EXTRA)
                probe(y, row)
            walk(passcost.ROW_EXTRA_LAST)
        else:
            walk(passcost.ROW_START_SAME)
        if end > p_end:  # this_row_ends_after $2741
            walk(passcost.ROW_END_AFTER + passcost.ROW_END_AFTER_TAIL)
            y = p_end
            while True:
                y = (y + 1) & 0xFF
                probe(y, row)
                if y == end:
                    break
                walk(passcost.ROW_EXTRA)
            walk(passcost.ROW_EXTRA_LAST)
        elif end < p_end:  # calculate_this_row_new_last_tiles $2737
            walk(passcost.ROW_END_BEFORE + passcost.ROW_END_BEFORE_TAIL)
            y = end
            while True:
                y = (y + 1) & 0xFF
                probe(y, row)
                if y == p_end:
                    break
                walk(passcost.ROW_EXTRA)
            walk(passcost.ROW_EXTRA_LAST)
        else:
            walk(passcost.ROW_END_SAME)
        walk(passcost.ROW_PLOT)
        rows.append((row, min(start, p_start), max(end, p_end)))
        stage(row)
    stage(None)  # $27CE the observer's own tile, and $26FC's exit
    return exam[0], exam[1], exam[2], rows, cache


def _occlusion_visible_py(state, observer=None):
    """Byte-exact port of populate_tile_visibility_bit_table ($245B): the raytraced
    ``$3E80``/``$24DA`` bitmap $2845 consults at $2911-$2919. ``visible[ty][tx]`` is
    True iff tile (tx,ty) is unoccluded; object tiles ($28F0) bypass it (terrain-only gate).
    Rays start at ``observer`` (the viewpoint object $0C63), defaulting to the player.
    """
    n = mm.N
    p = state.player if observer is None else observer
    objx, objy = state.obj_x[p], state.obj_y[p]
    objz, zfrac = state.obj_z_height[p], state.obj_z_frac[p]
    tz = [[0] * n for _ in range(n)]  # $25C4: (z<<1)|not_flat per tile
    for y in range(n):
        for x in range(n):
            z, slope = terrain.resolve_ground(state, x, y)
            tz[y][x] = ((z << 1) | (1 if slope else 0)) & 0xFF
    mz = [[0] * n for _ in range(n)]  # $25ED: min of the tile's 4 corner bytes, >>1
    for y in range(0x1E, -1, -1):
        for x in range(0x1E, -1, -1):
            b = tz[y][x]
            mz[y][x] = (
                (b >> 1)
                if not (b & 1)
                else (min(b, tz[y][x + 1], tz[y + 1][x + 1], tz[y + 1][x]) >> 1)
            )

    def maxz(row, xi):  # ($72),Y horizon lookup; off-table reads hit zeroed RAM
        return mz[row][xi] if 0 <= row <= 0x1E and 0 <= xi <= 0x1E else 0

    def trace(ty, tx):  # $24E2: ray-march observer->tile, True if unobstructed
        tile = (tx, tz[ty][tx] >> 1, ty)
        olo = (0x80, zfrac, 0x80)
        ohi = (objx, objz, objy)
        d = [0, 0, 0]
        hi = [0, 0, 0]
        ext = [0, 0, 0]
        maxd = 0
        for k in (2, 1, 0):  # per-axis signed delta ($2503), track max |hi|
            lo = (0 - olo[k]) & 0xFF
            cin = 1 if olo[k] == 0 else 0
            h = (tile[k] - ohi[k] - (1 - cin)) & 0xFF
            d[k], hi[k] = lo, h
            if h & 0x80:
                ext[k] = 0xFF
                nb = 1 if lo != 0 else 0
                a = (0 - h - nb) & 0xFF
            else:
                a = h
            if a >= maxd:
                maxd = a
        if ((maxd << 2) & 0xFF) < 6:  # $252A: within ~1 tile => visible
            return True
        step = 0xFF
        a = (maxd << 2) & 0xFF  # $2532 scale: ~2-4 substeps per tile
        while True:
            for k in range(3):
                c = (d[k] >> 7) & 1
                d[k] = (d[k] << 1) & 0xFF
                hi[k] = ((hi[k] << 1) | c) & 0xFF
            step >>= 1
            carry = (a >> 7) & 1
            a = (a << 1) & 0xFF
            if carry:
                break
        ax_lo, ax_int = 0x80, objx  # $37/$3A
        ay_lo, ay_row = 0x80, (objy + 0x40) & 0xFF  # $39/$73
        az_lo, az_mid, az_hi = 0, zfrac, objz  # $35/$38/$3B
        for _ in range(step):  # $2576 march; blocked when ray dips below horizon
            ax_lo += hi[0]
            ax_int = (ax_int + ext[0] + (ax_lo >> 8)) & 0xFF
            ax_lo &= 0xFF
            ay_lo += hi[2]
            ay_row = (ay_row + ext[2] + (ay_lo >> 8)) & 0xFF
            ay_lo &= 0xFF
            az_lo += d[1]
            az_mid += hi[1] + (az_lo >> 8)
            az_lo &= 0xFF
            az_hi = (az_hi + ext[1] + (az_mid >> 8)) & 0xFF
            az_mid &= 0xFF
            if az_hi < maxz((ay_row - 0x40) & 0xFF, ax_int):
                return False
        return True

    raw = [[trace(ty, tx) for tx in range(n)] for ty in range(n)]
    vis = [[False] * n for _ in range(n)]
    for y in range(0x1E, -1, -1):  # $248A combine: 2x2 raytrace dilation AND height
        for x in range(0x1E, -1, -1):
            b = tz[y][x]
            height_ok = (
                bool(b & 1) or (b >> 1) <= objz
            )  # hidden only if flat, above eye
            block = raw[y][x] or raw[y][x + 1] or raw[y + 1][x] or raw[y + 1][x + 1]
            vis[y][x] = block and height_ok
    return vis


def _occlusion_tables(state, observer):
    """One $245B raytrace as both shapes: (list-of-lists of bool, uint8[32,32])."""
    if not _HAVE_JIT:
        ref = _occlusion_visible_py(state, observer)
        return ref, np.array(ref, dtype=np.uint8)
    arr = projector_jit.occlusion_table(
        np.frombuffer(state.mem, dtype=np.uint8), observer
    )
    return [[bool(v) for v in row] for row in arr], arr


def _occlusion_visible(state, observer=None):
    """:func:`_occlusion_visible_py` via the numba twin (:mod:`sentinel.projector_jit`)
    when numba is present; the two are element-for-element identical
    (``tests/test_projector_jit.py``).  Returns a list-of-lists of bools either way."""
    obs = state.player if observer is None else observer
    return _occlusion_tables(state, obs)[0]


# Every byte plot_world reads: tiles_table ($0400) + the object flags/v_angle ($0100) and x/z/y/h_angle/z_frac/type ($0900) arrays + the $283D examine flag and the $26CD row hint -- 1538 bytes, ~1us to digest.
_SCENE_SPANS = (
    (0x0400, 0x0800),
    (0x0100, 0x0180),
    (0x0900, 0x0A80),
    (mm.PLAY_DISPLAY_FLAG, mm.PLAY_DISPLAY_FLAG + 1),
    (mm.FURTHEST_ROW_HINT, mm.FURTHEST_ROW_HINT + 1),
)
_CACHE_MAX = int(os.environ.get("RENDER_CACHE_MAX", "20000"))
_OCCLUSION_CACHE = {}
_COST_CACHE = {}


def scene_key(state):
    """Digest of every byte :func:`project_scene` reads: a sound memo key over a
    mutating ``State`` (creates, absorbs and transfers all land in these spans)."""
    mem = state.mem
    return hash(b"".join(bytes(mem[lo:hi]) for lo, hi in _SCENE_SPANS))


def memo(cache, key, cap, make):
    """Bounded memo: clear wholesale at ``cap`` rather than track an LRU, since a
    search walks scene keys forward and stale entries rarely return."""
    hit = cache.get(key)
    if hit is None:
        if len(cache) >= cap:
            cache.clear()
        hit = cache[key] = make()
    return hit


def _occlusion_memo(state, obs):
    """:func:`_occlusion_tables` memoized on (scene, observer): the $245B table is
    view-independent, so one raytrace serves every bearing at an observer."""
    return memo(
        _OCCLUSION_CACHE,
        (scene_key(state), obs, state.obj_x[obs], state.obj_y[obs]),
        _CACHE_MAX,
        lambda: _occlusion_tables(state, obs),
    )


def occlusion_visible(state, observer=None):
    """The memoized $245B table as a list-of-lists of bools."""
    return _occlusion_memo(state, state.player if observer is None else observer)[0]


def _project_scene_py(state, setup, observer, schedule=None):
    """The pure-Python tile-selection body of :func:`project_scene`."""
    visible = occlusion_visible(state, observer)
    n_examine, exam_cycles, walk_cycles, rows, cache = _scan_visible(
        state, setup, visible, schedule
    )
    staged = iter(
        [] if schedule is None else [e for e in schedule if e["row"] is not None]
    )

    def proj(col, row):
        col &= 0xFF
        cached = cache.get((col, row))
        if cached is None:
            cached = _project(state, setup, col, row, visible)
            cache[(col, row)] = cached
        return cached

    s1b = _OFFSET_TO_TILE[setup["quadrant"]]
    # plot_tile ($2A24) reads $0180 slot (($0025|$0005)+$001B)&$3F: drawn tile is examine (col+offc,row+offr); $001B=$27D3[quad], bit0=col, bit5=bank(row+1). prepare_polygon $2D6C takes $0025|$0005 instead, so a polygon's corners are (col,row) and its neighbours, not the offset slot's.
    offc, offr = s1b & 1, (s1b >> 5) & 1
    tiles = []
    fill = np.zeros(
        (sum(hi - lo for _r, lo, hi in rows), rendercost.TILE_COLS), np.int64
    )
    nfill = 0
    c3 = setup["c3"]
    for row, lo, hi in rows:
        re = (row + offr) & 0xFF
        # plot_row_of_tiles_or_block $295D: up from $0037 to $0003, then down from $0038.
        nfront = max(0, min(hi, c3) - lo)
        order = list(range(lo, lo + nfront)) + list(range(hi - 1, max(lo, c3) - 1, -1))
        per_col = [
            passcost.PLOT_ROW_FRONT if i < nfront else passcost.PLOT_ROW_BACK
            for i in range(len(order))
        ]
        row_tail = passcost.PLOT_ROW_DONE
        if nfront != hi - lo:  # $2963: else the front half ran to $0038 and left
            row_tail = passcost.PLOT_ROW_TURN
            if max(lo, c3) == 0:  # $2978 BMI: the descent ran off the bottom
                row_tail += passcost.PLOT_ROW_BACK_EMPTY
            elif c3 <= lo:  # $297E BCC: it reached $0037 before $0003
                row_tail += passcost.PLOT_ROW_BACK_LOW
            else:
                row_tail += passcost.PLOT_ROW_BACK_END
        walk_cycles += passcost.PLOT_ROW_HEAD + sum(per_col) + row_tail
        if schedule is not None:
            next(staged).update(
                cols=order,
                per_col=per_col,
                head=passcost.PLOT_ROW_HEAD,
                tail=row_tail,
                n0=nfill,
            )
        for col in order:
            ce = (col + offc) & 0xFF
            res = proj(ce, re)
            tb = res[4]
            tx, ty = _tile_xy(setup["quadrant"], ce, re)
            if tb and tb < mm.OBJECT_TILE and not visible[ty][tx]:
                tb = 0  # $291B zeroes $0180 for hidden non-object tiles
            fill[nfill, rendercost.T_BYTE] = tb
            fill[nfill, rendercost.T_PARITY] = (col ^ row) & 1
            for k, (cc, rr) in enumerate(
                ((col, row), (col, row + 1), (col + 1, row + 1), (col + 1, row))
            ):
                q = proj(cc, rr)
                b = rendercost.T_CORNER + 4 * k
                fill[nfill, b : b + 4] = q[0], q[1], q[2], q[3]
            nfill += 1
            if tb == 0:  # $0180 slot zero: nothing to plot ($2A27 BEQ)
                continue
            c1, r1 = min(ce + 1, _LAST), min(re + 1, _LAST)
            corners = (res, proj(c1, re), proj(ce, r1), proj(c1, r1))
            ys = [_signed16(c[3], c[2]) for c in corners]
            xs = [_signed16(c[1], c[0]) for c in corners]
            top = _clamp(min(ys), 0, _SCREEN_H)
            bot = _clamp(max(ys), 0, _SCREEN_H)
            span = (max(xs) - min(xs)) / _W_SCALE
            tiles.append(
                {
                    "col": ce,
                    "row": re,
                    "tile": (tx, ty),
                    "sx_lo": res[0],
                    "sx_hi": res[1],
                    "sy_lo": res[2],
                    "sy_hi": res[3],
                    "tile_byte": tb,
                    "onscreen": res[5],
                    "h": bot - top,
                    "w": max(min(span, _W_SCREEN), 0),
                }
            )
    return tiles, n_examine, exam_cycles, fill[:nfill], walk_cycles


def _project_scene_jit(state, setup, observer):
    """:func:`_project_scene_py` via :func:`sentinel.projector_jit.project_scene`; the
    env-tunable geometry constants are passed in so they stay monkeypatchable."""
    su = np.zeros(projector_jit.S_N, dtype=np.int64)
    su[projector_jit.S_QUAD] = setup["quadrant"]
    su[projector_jit.S_C3] = setup["c3"]
    su[projector_jit.S_C1D] = setup["c1d"]
    su[projector_jit.S_REF_LO] = setup["ref_lo"]
    su[projector_jit.S_REF_HI] = setup["ref_hi"]
    su[projector_jit.S_VANGLE] = setup["v_angle"]
    su[projector_jit.S_OBS_ZH] = state.obj_z_height[observer]
    su[projector_jit.S_OBS_ZF] = state.obj_z_frac[observer]
    su[projector_jit.S_LEFT] = setup["buf_left"]
    su[projector_jit.S_RIGHT] = setup["buf_right"]
    su[projector_jit.S_TABLE] = 1 if setup["table_examine"] else 0
    su[projector_jit.S_FRAC] = setup["buf_frac"]
    out, spans, n, n_examine, exam_cycles, fill, walk = projector_jit.project_scene(
        np.frombuffer(state.mem, dtype=np.uint8),
        su,
        _occlusion_memo(state, observer)[1],
        setup["row_hint"],
        _SCREEN_H,
        _W_SCALE,
    )
    # .tolist() once: the columns come back as Python ints/floats, as the reference has;
    # the width clamp stays here, applied by the same expression the reference uses.
    tiles = [
        {
            "col": r[0],
            "row": r[1],
            "tile": (r[2], r[3]),
            "sx_lo": r[4],
            "sx_hi": r[5],
            "sy_lo": r[6],
            "sy_hi": r[7],
            "tile_byte": r[8],
            "onscreen": r[9],
            "h": r[10],
            "w": max(min(span, _W_SCREEN), 0),
        }
        for r, span in zip(out[:n].tolist(), spans[:n].tolist())
    ]
    return tiles, int(n_examine), int(exam_cycles), fill, int(walk)


def project_scene(
    state, h_angle, v_angle, observer=None, mode=PLAY_MODE, window=None, ref_lo=0x00
):
    """Return (tiles, n_examine, exam_cycles, fill, walk_cycles) under ``mode``'s window.

    The plotted tile set and the $2845 examination count and cycles are all exact.
    Non-object tiles the occlusion table hides are examined but dropped from ``tiles``;
    ``fill`` keeps every tile of the plot range in render order for :mod:`rendercost`.
    """
    if observer is None:
        observer = state.player
    setup = _setup(
        state, h_angle & 0xFF, v_angle & 0xFF, observer, mode, window, ref_lo
    )
    if _HAVE_JIT:
        return _project_scene_jit(state, setup, observer)
    return _project_scene_py(state, setup, observer)


_OBJECT_MODEL_CACHE = []


def _object_backend():
    """(model tables, scratch) for the $21AE/$8533 emulation, or None.

    The model geometry lives in the local game image; without it, or without numba,
    the object term falls back to :func:`_inview_object_base`'s floor.
    """
    if not _OBJECT_MODEL_CACHE:
        try:
            from sentinel import objectcost, objmodel

            if not objmodel.available() or not _HAVE_JIT:
                _OBJECT_MODEL_CACHE.append(None)
            else:
                _OBJECT_MODEL_CACHE.append(
                    (
                        objectcost,
                        objectcost.model_arrays(),
                        objectcost.scratch_zp(),
                        objectcost.workspace(),
                        rendercost.tile_workspace(),
                    )
                )
        except ImportError:  # pragma: no cover - numba absent
            _OBJECT_MODEL_CACHE.append(None)
    return _OBJECT_MODEL_CACHE[0]


def fill_frames(state, setup, fill, mode, observer, columns=None, rows=None):
    """One plot_world pass's fill cycles: the terrain sequence, and the object stacks
    too when the game image is present ($21AE/$8533), else terrain only."""
    bufs, sect = rendercost.buffers(mode, columns)
    top, bot = rows if rows else (rendercost.SCREEN_TOP, rendercost.SCREEN_BOTTOM)
    args = (
        len(fill),
        (setup["quadrant"] - 2) & 0xFF,  # $004B ($267B)
        (setup["quadrant"] * 4) & 0xFF,  # $0066 ($2673)
        sect,
        bufs,
        top,
        bot,
    )
    backend = _object_backend()
    if backend is None:
        return rendercost.fill_cycles(fill, *args), False
    objectcost, model, zp, obj_work, tile_work = backend
    left, right = rendercost.edge_tables()
    return (
        objectcost.pass_cycles(
            np.frombuffer(state.mem, dtype=np.uint8),
            zp,
            model,
            fill,
            *args,
            observer,
            state.player,
            setup["h_angle"],
            setup["v_angle"],
            tile_work,
            obj_work,
            left,
            right,
        ),
        True,
    )


FRAME_CYCLES = 19656.0  # PAL frame
BASE_CYCLES = float(os.environ.get("RENDER_BASE_CYCLES", "0"))
# term (c) plot_object ($8533) base floor (docs/architecture.md); object fill = residual.
C_VERTEX = float(os.environ.get("RENDER_C_VERTEX", "2200"))  # transform_vertex trig
C_PREP_CALL = float(os.environ.get("RENDER_C_PREP_CALL", "625"))  # off-band prepare
SECTIONS = int(os.environ.get("RENDER_SECTIONS", "2"))  # wide play buffer ($2AAB)
# type ($004C) -> (vertices, polygons) from model tables $9CA0/$9CA1, $9CAB/$9CAC.
_OBJECT_MODEL = {
    0: (29, 27),
    1: (22, 25),
    2: (17, 15),
    3: (8, 10),
    4: (18, 25),
    5: (30, 35),
    6: (12, 11),
    7: (8, 4),
}


def _inview_object_base(state, tiles):
    """:func:`_object_base_cycles` over a :func:`project_scene` tile list."""
    return _object_base_cycles(state, [t["tile_byte"] for t in tiles])


def _object_base_cycles(state, tile_bytes):
    """plot_object base cost summed over objects on the plotted object-tiles ($21AE
    stack walk down the $0100 flags chain). Unknown types contribute nothing."""
    total = 0.0
    for tb in tile_bytes:
        if tb < mm.OBJECT_TILE:
            continue
        slot = tb & 0x3F
        for _ in range(mm.NUM_SLOTS):
            model = _OBJECT_MODEL.get(state.obj_type[slot])
            if model is not None:
                nv, npoly = model
                total += nv * C_VERTEX + npoly * SECTIONS * C_PREP_CALL
            flags = state.obj_flags[slot]
            if flags < 0x40:  # bottommost object, on the ground
                break
            slot = flags & 0x3F
    return total


_EXACT_WARNED = [False]


def _exact_render_cost(state, h, v, observer):
    """The py65 exact backend when ``RENDER_COST_BACKEND=py65`` selects it and the ROM
    is present, else None (warn once and fall back to the proxy). Player-view only."""
    if os.environ.get("RENDER_COST_BACKEND", "proxy").lower() != "py65":
        return None
    if observer is not None and observer != state.player:
        return None
    try:
        from sentinel.tests import oracle

        if not oracle.available():
            raise FileNotFoundError("ROM fixture absent")
        from sentinel import rendercost_py65

        return rendercost_py65.render_cost_exact(state, h, v)
    except (ImportError, FileNotFoundError) as exc:
        if not _EXACT_WARNED[0]:
            _EXACT_WARNED[0] = True
            print(f"RENDER_COST_BACKEND=py65 unavailable ({exc}); using proxy")
        return None


def render_cost(state, view, observer=None, mode=PLAY_MODE, window=None, rows=None):
    """One plot_world pass in PAL frames (docs/architecture.md): examine floor +
    terrain/object prepare_polygon floors + the area fill proxy, into ``mode``'s $2993
    buffer. ``view`` maps ``h_angle``/``v_angle``; 0.0 if none.
    ``RENDER_COST_BACKEND=py65`` (ROM present) replaces the proxy for the play buffer.
    """
    if not view or view.get("h_angle") is None:
        return 0.0
    h = view["h_angle"] & 0xFF
    v = (view.get("v_angle") or 0) & 0xFF
    ref_lo = (view.get("ref_lo") or 0) & 0xFF  # $001F, $1FD0 on an odd-column strip
    if mode == PLAY_MODE and window is None and not ref_lo:
        exact = _exact_render_cost(state, h, v, observer)
        if exact is not None:
            return exact
    obs = state.player if observer is None else observer

    def make():
        setup = _setup(state, h, v, obs, mode, window, ref_lo)
        tiles, _n, exam_cycles, fill, walk = project_scene(
            state, h, v, obs, mode, window, ref_lo
        )
        columns = None if window is None else _window_columns(window)
        fill_cycles, exact_objects = fill_frames(
            state, setup, fill, mode, obs, columns, rows
        )
        base = 0.0 if exact_objects else _inview_object_base(state, tiles)
        return (BASE_CYCLES + exam_cycles + walk + fill_cycles + base) / FRAME_CYCLES

    key = (scene_key(state), obs, h, v, mode, window, rows, ref_lo)
    return memo(_COST_CACHE, key, _CACHE_MAX, make)


def _window_columns(window):
    """The $0C69 column count a :func:`strip_window` triple came from ($29C9 halves it,
    $29CF keeping the odd bit in $0028)."""
    return window[0] * 2 + (1 if window[2] else 0)


def strip_window(columns):
    """$29C7 with A = $0C69: the buffer window $1F9F gives the strip plot.

    $29C9 halves the column count into $0007 and $29D3 folds that into $0012, the same
    pair $2993 sets from its own table for a full-screen mode; $29CF rotates the halving's
    carry into $0028, so an ODD strip drops the left column's near half."""
    left = columns >> 1
    return left, (left >> 1) ^ 0x80, 0x80 if columns & 1 else 0x00


MAX_CHUNK = 0x14  # $2105: $0C69 never exceeds 20, so a wider span replots in chunks


def replot_chunks(left, span):
    """$1FC2..$2029: the (left, columns) strips $2625 replots to cover ``span``.

    $2105 caps $0C69 at 20 and $201E/$2021 re-enter $1FC2 with what is left, so a
    span over 20 columns is two passes, each at its own $1FC2 camera shift."""
    out = []
    while span > 0:
        take = min(span, MAX_CHUNK)
        out.append((left, take))
        left += take
        span -= take
    return tuple(out)


def clear_strip_cycles(span):
    """clear_strip $2211 over ``span`` columns of 24 rows, from its own loops.

    $2247 stores 8 bytes a column and $2276 re-enters for the columns past 32."""
    if span < 32:
        entry, row = passcost.CLEAR_ENTRY, passcost.CLEAR_ROW
    elif span == 32:
        entry, row = passcost.CLEAR_ENTRY_EXACT, passcost.CLEAR_ROW
    else:
        entry = passcost.CLEAR_ENTRY_WIDE
        row = passcost.CLEAR_ROW + passcost.CLEAR_ROW_SPLIT
    body = passcost.CLEAR_COLUMN * span + row
    return entry + passcost.CLEAR_ROWS * body + passcost.CLEAR_TAIL


def flush_cycles(scroll):
    """One $9730 buffer flush, from the 24 $3A00 rows it copies at ``scroll`` = $0095.

    $9792 rolls X over the 25 screen banks, so the 24-row window skips bank
    ``scroll - 1``; a bank with a nonzero $3A40 entry buys a second $9888."""
    splits = sum(
        1
        for r in passcost.FLUSH_SPLIT_ROWS
        if (r - scroll) % passcost.FLUSH_BANKS < passcost.FLUSH_ROWS
    )
    wrap = passcost.FLUSH_WRAP if scroll else 0
    return passcost.FLUSH + passcost.FLUSH_SPLIT * splits + wrap


CHUNK_CYCLES = (
    passcost.REDRAW_CHUNK_HEAD
    + passcost.REDRAW_CHUNK_TAIL
    + passcost.BUF_WINDOW_CALL  # $1FE5 JSR $29C7
)


def strip_flush_cycles(span, scroll):
    """$202C..$1F9E: the $207E $9730 flush loop over ``span`` columns, and the exit."""
    tail = passcost.REDRAW_TAIL + passcost.REDRAW_FLUSH_ENTRY + passcost.REDRAW_EXIT
    return tail + span * (passcost.REDRAW_FLUSH_LOOP + flush_cycles(scroll))


def strip_line_cycles(span, chunks, scroll):
    """$1FA4..$1F9E: everything the strip replot costs but the $2625 chunks themselves.

    The $2211 clear and the $207E $9730 flush loop are both driven by the uncapped
    $211B span, the $29C7 window by each chunk."""
    cyc = passcost.REDRAW_CLEAR_CALL + clear_strip_cycles(span)
    cyc += chunks * CHUNK_CYCLES
    cyc += (chunks - 1) * (passcost.REDRAW_CHUNK_MORE + passcost.REDRAW_CHUNK_RESUME)
    return cyc + strip_flush_cycles(span, scroll)


SCREEN_SCROLL = 0x95  # $2043: the first screen bank $9730 flushes, $0097 for its loop
CAMERA_SAVED = mm.CAMERA_SAVED
STRIP_LEFT = 0x0C62  # $20D8: the column the strip being plotted starts at
STRIP_COLUMNS = 0x0C69  # $210B: that strip's own width, capped at 20
STRIP_REMAINING = 0x0C6A  # $20FD: the columns left to plot, this strip included
STRIP_SPAN = 0x211B  # $2100: the whole span, which the chunk loop never rewrites
PLOT_ROW = 0x26  # $26C9 seeds it $1F and $26EF walks it down to the observer row $001D
PLOT_COLUMN = 0x25  # $295D drives it across the row it plots
CAMERA_OBJECT = mm.CAMERA_OBJECT
BUF_LEFT, BUF_RIGHT, BUF_FRAC = (
    0x07,
    0x12,
    0x28,
)  # the $2993/$29C7 window plot_world is plotting into
CAMERA_REF_LO = mm.CAMERA_REF_LO


def strip_view(h_angle, v_angle, left):
    """$1FC2/$1FCC: the camera the ROM plots a strip starting at column ``left`` from.

    $209A holds $0C62 halved, and the halving's carry is the camera's own $001F half
    column ($1FCF/$1FD0)."""
    return {
        "h_angle": (h_angle + (left >> 1)) & 0xFF,
        "v_angle": v_angle,
        "ref_lo": 0x80 if left & 1 else 0x00,
    }


def hold_camera(state, left):
    """$1FCF..$1FDB with $211A already saved: $09C0,X and $001F for a strip at ``left``.

    Idempotent -- it re-derives the shifted bearing from $211A, never from itself."""
    mem = state.mem
    mem[CAMERA_REF_LO] = 0x80 if left & 1 else 0x00
    state.obj_h_angle[mem[CAMERA_OBJECT]] = (mem[CAMERA_SAVED] + (left >> 1)) & 0xFF


def shift_camera(state, left):
    """$1FC2..$1FDB: save the camera object's own bearing in $211A, then point it at
    the strip -- so the shift is in the object table, live to anything reading it."""
    state.mem[CAMERA_SAVED] = state.obj_h_angle[state.mem[CAMERA_OBJECT]]
    hold_camera(state, left)


def restore_camera(state):
    """$2003/$2008: put the camera's own bearing and half column back."""
    mem = state.mem
    mem[CAMERA_REF_LO] = 0
    state.obj_h_angle[mem[CAMERA_OBJECT]] = mem[CAMERA_SAVED]


def held_strip(state):
    """The $0C62 a still-shifted $09C0,X names, 0 when the camera is its own again.

    $1FCF/$1FD0 put $0C62's low bit in $001F and $1FD8 adds the rest to the saved
    bearing, so the two together give the column back."""
    mem = state.mem
    shift = (state.obj_h_angle[mem[CAMERA_OBJECT]] - mem[CAMERA_SAVED]) & 0xFF
    return (shift << 1) | (1 if mem[CAMERA_REF_LO] else 0)


def strip_replot_frames(state, target, left, columns, span=None):
    """$1FA4..$1F9E out of update_object_on_screen $1F9F, in PAL frames.

    Around the $2625 chunks sit the $2211 clear and the $9730 flush, both over the
    uncapped ``span`` ($0C6A; ``columns`` when not given).
    ``RENDER_COST_BACKEND=py65`` runs the real $1F9F and is exact."""
    if os.environ.get("RENDER_COST_BACKEND", "proxy").lower() == "py65":
        exact = _exact_strip_cost(state, target)
        if exact is not None:
            return exact
    span = columns if span is None else span
    player = state.player
    chunks = replot_chunks(left, span)
    frames = strip_line_cycles(span, len(chunks), state.mem[SCREEN_SCROLL])
    frames /= FRAME_CYCLES
    h, v = state.obj_h_angle[player], state.obj_v_angle[player]
    for chunk_left, chunk_columns in chunks:
        frames += render_cost(
            state,
            strip_view(h, v, chunk_left),
            player,
            window=strip_window(chunk_columns),
        )
    return frames


def _exact_strip_cost(state, target):
    """The py65 $1F9F backend, or None when the ROM fixture is absent."""
    try:
        from sentinel.tests import oracle

        if not oracle.available():
            raise FileNotFoundError("ROM fixture absent")
        from sentinel import rendercost_py65

        return rendercost_py65.strip_replot_cost_exact(state, target)
    except (ImportError, FileNotFoundError) as exc:
        if not _EXACT_WARNED[0]:
            _EXACT_WARNED[0] = True
            print(f"RENDER_COST_BACKEND=py65 unavailable ({exc}); using proxy")
        return None


def replot_owed(state, row, column, in_plot):
    """The cycles an interrupted $1FFC JSR $2625 still owes $1F9F, from its own progress.

    ``row`` is the next row $295D will plot, ``column`` the machine's $0025, and
    ``in_plot`` says the halt is inside $295D rather than that row's own $27D7 scan.
    Camera, view and window come from the image, where $1FC2/$1FE5 have set them.
    """
    mem = state.mem
    obs = mem[CAMERA_OBJECT]
    window = (mem[BUF_LEFT], mem[BUF_RIGHT], mem[BUF_FRAC])
    h, v = state.obj_h_angle[obs], state.obj_v_angle[obs]
    setup = _setup(state, h, v, obs, PLAY_MODE, window, mem[CAMERA_REF_LO])
    schedule = []
    _tiles, _n, _exam, fill, _walk = _project_scene_py(state, setup, obs, schedule)
    at = len(schedule) - 1  # past every plotted row: only $27CE's own tile is left
    if row >= _LAST:  # $26EF has not run once, so neither has one $295D
        at = 0
    else:
        for i, e in enumerate(schedule):
            if e["row"] is not None and e["row"] <= row:
                at = i
                break
    owed = sum(_stage_cycles(e) for e in schedule[at:])
    here = schedule[at]
    done = here["n0"] if "n0" in here else (0 if at == 0 else len(fill))
    if in_plot and "cols" in here:
        cols = here["cols"]
        spent = cols.index(column) if column in cols else len(cols)
        owed -= here["scan"] + here["head"] + sum(here["per_col"][:spent])
        done += spent
    columns = _window_columns(window)
    total, exact = fill_frames(state, setup, fill, PLAY_MODE, obs, columns)
    ahead, _e = fill_frames(state, setup, fill[:done], PLAY_MODE, obs, columns)
    if not exact:  # no game image: the object floor, split at the same tile
        total += _object_base_cycles(state, fill[:, rendercost.T_BYTE])
        ahead += _object_base_cycles(state, fill[:done, rendercost.T_BYTE])
    return owed + total - ahead + _strip_line_owed(state, obs)


def _strip_line_owed(state, obs):
    """$1FFF..$1F9E: what $1F9F owes once the interrupted $2625 returns.

    $2211 ran at $1FBA, before the chunk loop, so nothing of it is owed; the chunks
    past this one, the $9730 flush over the whole $211B span and the exit all are."""
    mem = state.mem
    chunk = mem[STRIP_COLUMNS]
    left = mem[STRIP_LEFT] + chunk
    owed = passcost.REDRAW_CHUNK_TAIL
    for nxt, cols in replot_chunks(left, mem[STRIP_REMAINING] - chunk):
        owed += passcost.REDRAW_CHUNK_MORE + passcost.REDRAW_CHUNK_RESUME
        owed += CHUNK_CYCLES + FRAME_CYCLES * render_cost(
            state,
            strip_view(mem[CAMERA_SAVED], state.obj_v_angle[obs], nxt),
            obs,
            window=strip_window(cols),
        )
    return owed + strip_flush_cycles(mem[STRIP_SPAN], mem[SCREEN_SCROLL])


def _stage_cycles(entry):
    """One :func:`_scan_visible` schedule entry's own $27D7 scan and $295D walk."""
    if "cols" not in entry:
        return entry["scan"]
    return entry["scan"] + entry["head"] + sum(entry["per_col"]) + entry["tail"]


# wait_for_end_of_tune ($35D5): #$19 tune ($1B82/$AB69) = FIXED 96 note-hold frames ($0CDF@$9630), == #$0 TUNE_FRAMES.  The transfer settle law lives in sentinel.settlecost.
TUNE_TRANSFER_FRAMES = float(os.environ.get("TUNE_TRANSFER_FRAMES", "96"))
