"""plot_world ($2625) terrain render-projector, ported bit-exactly from the ROM.

Walks the 32x32 tile grid furthest->nearest, projecting each grid point ($2845) to
a horizontal angle (screen x) and vertical angle (screen y). Trig cores reused from
:mod:`sentinel.relative`; projected corners feed :func:`render_cost`.
"""

import collections
import os

from sentinel import relative, terrain, memmap as mm

# initialise_buffer_variables ($2993) mode -> ($0007, $0012=($0007>>1)^$80), table $29C4; mode 0 = play buffer and vertical-pan strip ($9939), mode 2 = horizontal-pan strip ($994F).
BUF_WINDOW = {0: (0x14, 0x8A), 1: (0x14, 0x8A), 2: (0x08, 0x84)}
PLAY_MODE = 0
_BUF_LEFT, _BUF_RIGHT = BUF_WINDOW[PLAY_MODE]


def _neg16(hi, lo):
    """Two's-complement negate the 16-bit (hi:lo), as $2865/$286C do."""
    borrow = 1 if lo != 0 else 0
    return ((-hi - borrow) & 0xFF, (-lo) & 0xFF)


def _setup(state, h_angle, v_angle, observer, mode=PLAY_MODE):
    """plot_world setup ($2625-$26D6): the view-orientation case and screen-x
    reference angle from the observer's h_angle, against ``mode``'s $2993 window."""
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
    left, right = BUF_WINDOW[mode]
    return {
        "observer": observer,
        "quadrant": quadrant,
        "buf_left": left,  # $0007
        "buf_right": right,  # $0012
        "c3": c3,  # $0003
        "c1d": c1d,  # $001D
        "ref_lo": 0x00,  # $001F (play)
        "ref_hi": (folded - 0x0A) & 0xFF,  # $0020 ($267D)
        "v_angle": v_angle,  # objects_v_angle used by $933D
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
    bottommost object's z_height. Returns (height, tile_byte)."""
    tb = terrain.tile_byte(state, tx, ty)
    if tb >= mm.OBJECT_TILE:
        return state.obj_z_height[terrain.bottom_object(state, tb & 0x3F)], tb
    return tb >> 4, tb


def _project(state, setup, col, row):
    """$2845: project grid point (col,row). Returns (sx_lo, sx_hi, sy_lo, sy_hi,
    tile_byte, onscreen) matching the ROM plottables, visibility byte and $007F."""
    observer = setup["observer"]
    zp = collections.defaultdict(int)
    sx = (col - setup["c3"] - 1) & 0xFF  # signed column ($2858, CLC-before-SBC -1)
    zp[0x86], zp[0x80] = sx, 0x80
    if sx & 0x80:
        zp[0x83], zp[0x80] = _neg16(sx, 0x80)
    else:
        zp[0x83] = sx
    sr = (row - setup["c1d"] - 1) & 0xFF  # signed row ($2876)
    zp[0x88], zp[0x82] = sr, 0x80
    if sr & 0x80:
        zp[0x85], zp[0x82] = _neg16(sr, 0x80)
    else:
        zp[0x85] = sr
    relative._calc_angle(zp)  # $9287 -> zp[$8A]/zp[$8B]
    sx_lo = (zp[0x8A] - setup["ref_lo"]) & 0xFF  # screen x ($2891), carry stays set
    sx_hi = (zp[0x8B] - setup["ref_hi"]) & 0xFF
    relative._calc_hypotenuse(zp)  # $937F -> zp[$7C]/zp[$7D]
    tx, ty = _tile_xy(setup["quadrant"], col, row)
    height, tile_byte = _tile_height(state, tx, ty)
    zf = state.obj_z_frac[observer]  # tile height relative to eye ($291E)
    zp[0x80] = (-zf) & 0xFF
    rel_z_hi = (height - state.obj_z_height[observer] - (1 if zf else 0)) & 0xFF
    sy_hi = relative._vertical_angle(zp, rel_z_hi, setup["v_angle"])  # $933D
    sy_lo = zp[0x50]
    # $293C on-screen test against the mode's $0007/$0012; $0028=0 waives the fraction check.
    if sx_hi < setup["buf_left"]:
        onscreen = 0x00
    elif sx_hi < setup["buf_right"]:
        onscreen = 0x80
    else:
        onscreen = 0x81
    return sx_lo, sx_hi, sy_lo, sy_hi, tile_byte, onscreen


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


# $0C48 furthest-row extent hint ($26CD); 0 in every fresh play state, env-overridable.
_ROW_HINT = int(os.environ.get("RENDER_ROW_HINT", "0"))
_LAST = mm.N - 1  # 0x1F
_OFFSET_TO_TILE = (
    0x00,
    0x01,
    0x21,
    0x20,
)  # offset_to_tile_table $27D3, indexed by quadrant


def _scan_visible(state, setup):
    """Exact port of find_visible_extent ($27D7) + plot_rows_in_front_of_observer_loop
    ($26DE): the furthest->nearest walk that probes tiles via $2845. Returns
    (n_examine, rows, cache) with the exact $2845 call count and per-row (row, lo, hi)
    plotted extents; the on-screen byte drives every branch so the count is byte-exact.
    """
    cache = {}
    exam = [0]

    def probe(col, row):
        exam[0] += 1
        col &= 0xFF
        r = cache.get((col, row))
        if r is None:
            r = _project(state, setup, col, row)
            cache[(col, row)] = r
        return r[5]

    def find_end(row, col):  # find_end_of_row_loop $27E2
        while True:
            start = col
            if col == _LAST:
                return start, _LAST
            col += 1
            a = probe(col, row)
            if a == 0x81:
                continue
            if a == 0x80:
                return start, col
            while True:  # find_first_visible_tile_at_end_loop $27F3
                if col == _LAST:
                    return start, col
                col += 1
                a = probe(col, row)
                if a == 0:
                    continue
                return start, col

    def start_left(row, end, col):  # find_first_visible_tile_at_start_of_row_loop $2820
        while True:
            if col == 0:
                return 0, end
            col -= 1
            if probe(col, row) == 0:
                continue
            return col, end

    def crop_right(row, col):  # tile_is_cropped_to_right $27FF
        while True:
            end = col
            if col == 0:
                return 0, end
            col -= 1
            a = probe(col, row)
            if a == 0x80:
                continue
            if a != 0:  # into_find_first_visible_tile_at_start_of_row_loop $2825
                return col, end
            return start_left(row, end, col)

    def find_extent(row, hint):  # find_visible_extent $27D7
        col = hint & 0xFF
        a = probe(col, row)
        if a == 0x80:
            return crop_right(row, col)
        if a != 0:
            return find_end(row, col)
        while True:  # find_first_visible_tile_at_start_loop $2811
            if col == _LAST:
                return start_left(row, _LAST, hint & 0xFF)  # endRow2 $2818
            col += 1
            a = probe(col, row)
            if a == 0:
                continue
            return start_left(row, col, hint & 0xFF)

    c1d, c3 = setup["c1d"], setup["c3"]
    rows = []
    row = _LAST
    start, end = find_extent(row, _ROW_HINT)
    while True:
        row -= 1
        if row < 0:
            break
        if row == c1d:  # consider_plotting_observer_row $276F: last, observer row
            y = (start + 1) & 0xFF
            if y == c3:  # plot_observer_row $2786: plots the single tile $0037
                probe(start, row)
                probe(y, row)
                probe(c3, row)
                rows.append((row, start, (start + 1) & 0xFF))
            elif (end - 2) & 0xFF == c3:  # $277B: plots the single tile $0038-1
                probe((end - 1) & 0xFF, row)
                probe(end, row)
                probe(c3, row)
                rows.append((row, (end - 1) & 0xFF, end))
            else:  # skip_plotting_observer_row $2793: only the observer tile ($27CE)
                probe(c3, row)
            break
        p_start, p_end = start, end
        start, end = find_extent(row, p_start)
        if start < p_start:  # this_row_starts_before $2713
            y = (p_start - 1) & 0xFF
            probe(y, row)
            while y != start:
                y = (y - 1) & 0xFF
                probe(y, row)
        elif start > p_start:  # calculate_this_row_new_first_tiles $2709
            y = (start - 1) & 0xFF
            probe(y, row)
            while y != p_start:
                y = (y - 1) & 0xFF
                probe(y, row)
        if end > p_end:  # this_row_ends_after $2741
            y = p_end
            while True:
                y = (y + 1) & 0xFF
                probe(y, row)
                if y == end:
                    break
        elif end < p_end:  # calculate_this_row_new_last_tiles $2737
            y = end
            while True:
                y = (y + 1) & 0xFF
                probe(y, row)
                if y == p_end:
                    break
        rows.append((row, min(start, p_start), max(end, p_end)))
    return exam[0], rows, cache


def _occlusion_visible(state, observer=None):
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


# Every byte plot_world reads: tiles_table ($0400) + the object flags/v_angle ($0100) and x/z/y/h_angle/z_frac/type ($0900) arrays -- 1536 bytes, ~1us to digest.
_SCENE_SPANS = ((0x0400, 0x0800), (0x0100, 0x0180), (0x0900, 0x0A80))
_CACHE_MAX = int(os.environ.get("RENDER_CACHE_MAX", "20000"))
_OCCLUSION_CACHE = {}


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


def occlusion_visible(state, observer=None):
    """:func:`_occlusion_visible` memoized on (scene, observer): the $245B table is
    view-independent, so one raytrace serves every bearing at an observer."""
    obs = state.player if observer is None else observer
    return memo(
        _OCCLUSION_CACHE,
        (scene_key(state), obs, state.obj_x[obs], state.obj_y[obs]),
        _CACHE_MAX,
        lambda: _occlusion_visible(state, obs),
    )


def project_scene(state, h_angle, v_angle, observer=None, mode=PLAY_MODE):
    """Return (tiles, n_examine): the exactly-selected plotted tiles and the exact
    $2845 examination count under ``mode``'s $2993 buffer window. Non-object tiles the
    occlusion table hides are examined but dropped; each kept tile carries its H and W.
    """
    if observer is None:
        observer = state.player
    setup = _setup(state, h_angle & 0xFF, v_angle & 0xFF, observer, mode)
    n_examine, rows, cache = _scan_visible(state, setup)
    visible = occlusion_visible(state, observer)

    def proj(col, row):
        col &= 0xFF
        cached = cache.get((col, row))
        if cached is None:
            cached = _project(state, setup, col, row)
            cache[(col, row)] = cached
        return cached

    s1b = _OFFSET_TO_TILE[setup["quadrant"]]
    # plot_tile ($2A24) reads $0180 slot (($0025|$0005)+$001B)&$3F: drawn tile is examine (col+offc,row+offr); $001B=$27D3[quad], bit0=col, bit5=bank(row+1).
    offc, offr = s1b & 1, (s1b >> 5) & 1
    tiles = []
    for row, lo, hi in rows:
        re = (row + offr) & 0xFF
        for col in range(lo, hi):  # plot range [$0037, $0038); $0038 excluded
            ce = (col + offc) & 0xFF
            res = proj(ce, re)
            tb = res[4]
            if tb == 0:  # $0180 slot zero: nothing to plot ($2A27 BEQ)
                continue
            tx, ty = _tile_xy(setup["quadrant"], ce, re)
            if tb < mm.OBJECT_TILE and not visible[ty][tx]:
                continue  # $291B zeroes $0180 for hidden non-object tiles
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
    return tiles, n_examine


def visible_tiles(state, h_angle, v_angle, observer=None, mode=PLAY_MODE):
    """The plotted-tile list from :func:`project_scene` (drops n_examine)."""
    return project_scene(state, h_angle, v_angle, observer, mode)[0]


FRAME_CYCLES = 19656.0  # PAL frame
BASE_CYCLES = float(os.environ.get("RENDER_BASE_CYCLES", "0"))
# term (a) per-$2845-calltree cost ($2845+$9287+$937F+$933D), py65 mean (1551-2046).
C_EXAMINE = float(os.environ.get("RENDER_C_EXAMINE", "1737"))
PER_SCANLINE = float(os.environ.get("RENDER_PER_SCANLINE", "60"))  # term (b) edge/row
PER_PIXEL = float(os.environ.get("RENDER_PER_PIXEL", "1.75"))  # term (b) span_fill byte

# term (c) plot_object ($8533) base floor (docs/render_cost.md); object fill = residual.
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
    """plot_object base cost summed over objects on the plotted object-tiles ($21AE
    stack walk down the $0100 flags chain). Unknown types contribute nothing."""
    total = 0.0
    for tile in tiles:
        tb = tile["tile_byte"]
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


def render_cost(state, view, observer=None, mode=PLAY_MODE):
    """One plot_world pass in PAL frames (docs/render_cost.md):
    ``(BASE + N_examine*C_EXAMINE + sum_tiles(60*H + 1.75*H*W) + object_base)/19656``,
    into ``mode``'s $2993 buffer. ``view`` maps ``h_angle``/``v_angle``; 0.0 if none.
    ``RENDER_COST_BACKEND=py65`` (ROM present) replaces the proxy for the play buffer.
    """
    if not view or view.get("h_angle") is None:
        return 0.0
    h = view["h_angle"] & 0xFF
    v = (view.get("v_angle") or 0) & 0xFF
    if mode == PLAY_MODE:
        exact = _exact_render_cost(state, h, v, observer)
        if exact is not None:
            return exact
    tiles, n_examine = project_scene(state, h, v, observer, mode)
    area = sum(PER_SCANLINE * t["h"] + PER_PIXEL * t["h"] * t["w"] for t in tiles)
    obj_base = _inview_object_base(state, tiles)
    return (BASE_CYCLES + n_examine * C_EXAMINE + area + obj_base) / FRAME_CYCLES


# Transfer viewpoint-replot settle ($357D): two plot_world passes (docs/render_cost.md).
REPLOT_PASSES = float(os.environ.get("RENDER_REPLOT_PASSES", "2"))
# wait_for_end_of_tune ($35D5): #$19 tune ($1B82/$AB69) = FIXED 96 note-hold frames ($0CDF@$9630), == #$0 TUNE_FRAMES.
TUNE_TRANSFER_FRAMES = float(os.environ.get("TUNE_TRANSFER_FRAMES", "96"))
# Fixed $357D foreground before the tune, absent from render_cost: $245B occ + $3700 + fill + status (py65 ~176f).
SETTLE_FIXED_FRAMES = float(os.environ.get("SETTLE_FIXED_FRAMES", "176"))


def viewpoint_replot_frames(state, view, observer=None):
    """Transfer/viewpoint-change settle in frames (docs/render_cost.md): fixed tune
    wait + fixed $245B/$3700/fill/status foreground + ``REPLOT_PASSES`` plot_world
    passes, all seen from ``observer`` (the POST-transfer eye $0C63; default player).
    Live $9630 settle 259-460f; median abs error <15%
    (``test_viewpoint_replot_lands_in_live_settle_band``)."""
    return (
        TUNE_TRANSFER_FRAMES
        + SETTLE_FIXED_FRAMES
        + REPLOT_PASSES * render_cost(state, view, observer)
    )
