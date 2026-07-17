"""plot_world ($2625) terrain render-projector, ported bit-exactly from the ROM.

Walks the 32x32 tile grid furthest->nearest, projecting each grid point ($2845) to
a horizontal angle (screen x) and vertical angle (screen y). Trig cores reused from
:mod:`sentinel.relative`; projected corners feed :func:`render_cost`.
"""

import collections
import os

from sentinel import relative, terrain, memmap as mm

# initialise_buffer_variables ($2993) play buffer: $0007, $0012=($0007>>1)^$80.
_BUF_LEFT = 0x14  # $0007
_BUF_RIGHT = 0x8A  # $0012


def _neg16(hi, lo):
    """Two's-complement negate the 16-bit (hi:lo), as $2865/$286C do."""
    borrow = 1 if lo != 0 else 0
    return ((-hi - borrow) & 0xFF, (-lo) & 0xFF)


def _setup(state, h_angle, v_angle, observer):
    """plot_world setup ($2625-$26D6): the view-orientation case and screen-x
    reference angle from the observer's h_angle."""
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
    return {
        "observer": observer,
        "quadrant": quadrant,
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
    if sx_hi < _BUF_LEFT:  # on-screen test ($293C); $0028=0 waives the fraction check
        onscreen = 0x00
    elif sx_hi < _BUF_RIGHT:
        onscreen = 0x80
    else:
        onscreen = 0x81
    return sx_lo, sx_hi, sy_lo, sy_hi, tile_byte, onscreen


_SCREEN_H = int(os.environ.get("RENDER_SCREEN_H", "240"))  # $F0 fillable scanlines
_W_SCALE = int(os.environ.get("RENDER_W_SCALE", "32"))  # angle16 -> screen pixels
_W_SCREEN = int(os.environ.get("RENDER_W_SCREEN", "160"))  # screen width, pixels
_W_WRAP = int(os.environ.get("RENDER_W_WRAP", "200"))  # skip angle-wrapped tiles


def _signed16(hi, lo):
    """A corner's signed 16-bit projected coordinate (prepare_polygon $2F02/$2DD2):
    high byte selects the region/band, low byte the scanline/fine position."""
    return (hi - 256 if hi & 0x80 else hi) * 256 + lo


def _clamp(v, lo, hi):
    """Clamp ``v`` into ``[lo, hi]``."""
    return lo if v < lo else hi if v > hi else v


def project_scene(state, h_angle, v_angle, observer=None):
    """Return (tiles, n_examine): plotted tiles and the count of $2845 examinations,
    mirroring the furthest->nearest row loop ($26DE) and per-row visible-extent scan
    ($27D7). Each tile carries its projection and screen height H and width W."""
    if observer is None:
        observer = state.player
    setup = _setup(state, h_angle & 0xFF, v_angle & 0xFF, observer)
    n_examine = 0
    grid = {}

    def proj(col, row):
        cached = grid.get((col, row))
        if cached is None:
            cached = _project(state, setup, col, row)
            grid[(col, row)] = cached
        return cached

    tiles = []
    for row in range(mm.N - 1, -1, -1):  # $0026 31->0
        onscreen_cols = [c for c in range(mm.N) if proj(c, row)[5]]
        n_examine += mm.N
        if not onscreen_cols:
            continue
        n_examine += 2  # the two off-screen boundary probes per row scan
        for col in range(onscreen_cols[0], onscreen_cols[-1] + 1):
            res = proj(col, row)
            if not res[5]:
                continue
            c1, r1 = min(col + 1, mm.N - 1), min(row + 1, mm.N - 1)
            corners = (res, proj(c1, row), proj(col, r1), proj(c1, r1))
            ys = [_signed16(c[3], c[2]) for c in corners]
            xs = [_signed16(c[1], c[0]) for c in corners]
            top = _clamp(min(ys), 0, _SCREEN_H)
            bot = _clamp(max(ys), 0, _SCREEN_H)
            span = (max(xs) - min(xs)) / _W_SCALE
            if span > _W_WRAP:  # a tile straddling the angle wrap: not on screen
                continue
            width = min(span, _W_SCREEN)
            tx, ty = _tile_xy(setup["quadrant"], col, row)
            tiles.append(
                {
                    "col": col,
                    "row": row,
                    "tile": (tx, ty),
                    "sx_lo": res[0],
                    "sx_hi": res[1],
                    "sy_lo": res[2],
                    "sy_hi": res[3],
                    "tile_byte": res[4],
                    "onscreen": res[5],
                    "h": bot - top,
                    "w": max(width, 0),
                }
            )
    return tiles, n_examine


def visible_tiles(state, h_angle, v_angle, observer=None):
    """The plotted-tile list from :func:`project_scene` (drops n_examine)."""
    return project_scene(state, h_angle, v_angle, observer)[0]


FRAME_CYCLES = 19656.0  # PAL frame
BASE_CYCLES = float(os.environ.get("RENDER_BASE_CYCLES", "0"))
C_EXAMINE = float(os.environ.get("RENDER_C_EXAMINE", "900"))  # term (a) trig floor
PER_SCANLINE = float(os.environ.get("RENDER_PER_SCANLINE", "60"))  # term (b)
PER_PIXEL = float(os.environ.get("RENDER_PER_PIXEL", "1.75"))


def render_cost(state, view, observer=None):
    """One plot_world pass in PAL frames (docs/render_cost.md):
    ``(BASE + N_examine*C_examine + sum_tiles(60*H + 1.75*H*W)) / 19656``.

    ``view`` maps ``h_angle``/``v_angle`` (the aimed heading); 0.0 if no heading."""
    if not view or view.get("h_angle") is None:
        return 0.0
    h = view["h_angle"] & 0xFF
    v = (view.get("v_angle") or 0) & 0xFF
    tiles, n_examine = project_scene(state, h, v, observer)
    area = sum(PER_SCANLINE * t["h"] + PER_PIXEL * t["h"] * t["w"] for t in tiles)
    return (BASE_CYCLES + n_examine * C_EXAMINE + area) / FRAME_CYCLES


# Transfer viewpoint-replot settle ($357D): two plot_world passes (docs/render_cost.md).
REPLOT_PASSES = float(os.environ.get("RENDER_REPLOT_PASSES", "2"))


def viewpoint_replot_frames(state, view, observer=None):
    """The transfer/viewpoint-change settle in frames: ``REPLOT_PASSES`` blocking
    plot_world passes ($35C3/$35C6), the scene-dependent redraw docs/render_cost.md
    measures at ~306-420 frames (vs the old constant 47)."""
    return REPLOT_PASSES * render_cost(state, view, observer)
