"""Isometric SVG diagram of a board state, annotated with an action line.

Draws the 32x32 height field as a lit isometric mesh (flat tiles flat, sloped tiles
from their four ROM corner heights), the objects as glyphs, the enemies with their
facing cones, and any list of actions as numbered arrows with a side panel.
"""

import argparse
import html
import math
import os

from sentinel import aimcost, landscape, memmap as mm, terrain
from sentinel.game import Game
from sentinel.los import _slope_corner_z

FOV_HALF = 0x14 // 2  # $16F2: half the enemy scan cone's width
TILE_W = 34.0
TILE_H = 17.0
Z_H = 12.0
GRID = 32
PANEL_W = 440.0
PAD = 26.0
HEAD_H = 74.0

BG = "#11151c"
INK = "#e8edf5"
DIM = "#93a1b5"
RAMP = ("#1d3b2a", "#255138", "#2f6845", "#3d8054", "#559a63", "#77b276", "#a3c98e")
LIGHT = (-0.55, -0.62, 0.56)

VERB_COLOR = {
    "absorb": "#ff9f45",
    "create": "#4fd08a",
    "robot": "#4fd08a",
    "boulder": "#4fd08a",
    "transfer": "#59b8ff",
    "hyperspace": "#c58bff",
}
TYPE_COLOR = {
    mm.T_ROBOT: "#8fd4ff",
    mm.T_SENTRY: "#ff6b6b",
    mm.T_TREE: "#3fae72",
    mm.T_BOULDER: "#b9a37a",
    mm.T_MEANIE: "#ff3fb0",
    mm.T_SENTINEL: "#ff2f2f",
    mm.T_PLATFORM: "#ffd166",
}
GLYPH_H = {
    mm.T_ROBOT: 26.0,
    mm.T_SENTRY: 30.0,
    mm.T_TREE: 26.0,
    mm.T_BOULDER: 14.0,
    mm.T_MEANIE: 24.0,
    mm.T_SENTINEL: 36.0,
    mm.T_PLATFORM: 6.0,
}


def _org():
    """Screen origin: board x-y spans +-GRID/2 tiles wide, heights lift the top."""
    return (PAD + GRID * TILE_W / 2.0, PAD + HEAD_H + 15 * Z_H)


def project(x, y, z, org=None):
    """Tile-space (x, y, height) to screen coordinates in the 2:1 isometric view."""
    ox, oy = org or _org()
    return (ox + (x - y) * TILE_W / 2.0, oy + (x + y) * TILE_H / 2.0 - z * Z_H)


def _canvas():
    w = GRID * TILE_W + 2 * PAD + PANEL_W
    h = GRID * TILE_H + 2 * PAD + HEAD_H + 16 * Z_H
    return w, h


def _corners(state, x, y):
    """The tile's four corner heights (NE, SE, SW, NW order of the ROM square); a
    flat tile is its own height at every corner, a sloped one reads its neighbours."""
    z, slope = terrain.resolve_ground(state, x, y)
    if not slope:
        return (z, z, z, z)
    return (
        _slope_corner_z(state, x, y)[0],
        _slope_corner_z(state, x + 1, y)[0],
        _slope_corner_z(state, x + 1, y + 1)[0],
        _slope_corner_z(state, x, y + 1)[0],
    )


def _shade(color, f):
    r, g, b = (int(color[i : i + 2], 16) for i in (1, 3, 5))
    return "#%02x%02x%02x" % tuple(max(0, min(255, int(c * f))) for c in (r, g, b))


def _tile_fill(cz):
    """Height ramp colour with a Lambert term from the tile's own gradient."""
    c0, c1, c2, c3 = cz
    dzdx = ((c1 + c2) - (c0 + c3)) / 2.0
    dzdy = ((c2 + c3) - (c0 + c1)) / 2.0
    nx, ny, nz = -dzdx, -dzdy, 1.0
    ln = math.sqrt(nx * nx + ny * ny + nz * nz)
    lam = (nx * LIGHT[0] + ny * LIGHT[1] + nz * LIGHT[2]) / ln
    base = RAMP[min(len(RAMP) - 1, int(sum(cz) / 4.0 * (len(RAMP) - 1) / 11.0))]
    return _shade(base, 0.62 + 0.5 * max(0.0, lam))


def _poly(pts, fill, stroke=None, width=0.4, extra=""):
    d = " ".join(f"{x:.1f},{y:.1f}" for x, y in pts)
    sk = f' stroke="{stroke}" stroke-width="{width}"' if stroke else ""
    return f'<polygon points="{d}" fill="{fill}"{sk}{extra}/>'


def _text(x, y, s, size=12, fill=INK, anchor="start", weight="normal", extra=""):
    return (
        f'<text x="{x:.1f}" y="{y:.1f}" font-size="{size}" fill="{fill}" '
        f'text-anchor="{anchor}" font-weight="{weight}" '
        f'font-family="ui-monospace,Menlo,Consolas,monospace"{extra}>'
        f"{html.escape(str(s))}</text>"
    )


def _tile_top(state, x, y):
    """Screen point at the centre of a tile's surface (object tops included)."""
    top = terrain.top_object(state, x, y)
    z = terrain.resolve_ground(state, x, y)[0]
    if top is not None:
        z = state.obj_z_height[top] + GLYPH_H.get(state.obj_type[top], 0) / Z_H
    return project(x + 0.5, y + 0.5, z)


def _terrain_svg(state):
    """Every tile as a lit quad, drawn back to front along the isometric diagonal."""
    out = []
    for d in range(2 * GRID - 1):
        for x in range(max(0, d - GRID + 1), min(GRID, d + 1)):
            y = d - x
            cz = _corners(state, x, y)
            pts = [
                project(x, y, cz[0]),
                project(x + 1, y, cz[1]),
                project(x + 1, y + 1, cz[2]),
                project(x, y + 1, cz[3]),
            ]
            out.append(_poly(pts, _tile_fill(cz), "#0b0e13", 0.35))
    return out


def _signed_byte(b):
    return b - 256 if b >= 128 else b


def _cone(state, slot, reach=9.0):
    """The enemy's scan cone as a ground wedge on its current facing."""
    x, y = state.tile_of(slot)
    z = state.obj_z_height[slot]
    a = state.obj_h_angle[slot] / 256.0 * 2 * math.pi
    half = FOV_HALF / 256.0 * 2 * math.pi
    pts = [project(x + 0.5, y + 0.5, z)]
    for k in range(9):
        t = a - half + 2 * half * k / 8.0
        pts.append(
            project(x + 0.5 + reach * math.cos(t), y + 0.5 + reach * math.sin(t), z)
        )
    return _poly(pts, "#ff5a5a", "#ff8080", 0.6, ' fill-opacity="0.16"')


def _sweep(state, slot, reach=11.0):
    """Which way the cone is travelling: an arc off the leading edge, plus its sign.

    $1805 adds the per-enemy step from ROTATION_SPEED_TABLE ($14 or $EC, i.e. +20 or
    -20), so the sign of that byte is the scan's direction of travel.
    """
    step = _signed_byte(state.mem[mm.ROTATION_SPEED_TABLE + slot])
    if not step:
        return ""
    x, y = state.tile_of(slot)
    z = state.obj_z_height[slot]
    a = state.obj_h_angle[slot] / 256.0 * 2 * math.pi
    half = FOV_HALF / 256.0 * 2 * math.pi
    lead = a + (half if step > 0 else -half)
    span = abs(step) / 256.0 * 2 * math.pi
    pts = []
    for k in range(7):
        t = lead + (span if step > 0 else -span) * k / 6.0
        pts.append(
            project(x + 0.5 + reach * math.cos(t), y + 0.5 + reach * math.sin(t), z)
        )
    d = " ".join(f"{px:.1f},{py:.1f}" for px, py in pts)
    tip = pts[-1]
    label = "CW" if step > 0 else "CCW"
    return (
        f'<polyline points="{d}" fill="none" stroke="#ffd166" stroke-width="2.2" '
        f'stroke-linecap="round"/>'
        + _text(tip[0], tip[1] - 4, label, 9.5, "#ffd166", "middle", "bold")
    )


def _origin(state):
    """Mark tile 0,0 and the two axis directions, so the grid can be read."""
    z = 0
    o = project(0.5, 0.5, z)
    xa = project(4.5, 0.5, z)
    ya = project(0.5, 4.5, z)
    return [
        f'<circle cx="{o[0]:.1f}" cy="{o[1]:.1f}" r="4" fill="#ffd166"/>',
        _text(o[0], o[1] - 9, "0,0", 11, "#ffd166", "middle", "bold"),
        f'<line x1="{o[0]:.1f}" y1="{o[1]:.1f}" x2="{xa[0]:.1f}" y2="{xa[1]:.1f}" '
        f'stroke="#ffd166" stroke-width="1.6"/>',
        _text(xa[0] + 6, xa[1], "+x", 10.5, "#ffd166", "start", "bold"),
        f'<line x1="{o[0]:.1f}" y1="{o[1]:.1f}" x2="{ya[0]:.1f}" y2="{ya[1]:.1f}" '
        f'stroke="#ffd166" stroke-width="1.6"/>',
        _text(ya[0] - 6, ya[1], "+y", 10.5, "#ffd166", "end", "bold"),
    ]


def _glyph(state, slot, is_player):
    """One object drawn as a coloured prism with a type-specific cap."""
    x, y = state.tile_of(slot)
    otype = state.obj_type[slot]
    z = state.obj_z_height[slot] + state.obj_z_frac[slot] / 256.0
    col = "#ffffff" if is_player else TYPE_COLOR.get(otype, "#cccccc")
    hgt = GLYPH_H.get(otype, 20.0)
    cx, cy = project(x + 0.5, y + 0.5, z)
    w = TILE_W * 0.26
    out = [
        _poly(
            [
                (cx - w, cy),
                (cx, cy - TILE_H * 0.26),
                (cx + w, cy),
                (cx, cy + TILE_H * 0.26),
            ],
            _shade(col, 0.45),
            "#0b0e13",
            0.4,
        )
    ]
    if otype == mm.T_PLATFORM:
        return out
    out.append(
        f'<rect x="{cx - w * 0.5:.1f}" y="{cy - hgt:.1f}" width="{w:.1f}" '
        f'height="{hgt:.1f}" fill="{col}" stroke="#0b0e13" stroke-width="0.5" rx="1.5"/>'
    )
    if otype == mm.T_TREE:
        out.append(
            _poly(
                [
                    (cx - w * 1.3, cy - hgt * 0.55),
                    (cx, cy - hgt * 1.55),
                    (cx + w * 1.3, cy - hgt * 0.55),
                ],
                col,
                "#0b0e13",
                0.5,
            )
        )
    elif otype in (mm.T_SENTRY, mm.T_SENTINEL, mm.T_MEANIE, mm.T_ROBOT):
        out.append(
            f'<circle cx="{cx:.1f}" cy="{cy - hgt - 3:.1f}" r="{w * 0.62:.1f}" '
            f'fill="{col}" stroke="#0b0e13" stroke-width="0.5"/>'
        )
    if is_player:
        out.append(
            f'<circle cx="{cx:.1f}" cy="{cy - hgt - 3:.1f}" r="{w * 1.25:.1f}" '
            f'fill="none" stroke="#ffffff" stroke-width="1.6"/>'
        )
    return out


def _enemy_labels(state):
    """Number each enemy and give its height, so the diagram can be talked about.

    The Sentinel is slot 0 ($1B8E locks it last); sentries are numbered by slot in
    generation order, the numbering the docs use.
    """
    out = []
    n = 0
    for e in state.enemy_slots():
        x, y = state.tile_of(e)
        z = state.obj_z_height[e] + state.obj_z_frac[e] / 256.0
        if state.obj_type[e] == mm.T_SENTINEL:
            tag = "SENTINEL"
        else:
            n += 1
            tag = f"S{n}"
        cx, cy = project(x + 0.5, y + 0.5, z)
        out.append(
            _text(
                cx,
                cy - GLYPH_H.get(state.obj_type[e], 20.0) - 13,
                f"{tag}  {x},{y}  z={z:.2f}",
                10,
                TYPE_COLOR.get(state.obj_type[e], "#ffffff"),
                "middle",
                "bold",
            )
        )
    return out


def _tile_ring(state, x, y, color, label):
    """A coloured diamond on a tile's surface, with a label above it."""
    z = terrain.resolve_ground(state, x, y)[0]
    cx, cy = project(x + 0.5, y + 0.5, z)
    w, h = TILE_W * 0.5, TILE_H * 0.5
    return [
        _poly(
            [(cx - w, cy), (cx, cy - h), (cx + w, cy), (cx, cy + h)],
            "none",
            color,
            2.0,
        ),
        _text(cx, cy + h + 12, label, 10, color, "middle", "bold"),
    ]


def _objects_svg(state):
    """Cones behind everything, then glyphs back to front so near hides far."""
    out = [_cone(state, e) for e in state.enemy_slots()]
    out += [_sweep(state, e) for e in state.enemy_slots()]
    out += _origin(state)
    px, py = state.platform_xy
    out += _tile_ring(state, px, py, TYPE_COLOR[mm.T_PLATFORM], f"PLATFORM {px},{py}")
    for slot in sorted(
        state.occupied_slots(), key=lambda s: state.obj_x[s] + state.obj_y[s]
    ):
        out += _glyph(state, slot, slot == state.player)
    out += _enemy_labels(state)
    ex, ey = state.player_xy()
    out += [
        _text(
            *project(ex + 0.5, ey + 0.5, state.obj_z_height[state.player] + 3.4),
            f"YOU {ex},{ey} z={state.eye_z():.2f} E={state.energy}",
            10.5,
            "#ffffff",
            "middle",
            "bold",
        )
    ]
    return out


def _arrow(state, act, seq, stack=0):
    """A numbered action arc from the actor's tile to its target tile; ``stack`` lifts
    the badge clear of earlier actions on the same tile."""
    col = VERB_COLOR.get(act["verb"], "#ffffff")
    dash = ' stroke-dasharray="6 4"' if act.get("kind") == "planner" else ""
    x0, y0 = _tile_top(state, *act["from"])
    x1, y1 = _tile_top(state, *act["to"])
    lift = 17.0 * stack
    mx, my = (x0 + x1) / 2.0, (y0 + y1) / 2.0 - max(34.0, abs(x1 - x0) * 0.22) - lift
    bx, by = x1, y1 - lift
    return [
        f'<path d="M{x0:.1f},{y0:.1f} Q{mx:.1f},{my:.1f} {bx:.1f},{by:.1f}" '
        f'fill="none" stroke="{col}" stroke-width="2.2" opacity="0.92"{dash} '
        f'marker-end="url(#ah_{act["verb"]})"/>',
        f'<circle cx="{bx:.1f}" cy="{by:.1f}" r="7.5" fill="{col}" '
        f'stroke="#0b0e13" stroke-width="1"/>',
        _text(bx, by + 4, seq, 11, "#0b0e13", "middle", "bold"),
        _text(
            (x0 + mx) / 2.0,
            (y0 + my) / 2.0,
            f"{act['to'][0]},{act['to'][1]}",
            10,
            col,
            "middle",
        ),
    ]


def _panel(x, y, title, subtitle, state, acts, notes):
    """Right-hand column: board scalars, the action list, and free-text notes."""
    out = [_text(x, y, title, 16, INK, "start", "bold")]
    y += 20
    for line in subtitle:
        out.append(_text(x, y, line, 11, DIM))
        y += 15
    y += 8
    px, py = state.player_xy()
    foes = state.enemy_slots()
    counts = {}
    for s in state.occupied_slots():
        counts[mm.TYPES[state.obj_type[s]]] = (
            counts.get(mm.TYPES[state.obj_type[s]], 0) + 1
        )
    for line in (
        f"player   ({px},{py}) eye z={state.eye_z():.2f}",
        f"energy   {state.energy}",
        f"platform {state.platform_xy}",
        "objects  " + " ".join(f"{k[:4]}x{v}" for k, v in sorted(counts.items())),
        f"enemies  {len(foes)} (bearing to you, cone half-width {FOV_HALF})",
    ):
        out.append(_text(x, y, line, 11, INK))
        y += 15
    for e in foes:
        bearing = aimcost.bearing_to(*state.tile_of(e), px, py)
        out.append(
            _text(
                x,
                y,
                f"  {mm.TYPES[state.obj_type[e]][:4]} {tuple(state.tile_of(e))} "
                f"h={state.obj_h_angle[e]:3d} you={bearing} "
                f"d={aimcost.angle_dist(state.obj_h_angle[e], bearing)}",
                10,
                DIM,
            )
        )
        y += 14
    y += 12
    out.append(_text(x, y, "ACTIONS", 12, INK, "start", "bold"))
    y += 18
    for i, a in enumerate(acts, 1):
        col = VERB_COLOR.get(a["verb"], "#ffffff")
        who = "planner" if a.get("kind") == "planner" else "human"
        out.append(
            _text(
                x,
                y,
                f"{i:2d}. {who:5s} {a['verb']:9s} {a.get('otype_name') or '':7s} "
                f"{tuple(a['to'])}"
                + (f"  E={a['energy']}" if a.get("energy") is not None else ""),
                10.5,
                col,
            )
        )
        y += 14
    y += 14
    for line in notes:
        out.append(_text(x, y, line, 10.5, DIM))
        y += 14
    y += 10
    for otype, col in sorted(TYPE_COLOR.items()):
        out.append(
            f'<rect x="{x:.1f}" y="{y - 8:.1f}" width="9" height="9" fill="{col}"/>'
        )
        out.append(_text(x + 14, y, mm.TYPES[otype], 10, DIM))
        y += 13
    return out


def diagram(state, title="landscape", subtitle=(), acts=(), notes=()):
    """The full annotated isometric SVG for ``state`` as a string."""
    w, h = _canvas()
    defs = "".join(
        f'<marker id="ah_{v}" viewBox="0 0 10 10" refX="9" refY="5" markerWidth="6" '
        f'markerHeight="6" orient="auto-start-reverse">'
        f'<path d="M0,0 L10,5 L0,10 z" fill="{c}"/></marker>'
        for v, c in VERB_COLOR.items()
    )
    body = _terrain_svg(state) + _objects_svg(state)
    seen = {}
    for i, a in enumerate(acts, 1):
        key = tuple(a["to"])
        body += _arrow(state, a, i, seen.get(key, 0))
        seen[key] = seen.get(key, 0) + 1
    body += _panel(
        w - PANEL_W + 10, PAD + 24, title, list(subtitle), state, acts, notes
    )
    for t in range(0, GRID, 2):
        ez = terrain.resolve_ground(state, t, GRID - 1)[0]
        bx, by = project(t + 0.5, GRID + 0.9, ez)
        body.append(_text(bx, by, f"x{t}", 9.5, DIM, "middle"))
        ez = terrain.resolve_ground(state, GRID - 1, t)[0]
        bx, by = project(GRID + 0.9, t + 0.5, ez)
        body.append(_text(bx, by, f"y{t}", 9.5, DIM, "middle"))
    return (
        f'<svg xmlns="http://www.w3.org/2000/svg" width="{w:.0f}" height="{h:.0f}" '
        f'viewBox="0 0 {w:.0f} {h:.0f}">'
        f'<rect width="{w:.0f}" height="{h:.0f}" fill="{BG}"/>'
        f"<defs>{defs}</defs>" + "".join(body) + "</svg>"
    )


def write(svg, path):
    """Write an SVG string, creating the directory."""
    os.makedirs(os.path.dirname(os.path.abspath(path)), exist_ok=True)
    with open(path, "w", encoding="utf-8") as fh:
        fh.write(svg)
    return path


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("landscape", type=int, help="the landscape number you TYPE")
    parser.add_argument("--out", default=None)
    args = parser.parse_args(argv)
    game = Game.typed(args.landscape)
    svg = diagram(
        game.state,
        f"ls{args.landscape} at entry",
        [f"generate seed {landscape.seed_for(args.landscape)}"],
    )
    print(f"wrote {write(svg, args.out or f'renders/ls{args.landscape}_iso.svg')}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
