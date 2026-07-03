"""Compute a keyboard-lattice sights view (h_angle, v_angle, cursor) that aims at a tile,
by INVERTING the action-time aim (prepare_vector_from_player_sights $1C10) rather than
searching a cursor grid.

The aim vector is vx=sin(H)cos(V), vy=cos(H)cos(V), vz=sin(V), with z marching 1:1 with
tile widths, so the bearing is H=atan2(dx,dy) and the pitch is V=atan2(dz,dhoriz) (256=full
circle).  The forward keyboard map is H_eff=h+(cx>>3)-10 (cx low 3 bits -> 1/8-unit h_frac,
bit-reversed) and V_eff=v+((cy-5)>>4)+3 (cy low 4 bits -> v_frac).  Bearing inverts in closed
form; pitch is seeded from geometry then solved 1-D against the game's own marched tile
($003A/$003C after handle_player_actions $1B18), which is monotonic in cy.  On an occupied
(boulder/platform) tile the create needs a near-centre aim ($1E48 <$40), refined locally.
The real gate (CodeEngine.create_via_gate) is the arbiter throughout.
"""

import math

from game_state import (
    OBJECTS_X,
    OBJECTS_Y,
    OBJECTS_Z_HEIGHT,
    OBJECTS_Z_FRACTION,
    OBJECTS_H_ANGLE,
    OBJECTS_V_ANGLE,
)

TWO_PI = 2 * math.pi
_TIDX = lambda x, y: (x & 3) * 256 + ((x >> 2) & 7) * 32 + y


def _nearest_1mod4(x):
    return 4 * int(round((x - 1) / 4.0)) + 1


def _tile_top(eng, tile):
    b = eng.mem[0x0400 + _TIDX(*tile)]
    if b >= 0xC0:
        s = b & 0x3F
        return eng.mem[OBJECTS_Z_HEIGHT + s] + eng.mem[OBJECTS_Z_FRACTION + s] / 256.0
    return float(b >> 4)


def is_occupied(eng, tile):
    return eng.mem[0x0400 + _TIDX(*tile)] >= 0xC0


def _seed(eng, tile):
    ps = eng.player_slot
    px, py = eng.mem[OBJECTS_X + ps], eng.mem[OBJECTS_Y + ps]
    eye = eng.mem[OBJECTS_Z_HEIGHT + ps] + eng.mem[OBJECTS_Z_FRACTION + ps] / 256.0
    dx, dy = tile[0] - px, tile[1] - py
    dz = _tile_top(eng, tile) - eye
    dh = math.hypot(dx, dy)
    hf = (math.atan2(dx, dy) / TWO_PI * 256) % 256
    vf = math.atan2(dz, dh) / TWO_PI * 256
    base_h = (int(round(hf / 8.0)) * 8) & 0xFF
    need_h = ((hf - base_h + 10) + 256) % 256
    m = int(round(need_h))
    cx0 = min(143, max(16, 8 * m))
    base_v = _nearest_1mod4(vf - 8) & 0xFF
    return base_h, cx0, base_v, (px, py), (dx, dy)


def _marched(eng, otype, h, v, cx, cy):
    """Drive the real aim (handle_player_actions $1B18) and read the marched tile
    $003A/$003C without committing a create.  Restores memory."""
    ps = eng.player_slot
    m = eng.mem
    saved = bytes(m)
    m[OBJECTS_H_ANGLE + ps] = h & 0xFF
    m[OBJECTS_V_ANGLE + ps] = v & 0xFF
    m[0x0CC6] = cx & 0xFF
    m[0x0CC7] = cy & 0xFF
    m[0x0C61] = otype & 0xFF
    m[0x006E] = ps
    eng._call(0x1B18)
    mt = (m[0x003A], m[0x003C])
    eng.mem[:] = saved
    return mt


def _try(eng, otype, tile, h, v, cx, cy):
    """PURE: validate a candidate view against the real gate, always restoring memory
    (the caller commits the create once, on the returned view)."""
    if not (16 <= cx <= 143 and 32 <= cy <= 158):
        return False
    saved = bytes(eng.mem)
    ok = eng.create_via_gate(
        otype,
        tile,
        {"h_angle": h & 0xFF, "v_angle": v & 0xFF, "cursor": (cx & 0xFF, cy & 0xFF)},
    ).get("ok")
    eng.mem[:] = saved
    return ok


def _bisect_cy(eng, otype, tile, base_h, bv, cx, progress, target_prog):
    """Find a cursor-y that lands the marched ray ON `tile` for a fixed bearing/pitch
    lattice, by bisection (landing distance is monotonic in cy).  Returns cy or None."""
    lo, hi = 32, 158
    p_lo = progress(_marched(eng, otype, base_h, bv, cx, lo))
    p_hi = progress(_marched(eng, otype, base_h, bv, cx, hi))
    for _ in range(9):
        mid = (lo + hi) // 2
        mt = _marched(eng, otype, base_h, bv, cx, mid)
        if mt == tile:
            return mid
        pm = progress(mt)
        if (p_hi - p_lo) >= 0:
            lo, hi = (mid, hi) if pm < target_prog else (lo, mid)
        else:
            lo, hi = (lo, mid) if pm < target_prog else (mid, hi)
    return None


def solve_view(eng, tile, otype):
    """Return a keyboard view dict {h_angle, v_angle, cursor} the REAL ROM gate accepts
    for building `otype` on `tile` from the current player state, or None.  Deterministic:
    closed-form bearing + seeded pitch solve, all ROM-arbitrated (no committed side effect).

    Bare terrain builds need only line-of-sight, so the bearing seed + a 1-D pitch (cy)
    bisection suffices.  Occupied tiles (boulder/platform) need a near-centre aim
    ($1E48 <$40) whose window is narrow in BOTH cursor axes, so we ring-search the
    horizontal cursor (fine bearing) around the seed as well.
    """
    tile = tuple(tile)
    base_h, cx0, base_v0, (px, py), (dx, dy) = _seed(eng, tile)
    target_prog = dx * dx + dy * dy

    def progress(mt):
        return (mt[0] - px) * dx + (mt[1] - py) * dy

    occ = is_occupied(eng, tile)
    # horizontal cursor candidates (fine bearing): the seed first (usual first-try hit),
    # then a widening ring -- a grazing far shot or an occupied-tile centre window can sit
    # a few bearing units off the seed, supplied by the cursor.
    cx_list = [cx0] + [cx0 + s * d for d in range(2, 17, 2) for s in (-1, 1)]
    cx_list = [c for c in cx_list if 16 <= c <= 143]
    bv_list = [(base_v0 + 4 * i) & 0xFF for i in (0, -1, 1, -2, 2, -3, 3, -4, 4, -5, 5)]

    for bv in bv_list:
        for cx in cx_list:
            cy = _bisect_cy(eng, otype, tile, base_h, bv, cx, progress, target_prog)
            if cy is None:
                continue
            if _try(eng, otype, tile, base_h, bv, cx, cy):
                return {"h_angle": base_h, "v_angle": bv, "cursor": (cx, cy)}
            if occ:  # landed but centre gate refused: nudge cy within the centre window
                for dcy in (-2, 2, -4, 4, -1, 1, -3, 3, -5, 5):
                    if _try(eng, otype, tile, base_h, bv, cx, cy + dcy):
                        return {
                            "h_angle": base_h,
                            "v_angle": bv,
                            "cursor": (cx, cy + dcy),
                        }
    return None
