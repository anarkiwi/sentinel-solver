#!/usr/bin/env python3
"""Down-look launch enumeration (planner redesign T1.1).

The endgame shot is the player looking DOWN at the platform.  A down-shot is
legal where the reverse up-shot is blocked -- the ROM's looking-up waiver
(``$1D2E``).  So the launch-candidate set is asymmetric: it is NOT the
platform-vantage reverse sweep ``climb_search._launch_tiles`` uses (that
under-counts far launch tiles and over-counts near ones that cannot actually be
down-shot).  It is enumerated by testing the down-shot FROM each sufficiently
high tile, honoring the waiver.

:func:`launch_tiles` phantom-places a robot on each candidate tile (the placement
math is inlined from :func:`sentinel.threat._place_phantom`) and asks
:func:`sentinel.los.sees_tile` whether that observer, aimed down, has line of
sight to the platform tile.  :func:`down_look_los` / :func:`endgame_ready` answer
the same question for the player's CURRENT tile/eye.
"""

from sentinel import los, terrain
from sentinel import memmap as mm
from solver.plan_game import terrain_z, N

# A robot's top above its foot tile ($E0 render fraction); the bare-terrain
# phantom eye fraction (mirrors sentinel.threat._place_phantom / actions.create).
_ROBOT_ZF = 0xE0


def _free_slot(state):
    """The highest empty object slot, or None."""
    free = state.free_slots()
    return free[-1] if free else None


def _place_phantom(state, tile, slot):
    """Place a phantom ``T_ROBOT`` at ``tile`` in ``slot``, inlined from
    :func:`sentinel.threat._place_phantom`: bare-terrain / stack-on-boulder
    placement of ``obj_type``/``obj_flags``/``obj_x``/``obj_y``/
    ``obj_z_height``/``obj_z_frac`` plus the tile byte, so ``los.sees_tile``
    treats the phantom as the observer.  No energy is spent and no PRNG is drawn.
    Returns the phantom's foot ``z_height``, or None when the tile can't be stood
    on (a non-boulder object is on top)."""
    x, y = tile
    b = terrain.tile_byte(state, x, y)
    if b >= mm.OBJECT_TILE:  # stacking on an existing object
        below = b & 0x3F
        if state.obj_type[below] != mm.T_BOULDER:
            return None
        t = state.obj_z_frac[below] + 0x80  # boulders are half a unit high
        zf = t & 0xFF
        z = (state.obj_z_height[below] + (t >> 8)) & 0xFF
        flags = 0x40 | below
    else:  # bare terrain
        zf = _ROBOT_ZF
        z = (b >> 4) & 0xFF
        flags = 0x00
    state.obj_type[slot] = mm.T_ROBOT
    state.obj_flags[slot] = flags
    state.obj_x[slot] = x
    state.obj_y[slot] = y
    state.obj_z_height[slot] = z
    state.obj_z_frac[slot] = zf
    terrain.set_tile_byte(state, x, y, mm.OBJECT_TILE | slot)
    return z


def launch_tiles(state, plat, plat_ground):
    """Every tile from which a robot standing there has down-look line of sight to
    the platform tile ``plat``.

    A candidate is any tile with ``terrain_z >= plat_ground``.  A phantom robot is
    placed there and the down-shot to ``plat`` is tested via
    ``los.sees_tile(clone, plat, slot, eye_z=tile_eye)``.  ``tile_eye`` is the
    launch eye: the tile's own terrain height when that already overlooks the
    platform (``> plat_ground``), else the minimal built launch height
    ``plat_ground + 1`` -- a down-shot needs the eye strictly above the platform
    surface, and a landscape (like ls0) whose platform sits at the maximum terrain
    height is only launch-able from BUILT height.  A higher eye only ever adds
    down-look line of sight, so the minimal launch eye is the strictest test.

    Returns a ``set`` of ``(x, y)``.  ``plat`` itself is excluded.
    """
    plat = tuple(plat)
    launch_eye = plat_ground + 1
    result = set()
    for y in range(N):
        for x in range(N):
            if (x, y) == plat:
                continue
            tz = terrain_z(state, x, y)
            if tz is None or tz < plat_ground:
                continue
            clone = state.clone()
            slot = _free_slot(clone)
            if slot is None:
                continue
            if _place_phantom(clone, (x, y), slot) is None:
                continue
            tile_eye = max(int(tz), launch_eye)
            if los.sees_tile(clone, plat, slot, eye_z=tile_eye, max_steps=200):
                result.add((x, y))
    return result


def down_look_los(g, plat):
    """Line of sight from the player's CURRENT tile/eye down to ``plat``.

    A fractional eye above the platform ground sees DOWN onto it, so the observer
    is ceil'd when the eye carries a fraction (``seye = int(eye) + 1`` when
    ``eye > int(eye)``, else ``int(eye)``) -- matching ``climb_search.endgame``'s
    ``seye``.  Returns bool.
    """
    eye = g.eye
    ie = int(eye)
    seye = ie + 1 if eye > ie else ie
    return los.sees_tile(g.state, tuple(plat), g.player, eye_z=seye, max_steps=200)


def endgame_ready(g, plat, plat_ground):
    """Whether the endgame down-shot can launch from HERE: the eye strictly above
    the platform ground (a fractional eye counts) AND down-look line of sight to
    the platform tile."""
    return g.eye > plat_ground and down_look_los(g, plat)
