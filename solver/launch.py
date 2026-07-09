"""Down-look launch enumeration + terminal readiness.

The endgame absorb is the player looking DOWN at the platform tile, which is legal
on tiles where the reverse up-shot from the platform's own vantage is blocked (the
ROM ``$1D2E`` asymmetry). Launch tiles are therefore enumerated with the direct
observer->tile geometric march (:func:`sentinel.threat.player_sees_tile`), NOT the
symmetric platform-vantage sweep the old climb used.
"""

from sentinel import memmap as mm, terrain, threat
from solver.plan_game import terrain_z


def _place_phantom(state, tile, slot):
    """Place a phantom T_ROBOT observer at `tile` in `slot` (bare-terrain /
    stack-on-boulder placement, no energy, no PRNG). Returns True, or False when a
    non-boulder object already tops the tile. Inlined from ``threat._place_phantom``."""
    x, y = tile
    b = terrain.tile_byte(state, x, y)
    if b >= mm.OBJECT_TILE:  # stacking on an existing object
        below = b & 0x3F
        if state.obj_type[below] != mm.T_BOULDER:
            return False
        t = state.obj_z_frac[below] + 0x80  # boulders are half a unit high
        zf = t & 0xFF
        z = (state.obj_z_height[below] + (t >> 8)) & 0xFF
        flags = 0x40 | below
    else:  # bare terrain
        zf = 0xE0
        z = (b >> 4) & 0xFF
        flags = 0x00
    state.obj_type[slot] = mm.T_ROBOT
    state.obj_flags[slot] = flags
    state.obj_x[slot] = x
    state.obj_y[slot] = y
    state.obj_z_height[slot] = z
    state.obj_z_frac[slot] = zf
    terrain.set_tile_byte(state, x, y, mm.OBJECT_TILE | slot)
    return True


def _restore_tile(state, tile, old_tile_byte, slot):
    """Undo :func:`_place_phantom`: restore the tile byte and mark `slot` empty."""
    terrain.set_tile_byte(state, tile[0], tile[1], old_tile_byte)
    state.obj_flags[slot] |= 0x80


def launch_tiles(state, plat, plat_ground) -> set:
    """Every tile from which a robot standing at its terrain height has DOWN-look
    line of sight to the platform tile `plat`.

    Each candidate tile is phantom-placed with a robot observer and tested with the
    direct geometric march ``threat.player_sees_tile(clone, plat, slot, eye_z=tz)``
    (`tz` = the tile's terrain height). Only tiles with ``terrain_z >= plat_ground - 1``
    are probed: a launch robot must be able to build one unit and clear the platform
    ground. (ls0's terrain caps at ``plat_ground - 1``, so the plan's literal
    ``>= plat_ground`` floor would admit no tile; the launch height the human fired
    from -- ``(2,10)`` at terrain 8 -- sits exactly at ``plat_ground - 1``.) The
    caller's state is never mutated."""
    plat = tuple(plat)
    board = state.clone()
    free = board.free_slots()
    if not free:
        return set()
    slot = free[-1]
    floor = plat_ground - 1
    out = set()
    for y in range(mm.N):
        for x in range(mm.N):
            tz = terrain_z(board, x, y)
            if tz is None or tz < floor:
                continue
            old = terrain.tile_byte(board, x, y)
            if not _place_phantom(board, (x, y), slot):
                continue
            if threat.player_sees_tile(board, plat, slot, eye_z=tz):
                out.add((x, y))
            _restore_tile(board, (x, y), old, slot)
    return out


def down_look_los(g, plat) -> bool:
    """Line of sight from the PlanGame `g`'s CURRENT tile/eye down to `plat`, using
    the player's TRUE eye (``eye_z=None`` reads the real z_height + z_frac from the
    object, exactly what the ROM's aim march uses at fire time). An earlier ceil of
    the eye to the next integer over-estimated the observer by up to ~1 tile and
    granted a marginal far down-look LOS the real eye does not have -- the endgame
    then fired blind and missed live (the (5,16)->(12,4) ls0 launch)."""
    return threat.player_sees_tile(g.state, tuple(plat), g.player, eye_z=None)


def endgame_ready(g, plat, plat_ground) -> bool:
    """The player is above the platform ground and has down-look LOS to it -- the
    geometric precondition for the endgame absorb."""
    return g.eye > plat_ground and down_look_los(g, plat)
