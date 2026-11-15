"""Whether an observer can see a tile: the ROM's object visibility with the facing gate
dropped, a phantom robot standing in on bare terrain.

The phantom mirrors :func:`sentinel.actions.create`'s placement without spending energy
or drawing the PRNG; every query runs on a clone, so the caller's state is untouched.
"""

from sentinel import memmap as mm, relative, terrain

FOV_FULL = 0x100  # fov_width that drops the facing gate (in_fov always True)


def _free_slot(state):
    """The highest empty object slot, or None."""
    free = state.free_slots()
    return free[-1] if free else None


def _place_phantom(state, tile, slot):
    """Place a phantom T_ROBOT at `tile` in `slot`, mirroring actions.create's
    bare-terrain / stack-on-boulder placement (z_height, z_frac, obj_flags, tile
    byte, obj_type). No energy is spent and no PRNG is drawn. Returns True, or
    False if the tile can't be stood on (a non-boulder object is on top)."""
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


def player_sees_tile(state, tile, observer_slot, eye_z=None):
    """Whether the observer at `observer_slot` sees `tile` (x, y) by the ROM's direct
    line of sight (relative.can_see_object, facing gate dropped).

    An occupied tile tests the object in its slot, bare terrain a phantom T_ROBOT;
    `eye_z` overrides the observer's height.  Runs on a clone."""
    clone = state.clone()
    if eye_z is not None:
        clone.obj_z_height[observer_slot] = int(eye_z) & 0xFF
    x, y = tile
    b = terrain.tile_byte(clone, x, y)
    if b >= mm.OBJECT_TILE:  # occupied -> test the real object in its slot
        slot = b & 0x3F
        typ = clone.obj_type[slot]
        return (
            relative.can_see_object(clone, observer_slot, slot, typ, FOV_FULL)[
                "exposure"
            ]
            > 0
        )
    slot = _free_slot(clone)  # bare terrain -> phantom-place a robot and test it
    if slot is None or not _place_phantom(clone, (x, y), slot):
        return False
    return (
        relative.can_see_object(clone, observer_slot, slot, mm.T_ROBOT, FOV_FULL)[
            "exposure"
        ]
        > 0
    )
