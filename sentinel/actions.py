"""The player actions -- create, absorb, transfer, win -- as bit-exact ports of
the game's own routines, operating on the one :class:`sentinel.state.State`.

  create   try_to_create_object $1BBA (create_object_from_action $2120 +
           put_object_in_tile $1F16 + energy $2136)
  absorb   try_to_absorb_object $1B8E (remove_object $1EEF + energy $2136)
  transfer try_to_transfer_into_object $1B64

These are the *mechanics*; the line-of-sight gate that the game applies before
them ($1B46) is the caller's responsibility (see :mod:`sentinel.los`).  A create
targets the tile the sights marched to.
"""

from sentinel import memmap as mm
from sentinel import energy
from sentinel.prng import Prng
from sentinel.terrain import tile_byte, set_tile_byte

PLATFORM_SLOT = 0x3F  # the platform is always object $3F ($1B78)


def _find_empty_slot(state):
    """find_empty_slot_loop $2120: the highest slot with objects_flags bit7 set."""
    for slot in range(mm.NUM_SLOTS - 1, -1, -1):
        if state.obj_flags[slot] & 0x80:
            return slot
    return None


def can_create(state, otype, tile):
    """Whether a create of `otype` on `tile` is feasible on energy / free-slot /
    stackability grounds (the $1F38 checks), ignoring line of sight."""
    if energy.value(otype) > state.energy:
        return False
    if _find_empty_slot(state) is None:
        return False
    b = tile_byte(state, *tile)
    if b >= mm.OBJECT_TILE:
        below = b & 0x3F
        if state.obj_type[below] not in (mm.T_BOULDER, mm.T_PLATFORM):
            return False
    return True


def create(state, otype, tile):
    """Build `otype` on `tile`.  Returns the new object slot, or None if the
    create is infeasible (no free slot, out of energy, or the tile can't be
    stacked on).  Mirrors try_to_create_object $1BBA exactly, including the prnd
    draw that put_object_in_tile spends on the object's random facing."""
    slot = _find_empty_slot(state)
    if slot is None:
        return None
    # $1BBF-$1BC3: spend the energy first; underflow aborts before placement.
    if not energy.lose(state, otype):
        return None
    state.obj_type[slot] = otype
    b = tile_byte(state, *tile)
    if b >= mm.OBJECT_TILE:  # stacking on an existing object ($1F29)
        below = b & 0x3F
        btype = state.obj_type[below]
        if btype == mm.T_PLATFORM:  # $1F47: the platform is one unit high
            zf = state.obj_z_frac[below]
            z = state.obj_z_height[below] + 1
        elif btype == mm.T_BOULDER:  # $1F52: boulders are half a unit high
            t = state.obj_z_frac[below] + 0x80
            zf = t & 0xFF
            z = state.obj_z_height[below] + (t >> 8)
        else:  # $1F38: can't build on anything else -- refund and abort
            energy.gain(state, otype)
            return None
        state.obj_flags[slot] = 0x40 | below
    else:  # $1F66: bare terrain
        state.obj_flags[slot] = 0x00
        zf = 0xE0
        z = b >> 4
    state.obj_x[slot] = tile[0]
    state.obj_y[slot] = tile[1]
    state.obj_z_height[slot] = z & 0xFF
    state.obj_z_frac[slot] = zf & 0xFF
    set_tile_byte(state, tile[0], tile[1], mm.OBJECT_TILE | slot)
    state.obj_v_angle[slot] = 0xF5  # $1F7E: only the player's v_angle is meaningful
    # $1F83: give the object a random rotation (this prnd draw happens even for a
    # robot, whose facing is then overwritten below).
    prng = Prng().load(state.mem)
    rot = prng.next()
    prng.store(state.mem)
    state.obj_h_angle[slot] = ((rot & 0xF8) + 0x60) & 0xFF  # $1F86-$1F8B
    if otype == mm.T_ROBOT:  # $1BE0: a new synthoid faces the player
        state.obj_h_angle[slot] = state.obj_h_angle[state.player] ^ 0x80
    return slot


def _remove_object(state, slot):
    """remove_object $1EEF: unlink the object and repair the tile it stood on."""
    tile = (state.obj_x[slot], state.obj_y[slot])
    flags = state.obj_flags[slot]
    if flags >= 0x40:  # sat on another object -> that object becomes topmost
        set_tile_byte(state, *tile, mm.OBJECT_TILE | (flags & 0x3F))
    else:  # on the ground -> revert to terrain, slope nibble zeroed
        set_tile_byte(state, *tile, (state.obj_z_height[slot] << 4) & 0xFF)
    state.obj_flags[slot] = 0x80


def can_absorb(state, slot):
    """Whether the object in `slot` can be absorbed (occupied, not a platform),
    ignoring line of sight."""
    if state.obj_flags[slot] & 0x80:
        return False
    return state.obj_type[slot] != mm.T_PLATFORM


def absorb(state, slot):
    """Absorb the object in `slot`, gaining its energy (try_to_absorb_object
    $1B8E + absorb_object $1B9E).  Returns True, or False if the slot is empty or
    a platform.  (Meanie absorption $1BEC is handled with the enemy dynamics.)"""
    if not can_absorb(state, slot):
        return False
    otype = state.obj_type[slot]
    _remove_object(state, slot)
    energy.gain(state, otype)
    return True


def transfer(state, slot):
    """Move the player's viewpoint into robot `slot` (try_to_transfer_into_object
    $1B64).  Returns True, or False if the target isn't a robot.  The eye height
    follows automatically from the target's own z_height/z_fraction."""
    if state.obj_type[slot] != mm.T_ROBOT:
        return False
    state.player = slot
    return True


def on_platform(state):
    """Whether the player currently stands on the Sentinel's platform (slot
    $3F) -- the landscape-complete condition."""
    slot = state.player
    for _ in range(mm.NUM_SLOTS):
        flags = state.obj_flags[slot]
        if flags < 0x40:  # reached the ground without meeting the platform
            return False
        slot = flags & 0x3F
        if slot == PLATFORM_SLOT:
            return True
    return False


def won(state):
    """Whether the landscape is complete (the player is on the platform)."""
    return on_platform(state)


def win(state, tile=None):
    """The endgame: absorb the Sentinel, build a synthoid on its platform tile,
    and transfer onto it.  Returns True if the player ends on the platform.  The
    caller is responsible for having line of sight to each step."""
    sentinel = state.slot_of_type(mm.T_SENTINEL)
    if sentinel is not None:
        absorb(state, sentinel)
    if tile is None:
        tile = state.platform_xy
    slot = create(state, mm.T_ROBOT, tile)
    if slot is None:
        return False
    transfer(state, slot)
    return on_platform(state)
