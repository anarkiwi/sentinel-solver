"""The player actions as bit-exact ports of the game's routines on one :class:`State`.

Create $1BBA ($2120 + $1F16 + energy $2136), absorb $1B8E ($1EEF + $2136), transfer $1B64.
The line-of-sight gate the game applies first ($1B46) is the caller's (:mod:`sentinel.los`);
a create targets the tile the sights marched to.
"""

from sentinel import memmap as mm
from sentinel import energy
from sentinel.prng import Prng
from sentinel.terrain import tile_byte, set_tile_byte

# slot 0 is the Sentinel ($1553); $1B8E's `LDA $0100 / BMI` rejects every absorb once it is gone.
SENTINEL_SLOT = 0


def _find_empty_slot(state):
    """find_empty_slot_loop $2120: the highest slot with objects_flags bit7 set."""
    for slot in range(mm.NUM_SLOTS - 1, -1, -1):
        if state.obj_flags[slot] & 0x80:
            return slot
    return None


def _mark_player_acted(state):
    """$12E1: the player-action dispatch clears $0CE5 bit7 on the first action,
    unfreezing the enemy world clock ($3682/$9659 skip while it is set)."""
    state.mem[mm.PLAYER_NOT_ACTED] = 0x00


def create(state, otype, tile):
    """Build `otype` on `tile`.  Returns the new object slot, or None if the
    create is infeasible (no free slot, out of energy, or the tile can't be
    stacked on).  Mirrors try_to_create_object $1BBA exactly, including the prnd
    draw that put_object_in_tile spends on the object's random facing."""
    _mark_player_acted(state)
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
    # $1F83: a random rotation, drawn even for a robot whose facing is overwritten below.
    prng = Prng().load(state.mem)
    rot = prng.next()
    prng.store(state.mem)
    state.obj_h_angle[slot] = ((rot & 0xF8) + 0x60) & 0xFF  # $1F86-$1F8B
    if otype == mm.T_ROBOT:  # $1BE0: a new synthoid faces the player
        state.obj_h_angle[slot] = state.obj_h_angle[state.player] ^ 0x80
    return slot
