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
# slot 0 is ALWAYS the Sentinel (is_sentinel $1553).  try_to_absorb_object $1B8E opens with an
# ABSOLUTE `LDA $0100 / BMI` over objects_flags[0]; once the Sentinel is absorbed (its slot
# SLOT_EMPTY) that branch rejects EVERY absorb -- of any object, meanies included.
SENTINEL_SLOT = 0


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


# public alias: removing an object is also part of enemy draining (a tree drained
# to nothing) and the win sequence, not only player absorption.
remove_object = _remove_object


def can_absorb(state, slot):
    """Whether the object in `slot` can be absorbed (the Sentinel still exists, the
    slot is occupied and is not a platform), ignoring line of sight."""
    # $1B8E/$1B91: the Sentinel-still-exists gate comes FIRST -- an absolute read of
    # objects_flags[0].  Absorbing the Sentinel itself passes here (its slot is still
    # occupied at the time of the check).
    if state.obj_flags[SENTINEL_SLOT] & 0x80:
        return False
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


def hyperspace(state):
    """do_hyperspace $2156: the player's panic teleport, and the terminal move of a
    win.  Creates a new robot (type 0, $2158), places it on a random flat empty tile
    no higher than the player's ``z_height + 1`` ($215D-$2165) via the shared
    random-tile-below-z placement, and spends 3 energy -- the robot value -- through
    the energy economy ($216A).

    * underflow -> DEATH: the new robot is removed and $0CDE bit7 (only) is set, with
      NO relocation ($2170-$217A);
    * survived ($217F): if the player's CURRENT tile is the platform tile
      ($0C19/$0C1A) the landscape is complete, $0CDE = $C0 (bit7 hyperspaced + bit6
      complete, $2196); the player then transfers into the new robot ($21A5).

    The PRNG-driven landing is deliberately not steerable (a faithful solver must
    treat it as unknown).  This is the same ROM routine a meanie forces on the player
    (:func:`sentinel.enemies.do_hyperspace`); the win path drives it directly.
    Returns True if the player survived the hyperspace."""
    from sentinel import enemies  # deferred: enemies imports actions

    enemies.do_hyperspace(state)
    return not player_dead(state)


def won(state):
    """Whether the landscape is complete -- the player HYPERSPACED while standing on
    the platform tile, setting $0CDE to $C0 (player_survived_hyperspace $217F/$2196).
    Merely standing on the platform (:func:`on_platform`) is NOT a win."""
    return (state.mem[mm.PLAYER_HAS_HYPERSPACED] & 0xC0) == 0xC0


def player_dead(state):
    """Whether the player has been killed. Two ROM death paths:
    * drained at 0 energy -- kill_player $1A00 sets the $0C4E flag; and
    * a meanie forcibly hyperspaced the player with too little energy to survive it
      -- do_hyperspace $215F sets PLAYER_HAS_HYPERSPACED ($0CDE bit7) and does NOT
      relocate the player.  A WIN also touches $0CDE (bit6, landscape complete), so a
      death is bit7 set with bit6 clear ($0CDE & $C0 == $80).  The simulator never
      voluntarily hyperspaces, so a lone bit7 during enemy stepping is a meanie kill."""
    if state.mem[mm.PLAYER_DIED_BY_DRAINING] & 0x80:
        return True
    return (state.mem[mm.PLAYER_HAS_HYPERSPACED] & 0xC0) == 0x80


def win(state, tile=None):
    """The endgame (docs/gameplay.md §1 "How a human wins"): absorb the Sentinel,
    build a synthoid on its platform tile, transfer onto it, then HYPERSPACE from the
    platform -- the terminal move that actually sets the landscape-complete flag
    (player_survived_hyperspace $217F).  Returns :func:`won`.  The caller is
    responsible for having line of sight to each step."""
    sentinel = state.slot_of_type(mm.T_SENTINEL)
    if sentinel is not None:
        absorb(state, sentinel)
    if tile is None:
        tile = state.platform_xy
    slot = create(state, mm.T_ROBOT, tile)
    if slot is None:
        return False
    transfer(state, slot)
    hyperspace(state)
    return won(state)
