"""The game's RAM layout -- the one source of truth for every address and the
interleaved tile-index formula.

The simulator keeps its whole state in a 64 KB ``bytearray`` addressed exactly
like the C64's RAM, so these constants are shared by terrain, LOS, actions,
enemies and the landscape generator alike.  Each address is annotated with the
routine that reads or writes it.
"""

N = 32  # the board is 32x32 tiles
NUM_SLOTS = 64  # the object arrays are 64 entries each
MEM_SIZE = 0x10000  # full 64 KB address space

# ---- scalars --------------------------------------------------------------
PLAYER_OBJECT = 0x000B  # index of the player's object slot
CURSOR = 0x0090  # which enemy slot update_enemies processes this round (7->0)
MAX_ENEMIES = 0x0C07  # maximum_number_of_enemies
VERTICAL_SCALE = 0x0C08  # landscape_vertical_scale
PLAYER_ENERGY = 0x0C0A  # player_energy, masked AND #$3F by set_player_energy $2148
PLATFORM_X = 0x0C19  # tile x of the Sentinel's platform
PLATFORM_Y = 0x0C1A  # tile y of the Sentinel's platform
FOV_WIDTH = 0x0C68  # enemy horizontal FOV width, reloaded to $14 each scan ($16F2)
LANDSCAPE_COMPLETE = 0x0CDE  # bit6 set by landscape_completed $3603 on a win

# ---- the object arrays (64 slots each) ------------------------------------
OBJECTS_FLAGS = 0x0100  # bit7 => empty; <$40 on ground; $40-$7F stacked on (v&$3F)
OBJECTS_V_ANGLE = 0x0140  # vertical tilt
OBJECTS_X = 0x0900  # tile x (first horizontal axis)
OBJECTS_Z_HEIGHT = 0x0940  # the VERTICAL / height axis
OBJECTS_Y = 0x0980  # tile y (second horizontal axis)
OBJECTS_H_ANGLE = 0x09C0  # horizontal facing (0..255 => 0..360 deg)
OBJECTS_Z_FRACTION = 0x0A00  # sub-unit height fraction
OBJECTS_TYPE = 0x0A40  # 0..6 (see TYPES)

# ---- the terrain grid -----------------------------------------------------
TILES_TABLE = 0x0400  # $0400-$07FF, 32x32, interleaved (calculate_tile_address)

# ---- per-enemy phase arrays (8 entries each) ------------------------------
ENEMIES_DRAINING_COOLDOWN = 0x0C20  # $0C20
ENEMIES_ROTATION_COOLDOWN = 0x0C28  # $0C28
ENEMIES_UPDATE_COOLDOWN = 0x0C30  # $0C30
ENEMIES_MEANIE_SEARCH_OBJECT = 0x0C80  # $0C80
ENEMIES_ENERGY_TO_DISCHARGE = 0x0C88  # $0C88
ENEMIES_FAILED_MEANIE_MEMORY = 0x0C90  # $0C90
ENEMIES_MEANIE_ATTEMPT_SCANS = 0x0C98  # $0C98
ENEMIES_MEANIE_OBJECT = 0x0CA0  # $0CA0 (top bit set == no meanie)
ENEMIES_TARGETED_OBJECT = 0x0CA8  # $0CA8
ENEMIES_TARGETED_OBJECT_EXPOSURE = 0x0CB0  # $0CB0 (top bit == fully visible)
ENEMIES_CONSIDERING_MEANIE = 0x0CB8  # $0CB8
ROTATION_SPEED_TABLE = 0x9D37  # per-enemy rotation step (+$14 / $EC), indexed by slot

# ---- enemy-update scalars -------------------------------------------------
COOLDOWN_GATE = 0x0C50  # $0C50 gates update_enemy_cooldowns (1-in-3 cadence)
TARGETED_OBJECT_SLOT = 0x0C58  # $0C58 slot the LOS march recognises
FOV_RELATIVE_H_ANGLE = 0x0C57  # object_relative_h_angle_high (bearing + $0A)

# ---- the PRNG state -------------------------------------------------------
PRND_STATE = 0x0C7B  # 5-byte LFSR state $0C7B-$0C7F (prnd $31CA)

# ---- object types ---------------------------------------------------------
T_ROBOT = 0  # also the player synthoid
T_SENTRY = 1
T_TREE = 2
T_BOULDER = 3
T_MEANIE = 4
T_SENTINEL = 5
T_PLATFORM = 6

TYPES = {
    T_ROBOT: "ROBOT",
    T_SENTRY: "SENTRY",
    T_TREE: "TREE",
    T_BOULDER: "BOULDER",
    T_MEANIE: "MEANIE",
    T_SENTINEL: "SENTINEL",
    T_PLATFORM: "PLATFORM",
}

ENEMY_TYPES = (T_SENTRY, T_SENTINEL)

# Energy value of each object type (energy_in_objects table $214F): absorbing an
# object adds this, creating one subtracts it.  Platform (6) is never absorbed
# or created.
ENERGY_IN_OBJECTS = {
    T_ROBOT: 3,
    T_SENTRY: 3,
    T_TREE: 1,
    T_BOULDER: 2,
    T_MEANIE: 1,
    T_SENTINEL: 4,
    T_PLATFORM: 0,
}
ENERGY_MASK = 0x3F  # set_player_energy $2148 AND #$3F

# A tiles_table byte >= OBJECT_TILE holds an object index in its low 6 bits;
# below it the byte is (height<<4)|slope terrain.
OBJECT_TILE = 0xC0


def tidx(x, y):
    """Index into tiles_table for tile (x, y).

    The game's own interleaved layout (calculate_tile_address $28D4 / $2BA8):
    ``$0400 + (x&3)*256 + ((x>>2)&7)*32 + y`` -- NOT row-major.
    """
    return (x & 3) * 256 + ((x >> 2) & 7) * 32 + y
