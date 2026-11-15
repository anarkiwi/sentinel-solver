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
EXPOSURE_BAR = 0x0C4F  # set_bar_state $194D: bit7 fully seen, bit6 partly
PLAYER_DIED_BY_DRAINING = 0x0C4E  # bit7 set by kill_player $1A00 (drain at 0 energy)

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
ENEMIES_UPDATE_COOLDOWN = 0x0C30  # $0C30
ROTATION_SPEED_TABLE = 0x9D37  # per-enemy rotation step (+$14 / $EC), indexed by slot

# ---- enemy-update scalars -------------------------------------------------
COOLDOWN_GATE = 0x0C50  # $0C50 gates update_enemy_cooldowns (1-in-3 cadence)
# The running game ticks cooldowns ONCE PER FRAME (raster $9663 / scroll $3684, mutually
# exclusive via $0CD8) through this integer Bresenham accumulator: $130C does $1335 += $CD
# and only falls through to update_enemy_cooldowns ($1317) on carry (205/256).  Suppressed
# until the player's first action ($0CE5).
COOLDOWN_BRESENHAM_STEP = 0xCD  # $1310 ADC #$CD
PLAYER_NOT_ACTED = 0x0CE5  # bit7 set until the player's first action ($12E1 clears it)
# $9A51 is its only writer: $3577 stores $80 before play_landscape_loop $357D, $117E stores 0 for generation and the preview, so it is $80 for the whole of play.  $283D reads it to pick the plot_world examine ($2845's trig, or $37F2's table read), $95FA the raster split chain.
PLAY_DISPLAY_FLAG = 0x9AF6
FURTHEST_ROW_HINT = 0x0C48  # $26CD seeds $27D7's first column from it, $26D9 stores the furthest row's own start back: plot_world's only carry between passes
TARGETED_OBJECT_SLOT = 0x0C58  # $0C58 slot the LOS march recognises

# ---- the PRNG state -------------------------------------------------------
PRND_STATE = 0x0C7B  # 5-byte LFSR state $0C7B-$0C7F (prnd $31CA)

# ---- sound voices: $963D JSR $FFC2 runs the $8ED1 note tick every frame ----
SOUND_NOTE_TIMER = 0x8E86  # $8E86..$8E88: frames left of this note ($80 == voice idle)
SOUND_VOICE_FLAG = 0x8E96  # $8E96..$8E98: bit7 == the voice is off ($8EF8 sets it)
SOUND_VOICE_IDLE = 0x80  # the resting note timer, MEASURED at landscape entry

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
