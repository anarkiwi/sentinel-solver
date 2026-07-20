"""The terrain height field and the object-stack surface resolution.

``tiles_table`` ($0400-$07FF) is a 32x32 *vertex* height field in the game's
interleaved layout.  A byte below $C0 is terrain ``(height<<4) | slope`` (height
0-11, slope nibble 0 = flat); a byte >= $C0 means the tile holds an object whose
index is in the low 6 bits, and the ground there is the bottommost object's
height.

``tile_byte`` reproduces the ROM address arithmetic (calculate_tile_address
$2BA8) exactly -- including how it wraps at the board edge -- which matters
because check_sloping_tile reads a tile's four corner heights from its
*neighbours*, some of which sit off the 32x32 interior.
"""

from sentinel import memmap as mm


def tile_byte(state, x, y):
    """The raw tiles_table byte for tile (x, y), via the ROM addressing.

    ``((x<<3)&0xE0) | (y&0x1F)`` with page ``(x&3)+4`` is provably equal to
    ``TILES_TABLE + tidx(x, y)`` for in-range tiles; the masked 8-bit form is
    kept so edge reads (x+1, y+1 == 32) match the 6502 byte-for-byte.
    """
    x &= 0xFF
    y &= 0xFF
    lo = ((x << 3) & 0xE0) | (y & 0x1F)
    page = (x & 3) + 4
    return state.mem[page * 256 + lo]


def set_tile_byte(state, x, y, value):
    """Write the tiles_table byte for tile (x, y), via the same ROM addressing as
    :func:`tile_byte` (put_object_in_tile / remove_object write through it)."""
    x &= 0xFF
    y &= 0xFF
    lo = ((x << 3) & 0xE0) | (y & 0x1F)
    page = (x & 3) + 4
    state.mem[page * 256 + lo] = value & 0xFF


def bottom_object(state, slot):
    """Walk the flags chain down to the bottommost object slot of a stack."""
    for _ in range(mm.NUM_SLOTS):
        flags = state.obj_flags[slot]
        if flags < 0x40:  # on the ground
            return slot
        slot = flags & 0x3F
    return slot


def top_object(state, x, y):
    """The slot of the topmost object on (x, y), or None for bare terrain."""
    b = tile_byte(state, x, y)
    return (b & 0x3F) if b >= mm.OBJECT_TILE else None


def resolve_ground(state, x, y):
    """(ground_height, slope_nibble) at (x, y).

    For an object tile the ground is the bottommost object's z_height and the
    slope is 0 (object tiles carry no slope nibble)."""
    b = tile_byte(state, x, y)
    if b < mm.OBJECT_TILE:
        return (b >> 4), (b & 0x0F)
    bottom = bottom_object(state, b & 0x3F)
    return state.obj_z_height[bottom], 0
