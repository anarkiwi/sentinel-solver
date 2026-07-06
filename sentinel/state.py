"""The one game state: a 64 KB memory image plus live, O(1) views over the
object arrays.

Everything -- terrain, line-of-sight, actions, enemies -- reads and writes this
single ``bytearray`` at the game's own addresses, so a mutation made by one
mechanic (e.g. a create placing a boulder) is immediately visible to another
(e.g. the line-of-sight that must now occlude on it).  The object-array views
are thin proxies into ``mem`` rather than snapshots, which is what keeps them
consistent as the state changes.
"""

from sentinel import memmap as mm


class _ObjArray:
    """A live view of one 64-entry object array, indexed by slot."""

    __slots__ = ("mem", "base")

    def __init__(self, mem, base):
        self.mem = mem
        self.base = base

    def __getitem__(self, slot):
        return self.mem[self.base + slot]

    def __setitem__(self, slot, value):
        self.mem[self.base + slot] = value & 0xFF


class State:
    """A mutable game state backed by a 64 KB ``bytearray``.

    The object arrays (``obj_x``, ``obj_type``, ...) are live views over ``mem``;
    the scalars (``player``, ``energy``) are read straight from their addresses.
    """

    __slots__ = (
        "mem",
        "obj_x",
        "obj_y",
        "obj_z_height",
        "obj_z_frac",
        "obj_h_angle",
        "obj_v_angle",
        "obj_flags",
        "obj_type",
    )

    def __init__(self, mem):
        self.mem = mem
        self._bind()

    def _bind(self):
        mem = self.mem
        self.obj_x = _ObjArray(mem, mm.OBJECTS_X)
        self.obj_y = _ObjArray(mem, mm.OBJECTS_Y)
        self.obj_z_height = _ObjArray(mem, mm.OBJECTS_Z_HEIGHT)
        self.obj_z_frac = _ObjArray(mem, mm.OBJECTS_Z_FRACTION)
        self.obj_h_angle = _ObjArray(mem, mm.OBJECTS_H_ANGLE)
        self.obj_v_angle = _ObjArray(mem, mm.OBJECTS_V_ANGLE)
        self.obj_flags = _ObjArray(mem, mm.OBJECTS_FLAGS)
        self.obj_type = _ObjArray(mem, mm.OBJECTS_TYPE)

    @classmethod
    def from_mem(cls, mem):
        """Wrap a raw 64 KB memory image (bytes or bytearray)."""
        return cls(bytearray(mem))

    def clone(self):
        """A deep copy for branching search: duplicate ``mem`` and rebind the
        views onto it."""
        return State(bytearray(self.mem))

    # ---- scalars --------------------------------------------------------
    @property
    def player(self):
        return self.mem[mm.PLAYER_OBJECT]

    @player.setter
    def player(self, slot):
        self.mem[mm.PLAYER_OBJECT] = slot & 0xFF

    @property
    def energy(self):
        return self.mem[mm.PLAYER_ENERGY]

    @energy.setter
    def energy(self, value):
        self.mem[mm.PLAYER_ENERGY] = value & mm.ENERGY_MASK

    @property
    def platform_xy(self):
        return (self.mem[mm.PLATFORM_X], self.mem[mm.PLATFORM_Y])

    # ---- object queries -------------------------------------------------
    def is_empty(self, slot):
        return bool(self.obj_flags[slot] & 0x80)

    def occupied_slots(self):
        return [s for s in range(mm.NUM_SLOTS) if not self.is_empty(s)]

    def free_slots(self):
        return [s for s in range(mm.NUM_SLOTS) if self.is_empty(s)]

    def slot_of_type(self, otype):
        """Lowest occupied slot of the given object type, or None."""
        for s in range(mm.NUM_SLOTS):
            if not self.is_empty(s) and self.obj_type[s] == otype:
                return s
        return None

    def tile_of(self, slot):
        return (self.obj_x[slot], self.obj_y[slot])

    def player_xy(self):
        p = self.player
        return (self.obj_x[p], self.obj_y[p])

    def eye_z(self, slot=None):
        """The observer's eye height (z_height + z_fraction/256) for a slot,
        defaulting to the player."""
        if slot is None:
            slot = self.player
        return self.obj_z_height[slot] + self.obj_z_frac[slot] / 256.0
