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
        "cycle_residual",
        "pass_phase",
        "body_stage",
        "body_index",
        "body_partial",
        "body_paid",
        "camera_shift",
        "camera_clear",
        "steal_residue",
        "clock_overhang",
        "entry_b",
        "carry_step",
    )

    def __init__(
        self,
        mem,
        cycle_residual=0,
        pass_phase=0,
        body_stage=0,
        body_index=0,
        body_partial=-1,
        body_paid=0,
        camera_shift=0,
        camera_clear=0,
        steal_residue=0,
        clock_overhang=0,
        entry_b=0,
        carry_step=0,
    ):
        self.mem = mem
        self.cycle_residual = cycle_residual  # cycles owed at the sub-pass resume point
        self.pass_phase = pass_phase  # WHICH resume point: enemies.PHASE_*
        self.body_stage = body_stage  # resume point INSIDE $16E6: enemies.BODY_*
        self.body_index = body_index  # its scan slot; ~slot == charged, commit pending
        self.body_partial = body_partial  # $17B2's held partially-visible player
        self.body_paid = body_paid  # cycles that stage has paid: its committed writes
        self.camera_shift = camera_shift  # $1FC2's outstanding strip $0C62, 0 for none
        self.camera_clear = (
            camera_clear  # the residual $1FC2 waits for: the $2211 clear
        )
        self.steal_residue = steal_residue  # badline.WEIGHT_SHIFT refund carried over
        self.clock_overhang = clock_overhang  # clock past the raster: the caught term
        self.entry_b = entry_b  # of which the caught INSTRUCTION owes, before the IRQ
        self.carry_step = carry_step  # that term's per-window refund, for its own tail
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
        return State(
            bytearray(self.mem),
            self.cycle_residual,
            self.pass_phase,
            self.body_stage,
            self.body_index,
            self.body_partial,
            self.body_paid,
            self.camera_shift,
            self.camera_clear,
            self.steal_residue,
            self.clock_overhang,
            self.entry_b,
            self.carry_step,
        )

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
