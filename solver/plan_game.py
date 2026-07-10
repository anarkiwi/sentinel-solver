#!/usr/bin/env python3
"""The solver's mutable planning state on a bit-exact ``sentinel.State``.

``PlanGame`` tracks the planner's derived view of the climb -- ``col`` (tile ->
object-stack top height), ``eye``, ``steps``, and feasibility -- while every
mechanic is delegated to the ``sentinel`` package so the plan is ROM-faithful.
"""

from sentinel import landscape as _landscape
from sentinel import actions, aim
from sentinel import memmap as mm
from sentinel.state import State

N = mm.N

ROBOT_EYE_FUDGE = 2  # build-height slack: top <= eye+2 ($1F38 sightline)
_OBJECT_TILE = mm.OBJECT_TILE


def cheb(a, b):
    return max(abs(a[0] - b[0]), abs(a[1] - b[1]))


def terrain_z(mem_or_state, x, y):
    """Bare-terrain height nibble at (x, y), or None when off-board or the tile
    holds an object.  Accepts either a raw 64 KB mem (bytes/bytearray) or a
    :class:`sentinel.state.State`."""
    if not (0 <= x < N and 0 <= y < N):
        return None
    mem = getattr(mem_or_state, "mem", mem_or_state)
    b = mem[mm.TILES_TABLE + mm.tidx(x, y)]
    return (b >> 4) if b < _OBJECT_TILE else None


class PlanGame:
    """Mutable game state on a ``sentinel.State``, matching ``native_game.Game``.

    ``col`` (tile -> object-stack top height) and ``eye`` are tracked exactly as
    ``native_game`` did so the planners' height comparisons are unchanged; all
    object placement / removal / energy goes through :mod:`sentinel.actions`.
    """

    def __init__(self, landscape):
        state = _landscape.generate(landscape)
        state.mem[mm.CURSOR] = 7
        state.mem[mm.COOLDOWN_GATE] = 0
        self._init(state, landscape, seed_built_columns=False)

    @classmethod
    def from_mem(cls, mem, landscape=None, seed_built_columns=True):
        """Wrap a live/raw 64 KB memory image.  ``seed_built_columns`` mirrors
        ``native_game._init_from_mem``: True seeds ``col`` for every occupied tile
        (live resync mid-climb); False uses the fresh-landscape terrain-height
        start eye (ROM-validated offline baseline)."""
        g = cls.__new__(cls)
        g._init(State.from_mem(mem), landscape, seed_built_columns=seed_built_columns)
        return g

    def _init(self, state, landscape, seed_built_columns):
        self.state = state
        self.mem = state.mem
        self.landscape = landscape
        self.col = {}
        self.steps = []
        self.native_won = False
        if seed_built_columns:
            for ty in range(N):
                for tx in range(N):
                    b = self.mem[mm.TILES_TABLE + mm.tidx(tx, ty)]
                    if b >= _OBJECT_TILE:
                        top = b & 0x3F
                        self.col[(tx, ty)] = (
                            state.obj_z_height[top] + state.obj_z_frac[top] / 256.0
                        )
        px, py = self.player_xy()
        p = state.player
        if seed_built_columns:
            if (px, py) in self.col:
                self.eye = self.col[(px, py)]
            else:
                ground = terrain_z(self.mem, px, py)
                if ground is not None:
                    self.eye = float(ground)
                else:
                    self.eye = state.obj_z_height[p] + state.obj_z_frac[p] / 256.0
                    self.col[(px, py)] = self.eye
        else:
            tz = terrain_z(self.mem, px, py)
            self.eye = float(tz if tz is not None else state.obj_z_height[p])
        self.sentinel_slot = state.slot_of_type(mm.T_SENTINEL)
        self.plat = tuple(state.platform_xy)
        pg = terrain_z(self.mem, *self.plat)
        if pg is None:  # platform tile is object-occupied
            pslot = state.slot_of_type(mm.T_PLATFORM)
            if pslot is not None:
                pg = state.obj_z_height[pslot]
        self.plat_ground = pg

    # --- derived attributes -------------------------------------------------
    @property
    def player(self):
        return self.state.player

    @property
    def energy(self):
        return self.state.energy

    @energy.setter
    def energy(self, value):
        self.state.energy = value

    @property
    def free(self):
        """Empty slots, ascending (read-only; the model derives it from state)."""
        return self.state.free_slots()

    # --- queries ------------------------------------------------------------
    def player_xy(self):
        return tuple(self.state.player_xy())

    def top_of(self, tile):
        if tile in self.col:
            return self.col[tile]
        z = terrain_z(self.mem, *tile)
        return float(z) if z is not None else None

    def feasible(self, otype, tile):
        """Keyboard-feasibility of a create on ``tile`` (own-tile / energy /
        free-slot / stackability / build-height-limit); the LOS gate is supplied
        separately by the sweep."""
        if tuple(tile) == self.player_xy():
            return False  # cannot build on your own tile ($1F38)
        if self.energy < mm.ENERGY_IN_OBJECTS[otype] or not self.free:
            return False
        tb = self.mem[mm.TILES_TABLE + mm.tidx(tile[0], tile[1])]
        if tb >= _OBJECT_TILE:  # occupied tile: create only on boulder/plat
            below = tb & 0x3F
            if self.mem[mm.OBJECTS_TYPE + below] not in (mm.T_BOULDER, mm.T_PLATFORM):
                return False  # $1F38 leave_with_carry_set
            top = self.top_of(tile)
            if top is not None and top > self.eye + ROBOT_EYE_FUDGE:
                return False  # column top above sightline
        return True

    # --- actions ------------------------------------------------------------
    def create(self, otype, tile, view, note=""):
        """Build ``otype`` on ``tile`` (delegates to actions.create, so the
        objects_flags chain + tile byte + energy match the ROM).  Returns the new
        slot, or None if ROM-infeasible OR the sights ``view`` has no line of sight
        to ``tile`` at the true eye (``aim.gate`` -- the ROM action LOS gate $1B46,
        the same check the sim runner and driver apply)."""
        if view is not None and not aim.gate(self.state, view, tuple(tile)):
            return None  # no real-eye LOS on the resolved aim -- the ROM would abort
        slot = actions.create(self.state, otype, tuple(tile))
        if slot is None:
            return None
        self.col[tuple(tile)] = (
            self.state.obj_z_height[slot] + self.state.obj_z_frac[slot] / 256.0
        )
        self.steps.append(
            {
                "verb": "create",
                "otype": otype,
                "target": list(tile),
                "view": view,
                "player_tile": list(self.player_xy()),
                "eye_z": round(self.eye, 3),
                "note": note,
            }
        )
        return slot

    def transfer(self, slot, note=""):
        actions.transfer(self.state, slot)
        tile = (self.state.obj_x[slot], self.state.obj_y[slot])
        self.eye = self.top_of(tile)
        self.steps.append(
            {
                "verb": "transfer",
                "otype": None,
                "target": list(tile),
                "view": None,
                "player_tile": list(tile),
                "eye_z": round(self.eye, 3),
                "note": note,
            }
        )

    def absorb(self, slot, view, note=""):
        tile = (self.state.obj_x[slot], self.state.obj_y[slot])
        if view is not None and not aim.gate(self.state, view, tile):
            return False  # ROM action LOS gate $1B46: no real-eye LOS -> no absorb
        otype = int(self.state.obj_type[slot])
        flags = self.state.obj_flags[slot]
        actions.absorb(self.state, slot)  # gains energy, removes object, repairs tile
        if 0x40 <= flags <= 0x7F:  # an object sat below -> it becomes the tile top
            below = flags & 0x3F
            self.col[tile] = (
                self.state.obj_z_height[below] + self.state.obj_z_frac[below] / 256.0
            )
        else:
            self.col.pop(tile, None)
        self.steps.append(
            {
                "verb": "absorb",
                "otype": otype,
                "target": list(tile),
                "view": view,
                "player_tile": list(self.player_xy()),
                "eye_z": round(self.eye, 3),
                "note": note,
            }
        )
        return True

    def clone(self):
        """Independent branch copy for lookahead search."""
        g = PlanGame.__new__(PlanGame)
        g.state = self.state.clone()
        g.mem = g.state.mem
        g.landscape = self.landscape
        g.col = dict(self.col)
        g.eye = self.eye
        g.plat = self.plat
        g.plat_ground = self.plat_ground
        g.sentinel_slot = self.sentinel_slot
        g.steps = list(self.steps)
        g.native_won = self.native_won
        return g
