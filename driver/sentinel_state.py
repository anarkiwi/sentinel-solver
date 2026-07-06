#!/usr/bin/env python3
"""Read the live game state of The Sentinel (C64) out of memory.

The game keeps all its state in well-known memory locations. This module
abstracts *where* memory comes from behind a ``MemorySource`` (callable
``read(addr, length) -> bytes``) and turns a snapshot into a structured
``GameState``:

  * a 32x32 de-interleaved terrain height/slope grid (a VERTEX height field),
  * the object list (Sentinel, sentries, trees, the player robot, ...), and
  * the scalar play variables (player slot/energy, max enemies, vertical scale).

Two concrete sources are provided:
  * ``Py65Source``  — wraps a flat 64 KB memory image; ``from_landscape`` builds
                      one from scratch via the standalone :mod:`sentinel`
                      simulator (no emulator), so a driver can sanity-check the
                      live board against the generator.
  * ``ViceSource``  — reads from a live asid-vice binary monitor (``bm.mem_get``).

``mem_image(source)`` returns a full 64 KB image (for wrapping in a
:class:`sentinel.state.State` / ``plan_game.PlanGame.from_mem``), and
``read_game_state`` returns the structured :class:`GameState` the live driver
consumes.
"""

import sys
import os

_HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, _HERE)
sys.path.insert(0, os.path.dirname(_HERE))  # repo root, so `import sentinel` resolves

from sentinel import memmap as mm  # noqa: E402
from sentinel import landscape as _landscape  # noqa: E402

N = mm.N  # the board is 32x32 tiles

# ---- memory map (identical addresses to the ROM / sentinel.memmap) ----------
PLAYER_OBJECT = mm.PLAYER_OBJECT  # 0x000B
OBJECTS_FLAGS = mm.OBJECTS_FLAGS  # 0x0100
OBJECTS_V_ANGLE = mm.OBJECTS_V_ANGLE  # 0x0140
TILES_TABLE = mm.TILES_TABLE  # 0x0400
OBJECTS_X = mm.OBJECTS_X  # 0x0900
OBJECTS_Z_HEIGHT = mm.OBJECTS_Z_HEIGHT  # 0x0940
OBJECTS_Y = mm.OBJECTS_Y  # 0x0980
OBJECTS_H_ANGLE = mm.OBJECTS_H_ANGLE  # 0x09C0
OBJECTS_Z_FRACTION = mm.OBJECTS_Z_FRACTION  # 0x0A00
OBJECTS_TYPE = mm.OBJECTS_TYPE  # 0x0A40
MAX_ENEMIES = 0x0C07
VERTICAL_SCALE = 0x0C08
PLAYER_ENERGY = mm.PLAYER_ENERGY  # 0x0C0A

NUM_SLOTS = mm.NUM_SLOTS  # 64

TYPES = {
    0: "ROBOT",
    1: "SENTRY",
    2: "TREE",
    3: "BOULDER",
    4: "MEANIE",
    5: "SENTINEL",
    6: "PLATFORM",
}

# The live game keeps all its play state in the first 4 KB; a 64 KB image with
# that page range populated is enough to wrap in a sentinel State / PlanGame.
_LIVE_SNAPSHOT_END = 0x0FFF


# ---- memory sources -------------------------------------------------------
class MemorySource:
    """Abstract source of C64 memory: ``read(addr, length) -> bytes``."""

    def read(self, addr: int, length: int) -> bytes:
        raise NotImplementedError

    def byte(self, addr: int) -> int:
        return self.read(addr, 1)[0]


class Py65Source(MemorySource):
    """Wrap a flat 64 KB memory image (bytes/bytearray)."""

    def __init__(self, mem):
        self.mem = mem

    def read(self, addr: int, length: int) -> bytes:
        return bytes(self.mem[addr : addr + length])

    @classmethod
    def from_landscape(cls, landscape: int):
        """Generate a landscape from scratch with the standalone simulator and
        wrap its 64 KB memory image (no emulator)."""
        state = _landscape.generate(landscape)
        state.mem[mm.CURSOR] = 7
        state.mem[mm.COOLDOWN_GATE] = 0
        return cls(bytes(state.mem))


class ViceSource(MemorySource):
    """Read from a live asid-vice binary monitor via ``bm.mem_get(start, end)``.

    ``bm`` is a connected vice_driver.BinMon. ``mem_get``'s ``end`` is INCLUSIVE
    (see vice_driver/binmon.py), so a read of ``length`` bytes spans
    ``[addr, addr + length - 1]``. The CPU is left stopped by the monitor; call
    ``bm.exit()`` yourself to resume.
    """

    def __init__(self, bm):
        self.bm = bm

    def read(self, addr: int, length: int) -> bytes:
        if length <= 0:
            return b""
        return bytes(self.bm.mem_get(addr, addr + length - 1))


def mem_image(source: MemorySource, end: int = _LIVE_SNAPSHOT_END) -> bytearray:
    """A 64 KB memory image with ``[0, end]`` filled from ``source`` — ready to
    wrap in ``sentinel.state.State.from_mem`` / ``plan_game.PlanGame.from_mem``.

    The live game state lives in the first 4 KB (objects, tiles, cooldowns and
    the scalar play variables), which is all a planner resync needs."""
    if hasattr(source, "mem") and len(source.mem) >= 0x10000:
        return bytearray(source.mem)
    buf = bytearray(0x10000)
    buf[0 : end + 1] = source.read(0x0000, end + 1)
    return buf


# ---- parsed state ---------------------------------------------------------
class GameObject:
    """One object-array entry (a Sentinel, sentry, tree, boulder, the player,
    ...), decoded from the object arrays."""

    __slots__ = (
        "slot",
        "type",
        "type_name",
        "x",
        "y",
        "z",
        "z_fraction",
        "h_angle",
        "v_angle",
        "flags",
        "on_ground",
        "stacked_on",
        "is_player",
    )

    def __init__(self, **kw):
        for k in self.__slots__:
            setattr(self, k, kw[k])


class GameState:
    """The decoded live state: terrain grid, object list and scalars."""

    __slots__ = (
        "height",
        "slope",
        "objects",
        "player_slot",
        "player_energy",
        "max_enemies",
        "vertical_scale",
    )

    def __init__(
        self,
        player_slot=0,
        player_energy=0,
        max_enemies=0,
        vertical_scale=0,
    ):
        self.height = []
        self.slope = []
        self.objects = []
        self.player_slot = player_slot
        self.player_energy = player_energy
        self.max_enemies = max_enemies
        self.vertical_scale = vertical_scale

    def object_by_slot(self, slot):
        for o in self.objects:
            if o.slot == slot:
                return o
        return None

    def objects_at(self, x, y):
        """Every decoded object standing on tile (x, y)."""
        return [o for o in self.objects if o.x == x and o.y == y]

    @property
    def player(self):
        return self.object_by_slot(self.player_slot)


# ---- de-interleaving / ground-height resolution ---------------------------
def tidx(x: int, y: int) -> int:
    """Index into tiles_table for tile (x, y) (interleaved ROM layout)."""
    return mm.tidx(x, y)


def _resolve_ground(tiles, flags, objz, x, y):
    """Ground height + slope at (x, y). A tile byte >= $C0 holds an object index
    (low 6 bits); its terrain height is the bottommost (on-ground) object's z."""
    t = tiles[tidx(x, y)]
    if t < mm.OBJECT_TILE:
        return t >> 4, t & 0x0F
    o = t & 0x3F
    for _ in range(NUM_SLOTS):  # walk the stack down to the ground object
        if flags[o] < 0x40:
            break
        o = flags[o] & 0x3F
    return objz[o], 0  # object tiles are flat


# ---- the reader -----------------------------------------------------------
def read_game_state(source: MemorySource) -> GameState:
    """Extract the full game state from ``source`` into a ``GameState``."""
    tiles = source.read(TILES_TABLE, N * N)  # $0400-$07FF
    flags = source.read(OBJECTS_FLAGS, NUM_SLOTS)
    v_ang = source.read(OBJECTS_V_ANGLE, NUM_SLOTS)
    objx = source.read(OBJECTS_X, NUM_SLOTS)
    objz = source.read(OBJECTS_Z_HEIGHT, NUM_SLOTS)
    objy = source.read(OBJECTS_Y, NUM_SLOTS)
    h_ang = source.read(OBJECTS_H_ANGLE, NUM_SLOTS)
    zfrac = source.read(OBJECTS_Z_FRACTION, NUM_SLOTS)
    objt = source.read(OBJECTS_TYPE, NUM_SLOTS)

    player_slot = source.byte(PLAYER_OBJECT)

    state = GameState(
        player_slot=player_slot,
        player_energy=source.byte(PLAYER_ENERGY),
        max_enemies=source.byte(MAX_ENEMIES),
        vertical_scale=source.byte(VERTICAL_SCALE),
    )

    state.height = [[0] * N for _ in range(N)]
    state.slope = [[0] * N for _ in range(N)]
    for y in range(N):
        for x in range(N):
            h, s = _resolve_ground(tiles, flags, objz, x, y)
            state.height[y][x] = h
            state.slope[y][x] = s

    for i in range(NUM_SLOTS):
        f = flags[i]
        if f & 0x80:  # empty slot
            continue
        on_ground = f < 0x40
        t = objt[i]
        state.objects.append(
            GameObject(
                slot=i,
                type=t,
                type_name=TYPES.get(t, f"?{t}"),
                x=objx[i],
                y=objy[i],
                z=objz[i],
                z_fraction=zfrac[i],
                h_angle=h_ang[i],
                v_angle=v_ang[i],
                flags=f,
                on_ground=on_ground,
                stacked_on=None if on_ground else (f & 0x3F),
                is_player=(i == player_slot and t == 0),
            )
        )

    return state


def verify_entry(bm, landscape, log=print):
    """Confirm the live board matches the standalone generator for ``landscape``
    (:meth:`Py65Source.from_landscape`). Returns ``(matched, total)`` object
    position/type counts, or ``None`` if the reference could not be generated."""
    try:
        ref = read_game_state(Py65Source.from_landscape(landscape))
    except Exception as e:  # generator unavailable (no simulator import, bad seed)
        log(f"  (entry ref unavailable: {e})")
        return None
    live = read_game_state(ViceSource(bm))
    ref_objs = sorted((o.x, o.y, o.type) for o in ref.objects)
    live_objs = sorted((o.x, o.y, o.type) for o in live.objects)
    matched = sum(1 for o in ref_objs if o in live_objs)
    log(
        f"ENTRY MATCH: {matched}/{len(ref_objs)} objects vs from_landscape({landscape}) "
        f"(live has {len(live_objs)})"
    )
    return matched, len(ref_objs)


def dump(state: GameState) -> str:
    """A compact human-readable summary of a decoded ``GameState``."""
    out = ["== scalars =="]
    out.append(f"  player_slot     : {state.player_slot}")
    out.append(f"  player_energy   : {state.player_energy}")
    out.append(f"  max_enemies     : {state.max_enemies}")
    out.append(f"  vertical_scale  : {state.vertical_scale}")
    counts = {}
    for o in state.objects:
        key = "PLAYER" if o.is_player else o.type_name
        counts[key] = counts.get(key, 0) + 1
    out.append(f"== objects ({len(state.objects)}) ==")
    out.append("  " + "  ".join(f"{k.lower()}:{v}" for k, v in sorted(counts.items())))
    return "\n".join(out)


def main():
    for landscape in (0, 42, 9999):
        state = read_game_state(Py65Source.from_landscape(landscape))
        print(f"\n############ seed {landscape} ############")
        print(dump(state))


if __name__ == "__main__":
    main()
