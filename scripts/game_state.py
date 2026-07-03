#!/usr/bin/env python3
"""Read the live game state of The Sentinel (C64) out of memory.

The game keeps all its state in well-known memory locations (the game's memory
layout). This module abstracts *where* memory comes
from behind a `MemorySource` (callable `read(addr, length) -> bytes`) and turns a
snapshot into a structured `GameState`:

  * a 32x32 de-interleaved terrain height/slope grid (a VERTEX height field),
  * the object list (Sentinel, sentries, trees, the player robot, ...), and
  * the scalar play variables (player slot/energy, max enemies, vertical scale).

Two concrete sources are provided:
  * `Py65Source`  — reads from a py65 emulator memory (scripts/_emu.py), so the
                    reader can be tested fast and deterministically WITHOUT VICE.
  * `ViceSource`  — reads from a live asid-vice binary monitor (`bm.mem_get`).

Run `python3 scripts/game_state.py` to generate a few landscapes via the real
C64 code (_emu.py) and print their parsed state.
"""

import sys, os
from dataclasses import dataclass, field
from typing import List, Optional

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

N = 32  # the board is 32x32 tiles

# ---- verified memory map ----------------------------------------------------
PLAYER_OBJECT = 0x000B  # index of the player's object slot
OBJECTS_FLAGS = 0x0100  # bit7 ⇒ empty; <$40 on ground; $40-$7F stacked on (v&$3F)
OBJECTS_V_ANGLE = 0x0140  # vertical tilt
TILES_TABLE = 0x0400  # $0400-$07FF, 32x32, interleaved
OBJECTS_X = 0x0900  # tile x (first horizontal axis)
OBJECTS_Z_HEIGHT = 0x0940  # the VERTICAL/height axis
OBJECTS_Y = 0x0980  # tile y (second horizontal axis)
OBJECTS_H_ANGLE = 0x09C0  # horizontal facing
OBJECTS_Z_FRACTION = 0x0A00  # sub-unit height fraction
OBJECTS_TYPE = 0x0A40  # 0..6 (see TYPES)
MAX_ENEMIES = 0x0C07
VERTICAL_SCALE = 0x0C08
PLAYER_ENERGY = 0x0C0A

NUM_SLOTS = 64  # the object arrays are 64 entries each

TYPES = {
    0: "ROBOT",
    1: "SENTRY",
    2: "TREE",
    3: "BOULDER",
    4: "MEANIE",
    5: "SENTINEL",
    6: "PLATFORM",
}


# ---- memory sources -------------------------------------------------------
class MemorySource:
    """Abstract source of C64 memory: read(addr, length) -> bytes."""

    def read(self, addr: int, length: int) -> bytes:
        raise NotImplementedError

    # convenience helpers used by the reader
    def byte(self, addr: int) -> int:
        return self.read(addr, 1)[0]


class Py65Source(MemorySource):
    """Read from a flat 64K memory image (a py65 emulator's RAM, as produced by
    scripts/_emu.generate). `mem` is anything indexable/sliceable like bytes."""

    def __init__(self, mem):
        self.mem = mem

    def read(self, addr: int, length: int) -> bytes:
        return bytes(self.mem[addr : addr + length])

    @classmethod
    def from_landscape(cls, landscape: int):
        """Generate a landscape by running the real C64 code and wrap its RAM."""
        import _emu

        mem, ins = _emu.generate(landscape)
        src = cls(mem)
        src.instructions = ins
        return src


class ViceSource(MemorySource):
    """Read from a live asid-vice binary monitor via `bm.mem_get(start, end)`.

    `bm` is a connected vice_driver.BinMon. mem_get's `end` is INCLUSIVE (see
    vice_driver/binmon.py), so a read of `length` bytes spans [addr, addr+length-1].
    The CPU is left stopped by the monitor; call `bm.exit()` yourself to resume.
    """

    def __init__(self, bm):
        self.bm = bm

    def read(self, addr: int, length: int) -> bytes:
        if length <= 0:
            return b""
        return bytes(self.bm.mem_get(addr, addr + length - 1))


# ---- parsed state ---------------------------------------------------------
@dataclass
class GameObject:
    slot: int  # 0..63 array index
    type: int  # 0..6
    type_name: str
    x: int  # tile x (first horizontal axis)
    y: int  # tile y (second horizontal axis)
    z: int  # height (whole units), the vertical axis
    z_fraction: int  # sub-unit height fraction
    h_angle: int  # horizontal facing (0..255 ⇒ 0..360°)
    v_angle: int  # vertical tilt
    flags: int  # raw flags byte
    on_ground: bool  # flags < $40
    stacked_on: Optional[int]  # slot it sits on, or None if on the ground
    is_player: bool


@dataclass
class GameState:
    # terrain: 32x32 vertex height field, de-interleaved. height[y][x], slope[y][x].
    height: List[List[int]] = field(default_factory=list)
    slope: List[List[int]] = field(default_factory=list)
    # objects, in slot order (only non-empty slots)
    objects: List[GameObject] = field(default_factory=list)
    player_slot: int = 0
    player_energy: int = 0
    max_enemies: int = 0
    vertical_scale: int = 0

    def object_by_slot(self, slot: int) -> Optional[GameObject]:
        for o in self.objects:
            if o.slot == slot:
                return o
        return None

    @property
    def player(self) -> Optional[GameObject]:
        return self.object_by_slot(self.player_slot)


# ---- de-interleaving / ground-height resolution ---------------------------
def tidx(x: int, y: int) -> int:
    """Index into tiles_table for tile (x,y) — the game's own formula
    (calculate_tile_address $28D4): interleaved, NOT row-major."""
    return (x & 3) * 256 + ((x >> 2) & 7) * 32 + y


def _resolve_ground(tiles, flags, objz, x, y):
    """Ground height + slope at (x,y). A tile byte >=$C0 holds an object index
    (low 6 bits); its terrain height is the bottommost (on-ground) object's z."""
    t = tiles[tidx(x, y)]
    if t < 0xC0:
        return t >> 4, t & 0x0F
    o = t & 0x3F
    for _ in range(NUM_SLOTS):  # walk the stack down to the ground object
        if flags[o] < 0x40:
            break
        o = flags[o] & 0x3F
    return objz[o], 0  # object tiles are flat


# ---- the reader -----------------------------------------------------------
def read_game_state(source: MemorySource) -> GameState:
    """Extract the full game state from `source` into a GameState."""
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

    # terrain grid
    state.height = [[0] * N for _ in range(N)]
    state.slope = [[0] * N for _ in range(N)]
    for y in range(N):
        for x in range(N):
            h, s = _resolve_ground(tiles, flags, objz, x, y)
            state.height[y][x] = h
            state.slope[y][x] = s

    # object list
    for i in range(NUM_SLOTS):
        f = flags[i]
        if f & 0x80:  # empty slot
            continue
        on_ground = f < 0x40
        stacked_on = None if on_ground else (f & 0x3F)
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
                stacked_on=stacked_on,
                is_player=(i == player_slot and t == 0),
            )
        )

    return state


# ---- pretty-printer -------------------------------------------------------
def dump(state: GameState) -> str:
    out = []
    terr = [state.height[y][x] for y in range(N) for x in range(N)]
    hmin, hmax = min(terr), max(terr)
    # highest terrain tile (where the Sentinel's platform usually goes)
    hx = hy = 0
    for y in range(N):
        for x in range(N):
            if state.height[y][x] > state.height[hy][hx]:
                hx, hy = x, y
    n_slope = sum(1 for y in range(N) for x in range(N) if state.slope[y][x])

    out.append("== scalars ==")
    out.append(f"  player_slot     : {state.player_slot}")
    out.append(f"  player_energy   : {state.player_energy}")
    out.append(f"  max_enemies     : {state.max_enemies}")
    out.append(f"  vertical_scale  : {state.vertical_scale}")
    out.append("== terrain (32x32 vertex height field) ==")
    out.append(f"  height range    : {hmin}..{hmax}  (relief {hmax - hmin})")
    out.append(f"  highest tile    : ({hx},{hy}) height {hmax}")
    out.append(f"  sloped tiles    : {n_slope}/{N * N}")

    counts = {}
    for o in state.objects:
        key = "PLAYER" if o.is_player else o.type_name
        counts[key] = counts.get(key, 0) + 1
    out.append(f"== objects ({len(state.objects)}) ==")
    out.append("  " + "  ".join(f"{k.lower()}:{v}" for k, v in sorted(counts.items())))
    out.append(
        f"  {'slot':>4} {'type':>9} {'x':>3} {'y':>3} {'z':>3} {'zf':>4}"
        f" {'hang':>5} {'vang':>5}  placement"
    )
    for o in sorted(state.objects, key=lambda o: o.slot):
        if o.is_player:
            place = "PLAYER"
        elif o.on_ground:
            place = "ground"
        else:
            place = f"on #{o.stacked_on}"
        out.append(
            f"  {o.slot:>4} {o.type_name:>9} {o.x:>3} {o.y:>3} {o.z:>3}"
            f" {o.z_fraction:>4} {o.h_angle:>5} {o.v_angle:>5}  {place}"
        )
    return "\n".join(out)


# ---- self-test / demo -----------------------------------------------------
def _validate(landscape, state):
    """Light sanity checks; returns a list of warning strings (empty == good)."""
    warns = []
    types_seen = {o.type for o in state.objects}
    if 5 not in types_seen:
        warns.append("no SENTINEL present")
    if 6 not in types_seen:
        warns.append("no PLATFORM present")
    # the Sentinel should sit (stacked) on the highest terrain tile
    sentinels = [o for o in state.objects if o.type == 5]
    if sentinels:
        s = sentinels[0]
        hmax = max(state.height[y][x] for y in range(N) for x in range(N))
        if state.height[s.y][s.x] < hmax:
            warns.append(
                f"Sentinel tile ({s.x},{s.y}) height "
                f"{state.height[s.y][s.x]} is not the highest ({hmax})"
            )
        if s.on_ground:
            warns.append("Sentinel is on the ground, not on a platform")
    if (
        state.player_slot >= NUM_SLOTS
        or state.object_by_slot(state.player_slot) is None
    ):
        warns.append(f"player_slot {state.player_slot} has no object")
    return warns


def main():
    for landscape in (0, 42, 9999):
        src = Py65Source.from_landscape(landscape)
        state = read_game_state(src)
        print(
            f"\n############ seed {landscape}  "
            f"({getattr(src, 'instructions', 0):,} emulated instructions) ############"
        )
        print(dump(state))
        warns = _validate(landscape, state)
        if warns:
            print("  !! warnings: " + "; ".join(warns))
        else:
            print("  ok: placement looks sane")


if __name__ == "__main__":
    main()
