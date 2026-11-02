"""The player actions of :mod:`sentinel.actions` as scalar kernels over one board image:
``create``, ``absorb``, ``transfer`` and ``face``, byte-identical to the reference and compiled
for the CPU and, from the same source, for CUDA (:mod:`sentinel.kern`)."""

import numpy as np

from sentinel import memmap as mm
from sentinel.enemies_jit import (
    _prng_next,
    _remove_object,
    _set_tile_byte,
    _tile_byte,
)
from sentinel.kern import spec

_OFLAGS = mm.OBJECTS_FLAGS
_OTYPE = mm.OBJECTS_TYPE
_OX = mm.OBJECTS_X
_OY = mm.OBJECTS_Y
_OZH = mm.OBJECTS_Z_HEIGHT
_OZF = mm.OBJECTS_Z_FRACTION
_OHANG = mm.OBJECTS_H_ANGLE
_OVANGLE = mm.OBJECTS_V_ANGLE
_ENERGY = mm.PLAYER_ENERGY
_PLAYER = mm.PLAYER_OBJECT
_OBJECT_TILE = mm.OBJECT_TILE
_T_ROBOT = mm.T_ROBOT
_T_BOULDER = mm.T_BOULDER
_T_PLATFORM = mm.T_PLATFORM
VALUE = np.zeros(8, dtype=np.int64)
for _t, _v in mm.ENERGY_IN_OBJECTS.items():
    VALUE[_t] = _v
NO_SLOT = -1


@spec
def _acted(mem):
    """$12E1: the first action unfreezes the enemy world clock."""
    mem[mm.PLAYER_NOT_ACTED] = np.uint8(0)


@spec
def create(mem, otype, x, y):
    """:func:`sentinel.actions.create` of ``otype`` on tile ``(x, y)``: the new slot, or
    ``NO_SLOT`` when no slot is free, the purse is short or the tile cannot be stacked on.
    """
    _acted(mem)
    slot = NO_SLOT
    for s in range(63, -1, -1):
        if mem[_OFLAGS + s] & 0x80:
            slot = s
            break
    if slot == NO_SLOT:
        return NO_SLOT
    cost = VALUE[otype]
    energy = np.int64(mem[_ENERGY])
    if energy < cost:
        return NO_SLOT
    mem[_ENERGY] = np.uint8((energy - cost) & mm.ENERGY_MASK)
    mem[_OTYPE + slot] = np.uint8(otype)
    b = np.int64(_tile_byte(mem, x, y))
    zf = 0xE0
    z = b >> 4
    flags = 0
    if b >= _OBJECT_TILE:
        below = b & 0x3F
        btype = np.int64(mem[_OTYPE + below])
        if btype == _T_PLATFORM:
            zf = np.int64(mem[_OZF + below])
            z = np.int64(mem[_OZH + below]) + 1
        elif btype == _T_BOULDER:
            t = np.int64(mem[_OZF + below]) + 0x80
            zf = t & 0xFF
            z = np.int64(mem[_OZH + below]) + (t >> 8)
        else:
            mem[_ENERGY] = np.uint8((np.int64(mem[_ENERGY]) + cost) & mm.ENERGY_MASK)
            return NO_SLOT
        flags = 0x40 | below
    mem[_OFLAGS + slot] = np.uint8(flags)
    mem[_OX + slot] = np.uint8(x)
    mem[_OY + slot] = np.uint8(y)
    mem[_OZH + slot] = np.uint8(z & 0xFF)
    mem[_OZF + slot] = np.uint8(zf & 0xFF)
    _set_tile_byte(mem, x, y, _OBJECT_TILE | slot)
    mem[_OVANGLE + slot] = np.uint8(0xF5)
    rot = _prng_next(mem)
    mem[_OHANG + slot] = np.uint8(((rot & 0xF8) + 0x60) & 0xFF)
    if otype == _T_ROBOT:
        mem[_OHANG + slot] = np.uint8(mem[_OHANG + np.int64(mem[_PLAYER])] ^ 0x80)
    return slot


@spec
def face(mem, h, v):
    """Turn the player's body to bearing ``h`` and pitch ``v``: the facing an aim leaves."""
    player = np.int64(mem[_PLAYER])
    mem[_OHANG + player] = np.uint8(h)
    mem[_OVANGLE + player] = np.uint8(v)


@spec
def absorb(mem, slot):
    """:func:`sentinel.actions.absorb`: 1 when ``slot`` was absorbed, else 0."""
    _acted(mem)
    if mem[_OFLAGS] & 0x80 or mem[_OFLAGS + slot] & 0x80:
        return 0
    otype = np.int64(mem[_OTYPE + slot])
    if otype == _T_PLATFORM:
        return 0
    _remove_object(mem, slot)
    mem[_ENERGY] = np.uint8((np.int64(mem[_ENERGY]) + VALUE[otype]) & mm.ENERGY_MASK)
    return 1


@spec
def transfer(mem, slot):
    """:func:`sentinel.actions.transfer`: 1 when the player moved into robot ``slot``."""
    _acted(mem)
    if mem[_OTYPE + slot] != _T_ROBOT:
        return 0
    mem[_PLAYER] = np.uint8(slot)
    return 1
