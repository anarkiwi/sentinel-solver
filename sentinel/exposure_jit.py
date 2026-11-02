"""Which enemies have any sight of one body: the player-side exposure scan as a spec."""

import numpy as np
from numba import njit

from sentinel import memmap as mm
from sentinel.enemies_jit import ZP_HI, _can_see_object, _relative_angles
from sentinel.kern import spec

FOV_FULL = 0x100
_OFLAGS = mm.OBJECTS_FLAGS
_OTYPE = mm.OBJECTS_TYPE
_SENTRY = mm.T_SENTRY
_SENTINEL = mm.T_SENTINEL
_ROBOT = mm.T_ROBOT


@spec
def exposures(mem, zp, slot, out):
    """Rows ``(enemy slot, angle_hi, full)`` of ``out`` for every enemy, in slot order,
    with any sight of a robot body in ``slot`` at any facing; returns their count.  The
    marches write the LOS tracker bytes of ``mem`` as the ROM's own scan does."""
    n = 0
    for e in range(mm.NUM_SLOTS):
        kind = mem[_OTYPE + e]
        if mem[_OFLAGS + e] & 0x80 or (kind != _SENTRY and kind != _SENTINEL):
            continue
        seen = _can_see_object(mem, zp, e, slot, _ROBOT, FOV_FULL)
        if seen[2] != 0:
            angles = _relative_angles(mem, zp, e, slot)
            out[n, 0] = e
            out[n, 1] = angles[2]
            out[n, 2] = seen[3]
            n += 1
    return n


@njit(cache=True)
def exposures_of(mem, slot):
    """:func:`exposures` on fresh scratch -> int64 ``(n, 3)`` rows."""
    zp = np.zeros(ZP_HI, dtype=np.int64)
    out = np.zeros((mm.NUM_SLOTS, 3), dtype=np.int64)
    return out[: exposures(mem, zp, slot, out)]
