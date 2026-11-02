"""Per-row scalar kernels of the search engine: apply an action sequence to a row's image,
read its decision fields, hash its mutable region.  Each is one spec, so the CPU ``prange``
loops here and the device kernels of :mod:`sentinel.kern_cuda` run the same source."""

import numpy as np

from numba import njit, prange

from sentinel import memmap as mm
from sentinel.actions_jit import absorb, create, transfer
from sentinel.kern import spec

DEVICE_TABLES = True  # kern._cuda_module binds this module's tables as device arrays
CREATE, ABSORB, TRANSFER, NOOP = 0, 1, 2, -1
LAST = -1
FIELDS = ("energy", "player", "eye", "alive", "won", "enemies", "x", "y")
N_FIELDS = len(FIELDS)
_FNV = 16777619
_BASIS = 2166136261
_MIX = 0x2C1B3C6D
_SALT = 0x85EBCA6B
_M32 = 0xFFFFFFFF
_CHECK = mm.PLAYER_DIED_BY_DRAINING
_FLAGS = mm.PLAYER_HAS_HYPERSPACED


@spec
def apply_ops(mem, ops):
    """Run the ``(K, 3)`` ``(op, a, b)`` sequence on ``mem`` and return 1 when every op
    took effect, else 0 at the first that did not.  ``CREATE`` builds type ``a`` on the
    packed tile ``b`` (``x * 32 + y``); ``ABSORB`` and ``TRANSFER`` act on slot ``a``, or
    on the slot the last ``CREATE`` made when ``a`` is ``LAST``; ``NOOP`` rows pad a ragged
    batch."""
    last = LAST
    for k in range(ops.shape[0]):
        op = ops[k, 0]
        slot = ops[k, 1] if ops[k, 1] != LAST else last
        if op == CREATE:
            last = create(mem, ops[k, 1], ops[k, 2] >> 5, ops[k, 2] & 31)
            if last < 0:
                return 0
        elif op == ABSORB:
            if slot < 0 or absorb(mem, slot) == 0:
                return 0
        elif op == TRANSFER:
            if slot < 0 or transfer(mem, slot) == 0:
                return 0
    return 1


@spec
def fields(mem, out):
    """Write the :data:`FIELDS` of an image into ``out``: purse, player slot, eye height in
    1/256ths, alive, won, live enemy count and the player's tile."""
    player = np.int64(mem[mm.PLAYER_OBJECT])
    energy = np.int64(mem[mm.PLAYER_ENERGY])
    flags = np.int64(mem[_FLAGS]) & 0xC0
    dead = (mem[_CHECK] & 0x80) != 0 or flags == 0x80
    n = 0
    for slot in range(mm.NUM_SLOTS):
        if mem[mm.OBJECTS_FLAGS + slot] & 0x80 == 0:
            otype = mem[mm.OBJECTS_TYPE + slot]
            if otype == mm.T_SENTRY or otype == mm.T_SENTINEL:
                n += 1
    out[0] = energy
    out[1] = player
    out[2] = (np.int64(mem[mm.OBJECTS_Z_HEIGHT + player]) << 8) | np.int64(
        mem[mm.OBJECTS_Z_FRACTION + player]
    )
    out[3] = 1 if (not dead and energy > 0) else 0
    out[4] = 1 if flags == 0xC0 else 0
    out[5] = n
    out[6] = mem[mm.OBJECTS_X + player]
    out[7] = mem[mm.OBJECTS_Y + player]


@spec
def row_hash(mut, clock):
    """A 64-bit digest of a row: two FNV-style 32-bit lanes over the mutable bytes and the
    clock fields."""
    h1 = _BASIS
    h2 = _SALT
    for j in range(mut.shape[0]):
        b = np.int64(mut[j])
        h1 = ((h1 ^ b) * _FNV) & _M32
        h2 = ((h2 + b + 1) * _MIX) & _M32
    for j in range(clock.shape[0]):
        v = np.int64(clock[j])
        for s in range(8):
            b = (v >> (8 * s)) & 0xFF
            h1 = ((h1 ^ b) * _FNV) & _M32
            h2 = ((h2 + b + 1) * _MIX) & _M32
    return (h1 << 32) | h2


@njit(cache=True, parallel=True)
def apply_rows(imgs, ops, ok):
    """:func:`apply_ops` on every row of the ``(B, 65536)`` stack ``imgs``."""
    for b in prange(imgs.shape[0]):  # pylint: disable=not-an-iterable
        ok[b] = apply_ops(imgs[b], ops[b])


@njit(cache=True, parallel=True)
def fields_rows(imgs, out):
    """:func:`fields` on every row of ``imgs``."""
    for b in prange(imgs.shape[0]):  # pylint: disable=not-an-iterable
        fields(imgs[b], out[b])


@njit(cache=True, parallel=True)
def hash_rows(mut, clock, out):
    """:func:`row_hash` of every row."""
    for b in prange(mut.shape[0]):  # pylint: disable=not-an-iterable
        out[b] = row_hash(mut[b], clock[b])
