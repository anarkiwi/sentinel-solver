"""Numba twin of the enemy frame clock -- :func:`sentinel.enemies.advance_frames`.

:mod:`sentinel.enemies` remains the bit-exact reference and the numba-absent fallback,
mirroring :mod:`sentinel.los`/:mod:`sentinel.los_jit`; ``tests/test_enemies_jit.py``
asserts byte-identical 64 KB images between the two over hundreds of frames.
"""

import numpy as np

from sentinel import jitcache

jitcache.install()  # must precede numba: the cache key carries the cost constants

from numba import njit  # noqa: E402  pylint: disable=wrong-import-position

from sentinel import badline, memmap as mm, passcost  # noqa: E402
from sentinel.badline_jit import charge, frame_clock  # noqa: E402
from sentinel.relative import _ARCTAN_LO, _ARCTAN_HI, _HYP
from sentinel.los_jit import (  # noqa: E402
    march,
    LOS_CLEAR,
    _vsin_cos,
    _vproc_sc,
    _vmul_dbl_dbl,
    _vmul_dbl_by_byte,
    _vinvert16,
)

# Coefficient tables, shared with the reference so they cannot drift.
ARCTAN_LO = np.array(_ARCTAN_LO, dtype=np.int64)
ARCTAN_HI = np.array(_ARCTAN_HI, dtype=np.int64)
HYP = np.array(_HYP, dtype=np.int64)
HALF_ANGLE = np.array(
    [mm.OBJECT_SCREEN_HALF_ANGLE.get(t, 0) for t in range(8)], dtype=np.int64
)

# Addresses and tuning constants, inlined as njit-visible globals.
_OFLAGS = mm.OBJECTS_FLAGS
_OVANGLE = mm.OBJECTS_V_ANGLE
_OX = mm.OBJECTS_X
_OZH = mm.OBJECTS_Z_HEIGHT
_OY = mm.OBJECTS_Y
_OHANG = mm.OBJECTS_H_ANGLE
_OZF = mm.OBJECTS_Z_FRACTION
_OTYPE = mm.OBJECTS_TYPE

_PLAYER = mm.PLAYER_OBJECT
_SIZE_FLOOR = mm.OBJECT_SIZE_FLOOR
_ENERGY = mm.PLAYER_ENERGY
_CURSOR = mm.CURSOR
_FOV_WIDTH = mm.FOV_WIDTH
_PLATFORM_X = mm.PLATFORM_X
_PLATFORM_Y = mm.PLATFORM_Y
_DIED_DRAINING = mm.PLAYER_DIED_BY_DRAINING
_HYPERSPACED = mm.PLAYER_HAS_HYPERSPACED
_COMPLETE = mm.LANDSCAPE_COMPLETE
_BELOW_Z = mm.ENEMY_BELOW_Z
_NOT_ACTED = mm.PLAYER_NOT_ACTED
_BRESENHAM = mm.COOLDOWN_BRESENHAM
_BRESENHAM_STEP = mm.COOLDOWN_BRESENHAM_STEP
_GATE = mm.COOLDOWN_GATE
_PRND = mm.PRND_STATE
_SND_NOTE = mm.SOUND_NOTE_TIMER
_SND_LENGTH = mm.SOUND_NOTE_LENGTH
_SND_GATE = mm.SOUND_GATE_TIMER
_SND_FLAG = mm.SOUND_VOICE_FLAG
_SND_TABLE = mm.SOUND_TABLE
_SND_LENGTHS = mm.SOUND_LENGTHS
_SND_VOICE_ON = mm.SOUND_VOICE_ON
_SND_IDLE = mm.SOUND_VOICE_IDLE
_SND_ROTATE = mm.SOUND_ROTATE
_SND_MEANIE = mm.SOUND_MEANIE
_SND_DRAIN = mm.SOUND_DRAIN

_DRAIN_CD = mm.ENEMIES_DRAINING_COOLDOWN
_ROT_CD = mm.ENEMIES_ROTATION_COOLDOWN
_UPD_CD = mm.ENEMIES_UPDATE_COOLDOWN
_M_SEARCH = mm.ENEMIES_MEANIE_SEARCH_OBJECT
_DISCHARGE = mm.ENEMIES_ENERGY_TO_DISCHARGE
_M_FAILED = mm.ENEMIES_FAILED_MEANIE_MEMORY
_M_SCANS = mm.ENEMIES_MEANIE_ATTEMPT_SCANS
_M_OBJECT = mm.ENEMIES_MEANIE_OBJECT
_TARGET = mm.ENEMIES_TARGETED_OBJECT
_TARGET_EXP = mm.ENEMIES_TARGETED_OBJECT_EXPOSURE
_CONSIDERING = mm.ENEMIES_CONSIDERING_MEANIE
_ROT_SPEED = mm.ROTATION_SPEED_TABLE
_TARGETED_SLOT = mm.TARGETED_OBJECT_SLOT

_T_ROBOT = mm.T_ROBOT
_T_TREE = mm.T_TREE
_T_BOULDER = mm.T_BOULDER
_T_MEANIE = mm.T_MEANIE
_T_SENTRY = mm.T_SENTRY
_T_SENTINEL = mm.T_SENTINEL
_OBJECT_TILE = mm.OBJECT_TILE
_NUM_SLOTS = mm.NUM_SLOTS
_ENERGY_MASK = mm.ENERGY_MASK
_ROBOT_ENERGY = mm.ENERGY_IN_OBJECTS[mm.T_ROBOT]

_FOV_SCAN = 0x14
_FOV_CREATE_MEANIE = 0x28
_UPD_CD_SCAN = 0x04
_UPD_CD_DRAIN = 0x1E
_UPD_CD_MEANIE_ROTATE = 0x0A
_UPD_CD_MEANIE_MADE = 0x32
_ROT_CD_RELOAD = 0xC8
_DRAIN_CD_RELOAD = 0x78
_COOLDOWN_STICK = 0x02
_MEANIE_ROTATE_STEP = 0x08
_MEANIE_MAX_ATTEMPTS = 0x02

# Cycle costs, shared with the reference so the two clocks cannot drift.
_FOREGROUND_CYCLES = passcost.FOREGROUND_CYCLES
_IRQ_BODY = passcost.IRQ_BODY
_STEAL_CEILING = badline.FRAME_STEAL_CEILING

_EXPOSURE_FIXED = passcost.EXPOSURE_FIXED
_EXPOSURE_EMPTY = passcost.EXPOSURE_EMPTY
_EXPOSURE_OTHER = passcost.EXPOSURE_OTHER
_EXPOSURE_SENTRY = passcost.EXPOSURE_SENTRY
_EXPOSURE_SENTINEL = passcost.EXPOSURE_SENTINEL
_EXPOSURE_TARGETS_PLAYER = passcost.EXPOSURE_TARGETS_PLAYER
_EXPOSURE_DRAINING = passcost.EXPOSURE_DRAINING
_EXPOSURE_LAST = passcost.EXPOSURE_LAST

_UPDATE_NOT_ENEMY = passcost.UPDATE_NOT_ENEMY
_UPDATE_DISPATCH_SENTRY = passcost.UPDATE_DISPATCH_SENTRY
_UPDATE_DISPATCH_SENTINEL = passcost.UPDATE_DISPATCH_SENTINEL
_UPDATE_GATE_CLOSED = passcost.UPDATE_GATE_CLOSED
_UPDATE_ABSORBED = passcost.UPDATE_ABSORBED
_UPDATE_PRND = passcost.UPDATE_PRND
_UPDATE_CURSOR = passcost.UPDATE_CURSOR
_UPDATE_CURSOR_WRAP = passcost.UPDATE_CURSOR_WRAP
_PASS_HEAD = passcost.PASS_HEAD
_PASS_TAIL = passcost.PASS_TAIL
_CONSIDER_ENTRY = passcost.CONSIDER_ENTRY

_DISCHARGE_NONE = passcost.DISCHARGE_NONE
_DISCHARGE_CREATE = passcost.DISCHARGE_CREATE
_DISCHARGE_PLACE = passcost.DISCHARGE_PLACE
_DISCHARGE_ABANDON = passcost.DISCHARGE_ABANDON
_DISCHARGE_DONE = passcost.DISCHARGE_DONE
_CREATE_HEAD = passcost.CREATE_HEAD
_CREATE_SLOT = passcost.CREATE_SLOT
_CREATE_LAST = passcost.CREATE_LAST
_CREATE_HIT = passcost.CREATE_HIT
_CREATE_NONE = passcost.CREATE_NONE
_PLACE_HEAD = passcost.PLACE_HEAD
_PLACE_LAP = passcost.PLACE_LAP
_PLACE_WRAP = passcost.PLACE_WRAP
_PLACE_GIVE_UP = passcost.PLACE_GIVE_UP
_PLACE_OCCUPIED = passcost.PLACE_OCCUPIED
_PLACE_NOT_FLAT = passcost.PLACE_NOT_FLAT
_PLACE_TOO_HIGH = passcost.PLACE_TOO_HIGH
_PLACE_HIT = passcost.PLACE_HIT
_PUT_IN_TILE = passcost.PUT_IN_TILE
_DRAW = passcost.DRAW
_DRAW_REJECT = passcost.DRAW_REJECT

_SEE_SLOT_EMPTY = passcost.SEE_SLOT_EMPTY
_SEE_SLOT_WRONG_TYPE = passcost.SEE_SLOT_WRONG_TYPE
_SEE_PROLOGUE = passcost.SEE_PROLOGUE
_SEE_FOV = passcost.SEE_FOV
_SEE_FOV_REJECT = passcost.SEE_FOV_REJECT
_SEE_FOV_PASS = passcost.SEE_FOV_PASS
_SEE_ROBOT = passcost.SEE_ROBOT
_SEE_NOT_ROBOT = passcost.SEE_NOT_ROBOT
_SEE_PROBE_FIXED = passcost.SEE_PROBE_FIXED
_SEE_BASE = passcost.SEE_BASE
_SEE_BASE_AGAIN = passcost.SEE_BASE_AGAIN
_SEE_BASE_LAST = passcost.SEE_BASE_LAST
_SEE_TAIL = passcost.SEE_TAIL
_MARCH_ENTRY = passcost.MARCH_ENTRY
_PREP_VEC = passcost.PREP_VEC

_REL_XY = passcost.REL_XY
_REL_XY_ABS = passcost.REL_XY_ABS
_REL_Z = passcost.REL_Z
_REL_ANGLES = passcost.REL_ANGLES
_ANG_CMP = passcost.ANG_CMP
_ANG_X_LARGER = passcost.ANG_X_LARGER
_ANG_Y_LARGER = passcost.ANG_Y_LARGER
_ANG_EQ = passcost.ANG_EQ
_ANG_EQ_Y = passcost.ANG_EQ_Y
_ANG_EQ_X = passcost.ANG_EQ_X
_ANG_MIN_Y = passcost.ANG_MIN_Y
_ANG_MIN_X = passcost.ANG_MIN_X
_ANG_ZERO = passcost.ANG_ZERO
_ANG_NONZERO = passcost.ANG_NONZERO
_SCALE_SHIFT = passcost.SCALE_SHIFT
_SCALE_LOOP = passcost.SCALE_LOOP
_SCALE_LOOP_Y = passcost.SCALE_LOOP_Y
_SCALE_TAIL = passcost.SCALE_TAIL
_ANG_SIGN = passcost.ANG_SIGN
_ANG_SIGN_KEEP = passcost.ANG_SIGN_KEEP
_ANG_SIGN_NEGATE = passcost.ANG_SIGN_NEGATE
_ANG_QUAD = passcost.ANG_QUAD
_ANG_QUAD_LOW = passcost.ANG_QUAD_LOW
_ANG_QUAD_HIGH = passcost.ANG_QUAD_HIGH
_ANG_QUAD_TAIL = passcost.ANG_QUAD_TAIL

_DIV_FULL_SHIFT = passcost.DIV_FULL_SHIFT
_DIV_FULL_CARRY = passcost.DIV_FULL_CARRY
_DIV_FULL_UNDER = passcost.DIV_FULL_UNDER
_DIV_FULL_OVER = passcost.DIV_FULL_OVER
_DIV_FULL_EQ = passcost.DIV_FULL_EQ
_DIV_FULL_EQ_UNDER = passcost.DIV_FULL_EQ_UNDER
_DIV_FULL_EQ_OVER = passcost.DIV_FULL_EQ_OVER
_DIV_SUB = passcost.DIV_SUB
_DIV_SKIP_TEST = passcost.DIV_SKIP_TEST
_DIV_SKIP = passcost.DIV_SKIP
_DIV_NO_SKIP = passcost.DIV_NO_SKIP
_DIV_BYTE_SHIFT = passcost.DIV_BYTE_SHIFT
_DIV_BYTE_CARRY = passcost.DIV_BYTE_CARRY
_DIV_BYTE_UNDER = passcost.DIV_BYTE_UNDER
_DIV_BYTE_OVER = passcost.DIV_BYTE_OVER
_DIV_LAST_SHIFT = passcost.DIV_LAST_SHIFT
_DIV_LAST_CARRY = passcost.DIV_LAST_CARRY
_DIV_LAST_CMP = passcost.DIV_LAST_CMP
_DIV_LAST_TAIL = passcost.DIV_LAST_TAIL
_DIV_PLP = passcost.DIV_PLP
_DIV_ROUND3_UNDER = passcost.DIV_ROUND3_UNDER
_DIV_ROUND3_OVER = passcost.DIV_ROUND3_OVER
_DIV_NO_OVERFLOW = passcost.DIV_NO_OVERFLOW
_DIV_OVERFLOW = passcost.DIV_OVERFLOW
_DIV_ARCTAN = passcost.DIV_ARCTAN
_DIV_ARCTAN_DONE = passcost.DIV_ARCTAN_DONE
_DIV_DELTA_BR = passcost.DIV_DELTA_BR
_DIV_HALF_BR = passcost.DIV_HALF_BR
_DIV_DELTA = passcost.DIV_DELTA
_DIV_DELTA_KEEP = passcost.DIV_DELTA_KEEP
_DIV_DELTA_INVERT = passcost.DIV_DELTA_INVERT
_DIV_DELTA_HALVE = passcost.DIV_DELTA_HALVE
_DIV_NEXT = passcost.DIV_NEXT
_DIV_ADD_SKIP = passcost.DIV_ADD_SKIP
_DIV_ADD_HALF = passcost.DIV_ADD_HALF
_DIV_AVERAGE = passcost.DIV_AVERAGE
_DIV_TABLE_CROSS = passcost.DIV_TABLE_CROSS

_VANG_HEAD = passcost.VANG_HEAD
_VANG_POS = passcost.VANG_POS
_VANG_NEG = passcost.VANG_NEG
_VANG_SETUP = passcost.VANG_SETUP
_VANG_SHIFT = passcost.VANG_SHIFT
_VANG_SIGN_POS = passcost.VANG_SIGN_POS
_VANG_SIGN_NEG = passcost.VANG_SIGN_NEG
_VANG_TAIL = passcost.VANG_TAIL
_HYP_HEAD = passcost.HYP_HEAD
_HYP_TAIL = passcost.HYP_TAIL

_CONSIDER_MEANIE = passcost.CONSIDER_MEANIE
_CONSIDER_NO_MEANIE = passcost.CONSIDER_NO_MEANIE
_DISCHARGE_CALL = passcost.DISCHARGE_CALL
_DISCHARGED = passcost.DISCHARGED
_NO_DISCHARGE = passcost.NO_DISCHARGE
_HUNT_CLEAR = passcost.HUNT_CLEAR
_HUNT_CALL = passcost.HUNT_CALL
_HUNT_MISS = passcost.HUNT_MISS
_HUNT_HIT = passcost.HUNT_HIT
_HELD_NONE = passcost.HELD_NONE
_HELD_CALL = passcost.HELD_CALL
_HELD_LOST = passcost.HELD_LOST
_HELD_KEPT = passcost.HELD_KEPT
_SCAN_INIT = passcost.SCAN_INIT
_SCAN_SLOT_HIDDEN = passcost.SCAN_SLOT_HIDDEN
_SCAN_SLOT_UNSEEN = passcost.SCAN_SLOT_UNSEEN
_SCAN_SLOT_FULL = passcost.SCAN_SLOT_FULL
_SCAN_SLOT_OTHER = passcost.SCAN_SLOT_OTHER
_SCAN_SLOT_PARTIAL = passcost.SCAN_SLOT_PARTIAL
_SCAN_LAST = passcost.SCAN_LAST
_SCAN_END = passcost.SCAN_END
_SCAN_END_PARTIAL = passcost.SCAN_END_PARTIAL
_PARTIAL_KNOWN = passcost.PARTIAL_KNOWN
_PARTIAL_ARM = passcost.PARTIAL_ARM
_TREE_CALL = passcost.TREE_CALL
_TREE_NONE = passcost.TREE_NONE
_TREE_HIT = passcost.TREE_HIT
_DRAIN_CALL = passcost.DRAIN_CALL
_DRAIN_TAIL = passcost.DRAIN_TAIL
_BODY_TAIL = passcost.BODY_TAIL
_TARGET_HEAD = passcost.TARGET_HEAD
_TARGET_FIRST = passcost.TARGET_FIRST
_TARGET_WAIT = passcost.TARGET_WAIT
_TARGET_DUE = passcost.TARGET_DUE
_TARGET_MEANIE = passcost.TARGET_MEANIE
_TARGET_DRAIN = passcost.TARGET_DRAIN
_TARGET_DRAIN_OBJ = passcost.TARGET_DRAIN_OBJ
_TARGET_DRAIN_PLAYER = passcost.TARGET_DRAIN_PLAYER

_REDUCE_HEAD = passcost.REDUCE_HEAD
_REDUCE_KILL = passcost.REDUCE_KILL
_REDUCE_PLAYER = passcost.REDUCE_PLAYER
_REDUCE_OBJECT = passcost.REDUCE_OBJECT
_REDUCE_ROBOT = passcost.REDUCE_ROBOT
_REDUCE_TREE = passcost.REDUCE_TREE
_REDUCE_BOULDER = passcost.REDUCE_BOULDER
_REDUCE_BANK = passcost.REDUCE_BANK
_REMOVE_STACKED = passcost.REMOVE_STACKED
_REMOVE_GROUND = passcost.REMOVE_GROUND
_TUNE_DRAIN = passcost.TUNE_DRAIN
_STATUS_HEAD = passcost.STATUS_HEAD
_STATUS_BLOCK = passcost.STATUS_BLOCK
_STATUS_BLOCK_DONE = passcost.STATUS_BLOCK_DONE
_STATUS_UNIT_NONE = passcost.STATUS_UNIT_NONE
_STATUS_UNIT = passcost.STATUS_UNIT
_STATUS_PAD = passcost.STATUS_PAD
_STATUS_MID = passcost.STATUS_MID
_STATUS_TAIL = passcost.STATUS_TAIL
_STATUS_PAD_END = passcost.STATUS_PAD_END
_STATUS_PAD_LAPS = passcost.STATUS_PAD_LAPS

_TILE_SCAN_ENTRY = passcost.TILE_SCAN_ENTRY
_TILE_SCAN_EXHAUSTED = passcost.TILE_SCAN_EXHAUSTED
_TILE_SCAN_LAST = passcost.TILE_SCAN_LAST
_TILE_SCAN_SEE_BOULDER = passcost.TILE_SCAN_SEE_BOULDER
_TILE_SCAN_EMPTY = passcost.TILE_SCAN_EMPTY
_TILE_SCAN_OTHER = passcost.TILE_SCAN_OTHER
_TILE_SCAN_STACKED = passcost.TILE_SCAN_STACKED
_TILE_SCAN_LOOSE = passcost.TILE_SCAN_LOOSE
_TILE_SCAN_TILE = passcost.TILE_SCAN_TILE
_TILE_SCAN_NO_TILE = passcost.TILE_SCAN_NO_TILE
_TILE_SCAN_TOP = passcost.TILE_SCAN_TOP
_TILE_SCAN_WRONG_TOP = passcost.TILE_SCAN_WRONG_TOP
_TILE_SCAN_SEE = passcost.TILE_SCAN_SEE
_TILE_SCAN_NEXT = passcost.TILE_SCAN_NEXT
_TILE_SCAN_HIT = passcost.TILE_SCAN_HIT

_MEANIE_SCAN_SLOT = passcost.MEANIE_SCAN_SLOT
_MEANIE_SCAN_OTHER = passcost.MEANIE_SCAN_OTHER
_MEANIE_SCAN_DX = passcost.MEANIE_SCAN_DX
_MEANIE_SCAN_DY = passcost.MEANIE_SCAN_DY
_MEANIE_SCAN_SEE = passcost.MEANIE_SCAN_SEE
_MEANIE_SCAN_DONE = passcost.MEANIE_SCAN_DONE

_ROTATE_GATE = passcost.ROTATE_GATE
_ROTATE_GATE_HELD = passcost.ROTATE_GATE_HELD
_ROTATE = passcost.ROTATE
_REDRAW_CALL = passcost.REDRAW_CALL
_REDRAW_NONE = passcost.REDRAW_NONE
_REDRAW_PLOT_ENTRY = passcost.REDRAW_PLOT_ENTRY
_SPAN_HEAD = passcost.SPAN_HEAD
_SPAN_PLAYER = passcost.SPAN_PLAYER
_SPAN_ANGLES = passcost.SPAN_ANGLES
_SPAN_SIZE = passcost.SPAN_SIZE
_SPAN_SIZE_FLOOR = passcost.SPAN_SIZE_FLOOR
_SPAN_LEFT = passcost.SPAN_LEFT
_SPAN_LEFT_HI = passcost.SPAN_LEFT_HI
_SPAN_LEFT_NEG = passcost.SPAN_LEFT_NEG
_SPAN_LEFT_POS = passcost.SPAN_LEFT_POS
_SPAN_OFF_RIGHT = passcost.SPAN_OFF_RIGHT
_SPAN_LEFT_ONSCREEN = passcost.SPAN_LEFT_ONSCREEN
_SPAN_LEFT_OK = passcost.SPAN_LEFT_OK
_SPAN_RIGHT = passcost.SPAN_RIGHT
_SPAN_RIGHT_HI = passcost.SPAN_RIGHT_HI
_SPAN_BEHIND = passcost.SPAN_BEHIND
_SPAN_RIGHT_OK = passcost.SPAN_RIGHT_OK
_SPAN_WIDTH_CLIP = passcost.SPAN_WIDTH_CLIP
_SPAN_WIDTH = passcost.SPAN_WIDTH
_SPAN_ZERO_WIDTH = passcost.SPAN_ZERO_WIDTH
_SPAN_VISIBLE = passcost.SPAN_VISIBLE
_SPAN_WIDTH_CAP = passcost.SPAN_WIDTH_CAP
_MEANIE_ROTATE = passcost.MEANIE_ROTATE

_COOLDOWN_TICK_NO_CARRY = passcost.COOLDOWN_TICK_NO_CARRY
_COOLDOWN_TICK_GATE = passcost.COOLDOWN_TICK_GATE
_COOLDOWN_TICK_WALK = passcost.COOLDOWN_TICK_WALK
_COOLDOWN_TICK_BYTE_STICK = passcost.COOLDOWN_TICK_BYTE_STICK
_COOLDOWN_TICK_BYTE_DEC = passcost.COOLDOWN_TICK_BYTE_DEC
_COOLDOWN_TICK_LAST = passcost.COOLDOWN_TICK_LAST
_IRQ_GATE_SHUT = passcost.IRQ_GATE_SHUT
_IRQ_GATE_OPEN = passcost.IRQ_GATE_OPEN

_SOUND_TICK_FIXED = passcost.SOUND_TICK_FIXED
_SOUND_VOICE_READ = passcost.SOUND_VOICE_READ
_SOUND_VOICE_OFF = passcost.SOUND_VOICE_OFF
_SOUND_VOICE_SPENT = passcost.SOUND_VOICE_SPENT
_SOUND_VOICE_TICK = passcost.SOUND_VOICE_TICK
_SOUND_VOICE_MORE = passcost.SOUND_VOICE_MORE
_SOUND_VOICE_NOTE = passcost.SOUND_VOICE_NOTE
_SOUND_GATE_READ = passcost.SOUND_GATE_READ
_SOUND_GATE_OFF = passcost.SOUND_GATE_OFF
_SOUND_GATE_TICK = passcost.SOUND_GATE_TICK
_SOUND_GATE_MORE = passcost.SOUND_GATE_MORE
_SOUND_GATE_END = passcost.SOUND_GATE_END
_SOUND_VOICE_NEXT = passcost.SOUND_VOICE_NEXT
_SOUND_VOICE_LAST = passcost.SOUND_VOICE_LAST

_BODY_ENTRY = 0  # resume points inside $16E6, mirroring sentinel.enemies.BODY_*
_BODY_MEANIE = 1
_BODY_DISCHARGE = 2
_BODY_HUNT = 3
_BODY_HELD = 4
_BODY_SCAN = 5
_BODY_PARTIAL = 6
_BODY_TREE = 7
_BODY_ROTATE = 8
_BODY_MAKE_MEANIE = 9
_BODY_DONE = -1

_MAX_STEPS = 20000  # can_see_object's march bound (the ROM's board-edge exit)
ZP_LO = 0x50  # the zero-page window the geometry touches ($0050..$008B)
ZP_HI = 0x8C
ZP_REPLOT = 0x8C  # out of band: the object $1F9F found a screen span for, +1
ZP_REPLOT_LEFT = 0x8D  # ... and its $0C62 left column and $0C69 width; the
ZP_REPLOT_COLS = 0x8E  # $1FFC plot itself is priced outside the twin
ZP_N = 0x8F


@njit(cache=True, inline="always")
def _rd(mem, addr):
    return np.int64(mem[addr])


@njit(cache=True, inline="always")
def _wr(mem, addr, val):
    mem[addr] = np.uint8(val & 0xFF)


@njit(cache=True)
def _start_tune(mem, tune):
    """$3470 -> $8DB4: point a voice at tune's $AC00 descriptor (the tick's bytes only)."""
    desc = _SND_TABLE + tune * 8
    x = _rd(mem, desc)
    _wr(mem, _SND_LENGTH + x, _rd(mem, _SND_LENGTHS + (_rd(mem, desc + 3) & 0x0F)))
    if _rd(mem, desc + 7) & 0x80:
        _wr(mem, _SND_FLAG + x, _SND_IDLE)
    else:
        _wr(mem, _SND_FLAG + x, _rd(mem, _SND_VOICE_ON + x))
    _wr(mem, _SND_NOTE + x, _rd(mem, desc + 6))


@njit(cache=True, inline="always")
def _tile_byte(mem, x, y):
    """calculate_tile_address $2BA8, masked-8-bit form (edge reads wrap as on 6502)."""
    lo = (((x & 0xFF) << 3) & 0xE0) | (y & 0x1F)
    return np.int64(mem[((x & 3) + 4) * 256 + lo])


@njit(cache=True, inline="always")
def _set_tile_byte(mem, x, y, val):
    """Write a tiles_table byte through the same addressing as :func:`_tile_byte`."""
    lo = (((x & 0xFF) << 3) & 0xE0) | (y & 0x1F)
    mem[((x & 3) + 4) * 256 + lo] = np.uint8(val & 0xFF)


@njit(cache=True)
def _prng_next(mem):
    """prnd $31CA: 8 shuffles of the 5-byte LFSR at $0C7B, returning $0C7F."""
    s0 = np.int64(mem[_PRND])
    s1 = np.int64(mem[_PRND + 1])
    s2 = np.int64(mem[_PRND + 2])
    s3 = np.int64(mem[_PRND + 3])
    s4 = np.int64(mem[_PRND + 4])
    for _ in range(8):
        carry = ((s2 >> 3) ^ s4) & 1
        v = (s0 << 1) | carry
        carry = v >> 8
        s0 = v & 0xFF
        v = (s1 << 1) | carry
        carry = v >> 8
        s1 = v & 0xFF
        v = (s2 << 1) | carry
        carry = v >> 8
        s2 = v & 0xFF
        v = (s3 << 1) | carry
        carry = v >> 8
        s3 = v & 0xFF
        s4 = ((s4 << 1) | carry) & 0xFF
    mem[_PRND] = np.uint8(s0)
    mem[_PRND + 1] = np.uint8(s1)
    mem[_PRND + 2] = np.uint8(s2)
    mem[_PRND + 3] = np.uint8(s3)
    mem[_PRND + 4] = np.uint8(s4)
    return s4


@njit(cache=True, inline="always")
def _asl(v):
    return (v << 1) & 0xFF, (v >> 7) & 1


@njit(cache=True, inline="always")
def _rol(v, c):
    return ((v << 1) | c) & 0xFF, (v >> 7) & 1


@njit(cache=True, inline="always")
def _ror(v, c):
    return ((c << 7) | (v >> 1)) & 0xFF, v & 1


@njit(cache=True, inline="always")
def _full_over(ca, a, t74, t76, t77):
    """Rounds 1-3 of $0D4A: a >= b as a 16-bit compare, and its branches' cost."""
    if ca:
        return True, np.int64(_DIV_FULL_CARRY)
    if a > t76:
        return True, np.int64(_DIV_FULL_OVER)
    if a < t76:
        return False, np.int64(_DIV_FULL_UNDER)
    if t74 >= t77:
        return True, np.int64(_DIV_FULL_EQ + _DIV_FULL_EQ_OVER)
    return False, np.int64(_DIV_FULL_EQ + _DIV_FULL_EQ_UNDER)


@njit(cache=True)
def _finish_overflow(a, php_carry, t78, cyc):
    """consider_overflow $0DFC: the round-3 bit, the 45-degree clamp, then the
    arctan lookup + rounds-9/10 interpolation."""
    carry = 0
    cyc += np.int64(_DIV_PLP)
    if php_carry:
        s = a + 0x1F + 1
        a = s & 0xFF
        carry = 1 if s > 0xFF else 0
        cyc += np.int64(_DIV_ROUND3_OVER)
    else:
        cyc += np.int64(_DIV_ROUND3_UNDER)
    if carry:
        return np.int64(0x00), np.int64(0x20), np.int64(0xFF), cyc + _DIV_OVERFLOW
    if php_carry:
        cyc += np.int64(_DIV_NO_OVERFLOW)

    y = a & 0xFF
    ratio = y
    ang_lo = ARCTAN_LO[y]
    ang_hi = ARCTAN_HI[y]
    cyc += np.int64(_DIV_ARCTAN)
    if y >= 0xFF:
        cyc += np.int64(_DIV_TABLE_CROSS)
    b78_7 = (t78 >> 7) & 1
    b78_6 = (t78 >> 6) & 1
    if b78_7 == 0 and b78_6 == 0:
        return (
            np.int64(ang_lo),
            np.int64(ang_hi),
            np.int64(ratio),
            cyc + _DIV_ARCTAN_DONE,
        )
    # One $3B01,Y and one $3C02,Y read per interpolation block, a cycle dearer off-page
    cross = np.int64(0)
    if y >= 0xFF:
        cross += np.int64(_DIV_TABLE_CROSS)
    if y >= 0xFE:
        cross += np.int64(_DIV_TABLE_CROSS)

    nxt_lo = ARCTAN_LO[y + 1]
    nxt_hi = ARCTAN_HI[y + 1]
    d_lo = (ang_lo - nxt_lo) & 0xFF
    borrow = 1 if ang_lo < nxt_lo else 0
    d_hi = (ang_hi - nxt_hi - borrow) & 0xFF
    if b78_7:
        cyc += np.int64(_DIV_DELTA_BR + _DIV_DELTA) + cross
        if b78_6:  # round 9 over -> invert the delta
            d_hi, d_lo = _vinvert16(d_hi, d_lo)
            cyc += np.int64(_DIV_DELTA_INVERT)
        else:
            cyc += np.int64(_DIV_DELTA_KEEP)
        cyc += np.int64(_DIV_DELTA_HALVE)
    else:
        cyc += np.int64(_DIV_HALF_BR)  # $0E30 BVS $0E50: only the average is wanted
    # ROL A ; ROR $0075 ; ROR $0074 -- arithmetic >>1 keeping the sign.
    t75, c2 = _ror(d_hi, (d_hi >> 7) & 1)
    t74b, _c = _ror(d_lo, c2)
    s = ang_lo + nxt_lo
    ang_lo = s & 0xFF
    cc = 1 if s > 0xFF else 0
    ang_hi = (ang_hi + nxt_hi + cc) & 0xFF
    cyc += np.int64(_DIV_NEXT) + cross
    if b78_7:  # round 10 over -> add half-delta
        s = ang_lo + t74b
        ang_lo = s & 0xFF
        cc = 1 if s > 0xFF else 0
        ang_hi = (ang_hi + t75 + cc) & 0xFF
        cyc += np.int64(_DIV_ADD_HALF)
    else:
        cyc += np.int64(_DIV_ADD_SKIP)
    ang_hi, c3 = _ror(ang_hi, 0)
    ang_lo, _c2 = _ror(ang_lo, c3)
    return np.int64(ang_lo), np.int64(ang_hi), np.int64(ratio), cyc + _DIV_AVERAGE


@njit(cache=True)
def _divide_and_arctan(a_lo, a_hi, b_lo, b_hi):
    """$0D4A: shift/subtract divide of 16-bit a by 16-bit b, then arctan(a/b)."""
    t74 = a_lo & 0xFF
    a = a_hi & 0xFF
    t76 = b_hi & 0xFF
    t77 = b_lo & 0xFF
    t78 = np.int64(0)
    cyc = np.int64(0)

    c = 0
    php_carry = 0
    for rnd in range(1, 4):
        if rnd == 1:
            t74, c0 = _asl(t74)
        else:
            t74, c0 = _rol(t74, c)
        a, ca = _rol(a, c0)
        over, dcyc = _full_over(ca, a, t74, t76, t77)
        cyc += np.int64(_DIV_FULL_SHIFT) + dcyc
        if over:
            v = t74 - t77
            t74 = v & 0xFF
            a = (a - t76 - (1 if v < 0 else 0)) & 0xFF
            c = 1
            cyc += np.int64(_DIV_SUB)
        else:
            c = 0
        if rnd == 3:
            php_carry = c  # PHP after round 3

    cyc += np.int64(_DIV_SKIP_TEST)
    if a == t76:  # skip_further_division $0E10
        cyc += np.int64(_DIV_SKIP)
        a = np.int64(0)
        cc = 1
        t74, cc = _ror(t74, cc)
        a, cc = _ror(a, cc)
        t74, cc = _ror(t74, cc)
        a, cc = _ror(a, cc)
        return _finish_overflow(a | 0x20, php_carry, np.int64(0), cyc)
    cyc += np.int64(_DIV_NO_SKIP)

    for rnd in range(4, 10):
        if rnd == 4:
            t74, c0 = _asl(t74)
        else:
            t74, c0 = _rol(t74, c)
        a, ca = _rol(a, c0)
        cyc += np.int64(_DIV_BYTE_SHIFT)
        if ca:
            cyc += np.int64(_DIV_BYTE_CARRY)
        elif a >= t76:
            cyc += np.int64(_DIV_BYTE_OVER)
        else:
            cyc += np.int64(_DIV_BYTE_UNDER)
        if ca or a >= t76:
            a = (a - t76) & 0xFF
            c = 1
        else:
            c = 0
    # round 10 $0DF1: ROR $78 (round-9 bit), ROL A, compare, ROR $78
    t78, c0 = _ror(t78, c)
    a, ca = _rol(a, c0)
    cyc += np.int64(_DIV_LAST_SHIFT + _DIV_LAST_TAIL)
    if ca:
        c = 1
        cyc += np.int64(_DIV_LAST_CARRY)
    else:
        c = 1 if a >= t76 else 0
        cyc += np.int64(_DIV_LAST_CMP)
    t78, _c = _ror(t78, c)
    return _finish_overflow(t74, php_carry, t78, cyc)


@njit(cache=True)
def _normalise(zp, max_lo_a, max_hi_a, min_lo_a, min_hi_a, lap):
    """scale_using_x/_y $92C1/$92FF: shift max left until it overflows, min in
    lock-step, then back the max off by one.  ``lap`` is one loop-back's cost.
    Returns (b_lo, b_hi, a_lo, a_hi, cycles)."""
    max_lo = zp[max_lo_a]
    min_lo = zp[min_lo_a]
    min_hi = zp[min_hi_a]
    a = zp[max_hi_a]
    laps = 0
    while True:
        max_lo, c = _asl(max_lo)
        a, ca = _rol(a, c)
        if ca:  # max overflowed
            break
        min_lo, c2 = _asl(min_lo)
        min_hi, _c = _rol(min_hi, c2)
        laps += 1
    a, c = _ror(a, 1)
    max_lo, _c2 = _ror(max_lo, c)
    cyc = (
        np.int64(_SCALE_SHIFT) + np.int64(laps) * np.int64(lap) + np.int64(_SCALE_TAIL)
    )
    return max_lo & 0xFC, a, min_lo, min_hi, cyc


@njit(cache=True)
def _calc_angle(zp):
    """calculate_angle $9287: quadrant-folded arctan of (x, y), and its cycles."""
    x_lo = zp[0x80]
    x_hi = zp[0x83]
    y_lo = zp[0x82]
    y_hi = zp[0x85]
    sx = zp[0x86]
    sy = zp[0x88]
    cyc = np.int64(_ANG_CMP)
    if y_hi < x_hi:
        x_larger = True
        cyc += np.int64(_ANG_X_LARGER)
    elif y_hi > x_hi:
        x_larger = False
        cyc += np.int64(_ANG_Y_LARGER)
    else:
        x_larger = y_lo < x_lo
        cyc += np.int64(_ANG_EQ)
        cyc += np.int64(_ANG_EQ_X if x_larger else _ANG_EQ_Y)
    if x_larger:
        cyc += np.int64(_ANG_MIN_Y)
        zp[0x5D] = y_hi
        zp[0x5C] = y_lo
        zp[0x7A] = x_lo
        zp[0x7B] = x_hi
        b_lo, b_hi, a_lo, a_hi, ncyc = _normalise(
            zp, 0x80, 0x83, 0x82, 0x85, _SCALE_LOOP
        )
        ang_lo, ang_hi, ratio, dcyc = _divide_and_arctan(a_lo, a_hi, b_lo, b_hi)
        cyc += ncyc + dcyc + np.int64(_ANG_SIGN)
        if ((sx ^ sy) & 0x80) == 0:  # same sign -> negate angle
            ang_hi, ang_lo = _vinvert16(ang_hi, ang_lo)
            cyc += np.int64(_ANG_SIGN_NEGATE)
        else:
            cyc += np.int64(_ANG_SIGN_KEEP)
        base = 0x40 if (sx & 0x80) == 0 else 0xC0
        cyc += np.int64(_ANG_QUAD)
        cyc += np.int64(_ANG_QUAD_LOW if base == 0x40 else _ANG_QUAD_HIGH)
        ang_hi = (ang_hi + base) & 0xFF
    else:
        cyc += np.int64(_ANG_MIN_X)
        if (y_hi | y_lo) == 0:  # both zero
            zp[0x7E] = 0
            zp[0x8A] = 0
            zp[0x8B] = 0
            return cyc + np.int64(_ANG_ZERO)
        cyc += np.int64(_ANG_NONZERO)
        zp[0x5D] = x_hi
        zp[0x5C] = x_lo
        zp[0x7A] = y_lo
        zp[0x7B] = y_hi
        b_lo, b_hi, a_lo, a_hi, ncyc = _normalise(
            zp, 0x82, 0x85, 0x80, 0x83, _SCALE_LOOP_Y
        )
        ang_lo, ang_hi, ratio, dcyc = _divide_and_arctan(a_lo, a_hi, b_lo, b_hi)
        cyc += ncyc + dcyc + np.int64(_ANG_SIGN)
        if (sx ^ sy) & 0x80:  # opposite sign -> negate angle
            ang_hi, ang_lo = _vinvert16(ang_hi, ang_lo)
            cyc += np.int64(_ANG_SIGN_NEGATE)
        else:
            cyc += np.int64(_ANG_SIGN_KEEP)
        base = 0x00 if (sy & 0x80) == 0 else 0x80
        cyc += np.int64(_ANG_QUAD)
        cyc += np.int64(_ANG_QUAD_LOW if base == 0x00 else _ANG_QUAD_HIGH)
        ang_hi = (ang_hi + base) & 0xFF
    zp[0x8A] = ang_lo
    zp[0x8B] = ang_hi
    zp[0x7E] = ratio
    return cyc + np.int64(_ANG_QUAD_TAIL)


@njit(cache=True)
def _calc_hypotenuse(zp):
    """calculate_hypotenuse $937F: distance = max + f*min/512."""
    ratio = zp[0x7E]
    f = HYP[(ratio >> 1) + (ratio & 1)]
    res_lo, res_hi, mcyc = _vmul_dbl_by_byte(zp[0x5C], f, zp[0x5D])
    new_hi = res_hi >> 1
    new_lo = ((res_hi & 1) << 7) | (res_lo >> 1)
    s = new_lo + zp[0x7A]
    zp[0x7C] = s & 0xFF
    cc = 1 if s > 0xFF else 0
    zp[0x7D] = (new_hi + zp[0x7B] + cc) & 0xFF
    return np.int64(_HYP_HEAD + _HYP_TAIL) + np.int64(mcyc)


@njit(cache=True)
def _vertical_angle(zp, z_hi, v_angle):
    """calculate_object_relative_vertical_angle $933D."""
    sx = z_hi & 0xFF
    cyc = np.int64(_VANG_HEAD)
    if sx & 0x80:  # negative -> make positive
        zlo = (-zp[0x80]) & 0xFF
        borrow = 1 if zp[0x80] != 0 else 0
        sx_abs = (-sx - borrow) & 0xFF
        zp[0x80] = zlo
        cyc += np.int64(_VANG_NEG)
    else:
        sx_abs = sx
        cyc += np.int64(_VANG_POS)
    zp[0x83] = sx_abs
    zp[0x82] = zp[0x7C]
    zp[0x85] = zp[0x7D]
    zp[0x88] = 0
    zp[0x86] = sx
    cyc += np.int64(_VANG_SETUP) + _calc_angle(zp) + np.int64(_VANG_SHIFT)
    lo = zp[0x8A] - 0x20
    t50 = lo & 0xFF
    a = (zp[0x8B] - v_angle - (1 if lo < 0 else 0)) & 0xFF
    neg = a & 0x80
    for _ in range(4):
        c = a & 1
        a = a >> 1
        t50 = ((c << 7) | (t50 >> 1)) & 0xFF
    if neg:
        a = a | 0xF0
        cyc += np.int64(_VANG_SIGN_NEG)
    else:
        cyc += np.int64(_VANG_SIGN_POS)
    zp[0x50] = t50
    return a & 0xFF, cyc + np.int64(_VANG_TAIL)


@njit(cache=True)
def _relative_angles(mem, zp, observer, target):
    """calculate_object_relative_angles_and_distance $8401, play-mode path.

    The zero-page window is zeroed here, where the reference allocates its scratch.
    Returns (c57, angle_lo, angle_hi, z_lo, z_hi, cycles)."""
    for i in range(ZP_LO, ZP_HI):
        zp[i] = 0
    dx = (_rd(mem, _OX + target) - _rd(mem, _OX + observer)) & 0xFF
    zp[0x86] = dx
    zp[0x80] = 0
    zp[0x83] = (-dx) & 0xFF if dx & 0x80 else dx
    dy = (_rd(mem, _OY + target) - _rd(mem, _OY + observer)) & 0xFF
    zp[0x88] = dy
    zp[0x82] = 0
    zp[0x85] = (-dy) & 0xFF if dy & 0x80 else dy
    v = _rd(mem, _OZF + target) - _rd(mem, _OZF + observer)
    zp[0x81] = v & 0xFF
    zp[0x84] = (
        _rd(mem, _OZH + target) - _rd(mem, _OZH + observer) - (1 if v < 0 else 0)
    ) & 0xFF
    cyc = np.int64(_REL_ANGLES + _REL_XY + _REL_Z)
    cyc += np.int64(_REL_XY_ABS) * np.int64((dx >> 7) + (dy >> 7))
    cyc += _calc_angle(zp)
    c57 = (zp[0x8B] - _rd(mem, _OHANG + observer) + 0x0A) & 0xFF
    cyc += _calc_hypotenuse(zp)
    return c57, zp[0x8A], zp[0x8B], zp[0x81], zp[0x84], cyc


@njit(cache=True)
def _update_object_on_screen(mem, zp, target):
    """update_object_on_screen $1F9F via calculate_object_screen_span $209B.

    Returns (cycles, columns); columns is 0 when the object has no screen span and
    the cost is then whole, else the $0C69 strip width, the $1FFC replot unpriced."""
    cyc = np.int64(_REDRAW_CALL + _SPAN_HEAD)
    player = _rd(mem, _PLAYER)
    if target == player:  # $209F: the player is never drawn
        return cyc + np.int64(_SPAN_PLAYER + _REDRAW_NONE), 0
    c57, _alo, _ahi, _zlo, _zhi, rcyc = _relative_angles(mem, zp, player, target)
    c59 = zp[0x8A]
    cyc += np.int64(_SPAN_ANGLES + _SPAN_SIZE) + rcyc
    otype = _rd(mem, _OTYPE + target)
    half = HALF_ANGLE[otype]
    if half < _rd(mem, _SIZE_FLOOR):  # $20AC: a pending size wins
        half = _rd(mem, _SIZE_FLOOR)
        cyc += np.int64(_SPAN_SIZE_FLOOR)
    _wr(mem, _SIZE_FLOOR, 0)  # $20B8
    zp[0x80] = half
    cyc += _vertical_angle(zp, 0, _rd(mem, _OVANGLE + otype))[1]
    ang_lo = zp[0x8A]
    ang_hi = zp[0x8B]
    cyc += np.int64(_SPAN_LEFT + _SPAN_LEFT_HI)
    d = c59 - ang_lo
    hi = (c57 - ang_hi - (1 if d < 0 else 0)) & 0xFF
    if hi & 0x80:  # $20CB: the left edge is behind the view origin -> column 0
        left = 0
        cyc += np.int64(_SPAN_LEFT_NEG)
    else:
        cyc += np.int64(_SPAN_LEFT_POS)
        left = ((hi << 1) | ((d & 0xFF) >> 7)) & 0xFF
        if left >= 0x28:  # $20D6: off the right of the screen
            return cyc + np.int64(_SPAN_OFF_RIGHT + _REDRAW_NONE), 0
        cyc += np.int64(_SPAN_LEFT_ONSCREEN)
    cyc += np.int64(_SPAN_LEFT_OK + _SPAN_RIGHT + _SPAN_RIGHT_HI)
    s = c59 + ang_lo
    hi = (c57 + ang_hi + (1 if s > 0xFF else 0)) & 0xFF
    if hi & 0x80:  # $20EB: the right edge is left of the view
        return cyc + np.int64(_SPAN_BEHIND + _REDRAW_NONE), 0
    cyc += np.int64(_SPAN_RIGHT_OK)
    right = ((hi << 1) | ((s & 0xFF) >> 7)) & 0xFF
    if right >= 0x28:  # $20F2: clip to the last column
        right = 0x27
        cyc += np.int64(_SPAN_WIDTH_CLIP)
    cyc += np.int64(_SPAN_WIDTH)
    width = (right + 1 - left) & 0xFF
    if width == 0:  # $2103
        return cyc + np.int64(_SPAN_ZERO_WIDTH + _REDRAW_NONE), 0
    cyc += np.int64(_SPAN_VISIBLE)
    if width >= 0x15:  # $2105: a replot spans at most 20 columns at once
        width = 0x14
        cyc += np.int64(_SPAN_WIDTH_CAP)
    zp[ZP_REPLOT] = target + 1  # the $1FFC strip replot is the caller's to price
    zp[ZP_REPLOT_LEFT] = left
    zp[ZP_REPLOT_COLS] = width
    return cyc + np.int64(_REDRAW_PLOT_ENTRY), width


@njit(cache=True)
def _prep_vec_angle(h_angle, h_frac, v_angle, v_frac):
    """prepare_vector_from_angle $1C54, the standalone entry the enemy probes use.

    Returns (vx_lo, vx_hi, vz_lo, vz_hi, vy_lo, vy_hi, s30, cycles)."""
    sin_lo_v, cos_lo_v, sin_hi_v, cos_hi_v, cyc = _vsin_cos(v_angle, v_frac)
    cyc += _PREP_VEC
    s33, s32, c = _vproc_sc(cos_lo_v, cos_hi_v)
    cyc += c
    s30, s2d, c = _vproc_sc(sin_lo_v, sin_hi_v)
    cyc += c
    h_sin_lo, h_cos_lo, h_sin_hi, h_cos_hi, c = _vsin_cos(h_angle, h_frac)
    cyc += c
    vy_hi, vy_lo, c = _vmul_dbl_dbl(h_cos_lo, h_cos_hi, s32, s33)
    cyc += c
    vx_hi, vx_lo, c = _vmul_dbl_dbl(h_sin_lo, h_sin_hi, s32, s33)
    cyc += c
    return vx_lo, vx_hi, s2d, s30, vy_lo, vy_hi, s30, cyc


@njit(cache=True)
def _can_see_object(mem, zp, observer, target, expected_type, fov_width):
    """check_if_enemy_can_see_object $1887.  A robot is probed at its upper point
    first ($18DC, $0C6E bit7 set so the looking-up rejection is waived) then at its
    base; every other object only at its base.

    Returns (in_slot, in_fov, exposure, full, tree_in_los_head, cycles)."""
    mem[0x0014] = 0
    if mem[_OFLAGS + target] & 0x80:  # empty slot
        return 0, 0, np.int64(0), 0, 0, np.int64(_SEE_SLOT_EMPTY)
    if mem[_OTYPE + target] != expected_type:
        return 0, 0, np.int64(0), 0, 0, np.int64(_SEE_SLOT_WRONG_TYPE)

    c57, angle_lo, angle_hi, z_lo, z_hi, rcyc = _relative_angles(
        mem, zp, observer, target
    )
    cyc = np.int64(_SEE_PROLOGUE) + rcyc + np.int64(_SEE_FOV)
    a = (c57 - 0x0A + (fov_width >> 1)) & 0xFF
    if a >= fov_width:  # $18B8 FOV gate
        return 1, 0, np.int64(0), 0, 0, cyc + np.int64(_SEE_FOV_REJECT)
    cyc += np.int64(_SEE_FOV_PASS)

    v_angle_obs = _rd(mem, _OVANGLE + observer)
    mem[_TARGETED_SLOT] = np.uint8(target)
    ox = _rd(mem, _OX + observer)
    oy = _rd(mem, _OY + observer)
    obs_zf = _rd(mem, _OZF + observer)
    obs_zh = _rd(mem, _OZH + observer)
    n_probes = 2 if expected_type == _T_ROBOT else 1
    if n_probes == 2:
        cyc += np.int64(_SEE_ROBOT)
    else:
        cyc += np.int64(_SEE_NOT_ROBOT)
    # $1904..$1914 runs twice whatever the type: the $1E counter always reaches 0.
    cyc += np.int64(2 * _SEE_BASE + _SEE_BASE_AGAIN + _SEE_BASE_LAST)
    for probe in range(n_probes):
        if n_probes == 2 and probe == 0:
            plo = z_lo
            phi = z_hi
            do_los = 0x80
        else:
            plo = (z_lo - 0xE0) & 0xFF
            phi = (z_hi - (1 if z_lo < 0xE0 else 0)) & 0xFF
            do_los = 0x00
        zp[0x80] = plo
        _va, vcyc = _vertical_angle(zp, phi, v_angle_obs)
        cyc += np.int64(_SEE_PROBE_FIXED) + vcyc + np.int64(_MARCH_ENTRY)
        vx_lo, vx_hi, vz_lo, vz_hi, vy_lo, vy_hi, s30, pcyc = _prep_vec_angle(
            angle_hi, angle_lo, zp[0x8B], zp[0x8A]
        )
        cyc += pcyc  # $1C54, priced from its own shift-adds
        c56 = (_rd(mem, 0x0C56) >> 1) & 0xFF  # $1CDF LSR $0C56
        cdd = (_rd(mem, 0x0CDD) >> 1) & 0xFF  # $1CE2 LSR $0CDD
        mem[0x0C56] = np.uint8(c56)
        mem[0x0CDD] = np.uint8(cdd)
        res = march(
            mem,
            vx_lo,
            vx_hi,
            vz_lo,
            vz_hi,
            vy_lo,
            vy_hi,
            s30,
            0,
            0x80,
            ox,
            0,
            obs_zf,
            obs_zh,
            0,
            0x80,
            oy,
            ox,
            oy,
            do_los,
            _rd(mem, _TARGETED_SLOT),
            c56,
            cdd,
            _MAX_STEPS,
        )
        los_ok = res[0] == LOS_CLEAR
        cyc += np.int64(res[15])
        mem[0x0C56] = np.uint8(res[12] & 0xFF)
        mem[0x0CDD] = np.uint8(res[13] & 0xFF)
        # $18F9-$1901: the four-rotate chained-carry cascade.
        c56v = _rd(mem, 0x0C56)
        reached = (c56v >> 7) & 1
        mem[0x0C56] = np.uint8(((c56v << 1) | (0 if los_ok else 1)) & 0xFF)
        c14 = _rd(mem, 0x0014)
        c14_out = c14 & 1
        mem[0x0014] = np.uint8(((reached << 7) | (c14 >> 1)) & 0xFF)
        cddv = _rd(mem, 0x0CDD)
        tree_flag = (cddv >> 7) & 1
        mem[0x0CDD] = np.uint8(((cddv << 1) | c14_out) & 0xFF)
        c76 = _rd(mem, 0x0C76)
        mem[0x0C76] = np.uint8(((tree_flag << 7) | (c76 >> 1)) & 0xFF)
    exposure = _rd(mem, 0x0014)
    full = 1 if exposure & 0x80 else 0
    tree_head = 1 if _rd(mem, 0x0C76) & 0x40 else 0
    return 1, 1, exposure, full, tree_head, cyc + np.int64(_SEE_TAIL)


@njit(cache=True, inline="always")
def _exposure_byte(in_slot, in_fov, exposure):
    """The ROM's object_exposure ($0014) from a can-see check."""
    if in_slot == 0 or in_fov == 0:
        return np.int64(0)
    return exposure


@njit(cache=True)
def _exposure_cycles(mem, clk):
    """$191F for the current board: the 8-slot walk plus prologue/epilogue."""
    player = _rd(mem, _PLAYER)
    total = charge(clk, 0x191F, np.int64(_EXPOSURE_FIXED))
    for x in range(7, -1, -1):
        last = np.int64(_EXPOSURE_LAST) if x == 0 else np.int64(0)
        if mem[_OFLAGS + x] & 0x80:
            total += charge(clk, 0x1925, np.int64(_EXPOSURE_EMPTY) - last)
            continue
        otype = _rd(mem, _OTYPE + x)
        if otype == _T_SENTRY:
            slot = np.int64(_EXPOSURE_SENTRY)
        elif otype == _T_SENTINEL:
            slot = np.int64(_EXPOSURE_SENTINEL)
        else:
            total += charge(clk, 0x1925, np.int64(_EXPOSURE_OTHER) - last)
            continue
        if _rd(mem, _TARGET + x) != player:
            total += charge(clk, 0x1925, slot - last)
            continue
        total += charge(clk, 0x1925, slot)
        if mem[_DRAIN_CD + x] == 0:
            total += charge(clk, 0x193A, np.int64(_EXPOSURE_TARGETS_PLAYER) - last)
            continue
        total += charge(clk, 0x193A, np.int64(_EXPOSURE_TARGETS_PLAYER))
        if mem[_TARGET_EXP + x] & 0x80:
            return total + charge(clk, 0x193F, np.int64(_EXPOSURE_DRAINING + 1))
        total += charge(clk, 0x193F, np.int64(_EXPOSURE_DRAINING) - last)
    return total


@njit(cache=True)
def _remove_object(mem, slot):
    """remove_object $1EEF: unlink the object and repair the tile it stood on."""
    tx = _rd(mem, _OX + slot)
    ty = _rd(mem, _OY + slot)
    flags = _rd(mem, _OFLAGS + slot)
    if flags >= 0x40:
        _set_tile_byte(mem, tx, ty, _OBJECT_TILE | (flags & 0x3F))
    else:
        _set_tile_byte(mem, tx, ty, (_rd(mem, _OZH + slot) << 4) & 0xFF)
    _wr(mem, _OFLAGS + slot, 0x80)


@njit(cache=True)
def _create_object(mem, otype, clk):
    """create_object $211D: (the highest empty slot typed `otype`, or -1; cycles)."""
    cost = charge(clk, 0x211D, np.int64(_CREATE_HEAD))
    for slot in range(_NUM_SLOTS - 1, -1, -1):
        if mem[_OFLAGS + slot] & 0x80:
            _wr(mem, _OTYPE + slot, otype)
            return slot, cost + charge(clk, 0x212C, np.int64(_CREATE_HIT))
        cost += charge(
            clk, 0x2122, np.int64(_CREATE_SLOT - (_CREATE_LAST if slot == 0 else 0))
        )
    return -1, cost + charge(clk, 0x212A, np.int64(_CREATE_NONE))


@njit(cache=True)
def _random_tile_coord(mem, clk):
    """$1272: (a prnd draw masked to 0..31, rejecting 31; cycles)."""
    cost = charge(clk, 0x1272, np.int64(_DRAW))
    while True:
        v = _prng_next(mem) & 0x1F
        if v != 0x1F:
            return v, cost
        cost += charge(clk, 0x1279, np.int64(_DRAW_REJECT))


@njit(cache=True)
def _put_object_in_tile(mem, slot, tx, ty):
    """put_object_in_tile $1EFF for a bare flat tile."""
    b = _tile_byte(mem, tx, ty)
    _wr(mem, _OX + slot, tx)
    _wr(mem, _OY + slot, ty)
    _wr(mem, _OFLAGS + slot, 0x00)
    _wr(mem, _OZF + slot, 0xE0)
    _wr(mem, _OZH + slot, (b >> 4) & 0xFF)
    _set_tile_byte(mem, tx, ty, _OBJECT_TILE | slot)
    _wr(mem, _OVANGLE + slot, 0xF5)
    _wr(mem, _OHANG + slot, (_prng_next(mem) & 0xF8) + 0x60)


@njit(cache=True)
def _put_object_in_random_tile_below_z(mem, slot, z, clk):
    """$1238: a random flat, empty tile no higher than `z`; after 256 misses the
    ceiling rises, and it fails once it reaches 12.  Returns (placed, cycles)."""
    attempts = 0
    cost = charge(clk, 0x1238, np.int64(_PLACE_HEAD))
    while True:
        attempts = (attempts - 1) & 0xFF
        if attempts == 0:  # $1242: 256 misses -> relax the height ceiling
            z = (z + 1) & 0xFF
            if z >= 0x0C:
                return False, cost + charge(clk, 0x1270, np.int64(_PLACE_GIVE_UP))
            cost += charge(clk, 0x1240, np.int64(_PLACE_WRAP))
        cost += charge(clk, 0x123E, np.int64(_PLACE_LAP))
        tx, dx = _random_tile_coord(mem, clk)
        ty, dy = _random_tile_coord(mem, clk)
        cost += dx + dy
        b = _tile_byte(mem, tx, ty)
        if b >= _OBJECT_TILE:  # tile already holds an object
            cost += charge(clk, 0x125B, np.int64(_PLACE_OCCUPIED))
            continue
        if b & 0x0F:  # not flat
            cost += charge(clk, 0x125F, np.int64(_PLACE_NOT_FLAT))
            continue
        if (b >> 4) >= z:  # too high
            cost += charge(clk, 0x1269, np.int64(_PLACE_TOO_HIGH))
            continue
        _put_object_in_tile(mem, slot, tx, ty)
        cost += charge(clk, 0x126B, np.int64(_PLACE_HIT))
        return True, cost + charge(clk, 0x1F16, np.int64(_PUT_IN_TILE))


@njit(cache=True)
def _status_bar_cycles(energy, clk):
    """$9508 plot_status_bar: 15-blocks, 3-blocks, the odd unit, then the padding."""
    fifteens = energy // 15
    rest = energy - fifteens * 15
    threes = rest // 3
    unit = rest - threes * 3
    total = charge(clk, 0x9508, np.int64(_STATUS_HEAD))
    total += charge(
        clk,
        0x9515,
        np.int64((fifteens + threes) * _STATUS_BLOCK + 2 * _STATUS_BLOCK_DONE),
    )
    if unit == 0:
        total += charge(clk, 0x9543, np.int64(_STATUS_UNIT_NONE))
    else:
        total += charge(clk, 0x9547, np.int64(_STATUS_UNIT))
    chars = 1 + 2 * (fifteens + threes + (1 if unit else 0))
    laps = _STATUS_PAD_END - chars
    total += charge(clk, 0x9551, np.int64(_STATUS_PAD * (laps if laps > 1 else 1) - 1))
    total += charge(clk, 0x955D, np.int64(_STATUS_MID))
    total += charge(clk, 0x9551, np.int64(_STATUS_PAD * _STATUS_PAD_LAPS - 1))
    return total + charge(clk, 0x956E, np.int64(_STATUS_TAIL))


@njit(cache=True, inline="always")
def _discharge_bank(mem, enemy):
    """increase_enemy_energy_to_discharge $1A4F."""
    _wr(mem, _DISCHARGE + enemy, _rd(mem, _DISCHARGE + enemy) + 1)


@njit(cache=True)
def _reduce_object_energy(mem, target, enemy, clk):
    """$1A08: drain `target`, returning (drained the player, cycles spent)."""
    head = charge(clk, 0x1A08, np.int64(_REDUCE_HEAD))
    if target == _rd(mem, _PLAYER):
        if _rd(mem, _ENERGY) == 0:  # kill_player $1A00
            _wr(mem, _DIED_DRAINING, _rd(mem, _DIED_DRAINING) | 0x80)
            return True, head + charge(clk, 0x1A0D, np.int64(_REDUCE_KILL))
        _wr(mem, _ENERGY, (_rd(mem, _ENERGY) - 1) & _ENERGY_MASK)
        _discharge_bank(mem, enemy)
        cost = head + charge(clk, 0x1A14, np.int64(_REDUCE_PLAYER))
        cost += charge(clk, 0x1A4F, np.int64(_REDUCE_BANK))
        cost += _status_bar_cycles(_rd(mem, _ENERGY), clk)
        cost += charge(clk, 0x1A1D, np.int64(_TUNE_DRAIN))
        _start_tune(mem, _SND_DRAIN)
        return True, cost
    cost = head + charge(clk, 0x1A0D, np.int64(_REDUCE_OBJECT))
    cost += charge(clk, 0x1A4F, np.int64(_REDUCE_BANK))
    otype = _rd(mem, _OTYPE + target)
    if otype == _T_ROBOT:
        _wr(mem, _DRAIN_CD + enemy, 0)  # $1A31
        _wr(mem, _OTYPE + target, _T_BOULDER)
        cost += charge(clk, 0x1A2D, np.int64(_REDUCE_ROBOT))
    elif otype == _T_TREE:
        stacked = _rd(mem, _OFLAGS + target) >= 0x40  # $1EFF: the tile byte restored
        _remove_object(mem, target)
        cost += charge(clk, 0x1EEF, np.int64(_REDUCE_TREE))
        cost += charge(
            clk, 0x1EEF, np.int64(_REMOVE_STACKED if stacked else _REMOVE_GROUND)
        )
    else:  # boulder -> tree
        _wr(mem, _OTYPE + target, _T_TREE)
        cost += charge(clk, 0x1A44, np.int64(_REDUCE_BOULDER))
    _discharge_bank(mem, enemy)
    return False, cost


@njit(cache=True)
def _consider_discharging_enemy_energy(mem, enemy, clk):
    """consider_discharging_enemy_energy $1A5D: return one banked unit to the
    landscape as a tree on a random flat tile.  Returns (discharged, cycles, slot)."""
    if mem[_DISCHARGE + enemy] == 0:  # $1A63: nothing to discharge
        return False, charge(clk, 0x1A5D, np.int64(_DISCHARGE_NONE)), -1
    cost = charge(clk, 0x1A65, np.int64(_DISCHARGE_CREATE))
    slot, ccost = _create_object(mem, _T_TREE, clk)  # $1A67
    cost += ccost
    if slot < 0:
        return False, cost, -1
    cost += charge(clk, 0x1A6A, np.int64(_DISCHARGE_PLACE))
    placed, pcost = _put_object_in_random_tile_below_z(
        mem, slot, _rd(mem, _BELOW_Z), clk
    )
    cost += pcost
    if not placed:  # $1A70: no tile found -> abandon
        return False, cost + charge(clk, 0x1A70, np.int64(_DISCHARGE_ABANDON)), -1
    _wr(mem, _DISCHARGE + enemy, _rd(mem, _DISCHARGE + enemy) - 1)  # $1A7A
    return True, cost + charge(clk, 0x1A72, np.int64(_DISCHARGE_DONE)), slot


@njit(cache=True)
def _do_hyperspace(mem):
    """do_hyperspace $2147: a synthoid on a random low tile, energy spent, player
    transferred; too little energy kills, and doing it from the platform wins."""
    idle = frame_clock(False)  # the hyperspace's cycles are not budgeted
    slot, _ccost = _create_object(mem, _T_ROBOT, idle)
    if slot < 0:
        return
    player = _rd(mem, _PLAYER)
    z = (_rd(mem, _OZH + player) + 1) & 0xFF
    placed, _pcost = _put_object_in_random_tile_below_z(mem, slot, z, idle)
    if not placed:
        _wr(mem, _OFLAGS + slot, _rd(mem, _OFLAGS + slot) | 0x80)  # $2159
        return
    if _rd(mem, _ENERGY) < _ROBOT_ENERGY:  # $215F: out of energy -> death
        _remove_object(mem, slot)
        _wr(mem, _HYPERSPACED, 0x80)
        return
    _wr(mem, _ENERGY, (_rd(mem, _ENERGY) - _ROBOT_ENERGY) & _ENERGY_MASK)
    on_platform = _rd(mem, _OX + player) == _rd(mem, _PLATFORM_X) and _rd(
        mem, _OY + player
    ) == _rd(mem, _PLATFORM_Y)
    if on_platform:  # $2187: hyperspacing from the platform completes the landscape
        _wr(mem, _COMPLETE, 0xC0)
    _wr(mem, _PLAYER, slot)


@njit(cache=True)
def _find_drainable_boulder_or_tree(mem, zp, enemy, budget, index, clk):
    """find_drainable_boulder_or_tree_on_stack $1AB0, resumable, one slot per unit.

    Returns (budget, index, drained slot / -1 exhausted / -2 suspended)."""
    while True:
        if index <= -2:  # this slot's $1887 is paid; only its write is outstanding
            x = -2 - index
            tb = _tile_byte(mem, _rd(mem, _OX + x), _rd(mem, _OY + x))
            y = tb & 0x3F
            index = x - 1
            _i, _f, _e, full, _t, _c = _can_see_object(
                mem, zp, enemy, y, _rd(mem, _OTYPE + y), _FOV_SCAN
            )
            if full:
                _wr(mem, _TARGETED_SLOT, y)
                return budget - charge(clk, 0x1AE6, np.int64(_TILE_SCAN_HIT)), index, y
            budget -= charge(
                clk,
                0x1AE6,
                np.int64(_TILE_SCAN_NEXT - (_TILE_SCAN_LAST if x == 0 else 0)),
            )
            continue
        if index < 0:  # $1AF2: the whole scan came up empty
            gone = charge(clk, 0x1AF2, np.int64(_TILE_SCAN_EXHAUSTED))
            return budget - gone, index, -1
        if budget <= 0:
            return budget, index, -2
        x = index
        index -= 1
        last = np.int64(_TILE_SCAN_LAST if x == 0 else 0)  # $1AF0 BPL not taken
        flags = _rd(mem, _OFLAGS + x)
        if flags & 0x80:  # empty slot
            budget -= charge(clk, 0x1AB2, np.int64(_TILE_SCAN_EMPTY) - last)
            continue
        if not (flags >= 0x40 or _rd(mem, _OTYPE + x) == _T_BOULDER):
            budget -= charge(clk, 0x1AB7, np.int64(_TILE_SCAN_OTHER) - last)
            continue
        if flags >= 0x40:
            budget -= charge(clk, 0x1AB9, np.int64(_TILE_SCAN_STACKED))
        else:
            budget -= charge(clk, 0x1ABE, np.int64(_TILE_SCAN_LOOSE))
        budget -= charge(clk, 0x1AC2, np.int64(_TILE_SCAN_TILE))
        tb = _tile_byte(mem, _rd(mem, _OX + x), _rd(mem, _OY + x))
        if tb < _OBJECT_TILE:
            budget -= charge(clk, 0x1AD3, np.int64(_TILE_SCAN_NO_TILE) - last)
            continue
        y = tb & 0x3F  # topmost object of the tile
        otype = _rd(mem, _OTYPE + y)
        budget -= charge(clk, 0x1AD3, np.int64(_TILE_SCAN_TOP))
        if otype != _T_TREE and otype != _T_BOULDER:
            budget -= charge(clk, 0x1ADD, np.int64(_TILE_SCAN_WRONG_TOP) - last)
            continue
        _in_slot, _in_fov, _exp, full, _th, scost = _can_see_object(
            mem, zp, enemy, y, otype, _FOV_SCAN
        )
        if otype == _T_TREE:
            budget -= charge(clk, 0x1ADD, np.int64(_TILE_SCAN_SEE))
        else:
            budget -= charge(clk, 0x1ADF, np.int64(_TILE_SCAN_SEE_BOULDER))
        budget -= charge(clk, 0x1887, scost)
        if not full:
            budget -= charge(clk, 0x1AE6, np.int64(_TILE_SCAN_NEXT) - last)
            continue
        if budget <= 0:  # the ROM has not reached $1AEA yet
            return budget, -2 - x, -2
        _wr(mem, _TARGETED_SLOT, y)
        return budget - charge(clk, 0x1AE6, np.int64(_TILE_SCAN_HIT)), index, y


@njit(cache=True)
def _initialise_enemy_meanie_variables(mem, enemy):
    """initialise_enemy_meanie_variables $196A: (re)arm an enemy's meanie hunt."""
    _wr(mem, _M_OBJECT + enemy, 0x80)
    _wr(mem, _M_FAILED + enemy, 0x80)
    _wr(mem, _M_SCANS + enemy, 0)
    _wr(mem, _M_SEARCH + enemy, 0x40)


@njit(cache=True)
def _consider_creating_meanie(mem, zp, enemy, budget, index, clk):
    """consider_creating_meanie $197D plus target_object's $1852 tail, resumable.

    One search-counter step is one unit.  Returns (budget, stage, index)."""
    player = _rd(mem, _TARGET + enemy)
    while True:
        if index <= -2:  # this slot's $1887 is paid; only its write is outstanding
            slot = _rd(mem, _M_SEARCH + enemy)
            index = 0
            _i, _f, _e, full, _t, _c = _can_see_object(
                mem, zp, enemy, slot, _T_TREE, _FOV_CREATE_MEANIE
            )
            if full:
                _wr(mem, _M_OBJECT + enemy, slot)  # $19E1
                _wr(mem, _OTYPE + slot, _T_MEANIE)
                _wr(mem, _UPD_CD + enemy, _UPD_CD_MEANIE_MADE)
                return budget, _BODY_DONE, 0
            continue
        if budget <= 0:
            return budget, _BODY_MAKE_MEANIE, index
        sc = _rd(mem, _M_SEARCH + enemy)
        if sc == 0:  # $198D: scanned everything -> no meanie this pass
            budget -= charge(clk, 0x198F, np.int64(_MEANIE_SCAN_DONE))
            _wr(mem, _M_SCANS + enemy, _rd(mem, _M_SCANS + enemy) + 1)
            _wr(mem, _M_FAILED + enemy, player)
            if _rd(mem, _M_SCANS + enemy) >= _MEANIE_MAX_ATTEMPTS:
                _wr(mem, _DRAIN_CD + enemy, 0)  # give up on this player
            else:
                _wr(mem, _CONSIDERING + enemy, 0x80)  # keep trying next time
            return budget, _BODY_DONE, 0
        _wr(mem, _M_SEARCH + enemy, sc - 1)
        slot = sc - 1  # $199B DEY
        if mem[_OFLAGS + slot] & 0x80:
            budget -= charge(clk, 0x198F, np.int64(_MEANIE_SCAN_SLOT))
            continue
        if mem[_OTYPE + slot] != _T_TREE:
            budget -= charge(clk, 0x19AA, np.int64(_MEANIE_SCAN_OTHER))
            continue
        budget -= charge(clk, 0x19AA, np.int64(_MEANIE_SCAN_OTHER))
        budget -= charge(clk, 0x19B1, np.int64(_MEANIE_SCAN_DX))
        dx = (_rd(mem, _OX + player) - _rd(mem, _OX + slot)) & 0xFF
        if dx >= 0x80:
            dx = 0x100 - dx  # $19B5 abs
        if dx >= 0x0A:
            continue
        budget -= charge(clk, 0x19C7, np.int64(_MEANIE_SCAN_DY))
        dy = (_rd(mem, _OY + player) - _rd(mem, _OY + slot)) & 0xFF
        if dy >= 0x80:
            dy = 0x100 - dy
        if dy >= 0x0A:
            continue
        _in_slot, _in_fov, _exp, full, _th, scost = _can_see_object(
            mem, zp, enemy, slot, _T_TREE, _FOV_CREATE_MEANIE
        )
        budget -= charge(clk, 0x19D9, np.int64(_MEANIE_SCAN_SEE))
        budget -= charge(clk, 0x1887, scost)
        if not full:
            continue
        if budget <= 0:  # the ROM has not reached $19E1 yet
            return budget, _BODY_MAKE_MEANIE, -2 - slot
        _wr(mem, _M_OBJECT + enemy, slot)  # $19E1
        _wr(mem, _OTYPE + slot, _T_MEANIE)
        _wr(mem, _UPD_CD + enemy, _UPD_CD_MEANIE_MADE)
        return budget, _BODY_DONE, 0


@njit(cache=True)
def _remove_meanie(mem, enemy):
    """remove_meanie $1754: turn the meanie back into a tree."""
    meanie = _rd(mem, _M_OBJECT + enemy)
    _wr(mem, _M_OBJECT + enemy, 0x80)
    _wr(mem, _OTYPE + meanie, _T_TREE)


@njit(cache=True)
def _remove_meanie_and_reset_enemy(mem, enemy):
    """remove_meanie_and_reset_enemy $174F: also clear the draining cooldown."""
    _wr(mem, _DRAIN_CD + enemy, 0)
    _remove_meanie(mem, enemy)


@njit(cache=True)
def _update_meanie(mem, zp, enemy, budget, index, clk):
    """update_meanie $16F2, resumable: rotate toward the player, then hyperspace it.

    Its one $1887 call is the unit.  Returns (budget, stage, index)."""
    meanie = _rd(mem, _M_OBJECT + enemy)
    target = _rd(mem, _TARGET + enemy)
    if mem[_OFLAGS + target] & 0x80:  # $16F7: the object the player was in is gone
        _remove_meanie_and_reset_enemy(mem, enemy)
        return budget, _BODY_DONE, 0
    in_slot, in_fov, exposure, _full, _th, cost = _can_see_object(
        mem, zp, meanie, target, _T_ROBOT, _FOV_SCAN
    )
    if index >= 0:  # not yet charged
        budget -= charge(clk, 0x1887, cost)
        if budget <= 0:
            return budget, _BODY_MEANIE, -2
    if in_fov == 0:  # $1706: not yet looking at the player -> rotate
        c57 = _relative_angles(mem, zp, meanie, target)[0]
        step = _MEANIE_ROTATE_STEP
        if c57 & 0x80:
            step = 0x100 - _MEANIE_ROTATE_STEP
        _wr(mem, _OHANG + meanie, _rd(mem, _OHANG + meanie) + step)
        _wr(mem, _UPD_CD + enemy, _UPD_CD_MEANIE_ROTATE)
        _start_tune(mem, _SND_MEANIE)  # $1743 JSR $3470
        # $1755 JMP $187B: a meanie's turn redraws it too
        budget -= charge(clk, 0x1728, np.int64(_MEANIE_ROTATE))
        redraw = charge(clk, 0x1F9F, _update_object_on_screen(mem, zp, meanie)[0])
        return budget - redraw, _BODY_DONE, 0
    if target != _rd(mem, _PLAYER):  # $1708: player transferred out of the object
        _remove_meanie_and_reset_enemy(mem, enemy)
        return budget, _BODY_DONE, 0
    if _exposure_byte(in_slot, in_fov, exposure) == 0:  # $170E
        _remove_meanie(mem, enemy)
        return budget, _BODY_DONE, 0
    _do_hyperspace(mem)  # $1710: forced hyperspace
    return budget, _BODY_DONE, 0


@njit(cache=True)
def _target_object(mem, zp, enemy, target, exposure, clk):
    """target_object $1825 up to its $184D branch: record the target, drain it when
    the timer expires.  Returns (next stage, the cycles $1825's own line spent)."""
    _wr(mem, _TARGET + enemy, target)
    _wr(mem, _TARGET_EXP + enemy, exposure)
    cost = charge(clk, 0x1825, np.int64(_TARGET_HEAD))
    cd = _rd(mem, _DRAIN_CD + enemy)
    if cd < 0x01:  # first sight -> arm the drain timer
        _wr(mem, _DRAIN_CD + enemy, _DRAIN_CD_RELOAD)
        return _BODY_DONE, cost + charge(clk, 0x1833, np.int64(_TARGET_FIRST))
    if cd != 0x01:  # still counting down
        return _BODY_DONE, cost + charge(clk, 0x1833, np.int64(_TARGET_WAIT))
    cost += charge(clk, 0x183D, np.int64(_TARGET_DUE))
    if exposure & 0x80:  # fully visible -> drain
        _wr(mem, _TARGETED_SLOT, target)
        killed = target == _rd(mem, _PLAYER) and _rd(mem, _ENERGY) == 0
        cost += charge(clk, 0x1843, np.int64(_TARGET_DRAIN))
        cost += _reduce_object_energy(mem, target, enemy, clk)[1]
        if killed:  # kill_player $1A00 unwinds the stack
            return _BODY_DONE, cost
        _wr(mem, _UPD_CD + enemy, _UPD_CD_DRAIN)
        if target == _rd(mem, _PLAYER):  # $184D: a drained player skips the redraw
            return _BODY_DONE, cost + charge(
                clk, 0x1884, np.int64(_TARGET_DRAIN_PLAYER)
            )
        cost += charge(clk, 0x184D, np.int64(_TARGET_DRAIN_OBJ))
        cost += charge(clk, 0x1876, np.int64(_BODY_TAIL))
        redraw = charge(clk, 0x1F9F, _update_object_on_screen(mem, zp, target)[0])
        return _BODY_DONE, cost + redraw
    # $184D: only the head -> hunt a tree to convert
    return _BODY_MAKE_MEANIE, cost + charge(clk, 0x1841, np.int64(_TARGET_MEANIE))


@njit(cache=True)
def _rotate_enemy(mem, enemy):
    """rotate_enemy $1805: add the per-enemy step to the facing, reload $C8."""
    _wr(mem, _OHANG + enemy, _rd(mem, _OHANG + enemy) + _rd(mem, _ROT_SPEED + enemy))
    _wr(mem, _ROT_CD + enemy, _ROT_CD_RELOAD)
    _start_tune(mem, _SND_ROTATE)  # $180F JSR $3470
    _initialise_enemy_meanie_variables(mem, enemy)  # $1818


@njit(cache=True)
def _scan_for_robot(mem, zp, enemy, budget, index, partial, clk):
    """find_drainable_robot_loop $17B2, resumable, one slot per unit."""
    player = _rd(mem, _PLAYER)
    while True:
        if index <= -2:  # this slot's $1887 is paid; only its write is outstanding
            y = -2 - index
            in_slot, in_fov, exp_raw, _f, _t, _c = _can_see_object(
                mem, zp, enemy, y, _T_ROBOT, _FOV_SCAN
            )
            stage, cost = _target_object(
                mem, zp, enemy, y, _exposure_byte(in_slot, in_fov, exp_raw), clk
            )
            return budget - cost, stage, 0, -1
        if index < 0:  # $17CB: the scan is exhausted
            if partial >= 0:
                return budget, _BODY_PARTIAL, 0, partial
            _wr(mem, _DRAIN_CD + enemy, 0)  # $17E0
            budget -= charge(clk, 0x17CD, np.int64(_SCAN_END))
            budget -= charge(clk, 0x17E0, np.int64(_TREE_CALL))
            budget -= charge(clk, 0x1AB0, np.int64(_TILE_SCAN_ENTRY))
            return budget, _BODY_TREE, _NUM_SLOTS - 1, -1
        if budget <= 0:
            return budget, _BODY_SCAN, index, partial
        y = index
        index -= 1
        last = np.int64(_SCAN_LAST if y == 0 else 0)  # $17CB BPL not taken
        in_slot, in_fov, exp_raw, _full, tree_head, scost = _can_see_object(
            mem, zp, enemy, y, _T_ROBOT, _FOV_SCAN
        )
        budget -= charge(clk, 0x1887, scost)
        if tree_head:  # $17B7: a tree hides this robot's head
            budget -= charge(clk, 0x17B2, np.int64(_SCAN_SLOT_HIDDEN) - last)
            continue
        exposure = _exposure_byte(in_slot, in_fov, exp_raw)
        if exposure == 0:  # $17BE: not visible at all
            budget -= charge(clk, 0x17B2, np.int64(_SCAN_SLOT_UNSEEN) - last)
            continue
        if exposure & 0x80:  # $17BA: fully visible -> drain target
            budget -= charge(clk, 0x17B2, np.int64(_SCAN_SLOT_FULL))
            if budget <= 0:  # the ROM has not reached $1825 yet
                return budget, _BODY_SCAN, -2 - y, partial
            stage, cost = _target_object(mem, zp, enemy, y, exposure, clk)
            return budget - cost, stage, 0, -1
        if y == player:  # $17C0: head only -> meanie candidate
            partial = y
            budget -= charge(clk, 0x17B2, np.int64(_SCAN_SLOT_PARTIAL) - last)
        else:
            budget -= charge(clk, 0x17B2, np.int64(_SCAN_SLOT_OTHER) - last)


@njit(cache=True)
def _consider_enemy_state(mem, zp, enemy, budget, stage, index, partial, clk):
    """consider_enemy_state $16E6, resumable at its own write points.

    Returns (budget, stage, index, partial); stage BODY_DONE means it reached RTS."""
    while True:
        if stage == _BODY_DONE:
            return budget, _BODY_DONE, 0, -1
        if stage == _BODY_ENTRY:
            if mem[_UPD_CD + enemy] >= _COOLDOWN_STICK:
                gate = charge(clk, 0x16E6, np.int64(_UPDATE_GATE_CLOSED))
                return budget - gate, _BODY_DONE, 0, -1
            budget -= charge(clk, 0x16E6, np.int64(_CONSIDER_ENTRY))
            _wr(mem, _UPD_CD + enemy, _UPD_CD_SCAN)  # $16ED
            _wr(mem, _FOV_WIDTH, _FOV_SCAN)  # $16F0
            if not (mem[_M_OBJECT + enemy] & 0x80):  # $16EA: already owns a meanie
                budget -= charge(clk, 0x16F7, np.int64(_CONSIDER_MEANIE))
                stage = _BODY_MEANIE
                index = 0
                continue
            budget -= charge(clk, 0x16FC, np.int64(_CONSIDER_NO_MEANIE))
            budget -= charge(clk, 0x1773, np.int64(_DISCHARGE_CALL))
            stage = _BODY_DISCHARGE
            index = 0
            continue

        if index > -1 and budget < 1:  # nothing charged is awaiting its write
            return budget, stage, index, partial

        if stage == _BODY_MEANIE:
            budget, stage, index = _update_meanie(mem, zp, enemy, budget, index, clk)
            continue

        if stage == _BODY_DISCHARGE:
            discharged, dcost, slot = _consider_discharging_enemy_energy(
                mem, enemy, clk
            )
            budget -= dcost
            if discharged:  # $177A: the tail redraws, then the update is over
                budget -= charge(clk, 0x1778, np.int64(_DISCHARGED))
                budget -= charge(clk, 0x1876, np.int64(_BODY_TAIL))
                budget -= charge(
                    clk, 0x1F9F, _update_object_on_screen(mem, zp, slot)[0]
                )
                return budget, _BODY_DONE, 0, -1
            budget -= charge(clk, 0x177D, np.int64(_NO_DISCHARGE))
            if mem[_CONSIDERING + enemy] & 0x80:  # $177F: mid meanie-hunt
                budget -= charge(clk, 0x1784, np.int64(_HUNT_CALL))
                budget -= charge(clk, 0x1AB0, np.int64(_TILE_SCAN_ENTRY))
                stage = _BODY_HUNT
                index = _NUM_SLOTS - 1
                continue
            budget -= charge(clk, 0x1782, np.int64(_HUNT_CLEAR))
            stage = _BODY_HELD
            index = 0
            continue

        if stage == _BODY_HUNT:
            budget, index, tb = _find_drainable_boulder_or_tree(
                mem, zp, enemy, budget, index, clk
            )
            if tb == -2:  # suspended mid-scan
                return budget, stage, index, partial
            if tb >= 0:
                _wr(mem, _M_SEARCH + enemy, 0x40)  # $178B
                budget -= charge(clk, 0x178B, np.int64(_HUNT_HIT))
                budget -= charge(clk, 0x17EA, np.int64(_DRAIN_CALL))
                budget -= _reduce_object_energy(mem, tb, enemy, clk)[1]  # $17EA
                _wr(mem, _UPD_CD + enemy, _UPD_CD_DRAIN)
                budget -= charge(clk, 0x17ED, np.int64(_DRAIN_TAIL))
                budget -= charge(clk, 0x1876, np.int64(_BODY_TAIL))
                budget -= charge(clk, 0x1F9F, _update_object_on_screen(mem, zp, tb)[0])
                return budget, _BODY_DONE, 0, -1
            budget -= charge(clk, 0x1787, np.int64(_HUNT_MISS))
            _wr(mem, _CONSIDERING + enemy, _rd(mem, _CONSIDERING + enemy) >> 1)
            stage = _BODY_HELD
            index = 0
            continue

        if stage == _BODY_HELD:
            if mem[_DRAIN_CD + enemy] == 0:
                budget -= charge(clk, 0x1795, np.int64(_HELD_NONE))
                budget -= charge(clk, 0x17AC, np.int64(_SCAN_INIT))
                stage = _BODY_SCAN
                index = _NUM_SLOTS - 1
                partial = -1
                continue
            held = _rd(mem, _TARGET + enemy)  # $178C: re-check a held target
            in_slot, in_fov, exp_raw, _full, _th, scost = _can_see_object(
                mem, zp, enemy, held, _T_ROBOT, _FOV_SCAN
            )
            if index >= 0:
                budget -= charge(clk, 0x179A, np.int64(_HELD_CALL))
                budget -= charge(clk, 0x1887, scost)
                if budget <= 0:
                    return budget, stage, -2, partial
            exposure = _exposure_byte(in_slot, in_fov, exp_raw)
            if exposure != 0:
                budget -= charge(clk, 0x17A6, np.int64(_HELD_KEPT))
                stage, cost = _target_object(mem, zp, enemy, held, exposure, clk)
                budget -= cost
                index = 0
                continue
            _wr(mem, _DRAIN_CD + enemy, 0)  # target lost
            budget -= charge(clk, 0x17A2, np.int64(_HELD_LOST))
            budget -= charge(clk, 0x17AC, np.int64(_SCAN_INIT))
            stage = _BODY_SCAN
            index = _NUM_SLOTS - 1
            partial = -1
            continue

        if stage == _BODY_SCAN:
            budget, stage, index, partial = _scan_for_robot(
                mem, zp, enemy, budget, index, partial, clk
            )
            continue

        if stage == _BODY_PARTIAL:
            budget -= charge(clk, 0x17D1, np.int64(_SCAN_END_PARTIAL))
            if partial != _rd(mem, _M_FAILED + enemy):  # $17C4
                _initialise_enemy_meanie_variables(mem, enemy)
                budget -= charge(clk, 0x17D7, np.int64(_PARTIAL_ARM))
                stage, cost = _target_object(mem, zp, enemy, partial, 0x40, clk)
                budget -= cost
                index = 0
                partial = -1
                continue
            _wr(mem, _DRAIN_CD + enemy, 0)  # $17E0
            budget -= charge(clk, 0x17D5, np.int64(_PARTIAL_KNOWN))
            budget -= charge(clk, 0x17E0, np.int64(_TREE_CALL))
            budget -= charge(clk, 0x1AB0, np.int64(_TILE_SCAN_ENTRY))
            stage = _BODY_TREE
            index = _NUM_SLOTS - 1
            partial = -1
            continue

        if stage == _BODY_TREE:
            budget, index, tb = _find_drainable_boulder_or_tree(
                mem, zp, enemy, budget, index, clk
            )
            if tb == -2:
                return budget, stage, index, partial
            if tb >= 0:
                _wr(mem, _TARGETED_SLOT, tb)
                budget -= charge(clk, 0x17E8, np.int64(_TREE_HIT))
                budget -= charge(clk, 0x17EA, np.int64(_DRAIN_CALL))
                budget -= _reduce_object_energy(mem, tb, enemy, clk)[1]
                _wr(mem, _UPD_CD + enemy, _UPD_CD_DRAIN)
                budget -= charge(clk, 0x17ED, np.int64(_DRAIN_TAIL))
                budget -= charge(clk, 0x1876, np.int64(_BODY_TAIL))
                budget -= charge(clk, 0x1F9F, _update_object_on_screen(mem, zp, tb)[0])
                return budget, _BODY_DONE, 0, -1
            budget -= charge(clk, 0x17E8, np.int64(_TREE_NONE))
            stage = _BODY_ROTATE
            index = 0
            continue

        if stage == _BODY_ROTATE:
            if mem[_ROT_CD + enemy] < _COOLDOWN_STICK:  # $17F9 no_drain
                _rotate_enemy(mem, enemy)
                budget -= charge(clk, 0x17F9, np.int64(_ROTATE_GATE))
                budget -= charge(clk, 0x1805, np.int64(_ROTATE))
                budget -= charge(
                    clk, 0x1F9F, _update_object_on_screen(mem, zp, enemy)[0]
                )
                return budget, _BODY_DONE, 0, -1
            held_cd = charge(clk, 0x1802, np.int64(_ROTATE_GATE_HELD))
            return budget - held_cd, _BODY_DONE, 0, -1

        budget, stage, index = _consider_creating_meanie(
            mem, zp, enemy, budget, index, clk
        )


@njit(cache=True)
def _tick_cooldowns(mem, clk):
    """update_enemy_cooldowns $1317: the 1-in-3 gate, then every cooldown >= 2.

    Returns its cycles; the 24-byte walk is 6 dearer per byte that decrements."""
    if mem[_GATE] != 0:
        _wr(mem, _GATE, _rd(mem, _GATE) - 1)
        return charge(clk, 0x1317, np.int64(_COOLDOWN_TICK_GATE))
    cost = charge(clk, 0x131C, np.int64(_COOLDOWN_TICK_WALK - _COOLDOWN_TICK_LAST))
    for addr in range(_DRAIN_CD, _UPD_CD + 8):
        if mem[addr] >= _COOLDOWN_STICK:
            mem[addr] = np.uint8(mem[addr] - 1)
            cost += charge(clk, 0x1325, np.int64(_COOLDOWN_TICK_BYTE_DEC))
        else:
            cost += charge(clk, 0x131E, np.int64(_COOLDOWN_TICK_BYTE_STICK))
    _wr(mem, _GATE, 2)
    return cost


@njit(cache=True)
def _dispatch_cycles(mem, clk):
    """$16B5..$16E6: the type dispatch and the $16CC absorbed test, which write nothing."""
    otype = _rd(mem, _OTYPE + _rd(mem, _CURSOR))
    if otype == _T_SENTINEL:
        return charge(clk, 0x16B5, np.int64(_UPDATE_DISPATCH_SENTINEL))
    if otype == _T_SENTRY:
        return charge(clk, 0x16B5, np.int64(_UPDATE_DISPATCH_SENTRY))
    return charge(clk, 0x16B5, np.int64(_UPDATE_NOT_ENEMY))


@njit(cache=True)
def _update_body(mem, zp, budget, stage, index, partial, clk):
    """$16E6..$16D6: every state write $16B5 makes before the prnd, resumable."""
    x = _rd(mem, _CURSOR)
    otype = _rd(mem, _OTYPE + x)
    if otype != _T_SENTRY and otype != _T_SENTINEL:  # $16BB
        return budget, _BODY_DONE, 0, -1
    if mem[_OFLAGS + x] & 0x80:  # $16CE: an absorbed slot discharges its bank only
        budget -= charge(clk, 0x16CC, np.int64(_UPDATE_ABSORBED))
        return (
            budget - _consider_discharging_enemy_energy(mem, x, clk)[1],
            _BODY_DONE,
            0,
            -1,
        )
    return _consider_enemy_state(mem, zp, x, budget, stage, index, partial, clk)


@njit(cache=True)
def _update_cursor(mem, clk):
    """$16D9: store the advanced prnd stream and decrement the cursor (7->0 wrap)."""
    x = _rd(mem, _CURSOR)
    _prng_next(mem)
    _wr(mem, _CURSOR, (x - 1) if x > 0 else 7)
    if x == 0:
        return charge(clk, 0x16D9, np.int64(_UPDATE_CURSOR_WRAP))
    return charge(clk, 0x16D9, np.int64(_UPDATE_CURSOR))


_UNBOUNDED = np.int64(1) << 40  # a budget no single $16E6 can outrun


@njit(cache=True)
def _update_enemies(mem, zp):
    """update_enemies $16B5 whole: dispatch, body, prnd and cursor."""
    clk = frame_clock(False)  # the isolated round places no badline
    cost = _dispatch_cycles(mem, clk)
    left = _update_body(mem, zp, _UNBOUNDED, _BODY_ENTRY, 0, -1, clk)[0]
    cost += _UNBOUNDED - left
    return cost + np.int64(_UPDATE_PRND) + _update_cursor(mem, clk)


@njit(cache=True)
def _cooldown_frame(mem, clk):
    """$130C: the per-frame Bresenham gate on update_enemy_cooldowns, and its cycles."""
    if mem[_NOT_ACTED] & 0x80:  # $965C BMI $9669: no $130C and no $1635 either
        return charge(clk, 0x9659, np.int64(_IRQ_GATE_SHUT))
    cost = charge(clk, 0x965E, np.int64(_IRQ_GATE_OPEN))
    acc = _rd(mem, _BRESENHAM) + _BRESENHAM_STEP
    _wr(mem, _BRESENHAM, acc)
    if acc > 0xFF:  # $1315 BCC skip
        return cost + _tick_cooldowns(mem, clk)
    return cost + charge(clk, 0x130C, np.int64(_COOLDOWN_TICK_NO_CARRY))


@njit(cache=True)
def _sound_frame(mem, clk):
    """$8ED1: the three-voice note tick the raster IRQ runs every frame, and its cycles."""
    cyc = charge(clk, 0x963D, np.int64(_SOUND_TICK_FIXED))
    for x in range(2, -1, -1):
        cyc += charge(clk, 0x8ED3, np.int64(_SOUND_VOICE_READ))
        note = _rd(mem, _SND_NOTE + x)
        gate_due = note == 0
        if gate_due:
            cyc += charge(clk, 0x8ED6, np.int64(_SOUND_VOICE_SPENT))
        elif note & 0x80:
            cyc += charge(clk, 0x8ED6, np.int64(_SOUND_VOICE_OFF))
        else:
            note -= 1
            _wr(mem, _SND_NOTE + x, note)
            cyc += charge(clk, 0x8ED8, np.int64(_SOUND_VOICE_TICK))
            if note:
                cyc += charge(clk, 0x8EDD, np.int64(_SOUND_VOICE_MORE))
            else:
                _wr(mem, _SND_GATE + x, _rd(mem, _SND_LENGTH + x))
                cyc += charge(clk, 0x8EDF, np.int64(_SOUND_VOICE_NOTE))
                gate_due = True
        if gate_due:
            cyc += charge(clk, 0x8EEE, np.int64(_SOUND_GATE_READ))
            gate = _rd(mem, _SND_GATE + x)
            if gate == 0:
                cyc += charge(clk, 0x8EF1, np.int64(_SOUND_GATE_OFF))
            else:
                gate -= 1
                _wr(mem, _SND_GATE + x, gate)
                cyc += charge(clk, 0x8EF3, np.int64(_SOUND_GATE_TICK))
                if gate:
                    cyc += charge(clk, 0x8EF6, np.int64(_SOUND_GATE_MORE))
                else:
                    _wr(mem, _SND_FLAG + x, 0x80)
                    cyc += charge(clk, 0x8EF8, np.int64(_SOUND_GATE_END))
        cyc += charge(
            clk, 0x8F08, np.int64(_SOUND_VOICE_NEXT if x else _SOUND_VOICE_LAST)
        )
    return cyc


@njit(cache=True)
def _advance(mem, zp, n_frames, plotting, residual, phase, stage, index, partial):
    """The frame loop: the raster cooldown tick, then the foreground's cycle budget.

    Returns the sub-pass resume point the last frame stopped at: the cycles it
    overspent, which segment of the pass owes them, and where inside $16E6."""
    for done in range(n_frames):
        clk = frame_clock(True)  # the raster IRQ pins the frame's phase every frame
        irq = charge(clk, 0x95E9, np.int64(_IRQ_BODY)) - np.int64(_IRQ_BODY)
        irq += _sound_frame(mem, clk) + _cooldown_frame(mem, clk)
        if plotting:
            continue
        budget = residual + np.int64(_FOREGROUND_CYCLES - _STEAL_CEILING) - irq
        while budget > 0:
            if phase == 0:
                budget -= charge(clk, 0x1289, np.int64(_PASS_HEAD))
                budget -= _dispatch_cycles(mem, clk)
                phase = 1
                if budget <= 0:
                    break
            if phase == 1:
                budget, stage, index, partial = _update_body(
                    mem, zp, budget, stage, index, partial, clk
                )
                if stage != _BODY_DONE:  # the frame ran out INSIDE $16E6
                    break
                stage, index, partial = _BODY_ENTRY, 0, -1
                budget -= charge(clk, 0x16D6, np.int64(_UPDATE_PRND))
                phase = 2
                if budget <= 0 or zp[ZP_REPLOT] != 0:
                    break
            budget -= _update_cursor(mem, clk)
            budget -= charge(clk, 0x12A2, np.int64(_PASS_TAIL))
            budget -= _exposure_cycles(mem, clk)
            phase = 0
        residual = budget
        if zp[ZP_REPLOT] != 0:  # $1FFC: stop so the caller can price the replot
            return residual, phase, stage, index, partial, n_frames - done - 1
    return residual, phase, stage, index, partial, 0


def advance_frames(mem, n_frames, plotting, residual, phase, stage, index, partial):
    """Advance ``n_frames`` video frames on the caller's 64 KB ``bytearray``, carrying
    the sub-pass resume point (cycle residual, phase, $16E6 stage) in and out.

    Returns the resume point plus (frames left, replot target, replot left column):
    an on-screen $1F9F stops the run so the caller prices its $1FFC strip replot."""
    view = np.frombuffer(mem, dtype=np.uint8)
    zp = np.zeros(ZP_N, dtype=np.int64)
    res, ph, st, ix, pa, left_frames = _advance(
        view,
        zp,
        int(n_frames),
        bool(plotting),
        int(residual),
        int(phase),
        int(stage),
        int(index),
        int(partial),
    )
    target = int(zp[ZP_REPLOT]) - 1
    return (
        int(res),
        int(ph),
        int(st),
        int(ix),
        int(pa),
        int(left_frames),
        target,
        int(zp[ZP_REPLOT_LEFT]),
        int(zp[ZP_REPLOT_COLS]),
    )
