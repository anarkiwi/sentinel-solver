"""Per-verb settle frames, composed from the ROM's own settle paths.

Create/absorb settle through $12D0 -> $1B18 -> $12EC ($1F9F strip replot with the
$86A5 dither) and the pass tail; a transfer or u-turn starts its tune ($1B84), sets
$0C63 and settles through the $357D redraw with the tune wait overlapped.
"""

import numpy as np

from sentinel import memmap as mm
from sentinel import passcost, projector
from sentinel.passcost import PAL_FRAME_CYCLES

SETTLE_BODY_FLAT = 356  # $95E9..RTI, $0CE4 bit7 set, $130C Bresenham no-carry
SETTLE_BODY_CARRY = 704  # same, carry frame: the $131E cooldown walk + $1635
_BRES = mm.COOLDOWN_BRESENHAM_STEP / 256.0
SETTLE_BODY = SETTLE_BODY_FLAT + _BRES * (SETTLE_BODY_CARRY - SETTLE_BODY_FLAT)
SETTLE_STEAL = 1069.0  # badline steal of a store-heavy plot frame
SETTLE_FG = PAL_FRAME_CYCLES - passcost.SHORT_IRQ_FRAME - SETTLE_BODY - SETTLE_STEAL

BRACKET_FRAMES = 3.0  # tap_action: idle scan + press scan + the $130B=2 scan re-arm

REDRAW_HEAD = 13_249  # $357D..$35BA: $351C, $8643, the $3592 zeroing, sprites off
FILL_1090 = 108_935  # $35C0..$35C3, constant
STATUS_9508 = 1_946  # $35C6..$35C9
STATUS_98B2 = 130_553  # $35C9..$35CC
BUF_WAIT_HEAD = 75  # $35CC..$35D5: $2993 A=#$02 and the wait entry
EXIT_TAIL = 13_014  # $35DD..$363D: $114D, $33ED, $106C back to the loop
HEAD_BASE = 4_600  # $12D0 dispatch + handler bookkeeping, less the fire ray

TUNE_FRAMES = {0x19: 96.0, 0x28: 24.0}  # $AB50 tune-table note-holds, decoded


def _frames(cycles):
    return cycles / SETTLE_FG


def uturn_from_cost(cost):
    """:func:`uturn_settle_frames` from the ``render_cost`` of the flipped view, elementwise
    over an array of costs."""
    fg = (
        HEAD_BASE
        + REDRAW_HEAD
        + FILL_1090
        + projector.FRAME_CYCLES * cost
        + STATUS_9508
        + STATUS_98B2
        + BUF_WAIT_HEAD
        + EXIT_TAIL
    )
    return BRACKET_FRAMES + np.maximum(_frames(fg), TUNE_FRAMES[0x28])


def uturn_view(state, observer=None):
    """The flipped view :func:`uturn_settle_frames` prices: ``(observer, h, v)``."""
    obs = state.player if observer is None else observer
    return obs, (int(state.obj_h_angle[obs]) ^ 0x80) & 0xFF, int(state.obj_v_angle[obs])


def uturn_settle_frames(state, observer=None):
    """$1B2F: bearing ^ $80, tune #$28, then the $357D redraw with $245B/$3700
    SKIPPED ($1B2F's ASL leaves $0C51 bit7 set, so $35B5 branches to $35C0);
    no fire ray ($1B18 dispatches code $23 straight to $1B2F)."""
    obs, h, v = uturn_view(state, observer)
    return float(
        uturn_from_cost(projector.render_cost(state, {"h_angle": h, "v_angle": v}, obs))
    )
