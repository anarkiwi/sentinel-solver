"""Numba twin of the badline clock -- :func:`sentinel.badline.charge`.

:mod:`sentinel.badline` stays the reference and the numba-absent fallback; the tables
are the same objects, so the two cannot drift.
"""

import numpy as np

from sentinel import jitcache

jitcache.install()  # must precede numba: the cache key carries the cost constants

from numba import njit  # noqa: E402  pylint: disable=wrong-import-position

from sentinel import badline, passcost, writeruns  # noqa: E402

_EVENT_POS = badline.EVENT_POS
_EVENT_SHORT_IRQ = badline.EVENT_SHORT_IRQ
_START = writeruns.START
_LENGTH_AT = writeruns.LENGTH_AT
_RUN = writeruns.RUN
_STEAL = passcost.BADLINE_STEAL
_SHORT_IRQ = passcost.SHORT_IRQ
_N_EVENTS = badline.N_EVENTS
_PAL_FRAME_CYCLES = passcost.PAL_FRAME_CYCLES


_NO_EVENT = badline.NO_EVENT
_FIELDS = badline.CLOCK_FIELDS
_WEIGHT_SHIFT = badline.WEIGHT_SHIFT
_OWED = writeruns.OWED


@njit(cache=True)
def frame_clock(armed):
    """A clock at the raster IRQ, its events re-armed; ``armed`` False places none."""
    clk = np.zeros(_FIELDS, dtype=np.int64)
    clk[1] = _EVENT_POS[0] if armed else _NO_EVENT
    clk[2] = 0 if armed else _N_EVENTS
    return clk


@njit(cache=True, inline="always")
def owed_at(anchor, offset):
    """Cycles the instruction ``offset`` into the run at ``anchor`` has still to run."""
    start = _START[anchor]
    if start < 0 or offset < 0 or offset >= _LENGTH_AT[anchor]:
        return np.int64(0)
    return np.int64(_OWED[start + offset])


@njit(cache=True, inline="always")
def run_at(anchor, offset):
    """Consecutive write cycles ``offset`` cycles into the run counted from ``anchor``."""
    start = _START[anchor]
    if start < 0 or offset >= _LENGTH_AT[anchor]:
        return 0
    return _RUN[start + offset]


@njit(cache=True)
def _place(clk, anchor, cycles):
    """The slow half of :func:`charge`: the events this term's cycles reach."""
    pos = clk[0]
    index = clk[2]
    refund = 0
    start = pos
    end = pos + cycles
    while _EVENT_POS[index] < end:
        split = _EVENT_SHORT_IRQ[index]
        if split:
            pos += split
            end += split
        else:
            spend = _STEAL - run_at(anchor, _EVENT_POS[index] - pos)
            refund += _STEAL - spend
            pos += spend
            end += spend
        index += 1
    if start < _PAL_FRAME_CYCLES <= end:  # the raster crosses this term
        clk[5] = owed_at(anchor, _PAL_FRAME_CYCLES - pos)
    clk[0] = end
    clk[1] = _EVENT_POS[index]
    clk[2] = index
    clk[3] += refund
    return cycles - refund


@njit(cache=True, inline="always")
def charge(clk, anchor, cycles):
    """Spend ``cycles`` of the run at ``anchor``, less what its writes refund."""
    end = clk[0] + cycles
    if end <= clk[1]:
        if clk[0] < _PAL_FRAME_CYCLES <= end:  # the raster crosses this term
            clk[5] = owed_at(anchor, _PAL_FRAME_CYCLES - clk[0])
        clk[0] = end
        return cycles
    return _place(clk, anchor, cycles)


@njit(cache=True, inline="always")
def overhang(clk):
    """How far past the raster this frame's clock ran: the next frame's own carry."""
    return clk[0] - _PAL_FRAME_CYCLES + clk[7]


@njit(cache=True)
def spend(clk, anchor, budget):
    """Run the clock over every cycle ``budget`` buys of the run at ``anchor``."""
    pos = clk[0]
    index = clk[2]
    refund = 0
    end = pos + budget
    while _EVENT_POS[index] < end:
        split = _EVENT_SHORT_IRQ[index]
        if split:
            pos += split
            end += split
        else:
            back = run_at(anchor, _EVENT_POS[index] - pos)
            refund += back
            pos += _STEAL - back
            end += _STEAL
        index += 1
    if clk[0] < _PAL_FRAME_CYCLES <= end:  # the budget stops the run at the raster
        clk[5] = 0
        clk[6] = 0
    clk[0] = end
    clk[1] = _EVENT_POS[index]
    clk[2] = index
    clk[3] += refund


@njit(cache=True)
def stall(clk, cycles):
    """The whole video frames a strip replot stalls the play loop for."""
    index = clk[2]
    end = clk[0]
    while index < _N_EVENTS:
        split = _EVENT_SHORT_IRQ[index]
        end += split if split else _STEAL
        index += 1
    if clk[0] < _PAL_FRAME_CYCLES:  # the stall carries no map: nothing is owed at it
        clk[5] = 0
        clk[6] = 0
    clk[0] = end
    clk[1] = _EVENT_POS[index]
    clk[2] = index
    clk[7] += cycles


@njit(cache=True)
def _place_run(clk, cycles, step):
    """The slow half of :func:`_charge_step`: the events this term's cycles reach."""
    index = clk[2]
    refund = 0
    end = clk[0] + cycles
    while _EVENT_POS[index] < end:
        split = _EVENT_SHORT_IRQ[index]
        if split:
            end += split
        else:
            clk[4] += step
            back = clk[4] >> _WEIGHT_SHIFT
            clk[4] -= back << _WEIGHT_SHIFT
            refund += back
            end += _STEAL - back
        index += 1
    if clk[0] < _PAL_FRAME_CYCLES <= end:  # no map: the raster's own b is unknown
        clk[5] = 0
        clk[6] = step
    clk[0] = end
    clk[1] = _EVENT_POS[index]
    clk[2] = index
    clk[3] += refund
    return cycles - refund


@njit(cache=True, inline="always")
def _charge_step(clk, cycles, step):
    """Spend ``cycles`` whose every window refunds the fixed-point ``step``."""
    end = clk[0] + cycles
    if end <= clk[1]:
        if clk[0] < _PAL_FRAME_CYCLES <= end:  # no map: the raster's own b is unknown
            clk[5] = 0
            clk[6] = step
        clk[0] = end
        return cycles
    return _place_run(clk, cycles, step)


@njit(cache=True, inline="always")
def charge_run(clk, cycles, weight):
    """Spend ``cycles`` of a run no static map reaches, less what its writes refund."""
    return _charge_step(clk, cycles, (weight << _WEIGHT_SHIFT) // cycles)


@njit(cache=True, inline="always")
def carry(clk, cycles, step):
    """Advance over ``cycles`` a previous frame's budget paid; return their refund."""
    return cycles - _charge_step(clk, cycles, step)
