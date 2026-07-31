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


_NO_EVENT = badline.NO_EVENT
_FIELDS = badline.CLOCK_FIELDS


@njit(cache=True)
def frame_clock(armed):
    """A clock at the raster IRQ, its events re-armed; ``armed`` False places none."""
    clk = np.zeros(_FIELDS, dtype=np.int64)
    clk[1] = _EVENT_POS[0] if armed else _NO_EVENT
    clk[2] = 0 if armed else _N_EVENTS
    return clk


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
    end = pos + cycles
    while _EVENT_POS[index] < end:
        if _EVENT_SHORT_IRQ[index]:
            pos += _SHORT_IRQ
            end += _SHORT_IRQ
        else:
            spend = _STEAL - run_at(anchor, _EVENT_POS[index] - pos)
            refund += _STEAL - spend
            pos += spend
            end += spend
        index += 1
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
        clk[0] = end
        return cycles
    return _place(clk, anchor, cycles)
