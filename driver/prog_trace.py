#!/usr/bin/env python3
"""Trace a driver program on a clone of an emulated session, frame by frame.

:func:`trace` interprets ``program`` on a clone of an ``EmuMon`` with a decode hook,
recording ``addrs`` at every $9630 frame marker and the frame each of ``pcs`` is reached
on, so a move's stall can be read against the ROM state that caused it.
"""

from driver import clock, prog


def trace(bm, program, pcs=(), addrs=()):
    """``(outcome, frames, events)``: ``frames`` rows ``(frame, *bytes at addrs, keys
    held)``, ``events`` ``(frame, pc)`` for each arrival at one of ``pcs``."""
    twin = bm.clone()
    m, watch, rows, events = twin.m, frozenset(pcs), [], []
    frame = [0]

    def hook(machine):
        pc = machine.mpu.pc
        if pc == clock.FRAME_PC:
            frame[0] += 1
            ram = machine.ram
            held = tuple(i for i, b in enumerate(machine.keymatrix) if b)
            rows.append((frame[0], *(ram[a] for a in addrs), held))
        if pc in watch:
            events.append((frame[0], pc))

    m.set_decode_hook(hook)
    try:
        outcome = prog.interpret(twin, program)
    finally:
        m.set_decode_hook(None)
    return outcome, rows, events
