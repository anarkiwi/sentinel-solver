"""Unit tests for the keyboard-aim driver's reuse bookkeeping -- the sights-state read
and committed-bearing tracking that let a same-bearing follow-on step keep sights ON and
drive only the cursor (skipping the initialise_sights $134C recenter). No VICE/Docker: a
fake monitor backs a 64KB byte image."""

import os
import sys

import pytest

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.dirname(HERE))

from driver import kbd_aim  # noqa: E402


def _drv(bm):
    return kbd_aim.KbdDriver(bm, log=lambda *a: None)


def test_sights_live_on_reads_flag_bit7(fake_bm):
    d = _drv(fake_bm)
    d.bm.mem[kbd_aim.A_SFLAG] = 0x80
    assert d.sights_live_on() is True
    d.bm.mem[kbd_aim.A_SFLAG] = 0x00
    assert d.sights_live_on() is False
    d.bm.mem[kbd_aim.A_SFLAG] = 0x7F  # bit7 clear, other bits set
    assert d.sights_live_on() is False


def test_cur_reads_cursor_bytes(fake_bm):
    d = _drv(fake_bm)
    d.bm.mem[kbd_aim.A_CX] = 99
    d.bm.mem[kbd_aim.A_CY] = 43
    assert d.cur() == (99, 43)


def test_a_wait_through_a_redraw_is_one_wait_and_its_timeout_raises(fake_bm):
    """A redraw delays the pc, it does not stall it: one wait of every pass's window,
    and a wait that still times out raises."""
    d = _drv(fake_bm)
    calls = []

    def run_until_pc(pc, timeout):
        calls.append((pc, timeout))
        if len(calls) > 1:
            raise TimeoutError(pc)

    d.bm.run_until_pc = run_until_pc
    assert d._run_past_redraw(0x365D, 1.0)
    assert calls == [(0x365D, 1.0 * kbd_aim._SCAN_WAIT_PASSES)]
    with pytest.raises(TimeoutError):
        d._run_past_redraw(0x365D, 1.0)
