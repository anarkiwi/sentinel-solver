"""Unit tests for the keyboard-aim driver's reuse bookkeeping -- the sights-state read
and committed-bearing tracking that let a same-bearing follow-on step keep sights ON and
drive only the cursor (skipping the initialise_sights $134C recenter). No VICE/Docker: a
fake monitor backs a 64KB byte image."""

import os
import sys

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.dirname(HERE))

from driver import kbd_aim  # noqa: E402


class FakeBM:
    """A monitor over an in-memory 64KB image."""

    def __init__(self):
        self.mem = bytearray(0x10000)

    def mem_get(self, a, b):
        return bytes(self.mem[a : b + 1])


def _drv():
    return kbd_aim.KbdDriver(FakeBM(), log=lambda *a: None)


def test_sights_live_on_reads_flag_bit7():
    d = _drv()
    d.bm.mem[kbd_aim.A_SFLAG] = 0x80
    assert d.sights_live_on() is True
    d.bm.mem[kbd_aim.A_SFLAG] = 0x00
    assert d.sights_live_on() is False
    d.bm.mem[kbd_aim.A_SFLAG] = 0x7F  # bit7 clear, other bits set
    assert d.sights_live_on() is False


def test_committed_bearing_lifecycle():
    d = _drv()
    assert d.committed_bearing() is None
    d.set_bearing(0x60, 0xF5)
    assert d.committed_bearing() == (0x60, 0xF5)
    d.set_bearing(0x160, 0x1F5)  # masked to a byte each
    assert d.committed_bearing() == (0x60, 0xF5)
    d.clear_bearing()
    assert d.committed_bearing() is None


def test_cur_reads_cursor_bytes():
    d = _drv()
    d.bm.mem[kbd_aim.A_CX] = 99
    d.bm.mem[kbd_aim.A_CY] = 43
    assert d.cur() == (99, 43)
