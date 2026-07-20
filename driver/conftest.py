"""Shared fakes for the driver's no-VICE unit tests."""

import os
import sys

import pytest

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.dirname(HERE))

from driver import core  # noqa: E402


class FakeBM:
    """A monitor over an in-memory 64 KB image (the reads driver code does live)."""

    def __init__(self):
        self.mem = bytearray(0x10000)

    def mem_get(self, a, b):
        return bytes(self.mem[a : b + 1])

    def set_player(self, slot, x, y):
        self.mem[core.A_SLOT] = slot
        self.mem[core.A_X + slot] = x
        self.mem[core.A_Y + slot] = y

    def set_energy(self, e):
        self.mem[core.A_ENERGY] = e


@pytest.fixture
def fake_bm():
    return FakeBM()
