"""Shared fakes for the driver's no-VICE unit tests."""

import os
import sys

import pytest

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.dirname(HERE))


class FakeBM:
    """A monitor over an in-memory 64 KB image (the reads driver code does live)."""

    def __init__(self):
        self.mem = bytearray(0x10000)

    def mem_get(self, a, b):
        return bytes(self.mem[a : b + 1])


@pytest.fixture
def fake_bm():
    return FakeBM()
