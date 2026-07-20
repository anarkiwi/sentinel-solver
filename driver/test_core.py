"""Unit tests for the foundation driver's pure live reads and code-digit parsing.
No VICE/Docker: a fake monitor backs a 64KB byte image."""

import os
import sys

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.dirname(HERE))

from driver import core  # noqa: E402


class FakeBM:
    """A monitor over an in-memory 64KB image."""

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


def test_energy_is_six_bit():
    bm = FakeBM()
    bm.set_energy(0x4A)  # 0x4A & 0x3F == 0x0A
    assert core.energy(bm) == 0x0A


def test_player_tile():
    bm = FakeBM()
    bm.set_player(62, 8, 17)
    assert core.player_tile(bm) == (8, 17)


def test_landscape_from_digits_keeps_the_high_bcd_byte():
    """The seed is the whole packed-BCD code, not just its low byte: the PRNG is
    seeded from both bytes ($0C7B/$0C7C), so dropping the leading pair selects a
    different landscape for any code above 0099."""
    assert core.landscape_from_digits("0042") == 0x0042
    assert core.landscape_from_digits("0335") == 0x0335
    assert core.landscape_from_digits("2024") == 0x2024


def test_landscape_from_digits_inverts_enter_landscape():
    for seed in (0x0000, 0x0042, 0x0335, 0x2024, 0x9999):
        assert core.landscape_from_digits(f"{seed:04x}") == seed


def test_landscape_from_digits_matches_prng_seed_bytes():
    from sentinel import prng

    seed = core.landscape_from_digits("0335")
    assert prng.seed_state(seed)[:2] == [0x35, 0x03]
