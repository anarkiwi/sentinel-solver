"""Unit tests for the driver's code-digit parsing; no VICE or Docker."""

import os
import sys

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.dirname(HERE))

from driver import core  # noqa: E402


def test_landscape_from_digits_inverts_enter_landscape():
    """The seed is the whole packed-BCD code, not just its low byte: the PRNG is
    seeded from both bytes ($0C7B/$0C7C), so dropping the leading pair selects a
    different landscape for any code above 0099."""
    for seed in (0x0000, 0x0042, 0x0335, 0x2024, 0x9999):
        assert core.landscape_from_digits(f"{seed:04x}") == seed


def test_landscape_from_digits_matches_prng_seed_bytes():
    from sentinel import prng

    seed = core.landscape_from_digits("0335")
    assert prng.Prng.seeded(seed).s[:2] == [0x35, 0x03]
