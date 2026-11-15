"""A landscape number is the number a PLAYER TYPES, never a raw PRNG seed.

The ROM stores the typed code packed-BCD and seeds from those bytes, so the seed is the
digits read as hex. Mixing the two silently selects a different board; these pin the
conversion against the human-win fixtures, which record both names.
"""

import pytest

from sentinel import landscape
from sentinel.game import Game


@pytest.mark.parametrize(
    "typed,seed",
    [(0, 0), (42, 66), (335, 821), (2024, 8228), (1999, 6553)],
)
def test_seed_for_reads_the_typed_digits_as_hex(typed, seed):
    assert landscape.seed_for(typed) == seed


def test_seed_for_agrees_with_the_live_driver_conversion():
    """The driver types the digits at the real title screen; both paths must agree or
    the sim and the live run play different boards."""
    core = pytest.importorskip("driver.core")
    for typed in (0, 42, 335, 821, 2024):
        assert landscape.seed_for(typed) == core.landscape_from_digits(f"{typed:04d}")


def test_typed_and_raw_seed_are_different_boards():
    """The trap this module exists to catch: ``Game.new(335)`` is NOT landscape 335."""
    assert Game.typed(335).player_xy() != Game.new(335).player_xy()
