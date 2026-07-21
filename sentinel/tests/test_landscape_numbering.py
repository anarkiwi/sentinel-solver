"""A landscape number is the number a PLAYER TYPES, never a raw PRNG seed.

The ROM stores the typed code packed-BCD and seeds from those bytes, so the seed is the
digits read as hex. Mixing the two silently selects a different board; these pin the
conversion against the human-win fixtures, which record both names.
"""

import json
import os

import pytest

from sentinel import landscape
from sentinel.game import Game

FIXTURES = os.path.join(
    os.path.dirname(os.path.abspath(__file__)), "fixtures", "human_wins"
)


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


@pytest.mark.parametrize(
    "fixture,typed,seed,player",
    [("ls42.json", 42, 66, (13, 29)), ("ls335.json", 335, 821, (11, 17))],
)
def test_typed_reproduces_the_human_fixture_board(fixture, typed, seed, player):
    """``Game.typed(n)`` builds what the human played: the fixtures carry the typed code
    AND the seed the ROM derived, so they arbitrate."""
    with open(os.path.join(FIXTURES, fixture), encoding="utf-8") as fh:
        rec = json.load(fh)
    assert (rec["entered_code"], rec["landscape"]) == (typed, seed)
    game = Game.typed(typed)
    first = rec["events"][0]["player"]
    assert game.player_xy() == player == (first["x"], first["y"])


def test_typed_and_raw_seed_are_different_boards():
    """The trap this module exists to catch: ``Game.new(335)`` is NOT landscape 335."""
    assert Game.typed(335).player_xy() != Game.new(335).player_xy()
