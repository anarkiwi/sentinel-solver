"""The human-win fixtures carry recorder artifacts that are not player actions.

``_extract._classify`` mints a transfer whenever no object changed, so a drain tick
($1838) becomes a transfer onto the player's own tile, and enemy discharge trees
($1A5D) survive as creates -- ~23% noise on ls335 for anyone skipping ``human_events``.
"""

import collections
import os

import pytest

from sentinel import memmap as mm
from sentinel.tests.telemetry import human_events

FIXTURES = os.path.join(
    os.path.dirname(os.path.abspath(__file__)), "fixtures", "human_wins"
)


def _events(name):
    return human_events(os.path.join(FIXTURES, name))


@pytest.mark.parametrize(
    "name,absorbs,creates,transfers",
    [("ls335.json", 73, 38, 18), ("ls42.json", 19, 15, 8), ("ls0.json", 11, 9, 5)],
)
def test_real_action_counts(name, absorbs, creates, transfers):
    counts = collections.Counter(ev["verb"] for ev in _events(name))
    assert counts == {"absorb": absorbs, "create": creates, "transfer": transfers}


def test_ls335_has_129_real_actions_of_168_rows():
    """19 drain ticks + 20 discharge trees are recorder artifacts, not moves."""
    assert len(_events("ls335.json")) == 129


def test_a_player_never_creates_a_tree():
    """The rule the tree filter rests on: create makes boulders and robots only."""
    for name in ("ls0.json", "ls42.json", "ls335.json"):
        for ev in _events(name):
            if ev["verb"] == "create":
                assert ev["otype"] in (mm.T_BOULDER, mm.T_ROBOT)


def test_no_transfer_lands_on_the_tile_it_started_from():
    """A real transfer moves the player; an own-tile transfer is a drain tick."""
    for name in ("ls0.json", "ls42.json", "ls335.json"):
        for ev in _events(name):
            if ev["verb"] == "transfer":
                player = ev["player"]
                assert tuple(ev["target"]) != (player["x"], player["y"])
