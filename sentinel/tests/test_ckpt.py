"""The checkpoint fidelity gate: re-entry must be indistinguishable from never leaving.

Run on ls0 with the A* player, which needs no stance-graph sweep.
"""

import pytest

from sentinel.astar_player import AStarPlayer
from sentinel.game import Game
from sentinel.stance_player import StancePlayer
from sentinel.tests import ckpt

TICKS = 6


def _run(ticks=TICKS):
    player = AStarPlayer(Game.typed(0), verbose=False)
    snaps = []
    ckpt.capture(player, ticks, snaps)
    return player, snaps


def test_fields_all_exist_on_the_player():
    """A typo in FIELDS would silently drop state, since snapshot skips absentees."""
    player = StancePlayer(Game.typed(0), verbose=False)
    missing = [name for name in ckpt.FIELDS if not hasattr(player, name)]
    assert not missing


def test_snapshots_do_not_alias_each_other():
    """Live mutable fields must be copied, or every checkpoint holds the last tick."""
    _, snaps = _run()
    lengths = [len(s["fields"]["trace"]) for s in snaps]
    assert lengths == sorted(lengths)
    assert lengths[0] < lengths[-1]
    assert snaps[0]["fields"]["trace"] is not snaps[-1]["fields"]["trace"]


def test_restore_does_not_mutate_the_checkpoint():
    """A corpus is re-entered many times; replaying must leave the snapshot intact."""
    _, snaps = _run()
    snap = snaps[1]
    before = len(snap["fields"]["trace"])
    player = ckpt.restore(snap, cls=AStarPlayer, verbose=False)
    ckpt.capture(player, 2)
    assert len(snap["fields"]["trace"]) == before


def test_reentry_reproduces_the_run_tail():
    """The gate proper, and it must replay a NON-ZERO number of actions to mean anything."""
    player, snaps = _run()
    rows = ckpt.verify(snaps, backs=(2, 4), cls=AStarPlayer, verbose=False)
    assert rows, "no usable re-entry points"
    assert any(row["actions"] > 0 for row in rows), "vacuous pass over an empty tail"
    assert all(row["identical"] for row in rows)
    assert player.trace


def test_roundtrip_through_disk(tmp_path):
    _, snaps = _run(3)
    path = tmp_path / "run.ckpt"
    ckpt.save(snaps, str(path))
    back = ckpt.load(str(path))
    assert [s["tick"] for s in back] == [s["tick"] for s in snaps]
    assert back[-1]["mem"] == snaps[-1]["mem"]


def test_describe_names_the_stance():
    player, _ = _run(2)
    assert "tile=" in ckpt.describe(player) and "foes=" in ckpt.describe(player)


@pytest.mark.parametrize("cls", [AStarPlayer, StancePlayer])
def test_restore_accepts_either_player(cls):
    _, snaps = _run(2)
    player = ckpt.restore(snaps[-1], cls=cls, verbose=False)
    assert player.st.mem == snaps[-1]["mem"]
    assert player.frames == snaps[-1]["fields"]["frames"]
