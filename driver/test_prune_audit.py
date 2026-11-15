"""The prune audit: hop landings read from the sight table, rules scored on the forks a
beam kept, and a short recorded game."""

import json

import pytest

from driver import prune_audit as pa
from sentinel import actions, emu, memmap as mm
from sentinel.sight_table import SightTable, stack
from sentinel.state import State

oracle = pytest.mark.oracle


@pytest.fixture(scope="module", name="entry")
def _entry():
    return bytes(emu.enter(0).ram)


def _rows():
    full, partial, stale = [True, 2, 1], [True, 1, 0], [False, 0, 0]
    return [
        [True, ("hop", (1, 1), 0), full, 100, True],
        [False, ("hop", (1, 2), 0), full, 50, True],
        [False, ("hop", (1, 3), 1), partial, 30, False],
        [True, ("hop", (1, 4), 0), stale, 10, True],
        [False, ("absorb", (1, 5), 0), None, 5, True],
    ]


def test_rules_score_forks_frames_and_the_kept_they_would_prune():
    got = pa.score(_rows())
    assert (got["forks"], got["frames"]) == (5, 195)
    assert got["seen fully"] == {
        "forks": 2,
        "frames": 150,
        "kept_by_beam": 1,
        "alive": 2,
    }
    assert got["seen at all"] == {
        "forks": 3,
        "frames": 180,
        "kept_by_beam": 1,
        "alive": 2,
    }


@oracle
def test_the_table_answers_once_per_tile_and_height(entry):
    table = SightTable(entry)
    st = State.from_mem(entry)
    me = st.tile_of(st.player)
    tile = next(
        (x, y)
        for x in range(2, 29)
        for y in range(2, 29)
        if table.seen((x, y)) and (x, y) != me
    )
    assert table.seen(tile) is table.seen(tile, 0)
    assert all(e in pa.enemies(st) for e, _full in table.seen(tile))
    assert table.seen(me) is None
    assert table.fresh(st, tile)
    st.energy = 0x3F
    assert actions.create(st, mm.T_BOULDER, tile) is not None
    assert not table.fresh(st, tile)
    assert stack(st, *tile)[0][0] == mm.T_BOULDER
    assert pa._seen(table, st, ("hop", tile, 0)) == [False, 0, 0]
    assert pa._seen(table, st, ("absorb", tile, 0)) is None
    fresh = State.from_mem(entry)
    n, full = pa._seen(table, fresh, ("hop", tile, 1))[1:]
    assert n == len(table.seen(tile, 1)) and full <= n


@oracle
def test_a_recorded_game_marks_what_the_beam_kept():
    forks, won = pa.record(0, max_actions=1, width=2, depth=2, branch=1, commit=1)
    assert not won and forks
    assert any(kept for kept, *_ in forks) and not all(kept for kept, *_ in forks)
    assert all(f[3] >= 0 for f in forks)
    assert all(f[2] is not None for f in forks if f[1][0] == "hop")


def test_main_writes_the_scored_rows(tmp_path, monkeypatch, capsys):
    monkeypatch.setattr(pa, "record", lambda landscape, **knobs: (_rows(), True))
    assert pa.main(["7", "--out-dir", str(tmp_path)]) == 0
    with open(tmp_path / "prune_0007.json", encoding="utf-8") as fh:
        got = json.load(fh)
    assert got["won"] and got["seen fully"]["kept_by_beam"] == 1
    assert len(got["fork_rows"]) == 5
    assert '"landscape": 7' in capsys.readouterr().out
