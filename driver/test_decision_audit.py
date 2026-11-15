"""The decision audit: the chosen line's candidates are the search's own, forcing the
chosen move replays the game, and the report marks what was taken."""

import json

import pytest

from driver import decision_audit as da, kbd_prog, prog, search_player as sp
from driver.emu import EmuMon
from sentinel import emu

oracle = pytest.mark.oracle
KNOBS = {"width": 2, "depth": 2, "branch": 1, "commit": 1}


def _facts(**kw):
    return {"won": 0, "alive": 1, "energy": 7, "z": 3, "z_frac": 0x80, **kw}


def test_an_ending_reads_the_last_moves_facts():
    assert da.ending(_facts(), 4) == {
        "won": False,
        "alive": True,
        "energy": 7,
        "eye": 3.5,
        "actions": 4,
    }
    assert da.ending(_facts(alive=0, won=1), 1)["alive"]
    assert not da.ending(_facts(alive=0), 1)["alive"]


def test_the_report_stars_the_chosen_move_and_names_refusals():
    now = da.ending(_facts(), 0)
    took = ("absorb", (1, 2), 0)
    lines = list(
        da.report([(0, now, took, [(took, now), (("wait", (3, 4), 0), None)])])
    )
    assert lines[0].startswith("decision   0 E= 7 eye= 3.500 took")
    assert lines[1].lstrip().startswith("* ('absorb'") and "refused" in lines[2]


@pytest.fixture(scope="module")
def game():
    player = sp.SearchPlayer(EmuMon(emu.enter(0)), log=[].append, **KNOBS)
    player.progress = False
    player.run(3)
    return player


@oracle
def test_decisions_are_the_machines_before_each_move_the_game_played(game):
    real = da.decisions(0, KNOBS, 3)
    assert len(real) == 3
    assert [(m[0], list(m[1])) for _bm, m in real] == [(r[1], r[2]) for r in game.audit]
    for (bm, _m), row in zip(real[1:], game.audit):
        now = kbd_prog.move_facts(prog.execute(bm.clone(), sp.ENTRY))
        assert now["energy"] == row[4]["energy"] and now["z"] == row[4]["z"]


@oracle
def test_forcing_the_chosen_move_replays_the_rest_of_the_game(game):
    rows = da.audit(0, KNOBS, KNOBS, 1, max_actions=3)
    ((d, now, chosen, options),) = rows
    assert d == 0 and now["actions"] == 0
    (work,) = sp.batches([sp.Node(EmuMon(emu.enter(0)))], KNOBS["branch"])
    assert [m for m, _end in options] == [m for m, _p in work]
    end = dict(options)[chosen]
    assert end == da.ending(game.audit[-1][4], len(game.audit))
    assert any(e != end for _m, e in options)


@oracle
def test_the_cli_writes_the_audit(tmp_path, capsys, monkeypatch):
    monkeypatch.setitem(sp.PRESETS, "narrow", KNOBS)
    out = tmp_path / "audit.json"
    args = ["0", "--game", "narrow", "--decisions", "1", "--max-actions", "2"]
    assert da.main(args + ["--out", str(out)]) == 0
    text = capsys.readouterr().out
    assert text.startswith("decision   0") and "*" in text
    assert json.loads(out.read_text())[0][0] == 0
