"""The trace gate: it must pass an identical record and fail any divergence."""

import json

import pytest

from sentinel.tests import trace_suite
from sentinel.tests.test_kern import SIM, gpu

REC = {
    "landscape": 7,
    "enemies": 2,
    "trace": [[1, "robot", [2, 3], 7, 5.5]],
    "frames": 9,
    "energy": 3,
    "won": True,
}


def test_record_roundtrip_and_divergence(tmp_path, monkeypatch):
    path = str(tmp_path / "t.json")
    monkeypatch.setattr(
        trace_suite, "run", lambda boards, player="phase": {str(b): REC for b in boards}
    )
    assert trace_suite.main(["7", "8", "--out", path]) == 0
    assert trace_suite.main(["7", "8", "--compare", path]) == 0
    assert trace_suite.main(["7", "--compare", path]) == 0
    assert trace_suite.main(["9", "--compare", path]) == 1
    altered = json.load(open(path, encoding="utf-8"))
    assert "trace" not in altered["8"] and altered["8"]["actions"] == 1
    altered["8"]["digest"] = trace_suite.digest([[1, "robot", [2, 4], 7, 5.5]])
    json.dump(altered, open(path, "w", encoding="utf-8"))
    assert trace_suite.diff(altered, {"7": REC, "8": REC}) == ["8"]
    assert trace_suite.main(["7", "8", "--compare", path]) == 1


def test_compare_accepts_full_trace_baseline(tmp_path, monkeypatch):
    path = str(tmp_path / "full.json")
    monkeypatch.setattr(trace_suite, "run", lambda boards, player="phase": {"7": REC})
    json.dump({"7": {**REC, "dead": False}}, open(path, "w", encoding="utf-8"))
    assert trace_suite.main(["7", "--compare", path]) == 0
    json.dump({"7": {**REC, "energy": 4}}, open(path, "w", encoding="utf-8"))
    assert trace_suite.main(["7", "--compare", path]) == 1


def test_readme_flag_rewrites_the_block(monkeypatch, tmp_path):
    calls = []
    monkeypatch.setattr(trace_suite.readme_results, "rewrite", lambda: calls.append(1))
    assert trace_suite.main(["--readme"]) == 0
    monkeypatch.setattr(trace_suite, "run", lambda boards, player="phase": {"7": REC})
    out = str(tmp_path / "o.json")
    assert trace_suite.main(["7", "--out", out, "--readme"]) == 0
    assert calls == [1, 1]


def test_play_serialises_the_trace(monkeypatch):
    class Game:
        energy = 4
        state = None

        @staticmethod
        def enemy_slots():
            return [0, 5]

        @staticmethod
        def typed(_number):
            return Game()

    class Player:
        trace = [(1, "boulder", (2, 3), 8, 5.875)]
        frames = 11

        def __init__(self, _game, verbose):
            assert not verbose

        def run(self, max_actions):
            return max_actions > 0

    monkeypatch.setattr(trace_suite, "Game", Game)
    monkeypatch.setattr(trace_suite, "PhasePlayer", Player)
    assert trace_suite.play(5) == {
        "landscape": 5,
        "enemies": 2,
        "trace": [[1, "boulder", [2, 3], 8, 5.875]],
        "frames": 11,
        "energy": 4,
        "won": True,
    }


def test_the_player_picks_its_own_results_file(tmp_path, monkeypatch):
    seen = []
    monkeypatch.setattr(
        trace_suite,
        "run",
        lambda boards, player="phase": seen.append(player) or {"7": REC},
    )
    beam, phase = str(tmp_path / "beam.json"), str(tmp_path / "phase.json")
    monkeypatch.setattr(trace_suite, "RESULTS", {"beam": beam, "phase": phase})
    assert trace_suite.main(["7", "--player", "beam", "--out", beam]) == 0
    assert trace_suite.main(["7", "--out", phase]) == 0
    assert trace_suite.main(["7", "--player", "beam", "--compare"]) == 0
    other = json.load(open(phase, encoding="utf-8"))
    other["7"]["energy"] += 1
    json.dump(other, open(beam, "w", encoding="utf-8"))
    assert trace_suite.main(["7", "--player", "beam", "--compare"]) == 1
    assert trace_suite.main(["7", "--compare"]) == 0
    assert seen == ["beam", "phase", "beam", "beam", "phase"]


def test_make_builds_the_named_player():
    game = trace_suite.Game.typed(0)
    assert type(trace_suite.make("beam", game)).__name__ == "BeamPlayer"
    assert type(trace_suite.make("phase", game)).__name__ == "PhasePlayer"


def test_time_suite_warms_up_times_each_board_and_splits_a_profile(monkeypatch, capsys):
    from sentinel.tests import time_suite  # pylint: disable=import-outside-toplevel

    monkeypatch.setenv("SENTINEL_BACKEND", "cpu")
    calls = []

    def play(number, player, width):
        calls.append((number, player, width))
        return {"won": True, "trace": [1, 2], "frames": 7}

    monkeypatch.setattr(trace_suite, "play", play)
    assert time_suite.main(["5", "--width", "3", "--profile", "5"]) == 0
    assert calls == [(0, "beam", 3), (5, "beam", 3), (5, "beam", 3)]
    out = capsys.readouterr().out
    assert "    5" in out and "planning" in out and "cuda api waits" in out


def test_the_beam_width_reaches_the_player():
    game = trace_suite.Game.typed(0)
    assert trace_suite.make("beam", game, 5).width == 5
    assert trace_suite.make("beam", game).width == 16


@gpu
@pytest.mark.skipif(SIM, reason="the simulator has no profiler hooks")
def test_time_suite_compiles_the_device_kernel_and_brackets_the_capture(
    monkeypatch, capsys
):
    from sentinel.tests import time_suite  # pylint: disable=import-outside-toplevel

    monkeypatch.setenv("SENTINEL_BACKEND", "cuda")
    monkeypatch.setattr(
        trace_suite,
        "play",
        lambda number, player, width: {"won": False, "trace": [], "frames": 0},
    )
    assert time_suite.main(["3", "--capture"]) == 0
    assert "    3" in capsys.readouterr().out
