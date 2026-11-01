"""The trace gate: it must pass an identical record and fail any divergence."""

import json

from sentinel.tests import trace_suite

REC = {"trace": [[1, "robot", [2, 3], 7, 5.5]], "frames": 9, "energy": 3, "won": True}


def test_record_roundtrip_and_divergence(tmp_path, monkeypatch):
    path = str(tmp_path / "t.json")
    monkeypatch.setattr(
        trace_suite, "run", lambda boards: {str(b): REC for b in boards}
    )
    assert trace_suite.main(["7", "8", "--out", path]) == 0
    assert trace_suite.main(["7", "8", "--compare", path]) == 0
    assert trace_suite.main(["7", "--compare", path]) == 0
    assert trace_suite.main(["9", "--compare", path]) == 1
    altered = json.load(open(path, encoding="utf-8"))
    altered["8"]["trace"][0][2] = [2, 4]
    json.dump(altered, open(path, "w", encoding="utf-8"))
    assert trace_suite.diff(altered, {"7": REC, "8": REC}) == ["8"]
    assert trace_suite.main(["7", "8", "--compare", path]) == 1


def test_play_serialises_the_trace(monkeypatch):
    class Game:
        energy = 4
        state = None

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
    monkeypatch.setattr(trace_suite.actions, "player_dead", lambda _st: False)
    assert trace_suite.play(5) == {
        "trace": [[1, "boulder", [2, 3], 8, 5.875]],
        "frames": 11,
        "energy": 4,
        "won": True,
        "dead": False,
    }
