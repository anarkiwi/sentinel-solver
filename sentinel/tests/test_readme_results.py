"""The README results block is generated from the committed result files."""

import os

import pytest

from sentinel.tests import readme_results as rr
from sentinel.tests import trace_suite

REGEN = rr.COMMAND
OFFLINE = rr.load(rr.OFFLINE)
LIVE = rr.load(rr.LIVE)
REC = {"landscape": 7, "enemies": 3, "actions": 5, "frames": 9, "energy": 2}


def test_readme_block_matches_data():
    with open(rr.README, encoding="utf-8") as handle:
        text = handle.read()
    assert rr.splice(text, rr.block(OFFLINE, LIVE)) == text, f"run: {REGEN}"


def test_offline_data_covers_the_suite():
    assert sorted(OFFLINE, key=int) == [str(b) for b in trace_suite.SUITE]
    for key, rec in OFFLINE.items():
        assert rec["landscape"] == int(key)
        assert len(rec["digest"]) == 64 and int(rec["digest"], 16) >= 0


def test_live_data_schema():
    for run in LIVE:
        assert str(run["landscape"]) in OFFLINE
        assert run["mode"] in rr.MODES
        assert run["digest"] is None or int(run["digest"], 16) >= 0
        assert run["energy"] is None or isinstance(run["energy"], int)
        assert isinstance(run["actions"], int)
        assert run["media"] is None or os.path.exists(
            os.path.join(rr.ROOT, run["media"])
        )


def test_render_marks_stale_live_rows():
    off = {"7": {**REC, "won": True, "digest": "ab"}}
    runs = [
        {"landscape": 7, "mode": "frozen", "actions": 6, "energy": None, "media": None},
        {"landscape": 7, "mode": "live", "actions": 5, "energy": 4, "media": "m.png"},
    ]
    fresh = [{**r, "digest": "ab"} for r in runs]
    old = [{**runs[0], "digest": None}, {**runs[1], "digest": "cd"}]
    assert "†" not in rr.render(off, fresh)
    out = rr.render(off, old)
    assert out.count("†") == 3
    assert "**6 actions** (enemies frozen)†" in out
    assert "[**5 actions**, energy 4](m.png)†" in out
    assert f"`{REGEN}`" in out


def test_render_unwon_and_no_live():
    off = {"7": {**REC, "won": False, "digest": "ab"}}
    assert "| 7 | 3 | 5 (not won) | 2 | — |" in rr.render(off, [])


def test_splice_and_rewrite(tmp_path, monkeypatch):
    path = tmp_path / "README.md"
    path.write_text(f"head\n{rr.BEGIN}\nold\n{rr.END}\ntail\n")
    monkeypatch.setattr(rr, "load", lambda p: OFFLINE if p == rr.OFFLINE else LIVE)
    rr.rewrite(str(path))
    text = path.read_text()
    assert text.startswith("head\n") and text.endswith(f"{rr.END}\ntail\n")
    assert "\nold\n" not in text
    rr.rewrite(str(path))
    assert path.read_text() == text


@pytest.mark.parametrize("board", trace_suite.SUITE)
def test_offline_results_are_current(board):
    got = trace_suite.summary(trace_suite.play(board))
    assert got == OFFLINE[str(board)], f"stale offline data; run: {REGEN}"
