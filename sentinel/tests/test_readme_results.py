"""The README results block is generated from the committed result files, and the
search records are what ``driver.emu_suite`` writes."""

import os

from driver import emu_suite
from sentinel.tests import readme_results as rr

SEARCH = rr.load(rr.SEARCH)
LIVE = rr.load(rr.LIVE)
REC = {"landscape": 7, "enemies": 3, "actions": 5, "energy": 2, "won": True}


def test_readme_block_matches_data():
    with open(rr.README, encoding="utf-8") as handle:
        text = handle.read()
    assert rr.splice(text, rr.block(SEARCH, LIVE)) == text, f"run: {rr.COMMAND}"


def test_search_data_covers_the_suite():
    assert sorted(SEARCH, key=int) == [str(b) for b in emu_suite.SUITE]
    for key, rec in SEARCH.items():
        assert rec["landscape"] == int(key) and isinstance(rec["won"], bool)
        assert len(rec["digest"]) == 64 and int(rec["digest"], 16) >= 0
        assert 1 <= rec["enemies"] <= 8 and rec["preset"] in ("wide", "narrow")


def test_live_data_schema():
    for run in LIVE:
        assert set(run) == {"landscape", "actions", "energy", "media"}
        assert str(run["landscape"]) in SEARCH
        assert isinstance(run["actions"], int) and isinstance(run["energy"], int)
        assert run["media"] is None or os.path.exists(
            os.path.join(rr.ROOT, run["media"])
        )


def test_render_lists_each_live_run_linking_its_media():
    runs = [
        {"landscape": 7, "actions": 6, "energy": 1, "media": None},
        {"landscape": 7, "actions": 5, "energy": 4, "media": "m.png"},
    ]
    out = rr.render({"7": REC}, runs)
    assert "| 7 | 3 | 5 | 2 | **6 actions**, energy 1<br>" in out
    assert "[**5 actions**, energy 4](m.png) |" in out and f"`{rr.COMMAND}`" in out


def test_render_unwon_and_no_live():
    out = rr.render({"7": {**REC, "won": False}}, [])
    assert "| 7 | 3 | 5 (not won) | 2 | — |" in out


def test_splice_and_rewrite(tmp_path, monkeypatch):
    path = tmp_path / "README.md"
    path.write_text(f"head\n{rr.BEGIN}\nold\n{rr.END}\ntail\n")
    data = {rr.SEARCH: SEARCH, rr.LIVE: LIVE}
    monkeypatch.setattr(rr, "load", data.__getitem__)
    rr.rewrite(str(path))
    text = path.read_text()
    assert text.startswith("head\n") and text.endswith(f"{rr.END}\ntail\n")
    assert "\nold\n" not in text
    rr.rewrite(str(path))
    assert path.read_text() == text


def test_a_record_digests_its_trace_and_carries_the_preset():
    trace = [[10, "absorb", [1, 2], 11, 1.875], [20, "wait", [1, 2], 11, 1.875]]
    result = {"landscape": 7, "enemies": 3, "trace": trace, "final_energy": 11}
    rec = emu_suite.summary({**result, "won": True, "frames": 20}, "narrow")
    assert rec == {
        "landscape": 7,
        "enemies": 3,
        "actions": 2,
        "energy": 11,
        "frames": 20,
        "won": True,
        "preset": "narrow",
        "digest": emu_suite.digest(trace),
    }
    assert emu_suite.digest(trace[:1]) != rec["digest"]


def test_the_table_lines_set_each_run_beside_its_record():
    res = {"trace": [[1]] * 4, "won": True, "final_energy": 3, "wall_seconds": 9.5}
    res["lookahead_audit"] = [[1, "a", [0, 0], {"e": 1}, {"e": 1}]] * 2
    res["search"] = {"stats": {"seconds": 2.0, "cpu_seconds": 3.0}}
    head, line = emu_suite.rows({42: res}, {"42": {**REC, "actions": 7}})
    assert head.startswith("board") and line.split() == (
        "42 True 4 3 9.5 2/2 1.5 | True 7 2".split()
    )
