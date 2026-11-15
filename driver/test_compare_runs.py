"""compare_runs finds each audit's first divergent row."""

import json

from driver import compare_runs


def test_audits_report_the_first_divergent_row():
    a = {"trace": [[0, "boulder"], [1, "robot"]], "live_audit": [[0, "entry"]]}
    b = {"trace": [[0, "boulder"], [1, "tree"]], "live_audit": [[0, "entry"], [1, "a"]]}
    got = compare_runs.audits(a, b)
    assert got["trace"] == (2, 2, 1, 1)
    assert got["live_audit"] == (1, 2, 1, 1)
    assert got["lookahead_audit"] == (0, 0, 0, None)


def test_cli_prints_each_audits_first_divergent_row(tmp_path, capsys):
    a, b = tmp_path / "a.json", tmp_path / "b.json"
    a.write_text(json.dumps({"trace": [[0], [1]]}))
    b.write_text(json.dumps({"trace": [[0], [2]]}))
    assert compare_runs.main([str(a), str(b)]) == 0
    out = capsys.readouterr().out
    assert "1 differ first #1: [1] vs [2]" in out and "live_audit" in out
