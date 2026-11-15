"""compare_runs finds each audit's first divergent row and separates a tick image's
new differences from the entry image's own."""

import json

import numpy as np

from driver import compare_runs


def _tick(directory, n, mem):
    directory.mkdir(exist_ok=True)
    np.savez_compressed(directory / f"t{n:04d}.npz", mem=mem)


def test_audits_report_the_first_divergent_row():
    a = {"trace": [[0, "boulder"], [1, "robot"]], "frame_audit": [["p1", 5]]}
    b = {"trace": [[0, "boulder"], [1, "tree"]], "frame_audit": [["p1", 5], ["p2", 3]]}
    got = compare_runs.audits(a, b)
    assert got["trace"] == (2, 2, 1, 1)
    assert got["frame_audit"] == (1, 2, 1, 1)
    assert got["settle_audit"] == (0, 0, 0, None)


def test_ticks_separate_new_differences_from_the_entry_images(tmp_path, capsys):
    a, b = np.zeros(0x10000, np.uint8), np.zeros(0x10000, np.uint8)
    b[0x0800] = b[0xD020] = 1  # a loader leftover, and the I/O page
    for n in range(2):
        _tick(tmp_path / "a", n, a)
    _tick(tmp_path / "b", 0, b)
    b[0x0800], b[0x0C0A] = 0, 9  # the leftover overwritten alike, then a real split
    _tick(tmp_path / "b", 1, b)
    rows = compare_runs.ticks(tmp_path / "a", tmp_path / "b")
    assert rows == [("t0000.npz", 1, 0, 0, []), ("t0001.npz", 1, 1, 1, [0x0C0A])]
    run = tmp_path / "run.json"
    run.write_text(json.dumps({"trace": [[0]]}))
    argv = [str(run), str(run), "--ticks", str(tmp_path / "a"), str(tmp_path / "b")]
    assert compare_runs.main(argv) == 0
    out = capsys.readouterr().out
    assert "first new difference: t0001.npz" in out and "0 differ" in out
