"""Acceptance test for the direct-march LOS visibility.

Against ground-truth human-win telemetry (`out/play_*.jsonl`, each line a JSON
record with a base64 `mem` dump of [0, 0x0CFF] and a `player` position), replay
the player-tile ladder: for every consecutive (from_tile -> to_tile) build, the
observer standing on from_tile must be able to see to_tile via the ROM's direct
geometric march (:func:`sentinel.threat.player_sees_tile`).

The coarse fixed-cursor sights-aim sweep (`los.sees_tile`) misses the fine
fractional sub-angles a diagonally-driven cursor reaches, wrongly reporting these
far/launch build tiles unseen; the direct march sees all of them.

Logs are gitignored fixtures; the test skips cleanly when they are absent.
"""

import base64
import json
import os

import pytest

from sentinel import threat
from sentinel.state import State

ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
LOGS = {
    "ls0": (os.path.join(ROOT, "out", "play_20260707_193356.jsonl"), 5),
    "ls42": (os.path.join(ROOT, "out", "play_20260707_194413.jsonl"), 8),
}


def _state_from_record(rec):
    raw = base64.b64decode(rec["mem"])
    mem = bytearray(0x10000)
    mem[0 : len(raw)] = raw
    return State(mem)


def _build_ladder(path):
    """Consecutive (from_tile, to_tile, last_record_at_from_tile) triples, excluding
    everything at/after the first `done_flag` set (post-win noise)."""
    with open(path, encoding="utf-8") as fh:
        lines = [ln for ln in fh if ln.strip()]
    recs = [json.loads(ln) for ln in lines[1:]]  # line 0 is the header
    cut = len(recs)
    for i, r in enumerate(recs):
        if r.get("done_flag"):
            cut = i
            break
    recs = recs[:cut]
    # Collapse consecutive equal player tiles, keeping the LAST record of each run.
    runs = []
    for r in recs:
        p = r.get("player")
        if p is None:
            continue
        tile = (p["x"], p["y"])
        if runs and runs[-1][0] == tile:
            runs[-1] = (tile, r)
        else:
            runs.append((tile, r))
    return [(runs[a][0], runs[a + 1][0], runs[a][1]) for a in range(len(runs) - 1)]


@pytest.mark.parametrize("name", sorted(LOGS))
def test_human_win_builds_all_visible(name):
    path, expected = LOGS[name]
    if not os.path.exists(path):
        pytest.skip(f"needs human-win log fixture {path}")
    ladder = _build_ladder(path)
    assert len(ladder) == expected, f"{name}: {len(ladder)} builds, expected {expected}"
    for from_tile, to_tile, rec in ladder:
        st = _state_from_record(rec)
        seen = threat.player_sees_tile(
            st, to_tile, st.player, eye_z=st.obj_z_height[st.player]
        )
        assert seen, f"{name}: build {from_tile} -> {to_tile} reported unseen"
