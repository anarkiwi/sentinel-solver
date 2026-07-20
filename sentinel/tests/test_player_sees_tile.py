"""Acceptance test for the direct-march LOS visibility against human-win telemetry.

Every consecutive (from_tile -> to_tile) step of the recorded player-tile ladder --
builds AND transfers onto occupied tiles, which ``test_landable`` skips -- must be
seen from from_tile by :func:`sentinel.threat.player_sees_tile`. Gitignored logs skip.
"""

import os

import pytest

from sentinel import threat
from sentinel.tests.telemetry import log_path, state_from_record, tile_ladder

LOGS = {
    "ls0": (log_path("play_20260707_193356.jsonl"), 5),
    "ls42": (log_path("play_20260707_194413.jsonl"), 8),
}


@pytest.mark.parametrize("name", sorted(LOGS))
def test_human_win_builds_all_visible(name):
    path, expected = LOGS[name]
    if not os.path.exists(path):
        pytest.skip(f"needs human-win log fixture {path}")
    ladder = tile_ladder(path)
    assert len(ladder) == expected, f"{name}: {len(ladder)} builds, expected {expected}"
    for from_tile, to_tile, rec in ladder:
        st = state_from_record(rec)
        seen = threat.player_sees_tile(
            st, to_tile, st.player, eye_z=st.obj_z_height[st.player]
        )
        assert seen, f"{name}: build {from_tile} -> {to_tile} reported unseen"
