#!/usr/bin/env python3
"""Tests for the gaze timeline oracle -- the correctness anchor is exact
agreement with sentinel.threat.ticks_until_seen at t=0 over every ls0 tile."""

import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from sentinel.game import Game  # noqa: E402
from sentinel import enemies, threat  # noqa: E402
from solver.gaze import GazeTimeline, FOV_HALF, FOV_SCAN  # noqa: E402

N = 32
HORIZON = 256


def test_fov_constants():
    assert FOV_SCAN == 0x14
    assert FOV_HALF == enemies.FOV_SCAN // 2 == 10


def test_ticks_until_seen_matches_threat_all_tiles():
    state = Game.new(0).state
    gz = GazeTimeline(state, horizon=HORIZON)
    mismatches = []
    for x in range(N):
        for y in range(N):
            got = min(gz.ticks_until_seen(x, y, 0), 256)
            ref = min(threat.ticks_until_seen(state, x, y, horizon=HORIZON), 256)
            if got != ref:
                mismatches.append(((x, y), got, ref))
    assert not mismatches, f"{len(mismatches)} mismatches: {mismatches[:10]}"


def test_seen_at_and_timeline_consistency():
    state = Game.new(0).state
    gz = GazeTimeline(state, horizon=HORIZON)
    # a tile the Sentinel eventually sees, per threat
    x, y = None, None
    for tx in range(N):
        for ty in range(N):
            if threat.ticks_until_seen(state, tx, ty, horizon=HORIZON) < HORIZON:
                x, y = tx, ty
                break
        if x is not None:
            break
    assert x is not None, "expected some seen tile on ls0"
    tfirst = gz.ticks_until_seen(x, y, 0)
    assert 0 <= tfirst < HORIZON
    assert gz.seen_at(x, y, tfirst)
    if tfirst > 0:
        assert not gz.seen_at(x, y, tfirst - 1)


def test_safe_windows_and_is_safe():
    state = Game.new(0).state
    gz = GazeTimeline(state, horizon=HORIZON)
    for x in range(N):
        for y in range(N):
            windows = gz.safe_windows(x, y)
            # windows partition the not-seen ticks; each is fully safe.
            for t0, t1 in windows:
                assert t0 <= t1
                assert gz.is_safe(x, y, t0, t1)
                assert not gz.seen_at(x, y, t0)
                assert not gz.seen_at(x, y, t1)
            # a fully-hidden tile has one window spanning the whole horizon.
            if gz.ticks_until_seen(x, y, 0) >= HORIZON:
                seen_any = any(gz.seen_at(x, y, t) for t in range(HORIZON))
                if not seen_any and windows:
                    assert windows[0][0] == 0 and windows[-1][1] == HORIZON - 1


def test_is_safe_rejects_seen_span():
    state = Game.new(0).state
    gz = GazeTimeline(state, horizon=HORIZON)
    # find a seen tile and assert a span covering its first-seen tick is unsafe.
    for x in range(N):
        for y in range(N):
            t = gz.ticks_until_seen(x, y, 0)
            if 0 <= t < HORIZON:
                assert not gz.is_safe(x, y, 0, t)
                assert not gz.is_safe(x, y, t, t)
                return
    raise AssertionError("expected a seen tile on ls0")
