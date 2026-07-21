"""The recorded ls110 human win, replayed two moves from its end.

Retrograde replay of ``out/play_20260721_170900.jsonl`` (gitignored; skipped when
absent) found the planner takes ZERO actions from the record where the human had
already built the winning robot on the platform.
"""

import base64
import os

import pytest

from sentinel import actions, enemies, terrain, memmap as mm
from sentinel.astar_player import AStarPlayer
from sentinel.game import Game
from sentinel.state import State
from sentinel.tests import telemetry

_LOG = "play_20260721_170900.jsonl"
_LANDSCAPE = 110
_NODE_BUDGET = 400
_TIME_BUDGET = 60.0


def _state(rec, base):
    """The record's [0,$0CFF] dump over a FULL generated image: the enemy clock reads
    ROM tables above $0CFF, which zero-padding (``telemetry.state_from_record``) would
    leave blank."""
    mem = bytearray(base)
    raw = base64.b64decode(rec["mem"])
    mem[0 : len(raw)] = raw
    return State(mem)


def _two_moves_out():
    """The last pre-win record that is a transfer + hyperspace from the win: Sentinel
    absorbed, a robot standing on the platform, the player standing off it."""
    path = telemetry.log_path(_LOG)
    if not os.path.exists(path):
        pytest.skip(f"{_LOG} absent (gitignored telemetry)")
    recs = telemetry.records(path)
    live = []
    for rec in recs:
        if rec.get("done_flag"):
            break
        live.append(rec)
    base = bytes(Game.typed(_LANDSCAPE).state.mem)
    for rec in reversed(live):
        st = _state(rec, base)
        top = terrain.top_object(st, *st.platform_xy)
        if top is None or not st.is_empty(actions.SENTINEL_SLOT):
            continue
        if st.obj_type[top] == mm.T_ROBOT and not actions.on_platform(st):
            return st
    raise AssertionError("no two-moves-out record in the log")


def test_planner_closes_the_logged_ls110_endgame():
    """From the human's own second-to-last position the plan is transfer then
    hyperspace, and the player takes it -- it used to emit no action at all."""
    st = _two_moves_out()
    assert not enemies.enemy_slots(st) and st.energy >= 3
    player = AStarPlayer(Game(st), node_budget=_NODE_BUDGET, time_budget=_TIME_BUDGET)
    won = player.run(max_actions=6)
    verbs = [rec[1] for rec in player.trace]
    assert verbs, "planner froze two moves from the recorded win"
    assert won and verbs == ["transfer", "hyperspace"], verbs
