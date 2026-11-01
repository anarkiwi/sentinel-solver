"""Planner decisions at the hard boards' opening trap, replayed from the cached release.

Each state: on the opening robot with E=9, the best climb costs 7 and lands beside the
body it left, which is the income that pays it back.  The image is ROM memory, so
``ckpt.cached_replay`` regenerates it instead of a committed fixture.
"""

import functools

import pytest

from sentinel import actions, memmap as mm, terrain
from sentinel.playerbase import EYE_EPS, ROBOT_EYE, _Views
from sentinel.tests import ckpt

STATES = ((9760, 5), (9677, 5))
ROBOT = mm.ENERGY_IN_OBJECTS[mm.T_ROBOT]


@functools.lru_cache(maxsize=None)
def _snap(code, n_actions):
    return ckpt.cached_replay(code, n_actions)


def _player(where):
    return ckpt.restore(_snap(*where), verbose=False)


def _climbed(where):
    """The state after the trap climb: the top-scoring landable candidate, built."""
    player = _player(where)
    left = tuple(player.st.player_xy())
    views = _Views(player.st)
    _score, tile, k, cost, _gain = player._leaders(
        player._climb_candidates(views, True), True
    )[0]
    assert player.st.energy - cost < ROBOT  # the state really is the trap
    assert player._build_and_mount(views, tile, k)
    return player, left


@pytest.mark.parametrize("where", STATES)
def test_mount_never_takes_a_lower_body(where):
    """The body left behind is lower, but reading a robot's stored z as its foot made
    it look 0.875 higher, and the two bodies were each other's mount forever."""
    player, left = _climbed(where)
    st = player.st
    old = terrain.top_object(st, *left)
    assert st.obj_type[old] == mm.T_ROBOT
    assert st.eye_z(old) < st.eye_z() < st.eye_z(old) + ROBOT_EYE - EYE_EPS
    before = len(player.trace)
    assert not player._mount(_Views(st))
    assert len(player.trace) == before


@pytest.mark.parametrize("where", STATES)
def test_every_climb_predicts_the_eye_a_robot_built_there_gets(where):
    """The abandoned robot absorbed leaves its boulder as a stacked stance, whose eye
    was predicted a whole `ROBOT_EYE` high, so a descent ranked as a climb."""
    player, left = _climbed(where)
    st = player.st
    assert actions.absorb(st, terrain.top_object(st, *left))
    st.energy = mm.ENERGY_MASK  # the geometry is the subject, not the purse
    my_eye = st.eye_z()
    cands = player._climb_candidates(_Views(st), False)
    assert left in [c[1] for c in cands]
    for _score, tile, k, _cost, gain in cands:
        trial = st.clone()
        for _ in range(k):
            assert actions.create(trial, mm.T_BOULDER, tile) is not None
        slot = actions.create(trial, mm.T_ROBOT, tile)
        assert trial.eye_z(slot) == pytest.approx(my_eye + gain)
        assert gain > 0


@pytest.mark.parametrize("where", STATES)
def test_the_trap_climb_is_paid_back_by_the_body_it_left(where):
    """Played for real, the climb is followed by absorbing what it left behind, never
    by a transfer back to it, and the purse is a robot again."""
    player = _player(where)
    left = tuple(player.st.player_xy())
    start = len(player.trace)
    for _ in range(6):
        player._tick()
        if player.st.energy >= ROBOT and len(player.trace) > start + 4:
            break
    done = [(verb, tile) for _f, verb, tile, *_ in player.trace[start:]]
    assert [verb for verb, _tile in done[:4]] == ["boulder", "boulder", "robot"] + [
        "transfer"
    ]
    assert ("transfer", left) not in done
    assert ("absorb", left) in done
    assert player.st.energy >= ROBOT


def test_explain_shows_the_breakout_fork_refilling(capsys):
    """Arbitration's own view of the trap: the climb's fork ends a robot richer than
    the climb left it, having eaten the body behind it rather than mounting it."""
    where = STATES[0]
    _snap(*where)
    assert ckpt.main([str(c) for c in where]) == 0
    out = capsys.readouterr().out
    assert "climbs" in out and "option=_breakout" in out
    report = ckpt.explain(_player(where))
    assert all(c["lands"] for c in report["climbs"][:1])
    breakout = next(o for o in report["options"] if o["option"] == "_breakout")
    assert breakout["moved"] and breakout["score"][-1] >= ROBOT
    assert "transfer" not in [verb for verb, *_ in breakout["actions"][4:]]
