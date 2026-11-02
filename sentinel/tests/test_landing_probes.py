"""The landing probe is asked lazily and memoized: neither may change a decision."""

import functools
import inspect

import pytest

from sentinel import batch, engine, kern, phase_player
from sentinel.game import Game
from sentinel.phase_player import CLOCK_FIELDS, LOS_TRACKERS, PhasePlayer
from sentinel.playerbase import _Views
from sentinel.state import State

MID_GAME = ((335, 0), (335, 1), (335, 2), (335, 8), (42, 5), (42, 7))


@functools.lru_cache(maxsize=None)
def _solve(code):
    """Forks of the phase player's fixed-ladder play of ``code`` at each of its MID_GAME
    ticks: real positions, reached without arbitration's forks."""
    player = PhasePlayer(Game.typed(code), verbose=False, rollout=True)
    out, done = {}, 0
    for ticks in sorted(t for c, t in MID_GAME if c == code):
        player.run(max_actions=ticks - done)
        done = ticks
        out[ticks] = player._fork()
    return out


def _position(code, ticks):
    return _solve(code)[ticks]


def _eager(player, views):
    """The reference: probe every candidate, uncached, then take the top score."""
    held = [
        c
        for c in player._climb_candidates(views, True)
        if player._probe_landing(c[1], c[2])
    ]
    if not held:
        return []
    top = max(c[0] for c in held)
    return [c for c in held if c[0] == top]


@pytest.mark.parametrize("where", MID_GAME)
def test_lazy_probing_returns_what_eager_probing_does(where):
    player = _position(*where)
    views = _Views(player.st)
    want = _eager(player, views)
    phase_player._LANDING_CACHE.clear()
    cands = player._climb_candidates(views, True)
    assert player._leaders(cands, True) == want
    phase_player._LANDING_CACHE.clear()
    assert player._any_climb(views) == bool(want)


def test_the_positions_exercise_the_lazy_walk():
    """ls335 tick 1 has candidates and no landing; at tick 2 the best-scoring
    candidate fails its landing, so the walk goes past it to a tie."""
    stuck = _position(335, 1)
    assert stuck._climb_candidates(_Views(stuck.st), True)
    assert not _eager(stuck, _Views(stuck.st))
    player = _position(335, 2)
    views = _Views(player.st)
    want = _eager(player, views)
    assert want[0][0] < max(c[0] for c in player._climb_candidates(views, True))
    assert len(want) > 1


@pytest.mark.parametrize("backend", ("cpu", "cuda"))
def test_a_lazy_ask_probes_no_more_than_the_backend_walks(monkeypatch, backend):
    """Existence stops at the first landable candidate in band order.  The CPU probes
    exactly the candidates the serial walk visits; the cuda backend probes whole
    windows, and a re-ask on the same state, or on a fork of it, runs nothing."""
    monkeypatch.setattr(batch, "width", lambda: 4)
    player = _position(335, 2)
    views = _Views(player.st)
    cands = player._climb_candidates(views, True)
    first = next(i for i, c in enumerate(cands) if player._probe_landing(c[1], c[2]))
    asked = []
    verdicts = PhasePlayer._probe_verdicts

    def counted(self, group):
        asked.extend(c[1] for c in group)
        return verdicts(self, group)

    monkeypatch.setattr(PhasePlayer, "_probe_verdicts", counted)
    monkeypatch.setattr(kern, "backend", lambda: backend)
    monkeypatch.setattr(engine, "default_device", lambda rows: False)
    phase_player._LANDING_CACHE.clear()
    assert player._any_climb(views)
    walked = [c[1] for c in cands[: first + 1]]
    if backend == "cpu":
        assert asked == walked
    else:
        assert asked[: len(walked)] == walked and len(asked) < first + 1 + 4
    asked.clear()
    assert player._any_climb(views)
    assert player._fork()._landing_holds(cands[first][1], cands[first][2])
    assert not asked


@pytest.mark.parametrize("where", MID_GAME)
def test_batched_probes_fill_the_cache_with_the_serial_verdicts(where):
    player = _position(*where)
    cands = player._climb_candidates(_Views(player.st), True)
    want = [player._probe_landing(c[1], c[2]) for c in cands]
    phase_player._LANDING_CACHE.clear()
    player._probe_landings(cands)
    got = [player._landing_key(c[1], c[2]) for c in cands]
    assert [phase_player._LANDING_CACHE[k] for k in got] == want


@pytest.mark.parametrize("where", ((335, 2), (42, 5)))
def test_no_landing_depends_on_the_los_trackers(where):
    """The memo key leaves out the bytes pricing rewrites on the live state, so the
    probe's verdict must be the same whatever they hold."""
    player = _position(*where)
    cands = player._climb_candidates(_Views(player.st), True)
    verdicts = set()
    for fill in (0x00, 0x55, 0xAA, 0xFF):
        twin = player._fork()
        for addr in LOS_TRACKERS:
            twin.st.mem[addr] = fill
        assert twin._landing_key(*cands[0][1:3]) == player._landing_key(*cands[0][1:3])
        verdicts.add(tuple(twin._probe_landing(c[1], c[2]) for c in cands))
    assert len(verdicts) == 1


def test_the_clock_fields_are_every_resume_field_a_state_carries():
    params = set(inspect.signature(State.__init__).parameters) - {"self", "mem"}
    assert set(CLOCK_FIELDS) == params


@pytest.mark.parametrize("field", CLOCK_FIELDS)
def test_the_memo_key_changes_with_every_enemy_clock_field(field):
    player = PhasePlayer(Game.typed(0), verbose=False)
    tile = (4, 4)
    key = player._landing_key(tile, 1)
    assert PhasePlayer(Game(player.st.clone()))._landing_key(tile, 1) == key
    setattr(player.st, field, getattr(player.st, field) + 1)
    assert player._landing_key(tile, 1) != key


def test_the_memo_key_covers_memory_aim_and_hop():
    player = PhasePlayer(Game.typed(0), verbose=False)
    tile = (4, 4)
    seen = {player._landing_key(tile, 1)}
    seen.add(player._landing_key(tile, 2))
    seen.add(player._landing_key((4, 5), 1))
    player.cursor[0] += 1
    seen.add(player._landing_key(tile, 1))
    player.last_bearing = (1, 2)
    seen.add(player._landing_key(tile, 1))
    player.st.mem[0x8000] ^= 1
    seen.add(player._landing_key(tile, 1))

    class _Sub(PhasePlayer):
        pass

    sub = _Sub(Game(player.st.clone()))
    sub.cursor, sub.last_bearing = list(player.cursor), player.last_bearing
    seen.add(sub._landing_key(tile, 1))
    assert len(seen) == 7
