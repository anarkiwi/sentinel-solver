"""The emulated phase player's lookahead runs on machine clones: a rehearsal predicts
the real action exactly, leaves the real machine alone, and waits spend machine time."""

import pytest

from driver import core, emu_phase, emu_suite
from driver.emu_phase import EmuPhase, Rehearsal, _futile, _hop_steps
from driver.play_player import emu_session
from sentinel import memmap as mm, phase_player
from sentinel.playerbase import _Views

oracle = pytest.mark.oracle


def _quiet(_msg):
    pass


@pytest.fixture(autouse=True)
def _fresh_memos():
    """An identical machine in another test would answer from the shared memos."""
    emu_phase._REHEARSALS.clear()
    phase_player._LANDING_CACHE.clear()


def _player(number=0):
    result = {"actions": [], "energy_curve": []}
    player = EmuPhase(emu_session(number, _quiet), _quiet, result)
    player._observe()
    return player


def _absorb_target(player):
    views = _Views(player.st)
    for value, tile in sorted(
        player._reclaim_targets(player.st, True), key=lambda vt: -vt[0]
    ):
        view = views.get(tuple(tile), band=True)
        if view is not None:
            return tuple(tile), view, value
    raise AssertionError("no absorbable target in view")


def _bare_tile(player):
    for tile, view in _Views(player.st).primary().items():
        if player._top(tile) is None and player._stance_base(tile) is not None:
            return tile, view
    raise AssertionError("no bare tile in view")


@oracle
def test_a_rehearsal_predicts_the_real_action_exactly():
    """A scripted absorb, robot and transfer: each fork's energy, death and frames are
    what the real machine then does, and the actions really happened."""
    player = _player()
    tile, view, value = _absorb_target(player)
    before = player.st.energy
    assert player._fire("absorb", tile, view)
    assert player.st.energy == before + value
    tile, view = _bare_tile(player)
    assert player._fire("robot", tile, view)
    assert player._fire("transfer", tile, _Views(player.st).get(tile, band=True))
    assert tuple(player.st.player_xy()) == tile
    rows = player.result["lookahead_audit"]
    assert [r[1] for r in rows] == ["absorb", "robot", "transfer"]
    for _step, _verb, _tile, want, got in rows:
        assert want == got and want[0] and want[3] > 0


@oracle
def test_a_rehearsal_leaves_the_real_machine_alone_and_is_memoized():
    player = _player()
    tile, view, _value = _absorb_target(player)
    key = player._machine_key()
    image = core.live_image(player.bm)
    forks = []
    fork = player._fork
    player._fork = lambda: forks.append(1) or fork()
    first = player._rehearse((("absorb", tile, view),))
    assert player._rehearse((("absorb", tile, view),)) is first and len(forks) == 1
    assert player._machine_key() == key and core.live_image(player.bm) == image
    assert first.ok and not first.drained


@oracle
def test_waiting_runs_the_real_machine_and_rereads_it():
    player = _player()
    f0, cycle = player.ex.frames(), player.bm.m.cycle
    player._wait(phase_player.WAIT_QUANTUM)
    assert player.ex.frames() - f0 == phase_player.WAIT_QUANTUM
    assert player.bm.m.cycle > cycle
    assert bytes(player.st.mem) == bytes(core.live_image(player.bm))
    assert player.result["wait_audit"][-1][1:] == [phase_player.WAIT_QUANTUM] * 2


@oracle
def test_a_frozen_world_drains_nothing_while_idling():
    """$0CE5: nothing moves before the first action, so a forked idle loses nothing
    and the real machine has not moved."""
    player = _player()
    assert player.st.mem[mm.PLAYER_NOT_ACTED] & 0x80
    cycle = player.bm.m.cycle
    assert not player._idle_drains(phase_player.WAIT_QUANTUM)
    assert player.bm.m.cycle == cycle


@oracle
def test_a_rollout_adopts_the_machine_its_rehearsal_left():
    """Adopting the rehearsal's fork is the same machine as firing for real."""
    player = _player()
    tile, view, _value = _absorb_target(player)
    adopt, fire = player._fork(), player._fork()
    adopt._rehearse((("absorb", tile, view),))
    rehearsed = adopt._last[1].bm
    assert adopt._fire("absorb", tile, view) and adopt.bm is rehearsed
    assert fire._fire("absorb", tile, view) and fire.bm is not rehearsed
    assert adopt._machine_key() == fire._machine_key()
    assert adopt.trace[-1][1:] == fire.trace[-1][1:]


@oracle
def test_landing_and_mount_probes_play_the_hop_on_a_fork():
    player = _player()
    tile, _view = _bare_tile(player)
    jobs = []
    run = player._fork_jobs
    player._fork_jobs = lambda js: jobs.append(len(js)) or run(js)
    cycle = player.bm.m.cycle
    holds = player._landing_holds(tile, 0)
    assert player._landing_holds(tile, 0) is holds and jobs == [1]
    assert player.bm.m.cycle == cycle
    assert holds is True  # a frozen world cannot drain the landing
    assert _hop_steps(tile, 2)[0] == ("boulder", tuple(tile), None)
    assert player._mount_holds(player.st.player) is False  # no transfer into oneself


def test_a_refused_or_fatal_rehearsal_is_not_fired():
    assert not _futile(Rehearsal(True, 5, False, 300, False))
    assert not _futile(Rehearsal(False, 4, False, 300, True))  # landed, then drained
    assert _futile(Rehearsal(False, 5, False, 300, False))  # refused outright
    assert _futile(Rehearsal(True, 10, True, 9000, True))  # the body was lost


def test_the_suite_table_sets_each_run_beside_its_offline_record(tmp_path, monkeypatch):
    run = {"won": True, "trace": [[0]] * 3, "final_energy": 4, "wall_seconds": 9.5}
    run["lookahead_audit"] = [
        [1, "absorb", [0, 0], [1], [1]],
        [2, "robot", [0, 0], [1], [2]],
    ]
    lines = list(
        emu_suite.rows({7: run}, {"7": {"won": True, "actions": 5, "energy": 2}})
    )
    assert lines[1].split() == "7 True 3 4 9.5 1/2 | True 5 2".split()

    def fake_run(cmd, **_kw):
        if cmd[3] == "1":
            with open(cmd[-1], "w") as fh:
                fh.write('{"won": false, "trace": []}')

    monkeypatch.setattr(emu_suite.subprocess, "run", fake_run)
    assert emu_suite.play(1, str(tmp_path), 5) == {"won": False, "trace": []}
    assert emu_suite.play(2, str(tmp_path), 5)["won"] == "error"
