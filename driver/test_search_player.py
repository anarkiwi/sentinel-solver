"""The search core: move-to-program mapping, selection per machine state, fork
expansion against sequential execution, and a played line against its plan."""

import json

import pytest

from driver import kbd_prog, prog, search_player as sp
from driver.kbd_aim import A_SFLAG
from sentinel import actions, beam_player as bp, memmap as mm
from sentinel.game import Game

oracle = pytest.mark.oracle


def _facts(eye=0, energy=5, alive=1, won=0, frames=10, enemies=3):
    return {
        "status": prog.HALT_OK,
        "energy": energy,
        "slot": 1,
        "z": eye,
        "z_frac": 0,
        "enemies": enemies,
        "alive": alive,
        "won": won,
        "frames": frames,
    }


def test_identity_is_equal_exactly_where_the_machine_keys_are():
    keys = [(b"a", 1), (b"b", 1), (b"a", 1), (b"a", 2)]
    assert sp.identity(keys)[:, 0].tolist() == [0, 1, 0, 2]


def test_select_keeps_the_best_child_of_each_machine_state_and_drops_the_dead():
    player = sp.SearchPlayer(None, width=3)
    kids = [
        sp.Node(None, "s1", facts=_facts(eye=9, frames=90)),
        sp.Node(None, "s1", facts=_facts(eye=9, frames=40)),
        sp.Node(None, "s2", facts=_facts(eye=12, alive=0)),
        sp.Node(None, "s3", facts=_facts(eye=5)),
        sp.Node(None, "s4", facts=_facts(eye=1, alive=0, won=1)),
        sp.Node(None, "s5", facts=_facts(eye=5, enemies=2)),
    ]
    assert [kids.index(n) for n in player._select(kids)] == [4, 5, 1]
    assert not player._select([])


def test_a_line_runs_from_the_roots_child_down():
    root = sp.Node(None, "r")
    a = sp.Node(None, "a", root, (("wait", (0, 0), 0), None), _facts(frames=7))
    b = sp.Node(None, "b", a, (("absorb", (1, 1), 0), None), _facts(frames=5))
    assert b.line() == [a, b] and b.frames == 12 and not root.line()


def _cols(st, kind, k=0):
    """A move of ``kind`` whose first action aims on the bearing the body faces."""
    view = (st.obj_h_angle[st.player], st.obj_v_angle[st.player], 80, 70)
    return kind, k, (view,) + ((1, 2, 3, 4),) * (k + 1 if kind == bp.HOP else 0)


def test_each_move_class_maps_to_its_move_program():
    st = Game.typed(42).state
    assert sp.move_program(st, *_cols(st, bp.WAIT)) == (
        "wait",
        kbd_prog.compile_move("wait", None, 0, sp.WAIT_QUANTUM),
    )
    assert sp.move_program(st, *_cols(st, bp.RELOCATE))[0] == "relocate"
    kind, k, steps = _cols(st, bp.ABSORB)
    assert sp.move_program(st, kind, k, steps)[1] is kbd_prog.compile_move(
        "absorb", steps[0], 0, 0, False
    )
    st.mem[A_SFLAG] |= 0x80
    assert sp.move_program(st, kind, k, steps)[1] is kbd_prog.compile_move(
        "absorb", steps[0], 0, 0, True
    )
    kind, k, steps = _cols(st, bp.HOP, 1)
    assert sp.move_program(st, kind, k, steps)[1] is kbd_prog.compile_move(
        "hop", steps, 1, 0, True
    )
    actions.remove_object(st, actions.SENTINEL_SLOT)
    me = st.player
    st.obj_x[me], st.obj_y[me] = st.platform_xy
    assert sp.move_program(st, *_cols(st, bp.RELOCATE))[0] == "finish"


def test_a_move_on_the_previous_actions_bearing_drives_only_the_cursor():
    one = (16, 240, 80, 70)
    same = kbd_prog.compile_move("hop", (one, one, one), 1)
    assert same.ops == kbd_prog.compile_move("hop", one, 1).ops
    turned = kbd_prog.compile_move(
        "hop", (one, (32, 240, 80, 70), (32, 240, 81, 70)), 1
    )
    assert len(turned.ops) > len(same.ops)


# ---- on the emulated game ------------------------------------------------------


@pytest.fixture(scope="module")
def session():
    from driver.play_player import emu_session

    return emu_session(335, lambda _msg: None)


@oracle
def test_forked_children_equal_their_programs_run_one_by_one(session):
    root = sp.Node(session.bm)
    work = sp.batches([root], 2)
    ((outs, twins),) = sp.run_batches([root], work)
    assert len(outs) == len(work[0]) > 3
    for (_move, program), out, twin in zip(work[0], outs, twins):
        solo = session.bm.clone()
        assert prog.execute(solo, program) == out
        assert sp.machine_key(solo) == sp.machine_key(twin)


@oracle
def test_a_stacked_hop_aims_each_action_and_lands_on_its_robot(session):
    root = sp.Node(session.bm)
    st = sp._state(session.bm)
    (work,) = sp.batches([root], 3)
    hops = [(m, p) for m, p in work if m[0] == "hop" and m[2] > 0]
    assert hops
    (_name, tile, k), program = hops[0]
    facts = kbd_prog.move_facts(prog.execute(session.bm.clone(), program))
    for otype in (mm.T_BOULDER,) * k + (mm.T_ROBOT,):
        robot = actions.create(st, otype, tile)
    assert facts["status"] == prog.HALT_OK and facts["slot"] != st.player
    assert (facts["z"], facts["z_frac"]) == (
        st.obj_z_height[robot],
        st.obj_z_frac[robot],
    )
    assert facts["energy"] == st.energy


@oracle
def test_a_relocation_moves_the_body_for_the_toll(session):
    root = sp.Node(session.bm)
    st = sp._state(session.bm)
    (work,) = sp.batches([root], 1)
    ((_move, program),) = [(m, p) for m, p in work if m[0] == "relocate"]
    bm = session.bm.clone()
    facts = kbd_prog.move_facts(prog.execute(bm, program))
    after = sp._state(bm)
    assert facts["energy"] == st.energy - mm.ENERGY_IN_OBJECTS[mm.T_ROBOT]
    assert facts["slot"] != st.player and after.player_xy() != st.player_xy()


@oracle
def test_a_played_line_lands_where_its_plan_did_and_replays_identically():
    from driver.play_player import emu_session

    runs = []
    for _ in range(2):
        logged = []
        player = sp.SearchPlayer(
            emu_session(0, lambda _msg: None).bm, 3, 3, 2, log=logged.append
        )
        player.run(3)
        runs.append((player.trace, player.audit, sp.machine_key(player.bm)))
        assert len(player.audit) == 3 and not logged
        assert all(planned == got for _n, _m, _t, planned, got in player.audit)
    assert runs[0] == runs[1]


@oracle
def test_a_ply_workload_is_one_batch_per_beam_row():
    work = sp.ply_batches(0, 1, width=2, depth=3, branch=1)
    assert 1 <= len(work) <= 2
    for base, programs in work:
        assert programs and all(isinstance(p, prog.Prog) for p in programs)
        outs, _ = prog.forks(base, programs)
        assert all(o.code != prog.HALT_FALLBACK for o in outs)


@oracle
def test_a_live_mirror_of_the_same_game_agrees_on_every_move():
    from driver.play_player import emu_session

    live = emu_session(0, lambda _msg: None).bm
    player = sp.SearchPlayer(emu_session(0, lambda _m: None).bm, 3, 3, 2, live=live)
    logged = []
    player.log = logged.append
    assert player.run(3) == player.won
    assert [r[:2] for r in player.live_audit][:1] == [[0, "entry"]]
    assert [r[0] for r in player.live_audit] == list(range(4)) and not logged
    assert all(a == b for _n, _m, a, b in player.live_audit) and not player.diverged
    assert sp.machine_key(live) == sp.machine_key(player.bm)


@oracle
def test_a_live_mirror_that_disagrees_is_reported_and_stops_the_run(monkeypatch):
    from driver.play_player import emu_session

    live = emu_session(0, lambda _msg: None).bm
    logged = []
    player = sp.SearchPlayer(
        emu_session(0, lambda _m: None).bm, 3, 3, 2, 3, logged.append, live
    )
    search = player.search

    def tampered(root, desc):
        best = search(root, desc)
        live.m.ram[mm.PLAYER_ENERGY] ^= 1
        return best

    monkeypatch.setattr(player, "search", tampered)
    assert not player.run(5)
    assert len(player.trace) == 1 and len(player.live_audit) == 2
    assert player.diverged["move"] == 1 and "energy" in player.diverged["facts"]
    assert logged and logged[-1].startswith("LIVE MISMATCH at move 1")


@oracle
def test_a_live_mirror_that_differs_at_entry_commits_nothing():
    from driver.play_player import emu_session

    live = emu_session(0, lambda _msg: None).bm
    live.m.ram[mm.PLAYER_ENERGY] ^= 1
    player = sp.SearchPlayer(
        emu_session(0, lambda _m: None).bm, live=live, log=[].append
    )
    assert not player.run(3) and not player.trace
    assert player.diverged["name"] == "entry"


@oracle
def test_the_live_search_records_its_mirror_audit_and_the_rom_win_flag(
    tmp_path, monkeypatch
):
    from driver import core, play_player

    def boot_and_play(_tap, _renders, digits, _video, log, play_fn, result):
        play_fn(play_player.emu_session(int(digits), log))
        result["video"] = "none.avi"
        return result

    monkeypatch.setattr(core, "boot_and_play", boot_and_play)
    out = tmp_path / "run.json"
    args = "0 --player search --width 2 --depth 2 --branch 1 --max-actions 2 --out"
    assert play_player.main(args.split() + [str(out)]) == 1
    run = json.loads(out.read_text())
    assert [r[0] for r in run["live_audit"]] == [0, 1, 2] and run["divergence"] is None
    assert run["landscape_done"] == 0 and run["won"] is False


def _children(bm, branch=1):
    player = sp.SearchPlayer(bm.clone(), 2, 1, branch, log=[].append)
    root = sp.Node(player.bm)
    return player, root, player._expand([root], {})


@oracle
def test_an_executed_move_that_lands_elsewhere_than_planned_is_reported(session):
    player, _root, kids = _children(session.bm)
    planned = next(n for n in kids if n.move[0] == "wait")
    planned.facts = {**planned.facts, "energy": planned.facts["energy"] + 1}
    logged = []
    player.log = logged.append
    player._execute(planned)
    assert player.audit[0][3] != player.audit[0][4] and "!=" in logged[0]


@oracle
def test_an_executed_relocation_records_where_the_body_landed(session):
    player, _root, kids = _children(session.bm)
    planned = next(n for n in kids if n.move[0] == "relocate")
    player._execute(planned)
    st = sp._state(player.bm)
    assert player.trace[0][1] == "relocate"
    assert tuple(player.trace[0][2]) == tuple(map(int, st.player_xy()))
    assert sp.machine_key(player.bm) == planned.key


@oracle
def test_a_search_with_no_surviving_child_stops(session, monkeypatch):
    logged = []
    player = sp.SearchPlayer(session.bm.clone(), 2, 2, 1, log=logged.append)
    monkeypatch.setattr(player, "_select", lambda kids: [])
    assert not player.run(3) and not player.trace
    assert logged == ["search: no surviving move"]


@oracle
def test_the_workload_cli_times_each_program_of_a_ply(capsys):
    assert sp.main(["0", "0", "--preset", "narrow", "--each"]) == 0
    rows = json.loads(capsys.readouterr().out.strip().splitlines()[-1])
    assert rows and all(len(r["moves"]) == 1 and r["frames"] > 0 for r in rows)
