"""Driver programs: the native backend equals the BinMon interpreter, and each compiled
keyboard primitive and step segment equals the interactive driver on the emulated game.
"""

import hashlib

import pytest
from vagrant_moses import Machine

from driver import clock, kbd_prog, prog, sentinel_execute as sx
from driver.emu import EmuMon
from driver.kbd_prog import Compiler, ProgKbd
from driver.live_player import MeasuringKbdDriver
from sentinel import memmap as mm
from sentinel.playerbase import _Views
from sentinel.state import State

R = prog.R
oracle = pytest.mark.oracle
BUDGET = kbd_prog.budget(1, 0)
LOOP = bytes([0x78, 0xEE, 0x00, 0xC0, 0x4C, 0x30, 0x96])  # SEI; INC $C000; JMP $9630


def _quiet(_msg):
    pass


def _toy():
    """A monitor on a bare machine looping through the frame marker $9630."""
    m = Machine(backend="native")
    m.ram[clock.FRAME_PC - 1 : clock.FRAME_PC - 1 + len(LOOP)] = LOOP
    m.reset(clock.FRAME_PC - 1)
    bm = EmuMon(m)
    clock.frames(bm)
    return bm


def _machine(bm):
    m = bm.m
    regs = (m.mpu.a, m.mpu.x, m.mpu.y, m.mpu.pc, m.mpu.sp, m.mpu.p)
    return hashlib.blake2b(bytes(m.ram)).digest(), m.cycle, regs, bytes(m.keymatrix)


def _state(bm, kbd=None):
    """Everything a later step depends on: machine, monitor and timing records."""
    got = _machine(bm), bm._owed, bm._parked, clock.frames(bm)
    return got if kbd is None else got + (dict(kbd.subframes), list(kbd.passes))


def _both(bm, p):
    """``(native, interpreted)`` outcomes and end states of ``p`` from ``bm``."""
    a, b = bm.clone(), bm.clone()
    return (prog.execute(a, p), _state(a)), (prog.interpret(b, p), _state(b))


def test_ops_agree_between_native_and_interpreter():
    p = prog.Prog(BUDGET)
    loop, hit, end = p.new(3)
    with p.regs(3) as (a, b, n):
        p.mov(a, 0xF0).add8(a, 0x20).sub(a, 1).shl(a, 4).shr(a, 2).or_(a, 0x100)
        p.xor(a, 3).and_(a, 0xFFF).out(a).mov(n, 5).label(loop).step().djnz(n, loop)
        p.run(0x9633, 1.0).ld(b, 0xC000).out(b).ldio(b, 0xD012).out(b)
        p.ld(b, 0xC000, a).out(b).frames(b).out(b)
        p.key(1, 2).key(1, 2, False).key(3, 4)
        p.mov(b, 0x81).bmi8(b, hit).halt(9).label(hit).bpl8(b, end).sub8(b, 0x82)
        p.blt(b, 1, end).bge(b, 0x100, end).bne(b, 0xFF, end).beq(b, 0xFF, end)
        p.label(end).rec(7, b).clear()
    (no, ns), (io, is_) = _both(_toy(), p)
    assert no == io and ns == is_
    assert no.code == prog.HALT_OK and no.outputs[0] == ((0xF0 >> 2) | 0x100) ^ 3
    assert no.outputs[-2:] == (7, 0xFF)


def test_a_wait_past_its_timeout_replays_on_the_monitor():
    """The native run overshoots the timeout; the replay times out where the monitor
    does, takes the handler and clears the keys."""
    p = prog.Prog(BUDGET)
    caught = p.new()
    with p.regs(1) as (a,):
        with p.catch(caught):
            p.key(0, 1).run(0x1234, 0.001)
        p.halt(5).label(caught).frames(a).out(a)
    before = dict(prog.STATS)
    (no, ns), (io, is_) = _both(_toy(), p)
    assert prog.STATS["fallback"] == before["fallback"] + 1
    assert no == io and ns == is_ and no.code == prog.HALT_OK and no.frames > 0
    p = prog.Prog(BUDGET)
    p.run(0x1234, 0.001)
    (no, _), (io, _) = _both(_toy(), p)
    assert no.code == io.code == prog.HALT_RAISE


def test_a_wait_already_at_its_pc_is_skipped_until_the_machine_runs():
    bm = _toy()
    bm.advance_instructions(2)
    assert bm.m.mpu.pc == clock.FRAME_PC
    p = prog.Prog(BUDGET).run(clock.FRAME_PC, 1.0).run(clock.FRAME_PC, 1.0)
    (no, ns), (io, is_) = _both(bm, p)
    assert no == io and ns == is_ and no.frames == 0
    p = (
        prog.Prog(BUDGET)
        .step()
        .run(clock.FRAME_PC, 1.0)
        .step()
        .run(clock.FRAME_PC, 1.0)
    )
    (no, ns), (io, is_) = _both(bm, p)
    assert no == io and ns == is_ and no.frames == 3  # the owed test, then two frames


def test_branch_in_matches_set_membership():
    for values in ([], [0], [255], [3, 4, 5, 9, 200, 201], list(range(0, 256, 3))):
        c = Compiler()
        p = c.p
        with p.regs(1) as (r,):
            for v in range(256):
                yes, end = p.new(2)
                p.mov(r, v)
                c.branch_in(r, values, yes)
                p.rec(0).jmp(end).label(yes).rec(1).label(end)
        got = prog.interpret(_toy(), p).outputs
        assert got == tuple(int(v in values) for v in range(256))


def test_rec_refuses_its_own_staging_register():
    with pytest.raises(ValueError):
        prog.Prog(BUDGET).rec(1, prog.SCRATCH)
    with pytest.raises(AttributeError):
        prog.Prog(BUDGET).nope()


def test_forks_equal_sequential_runs():
    bm = _toy()
    progs = [prog.Prog(BUDGET).step().run(clock.FRAME_PC, 1.0)] * 3 + [
        prog.Prog(BUDGET).run(0x1234, 0.001)
    ]
    outs, twins = prog.forks(bm, progs)
    for p, o, twin in zip(progs, outs, twins):
        solo = bm.clone()
        assert prog.execute(solo, p) == o and _state(solo) == _state(twin)
    assert _machine(bm) == _machine(_toy())


# ---- on the emulated game ------------------------------------------------------


@pytest.fixture(scope="module")
def session():
    from driver.play_player import (
        emu_session,
    )  # pylint: disable=import-outside-toplevel

    return emu_session(0, _quiet)


def _drivers(bm):
    """``(interactive, compiled)`` measuring drivers on two clones of ``bm``."""
    a, b = bm.clone(), bm.clone()
    clock.frames(a)
    clock.frames(b)
    return (
        MeasuringKbdDriver(a, _quiet, quantized=True, ex=sx.Executor(a, _quiet)),
        ProgKbd(b, _quiet, sx.Executor(b, _quiet)),
    )


def _same(bm, act):
    ka, kb = _drivers(bm)
    ra, rb = act(ka), act(kb)
    assert ra == rb
    assert _state(ka.bm, ka) == _state(kb.bm, kb)
    return ra, kb


PRIMITIVES = {
    "sights_toggle": lambda k: (k.sights_set(not k.sights_live_on()), k.sights_on()),
    "coarse_h": lambda k: (k.sights_set(False), k.coarse_h(k.hang() + 0x28)),
    "uturn": lambda k: (k.sights_set(False), k.coarse_h(k.hang() ^ 0x88)),
    "coarse_v": lambda k: (k.sights_set(False), k.coarse_v(0xF1), k.coarse_v(0x80)),
    "coarse_hv": lambda k: (
        k.sights_set(False),
        k.coarse_hv({"h_angle": k.hang() - 16, "v_angle": 0x05}),
    ),
    "fine_cursor": lambda k: (k.sights_on(), k.fine_cursor(100, 60)),
    "tap_action": lambda k: (k.sights_on(), k.tap_action("A"), k.tap_action("U")),
    "hyperspace_tap": lambda k: k.tap_action("H", settle=False),
    "run_frames": lambda k: (
        k.run_frames(7) if isinstance(k, ProgKbd) else clock.run_frames(k.bm, 7)
    ),
}


@oracle
@pytest.mark.parametrize("name", sorted(PRIMITIVES))
def test_each_primitive_equals_the_interactive_driver(session, name):
    _same(session.bm, PRIMITIVES[name])


def _target(bm):
    st = State.from_mem(bytes(bm.mem_get(0, 0xFFFF)))
    views = _Views(st)
    for slot in st.occupied_slots():
        tile = tuple(st.tile_of(slot))
        view = views.get(tile, band=True)
        if st.obj_type[slot] == mm.T_TREE and view is not None:
            return tile, view
    raise AssertionError("no tree in view")


def _interactive_step(kbd, view, key):
    """perform_step's aim and fire as the interactive driver ran them."""
    kbd.sights_set(False)
    okh, okv, status = kbd.coarse_hv(view)
    if status == "hyperspace":
        return None
    hv = kbd.hang(), kbd.vang()
    kbd.sights_on()
    okc = kbd.fine_cursor(*view["cursor"])
    probe = (
        kbd.probe_ray()
        if isinstance(kbd, ProgKbd)
        else __import__("driver.core").core.probe_tile(kbd.bm)
    )
    return okh, okv, hv, okc, probe, kbd.cur(), kbd.sights_on(), kbd.tap_action(key)


@oracle
def test_a_step_program_equals_the_interactive_primitives(session):
    tile, view = _target(session.bm)
    ka, kb = _drivers(session.bm)
    want = _interactive_step(ka, view, "A")
    got = kb.step(kbd_prog.view_key(view), ("A", None))
    probe = kb.probe_ray()
    assert got.halt == prog.HALT_OK and probe[:2] == tile == want[4][:2]
    assert (
        kbd_prog.STATUS[got[kbd_prog.V_OKH]],
        kbd_prog.STATUS[got[kbd_prog.V_OKV]],
        (got[kbd_prog.V_ACH_H], got[kbd_prog.V_ACH_V]),
        bool(got[kbd_prog.V_OKC]),
        (got[kbd_prog.V_CX], got[kbd_prog.V_CY]),
        bool(got[kbd_prog.V_LATCHED]),
    ) == (*want[:4], want[5], want[7])
    assert _state(ka.bm, ka) == _state(kb.bm, kb)


@oracle
def test_split_and_merged_steps_and_the_interpreter_agree(session):
    _tile, view = _target(session.bm)
    key = kbd_prog.view_key(view)
    whole = kbd_prog.compiled("step", key, ("A", None))
    (no, ns), (io, is_) = _both(session.bm, whole)
    assert no == io and ns == is_
    bm = session.bm.clone()
    aim, fire = (
        kbd_prog.compiled("step", *a) for a in ((key, None), (None, ("A", None)))
    )
    first = prog.execute(bm, aim)
    second = prog.execute(bm, fire)
    assert _state(bm) == ns
    assert (
        first.outputs + second.outputs != no.outputs
    )  # frame reads restart per program
    assert first.frames + second.frames == no.frames


@oracle
def test_perform_step_reuses_the_aim_and_verifies(session):
    tile, view = _target(session.bm)
    bm = session.bm.clone()
    ex = sx.Executor(bm, _quiet)
    kbd = ProgKbd(bm, _quiet, ex)
    result = {"actions": [], "energy_curve": []}
    stp = {"verb": "absorb", "otype": mm.T_TREE, "target": list(tile), "view": view}
    assert sx.perform_step(ex, kbd, "p1", stp, _quiet, result) == "ok"
    assert result["actions"][-1]["aim"]["probe"][:2] == tile
    assert kbd.committed_bearing() == (view["h_angle"], view["v_angle"])
    stp = {"verb": "create", "otype": mm.T_ROBOT, "target": list(tile), "view": view}
    stp["min_energy"] = 64
    assert sx.perform_step(ex, kbd, "p2", stp, _quiet, result) == "drained"
    kbd.advisory = False
    stp["min_energy"] = 0
    assert sx.perform_step(ex, kbd, "p3", stp, _quiet, result) == "ok"
    assert [a[1] for a in result["settle_audit"]] == ["absorb", "create"]


@oracle
def test_forked_step_programs_equal_sequential_ones(session):
    _tile, view = _target(session.bm)
    key = kbd_prog.view_key(view)
    progs = [
        kbd_prog.compiled("step", key, ("A", None)),
        kbd_prog.compiled("cursor_probe", *view["cursor"]),
        kbd_prog.compiled("wait", 30),
        kbd_prog.compile_move("absorb", key),
    ]
    outs, twins = prog.forks(session.bm, progs)
    for p, o, twin in zip(progs, outs, twins):
        solo = session.bm.clone()
        assert prog.execute(solo, p) == o and _state(solo) == _state(twin)


@oracle
def test_a_move_program_reports_the_facts_a_search_needs(session):
    _tile, view = _target(session.bm)
    key = kbd_prog.view_key(view)
    st0 = State.from_mem(bytes(session.bm.mem_get(0, 0xFFFF)))
    outs, twins = prog.forks(
        session.bm,
        [
            kbd_prog.compile_move("absorb", key),
            kbd_prog.compile_move("wait", None, 0, 50),
        ],
    )
    for o, twin in zip(outs, twins):
        facts = kbd_prog.move_facts(o)
        st = State.from_mem(bytes(twin.mem_get(0, 0xFFFF)))
        assert facts == {
            "status": prog.HALT_OK,
            "energy": st.energy,
            "slot": st.player,
            "z": st.obj_z_height[st.player],
            "z_frac": st.obj_z_frac[st.player],
            "enemies": len(
                [s for s in st.occupied_slots() if st.obj_type[s] in mm.ENEMY_TYPES]
            ),
            "alive": int(
                not st.mem[mm.PLAYER_DIED_BY_DRAINING] & 0x80
                and not st.mem[mm.PLAYER_NOT_ACTED] & 0x80
            ),
            "won": 0,
            "frames": o.frames,
        }
    assert kbd_prog.move_facts(outs[0])["energy"] == st0.energy + 1
    assert kbd_prog.move_facts(outs[1])["frames"] == 50


@oracle
def test_drive_transfer_aim_probes_through_programs(session):
    from driver.live_player import (
        drive_transfer_aim,
    )  # pylint: disable=import-outside-toplevel

    tile, view = _target(session.bm)
    kbd = ProgKbd(session.bm.clone(), _quiet, None)
    kbd._ex = sx.Executor(kbd.bm, _quiet)
    assert drive_transfer_aim(kbd, tile, view, _quiet)
    assert drive_transfer_aim(kbd, tile, view, _quiet)  # the reused aim
    assert not drive_transfer_aim(kbd, (tile[0] ^ 1, tile[1]), view, _quiet)
    done, stopped = kbd.hyperspace_attempt()
    assert not stopped and not done & 0x40


@oracle
def test_a_hop_move_lands_the_body_on_its_robot(session):
    st = State.from_mem(bytes(session.bm.mem_get(0, 0xFFFF)))
    taken = {tuple(st.tile_of(s)) for s in st.occupied_slots()}
    tile, view = next((t, v) for t, v in _Views(st).primary().items() if t not in taken)
    (o,), (twin,) = prog.forks(
        session.bm, [kbd_prog.compile_move("hop", kbd_prog.view_key(view))]
    )
    facts = kbd_prog.move_facts(o)
    after = State.from_mem(bytes(twin.mem_get(0, 0xFFFF)))
    assert facts["status"] == prog.HALT_OK and facts["slot"] != st.player
    assert tuple(after.tile_of(facts["slot"])) == tile and facts["alive"]
    assert facts["energy"] == st.energy - mm.ENERGY_IN_OBJECTS[mm.T_ROBOT]


def test_a_guarded_wait_stops_before_a_guard_pc_on_both_backends():
    """A wait for a pc that never comes ends at the guard pc instead of timing out."""
    p = prog.Prog(BUDGET)
    hit = p.new()

    def check(q):
        q.ld(prog.GUARD, 0xC000).out(prog.GUARD).jmp(hit)

    with p.guard((0x9633,), check):
        p.run(0x1234, 1.0)
    p.halt(5).label(hit).halt(6)
    before = dict(prog.STATS)
    (no, ns), (io, is_) = _both(_toy(), p)
    assert no == io and ns == is_ and no.code == 6
    assert prog.STATS["fallback"] == before["fallback"]
    stopped = _toy()
    before = stopped.m.ram[0xC000]
    assert prog.execute(stopped, p).outputs == ((before + 1) & 0xFF,)
    assert stopped.m.mpu.pc == 0x9633
