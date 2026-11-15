"""Driver programs: the native backend equals the BinMon interpreter, and each compiled
keyboard primitive and step segment equals the interactive driver on the emulated game.
"""

import hashlib
import struct
import threading

import pytest
from vagrant_moses import Machine
from vice_driver.binmon import OPCODE, BinmonError, BinmonResponse

from driver import clock, kbd_prog, prog
from driver.emu import EmuMon
from driver.kbd_prog import Compiler
from sentinel import memmap as mm
from sentinel.views import _Views
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


def _state(bm):
    """Everything a later step depends on: machine, monitor and frame count."""
    return _machine(bm), bm._owed, bm._parked, clock.frames(bm)


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


def _key(view):
    return (view["h_angle"], view["v_angle"], *map(int, view["cursor"]))


def _target(bm):
    st = State.from_mem(bytes(bm.mem_get(0, 0xFFFF)))
    views = _Views(st)
    for slot in st.occupied_slots():
        tile = tuple(st.tile_of(slot))
        view = views.get(tile, band=True)
        if st.obj_type[slot] == mm.T_TREE and view is not None:
            return tile, view
    raise AssertionError("no tree in view")


@oracle
def test_a_move_program_reports_the_facts_a_search_needs(session):
    _tile, view = _target(session.bm)
    key = _key(view)
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
            "seen": st.mem[mm.EXPOSURE_BAR] >> 6,
            "away": (st.player_xy()[0] ^ st.platform_xy[0])
            | (st.player_xy()[1] ^ st.platform_xy[1]),
        }
    assert kbd_prog.move_facts(outs[0])["energy"] == st0.energy + 1
    assert kbd_prog.move_facts(outs[1])["frames"] == 50


@oracle
def test_a_hop_move_lands_the_body_on_its_robot(session):
    st = State.from_mem(bytes(session.bm.mem_get(0, 0xFFFF)))
    taken = {tuple(st.tile_of(s)) for s in st.occupied_slots()}
    tile, view = next((t, v) for t, v in _Views(st).primary().items() if t not in taken)
    (o,), (twin,) = prog.forks(session.bm, [kbd_prog.compile_move("hop", _key(view))])
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


class _Wire:
    """BinMon's socket side: the EXIT it was sent and the events then queued."""

    def __init__(self, events):
        self._lock = threading.Lock()
        self._sock = self
        self.events, self.sent, self.timeout = list(events), [], None

    def gettimeout(self):
        return self.timeout

    def settimeout(self, t):
        self.timeout = t

    def _next_req(self):
        return 7

    def _send(self, opcode, body, req):
        self.sent.append((opcode, body, req))

    def _read_response(self):
        if not self.events:
            raise BinmonError("socket timed out")
        return self.events.pop(0)


def _event(opcode, body=b"", req=0xFFFFFFFF):
    return BinmonResponse(opcode=opcode, err=0, req_id=req, body=body)


def _hit(num, hit=1):
    return _event(0x11, struct.pack("<IB", num, hit) + bytes(18))


def test_a_binmon_wait_returns_at_the_stop_of_any_of_its_checkpoints():
    stopped = _event(OPCODE.STOPPED)
    wire = _Wire([_event(0xAA, req=7), stopped, _hit(9), _hit(3, 0), _hit(4), stopped])
    assert prog.wait_any(wire, (3, 4), 2.0) == {4}
    assert wire.sent == [(OPCODE.EXIT, b"", 7)] and wire.timeout is None
    with pytest.raises(BinmonError):
        prog.wait_any(_Wire([_hit(9), stopped]), (3, 4), 2.0)


def test_an_emumon_wait_on_checkpoints_that_never_hit_raises():
    bm = _toy()
    here = bm.checkpoint_set(clock.FRAME_PC, stop_when_hit=True).checknum
    there = bm.checkpoint_set(0x1234, stop_when_hit=True).checknum
    assert prog.wait_any(bm, (there, here), 1.0) == {here}
    bm.advance_instructions(1)
    with pytest.raises(BinmonError, match="stopped elsewhere"):
        prog.wait_any(bm, (there,), 1.0)


def test_a_clones_stop_checkpoints_are_its_own():
    bm = _toy()
    shared = prog._stop_checkpoint(bm, 0x9633)
    twin = bm.clone()
    assert prog._stop_checkpoint(twin, 0x9633) == shared
    mine = prog._stop_checkpoint(twin, 0x1234)
    assert prog._stop_checkpoint(bm, 0x1234) == mine
    assert bm.checkpoint_get(mine).start == twin.checkpoint_get(mine).start == 0x1234
