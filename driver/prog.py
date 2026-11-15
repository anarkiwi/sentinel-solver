#!/usr/bin/env python3
"""Driver programs: keyboard-driver waits, reads and presses as one register program.

The ops are vagrant-moses driver-program ops plus a timed wait; :func:`interpret` runs them
call by call over any BinMon, :func:`execute` natively on an ``EmuMon``'s machine and
:func:`forks` on ``run_forks`` forks of one, replaying on the monitor any timed-out wait.
"""

import contextlib
import struct
import time

from vagrant_moses.program import Program, R, Reg, run_forks
from vice_driver.binmon import OPCODE, BinmonError

from driver import clock
from driver.emu import EmuMon, cycles
from sentinel.passcost import PAL_FRAME_CYCLES

HALT_OK, HALT_RAISE, HALT_FALLBACK = 0, 1, 0xFFFF
SCRATCH = R[11]  # immediate staging for :meth:`Prog.rec`; R0-R10 are allocatable
GUARD = R[14]  # :meth:`Prog.guard`'s checks
_T0, _T1, _LAST = R[12], R[13], R[15]  # lowering only: wait start/elapsed, last run op
_LAST_STEP, _LAST_RUN = 1, 2
_EVENT, _CHECKPOINT_INFO = 0xFFFFFFFF, 0x11  # binmon unsolicited request id, event
_RAISE = "__raise__"
_FALLBACK = "__fallback__"
_OPS = frozenset(
    "key clear ld ldio mov add sub and_ or_ xor shl shr add8 sub8 frames beq bne blt bge"
    " bmi8 bpl8 jmp djnz out halt".split()
)
_ALU = {
    "mov": lambda x, y: y,
    "add": lambda x, y: x + y,
    "sub": lambda x, y: x - y,
    "and_": lambda x, y: x & y,
    "or_": lambda x, y: x | y,
    "xor": lambda x, y: x ^ y,
    "shl": lambda x, y: x << (y & 31),
    "shr": lambda x, y: x >> (y & 31),
    "add8": lambda x, y: (x + y) & 0xFF,
    "sub8": lambda x, y: (x - y) & 0xFF,
}
_BRANCH = {
    "beq": lambda x, y: x == y,
    "bne": lambda x, y: x != y,
    "blt": lambda x, y: x < y,
    "bge": lambda x, y: x >= y,
}
STATS = {"native": 0, "fallback": 0, "interpreted": 0}
BACKSTOP = 30.0  # host seconds a VICE may go without reaching any stop: a dead VICE


class Outcome(tuple):
    """``(status, code, frames, outputs)``: the ``HALT`` code and the frame-marker
    arrivals the program ran over."""

    __slots__ = ()

    def __new__(cls, status, code, frames, outputs):
        return super().__new__(cls, (status, code, frames, tuple(outputs)))

    status = property(lambda self: self[0])
    code = property(lambda self: self[1])
    frames = property(lambda self: self[2])
    outputs = property(lambda self: self[3])


class Prog:
    """A driver program under construction, ``budget`` cycles at most natively: the
    vagrant-moses builder ops by name
    (``p.ld(R[0], addr)``, ``p.beq(R[0], 5, "done")``) plus timed :meth:`run`,
    :meth:`step`, fresh labels, scratch registers and catch handlers."""

    def __init__(self, budget):
        self.budget = budget  # cycles: a native run's hang guard
        self.ops = []
        self.handlers = []
        self.pcs = set()
        self._labels = 0
        self._free = list(range(SCRATCH.n))[::-1]
        self._catch = []
        self._guard = None
        self._lowered = {}
        self._resolved = None

    def __getattr__(self, name):
        if name not in _OPS:
            raise AttributeError(name)
        return lambda *args: self._emit(name, *args)

    def _emit(self, *op):
        self.ops.append(op)
        self.handlers.append(self.thrower)
        self._lowered, self._resolved = {}, None
        return self

    def new(self, n=1):
        """``n`` fresh label names (one name when ``n`` is 1)."""
        names = [f"L{self._labels + i}" for i in range(n)]
        self._labels += n
        return names[0] if n == 1 else names

    def label(self, name):
        """The next op is the branch target ``name``."""
        return self._emit("label", name)

    @contextlib.contextmanager
    def regs(self, n):
        """``n`` scratch registers for the block."""
        got = [self._free.pop() for _ in range(n)]
        try:
            yield [R[i] for i in got]
        finally:
            self._free.extend(reversed(got))

    @contextlib.contextmanager
    def catch(self, label):
        """Timeouts inside the block branch to ``label``, as ``except`` would."""
        self._catch.append(label)
        try:
            yield
        finally:
            self._catch.pop()

    @property
    def thrower(self):
        """Where a timeout here goes: the innermost handler, else the raise."""
        return self._catch[-1] if self._catch else _RAISE

    @contextlib.contextmanager
    def guard(self, pcs, check):
        """Every wait in the block also stops before any of ``pcs``, and ``check(self)``
        emits the ops run after each wait (in :data:`GUARD`)."""
        self._guard = (tuple(pcs), check)
        try:
            yield
        finally:
            self._guard = None

    def run(self, pc, seconds, on_timeout=None, stops=()):
        """``run_until_pc(pc, timeout=seconds)``, ending early before any of ``stops``:
        nothing when the CPU is about to execute ``pc`` already; a timeout branches to
        ``on_timeout``."""
        self.pcs.add(pc)
        guards, check = self._guard or ((), None)
        self._emit(
            "run", pc, seconds, on_timeout or self.thrower, tuple(stops) + guards
        )
        if check is not None:
            self._guard = None
            check(self)
            self._guard = (guards, check)
        return self

    def step(self):
        """``advance_instructions(1)``."""
        return self._emit("step")

    def rec(self, *vals):
        """Append ``vals`` (registers other than ``SCRATCH``, or immediates) to the
        outputs."""
        if SCRATCH in vals:
            raise ValueError("rec stages immediates in SCRATCH")
        for v in vals:
            if not isinstance(v, Reg):
                self.mov(SCRATCH, v)
                v = SCRATCH
            self.out(v)
        return self

    @property
    def code(self):
        """The ops, then the raise every unhandled timeout lands on."""
        return self.ops + [
            ("halt", HALT_OK),
            ("label", _RAISE),
            ("clear",),
            ("halt", HALT_RAISE),
        ]

    def resolved(self):
        """``(ops, each op's handler, label -> op index)``."""
        if self._resolved is None:
            ops = self.code
            handlers = self.handlers + [_RAISE] * (len(ops) - len(self.ops))
            at = {op[1]: i for i, op in enumerate(ops) if op[0] == "label"}
            self._resolved = ops, handlers, at
        return self._resolved

    def native(self, start_pc):
        """The vagrant-moses ``Program`` started about to execute ``start_pc``: a wait
        for it before any run op is skipped as ``run_until_pc`` skips it (after one, the
        machine's parked state is that test), and a wait past its timeout halts."""
        key = start_pc if start_pc in self.pcs else None
        hit = self._lowered.get(key)
        if hit is not None:
            return hit
        out = Program(self.budget, marker=clock.FRAME_PC)
        for n, op in enumerate(self.code):
            name, args = op[0], op[1:]
            if name == "label":
                out.label(args[0])
            elif name == "run":
                skip = f"__skip{n}"
                if args[0] == key:
                    out.beq(_LAST, 0, skip)
                out.cyc(_T0).run_to((args[0],) + args[3]).cyc(_T1).sub(_T1, _T0)
                out.bge(_T1, cycles(args[1]) + 1, _FALLBACK).beq(_T1, 0, skip)
                out.mov(_LAST, _LAST_RUN)
                out.label(skip)
            elif name == "step":
                out.step().mov(_LAST, _LAST_STEP)
            elif name == "halt":
                out.out(_LAST).halt(*args)
            else:
                getattr(out, name)(*args)
        out.label(_FALLBACK).halt(HALT_FALLBACK)
        self._lowered[key] = out
        return out


def _wait(bm, pc, seconds, stops=()):
    """One wait (``run_until_pc``) of at most ``seconds`` of EMULATED time, ending early
    before any of ``stops``: on VICE counted in frames at the $9630 stop checkpoint, so
    both backends bound it alike; the frame marker itself through that checkpoint."""
    if not isinstance(bm, EmuMon):
        _wait_frames(bm, pc, -(-cycles(seconds) // PAL_FRAME_CYCLES), stops)
        return
    if stops:
        if bm.registers_get()[3] == pc:
            return
        frame = pc == clock.FRAME_PC
        cps = [clock.stop_checkpoint(bm) if frame else _stop_checkpoint(bm, pc)]
        cps += [_stop_checkpoint(bm, at) for at in stops]
        for cp in cps:
            bm.checkpoint_toggle(cp, True)
        try:
            wait_any(bm, cps, seconds)
        finally:
            for cp in cps:
                bm.checkpoint_toggle(cp, False)
        return
    if pc != clock.FRAME_PC:
        bm.run_until_pc(pc, timeout=seconds)
        return
    if bm.registers_get()[3] == pc:
        return
    cp = clock.stop_checkpoint(bm)
    bm.checkpoint_toggle(cp, True)
    try:
        bm.wait_for_checkpoint(cp, timeout=seconds)
    finally:
        bm.checkpoint_toggle(cp, False)


def _wait_frames(bm, pc, limit, stops=()):
    """Resume VICE until it is about to execute ``pc`` or any of ``stops``, raising after
    ``limit`` frames; each $9630 stop between counts one, then steps off the marker. A
    VICE that stops answering raises after :data:`BACKSTOP` host seconds."""
    if bm.registers_get()[3] == pc:
        return
    marker = clock.stop_checkpoint(bm)
    ends = {marker if pc == clock.FRAME_PC else _stop_checkpoint(bm, pc)}
    ends |= {_stop_checkpoint(bm, at) for at in stops}
    for cp in ends | {marker}:
        bm.checkpoint_toggle(cp, True)
    try:
        for _ in range(limit):
            if wait_any(bm, ends | {marker}, BACKSTOP) & ends:
                return
            bm.advance_instructions(1)
        raise BinmonError(f"${pc:04x} not reached in {limit} frames")
    finally:
        for cp in ends | {marker}:
            bm.checkpoint_toggle(cp, False)


def wait_any(bm, checknums, timeout):
    """Resume ``bm`` until any of the enabled stop checkpoints ``checknums`` hits;
    the ones that did. A stop elsewhere or a spent ``timeout`` raises ``BinmonError``.
    """
    if isinstance(bm, EmuMon):
        return bm.wait_for_checkpoints(checknums, timeout)
    want, hits = set(checknums), set()
    with bm._lock:
        old = bm._sock.gettimeout()
        bm._sock.settimeout(timeout)
        try:
            bm._send(OPCODE.EXIT, b"", bm._next_req())
            deadline = time.monotonic() + timeout
            while time.monotonic() < deadline:
                resp = bm._read_response()
                if resp.req_id != _EVENT:
                    continue
                if resp.opcode == _CHECKPOINT_INFO and len(resp.body) >= 5:
                    num = struct.unpack("<I", resp.body[:4])[0]
                    if resp.body[4] and num in want:
                        hits.add(num)
                elif resp.opcode == OPCODE.STOPPED and hits:
                    return hits
        finally:
            bm._sock.settimeout(old)
    raise BinmonError(f"wait_any({sorted(want)}) timed out")


def _stop_checkpoint(bm, pc):
    """checknum of a persistent, disabled stop-on-hit checkpoint at ``pc`` on ``bm``."""
    cps = bm.__dict__.setdefault("_prog_stops", {})
    if pc not in cps:
        cps[pc] = bm.checkpoint_set(pc, stop_when_hit=True, enabled=False).checknum
    return cps[pc]


def interpret(bm, prog):
    """``prog`` executed call by call over the BinMon ``bm``; a call that raises
    branches to its op's handler as ``except`` would, and to the raise outside one."""
    STATS["interpreted"] += 1
    ops, handlers, at = prog.resolved()
    st = {"reg": [0] * len(R), "out": [], "i": 0}
    with bm.halted():
        st["f0"] = clock.frames(bm)
        while st["i"] < len(ops):
            i = st["i"]
            st["i"] += 1
            try:
                code = _call(bm, ops[i], st, at)
            except Exception:  # pylint: disable=broad-except
                if handlers[i] == _RAISE:
                    raise
                st["i"] = at[handlers[i]]
                continue
            if code is not None:
                return Outcome("halt", code, clock.frames(bm) - st["f0"], st["out"])
        return Outcome("halt", HALT_OK, clock.frames(bm) - st["f0"], st["out"])


def _call(bm, op, st, at):
    """One op of :func:`interpret` on ``st`` (registers, outputs, next op); the halt
    code when it halts."""
    # pylint: disable=too-many-branches
    name, args, reg = op[0], op[1:], st["reg"]

    def val(s):
        return reg[s.n] if isinstance(s, Reg) else s

    if name == "run":
        try:
            _wait(bm, args[0], args[1], args[3])
        except Exception:  # pylint: disable=broad-except
            st["i"] = at[args[2]]
    elif name == "step":
        bm.advance_instructions(1)
    elif name == "key":
        down = args[2] if len(args) > 2 else True
        bm.keymatrix_set([(args[0], args[1], int(bool(down)))])
    elif name == "clear":
        bm.keymatrix_release_all()
    elif name in ("ld", "ldio"):
        idx = args[2] if len(args) > 2 else None
        a = (args[1] + (reg[idx.n] if idx is not None else 0)) & 0xFFFF
        kw = {"bank": "io"} if name == "ldio" else {}
        reg[args[0].n] = bm.mem_get(a, a, **kw)[0]
    elif name == "frames":
        reg[args[0].n] = clock.frames(bm) - st["f0"]
    elif name in _ALU:
        reg[args[0].n] = _ALU[name](reg[args[0].n], val(args[1])) & 0xFFFFFFFF
    elif name in _BRANCH:
        if _BRANCH[name](reg[args[0].n], val(args[1])):
            st["i"] = at[args[2]]
    elif name in ("bmi8", "bpl8"):
        if bool(reg[args[0].n] & 0x80) == (name == "bmi8"):
            st["i"] = at[args[1]]
    elif name == "jmp":
        st["i"] = at[args[0]]
    elif name == "djnz":
        reg[args[0].n] = (reg[args[0].n] - 1) & 0xFFFFFFFF
        if reg[args[0].n]:
            st["i"] = at[args[1]]
    elif name == "out":
        st["out"].append(reg[args[0].n])
    elif name == "halt":
        return val(args[0])
    return None


def _settle(bm, prog, outcome, backup, owed):
    """``outcome`` of ``prog`` run natively on ``bm``'s machine, accounted on ``bm``
    (``owed``: the marker test a step left there, which the first run op counts); a
    timed-out wait replays the program interpreted from ``backup()``, the machine it
    started on."""
    if outcome.status == "halt" and outcome.code != HALT_FALLBACK:
        STATS["native"] += 1
        last = outcome.outputs[-1]
        frames = outcome.frames + bool(last and owed)
        bm.ran(frames, None if last == 0 else last == _LAST_STEP)
        return Outcome("halt", outcome.code, frames, outcome.outputs[:-1])
    STATS["fallback"] += 1
    bm.adopt(backup())
    return interpret(bm, prog)


def execute(bm, prog):
    """The :class:`Outcome` of ``prog`` on ``bm``: natively on an ``EmuMon`` whose
    checkpoints allow it, else interpreted (``HALT_RAISE``: a timeout nothing caught).
    """
    if not isinstance(bm, EmuMon):
        return interpret(bm, prog)
    clock.frame_checkpoint(bm)
    if not bm.native_ready():
        return interpret(bm, prog)
    backup, owed = bm.m.clone(), bm.owes_marker()
    outcome = bm.m.run_program(prog.native(bm.m.mpu.pc))
    return _settle(bm, prog, outcome, lambda: backup, owed)


def forks(bm, progs, threads=None):
    """``(outcomes, monitors)``: each of ``progs`` on its own fork of the ``EmuMon``
    ``bm`` by one ``vagrant_moses.run_forks`` batch (threads, GIL released), every
    fork an ``EmuMon`` left where its program ended; ``bm`` is untouched."""
    clock.frame_checkpoint(bm)
    if not bm.native_ready():
        twins = [bm.clone() for _ in progs]
        return [interpret(t, p) for t, p in zip(twins, progs)], twins
    pc = bm.m.mpu.pc
    outs, machines = run_forks(
        bm.m, [p.native(pc) for p in progs], threads=threads, keep=True
    )
    twins, owed = [bm.clone(machine) for machine in machines], bm.owes_marker()
    return [
        _settle(t, p, o, bm.m.clone, owed) for t, p, o in zip(twins, progs, outs)
    ], twins
