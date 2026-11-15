#!/usr/bin/env python3
"""The keyboard driver compiled to driver programs (:mod:`driver.prog`).

:class:`Compiler` emits each :class:`driver.kbd_aim.KbdDriver` primitive as a program
segment -- same keys, PCs, retry bounds and stop points, plus timing records;
:func:`compile_move` builds one whole macro move.
"""

import contextlib
import functools


from driver import prog
from driver.clock import FRAME_PC
from driver.emu import cycles
from driver.kbd_aim import (
    A_CX,
    A_CY,
    A_H,
    A_PLOT,
    A_SFLAG,
    A_SLOT,
    A_V,
    ACTION_CODE,
    K_DOWN,
    K_LEFT,
    K_RIGHT,
    K_UP,
    K_UTURN,
    PC_PAN_CALL,
    PC_PASS,
    _PAN_MAX_FRAMES,
    _PAN_STALL_FRAMES,
    _RU_COMMIT,
    _SCAN_WAIT_PASSES,
    KbdDriver,
    _k,
)
from sentinel import aimcost as ac, memmap as mm
from sentinel.passcost import PAL_FRAME_CYCLES

SCRATCH = prog.SCRATCH
UNREACHABLE, OK, HYPERSPACE = 0, 1, 2
T_BEGIN, T_END, T_NOW, T_PASS, T_VAL, T_LOG = range(1, 7)
ARITY = {T_BEGIN: 2, T_END: 2, T_NOW: 3, T_PASS: 1, T_VAL: 2, T_LOG: 3}
TIMERS = ("pan_h", "pan_v", "toggle", "cursor", "uturn")
LOG_UTURN, LOG_TAP_STOP = range(2)
TAPS = tuple(ACTION_CODE) + ("SPACE",)
(
    H_FAIL_OFF,
    H_HYPERSPACE,
    H_FAIL_ON,
    H_AIM_MISS,
    H_DRAINED,
    H_NOT_OK,
    H_DIED,
) = range(2, 9)
(
    V_RES,
    V_OKH,
    V_OKV,
    V_ACH_H,
    V_ACH_V,
    V_OKC,
    V_CX,
    V_CY,
    V_ENERGY,
    V_F0,
    V_F1,
    V_LATCHED,
    V_DONE,
    V_STATUS,
    V_SLOT,
    V_Z,
    V_ZFRAC,
    V_ENEMIES,
    V_ALIVE,
    V_WON,
    V_FRAMES,
    V_SEEN,
    V_AWAY,
) = range(23)
MOVE_FACTS = {
    "status": V_STATUS,
    "energy": V_ENERGY,
    "slot": V_SLOT,
    "z": V_Z,
    "z_frac": V_ZFRAC,
    "enemies": V_ENEMIES,
    "alive": V_ALIVE,
    "won": V_WON,
    "frames": V_FRAMES,
    "seen": V_SEEN,
    "away": V_AWAY,
}
MOVE_KEYS = {"absorb": ("A",), "mount": ("Q",)}
KILL_PC = 0x1A05  # kill_player $1A00, past its STA $0C4E
TOLL_PC = 0x217D  # do_hyperspace's toll underflow, past its STA $0CDE ($80)
WIN_PC = 0x219B  # player_survived_hyperspace, past its STA $0CDE ($C0)


def budget(steps, frames):
    """A native run's hang guard for ``steps`` actions and ``frames`` of waiting: the
    interactive driver's own worst pan, every pass at the commit timeout, per action."""
    return cycles(steps * _PAN_MAX_FRAMES * _RU_COMMIT) + frames * PAL_FRAME_CYCLES


def _runs(values):
    """Maximal ``(lo, hi)`` runs of the sorted byte values ``values``."""
    out = []
    for v in values:
        if out and out[-1][1] == v - 1:
            out[-1][1] = v
        else:
            out.append([v, v])
    return out


class Compiler:
    """One program: :class:`KbdDriver` primitives appended in call order, results in
    the caller's registers (``res``)."""

    def __init__(self, steps=1, frames=0):
        self.p = prog.Prog(budget(steps, frames))
        self.exits = None  # (status register, label) a refusal goes to, else it halts

    def exit(self, code):
        """End with ``code``: halt, or branch to :attr:`exits` with it."""
        if self.exits is None:
            self.p.halt(code)
        else:
            self.p.mov(self.exits[0], code).jmp(self.exits[1])

    # ---- plumbing ------------------------------------------------------------
    def branch_in(self, r, values, label):
        """To ``label`` when register ``r`` holds a byte in ``values``."""
        p = self.p
        for lo, hi in _runs(sorted(values)):
            if lo == hi:
                p.beq(r, lo, label)
            elif hi == 0xFF:
                p.bge(r, lo, label)
            elif lo == 0:
                p.blt(r, hi + 1, label)
            else:
                skip = p.new()
                p.blt(r, lo, skip).blt(r, hi + 1, label).label(skip)

    @contextlib.contextmanager
    def timed(self, name):
        """The block's frames, recorded only when it completes."""
        name = TIMERS.index(name)
        self.p.rec(T_BEGIN, name).frames(SCRATCH).out(SCRATCH)
        yield
        self.p.rec(T_END, name).frames(SCRATCH).out(SCRATCH)

    def val(self, key, src):
        """Output ``src`` as the result ``key``."""
        self.p.rec(T_VAL, key, src)

    def _sflag(self, on, label):
        """To ``label`` when the sights flag ($0C5F bit7) is ``on``."""
        p = self.p
        p.ld(SCRATCH, A_SFLAG).and_(SCRATCH, 0x80)
        (p.bne if on else p.beq)(SCRATCH, 0, label)

    # ---- waits ---------------------------------------------------------------
    def run_past_redraw(self, pc, seconds, res=None):
        """``_run_past_redraw``: a timeout re-arms while $0CE4 bit7 is set or was at the
        previous timeout, raises otherwise; ``res`` false once the re-arms are spent."""
        p = self.p
        loop, timeout, busy, spent, done = p.new(5)
        with p.regs(1) as (n,):  # re-arms left; bit 8: plotting at the last timeout
            p.mov(n, _SCAN_WAIT_PASSES).label(loop).run(pc, seconds, timeout)
            if res is not None:
                p.mov(res, 1)
            p.jmp(done).label(timeout)
            p.ld(SCRATCH, A_PLOT).and_(SCRATCH, 0x80).shl(SCRATCH, 1)
            p.bne(SCRATCH, 0, busy).blt(n, 0x100, p.thrower).label(busy)
            p.and_(n, 0xFF).sub(n, 1).beq(n, 0, spent).or_(n, SCRATCH).jmp(loop)
            p.label(spent)
            if res is not None:
                p.mov(res, 0)
            p.label(done)

    def run_to_scan(self, seconds=6.0):
        """``_run_to_scan``: the next gated full input scan ($9678)."""
        self.run_past_redraw(KbdDriver.PC_IRQ_SCAN, seconds)

    def run_frames(self, n):
        """``clock.run_frames``: step off and wait for the $9630 marker ``n`` times."""
        if n <= 0:
            return
        p = self.p
        loop = p.new()
        with p.regs(1) as (k,):
            p.mov(k, n).label(loop).step().run(FRAME_PC, 6.0).djnz(k, loop)

    def probe_settle(self):
        """Settle a redraw: two frames at a time while $0CE4 bit7 is set, eight times at most."""
        p = self.p
        loop, done = p.new(2)
        with p.regs(1) as (n,):
            p.mov(n, 8).label(loop)
            p.ld(SCRATCH, A_PLOT).and_(SCRATCH, 0x80).beq(SCRATCH, 0, done)
            self.run_frames(2)
            p.djnz(n, loop).label(done)

    def now(self):
        """Record the frame count and the raster line."""
        p = self.p
        p.rec(T_NOW).frames(SCRATCH).out(SCRATCH)
        p.ldio(SCRATCH, 0xD011).out(SCRATCH).ldio(SCRATCH, 0xD012).out(SCRATCH)

    def run_pass(self):
        """``_run_pass``: one main-loop pass to $365D split at its ROM boundaries, with the
        frozen flag."""
        p = self.p
        self.now()
        for pc in (PC_PASS, PC_PAN_CALL, KbdDriver.PC_PAN_DONE):
            self.run_past_redraw(pc, _RU_COMMIT)
            self.now()
            if pc != KbdDriver.PC_PAN_DONE:
                p.step()
        p.rec(T_PASS).ld(SCRATCH, mm.PLAYER_NOT_ACTED).shr(SCRATCH, 7).out(SCRATCH)

    # ---- primitives ----------------------------------------------------------
    def pan_angle(self, base, s, want, first, keys, res):
        """``_pan_angle`` on ``base + s``: hold ``keys[0]`` while the angle is in
        ``first``, else ``keys[1]``; ``res`` the outcome (``STATUS`` index)."""
        p = self.p
        loop, ok, hyper, pick, press, moved, nxt, end = p.new(8)
        with p.regs(5) as (slot0, n, stalled, held, cur):
            p.ld(slot0, A_SLOT).mov(stalled, 0).mov(held, 0).mov(res, UNREACHABLE)
            p.mov(n, _PAN_MAX_FRAMES).label(loop)
            p.ld(cur, base, s).beq(cur, want, ok)
            p.ld(SCRATCH, A_SLOT).bne(SCRATCH, slot0, hyper)
            self.branch_in(cur, first, pick)
            p.beq(held, 2, press).clear().key(*_k(keys[1])).mov(held, 2).jmp(press)
            p.label(pick).beq(held, 1, press).clear().key(*_k(keys[0])).mov(held, 1)
            p.label(press)
            self.run_pass()
            p.ld(SCRATCH, base, s).step().bne(SCRATCH, cur, moved)
            p.add(stalled, 1).bge(stalled, _PAN_STALL_FRAMES, end).jmp(nxt)
            p.label(moved).mov(stalled, 0)
            p.label(nxt).djnz(n, loop).jmp(end)
            p.label(ok).mov(res, OK).jmp(end)
            p.label(hyper).mov(res, HYPERSPACE)
            p.label(end).clear()

    def tap_action(self, name, settle, res):
        """``tap_action``: idle scan, press inside the next, check the want-flags
        $0CE8..$0CEB, settle to the next gated scan; up to 45 tries."""
        p = self.p
        want, (r, c) = ACTION_CODE[name], _k(name)
        loop, hit, stop, out = p.new(4)
        p.mov(res, 0)
        with p.regs(1) as (n,), p.catch(stop):
            p.mov(n, 45).label(loop)
            self.run_to_scan()
            p.step()
            self.run_to_scan()
            p.key(r, c).run(KbdDriver.PC_IRQ_SCAN_DONE, 6.0)
            for addr in range(0x0CE8, 0x0CEC):
                p.ld(SCRATCH, addr).beq(SCRATCH, want, hit)
            p.clear().djnz(n, loop).jmp(out)
            p.label(hit).clear().mov(res, 1)
            if settle:
                with p.catch(out):
                    self.run_to_scan()
            p.jmp(out)
        p.label(stop).rec(T_LOG, LOG_TAP_STOP, TAPS.index(name), 0)
        p.label(out).clear()

    def uturn(self, s, res):
        """``_uturn``: tap U until the bearing at ``A_H + s`` moved by $80, five tries."""
        p = self.p
        loop, nxt, ok, end = p.new(4)
        with self.timed("uturn"), p.regs(2) as (before, n):
            p.mov(n, 5).label(loop).ld(before, A_H, s)
            self.tap_action(K_UTURN, True, res)
            p.beq(res, 0, nxt)
            p.ld(SCRATCH, A_H, s).sub8(SCRATCH, before).beq(SCRATCH, 0x80, ok)
            p.label(nxt).djnz(n, loop).mov(res, 0).jmp(end)
            p.label(ok).mov(res, 1).label(end)

    def coarse_h(self, want, res):
        """``coarse_h``: a u-turn first when ``aimcost.h_press_count`` takes one, then
        the pan by the shorter way round."""
        p = self.p
        want &= 0xFF
        with self.timed("pan_h"), p.regs(1) as (s,):
            p.ld(s, A_SLOT)
            with p.regs(1) as (cur,):
                go, skip = p.new(2)
                p.ld(cur, A_H, s)
                self.branch_in(
                    cur, [c for c in range(256) if ac.h_press_count(c, want)[0]], go
                )
                p.jmp(skip).label(go)
                self.uturn(s, res)
                p.bne(res, 0, skip).rec(T_LOG, LOG_UTURN, cur, want).label(skip)
            right = [c for c in range(256) if ((want - c) & 0xFF) <= 0x80]
            self.pan_angle(A_H, s, want, right, (K_RIGHT, K_LEFT), res)

    def coarse_v(self, want, res):
        """``coarse_v``: the pitch pan, ``unreachable`` off the $CD..$35 band."""
        p = self.p
        want &= 0xFF
        with self.timed("pan_v"):
            if not (want >= 0xCD or want <= 0x35):
                p.mov(res, UNREACHABLE)
                return
            lin = KbdDriver._pitch_lin  # pylint: disable=protected-access
            down = [c for c in range(256) if lin(want) > lin(c)]
            with p.regs(1) as (s,):
                p.ld(s, A_SLOT)
                self.pan_angle(A_V, s, want, down, (K_DOWN, K_UP), res)

    def coarse_hv(self, view, okh, okv):
        """``coarse_hv``: bearing then pitch, once more when either missed and neither
        hyperspaced."""
        p = self.p
        loop, again, end = p.new(3)
        with p.regs(1) as (t,):
            p.mov(t, 2).label(loop)
            self.coarse_h(view["h_angle"], okh)
            self.coarse_v(view["v_angle"], okv)
            p.beq(okh, HYPERSPACE, end).beq(okv, HYPERSPACE, end)
            p.bne(okh, OK, again).beq(okv, OK, end)
            p.label(again).djnz(t, loop).label(end)

    def one_scan_press(self):
        """``_one_scan_press``: SPACE held for exactly one gated scan after an idle
        one."""
        p = self.p
        self.run_to_scan(10.0)
        p.step()
        self.run_to_scan(10.0)
        p.key(*_k("SPACE")).run(KbdDriver.PC_IRQ_SCAN_DONE, 10.0).clear()

    def sights_set(self, on, res):
        """``sights_set``: SPACE until the sights flag matches, six presses at most."""
        p = self.p
        loop, yes, end = p.new(3)
        with self.timed("toggle"), p.regs(1) as (n,):
            p.mov(n, 6).label(loop)
            self._sflag(on, yes)
            self.one_scan_press()
            p.djnz(n, loop)
            self._sflag(on, yes)
            p.mov(res, 0).jmp(end).label(yes).mov(res, 1).label(end)

    def fine_cursor(self, cx, cy, res):
        """``fine_cursor``: per pass one x and one y key, run to the store of the last
        axis pressed and over it; 160 passes."""
        p = self.p
        cx, cy = cx & 0xFF, cy & 0xFF
        loop, move, nox, right, ymove, xinc, down, fail, ok, end = p.new(10)
        with self.timed("cursor"), p.regs(3) as (n, x, y):
            p.mov(n, 160).label(loop).ld(x, A_CX).ld(y, A_CY)
            p.bne(x, cx, move).beq(y, cy, ok).label(move)
            p.beq(x, cx, nox).blt(x, cx, right).key(*_k(K_LEFT)).jmp(nox)
            p.label(right).key(*_k(K_RIGHT)).label(nox)
            with p.catch(fail):
                p.bne(y, cy, ymove).blt(x, cx, xinc)
                p.run(KbdDriver.PC_CX_DEC, 1.5).step().jmp(fail)
                p.label(xinc).run(KbdDriver.PC_CX_INC, 1.5).step().jmp(fail)
                p.label(ymove).blt(y, cy, down).key(*_k(K_UP))
                p.run(KbdDriver.PC_CY_DEC, 1.5).step().jmp(fail)
                p.label(down).key(*_k(K_DOWN)).run(KbdDriver.PC_CY_INC, 1.5).step()
            p.label(fail).clear().djnz(n, loop)
            p.ld(x, A_CX).ld(y, A_CY).mov(res, 0).bne(x, cx, end).bne(y, cy, end)
            p.label(ok).mov(res, 1).label(end)

    # ---- segments ------------------------------------------------------------
    def drive_to(self, view, fail):
        """``drive_to`` up to its ``ok``: to ``fail`` unless sights off, both pans,
        sights on and the cursor all landed; a hyperspace stops at once."""
        p = self.p
        with p.regs(2) as (okh, okv):
            self.sights_set(False, okh)
            p.beq(okh, 0, fail)
            self.coarse_hv(view, okh, okv)
            p.beq(okh, HYPERSPACE, fail).beq(okv, HYPERSPACE, fail)
            with p.regs(2) as (oks, okc):
                self.sights_set(True, oks)
                self.fine_cursor(*view["cursor"], okc)
                p.bne(okh, OK, fail).bne(okv, OK, fail)
                p.beq(oks, 0, fail).beq(okc, 0, fail)

    def aim(self, view):
        """``perform_step``'s full aim: sights off, pans, the angles read sights-off,
        sights on, cursor, the probe's settle and the cursor read, halting on each
        refusal and with ``H_AIM_MISS`` when what was read back is not ``view``."""
        p = self.p
        off, hyper, landed, on, miss, done = p.new(6)
        cx, cy = view["cursor"]
        with p.regs(2) as (h, v):
            self.sights_set(False, h)
            p.bne(h, 0, off)
            self.exit(H_FAIL_OFF)
            p.label(off)
            self.coarse_hv(view, h, v)
            self.val(V_OKH, h)
            self.val(V_OKV, v)
            p.beq(h, HYPERSPACE, hyper).bne(v, HYPERSPACE, landed).label(hyper)
            self.exit(H_HYPERSPACE)
            p.label(landed)
            p.ld(SCRATCH, A_SLOT).ld(h, A_H, SCRATCH).ld(v, A_V, SCRATCH)
            self.val(V_ACH_H, h)
            self.val(V_ACH_V, v)
            with p.regs(2) as (a, y):
                self.sights_set(True, a)
                p.bne(a, 0, on)
                self.exit(H_FAIL_ON)
                p.label(on)
                self.fine_cursor(cx, cy, a)
                self.val(V_OKC, a)
                self.probe_settle()
                p.ld(a, A_CX).ld(y, A_CY)
                self.val(V_CX, a)
                self.val(V_CY, y)
                p.bne(h, view["h_angle"], miss).bne(v, view["v_angle"], miss)
                p.bne(a, cx, miss).beq(y, cy, done)
            p.label(miss)
            self.exit(H_AIM_MISS)
            p.label(done)

    def fire(self, key, min_energy):
        """``perform_step``'s fire: the post-aim budget gate, sights on and the action
        key, between two frame reads."""
        p = self.p
        rich = p.new()
        with p.regs(1) as (a,):
            if min_energy is not None:
                p.ld(a, mm.PLAYER_ENERGY).and_(a, 0x3F)
                self.val(V_ENERGY, a)
                p.bge(a, min_energy, rich)
                self.exit(H_DRAINED)
            p.label(rich).frames(a)
            self.val(V_F0, a)
            self.sights_set(True, a)
            self.tap_action(key, True, a)
            self.val(V_LATCHED, a)
            p.frames(a)
            self.val(V_F1, a)


@functools.lru_cache(maxsize=1 << 12)
def compile_move(kind, view=None, k=0, frames=0, reuse=False):
    """One :class:`driver.prog.Prog` playing a macro move, read by :func:`move_facts`.

    ``kind``: ``"absorb"``/``"mount"`` on ``view``'s tile, ``"hop"`` (``k`` boulders, a robot,
    the transfer), ``"wait"`` ``frames``, ``"relocate"`` (escape hyperspace) or ``"finish"``.
    An action on the previous bearing (or, with ``reuse``, the sights) drives the cursor only.
    """

    keys = {"wait": (), "relocate": (), "finish": ()}.get(kind)
    keys = MOVE_KEYS.get(kind, ("B",) * k + ("R", "Q")) if keys is None else keys
    steps = view if keys and isinstance(view[0], tuple) else (view,) * len(keys)
    c = Compiler(k + 2, frames)
    p = c.p
    facts, died, done = p.new(3)

    def check(p):
        p.ld(prog.GUARD, mm.PLAYER_DIED_BY_DRAINING).bmi8(prog.GUARD, died)
        p.ld(prog.GUARD, mm.LANDSCAPE_COMPLETE).and_(prog.GUARD, 0xC0)
        p.beq(prog.GUARD, 0x80, died).bne(prog.GUARD, 0, done)

    with p.regs(1) as (status,), p.guard((KILL_PC, TOLL_PC, WIN_PC), check):
        p.mov(status, prog.HALT_OK)
        c.exits = (status, facts)
        if kind == "wait":
            c.run_frames(frames)
        elif kind == "relocate":
            with p.regs(1) as (a,):
                c.tap_action("H", True, a)
        elif kind == "finish":
            _hyperspace_attempt(c)
        for i, (key, step) in enumerate(zip(keys, steps)):
            if (steps[i - 1][:2] == step[:2]) if i else reuse:
                with p.regs(1) as (a,):
                    c.fine_cursor(*step[2:], a)
                    c.probe_settle()
            else:
                c.aim(_view(step))
            c.fire(key, None)
        p.jmp(facts).label(died).mov(status, H_DIED).label(done).clear()
        p.label(facts)
        c.val(V_STATUS, status)
        _facts(c)
    return p


def _facts(c):
    """Output the post-move facts: energy, player slot and its z, enemies left, alive
    (no drain death, no reset to unacted), won, frames run, the exposure bar (bit1
    fully seen, bit0 partly), the tile's difference from the platform's (0: home)."""
    p = c.p
    loop, skip = p.new(2)
    with p.regs(2) as (a, n):
        p.ld(a, mm.PLAYER_ENERGY).and_(a, 0x3F)
        c.val(V_ENERGY, a)
        p.ld(a, mm.PLAYER_OBJECT)
        c.val(V_SLOT, a)
        p.ld(n, mm.OBJECTS_Z_HEIGHT, a)
        c.val(V_Z, n)
        p.ld(n, mm.OBJECTS_Z_FRACTION, a)
        c.val(V_ZFRAC, n)
        p.mov(a, 0).mov(n, 0).label(loop)
        p.ld(SCRATCH, mm.OBJECTS_FLAGS, a).and_(SCRATCH, 0x80).bne(SCRATCH, 0, skip)
        p.ld(SCRATCH, mm.OBJECTS_TYPE, a)
        for otype in mm.ENEMY_TYPES:
            nxt = p.new()
            p.bne(SCRATCH, otype, nxt).add(n, 1).label(nxt)
        p.label(skip).add(a, 1).blt(a, mm.NUM_SLOTS, loop)
        c.val(V_ENEMIES, n)
        p.ld(a, mm.PLAYER_DIED_BY_DRAINING).ld(n, mm.PLAYER_NOT_ACTED).or_(a, n)
        p.and_(a, 0x80).xor(a, 0x80).shr(a, 7)
        c.val(V_ALIVE, a)
        p.ld(a, mm.LANDSCAPE_COMPLETE).and_(a, 0x40).shr(a, 6)
        c.val(V_WON, a)
        p.frames(a)
        c.val(V_FRAMES, a)
        p.ld(a, mm.EXPOSURE_BAR).shr(a, 6)
        c.val(V_SEEN, a)
        p.ld(a, mm.PLAYER_OBJECT).ld(n, mm.OBJECTS_X, a).ld(a, mm.OBJECTS_Y, a)
        p.ld(SCRATCH, mm.PLATFORM_X).xor(n, SCRATCH).ld(SCRATCH, mm.PLATFORM_Y)
        p.xor(a, SCRATCH).or_(a, n)
        c.val(V_AWAY, a)


def move_facts(outcome):
    """``{fact: value}`` of a :func:`compile_move` program's outcome; ``status`` is
    ``HALT_OK`` or the refusal (``H_*``) that cut the move short."""
    vals = {}
    out = outcome.outputs
    i = 0
    while i < len(out):
        if out[i] == T_VAL:
            vals[out[i + 1]] = out[i + 2]
        i += 1 + ARITY[out[i]]
    return {name: vals[key] for name, key in MOVE_FACTS.items() if key in vals}


def _hyperspace_attempt(c):
    """``fire_hyperspace``'s attempt: H without a settle, then three frames at a time
    until $0CDE bit6, 120 times or until the marker stops (``V_RES`` 1)."""
    p = c.p
    loop, stop, done = p.new(3)
    with p.regs(1) as (n,):
        c.tap_action("H", False, n)
        c.val(V_RES, 0)
        p.mov(n, 120).label(loop)
        p.ld(SCRATCH, mm.LANDSCAPE_COMPLETE).and_(SCRATCH, 0x40).bne(SCRATCH, 0, done)
        with p.catch(stop):
            c.run_frames(3)
        p.djnz(n, loop).jmp(done)
        p.label(stop)
        c.val(V_RES, 1)
        p.label(done).ld(n, mm.LANDSCAPE_COMPLETE)
        c.val(V_DONE, n)


def _view(key):
    h, v, cx, cy = key
    return {"h_angle": h, "v_angle": v, "cursor": (cx, cy)}
