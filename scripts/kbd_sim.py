#!/usr/bin/env python3
"""Headless (py65) proof of the deterministic keyboard-drive protocol for The
Sentinel (C64). Runs the REAL play loop + REAL IRQ handler ($95E9) + REAL input
scan ($9678 -> $119F -> $1363) with a modeled CIA1 keyboard matrix, and drives
the same checkpoint protocol the live KbdDriver uses:

    press key WHILE CPU HALTED -> run_until_pc(commit checkpoint) -> read memory
    -> release WHILE HALTED -> resume.

plot_world ($2625) is stubbed to CLC/RTS (carry clear == "plotted OK"): the
protocol only depends on its carry, never its pixels. Everything else -- the
input scan, want-flag latching, auto-repeat gate $0CC8, pan_viewpoint commits,
move_sights, unbuffer pacing ($0CC1/$0CD8), action consumption -- is the real
ROM code.

Assertions prove, at several IRQ cadences:
  * one _cursor_step == exactly 1 px, none after release
  * one pan commit == exactly +-8 (h) / +-4 (v); landing is exact, no overshoot
  * v clamp ($35/$CD) never overshoots
  * tap_action fires exactly once ($12DE executed once; u-turn flips h once)
  * final state identical across cadences (timing-independence)
"""

import sys
import os

_SCRIPTS = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, _SCRIPTS)
os.chdir(os.path.dirname(_SCRIPTS))

from py65.devices.mpu6502 import MPU
from py65.memory import ObservableMemory
import _emu

# game keynums (table $138D): pa bit = k&7 (driven low on $DC00), pb bit = k>>3
KEY = {
    "S": 0x29,
    "D": 0x12,
    "L": 0x15,
    "COMMA": 0x3D,
    "A": 0x11,
    "Q": 0x37,
    "R": 0x0A,
    "T": 0x32,
    "B": 0x23,
    "H": 0x2B,
    "U": 0x33,
    "SPACE": 0x27,
}

A_SLOT, A_H, A_V, A_CX, A_CY, A_SFLAG = 0x000B, 0x09C0, 0x0140, 0x0CC6, 0x0CC7, 0x0C5F

PC_H_COMMIT, PC_V_COMMIT = 0x10EE, 0x1135
PC_CX_INC, PC_CX_DEC = 0x997C, 0x9990
PC_CY_INC, PC_CY_DEC = 0x99B8, 0x99D2
PC_SCAN_ENTRY, PC_SCAN_TAIL = 0x1363, 0x1386
PC_IRQ_SCAN = 0x9678  # IRQ-side JSR check_for_full_player_input
PC_ACTION_ACCEPT = 0x12DE  # STA $0C61: action passed all gates this frame


class SimTimeout(Exception):
    pass


class Machine:
    def __init__(self, irq_period):
        img = open(_emu.IMG, "rb").read()
        mem = bytearray(0x10000)
        for a, b in _emu.LOADED:
            mem[a : b + 1] = img[a : b + 1]
        self.mem = mem
        self.matrix = set()  # pressed (pa_bit, pb_bit)
        self.dc00 = 0xFF
        self.raster = 0
        self.irq_period = irq_period
        self.irq_on = False
        self.icount = 0
        self.next_irq = irq_period
        self.watch = {}  # pc -> hit count

        om = ObservableMemory()

        def rd(addr):
            if addr == 0xDC01:
                v = 0xFF
                for pa, pb in self.matrix:
                    if not (self.dc00 >> pa) & 1:
                        v &= ~(1 << pb) & 0xFF
                return v
            if addr == 0xDC00:
                return self.dc00
            if addr == 0xD012:
                self.raster = (self.raster + 1) & 0xFF
                return self.raster
            return mem[addr]

        def wr(addr, v):
            if addr == 0xDC00:
                self.dc00 = v & 0xFF
            mem[addr] = v & 0xFF

        om.subscribe_to_read(range(0x10000), rd)
        om.subscribe_to_write(range(0x10000), wr)
        self.cpu = MPU(memory=om)
        for a in (
            0xFF81,
            0xFF90,
            0xFFBA,
            0xFFBD,
            0xFFD2,
            0xFFD5,
            0xFFE1,
            0xFFE4,
            0xFFE7,
            0xFF9F,
            0xFFCC,
        ):
            mem[a] = 0x60

    # ---- matrix (changes only between instructions == while halted) ----------
    def press(self, name):
        self.press_code(KEY[name])

    def press_code(self, k):
        self.matrix.add((k & 7, k >> 3))

    def release_all(self):
        self.matrix.clear()

    # ---- CPU stepping with IRQ injection -------------------------------------
    def _push(self, v):
        self.mem[0x0100 + self.cpu.sp] = v & 0xFF
        self.cpu.sp = (self.cpu.sp - 1) & 0xFF

    def step(self):
        cpu = self.cpu
        self.icount += 1
        if self.irq_on and self.icount >= self.next_irq:
            self.next_irq = self.icount + self.irq_period
            if not (cpu.p & 0x04):
                self.mem[0xD019] |= 0x81
                self._push(cpu.pc >> 8)
                self._push(cpu.pc & 0xFF)
                self._push((cpu.p | 0x20) & ~0x10)
                cpu.p |= 0x04
                cpu.pc = self.mem[0xFFFE] | (self.mem[0xFFFF] << 8)
                if cpu.pc in self.watch:
                    self.watch[cpu.pc] += 1
                return
        cpu.step()
        if cpu.pc in self.watch:
            self.watch[cpu.pc] += 1

    def run_until_pc(self, target, budget=3_000_000):
        for _ in range(budget):
            if self.cpu.pc == target:
                return
            self.step()
        raise SimTimeout(f"pc ${target:04x} not reached in {budget}")

    def advance(self, n):
        for _ in range(n):
            self.step()

    def run_frames(self, n):
        """Let ~n display frames elapse (5 IRQs per frame)."""
        self.advance(n * 5 * self.irq_period)

    # ---- build + enter play ---------------------------------------------------
    def build(self):
        """AUTHENTIC cold boot: start at the game entry $3F00 (init + JMP title)
        and navigate title -> 'LANDSCAPE NUMBER ?' -> 0000 (no secret code) ->
        generate -> play, by tapping '0' whenever the game waits for a key.
        Done when the IRQ input scan ($9678) runs, i.e. we are in play."""
        cpu, mem = self.cpu, self.mem
        mem[0x2625], mem[0x2626] = 0x18, 0x60  # plot_world -> CLC/RTS (carry clear)
        # at headless warp speed update_game_loop runs update_enemies thousands of
        # times before the first IRQ -> the Sentinel drains the player to death
        # before any input scan. Enemy AI is orthogonal to the input protocol.
        mem[0x16B5] = 0x60  # update_enemies -> RTS
        cpu.pc = 0x3F00
        cpu.sp = 0xFF
        cpu.p |= 0x04  # SEI until $3F12 CLI
        self.irq_on = True
        # get_number_into_input_buffer $32D5 exits ONLY on RETURN ($32EC CMP #$0D),
        # so the landscape prompt needs digit(s) then RETURN. Alternate '0' ($1C)
        # and RETURN ($08); neither is in the play key table $138D, so a stray tap
        # once in play is invisible to the game.
        for i in range(30):
            try:
                self.run_until_pc(PC_IRQ_SCAN, budget=1_500_000)
                return
            except SimTimeout:
                self.press_code(0x1C if i % 2 == 0 else 0x08)
                self.advance(20_000)
                self.release_all()
                self.advance(20_000)
        raise RuntimeError("never reached the play input scan")

    def snapshot(self):
        c = self.cpu
        return (bytes(self.mem), c.pc, c.sp, c.a, c.x, c.y, c.p, self.dc00, self.raster)

    def restore(self, snap):
        mem, pc, sp, a, x, y, p, dc00, raster = snap
        self.mem[:] = mem
        c = self.cpu
        c.pc, c.sp, c.a, c.x, c.y, c.p = pc, sp, a, x, y, p
        self.dc00, self.raster = dc00, raster
        self.matrix.clear()
        self.icount = 0
        self.next_irq = self.irq_period
        self.irq_on = True

    # ---- state reads -----------------------------------------------------------
    def rd(self, a):
        return self.mem[a]

    def slot(self):
        return self.rd(A_SLOT)

    def hang(self):
        return self.rd(A_H + self.slot())

    def vang(self):
        return self.rd(A_V + self.slot())

    def cur(self):
        return self.rd(A_CX), self.rd(A_CY)


class SimDriver:
    """The checkpoint protocol under test (press-while-halted ordering)."""

    def __init__(self, m: Machine):
        self.m = m

    # cursor: one checkpoint-confirmed pixel
    def cursor_step(self, key, sta_pc):
        m = self.m
        m.press(key)  # matrix mutates while CPU is halted
        m.run_until_pc(sta_pc)  # checkpoint armed before any time passes
        m.advance(1)  # execute the STA: exactly 1 px committed
        m.release_all()  # release while halted

    def cursor_axis(self, addr, want, key_inc, key_dec, inc_pc, dec_pc):
        m = self.m
        for _ in range(160):
            cur = m.rd(addr)
            if cur == want:
                return True
            if want > cur:
                self.cursor_step(key_inc, inc_pc)
            else:
                self.cursor_step(key_dec, dec_pc)
        return m.rd(addr) == want

    def fine_cursor(self, cx, cy):
        ok = self.cursor_axis(A_CX, cx, "D", "S", PC_CX_INC, PC_CX_DEC)
        return ok and self.cursor_axis(A_CY, cy, "L", "COMMA", PC_CY_INC, PC_CY_DEC)

    # pan: one committed step at a time, stop the instant memory equals want
    def pan_angle(self, addr, want, key, commit_pc, max_steps=64):
        m = self.m
        want &= 0xFF
        if m.rd(addr) == want:
            return True
        m.press(key)
        try:
            m.run_until_pc(commit_pc)
            for _ in range(max_steps):
                if m.rd(addr) == want:
                    break
                m.advance(1)
                m.run_until_pc(commit_pc)
        except SimTimeout:
            pass
        finally:
            m.release_all()
        return m.rd(addr) == want

    def coarse_h(self, want):
        m = self.m
        addr = A_H + m.slot()
        key = "D" if ((want - m.rd(addr)) & 0xFF) <= 0x80 else "S"
        return self.pan_angle(addr, want, key, PC_H_COMMIT)

    @staticmethod
    def _pitch_lin(v):
        v &= 0xFF
        return v - 0xCD if v >= 0xCD else v + 0x33

    def coarse_v(self, want):
        m = self.m
        addr = A_V + m.slot()
        key = "L" if self._pitch_lin(want) > self._pitch_lin(m.rd(addr)) else "COMMA"
        return self.pan_angle(addr, want, key, PC_V_COMMIT)

    # sights: anchored single-scan toggle
    def sights_set(self, on):
        m = self.m
        for _ in range(6):
            if bool(m.rd(A_SFLAG) & 0x80) == on:
                return True
            m.run_until_pc(PC_IRQ_SCAN)  # next guaranteed full input scan
            m.press("SPACE")
            m.run_until_pc(0x967B)  # scan (incl. SPACE toggle) done
            m.release_all()
        return bool(m.rd(A_SFLAG) & 0x80) == on

    # action: anchored single-scan latch + confirmed consumption. One full IDLE
    # scan first: update_game clears $0C51 ($1281) and only an idle full scan
    # re-arms it to $40 ($11EA) -- without it a u-turn is silently dropped at
    # $1B2F (ASL $0C51 / BPL reject).
    def tap_action(self, name, want_code, max_passes=45):
        m = self.m
        for _ in range(max_passes):
            m.run_until_pc(PC_IRQ_SCAN)  # full-scan boundary (keys released)
            m.advance(1)
            m.run_until_pc(PC_IRQ_SCAN)  # that idle scan completed; at the next
            m.press(name)
            m.run_until_pc(0x967B)  # exactly this one scan sees the key
            flags = bytes(m.mem[0x0CE8:0x0CEC])
            m.release_all()
            if want_code in flags:
                m.run_until_pc(PC_IRQ_SCAN)  # scans resume only after consumption
                return True
        return False


def run_suite(irq_period, snap, verbose=True):
    def log(s):
        if verbose:
            print(f"  [{irq_period}] {s}")

    m = Machine(irq_period)
    m.restore(snap)
    d = SimDriver(m)
    h0, v0, cur0 = m.hang(), m.vang(), m.cur()
    log(f"in play: h=${h0:02x} v=${v0:02x} cursor={cur0} sights={m.rd(A_SFLAG)>>7}")

    # 1. idle: nothing moves without keys
    m.run_frames(20)
    assert (m.hang(), m.vang()) == (h0, v0), "idle moved the view"

    # 2. sights on -> cursor recentred (80,95) by initialise_sights $134C
    assert d.sights_set(True), "sights would not turn on"
    assert m.cur() == (80, 95), f"sights-on cursor {m.cur()} != (80,95)"
    log("sights ON, cursor recentred (80,95)")

    # 3. cursor: exact pixel landing, one STA per step, none after release
    m.watch = {PC_CX_INC: 0, PC_CX_DEC: 0, PC_CY_INC: 0, PC_CY_DEC: 0}
    assert d.fine_cursor(66, 98), "fine_cursor(66,98) failed"
    assert m.cur() == (66, 98), f"cursor {m.cur()} != (66,98)"
    moves = sum(m.watch.values())
    assert moves == (80 - 66) + (98 - 95), f"extra cursor STAs: {moves}"
    m.run_frames(15)
    assert m.cur() == (66, 98), "cursor drifted after release"
    log(f"cursor -> (66,98) exact, {moves} committed moves, no drift")

    # 4. cursor: far target near the safe-band edge
    assert d.fine_cursor(140, 150) and m.cur() == (140, 150)
    assert (m.hang(), m.vang()) == (h0, v0), "cursor drive panned the view"
    assert d.fine_cursor(20, 40) and m.cur() == (20, 40)
    log("cursor -> (140,150) -> (20,40) exact, no wrap-pan")

    # 5. sights off, coarse_h: +5 lattice steps then back, exact
    assert d.sights_set(False)
    m.watch = {PC_H_COMMIT: 0}
    want = (h0 + 5 * 8) & 0xFF
    assert d.coarse_h(want) and m.hang() == want, "coarse_h overshoot/miss"
    assert m.watch[PC_H_COMMIT] == 5, f"h commits {m.watch[PC_H_COMMIT]} != 5"
    assert d.coarse_h(h0) and m.hang() == h0
    m.run_frames(20)
    assert m.hang() == h0, "h drifted after release"
    log(f"coarse_h +40 -> back, 5+5 commits, no drift")

    # 6. coarse_v: cross the $FF->$00 wrap inside the clamp band, then back
    want_v = 0x0D if v0 == 0xF5 else ((v0 + 6 * 4) & 0xFF)
    assert d.coarse_v(want_v) and m.vang() == want_v, "coarse_v miss"
    assert d.coarse_v(v0) and m.vang() == v0
    log(f"coarse_v ${v0:02x} -> ${want_v:02x} -> back, exact")

    # 7. clamp: drive to the down limit $35, then one more press never commits
    assert d.coarse_v(0x35) and m.vang() == 0x35
    m.advance(1)  # step off $1135: run_until_pc fast-paths if already at target
    m.press("L")
    try:
        m.run_until_pc(PC_V_COMMIT, budget=30 * 5 * irq_period)
        raise AssertionError("pan committed past the $35 clamp")
    except SimTimeout:
        pass
    m.release_all()
    assert m.vang() == 0x35, "clamp overshot"
    assert d.coarse_v(v0) and m.vang() == v0
    log("v clamp $35 holds: held key never commits, no overshoot")

    # 8. tap_action: u-turn fires exactly once
    m.watch = {PC_ACTION_ACCEPT: 0}
    h_before = m.hang()
    assert d.tap_action("U", 0x23), "u-turn never latched"
    assert m.hang() == (h_before ^ 0x80), f"u-turn h ${m.hang():02x}"
    m.run_frames(30)
    assert m.hang() == (h_before ^ 0x80), "u-turn re-fired after release"
    assert (
        m.watch[PC_ACTION_ACCEPT] == 1
    ), f"action accepted {m.watch[PC_ACTION_ACCEPT]}x"
    assert d.tap_action("U", 0x23) and m.hang() == h_before
    log("tap_action U: exactly one accept, no repeat, inverse restores")

    # 9. negative control: naive hold (no checkpoints) bursts the cursor
    assert d.sights_set(True)
    d.fine_cursor(80, 95)
    m.press("D")
    m.run_frames(12)
    m.release_all()
    burst = m.cur()[0] - 80
    assert burst >= 3, f"expected an auto-repeat burst, got {burst}px"
    log(f"negative control: 12-frame naive hold moved {burst}px (cadence-dependent)")
    # re-home via the protocol: the burst is cadence-dependent, the protocol is not
    assert d.fine_cursor(80, 95) and m.cur() == (80, 95)

    return {
        "h": m.hang(),
        "v": m.vang(),
        "cur": m.cur(),
        "icount": m.icount,
        "burst": burst,
    }


def main():
    periods = [int(p) for p in (sys.argv[1:] or ["1109"])]
    import pickle

    snap_path = os.path.join(os.path.dirname(_SCRIPTS), "out", "play_snapshot.pkl")
    if os.path.exists(snap_path):
        snap = pickle.load(open(snap_path, "rb"))
        print("loaded play snapshot")
    else:
        boot = Machine(1109)
        boot.build()
        snap = boot.snapshot()
        pickle.dump(snap, open(snap_path, "wb"))
        print(f"booted to play in {boot.icount:,} instructions; snapshot saved")
    results = {}
    for p in periods:
        print(f"IRQ period {p} instructions:")
        results[p] = run_suite(p, snap)
        print(
            f"  [{p}] OK  final h=${results[p]['h']:02x} v=${results[p]['v']:02x} "
            f"cur={results[p]['cur']} ({results[p]['icount']:,} instr)"
        )
    finals = {(r["h"], r["v"], r["cur"]) for r in results.values()}
    assert len(finals) == 1, f"cadence-dependent outcome: {results}"
    print(f"\nALL PASS: identical final state across cadences {periods}")


if __name__ == "__main__":
    main()
