#!/usr/bin/env python3
"""A vagrant-moses C64 behind the BinMon surface the live driver uses.

Waits are cycle budgets (``timeout`` seconds of PAL time) that raise like an expired
VICE wait. Checkpoints follow VICE's ``DO_INTERRUPT``: a step that ends at a boundary
stops before that boundary's checkpoint test, which runs on the next resume.
"""

import contextlib
import copy
import dataclasses

import numpy as np
from vagrant_moses import memmap as vmm
from vice_driver.binmon import CHECK_EXEC, TAP_MODE_FIXED, BinmonError, Checkpoint

from driver.clock import FRAME_PC
from sentinel.passcost import PAL_FRAME_CYCLES

_CPU_BANKS = (0, "default", "cpu")
_REG_A, _REG_X, _REG_Y, _REG_PC, _REG_SP, _REG_FLAGS = range(6)
_REG_RASTER, _REG_RASTER_CYCLE = 53, 54  # VICE monitor register ids


def cycles(seconds):
    """A wait's cycle budget: ``seconds`` of PAL time."""
    return max(1, int(seconds * vmm.PAL.clock_hz))


class EmuMon:
    """BinMon over a vagrant-moses ``Machine``, whose trap it owns.

    Every stop leaves the CPU about to execute the instruction at ``pc``. The machine's
    unread ``events`` log is emptied on every machine taken, so forks do not copy it."""

    def __init__(self, machine):
        self.m = machine
        self.auto_resume = False
        self._cps = {}
        self._at = {}  # pc -> enabled checkpoints there
        self._run_cp = {}  # run_until_pc's stop checkpoint per target
        self._steps = 0
        self._owed = False  # a step stopped here before this boundary's test
        self._parked = False  # stopped at a decode, by a step or a checkpoint
        self._hits = set()
        self._tap = None  # (release cycle, keys) of a TAP_MODE_FIXED chord
        machine.clear_events()
        machine.set_trap(self._trap, set())

    def clone(self, machine=None):
        """An independent monitor on ``machine`` (default ``self.m.clone()``) carrying
        every checkpoint and count, the checkpoint caches kept on it, the owed boundary
        test and a held tap, so both run identically from here."""
        twin = copy.copy(self)
        twin.m = self.m.clone() if machine is None else machine
        for name, value in vars(self).items():
            if isinstance(value, dict):  # checkpoint caches callers keep here too
                setattr(twin, name, dict(value))
        twin._cps = {n: dataclasses.replace(cp) for n, cp in self._cps.items()}
        twin._hits = set(self._hits)
        twin.m.clear_events()
        twin.m.set_trap(twin._trap, set())
        twin._watch()
        return twin

    def adopt(self, machine):
        """Run on ``machine`` from here (a clone of this one's), under this trap."""
        self.m = machine
        machine.clear_events()
        machine.set_trap(self._trap, set())
        self._watch()

    def native_ready(self):
        """Whether a driver program can run natively here: no held tap, and the only
        enabled checkpoints count the frame marker without stopping."""
        return self._tap is None and all(
            cp.start == cp.end == FRAME_PC and not cp.stop_when_hit
            for cp in self._cps.values()
            if cp.enabled
        )

    def owes_marker(self):
        """Whether a step stopped on the frame marker before its test, which the next
        resume counts (a native run's marker count has it already decoded)."""
        return self._owed and self.m.mpu.pc == FRAME_PC

    def ran(self, frames, owed):
        """Account a native driver program: ``frames`` marker arrivals counted, and
        (unless ``owed`` is None, no run op) parked owing the test when a step ended it.
        """
        for cp in self._cps.values():
            if cp.enabled:
                cp.hit_count += frames
        if owed is not None:
            self._owed, self._parked = owed, True

    def connect(self, *_a, **_kw):
        """No socket to open."""

    def close(self):
        """No socket to close."""

    def exit(self):
        """VICE free-runs here; the emulator runs only inside waits."""

    def halted(self):
        """BinMon.halted: hold ``auto_resume`` off for the block."""

        @contextlib.contextmanager
        def hold():
            saved, self.auto_resume = self.auto_resume, False
            try:
                yield
            finally:
                self.auto_resume = saved

        return hold()

    # ---- the boundary hook -------------------------------------------------
    def _trap(self, m):
        if self._steps:
            self._steps -= 1
            if not self._steps:
                self._owed = True
                self._watch()
                m.stop = True
                return False
        self._test(m.mpu.pc)
        return False

    def _test(self, pc):
        for cp in self._at.get(pc, ()):
            cp.hit_count += 1
            if cp.stop_when_hit:
                self._hits.add(cp.checknum)
                self.m.stop = True

    def _watch(self):
        self._at = {}
        for cp in self._cps.values():
            for pc in range(cp.start, cp.end + 1) if cp.enabled else ():
                self._at.setdefault(pc, []).append(cp)
        self.m.watch = self.m.WATCH_ALL if self._steps else frozenset(self._at)

    def _now(self):
        """VICE's clock at this stop: the boundary before the decode cycle."""
        return self.m.cycle - self._parked

    def _resume(self, budget):
        """Run until a step ends or a stop checkpoint hits; returns the hits. A spent
        budget raises, leaving the machine where it stood."""
        self._hits = set()
        if self._owed:
            self._owed = False
            self._test(self.m.mpu.pc)
            if self._hits:
                return self._hits
        self._parked = False
        end = self.m.cycle + budget
        while True:
            self.m.run(end if self._tap is None else min(end, self._tap[0]))
            if self._tap is not None and self.m.cycle >= self._tap[0]:
                self._release_tap()
            if self._hits or self._owed:
                self._parked = True
                return self._hits
            if self.m.jammed:
                raise BinmonError(f"CPU JAM at ${self.m.mpu.pc:04X}")
            if self.m.cycle >= end:
                raise BinmonError(f"wait timed out after {budget} cycles")

    # ---- execution -----------------------------------------------------------
    def advance_instructions(self, count=1, step_over_subroutines=False):
        if step_over_subroutines:
            raise NotImplementedError("step over subroutines")
        self._steps = count
        self._watch()
        self._resume(PAL_FRAME_CYCLES * count)

    def run_until_pc(self, target, timeout, condition=None):
        if condition is not None:
            raise NotImplementedError("conditional run_until_pc")
        if self.m.mpu.pc == target:
            return
        cp = self._run_cp.get(target)
        if cp is None:
            cp = self.checkpoint_set(target, stop_when_hit=True, enabled=False)
            self._run_cp[target] = cp
        self.checkpoint_toggle(cp.checknum, True)
        try:
            self.wait_for_checkpoint(cp.checknum, timeout)
        finally:
            self.checkpoint_toggle(cp.checknum, False)

    def wait_for_checkpoint(self, checknum, timeout):
        self.wait_for_checkpoints((checknum,), timeout)

    def wait_for_checkpoints(self, checknums, timeout):
        """Resume until any of ``checknums`` hits; the ones that did."""
        hits = self._resume(cycles(timeout)) & set(checknums)
        if not hits:
            raise BinmonError(
                f"checkpoints {sorted(checknums)} not hit: stopped elsewhere"
            )
        return hits

    # ---- checkpoints ---------------------------------------------------------
    def checkpoint_set(
        self,
        start,
        end=None,
        op=CHECK_EXEC,
        stop_when_hit=False,
        enabled=True,
        temporary=False,
        memspace=0,
        silent=False,
    ):
        if op != CHECK_EXEC or temporary or memspace:
            raise NotImplementedError("only plain exec checkpoints")
        num = len(self._cps) + 1
        end = start if end is None else end
        self._cps[num] = Checkpoint(
            num, False, start, end, stop_when_hit, enabled, op, False, 0, 0, False, 0
        )
        self._watch()
        return self.checkpoint_get(num)

    def checkpoint_get(self, checknum):
        return dataclasses.replace(self._cps[checknum])

    def checkpoint_toggle(self, checknum, enabled):
        self._cps[checknum].enabled = bool(enabled)
        self._watch()

    # ---- memory and registers --------------------------------------------------
    def _io(self):
        """The I/O page as VICE's monitor peeks it, without read side effects: the
        VIC-II live, colour RAM, the CIA register files; SID and expansion as zero."""
        m = self.m
        page = np.zeros(0x1000, np.uint8)
        page[:0x400] = np.tile([m.io_read(0xD000 + r) for r in range(0x40)], 16)
        page[0x800:0xC00] = np.frombuffer(m.colour, np.uint8) & 0x0F
        page[0xC00:0xD00] = np.tile(np.frombuffer(m.cia1.regs, np.uint8), 16)
        page[0xD00:0xE00] = np.tile(np.frombuffer(m.cia2.regs, np.uint8), 16)
        return page

    def mem_get(self, start, end, side_effects=False, memspace=0, bank=0):
        """``[start, end]`` of the CPU view (bank 0), RAM, or RAM with the I/O page."""
        if start > end:
            raise BinmonError(f"mem_get: start ({start:#x}) > end ({end:#x})")
        m = self.m
        img = np.frombuffer(m.ram, np.uint8)
        if bank in _CPU_BANKS:
            img = img.copy()
            for i, page in enumerate(m.pages):
                if page is not None and start >> 12 <= i <= end >> 12:
                    span = slice(i << 12, (i + 1) << 12)
                    img[span] = np.frombuffer(page, np.uint8)[span]
        elif bank not in ("ram", "io"):
            raise BinmonError(f"unknown bank {bank!r}")
        io = m.pages[0xD] is None if bank in _CPU_BANKS else bank == "io"
        if io and start <= 0xDFFF and end >= 0xD000:
            img = img.copy() if bank == "io" else img
            img[0xD000:0xE000] = self._io()
        return img[start : end + 1].tobytes()

    def mem_set(self, start, data, side_effects=False, memspace=0, bank=0):
        """CPU-view writes: the I/O page when mapped, RAM everywhere else (also
        under a ROM)."""
        io = bank == "io" or (bank in _CPU_BANKS and self.m.pages[0xD] is None)
        for addr, value in enumerate(bytes(data), start):
            if io and addr >> 12 == 0xD:
                self.m.io_write(addr, value)
            else:
                self.m.ram[addr & 0xFFFF] = value

    def registers_get(self, memspace=0):
        mpu, vic = self.m.mpu, self.m.vic
        return {
            _REG_A: mpu.a,
            _REG_X: mpu.x,
            _REG_Y: mpu.y,
            _REG_PC: mpu.pc,
            _REG_SP: mpu.sp,
            _REG_FLAGS: mpu.p,
            _REG_RASTER: vic.raster_y,
            _REG_RASTER_CYCLE: vic.line_cycle,
        }

    # ---- keyboard ----------------------------------------------------------------
    def keymatrix_set(self, keys):
        for r, c, v in keys:
            if r >= 0:
                self.m.set_key(r, c, bool(v))

    def keymatrix_release_all(self):
        self.m.release_all()

    def keymatrix_tap(self, keys, mode=TAP_MODE_FIXED, frames=60):
        """Hold a chord for ``frames`` PAL frames of cycles from now, released by a
        clock alarm as asid-vice's fixed tap is; a newer tap supersedes it."""
        if mode != TAP_MODE_FIXED:
            raise NotImplementedError("only TAP_MODE_FIXED taps")
        if self._tap is not None:
            self._release_tap()
        keys = [tuple(k) for k in keys]
        for r, c in keys:
            self.m.set_key(r, c, True)
        self._tap = (self._now() + frames * PAL_FRAME_CYCLES, keys)

    def _release_tap(self):
        for r, c in self._tap[1]:
            self.m.set_key(r, c, False)
        self._tap = None
