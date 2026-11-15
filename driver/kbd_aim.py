#!/usr/bin/env python3
"""KEYBOARD aim on live VICE: drive the real keys to an exact (h, v, cursor) view.

D/S pan h_angle by +-8 and L/COMMA v_angle by +-4 (the cursor wrapping on a 9px grid), U is
h_angle EOR $80; the ROM fires along h + cur_x>>3, v + (cur_y-5)>>4 ($1C10, modelled by
:mod:`sentinel.los`). Every step is verified from memory reads: no pixels, no angle pokes.
"""

import os

from sentinel import aimcost as ac
from sentinel import memmap as mm

# run_until_pc hang guard: a commit checkpoint recurs every emulated frame, so a timeout here is a WAIT ON A PC OR CONDITION THAT CANNOT RECUR (clamped pan, left the play loop) -- a bug to fix, not back-pressure to wait out.
_RU_COMMIT = float(
    os.environ.get("KBD_COMMIT_TIMEOUT", "4")
)  # socket backstop, 1 frame
_PAN_STALL_FRAMES = (
    24  # > one notch scroll (16 h / 8 v): no commit this long => clamped
)
_PAN_MAX_FRAMES = (
    400  # > a full pan (256 h / 208 v frames); real targets need far fewer
)
_SCAN_WAIT_PASSES = int(
    os.environ.get("KBD_SCAN_WAIT_PASSES", "10")
)  # hang guard only: a scan wait is bounded by the plot ($0CE4), never by a clock

A_SLOT = mm.PLAYER_OBJECT  # 0x000B
A_H = mm.OBJECTS_H_ANGLE  # 0x09C0 + slot
A_V = mm.OBJECTS_V_ANGLE  # 0x0140 + slot
A_CX = 0x0CC6  # sights cursor column (no memmap entry)
A_CY = 0x0CC7  # sights cursor row (no memmap entry)
A_SFLAG = 0x0C5F  # sights-on flag, bit7 (no memmap entry)
A_PLOT = 0x0CE4  # busy-plotting gate, bit7 (no memmap entry)

K_RIGHT, K_LEFT, K_DOWN, K_UP, K_UTURN = "D", "S", "L", "COMMA", "U"
PC_PASS = 0x363D  # update_game_and_continue: a main-loop pass begins
PC_PAN_CALL = 0x365A  # JSR pan_viewpoint $10B7

# action latch $0CE9 = action-table byte >> 2 (re-armed to $80 at $1363 each pass)
ACTION_CODE = {
    "R": 0x00,
    "T": 0x02,
    "B": 0x03,
    "A": 0x20,
    "Q": 0x21,
    "H": 0x22,
    "U": 0x23,
}


class KbdDriver:
    """Drive the real keys to an exact (h_angle, v_angle, cursor) using the verified
    pan cycles. All gating/feedback is from memory reads."""

    def __init__(self, bm, log, quantized=False):
        self.bm = bm
        self.log = log
        # quantized: the CPU stays halted between primitives, so only run windows advance it
        self.quantized = quantized
        # same-bearing aims skip the sights toggle whose $134C recenters the cursor; None=stale
        self._bearing = None

    def rd(self, a):
        return self.bm.mem_get(a, a)[0]

    def _resume(self):
        """Resume the CPU at a halted primitive's tail; a no-op in quantized mode, where
        the game advances only during explicit run_until_pc windows."""
        if not self.quantized:
            self.bm.exit()

    def slot(self):
        return self.rd(A_SLOT)

    def hang(self):
        return self.rd(A_H + self.slot())

    def vang(self):
        return self.rd(A_V + self.slot())

    def cur(self):
        return self.rd(A_CX), self.rd(A_CY)

    def sights_live_on(self):
        """True iff the sights flag ($0C5F bit7) is set. SPACE is the only sights toggle
        ($11B3), so this flag is stable regardless of panning -- safe to read any time.
        """
        return bool(self.rd(A_SFLAG) & 0x80)

    def committed_bearing(self):
        """The last committed (h_angle, v_angle) for the current player slot, or None when
        unknown/stale (never aimed, an aim did not converge, a monitor drop, a slot change).
        """
        return self._bearing

    def set_bearing(self, h, v):
        self._bearing = (h & 0xFF, v & 0xFF)

    def clear_bearing(self):
        self._bearing = None

    # after JSR pan_viewpoint: reached once per pan attempt on commit, undo and clamp alike
    PC_PAN_DONE = 0x365D
    # move_sights STA PCs (no undo path): run_until_pc halts before the STA, value in A
    PC_CX_INC = 0x997C  # STA $0CC6 (move_sights right, cx+1)
    PC_CX_DEC = 0x9990  # STA $0CC6 (move_sights left,  cx-1)
    PC_CY_INC = 0x99B8  # STA $0CC7 (move_sights down,  cy+1)
    PC_CY_DEC = 0x99D2  # STA $0CC7 (move_sights up,    cy-1)
    PC_IRQ_SCAN = 0x9678  # IRQ: JSR check_for_full_player_input (gated full scan)
    PC_IRQ_SCAN_DONE = 0x967B  # return address of that JSR: the scan has completed

    def _pan_angle(self, addr, want, dir_fn):
        """Hold the pan key, stepping passes at PC_PAN_DONE, until the angle at `addr` is `want`.

        Ends only on state: "ok" at want, "hyperspace" when player_object ($0B) changes, or
        "unreachable" when a full notch-scroll passes with no change (clamped/off-lattice).
        """
        want &= 0xFF
        slot0 = self.rd(A_SLOT)
        stalled = 0
        stall_bail = _PAN_STALL_FRAMES  # > one notch scroll (16 frames h / 8 v)
        held = None
        result = "unreachable"
        with self.bm.halted():
            try:
                for _ in range(_PAN_MAX_FRAMES):
                    cur = self.rd(addr)
                    if cur == want:
                        result = "ok"
                        break
                    if self.rd(A_SLOT) != slot0:
                        result = "hyperspace"
                        break
                    key = dir_fn(cur)
                    if (
                        key != held
                    ):  # (re)press only on a direction change; HOLD otherwise
                        self.bm.keymatrix_release_all()
                        r, c = _k(key)
                        self.bm.keymatrix_set([(r, c, 1)])
                        held = key
                    self._run_pass()
                    new = self.rd(addr)
                    self.bm.advance_instructions(1)  # step off $365D
                    if new == cur:
                        stalled += 1
                        if stalled >= stall_bail:
                            break  # a full scroll with no commit: clamped / off-lattice
                    else:
                        stalled = 0
            finally:
                self.bm.keymatrix_release_all()
        self._resume()
        return result

    def _run_pass(self):
        """Run one main-loop pass to PC_PAN_DONE ($365D): the previous notch's scroll
        loop, update_game $127C and this pass's pan_viewpoint $10B7."""
        self._run_past_redraw(self.PC_PAN_DONE, _RU_COMMIT)

    def _run_past_redraw(self, pc, timeout):
        """``run_until_pc(pc)``, re-armed while a timeout finds the world being plotted
        ($0CE4 bit7): a redraw still running delays ``pc``, it does not stall it."""
        for _ in range(_SCAN_WAIT_PASSES):
            try:
                self.bm.run_until_pc(pc, timeout=timeout)
                return True
            except Exception:
                if not self.rd(A_PLOT) & 0x80:
                    raise
        return False

    def _run_to_scan(self, timeout=6.0):
        """Run to the next GATED full input scan ($9678), reached only while the world is not
        plotted ($0CE4 bit7, held across a whole redraw) and the $130B re-arm has expired.
        A timeout with the gate set is a redraw still running, so re-arm rather than leak its
        frames into the next primitive; a timeout with the gate open is a real stall."""
        return self._run_past_redraw(self.PC_IRQ_SCAN, timeout)

    def _one_scan_press(self):
        """Hold SPACE for EXACTLY ONE gated full input scan ($9678->$967B), after ONE idle scan.

        The toggle is edge-latched: $11B5 tests $1236, which $11D4 clears only on a scan with
        SPACE up, so back-to-back presses with no released-key scan swallow the second.
        """
        r, c = _k("SPACE")
        with self.bm.halted():
            try:
                self._run_to_scan(10.0)
                self.bm.advance_instructions(1)  # off the anchor, into the scan
                self._run_to_scan(10.0)  # that idle scan re-armed the latch
                self.bm.keymatrix_set([(r, c, 1)])  # press WHILE HALTED
                self.bm.run_until_pc(self.PC_IRQ_SCAN_DONE, timeout=10.0)
            finally:
                self.bm.keymatrix_release_all()
        self._resume()

    def sights_set(self, on):
        """Toggle SPACE (edge-latched $1236) until the sights flag ($0C5F bit7) matches.
        One `_one_scan_press` == exactly one toggle (it re-arms the latch first), so the
        loop is a fault backstop, not the normal path."""
        for _ in range(6):
            if bool(self.rd(A_SFLAG) & 0x80) == on:
                return True
            self._one_scan_press()
        return bool(self.rd(A_SFLAG) & 0x80) == on

    def sights_on(self):
        return self.sights_set(True)

    def _uturn(self):
        """SIGHTS OFF: flip the bearing 180 degrees in ONE keystroke (handle_uturn $1B2F).

        The EOR $80 lands in update_game, not in the scan that latched the press, so the flip
        is confirmed only after ``tap_action`` lets the action be consumed; retrying on an
        earlier read would press again and an even number of flips cancels."""
        addr = A_H + self.slot()
        for _ in range(5):
            before = self.rd(addr)
            if not self.tap_action(K_UTURN):
                continue
            if ((self.rd(addr) - before) & 0xFF) == 0x80:
                return True
        return False

    def coarse_h(self, want):
        """SIGHTS OFF: rotate bearing h to `want` (±8 lattice, wraps mod 256), shorter way first.

        ``aimcost.h_press_count`` decides whether a U-turn plus a short correction beats
        panning round; the ±8 loop finishes the residual whether or not the U-turn latched.
        """
        addr = A_H + self.slot()
        want &= 0xFF
        cur = self.rd(addr)
        n_uturn, n_step = ac.h_press_count(cur, want)
        if n_uturn and not self._uturn():
            # the residual pan is the half-turn the cost model priced as one keystroke
            self.log(
                f"    uturn MISSED ${cur:02x}->${want:02x}: paying "
                f"{ac.h_steps(cur, want)} notches instead of {n_step}"
            )
        dir_fn = lambda cur: K_RIGHT if ((want - cur) & 0xFF) <= 0x80 else K_LEFT
        return self._pan_angle(addr, want, dir_fn)

    @staticmethod
    def _pitch_lin(v):
        """Linearise clamped pitch. The keyboard-reachable band is [$CD..$FF]U[$00..$35],
        contiguous through the $FF->$00 wrap; L (up, +4) advances it, COMMA (down, -4)
        retreats. Map to a monotonic 0..104 coordinate so direction is a plain compare
        (a raw signed/unsigned compare is wrong across the $00 wrap)."""
        v &= 0xFF
        return v - 0xCD if v >= 0xCD else v + 0x33

    def coarse_v(self, want):
        """SIGHTS OFF: pitch v to `want` (±4 lattice, clamped band wraps through $00).
        L raises pitch (+4), COMMA lowers it (-4); direction is chosen per attempt."""
        addr = A_V + self.slot()
        want &= 0xFF
        if not (want >= 0xCD or want <= 0x35):
            return "unreachable"  # off the $CD..$35 pan band: proposer bug, don't drive
        dir_fn = lambda cur: (
            K_DOWN if self._pitch_lin(want) > self._pitch_lin(cur) else K_UP
        )
        return self._pan_angle(addr, want, dir_fn)

    def coarse_hv(self, view):
        """SIGHTS OFF: pan bearing then pitch to `view`, retrying once when either missed.

        Returns ``(okh, okv, status)``. "hyperspace" skips the retry: the player teleported
        mid-aim, so the requested angles belong to the wrong eye."""
        okh = self.coarse_h(view["h_angle"])
        okv = self.coarse_v(view["v_angle"])
        if "hyperspace" not in (okh, okv) and (okh != "ok" or okv != "ok"):
            okh = self.coarse_h(view["h_angle"])
            okv = self.coarse_v(view["v_angle"])
        status = (
            "hyperspace"
            if "hyperspace" in (okh, okv)
            else ("unreachable" if "unreachable" in (okh, okv) else "ok")
        )
        return okh, okv, status

    def fine_cursor(self, cx, cy):
        """SIGHTS ON: drive the cursor to (cx, cy) DIAGONALLY in ONE halted section.

        move_sights ($9958) steps cx and cy in one call, so travel is max(|dx|,|dy|); the only
        CPU time is each pixel's run-to-commit, with no free-running gaps between pixels.
        """
        cx &= 0xFF
        cy &= 0xFF
        ok = False
        with self.bm.halted():
            for _ in range(160):
                curx, cury = self.rd(A_CX), self.rd(A_CY)
                if curx == cx and cury == cy:
                    ok = True
                    break
                keys, commit_pc = [], None
                if curx != cx:
                    keys.append(K_RIGHT if cx > curx else K_LEFT)
                    commit_pc = self.PC_CX_INC if cx > curx else self.PC_CX_DEC
                if cury != cy:  # cy write runs last in move_sights -> commit point
                    keys.append(K_DOWN if cy > cury else K_UP)
                    commit_pc = self.PC_CY_INC if cy > cury else self.PC_CY_DEC
                presses = [(*_k(key), 1) for key in keys]
                try:
                    self.bm.keymatrix_set(presses)
                    self.bm.run_until_pc(commit_pc, timeout=1.5)
                    self.bm.advance_instructions(1)  # execute the last STA
                except Exception:
                    pass
                finally:
                    self.bm.keymatrix_release_all()
        self._resume()
        return ok or (self.rd(A_CX) == cx and self.rd(A_CY) == cy)

    def tap_action(self, name, settle=True):
        """Fire an action key EXACTLY ONCE, pressed halted at $9678 and released at $967B.

        An idle full scan first re-arms $0C51 ($11EA), without which a u-turn latch drops at
        $1B2F. The caller verifies the memory effect and retries a miss (at-least-once).
        """
        want = ACTION_CODE[name]
        r, c = _k(name)
        latched = False
        with self.bm.halted():
            try:
                for _ in range(45):
                    self._run_to_scan()
                    self.bm.advance_instructions(1)  # off the anchor
                    self._run_to_scan()  # idle scan ran
                    self.bm.keymatrix_set([(r, c, 1)])  # press WHILE HALTED
                    self.bm.run_until_pc(self.PC_IRQ_SCAN_DONE, timeout=6.0)
                    flags = self.bm.mem_get(0x0CE8, 0x0CEB)
                    self.bm.keymatrix_release_all()  # before the next scan
                    if want in flags:
                        latched = True
                        if not settle:
                            break  # caller settles itself (a hyperspace scan may never recur)
                        # scans reopen after $12D0 consumes the action: this run spans the settle
                        try:

                            if self.quantized:
                                self._run_to_scan()
                            else:
                                self.bm.run_until_pc(self.PC_IRQ_SCAN, timeout=1.0)
                        except Exception:
                            pass
                        break
            except Exception as e:
                self.log(f"    tap_action {name} stop: {type(e).__name__}")
            finally:
                self.bm.keymatrix_release_all()
        self._resume()
        return latched

    def drive_to(self, view):
        """Drive to the snapped (h, v, cursor). COARSE with sights OFF, then FINE with
        sights ON (cursor). Every sub-step result is checked; returns the achieved dict.
        """
        if not self.sights_set(False):  # coarse: fast rotate
            return {"h": self.hang(), "v": self.vang(), "cur": self.cur(), "ok": False}
        okh, okv, status = self.coarse_hv(view)
        if status == "hyperspace":  # teleported mid-aim: abort, do not re-drive
            return {
                "h": self.hang(),
                "v": self.vang(),
                "cur": self.cur(),
                "ok": False,
                "status": "hyperspace",
            }
        oks = self.sights_set(True)  # fine: cursor selection
        cx, cy = view["cursor"]
        okc = self.fine_cursor(cx, cy)
        return {
            "h": self.hang(),
            "v": self.vang(),
            "cur": self.cur(),
            "ok": bool(okh == "ok" and okv == "ok" and oks and okc),
            "status": status,
        }


def _k(name):
    from vice_driver import keys

    return keys.lookup(name)
