#!/usr/bin/env python3
"""Precise KEYBOARD aim for the live VICE Sentinel replay, built on the EXPERIMENTALLY
VERIFIED pan primitive:

  * D (sight right): cursor_x cycles 80,89,98,...,143 then WRAPS to 80 and h_angle += 8.
  * S (sight left) : cursor_x cycles down 80,71,...,17 then WRAPS to 79 and h_angle -= 8.
  * L (sight down) : cursor_y cycles 95,104,...,158 then WRAPS to 96 and v_angle += 4.
  * COMMA (up)     : cursor_y cycles 95,86,...,32  then WRAPS to 95 and v_angle -= 4.
  * U (u-turn)     : h_angle EOR $80.

So h_angle is a +-8 keyboard grid (reachable = h0 + 8k), v_angle a +-4 grid, and the
cursor moves on a 9px grid (cx in {17..143}, cy in {32..158}). The aim that the action
fires on is computed by the ROM's prepare_vector_from_player_sights ($1C10):
h_eff = h + cur_x>>3, v_eff = v + (cur_y-5)>>4 -- which sentinel.los models exactly.

The caller proposes a (h, v, cursor) view (``sentinel.aim.propose`` / ``sentinel.los``,
bit-exact vs the ROM); this module drives the real keys to that exact view -- verified
from memory reads -- and the caller probes the live ROM LOS to confirm before pressing
the action key. No pixels, no angle pokes.
"""

import os

from sentinel import aimcost as ac

# run_until_pc hang guards. MEASURED: VICE services the binary monitor once per
# emulated frame while the CPU runs (halted 0.04 ms/cmd, warp ~1.4 ms, real-time
# pace ~23.5 ms == one PAL frame, independent of read size -- the encoder/disk are
# irrelevant; recording matters only because video_record forces warp off). A pan
# checkpoint recurs every frame, so a timeout here means a WAIT ON A PC OR
# CONDITION THAT CANNOT RECUR (clamped pan, left the play loop) -- a bug to fix,
# not back-pressure to wait out. Generous values only cost time on such bugs.
_RU_PAN = float(os.environ.get("KBD_PAN_TIMEOUT", "20"))
_RU_STA = float(os.environ.get("KBD_STA_TIMEOUT", "8"))
_RU_COMMIT = float(
    os.environ.get("KBD_COMMIT_TIMEOUT", "4")
)  # socket backstop, one frame
_PAN_STALL_FRAMES = (
    24  # > one notch scroll (16 h / 8 v): no commit this long => clamped
)
_PAN_MAX_FRAMES = (
    400  # > a full pan (256 h / 208 v frames); real targets need far fewer
)
_SCAN_WAIT_PASSES = int(
    os.environ.get("KBD_SCAN_WAIT_PASSES", "10")
)  # hang guard only: a scan wait is bounded by the plot ($0CE4), never by a clock

A_SLOT = 0x000B
A_H = 0x09C0
A_V = 0x0140
A_CX = 0x0CC6
A_CY = 0x0CC7
A_SFLAG = 0x0C5F
A_PLOT = 0x0CE4

K_RIGHT, K_LEFT, K_DOWN, K_UP, K_UTURN = "D", "S", "L", "COMMA", "U"

# the game's action latch $0CE9 = action-table byte >> 2 (re-armed to $80 each pass at
# check_for_player_input $1363); poll it to confirm an action key was consumed (C8).
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
        # QUANTIZED live cadence: when True the CPU stays HALTED between primitives -- the
        # emulator advances ONLY inside the explicit run_until_pc animation windows (pan,
        # cursor, action settle), never during the auto-resuming socket reads of the
        # bookkeeping between steps. That monitor free-run (game running at gameplay speed
        # while Python does round-trips) inflated live per-step frames ~3x past the ROM
        # animation the cost model prices ($1FA4 dither + $2625 replot), pushing the player
        # past the ~450-frame drain onset that the model keeps it under. Off (default) keeps
        # the legacy resume-between-primitives behaviour for boot/aim helpers.
        self.quantized = quantized
        # Committed view bearing (objects_h/v_angle for the CURRENT player slot), tracked
        # so a follow-on same-bearing aim can keep sights ON and drive ONLY the cursor,
        # skipping the sights OFF->ON toggle whose initialise_sights ($134C) recenters the
        # cursor to ($50,$5F). None == unknown/stale.
        self._bearing = None

    def rd(self, a):
        return self.bm.mem_get(a, a)[0]

    def _resume(self):
        """Resume the CPU at a halted primitive's tail. No-op in quantized mode: the CPU
        stays HALTED so the game advances only during explicit run_until_pc windows, and
        the auto-resuming reads of inter-step bookkeeping never free-run the emulator.
        """
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

    # ---- checkpoint-driven primitives (no wall-clock timing) ----
    # PCs from the game's ROM routines.
    # The one PC reached ONCE PER PAN ATTEMPT on EVERY outcome -- commit, undo, AND clamp --
    # and for BOTH axes: the instruction right after `JSR pan_viewpoint` in the foreground
    # loop ($365A). A pan-commit STA PC is reached only when plot_world returns carry-clear;
    # when it returns carry-set (check_if_player_still_wants_to_pan aborts the plot) the pan
    # is UNDONE and the commit PC is never hit -- which is why anchoring on a commit PC needs
    # a wall-clock timeout to notice a stall. $365D needs none: the game always reaches it, so
    # we step one attempt, read the settled angle, and decide from STATE, never from a clock.
    PC_PAN_DONE = 0x365D
    # Cursor STA PCs (move_sights; no undo path) -- run_until_pc halts BEFORE the STA, so the
    # value about to be stored is read from A.
    PC_CX_INC = 0x997C  # STA $0CC6 (move_sights right, cx+1)
    PC_CX_DEC = 0x9990  # STA $0CC6 (move_sights left,  cx-1)
    PC_CY_INC = 0x99B8  # STA $0CC7 (move_sights down,  cy+1)
    PC_CY_DEC = 0x99D2  # STA $0CC7 (move_sights up,    cy-1)
    PC_IRQ_SCAN = 0x9678  # IRQ: JSR check_for_full_player_input (gated full scan)
    PC_IRQ_SCAN_DONE = 0x967B  # return address of that JSR: the scan has completed

    def _pan_angle(self, addr, want, dir_fn):
        """Drive the view angle at `addr` to `want` with NO wall-clock control flow: HOLD
        the pan key and step frame-by-frame at PC_PAN_DONE ($365D, reached every frame),
        reading the SETTLED angle each frame until it equals `want`. Terminates only on a
        real state -- reached want ("ok"); player_object ($0B) changed, i.e. teleported
        mid-aim ("hyperspace"); or the angle stops advancing for a full notch-scroll while
        not at want, i.e. clamped/off-lattice ("unreachable", an aim-proposer bug)."""
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
                    self.bm.run_until_pc(self.PC_PAN_DONE, timeout=_RU_COMMIT)
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

    def _run_to_scan(self, timeout=6.0):
        """Run to the next GATED full input scan ($9678). The IRQ reaches it only with
        the world NOT being plotted ($0CE4 bit7, set by set_busy_plotting $1214 and held
        across a whole viewpoint redraw $3642->play_landscape_loop) and the $130B re-arm
        expired ($966E), so a timeout while $0CE4 is set is a redraw still running, not a
        stall: re-arm. Conceding there would leak the redraw's remaining frames into
        whatever primitive runs next. A timeout with the gate OPEN is a real stall."""
        for _ in range(_SCAN_WAIT_PASSES):
            try:
                self.bm.run_until_pc(self.PC_IRQ_SCAN, timeout=timeout)
                return True
            except Exception:
                if not self.rd(A_PLOT) & 0x80:
                    raise
        return False

    def _one_scan_press(self, key, timeout=10.0):
        """Hold `key` for EXACTLY ONE gated full input scan ($9678->$967B), after ONE
        IDLE scan with the key released. check_for_full_player_input latches an EDGE:
        SPACE toggles only when the previous scan saw it up ($11B5 LDA $1236 / BNE
        skip_sights_toggle; $11D4 clears $1236 only on a scan with SPACE not pressed).
        Without the idle re-arm, two presses with no released-key scan between them --
        sights OFF then ON across coarse pans that are both no-ops, so nothing runs
        frames in between -- have the second swallowed and retried by `sights_set`."""
        r, c = _k(key)
        with self.bm.halted():
            try:
                self._run_to_scan(timeout)
                self.bm.advance_instructions(1)  # off the anchor, into the scan
                self._run_to_scan(timeout)  # that idle scan re-armed the latch
                self.bm.keymatrix_set([(r, c, 1)])  # press WHILE HALTED
                self.bm.run_until_pc(self.PC_IRQ_SCAN_DONE, timeout=timeout)
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
            self._one_scan_press("SPACE")
        return bool(self.rd(A_SFLAG) & 0x80) == on

    def sights_on(self):
        return self.sights_set(True)

    def _uturn(self, max_passes=5):
        """SIGHTS OFF: flip the bearing 180 degrees in ONE keystroke (handle_uturn $1B2F,
        objects_h_angle EOR $80) -- the fast way across half the compass.

        U is an ordinary action key (want-flag $23), so ``tap_action`` already presses it
        correctly. The flip itself lands in update_game's $1B2F, NOT in the scan that
        latched the press, so reading objects_h_angle straight after that scan sees the
        OLD bearing; a retry loop keyed on that read presses again, and an even number of
        EOR $80s cancels. Confirm only after tap_action has let the action be consumed.
        """
        addr = A_H + self.slot()
        for _ in range(max_passes):
            before = self.rd(addr)
            if not self.tap_action(K_UTURN):
                continue
            if ((self.rd(addr) - before) & 0xFF) == 0x80:
                return True
        return False

    def coarse_h(self, want):
        """SIGHTS OFF: rotate bearing h to `want` (±8 lattice, wraps mod 256). D pans right
        +8, S pans left -8; the shorter direction is chosen per attempt (self-correcting).

        When `want` is more than half a turn away, ONE U-turn (EOR $80) + a short +-8
        correction is FEWER keystrokes -- and far less pan-scroll time, so less time held
        exposed while the Sentinel rotates -- than stepping up to 16 times round the circle.
        aimcost.h_press_count picks the minimal plan; the +-8 loop below finishes the
        residual whether or not the U-turn latched (so this only ever saves time)."""
        addr = A_H + self.slot()
        want &= 0xFF
        cur = self.rd(addr)
        n_uturn, n_step = ac.h_press_count(cur, want)
        if n_uturn and not self._uturn():
            # Never silent: the residual pan is then the whole half-turn the u-turn was meant to skip, which the cost model charged one keystroke for.
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

    def fine_cursor(self, cx, cy):
        """SIGHTS ON: drive the sights cursor to (cx, cy) DIAGONALLY -- move_sights ($9958)
        steps cx ($9965) and cy ($9994) in ONE call, so a held horizontal+vertical pair moves
        both 1px/frame and travel is max(|dx|,|dy|), not the axis-by-axis sum.  The whole
        drive runs in ONE halted section: the only CPU time is each pixel's run-to-commit
        (the ROM-intrinsic frame per move) -- no auto-resume gaps between pixels, which
        leaked free-running world frames on every interstitial read."""
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

    def tap_action(self, name, max_passes=45, settle=True):
        """Fire an action key EXACTLY ONCE. One full IDLE scan first: update_game
        zeroes $0C51 ($1281) and only an idle full scan re-arms $40 ($11EA); without it
        a u-turn latch is DROPPED at $1B2F (ASL $0C51 / BPL). Anchor at the gated full-scan
        call site $9678 (NOT $1363, which has three callers), press WHILE HALTED, run to
        $967B, snapshot the want-flags, release before the next scan. The caller MUST verify
        the action's memory effect and retry on a miss -- at-most-once by the single-scan
        press, at-least-once by the verify-retry loop."""
        want = ACTION_CODE[name]
        r, c = _k(name)
        latched = False
        with self.bm.halted():
            try:
                for _ in range(max_passes):
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
                            break  # caller settles by its own condition (e.g. a
                            # hyperspace: the play-loop scan PC may never recur)
                        # Scans reopen only after $12D0 consumed the action, so this run spans the whole settle ($1FA4 dither + $2625 plot_world); _run_to_scan's $0CE4 re-arm keeps a transfer's redraw from being cut short and leaking into the next aim. Legacy free-run caps it short (there the settle runs at gameplay speed during bookkeeping).
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
        okh = self.coarse_h(view["h_angle"])
        okv = self.coarse_v(view["v_angle"])
        if "hyperspace" in (okh, okv):  # teleported mid-aim: abort, do not re-drive
            return {
                "h": self.hang(),
                "v": self.vang(),
                "cur": self.cur(),
                "ok": False,
                "status": "hyperspace",
            }
        if okh != "ok" or okv != "ok":
            okh = self.coarse_h(view["h_angle"])
            okv = self.coarse_v(view["v_angle"])
        status = (
            "hyperspace"
            if "hyperspace" in (okh, okv)
            else ("unreachable" if "unreachable" in (okh, okv) else "ok")
        )
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
