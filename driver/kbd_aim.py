#!/usr/bin/env python3
"""Precise KEYBOARD aim for the live VICE Sentinel replay, built on the EXPERIMENTALLY
VERIFIED pan primitive (scripts/_pan_probe.py):

  * D (sight right): cursor_x cycles 80,89,98,...,143 then WRAPS to 80 and h_angle += 8.
  * S (sight left) : cursor_x cycles down 80,71,...,17 then WRAPS to 79 and h_angle -= 8.
  * L (sight down) : cursor_y cycles 95,104,...,158 then WRAPS to 96 and v_angle += 4.
  * COMMA (up)     : cursor_y cycles 95,86,...,32  then WRAPS to 95 and v_angle -= 4.
  * U (u-turn)     : h_angle EOR $80.

So h_angle is a +-8 keyboard grid (reachable = h0 + 8k), v_angle a +-4 grid, and the
cursor moves on a 9px grid (cx in {17..143}, cy in {32..158}). The aim that the action
fires on is computed by the ROM's prepare_vector_from_player_sights ($1C10):
h_eff = h + cur_x>>3, v_eff = v + (cur_y-5)>>4 -- which sentinel.los models exactly.

We (1) search the keyboard grid NATIVELY (sentinel.los.aim_target, bit-exact vs
the ROM) for a (h, v, cursor) that lands the ray on the target tile with LOS (and a
small tile-centre fraction when needed), then (2) drive the real keys to that exact
(h, v, cursor) -- verified from memory reads -- and (3) the caller probes the live ROM
LOS to confirm before pressing the action key. No pixels, no angle pokes.
"""

import os

from sentinel.state import State
from sentinel import los
from sentinel import aimcost as ac

# run_until_pc socket-guard timeouts. Under live AVI recording (warp OFF) the ZMBV
# encoder can back-pressure the binmon socket for several seconds, so the CPU takes
# longer than a few frames to next reach a checkpoint from the monitor's view. The
# old 4 s pan guard tripped on that backpressure and aborted the aim mid-pan (an
# aim_miss that then burned energy in re-plan churn). These are pure hang guards --
# the checkpoints recur every frame -- so a generous value only costs time on a truly
# dead socket, never on the happy path. Overridable via env for headless/warp runs.
_RU_PAN = float(os.environ.get("KBD_PAN_TIMEOUT", "20"))
_RU_STA = float(os.environ.get("KBD_STA_TIMEOUT", "8"))

A_SLOT = 0x000B
A_H = 0x09C0
A_V = 0x0140
A_CX = 0x0CC6
A_CY = 0x0CC7
A_SFLAG = 0x0C5F
A_PLOT = 0x0CE4
A_ZH = 0x0940

# verified cursor cycles (one key press each)
CX_GRID = [80, 89, 98, 107, 116, 125, 134, 143]  # D cycles up; S cycles down
CY_GRID_UP = [95, 86, 77, 68, 59, 50, 41, 32]  # COMMA
CY_GRID_DN = [95, 104, 113, 122, 131, 140, 149, 158]  # L
CX_CENTRE, CY_CENTRE = 80, 95

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


def _cursor_x_choices():
    # a scan-consumed press moves the cursor +-1px, so ANY pixel clear of the wrap
    # bands is reachable; step 3 keeps the search cheap.
    return list(range(20, 141, 3))


def _cursor_y_choices():
    return list(range(36, 153, 3))


def snap_keyboard_view(mem4k, tile, want_centre):
    """Search the keyboard grid (h0+8k, v0+4j, cursor on its 9px cycle) for a view
    whose NATIVE ray hits `tile` with LOS (and centre fraction < $40 if want_centre).
    Returns (view, info) where view = {h_angle, v_angle, cursor:[cx,cy]} or (None,..).
    """
    ps = mem4k[A_SLOT]
    eye_z = mem4k[A_ZH + ps]
    st = State.from_mem(bytes(mem4k))
    h0 = mem4k[A_H + ps]
    v0 = mem4k[A_V + ps]
    px, py = mem4k[0x0900 + ps], mem4k[0x0980 + ps]
    hbase, vbase = h0 % 8, v0 % 4
    # h grid: full circle on the 8-step lattice (u-turn keeps us on it), but ordered by
    # proximity to the analytic compass bearing to the target so a hit is found early.
    est = ac.bearing_to(px, py, tile[0], tile[1]) or 0
    full = [(hbase + 8 * k) & 0xFF for k in range(32)]
    hgrid = sorted(full, key=lambda h: ac.angle_dist(est, h))
    # v grid: real pan clamp is $CD..$35 ($1149), lattice v ≡ v0 (mod 4). The band is
    # [$CD..$FF]∪[$00..$35]; anything outside is physically unreachable by the keyboard.
    band = list(range(0xCD, 0x100)) + list(range(0x00, 0x36))
    vgrid = [v for v in band if v % 4 == vbase]
    cxs = _cursor_x_choices()
    cys = _cursor_y_choices()

    # v candidates ordered by proximity to current v0 (fewest pan steps first).
    vord = sorted(vgrid, key=lambda v: ac.angle_dist(v0, v))

    def search(cx_list, cy_list, accept_thresh):
        """Return the first view found with centre <= accept_thresh (good enough; the
        live probe confirms), else the best seen. Bearing-ordered h + v-near-current
        ordering makes the first hit a low-pan, near-centre view."""
        best = None
        for h in hgrid:
            for v in vord:
                for cx in cx_list:
                    for cy in cy_list:
                        rx, ry, los_hit, centre = los.aim_target(
                            st,
                            h,
                            v,
                            cx,
                            cy,
                            ps,
                            eye_z=eye_z,
                            max_steps=640,
                            return_centre=True,
                        )
                        if (rx, ry) != tile or not los_hit:
                            continue
                        if want_centre and centre >= 0x40:
                            continue
                        cur_pen = (cx != CX_CENTRE) + (cy != CY_CENTRE)
                        key = (cur_pen, centre, ac.h_steps(h0, h) + ac.v_steps(v0, v))
                        view = {"h_angle": h, "v_angle": v, "cursor": [cx, cy]}
                        if best is None or key < best[0]:
                            best = (key, view)
                        if centre <= accept_thresh:
                            return best  # good enough; stop early
        return best

    # phase 1: centre cursor only (the common, robust case).
    best = search([CX_CENTRE], [CY_CENTRE], 0x20)
    if best is not None:
        return best[1], {"centre": best[0][1], "cursor_used": False}
    # phase 2: open the cursor grid.
    best = search(cxs, cys, 0x20)
    if best is not None:
        return best[1], {"centre": best[0][1], "cursor_used": True}
    return None, {"reason": "no keyboard view hits tile with LOS"}


class KbdDriver:
    """Drive the real keys to an exact (h_angle, v_angle, cursor) using the verified
    pan cycles. All gating/feedback is from memory reads."""

    def __init__(self, bm, log):
        self.bm = bm
        self.log = log

    def rd(self, a):
        return self.bm.mem_get(a, a)[0]

    def slot(self):
        return self.rd(A_SLOT)

    def hang(self):
        return self.rd(A_H + self.slot())

    def vang(self):
        return self.rd(A_V + self.slot())

    def cur(self):
        return self.rd(A_CX), self.rd(A_CY)

    # ---- checkpoint-driven primitives (no wall-clock timing) ----
    # PCs from the game's ROM routines.
    # Angle COMMIT PCs: a pan's settled STA ($10EB/$110B/$1132) can be UNDONE later in the
    # same frame ($10E1/$1126 BCS -> undo_*_pan when plot_world returns carry set). So we
    # checkpoint where the pan is COMMITTED (only reached on the non-undo branch) and read
    # the settled value straight from MEMORY there -- not the A register at the STA.
    PC_H_COMMIT = (
        0x10EE  # is_panning_left: both left+right committed horizontal pans reach here
    )
    PC_V_COMMIT = (
        0x1135  # committed vertical pans (up branch + down-correction) reach here
    )
    # The one PC reached ONCE PER PAN ATTEMPT on EVERY outcome -- commit, undo, AND clamp --
    # and for BOTH axes: the instruction right after `JSR pan_viewpoint` in the foreground
    # loop ($365A). The commit PCs above are reached only when plot_world returns carry-clear;
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

    def _cursor_step(self, key, sta_pc):
        """One checkpoint-confirmed cursor move (exactly 1px). SIGHTS-ON cursor movement is
        auto-repeat gated ($11F6 ASL $0CC8, init $6B): a HELD key moves in an accelerating
        burst, so we press (WHILE HALTED, so the checkpoint is armed before any resume), run
        to the move_sights write ($997C/$9990/$99B8/$99D2), execute it (1px committed), then
        RELEASE -- one clean pixel per call, no burst overshoot."""
        r, c = _k(key)
        with self.bm.halted():
            try:
                self.bm.keymatrix_set([(r, c, 1)])  # press WHILE HALTED
                # A live pan/cursor step reaches its write PC within a frame or two; only a
                # CLAMPED move (cursor/angle at the band edge, PC never reached) runs the
                # full timeout -- and run_until_pc runs the CPU to get there, so at true
                # gameplay speed that is dead seconds of dwell in which the Sentinel spawns
                # a ring meanie. Cap short: detect the clamp fast, keep the dwell tiny.
                self.bm.run_until_pc(sta_pc, timeout=1.5)
                self.bm.advance_instructions(1)  # execute the STA -> 1px stored
            except Exception:
                pass
            finally:
                self.bm.keymatrix_release_all()
        self.bm.exit()

    def _pan_angle(
        self, addr, want, dir_fn, stall_bail=24, max_attempts=300, residue=8
    ):
        """Drive the view angle at `addr` to `want`. Fast path: HOLD the direction key
        and resume ONCE with a condition-gated checkpoint at PC_PAN_DONE ($365D) that
        stops the CPU exactly when the angle register reads `want` -- the scroll runs
        through every intermediate notch at full speed, no per-notch read-back, no
        self-correction, no stall_bail. `want` is only reachable on the +-`residue`
        lattice from `cur`, so a residue mismatch (or any monitor error) falls back to
        the stepwise loop below."""
        want &= 0xFF
        cur = self.rd(addr)
        if cur == want:
            return True
        if (want - cur) % residue == 0:
            key = dir_fn(cur)
            r, c = _k(key)
            with self.bm.halted():
                try:
                    self.bm.keymatrix_release_all()
                    self.bm.keymatrix_set([(r, c, 1)])
                    self.bm.run_until_pc(
                        self.PC_PAN_DONE,
                        timeout=_RU_PAN,
                        # VICE condition grammar (mon_parse.y): memory read is
                        # @BANKNAME:(addr); numbers are $-hex. Stop the scroll the
                        # instant the angle register reads `want`.
                        condition=f"@cpu:(${addr:04x}) == ${want:02x}",
                    )
                except Exception as e:  # off-lattice, clamp, or monitor rejects cond
                    self.log(f"    pan cond fallback: {type(e).__name__}")
                finally:
                    self.bm.keymatrix_release_all()
            self.bm.exit()
            if self.rd(addr) == want:
                return True
        return self._pan_angle_stepwise(addr, want, dir_fn, stall_bail, max_attempts)

    def _pan_angle_stepwise(self, addr, want, dir_fn, stall_bail=24, max_attempts=300):
        """Drive the view angle at `addr` to `want`, NO wall-clock wait. HOLD the pan key
        and step ONE ATTEMPT at a time by halting at PC_PAN_DONE ($365D) -- reached once per
        attempt on commit AND undo AND clamp, both axes -- then read the SETTLED angle from
        memory. `dir_fn(cur)` picks the key for the remaining delta and is re-evaluated each
        attempt (self-correcting across a wrap/overshoot). The key stays HELD across the plot
        so check_if_player_still_wants_to_pan ($1223) sees it and the plot completes.

        Stop conditions are STATE, never a timer: reached `want` (success), or `stall_bail`
        consecutive attempts that moved the angle by ZERO (genuinely clamped / can't reach --
        the caller re-snaps a different view). The run_until_pc timeout is a bare socket-hang
        guard, not a functional wait: $365D recurs every attempt, so it is hit promptly.
        """
        want &= 0xFF
        if self.rd(addr) == want:
            return True
        stalled = 0
        held = None
        with self.bm.halted():
            try:
                for _ in range(max_attempts):
                    cur = self.rd(addr)
                    if cur == want:
                        break
                    key = dir_fn(cur)
                    if key != held:  # (re)press for the direction the delta needs
                        self.bm.keymatrix_release_all()
                        r, c = _k(key)
                        self.bm.keymatrix_set([(r, c, 1)])
                        held = key
                    self.bm.run_until_pc(
                        self.PC_PAN_DONE, timeout=_RU_PAN
                    )  # one attempt
                    new = self.rd(addr)
                    self.bm.advance_instructions(1)  # step off $365D
                    if new == cur:  # undone or clamped this attempt
                        stalled += 1
                        if stalled >= stall_bail:
                            break
                    else:
                        stalled = 0
            except Exception as e:  # socket drop / genuinely wedged
                self.log(f"    pan ${self.PC_PAN_DONE:04x} stop: {type(e).__name__}")
            finally:
                self.bm.keymatrix_release_all()
        self.bm.exit()
        return self.rd(addr) == want

    def _one_scan_press(self, key, timeout=10.0):
        """Hold `key` for EXACTLY ONE gated full input scan ($9678->$967B). Anchor at the
        gated scan CALL site (it fires only when a scan will actually run), press WHILE
        HALTED, run to the return address, release -- the scan that consumed the key already
        latched, and the next scan sees it released."""
        r, c = _k(key)
        with self.bm.halted():
            try:
                self.bm.run_until_pc(self.PC_IRQ_SCAN, timeout=timeout)
                self.bm.keymatrix_set([(r, c, 1)])
                self.bm.run_until_pc(self.PC_IRQ_SCAN_DONE, timeout=timeout)
            finally:
                self.bm.keymatrix_release_all()
        self.bm.exit()

    def sights_set(self, on):
        """Toggle SPACE (edge-latched $1236) until the sights flag ($0C5F bit7) matches.
        One gated full scan == exactly one toggle."""
        for _ in range(6):
            if bool(self.rd(A_SFLAG) & 0x80) == on:
                return True
            self._one_scan_press("SPACE")
        return bool(self.rd(A_SFLAG) & 0x80) == on

    def sights_on(self):
        return self.sights_set(True)

    def _uturn(self, max_passes=5):
        """SIGHTS OFF: flip the bearing 180 degrees in ONE keystroke (handle_uturn $1B2F,
        objects_h_angle EOR $80) -- the fast way across half the compass. The u-turn is
        edge-latched and auto-repeat gated like an action key ($0C51 ASL/BPL): update_game
        zeroes the want-flag and only an IDLE full scan re-arms it, so a bare tap right after
        a pan burst is dropped ($1B2F). So per pass run one idle full scan to re-arm, press U
        WHILE HALTED through the next scan, release, and confirm the EOR $80 flip; retry
        until it takes. Returns True on a confirmed flip -- purely an optimisation, the +-8
        loop in coarse_h still converges if this is swallowed."""
        addr = A_H + self.slot()
        r, c = _k(K_UTURN)
        flipped = False
        with self.bm.halted():
            try:
                for _ in range(max_passes):
                    before = self.rd(addr)
                    self.bm.run_until_pc(self.PC_IRQ_SCAN, timeout=6.0)
                    self.bm.advance_instructions(1)  # off the anchor
                    self.bm.run_until_pc(
                        self.PC_IRQ_SCAN, timeout=6.0
                    )  # idle scan re-arms
                    self.bm.keymatrix_set([(r, c, 1)])  # press WHILE HALTED
                    self.bm.run_until_pc(
                        self.PC_IRQ_SCAN_DONE, timeout=6.0
                    )  # scan consumed
                    self.bm.keymatrix_release_all()  # before the next scan
                    if ((self.rd(addr) - before) & 0xFF) == 0x80:
                        flipped = True
                        break
            except Exception as e:
                self.log(f"    uturn stop: {type(e).__name__}")
            finally:
                self.bm.keymatrix_release_all()
        self.bm.exit()
        return flipped

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
        n_uturn, _n_step = ac.h_press_count(self.rd(addr), want)
        if n_uturn:
            self._uturn()
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
        dir_fn = lambda cur: (
            K_DOWN if self._pitch_lin(want) > self._pitch_lin(cur) else K_UP
        )
        return self._pan_angle(addr, want, dir_fn, residue=4)

    def fine_to_tile(self, target, probe_fn, want_centre=False, budget=24):
        """SIGHTS ON. Land the sights ray on `target` (with LOS, and centre fraction
        < $40 when want_centre) by an ABSOLUTE-position ring search around the snapped
        cursor: drive the cursor to each candidate pixel (fine_cursor, closed-loop) and
        probe. Every position is reached absolutely, so it never walks OFF a hit like the
        old greedy nudger; a cursor wrap aborts (caller re-aims)."""

        def good():
            r = probe_fn()
            rx, ry, los = r[0], r[1], r[2]
            centre = r[3] if len(r) > 3 else 0
            return (rx, ry) == target and los and (not want_centre or centre < 0x40)

        if good():
            return True
        cx0, cy0 = self.cur()
        offs = sorted(
            (
                (dx, dy)
                for dx in range(-12, 13, 3)
                for dy in range(-12, 13, 3)
                if (dx, dy) != (0, 0)
            ),
            key=lambda d: (abs(d[0]) + abs(d[1]), d),
        )
        tried = 0
        for dx, dy in offs:
            cx, cy = cx0 + dx, cy0 + dy
            if not (20 <= cx <= 140 and 36 <= cy <= 152):  # clear of wrap pixels
                continue
            if not self.fine_cursor(cx, cy):
                return False  # a wrap/pan happened; caller re-aims
            if good():
                return True
            tried += 1
            if tried >= budget:
                break
        return good()

    def aim_at_tile(self, target, want_centre, probe_fn):
        """Full keyboard aim onto `target`: COARSE rotate (sights off) to the snapped
        view's angles, then FINE cursor (sights on) closed-loop on the tile. Every
        sub-step result is checked; sights-on re-centres the cursor ($134C) so the snapped
        cursor must be driven explicitly."""
        m4k = bytearray(self.bm.mem_get(0x0000, 0x0FFF))
        view, _info = snap_keyboard_view(m4k, target, want_centre)
        if view is None:
            return None
        if not self.sights_set(False):
            self.log(f"    ABORT {target}: sights would not turn OFF")
            return None
        okh = self.coarse_h(view["h_angle"])
        okv = self.coarse_v(view["v_angle"])
        if not (okh and okv):
            okh = self.coarse_h(view["h_angle"])
            okv = self.coarse_v(view["v_angle"])
        self.log(
            f"    coarse {target}: h=${self.hang():02x}/want ${view['h_angle']:02x} "
            f"ok={okh}  v=${self.vang():02x}/want ${view['v_angle']:02x} ok={okv}"
        )
        if not (okh and okv):
            return None
        if not self.sights_set(True):
            return None
        self.fine_cursor(*view["cursor"])  # sights-on just re-centred it
        self.fine_to_tile(target, probe_fn, want_centre)
        return probe_fn()

    def _cursor_axis(self, addr, want, key_inc, key_dec, inc_pc, dec_pc, max_moves=160):
        """SIGHTS ON: drive one cursor axis to the exact pixel `want` one checkpoint-confirmed
        pixel at a time, re-reading the coordinate and re-choosing direction each step (so a
        stray move self-corrects and can never run away to the wrap band). Targets are
        interior, so no wrap/pan is triggered."""
        want &= 0xFF
        for _ in range(max_moves):
            cur = self.rd(addr)
            if cur == want:
                return True
            if want > cur:
                self._cursor_step(key_inc, inc_pc)
            else:
                self._cursor_step(key_dec, dec_pc)
        return self.rd(addr) == want

    def _cursor_step_diag(self, keys, commit_pc):
        """One move_sights call with `keys` (a horizontal and/or vertical direction) held
        together: $9958 runs $9965 (cx) then $9994 (cy), so a diagonal press moves BOTH 1px.
        `commit_pc` is the STA that executes LAST (the cy write when a vertical key is held,
        else the cx write), so both writes are committed when it is reached. Press WHILE
        HALTED, run to the write, execute it, RELEASE -- one clean diagonal pixel, no burst.
        """
        presses = [(*_k(key), 1) for key in keys]
        with self.bm.halted():
            try:
                self.bm.keymatrix_set(presses)
                self.bm.run_until_pc(commit_pc, timeout=1.5)
                self.bm.advance_instructions(1)  # execute the last STA
            except Exception:
                pass
            finally:
                self.bm.keymatrix_release_all()
        self.bm.exit()

    def fine_cursor(self, cx, cy):
        """SIGHTS ON: drive the sights cursor to (cx, cy) DIAGONALLY -- move_sights ($9958)
        steps cx ($9965) and cy ($9994) in ONE call, so a held horizontal+vertical pair moves
        both 1px/frame and travel is max(|dx|,|dy|), not the axis-by-axis sum. Re-reads each
        step so a stray move self-corrects; interior targets never wrap/pan. D/S move cx +/-
        ($997C/$9990), L/COMMA move cy +/- ($99B8/$99D2)."""
        cx &= 0xFF
        cy &= 0xFF
        for _ in range(160):
            curx, cury = self.rd(A_CX), self.rd(A_CY)
            if curx == cx and cury == cy:
                return True
            keys, commit_pc = [], None
            if curx != cx:
                keys.append(K_RIGHT if cx > curx else K_LEFT)
                commit_pc = self.PC_CX_INC if cx > curx else self.PC_CX_DEC
            if cury != cy:  # cy write runs last in move_sights -> commit point
                keys.append(K_DOWN if cy > cury else K_UP)
                commit_pc = self.PC_CY_INC if cy > cury else self.PC_CY_DEC
            self._cursor_step_diag(keys, commit_pc)
        return self.rd(A_CX) == cx and self.rd(A_CY) == cy

    def tap_action(self, name, max_passes=45):
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
                    self.bm.run_until_pc(self.PC_IRQ_SCAN, timeout=6.0)
                    self.bm.advance_instructions(1)  # off the anchor
                    self.bm.run_until_pc(self.PC_IRQ_SCAN, timeout=6.0)  # idle scan ran
                    self.bm.keymatrix_set([(r, c, 1)])  # press WHILE HALTED
                    self.bm.run_until_pc(self.PC_IRQ_SCAN_DONE, timeout=6.0)
                    flags = self.bm.mem_get(0x0CE8, 0x0CEB)
                    self.bm.keymatrix_release_all()  # before the next scan
                    if want in flags:
                        latched = True
                        # scans reopen only after $12D0 consumed the action. This is a
                        # CONFIRMATION wait (latch already observed); a transfer rebuilds
                        # the view from the new POV and the gated scan PC may not recur for
                        # seconds -- at true gameplay speed (warp off during recording)
                        # that is seconds of live dwell in which the Sentinel spawns a
                        # meanie. Cap it short: the caller resyncs+halts before the next
                        # action, so consumption always completes before it matters.
                        try:
                            self.bm.run_until_pc(self.PC_IRQ_SCAN, timeout=1.0)
                        except Exception:
                            pass
                        break
            except Exception as e:
                self.log(f"    tap_action {name} stop: {type(e).__name__}")
            finally:
                self.bm.keymatrix_release_all()
        self.bm.exit()
        return latched

    def drive_to(self, view):
        """Drive to the snapped (h, v, cursor). COARSE with sights OFF, then FINE with
        sights ON (cursor). Every sub-step result is checked (mirrors aim_at_tile);
        returns the achieved dict."""
        if not self.sights_set(False):  # coarse: fast rotate
            return {"h": self.hang(), "v": self.vang(), "cur": self.cur(), "ok": False}
        okh = self.coarse_h(view["h_angle"])
        okv = self.coarse_v(view["v_angle"])
        if not (okh and okv):
            okh = self.coarse_h(view["h_angle"])
            okv = self.coarse_v(view["v_angle"])
        oks = self.sights_set(True)  # fine: cursor selection
        cx, cy = view["cursor"]
        okc = self.fine_cursor(cx, cy)
        return {
            "h": self.hang(),
            "v": self.vang(),
            "cur": self.cur(),
            "ok": bool(okh and okv and oks and okc),
        }


def _k(name):
    from vice_driver import keys

    return keys.lookup(name)
