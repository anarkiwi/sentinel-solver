#!/usr/bin/env python3
"""Live keyboard aiming + action primitives for The Sentinel (C64) in asid-vice.

THE ORACLE (verified live, see sentinel_execute.py):
  When ANY action key is pressed, handle_player_actions ($1B43) runs
  prepare_vector_from_player_sights ($1C10) + check_for_line_of_sight_to_tile
  ($1CDD) and leaves the TILE the sights ray reached in $0024/$0026. On an
  occupied / unplaceable / LOS-blocked tile the action no-ops but the target is
  STILL written -- so an action key is a NON-COMMITTING probe of the real aim
  path. This is the ground truth for "what the sights are pointing at".

  Crucially, a COLD probe stub that just calls $1C10+$1CDD (sentinel_execute.Executor.
  probe) and the py65 code_engine running the same two routines BOTH DIVERGE from
  the live target -- they lack per-frame view state the live routines consult.
  Measured: live center cursor -> (12,4); cold stub -> (13,32); py65 -> (15,17).
  So we DO NOT use those; we press a real action key and read $24/$26.

CURSOR CONTROL (sights ON, $0C5F bit7; verified live):
  S = cursor left  ($0CC6 -1),  D = cursor right ($0CC6 +1)
  COMMA = cursor up ($0CC7 -1), L = cursor down  ($0CC7 +1)
  cursor X range ~0x10..0x90 (centre 0x50), Y ~0x20..0xA0 (centre 0x5F);
  past the edge the view PANS (move_sights $9958 wraps by 0x40 + sets $0009).
  We deliberately keep the cursor OFF the edges -- the pan is multi-frame and
  rapid taps across it desync input. To reach tiles outside the current cone we
  HOP (create robot + transfer) which re-centres the cone on the new position.

ACTIONS (sights ON for create/absorb/transfer; verified live they COMMIT):
  A=absorb, Q=transfer, R=create robot, T=create tree, B=create boulder,
  H=hyperspace (no sights needed), U=u-turn.

ENERGY: running player_energy ($0C0A) to 0 triggers a game reset. Callers must
  keep energy > 0; the win sequence re-absorbs its climb structure to recover.
"""

import time

# addresses
A_PLAYER_OBJECT = 0x000B
A_OBJECTS_X = 0x0900
A_OBJECTS_Z = 0x0940
A_OBJECTS_Y = 0x0980
A_OBJECTS_ZF = 0x0A00
A_OBJECTS_TYPE = 0x0A40
A_OBJECTS_FLAGS = 0x0100
A_SIGHTS_X = 0x0CC6
A_SIGHTS_Y = 0x0CC7
A_SIGHTS_FLAG = 0x0C5F
A_PLAYER_ENERGY = 0x0C0A
A_TGT_X = 0x0024
A_TGT_Y = 0x0026
A_LANDSCAPE_DONE = 0x0CDE
A_PLATFORM_X = 0x0C19
A_PLATFORM_Y = 0x0C1A

# cursor safe bounds (stay off the edge so we never trigger a view pan mid-aim)
CUR_X_LO, CUR_X_HI = 0x1C, 0x84
CUR_Y_LO, CUR_Y_HI = 0x2C, 0x96

from vice_driver import keys
from vice_driver.binmon import TAP_MODE_FIXED


class LiveAim:
    """Keyboard-driven aiming + actions against a live BinMon `bm`."""

    def __init__(self, bm, log=print):
        self.bm = bm
        self.log = log
        self.ps = self.rd(A_PLAYER_OBJECT)

    # ---- low level ----
    def rd(self, a):
        return self.bm.mem_get(a, a)[0]

    def wr(self, a, v):
        self.bm.mem_set(a, bytes([v & 0xFF]))

    def tap(self, name, frames=4, settle=0.02):
        # Binmon calls are ~1ms; the only real cost is letting the emulator advance
        # enough frames for the input loop to sample the key. Measured live:
        # settle=0.02s gives 100% key registration (cursor + action), settle=0.0 is
        # ~90%. So 0.02 is the fast, reliable default. (Earlier 0.07-0.18 values were
        # needlessly conservative and made sweeps take minutes.)
        self.bm.keymatrix_tap([keys.lookup(name)], mode=TAP_MODE_FIXED, frames=frames)
        time.sleep(settle)

    def cursor(self):
        return (self.rd(A_SIGHTS_X), self.rd(A_SIGHTS_Y))

    def player_tile(self):
        ps = self.rd(A_PLAYER_OBJECT)
        return (
            self.rd(A_OBJECTS_X + ps),
            self.rd(A_OBJECTS_Y + ps),
            self.rd(A_OBJECTS_Z + ps),
        )

    def sights_on(self):
        if not (self.rd(A_SIGHTS_FLAG) & 0x80):
            self.tap("SPACE", frames=18, settle=0.5)
        return self.rd(A_SIGHTS_FLAG) & 0x80

    # ---- the oracle: press an action key, read $24/$26 (non-committing) ----
    def probe_target(self, key="Q"):
        """Read the live target tile via an action-key press. $0024/$0026 is written
        by check_for_line_of_sight_to_tile DURING the ray march, for ANY action key,
        BEFORE the action's commit branch -- so it is the true aim signal.

        We default to Q (transfer): try_to_transfer_into_object ($1B64) only commits
        on a tile holding a ROBOT (BNE play_bad_action_sound otherwise), so a Q-probe
        is non-committing on every tile EXCEPT a robot tile. That is far safer for a
        long sweep than A (which would absorb any tree/object it crosses) or a create
        key (which would build on any empty visible tile). The caller avoids probing
        directly onto its own climb robot."""
        self.wr(A_TGT_X, 0xEE)
        self.wr(A_TGT_Y, 0xEE)
        self.tap(key, frames=4, settle=0.02)
        return (self.rd(A_TGT_X), self.rd(A_TGT_Y))

    # ---- cursor corner reset (off the edge, never pans) ----
    def move(self, key, frames=10):
        """One ~5-6px cursor step (a longer fixed tap moves further per round-trip,
        so a full sweep needs far fewer binmon calls)."""
        self.tap(key, frames=frames, settle=0.025)

    def reset_cursor_corner(self):
        """Drive the cursor to the safe upper-left corner with keys (stopping
        before the pan edge). Hard-capped so it can never spin."""
        for _ in range(30):
            if self.rd(A_SIGHTS_X) <= CUR_X_LO + 4:
                break
            self.move("S")
        for _ in range(30):
            if self.rd(A_SIGHTS_Y) <= CUR_Y_LO + 4:
                break
            self.move("COMMA")

    def sweep_targets(self, callback=None, max_probes=900):
        """Boustrophedon sweep of the sights cursor across its whole safe range
        (staying off the pan edges). After each ~5px cursor step, read the live
        target via the action-key probe. If `callback(target,cursor)` returns True
        the sweep stops early. Returns the set of reachable tiles seen. Hard-capped
        at `max_probes` so it always terminates (~40 s for a full cone)."""
        self.reset_cursor_corner()
        seen = set()
        going_right = True
        nprobe = 0
        while self.rd(A_SIGHTS_Y) < CUR_Y_HI and nprobe < max_probes:
            row_steps = 0
            while row_steps < 40 and nprobe < max_probes:
                t = self.probe_target()
                nprobe += 1
                cx, cy = self.cursor()
                if t[0] < 32 and t[1] < 32:
                    seen.add((t[0], t[1]))
                    if callback is not None and callback(t, (cx, cy)):
                        return seen
                cx = self.rd(A_SIGHTS_X)
                if going_right:
                    if cx >= CUR_X_HI:
                        break
                    self.move("D")
                else:
                    if cx <= CUR_X_LO:
                        break
                    self.move("S")
                row_steps += 1
            going_right = not going_right
            self.move("L")
        return seen

    def reachable_tiles(self):
        """All tiles the sights can currently hit (from the current view)."""
        return self.sweep_targets()

    def aim_at(self, gx, gy):
        """Aim the sights exactly at tile (gx,gy) by sweeping the cursor until the
        LIVE probe target == (gx,gy), then STOP -- so the cursor is provably on the
        goal (the very probe that confirmed it IS the action path). Self-correcting:
        no reliance on reproducing a recorded pixel. Returns (ok, target, cursor).

        Two-pass: a coarse sweep then a fine (1-px x-step) sweep, since some tiles
        are only hit on particular cursor columns. We stop the instant the probe
        equals the goal."""
        hit = {"v": None}

        def cb(t, c):
            if t == (gx, gy):
                hit["v"] = c
                return True
            return False

        self.sweep_targets(callback=cb)
        if hit["v"] is not None:
            return True, (gx, gy), hit["v"]
        return False, self.probe_target(), self.cursor()
