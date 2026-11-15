#!/usr/bin/env python3
"""Player-agnostic machinery for driving a sim player against the REAL game in VICE.

``LiveMixin`` carries ONLY observation + execution over live memory; decision
logic (``_tick``) comes from the composed sim player via MRO.
It goes first in the bases so its execution overrides win over ``BasePlayer``'s.
"""

import os

import numpy as np

from driver import clock, core, kbd_aim, sentinel_execute as sx
from sentinel import phase_player, memmap as mm, player as sim_player, playerbase
from sentinel.game import Game
from sentinel.state import State

PC_PASS = 0x363D  # update_game_and_continue: a main-loop pass begins
PC_PAN_CALL = 0x365A  # JSR pan_viewpoint $10B7
RASTER_LINES = 312  # PAL 6569


class MeasuringKbdDriver(kbd_aim.KbdDriver):
    """KbdDriver that times each aim primitive into ``subframes`` (exact $9630 frames
    via the shared Executor) for per-sub-term charge validation. Reset per step;
    sights_on delegates to sights_set so the toggle is timed once, not twice."""

    def __init__(self, bm, log, quantized=False, ex=None):
        super().__init__(bm, log, quantized=quantized)
        self._ex = ex
        self.subframes = {}
        self.passes = []  # (step, frozen, scroll, update, pan) per pan pass
        self._label = None

    def _timed(self, name, fn, *args):
        f0 = self._ex.frames()
        r = fn(*args)
        df = self._ex.frames() - f0
        self.subframes[name] = self.subframes.get(name, 0) + df
        return r

    def _run_pass(self):
        """Run one pan pass split at its ROM boundaries, appending ``(scroll, update,
        pan)`` frames (raster-resolved) to ``passes``: $365D->$363D is the previous
        notch's scroll loop, $363D->$365A update_game $127C, $365A->$365D $10B7."""
        t = [self._now()]
        for pc in (PC_PASS, PC_PAN_CALL, self.PC_PAN_DONE):
            self.bm.run_until_pc(pc, timeout=kbd_aim._RU_COMMIT)
            t.append(self._now())
            if pc != self.PC_PAN_DONE:
                self.bm.advance_instructions(1)
        self.passes.append(
            (self._label, self._ex.rd(mm.PLAYER_NOT_ACTED) >> 7)
            + tuple(round(b - a, 3) for a, b in zip(t, t[1:]))
        )

    def _now(self):
        """Frames since record start at raster resolution: the $9630 count plus the
        raster line ($D011 bit7:$D012) over the PAL frame's 312 lines."""
        ctrl, line = self.bm.mem_get(0xD011, 0xD012, bank="io")
        return self._ex.frames() + (((ctrl & 0x80) << 1) | line) / RASTER_LINES

    def coarse_h(self, want):
        return self._timed("pan_h", super().coarse_h, want)

    def _uturn(self, *a, **kw):
        # nested inside coarse_h's pan_h, so the books must subtract it, not add it
        return self._timed("uturn", super()._uturn, *a, **kw)

    def coarse_v(self, want):
        return self._timed("pan_v", super().coarse_v, want)

    def fine_cursor(self, cx, cy):
        return self._timed("cursor", super().fine_cursor, cx, cy)

    def sights_set(self, on):
        return self._timed("toggle", super().sights_set, on)


def drive_transfer_aim(kbd, tile, view, log):
    """Aim the sights onto `tile` for a transfer; the live ray probe is the arbiter.

    A reused aim whose probe misses re-drives, and a still-missing probe refuses the
    fire (frozen ls42 p25: a blind reuse transferred 61->61, the sim reaches 58)."""
    want = (view["h_angle"], view["v_angle"])
    reused = kbd.sights_live_on() and kbd.committed_bearing() == want
    if reused:
        kbd.fine_cursor(*view["cursor"])
    else:
        ach = kbd.drive_to(view)
        if not ach["ok"]:
            kbd.clear_bearing()
            return False
        kbd.set_bearing(*want)
    rx, ry, los_hit, _ = core.probe_tile(kbd.bm)
    if (rx, ry) == tuple(tile) and los_hit:
        return True
    log(f"    transfer aim probe ({rx},{ry}) los={los_hit} != {tuple(tile)}")
    if reused:  # the parked sights drifted: full re-drive, then re-probe
        ach = kbd.drive_to(view)
        if ach["ok"]:
            kbd.set_bearing(*want)
            rx, ry, los_hit, _ = core.probe_tile(kbd.bm)
            if (rx, ry) == tuple(tile) and los_hit:
                return True
        log(f"    transfer re-aim probe ({rx},{ry}) los={los_hit}; not firing")
    kbd.clear_bearing()
    return False


def save_tick(ticks_dir, n, player):
    """Write the observed live image and the aim state the player decides from to
    ``ticks_dir/tNNNN.npz``: everything :func:`load_tick` needs to replay that tick's
    decision on the simulator."""
    os.makedirs(ticks_dir, exist_ok=True)
    bearing = player.last_bearing
    np.savez_compressed(
        os.path.join(ticks_dir, f"t{n:04d}.npz"),
        mem=np.frombuffer(bytes(player.st.mem), dtype=np.uint8),
        cursor=np.array(player.cursor, dtype=np.int64),
        bearing=np.array((-1, -1) if bearing is None else bearing, dtype=np.int64),
        trace_len=np.int64(len(player.trace)),
    )


def load_tick(path, cls=phase_player.PhasePlayer, **kwargs):
    """A simulator player ``cls`` standing exactly where :func:`save_tick` saw the live
    one: the same memory image, cursor and committed bearing."""
    with np.load(path) as z:
        mem, cursor, bearing = (np.asarray(z[k]) for k in ("mem", "cursor", "bearing"))
    player = cls(Game(State.from_mem(mem.tobytes())), **kwargs)
    player.cursor = cursor.tolist()
    player.last_bearing = None if bearing[0] < 0 else tuple(bearing.tolist())
    return player


class LiveMixin:
    """Observation + execution over live VICE memory; no decision logic.

    ``__init__`` builds the sim ``Game`` from a live snapshot and forwards any
    player-specific kwargs to the composed sim player's constructor via
    ``super().__init__``."""

    def __init__(self, session, log, result, ticks_dir=None, **kwargs):
        game = Game(State.from_mem(core.live_image(session.bm)))
        self.ticks_dir = ticks_dir
        self.tick_no = 0
        super().__init__(game, verbose=True, **kwargs)
        self.bm = session.bm
        self.bm.auto_resume = False  # world runs ONLY in deliberate run windows
        self.live_log = log
        self.result = result
        self.ex = sx.Executor(session.bm, log)
        self.kbd = MeasuringKbdDriver(session.bm, log, quantized=True, ex=self.ex)
        self.step_no = 0
        self.acted = False
        self.life_lost = None
        self.audits = True  # price and record every step's charged-vs-measured frames

    def _observe(self):
        """:meth:`_refresh` at a decision tick, saving the tick when ``ticks_dir``."""
        self._refresh()
        if self.ticks_dir:
            save_tick(self.ticks_dir, self.tick_no, self)
            self.tick_no += 1

    def _refresh(self):
        """Snapshot live memory and LEAVE THE CPU HALTED: think time is a tooling
        artifact, so the world advances only under real input (pans/actions) and
        deliberate waits.  Death is the drain flag or a silent landscape auto-reset
        ($0CE5 re-frozen after we acted), NOT $0CDE bit7."""
        with self.bm.halted():
            mem = core.live_image(self.bm)
        self.st = State.from_mem(mem)
        self.g.state = self.st
        self._sync_aim_state()
        if self.acted and self.st.mem[mm.PLAYER_NOT_ACTED] & 0x80:
            self.life_lost = "died (landscape auto-reset observed)"

    def _sync_aim_state(self):
        """Adopt the DRIVER's aim state as the model's.  What decides an aim REUSE at
        execution time is ``sights_live_on() and committed_bearing() == view bearing``
        (``sentinel_execute.perform_step``, ``_drive_transfer_aim``); the live ``_fire``
        overrides ``BasePlayer._fire``, so the base's own ``last_bearing``/``cursor``
        bookkeeping never runs here and would leave every live step charged a full aim.
        Sights off == no committed bearing: the OFF->ON toggle re-centres the cursor
        ($134C), which is exactly the non-reuse branch of ``_aim_frames``."""
        mem = self.st.mem
        on = bool(mem[kbd_aim.A_SFLAG] & 0x80)
        self.last_bearing = self.kbd.committed_bearing() if on else None
        self.cursor = [int(mem[kbd_aim.A_CX]), int(mem[kbd_aim.A_CY])]

    def _lost(self):
        return bool(self.life_lost) or super()._lost()

    def _dead(self):
        if self.st.mem[mm.PLAYER_DIED_BY_DRAINING] & 0x80:
            self.life_lost = "drained at zero energy"
        if self.life_lost:
            self.live_log(f"DEAD: {self.life_lost}")
            self.result["death"] = self.life_lost
            return True
        return False

    def _advance(self, frames):
        """Real time passes in the live game; the model clock is a no-op.  This keeps
        the player's think time out of the live world -- the real world moves only
        when ``_fire``/``_wait`` drive the keyboard."""

    def _wait(self, frames=playerbase.WAIT_FRAMES):
        """A deliberate wait spends REAL world time, FRAME-EXACT: step the $9630
        per-frame marker ``frames`` times, leaving the CPU halted, and re-read the
        world.  charged == measured holds by construction whatever the warp state, so
        ``wait_audit`` is a regression pin; a stalled marker raises instead of being
        waited out."""
        got = clock.run_frames(self.bm, frames)
        if self.audits:
            self.result.setdefault("wait_audit", []).append([self.step_no, frames, got])
            self.live_log(
                f"    [clock] wait: charged={frames} measured={got} (exact frames)"
            )
        self._refresh()

    def _drive_transfer_aim(self, tile, view):
        return drive_transfer_aim(self.kbd, tile, view, self.live_log)

    def _fire(self, verb, tile, view):
        st = self.st
        if verb in ("boulder", "robot"):
            pverb = "create"
            otype = mm.T_BOULDER if verb == "boulder" else mm.T_ROBOT
        else:
            pverb = verb
            top = self._top(tile)
            if top is None:
                return False
            otype = st.obj_type[top]
        whole_f0 = self.ex.frames()  # exact whole-action bracket (incl. transfer aim)
        self.kbd.subframes = {}  # per-step aim sub-term (pan/cursor/toggle) accumulator
        self.kbd._label = f"p{self.step_no + 1}"
        if pverb == "transfer" and not self._drive_transfer_aim(tile, view):
            self._observe()
            return False
        self.step_no += 1
        stp = {
            "verb": pverb,
            "otype": otype,
            "target": list(tile),
            "view": {**view, "cursor": list(view["cursor"])},
        }
        if self.acted and self.ex.rd(mm.PLAYER_NOT_ACTED) & 0x80:
            self.life_lost = "died (landscape reset caught pre-fire)"
            return False  # never act on a silently-reset board (race with _observe)
        if pverb == "create":
            stp["min_energy"] = mm.ENERGY_IN_OBJECTS[otype] + self._reserve()
        out = sx.perform_step(
            self.ex, self.kbd, f"p{self.step_no}", stp, self.live_log, self.result
        )
        if self.audits:
            self._audit_step(pverb, tile, view, whole_f0)
        self.acted = True
        self._observe()
        ok = out in ("ok", "diverge")  # diverge: primary effect landed, world moved
        if ok:
            self._log(verb, tile)
        return ok

    def _audit_step(self, pverb, tile, view, whole_f0):
        """Record the step just performed: its charged aim and settle (the model's
        prices) against the exact frames the machine spent."""
        aim_charged = self._step_aim_frames(pverb, view)
        # View-aware transfer settle: scene-dependent projector replot, not flat 47.
        settle_charged = self._settle(pverb, view, self._settle_eye(pverb, tile))
        charged = aim_charged + settle_charged
        whole_exact = self.ex.frames() - whole_f0  # exact, wrap-free aim+settle
        sa = self.result.get("settle_audit") or []
        lbl = f"p{self.step_no}"
        if sa and sa[-1][0] == lbl:  # perform_step reached the fire this step
            settle_exact = sa[-1][2]
            self.result.setdefault("exact_audit", []).append(
                [
                    lbl,
                    pverb,
                    list(tile),
                    round(aim_charged),
                    whole_exact - settle_exact,
                    settle_charged,
                    settle_exact,
                ]
            )
            sf = self.kbd.subframes
            self.result.setdefault("aim_subframes", []).append(
                [
                    lbl,
                    pverb,
                    sf.get("pan_h", 0),
                    sf.get("pan_v", 0),
                    sf.get("toggle", 0),
                    sf.get("cursor", 0),
                    sf.get("uturn", 0),
                ]
            )
        self.result.setdefault("frame_audit", []).append(
            [lbl, pverb, list(tile), round(charged), whole_exact]
        )
        self.live_log(
            f"    [clock] p{self.step_no} {pverb}: charged={round(charged)} "
            f"measured={whole_exact} (exact frames)"
        )

    def _hyperspace(self):
        """Platform win: the verified multi-attempt primitive. Escape: ONE tap,
        then straight back to the reactive loop (re-tapping H after a survived
        escape would burn 3 energy per press, or stand still in a gaze)."""
        if self.st.player_xy() == self.ex.platform():
            won = sx.fire_hyperspace(
                self.ex, self.kbd, self.ex.platform(), self.live_log, self.result
            )
            self.result["won_flag"] = won
        else:
            self.live_log("-- ESCAPE HYPERSPACE (H, single tap) --")
            self.kbd.tap_action(sx.K_HYPERSPACE)
        self.acted = True
        self._observe()
        self._log("hyperspace", self.st.player_xy())


class LiveGreedy(LiveMixin, sim_player.Player):
    """The reactive greedy player over live VICE memory instead of the simulator."""


class LivePhase(LiveMixin, phase_player.PhasePlayer):
    """The phase player over live VICE memory; its tie rollouts run on simulator
    clones, so only the move it settles on reaches the real keyboard."""

    def _twin_class(self):
        """A fork is a SIMULATOR player: this class takes a live session, not a Game,
        and its executor drives a real keyboard."""
        return phase_player.PhasePlayer
