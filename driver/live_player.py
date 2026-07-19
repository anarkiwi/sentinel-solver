#!/usr/bin/env python3
"""Player-agnostic machinery for driving a sim player against the REAL game in VICE.

``LiveMixin`` carries ONLY observation + execution over live memory; decision
logic (``_tick``, A*'s ``_search``) comes from the composed sim player via MRO.
It goes first in the bases so its execution overrides win over ``BasePlayer``'s.
"""

from driver import clock, core, kbd_aim, sentinel_execute as sx
from sentinel import astar_player, memmap as mm, player as sim_player, playerbase
from sentinel.game import Game
from sentinel.state import State


class MeasuringKbdDriver(kbd_aim.KbdDriver):
    """KbdDriver that times each aim primitive into ``subframes`` (exact $9630 frames
    via the shared Executor) for per-sub-term charge validation. Reset per step;
    sights_on delegates to sights_set so the toggle is timed once, not twice."""

    def __init__(self, bm, log, quantized=False, ex=None):
        super().__init__(bm, log, quantized=quantized)
        self._ex = ex
        self.subframes = {}

    def _timed(self, name, fn, *args):
        f0 = self._ex.frames()
        r = fn(*args)
        df = self._ex.frames() - f0
        self.subframes[name] = self.subframes.get(name, 0) + df
        return r

    def coarse_h(self, want):
        return self._timed("pan_h", super().coarse_h, want)

    def coarse_v(self, want):
        return self._timed("pan_v", super().coarse_v, want)

    def fine_cursor(self, cx, cy):
        return self._timed("cursor", super().fine_cursor, cx, cy)

    def sights_set(self, on):
        return self._timed("toggle", super().sights_set, on)


class LiveMixin:
    """Observation + execution over live VICE memory; no decision logic.

    ``__init__`` builds the sim ``Game`` from a live snapshot and forwards any
    player-specific kwargs (A*'s ``node_budget``/``time_budget``/``weight``) to
    the composed sim player's constructor via ``super().__init__``."""

    def __init__(self, session, log, result, **kwargs):
        game = Game(State.from_mem(core.live_image(session.bm)))
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

    def _observe(self):
        """Snapshot live memory and LEAVE THE CPU HALTED: think time is a tooling
        artifact, so the world advances only under real input (pans/actions),
        already priced by the cost model.  Death is the drain flag or a silent
        landscape auto-reset ($0CE5 re-frozen after we acted), NOT $0CDE bit7."""
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

    def _dead(self):
        if self.st.mem[mm.PLAYER_DIED_BY_DRAINING] & 0x80:
            self.life_lost = "drained at zero energy"
        if self.life_lost:
            self.live_log(f"DEAD: {self.life_lost}")
            self.result["death"] = self.life_lost
            return True
        return False

    def _advance(self, frames):
        """Real time passes in the live game; the model clock is a no-op.  For A*
        this keeps heavy ``_search`` think time out of the live world -- the real
        world moves only when ``_fire`` replays the plan's keystrokes."""

    def _wait(self):
        """A deliberate wait spends REAL world time, FRAME-EXACT: step the $9630
        per-frame marker ``WAIT_FRAMES`` times, leaving the CPU halted.  charged ==
        measured now holds by construction whatever the warp state, so ``wait_audit``
        is a regression pin; a stalled marker raises instead of being waited out."""
        want = playerbase.WAIT_FRAMES
        got = clock.run_frames(self.bm, want)
        self.result.setdefault("wait_audit", []).append([self.step_no, want, got])
        self.live_log(f"    [clock] wait: charged={want} measured={got} (exact frames)")

    def _drive_transfer_aim(self, tile, view):
        """Aim the sights onto `tile` for a transfer (perform_step drives the aim
        only for create/absorb).  Reuses a matching committed bearing, else drives
        the full view; the live ray probe confirms the landing."""
        want = (view["h_angle"], view["v_angle"])
        if self.kbd.sights_live_on() and self.kbd.committed_bearing() == want:
            self.kbd.fine_cursor(*view["cursor"])
        else:
            ach = self.kbd.drive_to(view)
            if not ach["ok"]:
                self.kbd.clear_bearing()
                return False
            self.kbd.set_bearing(*want)
        rx, ry, los_hit, _ = core.probe_tile(self.bm)
        if (rx, ry) != tuple(tile) or not los_hit:
            self.live_log(
                f"    transfer aim probe ({rx},{ry}) los={los_hit} != {tuple(tile)}"
            )
        return True

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
        aim_charged = self._step_aim_frames(pverb, view)
        # View-aware transfer settle: scene-dependent projector replot, not flat 47.
        settle_charged = self._settle(pverb, view, self._settle_eye(pverb, tile))
        charged = aim_charged + settle_charged
        out = sx.perform_step(
            self.ex, self.kbd, f"p{self.step_no}", stp, self.live_log, self.result
        )
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
                ]
            )
        self.result.setdefault("frame_audit", []).append(
            [lbl, pverb, list(tile), round(charged), whole_exact]
        )
        self.live_log(
            f"    [clock] p{self.step_no} {pverb}: charged={round(charged)} "
            f"measured={whole_exact} (exact frames)"
        )
        self.acted = True
        self._observe()
        ok = out in ("ok", "diverge")  # diverge: primary effect landed, world moved
        if ok:
            self._log(verb, tile)
        return ok

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

    def _plan_step_stale(self, verb, tile, view):
        """Re-validate the next planned step against the LIVE enemy phase. The plan's
        drain-safety was gated on the sim's PREDICTED phase, which drifts from live
        over multi-step execution; if the player's own gaze window no longer covers
        this step's aim+settle plus one step's cost-interval margin (depth 0: the
        board is freshly observed), replan from the observed board rather than stand
        exposed for longer than the window.

        The margin absorbs prediction error; it may not deadlock.  Once ``_restale``
        has waited on this same step (``_stale`` count > 1, so the enemy phase behind
        the earlier verdict is gone) and the RAW budget clears, the step proceeds."""
        budget = self._step_aim_frames(verb, view) + self._settle(
            verb, view, self._settle_eye(verb, tile)
        )
        margin_fn = getattr(self, "_margin", None)  # planner-only (greedy has none)
        margin = margin_fn(0) if margin_fn is not None else 0.0
        window = self._player_window()
        if window >= budget + margin:
            return False
        stale = self._stale
        waited = stale is not None and stale[0] == (verb, tuple(tile)) and stale[1] > 1
        if waited and window >= budget:
            self.live_log(
                f"    (plan step {verb} {tile}: window {window:.0f}f clears the raw "
                f"budget {budget:.0f}f after a wait; margin-only block released)"
            )
            return False
        self.live_log(
            f"    (plan step {verb} {tile}: live gaze window {window:.0f}f "
            f"< step budget {budget:.0f}f + margin {margin:.0f}f; replan from live)"
        )
        return True


class LiveGreedy(LiveMixin, sim_player.Player):
    """The reactive greedy player over live VICE memory instead of the simulator."""


class LiveAStar(LiveMixin, astar_player.AStarPlayer):
    """The A* planner over live VICE memory; ``_search`` think time stays out of
    the live clock, the real world moving only as ``_fire``/``_hyperspace``
    replay the plan's keystrokes."""
