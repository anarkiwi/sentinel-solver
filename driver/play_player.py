#!/usr/bin/env python3
"""Run the reactive greedy player against the REAL game in VICE, recording an AVI.

LivePlayer reuses sentinel.player's decision logic verbatim; only observation
(re-read live memory each tick) and execution (the aim -> fire -> verify
primitive, real keystrokes) are overridden.  Usage: python -m driver.play_player
"""

import argparse
import json
import os
import time

from driver import core, kbd_aim, sentinel_execute as sx
from sentinel import actioncost, memmap as mm, player as sim_player
from sentinel.game import Game
from sentinel.state import State

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))


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


class LivePlayer(sim_player.Player):
    """The reactive player over live VICE memory instead of the simulator."""

    def __init__(self, session, log, result):
        game = Game(State.from_mem(core.live_image(session.bm)))
        super().__init__(game, verbose=True)
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
        """Snapshot live memory and LEAVE THE CPU HALTED: think time is a
        tooling artifact, so the world may advance only under real input
        (pans/actions), which the ROM-derived cost model already prices.
        A death is detected by the drain flag or by the landscape silently
        auto-resetting ($0CE5 re-frozen after we have acted) -- NOT by $0CDE
        bit7, which any survived hyperspace also sets."""
        with self.bm.halted():
            mem = core.live_image(self.bm)
        self.st = State.from_mem(mem)
        self.g.state = self.st
        if self.acted and self.st.mem[mm.PLAYER_NOT_ACTED] & 0x80:
            self.life_lost = "died (landscape auto-reset observed)"

    def _dead(self):
        if self.st.mem[mm.PLAYER_DIED_BY_DRAINING] & 0x80:
            self.life_lost = "drained at zero energy"
        if self.life_lost:
            self.live_log(f"DEAD: {self.life_lost}")
            self.result["death"] = self.life_lost
            return True
        return False

    def _advance(self, frames):
        """Real time passes in the live game; the model clock is a no-op."""

    def _wait(self):
        """A deliberate wait spends REAL world time: resume the CPU (observe
        left it halted), let PAL frames elapse, and re-observe."""
        self.bm.exit()
        time.sleep(sim_player.WAIT_FRAMES / 50.0)

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
        aim_charged = self._aim_frames(view)
        settle_charged = actioncost.SETTLE.get(pverb, 60)
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


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("landscape", nargs="?", type=int, default=0)
    parser.add_argument("--max-actions", type=int, default=120)
    parser.add_argument("--video", default=None)
    args = parser.parse_args(argv)
    digits = f"{args.landscape:04d}"
    video = args.video or f"player_ls{args.landscape}_win.avi"
    tap = os.path.join(ROOT, "sentinel-gold.tap")
    renders = os.path.join(ROOT, "renders")
    result = {"landscape": args.landscape, "actions": [], "energy_curve": []}

    def log(msg):
        print(f"[{time.strftime('%H:%M:%S')}] {msg}", flush=True)

    def play_fn(session):
        lp = LivePlayer(session, log, result)
        won = lp.run(max_actions=args.max_actions)
        result["won"] = bool(won and lp.ex.won())
        result["trace"] = [list(r) for r in lp.trace]
        result["final_energy"] = lp.st.energy
        log(f"play loop done: won={result['won']} actions={len(lp.trace)}")

    core.boot_and_play(tap, renders, digits, video, log, play_fn, result)
    ok, size, frames, msg = core.validate_avi(result.get("video", ""))
    result["avi"] = {"ok": ok, "bytes": size, "frames": frames, "msg": msg}
    out_path = os.path.join(ROOT, "out", f"play_player_{digits}.json")
    with open(out_path, "w") as fh:
        json.dump(result, fh, indent=1, default=str)
    log(
        f"RESULT: won={result.get('won')} avi={msg} ({frames} frames, {size} bytes) "
        f"-> {result.get('video')}; log {out_path}"
    )
    return 0 if result.get("won") and ok else 1


if __name__ == "__main__":
    raise SystemExit(main())
