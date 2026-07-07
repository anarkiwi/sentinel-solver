#!/usr/bin/env python3
"""The live plan runner: drive the solver in the REAL game (asid-vice) by keyboard,
verified by the ROM's own win flag ($0CDE bit6), optionally recorded to an AVI. This is
the glue that wires the solver (solver/) to the driver (driver/); it plans and verifies
nothing itself beyond executing steps.

It is the live counterpart of scripts/run_plan_simulated.py: the SAME bare loop --
resync the planner from the real game, ask the solver (climb_search's receding-horizon
lookahead) for the next move, drive it, replan -- with NO avoidance tricks or tuning
overrides layered on top. The only difference from the simulated version is the "real
game" here is asid-vice instead of the bit-exact simulator, so real timing / enemy /
meanie divergence self-heals (or defeats the plan) on the next resync.

NO PIXELS: aim and verification are entirely from MEMORY reads. Aiming uses the driver's
keyboard sights-cursor path (kbd_aim.KbdDriver): drive the view angles sights-off
(S/D/L/COMMA + U-turn) then the sights cursor sights-on, closed-loop on the live native-LOS
probe (sentinel.los on a RAM snapshot) until the target tile is hit with LOS. The action is
a real keystroke (R/B robot/boulder, Q transfer, A absorb, H hyperspace) fired via
tap_action, which polls the game's own $0CE9 action latch. Reads never change state.

Everything reusable lives in the driver, not here: boot + landscape entry + snapshot
caching + monitor-resilience + the 64 KB live-image read + the live sights-ray probe
(driver/core), reading/comparing the live board (driver/sentinel_state: read_game_state,
GameState.objects_at, verify_entry) and arbitrating an action's memory delta
(driver/sentinel_execute: verify, otype_cost, the Executor accessors). This file is just
the plan-execution loop that wires the solver to those pieces, plus the CLI.
"""

import os, sys, time, argparse

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.abspath(os.path.join(HERE, ".."))
sys.path.insert(0, os.path.dirname(HERE))

from driver.sentinel_execute import Executor, perform_step, fire_hyperspace
from driver import kbd_aim
from driver import core

TAP = os.path.join(ROOT, "sentinel-gold.tap")


def run(
    typed_digits,
    max_seconds,
    log,
    video_name="solver_run_0042.avi",
):
    """Glue: boot the real game (driver.core.boot_and_play), then run the plan loop
    (execute_live) inside the booted session. All emulator driving -- container, boot
    retries, binmon connect, landscape entry, video record -- lives in the driver."""
    renders_host = os.path.join(ROOT, "renders")
    result = {
        "won": False,
        "actions": [],
        "divergence": None,
        "energy_curve": [],
    }

    def play(session):
        ex = Executor(session.bm, log)
        st = session.state0
        result["entry_match"] = session.entry_match
        log(
            f"IN PLAY: slot {st.player_slot} @ ({st.player.x},{st.player.y}) "
            f"energy {st.player_energy} objs {len(st.objects)} platform {ex.platform()}"
        )
        result["won"] = execute_live(
            ex, log, result, session.t_start, max_seconds, session.landscape
        )

    return core.boot_and_play(
        TAP, renders_host, typed_digits, video_name, log, play, result
    )


def execute_live(
    ex,
    log,
    result,
    t_start,
    max_seconds,
    landscape,
    max_iterations=200,
):
    """Closed-loop climb: at each iteration, RESYNC the native climb model from live
    memory (ground truth, including any real enemy/meanie activity since the last
    look), compute the next foothold move fresh (climb_search.search_iterate), and
    drive it via real keystrokes immediately -- so a step that a stale precomputed
    plan assumed would land cleanly, but which real timing/enemies made infeasible,
    self-heals on the NEXT iteration (a fresh resync never has the failed object,
    so the planner routes around it) instead of cascading into a run-ending
    divergence."""
    from solver import plan_game
    from solver import climb_search as csearch

    # the same planner config the simulated runner uses (run_plan_simulated defaults):
    # drive toward the platform, gate the ROM-infeasible on-distant-boulder synthoid in
    # the steep ring, depth-2 / beam-2 lookahead.
    # Env-overridable so the live run uses the SAME planner config as the winning
    # simulated run (run_plan_simulated). Defaults preserve the prior live behaviour.
    toward_plat = os.environ.get("TOWARD_PLAT", "1") == "1"
    near_plat_radius = int(os.environ.get("NEAR_PLAT_RADIUS", "2"))
    search_depth = int(os.environ.get("SEARCH_DEPTH", "2"))
    search_beam = int(os.environ.get("SEARCH_BEAM", "2"))

    # decision function: the receding-horizon best-first lookahead (climb_search,
    # SEARCH_REDESIGN.md) which won't commit a move without a continuation within the
    # horizon (fixes the dead-end/reposition failure of a purely greedy picker).
    def decide(g, ctx, blocked_set, lg):
        return csearch.search_iterate(
            g, ctx, blocked_set, lg, depth=search_depth, beam=search_beam
        )

    log(f"LIVE decision: lookahead search D{search_depth} B{search_beam}")

    drv = kbd_aim.KbdDriver(ex.bm, log)
    blocked = set()
    label = 0
    # The lookahead search spends several real SECONDS per decision. Under warp the
    # emulator would run millions of cycles in that gap -- enemies rotate, a meanie can
    # arm/spawn, the player can be drained -- so the state the search read is stale by the
    # time it acts, and the idle monitor socket goes flaky (the observed aim TimeoutErrors).
    # FREEZE the CPU across resync+decide (bm.halted() leaves it stopped after the mem read,
    # and the search issues no monitor I/O, so it stays frozen), then resume before driving
    # keystrokes. Now meanie_safe decides against the true state at action time.
    halt_during_decide = True

    def resync(seed_built_columns=True, keep_halted=False):
        if keep_halted:
            with ex.bm.halted():  # read with the CPU LEFT stopped (no auto-resume)
                mem = core.live_image(ex.bm)
        else:
            mem = core.live_image(ex.bm)
        return plan_game.Game.from_mem(
            mem, landscape, seed_built_columns=seed_built_columns
        )

    # seed_built_columns must stay False until a create() has ACTUALLY landed in the
    # real game: before that, the only object standing on the player's tile is the
    # landscape generator's original spawn placement, which -- like any first-level
    # object -- carries the ROM's fixed z_fraction=$E0 render offset. Seeding that
    # into g.col raises eye by 0.875 above the ROM-validated offline baseline and
    # steers the climb onto different (worse) footholds (see plan_game.Game.from_mem
    # docstring). Once a real create() lands, the player's tile IS one this model
    # built, so the full z+zf/256 reconstruction becomes correct.
    built_anything = False
    g = resync(seed_built_columns=False)
    ctx = csearch.climb_ctx(g, toward_plat, near_plat_radius)
    log(
        f"LIVE climb start: {g.player_xy()} eye {g.eye} plat {ctx['plat']} energy {g.energy}"
    )

    status = "retry"
    for it in range(max_iterations):
        if time.time() - t_start > max_seconds:
            log(
                f"TIME BUDGET ({max_seconds}s) exceeded at live iteration {it}; aborting"
            )
            result["divergence"] = f"timeout at live iteration {it}"
            return False
        g = resync(seed_built_columns=built_anything, keep_halted=halt_during_decide)
        before_n = len(g.steps)
        status = decide(g, ctx, blocked, log)  # CPU frozen here if halt_during_decide
        if halt_during_decide:
            try:
                ex.bm.exit()  # resume the CPU (frozen during the search) for the keystrokes
            except Exception as e:
                log(f"    (resume after search failed: {e}; reconnecting)")
                core.reconnect(ex.bm, log)
        new_steps = g.steps[before_n:]
        blocked_this_round = False
        built_tiles_this_batch = set()
        for stp in new_steps:
            label += 1
            tile = tuple(stp["target"])
            outcome = perform_step(ex, drv, f"L{label}", stp, log, result)
            if outcome in ("ok", "best_effort_miss"):
                if outcome == "ok" and stp["verb"] == "create":
                    built_tiles_this_batch.add(tile)
                    built_anything = True
                continue
            if outcome == "aim_miss":
                # the aim couldn't reach the requested view, so NOTHING was fired -- the
                # tile is still valid, only the view needs another try. Resync + re-plan
                # (which re-snaps a fresh view) WITHOUT blocking the foothold.
                log(
                    f"    LIVE: {stp['verb']} {tile} -> aim_miss (nothing fired); "
                    "resync + re-plan, tile not blocked"
                )
                blocked_this_round = True
                break
            if stp["verb"] == "create":
                if tile in built_tiles_this_batch:
                    # the boulder half of this foothold already landed for real; only
                    # the synthoid-on-boulder half failed. Don't block the tile -- the
                    # next resync sees a real boulder there and the natural "hop onto
                    # an existing boulder" candidate is exactly the correct retry.
                    log(
                        f"    LIVE: {stp['verb']} {tile} -> {outcome} (boulder already "
                        "landed; not blocking, natural hop retry will pick it up)"
                    )
                else:
                    # candidate key is (tile, use_boulder) -- a plain hop is otype 0.
                    blocked.add((tile, stp["otype"] == 3))
                    log(
                        f"    LIVE: {stp['verb']} {tile} -> {outcome}; blocking foothold"
                    )
            else:
                log(
                    f"    LIVE: {stp['verb']} {tile} -> {outcome}; resyncing and replanning"
                )
            blocked_this_round = True
            break
        if blocked_this_round:
            continue
        if status in ("no_gain", "stuck"):
            log(
                f"  LIVE climb stopped ({status}); attempting endgame from current state"
            )
            break
        if status == "approach":
            break
    else:
        log(
            f"LIVE climb: hit max_iterations ({max_iterations}) without reaching approach"
        )

    # ---- ENDGAME: resync once more, then absorb Sentinel + platform synthoid ----
    g = resync()
    before_n = len(g.steps)
    won_native = csearch.endgame(g, ctx["plat"], log)
    for stp in g.steps[before_n:]:
        label += 1
        outcome = perform_step(ex, drv, f"E{label}", stp, log, result)
        if outcome not in ("ok", "best_effort_miss"):
            log(
                f"    LIVE endgame step {stp['verb']} {tuple(stp['target'])}: {outcome}"
            )
            result["divergence"] = (
                f"endgame {stp['verb']} {tuple(stp['target'])}: {outcome}"
            )
            return False
    if not won_native:
        result["divergence"] = result.get("divergence") or (
            f"endgame not reachable from final climb state {g.player_xy()} eye {g.eye}"
        )
        log(f"  LIVE: endgame not reachable from {g.player_xy()} eye {g.eye}")
        return False

    return fire_hyperspace(ex, drv, ctx["plat"], log, result)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--digits", default="0042")
    ap.add_argument("--max-seconds", type=int, default=1500)
    ap.add_argument("--video-name", default=None)
    args = ap.parse_args()
    video_name = args.video_name or f"solver_run_{args.digits}.avi"

    def log(m):
        print(m, flush=True)

    log(f"=== VICE keyboard record: type {args.digits!r} LIVE replanning ===")
    result = run(
        args.digits,
        args.max_seconds,
        log,
        video_name=video_name,
    )

    vid = result.get("video")
    if vid:
        ok, size, nfr, msg = core.validate_avi(vid)
        result["video_valid"] = ok
        result["video_size"] = size
        result["video_frames"] = nfr
        import subprocess

        ft = ""
        try:
            ft = subprocess.run(
                ["file", "-b", vid], capture_output=True, text=True, timeout=10
            ).stdout.strip()
        except Exception:
            pass
        log(
            f"AVI: {vid}\n   valid={ok} size={size}B ({size/1024:.1f} KiB) "
            f"frames={nfr} ({msg})\n   file: {ft}"
        )

    done = result.get("landscape_done", 0)
    n_ok = sum(1 for a in result.get("actions", []) if a["ok"])
    log("\n=== RESULT ===")
    log(f"  entry match  : {result.get('entry_match')}")
    log(f"  steps OK     : {n_ok}/{len(result.get('actions', []))} keyboard steps")
    if result.get("energy_block"):
        log(f"  ENERGY BLOCK : {result['energy_block']}")
    log(f"  $0CDE flag   : {done:#04x} (bit6 = landscape complete)")
    log(f"  WIN VERIFIED : {'PASS' if result.get('won') else 'FAIL'}")
    if result.get("divergence"):
        log(f"  DIVERGENCE   : {result['divergence']}")
    log(f"  wall seconds : {result.get('wall_seconds')}")
    log(
        f"  video        : {vid} valid={result.get('video_valid')} "
        f"{result.get('video_size',0)}B frames={result.get('video_frames',0)}"
    )
    # dump the energy curve for the report
    ec = result.get("energy_curve", [])
    if ec:
        log(
            "  energy curve : "
            + " ".join(f"{e['step']}:{e['energy_before']}" for e in ec)
        )
    return 0 if result.get("won") else 1


if __name__ == "__main__":
    sys.exit(main())
