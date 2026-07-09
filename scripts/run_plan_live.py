#!/usr/bin/env python3
"""The live plan runner: drive the solver in the REAL game (asid-vice) by keyboard,
verified by the ROM's own win flag ($0CDE bit6), optionally recorded to an AVI. This is
the glue that wires the solver (solver/) to the driver (driver/); it plans and verifies
nothing itself beyond executing steps.

It is the live counterpart of scripts/run_plan_simulated.py, mirroring its structure on
asid-vice instead of the bit-exact simulator: RESYNC a PlanGame from live memory, compute
one full HIDDEN offline plan (solver.astar_planner.plan) whose steps already END with the
endgame (absorb Sentinel -> create platform synthoid -> transfer onto it), execute those
steps one at a time (driver.sentinel_execute.perform_step), and on a step the live game
made infeasible (aim could not be driven / drained below a build cost) RESYNC + re-plan()
from the true live state, capped at --max-replans. A step whose fire verify() REJECTS
(wrong tile / count / energy delta) is a CRASH: the offline plan is aim-exact, so a
wrong-target fire means the model diverged from the ROM -- it raises, it does not smooth
over or replan. On a clean run of every step, the final hyperspace fires from the platform
tile (the endgame transfer landed the player there) and sets the ROM win flag.

NO PIXELS: aim and verification are entirely from MEMORY reads. Aiming uses the driver's
keyboard sights-cursor path (kbd_aim.KbdDriver): drive the view angles sights-off
(S/D/L/COMMA + U-turn) then the sights cursor sights-on, closed-loop on the live native-LOS
probe (sentinel.los on a RAM snapshot) until the target tile is hit with LOS. The action is
a real keystroke (R/B robot/boulder, Q transfer, A absorb, H hyperspace) fired via
tap_action, which polls the game's own $0CE9 action latch. Reads never change state.

Everything reusable lives in the driver/solver, not here: boot + landscape entry + snapshot
caching + monitor-resilience + the 64 KB live-image read + the live sights-ray probe
(driver/core), arbitrating an action's memory delta (driver/sentinel_execute), and the
offline planner itself (solver/astar_planner). This file is just the plan-execution loop
that wires the planner to the driver, plus the CLI.
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
    max_replans=4,
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
            ex,
            log,
            result,
            session.t_start,
            max_seconds,
            session.landscape,
            max_replans=max_replans,
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
    max_replans=4,
):
    """Live offline-plan loop, the asid-vice counterpart of run_plan_simulated.run.

    RESYNC a PlanGame from live memory, compute one full HIDDEN offline plan
    (solver.astar_planner.plan) whose steps already END with the endgame (absorb
    Sentinel -> create platform synthoid -> transfer onto the platform tile), then
    execute those steps one at a time via driver.sentinel_execute.perform_step:

      * "ok"/"best_effort_miss" -- continue (a landed create seeds built columns).
      * "aim_miss"/"drained"/"diverge" -- the AIM was not wrong: nothing fired (view
                                   undriveable / energy below cost), or the object DID
                                   land on the target tile but the world diverged under
                                   it (a concurrent meanie spawn / discharge / in-window
                                   drain -- "diverge"). RESYNC + re-plan() from the true
                                   live state and restart, capped at ``max_replans``.
      * "fail"                  -- verify() rejected the fire AND the primary on-tile
                                   effect did NOT happen: the object landed on the WRONG
                                   tile. The offline plan is aim-exact, so a wrong-tile
                                   landing means the MODEL diverged from the ROM: raise
                                   RuntimeError. Do NOT smooth it over or replan.

    On a clean run of every step the endgame transfer has landed the player ON the
    platform tile, so fire the final hyperspace from there (fire_hyperspace sets the
    ROM win flag $0CDE bit6 and is the sole win arbiter). A plan that reports not-won
    is a genuine loss: log it loudly, record the divergence, return False (no fake win).

    The emulator CPU is FROZEN across each resync+plan -- planning spends tens of
    seconds and issues no monitor I/O, so a warped emulator would drift the enemy
    round / drain the player under the snapshot the plan was built from -- and resumed
    before any keystroke.
    """
    from solver import plan_game
    from solver.astar_planner import plan

    drv = kbd_aim.KbdDriver(ex.bm, log)
    halt_during_decide = True
    label = 0

    def resync(seed_built_columns, keep_halted):
        if keep_halted:
            with ex.bm.halted():  # read with the CPU LEFT stopped (no auto-resume)
                mem = core.live_image(ex.bm)
        else:
            mem = core.live_image(ex.bm)
        return plan_game.PlanGame.from_mem(
            mem, landscape, seed_built_columns=seed_built_columns
        )

    def resume_cpu():
        if not halt_during_decide:
            return
        try:
            ex.bm.exit()  # resume the CPU (frozen during planning) for the keystrokes
        except Exception as e:  # noqa: broad-except
            log(f"    (resume after plan failed: {e}; reconnecting)")
            core.reconnect(ex.bm, log)

    # seed_built_columns must stay False until a create() has ACTUALLY landed in the
    # real game: before that, the only object on the player's tile is the landscape
    # generator's original spawn placement, which carries the ROM's fixed z_fraction=$E0
    # render offset. Seeding that into g.col raises eye by 0.875 above the ROM-validated
    # offline baseline and steers the plan onto different (worse) footholds. Once a real
    # create() lands, the player's tile IS one this model built, so seeding is correct.
    built_anything = False

    for replan in range(max_replans + 1):
        if time.time() - t_start > max_seconds:
            log(f"TIME BUDGET ({max_seconds}s) exceeded at replan {replan}; aborting")
            result["divergence"] = f"timeout at replan {replan}"
            return False

        # FREEZE the CPU across resync + plan, then resume before any keystroke.
        g = resync(seed_built_columns=built_anything, keep_halted=halt_during_decide)
        log(
            f"LIVE plan {replan}: {g.player_xy()} eye {g.eye} plat {g.plat} "
            f"energy {g.energy}"
        )
        result_plan = plan(g)  # CPU frozen here if halt_during_decide
        resume_cpu()

        if not result_plan.won:
            log("")
            log("!!! LIVE PLAN FAILED -- no hidden route found; this is a LOSS !!!")
            log(f"    failure: {result_plan.failure}")
            log(f"    stats  : {result_plan.stats}")
            result["divergence"] = f"plan failed: {result_plan.failure}"
            return False
        log(
            f"  plan {replan}: won route with {len(result_plan.steps)} steps "
            f"(nodes {result_plan.stats.get('nodes')}, "
            f"{result_plan.stats.get('wall_s')}s wall)"
        )

        replan_needed = False
        for stp in result_plan.steps:
            label += 1
            tile = tuple(stp["target"])
            if (
                os.environ.get("DUMP_LIVE")
                and stp["verb"] == "absorb"
                and stp.get("otype") == 5
            ):
                # Capture the PRE-fire live 64 KB image for the endgame Sentinel absorb,
                # to diff the live launch state (player pos/eye, objects on the down-look
                # ray) against the offline forecast when the far launch misses live.
                try:
                    with ex.bm.halted():
                        _mem = core.live_image(ex.bm)
                    _p = os.path.join(ROOT, "renders", f"live_endgame_L{label}.bin")
                    with open(_p, "wb") as _f:
                        _f.write(bytes(_mem))
                    _lg = plan_game.PlanGame.from_mem(_mem, landscape)
                    log(
                        f"    [DUMP_LIVE] pre-endgame -> {_p}; live player "
                        f"{_lg.player_xy()} eye {_lg.eye} energy {_lg.energy}"
                    )
                except Exception as _e:  # noqa: broad-except
                    log(f"    [DUMP_LIVE] failed: {_e}")
            outcome = perform_step(ex, drv, f"L{label}", stp, log, result)
            if outcome in ("ok", "best_effort_miss"):
                if outcome == "ok" and stp["verb"] == "create":
                    built_anything = True
                continue
            if outcome in ("aim_miss", "drained", "diverge"):
                # The AIM was not wrong. Either nothing fired (the live view could not be
                # driven / energy fell below cost), or the primary on-tile effect DID land
                # but the world diverged under it -- a concurrent meanie spawn / discharge
                # / in-window drain ("diverge"). Both are live-state divergences the loop is
                # designed to self-heal: resync + re-plan() from the true live state (which
                # now reflects the landed object and any new meanies) and restart.
                log(f"    LIVE: {stp['verb']} {tile} -> {outcome}; resync + re-plan")
                replan_needed = True
                break
            # outcome == "fail": verify() rejected the fire AND the primary on-tile effect
            # did NOT happen -- the object landed on the WRONG tile (or no tile). The offline
            # plan is aim-exact, so a wrong-tile landing means the MODEL is wrong.
            # Do NOT smooth it over, do NOT replan -- CRASH loudly and let it propagate.
            raise RuntimeError(
                f"LIVE aim-exact plan fired on the WRONG target: step {stp['verb']} "
                f"{tile} otype={stp.get('otype')} (label L{label}) -> verify() rejected "
                f"it (outcome={outcome}). The offline plan is aim-exact; a rejected fire "
                f"means the model diverged from the ROM."
            )
        if replan_needed:
            continue

        # Every step verified against live memory, including the endgame (absorb
        # Sentinel -> create platform synthoid -> transfer). That final transfer landed
        # the player ON the platform tile, so fire the winning hyperspace from there.
        return fire_hyperspace(ex, drv, g.plat, log, result)

    log(f"LIVE: replan cap ({max_replans}) reached without a clean execution")
    result["divergence"] = result.get("divergence") or (
        f"replan cap ({max_replans}) reached without a clean run"
    )
    return False


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--digits", default="0042")
    ap.add_argument("--max-seconds", type=int, default=1500)
    ap.add_argument("--max-replans", type=int, default=4)
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
        max_replans=args.max_replans,
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
