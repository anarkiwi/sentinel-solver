#!/usr/bin/env python3
"""Lockstep probe variant that DUMPS per-step (pre_img, live_img, live_frames,
live_cools) to scratchpad so the sim cadence can be reproduced offline, and does
NOT stop at the first divergence -- it records the locked diffs per step and keeps
going a few steps so we can see whether the off-by-one compounds.
"""

import os, sys, pickle

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.abspath(os.path.join(HERE, ".."))
sys.path.insert(0, ROOT)

from driver import core, kbd_aim
from driver.sentinel_execute import Executor, perform_step
from solver import plan_game
from solver.astar_planner import plan
from scripts.lockstep_probe import UPDATE_ENEMIES_PC

DUMP = os.path.join(ROOT, "out", "lockstep_steps.pkl")


def run_probe(session, log, result):
    bm = session.bm
    bm.auto_resume = False
    ex = Executor(bm, log)
    drv = kbd_aim.KbdDriver(bm, log, quantized=True)
    landscape = session.landscape
    from vice_driver.binmon import CHECK_EXEC

    cp = bm.checkpoint_set(UPDATE_ENEMIES_PC, op=CHECK_EXEC, silent=True)
    fp = bm.checkpoint_set(0x9630, op=CHECK_EXEC, silent=True)
    dp = bm.checkpoint_set(0x130C, op=CHECK_EXEC, silent=True)

    def hits():
        return bm.checkpoint_get(cp.checknum).hit_count

    def frames():
        return bm.checkpoint_get(fp.checknum).hit_count

    def cools():
        return bm.checkpoint_get(dp.checknum).hit_count

    with bm.halted():
        m0 = bytes(core.live_image(bm))
    g = plan_game.PlanGame.from_mem(m0, landscape, seed_built_columns=False)
    pr = plan(g)
    log(f"PLAN from LIVE entry: won={pr.won} steps={len(pr.steps)}")
    if not pr.won:
        result["divergence"] = "plan failed"
        return

    dumps = []
    STOP_AFTER = 4  # dump first few steps then stop the live run
    for i, stp in enumerate(pr.steps):
        if i > STOP_AFTER:
            break
        verb, tile = stp["verb"], tuple(stp["target"])
        with bm.halted():
            pre_img = bytes(core.live_image(bm))
            f_before = frames()
            h_before = hits()
            c_before = cools()
        outcome = perform_step(ex, drv, f"L{i}", stp, log, result)
        with bm.halted():
            live_img = bytes(core.live_image(bm))
            live_frames = frames() - f_before
            live_rounds = hits() - h_before
            live_cools = cools() - c_before
        dumps.append(
            {
                "i": i,
                "verb": verb,
                "tile": tile,
                "step": stp,
                "outcome": outcome,
                "pre_img": pre_img,
                "live_img": live_img,
                "live_frames": live_frames,
                "live_rounds": live_rounds,
                "live_cools": live_cools,
            }
        )
        log(
            f"DUMP step {i} {verb} {tile}: outcome={outcome} "
            f"frames={live_frames} cools={live_cools} UE={live_rounds}"
        )
        if outcome not in ("ok", "best_effort_miss"):
            break
    with open(DUMP, "wb") as f:
        pickle.dump(dumps, f)
    log(f"wrote {DUMP} ({len(dumps)} steps)")
    result["divergence"] = None


def main():
    def log(m):
        print(m, flush=True)

    os.environ.setdefault("NO_RECORD", "1")
    result = {"won": False, "energy_curve": [], "actions": []}
    tap = os.path.join(ROOT, "sentinel-gold.tap")
    renders = os.path.join(ROOT, "renders")

    def play(session):
        run_probe(session, log, result)

    core.boot_and_play(tap, renders, "0000", "lockstep.avi", log, play, result)


if __name__ == "__main__":
    main()
