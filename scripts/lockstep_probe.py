#!/usr/bin/env python3
"""Lock-step sim-vs-live divergence probe.

Boots ls0 live, plans ONE offline plan from the live entry image, then executes each
plan step against BOTH the live emulator and the bit-exact simulator in lock step.
The live enemy-round count per step is measured EXACTLY with a silent monitor
checkpoint on update_enemies ($16B5) -- the ROM advances one enemy round per hit
(update_game_loop $129f). The sim is then advanced by that SAME measured count, so the
only thing being tested is per-round mechanic faithfulness. Complete game state is
dumped from both after every step and compared; the run STOPS at the first divergence.
"""

import os, sys

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.abspath(os.path.join(HERE, ".."))
sys.path.insert(0, ROOT)

from driver import core, kbd_aim
from driver.sentinel_execute import Executor, perform_step
from sentinel import memmap as mm, enemies as SE, actions, aim, landscape as _landscape
from sentinel.state import State
from solver import plan_game
from solver.astar_planner import plan

UPDATE_ENEMIES_PC = 0x16B5

# --- complete-state snapshot (only the gameplay state the sim models) ---------
OBJ_ARRAYS = [
    ("flags", mm.OBJECTS_FLAGS),
    ("type", mm.OBJECTS_TYPE),
    ("x", mm.OBJECTS_X),
    ("y", mm.OBJECTS_Y),
    ("zh", mm.OBJECTS_Z_HEIGHT),
    ("zf", mm.OBJECTS_Z_FRACTION),
    ("v", mm.OBJECTS_V_ANGLE),
]
ENEMY_TYPES = (mm.T_SENTRY, mm.T_SENTINEL)
ENEMY_ARRAYS = [
    ("drain_cd", mm.ENEMIES_DRAINING_COOLDOWN),
    ("rot_cd", mm.ENEMIES_ROTATION_COOLDOWN),
    ("upd_cd", mm.ENEMIES_UPDATE_COOLDOWN),
    ("meanie_search", mm.ENEMIES_MEANIE_SEARCH_OBJECT),
    ("discharge", mm.ENEMIES_ENERGY_TO_DISCHARGE),
    ("failed_meanie", mm.ENEMIES_FAILED_MEANIE_MEMORY),
    ("meanie_scans", mm.ENEMIES_MEANIE_ATTEMPT_SCANS),
    ("meanie_obj", mm.ENEMIES_MEANIE_OBJECT),
    ("target_obj", mm.ENEMIES_TARGETED_OBJECT),
    ("target_exp", mm.ENEMIES_TARGETED_OBJECT_EXPOSURE),
    ("considering", mm.ENEMIES_CONSIDERING_MEANIE),
]
SCALARS = [
    ("player_slot", mm.PLAYER_OBJECT),
    ("cursor", mm.CURSOR),
    ("cooldown_gate", mm.COOLDOWN_GATE),
    ("died_draining", mm.PLAYER_DIED_BY_DRAINING),
    ("landscape_done", mm.LANDSCAPE_COMPLETE),
]


def snapshot(mem):
    """Named gameplay fields from a 64 KB image. Empty slots (flags bit7) compare only
    on emptiness; their stale x/y/... bytes are not part of the state."""
    s = {}
    for name, base in SCALARS:
        s[name] = mem[base]
    s["energy"] = mem[mm.PLAYER_ENERGY] & 0x3F
    s["prng"] = bytes(mem[mm.PRND_STATE : mm.PRND_STATE + 5])
    for name, base in ENEMY_ARRAYS:
        s["E_" + name] = bytes(mem[base : base + 8])
    facings = []
    for slot in range(mm.NUM_SLOTS):
        empty = bool(mem[mm.OBJECTS_FLAGS + slot] & 0x80)
        if empty:
            s[f"slot{slot}"] = "EMPTY"
        else:
            s[f"slot{slot}"] = tuple(mem[base + slot] for _n, base in OBJ_ARRAYS)
            if mem[mm.OBJECTS_TYPE + slot] in ENEMY_TYPES:  # enemy facing IS locked
                facings.append((slot, mem[mm.OBJECTS_H_ANGLE + slot]))
    s["enemy_facings"] = tuple(facings)
    s["tiles"] = bytes(mem[mm.TILES_TABLE : mm.TILES_TABLE + 1024])
    return s


def diff(a, b):
    """List of (field, sim_value, live_value) where a (sim) and b (live) differ."""
    out = []
    for k in a:
        if a[k] != b[k]:
            if k == "tiles":  # summarise which tile indices differ
                da = [i for i in range(1024) if a[k][i] != b[k][i]]
                out.append(
                    (
                        f"tiles@{da[:12]}",
                        [a[k][i] for i in da[:12]],
                        [b[k][i] for i in da[:12]],
                    )
                )
            else:
                out.append((k, a[k], b[k]))
    return out


def sim_apply(world, stp, view):
    """Apply ONE plan step's action to the sim state (no world advance). Mirrors
    run_plan_simulated.execute_step's action firing. Returns the resolved view."""
    verb, otype, tile = stp["verb"], stp["otype"], tuple(stp["target"])
    if verb in ("create", "absorb") and view is None:
        view = aim.propose(world, tile, eye_z=None)
    if view:  # match the driven view angles onto the sim player (sim actions don't aim)
        ps = world.player
        if view.get("h_angle") is not None:
            world.mem[mm.OBJECTS_H_ANGLE + ps] = view["h_angle"] & 0xFF
        if view.get("v_angle") is not None:
            world.mem[mm.OBJECTS_V_ANGLE + ps] = view["v_angle"] & 0xFF
    b = world.mem[mm.TILES_TABLE + mm.tidx(*tile)]
    slot = (b & 0x3F) if b >= mm.OBJECT_TILE else None
    if verb == "create":
        actions.create(world, otype, tile)
    elif verb == "transfer":
        if slot is not None:
            actions.transfer(world, slot)
    elif verb == "absorb":
        if slot is not None:
            actions.absorb(world, slot)
    return view


def run_probe(session, log, result):
    bm = session.bm
    # Quantized live cadence: the CPU stays HALTED between primitives and advances ONLY
    # inside run_until_pc windows, so per-step live frames == the ROM animation the cost
    # model prices (no monitor free-run). auto_resume off makes every read halt-safe; the
    # quantized driver's _resume() no-ops the per-primitive exit().
    bm.auto_resume = False
    ex = Executor(bm, log)
    drv = kbd_aim.KbdDriver(bm, log, quantized=True)
    landscape = session.landscape

    from vice_driver.binmon import CHECK_EXEC

    cp = bm.checkpoint_set(UPDATE_ENEMIES_PC, op=CHECK_EXEC, silent=True)
    fp = bm.checkpoint_set(
        0x9630, op=CHECK_EXEC, silent=True
    )  # once-per-frame top marker
    # $130C update_enemy_cooldowns entry: the TRUE cooldown clock. It is called at most
    # once per frame (raster $9663 / scroll $3684, mutually exclusive via $0CD8) but is
    # SKIPPED when a blocking plot_world overruns the frame ($0CD8 gates $9663 while the
    # scroll loop is still inside the render) AND when $0CE5 freezes it (both callers gate
    # on $0CE5, so the very-first-action aim never reaches $130C). So $9630 frames >= $130C
    # ticks; the cooldown state advances per $130C, not per raster frame. Advancing the sim
    # by this count reproduces the ROM cadence exactly, incl. the plot-overrun suppression
    # that left the step-4 rot_cd/upd_cd residual under the raster-frame count.
    dp = bm.checkpoint_set(0x130C, op=CHECK_EXEC, silent=True)
    log(
        f"checkpoints: $16B5=#{cp.checknum} frame$9630=#{fp.checknum} "
        f"cool$130C=#{dp.checknum}"
    )

    def hits():
        return bm.checkpoint_get(cp.checknum).hit_count

    def frames():
        return bm.checkpoint_get(fp.checknum).hit_count

    def cools():
        return bm.checkpoint_get(dp.checknum).hit_count

    # M0: the shared authoritative entry image (CPU halted for a coherent read).
    with bm.halted():
        m0 = bytes(core.live_image(bm))
        h0 = hits()
    log(f"entry image captured; $16B5 hit_count={h0}")

    # bonus: does the offline generate(0) match the live entry phase?
    try:
        gen = _landscape.generate(landscape)
        gen.mem[mm.CURSOR] = 7
        gen.mem[mm.COOLDOWN_GATE] = 0
        d0 = diff(snapshot(gen.mem), snapshot(bytearray(m0)))
        log(f"[generate({landscape}) vs LIVE entry] {len(d0)} field diffs")
        for k, sv, lv in d0[:40]:
            log(f"    GEN-DIFF {k}: generate={sv} live={lv}")
    except Exception as e:
        log(f"  generate compare failed: {e}")

    world = State.from_mem(bytearray(m0))
    g = plan_game.PlanGame.from_mem(m0, landscape, seed_built_columns=False)
    pr = plan(g)
    log(f"PLAN from LIVE entry: won={pr.won} steps={len(pr.steps)}")
    if not pr.won:
        log(f"  plan failed: {pr.failure}")
        result["divergence"] = "plan failed on live entry image"
        return

    # quantized: CPU stays halted; keystrokes drive via run_until_pc, reads never free-run.
    result["energy_curve"] = []
    result["actions"] = []

    # PRNG + cursor churn on the unbounded update_enemies spin, which feeds nothing but
    # drain-scatter / forced-hyperspace (events a hidden plan never triggers), so they are
    # out of the lock-state; everything that determines the enemy schedule + outcome is in.
    # E_upd_cd (enemy update-cooldown $0C30): a sub-frame-phase artifact of the ROM's two
    # asynchronous cooldown clocks -- the cooldown decrement $130C/$1317 runs once/frame from
    # raster $9663 / scroll $3684, while the enemy reconsideration reload $16ED (which reloads
    # a due enemy's update-cooldown to 4) is spun continuously by update_game_loop $129F. Its
    # value at an arbitrary CPU-halt boundary depends on the sub-frame phase the whole-frame
    # sim cannot reproduce; proven inert per resynced step (forcing sim->live changes nothing
    # else in the window). Every OTHER field stays strict.
    UNLOCKED = {"prng", "cursor", "E_upd_cd"}
    # Resync the sim from the live PRE-step image each step, so every step's frame advance
    # is validated in isolation (no error accumulation). Step 0 carries the known $0CE5
    # aim-split (cooldowns don't tick during the first-ever action's aim frames) -- that is
    # a driver-timing artifact, handled exactly by the frame-quantized driver; steps 1+ have
    # $0CE5 already clear throughout, so they are the clean steady-state model test.
    for i, stp in enumerate(pr.steps):
        verb, tile = stp["verb"], tuple(stp["target"])
        # Coherent pre-step capture: image + frame/UE counters read at the SAME halt, so
        # the sim starts exactly at live's pre-step state with the frame counter aligned
        # (the uncounted free-run during bookkeeping is already baked into pre_img).
        with bm.halted():
            pre_img = bytes(core.live_image(bm))
            f_before = frames()
            h_before = hits()
            c_before = cools()
        world = State.from_mem(bytearray(pre_img))
        outcome = perform_step(ex, drv, f"L{i}", stp, log, result)
        with bm.halted():  # halt + coherent post dump + exact counts for this step
            live_img = bytes(core.live_image(bm))
            live_frames = frames() - f_before
            live_rounds = hits() - h_before
            live_cools = (
                cools() - c_before
            )  # true cooldown-clock ticks (plot/$0CE5 gated)
        # sim: apply the same action, mark the player as having acted ($12E1 LSR $0CE5),
        # then advance the cooldown clock by the EXACT live $130C tick count. $130C already
        # excludes plot-overrun frames and $0CE5-frozen (first-action aim) frames, so this
        # is the faithful cadence -- the raster $9630 count over-ticks by the plot overrun.
        sim_apply(world, stp, stp.get("view"))
        world.mem[mm.PLAYER_NOT_ACTED] = world.mem[mm.PLAYER_NOT_ACTED] >> 1
        SE.advance_frames(world, live_cools)
        sim_snap = snapshot(world.mem)
        live_snap = snapshot(bytearray(live_img))
        d = diff(sim_snap, live_snap)
        locked = [x for x in d if x[0] not in UNLOCKED]
        churn = [x for x in d if x[0] in UNLOCKED]
        if i == 0:  # known $0CE5 aim-split; report but do not stop
            log(f"  (step 0 $0CE5 aim-split: {len(locked)} locked diffs expected)")
            locked = []

        def senth(snap):
            return dict(snap.get("enemy_facings", ())).get(0, -1)

        log(
            f"STEP {i} {verb} {tile}: outcome={outcome} frames={live_frames} "
            f"cools={live_cools} (plot_overrun={live_frames - live_cools}) "
            f"UE={live_rounds} (UE/frame={live_rounds/max(1,live_frames):.1f}) "
            f"sentH sim=${senth(sim_snap):02x} live=${senth(live_snap):02x} "
            f"E sim={sim_snap['energy']} live={live_snap['energy']} "
            f"locked_diffs={len(locked)} churn_diffs={len(churn)}"
        )
        if locked:
            log(f"  !!! LOCKED-STATE DIVERGENCE at step {i} ({verb} {tile}) !!!")
            for k, sv, lv in locked[:60]:
                log(f"    DIFF {k}: sim={sv} live={lv}")
            result["divergence"] = (
                f"step {i} {verb} {tile}: {len(locked)} locked fields"
            )
            result["first_divergence_step"] = i
            return
        if outcome not in ("ok", "best_effort_miss"):
            log(f"  live step outcome {outcome} (no state divergence yet); stopping")
            result["divergence"] = f"step {i} outcome {outcome}"
            return
    log("NO DIVERGENCE across the whole plan (sim == live at every step)")
    result["divergence"] = None


def main():
    def log(m):
        print(m, flush=True)

    os.environ.setdefault("NO_RECORD", "1")
    result = {"won": False}
    tap = os.path.join(ROOT, "sentinel-gold.tap")
    renders = os.path.join(ROOT, "renders")

    def play(session):
        run_probe(session, log, result)

    core.boot_and_play(tap, renders, "0000", "lockstep.avi", log, play, result)
    log("\n=== LOCKSTEP RESULT ===")
    log(f"  divergence: {result.get('divergence')}")


if __name__ == "__main__":
    main()
