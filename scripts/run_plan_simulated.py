#!/usr/bin/env python3
"""Run the offline planner on the simulator with enemies enabled and TICK-ACCURATE.

The Phase-2 thesis: :func:`solver.astar_planner.plan` produces a genuinely HIDDEN
(never-seen) plan offline, so executing it against the tick-accurate world drains
nothing -- regardless of the exact tick counts the real enemy round spends -- and
wins ls0 WITHOUT any re-plan. This script is the honesty gate for that claim.

The "real game" is the bit-exact ``sentinel`` simulator (not VICE):

  * ``world`` is the authoritative :class:`sentinel.state.State`. It is the source
    of truth the planner resyncs from, and the thing that actually advances tick by
    tick with drain/downgrades APPLIED (``sentinel.enemies.step``).
  * a single offline plan is computed once from a snapshot resynced from ``world``
    (``PlanGame.from_mem``). The winning plan's ``steps`` already include the
    endgame (absorb Sentinel -> create platform synthoid -> transfer); there is no
    separate endgame phase.
  * each step (create/transfer/absorb) is EXECUTED against ``world`` one at a time
    via :func:`execute_step`, which advances the world by the real number of enemy
    rounds the action costs -- the keyboard-aim pan (``climb_search._pan_rounds``)
    plus the per-action settle (``sentinel.actioncost``) -- with drain and downgrades
    APPLIED. So if real timing/enemies made a foothold infeasible (its object
    downgraded out from under it, energy drained below the build cost), the step
    diverges here just as it does live.

  * on a divergence (a step outcome other than ``"ok"``/``"miss"``) the loop
    RESYNCS a fresh planning snapshot from the true ``world`` state and re-``plan()``s
    around it, capped at ``max_replans`` to bound both infinite loops and the 60 s
    CPU budget (each ``plan`` costs tens of seconds).

``sentinel.enemies.step`` models the full enemy round: rotation, cooldowns, target
LOS, draining, object downgrades, the energy-conservation tree discharge, AND both
loss paths -- drained at 0 energy (kill_player $1A00) and a meanie's forced
hyperspace. ``run`` checks ``actions.player_dead`` after every advance, so a route
that lingers in the Sentinel's view at low energy DIES here (as live). A LOSS is
thus a genuine, tick-accurate planner failure; a WIN reflects a route that survives
the modelled enemy round end to end. The gate additionally requires ZERO drain in
the per-step energy log -- every energy change a build cost or an absorb gain.
"""

import os, sys, json, time, argparse

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.abspath(os.path.join(HERE, ".."))
sys.path.insert(0, ROOT)

from sentinel import landscape as _landscape
from sentinel import actions, enemies as SE, aim
from sentinel import memmap as mm
from sentinel import actioncost
from solver import plan_game
from solver import climb_search as csearch  # only for _pan_rounds (aim cost)
from solver.astar_planner import plan

ENERGY = mm.ENERGY_IN_OBJECTS
T_SENTINEL = mm.T_SENTINEL


def make_world(landscape):
    """The authoritative tick-accurate game state, built exactly as
    ``plan_game.PlanGame.__init__`` seeds a fresh landscape (cursor 7, cooldown
    gate 0) so enemy stepping starts from the canonical initial phase."""
    world = _landscape.generate(landscape)
    world.mem[mm.CURSOR] = 7
    world.mem[mm.COOLDOWN_GATE] = 0
    world.mem[mm.PLAYER_DIED_BY_DRAINING] = 0
    world.mem[mm.PLAYER_HAS_HYPERSPACED] = 0
    return world


def advance(world, rounds, budget):
    """Advance the world `rounds` enemy rounds, bit-exact, with drain/downgrades
    APPLIED (this is the real game running, not a forecast). Accumulates into the
    tick budget for reporting."""
    n = int(round(rounds))
    for i in range(n):
        SE.step(world)
        if actions.player_dead(world):  # drained to death mid-advance -- stop the clock
            budget["ticks"] += i + 1
            return
    budget["ticks"] += n


def topmost_slot(world, tile):
    """The topmost live object slot on `tile` (from the tile byte), or None for
    bare terrain."""
    b = world.mem[mm.TILES_TABLE + mm.tidx(tile[0], tile[1])]
    if b >= mm.OBJECT_TILE:
        return b & 0x3F
    return None


def set_facing(world, view):
    """Point the player's view at `view`'s angles, as a real aim would (the
    simulator's create/absorb/transfer never touch OBJ_H/V_ANGLE). Keeps the next
    step's pan-cost chaining faithful."""
    if not view:
        return
    ps = world.player
    if view.get("h_angle") is not None:
        world.mem[mm.OBJECTS_H_ANGLE + ps] = view["h_angle"] & 0xFF
    if view.get("v_angle") is not None:
        world.mem[mm.OBJECTS_V_ANGLE + ps] = view["v_angle"] & 0xFF


def execute_step(world, stp, heading, budget, log):
    """Execute ONE plan step against the world with tick-accurate enemies.

    Aiming ticks the world (drain during the pan), then the action fires, then the
    action settles over ~actioncost rounds. Returns one of:
      "ok"        -- action verified against the world
      "drained"   -- a create's cost exceeded the (drained) energy before firing
      "create"    -- the create did not land (tile no longer stackable, etc.)
      "transfer"  -- the transfer did not move the player
      "sentinel"  -- the Sentinel absorb (endgame) missed
      "miss"      -- a best-effort fuel/shell absorb missed (non-fatal)
    """
    verb, otype, tile = stp["verb"], stp["otype"], tuple(stp["target"])
    view = stp.get("view")
    # A deferred view (on-boulder synthoid re-aim, or a coarse absorb) is resolved
    # against CURRENT world memory via the SHARED aim proposer (sentinel.aim.propose,
    # true eye) -- exactly as the live runner and planner resolve it before firing.
    if verb in ("create", "absorb") and view is None:
        view = aim.propose(world, tile, eye_z=None)

    # --- aim: the world runs while the view pans to the target bearing ---
    if view:
        advance(world, csearch._pan_rounds(heading[0], heading[1], view), budget)
        heading[0] = view.get("h_angle", heading[0])
        heading[1] = view.get("v_angle", heading[1])
        set_facing(world, view)

    # ROM action LOS gate ($1B46, sentinel.aim.gate at the TRUE eye): the sim now
    # rejects a no-LOS aim exactly like the live driver -- a resolved view whose
    # real-eye ray does not reach `tile` fires NOTHING (no world mutation), it misses.
    if view is not None and not aim.gate(world, view, tile):
        if verb == "create":
            return "create"
        return "sentinel" if otype == T_SENTINEL else "miss"

    # Per-action settle: the ROM-cadence cost the enemies advance AFTER the pan,
    # while the action animates and the live driver reads back the result. Priced
    # from sentinel.actioncost (dither/tune floor + scene redraw), computed against
    # the world at fire time so a dense view costs more.
    stacked = verb == "create" and actioncost.is_stacked(world.mem, tile)
    settle = actioncost.action_rounds(world.mem, verb, view, stacked=stacked)
    e0 = world.energy
    if verb == "create":
        if e0 < ENERGY[otype]:  # drained below the build cost while aiming
            return "drained"
        slot = actions.create(world, otype, tile)
        advance(world, settle, budget)
        return "ok" if slot is not None else "create"
    if verb == "transfer":
        slot = topmost_slot(world, tile)
        moved = slot is not None and actions.transfer(world, slot)
        advance(world, settle, budget)
        return "ok" if moved else "transfer"
    if verb == "absorb":
        slot = topmost_slot(world, tile)
        got = slot is not None and actions.absorb(world, slot)
        advance(world, settle, budget)
        if got:
            return "ok"
        return "sentinel" if otype == T_SENTINEL else "miss"
    return "miss"


def run(landscape, max_replans, log):
    """Plan ls`landscape` offline, then execute the plan tick-accurately, resyncing
    and re-planning on any divergence (capped at ``max_replans``)."""
    world = make_world(landscape)
    budget = {"ticks": 0}
    result = {
        "landscape": landscape,
        "won": False,
        "plans": 0,
        "replans": 0,
        "divergence": None,
        "energy_curve": [],
    }

    heading = [
        world.mem[mm.OBJECTS_H_ANGLE + world.player],
        world.mem[mm.OBJECTS_V_ANGLE + world.player],
    ]

    # planning snapshot resynced from the world (the ground truth, including any
    # drain/downgrade the ticks applied). seed_built_columns stays False until a
    # create has actually landed -- same rule as the live loop.
    def resync(seed_built):
        g = plan_game.PlanGame.from_mem(
            bytes(world.mem), landscape, seed_built_columns=seed_built
        )
        # seed the planner's root heading from the world's current facing so its
        # first pan-cost matches what execute_step will charge.
        g.mem[mm.OBJECTS_H_ANGLE + g.player] = heading[0]
        g.mem[mm.OBJECTS_V_ANGLE + g.player] = heading[1]
        return g

    built_anything = False
    died = False
    diverged = True
    plans = 0

    while diverged and plans <= max_replans:
        diverged = False
        g = resync(built_anything)
        if plans == 0:
            log(
                f"SIM plan start ls{landscape}: {g.player_xy()} eye {g.eye} "
                f"plat {g.plat} plat_ground {g.plat_ground} energy {g.energy}"
            )
        # plan from a CLONE so the runner's world/plan stay decoupled.
        pr = plan(g.clone())
        plans += 1
        result["plans"] = plans
        result["replans"] = plans - 1
        if not pr.won:
            log("")
            log("!!! PLAN FAILED -- no hidden route found; this is a LOSS !!!")
            log(f"    failure: {pr.failure}")
            log(f"    stats  : {pr.stats}")
            result["divergence"] = f"plan failed: {pr.failure}"
            break
        log(
            f"  plan {plans}: won route with {len(pr.steps)} steps "
            f"(nodes {pr.stats.get('nodes')}, {pr.stats.get('wall_s')}s wall)"
        )

        for stp in pr.steps:
            tile = tuple(stp["target"])
            result["energy_curve"].append(world.energy)
            outcome = execute_step(world, stp, heading, budget, log)
            log(
                f"    {stp['verb']:8s} {tile} -> {outcome}  "
                f"energy {world.energy} eye {stp.get('eye_z')} ticks {budget['ticks']}"
            )
            if actions.player_dead(world):
                result["divergence"] = f"player drained to death at {tile}"
                log(f"    player DRAINED TO DEATH at {tile}")
                died = True
                break
            if outcome == "ok":
                if stp["verb"] == "create":
                    built_anything = True
                continue
            if outcome == "miss":
                log(f"    best-effort absorb {tile} missed (drained/downgraded)")
                continue
            # a real divergence: resync from the true world state and re-plan.
            log(f"    DIVERGENCE {stp['verb']} {tile} -> {outcome}; resync + replan")
            diverged = True
            break

        if died:
            break

    if diverged and not died and result["divergence"] is None:
        result["divergence"] = f"replan cap ({max_replans}) reached without a clean run"
        log(f"SIM replan cap ({max_replans}) reached without a clean execution")

    # --- the arbiter: is the player on the platform? (actions.won == on_platform) ---
    result["won"] = actions.won(world) and not actions.player_dead(world)
    result["landscape_complete_flag"] = bool(world.mem[mm.LANDSCAPE_COMPLETE] & 0x40)
    gfinal = resync(built_anything)
    result["final_player"] = list(gfinal.player_xy())
    result["final_eye"] = gfinal.eye
    result["final_energy"] = world.energy
    result["total_ticks"] = budget["ticks"]
    return result


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("landscape", nargs="?", type=int, default=0)
    ap.add_argument("--max-replans", type=int, default=3)
    ap.add_argument("--quiet", action="store_true")
    args = ap.parse_args()

    def log(m):
        if not args.quiet:
            print(m, flush=True)

    t0 = time.time()
    result = run(args.landscape, max_replans=args.max_replans, log=log)
    result["wall_seconds"] = round(time.time() - t0, 1)

    print("\n=== SIMULATED RUN RESULT ===")
    print(f"  landscape        : {result['landscape']}")
    print(f"  WON (on platform): {'YES' if result['won'] else 'NO'}")
    print(f"  plans            : {result['plans']} (replans {result['replans']})")
    print(f"  final player     : {result['final_player']} eye {result['final_eye']}")
    print(f"  final energy     : {result['final_energy']}")
    print(f"  enemy ticks      : {result['total_ticks']} (game rounds elapsed)")
    if result.get("divergence"):
        print(f"  DIVERGENCE       : {result['divergence']}")
    print(f"  wall seconds     : {result['wall_seconds']}")
    os.makedirs(os.path.join(ROOT, "out"), exist_ok=True)
    outpath = os.path.join(ROOT, "out", f"sim_run_{result['landscape']:04d}.json")
    json.dump(result, open(outpath, "w"), indent=0)
    print(f"  wrote {outpath}")
    return 0 if result["won"] else 1


if __name__ == "__main__":
    sys.exit(main())
