#!/usr/bin/env python3
"""Run the solver on the simulator with enemies enabled and TICK-ACCURATE.

The offline planner (``solver.climb_search.plan_search``) wins ls0, but only
because it never advances the *real* enemy state between the moves it commits: its
lookahead steps enemies inside throwaway clones purely as a rotation forecast and
restores the energy afterwards (``_advance_enemies(apply_drain=False)``), and the
committed climb never ticks the world at all. The live game does the opposite --
the emulator runs continuously, so every second the player spends aiming a move the
enemies rotate, drain -1/tick when they see the player, and downgrade
(robot->boulder->tree) any object they see. That is the divergence this script
exposes WITHOUT the emulator.

It mirrors ``run_plan_live.execute_live``'s closed loop, but the "real game" is the
bit-exact ``sentinel`` simulator instead of VICE:

  * ``world`` is the authoritative :class:`sentinel.state.State`. It is the source
    of truth the planner resyncs from each iteration and the thing that actually
    advances tick by tick.
  * each iteration builds a throwaway planning snapshot from ``world`` and asks the
    solver (``climb_search.search_iterate``) for the next decision, exactly as the
    live loop does;
  * the decision's steps (create/absorb/transfer) are then EXECUTED against
    ``world`` one at a time, and between/around each the world is advanced by the
    real number of enemy rounds that action costs -- derived from the SAME
    keyboard-aim geometry the solver's move-cost model uses
    (``climb_search._pan_rounds`` + ``ROUNDS_PER_ACTION``) -- with drain and
    downgrades APPLIED (``sentinel.enemies.step``). So a foothold the planner
    assumed it could build, but which real timing/enemies made infeasible (tile's
    boulder downgraded out from under it, energy drained below the build cost),
    fails here just as it does live, and the loop resyncs + replans around it.

It then reports whether the player ends up on the platform (``actions.won``).

Not modelled: the meanie forced-hyperspace side channel (``sentinel.enemies.step``
covers drain/rotation/downgrade but not meanie spawning -- see
``sentinel.threat.meanie_safe``). So a WIN here is necessary-but-not-sufficient for
a live win; a LOSS is a genuine, tick-accurate solver failure.
"""

import os, sys, json, time, argparse

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.abspath(os.path.join(HERE, ".."))
sys.path.insert(0, ROOT)

from sentinel import landscape as _landscape
from sentinel import actions, enemies as SE
from sentinel import memmap as mm
from sentinel import actioncost
from solver import plan_game
from solver import climb_search as csearch

ENERGY = mm.ENERGY_IN_OBJECTS
ROUNDS_PER_ACTION = csearch.ROUNDS_PER_ACTION
T_SENTINEL = mm.T_SENTINEL


def make_world(landscape):
    """The authoritative tick-accurate game state, built exactly as
    ``plan_game.PlanGame.__init__`` seeds a fresh landscape (cursor 7, cooldown
    gate 0) so enemy stepping starts from the canonical initial phase."""
    world = _landscape.generate(landscape)
    world.mem[mm.CURSOR] = 7
    world.mem[mm.COOLDOWN_GATE] = 0
    return world


def advance(world, rounds, budget):
    """Advance the world `rounds` enemy rounds, bit-exact, with drain/downgrades
    APPLIED (this is the real game running, not a forecast). Accumulates into the
    tick budget for reporting."""
    n = int(round(rounds))
    for _ in range(n):
        SE.step(world)
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
    decision's root-heading read and pan-cost chaining faithful."""
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
    action settles over another ~ROUNDS_PER_ACTION rounds. Returns one of:
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
    # against CURRENT world memory, exactly as the live runner does before firing.
    if verb in ("create", "absorb") and view is None:
        ps = world.player
        eye_z = world.mem[mm.OBJECTS_Z_HEIGHT + ps]
        view = plan_game.centre_view_for(bytes(world.mem), tile, ps, eye_z)

    # --- aim: the world runs while the view pans to the target bearing ---
    if view:
        advance(world, csearch._pan_rounds(heading[0], heading[1], view), budget)
        heading[0] = view.get("h_angle", heading[0])
        heading[1] = view.get("v_angle", heading[1])
        set_facing(world, view)

    # Per-action settle: the ROM-cadence cost the enemies advance AFTER the pan,
    # while the action animates and the live driver reads back the result. Priced
    # from sentinel.actioncost (dither/tune floor + scene redraw), computed against
    # the world at fire time so a dense view costs more -- replacing the old flat
    # ROUNDS_PER_ACTION, which under-counted by ~15x and let the sim survive drains
    # the live run does not.
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


def run(landscape, depth, beam, toward_plat, near_plat_radius, max_iterations, log):
    world = make_world(landscape)
    budget = {"ticks": 0}
    result = {
        "landscape": landscape,
        "won": False,
        "decisions": 0,
        "divergence": None,
        "energy_curve": [],
    }

    # planning snapshot resynced from the world each iteration (the ground truth,
    # including any drain/downgrade the ticks applied). seed_built_columns stays
    # False until a create has actually landed -- same rule as the live loop.
    def resync(seed_built):
        return plan_game.Game.from_mem(
            bytes(world.mem), landscape, seed_built_columns=seed_built
        )

    built_anything = False
    g = resync(False)
    ctx = csearch.climb_ctx(g, toward_plat, near_plat_radius)
    heading = [
        world.mem[mm.OBJECTS_H_ANGLE + world.player],
        world.mem[mm.OBJECTS_V_ANGLE + world.player],
    ]
    log(
        f"SIM climb start ls{landscape}: {g.player_xy()} eye {g.eye} "
        f"plat {ctx['plat']} plat_ground {ctx['plat_ground']} energy {g.energy}"
    )

    blocked = set()
    status = "retry"
    for it in range(max_iterations):
        g = resync(built_anything)
        # seed the planner's root heading from the world's current facing.
        world.mem[mm.OBJECTS_H_ANGLE + world.player] = heading[0]
        world.mem[mm.OBJECTS_V_ANGLE + world.player] = heading[1]
        g.mem[mm.OBJECTS_H_ANGLE + g.player] = heading[0]
        g.mem[mm.OBJECTS_V_ANGLE + g.player] = heading[1]
        before_n = len(g.steps)
        status = csearch.search_iterate(g, ctx, blocked, log, depth=depth, beam=beam)
        new_steps = g.steps[before_n:]

        blocked_this_round = False
        built_tiles = set()
        for stp in new_steps:
            tile = tuple(stp["target"])
            result["energy_curve"].append(world.energy)
            outcome = execute_step(world, stp, heading, budget, log)
            if outcome == "ok":
                if stp["verb"] == "create":
                    built_tiles.add(tile)
                    built_anything = True
                continue
            if outcome == "miss":
                log(f"  [{it}] best-effort absorb {tile} missed (drained/downgraded)")
                continue
            if outcome == "drained":
                log(
                    f"  [{it}] create {tile} DRAINED (energy {world.energy} < "
                    f"cost {ENERGY[stp['otype']]}); resync + replan"
                )
                blocked_this_round = True
                break
            if outcome == "create":
                if tile in built_tiles:
                    log(
                        f"  [{it}] on-boulder synthoid {tile} failed but boulder landed; "
                        "not blocking (hop retry picks it up)"
                    )
                else:
                    blocked.add((tile, stp["otype"] == 3))
                    log(f"  [{it}] create {tile} infeasible; blocking foothold")
                blocked_this_round = True
                break
            # transfer / sentinel absorb failure mid-climb: resync + replan.
            log(f"  [{it}] {stp['verb']} {tile} -> {outcome}; resync + replan")
            blocked_this_round = True
            break

        result["decisions"] += 1
        if blocked_this_round:
            continue
        if status in ("no_gain", "stuck"):
            log(f"  SIM climb stopped ({status}); attempting endgame")
            break
        if status == "approach":
            log("  SIM reached win-launch state; attempting endgame")
            break
    else:
        log(f"SIM climb hit max_iterations ({max_iterations}) without approach")

    # --- endgame: absorb Sentinel + platform synthoid + transfer, tick-accurate ---
    g = resync(built_anything)
    world.mem[mm.OBJECTS_H_ANGLE + world.player] = heading[0]
    world.mem[mm.OBJECTS_V_ANGLE + world.player] = heading[1]
    g.mem[mm.OBJECTS_H_ANGLE + g.player] = heading[0]
    g.mem[mm.OBJECTS_V_ANGLE + g.player] = heading[1]
    before_n = len(g.steps)
    endgame_planned = csearch.endgame(g, ctx["plat"], log)
    for stp in g.steps[before_n:]:
        outcome = execute_step(world, stp, heading, budget, log)
        if outcome not in ("ok", "miss"):
            result["divergence"] = (
                f"endgame {stp['verb']} {tuple(stp['target'])} -> {outcome}"
            )
            log(f"  SIM endgame step {stp['verb']} {tuple(stp['target'])}: {outcome}")
            break
    if not endgame_planned and result["divergence"] is None:
        gfinal = resync(built_anything)
        result["divergence"] = (
            f"endgame not reachable from final climb state "
            f"{gfinal.player_xy()} eye {gfinal.eye}"
        )

    # --- the arbiter: is the player on the platform? (actions.won == on_platform) ---
    result["won"] = actions.won(world)
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
    ap.add_argument(
        "--depth", type=int, default=int(os.environ.get("SEARCH_DEPTH", "2"))
    )
    ap.add_argument("--beam", type=int, default=int(os.environ.get("SEARCH_BEAM", "2")))
    ap.add_argument("--near-plat-radius", type=int, default=2)
    ap.add_argument("--max-iterations", type=int, default=200)
    ap.add_argument("--no-toward-plat", action="store_true")
    ap.add_argument("--quiet", action="store_true")
    args = ap.parse_args()

    def log(m):
        if not args.quiet:
            print(m, flush=True)

    t0 = time.time()
    result = run(
        args.landscape,
        depth=args.depth,
        beam=args.beam,
        toward_plat=not args.no_toward_plat,
        near_plat_radius=args.near_plat_radius,
        max_iterations=args.max_iterations,
        log=log,
    )
    result["wall_seconds"] = round(time.time() - t0, 1)

    print("\n=== SIMULATED RUN RESULT ===")
    print(f"  landscape        : {result['landscape']}")
    print(f"  WON (on platform): {'YES' if result['won'] else 'NO'}")
    print(f"  decisions        : {result['decisions']}")
    print(f"  final player     : {result['final_player']} eye {result['final_eye']}")
    print(f"  final energy      : {result['final_energy']}")
    print(f"  enemy ticks       : {result['total_ticks']} (game rounds elapsed)")
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
