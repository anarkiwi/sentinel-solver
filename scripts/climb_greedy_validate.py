#!/usr/bin/env python3
"""Validate a native greedy climb plan against the REAL ROM and REPAIR it by
blocklist-replan: plan natively (fast), replay through code_engine ($0CDE gate),
and if a build is occluded / gate-rejected, blocklist that foothold tile and replan.
A handful of native plans + one ROM replay each -- NO per-move emulation in the loop.
"""

import sys, os, json, time

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
os.chdir(os.path.join(os.path.dirname(os.path.abspath(__file__)), ".."))
import climb_greedy, code_engine, native_game, aim_invert
import validate_kbd_plan as VKP
from game_state import OBJECTS_Z_HEIGHT, PLAYER_ENERGY

# A6: enemy ticks stepped after each replayed action (~ live aim wall-time) so the
# validator sees the rotation/drain the live keyboard run is exposed to.
ENEMY_STRESS = 100


def _replay_rom(steps, landscape, verbose=False, poke_energy=None):
    """Replay greedy steps through the real ROM using NATIVE views only (no emulated
    search). Returns (won, fail_tile or None). On the first rejected build, fail_tile
    is that step's target so the caller can blocklist it.

    poke_energy: if None (the default now), KEEP the ROM's REAL generated player energy
    so the replay genuinely enforces the energy budget -- a faithful energy-10 result.
    (Previously this was hard-poked to 0x3F=63, which masked any energy infeasibility.)
    """
    eng = code_engine.CodeEngine(landscape)
    if poke_energy is not None:
        eng.mem[PLAYER_ENERGY] = poke_energy
    log = print if verbose else (lambda *a: None)
    for i, st in enumerate(steps):
        verb, tile, otype = st["verb"], tuple(st["target"]), st["otype"]
        ps = eng.player_slot
        ez = eng.mem[OBJECTS_Z_HEIGHT + ps]
        if verb == "create":
            # Compute the keyboard view by INVERTING the aim ($1C10): closed-form bearing
            # (H=atan2(dx,dy)) + seeded 1-D pitch solve against the real marched tile, plus a
            # centre refine for occupied (boulder/platform) tiles ($1E48 <$40).  The view is
            # on the keyboard lattice (h%8==0, v%4==1 in the pan band) and gate-accepted, so it
            # is faithfully keyboard-reproducible -- no poked angles, no cursor grid search.
            view = aim_invert.solve_view(eng, tile, otype)
            if view is None:
                log(f"  [{i}] create {otype}->{tile} REJECTED (no keyboard aim view)")
                return False, tile
            if not eng.create_via_gate(otype, tile, view).get("ok"):
                log(
                    f"  [{i}] create {otype}->{tile} REJECTED (view computed but gate refused)"
                )
                return False, tile
            st["view"] = view  # persist the gate-accepted view
        elif verb == "transfer":
            slot = VKP._slot_on_tile(eng, tile, want_type=0)
            if slot is None:
                return False, tile
            eng.transfer(slot)
        elif verb == "absorb":
            view = native_game.centre_view_for(eng.mem, tile, ps, ez)
            if view is not None and VKP._absorb_via_aim(eng, view).get("ok"):
                st["view"] = view  # A4: persist the aimed view
            elif otype == 5:
                # absorbs are best-effort (energy recovery); only the Sentinel (otype 5)
                # absorb is fatal when it can't be aimed.
                return False, tile
        # A6: age the enemies by the live aim wall-time so rotation/drain is exercised.
        eng.step_enemies(ENEMY_STRESS)
        st["plan_energy"] = int(
            eng.player_energy
        )  # D7: expected energy for the recorder
    if not eng.won():
        eng.mem[0x0C61] = 0x22
        eng.mem[0x006E] = eng.player_slot
        eng._call(0x1B18)
    return eng.won(), None


def solve(landscape, max_iters=25, verbose=True, toward_plat=False, near_plat_radius=0):
    t0 = time.time()
    log = lambda *a: verbose and print(*a)
    blocked = set()
    for it in range(max_iters):
        g = climb_greedy.plan_greedy(
            landscape,
            verbose=False,
            blocked=frozenset(blocked),
            toward_plat=toward_plat,
            near_plat_radius=near_plat_radius,
        )
        won_native = getattr(g, "native_won", False)
        if not won_native:
            # the native plan didn't even reach the platform with this blocklist -- the
            # ROM replay (slow) can't win either; the last blocked tile pruned the route.
            log(
                f"  iter {it}: native_won=False steps={len(g.steps)} (skip ROM replay) "
                f"blocked={len(blocked)}; native route lost -- stop"
            )
            break
        won, fail = _replay_rom(g.steps, landscape)
        log(
            f"  iter {it}: native_won={won_native} steps={len(g.steps)} "
            f"ROM_won={won} fail_tile={fail} blocked={len(blocked)}"
        )
        if won:
            log(
                f"=== ls{landscape} ROM-VALID WIN in {time.time()-t0:.1f}s, "
                f"{len(g.steps)} steps, {it+1} iters ==="
            )
            return True, g.steps
        if fail is None:
            log("  ROM replay completed all steps but no win flag; stop")
            break
        # blocklist the failing foothold tile (both hop & boulder forms) and replan.
        blocked.add((tuple(fail), True))
        blocked.add((tuple(fail), False))
    log(
        f"=== ls{landscape} NO ROM-valid plan after {max_iters} iters "
        f"({time.time()-t0:.1f}s) ==="
    )
    return False, None


if __name__ == "__main__":
    ls = int(sys.argv[1]) if len(sys.argv) > 1 else 66
    won, steps = solve(ls)
    if steps is not None:
        json.dump(
            {"landscape": ls, "won": won, "steps": steps},
            open(f"out/kbd_greedy_{ls:04d}.json", "w"),
            indent=0,
        )
        print(f"wrote out/kbd_greedy_{ls:04d}.json")
    sys.exit(0 if won else 1)
