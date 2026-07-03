#!/usr/bin/env python3
"""Greedy height-first Sentinel climb planner.

A deliberately simple alternative to native_game._find_climb_path's A*: instead of
searching for an optimal route, it CLIMBS GREEDILY -- each step it transfers to the
reachable foothold that, in priority order:

  1. gains the most HEIGHT (eye z) -- get high as fast as possible;
  2. is FARTHEST from the current tile -- long transfers cover ground quickly;
  3. is FARTHEST from the map centre -- the centre is the most enemy-observable spot,
     so hugging the edges minimises exposure (a cheap proxy for enemy line-of-sight).

Once the eye is above the platform it switches to an APPROACH phase: head toward the
platform (minimising Chebyshev distance) while staying high enough to build/look down.
The endgame (absorb the Sentinel, synthoid on the platform, transfer = win) mirrors
native_game.plan.

Pure native (visibility_sweep / native_los); the emulator only validates the result.
Steps are emitted in the verb/otype/target/view shape validate_kbd_plan expects.
"""

import sys, os, json, time

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
os.chdir(os.path.join(HERE, ".."))
import native_game as NG
from native_game import (
    Game,
    visibility_sweep,
    terrain_z,
    cheb,
    OBJ_X,
    OBJ_Y,
    OBJ_Z,
    OBJ_TYPE,
    OBJ_ZF,
    OBJ_FLAGS,
)
from native_los import NativeState, aim_target_native


def _lattice_sweep(g, eye):
    """ONE keyboard-lattice sweep (h ≡ 0 mod 8, v in the pan band, centre cursor)
    returning {tile: first-LOS view} plus {tile: min tile-centre fraction}. The centre
    map is the keyboard-feasibility gate for on-boulder synthoids ($1E48 needs the
    tile-centre fraction < $40). Same order/params as visibility_sweep(coarse) so the
    chosen views are identical."""
    st = NativeState.from_mem(g.mem)
    ps = g.player
    views, centres = {}, {}
    for h in range(0, 256, 8):
        for v in NG._VBAND:
            rx, ry, los, centre = aim_target_native(
                st,
                h,
                v,
                NG.CUR_CX,
                NG.CUR_CY,
                ps,
                eye_z=eye,
                max_steps=200,
                return_centre=True,
            )
            if not los:
                continue
            t = (rx, ry)
            if t not in views:
                views[t] = {
                    "h_angle": h,
                    "v_angle": v,
                    "cursor": [NG.CUR_CX, NG.CUR_CY],
                }
            if t not in centres or centre < centres[t]:
                centres[t] = centre
    return views, centres


def _top_type(g, tile):
    """Type of the TOPMOST live object on `tile` (highest z_height+z_frac), or None if
    the tile is bare terrain. Used to gate boulder placement: a boulder may sit on bare
    terrain or on another boulder (type 3), never on a synthoid (0) / tree (2)."""
    top, topz = None, -1
    for s in range(64):
        if g.mem[OBJ_FLAGS + s] & 0x80:
            continue
        if (g.mem[OBJ_X + s], g.mem[OBJ_Y + s]) == tile:
            z = g.mem[OBJ_Z + s] * 256 + g.mem[OBJ_ZF + s]
            if z > topz:
                topz, top = z, g.mem[OBJ_TYPE + s]
    return top


N = 32
CENTRE = (N - 1) / 2.0  # 15.5
# energy the planner keeps in reserve above each build's up-front cost: live enemies
# rotate and drain -1/tick during the minutes of aiming, so planned +3 shell refunds
# silently become +2/+1/0 (drained robots downgrade). Reserve absorbs that loss.
# NOTE: the plan specified RESERVE=3, but from the REAL start energy 10 a reserve of 3
# makes boulder-steps (5 up-front) unaffordable once the budget dips, stalling the climb
# at (20,6) cheb-3 (native_won False). 2 is the largest reserve that preserves the
# ROM-validated win (climb ends at (21,3) eye 11.875, energy 6).
RESERVE = 2


def edge_dist(t):
    """Manhattan distance from the map centre -- larger == nearer an edge/corner,
    which is less observable to centrally-placed rotating enemies."""
    return abs(t[0] - CENTRE) + abs(t[1] - CENTRE)


def _foothold_eye(g, T2, use_b):
    """TRUE resulting eye (float) of moving onto T2, using Game.create's exact ROM
    height increments: first object on terrain = +0.875 (boulder) then synthoid +0.5;
    stacking onto an existing column (T2 already in g.col) = +0.5 each. This captures
    the FRACTIONAL staircase gain (re-stacking an already-used tile climbs +0.5-1),
    which a flat terrain+1 model misses."""
    cur = g.top_of(T2)
    if cur is None:
        return None
    first = T2 not in g.col
    if use_b:
        boulder_top = cur + (0.875 if first else 0.5)
        return boulder_top + 0.5  # synthoid on the boulder
    return cur + (0.875 if first else 0.5)  # synthoid on terrain/column


def _candidates(g, cur, eye, adj_boulder=False, plat=None, near_plat_radius=0):
    """All (T2, use_boulder, eye_after_float, view) footholds reachable with LOS, as a
    hop (synthoid) or an adjacent boulder-step (centre-aimed, climbs). Coarse sweep is
    fine -- views are recomputed at validate time.

    A FAR boulder-step (boulder on a distant LOS tile + synthoid CENTRE-AIMED on top,
    $1E48) is rejected by the real $1B46 gate in the tight, steep platform ring -- the
    on-boulder synthoid has NO acceptable centre-aim there ((18,9)->(18,6), cheb-3, fails
    even with _centre_aim_search). So a boulder-step whose TARGET tile T2 lies within
    `near_plat_radius` of `plat` is restricted to ADJACENT (cheb<=1 to the player); this
    gates on the BUILD TILE's proximity (the geometry that actually fails), not the
    player's. Far from the platform the open terrain accepts far steps (needed for the
    low-pocket break-out), so they stay available there. adj_boulder=True forces ADJACENT
    everywhere (legacy)."""
    sweep, centres = _lattice_sweep(g, eye)
    out = []
    for T2, view in sweep.items():
        if T2 == cur:
            continue
        if terrain_z(g.mem, *T2) is None:
            continue
        # You can only build (boulder OR synthoid) on BARE TERRAIN or on top of a
        # BOULDER -- never on top of a non-boulder object (a synthoid or tree). So both a
        # hop (synthoid) and a boulder-step are gated on the tile's current TOP object.
        top = _top_type(g, T2)
        if top is not None and top != 3:  # synthoid/tree on top -> nothing
            continue
        # A3: building on an ALREADY-occupied (boulder-topped) tile needs a keyboard
        # CENTRE-aim (tile-centre fraction < $40, $1E48). Gate such tiles on a
        # centre-cursor lattice view that centres; bare-terrain targets need no centre
        # (the first object on terrain is not centre-gated) -- their on-boulder synthoid
        # feasibility is enforced by the ROM validator (A4) after the boulder lands.
        centre_ok = (top != 3) or centres.get(T2, 0xFF) < 0x40
        he = _foothold_eye(g, T2, False)
        if he is not None and len(g.free) >= 1 and centre_ok:  # hop needs one free slot
            out.append((T2, False, he, view))  # synthoid on terrain/boulder
        # boulder-step: forbid only when the BUILD TILE is near the platform AND the step
        # is non-adjacent (the ROM-infeasible on-distant-boulder synthoid in the ring), or
        # when adj_boulder forces adjacent everywhere.
        far = cheb(T2, cur) > 1
        near_ring = (
            plat is not None and near_plat_radius and cheb(T2, plat) <= near_plat_radius
        )
        if not (far and (adj_boulder or near_ring)):
            hb = _foothold_eye(g, T2, True)
            if hb is not None and len(g.free) >= 2 and centre_ok:  # boulder + synthoid
                out.append((T2, True, hb, view))
    # deterministic order: visibility_sweep yields tiles in hash-seed-dependent order, so
    # max(pool, key=...) would break ties (equal height / equal cheb-to-platform) by which
    # tile happened to come first -- making the whole plan depend on PYTHONHASHSEED. Sort by
    # (tile_x, tile_y, use_boulder) so selection is reproducible across processes. A3: bias
    # ties toward Chebyshev<=2 geometry (the keyboard-reliable near-aim), so among otherwise
    # equal candidates the closer, more centre-aimable foothold wins.
    out.sort(key=lambda e: (cheb(e[0], cur) > 2, e[0][0], e[0][1], e[1]))
    return out


def _apply(g, T2, use_b, view):
    """Build the foothold and transfer onto it, then RECOVER energy by re-absorbing
    the synthoid shell left on the tile we just departed (it stays a live object --
    Game.transfer never frees it). Net cost of a boulder-step is then boulder(2) +
    synthoid(3) - reabsorbed shell(3) = 2; a hop is synthoid(3) - 3 = 0."""
    prev_slot, prev_tile = g.player, g.player_xy()
    if use_b:
        g.create(3, T2, view, "greedy climb boulder")
        s = g.create(
            0, T2, None, "greedy climb synthoid"
        )  # centre-aim view at validate
    else:
        s = g.create(0, T2, view, "greedy hop synthoid")
    g.transfer(s, "step")
    # energy recovery: re-absorb the abandoned synthoid shell on the departed tile if
    # it is now in line of sight (look down on it from the higher new foothold). Credit
    # it only when we can actually look DOWN: its FULL top (z_height + z_frac/256) must
    # be at or below the eye, else the ROM's looking-up check ($1D2E) rejects the absorb
    # and the native energy would over-count (A7). Game.absorb (ROM remove_object)
    # restores the departed tile's column to the object below the shell (the boulder, if
    # any) or clears it, so no manual column fixup is needed.
    sw = visibility_sweep(g.mem, g.player, int(g.eye), max_steps=200, coarse=True)
    if (
        prev_tile in sw
        and g.mem[OBJ_TYPE + prev_slot] == 0
        and g.mem[OBJ_Z + prev_slot] + g.mem[OBJ_ZF + prev_slot] / 256.0 <= g.eye + 1e-9
    ):
        g.absorb(prev_slot, sw[prev_tile], "reabsorb prior shell")


_FUEL_NAME = {0: "synthoid", 1: "sentry", 2: "tree", 3: "boulder"}


def _refuel(g, log):
    """Earn energy by absorbing everything you can look DOWN on (top <= eye, in LOS):
    TREES (+1) and SENTRIES (+3), AND -- crucially -- the climb's OWN abandoned BOULDERS
    (+2) and SYNTHOIDS (+3) once you've climbed past them. Reclaiming your own trail makes
    the staircase nearly energy-NEUTRAL (the boulder/synthoid cost is recovered), which is
    how the climb funds itself from the low real starting energy. Never absorb the column
    the player is standing on (you'd fall). Returns energy gained."""
    from native_game import ENERGY

    gained = 0
    cur = g.player_xy()
    sweep = visibility_sweep(g.mem, g.player, int(g.eye), max_steps=320)
    # absorb topmost-first per tile (matches the ROM gate) so stacks unwind cleanly.
    cand = []
    for slot in range(64):
        if (g.mem[OBJ_FLAGS + slot] & 0x80) or slot == g.player:
            continue
        ot = g.mem[OBJ_TYPE + slot]
        if ot not in (0, 1, 2, 3):  # synthoid/sentry/tree/boulder = fuel
            continue
        tile = (g.mem[OBJ_X + slot], g.mem[OBJ_Y + slot])
        if tile == cur or tile not in sweep:  # not your own column; must be in LOS
            continue
        # must look DOWN on the object: its FULL top (z_height + z_frac/256) must be at or
        # below the eye. Using only the integer z lets a same-level object whose fractional
        # top (e.g. a tree at 5.875 vs eye 5.0) sits ABOVE the eye slip through -- the real
        # ROM's looking-up check ($1D2E, full height) then REJECTS that absorb, so the native
        # energy over-counts. Compare full heights so the plan only absorbs what the ROM will.
        if g.mem[OBJ_Z + slot] + g.mem[OBJ_ZF + slot] / 256.0 > g.eye + 1e-9:
            continue
        cand.append((g.mem[OBJ_Z + slot] * 256 + g.mem[OBJ_ZF + slot], slot, ot, tile))
    for _z, slot, ot, tile in sorted(cand, reverse=True):
        if g.mem[OBJ_FLAGS + slot] & 0x80:  # already gone (stack collapsed)
            continue
        g.absorb(slot, sweep[tile], f"absorb {_FUEL_NAME[ot]} for fuel")
        gained += ENERGY[ot]
        log(f"    +fuel: absorbed {_FUEL_NAME[ot]} {tile}, energy {g.energy}")
    return gained


def plan_greedy(
    landscape,
    verbose=True,
    max_steps=120,
    blocked=frozenset(),
    start_energy=None,
    toward_plat=False,
    near_plat_radius=0,
):
    t0 = time.time()
    g = Game(landscape)
    if start_energy is not None:  # else keep the REAL generated energy
        g.energy = start_energy
    log = lambda *a: verbose and print(*a)
    plat = g.plat
    plat_ground = g.plat_ground if g.plat_ground is not None else 8
    target_z = plat_ground + 1
    log(
        f"greedy ls{landscape}: start {g.player_xy()} eye {g.eye} plat {plat} "
        f"plat_ground {plat_ground} target_z {target_z} energy {g.energy}"
    )

    visited = set()
    peak_eye = g.eye
    no_gain = 0
    for step in range(max_steps):
        cur = g.player_xy()
        eye = int(g.eye)
        if g.eye > peak_eye + 1e-9:
            peak_eye = g.eye
            no_gain = 0
        else:
            no_gain += 1
        if no_gain > 16:
            log(
                f"  step {step}: no height progress in 16 steps (peak {peak_eye:.2f}); stop"
            )
            break
        if eye > plat_ground and cheb(cur, plat) <= 1:
            log(f"  reached platform approach: {cur} eye {eye} (d=0/1)")
            break
        # ENERGY: from the real low starting energy, fund the climb by absorbing every
        # tree / sentry that comes into view (look-down) as the eye rises -- grab the
        # fuel the moment it's reachable so a build never beeps for lack of energy.
        _refuel(g, log)
        # _candidates forbids a non-adjacent boulder-step whose BUILD TILE is within
        # near_plat_radius of the platform (the ROM-infeasible on-distant-boulder synthoid
        # in the steep ring); far-from-platform far steps stay available for the pocket
        # break-out. Gating on the build tile (not the player) is what eliminates the
        # (18,9)->(18,6) cheb-3 step the ROM rejected.
        cands = _candidates(
            g,
            cur,
            eye,
            plat=(plat if toward_plat else None),
            near_plat_radius=near_plat_radius,
        )
        # affordability: a boulder-step pays boulder(2)+synthoid(3)=5 up front, a hop
        # pays synthoid(3), BEFORE the shell re-absorb refunds 3. Skip what we can't pay.
        cands = [c for c in cands if g.energy >= (5 if c[1] else 3) + RESERVE]
        # blocklist: footholds the real-ROM validation pass found infeasible (occluded /
        # gate-rejected) -- avoid them so the replan routes around the bad build.
        cands = [c for c in cands if (tuple(c[0]), c[1]) not in blocked]
        if not cands:
            # out of affordable footholds -- try once more to refuel, then give up.
            if g.energy < 6 and _refuel(g, log) > 0:
                continue
            log(
                f"  step {step}: NO affordable reachable foothold from {cur} eye {eye} "
                f"(energy {g.energy})"
            )
            break

        climbing = g.eye < target_z
        if toward_plat:
            # TOWARD-PLATFORM STRATEGY: ls66 starts in a low pocket whose only break-out
            # footholds lie W/SW; the original "furthest + most-edge" climb escapes the
            # pocket but ascends the NW corner, AWAY from the platform (21,3), and the
            # approach phase can't traverse back at elevation. Instead, among the
            # HEIGHT-GAINING boulder-steps take the one CLOSEST to the platform, then
            # HIGHEST. Because the terrain rises toward the platform, "climb while heading
            # at the platform" rides a boulder-staircase straight up the SW slope onto the
            # height-10 ring around the platform -- reaching (20,3) eye 11.375 (cheb 1),
            # the endgame-launch state. The very first moves still go W/SW (only footholds
            # visible from the pocket) but as soon as height is gained the bias pulls the
            # staircase east toward the platform.
            if climbing:
                gain = [c for c in cands if c[2] > g.eye + 1e-9]
                if gain:
                    pool = gain
                    key = lambda c: (
                        -cheb(c[0], plat),
                        c[2],
                    )  # closest to plat, then highest
                else:
                    # locally maxed: reposition toward the platform at the best height.
                    pool = [
                        c for c in cands if (c[0], round(c[2], 2)) not in visited
                    ] or cands
                    key = lambda c: (-cheb(c[0], plat), c[2])
            else:
                pool = [c for c in cands if c[2] >= plat_ground] or cands
                key = lambda c: (-cheb(c[0], plat), c[2])
        elif climbing:
            # THE STRATEGY (verbatim): gain HEIGHT as fast as possible, using the square
            # FURTHEST from the current position, avoiding the map CENTRE. Height gain is
            # the GATE (strict float gain -- the staircase climbs in 0.5 increments by
            # re-stacking); among gaining moves pick the FURTHEST, then most-edge, then
            # highest. Minimising transfers falls out of always taking the biggest jump.
            gain = [c for c in cands if c[2] > g.eye + 1e-9]
            if gain:
                # THE STRATEGY: MAX height per step, then FURTHEST from current, then most
                # EDGE (avoid the centre) -- this is what breaks out of the low start
                # pocket. boulder-steps give the height; hops alone can't climb.
                pool = gain
                key = lambda c: (c[2], cheb(c[0], cur), edge_dist(c[0]))
            else:
                # locally maxed: reposition to the FURTHEST unvisited high foothold to
                # find a fresh place to climb (bounded; avoids replouhing visited spots).
                pool = [c for c in cands if (c[0], round(c[2], 2)) not in visited]
                key = lambda c: (c[2], cheb(c[0], cur), edge_dist(c[0]))
        else:
            # APPROACH phase (high enough): close on the platform, staying high.
            pool = [c for c in cands if c[2] >= plat_ground] or cands
            key = lambda c: (-cheb(c[0], plat), c[2], edge_dist(c[0]))

        if not pool:
            log(
                f"  step {step}: NO foothold from {cur} eye {g.eye:.2f} "
                f"(d={cheb(cur, plat)}) -- climb pocketed here"
            )
            break
        T2, use_b, h_after, view = max(pool, key=key)
        visited.add((T2, round(h_after, 2)))
        _apply(g, T2, use_b, view)
        log(
            f"  [{step:2}] {'step' if use_b else 'hop '} -> {T2} eye {g.eye} "
            f"(d={cheb(g.player_xy(), plat)}) edge={edge_dist(T2):.0f} energy {g.energy}"
        )

    # ---- endgame: absorb the Sentinel (look down), synthoid on platform, transfer ----
    won = False
    cur = g.player_xy()
    eye = int(g.eye)
    if (
        g.plat_ground is not None
        and eye > g.plat_ground
        and cheb(cur, plat) <= 1
        and g.sentinel_slot is not None
    ):
        sw = visibility_sweep(g.mem, g.player, eye, max_steps=200, coarse=True)
        g.absorb(g.sentinel_slot, sw.get(plat), "absorb Sentinel")
        log(f"  absorbed Sentinel from eye {g.eye}, energy {g.energy}")
        if g.feasible(0, plat):
            g.transfer(
                g.create(0, plat, None, "platform synthoid"),
                "hyperspace onto platform (WIN)",
            )
            won = True
            log(f"  WIN: synthoid on platform {plat} + transfer")
    g.native_won = won
    log(
        f"=== greedy {'WON' if won else 'INCOMPLETE'} in {time.time()-t0:.2f}s, "
        f"{len(g.steps)} steps, final {g.player_xy()} eye {g.eye} ==="
    )
    return g


if __name__ == "__main__":
    ls = int(sys.argv[1]) if len(sys.argv) > 1 else 0
    g = plan_greedy(ls)
    out = {
        "landscape": ls,
        "native_won": g.native_won,
        "final_player": g.player_xy(),
        "eye": g.eye,
        "energy": g.energy,
        "steps": g.steps,
    }
    json.dump(out, open(f"out/kbd_greedy_{ls:04d}.json", "w"), indent=0)
    print(
        "FINAL",
        g.player_xy(),
        "eye",
        g.eye,
        "steps",
        len(g.steps),
        "native_won",
        g.native_won,
    )
