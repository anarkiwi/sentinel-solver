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
The endgame absorbs the Sentinel, builds a synthoid on the platform and transfers = win.

Runs on the bit-exact ``sentinel`` simulator via the ``plan_game.PlanGame`` adapter
(line-of-sight, actions and enemy exposure all come from ``sentinel``). Steps are
emitted in the verb/otype/target/view shape the live keyboard driver replays.
"""

import sys, os, json, time

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
sys.path.insert(0, os.path.dirname(HERE))
os.chdir(os.path.join(HERE, ".."))
import plan_game as NG
from plan_game import (
    PlanGame as Game,
    visibility_sweep,
    terrain_z,
    cheb,
    OBJ_X,
    OBJ_Y,
    OBJ_Z,
    OBJ_TYPE,
    OBJ_ZF,
    OBJ_FLAGS,
    OBJ_HANG,
    OBJ_VANG,
)
from sentinel import los, threat
from sentinel import aimcost as ac


def _enemy_exposed_tiles(g, tiles, object_top=threat.ROBOT_EYE):
    """Which of `tiles` are within LOS of any enemy (sentry/Sentinel), using the
    existing game_model.is_exposed conservative check (any rotation, not just the
    enemy's current facing -- the same "could this ever be seen" question a planner
    wants when deciding where to build/leave objects). game_model reads the SAME
    tiles_table/object-array addresses native_game.py uses, so g.mem (live or
    offline, 4KB or 64KB) can be wrapped directly with no conversion.

    Real-game motivation: a live keyboard-driven climb takes real WALL-CLOCK seconds
    per action; an object left standing in enemy LOS for that long can be drained or
    downgraded (robot->boulder->tree, enemy_dynamics.py) before the plan gets back to
    it -- observed live as an absorb landing on a lower-value object than planned.
    This lets the planner prefer NOT to leave footholds in enemy view when an
    equally-good alternative exists (it usually can't avoid it entirely near the
    Sentinel's own platform -- everything overlooking it is exposed to the Sentinel
    itself by construction)."""
    return threat.exposed_tiles(g.state, set(tiles))


def _lattice_sweep(g, eye):
    """ONE keyboard-lattice sweep (h ≡ 0 mod 8, v in the pan band, centre cursor)
    returning {tile: first-LOS view} plus {tile: min tile-centre fraction}. The centre
    map is the keyboard-feasibility gate for on-boulder synthoids ($1E48 needs the
    tile-centre fraction < $40). Same order/params as visibility_sweep(coarse) so the
    chosen views are identical."""
    return los.sweep_with_centres(g.state, g.player, eye, max_steps=200)


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
RESERVE = int(os.environ.get("CLIMB_RESERVE", "2"))

# EXTRA reserve required when a candidate's DESTINATION tile is enemy-exposed (see
# _enemy_exposed_tiles): a live keyboard-driven climb can spend many real seconds
# choosing/aiming its NEXT move while standing there, and an exposed player can be
# drained during that wait. Without this margin a move that's merely affordable
# TODAY can leave no energy to afford any escape once drained a few points --
# observed live as energy draining 6->0 over repeated stuck retries at an exposed
# tile with no affordable move (see climb_iterate's affordability filter).
# NOTE: tuned like RESERVE above -- 2 stalls ls0 (oscillates, unable to ever afford
# a height-gaining boulder-step near the exposed platform ring, native_won False).
# 1 is the largest value that preserves the ROM-validated ls0 win.
EXPOSURE_RESERVE = int(os.environ.get("CLIMB_EXPOSURE_RESERVE", "1"))

# Accept a fractional eye above the platform ground as launch-ready in endgame(): the
# default int(eye)>plat_ground gate rejects eye 8.875 over a z8 platform even though the
# viewpoint is genuinely above it (see climb_search._reached_approach). Off by default.
_ENDGAME_FRAC_EYE = os.environ.get("ENDGAME_FRAC_EYE", "0") == "1"

# PAN-aware fuel: absorbing a fuel object requires panning the view to its bearing, and
# the Sentinel drains the player during that scroll (see climb_search pan-cost notes).
# The default _refuel greedily grabs EVERY visible object, chasing far-bearing fuel it
# doesn't need -- a long pan per grab, drained live. When on: bank fuel NEAR the current
# heading first, and SKIP a far-bearing (> _PAN_FUEL_MAX units) non-exposed grab once the
# player already holds a comfortable buffer (so the climb never starves, but doesn't chase
# distant fuel for a surplus it can bank locally later). Exposed trail is always reclaimed
# (it degrades if left). Off by default (keeps the ROM-validated ls42 refuel).
_PAN_COST = os.environ.get("PAN_COST", "0") == "1"
# skip a far-bearing non-exposed fuel grab past this many enemy rounds of panning.
_PAN_FUEL_MAX = float(os.environ.get("PAN_FUEL_MAX", "64"))


def _pan_units(cur_h, cur_v, view):
    """Enemy rounds to pan the view from heading (cur_h, cur_v) to `view`'s aim -- a
    proxy for scroll time (and thus mid-aim drain). Uses the canonical keyboard-aim
    geometry (sentinel.aimcost) weighted by the ROM scroll cadence (a bearing keystroke
    animates a 16-round scroll, a pitch keystroke 8), the same conversion the search's
    move cost uses. 0 when the view carries no h angle."""
    if not view or view.get("h_angle") is None:
        return 0.0
    return ac.bearing_rounds(cur_h, view["h_angle"], 16, 16) + (
        ac.v_steps(cur_v, view["v_angle"]) * 8 if view.get("v_angle") is not None else 0
    )


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


def _boulder_centre_feasible(g, T2, view, max_steps=2000):
    """Simulate landing a boulder at T2 (a direct byte-level mutation mirroring
    Game.create's own boulder-placement logic, on a throwaway mem copy -- no scratch
    Game/col reconstruction needed) and check whether the on-boulder synthoid then has
    an actual keyboard-reachable centre-aim ($1E48 <$40), from the PLAYER's current eye
    (the ray origin doesn't move until the transfer after this build). The pre-build
    lattice sweep's centre estimate (`centres`, in _candidates) is taken against the
    BARE-TERRAIN surface, not the post-build object surface, so a tile can look
    centre_ok there yet have NO valid on-boulder aim once the boulder is real --
    wasting a real build (energy + real-time aiming) to discover it, exactly what the
    ROM-validator's blocklist-replan exists to catch offline. Live driving has no such
    later pass, so catch it here instead: pure native math, no ROM replay."""
    if not g.free:
        return False
    mem = bytearray(g.mem)
    slot = g.free[-1]
    tb = mem[0x0400 + NG.TIDX(*T2)]
    if tb >= 0xC0:  # stacking a boulder onto an existing boulder/platform
        below = tb & 0x3F
        btype = mem[OBJ_TYPE + below]
        if btype not in (3, 6):
            return False
        t = mem[OBJ_ZF + below] + 0x80
        zf = t & 0xFF
        z = mem[OBJ_Z + below] + (t >> 8)
        mem[OBJ_FLAGS + slot] = 0x40 | below
    else:  # bare terrain
        zf = 0xE0
        z = tb >> 4
        mem[OBJ_FLAGS + slot] = 0x00
    mem[0x0400 + NG.TIDX(*T2)] = 0xC0 | slot
    mem[OBJ_X + slot] = T2[0]
    mem[OBJ_Y + slot] = T2[1]
    mem[OBJ_Z + slot] = z & 0xFF
    mem[OBJ_ZF + slot] = zf & 0xFF
    mem[OBJ_TYPE + slot] = 3
    return (
        NG.centre_view_for(mem, T2, g.player, int(g.eye), max_steps=max_steps)
        is not None
    )


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


def _refuel(g, log, sweep_max_steps=320, sweep_coarse=False):
    """Earn energy by absorbing everything you can look DOWN on (top <= eye, in LOS):
    TREES (+1) and SENTRIES (+3), AND -- crucially -- the climb's OWN abandoned BOULDERS
    (+2) and SYNTHOIDS (+3) once you've climbed past them. Reclaiming your own trail makes
    the staircase nearly energy-NEUTRAL (the boulder/synthoid cost is recovered), which is
    how the climb funds itself from the low real starting energy. Never absorb the column
    the player is standing on (you'd fall). Returns energy gained.

    sweep_max_steps/sweep_coarse tune the LOS sweep: the fine default is used on the real
    committed state; the lookahead search passes a coarser/shorter sweep (it only needs
    an APPROXIMATE recovered-energy figure for affordability, and this sweep dominates its
    per-node cost)."""
    from plan_game import ENERGY

    gained = 0
    cur = g.player_xy()
    sweep = visibility_sweep(
        g.mem, g.player, int(g.eye), max_steps=sweep_max_steps, coarse=sweep_coarse
    )
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
    # only the TOPMOST object per tile: `sweep[tile]`'s view/aim was computed for
    # whatever was on top when the sweep ran. A live keyboard aim centres on the
    # target's own height, so it does NOT carry over once the top object is gone --
    # absorbing a second, lower object at the same tile with the stale view can miss
    # in the real ROM even though the native model (which has no aim concept) sees it
    # as fine. Leave any object thus exposed for the NEXT refuel pass, which resyncs
    # (live) or re-sweeps (offline) and gets a fresh, correct view for it.
    #
    # Reclaim ENEMY-EXPOSED fuel (typically the climb's own abandoned boulder/synthoid
    # trail, left standing in the Sentinel's view) BEFORE safer fuel elsewhere: a live
    # keyboard-driven climb takes real wall-clock seconds per action, and an exposed
    # object left too long can be drained or downgraded before this batch gets to it.
    exposed = _enemy_exposed_tiles(g, {tile for _z, _slot, _ot, tile in cand})
    tiles_done = set()
    # PAN-aware ordering: keep exposed-trail reclaim first (it degrades if left), then --
    # when pan cost is on -- bank the NEAREST-bearing fuel before far grabs, so if the
    # Sentinel comes around mid-pass the cheap local energy is already secured. Default
    # order (exposed-first, height-first) is preserved when pan cost is off.
    comfort = RESERVE + 6  # energy buffer above which far-bearing fuel is skipped (PAN)
    if _PAN_COST:
        ch, cv = g.mem[OBJ_HANG + g.player], g.mem[OBJ_VANG + g.player]
        order = sorted(
            cand,
            key=lambda c: (
                c[3] not in exposed,
                _pan_units(ch, cv, sweep.get(c[3])),
                -c[0],
            ),
        )
    else:
        order = sorted(cand, key=lambda c: (c[3] in exposed, c[0]), reverse=True)
    for _z, slot, ot, tile in order:
        if tile in tiles_done:
            continue
        if g.mem[OBJ_FLAGS + slot] & 0x80:  # already gone (stack collapsed)
            continue
        # skip a far-bearing non-exposed grab once a comfortable buffer is banked: the long
        # pan to reach it would be drained for fuel not needed this pass (grab it later,
        # closer, or when energy demands). Exposed trail is never skipped (it degrades).
        if (
            _PAN_COST
            and tile not in exposed
            and g.energy >= comfort
            and _pan_units(ch, cv, sweep.get(tile)) > _PAN_FUEL_MAX
        ):
            continue
        g.absorb(slot, sweep[tile], f"absorb {_FUEL_NAME[ot]} for fuel")
        gained += ENERGY[ot]
        tiles_done.add(tile)
        log(f"    +fuel: absorbed {_FUEL_NAME[ot]} {tile}, energy {g.energy}")
    return gained


def climb_ctx(g, toward_plat=False, near_plat_radius=0):
    """Fixed climb parameters + mutable progress state for `climb_iterate`, derived
    from g's CURRENT state (so it works whether g is a fresh landscape or a live
    resync mid-climb)."""
    plat = g.plat
    plat_ground = g.plat_ground if g.plat_ground is not None else 8
    return {
        "plat": plat,
        "plat_ground": plat_ground,
        "target_z": plat_ground + 1,
        "toward_plat": toward_plat,
        "near_plat_radius": near_plat_radius,
        "visited": set(),
        "tiles_used": set(),  # TILES a foothold has ever been built on (any height) --
        # blocks the "mine a small cluster" cycle: build here, later absorb your OWN
        # structure back as fuel, then rebuild nearby at a similar height forever
        # without net progress (each (tile, height) pair in `visited` is distinct
        # enough to dodge that check since the exact height differs slightly each
        # lap). Once a tile has hosted a foothold, don't build there again.
        "peak_eye": g.eye,
        "no_gain": 0,
        "runtime_blocked": set(),
    }


def climb_iterate(g, ctx, blocked, log):
    """ONE iteration of the greedy climb decision, against g's CURRENT state (which
    a live driver may have just resynced from real memory). Mutates ctx's
    visited/peak_eye/no_gain in place and appends to g.steps via g.create/absorb/
    transfer. Returns a status string:
      'retry'    -- no foothold move this call (a refuel-only pass); call again
      'stepped'  -- a foothold move (+ any refuel absorbs) was applied
      'no_gain'  -- no height progress in 16 iterations; climb should stop
      'approach' -- reached the platform approach (eye>plat_ground, cheb<=1); stop
      'stuck'    -- no affordable/reachable foothold; climb should stop
    """
    plat, plat_ground, target_z = ctx["plat"], ctx["plat_ground"], ctx["target_z"]
    toward_plat, near_plat_radius = ctx["toward_plat"], ctx["near_plat_radius"]
    cur = g.player_xy()
    eye = int(g.eye)
    # ENERGY: from the real low starting energy, fund the climb by absorbing every
    # tree / sentry that comes into view (look-down) as the eye rises -- grab the
    # fuel the moment it's reachable. Called BEFORE the no_gain check (below) so a
    # long, deliberate absorb streak -- banking fuel for several iterations before
    # spending it -- counts as real progress, not stalling: a recorded human win on
    # ls0 spent ~15s at one vantage point absorbing 5+ energy before its next build,
    # which the OLD height-only no_gain counter would have aborted as "no progress"
    # since eye doesn't move while banking.
    gained_this_pass = _refuel(g, log)
    if g.eye > ctx["peak_eye"] + 1e-9:
        ctx["peak_eye"] = g.eye
        ctx["no_gain"] = 0
    elif gained_this_pass > 0:
        ctx["no_gain"] = 0
    else:
        ctx["no_gain"] += 1
    if ctx["no_gain"] > 16:
        log(f"  no height/fuel progress in 16 steps (peak {ctx['peak_eye']:.2f}); stop")
        return "no_gain"
    if eye > plat_ground and cheb(cur, plat) <= 1:
        log(f"  reached platform approach: {cur} eye {eye} (d=0/1)")
        return "approach"
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
    climbing = g.eye < target_z

    def _cost(c, exposed):
        return (
            (5 if c[1] else 3)
            + RESERVE
            + (EXPOSURE_RESERVE if tuple(c[0]) in exposed else 0)
        )

    # PATIENCE: a recorded human win on ls0 banked a real energy surplus (absorbing
    # everything visible from one vantage point, sometimes 5+ energy) BEFORE ever
    # committing to the next build, then took the single biggest height+edge jump
    # available -- landing +1.0 eye per move, roughly double this planner's old
    # "take the first affordable gaining move" behaviour (+0.5/move, spending down to
    # near-zero every time). Model that: if the BEST possible gaining move (any
    # height, any cost) isn't affordable yet, but this pass's refuel found more fuel,
    # hold off and keep banking instead of settling for a smaller move that IS
    # affordable now. Only settle once refueling plateaus (nothing left to absorb
    # from here) -- the no_gain counter above still bounds how long that can go on.
    if climbing and not toward_plat:
        gain_all = [c for c in cands if c[2] > g.eye + 1e-9]
        # prefer FRESH ground over rebuilding on a tile already used as a foothold
        # (only fall back to reuse if nothing fresh gains height) -- see tiles_used.
        fresh_gain = [c for c in gain_all if tuple(c[0]) not in ctx["tiles_used"]]
        gain_all = fresh_gain or gain_all
        if gain_all:
            exposed_all = _enemy_exposed_tiles(g, {tuple(c[0]) for c in gain_all})
            gaze_all = threat.gaze_distance(g.state, {tuple(c[0]) for c in gain_all})
            best_possible = max(
                gain_all,
                key=lambda c: (c[2], gaze_all.get(tuple(c[0]), 0), edge_dist(c[0])),
            )
            if g.energy < _cost(best_possible, exposed_all) and gained_this_pass > 0:
                log(
                    f"  banking fuel before committing: best available gain is "
                    f"{best_possible[2]:.3f} (needs {_cost(best_possible, exposed_all)}, "
                    f"have {g.energy}); absorbed {gained_this_pass} this pass, retrying"
                )
                return "retry"

    # affordability: a boulder-step pays boulder(2)+synthoid(3)=5 up front, a hop
    # pays synthoid(3), BEFORE the shell re-absorb refunds 3. Skip what we can't pay.
    # Exposed DESTINATIONS need the extra EXPOSURE_RESERVE margin (see its docstring):
    # landing there may strand the player waiting on an unaffordable next move while
    # actively being drained.
    exposed_dest = _enemy_exposed_tiles(g, {tuple(c[0]) for c in cands})
    cands = [c for c in cands if g.energy >= _cost(c, exposed_dest)]
    # blocklist: footholds a real ROM build/aim found infeasible (occluded / gate-
    # rejected / occupied by a meanie), plus ones THIS session's own centre-aim probe
    # found infeasible (runtime_blocked) -- avoid them so the replan routes around it.
    cands = [
        c
        for c in cands
        if (tuple(c[0]), c[1]) not in blocked
        and (tuple(c[0]), c[1]) not in ctx["runtime_blocked"]
    ]
    if not cands:
        # out of affordable footholds this pass. Retry rather than calling _refuel
        # again HERE: a same-tile object _refuel deferred (see its one-per-tile note)
        # may now be absorbable, but computing that from THIS SAME native state (no
        # real resync/live keypress has happened yet) would batch a second same-tile
        # absorb using a view computed for a hypothetical post-absorb board -- exactly
        # the stale-aim risk that regressed a real run (an abandoned robot the plan
        # expected was, live, already downgraded to a boulder by an enemy; the
        # precomputed second absorb then fired against a board state that never
        # actually existed). Returning "retry" lets the NEXT iteration's own fresh
        # resync (live) or fresh native sweep (offline) discover it correctly instead.
        # The no_gain counter above still ends the loop after 16 height-less iterations,
        # so this can't spin forever if the board is genuinely out of reachable fuel.
        if g.energy < 6:
            log(
                f"  no affordable reachable foothold from {cur} eye {eye} "
                f"(energy {g.energy}); retrying (deferred fuel?)"
            )
            return "retry"
        log(
            f"  NO affordable reachable foothold from {cur} eye {eye} (energy {g.energy})"
        )
        return "stuck"

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
                # HEIGHT FIRST (matches THE STRATEGY's own height-first doctrine below --
                # always take the biggest jump you can), platform-proximity only as a
                # tiebreaker among equal-height candidates. Sorting by proximity-to-plat
                # FIRST (as this used to) makes the climb take whatever minimal-gain step
                # is closest to the platform even when a much bigger jump is available
                # elsewhere -- visibly wasteful small increments, not "gain fastest".
                key = lambda c: (
                    c[2],
                    -cheb(c[0], plat),
                )  # highest, then closest to plat
            else:
                # locally maxed: reposition toward the platform at the best height.
                pool = [
                    c for c in cands if (c[0], round(c[2], 2)) not in ctx["visited"]
                ] or cands
                key = lambda c: (-cheb(c[0], plat), c[2])
        else:
            pool = [c for c in cands if c[2] >= plat_ground] or cands
            key = lambda c: (-cheb(c[0], plat), c[2])
    elif climbing:
        # THE STRATEGY (verbatim): gain HEIGHT as fast as possible, moving toward the
        # EDGE, while tracking the Sentinel's CURRENT gaze and preferring footholds
        # furthest from where it's actually looking right now (on top of the hard
        # is_exposed avoidance below, which only knows "could this ever be seen").
        # Height gain is the GATE (strict float gain -- the staircase climbs in 0.5
        # increments by re-stacking); among gaining moves pick the HIGHEST, then
        # furthest from the enemy's current gaze, then most-edge.
        gain = [c for c in cands if c[2] > g.eye + 1e-9]
        # prefer FRESH ground (see tiles_used): otherwise the climb can mine a small
        # cluster forever -- build here, later absorb the same structure back as
        # fuel, rebuild nearby at a near-identical height -- since each lap's exact
        # height differs slightly, `visited` (tile, height) pairs don't catch it.
        gain = [c for c in gain if tuple(c[0]) not in ctx["tiles_used"]] or gain
        if gain:
            gaze = threat.gaze_distance(g.state, {tuple(c[0]) for c in gain})
            # THE STRATEGY: MAX height per step, then furthest from the Sentinel's
            # CURRENT gaze, then most EDGE (avoid the centre) -- this is what breaks
            # out of the low start pocket. boulder-steps give the height; hops alone
            # can't climb.
            pool = gain
            key = lambda c: (c[2], gaze.get(tuple(c[0]), 0), edge_dist(c[0]))
        else:
            # locally maxed: reposition to the FURTHEST unvisited high foothold to
            # find a fresh place to climb (bounded; avoids replouhing visited spots).
            pool = [c for c in cands if (c[0], round(c[2], 2)) not in ctx["visited"]]
            key = lambda c: (c[2], cheb(c[0], cur), edge_dist(c[0]))
    else:
        # APPROACH phase (high enough): close on the platform, staying high.
        pool = [c for c in cands if c[2] >= plat_ground] or cands
        key = lambda c: (-cheb(c[0], plat), c[2], edge_dist(c[0]))

    if not pool:
        log(
            f"  NO foothold from {cur} eye {g.eye:.2f} "
            f"(d={cheb(cur, plat)}) -- climb pocketed here"
        )
        return "stuck"
    if not toward_plat and climbing:
        # THE STRATEGY's own key already ranks HEIGHT first with the Sentinel's
        # CURRENT gaze-distance as the safety tiebreaker (see threat.gaze_distance
        # above) -- that key IS the height/safety tradeoff for this mode. Layering
        # the static "prefer non-exposed" hard reorder below on TOP of it overrides
        # height with static is_exposed status whenever ANY non-exposed candidate
        # exists, even a far lower one -- confirmed live: it picked a +0.375 tile
        # over several centre-feasible +1.0 tiles purely because the big ones were
        # (statically) exposed and the small one wasn't, exactly the small-wasteful-
        # step behaviour a recorded human win showed THE STRATEGY should avoid (see
        # climb_iterate's PATIENCE comment). So for this mode, trust the key as-is.
        safe_first = sorted(pool, key=key, reverse=True)
    else:
        # Other modes (toward_plat, approach) don't yet weigh gaze in their own key,
        # so keep the static is_exposed PREFERENCE (not a hard filter) here: walk the
        # non-exposed candidates first, falling back to exposed ones only if none of
        # those are centre-feasible either.
        exposed = _enemy_exposed_tiles(g, {tuple(c[0]) for c in pool})
        ordered = sorted(pool, key=key, reverse=True)
        safe_first = [c for c in ordered if tuple(c[0]) not in exposed] + [
            c for c in ordered if tuple(c[0]) in exposed
        ]
    # Walk the pool in preference order, verifying boulder-step centre-feasibility only
    # for the candidate actually under consideration (not the whole pool up front --
    # that's O(pool size) native-LOS sweeps every iteration for a check only the
    # eventual pick needs). A rejected boulder-step is blocklisted so future iterations
    # don't retry it.
    chosen = None
    for cand in safe_first:
        cT2, cuse_b, _ch_after, cview = cand
        if cuse_b and not _boulder_centre_feasible(g, cT2, cview):
            ctx["runtime_blocked"].add((tuple(cT2), True))
            log(f"  boulder-step {cT2}: no on-boulder centre-aim once built; blocking")
            continue
        chosen = cand
        break
    if chosen is None:
        log(
            f"  NO feasible foothold from {cur} eye {g.eye:.2f} "
            f"(d={cheb(cur, plat)}) after centre-aim check -- climb pocketed here"
        )
        return "stuck"
    T2, use_b, h_after, view = chosen
    ctx["visited"].add((T2, round(h_after, 2)))
    ctx["tiles_used"].add(T2)
    _apply(g, T2, use_b, view)
    log(
        f"  {'step' if use_b else 'hop '} -> {T2} eye {g.eye} "
        f"(d={cheb(g.player_xy(), plat)}) edge={edge_dist(T2):.0f} energy {g.energy}"
    )
    return "stepped"


def endgame(g, plat, log):
    """Attempt the win sequence (absorb the Sentinel, build a platform synthoid,
    transfer) from g's CURRENT state. Returns True iff the transfer onto the
    platform (the win condition) was applied.

    The win is a LONG-RANGE hop: a synthoid can be created on any tile in line of sight
    (the sweep supplies the aim), so the endgame needs the eye above the platform ground
    and LOS to the platform tile -- NOT adjacency. This lets the climb top out on the
    highest ground it can find and fire the win from afar (the human ls0 win was cheb-10),
    instead of creeping into the platform ring."""
    eye = int(g.eye)
    if g.plat_ground is None or g.sentinel_slot is None:
        return False
    above = g.eye > g.plat_ground if _ENDGAME_FRAC_EYE else eye > g.plat_ground
    if not above:
        return False
    # A fractional eye above the platform ground sees DOWN onto it; sweeping at int(eye)
    # (== plat_ground, the player's own level) drops the platform as not-below. Ceil the
    # observer to reflect that the viewpoint is genuinely above (only when frac-eye is on).
    seye = eye + 1 if (_ENDGAME_FRAC_EYE and g.eye > eye) else eye
    sw = visibility_sweep(g.mem, g.player, seye, max_steps=200, coarse=True)
    if plat not in sw:  # need LOS to the platform to absorb the Sentinel + build on it
        return False
    g.absorb(g.sentinel_slot, sw.get(plat), "absorb Sentinel")
    log(f"  absorbed Sentinel from eye {g.eye}, energy {g.energy}")
    if g.feasible(0, plat):
        g.transfer(
            g.create(0, plat, None, "platform synthoid"),
            "hyperspace onto platform (WIN)",
        )
        log(f"  WIN: synthoid on platform {plat} + transfer")
        return True
    return False


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
    ctx = climb_ctx(g, toward_plat, near_plat_radius)
    log(
        f"greedy ls{landscape}: start {g.player_xy()} eye {g.eye} plat {ctx['plat']} "
        f"plat_ground {ctx['plat_ground']} target_z {ctx['target_z']} energy {g.energy}"
    )

    for _ in range(max_steps):
        status = climb_iterate(g, ctx, blocked, log)
        if status in ("no_gain", "approach", "stuck"):
            break

    won = endgame(g, ctx["plat"], log)
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
