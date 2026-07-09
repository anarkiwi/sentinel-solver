#!/usr/bin/env python3
"""Receding-horizon best-first lookahead climb planner (SEARCH_REDESIGN.md).

The per-decision climb function. A purely greedy picker commits to the single
best-LOOKING foothold each call and only discovers a dead end after arriving
(SEARCH_REDESIGN.md sec.1). This runs a bounded depth-`D` lookahead at every real
decision, scores the leaves, and commits the FIRST move of the best-scoring line --
so a move is taken ONLY if it has a continuation within the horizon (a node with no
non-dead-end successor scores -inf and is pruned by its parent). Re-runs fresh from
each real state, chess-engine style: the plan can be arbitrarily long though each
search is shallow (sec.3).

It REUSES the real, keyboard-faithful mechanics validated against plan_game (sec.4,
sec.8) -- _candidates, _boulder_centre_feasible, _refuel, _apply, edge_dist, defined
above -- and branches over plan_game.Game.clone()d states. Enemy timing is folded
in as a per-candidate SAFETY ANNOTATION via sentinel.threat (sec.5), not an adversarial
search dimension (sec.2): meanie_safe is a hard pre-filter, ticks_until_seen a soft
leaf-eval term. Each node's sentinel enemy state is advanced by the move's REAL tick cost
(_move_cost, derived from the keyboard-aim geometry: aim + build + transfer + the return-
pan to reabsorb the shell) as the lookahead descends (sec.6).
"""

import sys, os, json, time

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.dirname(HERE))
os.chdir(os.path.join(HERE, ".."))

from solver import plan_game as NG
from solver.plan_game import (
    PlanGame,
    cheb,
    visibility_sweep,
    terrain_z,
)
from sentinel import los, threat, enemies as SE, aimcost as ac, actioncost, actions
from sentinel import memmap as mm

# --- shared climb mechanics (keyboard-faithful, validated against plan_game) --------
# These were the reusable core the greedy picker and this search both drove; they now
# live here, this being the single solver.


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
    """ONE forward sweep of the REAL keyboard aim lattice (h ≡ 0 mod 8, sights cursor on
    its 9px grid, v_angle $F5) returning {tile: first-LOS view} plus {tile: min tile-centre
    fraction}. This is AIM-landability -- the tiles the sights actually land on, i.e. the
    tiles a live player can build on -- NOT the centred-cursor geometric visibility of the
    old sweep, which over/under-reported far/diagonal builds and let the planner commit
    footholds no keyboard aim could reach (the ls0 (4,10)/(14,4) mirages). The centre map
    is the keyboard-feasibility gate for on-boulder synthoids ($1E48 needs fraction < $40).
    """
    return los.landable_sweep_with_centres(g.state, g.player, eye)


def _top_type(g, tile):
    """Type of the TOPMOST live object on `tile` (highest z_height+z_frac), or None if
    the tile is bare terrain. Used to gate boulder placement: a boulder may sit on bare
    terrain or on another boulder (type 3), never on a synthoid (0) / tree (2)."""
    top, topz = None, -1
    for s in range(64):
        if g.mem[mm.OBJECTS_FLAGS + s] & 0x80:
            continue
        if (g.mem[mm.OBJECTS_X + s], g.mem[mm.OBJECTS_Y + s]) == tile:
            z = g.mem[mm.OBJECTS_Z_HEIGHT + s] * 256 + g.mem[mm.OBJECTS_Z_FRACTION + s]
            if z > topz:
                topz, top = z, g.mem[mm.OBJECTS_TYPE + s]
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
# tile with no affordable move (the affordability filter).
# NOTE: tuned like RESERVE above -- 2 stalls ls0 (oscillates, unable to ever afford
# a height-gaining boulder-step near the exposed platform ring, native_won False).
# 1 is the largest value that preserves the ROM-validated ls0 win.
EXPOSURE_RESERVE = int(os.environ.get("CLIMB_EXPOSURE_RESERVE", "1"))

# PAN-aware fuel: absorbing a fuel object requires panning the view to its bearing, and
# the Sentinel drains the player during that scroll (see the pan-cost notes below). The
# default _refuel greedily grabs EVERY visible object, chasing far-bearing fuel it
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


def _boulder_batch(g, T2):
    """RULE 5 (batch boulders): how many boulders can be stacked on T2 in ONE dwell,
    and the resulting synthoid-top eye, given the current eye. The ROM build-height gate
    ($1F38, plan_game.feasible) caps an object built on an ALREADY-occupied column at
    top <= eye + ROBOT_EYE_FUDGE; the first boulder on bare terrain is uncapped. So stack
    boulders while the next one -- and the capping synthoid on top -- still fit under the
    cap. Returns (n_boulders, final_eye), or (0, None) if not even a single boulder-step
    is feasible here. Batching gains the full ~2-unit slack per transfer, so the climb
    reaches launch height in far fewer (exposed) transfers -- before the Sentinel's gaze
    precesses onto the player."""
    base = g.top_of(T2)
    if base is None:
        return 0, None
    first = T2 not in g.col
    cap = g.eye + NG.ROBOT_EYE_FUDGE
    top = base + (0.875 if first else 0.5)  # first boulder
    if top + 0.5 > cap + 1e-9:  # no room for even the capping synthoid
        return 0, None
    # how many boulders the build-height SLACK allows (each +0.5, room for the synthoid).
    n_slack = 1
    while (top + 0.5) + 0.5 <= cap + 1e-9:
        top += 0.5
        n_slack += 1
    # ...but cap the batch by AFFORDABILITY: a batch costs 2*n + 3 (+ RESERVE) up front, so
    # maxing boulders can price the step out of a low buffer and drop it entirely -- exactly
    # when a CHEAPER 1-2 boulder step would still reach launch height. Keep the tallest batch
    # the current energy can pay for (>=1 so a genuinely unaffordable tile is still offered
    # and dropped by the affordability filter, not silently mis-sized).
    n_afford = int((g.energy - RESERVE - 3) // 2)
    n = max(1, min(n_slack, n_afford))
    return n, base + (0.875 if first else 0.5) + 0.5 * (n - 1) + 0.5


def _foothold_eye(g, T2, use_b):
    """TRUE resulting eye (float) of moving onto T2, using Game.create's exact ROM
    height increments: first object on terrain = +0.875 (boulder) then synthoid +0.5;
    stacking onto an existing column (T2 already in g.col) = +0.5 each. A boulder-step
    BATCHES boulders (RULE 5, _boulder_batch) to gain the full slack in one transfer.
    This captures the FRACTIONAL staircase gain (re-stacking an already-used tile climbs
    +0.5-1), which a flat terrain+1 model misses."""
    cur = g.top_of(T2)
    if cur is None:
        return None
    first = T2 not in g.col
    if use_b:
        return _boulder_batch(g, T2)[1]  # batched boulders + capping synthoid
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
    tb = mem[0x0400 + mm.tidx(*T2)]
    if tb >= 0xC0:  # stacking a boulder onto an existing boulder/platform
        below = tb & 0x3F
        btype = mem[mm.OBJECTS_TYPE + below]
        if btype not in (3, 6):
            return False
        t = mem[mm.OBJECTS_Z_FRACTION + below] + 0x80
        zf = t & 0xFF
        z = mem[mm.OBJECTS_Z_HEIGHT + below] + (t >> 8)
        mem[mm.OBJECTS_FLAGS + slot] = 0x40 | below
    else:  # bare terrain
        zf = 0xE0
        z = tb >> 4
        mem[mm.OBJECTS_FLAGS + slot] = 0x00
    mem[0x0400 + mm.tidx(*T2)] = 0xC0 | slot
    mem[mm.OBJECTS_X + slot] = T2[0]
    mem[mm.OBJECTS_Y + slot] = T2[1]
    mem[mm.OBJECTS_Z_HEIGHT + slot] = z & 0xFF
    mem[mm.OBJECTS_Z_FRACTION + slot] = zf & 0xFF
    mem[mm.OBJECTS_TYPE + slot] = 3
    return (
        NG.centre_view_for(mem, T2, g.player, int(g.eye), max_steps=max_steps)
        is not None
    )


def _candidates(g, cur, eye):
    """All (T2, use_boulder, eye_after_float, view) footholds reachable with LOS, as a
    hop (synthoid) or a boulder-step (centre-aimed, climbs). Coarse sweep is fine --
    views are recomputed at validate time.

    RULE 2: footholds are NOT filtered by Chebyshev distance to the platform. A
    boulder-step whose on-boulder synthoid has no ROM-reachable centre-aim in steep
    terrain is caught by the real LOS feasibility probe (`_boulder_centre_feasible`, run
    at the committed root ply) -- a line-of-sight gate on the specific tile, not a
    distance heuristic. So this offers every LOS-reachable foothold and lets the true
    keyboard/LOS feasibility (and the gaze/exposure safety in `_gen_candidates`) decide.
    """
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
        n, _ = _boulder_batch(g, T2)  # RULE 5: stack the whole batch in one dwell
        for i in range(max(1, n)):
            g.create(3, T2, view if i == 0 else None, "climb boulder")
        s = g.create(0, T2, None, "climb synthoid")  # centre-aim view at validate
    else:
        s = g.create(0, T2, view, "hop synthoid")
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
        and g.mem[mm.OBJECTS_TYPE + prev_slot] == 0
        and g.mem[mm.OBJECTS_Z_HEIGHT + prev_slot]
        + g.mem[mm.OBJECTS_Z_FRACTION + prev_slot] / 256.0
        <= g.eye + 1e-9
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

    gained = 0
    cur = g.player_xy()
    sweep = visibility_sweep(
        g.mem, g.player, int(g.eye), max_steps=sweep_max_steps, coarse=sweep_coarse
    )
    # absorb topmost-first per tile (matches the ROM gate) so stacks unwind cleanly.
    cand = []
    for slot in range(64):
        if (g.mem[mm.OBJECTS_FLAGS + slot] & 0x80) or slot == g.player:
            continue
        ot = g.mem[mm.OBJECTS_TYPE + slot]
        if ot not in (0, 1, 2, 3):  # synthoid/sentry/tree/boulder = fuel
            continue
        tile = (g.mem[mm.OBJECTS_X + slot], g.mem[mm.OBJECTS_Y + slot])
        if tile == cur or tile not in sweep:  # not your own column; must be in LOS
            continue
        # must look DOWN on the object: its FULL top (z_height + z_frac/256) must be at or
        # below the eye. Using only the integer z lets a same-level object whose fractional
        # top (e.g. a tree at 5.875 vs eye 5.0) sits ABOVE the eye slip through -- the real
        # ROM's looking-up check ($1D2E, full height) then REJECTS that absorb, so the native
        # energy over-counts. Compare full heights so the plan only absorbs what the ROM will.
        if (
            g.mem[mm.OBJECTS_Z_HEIGHT + slot]
            + g.mem[mm.OBJECTS_Z_FRACTION + slot] / 256.0
            > g.eye + 1e-9
        ):
            continue
        cand.append(
            (
                g.mem[mm.OBJECTS_Z_HEIGHT + slot] * 256
                + g.mem[mm.OBJECTS_Z_FRACTION + slot],
                slot,
                ot,
                tile,
            )
        )
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
    ch, cv = g.mem[mm.OBJECTS_H_ANGLE + g.player], g.mem[mm.OBJECTS_V_ANGLE + g.player]
    if _PAN_COST:
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
    # honest refuel: on the COMMITTED pass the enemies step (and drain) while the view pans
    # onto each fuel object, so absorbing while being drained cannot net-gain. Return the
    # REAL energy delta so a drained "refuel" registers as no progress (no_gain), not a
    # phantom bank. The coarse lookahead refuel stays instantaneous (approximate ranking).
    drain_aim = _REFUEL_DRAIN and not sweep_coarse
    e_start = g.energy
    for _z, slot, ot, tile in order:
        if tile in tiles_done:
            continue
        if g.mem[mm.OBJECTS_FLAGS + slot] & 0x80:  # already gone (stack collapsed)
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
        if drain_aim:
            view = sweep[tile]
            # Charge the WHOLE cost of one absorb: the aim pan (view scrolls onto the
            # fuel bearing) AND the execute/settle the action then spends before it is
            # confirmed (actioncost, the same figure run_plan_simulated advances the
            # real world by). While the player is seen the Sentinel drains -1 roughly
            # every drain-cooldown period THROUGHOUT that window, so a single tree
            # absorb (+1) taken while drained nets NEGATIVE -- absorbing "one tree at a
            # time while being drained" cannot climb out of a deficit, it dies.
            # Charging only the pan (as before) omitted the settle, so a nearby tree's
            # tiny pan let the drain cooldown never elapse and every tree booked a
            # phantom +1 -- the fabricated fuel that made the planner sit on a seen
            # tile refuelling fruitlessly. On a SAFE (unseen) tile no drain fires over
            # the same window, so legitimate refuelling still nets the full +1/tree.
            rounds = _pan_rounds(ch, cv, view) + actioncost.action_rounds(
                g.mem, "absorb", view
            )
            e_pre = g.energy
            _advance_enemies(g.state, int(round(rounds)), apply_drain=True)
            ch = view.get("h_angle", ch) if view else ch
            cv = view.get("v_angle", cv) if view else cv
            if actions.player_dead(g.state):
                break  # drained to death mid-refuel: credit no further absorb
            # RULE 3: you cannot recover energy by absorbing while drained. If the drain
            # over this absorb's aim+settle window already meets or exceeds the object's
            # value, the absorb nets <= 0 -- refuelling here loses energy. Stop the pass
            # (the player is seen; every further absorb bleeds too) rather than fund the
            # plan on a phantom bank. On a SAFE tile no drain fires, so this never trips.
            if e_pre - g.energy >= mm.ENERGY_IN_OBJECTS[ot]:
                break
        if not g.absorb(slot, sweep[tile], f"absorb {_FUEL_NAME[ot]} for fuel"):
            continue  # ROM action LOS gate rejected the aim -> no absorb, no credit
        gained += mm.ENERGY_IN_OBJECTS[ot]
        tiles_done.add(tile)
        log(f"    +fuel: absorbed {_FUEL_NAME[ot]} {tile}, energy {g.energy}")
    return g.energy - e_start if drain_aim else gained


def _launch_tiles(g, plat_ground):
    """The set of tiles with a CLEAR DIAGONAL to the plinth (README strategy 3): tiles that
    can see the platform tile from launch height. Computed as ONE reverse visibility sweep
    from the platform's own vantage at launch eye (plat_ground+1) -- terrain LOS is ~symmetric,
    so the tiles the plinth vantage sees are the tiles that can fire the endgame absorb at it.

    This is the climb's DIRECTION signal: it separates the winning side of the map (a foothold
    from which the plinth is reachable) from an equally-high DEAD-END corner that can never see
    it (the ls0 failure: the search fled to the tallest corner (5,16), launch_ready=False,
    instead of the corridor toward (12,4), launch_ready=True). Being a set membership, it does
    NOT create a distance gradient that would drag the climb off its covered edge into the
    exposed centre (the reason cheb-distance is the WRONG pull) -- edge_dist still governs cover.
    Empty when the platform slot is unavailable (bonus off -> prior height-only behaviour).
    """
    ps = g.state.slot_of_type(mm.T_PLATFORM)
    if ps is None:
        return frozenset()
    views = los.visible_tiles(g.state, ps, eye_z=plat_ground + 1, max_steps=200)
    return frozenset(views.keys())


def climb_ctx(g, toward_plat=False, near_plat_radius=0):
    """Fixed climb parameters + mutable progress state for the climb decision
    (search_iterate), derived from g's CURRENT state (so it works whether g is a fresh
    landscape or a live resync mid-climb)."""
    plat = g.plat
    plat_ground = g.plat_ground if g.plat_ground is not None else 8
    return {
        "plat": plat,
        "plat_ground": plat_ground,
        "target_z": plat_ground + 1,
        "launch_tiles": _launch_tiles(g, plat_ground),
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


def _platform_launch_view(g, plat, v_band=False):
    """The keyboard-aim view that lands the sights on the platform tile from g's CURRENT
    position, or None if no aim reaches it. A fractional eye above plat_ground sees DOWN
    onto the platform; ceil the observer so int(eye)==plat_ground doesn't drop it (matches
    endgame's seye, only when frac-eye is on). This is the endgame's real precondition:
    the platform synthoid can only be built where the sights can land on the platform.

    `v_band=True` sweeps the body v_angle too (the endgame looks DOWN onto the platform, a
    pitch below the $F5 default) -- COMPLETE but slower; used by the hard endgame/approach
    gate. The default (v=$F5) is the fast approximation for leaf-scoring launch-readiness.
    """
    ie = int(g.eye)
    seye = ie + 1 if (_ENDGAME_FRAC_EYE and g.eye > ie) else ie
    return los.landable_view(g.state, plat, g.player, eye_z=seye, v_band=v_band)


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
    # Need a keyboard aim that LANDS on the platform to absorb the Sentinel + build the
    # winning synthoid there -- not mere geometric LOS (which over-reported and let the
    # endgame "win" from an un-aimable cheb-1 launch in sim, a live failure). None -> the
    # platform can't be aimed from here; the climb must reach a real launch tile.
    view = _platform_launch_view(g, plat, v_band=True)
    if view is None:
        return False
    if not g.absorb(g.sentinel_slot, view, "absorb Sentinel"):
        return (
            False  # ROM action LOS gate: no real-eye LOS to the platform -> no launch
        )
    log(f"  absorbed Sentinel from eye {g.eye}, energy {g.energy}")
    if g.feasible(0, plat):
        g.transfer(
            g.create(0, plat, None, "platform synthoid"),
            "hyperspace onto platform (WIN)",
        )
        # RULE 1: the win is real only if the ROM win condition actually holds -- the
        # player must be standing on the platform tile ($3F in the flags chain). A
        # transfer that did NOT land on the platform (create failed, drained to death,
        # etc.) is NOT a win. Never declare native_won on the strength of "we fired the
        # sequence"; gate it on actions.on_platform of the real resulting state.
        if actions.on_platform(g.state) and not actions.player_dead(g.state):
            log(f"  WIN: synthoid on platform {plat} + transfer")
            return True
    return False


INF = float("inf")
_NOOP = lambda *a, **k: None

# --- search shape (SEARCH_REDESIGN.md sec.3, sec.7) --------------------------
DEFAULT_DEPTH = 3  # plies of lookahead; raise for landscapes that need it (sec.7)
DEFAULT_BEAM = 3  # top-N candidates expanded per node, independent of depth (sec.7)

# --- move-cost model: the REAL cost of moving, from the simulator (SEARCH_REDESIGN.md
# sec.6) ----------------------------------------------------------------------
# A move is not a single aim -- it is a KEYBOARD SEQUENCE, each aim panning the view from
# wherever the last one left it (verified against a live run's per-action views,
# out/ls0_pancost.log): aim+build the foothold (a boulder then a re-centred synthoid, or a
# lone synthoid), transfer, then SWING THE VIEW BACK to look down on the departed tile and
# reabsorb its shell. That return pan is often a ~180-degree bearing swing (h $90->$08 =
# 15 lattice steps in the log) and was the live killer the old flat TICKS_PER_HOP/STEP
# constants entirely ignored -- they charged every move the same regardless of how far the
# view actually had to scroll, so the search could not see that a grab on the opposite
# bearing holds the player exposed for many rounds while the Sentinel rotates onto it.
#
# _move_cost now derives each move's cost from the sentinel keyboard-aim GEOMETRY
# (sentinel.aimcost): the number of pan keystrokes for the whole sequence, converted to
# elapsed enemy rounds by two ROM-cadence scalars below. This feeds the bit-exact
# enemy-state advancement (_advance_enemies) so ticks_until_seen / meanie_safe / the drain
# forecast are all evaluated at the state the enemies REALLY reach while the move executes.
#
# enemy rounds (sentinel.enemies.step calls ~= frames) elapsed per keyboard action, from
# the ROM's own scroll cadence: the view scrolls one step per frame, and the scroll length
# for one keystroke is baked into the pan setup. A coarse +-8 BEARING keystroke animates a
# 16-step horizontal scroll ($10EE LDA #$10) -> ~16 rounds; a +-4 PITCH keystroke an 8-step
# scroll ($1135 LDA #$8) -> ~8 rounds; a fired create/absorb/transfer settles over its own
# plot cycle (~a scroll's worth). Bearing pans are thus ~2x costlier than pitch pans -- so
# the enemies rotate much further while the view swings across the map than while it tilts,
# which the old single flat per-move constant could not express. Overridable so a diverging
# live run can be recalibrated from its own telemetry without a code change (sec.9 step 4).
# Scroll frames -> enemy-round ticks via the shared $130C/$1335 Bresenham factor
# (actioncost.FRAME_TICKS == 0.80); the scene replot is folded into those frames.
ROUNDS_PER_H_STEP = float(
    os.environ.get("ROUNDS_PER_H_STEP", str(actioncost.FRAME_TICKS * 16))
)
ROUNDS_PER_V_STEP = float(
    os.environ.get("ROUNDS_PER_V_STEP", str(actioncost.FRAME_TICKS * 8))
)
ROUNDS_PER_ACTION = float(os.environ.get("ROUNDS_PER_ACTION", "16"))
# A U-turn (EOR $80) flips the bearing 180 degrees in ONE keystroke, so a far-bearing swing
# costs one U-turn + a short +-8 correction instead of up to sixteen +-8 pans -- which the
# live aim driver now does (kbd_aim.coarse_h). Cost it the same here so the planner does not
# over-charge a big bearing swing the driver will actually shortcut (~one keystroke's worth
# of plot; kept equal to a bearing pan step so the keystroke and rounds crossovers coincide
# at a bearing >= 72 units).
ROUNDS_PER_UTURN = float(
    os.environ.get("ROUNDS_PER_UTURN", str(actioncost.FRAME_TICKS * 16))
)

# ticks_until_seen / meanie forward-sim horizons: bounded so the per-decision search
# stays inside the 60s script budget. The safety margin only needs to distinguish
# "seen soon" from "safe for a while"; a short horizon suffices as a tie-break term.
_SAFETY_HORIZON = 48
# ray-march cap for the boulder centre-aim feasibility probe. The default 2000 is for
# the far offline validator; a boulder-step's target tile is by construction LOS-visible
# and near, so a short march reaches it -- and this probe dominates search cost.
_CENTRE_MAX_STEPS = 200
# ray-march cap for the coarse refuel sweep inside the lookahead (approximate energy only).
_REFUEL_MAX_STEPS = 160
# extra height-ranked candidates (beyond `beam`) to compute the enemy safety margin for
# and re-rank -- safety only reorders near-equal heights, so a small shortlist suffices.
_SHORTLIST = 4

# GAZE-AWARE exposure (opt-in via env; default off preserves the ROM-validated ls42 win).
# The static _enemy_exposed_tiles mask is "could the Sentinel EVER see this tile at any
# rotation" -- near the platform that's EVERY high tile, so any real EXPOSURE_RESERVE
# stalls the climb. With gaze-awareness we instead penalize only tiles the Sentinel is
# looking toward WITHIN one build/rotation window (ticks_until_seen < horizon), so a far
# safe high tile -- or a ring tile while the Sentinel faces away -- stays affordable, but
# a tile in its current/imminent view is avoided. This is the human ls0 win's actual
# tactic: not "avoid what it could see" but "be where it isn't looking". Pair with a
# larger CLIMB_EXPOSURE_RESERVE (now safe: it hits only currently-unsafe tiles).
# RULE 4: gaze is deterministic and schedulable. Default ON. The static
# _enemy_exposed_tiles mask ("could the Sentinel EVER see this tile at any rotation") is
# far too pessimistic near good high ground -- on ls0 nearly every high tile is
# ever-visible in principle yet has ticks_until_seen == horizon (the actual gaze
# precession never sweeps to its bearing for a long time). Gating on the FORECAST gaze
# (ticks_until_seen < horizon) instead keeps a briefly-visible high foothold usable
# during its long safe windows, exactly the lever README strategy 4 describes.
_GAZE_AWARE_COST = os.environ.get("GAZE_AWARE_COST", "1") == "1"
_GAZE_COST_HORIZON = int(os.environ.get("GAZE_COST_HORIZON", "250"))
# Treat a FRACTIONAL eye above the platform ground as launch-ready. The default gate is
# int(eye) > plat_ground, which discards the 0.875/0.5 fraction: a climb that tops out at
# eye 8.875 (genuinely above a z8 platform, LOS intact) is rejected and the search creeps
# on -- into the ring -- chasing int(eye)>=9. With the no-build ring closing that off, the
# far staircase strands at 8.875. Comparing the true float (8.875 > 8) recognises it as a
# valid long-range launch. RULE 5 (gain height): a natural z8 terrain tile already gives
# eye 8.875 > a z8 platform ground, so recognising the fraction lets the climb launch from
# natural high ground without over-building into the exposed ring. Default ON.
_ENDGAME_FRAC_EYE = os.environ.get("ENDGAME_FRAC_EYE", "1") == "1"
# SEEN-DRAIN: model being seen as a COSTED, survivable state rather than a hard veto. As
# the search simulates each move it debits the energy the player would actually lose while
# dwelling at the destination during that move (sentinel.enemies.step: 1 energy per
# ~120 seen ticks, nothing for a quick transit). Routes that linger in view bleed energy
# and score lower; brief crossings are free -- so the planner can TRANSIT unavoidable seen
# tiles (multi-sentry landscapes) and is pushed to reach a safe tile fast. RULES 1 & 3:
# exposure is a timed, survivable energy cost -- the lookahead must price it (and prune a
# move whose window drains the player to death). Default ON so the offline planner's
# forecast matches the tick-accurate runner's real drain.
_SEEN_DRAIN = os.environ.get("SEEN_DRAIN", "1") == "1"
# Advance the COMMITTED enemy state by each move's real tick cost (not just refuel), so
# the planner's own rotation forecast matches the game's. When ON, plan_search's Sentinel
# facing tracks reality (h=248 at the (11,8) node, matching the sim/live runners) and the
# offline climb HONESTLY dead-ends in the same (11,8) drain-trap instead of false-winning
# against a slow enemy clock. Default OFF pending the routing overhaul that avoids the trap
# (a real win under honest timing needs the human-style low-exposure route, not this fix).
_COMMIT_MOVE_TICKS = os.environ.get("COMMIT_MOVE_TICKS", "0") == "1"
# REFUEL-DRAIN: charge the drain incurred while RE-AIMING at each fuel object. A refuel is
# not free -- every absorb needs a keyboard pan onto the target, and if the Sentinel is
# currently draining the player, that pan bleeds energy the whole time. So absorbing trees
# to "catch up" while being drained nets <=0 (each +1 tree is offset by the drain over its
# re-aim), which the old instantaneous _refuel (credit every absorb, advance nothing) got
# wrong -- it fabricated the energy that funded the ls0 energy-1 endgame the live player
# (drained the whole time) can never bank. When on, the COMMITTED refuel steps the enemy
# state (bit-exact drain) over each re-aim pan before crediting the absorb, so the planner's
# banked energy is what the drained player could REALLY earn. RULE 3: you cannot recover
# by absorbing low-value trees while drained (each absorb spans multiple drain periods ->
# net negative), so the planner must not fabricate that energy. Default ON.
_REFUEL_DRAIN = os.environ.get("REFUEL_DRAIN", "1") == "1"


def _pan_rounds(cur_h, cur_v, view):
    """Enemy rounds to aim from heading (cur_h, cur_v) at `view`: the bit-exact keyboard
    lattice distance (sentinel.aimcost) weighted by the per-axis scroll cadence (bearing
    keystrokes cost ROUNDS_PER_H_STEP, pitch keystrokes ROUNDS_PER_V_STEP). 0 when the
    heading is unknown or the view carries no angle."""
    if cur_h is None or not view:
        return 0.0
    vh, vv = view.get("h_angle"), view.get("v_angle")
    r = (
        ac.bearing_rounds(cur_h, vh, ROUNDS_PER_H_STEP, ROUNDS_PER_UTURN)
        if vh is not None
        else 0.0
    )
    if vv is not None and cur_v is not None:
        r += ac.v_steps(cur_v, vv) * ROUNDS_PER_V_STEP
    return r


def _move_cost(g, c, cur_h, cur_v):
    """The REAL cost of executing candidate `c` from view heading (cur_h, cur_v):
    (ticks, end_h, end_v). Mirrors _apply's keyboard sequence and prices it from the
    keyboard-aim geometry (sentinel.aimcost) rather than a flat per-move constant.

    Sequence (see out/ls0_pancost.log): pan to the build tile and fire (a boulder then a
    re-centred on-boulder synthoid, or a lone synthoid), transfer (no pan -- already
    looking at the tile), then SWING THE VIEW BACK to look down on the departed tile and
    reabsorb its shell. The return swing is a near-180-degree bearing pan and is the bulk
    of a move's exposure window -- the cost the flat model missed. Returns the ending
    heading so the search can chain the next move's pan from where this one left the view.
    """
    T2, use_b, _he, view = c
    prev = g.player_xy()
    vh = view.get("h_angle") if view else None
    vv = view.get("v_angle") if view else None
    rounds = _pan_rounds(cur_h, cur_v, view)  # aim the build tile
    # Price each fire by its per-verb SETTLE floor (sentinel.actioncost), shared with
    # the simulated runner's world advance so the enemy forward-sim rotates/drains by
    # the SAME amount the real action costs -- the flat ROUNDS_PER_ACTION under-counted
    # a fire ~15x, so the planner forecast a route the real drain could not survive.
    # A stacked create (on an occupied column) costs STACK_CREATE more (taller redraw).
    if use_b:
        n, _ = _boulder_batch(g, T2)  # RULE 5: the whole boulder batch is priced
        n = max(1, n)
        rounds += ROUNDS_PER_H_STEP + ROUNDS_PER_V_STEP  # re-centre on-boulder synthoid
        first_stacked = T2 in g.col  # first boulder stacks if the tile is already built
        settle = actioncost.SETTLE["create"] + (
            actioncost.STACK_CREATE if first_stacked else 0.0
        )
        # boulders 2..n and the capping synthoid all stack -> STACK_CREATE each.
        settle += n * (actioncost.SETTLE["create"] + actioncost.STACK_CREATE)
    else:
        settle = actioncost.SETTLE["create"]  # a lone synthoid on terrain/column
    settle += actioncost.SETTLE[
        "transfer"
    ]  # the transfer up (was omitted -> under-count)
    end_h, end_v = vh, vv
    back_h = ac.bearing_to(T2[0], T2[1], prev[0], prev[1])  # look back at departed tile
    if back_h is not None and end_h is not None:
        # the return-pan (bearing), U-turn-aware: a swing back past half a turn is one
        # U-turn + a short correction, matching what the live driver keys.
        rounds += ac.bearing_rounds(end_h, back_h, ROUNDS_PER_H_STEP, ROUNDS_PER_UTURN)
        end_h = back_h
        settle += actioncost.SETTLE["absorb"]  # reabsorb-shell confirm
    ticks = int(round(rounds + settle))
    return ticks, end_h, end_v


def _cost(g, c, exposed):
    """Up-front energy a candidate needs before the shell-reabsorb refund: a boulder-step
    of n batched boulders costs 2*n + 3 (n boulders at 2 + synthoid at 3), a hop 3, plus
    RESERVE, plus EXPOSURE_RESERVE when the destination is enemy-exposed."""
    if c[1]:
        n, _ = _boulder_batch(g, tuple(c[0]))
        base = 2 * max(1, n) + 3
    else:
        base = 3
    return base + RESERVE + (EXPOSURE_RESERVE if tuple(c[0]) in exposed else 0)


def _read_state(g):
    return g.state


def _advance_enemies(state, ticks, apply_drain):
    """Advance the node's sentinel enemy state in place by `ticks` game rounds
    (SEARCH_REDESIGN.md sec.6). Enemy timing (rotation/cooldown) lives inside the State and
    is stepped bit-exact by sentinel.enemies.step, so a move behind a long scroll rotates
    the Sentinel further (foreseeing mid-pan exposure). `ticks` is the base action cost plus
    the pan time to aim at the move. When `apply_drain` is on the stepping debits the energy
    the player would lose while dwelling in view during the window (seen tiles are PRICED,
    not vetoed); when off, energy is restored so only the rotation forecast is kept (the
    default ROM-validated accounting)."""
    e0 = state.energy
    for _ in range(ticks):
        SE.step(state)
    if not apply_drain:
        state.energy = (
            e0  # rotation forecast only; restore energy when seen-drain is off
        )


def _reached_approach(g, ctx):
    """Terminal check ('won(g)' in the sec.7 pseudocode): can the endgame (absorb Sentinel
    + platform synthoid + transfer) launch from HERE? That needs the eye above the platform
    ground, the Sentinel present, and LINE OF SIGHT to the platform tile -- NOT adjacency.
    A synthoid hop lands on any LOS tile, so the win is a long-range hop onto the platform
    from wherever the climb topped out (the human ls0 win fired from cheb-10). This is the
    key to 'gain height as fast as possible': the climb never has to creep to the platform
    ring (where the meanie spawns); it just gets high with LOS, then hops on."""
    if g.plat_ground is None or g.sentinel_slot is None:
        return False
    above = g.eye > g.plat_ground if _ENDGAME_FRAC_EYE else int(g.eye) > g.plat_ground
    if not above:
        return False
    # The endgame builds a synthoid ON the platform tile, so the launch position must be
    # able to AIM at it -- adjacency is NOT sufficient (a cheb<=1 platform is in the sights
    # foreground, un-aimable; the human ls0 win launched from cheb-9). Gate on the SAME
    # aim-landability the endgame requires so the handoff never breaks the loop on a launch
    # the endgame can't actually fire (the (11,3) cheb-1 mirage). Only reached once the eye
    # is above the platform (late climb), so this per-node sweep runs on very few nodes.
    return _platform_launch_view(g, ctx["plat"], v_band=True) is not None


def _gen_candidates(g, ctx, blocked, state, beam, cur_h=None, cur_v=None):
    """Ordered successor list for a search node: the real _candidates footholds, cheaply
    hard-filtered (affordability, blocklist, meanie-safety) and sorted best-first
    (height, safety margin, edge distance -- the human-validated priority, sec.7).

    Two costs are deliberately NOT paid here: `_boulder_centre_feasible` (a full LOS
    lattice per boulder-step) is checked LAZILY by the caller while it fills `beam`
    feasible children -- so at most ~beam of them run, not one per candidate; and
    `ticks_until_seen` (enemy forward-sim) is computed only for the top `_SHORTLIST`
    height-ranked candidates, since it only refines ordering among near-equal heights.
    The full ordered list is returned so the caller can skip past boulder-infeasible
    entries; it does its own `beam` truncation."""
    cur = g.player_xy()
    eye = int(g.eye)
    raw = _candidates(g, cur, eye)
    if not raw:
        return []
    # NON-REGRESSION (SEARCH_REDESIGN.md sec.1/sec.9): only ever consider footholds whose
    # resulting eye is >= the current eye. The search selects a line by its deepest leaf
    # score, so without this a down-move that later reaches a higher peak could be
    # committed -- exactly the height regression the human never made and the redesign is
    # meant to eliminate. Enforcing it structurally (not just emergently) also prunes the
    # candidate set. A node with only down-moves returns [] -> a genuine dead end.
    raw = [c for c in raw if c[2] >= g.eye - 1e-9]
    if not raw:
        return []
    # ANTI-OSCILLATION: never step back onto an exact (tile, height) already stood on this
    # climb (ctx["visited"], grown on committed real steps). With non-regression above, a
    # tile can then only be revisited at a STRICTLY greater height, so no equal-height
    # reposition cycle is possible (the greedy (2,6)<->(1,7)-forever failure) -- each
    # (tile,height) is used at most once, a finite budget, so the climb must make height
    # progress or dead-end. Applied before affordability so a pruned-but-affordable gaining
    # move can't mask the plateau (the bug that let the cycle survive an earlier guard).
    raw = [c for c in raw if (tuple(c[0]), round(c[2], 2)) not in ctx["visited"]]
    if not raw:
        return []
    # RULE 2: there is NO distance "danger ring". A foothold's safety is decided purely by
    # line of sight / the Sentinel's forecast gaze to that specific tile (below, via the
    # gaze-filtered `exposed` set + meanie_safe), never by Chebyshev distance to the plinth
    # -- a boulder built adjacent to the platform while the Sentinel faces away is safe and
    # is a valid penultimate winning position. So no cheb-radius create filter here.
    # affordability + blocklist (cheap, no sweeps). The exposure set is either the static
    # could-ever-be-seen LOS mask (default) or, when gaze-aware, only the subset the
    # Sentinel is actually looking toward within a build/rotation window (see _cost notes).
    exposed = _enemy_exposed_tiles(g, {tuple(c[0]) for c in raw})
    if _GAZE_AWARE_COST:
        exposed = {
            t
            for t in exposed
            if threat.ticks_until_seen(state, t[0], t[1], horizon=_GAZE_COST_HORIZON)
            < _GAZE_COST_HORIZON
        }
    pool = [
        c
        for c in raw
        if g.energy >= _cost(g, c, exposed)
        and (tuple(c[0]), c[1]) not in blocked
        and (tuple(c[0]), c[1]) not in ctx["runtime_blocked"]
    ]
    if not pool:
        return []
    # meanie-safe HARD filter: don't dwell somewhere a tree could convert into a meanie
    # that then forces a hyperspace, IF a safe alternative exists (sec.5). Fall back to
    # the full pool only when nothing is meanie-safe (better a risky climb than none).
    safe = [c for c in pool if threat.meanie_safe(state, tuple(c[0]))]
    pool = safe or pool
    # cheap best-first order MIRRORING _evaluate so the beam keeps the winning lines, not just
    # the highest-climbing ones: capped height + LAUNCH-READINESS (clear diagonal to the plinth)
    # dominate, so a launch-ready corridor foothold is kept over a taller DEAD-END corner; edge
    # cover and the tiny cheb pull are low-order tie-breaks. Then compute the enemy safety
    # margin for the top shortlist and re-rank those.
    plat = ctx["plat"]
    launch = ctx["launch_tiles"]
    tz = ctx["target_z"]

    def _base(c):
        t = tuple(c[0])
        return (
            min(c[2], tz) * _W_EYE
            + (_W_LAUNCH if t in launch else 0.0)
            - cheb(t, plat) * _W_PLAT
            + edge_dist(t) * _W_EDGE
        )

    pool.sort(key=_base, reverse=True)
    head = pool[: beam + _SHORTLIST]
    safety = {
        tuple(c[0]): threat.ticks_until_seen(
            state, c[0][0], c[0][1], horizon=_SAFETY_HORIZON
        )
        for c in head
    }
    # pan keystrokes to AIM each head candidate's build tile from the current heading
    # (0 at the root when the heading is unknown). Prefer the smaller pan among otherwise
    # equal candidates: aiming a far-bearing target holds the player exposed on its current
    # tile for the whole scroll, and lets the Sentinel rotate further onto it.
    pan = {tuple(c[0]): _pan_rounds(cur_h, cur_v, c[3]) for c in head}
    # add the safety (gaze) margin and least-pan preference as sub-tie-breaks below the base
    # score, so among near-equal launch/height footholds the safer, cheaper-to-aim one wins.
    head.sort(
        key=lambda c: _base(c)
        + min(safety[tuple(c[0])], 255) * _W_SAFETY
        - pan[tuple(c[0])] * _W_PAN,
        reverse=True,
    )
    return head + pool[beam + _SHORTLIST :]


# leaf-eval weights, strictly lexicographic. HEIGHT is dominant, by orders of magnitude:
# the one rule is GAIN HEIGHT AS FAST AS POSSIBLE, so a 0.5-eye step (5e8) must swamp every
# lower term's whole span. Platform-proximity is only a FAR-BELOW tie-break (span 62*1e3 =
# 6.2e4 << 5e8) among EQUAL-height leaves -- just enough to drift toward a high tile that
# can see the platform for the endgame hop, never enough to trade a height gain for it.
# (An earlier version weighted cheb-to-plat == height and produced timid +0.5 creeping
# moves through the meanie ring -- exactly the failure the user called out.)
_W_EYE = 1_000_000_000.0  # height reached (capped at launch height): strictly dominant
# LAUNCH-READINESS (README strategy 3, the climb's DIRECTION signal): a tile with a clear
# diagonal to the plinth (ctx["launch_tiles"], the reverse-sweep set) is worth ~1.5 height
# units, so a launch-ready foothold outscores a DEAD-END corner up to ~1.5 units taller that
# can never see the plinth -- the ls0 fix (the search was fleeing to the tallest corner and
# stranding). Paired with the height CAP at target_z below: over-climbing a non-ready corner
# past launch height earns nothing, so raw height can't out-run launch-readiness near the top.
# Height still dominates when the gap is large (a launch-ready tile >1.5 units lower than the
# best non-ready option loses), so the climb never stalls low on the ready side. Set > _W_EYE.
_W_LAUNCH = float(os.environ.get("CLIMB_W_LAUNCH", "1500000000"))
# LOS-to-platform launch-readiness: a high tile is only useful if it can SEE the plinth
# (README strategy 3 -- a clear line to the platform, not raw height). Worth ~0.4 units of
# eye, so near launch height it redirects the climb from a dead-end high ridge with no
# platform line to a slightly-lower tile that overlooks the plinth, but never overrides a
# genuine >=0.5-unit height gain.
_W_LOS_PLAT = 400_000_000.0
# platform-ward pull: a goal-direction gradient (which way the launch is) among near-equal
# heights, weighted well below a real 0.5-unit height gain (5e8) over the ~30-tile span so
# height still dominates. NOT a distance safety gate (RULE 2 -- exposure safety is the
# gaze/drain model, never distance to the plinth). Kept TINY: a strong distance pull would
# drag the climb off its covered edge/corner into the exposed centre (README strategy 3);
# the real direction signal is LOS-to-platform (clear diagonal), _W_LAUNCH below, not cheb.
_W_PLAT = 1_000_000.0
_W_SAFETY = 100.0  # max span 255*1e2 = 2.55e4
_W_EDGE = 1.0  # max span ~62
_W_ENERGY = 0.01
_W_PAN = (
    0.1  # least-pan preference; kept below one safety tick (100) so safety wins first
)


def _sees_plat(g, plat):
    """Whether the player at its current tile/eye can AIM at the platform tile (the
    endgame-launch precondition, and the leaf-scoring launch-readiness signal). Uses the
    same aim-landability the endgame requires, so the score pulls the climb toward tiles
    the endgame can actually fire from, not merely tiles with geometric LOS."""
    return _platform_launch_view(g, plat) is not None


def _evaluate(g, ctx):
    """Score a cut leaf (depth exhausted, not yet won): height reached DOMINATES (gain
    height as fast as possible), then platform proximity / ticks_until_seen (safety) / edge
    / energy purely as tie-breaks among equal-height leaves (SEARCH_REDESIGN.md sec.7).
    """
    cur = g.player_xy()
    plat = ctx["plat"]
    state = _read_state(g)
    safety = threat.ticks_until_seen(state, cur[0], cur[1], horizon=_SAFETY_HORIZON)
    # Reward LINE OF SIGHT to the platform (launch-readiness) once the climb is near launch
    # height -- a high tile is only useful if it overlooks the plinth (README strategy 3),
    # so this redirects the late climb from a dead-end high ridge with no platform line to a
    # tile that can actually fire the endgame. Only computed near plat_ground (cost bound),
    # and worth < a 0.5-unit height gain so it never trades away real height. Among EQUAL-
    # height leaves, proximity is a mild goal-direction gradient (not a safety gate -- RULE
    # 2: safety is the gaze-filtered exposure set + drain, never distance to the plinth).
    los_bonus = 0.0
    if abs(g.eye - ctx["plat_ground"]) <= 2.5 and _sees_plat(g, plat):
        los_bonus = _W_LOS_PLAT
    # DIRECTION signal: a foothold with a clear diagonal to the plinth (README strategy 3).
    # Worth > 1 height unit so it outranks a taller dead-end corner, but height is CAPPED at
    # target_z so over-climbing a non-ready corner earns nothing past launch height.
    launch_bonus = _W_LAUNCH if cur in ctx["launch_tiles"] else 0.0
    return (
        min(g.eye, ctx["target_z"]) * _W_EYE
        + launch_bonus
        + los_bonus
        - cheb(cur, plat) * _W_PLAT
        + min(safety, 255) * _W_SAFETY
        + edge_dist(cur) * _W_EDGE
        + g.energy * _W_ENERGY
    )


def _lookahead(
    g, ctx, blocked, depth, beam, stats, is_root=True, cur_h=None, cur_v=None
):
    """Best-first bounded lookahead (SEARCH_REDESIGN.md sec.7). Returns
    (score, first_move): the score of the best line from this node and the candidate to
    commit at THIS node. A won node scores +inf; a node with no non-dead-end successor
    scores -inf and is pruned by its parent -- the guarantee greedy lacked (a move is
    never committed unless it has a continuation within the horizon).

    `_boulder_centre_feasible` (the dominant per-node cost, a full LOS lattice) is paid
    only at the ROOT ply -- the move actually committed MUST be keyboard-feasible, but
    deeper plies stay optimistic about boulder centre-aim and let the next real
    re-plan discover any infeasible continuation (the loop re-searches every step)."""
    if _reached_approach(g, ctx):
        return INF, None
    if depth <= 0:
        return _evaluate(g, ctx), None
    state = _read_state(g)
    cands = _gen_candidates(g, ctx, blocked, state, beam, cur_h, cur_v)
    stats["nodes"] += 1
    if not cands:
        return -INF, None  # dead end: no reachable/feasible foothold at all
    best_score, best_first, expanded = -INF, None, 0
    for c in cands:
        if expanded >= beam:
            break
        # LAZY boulder centre-feasibility (sec.7), root ply only: skip a boulder-step
        # whose on-boulder synthoid has no keyboard-reachable centre-aim, without
        # counting it against the beam -- so `beam` FEASIBLE children get expanded.
        if (
            is_root
            and c[1]
            and not _boulder_centre_feasible(g, c[0], c[3], max_steps=_CENTRE_MAX_STEPS)
        ):
            continue
        expanded += 1
        g2 = g.clone()
        _apply(g2, c[0], c[1], c[3])
        # model the fuel recovery that funds the next step, with a CHEAP coarse sweep --
        # the lookahead only needs approximate energy for affordability, and this sweep
        # is the per-node hot spot (the committed root refuel in search_iterate stays fine).
        _refuel(g2, _NOOP, sweep_max_steps=_REFUEL_MAX_STEPS, sweep_coarse=True)
        # advance g2's in-state enemy timing by this move's REAL tick cost (aim + build +
        # transfer + the return-pan to reabsorb the shell, priced from the keyboard-aim
        # geometry) so deeper plies see the Sentinel rotated by how far it actually turns
        # while the move executes. The ending heading chains into the child so its own moves
        # pan from where this one left the view. With _SEEN_DRAIN on, the stepping debits
        # g2.state.energy while the player dwells in view over that real window (a long-pan
        # route bleeds more energy, scores lower); with it off, energy is restored so only
        # the rotation forecast is kept (the ROM-validated default accounting).
        ticks, nh, nv = _move_cost(g, c, cur_h, cur_v)
        _advance_enemies(g2.state, ticks, apply_drain=_SEEN_DRAIN)
        # a move whose real drain window KILLS the player (drained at 0, or a meanie's
        # forced hyperspace) is a dead end -- never commit to it, so the search routes
        # around the exposed-launch drain instead of walking into it. Only bites when the
        # forecast actually applies drain (_SEEN_DRAIN); with drain off the state cannot die.
        if actions.player_dead(g2.state):
            continue
        sub_score, _ = _lookahead(
            g2,
            ctx,
            blocked,
            depth - 1,
            beam,
            stats,
            is_root=False,
            cur_h=nh,
            cur_v=nv,
        )
        if sub_score == -INF:
            continue  # dead-end line -- PRUNE (never commit to it)
        if sub_score > best_score:
            best_score, best_first = sub_score, c
    if best_first is None:
        return -INF, None  # every successor dead-ends: propagate the dead end up
    return best_score, best_first


def search_iterate(g, ctx, blocked, log, depth=DEFAULT_DEPTH, beam=DEFAULT_BEAM):
    """ONE lookahead decision against g's CURRENT state (status strings, ctx
    bookkeeping and g.steps side effects match the shared climb-decision contract).
    Banks fuel, checks the terminal/no-gain conditions, then runs _lookahead and
    COMMITS the first move of the best line.

    Returns: 'retry' (refuel-only pass; call again), 'stepped' (a foothold move was
    applied), 'no_gain' (no height/fuel progress in 16 iterations), 'approach' (reached
    the platform approach; hand to endgame), 'stuck' (no feasible line)."""
    plat = ctx["plat"]
    cur = g.player_xy()
    eye = int(g.eye)
    # bank fuel first (same as greedy): a deliberate absorb streak counts as progress.
    gained = _refuel(g, log)
    # A refuel that the drain-honest model runs the player out of energy (seen tile,
    # drained faster than trees bank) is a LOST game, not progress: don't loop retries
    # on it and don't hand a dead state to the endgame (which would report a false win).
    if _REFUEL_DRAIN and actions.player_dead(g.state):
        log(
            f"  refuel drained player to death at {cur} (seen while banking fuel); stuck"
        )
        return "stuck"
    if g.eye > ctx["peak_eye"] + 1e-9:
        ctx["peak_eye"] = g.eye
        ctx["no_gain"] = 0
    elif gained > 0:
        ctx["no_gain"] = 0
    else:
        ctx["no_gain"] += 1
    if ctx["no_gain"] > 16:
        log(f"  no height/fuel progress in 16 steps (peak {ctx['peak_eye']:.2f}); stop")
        return "no_gain"
    if _reached_approach(g, ctx):
        log(
            f"  reached win-launch state: {cur} eye {eye} "
            f"(d={cheb(cur, plat)}, LOS to platform); hand to endgame"
        )
        return "approach"

    stats = {"nodes": 0}
    t0 = time.time()
    # Seed the root view heading from the player's current facing so the first move's aim
    # pan is costed from where the view actually points. Live this is the resynced real
    # heading; offline it is the generator's starting facing. The search threads the ending
    # heading of each committed move onward (see _move_cost), so deeper plies never depend
    # on RAM (native create/absorb don't write mm.OBJECTS_H_ANGLE) -- only this root read touches it.
    root_h, root_v = (
        g.mem[mm.OBJECTS_H_ANGLE + g.player],
        g.mem[mm.OBJECTS_V_ANGLE + g.player],
    )
    score, move = _lookahead(
        g, ctx, blocked, depth, beam, stats, cur_h=root_h, cur_v=root_v
    )
    dt = time.time() - t0

    if move is None:
        # no line survives: either genuinely stuck, or just out of affordable fuel this
        # pass (retry so the next resync/refuel can discover deferred fuel -- mirrors
        # the out-of-footholds retry branch, incl. the no_gain loop bound).
        if g.energy < 6:
            log(
                f"  lookahead found no line from {cur} eye {eye} "
                f"(energy {g.energy}, {stats['nodes']} nodes, {dt:.2f}s); retrying"
            )
            return "retry"
        log(
            f"  lookahead: NO feasible line from {cur} eye {eye} "
            f"(energy {g.energy}, {stats['nodes']} nodes, {dt:.2f}s) -- stuck"
        )
        return "stuck"

    T2, use_b, h_after, view = move
    ctx["visited"].add((T2, round(h_after, 2)))
    ctx["tiles_used"].add(T2)
    # Advance the COMMITTED enemy state by this move's real duration (aim + build +
    # transfer + return-pan), the SAME tick cost the lookahead priced on its clones
    # (_move_cost / _advance_enemies at the eval ply). Without this the committed clock
    # ran slow -- it ticked enemies only on refuel, never on the move itself -- so the
    # planner's own g drifted behind the true Sentinel rotation and it committed routes
    # that die once the enemy actually turns (the (11,8) live/sim dead-end). Priced from
    # the PRE-move state (player still on the departed tile).
    if _COMMIT_MOVE_TICKS:
        mticks, _, _ = _move_cost(g, move, root_h, root_v)
    _apply(g, T2, use_b, view)
    if _COMMIT_MOVE_TICKS:
        _advance_enemies(g.state, mticks, apply_drain=_SEEN_DRAIN)
    log(
        f"  {'step' if use_b else 'hop '} -> {T2} eye {g.eye} "
        f"(d={cheb(g.player_xy(), plat)}) edge={edge_dist(T2):.0f} energy {g.energy} "
        f"[best {score:.3g}, {stats['nodes']} nodes, {dt:.2f}s]"
    )
    return "stepped"


def plan_search(
    landscape,
    verbose=True,
    max_steps=120,
    blocked=frozenset(),
    start_energy=None,
    toward_plat=True,
    near_plat_radius=2,
    depth=DEFAULT_DEPTH,
    beam=DEFAULT_BEAM,
):
    """Offline driver: runs the lookahead decision (search_iterate) to a win, then
    endgame().

    Defaults to the LIVE executor's config (toward_plat=True, near_plat_radius=2, see
    run_plan_live.execute_live): the lookahead already drives toward the platform via
    _evaluate's proximity term, and toward_plat gates the ROM-infeasible on-distant-
    boulder synthoid in the steep platform ring (_candidates)."""
    t0 = time.time()
    g = PlanGame(landscape)
    if start_energy is not None:
        g.energy = start_energy
    log = lambda *a: verbose and print(*a)
    ctx = climb_ctx(g, toward_plat, near_plat_radius)
    log(
        f"search ls{landscape} D{depth} B{beam}: start {g.player_xy()} eye {g.eye} "
        f"plat {ctx['plat']} plat_ground {ctx['plat_ground']} energy {g.energy}"
    )
    for _ in range(max_steps):
        status = search_iterate(g, ctx, blocked, log, depth=depth, beam=beam)
        if status in ("no_gain", "approach", "stuck"):
            break
    won = endgame(g, ctx["plat"], log)
    g.native_won = won
    log(
        f"=== search {'WON' if won else 'INCOMPLETE'} in {time.time()-t0:.2f}s, "
        f"{len(g.steps)} steps, final {g.player_xy()} eye {g.eye} ==="
    )
    return g


if __name__ == "__main__":
    ls = int(sys.argv[1]) if len(sys.argv) > 1 else 0
    # CLI default depth is 2, not DEFAULT_DEPTH (3): a full offline climb is many
    # decisions and must fit the 60s script budget (ls0 wins at D2 in ~53s; D3 is the
    # per-decision ceiling for harder landscapes in the live loop, sec.7). Override:
    # `climb_search.py <ls> <depth> <beam>`.
    d = int(sys.argv[2]) if len(sys.argv) > 2 else 2
    b = int(sys.argv[3]) if len(sys.argv) > 3 else DEFAULT_BEAM
    g = plan_search(ls, depth=d, beam=b)
    out = {
        "landscape": ls,
        "native_won": g.native_won,
        "final_player": g.player_xy(),
        "eye": g.eye,
        "energy": g.energy,
        "steps": g.steps,
    }
    json.dump(out, open(f"out/kbd_search_{ls:04d}.json", "w"), indent=0)
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
