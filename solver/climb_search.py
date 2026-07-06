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
    PlanGame as Game,
    cheb,
    visibility_sweep,
    terrain_z,
    OBJ_X,
    OBJ_Y,
    OBJ_Z,
    OBJ_TYPE,
    OBJ_ZF,
    OBJ_FLAGS,
    OBJ_HANG,
    OBJ_VANG,
)
from sentinel import los, threat, enemies as SE, aimcost as ac, actioncost

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
        g.create(3, T2, view, "climb boulder")
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
    from solver.plan_game import ENERGY

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
    """Fixed climb parameters + mutable progress state for the climb decision
    (search_iterate), derived from g's CURRENT state (so it works whether g is a fresh
    landscape or a live resync mid-climb)."""
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
ROUNDS_PER_H_STEP = float(os.environ.get("ROUNDS_PER_H_STEP", "16"))
ROUNDS_PER_V_STEP = float(os.environ.get("ROUNDS_PER_V_STEP", "8"))
ROUNDS_PER_ACTION = float(os.environ.get("ROUNDS_PER_ACTION", "16"))
# A U-turn (EOR $80) flips the bearing 180 degrees in ONE keystroke, so a far-bearing swing
# costs one U-turn + a short +-8 correction instead of up to sixteen +-8 pans -- which the
# live aim driver now does (kbd_aim.coarse_h). Cost it the same here so the planner does not
# over-charge a big bearing swing the driver will actually shortcut (~one keystroke's worth
# of plot; kept equal to a bearing pan step so the keystroke and rounds crossovers coincide
# at a bearing >= 72 units).
ROUNDS_PER_UTURN = float(os.environ.get("ROUNDS_PER_UTURN", "16"))

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
_GAZE_AWARE_COST = os.environ.get("GAZE_AWARE_COST", "0") == "1"
_GAZE_COST_HORIZON = int(os.environ.get("GAZE_COST_HORIZON", "250"))
# HARD no-build ring: forbid CREATE footholds within this Chebyshev radius of the
# platform during the climb. A boulder/synthoid built in the Sentinel's face (cheb<=2)
# sits in its view cone and is absorbed on the next scan -- which spawns meanies and
# drains the player even when the player itself is unseen (gaze-timing can't help; the
# hazard is the OBJECT's exposure, not the player's). The human ls0 win never built in
# the ring: it climbed to launch height at a far tile (cheb-10) and struck from range.
# This forces launch tiles to cheb>radius, replicating that as a constraint (cheb>radius
# tiles stay available, so unlike a cost crank it won't collapse the candidate set).
# Plain hops onto EXISTING ring objects stay allowed; the endgame (absorb Sentinel +
# platform synthoid, fired after the Sentinel is gone) runs in endgame() and is not
# gated here. 0 = off (default).
_RING_NOBUILD_RADIUS = int(os.environ.get("RING_NOBUILD_RADIUS", "0"))
# Treat a FRACTIONAL eye above the platform ground as launch-ready. The default gate is
# int(eye) > plat_ground, which discards the 0.875/0.5 fraction: a climb that tops out at
# eye 8.875 (genuinely above a z8 platform, LOS intact) is rejected and the search creeps
# on -- into the ring -- chasing int(eye)>=9. With the no-build ring closing that off, the
# far staircase strands at 8.875. Comparing the true float (8.875 > 8) recognises it as a
# valid long-range launch. Off by default (keeps the ROM-validated ls42 int gate).
_ENDGAME_FRAC_EYE = os.environ.get("ENDGAME_FRAC_EYE", "0") == "1"
# SEEN-DRAIN: model being seen as a COSTED, survivable state rather than a hard veto. As
# the search simulates each move it debits the energy the player would actually lose while
# dwelling at the destination during that move (sentinel.enemies.step: 1 energy per
# ~120 seen ticks, nothing for a quick transit). Routes that linger in view bleed energy
# and score lower; brief crossings are free -- so the planner can TRANSIT unavoidable seen
# tiles (multi-sentry landscapes) and is pushed to reach a safe tile fast. Off by default
# (keeps the ROM-validated ls42 energy accounting; ls0 relies on avoidance knobs instead).
_SEEN_DRAIN = os.environ.get("SEEN_DRAIN", "0") == "1"


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
    settle = actioncost.SETTLE["create"]  # the synthoid create (always fired)
    if use_b:
        rounds += ROUNDS_PER_H_STEP + ROUNDS_PER_V_STEP  # re-centre on-boulder synthoid
        settle += actioncost.SETTLE["create"]  # boulder create
        # the synthoid stacks ON the boulder -> taller redraw, ~STACK_CREATE more rounds
        # (the dominant per-move cost the flat model missed; see actioncost).
        settle += actioncost.STACK_CREATE
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


def _cost(c, exposed):
    """Up-front energy a candidate needs before the shell-reabsorb refund: boulder-step
    5 (boulder 2 + synthoid 3), hop 3, plus RESERVE, plus EXPOSURE_RESERVE when the
    destination is enemy-exposed."""
    return (
        (5 if c[1] else 3)
        + RESERVE
        + (EXPOSURE_RESERVE if tuple(c[0]) in exposed else 0)
    )


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
    plat = ctx["plat"]
    if cheb(g.player_xy(), plat) <= 1:
        return True
    # far win: check LOS to the platform. Only reached once the eye is already above the
    # platform (late climb), so this sweep runs on very few nodes. A fractional eye above
    # plat_ground sees DOWN onto it; ceil the observer so int(eye)==plat_ground doesn't
    # drop the platform (matches endgame's seye; only when frac-eye is on).
    ie = int(g.eye)
    seye = ie + 1 if (_ENDGAME_FRAC_EYE and g.eye > ie) else ie
    sw = visibility_sweep(g.mem, g.player, seye, max_steps=200, coarse=True)
    return plat in sw


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
    raw = _candidates(
        g,
        cur,
        eye,
        plat=(ctx["plat"] if ctx["toward_plat"] else None),
        near_plat_radius=ctx["near_plat_radius"],
    )
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
    # HARD no-build ring: drop CREATE footholds (c[1] is use_boulder) inside the deadly
    # cheb<=_RING_NOBUILD_RADIUS zone -- an object built there is absorbed by the Sentinel
    # and spawns meanies regardless of player exposure. Plain hops (c[1] False) onto
    # existing objects stay. Forces the launch stack to a safer far tile (the human route).
    if _RING_NOBUILD_RADIUS > 0:
        raw = [
            c
            for c in raw
            if not (c[1] and cheb(c[0], ctx["plat"]) <= _RING_NOBUILD_RADIUS)
        ]
        if not raw:
            return []
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
        if g.energy >= _cost(c, exposed)
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
    # cheap HEIGHT-first order (biggest eye gain wins; platform proximity only breaks
    # equal-height ties), then compute the enemy safety margin for the top shortlist and
    # re-rank those. Mirrors _evaluate so the beam keeps the highest-climbing lines.
    plat = ctx["plat"]
    pool.sort(key=lambda c: (c[2], -cheb(c[0], plat), edge_dist(c[0])), reverse=True)
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
    if _GAZE_AWARE_COST:
        # Among EQUAL-height candidates, prefer the one the Sentinel is NOT looking at
        # (safety) and the one that needs the least panning (least mid-aim exposure) OVER
        # the one closest to the platform. The default order prefers proximity, which --
        # since both a ring tile and a far tile at the Sentinel's height reach the endgame
        # trigger (both score INF) -- makes the search launch from the exposed ring.
        head.sort(
            key=lambda c: (
                c[2],
                safety[tuple(c[0])],
                -pan[tuple(c[0])],
                -cheb(c[0], plat),
                edge_dist(c[0]),
            ),
            reverse=True,
        )
    else:
        head.sort(
            key=lambda c: (
                c[2],
                -cheb(c[0], plat),
                safety[tuple(c[0])],
                -pan[tuple(c[0])],
                edge_dist(c[0]),
            ),
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
_W_EYE = 1_000_000_000.0  # height reached: strictly dominant
_W_PLAT = 1_000.0  # platform proximity: equal-height tie-break only (max span 6.2e4)
_W_SAFETY = 100.0  # max span 255*1e2 = 2.55e4
_W_EDGE = 1.0  # max span ~62
_W_ENERGY = 0.01


def _evaluate(g, ctx):
    """Score a cut leaf (depth exhausted, not yet won): height reached DOMINATES (gain
    height as fast as possible), then platform proximity / ticks_until_seen (safety) / edge
    / energy purely as tie-breaks among equal-height leaves (SEARCH_REDESIGN.md sec.7).
    """
    cur = g.player_xy()
    state = _read_state(g)
    safety = threat.ticks_until_seen(state, cur[0], cur[1], horizon=_SAFETY_HORIZON)
    # gaze-aware: SAFETY outranks platform-proximity among equal-height leaves (still far
    # below height). Default: proximity outranks safety (the ROM-validated ls42 order).
    w_safety = 100_000.0 if _GAZE_AWARE_COST else _W_SAFETY
    return (
        g.eye * _W_EYE
        - cheb(cur, ctx["plat"]) * _W_PLAT
        + min(safety, 255) * w_safety
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
    # on RAM (native create/absorb don't write OBJ_HANG) -- only this root read touches it.
    root_h, root_v = g.mem[OBJ_HANG + g.player], g.mem[OBJ_VANG + g.player]
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
    _apply(g, T2, use_b, view)
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
    g = Game(landscape)
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
