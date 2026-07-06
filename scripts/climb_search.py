#!/usr/bin/env python3
"""Receding-horizon best-first lookahead climb planner (SEARCH_REDESIGN.md).

Drop-in replacement for climb_greedy's per-decision function. Greedy commits to the
single best-LOOKING foothold each call and only discovers a dead end after arriving
(SEARCH_REDESIGN.md sec.1). This runs a bounded depth-`D` lookahead at every real
decision, scores the leaves, and commits the FIRST move of the best-scoring line --
so a move is taken ONLY if it has a continuation within the horizon (a node with no
non-dead-end successor scores -inf and is pruned by its parent). Re-runs fresh from
each real state, chess-engine style: the plan can be arbitrarily long though each
search is shallow (sec.3).

It REUSES the real, keyboard-faithful mechanics already validated in climb_greedy /
plan_game (sec.4, sec.8) -- _candidates, _boulder_centre_feasible, _refuel, _apply,
edge_dist -- and branches over plan_game.Game.clone()d states. Enemy timing is folded
in as a per-candidate SAFETY ANNOTATION via sentinel.threat (sec.5), not an adversarial
search dimension (sec.2): meanie_safe is a hard pre-filter, ticks_until_seen a soft
leaf-eval term. Each node's sentinel enemy state is advanced by the move's REAL tick cost
(_move_cost, derived from the keyboard-aim geometry: aim + build + transfer + the return-
pan to reabsorb the shell) as the lookahead descends (sec.6).
"""

import sys, os, json, time

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
os.chdir(os.path.join(HERE, ".."))

from climb_greedy import (
    _candidates,
    _boulder_centre_feasible,
    _refuel,
    _apply,
    edge_dist,
    climb_ctx,
    endgame,
    RESERVE,
    EXPOSURE_RESERVE,
    _enemy_exposed_tiles,
)
from plan_game import PlanGame as Game, cheb, visibility_sweep, OBJ_HANG, OBJ_VANG
from sentinel import threat, enemies as SE, aimcost as ac

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
# platform synthoid, fired after the Sentinel is gone) runs in climb_greedy.endgame and
# is not gated here. 0 = off (default).
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
    fires = 2 if use_b else 1  # boulder + synthoid, or lone synthoid
    if use_b:
        rounds += ROUNDS_PER_H_STEP + ROUNDS_PER_V_STEP  # re-centre on-boulder synthoid
    fires += 1  # transfer confirm
    end_h, end_v = vh, vv
    back_h = ac.bearing_to(T2[0], T2[1], prev[0], prev[1])  # look back at departed tile
    if back_h is not None and end_h is not None:
        # the return-pan (bearing), U-turn-aware: a swing back past half a turn is one
        # U-turn + a short correction, matching what the live driver keys.
        rounds += ac.bearing_rounds(end_h, back_h, ROUNDS_PER_H_STEP, ROUNDS_PER_UTURN)
        end_h = back_h
        fires += 1  # reabsorb-shell confirm
    ticks = int(round(rounds + fires * ROUNDS_PER_ACTION))
    return ticks, end_h, end_v


def _cost(c, exposed):
    """Up-front energy a candidate needs before the shell-reabsorb refund: boulder-step
    5 (boulder 2 + synthoid 3), hop 3, plus RESERVE, plus EXPOSURE_RESERVE when the
    destination is enemy-exposed. Same formula climb_iterate uses (climb_greedy)."""
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
    """ONE lookahead decision against g's CURRENT state -- drop-in for
    climb_greedy.climb_iterate (same signature, same status strings, same ctx
    bookkeeping / g.steps side effects). Banks fuel, checks the terminal/no-gain
    conditions, then runs _lookahead and COMMITS the first move of the best line.

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
        # climb_iterate's out-of-footholds branch, incl. the no_gain loop bound).
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
    """Offline driver mirroring climb_greedy.plan_greedy, but with the lookahead
    decision (search_iterate) in place of climb_iterate. Same endgame.

    Defaults to the LIVE executor's config (toward_plat=True, near_plat_radius=2, see
    record_win_0042.execute_live) rather than plan_greedy's legacy toward_plat=False:
    the lookahead already drives toward the platform via _evaluate's proximity term, and
    toward_plat gates the ROM-infeasible on-distant-boulder synthoid in the steep
    platform ring (climb_greedy._candidates)."""
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
