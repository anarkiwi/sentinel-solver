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
native_game (sec.4, sec.8) -- _candidates, _boulder_centre_feasible, _refuel, _apply,
edge_dist -- and branches over native_game.Game.clone()d states. Enemy timing is folded
in as a per-candidate SAFETY ANNOTATION via enemy_dynamics (sec.5), not an adversarial
search dimension (sec.2): meanie_safe is a hard pre-filter, ticks_until_seen a soft
leaf-eval term. The EnemyPhase is advanced by TICKS_PER_ACTION per simulated move
(sec.6) as the lookahead descends.
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
from native_game import Game, cheb
import enemy_dynamics as ED
import game_state as GS

INF = float("inf")
_NOOP = lambda *a, **k: None

# --- search shape (SEARCH_REDESIGN.md sec.3, sec.7) --------------------------
DEFAULT_DEPTH = 3  # plies of lookahead; raise for landscapes that need it (sec.7)
DEFAULT_BEAM = 3  # top-N candidates expanded per node, independent of depth (sec.7)

# --- tick-cost per real keyboard action (SEARCH_REDESIGN.md sec.6) -----------
# NOT invented: a placeholder pending empirical calibration from watch_play.py-style
# telemetry (aim-to-fire durations) against the live executor. The C64 advances
# update_enemies ~once per frame (~50 PAL fps), so seconds*50 ~= ticks. A hop
# (aim+build synthoid+transfer) is cheaper than a boulder-step (two builds+transfer).
# These feed EnemyPhase advancement only; if the live run's safety forecast diverges,
# recalibrate these from that run's log (sec.9 step 4) rather than adding static margin.
TICKS_PER_HOP = 150
TICKS_PER_BOULDER_STEP = 250

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


def _cost(c, exposed):
    """Up-front energy a candidate needs before the shell-reabsorb refund: boulder-step
    5 (boulder 2 + synthoid 3), hop 3, plus RESERVE, plus EXPOSURE_RESERVE when the
    destination is enemy-exposed. Same formula climb_iterate uses (climb_greedy)."""
    return (
        (5 if c[1] else 3)
        + RESERVE
        + (EXPOSURE_RESERVE if tuple(c[0]) in exposed else 0)
    )


def _ticks_for(c):
    return TICKS_PER_BOULDER_STEP if c[1] else TICKS_PER_HOP


def _read_state(g):
    return GS.read_game_state(GS.Py65Source(g.mem))


def _advance_phase(state, phase, c):
    """Advance the EnemyPhase by the estimated tick-cost of applying candidate `c`
    (SEARCH_REDESIGN.md sec.6). Enemies are fixed-rate automata, so rotation/cooldown
    advance is a pure function of tick count against the node's `state`."""
    ph = phase
    for _ in range(_ticks_for(c)):
        ph = ED.step_enemies(state, ph)
    return ph


def _reached_approach(g, ctx):
    """Terminal check: the eye is above the platform ground and adjacent, and the
    Sentinel is still present -- i.e. endgame (absorb Sentinel + platform synthoid +
    transfer) can launch from here (SEARCH_REDESIGN.md sec.8). This is `won(g)` in the
    sec.7 pseudocode; the endgame itself stays a separate terminal step, not something
    the lookahead re-derives."""
    if g.plat_ground is None or g.sentinel_slot is None:
        return False
    return int(g.eye) > g.plat_ground and cheb(g.player_xy(), ctx["plat"]) <= 1


def _gen_candidates(g, ctx, blocked, state, phase, beam):
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
    # affordability + blocklist (cheap, no sweeps)
    exposed = _enemy_exposed_tiles(g, {tuple(c[0]) for c in raw})
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
    safe = [c for c in pool if ED.meanie_safe(state, tuple(c[0]))]
    pool = safe or pool
    # cheap height-then-platform order, then compute the enemy safety margin only for the
    # top shortlist and re-rank those. Mirrors _evaluate's priority (height gain, platform
    # proximity, safety margin, edge distance) so the beam keeps the goal-directed lines.
    # cheap goal-potential order (mirrors _evaluate: climb while low, approach once high),
    # then compute the enemy safety margin only for the top shortlist and re-rank those.
    pool.sort(key=lambda c: (-_goal_h(c[0], c[2], ctx), edge_dist(c[0])), reverse=True)
    head = pool[: beam + _SHORTLIST]
    safety = {
        tuple(c[0]): ED.ticks_until_seen(
            state, phase, c[0][0], c[0][1], horizon=_SAFETY_HORIZON
        )
        for c in head
    }
    head.sort(
        key=lambda c: (
            -_goal_h(c[0], c[2], ctx),
            safety[tuple(c[0])],
            edge_dist(c[0]),
        ),
        reverse=True,
    )
    return head + pool[beam + _SHORTLIST :]


# leaf-eval weights, chosen so the ordering is strictly lexicographic. The GOAL potential
# dominates -- a 0.5 step of progress (5e5) swamps every lower term's whole span.
_W_GOAL = 1_000_000.0  # goal potential (see _goal_h): dominant
_W_SAFETY = 100.0  # max span 255*1e2 = 2.55e4
_W_EDGE = 1.0  # max span ~62
_W_ENERGY = 0.01


def _goal_h(tile, eye, ctx):
    """Distance-to-WIN potential, native_game._astar_terrain's validated A* heuristic:
    cheb-to-platform + remaining height deficit (target_z - eye, floored at 0). LOWER is
    closer to the win. Because non-regression is enforced at candidate generation (a
    down-move is never a candidate), this can safely be the dominant term without risking a
    height regression: while low, the deficit dominates so the search CLIMBS; once high
    enough (deficit 0), cheb dominates so it APPROACHES the platform (the phase greedy
    handled separately). This is what converges the climb onto the win instead of the
    nearest peak (ls9999/ls66 sit below surrounding terrain, so pure height-max walks past
    the platform)."""
    return cheb(tile, ctx["plat"]) + max(0.0, ctx["target_z"] - eye)


def _evaluate(g, ctx, phase):
    """Score a cut leaf (depth exhausted, not yet won): the goal potential (dominant --
    climb while low, approach once high), then ticks_until_seen (safety margin), edge
    distance and remaining energy as tie-breaks (SEARCH_REDESIGN.md sec.7)."""
    cur = g.player_xy()
    state = _read_state(g)
    safety = ED.ticks_until_seen(state, phase, cur[0], cur[1], horizon=_SAFETY_HORIZON)
    return (
        -_goal_h(cur, g.eye, ctx) * _W_GOAL
        + min(safety, 255) * _W_SAFETY
        + edge_dist(cur) * _W_EDGE
        + g.energy * _W_ENERGY
    )


def _lookahead(g, ctx, blocked, phase, depth, beam, stats, is_root=True):
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
        return _evaluate(g, ctx, phase), None
    state = _read_state(g)
    cands = _gen_candidates(g, ctx, blocked, state, phase, beam)
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
        phase2 = _advance_phase(state, phase, c)
        sub_score, _ = _lookahead(
            g2, ctx, blocked, phase2, depth - 1, beam, stats, is_root=False
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
    plat, plat_ground = ctx["plat"], ctx["plat_ground"]
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
    if eye > plat_ground and cheb(cur, plat) <= 1:
        log(f"  reached platform approach: {cur} eye {eye} (d=0/1)")
        return "approach"

    state = _read_state(g)
    phase = ED.init_phase_from_ram(state, g.mem)
    stats = {"nodes": 0}
    t0 = time.time()
    score, move = _lookahead(g, ctx, blocked, phase, depth, beam, stats)
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
