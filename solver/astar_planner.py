#!/usr/bin/env python3
"""Weighted-A* macro planner (T2.4).

Drives the macro expanders (:mod:`solver.macros`) with a weighted-A* frontier over
search :class:`~solver.search_node.Node`s.  The heuristic is an admissible lower
bound on the enemy-rounds still needed to climb from the current eye to launch
height; the endgame macro is the goal test.  On failure the planner returns a
structured diagnosis of the tightest blocker rather than a bare ``False``.

All tuning constants are module-level and env-overridable so a run can be
recalibrated without a code change.  Defaults come from the planner plan's
"Global config defaults".
"""

import os
import sys
import math
import time
import itertools
from heapq import heappush, heappop
from dataclasses import dataclass
from typing import Optional

if __package__ in (None, ""):  # run as a script: put the repo root on the path
    sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from solver.plan_game import PlanGame
from solver.search_node import Node, node_key
from solver.search_node import energy as node_energy
from solver.gaze import GazeTimeline
from solver.macros import expand_climb, expand_refuel, endgame_child


def _envf(name, default):
    return float(os.environ.get(name, default))


def _envi(name, default):
    return int(os.environ.get(name, default))


# --- tuning constants (all env-overridable) ---------------------------------
W_ASTAR = _envf("W_ASTAR", "1.5")  # weighted-A* inflation on the heuristic
MAX_H_PER_MOVE = _envf("MAX_H_PER_MOVE", "2.0")  # optimistic eye gain per macro
MIN_MOVE_ROUNDS = _envf("MIN_MOVE_ROUNDS", "290.0")  # cheapest possible move cost
T_BUCKET = _envi("T_BUCKET", "64")  # closed-set tick bucket (mirrors search_node)
NEXT_COST_FLOOR = _envi("NEXT_COST_FLOOR", "3")  # energy kept for one more synthoid
# HORIZON bounds both the gaze build and the ``c.t < HORIZON`` child filter. The plan's
# nominal 4000 is too small: the ls0 winning climb's cumulative world-tick is ~12000
# rounds (each macro's action window is ~2000), so 4000 prunes the winning path mid-climb.
HORIZON = _envi("HORIZON", "20000")  # gaze / world tick horizon
NODE_BUDGET = _envi("NODE_BUDGET", "20000")  # max node expansions
T_BUDGET_S = _envf("T_BUDGET_S", "45.0")  # wall-clock budget (< 60 s hard cap)
# BEAM: children pushed per expansion. The plan's nominal 8 cannot fit the 60 s CPU
# budget -- each expansion costs ~5 s (full survivability enemy-stepping over ~2000-round
# windows for ~50 candidates), so a wide beam drowns in the equal-eye plateau before
# reaching launch. BEAM=1 is a greedy commit (best child by height then cost) that wins
# ls0 in 6 expansions / ~33 s; raise it (with faster macros) for backtracking robustness.
BEAM = _envi("BEAM", "1")  # children pushed per expansion
BRANCH_HIGH = _envi("BRANCH_HIGH", "24")  # high-branch escalation threshold
SAFETY_HORIZON = _envi("SAFETY_HORIZON", "256")  # exposure look-ahead window


def heuristic(n, plat_ground) -> float:
    """Admissible lower bound on remaining enemy-rounds to reach launch height:
    the height still to climb, divided by the most any one macro can gain, times
    the cheapest a macro can cost."""
    dh = max(0.0, (plat_ground + 1) - n.g.eye)
    return math.ceil(dh / MAX_H_PER_MOVE) * MIN_MOVE_ROUNDS


@dataclass
class PlanResult:
    """Outcome of a :func:`plan` call."""

    won: bool
    steps: list  # flattened PlanGame.steps of the winning path
    failure: Optional[dict]  # {"reason","blocker","detail"} when not won
    stats: dict  # nodes, wall_s, peak_eye


def _facing_h(g0):
    return int(g0.state.obj_h_angle[g0.player])


def _facing_v(g0):
    return int(g0.state.obj_v_angle[g0.player])


def _diagnose(best_eye, g0, nodes, no_safe_seen, budget_hit) -> dict:
    """Pick the tightest blocker for a non-winning search.

    ``no_launch_los`` -- climbed to/above launch height but never had a down-look
    LOS to the platform; ``energy_deficit`` -- stalled below launch height with no
    affordable refuel; ``no_safe_window`` -- height-gaining children existed but all
    failed the survivability gate; ``budget_exhausted`` -- the node/time budget ran
    out first."""
    pg = g0.plat_ground if g0.plat_ground is not None else 8
    detail = {
        "best_eye": round(best_eye, 3),
        "plat_ground": pg,
        "nodes": nodes,
    }
    if best_eye > pg:
        return {
            "reason": "reached launch height but never gained down-look LOS to platform",
            "blocker": "no_launch_los",
            "detail": detail,
        }
    if budget_hit:
        return {
            "reason": "node/time budget exhausted before reaching launch height",
            "blocker": "budget_exhausted",
            "detail": detail,
        }
    if no_safe_seen:
        return {
            "reason": "all height-gaining moves failed the survivability window",
            "blocker": "no_safe_window",
            "detail": detail,
        }
    return {
        "reason": "climb stalled below launch height with no affordable refuel",
        "blocker": "energy_deficit",
        "detail": detail,
    }


def plan(landscape_or_game, cfg=None) -> PlanResult:
    """Weighted-A* over the macro expanders.  Returns a :class:`PlanResult`;
    ``won`` iff the endgame macro fired from a reached node within budget."""
    del cfg  # constants are module-level/env-overridable; no per-call config yet
    t_start = time.time()
    g0 = (
        PlanGame(landscape_or_game)
        if isinstance(landscape_or_game, int)
        else landscape_or_game
    )
    gaze = GazeTimeline(g0.state, horizon=HORIZON)
    start = Node(g=g0, t=0, vh=_facing_h(g0), vv=_facing_v(g0), cost=0.0)

    counter = itertools.count()
    openq = [(heuristic(start, g0.plat_ground), next(counter), start)]
    closed = {}
    best_eye = g0.eye
    nodes = 0
    no_safe_seen = False
    budget_hit = False

    while openq:
        if nodes >= NODE_BUDGET or (time.time() - t_start) > T_BUDGET_S:
            budget_hit = True
            break
        _f, _tb, n = heappop(openq)

        end = endgame_child(n, g0.plat, g0.plat_ground)
        if end is not None:
            stats = {
                "nodes": nodes,
                "wall_s": round(time.time() - t_start, 3),
                "peak_eye": round(max(best_eye, n.g.eye), 3),
            }
            return PlanResult(True, end.g.steps, None, stats)

        k = node_key(n)
        if k in closed and closed[k] <= n.cost:
            continue
        closed[k] = n.cost
        best_eye = max(best_eye, n.g.eye)

        children = expand_climb(n, gaze) + expand_refuel(n, gaze)
        children = [c for c in children if node_energy(c) > 0 and c.t < HORIZON]
        gaining = any(c.g.eye > n.g.eye + 1e-9 for c in children)
        if not gaining and n.g.eye < g0.plat_ground:
            no_safe_seen = True  # nothing raised the eye from this below-launch node
        children.sort(key=lambda c: (-c.g.eye, c.cost))
        for c in children[:BEAM]:
            f = c.cost + W_ASTAR * heuristic(c, g0.plat_ground)
            heappush(openq, (f, next(counter), c))
        nodes += 1

    stats = {
        "nodes": nodes,
        "wall_s": round(time.time() - t_start, 3),
        "peak_eye": round(best_eye, 3),
    }
    failure = _diagnose(best_eye, g0, nodes, no_safe_seen, budget_hit)
    return PlanResult(False, [], failure, stats)


def main(argv=None) -> int:
    """CLI: ``python3 solver/astar_planner.py [landscape]`` -- print won, step
    count, peak eye, wall seconds and (on failure) the diagnosis."""
    argv = list(sys.argv[1:] if argv is None else argv)
    landscape = int(argv[0]) if argv else 0
    result = plan(landscape)
    print(f"landscape {landscape}: won={result.won}")
    print(f"  steps={len(result.steps)}")
    print(f"  peak_eye={result.stats.get('peak_eye')}")
    print(f"  nodes={result.stats.get('nodes')}  wall_s={result.stats.get('wall_s')}")
    if not result.won:
        print(f"  failure={result.failure}")
    return 0 if result.won else 1


if __name__ == "__main__":
    sys.exit(main())
