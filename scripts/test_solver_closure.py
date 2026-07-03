#!/usr/bin/env python3
"""Unit tests for scripts/solver_closure.py (the monotone reachability-closure /
best-base-sweep planner).

Asserts, for landscapes 0000 / 0042 / 9999:
  * the closure solves and its model_replay projection replays through
    game_model.apply to `won`;
  * the closure collects >= greedy (solver.py) objects and >= greedy final energy;
  * the closure is deterministic (two solves identical).
Also exercises analyse_solvability's honest hyperspace wording on a broken state.
"""

import sys
import os

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

import solver as greedy
from solver_closure import solve, analyse_solvability
from solver import _replay_verify
from game_model import GameModel, T_SENTINEL

FAILS = 0
LANDSCAPES = (0, 42, 9999)


def check(cond, msg):
    global FAILS
    if cond:
        print(f"  ok: {msg}")
    else:
        FAILS += 1
        print(f"  FAIL: {msg}")


def test_closure_solves_and_replays():
    print("[closure solves + replays to won for 0/42/9999]")
    for ls in LANDSCAPES:
        m = GameModel.from_landscape(ls)
        rep = analyse_solvability(m)
        check(
            rep.solvable,
            f"ls{ls:04d} analyse_solvability solvable (reason={rep.reason!r})",
        )
        plan = solve(m)
        check(plan.solved, f"ls{ls:04d} closure returns a solved plan")
        check(plan.los_proven, f"ls{ls:04d} abstract LOS proof holds")
        check(
            _replay_verify(m.state, plan.model_replay),
            f"ls{ls:04d} model-replay reaches won through game_model.apply",
        )
        check(
            any(t == "SENTINEL" for t, _x, _y in plan.absorbed),
            f"ls{ls:04d} Sentinel is absorbed",
        )
        # win is the LAST collection in model_replay (Sentinel absorbed last).
        verbs = [a.verb for a in plan.model_replay]
        check(
            verbs and verbs[-1] == "win",
            f"ls{ls:04d} the win is the final model-replay action (Sentinel last)",
        )


def test_closure_beats_greedy():
    print("[closure objects/energy >= greedy for 0/42/9999]")
    for ls in LANDSCAPES:
        m = GameModel.from_landscape(ls)
        gp = greedy.solve(m)
        cp = solve(m)
        check(
            cp.absorbed_count >= gp.absorbed_count,
            f"ls{ls:04d} closure objects {cp.absorbed_count} >= greedy {gp.absorbed_count}",
        )
        check(
            cp.final_energy >= gp.final_energy,
            f"ls{ls:04d} closure energy {cp.final_energy} >= greedy {gp.final_energy}",
        )


def test_closure_deterministic():
    print("[closure is deterministic]")
    for ls in LANDSCAPES:
        m = GameModel.from_landscape(ls)
        p1 = solve(m)
        p2 = solve(m)
        check(
            [repr(a) for a in p1.actions] == [repr(a) for a in p2.actions],
            f"ls{ls:04d} two closure solves yield identical action lists",
        )


def test_solvability_honesty():
    print("[analyse_solvability honest reasons]")
    m = GameModel.from_landscape(0)
    # grounded Sentinel (no platform): proven impossible via (1b).
    grounded = greedy.break_state_grounded_sentinel(m)
    r = analyse_solvability(grounded)
    check(not r.solvable, "grounded-Sentinel not solvable")
    check(
        "platform" in r.reason.lower() or "ground" in r.reason.lower(),
        f"grounded-Sentinel reason mentions platform/ground: {r.reason!r}",
    )
    # no Sentinel: degenerate.
    import copy

    degen = copy.deepcopy(m.state)
    degen.objects = [o for o in degen.objects if o.type != T_SENTINEL]
    r2 = analyse_solvability(degen)
    check(
        not r2.solvable and r2.reason.strip() != "",
        f"no-Sentinel reports a reason: {r2.reason!r}",
    )


def main():
    test_closure_solves_and_replays()
    test_closure_beats_greedy()
    test_closure_deterministic()
    test_solvability_honesty()
    print()
    if FAILS:
        print(f"FAILED: {FAILS} check(s) failed")
        sys.exit(1)
    print("all solver_closure tests passed")


if __name__ == "__main__":
    main()
