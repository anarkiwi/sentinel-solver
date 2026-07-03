#!/usr/bin/env python3
"""Unit tests for scripts/solver.py against real generated landscapes.

Asserts that landscape 0000 is solvable and its plan replays through
game_model.apply to `won`, with a sane final-energy lower bound, and that
analyse_solvability returns a human-readable reason for deliberately-broken
(unsolvable) states.
"""

import sys
import os

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

import solver
from solver import solve, analyse_solvability, _replay_verify
from game_model import GameModel, T_SENTINEL

FAILS = 0


def check(cond, msg):
    global FAILS
    if cond:
        print(f"  ok: {msg}")
    else:
        FAILS += 1
        print(f"  FAIL: {msg}")


def test_ls0000_solvable():
    print("[landscape 0000 solvable + replays to won]")
    m = GameModel.from_landscape(0)
    start_energy = m.state.player_energy

    rep = analyse_solvability(m)
    check(rep.solvable, f"analyse_solvability says solvable (reason={rep.reason!r})")

    plan = solve(m)
    check(plan.solved, "solve() returns a solved plan")
    check(plan.los_proven, "abstract LOS proof holds (climb reveals the Sentinel)")

    # the emitted model-replay projection must reach won through game_model.apply
    won = _replay_verify(m.state, plan.model_replay)
    check(won, "model-replay projection replays through game_model.apply to won")

    # final energy lower bound: we win (+4 Sentinel) and recover the climb robot,
    # so the budget must be at least the start energy.
    check(
        plan.final_energy >= start_energy,
        f"final energy {plan.final_energy} >= start energy {start_energy}",
    )
    # and it must be within the 6-bit cap.
    check(0 <= plan.final_energy <= 63, f"final energy in 0..63 ({plan.final_energy})")

    # the Sentinel must be among the absorbed objects.
    check(
        any(t == "SENTINEL" for t, _x, _y in plan.absorbed),
        "Sentinel is absorbed in the plan",
    )

    # the executor-facing action list is non-empty and tile-addressed.
    check(
        len(plan.actions) >= len(plan.model_replay),
        f"executor actions ({len(plan.actions)}) >= model-replay "
        f"({len(plan.model_replay)})",
    )
    verbs = [a.verb for a in plan.actions]
    check("win" in verbs, "executor action list contains the win")
    # any post-win steps are Stage-D re-absorbs of the climb structure.
    win_i = verbs.index("win")
    check(
        all(v == "absorb" for v in verbs[win_i + 1 :]),
        "all post-win executor actions are Stage-D re-absorbs",
    )


def test_ls0042_9999_solvable():
    print("[landscapes 0042 / 9999 solvable + replay-verified]")
    for ls in (42, 9999):
        m = GameModel.from_landscape(ls)
        rep = analyse_solvability(m)
        check(rep.solvable, f"ls{ls:04d} solvable (reason={rep.reason!r})")
        plan = solve(m)
        check(
            plan.solved and _replay_verify(m.state, plan.model_replay),
            f"ls{ls:04d} plan solved and replays to won "
            f"(energy={plan.final_energy}, absorbed={plan.absorbed_count})",
        )


def test_unsolvable_reasons():
    print("[analyse_solvability returns a reason for broken states]")
    m = GameModel.from_landscape(0)

    # (a) zero energy: cannot build the climbing structure.
    no_e = solver.break_state_no_energy(m)
    r = analyse_solvability(no_e)
    check(not r.solvable, "zero-energy state is not solvable")
    check(
        isinstance(r.reason, str) and r.reason.strip() != "",
        f"zero-energy reason is a non-empty string: {r.reason!r}",
    )
    check("energy" in r.reason.lower(), "zero-energy reason mentions energy")

    # (b) Sentinel grounded (no platform): the winning transfer is impossible.
    grounded = solver.break_state_grounded_sentinel(m)
    r2 = analyse_solvability(grounded)
    check(not r2.solvable, "grounded-Sentinel state is not solvable")
    check(
        isinstance(r2.reason, str) and r2.reason.strip() != "",
        f"grounded-Sentinel reason is a non-empty string: {r2.reason!r}",
    )
    check(
        "platform" in r2.reason.lower() or "ground" in r2.reason.lower(),
        "grounded-Sentinel reason mentions platform/ground",
    )

    # (c) no Sentinel at all: degenerate map.
    import copy

    degen = copy.deepcopy(m.state)
    degen.objects = [o for o in degen.objects if o.type != T_SENTINEL]
    r3 = analyse_solvability(degen)
    check(
        not r3.solvable and r3.reason.strip() != "",
        f"no-Sentinel state reports a reason: {r3.reason!r}",
    )


def test_plan_is_deterministic():
    print("[solver is deterministic]")
    m = GameModel.from_landscape(0)
    p1 = solve(m)
    p2 = solve(m)
    check(
        [repr(a) for a in p1.actions] == [repr(a) for a in p2.actions],
        "two solves of ls0000 yield identical action lists",
    )


def main():
    test_ls0000_solvable()
    test_ls0042_9999_solvable()
    test_unsolvable_reasons()
    test_plan_is_deterministic()
    print()
    if FAILS:
        print(f"FAILED: {FAILS} check(s) failed")
        sys.exit(1)
    print("all solver tests passed")


if __name__ == "__main__":
    main()
