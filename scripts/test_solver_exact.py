#!/usr/bin/env python3
"""Tests for the exact / minimax planner (solver_exact.py).

Asserts:
  * ls0000 solves and its model_replay reaches `won` through game_model.apply;
  * the exact planner's final energy is >= the greedy solver.solve final energy
    for ls 0/42/9999 (it must never do worse than greedy);
  * determinism: two runs of solve() give identical results;
  * the enemy_dynamics phase cross-checks against the generated RAM.
"""

import sys
import os

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from game_model import GameModel
import solver as greedy
import solver_exact as sx
import enemy_dynamics as ed
import game_state as gs
from game_state import read_game_state

DEPTH = 5
BUDGET = 40.0
LANDSCAPES = (0, 42, 9999)


def test_ls0_solves_and_replays():
    m = GameModel.from_landscape(0)
    plan = sx.solve(m, depth=DEPTH, budget_s=BUDGET)
    assert plan.solved, f"ls0000 should solve: {plan.notes}"
    assert sx._replay_verify(
        m.state, plan.model_replay
    ), "ls0000 model-replay must reach won"
    print("ok: ls0000 solves and replays to won")


def test_exact_never_worse_than_greedy():
    for ls in LANDSCAPES:
        m = GameModel.from_landscape(ls)
        g = greedy.solve(m)
        x = sx.solve(m, depth=DEPTH, budget_s=BUDGET)
        assert x.solved, f"ls{ls:04d} exact should solve: {x.notes}"
        assert (
            x.final_energy >= g.final_energy
        ), f"ls{ls:04d}: exact energy {x.final_energy} < greedy {g.final_energy}"
        print(
            f"ok: ls{ls:04d} exact E={x.final_energy} >= greedy E={g.final_energy} "
            f"(abs {x.absorbed_count} vs {g.absorbed_count})"
        )


def test_determinism():
    for ls in LANDSCAPES:
        m = GameModel.from_landscape(ls)
        a = sx.solve(m, depth=DEPTH, budget_s=BUDGET)
        b = sx.solve(m, depth=DEPTH, budget_s=BUDGET)
        assert a.final_energy == b.final_energy, f"ls{ls}: energy differs"
        assert a.absorbed_count == b.absorbed_count, f"ls{ls}: absorbed differs"
        assert [repr(s.action) for s in a.steps] == [
            repr(s.action) for s in b.steps
        ], f"ls{ls}: step sequence differs between runs"
        print(f"ok: ls{ls:04d} deterministic (E={a.final_energy})")


def test_enemy_dynamics_crosscheck():
    # rotation speeds and cooldowns read from RAM must match the ROM init rules.
    for ls in LANDSCAPES:
        src = gs.Py65Source.from_landscape(ls)
        state = read_game_state(src)
        phase = ed.init_phase_from_ram(state, src.mem)
        assert phase.enemies, f"ls{ls}: no enemies parsed"
        for slot, e in phase.enemies.items():
            assert e.rotation_speed in (
                ed.ROTATION_STEP_CW,
                ed.ROTATION_STEP_CCW,
            ), f"ls{ls} slot{slot}: rotation speed {e.rotation_speed} not +/-20"
            # update cooldown initialised to (prnd & $3F) | $05 -> in 5..63
            assert (
                5 <= e.update_cooldown <= 63
            ), f"ls{ls} slot{slot}: update_cooldown {e.update_cooldown} out of 5..63"
        # phase advances deterministically
        p1 = phase.copy()
        p2 = phase.copy()
        for _ in range(50):
            p1 = ed.step_enemies(state, p1)
            p2 = ed.step_enemies(state, p2)
        for slot in phase.enemies:
            assert p1.enemies[slot].h_angle == p2.enemies[slot].h_angle
        print(
            f"ok: ls{ls:04d} enemy_dynamics cross-check ({len(phase.enemies)} enemies)"
        )


def main():
    test_enemy_dynamics_crosscheck()
    test_ls0_solves_and_replays()
    test_exact_never_worse_than_greedy()
    test_determinism()
    print("\nALL TESTS PASSED")


if __name__ == "__main__":
    main()
