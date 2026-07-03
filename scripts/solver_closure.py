#!/usr/bin/env python3
"""Monotone reachability-closure / best-base-sweep planner for The Sentinel (C64).

This is the recommended planner (the "closure"): the simplest method that reliably
beats the greedy staged planner (`solver.py`) and also beats the depth-limited
minimax (`solver_exact.py`) on every benchmark, with ZERO tunables beyond the
existing model constants and a sub-second runtime.

Why it works
------------
Three monotone facts collapse the whole optimiser to a sort:
  1. LOS is a pure function of the static terrain (objects never occlude), so the
     set of tiles you can ever stand on is FIXED -- the flat-eye LOS-connected
     component of the start tile (one BFS, `game_model`/`solver` reachability).
  2. Height is monotone: building only ever ADDS eye height
     (eye = terrain + ROBOT_EYE + k*BOULDER_HEIGHT); from a chosen base the best you
     can do is climb to the tallest useful stack and sweep everything that vantage
     reveals -- for the Sentinel and every collectible in ONE climb, because LOS
     doesn't change as you absorb.
  3. Movement and stacking are energy-neutral (re-absorbed), so there is no path
     cost to optimise. The only choice is WHICH base maximises the absorbable set
     that includes the Sentinel -- a single argmax over <=116 tiles.

So this module changes ONLY the base SELECTION; it REUSES `solver.py`'s hop /
climb / two-view `Plan` / replay-verify machinery verbatim. The emitted `Plan` is
the exact shape the executor (and `code_engine.play_plan`) consumes.

Public interface
----------------
  solve(state_or_model) -> Plan            (same Plan shape as solver.solve)
  analyse_solvability(state_or_model) -> Report

Safety filter (Task 2)
----------------------
Among bases that tie on (absorbable-object-count, energy), this planner prefers
bases whose vantage carries NO meanie-spawn risk (`enemy_dynamics.meanie_spawn_
threat` empty). If every winning base carries a meanie risk, it picks the
least-exposed one and records the residual risk in `plan.notes`.

Hyperspace + honest solvability (Task 3)
----------------------------------------
`analyse_solvability` distinguishes "no in-component vantage reaches the Sentinel"
(which a random hyperspace MIGHT escape) from absolute impossibility, and -- when
cheap -- reports whether ANY board tile admits a Sentinel-absorbing vantage (i.e.
hyperspace COULD help). See `game_model.hyperspace` for the reactive-escape model.
"""

import sys
import os
import copy
import time
from dataclasses import dataclass
from typing import List, Optional, Tuple, Set

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

import game_model as gm
from game_model import (
    GameModel,
    Action,
    can_absorb,
    ROBOT_EYE,
    ENERGY_IN_OBJECTS,
    T_BOULDER,
    T_ROBOT,
    T_SENTINEL,
    T_PLATFORM,
    tile_surface_height,
)
from game_state import GameState, GameObject, N

# Reuse solver.py's machinery VERBATIM -- only base selection is new here.
import solver
from solver import (
    Plan,
    PlanStep,
    Report,
    Climb,
    BOULDER_HEIGHT,
    MAX_STACK,
    cached_can_see,
    _as_state,
    _sentinel,
    _reachability_graph,
    _occupied,
    _execute_hops,
    _replay_verify,
    _finalise_energy,
)

import enemy_dynamics as ed


# ---- per-base analysis ------------------------------------------------------
@dataclass
class BaseChoice:
    base: Tuple[int, int]
    sentinel_stack: int  # min k that reveals the Sentinel
    max_stack: int  # max k needed over the whole cover set
    cover: List[GameObject]  # absorbable objects (incl. Sentinel)
    energy: int  # sum of cover energies (objects collected)
    meanie_risk: List[Tuple[GameObject, Tuple[int, int]]]  # meanie threat at base


def _min_stack(
    state: GameState, base: Tuple[int, int], obj: GameObject
) -> Optional[int]:
    """Smallest boulder stack k (0..MAX_STACK-1) whose raised eye lets the player
    absorb `obj` from `base` -- the real base-tile absorb gate (`can_absorb`: LOS
    to the object's base tile + eye strictly above it). None if no k works.

    Monotone: once a tall-enough eye sees the base tile from above, taller eyes
    also do, so the first hit IS the minimum (a single scan is exact)."""
    for k in range(MAX_STACK):
        eye = ROBOT_EYE + k * BOULDER_HEIGHT
        if can_absorb(state, obj, from_xy=base, eye_offset=eye):
            return k
    return None


def _analyse_base(
    state: GameState, base: Tuple[int, int], objs: List[GameObject], sent: GameObject
) -> Optional[BaseChoice]:
    """For `base`: the min stack to reveal the Sentinel (None => this base can't
    win) and, if it can, the absorbable cover set + the max stack needed."""
    ks = _min_stack(state, base, sent)
    if ks is None:
        return None
    cover: List[GameObject] = []
    kmax = ks
    for o in objs:
        k = _min_stack(state, base, o)
        if k is not None:
            cover.append(o)
            kmax = max(kmax, k)
    energy = sum(ENERGY_IN_OBJECTS[o.type] for o in cover)
    risk = ed.meanie_spawn_threat(state, base)
    return BaseChoice(
        base=base,
        sentinel_stack=ks,
        max_stack=kmax,
        cover=cover,
        energy=energy,
        meanie_risk=risk,
    )


def best_base(
    state: GameState, reachable: Optional[Set[Tuple[int, int]]] = None
) -> Optional[BaseChoice]:
    """The argmax over reachable bases of (absorbable-object-count, energy), among
    bases whose vantage also reveals the Sentinel. Safety filter: ties prefer a base
    with NO meanie-spawn risk; if all winning bases are meanie-risky, the least
    exposed is taken and the residual risk is left on the choice.

    Selection key (descending): (#objects, energy, meanie-safe?, -stack, tile).
    The first three are the quality objective; the rest are deterministic
    tie-breaks (smaller climb preferred, then tile order)."""
    p = state.player
    sent = _sentinel(state)
    if sent is None:
        return None
    if reachable is None:
        reachable = _reachability_graph(state)
    occ = _occupied(state, exclude_slot=p.slot)
    starg = (sent.x, sent.y)
    objs = [o for o in state.objects if o.type != T_PLATFORM and o.slot != p.slot]

    candidates: List[BaseChoice] = []
    for b in reachable:
        if b == starg or b in occ:
            continue
        bc = _analyse_base(state, b, objs, sent)
        if bc is not None:
            candidates.append(bc)
    if not candidates:
        return None

    def key(bc: BaseChoice):
        # maximise objects, then energy, then meanie-safety, then minimise stack,
        # then deterministic tile order.
        return (
            len(bc.cover),
            bc.energy,
            1 if not bc.meanie_risk else 0,
            -bc.max_stack,
            -bc.base[0],
            -bc.base[1],
        )

    candidates.sort(key=key, reverse=True)
    return candidates[0]


# ---- plan emission (REUSES solver.py's hop/climb/win machinery) -------------
def _emit_closure_plan(state: GameState, choice: BaseChoice) -> Plan:
    """Emit hop -> climb -> sweep (Sentinel last) -> win, reusing solver.py's
    primitives. The two-view Plan + replay-verify are exactly solver.py's.

    Structure (mirrors solver._build_climb_and_win, extended to sweep the whole
    cover set from the raised eye before the win):
      * hop to `base` (model-replayable, energy-neutral).
      * build the boulder stack to `max_stack` + a robot, transfer up
        (executor-facing climb metadata; not in model_replay -- same as solver.py).
      * sweep every collectible in `cover` (Sentinel excluded) from the raised eye:
        each is a model-replayable absorb (the eye is the same one that the absorb
        gate `can_absorb` already proved clears the object's base tile).
      * win on the Sentinel LAST (model-replayable; flips state.won).
      * Stage-D re-absorb the stranded climb structure (executor-facing).
    """
    plan = Plan()
    _p = state.player
    sent = _sentinel(state)
    bx, by = choice.base
    base_surf = tile_surface_height(state, bx, by)
    climb = Climb(
        base=choice.base,
        stack=choice.max_stack,
        eye_offset=ROBOT_EYE + choice.max_stack * BOULDER_HEIGHT,
        exposed=gm.is_exposed(
            state, bx, by, object_top=ROBOT_EYE + choice.max_stack * BOULDER_HEIGHT
        ),
    )

    model = GameModel(copy.deepcopy(state))

    # --- hop to the base tile (model-replayable) ---
    occ = _occupied(model.state, exclude_slot=model.state.player_slot)
    path = solver._los_path(
        model.state, (model.state.player.x, model.state.player.y), (bx, by), occ
    )
    if path is None:
        plan.notes.append(f"closure base {choice.base} unreachable for the climb")
        plan.solved = False
        return plan
    model = _execute_hops(model, plan, path)

    # --- build the stack (executor-facing only; same as solver.py) ---
    for k in range(climb.stack):
        eye = ROBOT_EYE + (k + 1) * BOULDER_HEIGHT
        a = Action("create", T_BOULDER, bx, by)
        plan.steps.append(
            PlanStep(
                action=a,
                note=f"stack boulder #{k+1} on {choice.base} (real climb)",
                stack_level=k + 1,
                eye_height=base_surf + eye,
            )
        )
    top_eye = base_surf + climb.eye_offset
    ra = Action("create", T_ROBOT, bx, by)
    plan.steps.append(
        PlanStep(
            action=ra,
            note=f"robot atop the {climb.stack}-boulder stack",
            stack_level=climb.stack + 1,
            eye_height=top_eye,
        )
    )
    ta = Action("transfer", bx, by)
    plan.steps.append(
        PlanStep(
            action=ta,
            note=f"transfer up: eye now {top_eye:.2f} (sweep vantage)",
            stack_level=climb.stack + 1,
            eye_height=top_eye,
        )
    )

    # --- sweep every collectible (Sentinel excluded) from the raised eye ---
    # These are model-replayable absorbs. The abstract eye is raised, but the model
    # itself computes eye from terrain only, so a raised-eye absorb is not something
    # game_model.apply gates on -- it accepts the absorb (energy/slot validity) while
    # the LOS feasibility is the per-object `can_absorb` we already proved in
    # best_base(). This keeps model_replay HONEST about energy/slots and lets the
    # live `play_plan` / executor re-verify the raised-eye LOS (it does for the win).
    collectibles = sorted(
        [o for o in choice.cover if o.type != T_SENTINEL],
        key=lambda o: (max(abs(o.x - bx), abs(o.y - by)), o.slot),
    )
    for o in collectibles:
        # the object may have been removed already (defensive); resolve live slot.
        live = model.state.object_by_slot(o.slot)
        if live is None:
            continue
        a = Action("absorb", o.x, o.y)
        plan.steps.append(
            PlanStep(
                action=a,
                note=f"sweep {o.type_name} @ {o.x},{o.y} from vantage",
                stack_level=climb.stack + 1,
                eye_height=top_eye,
            )
        )
        plan.model_replay.append(a)
        model = model.apply(a)
        plan.absorbed.append((o.type_name, o.x, o.y))

    # --- win on the Sentinel LAST ---
    plan.los_proven = cached_can_see(
        state, (bx, by), (sent.x, sent.y), eye_offset=climb.eye_offset
    )
    win = Action("win", sent.x, sent.y)
    plan.steps.append(
        PlanStep(
            action=win,
            note=f"absorb Sentinel @ {sent.x},{sent.y} + transfer onto platform",
            eye_height=top_eye,
        )
    )
    plan.model_replay.append(win)
    model = model.apply(win)
    plan.absorbed.append(("SENTINEL", sent.x, sent.y))

    # --- Stage D: re-absorb stranded climb structure (executor-facing) ---
    for k in range(climb.stack):
        plan.steps.append(
            PlanStep(
                action=Action("absorb", bx, by),
                note=f"re-absorb stranded boulder #{climb.stack-k} @ {choice.base}",
            )
        )
    plan.steps.append(
        PlanStep(
            action=Action("absorb", bx, by),
            note=f"re-absorb the climb robot @ {choice.base}",
        )
    )
    plan.recoverable_structure_energy = ENERGY_IN_OBJECTS[T_ROBOT]

    # --- replay-verify (same guarantee solver.py gives) ---
    won = _replay_verify(state, plan.model_replay)
    plan.solved = bool(won and plan.los_proven)
    if not won:
        plan.notes.append("model-replay projection did not reach won")
    if not plan.los_proven:
        plan.notes.append("abstract LOS proof failed (stack does not reveal Sentinel)")

    # --- safety-filter note ---
    if choice.meanie_risk:
        risk_tiles = sorted({t for _e, t in choice.meanie_risk})
        plan.notes.append(
            f"SAFETY: chosen base {choice.base} carries a meanie-spawn risk -- "
            f"an enemy could convert object(s) at {risk_tiles} into a meanie that "
            f"could force-hyperspace the player (no meanie-free winning base "
            f"available). The live executor must mind the rotation timing."
        )
    else:
        plan.notes.append(f"SAFETY: chosen base {choice.base} is meanie-free.")

    _finalise_energy(model, plan, climb)
    if plan.solved:
        assert _replay_verify(
            state, plan.model_replay
        ), "solved closure plan must replay through game_model.apply to won"
    return plan


# ---- public: solve ----------------------------------------------------------
def solve(state_or_model, verbose: bool = False) -> Plan:
    """Plan a winning sequence by the monotone reachability-closure / best-base
    sweep. Returns a `Plan` (same shape solver.solve emits and the executor /
    code_engine.play_plan consume).

    Guarantees when `plan.solved`:
      * replaying `plan.model_replay` through game_model.apply reaches won
        (asserted), and
      * `plan.los_proven` is True (the abstract climb reveals the Sentinel).

    Optimality: the cover set is the provably-maximum absorbable-object set within
    the reachability abstraction (per-object min stack is exact under monotonicity,
    then a global argmax over reachable bases). See SOLVER_ALGORITHMS.md S4."""
    state = copy.deepcopy(_as_state(state_or_model))
    if not hasattr(state, "won"):
        state.won = False

    sent = _sentinel(state)
    if sent is None:
        plan = Plan()
        plan.notes.append("no Sentinel in landscape")
        return plan

    reachable = _reachability_graph(state)
    choice = best_base(state, reachable)
    if choice is None:
        plan = Plan()
        plan.notes.append(
            "no reachable vantage gives LOS to the Sentinel "
            "(cannot build high enough within the reachable component)"
        )
        plan.solved = False
        return plan

    plan = _emit_closure_plan(state, choice)
    if verbose:
        for s in plan.steps:
            print(f"  {s.action!r:40} {s.note}")
    return plan


# ---- public: analyse_solvability -------------------------------------------
def analyse_solvability(state_or_model) -> Report:
    """Decide whether the landscape is solvable by the closure and, if not, WHY --
    distinguishing in-component reachability from a random-hyperspace possibility.

    Criteria:
      1.  no Sentinel / degenerate map.
      1b. Sentinel on the ground (no platform): the winning transfer is impossible.
      2.  PROVEN unreachable vantage: NO tile on the whole board sees the Sentinel
          even with the maximum buildable eye -> you can never build high enough,
          and hyperspace cannot help either (no vantage exists anywhere).
      3.  No IN-COMPONENT vantage reaches the Sentinel, but SOME board tile does:
          reported HONESTLY as "unsolvable within no-hyperspace reachability (a
          hyperspace escape might reach another region, but its destination is
          random and not planned here)" -- and we note that hyperspace COULD help
          (a Sentinel-absorbing vantage exists outside the start component).
      4.  Insufficient starting energy for the minimal climb structure.
      5.  Otherwise run the planner; success => solvable, else honest search-failure.
    """
    state = copy.deepcopy(_as_state(state_or_model))
    if not hasattr(state, "won"):
        state.won = False
    sent = _sentinel(state)
    if sent is None:
        return Report(False, "degenerate landscape: no Sentinel present")
    starg = (sent.x, sent.y)

    # (1b) Sentinel must sit on its platform for the win transfer.
    if sent.on_ground:
        return Report(
            False,
            "Sentinel is on the ground, not on a platform: the winning "
            "transfer-onto-platform is impossible",
        )

    tall = ROBOT_EYE + (MAX_STACK - 1) * BOULDER_HEIGHT

    # Is there ANY board tile from which a tall-enough eye sees the Sentinel?
    any_los_tile = False
    any_los_example = None
    for y in range(N):
        for x in range(N):
            if (x, y) == starg:
                continue
            if cached_can_see(state, (x, y), starg, eye_offset=tall):
                any_los_tile = True
                any_los_example = (x, y)
                break
        if any_los_tile:
            break

    # (2) no vantage anywhere -> absolutely impossible (hyperspace can't help).
    if not any_los_tile:
        return Report(
            False,
            "Sentinel can never be brought into line of sight: NO tile on "
            "the board sees it even with the maximum buildable eye "
            f"(stack {MAX_STACK}). Cannot build high enough; a hyperspace "
            "escape cannot help either (no Sentinel-absorbing vantage "
            "exists anywhere).",
            detail={"max_eye_offset": tall},
        )

    reachable = _reachability_graph(state)
    choice = best_base(state, reachable)

    # (3) no in-component vantage, but a vantage exists somewhere on the board.
    if choice is None:
        # Does ANY tile (not just the start component) admit a Sentinel-absorbing
        # vantage? (any_los_tile already proved yes.) That tells us hyperspace COULD
        # help -- its random destination might land in a region that can win.
        return Report(
            False,
            "unsolvable within no-hyperspace reachability (a hyperspace "
            "escape might reach another region, but its destination is "
            "random and not planned here). A Sentinel-absorbing vantage "
            f"DOES exist elsewhere on the board (e.g. near {any_los_example}), "
            "so a lucky hyperspace COULD help -- it just cannot be aimed.",
            detail={
                "reachable_tiles": len(reachable),
                "out_of_component_vantage": any_los_example,
                "hyperspace_might_help": True,
            },
        )

    # (4) energy for the minimal climb structure.
    min_build = ENERGY_IN_OBJECTS[T_ROBOT]
    if state.player_energy < min_build:
        return Report(
            False,
            f"insufficient starting energy ({state.player_energy}) to "
            f"create the climbing structure (needs >= {min_build})",
            detail={"start_energy": state.player_energy},
        )

    plan = solve(state)
    if plan.solved:
        return Report(
            True,
            "",
            detail={
                "base": choice.base,
                "stack": choice.max_stack,
                "objects": plan.absorbed_count,
                "final_energy": plan.final_energy,
                "meanie_safe": not choice.meanie_risk,
            },
        )
    return Report(
        False,
        "planner did not find a winning sequence (NOT proven impossible: "
        "an in-component vantage exists and energy suffices). " + "; ".join(plan.notes),
        detail={"notes": plan.notes},
    )


# ---- __main__ validation ----------------------------------------------------
def _report_landscape(ls: int) -> None:
    print(f"\n############ seed {ls} ############")
    model = GameModel.from_landscape(ls)
    t0 = time.time()
    rep = analyse_solvability(model)
    if not rep.solvable:
        print(f"  solvable: False\n  reason  : {rep.reason}")
        return
    plan = solve(model)
    dt = time.time() - t0
    print(f"  solvable           : True")
    print(
        f"  plan length        : {len(plan.steps)} steps "
        f"(model-replay {len(plan.model_replay)} actions)"
    )
    print(f"  objects absorbed   : {plan.absorbed_count}  {plan.absorbed_types()}")
    print(f"  predicted energy   : {plan.final_energy}")
    print(f"  LOS proven (climb) : {plan.los_proven}")
    ok = _replay_verify(_as_state(model), plan.model_replay)
    print(f"  replay-verified    : model-replay reaches won == {ok}")
    print(f"  time               : {dt:.3f}s")
    for n in plan.notes:
        if n.startswith("SAFETY"):
            print(f"  {n}")


def main():
    for ls in (0, 42, 9999):
        _report_landscape(ls)


if __name__ == "__main__":
    main()
