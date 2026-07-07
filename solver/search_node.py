#!/usr/bin/env python3
"""Search node + closed-set key for the macro planner.

A :class:`Node` wraps a :class:`solver.plan_game.PlanGame` (reusing its
``col``/``eye``/``steps`` tracking and bit-exact ``create``/``absorb``/``transfer``)
plus a world tick and the sights heading.  ``node_key`` collapses states that
continue identically enough to dedup in the closed set.
"""

from dataclasses import dataclass
from typing import Optional

from solver.plan_game import PlanGame
from sentinel import enemies, memmap as mm

T_BUCKET = 64


@dataclass
class Node:
    """One search state: a branchable ``PlanGame`` plus world tick and sights."""

    g: PlanGame  # holds sentinel State incl PRNG; .clone() branches
    t: int  # world tick: number of enemies.step applied since start
    vh: int  # sights bearing 0..255 (current heading)
    vv: int  # sights pitch (v_angle)
    cost: float  # g-cost: cumulative enemy-rounds
    parent: Optional["Node"] = None
    macro: Optional[dict] = None  # record of the macro that produced this node


def eye(n: Node) -> float:
    return n.g.eye


def energy(n: Node) -> int:
    return n.g.state.energy


def tile(n: Node) -> tuple:
    return n.g.player_xy()


def step_world(g: PlanGame, rounds: int) -> None:
    """Advance the real enemy world ``rounds`` rounds on ``g.state``
    (drain/rotate/meanie)."""
    for _ in range(int(rounds)):
        enemies.step(g.state)


def enemy_phase_hash(state) -> tuple:
    """Coarse enemy phase: per enemy (type, bearing>>3, mid-cooldown), plus the
    set of live meanie slots."""
    ph = tuple(
        sorted(
            (
                int(state.obj_type[e]),
                int(state.obj_h_angle[e]) >> 3,
                1 if state.mem[mm.ENEMIES_UPDATE_COOLDOWN + e] >= 2 else 0,
            )
            for e in enemies.enemy_slots(state)
        )
    )
    meanies = tuple(
        sorted(
            s
            for s in range(mm.NUM_SLOTS)
            if not state.is_empty(s) and state.obj_type[s] == mm.T_MEANIE
        )
    )
    return ph + (meanies,)


def node_key(n: Node) -> tuple:
    """Closed-set dedup key: tile, eye, energy, coarse tick bucket, enemy phase."""
    st = n.g.state
    return (
        n.g.player_xy(),
        round(n.g.eye, 3),
        st.energy,
        n.t // T_BUCKET,
        enemy_phase_hash(st),
    )
