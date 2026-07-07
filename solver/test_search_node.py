#!/usr/bin/env python3
"""Tests for the search Node + closed-set key."""

import dataclasses
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from solver.plan_game import PlanGame, terrain_z, N  # noqa: E402
from solver.search_node import (  # noqa: E402
    Node,
    eye,
    energy,
    tile,
    enemy_phase_hash,
    node_key,
    step_world,
)
from sentinel import actions, enemies  # noqa: E402
from sentinel import memmap as mm  # noqa: E402


def _adjacent_bare_tile(g):
    """A bare-terrain tile next to the player that isn't the player's own tile."""
    px, py = g.player_xy()
    for dx, dy in ((1, 0), (-1, 0), (0, 1), (0, -1), (1, 1), (-1, -1)):
        t = (px + dx, py + dy)
        if 0 <= t[0] < N and 0 <= t[1] < N and terrain_z(g.mem, *t) is not None:
            return t
    raise AssertionError("no adjacent bare tile")


def _start_node():
    g = PlanGame(0)
    g.energy = 30
    return Node(g=g, t=0, vh=0, vv=0, cost=0.0)


def test_accessors():
    n = _start_node()
    assert eye(n) == n.g.eye
    assert energy(n) == n.g.state.energy == 30
    assert tile(n) == n.g.player_xy()


def test_clone_apply_matches_raw_actions():
    """A boulder+synthoid create then transfer, applied via the node's PlanGame,
    matches raw sentinel.actions on an independent State cloned from the same
    start -- equal energy, player tile and eye."""
    base = _start_node()
    t2 = _adjacent_bare_tile(base.g)

    # path (a): via the node's PlanGame (independent clone of the start).
    n = dataclasses.replace(base, g=base.g.clone())
    n.g.create(mm.T_BOULDER, t2, None, "boulder")
    slot_s = n.g.create(mm.T_ROBOT, t2, None, "synthoid on boulder")
    n.g.transfer(slot_s, "step up")

    # path (b): raw sentinel.actions on a parallel independent State cloned from
    # the identical start (same otype/tile sequence, so PRNG advances identically).
    raw = base.g.state.clone()
    actions.create(raw, mm.T_BOULDER, t2)
    raw_slot = actions.create(raw, mm.T_ROBOT, t2)
    actions.transfer(raw, raw_slot)

    assert energy(n) == raw.energy
    assert tile(n) == tuple(raw.player_xy())
    assert eye(n) == raw.eye_z()
    assert tile(n) == t2  # transferred onto the built stack
    assert eye(n) > base.g.eye  # the stack raised the eye


def test_single_create_energy_matches_raw():
    """A lighter cross-check: one PlanGame.create moves energy by the same amount
    actions.create does on a raw clone of the same start."""
    base = _start_node()
    t2 = _adjacent_bare_tile(base.g)

    n = dataclasses.replace(base, g=base.g.clone())
    e_before = energy(n)
    n.g.create(mm.T_BOULDER, t2, None)
    plan_delta = e_before - energy(n)

    raw = base.g.state.clone()
    r_before = raw.energy
    actions.create(raw, mm.T_BOULDER, t2)
    raw_delta = r_before - raw.energy

    assert plan_delta == raw_delta == mm.ENERGY_IN_OBJECTS[mm.T_BOULDER]


def test_node_key_stable_across_clone():
    """Cloning a node's PlanGame yields an identical node_key."""
    n = _start_node()
    n2 = dataclasses.replace(n, g=n.g.clone())
    assert node_key(n) == node_key(n2)


def test_node_key_two_clone_determinism():
    """The same ops on two clones of a node give identical energy/tile/eye and an
    identical node_key."""
    base = _start_node()
    t2 = _adjacent_bare_tile(base.g)

    def apply(node):
        node.g.create(mm.T_BOULDER, t2, None)
        s = node.g.create(mm.T_ROBOT, t2, None)
        node.g.transfer(s)
        step_world(node.g, 5)
        return node

    a = apply(dataclasses.replace(base, g=base.g.clone(), t=0))
    b = apply(dataclasses.replace(base, g=base.g.clone(), t=0))
    a.t = a.t + 5
    b.t = b.t + 5
    assert energy(a) == energy(b)
    assert tile(a) == tile(b)
    assert eye(a) == eye(b)
    assert node_key(a) == node_key(b)


def test_node_key_changes_with_energy():
    n = _start_node()
    n2 = dataclasses.replace(n, g=n.g.clone())
    n2.g.energy = 10
    assert node_key(n) != node_key(n2)


def test_enemy_phase_hash_is_hashable_tuple():
    n = _start_node()
    ph = enemy_phase_hash(n.g.state)
    assert isinstance(ph, tuple)
    hash(ph)  # must not raise
    # one entry per enemy slot, plus the trailing meanie-set tuple.
    assert len(ph) == len(enemies.enemy_slots(n.g.state)) + 1
    assert isinstance(ph[-1], tuple)


def test_step_world_advances_tick():
    n = _start_node()
    before = enemy_phase_hash(n.g.state)
    step_world(n.g, 300)
    after = enemy_phase_hash(n.g.state)
    # some enemy phase should have advanced over 300 rounds (rotation/cooldown).
    assert before != after
