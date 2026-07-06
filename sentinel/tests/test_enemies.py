"""The enemy round advance reproduces the game's cooldown/rotation/target/drain
machinery over the enemy arrays.

``golden_enemies.json`` holds, per landscape, the enemy arrays + cursor + energy
captured from the ROM every 25 rounds for 400 rounds (rendering/sound stubbed);
``test_step_matches_golden`` replays them with no emulator. The remaining tests
exercise the pieces directly.
"""

import json
import os

from sentinel import landscape, enemies, memmap as mm

GOLDEN = os.path.join(os.path.dirname(__file__), "golden_enemies.json")


def _snap(m):
    return (
        bytes(m[0x09C0 : 0x09C0 + 8]).hex()
        + bytes(m[0x0C20:0x0C38]).hex()
        + bytes(m[0x0CA8:0x0CB8]).hex()
        + "%02x%02x%02x" % (m[0x0090], m[0x0C50], m[0x0C0A])
    )


def test_step_matches_golden():
    with open(GOLDEN) as f:
        data = json.load(f)
    assert data
    for ls, checkpoints in data.items():
        state = landscape.generate(int(ls))
        state.mem[mm.CURSOR] = 7
        state.mem[mm.COOLDOWN_GATE] = 0
        last = max(int(r) for r in checkpoints)
        for rnd in range(1, last + 1):
            enemies.step(state)
            if str(rnd) in checkpoints:
                assert _snap(state.mem) == checkpoints[str(rnd)], f"ls{ls} round {rnd}"


def test_cooldown_cadence_is_one_in_three():
    state = landscape.generate(42)
    m = state.mem
    m[mm.COOLDOWN_GATE] = 0
    enemy = enemies._enemy_slots(state)[0]
    m[mm.ENEMIES_ROTATION_COOLDOWN + enemy] = 9
    # gate 0 -> decrement now, reload gate to 2; then two rounds of no decrement.
    enemies.tick_cooldowns(state)
    assert m[mm.ENEMIES_ROTATION_COOLDOWN + enemy] == 8
    assert m[mm.COOLDOWN_GATE] == 2
    enemies.tick_cooldowns(state)
    enemies.tick_cooldowns(state)
    assert m[mm.ENEMIES_ROTATION_COOLDOWN + enemy] == 8  # unchanged for two rounds
    enemies.tick_cooldowns(state)
    assert m[mm.ENEMIES_ROTATION_COOLDOWN + enemy] == 7


def test_cooldown_sticks_at_one():
    state = landscape.generate(0)
    m = state.mem
    m[mm.COOLDOWN_GATE] = 0
    m[mm.ENEMIES_DRAINING_COOLDOWN] = 1
    enemies.tick_cooldowns(state)
    assert m[mm.ENEMIES_DRAINING_COOLDOWN] == 1  # 1 does not decrement to 0


def test_rotation_steps_by_speed_table():
    state = landscape.generate(42)
    enemy = enemies._enemy_slots(state)[0]
    before = state.obj_h_angle[enemy]
    step = state.mem[mm.ROTATION_SPEED_TABLE + enemy]
    enemies._rotate_enemy(state, enemy)
    assert state.obj_h_angle[enemy] == (before + step) & 0xFF
    assert state.mem[mm.ENEMIES_ROTATION_COOLDOWN + enemy] == 0xC8


def test_reduce_object_energy_downgrades():
    state = landscape.generate(42)
    enemy = enemies._enemy_slots(state)[0]
    # a tree is removed, a boulder becomes a tree, a robot becomes a boulder.
    tree = state.slot_of_type(mm.T_TREE)
    d0 = state.mem[mm.ENEMIES_ENERGY_TO_DISCHARGE + enemy]
    assert enemies._reduce_object_energy(state, tree, enemy) is False
    assert state.is_empty(tree)
    # every non-kill drain banks one unit of energy to discharge later as a tree.
    assert state.mem[mm.ENEMIES_ENERGY_TO_DISCHARGE + enemy] == (d0 + 1) & 0xFF
    # player loses one energy (and banks another discharge unit).
    e0 = state.energy
    drained = enemies._reduce_object_energy(state, state.mem[mm.PLAYER_OBJECT], enemy)
    assert drained is True and state.energy == e0 - 1
    assert state.mem[mm.ENEMIES_ENERGY_TO_DISCHARGE + enemy] == (d0 + 2) & 0xFF


def test_consider_discharging_scatters_tree():
    state = landscape.generate(42)
    enemy = enemies._enemy_slots(state)[0]
    # nothing banked -> no discharge.
    state.mem[mm.ENEMIES_ENERGY_TO_DISCHARGE + enemy] = 0
    assert enemies._consider_discharging_enemy_energy(state, enemy) is False
    # one unit banked -> one new tree placed, bank decremented.
    trees0 = sum(
        1
        for s in range(mm.NUM_SLOTS)
        if not state.is_empty(s) and state.obj_type[s] == mm.T_TREE
    )
    state.mem[mm.ENEMIES_ENERGY_TO_DISCHARGE + enemy] = 1
    assert enemies._consider_discharging_enemy_energy(state, enemy) is True
    trees1 = sum(
        1
        for s in range(mm.NUM_SLOTS)
        if not state.is_empty(s) and state.obj_type[s] == mm.T_TREE
    )
    assert trees1 == trees0 + 1
    assert state.mem[mm.ENEMIES_ENERGY_TO_DISCHARGE + enemy] == 0


def test_drain_at_zero_energy_kills_player():
    state = landscape.generate(42)
    enemy = enemies._enemy_slots(state)[0]
    player = state.mem[mm.PLAYER_OBJECT]
    state.mem[mm.PLAYER_DIED_BY_DRAINING] = 0
    # draining the player with energy left just decrements and does not kill.
    state.energy = 1
    assert enemies._reduce_object_energy(state, player, enemy) is True
    assert state.energy == 0
    assert not (state.mem[mm.PLAYER_DIED_BY_DRAINING] & 0x80)
    # draining again at zero energy sets the death flag (kill_player $1A00).
    assert enemies._reduce_object_energy(state, player, enemy) is True
    assert state.mem[mm.PLAYER_DIED_BY_DRAINING] & 0x80


def test_meanie_threat_signature():
    state = landscape.generate(42)
    enemy = enemies._enemy_slots(state)[0]
    # returns None or the player slot; never raises.
    res = enemies.meanie_threat(state, enemy)
    assert res in (None, state.mem[mm.PLAYER_OBJECT])
