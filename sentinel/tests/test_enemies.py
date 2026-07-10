"""The enemy round advance reproduces the game's cooldown/rotation/target/drain
machinery over the enemy arrays.

``golden_enemies.json`` holds, per landscape, the enemy arrays + cursor + energy
captured from the ROM every 25 rounds for 400 rounds (rendering/sound stubbed);
``test_step_matches_golden`` replays them with no emulator. The remaining tests
exercise the pieces directly.
"""

import hashlib
import json
import os

import pytest

from sentinel import landscape, enemies, memmap as mm
from sentinel.tests import oracle

GOLDEN = os.path.join(os.path.dirname(__file__), "golden_enemies.json")
GOLDEN_MEANIE = os.path.join(os.path.dirname(__file__), "golden_meanie.json")

# Landscape 2024's enemy 2 sees the player only partially (head, not base) at
# spawn, so the passive round-advance drives the full meanie lifecycle: an enemy
# targets the player ($110), spawns a meanie from a nearby tree ($470), the meanie
# rotates round to face the player and forces a hyperspace ($814, player object
# 62->45, energy 10->7), the spent meanie is dropped ($830), and much later a drain
# finishes the player off ($2486). Landscape 49 has no createable meanie for its
# partially-seen player, so it exercises the failed-attempt path instead: the enemy
# arms and re-arms its considering-meanie flag ($1039..), gives up after two scans
# ($1056), and decays the flag exactly once. Checkpoints bracket each transition.
MEANIE_RUNS = {
    2024: (
        110,
        469,
        470,
        471,
        500,
        813,
        814,
        815,
        829,
        830,
        831,
        1000,
        1200,
        1500,
        1800,
        2100,
        2400,
        2485,
        2486,
    ),
    49: (1000, 1039, 1040, 1056, 1063, 1080, 1200, 1400, 1599),
}


def _snap(m):
    return (
        bytes(m[0x09C0 : 0x09C0 + 8]).hex()
        + bytes(m[0x0C20:0x0C38]).hex()
        + bytes(m[0x0CA8:0x0CB8]).hex()
        + "%02x%02x%02x" % (m[0x0090], m[0x0C50], m[0x0C0A])
    )


def _meanie_snap(m):
    """A full-lifecycle snapshot: every object array (flags, v/h angle, x/y/z,
    z-fraction, type), the enemy cooldowns ($0C20-$0C38), the whole meanie array
    block ($0C80-$0CC0: search/discharge/failed-memory/attempts/meanie-object/
    targeted/exposure/considering), the PRNG, the player/energy/cursor/gate and
    the death+hyperspace flags, plus a digest of the 32x32 tile grid."""
    core = (
        bytes(m[0x0100:0x0140])  # flags
        + bytes(m[0x0140:0x0180])  # v_angle
        + bytes(m[0x0900:0x0940])  # x
        + bytes(m[0x0940:0x0980])  # z_height
        + bytes(m[0x0980:0x09C0])  # y
        + bytes(m[0x09C0:0x0A00])  # h_angle
        + bytes(m[0x0A00:0x0A40])  # z_fraction
        + bytes(m[0x0A40:0x0A80])  # type
        + bytes(m[0x0C20:0x0C38])  # draining/rotation/update cooldowns
        + bytes(m[0x0C80:0x0CC0])  # meanie arrays
        + bytes(m[0x0C7B:0x0C80])  # prng
        + bytes([m[0x000B], m[0x0C0A], m[0x0090], m[0x0C50], m[0x0C4E], m[0x0CDE]])
    )
    return core.hex() + hashlib.sha256(bytes(m[0x0400:0x0800])).hexdigest()[:16]


def _drive_meanie_rom(ls, checkpoints):
    """Drive the real 6502 enemy loop over `ls` and return {round: snap} at the
    given checkpoint rounds (needs the ROM image)."""
    cpu, mem, state = oracle.generate_machine(ls)
    oracle.prime_enemy_driver(cpu, mem, state)
    out = {}
    for rnd in range(1, max(checkpoints) + 1):
        oracle.step_enemy_round(cpu, mem, state)
        if rnd in checkpoints:
            out[str(rnd)] = _meanie_snap(mem)
    return out


def _drive_meanie_golden():
    """{landscape: {round: snap}} across every MEANIE_RUNS landscape (needs ROM)."""
    return {str(ls): _drive_meanie_rom(ls, cps) for ls, cps in MEANIE_RUNS.items()}


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


def _replay_meanie(ls):
    """A pure-sim state primed like the golden replay, run to the last checkpoint."""
    state = landscape.generate(ls)
    state.mem[mm.CURSOR] = 7
    state.mem[mm.COOLDOWN_GATE] = 0
    return state


def test_meanie_lifecycle_matches_golden():
    """Replay (no emulator) the frozen meanie golden per landscape and, for the
    full-lifecycle run, assert it actually fired: meanie spawned, player hyperspaced
    by it, player eventually drain-killed."""
    with open(GOLDEN_MEANIE) as f:
        data = json.load(f)
    assert set(data) == {str(ls) for ls in MEANIE_RUNS}
    for ls, checkpoints in MEANIE_RUNS.items():
        want = data[str(ls)]
        state = _replay_meanie(ls)
        saw_meanie = False
        saw_hyperspace = False
        player0 = state.mem[mm.PLAYER_OBJECT]
        for rnd in range(1, max(checkpoints) + 1):
            enemies.step(state)
            if any(
                not (state.mem[mm.ENEMIES_MEANIE_OBJECT + e] & 0x80) for e in range(8)
            ):
                saw_meanie = True
            if state.mem[mm.PLAYER_OBJECT] != player0:
                saw_hyperspace = True
            if str(rnd) in want:
                assert _meanie_snap(state.mem) == want[str(rnd)], f"ls{ls} round {rnd}"
        if ls == 2024:
            assert saw_meanie, "no meanie was ever created"
            assert saw_hyperspace, "the meanie never hyperspaced the player"
            assert (
                state.mem[mm.PLAYER_DIED_BY_DRAINING] & 0x80
            ), "player not drain-killed"


@pytest.mark.oracle
def test_meanie_lifecycle_matches_rom_live():
    """Drive the pure sim and the real 6502 code from identical play-setup memory,
    round by round through the whole meanie lifecycle, asserting the full state
    (object table, meanie/enemy arrays, player, energy, PRNG, tiles, death and
    hyperspace flags) matches bit-for-bit at every round (needs the ROM image)."""
    for ls, checkpoints in MEANIE_RUNS.items():
        cpu, mem, state = oracle.generate_machine(ls)
        oracle.prime_enemy_driver(cpu, mem, state)
        sim = _replay_meanie(ls)
        for rnd in range(1, max(checkpoints) + 1):
            oracle.step_enemy_round(cpu, mem, state)
            enemies.step(sim)
            assert _meanie_snap(sim.mem) == _meanie_snap(mem), f"ls{ls} round {rnd}"
