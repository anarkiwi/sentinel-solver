"""Enemy dynamics: rotation, targeting, draining and cooldowns, ported onto the
bit-exact state and the object-relative visibility of :mod:`sentinel.relative`.

One :func:`step` advances the world by a single game round, mirroring the play
loop's ``update_enemy_cooldowns`` ($1317) + ``update_enemies`` ($16B5) pair:

  * cooldowns tick on a 1-in-3 cadence, each sticking at 1 ($0C50 gate);
  * exactly one enemy -- the slot at the cursor $0090, counting 7->0 and
    wrapping -- is considered (``consider_enemy_state`` $16E6):
      - if its update cooldown is still >= 2 it is skipped;
      - otherwise it re-checks a held target, then scans slots $3F..0 for a
        drainable robot/player (``find_drainable_robot_loop`` $17B2), else a
        tree/boulder; a fully-visible target is drained (``reduce_object_energy``
        $1A08: player -1 energy, robot->boulder->tree->removed) and the update
        cooldown reloads to 30; with nothing to drain and its rotation cooldown
        low, the enemy rotates ($9D37,X added to its facing, cooldown -> 200).

The rendering-coupled side effects of the ROM routine (object re-plotting, sound,
the energy-discharge that scatters trees when an enemy is absorbed) do not change
the gameplay state a strategy search reasons about and are not modelled here; the
meanie lifecycle is exposed separately via :func:`meanie_threat`.

The cooldown, rotation, targeting and drain machinery reproduces the ROM round for
round (validated bit-exact over the enemy arrays for hundreds of rounds). The one
approximation is the exposure byte stored for a target: the ROM's two-probe
$0014 accumulates a full ($80) / partial ($40) classification whose multi-probe
bit-plumbing is not fully reconstructed, so a rare rotated-angle target may be
classed partial where the ROM classes it fully visible.
"""

from sentinel import memmap as mm, relative, actions, terrain
from sentinel.prng import Prng

FOV_SCAN = 0x14  # $16F2: enemy horizontal FOV width during a scan
UPDATE_COOLDOWN_SCAN = 0x04  # $16ED
UPDATE_COOLDOWN_DRAIN = 0x1E  # $17F1 / $1848: 30 rounds after a drain
ROTATION_COOLDOWN_RELOAD = 0xC8  # $1813: 200 rounds after a rotation
DRAINING_COOLDOWN_RELOAD = 0x78  # $1835: 120 rounds when first targeting
COOLDOWN_STICK = 0x02  # thresholds compare against 2 ($16E9/$17FE/$1321)


def _enemy_slots(state):
    return [
        s
        for s in range(mm.NUM_SLOTS)
        if not state.is_empty(s) and state.obj_type[s] in mm.ENEMY_TYPES
    ]


# ---------------------------------------------------------------------------
# update_enemy_cooldowns $1317
# ---------------------------------------------------------------------------
def tick_cooldowns(state):
    """$1317: on 2-of-3 rounds decrement the $0C50 gate; on the third, decrement
    every draining/rotation/update cooldown that is >= 2 (they stick at 1)."""
    mem = state.mem
    if mem[mm.COOLDOWN_GATE] != 0:
        mem[mm.COOLDOWN_GATE] = (mem[mm.COOLDOWN_GATE] - 1) & 0xFF
        return
    for addr in range(mm.ENEMIES_DRAINING_COOLDOWN, mm.ENEMIES_UPDATE_COOLDOWN + 8):
        if mem[addr] >= COOLDOWN_STICK:
            mem[addr] -= 1
    mem[mm.COOLDOWN_GATE] = 2


# ---------------------------------------------------------------------------
# reduce_object_energy $1A08
# ---------------------------------------------------------------------------
def _reduce_object_energy(state, target):
    """$1A08: drain `target`. The player loses one energy (returns True); a robot
    downgrades to a boulder, a boulder to a tree, a tree is removed (returns
    False -- no player energy change)."""
    mem = state.mem
    if target == mem[mm.PLAYER_OBJECT]:
        if state.energy == 0:
            return True  # kill_player (out of energy)
        state.energy = state.energy - 1
        return True
    otype = state.obj_type[target]
    if otype == mm.T_ROBOT:
        state.obj_type[target] = mm.T_BOULDER
    elif otype == mm.T_TREE:
        actions.remove_object(state, target)
    else:  # boulder -> tree
        state.obj_type[target] = mm.T_TREE
    return False


# ---------------------------------------------------------------------------
# target_object $1825
# ---------------------------------------------------------------------------
def _target_object(state, enemy, target, exposure):
    """$1825: record the target and, once the draining cooldown counts down to 1
    with the target fully visible, drain it."""
    mem = state.mem
    mem[mm.ENEMIES_TARGETED_OBJECT + enemy] = target
    mem[mm.ENEMIES_TARGETED_OBJECT_EXPOSURE + enemy] = exposure
    cd = mem[mm.ENEMIES_DRAINING_COOLDOWN + enemy]
    if cd < 0x01:  # first sight -> arm the drain timer
        mem[mm.ENEMIES_DRAINING_COOLDOWN + enemy] = DRAINING_COOLDOWN_RELOAD
        return
    if cd != 0x01:  # still counting down
        return
    if not (exposure & 0x80):  # not fully visible -> can't drain (meanie path)
        return
    mem[mm.TARGETED_OBJECT_SLOT] = target
    _reduce_object_energy(state, target)
    mem[mm.ENEMIES_UPDATE_COOLDOWN + enemy] = UPDATE_COOLDOWN_DRAIN


# ---------------------------------------------------------------------------
# rotate_enemy $1805
# ---------------------------------------------------------------------------
def _rotate_enemy(state, enemy):
    """$1805: add the per-enemy rotation step to its facing; reload the rotation
    cooldown to 200."""
    mem = state.mem
    step = mem[mm.ROTATION_SPEED_TABLE + enemy]
    state.obj_h_angle[enemy] = (state.obj_h_angle[enemy] + step) & 0xFF
    mem[mm.ENEMIES_ROTATION_COOLDOWN + enemy] = ROTATION_COOLDOWN_RELOAD


# ---------------------------------------------------------------------------
# consider_enemy_state $16E6 (no-meanie path)
# ---------------------------------------------------------------------------
def _consider_enemy_state(state, enemy):
    mem = state.mem
    if mem[mm.ENEMIES_UPDATE_COOLDOWN + enemy] >= COOLDOWN_STICK:
        return
    mem[mm.ENEMIES_UPDATE_COOLDOWN + enemy] = UPDATE_COOLDOWN_SCAN
    mem[mm.FOV_WIDTH] = FOV_SCAN

    # Re-check a held target ($1795): drop it if no longer fully visible.
    mem[mm.ENEMIES_CONSIDERING_MEANIE + enemy] = (
        mem[mm.ENEMIES_CONSIDERING_MEANIE + enemy] >> 1
    )
    if mem[mm.ENEMIES_DRAINING_COOLDOWN + enemy] != 0:
        held = mem[mm.ENEMIES_TARGETED_OBJECT + enemy]
        see = relative.can_see_object(state, enemy, held, mm.T_ROBOT, FOV_SCAN)
        if see["full"]:
            _target_object(state, enemy, held, 0x80)
            return
        mem[mm.ENEMIES_DRAINING_COOLDOWN + enemy] = 0  # target lost

    # find_drainable_robot_loop ($17B2): scan slots 63..0 for the player/a robot.
    player = mem[mm.PLAYER_OBJECT]
    partial_player = None
    for y in range(mm.NUM_SLOTS - 1, -1, -1):
        see = relative.can_see_object(state, enemy, y, mm.T_ROBOT, FOV_SCAN)
        if not see["in_slot"] or not see["in_fov"]:
            continue
        if see["full"]:
            _target_object(state, enemy, y, 0x80)
            return
        if y == player:  # in view but not fully visible -> meanie candidate
            partial_player = y
    if partial_player is not None:
        # The ROM arms the meanie search here; the meanie lifecycle is modelled by
        # meanie_threat(), so we only record the partial exposure and target.
        _target_object(state, enemy, partial_player, 0x40)
        return

    # reset_draining_cooldown_and_look_for_tree_or_boulder ($17E0).
    mem[mm.ENEMIES_DRAINING_COOLDOWN + enemy] = 0
    tb = _find_drainable_boulder_or_tree(state, enemy)
    if tb is not None:
        mem[mm.TARGETED_OBJECT_SLOT] = tb
        _reduce_object_energy(state, tb)
        mem[mm.ENEMIES_UPDATE_COOLDOWN + enemy] = UPDATE_COOLDOWN_DRAIN
        return

    # no_drain ($17F9): rotate if the rotation cooldown is low.
    if mem[mm.ENEMIES_ROTATION_COOLDOWN + enemy] < COOLDOWN_STICK:
        _rotate_enemy(state, enemy)


def _find_drainable_boulder_or_tree(state, enemy):
    """find_drainable_boulder_or_tree_on_stack ($1AB0): scanning slots 63..0, a
    boulder or a stacked object (flags >= $40) marks a candidate tile; the tile's
    topmost object -- if a tree or boulder the enemy can fully see -- is drained.
    Lone ground trees are not drainable this way."""
    for x in range(mm.NUM_SLOTS - 1, -1, -1):
        flags = state.obj_flags[x]
        if flags & 0x80:  # empty slot
            continue
        if not (flags >= 0x40 or state.obj_type[x] == mm.T_BOULDER):
            continue
        tb = terrain.tile_byte(state, state.obj_x[x], state.obj_y[x])
        if tb < mm.OBJECT_TILE:  # no object on the tile (shouldn't happen)
            continue
        y = tb & 0x3F  # topmost object of the tile
        otype = state.obj_type[y]
        if otype not in (mm.T_TREE, mm.T_BOULDER):
            continue
        see = relative.can_see_object(state, enemy, y, otype, FOV_SCAN)
        if see["full"]:
            state.mem[mm.TARGETED_OBJECT_SLOT] = y
            return y
    return None


# ---------------------------------------------------------------------------
# update_enemies $16B5
# ---------------------------------------------------------------------------
def _dec_cursor(state):
    mem = state.mem
    c = mem[mm.CURSOR]
    mem[mm.CURSOR] = (c - 1) if c > 0 else 7


def update_enemies(state):
    """$16B5: consider the enemy at the cursor, advance the PRNG ($16D6) and the
    cursor ($16D9, 7->0 wrap)."""
    mem = state.mem
    x = mem[mm.CURSOR]
    if state.obj_type[x] in mm.ENEMY_TYPES and not (state.obj_flags[x] & 0x80):
        _consider_enemy_state(state, x)
    prng = Prng().load(mem)  # $16D6 JSR prnd (advances the stream)
    prng.next()
    prng.store(mem)
    _dec_cursor(state)


def step(state):
    """Advance the world by one game round: cooldown tick + one enemy update."""
    tick_cooldowns(state)
    update_enemies(state)


# ---------------------------------------------------------------------------
# meanie threat (the forced-hyperspace side channel)
# ---------------------------------------------------------------------------
def meanie_threat(state, enemy):
    """Whether `enemy` currently sees the player partially (in its field of view
    but without a clear sight of the player's own tile) -- the condition under
    which the ROM arms a meanie that can later hyperspace the player. Returns the
    player slot when threatened, else None."""
    player = state.mem[mm.PLAYER_OBJECT]
    see = relative.can_see_object(state, enemy, player, mm.T_ROBOT, FOV_SCAN)
    if see["in_slot"] and see["in_fov"] and not see["full"]:
        return player
    return None
