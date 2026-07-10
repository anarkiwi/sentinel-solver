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

Landscape-energy conservation IS modelled: every drain banks a unit on the enemy
(``reduce_object_energy`` $1A4F) which a later ``consider_discharging_enemy_energy``
($1A5D) returns to the board as a tree on a random flat tile -- so a Sentinel that
drains the player over the time an action takes scatters trees exactly as the live
ROM does.  The purely rendering-coupled side effects (object re-plotting, sound) do
not change the gameplay state and are not modelled.

The full meanie lifecycle is a stateful side effect of the round advance, not a
side-channel: when an enemy sees the player only partially ($0014 == $40, head but
not base) it arms a meanie, ``_consider_creating_meanie`` ($197D) turns a nearby
tree into one, ``_update_meanie`` ($16F2) rotates it round to face the player, and
``do_hyperspace`` ($2156) relocates the player (spending energy) or kills them.
:func:`meanie_threat` is only a planner-facing query over the same visibility test.

The cooldown, rotation, targeting, drain and meanie machinery reproduce the ROM
round for round, validated bit-exact against the py65 oracle over the full state
(object table, enemy + meanie arrays, player/energy, PRNG, tiles, death/hyperspace
flags) across the whole meanie lifecycle -- spawn, hunt, forced hyperspace and a
later drain-death -- as well as the failed-attempt path.
"""

from sentinel import memmap as mm, relative, actions, terrain
from sentinel.prng import Prng
from sentinel.terrain import tile_byte, set_tile_byte

FOV_SCAN = 0x14  # $16F2: enemy horizontal FOV width during a scan
FOV_CREATE_MEANIE = 0x28  # $197F: two screen widths while hunting a tree to convert
UPDATE_COOLDOWN_SCAN = 0x04  # $16ED
UPDATE_COOLDOWN_DRAIN = 0x1E  # $17F1 / $1848: 30 rounds after a drain
UPDATE_COOLDOWN_MEANIE_ROTATE = 0x0A  # $173A: 10 rounds after rotating a meanie
UPDATE_COOLDOWN_MEANIE_MADE = 0x32  # $1869: 50 rounds after creating a meanie
ROTATION_COOLDOWN_RELOAD = 0xC8  # $1813: 200 rounds after a rotation
DRAINING_COOLDOWN_RELOAD = 0x78  # $1835: 120 rounds when first targeting
COOLDOWN_STICK = 0x02  # thresholds compare against 2 ($16E9/$17FE/$1321)
MEANIE_ROTATE_STEP = 0x08  # $171B: meanie turns +/-8 units/update toward the player
MEANIE_MAX_ATTEMPTS = 0x02  # $1857: stop hunting a tree after two failed full scans


def enemy_slots(state):
    """The occupied sentry/Sentinel slots."""
    return [
        s
        for s in range(mm.NUM_SLOTS)
        if not state.is_empty(s) and state.obj_type[s] in mm.ENEMY_TYPES
    ]


_enemy_slots = enemy_slots  # internal alias kept for existing call sites


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
def _discharge_bank(state, enemy):
    """increase_enemy_energy_to_discharge $1A4F: bank one unit of drained energy on
    `enemy`, to be returned to the landscape as a tree by a later discharge pass."""
    a = mm.ENEMIES_ENERGY_TO_DISCHARGE + enemy
    state.mem[a] = (state.mem[a] + 1) & 0xFF


def _reduce_object_energy(state, target, enemy):
    """$1A08: drain `target`. The player loses one energy (returns True); a robot
    downgrades to a boulder, a boulder to a tree, a tree is removed (returns
    False -- no player energy change).  Every drain that is not a kill banks one
    unit of energy on `enemy` ($1A4F/$1A4E) for later discharge as a tree -- the
    landscape-energy conservation that scatters trees while the Sentinel drains."""
    mem = state.mem
    if target == mem[mm.PLAYER_OBJECT]:
        if state.energy == 0:
            # kill_player $1A00: drained with no energy left -> mark the player dead.
            mem[mm.PLAYER_DIED_BY_DRAINING] |= 0x80
            return True  # no discharge
        state.energy = state.energy - 1
        _discharge_bank(state, enemy)
        return True
    otype = state.obj_type[target]
    if otype == mm.T_ROBOT:
        mem[mm.ENEMIES_DRAINING_COOLDOWN + enemy] = 0  # $1A31
        state.obj_type[target] = mm.T_BOULDER
    elif otype == mm.T_TREE:
        actions.remove_object(state, target)
    else:  # boulder -> tree
        state.obj_type[target] = mm.T_TREE
    _discharge_bank(state, enemy)
    return False


def _exposure_byte(see):
    """The ROM's object_exposure ($14) from a can-see check: $80 fully visible (the
    base was reached), $40 partial (only the robot's head), 0 not visible."""
    if not (see["in_slot"] and see["in_fov"]):
        return 0
    return see["exposure"]


# ---------------------------------------------------------------------------
# target_object $1825
# ---------------------------------------------------------------------------
def _target_object(state, enemy, target, exposure):
    """$1825: record the target and, once the draining cooldown counts down to 1,
    drain it if fully visible ($1838), otherwise try to spawn a meanie against a
    partially-visible player ($184D consider_creating_meanie)."""
    mem = state.mem
    mem[mm.ENEMIES_TARGETED_OBJECT + enemy] = target
    mem[mm.ENEMIES_TARGETED_OBJECT_EXPOSURE + enemy] = exposure
    cd = mem[mm.ENEMIES_DRAINING_COOLDOWN + enemy]
    if cd < 0x01:  # first sight -> arm the drain timer
        mem[mm.ENEMIES_DRAINING_COOLDOWN + enemy] = DRAINING_COOLDOWN_RELOAD
        return
    if cd != 0x01:  # still counting down
        return
    if exposure & 0x80:  # fully visible -> drain
        mem[mm.TARGETED_OBJECT_SLOT] = target
        killed = target == mem[mm.PLAYER_OBJECT] and state.energy == 0
        _reduce_object_energy(state, target, enemy)
        if killed:  # kill_player $1A00 unwinds the stack -> no update-cooldown reload
            return
        mem[mm.ENEMIES_UPDATE_COOLDOWN + enemy] = UPDATE_COOLDOWN_DRAIN
        return
    # enemy_can't_drain_object $184D: the player is only partially visible.  Try to
    # turn a nearby tree into a meanie; if the whole scan fails, remember the
    # attempt and either keep trying or give up after MEANIE_MAX_ATTEMPTS.
    if _consider_creating_meanie(state, enemy):
        mem[mm.ENEMIES_UPDATE_COOLDOWN + enemy] = UPDATE_COOLDOWN_MEANIE_MADE
        return
    if mem[mm.ENEMIES_MEANIE_ATTEMPT_SCANS + enemy] >= MEANIE_MAX_ATTEMPTS:
        mem[mm.ENEMIES_DRAINING_COOLDOWN + enemy] = 0  # give up on this player
    else:
        mem[mm.ENEMIES_CONSIDERING_MEANIE + enemy] = 0x80  # keep trying next time


# ---------------------------------------------------------------------------
# rotate_enemy $1805
# ---------------------------------------------------------------------------
def _rotate_enemy(state, enemy):
    """$1805: add the per-enemy rotation step to its facing; reload the rotation
    cooldown to 200; re-arm the meanie hunt ($1818)."""
    mem = state.mem
    step = mem[mm.ROTATION_SPEED_TABLE + enemy]
    state.obj_h_angle[enemy] = (state.obj_h_angle[enemy] + step) & 0xFF
    mem[mm.ENEMIES_ROTATION_COOLDOWN + enemy] = ROTATION_COOLDOWN_RELOAD
    _initialise_enemy_meanie_variables(state, enemy)  # $1818


# ---------------------------------------------------------------------------
# consider_enemy_state $16E6 (no-meanie path)
# ---------------------------------------------------------------------------
def _consider_enemy_state(state, enemy):
    mem = state.mem
    if mem[mm.ENEMIES_UPDATE_COOLDOWN + enemy] >= COOLDOWN_STICK:
        return
    mem[mm.ENEMIES_UPDATE_COOLDOWN + enemy] = UPDATE_COOLDOWN_SCAN
    mem[mm.FOV_WIDTH] = FOV_SCAN

    # $16EA: an enemy that already owns a meanie runs the meanie lifecycle instead
    # of scanning for a drain (top bit of enemies_meanie_object clear == has one).
    if not (mem[mm.ENEMIES_MEANIE_OBJECT + enemy] & 0x80):
        _update_meanie(state, enemy)
        return

    # no_meanie ($1773): before draining, return any energy banked from earlier drains
    # to the landscape as a tree. If one is discharged the enemy dithers and skips its
    # drain/rotate for this update ($177A) -- the conservation that scatters trees.
    if _consider_discharging_enemy_energy(state, enemy):
        return

    # $177F: only while mid meanie-hunt (considering_meanie bit7 set) does the enemy
    # act on the flag -- first draining a boulder/stacked tree it can fully see, which
    # re-arms its tree search ($178B); otherwise it shifts the flag down ($1792). With
    # the top bit already clear the flag is left untouched ($1782 BPL -> $1795), so a
    # considering byte decays exactly once (0x80 -> 0x40) and then sticks.
    if mem[mm.ENEMIES_CONSIDERING_MEANIE + enemy] & 0x80:
        tb = _find_drainable_boulder_or_tree(state, enemy)
        if tb is not None:
            mem[mm.ENEMIES_MEANIE_SEARCH_OBJECT + enemy] = 0x40
            _reduce_object_energy(state, tb, enemy)  # never the player -> no kill
            mem[mm.ENEMIES_UPDATE_COOLDOWN + enemy] = UPDATE_COOLDOWN_DRAIN
            return
        mem[mm.ENEMIES_CONSIDERING_MEANIE + enemy] = (
            mem[mm.ENEMIES_CONSIDERING_MEANIE + enemy] >> 1
        )
    # Re-check a held target ($178C): keep it while it is visible AT ALL (a
    # partially-visible player stays targeted so its drain timer runs down to the
    # meanie-creation point $184D); drop it only when out of sight.
    if mem[mm.ENEMIES_DRAINING_COOLDOWN + enemy] != 0:
        held = mem[mm.ENEMIES_TARGETED_OBJECT + enemy]
        see = relative.can_see_object(state, enemy, held, mm.T_ROBOT, FOV_SCAN)
        exposure = _exposure_byte(see)
        if exposure != 0:
            _target_object(state, enemy, held, exposure)
            return
        mem[mm.ENEMIES_DRAINING_COOLDOWN + enemy] = 0  # target lost

    # find_drainable_robot_loop ($17B2): scan slots 63..0 for the player/a robot.
    player = mem[mm.PLAYER_OBJECT]
    partial_player = None
    for y in range(mm.NUM_SLOTS - 1, -1, -1):
        see = relative.can_see_object(state, enemy, y, mm.T_ROBOT, FOV_SCAN)
        # $17B7 LDA $0C76 ; AND #$40 ; BNE consider_next_object: a non-target tree in
        # the enemy's sightline to this robot's HEAD hides it -- skip before the
        # exposure test ($17BE), exactly as the ROM gates the drainable-robot scan.
        if see["tree_in_los_head"]:
            continue
        exposure = _exposure_byte(see)
        if exposure == 0:  # $17BE/$17C0: not visible at all -> next slot
            continue
        if exposure & 0x80:  # $17BA: fully visible (base reached) -> drain target
            _target_object(state, enemy, y, exposure)
            return
        if y == player:  # only the head is visible -> meanie candidate ($17C0)
            partial_player = y
    if partial_player is not None:
        # $17C4: unless the player was already found un-meanie-able this episode
        # (failed_meanie_memory), fresh-arm the meanie search and target them.
        if partial_player != mem[mm.ENEMIES_FAILED_MEANIE_MEMORY + enemy]:
            _initialise_enemy_meanie_variables(state, enemy)
            _target_object(state, enemy, partial_player, 0x40)
            return

    # reset_draining_cooldown_and_look_for_tree_or_boulder ($17E0).
    mem[mm.ENEMIES_DRAINING_COOLDOWN + enemy] = 0
    tb = _find_drainable_boulder_or_tree(state, enemy)
    if tb is not None:
        mem[mm.TARGETED_OBJECT_SLOT] = tb
        _reduce_object_energy(state, tb, enemy)
        mem[mm.ENEMIES_UPDATE_COOLDOWN + enemy] = UPDATE_COOLDOWN_DRAIN
        return

    # no_drain ($17F9): rotate if the rotation cooldown is low.
    if mem[mm.ENEMIES_ROTATION_COOLDOWN + enemy] < COOLDOWN_STICK:
        _rotate_enemy(state, enemy)


# ---------------------------------------------------------------------------
# meanie lifecycle: creation ($197D), rotation/hyperspace ($16F2), removal
# ---------------------------------------------------------------------------
def _initialise_enemy_meanie_variables(state, enemy):
    """$196A: (re)arm an enemy's meanie hunt -- no meanie owned, no failed memory,
    zero attempts, and a fresh 64-slot tree search."""
    mem = state.mem
    mem[mm.ENEMIES_MEANIE_OBJECT + enemy] = 0x80  # top bit == no meanie
    mem[mm.ENEMIES_FAILED_MEANIE_MEMORY + enemy] = 0x80
    mem[mm.ENEMIES_MEANIE_ATTEMPT_SCANS + enemy] = 0
    mem[mm.ENEMIES_MEANIE_SEARCH_OBJECT + enemy] = 0x40


def _consider_creating_meanie(state, enemy):
    """consider_creating_meanie $197D: scan the object slots (walking the
    per-enemy search counter down) for a tree within 10 tiles of the targeted
    player, in both axes, that the enemy can fully see within a two-screen-width
    FOV.  The first such tree is turned into a meanie owned by `enemy`.  Returns
    True if a meanie was created, False if the whole scan came up empty."""
    mem = state.mem
    player = mem[mm.ENEMIES_TARGETED_OBJECT + enemy]
    while True:
        sc = mem[mm.ENEMIES_MEANIE_SEARCH_OBJECT + enemy]
        if sc == 0:  # $198D: scanned everything -> no meanie this pass
            mem[mm.ENEMIES_MEANIE_ATTEMPT_SCANS + enemy] = (
                mem[mm.ENEMIES_MEANIE_ATTEMPT_SCANS + enemy] + 1
            ) & 0xFF
            mem[mm.ENEMIES_FAILED_MEANIE_MEMORY + enemy] = player
            return False
        mem[mm.ENEMIES_MEANIE_SEARCH_OBJECT + enemy] = sc - 1
        slot = sc - 1  # $199B DEY: the object index this iteration tests
        if state.obj_flags[slot] & 0x80:  # empty slot
            continue
        if state.obj_type[slot] != mm.T_TREE:  # not a tree
            continue
        dx = (state.obj_x[player] - state.obj_x[slot]) & 0xFF
        if dx >= 0x80:
            dx = 0x100 - dx  # $19B5 abs
        if dx >= 0x0A:  # more than 10 tiles away in x
            continue
        dy = (state.obj_y[player] - state.obj_y[slot]) & 0xFF
        if dy >= 0x80:
            dy = 0x100 - dy
        if dy >= 0x0A:  # more than 10 tiles away in y
            continue
        see = relative.can_see_object(state, enemy, slot, mm.T_TREE, FOV_CREATE_MEANIE)
        if not see["full"]:  # enemy hasn't a clear sight of the tree
            continue
        # $19E1: convert the tree into a meanie owned by this enemy.
        mem[mm.ENEMIES_MEANIE_OBJECT + enemy] = slot
        state.obj_type[slot] = mm.T_MEANIE
        return True


def _remove_meanie(state, enemy):
    """remove_meanie $1754: drop the meanie (turn it back into a tree) and mark the
    enemy as owning none."""
    mem = state.mem
    meanie = mem[mm.ENEMIES_MEANIE_OBJECT + enemy]
    mem[mm.ENEMIES_MEANIE_OBJECT + enemy] = 0x80
    state.obj_type[meanie] = mm.T_TREE


def _remove_meanie_and_reset_enemy(state, enemy):
    """remove_meanie_and_reset_enemy $174F: also clear the draining cooldown."""
    state.mem[mm.ENEMIES_DRAINING_COOLDOWN + enemy] = 0
    _remove_meanie(state, enemy)


def _update_meanie(state, enemy):
    """update_meanie $16F2: the enemy's meanie rotates toward the player and, once
    it is looking at a player it can still see, forcibly hyperspaces them."""
    mem = state.mem
    meanie = mem[mm.ENEMIES_MEANIE_OBJECT + enemy]
    target = mem[mm.ENEMIES_TARGETED_OBJECT + enemy]
    if state.obj_flags[target] & 0x80:  # $16F7: the object the player was in is gone
        _remove_meanie_and_reset_enemy(state, enemy)
        return
    see = relative.can_see_object(state, meanie, target, mm.T_ROBOT, FOV_SCAN)
    if not see["in_fov"]:  # $1706: meanie not yet looking at the player -> rotate
        c57 = relative.relative_angles(state, meanie, target)["c57"]
        step = MEANIE_ROTATE_STEP if not (c57 & 0x80) else (0x100 - MEANIE_ROTATE_STEP)
        state.obj_h_angle[meanie] = (state.obj_h_angle[meanie] + step) & 0xFF
        mem[mm.ENEMIES_UPDATE_COOLDOWN + enemy] = UPDATE_COOLDOWN_MEANIE_ROTATE
        return
    if target != mem[mm.PLAYER_OBJECT]:  # $1708: player transferred out of the object
        _remove_meanie_and_reset_enemy(state, enemy)
        return
    if _exposure_byte(see) == 0:  # $170E: meanie can't actually see the player
        _remove_meanie(state, enemy)
        return
    do_hyperspace(state)  # $1710: forced hyperspace


# ---------------------------------------------------------------------------
# do_hyperspace $2147 (create a robot on a random low tile + transfer/energy)
# ---------------------------------------------------------------------------
def _create_object(state, otype):
    """create_object $210E: the highest empty slot, typed `otype`, or None."""
    for slot in range(mm.NUM_SLOTS - 1, -1, -1):
        if state.obj_flags[slot] & 0x80:
            state.obj_type[slot] = otype
            return slot
    return None


def _random_tile_coord(prng):
    """get_random_tile_coordinate $125A: a prnd draw masked to 0..31, rejecting 31
    (the 32x32 board's out-of-range edge)."""
    while True:
        v = prng.next() & 0x1F
        if v != 0x1F:
            return v


def _put_object_in_tile(state, slot, tx, ty, prng):
    """put_object_in_tile $1EFF for a bare flat tile (the only kind the hyperspace
    placement picks): ground flags, z from the tile height, and a random facing."""
    b = tile_byte(state, tx, ty)
    state.obj_x[slot] = tx
    state.obj_y[slot] = ty
    state.obj_flags[slot] = 0x00
    state.obj_z_frac[slot] = 0xE0
    state.obj_z_height[slot] = (b >> 4) & 0xFF
    set_tile_byte(state, tx, ty, mm.OBJECT_TILE | slot)
    state.obj_v_angle[slot] = 0xF5
    rot = prng.next()
    state.obj_h_angle[slot] = ((rot & 0xF8) + 0x60) & 0xFF


def _put_object_in_random_tile_below_z(state, slot, z, prng):
    """put_object_in_random_tile_below_z $1224: place `slot` on a random flat,
    empty tile no higher than `z`.  After 256 misses the height ceiling `z` is
    raised; it fails (returns False) once that ceiling reaches 12."""
    attempts = 0
    while True:
        attempts = (attempts - 1) & 0xFF
        if attempts == 0:  # $122E: 256 misses -> relax the height ceiling
            z = (z + 1) & 0xFF
            if z >= 0x0C:
                return False
        tx = _random_tile_coord(prng)
        ty = _random_tile_coord(prng)
        b = tile_byte(state, tx, ty)
        if b >= mm.OBJECT_TILE:  # tile already holds an object
            continue
        if b & 0x0F:  # not a flat tile
            continue
        if (b >> 4) >= z:  # tile too high
            continue
        _put_object_in_tile(state, slot, tx, ty, prng)
        return True


def _consider_discharging_enemy_energy(state, enemy):
    """consider_discharging_enemy_energy $1A5D: if `enemy` has banked drained energy,
    return one unit to the landscape as a TREE on a random flat tile no higher than
    the below-enemies ceiling ($0C06), and decrement the bank.  Returns True if a tree
    was discharged (the ROM then dithers and SKIPS this enemy's drain/rotate for the
    update, $177A), False if there was nothing to discharge or no tile could take it."""
    mem = state.mem
    if mem[mm.ENEMIES_ENERGY_TO_DISCHARGE + enemy] == 0:
        return False  # $1A63: carry set -- nothing to discharge
    prng = Prng().load(mem)
    slot = _create_object(state, mm.T_TREE)  # $1A65 create_object(type 2)
    if slot is None:
        prng.store(mem)
        return False
    placed = _put_object_in_random_tile_below_z(
        state, slot, mem[mm.ENEMY_BELOW_Z], prng
    )
    prng.store(mem)
    if not placed:  # $1A70: no tile found -> abandon (slot stays flagged empty)
        return False
    a = mm.ENEMIES_ENERGY_TO_DISCHARGE + enemy
    mem[a] = (mem[a] - 1) & 0xFF  # $1A7A DEC
    return True


def do_hyperspace(state):
    """do_hyperspace $2147: create a synthoid on a random tile no higher than the
    player, spend the robot's energy, and transfer the player into it.  A player
    with too little energy dies; hyperspacing off the platform is the meanie's win
    condition against the player, hyperspacing *from* the platform completes the
    landscape ($2187)."""
    mem = state.mem
    prng = Prng().load(mem)
    slot = _create_object(state, mm.T_ROBOT)
    if slot is None:
        prng.store(mem)
        return
    player = mem[mm.PLAYER_OBJECT]
    z = (state.obj_z_height[player] + 1) & 0xFF
    placed = _put_object_in_random_tile_below_z(state, slot, z, prng)
    prng.store(mem)
    if not placed:  # $2159: no tile found -> the hyperspace is abandoned
        state.obj_flags[slot] |= 0x80
        return
    if state.energy < mm.ENERGY_IN_OBJECTS[mm.T_ROBOT]:  # $215F: out of energy -> death
        actions.remove_object(state, slot)
        mem[mm.PLAYER_HAS_HYPERSPACED] = 0x80
        return
    state.energy = state.energy - mm.ENERGY_IN_OBJECTS[mm.T_ROBOT]
    on_platform = (
        state.obj_x[player] == mem[mm.PLATFORM_X]
        and state.obj_y[player] == mem[mm.PLATFORM_Y]
    )
    if on_platform:  # $2187: hyperspacing from the platform completes the landscape
        mem[mm.LANDSCAPE_COMPLETE] = 0xC0
    mem[mm.PLAYER_OBJECT] = slot  # transfer the player into the new synthoid


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
    if state.obj_type[x] in mm.ENEMY_TYPES:  # $16BB type == sentry(1) or Sentinel(5)
        if not (state.obj_flags[x] & 0x80):  # $16CC BPL: not absorbed -> normal update
            _consider_enemy_state(state, x)
        else:
            # $16CE: a slot still typed as an enemy but flagged absorbed (SLOT_EMPTY)
            # still returns any residual banked energy to the landscape as a tree.
            _consider_discharging_enemy_energy(state, x)
    prng = Prng().load(mem)  # $16D6 JSR prnd (advances the stream)
    prng.next()
    prng.store(mem)
    _dec_cursor(state)


def step(state):
    """Advance the world by one game round: cooldown tick + one enemy update.

    This is the ISOLATED-ROUTINE round (tick_cooldowns + update_enemies, 1:1) that the
    py65 oracle (tests/oracle.step_enemy_round) captures the golden with -- a valid
    transition-function unit, but NOT the running game's cadence.  Real-time advance goes
    through :func:`advance_frames` (the two decoupled ROM clocks)."""
    tick_cooldowns(state)
    update_enemies(state)


# ---------------------------------------------------------------------------
# real-game frame cadence: update_game_loop $1289 + raster IRQ $9663 / scroll $3684
# ---------------------------------------------------------------------------
CURSOR_SLOTS = 8  # $0090 cycles 7->0: one cursor sweep considers every enemy slot once


def _any_enemy_due(state):
    """True if some enemy's update_cooldown has ticked below the stick threshold, so a
    not-plotting update_enemies sweep would actually process it.  While every enemy is
    still cooling ($16E9 skips them), a sweep only churns the cursor/PRNG (out of the lock
    state), so it can be skipped -- the ~15x speedup that keeps the frame forecast fast.
    """
    mem = state.mem
    for e in range(CURSOR_SLOTS):
        if state.obj_flags[e] & 0x80:
            continue
        if state.obj_type[e] in mm.ENEMY_TYPES:
            if mem[mm.ENEMIES_UPDATE_COOLDOWN + e] < COOLDOWN_STICK:
                return True
    return False


def cooldown_frame(state):
    """The cooldown clock for ONE video frame -- $130C called once per frame (raster
    $9663, or the scroll loop $3684 while scrolling; the two are mutually exclusive via
    $0CD8, so exactly one $130C per frame).  Advance the integer Bresenham accumulator
    $1335 += $CD and, only on carry (205/256), run update_enemy_cooldowns ($1317, itself
    1-in-3 via the $0C50 gate).  Suppressed until the player's first action ($0CE5,
    $9659/$367f).  All integer -- no rounding to drift."""
    mem = state.mem
    if mem[mm.PLAYER_NOT_ACTED] & 0x80:  # player has not yet acted -> no cooldown ticks
        return
    acc = mem[mm.COOLDOWN_BRESENHAM] + mm.COOLDOWN_BRESENHAM_STEP
    mem[mm.COOLDOWN_BRESENHAM] = acc & 0xFF
    if acc > 0xFF:  # $1315 BCC skip -> only the carry runs the cooldown decrement
        tick_cooldowns(state)


def advance_frame(state, plotting=False):
    """Advance the world by ONE video frame.

    The cooldown clock ticks every frame (:func:`cooldown_frame`).  update_enemies
    executes the cooldown-gated decisions only in a NOT-plotting frame (the loop's
    is_plotting path $128c suppresses it while the world scrolls).  A full 8-slot cursor
    sweep lets every enemy that became due at this frame's cooldown tick act exactly once
    -- further sweeps are idempotent (its update_cooldown was reloaded).

    Validated against the live ROM (scripts/lockstep_probe.py): with the exact live frame
    count, the sentinel facing, energy, objects, tiles and every cooldown reproduce the
    running game byte-for-byte.  ``plotting`` gates the update_enemies spin per the ROM's
    is_plotting path; the caller supplies the plot schedule (the live driver reads $0CE4,
    the offline planner runs the cooldown clock which is plot-independent)."""
    cooldown_frame(state)
    if not plotting and _any_enemy_due(state):
        for _ in range(CURSOR_SLOTS):
            update_enemies(state)


def advance_frames(state, n_frames, plotting=False):
    """Advance ``n_frames`` video frames.  ``plotting`` marks a scroll/replot span in which
    update_enemies is suppressed (only the cooldown clock advances)."""
    for _ in range(int(n_frames)):
        advance_frame(state, plotting=plotting)


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
