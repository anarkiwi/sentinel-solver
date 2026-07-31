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
:func:`meanie_threat` is only a query over the same visibility test.

The cooldown, rotation, targeting, drain and meanie machinery reproduce the ROM
round for round, validated bit-exact against the py65 oracle over the full state
(object table, enemy + meanie arrays, player/energy, PRNG, tiles, death/hyperspace
flags) across the whole meanie lifecycle -- spawn, hunt, forced hyperspace and a
later drain-death -- as well as the failed-attempt path.
"""

from sentinel import (
    badline,
    memmap as mm,
    relative,
    actions,
    terrain,
    passcost,
    projector,
)
from sentinel.prng import Prng
from sentinel.terrain import tile_byte, set_tile_byte

try:
    from sentinel import enemies_jit

    _HAVE_JIT = True
except Exception:  # pragma: no cover - numba absent -> pure-Python fallback
    enemies_jit = None
    _HAVE_JIT = False

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

PHASE_HEAD = 0  # sub-pass resume at $1289: head and $16B5 dispatch unspent
PHASE_BODY = 1  # sub-pass resume at $16E6: consider_enemy_state has not run
PHASE_CURSOR = 2  # sub-pass resume at $16D9: the prnd ran, its result is not stored

BODY_ENTRY = 0  # $16E6: the gate, then the $16ED/$16F0 entry writes
BODY_MEANIE = 1  # $16F2 update_meanie: its one $1887 call
BODY_DISCHARGE = 2  # $1773: consider_discharging_enemy_energy $1A5D
BODY_HUNT = 3  # $177F: find_drainable_boulder_or_tree $1AB0, mid meanie-hunt
BODY_HELD = 4  # $178C: the held target's $1887 re-check
BODY_SCAN = 5  # $17B2 find_drainable_robot_loop: slots 63..0
BODY_PARTIAL = 6  # $17C4: the partially-visible player branch
BODY_TREE = 7  # $17E0: find_drainable_boulder_or_tree $1AB0
BODY_ROTATE = 8  # $17F9: the rotation gate
BODY_MAKE_MEANIE = 9  # $197D consider_creating_meanie, reached from $184D
BODY_DONE = -1  # the body reached its RTS; the pass may go on to $16D6


# Where a raster interrupt can catch the play loop, as the resume points above name it.
LOOP_HEAD = (0x1289, 0x12A2)  # the pass head, before its JSR $16B5
LOOP_TAIL = (0x12A2, 0x12C8)  # the tail; $191F and the $34xx sound run under it
LOOP_DISPATCH = (0x16B5, 0x16D6)  # $16B5's type dispatch, before the body
LOOP_PRND = (0x16D9, 0x16E6)  # the cursor step; $31CA itself runs under $16D6
LOOP_BODY = (0x16E6, 0x1887)  # consider_enemy_state, target_object and their tail
_BODY_LADDER = (  # the first address of each stage, in ROM order
    (0x16E6, BODY_ENTRY),
    (0x16FF, BODY_MEANIE),
    (0x1773, BODY_DISCHARGE),
    (0x177D, BODY_HUNT),
    (0x1795, BODY_HELD),
    (0x17AC, BODY_SCAN),
    (0x17CD, BODY_PARTIAL),
    (0x17E0, BODY_TREE),
    (0x17F9, BODY_ROTATE),
    (0x1825, BODY_DONE),
    (0x1852, BODY_MAKE_MEANIE),
    (0x1876, BODY_DONE),
)
SAVED_X = 0x191E  # $1889 STX $191E: $1887 saves its caller's X here
SAVED_Y = mm.TARGETED_OBJECT_SLOT  # $188C STY $0C58: ... and its caller's Y here
PARTIAL_SLOT = 0x000F  # $17AE/$17C8: the head-only player the robot scan remembers

_HEAD_SPENT = {  # cycles spent at each boundary of $1289..$16CC, the head and dispatch
    0x1289: 0,
    0x128C: 4,
    0x128E: 6,
    0x1291: 12,
    0x1294: 16,
    0x129F: 19,
    0x16B5: 25,
    0x16B6: 27,
    0x16B9: 31,
    0x16BB: 34,
    0x16BE: 38,
    0x16C0: 40,
    0x16C2: 42,
    0x16C4: 44,
}
_DISPATCH_SPENT = {  # $16C6..$16CC: the two enemy types take different compare chains
    mm.T_SENTRY: {0x16C6: 43, 0x16C9: 47, 0x16CC: 51},
    mm.T_SENTINEL: {0x16C6: 46, 0x16C9: 50, 0x16CC: 54},
}
_CURSOR_SPENT = {  # from $16D9; the wrap misses the $16DB BPL and costs four more
    False: {0x16D9: 0, 0x16DB: 5, 0x16E1: 8, 0x16E3: 11, 0x16E5: 14},
    True: {
        0x16D9: 0,
        0x16DB: 5,
        0x16DD: 7,
        0x16DF: 9,
        0x16E1: 12,
        0x16E3: 15,
        0x16E5: 18,
    },
}
_TAIL_SPENT = {  # from $12A2; the $191F walk and the sound bodies are added below
    0x12A2: 0,
    0x12A5: 4,
    0x12B1: 7,
    0x12B4: 13,
    0x12B6: 15,
    0x12B9: 19,
    0x12BB: 21,
    0x12BE: 27,
    0x12C1: 33,
    0x12C4: 39,
    0x12C7: 45,
}
_TAIL_SOUND = {0x12C1: 13, 0x12C4: 42, 0x12C7: 69}  # $34BA, then $352C, then $347D
_PRND_ROUND = {  # one $31CF lap of the 40-bit LFSR, 51 cycles
    0x31CF: 0,
    0x31D2: 4,
    0x31D3: 6,
    0x31D4: 8,
    0x31D5: 10,
    0x31D8: 14,
    0x31D9: 16,
    0x31DC: 22,
    0x31DF: 28,
    0x31E2: 34,
    0x31E5: 40,
    0x31E8: 46,
    0x31E9: 48,
}
_PRND_ENTRY = {0x31CA: 0, 0x31CD: 4}  # $31CA STY $31F2 4 + $31CD LDY #$08 2
_PRND_TAIL = {
    0x31EB: 413,
    0x31EE: 417,
}  # past the loop: 6 + 7*51 + the shorter last lap
_PRND_ROUNDS = 8  # $31CD loads the lap counter with eight
_PRND_LAP = 51  # one lap; the last is a cycle shorter, its $31E9 BNE not taken


def _prnd_spent(pc, y):
    """Cycles $31CA has already spent when the interrupt caught it at ``pc``."""
    if pc in _PRND_TAIL:
        return _PRND_TAIL[pc]
    if pc in _PRND_ROUND:
        laps = _PRND_ROUNDS - (y + 1 if pc == 0x31E9 else y)
        return 6 + laps * _PRND_LAP + _PRND_ROUND[pc]
    return _PRND_ENTRY.get(pc)


def _segment_offset(mem, pc, y):
    """The signed cycles between the machine's position and the model's resume point.

    Positive: the machine is past it, so the model is credited what it already paid;
    negative: the machine still owes cycles the model's resume point will not charge.
    None where the position is not on a straight line this can count."""
    if pc in _HEAD_SPENT:
        if pc >= 0x129F and mem[0x0CDC] & 0x80:  # $1296 took the longer $1223 path
            return None
        return _HEAD_SPENT[pc]
    dispatch = _DISPATCH_SPENT.get(mem[mm.OBJECTS_TYPE + mem[mm.CURSOR]], {})
    if pc in dispatch:
        return dispatch[pc]
    wrap = pc in (0x16DD, 0x16DF) or mem[mm.CURSOR] == 7
    if pc in _CURSOR_SPENT[wrap]:
        return _CURSOR_SPENT[wrap][pc]
    if pc in _TAIL_SPENT:  # the model has paid the whole tail; the machine has not
        exposure = passcost.exposure_cycles(mem)
        spent = (
            _TAIL_SPENT[pc] + _TAIL_SOUND.get(pc, 0) + (exposure if pc > 0x12BB else 0)
        )
        total = _TAIL_SPENT[0x12C7] + 3 + _TAIL_SOUND[0x12C7] + exposure
        return spent - total
    spent = _prnd_spent(pc, y)  # the prnd the model's PHASE_CURSOR does not charge
    return None if spent is None else spent - passcost.PRND


def _innermost_loop_address(pc, sp, page):
    """The deepest play-loop address the interrupt caught: the PC itself when it is
    already in the loop's own code, else the return address of the call that left it."""
    if LOOP_HEAD[0] <= pc < LOOP_BODY[1]:
        return pc, True
    off = (sp + 7) & 0xFF
    while off < 0xFF:
        ret = (page[off] | (page[(off + 1) & 0xFF] << 8)) + 1
        if LOOP_HEAD[0] <= ret < LOOP_BODY[1]:
            return ret, False
        off += 1
    return pc, False


def resume_from_stack(mem, sp, page):
    """The sub-pass position a $9630 halt exposes: (phase, stage, index, partial, cycles).

    The $95E9 frame holds Y, X, A, P, PCL, PCH from SP+1, and under it the foreground's
    own return addresses; $1887 saves its caller's X and Y at $191E and $0C58, so the
    scan indices survive a call as well.  The fifth value is the cycle offset between
    the machine's position and that resume point, where a straight line can count it."""
    y = page[(sp + 1) & 0xFF]
    x = page[(sp + 2) & 0xFF]
    pc = page[(sp + 5) & 0xFF] | (page[(sp + 6) & 0xFF] << 8)
    addr, live = _innermost_loop_address(pc, sp, page)
    offset = _segment_offset(mem, pc, y)  # None where no straight line counts it
    if not LOOP_BODY[0] <= addr < LOOP_BODY[1]:
        if LOOP_DISPATCH[0] <= addr < LOOP_DISPATCH[1]:
            return PHASE_HEAD, BODY_ENTRY, 0, -1, offset
        if LOOP_PRND[0] <= addr < LOOP_PRND[1]:
            return PHASE_CURSOR, BODY_ENTRY, 0, -1, offset
        if 0x16D6 <= addr < LOOP_PRND[0]:  # the JSR $31CA: the prnd is still owed
            return PHASE_BODY, BODY_DONE, 0, -1, offset
        return PHASE_HEAD, BODY_ENTRY, 0, -1, offset  # head, tail, or off the loop
    stage = BODY_ENTRY
    for first, st in _BODY_LADDER:
        if addr >= first:
            stage = st
    index, partial = 0, -1
    if stage == BODY_SCAN:
        index = (y if live else mem[SAVED_Y]) if addr >= 0x17B2 else mm.NUM_SLOTS - 1
    elif stage in (BODY_HUNT, BODY_TREE):
        first_call = 0x1784 if stage == BODY_HUNT else 0x17E5
        index = (x if live else mem[SAVED_X]) if addr > first_call else mm.NUM_SLOTS - 1
    if stage in (BODY_SCAN, BODY_PARTIAL, BODY_TREE, BODY_ROTATE):
        partial = -1 if mem[PARTIAL_SLOT] & 0x80 else mem[PARTIAL_SLOT]
    return PHASE_BODY, stage, index, partial, None


def stack_position(mem, sp, page):
    """:func:`resume_from_stack` plus the two addresses that say how exact it is.

    ``(resume, pc, addr)``: the PC the raster IRQ interrupted and the play-loop address
    it maps to.  A resume whose cycle offset is ``None`` starts the model at ``addr``
    while the machine is somewhere at or after it, and ``pc`` names where."""
    pc = page[(sp + 5) & 0xFF] | (page[(sp + 6) & 0xFF] << 8)
    addr, _live = _innermost_loop_address(pc, sp, page)
    return resume_from_stack(mem, sp, page), pc, addr


def paid(slot):
    """``State.body_index`` for "slot's $1887 is charged, its CORE write is not".

    The field is a scan position: >= 0 the next slot to query, -1 exhausted,
    <= -2 a charged-but-uncommitted slot."""
    return -2 - slot


def enemy_slots(state):
    """The occupied sentry/Sentinel slots."""
    return [
        s
        for s in range(mm.NUM_SLOTS)
        if not state.is_empty(s) and state.obj_type[s] in mm.ENEMY_TYPES
    ]


# ---------------------------------------------------------------------------
# update_enemy_cooldowns $1317
# ---------------------------------------------------------------------------
def tick_cooldowns(state, clk):
    """$1317: on 2-of-3 rounds decrement the $0C50 gate; on the third, decrement
    every draining/rotation/update cooldown that is >= 2 (they stick at 1).

    Returns its cycles: the 24-byte walk is 6 dearer per byte that actually decrements,
    so the raster IRQ's own length is a property of the board's cooldown state."""
    mem = state.mem
    if mem[mm.COOLDOWN_GATE] != 0:
        mem[mm.COOLDOWN_GATE] = (mem[mm.COOLDOWN_GATE] - 1) & 0xFF
        return badline.charge(clk, 0x1317, passcost.COOLDOWN_TICK_GATE)
    cost = badline.charge(
        clk, 0x131C, passcost.COOLDOWN_TICK_WALK - passcost.COOLDOWN_TICK_LAST
    )
    for addr in range(mm.ENEMIES_DRAINING_COOLDOWN, mm.ENEMIES_UPDATE_COOLDOWN + 8):
        if mem[addr] >= COOLDOWN_STICK:
            mem[addr] -= 1
            cost += badline.charge(clk, 0x1325, passcost.COOLDOWN_TICK_BYTE_DEC)
        else:
            cost += badline.charge(clk, 0x131E, passcost.COOLDOWN_TICK_BYTE_STICK)
    mem[mm.COOLDOWN_GATE] = 2
    return cost


# ---------------------------------------------------------------------------
# reduce_object_energy $1A08
# ---------------------------------------------------------------------------
def _discharge_bank(state, enemy):
    """increase_enemy_energy_to_discharge $1A4F: bank one unit of drained energy on
    `enemy`, to be returned to the landscape as a tree by a later discharge pass."""
    a = mm.ENEMIES_ENERGY_TO_DISCHARGE + enemy
    state.mem[a] = (state.mem[a] + 1) & 0xFF


def _reduce_object_energy(state, target, enemy, clk):
    """$1A08: drain `target`, returning (drained the player, cycles spent).

    The player loses one energy; a robot downgrades to a boulder, a boulder to a tree,
    a tree is removed.  Every drain that is not a kill banks one unit of energy on
    `enemy` ($1A4F/$1A4E) for later discharge as a tree."""
    mem = state.mem
    head = badline.charge(clk, 0x1A08, passcost.REDUCE_HEAD)
    if target == mem[mm.PLAYER_OBJECT]:
        if state.energy == 0:
            # kill_player $1A00: drained with no energy left -> mark the player dead.
            mem[mm.PLAYER_DIED_BY_DRAINING] |= 0x80
            return True, head + badline.charge(clk, 0x1A0D, passcost.REDUCE_KILL)
        state.energy = state.energy - 1
        _discharge_bank(state, enemy)
        cost = head + badline.charge(clk, 0x1A14, passcost.REDUCE_PLAYER)
        cost += badline.charge(clk, 0x1A4F, passcost.REDUCE_BANK)
        # $1A1A/$1A1F: the drained bar is replotted and the drain sound started
        cost += passcost.status_bar_cycles(state.energy, clk)
        cost += badline.charge(clk, 0x1A1D, passcost.TUNE_DRAIN)
        start_tune(state, mm.SOUND_DRAIN)
        return True, cost
    cost = head + badline.charge(clk, 0x1A0D, passcost.REDUCE_OBJECT)
    cost += badline.charge(clk, 0x1A4F, passcost.REDUCE_BANK)
    otype = state.obj_type[target]
    if otype == mm.T_ROBOT:
        mem[mm.ENEMIES_DRAINING_COOLDOWN + enemy] = 0  # $1A31
        state.obj_type[target] = mm.T_BOULDER
        cost += badline.charge(clk, 0x1A2D, passcost.REDUCE_ROBOT)
    elif otype == mm.T_TREE:
        stacked = state.obj_flags[target] >= 0x40  # $1EFF: the tile byte it restores
        actions.remove_object(state, target)
        cost += badline.charge(clk, 0x1EEF, passcost.REDUCE_TREE)
        cost += badline.charge(
            clk,
            0x1EEF,
            passcost.REMOVE_STACKED if stacked else passcost.REMOVE_GROUND,
        )
    else:  # boulder -> tree
        state.obj_type[target] = mm.T_TREE
        cost += badline.charge(clk, 0x1A44, passcost.REDUCE_BOULDER)
    _discharge_bank(state, enemy)
    return False, cost


def _see_cost(see, clk):
    """The $1887 cycles a :func:`relative.can_see_object` query cost the ROM."""
    return badline.charge(clk, 0x1887, see["cycles"])


def _redraw_cost(state, target, clk):
    """$187B STX $91 + $1881 JSR $1F9F: what redrawing `target` costs right now.

    Off screen ($209B carry set) that is the counted $1F9F alone; on screen it is
    that plus the $1FFC JSR $2625 replot of the strip, at the camera $1FC2 shifts to."""
    cycles, columns, left = relative.update_object_on_screen_cycles(state, target)
    cycles = badline.charge(clk, 0x1F9F, cycles)
    if columns == 0:
        return cycles
    frames = projector.strip_replot_frames(state, target, left, columns)
    # whole video frames of stall: PAL_FRAME_CYCLES already carries their own badlines
    return cycles + int(round(frames * passcost.PAL_FRAME_CYCLES))


def _body_tail(state, target, clk):
    """$1876..$1884: every exit that redraws the object it just touched ($187B STX
    $91 names it), and the $1F9F redraw that object's own screen span prices."""
    return badline.charge(clk, 0x1876, passcost.BODY_TAIL) + _redraw_cost(
        state, target, clk
    )


def _exposure_byte(see):
    """The ROM's object_exposure ($14) from a can-see check: $80 fully visible (the
    base was reached), $40 partial (only the robot's head), 0 not visible."""
    if not (see["in_slot"] and see["in_fov"]):
        return 0
    return see["exposure"]


# ---------------------------------------------------------------------------
# target_object $1825
# ---------------------------------------------------------------------------
def _target_object(state, enemy, target, exposure, clk):
    """$1825 up to its $184D branch: record the target and, once the draining
    cooldown counts down to 1, drain it if fully visible ($1838).  Returns the next
    stage -- BODY_MAKE_MEANIE when only the player's head is visible, else DONE --
    and the cycles $1825's own line spent reaching it."""
    mem = state.mem
    mem[mm.ENEMIES_TARGETED_OBJECT + enemy] = target
    mem[mm.ENEMIES_TARGETED_OBJECT_EXPOSURE + enemy] = exposure
    cost = badline.charge(clk, 0x1825, passcost.TARGET_HEAD)
    cd = mem[mm.ENEMIES_DRAINING_COOLDOWN + enemy]
    if cd < 0x01:  # first sight -> arm the drain timer
        mem[mm.ENEMIES_DRAINING_COOLDOWN + enemy] = DRAINING_COOLDOWN_RELOAD
        return BODY_DONE, cost + badline.charge(clk, 0x1833, passcost.TARGET_FIRST)
    if cd != 0x01:  # still counting down
        return BODY_DONE, cost + badline.charge(clk, 0x1833, passcost.TARGET_WAIT)
    cost += badline.charge(clk, 0x183D, passcost.TARGET_DUE)
    if exposure & 0x80:  # fully visible -> drain
        mem[mm.TARGETED_OBJECT_SLOT] = target
        killed = target == mem[mm.PLAYER_OBJECT] and state.energy == 0
        cost += badline.charge(clk, 0x1843, passcost.TARGET_DRAIN)
        cost += _reduce_object_energy(state, target, enemy, clk)[1]
        if killed:  # kill_player $1A00 unwinds the stack -> no update-cooldown reload
            return BODY_DONE, cost
        mem[mm.ENEMIES_UPDATE_COOLDOWN + enemy] = UPDATE_COOLDOWN_DRAIN
        if target == mem[mm.PLAYER_OBJECT]:  # $184D: a drained player skips the redraw
            return BODY_DONE, cost + badline.charge(
                clk, 0x1884, passcost.TARGET_DRAIN_PLAYER
            )
        cost += badline.charge(clk, 0x184D, passcost.TARGET_DRAIN_OBJ)
        return BODY_DONE, cost + _body_tail(state, target, clk)
    # $184D: only the head -> hunt a tree to convert
    return BODY_MAKE_MEANIE, cost + badline.charge(clk, 0x1841, passcost.TARGET_MEANIE)


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
    start_tune(state, mm.SOUND_ROTATE)  # $180F JSR $3470
    _initialise_enemy_meanie_variables(state, enemy)  # $1818


# ---------------------------------------------------------------------------
# consider_enemy_state $16E6 (no-meanie path)
# ---------------------------------------------------------------------------
def _consider_enemy_state(state, enemy, budget, stage, index, partial, clk):
    """$16E6, resumable at its own write points.

    Spends ``budget`` and suspends where the ROM would be when the frame ran out --
    between an $1887 query and the CORE write its answer causes.  Returns
    (budget, stage, index, partial); stage BODY_DONE means the routine reached RTS."""
    mem = state.mem
    while True:
        if stage == BODY_DONE:
            return budget, BODY_DONE, 0, -1
        if stage == BODY_ENTRY:
            if mem[mm.ENEMIES_UPDATE_COOLDOWN + enemy] >= COOLDOWN_STICK:
                gate = badline.charge(clk, 0x16E6, passcost.UPDATE_GATE_CLOSED)
                return budget - gate, BODY_DONE, 0, -1
            budget -= badline.charge(clk, 0x16E6, passcost.CONSIDER_ENTRY)
            mem[mm.ENEMIES_UPDATE_COOLDOWN + enemy] = UPDATE_COOLDOWN_SCAN  # $16ED
            mem[mm.FOV_WIDTH] = FOV_SCAN  # $16F0
            # $16EA: an enemy that owns a meanie (top bit clear) runs its lifecycle
            if not (mem[mm.ENEMIES_MEANIE_OBJECT + enemy] & 0x80):
                budget -= badline.charge(clk, 0x16F7, passcost.CONSIDER_MEANIE)
                stage, index = BODY_MEANIE, 0
                continue
            budget -= badline.charge(clk, 0x16FC, passcost.CONSIDER_NO_MEANIE)
            budget -= badline.charge(clk, 0x1773, passcost.DISCHARGE_CALL)
            stage, index = BODY_DISCHARGE, 0
            continue

        if index > -1 and budget < 1:  # nothing charged is awaiting its write
            return budget, stage, index, partial

        if stage == BODY_MEANIE:
            budget, stage, index = _update_meanie(state, enemy, budget, index, clk)
            continue

        if stage == BODY_DISCHARGE:
            # no_meanie ($1773): a discharge makes the enemy skip its drain ($177A)
            discharged, dcost, slot = _consider_discharging_enemy_energy(
                state, enemy, clk
            )
            budget -= dcost
            if discharged:  # $177A: the tail redraws, then the update is over
                budget -= badline.charge(clk, 0x1778, passcost.DISCHARGED)
                budget -= _body_tail(state, slot, clk)
                return budget, BODY_DONE, 0, -1
            budget -= badline.charge(clk, 0x177D, passcost.NO_DISCHARGE)
            # $177F: only mid meanie-hunt does the enemy act on the considering flag
            if mem[mm.ENEMIES_CONSIDERING_MEANIE + enemy] & 0x80:
                budget -= badline.charge(clk, 0x1784, passcost.HUNT_CALL)
                budget -= badline.charge(clk, 0x1AB0, passcost.TILE_SCAN_ENTRY)
                stage, index = BODY_HUNT, mm.NUM_SLOTS - 1
                continue
            budget -= badline.charge(clk, 0x1782, passcost.HUNT_CLEAR)
            stage, index = BODY_HELD, 0
            continue

        if stage == BODY_HUNT:
            budget, index, tb = _find_drainable_boulder_or_tree(
                state, enemy, budget, index, clk
            )
            if tb == -2:  # suspended mid-scan
                return budget, stage, index, partial
            if tb >= 0:
                mem[mm.ENEMIES_MEANIE_SEARCH_OBJECT + enemy] = 0x40  # $178B
                budget -= badline.charge(clk, 0x178B, passcost.HUNT_HIT)
                budget -= badline.charge(clk, 0x17EA, passcost.DRAIN_CALL)
                budget -= _reduce_object_energy(state, tb, enemy, clk)[1]  # $17EA
                mem[mm.ENEMIES_UPDATE_COOLDOWN + enemy] = UPDATE_COOLDOWN_DRAIN
                budget -= badline.charge(clk, 0x17ED, passcost.DRAIN_TAIL)
                return budget - _body_tail(state, tb, clk), BODY_DONE, 0, -1
            budget -= badline.charge(clk, 0x1787, passcost.HUNT_MISS)
            mem[mm.ENEMIES_CONSIDERING_MEANIE + enemy] = (  # $1792
                mem[mm.ENEMIES_CONSIDERING_MEANIE + enemy] >> 1
            )
            stage, index = BODY_HELD, 0
            continue

        if stage == BODY_HELD:
            # $178C: a held target is kept while visible AT ALL, dropped when not
            if mem[mm.ENEMIES_DRAINING_COOLDOWN + enemy] == 0:
                budget -= badline.charge(clk, 0x1795, passcost.HELD_NONE)
                budget -= badline.charge(clk, 0x17AC, passcost.SCAN_INIT)
                stage, index, partial = BODY_SCAN, mm.NUM_SLOTS - 1, -1
                continue
            held = mem[mm.ENEMIES_TARGETED_OBJECT + enemy]
            see = relative.can_see_object(state, enemy, held, mm.T_ROBOT, FOV_SCAN)
            if index >= 0:
                budget -= badline.charge(clk, 0x179A, passcost.HELD_CALL)
                budget -= _see_cost(see, clk)
                if budget <= 0:
                    return budget, stage, paid(0), partial
            exposure = _exposure_byte(see)
            if exposure != 0:
                budget -= badline.charge(clk, 0x17A6, passcost.HELD_KEPT)
                stage, cost = _target_object(state, enemy, held, exposure, clk)
                budget, index = budget - cost, 0
                continue
            mem[mm.ENEMIES_DRAINING_COOLDOWN + enemy] = 0  # target lost
            budget -= badline.charge(clk, 0x17A2, passcost.HELD_LOST)
            budget -= badline.charge(clk, 0x17AC, passcost.SCAN_INIT)
            stage, index, partial = BODY_SCAN, mm.NUM_SLOTS - 1, -1
            continue

        if stage == BODY_SCAN:
            budget, stage, index, partial = _scan_for_robot(
                state, enemy, budget, index, partial, clk
            )
            continue

        if stage == BODY_PARTIAL:
            budget -= badline.charge(clk, 0x17D1, passcost.SCAN_END_PARTIAL)
            # $17C4: fresh-arm the meanie hunt unless this player already failed one
            if partial != mem[mm.ENEMIES_FAILED_MEANIE_MEMORY + enemy]:
                _initialise_enemy_meanie_variables(state, enemy)
                budget -= badline.charge(clk, 0x17D7, passcost.PARTIAL_ARM)
                stage, cost = _target_object(state, enemy, partial, 0x40, clk)
                budget, index, partial = budget - cost, 0, -1
                continue
            mem[mm.ENEMIES_DRAINING_COOLDOWN + enemy] = 0  # $17E0
            budget -= badline.charge(clk, 0x17D5, passcost.PARTIAL_KNOWN)
            budget -= badline.charge(clk, 0x17E0, passcost.TREE_CALL)
            budget -= badline.charge(clk, 0x1AB0, passcost.TILE_SCAN_ENTRY)
            stage, index, partial = BODY_TREE, mm.NUM_SLOTS - 1, -1
            continue

        if stage == BODY_TREE:
            budget, index, tb = _find_drainable_boulder_or_tree(
                state, enemy, budget, index, clk
            )
            if tb == -2:
                return budget, stage, index, partial
            if tb >= 0:
                mem[mm.TARGETED_OBJECT_SLOT] = tb
                budget -= badline.charge(clk, 0x17E8, passcost.TREE_HIT)
                budget -= badline.charge(clk, 0x17EA, passcost.DRAIN_CALL)
                budget -= _reduce_object_energy(state, tb, enemy, clk)[1]
                mem[mm.ENEMIES_UPDATE_COOLDOWN + enemy] = UPDATE_COOLDOWN_DRAIN
                budget -= badline.charge(clk, 0x17ED, passcost.DRAIN_TAIL)
                return budget - _body_tail(state, tb, clk), BODY_DONE, 0, -1
            budget -= badline.charge(clk, 0x17E8, passcost.TREE_NONE)
            stage, index = BODY_ROTATE, 0
            continue

        if stage == BODY_ROTATE:
            # no_drain ($17F9): rotate if the cooldown is low; the turn redraws it
            if mem[mm.ENEMIES_ROTATION_COOLDOWN + enemy] < COOLDOWN_STICK:
                _rotate_enemy(state, enemy)
                budget -= badline.charge(clk, 0x17F9, passcost.ROTATE_GATE)
                budget -= badline.charge(clk, 0x1805, passcost.ROTATE)
                budget -= _redraw_cost(state, enemy, clk)
                return budget, BODY_DONE, 0, -1
            held = badline.charge(clk, 0x1802, passcost.ROTATE_GATE_HELD)
            return budget - held, BODY_DONE, 0, -1

        budget, stage, index = _consider_creating_meanie(
            state, enemy, budget, index, clk
        )


def _scan_for_robot(state, enemy, budget, index, partial, clk):
    """find_drainable_robot_loop ($17B2), resumable, one slot per unit.

    Scans slots 63..0 for the player or a robot; a fully-visible one is targeted,
    a head-only player is remembered in ``partial``."""
    mem = state.mem
    player = mem[mm.PLAYER_OBJECT]
    while True:
        if index <= -2:  # this slot's $1887 is paid; only its write is outstanding
            y = -2 - index
            see = relative.can_see_object(state, enemy, y, mm.T_ROBOT, FOV_SCAN)
            stage, cost = _target_object(state, enemy, y, _exposure_byte(see), clk)
            return budget - cost, stage, 0, -1
        if index < 0:  # $17CB: the scan is exhausted
            if partial >= 0:
                return budget, BODY_PARTIAL, 0, partial
            mem[mm.ENEMIES_DRAINING_COOLDOWN + enemy] = 0  # $17E0
            budget -= badline.charge(clk, 0x17CD, passcost.SCAN_END)
            budget -= badline.charge(clk, 0x17E0, passcost.TREE_CALL)
            budget -= badline.charge(clk, 0x1AB0, passcost.TILE_SCAN_ENTRY)
            return budget, BODY_TREE, mm.NUM_SLOTS - 1, -1
        if budget <= 0:
            return budget, BODY_SCAN, index, partial
        y = index
        index -= 1
        last = passcost.SCAN_LAST if y == 0 else 0  # $17CB BPL not taken
        see = relative.can_see_object(state, enemy, y, mm.T_ROBOT, FOV_SCAN)
        budget -= _see_cost(see, clk)
        # $17B7: a non-target tree in the sightline to this robot's HEAD hides it
        if see["tree_in_los_head"]:
            budget -= badline.charge(clk, 0x17B2, passcost.SCAN_SLOT_HIDDEN - last)
            continue
        exposure = _exposure_byte(see)
        if exposure == 0:  # $17BE/$17C0: not visible at all -> next slot
            budget -= badline.charge(clk, 0x17B2, passcost.SCAN_SLOT_UNSEEN - last)
            continue
        if exposure & 0x80:  # $17BA: fully visible (base reached) -> drain target
            budget -= badline.charge(clk, 0x17B2, passcost.SCAN_SLOT_FULL)
            if budget <= 0:  # the ROM has not reached $1825 yet
                return budget, BODY_SCAN, paid(y), partial
            stage, cost = _target_object(state, enemy, y, exposure, clk)
            return budget - cost, stage, 0, -1
        if y == player:  # only the head is visible -> meanie candidate ($17C0)
            partial = y
            budget -= badline.charge(clk, 0x17B2, passcost.SCAN_SLOT_PARTIAL - last)
        else:
            budget -= badline.charge(clk, 0x17B2, passcost.SCAN_SLOT_OTHER - last)


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


def _consider_creating_meanie(state, enemy, budget, index, clk):
    """consider_creating_meanie $197D plus the $1852 tail of target_object, resumable.

    Walks the per-enemy search counter down looking for a tree within 10 tiles of the
    targeted player that the enemy fully sees; the first such tree becomes a meanie.
    One counter step is one unit.  Returns (budget, stage, index)."""
    mem = state.mem
    player = mem[mm.ENEMIES_TARGETED_OBJECT + enemy]
    while True:
        if index <= -2:  # this slot's $1887 is paid; only its write is outstanding
            slot = mem[mm.ENEMIES_MEANIE_SEARCH_OBJECT + enemy]
            index = 0
            if relative.can_see_object(
                state, enemy, slot, mm.T_TREE, FOV_CREATE_MEANIE
            )["full"]:
                mem[mm.ENEMIES_MEANIE_OBJECT + enemy] = slot  # $19E1
                state.obj_type[slot] = mm.T_MEANIE
                mem[mm.ENEMIES_UPDATE_COOLDOWN + enemy] = UPDATE_COOLDOWN_MEANIE_MADE
                return budget, BODY_DONE, 0
            continue
        if budget <= 0:
            return budget, BODY_MAKE_MEANIE, index
        sc = mem[mm.ENEMIES_MEANIE_SEARCH_OBJECT + enemy]
        if sc == 0:  # $198D: scanned everything -> no meanie this pass
            budget -= badline.charge(clk, 0x198F, passcost.MEANIE_SCAN_DONE)
            mem[mm.ENEMIES_MEANIE_ATTEMPT_SCANS + enemy] = (
                mem[mm.ENEMIES_MEANIE_ATTEMPT_SCANS + enemy] + 1
            ) & 0xFF
            mem[mm.ENEMIES_FAILED_MEANIE_MEMORY + enemy] = player
            if mem[mm.ENEMIES_MEANIE_ATTEMPT_SCANS + enemy] >= MEANIE_MAX_ATTEMPTS:
                mem[mm.ENEMIES_DRAINING_COOLDOWN + enemy] = 0  # give up on this player
            else:
                mem[mm.ENEMIES_CONSIDERING_MEANIE + enemy] = 0x80  # keep trying
            return budget, BODY_DONE, 0
        mem[mm.ENEMIES_MEANIE_SEARCH_OBJECT + enemy] = sc - 1
        slot = sc - 1  # $199B DEY: the object index this iteration tests
        if state.obj_flags[slot] & 0x80:  # empty slot
            budget -= badline.charge(clk, 0x198F, passcost.MEANIE_SCAN_SLOT)
            continue
        if state.obj_type[slot] != mm.T_TREE:  # not a tree
            budget -= badline.charge(clk, 0x19AA, passcost.MEANIE_SCAN_OTHER)
            continue
        budget -= badline.charge(clk, 0x19AA, passcost.MEANIE_SCAN_OTHER)
        budget -= badline.charge(clk, 0x19B1, passcost.MEANIE_SCAN_DX)
        dx = (state.obj_x[player] - state.obj_x[slot]) & 0xFF
        if dx >= 0x80:
            dx = 0x100 - dx  # $19B5 abs
        if dx >= 0x0A:  # more than 10 tiles away in x
            continue
        budget -= badline.charge(clk, 0x19C7, passcost.MEANIE_SCAN_DY)
        dy = (state.obj_y[player] - state.obj_y[slot]) & 0xFF
        if dy >= 0x80:
            dy = 0x100 - dy
        if dy >= 0x0A:  # more than 10 tiles away in y
            continue
        see = relative.can_see_object(state, enemy, slot, mm.T_TREE, FOV_CREATE_MEANIE)
        budget -= badline.charge(clk, 0x19D9, passcost.MEANIE_SCAN_SEE)
        budget -= _see_cost(see, clk)
        if not see["full"]:  # enemy hasn't a clear sight of the tree
            continue
        if budget <= 0:  # the ROM has not reached $19E1 yet
            return budget, BODY_MAKE_MEANIE, paid(slot)
        mem[mm.ENEMIES_MEANIE_OBJECT + enemy] = slot  # $19E1
        state.obj_type[slot] = mm.T_MEANIE
        mem[mm.ENEMIES_UPDATE_COOLDOWN + enemy] = UPDATE_COOLDOWN_MEANIE_MADE
        return budget, BODY_DONE, 0


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


def _update_meanie(state, enemy, budget, index, clk):
    """update_meanie $16F2, resumable: the enemy's meanie rotates toward the player
    and, once it is looking at a player it can still see, forcibly hyperspaces them.

    Its one $1887 call is the unit; everything after it is the write.  Returns
    (budget, stage, index)."""
    mem = state.mem
    meanie = mem[mm.ENEMIES_MEANIE_OBJECT + enemy]
    target = mem[mm.ENEMIES_TARGETED_OBJECT + enemy]
    if state.obj_flags[target] & 0x80:  # $16F7: the object the player was in is gone
        _remove_meanie_and_reset_enemy(state, enemy)
        return budget, BODY_DONE, 0
    see = relative.can_see_object(state, meanie, target, mm.T_ROBOT, FOV_SCAN)
    if index >= 0:  # not yet charged
        budget -= _see_cost(see, clk)
        if budget <= 0:
            return budget, BODY_MEANIE, paid(0)
    if not see["in_fov"]:  # $1706: meanie not yet looking at the player -> rotate
        c57 = relative.relative_angles(state, meanie, target)["c57"]
        step = MEANIE_ROTATE_STEP if not (c57 & 0x80) else (0x100 - MEANIE_ROTATE_STEP)
        state.obj_h_angle[meanie] = (state.obj_h_angle[meanie] + step) & 0xFF
        mem[mm.ENEMIES_UPDATE_COOLDOWN + enemy] = UPDATE_COOLDOWN_MEANIE_ROTATE
        start_tune(state, mm.SOUND_MEANIE)  # $1743 JSR $3470
        # $1755 JMP $187B: a meanie's turn redraws it too
        budget -= badline.charge(clk, 0x1728, passcost.MEANIE_ROTATE)
        budget -= _redraw_cost(state, meanie, clk)
        return budget, BODY_DONE, 0
    if target != mem[mm.PLAYER_OBJECT]:  # $1708: player transferred out of the object
        _remove_meanie_and_reset_enemy(state, enemy)
        return budget, BODY_DONE, 0
    if _exposure_byte(see) == 0:  # $170E: meanie can't actually see the player
        _remove_meanie(state, enemy)
        return budget, BODY_DONE, 0
    do_hyperspace(state)  # $1710: forced hyperspace
    return budget, BODY_DONE, 0


# ---------------------------------------------------------------------------
# do_hyperspace $2147 (create a robot on a random low tile + transfer/energy)
# ---------------------------------------------------------------------------
def _create_object(state, otype, clk):
    """create_object $211D: the highest empty slot, typed `otype`, else None.

    Returns (slot, cycles): the walk is 11 a slot, the last one short by its
    untaken $2128 BPL."""
    cost = badline.charge(clk, 0x211D, passcost.CREATE_HEAD)
    for slot in range(mm.NUM_SLOTS - 1, -1, -1):
        if state.obj_flags[slot] & 0x80:
            state.obj_type[slot] = otype
            return slot, cost + badline.charge(clk, 0x212C, passcost.CREATE_HIT)
        cost += badline.charge(
            clk,
            0x2122,
            passcost.CREATE_SLOT - (passcost.CREATE_LAST if slot == 0 else 0),
        )
    return None, cost + badline.charge(clk, 0x212A, passcost.CREATE_NONE)


def _random_tile_coord(prng, clk):
    """$1272: a prnd draw masked to 0..31, rejecting 31 (the board's edge).

    Returns (coordinate, cycles); each rejected draw costs another prnd."""
    cost = badline.charge(clk, 0x1272, passcost.DRAW)
    while True:
        v = prng.next() & 0x1F
        if v != 0x1F:
            return v, cost
        cost += badline.charge(clk, 0x1279, passcost.DRAW_REJECT)


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


def _put_object_in_random_tile_below_z(state, slot, z, prng, clk):
    """$1238: place `slot` on a random flat, empty tile no higher than `z`.

    After 256 misses the height ceiling `z` is raised; it fails once that ceiling
    reaches 12.  Returns (placed, cycles), each lap charged by the test it failed."""
    attempts = 0
    cost = badline.charge(clk, 0x1238, passcost.PLACE_HEAD)
    while True:
        attempts = (attempts - 1) & 0xFF
        if attempts == 0:  # $1242: 256 misses -> relax the height ceiling
            z = (z + 1) & 0xFF
            if z >= 0x0C:
                return False, cost + badline.charge(clk, 0x1270, passcost.PLACE_GIVE_UP)
            cost += badline.charge(clk, 0x1240, passcost.PLACE_WRAP)
        cost += badline.charge(clk, 0x123E, passcost.PLACE_LAP)
        tx, dx = _random_tile_coord(prng, clk)
        ty, dy = _random_tile_coord(prng, clk)
        cost += dx + dy
        b = tile_byte(state, tx, ty)
        if b >= mm.OBJECT_TILE:  # tile already holds an object
            cost += badline.charge(clk, 0x125B, passcost.PLACE_OCCUPIED)
            continue
        if b & 0x0F:  # not a flat tile
            cost += badline.charge(clk, 0x125F, passcost.PLACE_NOT_FLAT)
            continue
        if (b >> 4) >= z:  # tile too high
            cost += badline.charge(clk, 0x1269, passcost.PLACE_TOO_HIGH)
            continue
        _put_object_in_tile(state, slot, tx, ty, prng)
        cost += badline.charge(clk, 0x126B, passcost.PLACE_HIT)
        return True, cost + badline.charge(clk, 0x1F16, passcost.PUT_IN_TILE)


def _consider_discharging_enemy_energy(state, enemy, clk):
    """consider_discharging_enemy_energy $1A5D: if `enemy` has banked drained energy,
    return one unit to the landscape as a TREE on a random flat tile no higher than
    the below-enemies ceiling ($0C06), and decrement the bank.  Returns True if a tree
    was discharged (the ROM then dithers and SKIPS this enemy's drain/rotate for the
    update, $177A), the cycles the call cost, and the slot $1A7D leaves in $91."""
    mem = state.mem
    if mem[mm.ENEMIES_ENERGY_TO_DISCHARGE + enemy] == 0:
        # $1A63: nothing to discharge
        return False, badline.charge(clk, 0x1A5D, passcost.DISCHARGE_NONE), -1
    prng = Prng().load(mem)
    cost = badline.charge(clk, 0x1A65, passcost.DISCHARGE_CREATE)
    slot, ccost = _create_object(state, mm.T_TREE, clk)  # $1A67 create_object(type 2)
    cost += ccost
    if slot is None:
        prng.store(mem)
        return False, cost, -1
    cost += badline.charge(clk, 0x1A6A, passcost.DISCHARGE_PLACE)
    placed, pcost = _put_object_in_random_tile_below_z(
        state, slot, mem[mm.ENEMY_BELOW_Z], prng, clk
    )
    prng.store(mem)
    cost += pcost
    if not placed:  # $1A70: no tile found -> abandon (slot stays flagged empty)
        return False, cost + badline.charge(clk, 0x1A70, passcost.DISCHARGE_ABANDON), -1
    a = mm.ENEMIES_ENERGY_TO_DISCHARGE + enemy
    mem[a] = (mem[a] - 1) & 0xFF  # $1A7A DEC
    return True, cost + badline.charge(clk, 0x1A72, passcost.DISCHARGE_DONE), slot


def do_hyperspace(state):
    """do_hyperspace $2147: create a synthoid on a random tile no higher than the
    player, spend the robot's energy, and transfer the player into it.  A player
    with too little energy dies; hyperspacing off the platform is the meanie's win
    condition against the player, hyperspacing *from* the platform completes the
    landscape ($2187)."""
    mem = state.mem
    idle = badline.frame_clock(armed=False)  # the hyperspace's cycles are not budgeted
    prng = Prng().load(mem)
    slot, _ccost = _create_object(state, mm.T_ROBOT, idle)
    if slot is None:
        prng.store(mem)
        return
    player = mem[mm.PLAYER_OBJECT]
    z = (state.obj_z_height[player] + 1) & 0xFF
    placed, _pcost = _put_object_in_random_tile_below_z(state, slot, z, prng, idle)
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


def _tile_top(state, x):
    """The topmost object of the tile slot ``x`` stands on, or -1 if there is none."""
    tb = terrain.tile_byte(state, state.obj_x[x], state.obj_y[x])
    return -1 if tb < mm.OBJECT_TILE else tb & 0x3F


def _find_drainable_boulder_or_tree(state, enemy, budget, index, clk):
    """find_drainable_boulder_or_tree_on_stack ($1AB0), resumable, one slot per unit.

    Scanning slots 63..0, a boulder or a stacked object (flags >= $40) marks a
    candidate tile whose topmost object, if a fully-visible tree or boulder, is
    drained.  Returns (budget, index, drained slot / -1 exhausted / -2 suspended)."""
    while True:
        if index <= -2:  # this slot's $1887 is paid; only its write is outstanding
            x = -2 - index
            y = _tile_top(state, x)
            index = x - 1
            if relative.can_see_object(state, enemy, y, state.obj_type[y], FOV_SCAN)[
                "full"
            ]:
                state.mem[mm.TARGETED_OBJECT_SLOT] = y
                hit = badline.charge(clk, 0x1AE6, passcost.TILE_SCAN_HIT)
                return budget - hit, index, y
            budget -= badline.charge(
                clk,
                0x1AE6,
                passcost.TILE_SCAN_NEXT - (passcost.TILE_SCAN_LAST if x == 0 else 0),
            )
            continue
        if index < 0:  # $1AF2: the whole scan came up empty
            gone = badline.charge(clk, 0x1AF2, passcost.TILE_SCAN_EXHAUSTED)
            return budget - gone, index, -1
        if budget <= 0:
            return budget, index, -2
        x = index
        index -= 1
        last = passcost.TILE_SCAN_LAST if x == 0 else 0  # $1AF0 BPL not taken
        flags = state.obj_flags[x]
        if flags & 0x80:  # empty slot
            budget -= badline.charge(clk, 0x1AB2, passcost.TILE_SCAN_EMPTY - last)
            continue
        if not (flags >= 0x40 or state.obj_type[x] == mm.T_BOULDER):
            budget -= badline.charge(clk, 0x1AB7, passcost.TILE_SCAN_OTHER - last)
            continue
        if flags >= 0x40:
            budget -= badline.charge(clk, 0x1AB9, passcost.TILE_SCAN_STACKED)
        else:
            budget -= badline.charge(clk, 0x1ABE, passcost.TILE_SCAN_LOOSE)
        budget -= badline.charge(clk, 0x1AC2, passcost.TILE_SCAN_TILE)
        y = _tile_top(state, x)
        if y < 0:  # no object on the tile (shouldn't happen)
            budget -= badline.charge(clk, 0x1AD3, passcost.TILE_SCAN_NO_TILE - last)
            continue
        otype = state.obj_type[y]
        budget -= badline.charge(clk, 0x1AD3, passcost.TILE_SCAN_TOP)
        if otype not in (mm.T_TREE, mm.T_BOULDER):
            budget -= badline.charge(clk, 0x1ADD, passcost.TILE_SCAN_WRONG_TOP - last)
            continue
        see = relative.can_see_object(state, enemy, y, otype, FOV_SCAN)
        if otype == mm.T_TREE:
            budget -= badline.charge(clk, 0x1ADD, passcost.TILE_SCAN_SEE)
        else:
            budget -= badline.charge(clk, 0x1ADF, passcost.TILE_SCAN_SEE_BOULDER)
        budget -= _see_cost(see, clk)
        if not see["full"]:
            budget -= badline.charge(clk, 0x1AE6, passcost.TILE_SCAN_NEXT - last)
            continue
        if budget <= 0:  # the ROM has not reached $1AEA yet
            return budget, paid(x), -2
        state.mem[mm.TARGETED_OBJECT_SLOT] = y
        hit = badline.charge(clk, 0x1AE6, passcost.TILE_SCAN_HIT)
        return budget - hit, index, y


# ---------------------------------------------------------------------------
# update_enemies $16B5
# ---------------------------------------------------------------------------
def _dec_cursor(state):
    mem = state.mem
    c = mem[mm.CURSOR]
    mem[mm.CURSOR] = (c - 1) if c > 0 else 7


def dispatch_cycles(state, clk):
    """$16B5..$16E6: the type dispatch and the $16CC absorbed test, which write
    nothing, so a frame boundary inside them leaves no trace."""
    otype = state.obj_type[state.mem[mm.CURSOR]]
    if otype == mm.T_SENTINEL:
        return badline.charge(clk, 0x16B5, passcost.UPDATE_DISPATCH_SENTINEL)
    if otype == mm.T_SENTRY:
        return badline.charge(clk, 0x16B5, passcost.UPDATE_DISPATCH_SENTRY)
    return badline.charge(clk, 0x16B5, passcost.UPDATE_NOT_ENEMY)


def update_body(state, budget, clk):
    """$16E6..$16D6: consider the enemy at the cursor, or discharge an absorbed one.

    Every state write $16B5 makes before the prnd is made here; a non-enemy slot
    makes none.  Spends ``budget`` and returns (budget, finished) -- unfinished means
    the frame ran out inside $16E6 and ``State.body_*`` names where."""
    mem = state.mem
    x = mem[mm.CURSOR]
    if state.obj_type[x] not in mm.ENEMY_TYPES:  # $16BB sentry(1)/Sentinel(5) only
        return budget, True
    if state.obj_flags[x] & 0x80:  # $16CC BPL: absorbed -> discharge its bank only
        budget -= badline.charge(clk, 0x16CC, passcost.UPDATE_ABSORBED)
        return budget - _consider_discharging_enemy_energy(state, x, clk)[1], True
    budget, stage, index, partial = _consider_enemy_state(
        state, x, budget, state.body_stage, state.body_index, state.body_partial, clk
    )
    if stage == BODY_DONE:
        state.body_stage, state.body_index, state.body_partial = BODY_ENTRY, 0, -1
        return budget, True
    state.body_stage, state.body_index, state.body_partial = stage, index, partial
    return budget, False


def update_cursor(state, clk):
    """$16D9: store the advanced $16D6 prnd stream and decrement the cursor (7->0
    wrap).  Returns the cycles after the prnd, which UPDATE_PRND already charged."""
    mem = state.mem
    x = mem[mm.CURSOR]
    prng = Prng().load(mem)
    prng.next()
    prng.store(mem)
    _dec_cursor(state)
    if x == 0:
        return badline.charge(clk, 0x16D9, passcost.UPDATE_CURSOR_WRAP)
    return badline.charge(clk, 0x16D9, passcost.UPDATE_CURSOR)


UNBOUNDED = 1 << 40  # a budget no single $16E6 can outrun: the isolated-round path


def update_enemies(state):
    """$16B5 whole: dispatch, body, prnd and cursor, for the isolated round.

    The frame loop runs the three parts apart -- see :func:`advance_frame_python`."""
    clk = badline.frame_clock(armed=False)  # the isolated round places no badline
    cost = dispatch_cycles(state, clk)
    left, _done = update_body(state, UNBOUNDED, clk)
    cost += UNBOUNDED - left
    return cost + passcost.UPDATE_PRND + update_cursor(state, clk)


def step(state):
    """Advance the world by one game round: cooldown tick + one enemy update.

    This is the ISOLATED-ROUTINE round (tick_cooldowns + update_enemies, 1:1) that the
    py65 oracle (tests/oracle.step_enemy_round) captures the golden with -- a valid
    transition-function unit, but NOT the running game's cadence.  Real-time advance goes
    through :func:`advance_frames` (the two decoupled ROM clocks)."""
    tick_cooldowns(state, badline.frame_clock(armed=False))
    update_enemies(state)


# ---------------------------------------------------------------------------
# real-game frame cadence: update_game_loop $1289 + raster IRQ $9663 / scroll $3684
# ---------------------------------------------------------------------------


def cooldown_frame(state, clk):
    """The cooldown clock for ONE video frame -- $130C called once per frame (raster
    $9663, or the scroll loop $3684 while scrolling; the two are mutually exclusive via
    $0CD8, so exactly one $130C per frame).  Advance the integer Bresenham accumulator
    $1335 += $CD and, only on carry (205/256), run update_enemy_cooldowns ($1317, itself
    1-in-3 via the $0C50 gate).  Suppressed until the player's first action ($0CE5,
    $9659/$367f).  All integer -- no rounding to drift.  Returns its cycles, which the
    foreground loses from that frame's budget."""
    mem = state.mem
    if mem[mm.PLAYER_NOT_ACTED] & 0x80:  # $965C BMI $9669: no $130C and no $1635 either
        return badline.charge(clk, 0x9659, passcost.IRQ_GATE_SHUT)
    cost = badline.charge(clk, 0x965E, passcost.IRQ_GATE_OPEN)
    acc = mem[mm.COOLDOWN_BRESENHAM] + mm.COOLDOWN_BRESENHAM_STEP
    mem[mm.COOLDOWN_BRESENHAM] = acc & 0xFF
    if acc > 0xFF:  # $1315 BCC skip -> only the carry runs the cooldown decrement
        return cost + tick_cooldowns(state, clk)
    return cost + badline.charge(clk, 0x130C, passcost.COOLDOWN_TICK_NO_CARRY)


def start_tune(state, tune):
    """$3470 -> $8DB4: point a voice at ``tune``'s eight-byte $AC00 descriptor.

    Only the three bytes :func:`sound_frame` reads are modelled -- the note timer
    $8E86,X, the gate reload $8E8E,X and the voice flag $8E96,X -- the rest of $8DB4
    is SID writes.  Its own cycles are already inside ROTATE/MEANIE_ROTATE/TUNE_DRAIN.
    """
    mem = state.mem
    desc = mm.SOUND_TABLE + tune * 8
    x = mem[desc]
    mem[mm.SOUND_NOTE_LENGTH + x] = mem[mm.SOUND_LENGTHS + (mem[desc + 3] & 0x0F)]
    mem[mm.SOUND_VOICE_FLAG + x] = (
        mm.SOUND_VOICE_IDLE  # $8E4F BMI: an un-gated tune leaves the voice off
        if mem[desc + 7] & 0x80
        else mem[mm.SOUND_VOICE_ON + x]
    )
    mem[mm.SOUND_NOTE_TIMER + x] = mem[desc + 6]  # $8E7A: frames of the first note


def _voice_next(x):
    """$8F08 DEX/BPL: voice 0 leaves the lap by the untaken branch."""
    return passcost.SOUND_VOICE_NEXT if x else passcost.SOUND_VOICE_LAST


def sound_frame(state, clk):
    """The note tick $8ED1 for ONE video frame ($963D JSR $FFC2), and its cycles.

    Three voices, X counting 2 down to 0.  A voice's note timer $8E86,X counts down a
    frame a lap; at 0 it reloads the gate timer $8E92,X from $8E8E,X, and when THAT
    runs out the voice is silenced ($8E96,X = $80).  $80 is the idle timer."""
    mem = state.mem
    cyc = badline.charge(clk, 0x963D, passcost.SOUND_TICK_FIXED)
    for x in (2, 1, 0):
        cyc += badline.charge(clk, 0x8ED3, passcost.SOUND_VOICE_READ)
        note = mem[mm.SOUND_NOTE_TIMER + x]
        gate_due = note == 0  # $8ED6 BEQ $8EEE: the note timer is already out
        if gate_due:
            cyc += badline.charge(clk, 0x8ED6, passcost.SOUND_VOICE_SPENT)
        elif note & 0x80:  # $8ED8 BMI $8F08: an idle voice ticks nothing
            cyc += badline.charge(clk, 0x8ED6, passcost.SOUND_VOICE_OFF)
        else:
            note -= 1
            mem[mm.SOUND_NOTE_TIMER + x] = note
            cyc += badline.charge(clk, 0x8ED8, passcost.SOUND_VOICE_TICK)
            if note:
                cyc += badline.charge(clk, 0x8EDD, passcost.SOUND_VOICE_MORE)
            else:  # $8EDF: the note ends, the gate timer reloads
                mem[mm.SOUND_GATE_TIMER + x] = mem[mm.SOUND_NOTE_LENGTH + x]
                cyc += badline.charge(clk, 0x8EDF, passcost.SOUND_VOICE_NOTE)
                gate_due = True
        if not gate_due:
            cyc += badline.charge(clk, 0x8F08, _voice_next(x))
            continue
        cyc += badline.charge(clk, 0x8EEE, passcost.SOUND_GATE_READ)
        gate = mem[mm.SOUND_GATE_TIMER + x]
        if gate == 0:
            cyc += badline.charge(clk, 0x8EF1, passcost.SOUND_GATE_OFF)
        else:
            gate = (gate - 1) & 0xFF
            mem[mm.SOUND_GATE_TIMER + x] = gate
            cyc += badline.charge(clk, 0x8EF3, passcost.SOUND_GATE_TICK)
            if gate:
                cyc += badline.charge(clk, 0x8EF6, passcost.SOUND_GATE_MORE)
            else:  # $8EF8: the gate runs out, the voice goes quiet
                mem[mm.SOUND_VOICE_FLAG + x] = 0x80
                cyc += badline.charge(clk, 0x8EF8, passcost.SOUND_GATE_END)
        cyc += badline.charge(clk, 0x8F08, _voice_next(x))
    return cyc


def advance_frame_python(state, plotting=False):
    """Advance the world by ONE video frame, faithful to the ROM cadence.

    The raster-IRQ cooldown tick ($9663/$1317) runs BEFORE the frame's foreground passes,
    so an enemy the tick makes due rotates in the SAME frame, not the next one. Ticking
    after the passes costs a whole rotation of phase. ``plotting`` suppresses the sweep.
    """
    clk = badline.frame_clock()  # the raster IRQ pins the frame's phase every frame
    irq = badline.charge(clk, 0x95E9, passcost.IRQ_BODY) - passcost.IRQ_BODY
    irq += sound_frame(state, clk) + cooldown_frame(state, clk)
    if plotting:
        return
    budget = state.cycle_residual + passcost.FOREGROUND_CYCLES - irq
    budget -= badline.FRAME_STEAL_CEILING  # the 25 windows, refunded by charge()
    phase = state.pass_phase
    while budget > 0:
        if phase == PHASE_HEAD:
            budget -= badline.charge(clk, 0x1289, passcost.PASS_HEAD)
            budget -= dispatch_cycles(state, clk)
            phase = PHASE_BODY
            if budget <= 0:
                break
        if phase == PHASE_BODY:
            budget, done = update_body(state, budget, clk)
            if not done:  # the frame ran out INSIDE $16E6; resume there next frame
                break
            budget -= badline.charge(clk, 0x16D6, passcost.UPDATE_PRND)
            phase = PHASE_CURSOR
            if budget <= 0:
                break
        budget -= update_cursor(state, clk)
        budget -= badline.charge(clk, 0x12A2, passcost.PASS_TAIL)
        budget -= passcost.exposure_cycles(state.mem, clk)
        phase = PHASE_HEAD
    state.cycle_residual = budget
    state.pass_phase = phase


def advance_frames_python(state, n_frames, plotting=False):
    """The reference frame loop -- pure Python, and the fallback when numba is absent."""
    for _ in range(int(n_frames)):
        advance_frame_python(state, plotting=plotting)


def _jit_usable(state):
    """The jit twin needs a writable buffer to view; a bytes-backed state has none."""
    return _HAVE_JIT and not isinstance(state.mem, bytes)


def advance_frames(state, n_frames, plotting=False):
    """Advance ``n_frames`` video frames.  ``plotting`` marks a scroll/replot span in which
    update_enemies is suppressed (only the cooldown clock advances).

    Dispatches to the numba twin (:mod:`sentinel.enemies_jit`) when numba is present,
    else to :func:`advance_frames_python`; the two are byte-identical."""
    if _jit_usable(state):
        remaining = int(n_frames)
        while remaining > 0:
            (
                state.cycle_residual,
                state.pass_phase,
                state.body_stage,
                state.body_index,
                state.body_partial,
                remaining,
                target,
                left,
                columns,
            ) = enemies_jit.advance_frames(
                state.mem,
                remaining,
                plotting,
                state.cycle_residual,
                state.pass_phase,
                state.body_stage,
                state.body_index,
                state.body_partial,
            )
            if target < 0:  # no $1FFC strip replot to price: the run is complete
                break
            frames = projector.strip_replot_frames(state, target, left, columns)
            state.cycle_residual -= int(round(frames * passcost.PAL_FRAME_CYCLES))
        return
    advance_frames_python(state, n_frames, plotting=plotting)


def advance_frame(state, plotting=False):
    """Advance the world by ONE video frame (the jit twin when numba is present)."""
    advance_frames(state, 1, plotting=plotting)


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
