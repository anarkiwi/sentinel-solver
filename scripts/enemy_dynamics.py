#!/usr/bin/env python3
"""Time-accurate, deterministic enemy dynamics for The Sentinel (C64).

This module is the *phase-aware* successor to `game_model.seen_by_enemy`, which is
deliberately conservative (it reports whether a tile is EVER exposed if the enemy
rotates toward it, dropping the instantaneous angular field). Here we faithfully
port the per-tick rotation, the angular field-of-view test, and the draining
cooldown logic of the game, so the planner can ask the
*time-accurate* question: "is enemy E looking at tile (x,y) AT this tick?".

Every constant cites its source address. The two layers we reuse from `game_model`:
  * `can_see(...)` for the terrain line-of-sight raytrace (the ROM's
    check_for_line_of_sight_to_tile $1CDD, called from $18F6), and
  * the object/tile geometry helpers.

----------------------------------------------------------------------------
HOW THE ROM RUNS ENEMIES (control flow)
----------------------------------------------------------------------------
* `update_game` ($127C) is one game ROUND. It calls `update_enemies` ($16B5)
  exactly once per round ($129F).
* `update_enemies` processes ONE enemy per round, indexed by `$0090`, which
  decrements 7->0 and wraps to 7 ($16D9 DEC $0090 / $16DD reload #$7). Slots
  whose objects_type isn't sentry(1)/Sentinel(5) are skipped ($16BE/$16C2).
* The outer play loop ($363D..$3694) calls `update_enemy_cooldowns` ($1317)
  once per loop iteration ($3684). That routine is gated by `$0C50`: it only
  actually decrements the cooldowns once every 3 calls (counts $0C50 2->1->0,
  decrements + reloads 2 at 0; $1331/$1317-$132D). Each cooldown "sticks at 1"
  -- it is decremented only while `>= 2` ($1321 CMP #$2 / BCC skip / $1325 DEC).
  The loop spans X=$17..0 = 24 bytes = enemies_draining_cooldown ($0C20, 8),
  enemies_rotation_cooldown ($0C28, 8), enemies_update_cooldown ($0C30, 8).

PER-ENEMY UPDATE (consider_enemy_state $16E6, when this enemy's turn comes up):
  * If `update_cooldown >= 2` -> skip this enemy entirely ($16E9 CMP #$2 / BCS).
    Otherwise set `update_cooldown = 4` ($16ED) and set the FOV width
    `$0C68 = $14` (20) ($16F2), then do the scan:
      - look for a drainable robot / the player (find_drainable_robot_loop $17B2),
        else a tree/boulder (reset_draining_cooldown_and_look_for_tree_or_boulder
        $17E0). check_if_enemy_can_see_object ($1887) does the angular-FOV test
        then the terrain-LOS test.
      - If a target is found (target_object $1825): set
        `draining_cooldown = $78` (120, $1835) the first time; when the draining
        cooldown is exactly 1 ($1831 CMP #$1 / $183D BNE), drain it
        (reduce_object_energy $1A08, 1 energy/tick or a type-downgrade) and set
        `update_cooldown = $1E` (30, $1848).
      - If no drainable target and the enemy isn't on draining cooldown, it
        ROTATES (rotate_enemy $1805) IF `rotation_cooldown < 2` ($17FE): add the
        per-enemy rotation speed `$9D37,X` to objects_h_angle ($180D) and reload
        `rotation_cooldown = $C8` (200, $1813).

We model TICK = one `update_enemies` round (one game frame). On each tick we:
  (1) advance the cooldown phase (the 1-in-3 cadence + stick-at-1), then
  (2) run the per-enemy scan/rotate/drain for the enemy whose turn it is.
This mirrors the ROM closely enough for the planner; the only abstraction is
that we don't simulate the meanie-creation side-channel (it does not change
where the enemy looks or what it drains from the player's standpoint).
"""

import sys
import os
import copy
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Tuple

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

import game_model as gm
from game_model import (
    can_see,
    tile_surface_height,
    ENEMY_TYPES,
    ENEMY_EYE,
    ROBOT_EYE,
    T_SENTINEL,
    T_ROBOT,
    T_BOULDER,
    T_TREE,
)
from game_state import GameState, GameObject

# ============================================================================
# CONSTANTS (every value cites the game address it was read from)
# ============================================================================

# --- Rotation -----------------------------------------------------------------
# Each enemy's per-rotation angular step lives in the table $9D37,X. It is set to
# +$14 (clockwise) or -$14 == $EC (anticlockwise) at init (set_enemies_rotation_speed
# $1586; loaded #$14 at $1580 / #$EC at $1584, chosen by a random bit). The comment
# at $1580 reads "rotate clockwise by one screen width (20 * 256/360 degrees)".
# h_angle is a full byte: 256 units == 360 degrees, so one unit == 1.40625 deg and
# $14 == 20 units == 28.125 deg.
ROTATION_SPEED_TABLE = 0x9D37  # objects' per-enemy rotation step, indexed by slot
ROTATION_STEP_CW = 0x14  # $1580: +20 units
ROTATION_STEP_CCW = 0xEC  # $1584: -20 units (two's complement of 20)
DEG_PER_UNIT = 360.0 / 256.0  # 1.40625 deg per angle unit

# --- Field of view ------------------------------------------------------------
# Horizontal FOV width, $0C68, reloaded to $14 (20 units == 28.125 deg) at the
# start of each enemy scan ($16F2 LDA #$14 / STA $0C68). The angular test in
# check_if_enemy_can_see_object ($18BE..$18CA):
#     A = ($0C57) ; relative angle, already offset by +$0A (half a screen width,
#                 ; added at calculate_object_relative_angles_and_distance $8423)
#     A = A - $0A + ($0C68 >> 1)        ; $18C2 SBC #$0A ; $18C5 ADC ($0C68>>1)
#     if A >= $0C68: out of FOV         ; $18C7 CMP $0C68 / $18CA BCS leave
# With $0C68 == $14 and its half $0A, this reduces to: in-FOV iff the *raw*
# relative angle (signed, units) lies in [-$0A, +$0A) == [-10, +10) units ==
# [-14.0625, +14.0625) degrees, i.e. a 28.125-deg-wide cone centred on the enemy
# facing. (The +$0A offset in $0C57 and the -$0A here cancel.)
FOV_ADDR = 0x0C68
FOV_WIDTH_UNITS = 0x14  # $16F2: 20 units == 28.125 deg total width
FOV_HALF_UNITS = FOV_WIDTH_UNITS // 2  # $0A == 10 units == 14.0625 deg each side
# Vertical FOV: the BBC notes ~16.875 deg high. The C64 code does NOT gate on a
# vertical angle directly; instead check_if_enemy_can_see_object runs the terrain
# LOS test TWICE ($18D4 LDA #$2 / STA $001E; loop at $1914 BNE), once from the
# object's upper point and once from $E0 lower ($1909 SBC #$E0) -- i.e. it checks
# both the top and (near) the base of the target. We reproduce that as two LOS
# probes (object top + object base) in `enemy_sees`. The 16.875-deg figure is the
# screen's vertical extent (aspect of the 28.125-deg-wide view); we don't need an
# explicit vertical-angle gate because the ROM doesn't apply one.
VERT_PROBE_LOW_OFFSET = 0xE0 / 256.0  # $1909 SBC #$E0: low probe is ~$E0/256 unit lower

# --- Cooldown reload values ---------------------------------------------------
UPDATE_COOLDOWN_SCAN = 0x04  # $16ED: reload 4 each time the enemy scans
UPDATE_COOLDOWN_DRAIN = 0x1E  # $17F1 / $1848: 30 after draining an object
ROTATION_COOLDOWN_RELOAD = 0xC8  # $1813: 200 after a rotation step
DRAINING_COOLDOWN_RELOAD = 0x78  # $1835: 120 when an enemy first targets an object
DRAIN_FIRE_VALUE = 0x01  # $1831/$183D: drain fires when draining_cooldown == 1
COOLDOWN_STICK = 0x02  # $1321/$16E9/$17FE: thresholds compare against 2
# update_enemy_cooldowns is gated by $0C50, which makes the decrement happen once
# every COOLDOWN_DECREMENT_PERIOD outer-loop iterations ($1331 DEC $0C50 / reload 2
# at $132B). For our per-round tick model we fold this into a 3-phase counter.
COOLDOWN_DECREMENT_PERIOD = 3  # $0C50 cycles 2->1->0 then decrements

# Cooldown-array base addresses (for cross-checking against generated RAM).
ENEMIES_DRAINING_COOLDOWN = 0x0C20  # $0C20
ENEMIES_ROTATION_COOLDOWN = 0x0C28  # $0C28
ENEMIES_UPDATE_COOLDOWN = 0x0C30  # $0C30


# ============================================================================
# PER-ENEMY PHASE STATE
# ============================================================================
@dataclass
class EnemyState:
    """The time-varying phase of one enemy, mirroring the ROM variables.

    Fields map directly to the C64 arrays:
      h_angle           -> objects_h_angle[slot]  ($09C0)
      rotation_speed    -> $9D37[slot]            (+$14 / $EC)
      update_cooldown   -> enemies_update_cooldown[slot]   ($0C30)
      rotation_cooldown -> enemies_rotation_cooldown[slot] ($0C28)
      draining_cooldown -> enemies_draining_cooldown[slot] ($0C20)
    """

    slot: int
    type: int
    x: int
    y: int
    h_angle: int  # 0..255, objects_h_angle
    rotation_speed: int  # +$14 or $EC (two's complement)
    update_cooldown: int = 0  # $0C30
    rotation_cooldown: int = 0  # $0C28
    draining_cooldown: int = 0  # $0C20
    targeted_slot: int = -1  # enemies_targeted_object ($0CA8), -1 == none


@dataclass
class EnemyPhase:
    """The full enemy-phase snapshot the planner advances tick by tick."""

    enemies: Dict[int, EnemyState] = field(default_factory=dict)
    # $0090: which enemy slot update_enemies will process this round (7->0 wrap).
    cursor: int = 7
    # 3-phase counter folding update_enemy_cooldowns' $0C50 gate (decrement at 0).
    cd_phase: int = 0
    tick: int = 0

    def copy(self) -> "EnemyPhase":
        return EnemyPhase(
            enemies={s: copy.copy(e) for s, e in self.enemies.items()},
            cursor=self.cursor,
            cd_phase=self.cd_phase,
            tick=self.tick,
        )


def _rotation_speed_for(state: GameState, slot: int) -> int:
    """The enemy's rotation step. We can't read $9D37 from a GameState (it isn't
    parsed), so we recover sign from the ROM rule: +$14 or $EC. When a live RAM
    image is available, prefer `enemy_phase_from_ram` which reads $9D37 exactly.
    Here we default every enemy to clockwise (+$14); callers that have the RAM
    should override via from_ram. The magnitude (20) is fixed; only the sign is
    landscape-random ($1578 'lowest bit of random number')."""
    return ROTATION_STEP_CW


def _enemies(state: GameState) -> List[GameObject]:
    return [o for o in state.objects if o.type in ENEMY_TYPES]


def init_phase(
    state: GameState,
    rotation_speeds: Optional[Dict[int, int]] = None,
    update_cooldowns: Optional[Dict[int, int]] = None,
) -> EnemyPhase:
    """Build the initial EnemyPhase from a GameState.

    Initial values per the ROM (set_palette_and_initialise_enemies loop $1500):
      * objects_h_angle[slot] is the generated facing (read straight from state),
      * update_cooldown = (prnd & $3F) | $05 ($157D), so 5..63; we don't have the
        prng draw from a GameState, so callers with a live RAM image should pass
        `update_cooldowns`/`rotation_speeds` (see `init_phase_from_ram`). Without
        them we start all enemies at update_cooldown=0 (ready to scan immediately)
        and draining/rotation cooldown=0, which is the WORST CASE for the player
        (enemy active as soon as possible) -- a deliberately conservative default.
      * draining_cooldown = 0, rotation_cooldown = 0 ($0C20/$0C28 start cleared).
    """
    rotation_speeds = rotation_speeds or {}
    update_cooldowns = update_cooldowns or {}
    enemies = {}
    for o in _enemies(state):
        enemies[o.slot] = EnemyState(
            slot=o.slot,
            type=o.type,
            x=o.x,
            y=o.y,
            h_angle=o.h_angle & 0xFF,
            rotation_speed=rotation_speeds.get(
                o.slot, _rotation_speed_for(state, o.slot)
            ),
            update_cooldown=update_cooldowns.get(o.slot, 0),
            rotation_cooldown=0,
            draining_cooldown=0,
        )
    return EnemyPhase(enemies=enemies, cursor=7, cd_phase=0, tick=0)


def init_phase_from_ram(state: GameState, mem) -> EnemyPhase:
    """Exact initial phase, reading the real $9D37 rotation speeds and $0C30/$0C28/
    $0C20 cooldowns out of a live RAM image (a Py65Source.from_landscape mem or a
    ViceSource snapshot). `mem` is anything indexable like bytes.

    This is the constructor to cross-check against Py65Source.from_landscape(0):
    the generated state's objects_h_angle and the $9D37/$0C30 arrays are the real
    initial enemy phase.
    """
    enemies = {}
    for o in _enemies(state):
        s = o.slot
        enemies[s] = EnemyState(
            slot=s,
            type=o.type,
            x=o.x,
            y=o.y,
            h_angle=o.h_angle & 0xFF,
            rotation_speed=mem[ROTATION_SPEED_TABLE + s],
            update_cooldown=mem[ENEMIES_UPDATE_COOLDOWN + s],
            rotation_cooldown=mem[ENEMIES_ROTATION_COOLDOWN + s],
            draining_cooldown=mem[ENEMIES_DRAINING_COOLDOWN + s],
        )
    return EnemyPhase(enemies=enemies, cursor=7, cd_phase=0, tick=0)


# ============================================================================
# ANGULAR FIELD-OF-VIEW TEST (check_if_enemy_can_see_object $1887 angular part)
# ============================================================================
def _signed_angle_delta(a: int, b: int) -> int:
    """Signed difference a-b on the 256-unit circle, in [-128, 127]."""
    d = (a - b) & 0xFF
    if d >= 128:
        d -= 256
    return d


def in_fov(enemy: EnemyState, x: int, y: int, state: GameState) -> bool:
    """True if tile (x,y) is within the enemy's *current* horizontal angular FOV.

    Faithful to the $18BE..$18CA test: in-FOV iff the signed relative angle (target
    bearing minus enemy facing) is in [-FOV_HALF_UNITS, +FOV_HALF_UNITS) units.
    The ROM bearing is computed by calculate_angle ($9287) from the relative x/z;
    we use atan2 over the board axes, which matches the same geometry. (We bear
    against the (x,y) tile axes, the game's two horizontal axes.)"""
    import math

    dx = x - enemy.x
    dy = y - enemy.y
    if dx == 0 and dy == 0:
        return True
    # bearing in angle units (0..255). The game's h_angle increases the same way
    # objects_h_angle is read for rotation; we measure atan2(dy,dx) and quantise.
    bearing = int(round((math.atan2(dy, dx) / (2 * math.pi)) * 256)) & 0xFF
    d = _signed_angle_delta(bearing, enemy.h_angle)
    return -FOV_HALF_UNITS <= d < FOV_HALF_UNITS


# ============================================================================
# enemy_sees: time-accurate LOS + FOV (the phase-aware seen_by_enemy)
# ============================================================================
def enemy_sees(
    state: GameState, enemy: EnemyState, x: int, y: int, object_top: float = ROBOT_EYE
) -> bool:
    """Whether `enemy`, at its CURRENT angle/phase, can see tile (x,y).

    Two gates, mirroring check_if_enemy_can_see_object $1887:
      (a) angular FOV: the tile is within +/-FOV_HALF_UNITS of the enemy facing
          ($18C7 angular test), and
      (b) terrain LOS: a clear ray over the height field, run at TWO probe heights
          (object top + object base) to mirror the two-pass $18E6 loop ($1909
          SBC #$E0). We reuse game_model.can_see for the raytrace.
    """
    if not in_fov(enemy, x, y, state):
        return False
    # The terrain-LOS part depends only on (enemy tile, target tile, heights), NOT
    # on the enemy facing, so it is cacheable per terrain (which is constant during
    # a solve). Memoise it in a module cache keyed by the terrain's content hash --
    # game_model.apply deep-copies the state every step, so we key on CONTENT.
    surf = tile_surface_height(state, float(x), float(y))
    tgt_top = surf + object_top
    tgt_low = max(surf, tgt_top - VERT_PROBE_LOW_OFFSET)
    cache = _los_cache_for(state)
    key = ((enemy.x, enemy.y), (x, y), round(tgt_top, 3), round(tgt_low, 3))
    v = cache.get(key)
    if v is None:
        v = can_see(
            state,
            (enemy.x, enemy.y),
            (x, y),
            eye_offset=ENEMY_EYE,
            target_height=tgt_top,
        ) or can_see(
            state,
            (enemy.x, enemy.y),
            (x, y),
            eye_offset=ENEMY_EYE,
            target_height=tgt_low,
        )
        cache[key] = v
    return v


_LOS_CACHE: Dict[int, Dict] = {}


def _los_cache_for(state: GameState) -> Dict:
    k = getattr(state, "_terrain_hash", None)
    if k is None:
        k = hash(tuple(tuple(row) for row in state.height))
        try:
            state._terrain_hash = k
        except Exception:
            pass
    return _LOS_CACHE.setdefault(k, {})


def enemies_seeing(
    state: GameState, phase: EnemyPhase, x: int, y: int, object_top: float = ROBOT_EYE
) -> List[int]:
    """Slots of enemies whose CURRENT phase sees tile (x,y)."""
    return [
        e.slot for e in phase.enemies.values() if enemy_sees(state, e, x, y, object_top)
    ]


# ============================================================================
# step_enemies: advance one game tick (mirror of update_enemies + cooldowns)
# ============================================================================
def _decrement_cooldowns(phase: EnemyPhase) -> None:
    """Mirror update_enemy_cooldowns ($1317): once every COOLDOWN_DECREMENT_PERIOD
    ticks, decrement every cooldown that is >= 2 (stick at 1). $0C50 gate."""
    phase.cd_phase = (phase.cd_phase + 1) % COOLDOWN_DECREMENT_PERIOD
    if phase.cd_phase != 0:
        return
    for e in phase.enemies.values():
        if e.draining_cooldown >= COOLDOWN_STICK:
            e.draining_cooldown -= 1
        if e.rotation_cooldown >= COOLDOWN_STICK:
            e.rotation_cooldown -= 1
        if e.update_cooldown >= COOLDOWN_STICK:
            e.update_cooldown -= 1


def _player_target(
    state: GameState, e: EnemyState
) -> Optional[Tuple[int, int, float, int]]:
    """The object this enemy would target this scan: the player robot if seen,
    else the nearest tree/boulder it can see. Returns (x,y,top,slot) or None.

    Simplification vs find_drainable_robot_loop $17B2 / find_drainable_boulder_or_
    tree_on_stack $1AB0: we prioritise the player (the planner cares about player
    drain) then any visible robot/boulder/tree, scanning slots high->low as the ROM
    does ($17B0 LDY #$3F downward)."""
    p = state.player
    # player first (find_drainable_robot_loop checks CPY player_object $17C4)
    if p is not None and enemy_sees(state, e, p.x, p.y, ROBOT_EYE):
        return (p.x, p.y, ROBOT_EYE, p.slot)
    best = None
    for o in sorted(state.objects, key=lambda o: -o.slot):
        if o.type in (T_ROBOT, T_BOULDER, T_TREE) and o.slot != (p.slot if p else -1):
            if enemy_sees(state, e, o.x, o.y, ROBOT_EYE):
                best = (o.x, o.y, ROBOT_EYE, o.slot)
                break
    return best


def step_enemies(state: GameState, phase: EnemyPhase) -> EnemyPhase:
    """Advance the enemy phase by ONE game tick (one update_enemies round).

    Faithful mirror of one update_game round:
      1. update_enemy_cooldowns: decrement cooldowns on the 1-in-3 cadence (stick
         at 1).  [update_enemy_cooldowns $1317]
      2. update_enemies processes the enemy at `cursor`; advance the cursor 7->0
         wrap.  [$16D9 DEC $0090]
         For that enemy, if update_cooldown < 2 (consider_enemy_state $16E6):
           * reload update_cooldown = 4 ($16ED)
           * scan for a target (player/robot/boulder/tree) within FOV+LOS
           * if a target found: set draining_cooldown=120 the first time ($1835);
             if draining_cooldown == 1, "drain" (handled by drain_tick) and reload
             update_cooldown = 30 ($1848).
           * else (no target, not on draining cooldown): if rotation_cooldown < 2,
             rotate (h_angle += rotation_speed; rotation_cooldown = 200).
                 [rotate_enemy $1805]
    Returns a NEW EnemyPhase (the input is not mutated)."""
    ph = phase.copy()
    _decrement_cooldowns(ph)

    # pick the enemy whose turn it is; cursor decrements 7->0 wrap ($16D9).
    slot = ph.cursor
    ph.cursor = (slot - 1) if slot > 0 else 7

    e = ph.enemies.get(slot)
    ph.tick += 1
    if e is None:
        return ph  # non-enemy slot: update_enemies skips it ($16BE/$16C2)

    # consider_enemy_state $16E6: skip if update_cooldown >= 2.
    if e.update_cooldown >= COOLDOWN_STICK:
        return ph
    e.update_cooldown = UPDATE_COOLDOWN_SCAN  # $16ED reload 4

    tgt = _player_target(state, e)
    if tgt is not None:
        _tx, _ty, _top, tslot = tgt
        e.targeted_slot = tslot
        if e.draining_cooldown < DRAIN_FIRE_VALUE:
            # first time targeting -> arm the draining cooldown ($1835 reload 120)
            e.draining_cooldown = DRAINING_COOLDOWN_RELOAD
        elif e.draining_cooldown == DRAIN_FIRE_VALUE:
            # drain fires this tick; reload update cooldown ($1848 = 30). The
            # actual energy change is applied by drain_tick (so callers can decide
            # whether the *player* is the target).
            e.update_cooldown = UPDATE_COOLDOWN_DRAIN
            # after firing, the enemy re-arms next time it re-targets; clear it.
            e.draining_cooldown = 0
        return ph

    # no target: enemy is free to rotate (if not waiting on draining cooldown).
    e.targeted_slot = -1
    if e.draining_cooldown >= DRAIN_FIRE_VALUE:
        # still nominally targeting/cooling: ROM clears it (set_enemies_draining_
        # cooldown $17A9 stores 0 when target lost). Do the same.
        e.draining_cooldown = 0
    if e.rotation_cooldown < COOLDOWN_STICK:
        e.h_angle = (e.h_angle + e.rotation_speed) & 0xFF  # $180D ADC $9D37,X
        e.rotation_cooldown = ROTATION_COOLDOWN_RELOAD  # $1813 reload 200
    return ph


# ============================================================================
# drain_tick: apply one tick of draining (reduce_object_energy $1A08)
# ============================================================================
@dataclass
class DrainEvent:
    enemy_slot: int
    target_slot: int
    kind: str  # "player" | "downgrade" | "remove"
    detail: str = ""


def drain_tick(state: GameState, phase: EnemyPhase) -> Tuple[int, List[DrainEvent]]:
    """Return (energy_delta_to_player, events) for draining that fires THIS tick.

    Mirrors the moment update_enemies calls reduce_object_energy ($1A08): an enemy
    whose draining_cooldown is about to fire (== DRAIN_FIRE_VALUE) and which sees
    the player drains the player by 1 ($1A15 SBC #$1). If it sees one of the
    player's robots/boulders/trees instead, the object is downgraded
    (robot->boulder->tree, tree->removed) rather than the player losing energy
    ($1A36/$1A49/$1A3E). This is the *time-accurate* per-tick drain the planner can
    use to debit energy lost during a climb.

    NOTE: this reports what WOULD happen at the firing tick; it does not mutate
    `state`. The planner calls it after `step_enemies` to debit player energy, and
    the executor must mind the same timing live (drains land on specific ticks)."""
    events: List[DrainEvent] = []
    delta = 0
    p = state.player
    for e in phase.enemies.values():
        # the drain fires on the tick draining_cooldown == DRAIN_FIRE_VALUE while a
        # target is in view (target_object $1825 -> consider_reducing_object $183D).
        if e.draining_cooldown != DRAIN_FIRE_VALUE:
            continue
        tgt = _player_target(state, e)
        if tgt is None:
            continue
        _tx, _ty, _top, tslot = tgt
        if p is not None and tslot == p.slot:
            delta -= 1
            events.append(
                DrainEvent(e.slot, tslot, "player", "player drained 1 energy")
            )
        else:
            o = state.object_by_slot(tslot)
            if o is None:
                continue
            if o.type == T_TREE:
                events.append(DrainEvent(e.slot, tslot, "remove", "tree removed"))
            else:
                events.append(
                    DrainEvent(e.slot, tslot, "downgrade", f"{o.type_name} downgraded")
                )
    return delta, events


# ============================================================================
# Convenience: exposure over a timing window (used by the robust min-layer)
# ============================================================================
def exposed_within_window(
    state: GameState,
    phase: EnemyPhase,
    x: int,
    y: int,
    ticks: int,
    object_top: float = ROBOT_EYE,
) -> bool:
    """Worst-case exposure: True if ANY enemy sees tile (x,y) at ANY tick in the
    next `ticks` ticks (advancing the phase deterministically). This is what the
    planner's MIN layer uses to be robust to executor timing jitter -- it widens
    the instantaneous `enemy_sees` over a +/- band of phase.

    The (x,y) target is assumed static over the window (the player is deciding
    whether standing/building here is safe across timing slack)."""
    ph = phase
    for _ in range(max(1, ticks)):
        if enemies_seeing(state, ph, x, y, object_top):
            return True
        ph = step_enemies(state, ph)
    return bool(enemies_seeing(state, ph, x, y, object_top))


def ticks_until_seen(
    state: GameState,
    phase: EnemyPhase,
    x: int,
    y: int,
    horizon: int = 256,
    object_top: float = ROBOT_EYE,
) -> int:
    """How many ticks until some enemy first sees (x,y) (>= horizon if never within
    the horizon). Used as the 'safety margin' eval term. 0 == seen right now."""
    ph = phase
    for t in range(horizon):
        if enemies_seeing(state, ph, x, y, object_top):
            return t
        ph = step_enemies(state, ph)
    return horizon


# ============================================================================
# MEANIE-SPAWN MODELLING (the forced-hyperspace side-channel)
# ============================================================================
# The ROM mechanic, traced end-to-end through update_enemies $16B5:
#
#  (1) An enemy scans for a drainable robot/player in find_drainable_robot_loop
#      $17B2. For each slot it runs check_if_enemy_can_see_object $1887 (angular
#      FOV $0C68 then terrain LOS). $0014's TOP bit = "fully visible" (clear LOS
#      to the object body at the upper probe, $18FC); the $40 bit = "partially
#      visible". At $17C2 BMI target_object: a FULLY-visible player/robot is just
#      DRAINED, no meanie. The meanie path is the PARTIAL case:
#  (2) $17C4 CPY player_object / $17C8 STY $000F: if the player was SEEN (in FOV)
#      but NOT fully visible, the enemy remembers the player slot in $000F. After
#      the loop, $17CD LDY $000F; if it is the player (BPL) and it differs from
#      enemies_failed_meanie_memory ($17D2), the enemy calls
#      initialise_enemy_meanie_variables $1973 (arming the meanie search,
#      $1980 enemies_meanie_search_object=$40) and sets $0014=$40 ("partially
#      visible") then target_object $1825 -> targets the player.
#      ==> CONDITION (A): the enemy SEES the player but NOT the player's base
#          square (partial visibility), so it cannot drain directly and instead
#          looks to spawn a meanie. ($17B2/$17C2/$17C8/$17D7)
#  (3) On the draining tick, consider_creating_meanie $1986 runs. It widens the
#      FOV to $28 (40 units = two screen widths, $1986) and scans every TREE
#      (type 2, $19AD) slot. For a candidate tree Y and the targeted player X:
#        * |objects_x[player] - objects_x[tree]| < $0A (10 tiles), $19B5-$19C5
#        * |objects_y[player] - objects_y[tree]| < $0A (10 tiles), $19C7-$19D7
#          ==> CONDITION (B): the tree is within 10 tiles of the player in BOTH
#              x and y.
#        * check_if_enemy_can_see_object $1887 on the tree with $0014 top bit set
#          ($19DB/$19DE/$19E0): the enemy has a CLEAR LOS to the tree's square.
#          ==> CONDITION (C): the enemy can see the object's base square.
#      If all hold, the tree's type is overwritten to 4 (meanie, $19F0) and the
#      enemy records it ($19EB enemies_meanie_object).
#  (4) The meanie then ROTATES toward the player each round (meanie_not_looking_
#      at_player $1728: $172A BIT object_relative_h_angle_high decides +8/-8 step
#      $1741 ADC $0C0E) and, when it acquires the player ($170B
#      check_if_enemy_can_see_object on the player from the meanie + $1719 LDA
#      $0014 nonzero = meanie can see the player), $171D JSR do_hyperspace forces
#      a teleport.
#      ==> CONDITION (D): the object's position must have LOS to the player's
#          square (otherwise the meanie can never acquire the player to fire the
#          hyperspace; $170B-$171D).
#
# `meanie_spawn_threat` ports conditions (A)-(D) geometrically. Because our LOS
# model drops the instantaneous rotation phase (game_model.seen_by_enemy is the
# conservative "ever sees if rotated toward"), we ask the *capability* question
# the planner needs: "could THIS enemy spawn a meanie from THIS object that could
# then acquire the player at `player_tile`?" -- which is exactly what makes a base
# or dwell tile risky.
MEANIE_RANGE = 0x0A  # $19C3/$19D5: |dx|<10 and |dy|<10 tiles
T_MEANIE_CONVERTIBLE = (T_TREE,)  # $19AD CMP #$2: only trees become meanies


def _enemy_sees_partial_not_square(
    state: GameState, enemy: GameObject, player_tile: Tuple[int, int]
) -> bool:
    """CONDITION (A): the enemy sees the player but NOT the player's base square.

    In the ROM, find_drainable_robot_loop $17B2 routes the player to the meanie
    path only when check_if_enemy_can_see_object reports the player as PARTIALLY
    visible -- in FOV with the upper probe blocked so the $0014 top bit (full
    visibility, $18FC) is clear (otherwise it would just drain, $17C2). We model
    that conservatively with game_model.can_see's two probes: the enemy can see
    the player BODY (ROBOT_EYE top) but the looking-DOWN sight to the player's bare
    BASE square is blocked (the enemy is not strictly above it). That is the
    geometric face of "sees the player but not the player's square"."""
    px, py = player_tile
    surf = tile_surface_height(state, float(px), float(py))
    sees_body = can_see(
        state,
        (enemy.x, enemy.y),
        (px, py),
        eye_offset=ENEMY_EYE,
        target_height=surf + ROBOT_EYE,
    )
    if not sees_body:
        return False
    # "the player's base square" -- LOS to the bare tile surface, looking down
    # (game_model.can_see's looking-up rejection makes this False when the enemy
    # eye is not above the square; that is precisely the partial-visibility case).
    sees_square = can_see(
        state, (enemy.x, enemy.y), (px, py), eye_offset=ENEMY_EYE, target_height=surf
    )
    return not sees_square


def meanie_spawn_threat(
    state: GameState, player_tile: Tuple[int, int]
) -> List[Tuple[GameObject, Tuple[int, int]]]:
    """Objects an enemy could convert into a meanie that could then acquire the
    player standing at `player_tile`. Returns a list of (enemy_object, (ox,oy))
    for each (enemy, convertible object) pair satisfying the ROM predicate.

    Ported conditions (addresses in the module header):
      (A) the enemy sees the player but NOT the player's base square
          (_enemy_sees_partial_not_square; find_drainable_robot_loop $17B2 ->
          partial-visibility meanie path $17C2/$17C8/$17D7).
      (B) the object is a TREE within 10 tiles of the player in BOTH x and y
          (consider_creating_meanie $19AD type==2; $19C3/$19D5 |d|<10).
      (C) the enemy has a clear LOS to the object's base square
          ($19DB check_if_enemy_can_see_object, $0014 top bit).
      (D) the object's position has LOS to the player's square, so the resulting
          meanie can rotate to and acquire the player ($170B-$171D do_hyperspace).

    An EMPTY list means standing at `player_tile` carries no meanie-spawn risk."""
    px, py = player_tile
    threats: List[Tuple[GameObject, Tuple[int, int]]] = []
    enemies = _enemies(state)
    if not enemies:
        return threats
    convertible = [o for o in state.objects if o.type in T_MEANIE_CONVERTIBLE]
    if not convertible:
        return threats
    surf_p = tile_surface_height(state, float(px), float(py))
    for e in enemies:
        # (A) enemy sees the player partially (sees player, not their square).
        if not _enemy_sees_partial_not_square(state, e, (px, py)):
            continue
        for o in convertible:
            # (B) within 10 tiles in both axes.
            if abs(o.x - px) >= MEANIE_RANGE or abs(o.y - py) >= MEANIE_RANGE:
                continue
            # (C) enemy has clear LOS to the object's base square ($19DB).
            o_surf = tile_surface_height(state, float(o.x), float(o.y))
            if not can_see(
                state,
                (e.x, e.y),
                (o.x, o.y),
                eye_offset=ENEMY_EYE,
                target_height=o_surf + ROBOT_EYE,
            ):
                continue
            # (D) the (future) meanie at the object's tile can see the player, so it
            # can rotate to and acquire them -> forced hyperspace ($170B-$171D).
            if not can_see(
                state,
                (o.x, o.y),
                (px, py),
                eye_offset=ENEMY_EYE,
                target_height=surf_p + ROBOT_EYE,
            ):
                continue
            threats.append((e, (o.x, o.y)))
    return threats


def meanie_safe(state: GameState, player_tile: Tuple[int, int]) -> bool:
    """True if standing at `player_tile` carries NO meanie-spawn risk (the
    convenience boolean the closure solver's safety filter consults)."""
    return not meanie_spawn_threat(state, player_tile)


# ============================================================================
# self-test / cross-check against Py65Source.from_landscape(0)
# ============================================================================
def _selftest():
    import game_state as gs
    from game_state import read_game_state

    print("== enemy_dynamics cross-check vs generated RAM ==")
    for ls in (0, 42, 9999):
        src = gs.Py65Source.from_landscape(ls)
        mem = src.mem
        state = read_game_state(src)
        phase = init_phase_from_ram(state, mem)
        print(f"\n-- seed {ls}: {len(phase.enemies)} enemies --")
        for slot, e in sorted(phase.enemies.items()):
            spd = (
                "+20(cw)"
                if e.rotation_speed == ROTATION_STEP_CW
                else (
                    "-20(ccw)"
                    if e.rotation_speed == ROTATION_STEP_CCW
                    else f"raw{e.rotation_speed}"
                )
            )
            print(
                f"   slot {slot} {gm.TYPES[e.type]:<8} "
                f"h_angle={e.h_angle:3d} ({e.h_angle*DEG_PER_UNIT:6.1f} deg) "
                f"rot={spd} upd_cd={e.update_cooldown} rot_cd={e.rotation_cooldown} "
                f"drain_cd={e.draining_cooldown}"
            )
        # advance a handful of ticks, show how the Sentinel's facing sweeps.
        sent = next((e for e in phase.enemies.values() if e.type == T_SENTINEL), None)
        if sent:
            _angs = []
            ph = phase
            for _ in range(2000):
                ph = step_enemies(state, ph)
            sent2 = ph.enemies[sent.slot]
            print(
                f"   Sentinel facing after 2000 ticks: "
                f"{sent2.h_angle} ({sent2.h_angle*DEG_PER_UNIT:.1f} deg)  "
                f"[was {sent.h_angle}]"
            )
        # who sees the player's start tile right now?
        p = state.player
        seers = enemies_seeing(state, phase, p.x, p.y)
        margin = ticks_until_seen(state, phase, p.x, p.y, horizon=512)
        print(
            f"   player start ({p.x},{p.y}): seen-now by {seers}; "
            f"first-seen in {margin} ticks (512 horizon)"
        )


if __name__ == "__main__":
    _selftest()
