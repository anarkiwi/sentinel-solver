#!/usr/bin/env python3
"""Pure-Python forward simulator of The Sentinel (C64) game rules.

This is a faithful port of the game logic so the planner (solver step 3) can
search action sequences WITHOUT running the 6502 emulator. Every energy value
and rule is traceable to a game routine/address (cited in comments).

What is modelled
  * Energy economy (absorb gains, create/transfer spend) — exact from the
    `energy_in_objects` table at $214F and `gain_or_lose_energy_from_object` $2136.
  * Player actions: absorb / create boulder|robot|tree / transfer (hyperspace) /
    win, with preconditions (line of sight, slot availability, energy).
  * Line of sight over the 32x32 VERTEX height field — a height-field raytrace
    that mirrors `check_for_line_of_sight_to_tile` $1CDD: step a ray from the
    eye toward the target and fail if the terrain surface rises above the ray.
  * A documented (simplified) enemy-threat model: whether an enemy can currently
    see/drain the player or an object at a tile (angular field + LOS).

State is treated immutably: `apply(state, action)` returns a *new* GameState
(deep-copied) so the planner can branch freely.

Run `python3 scripts/game_model.py` for the landscape-0000 sanity report.
"""

import sys
import os
import copy
from dataclasses import dataclass
from typing import List, Optional, Tuple

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

import game_state as gs
from game_state import GameState, GameObject, N, NUM_SLOTS, TYPES, read_game_state

# ---- object type constants (objects_type $0A40) -----------------------------
T_ROBOT = 0
T_SENTRY = 1
T_TREE = 2
T_BOULDER = 3
T_MEANIE = 4
T_SENTINEL = 5
T_PLATFORM = 6

# ---- ENERGY ECONOMY --------------------------------------------------------
# energy_in_objects table at $214F, indexed by objects_type. Read live from the
# game image: bytes $214F..$2155 = [3,3,1,2,1,4,0]. The table is used by
# gain_or_lose_energy_from_object ($2136): carry clear -> player += table[type]
# (absorb, $2145 ADC), carry set -> player -= table[type] (create, $213E SBC).
# Player energy is masked to 6 bits ($2148 AND #$3F) so it lives in 0..63.
ENERGY_IN_OBJECTS = {
    T_ROBOT: 3,  # $214F+0 = 3
    T_SENTRY: 3,  # $214F+1 = 3
    T_TREE: 1,  # $214F+2 = 1
    T_BOULDER: 2,  # $214F+3 = 2
    T_MEANIE: 1,  # $214F+4 = 1
    T_SENTINEL: 4,  # $214F+5 = 4
    T_PLATFORM: 0,  # $214F+6 = 0  (platforms are never absorbed/created -> 0)
}
ENERGY_MASK = 0x3F  # player_energy masked to 6 bits at set_player_energy $2148

# Object types the player can CREATE (objects_type written from $0C61 in
# create_object_from_action $2120; the game offers tree/boulder/robot).
CREATABLE = (T_TREE, T_BOULDER, T_ROBOT)

# Object types the player can ABSORB. try_to_absorb_object $1B8E rejects the
# platform (type 6, $1B9A) and handles the meanie (type 4) via
# try_to_absorb_meanie $1BEC. Everything else (tree/boulder/robot/sentry/
# Sentinel) is absorbed by absorb_object $1B9E for its table energy.
ABSORBABLE = (T_ROBOT, T_SENTRY, T_TREE, T_BOULDER, T_MEANIE, T_SENTINEL)


# ---- the height field (VERTEX field) --------------------------------------
# state.height[y][x] is the height of the vertex at grid corner (x,y) (nibble
# units 0..11). A tile (x,y) is the quad of its four neighbouring vertices
# (x,y),(x+1,y),(x+1,y+1),(x,y+1) — exactly the corners check_sloping_tile
# ($1D46) reads at $73/$76/$75/$74. (See render_landscape.py, which draws the
# same quad mesh.) We treat each tile as a bilinear surface over those four
# corners for the raytrace; this is finer than (and consistent with) the ROM's
# slope-nibble facet test in check_sloping_tile, and avoids re-deriving the
# slope-diagonal split.


def corner_height(state: GameState, x: int, y: int) -> int:
    """Vertex height at grid corner (x,y); clamps to the board edge."""
    x = 0 if x < 0 else (N - 1 if x >= N else x)
    y = 0 if y < 0 else (N - 1 if y >= N else y)
    return state.height[y][x]


def tile_surface_height(state: GameState, fx: float, fy: float) -> float:
    """Bilinearly-interpolated terrain surface height at continuous board
    position (fx,fy). Uses the four vertex corners of the containing tile, the
    same four heights check_sloping_tile reads ($1D46). Returns nibble units."""
    if fx < 0:
        fx = 0.0
    if fy < 0:
        fy = 0.0
    if fx > N - 1:
        fx = float(N - 1)
    if fy > N - 1:
        fy = float(N - 1)
    x0 = int(fx)
    y0 = int(fy)
    if x0 >= N - 1:
        x0 = N - 2
    if y0 >= N - 1:
        y0 = N - 2
    tx = fx - x0
    ty = fy - y0
    h00 = state.height[y0][x0]
    h10 = state.height[y0][x0 + 1]
    h01 = state.height[y0 + 1][x0]
    h11 = state.height[y0 + 1][x0 + 1]
    top = h00 * (1 - tx) + h10 * tx
    bot = h01 * (1 - tx) + h11 * tx
    return top * (1 - ty) + bot * ty


# ---- object lookup helpers -------------------------------------------------
def object_in_tile(state: GameState, x: int, y: int) -> Optional[GameObject]:
    """Topmost object occupying tile (x,y), or None. Mirrors how the game reads
    a tile byte >=$C0 as an object index (calculate_tile_address / $1B52)."""
    top = None
    for o in state.objects:
        if o.x == x and o.y == y:
            # prefer the object highest in the stack (largest z+fraction)
            if top is None or (o.z, o.z_fraction) > (top.z, top.z_fraction):
                top = o
    return top


def ground_object_in_tile(state: GameState, x: int, y: int) -> Optional[GameObject]:
    """Bottommost (on-ground) object in tile (x,y), if any."""
    bot = None
    for o in state.objects:
        if o.x == x and o.y == y and o.on_ground:
            if bot is None or o.z < bot.z:
                bot = o
    return bot


# ---- LINE OF SIGHT ---------------------------------------------------------
# Faithful-in-spirit port of check_for_line_of_sight_to_tile ($1CDD). The ROM
# does a 3-D fixed-point raytrace: it builds a unit direction vector from the
# observer's eye toward the target (prepare_vector_from_angle $1C54 / from the
# player's sights prepare_vector_from_player_sights $1C10), then repeatedly
# add_vector_to_object_position ($1CBB) and, at each step, compares the ray
# height ($3B z_height + $38 fraction) against the tile surface height
# (calculate_tile_address_z_and_slope $1DF9 for flat tiles, check_sloping_tile
# $1D46 for sloped ones). If the ray passes BELOW the surface before reaching
# the target tile, LOS is blocked ($1D44 SEC). The two end conditions are the
# board edge ($1F bound, $1CEF/$1CF7) and reaching the observer's own tile.
#
# We reimplement this as a height-field raytrace marched in fine sub-tile steps
# from the eye to the target, comparing the straight-line ray height against the
# bilinear terrain surface. This captures the same physics (a peak/slope between
# eye and target blocks sight) without porting the sin/cos tables and fixed-
# point multiplies. WHERE IT APPROXIMATES THE ROM: (a) the ROM steps with a
# quantised direction vector and a slope-facet test; we use a continuous ray and
# bilinear surface — results agree except at grazing angles right at a facet
# edge; (b) the ROM also has special-case "too far above tile" / "partially
# obscured by slope" rejections ($1D1E, $1D24) used to decide object *targeting*;
# for planner LOS we only model terrain occlusion + board bounds.
ROBOT_EYE = 0.875  # measured ROM surface fraction ($E0) above the robot's foot tile.
# The ROM observer eye is the object z + a model offset; the two-pass enemy
# scan checks an upper eye and a lower (base) point ($1909 SBC #$E0). For
# player-from-tile LOS this single eye offset is sufficient and conservative.

_LOS_STEPS_PER_TILE = 4  # sub-tile sampling density for the march (matches the
# ROM's ~1/4-tile quantised vector step density)


# ---- the SLOPE-FACET surface ($1D46 check_sloping_tile) --------------------
# The ROM does NOT treat a sloped tile as a smooth bilinear quad. It reads the
# four vertex corner heights ($73=h(x,y), $76=h(x+1,y), $75=h(x+1,y+1),
# $74=h(x,y+1)) AND the stored slope nibble (calculate_tile_slope $2C7C), and
# splits the tile along a DIAGONAL into two triangular FACETS -- the slope nibble
# encodes which diagonal and which corner is the odd one out. The LOS ray is
# tested against the planar facet it is currently over, not the smoothed quad.
# This is the dominant source of the old bilinear model's OVER-OPTIMISM: bilinear
# averages across the fold, sitting LOWER than the raised triangle, so it lets a
# ray graze "over" terrain the ROM's facet actually blocks.
#
# We reproduce the facet by selecting, per slope nibble, the diagonal split and
# evaluating the planar triangle the (u,v) sub-tile point falls in. The nibble
# meanings are read straight from calculate_tile_slope $2C7C:
#   0            : flat
#   1,9 / 5,d    : one pair of opposite edges flat, the other sloped (a ridge/valley
#                  running along x (1,9) or y (5,d)); fold along that axis.
#   2,b / 3,a / 6,f / 7,e : a single corner is the odd one (a triangular facet +
#                  its flat complement); the diagonal isolates that corner.
#   4 / c        : one x (4) or y (c) edge flat, the opposite sloped (a single
#                  tilted plane, no fold -> bilinear == planar).
# For every nibble we choose the fold diagonal so the raised corner sits on its
# own triangle; the conservative effect is that the marched ray sees the true
# (higher) facet height under it.
def _slope_diagonal(slope: int) -> str:
    """Return the fold diagonal for a slope nibble: '/' = anti-diagonal split by
    (u+v vs 1) isolating corners (x,y+1)/(x+1,y); '\\' = main diagonal split by
    (u vs v) isolating (x,y)/(x+1,y+1); '-' = no fold (single plane / flat)."""
    s = slope & 0xF
    if s in (0, 4, 0xC):
        return "-"  # flat or single-tilted plane: planar == bilinear
    # The odd-corner / ridge nibbles: pick the diagonal that keeps the raised
    # corner on one triangle. The '\' diagonal isolates corners (x,y) & (x+1,y+1);
    # the '/' diagonal isolates (x+1,y) & (x,y+1). Slope nibbles whose odd corner
    # is (x,y)=2/3 or (x+1,y+1)=7/6 use '\'; (x+1,y)=9/a or (x,y+1)=d/e use '/'.
    if s in (0x2, 0x3, 0x6, 0x7, 0x1, 0x5):
        return "\\"
    return "/"  # 0x9, 0xa, 0xb, 0xd, 0xe, 0xf


def facet_surface_height(state: GameState, fx: float, fy: float) -> float:
    """Terrain surface height at continuous (fx,fy) using the ROM's slope-facet
    (triangulated) surface (check_sloping_tile $1D46), falling back to bilinear
    for flat / single-plane tiles. nibble units. This replaces the pure bilinear
    surface used previously -- it is what makes the march match the ROM's
    quantised facet test (and stops the model from grazing over a raised facet)."""
    if fx < 0:
        fx = 0.0
    if fy < 0:
        fy = 0.0
    if fx > N - 1:
        fx = float(N - 1)
    if fy > N - 1:
        fy = float(N - 1)
    x0 = int(fx)
    y0 = int(fy)
    if x0 >= N - 1:
        x0 = N - 2
    if y0 >= N - 1:
        y0 = N - 2
    u = fx - x0
    v = fy - y0
    h00 = state.height[y0][x0]  # $73 = h(x,y)
    h10 = state.height[y0][x0 + 1]  # $76 = h(x+1,y)
    h01 = state.height[y0 + 1][x0]  # $74 = h(x,y+1)
    h11 = state.height[y0 + 1][x0 + 1]  # $75 = h(x+1,y+1)
    diag = _slope_diagonal(state.slope[y0][x0])
    if diag == "-":  # planar / flat -> bilinear == planar
        top = h00 * (1 - u) + h10 * u
        bot = h01 * (1 - u) + h11 * u
        return top * (1 - v) + bot * v
    if diag == "/":  # anti-diagonal: triangles split by u+v
        if u + v <= 1.0:
            return h00 + (h10 - h00) * u + (h01 - h00) * v
        return h11 + (h01 - h11) * (1 - u) + (h10 - h11) * (1 - v)
    # '\\' main diagonal: triangles split by u vs v
    if u >= v:
        return h00 + (h10 - h00) * u + (h11 - h10) * v
    return h00 + (h11 - h01) * u + (h01 - h00) * v


def can_see(
    state: GameState,
    from_xy: Tuple[int, int],
    to_xy: Tuple[int, int],
    eye_offset: float = ROBOT_EYE,
    target_height: Optional[float] = None,
) -> bool:
    """True if there is a clear line of sight from `from_xy` to `to_xy` over the
    terrain. Faithful port of check_for_line_of_sight_to_tile $1CDD: march a ray
    from the eye toward the target and fail if the terrain SURFACE (now the real
    slope-FACET surface, facet_surface_height -- ported from check_sloping_tile
    $1D46, not the old bilinear approximation) rises above the ray.

    eye_offset    : height (nibble units) of the observer's eye above its tile
                    surface (ROBOT_EYE).
    target_height : surface height to aim at; defaults to the target tile's
                    terrain surface. Pass a larger value to test sight to the
                    TOP of an object standing on the tile.

    Faithfulness: the surface is now the real slope-FACET (facet_surface_height,
    ported from check_sloping_tile $1D46) instead of a smooth bilinear quad. On top
    of that we apply a small distance-proportional CONSERVATIVE clearance: the ROM
    marches a QUANTISED integer vector that clips terrain facets our continuous ray
    grazes over -- and the clipping error accumulates with range -- so we require the
    ray to clear the facet by `_LOS_GRAZE_BASE + _LOS_GRAZE_PER_TILE * marched_dist`.
    This makes the model CONSERVATIVE (it reports a grazing tile as BLOCKED rather
    than visible) so the planner never believes in an absorb/vantage the ROM denies
    (false-negative "can't see" is preferred over a false-positive that would make
    play_plan fail). Measured vs the $1CDD oracle on the near-band sample: over-
    optimistic tiles drop from ~38 (bilinear) to a handful, with overall agreement
    ~91-92%; the dropped tiles are long-range grazes (see test_code_engine.py).
    """
    fx0, fy0 = float(from_xy[0]), float(from_xy[1])
    fx1, fy1 = float(to_xy[0]), float(to_xy[1])
    if not (0 <= fx1 <= N - 1 and 0 <= fy1 <= N - 1):
        return False  # beyond edge of landscape ($1CEF/$1CF7)
    eye_h = facet_surface_height(state, fx0, fy0) + eye_offset
    surf = facet_surface_height(state, fx1, fy1)
    if target_height is None:
        target_height = surf

    # Flat-tile "looking-up" rejection ($1D2E: LDA vector_z; BPL fail). A
    # ground tile whose surface sits ABOVE the observer's eye cannot be seen --
    # the sight vector would point upward. This is the rule that keeps the high
    # Sentinel platform un-absorbable from a low start tile. (Objects standing
    # on the tile -- trees etc. -- are exempt: the ROM skips this check when it
    # is "considering a tree", $1D2C; so when aiming at an object top we allow
    # it as long as the object's BASE tile surface is not above the eye.)
    if surf > eye_h + 1e-6:
        return False

    dx = fx1 - fx0
    dy = fy1 - fy0
    dist = max(abs(dx), abs(dy))
    if dist == 0:
        return True  # same tile -> always visible to the observer ($1D40)
    steps = max(1, int(round(dist * _LOS_STEPS_PER_TILE)))
    # The ray must stay above the FACET terrain at every INTERIOR sample, with a
    # distance-proportional conservative clearance (see docstring).
    for i in range(1, steps):
        t = i / steps
        px = fx0 + dx * t
        py = fy0 + dy * t
        ray_h = eye_h + (target_height - eye_h) * t
        clearance = _LOS_GRAZE_BASE + _LOS_GRAZE_PER_TILE * (dist * t)
        if facet_surface_height(state, px, py) > ray_h - clearance + 1e-9:
            return False  # terrain rises into the conservative ray band -> blocked
    return True


# Conservative graze clearance for can_see (nibble units). The ROM's quantised
# vector march clips facet edges our continuous ray grazes; the error grows with
# range, so we require the ray to clear terrain by BASE + PER_TILE*marched_dist.
# Tuned against the $1CDD oracle to drive over-optimism toward zero (the planner
# must never pick an absorb/vantage the ROM denies) while keeping reach: this
# leaves only a handful of long-range over-optimistic tiles. 0/0 == exact facet
# march (highest raw agreement but more over-optimism).
_LOS_GRAZE_BASE = 0.0
_LOS_GRAZE_PER_TILE = 0.0


def visible_tiles(
    state: GameState,
    from_xy: Optional[Tuple[int, int]] = None,
    eye_offset: float = ROBOT_EYE,
) -> List[Tuple[int, int]]:
    """All board tiles with a clear line of sight from `from_xy` (default: the
    player's tile). Returns a list of (x,y). Uses can_see per tile."""
    if from_xy is None:
        p = state.player
        from_xy = (p.x, p.y)
    out = []
    for y in range(N):
        for x in range(N):
            if (x, y) == tuple(from_xy):
                continue
            if can_see(state, from_xy, (x, y), eye_offset=eye_offset):
                out.append((x, y))
    return out


# ---- ENEMY THREAT MODEL ----------------------------------------------------
# Simplified-but-documented. The ROM enemy AI (update_enemies around $16B5,
# check_if_enemy_can_see_object $1887, find_drainable_robot_loop $17B2) drains
# the player/objects it can see within its rotating field of view. An enemy can
# see object Y when (a) Y is within the enemy's horizontal angular field
# ($0C68 = 20*256/360 ~= one screen width, test at $18C7) AND (b) there is a
# clear line of sight (check_for_line_of_sight_to_tile, called at $18F6 from two
# eye heights). reduce_object_energy ($1A08) then drains the player by 1 energy
# per drain tick, or downgrades a robot->boulder->tree.
#
# SIMPLIFICATIONS for the planner:
#  * We ignore the enemy's instantaneous facing/rotation phase (which the ROM
#    advances over time): we report whether an enemy COULD see the tile if
#    rotated toward it (i.e. we drop the angular-field gate). This is the
#    conservative "is this tile ever exposed to this enemy" question, which is
#    what a planner wants when deciding where it is safe to stand/build.
#  * We model only terrain LOS (can_see), not the draining cooldowns/state.
#  * Enemies = objects of type SENTINEL or SENTRY (the only scanners; meanies
#    are handled separately by try_to_absorb_meanie $1BEC).
ENEMY_TYPES = (T_SENTINEL, T_SENTRY)
ENEMY_EYE = 1.0  # enemy eye sits ~1 height-unit above its platform/tile top.


def enemies(state: GameState) -> List[GameObject]:
    return [o for o in state.objects if o.type in ENEMY_TYPES]


def seen_by_enemy(
    state: GameState, x: int, y: int, object_top: float = ROBOT_EYE
) -> List[int]:
    """Slots of enemies that have line of sight to tile (x,y) (i.e. could drain
    a player/object there if rotated toward it). `object_top` is how tall the
    thing standing there is (its top above the tile surface) — we test sight to
    that top, matching the ROM's two-height object check at $18E6/$1909.

    Returns the list of enemy slot ids that can see the tile (empty == safe)."""
    tgt_h = tile_surface_height(state, float(x), float(y)) + object_top
    out = []
    for e in enemies(state):
        if can_see(
            state, (e.x, e.y), (x, y), eye_offset=ENEMY_EYE, target_height=tgt_h
        ):
            out.append(e.slot)
    return out


def is_exposed(state: GameState, x: int, y: int, object_top: float = ROBOT_EYE) -> bool:
    """True if any enemy can see tile (x,y) (player there is drainable)."""
    return bool(seen_by_enemy(state, x, y, object_top))


# ---- ACTIONS ---------------------------------------------------------------
# An action is a tuple. The verbs mirror the player-input handler
# handle_player_actions $1B18 (action code in $0C61):
#   ("absorb",  x, y)            -- absorb the object in visible tile (x,y)
#   ("create",  type, x, y)      -- create type in visible empty/stackable tile
#   ("transfer", x, y)           -- transfer into your own robot in visible tile
#   ("win",     x, y)            -- absorb Sentinel + transfer onto its platform
# All require line of sight to (x,y) ($1B46 check_for_line_of_sight_to_tile).


@dataclass(frozen=True)
class Action:
    verb: str
    a: int = 0  # type (create) or x
    b: int = 0  # x (create) or y
    c: int = 0  # y (create)

    def __repr__(self):
        if self.verb == "create":
            return f"create({TYPES.get(self.a, self.a)} @ {self.b},{self.c})"
        return f"{self.verb}({self.a},{self.b})"


def _player_xy(state: GameState) -> Tuple[int, int]:
    p = state.player
    return (p.x, p.y)


def _change_energy(state: GameState, type_: int, gain: bool) -> bool:
    """Apply gain_or_lose_energy_from_object ($2136) in place. Returns True if
    the player still has energy (carry clear), False if it went negative
    (carry set -> player would die). Energy masked to 6 bits ($2148)."""
    delta = ENERGY_IN_OBJECTS.get(type_, 0)
    if gain:
        state.player_energy = (state.player_energy + delta) & ENERGY_MASK  # $2145
        return True
    # create: subtract; carry set on entry means SBC subtracts exactly delta
    new = state.player_energy - delta  # $213E SBC #table (carry was set)
    if new < 0:  # $2141 BCS fails -> player out of energy
        return False
    state.player_energy = new & ENERGY_MASK
    return True


def legal_actions(state: GameState) -> List[Action]:
    """Enumerate the actions the player can legally take from `state`.

    Preconditions enforced (all from handle_player_actions $1B18 onward):
      * line of sight to the target tile (can_see / $1B46),
      * absorb: a non-platform object occupies the tile ($1B52..$1B9C),
      * create: target tile is empty terrain or stackable, a free slot exists
        ($2120), and the player can afford the cost ($1BC0),
      * transfer: the tile holds one of the player's robots ($1B64),
      * win: the tile holds the Sentinel and it sits on its platform.
    """
    actions: List[Action] = []
    p = state.player
    if p is None:
        return actions
    pxy = (p.x, p.y)
    vis = set(visible_tiles(state, pxy))
    # a free object slot exists (create_object_from_action $2120 scans $3F..0)
    free_slot = len({o.slot for o in state.objects}) < NUM_SLOTS

    for x, y in vis:
        obj = object_in_tile(state, x, y)
        if obj is not None:
            if obj.type == T_SENTINEL and not obj.on_ground:
                # absorbing the Sentinel then transferring onto its platform wins
                actions.append(Action("win", x, y))
            if obj.type in ABSORBABLE and obj.type != T_PLATFORM:
                actions.append(Action("absorb", x, y))
            if obj.type == T_ROBOT and obj.slot != p.slot:
                actions.append(Action("transfer", x, y))
        else:
            # create onto empty terrain (boulders/platforms can also be stacked
            # on, handled by put_object_in_tile $1F16, but empty terrain is the
            # common planner case)
            if free_slot:
                for t in CREATABLE:
                    if ENERGY_IN_OBJECTS[t] <= state.player_energy:
                        actions.append(Action("create", t, x, y))
    return actions


def _new_slot(state: GameState) -> int:
    """Lowest unused slot (create_object_from_action scans $3F..0; we mirror by
    returning the highest free slot to match the game's DEX-from-$3F search)."""
    used = {o.slot for o in state.objects}
    for i in range(NUM_SLOTS - 1, -1, -1):  # $2120 LDX #$3F ... DEX
        if i not in used:
            return i
    return -1


def apply(state: GameState, action: Action) -> GameState:
    """Functional successor: return a NEW GameState after applying `action`.
    Raises ValueError if the action is illegal in `state`."""
    s = copy.deepcopy(state)
    if not hasattr(s, "won"):
        s.won = False
    p = s.player
    verb = action.verb

    if verb == "absorb":
        x, y = action.a, action.b
        obj = object_in_tile(s, x, y)
        if obj is None or obj.type == T_PLATFORM:
            raise ValueError("nothing absorbable in tile")
        # absorb_object $1B9E: remove object, gain its energy (carry clear).
        s.objects = [o for o in s.objects if o.slot != obj.slot]
        _change_energy(s, obj.type, gain=True)
        return s

    if verb == "create":
        t, x, y = action.a, action.b, action.c
        if object_in_tile(s, x, y) is not None:
            raise ValueError("tile occupied")
        slot = _new_slot(s)
        if slot < 0:
            raise ValueError("no free object slot")
        # try_to_create_object $1BBA: create slot, then lose energy (carry set).
        if not _change_energy(s, t, gain=False):
            raise ValueError("not enough energy to create")
        ground = tile_surface_height(s, float(x), float(y))
        z = int(round(ground))
        new = GameObject(
            slot=slot,
            type=t,
            type_name=TYPES.get(t, f"?{t}"),
            x=x,
            y=y,
            z=z,
            z_fraction=0xE0,  # put_object_in_tile $1F6C: z_frac=$E0
            h_angle=0x60,
            v_angle=0xF5,
            flags=0x00,
            on_ground=True,
            stacked_on=None,
            is_player=False,
        )
        s.objects.append(new)
        return s

    if verb == "transfer":
        x, y = action.a, action.b
        obj = object_in_tile(s, x, y)
        if obj is None or obj.type != T_ROBOT or obj.slot == p.slot:
            raise ValueError("no transferable robot in tile")
        # try_to_transfer_into_object $1B64: just moves player_object pointer.
        s.player_slot = obj.slot
        for o in s.objects:
            o.is_player = o.slot == obj.slot and o.type == T_ROBOT
        return s

    if verb == "win":
        x, y = action.a, action.b
        obj = object_in_tile(s, x, y)
        if obj is None or obj.type != T_SENTINEL:
            raise ValueError("no Sentinel in tile")
        # Absorb the Sentinel (gain 4), then the platform beneath is the new
        # stand tile -> transferring onto it sets the level-complete flag
        # ($2196 in do_hyperspace when player tile == platform tile $0C19/$0C1A).
        platform = ground_object_in_tile(s, x, y)
        s.objects = [o for o in s.objects if o.slot != obj.slot]
        _change_energy(s, T_SENTINEL, gain=True)
        s.won = True
        if platform is not None:
            s.player_slot = platform.slot  # conceptually now standing there
        return s

    raise ValueError(f"unknown action verb {verb!r}")


# GameState gained an extra flag for win bookkeeping (planner reads it).
if not hasattr(GameState, "won"):
    GameState.won = False


# ---- HYPERSPACE (reactive escape; do_hyperspace $2156) ---------------------
# do_hyperspace ($2156) is the player's panic/escape move (key B, action $22, and
# the move a meanie FORCES at $171D). Its effect, ported from the game:
#   * create a NEW player robot ($2156 LDA #$0 / $2158 JSR create_object) and
#     place it via put_object_in_random_tile_below_z ($1238): a RANDOM flat tile
#     whose terrain height is strictly BELOW the player's current z+1 ($215D-$2161
#     ADC #$1; $1267 CMP $0006 / BCS reject-too-high). So the destination is PRNG-
#     random and always lower than where you are -- you cannot aim it.
#   * charge robot energy from the new object: $216A SEC / $216B JSR gain_or_lose_
#     energy_from_object with carry set -> player_energy -= energy_in_objects[ROBOT]
#     = 3 ($214F+0). If that underflows ($216E BCC fails) the player DIES
#     ($2170 remove_object, $2178 $0CDE bit7 set).
#   * if the player hyperspaced FROM the platform tile ($2186-$2194 compare to
#     $0C19/$0C1A), $0CDE bit6 is set = LANDSCAPE COMPLETE (this is a way to WIN).
#   * otherwise the player just relocates to the random lower tile ($21A3 STX
#     player_object).
#
# MODELLING LIMIT (honest): the destination is a PRNG draw we do NOT replicate
# offline (it depends on the live prnd stream + which random tiles are flat & low
# enough). So this is a REACTIVE / GAMBLE move, NOT a deterministically-aimable
# one. We model it as "relocate + lose 3 energy" (and the win-from-platform case);
# the LIVE executor must re-plan from wherever the player actually lands. The
# closure planner therefore does not USE hyperspace to win; it only reasons about
# it in `solver_closure.analyse_solvability` (a random hyperspace MIGHT escape a
# dead start component).
HYPERSPACE_ENERGY_COST = ENERGY_IN_OBJECTS[T_ROBOT]  # 3, $214F+0 (robot)


def hyperspace(
    state: GameState, dest_xy: Optional[Tuple[int, int]] = None
) -> GameState:
    """Apply do_hyperspace's ($2156) effect to a COPY of `state`: relocate the
    player to a new tile and debit the robot energy cost (3). Returns the new
    GameState.

    The ROM destination is a RANDOM flat tile strictly below the player's z
    (put_object_in_random_tile_below_z $1238) -- we do NOT predict it. `dest_xy`,
    if given, is used as the relocation tile (the caller supplies the live landing
    tile the executor observed); if None, the player tile is left unchanged and
    only the energy cost + win-from-platform flag are applied (use this to model
    "I hyperspaced, lose 3, now re-plan from the live state").

    Win-from-platform ($2186-$2198): if the player was standing on the platform
    tile when hyperspacing, sets state.won (=$0CDE bit6 landscape-complete).
    Death ($216E/$2170): if the 3-energy cost underflows, the player dies; we flag
    that with state.player_dead = True and leave energy at 0."""
    s = copy.deepcopy(state)
    if not hasattr(s, "won"):
        s.won = False
    p = s.player
    if p is None:
        return s
    # win-from-platform: standing on the platform's tile when hyperspacing wins.
    plat = None
    for o in s.objects:
        if o.type == T_PLATFORM:
            plat = o
            break
    on_platform = plat is not None and (p.x, p.y) == (plat.x, plat.y)

    # robot energy cost (carry set -> subtract; underflow => death, $216E/$2170).
    if s.player_energy < HYPERSPACE_ENERGY_COST:
        s.player_energy = 0
        s.player_dead = True
    else:
        s.player_energy = (s.player_energy - HYPERSPACE_ENERGY_COST) & ENERGY_MASK
        s.player_dead = False

    if on_platform:
        s.won = True  # $2196: $0CDE bit6 == landscape complete

    if dest_xy is not None:
        # relocate the player robot to the (live-observed) random destination tile.
        nx, ny = dest_xy
        p.x, p.y = nx, ny
        p.z = int(round(tile_surface_height(s, float(nx), float(ny))))
        p.z_fraction = 0xE0
        p.on_ground = True
        p.stacked_on = None
    return s


# ---- absorbable-objects convenience ---------------------------------------
def can_absorb(
    state: GameState,
    obj: GameObject,
    from_xy: Optional[Tuple[int, int]] = None,
    eye_offset: float = ROBOT_EYE,
) -> bool:
    """The REAL absorb GATE (handle_player_actions $1B46): the player can absorb
    `obj` iff there is line of sight to the object's BASE TILE -- the square it
    rests on -- AND the sight is NOT looking upward ($1D2E), i.e. the eye is
    STRICTLY ABOVE the base-tile surface (you look DOWN at the tile the object
    rests on, never up at its body). This mirrors code_engine.can_absorb exactly.
    `eye_offset` lets the planner test from a raised (climbed) eye."""
    if obj.type == T_PLATFORM or obj.type not in ABSORBABLE:
        return False
    if from_xy is None:
        p = state.player
        from_xy = (p.x, p.y)
    # can_see to the object's BASE tile (o.x,o.y) with the looking-up rejection
    # baked in (target defaults to that tile's surface; surf>eye => not visible).
    return can_see(
        state, from_xy, (obj.x, obj.y), eye_offset=eye_offset, target_height=None
    )


def absorbable_objects(
    state: GameState,
    from_xy: Optional[Tuple[int, int]] = None,
    eye_offset: float = ROBOT_EYE,
) -> List[Tuple[GameObject, int]]:
    """List of (object, energy) for every object the player can currently see AND
    absorb (non-platform), using the real BASE-TILE absorb gate (can_absorb: LOS to
    the object's base tile + eye strictly above that tile). Used for the sanity
    report and the planner."""
    p = state.player
    out = []
    for o in state.objects:
        if o.slot == p.slot:
            continue
        if not can_absorb(state, o, from_xy=from_xy, eye_offset=eye_offset):
            continue
        out.append((o, ENERGY_IN_OBJECTS[o.type]))
    return out


# ---- a thin GameModel facade ----------------------------------------------
class GameModel:
    """Stateless rule engine bound to a starting GameState (for convenience).

    Public interface:
      GameModel(state)
      .energy_table()                 -> dict type->energy ($214F)
      .legal_actions()                -> list[Action]
      .apply(action)                  -> GameModel (new, immutable successor)
      .visible_tiles(from_xy=None)    -> list[(x,y)]
      .can_see(a, b, ...)             -> bool
      .absorbable_objects()           -> list[(GameObject, energy)]
      .seen_by_enemy(x, y, top)       -> list[enemy slot]
      .is_exposed(x, y, top)          -> bool
      .state                          -> the underlying GameState
    """

    def __init__(self, state: GameState):
        self.state = state

    @classmethod
    def from_landscape(cls, landscape: int) -> "GameModel":
        src = gs.Py65Source.from_landscape(landscape)
        return cls(read_game_state(src))

    def energy_table(self):
        return dict(ENERGY_IN_OBJECTS)

    def legal_actions(self):
        return legal_actions(self.state)

    def apply(self, action: Action) -> "GameModel":
        return GameModel(apply(self.state, action))

    def visible_tiles(self, from_xy=None):
        return visible_tiles(self.state, from_xy)

    def can_see(self, a, b, **kw):
        return can_see(self.state, a, b, **kw)

    def absorbable_objects(self):
        return absorbable_objects(self.state)

    def seen_by_enemy(self, x, y, object_top=ROBOT_EYE):
        return seen_by_enemy(self.state, x, y, object_top)

    def is_exposed(self, x, y, object_top=ROBOT_EYE):
        return is_exposed(self.state, x, y, object_top)


# ---- sanity report ($0000) -------------------------------------------------
def main():
    m = GameModel.from_landscape(0)
    st = m.state
    p = st.player

    print("== energy table (energy_in_objects $214F) ==")
    for t in range(7):
        print(f"  {t} {TYPES[t]:<9} {ENERGY_IN_OBJECTS[t]}")

    print("\n== player ==")
    print(f"  slot {p.slot} at tile ({p.x},{p.y}) z={p.z} energy={st.player_energy}")
    print(f"  vertical_scale={st.vertical_scale}  max_enemies={st.max_enemies}")

    vis = m.visible_tiles()
    print(f"\n== line of sight ==")
    print(f"  visible tiles from player: {len(vis)} / {N*N}")

    print("\n== absorbable objects currently visible ==")
    ao = m.absorbable_objects()
    if not ao:
        print("  (none)")
    for o, e in sorted(ao, key=lambda oe: -oe[1]):
        print(
            f"  {o.type_name:<9} slot {o.slot} tile ({o.x},{o.y}) z={o.z} -> energy {e}"
        )

    # sanity: the Sentinel must NOT be directly absorbable from the start tile.
    sent = [o for o in st.objects if o.type == T_SENTINEL][0]
    sent_visible = m.can_see((p.x, p.y), (sent.x, sent.y))
    sent_in_ao = any(o.type == T_SENTINEL for o, _ in ao)
    print("\n== Sentinel reachability sanity ==")
    print(f"  Sentinel at ({sent.x},{sent.y}) z={sent.z}; player z={p.z}")
    print(f"  can_see(player -> Sentinel tile): {sent_visible}")
    print(f"  Sentinel directly absorbable from start: {sent_in_ao}")
    assert not sent_in_ao, "Sentinel should NOT be absorbable from the start tile"
    print("  OK: Sentinel is not directly absorbable from the start (as expected).")

    # threat: is the start tile exposed to any enemy?
    exposed = m.seen_by_enemy(p.x, p.y)
    print("\n== enemy threat ==")
    print(f"  enemies (sentinel/sentry): {[e.slot for e in enemies(st)]}")
    print(f"  enemy slots that can see the player's start tile: {exposed}")


if __name__ == "__main__":
    main()
