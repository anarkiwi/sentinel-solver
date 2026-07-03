#!/usr/bin/env python3
"""Fully-native forward model + keyboard-win planner for The Sentinel (C64).

NO py65 in the search loop. The terrain height field stays pure in tiles_table,
so native_los (bit-exact for terrain) computes the action-time LOS gate at
~50us/call.  Objects (boulders/robots) and the player's raised eye after a climb
are tracked natively; energy/heights use the ROM-measured constants.  The final
plan is validated ONCE through the emulator (code_engine) to confirm the real
ROM win flag.

The keyboard rules this models (each verified against the ROM elsewhere):
  * You can only create/absorb on a tile your sights can aim at WITH line of
    sight (the real $1B46 gate) -> gated_view(tile, eye) must exist.
  * Absorb needs the eye strictly above the object's BASE tile (look down).
  * A boulder column can only be extended while its top is <= eye + ~2 (you must
    see the tile you build on); past that you must transfer up first (staged
    climb).  Build on an ADJACENT tile, never your own.
"""

import sys, os, json, time

_HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, _HERE)
os.chdir(os.path.join(_HERE, ".."))  # _emu reads out/sentinel_stage2.bin
import _emu
from native_los import NativeState, aim_target_native

N = 32


def TIDX(x, y):
    return (x & 3) * 256 + ((x >> 2) & 7) * 32 + y


PLAYER_OBJ = 0x000B
OBJ_X, OBJ_Y, OBJ_Z, OBJ_ZF = 0x0900, 0x0980, 0x0940, 0x0A00
OBJ_HANG, OBJ_VANG, OBJ_TYPE, OBJ_FLAGS = 0x09C0, 0x0140, 0x0A40, 0x0100
PLAYER_ENERGY, PLAT_X, PLAT_Y = 0x0C0A, 0x0C19, 0x0C1A
ENERGY = {0: 3, 1: 3, 2: 1, 3: 2, 4: 1, 5: 4, 6: 0}
CUR_CX, CUR_CY = 0x50, 0x5F  # sights centre ($1356)
ROBOT_EYE_FUDGE = 2  # build-height slack: top <= eye+2
NEIGH8 = [(0, -1), (0, 1), (-1, 0), (1, 0), (-1, -1), (1, 1), (1, -1), (-1, 1)]


def terrain_z(mem, x, y):
    if not (0 <= x < N and 0 <= y < N):
        return None
    b = mem[0x0400 + TIDX(x, y)]
    return (b >> 4) if b < 0xC0 else None  # None == occupied/object tile


# ---- the native LOS gate: ONE sweep gives every tile you can aim at + the view -
# full pitch circle: climbing needs same-level / near-horizontal footholds, not just
# the down-looking band used for absorbing.  Native LOS still rejects too-high tiles.
# Coarse step keeps the per-sweep cost low (A* calls it for many (tile,z) states).
# the keyboard-reachable pitch lattice: v ≡ 1 (mod 4), inside the pan clamp
# [$CD..$35] ($10FF/$1149; every body's v_angle=$F5 per put_object_in_tile $1F7E).
# ±4 v-pans preserve the residue and cannot cross the clamp, so views off this
# lattice are physically unreachable by the keyboard.
_VBAND = [v & 0xFF for v in list(range(0xCD, 0x100, 4)) + list(range(0x01, 0x36, 4))]
# h azimuth step: the keyboard can only reach h ≡ 0 (mod 8) (±8 pans / u-turn EOR
# $80 from any body's h_angle ≡ 0 mod 8), so a finer step would emit unreachable
# views. 8 IS the keyboard-faithful resolution for BOTH fine and coarse sweeps.
_HSTEP = 8
# Coarse sampling for the A* CANDIDATE-GENERATION sweeps only.  Profiling ls42
# showed each fine sweep (hstep4/vstep4 = 4096 rays) costs ~0.6-0.77s and a single
# A* pass fires ~64 of them (=49s) -- over the 60s budget.
#
# The AZIMUTH (h) resolution only decides which TILES a ray lands on, so hstep=8
# (1/2 the rays) keeps essentially every foothold.  The PITCH (v) resolution, by
# contrast, decides whether a ray threads a NARROW ridge-crossing LOS window: ls9999
# starts boxed in a pocket whose only break-out (eye 5 at (3,6) revealing the east
# tiles (8,6)/(8,7)) is a thin pitch window that vstep=8 MISSES (-> NO PATH) but
# vstep=4 catches.  So coarse = hstep8 (azimuth halved) / vstep4 (pitch kept fine):
# the critical break-out is found at ~0.31s/sweep (vs 0.62s fine, ~2x), and the climb
# A* for ls42 runs well under budget.  The OUTER plan-validate-replay path still
# re-sweeps each chosen foothold at FINE resolution (via the real gate), so a foothold
# coarse misses only costs the A* a candidate, never correctness.
_HSTEP_COARSE = 8
_VBAND_COARSE = _VBAND


def centre_view_for(mem, tile, player_slot, eye_z, hstep=8, max_steps=2000):
    """NATIVE centre-aimed view onto `tile`: among views whose ray lands on the tile
    WITH line of sight (the real $1B46 gate requires it), the one with the smallest
    get_minimum_x_or_y_fraction_from_tile_centre ($1EAF).  For a boulder tile this is
    the view that makes it targetable ($1E4B) -- so a synthoid can be built on it --
    AND, now that native LOS is bit-exact over object tiles, the returned view is
    directly accepted by code_engine.create_via_gate on the FIRST try (no window
    search).  Pure native (no emulation); one create_via_gate later only advances the
    real game.

    Returns None when no LOS view lands on the tile (the build is NOT gate-feasible
    from this eye -- e.g. the object's surface is above the eye-line and not centre-
    aimable): the caller must raise the eye / pick another foothold rather than emit
    an infeasible keyboard action.  Among LOS hits a no-LOS hit is never preferred."""
    st = NativeState.from_mem(mem)
    best = None  # (los_rank, centre, view) min-first
    cx, cy = CUR_CX, CUR_CY
    for h in range(0, 256, hstep):
        for v in _VBAND:
            tx, ty, los, centre = aim_target_native(
                st,
                h,
                v,
                cx,
                cy,
                player_slot,
                eye_z=eye_z,
                max_steps=max_steps,
                return_centre=True,
            )
            if (tx, ty) != tile:
                continue
            # rank LOS hits ahead of no-LOS hits, then by smallest centre fraction.
            key = (0 if los else 1, centre)
            if best is None or key < best[0]:
                best = (key, {"h_angle": h, "v_angle": v, "cursor": [cx, cy]})
    # Only return a LOS-positive view: a no-LOS view is rejected by the real gate.
    if best is not None and best[0][0] == 0:
        return best[1]
    return None


def visibility_sweep(mem, player_slot, eye_z, max_steps=320, coarse=False):
    """March the sights over all azimuths x the down-looking pitch band ONCE and
    return {tile: view} for every tile the player can aim at with LOS from its
    current position + eye_z.  Native + step-capped, so ~tens of ms, not a
    per-tile search.  The view is the (h_angle, v_angle, cursor) keystroke target.

    coarse=True halves the azimuth/pitch resolution (hstep8/vstep8) for the A*
    candidate-generation sweeps -- ~4x faster, finds nearly every foothold; the
    fine default is kept for the gate-faithful validate/replay views."""
    st = NativeState.from_mem(mem)
    seen = {}
    cx, cy = CUR_CX, CUR_CY
    hstep = _HSTEP_COARSE if coarse else _HSTEP
    vband = _VBAND_COARSE if coarse else _VBAND
    for h in range(0, 256, hstep):
        for v in vband:
            tx, ty, los = aim_target_native(
                st, h, v, cx, cy, player_slot, eye_z=eye_z, max_steps=max_steps
            )
            if los and (tx, ty) not in seen:
                seen[(tx, ty)] = {"h_angle": h, "v_angle": v, "cursor": [cx, cy]}
    return seen


# ---- incremental place-and-re-sweep: model the search's OWN occluders -------
# native_los (check_for_line_of_sight_to_tile $1CDD) marches the ray and at every
# intermediate tile calls _calc_tile_z_and_slope; for an object tile (byte >= $C0)
# it walks the object stack ($1E3F get_tile_z_from_object), raising the surface to
# whatever object sits there and BLOCKING the ray if it pokes above the eye-line.
# That engine reads the LIVE object arrays via NativeState.from_mem.  The climb A*
# therefore MUST write the stepping-stone boulders / synthoids it drops AND every
# abandoned synthoid shell (Game.transfer never frees the old body) into those
# arrays, or the sweep runs against a board missing the route's own occluders and
# the real $1B46 gate later refuses the build (false-positive LOS).
#
# A placed object is (tx, ty, z_height_int, otype).  Surface heights use the SAME
# constants Game.create uses (first-on-terrain +0.875 boulder / +0.5 other; stack
# +0.5); the ROM stores OBJ_Z = int(surface), which is what the LOS march reads.
# The tile byte at $0400+TIDX must be flagged >= $C0 so _calc_tile_z_and_slope
# takes the object branch and resolves the placed surface.


def _objtile_byte(state, x, y):
    """Tile byte that calculate_tile_address ($2BA8) reads for (x,y).  An object
    tile is byte >= $C0; the ROM's object code reads the surface from the object
    arrays, so the exact low bits don't matter -- only that bit7+bit6 are set."""
    return state.mem[0x0400 + TIDX(x, y)]


def sweep_with_placed(
    base_mem,
    player_slot,
    eye_z,
    player_tile,
    placed,
    max_steps=320,
    eye_frac=0x00,
    coarse=False,
):
    """visibility_sweep, but with `placed` objects materialized into a SCRATCH copy
    of the object arrays + tiles_table so native_los occludes on them exactly as the
    real ROM will at action time.

    `eye_frac` is the observer's z_frac ($0A00+slot): the FRACTIONAL height of the
    surface the player stands on (0x00 bare terrain, 0xE0 synthoid-on-terrain +0.875,
    0x60 synthoid-on-boulder +0.375).  check_for_line_of_sight_to_tile ($1CDD) seeds
    the ray's pz_sub ($0038) from this z_frac, so it MUST match the real stack height
    or the marched ray diverges (a higher frac lifts the eye and can clear an occluder
    the real, lower eye is blocked by -- the ls9999 (2,6)->(3,9) false-positive).

    `placed` is an iterable of (tx, ty, z_height_int, otype).  We write each into a
    free object slot (FLAGS=0x00, X/Y, Z=z_height_int, TYPE=otype) and mark its tile
    byte as an object tile ($C0+), then run the sweep, then restore.  The player's
    OWN tile is the observer (visibility_sweep places it via x,y/eye); placed objects
    sitting ON the player tile (its boulder/its own shell) are skipped so the eye is
    not occluded by the column it stands on.

    Returns {tile: view}.  Pure native (no emulation)."""
    # group placed objects per tile so a stacked tile chains through obj_flags.
    by_tile = {}
    for tx, ty, zh, zf, ot in placed:
        if (tx, ty) == player_tile:
            continue  # observer's own column -- skip
        by_tile.setdefault((tx, ty), []).append((zh, zf, ot))

    mem = bytearray(base_mem)  # scratch copy -- never mutate caller's
    # place the OBSERVER at player_tile (the A* moves the player from node to node;
    # base_mem still holds its original x,y).  eye_z is passed straight to the sweep;
    # eye_frac seeds the ray's pz_sub so the eye sits at the true stack height.
    mem[OBJ_X + player_slot] = player_tile[0] & 0xFF
    mem[OBJ_Y + player_slot] = player_tile[1] & 0xFF
    mem[OBJ_ZF + player_slot] = eye_frac & 0xFF
    if not by_tile:
        return visibility_sweep(mem, player_slot, eye_z, max_steps, coarse=coarse)

    # collect free slots NOT used by the player (so we don't clobber the observer).
    free = [s for s in range(64) if (mem[OBJ_FLAGS + s] & 0x80) and s != player_slot]
    for (tx, ty), stack in by_tile.items():
        if not (0 <= tx < N and 0 <= ty < N):
            continue
        stack.sort()  # base object first, then stacked
        prev = None  # slot of the object below (for flags chain)
        placed_any = False
        for zh, zf, ot in stack:
            if not free:
                break
            slot = free.pop()
            mem[OBJ_FLAGS + slot] = 0x00
            mem[OBJ_X + slot] = tx & 0xFF
            mem[OBJ_Y + slot] = ty & 0xFF
            mem[OBJ_Z + slot] = zh & 0xFF
            mem[OBJ_ZF + slot] = zf & 0xFF
            mem[OBJ_TYPE + slot] = ot & 0xFF
            if prev is not None:
                # objects_flags links a stacked object to the one below: the ROM's
                # get_height_of_lowest_object ($1EA4) walks flags>=$40 down the stack.
                mem[OBJ_FLAGS + slot] = 0x40 | (prev & 0x3F)
            prev = slot
            placed_any = True
        if placed_any:
            # flag the tile as object-occupied so _calc_tile_z_and_slope ($1DF9)
            # takes the get_tile_z_from_object branch.  The tiles_table low byte
            # encodes the TOP object's slot for the ROM (objects_flags chain head).
            mem[0x0400 + TIDX(tx, ty)] = 0xC0 | (prev & 0x3F)
    return visibility_sweep(mem, player_slot, eye_z, max_steps, coarse=coarse)


class Game:
    """Native mutable game state.  Terrain (tiles_table) is never mutated, so the
    native LOS stays bit-exact; objects live in `self.col` (tile -> top height as
    a float) and the free-slot list."""

    def __init__(self, landscape):
        mem, _ = _emu.generate(landscape)  # the ONLY py65 use (PRNG generation)
        self.mem = bytearray(mem)
        self.landscape = landscape
        self.player = self.mem[PLAYER_OBJ]
        self.energy = self.mem[PLAYER_ENERGY]
        self.plat = (self.mem[PLAT_X], self.mem[PLAT_Y])
        self.col = {}  # tile -> top height (float) of its object stack
        self._tiles0 = bytes(self.mem[0x0400:0x0800])  # pristine terrain byte snapshot
        self.free = [s for s in range(64) if self.mem[OBJ_FLAGS + s] & 0x80]
        # player eye = its tile's terrain height (a robot standing on the ground)
        px, py = self.mem[OBJ_X + self.player], self.mem[OBJ_Y + self.player]
        self.eye = float(terrain_z(self.mem, px, py) or self.mem[OBJ_Z + self.player])
        # sentinel + platform-ground for the win test
        self.sentinel_slot = next(
            (
                s
                for s in range(64)
                if not (self.mem[OBJ_FLAGS + s] & 0x80) and self.mem[OBJ_TYPE + s] == 5
            ),
            None,
        )
        self.plat_ground = terrain_z(self.mem, *self.plat)
        if self.plat_ground is None:  # platform tile is object-occupied
            pslot = next(
                (
                    s
                    for s in range(64)
                    if not (self.mem[OBJ_FLAGS + s] & 0x80)
                    and self.mem[OBJ_TYPE + s] == 6
                ),
                None,
            )
            if pslot is not None:
                self.plat_ground = self.mem[OBJ_Z + pslot]
        self.steps = []

    # --- queries ---
    def player_xy(self):
        return (self.mem[OBJ_X + self.player], self.mem[OBJ_Y + self.player])

    def top_of(self, tile):
        if tile in self.col:
            return self.col[tile]
        z = terrain_z(self.mem, *tile)
        return float(z) if z is not None else None

    # --- the gated, keyboard-faithful actions ---
    def feasible(self, otype, tile):
        """Keyboard-feasibility of a create on `tile` (energy / own-tile / free-slot
        / build-height-limit).  The LOS gate is supplied separately by the sweep
        (empty tiles, native-exact) or the height rule (occupied stacking tiles)."""
        if tile == self.player_xy():
            return False  # cannot build on your own tile ($1F38)
        if self.energy < ENERGY[otype] or not self.free:
            return False
        tb = self.mem[0x0400 + TIDX(*tile)]
        if tb >= 0xC0:  # occupied tile: create only on boulder/plat
            below = tb & 0x3F
            if self.mem[OBJ_TYPE + below] not in (3, 6):
                return False  # $1F38 leave_with_carry_set
            top = self.top_of(tile)
            if top is not None and top > self.eye + ROBOT_EYE_FUDGE:
                return False  # column top above sightline
        return True

    def create(self, otype, tile, view, note=""):
        """Build `otype` on `tile`, maintaining the tiles_table byte + objects_flags
        chain exactly like the ROM (put_object_in_tile $1F2F-$1F79) so native_los
        occludes on the plan's own objects.  Returns the new slot, or None if the
        create is ROM-infeasible (no free slot / wrong top-type)."""
        if not self.free:
            return None
        slot = self.free.pop()
        tb = self.mem[0x0400 + TIDX(*tile)]
        if tb >= 0xC0:  # stacking on an object
            below = tb & 0x3F
            btype = self.mem[OBJ_TYPE + below]
            if btype == 6:  # platform ($1F47-$1F50)
                zf = self.mem[OBJ_ZF + below]
                z = self.mem[OBJ_Z + below] + 1
            elif btype == 3:  # boulder ($1F52-$1F5F)
                t = self.mem[OBJ_ZF + below] + 0x80
                zf = t & 0xFF
                z = self.mem[OBJ_Z + below] + (t >> 8)
            else:
                self.free.append(slot)
                return None  # $1F38 leave_with_carry_set
            self.mem[OBJ_FLAGS + slot] = 0x40 | below
        else:  # bare terrain ($1F66-$1F76)
            self.mem[OBJ_FLAGS + slot] = 0x00
            zf = 0xE0
            z = tb >> 4  # z_fraction $E0 for ALL types
        self.mem[0x0400 + TIDX(*tile)] = 0xC0 | slot
        self.mem[OBJ_X + slot] = tile[0]
        self.mem[OBJ_Y + slot] = tile[1]
        self.mem[OBJ_Z + slot] = z & 0xFF
        self.mem[OBJ_ZF + slot] = zf & 0xFF
        self.mem[OBJ_TYPE + slot] = otype
        self.col[tile] = z + zf / 256.0
        self.energy -= ENERGY[otype]
        self.steps.append(
            {
                "verb": "create",
                "otype": otype,
                "target": list(tile),
                "view": view,
                "player_tile": list(self.player_xy()),
                "eye_z": round(self.eye, 3),
                "note": note,
            }
        )
        return slot

    def transfer(self, slot, note=""):
        tile = (self.mem[OBJ_X + slot], self.mem[OBJ_Y + slot])
        self.player = slot
        self.mem[PLAYER_OBJ] = slot
        self.eye = self.top_of(tile)
        # CRITICAL: the forward model must maintain the player's z_FRAC (sub-tile height
        # fraction), not just the integer z_height -- native_los reads it for the exact
        # ray origin, and the real eye is z_height + z_frac/256. Dropping it (leaving
        # 0x00) made the LOS observer ~0.375 too low and diverged from the ROM (which
        # carries e.g. 0x60 for a synthoid on a boulder). Derive it from self.eye.
        self.mem[OBJ_Z + slot] = int(self.eye)
        self.mem[OBJ_ZF + slot] = int(round((self.eye - int(self.eye)) * 256)) & 0xFF
        self.steps.append(
            {
                "verb": "transfer",
                "otype": None,
                "target": list(tile),
                "view": None,
                "player_tile": list(tile),
                "eye_z": round(self.eye, 3),
                "note": note,
            }
        )

    def absorb(self, slot, view, note=""):
        # energy wraps mod 64 (AND #$3F $2148), not clamped.
        self.energy = (self.energy + ENERGY[self.mem[OBJ_TYPE + slot]]) & 0x3F
        tile = (self.mem[OBJ_X + slot], self.mem[OBJ_Y + slot])
        flags = self.mem[OBJ_FLAGS + slot]
        self.mem[OBJ_FLAGS + slot] = 0x80
        self.free.append(slot)
        # ROM remove_object $1EEF-$1F15: if an object sat below, it becomes the tile top;
        # else the tile byte reverts to z<<4 with the slope nibble ZEROED.
        if 0x40 <= flags <= 0x7F:
            below = flags & 0x3F
            self.mem[0x0400 + TIDX(*tile)] = 0xC0 | below
            self.col[tile] = self.mem[OBJ_Z + below] + self.mem[OBJ_ZF + below] / 256.0
        else:
            self.mem[0x0400 + TIDX(*tile)] = (self.mem[OBJ_Z + slot] << 4) & 0xFF
            self.col.pop(tile, None)
        self.steps.append(
            {
                "verb": "absorb",
                "otype": int(self.mem[OBJ_TYPE + slot]),
                "target": list(tile),
                "view": view,
                "player_tile": list(self.player_xy()),
                "eye_z": round(self.eye, 3),
                "note": note,
            }
        )


# ---- the staged-climb keyboard planner ------------------------------------
def cheb(a, b):
    return max(abs(a[0] - b[0]), abs(a[1] - b[1]))


def _move_placed(placed, tile, z, T2, use_b, terrain_T2, z_frac=0xE0):
    """Forward-model the object set after a move from `tile` (eye z, z_frac) to `T2`.

    A move leaves an ABANDONED SHELL on the departed tile (Game.transfer never frees
    the old body -- $1B?? transfer just reassigns the player slot), and creates the
    stepping-stone object(s) on T2.  The (Z_height, z_frac) surface values are taken
    from the REAL ROM (measured via code_engine create+transfer, ls42):
      * hop          -> synthoid on terrain T2: Z=terrain, frac 0xE0 (surface +0.875);
                        the player eye on it = (terrain, 0xE0).
      * boulder-step -> boulder on T2: Z=terrain, frac 0xE0 (surface +0.875); THEN a
                        synthoid on the boulder: Z=terrain+1, frac 0x60 (surface +1.375);
                        the player eye on it = (terrain+1, 0x60).
    The shell left on the departed tile is a synthoid at the eye surface the player
    stood at there = (z, z_frac).  The player then stands ON T2 (the observer's own
    column, skipped by the sweep).

    Placed objects are (tx, ty, Z_height, z_frac, otype).  Returns
    (new_placed_frozenset, new_eye_int, new_eye_frac).  z_frac feeds the occlusion
    surface (get_tile_z_from_object $1E3F reads objects_z_frac); the LOS march ($1CDD)
    compares against (Z_height, z_frac)."""
    s = set(placed)
    # abandoned shell on the tile we transfer OFF: a synthoid (type 0) whose surface
    # is the eye surface (z, z_frac) the player stood at.
    s.add((tile[0], tile[1], int(z), z_frac & 0xFF, 0))
    if use_b:
        s.add((T2[0], T2[1], int(terrain_T2), 0xE0, 3))  # boulder (+0.875)
        s.add((T2[0], T2[1], int(terrain_T2) + 1, 0x60, 0))  # synthoid on boulder
        new_eye = int(terrain_T2) + 1
        new_frac = 0x60
    else:
        s.add((T2[0], T2[1], int(terrain_T2), 0xE0, 0))  # synthoid on terrain
        new_eye = int(terrain_T2)
        new_frac = 0xE0
    return frozenset(s), new_eye, new_frac


def _astar_terrain(
    g, target_z, blocked, sweep_cache, sweep_counter, max_nodes, sweep_fn=None
):
    """Fast A* over (tile, eye) using BARE-terrain sweeps (cached by (tile,eye)),
    avoiding any edge in `blocked` (a set of (from_tile, from_eye, to_tile, use_b)).
    Returns a list of (T2, use_boulder, view) moves or None.  This is the original
    (fast, cacheable) climb A*; the occlusion correctness is enforced by the OUTER
    plan-validate-blocklist loop in _find_climb_path, which re-sweeps each foothold
    against the route's TRUE accumulated objects and blocks the ones that are occluded.

    `sweep_fn(base, slot, eye, tile, placed)` (optional) lets the TIMED planner inject
    its exposure gate; here placed is always frozenset() (bare terrain)."""
    import heapq

    base = bytes(g.mem)
    ps = g.player
    start = (g.player_xy(), int(g.eye), int(g.mem[OBJ_ZF + ps]))
    goal = lambda s: cheb(s[0], g.plat) <= 1 and s[1] >= target_z
    hh = lambda s: cheb(s[0], g.plat) + max(0, target_z - s[1])

    def sweep_from(tile, z, efrac):
        c = sweep_cache.get((tile, z, efrac))
        if c is not None:
            return c
        # A* candidate-generation sweep: COARSE resolution (hstep8/vstep4) AND a
        # shortened ray march (max_steps=120).  The break-out tiles a sweep must find
        # sit within ~8 tiles of the eye, so 120 march steps suffice while ~halving the
        # per-ray cost vs 320; the chosen path is re-swept at FINE/full-march resolution
        # in _validate_path_occlusion and finally re-gated in the real ROM, so a coarse
        # miss costs a candidate, never correctness.
        sw = (
            sweep_fn(base, ps, z, tile, frozenset(), efrac)
            if sweep_fn
            else sweep_with_placed(
                base,
                ps,
                z,
                tile,
                frozenset(),
                max_steps=120,
                eye_frac=efrac,
                coarse=True,
            )
        )
        if sweep_counter is not None:
            sweep_counter[0] += 1
        sweep_cache[(tile, z, efrac)] = sw
        return sw

    openq = [(hh(start), 0.0, start)]
    gc = {start: 0.0}
    came = {start: (None, None)}
    found = None
    nodes = 0
    while openq:
        _f, c, s = heapq.heappop(openq)
        if goal(s):
            found = s
            break
        if c > gc.get(s, 1e18) or nodes > max_nodes:
            continue
        nodes += 1
        tile, z, efrac = s
        for T2, view in sweep_from(tile, z, efrac).items():
            if T2 == tile:
                continue
            tz2 = terrain_z(g.mem, *T2)
            if tz2 is None:
                continue
            moves = [(False, 1.0)]
            if cheb(T2, tile) <= 1 and tz2 <= z:
                moves.append((True, 1.3))
            for ub, cost in moves:
                if (tile, z, T2, ub) in blocked:
                    continue
                _np, neye, nfrac = _move_placed(
                    frozenset(), tile, z, T2, ub, tz2, z_frac=efrac
                )
                ns = (T2, neye, nfrac)
                nc = c + cost + 0.12 * max(0, cheb(T2, tile) - 2)
                if nc < gc.get(ns, 1e18):
                    gc[ns] = nc
                    came[ns] = (s, (T2, ub, view, z))
                    heapq.heappush(openq, (nc + hh(ns), nc, ns))
    if found is None:
        return None
    path, s = [], found
    while came[s][1] is not None:
        path.append(came[s][1])  # (T2, ub, view, from_eye)
        s = came[s][0]
    path.reverse()
    return path


def _validate_path_occlusion(g, path, sweep_counter, sweep_fn=None):
    """Replay `path` natively, accumulating the route's OWN objects (stepping-stone
    boulders/synthoids + abandoned shells) and at EACH step re-sweep WITH those objects
    materialized (sweep_with_placed) -- exactly what native_los ($1CDD) / the real $1B46
    gate sees at action time.  Returns (ok, bad_edge, refreshed_path).

    On the FIRST foothold that is occluded by the route's accumulated objects, returns
    ok=False and the offending edge (from_tile, from_eye, to_tile, use_b) so the outer
    loop can blocklist it and replan.  When ok=True the refreshed_path carries the
    re-swept (gate-faithful) view for each step."""
    base = bytes(g.mem)
    ps = g.player
    tile = g.player_xy()
    eye = int(g.eye)
    efrac = int(g.mem[OBJ_ZF + ps])
    placed = frozenset()
    refreshed = []
    for T2, ub, _view, _from_eye in path:
        sw = (
            sweep_fn(base, ps, eye, tile, placed, efrac)
            if sweep_fn
            else sweep_with_placed(base, ps, eye, tile, placed, eye_frac=efrac)
        )
        if sweep_counter is not None:
            sweep_counter[0] += 1
        if T2 not in sw:
            # this foothold is occluded once the route's own objects are present.
            return False, (tile, eye, T2, ub), None
        refreshed.append((T2, ub, sw[T2]))
        placed, eye, efrac = _move_placed(
            placed, tile, eye, T2, ub, terrain_z(g.mem, *T2), z_frac=efrac
        )
        tile = T2
    return True, None, refreshed


def _find_climb_path(
    g, target_z, sweep_counter=None, log=lambda *a: None, max_nodes=4000, sweep_fn=None
):
    """Climb planner that models its OWN occluders.  Returns a list of
    (tile, use_boulder, view) moves that travels to the platform AND ascends to
    >= target_z, or None.

    THE OCCLUSION FIX (incremental place-and-re-sweep, plan/validate/blocklist form).
    The original climb A* computed every visibility sweep over the BARE terrain, so
    native_los ($1CDD) never saw the stepping-stone boulders, synthoids, or abandoned
    synthoid shells (Game.transfer never frees the old body) the route drops.  By
    arrival an earlier-dropped object sits on the ray, the real $1B46 gate refuses the
    build, and the replay dies (ls9999 step ~5/10; ls42 step ~14).

    Carrying the full placed-object SET in the A* key is correct but intractable here:
    every node gets a distinct set, the sweep cache (each sweep ~0.75 s) never hits and
    the frontier explodes.  So we do place-and-re-sweep as a GREEDY-WITH-BACKTRACK at
    the PATH level (the brief sanctions this):

      1. Fast A* over (tile, eye) on bare terrain (sweeps cached by (tile, eye)),
         avoiding any edge in `blocked`.
      2. Validate the path: replay it, accumulating the route's TRUE objects, and at
         each step re-sweep WITH them (sweep_with_placed).  The placed set grows
         faithfully -- each move adds the boulder + synthoid (boulder-step) or a
         synthoid on terrain (hop), AND the abandoned shell on the tile transferred OFF
         -- with ROM int-floor Z (OBJ_Z), so native_los occludes on them exactly as the
         action-time gate will.
      3. On the first occluded foothold, blocklist that edge and replan (1).  Repeat
         until a path validates clean (its footholds stay LOS-visible with all the
         route's objects present) or no path remains.

    Each A* is fast and fully cached; only ~(path length) place-and-re-sweeps run per
    iteration, and a handful of iterations suffice -- so the loop stays cheap while the
    returned path is occlusion-faithful to the real ROM.  use_boulder steps (+1 eye)
    must be to an ADJACENT tile whose terrain <= eye; hops go to any visible terrain
    tile.  `sweep_fn` is the TIMED planner's exposure-gate injection point."""
    blocked = set()
    sweep_cache = {}
    for it in range(60):  # bounded blocklist-replan iterations
        path = _astar_terrain(
            g, target_z, blocked, sweep_cache, sweep_counter, max_nodes, sweep_fn
        )
        if path is None:
            log(
                f"  climb path: NO PATH after {it} blocklist iterations "
                f"({len(blocked)} edges blocked, {len(sweep_cache)} sweeps)"
            )
            return None
        ok, bad, refreshed = _validate_path_occlusion(g, path, sweep_counter, sweep_fn)
        if ok:
            log(
                f"  climb path: {len(refreshed)} moves, {it} replans, "
                f"{len(blocked)} blocked edges, {len(sweep_cache)} cached sweeps"
            )
            return refreshed
        # blocklist the occluded edge and the bare A* sweep that produced it (so the
        # replan does not just re-pick the same foothold from the same node).
        blocked.add(bad)
    log("  climb path: blocklist-replan budget exhausted")
    return None


def plan(landscape, verbose=True, top_energy=True):
    t0 = time.time()
    g = Game(landscape)
    if top_energy:
        g.energy = 63  # TODO: acquire energy by absorbing trees
    log = lambda *a: verbose and print(*a)
    log(
        f"ls{landscape}: start {g.player_xy()} eye {g.eye} platform {g.plat} "
        f"plat_ground {g.plat_ground} sentinel_slot {g.sentinel_slot}"
    )
    n_sweeps = [0]

    def sweep():
        n_sweeps[0] += 1
        # eye_z = the ROM's integer z_height (== floor of the float column top),
        # NOT round(): the LOS march reads obj_z_height, so floor makes the native
        # view bit-exactly replayable on the real machine.  COARSE: these per-step
        # sweeps only drive energy-recovery reabsorb (prev_tile membership) and the
        # final win FEASIBILITY check; the real absorb/create views are recomputed
        # at fine resolution in validate_kbd_plan (centre_view_for / native_view_for),
        # so coarse here is a pure speed win.
        return visibility_sweep(g.mem, g.player, int(g.eye), max_steps=120, coarse=True)

    # ---- A* over (tile, z_height): find a route of stepping-stones (hops + adjacent
    # boulder-steps) that BOTH travels to the platform AND ascends above it.  Sweeps
    # are a function of (tile, z) on the base terrain (boulders are local footholds we
    # leave behind), so we cache them.
    target_z = (g.plat_ground or 8) + 1  # eye must be above the platform
    path = _find_climb_path(g, target_z, sweep_counter=n_sweeps, log=log)
    if path is None:
        log("  NO PATH found by A*")
    else:
        for tile, use_b, view in path:
            prev_slot, prev_tile = g.player, g.player_xy()
            if use_b:
                g.create(3, tile, view, "climb boulder")
                sslot = g.create(
                    0, tile, None, "climb synthoid"
                )  # on boulder: emulated view
            else:
                sslot = g.create(0, tile, view, "hop synthoid")
            g.transfer(sslot, "step")
            sw2 = sweep()  # energy recovery
            if (
                prev_tile in sw2
                and g.mem[OBJ_TYPE + prev_slot] == 0
                and prev_tile not in g.col
            ):
                g.absorb(prev_slot, sw2[prev_tile], "reabsorb prior synthoid")
            log(
                f"  {'step' if use_b else 'hop'} -> {tile} eye {g.eye} "
                f"(d={cheb(g.player_xy(), g.plat)}) energy {g.energy}"
            )

    log(
        f"reached {g.player_xy()} eye {g.eye} (d={cheb(g.player_xy(), g.plat)} "
        f"from platform) in {time.time()-t0:.2f}s, {n_sweeps[0]} sweeps, "
        f"{len(g.steps)} steps, energy {g.energy}"
    )

    # ---- final: eye above the platform + adjacent, absorb the Sentinel (look down at
    # its base tile), put a synthoid on the platform and transfer onto it (win). ----
    won = False
    if g.plat_ground is not None:
        sw = sweep()
        if (
            int(g.eye) > g.plat_ground
            and cheb(g.player_xy(), g.plat) <= 1
            and g.sentinel_slot is not None
        ):
            g.absorb(g.sentinel_slot, sw.get(g.plat), "absorb Sentinel")
            log(f"  absorbed Sentinel from eye {g.eye}, energy {g.energy}")
            if g.feasible(0, g.plat):
                g.transfer(
                    g.create(0, g.plat, None, "platform synthoid"),
                    "hyperspace onto platform (WIN)",
                )
                won = True
                log(f"  WIN: synthoid on platform {g.plat} + transfer")
    log(
        f"=== plan {'WON' if won else 'INCOMPLETE'} in {time.time()-t0:.2f}s, "
        f"{len(g.steps)} steps, {n_sweeps[0]} sweeps ==="
    )
    g.native_won = won
    return g


if __name__ == "__main__":
    ls = int(sys.argv[1]) if len(sys.argv) > 1 else 0
    g = plan(ls)
    out = {
        "landscape": ls,
        "native_won": g.native_won,
        "final_player": g.player_xy(),
        "eye": g.eye,
        "energy": g.energy,
        "steps": g.steps,
    }
    json.dump(out, open(f"out/kbd_native_{ls:04d}.json", "w"), indent=0)
    print(
        "FINAL",
        g.player_xy(),
        "eye",
        g.eye,
        "energy",
        g.energy,
        "steps",
        len(g.steps),
        "native_won",
        g.native_won,
    )
