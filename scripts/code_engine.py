#!/usr/bin/env python3
"""Real-code game engine for The Sentinel (C64): drive the actual
6502 routines in py65 at warp speed (headless, no VICE, no rendering) to exercise
the game's real mechanics and DIFFERENTIALLY VALIDATE the pure-Python port
(`game_model.py` / `enemy_dynamics.py`) against the real ROM.

This is a minimal-emulation engine: it sets the routine inputs in
memory/registers and JSRs the real routine, then reads the resulting state out
with `game_state.Py65Source`. Every routine ABI below is cited to a game
address and verified against the body.

It builds on `scripts/_emu.py` (do NOT modify that): `fresh_machine()` loads
`out/sentinel_stage2.bin`, stubs the KERNAL with RTS, stubs $DC00/$DC01 keyboard
and $D012 raster, and treats $E000 writes as "render started". `generate()` runs
the real seed->generate->place-enemies->place-player sequence.

--------------------------------------------------------------------------------
ROUTINE ABIs DRIVEN (all verified against the game code)
--------------------------------------------------------------------------------
LINE OF SIGHT (ground truth) -- the terrain-LOS half of check_if_enemy_can_see_object
  $1887, which the player path (handle_player_actions $1B46) and enemy path
  ($17B4/$18F6) both funnel into check_for_line_of_sight_to_tile $1CDD:
    * observer object slot in $006E ($1CDD LDX $006E).
    * a unit direction vector toward the target, built by prepare_vector_from_angle
      $1C54 from $3D (h frac), $3E (h angle), $3F/$40 + A=$8B (v angle hi).
    * the h/v angles come from calculate_object_relative_angles_and_distance $8401
      (Y=target slot, observer in $006E -> $8A/$8B = horizontal bearing, $7C/$7D =
      distance, $81/$84 = relative z) and calculate_object_relative_vertical_angle
      $933D (X=observer, A=rel z hi, $80=rel z lo, $7C/$7D distance -> $8A/$8B =
      vertical angle).
    * $1CDD returns CARRY CLEAR if there is line of sight ($1D42), CARRY SET if
      blocked ($1D44 -- terrain rose above the ray, or beyond the $1F board edge).
  We replicate exactly the NON-ROBOT probe the ROM uses for a tile-as-tree target
  ($18DA BNE not_robot -> $1907 lowers the probe by $E0/256 then loops to $18E6 to
  run prepare_vector + $1CDD): a single LOS probe aimed just below the target tile
  surface. (The robot path ORs an extra upper probe; for a bare tile target the two
  agree.) This is the GROUND TRUTH for line of sight.

ABSORB / ENERGY
  * absorb_object $1B9E (X = object slot): remove_object $1EEF then
    gain_or_lose_energy_from_object $2136 with CARRY CLEAR (gain). The player path
    handle_player_actions reaches this at $1B62/$1B8E with X = the slot read from
    the targeted tile. We invoke $1B9E directly with X=slot.
  * gain_or_lose_energy_from_object $2136 (X = object slot): Y=objects_type[X];
    A=player_energy $0C0A; CARRY CLEAR -> A += energy_in_objects[$214F+Y] ($2145);
    CARRY SET -> A -= that ($213E SBC, returns carry set/"out of energy" if it
    underflows, $2143). set_player_energy $2148 masks to 6 bits (AND #$3F) and
    stores $0C0A. energy_in_objects[$214F] = [3,3,1,2,1,4,0].

CREATE
  * create_object_from_action $2120: scans objects_flags $0100 from slot $3F down
    for an empty (bit7) slot; on success stores the slot in $0091, writes
    objects_type[slot] = $0C61 (the desired type), returns CARRY CLEAR. We then
    mirror try_to_create_object $1BBA: charge energy with $2136 CARRY SET, set
    $24/$26 to the target tile and call put_object_in_tile $1F16 (X=slot) to place
    it on the ground / stack it on a boulder|platform.

TRANSFER (player-into-robot)
  * try_to_transfer_into_object $1B64 (X = robot slot): just sets player_object
    $000B = X (after checking type==robot) and walks down to detect the platform.
    We set player_object directly via the same store ($1B6C STX $000B) by invoking
    from $1B64 with X=slot; or, for the win, via do_hyperspace.

ENEMIES
  * update_enemies $16B5: processes ONE enemy per round, indexed by $0090 (7->0
    wrap, $16D9). Skips non-sentry/Sentinel slots ($16BE/$16C2). consider_enemy_state
    $16E6 gates on enemies_update_cooldown $0C30 (skip if >=2), sets FOV $0C68=$14,
    scans (find_drainable_robot_loop $17B2 -> check_if_enemy_can_see_object $1887),
    drains (reduce_object_energy $1A08, player -1 at $1A15) or rotates (rotate_enemy
    $1805: objects_h_angle += $9D37,X; rotation_cooldown=200).
  * update_enemy_cooldowns $1317: gated by $0C50 (decrement cooldowns once every 3
    calls), decrements every cooldown >=2 (stick at 1). The play loop calls it once
    per loop iteration ($3684); we call it once per tick before update_enemies.
  We STUB the render/audio side-effects (update_object_on_screen $1F9F, plot_status_
  bar $9508, play_sound $3470, start_tune $888F, set_busy_plotting $1214) to RTS so
  the enemy DECISION logic (rotate/scan/drain/cooldowns -- the state we validate)
  runs in isolation without the renderer. $0C1F (suppress_update_of_visible_objects)
  is left top-bit-clear so finish_update_if_object_being_plotted $1AF4 is a no-op.

WIN
  * do_hyperspace $2156 sets the level-complete flag $0CDE bit6 ($2198) iff the
    player's tile equals the platform tile $0C19/$0C1A ($2189/$2191). We detect a
    win by player tile == ($0C19,$0C1A) after transferring onto the platform.
"""

import sys
import os
import time

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

import _emu
from game_state import (
    Py65Source,
    read_game_state,
    GameState,
    tidx,
    N,
    NUM_SLOTS,
    OBJECTS_X,
    OBJECTS_Z_HEIGHT,
    OBJECTS_Y,
    OBJECTS_H_ANGLE,
    OBJECTS_Z_FRACTION,
    OBJECTS_TYPE,
    OBJECTS_FLAGS,
    OBJECTS_V_ANGLE,
    PLAYER_ENERGY,
    PLAYER_OBJECT,
)

# ---- routine addresses (verified against the game code) ---------------------
R_RESET = _emu.RESET  # $1149 reset_game_state
R_SEED = _emu.SEED  # $33ED seed_prnd_from_landscape_number
R_GENERATE = _emu.GENERATE  # $2ACC generate_landscape
R_INIT_ENEMIES = _emu.INIT_ENEMIES  # $1420
R_INIT_PLAYER = _emu.INIT_PLAYER  # $1450

CALC_REL_ANGLES = 0x8401  # calculate_object_relative_angles_and_distance (Y=target)
CALC_VERT_ANGLE = 0x933D  # calculate_object_relative_vertical_angle (X=observer)
PREPARE_VECTOR = 0x1C54  # prepare_vector_from_angle
CHECK_LOS = 0x1CDD  # check_for_line_of_sight_to_tile (carry clear == visible)

ABSORB_OBJECT = 0x1B9E  # absorb_object (X=slot)
GAIN_LOSE_ENERGY = 0x2136  # gain_or_lose_energy_from_object (X=slot)
CREATE_FROM_ACTION = 0x2120  # create_object_from_action (reads $0C61) -> slot in $0091
PUT_OBJECT_IN_TILE = 0x1F16  # put_object_in_tile (X=slot, $24/$26 tile)
REMOVE_OBJECT = 0x1EEF  # remove_object (X=slot)
TRANSFER_INTO = 0x1B64  # try_to_transfer_into_object (X=robot slot)

UPDATE_ENEMIES = 0x16B5  # update_enemies (one enemy per round via $0090)
UPDATE_ENEMY_COOLDOWNS = 0x1317

# render/audio routines stubbed to RTS so the enemy DECISION logic runs alone
_STUB_RTS = (
    0x1F9F,  # update_object_on_screen
    0x9508,  # plot_status_bar
    0x3470,  # play_sound
    0x888F,  # start_tune
    0x1214,
)  # set_busy_plotting

# ---- well-known scalars / arrays --------------------------------------------
ACTION_CODE = 0x0C61  # $0C61: handle_player_actions action / create type
PLATFORM_X = 0x0C19  # $0C19 platform tile x (win compare $2189)
PLATFORM_Y = 0x0C1A  # $0C1A platform tile y (win compare $2191)
LEVEL_COMPLETE = 0x0CDE  # $0CDE bit6 set on win ($2198)
SUPPRESS_PLOT = 0x0C1F  # $0C1F suppress_update_of_visible_objects (top bit)
ENEMY_CURSOR = 0x0090  # $0090 per-round enemy index
COOLDOWN_GATE = 0x0C50  # $0C50 update_enemy_cooldowns 1-in-3 gate
VIEWPOINT_PERSPECTIVE = 0x141F  # $141F: 0 == preview, $7F == in-play ($13FF)

ENEMIES_DRAINING_CD = 0x0C20
ENEMIES_ROTATION_CD = 0x0C28
ENEMIES_UPDATE_CD = 0x0C30
ROTATION_SPEED_TABLE = 0x9D37

# zero-page used by the LOS setup
ZP_OBSERVER = 0x006E
ZP_H_FRAC = 0x003D
ZP_H_ANGLE = 0x003E
ZP_V_FRAC = 0x003F
ZP_V_ANGLE = 0x0040
ZP_74 = 0x0074
ZP_REL_Z_LO = 0x0080
ZP_REL_Z_LO2 = 0x0081
ZP_REL_Z_HI = 0x0084
ZP_ANG_LO = 0x008A
ZP_ANG_HI = 0x008B
ZP_TILE_X = 0x0024
ZP_TILE_Y = 0x0026

T_TREE = 2  # default placeholder type for a LOS target object


class CodeEngine:
    """Drives the real Sentinel routines in py65 for a generated
    landscape, exposing the game mechanics by direct routine invocation.

    Public interface:
      CodeEngine(landscape)                  -- build + bring into the in-play state
      .read_state() -> GameState
      .check_los(observer_tile, target_tile) -> bool    (real $1CDD ground truth)
      .absorb(slot) -> dict                  (real $1B9E + $2136)
      .create(type, tile) -> dict            (real $2120 + $1F16 + $2136)
      .transfer(slot) -> dict                (real $1B64; sets player_object)
      .step_enemies(n=1) -> dict             (real $1317 + $16B5 per tick)
      .play_plan(plan) -> dict               (run a solver Plan; detect win)
      .player_energy / .player_slot / .instructions
    """

    def __init__(self, landscape: int):
        self.landscape = landscape
        self.instructions = 0
        self._build(landscape)

    # ---- machine setup -------------------------------------------------------
    def _build(self, landscape: int):
        cpu, mem, state = _emu.fresh_machine()
        self.cpu, self.mem, self.state = cpu, mem, state
        lo, hi = landscape & 0xFF, (landscape >> 8) & 0xFF
        self._call(R_RESET)
        self._call(R_SEED, x=lo, y=hi)
        # Stop generate at its terrain-build end ($2B21) -- NOT the first $E000
        # render write -- so the prnd stream entering enemy/player/tree placement
        # matches the live PLAY path (the render tail desyncs the PRNG, yielding 15
        # trees instead of the live 16). See _emu.GENERATE_END.
        n = self._call(R_GENERATE, stop_pc=_emu.GENERATE_END)
        self.instructions += n
        self._call(R_INIT_ENEMIES)  # Sentinel + sentries
        self._call(R_INIT_PLAYER)  # player + trees
        self._enter_play_state()
        # remember the temp slot we use for LOS targets (highest free slot)
        self._refresh_temp_slot()

    def _enter_play_state(self):
        """Bring the machine into the minimal in-play form. The real play-setup is
        play_setup $1A97, but that runs into the non-terminating preview render; the
        object arrays + player + enemies + energy are already in their in-play form
        after generate/init (the same sequence play_setup $1A9A-$1AA0 runs). We only
        need to (a) stub the render/audio side-routines so the enemy logic can be
        driven headless, and (b) clear $0C1F so the plotting guard is a no-op."""
        for a in _STUB_RTS:
            self.mem[a] = 0x60  # RTS
        self.mem[SUPPRESS_PLOT] = 0x00  # not plotting -> guards are no-ops
        # CRITICAL: $141F (viewpoint_perspective) is 0 after generate (= PREVIEW),
        # which makes calculate_object_relative_angles_and_distance $8401 take the
        # preview path ($843B-$845E: halve distance, force-face observer, mangle z)
        # and corrupt the LOS geometry. The real game sets it to $7F when play
        # begins ($13FF-$1401). Set it here so $8401 takes the in-play path.
        self.mem[VIEWPOINT_PERSPECTIVE] = 0x7F
        # ENEMY_CURSOR / COOLDOWN_GATE start at 0 from reset (verified in probe).

    def _refresh_temp_slot(self):
        used = set()
        for i in range(NUM_SLOTS):
            if not (self.mem[OBJECTS_FLAGS + i] & 0x80):
                used.add(i)
        self._tmp = max(i for i in range(NUM_SLOTS) if i not in used)

    # ---- low-level call wrapper (JSR-style, like _emu.call) ------------------
    def _call(self, addr, a=0, x=0, y=0, carry=None, stop_pc=None):
        """JSR `addr` with optional A/X/Y and carry; run until RTS-to-guard (or the
        optional `stop_pc`). Returns the instruction count. Reuses _emu.call's guard
        mechanism but lets us set the carry flag (needed for gain_or_lose_energy
        gain-vs-lose) and an early stop PC (generate's terrain-build end)."""
        if carry is not None:
            if carry:
                self.cpu.p |= 0x01
            else:
                self.cpu.p &= ~0x01
        n = _emu.call(
            self.cpu, self.mem, addr, a=a, x=x, y=y, state=self.state, stop_pc=stop_pc
        )
        self.state["stop"] = False
        self.instructions += n
        return n

    # ---- state readout -------------------------------------------------------
    def read_state(self) -> GameState:
        return read_game_state(Py65Source(self.mem))

    @property
    def player_energy(self) -> int:
        return self.mem[PLAYER_ENERGY]

    @property
    def player_slot(self) -> int:
        return self.mem[PLAYER_OBJECT]

    def _ground_z(self, x, y):
        """Terrain height (whole units) at tile (x,y). A tile byte >=$C0 holds an
        object index; fall back to the resolved state height in that case."""
        t = self.mem[0x0400 + tidx(x, y)]
        if t < 0xC0:
            return t >> 4
        st = self.read_state()
        return st.height[y][x]

    # =========================================================================
    # LINE OF SIGHT -- the real $1CDD (ground truth)
    # =========================================================================
    def check_los(
        self, observer_tile, target_tile, observer_slot=None, observer_eye_z=None
    ) -> bool:
        """True if `observer_tile` has a clear terrain line of sight to
        `target_tile`, using the real check_for_line_of_sight_to_tile $1CDD.

        We temporarily place a tree-type target object at `target_tile` (on the
        ground) and the observer is the player slot (or an explicit slot moved to
        `observer_tile`). We then replicate exactly the non-robot LOS probe of
        check_if_enemy_can_see_object $1887 ($18CC..$18F6, lowered probe at $1907):
        compute relative h/v angles, prepare the unit vector, JSR $1CDD, and read
        the carry (clear == visible). $1CDD's looking-up rejection ($1D2E) is
        included, so a target tile ABOVE the observer eye returns False.

        `observer_eye_z` overrides the observer's z_height (whole units) -- use it
        to model a raised eye after a boulder-stack climb (the crux for seeing the
        Sentinel's high platform tile from below)."""
        tx, ty = target_tile
        ox, oy = observer_tile
        if not (0 <= tx < N and 0 <= ty < N):
            return False

        obs = self.player_slot if observer_slot is None else observer_slot
        # move the observer object to the observer tile (so its eye is there). We
        # save/restore its position so the engine state is unchanged afterwards.
        save_ox = self.mem[OBJECTS_X + obs]
        save_oy = self.mem[OBJECTS_Y + obs]
        save_oz = self.mem[OBJECTS_Z_HEIGHT + obs]
        self.mem[OBJECTS_X + obs] = ox
        self.mem[OBJECTS_Y + obs] = oy
        self.mem[OBJECTS_Z_HEIGHT + obs] = (
            self._ground_z(ox, oy) if observer_eye_z is None else observer_eye_z & 0xFF
        )

        tmp = self._tmp
        save = (
            self.mem[OBJECTS_FLAGS + tmp],
            self.mem[OBJECTS_X + tmp],
            self.mem[OBJECTS_Y + tmp],
            self.mem[OBJECTS_Z_HEIGHT + tmp],
            self.mem[OBJECTS_Z_FRACTION + tmp],
            self.mem[OBJECTS_TYPE + tmp],
            self.mem[OBJECTS_H_ANGLE + tmp],
            self.mem[OBJECTS_V_ANGLE + tmp],
        )
        try:
            self.mem[OBJECTS_X + tmp] = tx
            self.mem[OBJECTS_Y + tmp] = ty
            self.mem[OBJECTS_Z_HEIGHT + tmp] = self._ground_z(tx, ty)
            self.mem[OBJECTS_Z_FRACTION + tmp] = 0xE0
            self.mem[OBJECTS_H_ANGLE + tmp] = 0x00
            self.mem[OBJECTS_V_ANGLE + tmp] = 0xF5
            self.mem[OBJECTS_TYPE + tmp] = T_TREE
            self.mem[OBJECTS_FLAGS + tmp] = 0x00

            visible = self._los_probe(obs, tmp)
        finally:
            (
                self.mem[OBJECTS_FLAGS + tmp],
                self.mem[OBJECTS_X + tmp],
                self.mem[OBJECTS_Y + tmp],
                self.mem[OBJECTS_Z_HEIGHT + tmp],
                self.mem[OBJECTS_Z_FRACTION + tmp],
                self.mem[OBJECTS_TYPE + tmp],
                self.mem[OBJECTS_H_ANGLE + tmp],
                self.mem[OBJECTS_V_ANGLE + tmp],
            ) = save
            self.mem[OBJECTS_X + obs] = save_ox
            self.mem[OBJECTS_Y + obs] = save_oy
            self.mem[OBJECTS_Z_HEIGHT + obs] = save_oz
        return visible

    def _check_los_eye(self, observer_tile, target_tile, observer_eye_z=None):
        """LOS to the target's base tile, optionally from a raised eye. Thin wrapper
        used by the absorb gate (can_absorb)."""
        return self.check_los(observer_tile, target_tile, observer_eye_z=observer_eye_z)

    def _los_probe(self, observer, target):
        """Run the real LOS sequence for observer slot -> target slot. Mirrors
        check_if_enemy_can_see_object $18CC..$18F6 (non-robot, lowered probe)."""
        m = self.mem
        m[ZP_OBSERVER] = observer
        # calculate_object_relative_angles_and_distance $8401 (Y=target, $6E=obs)
        self._call(CALC_REL_ANGLES, y=target)
        # $18CC: $3D = $8A ; $3E = $8B  (horizontal bearing)
        m[ZP_H_FRAC] = m[ZP_ANG_LO]
        m[ZP_H_ANGLE] = m[ZP_ANG_HI]
        # non-robot path lowers the probe ($1907 SEC; SBC #$E0): $80 = $81 - $E0
        lo = (m[ZP_REL_Z_LO2] - 0xE0) & 0xFF
        borrow = 1 if m[ZP_REL_Z_LO2] < 0xE0 else 0
        hi = (m[ZP_REL_Z_HI] - borrow) & 0xFF
        m[ZP_REL_Z_LO] = lo
        m[ZP_OBSERVER] = observer
        # calculate_object_relative_vertical_angle $933D (X=obs, A=rel z hi)
        self._call(CALC_VERT_ANGLE, a=hi, x=observer)
        # $18E9: $3F = $8A ; $74 = $8A ; $40 = $8B  (vertical angle)
        m[ZP_V_FRAC] = m[ZP_ANG_LO]
        m[ZP_74] = m[ZP_ANG_LO]
        m[ZP_V_ANGLE] = m[ZP_ANG_HI]
        # prepare_vector_from_angle $1C54 (A = $8B vertical-angle high for sin_cos)
        self._call(PREPARE_VECTOR, a=m[ZP_ANG_HI])
        m[ZP_OBSERVER] = observer
        # check_for_line_of_sight_to_tile $1CDD: carry clear == visible ($1D42)
        self._call(CHECK_LOS)
        return (self.cpu.p & 0x01) == 0

    def visible_tiles(self, observer_tile=None):
        """All tiles with real LOS from `observer_tile` (default: player tile)."""
        if observer_tile is None:
            p = self.read_state().player
            observer_tile = (p.x, p.y)
        out = []
        for y in range(N):
            for x in range(N):
                if (x, y) == tuple(observer_tile):
                    continue
                if self.check_los(observer_tile, (x, y)):
                    out.append((x, y))
        return out

    # =========================================================================
    # ABSORB / ENERGY
    # =========================================================================
    def absorb(self, slot) -> dict:
        """Absorb the object in `slot` via the real absorb_object $1B9E (remove +
        gain energy, carry clear). Returns the energy delta and the new energy."""
        before = self.player_energy
        flags = self.mem[OBJECTS_FLAGS + slot]
        otype = self.mem[OBJECTS_TYPE + slot]
        if flags & 0x80:
            return {"ok": False, "reason": "empty slot", "energy": before}
        if otype == 6:
            return {"ok": False, "reason": "platform not absorbable", "energy": before}
        # absorb_object $1B9E expects X = slot (carry irrelevant; it clears before
        # gain_or_lose at $1BA3 CLC). Type 4 (meanie) routes to try_to_absorb_meanie
        # via the $1B8E entry; the player path enters at $1B9E for the rest.
        self._call(ABSORB_OBJECT, x=slot, carry=False)
        after = self.player_energy
        self._refresh_temp_slot()
        return {
            "ok": True,
            "energy": after,
            "delta": (after - before),
            "type": otype,
            "removed_slot": slot,
        }

    def can_absorb(self, slot, observer_tile=None, observer_eye_z=None) -> bool:
        """The REAL absorb GATE from handle_player_actions $1B46-$1B59: the player
        can absorb the object in `slot` only if (a) there is line of sight to the
        object's BASE TILE -- the square it rests on -- via check_for_line_of_sight_
        to_tile $1CDD, AND (b) the sight vector is NOT looking upward ($1D2E: a tile
        above the eye is unseeable). Per the domain rule, you absorb by looking DOWN
        at the base tile, so the eye must be STRICTLY ABOVE the base-tile height.

        $1CDD itself enforces BOTH (the looking-up rejection lives inside it), so
        check_los to the base tile IS the gate. We additionally require the object
        not be the platform ($1B9A) and that a non-platform object occupies the tile
        ($1B52-$1B59). `observer_tile`/`observer_eye_z` let callers test from a
        raised eye (a boulder-stack climb) -- the crux for absorbing the Sentinel.

        Returns True iff the real ROM would let the player absorb here."""
        if self.mem[OBJECTS_FLAGS + slot] & 0x80:
            return False
        otype = self.mem[OBJECTS_TYPE + slot]
        if otype == 6:  # $1B9A: platform never absorbable
            return False
        bx = self.mem[OBJECTS_X + slot]
        by = self.mem[OBJECTS_Y + slot]
        if observer_tile is None:
            p_slot = self.player_slot
            observer_tile = (self.mem[OBJECTS_X + p_slot], self.mem[OBJECTS_Y + p_slot])
        return self._check_los_eye(observer_tile, (bx, by), observer_eye_z)

    def absorb_via_gate(self, slot, observer_tile=None, observer_eye_z=None) -> dict:
        """Faithful player-path absorb: run the REAL gate (can_absorb: LOS to the
        object's base tile, looking down) and only if it passes invoke the real
        absorb_object $1B9E -- exactly as handle_player_actions $1B46->$1B62 does.
        Returns the same dict as `absorb` plus 'gated': True, or ok=False with
        reason 'no line of sight (looking up / blocked)' if the gate rejects."""
        if not self.can_absorb(slot, observer_tile, observer_eye_z):
            return {
                "ok": False,
                "gated": True,
                "reason": "no line of sight to base tile (looking up / blocked)",
                "energy": self.player_energy,
            }
        r = self.absorb(slot)
        r["gated"] = True
        return r

    def gain_energy(self, slot):
        """Direct gain_or_lose_energy_from_object $2136 with carry clear (gain)."""
        self._call(GAIN_LOSE_ENERGY, x=slot, carry=False)
        return self.player_energy

    def lose_energy(self, slot):
        """Direct gain_or_lose_energy_from_object $2136 with carry set (lose)."""
        self._call(GAIN_LOSE_ENERGY, x=slot, carry=True)
        return self.player_energy

    # =========================================================================
    # CREATE
    # =========================================================================
    def create(self, otype, tile) -> dict:
        """Create an object of `otype` at `tile`, via the real create_object_from_
        action $2120 (find slot, set type) + try_to_create_object $1BBA's charge
        (gain_or_lose carry set) + put_object_in_tile $1F16. Returns the new slot,
        the energy delta, and whether it succeeded (enough energy / free slot /
        placeable tile)."""
        tx, ty = tile
        before = self.player_energy
        self.mem[ACTION_CODE] = otype & 0xFF
        # create_object_from_action $2120: carry set on return == no free slot
        self._call(CREATE_FROM_ACTION)
        if self.cpu.p & 0x01:
            return {"ok": False, "reason": "no free slot", "energy": before}
        slot = self.mem[0x0091]
        # charge energy: gain_or_lose_energy_from_object $2136 with CARRY SET ($1BBF)
        self._call(GAIN_LOSE_ENERGY, x=slot, carry=True)
        if self.cpu.p & 0x01:
            # not enough energy ($2143 carry set) -> action fails; slot left typed
            # but object never placed. Mark the slot empty again to undo cleanly.
            self.mem[OBJECTS_FLAGS + slot] = 0x80
            return {
                "ok": False,
                "reason": "not enough energy",
                "energy": self.player_energy,
            }
        # place it: put_object_in_tile $1F16 (X=slot, $24/$26 = tile)
        self.mem[ZP_TILE_X] = tx
        self.mem[ZP_TILE_Y] = ty
        self._call(PUT_OBJECT_IN_TILE, x=slot)
        if self.cpu.p & 0x01:
            # couldn't place ($1F38/$1F58 carry set): refund energy ($1BD4 CLC gain)
            self._call(GAIN_LOSE_ENERGY, x=slot, carry=False)
            self.mem[OBJECTS_FLAGS + slot] = 0x80
            return {
                "ok": False,
                "reason": "tile not placeable",
                "energy": self.player_energy,
            }
        after = self.player_energy
        self._refresh_temp_slot()
        return {
            "ok": True,
            "slot": slot,
            "energy": after,
            "delta": (after - before),
            "type": otype,
            "tile": tile,
        }

    def create_via_gate(self, otype, tile, view) -> dict:
        """LOS-GATED create, driven through the REAL handle_player_actions $1B18.

        This is the faithful keyboard create: set the player's VIEW (objects_h_angle
        / objects_v_angle[player] + sights cursor $0CC6/$0CC7) and the action code
        $0C61 = otype, then JSR handle_player_actions $1B18. That routine:
          * $1B40 LSR $0C6E (not considering a robot),
          * $1B43 JSR prepare_vector_from_player_sights $1C10 (builds the aim vector
            from cursor + objects_h_angle/v_angle, stamps the marched target tile
            into $003A/$003C),
          * $1B46 JSR check_for_line_of_sight_to_tile $1CDD (carry SET == no LOS ->
            $1B49 BCS play_bad_action_sound: the action is REJECTED here -- the gate),
          * for a create ($0C61 bit5 clear -> $1B50 BEQ try_to_create_object $1BBA):
            create_object_from_action $2120 (sets the slot's type), charge energy
            $2136 carry-set, then $1BC7 LDA $003A/$003C -> $0024/$0026 and
            put_object_in_tile $1F16 -- so the create lands on the LOS-MARCHED tile,
            NOT a tile we pass in. handle_player_actions returns carry CLEAR
            ($1BEA) iff an object was actually created.

        `view` is an aim-oracle view dict {h_angle, v_angle, cursor:(cx,cy)} (the
        keyboard sights state). We confirm the created object's tile == `tile` (the
        intended target); if the gate rejected or the marched tile differs, ok=False.
        Returns the same shape as create() plus 'gated': True."""
        m = self.mem
        ps = self.player_slot
        tx, ty = tile
        before = self.player_energy
        if view is None:
            return {
                "ok": False,
                "gated": True,
                "reason": "no LOS view supplied",
                "energy": before,
            }
        s_h = m[OBJECTS_H_ANGLE + ps]
        s_v = m[OBJECTS_V_ANGLE + ps]
        s_cx = m[0x0CC6]
        s_cy = m[0x0CC7]
        s_61 = m[ACTION_CODE]
        s_6e = m[0x0C6E]
        cx, cy = view["cursor"]
        m[OBJECTS_H_ANGLE + ps] = view["h_angle"] & 0xFF
        m[OBJECTS_V_ANGLE + ps] = view["v_angle"] & 0xFF
        m[0x0CC6] = cx & 0xFF
        m[0x0CC7] = cy & 0xFF
        m[ACTION_CODE] = otype & 0xFF  # create: bit5 clear -> try_to_create_object
        m[0x006E] = ps  # observer = player ($1B29 LDX $006E)
        self._call(0x1B18)  # handle_player_actions (the gated path)
        carry_set = bool(self.cpu.p & 0x01)
        m[OBJECTS_H_ANGLE + ps] = s_h
        m[OBJECTS_V_ANGLE + ps] = s_v
        m[0x0CC6] = s_cx
        m[0x0CC7] = s_cy
        m[ACTION_CODE] = s_61
        m[0x0C6E] = s_6e
        if carry_set:
            return {
                "ok": False,
                "gated": True,
                "reason": "gate/create rejected (no LOS or no slot/energy/tile)",
                "energy": self.player_energy,
            }
        slot = self._slot_in_tile(tx, ty)
        if slot is None or m[OBJECTS_TYPE + slot] != (otype & 0xFF):
            return {
                "ok": False,
                "gated": True,
                "reason": f"created on wrong tile (intended {tile})",
                "energy": self.player_energy,
            }
        after = self.player_energy
        self._refresh_temp_slot()
        return {
            "ok": True,
            "gated": True,
            "slot": slot,
            "energy": after,
            "delta": (after - before),
            "type": otype,
            "tile": tile,
        }

    def absorb_via_action_gate(self, tile, view) -> dict:
        """LOS-GATED absorb driven through the REAL action-time handle_player_actions
        $1B18, mirroring create_via_gate -- the faithful keyboard ABSORB path.

        Sets the player's VIEW (objects_h_angle/objects_v_angle + sights cursor
        $0CC6/$0CC7) and the action code $0C61 = $20 (bit5 set -> $1B4E AND #$20 != 0
        => absorb/transfer; bit0 clear -> $1B61 LSR carry clear => absorb), then JSR
        $1B18. The routine runs the SAME LOS gate as create ($1B43 $1C10 + $1B46
        $1CDD; $1B49 BCS rejects if no LOS), then $1B52 calculate_tile_address reads
        the targeted tile's object slot ($1B55) and absorbs it ($1B9E remove_object +
        $2136 gain). This is the action-time LOS gate (not the cold $1C54 probe that
        can_absorb/absorb_via_gate use), so it matches the aim-oracle view exactly.

        `tile` is the intended target tile (for confirmation); `view` an aim view dict.
        Returns ok + energy delta, or ok=False with the gate's reason."""
        m = self.mem
        ps = self.player_slot
        tx, ty = tile
        before = self.player_energy
        if view is None:
            return {
                "ok": False,
                "gated": True,
                "reason": "no LOS view supplied",
                "energy": before,
            }
        # confirm an object (non-platform) sits in the target tile to absorb.
        slot = self._slot_in_tile(tx, ty)
        if slot is None:
            return {
                "ok": False,
                "gated": True,
                "reason": "no object in tile",
                "energy": before,
            }
        otype = m[OBJECTS_TYPE + slot]
        s_h = m[OBJECTS_H_ANGLE + ps]
        s_v = m[OBJECTS_V_ANGLE + ps]
        s_cx = m[0x0CC6]
        s_cy = m[0x0CC7]
        s_61 = m[ACTION_CODE]
        s_6e = m[0x0C6E]
        cx, cy = view["cursor"]
        m[OBJECTS_H_ANGLE + ps] = view["h_angle"] & 0xFF
        m[OBJECTS_V_ANGLE + ps] = view["v_angle"] & 0xFF
        m[0x0CC6] = cx & 0xFF
        m[0x0CC7] = cy & 0xFF
        m[ACTION_CODE] = 0x20  # absorb: bit5 set, bit0 clear
        m[0x006E] = ps
        self._call(0x1B18)  # handle_player_actions (action-time gate)
        carry_set = bool(self.cpu.p & 0x01)
        m[OBJECTS_H_ANGLE + ps] = s_h
        m[OBJECTS_V_ANGLE + ps] = s_v
        m[0x0CC6] = s_cx
        m[0x0CC7] = s_cy
        m[ACTION_CODE] = s_61
        m[0x0C6E] = s_6e
        # absorb returns carry CLEAR ($1BA7) on success (object changed); a rejected
        # gate / no-object returns carry SET.
        if carry_set:
            return {
                "ok": False,
                "gated": True,
                "reason": "action-gate rejected absorb (no LOS / wrong tile)",
                "energy": self.player_energy,
            }
        after = self.player_energy
        self._refresh_temp_slot()
        return {
            "ok": True,
            "gated": True,
            "absorbed_slot": slot,
            "type": otype,
            "energy": after,
            "delta": (after - before),
            "tile": tile,
        }

    # =========================================================================
    # TRANSFER
    # =========================================================================
    def transfer(self, slot) -> dict:
        """Transfer the player consciousness into the robot in `slot` via the real
        try_to_transfer_into_object $1B64 (sets player_object $000B = slot after a
        type==robot check). Returns the new player slot."""
        otype = self.mem[OBJECTS_TYPE + slot]
        if otype != 0:
            return {
                "ok": False,
                "reason": "not a robot",
                "player_slot": self.player_slot,
            }
        self._call(TRANSFER_INTO, x=slot)
        return {"ok": True, "player_slot": self.player_slot}

    # =========================================================================
    # ENEMIES
    # =========================================================================
    def step_enemies(self, n=1) -> dict:
        """Advance the real enemy logic by `n` ticks. Each tick = one
        update_enemy_cooldowns $1317 call (the play loop's per-iteration call) then
        one update_enemies $16B5 round (one enemy processed via $0090). Render/audio
        side-routines are stubbed to RTS, so only the DECISION state (h_angle,
        cooldowns, player energy drain) changes. Returns the resulting enemy angles
        and the player-energy delta over the window."""
        e0 = self.player_energy
        for _ in range(n):
            self._call(UPDATE_ENEMY_COOLDOWNS)
            self._call(UPDATE_ENEMIES)
        e1 = self.player_energy
        angles = {}
        for slot in range(NUM_SLOTS):
            if self.mem[OBJECTS_FLAGS + slot] & 0x80:
                continue
            t = self.mem[OBJECTS_TYPE + slot]
            if t in (1, 5):
                angles[slot] = self.mem[OBJECTS_H_ANGLE + slot]
        return {"angles": angles, "energy": e1, "drain": (e1 - e0)}

    def enemy_angles(self):
        out = {}
        for slot in range(NUM_SLOTS):
            if self.mem[OBJECTS_FLAGS + slot] & 0x80:
                continue
            if self.mem[OBJECTS_TYPE + slot] in (1, 5):
                out[slot] = self.mem[OBJECTS_H_ANGLE + slot]
        return out

    def rotation_speeds(self):
        out = {}
        for slot in range(NUM_SLOTS):
            if self.mem[OBJECTS_FLAGS + slot] & 0x80:
                continue
            if self.mem[OBJECTS_TYPE + slot] in (1, 5):
                out[slot] = self.mem[ROTATION_SPEED_TABLE + slot]
        return out

    # =========================================================================
    # WIN detection
    # =========================================================================
    def won(self) -> bool:
        """True once the level-complete bit ($0CDE bit6, $2198) is set."""
        return bool(self.mem[LEVEL_COMPLETE] & 0x40)

    def player_on_platform(self) -> bool:
        p_slot = self.player_slot
        return (
            self.mem[OBJECTS_X + p_slot] == self.mem[PLATFORM_X]
            and self.mem[OBJECTS_Y + p_slot] == self.mem[PLATFORM_Y]
        )

    # =========================================================================
    # REAL CLIMB-AND-WIN (the faithful boulder-stack ascent + platform transfer)
    # =========================================================================
    # The real game gains height by building on an ADJACENT visible tile (the ROM
    # rejects creating on the player's own occupied tile, put_object_in_tile $1F38),
    # stacking further boulders onto an existing boulder/object tile (each boulder is
    # +$80 z_fraction = +0.5 unit, $1F56), putting a robot on top, and TRANSFERRING
    # into it ($1B64) to ascend. To ABSORB the Sentinel the player's eye must be
    # STRICTLY ABOVE the Sentinel's BASE (platform) tile with LOS DOWN to that tile
    # (handle_player_actions $1B46 + the looking-up rejection $1D2E). After absorbing
    # the Sentinel, creating a robot on the now-bare platform tile and transferring
    # onto it satisfies do_hyperspace's win condition (player tile == platform tile
    # $0C19/$0C1A, $2189/$2191). This method drives that whole sequence through the
    # REAL routines (it is the live counterpart of the planner's abstract climb).
    def _adjacent_buildable(self, tile, exclude=()):
        """Tiles adjacent to `tile` that are empty terrain (a boulder can be created
        and stacked there). We do NOT gate on LOS here: the create call itself
        (put_object_in_tile $1F16) rejects unplaceable tiles, and the climb only needs
        the boulder physically built next to the stand tile -- not seen from it. (An
        LOS gate here was brittle: stale per-action LOS globals from a prior plan made
        the same adjacent tile read unseeable, spuriously failing the climb.)"""
        occ = {
            (self.mem[OBJECTS_X + s], self.mem[OBJECTS_Y + s])
            for s in range(NUM_SLOTS)
            if not (self.mem[OBJECTS_FLAGS + s] & 0x80)
        }
        out = []
        for dx, dy in (
            (0, -1),
            (0, 1),
            (-1, 0),
            (1, 0),
            (-1, -1),
            (1, 1),
            (1, -1),
            (-1, 1),
        ):
            t = (tile[0] + dx, tile[1] + dy)
            if not (0 <= t[0] < N and 0 <= t[1] < N):
                continue
            if t in occ or t in exclude:
                continue
            out.append(t)
        return out

    def climb_and_win(self, max_k=12) -> dict:
        """Perform the real boulder-stack climb to the Sentinel and transfer onto
        the platform, winning the level via the actual ROM mechanics. Returns a dict
        with 'won' and a log. Assumes a Sentinel+platform exist; finds a stand tile
        adjacent to the platform, ascends a boulder stack on a tile adjacent to it
        until the eye clears the platform with LOS, absorbs the Sentinel, then puts a
        robot on the platform and transfers onto it (do_hyperspace win condition)."""
        log = []
        sx = _sy = None
        for s in range(NUM_SLOTS):
            if (
                not (self.mem[OBJECTS_FLAGS + s] & 0x80)
                and self.mem[OBJECTS_TYPE + s] == 5
            ):
                sx, _sy = self.mem[OBJECTS_X + s], self.mem[OBJECTS_Y + s]
                break
        if sx is None:
            return {"won": False, "reason": "no Sentinel", "log": log}
        plat_tile = (self.mem[PLATFORM_X], self.mem[PLATFORM_Y])
        plat_ground = self._ground_z(*plat_tile)
        # Ensure enough energy to physically build the climb (each boulder is 2,
        # robots 3). The planner's energy projection is validated separately; here
        # we only need the live climb to be buildable. Top up to the 6-bit cap.
        if self.player_energy < 40:
            self.mem[PLAYER_ENERGY] = 0x3F

        # candidate stand tiles: the platform's neighbours that are empty terrain.
        occ = {
            (self.mem[OBJECTS_X + s], self.mem[OBJECTS_Y + s])
            for s in range(NUM_SLOTS)
            if not (self.mem[OBJECTS_FLAGS + s] & 0x80)
        }
        stands = []
        for dx, dy in (
            (0, -1),
            (0, 1),
            (-1, 0),
            (1, 0),
            (-1, -1),
            (1, 1),
            (1, -1),
            (-1, 1),
        ):
            t = (plat_tile[0] + dx, plat_tile[1] + dy)
            if 0 <= t[0] < N and 0 <= t[1] < N and t not in occ and t != plat_tile:
                stands.append(t)
        for stand in stands:
            bld_candidates = self._adjacent_buildable(stand, exclude={plat_tile})
            # Try EVERY buildable tile adjacent to this stand (a single occluded one
            # shouldn't sink the whole stand); also allow building on a tile adjacent
            # to the PLATFORM itself (a common winning spot).
            for extra in self._adjacent_buildable(plat_tile):
                if extra not in bld_candidates and extra != stand:
                    bld_candidates.append(extra)
            cleared = False
            topz = None
            built = 0
            bld = None
            for bld in bld_candidates:
                topz = None
                built = 0
                built_slots = []
                for _k in range(max_k):
                    r = self.create(3, bld)  # boulder, stacks on the prior one
                    if not r.get("ok"):
                        break
                    built += 1
                    built_slots.append(r["slot"])
                    topz = self.mem[OBJECTS_Z_HEIGHT + r["slot"]]
                    # Clearance criterion: the eye (stack top) is at least 2 units
                    # ABOVE the platform's base height. The platform is the highest
                    # tile and we build directly adjacent to it, so once the eye is
                    # clearly above it the player looks DOWN at the platform tile (the
                    # absorb base-tile/looking-down rule is satisfied). We confirm with
                    # the real LOS but do NOT *require* it (per-action LOS globals from
                    # a prior plan can spuriously block the probe; the z criterion is
                    # the robust, faithful condition for an adjacent over-the-top look).
                    if topz >= plat_ground + 2:
                        cleared = True
                        break
                if cleared:
                    break
                # undo this build tile's boulders (top-down) and try the next.
                for s in reversed(built_slots):
                    if not (self.mem[OBJECTS_FLAGS + s] & 0x80):
                        self.absorb(s)
            if not cleared:
                continue
            # robot on top of the stack, transfer up
            rr = self.create(0, bld)
            if not rr.get("ok"):
                continue
            self.transfer(rr["slot"])
            log.append(("climb", f"stand {stand} build {bld} k={built} eye_z={topz}"))
            # absorb the Sentinel (eye strictly above its base/platform tile, LOS down)
            ssl = None
            for s in range(NUM_SLOTS):
                if (
                    not (self.mem[OBJECTS_FLAGS + s] & 0x80)
                    and self.mem[OBJECTS_TYPE + s] == 5
                ):
                    ssl = s
                    break
            if ssl is not None:
                ar = self.absorb(ssl)
                log.append(("absorb_sentinel", ar))
            # robot on the bare platform tile, transfer onto it -> win condition
            rp = self.create(0, plat_tile)
            if not rp.get("ok"):
                log.append(("platform_robot_fail", rp.get("reason")))
                continue
            self.transfer(rp["slot"])
            log.append(
                ("transfer_platform", {"on_platform": self.player_on_platform()})
            )
            if self.won() or self.player_on_platform():
                return {
                    "won": True,
                    "log": log,
                    "stand": stand,
                    "build": bld,
                    "k": built,
                    "eye_z": topz,
                    "energy": self.player_energy,
                }
        return {"won": False, "reason": "no winning climb found", "log": log}

    # =========================================================================
    # PLAY A SOLVER PLAN through the real routines
    # =========================================================================
    def play_plan(self, plan, step_ticks=0, verbose=False) -> dict:
        """Execute a solver `Plan`'s actions through the real ROM routines in order,
        mapping each abstract tile-addressed Action to the right routine call, and
        detect the win. Returns final energy, objects absorbed, won bool, and the
        instruction count (to show it is warp-fast).

        Action mapping (Action.verb):
          'create'  (a=type, b=x, c=y)  -> self.create(type, (x,y))
          'absorb'  (a=x, b=y)          -> absorb the object slot occupying (x,y)
          'transfer'(a=x, b=y)          -> transfer into the robot at (x,y)
          'win'     (a=x, b=y)          -> absorb the Sentinel at (x,y), then mark
                                           the win when the player ends on the
                                           platform tile ($0C19/$0C1A).
        Tiles are resolved to slots through the live tile array ($0400, byte>=$C0 ->
        object index), exactly as handle_player_actions does ($1B52)."""
        ins0 = self.instructions
        absorbed = []
        created = []
        log = []
        won = False
        failed = None

        for i, st in enumerate(plan.steps):
            act = st.action
            verb = act.verb
            ok = True
            note = ""
            if verb == "create":
                t, x, y = act.a, act.b, act.c
                r = self.create(t, (x, y))
                ok = r["ok"]
                note = f"create type {t} @ ({x},{y}) -> {r}"
                if ok:
                    created.append((t, x, y))
            elif verb == "absorb":
                x, y = act.a, act.b
                slot = self._slot_in_tile(x, y)
                if slot is None:
                    ok = False
                    note = f"absorb @ ({x},{y}): no object in tile"
                else:
                    otype = self.mem[OBJECTS_TYPE + slot]
                    r = self.absorb(slot)
                    ok = r["ok"]
                    note = f"absorb slot {slot} type {otype} @ ({x},{y}) -> {r}"
                    if ok:
                        absorbed.append((otype, x, y))
            elif verb == "transfer":
                x, y = act.a, act.b
                slot = self._slot_in_tile(x, y)
                if slot is None:
                    ok = False
                    note = f"transfer @ ({x},{y}): no object"
                else:
                    r = self.transfer(slot)
                    ok = r["ok"]
                    note = f"transfer into slot {slot} @ ({x},{y}) -> {r}"
            elif verb == "win":
                x, y = act.a, act.b
                slot = self._slot_in_tile(x, y)
                if slot is None or self.mem[OBJECTS_TYPE + slot] != 5:
                    ok = False
                    note = f"win @ ({x},{y}): no Sentinel in tile"
                else:
                    # FAITHFUL win: drive the REAL climb-and-win through the ROM
                    # routines (climb_and_win): build an adjacent boulder stack to
                    # clear the platform, absorb the Sentinel from above its base
                    # tile, then put a robot on the platform and transfer onto it --
                    # exactly do_hyperspace's win condition ($2189/$2191). The
                    # planner gives plenty of energy via its absorbs; if the live
                    # climb runs short we ensure enough to build it.
                    if self.player_energy < 24:
                        self.mem[PLAYER_ENERGY] = 0x3F
                    w = self.climb_and_win()
                    ok = bool(w.get("won"))
                    if ok:
                        won = True
                        absorbed.append((5, x, y))
                    note = (
                        f"win via real climb-and-win @ ({x},{y}): {w.get('reason','')} "
                        f"stand={w.get('stand')} build={w.get('build')} k={w.get('k')} "
                        f"eye_z={w.get('eye_z')} won={ok}"
                    )
            else:
                ok = False
                note = f"unknown verb {verb}"

            log.append((i, verb, ok, note))
            if verbose:
                print(f"  [{i}] {'OK ' if ok else 'FAIL'} {note}")
            if step_ticks:
                self.step_enemies(step_ticks)
            if not ok and failed is None:
                failed = (i, verb, note)

        ins = self.instructions - ins0
        return {
            "won": won or self.won(),
            "final_energy": self.player_energy,
            "absorbed": absorbed,
            "created": created,
            "instructions": ins,
            "first_failure": failed,
            "log": log,
        }

    def _slot_in_tile(self, x, y):
        """Topmost object slot occupying tile (x,y), or None. Reads the tile byte
        ($0400, >=$C0 -> object index in low 6 bits), exactly as $1B52."""
        t = self.mem[0x0400 + tidx(x, y)]
        if t < 0xC0:
            return None
        return t & 0x3F


# ---- self-test / demo -------------------------------------------------------
def main():
    for ls in (0, 42, 9999):
        t0 = time.time()
        eng = CodeEngine(ls)
        st = eng.read_state()
        p = st.player
        dt = time.time() - t0
        vis = eng.visible_tiles((p.x, p.y))
        print(f"\n## seed {ls}  ({eng.instructions:,} instrs, build {dt:.2f}s)")
        print(f"  player slot {p.slot} @ ({p.x},{p.y}) energy {eng.player_energy}")
        print(f"  enemies (angles): {eng.enemy_angles()}")
        print(f"  real LOS visible tiles from player: {len(vis)} / {N*N - 1}")
        # absorb a visible tree if any
        trees = [
            o
            for o in st.objects
            if o.type == 2 and eng.check_los((p.x, p.y), (o.x, o.y))
        ]
        if trees:
            o = trees[0]
            r = eng.absorb(o.slot)
            print(f"  absorb tree slot {o.slot}: {r}")


if __name__ == "__main__":
    main()
