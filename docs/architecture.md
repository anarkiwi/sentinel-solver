# Architecture

Three parts: a model of the game's board, geometry and keyboard aim (`sentinel/`), a driver
that plays the real game through the keyboard (`driver/`), and the real game itself running
cycle-exact on vagrant-moses (`sentinel/emu.py`, `driver/emu.py`), which is the only clock.
The model never advances game time: it builds boards, answers what a keyboard aim can land
on, and prices candidate views so the search can rank them; every outcome comes from the ROM
running on the emulator or in VICE. The 6502 is a test-time oracle for the model; its outputs
are frozen as golden fixtures so CI proves the model without the copyrighted ROM.

This document is the *machine's* structure. The player-facing rules are
[gameplay.md](gameplay.md); the player is [players.md](players.md); everything still wrong
is [open_items.md](open_items.md).

One object generates the rest: a 32×32 height field built from a typed number. What can be
seen, what can be landed on, what can be built on, where an enemy's gaze reaches and what a
frame costs to draw are all queries against that field. So the geometry comes first, and
aiming and render cost follow from it.

- [The landscape and its geometry](#the-landscape-and-its-geometry) — the board, the shared
  coordinate system, and what standing somewhere means.
- [What the geometry permits](#what-the-geometry-permits) — seen ⊃ landable ⊃ buildable.
- [The state machines](#the-state-machines) — the game's cadence, the enemy, the player
  action, the driver, the emulated game, the keyboard aim, driver programs.
- [ROM routines and the model](#rom-routines-and-the-model) — one row per address: routine,
  effect, model function, validation.
- Then the derived subsystems: [the model](#the-model-sentinel),
  [view selection](#view-selection-viewspy),
  [the landing sweep](#the-landing-sweep-landtablepy),
  [kernel specs](#kernel-specs-and-backends-kernpy-kern_cudapy),
  [batched queries](#batched-queries-batchpy),
  [pricing kernels](#pricing-kernels-pricing_jitpy-exposure_jitpy).

---

## The landscape and its geometry

### One number, one board

A landscape has one id: the number a player keys in. `Game.typed(n)` builds it. The ROM
stores that code packed-BCD and seeds `prnd` from those two bytes
(`seed_prnd_from_landscape_number $33ED`), so `landscape.seed_for` reads the typed digits as
hex; `landscape.generate` is the only entry point that takes the raw seed. Generation is
deterministic and consumes a fixed PRNG draw count, so a board is reproducible with no
emulator — the rule-level pipeline is [gameplay.md §3](gameplay.md#3-landscape-generation).

`landscape.generate` ports the ROM's play-setup sequence (`$1A97`) in order: seed → raw
heights → two smoothing passes → scale → spike level → slope nibbles → nibble swap
(`_generate_terrain`, `_process_landscape`, `_smooth_landscape`, `_set_tile_slopes`), then
`_initialise_enemies` (`$14FB`) and `_initialise_player_and_trees` (`$1450`). Draw order
matters more than the arithmetic: a single extra `prnd` call moves every object.

| landscape | `Game.typed(n)` |
|---|---|
| `42` | player (13,29), 2 enemies, 16 objects |
| `335` | player (11,17), eye 3.875, **7 enemies** (Sentinel (28,17) h12 + 6 sentries) |

```python
from sentinel import Game
g = Game.typed(42)                   # no emulator
g.player_xy(), g.energy              # ((13, 29), 10)
```

### The tile map, and the one coordinate system

Everything — terrain, LOS, actions, the renderer — reads one 64 KB `bytearray`
(`state.State`) laid out exactly like the game's RAM, so a boulder placed by `actions.create`
is immediately an occluder for the next `los` march. `memmap.py` holds every address; the
object arrays are live `_ObjArray` views over `mem`, never snapshots. The emulator's RAM is
the same layout, so `State.from_mem` reads a live or emulated machine directly.

| axis | where it lives | units |
|---|---|---|
| x, y | `objects_x $0900`, `objects_y $0980`; tiles table `$0400-$07FF` | tile indices 0..31; object placement uses 0..30 (`$1272` rejects 31) |
| z | `objects_z_height $0940` + `objects_z_fraction $0A00` | whole tile-height units 0..11 plus /256 fraction; `state.eye_z` is the pair |
| bearing | `objects_h_angle $09C0` | 256 units = 360° |
| pitch | `objects_v_angle $0140` | same units, clamped to `[$CD..$FF] ∪ [$00..$35]` |

The tile map is **interleaved, not row-major**: `calculate_tile_address $2BA8` puts tile
`(x,y)` at `$0400 + 256·(x&3) + 8·(x>>2) + y`. `memmap.tidx` is that formula and
`terrain.tile_byte` is the ROM's masked 8-bit form, kept because `check_sloping_tile` reads a
tile's four *corner* tiles, some of which sit off the interior and must wrap byte-for-byte
the way the 6502 does.

A tile byte below `$C0` is terrain, `(height<<4) | slope`; `>= $C0` it holds the slot of the
**topmost** object in that tile and the ground there is the *bottommost* object's height
(`terrain.resolve_ground`, `terrain.bottom_object`). So the same byte answers "how high is
this" and "what is standing here", and an object tile carries no slope nibble.

Height is a **corner field**, not a per-tile plateau. A tile's surface is interpolated from
the heights of `(x,y) (x+1,y) (x+1,y+1) (x,y+1)`, split into two triangles by the slope
nibble (`los._slope_corner_z`, `los._check_sloping_tile`, ported from `check_sloping_tile
$1D46` and the edge table `$1DF1`). Averaging the four corners instead over-estimates
visibility.

Vertically the whole model works in the ROM's 16-bit compare pair `z*256 + fraction`
(`$003B:$0038`): eye height, ray height and surface height are all that form, which is why
`landtable`'s landing condition can be written as one closed-form inequality.

### Standing somewhere

The player is not a camera, it is an object slot: `player_object $0B` indexes the same 64
arrays every tree and boulder lives in. So a stance is a slot, and its geometry is that
slot's row:

- **eye height** = `obj_z_height + obj_z_frac/256` (`state.eye_z`). A robot on the ground
  sits at the tile height with `z_fraction = $E0`; `views.ROBOT_EYE = 0.875` is that
  offset.
- **the body** occupies the tile, blocks rays like any other object, and is drainable while
  it is a type-0 robot — including after you leave it.
- **height is bought, never found.** `put_object_in_tile $1F16` accepts a create on bare
  ground, on a **boulder** (`+½` unit, `views.BOULDER_H = 0.5`) or on the **platform**
  (`+1`); `$1F38` refuses every other occupied tile. The only ladder is boulder → transfer →
  boulder, and `los._get_object_details` (`$1ECC`) re-seeds every ray at the new slot's tile
  centre and eye z.

A hop is therefore one move, not three: `moves.moves` offers `k` boulders, the robot and the
transfer as a single candidate, and `kbd_prog.compile_move("hop", ...)` plays them as one
program ([gameplay.md §7](gameplay.md#7-how-a-human-wins-quick-strategy)).

## What the geometry permits

Three questions, strictly nested. Each is the previous one plus a constraint the *machine*
imposes, and the gap between them is where most of the model's work is.

| question | test | ROM | model |
|---|---|---|---|
| Is there a clear line to this tile? | one ray march from the eye | `check_for_line_of_sight_to_tile $1CDD` | `los.check_for_line_of_sight_to_tile`, `los_jit` |
| Can a **keyboard aim** stop on it? | some lattice ray must *land*, not merely pass | `$1C10` + the sights clamps `$9965`/`$9994` | `landtable.landing_rays`, `batch.landings`, `views._Views` |
| Can I build on it? | landable **and** the stacking rule | `put_object_in_tile $1F16`/`$1F38` | `actions.create`, the `stack` mask of `moves.moves` |

**Seen.** One ray march serves the player and every enemy; they differ only in observer and
in how the aim vector is built. `$1CDD` seeds at the observer tile's centre (`$1ECC`), steps
≈1/16 tile per iteration (`add_vector_to_object_position $1CBB`), and at each step compares
the ray's 16-bit z against the tile's surface: below → march on, inside the `$000C` = `$80`
band → **land here**, above → **blocked**. It quits when a coordinate reaches `$1F`. The
reached tile is stamped into `$003A`/`$003C` and *that* is what an action fires on. Two
asymmetries matter: the look-up rejection (`$1D2C`-`$1D32`) blocks a flat tile at or above
eye level unless the target is an object *top*, and object tops require the ray to thread
near the tile centre (`$1EAF`), tightening the tolerance to `$10` for a robot or platform
(`$1E2C`).

**Landable is stricter than seen**, and this is the distinction the whole solver turns on.
The keyboard cannot aim anywhere: the body pans on an 8-unit bearing lattice and a 4-unit
pitch lattice inside the clamp, and the sights cursor moves ±1 px per gated scan within
cx `$10-$8F` / cy `$20-$9F` (`move_sights $9958`, `$9965`/`$9994`). `$1C10` turns each 1-px
cursor step into a distinct ray sub-angle, so the reachable aims are a finite lattice
(`los.CURSOR_CX`/`CURSOR_CY`, `PITCH_BAND`, `AZIMUTH_STEP`) — and a tile plainly visible on
screen is unbuildable if no lattice ray *stops* inside it. Answering that is a march over
262,144 rays for the `$F5` plane and 3,538,944 for the full pitch band, which is why
[the landing sweep](#the-landing-sweep-landtablepy) walks each ray's track in closed form
before marching any.

**Buildable is stricter again.** Beyond the stacking rule, a **sloping tile never lands** —
`check_sloping_tile $1D46` only continues or blocks, it has no hit case — and the observer's
own tile never lands (`$1D32`). Both are exact shortcuts in the sweep (`landtable.NEVER_LANDS`).
So the build set is: bare *flat* tiles, boulder tops, and the platform.

### What follows from the geometry

- **Aiming** is the cost of moving along that lattice: bearing notches and pitch notches each
  cost a `pan_viewpoint $10B7` redraw, cursor pixels cost gated scans. Priced in
  [view selection](#view-selection-viewspy), which picks the view a move is played on.
- **An enemy's gaze** is the same march plus a heading gate: the target must lie within
  `±($0C68/2)` of the enemy's *current* facing (`$1887`/`$18B8`), and `$18E6` runs `$1CDD`
  twice — at the target's top and `$E0` lower — to get full / partial / unseen
  (`relative.can_see_object`, `sight_jit._can_see_object`, `threat.player_sees_tile`).
- **Draining** is what a cone plus full sight buys the enemy after a countdown; see
  [the enemy machine](#the-enemy).
- **Occlusion and render cost** walk the same height field. `populate_tile_visibility_bit_table
  $245B` raytraces the field into a bitmap that only zeroes a tile's plot byte, so an occluded
  tile is still *examined* — it pays the trig floor and not the fill. That split is the
  [render-cost model](#pricing-kernels-pricing_jitpy-exposure_jitpy), and render cost is
  time, because the game is not frame-locked.

---

## The state machines

Four machines run the game, and more run the driver. Every edge below is a ROM branch with
an address. The game's machines run only on the real ROM — on vagrant-moses or in VICE — and
the model holds none of them; they are listed because the move set, the move programs and
their stop points are built on them.

### The game's cadence

The game is **not frame-locked**. `update_game_and_continue $363D` calls `update_game $127C`
with no vsync wait and re-enters `play_landscape_loop $357D` on carry, so the foreground rate
is compute-bound and dominated by the 3-D redraw. `update_game_loop $1289` calls
`update_enemies $16B5` once per pass, repeatedly, until an enemy forces a visible replot.
The fixed cadence is the raster: `$9663` (or the scroll loop `$3684`, the two being mutually
exclusive) runs `update_enemy_cooldowns $130C` once per video frame, a Bresenham divider that
decrements the cooldown arrays on a fraction of frames. How many enemy updates a frame holds
therefore depends on what the foreground is drawing, the badlines it meets and the board's
enemy count — which is why game time is taken from a cycle-exact emulator and not modelled.

### The enemy

An enemy is a per-slot state machine driven by three cooldown bytes, serviced round-robin —
`update_enemies $16B5` handles **one** slot per call via the cursor `$0090` (7→0).
`is_sentinel_or_sentry $16C6` sends the Sentinel and sentries down the identical path; the
Sentinel's only edges are positional.

| counter | array | reload | set by | meaning |
|---|---|---|---|---|
| update | `$0C30` | 4 | `$16ED` (gate `$16E9`) | how often the ladder below runs at all |
| | | 30 | `$17F1`/`$1848` | after a drain or downgrade |
| | | 10 | `$173A` | after a meanie turns |
| | | 50 | `$1869`/`$186E` | after a meanie is spawned |
| | | `(prnd & $3F)` OR 5 | spawn | initial phase |
| rotation | `$0C28` | 200 | `$1813` | one ±20-unit step per period |
| draining | `$0C20` | 120 | `$1835` | sight *arms* it; 0 = not targeting |

Sight alone costs nothing. `target_object $1825` merely arms `$0C20` to 120 rounds, and
`$1A31` re-zeroes it after each drain, so a cone sweeping across a body is free for its first
120 rounds.

**The ladder.** `consider_enemy_state $16E6` runs only when the update cooldown is below the
stick value, and **returns before the rotate at `$17F9` whenever an earlier branch fires** —
so a busy enemy stops sweeping.

| # | state entered | trigger | ROM | effect / next |
|:-:|---|---|---|---|
| — | **frozen** | player has not acted | `$0CE5` bit7; skipped at `$3682`/`$9659` | no enemy update until `$12E1` clears it |
| — | **absorbed** | slot empty | `$16B5` | discharge only, forever |
| 0 | **idle** | `$0C30 >= 2` | `$16E9` | return; the cooldown tick is the only edge |
| 1 | **driving a meanie** | `$0CA0` bit7 clear (owns one) | `$16EA` → `update_meanie $16F2` | no scanning of its own this update |
| 2 | **discharging** | `$0C88` non-zero | `$1775 JSR $1A5D` | emits a tree in a random low tile; `$177A` returns before the rotate |
| 3 | **hunting for a meanie** | `$0CB8` bit7 armed | `$177F` → `$1784 JSR $1AB0` | on a hit, drain it; else the flag decays `$80→$40` |
| 4 | **holding a target** | `$0C20 != 0` and still visible | `$178C` | re-`target_object`; lost sight → `$0C20 = 0` |
| 5 | **scanning** | nothing above fired | `find_drainable_robot_loop $17B2`, slots 63→0 | full sight of a type-0 robot → target it |
| 5a | **half-sees the player** | head-only exposure on the player | `$17C0`/`$17C4` | remembered unless `$0C90` already blames that slot |
| 6 | **dismantling** | a boulder, or anything with `flags >= $40` | `$1AB0` (`$17E0`) | drain it; update cd ← 30 |
| 7 | **rotating** | `$0C28 < 2` | `rotate_enemy $1805` | `+= $9D37[slot]` (fixed ±20), rotation cd ← 200 |

Seeing is two tests, both in `check_if_enemy_can_see_object $1887`: the target must lie
within `±($0C68/2)` of the *current* facing (`$0C68 = $14` scanning, `$28` for meanie search;
the halving is `$18B8 LDA $0C68 / LSR`), and
`check_if_enemy_has_line_of_sight_to_object $18E6` calls `$1CDD` twice — at the target's top
and `$E0` lower — writing `$0014`: `$80` full, `$40` partial, 0 unseen. A robot seen through a
tree is skipped (`$17B2` tests `$0C76` bit6). Only **full** sight drains (`$183F`/`$1841`).

**What a drain does.** `reduce_object_energy $1A08` reads its target from `$0C58`: the player
loses 1 energy, or dies if the meter is already 0 (`kill_player $1A00` sets `$0C4E` bit7);
any other object is **downgraded** one step, robot → boulder → tree → gone. Every drain banks
+1 into `$0C88` for later re-emission as a tree, so board energy is conserved.

**The meanie sub-machine.** A *partially* visible player is the meanie path (`$184D` →
`$1852` → `consider_creating_meanie $1986`).

| state | trigger | ROM | next |
|---|---|---|---|
| considering | drain countdown expires on a half-seen player | `$1852` | scan for a tree |
| scanning for a tree | FOV widened to `$28`, tree within 10 tiles in x **and** y, fully seen | `$19A1`, `$19C3`/`$19D5` | flip its type to 4 in place — no slot allocated (`$19F0`) |
| deferred | the scene is being plotted | `pause_meanie_creation $19FA` | retry next tick |
| given up | 2 scans found nothing | `$1857` | `$0C90` remembers the player slot; stop retrying |
| driving | a meanie exists | `update_meanie $16F2` | rotate ±8 (`$1728`) toward the player, update cd ← 10 |
| firing | bearing within one screen width **and** `$0014 != 0` | `$171B`/`$171D JSR $2156` | forces a hyperspace on the player |
| dissolved | it fired, the bound body was absorbed (`$1707`), or the player transferred out (`$1717`) | `remove_meanie $1754` / `$174F` | back to a tree |

The parent enemy holds at most one meanie and does nothing else while it lives. Absorbing the
meanie (`try_to_absorb_meanie $1BEC`) is +1 energy and clears the link.

### The player action

An action is not an instant: the world keeps running while the player aims, fires and the
screen settles. On the ROM one action is this sequence, which a move program reproduces key
for key ([driver programs](#driver-programs-driverprogpy-driverkbd_progpy)) and
[view selection](#view-selection-viewspy) prices to rank views:

| segment | ROM | runs `$16B5`? |
|---|---|---|
| sights toggle | `initialise_sights $134C` re-centre + `plot_sights` (none on a same-bearing reuse) | no |
| u-turn taps | action code `$23`: bearing `⊕ $80`, tune `#$28`, the `$357D` redraw less `$245B`/`$3700` (`$35B5 BMI`) | **yes** |
| pan notches | `$10EE`/`$1135` scroll + one `plot_world` per notch | no |
| cursor drive + firing tap | gated `move_sights`/`tap_action` scans | **yes** |
| settle | `$12EC` strip replot + `$86A5` dither, or the `$357D` redraw | no |

Two freezes decide which segments the enemies see.

- **The enemy freeze `$0CE5` bit7** holds until the player's *first* action. `consider_player_action
  $12D0` normally requires the sights to be active, but `$12D5 CMP #$22 / BCS` lets codes
  `>= $22` skip straight to `$12E1 LSR $0CE5`. So a u-turn — code `$23`, free, no LOS, no
  energy — **unfreezes the world mid-aim**.
- **The plot freeze `$0CE4` bit7** (`set_busy_plotting $1214`) is held across a whole viewpoint
  redraw; while it is set the foreground never reaches `update_enemies`, so only the raster
  cooldown tick advances.

Dispatch is `handle_player_actions $1B18` off the code latched in `$0C61`/`$0CE9`; for every
code `< $22` it builds the aim vector and calls `$1CDD`, and carry set at `$1B40`-`$1B46`
plays the bad-action sound and does nothing. `$1281` zeroes the action latch `$0C51` and only
an idle full scan re-arms it at `$11EA`, so one press fires at most one action. The code →
routine map is [gameplay.md §4](gameplay.md#4-actions).

### Hyperspace, death and the win

Three flag bits carry the terminal states, and the same routine produces all of them.
`do_hyperspace $2156` creates a robot on a random flat tile no higher than `player_z + 1`
(`$215D` adds 1, `$2165 JSR $1238`), then spends the 3-energy toll (`$216A`/`$216B JSR $2136`).

| state | bits | set by | read by |
|---|---|---|---|
| alive | `$0CDE & $C0 == 0`, `$0C4E` bit7 clear | — | `kbd_prog.move_facts` `alive` |
| drained to death | `$0C4E` bit7 | `kill_player $1A00`, a drain arriving at 0 energy | `Prog.guard` stop at `kbd_prog.KILL_PC` |
| died on the toll | `$0CDE` bit7 set, bit6 clear | `$216E`/`$2170`, underflow on the 3-energy spend | `Prog.guard` stop at `kbd_prog.TOLL_PC` |
| landscape complete | `$0CDE == $C0` | `player_survived_hyperspace $217F` / `landscape_completed $3603`, jumping from the platform tile `$0C19`/`$0C1A` | `kbd_prog.WIN_PC`, `move_facts` `won` |

Standing on the platform is not a win; hyperspacing from it is. Live, the driver reads
`$0CDE` bit 6 straight out of VICE (`landscape_done` in the run's JSON).

`$216A` is also the only move that changes position with **no line of sight**: it takes no
aim and no target, so it is the one action still available from a stance that can land
nothing. `moves.moves` therefore offers the relocation wherever the toll is in hand; its
landing is PRNG-driven (`$2165 JSR $1238`), and the search does not predict it — the child
is whatever the machine does.

### The driver: boot → enter → play

`driver/boot.py` and `driver/core.py` walk a fixed stage sequence, every transition gated on
a memory or PC predicate rather than a host clock.

| stage | function | ends when | on failure |
|---|---|---|---|
| container launch | `boot.boot_loaded` | `BinMon` connected to the container's bridge IP | relaunch (tape timing under warp can JAM the 6502) |
| tape load | `boot.wait_for_load` | `SIG_BYTES A5 0B 85` readable at `SIG_ADDR $35A4` | retry the whole container, up to `attempts` |
| snapshot | `save_boot_snapshot_if_missing` | `boot.vsf` written via `MON_CMD_DUMP` `0x41` | — |
| title → code entry | `core.navigate` | three `SPACE` taps, then `CODE_PATCHES` applied at `$14DF`/`$2565`/`$2570` | — |
| code entry | `core.navigate` | digits + RETURN; `landscape_from_digits` parses them as **hex** | — |
| generation | `core._enter_play` leg 1, `_generated` | `$0C0A` non-zero — the player object is installed | `RuntimeError`: generation never installed a player |
| preview → play | `core._enter_play` leg 2, `_in_play` | `$0CE4` bit7 released | `RuntimeError`: play never started |
| play | `core.boot_and_play` / `GameSession` | the run ends; AVI finalized in a `finally` | — |

`vice_code_entry.vsf` is restored when present, skipping the tape boot entirely.

### The emulated game (`sentinel/emu.py`, `driver/emu.py`)

The same stack plays the CSDb release offline on vagrant-moses, a cycle-exact PAL C64
(VIC-II badlines and raster IRQ, both CIAs, the 6510; native, GIL released).
`emu.boot()` powers on with the real KERNAL/BASIC/CHARGEN, RUNs the release's PRG from
BASIC's keyboard buffer, holds SPACE through its intro until the loader jumps to
`$3F00`, and stops at the title's first keyboard read `$8D17`; `emu.enter(n)` drives
`core.navigate` from there (three SPACE taps, `CODE_PATCHES`, digits, code, preview) and
checks `verify_entry` object for object. Both are cached templates handed out as
`clone()`s. `python -m driver.play_player 335 --emu` builds the `GameSession` from
`emu.enter` and plays the [search player](players.md) on it; `sentinel.tests.csdb` builds
the oracle image from `emu.boot()`.

`EmuMon` answers exactly the BinMon calls the driver makes. Its rules are VICE's
(`6510dtvcore.c` `DO_INTERRUPT`, `mon_breakpoint.c`, asid-vice's keymatrix core):

| call | semantics |
|---|---|
| `run_until_pc(pc, timeout)` | returns at once when already at `pc`; else a stop-on-hit checkpoint, so the CPU stops **before** executing `pc` |
| `advance_instructions(n)` | `n` decodes; the step test precedes the checkpoint test, so a step ending on a checkpointed pc counts and stops there only on the next resume |
| `checkpoint_set/get/toggle/delete`, `wait_for_checkpoint(s)` | exec checkpoints with `hit_count`; a silent one only counts (`clock.FRAME_PC`) |
| `mem_get(…, bank)` | bank 0: the CPU view through `$01`; `"io"`: RAM with the I/O page peeked side-effect free; `"ram"` |
| `keymatrix_set/release_all/tap` | the CIA 1 matrix, row = PA line, col = PB line; a fixed tap releases `frames · 19656` cycles after the press |
| timeouts | `timeout` seconds of PAL cycles, then `BinmonError`; a stop at another checkpoint also raises |
| `clone()` | `Machine.clone()` plus every checkpoint, count, owed boundary test and held tap |

A clone is a fork of the game: the search expands each parent by running its children's
programs on clones (`prog.forks`), so a child's outcome is the ROM's own and takes the game's
own time. Live, the search plans on an emulated session of the same landscape and every
committed move is also interpreted on VICE and its facts compared
([players](players.md#live-in-vice)); VICE times a wait in host seconds and the emulator in
PAL cycles, so only a wait that expires can split them. `python -m driver.compare_runs
A.json B.json` reports each audit's first divergent row.

### The keyboard aim primitive

`driver/kbd_aim.py` drives one aim as sub-machines, each ending on state and never on
elapsed time. Every outcome below is a literal returned by the code.

| sub-machine | sights | steps on | ends `ok` | other exits |
|---|---|---|---|---|
| `coarse_h` | off | `PC_PAN_DONE $365D` | `$09C0+slot == want` | `hyperspace` (`$0B` changed mid-aim), `unreachable` (angle unchanged for `_PAN_STALL_FRAMES`, or `_PAN_MAX_FRAMES` spent) |
| `coarse_v` | off | `PC_PAN_DONE` | `$0140+slot == want` | `unreachable` immediately if `want` is outside `[$CD..$35]` |
| `fine_cursor` | on | the per-axis store PCs `$997C`/`$9990`/`$99B8`/`$99D2` | `$0CC6`/`$0CC7` reached | `False` after 160 iterations |
| `tap_action` | — | the gated scan `$9678`→`$967B` | the want-flag `$0CE8..$0CEB` latched | not latched → retry, up to 45 times |
| `sights_set` | — | `_one_scan_press("SPACE")` | `$0C5F` bit7 == requested | up to 6 presses |

`coarse_h` takes the shorter of ±8 steps or a u-turn plus a correction (`aimcost.h_press_count`),
confirming the u-turn by `(angle_after − angle_before) & $FF == $80`. Fine cursor presses
while halted and releases after the store, because a held key auto-repeats in an accelerating
burst (`$11F6 ASL $0CC8`). `_run_to_scan` treats a timeout while `$0CE4` bit7 is set as a
redraw still running and re-arms; conceding there would leak the redraw's frames into the next
primitive.

### Driver programs (`driver/prog.py`, `driver/kbd_prog.py`)

Every primitive above, and a whole move, is compiled to one vagrant-moses driver program
(`vagrant_moses.program`, sixteen registers, run natively with the GIL released) with the
interactive driver's exact semantics: the same keys, PCs, retry bounds, stop points and
frames. `kbd_prog.Compiler` holds one compiler method per `kbd_aim` primitive.

A `Prog` is the vagrant-moses op set plus one extension, the timed wait `run(pc, seconds,
on_timeout)` (`run_until_pc` with its timeout), with `catch` blocks standing for the driver's
`try/except`. Three entry points execute the same `Prog`:

| backend | how | timeouts |
|---|---|---|
| `prog.interpret(bm, p)` | call by call over any BinMon: VICE, or `EmuMon` | exact: a raise branches to the op's handler |
| `prog.execute(emumon, p)` | `machine.run_program` on the native lowering | each wait checks its elapsed cycles; past its timeout the program is replayed by `interpret` from a clone taken before it |
| `prog.forks(emumon, ps)` | one `vagrant_moses.run_forks` batch, `keep=True`: one `EmuMon` per fork | as `execute`, per fork |

The lowering reproduces `EmuMon`'s monitor state: a wait for the pc the CPU stands at is skipped
before any run op (after one, the native parked state is that test), the frame checkpoint gains
the marker arrivals, and a program ending on a step leaves the boundary test owed.

| primitive | program |
|---|---|
| `sights_set` / `one_scan_press` | idle scan, step, scan, SPACE, run to `$967B`, release; ≤ 6 presses on `$0C5F` bit7 |
| `uturn` | `tap_action U` until `(h − h₀) & $FF == $80`, 5 tries |
| `coarse_h` / `coarse_v` | u-turn when `h_press_count` takes one; per pass the key chosen by `(want − cur) & $FF ≤ $80` (pitch: `_pitch_lin`), compiled as range tests over all 256 angles; stall, slot-change and pass bounds as `kbd_aim` |
| `run_pass` | `$363D`, step, `$365A`, step, `$365D`, each a `run_past_redraw` (re-arm while `$0CE4` bit7) with raster stamps |
| `fine_cursor` | per pass x and y keys, the last axis's store PC, step; 160 passes, a timeout skips the pass |
| `tap_action` | idle scan, press at the scan, run to `$967B`, want-flags `$0CE8..$0CEB`, settle to the next gated scan; 45 tries |
| `run_frames` | step + `$9630` wait, `n` times |
| `probe_settle` | 2 frames while `$0CE4` bit7, ≤ 8 times |
| `aim` + `fire` | the full aim with sights-off h/v reads, the settle, then sights on and the tap between frame reads |

`kbd_prog.compile_move(kind, view, k, frames, reuse)` compiles a macro move for the search to
expand by `prog.forks`: `"absorb"`, `"mount"` (a transfer), `"hop"` (`k` boulders, the robot,
the transfer), `"wait"`, `"relocate"` (one H tap, settled) or `"finish"` (the platform
hyperspace attempt). `view` is one view key or one per action; an action whose bearing is the
previous action's (or, with `reuse`, the sights') drives the cursor only, any other takes the
full aim (the ray is assumed to land; the facts tell whether it did); a refusal ends the move.
`Prog.guard(pcs, check)` makes every wait in its block also stop before `pcs` and runs
`check`'s ops after each wait, on both backends: a move's waits stop at `$1A05` (past
`kill_player`'s `STA $0C4E`), `$217D` (past the toll underflow's `STA $0CDE`) and `$219B`
(past the win's `STA $0CDE`) and end the move (`H_DIED`, or the win), so a death never
runs a wait into its hang guard. Interpreted, a guarded wait is one resume over the wait's
own stop checkpoint and the guard pcs' (`prog.wait_any`: `EmuMon.wait_for_checkpoints`, or
the binmon event stream on VICE), so on VICE too a guard stop returns at once instead of
waiting out the host timeout. `move_facts` decodes `status`, `energy`, `slot`, `z`, `z_frac`,
`enemies` (sentries and Sentinel left), `alive` (`$0C4E` and `$0CE5` bit7 clear), `won`
(`$0CDE` bit6) and `frames`.

---

## ROM routines and the model

The ROM addresses the model and the driver port, price or stop on, sorted. Names are the
ROM's own labels; a row without a label names the routine the address sits inside. The last
two columns are the mapping — the function or constant that ports, prices or depends on the
routine, and how that is proved (a golden fixture, a test, or the numbered open item where it
is approximate). An empty model cell is a ROM fact the code reads only through the running
game.

| Address | Routine | What it does | Model / driver (`module.name`) | Validation / approximation |
|---|---|---|---|---|
| `$0D03`/`$0D05` | `multiply_byte_by_byte` | 8×8 unsigned multiply, the trig primitive | `los._mul8` | `golden_los` |
| `$0D4A` | `divide_and_arctan` | shift/subtract 16-bit divide, then arctan of the quotient | `relative._divide_and_arctan` | `golden_relative` |
| `$0E75` | `sin_cos_lookup` | polynomial sine/cosine of a byte angle (256 = full circle) | `los.sin_cos_lookup` | `golden_los` |
| `$0F3E` | `multiply_double_A_by_pi` | scale a 16-bit value by π | `los.multiply_double_A_by_pi` | `golden_los` |
| `$0F4A` | `multiply_double_by_byte` | 16×8 fixed-point multiply | `los.multiply_double_by_byte` | `golden_los` |
| `$0F9E` | `multiply_double_by_double` | signed 16×16 fixed-point multiply | `los.multiply_double_by_double` | `golden_los` |
| `$1090` | `fill_screen_with_background` | clears the play buffer before a replot | `settlecost.FILL_1090` (108935 cycles, constant) |  |
| `$10B7` | `pan_viewpoint` | one keyboard pan notch: strip clear, one `plot_world` at the intermediate angle, queue the scroll | `pancost.notch_frames` | `golden_pan_cost` |
| `$10EE` | (in `pan_viewpoint`) | horizontal scroll — 16 steps per ±8 bearing notch | `views.H_SCROLL` | `golden_pan_cost` |
| `$1135` | (in `pan_viewpoint`) | vertical scroll — 8 steps per ±4 pitch notch | `views.V_SCROLL` | `golden_pan_cost` |
| `$1149` | pitch limits table | clamps the pitch band to `[$CD..$FF] ∪ [$00..$35]` | `los.PITCH_BAND` | `test_landtable.py`; [3](open_items.md#3-landability-filter-unproven-corners) |
| `$119F` | `check_for_full_player_input` | the full key scan, including the SPACE edge test |  | `test_kbd_scan_gate.py` |
| `$11E0` | (in the input scan) | reloads the cursor auto-repeat mask `$0CC8` with `#$6B` | `views.CURSOR_REPEAT_MASK` | [2](open_items.md#2-the-drivers-wall-clock-timeouts-are-the-residual-load-sensitivity) |
| `$11EA` | (in the input scan) | an idle full scan re-arms the action latch `$0C51` | `kbd_aim.tap_action` | [2](open_items.md#2-the-drivers-wall-clock-timeouts-are-the-residual-load-sensitivity) |
| `$11F6` | (in the input scan) | `ASL $0CC8 / BCS` — one gated scan skipped per set mask bit | `views.CURSOR_RAMP` | [2](open_items.md#2-the-drivers-wall-clock-timeouts-are-the-residual-load-sensitivity) |
| `$1214` | `set_busy_plotting` | sets `$0CE4` bit 7 for the duration of a redraw | `kbd_aim._run_to_scan` |  |
| `$125A`/`$1272` | `get_random_tile_coordinate` | a `prnd` draw masked to 0..31, rejecting 31 | `landscape._random_tile_coord` | `golden_landscape` |
| `$127C` | `update_game` | the per-pass game update |  |  |
| `$1281` | (in `update_game`) | zeroes the action latch `$0C51` | `kbd_aim.tap_action` | [2](open_items.md#2-the-drivers-wall-clock-timeouts-are-the-residual-load-sensitivity) |
| `$12D0` | `consider_player_action` | requires the sights active before create/absorb/transfer; dispatch + handler bookkeeping is `settlecost.HEAD_BASE` = 4600 cycles | `kbd_aim.tap_action`, `settlecost.HEAD_BASE` |  |
| `$12E1` | (in `consider_player_action`) | `LSR $0CE5` — the first action unfreezes the enemies | `actions._mark_player_acted` |  |
| `$134C` | `initialise_sights` | a sights-ON toggle re-centres the cursor to `$0CC6`=80 / `$0CC7`=95 | `views.SIGHTS_CENTRE`, `kbd_aim.sights_set` | [2](open_items.md#2-the-drivers-wall-clock-timeouts-are-the-residual-load-sensitivity) |
| `$1363` | `check_for_player_input` | the ungated input scan (three callers) | `kbd_aim.ACTION_CODE` | [2](open_items.md#2-the-drivers-wall-clock-timeouts-are-the-residual-load-sensitivity) |
| `$139C` | action-code table | maps a key to the action code latched in `$0CE9` | `kbd_aim.ACTION_CODE` |  |
| `$1420` | `set_palette_and_initialise_enemies` | enemy count, then placement | `landscape.generate` | `golden_landscape` |
| `$1450` | `initialise_player_and_trees` | player slot, energy 10, start tile, the tree scatter | `landscape._initialise_player_and_trees` | `golden_landscape` |
| `$14AA` | `generate_secret_code_validation_table` | builds the entry-code checker |  |  |
| `$14DC` | secret-entry-code gate | computes the jump-to-play from the validation result | `core.CODE_PATCHES` | `driver/test_core.py` |
| `$14FB` | `initialise_enemies` | 8×8 grid of 4×4 sections; Sentinel + platform, then sentries | `landscape._initialise_enemies` | `golden_landscape` |
| `$151B` | `choose_a_random_grid_section` | `prnd & mask`, rejecting out-of-range | `landscape._initialise_enemies` | `golden_landscape` |
| `$1553` | `is_sentinel` | slot 0 is the Sentinel | `actions.SENTINEL_SLOT` | `golden_actions` |
| `$1586` | `set_enemies_rotation_speed` | per-enemy fixed ±20 rotation step, direction from a random bit | `landscape._initialise_enemies` | `golden_landscape` |
| `$159D` | `find_grid_sections_at_given_z` | sections whose highest flat tile equals z | `landscape._initialise_enemies` | `golden_landscape` |
| `$15B5` | `calculate_mask` | selection mask for the `prnd` section draw | `landscape._initialise_enemies` | `golden_landscape` |
| `$15CC` | `find_highest_tiles_in_grid` | per-section highest flat tile | `landscape._find_highest_tiles` | `golden_landscape` |
| `$1887` | `check_if_enemy_can_see_object` | horizontal FOV `±($0C68/2)` of the *current* facing | `relative.can_see_object` | `golden_relative` |
| `$18B8` | (in `check_if_enemy_can_see_object`) | the cone gate itself | `relative.can_see_object` | `golden_relative` |
| `$18E6` | `check_if_enemy_has_line_of_sight_to_object` | two `$1CDD` probes → `$0014` full/partial/unseen | `relative.can_see_object` (two `los` probes) | `golden_relative` |
| `$1A00` | `kill_player` | a drain arriving at zero energy is death | `memmap.PLAYER_DIED_BY_DRAINING`, `kbd_prog.KILL_PC` |  |
| `$1A97` | play setup | the play-mode entry sequence | `landscape.generate` | `golden_landscape` |
| `$1B2F` | `handle_uturn` | `objects_h_angle ⊕ $80`, starts tune `#$28` (`$1B3C LDA #$28 / BNE $1B84`), sets `$0C63`; its ASL leaves `$0C51` bit7 set so `$35B5 BMI` skips `$245B`/`$3700` in the redraw that follows | `settlecost.uturn_settle_frames`, `aimcost.h_press_count`, `kbd_aim._uturn` | `test_aimcost.py` |
| `$1B64` | `try_to_transfer_into_object` | move `player_object` into a visible type-0 robot | `kbd_prog.compile_move` (`"mount"`) |  |
| `$1B82` | (in `try_to_transfer_into_object`) | starts the transfer tune (`start_tune $888F`, tune `#$19`) BEFORE the redraw — `$35D5` waits only for its remainder, so the tune overlaps, never adds | `settlecost.TUNE_FRAMES` |  |
| `$1B8E` | `try_to_absorb_object` | absorb the topmost object in the target tile | `moves.moves` |  |
| `$1B91` | (in `try_to_absorb_object`) | `LDA $0100` — absolute read of `objects_flags[0]`; the Sentinel lock | `moves.moves` (`actions.SENTINEL_SLOT`) |  |
| `$1BBA` | `try_to_create_object` | slot, energy, placement, refund on placement failure | `actions.create` | `golden_actions` |
| `$1BBF` | (in `try_to_create_object`) | a create may spend the meter down to 0 | `actions.create` | `golden_actions` |
| `$1BE0` | (in `try_to_create_object`) | a created robot faces `creator_angle ⊕ $80` | `actions.create` | `golden_actions` |
| `$1C10`/`$1C13` | `prepare_vector_from_player_sights` | cursor + view angles → aim vector; reads neither state nor slot | `los_jit.build_lattice` | `golden_los` |
| `$1C54` | `prepare_vector_from_angle` | unit direction vector from a horizontal/vertical angle pair | `los.prepare_vector_from_angle` | `golden_los` |
| `$1C9D` | `process_sine_or_cosine` | sign/magnitude unpack of the trig lookup | `los.process_sine_or_cosine` | `golden_los` |
| `$1CBB` | `add_vector_to_object_position` | one ray sub-step (≈ 1/16 tile) in 3-byte fixed point | `los._add_vector` | `golden_los` |
| `$1CDD` | `check_for_line_of_sight_to_tile` | the one ray-march; carry set = blocked. Loop at `$1CE8` | `los.check_for_line_of_sight_to_tile`, `los_jit` | `golden_los` |
| `$1D0D` | `check_flat_tile` | surface = height nibble; `$000C` = `$80` vertical tolerance | `los._march_python` | `golden_los` |
| `$1D2C`-`$1D32` | (in `check_flat_tile`) | the look-up rejection, waived when aiming at an object top | `los._march_python` | `golden_los` |
| `$1D46` | `check_sloping_tile` | picks the triangle, interpolates the sloped edge | `los._check_sloping_tile` | `golden_los` |
| `$1D8A` | `tile_is_corner_or_quadrilateral` | slope-shape decision | `los._slope_corner_or_quad` | `golden_los` |
| `$1D9D` | `use_corner_for_slope` | corner-split facet height | `los._slope_corner_or_quad` | `golden_los` |
| `$1DAF` | `use_edge_for_slope` | edge-split facet height | `los._slope_corner_or_quad` | `golden_los` |
| `$1DF1` | slope edge table | which corner pair each slope code interpolates along | `los._slope_corner_or_quad`, `los_jit._edge` | `golden_los` |
| `$1DF9` | `calculate_tile_address_z_and_slope` | tile byte → height, slope, or object slot | `los._calc_tile_z_and_slope` | `golden_los` |
| `$1E0E` | `get_tile_z_for_line_of_sight` | the blocking/target height at an object tile | `los._get_tile_z_from_object` | `golden_los` |
| `$1E30` | (in `$1E0E`) | a platform adds `+$20` on top | `los._get_tile_z_from_object` | `golden_los` |
| `$1E3F` | `get_tile_z_from_object` | walks the stack recursively | `los._get_tile_z_from_object` | `golden_los` |
| `$1E48` | `get_boulder_or_tree_z_for_line_of_sight` | boulder/tree top; needs fraction `< $40` | `los._get_tile_z_from_object` | `golden_los` |
| `$1E5A` | (in `$1E48`) | a boulder sits `-$60` at the bottom of the band | `los._get_tile_z_from_object`, `landtable.surface_maps` | `golden_los` |
| `$1E69` | `is_tree` | the enemy-can-see-a-tree marker `$0CDD` | `los._is_tree` | `golden_los` |
| `$1EA4` | `get_height_of_lowest_object` | resolves a stack down to its base | `los._get_tile_z_from_object` | `golden_los` |
| `$1EAF` | `get_minimum_x_or_y_fraction_from_tile_centre` | "targeted" only if the ray threads near centre | `los._get_min_xy_fraction` | `golden_los` |
| `$1ECC` | `get_object_details` | seeds the march at the observer tile's centre | `los._get_object_details`, `landtable.seed_z` | `golden_los` |
| `$1EFF`/`$1F16` | `put_object_in_tile` | ground, or stacked on a boulder (+½) or platform (+1) | `landscape._put_object` | `golden_actions`, `golden_landscape` |
| `$1F38` | (in `put_object_in_tile`) | refuses a create on a tile that already carries anything else | `actions.create`, `landscape._put_object` | `golden_actions` |
| `$1F83` | (in `put_object_in_tile`) | random initial facing `(prnd & $F8) + $60` | `actions.create`, `landscape._put_object` | `golden_landscape` |
| `$210E` | `create_object` | the highest empty slot, typed | `landscape._create_object` | `golden_actions` |
| `$2120`/`$2122` | `create_object_from_action` / `find_empty_slot_loop` | slots 63→0 | `actions._find_empty_slot` | `golden_actions` |
| `$2136` | `gain_or_lose_energy_from_object` | absorb adds, create subtracts | `energy.gain`/`lose` | `golden_actions` |
| `$2143` | (in `$2136`) | carry set on underflow = "not enough energy" | `energy.lose` | `golden_actions` |
| `$2148` | `set_player_energy` | every write masks `AND #$3F` — over-absorb wraps mod 64 | `energy.gain`, `memmap.ENERGY_MASK` | `golden_actions` |
| `$214F` | `energy_in_objects` | `03 03 01 02 01 04 00` by type | `energy.value`, `memmap.ENERGY_IN_OBJECTS` | `golden_actions` |
| `$2156` | `do_hyperspace` | new robot on a random flat tile of height `<= player_z + 1` | `kbd_prog.compile_move` (`"relocate"`, `"finish"`) | landing tile unread |
| `$216A` | (in `do_hyperspace`) | spends the 3-energy toll | `moves.moves` (`ROBOT_E` in hand) |  |
| `$2170` | (in `do_hyperspace`) | kills on underflow | `kbd_prog.TOLL_PC` |  |
| `$217F` | `player_survived_hyperspace` | sets `$0CDE` bit 6 when the jump left the platform tile | `kbd_prog.WIN_PC` |  |
| `$21AE` | `plot_stack_of_objects` | the per-tile object stack draw | `projector._inview_object_base` | `golden_projector` |
| `$22AA` | `span_fill` | middle-of-polygon fill, 8 cycles per byte, 4 px/byte | `rendercost._span_fill` | `golden_render_cost` |
| `$23D0` | `plot_middle_of_row` | per-row span emit; the `$23DB` branch offset selects the entry into a 31-store unrolled loop | `rendercost._span_fill` | `golden_render_cost` |
| `$245B` | `populate_tile_visibility_bit_table` | raytraced occlusion into the `$3E80`/`$24DA` bitmap, per (board, position) | `projector.occlusion_visible` | tile-for-tile against the ROM `$3E80` bitmap |
| `$24E2` | `trace_rays_from_observer_to_row_of_tiles` | the fixed-point DDA occlusion raytrace | `projector._occlusion_visible_py.trace` | tile-for-tile against the ROM `$3E80` bitmap |
| `$2565`/`$2570` | code-entry validation | the driver patches these to accept any code | `core.CODE_PATCHES` | `driver/test_core.py` |
| `$2625` | `plot_world` | the equirectangular rasteriser, 32×32 grid furthest-to-nearest | `projector.project_scene`, `projector_jit` | `golden_projector` |
| `$26DE` | `plot_rows_in_front_of_observer_loop` | counts `$0026` 31→0 | `projector._scan_visible` | `golden_projector` |
| `$2709` | `calculate_this_row_new_first_tiles` | row span start | `projector._scan_visible` | `golden_projector` |
| `$2737` | `calculate_this_row_new_last_tiles` | row span end | `projector._scan_visible` | `golden_projector` |
| `$276F` | `consider_plotting_observer_row` | the observer-row tail branch | `projector._scan_visible` | `golden_projector` |
| `$27CE` | `plot_checkerboard_tile` | the observer's own tile, outside the `$0180` gate | `projector._scan_visible` | `golden_projector` |
| `$27D3` | `offset_to_tile_table` | `[$00,$01,$21,$20]` — the drawn-tile offset by quadrant | `projector._project_scene_py` | `golden_projector` |
| `$27D7` | `find_visible_extent_of_row_of_tiles` | the plotted span of a row | `projector._scan_visible.find_extent` | `golden_projector` |
| `$283D` | — | `BIT $9AF6`: the examine is `$2845` with the flag clear, `$37F2` with it set | `projector._project`, `passcost.EXAM_ENTRY_*` | `golden_render_cost`, generated with `$9AF6` = `$80` |
| `$2845` | `check_if_tile_is_on_screen_and_calculate_screen_coordinates` | the per-tile examine (trig floor) | `projector._project` | `golden_projector` |
| `$28D4` | `calculate_tile_address` | render-path tile addressing | `memmap.tidx` | `golden_landscape` |
| `$295D` | `plot_row_of_tiles_or_block` | the plot loop over a row | `projector._project_scene_py` | `golden_projector` |
| `$2993` | `initialise_buffer_variables` | selects the buffer window (`$29C4`) for a pan or the play view; zeroes `$0028`/`$0029` | `projector.BUF_WINDOW`, `pancost.PAN_MODE` | `golden_pan_cost` |
| `$29C7` | `initialise_buffer_variables_for_updating` | the strip's own window: `$0007`/`$0012` from `$0C69`, the halving's carry into `$0028`, and `$0011`/`$0029`/`$0035`/`$0061` | `rendercost.buffers` |  |
| `$2A24` | `plot_tile` | gates only on `$0180 != 0` | `projector._project_scene_py` | `golden_projector` |
| `$2A8A` | `plot_two_triangles` | a sloped tile is two triangles, a flat tile one quad | `rendercost.fill_cycles` | `golden_render_cost` |
| `$2ACC` | `generate_landscape` | the whole deterministic board pipeline | `landscape._generate_terrain` | `golden_landscape`; `driver.dump_stage2.verify` requires 1024/1024 tiles against the ROM's own generator |
| `$2ACE` | `randomise_row_or_column_tile_z_table` | 81 throwaway `prnd` draws | `landscape._generate_terrain` | `golden_landscape` |
| `$2AE6` | `set_landscape_vertical_scale` | `$0C08` ∈ [14..36]; landscape 0 is fixed 24 | `landscape._generate_terrain` | `golden_landscape` |
| `$2AFD` | `set_tile_slopes` | slope nibble for every interior tile | `landscape._set_tile_slopes` | `golden_landscape` |
| `$2B22` | `process_landscape` | modes `$80` raw / `$01` scale / `$02` nibble swap | `landscape._process_landscape` | `golden_landscape` |
| `$2B4B` | `scale_tile_height` | the clamp to 1..11 | `landscape._scale_tile` | `golden_landscape` |
| `$2B83` | `smooth_landscape` | 2 passes, rows then columns | `landscape._smooth_landscape` | `golden_landscape` |
| `$2BA8` | `calculate_tile_address` | `$0400 + 256·(x&3) + 8·(x>>2) + y` — interleaved, not row-major | `memmap.tidx`, `terrain.tile_byte` | `golden_landscape` |
| `$2BBC` | `smooth_row_or_column` | one toroidal pass | `landscape._smooth_line` | `golden_landscape` |
| `$2BDF` | `level_spikes` | pulls a single-tile spike/pit to its nearer neighbour | `landscape._smooth_line` | `golden_landscape` |
| `$2BFB` | `middle_is_higher_than_last` | the spike comparison | `landscape._spike` | `golden_landscape` |
| `$2C2C` | `average_tile_heights` | toroidal width-4 box filter | `landscape._smooth_line` | `golden_landscape` |
| `$2C7C` | `calculate_tile_slope` | four corner heights → a 0..15 slope code (`$2CA8`-`$2D11`) | `landscape._tile_slope` | `golden_landscape` |
| `$2D6C` | `prepare_polygon` | per-polygon vertex projection; the second wide-buffer section runs only when the first clipped an edge | `rendercost.fill_cycles` | `golden_render_cost` |
| `$2D93`/`$2DCF` | `convert_angles_into_screen_coordinates` | vertex angles → `$A7A0`/`$0B40` screen coordinates | `rendercost.fill_cycles` | `golden_render_cost` |
| `$2DF2`/`$3002` | `process_line` | the DDA edge walk writing `$AD00`/`$AE00`, split into sections when a delta overflows a byte | `rendercost._edge`, `rendercost._section_line` | `golden_render_cost` |
| `$2F5F`/`$2FAD` | (in `rasterise_polygon_edge`) | `STX $AD00`/`$AE00` — the only writes to the edge tables, one row at a time, the address low byte self-modified | `rendercost._edge` | `golden_render_cost` |
| `$2F58` | (in `process_line`) | the steep inner loop | `projector` steep inner loop | `golden_render_cost` |
| `$2FB6`/`$2FDC` | (in `rasterise_polygon_edge`) | the second entry for a line starting above the inner area: walk those rows without storing, then rejoin at `$2F67`/`$2FB0`, one column step ahead of the first store | `rendercost._edge` | `golden_render_cost` |
| `$31CA` | `prnd` | 40-bit LFSR over `$0C7B-$0C7F`, 8 shuffles per call | `prng.Prng` | `golden_prng` |
| `$339A` | `get_random_two_digit_bcd_number` | one `prnd` draw per call | `landscape._initialise_player_and_trees` | `golden_landscape` |
| `$33ED` | `seed_prnd_from_landscape_number` | seeds `state[0..1]` from the typed number as packed BCD | `landscape.seed_for`, `core.landscape_from_digits` | `golden_prng`, `test_landscape_numbering.py` |
| `$3426` | `get_maximum_number_of_enemies` | geometric draw centred on the thousands digit + 2 | `landscape._max_enemies_second_cap` | `golden_landscape` |
| `$3451` | `get_random_number_between_0_and_22` | one draw, range-limited | `landscape._rnd_0_22` | `golden_landscape` |
| `$34DE` | `play_tune` | walks `$AB50 + tune_number`; note holds count down in `$0CDF` | `settlecost.TUNE_FRAMES` (`#$19` = 96 f, `#$28` = 24 f) |  |
| `$357D` | `play_landscape_loop` | the viewpoint redraw; a u-turn takes it with `$245B`/`$3700` skipped | `settlecost.uturn_settle_frames` |  |
| `$35A4` | load signature | `A5 0B 85` — the driver's proof the game is resident | `boot.SIG_ADDR` | `driver.dump_stage2.verify` |
| `$35BA` | (in `play_landscape_loop`) | calls the occlusion raytrace | `projector.occlusion_visible` | tile-for-tile against the ROM `$3E80` bitmap |
| `$35C3`/`$35C6`/`$35C9` | (in `play_landscape_loop`) | `$35C3` is the ONE `plot_world` pass; `$35C6`/`$35C9` are `plot_status_bar` `$9508`/`$98B2`, not a second pass | `projector.render_cost`, `settlecost.STATUS_9508`/`STATUS_98B2` |  |
| `$3603` | `landscape_completed` | sets `$0CDE` bit 6 — the win | `memmap.LANDSCAPE_COMPLETE` | read back out of the machine by the driver |
| `$3642` | viewpoint redraw entry | into `play_landscape_loop` | `kbd_aim._run_to_scan` |  |
| `$365A`/`$365D` | (in the main loop) | the `JSR pan_viewpoint` call site and the pan-done PC | `kbd_aim.PC_PAN_DONE` | [2](open_items.md#2-the-drivers-wall-clock-timeouts-are-the-residual-load-sensitivity) |
| `$3682` | (in the main loop) | skips the enemy update while `$0CE5` bit 7 is set | `actions._mark_player_acted` |  |
| `$37F2` | — | the play path's examine: the same plottables off `$3700`'s table less the camera reference `$001F`/`$00B0`/`$00B1`/`$00B2`, no trig and no object-stack walk | `projector._project`, `passcost.TAB_*` | `golden_render_cost`, cycle-exact on all 15 views |
| `$3B00`/`$3C01` | arctan coefficient tables | reproduced closed-form, byte-exact | `relative._ARCTAN_LO`/`_HI` | closed form, byte-exact against the ROM table |
| `$3D02` | hypotenuse coefficient table | reproduced closed-form, byte-exact | `relative._HYP` | closed form, byte-exact against the ROM table |
| `$8401` | `calculate_object_relative_angles_and_distance` | relative x/y (`$85C4`), z (`$85F5`), then the angles | `relative.relative_angles` | `golden_relative` |
| `$8475` | object transform loop | per-vertex `transform_vertex` | `projector.C_VERTEX` | `golden_render_cost` |
| `$8533` | `plot_object` | the object model draw | `objectcost.object_cycles` | `golden_object_cost`, every vertex byte-exact; `projector._inview_object_base`'s floor without the game image |
| `$0F70` | `calculate_sine_and_cosine` | \|sin\| and \|cos\| from the `$AC80` quarter-turn table | `objectcost._sin_cos` | `golden_object_cost` |
| `$9939`/`$994F` | the pan raster window | `$0051`/`$0052` from `$994B`/`$994D`: a pitch notch plots 64 rows, a bearing notch 192 | `pancost.PAN_ROWS` | `golden_pan_cost` |
| `$888F` | `start_tune` | begins a tune, number in `$0CE7` | `settlecost.TUNE_FRAMES` |  |
| `$9287` | `calculate_angle` | bearing from a relative x/y pair | `relative._calc_angle` | `golden_relative` |
| `$933D` | `calculate_object_relative_vertical_angle` | pitch from z and distance | `relative._vertical_angle` | `golden_relative` |
| `$937F` | `calculate_hypotenuse` | horizontal distance | `relative._calc_hypotenuse` | `golden_relative` |
| `$9508` | `plot_status_bar` (first section) | the `$35C6` status plot; pads to fixed columns so its length breathes with the energy | `settlecost.STATUS_9508` (1946 ± f(E)) |  |
| `$9630` | raster frame marker | `DEC $0CDF`; one `$9630`→`$9630` span is exactly one frame | `driver.clock.frames`/`run_frames` |  |
| `$9678`/`$967B` | gated full input scan | the driver's press window | `kbd_aim._run_to_scan`, `views.TAP_FRAMES` | [2](open_items.md#2-the-drivers-wall-clock-timeouts-are-the-residual-load-sensitivity) |
| `$98B2` | `plot_status_bar` (second section) | the `$35C9` status plot | `settlecost.STATUS_98B2` (130553 cycles, constant) |  |
| `$9925` | `PAN_DELTA` table | `$14/$F8/$04/$F4` added before the pan's `plot_world` | `pancost.PAN_DELTA` | `golden_pan_cost` |
| `$9939`/`$994F` | pan buffer-mode entries | vertical `A=#$00` (play window), horizontal `A=#$02` | `pancost.PAN_MODE`, `projector.BUF_WINDOW` | `golden_pan_cost` |
| `$9958` | `move_sights` | steps cx and cy in ONE call — the cursor moves diagonally | `kbd_aim.fine_cursor` | `test_landtable.py`; [3](open_items.md#3-landability-filter-unproven-corners) |
| `$9965`/`$9994` | (in `move_sights`) | ±1 px per gated scan; clamps cx `$10-$8F`, cy `$20-$9F` | `los.CURSOR_CX`/`CURSOR_CY` | `test_landtable.py`; [3](open_items.md#3-landability-filter-unproven-corners) |
| `$9CA0`/`$9CA1` | object vertex counts | per model type | `projector._OBJECT_MODEL` | `golden_render_cost` |
| `$9CAB`/`$9CAC` | object polygon counts | per model type | `projector._OBJECT_MODEL` | `golden_render_cost` |
| `$9D37` | rotation speed table | the per-enemy ±20 step, in RAM | `memmap.ROTATION_SPEED_TABLE` | `golden_landscape` |

---

## The model (`sentinel/`)

| Module | Role |
|--------|------|
| `memmap.py` | RAM addresses, object types, the interleaved tile index |
| `prng.py` | the 40-bit LFSR `prnd` and landscape seeding |
| `state.py` | the canonical state: a 64 KB `bytearray` laid out like the game's RAM, with typed object-array views |
| `terrain.py` | height/slope nibble decode and the slope-facet surface |
| `landscape.py` | `generate(landscape) -> State`, the board generator |
| `game.py` | `Game`: a board built the way the game enters it |
| `actions.py`, `energy.py` | `create` (the stacking rule) and the energy economy (`$2136`, table `$214F`, 6-bit mask, underflow) |
| `los.py`, `los_jit.py` | the LOS ray-march, the sights aim vector and the keyboard lattice; numba fast-march bit-identical to `los.py` |
| `landtable.py` | the closed-form landing sweep over the keyboard lattice |
| `relative.py`, `sight_jit.py` | object-relative bearing/distance/vertical angle (`$8401`) and enemy sight (`$1887`); the numba specs and their Python reference |
| `exposure_jit.py`, `threat.py` | which enemies see a body; whether an observer sees a robot on a tile |
| `aimcost.py` | keyboard-aim geometry: keystrokes to pan a heading, u-turn-aware |
| `pancost.py`, `settlecost.py` | per-notch pan redraw cost (`pan_viewpoint $10B7`); the u-turn redraw and its counted segment constants |
| `projector.py`, `projector_jit.py` | `plot_world $2625` terrain projector and occlusion, ported bit-exactly; the render-cost entry points |
| `rendercost.py`, `objectcost.py`, `objmodel.py`, `pricing_jit.py` | terrain fill, object stacks (tables from the game image) and the one-job `plot_world` pass in integer cycles |
| `passcost.py`, `writeweight.py` | counted cycle constants of the ROM routines the pricing and sight specs charge |
| `views.py` | view selection: per landable tile, the frame-cheapest keyboard view |
| `aimbatch.py` | `views.aim_table` for many boards at once |
| `moves.py` | the search's move generation over many boards, its objective and selection |
| `batch.py` | batched landing sweeps |
| `kern.py`, `kern_cuda.py`, `jitcache.py` | one-source CPU/CUDA kernel specs, the CUDA launch layer, and numba's cache key |
| `emu.py` | the CSDb release booted and entered on vagrant-moses |
| `statecache.py`, `atlas.py` | the landscape atlas: cached board images, and the metrics measured off them |
| `isoview.py` | isometric SVG of any `State` |

`State` is a mutable `bytearray` image; `State.clone()` deep-copies it.

### Validation

Mechanics are differentially validated against the 6502 via `sentinel/tests/oracle.py`, then
frozen as JSON goldens replayed by CI — the fifth column of the
[routine table](#rom-routines-and-the-model) says which golden covers which routine. Three
results are worth stating on their own:

- **The board is exact.** `driver.dump_stage2.verify` requires 1024/1024 tiles against the
  ROM's own generator run in the emulator.
- **No game data is embedded.** The arctan (`$3B00`/`$3C01`) and hypotenuse (`$3D02`)
  coefficient tables are reproduced from closed-form expressions verified byte-exact against
  the ROM.
- **A move's outcome is the game's.** The search's children are the ROM running on
  vagrant-moses; `driver/test_search_player.py` holds a forked child equal to its program run
  alone, an executed line equal to its plan, and a live mirror equal to the emulated session.

### Not modelled (deliberate scope)

- **Game time and the enemies.** The model has no frame clock and no enemy machine; both
  exist only as the ROM running on the emulator or in VICE.
- **PRNG-driven landing coordinates.** The landing tile of a hyperspace or a meanie
  relocation (`put_object_in_random_tile_below_z $1224`) is never predicted; the search
  takes whatever the machine does.
- **The settle of a create, absorb or transfer.** View selection prices the aim (toggle,
  u-turn, pan notches, cursor drive); what follows the tap is the same for every view of a
  tile, so it is not priced.
- Sound side effects carry no gameplay state.

## View selection (`views.py`)

A tile is landed by tens of thousands of lattice rays and a move is played on exactly one,
so the pick decides how many frames the aim takes. `views._Landings` keeps, per tile, the
**frame-minimal** ray under the aim model, priced mechanism for mechanism against the key
sequence the driver sends, over the lattice [the geometry imposes](#what-the-geometry-permits).

- **Body pan is two keystroke ramps** (`pancost.notch_frames`), each notch followed by one
  `plot_world`: horizontal `$10EE` = 16 scroll steps per ±8 bearing notch, vertical `$1135` =
  8 steps per ±4 pitch notch (`views.H_SCROLL`, `views.V_SCROLL`). Notch counts from
  `aimcost.h_press_count` (u-turn-aware, returns `(n_uturn, n_step)`) and `aimcost.v_steps`.
- **U-turn** = one action tap plus its own redraw: `$1B2F` flips the bearing, starts tune
  `#$28` (24 f) and takes the `$357D` redraw with `$245B`/`$3700` skipped (`$0C51` bit7,
  `$35B5 BMI`). Priced per scene by `settlecost.uturn_settle_frames`;
  `views.UTURN_FLOOR_FRAMES` (the tap bracket plus the 24-frame tune floor) is the
  admissible bound. Taken only when it strictly lowers the keystroke count.
- **Cursor is derived, not fitted.** `move_sights $9958` steps both axes in one call at 1 px
  per gated scan, so a drive costs `max(|Δcx|, |Δcy|)` scans plus `views.CURSOR_RAMP` =
  popcount(`$6B`) scans the `$0CC8` auto-repeat mask skips. Zero if the cursor is parked.
- **Sights toggle is a state transition.** A same-bearing reuse keeps sights on and drives
  from the live cursor at zero toggle cost; otherwise `views.TOGGLE_FRAMES` is charged and
  `$134C` re-centres to `views.SIGHTS_CENTRE = (80, 95)`.

**Why one ray per cell suffices.** The price decomposes — the pan term depends only on
`(h_angle, v_angle)`, the cursor term only on the cursor and monotonically in its drive
distance — so one representative per `(h, v)` cell (the cursor nearest `SIGHTS_CENTRE`, then
the lowest ray) attains the minimum. `views._group_min` finds the least `(drive, ray)` key of
every `(tile, cell)` as a min over a dense table. The cell prices come from
`views.aim_table`: `pancost.notch_plots` pans along the start pitch first, so every pitch of
one bearing shares its bearing notches and each pitch path is a prefix of the farthest one in
its direction; one notch list per (bearing, direction) prices the whole column
(`sentinel/tests/test_aim_table.py` pins it on every cell). The asked tiles then run one
branch-and-bound together in `views._aim_bound` order (the aim with every notch's
`render_cost` dropped: admissible, since a render is never negative); a priced cell prices
every tile's representative in it.

**Callers.** `views._Views` answers per-tile queries on one board: each lattice is swept whole
at its first query (`views._sweep`) and pinned to that board, memoized on the scene key and
the facing aims are priced from; `get`/`band_ordered`/`band`/`primary` read off the sweep.
`moves.views` is the search's batched form: both lattices are swept for every parent in one
launch each (`batch.landings`), the aimed `(bearing, pitch)` cells of every move are priced
together by `aimbatch.aim_parts` (pan notches and u-turn redraws become one pricing job each,
deduplicated on scene, observer and notch), and `_Landings.winners` gives each move tile's
view; `moves.action_views` re-aims each later action of a hop at the stack its creates leave.

## The landing sweep (`landtable.py`)

Which lattice rays land, and on which tile, for a whole board. A full march of the band
lattice is millions of rays; the sweep walks each ray's track in closed form against the
real surface map first and marches only the rays that can land.

`_get_object_details $1ECC` seeds every ray at `px_frac=py_frac=0`, `px_sub=py_sub=0x80` (the
eye tile's centre) and `prepare_vector_from_player_sights $1C10` reads neither state nor slot,
so `_add_vector $1CBB` gives, at sub-step `i`:

```
DX_i  = floor((0x8000 + i*vx) / 65536)             tile offset from the eye tile
DY_i  = floor((0x8000 + i*vy) / 65536)
z16_i = eye_z*256 + obj_z_frac + floor(i*vz / 256)  the $003B:$0038 compare pair
```

A ray's track is a pure function of its aim; terrain only decides where the march stops.
`obj_z_frac` and `eye_z` are additive offsets (`landtable.seed_z`, never table axes) and
`DX`/`DY` are position-independent, so one set of lattice vectors (`los_jit.build_lattice`,
cached) serves every observer.

**The condition.** `check_flat_tile $1D0D` lands only when `D = surface16 - z16` is in
`[0, $80)` (`$0079` vs `$000C`, tightened to `$10` on the object path), where
`surface16 = tile_z*256 + $0079`; `D < 0` marches on, `D >= $80` blocks. So the ray must, at
some sub-step inside the cell, have entered above the band (`z_entry > surface16 - $80`)
**and** reach the surface (`min z16 <= surface16`). `landtable.surface_maps` gives each
tile's `(lo, hi)` surface bracket — exact for bare terrain, over the whole stack otherwise
(platform `$1E30` `+$20` on top, boulder `$1E5A` `-$60` at the bottom) — and its flags. Two
exact shortcuts come straight from the geometry: the observer's own tile never lands
(`$1D32`), and a sloping tile never lands (`NEVER_LANDS`: `check_sloping_tile $1D46` only
loops or blocks).

**Partition by landing tile.** `landtable.stop_cells` walks each ray's track cell by cell and
names the tile(s) it can stop in; most of the lattice crosses no surface band before leaving
the board and is never marched. The walk may only stop where the ray **provably** stops
(`zmin <= surface_lo`); an undecidable cell (an object stack with `lo < hi`, or alias risk)
makes the ray multi-candidate and always marched (`test_stop_cell_partition_holds_every_landing`).
`landtable.landing_rays` is every ray naming a candidate, and marching those is the full sweep.

**Wrap safety.** The ROM compares z as bytes, so a ray far enough above a surface aliases onto
"equal" (`WRAP_D`). Such cells are kept as candidates unconditionally, so the answer is sound
either way — see [open item 3](open_items.md#3-landability-filter-unproven-corners).

**Climbing rays.** The ROM rejects a look-up (`vz >= 0`, `s30` bit 7 clear) at every flat or
object tile unless `$0C6E | $0C67` has bit 7 set; the sweeps keep `$0C6E` bit 7 clear, a
sloping tile never ends CLEAR, and `$0C67` is local to one object-stack walk (0, or `$80` when
that walk reaches a boulder), so it never carries across tiles. A climbing ray therefore lands
only on a tile whose stack holds a boulder, and on nothing on a board without one:
`surface_maps` flags both (`HAS_BOULDER`, `BOARD_BOULDER`) and `_stop_one` restricts to them.

**Claim order.** A consumer keeps the least ray per tile (`keep="first"`) or the least
`(drive, ray)` per tile and cursor cell (`views._Landings`, `keep="cell"`).
`landtable.claim_plan` orders the sweep that way, and `landing_pairs` marches in doubling
chunks, so a ray whose only candidate tile its group has already landed on is never marched.

**Validation.** `sentinel/tests/test_landtable.py` pins the two properties everything rests
on: the closed-form track equals `_add_vector` sub-step for sub-step
(`test_closed_form_track_matches_add_vector`), and every ray a full march lands on a tile
names that tile among its `stop_cells` candidates (`test_stop_cell_partition_holds_every_landing`).

## Kernel specs and backends (`kern.py`, `kern_cuda.py`)

The scalar integer kernels are the specification of the model's bit-exact arithmetic: the
ray march in `los_jit` (`march`, `_march_one` and their helpers), the landing track walk in
`landtable` (`_cell_last`, `_stop_one`), the sight specs in `sight_jit` and the pricing pass
in `pricing_jit`. Each is written once, as a plain Python function marked `@kern.spec`, and
compiled for two targets from that one source.

* **CPU.** `spec` compiles in place with `numba.njit(cache=True)` (`inline=True` keeps
  `inline="always"`), so a spec is an ordinary dispatcher and `py_func`, line numbers and
  coverage are unchanged.
* **CUDA.** `kern.build(fn, "cuda")` recreates each spec of the module from its own code
  object (`types.FunctionType`) over a globals dict in which every spec name is bound to its
  CUDA device function, then wraps it in `numba.cuda.jit(device=True)`. Callees therefore
  resolve to the same target's builds; nothing is re-typed or re-written.

Only the launch layer differs per target: the `prange` loops of the CPU callers
(`stop_cells`, `march_batch`) against the one-thread-per-ray kernels of `kern_cuda`.
Each thread writes its own cell of a dense per-ray result, compacted on the host in lattice
order, so results never depend on thread scheduling. The state-independent lattice vectors stay resident on the device between calls.

**Backend.** `SENTINEL_BACKEND=cpu|cuda|auto` (default `auto`: cuda when numba-cuda can
compile and launch a kernel). Strategy may differ per backend while results stay identical:
on cuda `batch.landings` marches the whole lattice (no `stop_cells` prefilter, no
doubling-chunk early exit). The CUDA extra is optional (`requirements-gpu.txt`; numba-cuda needs numpy below 2.5).

**The spec subset.** `sentinel/tests/test_kern.py` lints every `@spec` function's AST: scalar
integer arithmetic with explicit masking, no allocation, no exceptions, no recursion, no
`prange`, tables only as module-level arrays, tuple returns, floats only in `_cell_last`
(exact: see `_cell_span`'s derivation in its docstring), no `fastmath`. Two differences
between the targets are designed out: `math.ceil`/`math.floor` return floats on CUDA, so the
callers wrap them in `int`; the CUDA simulator keeps numpy scalar widths, so int16/int32
inputs are widened with `np.int64` at the first use. `test_kern.py` also asserts CPU/CUDA bit
identity on random lattice slices of several boards, on a real device or, with
`NUMBA_ENABLE_CUDASIM=1`, in the simulator (the `cudasim` CI job, built from the
`Dockerfile`).

Because the specs bind cost constants as compile-time globals and `@njit(cache=True)`
invalidates only on the *defining* file's source stamp, editing `passcost.py` alone would
leave a stale compilation charging the old cycles. `sentinel/jitcache.py` folds a digest of
the constants and kernel sources into `NUMBA_CACHE_DIR` before numba is imported, so a change
is a new cache.

## Batched queries (`batch.py`)

`batch.landings(states, grids, slots=None, eye_z=None, max_steps, keep, drive)` ->
`[(rays, tiles)]` answers the landing sweep for a list of board states, each result equal to
the single-state `landtable.landing_pairs`. On the CPU, identical images are computed once and fanned
back out (`batch.state_key`, `batch.first_of`). `views._sweeps` and `moves.views` build their
`_Landings` from it.

On cuda one launch covers a 2-D (state, ray) grid over a stacked `(B, 65536)` image array and
writes a dense `int16` result per (state, ray); the host compacts it in lattice order, so there
is no atomic compaction and no sort. On the CPU a lone state scales poorly over rays, so when
the batch fills the threads the unit of parallelism becomes the state
(`landtable.landing_pairs_many`, one thread per state running the `stop_cells` prefilter then
the march serially). Below that width the ray-parallel per-state path stays.

**Backend rule.** A backend pays when its work items fill the machine: cuda when
`rays x states` reaches the device's resident threads (`multiprocessors x max threads per
multiprocessor`, `kern_cuda.resident_threads`; `batch.cuda_pays`); the CPU goes
state-parallel when `states` reaches `numba.get_num_threads()` (`batch.width`). Both are the
same statement — parallel width at least the hardware's — so no constant is tuned.

## Pricing kernels (`pricing_jit.py`, `exposure_jit.py`)

**One job, one spec.** `pricing_jit.render_cycles` is the whole of one `plot_world` pass in
integer cycles: `projector_jit.scan_scene` (the $27D7/$26DE visibility walk and $2A24 plot range),
then `objectcost.pass_cycles` (terrain fill through `rendercost.tile_cycles`, plus the $21AE
object stacks), returning `exam + walk + fill`. The projection, fill and object code are specs
over caller-owned scratch (`occlusion_into`, `scan_scene`, `rendercost.fill_pass`,
`objectcost.stack_cycles` take the arrays they would otherwise allocate; `project_scene`,
`occlusion_table` and `fill_cycles` remain as allocating wrappers). A job is one int64 row of
`J_N` columns (the `projector_jit` setup vector, the $0C48 hint, the buffer window and raster
rows from `rendercost.buffers`, observer, player, view). The object model tables
(`objmodel.tables()`, read from the game image) are runtime arguments, not module constants.

**Bit-identity.** The host divides the integer total by `FRAME_CYCLES` in float, the same
expression the scalar `render_cost` evaluates (`BASE_CYCLES` is required to be integral;
a fractional one or a missing game image leaves the request to the scalar path).
All of it is integer and independent per job, so a batch equals the scalar calls bit for bit.

**Launches.** A launch prices the requests of any number of scenes: a job row names its board
image (`J_MEM`, an index into the stacked `(S, 65536)` images) and its occlusion table
(`J_VIS`). `projector._cost_jobs` builds a scene's rows column-wise and
`projector._render_cycles_many` stacks scenes, so `prime_costs` and `aimbatch.aim_parts` are
one launch per round. `pricing_jit.render_many` runs one job per `prange` iteration over
per-slot scratch; `render_serial` is the same loop on the calling thread for a lone job.
`kern_cuda.render_cycles_many` runs the same spec on the device, one thread per job, from
persistent device buffers (scratch grown on demand and capped by the resident threads and half
the free device memory; one result download). The device is chosen by
`kern_cuda.pricing_pays(jobs)`: jobs fill a block of `BLOCK` threads on every multiprocessor.

**Why a job is one thread.** A pass is a serial chain: `plot_world` threads `$0010` and the
never-cleared `$AD00/$AE00` edge tables through every tile and object stack in draw order, and
the visibility walk carries its column cache and extent between rows, so the tiles of one job
cannot be priced independently without changing the integer result. Parallelism is across jobs.

**Prefetch.** The scalar entry points stay the semantics; batching only fills their memos.
`projector.prime_costs(state, requests)` prices a list of `render_cost` argument tuples in one
launch (`pancost.prime_notches` builds the list for the notch plots of an `aim_table`).

**Exposure.** `exposure_jit.exposures(mem, zp, slot, out)` is the player-side exposure scan as
a spec over `sight_jit._can_see_object` and `_relative_angles`: the enemies in slot order that
have any sight of a robot body in `slot`, with their $8401 `angle_hi` and full-sight flag. It
writes the LOS tracker bytes as the ROM scan does; `sentinel/tests/test_exposure_jit.py` pins
results and every byte of memory to the Python scan over several boards.
