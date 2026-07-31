# Architecture

Three parts: a forward model of the game (`sentinel/`), a driver that executes against the
real game in VICE (`driver/`), and an instrument that races the two frame-for-frame. The
6502 is a test-time oracle only; its outputs are frozen as golden fixtures so CI proves the
model without the copyrighted ROM.

This document is the *machine's* structure. The player-facing rules are
[gameplay.md](gameplay.md); the players are [players.md](players.md); everything still
wrong is [open_items.md](open_items.md).

One object generates the rest: a 32×32 height field built from a typed number. What can be
seen, what can be landed on, what can be built on, where an enemy's gaze reaches and what a
frame costs to draw are all queries against that field. So the geometry comes first, and
aiming, occlusion, draining and render cost follow from it.

- [The landscape and its geometry](#the-landscape-and-its-geometry) — the board, the shared
  coordinate system, and what standing somewhere means.
- [What the geometry permits](#what-the-geometry-permits) — seen ⊃ landable ⊃ buildable, and
  what each of them costs.
- [The state machines](#the-state-machines) — the world clock, the enemy, the player action,
  the driver, the keyboard aim primitive.
- [ROM routines and the model](#rom-routines-and-the-model) — one row per address: routine,
  effect, model function, validation.
- Then the derived subsystems: [the model](#the-model-sentinel),
  [the landability filter](#the-landability-filter-landtablepy),
  [render cost](#render-cost-projectorpy-pancostpy-rendercost_py65py),
  [the live driver](#the-live-driver-driver),
  [the instrument](#the-divergence-instrument-driverinstrumentpy),
  [the measurement tools](#measurement-and-iteration-tools).

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

Both reproduce the `ls42.json`/`ls335.json` human-win fixtures object for object.

```python
from sentinel import Game
g = Game.typed(42)                   # (13, 29), energy 10 -- no emulator
g.create(g.state.obj_type, (x, y)); g.step_enemies(); g.won()
```

### The tile map, and the one coordinate system

Everything — terrain, LOS, actions, enemies, the renderer — reads one 64 KB `bytearray`
(`state.State`) laid out exactly like the game's RAM, so a boulder placed by `actions.create`
is immediately an occluder for the next `los` march. `memmap.py` holds every address; the
object arrays are live `_ObjArray` views over `mem`, never snapshots.

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
  sits at the tile height with `z_fraction = $E0`; `playerbase.ROBOT_EYE = 0.875` is that
  offset.
- **the body** occupies the tile, blocks rays like any other object, and is drainable while
  it is a type-0 robot — including after you leave it.
- **height is bought, never found.** `put_object_in_tile $1F16` accepts a create on bare
  ground, on a **boulder** (`+½` unit, `playerbase.BOULDER_H = 0.5`) or on the **platform**
  (`+1`); `$1F38` refuses every other occupied tile. The only ladder is boulder → transfer →
  boulder, and `los._get_object_details` (`$1ECC`) re-seeds every ray at the new slot's tile
  centre and eye z.

Two consequences the players are built on. A hop is atomic — a half-built pedestal is a
boulder standing in the open with no one on it — and a spent pedestal plus the abandoned
shell are 5 energy the enemy will otherwise dismantle, so the standard hop ends by absorbing
them back ([gameplay.md §7](gameplay.md#7-how-a-human-wins-quick-strategy)).

## What the geometry permits

Three questions, strictly nested. Each is the previous one plus a constraint the *machine*
imposes, and the gap between them is where most of the model's work is.

| question | test | ROM | model |
|---|---|---|---|
| Is there a clear line to this tile? | one ray march from the eye | `check_for_line_of_sight_to_tile $1CDD` | `los.check_for_line_of_sight_to_tile`, `los_jit` |
| Can a **keyboard aim** stop on it? | some lattice ray must *land*, not merely pass | `$1C10` + the sights clamps `$9965`/`$9994` | `los.landable_view`/`landable_views`, `landtable` |
| Can I build on it? | landable **and** the stacking rule | `put_object_in_tile $1F16`/`$1F38` | `actions.can_create` |

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
262,144 rays for the `$F5` plane and 3,538,944 for the full pitch band, which is why there is
a [closed-form filter](#the-landability-filter-landtablepy) in front of every query.

**Buildable is stricter again.** Beyond the stacking rule, a **sloping tile never lands** —
`check_sloping_tile $1D46` only continues or blocks, it has no hit case — and the observer's
own tile never lands (`$1D32`). Both are exact shortcuts in `landtable.never_lands`. So the
build set is: bare *flat* tiles, boulder tops, and the platform.

### What follows from the geometry

Everything else in this document is one of these queries wearing a different hat.

- **Aiming** is the cost of moving along that lattice: bearing notches and pitch notches each
  cost a `pan_viewpoint $10B7` redraw, cursor pixels cost gated scans. Priced in
  [`aimcost`/`pancost`/`playerbase._aim_frames`](#aim-cost-playerbase_aim_frames).
- **An enemy's gaze** is the same march plus a heading gate: the target must lie within
  `±($0C68/2)` of the enemy's *current* facing (`$1887`/`$18B8`), and `$18E6` runs `$1CDD`
  twice — at the target's top and `$E0` lower — to get full / partial / unseen. Safety
  therefore quantifies over every facing the enemy will rotate through, not the current one.
- **Draining** is what a cone plus full sight buys the enemy after a countdown; see
  [the enemy machine](#the-enemy).
- **Occlusion and render cost** walk the same height field. `populate_tile_visibility_bit_table
  $245B` raytraces the field into a bitmap that only zeroes a tile's plot byte, so an occluded
  tile is still *examined* — it pays the trig floor and not the fill. That split is the
  [render-cost model](#render-cost-projectorpy-pancostpy-rendercost_py65py), and render cost
  is time, because the game is not frame-locked.

---

## The state machines

Four machines run the game, and one more runs the driver. Every edge below is a ROM branch
with an address; nothing here is inferred from behaviour.

### The world clock — the only edge both sides can agree on

The game is **not frame-locked**. `update_game_and_continue $363D` calls `update_game $127C`
with no vsync wait and re-enters `play_landscape_loop $357D` on carry, so the foreground rate
is compute-bound and dominated by the 3-D redraw. `update_game_loop $1289` calls
`update_enemies $16B5` once per pass, repeatedly, until an enemy forces a visible replot.

The fixed cadence is the raster. `$9663` — or the scroll loop `$3684` while scrolling, the
two being mutually exclusive — runs `update_enemy_cooldowns $130C` **once per video frame**.
`$130C` is an integer Bresenham divider: `$1335 += $CD` each frame, falling through to
`$1317` only on the carry (205 of every 256 frames); `$1317` decrements the three cooldown
arrays only on every third carry, gated by `$0C50`. One cooldown "unit" is therefore
`3 · 256 / $CD` frames (`playerbase.UNIT_FRAMES`), and a byte only decrements while
`>= COOLDOWN_STICK` (2), sticking at 1 until reset. `$130C` is also the raster IRQ's only
variable-length part — 21 cycles on a frame that does not carry, 33 on a gate decrement,
33 plus 14 or 20 per byte on the walk — so `cooldown_frame` returns its cycles and the
foreground loses them from that frame's budget, a swing of up to 398 cycles per frame.

### Passes per frame is a cycle budget, not a constant (`passcost.py`)

The loop never counts frames. The raster IRQ pre-empts it, takes a fixed slice of the PAL
frame, and the foreground spends what is left, so

    passes in a frame = (PAL_FRAME_CYCLES − IRQ_CYCLES) ÷ pass cost in the current state

`enemies.advance_frame(state, plotting=False)` is one frame: `cooldown_frame`
(`$9663`/`$130C`) **first**, so an enemy the tick makes due acts in the same frame; then
`FOREGROUND_CYCLES` is added to a carried budget and the pass runs, charged its own cycles,
until the budget goes negative. The negative remainder (`State.cycle_residual`) is the next
frame's debt, so a pass that outruns a frame — a 64-slot scan whose ray-march walks off the
board costs 60-70k cycles, three whole frames — leaves the following frames with **no**
`$16B5` at all, which is what the ROM does. `plotting=True` suppresses the sweep and leaves
the residual alone, modelling the spans in which the foreground never reaches `$16B5`.

A pass is **not** atomic. `$129F JSR $16B5` sits 25 cycles into a 142-cycle straight line,
so a frame boundary can fall before the update, inside it, or after it, and the ROM's state
at `$9630` differs in each case. `State.pass_phase` names which of three resume points the
budget ran out at and `cycle_residual` is the debt owed there; together they are a position
*inside* a pass, not just a frame debt. Measured on the machine by reading the interrupted
PC off the `$95E9` stack frame at `$9630`, 1500 frames a board, the boundary lands:

| segment | ROM | cycles | writes | ls0042 | ls0335 | ls9795 |
|---|---|---|---|---|---|---|
| `PASS_HEAD` + `UPDATE_*` dispatch | `$1289..$16D8` | 25 + 22/29/32 | none | 6.9% | 2.6% | 1.7% |
| `update_body` | `$16E6` `consider_enemy_state` | its own | **every CORE field** | 2.4% | 48.5% | 65.0% |
| `UPDATE_PRND` | `$16D6 JSR $31CA` | 433 | prng, cursor (at its end) | 52.3% | 26.8% | 17.4% |
| `UPDATE_CURSOR` + `PASS_TAIL` + `$191F` | `$16D9..$12C7` | 20/24 + 117 + exposure | none | 38.4% | 22.1% | 15.9% |

`advance_frame` spends them in that order, applying each segment's writes at the point the
ROM makes them: the cursor decrement and the prnd result are stored only once `UPDATE_PRND`
is paid, which is why the cursor no longer diverges at frame 1 on any board.

#### `consider_enemy_state` is resumable too

The body is the expensive segment and the only one holding CORE writes, so it is itself
split. Every `$1887` visibility call writes only SCRATCH (`$0014`, `$0C56`, `$0CDD`,
`$0C76`, `$0C58`), so a frame boundary inside one leaves no trace; the CORE writes sit
between them. `State.body_stage` names the resume point, `body_index` the scan slot reached
and `body_partial` the `$17B2` head-only player candidate — three fields the 64 KB image
does not hold, carried through the numba twin's call and through `tests/ckpt.py`.

| `body_stage` | ROM | CORE writes it commits |
|---|---|---|
| `BODY_ENTRY` | `$16E6` gate, `$16ED`, `$16F0` | `update_cd = 4` (`$16F0` is SCRATCH) |
| `BODY_MEANIE` | `$16F2 update_meanie` | facing, `update_cd`, the meanie/hyperspace |
| `BODY_DISCHARGE` | `$1773`/`$1A5D` | the discharged tree and its tile |
| `BODY_HUNT` | `$177F`→`$1AB0` | `$178B` search reset + the drain |
| `BODY_HELD` | `$178C` re-check | `$1825 target_object`, or `drain_cd = 0` |
| `BODY_SCAN` | `$17B2` slots 63..0 | `$1825` on the first fully-visible robot |
| `BODY_PARTIAL` | `$17C4` | `$196A` re-arm + `$1825` |
| `BODY_TREE` | `$17E0`→`$1AB0` | the drain and its `update_cd` reload |
| `BODY_ROTATE` | `$17F9`/`$1805` | facing, `rotation_cd`, `$196A` |
| `BODY_MAKE_MEANIE` | `$184D`→`$197D` | `meanie_search` per step, then the meanie |

`body_index` is a scan position: `>= 0` the next slot to query, `-1` the scan is exhausted,
`<= -2` the slot `-2 - i` has been **charged** and owes only its write. That last encoding is
what makes the split exact rather than one-unit-coarse: when the budget runs out between an
`$1887` and the `$1825` its answer causes, the model suspends there and commits on resume,
recomputing the query for free because its cycles are already paid.

The body is the open residual for a different reason now — its cycle *cost*, not its
atomicity: [open_items.md 8](open_items.md#8-the-enemy-clock-what-is-left-is-the-redraw-and-the-frame-budget). `enemies.resume_from_stack` reads that same position back off a `$9630` halt's `$95E9` frame, so a seed or a resync starts the sim where the machine is rather than at a pass head.

Every term is an instruction count off the disassembly, reproduced by running the real code
in the jennings oracle:

| term | ROM | cycles |
|---|---|---|
| `PASS_HEAD`, `PASS_TAIL` | `$1289..$129F` and `$12A2..$12C7`, the `$16B5`/`$191F` bodies apart | 25 / 117 |
| `PRND` | `$31CA`, 8 rounds of the 40-bit LFSR at 51 each | 427 |
| `UPDATE_PRND`, `UPDATE_CURSOR` | `$16D6` JSR + `PRND`, then `$16D9` + RTS (+4 on the 7→0 wrap) | 433 / 20 |
| `UPDATE_*` | `$16B5` type dispatch: not an enemy / sentry / Sentinel / absorbed | 22 / 29 / 32 / +8 |
| `UPDATE_GATE_CLOSED` | `$16E6` `LDA $0C30,X` + `CMP #2` + `BCS` | 9 |
| `EXPOSURE_*` | `$191F` per enemy slot: empty / other type / sentry / Sentinel | 12 / 24 / 30 / 33 |
| `SEE_SLOT_EMPTY`, `SEE_SLOT_WRONG_TYPE` | `$1887` exits at `$1893` / `$189D` | 40 / 49 |
| `SEE_*` | `$1887`'s own line: prologue / `$18A2` FOV compare / reject / probe / `$1904` | 37 / 54 / 18 / 56 / 26 |
| `REL_*` | `$8401` and its `$85C4`/`$85F5`, `REL_XY_ABS` 6 per negative component delta | 119 / 66 / 36 |
| `ANG_*`, `SCALE_*` | `$9287` per branch; its `$92C1`/`$92FF` shift loop at 20 a lap (21 in y) | 2..33 |
| `DIV_*` | `$0D4A` per conditional-subtract round, then the `$0E1F` arctan interpolation | 1..28 |
| `VANG_*`, `HYP_*` | `$933D` per branch; `$937F` head/tail around its own `$0F4A` | 3..52 / 38 / 40 |
| `MARCH_ENTRY` | `$1CDD`'s own line + `JSR $1ECC get_object_details` | 74 |
| `MUL8` | `$0D03 multiply_byte_by_byte`, plus `MUL8_BIT` 4 per set bit of the multiplier | 102 |
| `ADD_VECTOR` | `$1CE8 JSR $1CBB`, plus `ADD_VECTOR_NEG` 4 per negative component | 163 |
| `TILE_ADDR` | `$1DF9 JSR $2BA8` + `calculate_tile_address` | 40 |
| `CONSIDER_*`, `HELD_*`, `HUNT_*` | `$16E6`'s own line per branch: entry / meanie gate / discharge / held target | 21 / 7 / 9 / 18 |
| `SCAN_SLOT_*` | one `$17B2` slot by its exit: tree-hidden / unseen / full / other / the player's head | 22 / 27 / 25 / 34 / 36 |
| `SCAN_END*`, `TREE_*` | `$17CD..$17E8` once the scan is exhausted, and the `$1AB0` call it makes | 6 / 11 / 13 |
| `TARGET_*` | `target_object $1825` per branch: first sight / waiting / due / drain / meanie | 21 / 12 / 9 / 18 / 3 |
| `REDUCE_*`, `REMOVE_*` | `$1A08` per target kind and `$1EEF` per stacked/ground removal | 7..46 / 86 / 94 |
| `STATUS_*` | `$9508 plot_status_bar`, recomputed from the energy byte the drain wrote | f(energy) |
| `PLACE_*`, `DRAW`, `CREATE_*` | `$1238`'s tile hunt per failed test, its `$1272` draws, `$211D`'s walk | 3..38 / 445 / 11 |
| `TILE_SCAN_*` | `$1AB0` walks its own loop: empty slot / rejected / tile fetch (`$2BA8`) | 12 / 24 / +61 |
| `MEANIE_SCAN_*` | `$198F` walks the search counter, not a slot index | 26 / 34 / +42 |
| `ROTATE` | `$1805..$1884`, its `$1AF4`/`$1973`/`$3470` callees at 31/32/`TUNE_ROTATE` | 454 |
| `REDRAW_CALL`/`REDRAW_NONE` | `$1881 JSR $1F9F` `update_object_on_screen`, off-screen | 6 / 23 |
| `SPAN_*` | `$209B calculate_object_screen_span`, branch by branch | 1..33 |

A rotation is the single most expensive thing a gated enemy does and none of it is the
turn: `$1805` adds the step in 44 cycles and then redraws the enemy through
`update_object_on_screen`. That redraw is **bimodal**, so it is not one number.
`$1F9F` first calls `$209B`, which takes the object's bearing and horizontal distance
from `$8401`, re-arctans its `$2112` half-angle over that distance through `$933D`, and
turns bearing +- half-angle into a left column `$0C62` and a width `$0C69`. An object
with no span on the 40-column screen ends at `$1F93`, and that whole path is counted
from state by `relative.update_object_on_screen_cycles` — 1568..1858 on ls0042/ls0335/
ls9795, cycle-exact against the ROM's own `$209B`/`$1F9F`
(`test_the_object_screen_span_is_exact_against_the_roms_own_209b`). All 16 live
rotations in `fixtures/live_pass_cycles.json` (1576..1843) are that branch.

An object that *does* have a span is a different animal: `$1FC2` re-points the camera at
the strip (`$09C0,X += $0C62/2`, `$001F` the fine angle) and `$1FFC JSR $2625` replots it,
0.40..0.85 M cycles on ls9795 — 250..500x the branch above, and a `plot_world` cost, not
an enemy-clock one. `projector.strip_replot_frames` prices it at that shifted camera —
never the player's own — and through the strip's own buffer window: `$1FE5 JSR $29C7`
takes `$0C69` and sets `$0007 = columns >> 1`, `$0012 = ($0007 >> 1) ^ $80`, the same pair
`$2993` sets from its table for a full-screen mode, so `projector.strip_window` feeds it
straight to `render_cost`. `RENDER_COST_BACKEND=py65` instead runs the real `$1F9F` and is
exact (19..29 frames on ls0042/ls0335/ls9795, ~1.3 s per uncached call, memoized); without
the window the proxy priced all 40 columns and was 1.3..2.8x dear, which showed up as the
clock over-stalling. `enemies` charges the result in cycles.

The numba twin cannot call the renderer, so `enemies_jit._advance` **stops** on an
on-screen `$1F9F`, hands back the object and its left column with the frames still owed,
and `enemies.advance_frames` prices the replot and resumes — the same number, charged at
the same point in the pass, which `test_jit_matches_python_across_an_on_screen_redraw`
holds to byte and residual identity.

`$191F` is why the cadence is a property of the board: it walks all 8 enemy slots on **every**
pass, so an 8-enemy board's pass costs 108 cycles more than a 1-enemy board's and the idle
rate falls from 19.7 to 16.8 passes per frame. Live, over 200 frames with the clock unfrozen:
ls0 19.09 mean, ls42 18.15, ls373 12.60, ls335 9.74, ls9795 7.22 — a range the retired
constant 8 could not span. The idle brackets are pinned in
`tests/fixtures/live_pass_rate.json` and the model must land inside every one
(`test_irq_cycles_matches_the_live_pass_rate`).

The march is charged **per sub-step by the branch its tile takes**, not by a mean:
`los.check_for_line_of_sight_to_tile` returns the cycles it cost and
`relative.can_see_object` sums them, so a 64-slot scan whose ray walks off the board is
priced as the 60-70k cycles it really is. `sentinel/tests/ckpt.py` carries
`cycle_residual` through a checkpoint for the same reason: it is state the image does not hold.

No sub-step term is a mean. `$1CE8` is charged as `$1CBB add_vector` (4 more per negative
vector component), the two edge tests, the `$1CFB` reset, `$1DF9` with `$2BA8` and either
its flat tail or the `$1E3F` object walk per stack level, then `$1D0D check_flat_tile` or
`$1D46 check_sloping_tile` — the latter carrying its three `$1DF9` corner reads and, on the
`$1D8A` quad path, a `$0D03` multiply priced by its own operand. A taken branch is 3, or 4
when it crosses a page (`$1CF1`/`$1CF9`→`$1D44`, `$1D18`/`$1D40`→`$1CE8`). `$1C54
prepare_vector_from_angle` is priced the same way, down to `$0D03`'s shift-adds, and so
is the whole `$1887` bearing chain — `$8401`, `$9287` with its variable-length
`$92C1`/`$92FF` shift loop, `$0D4A`, `$933D` and `$937F`. Every one of those is
cycle-exact against the jennings oracle; what is left over is upstream of them,
[open_items.md 8](open_items.md#8-the-enemy-clock-what-is-left-is-the-redraw-and-the-frame-budget).

`IRQ_CYCLES` is the one term measured rather than counted, and it is three mechanisms, all
read off the machine with VICE's `cpuhistory` (an absolute cycle stamp per instruction, so
a badline shows up as an instruction that took 43 cycles too long):

| mechanism | per frame | measured |
|---|---|---|
| VIC-II badline steal | 25 lines (`$30..$F7`, low 3 bits = YSCROLL 3) × 43 | 1075 |
| short raster interrupts | 4 × 119 (7 entry + 112 body) at raster 53/93/133/173 | 476 |
| the `$9630` body, less `$130C` | once, at raster 213 | 2491 |

`$D015 = 0` — no sprite is ever enabled in play — so there is **no** sprite-DMA term. The
`$95E9` split chain walks `$9588` down 4→0, programming `$D012` from the table at `$9589`
(`35 D5 AD 85 5D`) and taking the full `$9630` body only for the `$9593` entry.

All three are per-**frame** totals, not per-pass ones: the badline and split-IRQ raster
lines are fixed, so any 19656-cycle window contains exactly the same set whatever the phase.
Charging a pass the badlines it crosses and the interrupts it straddles is therefore
arithmetically identical to subtracting one lump per frame, and the lump is what the model
does. What is **not** fixed is `$130C`, which is charged separately (above), and with it out
the arithmetic closes exactly: the foreground gets `FOREGROUND_CYCLES − $130C`, so 15593 on
a frame with no Bresenham carry — which is precisely the maximum measured over 20 live
ls9795 frames (15124..15593). The handler is in the fixture — the game runs with `$01 = $25`,
KERNAL banked out, so `$FFC2`/`$FFC5` are its own RAM (`JMP $8ED1`/`JMP $8F0C`, the sound
engine) and `$FFFE` vectors to `$95E9`. Evidence: `fixtures/live_pass_cycles.json`,
`test_irq_cycles_is_the_measured_badline_steal_and_handler_time`,
`test_the_cooldown_tick_prices_every_live_130c_sample` and
`test_irq_cycles_matches_the_live_pass_rate`.

Because the numba twins bind these constants as compile-time globals and `@njit(cache=True)`
invalidates only on the *defining* file's source stamp, editing `passcost.py` alone would
leave a stale compilation charging the old cycles. `sentinel/jitcache.py` folds a digest of
the constants into `NUMBA_CACHE_DIR` before numba is imported, so a constant change is a new
cache; `test_enemies_jit.py` is the backstop.

### The enemy

An enemy is a per-slot state machine driven by three cooldown bytes, serviced round-robin —
`update_enemies $16B5` handles **one** slot per call via the cursor `$0090` (7→0).
`is_sentinel_or_sentry $16C6` sends the Sentinel and sentries down the identical path; the
Sentinel's only edges are positional.

| counter | array | reload | set by | meaning |
|---|---|---|---|---|
| update | `$0C30` | `UPDATE_COOLDOWN_SCAN` 4 | `$16ED` (gate `$16E9`) | how often the ladder below runs at all |
| | | `UPDATE_COOLDOWN_DRAIN` 30 | `$17F1`/`$1848` | after a drain or downgrade |
| | | `UPDATE_COOLDOWN_MEANIE_ROTATE` 10 | `$173A` | after a meanie turns |
| | | `UPDATE_COOLDOWN_MEANIE_MADE` 50 | `$1869`/`$186E` | after a meanie is spawned |
| | | `(prnd & $3F)` OR 5 | spawn | initial phase |
| rotation | `$0C28` | `ROTATION_COOLDOWN_RELOAD` 200 | `$1813` | one ±20-unit step per period |
| draining | `$0C20` | `DRAINING_COOLDOWN_RELOAD` 120 | `$1835` | sight *arms* it; 0 = not targeting |

Sight alone costs nothing. `target_object $1825` merely arms `$0C20` to 120 rounds, and
`$1A31` re-zeroes it after each drain, so a cone sweeping across a body is free for its first
120 rounds — the whole basis of `playerbase._gaze_window`.

**The ladder.** `consider_enemy_state $16E6` runs only when the update cooldown is below the
stick value, and **returns before the rotate at `$17F9` whenever an earlier branch fires** —
so a busy enemy stops sweeping. `playerbase._cone_onset` does not model that stall
([open item 3](open_items.md#3-the-gaze-forecast-assumes-rotation-never-stalls)); the two
the phase player works around it by never forecasting at all (`_drained_over` advances a clone
over the span and looks).

| # | state entered | trigger | ROM | effect / next |
|:-:|---|---|---|---|
| — | **frozen** | player has not acted | `$0CE5` bit7; skipped at `$3682`/`$9659` | no clock at all until `$12E1` clears it |
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
| given up | `MEANIE_MAX_ATTEMPTS` (2) scans found nothing | `$1857` | `$0C90` remembers the player slot; stop retrying |
| driving | a meanie exists | `update_meanie $16F2` | rotate ±8 (`$1728`) toward the player, update cd ← 10 |
| firing | bearing within one screen width **and** `$0014 != 0` | `$171B`/`$171D JSR $2156` | forces a hyperspace on the player |
| dissolved | it fired, the bound body was absorbed (`$1707`), or the player transferred out (`$1717`) | `remove_meanie $1754` / `$174F` | back to a tree |

The parent enemy holds at most one meanie and does nothing else while it lives. Absorbing the
meanie (`try_to_absorb_meanie $1BEC`) is +1 energy and clears the link.

Implemented in `sentinel/enemies.py` with a bit-identical numba twin `enemies_jit.py`;
`enemies.step` is the isolated-routine round the py65 oracle compares against, and
`advance_frame`/`advance_frames` are the running cadence.

### The player action

An action is not an instant. `playerbase._fire(verb, tile, view)` runs it as an ordered
sequence of `(frames, plotting)` segments so the world evolves underneath it exactly as the
ROM's does, and search and executor share the one sequence (`_aim_head_tail`).

| segment | ROM | frames | runs `$16B5`? |
|---|---|---|---|
| sights toggle | `initialise_sights $134C` re-centre + `plot_sights` | 0 on a same-bearing reuse, else `TOGGLE_FRAMES` | no |
| u-turn taps | action code `$23`, no scroll, no replot | `n · UTURN_FRAMES` | **yes** |
| pan notches | `$10EE`/`$1135` scroll + one `plot_world` per notch | `pancost.pan_frames` | no |
| cursor drive + firing tap | gated `move_sights`/`tap_action` scans | `max(abs Δcx, abs Δcy) + CURSOR_RAMP + TAP_FRAMES` | **yes** |
| settle | `$1FA4`/`$86A5` dither + replot, or the `$357D` redraw | `actioncost.SETTLE[verb]` | no |

Two freezes decide which segments the enemies see.

- **The enemy freeze `$0CE5` bit7** holds until the player's *first* action. `consider_player_action
  $12D0` normally requires the sights to be active, but `$12D5 CMP #$22 / BCS` lets codes
  `>= $22` skip straight to `$12E1 LSR $0CE5`. So a u-turn — code `$23`, free, no LOS, no
  energy — **unfreezes the world mid-aim**, which is why `_aim_head_tail` splits the aim there.
- **The plot freeze `$0CE4` bit7** (`set_busy_plotting $1214`) is held across a whole viewpoint
  redraw; while it is set the foreground never reaches `update_enemies`, so only the raster
  cooldown clock advances. That is the `plotting=True` segment.

Dispatch is `handle_player_actions $1B18` off the code latched in `$0C61`/`$0CE9`; for every
code `< $22` it builds the aim vector and calls `$1CDD`, and carry set at `$1B40`-`$1B46`
plays the bad-action sound and does nothing. `$1281` zeroes the action latch `$0C51` and only
an idle full scan re-arms it at `$11EA`, so one press fires at most one action. The code →
routine map is [gameplay.md §4](gameplay.md#4-actions); the routine → model map is the
[routine table](#rom-routines-and-the-model).

`_fire` then runs the head segments, clears `$0CE5`, runs the tail, re-gates the aim through
`aim.gate` (falling back to `aim.propose(v_band=True)`), checks `_affords`, applies the
action, and advances the settle with `plotting=True`.

### Hyperspace, death and the win

Three flag bits carry the terminal states, and the same routine produces all of them.
`do_hyperspace $2156` creates a robot on a random flat tile no higher than `player_z + 1`
(`$215D` adds 1, `$2165 JSR $1238`), then spends the 3-energy toll (`$216A`/`$216B JSR $2136`).

| state | bits | set by | model |
|---|---|---|---|
| alive | `$0CDE & $C0 == 0`, `$0C4E` bit7 clear | — | — |
| drained to death | `$0C4E` bit7 | `kill_player $1A00`, a drain arriving at 0 energy | `enemies._reduce_object_energy` → `actions.player_dead` |
| died on the toll | `$0CDE` bit7 set, bit6 clear | `$216E`/`$2170`, underflow on the 3-energy spend | `enemies.do_hyperspace` → `actions.player_dead` |
| landscape complete | `$0CDE == $C0` | `player_survived_hyperspace $217F` / `landscape_completed $3603`, jumping from the platform tile `$0C19`/`$0C1A` | `actions.won` |

Standing on the platform is not a win; hyperspacing from it is. Live, the driver reads
`$0CDE` bit 6 straight out of the machine, and `LivePlayer._dead` additionally treats a
re-frozen `$0CE5` after the player has acted as an observed landscape reset.

`$216A` is also the only move that changes position with **no line of sight**: it takes no
aim and no target, so it is the one action still available from a stance that can land
nothing. That makes relocation a move class, not just the win move — `phase_player._barren`
detects a stance where every generator is empty (no landable tile at all, or nothing
absorbable, reclaimable, climbable or mountable in view) and `_relocate` jumps out of it for
the 3-energy toll. Such a stance has no income and no climb, so it loses whether or not a
cone ever finds it, and waiting cannot change it — before this the planner idled 60 frames a
tick until a cone drained it to death. The landing tile is PRNG-driven (`$2165 JSR $1238`)
and treated as unknowable, so a relocation is never *scored*, only taken; the purse is the
whole gate, since `$2170` kills on underflow.

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

### The keyboard aim primitive

`driver/kbd_aim.py` drives one aim as three sub-machines, each ending on state and never on
elapsed time. Every outcome below is a literal returned by the code.

| sub-machine | sights | steps on | ends `ok` | other exits |
|---|---|---|---|---|
| `coarse_h` | off | `PC_PAN_DONE $365D` | `$09C0+slot == want` | `hyperspace` (`$0B` changed mid-aim), `unreachable` (angle unchanged for `_PAN_STALL_FRAMES`, or `_PAN_MAX_FRAMES` spent) |
| `coarse_v` | off | `PC_PAN_DONE` | `$0140+slot == want` | `unreachable` immediately if `want` is outside `[$CD..$35]` |
| `fine_cursor` | on | the per-axis store PCs `$997C`/`$9990`/`$99B8`/`$99D2` | `$0CC6`/`$0CC7` reached | `False` after 160 iterations |
| `tap_action` | — | the gated scan `$9678`→`$967B` | the want-flag `$0CE8..$0CEB` latched | not latched → retry, up to `max_passes` |
| `sights_set` | — | `_one_scan_press("SPACE")` | `$0C5F` bit7 == requested | up to 6 presses |

`coarse_h` takes the shorter of ±8 steps or a u-turn plus a correction (`aimcost.h_press_count`),
confirming the u-turn by `(angle_after − angle_before) & $FF == $80`. Fine cursor presses
while halted and releases after the store, because a held key auto-repeats in an accelerating
burst (`$11F6 ASL $0CC8`). `_run_to_scan` treats a timeout while `$0CE4` bit7 is set as a
redraw still running and re-arms; conceding there would leak the redraw's frames into the next
primitive.

`sentinel_execute.perform_step` wraps that in aim → fire → verify. Its outcomes are `ok`,
`best_effort_miss`, `drained`, `aim_miss`, `aim_hyperspace`, `diverge` (resync + replan) and
`fail`; `classify_outcome` checks the primary effect **before** the best-effort shortcut.

---

## ROM routines and the model

Every address cited anywhere in this repo, sorted. Names are the ROM's own labels; a row
without a label names the routine the address sits inside. The last two columns are the model
mapping — the function that ports or prices the routine, and how that is proved (a golden
fixture, an oracle or instrument gate, a live test, or the numbered open item where it is
approximate). **An empty cell is information**: the model has no counterpart for that row,
either deliberately ([Not modelled](#not-modelled-deliberate-scope)) or because it is
foreground work folded into a settle constant.
| Address | Routine | What it does | Model (`module.function`) | Validation / approximation |
|---|---|---|---|---|
| `$0D03`/`$0D05` | `multiply_byte_by_byte` | 8×8 unsigned multiply, the trig primitive | `los._mul8` | `golden_los` |
| `$0D4A` | `divide_and_arctan` | shift/subtract 16-bit divide, then arctan of the quotient | `relative._divide_and_arctan` | `golden_relative` |
| `$0E75` | `sin_cos_lookup` | polynomial sine/cosine of a byte angle (256 = full circle) | `los.sin_cos_lookup` | `golden_los` |
| `$0F3E` | `multiply_double_A_by_pi` | scale a 16-bit value by π | `los.multiply_double_A_by_pi` | `golden_los` |
| `$0F4A` | `multiply_double_by_byte` | 16×8 fixed-point multiply | `los.multiply_double_by_byte` | `golden_los` |
| `$0F9E` | `multiply_double_by_double` | signed 16×16 fixed-point multiply | `los.multiply_double_by_double` | `golden_los` |
| `$1090` | `fill_screen_with_background` | clears the play buffer before a replot |  | folded into `projector.SETTLE_FIXED_FRAMES`; [4](open_items.md#4-per-step-frame-drift-and-the-unattributed-createabsorb-settle-split) |
| `$10B7` | `pan_viewpoint` | one keyboard pan notch: strip clear, one `plot_world` at the intermediate angle, queue the scroll | `pancost.notch_frames` | `golden_pan_cost` |
| `$10EE` | (in `pan_viewpoint`) | horizontal scroll — 16 steps per ±8 bearing notch | `playerbase.H_SCROLL` | `golden_pan_cost` |
| `$1135` | (in `pan_viewpoint`) | vertical scroll — 8 steps per ±4 pitch notch | `playerbase.V_SCROLL` | `golden_pan_cost` |
| `$1149` | pitch limits table | clamps the pitch band to `[$CD..$FF] ∪ [$00..$35]` | `los.PITCH_BAND` | `test_landable.py`, `test_landtable.py`; [10](open_items.md#10-landability-filter-unproven-corners) |
| `$119F` | `check_for_full_player_input` | the full key scan, including the SPACE edge test |  | `test_kbd_scan_gate.py` |
| `$11E0` | (in the input scan) | reloads the cursor auto-repeat mask `$0CC8` with `#$6B` | `playerbase.CURSOR_REPEAT_MASK` | `driver/test_live_determinism.py`; [7](open_items.md#7-the-drivers-wall-clock-timeouts-are-the-residual-load-sensitivity) |
| `$11EA` | (in the input scan) | an idle full scan re-arms the action latch `$0C51` | `kbd_aim.tap_action` | `driver/test_live_determinism.py`; [7](open_items.md#7-the-drivers-wall-clock-timeouts-are-the-residual-load-sensitivity) |
| `$11F6` | (in the input scan) | `ASL $0CC8 / BCS` — one gated scan skipped per set mask bit | `playerbase.CURSOR_RAMP` | `driver/test_live_determinism.py`; [7](open_items.md#7-the-drivers-wall-clock-timeouts-are-the-residual-load-sensitivity) |
| `$1214` | `set_busy_plotting` | sets `$0CE4` bit 7 for the duration of a redraw | `kbd_aim._run_to_scan` | `test_settle_accuracy.py` |
| `$1224`/`$1238` | `put_object_in_random_tile_below_z` | random flat empty tile no higher than a given z | `enemies._put_object_in_random_tile_below_z` | landing tile deliberately unread ([Not modelled](#not-modelled-deliberate-scope)) |
| `$125A`/`$1272` | `get_random_tile_coordinate` | a `prnd` draw masked to 0..31, rejecting 31 | `enemies._random_tile_coord`, `landscape._random_tile_coord` | `golden_landscape` |
| `$127C` | `update_game` | the per-pass game update |  |  |
| `$1281` | (in `update_game`) | zeroes the action latch `$0C51` | `kbd_aim.tap_action` | `driver/test_live_determinism.py`; [7](open_items.md#7-the-drivers-wall-clock-timeouts-are-the-residual-load-sensitivity) |
| `$1289` | `update_game_loop` | calls `update_enemies` once per main-loop pass | `enemies.CURSOR_SLOTS` | instrument gate `test_enemy_sim_frame_locked_to_live_ls42`; [8](open_items.md#8-the-enemy-clock-what-is-left-is-the-redraw-and-the-frame-budget) |
| `$12D0` | `consider_player_action` | requires the sights active before create/absorb/transfer | `kbd_aim.tap_action` | `driver/test_live_determinism.py`; [7](open_items.md#7-the-drivers-wall-clock-timeouts-are-the-residual-load-sensitivity) |
| `$12D5` | (in `consider_player_action`) | `CMP #$22 / BCS $12DE` — codes `>= $22` skip the sights check | `playerbase._aim_unfreeze_split` | `test_settle_accuracy.py` |
| `$12E1` | (in `consider_player_action`) | `LSR $0CE5` — the first action unfreezes the enemy clock | `actions._mark_player_acted` | `test_settle_accuracy.py` |
| `$130C` | `update_enemy_cooldowns` | per-frame Bresenham: `$1335 += $CD`, call `$1317` on carry | `enemies.cooldown_frame` | instrument gate `test_enemy_sim_frame_locked_to_live_ls42`; [8](open_items.md#8-the-enemy-clock-what-is-left-is-the-redraw-and-the-frame-budget) |
| `$1317` | `update_enemy_cooldowns` | decrement stage, every third carry (gated by `$0C50`) | `enemies.tick_cooldowns` | instrument gate `test_enemy_sim_frame_locked_to_live_ls42`; [8](open_items.md#8-the-enemy-clock-what-is-left-is-the-redraw-and-the-frame-budget) |
| `$134C` | `initialise_sights` | a sights-ON toggle re-centres the cursor to `$0CC6`=80 / `$0CC7`=95 | `playerbase.SIGHTS_CENTRE`, `kbd_aim.sights_set` | `driver/test_live_determinism.py`; [7](open_items.md#7-the-drivers-wall-clock-timeouts-are-the-residual-load-sensitivity) |
| `$1363` | `check_for_player_input` | the ungated input scan (three callers) | `kbd_aim.ACTION_CODE` | `driver/test_live_determinism.py`; [7](open_items.md#7-the-drivers-wall-clock-timeouts-are-the-residual-load-sensitivity) |
| `$139C` | action-code table | maps a key to the action code latched in `$0CE9` | `sentinel_execute.CREATE_KEY` | `driver/test_live_determinism.py`; [7](open_items.md#7-the-drivers-wall-clock-timeouts-are-the-residual-load-sensitivity) |
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
| `$16B5` | `update_enemies` | services one enemy slot per call, round-robin via `$0090` | `enemies.update_enemies`, `enemies_jit` | `golden_enemies`, `oracle.step_enemy_round` |
| `$16C6` | `is_sentinel_or_sentry` | the Sentinel and sentries run identical AI | `enemies.update_enemies` (`memmap.ENEMY_TYPES`) | `golden_enemies`, `oracle.step_enemy_round` |
| `$16E6` | `consider_enemy_state` | discharge → dismantle → target → reduce → rotate | `enemies._consider_enemy_state` | `golden_enemies`, `oracle.step_enemy_round` |
| `$16E9` | (in `consider_enemy_state`) | the update-cooldown gate, reload 4 | `enemies.COOLDOWN_STICK`, `UPDATE_COOLDOWN_SCAN` | `golden_enemies`, `oracle.step_enemy_round` |
| `$16F2` | `update_meanie` | rotate the meanie ±8 units toward the player, then force a hyperspace | `enemies._update_meanie`, `enemies.FOV_SCAN` | `golden_meanie` |
| `$16FF` | (in `update_enemies`) | an enemy owning a meanie drives it instead of scanning | `enemies._consider_enemy_state` | `golden_meanie` |
| `$1707` | (in `update_meanie`) | meanie dissolves when its bound body is absorbed | `enemies._update_meanie` | `golden_meanie` |
| `$1717` | (in `update_meanie`) | meanie dissolves when the player transfers out of that body | `enemies._update_meanie` | `golden_meanie` |
| `$171B`/`$171D` | (in `update_meanie`) | requires `$0014` non-zero, then `JSR do_hyperspace` | `enemies.MEANIE_ROTATE_STEP` | `golden_meanie` |
| `$1728` | `meanie_not_looking_at_player` | the ±8-unit rotation step toward the player | `enemies._update_meanie` | `golden_meanie` |
| `$174F` | `remove_meanie_and_reset_enemy` | dissolve and clear the draining cooldown | `enemies._remove_meanie_and_reset_enemy` | `golden_meanie` |
| `$1754` | `remove_meanie` | turn the meanie back into a tree | `enemies._remove_meanie` | `golden_meanie` |
| `$1775` | (in `consider_enemy_state`) | `JSR $1A5D consider_discharging_enemy_energy` (`$1773` is the `STX $6E` before it) | `enemies._consider_enemy_state` | `golden_enemies`, `oracle.step_enemy_round` |
| `$177A` | (in `consider_enemy_state`) | `JMP $1876` — a discharge returns before the rotate | `enemies._consider_discharging_enemy_energy` | `golden_enemies`, `oracle.step_enemy_round` |
| `$177F` | (in `consider_enemy_state`) | the meanie-hunt gate `$0CB8`; on a hit `$1784` drains the stack object | `enemies._consider_enemy_state` | `golden_enemies`, `oracle.step_enemy_round` |
| `$178C` | (in `consider_enemy_state`) | still sees its held target — returns before the rotate | `enemies._consider_enemy_state` | `golden_enemies`, `oracle.step_enemy_round` |
| `$17B2` | `find_drainable_robot_loop` | scans all 64 slots for a visible type-0 robot | `enemies._consider_enemy_state` | `golden_enemies`, `oracle.step_enemy_round` |
| `$17F9` | (in `consider_enemy_state`) | the rotate branch, reached only when nothing else fired | `enemies._consider_enemy_state`; forecast `playerbase._cone_onset` | `golden_enemies`, `oracle.step_enemy_round`; the rotation stall is unmodelled [3](open_items.md#3-the-gaze-forecast-assumes-rotation-never-stalls) |
| `$17FB` | (in `consider_enemy_state`) | `LDA $0C28,X / CMP #$02 / BCC $1805` — the rotate fires only while the rotation cooldown is below the stick value; otherwise `JMP $16D6`, the round tail | `enemies._rotate_enemy`, `COOLDOWN_STICK` | `golden_enemies`, `oracle.step_enemy_round` |
| `$1805` | `rotate_enemy` | one fixed ±20-unit step, then `$187B JSR $1F9F` redraws the enemy | `enemies._rotate_enemy`, `passcost.ROTATE`, `relative.update_object_on_screen_cycles` | `golden_enemies`, `oracle.step_enemy_round`, `test_the_object_screen_span_is_exact_against_the_roms_own_209b` |
| `$1813` | (in `rotate_enemy`) | reloads the rotation cooldown to 200 | `enemies.ROTATION_COOLDOWN_RELOAD` | `golden_enemies`, `oracle.step_enemy_round` |
| `$1825` | `target_object` | records the target and ARMS `$0C20` to 120 rounds | `enemies._target_object`, `DRAINING_COOLDOWN_RELOAD` | `golden_enemies`, `oracle.step_enemy_round` |
| `$1838` | (in `consider_reducing_object`) | only FULL sight drains | `enemies._target_object` | `golden_enemies`, `oracle.step_enemy_round` |
| `$183D` | `consider_reducing_object` | fires the drain when the countdown expires | `playerbase._meanie_window` | `golden_meanie` |
| `$184D` | (in `consider_reducing_object`) | partially visible player → the meanie branch | `enemies._target_object` | `golden_meanie` |
| `$1852` | (in `consider_reducing_object`) | branch into meanie creation | `playerbase._meanie_window` | `golden_meanie` |
| `$1869`/`$186E` | (in the meanie path) | zeroes the draining cooldown, then reloads the update cooldown to 50 (`#$32`) | `enemies.UPDATE_COOLDOWN_MEANIE_MADE` | `golden_meanie` |
| `$1887` | `check_if_enemy_can_see_object` | horizontal FOV `±($0C68/2)` of the *current* facing | `relative.can_see_object` | `golden_relative` |
| `$18B8` | (in `check_if_enemy_can_see_object`) | the cone gate itself | `relative.can_see_object`, `playerbase._in_cone` | `golden_relative` |
| `$18E6` | `check_if_enemy_has_line_of_sight_to_object` | two `$1CDD` probes → `$0014` full/partial/unseen | `relative.can_see_object` (two `los` probes) | `golden_relative` |
| `$191F` | `calculate_player_exposure` | aggregates every enemy targeting the player |  | not modelled ([Not modelled](#not-modelled-deliberate-scope)) |
| `$194D` | `set_bar_state` | drives the on-screen exposure bar `$0C4F` |  | not modelled ([Not modelled](#not-modelled-deliberate-scope)) |
| `$197D`/`$1986` | `consider_creating_meanie` | deterministic slot scan for an eligible tree | `enemies._consider_creating_meanie` | `golden_meanie` |
| `$19A1` | `attempt_to_create_meanie` | FOV widened to `$28`; needs full sight of the tree | `enemies._consider_creating_meanie`, `FOV_CREATE_MEANIE` | `golden_meanie` |
| `$19C3`/`$19D5` | (in `attempt_to_create_meanie`) | the precondition: a tree within 10 tiles in x and y | `playerbase._tree_near` | `golden_meanie` |
| `$19F0` | (in `attempt_to_create_meanie`) | flips the tree's type to 4 in place — no slot allocated | `enemies._consider_creating_meanie` | `golden_meanie` |
| `$19FA` | `pause_meanie_creation` | retry next tick if the scene is being plotted |  | the sim has no plot state to defer on |
| `$1A00` | `kill_player` | a drain arriving at zero energy is death | `memmap.PLAYER_DIED_BY_DRAINING`, `actions.player_dead` | `golden_enemies`, `oracle.step_enemy_round` |
| `$1A08` | `reduce_object_energy` | −1 player energy, or robot → boulder → tree → gone | `enemies._reduce_object_energy` | `golden_enemies`, `oracle.step_enemy_round` |
| `$1A31` | (in `reduce_object_energy`) | re-zeroes the draining cooldown after each drain | `enemies._reduce_object_energy`, `playerbase._drain_clock` | `golden_enemies`, `oracle.step_enemy_round` |
| `$1A5D` | `consider_discharging_enemy_energy` | re-emits banked energy as a tree in a random low tile | `enemies._consider_discharging_enemy_energy` | `golden_enemies`, `oracle.step_enemy_round`; relocation tile unread |
| `$1A97` | play setup | the play-mode entry sequence | `landscape.generate` | `golden_landscape` |
| `$1AB0` | `find_drainable_boulder_or_tree_on_stack` | dismantles anything standing on a stack (`flags >= $40`) | `enemies._find_drainable_boulder_or_tree` | `golden_enemies`, `oracle.step_enemy_round` |
| `$1B00` | (in `consider_enemy_state`) | `SEC / BIT $0C1F / BPL $1B17` — returns carry set without doing anything unless the plot flag `$0C1F` bit 7 is set |  | deliberately unmodelled: a no-op on the common path |
| `$1B18` | `handle_player_actions` | dispatch; builds the aim vector for every code `< $22` | `aim.resolve` | `golden_actions` |
| `$1B1F` | `handle_hyperspace` | the hyperspace action | `actions.hyperspace` | `golden_actions` |
| `$1B2F` | `handle_uturn` | `objects_h_angle ⊕ $80` — free instant 180° flip | `aimcost.h_press_count`, `kbd_aim._uturn` | `test_aimcost.py`, `driver/test_live_determinism.py`; [7](open_items.md#7-the-drivers-wall-clock-timeouts-are-the-residual-load-sensitivity) |
| `$1B40`-`$1B46` | (in `handle_player_actions`) | the LOS gate; carry set → bad-action sound, no effect | `aim.resolve`/`gate` | `golden_actions` |
| `$1B64` | `try_to_transfer_into_object` | move `player_object` into a visible type-0 robot | `actions.transfer` | `golden_actions` |
| `$1B6E` | `find_platform_below_player_loop` | sets `$0CE6` when the new body stands on the platform | `actions.on_platform` | `golden_actions` |
| `$1B82` | (in `try_to_transfer_into_object`) | starts the transfer tune (`start_tune $888F`, tune `#$19`) | `projector.TUNE_TRANSFER_FRAMES` | `test_transfer_tune_is_96_frames` |
| `$1B8E` | `try_to_absorb_object` | absorb the topmost object in the target tile | `actions.absorb`/`can_absorb` | `golden_actions` |
| `$1B91` | (in `try_to_absorb_object`) | `LDA $0100` — absolute read of `objects_flags[0]`; the Sentinel lock | `actions.can_absorb` (`SENTINEL_SLOT`) | `golden_actions` |
| `$1B9A` | (in `try_to_absorb_object`) | the platform (type 6) can never be absorbed | `actions.can_absorb` | `golden_actions` |
| `$1B9E` | `absorb_object` | removes the object and banks its energy | `actions.absorb` | `golden_actions` |
| `$1BBA` | `try_to_create_object` | slot, energy, placement, refund on placement failure | `actions.create` | `golden_actions` |
| `$1BBF` | (in `try_to_create_object`) | a create may spend the meter down to 0 | `actions.create`, `playerbase._affords` | `golden_actions` |
| `$1BE0` | (in `try_to_create_object`) | a created robot faces `creator_angle ⊕ $80` | `actions.create`, `playerbase._settle_eye` | `golden_actions` |
| `$1BEC` | `try_to_absorb_meanie` | +1 energy and clears the parent enemy's meanie link | `actions.absorb` | `golden_actions`, `golden_meanie` |
| `$1C10`/`$1C13` | `prepare_vector_from_player_sights` | cursor + view angles → aim vector; reads neither state nor slot | `los.prepare_vector_from_player_sights` | `golden_los` |
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
| `$1E30` | (in `$1E0E`) | a platform adds `+$20` on top | `los._get_tile_z_from_object`, `landtable.surface_bounds` | `golden_los` |
| `$1E3F` | `get_tile_z_from_object` | walks the stack recursively | `los._get_tile_z_from_object` | `golden_los` |
| `$1E48` | `get_boulder_or_tree_z_for_line_of_sight` | boulder/tree top; needs fraction `< $40` | `los._boulder_or_tree_z` | `golden_los` |
| `$1E5A` | (in `$1E48`) | a boulder sits `-$60` at the bottom of the band | `los._boulder_or_tree_z`, `landtable.surface_bounds` | `golden_los` |
| `$1E69` | `is_tree` | the enemy-can-see-a-tree marker `$0CDD` | `los._is_tree` | `golden_los` |
| `$1EA4` | `get_height_of_lowest_object` | resolves a stack down to its base | `los._get_height_of_lowest_object` | `golden_los` |
| `$1EAF` | `get_minimum_x_or_y_fraction_from_tile_centre` | "targeted" only if the ray threads near centre | `los._get_min_xy_fraction` | `golden_los` |
| `$1ECC` | `get_object_details` | seeds the march at the observer tile's centre | `los._get_object_details`, `landtable.seed_z` | `golden_los` |
| `$1EEF` | `remove_object` | unlink the object and repair the tile it stood on | `actions._remove_object` | `golden_actions` |
| `$1EFF`/`$1F16` | `put_object_in_tile` | ground, or stacked on a boulder (+½) or platform (+1) | `enemies._put_object_in_tile`, `landscape._put_object` | `golden_actions`, `golden_landscape` |
| `$1F38` | (in `put_object_in_tile`) | refuses a create on a tile that already carries anything else | `actions.can_create` | `golden_actions` |
| `$1F83` | (in `put_object_in_tile`) | random initial facing `(prnd & $F8) + $60` | `actions.create`, `landscape._put_object` | `golden_landscape` |
| `$1FA4` | dither loop | the create/absorb post-action dither, loads `#$19` into `$2099` | `actioncost.DITHER_FRAMES` | `test_settle_accuracy.py`; [4](open_items.md#4-per-step-frame-drift-and-the-unattributed-createabsorb-settle-split) |
| `$2051` | (in the settle path) | loads `#$28` instead when `$0C4E` (meanie-made) is set |  | the meanie split, not the create/absorb one ([4](open_items.md#4-per-step-frame-drift-and-the-unattributed-createabsorb-settle-split)) |
| `$2099` | settle counter | the post-action dither countdown | `actioncost.SETTLE` | `test_settle_accuracy.py`; [4](open_items.md#4-per-step-frame-drift-and-the-unattributed-createabsorb-settle-split) |
| `$210E` | `create_object` | the highest empty slot, typed | `enemies._create_object`, `landscape._create_object` | `golden_actions` |
| `$2120`/`$2122` | `create_object_from_action` / `find_empty_slot_loop` | slots 63→0 | `actions._find_empty_slot` | `golden_actions` |
| `$2136` | `gain_or_lose_energy_from_object` | absorb adds, create subtracts | `energy.gain`/`lose` | `golden_actions` |
| `$2143` | (in `$2136`) | carry set on underflow = "not enough energy" | `energy.lose` | `golden_actions` |
| `$2148` | `set_player_energy` | every write masks `AND #$3F` — over-absorb wraps mod 64 | `energy.gain`, `memmap.ENERGY_MASK` | `golden_actions` |
| `$214F` | `energy_in_objects` | `03 03 01 02 01 04 00` by type | `energy.value`, `memmap.ENERGY_IN_OBJECTS` | `golden_actions` |
| `$2156` | `do_hyperspace` | new robot on a random flat tile of height `<= player_z + 1` | `actions.hyperspace`, `enemies.do_hyperspace` | `golden_actions`; landing tile unread |
| `$215F` | (in `do_hyperspace`) | kills below the 3-energy toll | `actions.player_dead` | `golden_actions` |
| `$216A` | (in `do_hyperspace`) | spends the 3-energy toll | `actions.hyperspace` | `golden_actions` |
| `$2170` | (in `do_hyperspace`) | kills on underflow | `actions.hyperspace` | `golden_actions` |
| `$217F` | `player_survived_hyperspace` | sets `$0CDE` bit 6 when the jump left the platform tile | `actions.won` | read back out of live memory by the driver |
| `$21AE` | `plot_stack_of_objects` | the per-tile object stack draw | `projector._inview_object_base` | `golden_projector` |
| `$22AA` | `span_fill` | middle-of-polygon fill, 4 px/byte |  | object `span_fill` unmodelled ([5](open_items.md#5-terrain-fill-cost-cannot-close-per-tile)) |
| `$23D0` | `plot_middle_of_row` | per-row span emit | `projector` fill proxy | [5](open_items.md#5-terrain-fill-cost-cannot-close-per-tile) |
| `$245B` | `populate_tile_visibility_bit_table` | raytraced occlusion into the `$3E80`/`$24DA` bitmap | `projector.occlusion_visible` | tile-for-tile against the ROM `$3E80` bitmap |
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
| `$2845` | `check_if_tile_is_on_screen_and_calculate_screen_coordinates` | the per-tile examine (trig floor) | `projector._project`, `C_EXAMINE` | `golden_projector` |
| `$28D4` | `calculate_tile_address` | render-path tile addressing | `memmap.tidx` | `golden_landscape` |
| `$295D` | `plot_row_of_tiles_or_block` | the plot loop over a row | `projector._project_scene_py` | `golden_projector` |
| `$2993` | `initialise_buffer_variables` | selects the buffer window (`$29C4`) for a pan or the play view | `projector.BUF_WINDOW`, `pancost.PAN_MODE` | `golden_pan_cost` |
| `$2A24` | `plot_tile` | gates only on `$0180 != 0` | `projector._project_scene_py` | `golden_projector` |
| `$2A8A` | `plot_two_triangles` | a sloped tile is two triangles, a flat tile one quad | `projector._terrain_poly_base` | `golden_projector` |
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
| `$2D6C` | `prepare_polygon` | per-polygon edge setup, run twice per wide-buffer section | `projector._terrain_poly_base` | `golden_projector`; per-call floor only ([5](open_items.md#5-terrain-fill-cost-cannot-close-per-tile)) |
| `$2D93`/`$2DCF` | `convert_angles_into_screen_coordinates` | vertex angles → `$A7A0`/`$0B40` screen coordinates | `projector` conv term | `golden_render_cost` |
| `$2DF2`/`$3002` | `process_line` | the DDA edge walk writing `$AD00`/`$AE00` | `projector` edge-walk term | `golden_render_cost`; [5](open_items.md#5-terrain-fill-cost-cannot-close-per-tile) |
| `$2EB2`/`$2EB7` | (in `process_line`) | `STA $AD00,Y` / `STA $AE00,Y` — the only writes to the left/right edge tables, one row at a time | `projector` edge-walk term | `golden_render_cost`; [5](open_items.md#5-terrain-fill-cost-cannot-close-per-tile) |
| `$2F58` | (in `process_line`) | the steep inner loop | `projector` steep inner loop | `golden_render_cost` |
| `$31CA` | `prnd` | 40-bit LFSR over `$0C7B-$0C7F`, 8 shuffles per call | `prng.Prng` | `golden_prng` |
| `$339A` | `get_random_two_digit_bcd_number` | one `prnd` draw per call | `landscape._initialise_player_and_trees` | `golden_landscape` |
| `$33ED` | `seed_prnd_from_landscape_number` | seeds `state[0..1]` from the typed number as packed BCD | `landscape.seed_for`, `core.landscape_from_digits` | `golden_prng`, `test_landscape_numbering.py` |
| `$3426` | `get_maximum_number_of_enemies` | geometric draw centred on the thousands digit + 2 | `landscape._max_enemies_second_cap` | `golden_landscape` |
| `$3451` | `get_random_number_between_0_and_22` | one draw, range-limited | `landscape._rnd_0_22` | `golden_landscape` |
| `$34DE` | `play_tune` | walks `$AB50 + tune_number`; note holds count down in `$0CDF` | `projector.TUNE_TRANSFER_FRAMES` | `test_transfer_tune_is_96_frames` |
| `$357D` | `play_landscape_loop` | the full viewpoint settle | `projector.viewpoint_replot_frames`, `playerbase._settle` | `test_settle_accuracy.py` |
| `$35A4` | load signature | `A5 0B 85` — the driver's proof the game is resident | `boot.SIG_ADDR` | `driver.dump_stage2.verify` |
| `$35BA` | (in `play_landscape_loop`) | calls the occlusion raytrace | `projector.occlusion_visible` | tile-for-tile against the ROM `$3E80` bitmap |
| `$35C3`/`$35C6` | (in `play_landscape_loop`) | the two `plot_world` passes | `projector.REPLOT_PASSES`, `playerbase._settle_eye` | `test_settle_accuracy.py` |
| `$35D5` | `wait_for_end_of_tune` | spins until the tune's bit 7 sets | `projector.TUNE_TRANSFER_FRAMES` | `test_transfer_tune_is_96_frames` |
| `$3603` | `landscape_completed` | sets `$0CDE` bit 6 — the win | `memmap.LANDSCAPE_COMPLETE`, `actions.won` | read back out of live memory by the driver |
| `$363D` | `update_game_and_continue` | the main loop; no vsync wait | `enemies.advance_frame` | instrument gate `test_enemy_sim_frame_locked_to_live_ls42`; [8](open_items.md#8-the-enemy-clock-what-is-left-is-the-redraw-and-the-frame-budget) |
| `$3642` | viewpoint redraw entry | into `play_landscape_loop` | `kbd_aim._run_to_scan` | `test_settle_accuracy.py` |
| `$365A`/`$365D` | (in the main loop) | the `JSR pan_viewpoint` call site and the pan-done PC | `kbd_aim.PC_PAN_DONE` | `driver/test_live_determinism.py`; [7](open_items.md#7-the-drivers-wall-clock-timeouts-are-the-residual-load-sensitivity) |
| `$3682` | (in the main loop) | skips the enemy clock while `$0CE5` bit 7 is set | `playerbase._frozen`, `actions._mark_player_acted` | `test_settle_accuracy.py` |
| `$3684` | scroll loop | ticks cooldowns while scrolling; mutually exclusive with `$9663` | `enemies.cooldown_frame` | instrument gate `test_enemy_sim_frame_locked_to_live_ls42`; [8](open_items.md#8-the-enemy-clock-what-is-left-is-the-redraw-and-the-frame-budget) |
| `$3700` | grid angle/hypotenuse pass | fixed per-settle foreground work | `projector.SETTLE_FIXED_FRAMES` | `test_settle_accuracy.py`; [4](open_items.md#4-per-step-frame-drift-and-the-unattributed-createabsorb-settle-split) |
| `$3B00`/`$3C01` | arctan coefficient tables | reproduced closed-form, byte-exact | `relative._ARCTAN_LO`/`_HI` | closed form, byte-exact against the ROM table |
| `$3D02` | hypotenuse coefficient table | reproduced closed-form, byte-exact | `relative._HYP` | closed form, byte-exact against the ROM table |
| `$8401` | `calculate_object_relative_angles_and_distance` | relative x/y (`$85C4`), z (`$85F5`), then the angles | `relative.relative_angles` | `golden_relative` |
| `$8475` | object transform loop | per-vertex `transform_vertex` | `projector.C_VERTEX` | `golden_render_cost` |
| `$8533` | `plot_object` | the object model draw | `projector.C_VERTEX` | `golden_render_cost`; object `span_fill` unmodelled ([5](open_items.md#5-terrain-fill-cost-cannot-close-per-tile)) |
| `$888F` | `start_tune` | begins a tune, number in `$0CE7` | `projector.TUNE_TRANSFER_FRAMES` | `test_transfer_tune_is_96_frames` |
| `$9287` | `calculate_angle` | bearing from a relative x/y pair | `relative._calc_angle` | `golden_relative` |
| `$933D` | `calculate_object_relative_vertical_angle` | pitch from z and distance | `relative._vertical_angle` | `golden_relative` |
| `$937F` | `calculate_hypotenuse` | horizontal distance | `relative._calc_hypotenuse` | `golden_relative` |
| `$9630` | raster frame marker | `DEC $0CDF`; one `$9630`→`$9630` span is exactly one frame | `driver.clock.frames`/`run_frames` | instrument gate `test_enemy_sim_frame_locked_to_live_ls42`; [8](open_items.md#8-the-enemy-clock-what-is-left-is-the-redraw-and-the-frame-budget) |
| `$9659` | (in the raster IRQ) | skips the enemy clock while frozen | `enemies.cooldown_frame` | instrument gate `test_enemy_sim_frame_locked_to_live_ls42`; [8](open_items.md#8-the-enemy-clock-what-is-left-is-the-redraw-and-the-frame-budget) |
| `$9663` | (in the raster IRQ) | the once-per-frame cooldown tick | `enemies.cooldown_frame` | instrument gate `test_enemy_sim_frame_locked_to_live_ls42`; [8](open_items.md#8-the-enemy-clock-what-is-left-is-the-redraw-and-the-frame-budget) |
| `$9678`/`$967B` | gated full input scan | the driver's press window | `kbd_aim._run_to_scan`, `playerbase.TAP_FRAMES` | `driver/test_live_determinism.py`; [7](open_items.md#7-the-drivers-wall-clock-timeouts-are-the-residual-load-sensitivity) |
| `$98B2` | `plot_status_bar` | fixed per-settle foreground work |  | folded into `projector.SETTLE_FIXED_FRAMES`; [4](open_items.md#4-per-step-frame-drift-and-the-unattributed-createabsorb-settle-split) |
| `$9925` | `PAN_DELTA` table | `$14/$F8/$04/$F4` added before the pan's `plot_world` | `pancost.PAN_DELTA` | `golden_pan_cost` |
| `$9939`/`$994F` | pan buffer-mode entries | vertical `A=#$00` (play window), horizontal `A=#$02` | `pancost.PAN_MODE`, `projector.BUF_WINDOW` | `golden_pan_cost` |
| `$9958` | `move_sights` | steps cx and cy in ONE call — the cursor moves diagonally | `los.landable_views`, `kbd_aim.fine_cursor` | `test_landable.py`, `test_landtable.py`; [10](open_items.md#10-landability-filter-unproven-corners) |
| `$9965`/`$9994` | (in `move_sights`) | ±1 px per gated scan; clamps cx `$10-$8F`, cy `$20-$9F` | `los.CURSOR_CX`/`CURSOR_CY` | `test_landable.py`, `test_landtable.py`; [10](open_items.md#10-landability-filter-unproven-corners) |
| `$9CA0`/`$9CA1` | object vertex counts | per model type | `projector._OBJECT_MODEL` | `golden_render_cost` |
| `$9CAB`/`$9CAC` | object polygon counts | per model type | `projector._OBJECT_MODEL` | `golden_render_cost` |
| `$9D37` | rotation speed table | the per-enemy ±20 step, in RAM — inside a ROM `LOADED` region, so `oracle.machine_from_image` overwrites it ([the 6502 oracle](#the-6502-oracle-sentineltestsoraclepy)) | `memmap.ROTATION_SPEED_TABLE` | `golden_enemies`, `oracle.step_enemy_round`; seeded from the live image by the instrument |

---
## The model (`sentinel/`)

| Module | Role |
|--------|------|
| `memmap.py` | RAM addresses, object types, the interleaved tile index |
| `prng.py` | the 40-bit LFSR `prnd` and landscape seeding |
| `state.py` | the canonical state: a 64 KB `bytearray` laid out like the game's RAM, with typed object-array views |
| `statecmp.py` | the labelled/tiered address schema shared with the instrument |
| `terrain.py` | height/slope nibble decode and the slope-facet surface |
| `los.py` | the LOS ray-march, the sights aim vector, and the keyboard-aim buildability oracle (`landable_views`/`landable_view`/`landable_sweep_with_centres`) |
| `los_jit.py` | numba fast-march of the hot LOS loop, bit-identical to `los.py` |
| `landtable.py` | closed-form landability superset filter in front of every per-tile aim query |
| `aim.py` | the one action aim/LOS layer (`resolve`/`gate`/`propose`) — the `$1B40`-`$1B46` gate |
| `aimcost.py` | keyboard-aim geometry: keystrokes to pan a heading, u-turn-aware |
| `pancost.py` | per-notch pan redraw cost, ported from `pan_viewpoint $10B7` |
| `projector.py`, `projector_jit.py` | `plot_world $2625` terrain projector, ported bit-exactly; feeds the render-cost proxy |
| `rendercost_py65.py` | exact `plot_world` frame cost by running the real 6502 in py65, memoized; ROM-gated |
| `actioncost.py` | per-action world advance: the ROM dither/replot frame counts and the `$1335`/`$0C50` frame→tick cadence |
| `actions.py` | absorb / create / transfer / hyperspace / win (the LOS gate is the caller's) |
| `energy.py` | the energy economy (`$2136`, table `$214F`, 6-bit mask, underflow) |
| `landscape.py` | `generate(landscape) -> State`, the board generator |
| `relative.py` | object-relative bearing/distance/vertical angle, enemy FOV and visibility |
| `enemies.py`, `enemies_jit.py` | the enemy machine and the frame clock (`advance_frame`/`advance_frames`) |
| `passcost.py` | cycle cost of one `$1289` play-loop pass, counted from the ROM: the enemy clock's cadence |
| `threat.py` | any-rotation tile exposure, gaze distance, ticks-until-seen, meanie safety, drain-over-window |
| `game.py` | `Game`, the facade |
| `playerbase.py` | shared player machinery: world clock, geometry, gaze windows, aim cost, firing, run loop |
| `phase_player.py`, `player.py` | the two players ([players.md](players.md)) |
| `statecache.py`, `atlas.py` | the landscape atlas: cached board images, and the metrics measured off them |
| `isoview.py` | isometric SVG of any `State` |

`State` is a mutable `bytearray` image; `Game.clone()` deep-copies it so a search branches
without side effects.

### Validation

Mechanics are differentially validated against the 6502 via `sentinel/tests/oracle.py`, then
frozen as JSON goldens replayed by CI — the fifth column of the
[routine table](#rom-routines-and-the-model) says which golden covers which routine. Four
results are worth stating on their own:

- **The board is exact.** `driver.dump_stage2.verify` requires 1024/1024 tiles against the
  ROM's own generator run in the emulator.
- **The enemy round is exact.** Seeded with a divergent ls335 state, `enemies.step` is
  byte-exact against `oracle.step_enemy_round` for 119 rounds — the transition function,
  branches included, is right. The meanie lifecycle is pinned round for round on landscape
  2024 to round 2486 plus the failed-attempt path (landscape 49).
- **No game data is embedded.** The arctan (`$3B00`/`$3C01`) and hypotenuse (`$3D02`)
  coefficient tables are reproduced from closed-form expressions verified byte-exact against
  the ROM.
- **A 6502 interpreter is a sound instrument.** Tracing generation, a full `plot_world` frame
  and 400 enemy rounds hits none of the 105 illegal opcodes.

The frame clock is the one mechanic with no golden: it is gated by
[the instrument](#the-divergence-instrument-driverinstrumentpy) and by
`test_irq_cycles_matches_the_live_pass_rate` instead.

### Not modelled (deliberate scope)

- **PRNG-driven landing coordinates.** `actions.hyperspace` (`do_hyperspace $2156`) is
  faithful and `win` is gated on `$0CDE` bit6+7 with the 3-energy cost and
  death-if-underfunded, but the landing tile of a hyperspace or meanie relocation is
  deliberately unread. The draw *rate* is unmodellable — the ROM draws many times per frame
  against a cursor that moves a few steps, so PRNG phase is not observable in play. This
  limits exactly two things, both through `put_object_in_random_tile_below_z $1224`: the
  discharge tree's tile and the hyperspace tile. Meanie creation is **not** one — `$197D` is
  a deterministic slot scan, as are the hunt and the hyperspace trigger.
- **The u-turn as a player action.** The free 180° flip (`$1B2F`) is priced in
  `aimcost`/`playerbase` but is not in `actions.py`.
- **Exposure-bar aggregation** (`$191F`/`set_bar_state $194D`/`$0C4F`); the underlying
  two-probe `$0014` is modelled.
- **Meanie death-credit `$0C1C = 4`** — affects only death-screen attribution.
- Sound side effects carry no gameplay state.

## Aim cost (`playerbase._aim_frames`)

Priced mechanism for mechanism against the executor's key sequence, over the aim lattice
[the geometry imposes](#what-the-geometry-permits).

- **Body pan is two keystroke ramps** (`pancost.pan_frames`), each notch followed by one
  `plot_world`: horizontal `$10EE` = 16 scroll steps per ±8 bearing notch, vertical `$1135` =
  8 steps per ±4 pitch notch. Notch counts from `aimcost.h_press_count` (u-turn-aware,
  returns `(n_uturn, n_step)`) and `aimcost.v_steps`.
- **U-turn** = one action tap (`UTURN_FRAMES`), no scroll and no redraw; taken only when it
  strictly lowers the keystroke count (crossover at `d >= 9` lattice steps). Pooled live
  n=9 (ls42 p1 plus the ls335 win's eight), mean 76.6 over samples spanning 33–180 f — a
  central value, not a bound.
- **Cursor is derived, not fitted.** `move_sights $9958` steps both axes in one call at 1 px
  per gated scan, so a drive costs `max(|Δcx|, |Δcy|)` scans plus `CURSOR_RAMP =
  popcount($6B) = 5` scans the `$0CC8` auto-repeat mask skips. Zero if the cursor is parked.
- **Sights toggle is a state transition.** A same-bearing reuse keeps sights on and drives
  from the live cursor at zero toggle cost; otherwise `TOGGLE_FRAMES` is charged and `$134C`
  re-centres to `SIGHTS_CENTRE = (80, 95)`.
- **A transfer charges 0 aim only on a bearing reuse** — the executor sends no aim keys then,
  `$21` firing on the object the preceding same-tile create/absorb parked the cursor over.
  Live, the predicate is the driver's, adopted by `LiveMixin._sync_aim_state`.
- `HOP_FRAMES` is the window a full hop (2 creates + transfer + aims) needs; `SAFE_FRAMES` is
  the window below which a tile is urgent. Both are pinned against the live ls42 whole-step
  hops in `live_ls42_hops.json` (`test_hop_budget.py`), and charge slightly under measured.

**Which view a tile gets.** A tile is landed by tens of thousands of lattice rays and the
players use exactly one, so the pick is part of the cost model, not a detail:
`playerbase._cheapest_ray` returns the **frame-minimal** ray, proved. The price decomposes —
the pan term depends only on `(h_angle, v_angle)`, the cursor term only on the cursor and
monotonically in its drive distance — so one representative per `(h, v)` cell (the cursor
nearest `SIGHTS_CENTRE`) attains the minimum, collapsing ~20k rays to ~10 groups. Groups are
then priced with the real `aim_frames` in `_aim_bound` order (`aim_frames` with every notch's
`render_cost` dropped: admissible, since a render is never negative), stopping at the first
bound that reaches the best price — 4–32 exact evaluations per tile, measured on ls0/42/110/335.
The predecessor was a `1000·bearing + 100·pitch + cursor` proxy whose argmin is not the
argmin of frames: it left 2/50 tiles suboptimal on ls0 and 6/23 on ls42, by up to 141 f.

## The landability filter (`landtable.py`)

A sound **superset filter** over the keyboard-aim lattice: given an observer and a target
tile it returns the lattice rays that *can* land there, so a per-tile query marches thousands
of rays instead of a whole heading cone. It is the path behind `los.landable_view_targeted`.

`_get_object_details $1ECC` seeds every ray at `px_frac=py_frac=0`, `px_sub=py_sub=0x80` (the
eye tile's centre) and `prepare_vector_from_player_sights $1C10` reads neither state nor slot,
so `_add_vector $1CBB` gives, at sub-step `i`:

```
DX_i  = floor((0x8000 + i*vx) / 65536)             tile offset from the eye tile
DY_i  = floor((0x8000 + i*vy) / 65536)
z16_i = eye_z*256 + obj_z_frac + floor(i*vz / 256)  the $003B:$0038 compare pair
```

A ray's track is a pure function of its aim; terrain only decides where the march stops.
`obj_z_frac` and `eye_z` are additive offsets (terms in the query threshold, never table
axes) and `DX`/`DY` are position-independent, so one condition serves every observer
(`test_closed_form_track_matches_add_vector`).

**The condition.** `check_flat_tile $1D0D` lands only when `D = surface16 - z16` is in
`[0, $80)` (`$0079` vs `$000C`, tightened to `$10` on the object path), where
`surface16 = tile_z*256 + $0079`; `D < 0` marches on, `D >= $80` blocks, and `|vz| <= 4095`
bounds `z16` to 16 per sub-step. So the ray must, at some sub-step inside the cell, have
entered above the band (`z_entry > surface16 - $80`) **and** reach the surface
(`min z16 <= surface16`); for a climbing ray the entry sub-step must satisfy both.
`crossing_mask` tests that in O(1) per ray, needs no storage, and composes with the
heading-arc bisection `los._tile_arc_indices`. It is a superset because terrain can only stop
a march *earlier*; a flat-terrain table would not be sound — keying on the crossing height is
what makes it hold. `surface_bounds` returns `(lo, hi)`, exact for bare terrain and over the
whole stack otherwise (platform `$1E30` `+$20` on top, boulder `$1E5A` `-$60` at the bottom).
Two exact shortcuts (`never_lands`) come straight from the geometry: the observer's own tile
never lands (`$1D32`), and a sloping tile never lands (`check_sloping_tile $1D46` only loops
or blocks).

**Wrap safety.** The ROM compares z as bytes, so a ray far enough above a surface aliases onto
"equal". Those visits are kept unconditionally (wildcards, and `crossing_mask`'s `wrap_z`
branch), at a small cost per arc-narrowed query, so the answer stays sound either way — see
[open item 10](open_items.md#10-landability-filter-unproven-corners).

**Callers.** `playerbase._Views` is the one entry point the players use. `get`/`band_get`
answer a **single** tile with one targeted march (`_cheap_view` over `landtable.candidates`,
narrowed by `crossing_mask`); the whole-lattice sweep `los._landable_batch` is built only when
a caller reads the entire dict (`views.band()`). Each lattice is pinned to the board at its
first query (`_Views._pinned`), so a caller that builds or transfers mid-tick keeps reading
one consistent board. Both paths hand the tile's landing rays to the same `_cheapest_ray`
(see [aim cost](#aim-cost-playerbase_aim_frames)), so they return the same view for the same
tile by construction.

**Candidate generators.** A generator that wants *which of these tiles land* asks
`_Views.band_ordered(pick)`, never the whole dict: `pick` is the caller's visibility-free
filter (terrain, the object table, the purse), and the survivors are answered per tile. The
answer must be in the sweep's own **order**, because the climb tie-break reads it
positionally (`docs/players.md`, "The tie that decided ls110"); the sweep inserts by winning
lattice ray and `_cheap_view` returns that ray, so sorting on it reproduces the order exactly
(`test_band_ordered_is_the_sweep_order_without_buying_it`). Which side gets scanned is a cost
decision only, since both return the same views: with a sweep in hand the landable set is
filtered, and an ask of `BAND_SWEEP_TILES` marchable tiles or more buys a sweep instead —
measured 3.7-4.4 ms per targeted band march against 0.83-1.08 s per band sweep, so the
crossover is 205-260 tiles. What remains sweep-bound is
[open item 1](open_items.md#1-a-per-tile-candidate-generator-only-pays-once-the-eye-is-up).

**Lattices.** `los._landable_sweep`'s plane and band (`landtable.MAX_STEPS = 6000`). Over
targeted queries from real solves (ls0/42/335), arc bisection alone marches a large fraction of
the arc; adding `crossing_mask` (`landable_view`) cuts the rays marched several-fold and answers a
majority of ls42 queries as a *proven* "no view" without marching at all. The expensive
residual is an **adjacent** cell: its arc is huge and a ray dwells long enough inside it to
cross almost any surface height, so most of the arc survives the filter.

**Whole-board set.** `landable_set` answers over the landset lattice — the band's `hgrid` and
`los._V_PRIORITY` with the cursor subsampled 2:1 (`COARSE_CX`/`COARSE_CY`), 884,736 rays.
Summing per-cell queries re-marches rays, so `stop_cells` inverts it: walk each ray's track
against the real surface map and name the tile(s) it can stop in, partitioning the lattice by
landing tile. Most of the lattice crosses no surface band before leaving the board and is
never marched, so the whole-board answer costs several times fewer rays than the per-cell sum
while staying exact tile for tile including the ring. Soundness needs one extra step: the walk
may only stop where the ray **provably** stops (`zmin <= surface_lo`); an undecidable cell
(object stack with `lo < hi`, or alias risk) makes the ray multi-candidate and always marched
(`test_stop_cell_partition_holds_every_landing`).

**Validation.** `test_landtable.py` pins the property everything rests on — over several
boards, a stacked/raised-eye stance and an `eye_z` override, every ray the full sweep lands on
a tile is in that tile's candidate set, on all three lattices including below-eye, object and
outer-ring tiles. `test_landable.py` pins the same through `los.landable_view_targeted` for
every tile of the band and coarse lattices, and through `_view_for` for the plane. Start
states alone do not exercise the object-stack surface bracket, so
`test_landable_view_matches_sweep_every_tile_midgame` re-checks every tile of a board the
player has already built and transferred on.

## Render cost (`projector.py`, `pancost.py`, `rendercost_py65.py`)

The game is compute-bound, so drawing the height field *is* the world clock: what a view costs
to plot is how many frames an action spends. `FRAME_CYCLES` = 19656 (PAL). Validated against
`golden_render_cost.json` (py65 cycle counts, 15 views over generated boards
0/42/66/335/777/2024) with the raytraced occlusion table active.

`plot_world $2625` is an equirectangular rasteriser walking the 32×32 grid furthest-to-nearest
(`$26DE` counts `$0026` 31→0; per row `$27D7` finds the span via `$2845`; each plotted tile
runs `plot_tile $2A24` → `prepare_polygon $2D6C` / `process_line` / `span_fill $22AA`, object
tiles adding `$21AE`/`$8533`).

`render_cost(state, view, observer, mode)` = examine floor + `prepare_polygon` floors + area
fill proxy, over `FRAME_CYCLES`, memoized on `(scene_key, observer, h, v, mode)`. With
`RENDER_COST_BACKEND=py65` and the ROM fixture present, the play-buffer player view is the
exact py65 cycle count instead ([open item 6](open_items.md#6-the-py65-exact-backend-skips-transfer-settles)).

| term | exactness |
| --- | --- |
| (a) examine trig floor: `$2845` + `$9287` + `$937F` + `$933D` | count **exact**; cost `N * C_EXAMINE` (py65-derived) |
| (b) terrain fill | plotted set **exact** (`$0180` gate); per-tile cycles approximate |
| (c) object fill | plotted set **exact**; per-object base floor, `span_fill` unmodelled |

**Occlusion is exact.** `projector._occlusion_visible` is a byte-exact port validated
tile-for-tile against the ROM `$3E80` bitmap: (1) temp height table `$25C4`, per tile
`(z<<1) | not_flat`; (2) horizon table `$25ED`, per tile the **minimum** of its four corner
bytes `>>1` (the CMP/BCC at `$2604`-`$2617` keeps the smaller — the ROM disassembly's
"maximum" label is a misnomer); (3) fixed-point DDA raytrace `$24E2` (`$2503` signed 3-axis
delta, `$2532` scale to ~2-4 substeps/tile, `$2576` march), blocking a tile whose ray dips
below the horizon table, then `$248A` ORs the 2×2 block and applies a height test, setting the
bit read at `$2911`/`$2916`. Occlusion changes **only** the plot byte: `$291B` zeroes
`$0180,X` so `plot_tile` skips at `$2A27 BEQ`, while `$007F` is untouched — occluded tiles are
still examined and pay the trig floor, removing roughly half the would-be-filled tiles. Object
tiles (`$28F0 CMP #$C0`) bypass occlusion. The raytrace starts at the passed observer, not
unconditionally at `state.player`.

**Tile selection and the `$0180` gate are exact.** `_scan_visible` ports `$27D7` + `$26DE` +
the observer-row tail `$276F` branch-for-branch off the byte-exact `$2845` result (the `$0C48`
furthest-row hint is 0 in every fresh play state). Three facts make it exact: the plot range is
`[$0037, $0038)` (split loops `$2961`/`$2975`, column `$0038` never plotted); there is **no**
on-screen filter, `plot_tile` gating only on `$0180 != 0` and height-0 flat tiles having byte
0; and the slot remap `(($0025|$0005)+$001B)&$3F` draws examine `(col+offc, row+offr)` with
`$001B = $27D3 = [$00,$01,$21,$20]` by quadrant. The observer's own tile is drawn by
`plot_checkerboard_tile $27CE`, outside the gate.

**Fill** is `_terrain_poly_base` (a `prepare_polygon` floor) plus
`sum(PER_SCANLINE*H + PER_PIXEL*H*W)` over kept tiles — an area proxy, not a fit. Vertex
projection `$2DCF`/`$2D93` is ported cycle-exact (`screen_x` = high byte of
`((h_angle16 + $0011:$0029) << 3)`), and the DDA edge walk reproduces the `$AD00`/`$AE00`
writes byte-for-byte on every narrow polygon-section swept. Per-block costs come from the loop
bodies: `process_line`'s steep inner loop `$2F58` is **23 cyc/row** (27 on a column step),
iterating exactly 2 × filled rows for an inside polygon; `span_fill` middle 8 cyc/byte at
4 px/byte; per-row edge plot `$23B5`/`$238C` ~55-70 cyc; rows walk `[$0052,$0051] = [48,240]`;
off-band `prepare_polygon` ~600 cyc/call (`C_PREP_CALL`). The fill is **prepare-dominated** —
some golden views fill zero pixels yet spend most of their terrain budget tracing edges for
polygons that clip out of the band — because `prepare_polygon` runs per polygon × 2
wide-buffer sections and a plotted tile is one quad or two triangles (`plot_two_triangles
$2A8A`).

**The edge tables carry state across polygons.** `polygon_left_edge_table $AD00` and
`polygon_right_edge_table $AE00` are **never cleared**: a linear scan of the image finds no
clear loop over those pages, and the only writes are `process_line`'s own per-row
`$2EB2 STA $AD00,Y` / `$2EB7 STA $AE00,Y`. A polygon clipping to a sliver therefore writes
only some of the `[$0004,$0006]` rows, and `span_fill $22AA` — whose middle-fill length is
`right_col − left_col` (`$22B7 LDA $AE00,X / $22BA CMP $AD00,X`) — reads columns a *previous*
polygon left behind. So the fill is a cross-polygon stateful sequence, not a per-tile
function, which is why the area proxy's residual cannot close:
[open item 5](open_items.md#5-terrain-fill-cost-cannot-close-per-tile).

**Object term.** `plot_object $8533` → transform loop `$8475`: per vertex `transform_vertex`
runs `calculate_sine_and_cosine` + two `multiply_byte_by_byte` + `$9287` + `$937F` + `$933D`,
charged as `C_VERTEX`, then per polygon the same `prepare_polygon`+`span_fill`. Model sizes
come from engine facts `$9CA0`/`$9CA1` (verts) and `$9CAB`/`$9CAC` (polys): type 0=(29,27)
1=(22,25) 2=(17,15) 3=(8,10) 4=(18,25) 5=(30,35) 6=(12,11) 7=(8,4). `_inview_object_base` sums
a fixed per-object base over plotted object tiles' `$0100` stacks and, with object `span_fill`
unmodelled, is a strict floor. Constants stay env-overridable (`RENDER_C_EXAMINE`,
`RENDER_PER_SCANLINE`, `RENDER_PER_PIXEL`, `RENDER_C_VERTEX`, `RENDER_C_PREP_CALL`,
`RENDER_SECTIONS`) but are ROM-derived: a perturbation smaller than the model's own error can
flip a knife-edge board, so tuning them to win one is evidence of nothing.

**Transfer settle `$357D`.**

    viewpoint_replot_frames = TUNE_TRANSFER_FRAMES + SETTLE_FIXED_FRAMES
                              + REPLOT_PASSES * render_cost(state, view, observer)

`$0C63` moves into the target in `try_to_transfer_into_object $1B64` **before**
`play_landscape_loop $357D` runs its two `plot_world` passes (`$35C3`/`$35C6`), so both
`render_cost` and the `$245B` raytrace run from the post-transfer eye at that body's own
bearing (a created robot faces `creator_angle ^ $80`, `$1BE0`) — not the aim view, which
belongs to the abandoned eye. `playerbase._settle_eye(verb, tile)` returns that slot.

`TUNE_TRANSFER_FRAMES = 96` is ROM-derived: `play_landscape_loop` ends at
`wait_for_end_of_tune $35D5`, spinning until the tune started at `$1B82` (`start_tune $888F`,
tune `#$19`) sets bit7; `play_tune $34DE` walks `$AB50 + tune_number`, a byte `>= $C8` setting
note length `$0C70 = (byte-$C8)*4` and a byte `< $C8` holding it in the `$0CDF` countdown
decremented once per frame by `$9630 DEC $0CDF` — note holds sum to 96 frames, the same as the
`#$0` hyperspace tune (`test_transfer_tune_is_96_frames`). `SETTLE_FIXED_FRAMES = 176` is a
stand-in for four foreground routines absent from `render_cost` — `$245B`, `$3700`,
`fill_screen_with_background $1090` and `plot_status_bar $98B2` — py65 cycle-counted on ls42
and ls335 and averaged, since the occlusion term is scene-dependent; raster-IRQ steal is folded
into it and the tune base.

`test_viewpoint_replot_lands_in_live_settle_band` asserts each prediction lands in
`[0.75*lo, 1.25*hi]` of the recorded live band with median abs error < 15%. That band was read
through a 6 s wall-clock `run_until_pc` in `tap_action` that caps a reading at ~300 frames, so
any value at or under that ceiling is indistinguishable from it. A u-turn scrolls 0 frames and
is not a viewpoint replot; `_exact_render_cost` returns `None` for any `observer !=
state.player`.

**Per-notch pan.** One keyboard notch is one `pan_viewpoint $10B7` call and `notch_frames` is a
direct port: the strip clear (`$3912` h / `$38AD` v), the ONE `plot_world` at the
**intermediate** angle in that direction's `$2993` buffer mode, and the notch's queued 16 h /
8 v scroll steps. The `$9925` delta (`PAN_DELTA = $14/$F8/$04/$F4`) is added before `JSR $2625`
and fixed up after, so a right pan plots at `h + $14` and a downward pitch at `v - $0C`, while
left pans and upward pitches land on the destination. A horizontal pan is **not** the play
buffer: `$10EE` reaches `$2993` through `$994F` with `A=#$02`, whose `$29C4` window culls tiles
the play window keeps; a vertical pan (`$9939`, `A=#$00`) shares the play window. Examined and
filled counts are byte-exact on every row of `golden_pan_cost.json` (288 notches over
ls0/42/335), and measured notch cost spans more than an order of magnitude — a swing no flat
base covers, which is what `test_pan_notch_cost_matches_the_measured_plot` (rms < 9 f) and
`test_derived_notch_beats_the_flat_base_it_replaced` pin. The residual is the fill proxy, not
the notch model — do not add a compensating constant to `pancost`. The view-independent `$245B`
raytrace is memoized per (scene, observer) as `projector.occlusion_visible`, and `notch_frames`
per (scene, observer, direction, plot angle), both keyed off `projector.scene_key`, a digest of
every byte `plot_world` reads.

## The live driver (`driver/`)

Executes a plan against the real game in [VICE](https://vice-emu.sourceforge.io/) (asid-vice)
inside Docker, headless, and verifies each result from the game's own memory. Imports only
`sentinel/`. The boot sequence and the aim primitive are
[state machines](#the-driver-boot--enter--play).

    python -m driver.play_player 335                  # phase player (default), records an AVI
    python -m driver.play_player 0 --player greedy

| Module | Role |
|--------|------|
| `core.py` | container/boot/connect/navigate/record lifecycle (`boot_and_play`, `GameSession`, `validate_avi`), `SentinelDriver`, `live_image`, live LOS probe `probe_tile` |
| `kbd_aim.py` | pan/cursor cycles, `KbdDriver` (checkpoint-driven, u-turn-aware) |
| `sentinel_execute.py` | `Executor`, `perform_step` (aim → fire → verify), `fire_hyperspace`, `verify` |
| `live_player.py` | `LiveMixin` (observation + execution over live memory, no decision logic) composed with the sim players into `LiveGreedy`/`LivePhase`; `MeasuringKbdDriver` |
| `play_player.py` | runner → `out/play_player_<digits>.json` |
| `clock.py` | machine-side clock: `frames` (wrap-free `$9630` checkpoint hits), `run_frames` |
| `boot.py` | tape boot with load-signature polling, bridge-IP lookup, container reaping, snapshots |
| `sentinel_state.py` | live memory → `GameState` (`ViceSource`/`Py65Source`), `verify_entry` |
| `dump_stage2.py` | regenerates `out/sentinel_stage2.bin` from the tape (the `oracle` fixture) |
| `instrument.py` | the frame-locked divergence race |
| `frozen_run.py` | RTS-stubs `update_enemies $16B5` live: isolates frame-cost fidelity |
| `plan_audit.py` | per-step audit of each `PlanStep`'s recorded budget/windows vs live |
| `replay_human.py` | replays a recorded human line into `<fixture>_truth.json` |
| `watch_play.py` | passive logger of a human playing; logs `[0,$0CFF]` plus the enemy clock |

**No wall-clock waits.** Every wait keys on a PC or memory predicate, never `time.sleep`: a
host delay is warp-dependent (warp on under `NO_RECORD=1`, off while recording), so it would
make measured frame counts differ between modes. `test_no_sleep.py` is an AST guard; waits
outside the emulated machine carry an inline `# sleep-ok: <reason>`. In play,
`bm.auto_resume = False`, so the world moves only in deliberate `run_frames`/checkpoint windows
and think time is free. The pan cycle is machine-clocked frame by frame off `PC_PAN_DONE`,
terminating on `_PAN_STALL_FRAMES`/`_PAN_MAX_FRAMES` rather than elapsed time; the residual host
clocks are the remaining `kbd_aim` socket timeouts
([open item 7](open_items.md#7-the-drivers-wall-clock-timeouts-are-the-residual-load-sensitivity)).

**Aim → fire → verify.** A view is a bearing (8-unit lattice), a pitch (4-unit lattice, band
`$CD..$35`) and the cursor `(cx, cy)`; `$1C10` combines them as `h_eff = h + cx>>3`,
`v_eff = v + (cy-5)>>4`. A settled press moves the cursor 9 px but `$9965`/`$9994` step 1 px,
so any pixel is reachable (the search uses a step-3 window). `sentinel.aim.propose` searches
that grid with `los.aim_target` for a `(h, v, cursor)` whose native ray lands the tile,
preferring a low pan and a small tile-centre fraction, with the CPU halted.

Read h/v **sights-off**, where `objects_h_angle $09C0+slot` and `objects_v_angle $0140+slot`
are settled — sights-on, the foreground loop `$363D` calls `$10B7` every frame and its settle
dance (`+$14`, `JSR $2625`, `−$0C`) leaves the byte transiently off-lattice. The cursor
`$0CC6`/`$0CC7` is stable sights-on. The aim-vector scratch `$003D`/`$003E`/`$0040` is shared
with the enemy-relative-angle math and is never a stable source. `perform_step` confirms the
read-back angles and cursor reached the request (a mismatch means a clamp or no-converge →
**do not fire**, `aim_miss`), fires once, then `verify` arbitrates the memory delta: the exact
on-tile object-count change **and** the exact energy delta, any other global object-count change
being a divergence. A win is `$0CDE` bit 6 after `fire_hyperspace` from the platform tile.
`probe_tile` is advisory — it reads the live CPU asynchronously. A create/absorb leaves sights
ON and the bearing untouched (SPACE at `$11B3` is the only toggle), so a matching next bearing
drives only the cursor, skipping the `$134C` re-centre; a slot change or a non-converged pan
clears the committed bearing.

**Plan steps.** A step is `{verb, otype, target tile, view}` plus `min_energy` on a create (a
post-aim gate: a mid-aim drain must not push it below the reserve). `view: None` is a **deferred
aim** — an on-boulder synthoid re-aiming after the boulder landed, or an absorb whose coarse
sweep resolved no view — re-proposed against current live memory via `aim.propose` at the
player's true eye, so sim and driver never diverge on how a tile is aimed. Transfer aims go
through `LiveMixin._drive_transfer_aim`. A missed aim is a crash: a step is aim-exact, so a miss
means the model diverged. `LiveMixin` keeps think time out of the world (`_observe` snapshots
and leaves the CPU halted, `_advance` is a no-op, `_wait` spends frames via `clock.run_frames`),
and `_plan_step_stale` re-validates the next step against the live enemy phase on the window the
plan gated it with.

**Plumbing.** Host `-p` publishing is unreachable here, so every boot path connects to the
container's docker bridge IP (`boot.bridge_ip`; `BINMON_HOST`/`BINMON_PORT` override, missing IP
falls back to `127.0.0.1`). The container launches `warp=True`; `WarpMode` is not settable on
this asid-vice build (opcode `0x52` → err `0x8f`), so a failed set is non-fatal. Concurrent runs
are safe: the publish is `-p 0:6502` and `boot.kill_stale` is scoped by `boot.stale_filter()` to
`asid-vice-<own pid>-*` (`VICE_REAP_ORPHANS=1` opts into a blanket sweep). Snapshot paths are
paths *inside* the emulator process, so they must be `/renders/...`. The monitor is
frame-quantized while the CPU runs — a round-trip costs orders of magnitude more running than
halted, and slowest of all with warp off, independent of read size — so read while halted and
treat a multi-second timeout as a wait on a PC that can never recur, not back-pressure. With
warp off, any dead dwell in which the CPU runs is live time in which the Sentinel can spawn a
ring meanie. Full-image reads are done in two 32 KB halves because `mem_get`'s response length
is a u16.

## The divergence instrument (`driver/instrument.py`)

Races the model against the real game frame-for-frame and reports the **first** state
disagreement, decoded to a named field. Both worlds keep their play state in a 64 KB image at
the same addresses, so one schema (`statecmp.FIELDS`) decodes either. The emulator clock is
`advance_instructions(1)` off the raster marker `$9630` then `run_until_pc($9630)` — one
`$9630`→`$9630` span is exactly one ROM frame; the sim clock is `enemies.advance_frame`. Both
are seeded from the emulator's own image at entry, so frame 0 is byte-identical, the sim gets
the real in-RAM tables (e.g. rotation speeds at `$9D37`), and board generation is skipped
entirely.

| Tier | Fields | Meaning |
|------|--------|---------|
| `CORE` | objects, enemy cooldowns, energy, tiles, discharge/meanie arrays, `$1335`, `$0C50` | a CORE divergence is a real model/ROM disagreement |
| `SWEEP` | cursor `$0090`, PRNG `$0C7B-$0C7F` | by-design non-goals: unreadable landing coords; `$0090` only orders slots within a frame |
| `SCRATCH` | `$0014`, `$0C56-$0C58`, `$0C68`, `$0C76`, `$0CDD` | LOS/targeting bytes rewritten every scan |

```bash
python -m driver.instrument 42 --frames 1200     # --follow keeps racing past a CORE event
```

Boots under warp with no recording (`NO_RECORD=1`), unfreezes the enemy clock on both sides by
clearing `$0CE5` bit7, then frame-locks and prints the per-tier first-divergence report.
`--follow` reseeds the sim from live memory on each CORE divergence and continues.

**Status:** no CORE divergence within 1200 frames on ls42;
`driver/test_enemy_sim_divergence.py::test_enemy_sim_frame_locked_to_live_ls42` gates 600 frames
as a plain assertion. Fidelity here is binary — a sim that reproduces enemy phase 97% of the time
is 0% correct on the outcome it decides, because one rotation step of drift puts a body in a gaze
the planner modelled empty.

## Measurement and iteration tools

### The 6502 oracle (`sentinel/tests/oracle.py`)

Runs the real ROM under py65 so a model routine can be diffed against it. Two setup steps
mutate state the caller usually means to control, and either one makes a comparison read as a
model defect:

- **`machine_from_image` overlays the ROM `LOADED` regions on top of the caller's image.**
  The board is in low RAM and survives, but per-enemy state *inside* a loaded region does not:
  `ROTATION_SPEED_TABLE $9D37` and the cooldown Bresenham accumulator `$1335` are replaced by
  the image's. A caller seeding a recorded clock must rewrite both afterwards, or the ROM
  rotates by the wrong step. (`test_human_clock.py` does this; `driver/instrument.py` seeds
  from the live image, so it gets the real table.)
- **`prime_enemy_driver` resets the round-robin cursor `$0090` to 7 and the cooldown gate
  `$0C50` to 0**, on top of RTS-stubbing the render/sound routines and clearing
  `WORLD_BUSY_PLOTTING $0C1F`. That is what makes `update_enemies` steppable one round at a
  time to match `enemies.step`, but it discards any recorded phase in those two bytes.

### The recorded clock (`sentinel/tests/human_clock.py`)

A `watch_play/3` fixture event carries the whole pre-action enemy clock, which recovers exactly
how many frames the game advanced between two recorded actions with no cost model in the loop.

| quantity | what it pins |
|---|---|
| `$1335` accumulator | the frame count **mod 256** (`$CD` is invertible mod 256) |
| `$0C50` gate | lifts that to **mod 768** (205 carries per 256 frames, `205 % 3 == 1`) |
| `$0C28` sawtooth | picks the multiple — 200 rounds, reloaded at `$1813`, sticking at 1 |

A span is exact only when every voter agrees; one voter suffices, because `span_frames` must
satisfy (bres, gate) AND the decrement count jointly, so a wrong delta yields no candidate rather
than a wrong one. `span_frames` is closed-form and `test_closed_form_matches_the_stepped_clock`
checks it against the stepped loop over the whole `(accumulator, gate)` space.

| board | enemies | capture | exact spans | clock round-trip | facings |
|---|---|---|---|---|---|
| ls0 | 1 | live | 16 | 16/16 | 16/16 |
| ls42 | 2 | live | 10 | 10/10 | 10/10 |
| ls335 | 7 | live | 18 | 18/18 | 12/18 |
| ls335 | 7 | async | 117 | 117/117 | 89/117 |

The cooldown clock round-trips perfectly everywhere; the ls335 facing gap is
[open](open_items.md#8-the-enemy-clock-what-is-left-is-the-redraw-and-the-frame-budget). In aggregate the
action-cost bill lands just under the measured span between genuine player actions, which is what
a correct bill must do — the human's think time sits on the measured side. Applying the action
last in its span, 83 of 91 exact-span actions reproduce the human's next energy; the misses are
off by exactly one in both directions, i.e. drain-timing scatter inside the span. A drain does not
decrement a counter — `$1A08` **downgrades** its target — so an absorb whose object was drained
mid-span yields one less.

**`$0C30` is not a score.** The recorded `update_cooldown` sits on its stick value 1 in most ls335
`watch_play` samples (async, free-running machine) but rarely in ls42 `replay_human` samples
(halted at a checkpoint): the same register reads differently by *where in the loop the capture
stops*, so scoring on it measures the recorder
(`test_update_cooldown_is_sampling_dependent_and_not_a_score`). Facings are the sound score — a
facing only moves when a rotation actually fires.

**Fixture hygiene.** The dither loop (`$1FA4`, `DITHER_FRAMES`) and the transfer tune wait
(`$35D5`, `TUNE_TRANSFER_FRAMES`) are hard floors on how close two real actions can be; 8 exact
ls335 spans fall below the floor for the action preceding them, so those bracket pairs are one
action recorded twice. ls335 also carries 33 events of the two known recorder classes (enemy
discharge trees, drain ticks minted as self-transfers); `human_replay` skips them.

### Retrograde regression (`sentinel/tests/human_regress.py`)

Hands the player the human's PRE-action state at event `i` and asks it to finish alone: the highest
handover it cannot convert is the board it fails on, and the human's own move there is the move it
missed.

```bash
python -m sentinel.tests.human_regress ls335.json --out out/ls335_regress.json --diagram
python -m sentinel.isoview 335                       # the board at entry, no annotations
```

`state_at(fixture, i)` rebuilds the board from `landscape.generate(seed)` (byte-exact terrain,
never stored in the fixture) plus the event's objects, player/energy, `enemy_clock` (true mid-game
facings and rotation/drain/update cooldowns) and `cooldown_*` (`$1335` and `$0C50`). `$0CE5` is
cleared: mid-game, the enemies run. `$0090` is not recovered, so a handover's enemy phase is right
to within one round of updates.

Handovers are **bisected**, not walked — each round probes `workers` evenly spaced handovers
concurrently, so a whole fixture settles in a handful of rounds (`--linear` keeps the exhaustive
backward scan, `--indices` runs a list). Attempts run one **spawned** process per index (a fork
aborts: the numba LOS march leaves an OpenMP runtime in the parent) and a `SIGALRM` cap does not
work — raised inside the numba march it corrupts the dispatcher. Each worker is pinned to
`cores // batch` numba threads, because an uncapped `parallel=True` `march_batch` takes a thread
per core and oversubscription inflates every attempt. Outcomes are `won`/`lost`/`capped`; `capped`
means undecided, and the top capped index is re-run alone at `escalate` × the cap before being
called.

`--diagram` writes an isometric SVG (`sentinel/isoview.py`) of the first losing handover — the
height field as a lit mesh from its ROM corner heights (`los._slope_corner_z`), typed object
glyphs at their own `z_height`, each live enemy's scan cone as a ground wedge on its recorded
facing, and the human's next actions as solid numbered arrows against the planner's dashed.

### Checkpointed iteration (`sentinel/tests/ckpt.py`)

A planner failure is a property of a **single state**, so re-enter that state instead of replaying
the board to it. `snapshot(player, tick)` stores the 64 KB image plus the player scalars it does
not carry; `restore(snap, cls=PhasePlayer)` rebuilds. Everything else the player holds
(`_view_memo`, `_cone_memo`, `_hop_price_memo`, `_hold_memo`, the module-level `_VIEW_CACHE`) is a
pure cache keyed on a state signature. Two images are stored — the board the player was
**constructed** on and the live one — and the deep copies on both sides are load-bearing: storing
references makes every checkpoint alias the final tick, and assigning the snapshot's own list back
to the player makes replay mutate the checkpoint.

| tier | question |
|---|---|
| 0 filter tally | which gate kills each `(tile, k)` here? |
| 1 probe | does the candidate generator (`_climb_candidates`/`_mount`) yield anything here? |
| 2 resume | does the run still die from this tick on? |
| 3 board | does the whole board flip to a win? |
| 4 matrix | did any other board regress? (`human_regress`) |

Exercised on a reduced ls335 board, enemies `{(4,18), (12,10)}` plus the Sentinel. The tiers are
ordered by cost, each far cheaper than the one below it: promote a change only when the tier below
passes, so a generator change is judged against the stance that defeats it rather than by replaying
a whole board. The fidelity gate — restoring at tick *t* and replaying to the end reproduces the
run's own trace tail entry for entry — must assert a non-zero `actions_replayed`; both deep-copy
bugs above first presented as `identical: true` over an empty tail.

**Determinism contract.** Anything whose result is compared must be bounded by node budget only,
never a wall clock: a wall-clock cut makes the search a function of host load, and with the clock
out of the loop parallelism changes wall time and never a verdict.

### The landscape atlas (`sentinel/atlas.py`, `sentinel/statecache.py`)

Per-landscape metrics over every board, in two layers that are versioned independently.

**Layer 1 — the state cache (`statecache.py`).** Generating a board is the whole cost
(~17 ms); its entire result is one 64 KB image, so an entry is that image zlib-compressed
(~1.1 KB) at `out/atlas/<signature>/ls<code>.z`, loaded in ~0.3 ms. The signature is 12 hex
chars of sha256 over `CACHE_VERSION` plus the sources of `landscape.py`, `prng.py`,
`state.py`, `memmap.py`, `game.py` — the generator and nothing else. Editing the generator
selects a new directory, so stale entries are never read; editing a metric cannot change the
key. `$SENTINEL_ATLAS_CACHE` overrides the root. `out/atlas/` is gitignored.

**Layer 2 — the metrics (`atlas.py`).** A `Board` wraps a cached `State` with the arrays
metrics read: the 32×32 resolved `heights`/`slopes` fields (object tiles resolved to their
stack floor, as `terrain.resolve_ground` defines them), the object arrays, and the live and
enemy slot indices — all numpy, no per-tile Python. A metric is one function registered by
`@metric("name")` taking that `Board`. Adding one is adding that function: re-running
recomputes over the cached images with zero generation.

| metric | value |
|---|---|
| `seed` | the ROM PRNG seed the typed code maps to |
| `enemies` | occupied enemy slots (Sentinel + sentries) |
| `enemy_list` | per enemy: slot, type name, tile, height |
| `landscape_energy` | `ENERGY_IN_OBJECTS` summed over every occupied slot but the player's own robot; **excludes** the player's 10 starting energy |
| `roughness` | mean absolute height step between neighbouring tiles, both axes |
| `relief`, `mean_z` | height span and mean over the 32×32 ground field |
| `flat_tiles` | tiles of slope 0 — the only standable ones |
| `start_tile`, `start_z`, `start_eye`, `start_energy` | where the player robot begins |

**Range.** A landscape code is the four digits a player types, `0000`–`9999`;
`landscape.seed_for` reads them as hex (`f"{code:04d}"` parsed base 16), so 10000 and above
are not codes at all. `statecache.valid_code` enforces it.

```bash
python -m sentinel.atlas --start 0 --stop 500              # readable table
python -m sentinel.atlas --codes 0,42,335 --format json    # machine-readable
python -m sentinel.atlas --start 0 --stop 400 --like 335   # nearest boards (absorbed landscan)
python -m sentinel.atlas --codes 335 --regen               # ignore the cache
```

Chunked and parallel (`--jobs`, default one worker per core) and resumable, because the
cache *is* the resume point — a chunk is a plain re-run of the same command over a
sub-range. Measured on 24 cores: 2000 codes cold 2.0 s wall / 33 s CPU, warm 0.5 s wall /
4.2 s CPU. The whole range is ~178 s CPU, so build it in five 2000-code chunks to stay
inside the 60 s-per-script budget; all 10000 warm then cost 2.8 s wall / 21 s CPU and
generate nothing. Cache: ~1.06 KB per landscape, 10.6 MB for all 10000.
