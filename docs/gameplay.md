# Gameplay model

A reference for both **playing** *The Sentinel* (C64, Geoff Crammond / Firebird
1986) and **writing an automated solver** for it. Every mechanic here is taken
from the original C64 6502 code; routine names and `$hex` addresses cite the exact
ROM source. No emulator or existing simulator was used to derive behaviour — the
ROM is the ground truth. The last section, [Writing a solver](#writing-a-solver),
distils the mechanics into what a search needs; the implemented players
([player.md](player.md), [astar_player.md](astar_player.md)) apply it.

The board and object placement are **deterministic given the landscape number**,
and the effect of each player action is deterministic. The one thing a player
cannot foresee is where a **hyperspace or meanie sends them**: those destinations
are drawn from the game's running PRNG, whose state a player cannot observe — so
they are unpredictable in play, and a faithful solver must treat them the same way
(see [PRNG](#prng-and-determinism)).

---

## 1. Objective

Each **landscape** (numbered 0000–9999) is a 32×32 tile board. One **Sentinel**
stands on a **platform** on the highest tile, slowly rotating and scanning;
higher landscapes add **sentries** (up to 7 more) that behave identically. When one
of these catches you half-exposed it can grow a **meanie** — a nearby tree animated
into a mobile hunter that turns to face you and teleports you off your position
([§6](#meanies--the-mobile-hunter-consider_creating_meanie-1986)). You start as a
low-energy robot far below. You win a landscape by:

1. **absorbing the Sentinel**, then
2. getting one of your own robots **onto the platform tile** it stood on, then
3. **hyperspacing while standing on that platform tile**.

That final hyperspace-from-the-platform is what sets the landscape-complete flag
(`$0CDE` bit 6 in `player_survived_hyperspace $217F`); simply *standing* on the
platform is **not** a win. Completing a landscape reveals a code that warps you
several landscapes forward.

The core tension: you can only see and act along an unobstructed line of sight,
your eye starts near ground level, and the only way to raise it is to build and
climb a stack of boulders — all while enemies rotate to catch your robots and
your stacks in their gaze and drain/dismantle them.

---

## 2. World model

### Board & tiles

- **32×32 tiles** (`N = 32`). Generation and most loops run tile indices `0..31`;
  object placement only ever uses tiles `0..30` (`get_random_tile_coordinate
  $1272` rejects 31), because a tile's slope facet needs its `+1` neighbours.
- `tiles_table` at **`$0400–$07FF`**, one byte per tile, stored in an
  **interleaved** (not row-major) layout. `calculate_tile_address $2BA8`
  computes the byte address of tile `(x,y)` as
  `$0400 + 256·(x&3) + 8·(x>>2) + y`.
- A tile byte encodes:
  - **`< $C0`** → terrain: **high nibble = height (0..11)**, **low nibble = slope
    code (0..15, 0 = flat)**.
  - **`>= $C0`** → the tile **contains an object**; low 6 bits = slot index of the
    **topmost** object in that tile (`put_object_in_tile $1F16`,
    `calculate_tile_address_z_and_slope $1DF9`).

The height field is a **vertex/corner field**: a tile's rendered/collision surface
is derived from the heights of its **four corner tiles** `(x,y) (x+1,y) (x+1,y+1)
(x,y+1)`, split into two triangles by the slope nibble (see
[Line of sight](#5-line-of-sight)).

### Object arrays

64 object slots (0..63), stored as parallel arrays indexed by slot:

| Array | Base | Meaning |
|-------|------|---------|
| `objects_x` | `$0900` | tile x |
| `objects_y` | `$0980` | tile y |
| `objects_z_height` | `$0940` | height (integer units) |
| `objects_z_fraction` | `$0A00` | height sub-fraction (/256 of a unit) |
| `objects_flags` | `$0100` | **bit7 = empty slot**; `< $40` = on ground; `$40–$7F` = stacked on object `flags & $3F` |
| `objects_h_angle` | `$09C0` | facing (256 units = 360°) |
| `objects_v_angle` | `$0140` | pitch |
| `objects_type` | `$0A40` | type code (below) |

`player_object` (zero-page `$0B`) is the slot the player currently inhabits.

**Object types** (`objects_type`): `0` robot (what the player inhabits, and
other drainable robots), `1` sentry, `2` tree, `3` boulder, `4` meanie, `5`
Sentinel, `6` platform. Player, sentries and the Sentinel are all "robot-shaped";
only type `0` is a *transferable* robot.

### Energy economy

One player energy meter, **`player_energy $0C0A`**, range **0..63** — every write
goes through `set_player_energy $2148` which masks `AND #$3F`. It starts at
**10** (`initialise_player_and_trees $1450`).

Each type has a fixed energy value, `energy_in_objects $214F` (bytes
`03 03 01 02 01 04 00`):

| Type | robot(0) | sentry(1) | tree(2) | boulder(3) | meanie(4) | Sentinel(5) | platform(6) |
|------|:-:|:-:|:-:|:-:|:-:|:-:|:-:|
| Energy | 3 | 3 | 1 | 2 | 1 | 4 | 0 |

`gain_or_lose_energy_from_object $2136`: **absorb adds** the value, **create
subtracts** it. Subtraction that would underflow returns carry set = "not enough
energy" and the action fails (`$2143`). Because the `AND #$3F` mask is applied
*after* the add, **absorb gains above 63 wrap mod 64** (62 + Sentinel 4 → 2) —
faithful ROM behaviour (`$214A`), a real trap for a greedy solver.

**Energy is conserved across the whole board**: total energy = player energy +
Σ (energy value of every object present) is invariant under absorb/create
(hyperspace and enemy drains move energy around but the enemy re-emits drained
energy as trees, `consider_discharging_enemy_energy $1A5D`).

---

## 3. Landscape generation

Fully deterministic from the landscape number. Reproduce the PRNG draw counts
exactly or the board diverges.

### PRNG and determinism

`prnd $31CA` — a 40-bit LFSR over `prnd_state $0C7B–$0C7F`, **8 shuffles per
call**, feedback bit = `bit3(state[2]) XOR bit0(state[4])`, returns `state[4]`
(`$0C7F`). `seed_prnd_from_landscape_number $33ED` seeds `state[0..1]` with the
landscape number (as two BCD bytes: low pair = tens|units, high pair =
thousands|hundreds) and leaves `state[2..4] = 0` (fresh RAM) — that zero-fill is
what makes landscapes reproducible across machines.

The same PRNG runs during play (hyperspace/meanie placement, enemy discharge). Its
state is used **only** to reproduce a landscape's fixed generation. A player cannot
observe it, so the runtime outcomes it drives are unpredictable in play — and a
faithful solver must not read or predict from PRNG state either (it may reproduce
generation, but must treat hyperspace/meanie landings as unknown, see
[Writing a solver](#writing-a-solver)).

### Terrain pipeline — `generate_landscape $2ACC`

In strict order (draw counts matter):

1. **81 throwaway draws** — `randomise_row_or_column_tile_z_table $2ACE` fills a
   scratch table (`$AD00`, later reused for the secret-code obfuscation); it does
   not produce terrain but advances the LFSR by 81.
2. **Vertical scale** — `set_landscape_vertical_scale $2AE6`: landscape 0000 →
   fixed `24`; else `get_random_number_between_0_and_22 $3451` (1 draw) `+ 14` →
   **`landscape_vertical_scale $0C08` ∈ [14..36]**.
3. **Raw heights** — `process_landscape $2B22` mode `$80`: every one of the 1024
   tiles gets a raw `prnd` byte (**1024 draws**, y-outer/x-inner).
4. **Average smooth** — `smooth_landscape $2B83` mode `$00`: **2 passes**, each
   pass smooths every row then every column with a **toroidal width-4 box filter**
   (`average_tile_heights $2C2C`: `t[x] = (t[x]+t[x+1]+t[x+2]+t[x+3])>>2`).
5. **Scale to 1..11** — `process_landscape` mode `$01`: for each smoothed byte
   `raw`, `h = clamp( sign(raw−128)·((|raw−128|·scale)>>8) + 7 , 1, 11 )`. Flat
   noise (`raw=128`) → 7. Larger `scale` → steeper terrain (more 1s and 11s).
6. **Spike-level smooth** — `smooth_landscape` mode `$40`: same 2-pass row/column
   structure but `level_spikes $2BDF` pulls any single-tile spike/pit to its
   nearer neighbour, preserving plateaus and monotone slopes.
7. **Slopes** — `set_tile_slopes $2AFD` computes a slope nibble for every interior
   tile (x,y ≤ 30) via `calculate_tile_slope $2C7C` and ORs it into the high
   nibble (heights are still in the low nibble at this stage).
8. **Nibble swap** — `process_landscape` mode `$02`: final format `(height<<4) |
   slope`.

Total draws before object placement: **81 + (0|1) + 1024 = 1105 or 1106**.

### Slope nibble — `calculate_tile_slope $2C7C`

Reads the four corner heights `a=(x,y) b=(x+1,y) c=(x+1,y+1) d=(x,y+1)` and
returns a 0..15 code by an equality/ordering decision tree (`$2CA8–$2D11`):

- **0** — flat (all four equal).
- **Single odd corner** ("corner" tiles, split into two triangles about that
  corner): `2`/`b` = origin `a` odd, `7`/`e` = `b` odd, `3`/`a` = `c` odd, `6`/`f`
  = `d` odd. The even/odd member of each pair distinguishes the odd corner being
  **higher vs lower**.
- **Edge/ridge tiles** (two parallel edges at different heights): `4` & `c` = one
  edge flat and the adjacent one sloped; `1`/`9` = the two x-edges differ; `5`/`d`
  = the two y-edges differ.

To reconstruct the facet offline, **port the `$2CA8–$2D11` compare-tree
verbatim** — the diagonal-choice low bits depend on exact `BCS/BCC` boundaries.

### Enemies — `set_palette_and_initialise_enemies $1420`

**Count.** Landscape 0000 → exactly **1** (the Sentinel). Otherwise
`min( get_maximum_number_of_enemies $3426, maximum_number_of_enemies $0C07 )`,
stored in `$0C6F`, where:
- `$0C07` (the cap) = `8` for landscapes ≥ 0100, else `1 + tens_digit` (≤ 8).
- `$3426` (the random value) is a geometric distribution centred on `thousands
  digit + 2`, clamped to 1..8 (each step half as likely; port `$3432–$3450` for
  exact draw counts — it consumes a variable number of `prnd` calls).

So low landscapes are cap-limited: 0000–0009 → 1 enemy, 0010–0019 → ≤ 2, etc.

**Placement** — `initialise_enemies $14FB` using an **8×8 grid of 4×4-tile
sections** (`find_highest_tiles_in_grid $15CC`; each section records its highest
**flat** tile). Enemy 0 is always the **Sentinel** (`is_sentinel $1553`, type 5),
placed on the globally highest flat tile; a separate **platform object (type 6)**
is created under it and the Sentinel stacks one unit up on it
(`create_object` A=6 at `$1560`; the platform tile is saved to `$0C19/$0C1A` — the
win target). Sentries (type 1) take the next-highest sections. Each chosen section
plus its 3×3 neighbourhood is marked used, so **no two enemies are placed in
adjacent sections** (`$1528–$1542`). The height nibble of the lowest enemy is
saved to `$0C06` as the ceiling for player/tree placement.

Each placed object gets a **random initial facing** `(prnd & $F8) + $60`
(`put_object_in_tile $1F83`). Each enemy also gets a random **update cooldown**
`(prnd & $3F)|5` and a fixed **rotation step** of `+20` (clockwise) or `−20`
(anticlockwise) units, direction from a random bit (`set_enemies_rotation_speed
$1586`).

### Player & trees — `initialise_player_and_trees $1450`

- Player object type 0, **energy 10**.
- **Start tile**: landscape 0000 → fixed **(8, 17)**; else a random **flat, empty**
  tile of height `< min($0C06, 6)` (`put_object_in_random_tile_below_z $1238`) —
  i.e. below the enemies and no higher than 6.
- Player eye sits at that tile's height (`z_fraction = $E0`); random initial
  facing as above.
- **Trees** (type 2): count = `min( get_random_0_22 + 10 , 48 − 3·enemies )`, each
  placed on a random flat tile below the enemy ceiling `$0C06`; the loop stops
  early if no tile qualifies. More enemies ⇒ fewer trees (less loose energy).

---

## 4. Actions

There are no turns. See [Timing](#timing--the-world-clock): the world clock keeps
running while you aim and act. All actions except hyperspace and u-turn require a
**line of sight** to a target tile.

### Aiming — the sights

With the sights on, direction keys move a 1-pixel cursor that pans the view only
when it **wraps at a band edge**; horizontal angle moves on an **8-unit lattice**,
vertical pitch on a **4-unit lattice**, pitch hard-clamped to the band
`[$CD..$FF] ∪ [$00..$35]` (`pan_viewpoint $10B7`, limits table at `$1149`). The
target tile is derived from cursor + view angles by
`prepare_vector_from_player_sights $1C10`, then the ray is marched by
`check_for_line_of_sight_to_tile $1CDD`, which **stamps the reached tile into
`$003A`(x)/`$003C`(y)** — that stamped tile is what every action fires on. (The
full key→angle→vector chain — cursor bands, pan lattice, aim-vector build — is a
subsystem of its own; the tile-targeting summary here is what actions depend on.)

Three control facts that matter for aim time (all verified in the ROM):

- **You can steer the cursor diagonally.** `move_sights $9958` applies the
  horizontal move (`$9965`, from want-flag `$0CE8`) and the vertical move (`$9994`,
  from `$0CEA`) in the *same* frame (vertical is skipped only on a frame where a
  horizontal edge-pan fires). Holding a left/right key **and** an up/down key
  together walks the cursor diagonally, reaching an off-axis target in far fewer
  frames than squaring the corner one axis at a time. Always aim diagonally when
  the target is off both axes.
- **The u-turn is an aiming tool, not an action to spend.** It flips your facing
  180° instantly (`handle_uturn $1B2F`: `objects_h_angle ⊕ $80`) — **free**, no LOS,
  no energy, with an auto-repeat guard. When your target is behind you, a u-turn is
  a one-shot substitute for panning the sights halfway around the horizon: flip,
  then fine-aim the short remaining angle. It costs (almost) no aim time, so it is
  the cheapest way to service a target to your rear (mechanically it is dispatched
  as action code `$23`, but you use it like a control).
- **Toggling the sights off re-centres them.** Turning the sights on runs
  `initialise_sights $134C`, which hard-resets the cursor to `$0CC6 = $50` (80) /
  `$0CC7 = $5F` (95) — the band centres. So switching the sights off and back on
  **throws away wherever the cursor was** and starts you centred again. Avoid
  toggling off unless that re-centre genuinely makes the next aim net faster (e.g.
  the cursor is pinned far from where you next want it).

The pending action is a code in `$0C61`/`$0CE9`, dispatched by
`handle_player_actions $1B18`:

| Action | Code | Routine |
|--------|:----:|---------|
| create robot | `$00` | `try_to_create_object $1BBA` |
| create tree | `$02` | `try_to_create_object` |
| create boulder | `$03` | `try_to_create_object` |
| absorb | `$20` | `try_to_absorb_object $1B8E` |
| transfer | `$21` | `try_to_transfer_into_object $1B64` |
| hyperspace | `$22` | `handle_hyperspace $1B1F` |
| u-turn | `$23` | `handle_uturn $1B2F` |

For every code `< $22`, `$1B18` first runs the aim vector + `$1CDD` and, on carry
set (no LOS), plays the bad-action sound and does nothing (`$1B46`). Create/
absorb/transfer also require the sights to be active (`consider_player_action
$12D0`); hyperspace/u-turn do not.

### Create — `try_to_create_object $1BBA`

Creatable types: **robot(0), tree(2), boulder(3)** only (the type written *is* the
action code). Steps:

1. **Find an empty slot** (`find_empty_slot_loop $2122`, slots 63→0); all 64 used
   → fail.
2. **Deduct energy** for the type (fails, no placement, if underflow).
3. **Place** on the LOS-target tile (`put_object_in_tile $1F16`); on failure the
   energy is **refunded**.
4. A created robot is turned to face the player (`objects_h_angle = player ⊕ $80`).

**Stacking rule** (`put_object_in_tile`): you may create on an empty tile
(placed on the ground) **or on a tile whose topmost object is a boulder(3) or a
platform(6)** — nothing else is stackable. On a boulder the new object sits `+½`
unit up; on a platform `+1` unit. This is the **only** way to gain height:
build a boulder, transfer up onto it, build the next boulder from there, repeat.

There is **no explicit "height slack" constant** for building upward — the
apparent limit is purely the LOS geometry. `$1CDD` sets a vertical tolerance
`$000C = $80` (½ unit) and rejects a flat tile the ray sits more than that above
(`$1D1E`). So treat create-reachability as simply "any tile with clear LOS per
`$1CDD`", not a fixed unit budget.

### Absorb — `try_to_absorb_object $1B8E`

Requires LOS + an object present in the target tile (both enforced by `$1B18`),
**and the Sentinel must still exist** (below). Absorbs the **topmost** object,
gaining its energy. Meanies route to `try_to_absorb_meanie $1BEC`, which also
clears the owning enemy's meanie link. The **platform (type 6) cannot be absorbed**
(`$1B9A`: `CMP #$6 / BEQ play_bad_action_sound`); every other present type is —
robot/tree/boulder, and, while the Sentinel lives, sentries(+3) and the
Sentinel(+4) itself.

**Once you absorb the Sentinel you can no longer absorb *anything* — only create
and transfer.** This is a hard rule, and it is the first thing
`try_to_absorb_object` checks:

```
1b8e  ad 00 01  LDA $0100                 ; objects_flags[0] — slot 0 is ALWAYS the Sentinel
1b91  30 16     BMI play_bad_action_sound ; Sentinel slot empty (already absorbed) -> reject
```

The load is **absolute `$0100`** (`ad 00 01`), i.e. `objects_flags[0]`, not the
target's slot — slot 0 is always the Sentinel (`is_sentinel $1553`, "Object &00 is
the sentinel"). When you absorb the Sentinel its slot becomes `SLOT_EMPTY`
(bit 7), so **every** later absorb attempt — of any object, including meanies —
bails here before it even looks at the target tile. (The ROM annotation "Can't
absorb if no object in slot" misreads this as a target check; the operand proves
it is the Sentinel's slot.) Create (`try_to_create_object`, branched at `$1B50`)
and transfer (`try_to_transfer_into_object $1B64`) use different checks and are
unaffected. The platform is thus effectively part of the landscape once the
Sentinel is gone: you can build a robot on it and transfer up, but you cannot
absorb it or anything else.

**Consequence: absorb the Sentinel dead last.** This is only an *ordering*
constraint, not a "harvest everything first" phase. Absorbing trees/sentries/
scenery stays opportunistic and time-gated (do it when a tile buys you the spare
time and the target is cheap to aim at — [§7](#7-how-a-human-wins-quick-strategy));
just take anything you actually want *before* the Sentinel, because the moment it
falls your only remaining tools are create, transfer and hyperspace.

There is **no "facing away" requirement** in the player-absorb code — you only
need LOS to the tile and an object there (and the Sentinel alive). (The folklore
"only when it can't see you" concerns the *enemy's* ability to drain *you*, not
your ability to absorb.) Absorbing an object that had things stacked on it exposes
the object below.

### Transfer — `try_to_transfer_into_object $1B64`

The target's topmost object must be a **robot (type 0)**; the viewpoint moves into
it (`player_object = slot`) and your eye rises to that robot's height. Any visible
robot works (there is no ownership check — but only type-0 robots exist to
transfer into, all of which are yours). If a boulder is stacked on your robot the
topmost object is the boulder, so you cannot transfer there. Transfer records
whether the new body stands on the platform (`find_platform_below_player_loop
$1B6E` → sets `$0CE6`).

### Hyperspace — `handle_hyperspace $1B1F` / `do_hyperspace $2156`

Panic teleport. Creates a **new robot**, places it on a random flat empty tile of
height `≤ player_z + 1` (`put_object_in_random_tile_below_z $1238`, so never
higher, usually lower), and **costs 3 energy** (the robot value). Hyperspacing
with `< 3` energy **kills you** (`$216E`). The old body is left behind as a robot
in the world. The destination is **PRNG-driven and unforeseeable** — the player
cannot observe the PRNG, so hyperspace is a gamble, not a steerable move (a
faithful solver must treat the landing as unknown). Hyperspacing while standing on
the platform tile (`$0C19/$0C1A`) is the **win** (`player_survived_hyperspace
$217F` sets `$0CDE` bit 6) — and that is a fixed check on your *current* tile, not
a PRNG outcome.

(**U-turn** is dispatched here too, as code `$23` → `handle_uturn $1B2F`, but it is
an aiming control — a free instant 180° facing flip — and is documented with the
[sights](#aiming--the-sights).)

---

## 5. Line of sight

Both **player actions** and **enemy scanning** use the same ray-march,
`check_for_line_of_sight_to_tile $1CDD` — they differ only in the observer and how
the aim vector is built. Result: **carry set = blocked / no LOS**; carry clear =
clear LOS to the reached tile.

### The ray march — `$1CDD` / loop `$1CE8`

- **Start** at the centre of the observer's tile, at the observer's stored
  top-of-stack height (`get_object_details $1ECC` seeds a 3-byte fixed-point
  position per axis: sub-fraction / fraction / tile; fractions start at `$80` =
  tile centre).
- **Step** each iteration by the aim unit vector scaled to **≈ 1/16 tile per
  step** (`add_vector_to_object_position $1CBB`). The vector's vertical component
  sign (`$30`) records whether the ray points up/level.
- Each step: advance; if tile x or y leaves `0..30` → **no LOS**. Look up the
  tile's surface height (terrain facet or object top) and compare to the ray
  height. There is **no fixed step count** — the march continues until it leaves
  the grid, rises clear of terrain, or hits a surface.
- The observer's own tile is skipped.

### Terrain surface under the ray

- **Flat tile** (`check_flat_tile $1D0D`): surface = height nibble. Ray below it →
  keep marching; ray ≥ 1 unit above → **blocked**; within the `$000C` (`$80`)
  tolerance → hit this tile.
  - **Look-up rule** (`$1D2C–$1D30`): if the ray points up or level (`$30 ≥ 0`)
    when it reaches a flat tile at its own height → **blocked**. You cannot see or
    act on ground at or above your eye by looking up at it.
  - **Exception**: waived when aiming at the **top of an object** (tree/boulder/
    robot — flags `$0C6E`/`$0C67`), so you *can* aim upward at object tops.
- **Sloping tile** (`check_sloping_tile $1D46`): reads the four corner heights,
  then:
  - Quadrilateral slopes 4/`c`: blocked only if the ray is below **all four**
    corners.
  - Other slopes: pick which **triangle** by comparing the ray's in-tile x-fraction
    vs y-fraction (the diagonal split), then **linearly interpolate the surface
    height along that triangle's sloped edge** (`use_corner_for_slope $1D9D`,
    `use_edge_for_slope $1DAF`, edge table `$1DF1`). Ray above the interpolated
    facet → continue; below → blocked/hit.

This is a **true triangular-facet surface**, not a bilinear average of the four
corners — a solver's LOS must interpolate along the chosen triangle edge or it
will over-estimate visibility.

### Object surfaces

At an object tile the blocking/target height is the object's true stacked surface
(`get_tile_z_for_line_of_sight $1E0E`): tree/boulder tops via
`get_minimum_x_or_y_fraction_from_tile_centre $1EAF` (only "targeted" if the ray
threads near the tile centre), robot/platform tops require the ray very near
centre and tighten the vertical tolerance to `$10`, and stacked objects resolve
down to the lowest object's height (`get_height_of_lowest_object $1EA4`).

### Player vs enemy LOS

Enemies do **not** use a different raytracer.
`check_if_enemy_has_line_of_sight_to_object $18E6` (from the `$1887` scan path)
sets the observer to the enemy, aims at the target's height, and calls the same
`$1CDD` — but **twice**, once at the target's top and once `$E0` lower, yielding
**full vs partial** visibility (`$0014`: 0 = unseen, `$40` = partial, top bit =
full). Enemy LOS is additionally gated by a horizontal field-of-view test (below).

> A separate, render-time visibility precompute (`populate_tile_visibility_bit_table
> $245B`, `trace_rays_from_observer_to_row_of_tiles $24E2`) drives drawing/culling
> and the on-screen exposure bar; it is **not** the authoritative action test. For
> "can I act on / can an enemy drain this tile", `$1CDD` is authoritative.

---

## 6. Enemies

You face **three** kinds of threat: the **Sentinel** (type 5), its **sentries**
(type 1), and the **meanies** (type 4) those two spawn from nearby trees. The
Sentinel and sentries are the stationary rotating scanners; a meanie is a mobile
hunter conjured on demand. All meanie behaviour is covered in
[Meanies](#meanies--the-mobile-hunter-consider_creating_meanie-1986) below — but
treat it as a real enemy, not a footnote.

Sentries (type 1) and the Sentinel (type 5) run **identical AI** — `update_enemies
$16B5` treats them the same (`is_sentinel_or_sentry $16C6`). The Sentinel's only
edges are **positional**: it stands highest, on a raised platform (widest LOS,
hardest to hide from), is always present, and its platform tile is the win target.

### Timing / the world clock

**Not frame-locked.** The game logic runs in the main loop
`update_game_and_continue $363D`; there is no vsync wait in the loop path, so the
loop rate is compute-bound (dominated by the 3-D redraw). **No wall-clock "ticks
per second" is derivable from the ROM** — only the *relative* cadence set by the
cooldown constants is fixed.

- **Enemies are frozen until the player's first action** (`$3682`: skip while
  `$0CE5` bit7 set).
- `update_enemies $16B5` services **one of the 8 enemy slots per call**
  (round-robin via `$0090`), and `update_game $127C` calls it repeatedly until an
  enemy causes a visible replot — so roughly **one visible enemy micro-action per
  redraw cycle**.
- Cooldowns live in three 8-byte arrays: **`enemies_draining_cooldown $0C20`**,
  **`enemies_rotation_cooldown $0C28`**, **`enemies_update_cooldown $0C30`**. They
  are decremented through a two-stage divider (`update_enemy_cooldowns $130C`: a
  fractional accumulator adds `$CD/256 ≈ 0.80` per call, then a `2→0` counter
  `$0C50`), so each cooldown unit ≈ **3.7 main-loop iterations**; a byte only
  decrements while ≥ 2 (sticks at 1 until reset).

Reset constants (in cooldown units / "rounds"):

| Counter | Reset | When |
|---------|:-----:|------|
| update | 4 | every service (enemy "thinks" when update ≤ 1) |
| update | 10 | after a meanie rotates toward you |
| update | 30 | after draining/reducing an object |
| update | 50 | after creating a meanie |
| update | 5–63 rand | at spawn |
| rotation | 200 (`$C8`) | each time it rotates |
| draining | 120 (`$78`) | when it first targets an object (also a "targeting" flag; 0 = not targeting) |

Treat these as relative cadences: a rotation step costs ~200 units, a drain
countdown ~120 units of continuous visibility before it fires.

### Scanning — a narrow forward cone

`check_if_enemy_can_see_object $1887`: the target must be within the enemy's
**horizontal field of view** of its *current facing* — `±($0C68/2)` units, where
`$0C68 = $14` (±10 units ≈ ±14°, one screen width) for normal drain targeting and
`$28` (±20 units) for meanie search — **and** pass the vertical `$1CDD` LOS. So an
enemy must physically **rotate to face** a target to see it.

### Rotation — `rotate_enemy $1805`

Each rotation event turns the enemy by its **fixed ±20-unit step** (set at spawn),
in one constant direction, and resets rotation cooldown to 200. It does **not**
track smoothly — it steps one screen-width at a time. A tile is only truly "safe"
from an enemy if the enemy's LOS to it is blocked from **every facing it will
rotate through**, not just its current one.

### Drain vs downgrade — `consider_enemy_state $16E6`

When an enemy's update cooldown reaches ≤ 1, per service, in order:

1. **Discharge** stored energy (`consider_discharging_enemy_energy $1A5D`): if it
   holds absorbed energy (`enemies_energy_to_discharge $0C88`), it spits out a new
   **tree** in a random low tile.
2. **Dismantle stacks** (`find_drainable_boulder_or_tree_on_stack $1AB0`): scans
   for a visible boulder, or **any object standing on a stack** (`flags ≥ $40`),
   and drains it — this is how it takes apart the towers you build.
3. **Target a robot** (`find_drainable_robot_loop $17B2`): scans all 64 slots for
   a visible **robot (type 0)**. A robot seen through a tree is skipped. A fully
   visible robot is targeted immediately; a *partially* visible one is remembered
   **only if it is the player** (→ the meanie path).
4. **Reduce** (`consider_reducing_object $183D`) once the 120-unit draining
   countdown reaches 1 **and** the target is still fully visible:
   `reduce_object_energy $1A08`.

`reduce_object_energy` effects (`$0C58` = target):
- **Player**: energy 0 already → `kill_player $1A00`; else **−1 energy** and the
  unit is banked into the enemy's discharge store.
- **robot → boulder → tree → (removed)**: one downgrade step per drain event.

Every player drain / object downgrade banks **+1** into the enemy's discharge
store, later re-emitted as trees (step 1) — so the enemy is energy-neutral and the
board's total energy is conserved.

### The exposure bar

`calculate_player_exposure $191F` scans all enemies targeting the player and drives
the on-screen bar (`set_bar_state $194D`, `$0C4F` = `$80` fully seen / `$40`
partial). A flickering/filling bar is your cue that an enemy has LOS and the
~120-unit drain countdown is running — move or break LOS before it completes.

### Meanies — the mobile hunter (`consider_creating_meanie $1986`)

A meanie is a **tree the Sentinel or a sentry animates into a hunter** to deal with
a player it can half-see but can't drain. It is a full enemy, but it is not placed
at generation — it is spawned at runtime, one per enemy, and driven *by* that enemy.

**Trigger**: an enemy has targeted the **player** but the player is only
*partially* visible, so at the drain countdown it cannot drain — it branches to
make a meanie instead (`consider_reducing_object $183D` → `$1852`). (A *fully*
visible player is just drained; a meanie is specifically the answer to a
half-exposed one.)

**Creation** (`attempt_to_create_meanie $19A1`): with FOV widened to `$28` (two
screen widths), the enemy scans objects for a **tree within 10 tiles in x *and* 10
in y of the player-body** that it can **fully see**, and flips that tree's type to
**4 (meanie)** in place (`$19F0`) — no slot is allocated, the tree *becomes* the
meanie. It records the meanie in `enemies_meanie_object,X` (so **each enemy holds at
most one meanie**). If the chosen tree is on a screen currently being plotted it
waits and retries next tick (`pause_meanie_creation $19FA`); if **two full scans**
(`enemies_meanie_attempt_scans ≥ 2`) find no eligible tree it gives up and stores
the player's slot in `enemies_failed_meanie_memory` so it won't keep retrying while
you stay put.

**While a meanie exists, its parent enemy drives it instead of doing its own
scanning/draining** (`update_enemies` takes the `update_meanie` branch at `$16FF`
whenever `enemies_meanie_object,X` is set). Each tick the meanie:
- **Rotates toward the player in ±8-unit steps** (`meanie_not_looking_at_player
  $1728`; direction from the player's relative bearing), updating every ~10 cooldown
  units.
- When it comes to **face the player** (relative bearing `< $14`, one screen width)
  **and has LOS** (`$0014 ≠ 0`), it **forces a hyperspace on the player**
  (`do_hyperspace $2156`, `$171D`) — flinging you to a random low, more-exposed tile
  and **charging you the 3-energy hyperspace cost** (so it **kills you if you can't
  afford the jump**; the meanie is then credited on the death screen, `$0C1C = 4`).

A meanie **never drains energy directly** — its whole purpose is to line up and
teleport you off your hard-won high ground.

**It is bound to the body you occupied when it was born** (`enemies_targeted_object`).
It dissolves back into a **tree (type 2)** (`remove_meanie $175D`) when:
- it hyperspaces you, or
- you **transfer out of** that targeted body (`$1717`, bearing check sees you left),
  or
- that body becomes empty because you **absorbed it** (`$1707`).

You can also **absorb the meanie itself** (type 4 → `try_to_absorb_meanie $1BEC`,
`+1` energy), which clears its parent enemy's link and reverts the enemy to normal —
but, like all absorbs, only while the **Sentinel still lives** (the `$1B8E` lock,
[§4](#absorb--try_to_absorb_object-1b8e)).

### Death

`kill_player $1A00` (drained at 0 energy) sets `$0C4E = $80`; the loop shows the
death screen. Forced hyperspace with insufficient energy kills via the
`$0CDE`/`$0C4C` path in `do_hyperspace`. In short: **each drain removes exactly 1
energy; a drain landing at 0 energy is death.**

---

## 7. How a human wins (quick strategy)

1. **Absorb trees opportunistically, not as a phase.** Each tree is +1 energy that
   widens your future boulder/robot options, so when your current tile buys you
   spare time and one is cheap to aim at, grab it. Under time pressure, don't —
   energy is a means, and a drain mid-aim costs more than a tree is worth. Watch the
   exposure bar.
2. **Build a boulder** on a visible tile, **transfer** onto it, build another from
   the higher vantage, and **climb** until you have a sight line over the terrain.
   Prefer tiles the enemies' rotating gaze cannot reach.
3. Work toward a line of sight to the **Sentinel's platform**, absorbing the
   Sentinel's **sentries** and any supporting scenery *when it pays* — to open a
   sight line, clear a threat, or bank energy you have time to take.
4. **The one hard ordering rule:** absorbing the Sentinel **permanently disables
   all further absorption** (`$1B8E` checks the Sentinel's slot 0), leaving you only
   create/transfer/hyperspace. So the Sentinel must be your **final** absorb — take
   whatever sentries/scenery/energy you still want *before* it, as time allows, but
   this is opportunistic harvesting, not a mandatory "absorb everything first" step.
5. When you can see the Sentinel — and have grabbed anything else you wanted —
   **absorb it** (+4).
6. **Create a robot on the platform tile** (still stackable, since the platform
   object remains), **transfer** into it, and **hyperspace** to complete the
   landscape. Note the completion code to skip ahead.

Constant pressure management: never sit fully visible to a rotating enemy long
enough for its ~120-unit drain countdown to finish.

**Deny meanies, and kill the ones that appear.** A meanie can only be born from a
**tree within 10 tiles of your body while an enemy half-sees you**
([§6](#meanies--the-mobile-hunter-consider_creating_meanie-1986)), so the first
defence is positional: don't loiter *partially* exposed near standing trees —
absorb such trees pre-emptively when you have the time, or don't stand there. If a
meanie does spawn, it will slowly turn to face you and then hyperspace you off your
hard-won height (or kill you if you can't pay the 3-energy jump), so deal with it
promptly — you have three outs: **absorb the meanie** (while the Sentinel still
lives), **absorb the body it's bound to**, or **transfer out of that body**; any of
the three dissolves it back into a harmless tree. Don't ignore it and keep climbing.

**Never enter the Sentinel's (or a sentry's) gaze unless there is literally no
other option.** Being in the gaze is a *guaranteed net energy loss*, twice over:

- **You lose energy before you can get out.** Leaving takes time — you must aim and
  then transfer, many world-clock ticks ([§6](#timing--the-world-clock)) — and the
  drain lands in that window, so you give up at least one unit no matter how fast
  you react.
- **Your abandoned body keeps bleeding after you leave.** The robot you transferred
  *out of* stays on its tile in the gaze; the enemy keeps working it, downgrading
  robot → boulder → tree and banking that energy. To get it back you must re-enter
  the gaze and fight for it, losing at least another unit in the process.

So gaze tiles are effectively off-limits, not merely "risky": treat exposure to an
enemy's current or reachable facing as a wall, and only cross it when every
alternative is worse.

**If there are sentries, hunt them down early and aggressively.** Every sentry is
another independent rotating gaze ([§6](#6-enemies)), and their swept cones
compound: each one shrinks the set of safe standing tiles, narrows your time
windows, and forces more moves. Removing a sentry is worth far more than its +3
energy — it permanently deletes a whole hazard and *loosens the time budget for the
entire rest of the solve*, which is your scarcest resource. So a sentry is a
high-priority absorb target, not incidental scenery: work to open a safe sight line
to each one and take it as soon as you can (from outside its gaze — you need LOS to
its tile, not to be in its cone). Clear the board down toward just the Sentinel
before you commit to the final climb; fewer gazes early pays off compoundingly
later. (They must precede the Sentinel anyway — the absorb-lock,
[§4](#absorb--try_to_absorb_object-1b8e).)

**Time is the scarce resource, and aiming spends it.** The world clock never stops
([§6](#timing--the-world-clock)): every pan step you take to swing the sights onto
a target is real time in which enemies keep rotating and draining. So an action's
true cost is *aim time + the act itself*, and you plan the whole route to minimise
it. Two practical rules:

- **Pick destinations that are both strong and cheap to aim at.** A transfer target
  isn't just "how high/safe is it" — it's also "how many pan steps from where I'm
  already looking." A slightly lower boulder you can fire on with a small swing
  often beats a marginally taller one that costs a long sweep of the sights (and
  the exposure that sweep buys the enemies).
- **Spend aim time in proportion to the time your current tile affords you.** On a
  tile safe from every enemy rotation you can afford a big, deliberate aim toward
  the best next position. Under time pressure — an enemy's gaze sweeping toward
  you, the exposure bar filling — don't linger to line up the perfect move; make a
  cheap, quick hop to somewhere *safer but still leverageable* and re-plan from
  there. A good-enough move made in time beats a perfect move that gets you drained
  mid-aim.

**Ping-pong across the gap, and go long.** A robot you create is always spawned
*facing back at you* (`objects_h_angle = your angle ⊕ $80`;
[§4 Create](#create--try_to_create_object-1bba), `$1BE0` "make new robot face
player"). So the instant you transfer into a new robot — however far and however
much higher — your view is already pointed back where you came from, so aiming to
your *next* spot (a **higher, cheap-to-aim tile**, not something "around the old
spot") starts from a useful heading. Skilled play is a **ping-pong**: alternate
placements between two widely separated high vantage points, each transfer landing
you pre-aimed across the gap.

**Long distance is the point, not a tolerated cost.** From a high tile you see much
more of the landscape at once, and far terrain is compressed into a small span of
angle — so the sights cursor sweeps a *long* world-distance for the *same* keyboard
input. Aiming between two distant high spots is therefore **cheap in keystrokes**
(the opposite of aiming among near tiles, which are spread across wide angles). So
reach for far, tall tiles: they give the commanding view, and — combined with the
auto-face-back and diagonal steering above — make the long exchange cheap to aim.
This pairs with the create/transfer climb ([§4](#create--try_to_create_object-1bba))
to gain height and cover ground fast.

**Reclaim what you climbed off — never abandon that energy.** Every transfer-up
leaves behind, at the tile you came from, a spent **boulder** pedestal (2 energy
each) and the **robot shell** you transferred out of (3 energy). That is your own
invested energy sitting in the world, and after the transfer you are already
pre-aimed back at it (the face-back rule above) and looking *down* at it, so
absorbing it is cheap and unblocked. So the standard hop is create→transfer→**absorb
your old boulder(s) and shell from the new vantage**, recovering the 2/3 units —
this both refills your budget for the next boulder and denies the enemy a target it
would otherwise dismantle and drain ([§6](#6-enemies)). Leave that energy behind
only when there is literally no other choice (e.g. a gaze makes reclaiming it a
worse trade than moving on).

Measured across the recorded human wins, this is an **inchworm**: a hop stacks at
most two boulders (the observed count is `k ∈ {1, 2}` — never a tall ground-up
tower), and every climb is followed by reclaiming the pedestal below, so energy
rides the ~3-unit reserve floor rather than being locked up in height. A solver's
directed climb should do the same — build ≤2, transfer up, recycle the abandoned
pedestal, repeat — instead of committing energy to a single deep stack it cannot
afford to complete.

---

## Writing a solver

This section frames the search problem the mechanics above define; the
implemented players are documented in [player.md](player.md) (reactive) and
[astar_player.md](astar_player.md) (A* search). (The sibling bit-exact forward
model is documented in [simulator.md](simulator.md); the ROM mechanics above are
its specification.)

### State

The minimal search state is the game's own RAM view: the 64 object slots
(`x,y,z_height,z_fraction,flags,h_angle,type`), `player_object`, `player_energy`,
the tile map (derivable from objects + terrain), and the per-enemy cooldowns/
targets. Terrain (height + slope field) is fixed per landscape and can be
precomputed once via [§3](#3-landscape-generation). **The PRNG state is
deliberately *not* part of the solver's search state** — a player cannot observe
it, so a faithful solver must not either (below).

### What is deterministic vs not

- **Deterministic** (fair game for planning): terrain, initial object/enemy/player
  placement, energy economy, LOS, enemy rotation direction/step, all cooldown
  cadences (in *units*).
- **PRNG-driven and off-limits to the solver**: hyperspace and meanie destinations,
  enemy discharge tree placement. These come from the PRNG, which the player cannot
  observe — so **a faithful solver must not read or predict them**. Model them as
  nondeterministic: treat a hyperspace as a jump to an unknown low tile, and plan
  so you never *depend* on where it lands. (The PRNG is used only to reproduce a
  landscape's fixed generation, never a runtime outcome.)
- **Not ROM-defined**: wall-clock timing. The world clock is compute-bound, so a
  solver should reason in **cooldown units / enemy service rounds**, not seconds.
  A move is a *duration*, not an instant — enemies keep rotating and draining
  throughout a build-and-transfer sequence.

### Core queries to implement (all reducible to the ROM routines above)

- **LOS(observer, target)** — the `$1CDD` march with true facet surfaces and the
  look-up rule. This is the workhorse; make it fast (it gates every action and
  every enemy sighting).
- **Buildable/absorbable/transferable set** from a given eye position — sweep the
  sights over the reachable angle lattice (8-unit h, 4-unit v, pitch-clamped) and
  collect the tiles `$1CDD` stamps as reachable.
- **Aim cost(target)** — the number of cursor frames to move the sights from the
  current cursor/facing onto a target tile's angle. Model it in **angle/cursor
  space**, not tile-distance, and get the mapping right, because it drives every
  routing decision:
  - The cost is roughly `max(Δcursor_x, Δcursor_y)` in cursor steps, **not the sum**
    — the cursor moves **diagonally** (both axes per frame, `move_sights $9958`), so
    off-axis targets are cheaper than they look.
  - Keystroke→world-distance is **non-linear and view-dependent**: from a high,
    long-range vantage, far tiles are compressed into a small angular span, so
    distant targets cost *few* cursor steps while near tiles are spread over wide
    angles and cost more. This is why long-range high placements are efficient, not
    expensive — favour them.
  - The facing after a transfer is **deterministic**: a created robot faces
    `creator_angle ⊕ $80` (`$1BE0`), so on transferring in you are pre-aimed back
    across the gap — compute that return aim as (near-)zero and exploit the
    **ping-pong** (alternating high placements across a large gap, paying the big
    aim only outbound).
  - Don't model a sights **off/on toggle** as free re-centring: `initialise_sights
    $134C` resets the cursor to (80, 95), discarding position — only ever a win when
    that centre is closer to the next target than the current cursor.
  Each cursor frame advances the world clock, so aim cost is a first-class term in
  the plan's cost, and cheap-to-aim targets are worth a lot.
- **Any-rotation exposure**: is a tile visible to an enemy at **any** facing it
  rotates through (not just now)? A tile is safe only if blocked from every such
  facing (§6 rotation + FOV + `$1CDD`). Treat a gaze-exposed tile as (near-)
  forbidden, not merely costly — see below.
- **Ticks-until-seen / drain-over-window**: given cooldown state and rotation
  step, how many service rounds until an enemy faces and can drain a body, and how
  much energy it costs to sit there.
- **Gaze-entry penalty**: standing (even briefly) in an enemy's gaze costs energy
  *twice*: ≥1 unit off the current body before a slow aim+transfer can leave, and
  the continued draining/downgrading of the **body you abandon** there (robot →
  boulder → tree, banked by the enemy). Both must be charged to any move that
  passes through an exposed tile, which is why such moves are last-resort.
- **Meanie safety**: a meanie spawns iff, at an enemy's drain countdown, the player
  is **partially** visible (not fully) **and** that enemy can fully see a **tree
  within 10 tiles in x and y** of the player-body. Model all three conditions and
  avoid the standing-position that satisfies them; also model the meanie as a
  *dynamic* enemy once spawned (turns ±8 units/tick toward you, hyperspaces you when
  it faces you with LOS) and the three dissolves — absorb the meanie (Sentinel
  alive), absorb the bound body, or transfer out of it — as available responses.
- **Sentry value ≫ its energy**: absorbing a sentry deletes one of the rotating
  gazes, which shrinks the *any-rotation exposure* hazard set and relaxes the time
  budget for the whole remaining plan. Value a sentry-absorb well above its +3
  energy — score it by the safe tiles / time windows it unlocks downstream — and
  prioritise clearing sentries early.

### Search shape

The action space is small and structured — the tile-targeted actions (absorb /
create{robot,tree,boulder} / transfer) each fire on a LOS-reachable tile, plus the
untargeted hyperspace and the u-turn (a free aiming flip) — but the game is a
**continuous-time, adversarial** problem. Key modelling points, all detailed in
[§7](#7-how-a-human-wins-quick-strategy):

- **Gaze is a near-hard wall.** The gaze-entry penalty makes exposed tiles
  last-resort, not merely expensive; route over safe standing tiles and climbed
  stacks toward a sight line on the platform.
- **Reducing the enemy set is its own objective.** Each sentry absorbed removes a
  wall, so clear sentries early rather than tiptoeing around them.
- **Energy is conserved and largely recoverable.** Reachable energy is bounded by
  the loose energy (trees/scenery) you can safely reach *plus* your own invested
  boulders/shells; model the climb as create→transfer→**reclaim** so the accounting
  nets out the hop cost.
- **Cost every plan in time.** Each action costs *aim + act* against a running
  world clock; pick targets jointly on position quality and aim cost from the
  current facing, and let the current tile's *ticks-until-seen* set the aim budget.
- **The win is a specific terminal sequence.** Absorb the Sentinel → robot onto the
  platform tile → transfer in → hyperspace; encode it explicitly. Because the
  Sentinel-absorb is the one-way absorb lock (`$1B8E`, [§4](#absorb--try_to_absorb_object-1b8e)),
  it must be the **last** absorb node — an ordering constraint, not a forced prefix.
- **Hyperspace landing is unknowable.** Treat it as a random relocation to a lower,
  more exposed tile; use only when any landing beats staying put.

### Faithfulness discipline

When the solver's model disagrees with the game, **read the cited ROM routine**
and fix the model to match what the 6502 does, **and trust operands over
comments** — annotator comments can mislead (e.g. `try_to_absorb_object $1B8E`
reads absolute `$0100`, the Sentinel's slot 0, not the target slot a comment might
imply). Known traps: the
absorb lock after the Sentinel falls (`$1B8E`, slot 0), the energy `AND #$3F` wrap
on over-absorb, the ½-unit (`$80`) LOS tolerance (there is no explicit multi-unit
build slack), the toroidal width-4 smoothing and 81 throwaway PRNG draws in
generation, and enemy FOV being relative to *current* facing so safety must
quantify over all rotations.
