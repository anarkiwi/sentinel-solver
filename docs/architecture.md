# Architecture

Three parts: a forward model of the game (`sentinel/`), a driver that executes against the
real game in VICE (`driver/`), and an instrument that races the two frame-for-frame. The
6502 is a test-time oracle only; its outputs are frozen as golden fixtures so CI proves the
model without the copyrighted ROM.

This document is the *machine's* structure. The player-facing rules are
[gameplay.md](gameplay.md); the players are [players.md](players.md); everything still
wrong is [open_items.md](open_items.md).

- [A. How the game works](#a-how-the-game-works) — the ROM's own structure.
- [B. ROM routines](#b-rom-routines) — address / name / what it does.
- [C. How the model mirrors it](#c-how-the-model-mirrors-it) — ROM → module → validation.
- Then the subsystems: [the model](#the-model-sentinel), aim cost,
  [landability](#landability-filter-landtablepy), render cost,
  [the driver](#the-live-driver-driver), the instrument, the measurement tools.

---

## A. How the game works

### The main loop

The game is **not frame-locked**. `update_game_and_continue $363D` runs the foreground with
no vsync wait, so the loop rate is compute-bound and dominated by the 3-D redraw.
`update_game $127C` → `update_game_loop $1289` calls `update_enemies $16B5` once per pass,
and keeps calling it until an enemy causes a visible replot.

The only fixed cadence is the raster: `$9663` (or the scroll loop `$3684` while scrolling —
the two are mutually exclusive) runs `update_enemy_cooldowns $130C` **once per video
frame**. `$130C` is a Bresenham divider that adds `$CD` to the accumulator `$1335` each
frame and calls `$1317` only on the carry (205 of every 256 frames); `$1317` decrements the
three cooldown arrays only on every third carry, gated by `$0C50`. So a cooldown "unit" is
`3 * 256 / $CD` frames, and the *frame* is the only clock either side of the instrument can
agree on.

Cooldowns live in `enemies_draining_cooldown $0C20`, `enemies_rotation_cooldown $0C28` and
`enemies_update_cooldown $0C30`, one byte per enemy slot; a byte only decrements while `>= 2`
and sticks at 1 until reset.

### Dispatching an action, and what freezes the world

Keyboard state is read by `check_for_player_input $1363` and the full scan
`check_for_full_player_input $119F`; the driver gates on the *gated* call site
`$9678`→`$967B`, which the IRQ reaches only with the want-flags `$0CE8..$0CEB` live.
`$1281` zeroes the action latch `$0C51` and only an idle full scan re-arms it at `$11EA`,
so an action fires at most once per press. The cursor auto-repeat mask `$0CC8` is reloaded
`#$6B` at `$11E0` and one gated scan is skipped per set bit (`$11F6 ASL / BCS`).

The pending action code lands in `$0C61`/`$0CE9` (action table `$139C`) and is dispatched
by `handle_player_actions $1B18`:

| Action | Code | Routine |
|---|:--:|---|
| create robot / tree / boulder | `$00` / `$02` / `$03` | `try_to_create_object $1BBA` |
| absorb | `$20` | `try_to_absorb_object $1B8E` |
| transfer | `$21` | `try_to_transfer_into_object $1B64` |
| hyperspace | `$22` | `handle_hyperspace $1B1F` |
| u-turn | `$23` | `handle_uturn $1B2F` |

Two gates run first. `consider_player_action $12D0` requires the sights to be active — but
`$12D5 CMP #$22 / BCS $12DE` lets codes `>= $22` skip that check and fall straight into
`$12E1 LSR $0CE5`. `$0CE5` bit 7 is the **enemy freeze**: it is set until the player's first
action (`$3682`/`$9659` skip the enemy clock while it is set), and `$12E1` clears it. So a
u-turn — code `$23`, free, no LOS, no energy — **unfreezes the world mid-aim**. Then, for
every code `< $22`, `$1B18` builds the aim vector and calls `check_for_line_of_sight_to_tile
$1CDD`; carry set at `$1B40`-`$1B46` plays the bad-action sound and does nothing.

The other freeze is the redraw. `set_busy_plotting $1214` sets `$0CE4` bit 7 and holds it
across a whole viewpoint redraw (`$3642` → `play_landscape_loop $357D`); while it is set the
foreground never reaches `update_enemies`, so only the raster cooldown clock advances. That
split — frames in which enemies are serviced versus frames in which the machine is plotting
— is the whole of the model's action-duration accounting.

Aiming itself is a state machine: `initialise_sights $134C` re-centres the cursor to
(80, 95) on every sights-**on** toggle, `move_sights $9958` steps both axes in one call
(`$9965` x, `$9994` y) at 1 px per gated scan within the clamps cx `$10-$8F` / cy `$20-$9F`,
and a body pan is one `pan_viewpoint $10B7` call per notch which queues 16 horizontal scroll
steps (`$10EE`) or 8 vertical (`$1135`).

### How enemies see and drain

`update_enemies $16B5` services **one** of the 8 slots per call, round-robin through the
cursor `$0090`. `is_sentinel_or_sentry $16C6` sends the Sentinel and sentries down the same
path; an enemy that owns a meanie takes the `$16FF` branch to `update_meanie $16F2` instead
and does no scanning of its own.

`consider_enemy_state $16E6` runs when the update cooldown reaches `<= 1` (reload 4 at
`$16E9`), in this order, and **returns before the rotate at `$17F9` whenever any earlier
branch fires**:

1. discharge banked energy as a tree — `consider_discharging_enemy_energy $1A5D` (`$177A`);
2. dismantle a stack — `find_drainable_boulder_or_tree_on_stack $1AB0` (`$1773`);
3. keep a still-visible held target (`$178C`), or find a new one —
   `find_drainable_robot_loop $17B2`;
4. reduce — `consider_reducing_object $183D` → `reduce_object_energy $1A08`;
5. otherwise rotate — `rotate_enemy $1805`, one fixed ±20-unit step, reloading the rotation
   cooldown to 200 at `$1813`.

Seeing is two tests. `check_if_enemy_can_see_object $1887` requires the target inside
`±($0C68/2)` of the enemy's *current* facing (`$0C68` = `$14` when scanning, `$28` for
meanie search, cone gate at `$18B8`), and
`check_if_enemy_has_line_of_sight_to_object $18E6` calls the same `$1CDD` **twice** — once
at the target's top, once `$E0` lower — writing `$0014`: `$80` full, `$40` partial, 0
unseen. Sight alone costs nothing: `target_object $1825` merely **arms** the draining
cooldown `$0C20` to 120 rounds, re-zeroed after each drain at `$1A31`. Only **full** sight
drains (`$1838`); `kill_player $1A00` fires when a drain arrives at zero energy.

A **partially** visible player is the meanie path (`$184D` → `$1852`).
`consider_creating_meanie $197D` walks the slots for a fully-visible tree within 10 tiles in
x and y of the player's body (`$19C3`/`$19D5`), flips its type to 4 in place (`$19F0`,
`attempt_to_create_meanie $19A1`), waits and retries if the scene is being plotted
(`pause_meanie_creation $19FA`), and holds the enemy's update cooldown at 50 rounds
afterwards (`$1869`). `update_meanie $16F2` then rotates the meanie ±8 units per update
until it faces the player and forces `do_hyperspace $2156` on them. `remove_meanie $1754` /
`remove_meanie_and_reset_enemy $174F` dissolve it back to a tree when it fires, when the
player transfers out of the bound body (`$1717`) or when that body is absorbed (`$1707`).

The endgame is two absolute reads. `$1B91` (inside `try_to_absorb_object $1B8E`) reads
`objects_flags[0]` — slot 0 is always the Sentinel — so once its slot empties **every**
later absorb is refused. And `do_hyperspace $2156` spends 3 energy at `$216A` and kills on
underflow at `$2170`; hyperspacing from the platform tile sets the landscape-complete flag
`$0CDE` bit 6 (`player_survived_hyperspace $217F`, `landscape_completed $3603`).

### How the scene is rendered

`plot_world $2625` is an equirectangular rasteriser walking the 32×32 grid
furthest-to-nearest: `plot_rows_in_front_of_observer_loop $26DE` counts `$0026` 31→0; per
row `find_visible_extent_of_row_of_tiles $27D7` finds the span via
`check_if_tile_is_on_screen_and_calculate_screen_coordinates $2845`; each plotted tile runs
`plot_tile $2A24` → `prepare_polygon $2D6C` / `process_line $2DF2`/`$3002` / `span_fill
$22AA` / `plot_middle_of_row $23D0`, object tiles adding `plot_stack_of_objects $21AE` and
`plot_object $8533`. Before the replot, `populate_tile_visibility_bit_table $245B` (from
`$35BA`) raytraces occlusion (`trace_rays_from_observer_to_row_of_tiles $24E2`) into the
`$3E80`/`$24DA` bitmap, which only zeroes the plot byte `$0180,X` — an occluded tile is
still *examined*, just not filled.

A whole viewpoint settle is `$3642` → `play_landscape_loop $357D`: the fixed
`$245B`/`$3700`/`fill_screen_with_background $1090`/`plot_status_bar $98B2` foreground, two
full `plot_world` passes (`$35C3`/`$35C6`), then `wait_for_end_of_tune $35D5` spinning on
the tune `play_tune $34DE` started at `$1B82`. A create or absorb instead runs the dither
loop `$1FA4` and a single replot.

---

## B. ROM routines

Every address cited anywhere in this repo, sorted. Names are the ROM's own labels; a row
without a label names the routine the address sits inside.

| Address | Routine | What it does |
|---|---|---|
| `$0D03`/`$0D05` | `multiply_byte_by_byte` | 8×8 unsigned multiply, the trig primitive |
| `$0D4A` | `divide_and_arctan` | shift/subtract 16-bit divide, then arctan of the quotient |
| `$0E75` | `sin_cos_lookup` | polynomial sine/cosine of a byte angle (256 = full circle) |
| `$0F3E` | `multiply_double_A_by_pi` | scale a 16-bit value by π |
| `$0F4A` | `multiply_double_by_byte` | 16×8 fixed-point multiply |
| `$0F9E` | `multiply_double_by_double` | signed 16×16 fixed-point multiply |
| `$1090` | `fill_screen_with_background` | clears the play buffer before a replot |
| `$10B7` | `pan_viewpoint` | one keyboard pan notch: strip clear, one `plot_world` at the intermediate angle, queue the scroll |
| `$10EE` | (in `pan_viewpoint`) | horizontal scroll — 16 steps per ±8 bearing notch |
| `$1135` | (in `pan_viewpoint`) | vertical scroll — 8 steps per ±4 pitch notch |
| `$1149` | pitch limits table | clamps the pitch band to `[$CD..$FF] ∪ [$00..$35]` |
| `$119F` | `check_for_full_player_input` | the full key scan, including the SPACE edge test |
| `$11E0` | (in the input scan) | reloads the cursor auto-repeat mask `$0CC8` with `#$6B` |
| `$11EA` | (in the input scan) | an idle full scan re-arms the action latch `$0C51` |
| `$11F6` | (in the input scan) | `ASL $0CC8 / BCS` — one gated scan skipped per set mask bit |
| `$1214` | `set_busy_plotting` | sets `$0CE4` bit 7 for the duration of a redraw |
| `$1224`/`$1238` | `put_object_in_random_tile_below_z` | random flat empty tile no higher than a given z |
| `$125A`/`$1272` | `get_random_tile_coordinate` | a `prnd` draw masked to 0..31, rejecting 31 |
| `$127C` | `update_game` | the per-pass game update |
| `$1281` | (in `update_game`) | zeroes the action latch `$0C51` |
| `$1289` | `update_game_loop` | calls `update_enemies` once per main-loop pass |
| `$12D0` | `consider_player_action` | requires the sights active before create/absorb/transfer |
| `$12D5` | (in `consider_player_action`) | `CMP #$22 / BCS $12DE` — codes `>= $22` skip the sights check |
| `$12E1` | (in `consider_player_action`) | `LSR $0CE5` — the first action unfreezes the enemy clock |
| `$130C` | `update_enemy_cooldowns` | per-frame Bresenham: `$1335 += $CD`, call `$1317` on carry |
| `$1317` | `update_enemy_cooldowns` | decrement stage, every third carry (gated by `$0C50`) |
| `$134C` | `initialise_sights` | a sights-ON toggle re-centres the cursor to `$0CC6`=80 / `$0CC7`=95 |
| `$1363` | `check_for_player_input` | the ungated input scan (three callers) |
| `$139C` | action-code table | maps a key to the action code latched in `$0CE9` |
| `$1420` | `set_palette_and_initialise_enemies` | enemy count, then placement |
| `$1450` | `initialise_player_and_trees` | player slot, energy 10, start tile, the tree scatter |
| `$14AA` | `generate_secret_code_validation_table` | builds the entry-code checker |
| `$14DC` | secret-entry-code gate | computes the jump-to-play from the validation result |
| `$14FB` | `initialise_enemies` | 8×8 grid of 4×4 sections; Sentinel + platform, then sentries |
| `$151B` | `choose_a_random_grid_section` | `prnd & mask`, rejecting out-of-range |
| `$1553` | `is_sentinel` | slot 0 is the Sentinel |
| `$1586` | `set_enemies_rotation_speed` | per-enemy fixed ±20 rotation step, direction from a random bit |
| `$159D` | `find_grid_sections_at_given_z` | sections whose highest flat tile equals z |
| `$15B5` | `calculate_mask` | selection mask for the `prnd` section draw |
| `$15CC` | `find_highest_tiles_in_grid` | per-section highest flat tile |
| `$16B5` | `update_enemies` | services one enemy slot per call, round-robin via `$0090` |
| `$16C6` | `is_sentinel_or_sentry` | the Sentinel and sentries run identical AI |
| `$16E6` | `consider_enemy_state` | discharge → dismantle → target → reduce → rotate |
| `$16E9` | (in `consider_enemy_state`) | the update-cooldown gate, reload 4 |
| `$16F2` | `update_meanie` | rotate the meanie ±8 units toward the player, then force a hyperspace |
| `$16FF` | (in `update_enemies`) | an enemy owning a meanie drives it instead of scanning |
| `$1707` | (in `update_meanie`) | meanie dissolves when its bound body is absorbed |
| `$1717` | (in `update_meanie`) | meanie dissolves when the player transfers out of that body |
| `$171B`/`$171D` | (in `update_meanie`) | faces the player, then calls `do_hyperspace` |
| `$1728` | `meanie_not_looking_at_player` | the ±8-unit rotation step toward the player |
| `$174F` | `remove_meanie_and_reset_enemy` | dissolve and clear the draining cooldown |
| `$1754` | `remove_meanie` | turn the meanie back into a tree |
| `$1773` | (in `consider_enemy_state`) | found a drainable boulder/tree — returns before the rotate |
| `$177A` | (in `consider_enemy_state`) | discharging banked energy — returns before the rotate |
| `$178C` | (in `consider_enemy_state`) | still sees its held target — returns before the rotate |
| `$17B2` | `find_drainable_robot_loop` | scans all 64 slots for a visible type-0 robot |
| `$17F9` | (in `consider_enemy_state`) | the rotate branch, reached only when nothing else fired |
| `$1805` | `rotate_enemy` | one fixed ±20-unit step |
| `$1813` | (in `rotate_enemy`) | reloads the rotation cooldown to 200 |
| `$1825` | `target_object` | records the target and ARMS `$0C20` to 120 rounds |
| `$1838` | (in `consider_reducing_object`) | only FULL sight drains |
| `$183D` | `consider_reducing_object` | fires the drain when the countdown expires |
| `$184D` | (in `consider_reducing_object`) | partially visible player → the meanie branch |
| `$1852` | (in `consider_reducing_object`) | branch into meanie creation |
| `$1869` | (in the meanie path) | holds the enemy's update cooldown at 50 rounds after a spawn |
| `$1887` | `check_if_enemy_can_see_object` | horizontal FOV `±($0C68/2)` of the *current* facing |
| `$18B8` | (in `check_if_enemy_can_see_object`) | the cone gate itself |
| `$18E6` | `check_if_enemy_has_line_of_sight_to_object` | two `$1CDD` probes → `$0014` full/partial/unseen |
| `$191F` | `calculate_player_exposure` | aggregates every enemy targeting the player |
| `$194D` | `set_bar_state` | drives the on-screen exposure bar `$0C4F` |
| `$197D`/`$1986` | `consider_creating_meanie` | deterministic slot scan for an eligible tree |
| `$19A1` | `attempt_to_create_meanie` | FOV widened to `$28`; needs full sight of the tree |
| `$19C3`/`$19D5` | (in `attempt_to_create_meanie`) | the precondition: a tree within 10 tiles in x and y |
| `$19F0` | (in `attempt_to_create_meanie`) | flips the tree's type to 4 in place — no slot allocated |
| `$19FA` | `pause_meanie_creation` | retry next tick if the scene is being plotted |
| `$1A00` | `kill_player` | a drain arriving at zero energy is death |
| `$1A08` | `reduce_object_energy` | −1 player energy, or robot → boulder → tree → gone |
| `$1A31` | (in `reduce_object_energy`) | re-zeroes the draining cooldown after each drain |
| `$1A5D` | `consider_discharging_enemy_energy` | re-emits banked energy as a tree in a random low tile |
| `$1A97` | play setup | the play-mode entry sequence |
| `$1AB0` | `find_drainable_boulder_or_tree_on_stack` | dismantles anything standing on a stack (`flags >= $40`) |
| `$1B00` | (in `consider_enemy_state`) | `SEC / BIT $0C1F / BPL` — a no-op on the common path |
| `$1B18` | `handle_player_actions` | dispatch; builds the aim vector for every code `< $22` |
| `$1B1F` | `handle_hyperspace` | the hyperspace action |
| `$1B2F` | `handle_uturn` | `objects_h_angle ⊕ $80` — free instant 180° flip |
| `$1B40`-`$1B46` | (in `handle_player_actions`) | the LOS gate; carry set → bad-action sound, no effect |
| `$1B64` | `try_to_transfer_into_object` | move `player_object` into a visible type-0 robot |
| `$1B6E` | `find_platform_below_player_loop` | sets `$0CE6` when the new body stands on the platform |
| `$1B82` | (in `try_to_transfer_into_object`) | starts the transfer tune (`start_tune $888F`, tune `#$19`) |
| `$1B8E` | `try_to_absorb_object` | absorb the topmost object in the target tile |
| `$1B91` | (in `try_to_absorb_object`) | `LDA $0100` — absolute read of `objects_flags[0]`; the Sentinel lock |
| `$1B9A` | (in `try_to_absorb_object`) | the platform (type 6) can never be absorbed |
| `$1B9E` | `absorb_object` | removes the object and banks its energy |
| `$1BBA` | `try_to_create_object` | slot, energy, placement, refund on placement failure |
| `$1BBF` | (in `try_to_create_object`) | a create may spend the meter down to 0 |
| `$1BE0` | (in `try_to_create_object`) | a created robot faces `creator_angle ⊕ $80` |
| `$1BEC` | `try_to_absorb_meanie` | +1 energy and clears the parent enemy's meanie link |
| `$1C10`/`$1C13` | `prepare_vector_from_player_sights` | cursor + view angles → aim vector; reads neither state nor slot |
| `$1C54` | `prepare_vector_from_angle` | unit direction vector from a horizontal/vertical angle pair |
| `$1C9D` | `process_sine_or_cosine` | sign/magnitude unpack of the trig lookup |
| `$1CBB` | `add_vector_to_object_position` | one ray sub-step (≈ 1/16 tile) in 3-byte fixed point |
| `$1CDD` | `check_for_line_of_sight_to_tile` | the one ray-march; carry set = blocked. Loop at `$1CE8` |
| `$1D0D` | `check_flat_tile` | surface = height nibble; `$000C` = `$80` vertical tolerance |
| `$1D2C`-`$1D32` | (in `check_flat_tile`) | the look-up rejection, waived when aiming at an object top |
| `$1D46` | `check_sloping_tile` | picks the triangle, interpolates the sloped edge |
| `$1D8A` | `tile_is_corner_or_quadrilateral` | slope-shape decision |
| `$1D9D` | `use_corner_for_slope` | corner-split facet height |
| `$1DAF` | `use_edge_for_slope` | edge-split facet height |
| `$1DF1` | slope edge table | which corner pair each slope code interpolates along |
| `$1DF9` | `calculate_tile_address_z_and_slope` | tile byte → height, slope, or object slot |
| `$1E0E` | `get_tile_z_for_line_of_sight` | the blocking/target height at an object tile |
| `$1E30` | (in `$1E0E`) | a platform adds `+$20` on top |
| `$1E3F` | `get_tile_z_from_object` | walks the stack recursively |
| `$1E48` | `get_boulder_or_tree_z_for_line_of_sight` | boulder/tree top; needs fraction `< $40` |
| `$1E5A` | (in `$1E48`) | a boulder sits `-$60` at the bottom of the band |
| `$1E69` | `is_tree` | the enemy-can-see-a-tree marker `$0CDD` |
| `$1EA4` | `get_height_of_lowest_object` | resolves a stack down to its base |
| `$1EAF` | `get_minimum_x_or_y_fraction_from_tile_centre` | "targeted" only if the ray threads near centre |
| `$1ECC` | `get_object_details` | seeds the march at the observer tile's centre |
| `$1EEF` | `remove_object` | unlink the object and repair the tile it stood on |
| `$1EFF`/`$1F16` | `put_object_in_tile` | ground, or stacked on a boulder (+½) or platform (+1) |
| `$1F38` | (in `put_object_in_tile`) | refuses a create on a tile that already carries anything else |
| `$1F83` | (in `put_object_in_tile`) | random initial facing `(prnd & $F8) + $60` |
| `$1FA4` | dither loop | the create/absorb post-action dither, loads `#$19` into `$2099` |
| `$2051` | (in the settle path) | loads `#$28` instead when `$0C4E` (meanie-made) is set |
| `$2099` | settle counter | the post-action dither countdown |
| `$210E` | `create_object` | the highest empty slot, typed |
| `$2120`/`$2122` | `create_object_from_action` / `find_empty_slot_loop` | slots 63→0 |
| `$2136` | `gain_or_lose_energy_from_object` | absorb adds, create subtracts |
| `$2143` | (in `$2136`) | carry set on underflow = "not enough energy" |
| `$2148` | `set_player_energy` | every write masks `AND #$3F` — over-absorb wraps mod 64 |
| `$214F` | `energy_in_objects` | `03 03 01 02 01 04 00` by type |
| `$2156` | `do_hyperspace` | new robot on a random flat tile of height `<= player_z + 1` |
| `$215F` | (in `do_hyperspace`) | kills below the 3-energy toll |
| `$216A` | (in `do_hyperspace`) | spends the 3-energy toll |
| `$2170` | (in `do_hyperspace`) | kills on underflow |
| `$217F` | `player_survived_hyperspace` | sets `$0CDE` bit 6 when the jump left the platform tile |
| `$21AE` | `plot_stack_of_objects` | the per-tile object stack draw |
| `$22AA` | `span_fill` | middle-of-polygon fill, 4 px/byte |
| `$23D0` | `plot_middle_of_row` | per-row span emit |
| `$245B` | `populate_tile_visibility_bit_table` | raytraced occlusion into the `$3E80`/`$24DA` bitmap |
| `$24E2` | `trace_rays_from_observer_to_row_of_tiles` | the fixed-point DDA occlusion raytrace |
| `$2565`/`$2570` | code-entry validation | the driver patches these to accept any code |
| `$2625` | `plot_world` | the equirectangular rasteriser, 32×32 grid furthest-to-nearest |
| `$26DE` | `plot_rows_in_front_of_observer_loop` | counts `$0026` 31→0 |
| `$2709` | `calculate_this_row_new_first_tiles` | row span start |
| `$2737` | `calculate_this_row_new_last_tiles` | row span end |
| `$276F` | `consider_plotting_observer_row` | the observer-row tail branch |
| `$27CE` | `plot_checkerboard_tile` | the observer's own tile, outside the `$0180` gate |
| `$27D3` | `offset_to_tile_table` | `[$00,$01,$21,$20]` — the drawn-tile offset by quadrant |
| `$27D7` | `find_visible_extent_of_row_of_tiles` | the plotted span of a row |
| `$2845` | `check_if_tile_is_on_screen_and_calculate_screen_coordinates` | the per-tile examine (trig floor) |
| `$28D4` | `calculate_tile_address` | render-path tile addressing |
| `$295D` | `plot_row_of_tiles_or_block` | the plot loop over a row |
| `$2993` | `initialise_buffer_variables` | selects the buffer window (`$29C4`) for a pan or the play view |
| `$2A24` | `plot_tile` | gates only on `$0180 != 0` |
| `$2A8A` | `plot_two_triangles` | a sloped tile is two triangles, a flat tile one quad |
| `$2ACC` | `generate_landscape` | the whole deterministic board pipeline |
| `$2ACE` | `randomise_row_or_column_tile_z_table` | 81 throwaway `prnd` draws |
| `$2AE6` | `set_landscape_vertical_scale` | `$0C08` ∈ [14..36]; landscape 0 is fixed 24 |
| `$2AFD` | `set_tile_slopes` | slope nibble for every interior tile |
| `$2B22` | `process_landscape` | modes `$80` raw / `$01` scale / `$02` nibble swap |
| `$2B4B` | `scale_tile_height` | the clamp to 1..11 |
| `$2B83` | `smooth_landscape` | 2 passes, rows then columns |
| `$2BA8` | `calculate_tile_address` | `$0400 + 256·(x&3) + 8·(x>>2) + y` — interleaved, not row-major |
| `$2BBC` | `smooth_row_or_column` | one toroidal pass |
| `$2BDF` | `level_spikes` | pulls a single-tile spike/pit to its nearer neighbour |
| `$2BFB` | `middle_is_higher_than_last` | the spike comparison |
| `$2C2C` | `average_tile_heights` | toroidal width-4 box filter |
| `$2C7C` | `calculate_tile_slope` | four corner heights → a 0..15 slope code (`$2CA8`-`$2D11`) |
| `$2D6C` | `prepare_polygon` | per-polygon edge setup, run twice per wide-buffer section |
| `$2D93`/`$2DCF` | `convert_angles_into_screen_coordinates` | vertex angles → `$A7A0`/`$0B40` screen coordinates |
| `$2DF2`/`$3002` | `process_line` | the DDA edge walk writing `$AD00`/`$AE00` |
| `$2F58` | (in `process_line`) | the steep inner loop |
| `$31CA` | `prnd` | 40-bit LFSR over `$0C7B-$0C7F`, 8 shuffles per call |
| `$339A` | `get_random_two_digit_bcd_number` | one `prnd` draw per call |
| `$33ED` | `seed_prnd_from_landscape_number` | seeds `state[0..1]` from the typed number as packed BCD |
| `$3426` | `get_maximum_number_of_enemies` | geometric draw centred on the thousands digit + 2 |
| `$3451` | `get_random_number_between_0_and_22` | one draw, range-limited |
| `$34DE` | `play_tune` | walks `$AB50 + tune_number`; note holds count down in `$0CDF` |
| `$357D` | `play_landscape_loop` | the full viewpoint settle |
| `$35A4` | load signature | `A5 0B 85` — the driver's proof the game is resident |
| `$35BA` | (in `play_landscape_loop`) | calls the occlusion raytrace |
| `$35C3`/`$35C6` | (in `play_landscape_loop`) | the two `plot_world` passes |
| `$35D5` | `wait_for_end_of_tune` | spins until the tune's bit 7 sets |
| `$3603` | `landscape_completed` | sets `$0CDE` bit 6 — the win |
| `$363D` | `update_game_and_continue` | the main loop; no vsync wait |
| `$3642` | viewpoint redraw entry | into `play_landscape_loop` |
| `$365A`/`$365D` | (in the main loop) | the `JSR pan_viewpoint` call site and the pan-done PC |
| `$3682` | (in the main loop) | skips the enemy clock while `$0CE5` bit 7 is set |
| `$3684` | scroll loop | ticks cooldowns while scrolling; mutually exclusive with `$9663` |
| `$3700` | grid angle/hypotenuse pass | fixed per-settle foreground work |
| `$3B00`/`$3C01` | arctan coefficient tables | reproduced closed-form, byte-exact |
| `$3D02` | hypotenuse coefficient table | reproduced closed-form, byte-exact |
| `$8401` | `calculate_object_relative_angles_and_distance` | relative x/y (`$85C4`), z (`$85F5`), then the angles |
| `$8475` | object transform loop | per-vertex `transform_vertex` |
| `$8533` | `plot_object` | the object model draw |
| `$888F` | `start_tune` | begins a tune, number in `$0CE7` |
| `$9287` | `calculate_angle` | bearing from a relative x/y pair |
| `$933D` | `calculate_object_relative_vertical_angle` | pitch from z and distance |
| `$937F` | `calculate_hypotenuse` | horizontal distance |
| `$9630` | raster frame marker | `DEC $0CDF`; one `$9630`→`$9630` span is exactly one frame |
| `$9659` | (in the raster IRQ) | skips the enemy clock while frozen |
| `$9663` | (in the raster IRQ) | the once-per-frame cooldown tick |
| `$9678`/`$967B` | gated full input scan | the driver's press window |
| `$98B2` | `plot_status_bar` | fixed per-settle foreground work |
| `$9925` | `PAN_DELTA` table | `$14/$F8/$04/$F4` added before the pan's `plot_world` |
| `$9939`/`$994F` | pan buffer-mode entries | vertical `A=#$00` (play window), horizontal `A=#$02` |
| `$9958` | `move_sights` | steps cx and cy in ONE call — the cursor moves diagonally |
| `$9965`/`$9994` | (in `move_sights`) | ±1 px per gated scan; clamps cx `$10-$8F`, cy `$20-$9F` |
| `$9CA0`/`$9CA1` | object vertex counts | per model type |
| `$9CAB`/`$9CAC` | object polygon counts | per model type |
| `$9D37` | rotation speed table | the per-enemy ±20 step, in RAM |

---

## C. How the model mirrors it

| ROM | Model | Validated by | Where it is approximate |
|---|---|---|---|
| `prnd $31CA`, `seed_prnd_from_landscape_number $33ED` | `prng.Prng`, `landscape.seed_for` | `golden_prng` | exact |
| `generate_landscape $2ACC` (`$2B22`, `$2B83`, `$2AFD`, `$2C7C`) | `landscape.generate`, `terrain` | `golden_landscape`; `driver.dump_stage2.verify` requires 1024/1024 tiles against the ROM's own generator | exact |
| `initialise_enemies $14FB`, `initialise_player_and_trees $1450` | `landscape._initialise_enemies`, `_initialise_player_and_trees` | `golden_landscape` | exact |
| `check_for_line_of_sight_to_tile $1CDD` (`$1D0D`, `$1D46`, `$1E0E`) | `los.check_for_line_of_sight_to_tile`, `los_jit` | `golden_los` | exact |
| `prepare_vector_from_player_sights $1C10`, `_add_vector $1CBB` | `los.prepare_vector_from_player_sights`, `los._add_vector` | `golden_los` | exact |
| the sights sweep — `move_sights $9958`, `$9965`/`$9994` | `los.landable_views`/`landable_view`/`landable_sweep_with_centres`, filtered by `landtable.crossing_mask` | `test_landable.py`, `test_landtable.py` | superset-soundness corners unproven ([10](open_items.md#10-landability-filter-unproven-corners)) |
| the action LOS gate `$1B18`/`$1B40`-`$1B46` | `aim.resolve`/`gate`/`propose` | `golden_actions` | exact |
| `try_to_create_object $1BBA`, `try_to_absorb_object $1B8E`, `try_to_transfer_into_object $1B64`, `do_hyperspace $2156` | `actions.create`/`absorb`/`transfer`/`hyperspace`/`win` | `golden_actions` | landing tile of a hyperspace deliberately unread ([Not modelled](#not-modelled-deliberate-scope)) |
| `gain_or_lose_energy_from_object $2136`, `set_player_energy $2148`, `energy_in_objects $214F` | `energy.value`/`gain`/`lose` | `golden_actions` | exact |
| `update_enemies $16B5`, `consider_enemy_state $16E6`, `rotate_enemy $1805`, `reduce_object_energy $1A08` | `enemies.update_enemies`/`_consider_enemy_state`/`_rotate_enemy`/`_reduce_object_energy`, `enemies_jit` | `golden_enemies`; `oracle.step_enemy_round` byte-exact per round | exact per round; the *cadence* is the open part (below) |
| the meanie lifecycle `$197D`/`$16F2`/`$1754` | `enemies._consider_creating_meanie`/`_update_meanie`/`_remove_meanie` | `golden_meanie` (full lifecycle + the failed-attempt path) | relocation tile unread |
| enemy sight `$1887`/`$18E6`, relative geometry `$8401`/`$9287`/`$937F`/`$933D` | `relative.relative_angles`/`can_see_object`, `threat` | `golden_relative` | `calculate_player_exposure $191F`/`set_bar_state $194D` not modelled |
| the frame clock `$9663`/`$130C`/`$1317`/`$0C50` | `enemies.advance_frame`/`cooldown_frame`, `human_clock.span_frames` | **not** golden-pinned — gated by `driver/instrument.py` and `test_enemy_sim_frame_locked_to_live_ls42` | the within-span cadence ([8](open_items.md#8-the-ls335-facing-gap-a-seven-enemy-board-diverges)) |
| the enemy freeze `$0CE5`/`$12E1`, the plot freeze `$0CE4`/`$1214` | `actioncost`, `playerbase._aim_phases` (`(frames, plotting)` segments) | `test_settle_accuracy.py` | phase split priced, not derived ([8](open_items.md#8-the-ls335-facing-gap-a-seven-enemy-board-diverges)) |
| the post-action settle `$1FA4`/`$2099`, `play_landscape_loop $357D`/`$35D5` | `actioncost.SETTLE`, `projector.viewpoint_replot_frames` | `test_settle_accuracy.py`, `test_transfer_tune_is_96_frames` | create and absorb share one constant ([4](open_items.md#4-per-step-frame-drift-and-the-unattributed-createabsorb-settle-split)) |
| `pan_viewpoint $10B7`, `$10EE`/`$1135`, `move_sights $9958` | `pancost.notch_frames`/`pan_frames`, `aimcost`, `playerbase._aim_frames` | `golden_pan_cost` — examined and filled counts byte-exact per notch | the fill proxy inside the notch ([5](open_items.md#5-terrain-fill-cost-cannot-close-per-tile)) |
| `plot_world $2625` (`$2845`, `$27D7`, `$2A24`) | `projector.project_scene`/`_scan_visible`, `projector_jit` | `golden_projector` — plotted set and examine count exact | per-tile fill cycles ([5](open_items.md#5-terrain-fill-cost-cannot-close-per-tile)) |
| occlusion `$245B`/`$24E2`/`$24DA` | `projector.occlusion_visible` | tile-for-tile against the ROM `$3E80` bitmap | exact |
| whole-frame render cost | `projector.render_cost`, `rendercost_py65` | `golden_render_cost` (py65 cycle counts) | a proxy unless `RENDER_COST_BACKEND=py65`, which skips transfer settles ([6](open_items.md#6-the-py65-exact-backend-skips-transfer-settles)) |
| enemy rotation forecasting (`$17F9` and the branches that skip it) | `playerbase._cone_onset`, `threat` | `test_plan_dwell_prediction_matches_live_ls42` (xfail) | the rotation stall is unmodelled ([3](open_items.md#3-the-gaze-forecast-assumes-rotation-never-stalls)) |
| keyboard execution `$9678`, `$365D`, `$11E0`, `$134C` | `driver/kbd_aim.py`, `driver/sentinel_execute.py` | `driver/test_live_determinism.py` | wall-clock timeouts remain ([7](open_items.md#7-the-drivers-wall-clock-timeouts-are-the-residual-load-sensitivity)) |
| the win flag `$0CDE` bit 6 (`$217F`/`$3603`) | `actions.won`/`win` | read back out of live memory by the driver | exact |

---

## The model (`sentinel/`)

| Module | Role |
|--------|------|
| `memmap.py` | RAM addresses, object types, the interleaved tile index |
| `prng.py` | the 40-bit LFSR `prnd` and landscape seeding |
| `state.py` | the canonical state: a 64 KB `bytearray` laid out like the game's RAM, with typed object-array views |
| `statecmp.py` | the labelled/tiered address schema shared with the instrument |
| `terrain.py` | height/slope nibble decode and the slope-facet surface |
| `los.py` | LOS ray-march and sights aim vector, plus the keyboard-aim buildability oracle (`landable_views`/`landable_view`/`landable_sweep_with_centres`) sweeping the cursor at 1 px — the ROM step `$9965`/`$9994` — over the full clamp (cx `$10-$8F`, cy `$20-$9F`) |
| `los_jit.py` | numba fast-march of the hot LOS loop, bit-identical to `los.py` |
| `landtable.py` | closed-form landability superset filter in front of every per-tile aim query |
| `aim.py` | the one action aim/LOS layer (`resolve`/`gate`/`propose`) — the `$1B40-$1B46` gate |
| `aimcost.py` | keyboard-aim geometry: keystrokes to pan a heading (bearing on the 8-unit lattice, pitch on 4, u-turn-aware) |
| `pancost.py` | per-notch pan redraw cost, ported from `pan_viewpoint $10B7` |
| `projector.py`, `projector_jit.py` | `plot_world $2625` terrain projector, ported bit-exactly; feeds the render-cost proxy |
| `rendercost_py65.py` | exact `plot_world` frame cost by running the real 6502 in py65, memoized; ROM-gated |
| `actioncost.py` | per-action world advance: the ROM dither/replot frame counts and the `$1335`/`$0C50` frame→tick cadence |
| `actions.py` | absorb / create / transfer / hyperspace / win (the LOS gate is the caller's) |
| `energy.py` | the energy economy (`$2136`, table `$214F`, 6-bit mask, underflow) |
| `landscape.py` | `generate(landscape) -> State`, the board generator |
| `relative.py` | object-relative bearing/distance/vertical angle, enemy FOV and visibility |
| `enemies.py`, `enemies_jit.py` | the enemy round and the frame clock (`advance_frame`/`advance_frames`) |
| `threat.py` | any-rotation tile exposure, gaze distance, ticks-until-seen, meanie safety, drain-over-window |
| `game.py` | `Game`, the facade |
| `playerbase.py` | shared player machinery: world clock, geometry, gaze windows, aim cost, firing, run loop |
| `phase_player.py`, `player.py` | the two players ([players.md](players.md)) |
| `landscan.py`, `isoview.py` | per-landscape enemy/terrain census; isometric SVG of any `State` |

### A landscape number is what you TYPE

A landscape has one id: the number a player keys in. `Game.typed(n)` builds it; the ROM
stores the typed code packed-BCD and seeds `prnd` from those two bytes
(`seed_prnd_from_landscape_number $33ED`), so `landscape.seed_for` reads the typed digits
as hex. `landscape.generate` is the only entry point that takes that PRNG value.

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

`State` is a mutable `bytearray` image; `Game.clone()` deep-copies it so a search branches
without side effects.

### The clock

`enemies.advance_frame(state, plotting=False)` is one video frame: the `$9663`/`$1317`
raster cooldown tick **first**, then `UPDATES_PER_FRAME` (8) `update_enemies` passes — one
full `$0090` sweep, so every slot is considered each frame. `plotting=True` suppresses the
sweep, modelling the replot/scroll spans in which the foreground never reaches
`update_enemies`; only the cooldown clock advances. Three mechanisms it rests on:

- **The cooldown ticks before the sweep** (`$9663`/`$1317` in the raster IRQ), so an enemy
  the tick makes due rotates in the same frame.
- **A u-turn unfreezes the world mid-aim** (`$12D5`/`$12E1`, [above](#dispatching-an-action-and-what-freezes-the-world)).
- **`UPDATES_PER_FRAME` is 8.** The foreground makes 2-4 passes a frame, but an enemy is
  rate-limited by its own `$16E9` update_cd gate (reload 4), which is tighter; sweeping
  every slot reproduces the ROM clock exactly, a literal 3 does not.

`BasePlayer._aim_phases` returns the aim as ordered `(frames, plotting)` segments, so an
action evolves the world the way the ROM does; search and executor share one sequence
(`_aim_head_tail` + `advance_phases`).

| segment | ROM | runs `$16B5`? |
|---|---|---|
| sights toggle | `$134C` re-centre + `plot_sights` | no |
| u-turn taps | action tap `$23`, no scroll, no replot | yes |
| pan notches | `$10EE`/`$1135` scroll + `plot_world` per notch | no |
| cursor drive + firing tap | gated `move_sights`/`tap_action` scans | yes |
| settle | `$1FA4`/`$86A5` dither + replot, or `$357D` redraw | no |

### Validation

Mechanics are differentially validated against the 6502 via `sentinel/tests/oracle.py`,
then frozen as JSON goldens replayed by CI: `golden_prng` (the PRNG stream), `golden_los`
(sampled aim rays), `golden_actions` (absorb/create/transfer/energy), `golden_landscape`
(terrain + object tables + PRNG state), `golden_relative` (relative geometry + full
visibility), `golden_enemies` (enemy-array trajectories every 25 rounds over 400),
`golden_meanie` (the full lifecycle plus the failed-attempt path), `golden_projector`,
`golden_pan_cost`, `golden_render_cost`.

The meanie lifecycle (tree → meanie → forced hyperspace → relocation, energy spend or
death) is pinned over the full object + enemy/meanie state round for round on landscape
2024 to round 2486, plus the failed-attempt path (landscape 49). The arctan
(`$3B00`/`$3C01`) and hypotenuse (`$3D02`) coefficient tables are reproduced from
closed-form expressions verified byte-exact against the ROM, so no game data is embedded.
The two-probe `$0014` exposure byte (`$80` full / `$40` partial / `0` unseen) — the meanie
trigger — is reconstructed bit-exact.

Seeded with a divergent ls335 state, `enemies.step` is byte-exact against
`oracle.step_enemy_round` for 119 rounds: the per-round transition function, branches
included, is right. The ROM uses **no undocumented opcodes** — tracing generation, a full
`plot_world` frame and 400 enemy rounds hits none of the 105 illegal opcodes, so a 6502
interpreter is a sound instrument. The frame clock is not golden-pinned; the instrument
gates it.

### Not modelled (deliberate scope)

- **PRNG-driven landing coordinates.** `actions.hyperspace` (`do_hyperspace $2156`) is
  faithful and `win` is gated on `$0CDE` bit6+7 with the 3-energy cost and
  death-if-underfunded, but the landing tile of a hyperspace or meanie relocation is
  deliberately unread. The draw *rate* is unmodellable — the ROM draws many times per frame
  against a cursor that moves a few steps, so PRNG phase is not observable in play. This
  limits exactly two things, both through `put_object_in_random_tile_below_z $1224`: the
  discharge tree's tile and the hyperspace tile. Meanie creation is **not** one — `$197D` is
  a deterministic slot scan, as are the hunt and the hyperspace trigger.
- **The u-turn as a player action.** The free 180° flip (`$1B2F` EOR `$80`) is priced in
  `aimcost`/`playerbase` but is not in `actions.py`.
- **Exposure-bar aggregation** (`$191F`/`set_bar_state $194D`/`$0C4F`); the underlying
  two-probe `$0014` is modelled.
- **Meanie death-credit `$0C1C = 4`** — affects only death-screen attribution.
- Sound side effects carry no gameplay state.

## Aim cost (`playerbase._aim_frames` / `_step_aim_frames`)

Priced mechanism for mechanism against the executor's key sequence.

- **Body pan is two keystroke ramps** (`pancost.pan_frames`), each notch followed by one
  `plot_world`: horizontal `$10EE` = 16 scroll steps per ±8 bearing notch, vertical `$1135`
  = 8 steps per ±4 pitch notch. Notch counts from `aimcost.h_press_count` (u-turn-aware,
  returns `(n_uturn, n_step)`) and `aimcost.v_steps`.
- **U-turn** = one action tap (`UTURN_FRAMES = 74`, measured live on ls42), 0 scroll frames,
  no redraw; taken only when it strictly lowers the keystroke count (crossover at `d >= 9`
  lattice steps).
- **Cursor is derived, not fitted.** `move_sights $9958` steps both axes in one call at
  1 px per gated scan, so a drive costs `max(|Δcx|, |Δcy|)` scans plus
  `CURSOR_RAMP = popcount($6B) = 5` scans skipped by the `$0CC8` auto-repeat mask (reloaded
  `#$6B` at `$11E0`, one skip per set bit at `$11F6 ASL / BCS`). Zero if the cursor is
  parked.
- **Sights toggle is a state transition.** A same-bearing reuse keeps sights on and drives
  from the live cursor at zero toggle cost; otherwise `TOGGLE_FRAMES` is charged and
  `initialise_sights $134C` re-centres to `SIGHTS_CENTRE = (80, 95)`.
- **A transfer charges 0 aim only on a bearing reuse** — the executor sends no aim keys
  then, `$21` firing on the object the preceding same-tile create/absorb parked the cursor
  over. Live, the predicate is the driver's, adopted by `LiveMixin._sync_aim_state`.
- `HOP_FRAMES = 700` is the window a full hop (2 creates + transfer + aims) needs;
  `SAFE_FRAMES = 250` is the window below which a tile is urgent. Both are pinned against
  the live ls42 whole-step hops recorded in `live_ls42_hops.json`
  (`test_hop_budget.py`), which charge slightly under measured.

## Landability filter (`landtable.py`)

A sound **superset filter** over the keyboard-aim lattice: given an observer and a target
tile it returns the lattice rays that *can* land there, so a per-tile query marches
thousands of rays instead of a whole heading cone. It is the path behind
`los.landable_view_targeted`.

`_get_object_details $1ECC` seeds every ray at `px_frac=py_frac=0`, `px_sub=py_sub=0x80`
(the eye tile's centre) and `prepare_vector_from_player_sights $1C10` reads neither state
nor slot, so `_add_vector $1CBB` gives, at sub-step `i`:

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
heading-arc bisection `los._tile_arc_indices`. It is a superset because terrain can only
stop a march *earlier*; a flat-terrain table would not be sound, keying on the crossing
height is what makes it hold. `surface_bounds` returns `(lo, hi)` — exact for bare terrain,
over the whole stack otherwise (platform `$1E30` `+$20` on top, boulder `$1E5A` `-$60` at
the bottom). Two exact shortcuts (`never_lands`): the observer's own tile never lands
(`$1D32`) and a sloping tile never lands (`check_sloping_tile $1D46` only loops or blocks).

**Wrap safety.** The ROM compares z as bytes, so a ray far enough above a surface aliases
onto "equal". Those visits are kept unconditionally (wildcards, and `crossing_mask`'s
`wrap_z` branch), at a small cost per arc-narrowed query, so the answer stays sound either
way — see [open item 10](open_items.md#10-landability-filter-unproven-corners).

**Lattices.** `los._landable_sweep`'s: the `$F5` **plane** (32 h × 64 cx × 128 cy =
262,144 rays) and the full **band** (× 27 pitches = 3,538,944 rays), `max_steps = 6000`.
Over targeted queries taken from real solves (ls0/42/335), arc bisection alone marches a
large fraction of the arc; adding `crossing_mask` (`landable_view`) cuts the rays marched
several-fold and answers a majority of ls42 queries as a *proven* "no view" without
marching at all. The expensive residual is an **adjacent** cell: its arc is huge and a ray
dwells long enough inside it to cross almost any surface height, so most of the arc
survives the filter.

**Whole-board set.** `landable_set` answers over the landset lattice — the band's `hgrid`
and `los._V_PRIORITY` with the cursor subsampled 2:1 (`COARSE_CX`/`COARSE_CY` ⊂
`los.CURSOR_CX`/`CURSOR_CY`), 884,736 rays. Summing per-cell queries re-marches rays, so
`stop_cells` inverts it: walk each ray's track against the real surface map and name the
tile(s) it can stop in, partitioning the lattice by landing tile. Most of the lattice
crosses no surface band before leaving the board and is never marched, so the whole-board
answer costs several times fewer rays than the per-cell sum while staying exact tile for
tile including the ring. Soundness needs one extra step: the walk may only stop where the
ray **provably** stops (`zmin <= surface_lo`); an undecidable cell (object stack with
`lo < hi`, or alias risk) makes the ray multi-candidate and always marched
(`test_stop_cell_partition_holds_every_landing`).

**Validation.** `test_landtable.py` pins the property everything rests on — over several
boards, a stacked/raised-eye stance and an `eye_z` override, every ray the full sweep lands
on a tile is in that tile's candidate set, on all three lattices including below-eye,
object and outer-ring tiles. `test_landable.py` pins the same through
`los.landable_view_targeted` for every tile of the band and coarse lattices, and through
`_view_for` for the plane. Start states alone do not exercise the object-stack surface
bracket, so `test_landable_view_matches_sweep_every_tile_midgame` re-checks every tile of a
board the player has already built and transferred on.

## Render cost (`projector.py`, `pancost.py`, `rendercost_py65.py`)

Addresses are ROM `$hex`; `FRAME_CYCLES` = 19656 (PAL). Validated against
`golden_render_cost.json` (py65 cycle counts, 15 views over generated boards
0/42/66/335/777/2024) with the raytraced occlusion table active. The `plot_world` pipeline
itself is [above](#how-the-scene-is-rendered).

`render_cost(state, view, observer, mode)` = examine floor + `prepare_polygon` floors +
area fill proxy, over `FRAME_CYCLES`, memoized on `(scene_key, observer, h, v, mode)`. With
`RENDER_COST_BACKEND=py65` and the ROM fixture present, the play-buffer player view is the
exact py65 cycle count instead.

| term | exactness |
| --- | --- |
| (a) examine trig floor: `$2845` + `calculate_angle $9287` + `calculate_hypotenuse $937F` + `calculate_object_relative_vertical_angle $933D` | count **exact**; cost `N * C_EXAMINE` (`C_EXAMINE = 1737` cyc/examine, py65-derived) |
| (b) terrain fill | plotted set **exact** (`$0180` gate); per-tile cycles approximate |
| (c) object fill | plotted set **exact**; per-object base floor, `span_fill` unmodelled |

**Occlusion `$245B` → `$24DA` → `$2845` is exact.** `projector._occlusion_visible` is a
byte-exact port validated tile-for-tile against the ROM `$3E80` bitmap: (1) temp height
table `$25C4`, per tile `(z<<1) | not_flat`; (2) horizon table `$25ED`, per tile the
**minimum** of its four corner bytes `>>1` (the CMP/BCC at `$2604-$2617` keeps the smaller
— the ROM disassembly's "maximum" label is a misnomer); (3) fixed-point DDA raytrace
`$24E2` (`$2503` signed 3-axis delta, `$2532` scale to ~2-4 substeps/tile, `$2576` march),
blocking a tile whose ray dips below the horizon table, then `$248A` ORs the 2×2 block and
applies a height test, setting the bit read at `$2911 LDA $3E80,Y / $2916 AND $24DA,Y`.
Occlusion changes **only** the plot byte: `$291B` zeroes `$0180,X` so `plot_tile` skips at
`$2A27 BEQ`, while `$007F` is untouched — occluded tiles are still examined (they pay the
trig floor) but not filled, removing roughly half the would-be-filled tiles. Object tiles
(`$28F0 CMP #$C0`) bypass occlusion. The raytrace starts at the passed observer, not
unconditionally at `state.player`.

**Tile selection and the `$0180` gate are exact.** `_scan_visible` ports `$27D7` + `$26DE`
+ the observer-row tail `$276F` branch-for-branch off the byte-exact `$2845` result, so
`N_examine` matches the 6502; the `$0C48` furthest-row hint is 0 in every fresh play state
(`$26CD`). `project_scene`'s plot loop ports `plot_row_of_tiles_or_block $295D` →
`plot_tile $2A24`, validated tile- and object-for-object against real `$0180` reads over the
15 golden views. Three facts make it exact: the plot range is `[$0037, $0038)` (split loops
`$2961`/`$2975`, column `$0038` never plotted); there is **no** on-screen filter
(`plot_tile` gates only on `$0180 != 0`, and height-0 flat tiles have byte 0); and the slot
remap `(($0025|$0005)+$001B)&$3F` means the drawn tile is examine `(col+offc, row+offr)`
with `$001B = offset_to_tile_table $27D3 = [$00,$01,$21,$20]` by quadrant, i.e. offsets
`(0,0)/(1,0)/(1,1)/(0,1)`. The observer row plots one extra tile (`$0037` when
`$0037+1==$0003`, `$0038-1` when `$0038-2==$0003`); the observer's own tile is drawn by
`plot_checkerboard_tile $27CE`, outside the gate.

**Fill** is `_terrain_poly_base` (a `prepare_polygon` floor) plus
`sum(PER_SCANLINE*H + PER_PIXEL*H*W)` over kept tiles — an area proxy, not a fit.
`convert_angles_into_screen_coordinates $2DCF`/`$2D93` is ported cycle-exact (per vertex
`screen_x` = high byte of `((h_angle16 + $0011:$0029) << 3)`; it reproduces `$A7A0`/`$0B40`
on every swept vertex and the ported cycle sum equals the ROM `conv` bucket exactly,
including the double-coordinate restart when any `h_angle16+$0011 >= $20`). Edge build is
per-polygon independent and the DDA edge walk reproduces the `$AD00`/`$AE00` writes
byte-for-byte on every narrow polygon-section swept. Per-block costs come from the loop
bodies: `process_line` steep inner loop `$2F58` = `ADC $0D`(3) `BCC`(3) `STX table`(4)
`DEC $2F60`(6) `BEQ`(2) `DEY`(2) `BNE`(3) = **23 cyc/row**, **27** on a column step, and
steep iterations are exactly 2 × filled rows for an inside polygon; `span_fill` middle
8 cyc/byte (`$23DC`, 4 px/byte); per-row edge plot `$23B5`/`$238C` ~55-70 cyc; per-8-rows
buffer advance `ADC #$39 $231F` ~15 cyc; rows walk `[$0052,$0051] = [48,240]`; off-band
`prepare_polygon` ~600 cyc/call (`C_PREP_CALL`). The fill is **prepare-dominated**: some
golden views fill zero pixels yet spend the bulk of their terrain "fill" budget on pure
`process_line` edge tracing for polygons clipping out of the band. `prepare_polygon` runs
per polygon × 2 wide-buffer sections (`$0010=0 < 2` at `$2AAB`), and a flat tile is one quad
while a sloped tile is two triangles (`plot_two_triangles $2A8A`), so a plotted tile costs
2-4 calls even when nothing fills. Why the per-tile residual cannot close:
[open item 5](open_items.md#5-terrain-fill-cost-cannot-close-per-tile).

**Object term.** `plot_object $8533` → transform loop `$8475`: per vertex `transform_vertex`
runs `calculate_sine_and_cosine` + two `multiply_byte_by_byte` + `calculate_angle` +
`calculate_hypotenuse` + `calculate_object_relative_vertical_angle`, charged as
`C_VERTEX = 2200` cyc, then per polygon the same `prepare_polygon`+`span_fill`. Model sizes
from engine facts `$9CA0`/`$9CA1` (verts) and `$9CAB`/`$9CAC` (polys): type 0=(29,27)
1=(22,25) 2=(17,15) 3=(8,10) 4=(18,25) 5=(30,35) 6=(12,11) 7=(8,4). An in-view object costs
a large fixed base plus distance-dependent fill; `_inview_object_base` sums that base over
plotted object-tiles' `$0100` stacks and, with object `span_fill` unmodelled, is a strict
floor. Constants stay env-overridable (`RENDER_C_EXAMINE`, `RENDER_PER_SCANLINE`,
`RENDER_PER_PIXEL`, `RENDER_C_VERTEX`, `RENDER_C_PREP_CALL`, `RENDER_SECTIONS`) but are
ROM-derived: a perturbation smaller than the model's own error can flip a knife-edge board,
so tuning them to win one is evidence of nothing.

**Transfer settle `$357D`.**

    viewpoint_replot_frames = TUNE_TRANSFER_FRAMES + SETTLE_FIXED_FRAMES
                              + REPLOT_PASSES * render_cost(state, view, observer)

`$0C63` moves into the target in `try_to_transfer_into_object $1B64` **before**
`play_landscape_loop $357D` runs its two `plot_world` passes (`$35C3`/`$35C6`), so both
`render_cost` and the `$245B` raytrace run from the post-transfer eye at that body's own
bearing (a created robot faces `creator_angle ^ $80`, `$1BE0`) — not the aim view, which
belongs to the abandoned eye. `playerbase._settle_eye(verb, tile)` returns that slot.

`TUNE_TRANSFER_FRAMES = 96` is ROM-derived: `play_landscape_loop` ends at
`wait_for_end_of_tune $35D5`, spinning until the tune started at `$1B82` (`start_tune
$888F`, tune `#$19` in `$0CE7`) sets bit7; `play_tune $34DE` walks `$AB50 + tune_number`
(`$AB69` for `#$19`), a byte >= `$C8` setting note length `$0C70 = (byte-$C8)*4` and a byte
< `$C8` holding it in the `$0CDF` countdown decremented once per frame by `$9630 DEC $0CDF`
— note holds sum to 96 frames, the same as the `#$0` hyperspace tune at `$AB50`
(`test_transfer_tune_is_96_frames`). `SETTLE_FIXED_FRAMES = 176` is a stand-in for four
foreground routines absent from `render_cost` — the occlusion raytrace `$245B`, the grid
angle/hypotenuse pass `$3700`, `fill_screen_with_background $1090` and `plot_status_bar
$98B2` — py65 cycle-counted on ls42 and ls335 and averaged, since the occlusion term is
scene-dependent; raster-IRQ steal is folded into it and the tune base.

`test_viewpoint_replot_lands_in_live_settle_band` asserts each prediction lands in
`[0.75*lo, 1.25*hi]` of the recorded live band (`_LIVE_SETTLES` in `test_render_cost.py`:
ls42 (338, 305, 435, 460), ls335 (259, 333, 371)) with median abs error < 15%. That band was
read through a 6 s wall-clock `run_until_pc` in `tap_action` that caps a reading at ~300
frames, so any value at or under that ceiling is indistinguishable from it. A u-turn scrolls
0 frames and is not a viewpoint replot; `_exact_render_cost` returns `None` for any
`observer != state.player`.

**Per-notch pan.** One keyboard notch is one `pan_viewpoint $10B7` call and `notch_frames`
is a direct port: the strip clear (`$3912` h / `$38AD` v, cycle counts in
`pancost._CLEAR_CYCLES_H`/`_V`) plus the ONE `plot_world` at the **intermediate** angle in
that direction's `$2993` buffer mode, plus the notch's queued 16 h / 8 v scroll steps
(`$10EE`/`$1135`). The `$9925` delta (`PAN_DELTA = $14/$F8/$04/$F4`) is added before
`JSR $2625` and fixed up after, so a right pan plots at `h + $14` (destination + `$0C`,
`$10E9 SBC #$0C`) and a downward pitch at `v - $0C` (destination − 8, `$1130 ADC #$08`);
left pans and upward pitches land on the destination. A horizontal pan is **not** the play
buffer: `$10EE` reaches `initialise_buffer_variables $2993` through `$994F` with `A=#$02`,
whose `$29C4` window (`$0007=$08`/`$0012=$84`) culls tiles the play window keeps; a
vertical pan (`$9939`, `A=#$00`) shares the play window. Examined (`$2845`) and filled
(`$2A24`) counts are byte-exact against the 6502 on every row of `golden_pan_cost.json`
(288 notches over ls0/42/335). Measured notch cost spans more than an order of magnitude, a
swing no flat base covers: `test_pan_notch_cost_matches_the_measured_plot` pins rms < 9 f,
median |error| < 6 f, |bias| < 3 f, and `test_derived_notch_beats_the_flat_base_it_replaced`
requires under half the rms of the best flat constant. The residual is the fill proxy, not
the notch model — do not add a compensating constant to `pancost`. The view-independent
`$245B` raytrace dominated a `render_cost` call and is now memoized per (scene, observer) as
`projector.occlusion_visible`, and `notch_frames` per (scene, observer, direction, plot
angle), both keyed off `projector.scene_key`, a digest of every byte `plot_world` reads.

## The live driver (`driver/`)

Executes a plan against the real game in [VICE](https://vice-emu.sourceforge.io/)
(asid-vice) inside Docker, headless: boots the tape, enters a landscape, drives the sights
by keyboard, fires create/absorb/transfer/hyperspace, and verifies each result from the
game's own memory. Imports only `sentinel/`.

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
| `sentinel_state.py` | live memory → `GameState` (`ViceSource`/`Py65Source`), `verify_entry`, `mem_image` |
| `dump_stage2.py` | regenerates `out/sentinel_stage2.bin` from the tape (the `oracle` fixture) |
| `instrument.py` | the frame-locked divergence race |
| `frozen_run.py` | RTS-stubs `update_enemies $16B5` live: isolates frame-cost fidelity |
| `plan_audit.py` | per-step audit of each `PlanStep`'s recorded budget/windows vs live |
| `replay_human.py` | replays a recorded human line into `<fixture>_truth.json` |
| `watch_play.py` | passive logger of a human playing; logs `[0,$0CFF]` plus the enemy clock |

**No wall-clock waits.** Every wait keys on a PC or memory predicate, never `time.sleep`: a
host delay is warp-dependent (warp on under `NO_RECORD=1`, off while recording), so it
would make measured frame counts differ between modes. `test_no_sleep.py` is an AST guard;
waits outside the emulated machine carry an inline `# sleep-ok: <reason>`. In play,
`bm.auto_resume = False`, so the world moves only in deliberate `run_frames`/checkpoint
windows and think time is free.

**Boot / enter / record.** `boot.boot_loaded` launches an `anarkiwi/asid-vice:latest`
container, connects a `BinMon` and polls for the load signature (`A5 0B 85` at `$35A4`);
tape timing under warp varies and can JAM the 6502, so the launch is retried. A boot
snapshot (`renders/boot.vsf`) is cached via the monitor's `MON_CMD_DUMP`/`MON_CMD_UNDUMP`
(`0x41`/`0x42`). `navigate` drives the real title menu: the "SECRET ENTRY CODE?" gate
`$14DC` computes its jump-to-play from the validation result and cannot be bypassed, so
`$14DF`, `$2565`, `$2570` are patched to accept any code. `landscape_from_digits` parses
the 4 digits as **hex** (packed BCD into `$0C7B`/`$0C7C`, decimal digits equal hex nibbles,
so `"0042"` → seed `0x42`). `_enter_play` polls emulated-frame predicates: the player is
installed when `$0C0A` is nonzero, play has started when the busy-plotting gate `$0CE4`
bit 7 releases; `vice_code_entry.vsf` is restored when present to skip the tape boot.
`boot_and_play` owns lifecycle, retries, navigation and AVI start/stop, finalizing the AVI
in a `finally` so container teardown cannot kill VICE mid-write; `validate_avi` checks the
RIFF/AVI header plus one frame in the `movi` list.

**Aim → fire → verify.** A view is a bearing (8-unit lattice), a pitch (4-unit lattice,
band `$CD..$35`) and the cursor `(cx, cy)`; `$1C10` combines them as `h_eff = h + cx>>3`,
`v_eff = v + (cy-5)>>4`. A settled press moves the cursor 9 px but `$9965`/`$9994` step
1 px, so any pixel is reachable (the search uses a step-3 window). `sentinel.aim.propose`
searches that grid with `los.aim_target` for a `(h, v, cursor)` whose native ray lands the
tile, preferring a low pan and a small tile-centre fraction, with the CPU halted.
`KbdDriver` then gates only on memory reads and checkpoint PCs:

- Coarse rotation runs **sights off**: `coarse_h` takes the shorter of ±8 steps or a u-turn
  (`$1B2F`) plus a correction, `coarse_v` pitches ±4 over the linearised band. Both HOLD
  the key and frame-step at `PC_PAN_DONE $365D` (after the foreground `JSR pan_viewpoint`
  at `$365A`, reached on commit, undo and clamp alike), ending only on state: `ok`,
  `hyperspace` (player slot `$0B` changed mid-aim) or `unreachable`.
- Fine cursor selection runs **sights on**: one checkpoint-confirmed pixel at a time
  (`move_sights` STAs `$997C`/`$9990`/`$99B8`/`$99D2`), pressing while halted and releasing
  after the store, since a held key auto-repeats in an accelerating burst (`$11F6 ASL
  $0CC8`). The sights-on toggle re-centres (`$134C`), so it is driven explicitly.
- `tap_action` fires exactly once: one idle full scan (re-arming `$0C51` at `$11EA`), then
  a press-while-halted across the gated scan call site `$9678`→`$967B` (want-flags
  `$0CE8..$0CEB`), released before the next scan — at-most-once by the single-scan press,
  at-least-once by the verify-retry loop. `sights_set` toggles SPACE (`$0C5F` bit 7), whose
  `$1236` edge latch needs that idle re-arm.
- `_run_to_scan` treats a timeout while `$0CE4` bit 7 is set as a redraw still running and
  re-arms; conceding there leaks the redraw's frames into the next primitive.

Read h/v **sights-off**, where `objects_h_angle $09C0+slot` and `objects_v_angle
$0140+slot` are settled — sights-on, the foreground loop `$363D` calls `$10B7` every frame
and its settle dance (`+$14`, `JSR $2625`, `−$0C` for a net `+8`) leaves the byte
transiently off-lattice. The cursor `$0CC6`/`$0CC7` is stable sights-on. The aim-vector
scratch `$003D`/`$003E`/`$0040` is shared with the enemy-relative-angle math and is never a
stable source. `perform_step` confirms the read-back angles and cursor reached the request
(a mismatch means a clamp or no-converge → **do not fire**, `aim_miss`), fires once, then
`verify` arbitrates the memory delta: the exact on-tile object-count change **and** the
exact energy delta, any other global object-count change being a divergence. Outcomes are
`ok`, `best_effort_miss`, `drained`, `aim_miss`, `aim_hyperspace`, `diverge` (resync +
replan) and `fail`; `classify_outcome` checks the primary effect **before** the
best-effort shortcut. A win is `$0CDE` bit 6 after `fire_hyperspace` from the platform
tile. `probe_tile` is advisory — it reads the live CPU asynchronously. A create/absorb
leaves sights ON and the bearing untouched (SPACE at `$11B3` is the only toggle), so a
matching next bearing drives only the cursor, skipping the `$134C` re-centre; a slot change
or a non-converged pan clears the committed bearing.

**Plan steps.** A step is `{verb, otype, target tile, view}` plus `min_energy` on a create
(a post-aim gate: a mid-aim drain must not push it below the reserve). `view: None` is a
**deferred aim** — an on-boulder synthoid re-aiming after the boulder landed, or an absorb
whose coarse sweep resolved no view — re-proposed against current live memory via
`aim.propose` at the player's true eye, so sim and driver never diverge on how a tile is
aimed. Transfer aims go through `LiveMixin._drive_transfer_aim`. A missed aim is a crash: a
step is aim-exact, so a miss means the model diverged. `LiveMixin` keeps think time out of
the world (`_observe` snapshots and leaves the CPU halted, `_advance` is a no-op, `_wait`
spends frames via `clock.run_frames`), and `_plan_step_stale` re-validates the next step
against the live enemy phase on the window the plan gated it with.

**Plumbing.** Host `-p` publishing is unreachable here, so every boot path connects to the
container's docker bridge IP (`boot.bridge_ip`; `BINMON_HOST`/`BINMON_PORT` override,
missing IP falls back to `127.0.0.1`). The container launches `warp=True`; `WarpMode` is
not settable on this asid-vice build (opcode `0x52` → err `0x8f`), so a failed set is
non-fatal. Concurrent runs are safe: the publish is `-p 0:6502` and `boot.kill_stale` is
scoped by `boot.stale_filter()` to `asid-vice-<own pid>-*` (`VICE_REAP_ORPHANS=1` opts into
a blanket sweep). Snapshot paths are paths *inside* the emulator process, so they must be
`/renders/...`. The monitor is frame-quantized while the CPU runs — a round-trip costs
orders of magnitude more running than halted, and slowest of all with warp off, independent
of read size — so read while halted and treat a multi-second timeout as a wait on a PC that
can never recur, not back-pressure. With warp off, any dead dwell in which the CPU runs is
live time in which the Sentinel can spawn a ring meanie. Full-image reads are done in two
32 KB halves because `mem_get`'s response length is a u16.

## The divergence instrument (`driver/instrument.py`)

Races the model against the real game frame-for-frame and reports the **first** state
disagreement, decoded to a named field. Both worlds keep their play state in a 64 KB image
at the same addresses, so one schema (`statecmp.FIELDS`) decodes either. The emulator clock
is `advance_instructions(1)` off the raster marker `$9630` then `run_until_pc($9630)` — one
`$9630`→`$9630` span is exactly one ROM frame; the sim clock is `enemies.advance_frame`.
Both are seeded from the emulator's own image at entry, so frame 0 is byte-identical, the
sim gets the real in-RAM tables (e.g. rotation speeds at `$9D37`), and board generation is
skipped entirely.

| Tier | Fields | Meaning |
|------|--------|---------|
| `CORE` | objects, enemy cooldowns, energy, tiles, discharge/meanie arrays, `$1335`, `$0C50` | a CORE divergence is a real model/ROM disagreement |
| `SWEEP` | cursor `$0090`, PRNG `$0C7B-$0C7F` | by-design non-goals: unreadable landing coords; `$0090` only orders slots within a frame |
| `SCRATCH` | `$0014`, `$0C56-$0C58`, `$0C68`, `$0C76`, `$0CDD` | LOS/targeting bytes rewritten every scan |

```bash
python -m driver.instrument 42 --frames 1200     # --follow keeps racing past a CORE event
```

Boots under warp with no recording (`NO_RECORD=1`), unfreezes the enemy clock on both sides
by clearing `$0CE5` bit7, then frame-locks and prints the per-tier first-divergence report.
`--follow` reseeds the sim from live memory on each CORE divergence and continues,
reporting event and resync counts and the frame gaps between events.

**Status:** no CORE divergence within 1200 frames on ls42;
`driver/test_enemy_sim_divergence.py::test_enemy_sim_frame_locked_to_live_ls42` gates 600
frames as a plain assertion. Fidelity here is binary — a sim that reproduces enemy phase
97% of the time is 0% correct on the outcome it decides, because one rotation step of drift
puts a body in a gaze the planner modelled empty.

## Measurement and iteration tools

### The recorded clock (`sentinel/tests/human_clock.py`)

A `watch_play/3` fixture event carries the whole pre-action enemy clock, which recovers
exactly how many frames the game advanced between two recorded actions with no cost model
in the loop. `$130C` adds `$CD` to the accumulator each frame and runs `$1317` only on the
carry (205/256 of frames); `$1317` decrements the cooldowns only on every third carry
(`$0C50`).

| quantity | what it pins |
|---|---|
| `$1335` accumulator | the frame count **mod 256** (`$CD` is invertible mod 256) |
| `$0C50` gate | lifts that to **mod 768** (205 carries per 256 frames, `205 % 3 == 1`) |
| `$0C28` sawtooth | picks the multiple — 200 rounds, reloaded at `$1813`, sticking at 1 |

A span is exact only when every voter agrees; one voter suffices, because `span_frames`
must satisfy (bres, gate) AND the decrement count jointly, so a wrong delta yields no
candidate rather than a wrong one. `span_frames` is closed-form and
`test_closed_form_matches_the_stepped_clock` checks it against the stepped loop over the
whole `(accumulator, gate)` space.

| board | enemies | capture | exact spans | clock round-trip | facings |
|---|---|---|---|---|---|
| ls0 | 1 | live | 16 | 16/16 | 16/16 |
| ls42 | 2 | live | 10 | 10/10 | 10/10 |
| ls335 | 7 | live | 18 | 18/18 | 12/18 |
| ls335 | 7 | async | 117 | 117/117 | 89/117 |

The cooldown clock round-trips perfectly everywhere; the ls335 facing gap is
[open](open_items.md#8-the-ls335-facing-gap-a-seven-enemy-board-diverges). In aggregate the
action-cost bill lands just under the measured span between genuine player actions, which is
what a correct bill must do — the human's think time sits on the measured side. Applying the
action last in its span (a bracket fires when the action lands), 83 of 91 exact-span actions
reproduce the human's next energy; the misses are off by exactly one in both directions,
i.e. drain-timing scatter inside the span. A drain does not decrement a counter — `$1A08`
**downgrades** its target (robot → boulder → tree → gone) — so an absorb whose object was
drained mid-span yields one less.

**`$0C30` is not a score.** The recorded `update_cooldown` sits on its stick value 1 in most
ls335 `watch_play` samples (async, free-running machine) but rarely in ls42 `replay_human`
samples (halted at a checkpoint): the same register reads differently by *where in the loop
the capture stops*, so scoring on it measures the recorder
(`test_update_cooldown_is_sampling_dependent_and_not_a_score`). Facings are the sound score
— a facing only moves when a rotation actually fires.

**Fixture hygiene.** The dither loop (`$1FA4`, `DITHER_FRAMES`) and the transfer tune wait
(`$35D5`, `TUNE_TRANSFER_FRAMES = 96`) are hard floors on how close two real actions can be;
8 exact ls335 spans fall below the floor for the action preceding them, so those bracket
pairs are one action recorded twice. ls335 also carries 33 events of the two known recorder
classes (enemy discharge trees, drain ticks minted as self-transfers); `human_replay` skips
them.

### Retrograde regression (`sentinel/tests/human_regress.py`)

Hands the player the human's PRE-action state at event `i` and asks it to finish alone: the
highest handover it cannot convert is the board it fails on, and the human's own move there
is the move it missed.

```bash
python -m sentinel.tests.human_regress ls335.json --out out/ls335_regress.json --diagram
python -m sentinel.isoview 335                       # the board at entry, no annotations
```

`state_at(fixture, i)` rebuilds the board from `landscape.generate(seed)` (byte-exact
terrain, never stored in the fixture) plus the event's objects, player/energy, `enemy_clock`
(true mid-game facings and rotation/drain/update cooldowns) and `cooldown_*` (`$1335` and
`$0C50`). `$0CE5` is cleared: mid-game, the enemies run. `$0090` is not recovered, so a
handover's enemy phase is right to within one round of updates.

Handovers are **bisected**, not walked — each round probes `workers` evenly spaced
handovers concurrently, so a whole fixture settles in a handful of rounds (`--linear` keeps
the exhaustive backward scan, `--indices` runs a list). Attempts run one **spawned** process
per index (a fork aborts: the numba LOS march leaves an OpenMP runtime in the parent) and a
`SIGALRM` cap does not work — raised inside the numba march it corrupts the dispatcher.
Each worker is pinned to `cores // batch` numba threads, because an uncapped
`parallel=True` `march_batch` takes a thread per core and oversubscription inflates every
attempt. Outcomes are `won`/`lost`/`capped`; `capped` means undecided, and the top capped
index is re-run alone at `escalate` × the cap before being called.

`--diagram` writes an isometric SVG (`sentinel/isoview.py`) of the first losing handover:
the 32×32 height field as a lit mesh (sloped tiles from their four ROM corner heights via
`los._slope_corner_z`), typed object glyphs at their own `z_height` drawn back-to-front,
each live enemy's scan cone as a ground wedge on its true recorded facing, the platform as a
gold diamond, the human's next actions as solid numbered arrows against the planner's
dashed, and a panel of board scalars and per-enemy bearing/cone offset.

### Checkpointed iteration (`sentinel/tests/ckpt.py`)

A planner failure is a property of a **single state**, so re-enter that state instead of
replaying the board to it. `snapshot(player, tick)` stores the 64 KB image plus the player
scalars it does not carry; `restore(snap, cls=PhasePlayer)` rebuilds. Everything else the
player holds (`_view_memo`, `_cone_memo`, `_hop_price_memo`, `_hold_memo`, the module-level
`_VIEW_CACHE`) is a pure cache keyed on a state signature. Two images are stored — the board
the player was **constructed** on (for any player whose caches snapshot it) and the live one
— and the deep copies on both sides are load-bearing: storing references makes every
checkpoint alias the final tick, and assigning the snapshot's own list back to the player
makes replay mutate the checkpoint.

| tier | question |
|---|---|
| 0 filter tally | which gate kills each `(tile, k)` here? |
| 1 probe | does the candidate generator (`_climb_candidates`/`_mount`) yield anything here? |
| 2 resume | does the run still die from this tick on? |
| 3 board | does the whole board flip to a win? |
| 4 matrix | did any other board regress? (`human_regress`) |

Exercised on a reduced ls335 board, enemies `{(4,18), (12,10)}` plus the Sentinel. The tiers
are ordered by cost, each far cheaper than the one below it: promote a change only when the
tier below passes, so a generator change is judged against the stance that defeats it rather
than by replaying a whole board. The fidelity gate — restoring at
tick *t* and replaying to the end reproduces the run's own trace tail entry for entry — must
assert a non-zero `actions_replayed`; both deep-copy bugs above first presented as
`identical: true` over an empty tail.

**Determinism contract.** Anything whose result is compared must be bounded by node budget
only, never a wall clock: a wall-clock cut makes the search a function of host load, and
with the clock out of the loop parallelism changes wall time and never a verdict.
