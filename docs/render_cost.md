# plot_world ($2625) render-projection frame-cost model

Reverse-engineered spec for the ROM's terrain rasteriser (`sentinel/projector.py`,
`sentinel/pancost.py`). Addresses are ROM ($ hex); PAL frame = 19656 cycles. Validated
against `golden_render_cost.json` (py65 cycle counts, 15 views over landscapes
0/42/66/335/777/2024) with the raytraced occlusion table active. Measured accuracy and the
ranked open problems: [plan_fidelity.md](plan_fidelity.md).

## What plot_world does

An equirectangular rasteriser walking the 32x32 tile grid furthest-to-nearest.
`plot_rows_in_front_of_observer_loop` ($26DE) counts `$0026` 31 -> 0; per row
`find_visible_extent_of_row_of_tiles` ($27D7) finds the on-screen span via
`check_if_tile_is_on_screen_and_calculate_screen_coordinates` ($2845); each plotted tile
runs `plot_tile` ($2A24) -> `prepare_polygon` ($2D6C) / `process_lines`+`process_line`
($2DF2/$3002) / `span_fill` ($22AA) / `plot_middle_of_row` ($23D0), object tiles adding
`plot_stack_of_objects` ($21AE). Before the replot,
`populate_tile_visibility_bit_table` ($245B, from $35BA) raytraces occlusion into the
`$3E80`/`$24DA` bitmap `plot_tile` consults.

`render_cost(state, view, observer, mode)` = examine floor + terrain/object
`prepare_polygon` floors + area fill proxy, over `FRAME_CYCLES`, memoized on
`(scene_key, observer, h, v, mode)`. With `RENDER_COST_BACKEND=py65` and the ROM fixture
present, the play-buffer player view is instead the exact py65 cycle count
(`sentinel/rendercost_py65.py`, memoized on the render-relevant board bytes + view).

| term | share of plot_world | exactness |
| --- | --- | --- |
| (a) examine trig floor: `$2845` + `calculate_angle` ($9287) + `calculate_hypotenuse` ($937F) + `calculate_object_relative_vertical_angle` ($933D) | 16-78% (median 35%) | count **exact**; cost `N * C_EXAMINE`, py65 mean **1737 cyc/examine** (spread 1551-2046) |
| (b) terrain fill | 15-84% (median 33%) | plotted set **exact** ($0180 gate); per-tile cycles approximate |
| (c) object fill | 0-42% (median 14%) | plotted set **exact**; per-object base floor, `span_fill` unmodelled |

## Occlusion: $245B -> $24DA -> $2845 (EXACT)

`projector._occlusion_visible(state, observer)` is a byte-exact port of
`populate_tile_visibility_bit_table` ($245B), validated tile-for-tile against the ROM
`$3E80` bitmap (`test_occlusion_table_is_byte_exact`).

1. **Temp height table** (`populate_temporary_tile_z_table` $25C4): per tile
   `(z<<1) | not_flat`, `z` from `terrain.resolve_ground`, `not_flat` = slope != 0.
2. **Horizon table** ($25ED): per tile the **minimum** of its four corner bytes, `>>1` (the
   CMP/BCC at $2604-$2617 keeps the smaller each step -- the "maximum" label in the ROM
   disassembly is a misnomer). Flat tiles use their own height.
3. **Raytrace** (`trace_rays_from_observer_to_row_of_tiles` $24E2): fixed-point DDA
   observer -> tile ($2503 signed 3-axis delta; $2532 scale to ~2-4 substeps/tile; $2576
   march), blocking the tile if the ray height dips below the horizon table at any stepped
   cell. `$248A` ORs the 2x2 raytrace block (dilation) and applies a height test (a flat
   tile above eye level is hidden), setting the `$3E80` bit read at
   `$2911 LDA $3E80,Y / $2916 AND $24DA,Y`.

Occlusion changes **only** the plot byte: at `$291B` a hidden non-object tile has `$0180,X`
zeroed, so `plot_tile` skips it at `$2A27 BEQ`. `$007F` (the on-screen result) is
untouched, so occluded tiles are still **examined** (they pay the trig floor) but not
filled; roughly half the would-be-filled tiles are removed. Object tiles ($28F0 `CMP #$C0`)
bypass occlusion and always plot. The raytrace starts at the passed `observer`, not
unconditionally at `state.player`, so a non-player eye never mixes the `$2625` setup of one
object with the `$245B` rays of another.

## Exact tile selection and the $0180 plotted-set gate

`projector._scan_visible` ports `find_visible_extent_of_row_of_tiles` ($27D7) +
`plot_rows_in_front_of_observer_loop` ($26DE) + the observer-row tail ($276F)
branch-for-branch, driven by the byte-exact `$2845` on-screen result, so `N_examine`
matches the 6502 exactly. The `$0C48` furthest-row extent hint is 0 in every fresh play
state ($26CD).

`project_scene`'s plot loop is a byte-exact port of `plot_row_of_tiles_or_block` ($295D) ->
`plot_tile` ($2A24), validated tile- and object-for-object against real `$0180` reads over
the 15 golden views. The examine pass writes each tile's content byte to
`$0180[col|$0005]`; `$291B` zeroes it if occlusion-hidden; the plot pass re-walks each row
and draws every column whose slot is nonzero. Three facts make it exact:

1. **Plot range is `[$0037, $0038)`** -- the split forward/backward loops
   (`plot_start_of_row_loop $2961` / `plot_end_of_row_end $2975`) cover
   `[$0037, $0038-1]`; column `$0038` is never plotted. `$0037/$0038` equal the merged
   extent `(min(start,p_start), max(end,p_end))` `_scan_visible` emits.
2. **No on-screen filter.** `plot_tile` gates only on `$0180 != 0`; off-screen tiles with a
   nonzero byte are still drawn (they clip inside the rasteriser). Height-0 flat tiles have
   byte 0 and are skipped.
3. **Slot remap.** `plot_tile` reads `$0180` at `(($0025|$0005)+$001B)&$3F`, so the drawn
   tile is examine `(col+offc, row+offr)` with `offc=$001B&1`, `offr=($001B>>5)&1`,
   `$001B = offset_to_tile_table $27D3 = [$00,$01,$21,$20]` by quadrant. `offr=1` reads the
   other buffer bank = the previous (further) row; drawn-tile offsets are
   `(0,0)/(1,0)/(1,1)/(0,1)` for quadrants 0/1/2/3.

The **observer row** ($276F) plots one extra tile: `$0037` when `$0037+1==$0003`, or
`$0038-1` when `$0038-2==$0003`; the observer's own tile is drawn by
`plot_checkerboard_tile` ($27CE), outside the `$0180` gate.

## Fill: what is exact, and why the residual cannot close per tile

The fill charge is `_terrain_poly_base` (a `prepare_polygon` floor) plus
`sum(PER_SCANLINE*H + PER_PIXEL*H*W)` over kept tiles -- an area proxy, not a fit.

- **`convert_angles_into_screen_coordinates` ($2DCF/$2D93) is ported cycle-exact.** Per
  vertex `screen_x = high byte of ((h_angle16 + $0011:$0029) << 3)`; the sign-extended
  `$0B40` high byte reproduces ROM `$A7A0`/`$0B40` byte-for-byte on all 3574 swept vertices
  and the ported cycle sum equals the ROM `conv` bucket exactly (258628 cyc), including the
  double-coordinate restart ($2D93, when any `h_angle16+$0011 >= $20`).
- **Edge build is per-polygon independent** -- convert, `process_lines` dispatch,
  `rasterise_polygon_edge` and `process_line` read only the polygon's own projected
  vertices and fixed buffer/band vars. The DDA edge walk reproduces the ROM `$AD00`/`$AE00`
  edge-table writes byte-for-byte on every narrow polygon-section of the sweep (534/534);
  its cycle count transcribes to within a few percent.
- **Per-block cycle costs from the loop bodies.** `process_line` steep inner loop ($2F58):
  `ADC $0D`(3) `BCC`(3 taken) `STX table`(4) `DEC $2F60`(6) `BEQ`(2) `DEY`(2) `BNE`(3) =
  **23 cyc/row**, **27** on a column step (`+SBC $0C`, `+INX`, `BCC` not taken); steep
  iterations = exactly 2 x filled rows for an inside polygon. `span_fill` middle 8 cyc/byte
  ($23DC: unrolled `LDY #imm` 2 + `STA ($70),Y` 6, 4 px/byte); per-row edge plot
  ($23B5/$238C) ~55-70 cyc; per-8-rows buffer advance (`ADC #$39` $231F) ~15 cyc; rows walk
  `[$0052,$0051]=[48,240]`. Off-band `prepare_polygon` ~600 cyc/call (`C_PREP_CALL`).
- **The fill is prepare-dominated, not span-dominated.** 5 of the 15 sweep views fill zero
  pixels yet spend 3k-450k cyc of terrain "fill" -- pure `process_line` edge tracing for
  polygons clipping out of the band. `prepare_polygon` runs per polygon x 2 wide-buffer
  sections (`$0010=0 < 2` at $2AAB), and a flat tile is one quad while a sloped tile is two
  triangles (`plot_two_triangles` $2A8A), so a plotted tile costs 2-4 `prepare_polygon`
  calls even when nothing fills; `_terrain_poly_base` charges exactly that
  (`test_offband_tiles_still_cost_their_prepare_polygon_calls`,
  `test_terrain_polygon_floor_counts_two_triangles_for_a_sloped_tile`).

**Blocking fact: `span_fill` cost is not a per-tile function.** `polygon_left_edge_table`
($AD00) and `polygon_right_edge_table` ($AE00) are **never cleared** between polygons. A
polygon clipping to a sliver writes only some of the `[$0004,$0006]` rows; `span_fill` then
reads **stale** left/right columns left by a previous polygon (verified: a row's `$AE00`
byte matched none of the current triangle's three `$A7A0` values, only a prior polygon's).
Middle-fill length is `right_col - left_col`, so exact `span_fill` cycles need a
**stateful emulation of the whole `plot_world` fill sequence in render order**, including
interleaved object polygons writing the same two tables. Hence no closed-form per-tile fill
(filled-rows/y-extent ratio 0.38-2.26); `H` is also 0 on views where the ROM fills 100k+
cyc (every corner `screen_y` below the inner band) and per-tile fill spans 2.5k-170k cyc,
so the residual is neither area- nor H-linear.

## Object term (c)

`plot_object` ($8533 -> transform loop $8475): per vertex `transform_vertex` runs
`calculate_sine_and_cosine` + two `multiply_byte_by_byte` + `calculate_angle` +
`calculate_hypotenuse` + `calculate_object_relative_vertical_angle` ~ **2200 cyc**
(`C_VERTEX`); then per polygon the same `prepare_polygon`+`span_fill`. Model sizes from
engine facts `$9CA0/$9CA1` (verts), `$9CAB/$9CAC` (polys): type 0=(29,27) 1=(22,25)
2=(17,15) 3=(8,10) 4=(18,25) 5=(30,35) 6=(12,11) 7=(8,4). An in-view object costs a
**~63k-95k base** (vertex trig + `np x SECTIONS` `prepare_polygon`) plus distance-dependent
fill up to ~213k when close. `_inview_object_base` walks the plotted object-tiles' `$0100`
flags stacks and sums that base; with object `span_fill` unmodelled it is a strict floor
that never overshoots (`test_object_base_never_overshoots_and_is_present`).

Constants stay env-overridable (`RENDER_C_EXAMINE`, `RENDER_PER_SCANLINE`,
`RENDER_PER_PIXEL`, `RENDER_C_VERTEX`, `RENDER_C_PREP_CALL`, `RENDER_SECTIONS`) but are
ROM-derived, not fitted: a perturbation smaller than the model's own error can flip a
knife-edge board's outcome, so tuning them to win a board is evidence of nothing.

## Transfer settle ($357D)

    viewpoint_replot_frames = TUNE_TRANSFER_FRAMES + SETTLE_FIXED_FRAMES
                              + REPLOT_PASSES * render_cost(state, view, observer)

The viewpoint object `$0C63` moves into the target in `try_to_transfer_into_object` ($1B64)
**before** `play_landscape_loop` ($357D) runs its two `plot_world` passes ($35C3/$35C6), so
both `render_cost` and the `$245B` raytrace run from the **post-transfer eye**, at that
body's own bearing (a created robot faces `creator_angle ^ $80`, $1BE0) -- not the aim
view, which belongs to the abandoned eye. `playerbase._settle_eye(verb, tile)` returns that
slot and `_settle` prices from it; the aim `view` is unused for a transfer. `2*plot_world`
alone is 4-158 f against a live settle of hundreds; the rest is two fixed terms.

**`TUNE_TRANSFER_FRAMES = 96` (ROM-derived).** `play_landscape_loop` ends at
`wait_for_end_of_tune` ($35D5), an `update_sound`/`BPL $0CE7` spin blocking until the tune
started at $1B82 (`start_tune $888F`, tune #$19 in `$0CE7`) sets bit7. `play_tune` ($34DE)
walks `$AB50 + tune_number` ($AB69 for #$19): a byte >=$C8 sets note length
`$0C70 = (byte-$C8)*4`, a byte <$C8 holds `$0C70` frames in the `$0CDF` countdown, $FF ends.
`$0CDF` decrements once per frame in the raster IRQ (`$9630 DEC $0CDF`, floored at 0). Note
holds sum to **96 frames**, the same as the #$0 hyperspace tune ($AB50,
`projector.TUNE_TRANSFER_FRAMES`); `test_transfer_tune_is_96_frames` decodes both from the
image.

**`SETTLE_FIXED_FRAMES ~ 176` (fitted stand-in).** Four fixed foreground routines run
before the two `plot_world` passes and are absent from `render_cost`; py65 foreground
cycle-counted (`/19656`):

| routine | ROM | ls42 | ls335 |
| --- | --- | --- | --- |
| occlusion raytrace `populate_tile_visibility_bit_table` | $245B | 110f | 63f |
| grid angle/hypotenuse pass | $3700 | 82f | 82f |
| `fill_screen_with_background` | $1090 | ~1f | ~1f |
| `plot_status_bar` | $98B2 | 7f | 7f |
| **sum** | | **199f** | **152f** |

Occlusion cost is scene-dependent, so 176 is the **mean of these two scenes** -- unlike the
ROM-derived 96. Raster-IRQ steal (~10-25%) over the settle is folded into it and the tune
base.

`test_viewpoint_replot_lands_in_live_settle_band` asserts, over the ls42/ls335 sweep views,
that every prediction lands in `[0.75*lo, 1.25*hi]` of the recorded live band and that the
median abs error is < 15%. That band is `ls42 (338, 305, 435, 460)`, `ls335 (259, 333, 371)`
(`_LIVE_SETTLES` in `sentinel/tests/test_render_cost.py`), i.e. 259-460 f. It was measured
through a 6 s wall-clock `run_until_pc` that caps a reading at ~300 frames, so any value at
or under ~300 is indistinguishable from that ceiling: re-measure before using it as ground
truth for anything beyond this test's loose bracket.

A u-turn (EOR $80 bearing flip) scrolls 0 frames and is not a viewpoint replot.
`_exact_render_cost` returns `None` for any `observer != state.player`, and a transfer
settle is always priced from the post-transfer slot, so `RENDER_COST_BACKEND=py65` falls
back to the proxy on that whole path.

## Per-notch pan redraw (`sentinel/pancost.py`)

One keyboard notch is one `pan_viewpoint` ($10B7) call, and `notch_frames` is a direct port
of it: the strip clear plus the ONE `plot_world` at the intermediate angle, in that
direction's `$2993` buffer mode. `notch_plots` enumerates the notches a keyboard aim
animates in executor order (u-turn-aware bearing steps, then pitch -- a u-turn contributes
none); `pan_frames` adds the caller's queued scroll steps. Three ROM facts:

- **One plot_world per notch**, not a fraction, plus the notch's queued 16 h / 8 v scroll
  steps ($10EE/$1135) and a strip clear ($3912 h / $38AD v; exact cycle counts in
  `pancost._CLEAR_CYCLES_H`/`_V`).
- **The plot runs at the INTERMEDIATE angle.** The $9925 delta
  (`PAN_DELTA = $14/$F8/$04/$F4`) is added *before* `JSR $2625` and fixed up after, so a
  right pan plots at `h + $14` (destination + $0C, `$10E9 SBC #$0C`) and a downward pitch
  at `v - $0C` (destination - 8, `$1130 ADC #$08`). Left pans and upward pitches land on
  the destination.
- **A horizontal pan is not the play buffer.** `$10EE` reaches
  `initialise_buffer_variables` ($2993) through `$994F` with `A=#$02`, whose `$29C4` window
  is `$0007=$08`/`$0012=$84` and culls tiles the play window keeps. A vertical pan
  (`$9939`, `A=#$00`) shares the play window, so only bearing notches need the extra mode.

`project_scene` takes the mode and threads its window through the `$293C` on-screen test,
so the examined ($2845) and filled ($2A24) counts are **byte-exact against the 6502 on
every row of `golden_pan_cost.json`** (288 notches over ls0/42/335). Measured notch cost
spans **3.8 to 99.8 frames** (median 22.2) -- a swing no flat base can cover;
`test_pan_notch_cost_matches_the_measured_plot` pins rms < 9 f, median |error| < 6 f,
|bias| < 3 f, and `test_derived_notch_beats_the_flat_base_it_replaced` requires under half
the rms of the best possible flat constant. The residual is the fill proxy above, not the
notch model: tile selection is exact and `C_EXAMINE` is centred (measured mean 1704
cyc/examine vs 1737 charged). Do not add a compensating constant to `pancost`.

**Evaluation cost.** ~62% of a `render_cost` call was the view-independent `$245B`
raytrace; it is memoized per (scene, observer) as `projector.occlusion_visible`, and
`notch_frames` per (scene, observer, direction, plot angle). Both key off
`projector.scene_key`, a digest of every byte `plot_world` reads -- a net planner speed-up
despite the extra plots per aim.
