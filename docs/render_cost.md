# plot_world ($2625) render-projection frame-cost model

Reverse-engineered spec for the ROM's terrain rasteriser. All addresses are ROM
($ hex). PAL frame = 19656 cycles. Validated against `golden_render_cost.json`
(py65 cycle-counts, 15 views x 5 landscapes) with the raytraced occlusion table
active, exactly as the live game runs it.

## What plot_world does

`plot_world` ($2625) is an equirectangular terrain rasteriser. It walks the 32x32
tile grid furthest-to-nearest:

- `plot_rows_in_front_of_observer_loop` ($26DE) iterates the row counter `$0026`
  from 31 down to 0 (<=32 rows).
- Per row, `find_visible_extent_of_row_of_tiles` ($27D7) finds the on-screen
  horizontal tile span via `check_if_tile_is_on_screen_and_calculate_screen_coordinates`
  ($2845).
- Each plotted tile is drawn by `plot_tile` ($2A24) -> `plot_polygon` /
  `prepare_polygon` ($2D6C) / `process_lines`+`process_line` ($2DF2/$3002) /
  `span_fill` ($22AA) / `plot_middle_of_row` ($23D0). Object tiles additionally draw
  a stack of object polygons via `plot_stack_of_objects` ($21AE).

Before the replot, `populate_tile_visibility_bit_table` ($245B, called from $35BA)
raytraces terrain occlusion into the `$3E80`/`$24DA` bitmap that `plot_tile` consults.

## Cost decomposition (three terms)

`plot_world` cost splits, per the golden breakdown, into:

- **(a) Per-EXAMINED-tile trig floor.** Each `check_if_tile_is_on_screen` ($2845)
  call runs `calculate_angle` ($9287) + `calculate_hypotenuse` ($937F) +
  `calculate_object_relative_vertical_angle` ($933D). py65 cycle-counting the whole
  $2845 call-tree gives **1737 cyc/examine** (mean; 1551-2046 across tiles, the
  scale-loop / divide-round spread). `N_examine` is the exact $2845 call count.
- **(b) Terrain fill.** `span_fill` fills each polygon row's middle at **8 cyc/byte**
  (`plot_middle_of_row` $23DC: unrolled `LDY #imm` 2 + `STA ($70),Y` 6), plus per-row
  edge plotting (`plot_left/right_edge_of_row` $23B5/$238C) and the `process_line`
  ($3002) edge rasteriser, all clipped per vertical buffer band.
- **(c) Object fill.** `plot_stack_of_objects` ($21AE) renders each object in the
  tile column as its own set of polygons (through the same `span_fill`).

Terms (b)+(c) are the "fill". Golden fractions across the sweep:

| term | share of plot_world | exactness in `render_cost` |
| --- | --- | --- |
| (a) examine | 16-78% (median 35%) | count **exact**; cost `N*1737` (median 5.6%, max 14%) |
| (b) terrain fill | 15-84% (median 33%) | approximated (residual, below) |
| (c) object fill | 0-42% (median 14%) | **not modelled** (residual, below) |

## Occlusion: $245B -> $24DA -> $2845 (EXACT)

`projector._occlusion_visible` is a byte-exact port of
`populate_tile_visibility_bit_table` ($245B), validated tile-for-tile against the
real ROM `$3E80` bitmap (0 mismatches, all sweep landscapes;
`test_occlusion_table_is_byte_exact`). Three stages:

1. **Temp height table** (`populate_temporary_tile_z_table` $25C4): per tile,
   `(z<<1) | not_flat` where `z` is the terrain/lowest-object height
   (`terrain.resolve_ground`) and `not_flat` = slope != 0.
2. **Horizon table** ($25ED): per tile, the **minimum** of the tile's four corner
   bytes, `>>1` (the CMP/BCC at $2604-$2617 keeps the smaller each step -- the
   "maximum" label is a misnomer). Flat tiles use their own height.
3. **Raytrace** (`trace_rays_from_observer_to_row_of_tiles` $24E2): for each tile a
   fixed-point DDA marches observer->tile ($2503 signed 3-axis delta; $2532 scale to
   ~2-4 substeps/tile; $2576 march), blocking the tile if the ray height dips below
   the horizon table at any stepped cell. `$248A` then ORs the 2x2 raytrace block
   (dilation) and a height test (a flat tile above eye level is hidden), setting the
   `$3E80` bit that $2845 reads at `$2911 LDA $3E80,Y / $2916 AND $24DA,Y`.

The occlusion decision changes **only** the plot byte: at `$291B` a hidden non-object
tile has `$0180,X` zeroed, so `plot_tile` skips it at `$2A27 BEQ`. It never touches
`$007F`, the on-screen result -- so occluded tiles are still **examined** (they cost
the $2845 trig floor) but not filled. `project_scene` mirrors this exactly: it keeps
the examine walk untouched (`N_examine` stays 0-mismatch) and drops hidden non-object
tiles before the fill sum. Object tiles ($28F0 `CMP #$C0`) bypass occlusion and always
plot, so the grid gates terrain only. In the sweep this removes roughly half of the
"would-be-filled" tiles (e.g. ls0 view 0,0,0: 61 plot_tile calls, 48 hidden -> 13 filled).

## Exact tile selection (find_visible_extent)

`projector._scan_visible` is a faithful port of `find_visible_extent_of_row_of_tiles`
($27D7) + `plot_rows_in_front_of_observer_loop` ($26DE) + the observer-row tail
($276F). Driven by the byte-exact on-screen result of $2845, it reproduces the ROM's
furthest->nearest scan branch-for-branch, so `N_examine` matches the real 6502
**exactly** (0 mismatches across the sweep). The $0C48 furthest-row extent hint is 0
in every fresh play state ($26CD).

## Fill term: the exact residual

`render_cost`'s fill is still `sum(60*H + 1.75*H*W)` over the kept tiles. Making it
frame-exact needs the full render engine ported and cycle-counted. This pass measured
every block's exact cost and its geometric driver (below) but does NOT ship a native
model, because the drivers cannot yet be computed exactly from the projector's geometry
(see "Why the native drivers diverge"); shipping a coefficient fit to close the gap is
the forbidden `97%=0%` anti-pattern.

### The fill is prepare-dominated, not span-dominated (measured)

Per-phase py65 cycle brackets over the 15 sweep views (`prepare_polygon` subtree vs
`span_fill` subtree vs `plot_stack_of_objects` subtree):

- **`span_fill` frequently never runs.** 5 of 15 views fill zero pixels (`nspan=0`)
  yet still spend 3k-450k cyc of terrain "fill". The cost is `process_line` ($3002)
  building the `polygon_left/right_edge_table`s ($AD00/$AE00) for polygons that then
  clip out of the band -- pure edge-trace overhead, no `span_fill`.
- **`prepare_polygon` ($2D6C) is called per polygon x 2 wide-buffer sections** (the
  play buffer is wide: `$0010=0 < 2` at $2AAB, so `plot_polygon` runs two
  `prepare_polygon`+`span_fill` passes). A flat tile is one quad, a sloped tile two
  triangles (`plot_two_triangles` $2A8A), so a plotted tile costs 2-4 `prepare_polygon`
  calls; every scan-visible non-hidden tile pays this even when nothing fills.

### Exact per-block cycle costs (derived from the loop bodies)

- **`process_line` steep inner loop** ($2F58): `ADC $0D`(3) `BCC`(3 taken) `STX
  table`(4) `DEC $2F60`(6) `BEQ`(2) `DEY`(2) `BNE`(3) = **23 cyc/row**, or **27** on a
  column step (`+SBC $0C`(3) `+INX`(2), `BCC` not taken). Steep-loop iteration count =
  **exactly 2 x filled-rows** for an inside polygon (each filled row is bounded by a
  left and a right edge) -- verified per tile (`steep = 2*srows`).
- **`span_fill` middle** (`plot_middle_of_row` $23DC): unrolled `LDY #imm`(2)+`STA
  ($70),Y`(6) = **8 cyc/byte** (4 px/byte). Per-row edge plot (`plot_left/right_edge_of_row`)
  ~55-70 cyc; per-8-rows buffer advance (`ADC #$39` $231F) ~15 cyc. Rows walk the band
  `[$0052,$0051] = [48,240]` top-to-bottom.
- **`prepare_polygon`** off-band (all clip, no fill): ~600 cyc/call; with tracing it
  carries the `process_line` cost above.

### Object renderer reuses the SAME rasteriser (measured)

`plot_object` ($8533 -> transform loop $8475): per vertex, `transform_vertex` runs
`calculate_sine_and_cosine` + two `multiply_byte_by_byte` + `calculate_angle` +
`calculate_hypotenuse` + `calculate_object_relative_vertical_angle` ~ **2200 cyc**;
then per polygon it calls the same `prepare_polygon`+`span_fill`. Per-type model sizes
(engine facts `$9CA0/$9CA1` verts, `$9CAB/$9CAC` polys): type 0=(29v,27p) 1=(22,25)
2=(17,15) 3=(8,10) 4=(18,25) 5=(30,35) 6=(12,11) 7=(8,4). An in-view object costs a
**~63k-95k base** (vertex trig + `np x 2` `prepare_polygon`) plus distance-dependent
fill up to ~213k when close -- the largest single error source (0-42% of plot_world).

### Why the native drivers diverge (the port gap)

The projector's per-tile `H`/`W` and kept-tile set are NOT the ROM's rasterised
polygons:

- **`H` is 0 where the ROM fills 100k+ cyc.** All-prep views (e.g. `0,48,8`,
  `335,64,16`) project every corner `screen_y` below the inner band, so the
  `[0,240]`-clamped `H` is 0 while `process_line` spends its full prep cost.
- **The kept-tile set undercounts.** `plot_tile` gates on the `$0180` scan buffer, a
  DIFFERENT table from the `$3E80` occlusion bitmap `project_scene` filters on; ROM
  `plot_object` calls exceed the projector's object-tile count (e.g. `777,32,0`: ROM
  plots 5 objects, projector finds 0). So even a perfect rasteriser fed the projector's
  tiles would undercount.

Cycle-exactness therefore requires porting three more subsystems, in order: the
`$0180` plotted-set gate (so the polygon set matches), the self-modifying
`process_line`/`span_fill` rasteriser (edge tables + cycle count per polygon), and the
`plot_object` vertex transform/projection (so object polygon geometry is native). All
were RE'd to the block level here; none is curve-fit into `render_cost`.

## Achieved accuracy (vs py65 exact plot_world cycles)

| term | model | accuracy vs py65 |
| --- | --- | --- |
| `N_examine` (count) | `_scan_visible` port | **exact** (0 mismatches) |
| occlusion `$3E80` bitmap | `_occlusion_visible` port | **exact** (0 mismatches) |
| examine cost | `N_examine * 1737` | median 5.6%, max 14% |
| object base floor | `_inview_object_base` | floor, ratio 0.16-0.92 (never overshoots) |
| total frames | + area-proxy fill + object base | median err 41% (was 53%) |

The **object-base term (c)** (`_inview_object_base`, `C_VERTEX`=2200, `C_PREP_CALL`=625,
`SECTIONS`=2, per-type `(verts,polys)` model sizes) adds `plot_object`'s per-object
vertex-trig + `prepare_polygon` floor over the plotted object-tiles' stacks. Because the
distance-dependent object `span_fill` is unmodelled, the term is a strict floor: it moves
the previously-zero object cost toward the truth and never overshoots (verified
`test_object_base_never_overshoots_and_is_present`). The remaining residual is the
multi-band terrain rasteriser and the object `span_fill` fill; the tile-selection,
examine-count and occlusion foundations for porting them are exact and in place. Fill
constants stay env-overridable (`RENDER_*`).

## Transfer settle: the full fixed base (tune + $357D foreground)

The live transfer viewpoint-replot settle ($357D) is 259-460 frames
(ls0042 [338,305,435,460], ls0335 [259,333,371]); isolated py65 `plot_world` is
1.8-79 frames. `viewpoint_replot_frames` models it as

    viewpoint_replot_frames = TUNE_TRANSFER_FRAMES + SETTLE_FIXED_FRAMES
                              + REPLOT_PASSES * render_cost

The `2*plot_world` term (REPLOT_PASSES=2 at $35C3/$35C6) is only 4-158 frames -- a
~10x under-prediction of the live settle. The missing frames are two fixed,
scene-general terms `render_cost` neither can nor should include, both ROM-derived.

### TUNE_TRANSFER_FRAMES = 96 (the #$19 transfer tune)

`play_landscape_loop` ends at `wait_for_end_of_tune` ($35D5): a tight
`update_sound`/`BPL $0CE7` spin that blocks until the tune started at $1B82
(`start_tune $888F`, tune number #$19 in $0CE7) sets its bit7. `play_tune` ($34DE)
walks the note table at **$AB50 + tune_number** ($AB69 for #$19): a byte >=$C8 sets the
note length `$0C70 = (byte-$C8)*4`, a byte <$C8 is a note that holds `$0C70` frames in
the `$0CDF` countdown, $FF ends the tune. `$0CDF` is decremented once per frame by the
raster IRQ (`$9630 DEC $0CDF`, floored at 0). Summing the note holds gives **96 frames**
for tune #$19 -- byte-for-byte the same duration as the #$0 hyperspace tune ($AB50,
`actioncost.TUNE_FRAMES = 96`). This is a fixed ROM constant, not a fit
(`test_transfer_tune_is_96_frames` decodes both tunes to 96).

### SETTLE_FIXED_FRAMES ~ 176 (the other once-per-settle $357D foreground)

Before the two `plot_world` passes, `play_landscape_loop` runs four fixed foreground
routines `render_cost` omits, py65 foreground cycle-counted (`/19656`):

| routine | ROM | ls42 | ls335 |
| --- | --- | --- | --- |
| occlusion raytrace `populate_tile_visibility_bit_table` | $245B | 110f | 63f |
| grid angle/hypotenuse pass | $3700 | 82f | 82f |
| `fill_screen_with_background` | $1090 | ~1f | ~1f |
| `plot_status_bar` | $98B2 | 7f | 7f |
| **sum** | | **199f** | **152f** |

Occlusion cost is scene-dependent (terrain complexity); the mean ~176f is modelled as a
constant (env `SETTLE_FIXED_FRAMES`). Raster-IRQ steal (~10-25%) on the whole settle is
folded into this and the tune base.

### Achieved settle accuracy

`settle = 96 + 176 + 2*render_cost` vs the seven live transfers (sweep-order pairing):

| landscape | live settles | predicted | median abs error |
| --- | --- | --- | --- |
| ls42 | 305,338,435,460 | 291-329 | ~9% |
| ls335 | 259,333,371 | 288-358 | ~9% |

Median **~9%**, max ~29% (was ~22%, and ~10x / ~90% under before the settle base). The
object-base term (c) closes most of the object-view gap; the residual is the documented
`render_cost` terrain fill-proxy swing plus the object `span_fill` fill and the
single-constant occlusion approximation.

A u-turn (EOR $80 bearing flip) scrolls 0 frames (instant) and is not a viewpoint
replot.
