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
frame-exact needs two more ROM subsystems, which this pass measures but does not port
(no curve-fitting of the gap):

- **Multi-band terrain rasteriser.** Each tile polygon is clipped to the current
  vertical buffer band ($0051/$0052) and rasterised by `process_line` ($3002), a
  self-modifying steep/shallow x inside/outside Bresenham edge tracer, then filled by
  `span_fill`. The filled scanline count and per-row byte count are NOT a clean
  function of the projected corner H/W: instrumentation shows the ROM's polygon rows
  ($0004/$0006) diverge from the projected y-extent because the edge tracer, not a
  linear map, sets them, and edge/`process_line` overhead (~hundreds of cyc/row)
  dominates thin polygons. Even the pure-terrain views (0% object) span ratio 0.38-2.26.
- **Object renderer.** `plot_stack_of_objects` ($21AE) is a distinct per-object
  polygon renderer; a single object tile costs 68k-213k cyc (vs a few k for terrain),
  so object-heavy views are the largest single error source (up to 42% of plot_world).

## Achieved accuracy (vs py65 exact plot_world cycles)

| term | model | accuracy vs py65 |
| --- | --- | --- |
| `N_examine` (count) | `_scan_visible` port | **exact** (0 mismatches) |
| occlusion `$3E80` bitmap | `_occlusion_visible` port | **exact** (0 mismatches) |
| examine cost | `N_examine * 1737` | median 5.6%, max 14% |
| total frames | + area-proxy fill | ratio 0.32-2.26, median 0.52 |

The two named residual subsystems (multi-band rasteriser, object renderer) plus the
per-call examine-trig spread are what stand between this and a few-% total. The
tile-selection, examine-count and occlusion foundations for porting them are exact and
in place. Fill constants stay env-overridable (`RENDER_*`).

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
| ls42 | 305,338,435,460 | 315-352 | ~15% |
| ls335 | 259,333,371 | 264-317 | ~25% |

Median ~22%, max ~29% (was ~10x / ~90% under). The residual is the documented
`render_cost` fill-proxy swing (ratio 0.32-2.26) plus the single-constant occlusion
approximation; the tune + fixed-foreground base is the win.

A u-turn (EOR $80 bearing flip) scrolls 0 frames (instant) and is not a viewpoint
replot.
