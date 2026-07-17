# plot_world ($2625) render-projection frame-cost model

Reverse-engineered spec for the ROM's terrain rasteriser, so a future exact
per-notch redraw cost can be ported. All addresses are ROM ($ hex). PAL frame =
19656 cycles.

## What plot_world does

`plot_world` ($2625) is an equirectangular terrain rasteriser. It walks the
32x32 tile grid row-by-row, furthest to nearest:

- `plot_rows_in_front_of_observer_loop` ($26DE) iterates the row counter `$0026`
  from 31 down to 0 (<=32 rows).
- Per row, `find_visible_extent_of_row_of_tiles` ($27D7) finds the on-screen
  horizontal tile span via `check_if_tile_is_on_screen_and_calculate_screen_coordinates`
  ($2845).
- Each visible tile is filled as 1-2 filled polygons: `plot_tile` ($2A24) ->
  `plot_polygon` / `prepare_polygon` ($2D6C) / `span_fill` ($22AA) /
  `plot_middle_of_row` ($23D0).

Pitch is read from `objects_v_angle`, shifted into `$00B0/$00B1` at $262B.

## Cost decomposition

`plot_world` cost is the sum of three terms:

- (a) Per-EXAMINED-tile trig floor, ~700-1100 cyc/tile: `calculate_angle`
  ($9287) + `calculate_hypotenuse` ($937F) + `calculate_object_relative_vertical_angle`
  ($933D). Over the examined tiles this is a ~30-frame floor.
- (b) Terrain projected-AREA fill term: ~7 cyc per buffer byte + ~60 cyc per
  scanline, i.e. `sum over visible tiles of (60*H + 1.75*H*W)` where H, W are the
  tile's projected screen height/width.
- (c) Minor object term (~5%): `plot_object` ($8533/$8579), per-object vertex
  rotation plus small polygon fills.

Formula:

    plot_world_frames ~= (BASE
                          + N_examine * C_examine
                          + sum_visible_tiles(60*H + 1.75*H*W)
                          + objects) / 19656

## FOV and pitch clipping

Horizontal FOV is ~20 angle-units: `$0020 = bearing - 10` at $27C2. The on-screen
test ($2845) is purely horizontal. Vertical/pitch clipping happens IN the fill:
tiles whose projected screen-y falls outside the ~240-line band fill zero
scanlines. Pitch (`objects_v_angle` shifted into `$00B0/$00B1` at $262B) slides
terrain into and out of the fillable band and changes projected tile heights.
This is the scene-dependent, pitch-driven cost swing (30 -> 140 frames).

## Key consequence for the cost model

The redraw cost is terrain-projected-AREA dominated, NOT object-polygon-edge-count
dominated. The old `visible_edges` edge-count proxy (`sentinel/actioncost.py`)
modeled only the ~5% object term (c). An exact per-notch redraw model requires
porting the ROM's tile projection ($2845 / $9287 / $937F + the vertical-band clip)
to compute per-tile H, W. This is future work.

## Measured ground truth

Exact, un-wrapped frame counts from the live game (silent $9630 checkpoint):

- Transfer viewpoint-replot settle = 2x plot_world + tune, scene-dependent
  ~306-420 frames (landscape 0000 ~420, 0042 ~306) vs the old constant 47.
- A u-turn (EOR $80 bearing flip) scrolls 0 frames (instant; was charged ~34).
- Bearing pan ~47 frames/notch; pitch pan ~58+ frames/notch.
