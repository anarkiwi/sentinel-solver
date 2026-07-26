# Landability candidates (`sentinel/landtable.py`)

A sound **superset filter** over the keyboard-aim lattice: given an observer and a target
tile, it returns the lattice rays that *can* land there, so a per-tile build query marches
thousands of rays instead of the whole heading cone. It is the path behind
`los.landable_view_targeted` (every per-tile query, all three lattices) and
`astar_player.AStarPlayer._coarse_landable` (the whole-board landset).

## Why a precomputation is possible

`los._get_object_details` ($1ECC) seeds **every** ray at `px_frac=py_frac=0`,
`px_sub=py_sub=0x80` — the horizontal seed is always exactly the eye tile's centre — and
`prepare_vector_from_player_sights` ($1C10) reads neither state nor slot. `_add_vector`
($1CBB) then adds the sign-extended 16-bit component into the 24-bit (whole:sub:frac)
accumulator, so at sub-step `i`:

```
DX_i  = floor((0x8000 + i*vx) / 65536)             tile offset from the eye tile
DY_i  = floor((0x8000 + i*vy) / 65536)
z16_i = eye_z*256 + obj_z_frac + floor(i*vz / 256)  the $003B:$0038 compare pair
```

A ray's **track** — which tile offsets it steps through and at what ray z — is therefore a
pure function of its aim. Terrain never alters the track; it only decides at which step the
march STOPS. Two consequences:

* `obj_z_frac` enters as a **pure additive offset** (`pz_frac` starts 0, so the seed
  fraction is a whole multiple of 256 in the accumulator). It is never a table axis — it is
  a term in the query threshold. Measured in real solves it takes exactly two values, `$60`
  and `$E0`, but the closed form makes that irrelevant.
* `eye_z` is likewise an additive offset, and `DX`/`DY` are position-independent, so one
  table serves every observer position and height.

`test_landtable.py::test_closed_form_track_matches_add_vector` pins all three closed forms
against the ROM `_add_vector` accumulation, for several `(eye_z, z_frac)`.

## The necessary condition

check_flat_tile ($1D0D) lands only when `D = surface16 - z16` is in `[0, $80)` (`$0079`
compared against `$000C`, which the object path can tighten to `$10`), where
`surface16 = tile_z*256 + $0079`. `D < 0` keeps marching, `D >= $80` blocks. `|vz| <= 4095`
so `z16` moves at most 16 per sub-step. Hence, for the ray to land on a cell it must, at
some sub-step **inside that cell**:

* have entered above the band — `z_entry > surface16 - $80` (otherwise it is blocked at
  entry), and
* reach the surface — `min z16 over the cell <= surface16`.

For a climbing ray (`vz > 0`) `z16` only leaves the band, so the entry sub-step itself must
satisfy both. This is what `crossing_mask` tests, and it is a superset because terrain can
only terminate the march *earlier* — it can never make a ray visit a cell its track misses,
nor change `z16_i`.

A **flat-terrain** table would not be sound (lower terrain lets a ray travel past where it
otherwise stops); keying on the crossing height is what makes the condition hold.

The query needs bounds, not the exact surface: `surface_bounds` returns `(lo, hi)` — exact
for bare terrain, and over the whole object stack otherwise (the platform form $1E30
`+$20` on top, the boulder form $1E5A `-$60` at the bottom). A wider bracket only widens
the candidate set.

Two exact shortcuts fall out (`never_lands`): the observer's own tile never lands ($1D32
keeps marching) and a **sloping** tile never lands (check_sloping_tile $1D46 only loops or
blocks).

### Wrap safety

The ROM compares z as bytes, so a ray more than `0xFF80` z-units *above* a surface aliases
onto "equal". Reaching that takes ~4000 sub-steps of ascent, which only the full-band
lattice can do (measured max `zrel`: 472 on the $F5 plane, 95976 on the band; 168,880 band
rays exceed the alias distance at a non-origin cell). Such visits are kept
**unconditionally** — as wildcards in the table, and by the `wrap_z` branch in
`crossing_mask` — so the superset holds without arguing about whether the alias is
reachable on a real board. In the arc-narrowed query this costs ~490 extra rays.

## Measurements

Lattices are exactly `los._landable_sweep`'s: the **$F5 plane** (32 h x 64 cx x 128 cy =
262,144 rays) and the **full band** (x 27 pitches, 3,538,944 rays), `max_steps` 6000.

Track census (`python -m sentinel.landtable`), 24 threads, 6.7 s for both lattices:

| | plane | band |
|---|---|---|
| distinct cells per ray | 43.7 | 43.3 |
| (ray, cell) visits | 11.5 M | 124.6 M |
| wildcard visits | 0 | 1.81 M |
| candidates per cell, `T <= 0` | mean 3035, p50 2138, p90 4861 | mean 20103, p50 12982, p90 29780 |
| candidates per cell, `T <= -4` | mean 282 | mean 13084 |

Query thresholds in real solves (landscapes 0, 42, 335 — 3775 targeted queries):
`T = surface16 - eye_z*256 - obj_z_frac` spans **-1120 to +1568**, i.e. whole-unit buckets
-5..+6; eye heights 5..11; `|dx|`, `|dy|` up to 30.

### Why the K-pruned table is the wrong representation (measured, then deleted)

Storing the first `K` candidates per (cell, T-bucket) in lattice order gives a clean
structural guarantee: a landing candidate at or below the row's **frontier** (its last
stored index) is provably the sweep's first hit, and an untruncated row with no landing
candidate is a proven "no view"; anything else defers to the exact path, so a too-small `K`
can only cost time. But the measured **rank of the first hit** within a row is mean 1083,
p50 604, p90 3067 — most candidates ahead of it are occluded on the real board. So:

| K | share of landing queries the row decides |
|---|---|
| 8 | 0 % |
| 64 | 4.8 % |
| 256 | 28.6 % |

`K = 64` costs 46–54 MB per lattice; a `K` large enough to decide ~99 % (~4096) would be
terabytes. On real queries the K=64 table is no faster than the exact path (band: 15.1 vs
15.5 ms/query, 35/40 fallbacks). The representation (`LandTable`, its builder and its
`out/landtable_*.npz` cache) was removed once the filter was wired in; only the track
census that sized it remains, as `python -m sentinel.landtable`.

### What works: the same condition as a closed-form filter

`crossing_mask` evaluates the condition in O(1) per ray (the cell's sub-step interval is
closed form), so it needs **no storage at all** and composes with the existing heading-arc
bisection (`los._tile_arc_indices`), which is itself a proven superset. On the same real
queries:

| | rays marched (band) | rays marched (plane) |
|---|---|---|
| arc only | mean 77393, p50 50367 | mean 5648, p50 3584 |
| arc + crossing | mean 18994, **p50 2010** | mean 2267, p50 578 |

| query cost | band | plane |
|---|---|---|
| arc only | 15.46 ms | 1.24 ms |
| `landable_view` (what `landable_view_targeted` runs) | **3.20 ms** | **0.92 ms** |

Whole-solve, with the filter in front of the targeted cone (identical plans: 23 actions /
6240 frames on ls0, 35 actions / 9810 frames on ls42):

| landscape | plan time before | after |
|---|---|---|
| 0 | 2.7 s | 2.2 s |
| 42 | 27.0 s | 10.9 s |

Mean marched rays per query over the ls42 solve: **6131** (vs ~77 k), with 1638 of
2817 queries answered as a *proven* "no view" — the case that costs a full cone today.

The remaining expensive case is an **adjacent** cell: its heading arc is huge and a ray
dwells long enough inside it to cross almost any surface height, so the filter keeps
~70 % of the arc (worst observed 623 k of 886 k rays).

## The whole-board set (`_coarse_landable`)

`AStarPlayer._coarse_landable` / `_landable` answer over the **landset lattice**: the same
`hgrid` and `los._V_PRIORITY` as the full band with the cursor subsampled 2:1
(`_COARSE_CX`, `_COARSE_CY` — `landtable.COARSE_CX`/`_CY`, subsets of
`los.CURSOR_CX`/`CURSOR_CY`), i.e.
884,736 rays whose directions are a strict subset of the band's. `landtable.lattice(coarse=True)`
is that lattice, pinned against `astar_player`'s constants and index-for-index against the
band's ray vectors by `test_coarse_lattice_is_the_landset_lattice`.

A per-tile candidate list is the wrong shape for a whole-board answer — one ray is a
candidate for several tiles, so summing 961 per-cell queries re-marches rays. Measured
over 7 stances the per-cell sum is 0.2–0.5x of the sweep (245 k–426 k rays), a win but a
weak one. Inverting it is much better: **`stop_cells` walks each ray's track against the
real surface map and names the tile(s) it can stop in**, which partitions the lattice by
landing tile, so `landable_set` marches each ray at most once — and only while its tile is
still unresolved. Rays that cross no surface band before leaving the board (77–88 % of the
lattice, mostly climbing rays) are never marched at all.

| stance | tiles | full sweep | `landable_set` | rays marched |
|---|---|---|---|---|
| ls0 | 49 | 83.8 ms / 884,736 | 25.6 ms | 152,025 |
| ls42 | 23 | 72.7 ms / 884,736 | 23.5 ms | 135,979 |
| ls335 | 7 | 77.2 ms / 884,736 | 23.6 ms | 120,588 |
| ls0 stacked+transferred | 194 | 90.5 ms / 884,736 | 28.7 ms | 160,224 |
| 3 recorded solve states | 49–98 | 56.7–97.7 ms | 17.8–29.7 ms | 95,984–152,025 |

So **5.5–9x fewer rays and ~3x less wall time**, and the set is exact — `landable_set`
returns `_coarse_landable`'s set tile for tile on every stance tested, ring included (the
whole board is one loop; the ring is not excluded anywhere in the whole-board path).
Per-tile queries on the same lattice (what `_landable` calls) go from 21,660 to 4,372 rays
(p50 14,698 -> 776) and 2.67 -> 0.55 ms.

Whole solves with **both** paths wired in, plans byte-identical (ls0: 23 actions / 6240
frames, energy 6; ls42: 35 actions / 9810 frames), warm runs of
`python -m sentinel.astar_player N --quiet`:

| landscape | wall before | after | rays marched before | after |
|---|---|---|---|---|
| 0 | 2.8 s | 1.4 s | 19.9 M | 3.8 M |
| 42 | 23.1 s | 5.0 s | 144.3 M | 25.8 M |

The partition's soundness rests on one extra step beyond the per-tile condition: the walk
may only stop at a cell the ray **provably** stops in (`zmin <= surface_lo`, so it reaches
even the lowest surface the cell can present). Sloping tiles never land AND are walked
through (their facet test can pass the ray on); the observer's own tile never lands and
never terminates the walk. Where the walk cannot decide — an object stack whose `lo < hi`,
or an 8-bit alias risk — the ray is marked multi-candidate and always marched.
`test_stop_cell_partition_holds_every_landing` pins the invariant directly: a landing ray
either names its tile or is multi-candidate.

## Validation

`sentinel/tests/test_landtable.py` asserts the property everything rests on: over several
boards, an object-tile/raised-eye stance (boulder+boulder+robot, then transfer) and an
`eye_z` override, **every ray the full-lattice sweep lands on a tile is in that tile's
candidate set** — for all three lattices, including below-eye and object tiles. Outer-ring tiles
are asserted like any other: soundness here is geometric, not read off a flat board, so the
`terrain.tile_byte` edge wrap that makes flat-board landable *sets* differ at the ring does
not enter. It also pins the closed forms against `_add_vector`, the never-lands shortcuts,
the `(lo, hi)` surface bracket, and `landable_view` against each lattice's own full sweep
(the aim of the first landing ray) on landable, non-landable, object and ring tiles.
`test_landable.py` pins the same equality through `los.landable_view_targeted` for every
tile of the band and coarse lattices, and through `_view_for` for the $F5 plane.

## Unproven corners

* **Alias landings** (the 8-bit z compare wrapping) are *handled* but not *proven
  unreachable* — the wildcard rule makes the answer sound either way.
* The filter inherits `los._tile_arc_indices`'s superset claim; the harness would catch a
  violation, but it is not independently proved here.
* Both are keyed to `max_steps = 6000`; a caller marching further must rebuild/re-query
  with the same cap (it is an explicit argument).
