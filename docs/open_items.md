# Open items

Everything known to be wrong or unproven, each stated as: what is wrong, the measurement that
shows it, what would resolve it. Anything not here is either correct or out of scope
([architecture.md](architecture.md), "Not modelled"); a mechanism once understood moves there
and stops being an open item.

## 1. A per-tile candidate generator only pays once the eye is up

**Wrong.** The phase player's candidate generators now ask per tile
([candidate generators](architecture.md#the-landability-filter-landtablepy)) instead of reading
the whole pitch-band dict, but that is cheaper only while the cheap filter prunes. Early, with
a low eye, it prunes nothing and the sweep is still the better answer, so `BAND_SWEEP_TILES`
buys one — an unresolved cost, not an unresolved decision.

**Measured — the change.** `_climb_candidates`, `_establish`'s supply hop and `_barren` ask
`_Views.band_ordered`; `_barren` short-circuits on the first available action and reads no
whole dict on a tick. Bit-identical on all eight suite boards and on the two relocation wins
(ls7414 59, ls8589 46). Solo, one numba thread, idle host:

| board | before | after |
|---|---|---|
| ls42 | 180.9 s | 157.4 s |
| ls335 | 78.8 s | 60.5 s |
| ls373 | 85.5 s | 69.2 s |
| ls7545 (dies in 17) | 37.0 s | 42.9 s |
| ls8271 (dies in 13) | 48.3 s | 53.2 s |
| ls9090 (dies in 13) | 23.4 s | 24.2 s |

Boards that climb get faster and boards that die in their first twenty actions get slower, and
those are the same regime split: a climb candidate needs `base >= my_eye - 2.275`, so at entry
`gain > 0` holds for ~930 of the ~950 stackable tiles and nothing is pruned. On the 24-board
sample of [12](#12-the-hardest-boards-are-unsolved-and-mostly-unfinished) the net is 1-2 more
boards reaching a verdict, both of them wins.

**Measured — why batching does not rescue the entry state.** At the ls42/ls335 entry the union
of the asked tiles' candidate ray sets is 721,275 / 495,372 rays against the 696,699 / 765,328
the whole sweep marches, so one batched multi-tile march is not cheaper than the sweep. The
per-tile path is linear and clean (3.7-4.4 ms/tile over 25..200 tiles on ls42/ls335/ls8644)
against 0.83-1.08 s/sweep, hence the 205-260-tile crossover `BAND_SWEEP_TILES` = 200 sits
under.

**Measured — `_settle_tie` does not re-sweep.** The allegation that a tie buys a sweep per fork
is false: `_VIEW_CACHE` keys on `projector.scene_key`, so a fresh clone of an unchanged board
hits the memo. Over a full ls110 solve, 6 ties and 22 forks, 45 sweeps are built and **0** of
them at a tie's own entry scene key.

**Measured — `player._climb_scan` is genuinely whole-set.** It ranks on each candidate's
`_aim_frames`, so every candidate needs its view, and its visibility-free gates leave 946
(ls42) and 932 (ls335) tiles — ~6 s of targeted marches against a 1.2-1.4 s sweep. Per tile is
4-5x *worse* there, so it keeps the sweep.

**Resolves.** A cheap sound prune that survives a low eye, which the gain test is not. Then
`BAND_SWEEP_TILES` and the sweep path go away.

## 2. Live frame accounting leaks across a tie

**Wrong.** Frame counts from a live run are not trustworthy across a tie-breaking rollout.

**Measured.** Two frozen live ls42 runs take an identical action sequence yet measure different
frames per step on the same creates and transfers; the same pair under the greedy player is
bit-identical. The monitor socket drops during the ~25 s think gap a tie opens and
`core.reconnect`/`robust`/`drop_guard` reconnect with no frame reconciliation, so frames that
elapsed while the CPU ran unattended are unattributed. The live *win* is unaffected (`$0CDE`
bit 6); `driver/test_live_determinism.py` pins the player to greedy for this reason.

**Resolves.** Holding the connection alive, or the machine verifiably halted, across an
arbitrarily long think gap.

## 3. The gaze forecast assumes rotation never stalls

**Wrong.** `playerbase._cone_onset` projects the rotation step and cooldown cadence as if the
enemy rotated every period, but a busy enemy returns before the `$17F9` rotate and its cone
holds ([the enemy ladder](architecture.md#the-enemy)).

**Measured.** Waiting under a cone is non-terminating: one live ls110 run spent a long run of
consecutive waits and took drains throughout. The error has both signs — a body under a busy
cone is exposed *longer* than forecast, and tiles ahead of that arc become safe *later*.

**Resolves.** Modelling the stall inside `_cone_onset` and re-running the three live gates — it
shortens every window, so the gates are the test. Price at the same time that an abandoned
object **pins** a gaze, since a sentry with something to drain stops rotating. Binds the
reactive player only; the phase player simulates the span (`_drained_over`).

## 4. Per-step frame drift, and the unattributed create/absorb settle split

**Wrong.** Charged frames drift against measured frames per step, and the settle side is two
constants merged into one.

**Measured.** A winning live ls42 run under-charges over its 36 steps, reproducibly run to run.
Over 15 runs a create's settle measures 96.00 f against an absorb's 85.87 f while
`actioncost.SETTLE` charges both 93.75 f — the separation pinned by
`test_absorb_and_create_measured_settles_are_separated`, the merge a strict xfail in
`test_settle_accuracy.py`. The only split `$2099` is known to have is the meanie flag, so the
create/absorb difference has no ROM path yet. The aim side is a separate over-charge, dominated
by large pans.

**Resolves.** The ROM path that lengthens a create's settle, and splitting `actioncost.SETTLE`
on it rather than on the fitted difference.

## 5. One object vertex angle is ten units out

**Wrong.** `rendercost` reproduces the ROM's own loop counts exactly on 14 of the 15 golden
views. The fifteenth, 2024,16,0, runs `$2FA1` twice more than the ROM, and the cause is not
the fill: one object vertex's 16-bit horizontal angle is 10 units low, which moves its
`screen_x` byte across a multiple of 32.

**Measured.** Tracing every `$2EE4` of that view, four edges (polygons 3, 12 and 13, one
object) disagree on their input `$0039` alone: 76 on the machine, 75 in the model. `$2D93`
builds that byte as `(($A800,X + $0011) : ($0BA0,X + $0029)) >> 5`, and the vertex's pair is
`$FF80` on the machine against `$FF76` in the model — every other vertex of those polygons is
byte-identical. The two extra `$2FA1` laps are what a one-column-wider shallow line costs.

**And the exact backend is itself dear on a live state.** One `$1FFC` strip replot caught on
the machine at `$1F9F` (ls9795, slot 3, the ROM's own span 37/3) takes **22 whole frames** to
the next `$1289`. On that captured image the proxy charges 387392 cycles, 1.12x, but
`RENDER_COST_BACKEND=py65` charges **534090**, 1.54x — which the machine cannot have spent in
22 PAL frames (24277 a frame against 19656). Calibrating the proxy against that backend
inherits the error. Fixture `strip_replot`, measured from
[8](#8-the-enemy-clock-commits-consider_enemy_states-core-writes-early).

**Resolves.** Stateful emulation of the whole `plot_world` fill sequence in render order, the
interleaved object polygons included. Short of that the only honest levers are
`projector.PER_SCANLINE`/`PER_PIXEL` — and first, the `$2993`/`$245B` context
`oracle.update_object_cost` builds, since that is what "exact" is being measured against.

**The suspect is `$0078`.** `divide_and_arctan`'s `$0E2C BIT $78` decides whether `$0E1F`
interpolates between two `$3B00`/`$3C01` table entries (`$0E35`/`$0E50`) or returns the raw
entry (`$0E32 JMP $0E74`), so a stale bit 6 or 7 changes the arctan's **value**, not just the
~90 cycles already recorded below. `relative._divide_and_arctan` starts `$0078` at 0 and
cannot know better without whole-zero-page emulation, and `objectcost`'s vertex transform runs
that chain per vertex.

**Closed — the storing loops' second entry.** `$2EE4`'s narrow steep and shallow DDAs each have
an entry for a line starting above the inner area (`$2FB6`, `$2FDC`) that walks those rows
without storing. Neither rejoins at the store: `$2FD9` lands on `$2F67`, so `DEY` then `$2F58`
accumulate before the first `$2F5F`, and `$2FFF` lands on `$2FB0`, so `DEY` then `$2FA1` step
before the first `$2FAD`. The model stored first, which put every column one row late and wrote
one row below the polygon's bottom. That was the whole residual on 335,0,0 (-2 filled bytes)
and 42,160,240 (-6), and the shallow entry was not modelled at all.

**Closed — the harness ran `$2993` before `$245B`.** `populate_tile_visibility_bit_table
$245B` calls `get_object_details $1ECC` per ray, which zeroes `$0034`-`$0036`, and `$2597`
accumulates the march fraction into `$0035`; nothing rewrites `$0036`. So the raytrace leaves
the buffer variables as march state (`$0035` = the last ray's fraction, `$0036` = 0), and the
ROM re-runs `$2993` before every `plot_world` — `$35C0 JSR $1090` on the play redraw, `$994F`
on a pan notch, `$1FE5 JSR $29C7` on a strip. The oracle harness called `$2993` first and
`$245B` second, so its `plot_world` ran with `$0036` = 0, where every row's left edge reads as
right of the buffer: the polygons before the first `$0010` toggle plotted no bytes and clipped,
re-paritying `$0010` for the rest of the pass. Swapping the two calls is the whole residual on
0,136,248 and 2024,184,244; it moves three golden views' cycles (0,136,248 +5350,
2024,184,244 +13739, 42,192,0 -161) and neither `golden_object_cost.json` nor
`golden_pan_cost.json` by a byte.

**There is no unread-before-written `$0B40` in the ROM.** The image has exactly nine accesses:
three writes (`$2DC0` in the wide vertex loop, `$2E58`/`$2E5B` in `$2E56`) and six reads
(`$2E3F`, `$2E42`, `$3011`, `$3014`, `$304E`, `$30AB`). Every read indexes Y or X — the current
line's own start and end vertex — and both entry paths to them write both: `$2E4F` only from a
wide polygon, where `$2D93` has written every vertex, and `$2E5E` only through `$2E56`, which
zeroes both endpoints first. So the ROM is self-consistent and a "stale `$0B40`" cannot be the
mechanism. `$2D93` itself is exact: fuzzed 4000 random (`$0BA0`, `$A800`, `$0011`, `$0029`)
against the real routine, **0 mismatches**. `$2F17` is not the mechanism either.

Frame cost is 0.94-1.00x (median 0.975, mean absolute error 2.5%).

**Also not derived.**

- **`$0078` carries a bit between calls.** `divide_and_arctan`'s `$0E30 BVS` reads bit 6 of
  `$0078`, which the *previous* call left there; `relative._divide_and_arctan` starts it at 0
  and cannot know better without whole-zero-page emulation. Bisected to that single byte
  against the live machine. It no longer reaches the examine — the play path's `$37F2` calls
  no trig at all — but `objectcost`'s per-vertex transform still runs that chain.
- **`$3030`'s two overflow guards** are modelled as a loop rather than the ROM's re-entry into
  `$3022`; equivalent in every case checked, but not the same control flow.
- **`$27CE`, the observer's own tile.** `$2793` forces its four corners off the edges of the
  screen and plots it through `plot_checkerboard_tile`; the model charges the branch and the
  `$279B` examine (at `$001D + 1`, which `$2797 INC $0026` makes it) but not that polygon,
  because two of its corners inherit a screen_y low byte from whichever row last used the other
  `$0005` bank. It is never reached on any of the 15 golden views, so it is unmeasured rather
  than known-small.
- **The object term without the game image.** `objectcost` needs the model geometry, so a
  checkout without `out/sentinel_stage2.bin` (or without numba) falls back to
  `_inview_object_base`'s floor and under-charges every object
  ([render cost](architecture.md#render-cost-projectorpy-pancostpy-rendercost_py65py)).

**Resolves.** Emulating `$0078` across a whole `$8475` object transform, which is the one byte
between the model's arctan and the machine's.

## 6. The py65 exact backend cannot price another slot's view

**Wrong.** `RENDER_COST_BACKEND=py65` falls back to the proxy on the whole transfer path.

**Measured — the transfer path.** `projector._exact_render_cost` returns `None` for any
`observer != state.player`, and a transfer settle is always priced from the post-transfer slot
(`playerbase._settle_eye`), which is never the player's own at pricing time.

**Closed — the play machine does not run `$2845` at all.** `$283D BIT $9AF6 / BPL $2845 /
JMP $37F2` picks the examine and `$9AF6` is `$80` for the whole play loop (`$3577`; `$9A51` is
its only writer, and `$117E` is the only other call, with 0, for generation and the preview).
The play examine is `$37F2`, a read of `$3700`'s per-position projection tables less the camera
reference — no trig, no object-stack walk, 193..251 cycles against `$2845`'s ~1750. The oracle
harness generated its goldens with `$9AF6` = 0, so `render_cost` could be exact against the
golden and 1.26x wrong in play. `oracle.prepare_render_context` now sets `$80` and runs
`$3700` in the ROM's own order (`$35BA`-`$35C0`), and `passcost.TAB_*`/`EXAM_ENTRY_*` price
`$37F2` branch by branch. Every golden view fell 0.23-0.85x; not one of the eight fill loop
counts, nor an examine or plotted-tile count, moved by a byte — `$3700` computes exactly what
`$2845` computes, once per position instead of once per grid point.

**Closed — the strip's window is three bytes.** `$29C7` puts the halving's carry in `$0028`,
`$29EA`-`$29F2` leave `$0029`, and `$1FCC`/`$1FCF` leave `$001F`; the model held all three at 0
because `$2993` zeroes them for every full-screen mode. With them modelled the strip examines
111 grid points against the ROM's 111 on ls0042, 85 against 85 on ls0335 and 52 against 53 on
ls9795, and `render_cost` is 0.98-0.99x the ROM's own `$2625` (was ~0.79x with only `$9AF6`
fixed, and 1.5x dear before).

**Closed — `$2793` examines the observer tile one row nearer.** `$2797 INC $0026` runs before
`$279B JSR $283D`, so the observer's own tile is probed at `$001D + 1`, not `$001D`. All three
`$276F` branches fall into it. The model probed the observer row itself, which cost one wrong
grid point per pass on every view.

**Closed — `$1F9F`'s own line is priced, and it is exact.** `strip_replot_frames` now covers
the whole of `$1FA4..$1F9E`, every term counted from its own instruction sequence:

| term | price | driven by |
|---|---|---|
| `clear_strip $2211` | `2232 * span + 1292` under 32 columns (24 rows x 8 bytes a column at 11 each, `$2247`) | `$211B`, the **uncapped** span |
| `$29C7` | 79, a straight line | once per chunk |
| `$1FFC JSR $2625` | `render_cost` at the `$1FC2` camera through the `$29C7` window | `$0C69`, capped at 20 |
| `$9730` | `4340 + 135 * splits + wrap`, 4880..5016 a call | `$211B` calls, `$0095` for the price |
| `$1FA4..$1FBF`, `$1FC2..$201C`, `$202C..$1F9E` | 33 / 120 a chunk / 106 + 32 a column + 36 | counted branch by branch |

`$2105` caps `$0C69` at 20 but leaves the whole span in `$0C6A`/`$211B`, so an object wider
than 20 columns replots in two chunks (`$201E`/`$2021` re-enter `$1FC2`) while the clear and
the flush run **once** over the whole span — which is why the clear was `$211B` and not
`$0C69`-driven, and `object_screen_span` now returns both. `$9730` copies 24 of the 25 `$3A00`
screen banks, skipping bank `$0095 - 1`; a bank whose `$3A40` entry is nonzero straddles two
buffer pages and buys a second `$9888`, so its price follows from `$0095` alone. Checked
against the real `$1F9F` on 84 (board, facing, slot, `$0095`) combinations spanning one and
two chunks and spans 1..25: **0 mismatches**, cycle for cycle
(`test_the_strip_replot_line_is_the_roms_own_1fa4`). `strip_replot_frames` as a whole is now
0.98-1.00x the ROM's own `$1F9F`, and `driver.instrument --frames 3000 --follow` reports
ls9795 **67 -> 64**, ls0335 **57 -> 57**, ls0042 **0 -> 0**.

**Not a threat to the render golden.** `golden_render_cost.json` comes from `_measure_plot_world`,
which runs `$2625` directly at an explicit (h, v) with `$2993`/`$245B`/`$3700` outside the
counted window; it asks what plot_world costs for a board and a view. The one assumption it
shares with the strip path is `$0C48` = 0, matched by `projector._ROW_HINT` and bounded at
±3% above.

## 7. The driver's wall-clock timeouts are the residual load sensitivity

**Wrong.** `kbd_aim` still holds host clocks in an otherwise machine-clocked driver, and a
timeout there is retried rather than raised.

**Measured.** On an idle host `driver/test_live_determinism.py` passes (2/2 serial, two full
ls42 runs frame-identical); under a saturated host (`pytest -n auto` plus two VICE containers)
it fails on differing step counts — a monitor round-trip is orders of magnitude slower at
real-time pace than halted, so contention pushes a wait past its timeout and the aim re-drives.
The pan cycle is now machine-clocked and `_RU_PAN` (20 s) and `_RU_STA` (8 s) are dead
constants; what remains is the 4 s `_RU_COMMIT` socket backstop and the swallowed timeouts in
`_run_to_scan` and `tap_action`.

**Resolves.** `$365D` recurs every frame, so a timeout there means the game left the play loop —
raise, do not retry — plus deleting the two dead constants.

## 8. The enemy clock commits `consider_enemy_state`'s CORE writes early

**Wrong.** `driver.instrument --frames 3000 --follow` still reports CORE divergences on
ls9795 (**14** events, the first at frame 151) and on ls335 (**4**, the first at 613) — was
19 and 14 before the march carried its own write weight, 15 and 9 before the `$8401` chain
and the `$9630` body carried theirs (*Done* below).
ls42 is clean: **0 over 3000 frames**. Every event is an enemy's `update_cd` reading 4 in the
machine where the sim still reads 1 — one `$16ED` reload the sim reaches a frame late — or
the `$1805` rotation that follows from it. Everything this item has blamed in turn is now
measured, and none of it is the cause.  Those counts are from a seed that carries no error of
its own; the 116/24 this item used to quote were **partly the instrument's own seeding**, see
*The seed's own error* below.

Ten of ls9795's 14 and all 4 of ls335's are `update_cd`. The other four of ls9795's catch the
machine inside a replot, at `$9786`, `$978D`, `$988E` and `$9899` — the `$9730` buffer flush,
i.e. the replot's *exit* — and they now diverge on `obj[62].h_angle`, not on `update_cd` at
all (*The camera the replot borrows* below), where the
model's debt has run out and the machine's has not. `render_cost` reads 332100 cycles for
that pass against the machine's 22-frame wall (~343500 foreground), 0.97x — the `$2625` area
fill, which is [5](#5-one-object-vertex-angle-is-ten-units-out), not this item.

**Measured — the counted redraw is not the residual.** Charging `$1F9F` from the object's
own screen span instead of the retired 1723 mean changes what an ls9795 rotation costs by
-31, -91, +35 and +62 cycles at frames 20, 50, 80 and 103, i.e. **-25 cycles of accumulated
phase** before the frame-129 event, where reaching it a pass early needs ~2180. The gate
does not move: ls9795 first divergence **129 -> 129** and events **112 -> 111**, ls335
**478 -> 478** and **12 -> 12**, ls42 **0 -> 0**.

**Measured — the frame is fully accounted, to under 4 cycles.** `cpuhistory` stamps every
instruction with an absolute cycle, so the whole 19656-cycle frame splits four ways with
no inference: the VIC steal (a delta 20 or more above the opcode's own minimum), the four
short raster interrupts, the `$9630` body (`$95E9` to its RTI) and what is left for the
play loop. Means over 140 frames a board, `fixtures/live_pass_cycles.json`:

| term | model | ls0042 | ls0335 | ls9795 | ls9795 frozen |
|---|---|---|---|---|---|
| four short IRQs | 477 | 477.00 | 477.00 | 477.00 | 477.00 |
| `$9630` body | `IRQ_BODY` + gate + `$130C` + `$8ED1` | 2633.49 (2633.67) | 2645.73 (2646.07) | 2648.15 (2648.50) | 2454.34 (2455) |
| badline steal | 25 windows, priced apart | 1072.91 | 1073.96 | 1073.93 | 1075.20 |
| foreground | the rest | 15472.54 | 15459.24 | 15456.96 | 15649.27 |

The short interrupts are **exactly** 477 in every one of 500 frames. The body is within
**one cycle** of the model on all four captures: `$FFC5` a flat 56, `$1635` taking its
25-cycle fast exit every time (so `IRQ_SPRITES` never fires), `$FFC2` exactly
`SOUND_TICK_IDLE` 63 in all 80 frozen frames, and no `$130C` and no `$1635` at all when
frozen — the `$9659` gate, as `IRQ_GATE_SHUT` says.

So the whole frame-budget error was the badline steal. This split's own steal column is an
**upper bound** — it charges any delta over the opcode's *table* minimum to the VIC, so a
taken branch on a badline reads 44 and ls9795-frozen reads 1075.20, above the physical
25 x 43. The exact steal, per instruction rather than per table, is 1070.2 / 1072.1 /
1072.5 / 1072.0 on the same four captures: **-0.8 to +1.5 cycles a frame**, and its sign
changes with the board. The derivation is under *Boundary* below.

**Measured — the clock does not drift.** Frame-locked against the machine with the sim's
memory replaced from live truth every frame and only its cycle residual carried, so
nothing but the clock can move, over 1200 frames: ls42 **21354** machine passes against
**21363**, ls335 **9410** against **9408**. Nine passes and two — under 0.01 a frame,
either sign.

**A quiet frame is usually not a replot.** ls9795 reaches no `$1289` on **445 of 700**
live frames, in runs of 3, 4 and 7 — and a checkpoint on `$1805`/`$1F9F`/`$1FFC`/`$2625`
fires in none of them. Those are simply passes longer than a frame: one `$1887` march is
64080 cycles and the pass before the frame-129 rotation is a single **274578**-cycle
query, four and eighteen frames of foreground. Only **one** run in 700 frames is a replot.

**Measured — the replot is billed in the right pass.** That one run is frames 112..151, 40
frames, with `$1805`, `$1F9F`, `$1FFC` and `$2625` all inside frame **130**. The model
enters the same 274578-cycle march during the same frame 111, drains it, and charges its
own single replot — 389179 cycles, slot 3, span (37, 3) — in frame **129**. One frame, and
it is not a mis-billing: over 400 frames the model charges **exactly one** replot against
the machine's exactly one `$2625`, so nothing is double-billed, and the span it computes is
the ROM's own `$0C62`/`$0C69` byte for byte. What puts it a frame early is below.

**NOT the march price: it is exact, three ways.** The frame-129 event turns on one
`$16E6` — slot 3's, entered in frame 111 — whose `$17B2` scan spends a single 274578-cycle
`$1887` on the player. Halting the machine at that very `$16E6` and capturing the live
image:

* **The body is cycle-exact.** jennings on the captured image runs `$16E6` in **280640**
  cycles and `enemies.update_body` charges **280640**, with `$1F9F` stubbed on both sides.
  The same round is in the headless comparison too — ls9795 round 90, slot 3, 280640
  against 280640 — so `test_the_body_cost_model_matches_the_roms_own_16e6_cycle_count`
  now asserts its longest checked round exceeds 250000 and a long march cannot leave it.
* **The real 6510 agrees with jennings instruction by instruction.** cpuhistory over the
  live march, aligned against jennings' own stream of the same query, matches at **every**
  site: 89941 instructions, no cost differs. (`$1DB9 BCC` reads 3 live against jennings'
  2 only because the live sample never sees it not taken.)
* **The frame budget the march really gets is the model's.** Taking each frame's
  foreground as jennings' count for the instructions the live stream executed in it — no
  threshold, no classification — eleven whole frames inside the march measure 15166..15591
  where the model's own per-frame budget is 15174..15591; model minus machine sums to
  **+12 cycles over 11 frames**.

**Measured — what is left is ~1 cycle a frame.** The same alignment gives the machine's
position *inside* the march at every frame boundary, against the model's
`274578 + cycle_residual`: the model leads by **+81** cycles at the march's first frame
boundary and **+97** at its fifteenth, i.e. ~1.1 a frame on top of ~80 already accumulated
over frames 1..111. That lead is the whole event. Live, `$16E6` to `$17CD` is 354664 wall
cycles (17.926 frames) and `$17F9` a further 4098 (18.138), so the machine crosses the
frame boundary **between the scan's end and the rotate gate** and turns in frame 130; the
model, ~100 cycles ahead, reaches `$17F9` with ~119 of budget still in frame 129 and turns
there. Rotating with 119 cycles left is not a modelling error — the ROM's own `$1810`
h_angle write is ~80 cycles past `$17F9` — so nothing about *where* the model suspends
moves it. Only the ~100 cycles do.

**Boundary — the badline steal, now derived exactly, and why no constant can carry it.**
Per frame, 19656 = foreground + 477 + the `$9630` body + steal, and the first three are
counted off the image, so the residual is the steal. `driver/badline.py` measures it
exactly: `cpuhistory` stamps every instruction, and an instruction's *unstolen* cost is
the minimum delta over its own (opcode, branch taken, index page crossed) class — a steal
is never negative, so that minimum is its true cost. Every instruction off a badline then
reads a steal of **exactly zero** (256827 of 258267 in one ls9795 capture; the rest are
the 7-cycle interrupt entries), which is what makes the badline numbers exact.

The law, and it is a derivation, not a fit. A badline pulls BA low three cycles before
the VIC takes the bus and the 6510 runs on to its first **read** cycle, so

    one badline = 43 - (consecutive CPU write cycles at the window's first cycle)

43 being the 40 c-accesses plus the AEC lag. A write run is at most **two** — an NMOS
read-modify-write's dummy-plus-real pair, or a JSR's two pushes — because every
instruction opens with an opcode fetch, so **40 is unreachable and 44 impossible**.
Over **6424 live badlines** on ls0042/ls0335/ls9795, ls9795 frozen and ls9795 mid-march,
the steal is 41, 42 or 43 and `sentinel/badline.py` reproduces **every one** of them from
the opcode alone, with the window solved to one cycle-in-line (11) that is the same on
all five captures
(`fixtures/live_badline.json`, `test_every_live_badline_steal_is_the_derived_one`).
The 44s this item used to quote were an artefact: a static opcode table charged a taken
branch's or a crossed page's own extra cycle to the VIC. So was ls9795-frozen's 1075.20,
which is above the physical maximum of 25 x 43.

**The frame total is state-dependent, so no constant can be right.** Exactly:

| board | per badline | per frame | complete-frame range |
|---|---|---|---|
| ls0042 | 42.81 | **1070.2** | 1066..1075 |
| ls0335 | 42.88 | **1072.1** | 1069..1075 |
| ls9795 | 42.90 | **1072.5** | 1071..1075 |
| ls9795 frozen | 42.88 | **1072.0** | 1069..1075 |

Those three play means are the run's **head** — 10 captures a frame apart, ~15 distinct
frames, standard error ~0.6. Sampled across the run they are 1070.6 / 1071.1 / 1071.8
(*The ls335 gap was the measurement, not the model* below); the spread survives, the
individual figures do not.
The fitted 1071 was *above* ls0042's true steal and *below* the other two. The
writers the window catches are named in the fixture and are dominated by `$31CA`'s LFSR
(`ROL abs` at `$31D9`-`$31E5`, a dummy-plus-real pair, hence the 41s), then `$16D9 DEC
$0090`, the `$1289`/`$12A2` loop's `LSR`/`ASL`/`JSR`, and `$194D`/`$195F STY`. ls0042
runs ~19 passes a frame against ls9795's ~7, so it spends far more of the frame inside
`$31CA` and its windows catch far more writes — which is exactly why one constant cannot
serve three boards.

**Probe — the steal IS the lever, and one number cannot pull it.** Setting the term to
each board's own exact steal (the fitted `BADLINE_FRAME` **and** `IRQ_CYCLES`, which was a
stale literal — editing the fit alone had been inert):

| board | steal | first CORE | follow events |
|---|---|---|---|
| ls0042 | 1071 → 1070 | none → none | 0 → 0 |
| ls0335 | 1071 → 1072 | 478 → **155** | 12 → **20** |
| ls9795 | 1071 → 1073 | 129 → **130** | 111 → **122** |
| ls9795 | 1071 → 1075 | 129 → **66** | 111 → **164** |

ls9795's frame-129 rotation does move to the machine's own frame 130 — the phase lead
this item measured is real and it is the steal — but every other board pays for it.

**What closing it needs, precisely.** The model would have to know *which instruction*
the raster caught at each of the 25 windows. Two things were said to stand in the way.
Both were measured, and neither is a blocker; what is left is plumbing, not physics.

**1. The frame origin IS instruction-aligned — the 4-6 cycle spread is an output, not
blur.** The raster IRQ is taken at an instruction boundary, and the `$9630` marker's
frame position is that boundary plus a constant. Over **204 live frames** on five
captures the gap is **88** — `IRQ_ENTRY` 7 plus the 81 cycles `$95E9..$9630`, exactly
`IRQ_BODY`'s own split — with **8** frames reading 87, and every one of those 8 is a
**branch**, whose IRQ poll the NMOS core takes a cycle earlier (1292 short-IRQ entries
say the same: 7, or 6 off a branch, 55 of 55; a wider sample adds 5 off a branch that
crosses a page, *The IRQ entry is 7, or a taken branch's own cycles less* below). The
boundaries land at frame position
**13421..13426** — raster 213 cycles 2..7 — and 13421+88 = 13509, 13426+88 = 13514, which
is precisely the observed marker spread. So the spread is the tail of the interrupted
instruction, and a model that knows the instruction stream *predicts* it
(`badline.marker_position`, `test_the_9630_anchor_is_instruction_aligned_and_its_spread_is_that_instruction`).
The four short interrupts land on rasters 53/93/133/173 by the same law, so the frame's
whole IRQ layout is placeable.

**2. The write-cycle map per term exists, and the march resolves.** `sentinel/writemap.py`
walks a cost term's ROM run over the image — jennings' own 6510 length/cycle tables, with
`$CE DEC abs` corrected from its table's 3 to the machine's 6 — and reads the write cycles
straight off the addressing mode, so no hand table is needed. Measured against **6424**
live BA windows over five captures, one of them taken 112 frames into ls9795 so the loop
is inside a single 274578-cycle `$1887` march:

| | |
|---|---|
| windows falling after a `$XXXX` the cost model itself counts from | **6424 of 6424** |
| distinct (anchor, offset) keys | 1081 |
| keys carrying more than one steal | **3** |
| windows the static walk resolves with *no* branch record at all | 6147 (**95.7%**) |
| complete frames `badline.frame_steal` reproduces from the stream | 192 of 192 |

The march is not the hard case it was called: 842 of the 1600 march-capture windows land
in `$1CBB`, 240 in `$2BA8` and 237 in `$0D03`, and every one of them is anchored. All
three ambiguous keys are a branch the charging term *already* decides — `$0D05+57` is
`$0D03`'s per-bit shift-add (`MUL8_BIT`), `$1CCC+33` its per-component negate
(`ADD_VECTOR_NEG`), `$193A+10` the `$191F` targeting walk (`EXPOSURE_TARGETS_PLAYER`).

**Done — the fit is retired and the steal is charged per window.** Every cost term now
reports the ROM address it is counted from: `badline.charge(clk, $XXXX, cycles)` through
`enemies.py`, `relative.py`'s `$1887` and both numba twins, against
`sentinel/writeruns.py` — the per-anchor write-cycle map, regenerated from the image by
`driver/writeruns.py` (307 anchors, 1312 write cycles, `$31CA`'s eight LFSR rounds and
the pass head carrying the branch record the charging term itself decides).
`badline.frame_clock` pins the frame at raster 213, places the 25 BA windows and the four
split interrupts at their own raster positions, and hands `charge` the offset into the
running term. `passcost.BADLINE_FRAME` is **gone**; the frame is charged
`BADLINES_PER_FRAME * BADLINE_STEAL` (1075, the derived ceiling) up front and each window
refunds the write run it lands on.

| board | model steal, mean | machine |
|---|---|---|
| ls0042 | **1070.8** (1063..1075) | 1070.2 |
| ls9795 | **1071.5** (1064..1075) | 1072.5 |

**Measured against the machine, frame-locked, 3000 frames, `--follow`:**

| board | CORE events | first CORE |
|---|---|---|
| ls0042 | 0 → **0** | none → none |
| ls9795 | 111 → **116** | 129 → **130** |
| ls0335 | 12 → **24** | 478 → **155** |

ls9795's frame-129 rotation is gone: the model no longer turns enemy 3 a frame early, and
the machine's own frame 130 is the first CORE. ls0335 moves to 155/24 — which is what the
probe above predicted for its own true steal (1072 → 155/20), so the model is now right
about the steal and ls0335's residue is elsewhere. ls0042 is untouched.

**Placement matters and is not free — and the reason is not calibration.** Charging each
window's steal at its own raster position (rather than the whole frame's ceiling up front)
is physically right, and it still makes the model measurably *worse*. Re-run on the tree
that has since lost the `$9AF6`/`$37F2`, `$1F9F`, `$0C48`, camera-shift and CORE-write-offset
errors that were said to be compensating for it, under exact seeding over 3000 frames:

| board | CORE, ceiling up front | CORE, steal at its own raster |
|---|---|---|
| ls9795 | 19 (first at frame 151) | **96** (first at 77) |
| ls0335 | 14 (first at 156) | **140** (first at 111) |
| ls0042 | 0 | **0** |

`test_human_clock`'s pinned counts move both ways and net roughly nil (facing cadence
89 → 91, split cadence 87 → 86, ls335 exact spans 12 → 11, facing errors 42 → 39), so the
gate is the whole verdict. **The mechanism is measured, and it is not the rest of the clock
being calibrated against the lump.** The event list `frame_clock` arms covers exactly one
frame, but a term is charged atomically and may outlive several: over 300 ls9795 frames
**253 run no foreground at all**, the clock stops at **2701** of 19656 cycles, and only the
four windows the `$9630` body contains are ever reached. Placement therefore charges those
four (~172 cycles) where the frame really pays 25 (~1072), handing the model ~900 free
foreground cycles in every frame a march spans. ls0042 is untouched for the same reason it
was untouched before: its clock reaches 19655 and all 25 windows in every frame.

**What the ceiling costs instead, and where the residual accrues.** Model per-frame steal
against the machine's own (`fixtures/live_badline.json`, complete frames only):

| board | machine | model |
|---|---|---|
| ls0042 | 1070.22 | 1070.37 |
| ls0335 | 1072.08 | 1070.71 |
| ls9795 loop | 1072.56 | — |
| ls9795 inside the march | **1071.46** | **1074.76** (1075.00 in the 253 quiet frames) |

(The two whole-run rows are head-of-run machine means against a model measured on another
landscape; only the march row survives. See *The ls335 gap was the measurement, not the
model* below.)

So across a march the model runs **~3.5 cycles a frame short of foreground** — the sign that
makes the sim late, which is what every one of the 19 and 14 `update_cd` events surviving then was.
And it is not a placement question at all: `charge(clk, 0x1887, 274578)` walks a 49-cycle
write map, so every window inside the march is charged the full 43 whatever its position.
Of the machine's own 3.48 cycles a frame of refund there, **3.09** lands on the march's own
writes (`$1CBF`, `$1CFF`, `$1CD6`, `$1D4C`, `$1DF9`, `$0D07`, `$2BB9` …) and 0.42 inside the
`$9630` body, on the `$8CFC`/`$8D01`/`$8D0D` CIA1 strobes the `$119F` keyboard walk drives
(called the sound engine here until the walk was read; `driver.badline`'s routine table now
names `$0F62`/`$119F`/`$1363`/`$8CF9`/`$8F78` the keyboard walk and leaves `$8ED1` the note
tick) — and the `$95E9` walk RTIs out of the short-IRQ chain, so the model
refunded none of it either.
`test_a_term_outliving_the_frame_pays_the_ceiling_with_nothing_to_refund_it` pins all of it.

**Done — the march is charged by its own laps' write weight, and it closes.** No static map
can reach a data-dependent loop of thousands of laps, but the loop's *write profile* is fixed
arithmetic: a window `d` cycles into a term refunds the consecutive write cycles at `d`, so
summed over the term that is one per one-cycle write (STA/STX/STY/PHA/PHP) and three per two
(an RMW's dummy-plus-real pair or a JSR's pushes) — and 40 being unreachable, no run merges
across an opcode fetch. `sentinel/writeweight.py` carries that weight for every `$1CDD` term,
packed above the term's cycles so the march's existing accumulator sums both and no second
one is needed; `badline.charge_run` prices each window at the term's own `weight / cycles`
and **wraps the event list**, since every frame a term outlives is charged its own ceiling.

The weight is checked against the ROM, not fitted. The flat sub-step `$1CE8 JSR $1CBB` through
the `$1D18 BMI` that closes the loop, walked over the image for all four component-sign
patterns, is **306/310/314/318** cycles carrying **30/33/36/39** — exactly what the terms
`writeweight` composes claim
(`test_the_march_laps_write_weight_is_its_own_instructions`). The gate's own `$1887` is
274578 cycles carrying **35667**, 0.1299 a window against the machine's 0.139.

| ls9795, the gate's 274578-cycle march (frames 110..124) | steal a frame |
|---|---|
| machine | **1071.46** |
| model, the ceiling with nothing to refund it | 1075.00 |
| model, charged by the march's own write weight | **1071.73** |

**Measured against the machine, frame-locked, 3000 frames, `--follow`, exact seeding:**

| board | CORE events | first CORE |
|---|---|---|
| ls0042 | 0 → **0** | none → none |
| ls0335 | 14 → **9** | 156 → **297** |
| ls9795 | 19 → **15** | 151 → 151 |

`test_human_clock`'s pinned counts are untouched (facing cadence 89, split cadence 87, facing
errors 42, exact spans 117, facings 89, energies 86), and the rollout costs 6.2 → 6.6 ms
(ls42), 6.6 → 8.0 (ls335) and 7.7 → 8.7 (ls9795) per 3000 `advance_frame`s.

**Done — the `$8401` chain and the `$9630` body carry their own weight, and ls335 halves.**
The 0.21 of the machine's 3.48 a frame that the march's weight did not reach was in two
places, and neither is the march. **The `$95E9` body is the bigger one.** Its write map is
walked from `$95E9`, and with every branch falling through that walk takes the split chain's
`$962D JMP $969A` and RTIs after 380 cycles — before the *first* of the four windows the
body contains (offsets 389/893/1397/1901 from the raster IRQ), so `charge` refunded
**nothing at all** for the whole body. It is not a sound engine: the 2156 cycles of `$119F`
are `$1363`'s keyboard walk, seventeen `$8CF9` CIA1 matrix scans reached through `$0F62`,
and the walk cannot be placed statically because the body's length moves 450 cycles frame to
frame (the `$130C` tick, the `$8ED1` branches). So it is charged like a march: `charge_run`
with the weight of its own sequence, **203** (`writeweight.IRQ_BODY_WRITES`, `$0F62` and
`$8CF9` repeating `$11D9 LDY #$0E` + 2 times, pinned to the image by
`test_the_irq_bodys_write_weight_is_its_own_instructions`) — 0.340 a frame. Live over 40
`$9630` bodies the machine's own walk carries **211.4**, of which the `$8ED1` tick and the
`$9659` gate own 16.1: 195.3, plus the entry's 8. The fixed-point refund residue is smaller
than one cycle a frame, so it is now carried on `State.steal_residue` rather than reset with
the clock; a per-frame stand-in would have been inert.

The `$8401`/`$9287`/`$0D4A` chain now carries its own weight too (`writeweight.TRIG`), each
term walked over the image to exactly the cycles and write cycles it claims
(`test_the_trig_chains_write_weight_is_its_own_instructions`). It is worth only **+0.025** a
frame inside a march — the coincidence was real — but it also stops a query that rejects at
the `$18CA` FOV gate being charged against `$1887`'s 49-cycle map, which its ~460 cycles
outran; those now price by their own density, and ls42's whole-run steal lands on the
machine's.

| model per-frame steal | machine | ceiling only | + march weight | + trig and `$9630` |
|---|---|---|---|---|
| ls9795, inside the gate's march | **1071.46** | 1075.00 | 1071.78 | **1071.41** |
| ls0042, whole run | 1070.22 | — | 1070.43 | 1070.23 |
| ls0335, whole run | 1072.08 | — | 1070.72 | 1070.44 |

The two whole-run rows are **retracted**: neither column is that board. See *The ls335
gap was the measurement, not the model* below.

**Measured against the machine, frame-locked, 3000 frames, `--follow`, exact seeding:**

| board | CORE events | first CORE |
|---|---|---|
| ls0042 | 0 → **0** | none → none |
| ls0335 | 9 → **4** | 297 → **613** |
| ls9795 | 15 → **14** | 151 → 151 |

`test_human_clock`'s pinned counts are untouched (facing cadence 89, split cadence 87, facing
errors 42, exact spans 117, facings 89, energies 86) and the rollout costs 11.4 → 11.5 ms
(ls42), 12.2 → 12.3 (ls335) and 9.9 → 10.0 (ls9795) per 3000 `advance_frame`s, i.e. +1%.

**The ls335 gap was the measurement, not the model.** The 1.64-cycle deficit the table
above records for ls0335 does not exist, and neither column was that board's own steal:

* The **model** column was measured on **another landscape**. The probe read its argument
  as hex and handed it to `Game.typed`, which takes the number a player TYPES -- so
  `Game.typed(int("335", 16))` generates the board typed 0821, seed `$0821`, not ls0335
  (seed `$0335`), while printing "$0335". Those stand-in boards read 1070.25 and 1070.49
  over 3000 frames, which is the table's 1070.23 and 1070.44; the boards themselves read
  **1070.52** and **1071.32**.
* The **machine** column came from 10 captures **one frame apart** at the head of the run.
  A cpuhistory window spans 4-5 frames, so that mean rests on ~15 distinct frames whose own
  per-frame total has sd 2.2 -- a standard error of ~0.6, and 1072.08 is a head-of-run
  fluctuation.

`driver/badline.py --stride` runs frames between captures, so the mean covers the run
rather than its head. Machine (40 captures, +/- one standard error) against the model's
own frame clock on the same board over its own first frames:

| board | machine, stride 20 | machine, stride 50 | model, 800 frames | model, 3000 |
|---|---|---|---|---|
| ls0042 | 1070.68 +/- 0.19 | 1070.55 +/- 0.22 | 1070.51 | 1070.52 |
| ls0335 | 1071.16 +/- 0.22 | 1071.11 +/- 0.21 | 1071.36 | 1071.32 |
| ls9795 | 1071.87 +/- 0.18 | 1071.82 +/- 0.19 | 1072.02 | 1071.68 |

**The model's per-frame steal is the machine's on all three boards, to within 0.25 cycles.**
`test_the_models_own_frame_steal_is_the_one_the_machine_pays_on_each_board` pins each board
against the fixture, which is now sampled 40 frames apart (20 captures a board, ~800 frames)
and names the board by its typed number.

Where each side's refund lands, per frame (machine from the fixture's own writers, model
from the anchor its clock refunds at, both over the same three boards):

| refund a frame | ls0042 | ls0335 | ls9795 |
|---|---|---|---|
| `$31CA` prnd | 2.69 / **3.39** | 1.74 / **1.42** | 0.38 / **0.37** |
| the `$1887` chain | 0.09 / **0.13** | 1.24 / **1.28** | 1.99 / **1.90** |
| the `$9630` body | 0.76 / **0.34** | 0.66 / **0.34** | 0.49 / **0.34** |
| `$1289` pass head/tail | 0.41 / **0.44** | 0.20 / **0.29** | 0.07 / **0.14** |
| `$16E6` body and the rest | 0.39 / **0.14** | 0.23 / **0.21** | 0.16 / **0.18** |
| total | 4.35 / **4.45** | 4.08 / **3.63** | 3.09 / **2.96** |

The one term whose sign is the same on every board is the `$9630` body: it refunds **0.34** a
frame where the machine pays 0.49..0.76. That is what is left of the steal, and it is worth
~0.3 a frame on every board rather than 1.64 on one.

**Measured — the body's weight is its real path's, and the gap is the interrupt's own cycle.**
The suspect was the fall-through walk. It is not that. `cpuhistory` over **188** raster bodies
on ls0042, ls0335 and ls9795 (20 captures a board, 40 frames apart, each verified on the board
it claims by regenerating `$0400-$07FF`) reads **one path**, the same on every board and every
frame, branch for branch:

| run | cycles | write weight | |
|---|---|---|---|
| the 6510 interrupt sequence | 7 | — | three pushes, before the `$95E9` fetch |
| `$95E9`..the `$963D` JSR | 153 | 18 | `$9621 BEQ $9630` taken, `$9633 BPL` not, and `$FFC5`'s three idle voices |
| the `$8ED1` note tick | 63..130 | charged apart | `sound_frame` |
| `$9640`..`$9652 BEQ` | 25 | 0 | four reads; the `$9652 BEQ $9659` taken |
| the `$9659` gate | 7, 64, 76 or ~470..550 | charged apart | `cooldown_frame`, `$130C`, `$1635` |
| `$9669`..the `$969F` RTI | 2200 | 185 | `$130B` is 0, so `$9671 BEQ` runs `$119F` every frame |

7 + 153 + 25 + 2200 = **2385** = `passcost.IRQ_BODY`, and 18 + 185 = **203**, the weight
`writeweight` already carried. So the composed weight was never the fall-through walk's; it is
the real path's, and `writemap.walk` from `$9669` with that path's branch record reproduces all
**739** instructions of the tail. `test_the_irq_bodys_write_weight_is_its_own_instructions` now
walks all three runs end to end instead of checking the pieces by repetition count. Nothing is
double-counted or missed at the `$8ED1`/`$9659` boundaries either: the tick is exactly
`$963D`..`$9640` and the gate exactly `$9659`..`$9669`, on 188 of 188 bodies.

**The four windows land 375..380 cycles past the `$95E9` fetch.** Measured, the body's four BA
windows sit at **375..380 / 879..884 / 1383..1388 / 1887..1892** from the `$95E9` opcode fetch
— the recorded 389/893/1397/1901 are from raster 213 cycle 0, and the difference is `b`, the
cycles from the raster assert to the instruction boundary the IRQ is taken at (2..7, mean 3.4;
the model's clock assumes 0). All four fall inside `$119F`'s seventeen `$8CF9` matrix scans,
and the machine's refund is two single-cycle stores in them: `$8CFC STA $DC02` and
`$8D0D STA $DC00`. Placing the four in the body's own write map, in ROM order, with the
machine's own `b`, reproduces the machine **body for body**:

| board | bodies | machine | placed at the machine's `b` | bodies agreeing exactly |
|---|---|---|---|---|
| ls0042 | 68 | 0.735 | **0.721** | 67 of 68 |
| ls0335 | 60 | 0.617 | **0.617** | 60 of 60 |
| ls9795 | 60 | 0.467 | **0.433** | 59 of 60 |

**So the residual is one datum, `b`, and it is not usable.** The refund is a knife edge on it —
placed at a fixed `b`, per frame:

| `b` | 0 | 1 | 2 | 3 | 4 | 5 |
|---|---|---|---|---|---|---|
| ls0042 | 0.04 | 0.28 | 0.63 | **1.28** | 0.02 | 0.22 |
| ls0335 | 0.00 | 0.23 | 0.48 | **1.15** | 0.38 | 0.00 |
| ls9795 | 0.05 | 0.25 | 0.45 | **0.92** | 0.28 | 0.07 |

— and `b` is the interrupted instruction's own remaining cycles, which the model has no way to
count: it charges the body first and hands the frame's foreground what is left, so it holds no
instruction boundary at the raster. Averaging the placement over `b` uniform on 2..7 gives
0.48 / 0.40 / 0.32, still short of 0.735 / 0.617 / 0.467 because the machine's `b` concentrates
at 2..3, and that distribution is assumed rather than derived. The uniform-density `charge_run`
the body has now **is** that expectation, taken over a phase spread wide against the 121-cycle
scan lap; its only correctable bias is that 18 of the 203 sits in the 185 cycles ahead of the
tick that no window can reach, worth 0.004 a frame.

**And closing the body alone would break the totals.** The model's whole-frame steal is already
inside 0.25 of the machine on all three boards (1070.52 against 1070.55..1070.68, 1071.32
against 1071.11..1071.16, 1071.68 against 1071.82..1071.87), because `$31CA`'s own refund runs
+0.70 high on ls0042 and −0.32 low on ls0335. Adding the body's +0.40 / +0.28 / +0.13 would put
ls0042 and ls9795 outside it. Whatever eventually carries the body's phase has to carry
`$31CA`'s at the same time.

**The IRQ entry is 7, or a taken branch's own cycles less.** The wider sample also carries
entries of **5** cycles, which the 6/7 law had no case for. Keyed by the interrupted
instruction's own unstolen cost, all 1940 live entries are one law and it is not a fit: 7
past anything else, but a TAKEN branch polls the IRQ two cycles in, so a raster already
asserted starts the sequence there -- **6** off its 3 cycles (108 of them), **5** off the 4
of a page crossing (4) -- and one asserting after that poll lets the branch finish and pays
7 (133 taken branches do). `badline.entry_cycles` is that law;
`test_the_interrupt_sequence_is_short_only_by_a_taken_branchs_own_cycles` checks every entry
against it. The static walk resolving windows with no branch record at all reads 93% across
a run rather than the head-of-run sample's 95%.

None of this is a model change: `entry_cycles` and `marker_position` are the frame-origin
law and nothing charges through them, no term's cycles or weight move, and the gate is
unchanged end to end — CORE **14 / 4 / 0** with the same first divergences (151, 613, none)
over 3000 frames under exact seeding, `test_human_clock`'s pins held, and 6.5 / 7.7 / 8.9 ms
per 3000 `advance_frame`s.

**What is left.** All eighteen survivors are `update_cd` — ls335's four and every one of
ls9795's fourteen, the four `$9730` buffer-flush positions (`$9786`, `$978D`, `$988E`,
`$9899`) included: `obj[62].h_angle` is closed and those four now diverge on `update_cd`
like the rest. The `$0F4A`/`$0D03` multiply beneath `$937F` still carries no weight of its
own; it is shared with `$1C54`'s sin/cos and with the march's `$1DAF`, so naming it means
unpacking at the projector and object-cost boundaries as well.

**Measured — every event is one `$16E6` gate decision, and the lead at it is under 140
cycles.** `resume_from_stack` prices the machine's own position off the ROM's straight
lines, so wherever both sides sit in the same segment the difference of the two residuals is
the model's lead **in cycles**. Raced frame for frame with no resync (each board verified by
regenerating `$0400-$07FF` against `landscape.generate`: **0** mismatched bytes on all
three), the lead does not run away — ls9795 +3 -> +88 over frames 1..102, ls0042 -13..+87
over 300 frames, ls0335 +65 down to -55 over 600. Under `--follow`, at the last comparable
frame before each event:

| board | the model's lead, cycles, at each event |
|---|---|
| ls9795 (14) | +88, -87, -7, -6, +74, +93, +114, +132, +132, +96, +110, +54, +71, +22 |
| ls0335 (4) | -55, +112, +68, +92 |

Every event is the same shape. One enemy reads `emu=1 sim=4` — the model reached `$16ED`
first — and in **9 of the 18** a second enemy reads `emu=4 sim=1`, the model arriving at
*that* enemy's body one tick early, finding `update_cd` still 2 and shutting the gate the
machine opened. Both signs are the same lead. Ninety cycles of 19656 is 0.5% of a frame and
ls9795 races 2638 frames at ~1.1 passes a frame, so ~11 gate decisions are expected inside
the band against the 14 seen. **No event carries a lead of a term's size**: there is no
mis-priced routine hiding in them.

**Measured — the per-window refund's per-frame value carries no information about the
machine's.** Reseeding the sim at the machine's exact cycle **every** frame reduces the lead
to one frame's budget error: mean **-0.5**, range -10..+8, sd **3.7**, the same on ls0042
(n=107) and ls9795 (n=35) and in every phase pair. Switching `badline.charge`/`charge_run`
to return their cycles unrefunded and charging a flat frame steal instead, on the *same*
frames, leaves the mean where it was and drops the spread to sd **3.2** (ls0042) and **3.1**
(ls9795) — Pitman-Morgan t = 3.49 and 2.25, the same sign on both boards. A refund that
tracked the machine would *shrink* the spread; this one adds its own variance to it, so its
covariance with the machine's own per-frame steal is ~0. Its **mean** is right, which is why
the whole-run steals match to 0.25 cycles, and its per-frame value is noise — which is what
random-walks the phase out to +/-100 cycles over the 60..290 frames between events. The gate
run confirms the term: flat steal (1072 / 1071) reads ls9795 14 -> **10** and ls0335 4 ->
**10** (first at 479, was 613), so the events are the steal's and no constant serves both
boards. Diagnostic only; nothing is kept.

**Measured — window by window: the write maps are right, the placement is not.** The
per-frame figure above is an aggregate; the same seeding prices each window directly. Seed the
sim at the machine's exact cycle, run ONE frame on both, and read the sim's own refund for
event *i* against the steal `cpuhistory` measures at that window. Over 120 exactly-seeded
frames a board — 1974 windows on ls0042, 1466 on ls0335:

| | ls0042 | ls0335 |
|---|---|---|
| model refund a window | 0.150 | 0.171 |
| machine refund a window | 0.162 | 0.158 |
| windows the model prices right | 80.3% | 79.5% |
| answering 0 to every window would price right | **88.2%** | **89.4%** |
| nonzero calls, of them real | 205, **31** | 179, **23** |

The mean is right to 0.01 and the per-window answer is *worse than a constant zero*: precision
15% and 13% against base rates of 12% and 11%, so the calls are chance. A uniform shift of
every offset over -8..+8 moves the agreement by under 4 points and has no peak, so it is not
one origin offset.

**Where the sim places a window exactly, it prices it exactly: 13 windows of 13** (10 of 10
within a cycle on ls0042 alone). Recording the sim's own `(anchor, offset)` per window and
finding that anchor in the machine's own instruction stream gives the placement error in
cycles, ls0042, n=1249:

| the sim's anchor | n | p25 | median | p75 |
|---|---|---|---|---|
| `$95E9` raster-IRQ body | 66 | +10 | **+10** | +12 |
| `$16D6` prnd | 671 | -483 | +5 | +146 |
| `$1925` exposure | 264 | -625 | -138 | -8 |
| `$12A2` pass tail | 162 | -622 | -518 | -38 |
| `$1289` pass head | 31 | -696 | -593 | -509 |

Inside the body the sim is a tight **+10** cycles ahead — `b` plus the 7-cycle entry, which
`badline.entry_cycles` already names and `advance_frame` does not apply. Past the body the
error is pass-scale, not cycle-scale: `$16D6`'s own term is 433 cycles and its quartiles span
-483..+146, so at a quarter of those windows the machine is in a *different pass's* prnd, and
`$1289`'s whole run is 25 cycles against machine offsets of 389..787. Median |error| by window
index runs 10 inside the body, ~150 over the next four windows and ~480 by mid-frame,
oscillating against ls0042's ~630-cycle idle pass. **The sim's offset into its own term is
not where the machine is**, so no per-window refund of any kind can track the machine until
the within-frame phase does. (Comparing the *anchor's* routine against the machine's PC
overstates this — a term legitimately spans into its callee, `$16D6` into `$31CA` and `$95E9`
into `$119F` — so it is the offset, not the routine, that carries the measurement.)

**So `charge_run`'s uniform smear is a defect, but not the binding one.** It has two live
callers — the `$9630` body and the `$1887` see cost — and in the frames this seeding can
measure at all it prices **4.1 of 25 windows** (ls0042 4.1, ls0335 4.1, ls9795 4.2), all of
them the body's: a march frame has no countable position, so it is skipped. The other ~20 go
through `run_at` against a real map, `$16D6`'s exactly-measured prnd map the largest share
(918 of 1974 on ls0042), and price right 73.2% against the 88.2% a constant zero scores. Both
paths are uninformative, so charging the see path's terms individually would replace one
uninformative refund with another while the placement stands.

**Charged in the ROM's own order: `$17B2`.** The scan loop charged its whole slot *after* the
`$1887` query the ROM runs in the middle of it, and its 22..36-cycle map walked through the
`JSR` into `$1887`'s own prologue — writes at offsets 10/14/18 belong to the callee the model
charges separately. It is now the ROM's three pieces: `SCAN_SLOT_CALL` 8 at `$17B2` (map
`(5,6)`, the JSR's pushes), the query, the branch's own test at `$17B7` (map `(22,)`, the
`$17C8 STY`) and the `$17CA` loop step (no writes). Cycle totals per slot are unchanged.
**The gate reshuffles rather than improves: ls9795 14 -> 11, ls0335 4 -> 8, ls0042 0 -> 0**
(18 events to 19), which is what a refund carrying no per-window information does under any
perturbation. Kept for the map and the order, not for the count.

**Not the frame origin.** `b`, the 2..7 cycles from the raster assert to the instruction
boundary the IRQ is taken at, is not the missing datum. Shifting the whole event list by a
constant b = 0 / 3 / 6 leaves the single-frame error at mean -0.44 / -0.52 / -0.44 and sd
3.91 / 3.99 / 3.42 on ls0042 (n=63). Whatever the refund is missing, it is not the origin.

**Four of ls9795's fourteen are the replot's price, not this item.** Frames 151, 906, 1657
and 2404 catch the machine in the `$9730` flush, ~5000 cycles of a ~180-frame gap, so chance
would put none of them there. At the first, the model charges **323117** cycles for the
frame-130 strip replot, pays it at ~15575 a frame and leaves frame 151 with **2004** cycles
to spare while the machine is still at `$9786`, ~4000 cycles short of the flush's end: the
pass is ~6000 of ~329000 cycles cheap, ~2%, which is `render_cost` and
[5](#5-one-object-vertex-angle-is-ten-units-out). The debt itself is charged with **no clock
at all** (`state.cycle_residual -= debt`), so every frame the replot spans pays the whole
1075 ceiling with nothing to refund against a fill that is almost all stores — worth ~4
cycles a frame the other way, so it cannot be closed on its own.

**The seed's own error, and its removal.** `resume_from_stack` counts the machine's position
off the ROM's own straight lines and returns no offset for a position *inside a call*, so a
resync there started the model at the call's head and injected everything the machine had
already spent in it. Measured over the gate's own 3000-frame races (`--seed-any`, the shipped
behaviour): **4 of 117** ls9795 seeds and **3 of 25** ls0335 seeds were at the machine's exact
cycle. `instrument._seed` now steps frames until the `$9630` marker catches the loop on a line
it can count, and reports any seed it cannot make exact per event instead of absorbing it:

| board | CORE, seeding anywhere | CORE, seeding exactly | exact seeds | frames raced |
|---|---|---|---|---|
| ls0042 | 0 | **0** | 1 of 1 | 3000 of 3000 |
| ls0335 | 24 | **14** | 15 of 15 | 2917 of 3000 |
| ls9795 | 116 | **18** | 19 of 19 | 2516 of 3000 |

Per raced frame that is 0.0387 -> 0.0072 on ls9795 and 0.0080 -> 0.0048 on ls0335, so it is
not the shorter race. ls42 stays clean. The waiting is the cost: a stretch in which no marker
lands on a countable line is skipped, up to **296 frames** on ls9795 — one `$17B2` scan whose
`$1887` marches run ~4.6 M cycles, during which every frame boundary falls inside a call.
Those are the segments that genuinely cannot be resolved to a cycle: an `$1887` march (the
missing datum is cycles already spent in the query) and a `$1FFC JSR $2625` replot (the
missing datum is `plot_world`'s own progress, [5](#5-one-object-vertex-angle-is-ten-units-out)).

**What the surviving events are, classified by the PC the raster IRQ interrupted** and by
`resume_from_stack`'s own chain, every one of them from an exact seed:

| the model's resume point | ls9795 (18) | ls0335 (14) |
|---|---|---|
| `$17B7` — `$17B2`'s scan slot, machine under `$1887` | **8** (`$1893`, `$189D`, `$1916`x2, `$191D`, `$1D16`, `$17B7`, `$85DE`) | **9** (`$1887`'s line, `$1D8A`, `$85FE`, `$92DB`, `$9306`) |
| `$17E8` — `$1AB0`'s tree/boulder walk | 2 (`$1AB5`, `$1ABE`) | 2 (`$1AB5`x2) |
| `$1884` — the `$1876` redraw tail, inside a strip replot | 4 (`$9786`, `$978D`, `$988E`, `$9899`, all the `$9730` flush) | 0 |
| the body's and the loop's own lines | 4 (`$16F4`, `$17B0`, `$17B2`, `$31D8`) | 3 (`$1773`, `$17B2`, `$17BC`) |

* **Both boards are now the same defect: the `$1887` see cost.** Ten of ls9795's fourteen
  `update_cd` events and eleven of ls0335's fourteen catch the machine inside `$17B2`'s or
  `$1AB0`'s scan, in the `$1887`/`$18E6`/`$1CDD`/`$8401`/`$9287` chain the clock charges as
  one opaque term. ls9795's renderer share is gone — 68 of 116 became 4 of 18.
* **The camera the replot borrows.** `$1FC2` adds `$0C62/2` to the *player's* own `$09C0,X`
  and `$2003`/`$2008` put it back, so for the whole stall the object table carries a bearing
  the model priced but never wrote: on ls9795's strip, `$0C62` is 37 and the machine read
  **18 above the model**. That was 5 of 22 events, every one of them `obj[62].h_angle` alone.
  The model now writes the shift and takes it off — `$187D`/`$187F` name the camera object
  (the player, not the drawn one), `$211A` holds its own bearing and `$001F` the odd half
  column, and the shift lands only once the replot's own `$1FBA JSR $2211` clear is spent,
  8021 cycles in. All five are gone and nothing else moved (ls0335 14 -> 14, ls42 0 -> 0).

**Measured — the length, and a backend that is dearer than the machine.** Caught at
`$1F9F` with a stopping checkpoint, that replot takes **22 whole frames** to the next
`$1289`. `strip_replot_frames`' proxy charges 387392 cycles = 24.6 frames of stall,
**1.12x** — inside its own 0.93..1.21 band. Its `RENDER_COST_BACKEND=py65` backend, on the
same captured image, returns **534090** cycles = 34.0 frames, **1.54x**. The machine cannot
have spent 534090 cycles in 22 PAL frames — that is 24277 a frame against the frame's
19656 — so the *exact* backend's render context is dear, and the proxy calibrated against
it inherits that. Both belong to [5](#5-one-object-vertex-angle-is-ten-units-out), not here.

`~17 cycles a frame` was therefore never there. The budget is right to under 4 and the
clock to under 0.01 of a pass a frame; what is left is a render cost and its timing, and the
`$1887` see cost the surviving events name.

ls9795 and on ls335. ls42 is clean: **0 over 3000 frames**. Every event is an
enemy's `update_cd` reading 4 in the sim where the machine reads 1 — one `$16ED` reload the
sim reaches a pass early — or the `$1805` rotation that follows from the same lead. The
`$1887` chain is no longer the cause: it is now cycle-exact against the oracle, and the
classification below puts more than a quarter of what is left inside `$1AB0`
`find_drainable_boulder_or_tree`, the tile scan `consider_enemy_state` ends on.

**Measured — the counts, and what each event is.** ls9795 **142 -> 64 -> 67 -> 64**, ls0335
**57 -> 57**, ls0042 **0 -> 0** over 3000 frames; the 64 -> 67 was pricing the play machine's
own `$37F2` examine, which took a strip replot's over-charge away, and the 67 -> 64 is
`$1F9F`'s own line replacing it with the real thing
([6](#6-the-py65-exact-backend-cannot-price-another-slots-view),
[architecture.md](architecture.md#the-divergence-instrument-driverinstrumentpy)).
Classifying every surviving event by the `$95E9` chain the halt exposes:

| where the machine was, at each of the 64 | events |
|---|---|
| `$1AB0 find_drainable_boulder_or_tree`, its trig included | **18** |
| `$1887 can_see_object`, its trig included | 14 |
| `$31CA update_prnd` under `$16D6` | 10 |
| `$17B2 find_drainable_robot_loop` | 5 |
| `$16E6` body head | 5 |
| inside the `$1FFC JSR $2625` replot | 4 |
| inside the `$207E JSR $9730` buffer flush | 4 |
| `$12A2` pass tail, `$3470`/`$3527`, one `$2A31` | 4 |

At 142 the replot alone was 80 of them: a replot's 21 frames produced 21 consecutive events,
where it now produces the one that finds it plus a couple at its exit. ls0335 is unchanged
because no resync in 3000 frames lands inside a replot at all. **`$1AB0` is now the largest
single group** and is what this item names: 18 of 64, ahead of `$1887`'s 14.

**Not a flag the model guessed.** At all 64 halts the live machine has `$0C6D`, `$0C4D`,
`$0C4E`, `$0C5F`, `$0008` and `$0095` **all zero** — the branch configuration `passcost`'s
`$1FA4..$1F9E` prices and the `$9730` row window it assumes, confirmed against play rather
than against the harness.

**Measured — where the frame boundary lands.** The raster IRQ interrupts the play loop at a
raster position, not at a pass boundary, and the interrupted PC is on the `$95E9` stack
frame, so halting at `$9630` and reading `SP+5/6` gives the ROM's sub-pass position
directly. Over **1500 live frames a board**:

| segment the IRQ interrupted | ls0042 | ls0335 | ls9795 |
|---|---|---|---|
| `$1289..$16D8` head and `$16B5` dispatch | 6.9% | 2.6% | 1.7% |
| `$16E6 consider_enemy_state` | **2.4%** | **48.5%** | **65.0%** |
| `$16D6 JSR $31CA` prnd | 52.3% | 26.8% | 17.4% |
| `$16D9..$12C7` cursor, loop tail and `$191F` | 38.4% | 22.1% | 15.9% |

That column orders the three boards exactly as the divergence count still does, and it is
what sent the search into the body — where five means were indeed found. It no longer says
where the residual is: the body it names is now cycle-exact, and the boards keep their
order because a long body is what makes a per-frame error visible at all.

**What the sub-pass and sub-body splits fixed.** A pass is spent in four segments with
`pass_phase` naming the resume point, and `consider_enemy_state` itself in ten stages with
`body_stage`/`body_index`/`body_partial` naming a position between an `$1887` and the CORE
write its answer causes ([architecture.md](architecture.md#consider_enemy_state-is-resumable-too)).
ls42 stays clean throughout. The staging is visible in the trace: at ls9795 frame 84 the sim enters slot 5's body,
applies `update_cd = 4` and suspends at `BODY_SCAN` slot 61 owing 50771 cycles; frames
85-87 pay it with the ROM inside `$1CBF..$1CEB`, the same march; frame 88 commits the
outcome. That is what the ROM does.

**What the cost derivation fixed.** Four terms were means over routines that branch,
and each is now computed from state both twins already hold:

* `SEE_PROBE` folded a 3870-cycle mean for `prepare_vector_from_angle $1C54`; `$1C54`
  is now charged from its own branches, with `$0D03 multiply_byte_by_byte` at 96 + 4
  per set bit of its multiplier and `$0E75`, `$1C9D`, `$0F9E`, `$0F4A`, `$0F3E` priced
  by the paths the ports take.
* `MARCH_OBJECT` was the depth-1 constant for `get_tile_z_from_object $1E3F`; the walk
  is charged per stack level by the branch it takes.
* `MARCH_STEP` lumped `$1CBB add_vector`, the `$1CEB`/`$1CF3` edge tests, `$1DF9` with
  `$2BA8`, and `$1D0D check_flat_tile`; each is now charged at its own exit, a taken
  branch 3 and 4 when it crosses a page (`$1CF1`/`$1CF9`->`$1D44`, `$1D18`/`$1D40`->`$1CE8`).
* `MARCH_SLOPE` (one constant, then two) was a mean over `check_sloping_tile $1D46`,
  whose cost is three `$1DF9` corner reads — each able to walk an object stack — plus,
  on the `$1D8A` quad path, a `$0D03` multiply and the `$1007` invert.

**What the geometry derivation fixed.** `SEE_GEOMETRY` (1128) and `SEE_PROBE` (719) were
the last two means. `$1887` is now charged per branch of its own line; `$8401` from
`$85C4`/`$85F5` and the sign of each component delta; `$9287` from its quadrant compare and
the **variable length** of the `$92C1`/`$92FF` shift loop (20 cycles a lap, 21 in y, whose
`$9306` branch crosses a page); `$0D4A` per conditional-subtract round plus the `$0E1F`
arctan interpolation; `$933D` and `$937F` from their own lines; and `$1CDD`'s `$1ECC` entry
as its own term. Deriving them exposed four march errors that the old means had absorbed:
the per-sub-step fixed cost charged `$1CFB..$1DFE` on a board-edge exit that never reaches
it, `$1E0E CPY $0C58` was priced as zero page when it is absolute, the `$1D18`/`$1D30`
not-taken branches and the `$1D44 SEC`/`RTS` after a slope block went uncharged, and in
`$1C54` the `$0E7B` JSR was counted twice while `$0E8F`'s taken `BVC` and `$1C76`'s
`JSR $1C7D` were not counted at all.

**Measured — the oracle.** Against jennings on the real image: `$1887` is cycle-exact on
ls42/ls335/ls9795 over every observer/target pair, both FOV widths, the reject paths and
the two-probe robot path, and over a whole 8x64 scan run as one sequence on one machine;
`$1CDD` is exact over random rays on five boards; `$1C54` over all 55296 angle/fraction
pairs; `$9287` and `$933D` over thousands of random inputs.

**Measured — `$1AB0` and `$1887` are not the residual: their price is exact.** Both are
now checked against the ROM directly rather than only through the rounds a generated board
happens to run (`test_the_see_cost_model_matches_the_roms_own_1887`,
`test_the_tile_scan_cost_model_matches_the_roms_own_1ab0`). `$1887` is cycle-exact, and
leaves the same `$0014`, over every occupied slot as target on ls42/ls335/ls9795, asked for
as the robot `$17B2` wants and as its own type, from every enemy aimed at the player and
turned away. `$1AB0` is cycle-exact over its whole loop, with the board's top-of-tile trees
restacked so its drain half runs at all: on a freshly generated board nothing stands on a
stack, so `TILE_SCAN_TILE`, `_SEE`, `_SEE_BOULDER`, `_NEXT`, `_HIT`, `SCAN_SLOT_FULL`,
`TARGET_*` and `DRAIN_*` were never once reached by the `$16E6` oracle rounds.

**Fixed, and it was a write's placement, not its price: the body committed every CORE
write EARLY.** Stepping the ROM's `$16E6` instruction by instruction, recording the cycle
each CORE byte changes at, and binary-searching the smallest budget at which
`enemies.update_body` makes the same write, every one of the 153 writes the aimed bodies
make on the three boards landed before the ROM's own offset, because the model applied a
segment's writes at the segment's *start*:

| write | ROM offset in its segment | early by |
|---|---|---|
| `update_cd` (`$16ED`) | 15 | 15 |
| `h_angle` (`$1810`) | rotate + 66 | 65 |
| `rotation_cd` (`$1815`) | rotate + 73 | 72 |
| `targeted_object` (`$1826`) | `$1825` + 7 | 66 |
| `targeted_exposure` (`$182B`) | `$1825` + 15 | 74 |
| `draining_cd` (`$1837`) | `$1825` + 30 | 89 |

`State.body_paid` and `enemies._reach` now charge each stage forward to a write's own
cycle and commit it only there: `$1825` is a stage (`BODY_TARGET`) rather than a call, the
rotate line is seven rungs, `$16E6`'s gate and reload are two, and `$17CD..$17E5` is one
laddered stage covering `$1973`'s four writes and the `$17E2` drop. Every offset is the
instruction sequence at its address and every ladder sums to the whole-segment term the
`$16E6` oracle already pins (`test_every_write_cycle_offset_sums_to_its_own_segment_term`).
`test_the_body_commits_its_core_writes_at_the_roms_own_cycle` now steps the ROM and
demands the model make **every** CORE write on the ROM's own cycle and not one cycle
earlier, on ls42/ls335/ls9795 with the meanie bytes seeded dirty so `$1973`'s four writes
are changes; `test_a_resumed_segment_picks_up_at_the_roms_own_cycle_inside_it` checks the
per-PC resume tables against the same stepped ROM, and
`test_a_body_split_at_every_single_cycle_repeats_and_skips_nothing` suspends a rotating
body at every one of its cycles.

**Measured — the body is now exact.** Stepping `$16E6 consider_enemy_state` one round at a
time against the same oracle, comparing **every** round the play loop dispatches, is
cycle-exact on ls42, ls335, ls9795, ls0, ls60, ls110, ls298 and ls373 — gated, marching,
rotating, held-target, draining and discharging rounds alike
(`test_the_body_cost_model_matches_the_roms_own_16e6_cycle_count`). Getting there priced
five regions that were means or were not charged at all:

**No longer a mean: the rotation's redraw.** `$1F9F update_object_on_screen` **does** run
headless — it needs no render context on the branch the enemy clock takes. It calls
`$209B calculate_object_screen_span` first, which takes the object's bearing and distance
from `$8401`, re-arctans its `$2112` half-angle over that distance through `$933D`, and
turns bearing +- half-angle into the left column `$0C62` and the width `$0C69`. With no
span on the 40-column screen it returns carry set and `$1FA2` exits at `$1F93` without
touching the render buffer: 1568..1858 cycles, now counted from state by
`relative.update_object_on_screen_cycles` and cycle-exact against the ROM's own
`$209B`/`$1F9F` on ls0042/ls0335/ls9795, every occupied slot at every 8th player facing.
All 16 live rotations (1576..1843) are that branch, as is every rotation the headless
enemy driver takes over 2000 rounds on ls9795 and ls0335. Every `$1876` exit spends it —
a turn's redraw, a drain's and a discharge's — and all three are now counted, not meaned.

**What is left is a `plot_world`, not a redraw.** When the object *does* have a span,
`$1FC2` re-points the camera at the strip (`$09C0,X += $0C62/2`, `$001F` the fine angle)
and `$1FFC JSR $2625` replots it: 0.40..0.85 M cycles on ls9795 against 0.76..2.67 M for
the same board's whole screen, i.e. 20..40 frames of foreground for one rotation. It is
a function of the strip's own scene, not of the player's facing — four facings that put
the same object at four different columns cost the identical 841221 cycles, because the
camera shift cancels the column shift. That branch is real (the ROM only suppresses it
mid-replot, at `$1AF4`/`$1B00`, where an on-screen object aborts the whole enemy update)
and it is reachable at 8% of facings on ls9795 — about four replots per 3000 frames on
each of ls0042/ls0335/ls9795 at a facing that puts one enemy on screen, i.e. some 4% of
the clock spent where the flat 1723 spent none.

It is now charged, by `projector.strip_replot_frames`, at the camera `$1FC2` shifts to
(`$09C0 += $0C62/2`) rather than the player's own, and through the strip's own buffer
window: `$1FE5 JSR $29C7` halves `$0C69` into `$0007` and folds that into `$0012`, exactly
as `$2993` does from its table, so `render_cost` takes the window as an argument.
`RENDER_COST_BACKEND=py65` runs the real `$1F9F` instead and is exact — 19..29 frames on
the three boards — but costs ~1.3 s per uncached call, so the default is the windowed
proxy. The window matters: without it the proxy priced all 40 columns, ran 1.3..2.8x dear
and the clock over-stalled (ls335 grew a second facing error of the wrong sign, which the
exact backend did not). With it, proxy and exact agree on the error structure.

**What it bought, and the one it cost.** Over the ls335 human replay the modelled facing
errors fall 40 -> 37 and 35 strip replots fire. Thirty-six of the 37 are still the old
one-sided "+1 rotation"; the thirty-seventh is new and the other way — span 13 slot 3
loses a rotation the ROM kept.

**That one is decided, and it is not `$1F9F`.** Both cheap alternatives are excluded on
the recorded state by running it: `$0C4D` bit 7 is clear, so `$1FEF` does not divert to
`$8533`, and `$0C1F` bit 7 is clear, so `$1B00` hands `$1AF4` a set carry and the update
runs. Stepped round by round with `$1F9F` live, the ROM reaches `$1FFC JSR $2625` on
**slot 4** — the very enemy the model picks — and pays 728868 cycles, 37.1 frames. `$209B`
agrees with the model on all seven occupied slots of that state, cycle for cycle
(`test_the_rom_really_replots_the_enemy_the_overshoot_blames`).

What differs is *order*. Slots 3 and 4 enter the span with the same `$0C28` of 24, so they
come due together, and the descending `$0090` cursor decides which is considered first.
The ROM rotates slot 3 at round 68 and only then slot 4, with its stall, at round 75. The
model reaches slot 4 first, at frame 90 of 111, and the stall runs the span out: slot 3 is
never considered again and ends holding `$0C28` = 1, due and unserved. That is the
frame-to-round cadence this item already carries, one round wide, now with a 30-frame
lever on it. `test_every_facing_error_is_exactly_one_extra_rotation` caps the overshoot at
one so a later fix cannot quietly widen it.

**NOT the frame budget.** `registers_get` returns the VIC raster line (id 53) and the cycle
within it (id 54), so `line * 63 + cyc` is an exact intra-frame stamp and the handler can be
timed directly: `$95E9` entry to the `$969A` exit, over ls42/ls335/ls9795. A frame takes
five raster interrupts — the `$9589` split table programs `$D012` for lines 53, 93, 133, 173
and 213, and only the line-213 entry passes the `$961E` compare into the `$9630` body. Every
short entry is `SHORT_IRQ` exactly, and the body is `IRQ_BODY` **plus the four badlines its
own window (lines 213..~255) encloses plus the `$130C` the model bills to the foreground** —
which is what the 2683..2705 / 3079..3153 spread in `full_9630_wall` is. The **+9 within each
of those clusters was real**, and it is the note tick above; `$119F` is a flat 2156 counted on
the image and its live 2291/2334 wall is that plus the four or five badlines its own window
happens to enclose. Two counted cycles were genuinely missing and are now charged: the one
split entry a frame whose `$9603 BPL` wraps the index costs 1 more (the fixture's own
`short_wall` histogram is 3:1 over 119:120), and the cooldown walk's last byte leaves by an
untaken `$1329 BPL`.

**NOT `$3470` either.** `start_tune` ends in `JMP $FFF1`, and with the KERNAL banked out
(`$01 = $25`) `$FFF1` is the game's own RAM: the image carries `JMP $8D81` there. Counted on
the image it is 323 for tune 0 (a rotation) and 1 (a meanie's turn) — the same number the
live rotation measures — and **431** for tune 5, the one a drain starts at `$1A1F`, which
was charged 323.

**Not `$0078`.** `$0D4A` does **not** clear `$0078` before round 10: `$0DF1 ROR $78` rotates
the previous call's residue out into the carry that `$0DF3 ROL A` consumes, so bit 0 of the
incoming `$0078` can change the quotient bit, the interpolation and the angle. Both twins
start it at 0. It is live in principle — `$0F9E` and `$1D9D`/`$1DC1` write `$0078` too — but
over the natural 8x64 scan on ls42/ls335/ls9795 the sim matches the machine cycle for cycle
and byte for byte, so it does not bite at these states.

**Not the atomic body.** It was, and it is fixed; the model still reaches the wrong pass,
which staging inside the body cannot correct.

**Not the resync's clock.** Carrying the sim's cycle residual across a follow-mode resync
instead of resetting it moves ls9795 415 -> 397 only; it is the resync's *position* that
mattered, and that is now read off the stack frame.

**Not the discarded replot either — and what its resume still owes.** A resync inside
`$1FFC JSR $2625` used to start the sim at a pass boundary, so the model ran a whole frame
of passes against a machine that ran none, every frame until the replot finished: 80 of
ls9795's 142 events. `projector.replot_owed` now splits the pass at the row `$0026` and the
tile `$0025` the machine has reached and the sim owes the suffix. Two residuals are left,
both measured against two whole captured replots (21 frames each, sampled every frame):

* **Sub-tile position is not readable.** The split is exact at a tile boundary and flat
  inside one, so the predicted remainder tracks the machine as a sawtooth: 0 error at each
  tile the machine enters, drifting to +3.4 f by the time it leaves a near-row tile that
  costs 3-6 frames on its own. At the frame the instrument actually resyncs on — one frame
  into the replot — the error is **-0.19 f** against the machine's 21. Going finer needs the
  polygon inside `plot_tile`/`$8533`, which is control flow, not a variable.
* **The pass total is now within 1% of the machine's own, and slightly short.** Off the
  raster clock (`registers_get` 53/54 plus the `$9630` counter), three ls9795 replots measure
  412912/413054/416845 wall cycles over 21 frames, i.e. **328030..331963 foreground** cycles
  once the frame's 4042 IRQ and badline cycles come out. `render_cost` read 412893 for the
  same pass — 1.26x, two exit events a replot — and with `$37F2`, the `$2993`/`$245B` context
  and the `$1FA4..$1F9E` line all counted it reads **332100**, so the debt now clears just
  *early* and each replot leaves exactly one. What is left is the `$2625` area fill,
  [5](#5-one-object-vertex-angle-is-ten-units-out).

**Not the resync.** Carrying the sim's clock across a follow-mode resync instead of
resetting it moves ls9795 415 -> 397 only.

**It is NOT the sound engine.** An earlier revision blamed the tune player. Live, `$0CEB`
never leaves `$80` and `$0C73` never leaves 0, so `$34BA`/`$352C`/`$347D` are constant at
13/29/27 -- exactly what the model charges -- and `$34DE play_music` never runs. `$3470`,
which does vary, is not the tune: it is the rotation's own effect, 323 cycles, and it is now
charged inside `ROTATE`. The SID is in any case a pure cycle sink: there is **no read of
`$D400-$D41C` anywhere in the image**, by any addressing mode.

**Inert by tier.** `prng[0..4] $0C7B..$0C7F` (SWEEP) still diverges at frame 1 on every
board: the prnd result is stored 433 cycles into `$16B5` and the model applies it at the
same point in the *sequence*, so every consumer draws the same values, but a RAM snapshot
taken mid-prnd lags by one step and no RAM image can seed that. They are **not**
unconditionally inert: the PRNG feeds `put_object_in_random_tile_below_z $1238`, so a
discharge or hyperspace drawn on a drifted stream lands on a different tile. On ls42 no
such draw fires in 3000 frames, which is why it stays clean.
`fov_relative_h $0C57` (SCRATCH) is written at exactly one site, `$8425`, inside
`calculate_relative_angles`, which every visibility query runs. Four of its six readers are
inside that same call chain (`$18BE` the FOV gate, `$170E`/`$172A` the meanie rotation,
`$84ED`); the other two, `$20C6` and `$20E6`, are in `update_object_on_screen $1F9F` and
compute an object's screen x into `$0C62`/`$211C`. Nothing reads it before `$8425` has
rewritten it, and the plotting readers touch no field the schema carries.

**Measured — the write placement was a real defect, and it was not the gate's.** With every
CORE write committed on the ROM's own cycle, `--frames 3000 --follow` reads ls9795 **19**
(was 18), ls0335 **14** (was 14) and ls42 **0** (was 0). Every survivor is still an enemy's
`update_cd` reading 1 on the machine against 4 in the sim, and the interrupted PC says why:
the model has already run that enemy's body when the frame ends and the machine has not.
Classified by the `$95E9` chain, ls9795's 19 are 8 with the machine inside an `$1887` march
(`$1893`, `$189D`, `$1916` x2, `$191D`, `$1D16`, `$85C4`, `$17B7`), 3 inside `$1AB0`
(`$1AB5` x2, `$1ABE`), 4 inside the `$1F9F` replot's `$9730` flush (`$9786`, `$978D`,
`$988E`, `$9899`), 3 on the body's own straight line (`$16F4`, `$17B0`, `$17B2`) and 1 in
the prnd (`$31D8`); ls0335's 14 are 9 under `$1887`, 2 under `$1AB0` and 3 on the straight
line. That distribution is unchanged by the fix, which is the point: where a write lands
*inside* a segment is now exact, so what is left is when the model **arrives** at the
segment at all.

**Resolves.** Not the frame budget, not the clock, not the replot's placement, not the march
price, not the write's placement inside its segment and not the frame origin `b` — all now
measured directly against the machine. Not the seed either: it starts at the machine's own
cycle on every board. What is left is a per-window refund that names the instruction the
**machine** is running when the window falls, rather than an offset into the model's own
composite term: its mean is the machine's to 0.25 cycles and its per-frame value is
uncorrelated with the machine's, which is the whole +/-100-cycle band the surviving gate
decisions sit in. The replot's *price* — both backends 0.90..0.92 of a machine wall measured
at 22 frames, and ~2% cheap on the ls9795 pass measured above — is
[5](#5-one-object-vertex-angle-is-ten-units-out)'s.

## 9. The human line does not replay to a win through the live executor

**Wrong.** A recorded human win does not replay to a win through the driver.

**Measured.** `ls42_truth.json`, a live asid-vice replay of the recorded human action line,
records `reproduced: 26` of `n_events: 42` with `won_at_step: null`; `replay_human` sets that
field only on `$0CDE` bit 6. There is no offline control: only ls0 replays completely, and
ls42/ls335 are pinned to a first-divergence floor of 14 and 19 actions (`test_human_replay.py`).

**Resolves.** Items 3, 4 and 7 are the candidate causes on the driver side; a sim-side replay
that reaches a win would separate an executor-timing limit from a forward-model one.

## 10. Landability filter: unproven corners

**Wrong.** Two soundness claims are handled but not proved.

**Measured.** Alias landings (the 8-bit z compare wrapping) are kept unconditionally as
wildcards (`crossing_mask`'s `wrap_z` branch), so the answer is sound either way, but they are
not proved unreachable: band rays do exceed the alias distance at a non-origin cell. The filter
also inherits `los._tile_arc_indices`'s superset claim; `test_landtable.py` would catch a
violation but it is not independently proved. Both are keyed to `landtable.MAX_STEPS = 6000`,
still an explicit argument, so a caller marching further must re-query with the same cap.

**Resolves.** A reachability proof for the alias distance on a real board, or an independent
proof of the arc bisection's superset property.

## 11. Gaze-entry double penalty is not a query

**Wrong.** Leaving a body under a gaze costs the continued draining and downgrading of the
abandoned body, but the players price only the body they stand in (`playerbase._player_window`).

**Measured.** `threat.py` carries no query over a vacated slot, and
`_gaze_window`/`_exposing_enemies` are never called on a transfer's source tile.
`phase_player._mount_holds` and `_drained_over` do simulate the span but judge it on the **new**
body's energy alone, so the abandoned robot's downgrade is simulated and then discarded.

**Resolves.** A threat query over the abandoned stack's own exposure, so a hop out of a gaze is
charged what it actually costs.

## 12. The hardest boards are unsolved, and mostly unfinished

**Wrong.** Against the 128 hardest landscapes of the 10000 the planner wins 3, and two
thirds of the runs do not finish at all.

**Measured.** `sentinel.atlas` ranked all 10000 by equal-weight percentile of enemy count,
roughness, and climb (highest enemy z minus start eye); the top 128 are all 8-enemy boards,
roughness 0.402-0.551, climb 5.12-9.12. Every one was solved with a 180 s cap, 32 at a
time. Codes and per-board results are in `out/hardest_128.json` and
`out/hardest128_results.jsonl`.

| outcome | boards |
|---|---|
| won | 3 (ls8761 in 35 actions, ls7500 in 41, ls9544 in 48) |
| lost | 41 |
| did not finish in 180 s | 84 |

For scale, the eight boards the suite validates against rank 487th (ls335) to 9874th
(ls0), and none has 8 enemies. The table is that one sweep: only the 8 paralysed boards
below have been re-run since, so the 84 unfinished and the 41 lost are **not** current
measurements.

Three distinct failures sit underneath that, and they need different fixes:

* **Runtime, 84 boards.** Two thirds never reach a verdict. This is not a strategy limit
  and it dominates every other signal here; see
  [1](#1-a-per-tile-candidate-generator-only-pays-once-the-eye-is-up). 82 of the 84 had
  climbs available at entry, so they were playing, not stuck. A 24-board **sample** —
  every 5th code of `out/hardest_128.txt`, indices 0..115 — re-run at the same 180 s cap,
  24 at a time, went from **10 finished / 0 won** to **11-12 finished / 1-2 won** on the
  per-tile generators: ls9856 wins in 34 actions on both of two runs of the identical
  code, ls8926 wins in 57 on one of them and hits the cap on the other, having finished at
  152.7 s. Every board that finished on both sides took the identical action count. 24 of
  128 is not the 128, it was run 24 at a time against the table's 32, and a board landing
  within 30 s of the cap is a coin toss under that contention — the 84/44 split above
  stands as the only whole-set measurement.
* **Paralysis, 10 boards, was 18.** The planner takes zero actions. The 8 with no landable
  tile at entry at all now play: relocation is a move class
  ([architecture.md](architecture.md#hyperspace-death-and-the-win)), and re-run uncapped
  they are **ls8589 won in 47**, **ls7414 won in 81**, ls9785 lost 28, ls9364 lost 14,
  ls9795 lost 8, ls5301 lost 7, ls6725 lost 6, ls5916 lost 5 -- a win where there were 0
  actions. Only the two wins are re-measured under the current clock; ls7414/ls8589 read
  59/46, then 63/86, then 81/47 as the enemy-clock terms and splits of
  [8](#8-the-enemy-clock-commits-consider_enemy_states-core-writes-early) landed, so
  every action count on this page is a world model, not a policy. The other 10 *can* land somewhere but generate no climb candidate from it and
  are **not** re-measured here; that group shows the trigger is "no move I will commit to",
  not "nowhere to stand", so `_barren` is the predicate to widen next.
* **Genuine loss, 23 boards.** Played and lost. Only these are strategy failures, and they
  are the smallest class.

**Resolves.** The order is forced by the numbers: make a solve finish before judging
whether it wins. Until the 84 are resolved the win rate is a floor, not a measurement.

## 13. Arbitration charges an option with its continuation's mistakes

**Wrong.** `_arbitrate` scores each option on a fork with `rollout=True` — the fixed
breakout/supply/harvest/mount ladder — so an option is judged by a continuation the
arbitrating player will never take. When that continuation blunders, the death is attributed
to the option under test, and `ARBITRATE_ACTIONS` sets how many of its blunders get charged.
Depth therefore subtracts information rather than adding it, which is why the constant, and
not the aim price, decides ls373.

**Measured — the attribution, node for node.** At ls373 tick 0 the `_breakout` fork and the
winning line agree exactly: eye 5.875 E=7, then 6.875 E=3, then 7.875 E=1. At that third node
the fork's ladder dies within one tick; the arbitrating player absorbs `(14,3)`, reaches E=3
and wins in 45. `_breakout` is not fatal — the continuation is, three ticks later, at a node
the real policy handles. Horizon 2 wins by stopping before the fork reaches it.

Two corollaries, both measured:

* **Playing to termination does not help.** The fixed ladder makes the same blunder at the
  same node however long it runs, so `_breakout` played out loses in 15 — `_settle_tie`'s
  method (no horizon, consult only the outcome) would reject it too.
* **Policy-consistent forks do not help by themselves.** A `rollout=False` fork at depth 4
  also dies (E=0), because its own internal arbitration runs at the same horizon and repeats
  the error.

**Measured — off the suite.** Twelve boards outside both the suite and the hardest-128: of the
four that both horizons finished, horizon 2 is shorter or equal on every one (35 v 59, 36 v
41, 28 v 28, 32 v 62). Deeper search produced longer solutions on boards it was never chosen
against, so the effect is not a fit to the eight.

**Measured — not the safety probes.** This item previously read that the probes rest on an
over-priced aim (`playerbase._cheapest_ray`, mean 9 f and up to 141 f cheaper per action, took
ls373 from a 65-action win to a loss in 13). That diagnosis does not survive measurement.

* One probe carries the board: `_landing_holds((8,8), 0)` inside the tick-0 `_breakout` fork,
  at stance (14,3), eye 6.875, energy 6. Forcing that single call to False and changing
  nothing else wins ls373 in **63 actions**; banning the fork's other new landing, (14,3),
  changes nothing at all.
* That landing is sound and every accurate model accepts it. Run in ROM order the body
  reaches (8,8) with energy 1, and (8,8) is unexposed — no drain lands after the transfer.
  The fork dies twelve actions later, at a different stance.
* The arrival stance owes *less*, and with mixed sign. Over the 14 climb candidates at the
  ls373 entry stance, a fresh aim from the arrival body's `$1BE0` facing at the shell it
  abandoned plus that absorb's settle is **176–627 f**, against the **268–682 f** `_hop_span`
  charges from the departure stance. Substituting it exactly leaves ls373 bit-identical
  (13 actions, 10706 frames).
* Five further probe models, none recovering ls373: window = build plus the cheapest exit
  priced from the arrival stance (20 actions); phase-ordered probe, source then destination
  (5); ROM-ordered purse, each create paid out of a purse the step's own frames have already
  drained (10); verdict "can still afford the exit" (13); verdict "did not bleed" (15).
* `ARBITRATE_ACTIONS` alone moves the verdict. 1 and 2 win ls373 in 45 actions; 3, 4, 5, 6
  and 8 lose it in 13. At 2 the whole suite wins; at 1 it is 6/8 (ls60 and ls298 lost). 2 is
  what ships, as a bound on how much of the ladder's error the score absorbs — it is not
  derived, and the mechanism above, not the number, is what needs resolving.
* Gating a climb's purse on `_affords(cost, _hop_span)` — the invariant `_affords` states it
  holds ("executor and search both gate on this, at the same instant"), which
  `_climb_candidates` does not — wins ls373 in 62 and loses ls110. It reaches (8,8) only by
  billing the source tile's drain clock across the ~1700 f the body spends at the destination,
  where nothing sees it. Billing each phase at its own clock instead wins ls110 in 49 and
  loses ls373. The two differ in which board they lose, not in a number.

**Resolves.** Attribution: when a fork dies at tick *j*, ask whether an alternative at *j*
survives before charging the death to the option under test. Equivalently, a continuation at
least as strong as the policy that will follow — which nesting would give and `rollout` exists
to prevent (4 options × N ticks squared per decision), and which scoring instead of playing is
disproved below. Until one exists `ARBITRATE_ACTIONS` bounds the damage and no probe change
can be credited with the eight-board result.

## 14. The stance-base convention double-counts a robot's eye

**Wrong.** `actions.create` gives an object on bare terrain `z_frac = $E0` (`$1F66`), so a
stored z **already** carries a robot's eye fraction. `phase_player._stance_base` returns a
*foot* for a bare tile but `_base_z(top) + BOULDER_H` — already a robot eye — for a stacked
one, and `_climb_candidates` adds `ROBOT_EYE` to both. So on any tile already carrying a
boulder or platform the predicted eye is a whole eye high: `gain` is overstated, `k` is
under-computed, and a climb that actually *descends* can rank first. `_mount` has the same
bug on a robot's own z, so it can transfer downward — the overstatement, 0.875, exceeds
`EYE_EPS` (0.1) by enough to select a body up to 0.775 *below* the current eye.
`playerbase._robot_eye_after_boulder` gets the convention right, so two functions in tree
disagree.

**Measured.** Built and compared on ls42, tile `(4,4)`, terrain height 6: what the ranker
predicts against the actual eye of a robot built there
(`test_the_predicted_climb_eye_is_the_eye_a_robot_built_there_gets`, a strict xfail).

| boulders already on tile | k | predicted | actual | error |
|---|---|---|---|---|
| 0 | 0 | 6.875 | 6.875 | 0 |
| 0 | 1 | 7.375 | 7.375 | 0 |
| 1 | 0 | 8.25 | 7.375 | **+0.875** |
| 1 | 1 | 8.75 | 7.875 | **+0.875** |
| 2 | 0 | 8.75 | 7.875 | **+0.875** |
| 2 | 1 | 9.25 | 8.375 | **+0.875** |

**Measured — why it is not simply fixed.** One convention (`_stance_base` returns the eye a
robot built there would have now; `_climb_candidates` drops `ROBOT_EYE`; `_mount` reads the
robot's own eye) **loses ls373** — 45-action win → dead in 17, E=0 — and that is the only
suite board it loses: ls0 16, ls42 35, ls60 44, ls110 51, ls298 32, ls321 35, ls335 55 all
still win. Off the suite it is an improvement: of the eight formerly paralysed boards
(item 12) it takes ls9785 from lost in 28 to **won in 95** and ls7414 from 59 to 54 actions.
Reverting the geometry alone, with the relocation, harvest and fork fixes left in place,
restores ls373 exactly (45 actions, 17403 f) — so the arithmetic is the sole cause.

**Resolves.** One convention across `_stance_base`, `_climb_candidates`, `_mount` and
`_robot_eye_after_boulder`, with ls373 *diagnosed* rather than traded away. The fix changes
which climb ranks first, and item 13 is why a different first climb can lose a board the
planner otherwise wins — so that attribution is the blocker here too.

## Disproved — do not resurrect

Each is a hypothesis and the measurement that killed it.

- **ls373's lost win was a safety-probe margin the aim over-charge was supplying.** The one
  probe that decides the board accepts a landing that is genuinely survivable, six probe
  models leave it lost, and the verdict moves with `ARBITRATE_ACTIONS` alone
  ([13](#13-arbitration-charges-an-option-with-its-continuations-mistakes)).
- **A `body_spent` resume would cut the follow-mode event count.** Over the gate's own
  3000-frame races, **0** of ls9795's 111 and **0** of ls335's 12 CORE events catch the
  machine inside `$1CDD`; 84 of ls9795's catch it inside the `$1FFC JSR $2625` replot
  ([8](#8-the-enemy-clock-commits-consider_enemy_states-core-writes-early)).
- **The `$1887` march that decides ls9795's frame 129 is ~1% under-priced.** On the live
  image captured at that very `$16E6`, jennings and `enemies.update_body` both give
  **280640**; the real 6510 matches jennings at all 89941 instruction sites; and the
  march's own frame budget measures 15166..15591 against the model's 15174..15591. The
  model's whole lead is **+81 to +97 cycles** ([8](#8-the-enemy-clock-commits-consider_enemy_states-core-writes-early)).
- **A relocation needs an energy floor above the ROM's own 3.** Floors of 5 and 8 leave every
  board that matters bit-identical — ls7414 won in 59 and ls8589 won in 46 at 3, 5 and 8;
  ls9364 lost 14 and ls6725 lost 6 at all three — and are strictly worse on two already-lost
  boards: at floor 8 ls9795 and ls5916 cannot afford the jump, stop acting after 4 and 1
  actions and idle to the cap alive, which is the paralysis the move exists to end.
- **The transfer settle over-charges systematically.** The measurement was clipped by the 6 s
  wall-clock `run_until_pc` in `tap_action`, which caps a reading at ~300 frames.
- **Correcting the settle's viewpoint reduces it.** It moves the settle **up**.
- **Aim mispricing is secondary.** It is the larger term, and the cause was a driver defect — a
  swallowed sights toggle burning a whole aim's worth of frames — not a missing cost term.
- **Rank fixes by *cumulative* frame drift.** Net drift at the failing step was small while that
  step's own phase was badly out; the cumulative figure hides it.
- **`HOP_FRAMES` under-budgets every hop 2-3×.** Live hops run modestly over it, not multiples
  over; replacing it with the computed budget took the live player to zero actions.
- **A hop is expensive because the tile is far away.** Aim cost is angular, not spatial: over the
  23 landable tiles at the ls42 start `corr(aim, manhattan)` is **−0.54** against +0.60 for pitch
  notches, and the fatal ls42 build's cost was almost entirely pitch, not bearing.
- **The climb ranker only needs the `$F5` up/level pitch plane.** A pedestal is aimed at by its
  TILE, routinely *below* the eye even when the robot on it is not.
- **Meanie spawn location is PRNG-driven.** `$197D` never touches the PRNG.
- **Enemy freeze under `plotting=True` is a fidelity knob.** It freezes enemies outright.
- **Asking only the best-scoring climb group.** `_best_climb` reads one score group, so the
  generator could stop at the leading landable one; the tallest tiles are the least landable,
  so it walks 46-98 of the 66-99 groups and 596-930 of the tiles anyway (ls42/110/335/373/9717
  entry states).
- **A `_settle_tie` fork re-sweeps the board it forked from.** `_VIEW_CACHE` keys on
  `projector.scene_key`, so a clone of an unchanged board hits the memo: 0 of 45 sweeps over an
  ls110 solve are at a tie's entry key ([1](#1-a-per-tile-candidate-generator-only-pays-once-the-eye-is-up)).
- **A K-pruned landability table.** The first landing hit sits deep inside a candidate row, so a
  stored-first-`K` row decides almost no landing query at any affordable `K`: at K=64 it costs
  tens of MB per lattice, falls back on nearly every query and is no faster than the exact path,
  and a `K` large enough to decide the row would be terabytes. `crossing_mask` needs no storage.
- **Score the phase player's tie instead of playing it.** Three scorings, each worse than
  rollout's 7/8 (8/8 before the aim price was made exact): absorb only when its value exceeds
  the drain over its own span (6/8, ls321 and
  ls373 lost); make `_supply`'s affordability test agree with `_best_climb`'s (6/8, ls42 and
  ls110 lost); route every climb over the board's exact stance geometry (5/8).
- **`_refuel` harvests at a loss under a cone.** It cannot: idling through the same span costs
  the same drain and yields nothing, so an absorb under fire is never worse than waiting.
- **The ls335 facing gap (item 8) is a wrong or missing enemy branch.** The rotate gate `$17FB
  LDA $0C28,X / CMP #$02 / BCC` fires exactly when `enemies._rotate_enemy` does. Every earlier
  `$16E6` branch was checked against the trace it would have to leave: no meanie object exists
  in either the recording or the sim; 39 of the 43 errors have `drain_cooldown == 0` on both
  sides; the model already downgrades boulders/trees *more* often than the ROM (9 vs 5, and 0/28
  correlation with the bad spans); only 7 of the 28 miss a discharge tree. `$1B00`, the one
  omitted branch, is a no-op on the common path. Nor is it one missing delay: the margin by
  which the model fires early spreads over two orders of magnitude in frames.
