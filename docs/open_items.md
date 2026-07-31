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

## 5. Terrain fill cost cannot close per tile

**Wrong.** `projector`'s fill term is an area proxy (`PER_SCANLINE*H + PER_PIXEL*H*W`) whose
residual is systematic in scene busy-ness.

**Measured.** Mean error changes sign and grows across measured-cost quartiles, tracking how
busy the scene is. It cannot close per tile: the `$AD00`/`$AE00` edge tables carry state across
polygons ([render cost](architecture.md#render-cost-projectorpy-pancostpy-rendercost_py65py)),
the filled-rows/y-extent ratio spans both sides of 1, `H` is 0 on views where the ROM still
fills heavily (every corner `screen_y` below the inner band), and per-tile fill spans nearly two
orders of magnitude.

**And the exact backend is itself dear on a live state.** One `$1FFC` strip replot caught on
the machine at `$1F9F` (ls9795, slot 3, the ROM's own span 37/3) takes **22 whole frames** to
the next `$1289`. On that captured image the proxy charges 387392 cycles, 1.12x, but
`RENDER_COST_BACKEND=py65` charges **534090**, 1.54x — which the machine cannot have spent in
22 PAL frames (24277 a frame against 19656). Calibrating the proxy against that backend
inherits the error. Fixture `strip_replot`, measured from
[8](#8-the-enemy-clock-is-not-the-residual-the-replots-frame-is).

**Resolves.** Stateful emulation of the whole `plot_world` fill sequence in render order, the
interleaved object polygons included. Short of that the only honest levers are
`projector.PER_SCANLINE`/`PER_PIXEL` — and first, the `$2993`/`$245B` context
`oracle.update_object_cost` builds, since that is what "exact" is being measured against.

## 6. The py65 exact backend skips transfer settles

**Wrong.** `RENDER_COST_BACKEND=py65` falls back to the proxy on the whole transfer path.

**Measured.** `projector._exact_render_cost` returns `None` for any `observer != state.player`,
and a transfer settle is always priced from the post-transfer slot (`playerbase._settle_eye`),
which is never the player's own at pricing time.

**Resolves.** Running the py65 backend from an arbitrary observer slot.

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

## 8. The enemy clock is not the residual: the replot's frame is

**Wrong.** `driver.instrument --frames 3000 --follow` still reports CORE divergences on
ls9795 (**111** events, the first at frame 129) and on ls335 (**12**, the first at 478).
ls42 is clean: **0 over 3000 frames**. Every event is an enemy's `update_cd` reading 4 in the
machine where the sim still reads 1 — one `$16ED` reload the sim reaches a frame late — or
the `$1805` rotation that follows from it. Everything this item has blamed in turn is now
measured, and none of it is the cause.

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
| badline steal | `BADLINE_FRAME` 1071 | 1072.91 | 1073.96 | 1073.93 | 1075.20 |
| foreground | the rest | 15472.54 | 15459.24 | 15456.96 | 15649.27 |

The short interrupts are **exactly** 477 in every one of 500 frames. The body is within
**one cycle** of the model on all four captures: `$FFC5` a flat 56, `$1635` taking its
25-cycle fast exit every time (so `IRQ_SPRITES` never fires), `$FFC2` exactly
`SOUND_TICK_IDLE` 63 in all 80 frozen frames, and no `$130C` and no `$1635` at all when
frozen — the `$9659` gate, as `IRQ_GATE_SHUT` says.

So the whole frame-budget error is `BADLINE_FRAME`, at **1.9 to 4.2 cycles a frame**. That
is also the one term that cannot be counted off the ROM: a badline steals 40..44 by where
in its instruction the CPU is, so its frame total is a property of the code mix — 1072.9
over ls42's cheap passes against 1075.2 over ls9795's frozen idle loop. 1071 is the
frozen-rate fit, and replacing one mean with another mean would buy nothing.

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

**Boundary — that lead is the badline steal, and it cannot be counted.** Per frame,
19656 = foreground + 477 + the `$9630` body + steal, and the first three are counted off
the image, so the residual is the steal. Live during the march, the per-badline excesses
are 41x1, 42x9, 43x84, 44x6 over 100 badlines — **1073.75 a frame** against
`BADLINE_FRAME` 1071, in a frame with `$D015` = 0 and therefore no sprite term. A badline
costs 43 less however many of the three BA-window cycles the CPU spends writing, so its
frame total is a property of *which instruction the raster caught*, and a budget model with
no raster position cannot count it. 1071 is a frozen-rate fit and 1075 (25 x the 43 mode)
would be another; either way the frame-129 rotation is decided by ~100 cycles accumulated
over 129 frames, which is 0.8 a frame. **Zero divergence on ls9795 therefore needs the
badline steal per frame, i.e. raster-accurate accounting, not a better cost term.**

**Measured — the length, and a backend that is dearer than the machine.** Caught at
`$1F9F` with a stopping checkpoint, that replot takes **22 whole frames** to the next
`$1289`. `strip_replot_frames`' proxy charges 387392 cycles = 24.6 frames of stall,
**1.12x** — inside its own 0.93..1.21 band. Its `RENDER_COST_BACKEND=py65` backend, on the
same captured image, returns **534090** cycles = 34.0 frames, **1.54x**. The machine cannot
have spent 534090 cycles in 22 PAL frames — that is 24277 a frame against the frame's
19656 — so the *exact* backend's render context is dear, and the proxy calibrated against
it inherits that. Both belong to [5](#5-terrain-fill-cost-cannot-close-per-tile), not here.

`~17 cycles a frame` was therefore never there. The budget is right to under 4 and the
clock to under 0.01 of a pass a frame; what is left is a render cost and its timing, plus,
for follow-mode resyncs only, a marker that catches the machine mid-`$1887`.

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

**Measured — the body is now exact.** Stepping `$16E6 consider_enemy_state` one round at a
time against the same oracle, comparing **every** round the play loop dispatches, is
cycle-exact on ls42, ls335, ls9795, ls0, ls60, ls110, ls298 and ls373 — gated, marching,
rotating, held-target, draining and discharging rounds alike
(`test_the_body_cost_model_matches_the_roms_own_16e6_cycle_count`). Getting there priced
five regions that were means or were not charged at all:

* **`$16E6`'s own line.** `CONSIDER_ENTRY` folded the `$16FA` meanie branch,
  `CONSIDER_PREAMBLE` folded the `$1782`/`$1798` gates *and* the scan init, and `SCAN_FIXED`
  charged that init a second time. `SCAN_SLOT` was one shape of five (22 hidden, 27 unseen,
  25 full, 34 another robot's head, 36 the player's). `$17CD..$17E8`, the held-target line
  `$1795..$17A9`, `target_object $1825` and the `$1876` redraw tail were charged **nothing**.
* **`$1AB0`.** Its entry and its exhausted exit were one constant, the last slot's untaken
  `$1AF0 BPL` was overcharged, a boulder top's `$1ADF` compare was priced as a tree's, and
  the hit exit omitted `$1AE6 LDA $14` and its own `RTS`.
* **`reduce_object_energy $1A08`.** Every drain was free. It is now charged per target kind,
  with `$1AF4`'s update gate, `remove_object $1EEF` (86 on a stack, 94 on the ground) and
  the `$1A4F` energy bank.
* **`plot_status_bar $9508`.** A player drain replots the bar. It pads to fixed columns, so
  its whole cost is a function of the energy byte the drain just wrote — exact over 0..39.
* **The discharge.** `DISCHARGE_FIXED` 100 and `DISCHARGE_TRY` 966-a-draw were means over
  `$211D create_object`, `$1238`'s tile hunt and `$1F16 put_object_in_tile`; each is now
  charged per slot walked, per lap by the test that lap failed, and per `$1272` draw (445,
  plus 440 for each draw masking to `$1F`).

**NOT the pass line either.** The same raster clock times one whole play-loop pass live:
consecutive `$1289` hits, less the interrupts the window contains and the badlines outside
them, against `passcost`'s own model of that pass. On ls42, ls335 and ls9795 the model is
**exact** (ls9795: 222 of 299 consecutive passes exact, the rest a one-badline or
one-interrupt edge in the probe's own window arithmetic, plus a ±2 stamp jitter). The head,
the dispatch, the prnd, the cursor, the `$12A2` tail and the `$191F` exposure walk are
therefore all right as charged.

**The seed, not the clock.** The model's clock is not biased: over 70 live ls9795 frames the
machine's `$16E6` hit count and the sim's own pass count agree to **±1 pass** at every
sample (1044 passes). What the sim could not do was start where the machine was — a `$9630`
halt catches the loop at a raster position, and 65% of ls9795's land inside `$16E6`. That
position is recoverable: the `$95E9` frame holds Y, X and the interrupted PC from SP+1, under
it the foreground's own return addresses, and `$1887` saves its caller's X and Y at `$191E`
and `$0C58`. `enemies.resume_from_stack` maps it onto the model's own resume points and the
instrument applies it at the seed and at every resync.

**Measured — same seed, same machine, two sims.** One restarting the interrupted pass, one
resuming it: on ls9795 the first CORE divergence moves **66 -> 129 frames**, and on ls335,
where that seed happened to land on the pass head, both give 155 — the control.

**And mid-segment, not just mid-pass.** Resuming at the interrupted segment's *head* still
lost what the machine had spent inside it — up to 433 cycles in the prnd, a whole `$191F`
walk in the tail. `resume_from_stack` now also returns the signed cycle offset between the
machine's position and the model's resume point, counted off the ROM's own straight lines
(`$1289` through the `$16B5` dispatch for both enemy types, `$16D9`'s cursor step both ways
round its wrap, `$12A2`'s tail with its exposure walk and three sound bodies, and `$31CA`
per LFSR lap off the interrupted Y), and the instrument hands it over as the opening
`cycle_residual`; every entry is checked against the ROM running that stretch. A position
off those lines resumes at its segment head as before, so the seed waits up to 40 frames
for a marker that catches the loop on one it can count. ls9795's first divergence
**66 -> 129**, ls335's **155 -> 194**, follow-mode events on ls335 **63 -> 31** with its
median gap **19 -> 86** frames; ls42 stays at 0 throughout.

**Where the seed work stopped.** Two things, both named by measurement rather than suspicion
(the two paragraphs after this one move ls335 on again):

* The frame-129 ls9795 event is `obj[3].h_angle` with `enemy[3].rotation_cd` — a **rotation**,
  and the sim takes it a frame *early*: it reads `rotation_cd` 200 already reloaded where the
  machine still reads 0. A rotation's own cost cannot make it early, and the redraw is the
  wrong lever by two orders of magnitude (above), so what is early is the phase that reached
  it.
* ls335's first event was then frame 194, a single `enemy[2].update_cd`: still one pass of
  phase, not a wrong cost. What is left of the seed is the **body** — a marker that catches
  the loop inside `$16E6` resumes at the interrupted `$1887` query's start, for the reason
  below.

**It WAS the `$9630` head: the note tick, and the badline mean.** `$963D JSR $FFC2` vectors
into the sound engine's note tick `$8ED1`, and 63 is only its all-idle cost: a voice whose
`$8E86,X` note timer is counting costs **+9**, one that has run out **+5**, and the frame the
gate expires **+67**. Live on ls335 those four land 20/26/73/1 in 120 frames. The tick is now
charged per frame from its own branches (`sound_frame`, exact against the ROM over 300 random
voice states), the three `$3470` sites reload the voice off the `$AC00` descriptor
(`start_tune`), and `IRQ_BODY` is **counted** off the image rather than fitted: 7 entry +
`$95E9` 81 + the `$9630` body 2275 + the `$969A` RTI tail 22 = 2385, with the `$9659`-gated
block (`JSR $130C`, `$1635`, and the branch itself) charged only when the clock runs — the
model was paying all 43 of it on frozen frames, and never paying the `JSR $130C` at all.
What that exposed: `BADLINE_STEAL` 43 is the **mode**, not the mean. With the clock frozen
(so every pass is the idle one) the live `$1289` rate fixes the frame's whole steal at
**1071**, 42.8 a badline; the model reproduces the measured pass count on ls42/ls335/ls9795
to under **4 passes in 50000**. Together: ls335's first CORE divergence **194 -> 478** and its
follow events **31 -> 12**, ls9795's **128 -> 112**, ls42 still 0 over 3000 frames.

`$963A JSR $FFC5` ($8F0C, the effect tick) stays inside `IRQ_BODY` at a measured constant 72:
it is gated on `$8E96,X` bit7 and no tune a play frame starts clears it. `$9633 BPL` is not a
branch in play either — `$0CDF` reads 0 at every marker, so `$9635 INC` runs every frame.

**The march resume is a FOLLOW-MODE fix only, and its unit is cycles, not steps.** Two
corrections to what this item used to claim. First, the seed is never inside `$1887`:
`instrument._exact_seed` waits for a marker whose position `_segment_offset` can count, and
every `PHASE_BODY` position returns `None`, so the *first* divergence on all three boards is
measured from an exact non-body seed and the march resume cannot move it. It moves the
**resyncs**, which is where ls9795's 1-frame median gap comes from.

Second, the resume unit is not a step index. The raster IRQ preempts the ROM at an
*instruction* boundary, not at an `$1CE8` sub-step boundary, so the position to reconstruct
is "cycles spent inside this `$1887`" and the model can occupy any of them: give `State` a
`body_spent`, have the query charge `min(cost, budget)` and suspend at the same
`(stage, index)`, and re-run it on resume charging `cost - body_spent`. The query is a pure
function of state that nothing between frames disturbs, so the re-run is exact; its only
memory effects are `$0014`/`$0C56`/`$0CDD`/`$0C76`, which must move from per-probe writes to
locals written once the query completes, or the re-run's `$1CDF LSR $0C56` shifts twice. That
needs no change to `sentinel/los.py` at all.

What it does need is `c`, the cycles the machine has already spent, and that IS recoverable:
`$001E` is the probe counter, `$0C58` the target, and the ray's own accumulators sit at
`$0034..$003C`. `check_for_line_of_sight_to_tile` already takes `max_steps` and leaves the
marched position in its `Vector`, so a **binary search on `max_steps`** against the machine's
`$0034..$003C` finds the interrupted sub-step and reads its cycles off the same call — no
instrumented march, no step index threaded anywhere. **Not implemented.**

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
of those clusters was real**, and it is the note tick above; `$119F` is a flat 2162 counted on
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

**Resolves.** Not the frame budget, not the clock, not the replot's placement and not the
march price — all four are now measured directly against the machine and none has room for
what this item is chasing. What is left on this side is the **badline steal per frame**,
which needs the raster position and the instruction it caught, and a `body_spent` resume so
a follow-mode resync inside `$1887` does not restart the query. The replot's *price* —
proxy 1.12x and py65 1.54x of a machine wall measured at 22 frames — is
[5](#5-terrain-fill-cost-cannot-close-per-tile)'s.

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
  [8](#8-the-enemy-clock-is-not-the-residual-the-replots-frame-is) landed, so
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
- **The `$1887` march that decides ls9795's frame 129 is ~1% under-priced.** On the live
  image captured at that very `$16E6`, jennings and `enemies.update_body` both give
  **280640**; the real 6510 matches jennings at all 89941 instruction sites; and the
  march's own frame budget measures 15166..15591 against the model's 15174..15591. The
  model's whole lead is **+81 to +97 cycles** ([8](#8-the-enemy-clock-is-not-the-residual-the-replots-frame-is)).
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
