# Open items

Everything known to be wrong or unproven, each stated as: what is wrong, the measurement
that shows it, what would resolve it. Anything not here is either correct or out of scope
([architecture.md](architecture.md), "Not modelled").

## 1. The band sweep dominates runtime

**Wrong.** `los._landable_batch` — the full pitch-band lattice sweep — dominates a run.

**Measured.** Profiled on ls335: the sweep is the great majority of run time, and the band
lattice is 3,538,944 rays. A tie multiplies that by the number of tied candidates, which is
why a tie at the ls110 opening is the most expensive event in a solve.

**Partly resolved.** `_Views.get`/`band_get` now answer a single tile with one targeted
march (`playerbase._cheap_view` over `landtable.candidates`, narrowed by the closed-form
crossing filter `landtable.crossing_mask`) instead of a whole-lattice sweep, against the
board pinned at that lattice's first query: ls335 93 s -> 66 s, all eight boards
bit-identical. What still buys the sweep is the callers that iterate a whole view dict
(`views.band()`), and the tie rollouts that multiply it.

## 2. Live frame accounting leaks across a tie

**Wrong.** The frame counts a live run reports are not trustworthy across a tie-breaking
rollout.

**Measured.** Two frozen live ls42 runs take an identical action sequence but measure
different frames per step on the same creates and transfers; the same pair driven by the
greedy player is bit-identical. The VICE binary-monitor socket drops during the long think
gap a tie opens, and the reconnect leaks emulated frames into the measurement.
The live *win* is unaffected — it is verified by `$0CDE` bit 6 — and
`driver/test_live_determinism.py` gates the driver with greedy for this reason.

**Resolves.** Keeping the monitor connection alive (or halting the machine verifiably)
across an arbitrarily long think gap, so no emulated frames elapse unattributed.

## 3. The gaze forecast assumes rotation never stalls

**Wrong.** `playerbase._cone_onset` projects the fixed rotation step and cooldown cadence,
but `consider_enemy_state` returns *before* the `$17F9` rotate whenever the enemy is
discharging (`$177A`), has found a drainable boulder/tree (`$1773`), or still sees its
target (`$178C`) — so a busy enemy stops sweeping and its cone holds.

**Measured.** On the live ls42 line, the forecast body window at `robot (6,20)` is more than
twice the live one and overruns the budget
(`test_plan_dwell_prediction_matches_live_ls42`, xfail). It also makes waiting under a cone
non-terminating: one live ls110 run spent a long run of consecutive waits and took drains
throughout. Two consequences of opposite sign: a body already under a busy cone is exposed
*longer* than forecast, and tiles ahead of that enemy's arc become safe *later* than
forecast.

**Resolves.** Modelling the stall in `_cone_onset` and re-running the three live gates —
modelling it shortens every window, and the one attempt lost ls42. Worth pricing at the same
time: **abandoned objects pin an enemy's gaze**, since a sentry with something to drain
stops rotating, so the inchworm recycle un-pins it. The phase player sidesteps the forecast
entirely by simulating the span (`_drained_over`); this item binds the reactive player and
every window query built on `_cone_onset`.

## 4. Per-step frame drift, and the unattributed create/absorb settle split

**Wrong.** Charged frames drift against measured frames per step.

**Measured.** A winning live ls42 run under-charges over its 36 steps, reproducibly run to
run. Decomposed over 15 runs, the settle side is two constants the model merges: **a
create's measured settle is consistently longer than an absorb's**, against the single
charge `DITHER_FRAMES + POST_ACTION_REPLOT_FRAMES`. The ROM counter behind it is `$2099`
(`$1FA4` loads `#$19`; `$2051` loads `#$28` when `$0C4E`, the meanie-made flag, is set) —
that is the meanie split, not the create/absorb one, so the difference is unattributed. The
aim side is a separate over-charge, dominated by large pans.

**Resolves.** Finding the ROM path that makes a create's post-action settle longer than an
absorb's, and splitting the constant on it rather than on a fitted difference.

## 5. Terrain fill cost cannot close per tile

**Wrong.** The `span_fill` term of `render_cost` is an area proxy and its residual is
systematic in scene busy-ness.

**Measured.** The mean error changes sign and grows across measured-cost quartiles — it
tracks how busy the scene is. The blocking fact:
`polygon_left_edge_table $AD00` and `polygon_right_edge_table $AE00` are
**never cleared** between polygons, so a polygon clipping to a sliver writes only some of
the `[$0004,$0006]` rows and `span_fill` then reads **stale** left/right columns left by a
previous polygon (verified: a row's `$AE00` byte matched none of the current triangle's
three `$A7A0` values, only a prior polygon's). Middle-fill length is
`right_col - left_col`, so there is no closed-form per-tile fill: the filled-rows/y-extent
ratio spans both sides of 1, `H` is 0 on views where the ROM still fills heavily (every
corner `screen_y` below the inner band), and per-tile fill spans nearly two orders of
magnitude, so the residual is neither area- nor H-linear.

**Resolves.** A stateful emulation of the whole `plot_world` fill sequence in render order,
including interleaved object polygons writing the same two tables. Short of that, the levers
are `projector.PER_SCANLINE`/`PER_PIXEL` and the cross-polygon span coupling.

## 6. The py65 exact backend skips transfer settles

**Wrong.** `RENDER_COST_BACKEND=py65` falls back to the proxy on the whole transfer path.

**Measured.** `_exact_render_cost` returns `None` for any `observer != state.player`, and a
transfer settle is always priced from the post-transfer slot, which is never the player's
own at pricing time.

**Resolves.** Running the py65 backend from an arbitrary observer slot.

## 7. The driver's wall-clock timeouts are the residual load sensitivity

**Wrong.** `_RU_PAN` (20 s), `_RU_STA` (8 s) and `_RU_COMMIT` (4 s) in `kbd_aim` are host
clocks in an otherwise machine-clocked driver.

**Measured.** On an idle host `driver/test_live_determinism.py` passes (2/2 serial, and two
full ls42 runs are frame-identical); under a saturated host (`pytest -n auto` plus two VICE
containers) it fails on differing step counts. A monitor round-trip is orders of magnitude
slower at real-time pace than halted, so enough contention pushes a checkpoint wait past its
timeout and the aim re-drives.

**Resolves.** Since `$365D` recurs every frame, a timeout there means the game left the play
loop — it should be an error, not a retry.

## 8. The ls335 facing gap: a seven-enemy board diverges

**Wrong.** Replayed against the recorded clock, ls335 enemy facings run ahead of the ROM's.

**Measured.** 28 of the 117 exact spans miss a facing, 43 enemy-facings in all, and every
single one is **exactly one extra rotation step** — never an under-rotation
(`test_every_facing_error_is_exactly_one_extra_rotation`, so an over-correcting fix shows up
as −1). One- and two-enemy boards are clean (ls0 16/16, ls42 10/10 live), and the gap
survives re-recording by the checkpoint method, so it is not an artifact of the async
recorder. What has been ruled out:

- **The rotate gate.** `$17FB LDA $0C28,X / CMP #$02 / BCC` fires when the rotation cooldown
  is below the stick value, which is what `_rotate_enemy` does — the error is in *when the
  consideration happens*, not whether it fires.
- **Every earlier `$16E6` branch**, each against a trace it would have to leave: owns a
  meanie (no `MEANIE` object exists in the recording or the sim), held drain target (39/43
  have `drain_cooldown == 0` on both sides), drains a boulder/tree (we downgrade *more*, 9
  vs 5, 0/28 correlation), discharges its bank (only 7 of 28 miss a tree we should have
  made). `$1B00`, omitted from the model, is a no-op on the common path (`SEC / BIT $0C1F /
  BPL` returns carry set unless a visibility flag is set).
- **A single missing delay.** Measuring how early we fire, the margin spreads over two
  orders of magnitude in frames — no constant covers it.
- **The round logic.** Seeded with a divergent ls335 state, `enemies.step` is byte-exact
  against `oracle.step_enemy_round` for 119 rounds on every span we get wrong.
- **A constant cadence.** `$1289` calls `$16B5` once per main-loop pass, so cadence is
  passes per frame; sweeping `UPDATES_PER_FRAME` over 1..8 leaves facings at 89/117
  throughout, and driving below one pass per frame reaches only 92/117 at K=16. Sweeping a
  uniform updates-per-round `U` over 1..40 on the 28 bad spans, `U = 1` fixes 10 and no `U`
  fixes the other 18.

**Resolves.** The cadence must vary *within* a span, which is what the phase split models
(foreground `plot_world`/dither/scroll stretches reach no `$16B5`) — so closing the gap
means pricing those stretches better, not picking a better number. The phase split currently
scores 90/117 against idle-only 89 and plotting-only 64: a one-span margin, so it rests on
the replay floors and action counts, not on facings. One asymmetry survives unexplained: on
live checkpoint-sampled captures ls42's `$0C30` sits at 3-4 while ls335's sits at 1 — same
tool, same sampling position, different boards.

Two traps when re-running this comparison: `machine_from_image` overlays `$9D37` and `$1335`
from the ROM image (a seeded rotation-speed table is silently replaced, and the ROM then
rotates by the wrong step), and `prime_enemy_driver` resets `$0C50` and the cursor. Both
make a comparison look like a model defect.

## 9. The human line does not replay to a win through the live executor

**Wrong.** A recorded human win replays to a win in the sim but not through the driver.

**Measured.** ls42: 21/42 steps; the committed `ls42_truth.json` records 26/42 with
`won_at_step: None`. By contrast the ls335 human line replays to a win in the **sim** — with
the enemy clock recorded straight into the fixture (`watch_play/3`), all 123 genuine player
actions are feasible and aim-landable against the true clock and `sentinel.actions.win`
drives the recorded endgame (absorb Sentinel → build on platform (28,17) → transfer →
hyperspace) to `$0CDE = $C0`.

**Resolves.** This is an executor/driver-timing limit, distinct from the forward model.
Items 3, 4 and 7 are its candidate causes.

## 10. Landability filter: unproven corners

**Wrong.** Two soundness claims are handled but not proved.

**Measured.** Alias landings (the 8-bit z compare wrapping) are kept unconditionally as
wildcards, so the answer is sound either way, but they are not proved unreachable: band rays
do exceed the alias distance at a non-origin cell. The filter also inherits
`los._tile_arc_indices`'s superset claim; the validation harness would catch a violation but
it is not independently proved. Both are keyed to `max_steps = 6000` (an explicit argument),
so a caller marching further must re-query with the same cap.

**Resolves.** Either a reachability proof for the alias distance on a real board, or an
independent proof of the arc bisection's superset property.

## 11. Gaze-entry double penalty is not a query

**Wrong.** Entering a gaze costs >= 1 off the current body *plus* continued draining and
downgrading of the abandoned body, but the players price only the body they stand in
(`playerbase._player_window`).

**Measured.** The abandoned-body loss is realised only by actually stepping the enemies, and
never priced up-front; there is no threat-layer query for it.

**Resolves.** A query over the abandoned stack's own exposure, so a hop out of a gaze is
charged what it actually costs.

## Disproved — do not resurrect

- **"Transfer settle over-charges systematically."** It was the 6 s wall-clock
  `run_until_pc` in `tap_action` clipping the measurement at its own ceiling.
- **"Correcting the settle's viewpoint will reduce it."** It moves **up**.
- **"Aim mispricing is secondary."** It was the larger term, and a driver defect (a
  swallowed sights toggle burning a whole aim's worth of frames), not a missing cost term.
- **Ranking fixes by *cumulative* frame drift.** Net drift at the failing step was small
  while that step's own phase was badly out; the cumulative figure hid it.
- **"`HOP_FRAMES` under-budgets every hop 2-3×."** Live hops run modestly over it, not
  multiples over; replacing it with the computed budget took the live player to zero
  actions.
- **"The fatal hop is expensive because it is 12 tiles away."** Aim cost is angular, not
  spatial: over the 23 landable tiles at the ls42 start `corr(aim, manhattan)` is **−0.54**
  against +0.60 for pitch notches, and that build's cost was almost entirely pitch, not
  bearing.
- **"The climb ranker only needs the `$F5` up/level pitch plane."** A pedestal is aimed at
  by its TILE, routinely *below* the eye even when the robot on it is not.
- **"Meanie spawn location is PRNG-driven."** `$197D` never touches the PRNG.
- **"Enemy freeze under `plotting=True` is a fidelity knob."** It freezes enemies outright.
- **A K-pruned landability table.** The first landing hit sits deep inside a candidate row,
  so a stored-first-`K` row decides almost no landing queries at any affordable `K`; at
  K=64 it costs tens of MB per lattice, falls back on nearly every query and is no faster
  than the exact path, and a `K` large enough to decide the row would be terabytes. The
  closed-form `crossing_mask` needs no storage at all.
- **Scoring the phase player's tie instead of playing it.** Absorb only when its value
  exceeds the drain over its own span: ls110 won, ls321 + ls373 lost (6/8). Make `_supply`'s
  affordability test agree with `_best_climb`'s: ls42 + ls110 lost (6/8). Route every climb
  over the board's exact stance geometry: 5/8, ls42 and ls373 lost and ls110 still lost.
- **"`_refuel` harvests at a loss under a cone."** It cannot: idling through the same span
  costs the same drain and yields nothing, so an absorb under fire is never worse than
  waiting.
