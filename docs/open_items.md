# Open items

Everything known to be wrong or unproven, each stated as: what is wrong, the measurement that
shows it, what would resolve it. Anything not here is either correct or out of scope
([architecture.md](architecture.md), "Not modelled"); a mechanism once understood moves there
and stops being an open item.

## 1. Whole-view-dict reads and tie rollouts still buy the lattice sweep

**Wrong.** Reading a whole view dict builds the full pitch-band lattice — 3,538,944 rays — and
a tie multiplies that by the number of tied candidates.

**Measured.** The sweep is the great majority of ls335 run time. Making the per-tile path
targeted ([the landability filter](architecture.md#the-landability-filter-landtablepy)) took
ls335 93 s → 66 s with all eight boards bit-identical. `los._landable_batch` survives in the
`views.band()` calls in `phase_player._climb_candidates` and its supply hop and in
`player._climb_scan`, and in `_settle_tie`, which builds a fresh `_Views` per forked candidate.

**Resolves.** Candidate generators that ask per tile, and one sweep shared across a tie's forks.

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
`_verify_starts` compensates downstream on a bit-exact clone; the forecast itself is wrong.

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

**Resolves.** Stateful emulation of the whole `plot_world` fill sequence in render order, the
interleaved object polygons included. Short of that the only honest levers are
`projector.PER_SCANLINE`/`PER_PIXEL`.

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

## 8. The ls335 facing gap: a seven-enemy board diverges

**Wrong.** Replayed against the recorded clock, ls335 enemy facings run ahead of the ROM's.

**Measured.** 89 of the 117 exact spans match. The 28 that miss are 43 enemy-facings, and every
one is **exactly one extra rotation step**, never an under-rotation
(`test_every_facing_error_is_exactly_one_extra_rotation`, so an over-correcting fix shows up as
−1). One- and two-enemy boards are clean (ls0 16/16, ls42 10/10 live) and the gap survives
re-recording by the checkpoint method, so it is not a recorder artifact. No *constant* cadence
covers it: `UPDATES_PER_FRAME` swept over 1..8 leaves facings at 89/117 throughout, driving
below one pass per frame reaches only 92/117 at K=16, and over a uniform updates-per-round `U`
in 1..40, `U = 1` fixes 10 of the 28 bad spans and no `U` fixes the other 18.

**Resolves.** The cadence must vary *within* a span — `$1289` calls `$16B5` once per main-loop
pass, so cadence is passes per frame and a foreground `plot_world`/dither/scroll stretch reaches
no `$16B5` at all. Closing the gap means pricing those stretches better, not picking a better
number. The phase split scores 90/117 against idle-only 89 and plotting-only 64 — a one-span
margin, so it rests on the replay floors and action counts, not on facings.

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

## 12. An entry stance with no landable tile freezes the planner

**Wrong.** Where the entry stance can land nothing, every generator returns empty and the
planner takes no action at all — it is not beaten, it never starts.

**Measured.** ls9795, the hardest of the 10000 by `sentinel.atlas` ranking (8 enemies,
roughness 0.514, climb 8.125). The run ends on the action cap: **0 actions, 12000 frames,
energy 10, alive**. From the entry tile `(9,18)` at eye 3.875 both landable sets are empty
— band 0, `$F5` plane 0 — so `_climb_candidates` and `_establish` have nothing to iterate,
no reclaim target is in view, and `_under_fire` is false, so the cornered fallback in
`_tick` never fires either. The eight enemies stand at heights 7, 8, 9, 9, 9, 10, 10, 12.
No board in the eight-board suite has an empty landable set at entry, so nothing exercises
this path.

**Resolves.** Hyperspace is the move class the planner lacks. `$216A` spends 3 and
relocates the body without needing line of sight, and `phase_player` calls `_hyperspace()`
from exactly one site — `_finish`, the win move. Jumping once from entry moves the body to
`(3,1)` for 3 energy and raises the landable set from 0 to 1 — but still with no affordable
climb, so the fix is not "hyperspace when stuck once". It is to treat relocation as a move
the planner can choose and evaluate like any other: the landing is judged by what it can
land and eat, and the purse bounds how many jumps are affordable ($2170 kills on underflow).

## Disproved — do not resurrect

Each is a hypothesis and the measurement that killed it.

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
- **A K-pruned landability table.** The first landing hit sits deep inside a candidate row, so a
  stored-first-`K` row decides almost no landing query at any affordable `K`: at K=64 it costs
  tens of MB per lattice, falls back on nearly every query and is no faster than the exact path,
  and a `K` large enough to decide the row would be terabytes. `crossing_mask` needs no storage.
- **Score the phase player's tie instead of playing it.** Three scorings, each worse than
  rollout's 8/8: absorb only when its value exceeds the drain over its own span (6/8, ls321 and
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
