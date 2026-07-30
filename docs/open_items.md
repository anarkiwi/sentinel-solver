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

## 8. The enemy clock prices a marching `$1887` to within 1-2%, not to the cycle

**Wrong.** `driver.instrument --frames 3000 --follow` reports **278** CORE divergences on
ls9795 and **173** on ls335. ls42 is clean: **0 over 3000 frames**. Every event is an
enemy's `update_cd` reading 4 in the sim where the machine reads 1 — one `$16ED` reload the
sim reaches a pass early — or the `$1805` rotation that follows from the same lead.

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

That column orders the three boards exactly as the divergence count does, and it is the
column that says the residual is a **body** problem: only `consider_enemy_state` writes a
CORE field, and only on the two boards whose rays reach the board edge.

**What the sub-pass and sub-body splits fixed.** A pass is spent in four segments with
`pass_phase` naming the resume point, and `consider_enemy_state` itself in ten stages with
`body_stage`/`body_index`/`body_partial` naming a position between an `$1887` and the CORE
write its answer causes ([architecture.md](architecture.md#consider_enemy_state-is-resumable-too)).
Same-seed, over 3000 frames: ls9795 **329 -> 278**, ls335 **191 -> 173**, ls42 **0 -> 0**.
The staging is visible in the trace: at ls9795 frame 84 the sim enters slot 5's body,
applies `update_cd = 4` and suspends at `BODY_SCAN` slot 61 owing 50771 cycles; frames
85-87 pay it with the ROM inside `$1CBF..$1CEB`, the same march; frame 88 commits the
outcome. That is what the ROM does.

**What the cost derivation fixed.** `SEE_PROBE` was **210** where the ROM spends **4589**:
`$1C54 prepare_vector_from_angle` (3870) and `$933D` (627) were never counted. Bracketing
every `$1887` on the 6502 stack in `sentinel.tests.oracle`, a 2-probe call's non-march part
is 4456..4903 per probe over 12 boards. With that, `MARCH_OBJECT` 24 -> 28 and `MARCH_SLOPE`
581 split into 335 (nibble 4/12) and 573 (the `$1D8A` quad), the model's `$16E6` cost
against the ROM's own cycle count on ls9795 goes from **0.87 to 0.99** (60694 -> 69212
against 70001 on the worst body) and the whole-round ratio from 0.888 to 0.981.

**Where it stops, exactly.** At the first ls9795 divergence (frame 88) the sim's slot-5 body
costs 69212 where the ROM's costs 70001 — **789 cycles short, 1.1%**. Four frames later that
is most of a pass, so the sim reaches slot 1 and writes `$16ED` while the ROM is still in the
previous pass's `$31CA`. A frame is 15593 foreground cycles and the write points are ~450
cycles apart, so a 1% error on a 70000-cycle body **cannot** place a write in the right
frame. Two known sources, both from the same oracle sweep: `SEE_PROBE` is a mean over a
4456..4903 spread, and `MARCH_OBJECT` is the depth-1 constant (342) where a deeper `$1E3F`
object stack measures up to 479.

**Resolves.** Pricing `$1C54` and `$1E3F` by their own operands instead of by a mean —
`$1C54`'s shift-adds are a function of the angle bytes the model already holds, and the
object-stack walk is a loop the model already runs. Then the three-board gate.

**Not the atomic body.** It was, and it is fixed; the staging above is worth 51 events on
ls9795 and 18 on ls335 and no more, because the model reaches the wrong pass whether or not
the body's writes are staged inside it.

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
  [8](#8-the-enemy-clock-prices-a-marching-1887-to-within-1-2-not-to-the-cycle) landed, so
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
