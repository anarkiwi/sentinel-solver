# Verified current state (ground truth for doc rewrites)

Everything below is measured or read from code in the working tree. Do not restate
anything here that the code already says; cite mechanism + address, not narrative.

## 1. Driver defects fixed (these were injecting enemy-phase error, not modelling it)

- **Sights toggle `$1236` edge latch.** `check_for_player_input` edge-latches SPACE
  (`$11B5 LDA $1236 / BNE skip_sights_toggle`); `$1236` clears only on a scan that sees
  SPACE *up* (`$11D4/$11D6`). `kbd_aim._one_scan_press` pressed without first running an
  idle re-arm scan, so on a step with no pan (nothing else ran a scan) the ON press was
  swallowed and `sights_set`'s 6-pass retry burned **171 frames**. Now one idle re-arm
  scan precedes the press; a toggle is exactly 2 scans.
- **`_run_to_scan()`** (new, `driver/kbd_aim.py`): re-arms the wait while `$0CE4` bit7
  says the scan gate is shut. Replaces `run_until_pc($9678, timeout=6.0)` in `tap_action`
  — a wall clock on a plot-bounded wait. At ~50 fps that 6.0 s == 300 frames, and the
  ls42 transfer settles measured 270/302/303, i.e. **clipped by the timeout, not
  measured**. Used by `_one_scan_press`, `_uturn`, `tap_action`.
- **`LiveMixin._wait` was wall-clock** (`bm.exit()` + `time.sleep(60/50)`), charging 60
  frames it never measured. Now polls `Executor.frames()` (wrap-free `$9630` u32
  checkpoint) until `WAIT_FRAMES` elapse, CPU left halted. Audited into
  `result["wait_audit"]`. **Measured: 103 waits, rms error 1.0 f, max 1 f.**
- **`last_bearing` was permanently `None` in the live player.** `LiveMixin._fire` fully
  overrides `BasePlayer._fire`, which is the only place `last_bearing`/`cursor` are
  maintained outside `_search`. So the model never took the reuse branch while the
  executor did (`sentinel_execute.py` `reuse = drv.sights_live_on() and
  drv.committed_bearing() == want_bearing`), over-charging aim by +42..+70 f per reused
  step. New `LiveMixin._sync_aim_state`, called from `_observe`, reads the driver's
  committed bearing (`$0C5F` bit7) and cursor (`$0CC6/$0CC7`) from the halted snapshot.

## 2. Cost-model corrections (all derived, none fitted)

- **Cursor term**: was a fitted `1.24 f/px`. Now `max(|dcx|,|dcy|) + popcount($6B)`:
  `move_sights $9958` moves 1 px per gated scan, and `$0CC8` (reloaded `#$6B` at `$11E0`)
  skips one scan per set bit before the first move (`$11F6 ASL / BCS`), i.e. +5. Exact on
  24/32 recorded drives, within 1 on 6 more, bounded `px <= measured <= px+5` on all.
- **`_step_aim_frames`**: a transfer charges 0 aim **only** on a bearing reuse (the
  executor sends no aim keys then); otherwise the full `_aim_frames`.
- **Transfer settle is priced from the POST-transfer eye.** The ROM moves `$0C63` in
  `try_to_transfer_into_object $1B64` *before* `play_landscape_loop $357D` runs its two
  `plot_world` passes (`$35C3/$35C6`), and `actions.create` gives the new robot bearing
  `creator_angle ^ $80` (`$1BE0`). New `playerbase._settle_eye(verb, tile)`;
  `_settle(verb, view=None, observer=None)`. Effect measured over 12 ls42 tiles: delta
  **-90 f to +129 f, median +28 f** on a ~325 f settle — i.e. the old pricing *under*-charged
  on the median tile.
- **`projector._occlusion_visible(state, observer)` / `project_scene`** now thread
  `observer`. Previously `render_cost`'s `observer` moved the `$2625` setup while the
  `$245B` occlusion raytrace still started at `state.player` — a non-player observer
  silently mixed two eyes.
- **Per-notch pan redraw is now derived** (`sentinel/pancost.py`, from `pan_viewpoint`
  `$10B7`). One notch = one strip clear (`$3912` h / `$38AD` v) + exactly **one**
  `plot_world` (`$2625`) + the notch's queued scroll steps (`$10EE` 16 h / `$1135` 8 v).
  The plot runs at the **intermediate** angle: the `$9925` delta is added before
  `JSR $2625` and fixed up after, so a right pan plots at h+`$14` (destination+`$0C`,
  fixed by `$10E9 SBC #$0C`) and a downward pitch at v-`$0C` (destination-8, fixed by
  `$1130 ADC #$08`); left pans and upward pitches plot on the destination.
- **A horizontal pan uses a different render window.** `$10EE` reaches
  `initialise_buffer_variables $2993` via `$994F` with A=`#$02`, whose `$29C4` table entry
  gives `$0007`=`$08` / `$0012`=`$84`, culling tiles the play window (`$14`/`$8A`) keeps.
  Vertical pans (`$9939`, A=`#$00`) share the play window. `projector.project_scene` takes
  the mode and threads the window through the `$293C` on-screen test.
- **`REDRAW_BASE = 34` is DELETED** from `playerbase._aim_frames`, with its registry
  entry. `actioncost.STEPS_PER_EDGE` still exists but no longer prices the aim path.
  Accuracy over `golden_pan_cost.json` (measured notch plot cost spans **3.8-99.8 f**,
  median 22.2): flat `REDRAW_BASE`+`STEPS_PER_EDGE` rms **18.3 f**, mean +9.3, median abs
  16.5, max 63.7; derived per-notch rms **7.6 f**, mean -3.3, median abs 4.5, max 37.7.
- **Memoization makes the model a net speed-up** despite ~24 extra plots per aim.
  `render_cost` was ~5.8 ms/call, ~62% of it the view-independent `$245B` occlusion
  raytrace; that is now `projector.occlusion_visible`, memoized per (scene, observer), and
  `pancost.notch_frames` per (scene, observer, direction, plot angle). Both key off
  `projector.scene_key`, a digest of every byte `plot_world` reads.
  `test_player_placement_invariant` **250 s -> 33 s**,
  `test_player_wins_landscape_0042` **183 s -> 21 s**.
- **`actioncost.action_rounds`**: removed the double-counted
  `STEPS_PER_EDGE * visible_edges(...)` term (already inside both settles). NOTE: this
  function has **zero callers repo-wide**; it is dead API surface duplicating
  `playerbase._settle`.

## 3. Planner robustness (new)

- **Interval drain gate.** Every drain gate compares the window against the pessimistic
  end `budget + _margin(depth)`, where
  `_margin(d) = SENTINEL_MARGIN_K * SENTINEL_STEP_SIGMA * sqrt(d + 1)`.
  Rationale: per-step charged-vs-measured error is ~zero-mean but does **not** cancel —
  each excursion permanently shifts *when* a rotation committed — so uncertainty
  accumulates as a random walk in plan depth. Defaults k=1.0, sigma=68.4 f (rms of
  per-step error). Both env-overridable.
- **`_defend()`** — non-conceding ladder (counterattack a landable dangerous seer, else
  escape-transfer to the widest-window body). `_react` = threat gate -> `_defend` ->
  hyperspace only as last resort.
- **`_restale(key)` progress guarantee.** `_stale = (step key, consecutive verdicts)`.
  First verdict runs the ladder (search -> defend -> zero-margin search -> wait); a
  **repeat** of the same key goes straight to `_wait()` (frame-exact, really advances the
  world). The live `_plan_step_stale` releases a step once a wait has been taken and the
  **raw** budget clears — margin-only blocking is never terminal; raw unsafety still is.

## 4. Test methodology

`pytest.ini` gained `--dist loadgroup`; both live-VICE tests carry
`@pytest.mark.xdist_group("vice")`. They previously ran concurrently under `-n auto`,
each booting its own docker VICE, and the contention flipped
`test_plan_dwell_prediction_matches_live_ls42` (strict xfail) between XFAIL and XPASS.

`sentinel/tests/golden_pan_cost.json` — 288 rows over landscapes 0, 42, 335. Examined
(`$2845`) and filled (`$2A24`) tile counts are **byte-exact against the 6502 on every
row**. `sentinel/tests/test_pan_cost.py` marks regeneration as oracle-only; the
non-oracle tests pin the byte-exact tile selection, the mode-2 window genuinely differing
from the play window, the accuracy bracket, and that the derived model beats the best
possible flat constant. New registry entries: DERIVED `_CLEAR_CYCLES_H` /
`_CLEAR_CYCLES_V`, MEASURED `CLEAR_FRAMES`, derivations in `test_timing_derivations.py`.

`test_human_audit.py`'s pinned ls335 `gate_reject` list lost step 40 — one **fewer**
false disagreement with the human win log.

## 5. Measured results (live ls42, `out/rerun_final_ls42.log`)

- Whole-step charged-vs-measured rms: **68.4 -> 60.8 -> 49.3 f** across the three stages.
- Waits: rms **1.0 f** over 103 waits.
- Transfer settle error: **+42 / -7 / -54 f** (was +101 / -10 / -129 with the clipping).
- Transfer aim error: **0 / 0 / 0** (was +70 / +45 / +42).
- Aim at the previously worst steps: p13 **-152 -> +14**; p15 -109 -> **-125** in the final
  run (now pan-dominated, see below).
- **ls42 is NOT won.** Failure mode moved: escape-hyperspace death at energy 0 ->
  livelock (alive, energy 8) -> energy-starved dead-end (alive, **energy 2**, `_search`
  expands 2 nodes and returns None, player waits safely).
- A second clean run with `SENTINEL_STEP_SIGMA=49.3` reaches the **same** energy-2
  dead-end (rms 46.4 f), so the dead-end is **not** caused by the margin. It is a search /
  energy-management problem, not a frame-cost fidelity problem.
- **These numbers predate `pancost`.** The live ls42 run has **not** been re-measured
  under the derived per-notch model; that needs VICE and is open.

Outcome checks under `pancost` (offline, A/B'd in one build):

- Frozen sim A* still wins ls42 in **36 actions, 0 breaches**, identical under both models.
- Unfrozen sim ls42 A* takes **0 actions under both** models — the same root-prune defect
  already recorded for ls335, so it is neither caused nor cured by pan cost.

## 6. Open problems (the honest next work)

1. **Terrain fill cost is now the dominant error.** The residual over
   `golden_pan_cost.json` is **not** the notch model's: tile selection is byte-exact and
   `C_EXAMINE` is centred (measured mean **1704** cycles/examine vs **1737** charged). It
   is the fill proxy (`docs/render_cost.md` "Known gaps" item 2), and it is systematic in
   scene busy-ness: binning the golden by measured cost, mean error runs **+1.8, -1.4,
   -4.5, -9.0 f** across quartiles whose mean costs are 9.5, 17.8, 29.0, 49.0 f. Busy
   scenes under-price.
2. **The live ls42 run has not been re-measured under `pancost`.** Every `pancost`
   outcome check above is offline (frozen and unfrozen sim). Needs VICE.
3. **py65 exact backend silently stopped covering transfer settles.**
   `projector._exact_render_cost` returns `None` for any non-player observer, and transfer
   settles are now always priced from a non-player slot at plan time, so
   `RENDER_COST_BACKEND=py65` no longer applies to that path.
4. **`SENTINEL_STEP_SIGMA` is stale.** 68.4 f came from a run contaminated by the clipping
   and wall-clock waits; clean runs measure **49.3 f and 46.4 f**. Still 68.4 in code:
   lowering it does not change the ls42 outcome, and the margin tests pin concrete
   window/budget numbers, so the update needs a test-pin review.
5. **The energy-2 dead-end is a search problem.** `_search` expands 2 nodes and returns
   None at energy 2 (below the 3 needed to create a robot). Reproduces at both sigma
   values. Why the preceding line spends energy into an unrecoverable position is
   unexamined — this, not cost fidelity, is what currently blocks ls42.
6. **Pan-commit wall-clock timeout, same defect class as the fixed `tap_action` one.**
   `kbd_aim` line ~329 `run_until_pc(self.PC_PAN_DONE, timeout=_RU_COMMIT)` (`_RU_COMMIT`
   = 4 s). `$365D` recurs every frame, so per the module's own comment a timeout there
   means the game left the play loop — a bug, not back-pressure. Observed once in 5 live
   runs; it aborts the whole run (`won=None`).

## 7. What is now DISPROVED (must not survive anywhere in the docs)

`docs/ls42_dwell_fidelity.md` ranked its fixes "transfer settle over-charge first
(+40..+102 f, systematic, `viewpoint_replot_frames` over-estimates), aim mispricing
second". Both halves are wrong:
- The over-charge was the `tap_action` 6.0 s wall-clock timeout clipping the measurement
  at ~300 frames. Unclipped, the same transfers show **no systematic sign** (+42/-7/-54,
  and -129 before the viewpoint fix).
- Correcting the settle's viewpoint moves it **up** (median +28 f), the opposite direction
  again.
- The aim error it ranked second was the larger term all along, and its cause was a
  **driver defect** (the 171 f toggle), not a missing cost term.
Also disproved by construction: ranking fixes by *cumulative* frame drift. Net drift at
the failing step was only ~-17 f while the phase was ~35 f out; per-step |error| is the
metric that matters, not the running sum.
