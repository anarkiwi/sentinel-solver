# Plan-vs-live frame fidelity

The planner gates every action on `window >= aim + settle`. Both sides are model
predictions: `BasePlayer._charge` advances the simulated enemy phase by exactly the
aim+settle frames it charges, so a charged-vs-measured error on one step shifts *when*
every later rotation commits. The metric is therefore **per-step |error|**, not
cumulative drift — a run can be near-zero net and still gate a step wrongly, because a
rotation-phase error is amplified by the window being a min over several enemies.

Audit hooks (live driver, `driver/live_player.py`): `frame_audit` (whole-step charged vs
`$9630` frame counter), `exact_audit` (aim/settle split), `aim_subframes`
(pan/toggle/cursor), `wait_audit`. `driver/plan_audit.py` replays the plan's forward
model against live per-step `rotation_cd` / window.

## Current error budget (live ls42)

Whole-step rms **49.3 f** (46.4 f in a second clean run). Waits rms 1.0 f over 103
waits. Transfer aim error 0 f. Transfer settle error +42 / -7 / -54 f — no systematic
sign. The residual now concentrates in **pan redraw**: steps needing several bearing
notches through busy scenes (worst observed ls335 p21 -383 f, p8 -268 f; ls42 p15 -125 f).

ls42 is **not** won. The current failure is a dead-end at energy 2 (alive, `_search`
expands 2 nodes and returns `None`, the player waits safely).

## Settled: the residual is fidelity, not search

The both-frozen discriminator (`driver/frozen_run.py`, sim side via
`freeze_sim_enemies`) resolves this. With enemies frozen on both sides, ls42 is **won**:

    sim   astar 0x0042 frozen: won=True actions=36 energy=11 (12.1 s, no emulator)
    live  astar ls0042 frozen: won=True actions=36 energy=11

All 35 pre-win steps match sim-vs-live on `(action, tile, energy)`. Step 36 is the
hyperspace landing, which differs by design (PRNG-driven, not steerable,
`sentinel/actions.py:184`) and wins on both sides. The frozen winning line therefore
**is** the line the live player attempts -- the check that a frozen win otherwise needs.

Conclusion: the planner solves ls42 when it models the world correctly, so the live loss
is enemy-phase timing (frame-cost fidelity shifting when rotations commit), not search
capability or energy management.

Freezing does **not** reduce cost error, as expected -- it removes enemy feedback, not
render cost. Action-step rms is ~47 f frozen and ~47 f unfrozen (the unfrozen run's
17.6 f aggregate is diluted by 103 near-exact `wait` steps, rms 1.0 f; do not quote it).
The frozen run's 35 residuals are uncontaminated by rotation feedback:

| action | n | rms | mean | range |
|---|---|---|---|---|
| create | 13 | 52.7 | +7.8 | -63 .. +146 |
| transfer | 7 | 46.9 | +27.3 | -43 .. +69 |
| absorb | 15 | 42.5 | -15.0 | -58 .. +113 |

Error is spread across all three action types with no systematic sign -- consistent with
per-notch pan redraw (every action pays pan cost to aim) and inconsistent with any one
action's settle model being wrong. `n` is thin: the +27.3 f transfer mean rests on 7
samples and must not be modelled on. The `+146` / `+137` outliers are the informative
samples -- busy-scene pans priced at the flat `REDRAW_BASE = 34`.

## Open problems, ranked

1. **Per-notch pan redraw (dominant cost error).** Each `pan_viewpoint` attempt is one
   `plot_world` at the *intermediate* bearing into the pan strip buffer (`$10D7`/`$111C`)
   plus the notch's 16 h / 8 v scroll steps. Measured 25-27 f in empty views, 73-112 f in
   busy ones. `_aim_frames` prices it as `REDRAW_BASE` (34, flat) + `STEPS_PER_EDGE`
   (0.02) per visible edge — both fitted, and no choice of the two spans that range.
   Correct fix is a `render_cost`-class scene model evaluated at each notch's bearing;
   blockers are that `projector.render_cost` models the full play buffer (`$14..$8A`),
   not the pan strip, and costs ~60 ms/call, so it needs a bearing-keyed memo before it
   can sit in the A* inner loop. Not yet attempted. Do not retune the two constants.
2. **The energy-2 dead-end is a symptom, not a cause.** Frozen, the same search wins ls42
   in 36 actions from the same start state, so `_search` is not deficient: the live run
   drifts into the energy-2 position once mispriced frames shift rotation phases. Fixing
   (1) is the way to close it. Do not investigate it as a search defect.
3. **`SENTINEL_STEP_SIGMA` is stale.** The in-code default 68.4 f came from a run
   contaminated by driver defects since fixed; clean runs measure 49.3 f and 46.4 f.
   Lowering it does not change the ls42 outcome, and the margin tests pin concrete
   window/budget numbers, so the update needs a test-pin review.
4. **py65 exact backend no longer covers transfer settles.**
   `projector._exact_render_cost` returns `None` for any non-player observer, and a
   transfer settle is always priced from a non-player slot at plan time (`_settle_eye`),
   so `RENDER_COST_BACKEND=py65` silently skips that path.
5. **Pan-commit wall-clock timeout.** `driver/kbd_aim.py` uses
   `run_until_pc(PC_PAN_DONE, timeout=_RU_COMMIT)` (4 s). `$365D` recurs every frame, so
   a timeout there means the game left the play loop — a bug, not back-pressure. Observed
   once in five live runs; it aborts the run (`won=None`). Same defect class as the
   `tap_action` timeout already replaced by `_run_to_scan()`.

## Disproved (do not resurrect)

- "Transfer settle over-charges systematically (+40..+102 f); fix
  `viewpoint_replot_frames` first." The over-charge was a 6.0 s wall-clock
  `run_until_pc` in `tap_action` **clipping the measurement** at ~300 frames. Unclipped,
  the same transfers show no systematic sign.
- "Correcting the settle's viewpoint will reduce it." Pricing the settle from the
  post-transfer eye moves it **up** (median +28 f over 12 ls42 tiles).
- "Aim mispricing is the secondary term." It was the larger term, and its cause was a
  driver defect (a swallowed sights toggle burning 171 frames), not a missing cost term.
- Ranking fixes by *cumulative* frame drift: at the failing step net drift was ~-17 f
  while the enemy phase was ~35 f out.
- "A* taking 0 actions on ls335 is the live-only freeze gating it out." The `--no-freeze`
  control does the same thing: A* loops `plan (1 nodes): None` -> `plan (2 nodes): None`
  from the start state at full energy, frozen and unfrozen alike. ls335 A* planning is a
  **separate, open defect**, unrelated to the enemy model, and it makes ls335 useless as
  a discriminator arm. Two nodes means the root's successors are pruned at generation --
  a gate rejecting everything, not a depth or budget limit (`--node-budget 200000` never
  mattered). Reproduce offline by feeding a live-read ls335 start state to `_search`.
- "Enemy freeze under `plotting=True` is the lever": it freezes enemies outright
  (`rcd=[0,0]`), it is not a fidelity knob.
