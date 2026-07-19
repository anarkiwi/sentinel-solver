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
sign. The residual concentrated in **pan redraw** (worst observed ls335 p21 -383 f,
p8 -268 f; ls42 p15 -125 f); that term is now modelled per notch (`sentinel/pancost.py`)
and the residual under it is the fill proxy — see open problem 1. These live numbers
predate the pan model and need re-measuring.

ls42 is **not** won. The current failure is a dead-end at energy 2 (alive, `_search`
expands 2 nodes and returns `None`, the player waits safely).

The `_search` returns-`None`-at-the-root defect listed under "Disproved" for ls335 also
reproduces on **unfrozen ls42 in the pure sim**: 0 actions, root expands 1 node, every
`_c_pursue` chain runs out of `_MAX_PURSUE` iterations. It is present with the flat pan
model and with the derived one alike (A/B'd in one build), so it is neither caused nor
cured by pan cost. Frozen, the same start state wins in 36 actions.

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
samples -- busy-scene pans priced at the flat `REDRAW_BASE = 34`, since replaced.

## Open problems, ranked

1. **Terrain fill cost (now the dominant cost error).** Per-notch pan redraw is modelled
   and its tile selection is byte-exact (`sentinel/pancost.py`, docs/render_cost.md);
   per-notch rms fell 18.3 -> 7.6 f over `golden_pan_cost.json`. What is left is the fill
   proxy underneath it, and it is systematic in scene busy-ness: binned by measured cost,
   mean error runs +1.8, -1.4, -4.5, -9.0 f across quartiles averaging 9.5, 17.8, 29.0,
   49.0 f. Busy scenes under-price, on both the pan path and the settle path, which is
   why the 5 live ls42 whole-step pans do not yet separate the derived model from the
   flat one (both miss p10 -82/-14 f and p15 -95/-104 f). The lever is
   `projector.PER_SCANLINE`/`PER_PIXEL` and the cross-polygon span coupling, NOT a
   compensating constant in `pancost`.
2. **The live ls42 run has not been re-measured under the new pan model.** The frozen
   search still wins in 36 actions with 0 breaches and the sim outcome is unchanged, so
   the model is not a regression, but whether it moves the live energy-2 dead-end is
   untested — that needs a VICE run.
3. **The energy-2 dead-end is a symptom, not a cause.** Frozen, the same search wins ls42
   in 36 actions from the same start state, so `_search` is not deficient: the live run
   drifts into the energy-2 position once mispriced frames shift rotation phases. Do not
   investigate it as a search defect.
4. **`SENTINEL_STEP_SIGMA` is stale.** The in-code default 68.4 f came from a run
   contaminated by driver defects since fixed; clean runs measure 49.3 f and 46.4 f.
   Lowering it does not change the ls42 outcome, and the margin tests pin concrete
   window/budget numbers, so the update needs a test-pin review.
5. **py65 exact backend no longer covers transfer settles.**
   `projector._exact_render_cost` returns `None` for any non-player observer, and a
   transfer settle is always priced from a non-player slot at plan time (`_settle_eye`),
   so `RENDER_COST_BACKEND=py65` silently skips that path.
6. **Pan-commit wall-clock timeout.** `driver/kbd_aim.py` uses
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
