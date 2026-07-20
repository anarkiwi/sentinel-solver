# Plan-vs-live frame fidelity

## READ FIRST: "landscape 42" is two different boards

`landscape_from_digits` (driver/core.py) parses the typed code as **hex**: typing `0042`
seeds internal landscape **0x42 = 66**. So:

- `driver/play_player.py 42` and the human logs (`ls42.json` carries
  `entered_code 42, landscape 66`) both play **internal 66** -- player starts at (13,29).
- `Game.new(42)` / `sentinel/tests/test_astar_player.py::_LANDSCAPE = 42` build **internal
  42** -- a DIFFERENT board, player at (14,27), 17 objects vs 66's 16, zero slot overlap.

`Game.new(66)` matches the human ls42 fixture exactly (16/16 objects, same slots), and the
live replay confirms it: `entry match vs generate(66): (16, 16)`.

**The sim tests and the live driver have therefore never exercised the same board.** Any
sim-vs-live comparison keyed on the number 42 is void, including "A* wins ls42 in 28
actions" (that is internal 42) against the live 12-action loss (internal 66).

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

## The live ls42 loss is a gate on the wrong window (measured, not inferred)

Live ls42 fails identically every run: 11 steps, then `boulder (1,24)`, then a forced
hyperspace off the climb to energy 0. `driver/plan_audit.py` isolates it, and **it is not
timing**. Over the 11 executed steps charged is 3108 f vs measured 3047 (**-2%**),
whole-step rms **24.1 f**, and the enemy clock is right (live rotations every ~700-900 f
against `ROT_PERIOD_FRAMES` 749). The plan's predicted enemy phase tracks live throughout.

The audit keeps two windows and only one is gated:

| step | pred_win | live_win | pred_pbody | live_pbody |
|---|---|---|---|---|
| `boulder (1,24)` FIRE | 8148 | 7983 | 655 | 490 |
| `robot (1,24)` STALE | 7751 | 7552 | 258 | **60** |

`win` is the **destination tile's** window -- 7983 f, enormous, and the plan is right
about it to within 2%. `pbody` is the **player's own body** window, and it collapses
490 -> 60.

`_pick_hop` drain-gates on `_gaze_window(tile) >= HOP_FRAMES + margin`, i.e. on the tile
being built on. **Nothing gates the ~750 f the hop takes against the window the player's
own body has left**, and the player stands on its old tile for all of it. Each individual
step passes its own gate (boulder: budget 398 < pbody 655) while the 3-step macro does not
(750 > 655), so the hop is committed and the body is drained mid-macro.

The hop cost itself is sound: `HOP_FRAMES = 700` sits under two live-measured hops
(745, 879 f) and within 25% (`test_hop_budget.py`). An earlier claim here that it
under-budgets 2-3x was wrong and is retracted below.

### What the shadow instrumentation showed: the pursuit is all-or-nothing

`AStarPlayer._hop_audit = []` makes `_pick_hop` record what the body-window gate WOULD
decide without enforcing it. One live ls42 run, behaviour unchanged (12 actions), 353
records. The gate is **not** unaffordable:

- 225/353 candidates pass the tile gate; **181 of those would also pass the body gate**.
- The margin is irrelevant here: `body_ok` equals `body_ok_raw` on every row.
- Every rejection sits at **depth 10 and nowhere else**, and all 44 are identical:
  `body_window = 656`, `need = 780`, short by **124 f**.

They are identical because the body window is a property of the PLAYER, not of the
candidate tile -- at depth 10 the player has 656 f wherever it builds, and every hop
needs 780, so all 22 candidate tiles fail at once. `(1,24)`, the tile the live run dies
on, is one of them. The gate is right.

The damage is structural: `_c_pursue` is an all-or-nothing macro. It chains hops until
the enemy is landable and returns ONE child, or `None`. One unsurvivable hop at depth 10
discards the ten good steps before it, the root loses its only child generator, and
`_search` returns `None` from the start -- which is exactly the 0-action live run.

Both failure modes are the same defect. Ungated, the pursuit "reaches" the enemy on paper
and the player dies executing step 11; gated, it returns nothing at all. The fix is
neither a looser gate nor a bigger margin: **the pursuit must be able to return partial
progress**, so the search can expand from a truncated climb and solve step 11 separately.

### The obvious fix does not work live (attempted, reverted)

Gating the hop on the player's own body window -- `_hot(HOP_FRAMES + (k-1)*SETTLE)`
alongside the existing tile gate -- takes the live player from 12 actions to **0**. It
was reverted. Two things this exposed, both of which invalidate the offline evidence
that made it look right:

- **`Game.new(42)` starts FROZEN**, so `_body_drain_window()` is `inf` and the gate is
  inert. Every pure-sim A/B of a body-window gate is therefore vacuous -- the sim "win"
  under the gate exercised a path where the gate never fired.
- Clearing the frozen bit offline reproduces a root that returns `None`, and the margin
  is implicated: with the gate on, `_search` finds a 29-step plan at
  `SENTINEL_STEP_SIGMA` 24.1 or 0, and **nothing** at 49.3 or 68.4. But live still
  returns `None` at sigma 24.1, so an unfrozen fresh board is *still* not a faithful
  stand-in for the live root (real facings and cooldowns differ).

So the diagnosis above (the hop is gated on the destination tile, never on the body)
is measured and stands, but no gate formulation has yet survived live. Anything
attempted next must be A/B'd **live**, not in the sim.

## The human's winning line is pruned by the beam, not gated out

Replaying the human ls42 (internal 66) line through the model: the energy curve matches
the log **exactly for 16 steps**, the first divergence is the enemy drain the audit
already records (`drain: [15]`), and the line reaches **eye 11.875** -- the Sentinel's
plinth. Our planner never passes 7.375. So the model can represent and execute the
human's line.

Asking `_pick_hop`, at each human hop, whether the human's destination tile is even a
candidate:

| human step | tile | landable | in top-8 candidates | rank |
|---|---|---|---|---|
| 1 | (9,30) | yes | **no** | - |
| 5 | (13,26) | yes | **no** | - |
| 12 | (2,24) | yes | **no** | - |
| 17 | (5,22) | yes | **no** | - |
| 23 | (15,11) | yes | yes | 2 |
| 29 | (4,18) | yes | yes | 4 |
| 34 | (21,4) | yes | yes | 1 |

The first four hops -- the ones that build the climb off the floor -- are **never
generated**. And they are not filtered: each passes `_tile_base`, `k=1`, energy,
raise-the-eye and the drain gate. They are **ranked out of `_TOP_HOPS = 8`**.

The rank key is `(sees, robot_eye, window)` descending, so it maximises the eye gained
per hop. The human does the opposite: every one of those hops raises the eye by exactly
**+0.5**, the minimum a single boulder buys, on a steady staircase. Our ranking prefers
the biggest available rise, and the rank key cannot see the aim it will pay to get there
-- the hop the live run dies on.

**Aim cost is angular, not spatial.** Over the 23 landable tiles at the ls42 start,
`corr(aim, manhattan distance)` is **-0.54** -- farther is CHEAPER -- against +0.60 for
pitch notches and +0.44 for bearing notches. An adjacent tile needs a steep down-look:
(12,29) at distance 1 costs **667 f**, while (8,28) at distance 6 costs **159 f**, because
a far tile sits near the horizon and needs almost no pitch change. The fatal `(1,24)`
build measured `pan_h 18 f` and `pan_v 271 f`: its bearing was nearly free and its whole
expense was swinging pitch. So the term a hop ranker needs is the angular delta from the
CURRENT stance -- and it is path-dependent, which is why alternating between near
recycles and far builds costs more than the human's consistent staircase.

So the search never had the human's line to reject. Beam width and rank order are the
lever, ahead of any further work on gates, margins or frame cost.

## The enemy PHASE is off by 243 f -- 10x the frame-cost error I was chasing

`driver/plan_audit.py` on the live ls42 (internal 66) run, plan's forward model vs the
live board at every fired step:

| | value |
|---|---|
| plan-vs-live window optimism | **mean +243 f**, min +165, max +333, **always positive** |
| identical on tile window and body window | **11/11 rows** -> ONE shared phase error |
| whole-step charged-vs-measured rms | 24.1 f |
| cumulative charged-measured over the 11 steps | -61 f |
| `_STEP_SIGMA` margin at depth 10 | 80 f |

The model believes it has **243 f more time than it does**, on every step, in the same
direction, from the first action onward. That is **32% of a 749 f rotation** and 3-14x the
safety margin -- it swamps the margin completely, so no choice of sigma can cover it.

It is NOT frame miscounting: cumulative charged-vs-measured is only -61 f over the same
11 steps, while the phase error is already +292 f at step 1. The error is roughly constant
rather than growing, which points at an initial-condition/phase problem in the enemy clock,
not at a rate error -- the live rotation intervals themselves match `ROT_PERIOD_FRAMES`.

A second signal in the same data: the model keeps its two enemies locked exactly 8 cooldown
units apart for the whole run, while the live pair drifts (8 -> 9 -> 27). The model's
enemies rotate in lockstep; the real ones do not.

This subsumes the earlier "gate uses the wrong window" finding. The gate is on the wrong
window AND the window it reads is 243 f optimistic. Fix the phase before any further
gate, margin or frame-cost work: every gate in the planner is a comparison against a
number that is systematically wrong by a third of a rotation.

### FIXED: the cooldown tick ran AFTER the frame's enemy passes

`advance_frame` swept the enemy slots and THEN called `cooldown_frame`. The raster-IRQ
cooldown tick ($9663/$1317) runs BEFORE the foreground passes, so an enemy the tick makes
due rotates in the SAME frame; ticking last defers every rotation to the next frame.

Isolated with `driver/instrument.py` plus a captured 400-frame live image trace, racing
variants offline on a clock+facing criterion (cooldown bresenham/gate, per-enemy
rotation/update/draining cooldowns, enemy `h_angle`):

| passes/frame | tick first | tick last |
|---|---|---|
| 2 | frame 50 | frame 50 |
| 3 | frame 50 | frame 50 |
| 4 | frame 73 | frame 50 |
| 8 | **none in 400** | frame 50 |

Live re-race after the fix: **`[CORE] no divergence within 1200 frames`**, against frame 50
before. Every enemy facing, rotation cooldown, update cooldown and drain cooldown now
tracks the real game exactly for 1200 frames.

**`UPDATES_PER_FRAME` is back to 8, and the earlier "the ROM does ~3/frame" change is
reverted.** The cursor measurement behind it was right -- the ROM's foreground loop makes
2/3/4 passes a frame -- but the cursor is not what rate-limits an enemy: its own $16E9
`update_cd` gate (reload 4) is, and it is far tighter. Considering every slot each frame
reproduces the ROM's clock exactly; a literal 3 does not (frame 50 either way). The
remaining `cursor`/`prng` divergence is SWEEP-tier and does not move the modelled dynamics.

### The PRNG rate is unmodellable -- but it touches far less than I claimed

Measured live: recover k such that k model `prnd` calls take the state at frame N to
frame N+1, over 200 frames.

    prnd calls per frame: {19: 40, 11: 2, no-match-within-32: 158}
    cursor decrements per frame: {2: 17, 3: 128, 4: 53}

The ROM draws the stream ~19+ times a frame, often >32, while the cursor moves 3, so
`update_enemies` is not the main consumer and the rate cannot be reproduced without
porting every caller. The LFSR itself is correct -- `prng._shuffle` matches `$31CA`
instruction for instruction.

**What that actually limits is only two things**, the model's only PRNG consumers:

- the discharge tree's landing tile (`$1A5D` -> `put_object_in_random_tile_below_z $1224`)
- the hyperspace landing tile (`$2156` -> the same `$1224`)

**Meanie creation is NOT one of them.** `consider_creating_meanie $197D` never touches the
PRNG: it walks the per-enemy search counter down, takes the FIRST tree slot within 10
tiles of the TARGETED player in both axes that the enemy fully sees (`FOV_CREATE_MEANIE`
$28), and converts it. Object table, positions, visibility -- all of which the model holds
exactly. So **which tree becomes a meanie, and whether one does at all, is fully
predictable**, as is the meanie's hunt (`$16F2`, a fixed $8/update turn toward the player)
and the hyperspace TRIGGER. Only where the hyperspace drops you is random.

That makes the meanie threat plannable rather than a hazard to be robust to: a plan can
see the spawn coming from the trees it leaves within 10 tiles of a seer's full sight.
An earlier revision of this file claimed the opposite; it was wrong.

### Rotation phase: still open

At the frame-50 CORE divergence the emulator has `update_cd = 4` -- exactly
`UPDATE_COOLDOWN_SCAN`, i.e. it *just* ran `consider_enemy_state` and rotated -- while the
sim has `update_cd = 1` and `rotation_cd = 0`: it ran its consideration EARLIER, when the
rotation was not yet due, and must now wait for the next one. The two clocks (update
cooldown vs rotation cooldown) are phased differently against the frame.

Leading hypothesis, untested: the ROM's cooldown tick is a raster IRQ ($9663) that can
fire BETWEEN foreground passes, whereas `advance_frame` runs all `UPDATES_PER_FRAME`
passes and only then calls `cooldown_frame`. That ordering would shift by one
consideration when an enemy first sees `rotation_cd == 0`.

### Root cause of the +243 f: a u-turn unfreezes the world, mid-aim

`$12D5 CMP #$22 / BCS $12DE` lets action codes >= $22 skip the sights-on check and fall
into `$12E1 LSR $0CE5`. A u-turn is code **$23**, so keying one **starts the enemy clock
part-way through the aim**, before any real action fires. The model only cleared
`PLAYER_NOT_ACTED` when the action applied (`actions.py`), so it held the world frozen for
the WHOLE first aim -- and ls42's first aim keys a u-turn.

Measured at the first action, model vs live:

| | rotation_cd | facings |
|---|---|---|
| before the fix | [0, 0] | [128, 184] |
| **after the fix** | [146, 138] | **[148, 204]** |
| live | [130, 122] | **[148, 204]** |

The facings now match live **exactly**; before, the model was one full 20-unit rotation
step behind, which is the +243 f window optimism. Residual cooldown lag is 16 units
(~60 f), down from ~243 f -- the split point (`toggles + UTURN_FRAMES`) is approximate,
since the ROM may unfreeze at the first keypress rather than at the u-turn's own tap.

**Consequence: the planner now takes 0 actions on live ls42.** Correcting the clock
removed roughly one action's worth of free frozen time the search had been planning
against, and its gates cannot find a line without it. That is a real result, not a
regression to paper over: every line the planner has ever found on this board was priced
against a world that did not start until later than it really does.

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
- "`HOP_FRAMES` under-budgets every hop 2-3x (700 vs a real 1400-2300)." Wrong: that
  came from a broken offline replay (diverged state, `last_bearing` reset to None so
  every action paid a full toggle+pan, transfer settle from the pre-transfer eye).
  Live hops measure 745 and 879 f. Replacing the flat budget with the computed one
  made the live player take ZERO actions, because it blocked every hop.
- "Enemy freeze under `plotting=True` is the lever": it freezes enemies outright
  (`rcd=[0,0]`), it is not a fidelity knob.
