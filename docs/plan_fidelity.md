# Plan-vs-live fidelity

State: **the clocks are correct; the players are broken.** The enemy simulation now
tracks the real game exactly, and no player can find a line under it.

## READ FIRST: "landscape 42" is two boards

`driver.core.landscape_from_digits` parses the typed code as **hex**, so typing `0042`
seeds internal landscape **0x42 = 66**.

- `driver/play_player.py 42` and the human logs (`ls42.json` carries `entered_code 42,
  landscape 66`) play **internal 66** — player starts at (13,29).
- `Game.new(42)` and `test_astar_player._LANDSCAPE = 42` build **internal 42** — player at
  (14,27), 17 objects against 66's 16, zero slot overlap.

`Game.new(66)` matches the human ls42 fixture exactly (16/16 objects, same slots) and the
live replay agrees (`entry match vs generate(66): (16, 16)`). **Sim tests and the live
driver have never exercised the same board**; any sim-vs-live comparison keyed on "42" is
void.

## Clocks: exact

Frame-locked against the running game from a byte-identical seed
(`python -m driver.instrument 42`): **`[CORE] no divergence within 1200 frames`** — every
enemy facing, rotation/update/draining cooldown and the Bresenham clock.

Two defects got it there:

1. **Cooldown tick order.** `advance_frame` swept the enemy slots and *then* ticked. The
   raster IRQ ($9663/$1317) ticks **before** the foreground passes, so an enemy the tick
   makes due rotates in the same frame. Isolated by racing variants offline against a
   captured 400-frame live trace on a clock+facing criterion: tick-last diverged at frame
   50 for every pass count; tick-first with all 8 slots swept diverged nowhere in 400.
2. **A u-turn unfreezes the world, mid-aim.** `$12D5 CMP #$22 / BCS $12DE` lets action
   codes >= $22 skip the sights-on check and fall into `$12E1 LSR $0CE5`. A u-turn is
   code **$23**, so keying one starts the enemy clock part-way through the aim. The model
   cleared `PLAYER_NOT_ACTED` only when the action applied, freezing the whole first aim —
   and ls42's first aim keys a u-turn.

Effect of (2) at the first action, model vs live:

| | rotation_cd | facings |
|---|---|---|
| before | [0, 0] | [128, 184] |
| after | [146, 138] | **[148, 204]** |
| live | [130, 122] | **[148, 204]** |

Facings now match live exactly; before, the model sat one full 20-unit rotation step
behind, which was the **+243 f window optimism** every gate was reading (mean +243, range
+165..+333, identical on the tile window and the body window on 11/11 audited steps).
Residual cooldown lag is 16 units (~60 f): the split point (`toggles + UTURN_FRAMES`) is
approximate, since the ROM may unfreeze at the first keypress rather than the u-turn's tap.

`UPDATES_PER_FRAME` stays 8. The ROM's foreground loop makes only 2/3/4 passes a frame
(measured cursor `$0090` decrements `{2:27, 3:192, 4:79}`), but the cursor is not what
rate-limits an enemy — its own `$16E9` update_cd gate (reload 4) is, and it is far
tighter. Sweeping every slot reproduces the ROM's clock exactly; a literal 3 does not.

## Frame cost: good enough to stop working on

Live ls42 whole-step charged-vs-measured: **rms 24.1 f, mean −12.0, max |e| 46, n=11**
(`live_ls42_hops.json`, `test_hop_budget.py`), against 49.3 f before this work. Cumulative
charged vs measured over the 11 steps is −61 f (−2%).

Constants that moved UNVALIDATED → MEASURED: `HOP_FRAMES` (700, under two live hops of 745
and 879 f), `UTURN_FRAMES` (74 — a u-turn is an action tap, not a keystroke; charging
`TAP_FRAMES = 3` left p1 mispriced by +64 f), `_STEP_SIGMA` (24.1, the measured per-step
rms).

Per-notch pan redraw is derived rather than fitted (`sentinel/pancost.py`,
[render_cost.md](render_cost.md)): tile selection byte-exact on all 288 golden notches,
rms 18.3 → 6.4 f.

## Players: broken

With the clocks correct, **live ls42 takes 0 actions** — `plan (2 nodes): None`
throughout. Correcting the freeze removed roughly one action's worth of free frozen time
the search had been planning against. Every line this planner has found on this board was
priced against a world that started later than it really does.

Before the clock fixes it reached 12 actions and lost the same way every run: the climb to
eye 7.375, then `boulder (1,24)`, then a forced hyperspace back to eye 5.875 and death at
energy 0.

Three player defects are measured and still open:

1. **The hop gates the wrong window.** `_pick_hop` drain-gates on `_gaze_window(tile)` —
   the tile being built on, 7983 f and predicted to within 2% — while the player's own
   body window collapses 490 → 60 f standing on its old tile for the ~750 f the macro
   takes. Each step passes its own gate (boulder needs 398 f, body has 655 f) while the
   3-step macro does not. Shadow instrumentation (`AStarPlayer._hop_audit = []`, records
   without enforcing) shows the body gate would reject only 44 of 205 candidates, all at
   one depth, all short by the same 124 f — it is affordable, not the blocker.
2. **`_c_pursue` is all-or-nothing.** It chains hops and returns ONE child or `None`, so a
   single unsurvivable hop discards the ten good steps before it and the root loses its
   only child generator. Enforcing the body gate therefore yields 0 actions; leaving it
   off yields a plan that dies executing it. Same defect either way. The fix is partial
   progress from `_c_pursue`.
3. **The human's line is pruned by the beam.** Replaying the human ls42 (internal 66) line
   through the model matches its energy curve exactly for 16 steps and reaches eye 11.875,
   the Sentinel's plinth, against the planner's ceiling of 7.375 — the model can represent
   it. But the first FOUR hops, (9,30), (13,26), (2,24), (5,22), are landable and pass
   every filter yet never enter `_pick_hop`'s top 8: they are ranked out by `_TOP_HOPS`.
   The key `(sees, robot_eye, window)` maximises eye gained per hop; the human raises the
   eye by exactly +0.5 each time, on a staircase. Hops 5-7 do appear, at ranks 2, 4, 1.

Aim cost is **angular, not spatial**: over the 23 landable tiles at the ls42 start,
`corr(aim, manhattan distance)` is **−0.54** (farther is cheaper — far terrain compresses
into a small angle) against +0.60 for pitch notches and +0.44 for bearing notches. An
adjacent tile needs a steep down-look: (12,29) at distance 1 costs 667 f, (8,28) at
distance 6 costs 159 f. The rank key sees none of this.

Our aim is **not** slower than the human's: priced by the same model on the same board,
2526 f against 3285 f over 18 non-transfer steps. (Transfers are excluded — the recorded
bearing on every one is `previous ^ $80`, the new body inheriting `creator_angle ^ $80` at
`$1BE0`, not an aim the human keyed.)

## Limits

- **The PRNG rate is unmodellable.** The ROM draws the stream ~19+ times a frame, often
  more than 32, while the cursor moves 3 — callers the model does not have. The LFSR
  itself matches `$31CA` instruction for instruction. This limits exactly two things, the
  model's only PRNG consumers, both through `put_object_in_random_tile_below_z $1224`: the
  discharge tree's landing tile and the hyperspace landing tile. **Meanie creation is not
  one of them** — `$197D` is a deterministic slot scan (first tree within 10 tiles of the
  targeted player that the enemy fully sees), as are the hunt and the hyperspace trigger.
- **The human line does not replay to a win** through the live executor: 21/42 steps, and
  the committed `ls42_truth.json` records 26/42 with `won_at_step: None` from before this
  work. It has never replayed to a win in this repo.

## Open, ranked

1. **`_c_pursue` partial progress** — unblocks the 0-action state and both gate variants.
2. **`_pick_hop` rank order and beam width** — the human's line is generated and then
   discarded. Two cheap offline experiments: rank by minimum sufficient rise rather than
   maximum, and widen `_TOP_HOPS`.
3. **Point the sim tests at internal 66** so they are a valid control at all.
4. **Terrain fill cost** — the residual under the pan model, systematic in scene
   busy-ness: mean error +1.8, −1.4, −4.5, −9.0 f across measured-cost quartiles. The
   lever is `projector.PER_SCANLINE`/`PER_PIXEL` and the cross-polygon span coupling.
5. **py65 exact backend skips transfer settles** — `_exact_render_cost` returns `None` for
   any non-player observer, and a transfer settle is always priced from one.
6. **Pan-commit wall-clock timeout** — `run_until_pc(PC_PAN_DONE, timeout=_RU_COMMIT)`
   (4 s). `$365D` recurs every frame, so a timeout there means the game left the play
   loop. Aborted 1 live run in 5 before the conversion; confirm it can no longer fire.

## Disproved (do not resurrect)

- "Transfer settle over-charges systematically." It was a 6.0 s wall-clock `run_until_pc`
  in `tap_action` clipping the measurement at ~300 frames.
- "Correcting the settle's viewpoint will reduce it." It moves **up** (median +28 f).
- "Aim mispricing is secondary." It was the larger term, and a driver defect (a swallowed
  sights toggle burning 171 frames), not a missing cost term.
- Ranking fixes by *cumulative* frame drift: net drift at the failing step was ~−17 f
  while the phase was ~35 f out.
- "`HOP_FRAMES` under-budgets every hop 2-3x." From a broken offline replay; live hops
  measure 745 and 879 f against 700. Replacing it with the computed budget took the live
  player to zero actions.
- "The u-turn's cost is a keystroke." It is an action tap: 74 f measured, not 3.
- "The fatal hop is expensive because it is 12 tiles away." Distance is *negatively*
  correlated with aim cost; that build measured `pan_h 18 f` against `pan_v 271 f`.
- "Meanie spawn location is PRNG-driven." `$197D` never touches the PRNG.
- "A* taking 0 actions on ls335 is the live-only freeze." The `--no-freeze` control does
  the same; ls335 planning is a separate open defect that voids it as an experiment arm.
- "Enemy freeze under `plotting=True` is a fidelity knob." It freezes enemies outright.
