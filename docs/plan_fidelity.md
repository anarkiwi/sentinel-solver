# Plan-vs-live fidelity

State: **the clocks are exact, the A\* player wins ls42 live on the real game, and the
run reproduces frame for frame.** `python -m driver.play_player 42 --player astar` wins in
**36 actions, final energy 10** (10962 recorded frames,
[media/ls42_astar_win.png](media/ls42_astar_win.png)), and offline
(`python -m sentinel.astar_player 66`) in 41 actions / 11263 f, eye 11.875 on the plinth —
the human line's height. Landscape 0 wins offline (26 actions). The reactive greedy player
loses both ls42 boards.

**The plan is a pure function of the board.** Two live ls42 runs produce identical action
sequences and identical per-step measured frame counts. `_search` is bounded by
`node_budget` alone; `time_budget` (a wall-clock cut) is **off by default**, because a
loaded host truncates the search sooner and plays a different — not cheaper — line. That
wall-clock cut was worth 3 actions and 2684 frames on this board: it was ending the search
early enough to miss the better plan.

## READ FIRST: "landscape 42" is two boards

`driver.core.landscape_from_digits` parses the typed code as **hex**, so typing `0042`
seeds internal landscape **0x42 = 66**.

- `driver/play_player.py 42` and the human logs (`ls42.json`: `entered_code 42, landscape
  66`) play **internal 66** — player starts at (13,29).
- `Game.new(42)` and `test_astar_player._LANDSCAPE = 42` build **internal 42** — player at
  (14,27), 17 objects against 66's 16, zero slot overlap.

`Game.new(66)` matches the human ls42 fixture exactly (16/16 objects, same slots) and the
live replay agrees. Sim tests and the live driver therefore do **not** exercise the same
board; any sim-vs-live comparison keyed on "42" is void.

## Clocks: exact

Frame-locked against the running game from a byte-identical seed
(`python -m driver.instrument 42`): **no divergence within 1200 frames** — every enemy
facing, rotation/update/draining cooldown and the Bresenham clock. Gated by
`driver/test_enemy_sim_divergence.py` (600 frames, strict). Mechanisms:

- **The cooldown ticks before the enemy sweep.** The raster IRQ (`$9663`/`$1317`) ticks
  ahead of the foreground passes, so an enemy the tick makes due rotates in the same
  frame.
- **A u-turn unfreezes the world mid-aim.** `$12D5 CMP #$22 / BCS $12DE` lets action codes
  >= $22 skip the sights-on check and fall into `$12E1 LSR $0CE5`; a u-turn is code `$23`,
  so keying one starts the enemy clock part-way through the aim. ls42's first aim keys
  one.
- **`UPDATES_PER_FRAME` is 8.** The foreground loop makes 2-4 passes a frame, but what
  rate-limits an enemy is its own `$16E9` update_cd gate (reload 4), which is tighter;
  sweeping every slot reproduces the ROM clock exactly, a literal 3 does not.

## Frame cost

Live ls42 whole-step charged-vs-measured: **rms 24.1 f, mean −12.0, max |e| 46, n=11**
(`live_ls42_hops.json`, `test_hop_budget.py`), against 49.3 f before this work.
`HOP_FRAMES` (700), `UTURN_FRAMES` (74 — a u-turn is an action tap, not a keystroke) and
`_STEP_SIGMA` (24.1, the measured per-step rms) are measured, not fitted. Per-notch pan
redraw is derived (`sentinel/pancost.py`, [render_cost.md](render_cost.md)): tile
selection byte-exact on all 288 golden notches, rms 18.3 → 6.4 f.

A cold ls66 search runs ~32 s, a warm one ~2.5 s. Think time is free live
(`bm.auto_resume = False`: the world runs only in deliberate run windows), which is why
bounding it by wall clock buys nothing and costs reproducibility.

## Open, ranked

1. **Per-step frame drift.** −208 f over the 36 steps of the winning live run (mean −5.8,
   rms 58.1), reproducible run to run. Decomposed over 15
   runs, the settle side is two constants the model merges: **create measures 99 f (n=71,
   sd 7.8), absorb ~90 f (n=65)** against a shared charge of 93.75 (`DITHER_FRAMES +
   POST_ACTION_REPLOT_FRAMES`). The ROM counter behind it is `$2099` (`$1FA4` loads #$19;
   `$2051` loads #$28 when `$0C4E`, the meanie-made flag, is set) — that is the meanie
   split, not the create/absorb one, so the difference is unattributed. The aim side is a
   separate +8.7 mean (rms 15), dominated by large pans.
2. **`_pick_hop` rank order and beam width.** Replaying the human ls42 line matches its
   energy curve exactly for 16 steps, yet its first four hops — (9,30), (13,26), (2,24),
   (5,22) — are landable, pass every filter, and never enter `_pick_hop`'s top 8: the key
   `(sees, robot_eye, window)` maximises eye gained per hop while the human raises the eye
   by exactly +0.5 each time. A cost question, not a feasibility one. Aim cost is
   **angular, not spatial** — over the 23 landable tiles at the ls42 start,
   `corr(aim, manhattan)` is **−0.54** against +0.60 for pitch notches; the rank key sees
   none of it. Two cheap experiments: rank by minimum sufficient rise rather than maximum,
   and widen `_TOP_HOPS`.
3. **Point the sim tests at internal 66** so they are a valid control at all.
4. **Terrain fill cost** — the residual under the pan model, systematic in scene
   busy-ness: mean error +1.8, −1.4, −4.5, −9.0 f across measured-cost quartiles. Lever:
   `projector.PER_SCANLINE`/`PER_PIXEL` and the cross-polygon span coupling.
5. **py65 exact backend skips transfer settles** — `_exact_render_cost` returns `None` for
   any non-player observer, and a transfer settle is always priced from one.
6. **The DRIVER's wall-clock timeouts are the residual load sensitivity** — `_RU_PAN`
   (20 s), `_RU_STA` (8 s), `_RU_COMMIT` (4 s) in `kbd_aim`. The planner is now
   reproducible, so these are what is left: on an idle host
   `driver/test_live_determinism.py` passes (2/2 serial, and two full ls42 runs are
   frame-identical), while under a saturated host (`pytest -n auto`, ~14 workers plus two
   VICE containers) it still fails on differing step counts. A monitor round-trip costs
   ~23.5 ms at real-time pace, so enough contention pushes a checkpoint wait past its
   timeout and the aim re-drives. Since `$365D` recurs every frame, a timeout there means
   the game left the play loop — it should be an error, not a retry.

## Limits

- **The PRNG rate is unmodellable.** The ROM draws the stream ~19+ times a frame, often
  more than 32, while the cursor moves 3 — callers the model does not have. The LFSR
  itself matches `$31CA` instruction for instruction. This limits exactly two things, both
  through `put_object_in_random_tile_below_z $1224`: the discharge tree's landing tile and
  the hyperspace landing tile. **Meanie creation is not one of them** — `$197D` is a
  deterministic slot scan, as are the hunt and the hyperspace trigger.
- **The human line does not replay to a win** through the live executor: 21/42 steps
  (committed `ls42_truth.json` records 26/42, `won_at_step: None`). It has never replayed
  to a win in this repo.

## Disproved (do not resurrect)

- "A bound on recoverable energy can replace the strand probe's simulated reclaims." It
  accepts landings whose abandoned stack is not keyboard-aimable from them; the pursuit
  commits to one and dead-ends. Back to 0 actions.
- "The climb ranker only needs the `$F5` up/level pitch plane." A pedestal is aimed at by
  its TILE, routinely *below* the eye even when the robot on it is not. Empties
  `_pick_hop` at the first hop.
- "Transfer settle over-charges systematically." It was a 6.0 s wall-clock `run_until_pc`
  in `tap_action` clipping the measurement at ~300 frames.
- "Correcting the settle's viewpoint will reduce it." It moves **up** (median +28 f).
- "Aim mispricing is secondary." It was the larger term, and a driver defect (a swallowed
  sights toggle burning 171 frames), not a missing cost term.
- Ranking fixes by *cumulative* frame drift: net drift at the failing step was ~−17 f
  while the phase was ~35 f out.
- "`HOP_FRAMES` under-budgets every hop 2-3x." Live hops measure 745 and 879 f against
  700; replacing it with the computed budget took the live player to zero actions.
- "The fatal hop is expensive because it is 12 tiles away." Distance is *negatively*
  correlated with aim cost; that build measured `pan_h 18 f` against `pan_v 271 f`.
- "Meanie spawn location is PRNG-driven." `$197D` never touches the PRNG.
- "A\* taking 0 actions on ls335 is the live-only freeze." The `--no-freeze` control does
  the same; ls335 planning is a separate open defect.
- "Enemy freeze under `plotting=True` is a fidelity knob." It freezes enemies outright.
