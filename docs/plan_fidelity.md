# Plan-vs-live fidelity

State: **the clocks are correct; the A\* player wins ls42 offline.** The enemy simulation
tracks the real game exactly, and under it `sentinel.astar_player` now plans and executes
a win on internal 66 (the board typed `0042`) — 41 actions, 11263 f, eye 11.875 on the
plinth, the human line's height. The reactive greedy player still loses both boards.

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

## Players: the ls42 climb, fixed

Offline (`python -m sentinel.astar_player 66`): **WON in 41 actions / 11263 f, energy 11**.
Landscapes 0 and internal 42 still win (26 and 29 actions). The reactive player is
untouched and still loses (17 actions on 66, 12 on 42).

The 0-action state was **two** defects in the pursuit macro, both in the same place:

1. **`_c_pursue` was all-or-nothing.** It chained hops and returned ONE child or `None`,
   so a chain that stalled short of its enemy discarded the ten good steps before it and
   left the root with no child generator. It now returns the node it reached — the search
   resumes from a stalled climb instead of losing it.
2. **`_climb_continues` did not model the loop's own reclaim.** It called a landing
   stranded whenever no further hop was affordable *at that instant*, but the pursuit's
   next iteration reclaims first when energy is short. On this board a k=1 hop leaves
   E=6 against the 8 the next needs, so **every** landing above eye 7.375 was rejected —
   the exact ceiling the planner sat at. It now simulates the reclaim chain (capped at
   `_MAX_RECLAIM`, short-circuited once energy stops being the binding filter).

Simulating the reclaims rather than bounding the recoverable energy is load-bearing: an
energy bound accepts landings whose abandoned stack is not keyboard-aimable from them, and
the pursuit then commits to one and dies there (measured: back to 0 actions).

Search cost fell out of the same work — a cold ls66 search is ~25 s, warm ~2.5 s, against
~30 s / ~15 s before:

- `_view_for`'s targeted band march is memoized per `(sig, tile)`, not just per sig. It was
  the single largest cost (130 marches, 9.7 s of a 14.7 s warm search); every trial hop,
  strand probe and re-search at a stance re-marched the same below-eye tiles.
- The strand probe re-ranks against the landing's own tile set instead of re-sweeping the
  board per absorbed object (`_landable_batch` is the other half of the profile).

`time_budget` defaults moved 30 → 60 s. Think time is free live (`bm.auto_resume = False`:
the world runs only in deliberate run windows), and at 30 s the cold search was a coin
flip on a loaded machine — the losing side executes a truncated line and dies at eye 7.375.

Still open on the ranker: **the human's line is pruned by the beam.** Replaying the human
ls42 (internal 66) line matches its energy curve exactly for 16 steps. Its first FOUR hops,
(9,30), (13,26), (2,24), (5,22), are landable and pass every filter yet never enter
`_pick_hop`'s top 8: the key `(sees, robot_eye, window)` maximises eye gained per hop while
the human raises the eye by exactly +0.5 each time, on a staircase. We reach the same
plinth by a different route, so this is now a cost question, not a feasibility one.

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

1. **The live ls42 run: 14 actions, then an unrecoverable hyperspace.** The live A* now
   plans the same 42-step win from live memory (`plan (7 nodes)`) and executes 13 steps of
   it, through two transfers to eye 7.375 (`renders/player_ls42_astar_win.avi`, a LOSS).
   It dies in two stages:
   - **Drift makes a planned window stale.** Charged exceeds measured on nearly every
     step (355/347, 316/270, 205/164, 331/289, ...), ~+250 f over 13 steps, so the live
     enemy phase lags the plan. At p14 the `robot (2,24)` step reads a live gaze window of
     30 f against a 179 f budget, re-plans, and concedes an escape hyperspace. The
     `_margin` covers one step's sigma (24 f), not accumulated drift.
   - **Post-hyperspace the root has no children at all.** Landed at eye 5.875 with E=6:
     under `HOP_COST + reserve` so `_c_pursue` must reclaim first, nothing is reclaimable
     from there, no enemy is landable — `plan (1 nodes): None` until it is drained. The
     search has no hyperspace-as-relocation generator and cannot get one (the landing tile
     is `$1224` PRNG, see Limits), so the only fix is not entering that state.
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

- "A bound on recoverable energy can replace the strand probe's simulated reclaims." It
  accepts landings whose abandoned stack is not keyboard-aimable from them; the pursuit
  commits to the first such landing and dead-ends on it. Back to 0 actions.
- "The climb ranker only needs the $F5 up/level pitch plane, so its sweep can drop the
  other 26." A pedestal is aimed at by its TILE, which is routinely *below* the eye even
  when the robot on top of it will not be. Restricting the plane empties `_pick_hop` at
  the first hop.

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
