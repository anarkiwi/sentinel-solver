# Outstanding solver work

The simulator (`sentinel/`) is bit-exact and the driver (`driver/`) executes and
verifies live keyboard operations. The **solver does not yet win** on the
tick-accurate simulator or live. This is a strategy-model problem, not a mechanics
problem, and the fix is a new plan model — not more tuning of the existing one.

## What was fixed

- **Climb direction (launch-readiness).** The height-dominant leaf score used to
  send the climb to the *tallest reachable* foothold, which on ls0 is a corner on
  the wrong side of the map — a dead end with no line to the plinth — where
  non-regression then stranded it. `climb_search` now scores a **clear diagonal to
  the plinth** (`ctx["launch_tiles"]`, a reverse LOS sweep from the platform
  vantage) above raw height, with height capped at launch height, so the climb
  heads to a tile that can actually fire the endgame instead of the tallest corner.
  Edge/corner cover is preserved (it is a set-membership signal, not a
  centre-pulling distance term). This stopped the corner-flight; it did **not** by
  itself produce a win.

- **Live runner is now glue.** All emulator driving moved into the driver
  (`driver.core.boot_and_play` for the container/boot/connect/navigate/record
  lifecycle and `validate_avi`; `driver.sentinel_execute.perform_step` /
  `fire_hyperspace` for the keyboard-aim → fire → memory-verify primitive). The
  runner keeps only the plan-execution loop and the CLI. This is the surface the
  new plan model will drive.

## Why it still loses: the model does not match how the game is won

A recorded **human win** on ls0 (`git show <old-commit>:out/play_20260704_171942.jsonl`,
polled live) shows a strategy the current greedy, cost-weighted `climb_search`
does not embody:

1. **Never seen — zero drain.** Every energy change in the human run is build/absorb
   accounting; the Sentinel never had line of sight to the player, even while it stood
   **cheb-3 from the plinth**. Safety is the deterministic **gaze forecast**
   (`threat.ticks_until_seen`), not the static "could ever be seen" mask
   (`threat.is_exposed`) — that mask flags nearly every useful high tile (all of the
   human's) and is what previous agents mis-read as "exposure", excluding the winning
   routes. Real exposure = *the gaze catches you during your action window*.

2. **Aim-coherent ping-pong.** Consecutive aim bearings flip by ≈180° each move
   (measured Δ ≈ 128 units), so every aim is a single U-turn keystroke, never an
   arbitrary sweep. The XY looks scattered; the *bearing sequence* is maximally tight.
   Cheap actions and gaze-avoidance are the **same lever** — a cheap (short-aim) action
   finishes inside a gaze gap; an expensive one (long pan, tall build) overruns the gap
   and gets caught.

3. **Height first, cheaply.** Height comes from **transferring to naturally-high
   terrain** (landing-tile terrain 5→6→7→8, +1 each) with *minimal* builds, not from
   tall boulder towers on low ground. Build/absorb at a **distance** (shallow aim,
   cheap), never adjacent (steep down-look, expensive); ≤2 boulders, far transfer,
   shallow look-back reabsorb.

4. **Fuel is deferred by visibility.** Don't grab low-value fuel early. Fuel that is
   visible from many places is deferred (get it later from height); fuel that is hard
   to see from elsewhere, or needed to escape a terrain pocket on a steep landscape, is
   taken opportunistically now.

5. **Launch from afar.** The win is fired from a far high tile looking *down* onto the
   platform. NB the endgame/launch LOS query is currently treated as symmetric and
   under-counts exactly these far high launch tiles (the player looking *down* is
   allowed where the platform-vantage looking *up* is blocked, `$1D2E`).

The current `climb_search` is a receding-horizon best-first search whose objective is
height with a stack of soft cost penalties (`RESERVE`, `EXPOSURE_RESERVE`, seen-drain,
pan cost, launch-readiness, …). Tuning those against each other does not reproduce the
strategy above — it takes *expensive* actions (batched-boulder towers, ~180° return
pans, adjacent builds), overruns gaze gaps, is drained, and starves before launch. On
the tick-accurate sim it only wins when the modelled per-action drain window is cut
~2.5× (consistent with the open scan/cooldown cadence calibration gap), which is the
wrong fix.

## The plan model to build

A **deterministic, constructive planner** (everything — gaze precession, terrain, aim
geometry — is deterministic), not a cost-weighted search:

- **Gaze oracle as a hard constraint.** Precompute the Sentinel's gaze over time; a
  build/transfer/absorb is only ever scheduled onto a (tile, time-window) the gaze is
  provably off. No soft "exposure reserve" — a move into the gaze is illegal.
- **Distance-priced aim.** Replace the pan-only cost with a real aim cost dominated by
  pitch steepness, so the planner prefers far/shallow builds and absorbs; the move
  primitive is *build ≤2 boulders at a far tile → transfer → shallow look-back reabsorb*.
- **Height-first phases**, self-funded by the trail, fuel deferred by the visibility
  rule above; launch from a far out-of-gaze tile with LOS to the plinth.
- **Offline plan, live re-schedule.** Solve the full plan offline; re-solve on live
  divergence (meanie, etc.). A **missed aim is a crash** — the offline plan is
  aim-exact, so a miss means the model is wrong and must be investigated, not smoothed
  over.

See `SEARCH_REDESIGN.md` for the (now superseded) receding-horizon design and the
reasoning that led here.
