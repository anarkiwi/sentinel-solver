# Replacing the greedy climb with best-first lookahead search

## 1. Why the current greedy planner fails

`climb_greedy.py`'s `climb_iterate` picks the single best-looking candidate
foothold each call (by height, then Sentinel-gaze distance, then edge
distance) and commits immediately. It has no way to know, before committing,
whether that foothold leads anywhere further — it discovers a dead end only
after arriving, at which point its only recourse is a "reposition" fallback
that will happily move to a *lower* tile just because it hasn't been visited.

This was confirmed directly, not assumed: a recorded human win on landscape 0
(`out/play_20260704_171942.jsonl`, cross-checked against the terrain) never
lost height across 5 real moves —
5.875 → 6.375 → 7.375 → 8.375 → 9.375 → 9.875 (win) — because every landing
tile's *bare terrain* was one full unit higher than the last (5, 6, 7, 8).
The player never moved unless a further way up was already visible. Greedy,
with no lookahead, structurally cannot guarantee that; only a search that
looks ahead before committing can.

## 2. Why not adversarial minimax

The original re-sentinel project already ran this experiment
(`../re-sentinel/disasm/SOLVER_ALGORITHMS.md`) and has hard numbers: their
minimax planner (`solver_exact.py`, depth-limited negamax with a worst-case
enemy band, ~930 LOC, 6 eval weights) is **beaten** on all three benchmark
landscapes by a ~30-40 line closure algorithm with no adversarial layer, at
roughly 1/100th the runtime.

Their reasoning, which holds here too: the Sentinel and sentries are
**fixed-rate automata** — `enemy_dynamics.step_enemies` advances rotation and
cooldowns as a pure function of tick count, and nothing in that function
branches on what the player does. There is no opponent making choices to
minimize against. A classic MAX/MIN alternation models an adversary that
isn't there, and pays search cost for it.

Their own recommended alternative ("closure," a one-vantage best-base sweep)
doesn't transfer directly to this codebase, though: it assumes **free,
arbitrary-height stacking from one base tile** and **terrain-only LOS that
objects never occlude**. Neither holds in the keyboard-real pipeline this
project actually drives: `native_los`'s LOS walks the live object stack (a
boulder occludes), and each individual boulder-step needs its own
keyboard-reachable centre-aim (`_boulder_centre_feasible`, which rejects a
large fraction of candidates in practice — see today's repeated "no on-boulder
centre-aim once built" log lines). You cannot climb to an arbitrary height in
one move here; every step is real, individually gated, and has to be found.

Their own doc lists the right shape for that situation: **best-first / A*
search over the real reachable states**, with enemy timing folded in as a
per-candidate safety annotation rather than an adversarial search dimension.
That's what this doc designs.

## 3. Search shape: receding-horizon best-first, not a single giant search

"20-30 moves for a hard landscape" is the length of the final *plan*, not a
search depth to explore exhaustively in one shot — branching factor is dozens
of candidates per node (`_candidates` returns 30-60+ on landscape 0), so a
naive full-width search to depth 20 is combinatorially impossible.

Instead: at each real decision point, run a bounded lookahead of depth `D`
(default 3, raised for landscapes that need it — see §7), score the leaves,
and commit to the first move of the best-scoring line. Re-run from the new
real state for the next decision. This is the same pattern a chess engine
uses per move; the total game (plan) can be arbitrarily long even though each
search is shallow. `D=3` for landscape 0 matches the finding that a human
needed only ~5 real moves to win it — 3 plies of lookahead is enough to see
past a would-be dead end that far out.

## 4. State representation and branching

Reuse `native_game.Game` as the search state — it already IS the real,
keyboard-faithful native model (`mem`, `col`, `energy`, `free`, `player`,
`plat`). It needs one addition: a cheap `clone()` (bytearray copy + dict/list
copies of `col`/`free`/`steps`; everything else is scalar) so the search can
branch without mutating the state other branches still need. `plan_greedy`'s
existing pattern of mutating one persistent `Game` in place is fine for a
single committed line, but a lookahead tree needs branches that don't step on
each other.

Successor generation reuses the REAL functions already built and validated
today, not a re-derived abstract model:
- `_candidates(g, cur, eye, plat, near_plat_radius)` for foothold options
  (hop / boulder-step, with `_foothold_eye`'s real height arithmetic),
- `_boulder_centre_feasible(g, T2, view)` to gate boulder-steps on actual
  keyboard-reachable centre-aim (this is the check that rejects most of the
  raw candidate list — cheap to evaluate, so filter with it before expanding
  a node, not after),
- `_refuel(g, log)` for the absorb/fuel side of a node (one-object-per-tile,
  per today's fix, to avoid the stale-aim batching bug).

Each successor also needs its own enemy-safety annotation (§5) computed
against the *tick* the search estimates the game will actually be at when
that move executes live.

## 5. Enemy modelling: annotation, not adversary

`scripts/enemy_dynamics.py` already has everything needed and is currently
unused by the climb planner — this is the concrete gap that caused today's
live failures (a meanie spawned near the player during a long climb; the
model had no way to see that coming because nothing in `climb_greedy.py`
consults this module at all):

- `EnemyPhase` / `step_enemies(state, phase)` — deterministic per-tick
  rotation/cooldown advance (rotation ±20 units/step ≈ ±28.125°, FOV width 20
  units ≈ 28.125° full / ±14.0625° half, update cooldown 4 (scan) / 30 (post-
  drain), rotation cooldown 200, draining cooldown 120 firing at 1 — every
  constant cited to its ROM address in the module header).
- `exposed_within_window(state, phase, x, y, ticks)` — **this already is**
  the "robust to timing jitter" check the old minimax was trying to search
  for, but as an O(ticks) forward simulation instead of a search branch: is
  this tile seen by any enemy at any point in the next N ticks. Use it to
  score/reject candidate footholds and dwell points.
- `ticks_until_seen(state, phase, x, y)` — safety margin in ticks; use as a
  tie-break or soft penalty in the leaf evaluation.
- `meanie_spawn_threat(state, player_tile)` / `meanie_safe(state,
  player_tile)` — ports the exact ROM conditions (A)-(D) for a tree near the
  player converting into a meanie that can then force a hyperspace. **This
  is precisely today's observed live failure** (an unexplained +3 global
  object-count jump right when a create failed, at (7,7) on a five-and-a-
  half-minute-old climb) — a threat this function would have flagged in
  advance had anything consulted it. Use `meanie_safe` as a hard filter on
  candidate footholds: don't dwell somewhere meanie-unsafe if a safe
  alternative with comparable value exists.

`enemy_dynamics.py` operates on `game_state.GameState`, not `native_game`'s
raw `mem` — reuse the conversion already built and validated today
(`GS.read_game_state(GS.Py65Source(g.mem))`) to get a `GameState` view of a
search node before calling into it. Build an `EnemyPhase` once per real
decision (`init_phase_from_ram(state, g.mem)`) and advance it by the
estimated tick-cost of each simulated move (§6) as the lookahead descends —
this is what turns "the enemy will do something" into a concrete forecast
instead of a guess.

## 6. Tick-cost-per-move: derived from the simulator's own aim geometry

Advancing the enemy state correctly during lookahead requires knowing how many
game rounds elapse per real keyboard-driven move. This is NOT a flat constant
per move type — a move is a *keyboard sequence* whose cost is dominated by how
far the view has to **pan** to aim each sub-action, and that pan distance is
computable exactly from the simulator (`sentinel.aimcost` +
`sentinel.los`'s keyboard lattice), not something to guess or crudely measure.

`climb_search._move_cost` mirrors `_apply`'s real sequence — aim + build the
foothold (a boulder then a re-centred on-boulder synthoid, or a lone synthoid),
transfer, then **swing the view back** to look down on the departed tile and
reabsorb its shell — and prices each aim by the keyboard lattice distance from
the previous heading (`aimcost.h_steps`/`v_steps`). That return swing is often a
near-180° bearing pan (verified in `out/ls0_pancost.log`: `h $90→$08` = 15
lattice steps) and is the bulk of a move's exposure window — the cost the old
flat `TICKS_PER_HOP`/`TICKS_PER_BOULDER_STEP` constants entirely ignored.

Keystrokes convert to enemy rounds by the ROM's own scroll cadence (the view
scrolls one step per frame, and `enemies.step` ≈ one frame): a coarse ±8
**bearing** keystroke animates a 16-step scroll (≈16 rounds), a ±4 **pitch**
keystroke an 8-step scroll (≈8 rounds), and a fired create/absorb/transfer
settles over its own plot cycle (≈16 rounds). These are the `ROUNDS_PER_H_STEP`
/ `ROUNDS_PER_V_STEP` / `ROUNDS_PER_ACTION` tunables (env-overridable so a
diverging live run can be recalibrated from its own telemetry, sec.9 step 4).
`_move_cost` also returns the **ending heading**, which the search threads into
the next move so each move's pan is charged from where the last one left the
view — the whole reason the enemy-rotation and drain forecasts a move descends
into are evaluated at the state the enemies *really* reach while it executes.

## 7. Search algorithm

Best-first, not plain DFS/BFS, so the search explores promising lines first
and can cut off early once a good-enough plan is found (important since
`D` may need to scale up for harder landscapes):

```
def lookahead(g, ctx, enemy_state, enemy_phase, depth):
    if won(g):
        return LEAF(score=+inf, path=[])
    if depth == 0:
        return LEAF(score=evaluate(g, enemy_state, enemy_phase), path=[])

    raw = _candidates(g, g.player_xy(), int(g.eye), ...)
    cands = [c for c in raw if not c.use_boulder or _boulder_centre_feasible(g, c)]
    cands = [c for c in cands if meanie_safe(enemy_state_after(c), c.tile)]
    cands.sort(key=lambda c: heuristic(c), reverse=True)   # best-first order
    cands = cands[:BEAM_WIDTH]                             # cap branching

    best = None
    for c in cands:
        g2 = g.clone()
        apply_candidate(g2, c)
        phase2 = advance_phase(enemy_phase, ticks_for(c))
        result = lookahead(g2, ctx, enemy_state_after(c), phase2, depth - 1)
        if result.score == -inf:      # dead end: no continuation at all
            continue                  # PRUNE -- this is the guarantee greedy lacked
        total = combine(c, result)
        if best is None or total.score > best.score:
            best = total
    if best is None:
        return LEAF(score=-inf, path=[])   # genuine dead end, propagate up
    return best
```

Key properties this gives that greedy didn't:
- **No move is taken unless it has a continuation within the horizon** — a
  node with zero non-dead-end successors returns `-inf` and is pruned by its
  parent, so the search backtracks to the next-best sibling instead of ever
  committing to the dead end. This directly fixes the diagnosed bug.
- **Best-first ordering + beam width** keeps branching tractable: sort
  candidates by a cheap heuristic (height gain, then enemy safety margin,
  then edge distance — the same priority order validated against human play)
  before expanding, and only expand the top `BEAM_WIDTH` (a tunable,
  independent of search depth `D`).
- `_boulder_centre_feasible` and `meanie_safe` are applied as **pre-filters
  before expansion**, not discovered after — cheap checks first, so the
  search never sinks lookahead budget into a branch the keyboard can't
  actually execute.

`evaluate()` at a cut leaf (depth exhausted, not yet won) scores on: eye
height reached (primary, matching "gain height as fast as possible"),
`ticks_until_seen` at the final tile (safety margin), edge distance (matching
the validated heuristic), and remaining energy (a state with 0 energy and no
refuel prospects is worse than one with a cushion).

## 8. What doesn't change

- `_refuel`, `_boulder_centre_feasible`, `_candidates`, `_foothold_eye`,
  `_apply` — all real, already-validated mechanics; the search calls them,
  it doesn't reimplement them.
- The live executor (`record_win_0042.py`'s `execute_live`) still resyncs
  from real memory every iteration and re-plans; the search runs fresh each
  time using the live-observed state, same as `climb_iterate` does today —
  only the *decision* function inside that loop changes.
- The endgame (absorb Sentinel, build platform synthoid, transfer) stays as
  its own terminal check (`won(g)` in the pseudocode above), not something
  the lookahead needs to discover generically.

## 9. Validation plan

1. Offline: `plan_greedy`-equivalent driven by the new search on landscape 0
   with `D=3`; confirm it reaches a native win, and specifically confirm it
   does not regress height at any step (the property the old greedy walk
   violated) and ideally with a much closer step count to the human's 5 real
   moves than the old planner's 41.
2. Cross-check landscape 66/9999 with the same `D=3`; raise `D` (up to the
   20-30 ceiling) only for landscapes that don't close at the lower depth,
   confirming search time stays practical (log node counts / wall time per
   decision).
3. ROM-oracle validate (`_finalize0.py`-style, py65) — must ROM-validate a
   real win, not just the native model.
4. Live VICE recording (`record_win_0042.py --live`) as the final check,
   since that's the only place enemy timing is real rather than estimated
   (§6) — if it diverges, that's a signal to recalibrate `TICKS_PER_ACTION`
   from the run's own `watch_play.py`-style log, not to add more static
   safety margin.
