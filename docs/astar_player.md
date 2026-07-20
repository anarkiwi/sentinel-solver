# The A* planning player (`sentinel/astar_player.py`)

A weighted best-first search that plans one winning line over the `sentinel/` model, then
executes it. Shares `BasePlayer` (`sentinel/playerbase.py`) with the
[reactive player](player.md) — world clock, gaze windows, aim cost, `_fire`/`_settle`, and
the `--audit` post-settle invariant check. `driver/live_player.py`'s `LiveMixin` swaps the
model for real VICE memory and keystrokes under either player.

```bash
python -m sentinel.astar_player 66      # plan and play landscape 0042 (seed 66)
```

## What the search plans over

The game `State`. Enemies only rotate, so each tile has a closed-form gaze window (frames
until a cone rotates onto it); the search carries the cheap enemy phase, gates every move
on `window >= aim + settle`, and defers the keyboard-aim cursor sweep to execution.
Landing coordinates of PRNG-driven hyperspace/meanie moves are never read.

- **Node** (`_Node`): state, cost-so-far `g`, the `(verb, tile)` path, committed bearing
  and cursor. **Dedup key** (`_key`): player tile + eye, energy, remaining enemies
  (bucketed facing), built boulder/robot stacks.
- **Frontier**: `f = g + weight * h`, `weight = 1.4`; `node_budget` 200000 expansions,
  `time_budget` 60 s per search (each replan gets its own window; a cold ls42 search
  measures ~25 s, a warm one ~2.5 s, and think time is free live).

## Candidate generators (`_expand`)

A child is the next *strategic sub-goal*, not a primitive step: the multi-hop climb to
reach a goal is solved by a directed inner routine and bundled into one child, so depth is
about the number of enemies, not the number of hops.

- `_c_absorb` — terminal strike on an already-landable sentry/Sentinel (Sentinel dead
  last, the `$1B8E` lock).
- `_c_pursue` — one child per not-yet-landable living enemy (nearest first, `_TOP_TARGETS`):
  directed climb via `_pick_hop`/`_hop_exec`, interleaving `_reclaim_one` when short, until
  the enemy is landable, then absorb. A climb that stalls short still returns the node it
  reached (**partial progress**) — returning `None` there discards every step before the
  first unsurvivable one, and on ls42 that left the root with no children at all.
  `_climb_continues` decides a landing is not stranded when the pursuit's own next
  iteration could reclaim its way to an affordable hop, simulating that chain (up to
  `_MAX_RECLAIM`) against the landing's frozen tile set.
- `_c_reclaim` — absorb landable own boulders/shells (base <= eye) and, when short, trees;
  the player stays put so its own window bounds the aim.
- `_c_endgame` — Sentinel gone: robot on the platform tile, transfer, hyperspace.

The inner climb **inchworms** (the measured human pattern,
[gameplay §7](gameplay.md#7-how-a-human-wins-quick-strategy)): each hop stacks at most
`_HOP_BOULDERS` (2) boulders, and after every transfer-up it reclaims the pedestal now
below the new eye, so energy rides the reserve floor instead of being locked in a tower.

## Cost model

`_charge` = `_step_aim_frames(verb, view) + _settle(verb, view, _settle_eye(verb, tile))`,
advancing the enemies by that many frames **before** the action — the same charge the live
executor prices with, so plan and execution agree. A transfer over a reused committed
bearing charges 0 aim (the executor sends no aim keys); a transfer's settle is priced from
the **post-transfer** eye, since `$0C63` moves before `play_landscape_loop`'s two
`plot_world` passes. Settle internals: [render_cost.md](render_cost.md).

`h` sums floors derived from the same charged primitives:
`remaining_enemies * absorb_floor + hops * hop_floor + endgame_floor`, each floor being the
minimal aim latch plus the per-verb settle floor.

Accuracy of the charge against live measurement, and the open work on it, is
[plan_fidelity.md](plan_fidelity.md).

## Step cost is an interval (`_margin` / `_hot`)

Every drain gate compares a *predicted* window against a *predicted* budget, so with zero
headroom any residual cost error flips safe -> hot, and the error amplifies (the window is
a min over several enemies' phases). Gates therefore fire on the pessimistic end,
`budget + _margin(d)`:

```
_margin(d) = k * sigma * sqrt(d + 1)   sigma = SENTINEL_STEP_SIGMA (in code: 68.4)
                                       k     = SENTINEL_MARGIN_K   (1.0)
```

Shape: per-step charged-vs-measured error is ~zero-mean but does **not** cancel over a
plan — each excursion permanently shifts *when* a rotation committed — so phase uncertainty
accumulates as a random walk in plan depth. `d` is the step's depth (`_begin` seeds it from
`len(node.path)`, `_charge` increments it, a rejected hop trial restores it).

`sigma` is an **rms of measurement**, not a derived constant, and `k` is a chosen 1-sigma.
Both are env-overridable. The in-code default 68.4 f is **stale**: it came from a run
contaminated by driver defects since fixed; clean runs measure 49.3 f and 46.4 f, and the
margin tests pin concrete numbers, so changing it needs a test-pin review.

Gated on the margin: `_c_absorb`, `_reclaim_one` (via `_hot`), `_pick_hop` and `_hop_exec`
(via `_drain_gate`'s budget), and the live `_plan_step_stale` at depth 0 (the board is
freshly observed, so one step of uncertainty remains). `_search(margin_k=0.0)` restores the
raw gate.

## Execution and re-planning

`_tick` follows the plan step by step. Before each step `_react` is a survival override: if
the player's window is under `SAFE_FRAMES` it runs `_defend`, and only if that fails does
it hyperspace — last resort only, once per streak, and only when a drain is immediate and
energy exceeds the robot cost. Any live/plan divergence (`_fire` gate fails, or no view
lands the planned tile) triggers a fresh `_search`.

`_defend` is the non-conceding ladder: **counterattack** — absorb the cheapest-to-aim
landable dangerous seer (`_dangerous_seers`, honouring the Sentinel-last lock) — else
`_escape_transfer` to the landable robot with the widest window, and only if it is strictly
wider than staying put.

**Stale step (`_restale`).** `_plan_step_stale` (live override) fires when the player's own
window no longer covers the next step's aim+settle plus `_margin(0)`. The ladder on a first
verdict: `_search()`, else `_defend()`, else `_search(margin_k=0.0)`, else `_wait()`.

**Progress guarantee.** `_search` is a pure function of the board and does not advance it
(`_observe` leaves the CPU halted; `LiveMixin._advance` is a no-op), so re-planning after
the same verdict re-derives the same head and re-gates on an identical phase — the observed
livelock. So `_stale` holds `(step key, consecutive verdicts)` and:

- the consecutive counter strictly increases while the same `(verb, tile)` stays stale;
- a **repeat** skips the ladder and `_wait`s, which is frame-exact and really advances the
  world (`LiveMixin._wait` polls the wrap-free `$9630` counter for `WAIT_FRAMES` rather
  than sleeping on an assumed 50 fps; charged vs measured goes to `wait_audit`);
- after a wait (`count > 1`), `_plan_step_stale` releases a step whose **raw** budget
  clears — the margin absorbs prediction error and may not deadlock — while raw unsafety
  still blocks.

So the same verdict cannot recur a third time without either the world advancing or the
step firing.

## Status

Landscape 0 wins end-to-end. ls42 (seed 66) does not: the current failure is an
energy-starved search dead-end, not a threat-model or frame-cost failure. See
[plan_fidelity.md](plan_fidelity.md).
