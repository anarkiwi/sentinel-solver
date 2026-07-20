# The A* planning player (`sentinel/astar_player.py`)

A weighted best-first search that plans one winning line over the `sentinel/` model, then
executes it. Shares `BasePlayer` (`sentinel/playerbase.py`) with the
[reactive player](player.md) — world clock, gaze windows, aim cost, `_fire`/`_settle`,
`--audit`. `driver/live_player.py`'s `LiveMixin` swaps the model for real VICE memory and
keystrokes under either player.

```bash
python -m sentinel.astar_player 66              # offline: internal seed 66 == typed 0042
python -m driver.play_player 42 --player astar  # live in VICE (digits parsed as hex)
```

Which boards win, and measured cost accuracy: [plan_fidelity.md](plan_fidelity.md) (which
also covers why typed `0042` is internal 66 and `Game.new(42)` is a different board).

## What the search plans over

The game `State`. Enemies only rotate, so each tile has a closed-form gaze window (frames
until a cone rotates onto it); the search carries the cheap enemy phase, gates every move
on `window >= aim + settle`, and defers the keyboard-aim cursor sweep to execution.
PRNG-driven hyperspace/meanie landings are never read.

- **Node** (`_Node`): state, cost-so-far `g`, the `(verb, tile)` path, committed bearing
  and cursor. **Dedup key** (`_key`): player tile + eye, energy, remaining enemies
  (facing >> 3), built boulder/robot stacks.
- **Frontier**: `f = g + weight * h`, `weight = 1.4`; `node_budget` 200000 expansions,
  `time_budget` 60 s per search (each replan gets its own window; think time is free live,
  `LiveMixin._advance` being a no-op).
- Landable tile sets, stance view dicts and below-eye band marches are memoized per
  (terrain map, observer) signature; the band march otherwise dominates search cost.

## Candidate generators (`_expand`)

A child is the next *strategic sub-goal*, not a primitive step: a multi-hop climb is solved
by a directed inner routine and bundled into one child, so depth is about the number of
enemies, not the number of hops.

- `_c_absorb` — terminal strike on an already-landable enemy (Sentinel last, `$1B8E` lock).
- `_c_pursue` — one child per not-yet-landable enemy (nearest first, `_TOP_TARGETS` = 4):
  chain minimal pedestal hops (`_pick_hop` ranks `_TOP_HOPS` = 8 candidates by LOS gain,
  eye height, window; `_hop_exec` builds k boulders + robot, then transfers), interleaving
  `_reclaim_one` when short, until the enemy is landable, then absorb. A climb that stalls
  short still returns the hops it made; `None` only when nothing was done.
  `_climb_continues` calls a landing unstranded when the target is landable, another hop is
  affordable, or a simulated reclaim chain (`_MAX_RECLAIM` = 8, against the landing's
  frozen tile set) makes one affordable.
- `_c_reclaim` — absorb landable own boulders/shells (base <= eye) and, when short, trees,
  up to `_MAX_RECLAIM` per macro; the player stays put so its own window bounds the aim.
- `_c_endgame` — Sentinel gone: robot on the platform tile, transfer, hyperspace.

The climb **inchworms** (the measured human pattern,
[gameplay §7](gameplay.md#7-how-a-human-wins-quick-strategy)): a hop stacks at most
`_HOP_BOULDERS` (2) boulders, and each transfer-up reclaims the pedestal now below the new
eye, so energy rides the reserve floor instead of being locked in a tower.

## Cost model

`_charge` = `_step_aim_frames(verb, view) + _settle(verb, view, _settle_eye(verb, tile))`,
advancing the enemies by that many frames **before** the action — the same charge the live
executor pays, so plan and execution agree. A transfer over a reused committed bearing
charges 0 aim (no aim keys sent); its settle is priced from the **post-transfer** eye
(`$0C63` moves before the `$35C3`/`$35C6` `plot_world` passes). A u-turn keyed mid-aim
unfreezes the world early (`$12E1`), so the advance splits at `_aim_unfreeze_split`. Settle
internals: [render_cost.md](render_cost.md).

`h` = `remaining_enemies * _ABSORB_EST + hops * _HOP_EST + _ENDGAME_EST`, each floor being
the minimal aim latch (`TAP_FRAMES`) plus the per-verb settle floor; `hops` is the eye
deficit to `_TARGET_EYE` (9.0) over `_EYE_PER_HOP` (0.9).

## Step cost is an interval (`_margin` / `_hot`)

A drain gate compares a *predicted* window against a *predicted* budget, so with zero
headroom any residual cost error flips safe -> hot. Gates fire on the pessimistic end,
`budget + _margin(d)`:

```
_margin(d) = k * sigma * sqrt(d + 1)   sigma = SENTINEL_STEP_SIGMA (default 24.1)
                                       k     = SENTINEL_MARGIN_K   (default 1.0)
```

Per-step charged-vs-measured error is ~zero-mean but does not cancel over a plan (an
excursion permanently shifts *when* a rotation committed), so it accumulates as a random
walk in depth `d` (`_begin` seeds it from `len(node.path)`, `_charge` increments it, a
rejected hop trial restores it). `sigma` is a measured rms, test-pinned below.

Margin-gated: `_c_absorb`, `_reclaim_one` (via `_hot`), `_pick_hop`/`_hop_exec` (via
`_drain_gate`'s budget), and the live `_plan_step_stale` at depth 0.
`_search(margin_k=0.0)` restores the raw gate.

## Execution and re-planning

`_tick` follows the plan step by step. `_react` overrides it when the player's window is
under `SAFE_FRAMES`: `_defend`, then `_plan_escape_transfer` (take the plan's own next
transfer when the pedestal just built IS the escape), and only then a hyperspace — once per
streak, only with an immediate drain and energy above the robot cost. Any live/plan
divergence (`_fire` gate fails, or no view lands the planned tile) re-`_search`es.

`_defend`: counterattack the cheapest-to-aim landable dangerous seer (`_dangerous_seers`,
Sentinel-last lock honoured), else `_escape_transfer` — cheapest aim, widest window as
tiebreak, only bodies beating the current window whose aim+settle fits inside it.

**Stale step.** `_plan_step_stale` (live override in `driver/live_player.py`) re-checks the
next step against the live enemy phase **on the window the plan gated it with**: the body
window for an absorb, the target tile's gaze window for a build or transfer, against
`budget + _margin(0)`. First verdict runs `_restale`'s ladder: `_search()`, else
`_defend()`, else `_search(margin_k=0.0)`, else `_wait()`.

**Progress guarantee.** `_search` does not advance the world (`_observe` leaves the CPU
halted, `LiveMixin._advance` is a no-op), so re-planning after the same verdict re-derives
the same head on an identical phase. `_stale` holds `(step key, consecutive verdicts)`: a
repeat skips the ladder and `_wait`s, which does advance the world, frame-exact
(`LiveMixin._wait` steps the wrap-free `$9630` counter `WAIT_FRAMES` times; charged vs
measured goes to `wait_audit`). After a wait (`count > 1`) a step whose **raw** budget
clears is released — the margin may not deadlock — while raw unsafety still blocks. So a
verdict cannot recur a third time without the world advancing or the step firing.

## Tests

`sentinel/tests/test_astar_player.py` pins: charge == executor `_step_aim_frames +
_settle`, advancing the enemies by it; 0-aim transfer only on a committed bearing;
below-eye builds charged at the real pitched view; transfer settle from the post-transfer
eye; no audited body left in a live cone; wins on landscape 0 and internal 66; a stalled
pursuit still returns its climb; `_margin` rejects a step inside the cost interval and
widens as `sqrt(depth+1)`; `_restale` and `_react` never concede a hyperspace early.

`sentinel/tests/test_hop_budget.py` pins `HOP_FRAMES`, `UTURN_FRAMES` and `_STEP_SIGMA`
against the live whole-step books.
