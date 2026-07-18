# The A* planning player (`sentinel/astar_player.py`)

A weighted best-first (A*) search that plans one winning line over the
`sentinel/` model, then executes it. Shares `BasePlayer` (`sentinel/playerbase.py`)
with the [reactive player](player.md) — world clock, gaze windows, aim cost,
`_fire`/`_settle`, and the `--audit` post-settle invariant check are all common.

```bash
python -m sentinel.astar_player 66      # plan and play landscape 0042 (seed 66)
```

## What the search plans over

The game `State`. Enemies only rotate, so each tile has a closed-form gaze
window (frames until a cone rotates onto it); the search carries the cheap enemy
phase, gates every move on `window >= aim + settle`, and defers the keyboard-aim
cursor sweep to execution. Landing coordinates of PRNG-driven hyperspace/meanie
moves are never read.

- **Node** (`_Node`): `state`, cost-so-far `g`, the `(verb, tile)` path, the
  committed bearing and cursor. **Dedup key** (`_key`): player tile + eye,
  energy, remaining enemies (bucketed facing), and the built boulder/robot stacks.
- **Frontier**: `f = g + weight * h`, `weight = 1.4`. Budgets: `node_budget`
  200000 expansions, `time_budget` 30 s wall-clock.

## Candidate generators (`_expand`)

A node is the next *strategic sub-goal*, not a primitive step: the multi-hop
climb to reach a goal is solved by a directed inner routine and bundled into one
child, so search depth is ≈ the number of enemies, not the number of hops.

- **Absorb enemy** (`_c_absorb`) — terminal strike on an already-landable
  sentry/Sentinel (Sentinel dead last, the `$1B8E` lock).
- **Pursue enemy** (`_c_pursue`) — "absorb enemy E": from the current stance run
  a directed climb toward E, interleaving reclaim when short, until E is landable,
  then absorb it — all one child. One pursuit per not-yet-landable living enemy
  (nearest first).
- **Reclaim** (`_c_reclaim`) — absorb landable own boulders/shells (base ≤ eye)
  and, when short, trees; the player stays put so its own window bounds the aim.
- **Endgame** (`_c_endgame`) — Sentinel gone: robot on the platform tile →
  transfer → hyperspace (the win).

The inner climb **inchworms** (the measured human pattern,
[gameplay §7](gameplay.md#7-how-a-human-wins-quick-strategy)): each hop stacks at
most `_HOP_BOULDERS` (= 2) boulders, and after every transfer-up it reclaims the
pedestal now below the new eye, so energy rides the reserve floor instead of
being locked up in a tall tower.

## Cost model

`g` accumulates each action's **real** aim+settle via `_charge` =
`_aim_frames(view) + _settle(verb, view)`, advancing the enemies by that many
frames **before** the action — the same faithful charge the executor prices with,
so plan and execution agree (see [player.md](player.md); settle internals in
[render_cost.md](render_cost.md)). The heuristic `h` is a sum of floors derived
from the charged primitives: `remaining_enemies * absorb_floor + hops *
hop_floor + endgame_floor`, where each floor is the minimal aim latch plus the
per-verb settle floor.

## Execution and re-planning

`_tick` follows the plan step by step. Before each step, `_react` is a survival
override (a precomputed plan cannot predict enemy phase exactly): if a drainer
holds the current body it **counterattacks** (absorb a landable seer), else
**escapes** (transfer to the widest-window robot), else hyperspaces as the last
resort — and re-plans from the new state. Any live/plan divergence (`_fire` gate
fails, or no view lands the planned tile) triggers a fresh `_search`.

## Open problem — landscape 42 (seed 66)

Landscape 0 wins end-to-end; seed 66 does not, and the barrier is isolated to
**enemy-phase / threat fidelity, not search, node cost, or energy**. The energy
model is faithful — replayed against the recorded human win the inchworm
reproduces the human's energy penny-for-penny through the first 16 steps (audit:
ls42 41/42 exact, the lone gap a real mid-step drain). The failure is an
enemy-facing discrepancy: during the *search*, the sentry's cone rotates (by the
accumulated aim-cost frames) onto the human's own winning tiles (2,24)/(5,22) —
`window = 0` — so the pursuit gates refuse to build there and the drain bleeds
energy the human kept. Yet reconstructing those fixture states *alone* gives a
benign gaze (`window ≈ 1498`, `seen_now = False`), because the human fixtures
record object positions but **not enemy `h_angle` / rotation cooldowns** — so any
gaze verdict taken off a fixture state uses baseline (`landscape.generate`)
facings, not the true mid-game phase.

**Diagnosis (resolved by the human-log replay).** `driver/replay_human.py`
re-runs the recorded human line in VICE and captures the true per-enemy
`h_angle` / rotation cooldowns, committed as `human_wins/*_truth.json`. With true
facings the aim-cost clock is exonerated — our sim's enemy phase drifts at most
one rotation step (±20) from truth across the suspect steps. The defect is the
**gaze/exposure model over-classifying** the sentry's real sight at the human's
own winning tiles, two ways:

- **Partial sight scored as an immediate drain.** At (2,24) the Sentinel's cone
  is on the tile with only *partial* (head) sight; the ROM drains only on *full*
  sight (`$1838`) — partial merely arms a meanie when a tree is near. But
  `_gaze_window` promotes partial → dangerous whenever `_tree_near`, and
  `_account` flags any create exposure, so an undrainable glimpse scores
  `window = 0` / breach.
- **Placement judged as a body that never stands there.** At (5,22) the human
  places only a *boulder* while standing safely on (2,24), and the Sentinel
  rotates off before the later robot/transfer; but `_exposing_enemies` evaluates
  a phantom *robot* at placement and flags it, missing that the player never
  occupies the tile while exposed (and a boulder is not drainable).

**Fix:** in `BasePlayer`, stop treating partial sight as a `window = 0` drain
(partial + tree is the slower meanie-arming path, not an immediate loss), and
gate placement exposure on whether a *drainable body* actually stands exposed
(boulder creates and transit tiles are not bodies). Validate against the truth
fixtures (`test_human_audit` pins the current over-classifications). This is the
last gate before an ls42 win — for both this planner and the reactive player
(shared threat model).
